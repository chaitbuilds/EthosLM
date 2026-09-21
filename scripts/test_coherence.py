"""Offline coherence cases; recorded evidence is read, never overwritten."""
import copy
import functools
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethoslm import (groundread, observe, offline, pipeline, placeplan,
                   regions as _regions, spec, styles, voices)

loader = importlib.util.spec_from_file_location("site_search_coherence", ROOT / "scripts/find_site.py")
fs = importlib.util.module_from_spec(loader)
loader.loader.exec_module(fs)


class Coherence(unittest.TestCase):
    def test_virgin_ground_known_sites_and_clean_fixture(self):
        self.assertIn("origin_builds", groundread.excluded(-256, 0, 372, 372))
        self.assertIn("site_b", groundread.excluded(1536, -768, 192, 192))
        v = observe.Volume(5000, 60, 5000, np.ones((48, 12, 48), np.uint16), ["air", "stone", "bricks"])
        clean = fs.measure(fs.field_from_volume(v), 5000, 5000, 48, 12)
        self.assertEqual(clean["man_made_share"], 0)
        self.assertEqual(groundread.virgin_failures(clean), [])
        v.codes[:, 10:, :] = 2
        dirty = fs.measure(fs.field_from_volume(v), 5000, 5000, 48, 12)
        self.assertGreater(dirty["man_made_share"], 0.005)
        self.assertTrue(groundread.virgin_failures(dirty))
        self.assertTrue(groundread.virgin_failures({"man_made_share": None}))

    def test_the_man_made_census_does_not_count_a_forest_as_a_build(self):
        """`observe._is_natural` is blind to vegetation on purpose -- a placed log is a
                corner post when the question is a room's wall -- and the man-made census
                borrowed it as it stood, so a square kilometre of forest read as a build. On the
                cached worlds 92 to 98 per cent of what the census counted was leaves and logs;
                90 of the 111 squares a city's search scanned were refused for it; and the one
                green square that met every other need read 1.65% "man-made" with no build in
                it. Here: a wood on grass counts for nothing, and a cobblestone wall still does.
                
        """
        n = 48
        codes = np.zeros((n, 12, n), np.uint16)
        palette = ["air", "stone", "grass_block", "oak_log", "oak_leaves", "lava",
                   "tube_coral_block", "cobblestone"]
        codes[:, :4, :] = 1
        codes[:, 4, :] = 2
        # a wood: trunks on a lattice, a canopy over the lot, a lava pool, a reef
        codes[::4, 5:9, ::4] = 3
        codes[:, 9:11, :] = 4
        codes[0:6, 4, 0:6] = 5
        codes[40:48, 4, 40:48] = 6
        v = observe.Volume(5000, 60, 5000, codes, palette)
        wood = fs.measure(fs.field_from_volume(v), 5000, 5000, n, 12)
        self.assertEqual(wood["man_made_share"], 0, wood["man_made_blocks"])
        self.assertEqual(groundread.virgin_failures(wood), [])
        # ...and a wall of cobblestone through it is a build
        codes[:, 5:8, 20:22] = 7
        v = observe.Volume(5000, 60, 5000, codes, palette)
        walled = fs.measure(fs.field_from_volume(v), 5000, 5000, n, 12)
        self.assertGreater(walled["man_made_share"], groundread.MAX_MAN_MADE_SHARE)
        self.assertTrue(groundread.virgin_failures(walled))
        for name in ("oak_leaves", "oak_log", "lava", "cactus", "kelp", "bone_block",
                     "tube_coral_block", "mangrove_propagule", "snow"):
            self.assertFalse(groundread.made(name), name)
        for name in ("cobblestone", "stone_bricks", "oak_planks", "oxidized_copper",
                     "deepslate_bricks", "lantern"):
            self.assertTrue(groundread.made(name), name)
        # ...and the surface census reads the ground under the wood as green, with the
        # lava pool and the reef unclassed rather than charged to any setting
        s = wood["surface"]
        self.assertTrue(s["read"])
        self.assertGreater(s["classes"]["green"], 90.0, s)
        self.assertEqual(s["classes"]["badlands"], 0.0, s)
        self.assertEqual(s["top"][0][0], "grass_block", s["top"])

    def test_the_region_files_read_as_the_server_serves_them(self):
        """A square off the save's own files is the square a session read, byte for byte,
        and ground the files do not hold is refused, never made.
        """
        from ethoslm import regions
        cached = ROOT / "out" / "sites" / "ground_0_-1024_512x512.npz"
        if not (Path(regions.REGIONS).is_dir() and cached.exists()):
            self.skipTest("no region files or no cached square on this checkout")
        regs = regions.Regions()
        try:
            f = fs.field_from_regions(regs, 0, -1024, 64, 64, log=lambda *a: None)
            self.assertIsNotNone(f)
            old = np.load(cached)
            np.testing.assert_array_equal(f.h, old["h"][:64, :64])
            np.testing.assert_array_equal(f.wet, old["wet"][:64, :64])
            np.testing.assert_array_equal(f.canopy, old["canopy"][:64, :64])
            np.testing.assert_array_equal(f.occupied, old["occupied"][:64, :64])
            # ...and it carries what the session's read could not: the surface
            self.assertIsNotNone(f.surface)
            self.assertTrue(all(isinstance(n, str) for n in f.surface_palette))
            self.assertIsNotNone(f.gravity)
            # ...and ground past the generated world is NotGenerated, not made
            span = fs.region_span(fs.generated_regions())
            with self.assertRaises(regions.NotGenerated):
                regs.world_slice(span["x"][1] + 1, 0, 16, 16)
            self.assertIsNone(fs.field_from_regions(regs, span["x"][1] + 1, 0, 16, 16,
                                                    log=lambda *a: None))
        finally:
            regs.close()

    def test_the_recorded_town_is_no_longer_foreign_to_its_own_voice(self):
        """Every one of that town's 63 leaves declared a voice other than the one it was
                built in, and twelve of them declared a Japanese one -- which read as a coherence
                failure and was really a contract failure: the palette was never the type's to
                declare. A type declares a **form** now, so the same plan, unchanged, is checked
                against the place's form family instead, and the whole class is gone.
                
        """
        # Asked of the round rather than of a path under `out/`, so a checkout that has
        # the recorded plan as a shipped fixture and has never run the round reads the
        # same bytes. See `Round.state`.
        recorded = pipeline.Round.load(str(ROOT / "rounds/town.json"))
        plan = json.loads(Path(recorded.rel("plan.json")).read_text())
        parts = pipeline.plan_parts(plan)
        decls = pipeline.type_declarations(parts)
        # No type declares a palette any more, so there is nothing for a voice to
        # disagree with: the same plan under the voice it was built in has no failure of
        # that kind at all.
        self.assertFalse([f for f in pipeline.plan_failures(parts, decls)
                          if f["check"] == "form"])
        forms = {decls[p["type"]]["form"] for p in parts}
        self.assertEqual(forms, {"european_vernacular", "east_asian", "fortification",
                                 "civic"})
        # A place *does* still have a family, and a leaf outside it is named. The town
        # is European vernacular plus the two functional forms; its twelve east-Asian
        # leaves are the ones that do not belong.
        named = [f["part"] for f in
                 pipeline.plan_failures(parts, decls, form="european_vernacular")
                 if f["check"] == "form"]
        self.assertEqual(len(named), 12)
        self.assertTrue(all(decls[p["type"]]["form"] == "east_asian"
                            for p in parts if p["name"] in named))
        # ...and the brief a planner is given carries only the admissible types.
        table, allowed = placeplan.types_card(form="european_vernacular")
        self.assertTrue(allowed)
        self.assertNotIn("`minka`", table)
        self.assertIn("`cottage`", table)
        self.assertIn("`wall`", table)          # fortification, admissible anywhere
        self.assertIn("`square`", table)        # civic, likewise
        # Re-typed within the family, without moving a footprint, the plan passes.
        corrected = copy.deepcopy(parts)
        for p in corrected:
            if pipeline.form_ok(decls[p["type"]]["form"], "european_vernacular"):
                continue
            candidates = [n for n, d in allowed.items() if d["kind"] == p.get("kind", "plot")
                          and not pipeline.needs_footprint_failure(p, d["needs"])]
            self.assertTrue(candidates, p["name"])
            p["type"] = sorted(candidates)[0]
            p["params"] = pipeline.check_params(allowed[p["type"]]["params"], {})
        left = pipeline.plan_failures(corrected, allowed, form="european_vernacular")
        # **No form failure is left**, which is what this case is about. Not "no failure
        # of any kind": that assertion coupled a claim about *palettes* to every number
        # in every type's `NEEDS`, and it broke the day `wall` was re-authored with a
        # clearance of 4 against the old one's 2 -- five of the recorded town's plots
        # stand closer than that to its own wall. That is a true fact about a bigger
        # wall and it is not this test's.
        self.assertEqual([f for f in left if f["check"] == "form"], [])
        self.assertEqual([f for f in left if f["check"] not in ("overlap",)], [])
        near = sorted({f["other"] for f in left if f["check"] == "overlap"})
        print(f"re-typed plan: 0 form failures; {len(near)} plot(s) within the "
              f"re-authored wall's clearance of "
              f"{pipeline.load_type(str(ROOT / 'types/wall.py'))['needs']['clearance']}"
              f": {near}")

    def test_a_type_declares_a_form_and_never_a_palette(self):
        self.assertEqual(pipeline.load_type(str(ROOT / "types/cottage.py"))["form"],
                         "european_vernacular")
        self.assertNotIn("style", pipeline.load_type(str(ROOT / "types/cottage.py")))
        # The functional forms stand in a place of any family; the regional ones do not.
        self.assertTrue(pipeline.form_ok("fortification", "east_asian"))
        self.assertTrue(pipeline.form_ok("civic", "european_vernacular"))
        self.assertFalse(pipeline.form_ok("east_asian", "european_vernacular"))
        self.assertTrue(pipeline.form_ok("east_asian", None))
        # And a voice is a file, validated, naming every role a type may ask for and no
        # role the shell does not know. Asked as the contract and not as a list.
        known = set(voices.ROLES) | set(voices.OPTIONAL)
        for name in voices.names():
            v = voices.load(name)
            self.assertTrue(set(voices.ROLES) <= set(v["roles"]), name)
            self.assertEqual(set(v["roles"]) - known, set(), name)

    def test_the_fresh_ground_search_reads_from_a_cache_and_needs_no_second_server(self):
        """A5: the grid is walked square by square, and a square is read once, ever.

                Three things, and the third is the one that matters. A square the region files
                cover is ground the world has; a square they do not is ground the server would
                have to **make**, which is a change to the save and is capped at a number the
                caller says out loud. And a square that has been read once is in `out/sites/`,
                so the same search over the same grid runs again with no server at all and gives
                the same ranked list -- which is what makes a scan of a city's ground affordable
                rather than a session apiece.
                
        """
        import numpy as np
        import os
        import tempfile
        # What the save covers is a fact about this host and not about the search. A
        # checkout with no save still runs everything below, against the cache this test
        # writes itself.
        regions = fs.generated_regions()
        if regions:
            span = fs.region_span(regions)
            self.assertTrue(fs.covers(regions, -256, 0, 372, 372))
            self.assertFalse(fs.covers(regions, span["x"][1] + 1, 0, 372, 372))

        s = spec.read_spec(dict(fs.CHECK_SPEC), fs.CHECK_SENTENCE)
        size = int(s["needs"]["footprint"])
        with tempfile.TemporaryDirectory() as d:
            # **The save is not this suite's subject and is not on a stranger's
            # machine.** Every search below is handed a region directory with no files
            # in it, so what it reads is the cache this test writes and nothing else.
            # Given the host's own save it reads three squares off it instead, scores
            # them, and the synthetic caches under test are never looked at.
            empty = _regions.Regions(os.path.join(d, "no-regions"))
            # A radius wide enough that some square of it is outside the reservation
            # ledger at this spec's footprint: the craft round (E1) re-expressed the
            # size bands at the fabric the library lays, so the hamlet this case checks
            # asks for more ground and every square within 512 of the origin is now
            # reserved. What this case is about is the cache, not the ledger.
            fresh = functools.partial(fs.search_fresh, editor=None, radii=(1024,),
                                      stride=512, directory=d, regions=empty,
                                      log=lambda *a: None)
            # nothing cached and no session: every square is unread and says so
            cold = fresh(s)
            self.assertEqual(cold["candidates"], 0)
            self.assertGreater(cold["squares"]["unread"], 0)
            self.assertIsNone(cold["chosen"])

            # ...write what a read of one square leaves behind, for every square.
            def write(x, z, surface=False):
                h = 64 + ((np.indices((size, size)).sum(axis=0)
                           + abs(x) // 64 + abs(z) // 32) % 11)
                extra = ({"surface_codes": np.zeros((size, size), np.int32),
                          "surface_palette": np.array(["grass_block"], dtype=str)}
                         if surface else {})
                np.savez_compressed(
                    fs.tile_cache(x, z, size, size, d), x0=x, z0=z, h=h,
                    wet=np.zeros((size, size), bool),
                    canopy=np.zeros((size, size), bool),
                    manmade=np.zeros((size, size), np.int64),
                    occupied=np.full((size, size), 100, np.int64), **extra)
            for (x, z) in fs.candidates_at(1024, size, 512):
                write(x, z)

            one = fresh(s)
            two = fresh(s)
            self.assertEqual(json.dumps(one, sort_keys=True),
                             json.dumps(two, sort_keys=True))
            self.assertGreater(one["candidates"], 0)
            self.assertEqual(one["squares"]["unread"], 0)
            self.assertEqual(one["squares"]["server"], 0)
            # A square read before the cache carried its surface is **stale**: still
            # scored on everything a hamlet with no setting asks, and counted as such.
            self.assertEqual(one["squares"]["stale"], one["candidates"])
            self.assertEqual(one["squares"]["cache"], 0)
            self.assertIsNotNone(one["chosen"])
            self.assertFalse(one["chosen"]["measures"]["surface"]["read"])
            # ...and the cap on making new ground is off unless it is asked for
            self.assertEqual(one["max_new"], 0)

            # ...but against a setting a stale square cannot be certified, and the
            # search says so by name rather than scoring an unread surface at zero
            want = spec.read_spec(dict(fs.CHECK_SPEC, setting={"surface": "green"}),
                                  fs.CHECK_SENTENCE)
            no = fresh(want)
            self.assertEqual(no["meeting"], 0)
            self.assertIn("surface unread",
                          no["top"][0]["excess"]["setting_failures"][0])
            self.assertEqual(no["setting"]["surface"], "green")
            # ...and a square read since carries it, and meets
            for (x, z) in fs.candidates_at(1024, size, 512):
                write(x, z, surface=True)
            yes = fresh(want)
            self.assertEqual(yes["squares"]["cache"], yes["candidates"])
            self.assertEqual(yes["squares"]["stale"], 0)
            self.assertGreater(yes["meeting"], 0)
            self.assertEqual(yes["chosen"]["measures"]["surface"]["classes"]["green"],
                             100.0)

    def test_relief_band_scores_real_cached_patches(self):
        # Fixed 40-column diagnostic windows match the registered terrain bank's scale.
        # This tests the ranking rule, not the capacity of a forty-column village.
        needs = {"max_relief": 100, "max_water_pct": 100, "max_forest_pct": 100,
                 "relief_band": groundread.RELIEF_BANDS["village"]}
        rows = []
        for name, field in fs.cached_fields(log=lambda *a: None):
            for i in range(0, field.shape[0] - 39, 32):
                for j in range(0, field.shape[1] - 39, 32):
                    m = fs.measure(field, field.x0 + i, field.z0 + j, 40, 12)
                    m["site"] = name
                    rows.append({"measures": m, "excess": fs.excess(m, needs, plateau_relief=100)})
        flat = sorted([r for r in rows if r["measures"]["relief"] < 8], key=fs.rank_key)
        moderate = sorted([r for r in rows if 8 <= r["measures"]["relief"] <= 25], key=fs.rank_key)
        self.assertTrue(flat and moderate)
        # Compare the relief component independently of exclusion and gravity
        # differences.
        self.assertLess(moderate[0]["excess"]["relief_preference"],
                        flat[0]["excess"]["relief_preference"])
        self.assertLess(fs.rank_key(moderate[0]), fs.rank_key(flat[0]))
        print("relief comparison:", json.dumps({"moderate": moderate[0], "flat": flat[0]}))

    def test_failed_search_does_not_publish_an_ineligible_choice(self):
        """A fully refused scan is a stop, not a best-effort site coordinate."""
        measures = {"x": 0, "z": 0, "size": 48, "relief": 9,
                    "plateau": {"relief": 0}, "man_made_share": 0.01}
        rows = [{"measures": measures,
                 "excess": {"total": 0, "meets": False,
                            "virgin_failures": ["man-made share exceeds 0.5%"]}}]
        got = fs._answer({"sentence": "x", "kind": "village",
                          "needs": {}, "defining_parts": []},
                         "nothing met the needs", 48, 12, rows, [], None, 16,
                         failed=True)
        self.assertTrue(got["failed"])
        self.assertIsNone(got["chosen"])

    def test_plan_only_round_cannot_reach_a_build_stage(self):
        rnd = pipeline.Round(name="proof", sentence="a place", flags={"plan_only": True})
        # `reading` is the architecture round's first plan-layer stage: what the
        # sentence requires outright, and what was found out about it, before the spec
        # is written. A plan-only round still stops at `plan`. ...and `interpret` is the
        # realization round's, between them: what the sentence *means*, read by an agent
        # and cross-checked by the rules, before the programme is designed from it.
        self.assertEqual(pipeline.default_stages(rnd),
                         ("reading", "interpret", "place_spec", "site_search", "site",
                          "plateau", "plan"))


if __name__ == "__main__":
    unittest.main()
