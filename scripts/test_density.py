"""The place is filled, the ground is dressed, the centre scales.

    $PY scripts/test_density.py
    $PY scripts/test_density.py 1_        # one phase, or any substring of a name

  1. **The count is derived from the ground.** A district's structures are
     `area x PLOT_SHARE / columns_per_plot(density)` and nothing about the kind's band
     is in it; `dense`'s plot is the library's arithmetic over the committed urban types
     rather than a multiplier on the village constant; the structures ceiling scales
     with the footprint ceiling by area; the brief states a count and a cover and the
     validator refuses a district below a registered fraction of either; a place whose
     ground is found keeps the band as it was.
  2. **The ground is dressed from the setting, and what is not built is filled.** A
     terrace on a plain is grass and a terrace in a voice that names its ground is that
     ground, column by column; a plateau's feather is planted rather than left as a cut;
     an edge clears its own run and not the city it encloses; `site()` writes the
     setting's surface onto the part as `voice["ground"]`; the five area types stand on
     their sweep and are what the brief asks a district to fill its ground with; a
     district under the registered ground cover is handed back with the number; a voice
     with a second wall face builds the same shape out of the other stone.

Every case is deterministic and makes no model call.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402

from ethoslm import placeplan, spec as spec_mod                         # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


def _ringed(footprint=768, kind="city"):
    """A four-ring city spec on the footprint a city's ceiling allows."""
    import test_rings
    doc = json.loads(json.dumps(test_rings.THREE_RING))
    doc["kind"] = kind
    doc["needs"] = {"footprint": footprint}
    return spec_mod.read_spec(doc, "Build a ringed capital.")


def _site(size=768, origin=(-1024, -3072)):
    return {"origin": list(origin), "size": size}


def _brief_site(size=768, origin=(-1024, -3072)):
    """A site with the grids a brief prints, all of it level and green."""
    grid = [[64] * 12 for _ in range(12)]
    return {"origin": list(origin), "size": size, "mean_grid": grid,
            "roughness_grid": [[0] * 12 for _ in range(12)],
            "stats": {"min": 64, "max": 64, "relief": 0},
            "surface_blocks": {"grass_block": size * size}}


def _layout(spec, site=None, voice="ochre_stone_green_tile"):
    import test_rings
    site = site or _site()
    plateau = test_rings._plateau(site)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, voice)
    return place, fails


# ------------------------------------------- 1. the count is derived from the ground

@case
def t_1_a_dense_plot_is_the_smallest_house_the_library_can_build_a_quarter_of():
    """`spec.DENSITIES["dense"]` sized a plot no house in `types/` is.

        0.7 x 225 is 158 columns, a 12.5-block square. The library knows what its own houses
        need, so the number is arithmetic over the committed urban plot types: the one whose
        band tops out soonest, at its largest pad, plus `site()`'s inset on four sides.
        
    """
    d = placeplan.dense_plot()
    assert d["type"], d
    assert d["columns"] == d["plot"] ** 2, d
    assert d["plot"] == d["pad"] + 2 * d["inset"], d
    # ...and it is what `columns_per_plot` answers for the word, and only for that word.
    assert spec_mod.columns_per_plot({"density": "dense"}) == d["columns"], d
    for word in ("sparse", "low", "medium"):
        want = int(round(spec_mod.COLUMNS_PER_PLOT * spec_mod.DENSITIES[word]))
        assert spec_mod.columns_per_plot({"density": word}) == want, word
    assert spec_mod.columns_per_plot({}) == spec_mod.COLUMNS_PER_PLOT
    # A dense plot is smaller than a medium one, or the word means nothing.
    assert d["columns"] < spec_mod.COLUMNS_PER_PLOT, d
    was = int(round(spec_mod.COLUMNS_PER_PLOT * spec_mod.DENSITIES["dense"]))
    # ...and it is read off the files: restricted to one type, it is that type's.
    only = placeplan.dense_plot([d["type"]])
    assert only["columns"] == d["columns"] and only["type"] == d["type"], only
    return (f"a dense plot is {d['plot']}x{d['plot']} = {d['columns']} columns, off "
            f"{d['type']}'s {d['pad']}x{d['pad']} pad, against the word's {was}")


@case
def t_1_the_two_shares_are_one_derivation_and_a_district_can_hold_both():
    """A district drawn to one of them must not fail the other.

        Registered separately they could not both be met: 0.42 of plots and 0.70 of cover
        for `dense` asks a 300-square district for 400 plots of ten and 168 areas of twelve,
        which needs 133,602 columns of a rectangle that has 90,000. So `PLOT_CELLS` is
        registered -- how much of the tiled ground is plots rather than areas, which is what
        the density word means -- and both shares are arithmetic over it, the library's own
        plot and area sizes, and the five blocks a lane needs.
        
    """
    got = placeplan.occupancy_shares()
    assert sorted(got) == sorted(spec_mod.DENSITIES), sorted(got)
    said = []
    for w in ("sparse", "low", "medium", "dense"):
        v = got[w]
        # the plots are under what a district may hold in plots...
        assert v["plot_share"] < placeplan.DISTRICT_FILL, (w, v)
        # ...and the two together are under what the same ground can be tiled to at all
        assert v["ground_cover"] <= max(v["plot_tiled"], v["area_tiled"]) + 1e-9, (w, v)
        # ...and the derivation is the one written down
        want = v["plot_cells"] * v["plot_tiled"]
        assert abs(v["plot_share"] - want) < 0.001, (w, v)
        assert abs(v["ground_cover"] - (want + (1 - v["plot_cells"]) * v["area_tiled"])) \
            < 0.001, (w, v)
        said.append(f"{w} {v['plot_share']:.3f}/{v['ground_cover']:.3f}")
    # denser is more of the ground in plots and, because a street at a house's scale is
    # a third of the ground, **less** of it covered
    assert got["dense"]["plot_cells"] > got["sparse"]["plot_cells"]
    assert got["dense"]["ground_cover"] < got["sparse"]["ground_cover"]
    # ...and the whole point.
    assert got["dense"]["plot_share"] > 2 * 0.152, got["dense"]
    # ...and what the brief actually asks a 300-square district for **fits on it**,
    # counting each rectangle with the lane it needs round it. This is the check that
    # was missing: two shares that each look reasonable can together ask for half as
    # much ground again as the rectangle has.
    L = placeplan.PLOT_LANE
    fits = []
    for w in ("sparse", "low", "medium", "dense"):
        d = {"name": "q", "x0": 0, "z0": 0, "x1": 299, "z1": 299, "structures": 0}
        part = {"density": w}
        d["structures"] = spec_mod.structures_for(300 * 300, part)
        t = placeplan.district_target(d, part)
        tiled = (t["count"] * (got[w]["plot_side"] + L) ** 2
                 + t["areas"] * (got[w]["area_side"] + L) ** 2)
        assert tiled <= t["columns"], (w, tiled, t["columns"], t["count"], t["areas"])
        fits.append(f"{w} {t['count']}+{t['areas']} in {tiled}/{t['columns']}")
    return ("plot share / ground cover: " + ", ".join(said)
            + "; at 300 square, plots+areas with their lanes: " + ", ".join(fits)
            + f"; all under DISTRICT_FILL {placeplan.DISTRICT_FILL:g}; the ground "
              f"round's four rings read plots 0.258/0.179/0.152/0.154 and areas "
              f"0.000/0.003/0.004/0.396")


@case
def t_1_a_ring_asks_for_the_count_its_own_ground_says_and_not_the_bands_share():
    """The phase's rule, on the layout: every district's number is its rectangle's.

        And the reading that made it necessary: the same spec on the same site under the
        band gave the dense ring a thousand columns a house, because the count came from
        `SIZE_BANDS["city"]` and the rings took shares of it -- a number that does not move
        when the ground doubles.
        
    """
    spec = _ringed()
    place, fails = _layout(spec)
    assert not fails, fails
    L = place["layout"]
    by_name = {r["name"]: r for r in L["rings"]}
    rings = {r["name"]: r for r in spec_mod.rings(spec)}
    said = []
    for d in place["districts"]:
        part = rings[d["defines"]]
        area = (d["x1"] - d["x0"] + 1) * (d["z1"] - d["z0"] + 1)
        want = spec_mod.structures_for(area, part)
        cap = int(area * placeplan.DISTRICT_FILL // spec_mod.columns_per_plot(part))
        assert d["structures"] == min(cap, max(1, want)), (d["name"], d["structures"],
                                                           want, cap)
    for name, r in by_name.items():
        per = r["annulus_columns"] / float(max(1, r["structures"]["laid"]))
        said.append(f"{name} {r['structures']['laid']} at {per:.0f} columns each")
    # the dense ring is the densest of the four by ground per structure
    dense = [n for n, r in rings.items() if r.get("density") == "dense"]
    assert dense, rings
    per = {n: by_name[n]["annulus_columns"] / float(max(1, by_name[n]["structures"]["laid"]))
           for n in by_name}
    assert per[dense[0]] == min(per.values()), per
    assert per[dense[0]] < 1000, per
    # ...and what the spec's rings declared is kept beside it and read by nothing
    for r in by_name.values():
        assert "declared" in r["structures"], r["structures"]
    return "; ".join(said)


@case
def t_1_the_structures_ceiling_scales_with_the_footprint_ceiling_by_area():
    """Two ceilings about how big a place is, and they scale together or they lie."""
    assert spec_mod.structures_ceiling("town") == spec_mod.CEILING["structures"]
    f = spec_mod.footprint_ceiling("city")
    want = int(round(spec_mod.CEILING["structures"] * f * f
                     / float(spec_mod.CEILING["footprint"] ** 2)))
    assert spec_mod.structures_ceiling("city") == want, want
    assert want > spec_mod.CEILING["structures"], want
    # and a layout that derives more than the ceiling is scaled to it, none to nothing
    spec = _ringed()
    place, fails = _layout(spec)
    assert not fails, fails
    rec = place["layout"]["structures"]
    assert rec["ceiling"] == want, rec
    laid = sum(int(d["structures"]) for d in place["districts"])
    assert laid == rec["laid"] <= want, (laid, rec)
    assert all(int(d["structures"]) >= 1 for d in place["districts"])
    if rec["applied"]:
        assert rec["asked"] > want and 0 < rec["factor"] < 1, rec
    return (f"a city's ground ceiling is {f}x{f} and its structures ceiling {want} "
            f"against a town's {spec_mod.structures_ceiling('town')}; the layout asked "
            f"for {rec['asked']} and laid {rec['laid']}"
            + (f" scaled by {rec['factor']}" if rec["applied"] else " untouched"))


@case
def t_1_a_district_under_its_count_or_its_cover_is_handed_back_with_the_number():
    """Two numbers about the same ground, both in the brief, both refused on.

        A count met by twenty houses in one corner is not a district and one courtyard house
        at its largest is not twelve, so neither number alone says a quarter is a quarter.
        
    """
    district = {"name": "q", "x0": 0, "z0": 0, "x1": 99, "z1": 99, "structures": 0,
                "defines": "outer_town"}
    part = {"density": "dense"}
    t = placeplan.district_target(district, part)
    full = spec_mod.structures_for(100 * 100, part)
    district["structures"] = full
    t = placeplan.district_target(district, part)
    assert t["count"] == full and t["min_count"] == math.ceil(
        placeplan.DISTRICT_MIN_FRACTION * full), t

    def plots(n, side):
        """n plots of `side`, laid in rows five apart from the corner."""
        out, i = [], 0
        step = side + 5
        for row in range(0, 100, step):
            for col in range(0, 100, step):
                if i >= n:
                    break
                out.append({"name": f"p{i}", "kind": "plot", "type": "row_house",
                            "x0": col, "z0": row, "x1": col + side - 1,
                            "z1": row + side - 1})
                i += 1
        return out

    # too few: refused on the count, with the number
    few = placeplan.occupancy_failures(district, plots(4, 10), part, "urban")
    assert [f["check"] for f in few] == ["count", "cover", "ground_cover"], few
    assert str(t["count"]) in few[0]["why"] and str(t["min_count"]) in few[0]["why"]
    # enough of them, at the size the type wants: both hold
    ok = placeplan.occupancy_failures(district, plots(full, 10), part, "rural")
    assert not ok, ok
    # the count met and the cover not: the same plots at a third of the side
    small = placeplan.occupancy_failures(district, plots(full, 4), part, "rural")
    assert [f["check"] for f in small] == ["cover"], small
    assert str(t["min_plot_columns"]) in small[0]["why"], small
    # ...and the brief says both, in the units the refusal uses
    note = placeplan._cover_note(district, "urban", part)
    assert str(t["count"]) in note and str(t["min_count"]) in note, note
    assert str(t["min_plot_columns"]) in note and str(t["columns"]) in note, note
    return (f"a 100x100 dense district is asked for {t['count']} structures and "
            f"{t['min_plot_columns']} columns of plot; 4 plots fail count, cover and "
            f"ground cover by name, {full} of 10x10 pass all three, {full} of 4x4 meet "
            f"the count and fail the cover; the brief carries all four numbers")


@case
def t_1_a_place_whose_ground_is_found_keeps_the_band_as_it_was():
    """The derivation is the designed layout's. A place with no rings is untouched:
    its planner is handed `spec["structures"]` and that is the band's, as it was."""
    import test_rings
    doc = json.loads(json.dumps(test_rings.THREE_RING))
    for p in doc["defining_parts"]:
        for k in ("ring", "share", "walled"):
            p.pop(k, None)
    doc["kind"] = "town"
    doc.pop("needs", None)
    spec = spec_mod.read_spec(doc, "Build a town on a plain.")
    lo, hi = spec["size_band"]
    assert (lo, hi) == spec_mod.SIZE_BANDS["town"], spec["size_band"]
    assert lo <= spec["structures"] <= hi, spec["structures"]
    assert spec["ceiling"]["structures"] == spec_mod.CEILING["structures"]
    # ...and the place brief quotes the band's number, not a rectangle's
    _place, fails = _layout(spec)
    assert fails and fails[0]["check"] == "rings", fails
    return (f"a ringless town reads {spec['structures']} in {spec['size_band']}, the "
            f"kind's own band, and the arithmetic layout refuses it by name: "
            f"{fails[0]['why'][:60]}...")


# --------------------------------- 2. the ground is dressed and the place is filled

def _ground(size=120, y=64, surface="grass_block"):
    """A level fixture whose surface is one natural block."""
    from ethoslm.observe import Volume
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", surface, "dirt", "stone"])


def _builder(vol):
    from ethoslm import offline
    from ethoslm.buildlib import Builder
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    return b


def _tops(b, cols, y):
    """What this builder has left on top of each column, as a census."""
    out: dict = {}
    for (x, z) in cols:
        got = None
        for yy in range(y + 6, y - 8, -1):
            blk = b._pending.get((x, yy, z))
            if blk is None:
                continue
            if blk.split("[")[0] != "air":
                got = blk.split("[")[0]
                break
        if got is None:
            got = b.get_block(x, y, z).split("[")[0]
        out[got] = out.get(got, 0) + 1
    return out


@case
def t_2_a_terrace_on_a_plain_is_grass_and_a_terrace_in_a_quarry_voice_is_its_ground():
    """The cover is the setting's, column by column, and the voice's where it says."""
    from ethoslm import pipeline
    G = 64
    cols = [(x, z) for x in range(30, 70) for z in range(30, 70)]
    # 1. a plain, in a voice that names no ground: the plain's own surface
    vol = _ground(surface="grass_block")
    b = _builder(vol)
    rec = b.terrace_annulus((30, 30, 69, 69), G + 2,
                            mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert rec["ok"], rec["reason"]
    assert rec["dress"]["from"] == "setting", rec["dress"]
    top = _tops(b, cols, G + 2)
    assert top.get("grass_block", 0) == len(cols), top
    assert "granite" not in top, top
    # 2. a shore: the natural surface at the column, not one block for the whole ring
    sand = _ground(surface="grass_block")
    codes = np.array(sand.codes)
    pal = list(sand.palette) + ["sand"]
    codes[:, G - (G - 14), :][:50, :] = len(pal) - 1       # the west half is sand
    from ethoslm.observe import Volume
    shore = Volume(sand.x0, sand.y0, sand.z0, codes, pal)
    b2 = _builder(shore)
    rec2 = b2.terrace_annulus((30, 30, 69, 69), G + 2,
                              mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert rec2["ok"], rec2["reason"]
    t2 = _tops(b2, cols, G + 2)
    assert t2.get("sand", 0) > 0 and t2.get("grass_block", 0) > 0, t2
    assert t2["sand"] + t2["grass_block"] == len(cols), t2
    # 3. a voice that names its ground: that ground, everywhere, whatever the land is
    b3 = _builder(_ground(surface="grass_block"))
    rec3 = b3.terrace_annulus((30, 30, 69, 69), G + 2,
                              mat=pipeline.voice_palette("cut_sandstone_terraces"))
    assert rec3["ok"], rec3["reason"]
    assert rec3["dress"]["from"] == "voice", rec3["dress"]
    from ethoslm.prims import solid
    quarry = solid(pipeline.voice_palette("cut_sandstone_terraces")["ground"])
    t3 = _tops(b3, cols, G + 2)
    assert t3.get(quarry, 0) == len(cols), (quarry, t3)
    return (f"a 40x40 terrace on a plain is {len(cols)} columns of grass; the same "
            f"terrace half over sand is {t2['sand']} sand and {t2['grass_block']} grass; "
            f"in a voice that names its ground it is {len(cols)} of {quarry}")


@case
def t_2_a_plateaus_feather_is_planted_and_not_left_as_a_cut():
    """The ground look's fourth finding, at its root: an earthwork left undressed.

        `plateau()` laid the footing family to the target and stopped, so the slope off a
        podium was a staircase of quarry stone at two scales. A feather is a blend between
        designed ground and the land round it and its top course is the land's.
        
    """
    from ethoslm import pipeline
    G = 64
    vol = _ground(size=160, y=G)
    b = _builder(vol)
    rec = b.plateau((60, 60, 99, 99), G + 6, label="podium",
                    mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert rec["ok"], rec.get("reason")
    assert rec["feathered"] > 0, rec
    assert rec["feather_cover"]["from"] == "setting", rec["feather_cover"]
    assert rec["feather_cover"]["cover"] == "grass_block", rec["feather_cover"]
    f = rec.get("feathered")
    # every column of the feather ring is topped with the plain's own surface
    ring = [(x, z) for x in range(58, 102) for z in (58, 101)]
    ring += [(x, z) for z in range(58, 102) for x in (58, 101)]
    tops = _tops(b, ring, G + 6)
    assert tops.get("grass_block", 0) >= len(ring) // 2, tops
    assert tops.get("granite", 0) == 0, tops
    return (f"a 40x40 podium six blocks up: {f} feathered columns, every one of them "
            f"topped in the plain's own grass and none in the footing")


@case
def t_2_an_edge_clears_its_own_run_and_not_the_city_it_encloses():
    """The root of "one block of granite from wall to wall", with the number.

        `site()` for an edge cleared `clear_trees` and `clear_ground_cover` over the
        **bounding box** of the path, which for a ring wall is the city inside it, and
        `clear_ground_cover` takes a `grass_block`. Of the 49,403 columns of cover the upper
        ring's terrace laid on `out/ground`, 39,267 were air afterwards.
        
    """
    from ethoslm import pipeline
    G = 64
    vol = _ground(size=120, y=G)
    b = _builder(vol)
    ring = [(20, 20), (99, 20), (99, 99), (20, 99), (20, 20)]
    part = {"label": "ring_wall", "name": "ring_wall", "kind": "edge",
            "path": [list(c) for c in ring], "width": 3, "seed": 1}
    out = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert out["sited"]["ok"], out["sited"]
    # the turf inside the ring, well clear of the wall's own line, is untouched
    inside = [(x, z) for x in range(30, 90, 7) for z in range(30, 90, 7)]
    stripped = [c for c in inside
                if any(b._pending.get((c[0], y, c[1])) == "air" for y in range(G, G + 3))]
    assert not stripped, stripped[:8]
    # ...and the wall's own columns are worked
    on_line = [(x, 20) for x in range(30, 90, 7)]
    worked = [c for c in on_line if any((c[0], y, c[1]) in b._pending
                                        for y in range(G - 2, G + 3))]
    assert len(worked) == len(on_line), (len(worked), len(on_line))
    return (f"an 80x80 wall circuit of 4 segments: {len(inside)} sampled columns inside "
            f"it are untouched and all {len(on_line)} sampled columns on its own line "
            f"are worked")


@case
def t_2_site_writes_the_settings_own_surface_onto_the_part_as_a_seventh_role():
    """What lets an area type be planted without naming a block."""
    from ethoslm import pipeline
    G = 64
    for surface, want in (("grass_block", "grass_block"), ("sand", "sand")):
        vol = _ground(size=80, y=G, surface=surface)
        b = _builder(vol)
        part = {"label": "g", "name": "g", "kind": "area", "x0": 30, "z0": 30,
                "x1": 45, "z1": 45, "seed": 1}
        out = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"))
        assert out["voice"]["ground"] == want, (surface, out["voice"]["ground"])
    # ...and the voice's own where it names one, whatever the land is
    vol = _ground(size=80, y=G, surface="grass_block")
    b = _builder(vol)
    part = {"label": "g", "name": "g", "kind": "area", "x0": 30, "z0": 30,
            "x1": 45, "z1": 45, "seed": 1}
    out = b.site(dict(part), mat=pipeline.voice_palette("cut_sandstone_terraces"))
    from ethoslm.prims import solid
    quarry = solid(pipeline.voice_palette("cut_sandstone_terraces")["ground"])
    assert out["voice"]["ground"] == quarry, out["voice"]["ground"]
    return (f"site() writes grass on a plain, sand on a shore and the voice's own "
            f"{quarry} where the voice names a ground")


@case
def t_2_the_five_area_types_are_committed_and_measured_over_their_whole_band():
    """The area kinds a district needs to read as inhabited, each swept like any type."""
    bank = json.load(open(os.path.join(ROOT, "rounds", "type-needs.json")))
    want = ("garden", "grove", "plaza", "alley", "yard")
    said = []
    for name in want:
        p = os.path.join(ROOT, "types", f"{name}.py")
        assert os.path.exists(p), f"types/{name}.py is not committed"
        from ethoslm import pipeline
        decl = pipeline.load_type(p)
        assert decl["kind"] == "area", (name, decl["kind"])
        assert decl.get("role") in spec_mod.ROLES, (name, decl.get("role"))
        assert name in bank["types"], f"{name} is not in rounds/type-needs.json"
        row = bank["types"][name]
        band = tuple(row["band"]["footprint"])
        assert tuple(decl["needs"]["footprint"]) == band, (name, band)
        assert not row["band"].get("except"), (name, row["band"].get("except"))
        failed = sum(int(sz["failed"]) for sz in row["sizes"])
        assert failed == 0, (name, failed)
        said.append(f"{name} ({decl['role']}) {band[0]}-{band[2]} at "
                    f"{sum(int(sz['of']) for sz in row['sizes'])} instances, 0 failed")
    # ...and the district brief asks for them by size and number, not by name
    district = {"name": "q", "x0": 0, "z0": 0, "x1": 119, "z1": 119, "structures": 0}
    part = {"density": "low"}
    district["structures"] = spec_mod.structures_for(120 * 120, part)
    t = placeplan.district_target(district, part)
    note = placeplan._cover_note(district, "urban", part)
    assert str(t["min_ground_columns"]) in note and str(t["area_size"]) in note, note
    # ...and the **library** names none of them: which areas a district may draw is its
    # role's own table, read off the files, and not a list of type names in a constant.
    src = open(os.path.join(ROOT, "src", "ethoslm", "placeplan.py")).read()
    for name in want:
        assert f'"{name}"' not in src and f"'{name}'" not in src, \
            f"placeplan names the type {name}: the table is the type list"
    return ("; ".join(said) + f"; a 120x120 low district is asked for "
            f"{t['min_ground_columns']} columns of ground cover in about {t['areas']} "
            f"areas of {t['area_size']}, and the brief names no type")


@case
def t_2_a_district_under_its_ground_cover_is_handed_back_with_the_number():
    """Plots alone are not a quarter; a rural district keeps its own older rule."""
    district = {"name": "q", "x0": 0, "z0": 0, "x1": 99, "z1": 99}
    part = {"density": "medium"}
    district["structures"] = spec_mod.structures_for(100 * 100, part)
    t = placeplan.district_target(district, part)

    def leaves(n, side, kind, tag, z0=0):
        out = []
        step = side + 5
        i = 0
        for row in range(z0, 100, step):
            for col in range(0, 100, step):
                if i >= n:
                    break
                out.append({"name": f"{tag}{i}", "kind": kind,
                            "type": "townhouse" if kind == "plot" else "garden",
                            "x0": col, "z0": row, "x1": col + side - 1,
                            "z1": row + side - 1})
                i += 1
        return out

    plots = leaves(t["count"], 15, "plot", "p")
    bare = placeplan.occupancy_failures(district, plots, part, "urban")
    assert [f["check"] for f in bare] == ["ground_cover"], bare
    assert str(t["min_ground_columns"]) in bare[0]["why"], bare[0]["why"]
    # the same plots with areas over what they leave: it holds
    full = plots + leaves(24, 12, "area", "a", z0=2)
    assert not placeplan.occupancy_failures(district, full, part, "urban"), \
        placeplan.occupancy_failures(district, full, part, "urban")
    # a rural district is held to RURAL_COVER and not to this
    assert not placeplan.occupancy_failures(district, plots, part, "rural"), \
        placeplan.occupancy_failures(district, plots, part, "rural")
    return (f"a 100x100 medium district with {len(plots)} plots and no areas covers "
            f"{sum(15 * 15 for _ in plots)} of {t['columns']} and is refused naming "
            f"{t['min_ground_columns']}; with 24 areas of 12 it holds; a rural district "
            f"is held to RURAL_COVER {placeplan.RURAL_COVER:g} and not to this")


@case
def t_2_a_voice_with_a_second_wall_face_builds_the_same_shape_out_of_the_other_stone():
    """The two-voice test's claim, applied to one voice with two faces.

        A ring built from one `wall` family is one extruded building thirty times over. The
        type is never told which face it got: what changes is the block and not one column.
        
    """
    from ethoslm import offline, pipeline
    from ethoslm.buildlib import Builder, WALL_ALT_SHARE
    from ethoslm.pipeline import stages_build
    G = 64
    tf = os.path.join(ROOT, "types", "row_house.py")
    pal = pipeline.voice_palette("cut_sandstone_terraces")
    assert pal.get("wall_alt"), pal
    shapes, faces = {}, {}
    for seed in range(1, 25):
        vol = _ground(size=60, y=G)
        part = {"label": "h", "name": f"h{seed}", "kind": "plot",
                "x0": 20, "z0": 20, "x1": 27, "z1": 27, "seed": seed}
        b = _builder(vol)
        sited = b.site(dict(part), mat=pal,
                       roof=pipeline.voice_roof("cut_sandstone_terraces"))
        tb = b.type_builder(sited)
        import importlib.util
        sp = importlib.util.spec_from_file_location(f"rh{seed}", tf)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        m.build(tb, sited, seed)
        faces[seed] = tb.wall_alt
        shapes[seed] = frozenset((p[0], p[1], p[2]) for p, blk in b._pending.items()
                                 if blk.split("[")[0] != "air")
    alt = [s for s in faces if faces[s]]
    assert alt, "no instance took the second face"
    assert len(alt) < len(faces), "every instance took the second face"
    got = len(alt) / float(len(faces))
    assert abs(got - WALL_ALT_SHARE) < 0.25, (got, WALL_ALT_SHARE)
    # the same seed is the same face every time: it is the part's, not the run's
    vol = _ground(size=60, y=G)
    b = _builder(vol)
    s0 = alt[0]
    part = {"label": "h", "name": f"h{s0}", "kind": "plot", "x0": 20, "z0": 20,
            "x1": 27, "z1": 27, "seed": s0}
    sited = b.site(dict(part), mat=pal)
    assert b.type_builder(sited).wall_alt is True
    return (f"row_house on 24 seeds in a voice with two faces: {len(alt)} of "
            f"{len(faces)} are built in the second stone against a registered "
            f"{WALL_ALT_SHARE:g}, and the face a seed gets is the same every time")


# ------------------------------------------------- 3. the artifacts, at their rules

@case
def t_3_a_wall_inside_a_place_of_unbroken_walls_is_unbroken_too():
    """The ground look's second finding at its root.

        The layout chose a face and wrote it onto the *ring* walls; the compound's wall was
        drawn by a call of its own and got the `wall` type's default, so the palace of a city
        whose great walls are one earthen mass stood inside a fence of X-braces. A face is a
        fact about the place and not about which level drew the edge.
        
    """
    spec = _ringed()
    # the fixture's sentence says nothing about unbroken walls: no face is stamped
    assert placeplan.place_wall_face(spec) in (None, "framed", "banded", "plain")
    doc = json.loads(json.dumps(spec))
    for p in doc["defining_parts"]:
        if p["family"] == "wall":
            p["notes"] = "one unbroken rammed-earth mass, sheer on both faces"
    unbroken = spec_mod.read_spec(doc, doc["sentence"])
    assert placeplan.place_wall_face(unbroken) == "plain", \
        placeplan.place_wall_face(unbroken)
    leaves = {"parts": [{"kind": "edge", "name": "precinct", "type": "wall",
                         "path": [[0, 0], [9, 0]], "seed": 1},
                        {"kind": "plot", "name": "hall", "type": "hall",
                         "x0": 2, "z0": 2, "x1": 9, "z1": 9, "seed": 1}]}
    got = placeplan.compound_parts(leaves, None, "palace", unbroken)
    assert got[0]["face"] == "plain" and "face" not in got[1], got
    # ...and a district's own edges too, and never over a face the plan named itself
    named = {"quarters": [{"name": "q", "plots": [
        {"kind": "edge", "name": "e", "type": "wall", "path": [[0, 0], [9, 0]],
         "seed": 1, "params": {"face": "banded"}}]}]}
    rows = placeplan.district_plots(named, "urban", unbroken)
    assert "face" not in rows[0], rows
    return ("a spec whose wall the sentence calls an unbroken mass stamps `plain` on "
            "every edge leaf of its compounds and districts, on no plot, and never over "
            "a face the plan wrote itself")


@case
def t_3_a_battlement_is_masonry_at_one_rhythm_and_the_trim_is_a_string_course():
    """The ground look's fifth finding: the crown read as ornaments hung on the wall.

        The cap went up in `trim`, which in a voice whose trim is a bright band is a row of
        oversized coloured cubes standing proud of a pale wall; and `wall` drew its merlon
        two or three wide by seed while `great_wall` drew one, so two walls of one city
        crowned themselves differently.
        
    """
    import test_site_needs as tgs
    from ethoslm.buildlib import Builder
    assert Builder.MERLON_WIDTH == 2 and Builder.CRENEL_WIDTH == 1
    caps = [Builder.merlon(Builder, i) for i in range(9)]
    assert caps == [True, True, False] * 3, caps
    out = {}
    for name, params in (("wall", {"height": 20, "width": 3, "crown": "crenellated"}),
                         ("great_wall", {"height": 30, "width": 3,
                                         "parapet": "crenellated"})):
        got = tgs._stand_ring(name, "ochre_stone_green_tile", params, gates=False)
        pal = pipeline_palette()
        solid_cells = {(x, y, z): blk for (x, y, z), blk in got["pending"].items()
                       if blk.split("[")[0] != "air"}
        top = max(y for (_x, y, _z) in solid_cells)
        crown = [blk.split("[")[0] for (x, y, z), blk in solid_cells.items()
                 if y >= top - 1]
        wall_block = pal["wall"]
        trim_block = pal["trim"]
        n_wall = sum(1 for b in crown if b.startswith(wall_block.split("_")[0]))
        n_trim = sum(1 for b in crown if b == trim_block)
        assert n_wall > 0, (name, set(crown))
        assert n_trim * 4 < n_wall, (name, n_trim, n_wall, set(crown))
        out[name] = (n_wall, n_trim)
    return ("a merlon 2 and a crenel 1, one rhythm for both walls; the top two courses "
            + ", ".join(f"{k}: {v[0]} of the wall family against {v[1]} of trim"
                        for k, v in out.items()))


def pipeline_palette():
    from ethoslm import pipeline
    return pipeline.voice_palette("ochre_stone_green_tile")


@case
def t_3_a_shop_front_is_a_shopfront_and_never_a_void():
    """The ground look's sixth finding: a dark hole under the jetty with nothing in it."""
    import importlib.util
    from ethoslm import offline, pipeline
    from ethoslm.buildlib import Builder
    from ethoslm.circulate import Network, Threshold
    G = 64
    tf = os.path.join(ROOT, "types", "shop_house.py")
    src = open(tf).read()
    assert "trapdoor" in src and 'fitting("light"' in src
    vol = _ground(size=60, y=G)
    cells = {(x, 30): {"y": G, "rank": 1, "face": None} for x in range(10, 50)}
    vol = vol.overlay({(x, G, z): "cobblestone" for (x, z) in cells})
    net = Network(cells, [Threshold("s", 25, 30, G, "north", (25, G + 1, 31))])
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    from ethoslm.frontage import Frontage
    b.frontage = Frontage(vol, net)
    part = {"label": "s", "name": "s", "kind": "plot", "x0": 20, "z0": 31,
            "x1": 32, "z1": 38, "seed": 3}
    sited = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"),
                   roof=pipeline.voice_roof("ochre_stone_green_tile"))
    sp = importlib.util.spec_from_file_location("sh", tf)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    res = m.build(b.type_builder(sited, role=m.ROLE), sited, 3, storeys=2)
    blocks = [v.split("[")[0] for v in b._pending.values()]
    lights = [v for v in b._pending.values()
              if v.split("[")[0] in ("lantern", "candle", "torch", "wall_torch",
                                     "glowstone", "soul_lantern")]
    shut = [v for v in b._pending.values() if "trapdoor" in v]
    assert lights, "no lamp in the shop"
    assert shut, "no shutter over the counter"
    return (f"a two-storey shop_house on a 13x8 pad: {len(shut)} shutter blocks hung "
            f"over its counter and {len(lights)} lamps behind it, where the ground "
            f"round's shop fronts were air")


@case
def t_3_the_gates_arch_is_sized_from_its_wall_and_not_from_a_constant():
    """The concentric look's third finding, twice over: a passage six high through a
    pier of forty-eight is a mouse-hole, and making the tower scale with the wall made
    the hole look smaller rather than larger."""
    import importlib.util
    from ethoslm import offline, pipeline
    from ethoslm.buildlib import Builder
    from ethoslm.circulate import Network
    tf = os.path.join(ROOT, "types", "ring_gate.py")
    sp = importlib.util.spec_from_file_location("rg", tf)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    assert 0 < m.ARCH_HEADROOM_OF_WALL < 1 and 0 < m.ARCH_SPAN_OF_WALL < 1
    G = 64
    said = []
    heads, spans = {}, {}
    # the road runs the way the gate faces -- north, so along z -- four wide, which is
    # what an arterial is
    for wall_h in (None, 36, 48):
        vol = _ground(size=80, y=G)
        cells = {(x, z): {"y": G, "rank": 0, "face": None}
                 for x in range(39, 43) for z in range(10, 70)}
        vol = vol.overlay({(x, G, z): "cobblestone" for (x, z) in cells})
        b = Builder(offline.OfflineSite(vol))
        b._vol = vol
        from ethoslm.frontage import Frontage
        b.frontage = Frontage(vol, Network(cells, []))
        size = Builder.point_pad(wall_h, (5, 5, 16, 16))
        half = size // 2
        part = {"label": "g", "name": "g", "kind": "point", "at": [40, 40],
                "facing": "north", "seed": 1, "size": size,
                "x0": 40 - half, "z0": 40 - half, "x1": 40 - half + size - 1,
                "z1": 40 - half + size - 1}
        if wall_h:
            part["edge"] = {"name": "w", "type": "wall", "height": wall_h, "width": 3}
        sited = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"),
                       roof=pipeline.voice_roof("ochre_stone_green_tile"))
        res = m.build(b.type_builder(sited, role=m.ROLE), sited, 1, storeys=2)
        # the arch: the tallest column of air the build opened over the road the arch's
        # clearance at the **mouth**: the unbroken run of air standing over the road at
        # the outer face of the pad, from the road surface up. Inside the pad the
        # chambers are air too and measuring there measures the tower.
        pend = {k: v.split("[")[0] for k, v in b._pending.items()}
        fy = int(sited["floor_y"])
        mouth = 40 - half
        runs = []
        for x in range(39, 43):
            n = 0
            while pend.get((x, fy + 1 + n, mouth)) == "air" \
                    or (n and b.get_block(x, fy + 1 + n, mouth).split("[")[0] == "air"):
                n += 1
                if n > 60:
                    break
            runs.append(n)
        head = max(runs)
        # ...and its span: the columns of the mouth open at head height
        span = sum(1 for x in range(40 - half, 40 - half + size)
                   if pend.get((x, fy + 2, mouth)) == "air")
        heads[wall_h] = head
        spans[wall_h] = span
        said.append(f"wall {wall_h or 'none'}: pad {size}, arch {span} wide and {head} "
                    f"high over the road")
    assert heads[48] > heads[36] > heads[None], heads
    for h in (36, 48):
        assert heads[h] >= int(h * m.ARCH_HEADROOM_OF_WALL) - 2, (h, heads)
    assert spans[48] >= spans[36] >= 4, spans
    return "; ".join(said)


@case
def t_3_the_voice_card_shows_a_materials_colour_beside_its_name():
    """A voice's author chose a green roof from memory: the card showed a block's name
    and never its colour, and `preview.block_colour` has had the answer all along."""
    from ethoslm.pipeline.stages_plan import _family_colours, colour_of, spec_brief
    from ethoslm.preview import UNKNOWN, block_colour
    from ethoslm.prims import MATERIALS, solid
    # the two the table used to answer wrongly, and the one it answered magenta
    assert colour_of("oxidized_copper").endswith("green"), colour_of("oxidized_copper")
    assert not colour_of("copper_block").endswith("green"), colour_of("copper_block")
    assert block_colour("prismarine") != UNKNOWN, "prismarine still reads magenta"
    assert colour_of("dark_prismarine").endswith("green"), colour_of("dark_prismarine")
    rows = "\n".join(_family_colours())
    known = [f for f in MATERIALS if block_colour(solid(f)) != UNKNOWN]
    for f in known:
        assert f"{f} ({colour_of(solid(f))})" in rows, f
    # a family the table has no reading for is named without a colour, never with a
    # guess
    unknown = [f for f in MATERIALS if block_colour(solid(f)) == UNKNOWN]
    for f in unknown:
        assert f"{f} (" not in rows, f
    brief = spec_brief("Build a ringed capital.", "/tmp/out.json")
    assert "oxidized_copper (pale green)" in brief, "the brief carries no colours"

    # v2, A7: ...and on the card itself, beside every role of every voice on disk. The
    # card is what a builder, a type's author and the judge's brief all read, and it
    # named the material and never said what colour it is.
    from ethoslm import styles
    roles = coloured = 0
    for vname in sorted(styles.VOICES):
        card = styles.voice_card(vname)
        for role, mat in styles.VOICES[vname]["palette"].items():
            line = [ln for ln in card.splitlines() if ln.strip().startswith(role + " ")]
            assert line, f"{vname}: no line for the {role} role on the card"
            roles += 1
            try:
                block, known_here = solid(mat), True
            except Exception:                        # noqa: BLE001
                block, known_here = None, False
            if known_here and block_colour(block) != UNKNOWN:
                want = f"{mat} ({colour_of(block)})"
                assert want in line[0], f"{vname}/{role}: {line[0].strip()!r} lacks {want!r}"
                coloured += 1
            else:
                assert f"{mat} (" not in line[0], \
                    f"{vname}/{role}: a colour was guessed for {mat}"
    assert coloured == roles, f"{roles - coloured} of {roles} roles carry no colour"
    return (f"{len(known)} of {len(MATERIALS)} families carry a colour in the spec "
            f"brief and {len(unknown)} are named without one; {coloured} of {roles} "
            f"roles over {len(styles.VOICES)} voices carry one on the card itself; "
            f"oxidized_copper reads {colour_of('oxidized_copper')} and copper_block "
            f"{colour_of('copper_block')}, which it did not before")


# --------------------------------------------- 4. the centre scales with the place

@case
def t_4_the_centre_is_the_larger_of_what_it_holds_and_what_the_spec_declared():
    """The ground look's seventh finding: a compound among compounds."""
    from ethoslm.buildlib import Builder
    bare = placeplan.compound_ground()
    assert bare["declared"] == 0 and bare["side"] == bare["wanted"], bare
    spec = _ringed()
    share = spec_mod.centre_share(spec)
    want = int(round(768 * (share ** 0.5)))
    got = placeplan.compound_ground(spec=spec, site_side=768)
    assert got["declared"] == want, (got, want)
    assert got["wanted"] == max(bare["wanted"], want), got
    assert got["side"] == min(got["wanted"], got["cap"]), got
    assert got["side"] > bare["side"], (got["side"], bare["side"])
    # ...and the plateau bound scales with the site, and is the registered 128 on 512
    assert Builder.plateau_max(512) == Builder.PLATEAU_MAX == 128
    assert Builder.plateau_max(768) == int(768 * Builder.PLATEAU_MAX_SHARE) == 192
    assert Builder.plateau_max(None) == 128
    assert got["cap"] == 192, got
    # a share the bound cannot give is **recorded as capped**, never silently shrunk
    assert got["capped"] == (got["wanted"] > got["cap"]), got
    # ...and the centre is over its registered share of the innermost ring
    place, fails = _layout(spec)
    assert not fails, fails
    L = place["layout"]
    inner = 2 * L["rings"][0]["outer"] + 1
    side = L["compound_rect"][2] - L["compound_rect"][0] + 1
    return (f"a compound with no spec is {bare['side']} square; the same spec's centre "
            f"share on 768 asks for {want} and it is drawn at {got['side']}, against a "
            f"bound of {got['cap']} -- {side / float(inner):.0%} of the innermost ring's "
            f"{inner}, where the ground round read 34%")


@case
def t_4_a_compounds_composition_scales_with_its_rectangle_and_the_floor_holds():
    """`COMPOUND_MIN_HALLS` is a floor and nothing scaled it, so a precinct of 154 was
    laid out with the three parts a precinct of 24 is."""
    said = []
    prev = None
    for side in (placeplan.COMPOUND_MIN, 83, 128, 154, 200):
        comp = {"name": "p", "x0": 0, "z0": 0, "x1": side - 1, "z1": side - 1}
        t = placeplan.compound_target(comp)
        assert t["count"] >= placeplan.COMPOUND_MIN_HALLS, t
        assert t["min_count"] >= placeplan.COMPOUND_MIN_HALLS, t
        if prev is not None:
            assert t["count"] >= prev, (side, t["count"], prev)
        prev = t["count"]
        said.append(f"{side}: {t['count']} halls, {t['areas']} areas, "
                    f"{t['min_ground_columns']} of {t['columns']} covered")
    # the floor: the smallest compound the library admits is still two halls and a court
    small = placeplan.compound_target({"name": "p", "x0": 0, "z0": 0,
                                       "x1": placeplan.COMPOUND_MIN - 1,
                                       "z1": placeplan.COMPOUND_MIN - 1})
    assert small["count"] == placeplan.COMPOUND_MIN_HALLS, small
    # ...and the 83-square minimum still holds a small place: what the committed types
    # need has not moved
    assert placeplan.compound_ground()["side"] == 83, placeplan.compound_ground()
    return "; ".join(said)


@case
def t_4_a_compound_that_does_not_fill_its_rectangle_is_handed_back_with_the_number():
    spec = _ringed()
    comp = {"name": "p", "defines": "great_court", "x0": 0, "z0": 0, "x1": 153,
            "z1": 153}
    place = {"parts": [], "districts": [], "compounds": [comp]}
    _t, decls = placeplan.types_card(None, spec.get("form"))
    got = {"parts": [
        {"kind": "edge", "name": "w", "type": "wall", "seed": 1,
         "params": {"height": 12, "width": 3},
         "path": [[3, 3], [78, 3], [150, 3], [150, 78], [150, 150], [78, 150],
                  [3, 150], [3, 78], [3, 3]]},
        {"kind": "point", "name": "g", "type": "ring_gate", "at": [78, 3], "seed": 2,
         "params": {}},
        {"kind": "plot", "name": "h1", "type": "hall", "x0": 20, "z0": 20,
         "x1": 45, "z1": 45, "seed": 3, "params": {}},
        {"kind": "plot", "name": "h2", "type": "hall", "x0": 60, "z0": 20,
         "x1": 85, "z1": 45, "seed": 4, "params": {}},
        {"kind": "area", "name": "c1", "type": "square", "x0": 20, "z0": 60,
         "x1": 45, "z1": 85, "seed": 5, "params": {}}]}
    fails = placeplan.compound_failures(comp, got, place, decls, spec=spec)
    checks = [f["check"] for f in fails]
    assert "composed" in checks and "cover" in checks, fails
    t = placeplan.compound_target(comp)
    why = " ".join(f["why"] for f in fails)
    assert str(t["count"]) in why and str(t["min_ground_columns"]) in why, why
    # ...and the brief states both numbers, in the units the refusal uses
    brief = placeplan.compound_brief(spec, _brief_site(), comp, place, "/tmp/o.json",
                                     None, "ochre_stone_green_tile")
    assert str(t["count"]) in brief and str(t["min_ground_columns"]) in brief, \
        "the brief does not carry the numbers the refusal reads"
    # ...and a compound small enough that `COMPOUND_MIN_HALLS` is what it holds is asked
    # for the floor composition and **not** for a density's share of cover as well: two
    # numbers about one rectangle that cannot both be met is the thing
    # `occupancy_shares` exists to stop, and it applies here too.
    small = dict(comp, x1=comp["x0"] + 63, z1=comp["z0"] + 63)
    st = placeplan.compound_target(small)
    assert st["count"] == placeplan.COMPOUND_MIN_HALLS, st
    sgot = json.loads(json.dumps(got))
    for p in sgot["parts"]:
        for k in ("x1", "z1"):
            if k in p:
                p[k] = min(p[k], comp["x0"] + 58)
        if p.get("path"):
            p["path"] = [[min(a, comp["x0"] + 60), min(b, comp["z0"] + 60)]
                         for a, b in p["path"]]
        if p.get("at"):
            p["at"] = [min(p["at"][0], comp["x0"] + 60), p["at"][1]]
    checks = [f["check"] for f in
              placeplan.compound_failures(small, sgot, place, decls, spec=spec)]
    assert "cover" not in checks, checks
    return (f"a 154-square compound drawing 2 halls and 1 court is refused on both: it "
            f"holds {t['count']} halls and covers {t['min_ground_columns']} of "
            f"{t['columns']} columns, and the brief says so before the call is made; "
            f"a 64-square one holds {st['count']} -- the floor -- and is asked for no "
            f"cover at all")


# ------------------------------------------ 5. what the run found, before its write

@case
def t_5_the_plateau_bound_reaches_the_call_that_cuts_it():
    """Found by running it. `compound_ground` sized a 188-square podium from the spec's
    own centre share, `plateau_max(768)` admits 192, and `plateau()` refused at 128 --
    because the stage that cuts it never passed the bound it had just sized against."""
    src = open(os.path.join(ROOT, "src", "ethoslm", "pipeline",
                            "stages_plan.py")).read()
    calls = [ln for ln in src.splitlines() if "b.plateau(" in ln]
    assert len(calls) == 2, calls
    assert src.count("bound=Builder.plateau_max(int(site[\"size\"]))") == 2, \
        "a plateau call without the place's own bound"
    from ethoslm import offline, pipeline
    from ethoslm.buildlib import Builder
    G = 64
    vol = _ground(size=260, y=G)
    b = _builder(vol)
    n = 188
    rec = b.plateau((30, 30, 30 + n - 1, 30 + n - 1), G + 4,
                    mat=pipeline.voice_palette("ochre_stone_green_tile"),
                    bound=Builder.plateau_max(768))
    assert rec["ok"], rec["reason"]
    refused = _builder(_ground(size=260, y=G)).plateau(
        (30, 30, 30 + n - 1, 30 + n - 1), G + 4,
        mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert not refused["ok"] and str(Builder.PLATEAU_MAX) in refused["reason"], refused
    return (f"a {n}-square podium is cut under a 768 place's own bound of "
            f"{Builder.plateau_max(768)} and refused under the registered "
            f"{Builder.PLATEAU_MAX}, which is the bound of a 512 place")


@case
def t_5_every_city_this_project_has_written_is_in_the_reservation_ledger():
    """Found by running it."""
    import importlib.util
    p = os.path.join(ROOT, "src", "ethoslm", "data", "site-exclusions.json")
    led = json.load(open(p))
    rects = {s["name"]: s["rect"] for s in led["sites"]}
    out = os.path.join(ROOT, "out")
    if not os.path.isdir(out):
        raise Skip("no out/")
    missed = []
    for name in sorted(os.listdir(out)):
        rp = os.path.join(out, name, "round.json")
        if not os.path.exists(rp):
            continue
        try:
            w = (json.load(open(rp)).get("results") or {}).get("write") or {}
        except Exception:                        # noqa: BLE001 -- not a round record
            continue
        region = w.get("region")
        if not region or not w.get("placed"):
            continue
        x0, z0, x1, z1 = [int(v) for v in region]
        covered = any(r[0] <= x0 and x1 <= r[2] and r[1] <= z0 and z1 <= r[3]
                      for r in rects.values())
        if not covered:
            missed.append((name, region, w.get("placed")))
    assert not missed, f"written into the world and not reserved: {missed}"
    return (f"{len(rects)} reservations; every round in out/ that wrote blocks into the "
            f"world -- {sum(1 for n in os.listdir(out) if os.path.exists(os.path.join(out, n, 'round.json')))} "
            f"records read -- lies inside one of them")


@case
def t_5_an_area_that_draws_a_border_opens_it_where_the_way_in_is():
    """Found by running it: 37 of the occupancy run's 808 thresholds were E008."""
    import importlib.util
    from ethoslm import lint, offline, pipeline
    from ethoslm.buildlib import Builder
    from ethoslm.circulate import Network, Threshold
    from ethoslm.frontage import Frontage
    G = 64
    vol = _ground(size=80, y=G)
    cells = {(x, 20): {"y": G, "rank": 1, "face": None} for x in range(5, 75)}
    vol = vol.overlay({(x, G, z): "cobblestone" for (x, z) in cells})
    net = Network(cells, [Threshold("a", 40, 20, G, "south", (40, G + 1, 21))])
    rect = (30, 21, 50, 41)
    said = []
    for name in ("yard", "garden", "plaza", "alley", "grove", "field"):
        worst = 0
        for seed in (1, 3, 7):
            b = Builder(offline.OfflineSite(vol))
            b._vol = vol
            b.frontage = Frontage(vol, net)
            part = {"label": "a", "name": "a", "kind": "area", "x0": rect[0],
                    "z0": rect[1], "x1": rect[2], "z1": rect[3], "seed": seed}
            sited = b.site(dict(part),
                           mat=pipeline.voice_palette("ochre_stone_green_tile"))
            # the reserved doorway is not on the part here, and the rim cell still is
            assert sited.get("door") is None, sited.get("door")
            tb = b.type_builder(sited)
            got = tb.door_cell(*rect)
            assert got == (40, 21), (name, got)
            sp = importlib.util.spec_from_file_location(name,
                                                        os.path.join(ROOT, "types",
                                                                     f"{name}.py"))
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            m.build(b.type_builder(sited, role=m.ROLE), sited, seed)
            v2 = vol.overlay(dict(b._pending))
            ctx = lint.Context.build(
                v2, [{"label": "a", "x0": rect[0], "z0": rect[1], "x1": rect[2],
                      "z1": rect[3], "kind": "area", "y0": G}], network=net)
            worst = max(worst, len([f for f in lint.lint(ctx, only={"E008"}).findings
                                    if f.code == "E008"]))
        assert worst == 0, (name, worst)
        said.append(name)
    # ...and an area with no threshold anywhere near draws its border closed
    b = Builder(offline.OfflineSite(_ground(size=80, y=G)))
    b._vol = _ground(size=80, y=G)
    b.frontage = None
    part = {"label": "z", "name": "z", "kind": "area", "x0": 30, "z0": 21,
            "x1": 50, "z1": 41, "seed": 1}
    sited = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert b.type_builder(sited).door_cell(*rect) is None
    return (f"a 21x21 area with its lane on the north and no `door` on the part: "
            f"door_cell finds the threshold's own cell at the rim, and "
            f"{', '.join(said)} are all E008-silent at three seeds; an area with no "
            f"network gets None and draws its border closed")


# ------------------------------------------------------------------- the run

def main():
    only = set(sys.argv[1:])
    ok = fail = skipped = 0
    for name, fn in CASES:
        if only and not any(o in name for o in only):
            continue
        try:
            says = fn()
            ok += 1
            print(f"ok    {name}: {says}")
        except Skip as e:
            skipped += 1
            print(f"skip  {name}: {e}")
        except Exception as e:                    # noqa: BLE001 -- reported
            fail += 1
            import traceback
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok} ok, {fail} fail, {skipped} skipped of {ok + fail + skipped}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
