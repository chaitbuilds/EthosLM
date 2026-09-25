"""The site is chosen for what it is, and the place shapes what it stands on.

    $PY scripts/test_site_needs.py
    $PY scripts/test_site_needs.py 1_        # one phase, or any substring of a name

  1. **The site is chosen for what it is.** A setting may prefer biomes and the search
     scores them as a need; `flat` and `water: none` gate and a search with nothing
     meeting chooses nothing and says so; `steep` is a floor on relief with a preference
     for high ground; a square is assembled from cached squares that tile it; the biome
     share off the chunk NBT matches the census off the parsed field; the locate path
     reads a recorded console answer; the preflight prints the cost and refuses over
     the bound.
  2. **The ground under a place is designed.** A doorstep recorded under a pad is
     followed no lower than the pad's own ground; a concentric place's rings rise
     inward by `TERRACE_STEP` from the site's median to a podium a step above the
     innermost, the layout carries the levels and the walls stand on them at one
     level; a three-ring spec on a relief-40 fixture with a pond stands on three dry
     terraces and every plot on them is a plinth; a wall never deletes itself over a
     hollow.
  3. **Form and silhouette.** A chimney is a voice's word and the place's form decides
     where it says nothing; the great hall in the ochre voice builds no stack and a
     roof under `ROOF_RISE_MAX`; a cottage in a European voice keeps its chimney,
     capped at the eave plus `CHIMNEY_ABOVE_EAVE`; a roof over the cap is eased, one
     under it is untouched.
  4. **The wall's two faces.** The great wall carries `sparse` stairs -- at its corners
     and beside its gates only -- where the spec says unbroken, and `every` as it did
     by default; the wall honours a plain face above `FACE_FROM`; the layout chooses
     both from the spec's words.
  4b. **A ring may be round.** A 45-degree run is sited as a staircase of the wall's
     width and a gate on it is annotated; the great wall builds an octagon clean at
     every face in both voices; the layout draws octagons where the spec's words say
     round, for a type that declares a diagonal run, and squares with the miss named
     for one that does not; a place that is not round is byte-identical.
  5. **The bars tell the truth, and the harness wastes nothing.** E008 is counted
     place-wide in the own-errors bar; the parts and render preflights print the
     cost and refuse on their registered bounds and on a voice that will not
     resolve; a parallel parts run of a small fixture place is byte-identical to the
     sequential one; the district calls are asked for as a batch.
  6. **Cameras.** The flythrough rises to clear every wall it crosses, lifting ahead
     and settling after, and open ground is byte-identical; a gate's eye rung in a
     canopy is passed over for the next; a frame renders only the ground the world
     has, and a camera over unmade ground is dollied onto it.
  7. **The harness for the run.** The footprint ceiling is a number per kind (a city
     at 768) and a recorded spec reads as it was; the block guard scales by area; the
     terraces preflight estimates the fill and refuses over its bound; the ground
     measure reads its five clauses and fails by name.
  8. **The second run's two registered changes.** Where the ground is designed, water
     and the core are fill against `DESIGNED_FILL_MAX`, not needs, and the biome share
     is read over the core and inner rings; a spec hand-back changes only the field
     that was refused.
  9. **What the second run's dry run found, before its write.** The plateau's record
     carries the terrace it was cut at and the layout stands on it (one median, read
     before any ground moved); a designed place's arterials are routed on the designed
     ground, keyed by its terrace; a dry stage that writes the base volume back
     refreshes the backend that cached it; the ground measure reads the inner census
     where the search held the site to it; every gate has a level approach and a ramp
     in the designed ground and the terraces stage lays them; the road is solved with
     the lanes as one surface; a lane arrives at the middle of a face on a tie and
     never at a corner; an area's door is where the circulation reserved it.

Every case is deterministic and makes no model call. Cases that need `out/` skip on a
worktree without it, exactly as `test_place_spec`'s do.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                    # noqa: E402

from ethoslm import groundread, pipeline, spec as spec_mod              # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something `out/` has and this checkout does not."""


def _find_site():
    import importlib.util
    p = os.path.join(ROOT, "scripts", "find_site.py")
    sp = importlib.util.spec_from_file_location("find_site", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _spec(setting=None, sentence="Build a ringed capital on a plain."):
    import test_rings
    doc = json.loads(json.dumps(test_rings.THREE_RING))
    doc["needs"] = {"footprint": 512}       # the fixture's 512 squares; see THREE_RING
    if setting is not None:
        doc["setting"] = setting
    return spec_mod.read_spec(doc, sentence)


def _flat_spec(ringed):
    """The same spec with its rings taken off: a place that designs no ground and is
    held to the ground as found."""
    doc = json.loads(json.dumps(ringed))
    for p in doc["defining_parts"]:
        for k in ("ring", "share", "walled"):
            p.pop(k, None)
    doc = {k: v for k, v in doc.items() if k != "needs"}
    doc["defining_parts"] = [p for p in doc["defining_parts"]
                             if p["family"] not in ("wall", "gate")]
    doc["needs"] = {"footprint": 512}
    return spec_mod.read_spec(doc, "Build a town on a plain.")


def _square(fs, size, *, biome, relief=20, water_pct=0.0, origin=200_000,
            surface="grass_block", mean_y=64):
    """A synthetic candidate whose cells are one biome, with a lake in a corner."""
    import test_place_spec
    f = test_place_spec._settled(fs, size, 96, 2, relief, surface, water_pct=water_pct,
                           origin=origin)
    cw = -(-size // fs.BIOME_CELL)
    pal = [biome]
    codes = np.zeros((cw, cw), np.int32)
    return fs.Field(origin, origin, f.h + int(mean_y), f.wet, f.canopy,
                    gravity=f.gravity, source="fixture", manmade=f.manmade,
                    occupied=f.occupied, surface=f.surface,
                    surface_palette=f.surface_palette, biome=codes, biome_palette=pal)


def _rank(fs, spec, fields, size=512):
    needs = fs.search_needs(spec)
    rows = []
    for f in fields:
        m = fs.measure(f, f.x0, f.z0, size, 40, core=fs.core_size(spec),
                       design=needs.get("designed"))
        rows.append({"measures": m,
                     "excess": fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)})
    rows.sort(key=fs.rank_key)
    return rows


# ------------------------------------------- phase 1: the site is chosen for what it is

@case
def t_1_the_setting_may_prefer_biomes_and_the_words_are_the_censuss():
    s = _spec({"surface": "green", "relief": "flat", "biome": ["plains", "any"]})
    assert s["setting"]["biome"] is None, "a list with `any` in it is no preference"
    s = _spec({"surface": "green", "relief": "flat", "biome": "plains"})
    assert s["setting"]["biome"] == ["plains"], s["setting"]
    s = _spec({"surface": "green", "relief": "flat", "biome": ["savanna", "plains"]})
    assert s["setting"]["biome"] == ["plains", "savanna"], s["setting"]
    again = spec_mod.read_spec(json.loads(json.dumps(s)), s["sentence"])
    assert again["setting"] == s["setting"] and again["needs"] == s["needs"]
    for bad in ("ocean", ["plains", "river"], 7):
        try:
            _spec({"surface": "green", "biome": bad})
            raise AssertionError(f"a biome of {bad!r} was accepted")
        except spec_mod.SpecError as e:
            assert "biome" in str(e), str(e)
    assert set(groundread.BIOMES) >= {"plains", "savanna", "forest", "desert", "snowy",
                                      "any"}, groundread.BIOMES
    assert groundread.biome_class("minecraft:sunflower_plains") == "plains"
    assert groundread.biome_class("river") is None and groundread.biome_class("ocean") is None
    fs = _find_site()
    assert fs.search_needs(s)["biome"] == ["plains", "savanna"]
    assert fs.search_needs(s)["biome_share"] == spec_mod.BIOME_SHARE == 0.5
    brief = pipeline.spec_brief("Build a hold on a crag.", "/tmp/x.json")
    assert "biome" in brief and '"steep"' in brief and "plains" in brief
    return ("biome is a list of the census's words, `any` is no preference, a river is "
            "no biome; BIOME_SHARE 0.5 registered; the brief names the words")


@case
def t_1_a_savanna_site_loses_to_a_plains_one_at_equal_relief():
    fs = _find_site()
    s = _spec({"surface": "green", "relief": "flat", "biome": ["plains"]})
    sav = _square(fs, 512, biome="savanna", origin=200_000)
    pla = _square(fs, 512, biome="plains", origin=210_000)
    rows = _rank(fs, s, [sav, pla])
    top, second = rows[0], rows[1]
    assert top["measures"]["x"] == 210_000, top["measures"]["x"]
    assert top["excess"]["meets"] and top["excess"]["biome"] == 0.0, top["excess"]
    assert not second["excess"]["meets"] and second["excess"]["biome"] == 1.0, second["excess"]
    assert any("biome plains is 0.0%" in w for w in second["excess"]["setting_failures"])
    assert top["measures"]["biome"]["classes"]["plains"] == 100.0
    # ...and with no biome preference the two rank as they did: by coordinates
    plain = _spec({"surface": "green", "relief": "flat"})
    rows = _rank(fs, plain, [sav, pla])
    assert all(r["excess"]["meets"] for r in rows) and rows[0]["measures"]["x"] == 200_000
    assert all(r["excess"]["biome"] == 0.0 for r in rows)
    # a square whose biome was never read is refused by name, as an unread surface is
    blind = fs.Field(sav.x0, sav.z0, sav.h, sav.wet, sav.canopy, gravity=sav.gravity,
                     source="fixture", manmade=sav.manmade, occupied=sav.occupied,
                     surface=sav.surface, surface_palette=sav.surface_palette)
    e = _rank(fs, s, [blind])[0]["excess"]
    assert not e["meets"] and any("biome unread" in w for w in e["setting_failures"]), e
    return ("plains wanted: the plains square meets, the savanna one fails 'biome plains "
            "is 0.0% of the cells against 50%'; nothing wanted: unchanged; unread: refused")


@case
def t_1_a_site_over_its_water_cap_is_refused_for_water_none_and_nothing_is_chosen():
    """Amended under the second run's first registered change (phase 8): the cap is a
    need of a place that does not design its ground. A ringed place's water is fill
    against `DESIGNED_FILL_MAX`, and the same square meets for it with the fill named."""
    fs = _find_site()
    ringed = _spec({"surface": "green", "relief": "flat", "water": "none"})
    s = _flat_spec(ringed)
    assert fs.search_needs(s)["max_water_pct"] == spec_mod.WATER_NONE_PCT == 2.0
    assert "designed" not in fs.search_needs(s)
    wet = _square(fs, 512, biome="plains", water_pct=25.7)
    e = _rank(fs, s, [wet])[0]["excess"]
    assert not e["meets"] and e["water"] > 0, e
    assert any(w.startswith("water over by") for w in fs.failed_needs(e)), fs.failed_needs(e)
    row = _rank(fs, ringed, [wet])[0]
    assert row["excess"]["meets"] and row["excess"]["water"] == 0, row["excess"]
    assert row["measures"]["fill"]["blocks"] > 0
    # ...and the fresh-search path, over a cache holding only that square, chooses
    # nothing, writes the record, and says which need the best square failed
    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(fs.tile_cache(wet.x0, wet.z0, 512, 512, tmp),
                            x0=wet.x0, z0=wet.z0, h=wet.h, wet=wet.wet,
                            canopy=wet.canopy, manmade=wet.manmade,
                            occupied=wet.occupied, surface_codes=wet.surface,
                            surface_palette=np.array(wet.surface_palette, dtype=str),
                            biome_codes=wet.biome,
                            biome_palette=np.array(wet.biome_palette, dtype=str))

        class NoRegions:
            directory = os.path.join(tmp, "no-regions")
        lines = []
        got = fs.search_fresh(s, editor=None, radii=(300_000,), stride=100_000,
                              directory=tmp, regions=NoRegions(), log=lines.append)
    assert got["chosen"] is None, got["chosen"]
    assert got["attempt"] == "nothing met the needs", got["attempt"]
    assert got["candidates"] == 1 and got["meeting"] == 0, (got["candidates"], got["meeting"])
    assert got["best_failed"]["origin"] == [wet.x0, wet.z0], got["best_failed"]
    assert any(f.startswith("water over by") for f in got["best_failed"]["failures"])
    assert any("no site is chosen" in ln for ln in lines), lines
    assert got["squares"]["unread"] >= 1 and got["preflight"], got["squares"]
    return (f"25.7% water against a cap of 2: refused; the search over one square chooses "
            f"nothing, records the best as failing {got['best_failed']['failures'][0]!r}, "
            f"and says so")


@case
def t_1_steep_is_a_floor_on_relief_with_a_preference_for_high_ground():
    fs = _find_site()
    assert spec_mod.RELIEF_WORDS == {"flat": 0.25, "rolling": 0.55, "steep": None,
                                     "any": None}, spec_mod.RELIEF_WORDS
    assert spec_mod.RELIEF_FLOORS == {"steep": 0.55} and spec_mod.STEEP_CAP == 1.5
    # a hold on a crag designs no rings: held to the ground as found (phase 8 amended
    # this case to say so; a ringed place ranks by its fill as well)
    s = _flat_spec(_spec({"surface": "stone", "relief": "steep"}, "Build a hold on a crag."))
    assert s["needs"]["min_relief"] == round(0.55 * 512, 1) == 281.6, s["needs"]
    assert s["needs"]["max_relief"] == 1.5 * 512, s["needs"]
    needs = fs.search_needs(s)
    assert needs["min_relief"] == 281.6 and needs["prefer_high"] is True
    flat = _square(fs, 512, biome="mountain", relief=20, surface="stone",
                   origin=200_000, mean_y=64)
    low = _square(fs, 512, biome="mountain", relief=300, surface="stone",
                  origin=210_000, mean_y=64)
    high = _square(fs, 512, biome="mountain", relief=300, surface="stone",
                   origin=220_000, mean_y=120)
    rows = _rank(fs, s, [flat, low, high])
    assert rows[0]["measures"]["x"] == 220_000, [r["measures"]["x"] for r in rows]
    assert rows[1]["measures"]["x"] == 210_000
    assert rows[0]["excess"]["meets"] and rows[1]["excess"]["meets"]
    assert not rows[2]["excess"]["meets"] and rows[2]["excess"]["relief_min"] > 0, rows[2]["excess"]
    assert rows[0]["excess"]["height_preference"] < rows[1]["excess"]["height_preference"]
    # a flat word still refuses the steep ground, as it did
    f = _flat_spec(_spec({"surface": "stone", "relief": "flat"}))
    assert "min_relief" not in f["needs"]
    e = _rank(fs, f, [low])[0]["excess"]
    assert e["relief"] > 0 and e["height_preference"] == 0.0, e
    return ("steep: min_relief 281.6 over 512, cap 768; relief 20 fails the floor, 300 "
            "meets, and of two that meet the higher (mean y 120 against 64) ranks first; "
            "flat still refuses 300")


@case
def t_1_a_square_is_assembled_from_the_cached_squares_that_tile_it():
    fs = _find_site()
    with tempfile.TemporaryDirectory() as tmp:
        subs = [(0, 0), (256, 0), (0, 256), (256, 256)]
        big = np.arange(768 * 768, dtype=np.int32).reshape(768, 768) % 50 + 60
        for (sx, sz) in subs:
            h = big[sx:sx + 512, sz:sz + 512]
            np.savez_compressed(fs.tile_cache(sx, sz, 512, 512, tmp), x0=sx, z0=sz,
                                h=h, wet=h > 100, canopy=h < 62,
                                manmade=np.zeros(h.shape, np.int64),
                                occupied=np.ones(h.shape, np.int64),
                                # the same names per column, in a palette whose order
                                # differs between the squares
                                surface_codes=((h % 2) if sx else (1 - h % 2)).astype(np.int32),
                                surface_palette=np.array(["grass_block", "stone"] if sx
                                                         else ["stone", "grass_block"],
                                                         dtype=str),
                                biome_codes=np.zeros((128, 128), np.int32),
                                biome_palette=np.array(["plains" if sz else "forest"],
                                                       dtype=str))
        assert fs.composable(0, 0, 768, tmp) == [(0, 0, 512), (0, 256, 512),
                                                 (256, 0, 512), (256, 256, 512)]
        assert fs.composable(256, 0, 768, tmp) is None
        lines = []
        f = fs.compose_from_cache(0, 0, 768, tmp, log=lines.append)
        assert f is not None and f.shape == (768, 768) and np.array_equal(f.h, big)
        assert f.biome.shape == (192, 192)
        names = np.array(f.surface_palette)[f.surface]
        assert (names == np.where(big % 2, "stone", "grass_block")).all(), "palettes remapped"
        bnames = np.array(f.biome_palette)[f.biome]
        assert (bnames[:, :64] == "forest").all() and (bnames[:, 64:] == "plains").all()
        assert fs._cached(0, 0, 768, tmp), "the assembled square is cached at its size"
        # a sub-square that disagrees with its neighbour on the overlap refuses it
        p = fs.tile_cache(256, 256, 512, 512, tmp)
        got = dict(np.load(p))
        got["h"] = got["h"] + 1
        np.savez_compressed(p, **got)
        os.remove(fs.tile_cache(0, 0, 768, 768, tmp))
        lines = []
        assert fs.compose_from_cache(0, 0, 768, tmp, log=lines.append) is None
        assert any("disagrees" in ln for ln in lines), lines
    return ("four cached 512s tile a 768 on the 256 lattice: assembled, palettes and "
            "biomes remapped, cached; a disagreeing overlap refuses the assembly")


@case
def t_1_biome_shares_off_the_chunk_nbt_match_the_parsed_field():
    fs = _find_site()
    from ethoslm import regions as _regions
    x, z = -512, 2816
    p = fs.tile_cache(x, z, 512, 512)
    if not os.path.exists(p) or not os.path.isdir(_regions.REGIONS):
        raise Skip("no cached (-512,2816) 512x512 or no region files")
    got = np.load(p)
    if "biome_codes" not in got:
        raise Skip("the cached square carries no biomes")
    f = fs._field_from_cache(got, x, z, 512, 512, p)
    m = fs.measure(f, x, z, 512, 48, core=48)
    regs = _regions.Regions()
    try:
        b = fs.biome_share_from_regions(regs, x, z, 512, ["plains"])
    finally:
        regs.close()
    assert b["read"], b
    for k, v in m["biome"]["classes"].items():
        assert abs(b["classes"][k] - v) <= 0.5, (k, b["classes"][k], v)
    assert b["share"] == groundread.biome_share(m["biome"], ["plains"])
    assert b["seconds"] < 10.0, b["seconds"]
    return (f"({x},{z}): {b['top'][:3]} off the chunk NBT in {b['seconds']}s, within "
            f"0.5% of the parsed field's census on every class")


@case
def t_1_the_locate_path_reads_a_recorded_console_answer():
    fs = _find_site()
    recorded = {
        "execute positioned 0 64 0 run locate biome minecraft:plains":
            "[12:00:01] [Server thread/INFO]: The nearest minecraft:plains is at "
            "[1216, ~, -2048] (2381 blocks away)",
        "execute positioned 0 64 0 run locate biome minecraft:sunflower_plains":
            "[12:00:02] [Server thread/INFO]: Could not find a biome of type "
            "minecraft:sunflower_plains within reasonable distance",
        "execute positioned 4096 64 0 run locate biome minecraft:plains":
            "[12:00:03] [Server thread/INFO]: The nearest minecraft:plains is at "
            "[1216, ~, -2048] (3521 blocks away)",
        "execute positioned 4096 64 0 run locate biome minecraft:meadow":
            "[12:00:04] [Server thread/INFO]: The nearest minecraft:meadow is at "
            "[5000, ~, 300] (950 blocks away)",
    }
    asked = []

    def ask(cmd, pattern):
        asked.append(cmd)
        return recorded.get(cmd)
    pts = fs.locate_biomes(ask, [(0, 0), (4096, 0)], ["plains"], log=lambda *a: None)
    assert fs.parse_locate(recorded[asked[0]]) == ("plains", 1216, -2048, 2381)
    assert pts == [{"biome": "meadow", "x": 5000, "z": 300, "from": [4096, 0],
                    "distance": 950},
                   {"biome": "plains", "x": 1216, "z": -2048, "from": [0, 0],
                    "distance": 2381}], pts
    assert len(asked) == 2 * len(fs.LOCATE_IDS["plains"]), asked
    return (f"{len(asked)} locate commands from two origins; two points, the nearer "
            f"first, one answer deduplicated, one 'could not find' dropped")


@case
def t_1_the_preflight_prints_the_cost_and_refuses_over_the_bound():
    fs = _find_site()
    lines = []
    ok = fs.preflight_search({"cached": 700, "composed": 10, "biome_reads": 40,
                              "files": 30, "generate": 4}, 768, log=lines.append)
    # COST_GENERATE_S was 240 from the concentric log and is 30 from the ground run's
    # own first live session (six 768 squares at 19-51 s each); the case reads it
    est = ((30 * 15.0 + 4 * fs.COST_GENERATE_S + 40 * 3.0) * (768 / 512) ** 2)
    assert fs.COST_GENERATE_S == 30.0
    assert ok["estimated_seconds"] == round(est) and not ok["refused"], ok
    assert ok["bound_seconds"] == fs.SEARCH_BOUND_S == 7200.0
    assert "preflight" in lines[0] and "REFUSED" not in lines[0], lines
    bad = fs.preflight_search({"files": 400, "generate": 48}, 512, log=lines.append)
    assert bad["refused"] and bad["estimated_seconds"] > 7200, bad
    assert "REFUSED" in lines[-1], lines[-1]
    tight = fs.preflight_search({"files": 10}, 512, bound=60.0, log=lambda *a: None)
    assert tight["refused"], tight
    return (f"30 file reads, 4 generated and 40 biome reads at 768: {ok['estimated_seconds']}s "
            f"against 7200, allowed; 400 reads and 48 generated at 512 refused; a bound "
            f"of 60s refuses ten reads")


# ------------------------------------------- phase 2: the ground under a place is
# designed

def _volume(X, Z, S, *, base=60, relief=40, margin=16, pond=None, y0=30, sy=90):
    """A site of `S` on a slope of `relief` from west to east, on stone under grass,
    with `pond` = (x0, z0, x1, z1, surface_y) a lake sunk into it, as a Volume."""
    from ethoslm.observe import Volume
    sx = sz = S + 2 * margin
    codes = np.zeros((sx, sy, sz), np.uint16)
    pal = ["air", "stone", "grass_block", "water", "dirt"]
    xs = np.arange(sx)
    g = base + ((xs.astype(float) / max(1, sx - 1)) * relief).astype(int)
    for i in range(sx):
        codes[i, :g[i] - y0, :] = 1
        codes[i, g[i] - y0, :] = 2
    if pond:
        px0, pz0, px1, pz1, wy = pond
        i0, i1 = px0 - (X - margin), px1 - (X - margin) + 1
        j0, j1 = pz0 - (Z - margin), pz1 - (Z - margin) + 1
        codes[i0:i1, :, j0:j1] = 0
        codes[i0:i1, :wy - 6 - y0, j0:j1] = 1
        codes[i0:i1, wy - 6 - y0, j0:j1] = 4
        codes[i0:i1, wy - 5 - y0:wy + 1 - y0, j0:j1] = 3
    return Volume(X - margin, y0, Z - margin, codes, pal)


@case
def t_2_a_doorstep_below_the_pad_is_followed_no_lower_than_the_pads_ground():
    from ethoslm import offline
    from ethoslm.buildlib import Builder
    from ethoslm.circulate import Network, Threshold
    from ethoslm.frontage import Frontage
    X, Z, S = 1000, 2000, 64
    vol = _volume(X, Z, S, base=56, relief=6, margin=8)          # ground 56..62
    # the circulation pass's record says the doorstep is at y=38, eighteen blocks under
    # the pad -- the concentric run's `upper_ring_south_west_garden_estate`
    cells = {(x, Z + 40): {"y": 38, "rank": 0, "face": None} for x in range(X + 10, X + 50)}
    net = Network(cells, [Threshold("court", X + 30, Z + 40, 38, "north",
                                    (X + 30, 39, Z + 38))])
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net)
    part = {"label": "court", "kind": "plot", "x0": X + 20, "z0": Z + 20,
            "x1": X + 40, "z1": Z + 38}
    grade_before = {(x, z): b.bed(x, z) for x in range(X + 8, X + 56) for z in range(Z + 8, Z + 52)}
    got = b.site(part, mat={"wall": "sandstone", "footing": "granite", "frame": "dark_oak",
                            "roof": "oxidized_copper", "trim": "red_sandstone",
                            "floor": "smooth_stone"})
    lo, hi = got["sited"]["grade"]
    assert got["floor_y"] >= lo, (got["floor_y"], got["sited"]["grade"])
    assert got["floor_y"] == lo == 56 or got["floor_y"] >= 56, got["sited"]["reason"]
    assert got["ground"] in ("platform", "plinth"), got["ground"]
    # nothing was placed below the pad's own ground, and the neighbour's ground --
    # columns three or more outside the pad and its way in -- is untouched
    x0, z0, x1, z1 = got["footprint"]
    way = {(c[0], c[2]) for c in got["way"]}
    below = [p for p in b._pending if p[1] < lo - 1]
    assert not below, f"{len(below)} blocks below the pad's lowest ground"
    for (x, z), g in grade_before.items():
        if x0 - 3 <= x <= x1 + 3 and z0 - 3 <= z <= z1 + 3:
            continue
        if any(abs(x - wx) <= 1 and abs(z - wz) <= 1 for (wx, wz) in way):
            continue
        assert not any(p[0] == x and p[2] == z for p in b._pending), (x, z)
    return (f"a doorstep recorded at y=38 under a pad on ground {lo}..{hi}: the floor "
            f"is {got['floor_y']}, nothing is laid below {lo}, the neighbours are untouched")


@case
def t_2_terrace_levels_rise_inward_and_the_layout_carries_them():
    from ethoslm import placeplan
    import test_rings
    assert placeplan.TERRACE_STEP == 4
    lv = placeplan.terrace_levels(64, 3)
    assert lv == {"rings": [72, 68, 64], "podium": 76, "median": 64, "step": 4}, lv
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z = site["origin"]
    vol = _volume(X, Z, 512, base=50, relief=40)
    med = placeplan.site_median(vol, site)
    assert 68 <= med <= 72, med
    _t, decls = placeplan.types_card(None, s.get("form"))
    plateau = test_rings._plateau(site)
    place, fails = placeplan.concentric_layout(s, site, plateau, decls,
                                               "ochre_stone_green_tile", vol=vol)
    assert not fails, fails
    lay = place["layout"]
    assert lay["terrace"] == {"rings": [med + 8, med + 4, med], "podium": med + 12,
                              "median": med, "step": 4}, lay["terrace"]
    assert [r["level"] for r in lay["rings"]] == [med + 8, med + 4, med]
    walls = [p for p in place["parts"] if p["kind"] == "edge"]
    assert {p["ring"]: p["level"] for p in walls} == {1: med + 4, 2: med}, walls
    # a plateau record that carries the levels wins over the ground
    plateau2 = {**plateau, "terrace": {"rings": [90, 86, 82], "podium": 94,
                                       "median": 82, "step": 4}}
    place2, _f = placeplan.concentric_layout(s, site, plateau2, decls,
                                             "ochre_stone_green_tile", vol=vol)
    assert [r["level"] for r in place2["layout"]["rings"]] == [90, 86, 82]
    # ...and with no ground and no record there are no levels and no wall carries one
    place3, _f = placeplan.concentric_layout(s, site, plateau, decls,
                                             "ochre_stone_green_tile")
    assert place3["layout"]["terrace"] is None
    assert all("level" not in p for p in place3["parts"] if p["kind"] == "edge")
    assert lay["registered"]["TERRACE_STEP"] == 4
    return (f"median {med} on a 40-relief slope: rings at {med + 8}/{med + 4}/{med}, the "
            f"podium {med + 12}; the walls carry their ring's level; a plateau record's "
            f"levels win; no ground, no levels")


#: How far a terrace's own works reach into the ground it refused: the retaining face
#: one column in, and the feathered slope a neighbouring piece dresses beyond it.
SEAM_REACH = 3


@case
def t_2_a_three_ring_spec_on_a_relief_40_fixture_stands_on_three_terraces():
    from ethoslm import offline, observe, placeplan, pipeline
    from ethoslm.buildlib import Builder
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40,
                      pond=(X + 300, Z + 120, X + 340, Z + 160, 68))
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        json.dump(json.loads(json.dumps(test_rings.THREE_RING)),
                  open(os.path.join(tmp, "place.json"), "w"))
        json.dump({**site, "mean_grid": [[70] * 12] * 12, "roughness_grid": [[3] * 12] * 12,
                   "stats": {"min": 50, "max": 90, "relief": 40}},
                  open(os.path.join(tmp, "site.json"), "w"))
        json.dump({"chosen": None, "terraform": None, "attempt": "as asked"},
                  open(os.path.join(tmp, "site_search.json"), "w"))
        rnd = pipeline.Round(name="terrace_fixture", sentence="Build a ringed town.",
                             state_dir=tmp, voice="ochre_stone_green_tile",
                             flags={"dry_run": True})
        be = pipeline.OfflineBackend(rnd, dry_run=True)
        # the podium: cut for a concentric place whether or not the search marked it, at
        # a step above the innermost ring
        med = placeplan.site_median(rnd.volume(), site)
        pl = stages_plan.stage_plateau(rnd, be, {})
        assert pl.get("plateau", {}).get("ok"), pl
        assert pl["terrace"]["podium"] == med + 12 == pl["plateau"]["y"], pl["terrace"]
        assert os.path.exists(os.path.join(tmp, "world.before-plateau.npz"))
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(s, site, json.load(open(
            os.path.join(tmp, "plateau.json"))), decls, "ochre_stone_green_tile",
            vol=rnd.volume())
        assert not fails, fails
        json.dump(place, open(os.path.join(tmp, "plan.json"), "w"))
        got = stages_plan.stage_terraces(rnd, be, {})
        assert "rings" in got and got.get("placed"), got
        assert [r["level"] for r in got["rings"]] == [med, med + 4, med + 8], got["rings"]
        assert all(r["terrace"]["ok"] for r in got["rings"])
        est = [r["terrace"]["estimated_blocks"] for r in got["rings"]]
        assert all(e <= Builder.TERRACE_MAX_BLOCKS for e in est), est
        assert got["registered"] == {"TERRACE_STEP": 4, "TERRACE_MAX_BLOCKS": 1_500_000,
                                     "TERRACES_BOUND_BLOCKS": 12_000_000}, got["registered"]
        assert got["preflight"]["estimated_blocks"] > 0 and not got["preflight"]["refused"]
        again = stages_plan.stage_terraces(rnd, be, {})
        assert "skipped" in again
        # the ground: every ring at its design's own decision, no water where it was
        # prepared, the podium above the upper ring
        v2 = rnd.volume()
        h, wet = observe.ground_heights(v2)
        base_vol = offline.load_volume(os.path.join(tmp, "world.before-plateau.npz"))
        base_h, _base_wet = observe.ground_heights(base_vol)
        cx, cz = place["layout"]["centre"]
        lay = place["layout"]
        levels = {}
        gates = [p for p in place["parts"] if p["kind"] == "point"]
        points = placeplan.approach_points(lay, gates)
        assert len(points) == len(gates) + 1 and points[-1]["virtual"]   # the unwalled ring's ramp
        assert got["approaches"] and len(got["approaches"]) == len(points)
        def on_approach(x, z):
            for ap in got["approaches"]:
                for piece in ap["pieces"]:
                    r = piece["rect"]
                    if r[0] - 1 <= x <= r[2] + 1 and r[1] - 1 <= z <= r[3] + 1:
                        return True
            return False
        for r in lay["rings"]:
            # clear of the feathers -- **the library's own maximum**, not 12. The
            # neighbourhood delivery round: a fall of four between two rings feathers
            # further than twelve columns on a relief-40 fixture, so the sample of
            # `outer_town` took ground the ring inside it had dressed and read 75 where
            # its own level is 73. The margin is `Builder.PLATEAU_FEATHER_MAX`, which is
            # the widest slope any piece of this library dresses.
            a = r["inner"] + max(12, int(Builder.PLATEAU_FEATHER_MAX) + 2)
            b = r["outer"] - 4
            xs = np.arange(cx - b, cx + b + 1) - v2.x0
            zs = np.arange(cz - b, cz + b + 1) - v2.z0
            cheb = np.maximum(np.abs(np.arange(-b, b + 1))[:, None],
                              np.abs(np.arange(-b, b + 1))[None, :])
            ring = (cheb > a) & (cheb <= b)
            m = 2 * placeplan.TERRACE_STEP + 4           # ...and of the gates' approaches,
            for ap in got["approaches"]:                 # with their lateral feathers
                for piece in ap["pieces"]:
                    q = piece["rect"]
                    ring[max(0, q[0] - m - (cx - b)):q[2] + m + 1 - (cx - b),
                         max(0, q[1] - m - (cz - b)):q[3] + m + 1 - (cz - b)] = False
            # **Re-registered by the neighbourhood delivery round**, and the rule under
            # it changed: `terrace_annulus` no longer levels every column of the
            # rectangle it is given, because a mask that excludes housing from a
            # hillside on the ground that it must not be cut, beside a builder that cuts
            # it anyway, is two rules about one question. On this `relief=40` fixture
            # thirteen to nineteen blocks of cut stand over the belt's east strip and
            # `feasible.terrain` at the mask's own bound refuses all 27,920 of its
            # columns, so the ring is deliberately **not** one level. The question is
            # kept and made two-sided, which is more than it asked before: every column
            # the design **moved** reads the ring's level and is dry, and every column
            # it **kept** reads the bed it was found at. A terrace that levelled nothing
            # passes neither half.
            hh = h[np.ix_(xs, zs)][ring]
            ww = wet[np.ix_(xs, zs)][ring]
            hb = base_h[np.ix_(xs, zs)][ring]
            from ethoslm import feasible as _feas
            reach = int(placeplan.DISTRICT_TERRACE_REACH)
            dec = _feas.record(base_vol, (cx - b, cz - b, cx + b, cz + b),
                               level=r["level"], relief=reach, fill=reach, window=0)
            moved_mask = _feas.mask_of(dec)[ring]
            n_moved, n_kept = int(moved_mask.sum()), int((~moved_mask).sum())
            levels[r["name"]] = (int(hh.min()), int(hh.max()), int(ww.sum()),
                                 n_moved, n_kept)
            if n_kept:
                # ...and a kept column **at the seam** is the one exception, by design:
                # where prepared ground stands over kept ground the drop is carried as a
                # retaining face one column into the kept ground, exactly as the rim is
                # faced, so a person meets a wall's top and not a hole. Those columns
                # are excluded and counted; every other kept column is the ground as
                # found.
                import numpy as _np
                mm = _np.zeros_like(moved_mask)
                grid = _np.zeros(ring.shape, dtype=bool)
                grid[ring] = moved_mask
                near = grid.copy()
                for _ in range(SEAM_REACH):
                    near[1:, :] |= grid[:-1, :]
                    near[:-1, :] |= grid[1:, :]
                    near[:, 1:] |= grid[:, :-1]
                    near[:, :-1] |= grid[:, 1:]
                    grid = near.copy()
                mm = near[ring]
                # ...and the podium's own feathered slope, which is the plateau's work
                # and not this ring's: `inner_town` abuts the compound rectangle and the
                # feather is dressed outward from it.
                px0f, pz0f, px1f, pz1f = lay["compound_rect"]
                fmax = int(Builder.PLATEAU_FEATHER_MAX)
                gx = _np.arange(cx - b, cx + b + 1)[:, None] * _np.ones(
                    (1, 2 * b + 1), dtype=int)
                gz = _np.ones((2 * b + 1, 1), dtype=int) * _np.arange(
                    cz - b, cz + b + 1)[None, :]
                pod_near = ((gx >= px0f - fmax) & (gx <= px1f + fmax)
                            & (gz >= pz0f - fmax) & (gz <= pz1f + fmax))[ring]
                # the same exclusion, both ways: a seam is where two decisions meet and
                # neither side of it is a statement about one decision alone
                grid2 = _np.zeros(ring.shape, dtype=bool)
                grid2[ring] = ~moved_mask
                near2 = grid2.copy()
                for _ in range(SEAM_REACH):
                    near2[1:, :] |= grid2[:-1, :]
                    near2[:-1, :] |= grid2[1:, :]
                    near2[:, 1:] |= grid2[:, :-1]
                    near2[:, :-1] |= grid2[:, 1:]
                    grid2 = near2.copy()
                near_kept = near2[ring]
                far = (~moved_mask) & (~mm) & (~pod_near)
                clean = moved_mask & (~near_kept) & (~pod_near)
                levels[r["name"]] = levels[r["name"]] + (int(clean.sum()),
                                                        int(far.sum()))
                if clean.any():
                    assert hh[clean].min() == hh[clean].max() == r["level"], (
                        r["name"], "a column the design moved does not read its level",
                        int(hh[clean].min()), int(hh[clean].max()), r["level"])
                    assert ww[clean].sum() == 0, (r["name"], int(ww[clean].sum()))
                if far.any():
                    assert (hh[far] == hb[far]).all(), (
                        r["name"], "a column the design refused, away from any seam, "
                        "was moved anyway",
                        int((hh[far] != hb[far]).sum()), int(far.sum()))
        px0, pz0, px1, pz1 = lay["compound_rect"]
        pod = h[px0 - v2.x0:px1 - v2.x0 + 1, pz0 - v2.z0:pz1 - v2.z0 + 1]
        assert pod.min() == pod.max() == med + 12, (int(pod.min()), int(pod.max()))
        assert med + 12 > max(r["level"] for r in lay["rings"])
        # every gate's approach as ground: level for the wall's reach and the run, then
        # a block down every two columns to the ring beyond
        for ap in got["approaches"]:
            assert ap["ok"] and ap["placed"] > 0, ap
            gx, gz = ap["at"]
            ox, oz = ap["outward"]
            prof = [int(h[gx + ox * k - v2.x0, gz + oz * k - v2.z0]) for k in range(0, 30)
                    if 0 <= gx + ox * k - v2.x0 < h.shape[0] and 0 <= gz + oz * k - v2.z0 < h.shape[1]]
            assert prof[:15] == [ap["inside"]] * 15, (ap["gate"], prof)
            if ap["foot"] is not None and ap["foot"] < ap["inside"]:
                n_ramp = 2 * (ap["inside"] - ap["foot"] - 1)
                want = [ap["inside"] - 1 - (k // 2) for k in range(n_ramp)]
                assert prof[15:15 + n_ramp] == want, (ap["gate"], prof[15:15 + n_ramp], want)
                # past the last piece its dressed edge, then the ring: never a drop over
                # one
                tail = prof[15 + n_ramp:]
                assert all(a - b <= 1 for a, b in zip(tail, tail[1:])), (ap["gate"], tail)
                assert tail[-1] == ap["foot"], (ap["gate"], tail)
        # every wall at its ring's level, on one level; every plot a plinth, no deck
        b = Builder(offline.OfflineSite(v2))
        b._vol = v2
        mat = pipeline.voice_palette("ochre_stone_green_tile")
        walls = {}
        for p in place["parts"]:
            if p["kind"] != "edge":
                continue
            sited = b.site(dict(p, label=p["name"]), mat=mat)
            floors = {seg["floor_y"] for seg in sited["segments"]}
            assert floors == {p["level"]}, (p["name"], floors, p["level"])
            walls[p["name"]] = p["level"]
        grounds = {}
        for d in place["districts"][:6]:
            x0, z0 = d["x0"] + 4, d["z0"] + 4
            sited = b.site({"label": d["name"], "kind": "plot", "x0": x0, "z0": z0,
                            "x1": x0 + 12, "z1": z0 + 12}, mat=mat)
            grounds[d["name"]] = sited["ground"]
        assert set(grounds.values()) == {"plinth"}, grounds
        return (f"podium {med + 12} over rings at {med + 8}/{med + 4}/{med} on a 40-relief "
                f"slope with a pond: every column each ring's design **moved** reads its "
                f"level and is dry, and every column it **refused** at a reach of "
                f"{int(placeplan.DISTRICT_TERRACE_REACH)} reads the bed it was found at "
                f"-- per ring (min, max, wet, moved, kept, moved clear of a seam, kept "
                f"clear of one) {levels}, {got['placed']} blocks in "
                f"{got['seconds']}s; {len(got['approaches'])} gate approaches level then "
                f"ramped; walls {walls}; six plots all plinths")


@case
def t_2_a_wall_over_a_hollow_never_deletes_itself():
    from ethoslm import offline, pipeline
    from ethoslm.buildlib import Builder
    import importlib.util
    X, Z, S = 3000, 4000, 96
    vol = _volume(X, Z, S, base=60, relief=0, margin=8,
                  pond=(X + 40, Z + 30, X + 56, Z + 50, 54))     # a hollow 12 deep
    v = vol.codes
    # the hollow is dry: a pit, not a pond
    v[v == 3] = 0
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    mat = pipeline.voice_palette("ochre_stone_green_tile")
    part = {"label": "ring", "kind": "edge", "path": [[X + 20, Z + 40], [X + 76, Z + 40]],
            "width": 3}
    sited = b.site(dict(part), mat=mat)
    assert sited["sited"]["ok"], sited["sited"]
    sp = importlib.util.spec_from_file_location("wall_t", os.path.join(ROOT, "types", "wall.py"))
    wall = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(wall)
    tb = b.type_builder(sited)
    wall.build(tb, sited, 1, height=8, width=3)
    pend = b._pending
    gaps = []
    for x in range(X + 22, X + 75):
        col = [p for p in pend if p[0] == x and p[2] == Z + 40 and p[1] > sited["floor_y"] + 2]
        if not col:
            gaps.append(x)
    assert not gaps, f"the wall omits {len(gaps)} column(s): {gaps[:6]}"
    over = [p for p in pend if X + 40 <= p[0] <= X + 56 and p[2] == Z + 40
            and p[1] == sited["floor_y"] + 8]
    assert over, "no crown over the hollow"
    return (f"a wall of 57 columns over a 12-deep hollow: every column stands, the "
            f"crown runs over the hollow at y={sited['floor_y'] + 8}")


# ----------------------------------------------------- phase 3: form and silhouette

def _stand(name, voice, size, *, chimney=None, seed=1, params=None):
    """One committed type on flat ground in a voice, the voice's chimney word set as
    the driver would set it. `test_types._stand`'s fixture."""
    import collections
    import test_types
    from ethoslm import offline
    from ethoslm.buildlib import Builder
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
    mat = pipeline.voice_palette(voice)
    roof = pipeline.voice_roof(voice) or {}
    if chimney is not None:
        roof["chimney"] = chimney
    w = size + 4
    part = {"label": "t", "kind": "plot", "x0": 20, "z0": 20, "x1": 19 + w, "z1": 19 + w}
    x0, z0, x1, z1 = pipeline.part_rect(part)
    vol = test_types._flat_world(size + 45)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = None
    b.registry = test_types._OnePlot({"label": "t", "x0": x0, "z0": z0, "x1": x1, "z1": z1})
    sited = b.site(dict(part), mat=mat, roof=roof)
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)                  # noqa: S102
    tb = b.type_builder(sited, role=ns.get("ROLE"))
    res = ns["build"](tb, sited, seed, **(params or {}))
    b.resolve_steps()
    return {"ok": bool(res and res.get("ok", True)), "floor_y": int(sited["floor_y"]),
            "pending": dict(b._pending), "mat": mat, "voice": mat,
            "blocks": collections.Counter(v.split("[")[0] for v in b._pending.values())}


@case
def t_3_a_chimney_is_a_voices_word_and_the_form_decides_where_it_says_nothing():
    from ethoslm import voices
    assert voices.chimney_default("east_asian") is False
    assert voices.chimney_default("european_vernacular") is True
    assert voices.chimney_default(None) is True and voices.chimney_default("civic") is True
    v = voices.load("ochre_stone_green_tile")
    assert v["chimney"] is None, "a voice that says nothing carries None"
    doc = json.load(open(os.path.join(ROOT, "voices", "ochre_stone_green_tile.json")))
    no = voices.validate(dict(doc, chimney=False))
    assert no["chimney"] is False
    assert voices.validate(no) == no, "the validator reads what it writes"
    try:
        voices.validate(dict(doc, chimney="yes"))
        raise AssertionError("'yes' was accepted")
    except voices.VoiceError as e:
        assert "chimney" in str(e)
    assert pipeline.voice_roof("ochre_stone_green_tile")["chimney"] is None
    from ethoslm.buildlib import Builder
    assert Builder.CHIMNEY_ABOVE_EAVE == 4 and Builder.ROOF_RISE_MAX == 1.5
    return ("chimney: true/false/absent; absent is the form's -- none for east_asian, "
            "one for european_vernacular and for no form; CHIMNEY_ABOVE_EAVE 4 and "
            "ROOF_RISE_MAX 1.5 registered")


@case
def t_3_the_great_hall_in_the_ochre_voice_builds_no_stack_and_a_roof_under_the_cap():
    from ethoslm.buildlib import Builder
    got = _stand("hall", "ochre_stone_green_tile", 32, chimney=False)
    assert got["ok"], got["blocks"]
    fy = got["floor_y"]
    eave = fy + 2 * Builder.STOREY
    footing = got["mat"]["footing"]
    stack = [p for p, v in got["pending"].items()
             if v.split("[")[0] == footing and p[1] > eave + 1]
    assert not stack, f"{len(stack)} footing blocks above the eave: a stack"
    ridge = max(p[1] for p in got["pending"])
    cap = int(Builder.ROOF_RISE_MAX * (eave - fy))
    assert ridge - eave <= cap + 1, (ridge - eave, cap)
    # ...and the same hall with the word withheld (a stored program's terms) carries its
    # stacks to the ridge, which is what the concentric run rendered
    was = _stand("hall", "ochre_stone_green_tile", 32, chimney=None)
    tall = [p for p, v in was["pending"].items()
            if v.split("[")[0] == footing and p[1] > eave + 1]
    assert tall, "with no word the type's stack still stands (capped)"
    top_was = max(p[1] for p in tall)
    assert top_was <= eave + Builder.CHIMNEY_ABOVE_EAVE + 1, (top_was, eave)
    # the civic silhouette uncapped rises past the cap on a 32-plot; capped, it eases
    from ethoslm.prims import Primitives
    prof = Primitives._roof_profile(Builder.__new__(Builder), 16, "gable", 1, 1,
                                    [[1, 2], [2, 1]], "upturned")
    capped = Primitives._roof_profile(Builder.__new__(Builder), 16, "gable", 1, 1,
                                      [[1, 2], [2, 1]], "upturned", rise_max=12)
    assert max(h for h, _k in prof) > 12 >= max(h for h, _k in capped), (
        max(h for h, _k in prof), max(h for h, _k in capped))
    return (f"no footing block above the eave; ridge {ridge - eave} over walls of "
            f"{eave - fy} against a cap of {cap}; the civic profile on a half-span of 16 "
            f"rises {max(h for h, _k in prof)} uncapped and {max(h for h, _k in capped)} "
            f"under 12; with no word the stack stops at eave + {top_was - eave}")


@case
def t_3_a_cottage_in_a_european_voice_keeps_its_chimney_capped_at_the_eave():
    from ethoslm.buildlib import Builder
    from ethoslm import voices
    voice = "white_render_dark_frame"
    got = _stand("cottage", voice, 9, chimney=voices.chimney_default("european_vernacular"))
    assert got["ok"], got["blocks"]
    fy = got["floor_y"]
    footing = got["mat"]["footing"]
    above = [p for p, v in got["pending"].items()
             if v.split("[")[0] == footing and p[1] > fy + Builder.STOREY + 1]
    assert above, "the cottage lost its chimney"
    cols = {(p[0], p[2]) for p in above}
    assert len(cols) <= 2, f"the stack is {len(cols)} columns wide"
    top = max(p[1] for p in above)
    # the eave is the wall height, one or two storeys; the stack is at most
    # CHIMNEY_ABOVE_EAVE above it and its cap slab one more
    storeys = 2 if any(p[1] > fy + 2 * Builder.STOREY + 2 for p in got["pending"]
                       if got["pending"][p].split("[")[0] == got["mat"]["wall"]) else 1
    eave = fy + storeys * Builder.STOREY
    assert top <= eave + Builder.CHIMNEY_ABOVE_EAVE + 1, (top, eave, storeys)
    none = _stand("cottage", voice, 9, chimney=False)
    gone = [p for p, v in none["pending"].items()
            if v.split("[")[0] == footing and p[1] > fy + Builder.STOREY + 1]
    assert not gone, "the word 'no' left a stack"
    return (f"a {storeys}-storey cottage keeps its stack, topping out {top - eave} above "
            f"the eave against {Builder.CHIMNEY_ABOVE_EAVE} + the cap; the word 'no' "
            f"drops it")


@case
def t_3_a_roof_over_the_cap_is_eased_and_one_under_it_is_untouched():
    from ethoslm.prims import Primitives
    from ethoslm.buildlib import Builder
    b = Builder.__new__(Builder)
    for style, pitch, half in (("gable", (1, 1), 5), ("hip", (1, 2), 8),
                               ("gambrel", (2, 1), 6), ("mansard", (2, 1), 6)):
        plain = Primitives._roof_profile(b, half, style, *pitch, None, "straight")
        same = Primitives._roof_profile(b, half, style, *pitch, None, "straight",
                                        rise_max=99)
        assert plain == same, (style, "a cap nothing reaches changes nothing")
    steep = Primitives._roof_profile(b, 8, "gable", 2, 1, None, "straight")
    eased = Primitives._roof_profile(b, 8, "gable", 2, 1, None, "straight", rise_max=6)
    assert max(h for h, _k in steep) > 6 >= max(h for h, _k in eased)
    assert len(eased) == len(steep) == 8, "the span is the span"
    # eased means the run doubled, never a flat cut: the profile still climbs
    hs = [h for h, _k in eased]
    assert hs == sorted(hs) and hs[-1] > hs[0], hs
    return (f"presets with a cap nothing reaches are byte-identical; a (2,1) gable on "
            f"a half-span of 8 rises {max(h for h, _k in steep)} and eased under 6 "
            f"rises {max(hs)}, still climbing")


# ------------------------------------------------------- phase 4: the wall's two faces

def _stand_ring(name, voice, params, *, gates=True, half=30, seed=1, face=None):
    """A closed square ring wall of a committed edge type, width 3, with one gate
    annotated on its north side, on flat ground. Returns the build's answer and the
    pending blocks."""
    import collections
    import test_types
    from ethoslm import offline
    from ethoslm.buildlib import Builder
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
    mat = pipeline.voice_palette(voice)
    roof = pipeline.voice_roof(voice)
    c = half + 40
    path = [[c - half, c - half], [c + half, c - half], [c + half, c + half],
            [c - half, c + half], [c - half, c - half]]
    part = {"label": "t", "kind": "edge", "width": 3, "path": path}
    if face:
        part["face"] = face
    if gates:
        part["gates"] = [{"name": "g", "at": [c, c - half], "size": 11}]
    vol = test_types._flat_world(2 * c)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = None
    sited = b.site(dict(part), mat=mat, roof=roof)
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)                  # noqa: S102
    res = ns["build"](b.type_builder(sited, role=ns.get("ROLE")), sited, seed, **params)
    b.resolve_steps()
    return {"res": res, "shape": tuple(sorted(b._pending)),
            "blocks": collections.Counter(v.split("[")[0] for v in b._pending.values()),
            "pending": dict(b._pending), "floor_y": int(sited["floor_y"])}


@case
def t_4_the_great_wall_carries_sparse_stairs_at_its_corners_and_gates_only():
    decl = pipeline.load_type(os.path.join(ROOT, "types", "great_wall.py"))
    # the craft round (E4): `masonry` is what an unbroken wall gets, and `unbroken` --
    # the plain face on both sides with sparse stairs -- says nothing it does not
    assert list(decl["params"]["face"][1]) == ["framed", "banded", "plain", "masonry"]
    assert len(pipeline.param_combinations(decl["params"])) <= 27
    out = {}
    for voice in ("ochre_stone_green_tile", "white_render_dark_frame"):
        # a face with no articulation on it has nothing for a switchback every 26
        # columns to belong to, and the ground look read the inner rings' mural stairs
        # as diagonal bracing painted on the wall. `framed` is the default and is
        # `every` as it was.
        for stairs, face in (("every", "framed"), ("sparse", "masonry")):
            got = _stand_ring("great_wall", voice, {"height": 30, "width": 3,
                                                    "parapet": "crenellated",
                                                    "face": face})
            assert got["res"] and got["res"].get("closed"), got["res"]
            out[(voice, stairs)] = got
        every, sparse = out[(voice, "every")], out[(voice, "sparse")]
        assert sparse["res"]["stair_blocks"] < every["res"]["stair_blocks"], (
            sparse["res"]["stair_blocks"], every["res"]["stair_blocks"])
        # four corners and one gate: five ways down, one beside each
        assert sparse["res"]["stair_blocks"] == 5, sparse["res"]["stair_blocks"]
        assert len(sparse["shape"]) < len(every["shape"])
        # the default is today's bytes
        dflt = _stand_ring("great_wall", voice, {"height": 30, "width": 3,
                                                 "parapet": "crenellated"})
        assert dflt["shape"] == every["shape"], "the default moved"
        # ...and a plain face is sparse too, on the same five ways down
        plain = _stand_ring("great_wall", voice, {"height": 30, "width": 3,
                                                  "parapet": "crenellated",
                                                  "face": "plain"})
        assert plain["res"]["stair_blocks"] == sparse["res"]["stair_blocks"] == 5, \
            (plain["res"]["stair_blocks"], sparse["res"]["stair_blocks"])
    e, sp = out[("ochre_stone_green_tile", "every")], out[("ochre_stone_green_tile", "sparse")]
    return (f"a closed ring of four sides with one gate: {e['res']['stair_blocks']} ways "
            f"down on a framed face, {sp['res']['stair_blocks']} on a masonry one and on a "
            f"plain one (four corners, one gate), in both voices; the default is "
            f"'framed' and 'every' byte for byte")


@case
def t_4_the_wall_honours_a_plain_face_above_the_face_line():
    import importlib.util
    sp = importlib.util.spec_from_file_location("wall_t", os.path.join(ROOT, "types", "wall.py"))
    wall = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(wall)
    decl = pipeline.load_type(os.path.join(ROOT, "types", "wall.py"))
    assert "face" not in decl["params"] and wall.FACES == ("framed", "plain")
    assert len(pipeline.param_combinations(decl["params"])) <= 27
    for voice in ("ochre_stone_green_tile", "white_render_dark_frame"):
        framed = _stand_ring("wall", voice, {"height": 20, "width": 3, "crown": "crenellated"},
                             gates=False, face="framed")
        plain = _stand_ring("wall", voice, {"height": 20, "width": 3, "crown": "crenellated"},
                            gates=False, face="plain")
        assert set(plain["shape"]) < set(framed["shape"]), voice
        # what plain drops is outside the wall's own three columns: the buttresses and
        # the string course a block proud of the face
        cols_f = {(p[0], p[2]) for p in framed["shape"]}
        cols_p = {(p[0], p[2]) for p in plain["shape"]}
        assert cols_p < cols_f, "plain should stand in fewer columns"
        dflt = _stand_ring("wall", voice, {"height": 20, "width": 3, "crown": "crenellated"},
                           gates=False)
        assert dflt["shape"] == framed["shape"], "the default moved"
        low_f = _stand_ring("wall", voice, {"height": 8, "width": 3, "crown": "crenellated"},
                            gates=False, face="framed")
        low_p = _stand_ring("wall", voice, {"height": 8, "width": 3, "crown": "crenellated"},
                            gates=False, face="plain")
        assert low_f["shape"] == low_p["shape"], "below FACE_FROM there is no face to drop"
    return (f"at 20 the plain face drops {len(framed['shape']) - len(plain['shape'])} blocks "
            f"proud of the face in {len(cols_f) - len(cols_p)} columns, in both voices; at 8 "
            f"the two are identical; the default is framed")


@case
def t_4_the_layout_chooses_both_faces_from_the_specs_words():
    from ethoslm import placeplan
    import test_rings
    assert placeplan.wall_stairs_for({"notes": "an unbroken wall"}) == "sparse"
    assert placeplan.wall_stairs_for({"notes": "a timber palisade"}) == "every"
    assert placeplan.wall_stairs_for({"notes": "the outer wall"},
                                     {"invariants": "sheer, seamless walls"}) == "sparse"
    assert placeplan.wall_stairs_for({}) == "every"
    assert placeplan.WALL_STAIR_WORDS == ("unbroken", "sheer", "seamless", "smooth")
    s4 = test_rings.ring_spec()             # "monolithic and unbroken"
    place, fails, decls, site, plateau = test_rings._layout(s4)
    assert not fails, fails
    walls = [p for p in place["parts"] if p["kind"] == "edge"]
    chosen = {p["type"]: (p["params"].get("face"), p.get("face")) for p in walls}
    assert chosen.get("great_wall") == ("masonry", "plain"), chosen
    for p in walls:
        if p["type"] == "wall":
            assert p.get("face") == "plain" and "face" not in p["params"], p
    s3 = test_rings.three_ring_spec()
    place3, _f, _d, _s, _p = test_rings._layout(s3)
    walls3 = {p["type"]: p["params"].get("face") for p in place3["parts"] if p["kind"] == "edge"}
    return (f"'unbroken' chooses the masonry face (dressed, sparse stairs) on the great "
            f"wall and the plain face on the wall part: {chosen}; the three-ring spec's "
            f"walls carry {walls3}")


@case
def t_4_both_wall_types_sweep_clean_in_every_face_in_both_voices():
    """Every parameter combination of `great_wall` (24) and `wall` (27), each in both
    voices, on a closed ring with a gate, read by the build family of the lint on its
    own swept line -- the check `pipeline.check_type` applies, bounded to one fixture
    so it runs in minutes rather than hours."""
    from ethoslm import lint
    results = {}
    for name in ("great_wall", "wall"):
        decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
        combos = pipeline.param_combinations(decl["params"])
        assert len(combos) <= 27, (name, len(combos))
        faces = ["framed", "plain"] if name == "wall" else [None]
        n_ok = n_err = 0
        worst = None
        for voice in ("ochre_stone_green_tile", "white_render_dark_frame"):
            for params in combos:
                for face in faces:
                    got = _stand_ring(name, voice, dict(params), gates=(name == "great_wall"),
                                      half=24, face=face)
                    assert got["res"] is not None, (name, voice, params)
                    # the lint, on the wall's own swept line, as the checker reads it
                    import test_types
                    base = test_types._flat_world(2 * (24 + 40))
                    built = base.overlay(got["pending"])
                    row = pipeline.part_registry_row({
                        "name": "t", "label": "t", "kind": "edge", "width": 3,
                        "path": [[40, 40], [88, 40], [88, 88], [40, 88], [40, 40]]})
                    row["y0"] = got["floor_y"]
                    ctx = lint.Context.build(built, plots=[row], network=None,
                                             region=(30, 30, 98, 98), base=base)
                    rep = lint.lint(ctx, family=lint.BUILD).within([row])
                    errs = [f for f in rep.findings if f.code.startswith("E")]
                    if errs:
                        n_err += 1
                        worst = worst or (voice, params, face, [f.message for f in errs][:3])
                    else:
                        n_ok += 1
        results[name] = (n_ok, n_err, worst)
        assert n_err == 0, (name, n_err, worst)
    return (f"great_wall {results['great_wall'][0]} instances clean, wall "
            f"{results['wall'][0]} clean (every combination, both faces where the face "
            f"is the part's, both voices), 0 with an error")


# ------------------------------------------------- phase 4b: a ring may be round

@case
def t_4b_an_octagonal_rings_every_segment_is_sited_and_a_gate_on_a_diagonal_run_is_annotated():
    from ethoslm import offline, placeplan
    from ethoslm.buildlib import Builder
    from ethoslm.pipeline import stages_build
    import test_types
    path = placeplan._octagon_path(80, 80, 40, 128)
    assert len(path) == 9 and path[0] == path[-1]
    diag = [(a, b) for a, b in zip(path, path[1:]) if a[0] != b[0] and a[1] != b[1]]
    assert len(diag) == 4 and all(abs(b[0] - a[0]) == abs(b[1] - a[1]) for a, b in diag)
    vol = test_types._flat_world(160)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    part = {"label": "ring", "kind": "edge", "width": 3, "path": path}
    sited = b.site(dict(part), mat=pipeline.voice_palette("ochre_stone_green_tile"))
    assert sited["sited"]["ok"], sited["sited"]
    axes = [sg["axis"] for sg in sited["segments"]]
    assert axes.count("d") == 4 and set(axes) == {"x", "z", "d"}, axes
    # the diagonal's cells are a staircase of the width: three parallel diagonals
    dseg = next(sg for sg in sited["segments"] if sg["axis"] == "d")
    cells = {tuple(c) for c in dseg["cells"]}
    a = dseg["a"]
    assert (a[0], a[1]) in cells and len(cells) == 3 * (abs(dseg["b"][0] - a[0]) + 1)
    # the plan's edge cells and the registry's rectangles agree with the siting
    plan_cells = set(placeplan._edge_cells(part))
    all_sited = {tuple(c) for sg in sited["segments"] for c in sg["cells"]}
    assert plan_cells == all_sited, (len(plan_cells), len(all_sited))
    rects = pipeline.part_rects(part)
    covered = {(x, z) for (x0, z0, x1, z1) in rects for x in range(x0, x1 + 1)
               for z in range(z0, z1 + 1)}
    assert all_sited <= covered and len(covered) <= len(all_sited) * 2, "the boxes are the line"
    # a gate on a diagonal run is found by its edge and sized by it; one on the axial
    # side likewise
    mid = ((diag[0][0][0] + diag[0][1][0]) // 2, (diag[0][0][1] + diag[0][1][1]) // 2)
    wall_part = {"kind": "edge", "name": "ring", "type": "great_wall", "width": 3,
                 "path": path, "params": {"height": 30}}
    gate_d = {"kind": "point", "name": "g_d", "type": "ring_gate", "at": [mid[0], mid[1]],
              "facing": "north"}
    gate_a = {"kind": "point", "name": "g_a", "type": "ring_gate", "at": [80, 40],
              "facing": "north"}
    done = stages_build.annotate_gates([wall_part, gate_d, gate_a])
    assert set(done) == {"g_d", "g_a"}, done
    assert gate_d["edge"]["name"] == "ring" and gate_d["size"] == Builder.point_pad(30, None)
    assert {g["name"] for g in wall_part["gates"]} == {"g_d", "g_a"}
    return (f"an octagon of 8 segments, 4 of them 45-degree runs, sited on every segment; the "
            f"diagonal's {len(cells)} cells are three parallel staircases; the plan's cells, "
            f"the registry's boxes and the siting agree; a gate on a diagonal run and one on "
            f"an axial run are both annotated to the ring")


@case
def t_4b_the_great_wall_builds_an_octagon_clean_in_every_face_in_both_voices():
    """The gate of phase 4b: the diagonal run is clean on the type's sweep, both faces,
    both voices, or the type may not declare `DIAGONAL_RUNS` and the layout draws
    squares."""
    from ethoslm import lint, offline, placeplan
    from ethoslm.buildlib import Builder
    import test_types
    decl = pipeline.load_type(os.path.join(ROOT, "types", "great_wall.py"))
    assert decl["diagonal"] is True
    assert pipeline.load_type(os.path.join(ROOT, "types", "wall.py"))["diagonal"] is False
    path = placeplan._octagon_path(64, 64, 24, 128)
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)                  # noqa: S102
    n_ok = n_err = 0
    worst = None
    for voice in ("ochre_stone_green_tile", "white_render_dark_frame"):
        mat = pipeline.voice_palette(voice)
        roof = pipeline.voice_roof(voice)
        for params in pipeline.param_combinations(decl["params"]):
            base = test_types._flat_world(128)
            b = Builder(offline.OfflineSite(base))
            b._vol = base
            b.frontage = None
            part = {"label": "t", "kind": "edge", "width": 3, "path": path,
                    "gates": [{"name": "g", "at": [64, 40], "size": 11}]}
            sited = b.site(dict(part), mat=mat, roof=roof)
            res = ns["build"](b.type_builder(sited, role=ns.get("ROLE")), sited, 1, **params)
            b.resolve_steps()
            assert res and res.get("closed"), (voice, params, res)
            built = base.overlay(dict(b._pending))
            row = pipeline.part_registry_row({"name": "t", "label": "t", "kind": "edge",
                                              "width": 3, "path": path})
            row["y0"] = sited["floor_y"]
            ctx = lint.Context.build(built, plots=[row], network=None,
                                     region=(30, 30, 98, 98), base=base)
            rep = lint.lint(ctx, family=lint.BUILD).within([row])
            errs = [f for f in rep.findings if f.code.startswith("E")]
            # every diagonal cell carries the wall's body
            dsegs = [sg for sg in sited["segments"] if sg["axis"] == "d"]
            floor = sited["floor_y"]
            for sg in dsegs:
                for c in sg["cells"]:
                    assert (c[0], floor + 3, c[1]) in b._pending, ("a diagonal cell is empty", c)
            if errs:
                n_err += 1
                worst = worst or (voice, params, [f.message for f in errs][:3])
            else:
                n_ok += 1
    assert n_err == 0, (n_err, worst)
    return (f"{n_ok} octagonal great walls (every parameter combination, both voices) "
            f"linted clean on their own swept line, every diagonal cell built")


@case
def t_4b_the_layout_draws_octagons_where_the_spec_says_round_and_squares_where_a_type_cannot():
    from ethoslm import placeplan
    import test_rings
    doc = json.loads(json.dumps(test_rings.THREE_RING))
    square = spec_mod.read_spec(json.loads(json.dumps(doc)), "Build a ringed town.")
    doc["invariants"] = "A round walled capital: its rings are circular, one within the next."
    rnd_spec = spec_mod.read_spec(doc, "Build a ringed town.")
    assert placeplan.wall_round_for({}, rnd_spec) and not placeplan.wall_round_for({}, square)
    assert placeplan.RING_CHAMFER == 0.25
    place, fails, decls, site, plateau = test_rings._layout(rnd_spec)
    assert not fails, fails
    lay = place["layout"]
    assert lay["round"]["asked"] and lay["registered"]["RING_CHAMFER"] == 0.25
    walls = [p for p in place["parts"] if p["kind"] == "edge"]
    for p in walls:
        n_diag = sum(1 for a, b in zip(p["path"], p["path"][1:]) if a[0] != b[0] and a[1] != b[1])
        if p["type"] == "great_wall":
            assert n_diag == 4 and p.get("shape") == "octagon", p["name"]
            assert lay["round"]["rings"][p["ring"] and next(r["name"] for r in lay["rings"] if r["ring"] == p["ring"])] == "octagon"
        else:
            assert n_diag == 0 and "shape" not in p, p["name"]
    assert all(v == "octagon" for v in lay["round"]["rings"].values()), lay["round"]
    # ...and for a type that draws no diagonal run the same words draw squares, the miss
    # named per ring: the gate, asked by handing the layout a declaration that says so
    gated = {n: dict(d, diagonal=False) for n, d in decls.items()}
    place_g, fails_g = placeplan.concentric_layout(rnd_spec, site, plateau, gated,
                                                   "ochre_stone_green_tile")
    assert not fails_g, fails_g
    misses = place_g["layout"]["round"]["rings"]
    assert misses and all("draws no diagonal run" in v for v in misses.values()), misses
    assert all(len(p["path"]) == 5 or all(a[0] == b[0] or a[1] == b[1]
                                          for a, b in zip(p["path"], p["path"][1:]))
               for p in place_g["parts"] if p["kind"] == "edge")
    # the octagon layout is held to the same validator and concentric clauses
    got = placeplan.place_failures(place, rnd_spec, site, decls, ground={}, plateau=plateau)
    assert got == [], got
    assert placeplan.concentric_failures(place, rnd_spec, decls) == []
    # ...and every ring is still covered
    assert all(r["coverage"] >= placeplan.RING_COVERAGE for r in lay["rings"]), \
        [(r["name"], r["coverage"]) for r in lay["rings"]]
    # a place that is not round is byte-identical to the layout it always had
    place2, _f, _d, _s, _p = test_rings._layout(square)
    assert place2["layout"]["round"] is None
    assert all(len(p["path"]) == 5 or all(a[0] == b[0] or a[1] == b[1]
                                          for a, b in zip(p["path"], p["path"][1:]))
               for p in place2["parts"] if p["kind"] == "edge")
    return (f"'circular' in the invariants: the great_wall rings are octagons (chamfer 0.25); "
            f"a type that draws no diagonal gets squares with the miss named "
            f"({list(misses.values())[0]}); the "
            f"validator and the concentric clauses hold; coverage "
            f"{[r['coverage'] for r in lay['rings']]}; a square spec is untouched")


# ---------------------------------- phase 5: the bars tell the truth, the harness
# wastes nothing

def _flat_volume(size, y=64):
    from ethoslm.observe import Volume
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


def _two_district_state(tmp, workers):
    """A place of two quarters far apart, four cottages each, on flat ground."""
    from ethoslm import offline
    vol = _flat_volume(200)
    offline.save_volume(vol, os.path.join(tmp, "world.npz"))
    leaves = {"q_west": [(20 + 18 * i, 40) for i in range(4)],
              "q_east": [(20 + 18 * i, 140) for i in range(4)]}
    parts = []
    for q, cells in leaves.items():
        kids = [{"kind": "plot", "name": f"{q}_{k}", "type": "cottage", "seed": k + 1,
                 "params": {}, "x0": x, "z0": z, "x1": x + 12, "z1": z + 12}
                for k, (x, z) in enumerate(cells)]
        parts.append({"kind": "quarter", "name": q, "children": kids})
    plan = {"parts": parts, "voice": "white_render_dark_frame", "centre": None}
    json.dump(plan, open(os.path.join(tmp, "plan.json"), "w"))
    rows = [pipeline.part_registry_row(q) for q in pipeline.plan_parts(plan)]
    json.dump(rows, open(os.path.join(tmp, "plots.json"), "w"))
    rnd = pipeline.Round(name="parallel_fixture", state_dir=tmp,
                         voice="white_render_dark_frame",
                         flags={"dry_run": True, "workers": workers})
    return rnd, pipeline.OfflineBackend(rnd, dry_run=True)


def _names(vol):
    return np.array(vol.palette, dtype=object)[vol.codes]


@case
def t_5_e008_is_counted_in_the_own_errors_bar():
    from ethoslm import offline
    from ethoslm.circulate import Network, Threshold
    from ethoslm.pipeline import stages_measure
    G = 64
    with tempfile.TemporaryDirectory() as tmp:
        base = _flat_volume(96, y=G)
        offline.save_volume(base, os.path.join(tmp, "world.npz"))
        built = _flat_volume(96, y=G)
        # a hut, and a stone block standing on its reserved doorstep
        blocks = {}
        for x in range(30, 40):
            for z in range(30, 40):
                for y in range(G + 1, G + 5):
                    if x in (30, 39) or z in (30, 39) or y == G + 4:
                        blocks[(x, y, z)] = "stone"
        blocks[(35, G + 1, 29)] = "stone"
        built = built.overlay(blocks)
        offline.save_volume(built, os.path.join(tmp, "world_built.npz"))
        cells = {(x, 26): {"y": G, "rank": 0, "face": None} for x in range(20, 50)}
        cells[(35, 27)] = {"y": G, "rank": 1, "face": None}
        cells[(35, 28)] = {"y": G, "rank": 1, "face": None}
        net = Network(cells, [Threshold("hut", 35, 28, G, "south", (35, G + 1, 29))])
        net.save(os.path.join(tmp, "network.json"))
        json.dump([{"label": "hut", "x0": 30, "z0": 30, "x1": 39, "z1": 39,
                    "kind": "plot", "y0": G}], open(os.path.join(tmp, "plots.json"), "w"))
        json.dump({"parts": [{"kind": "quarter", "name": "q", "children": [
            {"kind": "plot", "name": "hut", "x0": 30, "z0": 30, "x1": 39, "z1": 39}]}]},
            open(os.path.join(tmp, "plan.json"), "w"))
        json.dump({"origin": [0, 0], "size": 96}, open(os.path.join(tmp, "site.json"), "w"))
        rnd = pipeline.Round(name="e008_fixture", state_dir=tmp, flags={"dry_run": True})
        be = pipeline.OfflineBackend(rnd, dry_run=True)
        got = stages_measure._m_own_lint_errors(rnd, be, {}, "0")
    assert got["e008_place_wide"] >= 1, got
    assert got["got"] == got["own_ground"] + got["e008_place_wide"], got
    assert "E008" in got["scoped_by"], got["scoped_by"]
    assert any("hut" in m for m in got["e008_examples"]), got["e008_examples"]
    return (f"a doorstep built over: own ground {got['own_ground']} + E008 place-wide "
            f"{got['e008_place_wide']} = {got['got']}; the record says which")


@case
def t_5_the_preflights_refuse_on_fixtures():
    from ethoslm.pipeline import stages_build, stages_media
    lines = []
    parts = [{"kind": "plot"}] * 300 + [{"kind": "edge"}] * 4
    ok = stages_build.preflight_parts(parts, ["ochre_stone_green_tile"], workers=4,
                                      log=lines.append)
    assert not ok["refused"] and not ok["failures"], ok
    assert ok["estimated_seconds"] == round(304 * 31.0 / 4), ok
    assert ok["estimated_blocks"] == 304 * 5600 and ok["by_kind"] == {"plot": 300, "edge": 4}
    assert ok["registered"] == {"COST_PART_S": 31.0, "COST_PART_BLOCKS": 5600,
                                "PARTS_BOUND_S": 21600.0}
    over = stages_build.preflight_parts(parts, ["ochre_stone_green_tile"], workers=1,
                                        bound=3600.0, log=lines.append)
    assert over["refused"] and "REFUSED" in lines[-1], lines[-1]
    bad = stages_build.preflight_parts(parts, ["no_such_voice"], workers=4, log=lines.append)
    assert bad["refused"] and bad["failures"] and "no palette" in bad["failures"][0], bad
    rows = [{"name": "a", "why": "place"}] + [{"name": f"b{i}", "why": "defining"}
                                            for i in range(20)]
    pr = stages_media.preflight_render(rows, {"flythrough": True}, log=lines.append)
    assert pr["frames"] == 21 * 4 + 48 and not pr["refused"], pr
    assert pr["estimated_seconds"] == round(pr["frames"] * 14.0)
    tight = stages_media.preflight_render(rows, {"flythrough": False}, bound=60.0,
                                          log=lines.append)
    assert tight["refused"] and tight["frames"] == 84, tight
    return (f"parts: 304 leaves at 31s over 4 workers = {ok['estimated_seconds']}s, "
            f"allowed; one worker against an hour refused; an unknown voice refused by "
            f"name; render: {pr['frames']} frames = {pr['estimated_seconds']}s allowed, "
            f"84 frames against 60s refused")


@case
def t_5_a_parallel_parts_run_is_byte_identical_to_the_sequential_one():
    from ethoslm.pipeline import stages_build
    import shutil
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for workers in (1, 2):
            d = os.path.join(tmp, f"w{workers}")
            os.makedirs(d)
            rnd, be = _two_district_state(d, workers)
            os.environ.pop("ETHOSLM_WORKERS", None)
            res = stages_build.stage_parts(rnd, be, {})
            assert res.get("built") == 8 and not res.get("failed"), (workers, res.get("failed"))
            rows = [{k: v for k, v in r.items() if k not in ("seconds", "program")}
                    for w in res["waves"] for r in w["parts"]]
            waves = [(w["wave"], w["errors"], w["scope"]) for w in res["waves"]]
            out[workers] = {"names": _names(be.volume), "rows": rows, "waves": waves,
                            "parallel": res.get("parallel"),
                            "paths": json.load(open(os.path.join(d, "paths.json")))
                            if os.path.exists(os.path.join(d, "paths.json")) else []}
    assert out[2]["parallel"] and out[2]["parallel"]["workers"] == 2, out[2]["parallel"]
    assert out[1]["parallel"] is None
    assert np.array_equal(out[1]["names"], out[2]["names"]), "the built volumes differ"
    assert out[1]["rows"] == out[2]["rows"], "the parts records differ"
    assert out[1]["waves"] == out[2]["waves"], "the waves' lints differ"
    assert sorted(map(json.dumps, out[1]["paths"])) == sorted(map(json.dumps, out[2]["paths"]))
    return (f"two quarters of four cottages: one worker and two lay the same "
            f"{int((out[1]['names'] != 'air').sum())}-block world, the same rows, the same "
            f"lints and the same paths")


@case
def t_5_the_district_calls_are_asked_for_as_a_batch():
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    place, fails, decls, site, plateau = test_rings._layout(s)
    assert not fails
    site = {**site, "mean_grid": [[64] * 12] * 12, "roughness_grid": [[1] * 12] * 12,
            "stats": {"min": 60, "max": 70, "relief": 10}}
    with tempfile.TemporaryDirectory() as tmp:
        rnd = pipeline.Round(name="batch_fixture", sentence="Build a ringed town.",
                             state_dir=tmp, voice="ochre_stone_green_tile")
        asks = stages_plan.district_asks(rnd, s, site, place, None, "ochre_stone_green_tile")
        n = len(place["districts"])
        assert len(asks) == n >= 4, (len(asks), n)
        for k, v in asks.items():
            assert v["status"] == "needs_model" and os.path.exists(v["request"]), k
            assert k.startswith("plan/district/")
        # an answered district drops out of the batch; the rest stay
        first = next(iter(asks.values()))
        json.dump({}, open(first["write"], "w"))
        again = stages_plan.district_asks(rnd, s, site, place, None, "ochre_stone_green_tile")
        assert len(again) == n - 1
        # **The driver waits on every one of them.** `_needs_model` returns the agent
        # jobs themselves and not their keys -- `model.staged`'s rule since the
        # unification round, so that a stage answering *flat* is visible to the loop
        # that waits -- so what is asserted is that the flat entry and every district's
        # are all in the list, one each.
        waiting = pipeline._needs_model({"plan": {"status": "needs_model"}, **asks})
        assert len(waiting) == 1 + len(asks), waiting
        assert {w.get("write") for w in waiting if w.get("write")} == \
            {v["write"] for v in asks.values()}, waiting
    return (f"{n} districts asked for at once, each its own entry with its brief on "
            f"disk; one answered leaves {n - 1}; the driver waits on all of them")


# ---------------------------------------------------------------- phase 6: cameras

@case
def t_6_the_flythrough_rises_to_clear_every_wall_it_crosses_and_settles_after():
    import test_camera
    from ethoslm import render
    from ethoslm.observe import Volume
    plan, spec, site, sm = test_camera._two_ring_plan()
    path = sm.flythrough_path(plan, spec, site)
    G, Y0 = test_camera.GROUND, test_camera.Y0
    flat = {(x, G, z): "stone" for x in range(-64, 160) for z in range(-64, 160)}
    open_vol = Volume.from_blocks(flat, -64, Y0, -64, 224, 40, 224)
    before = render.flythrough_shots(path, open_vol)
    n = len(path)
    # a wall 20 high across the path a third of the way along, well short of the
    # subject.
    k = n // 3
    ax, az = path[k]["at"]
    walled = dict(flat)
    for x in range(int(ax) - 30, int(ax) + 31):
        for y in range(G + 1, G + 21):
            walled[(x, y, int(az))] = "stone"
            walled[(int(az) + 0, y, x)] = "stone"
    wall_vol = Volume.from_blocks(walled, -64, Y0, -64, 224, 40, 224)
    after = render.flythrough_shots(path, wall_vol)
    ys_b = [before[f"fly_{i:03d}"].view.position[1] for i in range(n)]
    ys_a = [after[f"fly_{i:03d}"].view.position[1] for i in range(n)]
    crown = G + 20
    # every frame that crosses the wall stands FLY_RISE above its crown
    crossing = [i for i in range(n - 1)
                if render.tallest_between(wall_vol, tuple(path[i]["at"]),
                                          tuple(path[i + 1]["at"]), margin=0.0,
                                          stop=1e9) == crown]
    assert crossing, "the fixture's wall is not on the path"
    for i in crossing:
        assert ys_a[i] >= crown + render.FLY_RISE, (i, ys_a[i], crown)
        assert after[f"fly_{i:03d}"].repair.get("crown", {}).get("crown") == crown
    # ...lifting ahead: the FLY_LIFT_FRAMES frames before the wall are already up
    first = min(crossing)
    for i in range(max(0, first - render.FLY_LIFT_FRAMES), first):
        assert ys_a[i] >= crown + render.FLY_RISE, (i, ys_a[i])
    # ...and settling after: no frame drops more than FLY_SETTLE below the one before
    for i in range(1, n):
        assert ys_a[i] >= ys_a[i - 1] - render.FLY_SETTLE - 1e-6, (i, ys_a[i - 1], ys_a[i])
    assert any(after[f"fly_{i:03d}"].repair.get("crown", {}).get("settling")
               for i in range(max(crossing) + 1, n)), "nothing settled after the wall"
    # frames well before the wall are untouched, and open ground is byte for byte
    for i in range(0, max(0, first - render.FLY_LIFT_FRAMES)):
        assert after[f"fly_{i:03d}"].view.position == before[f"fly_{i:03d}"].view.position, i
    again = render.flythrough_shots(path, open_vol)
    assert all(again[q].view.position == before[q].view.position for q in before)
    assert not any(again[q].repair for q in again)
    assert render.FLY_SETTLE == 3
    return (f"a 20-high wall across frame {first}: frames {first - render.FLY_LIFT_FRAMES}.."
            f"{max(crossing)} stand at y>={crown + render.FLY_RISE} (from {ys_b[first]:.0f}), "
            f"the move settles at {render.FLY_SETTLE} a frame after it, the frames before "
            f"and open ground are byte-identical")


@case
def t_6_a_gate_eye_rung_in_a_canopy_is_passed_over_for_the_next():
    from ethoslm import render
    from ethoslm.circulate import Threshold
    from ethoslm.observe import Volume
    G = 64
    flat = {(x, G, z): "stone" for x in range(0, 120) for z in range(0, 120)}
    vol = Volume.from_blocks(flat, 0, G - 10, 0, 120, 50, 120)
    t = Threshold("gate", 60, 40, G, "south", (60, G + 1, 41))
    subject = {"centre": (60.0, G + 8, 40.0), "span": 30.0, "outward": (0, -1),
               "rise": 20}
    clear = render.gate_card_ladder(subject, t, sight=render.Sightline(vol))["eye"]
    assert clear and not any(r.repair.get("foliage") for r in clear)
    pos0 = clear[0].view.position
    # an acacia canopy round the first rung's eye and nothing else
    leafy = dict(flat)
    x, y, z = (int(pos0[0]), int(pos0[1]), int(pos0[2]))
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            for dy in range(-1, 2):
                leafy[(x + dx, y + dy, z + dz)] = "acacia_leaves"
    lv = Volume.from_blocks(leafy, 0, G - 10, 0, 120, 50, 120)
    sight = render.Sightline(lv)
    assert sight.foliage(pos0) > 0 and sight.foliage((10.5, G + 3, 10.5)) == 0
    got = render.gate_card_ladder(subject, t, sight=sight)["eye"]
    assert got[0].view.position != pos0, "the leafy rung is still first"
    assert sight.foliage(got[0].view.position) == 0, "the leading rung is clear"
    leafy_rungs = [r for r in got if r.repair.get("foliage")]
    assert leafy_rungs, "no rung was marked"
    assert all(sight.foliage(r.view.position) > 0 for r in leafy_rungs)
    assert got[-len(leafy_rungs):] == leafy_rungs, "leafy rungs come last"
    assert len(got) == len(clear)
    assert render.FOLIAGE_REACH == 3
    return (f"a canopy of {sight.foliage(pos0)} leaf blocks round the first eye rung: it "
            f"is passed over for the next and marked; without leaves the rungs are the "
            f"bytes they were")


@case
def t_6_a_frame_renders_only_the_ground_the_world_has():
    from ethoslm import render
    from ethoslm.observe import Volume

    class Regs:
        directory = "fixture"

        def chunk(self, cx, cz):
            from ethoslm.regions import NotGenerated
            if cx < 2:
                raise NotGenerated(f"chunk ({cx},{cz}) is not in its region file")
            return {}
    chunks = render.chunk_list(0, 0, 100, 40, pad=0)
    kept, dropped = render.finished_chunks(chunks, regions=Regs())
    assert all(c[0] >= 2 for c in kept) and all(c[0] < 2 for c in dropped)
    assert len(kept) + len(dropped) == len(chunks) and dropped
    G = 64
    flat = {(x, G, z): "stone" for x in range(0, 120) for z in range(0, 40)}
    vol = Volume.from_blocks(flat, 0, G - 10, 0, 120, 50, 40)
    shot = render.aim("place_skyline", (8.5, G + 10, 20.5), (100.5, G + 4, 20.5), 70,
                      render.Sightline(vol))
    have = {(c[0], c[1]) for c in kept}
    moved = render.onto_finished(shot, have)
    assert moved.view.position[0] >= 32 and moved.view.position[1] == shot.view.position[1]
    assert moved.repair.get("dollied_onto_finished_ground"), moved.repair
    same = render.onto_finished(render.aim("s", (50.5, G + 10, 20.5), (100.5, G + 4, 20.5),
                                           70, None), have)
    assert not same.repair.get("dollied_onto_finished_ground")
    return (f"{len(dropped)} of {len(chunks)} chunks the world never made are left out; a "
            f"camera over them is dollied {moved.repair['dollied_onto_finished_ground']} "
            f"blocks in along its line of sight onto finished ground, its height kept")


# ------------------------------------------------- phase 7: the harness for the run

@case
def t_7_the_footprint_ceiling_is_a_number_per_kind_and_the_record_reads_as_it_was():
    import test_rings
    assert spec_mod.footprint_ceiling("city") == 768
    assert all(spec_mod.footprint_ceiling(k) == 512 for k in ("town", "village", "hamlet",
                                                              "keep", "monument", None))
    assert spec_mod.footprint_for(260, "city") == 768 and spec_mod.footprint_for(260) == 512
    doc = json.loads(json.dumps(test_rings.THREE_RING))
    doc.pop("needs", None)                    # the fixture pins 512; a fresh city reads 768
    fresh = spec_mod.read_spec(doc, "Build a ringed capital on a plain.")
    assert fresh["needs"]["footprint"] == 768 and fresh["ceiling"]["footprint"] == 768
    again = spec_mod.read_spec(json.loads(json.dumps(fresh)), fresh["sentence"])
    assert again["needs"] == fresh["needs"]
    # a spec recorded under the old ceiling reads back at the footprint it was built at
    old = json.loads(json.dumps(fresh))
    old["needs"]["footprint"] = 512
    back = spec_mod.read_spec(old, fresh["sentence"])
    assert back["needs"]["footprint"] == 512, back["needs"]
    bad = json.loads(json.dumps(fresh))
    bad["needs"]["footprint"] = 600
    try:
        spec_mod.read_spec(bad, fresh["sentence"])
        raise AssertionError("a footprint nobody derived was accepted")
    except spec_mod.SpecError as e:
        assert "footprint" in str(e)
    # the round reads its checked spec where it wrote one
    with tempfile.TemporaryDirectory() as tmp:
        json.dump(doc, open(os.path.join(tmp, "place.json"), "w"))
        rnd = pipeline.Round(name="ceiling_fixture", sentence=fresh["sentence"], state_dir=tmp)
        assert rnd.place_spec()["needs"]["footprint"] == 768
        json.dump(old, open(os.path.join(tmp, "place.checked.json"), "w"))
        assert rnd.place_spec()["needs"]["footprint"] == 512
    from ethoslm.buildlib import max_blocks_for, MAX_BLOCKS
    assert max_blocks_for(512) == MAX_BLOCKS == 2_000_000 and max_blocks_for(None) == MAX_BLOCKS
    assert max_blocks_for(768) == int(2_000_000 * 2.25)
    return ("city 768, every other kind 512; a fresh city reads 768 and a recorded 512 "
            "reads 512, 600 refused; the round reads its checked spec; the block guard "
            "is 4.5M at 768")


@case
def t_7_the_terraces_preflight_estimates_the_fill_and_refuses_over_its_bound():
    from ethoslm import placeplan
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z = site["origin"]
    vol = _volume(X, Z, 512, base=50, relief=40)
    _t, decls = placeplan.types_card(None, s.get("form"))
    place, fails = placeplan.concentric_layout(s, site, test_rings._plateau(site), decls,
                                               "ochre_stone_green_tile", vol=vol)
    assert not fails
    lines = []
    pre = stages_plan.preflight_terraces(vol, place["layout"], log=lines.append)
    assert len(pre["rings"]) == 3 and pre["estimated_blocks"] > 1_000_000, pre
    assert not pre["refused"] and pre["bound_blocks"] == 12_000_000
    assert pre["registered"] == {"COST_TERRACE_BLOCK_S": 6e-6, "TERRACES_BOUND_BLOCKS": 12_000_000}
    tight = stages_plan.preflight_terraces(vol, place["layout"], bound=100_000, log=lines.append)
    assert tight["refused"] and "REFUSED" in lines[-1]
    return (f"three terraces over {pre['columns']:,} columns: about {pre['fill']:,} of fill "
            f"and {pre['cut']:,} of cut, {pre['estimated_seconds']}s, allowed against 12M; "
            f"refused against 100k")


@case
def t_7_the_ground_measure_reads_its_five_clauses_and_fails_by_name():
    from ethoslm import offline, placeplan, pipeline as pl
    from ethoslm.circulate import Network
    from ethoslm.pipeline import stages_measure, stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40)
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        doc = json.loads(json.dumps(test_rings.THREE_RING))
        doc["setting"]["biome"] = ["plains"]
        json.dump(doc, open(os.path.join(tmp, "place.json"), "w"))
        json.dump({**site, "mean_grid": [[70] * 12] * 12, "roughness_grid": [[3] * 12] * 12,
                   "stats": {"min": 50, "max": 90, "relief": 40}},
                  open(os.path.join(tmp, "site.json"), "w"))
        json.dump({"chosen": {"origin": [X, Z], "size": S, "measures": {
            "biome": {"read": True, "classes": {"plains": 81.0, "savanna": 4.0}}}},
                   "terraform": None, "attempt": "as asked"},
                  open(os.path.join(tmp, "site_search.json"), "w"))
        rnd = pl.Round(name="ground_measure_fixture", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        be = pl.OfflineBackend(rnd, dry_run=True)
        stages_plan.stage_plateau(rnd, be, {})
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(
            s, site, json.load(open(os.path.join(tmp, "plateau.json"))), decls,
            "ochre_stone_green_tile", vol=rnd.volume())
        assert not fails
        json.dump(place, open(os.path.join(tmp, "plan.json"), "w"))
        got = stages_plan.stage_terraces(rnd, be, {})
        assert got.get("placed")
        # the walls, sited on the terraces, as parts.json records them
        from ethoslm.buildlib import Builder
        v2 = rnd.volume()
        b = Builder(offline.OfflineSite(v2))
        b._vol = v2
        mat = pl.voice_palette("ochre_stone_green_tile")
        rows = []
        for q in place["parts"]:
            if q["kind"] != "edge":
                continue
            sited = b.site(dict(q, label=q["name"]), mat=mat)
            rows.append({"part": q["name"], "status": "built", "type": q["type"],
                         "kind": "edge", "ground": "footing", "floor_y": sited["floor_y"],
                         "sited": sited["sited"]["reason"]})
        rows.append({"part": "h1", "status": "built", "type": "cottage", "kind": "plot",
                     "ground": "plinth", "floor_y": 70, "sited": "plinth"})
        json.dump({"waves": [{"wave": "walls", "parts": rows}]},
                  open(os.path.join(tmp, "parts.json"), "w"))
        offline.save_volume(v2, os.path.join(tmp, "world_built.npz"))
        Network({}, []).save(os.path.join(tmp, "network.json"))
        json.dump([], open(os.path.join(tmp, "plots.json"), "w"))
        m = stages_measure._m_ground(rnd, be, {}, {})
        assert m["got"] == 1 and m["failed"] == [], m
        assert m["biome"]["share"] == 0.81 and m["walls"]["holds"] and m["podium"]["holds"]
        assert m["decks"]["decks"] == 0 and m["e008"]["place_wide"] == 0
        # ...and each clause fails by name
        rows[-1]["ground"] = "deck"
        rows[0]["sited"] = "a level per segment at y=60,64,60,64"
        json.dump({"waves": [{"wave": "walls", "parts": rows}]},
                  open(os.path.join(tmp, "parts.json"), "w"))
        m2 = stages_measure._m_ground(rnd, be, {}, {})
        assert m2["got"] == 0 and set(m2["failed"]) == {"decks", "walls"}, m2["failed"]
        lv = stages_measure._m_levels(rnd, be, {}, {})
        assert lv["podium"] == lv["terrace"]["podium"] and len(lv["rings"]) == 3
        pf = stages_measure._m_preflights(rnd, be, {}, {})
        assert "terraces" in pf and pf["refused"] == []
    return (f"biome {m['biome']['share']}, 0 decks, {len(m['walls']['parts'])} walls at "
            f"their ring's level, podium {m['podium']['podium']} over {m['podium']['ring_levels']}, "
            f"E008 0: holds; a deck and a stepping wall fail as 'decks' and 'walls'")


# ---------------------------- the second run: designed ground at the search, the hand-
# back

@case
def t_8_where_the_ground_is_designed_water_and_the_core_are_fill_against_a_budget():
    fs = _find_site()
    s = _spec({"surface": "green", "relief": "flat", "water": "none", "biome": ["plains"]})
    needs = fs.search_needs(s)
    assert needs["designed"] and needs["designed"]["n"] == 3 and needs["fill_max"] == 12_000_000
    assert needs["biome_window"] == fs.BIOME_INNER == "core and inner rings"
    assert fs.DESIGNED_FILL_MAX == 12_000_000
    lay = needs["designed"]
    assert lay["halves"][-1] == 512 // 2 - 4 and lay["inner_window"] == lay["halves"][-2]
    # a wet plains square with a hilly core: refused as a site as found, chosen as
    # ground to be designed -- the water and the core are fill inside the budget
    wet = _square(fs, 512, biome="plains", relief=20, water_pct=15.0)
    wet.h = np.where(wet.wet, wet.h - 6, wet.h)              # the lake has a bed
    m = fs.measure(wet, wet.x0, wet.z0, 512, 40, core=fs.core_size(s), design=lay)
    e = fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert m["fill"]["blocks"] > 0 and len(m["fill"]["per_ring"]) == 3, m["fill"]
    assert e["designed"] and e["water"] == 0.0 and e["core_water"] == 0.0 and e["fill"] == 0.0
    assert e["meets"], fs.failed_needs(e)
    assert 0 < e["fill_preference"] < 1
    # the same square held as found: refused for its water
    plain = dict(needs)
    plain.pop("designed"); plain.pop("fill_max")
    m2 = fs.measure(wet, wet.x0, wet.z0, 512, 40, core=fs.core_size(s))
    e2 = fs.excess(m2, plain, plateau_relief=fs.PLATEAU_RELIEF)
    assert not e2["meets"] and e2["water"] > 0 and not e2["designed"], fs.failed_needs(e2)
    # over the budget the fill is a need that fails by name
    tight = dict(needs, fill_max=1000.0)
    e3 = fs.excess(m, tight, plateau_relief=fs.PLATEAU_RELIEF)
    assert not e3["meets"] and e3["fill"] > 0 and any(w.startswith("fill") for w in e3["setting_failures"])
    assert "fill over by" in " ".join(fs.failed_needs(e3))
    # ...and among two that meet, the one with less to fill ranks first
    dry = _square(fs, 512, biome="plains", relief=20, water_pct=0.0, origin=210_000)
    md = fs.measure(dry, dry.x0, dry.z0, 512, 40, core=fs.core_size(s), design=lay)
    ed = fs.excess(md, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert md["fill"]["blocks"] < m["fill"]["blocks"]
    assert fs.rank_key({"measures": md, "excess": ed}) < fs.rank_key({"measures": m, "excess": e})
    return (f"a designed place: 15% water and a hilly core are {m['fill']['blocks']:,} blocks "
            f"of fill against 12M and the square meets; held as found it fails water; a "
            f"budget of 1,000 fails 'fill' by name; less to fill ranks first")


@case
def t_8_the_biome_share_of_a_designed_place_is_read_over_the_core_and_inner_rings():
    fs = _find_site()
    s = _spec({"surface": "green", "relief": "flat", "biome": ["plains"]})
    needs = fs.search_needs(s)
    lay = needs["designed"]
    w = lay["inner_window"]
    # plains inside the inner window, river across the belt: a plain with a river round
    # it, and a city on a plain
    f = _square(fs, 512, biome="plains")
    cells = np.ones(f.biome.shape, np.int32)                  # 1 = river
    c0, c1 = (256 - w) // fs.BIOME_CELL, -(-(256 + w) // fs.BIOME_CELL)
    cells[c0:c1, c0:c1] = 0                                   # 0 = plains
    f2 = fs.Field(f.x0, f.z0, f.h, f.wet, f.canopy, gravity=f.gravity, source="fixture",
                  manmade=f.manmade, occupied=f.occupied, surface=f.surface,
                  surface_palette=f.surface_palette, biome=cells,
                  biome_palette=["plains", "river"])
    m = fs.measure(f2, f2.x0, f2.z0, 512, 40, core=fs.core_size(s), design=lay)
    whole = m["biome"]["classes"]["plains"]
    inner = m["biome"]["inner"]["classes"]["plains"]
    assert whole < 50.0 < inner, (whole, inner)
    e = fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert e["meets"] and e["biome"] == 0.0, fs.failed_needs(e)
    # the same square for a place with no rings is read over the whole and fails
    flat = json.loads(json.dumps(s))
    for p in flat["defining_parts"]:
        for k in ("ring", "share", "walled"):
            p.pop(k, None)
    flat = {k: v for k, v in flat.items() if k not in ("needs",)}
    flat["defining_parts"] = [p for p in flat["defining_parts"] if p["family"] not in ("wall", "gate")]
    flat["needs"] = {"footprint": 512}
    fs2 = spec_mod.read_spec(flat, "Build a town on a plain.")
    n2 = fs.search_needs(fs2)
    assert "designed" not in n2
    m2 = fs.measure(f2, f2.x0, f2.z0, 512, 40, core=fs.core_size(fs2))
    e2 = fs.excess(m2, n2, plateau_relief=fs.PLATEAU_RELIEF)
    assert not e2["meets"] and e2["biome"] > 0
    return (f"plains {inner}% over the core and inner rings (a {2 * w}-square) against "
            f"{whole}% over the whole: the designed place meets; a place with no rings "
            f"reads the whole and fails")


@case
def t_8_a_spec_hand_back_changes_only_the_field_that_was_refused():
    from ethoslm.pipeline import stages_plan
    first = {"kind": "city", "invariants": "rings on a plain", "voice": None,
             "setting": {"surface": "green", "relief": "flat"},
             "defining_parts": [
                 {"name": "court", "kind": "group", "family": "palace", "relation": "centre",
                  "count": 1, "structures": 0},
                 {"name": "ring_a", "kind": "group", "family": "district", "relation": "concentric",
                  "count": 1, "structures": 0.3, "ring": 0, "share": 0.4, "walled": True},
                 {"name": "ring_b", "kind": "group", "family": "district", "relation": "concentric",
                  "count": 1, "structures": 0.6, "ring": 1, "share": 0.5, "walled": True},
                 {"name": "walls", "kind": "edge", "family": "wall", "relation": "concentric",
                  "count": 2, "structures": 0},
                 {"name": "gates", "kind": "point", "family": "gate", "relation": "gateway",
                  "count": 2, "structures": 0}]}
    try:
        spec_mod.read_spec(json.loads(json.dumps(first)), "Build a ringed town.")
        raise AssertionError("shares as counts were accepted")
    except spec_mod.SpecError as e:
        assert e.field == "structures" and e.part == "ring_a", (e.field, e.part)
    second = json.loads(json.dumps(first))
    for p in second["defining_parts"]:
        p["structures"] = 0
    second["defining_parts"][1]["share"] = 0.1                 # changed more than asked
    second["defining_parts"][2]["walled"] = False
    second["setting"] = {"surface": "green", "relief": "rolling"}
    merged = spec_mod.merge_hand_back(first, second, ["structures"])
    parts = {p["name"]: p for p in merged["defining_parts"]}
    assert parts["ring_a"]["structures"] == 0 and parts["ring_b"]["structures"] == 0
    assert parts["ring_a"]["share"] == 0.4 and parts["ring_b"]["walled"] is True
    assert merged["setting"]["relief"] == "flat"
    got = spec_mod.read_spec(merged, "Build a ringed town.")
    assert len(spec_mod.walled_rings(got)) == 2
    assert spec_mod.merge_hand_back(first, second, []) == second, "an unnamed refusal takes the answer"
    # the stage: a refused answer is set aside, the brief carries the field, the call is
    # asked again; the next answer is merged; a second refusal stops the round
    with tempfile.TemporaryDirectory() as tmp:
        rnd = pipeline.Round(name="hand_back_fixture", sentence="Build a ringed town.",
                             state_dir=tmp)
        be = pipeline.OfflineBackend(rnd)
        open(os.path.join(tmp, "place_spec_prompt.md"), "w").write("# brief\n")
        json.dump(first, open(os.path.join(tmp, "place.json"), "w"))
        res = stages_plan.stage_place_spec(rnd, be, {})
        assert res["status"] == "needs_model" and res["attempt"] == 2, res
        assert res["handed_back"]["check"] == "structures" and res["handed_back"]["part"] == "ring_a"
        assert os.path.exists(os.path.join(tmp, "place.rejected.1.json"))
        text = open(os.path.join(tmp, "place_spec_prompt.md")).read()
        assert "change **only** the field named" in text and "ring_a" in text
        json.dump(second, open(os.path.join(tmp, "place.json"), "w"))
        res2 = stages_plan.stage_place_spec(rnd, be, {})
        assert res2["status"] == "read", res2
        checked = json.load(open(os.path.join(tmp, "place.checked.json")))
        cp = {p["name"]: p for p in checked["defining_parts"]}
        assert cp["ring_a"]["share"] == 0.4 and cp["ring_b"]["walled"] is True
        assert checked["setting"]["relief"] == "flat"
        assert os.path.exists(os.path.join(tmp, "place.attempt.2.json"))
        # a refusal twice stops
        bad = json.loads(json.dumps(first))
        rnd2 = pipeline.Round(name="hand_back_twice", sentence="Build a ringed town.",
                              state_dir=os.path.join(tmp, "twice"))
        os.makedirs(rnd2.state)
        open(rnd2.rel("place_spec_prompt.md"), "w").write("# brief\n")
        json.dump(bad, open(rnd2.rel("place.json"), "w"))
        assert stages_plan.stage_place_spec(rnd2, be, {})["status"] == "needs_model"
        json.dump(bad, open(rnd2.rel("place.json"), "w"))
        stop = stages_plan.stage_place_spec(rnd2, be, {})
        assert stop.get("stop") and "twice" in stop["error"], stop
    return ("'structures' refused on ring_a: the second answer's structures are taken and "
            "its changed share, wall and relief are not; the stage sets the answer aside, "
            "names the field on the brief, merges the next answer and stops on a second refusal")


# ------------------------- what the second run's dry run found, before its write

@case
def t_9_the_plateau_record_carries_its_terrace_and_the_layout_stands_on_it():
    """The plateau is cut a step above the innermost ring off the site's median as it
    stood before any ground moved; the layout takes the same levels from the record and
    never reads a median again off the volume the cut has since changed."""
    from ethoslm import offline, placeplan, pipeline as pl
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40)
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        json.dump(json.loads(json.dumps(test_rings.THREE_RING)),
                  open(os.path.join(tmp, "place.json"), "w"))
        json.dump({**site, "mean_grid": [[70] * 12] * 12, "roughness_grid": [[3] * 12] * 12,
                   "stats": {"min": 50, "max": 90, "relief": 40}},
                  open(os.path.join(tmp, "site.json"), "w"))
        json.dump({"chosen": None, "terraform": None, "attempt": "as asked"},
                  open(os.path.join(tmp, "site_search.json"), "w"))
        rnd = pl.Round(name="one_median", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        be = pl.OfflineBackend(rnd, dry_run=True)
        med_before = placeplan.site_median(rnd.volume(), site)
        assert be.volume is not None                   # cached before the cut, as a run's is
        pl_rec = stages_plan.stage_plateau(rnd, be, {})
        assert pl_rec["plateau"]["ok"]
        rec = stages_plan.plateau_record(rnd)
        assert rec["terrace"] == pl_rec["terrace"] == placeplan.terrace_levels(med_before, 3)
        assert rec["y"] == rec["terrace"]["podium"] == med_before + 12
        # the backend forgot its cached volume when the cut was written under it
        med_after = placeplan.site_median(be.volume, site)
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(s, site, rec, decls,
                                                   "ochre_stone_green_tile", vol=be.volume)
        assert not fails, fails
        lay = place["layout"]
        assert lay["terrace"] == rec["terrace"], (lay["terrace"], rec["terrace"])
        assert lay["terrace"]["podium"] == rec["y"] == max(r["level"] for r in lay["rings"]) + 4
        # ...and the record read the old way -- part, rect, y and nothing else -- is
        # what the run did: a median off the cut volume, one block off where the fill
        # moved it, and a podium the measure refuses
        old = {k: rec[k] for k in ("part", "rect", "y")}
        place2, _f = placeplan.concentric_layout(s, site, old, decls,
                                                 "ochre_stone_green_tile", vol=be.volume)
        assert place2["layout"]["terrace"]["median"] == med_after
    return (f"median {med_before} before the cut, {med_after} off the cut volume: the "
            f"record carries rings {rec['terrace']['rings']} and the podium {rec['y']}, "
            f"and the layout stands on them")


@case
def t_9_a_designed_places_arterials_are_routed_on_the_designed_ground():
    """Where the layout designs the ground, the road between the gates is graded on the
    terraces and the podium it will stand on -- not on the hillside the search chose --
    so the road meets every ring at its level and climbs a step at each gate."""
    from ethoslm import observe, offline, placeplan, pipeline as pl
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    assert placeplan.TERRACE_REACH == 2
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40)
        rnd = pl.Round(name="designed_road", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        os.makedirs(rnd.state, exist_ok=True)
        _t, decls = placeplan.types_card(None, s.get("form"))
        med = placeplan.site_median(vol, site)
        plateau = {**test_rings._plateau(site), "y": med + 12,
                   "terrace": placeplan.terrace_levels(med, 3)}
        place, fails = placeplan.concentric_layout(s, site, plateau, decls,
                                                   "ochre_stone_green_tile", vol=vol)
        assert not fails, fails
        lay = place["layout"]
        h, _w = observe.ground_heights(vol)
        d = placeplan.designed_heights(h, vol.x0, vol.z0, lay, approaches=False)
        assert d.shape == h.shape and d is not h
        cx, cz = lay["centre"]
        gates = [p for p in place["parts"] if p["kind"] == "point"
                 and (decls.get(p["type"]) or {}).get("passage")]
        points = placeplan.approach_points(lay, gates)
        assert [p for p in points if p.get("virtual")], "the unwalled ring has a ramp"
        def outward(g):
            gx, gz = g["at"]
            if abs(gx - cx) >= abs(gz - cz):
                return (1 if gx > cx else -1), 0
            return 0, (1 if gz > cz else -1)
        def in_approach(x, z):
            for g in points:
                gx, gz = g["at"]
                ox, oz = outward(g)
                along = (x - gx) * ox + (z - gz) * oz
                across = abs((x - gx) * oz) + abs((z - gz) * ox)
                if -2 <= along <= 2 + 12 + 2 * 4 + 2 and across <= 4 + 1:
                    return True
            return False
        # every annulus at its level, the reach included, off the approaches; the
        # compound at the podium; outside the outermost ring the ground as found
        for r in lay["rings"]:
            a, b = r["inner"] + 3, r["outer"] + 2
            cheb = np.maximum(np.abs(np.arange(-b, b + 1))[:, None],
                              np.abs(np.arange(-b, b + 1))[None, :])
            win = d[cx - b - vol.x0:cx + b + 1 - vol.x0, cz - b - vol.z0:cz + b + 1 - vol.z0]
            mask = (cheb > a) & (cheb <= b)
            for i, j in zip(*np.nonzero(mask)):
                if in_approach(cx - b + i, cz - b + j):
                    mask[i, j] = False
            band = win[mask]
            assert band.min() == band.max() == r["level"], (r["name"], int(band.min()), int(band.max()))
        px0, pz0, px1, pz1 = lay["compound_rect"]
        pod = d[px0 - vol.x0:px1 - vol.x0 + 1, pz0 - vol.z0:pz1 - vol.z0 + 1]
        assert pod.min() == pod.max() == lay["terrace"]["podium"]
        edge = lay["rings"][-1]["outer"] + 2
        assert (d[:cx - edge - vol.x0, :] == h[:cx - edge - vol.x0, :]).all()
        # every gate has a level approach: the ground inside it carried outward past the
        # wall's reach for GATE_APPROACH columns, so the road climbs before the wall
        assert placeplan.GATE_APPROACH == 12 and placeplan.GATE_APPROACH_HALF == 4
        assert len(gates) >= 2, [p["name"] for p in gates]
        dg = placeplan.designed_heights(h, vol.x0, vol.z0, lay, gates=gates)
        approaches = 0
        def at(arr, g, k, side=0):
            gx, gz = g["at"]
            ox, oz = outward(g)
            return int(arr[gx + ox * k + oz * side - vol.x0, gz + oz * k + ox * side - vol.z0])
        assert placeplan.GATE_RAMP_RUN == 2
        for g in gates:
            inside = at(dg, g, -1)
            run = [at(dg, g, k) for k in range(0, 15)]
            assert run == [inside] * 15, (g["name"], run)
            ap = placeplan.gate_approach_pieces(d, vol.x0, vol.z0, lay, g)
            assert ap["inside"] == inside and ap["pieces"][0]["level"] == inside
            if g["ring"] < len(lay["rings"]) - 1:                    # a gate onto another ring
                below = lay["rings"][g["ring"] + 1]["level"]
                assert ap["foot"] == below and len(ap["pieces"]) == 1 + (inside - below - 1), \
                    (g["name"], ap["foot"], below, len(ap["pieces"]), inside)
                # the ramp: a block every two columns, down to the ring beyond
                ramp = [at(dg, g, k) for k in range(15, 15 + 2 * (inside - below))]
                want = [inside - 1 - (k // 2) for k in range(2 * (inside - below - 1))] + [below, below]
                assert ramp == want, (g["name"], ramp, want)
                assert at(d, g, 6) < inside, (g["name"], at(d, g, 6), inside)      # ...and was not, before
                assert at(dg, g, 6, side=4) == inside and at(dg, g, 6, side=5) < inside
            approaches += 1
        d = dg
        # the stage routes on it, says so, and keys the road by the terrace
        got = stages_plan._stage_arterials(rnd, place, decls, vol)
        assert got and got["ground"] == "as designed", got and got.get("ground")
        levels = {tuple(int(v) for v in k.split(",")): int(y) for k, y in got["levels"].items()}
        off = max(abs(y - int(d[x - vol.x0, z - vol.z0])) for (x, z), y in levels.items())
        assert off <= placeplan.TERRACE_STEP, off
        # ...and the road arrives at every gate level, for the wall's reach and beyond
        for g in gates:
            gx, gz = g["at"]
            ox, oz = outward(g)
            inside = at(dg, g, -1)
            at_gate = [levels.get((gx + ox * k + oz * side, gz + oz * k + ox * side))
                       for k in range(-1, 6) for side in (-1, 0, 1)]
            at_gate = [v for v in at_gate if v is not None]
            assert at_gate and set(at_gate) == {inside}, (g["name"], inside, at_gate)
        deep = 0
        for (x, z), y in levels.items():
            c = max(abs(x - cx), abs(z - cz))
            if in_approach(x, z):
                continue
            for r in lay["rings"]:
                if r["inner"] + 8 < c < r["outer"] - 8:
                    deep += 1
                    assert y == r["level"], ((x, z), y, r["name"], r["level"])
        assert deep > 0
        again = stages_plan._stage_arterials(rnd, place, decls, vol)
        assert again["of_plan"] == got["of_plan"] and again["cells"] == got["cells"]
        # the same road as found: graded on the hillside, off the designed ground by
        # more than a step somewhere -- the first run's road, through the terraces
        flat = json.loads(json.dumps(place))
        for r in flat["layout"]["rings"]:
            r.pop("level", None)
        for q in flat["parts"]:
            q.pop("level", None)
        rnd2 = pl.Round(name="found_road", sentence="Build a ringed town.",
                        state_dir=os.path.join(tmp, "found"), voice="ochre_stone_green_tile")
        os.makedirs(rnd2.state, exist_ok=True)
        found = stages_plan._stage_arterials(rnd2, flat, decls, vol)
        assert found["ground"] == "as found" and found["of_plan"] != got["of_plan"]
        fl = {tuple(int(v) for v in k.split(",")): int(y) for k, y in found["levels"].items()}
        off_found = max(abs(y - int(d[x - vol.x0, z - vol.z0])) for (x, z), y in fl.items())
        assert off_found > placeplan.TERRACE_STEP, off_found
    return (f"{len(got['cells'])} road columns on the designed ground, every one within "
            f"{off} of it and {deep} deep in a ring at its level; {approaches} gates each "
            f"with a level approach and the road level through it; the same road as found "
            f"stands {off_found} off the terraces")


@case
def t_9_the_offline_backend_reads_the_terraces_the_stage_laid():
    """A dry stage that writes the base volume back tells the backend, else the stage
    after it reads the ground as it stood before -- which is what the circulation of a
    run whose plan stage had cached the volume would have routed on."""
    from ethoslm import observe, offline, placeplan, pipeline as pl
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40)
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        json.dump(json.loads(json.dumps(test_rings.THREE_RING)),
                  open(os.path.join(tmp, "place.json"), "w"))
        json.dump({**site, "mean_grid": [[70] * 12] * 12, "roughness_grid": [[3] * 12] * 12,
                   "stats": {"min": 50, "max": 90, "relief": 40}},
                  open(os.path.join(tmp, "site.json"), "w"))
        json.dump({"chosen": None, "terraform": None, "attempt": "as asked"},
                  open(os.path.join(tmp, "site_search.json"), "w"))
        rnd = pl.Round(name="fresh_volume", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        be = pl.OfflineBackend(rnd, dry_run=True)
        stages_plan.stage_plateau(rnd, be, {})
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(
            s, site, stages_plan.plateau_record(rnd), decls, "ochre_stone_green_tile",
            vol=be.volume)
        assert not fails, fails
        json.dump(place, open(os.path.join(tmp, "plan.json"), "w"))
        lay = place["layout"]
        r = lay["rings"][-1]
        cx, cz = lay["centre"]
        # **a column the terrace will actually move.** The neighbourhood delivery round:
        # this took the midpoint of the belt, and on a relief-40 fixture that column is
        # 13 blocks off the ring's level, so the bounded terrace correctly leaves it
        # exactly as found and the case read 54 before and 54 after. The subject here is
        # the *backend's cache*, not the ground policy -- so the sample is a column the
        # design decided to move, found through the same mask the builder consults.
        from ethoslm import feasible as _feas
        reach = int(placeplan.DISTRICT_TERRACE_REACH)
        a_, b_ = r["inner"] + 8, r["outer"] - 4
        dec = _feas.record(be.volume, (cx - b_, cz - b_, cx + b_, cz + b_),
                           level=r["level"], relief=reach, fill=reach, window=0)
        mask = _feas.mask_of(dec)
        x = z = None
        if mask is not None:
            import numpy as _np
            cheb = _np.maximum(_np.abs(_np.arange(-b_, b_ + 1))[:, None],
                               _np.abs(_np.arange(-b_, b_ + 1))[None, :])
            here = mask & (cheb > a_) & (cheb <= b_)
            if here.any():
                i, j = (int(v[0]) for v in _np.nonzero(here))
                x, z = cx - b_ + i, cz - b_ + j
        assert x is not None, ("no column of this belt is inside the mask's own bound at "
                               "the ring's level", r["level"], reach)
        h0, _ = observe.ground_heights(be.volume)          # cached: the plan stage's read
        before = int(h0[x - be.volume.x0, z - be.volume.z0])
        got = stages_plan.stage_terraces(rnd, be, {})
        assert got.get("placed")
        h1, _ = observe.ground_heights(be.volume)
        after = int(h1[x - be.volume.x0, z - be.volume.z0])
        assert after == r["level"] != before, (before, after, r["level"])
        be.refresh()
        assert be.volume is not None
    return (f"the belt's ground read {before} off the backend's cache before the terraces "
            f"and {after}, its level, after: the stage refreshed it")


@case
def t_9_the_ground_measure_reads_the_inner_census_of_a_designed_place():
    """The measure reads the biome share the search held the site to: over the core and
    the inner rings where the record carries that census, the whole footprint reported
    beside it; over the whole where it does not."""
    from ethoslm import pipeline as pl
    from ethoslm.pipeline import stages_measure
    with tempfile.TemporaryDirectory() as tmp:
        rnd = pl.Round(name="inner_biome", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        os.makedirs(rnd.state, exist_ok=True)
        import test_rings
        doc = json.loads(json.dumps(test_rings.THREE_RING))
        doc["setting"]["biome"] = ["plains"]
        json.dump(doc, open(rnd.rel("place.json"), "w"))
        whole = {"read": True, "classes": {"plains": 36.6, "forest": 30.7}}
        json.dump({"chosen": {"origin": [0, 0], "size": 768, "measures": {"biome": {
            **whole, "inner": {"read": True, "classes": {"plains": 54.1, "forest": 20.0},
                               "window": 200}}}}},
                  open(rnd.rel("site_search.json"), "w"))
        be = pl.OfflineBackend(rnd, dry_run=True)
        m = stages_measure._m_ground(rnd, be, {}, {})
        b = m["biome"]
        assert b["share"] == 0.541 and b["holds"] and b["whole"] == 0.366, b
        assert "core and the inner rings" in b["over"] and "400" in b["over"], b["over"]
        json.dump({"chosen": {"origin": [0, 0], "size": 768, "measures": {"biome": whole}}},
                  open(rnd.rel("site_search.json"), "w"))
        m2 = stages_measure._m_ground(rnd, be, {}, {})
        assert m2["biome"]["share"] == 0.366 and not m2["biome"]["holds"]
        assert m2["biome"]["over"] == "the whole footprint" and m2["biome"]["whole"] is None
    return ("plains 0.541 over the core and inner rings holds with 0.366 over the whole "
            "reported beside it; the whole alone reads 0.366 and fails")


@case
def t_9_the_road_is_solved_with_the_lanes_as_one_surface():
    """The arterial record carries the level and the stair face the router solved for
    the road, for the record; the circulation puts every road column into the network
    at rank 0 before the surface is solved, so the road and the lanes are one field on
    the ground as it stands and a walk over the whole network reaches every cell."""
    from ethoslm import circulate, observe, placeplan, pipeline as pl
    from ethoslm.pipeline import stages_plan
    import test_rings
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    with tempfile.TemporaryDirectory() as tmp:
        vol = _volume(X, Z, S, base=50, relief=40)
        rnd = pl.Round(name="one_surface", sentence="Build a ringed town.",
                       state_dir=tmp, voice="ochre_stone_green_tile", flags={"dry_run": True})
        os.makedirs(rnd.state, exist_ok=True)
        _t, decls = placeplan.types_card(None, s.get("form"))
        med = placeplan.site_median(vol, site)
        plateau = {**test_rings._plateau(site), "y": med + 12,
                   "terrace": placeplan.terrace_levels(med, 3)}
        place, fails = placeplan.concentric_layout(s, site, plateau, decls,
                                                   "ochre_stone_green_tile", vol=vol)
        assert not fails, fails
        got = stages_plan._stage_arterials(rnd, place, decls, vol)
        assert got["faces"] and got["levels"], "the record carries the road's levels and faces"
        h, _w = observe.ground_heights(vol)
        gates = [p for p in place["parts"] if p["kind"] == "point"
                 and (decls.get(p["type"]) or {}).get("passage")]
        dg = placeplan.designed_heights(h, vol.x0, vol.z0, place["layout"], gates=gates)
        art = circulate.arterial_cells(got, h, vol.x0, vol.z0)
        assert len(art) == len(got["cells"])
        # the lanes: to every district, on the designed ground, with the road given and
        # the walls as obstacles crossed at their gates -- as the circulation stage does
        routing = circulate.parts_to_routing(
            [p for p in place["parts"] if p["kind"] in ("edge", "point")],
            passage={n for n, dd in decls.items() if dd.get("passage")})
        avoid = np.zeros(dg.shape, float)
        for (wx, wz) in routing["obstacles"]:
            i, j = wx - vol.x0, wz - vol.z0
            if 0 <= i < avoid.shape[0] and 0 <= j < avoid.shape[1]:
                avoid[i, j] = np.inf
        sites = [{"id": d["name"], "x0": min(d["x0"], d["x1"]), "z0": min(d["z0"], d["z1"]),
                  "x1": max(d["x0"], d["x1"]), "z1": max(d["z0"], d["z1"])}
                 for d in place["districts"]]
        band = placeplan.terrace_edge_band(
            dg.shape, vol.x0, vol.z0, place["layout"], gates=gates,
            designed=placeplan.designed_heights(h, vol.x0, vol.z0, place["layout"],
                                                gates=gates, approaches=False))
        assert placeplan.EDGE_BAND == 10 and placeplan.EDGE_BAND_COST == 40.0
        # the road keeps off the steep half of the band -- the face and the feather
        # under it -- but for the columns the widening puts there; the band's outer
        # columns are a preference and a road runs beside a wall at ten columns
        steep = 0
        cx, cz = place["layout"]["centre"]
        for (x, z) in art:
            c = max(abs(x - cx), abs(z - cz))
            for r in place["layout"]["rings"]:
                if r["outer"] < c <= r["outer"] + placeplan.TERRACE_REACH + placeplan.TERRACE_STEP \
                        and band[x - vol.x0, z - vol.z0] > 0:
                    steep += 1
        assert (band > 0).sum() > 0 and steep * 100 < len(art), (steep, len(art))
        net = circulate.plan_network(dg, vol.x0, vol.z0, sites, max_step=3, arterial=art,
                                     avoid_extra=np.maximum(avoid, band),
                                     passable=routing["passable"])
        on_road = [c for c in art if c in net.cells]
        assert len(on_road) == len(art), (len(on_road), len(art))
        assert all(net.cells[c]["rank"] == 0 for c in on_road)
        bare = [(c, n) for c, r in net.cells.items()
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                for n in [(c[0] + dx, c[1] + dz)] if n in net.cells
                and net.cells[n]["y"] - r["y"] == 1 and not net.cells[n]["face"]]
        assert not bare, bare[:6]
        wc = circulate.walk_check(net)
        assert wc["reached"] == wc["cells"], (wc["reached"], wc["cells"])
        assert net.notes.get("raised_max_step") is None, net.notes       # no cliff anywhere
        assert got["notes"].get("raised_max_step") is None, got["notes"]
    return (f"{len(art)} road columns in a network of {len(net.cells)} at rank 0, one "
            f"surface: no rise without a stair, {wc['reached']} of {wc['cells']} walked to, "
            f"max_step never raised")


@case
def t_9_a_lane_arrives_at_the_middle_of_a_face_and_never_at_a_corner():
    """On a levelled terrace every cell of a site's side is as smooth as the next, and
    the first cell of a side is its corner: the doorway reserved one step in from it
    was the site's corner, where a field stands its post -- 168 of 168 areas refused.
    The smoothest cell still wins; on a tie the middle of the side does."""
    from ethoslm import circulate
    from ethoslm.circulate import _approach_candidates
    flat = np.full((60, 60), 70, np.int32)
    got = _approach_candidates(flat, 0, 0, (10, 10, 20, 20))
    at = {c["face"]: (c["x"], c["z"]) for c in got}
    assert at == {"south": (15, 9), "north": (15, 21), "east": (9, 15), "west": (21, 15)}, at
    # ...and on a flat fixture no reserved doorway is a corner of its site
    sites = [{"id": "a", "x0": 10, "z0": 10, "x1": 20, "z1": 20},
             {"id": "b", "x0": 35, "z0": 12, "x1": 48, "z1": 30},
             {"id": "c", "x0": 14, "z0": 36, "x1": 30, "z1": 50}]
    net = circulate.plan_network(flat, 0, 0, sites, max_step=3)
    corners = {c for s in sites
               for c in ((s["x0"], s["z0"]), (s["x1"], s["z0"]), (s["x0"], s["z1"]), (s["x1"], s["z1"]))}
    doors = [(t.door[0], t.door[2]) for t in net.thresholds]
    assert doors and not any(d in corners for d in doors), doors
    return (f"flat: candidates at the middle of each side {sorted(at.values())}; "
            f"{len(doors)} doorways on a flat fixture, none a corner")


@case
def t_9_an_areas_door_is_where_the_circulation_reserved_it():
    """The way into a square or a field is the doorway the circulation pass reserved,
    on whichever side the lane came to -- not the rectangle's centre snapped to its
    nearest side. A field on a flat fixture with its lane on the east opens its
    border at the reserved doorway and E008 is silent."""
    from ethoslm import lint, offline, pipeline as pl
    from ethoslm.buildlib import Builder
    from ethoslm.circulate import Network, Threshold
    from ethoslm.pipeline import stages_build
    G = 64
    vol = _flat_volume(96, y=G)
    # a wide field, its lane along the east side, the doorway mid-side
    fx0, fz0, fx1, fz1 = 20, 30, 59, 49                    # 40 x 20: the centre is far from the east
    cells = {(61, z): {"y": G, "rank": 1, "face": None} for z in range(20, 60)}
    cells[(60, 40)] = {"y": G, "rank": 2, "face": None}
    net = Network(cells, [Threshold("field_a", 60, 40, G, "west", (59, G + 1, 40))])
    vol = vol.overlay({(x, G, z): "cobblestone" for (x, z) in cells})   # the lane, laid
    part = {"label": "field_a", "kind": "area", "x0": fx0, "z0": fz0, "x1": fx1, "z1": fz1,
            "seed": 3, "params": {"crop": "grain", "layout": "strips"}}
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    from ethoslm.frontage import Frontage
    b.frontage = Frontage(vol, net)
    mat = pl.voice_palette("ochre_stone_green_tile")
    sited = b.site(dict(part), mat=mat)
    assert sited["sited"]["ok"], sited["sited"]["reason"]
    door = b.frontage.sited["field_a"]["door"]          # what the type builder hands the type
    assert (int(door[0]), int(door[-1])) == (59, 40), door
    # ...and the field type, instantiated on it, opens its border there
    tf = os.path.join(ROOT, "types", "field.py")
    b2 = stages_build.instantiate(tf, {k: v for k, v in part.items() if k != "params"},
                                  vol, seed=3, network=net, params=part["params"],
                                  voice="ochre_stone_green_tile")
    v2 = vol.overlay(dict(b2._pending))
    ctx = lint.Context.build(v2, [{"label": "field_a", "x0": fx0, "z0": fz0, "x1": fx1,
                                   "z1": fz1, "kind": "area", "y0": G}], network=net)
    rep = lint.lint(ctx, only={"E008"})
    e008 = [f.message for f in rep.findings if f.code == "E008"]
    assert not e008, e008
    return (f"a 40x20 field with its lane on the east: sited door {door}, the "
            f"reserved doorway; the field opens its border there and E008 is silent")


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
