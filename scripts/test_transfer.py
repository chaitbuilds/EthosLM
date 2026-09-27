#!/usr/bin/env python3
"""The design path's small-place capabilities, self-contained.

    $PY scripts/test_transfer.py

Every case builds its own input -- synthetic terrain, a synthetic atlas, a flat lot --
and reads nothing from a run's state, so it runs the same in the public tree:

1. the atlas reads a section's biomes and the site scan holds a candidate to the
   setting's biomes, and says nothing about biomes where the atlas has none;
2. a grain's dwelling is any dwelling form in the library and nothing else;
3. a small settlement composes round its middle: lots on the lanes that follow the
   outline, never on water as found, the pool one basin that nothing else stands on,
   open ground laid only on open land, and the lanes shaded by the houses either side;
4. a changed request changes the decisions through the same contracts: another dwelling
   form fills the same grain and a form with no lane shade is asked for none;
5. the flat-roofed house delivers a flat roof, a terrace reached from inside or out and
   its shade, measured off the blocks -- and a pitched house does not read as flat;
6. the pool holds its water, a palm is its own timber and a garden bed is soil under a
   voice whose ground is dressed stone;
7. the design job is offered the forms the library has, discovered from `types/`;
8. the newcomer entry point says waiting, blocked or built from the controller's own
   record, and refuses a directory that is not a save before any work;
9. the designed ground takes its skin from the setting and its made ground from the
   design's palette.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from ethoslm import atlas, citydesign as CD, cityresolve as C  # noqa: E402

RESULTS = []


def case(fn):
    def run():
        try:
            why = fn()
            RESULTS.append((fn.__name__, True, why or ""))
        except AssertionError as e:
            RESULTS.append((fn.__name__, False, str(e)))
        except Exception as e:                     # noqa: BLE001 -- reported
            RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {e}"))
    run.__name__ = fn.__name__
    return run


# ------------------------------------------------------------------ 1. the atlas

class _Tag:
    def __init__(self, value):
        self.value = value


@case
def c1_biomes_are_read_and_held():
    # one section: a two-entry palette, bits 1, the first 32 cells desert, the rest
    # plains
    pal = [_Tag("minecraft:desert"), _Tag("minecraft:plains")]
    bits = [0] * 32 + [1] * 32
    word = sum(b << i for i, b in enumerate(bits))
    got = atlas._biome_codes({"palette": pal, "data": _Tag([word])})
    assert list(got[:32]) == [atlas.BIOME_CLASSES.index("desert")] * 32
    assert list(got[32:]) == [atlas.BIOME_CLASSES.index("plains")] * 32
    one = atlas._biome_codes({"palette": [_Tag("minecraft:river")]})
    assert set(one) == {0}, "a river is unclassed"
    # a synthetic coarse atlas: flat, dry, desert in the west half and plains in the
    # east
    n = 64
    co = {"step": 8, "x0": 0, "z0": 0,
          "median": np.full((n, n), 64, np.int16), "low": np.full((n, n), 64, np.int16),
          "high": np.full((n, n), 64, np.int16), "wet": np.zeros((n, n), np.float32),
          "biome": np.full((n, n), atlas.BIOME_CLASSES.index("plains"), np.uint8),
          "biomes": True}
    co["biome"][: n // 2] = atlas.BIOME_CLASSES.index("desert")
    held = atlas.scan_sites(64, co=co, stride=32, water=(0, 0.2), top=20,
                            biomes=["desert"])
    assert held and all(s["centre"][0] <= 8 * n // 2 + 8 for s in held), \
        "a candidate mostly off the desert was offered"
    assert all(s["setting_share"] >= 0.5 for s in held)
    free = atlas.scan_sites(64, co=dict(co, biomes=False), stride=32, water=(0, 0.2),
                            top=20, biomes=["desert"])
    assert any(s["centre"][0] > 8 * n // 2 + 8 for s in free), \
        "with no biome layer the setting cannot be held and is not"
    return f"{len(held)} desert candidates held, {len(free)} unheld without the layer"


# ------------------------------------------------------------------ 2-4. the compiler

VILLAGE = {
    "schema": "ethoslm.design/1", "id": "probe-village", "extent": {"radius": 64},
    "site": {"candidate": "given", "centre": [0, 0]},
    "boundary": {"shape": "circle"}, "axis": {"bearing": 180},
    "rings": [
        {"name": "green", "outer": 0.42, "grain": "open",
         "params": {"cover": ["grove", "garden"], "trees": "palm"},
         "ground": {"policy": "graded"}},
        {"name": "homes", "outer": 1.0, "grain": "clusters",
         "params": {"dwelling": "flat_house", "storeys": [1, 2], "lot_width": [6, 9],
                    "lot_depth": [8, 11], "lane_width": 3, "spokes": 4, "run": 4},
         "roads": {"inner": 3}, "shade": "lanes", "ground": {"policy": "graded"}},
    ],
    "landmarks": [{"name": "pool", "ring": "green", "bearing": 0, "at": 0.0,
                   "form": "pool", "size": [22, 18]}],
}


def _ground(W, wet_box=None):
    g = np.full((W, W), 64, np.int16)
    wet = np.zeros((W, W), bool)
    if wet_box is not None:
        a, b, c, d = wet_box
        wet[a:b, c:d] = True
    return g, g.copy(), wet


def _compile(design, wet_box=None):
    d = CD.read(copy.deepcopy(design))
    W = 2 * (d["extent"]["radius"] + C.MARGIN) + 1
    return C.compile_design(d, (0, 0), ground=_ground(W, wet_box))


@case
def c2_a_dwelling_is_any_dwelling_in_the_library():
    CD.read(copy.deepcopy(VILLAGE))
    for bad, why in (("ring_gate", "a gate"), ("worship", "a chapel"),
                     ("no_such_form", "a name with no file")):
        d = copy.deepcopy(VILLAGE)
        d["rings"][1]["params"]["dwelling"] = bad
        try:
            CD.read(d)
        except CD.DesignError:
            continue
        raise AssertionError(f"{why} ({bad}) was accepted as a dwelling")
    d = copy.deepcopy(VILLAGE)
    d["rings"][1]["shade"] = "awnings"
    try:
        CD.read(d)
        raise AssertionError("an unknown shade word was accepted")
    except CD.DesignError:
        pass
    return "flat_house accepted; a gate, a chapel and a missing file refused"


@case
def c3_a_small_settlement_round_its_middle():
    # a wet patch where the homes would stand, to be kept dry of lots
    city = _compile(VILLAGE, wet_box=(C.MARGIN + 100, C.MARGIN + 112,
                                      C.MARGIN + 60, C.MARGIN + 70))
    homes = [lf for lf in city.leaves if lf["type"] == "flat_house"]
    assert len(homes) >= 30, f"only {len(homes)} homes on a 64-radius ring"
    occ = np.zeros(city.use.shape, np.int16)
    for lf in homes:
        sl = (slice(lf["x0"] - city.X0, lf["x1"] - city.X0 + 1),
              slice(lf["z0"] - city.Z0, lf["z1"] - city.Z0 + 1))
        occ[sl] += 1
        assert not city.wet[sl].any(), f"{lf['name']} stands on water as found"
        assert lf["site"] or lf.get("front"), lf["name"]
    assert occ.max() == 1, "two lots overlap"
    pools = [lf for lf in city.leaves if lf["type"] == "pool"]
    assert len(pools) == 1, f"{len(pools)} pools"
    p = pools[0]
    psl = (slice(p["x0"] - city.X0, p["x1"] - city.X0 + 1),
           slice(p["z0"] - city.Z0, p["z1"] - city.Z0 + 1))
    assert (city.use[psl] == C.WATER).all(), "the pool's ground is not water"
    opens = [lf for lf in city.leaves if lf["kind"] == "area" and lf["type"] != "pool"]
    for lf in opens:
        sl = (slice(lf["x0"] - city.X0, lf["x1"] - city.X0 + 1),
              slice(lf["z0"] - city.Z0, lf["z1"] - city.Z0 + 1))
        assert not (city.use[sl] == C.WATER).any(), f"{lf['name']} is laid over the pool"
        if lf["ring"] == 0:
            assert (city.use[sl] == C.OPEN).all(), f"{lf['name']} leaves its open ground"
    kinds = {lf["type"] for lf in opens if lf["ring"] == 0}
    assert kinds == {"grove", "garden"}, f"the mixed cover laid {kinds}"
    palms = [lf for lf in opens if lf["type"] == "grove"]
    assert all(lf["params"].get("crown") == "palm" for lf in palms)
    shaded = [lf for lf in homes if lf["params"].get("shade") == "lane"]
    # a house fronting a crossing looks down a street too long to roof, and keeps its
    # sky
    assert len(shaded) >= 0.6 * len(homes), f"{len(shaded)}/{len(homes)} shade their lane"
    assert all(1 <= lf["params"]["reach"] <= 4 for lf in shaded)
    att = sum(1 for lf in homes if lf["attached"])
    assert att, "no house stands in a run"
    return (f"{len(homes)} homes ({att} in runs, {len(shaded)} shading their lane), one "
            f"pool, {len(opens)} open tiles, none on water")


@case
def c4_a_changed_request_changes_the_decisions():
    d = copy.deepcopy(VILLAGE)
    d["rings"][1]["params"].update(dwelling="cottage", lot_width=[9, 12],
                                   lot_depth=[10, 13])
    d["landmarks"] = []
    d["rings"][0]["params"] = {"cover": "garden"}
    city = _compile(d)
    homes = [lf for lf in city.leaves if lf["kind"] == "plot"]
    assert homes and {lf["type"] for lf in homes} == {"cottage"}
    assert not any("shade" in lf["params"] for lf in homes), \
        "a form with no lane shade was asked for one"
    assert not any(lf["type"] == "pool" for lf in city.leaves), "a pool nobody asked for"
    assert {lf["type"] for lf in city.leaves if lf["ring"] == 0} <= {"garden"}
    # and the accepted grains are untouched by the new one: a rows ring still packs rows
    d2 = copy.deepcopy(VILLAGE)
    d2["rings"][1] = {"name": "homes", "outer": 1.0, "grain": "rows",
                      "params": {"lane_dir": "east_west"}, "ground": {"policy": "terrace"}}
    d2["landmarks"] = []
    rows = {lf["type"] for lf in _compile(d2).leaves if lf["kind"] == "plot"}
    assert rows <= {"row_house", "shop_house"} and rows, rows
    return f"{len(homes)} cottages, unshaded; rows still {sorted(rows)}"


# ------------------------------------------------------------------ 5-6. the forms

def _probe(t, w, d, params, seed=1, attached=0):
    from ethoslm import construction, pipeline
    b, sited, res = construction.probe_build(t, w, d, params, seed=seed,
                                             voice="cut_sandstone_terraces",
                                             attached=attached, front="south")
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{t}.py"))
    out = construction.outcome(b, {**sited, "name": "probe", "kind": "plot",
                                   "params": params, "build": res}, decl, params)
    return b, sited, res, out


@case
def c5_a_flat_roof_is_measured_not_claimed():
    rows = []
    for w, d, prm, att in ((8, 14, {"storeys": 1, "court": "rear", "shade": "pergola"}, 0),
                           (7, 7, {"storeys": 1, "court": "none", "shade": "pergola"}, 0),
                           (10, 16, {"storeys": 2, "court": "rear", "shade": "lane",
                                     "reach": 3}, 1)):
        b, sited, res, out = _probe("flat_house", w, d, prm, attached=att)
        assert res.get("ok"), f"{w}x{d}: {res.get('reason')}"
        src = out["features_source"]
        for f in ("flat_roof", "roof_terrace", "canopy"):
            assert src.get(f) == "verified", f"{w}x{d} {prm}: {f} is {src.get(f)}"
        rows.append(f"{w}x{d}:{res['emitted']['roof']['way_up']}")
    # the negative: a pitched house's roof is not read as flat
    from ethoslm import construction
    b, sited, res = construction.probe_build("row_house", 6, 12, {"storeys": 1},
                                             voice="cut_sandstone_terraces")
    cols = construction._window(b, sited)
    fp = sited["footprint"]
    eave = min(y for c, ys in cols.items() for y, n in ys.items()
               if fp[0] < c[0] < fp[2] and fp[1] < c[1] < fp[3] and "stairs" in n)
    assert not construction._verify_flat(cols, fp, eave), "a pitched roof read as flat"
    return "flat roof, terrace and shade verified on " + ", ".join(rows)


@case
def c6_water_palms_and_soil():
    from ethoslm import construction, offline, pipeline
    from ethoslm.buildlib import Builder

    def area(t, w, d, params):
        ns = construction._type_ns(t, None)
        vol = construction._flat_volume(96)
        b = Builder(offline.OfflineSite(vol))
        b._vol, b.frontage = vol, None
        x0 = z0 = (96 - max(w, d)) // 2
        lot = {"label": "probe", "x0": x0, "z0": z0, "x1": x0 + w - 1, "z1": z0 + d - 1}
        b.registry = construction._OnePlot(dict(lot))
        v = "cut_sandstone_terraces"
        sited = b.site({**lot, "kind": "area"}, mat=pipeline.voice_palette(v),
                       roof=pipeline.voice_roof(v))
        res = ns["build"](b.type_builder(sited, role=ns.get("ROLE")), sited, 1, **params)
        return b, sited, res

    b, sited, res = area("pool", 20, 16, {"shape": "round"})
    water = {k for k, v in b._pending.items() if v == "water"}
    assert res["ok"] and water
    for (x, y, z) in water:
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = b._pending.get((x + dx, y, z + dz))
            assert n is not None and n != "air", f"water at {(x, y, z)} is open to the side"
        assert b._pending.get((x, y - 1, z)) not in (None, "air") or (x, y - 1, z) in water
    b, sited, res = area("grove", 16, 16, {"planting": "copse", "crown": "palm"})
    logs = {v.split("[")[0] for v in b._pending.values() if "_log" in v}
    assert res["trees"] >= 2 and logs == {"stripped_jungle_log"}, (res, logs)
    b, sited, res = area("garden", 14, 12, {"layout": "beds", "edge": "hedge"})
    fy = sited["floor_y"]
    beds = [v for (x, y, z), v in b._pending.items() if y == fy and "grass_block" in v]
    assert beds, "a garden under dressed stone planted its beds in stone"
    return f"{len(water)} water cells held, {res['flowers']} flowers in soil, palms"


# ------------------------------------------------------------------ 7. the forms card

@case
def c7_the_design_is_offered_the_library():
    from ethoslm.pipeline import stages_design as SD
    card = SD._forms_card()
    names = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "types"))
                   if f.endswith(".py") and not f.startswith("_"))
    missing = [n for n in names if f"`{n}`" not in card]
    assert not missing, f"not offered: {missing}"
    dwell = card.split("**Other buildings**")[0]
    assert "`flat_house`" in dwell and "`cottage`" in dwell and "`wall`" not in dwell
    return f"{len(names)} forms offered, dwellings apart"


# ------------------------------------------------------------------ 8. the entry point

@case
def c8_the_entry_point_says_where_a_place_stands():
    from ethoslm import cli
    with tempfile.TemporaryDirectory() as tmp:
        old = cli._state
        cli._state = lambda name: os.path.join(tmp, name)
        try:
            st = os.path.join(tmp, "p")
            os.makedirs(st)
            assert cli.state_of("p")["state"] == "new"
            rec = {"results": {"pending": {"stage": "design", "waiting": [
                {"role": "design", "request": "/x/design_prompt.md",
                 "write": "/x/design.proposals.json"}]}}}
            json.dump(rec, open(os.path.join(st, "round.json"), "w"))
            s = cli.state_of("p")
            assert s["state"] == "waiting" and s["jobs"][0]["write"].endswith(
                "proposals.json") and "resume p" in s["next"]
            rec = {"results": {"stopped": {"stage": "design_resolve", "why": "x"}}}
            json.dump(rec, open(os.path.join(st, "round.json"), "w"))
            assert cli.state_of("p")["state"] == "blocked"
            rec = {"results": {"regions": {"built": ["0_0"], "kept": [], "failed": []},
                               "region_views": {"views": ["a.png"]}}}
            json.dump(rec, open(os.path.join(st, "round.json"), "w"))
            s = cli.state_of("p")
            assert s["state"] == "designed" and "deliver" in s["next"]
            open(os.path.join(st, ".running"), "w").write(f"{os.getpid()} now\n")
            assert cli.state_of("p")["state"] == "working"
        finally:
            cli._state = old
        bad = cli.doctor(tmp, quiet=True)
        assert any("level.dat" in b for b in bad), bad
    return "new, waiting, blocked, designed and working read off the record"


# ------------------------------------------------------------------ 9. the ground

@case
def c9_the_ground_is_the_settings_and_the_palettes():
    from ethoslm.pipeline import stages_design as SD

    class R:
        flags = {"design": {}}

        def __init__(self, tmp, surface, ground=None):
            self.tmp = tmp
            json.dump({"setting": {"surface": surface}},
                      open(os.path.join(tmp, "place.json"), "w"))
            json.dump({"palette": {"ground": ground} if ground else {}},
                      open(os.path.join(tmp, "design.json"), "w"))

        def rel(self, *p):
            return os.path.join(self.tmp, *p)

        def place_spec(self):
            return json.load(open(self.rel("place.json")))

    with tempfile.TemporaryDirectory() as tmp:
        green = SD.ground_materials(R(tmp, "green"), {})
        assert green["cover"] == "grass_block" and green["retain"] == "stone_bricks"
        sand = SD.ground_materials(R(tmp, "sand", "cut_sandstone_terraces"), {})
        assert sand["cover"] == "sand" and sand["retain"] == "cut_sandstone" \
            and sand["pave"][2] == "smooth_sandstone", sand
    return "grass on a plain as before; sand and cut sandstone in a desert of sandstone"


def main() -> int:
    for fn in (c1_biomes_are_read_and_held, c2_a_dwelling_is_any_dwelling_in_the_library,
               c3_a_small_settlement_round_its_middle,
               c4_a_changed_request_changes_the_decisions,
               c5_a_flat_roof_is_measured_not_claimed, c6_water_palms_and_soil,
               c7_the_design_is_offered_the_library,
               c8_the_entry_point_says_where_a_place_stands,
               c9_the_ground_is_the_settings_and_the_palettes):
        fn()
    for name, ok, why in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {why}" if why else ""))
    n = sum(ok for _, ok, _ in RESULTS)
    print(f"{n}/{len(RESULTS)}")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
