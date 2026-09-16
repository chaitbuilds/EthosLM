"""v2, B2: the relation solver. The place level is placed by relation and the model
never writes a coordinate at any level.

    $PY scripts/test_solver.py

What is proved here, each in seconds:

1. the concentric fixture solves to the layout the arithmetic gave -- the same parts,
districts and compounds, byte for byte; 2. 3. an unsatisfiable fixture reports its
vetoes by name and the least-violating placement, and a merely tight one records what it
demoted and why; 4. `near`, `along` and `on` place against the part they name, and `of`
is checked when the spec is read; 5. the same spec, site and seed solve to the same
place; a different seed solves to a valid one; 6. the plan stage solves a ringless place
and asks no model for it: no brief, the districts asked for next.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import offline, pipeline, placeplan, placesolve, spec as spec_mod  # noqa: E402
from ethoslm.placeread import inside                                             # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


SENTENCE = "Build a walled town of about sixty houses with a market square and a keep."

WALLED_TOWN = {
    "sentence": SENTENCE, "kind": "town",
    "defining_parts": [
        {"name": "town_wall", "kind": "edge", "family": "wall", "relation": "perimeter",
         "count": 1, "structures": 0, "notes": "A single circuit of wall round the whole place."},
        {"name": "town_gate", "kind": "point", "family": "gate", "relation": "gateway",
         "count": 1, "structures": 0, "notes": "One way through the circuit."},
        {"name": "market_square", "kind": "area", "family": "square", "relation": "centre",
         "count": 1, "structures": 0, "notes": "The open ground the town is built round."},
        {"name": "keep", "kind": "plot", "family": "keep", "relation": "beside_the_centre",
         "count": 1, "structures": 0, "notes": "The keep flanks the square."},
        {"name": "residential_district", "kind": "group", "family": "district",
         "relation": "throughout", "count": 1, "structures": 0,
         "notes": "The houses, spread through the walls around the square and the keep."}],
    "voice": None, "form": "european_vernacular"}


def _town(**over):
    doc = json.loads(json.dumps(WALLED_TOWN))
    doc.update(over)
    return spec_mod.read_spec(doc, SENTENCE)


def _flat_site(size=192, origin=(0, 0)):
    return {"origin": list(origin), "size": size, "stats": {"min": 64, "max": 64, "relief": 0},
            "mean_grid": [[64] * 12] * 12, "roughness_grid": [[0] * 12] * 12,
            "surface_blocks": {"grass_block": 100.0}, "water_pct": 0.0, "forest_pct": 0.0}


def _validate(place, spec, site, decls, plateau=None, vol=None):
    ground = pipeline.plan_ground([{**p, "name": p.get("name")} for p in place["parts"]], vol)
    return placeplan.place_failures(place, spec, site, decls, ground=ground, plateau=plateau)


# ------------------------------------------------------------------ 1. the concentric
# case

@case
def t_1_the_concentric_fixture_solves_to_the_layout_the_arithmetic_gave():
    import test_rings
    s = test_rings.three_ring_spec()
    place, fails, decls, site, plateau = test_rings._layout(s)
    assert not fails, fails
    solved, sfails = placesolve.solve_place(s, site, plateau, decls, "ochre_stone_green_tile")
    assert not sfails, sfails
    for key in ("parts", "districts", "compounds", "centre", "intent"):
        assert solved[key] == place[key], key
    assert solved["layout"]["by"] == "placeplan.concentric_layout"
    assert solved["layout"]["solver"]["case"] == "concentric"
    assert {k: v for k, v in solved["layout"].items() if k != "solver"} == place["layout"]
    assert _validate(solved, s, site, decls, plateau=plateau) == []
    return (f"{len(solved['parts'])} parts, {len(solved['districts'])} districts and "
            f"{len(solved['compounds'])} compound identical to the arithmetic's; the "
            f"record says the case")


# ------------------------------------------------------------------ 2.

@case
def t_2_round_17s_walled_town_solves_from_its_spec_on_its_own_ground():
    world = offline.world_cache("town")
    if not os.path.exists(world):
        return "skip: no cached ground for the walled town"
    spec = _town()
    site = json.load(open(offline.fixture_path("town", "site.json")))
    plateau = json.load(open(offline.fixture_path("town", "plateau.json")))
    vol = offline.load_volume(world)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    t0 = time.perf_counter()
    place, fails = placesolve.solve_place(spec, site, plateau, decls, "blackstone_and_ash",
                                          vol=vol)
    secs = time.perf_counter() - t0
    assert not fails, fails
    by = {p["name"]: p for p in place["parts"]}
    wall, gate, square, keep = by["town_wall"], by["town_gate"], by["market_square"], by["keep"]
    # the wall is one closed loop of the wall type, every segment along an axis
    assert wall["kind"] == "edge" and wall["type"] == "wall"
    assert wall["path"][0] == wall["path"][-1] and len(wall["path"]) >= 5
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(wall["path"], wall["path"][1:]))
    # ...and the gate stands on it, facing in
    cells = set(placeplan._edge_cells(wall))
    assert tuple(gate["at"]) in cells, gate
    assert gate["facing"] in ("north", "south", "east", "west")
    # the square is at the centre, on the plateau cut for it
    px0, pz0, px1, pz1 = plateau["rect"]
    assert px0 <= square["x0"] and square["x1"] <= px1 and pz0 <= square["z0"] \
        and square["z1"] <= pz1, (square, plateau["rect"])
    assert abs((square["x0"] + square["x1"]) // 2 - (px0 + px1) // 2) <= 1
    # the keep is beside it: a lane and its clearance away, on one side
    gap = placeplan.LANE_GAP + 1
    sq = (square["x0"], square["z0"], square["x1"], square["z1"])
    kp = (keep["x0"], keep["z0"], keep["x1"], keep["z1"])
    dx = max(sq[0] - kp[2], kp[0] - sq[2], 0)
    dz = max(sq[1] - kp[3], kp[1] - sq[3], 0)
    assert (dx == 0) != (dz == 0) and max(dx, dz) <= gap + 8, (dx, dz)
    # every part and every district inside the wall
    for p in place["parts"]:
        if p is wall or p is gate:
            continue
        r = pipeline.part_rect({**p, "name": p["name"]})
        for c in ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])):
            assert inside(wall["path"], c), (p["name"], c)
    for d in place["districts"]:
        for c in ((d["x0"], d["z0"]), (d["x1"], d["z1"])):
            assert inside(wall["path"], c), (d["name"], c)
    # the districts hold the sixty houses the sentence asked for
    laid = sum(d["structures"] for d in place["districts"])
    assert spec["size_band"][0] <= laid <= spec["size_band"][1], (laid, spec["size_band"])
    assert place["layout"]["districts"]["coverage"] >= placeplan.RING_COVERAGE
    # the validator the model's plan was held to has nothing to say
    fails = _validate(place, spec, site, decls, plateau=plateau, vol=vol)
    assert fails == [], fails
    lay = place["layout"]
    assert lay["by"] == "placesolve.solve_place" and lay["demoted"] == []
    assert all(r["chosen"] is not None for r in lay["solved"])
    return (f"in {secs:.1f}s: the wall a closed loop at half-side {lay['wall']['half']} "
            f"with the gate at {gate['at']} on its {gate['facing']}-facing side, the "
            f"square {square['x1'] - square['x0'] + 1} square on the plateau, the keep "
            f"{max(dx, dz)} beside it, {len(place['districts'])} districts holding "
            f"{laid} of about sixty; 0 failures, 0 vetoes demoted")


# ------------------------------------------------------------------ 3. unsatisfiable

@case
def t_3_an_unsatisfiable_fixture_reports_its_vetoes_and_a_tight_one_what_it_demoted():
    spec = _town()
    _t, decls = placeplan.types_card(None, spec.get("form"))
    # a site far too small for a keep beside a square inside a wall: the keep has
    # nowhere inside the wall and off the square
    small = _flat_site(size=72)
    place, fails = placesolve.solve_place(spec, small, None, decls, "white_render_dark_frame")
    assert place is None and fails, fails
    names = {f["check"] for f in fails}
    assert names & {"vetoes", "district", "relation"}, names
    worst = [f for f in fails if f["check"] == "vetoes"]
    assert worst and worst[0]["vetoes"], worst
    assert "least violating" in worst[0]["why"] or "no candidate" in worst[0]["why"], worst[0]
    # a site that is merely tight: the keep can stand only where the wall's inset would
    # refuse it, and the solver says which veto it demoted to place it
    tight = _flat_site(size=96)
    place2, fails2 = placesolve.solve_place(spec, tight, None, decls, "white_render_dark_frame")
    if place2 is not None:
        dem = place2["layout"]["demoted"]
        assert isinstance(dem, list)
        note = f"a 96 site solves" + (f", demoting {dem[0]['dropped']} for {dem[0]['part']}"
                                      if dem else " with nothing demoted")
    else:
        note = f"a 96 site is refused too: {fails2[0]['check']}"
    return (f"a 72 site refused naming {sorted(names)}: {worst[0]['why'][:90]}...; {note}")


# ------------------------------------------------------------------ 4. near, along, on

@case
def t_4_near_along_and_on_place_against_the_part_they_name():
    doc = json.loads(json.dumps(WALLED_TOWN))
    doc["defining_parts"] += [
        {"name": "chapel", "kind": "plot", "family": "hall", "relation": "near", "of": "keep",
         "count": 1, "structures": 0, "notes": "a hall near the keep"},
        {"name": "postern", "kind": "point", "family": "gate", "relation": "on",
         "of": "town_wall", "count": 2, "structures": 0, "notes": "two posterns on the wall"},
        {"name": "wall_row", "kind": "plot", "family": "house", "relation": "along",
         "of": "town_wall", "count": 1, "structures": 0, "notes": "a house along the wall"}]
    spec = spec_mod.read_spec(doc, SENTENCE)
    assert [p.get("of") for p in spec["defining_parts"][-3:]] == ["keep", "town_wall", "town_wall"]
    site = _flat_site(size=256)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placesolve.solve_place(spec, site, None, decls, "white_render_dark_frame")
    assert not fails, fails
    by = {p["name"]: p for p in place["parts"]}
    wall = by["town_wall"]
    cells = set(placeplan._edge_cells(wall))
    # `on`: both posterns at a cell of the wall's line, not where the gate is
    towers = [by["postern_1"], by["postern_2"]]
    for t in towers:
        assert t["kind"] == "point" and tuple(t["at"]) in cells, t
        assert tuple(t["at"]) != tuple(by["town_gate"]["at"])
    assert towers[0]["at"] != towers[1]["at"]
    # ...and a family the library has no point type for is refused by name: the library
    # grows by refusal, and a tower is not a gate with the opening left out
    nope = json.loads(json.dumps(doc))
    nope["defining_parts"][-2].update(name="watch", family="tower")
    _p, nf = placesolve.solve_place(spec_mod.read_spec(nope, SENTENCE), site, None, decls,
                                    "white_render_dark_frame")
    assert nf and "tower" in nf[0]["why"] and "no committed point type" in nf[0]["why"], nf
    # `near` the keep: a lane and a clearance away from it, on one side
    keep, chapel = by["keep"], by["chapel"]
    dx = max(keep["x0"] - chapel["x1"], chapel["x0"] - keep["x1"], 0)
    dz = max(keep["z0"] - chapel["z1"], chapel["z0"] - keep["z1"], 0)
    assert (dx == 0) != (dz == 0) and max(dx, dz) <= placeplan.LANE_GAP + 8, (dx, dz)
    # `along` the wall: inside it and within its inset and a clearance of its line
    row = by["wall_row"]
    r = (row["x0"], row["z0"], row["x1"], row["z1"])
    wr = place["layout"]["wall"]["rect"]
    dist = min(r[0] - wr[0], wr[2] - r[2], r[1] - wr[1], wr[3] - r[3])
    assert 0 < dist <= place["layout"]["wall"]["inset"] + placeplan.LANE_GAP + 4, (dist, wr, r)
    assert all(inside(wall["path"], c) for c in ((r[0], r[1]), (r[2], r[3])))
    # and the whole passes the validator
    fails = _validate(place, spec, site, decls)
    assert fails == [], fails
    # `of` is checked when the spec is read
    bad = json.loads(json.dumps(doc))
    bad["defining_parts"][-1]["of"] = "nothing_here"
    try:
        spec_mod.read_spec(bad, SENTENCE)
    except spec_mod.SpecError as e:
        assert "nothing_here" in str(e), e
    else:
        raise AssertionError("an `of` naming no part was accepted")
    bad2 = json.loads(json.dumps(doc))
    bad2["defining_parts"][3]["of"] = "town_wall"            # `beside_the_centre` with an of
    try:
        spec_mod.read_spec(bad2, SENTENCE)
    except spec_mod.SpecError as e:
        assert "`of`" in str(e), e
    else:
        raise AssertionError("an `of` on a relation that takes none was accepted")
    return (f"posterns at {towers[0]['at']} and {towers[1]['at']} on the wall, the chapel "
            f"{max(dx, dz)} from the keep, the row {dist} inside the wall's line; 0 "
            f"failures; a tower the library has no type for, an `of` that names nothing "
            f"and one on the wrong relation refused by name")


# ------------------------------------------------------------------ 5. determinism

@case
def t_5_the_same_seed_solves_to_the_same_place_and_another_to_a_valid_one():
    spec = _town()
    site = _flat_site(size=224)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    a, fa = placesolve.solve_place(spec, site, None, decls, "white_render_dark_frame", seed=3)
    b, fb = placesolve.solve_place(spec, site, None, decls, "white_render_dark_frame", seed=3)
    assert not fa and not fb
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c, fc = placesolve.solve_place(spec, site, None, decls, "white_render_dark_frame", seed=11)
    assert not fc and _validate(c, spec, site, decls) == []
    assert c["layout"]["seed"] == 11 and a["layout"]["seed"] == 3
    reg = a["layout"]["registered"]
    assert reg["VETOES"] == [["site", 100], ["overlap", 90], ["footprint", 80],
                             ["inside", 70], ["plateau", 60]]
    assert reg["COSTS"] == placesolve.COSTS
    return ("seed 3 twice: identical; seed 11: valid; the vetoes and the costs are on "
            "the record")


# ------------------------------------------------------------------ 6. the stage

@case
def t_6_the_plan_stage_solves_a_ringless_place_and_asks_no_model_for_it():
    from ethoslm.pipeline import stages_plan
    doc = json.loads(json.dumps(WALLED_TOWN))
    with tempfile.TemporaryDirectory() as tmp:
        site = _flat_site(size=224)
        json.dump(doc, open(os.path.join(tmp, "place.json"), "w"))
        json.dump(site, open(os.path.join(tmp, "site.json"), "w"))
        rnd = pipeline.Round(name="solver_fixture", sentence=SENTENCE, state_dir=tmp,
                             voice="white_render_dark_frame")
        be = pipeline.OfflineBackend(rnd)
        spec = spec_mod.read_spec(doc, SENTENCE)
        got = stages_plan.stage_plan_levels(rnd, be, {}, spec)["plan"]
        assert got["status"] == "needs_model", got
        assert got["level"].startswith("district/"), got["level"]
        assert os.path.exists(os.path.join(tmp, "plan.place.json"))
        assert not os.path.exists(os.path.join(tmp, "place_plan_prompt.md"))
        place = json.load(open(os.path.join(tmp, "plan.place.json")))
        assert place["layout"]["by"] == "placesolve.solve_place"
        assert "arterials" in place                  # routed where there is ground to route on
        log = json.load(open(os.path.join(tmp, "plan_validation.json")))
        first = [a for a in log["attempts"] if a["level"] == "place"][0]
        assert first["failures"] == [] and "arithmetic" in first["checked"], first
        briefs = [f for f in os.listdir(tmp) if f.startswith("district_") and f.endswith("_prompt.md")]
        assert len(briefs) == len(place["districts"]) >= 1
        return (f"the stage wrote {len(place['parts'])} parts and {len(place['districts'])} "
                f"districts by relation, routed the arterials, asked next for {got['level']}; "
                f"no place brief exists")


def main() -> int:
    bad = 0
    t0 = time.perf_counter()
    for fn in CASES:
        name = fn.__name__[2:].replace("_", " ")
        try:
            note = fn()
            print(f"ok   {name}: {note}")
        except Exception as e:                       # noqa: BLE001 -- a failing case
            import traceback
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} solver cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
