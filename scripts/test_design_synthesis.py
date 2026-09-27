#!/usr/bin/env python3
"""Focused checks of the design synthesis contracts, with controls.

    $PY scripts/test_design_synthesis.py

Probes of the production compiler (`ethoslm.cityresolve`) on synthetic ground, each a
case that would break one contract or a positive control:

1. a round request's walls read round: the wall path hugs the circle and every segment
   is axial or 45 degrees; the ring land ends at the wall (no unowned corner);
2. a non-round request does not inherit circles: a square outline's wall path is four
   axial runs and its land fills the corners; a ring with no wall lays no wall;
3. a terrain-respecting small settlement stays possible: a preserved grain on a slope
   moves almost no ground and keeps its pads at their own ground;
4. a street does not run into a wall with no gate: the radial stops at the first ungated
   wall outward, and a gated one passes;
5. every lot fits its form's measured footprint and every compiled door lands on a street;
6. a changed region does not discard its neighbours: changing one region's leaf changes
   that region's build inputs only;
7. a landmark that cannot stand where the design put it is a finding, not a silent move;
8. the monument's axial sequence is laid on its axis with the principal hall's storeys
   carried to the leaf.
"""
from __future__ import annotations

import copy
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from ethoslm import boundary as B, cityresolve as C  # noqa: E402

BASE = {
    "schema": "ethoslm.design/1", "id": "probe", "extent": {"radius": 260},
    "boundary": {"shape": "circle"}, "axis": {"bearing": 180},
    "gates": [{"bearing": 180, "rings": "all"}, {"bearing": 90, "rings": ["outer"]}],
    "rings": [
        {"name": "core", "outer": 0.3, "role": "monument", "wall": {"height": 10, "width": 3},
         "ground": {"policy": "podium", "rise": 3}},
        {"name": "inner", "outer": 0.6, "grain": "courts", "params": {"lane_dir": "east_west"},
         "wall": {"height": 12, "width": 3}, "ground": {"policy": "terrace"},
         "roads": {"outer": 6}},
        {"name": "outer", "outer": 1.0, "grain": "rows", "params": {"lot_width": [5, 6]},
         "wall": {"height": 14, "width": 3}, "ground": {"policy": "terrace"},
         "roads": {"outer": 5}},
    ],
    "radials": [{"bearing": 180, "width": 9}, {"bearing": 90, "width": 7}],
    "monument": {"name": "palace", "ring": "core", "enter": 180, "sequence": [
        {"kind": "forecourt", "depth": 12},
        {"kind": "gate_hall", "width": 21, "depth": 11, "storeys": 1},
        {"kind": "court", "depth": 20},
        {"kind": "hall", "role": "principal", "width": 33, "depth": 21, "storeys": 3,
         "eaves": 2, "platform": 3, "roof": "hip"},
        {"kind": "garden", "depth": 14}]},
}

RESULTS = []


def case(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, True, ""))
        except AssertionError as e:
            RESULTS.append((fn.__name__, False, str(e)))
    return run


def _ground(W, slope=0.0):
    x = np.arange(W)[:, None] * np.ones((1, W))
    g = (64 + slope * x).astype(np.int16)
    return (g, g.copy(), np.zeros((W, W), bool))


def compile_(d, slope=0.0):
    R = d["extent"]["radius"]
    W = 2 * (R + C.MARGIN) + 1
    return C.compile_design(d, (0, 0), ground=_ground(W, slope))


@case
def c1_round_walls_hug_the_circle():
    city = compile_(copy.deepcopy(BASE))
    w = [w for w in city.walls if w["name"] == "wall_outer"][0]
    assert not B.check_polyline(w["path"]), "a wall segment off the lattice's axes"
    cells = B.polyline_cells(w["path"])
    dev = [abs(math.hypot(x + .5 - city.outline.centre[0], z + .5 - city.outline.centre[1])
               - (city.R - 2)) for x, z in cells]
    assert max(dev) < 8, f"wall strays {max(dev):.1f} from its circle"
    # no land outside the boundary, none unowned inside it
    inside = city.ring >= 0
    spill = ~inside & (city.use != C.OUT)
    assert (city.use[spill] == C.WALL).all(), "something but a wall stands outside"
    assert (city.ratio[spill] < 1 + 3.0 / city.R).all(), "a wall spills over 3 blocks out"
    assert not ((city.use == C.LAND) & inside).any(), "land left unowned inside the rings"


@case
def c2_a_square_request_is_square():
    d = copy.deepcopy(BASE)
    d["boundary"] = {"shape": "polygon", "points": [[1, 1], [-1, 1], [-1, -1], [1, -1]]}
    d["rings"][1]["wall"] = None
    city = compile_(d)
    names = [w["name"] for w in city.walls]
    assert "wall_inner" not in names, "a ring with no wall laid one"
    w = [w for w in city.walls if w["name"] == "wall_outer"][0]
    diag = [(a, b) for a, b in zip(w["path"], w["path"][1:])
            if a[0] != b[0] and a[1] != b[1] and abs(b[0] - a[0]) > 1]
    assert not diag, f"a square outline's wall has {len(diag)} diagonal run(s): {diag[:3]}"
    straight = sum(max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in
                   zip(w["path"], w["path"][1:]) if a[0] == b[0] or a[1] == b[1])
    assert straight > 0.97 * 8 * 258, "a square's wall is not four straight sides"
    corner = city.ring[city.ix(-240), city.iz(-240)]
    assert corner >= 0, "the square's corner is not owned by a ring"


@case
def c3_a_small_settlement_keeps_its_ground():
    d = {"schema": "ethoslm.design/1", "id": "hamlet", "extent": {"radius": 110},
         "boundary": {"shape": "ellipse", "aspect": 0.7},
         "rings": [{"name": "village", "outer": 1.0, "grain": "fields",
                    "params": {"street_width": 3}, "ground": {"policy": "preserve"}}],
         "radials": [], "gates": []}
    city = compile_(d, slope=0.25)
    assert not city.walls, "an unwalled request laid a wall"
    found = city.ground[0].astype(int)
    t = city.target.astype(int)
    ch = (t != C.NO_TARGET) & (city.ring >= 0)
    moved = int(np.abs(found - t)[ch].sum())
    area = int((city.ring >= 0).sum())
    assert moved < 0.5 * area, f"a preserved hamlet moved {moved} blocks over {area} columns"
    keep = ((city.treat == C.KEEP) & (city.ring >= 0)).sum() / area
    assert keep > 0.6, f"only {keep:.0%} of a preserved hamlet kept as found"


@case
def c4_no_street_into_an_ungated_wall():
    city = compile_(copy.deepcopy(BASE))
    # the east radial: the outer wall is gated at 90, the inner wall is not
    x_in = city.ix(int(0.45 * 260))
    assert city.use[x_in, city.iz(0)] != C.ROAD or city.rank[x_in, city.iz(0)] < 3, \
        "the east radial runs inside a wall with no east gate"
    x_out = city.ix(int(0.8 * 260))
    assert city.rank[x_out, city.iz(0)] == 3, "the east radial is missing outside"
    assert not [f for f in city.findings if "crosses a wall" in f["what"]]


@case
def c5_lots_fit_forms_and_doors_land_on_streets():
    from ethoslm.pipeline import needs_footprint_failure
    city = compile_(copy.deepcopy(BASE))
    bad, doors = [], []
    for lf in city.leaves:
        if lf["kind"] in ("plot", "area"):
            nd = C._needs_of(lf["type"])
            if nd and needs_footprint_failure(lf, nd):
                bad.append(lf["name"])
        st = lf.get("site")
        if lf["kind"] == "plot" and st and lf["type"] != "ceremonial_hall":
            x, z = st["landing"]
            if city.use[city.ix(x), city.iz(z)] not in (C.ROAD, C.LANE, C.COURT, C.GATE):
                doors.append(lf["name"])
    assert not bad, f"{len(bad)} leaf/leaves outside their form's footprint: {bad[:3]}"
    assert not doors, f"{len(doors)} door(s) off the street: {doors[:3]}"
    plots = [lf for lf in city.leaves if lf["kind"] == "plot"]
    assert len(plots) > 100, f"only {len(plots)} plots"


@case
def c6_a_changed_region_keeps_its_neighbours():
    from ethoslm.pipeline import stages_design as SD
    city = compile_(copy.deepcopy(BASE))
    rec = {"frame": [city.X0, city.Z0, city.W]}
    ras = {"owner": city.owner, "target": city.target, "treat": city.treat,
           "pave": city.pave}
    leaves = [dict(lf) for lf in city.leaves]
    rids = sorted({lf["region"] for lf in leaves if lf.get("region")})
    before = {r: SD.region_inputs(None, rec, ras, leaves, r) for r in rids}
    victim = next(lf for lf in leaves if lf["kind"] == "plot" and lf.get("region"))
    victim["params"] = {**victim["params"], "storeys": 2}
    after = {r: SD.region_inputs(None, rec, ras, leaves, r) for r in rids}
    changed = [r for r in rids if before[r] != after[r]]
    assert changed == [victim["region"]], f"changing one leaf changed {changed}"


@case
def c7_a_misplaced_landmark_is_a_finding():
    d = copy.deepcopy(BASE)
    d["landmarks"] = [{"name": "huge", "ring": "inner", "bearing": 45, "at": 0.5,
                       "form": "market", "size": [120, 120]}]
    city = compile_(d)
    assert any(f["field"] == "landmarks[0]" for f in city.findings), \
        "an oversized landmark was placed or dropped silently"


@case
def c8_the_axis_and_the_principal_hall():
    city = compile_(copy.deepcopy(BASE))
    halls = [lf for lf in city.leaves if lf["type"] == "ceremonial_hall"
             and lf["name"].startswith("palace_")]
    principal = [h for h in halls if h["params"].get("use") == "principal"]
    assert principal, "no principal hall"
    p = principal[0]
    assert p["params"]["storeys"] == 3 and p["params"]["eaves"] == 2
    cx = (p["x0"] + p["x1"]) / 2
    assert abs(cx - 0) <= 1, f"the principal hall stands off the axis ({cx})"
    assert p["front"] == "south" and p["site"]["door"][0] in (-1, 0, 1)


def main() -> int:
    for fn in [c1_round_walls_hug_the_circle, c2_a_square_request_is_square,
               c3_a_small_settlement_keeps_its_ground, c4_no_street_into_an_ungated_wall,
               c5_lots_fit_forms_and_doors_land_on_streets,
               c6_a_changed_region_keeps_its_neighbours,
               c7_a_misplaced_landmark_is_a_finding, c8_the_axis_and_the_principal_hall]:
        fn()
    for name, ok, why in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {why}" if why else ""))
    n = sum(ok for _, ok, _ in RESULTS)
    print(f"{n}/{len(RESULTS)}")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
