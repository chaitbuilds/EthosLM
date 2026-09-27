#!/usr/bin/env python3
"""**The parent composition round's contracts, on synthetic ground.** Seconds, no state.

    $PY scripts/test_parent_composition.py

    c1  the forms' section: on a strip 51 deep behind its street, lanes across the strip
        hold two courtyard houses a side and a module of two rows is 49 wide; a lane along
        the strip with a row facing it from each side needs 59 and is 8 short, with the
        parent decision that would supply it named
    c2  `lanes: between` lays the rows facing each other across their lane: the lane is
        fronted on both sides, where the edge-first sweep fronts each lane on one
    c3  a lane no front faces is a failed lane (no "footway" exemption), and a proposal
        withdraws a lane whose rows did not stand rather than keep it blank
    c4  a shop facing the market lines the principal street and is not a front on it
    c5  `refit` levels the pieces beside the arrival for their modules and the least
        step to it: the ground's own choice (the arrival high, the homes low) is revised
        to the pair that keeps every module the ground admits within a terrace step
    c6  the built reading: doors facing each other across a lane are `faced`, a door
        onto a routed street `on_street`
    c7  the promotion guard protects the spatial role: a trial keeping every home and
        shop working but losing the lane's facing gates regresses
    c8  water is judged where it stands: a trial that moves water out of the building
        pieces' count while the section's standing water rises regresses; landscape kept
        as found owes no court
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from ethoslm import parentdemand as pd  # noqa: E402
from ethoslm import section  # noqa: E402
from ethoslm import streetplan as sp  # noqa: E402
from ethoslm.pipeline import promote  # noqa: E402

CH = {"forms": {"dwelling": {"court": 7}, "trade": {"storeys": 2}}, "court_least": 7}
HOUSE = {"widths": (17, 19), "pref": 17, "depth": 20, "depth_lo": 20, "depth_hi": 22,
         "end_extra": 1, "type": "courtyard_house"}
SHOP = {"widths": (7, 10), "pref": 8, "depth": 12, "depth_lo": 12, "end_extra": 1,
        "type": "shop_house"}


def _ground(w=72, d=51, road_z=True):
    """A piece `w` along x and `d` deep, with its principal street along the north."""
    X0, Z0 = 0, 0
    ok = np.ones((w, d), bool)
    road = {(x, Z0 - 3) for x in range(-10, w + 10)} | {(x, Z0 - 4) for x in range(-10, w + 10)}
    return (X0, Z0, X0 + w - 1, Z0 + d - 1), ok, road


def c1_sections():
    f = pd.forms(CH, "courtyard_house")
    assert f and f["house"]["width"] == 17 and f["house"]["depth"] == 20, f
    s = pd.sections(f, 51)
    a, b = s["across"], s["along"]
    assert a["fits"] and a["houses_per_row"] == 2, a
    assert a["module"]["rows_2"] == 49 and a["module"]["rows_3"] == 76, a
    assert not b["fits"] and b["short"] == 8 and "supply" in b, b
    return (f"across: {a['houses_per_row']} a side, module {a['module']}; along needs "
            f"{b['needs']} of 51 ({b['short']} short)")


def _prop(fam_extra):
    # a piece the width of the lane piece's whole ground behind its shops (the site's
    # -5704..-5650): one module of two rows, not three
    rect, ok, road = _ground(w=56)
    fams = [{"axis": "z", "depth": 20, "shop_reach": 96, **fam_extra}]
    got = sp.compose(rect, ok, road, access=(-20, -10), house=HOUSE, shop=SHOP,
                     families=fams)
    return got["proposals"][0]


def c2_rows_face_across_their_lane():
    p = _prop({"lanes": "between"})
    lanes = [r for r in p["relationships"] if r["name"].startswith("lane_")
             and r.get("sides")]
    both = [r for r in lanes if min(r["sides"].values()) >= sp.LANE_FRONTED]
    assert p["admissible"] and both, (p["failed"], lanes)
    q = _prop({})
    lanes_q = [r for r in q["relationships"] if r["name"].startswith("lane_")
               and r.get("sides")]
    assert not any(min(r["sides"].values()) >= sp.LANE_FRONTED for r in lanes_q), lanes_q
    return (f"between: {len(both)} lane(s) fronted both sides, {p['houses']} houses; "
            f"edge-first: {[r['sides'] for r in lanes_q]}")


def c3_lane_without_fronts_fails():
    rect, ok, road = _ground()
    P = sp._Plan(rect[0], rect[1], ok, road)
    P.lines = sp.principal_lines(P)
    P.streets = [{"kind": "lane", "rect": [20, 0, 24, 50], "axis": "z"}]
    P.take([20, 0, 24, 50], street=True)
    v = sp.judge(P)
    assert "lane_0_fronted" in v["failed"], v["failed"]
    # a proposal whose rows did not stand withdraws the lane it had laid for them
    sp._prune_lanes(P)
    assert not P.streets and not P.street.any(), P.streets
    return f"judged {v['failed']}; withdrawn: {P.notes[-1]}"


def c4_market_shops_are_not_street_fronts():
    rect, ok, road = _ground(w=40, d=40)
    P = sp._Plan(rect[0], rect[1], ok, road)
    P.lines = sp.principal_lines(P)
    # a row of shops on the north street, all facing east into a market beside them
    P.lots = [{"rect": [x, 0, x + 7, 11], "front": "east", "use": "shop",
               "street": "anchor", "attached": []} for x in range(0, 32, 8)]
    v = sp.judge(P, house_needed=False)
    pn = next(r for r in v["relationships"] if r["name"] == "principal_north_fronted")
    assert pn["measure"]["fronts"] == 0.0 and not pn["held"], pn
    return f"north street lined {pn['measure']['lined']:.0%}, fronted {pn['measure']['fronts']:.0%}"


def c5_refit_levels_for_modules():
    f = pd.forms(CH, "courtyard_house")
    n, across = 160, 60
    # an arrival end whole at 72 and 70 (its first columns only at 72), a homes run
    # whole at 64 and 66 for 55 columns, a gully between
    prof = {}
    for L in (64, 66, 68, 70, 72):
        row = np.zeros(n, int)
        if L in (70, 72):
            row[0 if L == 72 else 4:42] = across
        if L in (64, 66):
            row[50:105] = across
        if L == 64:
            row[105:115] = across
        prof[L] = row
    districts = [{"name": "s", "x0": 0, "z0": 0, "x1": n - 1, "z1": across - 1}]
    alt = {"arrangement": "ground_2", "cuts": 1, "pieces": [
        {"from": "s", "rect": [0, 0, 42, across - 1], "level": 72,
         "feasible_columns": 43 * across, "columns": 43 * across, "share": 1.0,
         "open": False},
        {"from": "s", "rect": [49, 0, 124, across - 1], "level": 64,
         "feasible_columns": 65 * across, "columns": 76 * across, "share": 0.86,
         "open": False}]}
    got = pd.refit(alt, districts, [(0, 0, n - 1, across - 1)], {0: prof}, f, depth=51,
                   ref=72, gap=6, along_x=True, access=(-5, -10))
    lv = [p["level"] for p in got["pieces"]]
    homes = got["pieces"][1]
    assert lv == [70, 66] and homes["module"]["rows"] == 2, (lv, got.get("modules"))
    assert homes["programme"] == homes["module"]["houses"] + homes["module"]["shops"]
    return f"levels {lv} (the ground's own 72/64), step {got['step']}; {got['modules']}"


def _row(part, t, rect, door, working=True):
    return {"part": part, "type": t, "stood": True, "emitted": {
        "storeys": 1, "rects": {"main": rect},
        "usable": {"entrance_connected": {"holds": working}}},
            "door": door, "params": {"trade": "x"} if t == "shop_house" else {}}


def c6_reading_roles():
    rows = [_row("w", "courtyard_house", [0, 0, 17, 17], [17, 8]),
            _row("e", "courtyard_house", [25, 0, 42, 17], [25, 8]),
            _row("s", "shop_house", [0, 30, 7, 39], [3, 30])]
    faced = {}
    for r in rows:
        faced[r["part"]] = section._door_face(r, r["emitted"]["rects"]["main"])
    a = {"working": True, **faced["w"]}
    b = {"working": True, **faced["e"]}
    assert section._faces_across(a, b) and section._faces_across(b, a), (a, b)
    s = {"working": True, **faced["s"]}
    road = {(3, 27), (3, 26)}
    assert section._opens_on(s, road, section.FACE_STREET_REACH), s
    assert not section._opens_on(a, road, section.FACE_STREET_REACH)
    return f"w {faced['w']['door_faces']} / e {faced['e']['door_faces']} faced; shop on street"


def c7_guard_protects_the_lane():
    def sec(facing):
        return {"relationships": [{"id": "functions", "status": "demonstrated",
                                   "measured": {"homes": 4, "homes_working": 4,
                                                "homes_entered": 4, "shops": 7,
                                                "shops_working": 7,
                                                "homes_facing": facing,
                                                "shops_on_street": 7}}],
                "fabric": {}}
    want = {"relationships": {"functions": "demonstrated"},
            "quantities": promote._quantities(sec(4)), "subjects": {}}
    regs = promote.between(want, sec(0), None)
    ids = [r["id"] for r in regs]
    assert "homes facing across their lane" in ids, ids
    assert not promote.between(want, sec(4), None)
    return f"rejected on {ids}"


def c8_water_where_it_stands():
    from ethoslm import demand
    def sec(in_pieces, total):
        return {"relationships": [], "fabric": {"ground.water_in_pieces": in_pieces,
                                                "ground.water": total}}
    want = {"relationships": {}, "quantities": promote._quantities(sec(83, 92)),
            "subjects": {}}
    ids = [r["id"] for r in promote.between(want, sec(51, 108), None)]
    assert "ground water standing" in ids, ids
    assert not promote.between(want, sec(83, 90), None)
    hill = {"name": "h", "density": "dense", "character": {},
            "sector": {"open": True, "module": {"role": "landscape", "ground": "as_found"}}}
    assert demand.court_obligation(hill) is None
    return f"moved water rejected on {ids}; landscape owes no court"


def main() -> int:
    t0 = time.time()
    cases = [c1_sections, c2_rows_face_across_their_lane, c3_lane_without_fronts_fails,
             c4_market_shops_are_not_street_fronts, c5_refit_levels_for_modules,
             c6_reading_roles, c7_guard_protects_the_lane, c8_water_where_it_stands]
    bad = 0
    for c in cases:
        try:
            print(f"ok   {c.__name__}: {c()}")
        except Exception as e:                            # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {c.__name__}: {e}")
    print(f"\n{len(cases) - bad}/{len(cases)} parent composition cases pass "
          f"({time.time() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
