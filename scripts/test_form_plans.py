#!/usr/bin/env python3
"""**Admission and emission read one form plan.** The design resolution round's focused
checks, offline, a few seconds each.

    $PY scripts/test_form_plans.py [name-fragment]

    f1  a courtyard house built on the plan's pad has the plan's rooms and court, measured
        on the blocks (flat and bank; middle and end of a run)
    f2  a shop house stands the storeys its plan admits, and its shop is at least as wide
        as the plan says (the plan never promises more than the builder lays)
    f3  a pad the rooms and court do not fit is refused with the pad that fits; a lot
        admitted by the compiler's own lot band holds the form in every flank
        configuration a run can leave it in
    f4  a leaf is certified in its own type's storeys band, not the leading type's: a
        shop house in a quarter led by one-storey courtyard houses is asked about two
    f5  a run's free ends are cut wider by the inset they keep (`end_extra`)
    f6  a knoll of a few columns over the reach is cut; a hillside over it is not
    f7  the strip remnant between two stepped districts, and the band against a lower
        inner ring, are seams at the lower level (fr's two ridges)
"""
from __future__ import annotations

import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _lab():
    import form_lab
    return form_lab


@case
def f1_courtyard_rooms_as_planned():
    lab = _lab()
    out = []
    for tag in ("court_c7", "court_bank_c7", "court_c9"):
        r = lab.run(tag)
        assert not r["refused"] and not r["lint_errors"] and not r["party_gaps"], r
        for h in r["houses"]:
            pl, me = h["planned"], h["measured"]
            assert pl["ok"], (tag, h["label"], pl)
            assert me["court"] == pl["court"], (tag, h["label"], me["court"], pl["court"])
            clear = pl["clear"]
            for k, v in me["clear"].items():
                want = clear["gate" if k == "gate" else "hall" if k == "hall" else "wing"]
                assert v >= want, (tag, h["label"], k, v, want)
        out.append(f"{tag} {[h['measured']['court'] for h in r['houses']]}")
    return "; ".join(out)


@case
def f2_shop_storeys_and_floor_as_planned():
    lab = _lab()
    out = []
    for tag in ("shop_8x12", "shop_7x13"):
        r = lab.run(tag)
        assert not r["refused"] and not r["lint_errors"], r
        for h in r["houses"]:
            pl, me = h["planned"], h["measured"]
            assert me["storeys"] == pl["storeys"] == 2, (tag, h["label"], pl, me)
            assert me["clear_width"] >= pl["clear_width"], (tag, h["label"], pl, me)
        out.append(f"{tag} planned clear {[h['planned']['clear_width'] for h in r['houses']]}"
                   f" built {[h['measured']['clear_width'] for h in r['houses']]}")
    return "; ".join(out)


@case
def f3_refusal_names_the_pad_that_fits():
    from ethoslm import formplan
    fn = formplan.plan_fn("courtyard_house")
    got = fn(15, 13, params={"court": 7}, front="north")
    assert not got["ok"] and got["need"] == [17, 18], got
    lot = formplan.least_lot("courtyard_house", {"court": 7}, attached=("west", "east"))
    end = formplan.least_lot("courtyard_house", {"court": 7}, attached=("west",))
    assert lot["lot"] == [17, 20] and end["lot"] == [18, 20], (lot["lot"], end["lot"])
    assert formplan.admits("courtyard_house", 17, 20, {"court": 7},
                           configs_=[("west", "east")])
    assert not formplan.admits("courtyard_house", 17, 20, {"court": 7}, configs_=[("west",)])
    return (f"15x13 refused, needs {got['need']}; least lot {lot['lot']} between party "
            f"walls, {end['lot']} at an end")


@case
def f4_leaf_in_its_own_band():
    from ethoslm import demand, placeplan
    _t, decls = placeplan.types_card(["courtyard_house", "shop_house", "row_house"])
    part = {"name": "middle_ring", "character": {"storeys": [1, 2], "attached": True}}
    d = demand.resolve({}, {}, part, decls,
                       context={"capabilities": None})
    d["types"] = ["courtyard_house", "shop_house", "row_house"]
    lead = d["storeys_band"]
    shop = (d.get("bands") or {}).get("shop_house")
    assert lead == [1, 1] and shop == [1, 2], (lead, shop)
    got = demand.storeys_admitted(d, 8, 12, flanks=2, type_name="shop_house")
    assert got["band"] == [1, 2], got
    return f"leading band {lead}, shop band {shop}; an 8x12 shop answered in {got['band']}"


@case
def f5_free_ends_cut_wider():
    from ethoslm import streetplan as sp
    ok = np.ones((80, 20), bool)
    P = sp._Plan(0, 0, ok, set())
    lots = sp.cut_run(P, {"axis": "x", "a0": 0, "a1": 71, "b0": 0, "b1": 19,
                          "front": "north", "use": "house", "street": "lane", "id": "t"},
                      {"widths": (17, 19), "pref": 17, "depth": 20, "end_extra": 1},
                      random.Random(1))
    ws = [lt["rect"][2] - lt["rect"][0] + 1 for lt in lots]
    assert len(ws) >= 3 and ws[0] >= 18 and ws[-1] >= 18, ws
    assert all(17 <= w <= 19 for w in ws[1:-1]), ws
    return f"widths along a 72-column run: {ws}"


@case
def f6_knoll_cut_hill_kept():
    from ethoslm import feasible
    from ethoslm.observe import Volume
    size = 60
    h = np.full((size, size), 64, np.int32)
    h[20:24, 20:25] = 75                                   # a knoll, 20 columns, 11 up
    h[40:, :] = 64 + np.arange(size - 40)[:, None] * 2       # a hillside rising out of it
    codes = np.zeros((size, 120, size), np.uint16)
    for x in range(size):
        for z in range(size):
            codes[x, :int(h[x, z]) - 20 + 1, z] = 1
    vol = Volume(0, 20, 0, codes, ["air", "stone"])
    got = feasible.record(vol, (0, 0, size - 1, size - 1), level=64, relief=8, fill=8)
    m = feasible.mask_of(got)
    assert m[21, 22], "the knoll was left as found"
    assert not m[55, 10], "the hillside was cut"
    return f"knolls {got.get('knolls')}"


@case
def f7_seams_at_the_lower_level():
    import json
    from ethoslm.pipeline import stages_plan as sp
    place = json.load(open(os.path.join(ROOT, "out", "fr-frozen-city", "plan.place.json")))
    lay = place["layout"]
    rings = lay["rings"]
    cx, cz = int(lay["centre"][0]), int(lay["centre"][1])
    comp = lay.get("compound_rect")
    ring_pieces = {}
    for k in range(len(rings)):
        oh = int(rings[k]["outer"]) + 2
        inner = (tuple(int(v) for v in comp) if k == 0 and comp else
                 ((cx - (int(rings[k - 1]["outer"]) + 2), cz - (int(rings[k - 1]["outer"]) + 2),
                   cx + (int(rings[k - 1]["outer"]) + 2), cz + (int(rings[k - 1]["outer"]) + 2))
                  if k else None))
        ring_pieces[k] = {"outer": (cx - oh, cz - oh, cx + oh, cz + oh), "inner": inner}
    got = {g["label"]: g for g in sp._seam_pieces(place, rings, ring_pieces)}
    gap = got.get("seam/middle_ring_north_east_1/middle_ring_north_east_2")
    band = got.get("seam/middle_ring_north_east_2/south")
    assert gap and tuple(gap["rect"])[0] == -5700 and gap["level"] == 64, gap
    assert band and band["rect"][1] == 762 and band["level"] == 64, band
    return f"the x=-5695 remnant {gap['rect']} at {gap['level']}; the z=762 band at {band['level']}"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    ok = fail = 0
    t0 = time.time()
    for fn in CASES:
        if only and only not in fn.__name__:
            continue
        try:
            said = fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            fail += 1
            continue
        except Exception as e:                          # noqa: BLE001 -- reported
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            fail += 1
            continue
        ok += 1
        print(f"ok   {fn.__name__}: {said}")
    print(f"\n{ok}/{ok + fail} form plan cases pass ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
