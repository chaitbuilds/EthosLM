#!/usr/bin/env python3
"""**The allocation contracts of a city's rings.** Seconds, no state.

    $PY scripts/test_city_attempt.py

    a1  a ring's section from its forms: with a 17x20 courtyard lot and a 12-deep shop,
        a principal street along the strip's edge and one lane along the strip behind
        its frontage needs a district 70 deep; the three other proposals are recorded
        deeper, and `across` holds `LANE_LEAST` houses a side
    a2  the section governs an **inferred** ring's width: a ring whose count is the
        ground's own and whose share gives it less than its section is widened to it,
        the rings with slack are squeezed, and no counted sentence is needed
    a3  the control: a ring whose fabric is not composed from its streets asks for
        no section and keeps its share's width
    a4  a site too small for the sections refuses by name, with the shortfall and the
        rings whose sections set their least widths, instead of clipping them
    a5  the section reaches the districts: every strip of the ring carries it, owed,
        and the arterial is joined at the strip's inner edge rather than its centre
    a6  an owed section the composer cannot lay makes the composition inadmissible
        (`section_owed`) and says who owns the answer
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import parentdemand as pd  # noqa: E402
from ethoslm import placeplan  # noqa: E402
from ethoslm import spec as spec_mod  # noqa: E402

CH = {"forms": {"dwelling": {"court": 7}, "trade": {"storeys": 2}}, "court_least": 7,
      "layout": "street", "attached": True, "frontage": "street"}

SPEC = {
    "kind": "city", "invariants": None,
    "defining_parts": [
        {"name": "palace", "kind": "group", "family": "palace", "relation": "centre",
         "count": 1, "structures": 0, "notes": "the centre"},
        {"name": "inner_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "ring": 0, "share": 0.1, "walled": True,
         "density": "sparse", "role": "urban", "notes": "grid fabric"},
        {"name": "hutong_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "ring": 1, "share": 0.12, "walled": True,
         "density": "dense", "role": "urban", "character": dict(CH),
         "notes": "a street-composed courtyard quarter"},
        {"name": "outer_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "ring": 2, "share": 0.25, "walled": True,
         "density": "dense", "role": "urban", "notes": "rows"},
        {"name": "belt", "kind": "group", "family": "district", "relation": "concentric",
         "count": 1, "ring": 3, "share": 0.45, "walled": True, "density": "sparse",
         "role": "rural", "notes": "farms"},
        {"name": "walls", "kind": "edge", "family": "wall", "relation": "concentric",
         "count": 4, "structures": 0, "notes": "four walls"},
    ],
    "setting": {"surface": "green", "relief": "flat"}, "voice": "ochre_stone_green_tile",
    "form": "east_asian", "notes": "a fixture",
}
TYPES = ["courtyard_house", "shop_house", "row_house"]


def _layout(spec_raw=SPEC, size=768):
    spec = spec_mod.read_spec(json.loads(json.dumps(spec_raw)), "Build a ringed city.")
    for p in spec["defining_parts"]:
        if p.get("ring") is not None:
            p["demand"] = {"types": list(TYPES)}
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placeplan.concentric_layout(spec, {"origin": [0, 0], "size": size},
                                               None, decls, "ochre_stone_green_tile")
    return place, fails


def a1_section_from_forms():
    f = pd.forms(CH, "courtyard_house")
    got = pd.ring_section(f, road=placeplan.ARTERIAL_WIDTH)
    c = got["chosen"]
    assert (c["street"], c["arrangement"], c["depth"]) == ("edge", "along", 70), c
    others = [p for p in got["proposals"] if p is not c]
    assert len(others) == 3 and all(p["depth"] > 70 for p in others), others
    across = next(p for p in got["proposals"]
                  if p["arrangement"] == "across" and p["street"] == "edge")
    assert across["per_side"] == 14 + pd.row_run(f, pd.LANE_LEAST), across


def a2_section_widens_an_inferred_ring():
    place, fails = _layout()
    assert not fails, fails
    rings = {r["name"]: r for r in place["layout"]["rings"]}
    h = rings["hutong_ring"]
    assert h["section"] and h["width"] >= h["section"]["width"] == 70 + sum(h["insets"]), h
    assert h["share_got"] > h["share_asked"], h
    assert not h["target"]["counted"], h["target"]
    assert "hutong_ring" not in place["layout"]["squeezed"], place["layout"]["squeezed"]
    assert set(place["layout"]["squeezed"]) & {"outer_ring", "belt"}, place["layout"]


def a3_no_section_without_street_composition():
    place, _ = _layout()
    rings = {r["name"]: r for r in place["layout"]["rings"]}
    for n in ("inner_ring", "outer_ring", "belt"):
        assert rings[n]["section"] is None, (n, rings[n]["section"])


def a4_too_small_refuses_by_name():
    raw = json.loads(json.dumps(SPEC))
    for p in raw["defining_parts"]:
        if p["name"] in ("inner_ring", "outer_ring"):
            p["character"] = dict(CH)
    place, fails = _layout(raw, size=320)
    assert place is None, "expected a refusal"
    f = next(x for x in fails if x["check"] == "shares")
    assert f["short"] > 0 and "hutong_ring" in f["sections"], f
    assert "neighbourhood's section" in f["why"], f["why"]


def a5_section_reaches_the_districts():
    place, _ = _layout()
    mine = [d for d in place["districts"] if d["defines"] == "hutong_ring"]
    assert mine and all((d.get("section") or {}).get("owed") for d in mine), mine[:1]
    cx, cz = place["layout"]["centre"]
    for d in mine:
        node = placeplan.section_street_node(d, (cx, cz))
        assert node is not None
        wide = (d["x1"] - d["x0"]) >= (d["z1"] - d["z0"])
        if wide:
            inner = d["z1"] if d["z1"] < cz else d["z0"]
            assert abs((node[1] + node[3]) // 2 - inner) <= 4, (d["name"], node, inner)
        else:
            inner = d["x1"] if d["x1"] < cx else d["x0"]
            assert abs((node[0] + node[2]) // 2 - inner) <= 4, (d["name"], node, inner)
    others = [d for d in place["districts"] if d["defines"] == "outer_ring"]
    assert all(placeplan.section_street_node(d, (cx, cz)) is None for d in others)


def a6_owed_section_unmet_is_inadmissible():
    import numpy as np
    from ethoslm import streetplan as sp
    house = {"widths": (17, 19), "pref": 17, "depth": 20, "depth_lo": 20,
             "depth_hi": 22, "end_extra": 1, "type": "courtyard_house"}
    # 40 deep behind a street along the north: no lane along the strip fits
    ok = np.ones((120, 40), bool)
    road = {(x, -3) for x in range(-10, 130)} | {(x, -4) for x in range(-10, 130)}
    got = sp.compose((0, 0, 119, 39), ok, road, house=house, shop=None,
                     prefer={"axis": "x", "lanes": None})
    mod = got["module"]
    assert mod is not None and "realized" in mod, got["module"]
    # the compiler's rule, as it applies it: owed and not realized is inadmissible
    src = open(os.path.join(ROOT, "src", "ethoslm", "district_compile.py")).read()
    assert '"section_owed"' in src and 'composition["admissible"] = False' in src


def main() -> int:
    t0 = time.time()
    cases = [a1_section_from_forms, a2_section_widens_an_inferred_ring,
             a3_no_section_without_street_composition, a4_too_small_refuses_by_name,
             a5_section_reaches_the_districts, a6_owed_section_unmet_is_inadmissible]
    bad = 0
    for c in cases:
        try:
            c()
            print(f"  ok   {c.__name__}")
        except Exception as e:                   # noqa: BLE001
            bad += 1
            print(f"  FAIL {c.__name__}: {type(e).__name__}: {str(e)[:400]}")
    print(f"{len(cases) - bad}/{len(cases)} in {time.time() - t0:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
