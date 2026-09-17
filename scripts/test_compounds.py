"""The monument: a great thing is a nested place, and the sweep no longer decides scale.

    $PY scripts/test_compounds.py

  M1. **A compound is a kind of defining part.** A palace is a compound whatever kind
      the spec call guessed; a keep is a building unless the spec makes it a `group`;
      the size of the place is counted over its districts and never over a compound.
  M2. **The place level draws a compound as a rectangle on its cut ground.** A place
      plan with the compound on its plateau passes; one that draws the palace as a leaf,
      leaves it out, draws it off the plateau, over a corner of it, too small, or on top
      of a district is refused by name.
  M3. **The road arrives at the compound's edge and never crosses it.** The arterial is
      routed to the compound as a node and the record says where it arrives.
  M4. **The compound level.** A wall, a gate where the road arrives, two halls and a
      court inside the wall pass; an open wall, no gate, a gate off the wall or away
      from the road, a hall outside the wall or the rectangle, one hall, and a shop in a
      palace are each refused by name -- and the wall, which is defensive, is admitted
      in a civic compound.
  M5. **Assembly.** The compound is a quarter of its own; its parts are ordinary leaves
      carrying what they answer; the wall goes in the walls wave; the assembled plan
      passes the whole-tree check.
  M6. **The place read.** `present/`, `compound/.../scale` and `compound/.../ground` hold
      on a standing compound and fail by name on a wall that fell, a hall on a deck, a
      compound off its plateau and one no bigger than a shop; the great-wall clause holds
      at 30 and fails at 20.
  M7. **The sweep is a floor and not a ceiling.** A type clean over two runs declares
      the envelope and names the sizes between them; the plan refuses a named size on
      either axis and admits a clean size above the lower run; a malformed `except` is
      refused by name; the brief's table says where a type is never at.
  M8. **Built.** A compound on a plateau -- wall, gate, court, three halls -- is sited,
      built and read offline through the same stages a city is: every part stands, the
      build family reports no error on it, every door is reachable from the lane, the
      interior floor is walkable, and it is larger than the house beside it by the
      registered margin.
  M9. **The composition is the family's and the rectangle's, not a palace's** (v2,
      C0). A monument -- one hall and its precinct, no wall, no gate -- a shrine
      precinct -- a temple and its garden -- and a castle -- a keep, a curtain wall, a
      gatehouse and a bailey -- each pass the compound validator, assemble, and hold
      the place read's `present/` clause; a castle without its wall, a monument with an
      open wall drawn, and a gate or a keep in a shrine (an unwalled precinct admits
      no defensive type) are each refused by name; the ground a monument needs is less than a palace's; a
      monument does not scale with its rectangle and a palace does; the brief says what
      each is made of.

M8 needs `types/` and the library and nothing under `out/` but a scratch directory of
its own; nothing here needs a cached world.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (  # noqa: E402
    circulate, observe, offline, pipeline, placeplan, placeread, spec as spec_mod,
)
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# ------------------------------------------------------------------ the fixture A town
# with a palace at its middle and a district of houses: the smallest place that has a
# great thing in it. The spec is written here as a fixture, for the reason every fixture
# in this project is: a case whose input is a model call measures the model.

SENTENCE = "Build a town with a palace at its centre."

TOWN_SPEC = {
    "kind": "town",
    "defining_parts": [
        {"name": "palace", "kind": "plot", "family": "palace", "relation": "centre",
         "count": 1, "structures": 1, "needs": {"plateau": 64},
         "notes": "the palace: halls and courts inside a wall of its own"},
        {"name": "houses", "kind": "group", "family": "district",
         "relation": "throughout", "count": 1, "structures": 4,
         "notes": "the houses"},
    ],
    "voice": None,
}

SIZE = 192
GROUND = 64
Y0 = 40
VOICE = "white_render_dark_frame"

SITE = {"origin": [0, 0], "size": SIZE,
        "stats": {"min": GROUND - 8, "max": GROUND, "relief": 8, "std": 2.0},
        "mean_grid": [[GROUND] * 12 for _ in range(12)],
        "roughness_grid": [[2] * 12 for _ in range(12)],
        "surface_blocks": {"grass_block": 100}}

#: The 64x64 the site search would have levelled for the palace, at the middle.
PLATEAU = {"part": "palace", "rect": [64, 64, 127, 127], "y": GROUND}


def _spec():
    spec = spec_mod.read_spec(json.loads(json.dumps(TOWN_SPEC)), SENTENCE)
    # A district-sized town and not a forty-house one: the spec under test is the shape
    # of the thing, and the counts are the run's business (`test_place` does the same).
    spec["structures"], spec["size_band"] = 4, [4, 12]
    return spec


def _place(**over):
    place = {
        "intent": "a fixture", "centre": "palace", "voice": VOICE,
        "circulation_material": "cobblestone",
        "parts": [],
        "compounds": [{"name": "palace", "defines": "palace",
                       "x0": 64, "z0": 64, "x1": 127, "z1": 127,
                       "notes": "the palace on its plateau"}],
        "districts": [{"name": "houses", "defines": "houses",
                       "x0": 20, "z0": 12, "x1": 171, "z1": 48,
                       "structures": 4, "purpose": "houses", "notes": ""}],
    }
    place.update(over)
    return place


def _compound(**over):
    """The palace: a wall just inside the rectangle, a gate on its north side where the
    road from the houses arrives, a court behind the gate, two side halls and the
    great hall at the far end of the axis. Every gap is the clearance the types
    declare plus the lanes."""
    parts = [
        {"kind": "edge", "name": "palace_wall", "type": "wall", "seed": 1,
         "params": {"height": 6, "width": 1, "crown": "solid"}, "width": 1,
         "path": [[67, 67], [124, 67], [124, 124], [67, 124], [67, 67]],
         "notes": "the compound wall"},
        {"kind": "point", "name": "palace_gate", "type": "ring_gate", "seed": 2,
         "params": {"storeys": 2, "crown": "hip"}, "at": [96, 67],
         "facing": "north", "size": 5, "notes": "the gate, where the road arrives"},
        {"kind": "area", "name": "great_court", "type": "square", "seed": 3,
         "params": {"paving": "banded", "canopy": "hip"},
         "x0": 90, "z0": 72, "x1": 103, "z1": 85, "notes": "the court"},
        {"kind": "plot", "name": "west_hall", "type": "hall", "seed": 4,
         "params": {"dormers": 1, "use": "moot"},
         "x0": 72, "z0": 90, "x1": 83, "z1": 105, "notes": "a side hall"},
        {"kind": "plot", "name": "east_hall", "type": "hall", "seed": 5,
         "params": {"dormers": 2, "use": "refectory"},
         "x0": 108, "z0": 90, "x1": 119, "z1": 105, "notes": "a side hall"},
        {"kind": "plot", "name": "great_hall", "type": "hall", "seed": 6,
         "params": {"dormers": 2, "use": "moot"},
         "x0": 88, "z0": 105, "x1": 105, "z1": 118, "notes": "the great hall"},
    ]
    got = {"notes": "the palace", "axis": "north", "parts": parts}
    got.update(over)
    return got


def _district():
    return {"notes": "houses", "quarters": [{"name": "row", "notes": "", "plots": [
        {"kind": "plot", "name": f"house_{i}", "type": "townhouse", "seed": i + 1,
         "params": {}, "x0": 30 + i * 34, "z0": 22, "x1": 44 + i * 34, "z1": 36,
         "notes": ""} for i in range(4)]}]}


def _ground(size: int = SIZE, relief: int = 8) -> Volume:
    """Grass falling `relief` blocks from north to south."""
    codes = np.zeros((size, GROUND + 24 - Y0, size), np.uint16)
    for z in range(size):
        g = GROUND - int(round(relief * z / (size - 1)))
        codes[:, :g - Y0, z] = 2
        codes[:, g - Y0, z] = 1
    return Volume(0, Y0, 0, codes, ["air", "grass_block", "stone"])


def _heights(vol):
    h, _wet = observe.ground_heights(vol)
    return h


def _decls():
    _t, decls = placeplan.types_card()
    return decls


def _stood(plan, **over):
    rows = []
    for p in pipeline.plan_parts(plan):
        rows.append({"part": p["name"], "status": "built", "stood": True,
                     "blocks": 2000 if p.get("compound") else 900,
                     "ground": "plinth" if p.get("kind") != "edge" else "footing"})
    for r in rows:
        r.update(over.get(r["part"], {}))
    return {"waves": [{"wave": "all", "parts": rows}]}


# ------------------------------------------------------- M1. a compound is a part

@case
def t_m1_a_palace_is_a_compound_and_a_keep_is_a_building_unless_the_spec_says_group():
    spec = _spec()
    pal = next(p for p in spec["defining_parts"] if p["name"] == "palace")
    assert spec_mod.compound(pal), pal
    assert pal["role"] == "civic", pal["role"]
    assert not spec_mod.compound(next(p for p in spec["defining_parts"]
                                      if p["name"] == "houses"))
    # A keep as a plot is a building; a keep as a group is a castle.
    keep = spec_mod.read_part({"name": "keep", "kind": "plot", "family": "keep",
                               "relation": "centre"}, "x")
    castle = spec_mod.read_part({"name": "castle", "kind": "group", "family": "keep",
                                 "relation": "centre", "structures": 3}, "x")
    assert not spec_mod.compound(keep) and spec_mod.compound(castle)
    assert castle["role"] == "defensive"
    # ...and the size of the place is counted over its districts alone, so a compound
    # declaring structures does not make it a place of one building or scale the rest.
    doc = json.loads(json.dumps(TOWN_SPEC))
    doc["defining_parts"][0]["kind"] = "group"
    doc["defining_parts"][0]["structures"] = 9
    doc["defining_parts"][1]["structures"] = 60
    got = spec_mod.read_spec(doc, SENTENCE)
    houses = next(p for p in got["defining_parts"] if p["name"] == "houses")
    assert got["structures"] == 60 and houses["structures"] == 60, \
        (got["structures"], houses["structures"])
    p = os.path.join(ROOT, "out", "setting", "place.json")
    note = ""
    if os.path.exists(p):
        bss = spec_mod.read_spec(json.load(open(p)), "Build Ringed City.")
        comps = [q["name"] for q in spec_mod.compounds(bss)]
        assert comps == ["royal_palace"], comps
        note = "; the recorded Ringed City spec's royal_palace reads as one"
    return ("a palace is a compound whatever kind the spec guessed, a keep is a "
            "building and a group of keep is a castle; a compound's structures never "
            "size the place" + note)


# ------------------------------------------------- M2. the place level draws it

@case
def t_m2_the_place_level_draws_a_compound_on_its_cut_ground_or_is_refused_by_name():
    spec, decls = _spec(), _decls()
    ok = placeplan.place_failures(_place(), spec, SITE, decls, plateau=PLATEAU)
    assert not ok, ok

    def why(place, plateau=PLATEAU):
        return [(f["part"], f["check"]) for f in
                placeplan.place_failures(place, spec, SITE, decls, plateau=plateau)]

    # as a leaf, and not as a compound
    leaf = _place(compounds=[], parts=[
        {"kind": "area", "name": "palace", "type": "square", "seed": 1, "params": {},
         "x0": 80, "z0": 80, "x1": 95, "z1": 95, "notes": ""}])
    got = why(leaf)
    assert ("palace", "spec") in got and ("palace", "compound") in got, got
    # left out
    assert ("palace", "spec") in why(_place(compounds=[]))
    # off its plateau
    off = _place(compounds=[dict(_place()["compounds"][0], x0=30, z0=60, x1=93,
                                 z1=123)])
    assert ("palace", "plateau") in why(off), why(off)
    # over a corner of it
    corner = _place(compounds=[dict(_place()["compounds"][0], x0=64, z0=64, x1=95,
                                    z1=95)])
    got = why(corner)
    assert ("palace", "plateau") in got, got
    # ...and with no plateau cut, a corner-sized compound is only too small
    assert ("palace", "plateau") not in why(corner, plateau=None)
    small = _place(compounds=[dict(_place()["compounds"][0], x0=64, z0=64, x1=80,
                                   z1=80)])
    assert ("palace", "compound") in why(small, plateau=None), why(small, None)
    # on top of a district
    over = _place(districts=[dict(_place()["districts"][0], z1=70)])
    got = why(over)
    assert any(c == "overlap" for _n, c in got), got
    return ("a compound on its plateau passes; drawn as a leaf, left out, off the "
            "plateau, over a corner, too small and over a district are each refused by "
            "name")


# --------------------------------------------------- M3. the road arrives at it

@case
def t_m3_the_road_arrives_at_the_compounds_edge_and_never_crosses_it():
    decls = _decls()
    place = _place()
    vol = _ground()
    net, nodes = placeplan.plan_arterials(place, decls, _heights(vol), 0, 0)
    assert net is not None and {n["id"] for n in nodes} == {"houses", "palace"}, nodes
    rec = placeplan.arterial_record(net, nodes, place)
    place["arterials"] = rec
    joins = rec["joins"].get("palace") or []
    assert joins, "the road never reached the compound"
    r = placeplan.compound_rects(place)["palace"]
    inside = [c for c in rec["cells"] if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]]
    assert not inside, f"{len(inside)} arterial cells inside the compound"
    assert all(r[0] - 1 <= x <= r[2] + 1 and r[1] - 1 <= z <= r[3] + 1
               for x, z in joins), joins[:3]
    fails = placeplan.arterial_failures(place, rec, decls)
    assert not fails, fails
    side = placeplan._arrival_side(place["compounds"][0], joins)
    note = placeplan._arrival_note(place, place["compounds"][0])
    assert side in note and "gate" in note, note[:200]
    # ...and a road through the compound is refused by name
    bad = dict(rec, cells=rec["cells"] + [[96, 96]])
    got = placeplan.arterial_failures(place, bad, decls)
    assert any(f["part"] == "palace" and "through" in f["why"] for f in got), got
    return (f"{len(rec['cells'])} arterial columns from the houses to the palace's "
            f"{side} side, {len(joins)} of them against its edge and none inside it; "
            f"a road through it is refused")


# ----------------------------------------------------- M4. the compound level

@case
def t_m4_a_wall_a_gate_two_halls_and_a_court_pass_and_each_failure_is_named():
    spec, decls = _spec(), _decls()
    place = _place()
    vol = _ground()
    net, nodes = placeplan.plan_arterials(place, decls, _heights(vol), 0, 0)
    place["arterials"] = placeplan.arterial_record(net, nodes, place)
    comp = place["compounds"][0]
    part = spec_mod.compounds(spec)[0]
    _t, cdecls = placeplan.compound_types(None, spec, part)
    assert {"wall", "ring_gate", "hall", "square"} <= set(cdecls), sorted(cdecls)
    assert "shop_house" not in cdecls and "keep" in cdecls, sorted(cdecls)
    ok = placeplan.compound_failures(comp, _compound(), place, cdecls, spec=spec)
    assert not ok, ok

    def why(got):
        return [(f["part"], f["check"]) for f in
                placeplan.compound_failures(comp, got, place, cdecls, spec=spec)]

    def with_part(name, **change):
        got = _compound()
        for p in got["parts"]:
            if p["name"] == name:
                p.update(change)
        return got

    def without(name):
        got = _compound()
        got["parts"] = [p for p in got["parts"] if p["name"] != name]
        return got

    # an open wall
    got = why(with_part("palace_wall",
                        path=[[67, 67], [124, 67], [124, 124], [67, 124]]))
    assert ("palace_wall", "wall") in got, got
    # a wall on the rim, whose gate's pad would reach outside
    got = why(with_part("palace_wall",
                        path=[[64, 64], [127, 64], [127, 127], [64, 127], [64, 64]]))
    assert ("palace_wall", "inset") in got, got
    # no wall at all
    got = why(without("palace_wall"))
    assert ("palace", "wall") in got, got
    # no gate; a gate off the wall; a gate away from where the road arrives
    assert ("palace", "gate") in why(without("palace_gate"))
    assert ("palace_gate", "gate") in why(with_part("palace_gate", at=[96, 71]))
    far = why(with_part("palace_gate", at=[124, 96], facing="east"))
    assert ("palace_gate", "gate") in far, far
    # a hall outside the wall, and one outside the rectangle
    got = why(with_part("east_hall", x0=122, x1=133))
    assert ("east_hall", "compound") in got, got
    got = why(with_part("east_hall", x0=115, x1=126))
    assert ("east_hall", "inside") in got or ("east_hall", "overlap") in got, got
    # one hall only
    got = why(without("west_hall") | {"parts": [p for p in without("west_hall")["parts"]
                                               if p["name"] != "east_hall"]})
    assert ("palace", "composed") in got, got
    # a shop in a palace is refused by role; the wall, which is defensive, is not
    got = why(with_part("west_hall", type="shop_house",
                        params={"storeys": 2, "trade": "cloth"}))
    assert ("west_hall", "role") in got, got
    assert not any(n == "palace_wall" for n, _c in got), got
    return ("a wall, a gate at the road, a court and three halls pass; an open wall, "
            "no wall, no gate, a gate off the wall or away from the road, a hall "
            "outside the wall or the rectangle, too few halls and a shop are each "
            "refused by name, and the defensive wall is admitted in a civic compound")


# ------------------------------------------------------------- M5. assembly

def _assembled():
    spec, decls = _spec(), _decls()
    place = _place()
    vol = _ground()
    net, nodes = placeplan.plan_arterials(place, decls, _heights(vol), 0, 0)
    place["arterials"] = placeplan.arterial_record(net, nodes, place)
    plan = placeplan.assemble(place, {"houses": _district()}, spec,
                              {"palace": _compound()})
    return spec, decls, place, plan


@case
def t_m5_a_compound_is_a_quarter_of_its_own_and_its_parts_are_ordinary_leaves():
    spec, decls, place, plan = _assembled()
    parts = pipeline.plan_parts(plan)
    mine = [p for p in parts if p.get("compound") == "palace"]
    assert len(mine) == 6 and all(p.get("defines") == "palace" for p in mine), \
        [(p["name"], p.get("defines")) for p in mine]
    assert all(p["in"][-1] == "palace" for p in mine)
    assert all(p.get("role") == "civic" for p in mine)
    assert plan["compounds"][0]["name"] == "palace"
    assert plan["levels"]["compounds"] == {"palace": 6}, plan["levels"]
    waves = dict(pipeline.part_waves(parts))
    assert {p["name"] for p in waves["walls"]} == {"palace_wall", "palace_gate"}
    assert {p["name"] for p in waves["squares"]} == {"great_court"}
    assert {p["name"] for p in waves["palace"]} == {"west_hall", "east_hall",
                                                     "great_hall"}
    whole = pipeline.plan_failures(
        parts, pipeline.type_declarations(parts),
        ground={p["name"]: {"relief": 2, "water_pct": 0.0, "class": "dry",
                            "columns": 1} for p in parts})
    assert not whole, whole
    # ...and a name two levels share is qualified by the compound
    twice = _district()
    twice["quarters"][0]["plots"][0]["name"] = "great_hall"
    plan2 = placeplan.assemble(place, {"houses": twice}, spec, {"palace": _compound()})
    names = [p["name"] for p in pipeline.plan_parts(plan2)]
    assert "palace_great_hall" in names and "houses_great_hall" in names, names
    return (f"{len(parts)} leaves, {len(mine)} of them the palace's, carrying defines "
            f"and role; the wall and gate in the walls wave, the court in the squares "
            f"wave, the halls in the palace's own; the whole tree passes; a clashing "
            f"name is qualified")


# ----------------------------------------------------------- M6. the place read

@case
def t_m6_the_compound_clauses_hold_on_a_standing_compound_and_fail_by_name():
    spec, decls, place, plan = _assembled()
    parts = pipeline.plan_parts(plan)

    def read(rec, plateau=PLATEAU):
        got = placeread.read(spec, plan, rec, voice=VOICE, site=SITE, plateau=plateau)
        return {c["clause"]: c for c in got["clauses"]}

    c = read(_stood(plan))
    for k in ("present/palace", "compound/palace/scale", "compound/palace/ground"):
        assert c[k]["holds"], c[k]["says"]
    assert c["present/houses"]["holds"]
    assert "great/" not in "".join(c), "a town of one wall has no great wall"
    # the wall fell: not present
    got = read(_stood(plan, palace_wall={"stood": False}))
    assert not got["present/palace"]["holds"] and "NO closed wall" in \
        got["present/palace"]["says"], got["present/palace"]["says"]
    # a hall on a deck: the ground clause fails and names it
    got = read(_stood(plan, east_hall={"ground": "deck"}))
    assert not got["compound/palace/ground"]["holds"] and "east_hall" in \
        got["compound/palace/ground"]["says"]
    # off its plateau
    got = read(_stood(plan), plateau={"part": "palace", "rect": [0, 0, 63, 63],
                                      "y": GROUND})
    assert not got["compound/palace/ground"]["holds"] and "off the ground" in \
        got["compound/palace/ground"]["says"], got["compound/palace/ground"]["says"]
    # no bigger than a shop: a house outside it laid more blocks than the whole palace
    got = read(_stood(plan, house_1={"blocks": 20000}))
    assert not got["compound/palace/scale"]["holds"], got["compound/palace/scale"]["says"]
    assert "house_1" in got["compound/palace/scale"]["says"]
    # the great wall: a three-ring city's outermost ring, at 20 and at 30
    city = json.loads(json.dumps(spec))
    city["defining_parts"].append(
        spec_mod.read_part({"name": "ring_wall", "kind": "edge", "family": "wall",
                            "relation": "concentric", "count": 1}, "x"))
    ring = {"kind": "edge", "name": "ring_wall_1", "defines": "ring_wall",
            "type": "wall", "seed": 9, "params": {"height": 20, "width": 3},
            "width": 3, "path": [[4, 4], [187, 4], [187, 187], [4, 187], [4, 4]],
            "in": [], "notes": ""}
    stood = {"ring_wall_1": True}
    low = placeread.great_wall_clauses(city, plan, parts + [ring], decls, stood)
    assert len(low) == 1 and not low[0]["holds"] and low[0]["height"] == 20, low
    ring["params"]["height"] = placeread.GREAT_WALL_HEIGHT
    high = placeread.great_wall_clauses(city, plan, parts + [ring], decls, stood)
    assert high[0]["holds"], high
    fell = placeread.great_wall_clauses(city, plan, parts + [ring], decls, {})
    assert not fell[0]["holds"] and "DOES NOT STAND" in fell[0]["says"]
    # ...and the concentric gates clause judges the gates on the rings, not the
    # compound's own: a ring gate on the road and a palace gate off it hold
    ring_gate = {"kind": "point", "name": "ring_gate_1", "defines": "ring_gate",
                 "type": "ring_gate", "seed": 8, "params": {}, "at": [96, 4],
                 "facing": "north", "in": [], "notes": ""}
    city["defining_parts"].append(
        spec_mod.read_part({"name": "ring_gate", "kind": "point", "family": "gate",
                            "relation": "gateway", "count": 1}, "x"))
    road = dict(plan, arterials={"cells": [[x, z] for z in range(0, 70)
                                           for x in (95, 96, 97)]})
    got = {c["clause"]: c for c in placeread.concentric_clauses(
        city, road, parts + [ring, ring_gate], decls,
        {**{p["name"]: True for p in parts}, "ring_wall_1": True, "ring_gate_1": True})}
    assert got["concentric/gates"]["holds"], got["concentric/gates"]["says"]
    assert "palace_gate" not in got["concentric/gates"].get("off_arterial", [])
    return (f"present, scale and ground hold on the standing palace "
            f"({c['compound/palace/scale']['says'].split(';')[0]}); a fallen wall, a hall "
            f"on a deck, a compound off its plateau and a house bigger than the palace "
            f"each fail by name; the great wall holds at "
            f"{placeread.GREAT_WALL_HEIGHT} and fails at 20 and when it falls")


# ------------------------------------------------------------- M7. the sweep

@case
def t_m7_the_sweep_is_a_floor_inside_the_range_a_type_reaches_and_not_a_ceiling():
    import importlib.util
    p = os.path.join(ROOT, "scripts", "type_needs.py")
    s = importlib.util.spec_from_file_location("type_needs", p)
    tn = importlib.util.module_from_spec(s)
    s.loader.exec_module(tn)
    # clean at 8-12 and 19-28, dirty at 13-18: the palace's own shape of answer
    rows = [{"size": [n, n], "of": 18,
             "failed": 0 if (8 <= n <= 12 or 19 <= n <= 28) else 18, "first": None}
            for n in tn.AREA_SIZES]
    band = tn.band("area", rows)
    assert band["footprint"] == [8, 8, 28, 28], band
    assert band["runs"] == [[8, 12], [19, 28]], band
    assert band["except"] == [13, 14, 15, 16, 17, 18, 19 - 1][:6] or \
        band["except"] == [13, 14, 15, 16, 17, 18], band
    assert "broken at 13, 14, 15, 16, 17, 18" in band["note"], band["note"]
    # the declaration is the envelope and the exceptions, and the plan reads both
    needs = pipeline.read_needs({"NEEDS": {"footprint": (8, 8, 28, 28),
                                           "except": (13, 14, 15, 16, 17, 18),
                                           "frontage": "lane"}})
    assert needs["except"] == (13, 14, 15, 16, 17, 18), needs

    def pad(w, d):
        return {"kind": "area", "x0": 0, "z0": 0, "x1": w - 1, "z1": d - 1}

    assert pipeline.needs_footprint_failure(pad(24, 24), needs) is None
    assert pipeline.needs_footprint_failure(pad(10, 10), needs) is None
    why = pipeline.needs_footprint_failure(pad(15, 15), needs)
    assert why and "broken on 15" in why, why
    why = pipeline.needs_footprint_failure(pad(24, 13), needs)
    assert why and "broken on 13" in why, why
    assert "over 28" in pipeline.needs_footprint_failure(pad(30, 30), needs)
    # a plot is said as the plot drawn **and** the pad it gives, because a plot with a
    # side under nine is inset by one: an 8x8 plot is a 6x6 pad
    plot = pipeline.read_needs({"NEEDS": {"footprint": (5, 5, 22, 22),
                                          "except": (6, 9), "frontage": "lane"}})
    why = pipeline.needs_footprint_failure(
        {"kind": "plot", "x0": 0, "z0": 0, "x1": 12, "z1": 12}, plot)
    assert why and "13x13" in why and "pad of 9x9" in why and "broken on a pad of 9" \
        in why, why
    why = pipeline.needs_footprint_failure(
        {"kind": "plot", "x0": 0, "z0": 0, "x1": 7, "z1": 7}, plot)
    assert why and "8x8" in why and "pad of 6x6" in why and "inset by one" in why, why
    table = pipeline.needs_table({"x": {"needs": plot, "kind": "plot"}})
    assert "never on a pad of" in table and "| 6, 9 |" in table \
        and "keeps clear" in table and "8x8 plot is a 6x6 pad" in table, table
    # malformed, refused by name
    for bad in ({"footprint": (8, 8, 28, 28), "except": 13},
                {"footprint": (8, 8, 28, 28), "except": (30,)},
                {"footprint": (8, 8, 28, 28), "except": ("13",)}):
        try:
            pipeline.read_needs({"NEEDS": bad})
        except ValueError as e:
            assert "except" in str(e), e
        else:
            raise AssertionError(f"{bad} was not refused")
    # ...and an edge's band is untouched
    erows = [{"size": [w, r], "of": 2, "failed": 0, "first": None}
             for w in tn.EDGE_WIDTHS for r in tn.EDGE_RUNS]
    assert tn.band("edge", erows)["footprint"] == [1, 4, 3, 128]
    assert 32 in tn.PLOT_SIZES and 48 in tn.AREA_SIZES
    return ("a type clean at 8-12 and 19-28 declares 8-28 except 13-18; the plan "
            "refuses 15x15 and 24x13 by name and admits 24x24; a plot's exception is "
            "said in plots; three malformed declarations are refused; the plot sweep "
            f"reaches {max(tn.PLOT_SIZES)}")


# ------------------------------------------------------------------ M8. built

FIXTURE = os.path.join("checker_fixtures", "monument_compound")


@case
def t_m8_a_compound_on_its_plateau_is_built_walkable_and_lint_clean_and_larger_than_a_house():
    """The spec's own case: a monument defining part plans into its parts, every part
    sited on its prepared ground, walkable, lint-clean, and larger than a single house
    by the registered margin -- through the same stages a city runs, offline."""
    from ethoslm.pipeline import stages_measure
    state = os.path.join(ROOT, "out", FIXTURE)
    shutil.rmtree(state, ignore_errors=True)
    os.makedirs(state, exist_ok=True)
    spec, decls, place, plan = _assembled()
    # the ground, with the plateau the search would have cut for the palace
    vol = _ground()
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    rec = b.plateau(tuple(PLATEAU["rect"]), PLATEAU["y"], mat=pipeline.voice_palette(VOICE),
                    label="palace")
    assert rec["ok"], rec
    b.resolve_steps()
    vol.overlay(b._pending)
    offline.save_volume(vol, os.path.join(state, "world.npz"))
    json.dump({"part": "palace", "rect": PLATEAU["rect"], "plateau": rec},
              open(os.path.join(state, "plateau.json"), "w"))
    json.dump(SITE, open(os.path.join(state, "site.json"), "w"))
    json.dump(spec, open(os.path.join(state, "place.json"), "w"))
    rnd = pipeline.Round(name=FIXTURE, state_dir=state, sentence=SENTENCE,
                         site={"origin": [0, 0], "size": SIZE})
    parts = pipeline.plan_parts(plan)
    json.dump(plan, open(rnd.rel("plan.json"), "w"), indent=1)
    pipeline._write_registry(rnd, plan, parts, levels=plan["levels"], voice=VOICE)
    be = pipeline.OfflineBackend(rnd, dry_run=True)
    circ = pipeline.stage_circulation(rnd, be, {})
    assert circ.get("thresholds"), circ
    built = pipeline.stage_parts(rnd, be, {})
    rows = [r for w in built["waves"] for r in w["parts"]]
    not_up = [(r["part"], r["status"], r.get("error", r.get("type_said")))
              for r in rows if r["status"] != "built" or not r.get("stood")]
    assert not not_up, not_up
    wave_errors = {w["wave"]: w["errors"] for w in built["waves"] if w["errors"]}
    be.save(rnd.rel("world_built.npz"))
    own = stages_measure._m_own_lint_errors(rnd, be, {}, None)
    doors = stages_measure._m_e002_from_the_lane(rnd, be, {}, None)
    walk = stages_measure._m_walk_from_outdoors_pct(rnd, be, {}, None)
    read = placeread.read(spec, plan, built, voice=VOICE, site=SITE,
                          built=offline.load_volume(rnd.rel("world_built.npz")),
                          base=offline.load_volume(rnd.rel("world.npz")),
                          plateau=pipeline.plateau_record(rnd))
    clauses = {c["clause"]: c for c in read["clauses"]}
    grounds = {r["part"]: r.get("ground") for r in rows}
    assert not [n for n, g in grounds.items() if g == "deck"], grounds
    failed = [k for k, c in clauses.items() if not c["holds"]]
    assert not failed, {k: clauses[k]["says"] for k in failed}
    assert own["got"] == 0, own["per_wave"]
    assert doors["got"] == 0, doors["findings"]
    assert walk["got"] >= 90, walk
    scale = clauses["compound/palace/scale"]
    return (f"{len(rows)} parts stand -- {', '.join(sorted(r['part'] for r in rows if r.get('kind') != 'plot' or True))[:0]}"
            f"wall, gate, court and three halls on the plateau beside four houses; "
            f"0 own lint errors (wave errors {wave_errors or 'none'}), "
            f"{doors['got']} doors unreachable, {walk['got']}% of the interior floor "
            f"walkable from outdoors over {walk['floor_cells']} cells in "
            f"{walk['rooms']} rooms; the palace is {scale['footprint']} columns and "
            f"{scale['blocks']} blocks against a house of "
            f"{scale['largest_footprint'][1]} and {scale['largest_blocks'][1]}, "
            f"{scale['footprint'] / scale['largest_footprint'][1]:.0f}x and "
            f"{scale['blocks'] / max(1, scale['largest_blocks'][1]):.1f}x; every "
            f"clause of the place read holds")


# ----------------------------------------------------------------- the runner

# ------------------------------------ M9. the composition is the family's (v2, C0)

def _spec_with(part: dict):
    doc = json.loads(json.dumps(TOWN_SPEC))
    doc["defining_parts"][0] = part
    spec = spec_mod.read_spec(doc, SENTENCE)
    spec["structures"], spec["size_band"] = 4, [4, 12]
    return spec


def _place_for(name: str):
    place = _place()
    place["centre"] = name
    place["compounds"][0].update({"name": name, "defines": name,
                                  "notes": f"the {name} on its plateau"})
    vol = _ground()
    net, nodes = placeplan.plan_arterials(place, _decls(), _heights(vol), 0, 0)
    place["arterials"] = placeplan.arterial_record(net, nodes, place)
    return place


def _wall_gate():
    return [
        {"kind": "edge", "name": "curtain_wall", "type": "wall", "seed": 1,
         "params": {"height": 6, "width": 1, "crown": "solid"}, "width": 1,
         "path": [[67, 67], [124, 67], [124, 124], [67, 124], [67, 67]],
         "notes": "the wall"},
        {"kind": "point", "name": "gatehouse", "type": "ring_gate", "seed": 2,
         "params": {"storeys": 2, "crown": "hip"}, "at": [96, 67],
         "facing": "north", "size": 5, "notes": "the gate, where the road arrives"}]


COMPOSITIONS_FIXTURE = {
    # a monument: one thing and its setting -- no wall, no gate
    "monument": ({"name": "monument", "kind": "plot", "family": "monument",
                  "relation": "centre", "count": 1, "structures": 1,
                  "needs": {"plateau": 64}, "notes": "the monument and its precinct"},
                 [{"kind": "area", "name": "precinct", "type": "square", "seed": 3,
                   "params": {"paving": "banded", "canopy": "hip"},
                   "x0": 76, "z0": 70, "x1": 115, "z1": 95, "notes": "the precinct"},
                  {"kind": "plot", "name": "the_monument", "type": "hall", "seed": 4,
                   "params": {"dormers": 2, "use": "moot"},
                   "x0": 88, "z0": 102, "x1": 105, "z1": 117, "notes": "the thing"}]),
    # a shrine precinct: a group of the temple family -- the shrine and its garden
    "shrine": ({"name": "shrine", "kind": "group", "family": "temple",
                "relation": "centre", "count": 1, "structures": 2,
                "needs": {"plateau": 64}, "notes": "a shrine precinct"},
               [{"kind": "area", "name": "shrine_garden", "type": "garden", "seed": 3,
                 "params": {}, "x0": 72, "z0": 70, "x1": 119, "z1": 99,
                 "notes": "the garden before it"},
                {"kind": "plot", "name": "main_shrine", "type": "temple", "seed": 4,
                 "params": {}, "x0": 78, "z0": 104, "x1": 93, "z1": 119,
                 "notes": "the shrine"},
                {"kind": "plot", "name": "lesser_shrine", "type": "temple", "seed": 5,
                 "params": {}, "x0": 100, "z0": 104, "x1": 115, "z1": 119,
                 "notes": "a lesser shrine"}]),
    # a castle: a group of the keep family -- the keep inside its curtain wall, a
    # gatehouse on it, and the bailey
    "castle": ({"name": "castle", "kind": "group", "family": "keep",
                "relation": "centre", "count": 1, "structures": 3,
                "needs": {"plateau": 64}, "notes": "a castle"},
               [*_wall_gate(),
                {"kind": "area", "name": "bailey", "type": "square", "seed": 3,
                 "params": {"paving": "banded", "canopy": "hip"},
                 "x0": 84, "z0": 72, "x1": 107, "z1": 90, "notes": "the bailey"},
                {"kind": "plot", "name": "the_keep", "type": "keep", "seed": 4,
                 "params": {}, "x0": 84, "z0": 94, "x1": 107, "z1": 117,
                 "notes": "the keep"}]),
}


@case
def t_m9_a_monument_a_shrine_precinct_and_a_castle_are_each_their_own_composition():
    said = []
    for label, (part, parts) in COMPOSITIONS_FIXTURE.items():
        spec = _spec_with(part)
        d = spec_mod.compounds(spec)[0]
        assert d["name"] == part["name"], d
        made_of = placeplan.compound_composition(d, spec=spec)
        place = _place_for(part["name"])
        comp = place["compounds"][0]
        _t, cdecls = placeplan.compound_types(None, spec, d)
        got = {"notes": label, "axis": "north", "parts": parts}
        fails = placeplan.compound_failures(comp, got, place, cdecls, spec=spec)
        assert not fails, (label, [(f["part"], f["check"], f["why"]) for f in fails])
        # ...assembled and read: `present/<name>` holds
        plan = placeplan.assemble(place, {"houses": _district()}, spec,
                                  {part["name"]: got})
        leaves = pipeline.plan_parts(plan)
        mine = [p for p in leaves if p.get("compound") == part["name"]]
        assert all(p.get("admits") == list(made_of["admits"]) for p in mine), \
            [(p["name"], p.get("admits")) for p in mine]
        whole = pipeline.plan_failures(
            leaves, pipeline.type_declarations(leaves),
            ground={p["name"]: {"relief": 2, "water_pct": 0.0, "class": "dry",
                                "columns": 1} for p in leaves})
        assert not whole, (label, whole)
        read = placeread.read(spec, plan, _stood(plan), voice=VOICE, site=SITE,
                              plateau=dict(PLATEAU, part=part["name"]))
        c = {x["clause"]: x for x in read["clauses"]}
        assert c[f"present/{part['name']}"]["holds"], c[f"present/{part['name']}"]["says"]
        # ...and the brief says what this one is made of
        brief = placeplan.compound_brief(spec, SITE, comp, place, "/tmp/x.json", None,
                                         VOICE)
        t = placeplan.compound_target(comp, spec)
        if made_of["walled"]:
            assert "One closed wall of its own" in brief, label
        else:
            assert f"No wall is asked of a {d['family']}" in brief, label
            assert "closed wall of its own" not in brief, label
        said.append(f"{label} ({d['family']}): {t['count']} hall(s), {t['courts']} "
                    f"court(s), {'walled' if t['walled'] else 'unwalled'}, "
                    f"{'gated' if t['gated'] else 'ungated'}, admits "
                    f"{list(made_of['admits']) or 'nothing'}: passes, assembles, "
                    f"present/ holds")

    def why(label, parts, **place_over):
        part = COMPOSITIONS_FIXTURE[label][0]
        spec = _spec_with(part)
        d = spec_mod.compounds(spec)[0]
        place = _place_for(part["name"])
        _t, cdecls = placeplan.compound_types(None, spec, d)
        got = {"notes": label, "axis": "north", "parts": parts}
        return [(f["part"], f["check"]) for f in
                placeplan.compound_failures(place["compounds"][0], got, place, cdecls,
                                            spec=spec)]

    # a castle without its wall is refused by name; without its gate too
    castle = COMPOSITIONS_FIXTURE["castle"][1]
    got = why("castle", [p for p in castle if p["kind"] not in ("edge", "point")])
    assert ("castle", "wall") in got, got
    got = why("castle", [p for p in castle if p["kind"] != "point"])
    assert ("castle", "gate") in got, got
    # a monument with an open wall drawn is refused for the wall it drew, and with a
    # closed one it is not asked for a gate
    monument = COMPOSITIONS_FIXTURE["monument"][1]
    open_wall = dict(_wall_gate()[0], path=[[67, 67], [124, 67], [124, 124], [67, 124]])
    got = why("monument", [open_wall, *monument])
    assert ("curtain_wall", "wall") in got, got
    got = why("monument", [_wall_gate()[0], *monument])
    assert not any(c in ("wall", "gate") for _n, c in got), got
    # a gate in a shrine, and a keep in one, are refused by role, because an unwalled
    # precinct admits nothing over its own: the defensive types are a walled compound's
    shrine = COMPOSITIONS_FIXTURE["shrine"][1]
    got = why("shrine", [_wall_gate()[1], *shrine])
    assert ("gatehouse", "role") in got, got
    keep = {"kind": "plot", "name": "a_keep", "type": "keep", "seed": 9, "params": {},
            "x0": 100, "z0": 104, "x1": 115, "z1": 119, "notes": ""}
    got = why("shrine", [*shrine[:2], keep])
    assert ("a_keep", "role") in got, got
    # ...and the keep in the castle is admitted, as is the wall
    assert not why("castle", castle)
    # the ground: a monument needs less than a palace, a castle as much
    pal = placeplan.compound_ground(spec=_spec(), site_side=SIZE)
    mon = placeplan.compound_ground(spec=_spec_with(COMPOSITIONS_FIXTURE["monument"][0]),
                                    site_side=SIZE)
    cas = placeplan.compound_ground(spec=_spec_with(COMPOSITIONS_FIXTURE["castle"][0]),
                                    site_side=SIZE)
    assert mon["side"] < pal["side"] and not mon["walled"] and pal["walled"], (mon, pal)
    assert cas["side"] == pal["side"] and cas["walled"], (cas, pal)
    # a monument does not scale with its rectangle; a palace does
    big = {"name": "b", "x0": 0, "z0": 0, "x1": 199, "z1": 199}
    m_t = placeplan.compound_target(dict(big, defines="monument"),
                                    _spec_with(COMPOSITIONS_FIXTURE["monument"][0]))
    p_t = placeplan.compound_target(dict(big, defines="palace"), _spec())
    assert m_t["count"] == 1 and not m_t["scales"], m_t
    assert p_t["count"] > placeplan.COMPOUND_MIN_HALLS and p_t["scales"], p_t
    # the composition table is the spec's and a family not in it is the palace's
    assert spec_mod.composition("bridge") == dict(spec_mod.COMPOSITION_DEFAULT,
                                                   family="bridge")
    return "; ".join(said) + (f"; a castle without its wall or gate, a monument with an "
                              f"open wall, and a gate or a keep in a shrine are refused "
                              f"by name; a monument's ground "
                              f"{mon['side']} against a palace's {pal['side']}; a "
                              f"monument on 200 square is still 1 hall, a palace "
                              f"{p_t['count']}")


def main():
    ok = bad = skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok} of {ok + bad} cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
