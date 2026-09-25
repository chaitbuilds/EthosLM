#!/usr/bin/env python3
"""**One compatible spatial solution, chosen by the compiler and consumed verbatim.**

    $PY scripts/test_block_site.py [name-fragment]

The quarter design round, block/building worker. The block design audit's second cause:
`pad_founded` and `entrance_ok` answered "is there one" and every later stage chose the
pad, the floor, the facing, the door and the landing again. These are the counterexamples
at the boundaries the round changed; nothing here needs a cached world or a model call.

    s1  every plot leaf of a district on measured ground carries one `site`: the pad the
        builder's own inset gives for its settled attachment, the district level, the
        street side, a door on the pad's street edge and a landing one column outside
        the plot -- and a court carries `court_site` whose passage shares columns with
        the court and ends at it.
    s2  a pad and a door are a *pair*: ground the design refuses along the middle of a
        lot's street edge moves the door to prepared columns or refuses the lot; no
        site's door line crosses a refused column.
    s3  a lot whose own envelope admits one storey of a type whose floor is two is
        re-typed to another type of its use or refused -- never emitted at two.
    s4  the generated production call builds exactly the compiled pad, floor, facing and
        door (off the middle of the pad, where the old path would have put it), and a
        compiled floor the ground contract cannot hold within a step is refused
        visibly (`SiteRefused`), not sunk.
    s5  the router approaches a sited leaf at its landing and reserves the compiled door;
        a court's margin is an obstacle and its passage is its way in.
    s6  `usable.court_enclosed` on a synthetic assembled block: holds for a court whose
        named ranges stand round its owned margin and whose passage is walked into it;
        fails when a face's mass belongs to another part, when the passage shares no
        column with the court, and when the passage is blocked.
    s7  a leaf without a site (every older plan) is sited exactly as before.
    s8  a court's margin is laid by `site()` as the court's own ground -- a walk round the
        paving and through the passage's columns, the rest planted -- and the court's
        door is its compiled edge cell.
    s10 a compiled doorway has its floor: with the network reserving the compiled door,
        the door cell at the floor is solid and the leaf stands on it (qd-city E008).
    s9  shop_house, court_large, courtyard_house and row_house hang their door leaf on
        the compiled door.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (circulate, district_compile as dc, feasible, lint,  # noqa: E402
                   offline, pipeline, placeplan, stages, usable)
from ethoslm.buildlib import (Builder, SiteRefused, WALK_IN,  # noqa: E402
                            site_pad_rect)
from ethoslm.observe import Volume  # noqa: E402

CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------ a district on measured
# ground

FORM = "east_asian"
RECT = (0, 0, 119, 79)
LEVEL = 64


def _part(character=None, structures=40, density="dense"):
    return {"name": "houses", "kind": "defining_part", "density": density,
            "role": "urban", "structures": structures,
            "character": dict(character or {})}


def _ground(rect, refuse=None):
    """A measured ground record over `rect`, all prepared except the cells `refuse`
    names (a predicate on world (x, z))."""
    x0, z0, x1, z1 = rect
    w, d = x1 - x0 + 1, z1 - z0 + 1
    mask = np.ones((w, d), bool)
    if refuse:
        for i in range(w):
            for j in range(d):
                if refuse(x0 + i, z0 + j):
                    mask[i, j] = False
    return {"measured": True, "level": LEVEL, "origin": [x0, z0],
            "mask_bits": feasible.pack(mask, (x0, z0)),
            "columns": int(w * d), "feasible_columns": int(mask.sum())}


def _compile(character=None, refuse=None, rect=RECT, structures=40):
    p = _part(character, structures)
    spec = {"sentence": "a quarter", "form": FORM, "structures": structures,
            "defining_parts": [p], "voice": "japanese_minka"}
    d = {"name": "houses_1", "defines": "houses", "kind": "district",
         "x0": rect[0], "z0": rect[1], "x1": rect[2], "z1": rect[3],
         "structures": structures, "density": "dense", "role": "urban",
         "ground": _ground(rect, refuse)}
    place = {"districts": [d], "layout": {"seed": 1}, "structures": structures}
    _t, decls = placeplan.types_card(None, FORM, "urban")
    got, rec = dc.compile_district(d, p, place, decls, spec=spec, seed=1)
    return got, rec, d, decls


def _leaves(got):
    return [q for r in (got.get("quarters") or []) for q in (r.get("plots") or [])]


_OUT = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


def _site_ok(p) -> str | None:
    """None, or why this leaf's site is not the one compatible solution."""
    st = p.get("site")
    if not st:
        return "no site"
    pad = list(site_pad_rect(p["x0"], p["z0"], p["x1"], p["z1"], p.get("attached") or []))
    sp = st["pad"]
    if not (pad[0] <= sp[0] <= sp[2] <= pad[2] and pad[1] <= sp[1] <= sp[3] <= pad[3]):
        return f"pad {sp} is not inside the inset plot {pad}"
    if st["floor"] != LEVEL:
        return f"floor {st['floor']} is not the district level {LEVEL}"
    side = st["facing"]
    if p.get("front") and side != p["front"]:
        return f"faces {side} and the leaf fronts {p['front']}"
    dx, dz = st["door"]
    edge = {"north": sp[1], "south": sp[3], "west": sp[0], "east": sp[2]}[side]
    on = (dz == edge and sp[0] < dx < sp[2]) if side in ("north", "south") else \
        (dx == edge and sp[1] < dz < sp[3])
    if not on:
        return f"door {st['door']} is not off-corner on the pad's {side} edge {sp}"
    lx, lz = st["landing"]
    want = {"north": (dx, p["z0"] - 1), "south": (dx, p["z1"] + 1),
            "west": (p["x0"] - 1, dz), "east": (p["x1"] + 1, dz)}[side]
    if (lx, lz) != want:
        return f"landing {st['landing']} is not the cell outside the plot at {want}"
    return None


@case
def s1_every_plot_carries_one_site_and_a_court_its_passage():
    got, rec, d, decls = _compile({"attached": True, "courtyard_share": 0.05})
    plots = [p for p in _leaves(got) if p.get("kind", "plot") == "plot"]
    assert plots, "the fixture laid nothing"
    bad = [(p["name"], _site_ok(p)) for p in plots if _site_ok(p)]
    assert not bad, f"{len(bad)} of {len(plots)} sites are not compatible: {bad[:4]}"
    # the pad is the builder's own: an attached middle lot reaches its party walls
    mids = [p for p in plots if len(p.get("attached") or ()) == 2]
    assert mids, "no lot with both flanks attached"
    m = mids[0]
    assert m["site"]["pad"][0] == m["x0"] and m["site"]["pad"][2] == m["x1"] or \
        m["site"]["pad"][1] == m["z0"] and m["site"]["pad"][3] == m["z1"], m
    courts = [p for p in _leaves(got) if p.get("court_site")]
    assert courts, f"no court carries a court_site: {rec.get('compositions')}"
    for c in courts:
        cs = c["court_site"]
        cr, pa = cs["court"], cs["passage"]
        along_x = cs["street_side"] in ("north", "south")
        lo, hi = ((max(cr[0], pa[0]), min(cr[2], pa[2])) if along_x
                  else (max(cr[1], pa[1]), min(cr[3], pa[3])))
        assert hi >= lo, f"{c['name']}: passage {pa} shares no column with court {cr}"
        mg = cs["margin"]
        assert mg[0] <= cr[0] and mg[1] <= cr[1] and mg[2] >= cr[2] and mg[3] >= cr[3]
        assert cs["door"][0] in range(cr[0], cr[2] + 1) and \
            cs["door"][1] in range(cr[1], cr[3] + 1), cs
        assert set(cs["ranges"]) and set(cs["ranges"]) <= {p["name"] for p in plots}
        assert c["enclosure"]["passage"] == pa and c["enclosure"]["margin"] == mg
    assert rec["sites"]["sited"] == len(plots), rec["sites"]
    return (f"{len(plots)} plot(s), every one with one compatible site "
            f"({len(mids)} with both party walls reaching their lot line); "
            f"{len(courts)} court(s) with a passage sharing "
            f"{hi - lo + 1} column(s) with the court and a margin "
            f"{mg[2] - mg[0] + 1}x{mg[3] - mg[1] + 1} round a "
            f"{cr[2] - cr[0] + 1}x{cr[3] - cr[1] + 1} court")


@case
def s2_a_pad_and_a_door_are_a_pair():
    # refuse every column whose x is 3 mod 13 -- a stripe through the middle of many
    # street edges, and the lanes between them
    def refuse(x, z):
        return x % 13 == 6
    got, rec, d, decls = _compile({}, refuse=refuse)
    plots = [p for p in _leaves(got) if p.get("kind", "plot") == "plot"]
    assert plots, "the fixture laid nothing"
    crossed = []
    for p in plots:
        st = p["site"]
        sp = st["pad"]
        cells = [(x, z) for x in range(sp[0], sp[2] + 1) for z in range(sp[1], sp[3] + 1)]
        dx, dz = st["door"]
        lx, lz = st["landing"]
        line = [(dx, z) for z in range(min(dz, lz), max(dz, lz) + 1)] if dx == lx else \
            [(x, dz) for x in range(min(dx, lx), max(dx, lx) + 1)]
        if any(refuse(*c) for c in cells + line):
            crossed.append(p["name"])
    assert not crossed, f"sites on refused ground: {crossed[:5]}"
    shrunk = sum(1 for p in plots if "prepared part" in p["site"]["why"])
    return (f"{len(plots)} sited lot(s) on a striped mask: no pad and no door line "
            f"crosses a refused column; {shrunk} pad(s) cut to the prepared part of the "
            f"the street; drops {rec['dropped']}, refused after drawing "
            f"{rec['sites']['refused']}")


@case
def s3_a_one_storey_admission_under_a_two_storey_floor_is_not_emitted():
    """The shop house of `middle_ring_north_east_b2_0_00`, made deterministic: every
    type's band is raised to a floor of two and the envelope is made to answer one for
    the fabric's own house type. The leaf must come back re-typed to a type of the same
    use that admits its floor, or refused with the demand returned -- never at two
    beside an admission of one."""
    real_fit, real_band = dc._storeys_fit, dc._storeys_band
    victim = {}

    def band(decl, ch):
        lo, hi = real_band(decl, ch)
        return (max(lo, 2), max(hi, 2))

    def fit(tname, lo, hi, w, d, cache=None, demand=None, flanks=0):
        if tname == victim.get("type"):
            return {"storeys": 1, "holds": False, "required": False, "type": tname,
                    "why": f"a {w}x{d} lot admits 1 storey of {tname} (test)"}
        return {"storeys": int(hi), "holds": True, "required": False, "type": tname,
                "why": f"a {w}x{d} lot admits {hi} of {tname} (test)"}
    try:
        dc._storeys_band = band
        dc._storeys_fit = fit
        got0, rec0, _d, decls = _compile({})
        victim["type"] = rec0["house"]
        got, rec, _d, decls = _compile({})
    finally:
        dc._storeys_fit, dc._storeys_band = real_fit, real_band
    plots = [p for p in _leaves(got) if p.get("kind", "plot") == "plot"]
    bad = [p["name"] for p in plots
           if (p.get("envelope") or {}).get("storeys_admitted") is not None
           and int(p["envelope"]["storeys_admitted"])
           < int((p["envelope"].get("storeys_band") or [0])[0])]
    assert not bad, f"leaves emitted over their admission: {bad[:5]}"
    at_victim = [p["name"] for p in plots if p.get("type") == victim["type"]]
    assert not at_victim, f"{victim['type']} still emitted: {at_victim[:5]}"
    moved = rec["sites"]["retyped"]
    refused = rec["sites"]["lots_refused"]
    assert moved or refused, "nothing was re-typed or refused"
    if refused:
        assert any(x.get("what") == "storeys" for x in rec["demand_short"]), \
            "a storeys refusal was not returned to the district as demand"
    return (f"{victim['type']} admitting 1 under a floor of 2: {len(moved)} lot(s) "
            f"re-typed ({sorted({m['to'] for m in moved})}), {len(refused)} refused and "
            f"returned as demand; none emitted at storeys it does not admit")


# ------------------------------------------------------------ the generated production
# call

MAT = {"wall": "mud_brick", "footing": "cobblestone", "frame": "spruce",
       "roof": "deepslate_tile", "trim": "spruce", "floor": "packed_mud"}
ROOF = {"profile": [[1, 1]], "ends": "hip", "eave": "straight", "tiers": 1,
        "overhang": 1, "chimney": False}
GROUND = 64


def _flat(size: int, y: int = GROUND) -> Volume:
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


def _network(leaves):
    """A lane along each leaf's front with the doorstep the old pass would reserve:
    **at the middle of the plot's edge**, so a built door anywhere else is the site's."""
    from ethoslm.circulate import Network, Threshold
    cells, ths = {}, []
    for leaf in leaves:
        x0, z0, x1, z1 = leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"]
        mx = (x0 + x1) // 2
        lz = z0 - 1
        for x in range(x0 - 2, x1 + 3):
            for k in (0, 1):
                cells[(x, lz - k)] = {"y": GROUND, "rank": 1, "face": None}
        ths.append(Threshold(id=leaf["label"], x=mx, z=lz, y=GROUND, facing="south",
                             door=(mx, GROUND + 1, z0)))
    return Network(cells=cells, thresholds=ths)


def _network_compiled(leaves):
    """The network the router now plans for sited leaves (`circulate.site_way`): a lane
    along the street, the threshold at the compiled landing and its door the compiled
    door on the pad's edge at floor+1."""
    from ethoslm.circulate import Network, Threshold
    cells, ths = {}, []
    for leaf in leaves:
        st = leaf["site"]
        lx, lz = st["landing"]
        for x in range(leaf["x0"] - 2, leaf["x1"] + 3):
            for k in (0, 1):
                cells[(x, lz - k)] = {"y": GROUND, "rank": 1, "face": None}
        ths.append(Threshold(id=leaf["label"], x=lx, z=lz, y=GROUND,
                             facing=WALK_IN[st["facing"]],
                             door=(st["door"][0], int(st["floor"]) + 1, st["door"][1])))
    return Network(cells=cells, thresholds=ths)


def _run(leaves, type_name="row_house", params=None, network=None):
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{type_name}.py"))
    inst = [({k: leaf[k] for k in pipeline.PART_GEOMETRY if k in leaf}, 3,
             dict(params or {"storeys": 1})) for leaf in leaves]
    src = pipeline.instantiated_source(decl["src"], inst, mat=MAT, roof=ROOF)
    vol = _flat(64)
    with tempfile.TemporaryDirectory() as tmp:
        json.dump([{"label": lf["label"], "x0": lf["x0"], "z0": lf["z0"],
                    "x1": lf["x1"], "z1": lf["z1"]} for lf in leaves],
                  open(os.path.join(tmp, "plots.json"), "w"))
        prog = os.path.join(tmp, "program.py")
        open(prog, "w").write(src)
        b = offline.run_program(prog, vol, network=(network or _network)(leaves),
                                plots=stages._Registry(tmp), src=src)
    return b


def _row_leaf(label, x0, attached, door_dx, floor=GROUND):
    lf = {"label": label, "kind": "plot", "x0": x0, "z0": 20, "x1": x0 + 5, "z1": 32,
          "front": "north", "attached": list(attached), "type": "row_house"}
    pad = list(site_pad_rect(lf["x0"], lf["z0"], lf["x1"], lf["z1"], attached))
    lf["site"] = {"pad": pad, "floor": int(floor), "facing": "north",
                  "door": [pad[0] + door_dx, pad[1]], "landing": [pad[0] + door_dx, 19],
                  "attached": sorted(attached), "why": "test"}
    return lf


@case
def s4_the_production_call_builds_the_compiled_site():
    leaves = [_row_leaf("a", 10, ["east"], 1), _row_leaf("b", 16, ["west", "east"], 4),
              _row_leaf("c", 22, ["west"], 3)]
    b = _run(leaves)
    sited = {p["label"]: p for p in b.parts}
    wrong = []
    for lf in leaves:
        s, st = sited[lf["label"]], lf["site"]
        if s["footprint"] != st["pad"]:
            wrong.append((lf["label"], "pad", s["footprint"], st["pad"]))
        if s["floor_y"] != st["floor"]:
            wrong.append((lf["label"], "floor", s["floor_y"], st["floor"]))
        if list(s["door"]) != st["door"]:
            wrong.append((lf["label"], "door", s["door"], st["door"]))
        if s["facing"] != WALK_IN[st["facing"]]:
            wrong.append((lf["label"], "facing", s["facing"]))
        if not (s["sited"].get("site") or {}).get("door_on_pad_edge"):
            wrong.append((lf["label"], "door off the pad edge"))
        # the type's own door leaf is the compiled door, not the middle of the lot
        dx, dz = st["door"]
        leaf_here = [p for p, blk in b._pending.items()
                     if p[0] == dx and p[2] == dz and p[1] == st["floor"] + 1
                     and blk.split("[")[0].endswith("_door")]
        if not leaf_here:
            wrong.append((lf["label"], "no door leaf at the compiled door", st["door"]))
    assert not wrong, wrong
    # the old path would have put b's door at the network's middle cell
    assert leaves[1]["site"]["door"][0] != (leaves[1]["x0"] + leaves[1]["x1"]) // 2
    # a compiled floor six above flat ground is refused, not sunk to the contract's
    high = [_row_leaf("h", 10, [], 2, floor=GROUND + 6)]
    try:
        _run(high)
    except SiteRefused as e:
        said = str(e)
    else:
        raise AssertionError("a floor the contract holds 3 blocks off was built")
    assert "Refused rather than built" in said, said
    return (f"3 row houses built through the generated production call on exactly "
            f"their compiled pads, floors and doors (door columns "
            f"{[lf['site']['door'][0] for lf in leaves]}, the network's middle cells "
            f"{[(lf['x0'] + lf['x1']) // 2 for lf in leaves]}); a floor 6 above its "
            f"ground raised SiteRefused")


@case
def s5_the_router_takes_the_compiled_landing_and_door():
    plot = _row_leaf("p", 20, [], 2)
    court = {"label": "c", "name": "c", "kind": "area", "type": "plaza",
             "x0": 40, "z0": 30, "x1": 47, "z1": 37,
             "court_site": {"court": [40, 30, 47, 37], "margin": [37, 27, 50, 40],
                            "passage": [41, 21, 45, 29], "landing": [43, 20],
                            "approach": [43, 29], "door": [43, 30], "floor": GROUND,
                            "street_side": "north", "ranges": []}}
    parts = [dict(plot, name="p"), court]
    routing = circulate.parts_to_routing(parts)
    assert (38, 28) in routing["obstacles"], "the margin is not an obstacle"
    assert (43, 28) not in routing["obstacles"], "the passage through the margin is shut"
    h = np.full((64, 64), GROUND, np.int32)
    avoid = np.zeros((64, 64))
    for (x, z) in routing["obstacles"]:
        avoid[x, z] = np.inf
    net = circulate.plan_network(h, 0, 0, routing["sites"], avoid_extra=avoid)
    tp, tc = net.threshold("p"), net.threshold("c")
    st = plot["site"]
    assert (tp.x, tp.z) == tuple(st["landing"]), (tp, st)
    assert list(tp.door) == [st["door"][0], GROUND + 1, st["door"][1]], tp
    assert tp.facing == "south"
    assert (tc.x, tc.z) == (43, 29) and list(tc.door) == [43, GROUND + 1, 30], tc
    assert net.notes.get("compiled_landings") == 2, net.notes
    return (f"the plot is approached at its landing {st['landing']} with its compiled "
            f"door {list(tp.door)}; the court at the inner end of its passage with its "
            f"edge cell as the door; the margin is {len(routing['obstacles'])} obstacle "
            f"column(s) with the passage left open")


# ------------------------------------------------------------ court_enclosed, assembled

def _block_world(*, foreign_east=False, passage=(26, 14, 30, 23), block_passage=False):
    """A court (24..31, 24..31) with its margin (21..34), four ranges of lots round it
    and a passage through the north range; each range's pad (inset 2) stands as mass."""
    vol = _flat(64)
    lots = {"d_b0_0_00": (12, 14, 25, 20), "d_b0_0_01": (31, 14, 43, 20),
            "d_b0_0_10": (12, 35, 43, 41), "d_x0_0_00": (14, 21, 20, 34),
            "d_x0_0_10": (35, 21, 41, 34)}
    if foreign_east:
        lots["other_part"] = lots.pop("d_x0_0_10")
    stone = vol.palette.index("stone")
    for (x0, z0, x1, z1) in lots.values():
        vol.codes[x0 + 2:x1 - 1, GROUND + 1 - vol.y0:GROUND + 4 - vol.y0,
                  z0 + 2:z1 - 1] = stone
    if block_passage:
        vol.codes[26:31, GROUND + 1 - vol.y0:GROUND + 3 - vol.y0, 18] = stone
    reg = [{"label": n, "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]}
           for n, r in lots.items()]
    reg.append({"label": "d_pc0_0_0", "x0": 24, "z0": 24, "x1": 31, "z1": 31,
                "kind": "area"})
    ctx = lint.Context.build(vol, plots=reg, region=(0, 0, 63, 63))
    rows = [{"part": r["label"], "status": "built", "stood": True} for r in reg]
    world = usable.World(ctx, rows)
    claim = {"district": "d", "court": [24, 24, 31, 31], "margin": [21, 21, 34, 34],
             "passage": list(passage), "street_side": "north", "gap_bar": 4,
             "entry_bar": 5,
             "ranges": ["b0_0_00", "b0_0_01", "b0_0_10", "x0_0_00", "x0_0_10"]}
    row = {"part": "d_pc0_0_0", "floor_y": GROUND, "status": "built",
           "enclosure": claim}
    return usable.court_enclosed(world, "d_pc0_0_0", row, world.provenance())


@case
def s6_court_enclosed_needs_its_own_ranges_and_a_passage_into_it():
    ok = _block_world()
    assert ok["holds"] is True, ok["why"]
    assert ok["evidence"]["passage"]["established"], ok["evidence"]["passage"]
    foreign = _block_world(foreign_east=True)
    assert foreign["holds"] is False, foreign["why"]
    assert "other_part" in foreign["evidence"]["foreign_mass"], foreign["evidence"]
    missed = _block_world(passage=(14, 14, 18, 23))
    assert missed["holds"] is False and "shares no column" in missed["why"], missed["why"]
    blocked = _block_world(block_passage=True)
    assert blocked["holds"] is False and "cannot be walked" in blocked["why"], \
        blocked["why"]
    return (f"holds with its named ranges and a walked passage "
            f"({ok['evidence']['passage']['why']}); fails with the east range's mass "
            f"another part's ({foreign['evidence']['foreign_mass']}), with a passage "
            f"that shares no column, and "
            f"with the passage blocked")


@case
def s7_a_leaf_without_a_site_is_sited_as_before():
    lf = {"label": "old", "kind": "plot", "x0": 10, "z0": 20, "x1": 15, "z1": 32,
          "front": "north", "attached": ["west", "east"], "type": "row_house"}
    b = _run([lf])
    s = b.parts[-1]
    pad = list(site_pad_rect(10, 20, 15, 32, ["west", "east"]))
    assert s["footprint"] == pad, (s["footprint"], pad)
    assert "site" not in (s.get("sited") or {}), s["sited"]
    # the old path takes the door column from the network's reserved doorstep
    assert s["door"][0] == (10 + 15) // 2, s["door"]
    return (f"a leaf with no site is inset by the one shared definition to {pad} and "
            f"takes its door from the reserved doorstep, as before")


@case
def s8_a_courts_margin_is_laid_as_its_own_ground():
    """The court's owned ring is real blocks the court lays: a paved walk round the
    paving and through the passage's columns, the rest planted -- and its door is the
    compiled edge cell, not the rectangle's middle."""
    court = {"label": "c", "kind": "area", "type": "plaza",
             "x0": 24, "z0": 24, "x1": 31, "z1": 31,
             "court_site": {"court": [24, 24, 31, 31], "margin": [21, 21, 34, 34],
                            "passage": [25, 14, 29, 23], "landing": [27, 13],
                            "approach": [27, 23], "door": [27, 24], "floor": GROUND,
                            "street_side": "north", "ranges": []}}
    b = _run([court], type_name="plaza", params={})
    s = b.parts[-1]
    got = (s.get("sited") or {}).get("margin")
    assert got and got["walk"] > 0 and got["planted"] > 0, s.get("sited")
    assert list(s["door"]) == [27, 24] if s.get("door") else True
    pend = b._pending
    ring_top = {(x, z): pend.get((x, GROUND, z)) for x in range(21, 35)
                for z in range(21, 35) if not (24 <= x <= 31 and 24 <= z <= 31)}
    laid = sum(1 for v in ring_top.values() if v)
    assert laid >= 0.9 * len(ring_top), f"{laid} of {len(ring_top)} ring columns laid"
    paths = [p for p in b.paths if p.get("kind") == "margin"]
    assert paths and [27, 21] in paths[0]["cells"], "the passage line is not walked"
    entry = b.floor_from_threshold("c")
    assert list(entry["door"])[::2] == [27, 24], entry
    return (f"the margin 14x14 round an 8x8 court laid as {got['walk']} walk and "
            f"{got['planted']} planted column(s), owned by the court; its door is the "
            f"compiled edge cell (27, 24)")


@case
def s9_the_types_hang_their_door_on_the_compiled_door():
    """`row_house._front_edge`, `shop_house._frame`, `court_large`'s gate side and
    `courtyard_house`'s door all read what `site()` hands them; with a compiled site
    that is the compiled door, off the middle of the pad."""
    out = []
    for tn, params in (("shop_house", {"storeys": 2, "trade": "tea"}),
                       ("court_large", {"storeys": 1, "yard": "well"}),
                       ("courtyard_house", {"storeys": 1}),
                       ("row_house", {"storeys": 1})):
        lf = {"label": "L", "kind": "plot", "x0": 10, "z0": 20, "x1": 23, "z1": 33,
              "front": "north", "attached": [], "type": tn}
        pad = list(site_pad_rect(10, 20, 23, 33, []))
        door = [pad[0] + 3, pad[1]]
        lf["site"] = {"pad": pad, "floor": GROUND, "facing": "north", "door": door,
                      "landing": [door[0], 19], "attached": [], "why": "test"}
        b = _run([lf], type_name=tn, params=params)
        leaves = sorted({(p[0], p[2]) for p, blk in b._pending.items()
                         if blk.split("[")[0].endswith("_door")
                         and p[1] == GROUND + 1})
        assert (door[0], door[1]) in leaves, (tn, door, leaves)
        out.append(f"{tn} at {tuple(door)}")
    return "door leaves on the compiled door (pad middle x=16): " + ", ".join(out)


@case
def s10_a_compiled_doorway_has_its_floor_under_the_door():
    """`out/qd-city`: seven sited court houses stood with air at floor level under their
    door. The router reserves the compiled door on the pad's edge, `_site_lanes` counted
    every threshold door as a lane column, and `_site_lay` left that column unlaid -- so
    `clear_ground_cover`'s air stayed and the type's doorstep put-back restored it. Built
    here with the network the router now plans: the door cell has a solid floor."""
    out = []
    for tn, params in (("court_large", {"storeys": 1, "yard": "well"}),
                       ("court_small", {"storeys": 1}),
                       ("courtyard_house", {"storeys": 1})):
        lf = {"label": "L", "kind": "plot", "x0": 10, "z0": 20, "x1": 22, "z1": 32,
              "front": "north", "attached": [], "type": tn}
        pad = list(site_pad_rect(10, 20, 22, 32, []))
        door = [pad[0] + 4, pad[1]]
        lf["site"] = {"pad": pad, "floor": GROUND, "facing": "north", "door": door,
                      "landing": [door[0], 19], "attached": [], "why": "test"}
        b = _run([lf], type_name=tn, params=params, network=_network_compiled)
        under = b._pending.get((door[0], GROUND, door[1]))
        assert under and under.split("[")[0] not in ("air", "cave_air"), (tn, under)
        leaf = b._pending.get((door[0], GROUND + 1, door[1])) or ""
        assert leaf.split("[")[0].endswith("_door"), (tn, leaf)
        out.append(f"{tn}: {under.split('[')[0]} under the door")
    return "; ".join(out)


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__
        if only and only not in name:
            continue
        try:
            said = fn()
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        except Exception as e:                   # noqa: BLE001 -- reported
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} block site cases pass ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
