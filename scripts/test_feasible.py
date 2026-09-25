#!/usr/bin/env python3
"""`ethoslm.feasible`: which columns construction can stand on, and why not, where not.

The neighbourhood review's first remaining cause is that *"a district's developable
ground and the ground its count is derived from are still two different numbers"* --
`developable_columns` subtracts roads and standing parts but not water and not an
impossible grade, and `arrange.certificate_for` certifies with `ground=None`. These cases
establish that `feasible.terrain` answers that question, that each of its four clauses is
separately measurable (a counterexample and a positive control for each), and that its
answer survives being written onto a district record and read back.

    F1  flat dry ground at the design's own level is wholly feasible -- the control, so
        no later refusal can be dismissed as a mask that refuses everything.
    F2  a lake in the rectangle is refused as `wet`, at the column.
    F3  the same lake under a level above its waterline is **reclaimed by fill** and is
        feasible: water is not universally forbidden and this is not a dry-land mask.
    F4  a shelf ten below the ring's level is refused as `off_level` -- and the same
        shelf two below is not. The refusal is checked against the **real**
        `ground.Contract`, which clamps a lot declared there back onto the shelf: the
        module restates the resolver's rule and must not be free to restate it wrongly.
    F5  a twelve-block step is `broken` under a lot-sized window and is not under the
        column-sized default: clause 3 is inert until a caller asks it something.
    F6  an island of feasible ground no route reaches is reported as `unreached` and
        **stays in the mask**: a component cut off is not a door that failed.
    F7  a record round-trips exactly through `json.dumps`/`json.loads` to the same mask.
    F8  where no ground was read the record says so -- mask refusing nothing, every
        count `None`, `measured` false -- so "no terrain was read" is never "the terrain
        is fine".
    F9  **the real ground**: the retained section's own lower- and middle-ring districts
        on `out/nb-city/world.before-plateau.npz`, at their ring's own terrace level,
        with the clause split and the arithmetic checked. The figures are reported, not
        asserted: this case establishes that the answer exists and is self-consistent.

Seconds: eight synthetic 60-square fixtures and one read of the retained baseline.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import feasible as F, observe  # noqa: E402

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the fixtures

def world(bed, water=None, y0: int = 30, x0: int = 0, z0: int = 0):
    """A synthetic volume from a bed heightmap and an optional waterline map.

        Synthetic on purpose and labelled so wherever it is reported: these are unit
        fixtures for the four clauses, and F9 is the case that runs on recorded terrain.
        `water[i, j] <= bed[i, j]` is a dry column.
        
    """
    bed = np.asarray(bed, int)
    water = np.full(bed.shape, -10_000, int) if water is None else np.asarray(water, int)
    w, d = bed.shape
    top = int(max(bed.max(), water.max())) + 2
    codes = np.zeros((w, top - y0 + 1, d), np.uint16)
    for i in range(w):
        for j in range(d):
            codes[i, :bed[i, j] - y0 + 1, j] = 1                      # stone
            if water[i, j] > bed[i, j]:
                codes[i, bed[i, j] - y0 + 1:water[i, j] - y0 + 1, j] = 2   # water
    return observe.Volume(x0, y0, z0, codes, ["air", "stone", "water"])


def consistent(got: dict) -> None:
    """The arithmetic every measured record owes: each column is refused by exactly one
    clause or by none, and the clauses do not double-count."""
    assert got["measured"] is True, got["why"]
    total = (got["feasible_columns"] + got["wet_columns"] + got["off_level_columns"]
             + got["broken_columns"] + got["outside_columns"])
    assert total == got["columns"], (total, got["columns"], got)
    assert got["feasible_columns"] == int(np.asarray(got["mask"]).sum()), got
    assert got["reclaimed_columns"] <= got["feasible_columns"], got
    if got["unreached_columns"] is not None:
        assert 0 <= got["unreached_columns"] <= got["feasible_columns"], got


# ------------------------------------ F1: the positive control, on ground with no fault

@case
def t_f1_flat_dry_ground_at_the_designs_level_is_wholly_feasible():
    """The control. A mask that refused everything would pass every counterexample
    below, so the first thing to establish is that good ground comes back good."""
    n, y = 60, 66
    vol = world(np.full((n, n), y))
    got = F.terrain(vol, (0, 0, n - 1, n - 1), level=y,
                    routes=[(0, 0)], window=4)
    consistent(got)
    assert got["feasible_columns"] == n * n, got
    assert got["mask"].all()
    for k in ("wet_columns", "off_level_columns", "broken_columns",
              "unreached_columns", "outside_columns"):
        assert got[k] == 0, (k, got[k], got["why"])
    assert got["reasons"]["components"] == 1, got["reasons"]
    return (f"{n}x{n} of flat dry ground at y={y}, the level the design asks for: "
            f"{got['feasible_columns']}/{got['columns']} columns feasible in "
            f"{got['reasons']['components']} reached component, every clause zero "
            f"(synthetic fixture)")


# ------------------------------------------------------ F2: the lake, refused as water

@case
def t_f2_a_lake_in_the_rectangle_is_refused_as_wet():
    """`developable_columns` subtracts roads and standing parts; this is the water it
    does not subtract, and the 96-house count that was derived from it."""
    n, y = 60, 64
    bed = np.full((n, n), y)
    water = np.full((n, n), -10_000)
    bed[10:30, 10:30] = 58
    water[10:30, 10:30] = 62                      # a lake 20x20, its surface at y=62
    vol = world(bed, water)
    got = F.terrain(vol, (0, 0, n - 1, n - 1), level=62, routes=[(0, 0)])
    consistent(got)
    assert got["wet_columns"] == 400, got
    assert got["reclaimed_columns"] == 0, got
    assert got["feasible_columns"] == n * n - 400, got
    assert not got["mask"][10:30, 10:30].any()
    assert got["reasons"]["waterline"] == 62, got["reasons"]
    assert "1." in got["why"][0] and "water" in got["why"][0]
    return (f"a 20x20 lake at waterline y=62 in a {n}x{n} rectangle brought to y=62: "
            f"{got['wet_columns']} column(s) refused as standing water, "
            f"{got['feasible_columns']} feasible, and the record names clause 1 "
            f"(synthetic fixture)")


# ------------------------------------------- F3: the same lake, reclaimed by the design

@case
def t_f3_the_same_lake_above_its_waterline_is_reclaimed_by_fill():
    """*"Water is not universally forbidden: retaining it, earthworks or bridging must be
    explicit design choices with feasible access."* The same ground, one level higher,
    is a design that fills it -- and the record says how deep the fill is."""
    n, y = 60, 64
    bed = np.full((n, n), y)
    water = np.full((n, n), -10_000)
    bed[10:30, 10:30] = 58
    water[10:30, 10:30] = 62
    vol = world(bed, water)
    got = F.terrain(vol, (0, 0, n - 1, n - 1), level=64, routes=[(0, 0)])
    consistent(got)
    assert got["wet_columns"] == 0, got
    assert got["reclaimed_columns"] == 400, got
    assert got["feasible_columns"] == n * n, got
    assert got["mask"][10:30, 10:30].all()
    assert got["reasons"]["reclaimed_fill"] == 6, got["reasons"]
    assert "reclaimed" in got["why"][0], got["why"][0]
    # ...and it is a *decision*: one block lower and the same water is refused again
    low = F.terrain(vol, (0, 0, n - 1, n - 1), level=62)
    assert low["wet_columns"] == 400 and low["reclaimed_columns"] == 0, low
    return (f"the same 20x20 lake under a design level of y=64, one above its waterline: "
            f"{got['reclaimed_columns']} column(s) reclaimed by up to "
            f"{got['reasons']['reclaimed_fill']} block(s) of fill and feasible, "
            f"{got['wet_columns']} refused -- against {low['wet_columns']} refused at "
            f"y=62, so the water is a choice and not a rule (synthetic fixture)")


# ------------------------------------------ F4: the level the design cannot deliver

@case
def t_f4_a_shelf_ten_below_the_ring_level_is_refused_and_two_below_is_not():
    """`ground._held_level`: a footprint asked for a level more than `RELIEF` from its
    own found ground is clamped **back to its own ground**. The lot on the low shelf does
    not get the terrace; it gets a hole, and its door does not meet the lane."""
    n, y = 60, 66
    assert F.RELIEF == 3, F.RELIEF
    deep = np.full((n, n), y)
    deep[:, :20] = y - 10                          # a shelf ten below the ring's level
    shallow = np.full((n, n), y)
    shallow[:, :20] = y - 2                        # ...and the same shelf two below
    # the lane runs along the far edge, on the ground that stands at the ring's level
    lane = [(i, n - 1) for i in range(n)]
    a = F.terrain(world(deep), (0, 0, n - 1, n - 1), level=y, routes=lane)
    b = F.terrain(world(shallow), (0, 0, n - 1, n - 1), level=y, routes=lane)
    consistent(a)
    consistent(b)
    assert a["off_level_columns"] == 20 * n, a
    assert not a["mask"][:, :20].any() and a["mask"][:, 20:].all()
    assert b["off_level_columns"] == 0, b
    assert b["feasible_columns"] == n * n, b
    # the refused shelf is not merely unreached: it is ground nothing can be founded on,
    # and what survives the clause is reached from the lane
    assert a["unreached_columns"] == 0, a

    # ...and the refusal is the **resolver's** and not this module's restatement of it:
    # the same lot declared to the real `ground.Contract` on the refused shelf comes
    # back clamped to the shelf, ten below the terrace its lane was laid at.
    from ethoslm import ground as G
    vol = world(deep)
    found = G.Found(vol)
    c = G.Contract()
    c.platform("lot_on_the_shelf", (4, 4, 11, 11), y, cls="footprint")
    c.platform("lot_on_the_terrace", (4, 30, 11, 37), y, cls="footprint")
    res = c.resolve(found.bed, found.wet, relief=F.RELIEF)
    assert res.level(4, 4) == y - 10, (res.level(4, 4), y)
    assert res.level(4, 30) == y, (res.level(4, 30), y)
    assert not a["mask"][4, 4] and a["mask"][4, 30], "the clauses disagree with ground"
    return (f"a {n}x20 shelf 10 below a ring level of y={y}: "
            f"{a['off_level_columns']} column(s) refused because `_held_level` clamps a "
            f"footprint back to its own ground beyond relief {F.RELIEF}; the same shelf "
            f"2 below refuses {b['off_level_columns']} and stands at "
            f"{b['feasible_columns']}/{b['columns']} feasible. The real "
            f"`ground.Contract` agrees on the same fixture: an 8x8 lot asked for y={y} "
            f"on the shelf resolves to y={res.level(4, 4)} and the same lot on the "
            f"terrace to y={res.level(4, 30)} (synthetic fixture)")


# ---------------------------------------------- F5: the relief one platform cannot span

@case
def t_f5_a_step_is_broken_under_a_lot_sized_window_and_not_under_a_column():
    """One lot is one platform and one platform is one level. The clause is inert at the
    default `window=0` -- "this column only" -- and the caller passes its lot's radius."""
    n = 60
    bed = np.full((n, n), 64)
    bed[40:, :] = 52                               # a twelve-block step across the rect
    vol = world(bed)
    wide = F.terrain(vol, (0, 0, n - 1, n - 1), window=4)
    point = F.terrain(vol, (0, 0, n - 1, n - 1), window=0)
    consistent(wide)
    consistent(point)
    # the columns whose 9-column window straddles the step: i = 36..43
    assert wide["broken_columns"] == 8 * n, wide
    assert not wide["mask"][36:44, :].any()
    assert point["broken_columns"] == 0, point
    assert point["feasible_columns"] == n * n, point
    assert "inert" in point["why"][2], point["why"][2]
    return (f"a 12-block step across a {n}x{n} rectangle: under a 9-column (lot-sized) "
            f"window {wide['broken_columns']} column(s) are refused because no one "
            f"platform spans more than {2 * F.RELIEF}; under the default window=0 the "
            f"clause is inert and refuses {point['broken_columns']} (synthetic fixture)")


# --------------------------------------------------- F6: the component nothing reaches

@case
def t_f6_an_island_no_route_reaches_is_unreached_and_stays_in_the_mask():
    """*"Distinguish door failures within the section from disconnected route components
    outside it; do not assume all unreachable cells have one cause."* So the unreached
    component is counted and left in the mask, for the caller to tell apart."""
    n = 60
    bed = np.full((n, n), 64)
    water = np.full((n, n), -10_000)
    bed[20:25, :] = 58
    water[20:25, :] = 66                           # a moat, cutting the rectangle in two
    vol = world(bed, water)
    routes = [(0, j) for j in range(n)]            # a lane along the western edge only
    got = F.terrain(vol, (0, 0, n - 1, n - 1), level=64, routes=routes)
    consistent(got)
    assert got["wet_columns"] == 5 * n, got
    assert got["reasons"]["components"] == 2, got["reasons"]
    assert got["reasons"]["reached_components"] == 1, got["reasons"]
    assert got["unreached_columns"] == 35 * n, got
    # the island is feasible ground, and it is still in the mask
    assert got["mask"][25:, :].all(), "the unreached component was removed from the mask"
    assert got["feasible_columns"] == 55 * n, got
    blind = F.terrain(vol, (0, 0, n - 1, n - 1), level=64)
    assert blind["unreached_columns"] is None, blind
    assert "not measured" in blind["why"][3], blind["why"][3]
    return (f"a moat splits a {n}x{n} rectangle into "
            f"{got['reasons']['components']} components and the lane touches "
            f"{got['reasons']['reached_components']}: "
            f"{got['unreached_columns']} feasible column(s) reported unreached and all "
            f"{got['feasible_columns']} still in the mask; with no routes given the "
            f"clause answers `None` and not 0 (synthetic fixture)")


# ------------------------------------------------- F7: the answer a plan can carry

@case
def t_f7_a_record_round_trips_through_json_to_the_same_mask():
    """The answer belongs on the district's record in `plan.place.json` beside the four
    columns of `placeregion`, or the next stage reads the rectangle again and gets a
    fifth different number."""
    n = 60
    bed = np.full((n, n), 64)
    bed[10:30, 10:40] = 50                         # a hollow the level cannot reach
    bed[45:, :10] = 71                             # ...and a knoll it can
    # off the origin, so a record that quietly assumed a mask starts at (0, 0) fails
    # here
    vol = world(bed, x0=100, z0=200)
    rect = (100, 200, 100 + n - 1, 200 + n - 1)
    got = F.terrain(vol, rect, level=64, routes=[(100, 200)], window=3)
    rec = F.record(vol, rect, level=64, routes=[(100, 200)], window=3)
    consistent(got)
    assert 0 < got["feasible_columns"] < got["columns"], got
    assert got["off_level_columns"] == 20 * 30 + 15 * 10, got   # the hollow and the knoll
    assert "mask" not in rec, list(rec)
    text = json.dumps(rec)
    back = F.mask_of(json.loads(text))
    assert back is not None and back.shape == got["mask"].shape
    assert (back == got["mask"]).all(), "the packed mask is not the measured mask"
    assert F.mask_of({}) is None, "an absent bitmap must answer None, not all-true"
    # and a sub-rectangle of it reads off the record without re-measuring the ground
    sub = F.cover(rec, (105, 205, 134, 234))
    live = int(got["mask"][5:35, 5:35].sum())
    assert sub["columns"] == 30 * 30 and sub["feasible_columns"] == live, (sub, live)
    assert 0 < live < 900, live         # a sub-rectangle spanning good and bad ground
    empty = F.cover(F.record(None, (0, 0, 9, 9)), (0, 0, 9, 9))
    assert empty["feasible_columns"] is None, empty
    return (f"a {n}x{n} record is {len(text)} byte(s) of JSON of which "
            f"{len(rec['mask_bits']['bits'])} are the packed mask, round-trips to the "
            f"identical {got['feasible_columns']}-column answer, and `cover` reads a "
            f"30x30 sub-rectangle off it as {sub['feasible_columns']} feasible of "
            f"{sub['columns']} without touching the volume (synthetic fixture)")


# ----------------------------------------------------- F8: absence is never an answer

@case
def t_f8_where_no_ground_was_read_the_record_says_so():
    """`arrange.certificate_for` certifies with `ground=None` and the certificate reads
    exactly like ground that was found to be fine. It must not be possible to confuse
    the two again."""
    got = F.terrain(None, (0, 0, 99, 49))
    assert got["measured"] is False, got
    assert got["mask"].all() and got["mask"].shape == (100, 50)
    assert got["columns"] == 5000, got
    for k in ("feasible_columns", "wet_columns", "off_level_columns", "broken_columns",
              "unreached_columns", "reclaimed_columns"):
        assert got[k] is None, (k, got[k])
    assert "not read" in " ".join(got["why"]), got["why"]
    rec = F.record(None, (0, 0, 99, 49))
    assert json.loads(json.dumps(rec))["measured"] is False
    # a rectangle laid past the edge of the volume observed: read where it was read, and
    # refused apart -- not counted as ground found good and not as ground found bad
    part = F.terrain(world(np.full((40, 40), 64)), (20, 20, 59, 59), level=64, window=4)
    consistent(part)
    assert part["outside_columns"] == 40 * 40 - 20 * 20, part
    assert part["feasible_columns"] == 400, part
    # ...and the unread ground is not a step: a column at y=0 in a window would make
    # every rectangle that leaves the baseline look broken
    assert part["broken_columns"] == 0, part
    return (f"with no volume the record refuses nothing over its "
            f"{got['columns']} column(s), every count is None rather than 0 and "
            f"`measured` is false; a 40x40 rectangle half off the observed volume "
            f"reports {part['outside_columns']} column(s) outside and "
            f"{part['feasible_columns']} feasible, which are different refusals")


# --------------------------------------------------------------- F9: the real ground

_LOADED: dict = {}


def baseline():
    """The observed baseline of the retained section, read once. Immutable input."""
    if "vol" in _LOADED:
        return _LOADED["vol"]
    path = os.path.join(ROOT, "out", "nb-city", "world.before-plateau.npz")
    if not os.path.exists(path):
        raise Skip("no out/nb-city/world.before-plateau.npz")
    from ethoslm import offline
    _LOADED["vol"] = offline.load_volume(path)
    return _LOADED["vol"]


def ring_levels(vol, plan: dict, spec: dict, site: dict) -> tuple:
    """`(levels by ring, why)` -- the terrace level each ring's ground is brought to.

        The layout's own record where it carries one. The retained plan does **not**: every
        ring of `out/nb-city/plan.place.json` records `level: null` and `layout.terrace` is
        null, although the ground stage plainly computed a terrace (the palace podium the
        proposal cut at y=83 is `terrace_levels(...)["podium"]` exactly). So it is recomputed
        here the way `pipeline.stages_plan.terrace_for` computes it -- the site's median
        land, `TERRACE_STEP` a ring, ordered by the rings' own elevation words -- and the
        reported figure says which of the two it used.
        
    """
    from ethoslm import placeplan, spec as spec_mod
    rings = spec_mod.rings(spec)
    lay = plan.get("layout") or {}
    named = {r["name"]: r.get("level") for r in (lay.get("rings") or [])}
    if named and all(v is not None for v in named.values()):
        return {k: int(v) for k, v in named.items()}, "the layout's own ring levels"
    med = placeplan.site_median(vol, site)
    if med is None:
        raise Skip("no site median: the baseline does not cover the site")
    ranks, _why = placeplan.terrace_ranks(rings)
    terr = placeplan.terrace_levels(med, len(rings), ranks=ranks)
    return ({str(r.get("name")): int(terr["rings"][k]) for k, r in enumerate(rings)},
            f"recomputed as `stages_plan.terrace_for` does -- the plan records no ring "
            f"level at all -- from the site's median land y={med} by TERRACE_STEP "
            f"{terr['step']}, ordered by the rings' elevation words; its podium "
            f"y={terr['podium']} is the level the retained ground proposal cut")


@case
def t_f9_the_sections_own_districts_measured_at_their_ring_level():
    """**The number the review says does not exist.** *"A district's developable ground
        and the ground its count is derived from are still two different numbers."*

        Asserted: the answer is produced, its arithmetic closes, no clause double-counts,
        and the record persists. Reported and not asserted: the figures themselves. This
        case measures the retained plan; it does not judge it, and a bar on any of these
        numbers belongs to the acceptance runner and not here.
        
    """
    vol = baseline()
    d = os.path.join(ROOT, "out", "nb-city")
    plan = json.load(open(os.path.join(d, "plan.place.json")))
    spec = json.load(open(os.path.join(d, "place.json")))
    prop = json.load(open(os.path.join(d, "ground_proposal.json")))
    site = prop.get("site") or {}
    if not site.get("size"):
        raise Skip("the retained ground proposal records no site")
    levels, how = ring_levels(vol, plan, spec, site)
    lots = {r["name"]: int((r.get("target") or {}).get("lot", [8, 8])[0])
            for r in (plan["layout"].get("rings") or [])}
    cells = (plan.get("arterials") or {}).get("cells") or []
    section = (-5900, 596, -5620, 766)             # the retained section, x/z inclusive
    said = []
    for q in plan["districts"]:
        name = q["name"]
        if not (name.startswith("lower_ring_north")
                or name.startswith("middle_ring_north")):
            continue
        rect = (int(q["x0"]), int(q["z0"]), int(q["x1"]), int(q["z1"]))
        ring = q["defines"]
        lvl, lot = levels[ring], lots.get(ring, 8)
        near = [c for c in cells if rect[0] - 2 <= c[0] <= rect[2] + 2
                and rect[1] - 2 <= c[1] <= rect[3] + 2]
        got = F.terrain(vol, rect, level=lvl, routes=near, window=lot // 2)
        consistent(got)
        assert got["outside_columns"] == 0, (name, got["outside_columns"])
        # no clause double-counts: every refusal is one clause's, and the four clause
        # counts plus the feasible ground are exactly the rectangle
        assert (got["wet_columns"] + got["off_level_columns"] + got["broken_columns"]
                + got["feasible_columns"]) == got["columns"], got
        assert got["columns"] == int(q["scope_columns"]), (name, q["scope_columns"])
        # the answer persists on the district's record, and reads back the same
        rec = F.record(vol, rect, level=lvl, routes=near, window=lot // 2)
        text = json.dumps(rec)
        assert (F.mask_of(json.loads(text)) == got["mask"]).all(), name
        assert len(text) < 4096, (name, len(text))
        # ...and the part of it that lies inside the retained section
        cut = (max(rect[0], section[0]), max(rect[1], section[1]),
               min(rect[2], section[2]), min(rect[3], section[3]))
        inside = (F.cover(rec, cut) if cut[0] <= cut[2] and cut[1] <= cut[3] else None)
        # a control on the level, not on the ground: what the same rectangle could take
        # if it were brought to its own median bed rather than to the ring's one terrace
        own = got["reasons"]["bed"][1]
        alt = F.terrain(vol, rect, level=own, window=lot // 2)
        lost = got["reasons"]["components"] - (got["reasons"]["reached_components"] or 0)
        said.append(
            f"{name} {q['x1'] - q['x0'] + 1}x{q['z1'] - q['z0'] + 1}="
            f"{got['columns']} column(s) asked for {q['structures']} structure(s): at "
            f"the {ring} terrace y={lvl}, {got['feasible_columns']} feasible "
            f"({got['feasible_columns'] / got['columns']:.1%}) = "
            f"{got['wet_columns']} wet + {got['off_level_columns']} off-level + "
            f"{got['broken_columns']} broken refused, of which "
            f"{got['reclaimed_columns']} reclaimed by up to "
            f"{got['reasons']['reclaimed_fill']} block(s) of fill and "
            f"{got['unreached_columns']} unreached in "
            f"{lost} of {got['reasons']['components']} component(s); bed "
            f"y={got['reasons']['bed'][0]}..{got['reasons']['bed'][2]} median "
            f"y={own}, at which level the same rectangle would take "
            f"{alt['feasible_columns']} ({alt['feasible_columns'] / alt['columns']:.1%})"
            + (f"; {inside['feasible_columns']} of {inside['columns']} column(s) "
               f"feasible inside the retained section" if inside else "")
            + f"; {len(text)} byte(s) as a record")
    assert said, "no lower- or middle-ring north district in the retained plan"
    return (f"on out/nb-city/world.before-plateau.npz, {how}. " + " | ".join(said))


# ------------------------------------------------------------------ the runner

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = skipped = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__[2:]
        if only and only not in name:
            continue
        try:
            said = fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            skipped += 1
            continue
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} feasible ground cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
