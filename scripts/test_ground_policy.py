#!/usr/bin/env python3
"""**One ground policy decides both capacity and earthwork.**

The neighbourhood delivery round's third connected cause. Two rules were in play about
one question and they disagreed:

  * `placeplan.district_ground` measures a district's ground through
    `feasible.record(..., relief=DISTRICT_TERRACE_REACH, fill=DISTRICT_TERRACE_REACH)`
    -- a column needing a deeper cut or fill leaves the developable set, and every
    count, band and cover clause in the place divides by what is left;
  * `Builder.terrace_annulus` levelled **every** column of the rectangle it was handed,
    with no per-column bound at all. `TERRACE_MAX_BLOCKS` is a total budget and a budget
    is not a bound.

So housing was excluded from a hillside because that hillside should not be cut, and
construction cut it anyway. Measured here on the retained section's own observed
baseline, `out/nd-city/world.before-plateau.npz`, which is the immutable observation
every number in this project is taken against.

    P1  a column the mask refuses for grade is a column the terrace does not move, and a
        column the mask accepts is brought to the level. The counterexample and the
        positive control in one fixture, against the unbounded call on the same ground
        so the bound is visibly what changed it.
    P2  the same, for water: a lake the level does not stand over is kept, and a puddle
        the level does stand over inside the bound is reclaimed. The reclamation clause
        is not quietly disabled by the bound.
    P3  the totals in a terrace record reconcile with `feasible.record`'s own counts
        over the same rectangle at the same level -- column for column, on the real
        baseline, not by argument.
    P4  **the mesa.** `middle_ring_north_west`, bed y=58..127, at the middle ring's own
        settled level: the district keeps a buildable envelope rather than becoming a
        levelled plateau, the envelope is mostly ground a building can stand on, and the
        unmet programme stays owed on the rest.
    P5  the seam a terrace has **inside** itself: every kept column beside worked ground
        is a named seam, the ones the prepared ground stands over are carried as a
        retaining face, and the ones that stand over it are the hillside they were. No
        worked column has an unfaced drop into kept ground.
    P6  the earthwork actually performed is inside the bound the mask was computed with,
        on every ring strip and district terrace of the real section -- and where the
        builder's own sounding and the reading the decision was taken on disagree, the
        record counts those columns rather than hiding them in a maximum.
    P7  a caller that gets `None` from `developable_rect` behaves exactly as it does
        today: whole-rectangle ground, unread ground and ground with no envelope worth
        laying are three different answers and none of them is an exception.

Seconds: six synthetic fixtures and three reads of the retained baseline, about a
minute. P6 lays no blocks it does not have to: it runs the decision, not the build.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import feasible as F, ground as G, observe, placeplan  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402

#: The retained section's immutable observed baseline. No case writes to it; every case
#: that reads real ground reads this and says so in its own report.
BASELINE = os.path.join(ROOT, "out", "nd-city", "world.before-plateau.npz")
PLAN = os.path.join(ROOT, "out", "nd-city", "plan.place.json")

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the fixtures

def world(bed, water=None, y0: int = 30, x0: int = 0, z0: int = 0):
    """A synthetic volume from a bed heightmap and an optional waterline map.

        The same fixture `scripts/test_feasible.py` measures the mask on, so the two suites
        are asking one question of one piece of ground: that is the whole point of this
        file. `water[i, j] <= bed[i, j]` is a dry column.
        
    """
    bed = np.asarray(bed, int)
    water = np.full(bed.shape, -10_000, int) if water is None else np.asarray(water, int)
    w, d = bed.shape
    top = int(max(bed.max(), water.max())) + 2
    codes = np.zeros((w, top - y0 + 1, d), np.uint16)
    for i in range(w):
        for j in range(d):
            codes[i, :bed[i, j] - y0 + 1, j] = 1                        # stone
            if water[i, j] > bed[i, j]:
                codes[i, bed[i, j] - y0 + 1:water[i, j] - y0 + 1, j] = 2  # water
    return observe.Volume(x0, y0, z0, codes, ["air", "stone", "water"])


def builder(vol):
    from ethoslm import offline
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    return b


def column_touched(b, x: int, z: int) -> bool:
    """Did this builder decide anything at all in this column?

        The test of "left exactly as it was found" that does not depend on what the answer
        would have been: not one block pending anywhere in the column, and not recorded in
        `_sited`, which is where the library keeps the columns whose level is somebody's
        decision.
        
    """
    if (x, z) in getattr(b, "_sited", {}):
        return True
    return any(p[0] == x and p[2] == z for p in b._pending)


def baseline():
    from ethoslm import offline
    if not os.path.exists(BASELINE):
        raise Skip(f"no observed baseline at {os.path.relpath(BASELINE, ROOT)}")
    return offline.load_volume(BASELINE)


def plan():
    import json
    if not os.path.exists(PLAN):
        raise Skip(f"no plan at {os.path.relpath(PLAN, ROOT)}")
    return json.load(open(PLAN))


# -------------------------------------------- 1. the bound, at the column

@case
def p1_the_grade_the_mask_refuses_is_the_grade_the_terrace_does_not_cut():
    """A counterexample and a positive control on one hillside.

        Twelve columns of a 40-column strip stand twenty blocks over the design's level and
        the rest stand three below it. At `reach=8` the mask refuses the first twelve for
        grade (`off_level`) and accepts the rest, and the terrace must move exactly the
        accepted ones. The same call with no bound is run on the same ground first, so this
        case shows the bound doing the work rather than a fixture that had nothing to cut.
        
    """
    LVL, REACH = 64, 8
    bed = np.full((40, 40), LVL - 3, int)
    bed[:12, :] = LVL + 20                        # a mesa: twenty blocks of cut
    vol = world(bed)
    rect = (0, 0, 39, 39)

    got = F.terrain(vol, rect, level=LVL, relief=REACH, fill=REACH)
    assert got["off_level_columns"] == 12 * 40, got["why"]
    assert got["feasible_columns"] == 28 * 40, got["why"]

    loose = builder(world(bed))
    a = loose.terrace_annulus(rect, LVL, label="unbounded")
    assert a["ok"], a["reason"]
    assert a["disposition"]["worked"] == 1600, a["disposition"]
    assert a["disposition"]["max_cut"] == 20, a["disposition"]
    assert column_touched(loose, 0, 0), "the unbounded call did not cut the mesa"

    b = builder(world(bed))
    rec = b.terrace_annulus(rect, LVL, label="bounded", reach=REACH, base=vol)
    assert rec["ok"], rec["reason"]
    d = rec["disposition"]
    # the counterexample: not one block, not one `_sited` entry, in the refused columns
    for (x, z) in ((0, 0), (5, 20), (11, 39)):
        assert not column_touched(b, x, z), (x, z, "a refused column was moved")
    # the positive control: the accepted ones are at the level
    for (x, z) in ((12, 0), (25, 20), (39, 39)):
        assert b._sited.get((x, z)) == LVL, (x, z, b._sited.get((x, z)))
    assert d["worked"] == 28 * 40 == got["feasible_columns"], (d, got)
    assert d["keep_high"] == 12 * 40, d
    assert d["max_cut"] == 0 and d["max_fill"] == 3, d
    assert d["within_reach"] is True, d
    return (f"a 40x40 strip at y={LVL} with a {12 * 40}-column mesa twenty blocks over "
            f"it: unbounded the terrace moves all 1,600 columns and cuts "
            f"{a['disposition']['max_cut']} blocks off the deepest; at reach={REACH} it "
            f"moves {d['worked']} -- exactly the mask's {got['feasible_columns']} -- "
            f"leaves {d['keep_high']} standing and its deepest column is "
            f"{max(d['max_cut'], d['max_fill'])}")


@case
def p2_the_water_the_level_does_not_stand_over_is_water_the_terrace_keeps():
    """Clause 1 survives the bound, in both directions.

        Half the rectangle is a lake whose bed is thirty below the level, half a puddle two
        below it. The design's level stands over both waterlines, so the unbounded clause
        would reclaim both -- and reclaiming thirty blocks of lake is a dam, which is the
        measurement `DISTRICT_TERRACE_REACH` was registered on. At `reach=8` the puddle is
        reclaimed and the lake is kept, and the terrace fills exactly the puddle.
        
    """
    LVL, REACH = 64, 8
    bed = np.full((40, 40), LVL - 30, int)
    bed[20:, :] = LVL - 2
    water = np.full((40, 40), LVL - 1, int)       # a waterline the level stands over
    vol = world(bed, water)
    rect = (0, 0, 39, 39)

    got = F.terrain(vol, rect, level=LVL, relief=REACH, fill=REACH)
    assert got["reclaimed_columns"] == 20 * 40, got["why"]
    assert got["wet_columns"] == 20 * 40, got["why"]

    b = builder(world(bed, water))
    rec = b.terrace_annulus(rect, LVL, label="lake", reach=REACH, base=vol)
    assert rec["ok"], rec["reason"]
    d = rec["disposition"]
    assert d["keep_water"] == 20 * 40, d
    assert d["worked"] == 20 * 40 == got["feasible_columns"], (d, got)
    assert d["reclaimed"] == 20 * 40, d           # the builder calls its own fill flooded
    assert d["max_fill"] == 2, d
    assert not column_touched(b, 0, 0), "a lake the level cannot reach was filled"
    assert b._sited.get((30, 20)) == LVL, b._sited.get((30, 20))
    # ...and unbounded, the same ground is a dam: thirty blocks of fill over a lake
    loose = builder(world(bed, water))
    a = loose.terrace_annulus(rect, LVL, label="dam")
    assert a["disposition"]["max_fill"] == 30, a["disposition"]
    return (f"a lake thirty blocks under y={LVL} and a puddle two under it: unbounded "
            f"the terrace fills both and its deepest column is "
            f"{a['disposition']['max_fill']} blocks of fill; at reach={REACH} it "
            f"reclaims the {d['reclaimed']}-column puddle, keeps the "
            f"{d['keep_water']}-column lake as water, and its deepest fill is "
            f"{d['max_fill']}")


@case
def p3_the_terrace_record_reconciles_with_the_mask_over_the_same_ground():
    """The two instruments, over one real rectangle at one real level.

        A ring strip of the retained section on the observed baseline. Every column of the
        piece is in exactly one of `feasible.DISPOSITIONS`, the four that move earth sum to
        the mask's own `feasible_columns`, `keep_water` is the mask's `wet_columns`, and the
        builder's own tally of what it filled, reclaimed and left alone equals the
        disposition it was given. This is the reconciliation the round asks for: column for
        column, not by argument.
        
    """
    vol = baseline()
    rect, LVL, REACH = (-5966, 690, -5556, 765), 72, 8       # middle_ring/0, settled
    got = F.terrain(vol, rect, level=LVL, relief=REACH, fill=REACH)
    dis = placeplan.terrace_ground(vol, rect, LVL, reach=REACH, label="middle_ring/0")
    c = dis["counts"]
    assert sum(c.values()) == dis["columns"] == got["columns"], (c, dis["columns"])
    assert sum(c[k] for k in F.MOVED) == got["feasible_columns"], (c, got)
    assert c["keep_water"] == got["wet_columns"], (c, got)
    assert c["unread"] == got["outside_columns"], (c, got)
    assert c["keep_high"] + c["keep_low"] == (got["off_level_columns"]
                                              + got["broken_columns"]), (c, got)
    assert dis["max_cut"] <= REACH, dis
    # the quarter design round's pit rule: fill beyond the reach only in a recorded pit
    assert dis["max_fill"] <= REACH or (
        got["pits"]["columns"] and dis["max_fill"] <= F.PIT_DEPTH_REACHES * REACH), dis

    b = builder(vol)
    rec = b.terrace_annulus(rect, LVL, label="middle_ring/0", reach=REACH, base=vol,
                            sides={"north", "west", "east"})
    assert rec["ok"], rec["reason"]
    d = rec["disposition"]
    assert d["columns"] == d["worked"] + d["left_alone"] + d["lanes"], d
    assert d["worked"] == got["feasible_columns"], (d, got)
    for k in F.KEPT:
        assert d[k] == c[k], (k, d[k], c[k])
    return (f"middle_ring/0, {rect}, at its settled y={LVL} on the observed baseline: "
            f"the mask makes {got['feasible_columns']:,} of {got['columns']:,} columns "
            f"feasible and the terrace moves {d['worked']:,} -- the same columns -- "
            f"leaving {d['keep_high']:,} standing, {d['keep_low']:,} open and "
            f"{d['keep_water']:,} under water; {d['fill_blocks']:,} blocks of fill and "
            f"{d['cut_blocks']:,} of cut against {rec['blocks']:,} laid in all")


# ------------------------------------------- 2. the calm side, and its envelope

@case
def p4_a_district_on_a_mesa_keeps_a_buildable_envelope():
    """`middle_ring_north_west`: 11,400 columns of hillside, bed y=58..127.

        The audit's case. It records 2,969 feasible columns of 11,400 at the middle ring's
        settled level, proposed nineteen houses and realized one, and was written down as
        "open ground with an owner" while the ring's strip levelled the whole rectangle
        under it. Two things have to be true together and this case asserts both:

          * the **terrace** leaves the mesa where it is -- the columns the mask refuses are
            not moved, so the district does not become a levelled plateau;
          * the **grid** is offered a rectangle it can actually be laid over, mostly ground
            a building can stand on, and the rest of the district stays open with its
            programme owed.
        
    """
    vol = baseline()
    place = plan()
    d = next((q for q in (place.get("districts") or [])
              if q.get("name") == "middle_ring_north_west"), None)
    if d is None:
        raise Skip("no middle_ring_north_west in the retained plan")
    rec = d.get("ground") or {}
    assert rec.get("measured"), "the district carries no measured ground record"
    lvl = int(rec["level"])
    drect = (min(d["x0"], d["x1"]), min(d["z0"], d["z1"]),
             max(d["x0"], d["x1"]), max(d["z0"], d["z1"]))
    bed = (rec.get("reasons") or {}).get("bed")

    # 1. the terrace, over the district's own rectangle at its own level
    dis = placeplan.terrace_ground(vol, drect, lvl, label="middle_ring_north_west")
    kept = dis["kept_columns"]
    assert kept > 0.5 * dis["columns"], dis["why"]
    assert dis["max_cut"] <= placeplan.DISTRICT_TERRACE_REACH, dis["why"]
    # beyond the reach only in a pit the rule records (`feasible.PIT_COLUMNS`)
    assert dis["max_fill"] <= placeplan.DISTRICT_TERRACE_REACH or (
        (dis["terrain"].get("pits") or {}).get("columns")
        and dis["max_fill"] <= F.PIT_DEPTH_REACHES * placeplan.DISTRICT_TERRACE_REACH), \
        dis["why"]
    # ...and it is not a mask that refuses everything: ground is prepared here
    assert dis["moved_columns"] > 2000, dis["why"]

    # 2. the envelope the grid should be laid over
    env = placeplan.developable_envelope(d, place)
    r = placeplan.developable_rect(d, place)
    assert r is not None, env["why"]
    assert r == tuple(env["rect"]), (r, env["rect"])
    assert (drect[0] <= r[0] <= r[2] <= drect[2]
            and drect[1] <= r[1] <= r[3] <= drect[3]), (r, drect)
    assert env["cover"] >= placeplan.DEVELOPABLE_RECT_COVER, env
    assert env["envelope_columns"] >= placeplan.DEVELOPABLE_RECT_MIN_COLUMNS, env
    # the envelope is a quarter and not the district: the programme stays owed outside
    # it
    assert env["envelope_columns"] < 0.5 * env["columns"], env
    from ethoslm.district_compile import GROUND_FOUNDED
    assert env["cover"] > GROUND_FOUNDED, (env["cover"], GROUND_FOUNDED)
    w, dd = r[2] - r[0] + 1, r[3] - r[1] + 1
    return (f"middle_ring_north_west {drect}, bed y={bed[0]}..{bed[2]} at its own "
            f"y={lvl}: the terrace moves {dis['moved_columns']:,} of {dis['columns']:,} "
            f"columns and keeps {kept:,} ({kept / dis['columns']:.0%}) exactly as found "
            f"-- {dis['counts']['keep_high']:,} of hillside left standing -- its "
            f"deepest column {max(dis['max_cut'], dis['max_fill'])} against a reach of "
            f"{placeplan.DISTRICT_TERRACE_REACH}; the grid is offered {w}x{dd} = "
            f"{env['envelope_columns']:,} columns at {env['cover']:.0%} feasible, "
            f"against {GROUND_FOUNDED:.0%} the compiler needs of one lot, and "
            f"{env['columns'] - env['envelope_columns']:,} columns stay open ground")


# ----------------------------------------------- 3. the seam inside a terrace

@case
def p5_the_seam_between_prepared_and_unprepared_ground_is_a_deliberate_condition():
    """A terrace that leaves ground alone has edges in the middle of itself.

        A hillside with a twenty-block knoll in the middle of it and a twenty-block hollow
        beside it: at `reach=8` both are kept, and the seam between them and the prepared
        ground is two different conditions that must be handled two different ways.

          * the prepared ground stands **over** the hollow: a retaining face in the footing,
            one column into the kept ground and down to its own bed, so there is a wall's
            top at the terrace's edge and the fill behind it is not spilling into the hole;
          * the knoll stands **over** the prepared ground: nothing is laid, because laying
            anything is cutting the ground this decision just refused to cut.

        Asserted at the block, and asserted **exhaustively**: no worked column anywhere in
        the rectangle has an unfaced drop of more than a kerb into kept ground.
        
    """
    LVL, REACH = 64, 8
    bed = np.full((40, 40), LVL, int)
    bed[5:15, 5:15] = LVL + 20                      # a knoll, kept: it stands over
    bed[25:35, 25:35] = LVL - 20                    # a hollow, kept: it stands under
    vol = world(bed)
    b = builder(vol)
    rec = b.terrace_annulus((0, 0, 39, 39), LVL, label="seam", reach=REACH, base=vol)
    assert rec["ok"], rec["reason"]
    seams = rec["disposition"]["seams"]
    # the seam of a 10x10 kept patch is its own outer ring, 100 - 8*8 = 36 columns, and
    # the corners are in it: a corner has two four-neighbours on worked ground
    assert seams["columns"] == 2 * 36, seams
    assert seams["faced"] == 36, seams               # the hollow's ring
    assert seams["natural"] == 36, seams             # the knoll's
    assert seams["kinds"] == {"face": 72}, seams
    assert seams["face_blocks"] == 36 * (LVL - (LVL - 20) + 1), seams

    # the hollow's own rim column -- kept ground, one column in -- carries footing from
    # its own bed up to the level: a wall's top at the terrace's edge and no cavity
    foot = [yy for (x, yy, z) in b._pending if (x, z) == (25, 30)]
    assert min(foot) == LVL - 20 and max(foot) == LVL, (min(foot), max(foot))
    # ...and one column further in is kept ground and is not touched at all
    assert not column_touched(b, 27, 30), "the hollow's floor was filled"
    # the knoll's rim is untouched, and so is the knoll
    assert not column_touched(b, 5, 5), "the knoll's own ground was moved"
    for (x, z) in ((4, 9), (15, 9), (9, 4), (9, 15)):
        assert (x, z) not in b._sited or b._sited[(x, z)] == LVL, (x, z)

    # ...and the exhaustive form: every kept column beside worked ground is either
    # faced, or is ground that stands at or over the level.
    kept = {(x, z) for x in range(40) for z in range(40)
            if abs(int(bed[x, z]) - LVL) > REACH}
    worked = {(x, z) for x in range(40) for z in range(40)} - kept
    unfaced = []
    for (x, z) in kept:
        if not any((x + dx, z + dz) in worked
                   for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            continue
        if int(bed[x, z]) >= LVL:
            continue
        if G.seam_kind(LVL - int(bed[x, z])) in (None, "kerb"):
            continue
        if b._sited.get((x, z)) != LVL:
            unfaced.append((x, z))
    assert not unfaced, f"{len(unfaced)} seam column(s) with an unfaced drop: {unfaced[:5]}"
    return (f"a 40x40 terrace at y={LVL} with a knoll twenty over and a hollow twenty "
            f"under: {seams['columns']} seam column(s) inside the rectangle, all of them "
            f"`{G.seam_kind(20)}` by the drop -- {seams['faced']} carried as a retaining "
            f"face for {seams['face_blocks']} blocks of footing, {seams['natural']} left "
            f"as the hillside they are, and no worked column with an unfaced drop")


# -------------------------------------- 4. the whole section, and the residual

@case
def p6_every_piece_of_the_sections_designed_ground_is_inside_the_bound():
    """The policy, over the real section's twenty-four ring and district pieces.

        Read off the retained run's own ground contract (`out/nd-city/terraces.json`), at
        the levels that run settled, on the observed baseline. For each piece the decision
        is taken and its earthwork measured; the deepest column of every one of them must be
        inside `DISTRICT_TERRACE_REACH`, and the totals against the unbounded earthwork the
        same pieces would do are reported so the size of the disagreement is on the record.

        The gate approaches are **excluded and named**, because they are not the same
        question: a ramp exists to move earth so that a gate can be reached, and bounding it
        refuses the palace gate's approach outright (135 columns, 0 feasible at y=83) and
        leaves the gate unreachable. Together the approaches move 6,435 blocks, 0.09% of
        what this stage lays. Reshaping access is a design choice the round authorizes; it
        is recorded here rather than left to be discovered.
        
    """
    import json
    p = os.path.join(ROOT, "out", "nd-city", "terraces.json")
    if not os.path.exists(p):
        raise Skip("no terraces.json from a real run to read the settled levels off")
    pieces = [q for q in (json.load(open(p)).get("contract") or {}).get("pieces") or []
              if q.get("what") != "gate approach"]
    if not pieces:
        raise Skip("the contract record carries no ring or district piece")
    vol = baseline()
    r = F.reading(vol)
    REACH = placeplan.DISTRICT_TERRACE_REACH
    tot = {"columns": 0, "moved": 0, "kept": 0, "cut": 0, "fill": 0,
           "loose_cut": 0, "loose_fill": 0}
    worst = None
    for q in pieces:
        rect, lvl = tuple(q["rect"]), int(q["level"])
        dis = placeplan.terrace_ground(r, rect, lvl, reach=REACH, label=q["label"])
        assert dis["max_cut"] <= REACH, (q["label"], dis["max_cut"])
        assert dis["max_fill"] <= REACH or (
            (dis["terrain"].get("pits") or {}).get("columns")
            and dis["max_fill"] <= F.PIT_DEPTH_REACHES * REACH), (q["label"],
                                                                   dis["max_fill"])
        assert sum(dis["counts"].values()) == dis["columns"], q["label"]
        loose = F.dispositions(r, rect, level=lvl, relief=10_000, fill=None)
        tot["columns"] += dis["columns"]
        tot["moved"] += dis["moved_columns"]
        tot["kept"] += dis["kept_columns"]
        tot["cut"] += dis["cut_blocks"]
        tot["fill"] += dis["fill_blocks"]
        tot["loose_cut"] += loose["cut_blocks"]
        tot["loose_fill"] += loose["fill_blocks"]
        deep = max(loose["max_cut"], loose["max_fill"])
        if worst is None or deep > worst[1]:
            worst = (q["label"], deep)
    saved = (tot["loose_cut"] + tot["loose_fill"]) - (tot["cut"] + tot["fill"])
    assert saved > 0, tot
    return (f"{len(pieces)} ring and district piece(s) of the retained section, "
            f"{tot['columns']:,} columns at their settled levels on the observed "
            f"baseline: {tot['moved']:,} moved ({tot['moved'] / tot['columns']:.0%}) "
            f"and {tot['kept']:,} kept, for {tot['fill']:,} blocks of fill and "
            f"{tot['cut']:,} of cut, every piece inside a reach of {REACH}; unbounded "
            f"the same pieces move {tot['loose_fill']:,} and {tot['loose_cut']:,} -- "
            f"{saved:,} blocks of difference, the deepest single column being "
            f"{worst[1]} on {worst[0]}")


@case
def p7_no_envelope_and_no_ground_read_are_two_different_answers():
    """`developable_rect` never guesses, and `None` is always the caller's old behaviour.

        Three cases a consumer has to be able to tell apart, and the fourth that must not
        happen: a district whose ground was never read, a district whose rectangle is wholly
        usable, a district whose feasible ground is real but too fragmented to lay a grid
        over -- and never an exception on a district with no ground record at all.
        
    """
    d0 = {"name": "unread", "x0": 0, "z0": 0, "x1": 59, "z1": 59}
    assert placeplan.developable_rect(d0, {}) is None, "a district with no record"
    assert "not measured" in placeplan.developable_envelope(d0, {})["why"]

    LVL = 64
    flat = world(np.full((60, 60), LVL, int))
    whole = dict(d0, name="whole",
                 ground=F.record(flat, (0, 0, 59, 59), level=LVL,
                                 relief=placeplan.DISTRICT_TERRACE_REACH,
                                 fill=placeplan.DISTRICT_TERRACE_REACH))
    assert whole["ground"]["feasible_columns"] == 3600, whole["ground"]["why"]
    assert placeplan.developable_rect(whole, {}) is None
    assert "whole rectangle" in placeplan.developable_envelope(whole, {})["why"]

    # a lattice of one-column ribs over thirty-block pits: 43.75% of the rectangle is
    # ground a building can stand on and no rectangle of it is more than one column
    # wide, which is real feasible ground that no grid can be laid over
    bed = np.full((60, 60), LVL - 30, int)
    bed[::4, :] = LVL
    bed[:, ::4] = LVL
    broke = world(bed)
    frag = dict(d0, name="fragmented",
                ground=F.record(broke, (0, 0, 59, 59), level=LVL,
                                relief=placeplan.DISTRICT_TERRACE_REACH,
                                fill=placeplan.DISTRICT_TERRACE_REACH))
    got = placeplan.developable_envelope(frag, {})
    assert placeplan.developable_rect(frag, {}) is None, got
    assert got["feasible_columns"] > 1000, got
    assert got["why"].startswith("no rectangle of this district"), got["why"]

    # ...and the positive control, so `None` is not this function's only answer: the
    # same ground with one solid half is offered that half.
    bed2 = np.full((60, 60), LVL - 30, int)
    bed2[:, 30:] = LVL
    half = dict(d0, name="half",
                ground=F.record(world(bed2), (0, 0, 59, 59), level=LVL,
                                relief=placeplan.DISTRICT_TERRACE_REACH,
                                fill=placeplan.DISTRICT_TERRACE_REACH))
    r = placeplan.developable_rect(half, {})
    assert r == (0, 30, 59, 59), r
    return (f"a district with no ground record, one wholly usable and one whose "
            f"{got['feasible_columns']} feasible columns hold no rectangle over "
            f"{placeplan.DEVELOPABLE_RECT_MIN_SIDE} a side all answer None, each with "
            f"its own reason; the same ground with one solid half is offered "
            f"{r} and nothing is raised anywhere")


# ------------------------------------------------------------------ the runner

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = skipped = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__[3:]
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
    print(f"\n{ok}/{ok + fail} ground policy cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
