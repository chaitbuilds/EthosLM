"""Where construction can actually stand: the ground a district's count is derived from.

The neighbourhood review's first remaining cause, in its own words: *"Planning capacity is
not terrain-and-access feasibility."* `district_compile` does not read terrain at all,
`placeplan.developable_columns` subtracts the arterial's band and the standing parts'
clearances from a rectangle but not water and not an impossible grade, and
`arrange.certificate_for` calls the district validator with `ground=None`. So the crowded
ring's rectangles ran past the ring terrace onto low wet ground, the denser fabric filled
what it was told was developable, houses were sited on decks over water, and the walk
network broke -- 973 unreachable lane stances in the retained reading. Three numbers were
in play and none of them was the same number: the rectangle's columns, the columns a
count was derived from, and the columns a building could be founded on.

This module is the third one, measured, and nothing else. It builds nothing, it decides
nothing, and it refuses to guess: it reads the observed baseline under a rectangle, is
told the level this design proposes to bring that ground to, and answers **which columns
a building may be founded on, and for each column it refuses, which clause refused it**.

The governing physical rule is not ours; it is `ground._held_level`, which every platform
in the place already passes through:

    a `footprint` or a `field` platform asked for a level more than `RELIEF` from its
    own ground as found is **clamped back to its own ground**.

A lot laid on ground fifteen blocks below the ring's terrace does not get the terrace. It
gets a hole, its floor stands where the hillside was, and its door does not meet the lane
that was laid at the terrace level. That is the whole of clause 2, and it is why this
module re-exports `RELIEF` from `ground` rather than choosing a tolerance of its own: a
feasibility rule that disagreed with the resolver would be a fourth different number.

Four clauses, in precedence, each **independently** counted. The review is explicit that
*"the records do not establish one cause for every unreachable stance"*, so they are never
collapsed into a single refusal:

  1. **wet** -- the observed surface is water. Refused, *unless* this design's level
     stands at or above the waterline + 1, in which case the water is reclaimed by fill,
     the column is feasible, and the record says so and counts it. Water is not
     universally forbidden. Retaining it, bridging it and reclaiming it are design
     choices, and each one has to be explicit and carry its own number.
  2. **off_level** -- `abs(bed - level) > relief`: `_held_level` will not deliver this
     design's level here, whatever the plan says.
  3. **broken** -- the bed over a lot-sized neighbourhood spans more than `2 * relief`,
     so no single platform covers a lot there however it is placed. Inert at the default
     `window=0`, which is "this column only"; the caller passes its own lot's radius.
  4. **unreached** -- of the ground that survives 1-3, the connected components no route
     cell touches. Reported, and deliberately **left in the mask**: a component cut off
     outside the section is a different failure from a door that fails inside it, and
     collapsing them is how 973 unreachable stances came to have no attributable cause.

Absence is never an answer. With no volume to read, `terrain` returns a mask that refuses
nothing, `"level": None`, `"measured": False` and a `why` that says the ground was not
read; every measured record carries `"measured": True` and every count it could not take
is `None` rather than 0. A caller must always be able to tell "no terrain was read" from
"the terrain is fine", because the first of those is what `ground=None` looked like.

`pack`/`unpack`/`record`/`mask_of`/`cover` are the persistence half: a district's answer
belongs on its record in `plan.place.json`, and a 183x69 bitmap is about 600 bytes packed,
which is a thing a plan can carry and a live numpy array is not.
"""
from __future__ import annotations

import base64
import weakref
import zlib

import numpy as np

from .ground import RELIEF as _GROUND_RELIEF, bed_heights, waterlines

#: How far a platform's level may stand from its own ground as found, re-exported from
#: `ground.RELIEF` (which restates `Builder.SITE_RELIEF`). Not a tolerance of this
#: module's own: `ground._held_level` clamps a footprint asked for more than this back
#: to its own ground, so a column further than `RELIEF` from the design's level is a
#: column where the design's level will not be delivered. One rule, one number, one
#: owner.
RELIEF = _GROUND_RELIEF

#: The clauses, in the order they own a column. The first one to fire decides it -- a
#: column reclaimed by fill under clause 1 is not then re-refused by clause 2 for the
#: depth of its own fill -- and `reasons["order"]` publishes this so a reader of the
#: record does not have to infer it from the counts.
CLAUSES = ("wet", "off_level", "broken", "unreached")

#: The largest enclosed pocket below a design's level that is filled whatever its depth
#: (`terrain`'s pit rule): nine by nine, a shaft or a sink in a terrace and not a valley
#: or a lake. Registered after measuring the one it was for: 57 columns, 19 deep.
PIT_COLUMNS = 81

#: How far beyond a rectangle a pocket is looked for, so one its edge crosses is whole.
PIT_PAD = 8

#: ...and how deep, in reaches: a shaft three reaches deep is filled, a well deeper than
#: that stays a well. With the size bound, the pit rule's worst case is 81 columns of 24
#: blocks, and a lattice of deep shafts is not quietly made a plateau.
PIT_DEPTH_REACHES = 3

#: **...and a knoll** (the design resolution round): the same pocket rule above the
#: level. A boulder of ground five columns across and ten blocks over a terrace's level
#: refused the market piece's market (the one lot in from its corner it could stand on)
#: at the ring's own level, and the piece was laid two blocks higher to step round it --
#: a ten- block retaining face to its neighbour for twenty columns of cut. A pocket over
#: the bound of at most `KNOLL_COLUMNS` columns, found whole on the padded window so a
#: hillside's edge is never a pocket, and at most `KNOLL_DEPTH_REACHES` reaches over the
#: level, is cut; `knolls` costs it.
KNOLL_COLUMNS = 64
#: How many columns in from each side a slot below the level is filled.
SLOT_PASSES = 2
KNOLL_DEPTH_REACHES = 2

#: The fields a consumer may assert it read the one it meant to read, as
#: `placeregion.COLUMN_FIELDS` does for the four columns of a region.
TERRAIN_FIELDS = ("columns", "feasible_columns", "wet_columns", "reclaimed_columns",
                  "off_level_columns", "broken_columns", "unreached_columns",
                  "outside_columns", "measured")

#: **What a design does to one column of ground it has designed.** The neighbourhood
#: delivery round, and the half of the question this module did not answer. The mask
#: above says which columns may be **developed**. Nothing said which columns would be
#: **moved**, so `Builder.terrace_annulus` levelled every column of the rectangle it was
#: handed and the two instruments disagreed by 3,767,160 blocks of cut on the retained
#: section's rings alone -- housing excluded from a hillside because the hillside should
#: not be cut, and the hillside cut anyway. These eight answers are exhaustive over a
#: rectangle and they partition it: four the design moves, three it leaves exactly where
#: it found them, one it never read. `level` already at the level: nothing is moved
#: `fill` dry ground below the level, brought up to it `reclaim` standing water the
#: level stands over, filled from the bed (clause 1) `cut` ground above the level, taken
#: down to it `keep_water` water this level does not stand over, or could not fill
#: inside the bound: it stays water `keep_high` ground further above the level than the
#: bound: left standing `keep_low` ground further below the level than the bound: left
#: open `unread` outside the observed volume: never moved on ground nobody read A column
#: is moved **iff** it is in the mask, which is the whole point: the columns `terrain`
#: refuses are exactly the columns the builder does not touch, so the two reconcile
#: column for column rather than by argument.
DISPOSITIONS = ("level", "fill", "reclaim", "cut",
                "keep_water", "keep_high", "keep_low", "unread")

#: The four of `DISPOSITIONS` that move earth, and the four that do not.
MOVED = DISPOSITIONS[:4]
KEPT = DISPOSITIONS[4:]


# ------------------------------------------------------- the ground as found, once

class Reading:
    """The observed ground under one volume: the bed, the waterline, and the wet mask.

        `ground.bed_heights` and `ground.waterlines` are the authority here and not
        `observe.ground_heights`, on purpose. The router's heightmap answers *"what would a
        lane be laid on"* and so reports the **water surface** over a pond; the resolver's
        `_held_level` sounds to the **bed**, and this module exists to agree with the
        resolver. On the retained baseline the two disagree about no column's wetness
        (both 14.18%), which is the check that says composing them is safe; they disagree
        about the height of every wet column, which is the reason to be careful.
        
    """

    __slots__ = ("x0", "z0", "shape", "bed", "water", "wet")

    def __init__(self, vol):
        self.x0, self.z0 = int(vol.x0), int(vol.z0)
        self.bed = np.asarray(bed_heights(vol), int)
        self.water = np.asarray(waterlines(vol), int)
        self.wet = self.water > self.bed
        self.shape = self.bed.shape


#: One memo, weakly held. Reading an 864-square baseline is about a second and a
#: district measurement is a slice of it, so a suite that asks about five districts
#: should read the volume once. Weak, because the arrays are cheap to keep and the
#: volume is not.
_MEMO: list = []


def reading(vol) -> Reading:
    """The `Reading` of `vol`, memoised while the caller still holds the volume.

        Accepts a `Reading` and hands it straight back, so a caller with many rectangles can
        read once and pass the reading everywhere `vol` is taken.
        
    """
    if isinstance(vol, Reading):
        return vol
    for ref, got in _MEMO:
        if ref() is vol:
            return got
    got = Reading(vol)
    _MEMO[:] = [(weakref.ref(vol), got)][:1]
    return got


# ------------------------------------------------------------------ the measurement

def _rect(rect) -> tuple:
    """An inclusive `(x0, z0, x1, z1)`, ordered."""
    a, b, c, d = (int(v) for v in rect)
    return min(a, c), min(b, d), max(a, c), max(b, d)


def _span(bed, window: int, valid=None):
    """The range of the bed over a `window`-radius square at every column.

        `scipy.ndimage` at the edges reflects the nearest value rather than inventing ground
        outside the rectangle: a lot lies inside the district it is drawn in, so the ground
        that decides whether one platform covers it is the ground inside the rectangle.
        Columns the volume never read are excluded from both filters rather than counted as
        y=0, which would make every rectangle that leaves the baseline look broken.
        
    """
    if window <= 0:
        return np.zeros(bed.shape, int)
    from scipy import ndimage
    n = 2 * int(window) + 1
    big = np.iinfo(np.int32).max // 4
    up = bed if valid is None else np.where(valid, bed, -big)
    down = bed if valid is None else np.where(valid, bed, big)
    hi = ndimage.maximum_filter(up, size=n, mode="nearest")
    lo = ndimage.minimum_filter(down, size=n, mode="nearest")
    return np.where(np.abs(hi) >= big, 0, hi - lo).astype(int)


def _unmeasured(rect, why: str) -> dict:
    """The record of a rectangle whose ground was not read: refuses nothing, measures
    nothing, and says which of those two it is doing."""
    x0, z0, x1, z1 = _rect(rect)
    w, d = x1 - x0 + 1, z1 - z0 + 1
    return {"mask": np.ones((w, d), bool), "origin": [x0, z0], "shape": [w, d],
            "level": None, "relief": int(RELIEF), "window": 0,
            "columns": int(w * d), "measured": False,
            "feasible_columns": None, "wet_columns": None, "reclaimed_columns": None,
            "off_level_columns": None, "broken_columns": None,
            "unreached_columns": None, "outside_columns": None,
            "reasons": {"order": list(CLAUSES),
                        **{k: None for k in CLAUSES}, "reclaimed": None,
                        "outside": None, "bed": None, "waterline": None,
                        "components": None, "reached_components": None},
            "by": "ethoslm.feasible.terrain",
            "why": [f"0. the ground was not read ({why}): this record refuses nothing "
                    f"and measures nothing, and `measured` is false so no consumer can "
                    f"read it as ground that was found to be fine"]}


def terrain(vol, rect, *, level=None, relief: int = RELIEF, routes=None,
            window: int = 0, fill: int | None = None) -> dict:
    """Which columns of `rect` construction can stand on under this design's own ground.

        `vol` is the observed baseline (an `ethoslm.offline`/`observe` Volume, or a `Reading`
        of one), `rect` an inclusive `(x0, z0, x1, z1)` in world columns, `level` the level
        this design brings that ground to -- a ring's terrace -- or None where it brings it
        to none. `routes` is any iterable of `(x, z)` cells that reach this rectangle: the
        arterial, the lanes. `window` is the radius of the neighbourhood clause 3 asks about;
        the default of 0 makes it inert, and the caller passes its own lot's radius.

        Returns the record described in this module's docstring. `mask[i, j]` is the column
        `(x0 + i, z0 + j)` and is True where a building may be founded -- including where
        clause 4 found it unreached, which is counted and never silently removed.
        
    """
    if vol is None:
        return _unmeasured(rect, "no volume")
    x0, z0, x1, z1 = _rect(rect)
    w, d = x1 - x0 + 1, z1 - z0 + 1
    r = reading(vol)
    relief = int(relief)
    lvl = None if level is None else int(level)

    # The rectangle's window on the baseline. A district may be laid past the edge of
    # the volume that was observed -- the section's own rectangles are not -- and those
    # columns have no reading at all, so they are their own refusal (`outside`) and are
    # never counted as ground that was found to be good or found to be bad.
    i0, j0 = x0 - r.x0, z0 - r.z0
    ai0, aj0 = max(0, i0), max(0, j0)
    ai1, aj1 = min(r.shape[0], i0 + w), min(r.shape[1], j0 + d)
    inside = np.zeros((w, d), bool)
    bed = np.zeros((w, d), int)
    water = np.zeros((w, d), int)
    if ai1 > ai0 and aj1 > aj0:
        sl = (slice(ai0 - i0, ai1 - i0), slice(aj0 - j0, aj1 - j0))
        inside[sl] = True
        bed[sl] = r.bed[ai0:ai1, aj0:aj1]
        water[sl] = r.water[ai0:ai1, aj0:aj1]
    wet = inside & (water > bed)

    # --- 1. wet. Refused, unless this design's level stands above the waterline, in
    # which case the water is reclaimed by fill and the column is feasible. The fill is
    # the design's explicit choice and it owns the column: clause 2 does not then refuse
    # the reclaimed ground for the depth of the fill that reclaimed it, because the fill
    # *is* the ground there. `reclaimed_fill` records how deep it has to be, so a caller
    # can see the difference between filling a puddle and filling a lake. **...and the
    # fill has a bound, the coordinator's addition.** The clause as first written had no
    # stop: measured on the retained section, the middle ring's level of y=79 reclaims
    # 4,934 columns of `middle_ring_north_east` under up to **44 blocks** of fill and
    # calls every one of them feasible. That is not an earthwork, it is a dam, and a
    # design that can buy any amount of ground by raising the water is back to deriving
    # a count from ground nothing can be built on. `fill` is the most a reclamation may
    # raise the bed; None leaves the clause unbounded, which is what a caller asking
    # "what is under this ground" wants and what every existing call does.
    reclaimed = ((wet & (water + 1 <= lvl)) if lvl is not None
                 else np.zeros((w, d), bool))
    if fill is not None and lvl is not None:
        reclaimed &= (lvl - bed) <= int(fill)
    refused_wet = wet & ~reclaimed
    decided = refused_wet | reclaimed

    # --- 2. off_level. `_held_level` clamps a footprint asked for a level more than
    # `relief` from its own found ground back to that ground, so this is not a matter of
    # how much earth the design is willing to move: the design's level is not delivered
    # here and the lot's floor and its door stand where the hillside was.
    off = np.zeros((w, d), bool)
    if lvl is not None:
        off = inside & ~decided & (np.abs(bed - lvl) > relief)
    # **...except a pit.** The quarter design round. The bound is about hillsides and
    # lakes -- earth a design should not move -- and it also refused a shaft five
    # columns across and nineteen deep in the middle of a terraced quarter's own street:
    # left as found, the street dipped eleven blocks into it and the doorways facing it
    # could not be walked into. A pocket below the level of at most `PIT_COLUMNS`
    # columns -- measured whole, on a window `PIT_PAD` wider than the rectangle, so a
    # trench or a valley is never a pocket -- is filled to the level whatever its depth;
    # its columns are this design's ground, and `pits` says how many and how deep, so
    # the earthwork is costed rather than hidden.
    pits = {"columns": 0, "pockets": 0, "max_fill": 0}
    if lvl is not None and PIT_COLUMNS:
        from scipy import ndimage
        # found on a window `PIT_PAD` wider than the rectangle, so a pocket the
        # rectangle's edge happens to cross is still seen whole, and enclosed or not
        P = PIT_PAD
        e0, f0 = max(0, i0 - P), max(0, j0 - P)
        e1, f1 = min(r.shape[0], i0 + w + P), min(r.shape[1], j0 + d + P)
        if e1 > e0 and f1 > f0:
            eb, ew = r.bed[e0:e1, f0:f1], r.water[e0:e1, f0:f1]
            ewet = ew > eb
            elow = (ewet & ~((ew + 1 <= lvl) & ((lvl - eb) <= (fill if fill is not None
                                                                else 10 ** 6))))
            elow |= (~ewet & (lvl - eb > relief))
            elow &= eb < lvl
            lab, n = ndimage.label(elow)
            deep_ok = int(PIT_DEPTH_REACHES) * int(relief)
            for k in range(1, n + 1):
                m_e = lab == k
                size = int(m_e.sum())
                if size > PIT_COLUMNS:
                    continue
                ii, jj = np.nonzero(m_e)
                m = np.zeros((w, d), bool)
                for a, b in zip(ii + e0 - i0, jj + f0 - j0):
                    if 0 <= a < w and 0 <= b < d:
                        m[a, b] = True
                m &= inside & (refused_wet | off)
                if not m.any() or int((lvl - bed[m]).max()) > deep_ok:
                    continue
                reclaimed |= m & wet
                refused_wet &= ~m
                off &= ~m
                pits["columns"] += int(m.sum())
                pits["pockets"] += 1
                pits["max_fill"] = max(pits["max_fill"], int((lvl - bed[m]).max()))
        decided = refused_wet | reclaimed
    knolls = {"columns": 0, "pockets": 0, "max_cut": 0}
    if lvl is not None and KNOLL_COLUMNS and off.any():
        from scipy import ndimage
        P = PIT_PAD
        e0, f0 = max(0, i0 - P), max(0, j0 - P)
        e1, f1 = min(r.shape[0], i0 + w + P), min(r.shape[1], j0 + d + P)
        if e1 > e0 and f1 > f0:
            eb, ew = r.bed[e0:e1, f0:f1], r.water[e0:e1, f0:f1]
            high = (ew <= eb) & (eb - lvl > relief)
            lab, n = ndimage.label(high)
            deep_ok = int(KNOLL_DEPTH_REACHES) * int(relief)
            for k in range(1, n + 1):
                m_e = lab == k
                if int(m_e.sum()) > KNOLL_COLUMNS:
                    continue
                ii, jj = np.nonzero(m_e)
                m = np.zeros((w, d), bool)
                for a, b in zip(ii + e0 - i0, jj + f0 - j0):
                    if 0 <= a < w and 0 <= b < d:
                        m[a, b] = True
                m &= inside & off
                if not m.any() or int((bed[m] - lvl).max()) > deep_ok:
                    continue
                off &= ~m
                knolls["columns"] += int(m.sum())
                knolls["pockets"] += 1
                knolls["max_cut"] = max(knolls["max_cut"], int((bed[m] - lvl).max()))
    # **...and a slot** (the design resolution round, the independent reader's n1): a
    # pocket below the level too large for the pit rule can still be one or two columns
    # wide where it meets the terrace -- a shaft nine to sixteen deep along the ring
    # street's edge, left open because the whole hole it belonged to was more than
    # `PIT_COLUMNS`. A low column with feasible ground on both sides along one axis is
    # filled, `SLOT_PASSES` columns in from each side; `slots` costs it.
    slots = {"columns": 0}
    if lvl is not None and SLOT_PASSES and off.any():
        low = off & (bed < lvl)
        for _ in range(int(SLOT_PASSES)):
            okm = inside & ~off & ~refused_wet
            px = np.zeros_like(okm)
            pz = np.zeros_like(okm)
            px[1:-1, :] = okm[:-2, :] & okm[2:, :]
            pz[:, 1:-1] = okm[:, :-2] & okm[:, 2:]
            take = low & (px | pz) & ((lvl - bed) <= int(PIT_DEPTH_REACHES) * int(relief))
            if not take.any():
                break
            off &= ~take
            low &= ~take
            slots["columns"] += int(take.sum())
    decided = decided | off

    # --- 3. broken. One lot is one platform, and one platform is one level: where the
    # bed under a lot-sized neighbourhood spans more than the platform can reach in both
    # directions (`2 * relief`), no placement of a lot there is founded on it. Measured
    # on the bed the design's own proposal leaves -- reclaimed water stands at `level`,
    # because that is what reclaiming it means -- so a filled edge is not then reported
    # as broken against the bed it no longer has.
    eff = bed if lvl is None else np.where(reclaimed, lvl, bed)
    broken = np.zeros((w, d), bool)
    if window and window > 0:
        broken = inside & ~decided & (_span(eff, int(window), inside) > 2 * relief)
    decided = decided | broken

    # `decided` is what the clauses above have *owned*; `refused` is what they refused.
    # The two differ by exactly the reclaimed columns, which clause 1 decided in the
    # design's favour: they are feasible ground and they are also out of the reach of
    # clauses 2 and 3, because the fill is the ground there.
    feasible = inside & ~(refused_wet | off | broken)

    # --- 4. unreached. The connected components (4-neighbour) of the feasible ground
    # that no route cell touches. Left **in** the mask on purpose: this clause separates
    # "a door fails inside the section" from "a component of the fabric is disconnected
    # from everything outside it", and the review's finding is precisely that those two
    # were one undifferentiated number.
    unreached = None
    comps = reached = None
    if routes is not None:
        from scipy import ndimage
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        lab, n = ndimage.label(feasible, structure=cross)
        comps = int(n)
        touch = np.zeros((w, d), bool)
        for c in routes:
            i, j = int(c[0]) - x0, int(c[1]) - z0
            # a route cell one column outside the rectangle still reaches the rectangle:
            # the arterial runs *beside* a district far more often than through it
            for (a, b) in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < w and 0 <= b < d:
                    touch[a, b] = True
        hit = sorted(set(np.unique(lab[touch & (lab > 0)]).tolist()))
        reached = len(hit)
        got = int(np.isin(lab, hit).sum()) if hit else 0
        unreached = int(feasible.sum()) - got

    n_wet = int(refused_wet.sum())
    n_rec = int(reclaimed.sum())
    n_off = int(off.sum())
    n_brk = int(broken.sum())
    n_out = int((~inside).sum())
    n_ok = int(feasible.sum())
    fill = (int((lvl - bed)[reclaimed].max()) if n_rec and lvl is not None else None)
    beds = bed[inside]
    why = [
        f"1. water: {n_wet} column(s) refused as standing water"
        + (f" and {n_rec} reclaimed by fill, the design's level y={lvl} standing at or "
           f"above the waterline + 1 (up to {fill} block(s) of fill)" if n_rec else
           (" and none reclaimed: this design brings no level to this ground"
            if lvl is None else
            f" and none reclaimed: this design's level y={lvl} does not stand above "
            f"the waterline here")),
        (f"2. level: {n_off} column(s) where the design's level y={lvl} is more than "
         f"relief {relief} from the ground as found, so `ground._held_level` clamps a "
         f"footprint back to its own ground and the level is not delivered"
         if lvl is not None else
         "2. level: not measured -- this design brings no level to this ground, so "
         "there is no level for `_held_level` to fail to deliver"),
        (f"3. relief: {n_brk} column(s) whose bed over a {2 * int(window) + 1}-column "
         f"neighbourhood spans more than {2 * relief}, so no one platform covers a lot "
         f"there" if window else
         "3. relief: not measured at window=0 -- this clause is inert until a caller "
         "passes its own lot's radius"),
        (f"4. reach: {unreached} feasible column(s) in component(s) no route cell "
         f"touches, of {comps} component(s) of which {reached} are reached; they stay "
         f"in the mask, because ground cut off outside the section is not a door that "
         f"failed inside it" if unreached is not None else
         "4. reach: not measured -- no routes were given, so no component can be said "
         "to be reached or unreached"),
    ]
    if n_out:
        why.append(f"0. {n_out} column(s) of this rectangle lie outside the observed "
                   f"volume and have no reading: refused, and counted apart from the "
                   f"ground that was read")
    return {
        "mask": feasible, "origin": [x0, z0], "shape": [w, d],
        "level": lvl, "relief": relief, "window": int(window),
        "fill_max": None if fill is None else int(fill),
        "columns": int(w * d), "measured": True,
        "feasible_columns": n_ok, "wet_columns": n_wet, "reclaimed_columns": n_rec,
        "off_level_columns": n_off, "broken_columns": n_brk,
        "unreached_columns": unreached, "outside_columns": n_out,
        "pits": pits,
        "knolls": knolls,
        "slots": slots,
        "reasons": {
            "order": list(CLAUSES),
            "wet": n_wet, "off_level": n_off, "broken": n_brk, "unreached": unreached,
            "reclaimed": n_rec, "reclaimed_fill": fill, "outside": n_out,
            "components": comps, "reached_components": reached,
            "bed": ([int(beds.min()), int(np.median(beds)), int(beds.max())]
                    if beds.size else None),
            "waterline": (int(water[wet].max()) if wet.any() else None),
            "of": "ethoslm.ground.bed_heights / waterlines -- the bed the resolver "
                  "sounds to, not the router's surface over a pond",
        },
        "by": "ethoslm.feasible.terrain", "why": why,
    }


# --------------------------------------------------- what the design does to it
# `terrain` answers "may this column carry a building". `dispositions` answers "and what
# does this design do to it", over the same rectangle at the same level with the same
# bound -- derived from `terrain`'s own published mask and never from a second rule, so
# the two cannot drift apart. The builder executes this; the record publishes it; a
# reader reconciles the earthwork against the mask column for column.

def dispositions(vol, rect, *, level, relief: int = RELIEF, fill: int | None = None,
                 window: int = 0, routes=None) -> dict:
    """The `DISPOSITIONS` of every column of `rect` under a design that brings it to
        `level`, moving no column further than `relief` (cut) or `fill` (reclamation).

        `level` is required: a piece of ground with no level is ground this design does not
        design, and there is no disposition to publish for it. Returns

            code      a `uint8` array, `code[i, j]` indexing `DISPOSITIONS`, for the
                      column `(origin[0] + i, origin[1] + j)`
            mask      `terrain`'s own mask -- True exactly where `code` is one of `MOVED`
            counts    every name of `DISPOSITIONS` to its column count, summing to `columns`
            cut_blocks / fill_blocks, max_cut / max_fill
                      the earthwork the moved columns alone come to, and the deepest single
                      column of it: `max_cut` and `max_fill` are the numbers that have to be
                      inside the bound, and they are measured and not asserted
            terrain   `terrain`'s record over the same ground, without its live array

        The reconciliation a consumer may assert, and `why` states it:

            level + fill + reclaim + cut  == terrain.feasible_columns
            keep_water                    == terrain.wet_columns
            unread                        == terrain.outside_columns
            keep_high + keep_low          == the rest, which clauses 2 and 3 refused
        
    """
    if level is None:
        raise ValueError("dispositions() needs the level this design brings the ground "
                         "to: ground with no level is ground this design does not design")
    got = terrain(vol, rect, level=level, relief=relief, fill=fill, window=window,
                  routes=routes)
    mask = got.pop("mask")
    x0, z0 = int(got["origin"][0]), int(got["origin"][1])
    w, d = mask.shape
    lvl = int(level)
    code = np.full((w, d), DISPOSITIONS.index("unread"), np.uint8)
    if not got.get("measured"):
        # No volume was read. Nothing is refused, so nothing is *kept* either: this
        # record measures nothing and says so, exactly as `_unmeasured` does.
        return {"code": code, "mask": mask, "origin": [x0, z0], "shape": [w, d],
                "level": lvl, "relief": int(relief),
                "fill": None if fill is None else int(fill),
                "columns": int(w * d), "measured": False,
                "counts": {n: (int(w * d) if n == "unread" else 0)
                           for n in DISPOSITIONS},
                "moved_columns": None, "kept_columns": None,
                "cut_blocks": None, "fill_blocks": None,
                "max_cut": None, "max_fill": None,
                "terrain": got, "by": "ethoslm.feasible.dispositions",
                "why": ["the ground was not read: no column has a disposition and none "
                        "is reported as kept, because keeping ground is a decision and "
                        "this record took none"]}
    r = reading(vol)
    i0, j0 = x0 - r.x0, z0 - r.z0
    ai0, aj0 = max(0, i0), max(0, j0)
    ai1, aj1 = min(r.shape[0], i0 + w), min(r.shape[1], j0 + d)
    inside = np.zeros((w, d), bool)
    bed = np.zeros((w, d), int)
    water = np.zeros((w, d), int)
    if ai1 > ai0 and aj1 > aj0:
        sl = (slice(ai0 - i0, ai1 - i0), slice(aj0 - j0, aj1 - j0))
        inside[sl] = True
        bed[sl] = r.bed[ai0:ai1, aj0:aj1]
        water[sl] = r.water[ai0:ai1, aj0:aj1]
    wet = inside & (water > bed)
    idx = {n: DISPOSITIONS.index(n) for n in DISPOSITIONS}
    # the four the design moves, read off `terrain`'s own mask and nothing else
    code[mask & wet] = idx["reclaim"]
    code[mask & ~wet & (bed < lvl)] = idx["fill"]
    code[mask & ~wet & (bed > lvl)] = idx["cut"]
    code[mask & ~wet & (bed == lvl)] = idx["level"]
    # ...and the three it leaves alone. `unread` is already the fill value, so a column
    # outside the observed volume keeps it whatever else is true of the zeros there.
    kept = inside & ~mask
    code[kept & wet] = idx["keep_water"]
    code[kept & ~wet & (bed > lvl)] = idx["keep_high"]
    code[kept & ~wet & (bed <= lvl)] = idx["keep_low"]
    counts = {n: int((code == idx[n]).sum()) for n in DISPOSITIONS}
    moved = mask & inside
    cut_b = int(np.where(moved & (bed > lvl), bed - lvl, 0).sum())
    fill_b = int(np.where(moved & (bed < lvl), lvl - bed, 0).sum())
    max_cut = int((bed - lvl)[moved & (bed > lvl)].max()) if (moved & (bed > lvl)).any() else 0
    max_fill = int((lvl - bed)[moved & (bed < lvl)].max()) if (moved & (bed < lvl)).any() else 0
    n_moved = int(sum(counts[n] for n in MOVED))
    n_kept = int(sum(counts[n] for n in KEPT))
    return {
        "code": code, "mask": mask, "origin": [x0, z0], "shape": [w, d],
        "level": lvl, "relief": int(relief),
        "fill": None if fill is None else int(fill),
        "columns": int(w * d), "measured": True,
        "counts": counts, "moved_columns": n_moved, "kept_columns": n_kept,
        "cut_blocks": cut_b, "fill_blocks": fill_b,
        "max_cut": max_cut, "max_fill": max_fill,
        "terrain": got, "by": "ethoslm.feasible.dispositions",
        "why": [
            f"{n_moved} column(s) of {w * d} are moved to y={lvl} "
            f"({counts['level']} already there, {counts['fill']} filled, "
            f"{counts['reclaim']} reclaimed from under water, {counts['cut']} cut): "
            f"{fill_b} block(s) of fill and {cut_b} of cut, the deepest single column "
            f"{max_fill} of fill and {max_cut} of cut against a bound of "
            f"{relief} (cut) and {fill} (fill)",
            f"{n_kept} column(s) are left exactly as found ({counts['keep_water']} "
            f"water this level does not stand over, {counts['keep_high']} ground the "
            f"design will not cut, {counts['keep_low']} ground it will not fill, "
            f"{counts['unread']} outside the observed volume)",
            f"reconciles with the mask: level+fill+reclaim+cut = {n_moved} = "
            f"feasible_columns {got.get('feasible_columns')}; keep_water = "
            f"{counts['keep_water']} = wet_columns {got.get('wet_columns')}; unread = "
            f"{counts['unread']} = outside_columns {got.get('outside_columns')}",
        ],
    }


# ----------------------------------------------------------------- the persistence A
# district's answer belongs on the district's own record, beside the four columns of
# `placeregion`, or the next stage reads the rectangle again and gets a fifth different
# number. What a plan can carry is a packed bitmap: 183x69 is 12,627 bits, 1,579 bytes
# packed, and under a kilobyte through zlib and base64 -- against roughly 40 kB as a
# list of booleans and a live numpy array that `json.dumps` cannot write at all.

def pack(mask, origin) -> dict:
    """A boolean mask as `{"origin", "shape", "bits"}`, JSON-safe. `unpack` inverts it."""
    m = np.asarray(mask, bool)
    raw = zlib.compress(np.packbits(m.ravel()).tobytes(), 9)
    return {"origin": [int(origin[0]), int(origin[1])],
            "shape": [int(m.shape[0]), int(m.shape[1])],
            "bits": base64.b64encode(raw).decode("ascii")}


def unpack(rec: dict) -> tuple:
    """`(mask, [x0, z0])` from what `pack` wrote, or from the record that carries it.

        Both, so a caller holding a district record does not have to know the field the
        bitmap is under: it is the same answer either way.
        
    """
    rec = rec.get("mask_bits") or rec
    w, d = int(rec["shape"][0]), int(rec["shape"][1])
    raw = zlib.decompress(base64.b64decode(rec["bits"]))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), count=w * d)
    return bits.astype(bool).reshape(w, d), [int(rec["origin"][0]),
                                             int(rec["origin"][1])]


def record(vol, rect, **kw) -> dict:
    """`terrain()` as a record a plan can carry: the live array out, `mask_bits` in.

        Plain ints, strings and lists throughout, so it round-trips exactly through
        `json.dumps`/`json.loads` and back to a mask through `mask_of`.
        
    """
    got = terrain(vol, rect, **kw)
    mask = got.pop("mask")
    got["mask_bits"] = pack(mask, got["origin"])
    return got


def mask_of(rec: dict):
    """The mask of a record `record()` wrote, or None where it carries none.

        None and not an all-true mask: a record without a bitmap is a record whose ground
        nobody measured, and answering "all of it" would be the `ground=None` defect again.
        
    """
    if not rec or not rec.get("mask_bits"):
        return None
    return unpack(rec["mask_bits"])[0]


def cover(rec: dict, sub_rect) -> dict:
    """`{"columns", "feasible_columns"}` of a sub-rectangle of a measured record.

        What a lot, a block or a quarter asks of the district's answer. `feasible_columns`
        is None where the record was never measured or does not cover this ground, which the
        caller has to handle rather than read as zero.
        
    """
    x0, z0, x1, z1 = _rect(sub_rect)
    w, d = x1 - x0 + 1, z1 - z0 + 1
    out = {"columns": int(w * d), "feasible_columns": None,
           "measured": bool(rec.get("measured")) if rec else False,
           "outside_columns": None}
    mask = mask_of(rec) if rec else None
    if mask is None or not out["measured"]:
        out["why"] = ("this ground was not measured: `feasible_columns` is unmeasured "
                      "and is not zero")
        return out
    origin = (rec.get("mask_bits") or rec).get("origin") or rec.get("origin")
    ox, oz = int(origin[0]), int(origin[1])
    i0, j0 = max(0, x0 - ox), max(0, z0 - oz)
    i1, j1 = min(mask.shape[0], x1 - ox + 1), min(mask.shape[1], z1 - oz + 1)
    got = mask[i0:i1, j0:j1] if (i1 > i0 and j1 > j0) else np.zeros((0, 0), bool)
    out["feasible_columns"] = int(got.sum())
    out["outside_columns"] = int(w * d - got.size)
    if out["outside_columns"]:
        out["why"] = (f"{out['outside_columns']} column(s) of this sub-rectangle lie "
                      f"outside the measured record and are not in either count")
    return out
