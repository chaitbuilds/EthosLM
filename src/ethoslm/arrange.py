"""**One construction logic answers "how many" and "which ones".**

The realization round's second boundary, and the review's second finding:

  > `placeplan.fabric_fit` predicts a count from generic lot/block arithmetic. It does
  > not consume the selected type envelopes, actual lot arrangements, terrain mask or
  > access constraints. Extent repair reuses this estimate, so it can enlarge a region
  > and promise more of the same unrealizable housing.
  >
  > **Required boundary:** planning needs an actual feasible arrangement, or a reasoned
  > failure, from the same construction logic that will realize it. Counts alone are
  > not capacity certificates.

So there is no second opinion here. `arrange` runs `district_compile` -- the pass that
actually lays the streets, the blocks and the lots that get built -- and returns what it
laid. The number of houses a district is asked for is the number that pass produced on
that ground with those types, and the file it wrote is the file that is built. A
district's promise and its plan are the same act.

What was in the way was cost, not principle: one compile of an 84x35 rectangle took 2.9
seconds and the compiler's own search runs it up to 28 times, so asking the real thing
cost 96 seconds a district. Almost all of it was `pipeline.load_type` executing type
source 768 times per compile; cached, the same district arranges in 0.16 seconds and
planning can afford to ask.

**The recovery ladder lives here too**, and that is the other half of the finding. An
arrangement short of its proposal is not adopted until the two spatial answers have been
tried on the ground itself -- a bigger rectangle where there is free ground beside it,
and a smaller-footprint approved type -- each re-arranged and *measured*, so a repair
that changed nothing is visible as one. Only then does the promise move, and what it
moves to is what the ground gave. `attempts` is the record of that ladder: every rung,
what it changed, and how many lots came back.

Nothing here decides *policy*. Which rectangle a district is and which types it may use
are the layout's and the capability record's; this says what can be built on that.
"""
from __future__ import annotations

import contextlib

#: How much bigger a district may be grown, as a share of the rectangle it was **first**
#: laid out as, and the least gain in columns worth taking. Bounded against the original
#: so that a ladder run twice is not a ladder with twice the budget.
GROWTH_MAX = 0.5
GROWTH_MIN_EDGE = 4

#: The rungs of the ladder, in the order they are tried. Ground and types before the
#: promise, which is the order the review requires and the order a designer would use.
#: `fill` is the rung a *cover* shortfall needs: a district short of the ground its
#: density asks it to cover, whose own rectangle measurably holds more houses than it
#: was asked for, is asked for them. It is tried first because it changes nothing about
#: the place -- not the rectangle, not the approved pool -- and the other two do.
ACTIONS = ("fill", "extent", "fabric")

#: **The child arrangements a parent may negotiate over**, and the whole list. The
#: expression round's named dead end, in one line: "the lower ring stands at its least
#: width (insets 6+3 plus a district depth of 28 where the count needs 22), and the
#: fabric's wider lot changed nothing because the attached row house is a fixed six-wide
#: bay. they were simply not on anybody's list. A district's depth is two rows of lots
#: because nothing ever asked for one; a row house's bay is six because the density
#: word's own lot side is six and the type admits up to ten. row_depth one row of houses
#: along a street rather than two back to back. The parent gets a band a third shallower
#: and the same houses. bay_width the widest (or narrowest) lot the approved type
#: actually admits, rather than the density word's own side. frontage lots a lane apart
#: on open frontage rather than a clearance apart on a street -- more ground a house,
#: which is what a loose fabric means. compound the houses about a court rather than
#: along a street: a courtyard block, where the role's types and the block's depth admit
#: one. Every one of them is **proposed** here and **certified** by `district_compile`,
#: which stays the capacity authority. Nothing in this list is a second estimator.
#: compact the same lots on a longer block: fewer street-ends between the houses, which
#: is the only lever that turns the ground *between* blocks into frontage without
#: enlarging a lot (the composition round). **The neighbourhood round's three**, added
#: because the five above could not shape the street the round is about. Each is one
#: decision, named, and certified like the rest: terrace party walls or none: an
#: attached row where the fabric declares a type that can stand against its neighbours,
#: a detached street where it already does. It is the one thing in `ARRANGEMENT_FIELDS`
#: no action proposed -- the field was carried through `character_of` and nothing ever
#: wrote it -- and it is the difference between a continuous street wall and buildings
#: with gaps between them. compact_bay the narrowest bay the fabric admits, one row to a
#: block, on the longest block: all three compaction levers at once. Every action above
#: moves one lever from the character's own declaration, so the best crowded street any
#: of them could offer was the best *single* move -- and a crowded quarter is not one
#: move from a loose one. Composed deliberately and named as a composition. perimeter
#: lots on all four faces of a block with the court in the middle: a perimeter block.
#: `compound` raises the courtyard share, which makes the *back row* of a block its
#: court -- open on two sides, fronting nothing, and no lot has ever been laid along the
#: short faces of a block in this compiler. This is the arrangement that puts building
#: on the cross streets and encloses a court on four sides.
ARRANGEMENT_ACTIONS = ("row_depth", "bay_width", "frontage", "compound", "compact",
                       "terrace", "compact_bay", "perimeter")


def _rect_of(d: dict) -> list:
    return [min(d["x0"], d["x1"]), min(d["z0"], d["z1"]),
            max(d["x0"], d["x1"]), max(d["z0"], d["z1"])]


def _area(rect) -> int:
    return (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)


def compile_once(district: dict, part: dict, place: dict, decls: dict, *,
                 spec: dict | None = None, seed: int = 1) -> dict:
    """The arrangement this rectangle actually holds: the compiler's own answer.

        Returns `{"ok", "why", "lots", "plan", "record", ...}`. `plan` is the district file
        that will be built -- not a description of one -- and `lots` is how many houses are
        in it. A rectangle the compiler cannot lay at all is `ok: False` with the reason it
        gave, which is a reasoned failure and not a zero.
        
    """
    from . import district_compile, placeplan
    try:
        got, rec = district_compile.compile_district(district, part, place, decls,
                                                     spec=spec, seed=seed)
    except (ValueError, KeyError) as e:          # noqa: PERF203 -- reported, not raised
        return {"ok": False, "lots": 0, "plan": None, "record": None,
                "rect": _rect_of(district),
                "why": f"the construction logic cannot lay this district: {e}"}
    # a compiled district is `{quarters: [{plots: [...]}]}` at the seam every consumer
    # reads it at, which is `district_plots` and not the assembled tree's flattener
    plots = [p for p in placeplan.district_plots(got, part.get("role"), spec)
             if p.get("kind", "plot") == "plot"]
    ledger = rec.get("assigned") or {}
    return {
        "ok": True, "lots": int(rec.get("lots") or len(plots)), "plan": got,
        "record": rec, "rect": _rect_of(district),
        "house": rec.get("house"), "lot": rec.get("lot"),
        "types": sorted({str(p.get("type")) for p in plots if p.get("type")}),
        # **What the arrangement owes the place beside its houses.** A repair that
        # trades houses for grass, or lanes for houses, is visible in these three.
        "plot_cover": rec.get("plot_cover"), "ground_cover": rec.get("ground_cover"),
        "undeveloped_share": rec.get("undeveloped_share"),
        # **the lots drawn and the mass they admit, apart** (the design round's second
        # contract): a cover figure improved only by enlarging empty lots moves the
        # first and leaves the second where it was
        "allocated_columns": rec.get("allocated_columns"),
        "footprint_columns": rec.get("footprint_columns"),
        # **the pad figure under the name it earns** (the neighbourhood round): the same
        # number as `footprint_columns`, marked as the estimate it is, so a caller that
        # wants the plan's own answer and a caller that wants what stands are asking two
        # different questions rather than the same ambiguous one
        "pad_columns": rec.get("pad_columns"),
        "footprint_estimate": rec.get("footprint_estimate"),
        "footprint_basis": rec.get("footprint_basis"),
        # **what this quarter holds and why** (`district_compile.use_mix`): the inferred
        # programme, so a caller comparing arrangements can see that one of them lost
        # the quarter's work and another kept it
        "programme": (got.get("programme") if isinstance(got, dict) else None),
        "courts": rec.get("courts"),
        "perimeter_blocks": rec.get("perimeter_blocks"),
        # ...and how many of them actually closed on four sides, which is the number the
        # arrangement's claim is about
        "perimeter_shut": rec.get("perimeter_shut"),
        "lot_refused": rec.get("lot_refused"),
        # **What the arrangement could not hold, and what it held instead.** The
        # composition round's first change: a required market that did not fit became a
        # row of houses and nothing anywhere said so. `demand_short` is the unmet demand
        # with feasible alternatives, `reservations` what was asked of the ground before
        # the housing took it, and `shares_spent` the court share the count was met
        # with.
        "demand_short": [dict(x) for x in (rec.get("demand_short") or [])],
        "reservations": [dict(x) for x in (rec.get("reservations") or [])],
        "reservations_kept": rec.get("reservations_kept"),
        "reservations_required": rec.get("reservations_required"),
        "shares_spent": rec.get("shares_spent"),
        "open_land": {k: int(v) for k, v in ledger.items()
                      if k in ("court", "garden", "plaza", "field", "grove", "verge")},
        "streets": int(ledger.get("street") or 0),
        "why": (f"{rec.get('lots')} lot(s) of "
                f"{(rec.get('lot') or ['?', '?'])[0]}x{(rec.get('lot') or ['?', '?'])[1]} "
                f"({rec.get('house')}) on {rec.get('blocks')} block(s)"),
    }


def _lot_for(decl: dict, side: int, depth: int | None, *, attached: bool,
             area: int | None = None, shape: tuple | None = None) -> tuple | None:
    """`(w, d)` of a lot of about `side` (or of about `area` columns) this type admits,
        attached or not. None where nothing does -- which is the honest answer and not a
        smaller lot.

        **A depth the caller named is honoured or refused**, the neighbourhood round. This
        took `_attached_lot`'s answer, tried the asked depth at *that width only*, and
        returned the other depth where it did not stand -- so `arrangements` offered lots the
        compiler would never lay, and the row was then ranked on the cover and the enclosure
        of a fabric it did not lay. Measured on the retained section: `compact_bay` asked for
        8x6 at one row and the compiler laid 64 lots of 8x8.

        **`shape` is the (frontage, depth) the density asked for**, the neighbourhood
        delivery round, and it is the same argument `district_compile._compile_once` has
        passed `_attached_lot` since the spatial design round. The audit's second cause is
        that this function did not: the fresh compiler asked for the shape
        `placeplan.fabric` resolves a dense attached fabric to and got **6x13**, and
        `arrangements` rebuilt its base through this call without it and got **10x10** -- so
        the incumbent and every alternative offered against it were fabrics of two different
        lot models, and a nominal row-count change silently changed the building's geometry
        as well. Two rules about one question, on the two sides of one comparison.
        
    """
    from . import district_compile as dc
    atts = (("west", "east"), ("west",), ("east",), ())
    if attached:
        if depth:
            d2 = int(depth)
            lo_w, _lo_d, hi_w, _hi_d = decl["needs"]["footprint"]
            i = 2 * dc.Builder.SITE_INSET
            # the width nearest the frontage the shape asked for, where one was given,
            # and nearest the area otherwise: the same preference order the compiler's
            # `_attached_lot` ranks by, so the two answer the same question
            ws = sorted(range(max(3, lo_w), hi_w + i + 1),
                        key=lambda v: ((abs(v - int(shape[0])),) if shape else ())
                        + (abs(v * d2 - int(area or side * side)), v))
            w2 = next((w for w in ws
                       if all(dc._admits(decl, w, d2, att, True) for att in atts)), None)
            # no width of this type's band stands at the depth that was asked for: the
            # caller refuses the option by name rather than offering another lot
            return None if w2 is None else (int(w2), d2)
        got = dc._attached_lot(decl, int(side), ("west", "east"), True, area=area,
                               shape=shape)
        return None if got is None else (int(got[0]), int(got[1]))
    w = dc._clamp_side(int(side), decl)
    ld = dc._clamp_side(int(depth or side), decl)
    if not dc._admits(decl, w, ld, None, True):
        ld = dc._deepest(decl, w, ld, (), True) or w
    return int(w), int(ld)


def _depth_band(decl: dict) -> tuple:
    """`(lo, hi)` of a lot's **depth** in plot columns, off the type's declared
    footprint: `needs.footprint` is `(lo_w, lo_d, hi_w, hi_d)` and the two axes are not
    the same question for a type written narrow to the street and deep into the plot.

    `district_compile._plot_range` collapses them (`min(hi_w, hi_d)`) because it is
    answering "what *square* plot does this stand on", which is the right question for a
    clamp and the wrong one for a depth."""
    from .buildlib import Builder
    try:
        lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    except (KeyError, TypeError, ValueError):
        return (3, 3)
    i = 2 * Builder.SITE_INSET
    return (max(3, int(lo_d)), int(hi_d) + i)


def _tight_lot(decl: dict, side: int, *, attached: bool) -> tuple | None:
    """**The smallest lot this type admits**: the narrowest bay at the band's own floor,
        and then the shallowest depth that stands at that bay.

        `_lot_for` answers "a lot of about `side`", and for an attached type `_attached_lot`
        orders its candidates by distance from `side` on **both** axes -- so asked for the
        floor of `row_house`'s 8..10 band it returns 6x8, because 8 is nearer 8 than 6 is.
        That is right for a bay question and wrong for a compaction one: the lot that puts
        the most houses on a rectangle is the narrowest *and* the shallowest the type stands
        on, which for `row_house` is 6x6 -- the arrangement the composition round certified
        and could not get into blocks.
        
    """
    from . import district_compile as dc
    got = _lot_for(decl, int(side), None, attached=attached)
    if got is None:
        return None
    w0 = int(got[0])
    atts = ([("west", "east"), ("west",), ("east",), ()] if attached else [()])
    for d in range(3, int(got[1]) + 1):
        if all(dc._admits(decl, w0, d, att, True) for att in atts):
            return (w0, int(d))
    return (w0, int(got[1]))


def arrangements(part: dict, decls: dict, *, spec: dict | None = None,
                 pool: list | None = None, allocation: dict | None = None,
                 district: dict | None = None) -> list:
    """**The child arrangements this district's fabric really admits**, in order.

        One entry per alternative: the arrangement to write on a district record, the lot it
        lays, how many rows of lots a block holds, and the **district depth** that
        arrangement needs -- which is the number a parent negotiates its own dimension
        against. Returns `[]` where the role has no house type at all.

        Nothing here is a capacity claim. `capacity` -- the actual compiler -- answers how
        many houses each one holds, on the ground the parent proposes for it.

        **Every alternative is generated from the design that was adopted.** The
        neighbourhood delivery round, and the audit's second cause. This read
        `character_of(part)` with no district, so the base fabric it varied was the *brief's*
        fabric and not the one the layout had negotiated for this rectangle -- and the
        `allocation` argument it already took was never used for anything. So the incumbent
        laid 6x13 lots while `compound` and `row_depth` were both offered at **10x10**, and a
        trial that named a row count changed the building as well as the number of rows.
        `district` carries the adopted arrangement (`character_of` puts it over the brief and
        says that it did); `allocation` answers the same question for a caller that has the
        spec's allocation and not the district record.
        
    """
    from . import district_compile as dc, placeplan, spec as spec_mod
    role = part.get("role") or placeplan.DENSITY_ROLE.get(part.get("density") or "medium")
    houses = dc.house_types(decls, role, (spec or {}).get("form"),
                            approved=list(pool or part.get("fabric_types") or []) or None)
    if not houses:
        return []
    here = district
    adopted = dict((district or {}).get("arrangement") or {})
    if not adopted and allocation:
        from . import placesolve as _ps
        got = _ps.adopted_arrangement(allocation, part.get("name"))
        adopted = dict(got or {})
        here = {"arrangement": adopted} if adopted else None
    ch = dc.character_of(part, here)
    density = part.get("density") or "medium"
    declared = {k for k, v in (part.get("character") or {}).items() if v is not None}
    attached = bool(ch.get("attached"))
    name, decl = houses[0]
    if attached:
        # the compiler's terrace (`district_compile.terrace_order`), not the table's
        # order, which made the comparison's base a courtyard house once that type could
        # stand attached (the fabric reset round)
        pick = next(iter(dc.terrace_order([(n, d) for n, d in houses
                                           if d.get("attached")])), None)
        if pick is None:
            attached = False
        else:
            name, decl = pick
    lo_side, hi_side, _ex = dc._plot_range(decl)
    own = int(ch.get("lot_width") or dc._lot_side(density))
    # **The shape the density asks for, where nobody declared one** -- the same call
    # `district_compile._compile_once` makes before it asks `_attached_lot`, so the base
    # this comparison varies is the fabric the compiler would lay and not another one.
    shape = None
    if not (ch.get("lot_width") and ch.get("lot_depth")):
        with contextlib.suppress(Exception):
            shape = tuple(int(v) for v in placeplan.fabric(density, role, ch)["lot"])[:2]
    base = _lot_for(decl, own, ch.get("lot_depth"), attached=attached, shape=shape)
    if base is None:
        return []
    out, seen = [], set()

    def add(action, rows, lot, frontage, courtyard, why, refused=None, block=None,
            attach=None, house=None, perimeter=None):
        """One row of the list. `attach`, `house` and `perimeter` are the neighbourhood
        round's additions: an alternative may change which type the fabric is built of
        (a terrace is a type that declares `ATTACHED`) and whether the lots stand against
        each other, and both of those are already fields `district_compile` reads."""
        who = house or name
        if refused is not None or lot is None:
            out.append({"action": action, "house": who, "arrangement": None,
                        "refused": refused or "this type admits no such lot",
                        "why": why})
            return
        w, ld = lot
        att = bool(attached if attach is None else attach)
        gap = 0 if att else (placeplan.PLOT_LANE if frontage == "open" else dc.LOT_GAP)
        arr = {"rows": int(rows), "lot_width": int(w), "lot_depth": int(ld),
               "frontage": frontage, "courtyard_share": round(float(courtyard), 2),
               **({"block": int(block)} if block else {}),
               **({"attached": att} if attach is not None else {}),
               **({"perimeter": True} if perimeter else {})}
        key = (int(w), int(ld), int(rows), frontage, arr["courtyard_share"],
               int(block or 0), att, bool(perimeter))
        if key in seen:
            return
        seen.add(key)
        out.append({"action": action, "house": who, "attached": att,
                    "arrangement": arr, "lot": [int(w), int(ld)], "rows": int(rows),
                    "depth": dc.district_depth(ld, rows, gap), "frontage": frontage,
                    # which words of the district brief this alternative overrules, so a
                    # revision of an inferred choice is visible as one
                    "revises": sorted(k for k in arr if k in declared
                                      and ch.get(k) != arr[k]),
                    # **is this the fabric that is standing?** The delivery round: the
                    # incumbent has to be *in* the comparison, on the same ruler, or
                    # "the best alternative" is a claim about a field of one.
                    "incumbent": bool(adopted) and all(
                        adopted.get(k) == v for k, v in arr.items() if k in adopted),
                    "refused": None, "why": why})

    front0 = str(ch.get("frontage") or "street")
    court0 = float(ch.get("courtyard_share") or 0.0)
    add("as_declared", dc.BLOCK_ROWS, base, front0, court0,
        f"the fabric as the character declares it: {base[0]}x{base[1]} lots, "
        f"{dc.BLOCK_ROWS} rows to a block")
    add("row_depth", 1, base, front0, court0,
        f"one row of houses along a street rather than {dc.BLOCK_ROWS} back to back: "
        f"the same {base[0]}x{base[1]} lots in a band "
        f"{dc.district_depth(base[1], 1, dc.LOT_GAP if front0 != 'open' else placeplan.PLOT_LANE)} "
        f"deep")
    # the widest and the narrowest bay this type admits, each at one and at two rows: a
    # wider bay is more ground a house on the same frontage, a narrower one is more
    # houses in the same length
    for side in sorted({int(hi_side), int(lo_side)}):
        lot = _lot_for(decl, side, None, attached=attached)
        word = "widest" if side >= hi_side else "narrowest"
        why = (f"the {word} bay `{name}` admits inside its {lo_side}..{hi_side} plot "
               f"band, asked at {side}")
        if lot is None or lot == base:
            add("bay_width", dc.BLOCK_ROWS, None, front0, court0, why,
                refused=(f"`{name}` lays {base[0]}x{base[1]} at every side of its band "
                         f"that stands"
                         + (" with its flanks attached" if attached else "")
                         + f": a {word} bay is not a decision this fabric offers"))
            continue
        for rows in (1, dc.BLOCK_ROWS):
            add("bay_width", rows, lot, front0, court0,
                f"{why}: {lot[0]}x{lot[1]} lots, {rows} row(s) to a block")
    # ...and the deepest lot this type admits at the bay it does lay. A row house whose
    # width is fixed by its party walls is not a fabric with one lot: the expression
    # round read "the attached row house is a fixed six-wide bay" as the end of the
    # question, and six wide by ten deep is sixty columns of ground a house against
    # forty-eight, on the same length of street. **...over the type's own depth band and
    # not over `_plot_range`'s square one.** `_plot_range` answers "what square plot
    # does this type stand on", so its top is `min(hi_w, hi_d)` -- and a type written
    # narrow to the street and deep into the plot has its whole depth thrown away by
    # that `min`. Measured the day the library's row house went from a 6x6 band to
    # 4x4..6x12: every depth this loop could ask for was capped at 10 plot columns and
    # the 16 the type had just been given was unreachable from any arrangement. The
    # depth candidates come off the declared band.
    _lo_d, _hi_d = _depth_band(decl)
    for depth in sorted({int(hi_side), int(lo_side), int(_lo_d), int(_hi_d)},
                        reverse=True):
        lot = _lot_for(decl, base[0], depth, attached=attached)
        if lot is None or lot == base or lot[0] != base[0]:
            continue
        for rows in (1, dc.BLOCK_ROWS):
            add("bay_width", rows, lot, front0, court0,
                f"the deepest lot `{name}` admits at its {base[0]}-wide bay: "
                f"{lot[0]}x{lot[1]}, {rows} row(s) to a block")
    other = "open" if front0 != "open" else "street"
    add("frontage", dc.BLOCK_ROWS, base, other, court0,
        f"{'a lane' if other == 'open' else 'a clearance'} between neighbours "
        f"instead of {'a clearance' if other == 'open' else 'a lane'}: "
        f"{other} frontage"
        + (" (the character declares the other)" if "frontage" in declared else ""))
    if dc.area_types(decls, role, (spec or {}).get("form")):
        add("compound", dc.BLOCK_ROWS, base, front0, max(0.5, court0 + 0.3),
            "the houses about a court rather than along a street: the back row of "
            "every other block is the court its front row shares")
    else:
        add("compound", dc.BLOCK_ROWS, None, front0, court0,
            "the houses about a court rather than along a street",
            refused="this role admits no area type a court could be laid as")
    # **`compact`: the same houses, closer together, on a longer block.** The
    # composition round, and the user's own reading of the built section: "the crowded
    # ring is not crowded -- its terraces stand in stripes with grass voids as wide as
    # the terraces". Every alternative above changes the *lot* -- its width, its depth,
    # how many rows of them a block holds -- and none of them changes what lies
    # **between** the blocks, which on a dense ring is where the ground went: a lane
    # across every block's end and a verge beyond the last one. A longer block is the
    # same houses with fewer streets between them, and it is the one lever that turns
    # street and verge ground into frontage without enlarging a single lot.
    # `BLOCK_LOTS_MAX` is the registered stop. It is not a cover trick and the
    # distinction is the point: the lots do not grow, the count does not move, and what
    # changes is how close the houses stand to each other and to their street.
    # `street_enclosure` is what reads the result.
    here = int(ch.get("_block_lots") or dc.BLOCK_LOTS.get(density, 3))
    for lots_per in sorted({min(dc.BLOCK_LOTS_MAX, here + 2), dc.BLOCK_LOTS_MAX}):
        if lots_per <= here:
            continue
        # a row of party walls has no gap between lots, so its block is its lots end to
        # end; anything else keeps its clearance or its lane
        gap0 = (0 if attached else
                (placeplan.PLOT_LANE if front0 == "open" else dc.LOT_GAP))
        blk = lots_per * base[0] + (lots_per - 1) * gap0
        if ch.get("block") and int(ch["block"]) >= blk:
            add("compact", dc.BLOCK_ROWS, None, front0, court0,
                f"a block of {lots_per} lots rather than {here}",
                refused=(f"the character declares a block of {ch['block']} columns, which "
                         f"already holds {lots_per} lots of {base[0]}: a longer block is "
                         f"not a decision this fabric offers"))
            continue
        add("compact", dc.BLOCK_ROWS, base, front0, court0,
            f"{lots_per} lots to a block rather than {here}: the same "
            f"{base[0]}x{base[1]} lots with {lots_per - here} fewer street-end(s) "
            f"between them, a block {blk} columns long",
            block=blk)

    # Each moves exactly one value of the character, so the best any of them can offer
    # is the best single move from a fabric the brief already called loose -- and on the
    # retained section's crowded ring the best single move (`bay_width` 6x6, one row) is
    # 82 houses at a 28.4% pad cover, still with a lane across the end of every
    # 40-column block and a verge beyond the last one. None of them has ever put a
    # building on a cross street, none of them can make a street wall out of a fabric
    # the character declares detached, and `compound`'s court is open on two sides.

    # **`terrace`: party walls, or none.** `attached` has been an `ARRANGEMENT_FIELD`
    # since the design round and no action ever wrote one, so the single most
    # consequential thing about a street -- whether the buildings touch -- was settled
    # once by the district brief's author and was never a decision the layout could
    # revise. Both directions: a detached fabric with an `ATTACHED` type in its pool can
    # become a terrace, and a terrace can become a detached street.
    terraced = dc.terrace_order([(n2, d2) for n2, d2 in houses if d2.get("attached")])
    if not attached and terraced:
        t_name, t_decl = terraced[0]
        t_lot = _lot_for(t_decl, own, ch.get("lot_depth"), attached=True, shape=shape)
        for rows in (1, dc.BLOCK_ROWS):
            add("terrace", rows, t_lot, front0, court0,
                f"a terrace: `{t_name}` declares party walls, so its lots stand against "
                f"each other with no clearance between them and the street has a "
                f"continuous front -- {t_lot[0]}x{t_lot[1]} lots, {rows} row(s) to a "
                f"block" if t_lot else "a terrace of party walls",
                refused=(None if t_lot else
                         f"`{t_name}` declares `ATTACHED` and admits no lot with its "
                         f"flanks against its neighbours' at any side of its band"),
                attach=True, house=t_name)
    elif attached:
        d_lot = _lot_for(decl, own, ch.get("lot_depth"), attached=False, shape=shape)
        add("terrace", dc.BLOCK_ROWS, d_lot, front0, court0,
            f"the same fabric detached: `{name}`'s lots a clearance apart rather than "
            f"against each other, which costs the street its continuous front and buys "
            f"each house light on both flanks"
            + (f" -- {d_lot[0]}x{d_lot[1]} lots" if d_lot else ""),
            attach=False)
    else:
        add("terrace", dc.BLOCK_ROWS, None, front0, court0,
            "party walls between the houses",
            refused="no committed type this district's role admits declares `ATTACHED`")

    # **`compact_bay`: the narrowest bay, one row, on the longest block.** Composed on
    # purpose, and named as a composition so a reader can see that it is one. A crowded
    # quarter is not one move from a loose one: the narrowest lot the fabric admits puts
    # more doors on the same length of street, one row to a block halves the depth the
    # street costs, and the longest block removes the lanes between them -- and each of
    # those is already an action here, offered alone, so the only thing new is taking
    # all three. The stop on each is the same registered stop it has alone: the type's
    # own plot band and `BLOCK_LOTS_MAX`.
    tight = _tight_lot(decl, int(lo_side), attached=attached)
    if tight is None:
        add("compact_bay", 1, None, front0, court0,
            "the narrowest bay on the longest block",
            refused=f"`{name}` admits no lot at the {lo_side} its band starts at")
    else:
        gap1 = (0 if attached else
                (placeplan.PLOT_LANE if front0 == "open" else dc.LOT_GAP))
        blk1 = dc.BLOCK_LOTS_MAX * tight[0] + (dc.BLOCK_LOTS_MAX - 1) * gap1
        for rows in (1, dc.BLOCK_ROWS):
            add("compact_bay", rows, tight, front0, court0,
                f"the narrowest bay `{name}` admits ({tight[0]}x{tight[1]}), {rows} "
                f"row(s) to a block, on a block of {dc.BLOCK_LOTS_MAX} lots "
                f"({blk1} columns): the three compaction levers together rather than one "
                f"at a time",
                block=blk1)

    # **`perimeter`: a block built round its court.** `compound` raises the courtyard
    # share, and a courtyard block in this compiler is a block whose *back row* is the
    # court -- so the court is open along both short faces of the block, the cross
    # streets have nothing standing on them, and the enclosure the arrangement's own
    # name promises is not there. A perimeter block lays lots on all four faces with the
    # court inside them, which is the ordinary urban block of most cities that have one
    # and the only arrangement here that builds a corner. It needs a deeper block than
    # two rows back to back: front row, court, back row, and the side faces' own depth
    # at each end. Three rows' depth is what that comes to, and `district_compile` is
    # still the authority on whether it fits -- an alternative that does not is refused
    # by the certificate like any other.
    if dc.area_types(decls, role, (spec or {}).get("form")):
        for lot_p in ([base] if base == tight or tight is None else [base, tight]):
            gap2 = (0 if attached else
                    (placeplan.PLOT_LANE if front0 == "open" else dc.LOT_GAP))
            # **the least block that holds one**, and the character's own where it is
            # longer. Asking for four lots' length instead made the retained middle
            # ring's 190-column sector three 61-column blocks, none of which was clear
            # of the ring wall and the arterial -- so the reservation had nowhere to
            # stand and the alternative was refused for a reason that was about the
            # block size this function chose rather than about perimeter blocks.
            blk2 = max(int(ch.get("block") or 0),
                       2 * lot_p[1] + 2 * dc.LOT_GAP + 3, 2 * lot_p[0] + gap2)
            # the character's own courtyard share stands: a perimeter block lays its own
            # court from its own four faces and does not need a share to be one
            add("perimeter", 3, lot_p, front0, court0,
                f"a perimeter block: {lot_p[0]}x{lot_p[1]} lots on all four faces with "
                f"the court inside them, on a block {blk2} columns long and three rows "
                f"deep -- building on the cross streets and a court enclosed on four "
                f"sides rather than two",
                block=blk2, perimeter=True)
    else:
        add("perimeter", 3, None, front0, court0,
            "a perimeter block round a court",
            refused="this role admits no area type a court could be laid as")
    return out


#: **The validator's clauses that a negotiation may not negotiate away.** The
#: certificate below carries the validator's verdict whole -- every failure, unmodified.
#: But an alternative is *offered for selection* against these, and the distinction
#: earns its place by a measurement: certifying on the whole verdict makes every
#: alternative of the retained hill town's dense lower ring refused, because that ring
#: genuinely covers 5% where its word asks 43% -- which is the finding the negotiation
#: exists to answer. A gate that refuses every candidate leaves the ring unlayable and
#: reports nothing, which is worse than the defect it was closing. So: `cover`,
#: `cover_over`, `ground_cover`, `count` and `farmland_cover` are the bands the
#: arrangement is *being chosen against* -- recorded, ranked, and a finding for the
#: layout owner where no alternative reaches them. Everything else is a statement that
#: the arrangement is **invalid or loses something the request required**: a plot
#: outside its own district, a plot on the arterial routed to it, a reservation the
#: demand requires, a type or footprint the library refuses. Those are refusals, and an
#: alternative carrying one is not offered. **`ground` is a refusal and not a band, and
#: it took one round to earn that.** The spatial design round adds a terrain clause to
#: `placeplan.district_failures`: a leaf most of whose columns are ground the design
#: cannot prepare is named, and so is the district that lays them. A lot with nothing
#: under it is invalid in exactly the way a lot standing on the arterial is. and while
#: that was true, **every** arrangement on the section's wet and sloped districts put
#: lots on ground that could not be prepared, so refusing them would have left the ring
#: unlayable and reported nothing, which is the failure mode the paragraph above exists
#: to avoid. The compiler reads the mask now
#: (`district_compile.LAYS_ON_FEASIBLE_GROUND`), so the clause is one an arrangement can
#: answer and a refusal is a refusal. Measured on the section's own districts at their
#: chosen levels: `lower_ring_north_2` lays 88 lots with the mask against 96 without and
#: refuses 9 for ground; `middle_ring_north_west` lays 3 against 14, and the 14 leaves
#: that stood mostly on ground nothing could prepare become **0**.
NEGOTIABLE_CHECKS = frozenset(("cover", "cover_over", "ground_cover", "count",
                               "farmland_cover"))


def certificate_for(district: dict, got: dict, place: dict, decls: dict,
                    part: dict, *, spec: dict | None = None) -> dict:
    """**The verdict of the validator that will actually judge this arrangement.**

        The composition round's second change, and A4's whole content. `capacity_of` returned
        what the compiler laid; whether that arrangement would survive
        `placeplan.district_failures` was asked later, at
        `pipeline/stages_plan.py`'s district loop, after selection -- and `arrange._covers`,
        the only test selection had, is a local approximation of one of the validator's
        clauses. So an alternative could be chosen, its ring width adopted, and then be refused
        by the plan it was chosen for.

        This is the real thing, on the alternative's own compiled result: no ground volume
        (`ground=None`), which keeps it to the clauses a plan can answer -- the district
        rectangle, the arterial, the count, the cover, the reservations, the types and the
        footprints. Cheap: a tenth of the compile it is certifying.

        Returns `{"from", "verdict", "failures", "checks", "refuses"}`. `verdict` is the
        validator's own, over every clause; `refuses` is the subset that makes the alternative
        ineligible rather than merely short of the band it is being chosen against
        (`NEGOTIABLE_CHECKS`). `"unavailable"` where the compile itself failed, which is not the
        same as a refusal and is not recorded as one.
        
    """
    from . import placeplan
    if not got.get("ok") or got.get("plan") is None:
        return {"from": "placeplan.district_failures", "verdict": "unavailable",
                "failures": [], "checks": [], "refuses": [],
                "why": str(got.get("why") or "nothing was compiled to certify")}
    try:
        fails = placeplan.district_failures(
            district, got["plan"], place, decls, ground=None,
            form=(spec or {}).get("form"), role=part.get("role"), part=part, spec=spec)
    except (ValueError, KeyError, TypeError) as e:     # noqa: PERF203 -- reported
        return {"from": "placeplan.district_failures", "verdict": "unavailable",
                "failures": [], "checks": [], "refuses": [],
                "why": f"the district validator could not judge this arrangement: {e}"}
    hard = [dict(f) for f in fails
            if str(f.get("check")) not in NEGOTIABLE_CHECKS]
    return {"from": "placeplan.district_failures",
            "verdict": "refused" if fails else "ok",
            "failures": [dict(f) for f in fails],
            "checks": sorted({str(f.get("check")) for f in fails}),
            "refuses": hard,
            "why": ("; ".join(str(f.get("why"))[:120] for f in hard[:2]) if hard else
                    None)}


def capacity_of(district: dict, part: dict, place: dict, decls: dict, arrangement: dict,
                *, spec: dict | None = None, seed: int = 1,
                ceiling: int | None = None, certify: bool = True) -> dict:
    """**What the actual compiler lays on this rectangle under this arrangement.**

        Returns the compile result with `arrangement` on it. There is no second estimator
        here and there is not going to be one: a parent negotiating its own dimension
        against a child arrangement is negotiating against what `district_compile` produced,
        or it is negotiating against a guess.

        ...and with `certificate` on it: the verdict of `placeplan.district_failures` on that
        same compiled result (A4). `certify=False` is for a caller that only wants the count
        -- `placeplan._arrange_capacity`, which is asking a width question and not offering an
        alternative for selection -- and it says so in the record it returns.
        
    """
    trial = dict(district, arrangement=dict(arrangement or {}))
    if ceiling is not None:
        trial["structures"] = max(1, int(ceiling))
    got = compile_once(trial, part, place, decls, spec=spec, seed=seed)
    got["arrangement"] = dict(arrangement or {})
    got["certificate"] = (
        certificate_for(trial, got, place, decls, part, spec=spec) if certify
        else {"from": None, "verdict": "unasked", "failures": [], "checks": [],
              "why": "this probe asked the compiler for a count and did not offer an "
                     "arrangement for selection"})
    return got


#: **What a comparison row can say about a finding's own measure.** The neighbourhood
#: delivery round, and the audit's fifth cause: "before an expensive rebuild, establish
#: that the proposed change affects the finding's subjects, survives replanning and has
#: a plausible beneficial effect". A reading names the measure it is a finding about
#: (`pipeline/inspect.MEASURES`, and the section's own `section.<side>.<measure>`); this
#: is the estimate of that measure each compiled alternative already carries, so the
#: comparison can be asked which rows could move the number the finding is about. `(key,
#: direction)`, direction being the way the measure has to go to be better. A measure
#: that is not here has no estimate at this level and the comparison says so rather than
#: guessing: the ranking then falls back to its standing priority, which is what it
#: always did.
FINDING_ESTIMATE = {
    "court_share": ("courts", "up"),
    "courts": ("courts", "up"),
    "undeveloped_share": ("undeveloped_share", "down"),
    "structures": ("lots", "up"),
    "plots": ("lots", "up"),
    "lots": ("lots", "up"),
    "built_cover": ("built_cover", "up"),
    "plot_cover": ("built_cover", "up"),
    "median_neighbour_gap": ("neighbour_gap", "down"),
    "enclosure": ("enclosure", "up"),
    "street_enclosure": ("enclosure", "up"),
    "continuity": ("continuity", "up"),
    "frontage_length": ("frontage_length", "up"),
    "median_built_footprint": ("median_footprint", "up"),
}


def finding_measure(finding: dict | None) -> tuple:
    """`(key, direction)` of the comparison estimate a finding is about, or `(None, None)`.

        The measure name is the reading's, and a section measure carries its side with it
        (`section.crowded.court_share`); only the last word names the quantity.
        
    """
    if not isinstance(finding, dict):
        return None, None
    name = str((finding.get("target") or {}).get("measure")
               or finding.get("measure") or "")
    if not name:
        return None, None
    got = FINDING_ESTIMATE.get(name.split(".")[-1])
    if not got:
        return None, None
    key, way = got
    said = str((finding.get("target") or {}).get("direction") or "").lower()
    return key, (said if said in ("up", "down") else way)


def _estimate_of(row: dict, key: str):
    """One comparison row's estimate of `key`, or None where it carries none."""
    if key == "neighbour_gap":
        # a row of party walls has no gap between neighbours; anything else has its own
        # clearance or its lane, which is what the arrangement already states
        arr = row.get("arrangement") or {}
        if not arr:
            return None
        if row.get("attached") or arr.get("attached"):
            return 0.0
        return float(LOT_GAP_ESTIMATE)
    if key == "median_footprint":
        lots = int(row.get("lots") or 0)
        return (float(row.get("pad_columns") or 0) / lots) if lots else None
    v = row.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


#: The clearance a detached fabric leaves between neighbours, as the estimate of a
#: neighbour gap. The compiler's own `LOT_GAP`; read here rather than imported at module
#: scope because `district_compile` imports this module.
LOT_GAP_ESTIMATE = 3


def alternatives(district: dict, part: dict, place: dict, decls: dict, *,
                 spec: dict | None = None, seed: int = 1, ceiling: int | None = None,
                 parts_record: dict | None = None, most: int | None = None,
                 finding: dict | None = None) -> list:
    """**A few genuinely different arrangements for one district's own rectangle, each
        certified by the district validator and measured on what it would build.**

        The composition round's answer to "keep this section and terrain, compare arrangements
        cheaply". `negotiate_ring` does this for a whole ring, where the question is the ring's
        *width*; this does it for a district whose rectangle is already fixed, which is the
        question the section asks. One compile each, no world volume, no second estimator.

        Each row carries:

          `action`, `arrangement`  what it is, and the field the layout writes to adopt it
          `lots`                   houses the compiler actually laid
          `allocated_columns`      the lots' ground
          `pad_columns`            **the estimate**: the mass those lots admit, which is the
                                   pad `site()` will hand `build()` for each plot, summed
                                   (`district_compile.pad_columns`). The two apart, always,
                                   because a figure improved only by enlarging empty lots
                                   moves the first and leaves the second (the design round's
                                   E2)
          `built_columns`          the mass that **stands**, where a `parts_record` was given
                                   and construction reported rectangles for these leaves; the
                                   pad estimate where it was not. `built_from` says which --
                                   `emitted` or `planned pads` -- and `built_estimate` is the
                                   same fact as a boolean.

                                   The neighbourhood round's instruction, and the defect it
                                   names: "Alternative 'built cover' is currently pad area,
                                   not construction. Label estimates honestly and compare them
                                   with emitted geometry after building." This function
                                   already *took* a `parts_record` and already called
                                   `region_columns` with it -- and then read the pad figure
                                   off the compile record anyway and called it built, so the
                                   emitted path it was given was computed and discarded. It is
                                   read now, and `pad_columns` is beside it so the estimate
                                   and the observation can be compared on the same row.
          `allocated_cover`,
          `pad_cover`, `built_cover`
                                   each over the district's developable ground
          `enclosure`,
          `frontage_length`,
          `mean_setback`           `placeplan.street_enclosure`: how much of the district's own
                                   street length has a building's declared front within a lot's
                                   gap of it. A street with grass verges on both sides is a
                                   village road, and no cover ratio says so.
          `reservations_kept`      of `reservations_required`
          `verdict`, `refuses`     `placeplan.district_failures` on this arrangement's own
                                   compiled result; `refuses` is the subset that makes it
                                   ineligible (`NEGOTIABLE_CHECKS`)
          `demand_short`           the required reservations it could not hold

        The disagreement is real and the numbers it turns on are kept here rather than
        settled quietly. On `lower_ring_north_2`, the retained section's crowded ring, under
        the committed `row_house` band:

            as declared  6x8, 2 rows   69 lots   pad cover 23.9%   frontage 414   enclosure 23.3%
            compact_bay  6x6, 2 rows   94 lots   pad cover 21.7%   frontage 564   enclosure 27.0%

        Built cover first keeps the declaration: the fabric the composition round shipped and
        the user read as "terraces separated by voids as wide as the terraces". Enclosure
        first takes the second, which is 36% more houses and 36% more frontage on an unmoved
        rectangle with lots that did not grow. A reader who thinks mass is the subject should
        swap these two keys back; the round's own finding `s1` names both halves ("covers
        16.8% of its ground with building **and** its houses stand in terraces separated by
        voids"), and this is a judgement about which half a *comparison* should lead with.

        `most` bounds the list; the default is the whole of `arrangements`.
        
    """
    # **imported here, and it was a latent `NameError`.** `want_front` below reads
    # `dc.character_of` and nothing in this module bound `dc` at all -- every other
    # function imports it locally. The line only runs when the part's own character
    # declares no `frontage`, which the retained city's parts all do, so the whole
    # composition round exercised this function without ever reaching it.
    from . import district_compile as dc, placeplan, spec as spec_mod
    # **the district, so the base every alternative varies is the adopted design** --
    # the audit's second cause, and the reason `arrangements` took an `allocation` it
    # never read. `as_declared` is the incumbent where one was adopted, and every other
    # row is one move from it rather than one move from the brief.
    opts = arrangements(part, decls, spec=spec, district=district,
                        pool=list(district.get("fabric_types") or []) or None)
    # **An arrangement is asked for the count its own lot earns**, not for the count the
    # incumbent fabric earned. The neighbourhood round's last seam: every option was
    # compiled at `district["structures"]`, which is a number derived from the lot the
    # district already had -- so an arrangement of smaller lots, whose whole claim is
    # that the rectangle then holds more houses, was measured at the old ceiling and
    # came back with the old count. Measured on the retained crowded ring: 6x6 asked at
    # the baseline's 69 lays 82 and asked at its own band's number lays 94. **Except
    # where the count is the sentence's own.** An `exact` district, or a spec carrying
    # an explicit count that is not an approximation, lays the number it was given and
    # nothing here moves it.
    exact = bool(district.get("exact")) or bool(
        (spec or {}).get("explicit_count")
        and not ((spec or {}).get("explicit_count") or {}).get("about"))

    def _ask_for(arrangement):
        """`(ask, band, columns_per_plot)` for one arrangement, or the caller's own
        ceiling where the count is fixed or the arithmetic cannot answer."""
        if exact:
            return ceiling, None, None
        per = band = None
        with contextlib.suppress(Exception):
            per = int(spec_mod.columns_per_plot(part, arrangement))
        with contextlib.suppress(Exception):
            band = placeplan.count_band(district, part, place, decls,
                                        arrangement=arrangement)
        if not per or not band:
            return ceiling, band, per
        got = None
        with contextlib.suppress(Exception):
            cols = int(placeplan.developable_columns(district, place, decls))
            got = int(cols * placeplan.DISTRICT_FILL // per)
        ask = min(int(got), int(band["mid"])) if got else int(band["mid"])
        return max(1, int(ask)), band, per

    out = []
    for a in opts:
        if a.get("refused"):
            out.append({"action": a["action"], "arrangement": None,
                        "refused": a["refused"], "why": a["why"], "lots": None,
                        "verdict": "unavailable"})
            continue
        ask, band, per = _ask_for(a["arrangement"])
        got = capacity_of(district, part, place, decls, a["arrangement"], spec=spec,
                          seed=seed, ceiling=ask, certify=True)
        leaves = [p for q in ((got.get("plan") or {}).get("quarters") or [])
                  for p in (q.get("plots") or [])
                  if p.get("kind", "plot") == "plot"]
        cert = got.get("certificate") or {}
        # **Every proposal is measured on its own geometry.** The spatial design round,
        # and the review's words: "`arrange.alternatives` passes the previous
        # candidate's `parts_record` into `region_columns` and `street_enclosure` for
        # hypothetical newly compiled leaves. Both match emitted measurements by part
        # name, without proving unchanged geometry. A reused leaf name can therefore
        # attach the old footprint to a new arrangement." The compiler's leaf names are
        # positional -- `b2_0_06` is block 2, row 0, lot 6 -- so a *different*
        # arrangement of the same rectangle re-uses almost every one of them for a lot
        # of a different size in a different place. Joining the previous build's emitted
        # footprints to those names does not reuse an observation; it mislabels an
        # estimate. There is no test available at this level that the relevant
        # construction inputs agree, because the parts record does not carry the plot
        # each part was laid on -- so none is reused here, and every row is the
        # compiler's own pad arithmetic, uniformly, which is what makes the ranking a
        # comparison of like with like. The observation has a place and it is after the
        # build: `improve._estimate_vs_built` writes the estimate this row was chosen on
        # beside what construction emitted over the same ground, per district the action
        # refabricated. `parts_record` is still taken -- callers pass it and a later
        # reader may want the incumbent's figures -- and it is recorded as unused rather
        # than silently ignored.
        cols = placeplan.region_columns(district, place, decls,
                                        record=got.get("record"), leaves=leaves,
                                        parts_record=None)
        over = float(cols.get("developable_columns")
                     or cols.get("scope_columns") or 0) or 1.0
        # **the street this arrangement's own grid was cut with**, not the default lane:
        # `street_enclosure`'s reach is a lane's width from a built face and an
        # arrangement that widened its streets is otherwise measured against somebody
        # else's (the coordinator's correction, neighbourhood round)
        street = placeplan.street_enclosure(
            district, place, leaves, parts_record=None,
            street_width=((got.get("record") or {}).get("street")))
        # **the estimate and the observation, apart and both named.** `pad_columns` is
        # always the plan's own pads; `built_columns` is the emitted extent where
        # `region_columns` found one on this district's leaves and the pad estimate
        # where it did not, and `built_from`/`built_estimate` say which was read.
        pad = int(got.get("footprint_columns") or 0)
        built = cols.get("built_columns")
        built = pad if built is None else int(built)
        how = str(cols.get("built_from") or "planned pads")
        # the record says outright that this is a prediction and why no observation was
        # joined to it, so a reader never has to infer it from `built_estimate` alone
        predicted_why = (
            "this row is a prediction: the compiler's pad arithmetic on leaves that do "
            "not exist yet. The previous build's emitted footprints are not joined to "
            "them -- the compiler's leaf names are positional and a different "
            "arrangement re-uses them for different lots -- so the comparison ranks "
            "every option on its own geometry and the observation is taken after the "
            "build (`improve._estimate_vs_built`)")
        out.append({
            "action": a["action"], "arrangement": dict(a["arrangement"]),
            "lot": list(a["lot"]), "rows": int(a["rows"]),
            "house": a.get("house"), "revises": a.get("revises") or [],
            "lots": int(got["lots"]),
            # **the count this row was measured at, and where it came from.** Rows may
            # be measured at different counts -- that is the point, an arrangement's
            # count is part of what it proposes -- and a row that was must say so.
            "asked": (None if ask is None else int(ask)),
            "asked_from": ("the district's own `structures`, which the sentence fixed"
                           if exact else
                           f"this arrangement's own lot: {per} column(s) a plot over the "
                           f"district's developable ground, capped at the density band's "
                           f"{band['mid']}" if per and band else
                           "the caller's ceiling; the count arithmetic could not answer"),
            "count_band": (dict(band) if band else None),
            "columns_per_plot": per,
            "allocated_columns": got.get("allocated_columns"),
            "pad_columns": pad,
            "built_columns": built,
            "built_from": how,
            "built_estimate": not how.startswith("emitted"),
            "predicted": True, "observation_reused": None,
            "predicted_why": predicted_why,
            "allocated_cover": round(float(got.get("allocated_columns") or 0) / over, 4),
            "pad_cover": round(pad / over, 4),
            "built_cover": round(built / over, 4),
            "enclosure": street.get("enclosure"),
            "frontage_length": street.get("frontage_length"),
            "mean_setback": street.get("mean_setback"),
            "street_columns": street.get("street_columns"),
            # the corrected measure's own new keys, carried through so a caller ranking
            # or reporting an alternative sees continuity and not only a ratio
            "continuity": street.get("continuity"),
            "longest_enclosed_run": street.get("longest_enclosed_run"),
            "longest_street_run": street.get("longest_street_run"),
            "unclaimed_columns": street.get("unclaimed_columns"),
            # the lot that was **laid**, beside the lot that was asked for: a row ranked
            # on a fabric it did not lay is the defect `placeplan.arrangement_failures`
            # refuses, and this is the number that refusal is about
            "lot_laid": (got.get("record") or {}).get("lot_laid"),
            "lot_asked": (got.get("record") or {}).get("lot_asked"),
            # the compiler's own share of ground nobody owns, which is what a finding
            # about leftover ground is a finding about
            "undeveloped_share": (got.get("record") or {}).get("undeveloped_share"),
            "incumbent": bool(a.get("incumbent")),
            "attached_note": (got.get("record") or {}).get("attached_note"),
            "reservations_kept": got.get("reservations_kept"),
            "reservations_required": got.get("reservations_required"),
            "demand_short": [dict(x) for x in (got.get("demand_short") or [])],
            # what this arrangement made of the quarter's programme, and its courts
            "programme": got.get("programme"),
            "courts": got.get("courts"),
            "perimeter_blocks": got.get("perimeter_blocks"),
            "perimeter_shut": got.get("perimeter_shut"),
            # **how much of this arrangement's fabric stands on ground the design cannot
            # prepare**, off the certificate's own terrain clause. A key and not a
            # footnote.
            "off_ground_leaves": sum(1 for f in (cert.get("failures") or [])
                                     if str(f.get("check")) == "ground"),
            "off_ground_share": (round(sum(1 for f in (cert.get("failures") or [])
                                           if str(f.get("check")) == "ground")
                                       / float(len(leaves)), 4) if leaves else None),
            "verdict": str(cert.get("verdict") or "unasked"),
            "refuses": [dict(f) for f in (cert.get("refuses") or [])],
            "short_of": sorted(set(cert.get("checks") or ())
                               & set(NEGOTIABLE_CHECKS)),
            "why": a["why"],
            "refused": (None if not cert.get("refuses") else
                        f"the district validator refuses this arrangement on "
                        f"{', '.join(sorted({str(f.get('check')) for f in cert['refuses']}))}"),
        })
    declared_ch = part.get("character") or {}
    want_front = str(declared_ch.get("frontage")
                     or dc.character_of(part, district).get("frontage") or "street")
    want_att = declared_ch.get("attached")

    def keeps_kind(r) -> bool:
        """**Does this alternative keep the kind of street its character declared?**

                `frontage` and `attached` are the two character values that say what a street
                *is* -- lots on a lane or buildings in their own ground, a continuous front or
                gaps between neighbours. Everything else in `ARRANGEMENT_FIELDS` is a dimension
                the layout exists to negotiate, which is why `block` or `rows` moving is not
                counted here: an alternative is demoted for contradicting the brief about the
                kind of place, not for being a different size of it.

                Earned by a measurement, with the enclosure key above it: a terrace measures more
                enclosure than anything else can, because a party wall is a continuous front. On
                the retained middle ring -- "courtyard houses again, on a tighter grain", with
                `attached: false` declared -- the terrace alternative therefore ranked **first**
                and turned the traders' quarter into 48 row houses with no shops in it. A
                declaration is still the principal's and an arrangement is still an inference
                about one rectangle, in that order; `revises` says which word was overruled
                wherever a caller chooses one anyway.
                
        """
        arr = r.get("arrangement") or {}
        if str(arr.get("frontage") or want_front) != want_front:
            return False
        return want_att is None or bool(arr.get("attached", want_att)) == bool(want_att)

    # an arrangement that lays no house is last whatever else it measures: a rectangle
    # with nothing on it has no cover, no frontage and no fabric to compare **...and the
    # ground, above the street.** The spatial design round's one addition to this order,
    # and it is above enclosure deliberately: a terrace of houses on a lake measures
    # frontage exactly as a terrace of houses on land does, and the retained section's
    # denser revision is what that looks like built -- 128 houses, 31.8% mass, decks
    # over water and a walk network with 973 unreachable stances. A street nobody can
    # stand in is not a better street. The share, not the count, so an arrangement is
    # not punished for laying more lots. **...and, above the standing priority, whether
    # this row can move the number the finding is about.** The neighbourhood delivery
    # round, and the audit's fifth cause. The order below is a stated priority over
    # measured quantities and it is about a fabric in general; a *trial* is applied for
    # one finding, and a proposal that cannot affect that finding's own measure is a
    # rebuild spent on something else. Measured on the spatial design round: three
    # actions were spent resizing the palace in answer to a question about the crowded
    # ring's courtyards. It sits **below** the obligations -- the reservations, the kind
    # of street the brief declared, and the ground -- because helping one finding at the
    # cost of a required reservation is not help; and **above** enclosure and mass,
    # because those are the comparison's general preferences and this is what the trial
    # is for. Where the reading names no measure this comparison can estimate, `helps`
    # is None on every row and the order is exactly what it was.
    want_key, want_way = finding_measure(finding)
    incumbent = next((r for r in out if r.get("incumbent")), None)
    base_est = _estimate_of(incumbent, want_key) if (incumbent and want_key) else None
    for r in out:
        r["finding_measure"] = want_key
        r["finding_estimate"] = _estimate_of(r, want_key) if want_key else None
        r["finding_incumbent_estimate"] = base_est
        if want_key is None or r.get("finding_estimate") is None or base_est is None:
            r["helps"] = None
            r["helps_why"] = (
                "this reading names no measure the comparison can estimate"
                if want_key is None else
                f"no estimate of `{want_key}` for "
                + ("this row" if r.get("finding_estimate") is None
                   else "the incumbent") + ", so whether this row helps is not known")
            continue
        est, was = float(r["finding_estimate"]), float(base_est)
        r["helps"] = (est > was) if want_way == "up" else (est < was)
        r["helps_why"] = (f"`{want_key}` {was:g} -> {est:g}, and the finding asks for it "
                          f"to go {want_way}")
    out.sort(key=lambda r: (
        int(r.get("lots") or 0) > 0 and not r.get("refused"),
        int(r.get("reservations_kept") or 0) >= int(r.get("reservations_required") or 0),
        keeps_kind(r),
        -float(r.get("off_ground_share") if r.get("off_ground_share") is not None else 0.0),
        bool(r.get("helps")),
        float(r.get("enclosure") or 0.0), float(r.get("built_cover") or 0.0),
        int(r.get("lots") or 0)), reverse=True)
    return out[:most] if most else out


def _ceiling(district: dict, part: dict) -> int:
    """The generous ask a capacity probe puts to the compiler.

        The grid estimate, which survives as a *proposal* and never as a certificate. It has
        to be generous rather than right: the compiler sizes its blocks from the ask, so
        asking it for the number a previous probe *returned* gets fewer than that number
        back, and a rung that asks for the capacity it measured never reaches it.
        
    """
    from . import placeplan, spec as spec_mod
    rect = _rect_of(district)
    w, d = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
    got = None
    with contextlib.suppress(Exception):
        got = int(placeplan.fabric_fit(w, d, part.get("density") or "medium",
                                       part.get("role"), part.get("character")))
    if not got:
        with contextlib.suppress(Exception):
            got = int(spec_mod.structures_for(w * d, part))
    return max(1, int(got or 1))


def capacity(district: dict, part: dict, place: dict, decls: dict, *,
             spec: dict | None = None, seed: int = 1, ceiling: int | None = None,
             arrangement: dict | None = None) -> int:
    """**How many houses this rectangle actually holds**, from the compiler itself.

        The count the compiler lays when it is asked for as many as the ground could give:
        `ceiling` where one is supplied, and otherwise the grid estimate, which survives
        here as a *proposal* and not as a certificate. Whatever comes back is an arrangement
        that exists, so a repair that claims more ground is claiming houses that can stand
        on it rather than houses an area divide predicted.

        This is what replaced `placeplan.fabric_fit` at every point where a number was being
        treated as capacity. `fabric_fit` is still what proposes; it no longer certifies.
        
    """
    # **The most it lays over a few asks, not the one ask.** The compiler sizes its lots
    # and its blocks *from* the ask, so a single probe measures one arrangement rather
    # than the rectangle: asked for the grid estimate this district laid 3, and asked
    # for twice it laid 4. A capacity that can come back under a number the same
    # compiler has already produced is not a capacity, so the probe is run at the
    # estimate, at the district's own ask and at twice the estimate, and what comes back
    # is the largest arrangement any of them produced.
    if arrangement:
        district = dict(district, arrangement=dict(arrangement))
    asks = {max(1, int(ceiling))} if ceiling is not None else set()
    if not asks:
        base = _ceiling(district, part)
        asks = {base, 2 * base, max(1, int(district.get("structures") or 0))}
    best = 0
    for ask in sorted(a for a in asks if a > 0):
        got = compile_once(dict(district, structures=int(ask)), part, place, decls,
                           spec=spec, seed=seed)
        if got["ok"]:
            best = max(best, int(got["lots"]))
    return best


def _grown(place: dict, d: dict, site: dict | None) -> list | None:
    """A larger rectangle for `d` inside the site, touching nothing else. None if none.

        A ring sector is **not** grown: its rectangle is a chord of an annulus and moving it
        is the ring arithmetic's decision, which is a different action from this one and is
        named unavailable rather than attempted badly.
        
    """
    from . import pipeline as _pipeline
    if not site:
        return None
    if d.get("ring") is not None or (d.get("level") is not None
                                     and (place.get("layout") or {}).get("rings")):
        return None
    ox, oz = int(site["origin"][0]), int(site["origin"][1])
    n = int(site["size"])
    bound = [ox, oz, ox + n - 1, oz + n - 1]
    got = _rect_of(d)
    was = d.get("extent_from") or list(got)
    most = int(_area([int(v) for v in was]) * (1.0 + GROWTH_MAX))
    if _area(got) >= most:
        return None
    others = []
    for key in ("districts", "compounds"):
        for r in place.get(key) or []:
            if r.get("name") != d.get("name") and r.get("x1") is not None:
                others.append(_rect_of(r))
    for p in place.get("parts") or []:
        with contextlib.suppress(Exception):
            others.append([int(v) for v in _pipeline.part_rect(p)])

    def clear(rect):
        if not (bound[0] <= rect[0] and rect[2] <= bound[2]
                and bound[1] <= rect[1] and rect[3] <= bound[3]):
            return False
        return not any(rect[0] <= o[2] and o[0] <= rect[2]
                       and rect[1] <= o[3] and o[1] <= rect[3] for o in others)

    start = list(got)
    for i, step in ((0, -1), (1, -1), (2, 1), (3, 1)):
        while True:
            nxt = list(got)
            nxt[i] += step
            if _area(nxt) > most or not clear(nxt):
                break
            got = nxt
    # **never into the water**: a shoreline district grows inland and along the shore,
    # and the shore road and setback stay between it and the water (the closure round)
    from .placeshore import clamp_to_land
    clamped = clamp_to_land(place, got)
    if clamped is None:
        return None
    got = [max(a, b) if i < 2 else min(a, b) for i, (a, b) in enumerate(zip(got, clamped))]
    got = [got[0], got[1], max(got[2], got[0]), max(got[3], got[1])]
    if not (got[0] <= start[0] and got[1] <= start[1] and got[2] >= start[2]
            and got[3] >= start[3]):
        got = [min(got[0], start[0]), min(got[1], start[1]),
               max(got[2], start[2]), max(got[3], start[3])]
    gain = _area(got) - _area(start)
    longest = max(got[2] - got[0] + 1, got[3] - got[1] + 1)
    return got if gain >= GROWTH_MIN_EDGE * longest else None


def _smaller_fabric(d: dict, decls: dict) -> list | None:
    """The approved pool re-ordered smallest admitted footprint first, or None.

        The **capability** alternative: the compiler reaches for the widest-envelope type of
        the role first, which is right by default and wrong in a thirty-column strip.

        The review found the previous version of this ineffective rather than wrong:
        `district_compile.house_types` sorted the pool by role and name, so a re-ordering
        never reached the compiler. The order is now honoured there (`preferred`), and this
        returns None where re-ordering would change nothing -- a repair that cannot move its
        consumer reports that it cannot, instead of being counted as an action tried.
        
    """
    from .district_compile import _plot_range
    pool = list(d.get("fabric_types") or [])
    if len(pool) < 2:
        return None
    sized = []
    for name in pool:
        decl = (decls or {}).get(name)
        if not decl or decl.get("kind", "plot") != "plot":
            continue
        with contextlib.suppress(Exception):
            lo, _hi, _ex = _plot_range(decl)
            sized.append((lo, name))
    sized.sort()
    if len(sized) < 2:
        return None
    order = [n for _lo, n in sized]
    return None if order == pool else order


def _covers(got: dict, floor: int) -> bool:
    """Do this arrangement's plots cover the ground its district is held to?

        A district is refused on **two** numbers -- its count and its cover -- and the ladder
        climbed for the first only. So an arrangement that laid every house it was asked for
        and covered a fifth of its rectangle went to the validator with no action taken, was
        refused there, and the revision that produced it was rolled back. Both numbers are
        what the district is held to, so both are what the ladder answers.
        
    """
    if not floor or not got.get("ok"):
        return True
    rec = got.get("record") or {}
    cols = float(rec.get("columns") or 0)
    return (float(rec.get("plot_cover") or 0.0) * cols) >= float(floor)


def arrange(district: dict, part: dict, place: dict, decls: dict, *,
            spec: dict | None = None, site: dict | None = None, seed: int = 1,
            proposed: int | None = None, cover_floor: int = 0,
            actions=ACTIONS, intent: dict | None = None) -> dict:
    """The arrangement this district is built from, after the ladder. The whole point.

        `proposed` is what the allocator asked for; the district's own `structures` where it
        is not given. The result carries the arrangement that was **adopted** -- its plan,
        its lots, the rectangle and the fabric pool it was laid with -- and `attempts`, the
        rung-by-rung record of what was tried and what each produced.

        The promise moves last and only to what the ground gave. Where it still falls short,
        `short` says so and by how much: that is a capacity finding for the scale owner and
        never a quiet reduction.
        
    """
    want = int(proposed if proposed is not None else (district.get("structures") or 0))
    # **Compiled from the proposal, never from the adopted count.** The review's second
    # finding, reproduced: an unchanged rectangle with a proposal of 20 laid 15;
    # adopting 15 and regenerating with the same proposal laid 12, because this compiled
    # `district["structures"]` -- the adopted number -- while `want` was only the bar it
    # was measured against. The compiler sizes its lots from the ask, so the ask has to
    # be the proposal every time, and the same proposal on the same rectangle with the
    # same pool is the same arrangement.
    base = dict(district, structures=int(want))
    # an exact count is the district's whatever `intent` says; the flag is the solver's
    if intent is not None and (spec or {}).get("explicit_count") \
            and not (spec or {}).get("explicit_count", {}).get("about"):
        base.setdefault("exact", True)
    first = compile_once(base, part, place, decls, spec=spec, seed=seed)
    attempts = [{"action": "as allocated", "rect": first["rect"],
                 "fabric_types": list(district.get("fabric_types") or []),
                 "lots": first["lots"], "ok": first["ok"], "why": first["why"],
                 "demand_short": [x.get("subject")
                                  for x in (first.get("demand_short") or [])
                                  if x.get("required")]}]
    best, chosen = first, dict(base)
    attempts[0]["covers"] = _covers(first, cover_floor)
    if first["ok"] and first["lots"] >= want and _covers(first, cover_floor):
        return _result(best, chosen, want, attempts, part, cover_floor)
    # **A street-composed district is its own arrangement search** (the fabric reset
    # round): `ethoslm.streetplan` tried its proposal families and judged them on their
    # relationships; the ladder below re-cuts a grid's lots and rectangle, which would
    # trade the composed streets for a count.
    if first["ok"] and (first.get("record") or {}).get("composition") is not None:
        attempts[0]["composition"] = first["record"]["composition"]["why"]
        return _result(best, chosen, want, attempts, part, cover_floor)
    for action in actions:
        if best["lots"] >= want and _covers(best, cover_floor):
            break
        trial = dict(chosen)
        if action == "fill":
            if _covers(best, cover_floor) or base.get("exact"):
                # the count is what is short and `want` already says so -- or the count
                # is the sentence's own and the cover shortfall is the layout's finding
                continue
            ask = _ceiling(trial, part)
            held = capacity(trial, part, place, decls, spec=spec, seed=seed,
                            ceiling=ask)
            if held <= int(best["lots"]):
                attempts.append({"action": "fill", "lots": best["lots"],
                                 "changed": False,
                                 "why": (f"this rectangle holds {held} house(s) at this "
                                         f"fabric and {best['lots']} are laid: there are "
                                         f"no more to ask for")})
                continue
            # ask with the ceiling that *produced* this capacity, not with the capacity
            # itself: the compiler sizes its blocks from the ask
            trial["structures"] = int(max(ask, held))
            want = max(want, int(held))
        elif action == "extent":
            bigger = _grown(place, trial, site)
            if bigger is None:
                attempts.append({"action": "extent", "lots": best["lots"],
                                 "changed": False,
                                 "why": ("no free ground beside this district inside the "
                                         "site, or its rectangle is a ring sector the "
                                         "ring arithmetic owns")})
                continue
            trial.setdefault("extent_from", _rect_of(trial))
            trial["x0"], trial["z0"], trial["x1"], trial["z1"] = [int(v) for v in bigger]
        elif action == "fabric":
            pool = _smaller_fabric(trial, decls)
            if pool is None:
                attempts.append({"action": "fabric", "lots": best["lots"],
                                 "changed": False,
                                 "why": ("the approved pool has no smaller-footprint "
                                         "alternative to re-order to")})
                continue
            trial["fabric_types"] = pool
        else:
            continue
        trial["structures"] = int(trial.get("structures") or want)
        got = compile_once(trial, part, place, decls, spec=spec, seed=seed)
        # **Did the consumer's decision actually change?** The review's test of a
        # repair, applied to the repair itself: a rung that produces the identical
        # arrangement is recorded as having changed nothing and is not counted as
        # recovery.
        moved = (got["ok"] and (got["lots"] != best["lots"]
                                or got.get("types") != best.get("types")
                                or got.get("lot") != best.get("lot")
                                or got.get("plot_cover") != best.get("plot_cover")))
        attempts.append({"action": action, "rect": got["rect"], "lots": got["lots"],
                         "fabric_types": list(trial.get("fabric_types") or []),
                         "changed": bool(moved), "ok": got["ok"],
                         "covers": _covers(got, cover_floor),
                         "demand_short": [x.get("subject")
                                          for x in (got.get("demand_short") or [])
                                          if x.get("required")],
                         "why": got["why"] if moved else
                                (f"{action} changed the input and the arrangement came "
                                 f"back the same: {got['lots']} lot(s)")})
        # **Better is more houses, or the same houses covering the ground.** A rung that
        # turns a refused district into one that meets its cover has improved it even
        # where the count did not move. **...and never at the price of a reservation the
        # request requires.** The composition round: the ladder's rungs re-cut the block
        # grid (`extent` moves the rectangle, `fabric` changes the lot), so a rung could
        # buy two houses by losing the market. A rung that drops a required reservation
        # the incumbent kept is not an improvement whatever it does to the count.
        lost = _required_short(got) - _required_short(best)
        better = got["ok"] and not lost and (
            got["lots"] > best["lots"]
            or (_covers(got, cover_floor) and not _covers(best, cover_floor))
            or (_required_short(best) - _required_short(got)))
        if better:
            best, chosen = got, trial
    return _result(best, chosen, want, attempts, part, cover_floor)


def _required_short(got: dict) -> set:
    """The subjects of the **required** reservations this arrangement could not hold."""
    return {str(x.get("subject")) for x in (got.get("demand_short") or [])
            if x.get("required")}


def why_short(got: dict, want: int) -> str | None:
    """**Why this rectangle holds fewer houses than it was asked for**, in the
        compiler's own terms. None where it holds them.

        The design round: a ring sector asked for seven laid none and became "open ground
        with an owner", three times over, and half a town's population went with it. A
        region that cannot hold its count is a finding; what it must never be is silence.
        The record has the reason in it -- no block fitted, the road took the band, the
        ceiling capped it -- and this is where it is said.
        
    """
    if not got.get("ok"):
        return str(got.get("why") or "the construction logic cannot lay this district")
    lots = int(got.get("lots") or 0)
    if lots >= int(want):
        return None
    rec = got.get("record") or {}
    grid = rec.get("grid") or {}
    lot = rec.get("lot") or [0, 0]
    rect = got.get("rect") or [0, 0, 0, 0]
    w, d = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
    why = [f"{w}x{d} was asked for {want} house(s) and holds {lots}"]
    if not int(grid.get("rows") or 0) or not int(rec.get("blocks") or 0):
        why.append(f"no row of {lot[1]}-deep lots fits across it: the grid laid "
                   f"{int(grid.get('rows') or 0)} row(s) of blocks"
                   + (", the arterial's band taking the depth between them"
                      if rec.get("axial_arterial") and any(rec["axial_arterial"])
                      else ""))
    dropped = rec.get("dropped") or {}
    if int(dropped.get("arterial") or 0):
        why.append(f"{dropped['arterial']} lot(s) dropped for the arterial's band")
    if int(dropped.get("standing") or 0):
        why.append(f"{dropped['standing']} lot(s) dropped for a standing part")
    if rec.get("capped_to") is not None:
        why.append(f"the density word's ceiling capped it at {rec['capped_to']}")
    return "; ".join(why)


def _result(best: dict, chosen: dict, want: int, attempts: list, part: dict,
            cover_floor: int = 0) -> dict:
    from . import placeplan
    least = placeplan.DISTRICT_MIN_STRUCTURES
    lots = int(best["lots"]) if best["ok"] else 0
    short = max(0, want - lots)
    said = why_short(best, want)
    return {
        **best,
        "district": chosen.get("name"),
        "part": part.get("name"),
        "proposed": int(want), "realized": lots, "short": int(short),
        "adopted_rect": _rect_of(chosen) if chosen.get("x1") is not None else None,
        "adopted_fabric": list(chosen.get("fabric_types") or []),
        "grew": chosen.get("extent_from") is not None,
        "attempts": attempts,
        # a region the ground fits fewer than a district's worth of houses in is not a
        # district; the caller turns it into open ground with an owner. **Fits**, not
        # "was asked for": a quarter of fields asked for one farm cottage and laying it
        # is not thin ground, it is a district doing what it was asked (the closure
        # round, where the thin rule cost the village its sixteenth cottage).
        "thin": bool(best["ok"] and 0 <= lots < least and want >= least),
        "least": int(least),
        # **a region short of its count says why, in the compiler's own terms.** A
        # caller that turns a thin district into open ground carries this with it, so
        # the houses that were promised to it are accounted for and not merely gone.
        "why_short": said,
        "cover_floor": int(cover_floor),
        "covers": _covers(best, cover_floor),
        # **the unmet demand, returned to the caller.** `short` is houses; this is the
        # usable space the programme required and the ground would not give, each row
        # carrying what was needed, what was available and what would make it fit. The
        # layout owner reads it; `placeplan.district_failures` refuses on it.
        "demand_short": [dict(x) for x in (best.get("demand_short") or [])],
        "reservations": [dict(x) for x in (best.get("reservations") or [])],
        "shares_spent": best.get("shares_spent"),
        "limit": (None if not short and _covers(best, cover_floor)
                  and not _required_short(best) else
                  "; ".join([x for x in [said] if x]
                            + [f"{x.get('subject')}: {x.get('why')}"
                               for x in (best.get("demand_short") or [])
                               if x.get("required")]
                            + [f"{a['action']}: {a['why']}" for a in attempts[1:]])
                  or "no supported action changed the arrangement"),
    }
