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
#: enlarging a lot (the composition round).
ARRANGEMENT_ACTIONS = ("row_depth", "bay_width", "frontage", "compound", "compact")


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


def _lot_for(decl: dict, side: int, depth: int | None, *, attached: bool) -> tuple | None:
    """`(w, d)` of a lot of about `side` this type admits, attached or not. None where
    nothing does -- which is the honest answer and not a smaller lot."""
    from . import district_compile as dc
    if attached:
        got = dc._attached_lot(decl, int(side), ("west", "east"), True)
        if got is None:
            return None
        w, ld = got
        if depth:
            d2 = int(depth)
            if all(dc._admits(decl, w, d2, att, True)
                   for att in (("west", "east"), ("west",), ("east",), ())):
                ld = d2
        return int(w), int(ld)
    w = dc._clamp_side(int(side), decl)
    ld = dc._clamp_side(int(depth or side), decl)
    if not dc._admits(decl, w, ld, None, True):
        ld = dc._deepest(decl, w, ld, (), True) or w
    return int(w), int(ld)


def arrangements(part: dict, decls: dict, *, spec: dict | None = None,
                 pool: list | None = None, allocation: dict | None = None) -> list:
    """**The child arrangements this district's fabric really admits**, in order.

        One entry per alternative: the arrangement to write on a district record, the lot it
        lays, how many rows of lots a block holds, and the **district depth** that
        arrangement needs -- which is the number a parent negotiates its own dimension
        against. Returns `[]` where the role has no house type at all.

        Nothing here is a capacity claim. `capacity` -- the actual compiler -- answers how
        many houses each one holds, on the ground the parent proposes for it.
        
    """
    from . import district_compile as dc, placeplan, spec as spec_mod
    role = part.get("role") or placeplan.DENSITY_ROLE.get(part.get("density") or "medium")
    houses = dc.house_types(decls, role, (spec or {}).get("form"),
                            approved=list(pool or part.get("fabric_types") or []) or None)
    if not houses:
        return []
    ch = dc.character_of(part)
    density = part.get("density") or "medium"
    declared = {k for k, v in (part.get("character") or {}).items() if v is not None}
    attached = bool(ch.get("attached"))
    name, decl = houses[0]
    if attached:
        pick = next(((n, d) for n, d in houses if d.get("attached")), None)
        if pick is None:
            attached = False
        else:
            name, decl = pick
    lo_side, hi_side, _ex = dc._plot_range(decl)
    own = int(ch.get("lot_width") or dc._lot_side(density))
    base = _lot_for(decl, own, ch.get("lot_depth"), attached=attached)
    if base is None:
        return []
    out, seen = [], set()

    def add(action, rows, lot, frontage, courtyard, why, refused=None, block=None):
        if refused is not None or lot is None:
            out.append({"action": action, "house": name, "arrangement": None,
                        "refused": refused or "this type admits no such lot",
                        "why": why})
            return
        w, ld = lot
        gap = placeplan.PLOT_LANE if frontage == "open" else dc.LOT_GAP
        arr = {"rows": int(rows), "lot_width": int(w), "lot_depth": int(ld),
               "frontage": frontage, "courtyard_share": round(float(courtyard), 2),
               **({"block": int(block)} if block else {})}
        key = (int(w), int(ld), int(rows), frontage, arr["courtyard_share"],
               int(block or 0))
        if key in seen:
            return
        seen.add(key)
        out.append({"action": action, "house": name, "attached": bool(attached),
                    "arrangement": arr, "lot": [int(w), int(ld)], "rows": int(rows),
                    "depth": dc.district_depth(ld, rows, gap), "frontage": frontage,
                    # which words of the district brief this alternative overrules, so a
                    # revision of an inferred choice is visible as one
                    "revises": sorted(k for k in arr if k in declared
                                      and ch.get(k) != arr[k]),
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
    # forty-eight, on the same length of street.
    for depth in sorted({int(hi_side), int(lo_side)}, reverse=True):
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
#: alternative carrying one is not offered.
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


def alternatives(district: dict, part: dict, place: dict, decls: dict, *,
                 spec: dict | None = None, seed: int = 1, ceiling: int | None = None,
                 parts_record: dict | None = None, most: int | None = None) -> list:
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
          `built_columns`          the mass those lots admit -- the two apart, always, because
                                   a figure improved only by enlarging empty lots moves the
                                   first and leaves the second (the design round's E2)
          `allocated_cover`,
          `built_cover`            each over the district's developable ground
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

        `most` bounds the list; the default is the whole of `arrangements`.
        
    """
    from . import placeplan
    opts = arrangements(part, decls, spec=spec,
                        pool=list(district.get("fabric_types") or []) or None)
    out = []
    for a in opts:
        if a.get("refused"):
            out.append({"action": a["action"], "arrangement": None,
                        "refused": a["refused"], "why": a["why"], "lots": None,
                        "verdict": "unavailable"})
            continue
        got = capacity_of(district, part, place, decls, a["arrangement"], spec=spec,
                          seed=seed, ceiling=ceiling, certify=True)
        leaves = [p for q in ((got.get("plan") or {}).get("quarters") or [])
                  for p in (q.get("plots") or [])
                  if p.get("kind", "plot") == "plot"]
        cert = got.get("certificate") or {}
        cols = placeplan.region_columns(district, place, decls,
                                        record=got.get("record"), leaves=leaves,
                                        parts_record=parts_record)
        over = float(cols.get("developable_columns")
                     or cols.get("scope_columns") or 0) or 1.0
        street = placeplan.street_enclosure(district, place, leaves,
                                            parts_record=parts_record)
        out.append({
            "action": a["action"], "arrangement": dict(a["arrangement"]),
            "lot": list(a["lot"]), "rows": int(a["rows"]),
            "house": a.get("house"), "revises": a.get("revises") or [],
            "lots": int(got["lots"]),
            "allocated_columns": got.get("allocated_columns"),
            "built_columns": got.get("footprint_columns"),
            "allocated_cover": round(float(got.get("allocated_columns") or 0) / over, 4),
            "built_cover": round(float(got.get("footprint_columns") or 0) / over, 4),
            "enclosure": street.get("enclosure"),
            "frontage_length": street.get("frontage_length"),
            "mean_setback": street.get("mean_setback"),
            "street_columns": street.get("street_columns"),
            "reservations_kept": got.get("reservations_kept"),
            "reservations_required": got.get("reservations_required"),
            "demand_short": [dict(x) for x in (got.get("demand_short") or [])],
            "verdict": str(cert.get("verdict") or "unasked"),
            "refuses": [dict(f) for f in (cert.get("refuses") or [])],
            "short_of": sorted(set(cert.get("checks") or ())
                               & set(NEGOTIABLE_CHECKS)),
            "why": a["why"],
            "refused": (None if not cert.get("refuses") else
                        f"the district validator refuses this arrangement on "
                        f"{', '.join(sorted({str(f.get('check')) for f in cert['refuses']}))}"),
        })
    want_front = str((part.get("character") or {}).get("frontage")
                     or dc.character_of(part, district).get("frontage") or "street")
    # an arrangement that lays no house is last whatever else it measures: a rectangle
    # with nothing on it has no cover, no frontage and no fabric to compare
    out.sort(key=lambda r: (
        int(r.get("lots") or 0) > 0 and not r.get("refused"),
        int(r.get("reservations_kept") or 0) >= int(r.get("reservations_required") or 0),
        str((r.get("arrangement") or {}).get("frontage") or want_front) == want_front,
        float(r.get("built_cover") or 0.0), float(r.get("enclosure") or 0.0),
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
