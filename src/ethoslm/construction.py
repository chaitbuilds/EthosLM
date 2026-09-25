"""**What construction actually delivered**, measured off the emitted geometry.

The closure round's construction boundary, and the review's second finding in its
constructed form: `cottage.build(storeys=3)` on a 9x9 lot emits one storey and returns
`ok`; the production height clause read the *planned* `params.storeys` and called the
place tall. A type reduces its storeys and drops its features internally, for reasons of
its own that are usually right, and nothing downstream could tell a house built as asked
from a house built as the pad allowed.

So every built part gets an **outcome**: what was attempted, what stands, and the
difference -- read off the blocks the builder emitted, not copied from the parameters
that were handed in. Where the type's own return says what it gave up (`emitted` on the
build result), that is recorded beside the measurement and never instead of it.

    outcome(builder, part, decl, params)   the emitted record for one built part
    surfaces(builder, part)                compact material-role / exposure context
    constraint(part, decl, emitted)        an actionable constraint where a requested
                                           feature was lost, found by probing the type
    probe_storeys(...)                     the cottage counterexample, as a measurement

`instantiate_part` (the coordinator's) calls the first three and writes them onto the
part's row in `parts.json`; the checker's height clause and the recovery ladder read
them there. Nothing here is a helper the production path does not call.
"""
from __future__ import annotations

import os

AIR = frozenset(("air", "cave_air", "void_air"))
#: Blocks that are an opening in a wall: light or a way in.
OPENING_WORDS = ("glass", "door")
#: The storey pitch `Builder.building` lays floors at, and how far up a shell is scanned
#: for them. Scanning further costs nothing and finds nothing.
STOREY_PITCH = 4
SCAN_STOREYS = 8
#: What share of the interior has to be solid for a layer to be a floor, and what share
#: has to be clear above it for the floor to be a floor and not the underside of a roof.
FLOOR_SOLID = 0.5
FLOOR_CLEAR = 0.5
#: What share of a side of the storey's wall box has to stand for the side to count, and
#: three of four sides have to.
RING_STANDS = 0.3
#: How far a probe grows a lot, per axis, looking for the pad a lost feature needs.
PROBE_GROWTH = 12

_PROBES: dict = {}


def _name(block: str) -> str:
    return str(block).split("[")[0].split(":")[-1]


def _rect(part: dict) -> tuple:
    """The ground a built part is measured over.

        A **plot** with a sited footprint is that footprint: it is what `site()` handed the
        type and it is the ground the type was answerable for. Everything else is
        `stages_plan.part_rect`'s answer, which is the one definition of "this part's
        ground" the rest of the project uses -- an **edge** is its swept polyline and a
        **point** is its pad.

        **A gate is not a rectangle and this used to raise.** The neighbourhood delivery
        round. This fell through to `part["x0"]`, which a `point` (`at`, `size`) and an
        `edge` (`path`, `width`) do not carry, so `outcome` raised `KeyError: 'x0'` and
        the caller swallowed it into a status: `out/sd-city/parts.json` has both boundary
        gates `status: "built"` with `emitted: {"measured": false, "why": "the outcome
        could not be measured: KeyError: 'x0'"}`, beside `failed: 0`. The geometry was
        never missing -- `Builder.pad_extent` has known how to turn both kinds into an
        extent since A1 -- this line just never asked for it.
        
    """
    from .pipeline import part_rect
    fp = part.get("footprint")
    if fp and len(fp) == 4 and part.get("kind", "plot") == "plot":
        x0, z0, x1, z1 = [int(v) for v in fp]
    else:
        x0, z0, x1, z1 = part_rect(part)
    return (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))


def _window(builder, part: dict, margin: int = 2) -> dict:
    """`{(x, z): {y: name}}` -- every emitted block within the part's pad plus a margin."""
    x0, z0, x1, z1 = _rect(part)
    out: dict = {}
    for (x, y, z), blk in (getattr(builder, "_pending", None) or {}).items():
        if x0 - margin <= x <= x1 + margin and z0 - margin <= z <= z1 + margin:
            out.setdefault((x, z), {})[y] = _name(blk)
    return out


def _floor_y(builder, part: dict, cols: dict) -> int | None:
    fy = part.get("floor_y")
    if isinstance(fy, int):
        return fy
    ys = [y for c in cols.values() for y in c]
    return min(ys) if ys else None


def measure(builder, part: dict) -> dict:
    """The geometry's own answer: storeys, levels, the shell, openings, the chimney.

        A **floor** is a layer solid over the shell's interior with two clear layers above
        it inside a standing wall ring; a roof is solid over solid and is never counted.
        That is what tells a one-storey cottage under a tall roof from a three-storey one,
        and it needs nothing from the type.
        
    """
    cols = _window(builder, part)
    fy = _floor_y(builder, part, cols)
    if fy is None or not cols:
        return {"storeys": None, "levels": [], "footprint": None, "openings": None,
                "chimney": None, "height": None, "columns": 0,
                "why": "nothing was emitted inside this part's pad"}
    # the shell: every column written at wall height
    shell = [c for c, ys in cols.items() if any(fy + 1 <= y <= fy + 3 for y in ys)]
    if not shell:
        return {"storeys": 0, "levels": [], "footprint": None, "openings": 0,
                "chimney": False, "height": 0, "columns": len(cols),
                "why": "nothing stands above the floor inside this part's pad"}
    rx0, rz0 = min(c[0] for c in shell), min(c[1] for c in shell)
    rx1, rz1 = max(c[0] for c in shell), max(c[1] for c in shell)

    def solid(c, y):
        return _name(cols.get(c, {}).get(y, "air")) not in AIR

    def share(cells, y, want_solid):
        if not cells:
            return 0.0
        n = sum(1 for c in cells if solid(c, y) == want_solid)
        return n / float(len(cells))
    # **Each floor is judged inside the walls that stand on it.** The ring is the
    # bounding box of the columns solid one course above the candidate floor -- the
    # walls of *that* storey -- and not the shell's whole bounding box, which takes in
    # the lean-to, the porch and the skirt and made a three-storey cottage with an
    # outshot read as one storey: its second-floor walls were a third of a ring that was
    # mostly the lean-to's roof.
    levels = []
    for y in range(fy, fy + STOREY_PITCH * SCAN_STOREYS + 1):
        # a wall stands three courses; an eave, a lean-to's roof and a bed stand one
        walls = [c for c in cols if solid(c, y + 1) and solid(c, y + 2) and solid(c, y + 3)]
        if len(walls) < 4:
            continue
        wx0, wz0 = min(c[0] for c in walls), min(c[1] for c in walls)
        wx1, wz1 = max(c[0] for c in walls), max(c[1] for c in walls)
        if wx1 - wx0 < 2 or wz1 - wz0 < 2:
            continue
        # **A floor is judged over the roofed columns of the ring.** The expression
        # round (worker B's request): a courtyard house is a ring of ranges round an
        # open yard, and a yard paved at ground level and open to the sky is not a
        # missing upper floor -- counted as one, a two-storey courtyard house measured
        # one storey. A column with nothing above the storey's headroom is the yard; a
        # plain block has every interior column roofed and measures as it did.
        inner = [(x, z) for x in range(wx0 + 1, wx1) for z in range(wz0 + 1, wz1)
                 if any(yy > y + 3 and _name(n) not in AIR
                        for yy, n in cols.get((x, z), {}).items())]
        if not inner or share(inner, y, True) < FLOOR_SOLID:
            continue
        if share(inner, y + 1, False) < FLOOR_CLEAR or \
                share(inner, y + 2, False) < FLOOR_CLEAR:
            continue
        sides = [[(x, wz0) for x in range(wx0, wx1 + 1)],
                 [(x, wz1) for x in range(wx0, wx1 + 1)],
                 [(wx0, z) for z in range(wz0, wz1 + 1)],
                 [(wx1, z) for z in range(wz0, wz1 + 1)]]
        standing = sum(1 for side in sides if share(side, y + 1, True) >= RING_STANDS)
        if standing < 3:
            continue
        if levels and y - levels[-1] < STOREY_PITCH - 1:
            continue
        levels.append(y)
    openings = sum(1 for c in cols for y, n in cols[c].items()
                   if y > fy and any(w in n for w in OPENING_WORDS))
    tops = {c: max(y for y, n in ys.items() if n not in AIR)
            for c, ys in cols.items() if any(n not in AIR for n in ys.values())}
    chimney, chimney_at, top = False, None, None
    if tops:
        # **A stack is a column standing clear of everything round it.** Not the highest
        # thing on the building: a three-storey cottage's stack is capped at the eave
        # plus `CHIMNEY_ABOVE_EAVE`, well under its ridge. A chimney column tops every
        # one of its eight neighbours by a course, allowing a 2x2 stack to share its top
        # with up to three of them, and stands above the ground storey's walls.
        stacks = []
        for (x, z), t in tops.items():
            if t < fy + 4:
                continue
            nb = [tops.get((x + dx, z + dz)) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                  if (dx or dz)]
            equal = sum(1 for v in nb if v is not None and v == t)
            higher = sum(1 for v in nb if v is not None and v > t)
            if higher == 0 and equal <= 3 and all(v is None or v <= t for v in nb) \
                    and any(v is not None and v < t for v in nb):
                stacks.append((x, z))
        if stacks and len(stacks) <= 4:
            chimney, chimney_at = True, sorted(stacks)
        top = max(t for c, t in tops.items() if c not in set(stacks or []))
    return {"storeys": len(levels), "levels": levels,
            "footprint": [rx0, rz0, rx1, rz1], "openings": openings,
            "chimney": chimney, "chimney_at": chimney_at,
            "height": (top - fy) if top is not None else None, "ridge_y": top,
            "columns": len(cols), "floor_y": fy,
            "why": (f"{len(levels)} floor level(s) at {levels} inside a "
                    f"{rx1 - rx0 + 1}x{rz1 - rz0 + 1} shell, {openings} opening(s)"
                    + (", a chimney" if chimney else ""))}


def _declared(part: dict) -> dict:
    """What the type's own return says survived, if it says anything."""
    b = part.get("build") if isinstance(part.get("build"), dict) else {}
    em = b.get("emitted") if isinstance(b.get("emitted"), dict) else {}
    return {"type_said": b, "emitted": em}


#: Features that are **open** -- a paved court, a yard, an aisle -- and are verified by
#: their openness rather than by a mass standing on them (worker B's request, the
#: expression round): paved at the floor and clear for three courses above.
OPEN_FEATURES = ("courtyard", "yard", "court", "aisle")
#: What share of an open feature's cells must be paved and clear for it to stand.
OPEN_STANDS = 0.8
#: Every feature a type may claim by rectangle; measured where it can be.
CLAIMED_FEATURES = ("outshot", "wing", "porch", "dormers", "canopy", "jetty", "oriel",
                    "courtyard", "gate", "main_hall", "screen", "stalls", "aisle",
                    "forge", "hearth", "shopfront", "counter", "altar", "dais", "benches", "bell")


def _verify_open(cols: dict, rect, fy: int) -> bool:
    """Does an open feature the type claims stand: paved at `fy`, clear above?"""
    if not rect or len(rect) != 4:
        return False
    x0, z0, x1, z1 = [int(v) for v in rect]
    cells = [(x, z) for x in range(min(x0, x1), max(x0, x1) + 1)
             for z in range(min(z0, z1), max(z0, z1) + 1)]
    if not cells:
        return False
    ok = 0
    for c in cells:
        ys = cols.get(c, {})
        paved = _name(ys.get(fy, "air")) not in AIR
        clear = all(_name(ys.get(fy + k, "air")) in AIR for k in (1, 2, 3))
        ok += bool(paved and clear)
    return ok >= OPEN_STANDS * len(cells)


def rects_of(claim) -> list:
    """A feature's claimed ground, as a **list** of rectangles.

        `[x0, z0, x1, z1]` is one rectangle; a list of those is a feature that stands in
        several places at once. The design round's defect, found on the `des-farm` market
        square: `types/square.py` published `rects.stalls` as the bounding box of its four
        corner booths, which on a 40x40 square is a 38x38 rectangle with the whole middle of
        the market inside it. `_verify_rect` asks whether half the columns of the rectangle
        carry something, four small booths give it three per cent, and the square's stalls
        read `claimed_not_found` **however well they stood**. A feature in four places is
        four rectangles and not the box round them.
        
    """
    if not claim:
        return []
    if len(claim) == 4 and all(isinstance(v, (int, float)) for v in claim):
        return [[int(v) for v in claim]]
    out = []
    for r in claim:
        if r and len(r) == 4 and all(isinstance(v, (int, float)) for v in r):
            out.append([int(v) for v in r])
    return out


def _verify_all(verify, cols: dict, claim, fy: int) -> tuple:
    """`(holds, standing, claimed)` over every rectangle a feature claims.

        Every one of them has to stand. A market whose two western booths a neighbour's
        tree-clearing pass razed is not a market that delivered its stalls, and the count is
        on the record so a reader can see it was two of four rather than none of one.
        
    """
    rs = rects_of(claim)
    if not rs:
        return (False, 0, 0)
    n = sum(1 for r in rs if verify(cols, r, fy))
    return (n == len(rs), n, len(rs))


#: How far above the floor a feature claimed by a rectangle is looked for: the storey it
#: stands in, which is `STOREY_PITCH` less the floor itself. **A rectangle is two-
#: dimensional and the column above it is not.** The design round's `des-farm` proof, on
#: `homes_south_1_1_b0_0_00`: the cottage published a 1x1 `rects.hearth` and
#: `prims.fitting` left no fire in it, and this function answered `verified` -- because
#: it asked `y >= fy + 1` with no ceiling and found the **second storey's floor**, four
#: courses up, standing in the hearth's own column. Seven of the farm's seventeen
#: cottages certified a hearth that is not there, and the disagreement only surfaced
#: when `usable` re-read the assembled world in a narrower window and `confirm` called
#: the difference `overwritten`.
FEATURE_COURSES = STOREY_PITCH - 1


def _verify_rect(cols: dict, rect, fy: int) -> bool:
    """Does an attached mass the type claims actually stand on these columns?

        Inside the storey the rectangle is drawn on (`FEATURE_COURSES`), not anywhere up
        the column. `usable._stands_in` reads the same window on the assembled world, so the
        two instruments cannot drift: a disagreement between them is then a fact about the
        world and not about their arithmetic.
        
    """
    if not rect or len(rect) != 4:
        return False
    x0, z0, x1, z1 = [int(v) for v in rect]
    cells = [(x, z) for x in range(min(x0, x1), max(x0, x1) + 1)
             for z in range(min(z0, z1), max(z0, z1) + 1)]
    if not cells:
        return False
    stood = sum(1 for c in cells
                if any(fy + 1 <= y <= fy + FEATURE_COURSES and n not in AIR
                       for y, n in cols.get(c, {}).items()))
    return stood >= 0.5 * len(cells)


def _identity_words(feature: str) -> tuple:
    """The blocks that make `feature` that feature. `usable.FEATURE_BLOCKS` is the one
    table; imported here so the emission check and the assembled-world check cannot come
    to differ about what a hearth is."""
    from . import usable
    return usable.identifies(feature)


def _identified(cols: dict, rect, fy: int, words) -> int:
    """How many identifying blocks stand in `rect`, in the window `_verify_rect` reads."""
    if not rect or len(rect) != 4:
        return 0
    x0, z0, x1, z1 = [int(v) for v in rect]
    n = 0
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for z in range(min(z0, z1), max(z0, z1) + 1):
            for y, blk in (cols.get((x, z)) or {}).items():
                if fy + 1 <= y <= fy + FEATURE_COURSES and blk not in AIR \
                        and any(w in blk for w in words):
                    n += 1
    return n


def _verify_feature(cols: dict, name: str, claim, fy: int) -> tuple:
    """`(holds, standing, claimed, source)` for one claimed feature, at emission.

        The mass first (`_verify_rect`, or `_verify_open` where the feature is open ground),
        then the **identity** where this build lays a block that carries it. The composition
        round's second evidence connection: half a rectangle of the voice's own masonry is
        not a hearth, and `source: "not_identified"` is a different fact from
        `claimed_not_found` -- the type laid something there and it is not the thing.
        
    """
    verify = _verify_open if name in OPEN_FEATURES else _verify_rect
    ok, stood_n, want_n = _verify_all(verify, cols, claim, fy)
    words = _identity_words(name)
    if not ok or not words:
        return (ok, stood_n, want_n, "verified" if ok else "claimed_not_found")
    named = sum(1 for r in rects_of(claim) if _identified(cols, r, fy, words))
    if named == want_n:
        return (True, stood_n, want_n, "verified")
    return (False, named, want_n, "not_identified")


def outcome(builder, part: dict, decl: dict | None = None,
            params: dict | None = None) -> dict:
    """The emitted outcome of one built part. Never raises; a part that cannot be
        measured says so.

        `storeys` is the measurement. `features` are measured where geometry can say
        (chimney, an attached mass the type named), verified against the type's claim
        where it made one, and `declared` where only the type says so; `features_source`
        records which. `omitted` and `fallback` are what was asked for and not delivered,
        from the attempted parameters, the type's own account and the measurement together.
        
    """
    params = dict(params or (part.get("params") or {}))
    try:
        m = measure(builder, part)
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        m = {"storeys": None, "levels": [], "footprint": None, "openings": None,
             "chimney": None, "height": None, "why": f"not measured: {e}"}
    said = _declared(part)
    em = said["emitted"]
    cols = _window(builder, part)
    fy = m.get("floor_y") if m.get("floor_y") is not None else part.get("floor_y")
    features: dict = {}
    source: dict = {}
    if m.get("chimney"):
        features["chimney"] = True
        source["chimney"] = "measured"
    else:
        # a stack capped under a gable's slope is not the highest thing near it, so the
        # type's own column is verified instead: solid for three courses above the
        # ground storey's walls, where the type says it stands
        crect = (em.get("rects") or {}).get("chimney")
        if crect and fy is not None:
            ys = cols.get((int(crect[0]), int(crect[1])), {})
            stood = sum(1 for y, n in ys.items() if y >= int(fy) + 4 and n not in AIR)
            features["chimney"] = stood >= 3
            source["chimney"] = "verified" if stood >= 3 else "claimed_not_found"
        elif m.get("chimney") is not None:
            features["chimney"] = False
            source["chimney"] = "measured"
    #: `{feature: [claimed, standing]}` where the type gave rectangles. Two numbers
    #: rather than a bool, because "two of the four booths stand" is the fact and
    #: `False` is the summary of it.
    counted: dict = {}
    for name in CLAIMED_FEATURES:
        claim = (em.get("features") or {}).get(name)
        rect = (em.get("rects") or {}).get(name)
        if rect and fy is not None:
            ok, stood_n, want_n, how = _verify_feature(cols, name, rect, int(fy))
            features[name] = bool(ok)
            source[name] = how
            counted[name] = [want_n, stood_n]
        elif claim is not None:
            features[name] = bool(claim)
            source[name] = "declared"
    requested = dict(em.get("requested") or {})
    for k, v in params.items():
        requested.setdefault(k, v)
    omitted = list(em.get("omitted") or [])
    st_req = requested.get("storeys")
    if isinstance(st_req, int) and isinstance(m.get("storeys"), int) \
            and m["storeys"] < st_req and "storeys" not in omitted:
        omitted.append("storeys")
    for name, want in (("outshot", requested.get("outshot")),):
        if want and features.get(name) is False and name not in omitted:
            omitted.append(name)
    # **A rectangle whose blocks are not the feature is a loss, not a wording.** The
    # composition round: `not_identified` means the type laid mass where it said a
    # hearth or a stall stands and the fire, the counter or the goods are not in it.
    # That is the same kind of fact as a feature that did not stand, so it reaches
    # `constraint` and the obligation ledger by the same route.
    for name, how in source.items():
        if how == "not_identified" and name not in omitted:
            omitted.append(name)
    fallback = em.get("fallback")
    if fallback is None and isinstance(m.get("storeys"), int) \
            and isinstance(st_req, int) and m["storeys"] < st_req:
        fallback = f"storeys: {st_req} asked, {m['storeys']} measured"
    # **A shell that did not stand says so here, with the type's own reason.** The shore
    # probe: two cottages on 12x11 and 10x11 lots stood nothing on real frontage and
    # their rows read `built` -- the program ran -- with the refusal only in the type's
    # return. `stood` and `type_reason` put it where the checker reads.
    stood = bool((said["type_said"] or {}).get("ok", True))
    if not stood:
        fallback = fallback or "no shell stood"
        for k in ("storeys",):
            if k not in omitted:
                omitted.append(k)
    disagree = None
    if isinstance(em.get("storeys"), int) and isinstance(m.get("storeys"), int) \
            and em["storeys"] != m["storeys"]:
        disagree = (f"the type says {em['storeys']} storey(s) and the geometry measures "
                    f"{m['storeys']}; the measurement stands")
    return {
        "storeys": m.get("storeys"), "levels": list(m.get("levels") or []),
        "features": features, "features_source": source,
        # **How each answer was got, and of which world.** The design round's third
        # contract. Every answer here is read off *this part's own pending blocks*,
        # before the next part is built, so the strongest thing it can be is `inferred`:
        # the geometry implies the feature at emission. Only `usable.check` on the
        # assembled volume produces `observed`, and `construction.confirm` writes it
        # back. A courtyard a later wall filled in was `verified` here and is `False`
        # there, and the two words are why that can now be told.
        "features_method": {f: _METHOD.get(source.get(f), "declared")
                            for f in features},
        "features_read_at": "emission",
        # the rectangles the type claimed, carried so the assembled world can be asked
        # about them again (`usable._claimed_rects`). Without these the only record of
        # where a stall was is inside the builder that has been thrown away. A feature
        # that stands in several places carries several rectangles; see `rects_of`.
        "rects": {str(k): (rs[0] if len(rs) == 1 else rs)
                  for k, r in (em.get("rects") or {}).items()
                  for rs in [rects_of(r)] if rs},
        #: `{feature: [rectangles claimed, rectangles standing]}`
        "features_rects": counted,
        "omitted": omitted,
        "fallback": fallback, "openings": m.get("openings"),
        "footprint": m.get("footprint"), "height": m.get("height"),
        "attempted": requested, "type_storeys": em.get("storeys"),
        "type_attempt": em.get("attempt"),
        "stood": stood,
        "type_reason": (said["type_said"] or {}).get("reason") or em.get("fallback"),
        "measured": True,
        "why": (m.get("why") or "") + (f"; {disagree}" if disagree else ""),
    }


#: What each `features_source` word is worth as evidence, in `usable.METHODS`. Nothing
#: measured here is ever `observed`: see `features_method` above.
_METHOD = {"measured": "inferred", "verified": "inferred",
           "claimed_not_found": "observed", "not_identified": "observed",
           "not_in_world": "observed", "declared": "declared"}


#: The wants `confirm` asks of every built part, and the features each one is about. A
#: part that claims none of a want's features is answered `unsupported` by `usable` and
#: is not thereby a failure. **The floor and not the whole set**: see `wants_for`, which
#: adds the predicates a part's *required* features make applicable. **`court_enclosed`
#: is on the floor and not in `wants_for`**, the block design round. The claim it
#: answers is written by the *compiler* onto the court leaf and travels in the plot
#: registry, not in the part's own `emitted` record -- so `wants_for`, which reads the
#: row and the demand binding, has nothing to key on.
CONFIRM_WANTS = ("entrance_connected", "equipment_reachable", "court_accessible",
                 "court_enclosed")

#: Which predicate decides each **required** feature token, and what asking it also
#: makes applicable. The composition round's first evidence connection: `confirm` asked
#: a fixed three of the six predicates of every part whatever it was required to
#: deliver, so a part required to hold a court was never asked whether its ranges
#: enclose one and a part required to hold equipment indoors was never asked whether its
#: rooms can be walked. Read off `usable.EQUIPMENT` and `usable.COURTS`, which are the
#: two lists this build's feature tokens fall into. The **required** set and not the
#: claimed set, deliberately. A feature a type chose to lay of its own accord keeps the
#: three predicates it always had; a feature a requirement made mandatory gets the
#: predicates that decide it. That is the difference between an evidence connection and
#: a new bar applied to everything at once.
WANTS_FOR_FEATURE = {
    "equipment": ("equipment_reachable", "passage_connected", "circulation_clear"),
    "court": ("court_accessible", "range_relation", "circulation_clear"),
}


def feature_kind(token: str) -> str | None:
    """`"equipment"`, `"court"`, or None where no assembled-world predicate decides it.

        None is the honest answer for `storeys` (a geometric measurement, not a predicate)
        and for `gate` or `chimney` (openings in the fabric, which `usable.EQUIPMENT`
        deliberately excludes). A caller that needs evidence for one of those reads
        `evidence_for`, which says so rather than guessing.
        
    """
    from . import usable
    t = str(token)
    if t in usable.EQUIPMENT:
        return "equipment"
    if t in usable.COURTS:
        return "court"
    return None


def wants_for(row: dict, required=()) -> tuple:
    """The predicates applicable to one built part: `CONFIRM_WANTS` plus what it owes.

        `row` is the part's row in `parts.json` (its `emitted` record carries what it claims);
        `required` are the feature tokens the **demand binding** makes mandatory of this part
        (`demand.required_by_part`). The result is in `usable.WANTS` order so a reader meets
        the predicates in the order the contract names them.
        
    """
    from . import usable
    got = set(CONFIRM_WANTS)
    em = row.get("emitted") if isinstance(row.get("emitted"), dict) else {}
    claimed = set(k for k, v in (em.get("features") or {}).items() if v) \
        | set(em.get("rects") or {})
    for t in required or ():
        kind = feature_kind(t)
        if kind:
            got |= set(WANTS_FOR_FEATURE[kind])
    # a court a part claims and was required to hold is a court whose ranges are asked
    # about; a court it merely chose to lay keeps `court_accessible` alone
    if claimed & set(usable.COURTS) & set(required or ()):
        got |= set(WANTS_FOR_FEATURE["court"])
    return tuple(w for w in usable.WANTS if w in got)


#: Why a required feature has no affirmative answer. `unmeasured` is the one that
#: matters most: nothing asked. The four the round names, plus `unreachable` -- the
#: feature stands, is the thing it claims to be, and no stance beside it is on foot --
#: because calling that one of the other four would be the same kind of lie this record
#: exists to stop.
OWED_REASONS = ("unsupported", "declared", "not_in_world", "not_identified",
                "unreachable", "unmeasured")


#: How far above a part's floor `occupied_columns` looks when the emission recorded no
#: height. Twenty-four courses: taller than anything this library builds on a plot and
#: cheap on the volume, and a part that records a height is measured to its own.
OCCUPIED_COURSES = 24


def occupied_columns(world, row: dict) -> dict | None:
    """**The columns of a part's footprint that carry its mass**, or None.

        `{"columns": int, "of": int, "from": "assembled", "why": str}`. The round's
        "separate occupied mass from court interiors and enclosing rectangles where that
        distinction matters", measured in the pass that is already reading the assembled
        world.

        What counts is a column inside the footprint carrying a block between the floor and
        the part's own measured height, on ground the plot registry does not give to somebody
        else. **None, not zero**, where the part is not standing or has no rectangle to
        measure: `emitted_columns`' contract is that an unreported part is absent from the
        dict, and a 0 would be read as a part that built nothing.
        
    """
    from . import usable
    name = str(row.get("part") or row.get("name") or "")
    if not row.get("stood", row.get("status") == "built"):
        return None
    em = row.get("emitted") if isinstance(row.get("emitted"), dict) else {}
    fp = em.get("footprint") or ([row["x0"], row["z0"], row["x1"], row["z1"]]
                                 if row.get("x0") is not None else None)
    if not fp or len(fp) != 4:
        return None
    fy = usable._floor_of(world, name, row)
    if fy is None:
        return None
    x0, z0, x1, z1 = (min(fp[0], fp[2]), min(fp[1], fp[3]),
                      max(fp[0], fp[2]), max(fp[1], fp[3]))
    vol, ctx = world.ctx.vol, world.ctx
    h = em.get("height")
    top = int(fy) + (int(h) + 1 if isinstance(h, int) and h > 0 else OCCUPIED_COURSES)
    top = min(top, vol.y0 + vol.shape[1] - 1)
    n = tot = elsewhere = 0
    for x in range(int(x0), int(x1) + 1):
        for z in range(int(z0), int(z1) + 1):
            tot += 1
            # a neighbour's mass standing inside this rectangle is the neighbour's; a
            # column the registry gives to nobody is this part's to have built on
            owner = ctx.plot_at(x, z)
            if owner is not None and str(owner) != name:
                elsewhere += 1
                continue
            if any(vol.name(x, y, z) != "air" for y in range(int(fy) + 1, top + 1)):
                n += 1
    return {"columns": int(n), "of": int(tot), "from": "assembled",
            "why": (f"{n} of the {tot} column(s) of {name}'s footprint carry its mass "
                    f"between the floor and {top - int(fy)} course(s) above it"
                    + (f"; {elsewhere} belong to another part on the registry"
                       if elsewhere else "")
                    + (f". The enclosing rectangle is {tot}" if tot != n else ""))}


def confirm(world, parts_record: dict, *, wants=None, required=None) -> dict:
    """Ask the **assembled** world what each built part actually delivers.

        The other half of `outcome`. `outcome` measures a part's own emission before the
        next part is built; this runs after all construction, through `ethoslm.usable`, and
        writes what it finds back onto each row:

            emitted.features[name]         -> False where the assembled world does not have it
            emitted.features_source[name]  -> "not_in_world"
            emitted.features_method[name]  -> "observed"
            emitted.lost[name]             -> {"was": the emission verdict, "why": ...}
            emitted.usable                 -> {want: the predicate's whole answer}
            emitted.required               -> the feature tokens this part owes, from the
                                              demand binding
            emitted.owed                   -> [{feature, holds, method, reason, why}, ...]
                                              the required features with no affirmative
                                              answer, and why each one has none
            emitted.occupied_columns       -> the columns of the footprint carrying this
                                              part's mass (`occupied_columns`), absent where
                                              it could not be measured and never 0 for one
            emitted.occupied_of            -> the columns of the enclosing rectangle, beside
                                              it, so the two are readable apart
            emitted.occupied_from          -> "assembled"
            emitted.features_read_at       -> "assembled"

        **"not_in_world" and not "overwritten".** This said `overwritten` where a feature
        stood at emission and does not stand now, which is a claim about *history* from two
        observations of *state* -- and the `des-farm` cottages proved it can be wrong: their
        hearths were never laid at all, and the emission verdict that made the difference
        look like a loss was `_verify_rect` reading the storey floor above the hearth's
        rectangle. The emission verdict is kept under `lost[name]["was"]` so a reader can
        see the disagreement; naming its cause is somebody's next measurement, not this
        function's.

        **`wants` is per part unless a caller fixes it.** `wants=None` asks `wants_for` of
        every row, which is `CONFIRM_WANTS` plus the predicates that row's `required`
        features make applicable. `required` is `{part_name: (token, ...)}` -- the per-part
        binding `demand.required_by_part` answers -- and with none the answer is exactly what
        the fixed three gave before, so no caller loses a check by this changing.

        **An unknown answer stays unresolved and is visible as owed.** The round's first
        evidence connection: `intent._usable_verdict` counted a `declared` answer as a
        predicate that ran and reported the result `observed`. Nothing here counts predicates.
        `owed` is a list of the required features that have **no affirmative answer**, each
        with the reason (`OWED_REASONS`), and a caller that wants to know whether a function
        is delivered reads that list rather than a total.

        Returns `{"parts": n, "changed": [...], "owed": [...], "why": str}` and **mutates the
        record**, so the caller writes `parts.json` back.
        
    """
    from . import usable
    rows = [r for w in (parts_record or {}).get("waves") or []
            for r in (w.get("parts") or [])]
    req_of = dict(required or {})
    changed, owed_all, n, asked = [], [], 0, set()
    for r in rows:
        em = r.get("emitted")
        if not isinstance(em, dict) or not r.get("part"):
            continue
        n += 1
        need = tuple(req_of.get(str(r["part"])) or ())
        ask = tuple(wants) if wants else wants_for(r, need)
        asked |= set(ask)
        got = {w: usable.check(world, r, w) for w in ask}
        em["usable"] = got
        em["features_read_at"] = "assembled"
        em["required"] = list(need)
        # **the mass, beside the rectangle it stands in.** See `occupied_columns`: the
        # key is written only where it could be measured, because
        # `placeplan.emitted_columns` reads its absence as "this part did not report"
        # and a 0 as "this part built nothing".
        mass = occupied_columns(world, r)
        if mass is not None:
            em["occupied_columns"] = int(mass["columns"])
            em["occupied_from"] = mass["from"]
            em["occupied_of"] = int(mass["of"])
            em["occupied_why"] = mass["why"]
        ev = {w: (a.get("evidence") or {}) for w, a in got.items()}
        lost = sorted(set(ev.get("equipment_reachable", {}).get("gone") or [])
                      | set(ev.get("court_accessible", {}).get("filled") or []))
        # a feature whose mass stands in its rectangle and whose identifying block is
        # not there: a different fact from a feature that is gone, and a different owner
        wrong = sorted(set(ev.get("equipment_reachable", {}).get("unidentified") or [])
                       | set(ev.get("court_accessible", {}).get("roofed") or []))
        for f, how, why in ([(f, "not_in_world",
                              f"the assembled world does not carry it within "
                              f"{FEATURE_COURSES} course(s) of the floor") for f in lost]
                            + [(f, "not_identified",
                                f"its rectangle is occupied and carries none of "
                                f"{list(_identity_words(f)) or 'the blocks that identify it'}")
                               for f in wrong]):
            was = (em.get("features_source") or {}).get(f)
            em.setdefault("lost", {})[f] = {
                "was": was, "why": f"emission recorded `{was}` and {why}"}
            em.setdefault("features", {})[f] = False
            em.setdefault("features_source", {})[f] = how
            em.setdefault("features_method", {})[f] = "observed"
            if f not in (em.get("omitted") or []):
                em.setdefault("omitted", []).append(f)
        if lost or wrong:
            changed.append({"part": r["part"], "lost": lost, "unidentified": wrong})
        # **what this part still owes, feature by feature.** Never a count: see above.
        mine = [a for a in (evidence_for(r, r["part"], f, answers=got) for f in need)
                if a.get("owed")]
        em["owed"] = [{k: a[k] for k in ("feature", "holds", "method", "reason", "why")}
                      for a in mine]
        owed_all.extend(mine)
        for w, a in got.items():
            if a["holds"] is False:
                changed.append({"part": r["part"], "want": w, "why": a["why"]})
    return {"parts": n, "changed": changed, "owed": owed_all,
            "why": (f"{n} built part(s) re-read on the assembled world for "
                    f"{sorted(asked)}; {len(changed)} answer(s) moved"
                    + (f"; {len(owed_all)} required feature(s) on "
                       f"{len({a['subject'] for a in owed_all})} part(s) have no "
                       f"affirmative answer" if owed_all else
                       "; every required feature named by the binding is answered"))}


def evidence_for(source, part, feature: str, *, answers: dict | None = None) -> dict:
    """**Is this required feature affirmatively demonstrated on the assembled world?**

            {"subject": part name, "feature": token,
             "holds": True | False | None,
             "method": "observed" | "inferred" | "declared" | "unsupported",
             "owed": bool,                  # no affirmative answer at observed or inferred
             "reason": one of OWED_REASONS, or None where it holds,
             "want": the predicate that decided it, or None,
             "read_at": "assembled" | "emission" | None,
             "identity": "unsupported" where this build lays no block that identifies it,
             "why": str}

        The one helper the round asks for, and `intent._function_measure`'s replacement for
        counting predicates. `source` is whatever the caller has: a `usable.World` (the
        predicate is asked live), a parts record (`{"waves": [...]}`), a list of rows, or one
        row. `answers` is `confirm`'s answer map for this part where the caller already has it.

        **`owed` is the answer, not a score.** `holds: True` with `method: "declared"` cannot
        happen -- `usable.answer` refuses to build one -- so the only affirmative answers are
        `observed` (the assembled volume) and `inferred` (a geometric measurement at
        emission, which is what `storeys` is). Anything else is owed and says why.
        
    """
    from . import usable
    world = source if isinstance(source, usable.World) else None
    row = _row_for(source, part)
    name = str((row or {}).get("part") or (row or {}).get("name") or part)
    f = str(feature)
    if row is None:
        return _owe(name, f, None, "unsupported", "unmeasured", None, None,
                    f"no row for `{name}` in this record, so nothing about `{f}` on it "
                    f"can be read")
    em = row.get("emitted") if isinstance(row.get("emitted"), dict) else {}
    kind = feature_kind(f)
    if kind is None:
        # no assembled-world predicate decides this token. `storeys` has a geometric
        # measurement at emission and is answered from it; anything else says so.
        if f == "storeys":
            got, ask = em.get("storeys"), (em.get("attempted") or {}).get("storeys")
            if isinstance(got, int) and isinstance(ask, int):
                return _owe(name, f, got >= ask, "inferred",
                            None if got >= ask else "not_in_world", None, "emission",
                            f"{name} stands {got} storey(s) measured off the emitted "
                            f"geometry against the {ask} asked for")
            return _owe(name, f, None, "unsupported", "unmeasured", None, "emission",
                        f"{name}'s emission records no measured storey count to compare "
                        f"with what was asked")
        return _owe(name, f, None, "unsupported", "unsupported", None, None,
                    f"no assembled-world predicate in this build decides `{f}`, so "
                    f"whether {name} delivers it is not established here")
    want = WANTS_FOR_FEATURE[kind][0]
    got = (answers or {}).get(want)
    if got is None and world is not None:
        got = usable.check(world, row, want)
    if got is None:
        got = ((em.get("usable") or {}) if isinstance(em.get("usable"), dict) else {}) \
            .get(want)
    if not isinstance(got, dict):
        return _owe(name, f, None, "unsupported", "unmeasured", want, None,
                    f"`{want}` has not been asked of {name} on the assembled world, so "
                    f"`{f}` is unmeasured: nothing ran, and no other predicate answers "
                    f"for it")
    ev = got.get("evidence") or {}
    rows = [x for x in (ev.get("features") or ev.get("courts") or [])
            if str(x.get("feature")) == f]
    words = _identity_words(f)
    ident = "unsupported" if not words and kind == "equipment" else None
    if f in (ev.get("gone") or []) or f in (ev.get("filled") or []):
        return _owe(name, f, False, "observed", "not_in_world", want, "assembled",
                    f"{got.get('why')}", ident)
    if f in (ev.get("unidentified") or []) or f in (ev.get("roofed") or []):
        return _owe(name, f, False, "observed", "not_identified", want, "assembled",
                    f"{got.get('why')}", ident)
    if f in (ev.get("unreachable") or []) or f in (ev.get("cut_off") or []):
        return _owe(name, f, False, "observed", "unreachable", want, "assembled",
                    f"{got.get('why')}", ident)
    # **An unknown answer is unresolved, whatever else the predicate found.** Checked
    # before the rectangles, because an `unsupported` or `declared` answer carries no
    # rectangles at all and reading its emptiness as "the feature is not there" would be
    # a verdict where the honest answer is that nothing decided.
    if got.get("method") in ("declared", "unsupported") or got.get("holds") is None:
        return _owe(name, f, None, str(got.get("method")),
                    "declared" if got.get("method") == "declared" else "unsupported",
                    want, "assembled", str(got.get("why")), ident)
    if not rows:
        # the predicate ran and this feature was not among what it looked at: the part
        # claims no rectangle for it. A declaration with no rectangle is `declared`; no
        # claim at all is a feature that is not in the world.
        claimed = (em.get("features") or {}).get(f)
        if claimed:
            return _owe(name, f, None, "declared", "declared", want, "assembled",
                        f"{name} says it emitted `{f}` and published no rectangle for "
                        f"it, so `{want}` had nothing to look for; a declaration is not "
                        f"evidence", ident)
        return _owe(name, f, False, "observed", "not_in_world", want, "assembled",
                    f"{name} is required to deliver `{f}` and its emission claims none: "
                    f"`{want}` found no rectangle for it in the assembled world", ident)
    # the predicate looked at this feature's own rectangle(s) and none of its failure
    # lists names it: this feature holds, whatever the predicate answered about the
    # others
    return _owe(name, f, True, str(got.get("method") or "observed"), None, want,
                "assembled",
                (f"`{want}` holds for `{f}` on {name} in the assembled world: "
                 + "; ".join(f"{x.get('places', 1)} place(s), "
                             f"{x.get('standing', x.get('open'))} standing, "
                             f"identity {x.get('identity')}" for x in rows[:2])), ident)


def _owe(subject, feature, holds, method, reason, want, read_at, why,
         identity=None) -> dict:
    """One `evidence_for` answer. `owed` is computed here and in one place."""
    return {"subject": str(subject), "feature": str(feature), "holds": holds,
            "method": method, "reason": reason, "want": want, "read_at": read_at,
            "identity": identity,
            "owed": not (holds is True and method in ("observed", "inferred")),
            "why": why}


def _row_for(source, part):
    """The part's row, out of a world, a parts record, a list of rows or one row."""
    from . import usable
    name = str(part if not isinstance(part, dict)
               else (part.get("part") or part.get("name")))
    if isinstance(source, usable.World):
        return source.by_name.get(name) or (part if isinstance(part, dict) else None)
    rows = None
    if isinstance(source, dict) and source.get("waves") is not None:
        rows = [r for w in (source.get("waves") or []) for r in (w.get("parts") or [])]
    elif isinstance(source, dict):
        return source if str(source.get("part") or source.get("name")) == name else None
    elif isinstance(source, (list, tuple)):
        rows = list(source)
    for r in rows or []:
        if str(r.get("part") or r.get("name")) == name:
            return r
    return part if isinstance(part, dict) else None


#: Which block families each material role is, so a block can be charged to a role.
_ROLE_SUFFIXES = ("_stairs", "_slab", "_planks", "_wall", "_fence", "_log", "_wood",
                  "_bricks", "_brick", "_tiles", "_tile", "_block")


def _family(name: str) -> str:
    n = name
    for s in ("stripped_",):
        if n.startswith(s):
            n = n[len(s):]
    for s in _ROLE_SUFFIXES:
        if n.endswith(s):
            n = n[: -len(s)]
            break
    return n


def surfaces(builder, part: dict) -> dict:
    """Compact material-role and exposure context for one built part.

        `{role: blocks}` charged through the voice the part was built in (`part["voice"]`,
        the six roles), `unknown` for a block no role names -- a fitting, a crop, glass --
        and never guessed. `exposed` counts wall-role blocks with open air on a side, which
        is what a later material pass would weather; `by_face` says which side. Ground the
        part did not lay is not in the builder's pending set and is not charged here.
        
    """
    # **Recorded at the write, where the builder has the record.** The expression round:
    # the role is what the primitive laid the block as, not a guess from its family; a
    # cobblestone floor and a cobblestone wall are told apart, the part that laid a
    # block is known, and a tread or a door is protected. The census below is the
    # fallback for a builder that predates the record.
    if getattr(builder, "_owner", None):
        from . import surfaces as _surfaces
        rec = _surfaces.record(builder, part)
        if rec.get("owned"):
            return _surfaces.histogram(rec)
    cols = _window(builder, part, margin=2)
    voice = part.get("voice") if isinstance(part.get("voice"), dict) else {}
    fam_role = {}
    for role, mat in voice.items():
        if isinstance(mat, str):
            fam_role.setdefault(_family(_name(mat)), role)
    roles: dict = {}
    exposed = 0
    by_face: dict = {"north": 0, "south": 0, "east": 0, "west": 0, "up": 0}
    total = 0
    for (x, z), ys in cols.items():
        for y, n in ys.items():
            if n in AIR:
                continue
            total += 1
            role = fam_role.get(_family(n), "unknown")
            roles[role] = roles.get(role, 0) + 1
            if role == "wall":
                for dx, dz, face in ((0, -1, "north"), (0, 1, "south"),
                                     (1, 0, "east"), (-1, 0, "west")):
                    if _name(cols.get((x + dx, z + dz), {}).get(y, "air")) in AIR:
                        by_face[face] += 1
                        exposed += 1
                if _name(ys.get(y + 1, "air")) in AIR:
                    by_face["up"] += 1
    return {"blocks": total, "roles": roles, "exposed_wall_faces": exposed,
            "by_face": by_face, "known": bool(fam_role),
            "why": ("charged through the part's voice" if fam_role else
                    "no voice on this part: every block is unknown")}


# ------------------------------------------------------------------ probes

def _flat_volume(size: int = 96, y: int = 64):
    import numpy as np
    from .observe import Volume
    y0, y1 = y - 14, y + 56
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


class _OnePlot:
    def __init__(self, plot):
        self.plots = [dict(plot)]
        self.claimed_this_pass = [dict(plot)]

    def plots_list(self):
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k):
        return True


def _type_ns(type_name: str, source: str | None = None) -> dict:
    """The executed namespace of a type file.

        `source` is a path to a file that is **not** in `types/` -- a candidate the growth
        gate is deciding about, which by definition is not committed yet. Probing one used
        to mean putting it in `types/` for the length of the probe, which is a shared
        directory a concurrent run also reads; naming the file is the same answer without
        the hazard.
        
    """
    from . import pipeline
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    decl = pipeline.load_type(source or os.path.join(root, "types", f"{type_name}.py"))
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)               # noqa: S102
    return ns


def probe_flanks(front: str, attached: int) -> list:
    """The sides a probe lot is built attached on: `attached` of the two perpendicular
    to `front`, in a fixed order so one number always names one lot.

    A row house's party walls are its flanks; its front is the street and its back is
    the rear strip, and neither is ever a party wall. `envelope.FLANK_SIDES` is the same
    table, read from the module that asks the question."""
    from . import envelope
    n = max(0, min(envelope.FLANKS_MAX, int(attached or 0)))
    return list(envelope.FLANK_SIDES.get(str(front or "north"), ("west", "east")))[:n]


def probe_build(type_name: str, w: int, d: int, params: dict | None = None, *,
                seed: int = 1, front: str = "north", size: int = 96,
                voice: str | None = None, source: str | None = None,
                attached: int = 0) -> tuple:
    """Build one type on a flat lot of `w`x`d`, no voice. Returns (builder, sited, result).

        The lot is the plan's lot; `site()` insets it as it insets every plot, so what the
        type sees is the pad the same lot gives it in a real run. `source` names a file to
        build instead of the committed `types/<type_name>.py`; see `_type_ns`.

        **`attached` is how many of the lot's flanks a neighbour stands against** -- 0 for a
        free-standing lot, 1 for a row's end, 2 for a lot in the middle of one. The
        neighbourhood delivery round: this argument did not exist and every probe was built
        free on all four sides, while `Builder._insets` drops the inset on an attached side.
        A 6x13 lot is a 4x11 pad detached and a 6x9 pad between party walls, and the second
        is what a terrace leaf is actually handed. Nothing stands next door in the probe --
        what is being measured is the pad, and the pad is decided by the plan's word, not by
        whether the neighbour has been built yet.
        
    """
    from . import offline, pipeline
    from .buildlib import Builder
    ns = _type_ns(type_name, source)
    role = ns.get("ROLE")
    vol = _flat_volume(size)
    b = Builder(offline.OfflineSite(vol))
    b._vol, b.frontage = vol, None
    x0 = z0 = (size - max(w, d)) // 2
    lot = {"label": "probe", "x0": x0, "z0": z0, "x1": x0 + int(w) - 1,
           "z1": z0 + int(d) - 1}
    b.registry = _OnePlot(dict(lot))
    mat = roof = None
    if voice:
        mat, roof = pipeline.voice_palette(voice), pipeline.voice_roof(voice)
        if roof is not None and roof.get("chimney") is None:
            from . import voices as _voices
            roof = dict(roof, chimney=_voices.chimney_default(ns.get("FORM")))
    sited = b.site({**lot, "kind": "plot", "front": front,
                    "attached": probe_flanks(front, attached)}, mat=mat, roof=roof)
    try:
        res = ns["build"](b.type_builder(sited, role=role), sited, int(seed),
                          **dict(params or {}))
    except Exception as e:                       # noqa: BLE001 -- a probe reports
        res = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    sited["build"] = res
    with_steps = getattr(b, "resolve_steps", None)
    if with_steps:
        try:
            with_steps()
        except Exception:                        # noqa: BLE001 -- the probe stands
            pass
    return b, sited, res


def probe_storeys(type_name: str, storeys: int, small_lot, control_lot, *,
                  seed: int = 1, params: dict | None = None,
                  voice: str | None = None) -> dict:
    """The counterexample as a measurement: the same request on two lots.

        `{"small": {"planned", "emitted", "lot"}, "control": {...}}`, each `emitted` read off
        the geometry by `outcome` and never off the parameters.
        
    """
    out = {}
    for key, lot in (("small", small_lot), ("control", control_lot)):
        p = dict(params or {}, storeys=int(storeys))
        b, sited, res = probe_build(type_name, int(lot[0]), int(lot[1]), p, seed=seed,
                                    voice=voice)
        got = outcome(b, sited, None, p)
        out[key] = {"planned": int(storeys), "emitted": got.get("storeys"),
                    "lot": [int(lot[0]), int(lot[1])], "ok": bool(res.get("ok")),
                    "type_said": got.get("type_storeys"), "why": got.get("why")}
    return out


def constraint(part: dict, decl: dict | None, emitted: dict | None, *,
               params: dict | None = None, seed: int | None = None,
               voice: str | None = None) -> dict | None:
    """An actionable constraint where a requested feature was lost, or None.

        For a lost storey the type is **probed** -- the same type, parameters, seed and
        door side on flat ground, the lot grown one column at a time up to `PROBE_GROWTH`
        on each axis -- and the smallest lot on which the request survives is the
        constraint: `needs.lot_min`. That derives the envelope from the type's own rules by
        executing them, which is the only statement of those rules that cannot drift from
        the code. The owner is `layout` (the lot is the layout's decision) where a lot
        within reach delivers it, and `build` (the type cannot) where none does.
        
    """
    em = emitted or {}
    params = dict(params or em.get("attempted") or part.get("params") or {})
    omitted = list(em.get("omitted") or [])
    if not omitted:
        return None
    tname = str(part.get("type") or "")
    # The **plot** as the plan drew it, which is what a lot constraint is about and is
    # unchanged. A part with no rectangle of its own -- a gate (`point`), a wall
    # (`edge`) -- falls back to the one definition of its ground rather than raising
    # `KeyError: 'x0'` at a caller that turns the exception into a `built` status.
    if part.get("x0") is not None:
        x0, z0, x1, z1 = int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])
    else:
        x0, z0, x1, z1 = _rect(part)
    w, d = abs(x1 - x0) + 1, abs(z1 - z0) + 1
    seed = int(seed if seed is not None else part.get("seed", 0) or 0)
    if "storeys" in omitted and isinstance(params.get("storeys"), int) and tname:
        want = int(params["storeys"])
        got = em.get("storeys")
        key = (tname, w, d, want, seed, voice, tuple(sorted(params.items())))
        found = _PROBES.get(key)
        if found is None:
            found = _find_lot(tname, w, d, params, want, seed, voice)
            _PROBES[key] = found
        if found and tuple(found) == (w, d):
            # the same lot delivers it on flat ground: what differs here is the ground
            # under the pad or the lane beside it, which set the pad the type was given
            return {"what": "storeys", "requested": want, "emitted": got,
                    "part": part.get("name"), "type": tname,
                    "needs": {"lot": [w, d], "lot_min": [w, d],
                              "cause": "ground_or_frontage"},
                    "owner": "layout",
                    "why": (f"{tname} asked for {want} storey(s) on a {w}x{d} lot and "
                            f"emitted {got}; probed on flat ground at seed {seed} the "
                            f"same lot delivers {want}, so the lot is not what is "
                            f"short: the ground under this pad or the lane beside it "
                            f"took the room the type needed")}
        if found:
            return {"what": "storeys", "requested": want, "emitted": got,
                    "part": part.get("name"), "type": tname,
                    "needs": {"lot": [w, d], "lot_min": list(found)},
                    "owner": "layout",
                    "why": (f"{tname} asked for {want} storey(s) on a {w}x{d} lot and "
                            f"emitted {got}; probed on flat ground at seed {seed} it "
                            f"delivers {want} on a lot of {found[0]}x{found[1]}, so the "
                            f"lot is what is short")}
        return {"what": "storeys", "requested": want, "emitted": got,
                "part": part.get("name"), "type": tname,
                "needs": {"lot": [w, d], "lot_min": None},
                "owner": "build",
                "why": (f"{tname} asked for {want} storey(s) emitted {got}, and no lot "
                        f"up to {w + PROBE_GROWTH}x{d + PROBE_GROWTH} delivers them at "
                        f"seed {seed}: the type, not the lot, is what is short")}
    return {"what": omitted[0], "requested": params.get(omitted[0], True),
            "emitted": em.get("features", {}).get(omitted[0]),
            "part": part.get("name"), "type": tname,
            "needs": {"lot": [w, d], "lot_min": None}, "owner": "build",
            "why": (f"{tname} was asked for `{omitted[0]}` and did not deliver it "
                    f"({em.get('fallback') or 'no reason given'})")}


def _find_lot(tname: str, w: int, d: int, params: dict, want: int, seed: int,
              voice: str | None = None):
    """The smallest lot (by area) within `PROBE_GROWTH` on which `want` storeys stand."""
    best = None
    tried = 0
    for k in range(0, PROBE_GROWTH + 1):
        cands = sorted({(w + k, d + j) for j in range(0, k + 1)}
                       | {(w + j, d + k) for j in range(0, k + 1)},
                       key=lambda c: (c[0] * c[1], c))
        for (cw, cd) in cands:
            if best and cw * cd >= best[0] * best[1]:
                continue
            tried += 1
            try:
                b, sited, res = probe_build(tname, cw, cd, params, seed=seed, voice=voice)
            except Exception:                    # noqa: BLE001 -- a probe reports
                continue
            if not res.get("ok"):
                continue
            got = outcome(b, sited, None, params)
            if isinstance(got.get("storeys"), int) and got["storeys"] >= want:
                best = (cw, cd)
        if best:
            return best
    return best
