"""A town house on a city street: narrow to the lane, deep into the plot.

The two long flanks are party walls -- flat, unbroken, no eave and no opening --
because the next house stands against them. The ridge runs front to back, which
is the only way it can run on a frontage of five or six columns and a depth of
twenty: the roof slopes fall toward the two flanks and the hipped ends land over
the street and over the rear veranda. So a flank is a **verge**, not an eave: on
a side the plan says is attached the roof is closed with masonry from the wall
head to the ridge and nothing oversails the plot line -- see the verge in
`build()` for the five rooms-with-no-way-in that measured why. Inside: a front
room on the lane, a passage beside it running the depth of the house, rooms
behind it, and the stair in the passage where the passage runs out. The engawa is
at the back, under the eave, facing the strip of ground the street side does not
have.
"""

import random

KIND = "plot"
FORM = "east_asian"
ROLE = "urban"

#: **What this type is for.** The realization round: a `ROLE` says what work a building
#: is for and is satisfied by a hall, a barn or a temple alike; a sentence asking for
#: houses people live in is asking for a `dwelling`. Declared so that the function can
#: be checked rather than inferred from a label.
FUNCTION = "dwelling"

#: v2, C2: the two long flanks are party walls, so this type may stand **attached** --
#: the next house against it, the pad reaching the plot's edge on that side, the way in
#: on the street. The plan says which sides (`part["attached"]`); the type builds its
#: flanks blank whatever the plan says, as it always has.
ATTACHED = True

PARAMS = {
    "storeys": ("int", 1, 3),
    "front": ("choice", ["lattice", "screen", "open"]),
}

NEEDS = {
    # **The depth is taken, and the whole of the measurement is declared.** The band
    # read 4x4 to 6x6 against the 6x6 to 16x24 its author declared, and open thread 15
    # recorded the disagreement as unresolved: "the two instruments disagree by a factor
    # of four on the same file". The neighbourhood round re-ran the sweep's own
    # instrument (`type_needs.instance`, nine parameter sets x two seeds x flat and bank
    # = 36 instances a size) over the **square** sizes 4, 6, 8, 9, 10, 11, 12, 13, 16,
    # 20, 24 and 32 and found 0 of 36 failing at every one of them. Two things had
    # moved. The `E004` down-slope readings at 7, 8 and 9 that closed the old band were
    # the roof-stair carve-outs, fixed two rounds ago and never re-swept; the
    # `E003`/`E011` pairs at 10 and 11 were this file's own storey-line lip laid under
    # the main eave, fixed above. A square sweep cannot widen a band whose two axes are
    # different numbers, and this type's whole sentence is that they are. The spatial-
    # design round swept the **rectangle**: short sides 4, 5 and 6 against long sides 7
    # to 32 -- 38 sizes, 36 instances each, **1368 instances, 0 failures**, on the plane
    # and on the bank: 4x8 4x9 4x10 4x11 4x12 4x13 4x16 4x20 4x24 4x32 0 of 36 fail at
    # each 5x8 5x9 5x10 5x11 5x12 5x13 5x14 5x16 5x20 5x24 5x32 6x7 6x8 6x9 6x10 6x11
    # 6x12 6x13 6x14 6x16 6x18 6x20 6x24 6x28 6x32 4x4 5x5 6x6 (the old band's own
    # sizes, re-stood) So the long side is clean to the end of what was swept and the
    # declaration is that end, which is this file following the rule `type_needs` states
    # for every type: the band is the measured envelope, and where the envelope reaches
    # the end of the sweep the ceiling is a bound on the measurement rather than on the
    # type. **Why the short side stays at 6 and the long side moves to 32.** Two
    # reasons, and the second is not a reason and says so. The first is architectural
    # and is this type's own. `needs_footprint_failure` sorts a pad before it compares,
    # so `hi_w` is the ceiling on the *short* side and `hi_d` on the long one. it is a
    # hall with two party walls. A declaration is allowed to ask for less ground than
    # the sweep stood it on, and `test_types` holds it to exactly that: the measurement
    # is recorded in the bank, the frontage this house is for is declared here, and the
    # two do not disagree. The second is that `placeplan.dense_plot` reads only the
    # **square** sizes a type admits, takes the urban type whose square ceiling is
    # lowest, and makes that the lot every dense district in the library is built of --
    # so at `hi_w` of 32 the dense lot would be a 36-column square. Measured through the
    # real path after this edit: `dense_plot` 6/10/100 unchanged, and `density_lot`
    # 20/15/12/10 a side for sparse/low/medium/dense unchanged. Widening the **depth**
    # does not move any of them, because nothing on that path reads the depth. **What
    # the stale band cost**, on the retained section: ninety-six row houses on 5x6 and
    # 5x7 pads, every one of them one storey and six blocks high, half of them having
    # asked for two. `top` needs eight columns of house depth to carry a longitudinal
    # flight and the band admitted six. A ring with no variation in height was a ring
    # whose houses were never offered a lot they could vary on. Pad 5x6, 6x6 asked 1/2/3
    # -> stood 1,1,1 one height, 6 blocks pad 5x9 asked 1/2/3 -> stood 1,2,2 heights 5,
    # 7, 9, 11 pad 6x11 asked 1/2/3 -> stood 1,2,2|3 heights 7, 11, 13, 15 pad 6x12
    # asked 1/2/3 -> stood 1,2,2|3 heights 6,7,11,13,14,15,17 pad 6x16, 7x11 asked 1/2/3
    # -> stood 1,2,3 three storeys at every seed The ceiling used to be held down by a
    # coupling that is now cut. `_attached_lot` ranked candidate lots by nearness to the
    # density's own **square side**, so a wider band moved the dense fabric lot from 6x8
    # to 9x10 and then to 10x10 and the ground a dense house cost from 102 columns to
    # 166 and 182. It ranks by area and by the **shape the density asks for** now
    # (`placeplan.density_lot` carries a width and a depth), so this type's own sentence
    # -- narrow to the lane, deep into the plot -- is a thing the layout can ask for and
    # this band is a thing it can answer with.
    "footprint": (4, 4, 6, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

#: **What this house can do with a piece of ground: the least pad each storey count
#: stands on.** In pad columns (across the frontage, along the depth), which is the unit
#: `NEEDS["footprint"]` is in and `Builder.pad_extent` converts a plot to. The spatial-
#: design round. `NEEDS` says what this type may stand on; it does not say what standing
#: there *gets you*, and the layout was inferring that from the band's ceiling -- a
#: capability limit standing in for a demand. A density word asks for an amount of
#: ground and a shape; this is the answer to the other half of the question, said by the
#: house that has to stand it. `district_compile.storey_pad` reads it; a type that
#: declares nothing is unaffected and nothing here is required of any other file.
#: Measured by `construction.probe_build` at three seeds x three fronts, and a row is
#: the least pad at which **every** instance reached that many storeys: 1 storey 4x4 the
#: band's own floor 2 storeys 5x9 `top` needs `house_d >= 8` and `w >= 5`; `_sizes`
#: takes one row of the pad's depth for the rear strip once the pad is 8 deep, so nine
#: columns of pad is eight columns of house 3 storeys 6x16 `top` needs `house_d >= 10`
#: and `w >= 6`; at 6x11 and 6x12 the third storey stands at some seeds and not others,
#: because `_sizes` drops the frontage to 5 about half the time. That is the variation,
#: not a failure: a terrace of 6x12 lots stands at seven distinct heights in one
#: language. 6x16 is where three is certain at every seed **and** the pad is still
#: within this type's 6-column short side; 7x11 also gives three and is not offered,
#: because a 7-column frontage is not this house. **Nothing declares 4 across.** The
#: band's floor is 4 and the sweep stands clean there, but the sweep stands its
#: instances on an *inset plot*: at a 4-wide pad this type laid a five-wide house in **9
#: of 9** instances and the extra column landed in the plot's inset, which a party wall
#: does not have. `_sizes` no longer lets the house leave its pad (see there), and a lot
#: this type is asked to stand attached on is 5 across at the narrowest whatever the
#: band admits.
STOREY_PAD = {1: (4, 4), 2: (5, 9), 3: (6, 16)}

_NAME = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
_OPP = {"east": "west", "west": "east", "north": "south", "south": "north"}

# the street elevation of the ground floor, by parameter: (bay width, gap between bays,
# sill above the floor, head above the floor)
_FRONT = {
    "lattice": (1, 1, 1, 3),
    "screen": (3, 2, 2, 3),
    "open": (4, 1, 1, 3),
}

# fittings that may take more than the cell they are asked for
_WIDE = ("hearth", "store", "bed", "oven", "well", "table", "forge", "trough")

#: How far above a storey-line lip the sky has to be clear before it is laid: to the
#: ridge and two courses past it. Not a fixed few courses -- measured at pad 11x11 with
#: three storeys, where the main eave stands eight above the lip and the pocket between
#: them was as sealed as the two-storey one. If anything of this house stands over that
#: ground, the ground does not get a floor. See the lip.
LIP_HEADROOM = 2


def _sizes(rng, pad_d, pad_w, att_lo=False, att_hi=False):
    """Rear strip, depth of the house, width to the street.

        `att_lo`/`att_hi` say whether a party wall stands on the pad's low and high lateral
        edge. They are not decoration: see the width below.
        
    """
    if pad_d >= 12:
        rear = rng.choice([1, 2])
    elif pad_d >= 8:
        rear = 1
    else:
        rear = 0
    n_att = int(bool(att_lo)) + int(bool(att_hi))
    # A side veranda needs a free flank to stand beside. Between two party walls there
    # is no side to put one on, and taking a column for it there is what leaves a slot.
    side = 0 if (rear or n_att >= 2) else 1
    house_d = pad_d - rear
    if house_d >= 15:
        house_d -= rng.choice([0, 1, 2])
    avail = pad_w - side
    if n_att >= 2:
        # **Between two party walls the house is exactly as wide as its pad.** The
        # neighbourhood delivery round, measured on `out/nd-block`. v2 C2 drops the
        # pad's inset on an attached side *so that* the terrace is continuous; this line
        # then chose a width anyway, and a 6-wide pad got a 5-wide house in about half
        # the seeds. The spare column is a slot between two party walls, and both
        # neighbours' eaves roof over it: five `E003`/`E011` pairs, 10 to 15 cells each,
        # at the flank of every one-storey house in the block. Width is not a lever on a
        # lot with two party walls -- depth, the rear strip, the roof and the storeys
        # are, and the pad's own width still varies lot to lot. The plan cannot hand
        # this branch a pad wider than this type is written for:
        # `stages_plan.needs_footprint_failure` compares `Builder.pad_extent` against
        # `NEEDS["footprint"]`, whose short side is 6, in pad columns.
        return rear, house_d, max(1, pad_w)
    w = min(avail, 9, max(6, house_d - 1))
    if w >= 6 and rng.random() < 0.45:
        w -= 1
    # **The house does not leave its pad.** The spatial-design round, measured through
    # `construction.probe_build`: this line read `max(5, min(w, avail))`, so on a pad
    # with four columns to spare it returned five and the frontage ran one column past
    # the plot's edge in **9 of 9** instances at a 4x9 pad. The sweep never saw it
    # because `type_needs` stands its instances on a plot with `site()`'s inset round
    # the pad, and a column of overhang lands in the inset. A **party wall has no
    # inset**: the column it would land in is the next house. Five is still the floor
    # this house wants -- below it the passage and the rooms beside it do not both fit
    # -- but where the pad cannot give five it gets what there is, and the `attempts`
    # ladder below narrows again if the shell refuses.
    return rear, house_d, max(1, min(max(5, min(w, avail)), avail))


#: Which edge of the pad each compass front names: the axis the **depth** runs on, and
#: the sign it runs in. `front: "north"` means the street is north of the lot, so the
#: front wall is the pad's low-z row and the house runs south into the plot.
_FRONT_EDGE = {"north": ("z", 1), "south": ("z", -1),
               "west": ("x", 1), "east": ("x", -1)}

#: The two sides that are party walls, by the axis the depth runs on, **lower lateral
#: coordinate first**. The flanks are always the pair parallel to the depth: the front
#: is the street and the back is the rear strip, and neither is ever a party wall.
_FLANK_SIDES = {"z": ("west", "east"), "x": ("north", "south")}

#: How far the reserved doorstep may sit from the declared front's edge and still be
#: called a door in it. One, because `site()` reserves the doorstep on the pad's edge
#: row and `approach()` may take the step one column in from it.
FRONT_DOOR_TOL = 1


def _front_edge(part):
    """Which edge of the pad is the front: **the plan's**, where the doorstep is in it.

        Returns `(axis, sign, source)`.

        Two questions with the same answer most of the time, and until the neighbourhood
        delivery round only the second was asked. `part["front"]` is the compass side the
        street is on -- the plan's own decision, now carried through `PART_GEOMETRY` into
        `site()`; the reserved doorstep is where the circulation pass levelled a way in.
        They agree whenever the plan was carried through, which is what `test_compile`
        case 7 asserts of every leaf in a row. They come apart at a corner: the doorstep
        of a wide shallow lot sitting on the corner of its front wall is exactly as near
        the flank, and the tie-break below then reads the flank as the front and turns the
        house ninety degrees.

        **Where they disagree the doorstep still wins**, and the disagreement is reported
        (`emitted.front_from`) rather than resolved silently: a front wall with no door in
        it is a house nobody can walk into, which is a worse fault than a house facing the
        wrong way. That fallback is not hypothetical -- `Builder.site()` reserves a probe's
        doorstep on the south edge whatever `front` it is handed, so every
        `construction.probe_build(front=...)` is a part whose two answers differ. That is a
        gap in siting rather than in the plan, and it is why this keeps both.
        
    """
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    said = _FRONT_EDGE.get(str(part.get("front") or "").strip().lower())
    if said is not None:
        ax, s = said
        near = {("z", 1): abs(dz - pz0), ("z", -1): abs(dz - pz1),
                ("x", 1): abs(dx - px0), ("x", -1): abs(dx - px1)}[(ax, s)]
        if near <= FRONT_DOOR_TOL:
            return ax, s, "plan"
    runs = {"x": px1 - px0 + 1, "z": pz1 - pz0 + 1}
    cand = [("x", 1, abs(dx - px0)), ("x", -1, abs(dx - px1)),
            ("z", 1, abs(dz - pz0)), ("z", -1, abs(dz - pz1))]
    cand.sort(key=lambda c: (c[2], -runs[c[0]]))
    return cand[0][0], cand[0][1], ("door" if said is None else "door_over_plan")


def _partitions(rng, house_d):
    """Where the cross walls fall, in depth from the front wall.

        Every room they leave is at least two rows deep, so that a chest against
        one wall can never cut the room it stands in in half.
        
    """
    back = house_d - 2
    lo, hi = 3, back - 2
    if hi < lo:
        return []
    span = hi - lo
    if span >= 6:
        n = rng.choice([2, 3])
    elif span >= 3:
        n = rng.choice([1, 2])
    else:
        n = 1
    while n > 0:
        step = (back + 1) / float(n + 1)
        cand = []
        for k in range(n):
            dp = int((k + 1) * step + 0.5) + rng.choice([0, 0, 1, -1])
            cand.append(max(lo, min(hi, dp)))
        cand = sorted(set(cand))
        if len(cand) == n and all(cand[i + 1] - cand[i] >= 3
                                  for i in range(n - 1)):
            return cand
        n -= 1
    return []


def build(b, part, seed, storeys=None, front=None, **kw):
    rng = random.Random(seed * 7919 + 104729)
    voice = part["voice"]
    fy = part["floor_y"]
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    label = part["label"]

    ax, s, front_from = _front_edge(part)
    # **Which flanks a neighbour stands against.** The plan says (`part["attached"]`),
    # `Builder._insets` has already dropped the pad's inset on those sides, and from
    # here the type builds to the party wall instead of pretending the side is open.
    att = {str(v).strip().lower() for v in (part.get("attached") or ())}
    lo_side, hi_side = _FLANK_SIDES[ax]
    att_lo, att_hi = lo_side in att, hi_side in att
    if ax == "x":
        pad_d, pad_w = px1 - px0 + 1, pz1 - pz0 + 1
        front_c = px0 if s > 0 else px1
        latlo, lathi, dlat = pz0, pz1, dz
        inward = _NAME[(s, 0)]
    else:
        pad_d, pad_w = pz1 - pz0 + 1, px1 - px0 + 1
        front_c = pz0 if s > 0 else pz1
        latlo, lathi, dlat = px0, px1, dx
        inward = _NAME[(0, s)]
    outward = _OPP[inward]

    def cell(dd, tt):
        c = front_c + s * dd
        return (c, tt) if ax == "x" else (tt, c)

    def uncell(X, Z):
        return ((X - front_c) * s, Z) if ax == "x" else ((Z - front_c) * s, X)

    def lat_name(sign):
        return _NAME[(0, sign)] if ax == "x" else _NAME[(sign, 0)]

    def rect(hd, ww, aa):
        p = cell(0, aa)
        q = cell(hd - 1, aa + ww - 1)
        return (min(p[0], q[0]), min(p[1], q[1]),
                max(p[0], q[0]), max(p[1], q[1]))

    rear, house_d, w = _sizes(rng, pad_d, pad_w, att_lo, att_hi)

    def pin(ww):
        """Where the frontage of a `ww`-wide house starts when a party wall decides it.

        **A party wall is on the plot's edge**, so the house is against it: the pad's
        inset was dropped on that side for exactly this. Where both flanks are attached
        `_sizes` has already made `ww` the whole pad and this is the only position.
        None where both flanks are free and the doorstep decides instead."""
        if att_lo:
            return latlo
        if att_hi:
            return lathi - ww + 1
        return None

    def held(ww):
        """`(width, start)` for a frontage a party wall decides, **widened into a free
                flank's slack rather than standing the house's corner on the doorstep**.

                The neighbourhood delivery round's section build, and the one house in
                eighty-three whose door the check could not walk into:

                    E008 the doorway reserved for lower_ring_north_2_b0_0_40 at
                         (-5828,67,621) cannot be walked into off its own threshold

                It is a row's **end**. Its 6x9 lot insets to a 5x7 pad (`_insets` drops the east
                side and takes one off the rest); `_sizes` gives the side veranda a free flank
                allows the column it needs, leaving four; and four pinned to the attached flank
                put the house's *corner* on the reserved doorstep, where the shell lays a corner
                post and not a door. Swept through the generated production call at every column
                of the frontage, before the fix: an east-attached end failed on the one column
                its corner landed on and a west-attached end on the opposite one; a lot attached
                on both flanks passed at all six, because there the house already fills the pad.

                So an end spends its slack: one free flank is a column the terrace does not need,
                and a side deck is worth less than a front door. The widths are tried from `ww`
                up to the whole pad for a doorstep strictly inside the wall, then again for one
                merely in it, and the pin itself is never given up -- a hole in a terrace is not
                the alternative to a door.
                
        """
        for strict in (True, False):
            for cand in range(ww, max(ww, pad_w) + 1):
                aa = pin(cand)
                if aa is None or aa < latlo or aa + cand - 1 > lathi:
                    continue
                if ((aa < dlat < aa + cand - 1) if strict
                        else (aa <= dlat <= aa + cand - 1)):
                    return cand, aa
        return ww, pin(ww)

    # Slide the frontage along the pad so the reserved doorstep lands in the front wall,
    # and off its corner where there is room for that.
    a = pin(w)
    if a is None:
        spans = [aa for aa in range(latlo, lathi - w + 2)]
        strict = [aa for aa in spans if aa < dlat < aa + w - 1]
        loose = [aa for aa in spans if aa <= dlat <= aa + w - 1]
        a = rng.choice(strict) if strict else (loose[0] if loose else latlo)
    else:
        w, a = held(w)

    # How many floors this plot can actually carry a walkable stair between.
    top = 1
    if house_d >= 8 and w >= 5:
        top = 2
    if house_d >= 10 and w >= 6:
        top = 3
    if storeys is None:
        storeys = rng.choice([1, 2, 2, 3])
    st = max(1, min(int(storeys), top))
    asked_storeys = int(storeys)
    capped_by_lot = st < asked_storeys
    if front not in _FRONT:
        front = rng.choice(list(_FRONT.keys()))

    vr = part.get("roof") or {}
    profile = vr.get("profile") or rng.choice(
        [[(1, 2), (2, 1)], [(1, 2), (3, 1)], [(1, 3), (2, 1)]])
    eave_kind = vr.get("eave") or "upturned"
    tiers = 1
    # **The ends are not the voice's to give this house.** Composition round. A row
    # house's gable stands against the next house, so what happens at the end of the
    # ridge is a fact about the end of the **row**; a voice that says "irimoya" is
    # describing a building that stands free, and this one does not. The eave's reach
    # *is* the voice's -- a crowded ring in a four-block lane cannot afford the two
    # blocks a ring of courtyard houses standing back behind their own gates can -- and
    # it arrives on `building()` from `TypeBuilder` without this file naming it.
    ends = ("gable", "gable") if rng.random() < 0.5 else ("half-hip", "half-hip")
    roofspec = {"style": "gable", "axis": ax, "profile": profile,
                "ends": ends, "eave": eave_kind, "tiers": tiers}
    if min(w, house_d) <= 6:
        # on a short span the bare gable leaves the course under the ridge facing down-
        # slope; hipping the top of it lands cleanly.
        roofspec["ends"] = ("half-hip", "half-hip")

    chim = True if st >= 2 else None

    attempts = []
    for stt in range(st, 0, -1):
        attempts.append((house_d, w, stt))
    if house_d > 6:
        attempts.append((house_d - 1, w, 1))
    if w > 5 and not (att_lo and att_hi):
        # narrowing is a rung on the ladder only where there is a free flank to narrow
        # toward: between two party walls it would lay the slot back again
        attempts.append((house_d, w - 1, 1))
    if (att_lo or att_hi) and w < pad_w:
        # the last resort where a frontage pinned to one party wall misses the reserved
        # doorstep on the free side: fill the pad, which always covers it
        attempts.append((house_d, pad_w, 1))

    res = None
    rung = None
    for i, (hd, ww, stt) in enumerate(attempts):
        if pin(ww) is not None:
            # every rung of the ladder is placed the way the first one was, so a
            # narrower shell does not put the door back on the corner
            ww, aa = held(ww)
        else:
            aa = min(max(a, dlat - ww + 1), dlat)
        aa = min(max(aa, latlo), lathi - ww + 1)
        if not (aa <= dlat <= aa + ww - 1):
            continue
        a = aa
        x0, z0, x1, z1 = rect(hd, ww, aa)
        res = b.building(label, x0, z0, x1, z1, stt, roofspec,
                         openings="none", stair="none", mat=voice,
                         brackets=(ww >= 8), chimney=(chim if stt >= 2 else None))
        if res and res.get("ok"):
            house_d, w, st, rung = hd, ww, stt, i
            break
    if not (res and res.get("ok")):
        return dict(res or {"ok": False}, emitted={
            "requested": {"storeys": asked_storeys}, "storeys": 0, "attempt": None,
            "fallback": "no shell stood", "omitted": ["storeys"], "features": {}})
    # **What survived, said by the type.** The closure round: the lot's depth and width
    # cap the storeys before anything is built, and the ladder then walks the storeys
    # down; both are on the record beside the geometry `construction.outcome` measures.
    emitted = {
        "requested": {"storeys": asked_storeys},
        # the rung of the ladder that stood, by position: `held()` may have widened the
        # frontage off it, so the triple is no longer a key into the list
        "storeys": int(st), "attempt": int(rung),
        "fallback": ("; ".join(
            ([f"lot: storeys {asked_storeys} -> {min(asked_storeys, top)} for a "
              f"{house_d}x{w} house"] if capped_by_lot else [])
            + ([f"ladder: storeys walked down to {st}"]
               if st < min(asked_storeys, top) else [])) or None),
        "omitted": [] if st >= asked_storeys else ["storeys"],
        "features": {"chimney": bool(chim if st >= 2 else None),
                     "brackets": bool(w >= 8)},
        "rects": {"main": list(rect(house_d, w, a))},
        "floors": [fy + 4 * i for i in range(st)],
        # what the plan said about this lot and what the house did with it
        "front_from": front_from,
        "attached": sorted(s2 for s2, on in ((lo_side, att_lo), (hi_side, att_hi)) if on),
        "fills_pad": bool(w == pad_w),
        # **The one case `held()` cannot solve, said out loud.** Where both flanks are
        # attached the house already fills its pad and there is no slack to spend, so a
        # doorstep reserved on the pad's own first or last column is at the end of the
        # wall whatever this file does. Nothing in the section hit it -- it is recorded
        # rather than assumed absent, because the alternative is a silent E008.
        "door_at_corner": bool(dlat in (a, a + w - 1)),
    }

    floors = [fy + 4 * i for i in range(st)]
    ridge_y = res.get("ridge_y") or (fy + 4 * st + 4)
    inner_lo, inner_hi = a + 1, a + w - 2
    back = house_d - 2                     # last interior depth
    tp = min(max(dlat, inner_lo), inner_hi)  # the passage, in line with the door

    wall_blk = b.block(voice["wall"], "full")
    frame_blk = b.block(voice["frame"], "full")
    board_blk = b.block(voice["trim"], "full")
    fence = b.joinery(voice, "fence")

    # --- the party-wall verge: no roof over the next house's ground --- **A pocket
    # between a low roof and a tall neighbour is a room nobody can enter.** The
    # neighbourhood delivery round, measured in `out/nd-block` and reproduced offline on
    # a plane (two adjacent lots, one storey beside two): `E003 a room of 15 cells at
    # (-5706,72,656) cannot be walked into`, with `E011` beside it, five times -- once
    # for every one-storey house in the block that had a taller neighbour. The geometry
    # is a single column, the depth of the house, two blocks high: its floor is this
    # house's own roof where it meets the party wall, its walls are that roof rising on
    # one side and the neighbour's on the other, and its ceiling is the neighbour's eave
    # oversailing the plot line. The same junction gave `E004` eleven times, a roof
    # stair reading as down-slope because the ground ahead of it fell into the pocket.
    # The ridge of this house runs front to back, so what faces a party wall is a roof
    # **slope** and not a gable -- and a slope stopping short of the plot line is what
    # leaves the pocket. So on an attached flank the roof is closed with masonry from
    # the wall head to the ridge: a verge standing on the party wall, which is what a
    # terrace has and is the `udatsu` of the machiya this house is. Nothing is left for
    # a neighbour's eave to roof over, whatever height the neighbour is, and a free
    # flank keeps its eave because there is no plot line under it.
    verge_cells = 0
    for on, tt in ((att_lo, a), (att_hi, a + w - 1)):
        if not on:
            continue
        for dd in range(0, house_d):
            X, Z = cell(dd, tt)
            for yy in range(fy + 4 * st, ridge_y + 1):
                b.place_block(X, yy, Z, wall_blk)
                verge_cells += 1
    emitted["verge"] = verge_cells

    # **A verge carries the roof, so the roof it carries may not have a channel in it.**
    # The same measurement, one step on. With the verge standing to the ridge, `E004`
    # appeared at two depths of every attached house: `deepslate_tile_stairs at
    # [24,70,27] faces down-slope (ahead y=69, behind y=71)`. The cause is not the verge
    # -- it is a **hip** end, which the voice writes over this file's own choice of ends
    # (`TypeBuilder._silhouette`). Measured by standing one house five times with only
    # the voice's roof key changed: `ends: "hip"` leaves the ridge one course below the
    # slope on either side of it for the two rows where the hip ramps -- one column on a
    # five-wide house, two on a six-wide one -- and `gable`, `half-hip`, no ends at all
    # and no profile all stand clean. The channel was always there; until a party wall
    # stood beside it nothing on the other side was high enough for the linter to call
    # the tread down-slope. So the channel is closed where it is found: a run of columns
    # lying below **both** its shoulders comes up to the lower shoulder, in one pass off
    # tops read before anything is laid, so nothing cascades. The verge is never a
    # shoulder -- the valley between the ridge and a party wall is a gutter and is meant
    # to be there, and treating the verge as a shoulder would fill the roof solid from
    # the ridge to the wall. Only on a house that has a party wall at all; whether a
    # voice's `ends` should reach a type that declares `ATTACHED` is `buildlib`'s
    # question, not this file's.
    if att_lo or att_hi:
        roof_blk = b.block(voice["roof"], "full")

        def roof_top(tt, dd):
            X, Z = cell(dd, tt)
            for yy in range(ridge_y, fy + 4 * st - 1, -1):
                got = b.get_block(X, yy, Z)
                if got and got != "air":
                    return yy
            return None

        # the columns a shoulder may be: the roof's own, never a verge
        s_lo = a + 1 if att_lo else a
        s_hi = a + w - 2 if att_hi else a + w - 1
        closed = 0
        for dd in range(0, house_d):
            tops = {tt: roof_top(tt, dd) for tt in range(s_lo, s_hi + 1)}
            if any(v is None for v in tops.values()):
                continue
            tt = s_lo + 1
            while tt <= s_hi - 1:
                run = tt
                while run + 1 <= s_hi - 1 and tops[run + 1] == tops[tt]:
                    run += 1
                left, right = tops[tt - 1], tops[run + 1]
                if left > tops[tt] and right > tops[tt]:
                    want = min(left, right)
                    for c in range(tt, run + 1):
                        X, Z = cell(dd, c)
                        for yy in range(tops[tt] + 1, want + 1):
                            b.place_block(X, yy, Z, roof_blk)
                            closed += 1
                tt = run + 1
        emitted["ridge_closed"] = closed

    # --- the cross walls: a front room on the lane, rooms running back ---
    dps = [dp for dp in _partitions(rng, house_d) if 2 <= dp <= back - 1]
    if st >= 2:
        # the flight stands in the last stretch of the passage, so no cross wall may
        # fall so late that a room behind it is entered only over the treads.
        dps = [dp for dp in dps if dp <= back - 6]
    for dp in dps:
        for tt in range(inner_lo, inner_hi + 1):
            if tt == tp:
                continue
            blk = frame_blk if abs(tt - tp) == 1 else wall_blk
            X, Z = cell(dp, tt)
            for yy in range(fy + 1, fy + 4):
                b.place_block(X, yy, Z, blk)
        X, Z = cell(dp, tp)
        b.place_block(X, fy + 3, Z, frame_blk)          # header over the passage

    # one screen upstairs, never reaching across, so nothing can be shut in **...and the
    # furniture has to know it is there.** The neighbourhood delivery round, measured
    # through `test_compile` case 7's own build: the upper floor of a two-storey terrace
    # house read 15 of 42 floor cells unreachable on foot, 100% with a jump allowed.
    # `holds()` decides whether a fitting cuts a room in half by flooding the room with
    # `occ` blocked -- and these cells were never in `occ`, because `occ` is built below
    # this line. So the flood walked straight through the screen, judged the far strip
    # connected, and `lamps()` put a lantern in the one cell that actually joined it to
    # the stair. The screen is recorded here and `occ` starts from it.
    screened = set()
    if st >= 2 and inner_hi - inner_lo >= 2 and rng.random() < 0.6:
        dscr = max(2, min(back - 1, back // 2 + 1))
        edge = inner_lo if tp >= (inner_lo + inner_hi) / 2.0 else inner_hi
        step = 1 if edge == inner_lo else -1
        for k in range(max(1, (inner_hi - inner_lo) // 2)):
            tt = edge + step * k
            X, Z = cell(dscr, tt)
            screened.add((dscr, tt))
            for yy in range(floors[1] + 1, floors[1] + 4):
                b.place_block(X, yy, Z, wall_blk if k else frame_blk)

    # --- the stair, in the passage, where the passage runs out ---
    busy = set()      # cells the flight itself stands in
    keep = set()      # cells left clear so the flight can be got on and off
    landing = {}
    for i in range(st - 1):
        if i == 0:
            cols = [tp]
        else:
            cols = [c for c in (tp + 1, tp - 1) if inner_lo <= c <= inner_hi] + [tp]
        runs = []
        for c in cols:
            if i == 0:
                for d0 in (back - 4, back - 5, 2):
                    if 2 <= d0 and d0 + 4 <= back:
                        runs.append((c, d0, inward, 1))
            else:
                for d0 in (back - 1, back - 2):
                    if d0 - 4 >= 1:
                        runs.append((c, d0, outward, -1))
                for d0 in (2, 3):
                    if d0 + 4 <= back:
                        runs.append((c, d0, inward, 1))
        for (c, d0, dirn, step) in runs:
            X, Z = cell(d0, c)
            r = b.flight(label, X, Z, floors[i], floors[i + 1], dirn,
                         mat=voice["floor"])
            if r and r.get("ok"):
                landing[i + 1] = (d0 + step * 4, c)
                for k in range(0, 4):
                    busy.add((d0 + step * k, c))
                for k in (-1, 4, 5):
                    keep.add((d0 + step * k, c))
                for k in range(1, 5):        # box the flight in to the floor
                    X, Z = cell(d0 + step * k, c)
                    for yy in range(floors[i] + 1, floors[i] + 1 + k):
                        b.place_block(X, yy, Z, wall_blk)
                break

    # --- the street front, the back wall, and nothing on the party walls --- Screened,
    # not glazed: the bay is knocked out of the boarding and filled with a lattice, so
    # the light comes through it and a person does not.
    ow, ogap, osill, ohead = _FRONT[front]

    def screen(dd, lo, hi, yy, wid, gap, sill, head):
        run = hi - lo + 1
        wid = min(wid, run)
        if run < 1 or wid < 1:
            return
        n = (run + gap) // (wid + gap)
        if n < 1:
            return
        start = lo + (run - (n * wid + (n - 1) * gap)) // 2
        for k in range(n):
            for tt in range(start + k * (wid + gap),
                            start + k * (wid + gap) + wid):
                X, Z = cell(dd, tt)
                for y2 in range(yy + sill, yy + min(head, 3) + 1):
                    b.place_block(X, y2, Z, fence)

    for i in range(st):
        yy = floors[i]
        segs = []
        if i == 0:
            if dlat - 2 >= a + 1:
                segs.append((a + 1, dlat - 2))
            if dlat + 2 <= a + w - 2:
                segs.append((dlat + 2, a + w - 2))
        else:
            segs.append((a + 1, a + w - 2))
        for (lo, hi) in segs:
            screen(0, lo, hi, yy, ow, ogap, osill if i == 0 else 2, ohead)
        screen(house_d - 1, a + 1, a + w - 2, yy,
               3 if w >= 7 else 2, 2, 2, 3)

    # The veranda is entered from the house, at the back of the passage: it is covered
    # ground, and covered ground nobody can walk onto is a fault. **An opening, not a
    # door, and the refusal it replaces was not about the front.** `doorway()` is the
    # arrival's: `TypeBuilder._doorway` admits it only at the cell `site()` reserved,
    # because a type does not decide where the town arrives. This call was for the
    # *back* of the house and was refused every time -- seventy-six times in the
    # delivered section -- and an independent reader read those refusals as "every
    # `row_house` put its door in the wrong place and had it overridden". The front door
    # was never wrong; the veranda simply never got its way in, and the engawa stayed
    # covered ground that could not be walked onto from inside. So the rear threshold is
    # cut as what it is: a one-column opening in the back wall at the end of the
    # passage, sill on the floor and head two above it, laid by `openings()` -- which is
    # the library's own call for a hole in a wall run and needs no leaf. A machiya's
    # engawa is behind a sliding screen, not a hinged door; the lattice the back wall
    # already carries stands over the head as its transom.
    if rear >= 1:
        e0 = cell(house_d - 1, a)
        e1 = cell(house_d - 1, a + w - 1)
        b.openings(e0[0], fy, e0[1], e1[0], e1[1], at=[tp], width=1,
                   sill=1, head=2, block="air")

    # --- what each room is for ---
    zones = []
    lo = 1
    for dp in sorted(dps):
        if dp - 1 >= lo:
            zones.append((lo, dp - 1))
        lo = dp + 1
    if back >= lo:
        zones.append((lo, back))
    if not zones:
        zones = [(1, back)]

    def zone_cells(zone):
        d0, d1 = zone
        return set((dd, tt) for dd in range(d0, d1 + 1)
                   for tt in range(inner_lo, inner_hi + 1))

    def foot(kind, dd, tt):
        """What a fitting might take up: its own cell, and for the ones that
        run along a wall or lie down, the cells beside it as well."""
        if kind not in _WIDE:
            return {(dd, tt)}
        return set((dd + i, tt + j) for i in (-1, 0, 1) for j in (-1, 0, 1))

    def holds(zone, blocked, roots):
        """Is every cell of this room still walkable from its way in?"""
        free = set(c for c in zone_cells(zone) if c not in blocked)
        seeds = [c for c in roots if c in free]
        if not seeds:
            return False
        seen, stack = set(seeds), list(seeds)
        while stack:
            dd, tt = stack.pop()
            for nb in ((dd + 1, tt), (dd - 1, tt), (dd, tt + 1), (dd, tt - 1)):
                if nb in free and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(free)

    def wall_spots(zone, from_back, side, anywhere=False, corners=False):
        """Cells with their back to a wall, in the order asked for."""
        d0, d1 = zone
        rows = list(range(d1, d0 - 1, -1)) if from_back else list(range(d0, d1 + 1))
        cols = [inner_hi, inner_lo] if side >= 0 else [inner_lo, inner_hi]
        cols += [t for t in range(inner_lo, inner_hi + 1) if t not in cols]
        out, spare = [], []
        for tt in cols:
            for dd in rows:
                if tt == tp or (dd, tt) in occ or (dd, tt) in keep:
                    continue
                if corners:
                    if tt in (inner_lo, inner_hi) and dd in (d0, d1):
                        out.append((dd, tt))
                elif tt in (inner_lo, inner_hi) or dd in (d0, d1):
                    out.append((dd, tt))
                elif anywhere:
                    spare.append((dd, tt))
        return out + spare

    occ = set(busy) | screened
    # the way in, the length of the passage and the foot and head of every flight:
    # nothing may lean into these, however well it fits.
    sacred = set(keep)
    sacred.update((dd, tp) for dd in range(0, back + 1))
    for dp in dps:
        for tt in range(inner_lo, inner_hi + 1):
            occ.add((dp, tt))

    def roots_of(zone, level):
        if level == 0:
            return [(dd, tp) for dd in range(zone[0], zone[1] + 1)]
        ld, lc = landing.get(level, (zone[1], tp))
        return [(ld + 1, lc), (ld - 1, lc), (ld, lc + 1), (ld, lc - 1), (ld, lc)]

    def fit(kind, zone, level, facing, from_back=False, side=0,
            anywhere=False, **kwargs):
        yy = floors[level] + 1
        roots = roots_of(zone, level)
        # Anything that runs along a wall or lies down goes in a corner: a corner it
        # overflows leaves the rest of the floor in one piece, and the middle of a wall
        # does not.
        for (dd, tt) in wall_spots(zone, from_back, side, anywhere,
                                   kind in _WIDE):
            spread = foot(kind, dd, tt)
            if spread & sacred:
                continue
            if not holds(zone, occ | spread, roots):
                continue
            X, Z = cell(dd, tt)
            r = b.fitting(kind, X, yy, Z, facing, **kwargs)
            if r and r.get("ok"):
                occ.update(spread)
                took = r.get("cells")
                if isinstance(took, (list, tuple)):
                    for c in took:
                        if isinstance(c, (list, tuple)) and len(c) >= 3:
                            occ.add(uncell(c[0], c[2]))
                return (dd, tt)
        return None

    def lamps(zone, level, n):
        """Light, and keep asking until the room has some. A lantern hung from
        the tie beam wants headroom a low room may not have; a torch on the
        wall does not."""
        got = 0
        for k in range(2 * n + 4):
            if got >= n:
                break
            if fit("light", zone, level, inward, from_back=bool(k % 2),
                   side=(1 if k % 4 < 2 else -1), mat=voice["trim"],
                   room=("hall", "store", "shrine", "house")[k % 4]):
                got += 1

    wallside = 1 if tp - inner_lo <= inner_hi - tp else -1

    # light first: a small room fills up, and a dark room is the one fault a chair in it
    # will not excuse.
    for zn in zones:
        cnt = (zn[1] - zn[0] + 1) * (inner_hi - inner_lo + 1)
        lamps(zn, 0, 2 if cnt >= 20 else 1)

    front_zone = zones[0]
    fit("table", front_zone, 0, outward, side=wallside,
        mat=voice["floor"], room="house")
    fit("bench", front_zone, 0, inward, side=-wallside,
        mat=voice["floor"], room="house")

    back_zone = zones[-1]
    hearth_kw = {"mat": voice["footing"], "room": "house"}
    if st == 1:
        hearth_kw["flue_to"] = ridge_y
    fit("hearth", back_zone, 0, outward, from_back=True, side=wallside,
        **hearth_kw)
    fit("store", back_zone, 0, lat_name(-wallside), from_back=True,
        side=-wallside, mat=voice["floor"], extent=2,
        room="store")

    mids = zones[1:-1] if len(zones) > 2 else []
    for k, zn in enumerate(mids):
        fit("shelf" if k % 2 else "bench", zn, 0, lat_name(wallside),
            side=-wallside, mat=voice["floor"], room="house")
    if st == 1:
        zn = mids[-1] if mids else back_zone
        fit("bed", zn, 0, lat_name(wallside), from_back=True, side=-wallside,
            mat=voice["floor"], room="house")

    for i in range(1, st):
        whole = (1, back)
        lamps(whole, i, 2 if back * (inner_hi - inner_lo + 1) >= 20 else 1)
        fit("bed", whole, i, lat_name(-wallside), from_back=True, side=wallside,
            mat=voice["floor"], room="house")
        fit("bed", whole, i, lat_name(wallside), side=-wallside,
            mat=voice["floor"], room="house")
        fit("table", whole, i, outward, side=-wallside,
            mat=voice["floor"], room="house")
        fit("bookshelf" if i == 1 else "shelf", whole, i, lat_name(wallside),
            from_back=True, side=-wallside, mat=voice["floor"], room="house")

    # --- the engawa: boards the length of the house, under the eave ---
    if rear >= 1:
        nout = rear
        runs = list(range(a, a + w))

        def deck_cell(k, t):
            return cell(house_d - 1 + k, t)
    else:
        left, right = a - latlo, lathi - (a + w - 1)
        sd = 1 if right >= left else -1
        nout = min(2, right if sd > 0 else left)
        runs = list(range(1, house_d))
        edge = (a + w - 1) if sd > 0 else a

        def deck_cell(k, t):
            return cell(t, edge + sd * k)

    have = set()
    for k in range(1, nout + 1):
        for tt in runs:
            X, Z = deck_cell(k, tt)
            if not (px0 <= X <= px1 and pz0 <= Z <= pz1):
                continue
            if fy - 1 <= b.get_height(X, Z) <= fy:
                b.place_block(X, fy, Z, board_blk)
                have.add((k, tt))
    door_t = tp if rear >= 1 else runs[0]
    if have:
        gap = rng.choice([2, 3])
        posts = set()
        # Posts and rail stand on the outer course, so the boards next to the house stay
        # a walk from end to end. On a veranda one board wide there is no outer course:
        # leave it bare and let the eave carry itself.
        if nout >= 2:
            step_at = runs[len(runs) // 2]
            for tt in runs:
                if (nout, tt) not in have:
                    continue
                X, Z = deck_cell(nout, tt)
                # **the step is never a post** (the fabric reset round): with a gap of 3
                # on a six-wide deck the middle column is a post's, and the deck's one
                # way down was walled -- found once party walls stopped leaving a slot
                # at the deck's ends, which had been the only other way onto it
                if (tt - runs[0]) % gap == 0 and tt != step_at:
                    for yy in range(fy + 1, fy + 4):
                        b.place_block(X, yy, Z, frame_blk)
                    posts.add(tt)
                elif tt != step_at:
                    b.place_block(X, fy + 1, Z, fence)
        if st >= 2:
            # **A ledge with a roof over it is a room nobody can get into.** The
            # neighbourhood round, measured through `type_needs.instance` at pads 10x10
            # and 11x11: this laid the storey line as a slab the whole length of the
            # veranda, the main roof's eave already reached over the same ground two
            # courses higher, and the void between them came back as `E003 a room of 8
            # cells at (23,68,22) cannot be walked into` with `E011` beside it -- an
            # enclosed gallery at first-floor level, floored, walled, roofed and with no
            # way in. It is the same fault this file names twenty lines above about the
            # veranda itself, one storey up. So the lip is laid where the sky is over it
            # -- an eyebrow on an open deck, which is what it is for -- and not where
            # something already stands there.
            lip = b.block(voice["trim"], "slab")
            for (k, tt) in sorted(have):
                X, Z = deck_cell(k, tt)
                if any(b.get_block(X, yy, Z) != "air"
                       for yy in range(fy + 5, ridge_y + LIP_HEADROOM)):
                    continue
                b.place_block(X, fy + 4, Z, lip)
        # a lantern under the eave: the veranda is covered ground and stays dark all day
        # unless something is hung over it.
        ends = [tt for tt in (runs[0], runs[-1])
                if (1, tt) in have and tt != door_t]
        for tt in ends:
            X, Z = deck_cell(1, tt)
            b.fitting("light", X, fy + 1, Z, outward if rear else lat_name(sd),
                      mat=voice["trim"], room="hall")

    # anything the furniture shut in behind it, if it can be proved shut in
    fx0, fz0, fx1, fz1 = rect(house_d, w, a)
    b.seal_voids(fx0, fz0, fx1, fz1, wall_blk)
    # ...and on the whole pad, where a party wall now stands on the plot line (the
    # fabric reset round: `site()` used to cut a slot through the neighbour's wall, and
    # the slot was the only way into the strip of pad in front of a set-back house; with
    # the wall kept, that strip is a pocket nobody can enter). Only a pocket that is
    # provably enclosed, unreachable and wholly on this pad is closed.
    b.seal_voids(part["x0"], part["z0"], part["x1"], part["z1"], wall_blk)

    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "storeys": st, "depth": house_d, "width": w, "emitted": emitted,
            "front": front, "rear": rear}
