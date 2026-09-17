"""A ring-wall gatehouse: an arch a road walks through, guard chambers over and
beside it, a mural stair carrying the wall walk up through the gate, a tiered crown.

**A gate is sized by its wall.** Where the pad stands on an edge, siting hands this
type the edge's height in `part["edge"]` and sizes the pad from it; a gate in a wall
taller than a gatehouse's own deck (`TALL_FROM`) is then built as a **gate tower**: a
solid base to the wall's crown with the arch through it and a switchback stair in one
pier, the wall walk crossing on the base's top, and the chambers as a pavilion above
the crown under the tiered roof. In a low wall, or with no wall named, it is the
gatehouse it always was.
"""

import random

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"
KIND = "point"
PASSAGE = True

PARAMS = {
    "storeys": ("int", 1, 3),
    "crown": ("choice", ["gable", "hip", "pavilion"]),
}

NEEDS = {
    "footprint": (5, 5, 16, 16),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

AIR = "air"

#: A wall this many blocks over the gate's floor is one a gatehouse's deck cannot clear:
#: the gate is built as a tower to the wall's crown. A gatehouse's own deck is four to
#: six over the road and its chambers stop under twenty; a town wall of five and a
#: precinct wall of eight stay under this and keep the gatehouse.
TALL_FROM = 12
#: The arch through a gate tower's base, over the road: this many blocks of headroom,
#: **at the least**.
ARCH_HEADROOM = 6

#: A passage six high and the road's width through a pier of forty-eight is a mouse-hole
#: at the foot of a cliff, and making the *tower* scale with the wall (demo-polish, 2b)
#: made the hole look smaller rather than larger. The headroom and the span are
#: fractions of the wall's own height -- 48 gives 16 and 12, 20 gives 7 and 5 -- floored
#: at `ARCH_HEADROOM` and at the road's own width, and bounded above by the pad (a pier
#: two thick either side) and by the arch head standing under the deck. A gate that is
#: not on a wall keeps what it had.
ARCH_HEADROOM_OF_WALL = 0.35
ARCH_SPAN_OF_WALL = 0.25

def _dirname(along_x, step):
    if along_x:
        return "east" if step > 0 else "west"
    return "south" if step > 0 else "north"

def build(b, part, seed, **params):
    rnd = random.Random((seed * 6364136223846793005 + 1442695040888963407) % (1 << 61))
    voice = part["voice"]
    storeys = int(params.get("storeys", 2))
    storeys = max(1, min(3, storeys))
    crown = params.get("crown", "hip")

    WALL = b.block(voice["wall"], "full")
    FOOT = b.block(voice["footing"], "full")
    FRAME = b.block(voice["frame"], "full")
    TRIM = b.block(voice["trim"], "full")
    FLOOR = b.block(voice["floor"], "full")
    FOOT_SLAB = b.block(voice["footing"], "slab")
    WALL_SLAB = b.block(voice["wall"], "slab")

    x0, x1 = sorted((part["x0"], part["x1"]))
    z0, z1 = sorted((part["z0"], part["z1"]))
    F = part["floor_y"]
    facing = part.get("facing") or "north"
    at = part.get("at") or [(x0 + x1) // 2, (z0 + z1) // 2]
    along_x = facing in ("east", "west")

    # u runs along the way the gate faces (the road), v runs across it (the wall line).
    if along_x:
        u0, u1, v0, v1 = x0, x1, z0, z1
        v_at = int(at[1])
    else:
        u0, u1, v0, v1 = z0, z1, x0, x1
        v_at = int(at[0])
    U = u1 - u0 + 1
    V = v1 - v0 + 1
    v_at = max(v0 + 1, min(v1 - 1, v_at))

    def XZ(u, v):
        return (u, v) if along_x else (v, u)

    def cell(u, v, y):
        x, z = XZ(u, v)
        return (x, y, z)

    def put(u, v, y, blk):
        x, z = XZ(u, v)
        b.place_block(x, y, z, blk)

    def box(ua, va, ya, ub, vb, yb, blk):
        xa, za = XZ(ua, va)
        xb, zb = XZ(ub, vb)
        b.place_cuboid(min(xa, xb), min(ya, yb), min(za, zb),
                       max(xa, xb), max(ya, yb), max(za, zb), blk)

    lane = set()
    for xx in range(x0, x1 + 1):
        for zz in range(z0, z1 + 1):
            nl = b.nearest_lane(xx, zz)
            if nl and int(nl["distance"]) == 0:
                lane.add((xx, zz))

    def is_lane(u, v):
        return XZ(u, v) in lane

    # ---- proportions, all seed-driven -------------------------------------
    span = 5 if (V >= 11 and rnd.random() < 0.5) else 3
    door = part.get("door")
    if door:
        d_v = int(door[1]) if along_x else int(door[0])
        v_at = max(v0 + 1, min(v1 - 1, (v_at + d_v) // 2))
    pv0 = max(v0 + 1, min(v_at - span // 2, v1 - span))
    pv1 = pv0 + span - 1
    if door:
        if d_v < pv0:
            pv0 = max(v0 + 1, d_v)
        elif d_v > pv1:
            pv0 = min(v1 - span, d_v - span + 1)
        pv1 = pv0 + span - 1
    rise = min(U - 1, rnd.choice([4, 5, 5, 6]))      # ground -> wall-walk deck
    rise = max(3, rise)
    # The wall this gate stands in, where siting named one: the deck is the wall's
    # crown, and the gate is a tower.
    edge = part.get("edge") or {}
    crown = None
    if edge.get("height"):
        wall_floor = edge.get("floor_y")
        crown = int(wall_floor if wall_floor is not None else F) + int(edge["height"])
    # ...on a pad with piers two lanes thick either side of a three-wide arch, which is
    # what the mural stair needs; siting sizes the pad from the wall's height so a wall
    # this tall never arrives on less
    tall = crown is not None and (crown - F) >= TALL_FROM and U >= 9 and V >= 9
    # ...and only where one pier has two lanes the road does not run through, for the
    # mural stair: a road that crosses the pad on the skew leaves no pier, and then the
    # gate is the gatehouse it always was rather than a tower nobody can climb. Decided
    # here, before the deck is, so nothing is built to a crown it cannot reach.
    stair_sides = []
    if tall:
        for side_ in (0, 1):
            lanes_ = [v0 + 1, v0 + 2] if side_ == 0 else [v1 - 1, v1 - 2]
            if not any(is_lane(uu, lv) for uu in range(u0 + 1, u1) for lv in lanes_):
                stair_sides.append(side_)
        tall = bool(stair_sides)
    if tall:
        rise = crown - F
    DECK = F + rise                                   # the walk / first chamber floor
    CH = max(3, min(4, U - 2))                        # chamber height
    # the arch head over the road: the deck in a gatehouse, a course of headroom over
    # the road in a tower
    wall_h = int(((part.get("edge") or {}).get("height") or 0))
    headroom = max(ARCH_HEADROOM, int(round(wall_h * ARCH_HEADROOM_OF_WALL)))
    # ...and never so high that the mass over it is a lintel: four courses at least
    # between the arch head and the deck the wall's walk stands on.
    HEAD = (DECK - 1 if not tall
            else F + max(ARCH_HEADROOM, min(headroom, DECK - F - 5)) + 1)
    foot_courses = rnd.choice([1, 2])
    crest = rnd.choice([0, 1, 1])                     # a parapet course, or none
    top_floor = DECK + (storeys - 1) * CH
    eave_y = top_floor + CH + crest

    # **The arch is as wide as the road.** Where the lane runs through the pad, the
    # passage spans the lane's columns across the wall line -- a city's arterial is four
    # wide and a three-wide arch beside it is an arch the road goes round -- bounded so
    # that a pier at least two thick stands either side of it. ...the road being the
    # lane that runs **through** the pad, along the way the gate faces: a lane column
    # present at both mouths. A lane along the foot of the wall inside the city crosses
    # the pad too, and an arch spanning it would be a hall, not a gate.
    road_v = sorted({v for v in range(v0 + 2, v1 - 1)
                     if is_lane(u0, v) and is_lane(u1, v)})
    if not road_v:
        road_v = sorted({v for v in range(v0 + 2, v1 - 1)
                         if is_lane((u0 + u1) // 2, v)})
    if road_v and V >= 7:
        # the arch's span: the road's width, or the wall's own fraction where that is
        # wider, bounded by a pier two thick either side
        from_wall = int(round(wall_h * ARCH_SPAN_OF_WALL))
        want = max(span, min(V - 4, max(road_v[-1] - road_v[0] + 1, from_wall)))
        pv0 = max(v0 + 2, min(road_v[0], v1 - 1 - want))
        pv1 = pv0 + want - 1
        span = want

    def clear_over_lane(y_lo, y_hi):
        for u in range(u0, u1 + 1):
            for v in range(v0, v1 + 1):
                if is_lane(u, v):
                    box(u, v, y_lo, u, v, y_hi, AIR)

    def stair_col(side):
        return v0 if side == 0 else v1

    def inward(sv):
        return sv + 1 if sv == v0 else sv - 1

    # ---- the mass, and the road cut through it ---------------------------- Where the
    # lane already runs across this cell, the gate stands over it and not in it: nothing
    # at all between the lane surface and the deck -- and in a tower, nothing between
    # the lane surface and the arch head, the base solid above.
    box(u0, v0, F + 1, u1, v1, DECK - 1, AIR)
    for u in range(u0, u1 + 1):
        for v in range(v0, v1 + 1):
            if is_lane(u, v) or pv0 <= v <= pv1:
                if tall and HEAD < DECK - 1:
                    box(u, v, HEAD, u, v, DECK - 1, WALL)
                continue
            box(u, v, F, u, v, DECK - 1, WALL)
            box(u, v, F, u, v, F + foot_courses - 1, FOOT)
    for u in range(u0, u1 + 1):
        for v in range(pv0, pv1 + 1):
            if not is_lane(u, v):
                put(u, v, F, FOOT)

    # the arch head: the springing course solid, the span open, the mouths chamfered
    for v in (pv0, pv1):
        for u in range(u0, u1 + 1):
            if not is_lane(u, v):
                put(u, v, HEAD, FOOT)
    for u in (u0, u1):
        for v in (pv0, pv1):
            if not is_lane(u, v):
                put(u, v, HEAD, FOOT_SLAB + "[type=top]")

    # **The arch is framed**, the craft round (E4). Sized from its wall the opening is
    # eleven wide and seventeen high in a pier of forty-eight, and from the air it still
    # reads as a dark slit in an otherwise unbroken face, because nothing frames it. A
    # jamb down each side of the mouth, an arch ring over the head, a relieving course
    # two above it and the pier stepping back either side of that: the opening reads as
    # an entrance rather than a hole cut in a cliff.
    for u in (u0, u1):
        for v in (pv0 - 1, pv1 + 1):
            if v0 <= v <= v1 and not is_lane(u, v):
                box(u, v, F + 1, u, v, HEAD - 1, TRIM)
        for v in range(pv0 - 1, pv1 + 2):
            if v0 <= v <= v1 and not is_lane(u, v):
                put(u, v, HEAD, TRIM)
        if HEAD + 2 <= DECK - 1:
            for v in range(pv0 - 2, pv1 + 3):
                if v0 <= v <= v1 and not is_lane(u, v):
                    put(u, v, HEAD + 2, TRIM)
        # the pier's mass stepping back either side of the relieving course
        if HEAD + 3 <= DECK - 1:
            for v in (pv0 - 2, pv1 + 2):
                if v0 <= v <= v1 and not is_lane(u, v):
                    put(u, v, HEAD + 3, TRIM)

    # guard alcoves beside the arch, where a pier is thick enough to hold one; in a
    # tower they are guard rooms at the foot of the base, a person's height and a
    # course, and the mural stair rises out of one of them
    alcoves = []
    alc_top = DECK - 1 if not tall else min(DECK - 1, F + 3)
    for side in (0, 1):
        sv = stair_col(side)
        cols = [v for v in (range(v0 + 1, pv0) if side == 0 else range(pv1 + 1, v1))
                if v != sv]
        if len(cols) >= 1 and cols == list(range(cols[0], cols[-1] + 1)):
            box(u0 + 1, cols[0], F + 1, u1 - 1, cols[-1], alc_top, AIR)
            # the alcove's floor, on every column of it that is not the road's: a floor
            # course laid on a lane cell is a lane cell nobody can stand on
            for u in range(u0 + 1, u1):
                for v in cols:
                    if not is_lane(u, v):
                        put(u, v, F, FLOOR)
            alcoves.append((u0 + 1, cols[0]))

    # the signature: where the gate stands over open ground the storey above is carried
    # out over it, and the corbels that carry it are shown
    bracket = rnd.choice([WALL_SLAB, FOOT_SLAB]) + "[type=top]"
    for v in (v0, v1):
        for u in range(u0, u1 + 1):
            if is_lane(u, v):
                put(u, v, HEAD, bracket)

    # ---- the chambers over the arch ---------------------------------------
    def perimeter(y_lo, y_hi, blk):
        box(u0, v0, y_lo, u1, v0, y_hi, blk)
        box(u0, v1, y_lo, u1, v1, y_hi, blk)
        box(u0, v0, y_lo, u0, v1, y_hi, blk)
        box(u1, v0, y_lo, u1, v1, y_hi, blk)

    posts_u = [u0, u1] + ([u0 + (U - 1) // 2] if U >= 7 else [])
    posts_v = [v0, v1] + ([v0 + (V - 1) // 2] if V >= 7 else [])

    for s in range(storeys):
        fy = DECK + s * CH
        box(u0, v0, fy, u1, v1, fy, FLOOR)
        perimeter(fy, fy, TRIM)
        perimeter(fy + 1, fy + CH - 1, WALL)
        box(u0 + 1, v0 + 1, fy + 1, u1 - 1, v1 - 1, fy + CH - 1, AIR)
        for pu in posts_u:
            for pv in posts_v:
                if pu in (u0, u1) or pv in (v0, v1):
                    box(pu, pv, fy + 1, pu, pv, fy + CH - 1, FRAME)
        box(u0, v0, fy + CH, u1, v1, fy + CH, FLOOR)

    # ---- the stairs: ground to wall walk, then flank to flank --------------
    climbs = [(F, DECK)] + [(DECK + s * CH, DECK + (s + 1) * CH)
                            for s in range(storeys - 1)]
    stair_report = []
    prev_side = None
    keep = {}
    treads = set()
    needed = set()
    must_air = []

    def mural_stair():
        """A switchback stair inside one pier of a tower's base, from the guard room
        at its foot to the deck at the crown: flights along the road's axis in two
        lanes against the outer face, each doubling back on the one below it, and
        every one of them laid, cleared and walked by the library's `flight()`."""
        side = rnd.choice(stair_sides)
        lanes = ([v0 + 1, v0 + 2] if side == 0 else [v1 - 1, v1 - 2])
        if not all((pv0 > lv if side == 0 else lv > pv1) for lv in lanes):
            return {"ok": False, "reason": "no pier two lanes thick for a mural stair"}
        r_max = U - 4                      # foot, treads, landing inside u0+1..u1-1
        need = DECK - F
        n_fl = max(1, -(-need // r_max))
        base_r, extra = divmod(need, n_fl)
        runs = [base_r + (1 if q < extra else 0) for q in range(n_fl)]
        du = 1
        foot_u, y = u0 + 1, F
        reps = []
        for k, r in enumerate(runs):
            lane = lanes[k % 2]
            first = foot_u + du
            land_u = foot_u + du * (r + 1)
            if not (u0 + 1 <= first <= u1 - 1 and u0 + 1 <= land_u <= u1 - 1):
                return {"ok": False, "reason": f"flight {k} of {r} does not fit the pier"}
            # the foot: a floor block to stand on and room to stand
            put(foot_u, lane, y, FLOOR)
            box(foot_u, lane, y + 1, foot_u, lane, y + 3, AIR)
            needed.add((foot_u, lane, y))
            must_air.extend([(foot_u, lane, y + 1), (foot_u, lane, y + 2)])
            # the treads' cells and the landing's, opened out of the mass; the library
            # clears the headroom over each and refuses a tread into a wall
            for i in range(r):
                box(first + du * i, lane, y + 1 + i, first + du * i, lane, y + 1 + i, AIR)
                treads.add((first + du * i, lane, y + 1 + i))
            box(land_u, lane, y + r, land_u, lane, y + r, AIR)
            box(land_u, lane, y + r + 1, land_u, lane, y + r + 3, AIR)
            needed.add((land_u, lane, y + r))
            must_air.extend([(land_u, lane, y + r + 1), (land_u, lane, y + r + 2)])
            sx, sz = XZ(first, lane)
            rep_ = b.flight(part["label"], sx, sz, y, y + r, _dirname(along_x, du),
                            mat=voice["footing"])
            reps.append(rep_)
            if not rep_.get("ok"):
                return {"ok": False, "reason": f"flight {k}: {rep_.get('reason')}",
                        "flights": reps}
            # the landing spans both lanes, so the next flight's foot is a step away
            other = lanes[(k + 1) % 2]
            put(land_u, other, y + r, FLOOR)
            box(land_u, other, y + r + 1, land_u, other, y + r + 3, AIR)
            needed.add((land_u, other, y + r))
            must_air.extend([(land_u, other, y + r + 1), (land_u, other, y + r + 2)])
            reserve((y + r - DECK) // CH if y + r >= DECK else -1, land_u, other)
            reserve((y + r - DECK) // CH if y + r >= DECK else -1, land_u, lane)
            foot_u, y, du = land_u, y + r, -du
        return {"ok": True, "flights": reps, "side": side, "lanes": lanes,
                "reason": f"{len(runs)} flights of {runs} in the pier's two lanes"}

    def reserve(st, cu, cv):
        s_ = keep.setdefault(st, set())
        s_.add((cu, cv))
        for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s_.add((cu + dd[0], cv + dd[1]))

    def lay(r, y_lo, y_hi, inside, du, sv, u_start):
        iv = inward(sv)
        u_land = u_start + r * du
        u_foot = u_start - du
        for i in range(r):
            u = u_start + i * du
            box(u, sv, y_lo + 1 + i, u, sv, y_lo + 3 + i, AIR)
            treads.add((u, sv, y_lo + 1 + i))
            needed.add((u, sv, y_lo + i))
            must_air.extend([(u, sv, y_lo + 2 + i), (u, sv, y_lo + 3 + i)])
        box(u_land, sv, y_hi + 1, u_land, sv, y_hi + 2, AIR)
        needed.add((u_land, sv, y_hi))
        must_air.extend([(u_land, sv, y_hi + 1), (u_land, sv, y_hi + 2)])
        if inside:
            # the stairwell is open over the foot of the flight, or a person standing at
            # the bottom of it has the floor above on their head
            box(u_foot, sv, y_lo + 1, u_foot, sv, y_lo + 3, AIR)
            box(u_foot, iv, y_lo + 1, u_foot, iv, y_lo + 2, AIR)
            needed.add((u_foot, sv, y_lo))
            must_air.extend([(u_foot, sv, y_lo + 1), (u_foot, sv, y_lo + 2),
                             (u_foot, sv, y_lo + 3),
                             (u_foot, iv, y_lo + 1), (u_foot, iv, y_lo + 2)])
        box(u_start, iv, y_lo + 1, u_start, iv, y_lo + 2, AIR)
        box(u_land, iv, y_hi + 1, u_land, iv, y_hi + 2, AIR)
        must_air.extend([(u_start, iv, y_lo + 1), (u_start, iv, y_lo + 2),
                         (u_land, iv, y_hi + 1), (u_land, iv, y_hi + 2)])
        reserve((y_hi - DECK) // CH, u_land, iv)
        if y_lo >= DECK:
            reserve((y_lo - DECK) // CH, u_start, iv)
            if inside:
                reserve((y_lo - DECK) // CH, u_foot, iv)
        # a flight below cuts its stairwell through this floor: put back what this one
        # has to stand on, so its foot is on a floor and not a hole
        for (nu, nv, ny) in sorted(needed):
            if ny >= DECK or not is_lane(nu, nv):
                if b.get_block(*cell(nu, nv, ny)) == "air":
                    put(nu, nv, ny, FLOOR)
        sx, sz = XZ(u_start, sv)
        return b.flight(part["label"], sx, sz, y_lo, y_hi,
                        _dirname(along_x, du), mat=voice["footing"])

    for n, (y_lo, y_hi) in enumerate(climbs):
        if n == 0 and tall:
            rep = mural_stair()
            stair_report.append(rep)
            continue
        r = y_hi - y_lo
        off = max(0, (U - (r + 1)) // 2)
        cands = []
        o_in = max(0, (U - (r + 2)) // 2)
        for side in (0, 1):
            for du in (1, -1):
                sv = stair_col(side)
                starts = ([u0 + 1 + o_in, u0 + off] if du > 0
                          else [u1 - 1 - o_in, u1 - off])
                for u_start in starts:
                    u_land = u_start + r * du
                    u_foot = u_start - du
                    if not (u0 <= u_start <= u1 and u0 <= u_land <= u1):
                        continue
                    if any(is_lane(u_start + i * du, sv) and y_lo + 1 + i < DECK
                           for i in range(r)):
                        continue
                    inside = u0 <= u_foot <= u1
                    if not inside:
                        fx, fz = XZ(u_foot, sv)
                        if (b.get_block(fx, y_lo, fz) == "air"
                                or b.get_block(fx, y_lo + 1, fz) != "air"
                                or b.get_block(fx, y_lo + 2, fz) != "air"):
                            continue
                    cands.append((1 if inside else 0, side, du, sv, u_start))
                    break
        if not cands:
            continue
        rnd.shuffle(cands)
        cands.sort(key=lambda c: (-c[0],
                                  0 if (prev_side is None or c[1] != prev_side)
                                  else 1))
        for inside, side, du, sv, u_start in cands:
            rep = lay(r, y_lo, y_hi, inside, du, sv, u_start)
            if rep.get("ok"):
                break
        prev_side = side
        stair_report.append(rep)

    # ---- the wall walk crosses the gate at the deck ------------------------
    mid = u0 + (U - 1) // 2
    order = sorted(range(u0 + 1, u1), key=lambda u: abs(u - mid))
    for v in (v0, v1):
        for cand in order:
            want = [(cand, v, DECK + 1), (cand, v, DECK + 2)]
            if any(c in treads or c in needed for c in want):
                continue
            for c in want:
                box(c[0], c[1], c[2], c[0], c[1], c[2], AIR)
            must_air.extend(want)
            if CH >= 4:
                put(cand, v, DECK + 3, TRIM)
            break

    # ---- the gate itself: leaves across the outer mouth --------------------
    mouth_lane = {u: sum(1 for v in range(pv0, pv1 + 1) if is_lane(u, v))
                  for u in (u0, u1)}
    outer = min((u0, u1), key=lambda u: mouth_lane[u])
    if mouth_lane[outer] == 0:
        gate = b.joinery(voice, "gate")
        gdir = _dirname(along_x, 1 if outer == u1 else -1)
        for v in range(pv0, pv1 + 1):
            put(outer, v, F + 1, gate + "[facing=" + gdir + "]")

    # a light in the gate hall, set against a pier and never in the road itself
    lit = 0
    for v in (pv0, pv1, (pv0 + pv1) // 2):
        for u in range(u0, u1 + 1):
            if lit >= 2 or is_lane(u, v) or u == outer:
                continue
            r = b.fitting("light", *cell(u, v, F + 1), facing=facing,
                          mat=voice["footing"], room="gate hall")
            if r and r.get("ok"):
                lit += 1
                break

    # ---- loops and lights --------------------------------------------------
    def slits(u_or_v, is_u_face, fy, sp):
        if is_u_face:
            xa, za = XZ(u_or_v, v0 + 1)
            xb, zb = XZ(u_or_v, v1 - 1)
        else:
            xa, za = XZ(u0 + 1, u_or_v)
            xb, zb = XZ(u1 - 1, u_or_v)
        return b.openings(xa, fy, za, xb, zb, spacing=sp, width=1, sill=1,
                          head=1 if CH < 4 else 2)

    for s in range(storeys):
        fy = DECK + s * CH
        sp = 2 if s == 0 else 3
        for u in (u0, u1):
            slits(u, True, fy, sp)

    # ---- what a watch is kept with ----------------------------------------
    room = "hall"
    ic0, ic1 = u0 + 1, u1 - 1
    jc0, jc1 = v0 + 1, v1 - 1

    def whole(cells, start):
        if start not in cells:
            return False
        seen, stack = {start}, [start]
        while stack:
            cu_, cv_ = stack.pop()
            for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cu_ + dd[0], cv_ + dd[1])
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(cells)

    for s in range(storeys):
        fy = DECK + s * CH
        free = set((u, v) for u in range(ic0, ic1 + 1) for v in range(jc0, jc1 + 1))
        held = [c for c in keep.get(s, ()) if c in free]
        start = held[0] if held else sorted(free)[0]
        edge = [c for c in free if c[0] in (ic0, ic1) or c[1] in (jc0, jc1)]
        rnd.shuffle(edge)
        edge.sort(key=lambda c: 0 if (c[0] in (ic0, ic1) and c[1] in (jc0, jc1))
                  else 1)
        # no bench: a bench is a stair block, and a stair block with a table behind it
        # reads to the walk model as a tread facing down-slope
        kinds = ["store", "table", "shelf", "bookshelf"]
        rnd.shuffle(kinds)
        kinds = ["light"] + kinds
        want = 3 if len(free) > 6 else 2
        for k in kinds:
            if want <= 0:
                break
            for sp_ in list(edge):
                if sp_ in held or sp_ not in free:
                    continue
                if not whole(free - {sp_}, start):
                    continue
                # a bench is a stair block, and a stair block beside a stairwell reads
                # as a tread facing down it: no seat next to a hole in the floor
                if k == "bench" and any(
                        b.get_block(*cell(sp_[0] + dd[0], sp_[1] + dd[1], fy)) == "air"
                        for dd in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    continue
                r = b.fitting(k, *cell(sp_[0], sp_[1], fy + 1), facing=facing,
                              mat=voice["footing"] if k == "light"
                              else voice["frame"], room=room)
                if r and r.get("ok"):
                    free.discard(sp_)
                    edge.remove(sp_)
                    want -= 1
                    break

    # ---- the crown ---------------------------------------------------------
    if crest:
        perimeter(eave_y - crest + 1, eave_y, WALL)
        for u in range(u0, u1 + 1, 2):
            put(u, v0, eave_y, TRIM)
            put(u, v1, eave_y, TRIM)
    style = "gable" if crown == "gable" else "hip"
    tiers = 3 if crown == "pavilion" else rnd.choice([2, 2, 3])
    pitch = (3, 1) if crown == "pavilion" else rnd.choice([(2, 1), (2, 1), (3, 1)])
    kw = {"style": style, "axis": ("x" if along_x else "z"),
          "pitch": pitch, "overhang": 0, "tiers": tiers}
    rspec = part.get("roof")
    if isinstance(rspec, dict):
        for k in ("profile", "ends", "eave"):
            if rspec.get(k):
                kw[k] = rspec[k]
    b.roof(x0, z0, x1, z1, eave_y, voice["roof"], **kw)

    # nothing laid since may stand in the way up: the stair keeps its air
    for c in must_air:
        if (c not in treads and c not in needed
                and u0 <= c[0] <= u1 and v0 <= c[1] <= v1):
            box(c[0], c[1], c[2], c[0], c[1], c[2], AIR)

    return {"deck": DECK, "eave": eave_y, "stairs": stair_report,
            "tower": bool(tall), "crown": crown}
