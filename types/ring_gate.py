"""A city gate: a wide arched passage through the wall body, and a timber gate tower
standing on the wall's top over it.

The city round (axis views, two readers). Every ring gate read as the same narrow
stacked tower -- five wide, a flat black cap, a door-sized hole -- and the axis road,
eleven wide, ended at what looked like a blank wall. A Chinese city gate is two things
(the Pingyao and Xi'an references): a **gate base** of the wall's own masonry, deeper
than the wall and as high as its walk, pierced by a vaulted passage as wide as the
road; and a **gate tower** (chenglou) of posts, railings and stacked eaves standing on
the wall's top over it. This type builds both, in that order:

  base     the wall's body filled solid across the whole pad, `D` deep (the wall's
           width and a projection either side), up to the wall's walk; a plinth, a
           string course and a parapet, crenellated outside and low inside, the walk
           carried straight across the top so the wall's walk continues over the gate.
  passage  the road's width at least (`passage`, or the lane's own width where a lane
           runs through the pad), vaulted: vertical jambs to a springing course and a
           round head stepping in on inverted stairs, a voussoir ring and keystone on
           both faces, the head about six to nine over the road and more in a tall
           wall; paved at the pad's floor right through and past both faces, so the
           road and its ramps meet it level.
  tower    on the deck: a colonnade of frame posts on bays with railings between them,
           open at both ends so the walk passes through; over it a broad eave, then
           each further tier a storey of posts and panels stepped in under its own
           eave, and the crown -- hip, pavilion or gable -- in the voice's roof role,
           with a ridge course, raised ridge ends and lifted corners.
  way up   a switchback mural stair in one pier where a pier is thick enough, from a
           doorway in the passage to the deck.

**Rank shows.** `storeys` (1-3) is the tower's number of eave tiers; a great wall
(`edge` height from `TIERS_FROM`) always gets three and the widest passage; a low
precinct wall gets a smaller tower (fewer bays) that keeps every ornament.

**A gate is sized by its wall.** Siting hands this type its wall in `part["edge"]`
(height, type, width, floor) and sizes the pad from the height inside `NEEDS`; the
walk's level is the wall type's own (a `wall` walks two under its height, a
`great_wall` on it), read off the wall beside the pad where the wall is already
built. With no wall named it is the same gate, standing on its own.
"""

import math
import random

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"
KIND = "point"
PASSAGE = True
#: **What family of part this builds**, said outright. `capability._named_for` reads a
#: family off the committed name and `ring_gate` neither is `gate` nor starts with
#: `gate_`, while the ring layout's own chooser matched it on the substring -- so the
#: capability record said the place's gate was `gate_tower` and the place had a
#: `ring_gate` in it. Two rules, one library, two answers: a gate a ring wall is passed
#: through.
FAMILY = "gate"

PARAMS = {
    "storeys": ("int", 1, 3),
    "crown": ("choice", ["gable", "hip", "pavilion"]),
    # the passage's width; where a lane runs through the pad the lane's width is the
    # least it is. Unset: 5, or 7 in a wall from `WIDE_FROM`.
    "passage": ("int", 3, 11),
}

#: **The pad grew**, the city round: a passage as wide as an eleven-wide road with a
#: pier either side does not fit in the five a twelve-high wall used to get
#: (`Builder.point_pad` is a third of the height), so the band's floor is the pad a gate
#: needs whatever its wall. Measured by `scripts/type_needs.py`.
NEEDS = {
    "footprint": (15, 15, 16, 16),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

AIR = "air"

#: The arch through the base, over the road: this many blocks of headroom at the least,
#: and in a tall wall a share of its height -- a passage six high in a pier of forty-
#: eight is a mouse-hole at the foot of a cliff.
ARCH_HEADROOM = 6
ARCH_HEADROOM_OF_WALL = 0.3
#: ...and never more than this many times its span, or it is a slot.
ARCH_TALLEST_OF_SPAN = 2
#: The span from the wall's height, where no lane and no param says it.
ARCH_SPAN_OF_WALL = 0.25
#: A wall this high gets the wide passage by default, and one from `TIERS_FROM` a three-
#: eave tower whatever its `storeys` says.
WIDE_FROM = 20
TIERS_FROM = 30
#: The deck may stand this many courses over the wall's walk (with a step at each end)
#: so a low wall's passage still reads as a city gate.
DECK_OVER_WALK = 1

#: A pent eave's pitch, rise over run, where the voice's own is shallower.
SKIRT_PITCH = (1, 1)

RIDGE_ENDS = {"hip": ("hip", "hip"), "gable": ("gable", "gable"),
              "pavilion": ("hip", "hip")}
PITCH = {"hip": 0.6, "pavilion": 0.75, "gable": 0.55}


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _dirname(along_x, step):
    if along_x:
        return "east" if step > 0 else "west"
    return "south" if step > 0 else "north"


# ---------------------------------------------------------------- roofs After
# `ceremonial_hall`: the voice owns the roof's character (its eave and its profile, off
# `roof_spec`); this type decides its rank (which ends, how many eaves) and draws it on
# the library's own `roof()`.

def _raw(b):
    return getattr(b, "_b", b)


def _voice_roof(b):
    spec = getattr(b, "roof_spec", None) or {}
    prof = spec.get("profile")
    prof = [tuple(p) for p in prof] if prof else None
    return prof, spec.get("eave")


def _skirt_pitch(b):
    prof, _eave = _voice_roof(b)
    if not prof:
        return (1, 2)
    r, run = prof[0]
    r, run = max(1, int(r)), max(1, int(run))
    if r > run:
        r, run = 1, 1
    return (r, run)


class Frame:
    """u runs the way the gate faces (the road), v along the wall line."""

    def __init__(self, b, along_x):
        self.b = b
        self.along_x = along_x

    def xz(self, u, v):
        return (u, v) if self.along_x else (v, u)

    def rect(self, ua, va, ub, vb):
        xa, za = self.xz(ua, va)
        xb, zb = self.xz(ub, vb)
        return min(xa, xb), min(za, zb), max(xa, xb), max(za, zb)

    def put(self, u, v, y, blk):
        x, z = self.xz(u, v)
        self.b.place_block(x, y, z, blk)

    def box(self, ua, va, ya, ub, vb, yb, blk):
        x0, z0, x1, z1 = self.rect(ua, va, ub, vb)
        self.b.place_cuboid(x0, min(ya, yb), z0, x1, max(ya, yb), z1, blk)

    def get(self, u, v, y):
        x, z = self.xz(u, v)
        return self.b.get_block(x, y, z)

    def solid(self, u, v, y):
        return self.get(u, v, y).split("[")[0] not in ("air", "cave_air", "void_air")

    def axis_along(self, along_u):
        # roof(): axis "z" is a ridge running along x
        ridge_x = along_u == self.along_x
        return "z" if ridge_x else "x"

    def dir(self, du, dv):
        if du:
            return _dirname(self.along_x, du)
        return _dirname(not self.along_x, dv)


def _roof(b, G, rect, y, over, kind, band=None, cap=None):
    """One eave tier over `rect` (u0, v0, u1, v1): a skirt stopped `band` courses in
    from its eave, or the crown. Returns the top y."""
    prof, eave = _voice_roof(b)
    ends = RIDGE_ENDS[kind]
    style = "gable" if kind == "gable" else "hip"
    x0, z0, x1, z1 = G.rect(*rect)
    du = rect[2] - rect[0]
    dv = rect[3] - rect[1]
    kw = {"style": style, "axis": G.axis_along(du >= dv), "overhang": over, "ends": ends}
    if eave:
        kw["eave"] = "straight" if (eave == "upturned" and over < 2) else eave
    if band is not None:
        # a gate tower's pent eaves are carried at a full course a column, or from the
        # street they read as flat rings: the voice's eave pitch where it is steeper
        er, erun = _skirt_pitch(b)
        kw["profile"] = [(er, erun) if er >= erun else SKIRT_PITCH]
        kw["_stop_out"] = band
        kw["ends"] = RIDGE_ENDS["hip"]
        kw["style"] = "hip"
        return _raw(b).roof(x0, z0, x1, z1, y, b.voice["roof"], **kw)
    n = (min(du, dv) + 1 + 2 * over + 1) // 2
    er, erun = _skirt_pitch(b)
    run1 = max(erun, (n // 3) // erun * erun)
    r1 = run1 * er // erun
    rise = max(2, int(round(n * (cap or 0.7))))
    if n <= 4:
        kw["profile"] = [(1, 1)]
    else:
        kw["profile"] = [(r1, run1), (max(1, rise - r1), max(1, n - run1 - 1))]
    return _raw(b).roof(x0, z0, x1, z1, y, b.voice["roof"], **kw)


def _top_at(G, u, v, lo, hi):
    for y in range(hi, lo - 1, -1):
        if G.solid(u, v, y):
            return y
    return None


def _corners(b, G, rect, over, y_e):
    """The four eave corners carried up a course: the lifted corner."""
    u0, v0, u1, v1 = rect[0] - over, rect[1] - over, rect[2] + over, rect[3] + over
    slab = b.block(b.voice["roof"], "slab")
    for (u, v) in ((u0, v0), (u1, v0), (u0, v1), (u1, v1)):
        t = _top_at(G, u, v, y_e - 1, y_e + 3)
        if t is not None:
            G.put(u, v, t + 1, slab)


def _ridge(b, G, rect, over, ridge_y):
    """A course along the ridge in the trim, raised ends, a finial on a pyramid."""
    trim = b.block(b.voice["trim"])
    acc = b.block(b.voice["trim"], "accent")
    u0, v0, u1, v1 = rect[0] - over, rect[1] - over, rect[2] + over, rect[3] + over
    m = over + 1
    cells = [(u, v) for u in range(u0 + m, u1 - m + 1) for v in range(v0 + m, v1 - m + 1)
             if G.solid(u, v, ridge_y) and not G.solid(u, v, ridge_y + 1)]
    if not cells:
        return 0
    for (u, v) in cells:
        G.put(u, v, ridge_y + 1, trim)
    along_u = (u1 - u0) >= (v1 - v0)
    key = (lambda c: c[0]) if along_u else (lambda c: c[1])
    lo, hi = min(cells, key=key), max(cells, key=key)
    if key(hi) - key(lo) >= 2:
        for (u, v) in (lo, hi):
            G.put(u, v, ridge_y + 2, acc)
    else:
        cu = sorted(c[0] for c in cells)[len(cells) // 2]
        cv = sorted(c[1] for c in cells)[len(cells) // 2]
        G.put(cu, cv, ridge_y + 2, b.block(b.voice["trim"], "slab"))
    return len(cells)


# ---------------------------------------------------------------- the gate

def build(b, part, seed, **params):
    rnd = random.Random((seed * 6364136223846793005 + 1442695040888963407) % (1 << 61))
    voice = part["voice"]
    storeys = _clamp(int(params.get("storeys", 2) or 2), 1, 3)
    crown_kind = params.get("crown", "hip")
    if crown_kind not in RIDGE_ENDS:
        crown_kind = "hip"

    WALL = b.block(voice["wall"], "full")
    FOOT = b.block(voice["footing"], "full")
    POST = b.block(voice["frame"], "post")
    PLANK = b.block(voice["frame"], "full")
    PLANK_SLAB = b.block(voice["frame"], "slab")
    TRIM = b.block(voice["trim"], "full")
    ACC = b.block(voice["wall"], "accent")
    FLOOR = b.block(voice["floor"], "full")
    WALL_SLAB = b.block(voice["wall"], "slab")
    TRIM_SLAB = b.block(voice["trim"], "slab")
    FLOOR_SLAB = b.block(voice["floor"], "slab")
    FENCE = b.joinery(voice, "fence")
    LATTICE = b.joinery(voice, "trapdoor")

    x0, x1 = sorted((part["x0"], part["x1"]))
    z0, z1 = sorted((part["z0"], part["z1"]))
    F = int(part["floor_y"])
    facing = part.get("facing") or "north"
    at = part.get("at") or [(x0 + x1) // 2, (z0 + z1) // 2]
    along_x = facing in ("east", "west")
    G = Frame(b, along_x)
    put, box, get = G.put, G.box, G.get

    if along_x:
        u0, u1, v0, v1 = x0, x1, z0, z1
        uc, vc = int(at[0]), int(at[-1])
    else:
        u0, u1, v0, v1 = z0, z1, x0, x1
        uc, vc = int(at[-1]), int(at[0])
    U = u1 - u0 + 1
    V = v1 - v0 + 1
    uc = _clamp(uc, u0, u1)
    vc = _clamp(vc, v0, v1)
    out_du = 1 if facing in ("east", "south") else -1

    lane = set()
    for xx in range(x0, x1 + 1):
        for zz in range(z0, z1 + 1):
            nl = b.nearest_lane(xx, zz)
            if nl and int(nl["distance"]) == 0:
                lane.add((xx, zz))

    def is_lane(u, v):
        return G.xz(u, v) in lane

    # ---- the wall this gate stands in ----------------------------------------
    edge = part.get("edge") or {}
    H = int(edge.get("height") or 0)
    wtype = str(edge.get("type") or "")
    w = max(1, int(edge.get("width") or 3))
    wall_floor = int(edge["floor_y"]) if edge.get("floor_y") is not None else F
    if H:
        if wtype == "wall":
            walk_h = _clamp(H, 3, 20) - 2
        elif wtype == "great_wall":
            walk_h = _clamp(H, 24, 48)
        else:
            walk_h = H
    else:
        walk_h = _clamp(U // 2 + 2, 4, 8)
    WALK = max(F + 3, wall_floor + walk_h)
    # the walk as built beside the pad, where the wall is already standing: the lowest
    # top across its band (a parapet is higher than the walk it guards)
    bw = (w - 1) // 2 + 1

    def walk_beside(v):
        tops = []
        for u in range(uc - bw, uc + bw + 1):
            t = _top_at(G, u, v, WALK - 5, WALK + 4)
            if t is not None:
                tops.append(t)
        return min(tops) if len(tops) >= 2 else None
    seen = [walk_beside(v0 - 1), walk_beside(v1 + 1)] if H else [None, None]
    walks = [s if s is not None else WALK for s in seen]

    # ---- proportions ------------------------------------------------------------- the
    # base: the wall's width and a projection either side, inside the pad
    if U >= 13:
        D = min(U - 2, max(11, w + 4))
    else:
        D = U
    ua = _clamp(uc - D // 2, u0, u1 - D + 1)
    ub = ua + D - 1
    u_out = ub if out_du > 0 else ua
    u_in = ua if out_du > 0 else ub

    # the passage: the road's width at least
    pier_min = 2 if V >= 9 else 1
    road_v = sorted({v for v in range(v0 + 1, v1)
                     if is_lane(u0, v) and is_lane(u1, v)})
    if not road_v:
        road_v = sorted({v for v in range(v0 + 1, v1) if is_lane((u0 + u1) // 2, v)})
    want = params.get("passage")
    if want:
        want = int(want)
    elif H:
        want = max(7 if H >= WIDE_FROM else 5, int(round(H * ARCH_SPAN_OF_WALL)))
    else:
        want = 5 if V >= 9 else 3
    centre = vc
    if road_v:
        want = max(want, road_v[-1] - road_v[0] + 1)
        centre = (road_v[0] + road_v[-1] + 1) // 2
    P = _clamp(want, 1, max(1, V - 2 * pier_min))
    pv0 = _clamp(centre - P // 2, v0 + pier_min, v1 - pier_min - P + 1)
    pv1 = pv0 + P - 1

    # the head and the deck
    if H:
        A_want = max(ARCH_HEADROOM, int(round(H * ARCH_HEADROOM_OF_WALL)) + 3)
        A_want = min(A_want, ARCH_TALLEST_OF_SPAN * P + 2)
    else:
        A_want = max(3, walk_h - 2)
    lo_walk, hi_walk = min(walks), max(walks)
    DECK = min(max(hi_walk, F + A_want + 2), lo_walk + DECK_OVER_WALK)
    DECK = max(DECK, F + 4)
    A = max(3, min(A_want, DECK - F - 2))

    # the vault: jambs to the springing, then a round head
    pc = (pv0 + pv1) / 2.0
    half = max(0.5, P / 2.0)
    rise = max(0, min((P + 1) // 2, A - 4))
    S = A - rise
    head = {}
    for v in range(pv0, pv1 + 1):
        t = (v - pc) / half
        head[v] = max(3, S + int(round(rise * math.sqrt(max(0.0, 1.0 - t * t)))))

    # ---- the base --------------------------------------------------------------
    for u in range(ua, ub + 1):
        for v in range(v0, v1 + 1):
            if pv0 <= v <= pv1:
                box(u, v, F + head[v] + 1, u, v, DECK - 1, WALL)
                box(u, v, F + 1, u, v, F + head[v], AIR)
                if not is_lane(u, v):
                    put(u, v, F, FLOOR)
            else:
                box(u, v, F, u, v, DECK - 1, WALL)
    box(ua, v0, DECK, ub, v1, DECK, FLOOR)
    # the plinth and the string course round the faces
    for u in (ua, ub):
        for v in range(v0, v1 + 1):
            if not (pv0 <= v <= pv1):
                box(u, v, F, u, v, F + 1, FOOT)
            if DECK - 1 > F + head.get(v, 0) + 2:
                put(u, v, DECK - 1, TRIM)
    for v in (v0, v1):
        for u in range(ua, ub + 1):
            if abs(u - uc) > bw:
                box(u, v, F, u, v, F + 1, FOOT)
                put(u, v, DECK - 1, TRIM)
    # the road paved through, and past both faces to the pad's edge (a lane's own
    # surface is the town's and is left as it is; its headroom is kept)
    for u in range(u0, u1 + 1):
        for v in range(pv0 - 1, pv1 + 2):
            if not (ua <= u <= ub):
                if not is_lane(u, v):
                    put(u, v, F, FLOOR)
                box(u, v, F + 1, u, v, F + 3, AIR)

    # the head's steps in, as half courses, the length of the vault: the lower of two
    # neighbouring columns is closed by a slab in the top of the course over it
    for v in range(pv0, pv1 + 1):
        for n in (v - 1, v + 1):
            if n in head and head[n] < head[v]:
                for u in range(ua, ub + 1):
                    put(u, n, F + head[n] + 1, WALL_SLAB + "[type=top]")
    # the faces of the arch: jambs, a voussoir ring, a keystone, a tablet over it
    for u in (ua, ub):
        for v in (pv0 - 1, pv1 + 1):
            box(u, v, F + 2, u, v, F + S, TRIM)
        for v in range(pv0 - 1, pv1 + 2):
            hv = head.get(v, S)
            top = F + hv + 1
            nb = max(head.get(v - 1, hv), head.get(v + 1, hv))
            for y in range(top, max(top, F + nb + 1) + 1):
                if y < DECK - 1 and not (v in head and y <= F + hv):
                    put(u, v, y, TRIM)
        kv = int(round(pc))
        if F + A + 1 < DECK - 1:
            put(u, kv, F + A + 1, ACC)
        if F + A + 4 < DECK - 1 and P >= 5:
            box(u, kv - 1, F + A + 3, u, kv + 1, F + A + 3, ACC)

    # ---- the wall joined to the gate ------------------------------------------------
    # Where the wall is already standing round the pad, every column of it the pad holds
    # is carried on to the base: a wall that turns off the gate's straight run inside
    # the pad (a 45-degree run, or a run shorter than the pad) would otherwise stop at
    # the pad's edge with a gap between it and the base. Each such column is the wall's
    # own section -- the walk's level or the parapet's -- bridged along the road's axis
    # to the base's face, and where it meets the deck the parapet opens.
    in_base = lambda u, v: ua <= u <= ub and v0 <= v <= v1          # noqa: E731
    lo_top = min(walks)
    joins = {}
    bridges = {}
    mended = 0
    if H:
        # siting cuts the pad's one-column ledge to a person's headroom, and where the
        # wall stands on that ledge the cut is a slot under the wall beside the gate:
        # put the wall's courses back wherever masonry stands over the cut
        for u in range(u0 - 1, u1 + 2):
            for v in range(v0 - 1, v1 + 2):
                if u0 <= u <= u1 and v0 <= v <= v1:
                    continue
                if is_lane(u, v) or not all(G.solid(u, v, y) for y in
                                            range(F + 4, lo_top - 1)):
                    continue
                t = _top_at(G, u, v, F + 4, DECK + 3)
                if t is None or t < lo_top - 2:
                    continue
                for y in range(F + 1, F + 4):
                    if not G.solid(u, v, y):
                        put(u, v, y, WALL)
                        mended += 1
        rim = []
        for u in range(u0, u1 + 1):
            rim.append((u, v0, 0, -1))
            rim.append((u, v1, 0, 1))
        for v in range(v0, v1 + 1):
            rim.append((u0, v, -1, 0))
            rim.append((u1, v, 1, 0))
        for (u, v, du, dv) in rim:
            ou, ov = u + du, v + dv
            t = _top_at(G, ou, ov, F + 2, DECK + 3)
            if t is None or t < lo_top - 4:
                continue
            if not (G.solid(ou, ov, F + 2) and G.solid(ou, ov, (F + t) // 2)):
                continue
            if in_base(u, v):
                joins[(u, v)] = min(joins.get((u, v), t), t)
                continue
            step = 1 if u < ua else -1
            uu = u
            while not (ua <= uu <= ub):
                bridges[(uu, v)] = max(bridges.get((uu, v), t), t)
                uu += step
            joins[(uu, v)] = min(joins.get((uu, v), t), t)
        for (u, v), t in bridges.items():
            box(u, v, F, u, v, t - 1, WALL)
            put(u, v, t, FLOOR if t <= lo_top else WALL)
            if t <= lo_top:
                box(u, v, t + 1, u, v, t + 3, AIR)
    open_at = set()
    for (ju, jv), t in sorted(joins.items()):
        if t > DECK:
            continue
        # the walk arrives here: this cell and its neighbours on the base's rim open, so
        # a walk that meets the base at a corner can turn onto the deck
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                cu, cv = ju + du, jv + dv
                if ua <= cu <= ub and v0 <= cv <= v1 and \
                        (cu in (ua, ub) or cv in (v0, v1)):
                    open_at.add((cu, cv))
                    joins.setdefault((cu, cv), t)
    ends_seen = {v for (u, v) in joins if v in (v0, v1)}

    # ---- the deck: parapets, the walk carried across ------------------------------
    for v in range(v0, v1 + 1):
        if (u_out, v) not in open_at:
            put(u_out, v, DECK + 1, WALL)
            if (v - v0) % 2 == 0:
                put(u_out, v, DECK + 2, WALL)
        if (u_in, v) not in open_at:
            put(u_in, v, DECK + 1, WALL)
            put(u_in, v, DECK + 2, TRIM_SLAB)
    for v in (v0, v1):
        for u in range(ua + 1, ub):
            walked = ((u, v) in open_at) if v in ends_seen else abs(u - uc) <= bw
            if not walked:
                put(u, v, DECK + 1, WALL)
    # a half step up from the walk where the deck stands over it
    for (u, v) in sorted(open_at):
        if DECK - joins[(u, v)] >= 1:
            put(u, v, DECK, FLOOR_SLAB + "[type=bottom]")
    for side, v in ((0, v0), (1, v1)):
        if v not in ends_seen and DECK - walks[side] >= 1:
            for u in range(max(ua + 1, uc - bw), min(ub - 1, uc + bw) + 1):
                put(u, v, DECK, FLOOR_SLAB + "[type=bottom]")

    # ---- the way up: a mural stair in one pier -------------------------------------
    must_air = []
    stair = {"ok": False, "reason": "no pier four thick for a mural stair"}
    sides = []
    if pv0 - v0 >= 4:
        sides.append(0)
    if v1 - pv1 >= 4:
        sides.append(1)
    if sides and D >= 7:
        stair = _mural_stair(b, G, part, voice, rnd.choice(sides), ua, ub, v0, v1,
                             pv0, pv1, F, DECK, FLOOR, AIR, must_air, along_x)

    # ---- the tower -----------------------------------------------------------------
    tiers = storeys if H < TIERS_FROM or not H else 3
    small = bool(H) and H < 12
    m_v = 2 if not small else 3
    R = (ua + 1, v0 + m_v, ub - 1, v1 - m_v)
    grand = bool(H) and H >= TIERS_FROM
    tower = _tower(b, G, R, DECK, tiers, crown_kind, small, grand, rnd,
                   POST, PLANK, PLANK_SLAB, TRIM, FENCE, LATTICE, uc, bw)

    # lanterns on the inner parapet either side of the tower
    lit = 0
    for v in (R[1] - 1, R[3] + 1):
        if (u_in, v) in open_at or not (v0 <= v <= v1):
            continue
        lx, lz = G.xz(u_in, v)
        put(u_in, v, DECK + 2, AIR)
        r = b.fitting("light", lx, DECK + 2, lz, facing=facing, mat=voice["footing"],
                      room="gate deck")
        if r and r.get("ok"):
            lit += 1

    # nothing laid since stands in the way up
    for (u, v, y) in must_air:
        if G.solid(u, v, y):
            put(u, v, y, AIR)

    return {"deck": DECK, "eave": tower.get("eave"), "stairs": [stair],
            "tower": True, "crown": crown_kind, "crown_y": WALK if H else None,
            "passage": P, "arch": A, "base_depth": D, "walk": walks,
            "tiers": tower.get("tiers"), "ridge": tower.get("ridge"),
            "pad": [U, V], "joins": len(open_at), "bridges": len(bridges),
            "mended": mended}


def _mural_stair(b, G, part, voice, side, ua, ub, v0, v1, pv0, pv1, F, DECK, FLOOR,
                 AIR, must_air, along_x):
    """A switchback stair in one pier, from a doorway in the passage to the deck:
    flights along the road's axis in two lanes, each doubling back on the one below,
    every one laid, cleared and walked by the library's `flight()`."""
    put, box = G.put, G.box
    lanes = [v0 + 1, v0 + 2] if side == 0 else [v1 - 1, v1 - 2]
    # the doorway: every column of the pier between the stair's inner lane and the
    # passage, opened a person's height
    door = list(range(v0 + 3, pv0)) if side == 0 else list(range(pv1 + 1, v1 - 2))
    r_max = (ub - 1) - (ua + 1) - 1
    need = DECK - F
    if r_max < 2:
        return {"ok": False, "reason": "the pier is too short for a flight"}
    n_fl = max(1, -(-need // r_max))
    base_r, extra = divmod(need, n_fl)
    runs = [base_r + (1 if q < extra else 0) for q in range(n_fl)]
    # the flights alternate; start so the last lands at the end the walk is not on
    du = 1 if n_fl % 2 == 1 else -1
    foot_u = ua + 1 if du > 0 else ub - 1
    # the doorway from the passage into the foot of the stair
    for dv in door:
        box(foot_u, dv, F + 1, foot_u, dv, F + 2, AIR)
        put(foot_u, dv, F, FLOOR)
        must_air.extend([(foot_u, dv, F + 1), (foot_u, dv, F + 2)])
    box(foot_u, lanes[1], F + 1, foot_u, lanes[1], F + 3, AIR)
    put(foot_u, lanes[1], F, FLOOR)
    must_air.extend([(foot_u, lanes[1], F + 1), (foot_u, lanes[1], F + 2)])
    y = F
    reps = []
    for k, r in enumerate(runs):
        ln = lanes[(k + 1) % 2]
        first = foot_u + du
        land_u = foot_u + du * (r + 1)
        if not (ua + 1 <= first <= ub - 1 and ua + 1 <= land_u <= ub - 1):
            return {"ok": False, "reason": f"flight {k} of {r} does not fit the pier"}
        put(foot_u, ln, y, FLOOR)
        box(foot_u, ln, y + 1, foot_u, ln, y + 3, AIR)
        must_air.extend([(foot_u, ln, y + 1), (foot_u, ln, y + 2)])
        for i in range(r):
            box(first + du * i, ln, y + 1 + i, first + du * i, ln, y + 1 + i, AIR)
        box(land_u, ln, y + r, land_u, ln, y + r + 3, AIR)
        must_air.extend([(land_u, ln, y + r + 1), (land_u, ln, y + r + 2)])
        sx, sz = G.xz(first, ln)
        rep = b.flight(part["label"], sx, sz, y, y + r, _dirname(along_x, du),
                       mat=voice["footing"])
        reps.append(rep)
        if not rep.get("ok"):
            return {"ok": False, "reason": f"flight {k}: {rep.get('reason')}",
                    "flights": reps}
        other = lanes[k % 2]
        put(land_u, other, y + r, FLOOR)
        box(land_u, other, y + r + 1, land_u, other, y + r + 3, AIR)
        must_air.extend([(land_u, other, y + r + 1), (land_u, other, y + r + 2)])
        foot_u, y, du = land_u, y + r, -du
    return {"ok": True, "flights": reps, "side": side, "lanes": lanes,
            "reason": f"{len(runs)} flights of {runs} in the pier's two lanes"}


def _tower(b, G, R, DECK, tiers, kind, small, grand, rnd, POST, PLANK, PLANK_SLAB, TRIM,
           FENCE, LATTICE, uc, bw):
    """The gate tower on the deck over `R` (u0, v0, u1, v1): a colonnade open at both
    ends under a broad eave, then a storey stepped in under its own eave per tier, and
    the crown."""
    put, box = G.put, G.box
    ra, rv0, rb, rv1 = R
    if rb - ra < 2 or rv1 - rv0 < 2:
        return {"tiers": 0}
    CH0 = 4 if small else (6 if grand else 5)
    bay = 3 if min(rb - ra, rv1 - rv0) >= 6 else 2
    # the colonnade: posts on the bays round the ring
    posts = set()
    for v in range(rv0, rv1 + 1):
        if (v - rv0) % bay == 0 or v == rv1:
            posts.add((ra, v))
            posts.add((rb, v))
    for u in range(ra, rb + 1):
        if u in (ra, rb) or ((u - ra) % bay == 0 and abs(u - uc) > bw):
            posts.add((u, rv0))
            posts.add((u, rv1))
    top0 = DECK + CH0
    for (u, v) in posts:
        box(u, v, DECK + 1, u, v, top0, POST)
    # railings between the posts along both long faces, and on the ends outside the walk
    for u in (ra, rb):
        for v in range(rv0 + 1, rv1):
            if (u, v) not in posts:
                put(u, v, DECK + 1, FENCE)
    for v in (rv0, rv1):
        for u in range(ra + 1, rb):
            if (u, v) not in posts and abs(u - uc) > bw:
                put(u, v, DECK + 1, FENCE)
    # the architrave ring over the posts, and a lattice frieze under it on the faces
    ring = [(u, v) for u in range(ra, rb + 1) for v in range(rv0, rv1 + 1)
            if u in (ra, rb) or v in (rv0, rv1)]
    for (u, v) in ring:
        put(u, v, top0 + 1, PLANK)
    eave = top0 + 2
    rect = R
    eaves = []
    ridge = None
    ov = 1 if small else 2
    for k in range(tiers):
        last = (k == tiers - 1)
        inset = 1
        nxt = (rect[0] + inset, rect[1] + inset, rect[2] - inset, rect[3] - inset)
        if not last and (nxt[2] - nxt[0] < 2 or nxt[3] - nxt[1] < 2):
            last = True
        if not last:
            band = ov + inset - 1
            _roof(b, G, rect, eave, ov, "hip", band=band)
            eaves.append(eave)
            s_top = _top_at(G, nxt[0] - 1, (nxt[1] + nxt[3]) // 2, eave, eave + 12)
            s_top = eave + 1 if s_top is None else s_top
            # the storey above: posts at its corners carried down to the deck inside the
            # hall, walls of planks between them, lattice windows, a frieze
            wall_lo = eave
            h = 3 if small else (5 if grand else 4)
            wall_hi = s_top + h
            for (u, v) in ((nxt[0], nxt[1]), (nxt[2], nxt[1]),
                           (nxt[0], nxt[3]), (nxt[2], nxt[3])):
                box(u, v, DECK + 1 if k == 0 else wall_lo, u, v, wall_hi, POST)
            for u in range(nxt[0], nxt[2] + 1):
                for v in range(nxt[1], nxt[3] + 1):
                    if not (u in (nxt[0], nxt[2]) or v in (nxt[1], nxt[3])):
                        continue
                    if (u in (nxt[0], nxt[2])) and (v in (nxt[1], nxt[3])):
                        continue
                    box(u, v, wall_lo, u, v, wall_hi, PLANK)
            # the windows: a lattice panel in every other bay, facing out
            for u in range(nxt[0] + 1, nxt[2]):
                if (u - nxt[0]) % 2 == 1:
                    for v, dv in ((nxt[1], -1), (nxt[3], 1)):
                        put(u, v, s_top + 2, LATTICE +
                            f"[facing={G.dir(0, dv)},half=bottom,open=true]")
            for v in range(nxt[1] + 1, nxt[3]):
                if (v - nxt[1]) % 2 == 1:
                    for u, du in ((nxt[0], -1), (nxt[2], 1)):
                        put(u, v, s_top + 2, LATTICE +
                            f"[facing={G.dir(du, 0)},half=bottom,open=true]")
            for u in range(nxt[0], nxt[2] + 1):
                for v in (nxt[1], nxt[3]):
                    put(u, v, wall_hi + 1, TRIM)
            for v in range(nxt[1], nxt[3] + 1):
                for u in (nxt[0], nxt[2]):
                    put(u, v, wall_hi + 1, TRIM)
            rect = nxt
            eave = wall_hi + 2
            ov = max(1, ov)
            continue
        crown_rect = rect
        if kind == "pavilion":
            # the pyramid stands on a square
            su, sv = rect[2] - rect[0], rect[3] - rect[1]
            s = min(su, sv)
            cu, cv = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
            crown_rect = (cu - s // 2, cv - s // 2, cu - s // 2 + s, cv - s // 2 + s)
            if crown_rect != rect:
                # the walls round the ends the square does not cover, capped
                box(rect[0], rect[1], eave, rect[2], rect[3], eave, PLANK)
        ridge = _roof(b, G, crown_rect, eave, ov + (0 if small else 1), kind,
                      cap=PITCH[kind])
        if _voice_roof(b)[1] in ("upturned", "flared"):
            _corners(b, G, crown_rect, ov + (0 if small else 1), eave)
        eaves.append(eave)
        _ridge(b, G, crown_rect, ov + (0 if small else 1), ridge)
        break
    return {"tiers": len(eaves), "eave": eaves, "ridge": ridge}
