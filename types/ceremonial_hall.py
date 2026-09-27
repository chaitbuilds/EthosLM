"""A ceremonial hall: the hall a palace axis is made of, ranked by what it is for.

A type, not a building. One file realizes the halls of an axial palace -- the gate hall
the axis passes through, the principal hall on its high terrace, the rear hall, the side
halls that flank the courts, the square pavilion, and the small gate hall of an elite
compound -- and tells them apart by **purpose and rank**, not by pasting a different
roof on a house:

  platform   stepped stone tiers (xumizuo) under the hall, each coped and railed with a
             balustrade, a central flight up the front on the axis (and down the back
             for the halls the axis walks *through*: the gate and the principal hall),
             its middle column in the trim stone -- the imperial way.
  colonnade  frame-role posts on an odd number of bays, the central bay wider, round a
             wall set back behind a porch; an architrave ring on the post heads and a
             bracket course (a frieze band with corbels stepping out under the eave).
  walls      lattice doors across the front bays with the central bay open; solid
             panels between posts on the flanks and the back; lattice windows above.
  eaves      the roof carried out on a deep overhang past the posts. `eaves=2` puts a
             lower skirt roof round the colonnade under the main roof, with a short
             clerestory between them (the double eave of the highest halls).
  storeys    each storey above the first is a real floor, reached by a flight, walled
             and latticed, standing on a pent eave that rings the storey below it; so
             every extra storey is an extra eave tier and a 3-storey hall is visibly
             taller than a 1-storey one of the same footprint.
  roof       hip (wudian, the highest rank), hip_gable (xieshan), gable, pavilion (the
             square pyramid), with a ridge course and raised ridge ends, a finial on a
             pyramid, and the eave corners lifted.
  use        principal: the widest bays, the tallest posts, the deepest front terrace,
             a throne dais; gate: open front and back with the wall on the ridge line
             pierced by three openings, the centre widest, so a person walks straight
             through on the axis; side / rear: front porch, closed flanks; pavilion:
             open on all four sides.

**The silhouette.** The voice owns the roof's *character*: its eave (upturned, flared,
straight) and its profile (the pitch from the eave to the ridge) are read off the voice
the part arrived with, exactly as `TypeBuilder.roof` hands them. What this type decides
is the roof's **rank**, which is the thing it exists to express -- which ends the ridge
has (hip, xieshan, gable, pyramid) and how many eaves stand one over another. Those two
are the voice-contract keys `ends` and `tiers`, and a voice that names them for every
civic building would make every hall the same rank, so this type passes them itself,
on the library's own `roof()`, and names no literal silhouette anywhere (E015 is
clean). See `_roof()`. Every material is a voice role; no block is named here.

The seed changes ornament only (which lanterns, where the interior columns stop, the
corner pieces); it never changes the hierarchy, which is all in the parameters.
"""

import random

FORM = "east_asian"
ROLE = "civic"

PARAMS = {
    "storeys": ("int", 1, 3),
    "eaves": ("int", 1, 2),
    "platform": ("int", 0, 3),
    "roof": ("choice", ["hip", "hip_gable", "gable", "pavilion"]),
    "use": ("choice", ["principal", "gate", "side", "rear", "pavilion"]),
    # the composition round: a principal mass as the show's palace has it -- a tall
    # faced base under the stone tiers, several eave tiers stacked and stepping in, and
    # projecting gabled roofs over the entrance. Unset, the hall is what it was.
    "tiers": ("int", 1, 5),
    "base": ("int", 0, 14),
    "entrance": ("choice", ["plain", "porch"]),
}

#: What this type needs from the ground before it can stand. Measured by
#: scripts/test_ceremonial_hall.py --needs on the plane over a sample of every use at
#: sizes from 7x5 to 64x48 (the pad `site()` hands `build()`).
NEEDS = {
    "footprint": (7, 5, 112, 96),
    "except": (),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
USES = ("principal", "gate", "side", "rear", "pavilion")
ROOFS = ("hip", "hip_gable", "gable", "pavilion")

#: Rank, by purpose. Every number here is a proportion of the hall, never a material:
#: post height (ground storey), bay width, front terrace, flank and back margins on the
#: top tier, porch depth (front, flank, back), eave overhang, central stair width.
RANK = {
    "principal": {"post": 6, "bay": 4, "front": 5, "side": 2, "back": 2,
                  "porch": (2, 2, 2), "over": 3, "stair": 7, "rear_stair": True},
    "rear":      {"post": 5, "bay": 4, "front": 3, "side": 2, "back": 2,
                  "porch": (2, 0, 0), "over": 2, "stair": 5, "rear_stair": False},
    "gate":      {"post": 5, "bay": 4, "front": 2, "side": 1, "back": 2,
                  "porch": (0, 0, 0), "over": 2, "stair": 5, "rear_stair": True},
    "side":      {"post": 4, "bay": 3, "front": 2, "side": 1, "back": 1,
                  "porch": (2, 0, 0), "over": 2, "stair": 3, "rear_stair": False},
    "pavilion":  {"post": 4, "bay": 3, "front": 1, "side": 1, "back": 1,
                  "porch": (0, 0, 0), "over": 2, "stair": 3, "rear_stair": False},
}

#: The ends of the ridge each rank of roof has, and its style. Rank, not silhouette: see
#: the module docstring.
RIDGE_ENDS = {"hip": ("hip", "hip"), "hip_gable": ("irimoya", "irimoya"),
              "gable": ("gable", "gable"), "pavilion": ("hip", "hip")}
#: The rise of the main roof at its ridge, as a share of its half-span, by rank.
PITCH = {"hip": 0.95, "pavilion": 1.05, "hip_gable": 0.9, "gable": 0.8}
UPPER_POST = 3                    # wall courses of a storey above the first


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------- the local frame

class Frame:
    """(u, v) on the pad: u across the front, v back from the front edge (v=0 is the
    row the doorstep is in). Every placement goes through here."""

    def __init__(self, b, part, front):
        self.b = b
        x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
        self.pad = (x0, z0, x1, z1)
        self.front = front
        if front == "north":
            self.W, self.D = x1 - x0 + 1, z1 - z0 + 1
            self.f = lambda u, v: (x0 + u, z0 + v)
            self.plus_u, self.axis_u = "east", "x"
        elif front == "south":
            self.W, self.D = x1 - x0 + 1, z1 - z0 + 1
            self.f = lambda u, v: (x1 - u, z1 - v)
            self.plus_u, self.axis_u = "west", "x"
        elif front == "west":
            self.W, self.D = z1 - z0 + 1, x1 - x0 + 1
            self.f = lambda u, v: (x0 + v, z1 - u)
            self.plus_u, self.axis_u = "north", "z"
        else:
            self.W, self.D = z1 - z0 + 1, x1 - x0 + 1
            self.f = lambda u, v: (x1 - v, z0 + u)
            self.plus_u, self.axis_u = "south", "z"
        self.axis_v = "z" if self.axis_u == "x" else "x"
        self.out = front                       # the way out of the front, world
        self.back = OPP[front]
        self.minus_u = OPP[self.plus_u]
        # roof(): axis "z" is a ridge running along x.
        self.ridge_along_u = "z" if self.axis_u == "x" else "x"
        self.ridge_along_v = "x" if self.axis_u == "x" else "z"

    def rect(self, u0, v0, u1, v1):
        ax, az = self.f(u0, v0)
        bx, bz = self.f(u1, v1)
        return min(ax, bx), min(az, bz), max(ax, bx), max(az, bz)

    def put(self, u, y, v, block):
        x, z = self.f(u, v)
        self.b.place_block(x, y, z, block)

    def box(self, u0, y0, v0, u1, y1, v1, block):
        x0, z0, x1, z1 = self.rect(u0, v0, u1, v1)
        self.b.place_cuboid(x0, min(y0, y1), z0, x1, max(y0, y1), z1, block)

    def get(self, u, y, v):
        x, z = self.f(u, v)
        return self.b.get_block(x, y, z)

    def side_dir(self, side):
        """World direction a face of a local rectangle looks out to."""
        return {"front": self.out, "back": self.back, "u0": self.minus_u,
                "u1": self.plus_u}[side]


def _front_of(part):
    """Which pad edge the door is on: that is the front. `facing` is the direction a
    person walks *in*, so it only breaks a tie at a corner."""
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    d = part.get("door")
    walk = part.get("facing")
    by_walk = {"south": "north", "north": "south", "east": "west", "west": "east"}
    if d is None:
        return by_walk.get(walk, "north")
    dx, dz = int(tuple(d)[0]), int(tuple(d)[-1])
    reach = {"north": dz - z0, "south": z1 - dz, "west": dx - x0, "east": x1 - dx}
    best = min(reach.values())
    tied = [k for k, r in reach.items() if r == best]
    if len(tied) > 1 and by_walk.get(walk) in tied:
        return by_walk[walk]
    return tied[0]


# ---------------------------------------------------------------- bays

def _bays(span, bay):
    """Post offsets 0..span along a colonnade `span+1` wide (span even): an odd number
    of bays, the central one wider. Returns the sorted offsets."""
    if span <= 2:
        return [0, span] if span > 0 else [0]
    best = None
    for b in (bay, bay - 1, bay + 1, bay - 2):
        if b < 2:
            continue
        for nb in range(1, span // b + 2, 2):
            dc = span - (nb - 1) * b
            if dc < b:
                continue
            score = abs(dc - (b + 2)) + (0 if b == bay else 1) - 0.01 * nb
            if dc > b + 4:
                score += (dc - b - 4) * 2
            if best is None or score < best[0]:
                best = (score, b, nb, dc)
    if best is None:
        return [0, span]
    _s, b, nb, dc = best
    c = span // 2
    pos = {c - dc // 2, c + dc // 2}
    k = c - dc // 2
    while k - b >= 0:
        k -= b
        pos.add(k)
    k = c + dc // 2
    while k + b <= span:
        k += b
        pos.add(k)
    pos.add(0)
    pos.add(span)
    out = sorted(pos)
    # an end bay squeezed below two cells is folded into its neighbour
    if len(out) > 3 and out[1] - out[0] < 2:
        out.pop(1)
        out.pop(-2)
    return out


def _even_posts(n, bay):
    """Offsets 0..n evenly, about `bay` apart, always both ends."""
    if n <= 0:
        return [0]
    k = max(1, round(n / float(bay)))
    return sorted({round(i * n / float(k)) for i in range(k + 1)})


# ---------------------------------------------------------------- the plan

def _inset(outer, inner):
    """The deepest of the four insets of `inner` inside `outer`."""
    return max(inner[0] - outer[0], inner[1] - outer[1], outer[2] - inner[2],
               outer[3] - inner[3], 0)


def _plan(G, use, storeys, eaves, platform, roof, ud, pitch, base=0):
    """Every rectangle and level of the hall, decided before one block is placed.
    Where the pad cannot carry what was asked the plan gives up platform tiers first,
    then margins, then eave depth, then the base, and says so.

    `base` is a tall faced podium under the stone tiers: one tier `base` courses high
    across the whole pad, set back from the front by its own height so the central
    flight climbs it at one course per row."""
    R = dict(RANK[use])
    W, D = G.W, G.D
    notes = []
    uc = _clamp(ud, 1, W - 2) if W >= 3 else W // 2
    hw = min(uc, W - 1 - uc)
    small = min(2 * hw + 1, D) < 12
    ht = 1 if small else 2
    rear = R["rear_stair"]
    over = R["over"] if not small else 1
    if min(2 * hw + 1, D) >= 30 and use == "principal":
        over = 3
    P = platform
    Bh = base if not small else 0
    front, side, back = R["front"], R["side"], R["back"]
    if small:
        front, side, back = 0, 0, 0
    if use == "principal":
        front = max(front, D // 7)

    def heights(P, Bh):
        return ([Bh] if Bh else []) + [ht] * P

    def attempt(P, front, side, back, over, Bh=0):
        u0, u1 = uc - hw, uc + hw
        tiers = []
        cum = 0
        hs = heights(P, Bh)
        for k, h in enumerate(hs, 1):
            prev = cum
            cum += h
            step = (k - 1) * ht
            tu0 = u0 + step
            tu1 = u1 - step
            tv0 = cum
            tv1 = (D - 1 - cum) if rear else (D - 1 - prev)
            tiers.append((tu0, tv0, tu1, tv1))
        top = tiers[-1] if tiers else (u0, 0, u1, D - 1)
        cu0, cu1 = top[0] + side, top[2] - side
        cv0, cv1 = top[1] + front, top[3] - back
        # the eave may not leave the pad
        cu0, cu1 = max(cu0, u0 + over), min(cu1, u1 - over)
        cv0, cv1 = max(cv0, over), min(cv1, D - 1 - over)
        if use == "pavilion":                  # square on plan
            s = min(cu1 - cu0, cv1 - cv0)
            du = (cu1 - cu0) - s
            cu0, cu1 = cu0 + du // 2, cu1 - (du - du // 2)
            dv = (cv1 - cv0) - s
            cv0, cv1 = cv0 + dv // 2, cv1 - (dv - dv // 2)
        # symmetric about the axis
        half = min(uc - cu0, cu1 - uc)
        cu0, cu1 = uc - half, uc + half
        return tiers, (cu0, cv0, cu1, cv1)

    # a railed tier (two courses and up) keeps a walk of two rows between its rail and
    # the hall on every side: a one-row strip between a balustrade and a wall is a
    # trench nobody can get into
    if ht >= 2 and P >= 1:
        front, side, back = max(front, 2), max(side, 2), max(back, 2)
    while True:
        tiers, C = attempt(P, front, side, back, over, Bh)
        cw, cd = C[2] - C[0] + 1, C[3] - C[1] + 1
        if cw >= min(7, 2 * hw - 1) and cd >= min(5, D - 2):
            break
        floor = 2 if (ht >= 2 and P >= 1) else 1
        if front > floor or back > floor or side > floor:
            front, back, side = (max(floor, front - 1), max(floor, back - 1),
                                 max(floor, side - 1))
            continue
        if P > 0:
            P -= 1
            continue
        if over > 1:
            over -= 1
            continue
        if Bh:
            Bh = 0
            continue
        if front > 0 or back > 0 or side > 0:
            front = back = side = 0
            continue
        break
    if P < platform:
        notes.append(f"platform: {platform} asked, {P} fit the pad")
    if Bh < base:
        notes.append(f"base: {base} asked, {Bh} fit the pad")
    cu0, cv0, cu1, cv1 = C
    cw, cd = cu1 - cu0 + 1, cv1 - cv0 + 1

    posts_u = [cu0 + o for o in _bays(cw - 1, R["bay"] if cw >= 13 else
                                      (3 if cw >= 9 else 2))]
    posts_v = [cv0 + o for o in _even_posts(cd - 1, R["bay"])]

    # the wall behind the porch
    pf, ps, pb = R["porch"]
    if cd - pf - pb < 5:
        pf, pb = min(pf, 1), min(pb, 1)
    if cd - pf - pb < 4:
        pf = pb = 0
    if cw - 2 * ps < 7:
        ps = 0
    # a porch is a walk: one row deep it is a rail against a wall, so it is none; and a
    # porch down the flanks is reached round the front one or not at all
    pf, ps, pb = [0 if x < 2 else x for x in (pf, ps, pb)]
    if pf < 2:
        ps = 0
    W1 = (cu0 + ps, cv0 + pf, cu1 - ps, cv1 - pb)

    post = R["post"]
    if cd < 7 or cw < 9:
        post = min(post, 4)
    if cw < 9:
        post = 3
    fy = G.floor_y
    hs = heights(P, Bh)
    F1 = fy + sum(hs)

    # storeys: each above the first stands on a pent eave, one course in
    n = storeys
    levels = []
    F = F1
    ring = C
    wall = W1 if use not in ("gate", "pavilion") else (cu0 + 1, cv0 + 1, cu1 - 1, cv1 - 1)
    for k in range(1, n + 1):
        H = post if k == 1 else UPPER_POST
        levels.append({"k": k, "floor": F, "H": H, "ring": ring,
                       "wall": W1 if k == 1 else ring})
        if k == n:
            break
        t = 2 if (wall[2] - wall[0]) >= 22 and (wall[3] - wall[1]) >= 12 else 1
        nxt = (wall[0] + t, wall[1] + t, wall[2] - t, wall[3] - t) if k > 1 else wall
        if nxt[2] - nxt[0] < 6 or nxt[3] - nxt[1] < 4:
            n = k
            notes.append(f"storeys: {storeys} asked, {k} fit the plan")
            break
        ye = F + H + 3
        ov = over if k == 1 else max(1, over - 1)
        band = ov + _inset(ring, nxt) - 1
        nF = ye + _skirt_rise(band, pitch)
        # the flight up has to fit inside the storey it arrives in, along its length
        if nF - F > max(nxt[2] - nxt[0], nxt[3] - nxt[1]) - 3:
            n = k
            notes.append(f"storeys: {storeys} asked, {k} fit the plan (no room for "
                         f"a {nF - F}-tread flight)")
            break
        F = nF
        ring, wall = nxt, nxt
    return {"uc": uc, "hw": hw, "ht": ht, "P": P, "base": Bh, "hs": hs,
            "tiers": tiers, "C": C,
            "posts_u": posts_u, "posts_v": posts_v, "W1": W1, "porch": (pf, ps, pb),
            "over": over, "post": post, "F1": F1, "levels": levels, "n": n,
            "rear": rear, "stair": R["stair"], "notes": notes, "small": small,
            "front": front}


def _skirt_rise(band, pitch=(1, 2)):
    """How far a pent eave rises over `band` courses at the voice's eave pitch."""
    r, run = pitch
    return max(1, (band * r) // max(1, run))


def _skirt_pitch(b):
    prof, _eave = _voice_roof(b)
    if not prof:
        return (1, 2)
    r, run = prof[0]
    r, run = max(1, int(r)), max(1, int(run))
    if r > run:                     # a pent eave is never steeper than 1 in 1
        r, run = 1, 1
    return (r, run)


# ---------------------------------------------------------------- roofs

def _raw(b):
    """The library's own builder under the TypeBuilder: its `roof()` is the one the
    TypeBuilder's calls, without the voice's `ends`/`tiers` written over the rank."""
    return getattr(b, "_b", b)


def _voice_roof(b):
    spec = getattr(b, "roof_spec", None) or {}
    prof = spec.get("profile")
    prof = [tuple(p) for p in prof] if prof else None
    return prof, spec.get("eave")


def _roof(b, G, rect, y, over, roof, band=None, cap=None):
    """One eave tier over the local rectangle `rect`: the main roof, or -- with `band`
    -- a pent/skirt roof stopped `band` courses in from its own eave. Returns the ridge
    (or the skirt's top) y."""
    prof, eave = _voice_roof(b)
    style = "gable" if roof == "gable" else "hip"
    ends = RIDGE_ENDS[roof]
    x0, z0, x1, z1 = G.rect(*rect)
    w = rect[2] - rect[0]
    d = rect[3] - rect[1]
    axis = G.ridge_along_u if w >= d else G.ridge_along_v
    kw = {"style": style, "axis": axis, "overhang": over, "ends": ends}
    if eave:
        # an upturn one course deep lifts the whole eave into a rim that reads as a
        # parapet; on a one-block eave the voice's upturn is drawn straight
        kw["eave"] = "straight" if (eave == "upturned" and over < 2) else eave
    if band is not None:
        kw["profile"] = [_skirt_pitch(b)]
        kw["_stop_out"] = band
        kw["ends"] = RIDGE_ENDS["hip"]
        kw["style"] = "hip"
    else:
        # the main roof: the voice's eave pitch over the lower third of the slope, then
        # one steeper run that brings it to the rise its rank asks at the ridge
        n = (min(w, d) + 1 + 2 * over + 1) // 2
        er, erun = _skirt_pitch(b)
        run1 = max(erun, (n // 3) // erun * erun)
        r1 = run1 * er // erun
        rise = max(2, int(round(n * (cap or 0.7))))
        if n <= 4:
            kw["profile"] = [(1, 1)]
            return _raw(b).roof(x0, z0, x1, z1, y, b.voice["roof"], **kw)
        # (the last segment reaches its rise one course past the ridge column)
        kw["profile"] = [(r1, run1), (max(1, rise - r1), max(1, n - run1 - 1))]
    return _raw(b).roof(x0, z0, x1, z1, y, b.voice["roof"], **kw)


def _top_at(G, u, v, lo, hi):
    for y in range(hi, lo - 1, -1):
        if G.get(u, y, v) not in ("air", "cave_air"):
            return y
    return None


def _ridge(b, G, rect, over, ridge_y, roof, rng):
    """A course along the ridge line in the trim, raised ends, a finial on a pyramid."""
    trim = b.block(b.voice["trim"])
    acc = b.block(b.voice["trim"], "accent")
    u0, v0, u1, v1 = rect[0] - over, rect[1] - over, rect[2] + over, rect[3] + over
    m = over + 1                     # never the eave: an upturned corner is not ridge
    cells = [(u, v) for u in range(u0 + m, u1 - m + 1) for v in range(v0 + m, v1 - m + 1)
             if G.get(u, ridge_y, v) not in ("air", "cave_air")]
    if not cells:
        return 0
    for (u, v) in cells:
        G.put(u, ridge_y + 1, v, trim)
    # the two ends of the ridge line
    along_u = (u1 - u0) >= (v1 - v0)
    key = (lambda c: c[0]) if along_u else (lambda c: c[1])
    lo, hi = min(cells, key=key), max(cells, key=key)
    if key(hi) - key(lo) >= 2:
        for (u, v) in (lo, hi):
            G.put(u, ridge_y + 2, v, acc)
    else:
        cu = sorted(c[0] for c in cells)[len(cells) // 2]
        cv = sorted(c[1] for c in cells)[len(cells) // 2]
        G.put(cu, ridge_y + 2, cv, acc)
        G.put(cu, ridge_y + 3, cv, b.block(b.voice["trim"], "slab"))
    return len(cells)


def _corners(b, G, rect, over, y_e):
    """The four eave corners carried up a course: the upturned corner."""
    u0, v0, u1, v1 = rect[0] - over, rect[1] - over, rect[2] + over, rect[3] + over
    slab = b.block(b.voice["roof"], "slab")
    n = 0
    for (u, v) in ((u0, v0), (u1, v0), (u0, v1), (u1, v1)):
        t = _top_at(G, u, v, y_e - 1, y_e + 3)
        if t is None:
            continue
        G.put(u, t + 1, v, slab)
        n += 1
    return n


# ---------------------------------------------------------------- the platform

def _platform(b, G, pl):
    """Tiers of stone, coped, railed, with the flight up the axis; under them, where the
    hall asks it, a tall base faced in the wall's material."""
    fy = G.floor_y
    foot = b.block(b.voice["footing"])
    fine = b.block(b.voice["footing"], "fine")
    rail = b.block(b.voice["footing"], "wall")
    faced = b.block(b.voice["wall"])
    ht, uc = pl["ht"], pl["uc"]
    sw = pl["stair"]
    hw = pl["hw"]
    sw = min(sw, max(1, 2 * ((hw - 2) // 2) + 1))
    s0, s1 = uc - sw // 2, uc + sw // 2
    P = len(pl["tiers"])
    hs = pl.get("hs") or [ht] * P
    tops = []
    c = 0
    for h in hs:
        c += h
        tops.append(fy + c)
    N = c
    based = bool(pl.get("base"))
    # Paved, the way in included (at floor_y itself)
    G.box(uc - hw, fy, 0, uc + hw, fy, G.D - 1, fine)
    for k, (tu0, tv0, tu1, tv1) in enumerate(pl["tiers"], 1):
        top = tops[k - 1]
        body = faced if (based and k == 1) else foot
        G.box(tu0, fy + 1, tv0, tu1, top, tv1, body)
        if based and k == 1 and top - fy >= 6:
            # a plinth and a string course in the stone: the base reads as masonry
            for (a0, b0, a1, b1) in ((tu0, tv0, tu1, tv0), (tu0, tv1, tu1, tv1),
                                     (tu0, tv0, tu0, tv1), (tu1, tv0, tu1, tv1)):
                G.box(a0, fy + 1, b0, a1, fy + 1, b1, fine)
                G.box(a0, top - 2, b0, a1, top - 2, b1, fine)
        # the coping: the lip of each tier in the finer stone
        for (a0, b0, a1, b1) in ((tu0, tv0, tu1, tv0), (tu0, tv1, tu1, tv1),
                                 (tu0, tv0, tu0, tv1), (tu1, tv0, tu1, tv1)):
            G.box(a0, top, b0, a1, top, b1, fine)
    if P == 0:
        return {"tiers": 0, "rail": 0, "treads": 0}
    # the flights: front, and back where the axis walks through
    flights = [("front", lambda j: j)]
    if pl["rear"]:
        flights.append(("back", lambda j: G.D - 1 - j))
    treads = 0
    for name, vof in flights:
        for u in range(s0, s1 + 1):
            cells = []
            for j in range(1, N + 1):
                v = vof(j)
                y = fy + j
                if j > 1:
                    G.box(u, fy + 1, v, u, y - 1, v, foot)
                G.box(u, y, v, u, y + 3, v, "air")
                cells.append((G.f(u, v)[0], y, G.f(u, v)[1]))
            mat = b.voice["trim"] if (u == uc and sw >= 5) else b.voice["footing"]
            b.steps(cells, mat, axis=G.axis_v)
            treads += len(cells)
        # the cheeks either side, solid to the tread and railed on top
        for u in (s0 - 1, s1 + 1):
            for j in range(1, N + 1):
                v = vof(j)
                if (fy + j) in tops or (based and j <= hs[0]):
                    G.box(u, fy + 1, v, u, fy + j, v, fine)
    # the balustrade: along the lip of every tier, open where a flight comes up
    railed = 0
    for k, (tu0, tv0, tu1, tv1) in enumerate(pl["tiers"] if ht >= 2 else [], 1):
        y = tops[k - 1] + 1
        runs = []
        # front and back lips, broken for the flight and its cheeks
        for vv, has in ((tv0, True), (tv1, pl["rear"])):
            if has:
                runs += [(tu0, vv, s0 - 2, vv), (s1 + 2, vv, tu1, vv)]
            else:
                runs.append((tu0, vv, tu1, vv))
        runs += [(tu0, tv0, tu0, tv1), (tu1, tv0, tu1, tv1)]
        for (a0, b0, a1, b1) in runs:
            if a1 < a0 or b1 < b0:
                continue
            G.box(a0, y, b0, a1, y, b1, rail)
            railed += (a1 - a0 + 1) * (b1 - b0 + 1)
        # the rail turns down beside the flight: close the cheek tops at this lip
        for vv, has in ((tv0, True), (tv1, pl["rear"])):
            if has:
                for u in (s0 - 1, s1 + 1):
                    G.put(u, y, vv, rail)
    return {"tiers": P, "rail": railed, "treads": treads}


# ---------------------------------------------------------------- walls and posts

def _post(b):
    return b.axial(b.block(b.voice["frame"], "post"), "y")


def _lattice(b, G, facing_out):
    """An open lattice leaf standing in the outer face of a wall cell."""
    t = b.joinery(b.voice, "trapdoor")
    return f"{t}[facing={OPP[facing_out]},half=bottom,open=true]"


def _run(G, side, rect):
    """The cells of one side of a local rectangle, in order, and whether posts on this
    side sit at the colonnade's u positions."""
    u0, v0, u1, v1 = rect
    if side == "front":
        return [(u, v0) for u in range(u0, u1 + 1)]
    if side == "back":
        return [(u, v1) for u in range(u0, u1 + 1)]
    if side == "u0":
        return [(u0, v) for v in range(v0, v1 + 1)]
    return [(u1, v) for v in range(v0, v1 + 1)]


def _posts_on(side, rect, posts_u, posts_v, bay):
    u0, v0, u1, v1 = rect
    if side in ("front", "back"):
        ps = sorted({u for u in posts_u if u0 <= u <= u1} | {u0, u1})
        return ps
    inner = [v for v in posts_v if v0 < v < v1]
    if not inner:
        inner = [v0 + o for o in _even_posts(v1 - v0, bay)]
    return sorted(set(inner) | {v0, v1})


def _wall_side(b, G, pl, side, rect, y0, y1, kind, uc, rng):
    """One side of a walled ring from course y0 to y1 inclusive.

        kind: "front"   lattice doors in every bay, the central bay open 3 high
              "through" the central bay open, the rest solid with a lattice window
              "solid"   panels with a lattice window high in each bay wide enough
              "upper"   a dado course and lattice above it in every bay
              "clere"   the clerestory: wall, lattice, wall

    """
    wallb = b.block(b.voice["wall"])
    post = _post(b)
    out = G.side_dir(side)
    lat = _lattice(b, G, out)
    cells = _run(G, side, rect)
    along = [c[0] for c in cells] if side in ("front", "back") else [c[1] for c in cells]
    posts = _posts_on(side, rect, pl["posts_u"], pl["posts_v"], 4)
    fixed = [c for c in cells if (c[0] if side in ("front", "back") else c[1]) in posts]
    for (u, v) in fixed:
        G.box(u, y0, v, u, y1, v, post)
    opened = 0
    for a, c in zip(posts, posts[1:]):
        gap = [cell for cell in cells
               if a < (cell[0] if side in ("front", "back") else cell[1]) < c]
        if not gap:
            continue
        centre = side in ("front", "back") and a < uc < c
        for (u, v) in gap:
            G.box(u, y0, v, u, y1, v, wallb)
        h = y1 - y0 + 1
        if kind == "front":
            if centre:
                for (u, v) in gap:
                    G.box(u, y0, v, u, y0 + 2, v, "air")
                    if h >= 5:
                        G.put(u, y0 + 3, v, lat)
                opened += 1
            else:
                for (u, v) in gap:
                    for y in range(y0, min(y1, y0 + max(2, h - 2)) + 1):
                        G.put(u, y, v, lat)
        elif kind == "through":
            if centre:
                for (u, v) in gap:
                    G.box(u, y0, v, u, y0 + 2, v, "air")
                opened += 1
            elif len(gap) >= 2 and h >= 4:
                for (u, v) in gap:
                    G.put(u, y0 + 2, v, lat)
        elif kind == "solid":
            # one course of lattice, never two: a two-high light in a wall is a niche a
            # person could stand in and nobody can reach
            if len(gap) >= 2 and h >= 4:
                for (u, v) in gap:
                    G.put(u, y0 + 2, v, lat)
        elif kind == "upper":
            for (u, v) in gap:
                for y in range(y0 + 1, y1 + 1):
                    G.put(u, y, v, lat)
        elif kind == "clere":
            if h >= 3:
                for (u, v) in gap:
                    G.put(u, y0 + 1 + (h - 3) // 2, v, lat)
    return opened


def _frieze(b, G, ring, y, out_row=True):
    """The architrave on the post heads (y) and the bracket course over it (y+1): a
    band of the trim stone on the ring line, and corbels one out in the frame, every
    other cell, under the eave."""
    u0, v0, u1, v1 = ring
    beam_u = b.axial(b.block(b.voice["frame"], "post"), "x" if G.axis_u == "x" else "z")
    beam_v = b.axial(b.block(b.voice["frame"], "post"), "z" if G.axis_u == "x" else "x")
    band = b.block(b.voice["trim"], "accent")
    corbel = b.block(b.voice["frame"])
    G.box(u0, y, v0, u1, y, v0, beam_u)
    G.box(u0, y, v1, u1, y, v1, beam_u)
    G.box(u0, y, v0, u0, y, v1, beam_v)
    G.box(u1, y, v0, u1, y, v1, beam_v)
    for side in ("front", "back", "u0", "u1"):
        for (u, v) in _run(G, side, ring):
            G.put(u, y + 1, v, band)
    if not out_row:
        return
    for u in range(u0 - 1, u1 + 2):
        if (u - u0) % 2 == 0:
            G.put(u, y + 1, v0 - 1, corbel)
            G.put(u, y + 1, v1 + 1, corbel)
    for v in range(v0, v1 + 1):
        if (v - v0) % 2 == 0:
            G.put(u0 - 1, y + 1, v, corbel)
            G.put(u1 + 1, y + 1, v, corbel)


# ---------------------------------------------------------------- the storeys

def _storey_one(b, G, pl, use, rng):
    """The ground storey: floor, colonnade, the wall behind the porch (or the gate's
    wall on the ridge line), architrave and brackets. Returns the eave y."""
    lv = pl["levels"][0]
    F, H = lv["floor"], lv["H"]
    C, W1 = pl["C"], pl["W1"]
    cu0, cv0, cu1, cv1 = C
    uc = pl["uc"]
    floor = b.block(b.voice["floor"])
    post = _post(b)
    G.box(cu0, F, cv0, cu1, F, cv1, floor)
    G.box(cu0, F + 1, cv0, cu1, F + H + 2, cv1, "air")
    top = F + H                                  # last course of post
    # posts round the colonnade
    ring_posts = set()
    for u in pl["posts_u"]:
        ring_posts |= {(u, cv0), (u, cv1)}
    for v in pl["posts_v"]:
        ring_posts |= {(cu0, v), (cu1, v)}
    for (u, v) in ring_posts:
        G.box(u, F + 1, v, u, top, v, post)
    walls_top = top + 2                          # under the eave
    gate_openings = 0
    if use == "gate":
        vm = (cv0 + cv1) // 2
        # the flanks, closed; the wall on the ridge line, pierced three times
        for u in (cu0, cu1):
            G.box(u, F + 1, cv0 + 1, u, walls_top, cv1 - 1, b.block(b.voice["wall"]))
            for v in pl["posts_v"]:
                G.box(u, F + 1, v, u, walls_top, v, post)
        G.box(cu0, F + 1, vm, cu1, walls_top, vm, b.block(b.voice["wall"]))
        ps = [u for u in pl["posts_u"]]
        for u in ps:
            G.box(u, F + 1, vm, u, top, vm, post)
        bays = list(zip(ps, ps[1:]))
        ci = next((i for i, (a, c) in enumerate(bays) if a < uc < c), None)
        if ci is not None:
            for i in (ci - 1, ci, ci + 1):
                if not (0 <= i < len(bays)):
                    continue
                a, c = bays[i]
                if i != ci and (c - a) < 2:
                    continue
                if i != ci and (i == 0 or i == len(bays) - 1) and len(bays) > 3:
                    pass
                if i != ci and (a == cu0 or c == cu1):
                    continue                      # never the end bay against a flank
                hgt = H if i == ci else max(3, H - 1)
                G.box(a + 1, F + 1, vm, c - 1, F + hgt - (0 if i == ci else 0), vm,
                      "air")
                gate_openings += 1
        if gate_openings == 0:
            G.box(uc - 1, F + 1, vm, uc + 1, F + 3, vm, "air")
            gate_openings = 1
    elif use != "pavilion":
        pf, ps_, pb = pl["porch"]
        through = pl["rear"]
        _wall_side(b, G, pl, "front", W1, F + 1, walls_top, "front", uc, rng)
        _wall_side(b, G, pl, "back", W1, F + 1, walls_top,
                   "through" if through else "solid", uc, rng)
        _wall_side(b, G, pl, "u0", W1, F + 1, walls_top, "solid", uc, rng)
        _wall_side(b, G, pl, "u1", W1, F + 1, walls_top, "solid", uc, rng)
    _knee_rail(b, G, pl, use, F)
    _frieze(b, G, C, top + 1)
    return top + 3, gate_openings


def _knee_rail(b, G, pl, use, F):
    """A low stone rail between the colonnade's posts wherever the colonnade stands
    in front of open porch, left open at the bays a person walks in by: the central
    bay at the front (and the back where the axis goes through); for a gate the three
    bays of its three openings; for a pavilion the central bay of every side."""
    rail = b.block(b.voice["footing"], "wall")
    cu0, cv0, cu1, cv1 = pl["C"]
    W1 = pl["W1"]
    uc = pl["uc"]
    ps = pl["posts_u"]
    bays = list(zip(ps, ps[1:]))
    ci = next((i for i, (a, c) in enumerate(bays) if a < uc < c), None)
    open_u = set()
    span = [ci] if ci is not None else []
    if use == "gate" and ci is not None:
        span = [i for i in (ci - 1, ci, ci + 1) if 0 < i < len(bays) - 1 or i == ci]
    for i in span:
        a, c = bays[i]
        open_u |= set(range(a + 1, c))
    pv = pl["posts_v"]
    vb = list(zip(pv, pv[1:]))
    mid = (cv0 + cv1) // 2
    open_v = set()
    if use == "pavilion":
        k = next((i for i, (a, c) in enumerate(vb) if a < mid < c), None)
        if k is not None:
            open_v = set(range(vb[k][0] + 1, vb[k][1]))
        else:
            open_v = {mid}
    laid = 0
    walled = {"front": W1[1] == cv0, "back": W1[3] == cv1,
              "u0": W1[0] == cu0, "u1": W1[2] == cu1}
    if use == "gate":
        walled = {"front": False, "back": False, "u0": True, "u1": True}
    if use == "pavilion":
        walled = {"front": False, "back": False, "u0": False, "u1": False}
    through = pl["rear"] or use == "pavilion"
    for side in ("front", "back", "u0", "u1"):
        if walled[side]:
            continue
        for (u, v) in _run(G, side, pl["C"]):
            if side in ("front", "back"):
                if u in ps:
                    continue
                if u in open_u and (side == "front" or through):
                    continue
            else:
                if v in pv or v in (cv0, cv1):
                    continue
                if v in open_v:
                    continue
            G.put(u, F + 1, v, rail)
            laid += 1
    return laid


def _storey_up(b, G, pl, lv, kind="upper"):
    """A storey above the first: floor, walls with lattice on every side, frieze."""
    F, H, ring = lv["floor"], lv["H"], lv["ring"]
    u0, v0, u1, v1 = ring
    G.box(u0, F, v0, u1, F, v1, b.block(b.voice["floor"]))
    G.box(u0 + 1, F + 1, v0 + 1, u1 - 1, F + H + 2, v1 - 1, "air")
    for side in ("front", "back", "u0", "u1"):
        _wall_side(b, G, pl, side, ring, F + 1, F + H, kind, pl["uc"], None)
    _frieze(b, G, ring, F + H + 1)
    return F + H + 3


def _clerestory(b, G, pl, ring, y0, y1):
    for side in ("front", "back", "u0", "u1"):
        _wall_side(b, G, pl, side, ring, y0, y1, "clere", pl["uc"], None)


# ---------------------------------------------------------------- inside

def _flight_up(b, G, label, lower, upper, rng):
    """The stair from one floor to the next, inside the upper storey's walls, run along
    the back wall. Tries both directions and both inner rows."""
    Fl, Fu = lower["floor"], upper["floor"]
    rise = Fu - Fl
    u0, v0, u1, v1 = upper["ring"]
    tries = []
    for v in (v1 - 1, v1 - 2, v0 + 1, v0 + 2):
        tries.append((u0 + 2, v, "plus"))
        tries.append((u1 - 2, v, "minus"))
    for (us, v, way) in tries:
        end = us + rise if way == "plus" else us - rise
        if not (u0 + 1 <= end <= u1 - 1):
            continue
        x, z = G.f(us, v)
        facing = G.plus_u if way == "plus" else G.minus_u
        r = b.flight(label, x, z, Fl, Fu, facing, mat=b.voice["frame"])
        if r and r.get("ok"):
            return {"ok": True, "at": (us, v), "facing": facing, "treads": rise,
                    "along": "u"}
    # a hall deeper than it is wide climbs along a flank instead
    for u in (u0 + 1, u1 - 1, u0 + 2, u1 - 2):
        for (vs, way) in ((v0 + 2, "plus"), (v1 - 2, "minus")):
            end = vs + rise if way == "plus" else vs - rise
            if not (v0 + 1 <= end <= v1 - 1):
                continue
            x, z = G.f(u, vs)
            facing = G.back if way == "plus" else G.out
            r = b.flight(label, x, z, Fl, Fu, facing, mat=b.voice["frame"])
            if r and r.get("ok"):
                return {"ok": True, "at": (u, vs), "facing": facing, "treads": rise,
                        "along": "v"}
    return {"ok": False}


def _furnish(b, G, pl, use, rng, avoid):
    """Modest: a few inner columns, lanterns, the throne dais in the principal hall."""
    lv = pl["levels"][0]
    F = lv["floor"]
    uc = pl["uc"]
    y = F + 1
    if use == "pavilion":
        inner = pl["C"]
    elif use == "gate":
        inner = pl["C"]
    else:
        inner = pl["W1"]
    iu0, iv0, iu1, iv1 = inner[0] + 1, inner[1] + 1, inner[2] - 1, inner[3] - 1
    if iu1 - iu0 < 2 or iv1 - iv0 < 2:
        return 0
    laid = 0
    ceiling = pl["ceiling1"]
    # inner columns: two rows, at the bay lines, where the hall is deep enough
    if use in ("principal", "rear") and iv1 - iv0 >= 8:
        rows = [iv0 + (iv1 - iv0) // 3, iv1 - (iv1 - iv0) // 3]
        for v in rows:
            for u in pl["posts_u"]:
                if iu0 + 1 < u < iu1 - 1 and abs(u - uc) > 1 and (u, v) not in avoid:
                    G.box(u, y, v, u, ceiling, v, _post(b))
                    avoid.add((u, v))
    # the throne dais at the back of the principal hall, on the axis
    if use == "principal" and iv1 - iv0 >= 6 and iu1 - iu0 >= 8:
        dz0 = iv1 - 2
        a0, a1 = uc - 3, uc + 3
        if not any((u, v) in avoid for u in range(a0, a1 + 1) for v in range(dz0, iv1 + 1)):
            x0, z0, x1, z1 = G.rect(a0, dz0, a1, iv1)
            r = b.dais(x0, z0, x1, z1, F, b.voice["footing"])
            if r and r.get("ok"):
                tx, tz = G.f(uc, iv1)
                b.fitting("table", tx, y + 1, tz, G.out, mat=b.voice["frame"],
                          room="hall")
                laid += 1
                for u in (a0, a1):
                    lx, lz = G.f(u, iv1)
                    b.fitting("light", lx, y + 1, lz, G.out, mat=b.voice["frame"],
                              room="hall")
                for u in range(a0, a1 + 1):
                    for v in range(dz0 - 1, iv1 + 1):
                        avoid.add((u, v))
    # lanterns: hung in the corners and along the flanks
    spots = [(iu0, iv0), (iu1, iv0), (iu0, iv1), (iu1, iv1)]
    if iu1 - iu0 >= 12:
        spots += [(iu0 + (iu1 - iu0) // 4, iv0), (iu1 - (iu1 - iu0) // 4, iv0)]
    rng.shuffle(spots)
    for (u, v) in spots[:max(2, len(spots) - rng.randint(0, 2))]:
        if (u, v) in avoid or abs(u - uc) <= 2:
            continue
        x, z = G.f(u, v)
        r = b.fitting("light", x, y, z, G.out, mat=b.voice["frame"], room="hall")
        if r and r.get("ok"):
            laid += 1
    return laid


def _terrace_lanterns(b, G, pl, rng):
    """Two lanterns on the front terrace, either side of the axis."""
    if not pl["tiers"] or pl["C"][1] - pl["tiers"][-1][1] < 3:
        return 0
    v = pl["tiers"][-1][1] + 1
    n = 0
    for du in (-1, 1):
        u = pl["uc"] + du * (pl["stair"] // 2 + 2 + rng.randint(0, 1))
        x, z = G.f(u, v)
        r = b.fitting("light", x, pl["F1"] + 1, z, G.out, mat=b.voice["frame"],
                      room="hall")
        n += bool(r and r.get("ok"))
    return n


# ---------------------------------------------------------------- the entrance roofs

def _porches(b, G, pl, tier_rings, rng):
    """Projecting gabled roofs over the entrance, one on each of the lowest eave tiers,
    each narrower than the one below: the principal mass's front. The lowest stands on
    its own posts on the top terrace in front of the colonnade; the one above rides on
    the skirt roof below it. Returns how many were built."""
    if not tier_rings:
        return 0
    uc = pl["uc"]
    cu0, cv0, cu1, cv1 = pl["C"]
    top_front = pl["tiers"][-1][1] if pl["tiers"] else 0
    F1 = pl["F1"]
    post = _post(b)
    n = 0
    pw = max(9, ((cu1 - cu0) // 4) | 1)
    for i, (y_e, ring, ov) in enumerate(tier_rings[:2]):
        w = pw - 4 * i
        if w < 7:
            break
        h = w // 2
        room = (ring[1] - top_front - 1) if i == 0 else 3
        proj = max(2, min(5, room))
        v0 = ring[1] - proj
        depth = w + 3
        rect = (uc - h, v0, uc + h, v0 + depth)
        if i == 0:
            for u in (uc - h, uc + h):
                G.box(u, F1 + 1, v0, u, y_e - 1, v0, post)
            ring_p = (uc - h, v0, uc + h, ring[1])
            _frieze(b, G, ring_p, y_e - 2, out_row=False)
        else:
            # the gable's wall under the upper projecting roof
            G.box(uc - h, y_e - 2, v0, uc + h, y_e - 1, v0 + 2, b.block(b.voice["wall"]))
        top = _roof(b, G, rect, y_e, 2, "hip_gable", cap=PITCH["hip_gable"])
        _corners(b, G, rect, 2, y_e)
        n += 1
    return n


# ---------------------------------------------------------------- build

def build(b, part, seed, **params):
    rng = random.Random((int(seed) * 2862933555777941757 + 3037000493) % (2 ** 61))
    storeys = _clamp(int(params.get("storeys", 1) or 1), 1, 3)
    eaves = _clamp(int(params.get("eaves", 1) or 1), 1, 2)
    platform = _clamp(int(params.get("platform", 1) if params.get("platform") is not None
                          else 1), 0, 3)
    use = params.get("use") or "side"
    if use not in USES:
        use = "side"
    roof = params.get("roof") or ("pavilion" if use == "pavilion" else "hip_gable")
    if roof not in ROOFS:
        roof = "hip_gable"
    tiers = _clamp(int(params.get("tiers") or eaves), 1, 5)
    eaves = max(eaves, min(2, tiers))
    base = _clamp(int(params.get("base") or 0), 0, 14)
    entrance = params.get("entrance") or "plain"
    requested = {"storeys": storeys, "eaves": eaves, "platform": platform,
                 "roof": roof, "use": use}
    if tiers > 2 or base or entrance != "plain":
        requested.update(tiers=tiers, base=base, entrance=entrance)
    label = part.get("label")
    front = _front_of(part)
    G = Frame(b, part, front)
    G.floor_y = int(part["floor_y"])
    dx, dz = tuple(part["door"])[0], tuple(part["door"])[-1]
    # u of the door on the front edge
    ud = None
    for u in range(G.W):
        if G.f(u, 0) == (int(dx), int(dz)):
            ud = u
    if ud is None:
        ud = G.W // 2
    pl = _plan(G, use, storeys, eaves, platform, roof, ud, _skirt_pitch(b), base=base)
    n = pl["n"]
    over = pl["over"]

    plat = _platform(b, G, pl)
    y_e, gate_openings = _storey_one(b, G, pl, use, rng)
    eave_ys = []
    floors = [pl["F1"]]
    ridge_y = None
    ceiling1 = None
    flights = []
    levels = pl["levels"]
    tier_rings = []
    # the eave tiers, from the bottom up
    for i, lv in enumerate(levels):
        k = lv["k"]
        if k > 1:
            y_e = _storey_up(b, G, pl, lv)
            floors.append(lv["floor"])
        ring = lv["ring"]
        ov = over if k == 1 else max(1, over - 1)
        last = (k == n)
        if not last:
            nxt = levels[i + 1]["ring"]
            band = ov + _inset(ring, nxt) - 1
            _roof(b, G, ring, y_e, ov, roof, band=band)
            _corners(b, G, ring, ov, y_e)
            eave_ys.append(y_e)
            # the wall of the storey above carried down through the skirt to its floor
            Fn = levels[i + 1]["floor"]
            wallb = b.block(b.voice["wall"])
            for side in ("front", "back", "u0", "u1"):
                for (u, v) in _run(G, side, nxt):
                    G.box(u, y_e, v, u, Fn - 1, v, wallb)
            if ceiling1 is None:
                ceiling1 = levels[i + 1]["floor"] - 1
            continue
        # the eave tiers of the top storey: a skirt round it, a clerestory, and the next
        # tier stepped in -- once for the double eave, up to four times for a stacked
        # principal mass -- then the main roof
        n_skirts = (tiers - 1) if tiers > 2 else (1 if eaves == 2 else 0)
        stepv = 0
        if n_skirts > 1:
            span = min(ring[2] - ring[0], ring[3] - ring[1])
            stepv = max(2, span // (2 * (n_skirts + 2)))
        for t in range(n_skirts):
            if t == 0:
                inner = (ring[0] + 1, ring[1] + 1, ring[2] - 1, ring[3] - 1) \
                    if (k > 1 or use in ("gate", "pavilion")
                        or _inset(ring, pl["W1"]) < 1) else pl["W1"]
                if use == "pavilion" and k == 1:
                    inner = (ring[0] + 1, ring[1] + 1, ring[2] - 1, ring[3] - 1)
                if n_skirts > 1:
                    s_ = max(stepv, _inset(ring, inner))
                    inner = (ring[0] + s_, ring[1] + s_, ring[2] - s_, ring[3] - s_)
            else:
                inner = (ring[0] + stepv, ring[1] + stepv, ring[2] - stepv,
                         ring[3] - stepv)
            if not (inner[2] - inner[0] >= 4 and inner[3] - inner[1] >= 2):
                break
            band = ov + _inset(ring, inner) - 1
            tier_rings.append((y_e, ring, ov))
            _roof(b, G, ring, y_e, ov, roof, band=band)
            _corners(b, G, ring, ov, y_e)
            eave_ys.append(y_e)
            s_top = y_e + _skirt_rise(band)
            t_top = _top_at(G, inner[0] - 1, (inner[1] + inner[3]) // 2, y_e,
                            y_e + 16)
            if t_top is not None:
                s_top = t_top
            # a stacked tier shows a band of wall between its eaves
            y_c = s_top + (1 if n_skirts == 1 else 2)
            _clerestory(b, G, pl, inner, y_e, y_c)
            if t == 0 and use in ("gate", "pavilion") and k == 1:
                # a gate or pavilion carries its clerestory on its own posts: the inner
                # ring's corners stand on the floor
                for (u, v) in ((inner[0], inner[1]), (inner[2], inner[1]),
                               (inner[0], inner[3]), (inner[2], inner[3])):
                    G.box(u, lv["floor"] + 1, v, u, y_e, v, _post(b))
            ins = (inner[0] - ring[0], inner[1] - ring[1], ring[2] - inner[2],
                   ring[3] - inner[3])
            if (use in ("gate", "pavilion") and k == 1) or min(ins) != max(ins) \
                    or n_skirts > 1:
                # a skirt that runs further in on some sides than others leaves its top
                # as a ledge inside the room; the hall is ceiled at the skirt instead
                # and the clerestory is a lantern of roof, solid
                G.box(inner[0] + 1, y_e, inner[1] + 1, inner[2] - 1, y_c + 2,
                      inner[3] - 1, b.block(b.voice["wall"]))
                if k == 1 and t == 0:
                    ceiling1 = y_e - 1
            _frieze(b, G, inner, y_c + 1)
            ring = inner
            y_e = y_c + 3
            ov = max(1, ov - 1) if min(inner[2] - inner[0], inner[3] - inner[1]) < 8 \
                else ov
        ridge_y = _roof(b, G, ring, y_e, ov, roof, cap=PITCH[roof])
        _corners(b, G, ring, ov, y_e)
        eave_ys.append(y_e)
        _ridge(b, G, ring, ov, ridge_y, roof, rng)
        if ceiling1 is None:
            ceiling1 = y_e - 1
    pl["ceiling1"] = ceiling1
    porches = _porches(b, G, pl, tier_rings, rng) if entrance == "porch" else 0

    # the ways up, before anything is stood inside
    avoid = set()
    for lower, upper in zip(levels, levels[1:]):
        r = _flight_up(b, G, label, lower, upper, rng)
        flights.append(r)
        if r.get("ok"):
            us, v = r["at"]
            step = 1 if r["facing"] in (G.plus_u, G.back) else -1
            for j in range(-1, r["treads"] + 2):
                avoid.add((us + step * j, v) if r["along"] == "u" else (us, v + step * j))
    fitted = _furnish(b, G, pl, use, rng, avoid)
    fitted += _terrace_lanterns(b, G, pl, rng)

    cu0, cv0, cu1, cv1 = pl["C"]
    b.seal_voids(*G.rect(cu0, cv0, cu1, cv1), b.block(b.voice["wall"]))
    b.check_walkable(label)
    b.check_attached()

    omitted = []
    if n < storeys:
        omitted.append("storeys")
    if pl["P"] < platform:
        omitted.append("platform")
    if any(not f.get("ok") for f in flights):
        omitted.append("flight")
    emitted = {
        "requested": requested,
        "storeys": n,
        "eaves": eaves,
        "platform": pl["P"],
        "roof": roof,
        "use": use,
        "attempt": 0,
        "fallback": "; ".join(pl["notes"]) or None,
        "omitted": omitted,
        "floors": floors,
        "eave_tiers": len(eave_ys),
        "eave_ys": eave_ys,
        "ridge_y": ridge_y,
        "height": (ridge_y - G.floor_y) if ridge_y is not None else None,
        "features": {"platform_tiers": pl["P"], "balustrade": plat["rail"] > 0,
                     "treads": plat["treads"], "rear_stair": bool(pl["rear"] and pl["P"]),
                     "bays": len(pl["posts_u"]) - 1,
                     "gate_openings": gate_openings, "double_eave": eaves == 2,
                     "base": pl.get("base", 0), "entrance_roofs": porches,
                     "flights": sum(1 for f in flights if f.get("ok")),
                     "fittings": fitted},
        "rects": {"colonnade": list(G.rect(*pl["C"])),
                  "wall": list(G.rect(*pl["W1"])),
                  "tiers": [list(G.rect(*t)) for t in pl["tiers"]]},
        "front": front,
    }
    return {"ok": True, "footprint": G.rect(*pl["C"]), "storeys": n, "emitted": emitted}
