"""A town house on a city street: narrow to the lane, deep into the plot.

The two long flanks are party walls -- flat, unbroken, no eave and no opening --
because the next house stands against them. The roof runs front to back, so the
slopes fall over the street and over the rear veranda and the gable ends land on
the party walls. Inside: a front room on the lane, a passage beside it running
the depth of the house, rooms behind it, and the stair in the passage where the
passage runs out. The engawa is at the back, under the eave, facing the strip of
ground the street side does not have.
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
    # Cut to the band `scripts/type_needs.py` measured -- 4x4 to 6x6, 18 of 18 at every
    # size in it -- against the 6x6 to 16x24 its author declared. The measurement is
    # what is on the file. Its author is not wrong about the house: the type is 14 of 14
    # clean on the six real fixtures at both seeds, on pads up to 20x20, and the sweep
    # is a plane at one **square** size at a time, so a type whose whole idea is narrow-
    # to-the-street and deep-into-the-plot is measured on the one shape it is not. That
    # is open thread 15 a third time and it is the sharpest case of it: the two
    # instruments disagree by a factor of four on the same file.
    "footprint": (4, 4, 6, 6),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

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


def _sizes(rng, pad_d, pad_w):
    """Rear strip, depth of the house, width to the street."""
    if pad_d >= 12:
        rear = rng.choice([1, 2])
    elif pad_d >= 8:
        rear = 1
    else:
        rear = 0
    side = 0 if rear else 1
    house_d = pad_d - rear
    if house_d >= 15:
        house_d -= rng.choice([0, 1, 2])
    avail = pad_w - side
    w = min(avail, 9, max(6, house_d - 1))
    if w >= 6 and rng.random() < 0.45:
        w -= 1
    return rear, house_d, max(5, min(w, avail))


def _front_edge(part):
    """Which edge of the pad the reserved door sits in."""
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    runs = {"x": px1 - px0 + 1, "z": pz1 - pz0 + 1}
    cand = [("x", 1, abs(dx - px0)), ("x", -1, abs(dx - px1)),
            ("z", 1, abs(dz - pz0)), ("z", -1, abs(dz - pz1))]
    cand.sort(key=lambda c: (c[2], -runs[c[0]]))
    return cand[0][0], cand[0][1]


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

    ax, s = _front_edge(part)
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

    rear, house_d, w = _sizes(rng, pad_d, pad_w)

    # Slide the frontage along the pad so the reserved doorstep lands in the front wall,
    # and off its corner where there is room for that.
    spans = [aa for aa in range(latlo, lathi - w + 2)]
    strict = [aa for aa in spans if aa < dlat < aa + w - 1]
    loose = [aa for aa in spans if aa <= dlat <= aa + w - 1]
    a = rng.choice(strict) if strict else (loose[0] if loose else latlo)

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
    if w > 5:
        attempts.append((house_d, w - 1, 1))

    res = None
    for (hd, ww, stt) in attempts:
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
            house_d, w, st = hd, ww, stt
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
        "storeys": int(st), "attempt": int(attempts.index((house_d, w, st))),
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

    # one screen upstairs, never reaching across, so nothing can be shut in
    if st >= 2 and inner_hi - inner_lo >= 2 and rng.random() < 0.6:
        dscr = max(2, min(back - 1, back // 2 + 1))
        edge = inner_lo if tp >= (inner_lo + inner_hi) / 2.0 else inner_hi
        step = 1 if edge == inner_lo else -1
        for k in range(max(1, (inner_hi - inner_lo) // 2)):
            tt = edge + step * k
            X, Z = cell(dscr, tt)
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
    # ground, and covered ground nobody can walk onto is a fault.
    if rear >= 1:
        DX, DZ = cell(house_d - 1, tp)
        b.doorway(DX, fy + 1, DZ, outward, voice["wall"],
                  leaf=b.joinery(voice, "door"), jamb="build")

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

    occ = set(busy)
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
                if (tt - runs[0]) % gap == 0:
                    for yy in range(fy + 1, fy + 4):
                        b.place_block(X, yy, Z, frame_blk)
                    posts.add(tt)
                elif tt != step_at:
                    b.place_block(X, fy + 1, Z, fence)
        if st >= 2:
            lip = b.block(voice["trim"], "slab")
            for (k, tt) in sorted(have):
                X, Z = deck_cell(k, tt)
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

    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "storeys": st, "depth": house_d, "width": w, "emitted": emitted,
            "front": front, "rear": rear}
