import math
import random

KIND = "edge"
FORM = "fortification"
#: This wall's spine, its outward normal and its ways down are written along x or z, so
#: it draws no diagonal run.
DIAGONAL_RUNS = False
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

PARAMS = {
    "height": ("int", 3, 20),
    "width": ("int", 1, 5),
    "crown": ("choice", ["crenellated", "machicolated", "solid"]),
}

#: `part["face"]` is `plain` for the flush face of a wall the spec calls unbroken,
#: written by the layout beside the part's level, and anything else is the buttress at
#: every pier and the string course between them this wall has carried. A part field
#: rather than a fourth parameter because the sweep is held to 27 points.
FACES = ("framed", "plain")

#: **A tall wall has a face.** Above this height the outer face carries a buttress a
#: block proud of the parapet line at every pier, three quarters of the way up and
#: capped in trim, and a string course at mid-height between them. Below it the wall
#: keeps its flush piers: a wall of eight is a wall of eight. The city's palace ring is
#: twenty high, and on the volume its flush red piers every five columns read as a blank
#: pale plane from fifty blocks off; relief is what reads at that distance. Registered
#: before the wall was re-swept.
FACE_FROM = 12

NEEDS = {
    # The band `scripts/type_needs.py` measured. **The width reaches a rampart's**, the
    # craft round (E4): the sweep only ever tried 1, 2 and 3, and the re-sweep over the
    # widths a place may declare as a wall's mass stands 2,520 instances of this type at
    # 1, 2, 3, 5, 7, 9 and 12 with none failing. A ring wall of a 512-block city takes a
    # vertex at least every 128 columns.
    "footprint": (1, 4, 12, 128),
    "frontage": "any",
    "ground": "any",
    "clearance": 4,
}


def _line(a, c, axis):
    """Every column from a to c inclusive, in order along the line."""
    x0, z0 = int(a[0]), int(a[1])
    x1, z1 = int(c[0]), int(c[1])
    out = []
    if axis == "x":
        st = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + st, st):
            out.append((x, z0))
    else:
        st = 1 if z1 >= z0 else -1
        for z in range(z0, z1 + st, st):
            out.append((x0, z))
    return out


def _terrain_bias(b, pts, axis):
    """Positive when the +perpendicular side of this run is the lower ground."""
    step = max(1, len(pts) // 8)
    hp = 0
    hm = 0
    c = 0
    for i in range(0, len(pts), step):
        x, z = pts[i]
        if axis == "x":
            hp += b.get_height(x, z + 5)
            hm += b.get_height(x, z - 5)
        else:
            hp += b.get_height(x + 5, z)
            hm += b.get_height(x - 5, z)
        c += 1
    if c == 0:
        return 1.0
    return float(hm - hp) + 0.001


def build(b, part, seed, **params):
    rnd = random.Random(seed * 7919 + 13)
    height = max(3, min(20, int(params.get("height", 8))))
    width = max(1, min(5, int(params.get("width", 3))))
    crown = params.get("crown", "crenellated")
    if crown not in ("crenellated", "machicolated", "solid"):
        crown = "crenellated"
    face_kind = part.get("face") or "framed"
    if face_kind not in FACES:
        face_kind = "framed"

    voice = part["voice"]
    WALL = b.block(voice["wall"], "full")
    FOOT = b.block(voice["footing"], "full")
    TRIM = b.block(voice["trim"], "full")
    PAVE = b.block(voice["roof"], "full")
    STEPMAT = voice["roof"]
    GF = int(part["floor_y"])

    out = width // 2
    inn = width - 1 - out
    M = out + 1

    # ---- the spine: one ordered run of columns down the whole line -----------
    segs = part["segments"]
    spine = []
    sfloor = []
    sidx = []
    junctions = []
    seg_pts = []
    for si, s in enumerate(segs):
        cells = _line(s["a"], s["b"], s["axis"])
        seg_pts.append(cells)
        if si > 0:
            junctions.append(len(spine))
            cells = cells[1:]
        for (x, z) in cells:
            spine.append((x, z))
            sfloor.append(int(s["floor_y"]))
            sidx.append(si)
    if not spine:
        return None
    closed = len(spine) > 3 and spine[0] == spine[-1]
    if closed:
        spine.pop()
        sfloor.pop()
        sidx.pop()
        junctions.append(0)
    n = len(spine)

    segstart = {}
    segend = {}
    for i in range(n):
        si = sidx[i]
        if si not in segstart:
            segstart[si] = i
        segend[si] = i

    # what an empty cell is called here, read off the sky rather than named
    probe = []
    for k in (0, n // 3, (2 * n) // 3):
        px0, pz0 = spine[k % n]
        probe.append(b.get_block(px0, GF + 58, pz0))
    AIR = probe[0] if (probe[0] == probe[1] and probe[1] == probe[2]) else None

    # ---- which way is out ----------------------------------------------------
    cx = sum(p[0] for p in spine) / float(n)
    cz = sum(p[1] for p in spine) / float(n)
    outv = []
    for si, s in enumerate(segs):
        if s["axis"] == "x":
            d = float(s["a"][1]) - cz
            if abs(d) < 1.0:
                d = _terrain_bias(b, seg_pts[si], "x")
            outv.append((0, 1 if d >= 0 else -1))
        else:
            d = float(s["a"][0]) - cx
            if abs(d) < 1.0:
                d = _terrain_bias(b, seg_pts[si], "z")
            outv.append((1 if d >= 0 else -1, 0))

    # ---- the height of the crown, column by column --------------------------- each
    # segment wants its own footing plus `height`; where two segments meet, the higher
    # of the two wins over a small plateau and the wall ramps a block a column back down
    # to its own level, so the walk never steps more than one and a person walks the
    # whole length without stopping.
    T = [sfloor[i] + height for i in range(n)]
    top = list(T)
    for j in junctions:
        v = max(T[(j - 1) % n], T[j], T[(j + 1) % n])
        for d in range(-M, M + 1):
            k = j + d
            if closed:
                k %= n
            elif k < 0 or k >= n:
                continue
            if v > top[k]:
                top[k] = v
    for _ in range(2):
        for i in range(n):
            k = i - 1
            if k < 0:
                if not closed:
                    continue
                k = n - 1
            if top[k] - 1 > top[i]:
                top[i] = top[k] - 1
        for i in range(n - 1, -1, -1):
            k = i + 1
            if k >= n:
                if not closed:
                    continue
                k = 0
            if top[k] - 1 > top[i]:
                top[i] = top[k] - 1
    YMAX = GF + 40
    for i in range(n):
        if top[i] > YMAX:
            top[i] = YMAX
    walk = [t - 2 for t in top]

    # ---- ground, and the record of what stands where --------------------------
    ghc = {}

    def gh(x, z):
        k = (x, z)
        v = ghc.get(k)
        if v is None:
            v = b.get_height(x, z)
            ghc[k] = v
        return v

    def base_at(x, z, sf):
        g = gh(x, z)
        v = sf + 1
        if g + 1 < v:
            v = g + 1
        if v < GF:
            v = GF
        return v

    # A column the library sited -- every column of every segment's swept width -- has a
    # footing laid to its segment's level, so it is always usable. A column outside that
    # width (a parapet's overhang, a buttress, a flight) is usable where its own ground
    # is within reach of the **local segment's** level, never the polyline's minimum:
    # the concentric run's ring walls omitted every column whose bed lay three below the
    # lowest segment of a ring, and read as walls that vanished into the earth over
    # every hollow.
    sited = set()
    for s in segs:
        for c in s.get("cells") or []:
            sited.add((int(c[0]), int(c[1])))

    def usable(x, z, sf=None):
        if (x, z) in sited:
            return True
        return gh(x, z) > (GF if sf is None else sf) - 3

    crest = {}
    placed = set()

    def mark(x, z, y):
        k = (x, z)
        v = crest.get(k)
        if v is None or y > v:
            crest[k] = y

    # ---- the columns the wall stands in --------------------------------------
    wcols = {}          # (x,z) -> [walk_y, seg_floor, spine_index, face]
    pcols = {}          # (x,z) -> [walk_y, seg_floor, spine_index, px, pz]
    for i in range(n):
        x, z = spine[i]
        px, pz = outv[sidx[i]]
        for o in range(-inn, out + 1):
            key = (x + px * o, z + pz * o)
            face = 0
            if o == out:
                face = 1
            elif o == -inn:
                face = -1
            cur = wcols.get(key)
            if cur is None or walk[i] > cur[0]:
                wcols[key] = [walk[i], sfloor[i], i, face]
        key = (x + px * (out + 1), z + pz * (out + 1))
        cur = pcols.get(key)
        if cur is None or walk[i] > cur[0]:
            pcols[key] = [walk[i], sfloor[i], i, px, pz]

    # the corner: a square bastion both segments run into, so the walk turns without a
    # gap and the parapet carries round the outside of it.
    for j in junctions:
        x, z = spine[j]
        p1 = outv[sidx[(j - 1) % n]]
        p2 = outv[sidx[j]]
        if p1[0] * p2[1] - p1[1] * p2[0] == 0:
            continue
        sf = max(sfloor[(j - 1) % n], sfloor[j])
        for a in range(-M, M + 1):
            for c in range(-M, M + 1):
                key = (x + p1[0] * a + p2[0] * c, z + p1[1] * a + p2[1] * c)
                if a == out + 1 or c == out + 1:
                    qx, qz = (p1 if a == out + 1 else p2)
                    cur = pcols.get(key)
                    if cur is None or walk[j] > cur[0]:
                        pcols[key] = [walk[j], sf, j, qx, qz]
                else:
                    cur = wcols.get(key)
                    if cur is None or walk[j] > cur[0]:
                        wcols[key] = [walk[j], sf, j, 0]
    for key in list(pcols.keys()):
        if key in wcols:
            del pcols[key]

    # ---- masonry -------------------------------------------------------------
    pierspace = rnd.choice([4, 5])
    talus = rnd.random() < 0.55
    for (x, z) in wcols:
        wy, sf, i, face = wcols[(x, z)]
        if not usable(x, z, sf):
            continue
        placed.add((x, z))
        bs = base_at(x, z, sf)
        if bs > wy:
            bs = wy
        ftop = bs + 1
        if ftop > wy - 1:
            ftop = wy - 1
        if ftop >= bs:
            b.place_cuboid(x, bs, z, x, ftop, z, FOOT)
            body0 = ftop + 1
        else:
            body0 = bs
        if body0 <= wy - 1:
            mat = WALL
            if face != 0 and (i % pierspace) == 0:
                mat = TRIM
            b.place_cuboid(x, body0, z, x, wy - 1, z, mat)
            if face != 0 and mat != TRIM:
                b.place_block(x, wy - 1, z, TRIM)
        b.place_block(x, wy, z, PAVE)
        mark(x, z, wy)

    # two walls of one city crowned differently and neither reading as a parapet.
    # `Builder.merlon()` is the rhythm, the wall family is what it is built of, and the
    # trim is left to the string course and the corbel, which is what a string course
    # is.
    brk = rnd.choice([2, 3])
    for (x, z) in pcols:
        wy, sf, i, px, pz = pcols[(x, z)]
        if not usable(x, z, sf):
            continue
        bs = base_at(x, z, sf)
        if bs > wy:
            bs = wy
        ftop = bs + 1
        if ftop > wy:
            ftop = wy
        if ftop >= bs:
            b.place_cuboid(x, bs, z, x, ftop, z, FOOT)
            body0 = ftop + 1
        else:
            body0 = bs
        if body0 <= wy:
            b.place_cuboid(x, body0, z, x, wy, z, WALL)
        b.place_block(x, wy + 1, z, WALL)
        if crown == "crenellated":
            cap = b.merlon(i)
        elif crown == "machicolated":
            cap = (i % 2) == 0
        else:
            cap = True
        if cap:
            b.place_block(x, wy + 2, z, WALL)
            mark(x, z, wy + 2)
        else:
            mark(x, z, wy + 1)
        # the overhang: a corbel bracket standing a block proud of the face
        if talus and (i % brk) == 0 and wy - 1 >= bs:
            kx, kz = x + px, z + pz
            if (kx, kz) not in wcols and (kx, kz) not in pcols \
                    and usable(kx, kz, sf):
                b.place_block(kx, wy - 1, kz, TRIM)
                mark(kx, kz, wy - 1)
        # the face, above FACE_FROM: a buttress at every pier, proud of the parapet
        # line, and a string course between the buttresses at mid-height -- unless the
        # spec called the wall unbroken, in which case the face is the face
        if height >= FACE_FROM and face_kind != "plain":
            kx, kz = x + px, z + pz
            if (kx, kz) not in wcols and (kx, kz) not in pcols and usable(kx, kz, sf):
                bb = base_at(kx, kz, sf)
                if (i % pierspace) == 0:
                    tb = bs + ((wy - bs) * 3) // 4
                    if tb >= bb:
                        b.place_cuboid(kx, bb, kz, kx, tb, kz, WALL)
                        b.place_block(kx, tb + 1, kz, b.block(voice["trim"], "slab")
                                      + "[type=bottom]")
                        mark(kx, kz, tb + 1)
                else:
                    sc = bs + (wy - bs) // 2
                    if sc > bb:
                        b.place_block(kx, sc, kz, b.block(voice["trim"], "slab")
                                      + "[type=top]")
                        mark(kx, kz, sc)

    # ---- where the walk changes level, it changes on treads ------------------
    ramp = [False] * n
    for i in range(n):
        if i > 0:
            p = walk[i - 1]
        elif closed:
            p = walk[n - 1]
        else:
            p = walk[i]
        if i < n - 1:
            q = walk[i + 1]
        elif closed:
            q = walk[0]
        else:
            q = walk[i]
        ramp[i] = (walk[i] != p) or (walk[i] != q)
    i = 0
    while i < n:
        if not ramp[i]:
            i += 1
            continue
        j = i
        while j < n and ramp[j] and sidx[j] == sidx[i]:
            j += 1
        for o in range(-inn, out + 1):
            cells = []
            for k in range(i, j + 1):
                key = None
                if k < j:
                    x, z = spine[k]
                    px, pz = outv[sidx[k]]
                    key = (x + px * o, z + pz * o)
                if key is not None and key in placed:
                    cells.append((key[0], walk[k], key[1]))
                else:
                    if len(cells) > 1:
                        b.steps(cells, STEPMAT)
                    cells = []
        i = j

    # ---- getting off the wall ------------------------------------------------ a
    # flight running along the inner face, just outside it, on a solid ramp of its own,
    # doubling back once where the drop is too long for a single run.
    def lay_flight(cells, gbase):
        kept = []
        for (x, y, z) in cells:
            if not usable(x, z):
                break
            base = gh(x, z) + 1
            if base > gbase + 1:
                base = gbase + 1
            if base < GF:
                base = GF
            if base <= y - 1:
                b.place_cuboid(x, base, z, x, y - 1, z, FOOT)
            mark(x, z, y)
            kept.append((x, y, z))
        if kept:
            b.steps(kept, STEPMAT)
        return len(kept)

    # **A plain face carries sparse ways down**, the craft round. The layout stamps
    # `stairs` on the part beside its `face`; `sparse` puts a way down at each vertex
    # and beside each gate, where a person actually climbs, and nowhere else.
    sparse = str(part.get("stairs") or "") == "sparse"
    spacing = rnd.choice([14, 16, 18])
    if sparse:
        anchors = []
        for si in range(len(seg_pts)):
            a = segstart[si] + M + 2
            if segstart[si] <= a <= segend[si]:
                anchors.append(a)
    else:
        anchors = list(range(rnd.randrange(2, 8), n, spacing))
    if not anchors:
        anchors = [n // 2]
    # A way down never lands in a gate's pad: siting hands this wall the points standing
    # on it (`part["gates"]`, each with its anchor and pad), and a flight laid where a
    # gate is then sited is a flight the gate's pad cuts through -- the city's palace
    # gate had this wall's own treads facing down its quarried cut.
    keep_off = set()
    for g in (part.get("gates") or []):
        gx, gz = int(g["at"][0]), int(g["at"][-1])
        half = int(g.get("size", 5)) // 2 + M + 3
        near = [i for i in range(n) if abs(spine[i][0] - gx) + abs(spine[i][1] - gz) <= half]
        keep_off.update(near)
    anchors = [a for a in anchors if a not in keep_off] or anchors
    for ai in anchors:
        si = sidx[ai]
        lo = segstart[si]
        hi = segend[si]
        px, pz = outv[sidx[ai]]
        pts = seg_pts[si]
        if len(pts) < 2:
            continue
        if pts[1][0] != pts[0][0]:
            ux, uz = (1 if pts[1][0] > pts[0][0] else -1), 0
        else:
            ux, uz = 0, (1 if pts[1][1] > pts[0][1] else -1)
        if (hi - ai) >= (ai - lo):
            dx, dz = ux, uz
        else:
            dx, dz = -ux, -uz
        # nudge the flight to ground it can actually land on
        best = None
        besta = None
        for shift in (0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 13, -13):
            a = ai + shift
            if a - lo < M + 1:
                a = lo + M + 1
            if hi - a < M + 1:
                a = hi - M - 1
            if a < lo or a > hi:
                continue
            room = (hi - a) if (dx * ux + dz * uz) > 0 else (a - lo)
            lim = max(2, min(18, room - M))
            gmin = None
            for o in (inn + 1, inn + 2):
                for k in range(0, lim + 1, 2):
                    g = gh(spine[a][0] - px * o + dx * k,
                           spine[a][1] - pz * o + dz * k)
                    if gmin is None or g < gmin:
                        gmin = g
            if gmin is not None and (best is None or gmin > best):
                best = gmin
                besta = a
            if best is not None and best > GF - 3:
                break
        if besta is None:
            continue
        a = besta
        room = (hi - a) if (dx * ux + dz * uz) > 0 else (a - lo)
        limit = max(2, min(18, room - M))
        W = walk[a]
        sx, sz = spine[a]
        l1x = sx - px * (inn + 1)
        l1z = sz - pz * (inn + 1)
        l2x = sx - px * (inn + 2)
        l2z = sz - pz * (inn + 2)
        low = None
        for k in range(0, limit + 1):
            for (qx, qz) in ((l1x + dx * k, l1z + dz * k),
                             (l2x + dx * k, l2z + dz * k)):
                g = gh(qx, qz)
                if g <= GF - 3:
                    continue
                if low is None or g < low:
                    low = g
        if low is None or low < GF:
            low = GF
        need = W - low
        if need <= 0:
            continue
        n1 = need if need < limit else limit
        run1 = []
        for k in range(0, n1):
            run1.append((l1x + dx * k, W - k, l1z + dz * k))
        got = lay_flight(run1, low)
        if need <= n1 or got < n1:
            continue
        # a landing, then back the other way one column further in
        landy = W - n1
        n2 = need - n1
        if n2 > n1:
            n2 = n1
        stop = False
        for (bx, bz) in ((l1x, l1z), (l2x, l2z)):
            x = bx + dx * n1
            z = bz + dz * n1
            if not usable(x, z):
                stop = True
                continue
            base = gh(x, z) + 1
            if base > low + 1:
                base = low + 1
            if base < GF:
                base = GF
            if base <= landy:
                b.place_cuboid(x, base, z, x, landy, z, FOOT)
            mark(x, z, landy)
        if stop:
            continue
        run2 = []
        for k in range(0, n2):
            run2.append((l2x + dx * (n1 - 1 - k), landy - k,
                         l2z + dz * (n1 - 1 - k)))
        lay_flight(run2, low)

    # ---- the gate ------------------------------------------------------------ a wall
    # with no way through it is a wall a city cannot use. It goes only where the wall is
    # deep enough to carry the walk over the opening and the ground stands level on both
    # sides of it -- and only where no gate part stands on this wall: siting hands the
    # points standing on an edge back as `part["gates"]`, and a gate's pad is the way
    # through.
    if AIR is not None and closed and not part.get("gates"):
        best = None
        for si in range(len(segs)):
            ln = segend[si] - segstart[si]
            if best is None or ln > best[0]:
                best = (ln, si)
        gi = None
        if best is not None and best[0] > 2 * M + 6:
            si = best[1]
            mid = (segstart[si] + segend[si]) // 2
            for shift in (0, 3, -3, 6, -6, 9, -9, 12, -12):
                a = mid + shift
                if a - segstart[si] < M + 3 or segend[si] - a < M + 3:
                    continue
                base = sfloor[a] + 1
                if walk[a] - base < 4:
                    continue
                px, pz = outv[sidx[a]]
                sx, sz = spine[a]
                ox, oz = sx + px * (out + 2), sz + pz * (out + 2)
                ix, iz = sx - px * (inn + 2), sz - pz * (inn + 2)
                if gh(ox, oz) != base - 1 or gh(ix, iz) != base - 1:
                    continue
                gi = a
                break
        if gi is not None:
            a = gi
            base = sfloor[a] + 1
            px, pz = outv[sidx[a]]
            sx, sz = spine[a]
            for o in range(-(inn + 2), out + 3):
                b.fill_region(sx + px * o, base, sz + pz * o,
                              sx + px * o, base + 2, sz + pz * o, AIR)
            if px == 0:
                face = "south" if pz > 0 else "north"
            else:
                face = "east" if px > 0 else "west"
            b.doorway(sx, base, sz, face, voice["frame"],
                      leaf=b.joinery(voice, "door"), jamb="build", lintel=True)
        else:
            # too low for a gateway under the walk: a postern through the parapet, where
            # the ground outside stands level with the walk.
            cands = []
            near = []
            start = rnd.randrange(2, 7)
            for a in range(start, n, 2):
                atjoin = False
                for j in junctions:
                    if abs(a - j) <= M + 2:
                        atjoin = True
                if atjoin:
                    continue
                px, pz = outv[sidx[a]]
                sx, sz = spine[a]
                qx, qz = sx + px * (out + 1), sz + pz * (out + 1)
                if (qx, qz) not in pcols or not usable(qx, qz):
                    continue
                if (sx + px * out, sz + pz * out) not in placed:
                    continue
                wy = walk[a]
                ox, oz = sx + px * (out + 2), sz + pz * (out + 2)
                d0 = gh(ox, oz) - wy
                if abs(d0) > 1:
                    continue
                fx, fz = sx + px * (out + 3), sz + pz * (out + 3)
                if abs(gh(fx, fz) - wy) > 2:
                    continue
                if d0 == 0:
                    cands.append((a, qx, qz, wy, px, pz))
                elif len(near) < 8:
                    near.append((a, qx, qz, wy, px, pz))
                if len(cands) >= 10:
                    break
            chosen = None
            for (a, qx, qz, wy, px, pz) in cands + near:
                b.fill_region(qx, wy + 1, qz, qx, wy + 2, qz, AIR)
                r = b.check_door(qx, wy + 1, qz)
                if r is not None and r.get("ok"):
                    chosen = (a, qx, qz, wy, px, pz)
                    break
                cap = b.merlon(a) if crown == "crenellated" else (
                    (a % 2) == 0 if crown == "machicolated" else True)
                b.place_block(qx, wy + 1, qz, WALL)
                if cap:
                    b.place_block(qx, wy + 2, qz, WALL)
            if chosen is not None:
                (a, qx, qz, wy, px, pz) = chosen
                if px == 0:
                    face = "south" if pz > 0 else "north"
                else:
                    face = "east" if px > 0 else "west"
                b.doorway(qx, wy + 1, qz, face, voice["frame"],
                          leaf=b.joinery(voice, "door"), jamb="build",
                          lintel=False)
                mark(qx, qz, wy + 2)

    # ---- the wall cuts what stood in its columns; nothing is left hanging -----
    if AIR is not None:
        for (x, z) in crest:
            y0 = crest[(x, z)] + 1
            y1 = gh(x, z) + 4
            if y1 < y0 + 2:
                y1 = y0 + 2
            if y1 > YMAX:
                y1 = YMAX
            if y0 <= y1:
                b.fill_region(x, y0, z, x, y1, z, AIR)

    # ---- anything the mass roofed over that nobody can reach ------------------
    pad = M + 3
    for si in range(len(segs)):
        pts = seg_pts[si]
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        b.seal_voids(min(xs) - pad, min(zs) - pad, max(xs) + pad,
                     max(zs) + pad, FOOT, max_cells=900)

    return {"columns": len(wcols), "parapet": len(pcols), "top": max(top)}
