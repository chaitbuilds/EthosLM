import math
import random

KIND = "area"
FORM = "civic"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "civic"

PARAMS = {
    "crop": ("choice", ["grain", "roots", "beet", "mixed"]),
    "layout": ("choice", ["strips", "quarters", "plots"]),
}

NEEDS = {
    "footprint": (3, 3, 48, 48),
    "except": (11, 16, 18, 20, 24, 28),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

_CROPS = {
    "grain": (("wheat", 7),),
    "roots": (("carrots", 7), ("potatoes", 7)),
    "beet": (("beetroots", 3),),
    "mixed": (("wheat", 7), ("carrots", 7), ("potatoes", 7), ("beetroots", 3)),
}


def _rect(part):
    fp = part.get("footprint")
    if fp is not None:
        return int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3])
    return int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])


def _xz(v):
    if v is None:
        return None
    if isinstance(v, dict):
        if "x" in v and "z" in v:
            return int(v["x"]), int(v["z"])
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 2:
            return int(v[0]), int(v[1])
        if len(v) == 3:
            return int(v[0]), int(v[2])
    return None


def _face_to(x, z, cx, cz):
    dx, dz = cx - x, cz - z
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def _outward(x0, z0, x1, z1, x, z):
    if x == x0:
        return x0 - 1, z
    if x == x1:
        return x1 + 1, z
    if z == z0:
        return x, z0 - 1
    return x, z1 + 1


def _snap(x0, z0, x1, z1, x, z):
    """The perimeter cell nearest a column outside it."""
    x = min(max(x, x0), x1)
    z = min(max(z, z0), z1)
    reach = {"west": x - x0, "east": x1 - x, "north": z - z0, "south": z1 - z}
    side = min(reach, key=lambda k: reach[k])
    if side == "west":
        return x0, z
    if side == "east":
        return x1, z
    if side == "north":
        return x, z0
    return x, z1


def _side_of(x0, z0, x1, z1, x, z):
    if z == z0:
        return "north"
    if z == z1:
        return "south"
    if x == x0:
        return "west"
    return "east"


def _gates(b, part, x0, z0, x1, z1, y, rng):
    """Where a person can get in. Read off the ground, not guessed from a name.

        A perimeter cell is a way in when the ground immediately outside it is at
        the field's own level; the gate goes at the one nearest the doorstep the
        circulation pass reserved, and a second one on another side where the
        ground offers it.
        
    """
    ring = [c for c in _ring_cells(x0, z0, x1, z1)
            if not (c[0] in (x0, x1) and c[1] in (z0, z1))]
    open_cells = []
    for (x, z) in ring:
        ox, oz = _outward(x0, z0, x1, z1, x, z)
        if abs(b.get_height(ox, oz) - y) <= 1:
            open_cells.append((x, z))

    ranked = []
    pool = open_cells if len(open_cells) >= 3 else ring
    step = 1 if len(pool) <= 60 else 2
    for c in pool[::step]:
        ln = b.nearest_lane(c[0], c[1])
        if ln is not None:
            ranked.append((float(ln.get("distance", 999.0)), c))
    ranked.sort(key=lambda p: p[0])

    want = _xz(part.get("door"))
    if want is None:
        th = b.threshold(part.get("label"))
        if th:
            want = _xz(th.get("door")) or _xz(th.get("lane"))
    near = bool(ranked) and ranked[0][0] <= 36.0

    good = []
    if near or want is not None:
        cands = []
        if want is not None:
            cands.append(_snap(x0, z0, x1, z1, want[0], want[1]))
        for _, c in ranked[:12]:
            if c not in cands:
                cands.append(c)
        for c in cands[:10]:
            r = b.check_door(c[0], y + 1, c[1])
            if r is not None and r.get("ok"):
                good.append(c)
            if len(good) >= 4:
                break

    if good:
        gates = [good[0]]
        first = _side_of(x0, z0, x1, z1, gates[0][0], gates[0][1])
        other = [c for c in good[1:]
                 if _side_of(x0, z0, x1, z1, c[0], c[1]) != first]
        if other and rng.random() < 0.7:
            gates.append(rng.choice(other))
    else:
        # no lane comes near this ground: the gates go where the ground itself lets a
        # person on, in the middle of the longest open run on a side
        gates = _open_runs(x0, z0, x1, z1, open_cells or ring, rng)
    return gates, open_cells, ring, near


def _open_runs(x0, z0, x1, z1, cells, rng):
    sides = {}
    for c in cells:
        sides.setdefault(_side_of(x0, z0, x1, z1, c[0], c[1]), []).append(c)
    order = sorted(sides, key=lambda s: (-len(sides[s]), s))
    picks = []
    for side in order[:2]:
        run = sorted(sides[side])
        picks.append(run[len(run) // 2 if len(run) < 4
                         else rng.randrange(1, len(run) - 1)])
        if len(picks) == 1 and rng.random() < 0.5:
            break
    return picks or [((x0 + x1) // 2, z1)]


def _gap_cells(x0, z0, x1, z1, gates, half):
    gap = set()
    for (gx, gz) in gates:
        side = _side_of(x0, z0, x1, z1, gx, gz)
        if side in ("north", "south"):
            for x in range(max(x0 + 1, gx - half), min(x1 - 1, gx + half) + 1):
                gap.add((x, gz))
        else:
            for z in range(max(z0 + 1, gz - half), min(z1 - 1, gz + half) + 1):
                gap.add((gx, z))
    return gap


def _ring_cells(x0, z0, x1, z1):
    out = []
    for x in range(x0, x1 + 1):
        out.append((x, z0))
        if z1 != z0:
            out.append((x, z1))
    for z in range(z0 + 1, z1):
        out.append((x0, z))
        if x1 != x0:
            out.append((x1, z))
    return out


def _hydrated(water, x, z):
    for wx in range(x - 4, x + 5):
        for wz in range(z - 4, z + 5):
            if (wx, wz) in water:
                return True
    return False


def _plan(rng, layout, px0, pz0, px1, pz1, axis, bed):
    """Role for every cell of the planting rectangle: till, water or path."""
    roles = {}
    for x in range(px0, px1 + 1):
        for z in range(pz0, pz1 + 1):
            roles[(x, z)] = "till"
    if layout == "quarters":
        mx, mz = (px0 + px1) // 2, (pz0 + pz1) // 2
        for x in range(px0, px1 + 1):
            for z in range(pz0, pz1 + 1):
                if x in (mx, mx + 1) or z in (mz, mz + 1):
                    roles[(x, z)] = "path"
    elif layout == "plots":
        for x in range(px0, px1 + 1):
            for z in range(pz0, pz1 + 1):
                if (x - px0) % 5 == 4 or (z - pz0) % 5 == 4:
                    roles[(x, z)] = "path"
    water = set()
    if layout == "plots":
        for x in range(px0, px1 + 1, 5):
            for z in range(pz0, pz1 + 1, 5):
                if roles.get((x, z)) == "till":
                    roles[(x, z)] = "water"
                    water.add((x, z))
    else:
        for cell in roles:
            if roles[cell] != "till":
                continue
            u = cell[0] if axis == "z" else cell[1]
            u0 = px0 if axis == "z" else pz0
            if (u - u0) % (bed + 1) == 0:
                roles[cell] = "water"
                water.add(cell)
    for x in range(px0, px1 + 1):
        for z in range(pz0, pz1 + 1):
            if roles[(x, z)] == "till" and not _hydrated(water, x, z):
                roles[(x, z)] = "water"
                water.add((x, z))
    return roles


def _border(b, voice, x0, z0, x1, z1, y, style, gap, rng):
    """A low wall or bank round the ground, broken where the lane comes in."""
    footing = b.block(voice["footing"], "full")
    post = b.block(voice["frame"], "full")
    cap = b.block(voice["trim"], "slab")
    rail = b.joinery(voice, "fence")
    corners = {(x0, z0), (x1, z0), (x0, z1), (x1, z1)}
    spacing = 4 if (x1 - x0) < 20 else 6
    for (x, z) in _ring_cells(x0, z0, x1, z1):
        b.place_block(x, y, z, footing)
        if (x, z) in gap:
            b.place_block(x, y + 1, z, "air")
            b.place_block(x, y + 2, z, "air")
            continue
        on_post = (x, z) in corners or ((x - x0) % spacing == 0 and (z - z0) % spacing == 0)
        if style == "fence":
            if on_post:
                b.place_block(x, y + 1, z, post)
                b.place_block(x, y + 2, z, post)
            else:
                b.place_block(x, y + 1, z, rail)
                b.place_block(x, y + 2, z, "air")
        elif style == "capped":
            b.place_block(x, y + 1, z, footing)
            b.place_block(x, y + 2, z, post if on_post else cap)
        else:
            b.place_block(x, y + 1, z, post if on_post else footing)
            b.place_block(x, y + 2, z, post if on_post else "air")


def _scarecrow(b, voice, x, y, z):
    post = b.block(voice["frame"], "full")
    head = b.block(voice["trim"], "full")
    b.place_block(x, y + 1, z, post)
    b.place_block(x, y + 2, z, post)
    b.place_block(x, y + 3, z, head)


def _furnish(b, voice, rng, cells, cx, cz, y, size):
    """The gear of a worked field, set down on the headland where it is used."""
    wanted = ["trough", "light", "store", "fodder", "bench", "shelf"]
    if size >= 24:
        wanted = ["well"] + wanted
    rng.shuffle(wanted)
    wanted = wanted[:2 if size < 14 else (4 if size < 24 else 6)]
    rng.shuffle(cells)
    taken = []
    for kind in wanted:
        for (x, z) in cells:
            if any(abs(x - ox) < 4 and abs(z - oz) < 4 for (ox, oz) in taken):
                continue
            mat = voice["trim"] if kind in ("store", "bench", "shelf") else voice["footing"]
            res = b.fitting(kind, x, y + 1, z, _face_to(x, z, cx, cz),
                            mat=mat, extent=2 if kind == "store" else 1,
                            room="field")
            if res and res.get("ok"):
                taken.append((x, z))
                break
    return taken


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    voice = part["voice"]
    w, d = x1 - x0 + 1, z1 - z0 + 1
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2

    crop = params.get("crop", "mixed")
    if crop not in _CROPS:
        crop = "mixed"
    layout = params.get("layout", "strips")
    if layout not in ("strips", "quarters", "plots"):
        layout = "strips"

    axis = "z" if (w >= d) == (rng.random() < 0.5) else "x"
    bed = rng.choice([3, 4])
    style = rng.choice(["bank", "capped", "fence"])
    fallow = rng.choice([0.0, 0.06, 0.12])

    gates, open_cells, ring, near = _gates(b, part, x0, z0, x1, z1, y, rng)
    gap = _gap_cells(x0, z0, x1, z1, gates, 1)
    gx, gz = gates[0]
    face = _side_of(x0, z0, x1, z1, gx, gz)

    # the headland: a walking margin inside the wall, all the way round
    ix0, iz0, ix1, iz1 = x0 + 1, z0 + 1, x1 - 1, z1 - 1
    hw = rng.choice([1, 2]) if min(w, d) >= 18 else 1
    px0, pz0, px1, pz1 = ix0 + hw, iz0 + hw, ix1 - hw, iz1 - hw
    if px1 - px0 < 3 or pz1 - pz0 < 3:
        hw = 1
        px0, pz0, px1, pz1 = ix0 + 1, iz0 + 1, ix1 - 1, iz1 - 1
    if layout == "quarters" and min(px1 - px0, pz1 - pz0) < 14:
        layout = "strips"
    if layout == "plots" and min(px1 - px0, pz1 - pz0) < 8:
        layout = "strips"

    roles = _plan(rng, layout, px0, pz0, px1, pz1, axis, bed)

    pave = b.block(voice["footing"], "full")
    trim_pave = b.block(voice["trim"], "full")
    band_crop = {}
    for x in range(ix0, ix1 + 1):
        for z in range(iz0, iz1 + 1):
            role = roles.get((x, z), "walk")
            if role == "walk" or role == "path":
                on_axis = (x == gx or z == gz)
                b.place_block(x, y, z, trim_pave if (role == "path" and on_axis) else pave)
                b.place_block(x, y + 1, z, "air")
                b.place_block(x, y + 2, z, "air")
            elif role == "water":
                b.place_block(x, y, z, "water")
                b.place_block(x, y + 1, z, "air")
            else:
                u = x if axis == "z" else z
                v = z if axis == "z" else x
                band = ((u - px0) // (bed + 1), (v - pz0) // 12 if layout == "plots" else 0)
                if band not in band_crop:
                    band_crop[band] = rng.choice(_CROPS[crop])
                name, ripe = band_crop[band]
                b.place_block(x, y, z, "farmland[moisture=7]")
                if rng.random() < fallow:
                    b.place_block(x, y + 1, z, "air")
                else:
                    age = ripe if rng.random() < 0.8 else max(0, ripe - rng.randint(1, 2))
                    b.place_block(x, y + 1, z, name + "[age=" + str(age) + "]")

    _border(b, voice, x0, z0, x1, z1, y, style, gap, rng)

    # the boarded deck under the eave has no eave here: a drying stage on the headland,
    # boards laid a step up, on the side of the field away from the gate
    if min(w, d) >= 16:
        run = max(4, min(w, d) // 3)
        if face == "south":
            dx0, dz0, dx1, dz1 = cx - run // 2, iz0, cx - run // 2 + run, iz0 + hw - 1
        elif face == "north":
            dx0, dz0, dx1, dz1 = cx - run // 2, iz1 - hw + 1, cx - run // 2 + run, iz1
        elif face == "east":
            dx0, dz0, dx1, dz1 = ix0, cz - run // 2, ix0 + hw - 1, cz - run // 2 + run
        else:
            dx0, dz0, dx1, dz1 = ix1 - hw + 1, cz - run // 2, ix1, cz - run // 2 + run
        dx0 = max(dx0, ix0)
        dz0 = max(dz0, iz0)
        dx1 = min(dx1, ix1)
        dz1 = min(dz1, iz1)
        if dx1 > dx0 and dz1 >= dz0:
            b.dais(dx0, dz0, dx1, dz1, y, voice["trim"])

    # scarecrows, standing in the crop where they can be seen
    posts = 1 + (1 if min(w, d) >= 24 else 0) + (1 if min(w, d) >= 40 else 0)
    tills = [c for c in roles if roles[c] == "till"
             and px0 + 1 < c[0] < px1 - 1 and pz0 + 1 < c[1] < pz1 - 1]
    rng.shuffle(tills)
    placed = []
    for (sx, sz) in tills:
        if len(placed) >= posts:
            break
        if any(abs(sx - ox) < 8 and abs(sz - oz) < 8 for (ox, oz) in placed):
            continue
        _scarecrow(b, voice, sx, y, sz)
        placed.append((sx, sz))

    # the gear, on the headland, clear of the gate
    head_cells = [(x, z) for x in range(ix0, ix1 + 1) for z in range(iz0, iz1 + 1)
                  if (x, z) not in roles
                  and (abs(x - gx) > 2 or abs(z - gz) > 2)
                  and (x, z) not in gap]
    _furnish(b, voice, rng, head_cells, cx, cz, y, min(w, d))

    # the gate has to be a gate: if nobody can walk in at one, open the wall where the
    # lane actually is
    if near:
        reached = False
        targets = []
        for (ex, ez) in gates:
            r = b.check_door(ex, y + 1, ez)
            if r is not None and r.get("ok"):
                reached = True
            elif r is not None:
                t = _xz(r.get("nearest_lane"))
                if t is not None:
                    targets.append(t)
        if not reached and targets:
            t = targets[0]
            spot = min(open_cells or ring,
                       key=lambda c: abs(c[0] - t[0]) + abs(c[1] - t[1]))
            for (ax, az) in _gap_cells(x0, z0, x1, z1, [spot], 1):
                b.place_block(ax, y, az, pave)
                b.place_block(ax, y + 1, az, "air")
                b.place_block(ax, y + 2, az, "air")

    b.check_attached()
    return {"layout": layout, "crop": crop, "axis": axis, "border": style}
