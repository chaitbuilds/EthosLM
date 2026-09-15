"""minka -- the farmhouse of this settlement.

A heavy exposed frame with pale boarding between it, the floor lifted clear of the
ground on its footing, and a roof that is by far the biggest thing about the building --
shallow off the ridge, steep at the top, eaves turned up over a boarded veranda, the
engawa, that runs the length of the long side under the overhang. this file names no
material.
"""

import random

FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "rural"

PARAMS = {
    "storeys": ("int", 1, 3),
    "plan": ("choice", ["hall", "wing", "outshot"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3x3 to 7x7 measured; above seven the wing and the outshot leave a room off the
    # stair. Measured by scripts/type_needs.py; the band is rounds/type-needs.json. The
    # pair is the pad site() hands build(), after siting's inset.
    "footprint": (3, 3, 7, 7),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

LANTERN = "lantern[hanging=true]"


# The boards, the posts and the screens were four block names in this file. They are the
# voice's now, and the engawa is the same engawa in any palette.

def _deck(b):
    return b.block(b.voice["trim"])


def _post(b):
    p = b.block(b.voice["frame"], "post")
    return f"{p}[axis=y]" if p.endswith(("_log", "_pillar", "_wood")) else p


def _screen(b):
    return b.joinery(b.voice, "fence")

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _purpose(label):
    """What this one is for. Fittings and a little of the massing follow from it."""
    l = (label or "").lower()
    if "barn" in l or "fold" in l or "byre" in l or "cattle" in l:
        return "barn"
    if "well" in l:
        return "wellhouse"
    if "gate" in l or "watch" in l:
        return "gate"
    if "landing" in l or "boat" in l or "tarn" in l or "quay" in l:
        return "landing"
    if "row" in l or "terrace" in l:
        return "row"
    return "house"


def _door_side(px0, pz0, px1, pz1, dx, dz):
    edges = {"west": abs(dx - px0), "east": abs(px1 - dx),
             "north": abs(dz - pz0), "south": abs(pz1 - dz)}
    return min(edges, key=lambda k: edges[k])


def _rect(px0, pz0, px1, pz1, m):
    return (px0 + m["west"], pz0 + m["north"], px1 - m["east"], pz1 - m["south"])


def _fits(r, minw, mind):
    return (r[2] - r[0] + 1) >= minw and (r[3] - r[1] + 1) >= mind


def _clamp_margins(px0, pz0, px1, pz1, m, minw=5, mind=5):
    """Give back margins until the footprint is big enough to be a building."""
    order = ["north", "south", "east", "west"]
    guard = 0
    while guard < 24:
        r = _rect(px0, pz0, px1, pz1, m)
        if _fits(r, minw, mind):
            return m
        w = r[2] - r[0] + 1
        d = r[3] - r[1] + 1
        shrunk = False
        for side in order:
            if m[side] <= 0:
                continue
            if side in ("east", "west") and w >= minw:
                continue
            if side in ("north", "south") and d >= mind:
                continue
            m[side] -= 1
            shrunk = True
            break
        if not shrunk:
            for side in order:
                if m[side] > 0:
                    m[side] -= 1
                    shrunk = True
                    break
        if not shrunk:
            return m
        guard += 1
    return m


def _roofspec(rnd, axis, purpose, storeys):
    """Irimoya or hip; shallow off the eave, steep at the ridge; turned-up eaves."""
    ends = rnd.choice([("irimoya", "irimoya"),
                       ("irimoya", "hip"),
                       ("hip", "irimoya"),
                       ("half-hip", "half-hip"),
                       ("hip", "hip")])
    if purpose in ("barn", "row"):
        ends = rnd.choice([("irimoya", "irimoya"), ("half-hip", "half-hip")])
    profile = rnd.choice([[(1, 2), (2, 1)],
                          [(1, 2), (2, 1)],
                          [(1, 2), (1, 1), (2, 1)],
                          [(1, 3), (2, 1)]])
    return {"style": "hip", "axis": axis, "profile": profile,
            "ends": ends, "eave": "upturned", "tiers": 1}


def _strip(side, fx0, fz0, fx1, fz1):
    """The one-block run of ground just outside the footprint on that side."""
    if side == "north":
        return [(x, fz0 - 1) for x in range(fx0, fx1 + 1)]
    if side == "south":
        return [(x, fz1 + 1) for x in range(fx0, fx1 + 1)]
    if side == "west":
        return [(fx0 - 1, z) for z in range(fz0, fz1 + 1)]
    return [(fx1 + 1, z) for z in range(fz0, fz1 + 1)]


def _longest_run(flags):
    best = (0, -1)
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    return best


def _engawa(b, sides, foot, plot, fy, eave_y, door, rnd):
    """The boarded veranda: one unbroken deck at floor level under the eave,
    a post at each end of it carrying the overhang."""
    fx0, fz0, fx1, fz1 = foot
    px0, pz0, px1, pz1 = plot
    laid = []
    for side in sides:
        cand = _strip(side, fx0, fz0, fx1, fz1)
        flags = []
        for (x, z) in cand:
            good = px0 <= x <= px1 and pz0 <= z <= pz1
            if good and door is not None:
                good = abs(x - door[0]) + abs(z - door[1]) > 1
            if good:
                good = (b.get_block(x, fy - 1, z) != "air"
                        or b.get_block(x, fy, z) != "air")
            flags.append(good)
        a, c = _longest_run(flags)
        if c - a + 1 < 4:
            continue
        run = cand[a:c + 1]
        top = min(fy + 3, max(fy + 1, eave_y - 1))
        for (x, z) in run:
            b.place_block(x, fy, z, _deck(b))
            for yy in range(fy + 1, top + 1):
                if b.get_block(x, yy, z) != "air":
                    b.place_block(x, yy, z, "air")
        for i in (0, len(run) - 1):
            x, z = run[i]
            for yy in range(fy + 1, max(fy + 2, eave_y)):
                b.place_block(x, yy, z, _post(b))
        walk = run[1:-1]
        # a lantern hung off the underside of the overhanging eave
        if eave_y - 2 > fy + 1 and len(walk) > 2:
            x, z = walk[len(walk) // 2]
            if b.get_block(x, eave_y, z) != "air" and \
                    b.get_block(x, eave_y - 1, z) == "air":
                b.place_block(x, eave_y - 1, z, LANTERN)
        laid.append((side, walk))
    return laid


def _screens(b, side, foot, fy, walk, door):
    """Wide low bays on the engawa wall -- screened, one left open to step out."""
    fx0, fz0, fx1, fz1 = foot
    horiz = side in ("north", "south")
    if horiz:
        fixed = fz0 if side == "north" else fz1
        lo, hi = fx0 + 1, fx1 - 1
    else:
        fixed = fx0 if side == "west" else fx1
        lo, hi = fz0 + 1, fz1 - 1
    avail = sorted(set((c[0] if horiz else c[1]) for c in walk))
    avail = [a for a in avail if lo <= a <= hi]
    aset = set(avail)
    if len(avail) < 2:
        return []
    mid = avail[len(avail) // 2]
    start = mid - 1 if (mid - 1) in aset else mid
    bays = []
    a = start
    while a >= lo:
        if a in aset and (a + 1) in aset:
            bays.append((a, a + 1))
        a -= 4
    a = start + 4
    while a <= hi:
        if a in aset and (a + 1) in aset:
            bays.append((a, a + 1))
        a += 4
    if not bays:
        return []
    open_bay = (start, start + 1)
    if open_bay not in bays:
        open_bay = bays[0]
    for bay in bays:
        for a in bay:
            cx, cz = (a, fixed) if horiz else (fixed, a)
            if door is not None and (cx, cz) == (door[0], door[1]):
                continue
            b.place_block(cx, fy + 1, cz, "air")
            b.place_block(cx, fy + 2, cz, "air")
            if bay != open_bay:
                b.place_block(cx, fy + 1, cz, _screen(b))
    return bays


def _stand_y(b, x, y, z):
    """rooms[] gives a floor level; find the cell a person stands in above it."""
    if b.get_block(x, y, z) == "air":
        return y
    return y + 1


def _wall_spots(rx0, rz0, rx1, rz1, rnd, keep=None):
    spots = []
    for x in range(rx0, rx1 + 1):
        spots.append((x, rz0, "south"))
        spots.append((x, rz1, "north"))
    for z in range(rz0, rz1 + 1):
        spots.append((rx0, z, "east"))
        spots.append((rx1, z, "west"))
    if keep:
        spots = [s for s in spots if (s[0], s[1]) not in keep] or spots
    rnd.shuffle(spots)
    return spots


def _inner_spots(rx0, rz0, rx1, rz1, rnd, keep=None):
    spots = []
    for x in range(rx0 + 1, rx1):
        for z in range(rz0 + 1, rz1):
            if keep and (x, z) in keep:
                continue
            spots.append((x, z, rnd.choice(["north", "south", "east", "west"])))
    rnd.shuffle(spots)
    return spots


def _step_stones(b, rooms, fy, avoid=None):
    """A block standing a whole step above the floor is a floor nobody can walk
    onto. Lay a half-step beside anything that ended up one block up."""
    for room in rooms:
        rx0, ry, rz0, rx1, rz1 = room
        sy = _stand_y(b, (rx0 + rx1) // 2, ry, (rz0 + rz1) // 2)
        tops = set()
        for x in range(rx0, rx1 + 1):
            for z in range(rz0, rz1 + 1):
                if (b.get_block(x, sy, z) != "air"
                        and b.get_block(x, sy + 1, z) == "air"
                        and b.get_block(x, sy + 2, z) == "air"):
                    tops.add((x, z))
        seen = set()
        for cell in sorted(tops):
            if cell in seen:
                continue
            comp = []
            stack = [cell]
            while stack:
                c = stack.pop()
                if c in seen or c not in tops:
                    continue
                seen.add(c)
                comp.append(c)
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (c[0] + d[0], c[1] + d[1])
                    if n in tops and n not in seen:
                        stack.append(n)
            done = False
            for (x, z) in comp:
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, nz = x + d[0], z + d[1]
                    if not (rx0 <= nx <= rx1 and rz0 <= nz <= rz1):
                        continue
                    if (nx, nz) in tops:
                        continue
                    if avoid and (nx, nz) in avoid:
                        continue
                    if (b.get_block(nx, sy, nz) == "air"
                            and b.get_block(nx, sy + 1, nz) == "air"
                            and b.get_block(nx, sy - 1, nz) != "air"):
                        b.place_block(nx, sy, nz, b.block(b.voice["trim"], "slab"))
                        done = True
                        break
                if done:
                    break
            if not done:
                # nowhere to step up from: this is clutter, not furniture
                for (x, z) in comp:
                    b.place_block(x, sy, z, "air")


def _put(b, kind, spots, y, rnd, tries=10, **kw):
    n = 0
    for (x, z, facing) in spots:
        if n >= tries:
            break
        n += 1
        r = b.fitting(kind, x, y, z, facing, **kw)
        if r and r.get("ok"):
            return (x, y, z)
    return None


AIRY = ("air", "cave_air", "void_air")


PASSABLE = ("door", "carpet", "sign", "banner", "button", "rail", "short_grass",
            "fern", "snow", "pressure_plate")


def _free(blk):
    """A person can be in this cell. A lantern or a pot on the floor cannot:
    they are something to climb, and the checker counts them as such."""
    if blk in AIRY:
        return True
    for p in PASSABLE:
        if p in blk:
            return True
    return False


def _feet(b, x, z, fy):
    """The height a person's feet reach standing in this column, or None."""
    for y in range(fy, fy + 4):
        here = b.get_block(x, y, z)
        if "slab" in here and "double" not in here:
            if _free(b.get_block(x, y + 1, z)) and _free(b.get_block(x, y + 2, z)):
                return y + 0.5
            continue
        below = b.get_block(x, y - 1, z)
        if below in AIRY or "water" in below:
            continue
        if _free(here) and _free(b.get_block(x, y + 1, z)):
            return float(y)
    return None


def _sweep(b, px0, pz0, px1, pz1, fy):
    """Walk the plot the way the checker does -- no jumping -- and answer for
    every standing place in it. A block a whole step above the floor with no
    way up onto it is a floor nobody can use: give it a half-step, or take it
    away again."""
    h = {}
    for x in range(px0 - 1, px1 + 2):
        for z in range(pz0 - 1, pz1 + 2):
            f = _feet(b, x, z, fy)
            if f is not None:
                h[(x, z)] = f
    seeds = [c for c in h
             if c[0] in (px0 - 1, px1 + 1) or c[1] in (pz0 - 1, pz1 + 1)]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        c = stack.pop()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in seen or n not in h:
                continue
            if h[n] - h[c] <= 0.55 and h[c] - h[n] <= 3.1:
                seen.add(n)
                stack.append(n)
    stray = [c for c in h if c not in seen
             and px0 <= c[0] <= px1 and pz0 <= c[1] <= pz1]
    for c in sorted(stray, key=lambda k: h[k]):
        if h[c] < fy + 1.9 or h[c] > fy + 3.1:
            continue
        stepped = False
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n not in seen or n not in h:
                continue
            if abs(h[n] - (h[c] - 1.0)) > 0.01:
                continue
            if b.get_block(n[0], int(h[n]), n[1]) in AIRY:
                b.place_block(n[0], int(h[n]), n[1], b.block(b.voice["trim"], "slab"))
                h[n] = h[n] + 0.5
                seen.add(c)
                stepped = True
                break
        if not stepped:
            for y in range(int(h[c]) - 1, fy, -1):
                if b.get_block(c[0], y, c[1]) not in AIRY:
                    b.place_block(c[0], y, c[1], "air")
    return stray


PLAN_FOR = {
    "barn": ["store", "fodder", "workbench", "bench", "shelf"],
    "wellhouse": ["store", "bench", "shelf", "table", "workbench"],
    "gate": ["bench", "store", "shelf", "table"],
    "landing": ["store", "workbench", "bench", "table", "shelf"],
    "row": ["table", "bench", "store", "shelf", "bed"],
    "house": ["table", "bench", "store", "shelf", "bed"],
}


BIG = {"well": 25, "hearth": 25, "oven": 20, "forge": 20, "anvil": 12,
       "trough": 12, "fodder": 12, "workbench": 9, "bed": 9, "table": 9,
       "store": 6, "bench": 4, "shelf": 4, "bookshelf": 6}


def _furnish(b, rooms, purpose, ridge_y, rnd, fy, keep=None):
    """A hearth in the biggest room on the ground, then what the building is for.
    A room is mostly floor: what goes in it is rationed by how big it is."""
    if not rooms:
        return
    ranked = sorted(rooms, key=lambda r: -((r[3] - r[0] + 1) * (r[4] - r[2] + 1)))
    wants = list(PLAN_FOR.get(purpose, PLAN_FOR["house"]))
    hearth_done = False
    for idx, room in enumerate(ranked):
        rx0, ry, rz0, rx1, rz1 = room
        w = rx1 - rx0 + 1
        d = rz1 - rz0 + 1
        if w < 2 or d < 2:
            continue
        area = w * d
        # a small room is a mat and a light: anything standing in it is in the way
        budget = 0 if area < 18 else max(1, area // 9)
        sy = _stand_y(b, (rx0 + rx1) // 2, ry, (rz0 + rz1) // 2)
        walls = _wall_spots(rx0, rz0, rx1, rz1, rnd, keep)
        inner = _inner_spots(rx0, rz0, rx1, rz1, rnd, keep)
        upper = ry > fy + 1
        room_word = "hall" if (idx == 0 and not upper) else (
            "store" if purpose in ("barn", "landing") and upper else "house")

        if (not hearth_done and not upper and idx == 0
                and w >= 5 and d >= 5 and area >= BIG["hearth"]):
            cx = (rx0 + rx1) // 2
            cz = (rz0 + rz1) // 2
            got = None
            if ridge_y and ridge_y > sy + 3:
                r = b.fitting("hearth", cx, sy, cz, "north",
                              mat=b.voice["footing"], room=room_word,
                              flue_to=ridge_y)
                got = r if (r and r.get("ok")) else None
            if got is None:
                got = _put(b, "hearth", [(cx, cz, "north")] + inner, sy, rnd,
                           tries=6, mat=b.voice["footing"], room=room_word)
            if got is not None:
                hearth_done = True
                budget -= 1

        shift = (idx * 2) % len(wants)
        order = wants[shift:] + wants[:shift]
        for kind in order:
            if budget <= 0:
                break
            if area < BIG.get(kind, 6):
                continue
            if kind == "store":
                got = _put(b, kind, walls, sy, rnd, tries=12, mat=b.voice["wall"],
                           extent=max(1, min(3, area // 8)), room=room_word)
            elif kind == "fodder":
                got = _put(b, kind, inner + walls, sy, rnd, tries=10,
                           mat=b.voice["footing"], room=room_word)
            else:
                got = _put(b, kind, walls, sy, rnd, tries=12, mat=b.voice["wall"],
                           room=room_word)
            if got is not None:
                budget -= 1
        if upper and area >= 18:
            _put(b, "bed", walls, sy, rnd, tries=12, mat=b.voice["wall"], room="house")
        if area >= 12:
            _put(b, "rug", inner, sy, rnd, tries=6, mat=b.block(b.voice["trim"]),
                 room=room_word)
        _put(b, "light", walls + inner, sy, rnd, tries=14, mat=b.block(b.voice["trim"]),
             room=room_word)


def build(b, part, seed, **params):
    rnd = random.Random((int(seed) * 2654435761) % 2147483647 + 17)
    storeys = max(1, min(3, int(params.get("storeys", 1))))
    plan = params.get("plan", "hall")

    px0 = min(part["x0"], part["x1"])
    px1 = max(part["x0"], part["x1"])
    pz0 = min(part["z0"], part["z1"])
    pz1 = max(part["z0"], part["z1"])
    fy = int(part["floor_y"])
    label = part.get("label")
    purpose = _purpose(label)
    W = px1 - px0 + 1
    D = pz1 - pz0 + 1

    door = part.get("door")
    if door is None:
        th = b.threshold(label)
        if th and th.get("door"):
            door = th["door"]
    if door is None:
        door = (px0, pz0)
    dx, dz = int(door[0]), int(door[1])
    dside = _door_side(px0, pz0, px1, pz1, dx, dz)

    if W > D:
        along = "x"
    elif D > W:
        along = "z"
    else:
        along = "x" if dside in ("west", "east") else "z"
    raxis = "z" if along == "x" else "x"
    perp = D if along == "x" else W
    alongdim = W if along == "x" else D

    cands = ["north", "south"] if along == "x" else ["west", "east"]
    cands = [s for s in cands if s != dside] or cands
    far = OPPOSITE[dside]

    layout = "one"
    if perp >= 8 and len(cands) == 2 and rnd.random() < 0.35:
        layout = "wrap"
    elif alongdim >= 9 and rnd.random() < 0.30:
        layout = "L"
    eng_sides = list(cands) if layout == "wrap" else [rnd.choice(cands)]
    if layout == "L" and far not in eng_sides:
        eng_sides = eng_sides + [far]

    # a block of plot is kept all round: the overhanging eave lands on it rather than
    # out on the lane, and on the long side it lands on the engawa
    m = {"north": 1, "south": 1, "east": 1, "west": 1}
    if alongdim >= 12 and rnd.random() < 0.45:
        m[far] = 2

    free = [s for s in cands if s not in eng_sides]
    out_side = free[0] if free else (far if m[far] == 0 else None)
    out_depth = 2 if perp <= 10 else 3
    m_out = dict(m)
    if out_side is not None:
        m_out[out_side] = m_out[out_side] + out_depth

    m = _clamp_margins(px0, pz0, px1, pz1, m, 5, 5)
    m_out = _clamp_margins(px0, pz0, px1, pz1, m_out, 5, 5)

    base = _rect(px0, pz0, px1, pz1, m)
    with_out = _rect(px0, pz0, px1, pz1, m_out)
    mat = dict(b.voice)
    roofspec = _roofspec(rnd, raxis, purpose, storeys)
    plain = {"style": rnd.choice(["hip", "hip", "gable"]), "axis": raxis,
             "pitch": (1, 2)}

    extras = {}
    if storeys >= 2:
        extras["dormers"] = rnd.choice([1, 2])

    # the wing: the far end of the long axis carried up under its own ridge, crossing
    # the main one where the two roofs meet
    wing = None
    wf = base
    wlen = 4 if alongdim < 16 else 5
    fx0, fz0, fx1, fz1 = base
    if along == "x" and (fx1 - fx0 + 1) >= wlen + 6:
        if far == "east":
            wing = (fx1 - wlen + 1, fz0, fx1, fz1)
            wf = (fx0, fz0, fx1 - wlen, fz1)
        else:
            wing = (fx0, fz0, fx0 + wlen - 1, fz1)
            wf = (fx0 + wlen, fz0, fx1, fz1)
    elif along == "z" and (fz1 - fz0 + 1) >= wlen + 6:
        if far == "south":
            wing = (fx0, fz1 - wlen + 1, fx1, fz1)
            wf = (fx0, fz0, fx1, fz1 - wlen)
        else:
            wing = (fx0, fz0, fx1, fz0 + wlen - 1)
            wf = (fx0, fz0 + wlen, fx1, fz1)

    tries = []
    if plan == "wing" and wing is not None:
        tries.append((wf, dict(extras, wing=wing), roofspec, storeys, base))
    if plan == "outshot" and out_side is not None:
        tries.append((with_out,
                      dict(extras, outshot={"side": out_side, "depth": out_depth}),
                      roofspec, storeys, with_out))
    tries.append((base, dict(extras), roofspec, storeys, base))
    tries.append((base, {}, roofspec, storeys, base))
    tries.append((base, {}, plain, storeys, base))
    if storeys > 1:
        tries.append((base, {}, plain, storeys - 1, base))
    tries.append((base, {}, {"style": "hip", "axis": raxis}, 1, base))


    res = None
    used = base
    used_storeys = storeys
    for (foot, kw, rspec, st, outer) in tries:
        r = b.building(label, foot[0], foot[1], foot[2], foot[3],
                       storeys=st, roof=rspec, mat=mat, **kw)
        if r and r.get("ok"):
            res = r
            used = outer
            used_storeys = st
            break
    if res is None:
        return {"ok": False}


    eave_y = res.get("eave_y") or (fy + 4 * used_storeys)
    ridge_y = res.get("ridge_y") or (eave_y + 4)
    rooms = res.get("rooms") or []

    # the doorway and the ground in front of it are circulation, not furniture. **The
    # shell's own door**, where the shell hung it: on a pad the hall does not fill, that
    # is a few columns off the cell the lane reserved, and an engawa or a screen laid
    # against the reserved cell walls the real one up.
    door_cell = (dx, dz)
    d0 = res.get("door")
    if isinstance(d0, (list, tuple)) and len(d0) == 3:
        door_cell = (int(d0[0]), int(d0[2]))
    elif isinstance(d0, (list, tuple)) and len(d0) == 2:
        door_cell = (int(d0[0]), int(d0[1]))
    # the engawa, and the wide screened bays that open onto it
    laid = _engawa(b, eng_sides, used, (px0, pz0, px1, pz1), fy, eave_y,
                   door_cell, rnd)
    keep = set()
    for ddx in range(-2, 3):
        for ddz in range(-2, 3):
            if abs(ddx) + abs(ddz) <= 2:
                keep.add((door_cell[0] + ddx, door_cell[1] + ddz))
                keep.add((dx + ddx, dz + ddz))
    for (s, walk) in laid:
        _screens(b, s, used, fy, walk, door_cell)
        ux0, uz0, ux1, uz1 = used
        if s == "north":
            keep.update((x, uz0 + 1) for x in range(ux0, ux1 + 1))
        elif s == "south":
            keep.update((x, uz1 - 1) for x in range(ux0, ux1 + 1))
        elif s == "west":
            keep.update((ux0 + 1, z) for z in range(uz0, uz1 + 1))
        else:
            keep.update((ux1 - 1, z) for z in range(uz0, uz1 + 1))

    near_door = set()
    for ddx in range(-1, 2):
        for ddz in range(-1, 2):
            if abs(ddx) + abs(ddz) <= 1:
                near_door.add((door_cell[0] + ddx, door_cell[1] + ddz))

    _furnish(b, rooms, purpose, ridge_y, rnd, fy, keep)
    _step_stones(b, rooms, fy, near_door)
    _sweep(b, px0, pz0, px1, pz1, fy)
    _sweep(b, px0, pz0, px1, pz1, fy)

    d = res.get("door")
    if isinstance(d, (list, tuple)) and len(d) == 3:
        cd = b.check_door(int(d[0]), int(d[1]), int(d[2]))
    elif isinstance(d, (list, tuple)) and len(d) == 2:
        cd = b.check_door(int(d[0]), fy + 1, int(d[1]))
    else:
        cd = b.check_door(dx, fy + 1, dz)

    sv = b.seal_voids(px0, pz0, px1, pz1, b.block(b.voice["wall"]), max_cells=256)
    cw = b.check_walkable(label)
    ca = b.check_attached()
    return {"ok": True, "purpose": purpose, "storeys": used_storeys}
