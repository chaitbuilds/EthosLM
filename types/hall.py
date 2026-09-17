import math
import random

FORM = "civic"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian. which is civic -- is made of halls. nothing the
#: type builds changes.
ROLE = "civic"

PARAMS = {
    "dormers": ("int", 1, 2),
    "use": ("choice", ["moot", "market", "refectory"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 5x5 and up measured; at four across, its roof puts a tread where the linter will
    # not have one. Measured by scripts/type_needs.py; the band is rounds/type-
    # needs.json. The pair is the pad site() hands build(), after siting's inset.
    "footprint": (4, 4, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

# What used to be six module constants of block names. A type is a form and the palette
# is the settlement's, so each of these is now a question put to the voice the part
# arrived with, and the same hall stands in any of them.

def _axial(b, block, axis):
    """`block` turned along `axis`, where the block is one that has an axis."""
    return b.axial(block, axis)


def _wall_b(b):
    return b.block(b.voice["wall"])


def _pier_b(b):
    return b.block(b.voice["wall"], "accent")


def _floor_b(b):
    return b.block(b.voice["floor"])


def _post_b(b):
    return _axial(b, b.block(b.voice["frame"], "post"), "y")


def _trim(b, axis):
    return _axial(b, b.block(b.voice["trim"], "bare"), axis)


def _slab_top(b):
    return b.block(b.voice["floor"], "slab") + "[type=top]"

PANE_B = "glass_pane"

SIDES = ("north", "south", "east", "west")
INWARD = {"north": "south", "south": "north", "east": "west", "west": "east"}
STEP = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
SKIP = ("air", "glass", "door", "stairs", "slab", "lantern", "torch", "water",
        "ladder", "fence", "pane", "candle", "bars", "chain", "sign", "carpet",
        "bed", "barrel", "campfire", "smoker", "furnace", "table", "wall")


def _solid(b, x, y, z):
    bid = b.get_block(x, y, z)
    if not bid:
        return False
    for s in SKIP:
        if s in bid:
            return False
    return True


def _posts(a0, a1):
    span = a1 - a0
    n = max(1, int(math.ceil(span / 5.0)))
    out = set()
    for i in range(n + 1):
        out.add(a0 + int(round(i * span / float(n))))
    return sorted(out)


def _door_side(px0, pz0, px1, pz1, door, facing):
    dx, dz = door[0], door[1]
    if dx <= px0:
        return "west"
    if dx >= px1:
        return "east"
    if dz <= pz0:
        return "north"
    if dz >= pz1:
        return "south"
    return INWARD.get(facing, "north")


def _rect(px0, pz0, px1, pz1, ins):
    return (px0 + ins["west"], pz0 + ins["north"],
            px1 - ins["east"], pz1 - ins["south"])


def _fits(rect):
    """Big enough that the library will carry its own straight flight: it wants
    six columns along the flight and three across."""
    w = rect[2] - rect[0] + 1
    d = rect[3] - rect[1] + 1
    return max(w, d) >= 8 and min(w, d) >= 5


def _plan(px0, pz0, px1, pz1, dside):
    W = px1 - px0 + 1
    D = pz1 - pz0 + 1
    ins = dict((s, 1) for s in SIDES)
    if _fits(_rect(px0, pz0, px1, pz1, ins)):
        return _rect(px0, pz0, px1, pz1, ins), ins, True
    order = [INWARD[dside]]
    order += [s for s in SIDES if s != dside and s != INWARD[dside]]
    for s in order:
        ins[s] = 0
        if _fits(_rect(px0, pz0, px1, pz1, ins)):
            return _rect(px0, pz0, px1, pz1, ins), ins, True
    ins = dict((s, 0) for s in SIDES)
    if (W if dside in ("east", "west") else D) >= 6:
        ins[dside] = 1
    return _rect(px0, pz0, px1, pz1, ins), ins, False


def _shrink(px0, pz0, px1, pz1, ins, dside, rng):
    """Seed variation in the footprint itself: a wall pulled in where there is
    slack for it, never the wall the door is in."""
    for pair in (("west", "east"), ("north", "south")):
        for s in pair:
            if s == dside:
                continue
            cut = rng.choice([0, 0, 1, 2])
            while cut > 0:
                trial = dict(ins)
                trial[s] = trial[s] + cut
                if _fits(_rect(px0, pz0, px1, pz1, trial)):
                    ins = trial
                    break
                cut -= 1
    return ins


def _face_line(rect, side):
    x0, z0, x1, z1 = rect
    if side == "north":
        return [(x, z0) for x in range(x0, x1 + 1)]
    if side == "south":
        return [(x, z1) for x in range(x0, x1 + 1)]
    if side == "west":
        return [(x0, z) for z in range(z0, z1 + 1)]
    return [(x1, z) for z in range(z0, z1 + 1)]


def _cells_of(obj, out, depth=0):
    if depth > 4:
        return out
    if isinstance(obj, dict):
        if "x" in obj and "z" in obj:
            out.append((int(obj["x"]), int(obj.get("y", 0)), int(obj["z"])))
        for v in obj.values():
            _cells_of(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 3 and all(isinstance(v, int) for v in obj):
            out.append((obj[0], obj[1], obj[2]))
        else:
            for v in obj:
                _cells_of(v, out, depth + 1)
    return out


def _frame(b, rect, ylo, yhi, band_y):
    """Exposed frame over the rendered panel: a post on every corner, no run of wall longer
    than five without one, a plate along the head of the storey. Painted only onto wall
    already standing, so glazing and doorways stay open.
    """
    x0, z0, x1, z1 = rect
    cols = set()
    for x in _posts(x0, x1):
        cols.add((x, z0))
        cols.add((x, z1))
    for z in _posts(z0, z1):
        cols.add((x0, z))
        cols.add((x1, z))
    for (x, z) in sorted(cols):
        for y in range(ylo, yhi + 1):
            if _solid(b, x, y, z):
                b.place_block(x, y, z, _post_b(b))
    for side in SIDES:
        run = "x" if side in ("north", "south") else "z"
        for (x, z) in _face_line(rect, side):
            if (x, z) in cols:
                continue
            if _solid(b, x, band_y, z):
                b.place_block(x, band_y, z, _trim(b, run))


def _dogleg(b, label, ix0, iz0, ix1, iz1, fy, dcell, dside):
    """Two short flights round a half-landing, for an inside too small to take
    a straight one. The landing is a pier across the end of the room furthest
    from the door, and each flight runs against a wall, so every tread has
    something beside it holding it up."""
    w = ix1 - ix0 + 1
    d = iz1 - iz0 + 1
    walk_z = dside in ("north", "south")
    order = [("z", d, w), ("x", w, d)]
    if not walk_z:
        order.reverse()
    axis = None
    for (a, along, across) in order:
        if along >= 4 and across >= 2:
            axis = a
            break
    if axis is None:
        return False, [], None
    if axis == "z":
        u0, u1, v0, v1 = iz0, iz1, ix0, ix1
        du = dcell[2] if dcell else u0
        dv = dcell[0] if dcell else v0
    else:
        u0, u1, v0, v1 = ix0, ix1, iz0, iz1
        du = dcell[0] if dcell else u0
        dv = dcell[2] if dcell else v0

    def cell(u, v):
        return (v, u) if axis == "z" else (u, v)

    if abs(du - u0) <= abs(du - u1):
        land, t1, t2, step = u1, u1 - 2, u1 - 1, 1
    else:
        land, t1, t2, step = u0, u0 + 2, u0 + 1, -1
    if abs(dv - v0) <= abs(dv - v1):
        va, vb = v1, v0
    else:
        va, vb = v0, v1
    up = {("x", 1): "east", ("x", -1): "west",
          ("z", 1): "south", ("z", -1): "north"}[(axis, step)]

    pa, pb = cell(land, v0), cell(land, v1)
    b.place_cuboid(pa[0], fy + 1, pa[1], pb[0], fy + 2, pb[1], _pier_b(b))
    for v in range(v0, v1 + 1):
        x, z = cell(land, v)
        for y in (fy + 3, fy + 4):
            b.place_block(x, y, z, "air")
    fa = cell(t1, va)
    fb = cell(t2, vb)
    r1 = b.flight(label, fa[0], fa[1], fy, fy + 2, up)
    r2 = b.flight(label, fb[0], fb[1], fy + 2, fy + 4, INWARD[up])
    used = [cell(land, v) for v in range(v0, v1 + 1)]
    for u in (t1, t2, land - step * 3):
        used.append(cell(u, va))
        used.append(cell(u, vb))
    return bool(r1.get("ok") and r2.get("ok")), used, cell(land - step * 3, vb)


def _oriel(b, rect, side, door, y, width):
    """The signature move by hand, for when the library's own will not go: the
    first-floor bay carried out over the entrance on a corbel of slabs."""
    dx, dz = STEP[side]
    line = _face_line(rect, side)
    if side in ("north", "south"):
        c = min(max(door[0], rect[0] + 1), rect[2] - 1)
        cells = [(c + i, line[0][1])
                 for i in range(-(width // 2), width - width // 2)]
    else:
        c = min(max(door[1], rect[1] + 1), rect[3] - 1)
        cells = [(line[0][0], c + i)
                 for i in range(-(width // 2), width - width // 2)]
    cells = [q for q in cells if q in line]
    if len(cells) < 2:
        return False
    out = [(x + dx, z + dz) for (x, z) in cells]
    for (x, z) in out:
        b.place_block(x, y - 1, z, _slab_top(b))
        b.place_block(x, y, z, _floor_b(b))
        b.place_block(x, y + 4, z, _floor_b(b))
    for i, (x, z) in enumerate(out):
        edge = (i == 0 or i == len(out) - 1)
        for yy in range(y + 1, y + 4):
            b.place_block(x, yy, z, _post_b(b) if edge else PANE_B)
    run = "x" if side in ("north", "south") else "z"
    for (x, z) in cells:
        for yy in (y + 1, y + 2):
            b.place_block(x, yy, z, "air")
        b.place_block(x, y + 3, z, _trim(b, run))
    return True


def _connect(b, dc, side, rect):
    """The door the library hangs is in the wall it built; make sure the way
    from it into the room is open, whatever else has gone in since."""
    dx, dz = STEP[side]
    x, y, z = dc[0], dc[1], dc[2]
    cut = []
    for i in range(1, 4):
        cx, cz = x - dx * i, z - dz * i
        for yy in (y, y + 1):
            bid = b.get_block(cx, yy, cz) or "air"
            if "door" in bid:
                continue
            if bid != "air":
                b.place_block(cx, yy, cz, "air")
                cut.append((cx, yy, cz))
        if rect[0] < cx < rect[2] and rect[1] < cz < rect[3]:
            break
    return cut


def _unfloat(b, at, px0, pz0, px1, pz1, keep):
    """Anything left hanging on nothing inside the plot, taken down again --
    never a tread, which is held by the wall it runs against."""
    gone = []
    for item in (at.get("floating") or []):
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 6):
            continue
        if int(item.get("cells") or 99) > 12:
            continue
        if "stairs" in str(item.get("example") or ""):
            continue
        hit = [(x, z) for x in range(int(bb[0]), int(bb[3]) + 1)
               for z in range(int(bb[2]), int(bb[5]) + 1) if (x, z) in keep]
        if hit:
            continue
        if bb[0] < px0 - 1 or bb[3] > px1 + 1:
            continue
        if bb[2] < pz0 - 1 or bb[5] > pz1 + 1:
            continue
        for x in range(int(bb[0]), int(bb[3]) + 1):
            for y in range(int(bb[1]), int(bb[4]) + 1):
                for z in range(int(bb[2]), int(bb[5]) + 1):
                    b.place_block(x, y, z, "air")
                    gone.append((x, y, z))
    return gone


FLAT = ("rug", "light")


def _flood(free, seed):
    if seed is None:
        stack = sorted(free)[:1]
    else:
        stack = [q for q in free
                 if q == seed or abs(q[0] - seed[0]) + abs(q[1] - seed[1]) == 1]
    seen = set(stack)
    while stack:
        x, z = stack.pop()
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, z + dz)
            if q in free and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def _reserve(free, start, targets):
    """The way from the door to the foot of the stair, kept clear before a
    stick of furniture goes in. Everything else can be argued about."""
    if start not in free:
        return set()
    prev = {start: None}
    q = [start]
    i = 0
    while i < len(q):
        (x, z) = q[i]
        i += 1
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, z + dz)
            if n in free and n not in prev:
                prev[n] = (x, z)
                q.append(n)
    out = set()
    for t in targets:
        c = t if t in prev else None
        while c is not None:
            out.add(c)
            c = prev[c]
    return out


def _open(ctx, take):
    """Would the room still be all of one piece with those cells furnished? A
    room you can only cross by climbing over the table is a room half laid."""
    free = ctx["cells"] - ctx["placed"] - ctx["solid"]
    before = _flood(free, ctx.get("seed"))
    for c in take:
        if c not in before:
            return False
    grow = [set(take)]
    for c in take:
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (c[0] + dx, c[1] + dz)
            if q in before and q not in take:
                grow.append(set(take) | set([q]))
    for gone in grow:
        if len(_flood(free - gone, ctx.get("seed"))) < len(before) - len(gone):
            return False
    return True


def _fit(b, kind, cells, facing, ctx, run=2, **kw):
    """Put one piece of equipment somewhere it fits and blocks nothing. `run`
    is how many cells along the wall it may take up, so a rank of barrels
    cannot creep across the way to the stair."""
    flat = kind in FLAT
    if flat:
        run = 1
    for i in range(len(cells)):
        span = cells[i:i + run]
        if len(span) < run:
            break
        take = [(c[0], c[2]) for c in span]
        bad = [q for q in take if q in ctx["avoid"] or q in ctx["placed"]
               or q in ctx["solid"]]
        if bad:
            continue
        if not flat and not _open(ctx, take):
            continue
        c = span[0]
        r = b.fitting(kind, c[0], c[1], c[2], facing, **kw)
        if r and r.get("ok"):
            if not flat:
                for q in take:
                    ctx["placed"].add(q)
                for q in _cells_of(r.get("cells"), []):
                    if (q[0], q[2]) in ctx["cells"]:
                        ctx["placed"].add((q[0], q[2]))
            return r
    return None


def _line(x0, cy, z0, x1, z1, side, back=0):
    if side == "north":
        return [(x, cy, z0 + back) for x in range(x0, x1 + 1)]
    if side == "south":
        return [(x, cy, z1 - back) for x in range(x0, x1 + 1)]
    if side == "west":
        return [(x0 + back, cy, z) for z in range(z0, z1 + 1)]
    return [(x1 - back, cy, z) for z in range(z0, z1 + 1)]


def _all(x0, cy, z0, x1, z1):
    return [(x, cy, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]


def _mid(x0, cy, z0, x1, z1):
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    out = [(cx, cy, cz)]
    out += [c for c in _all(x0, cy, z0, x1, z1) if c[0] != cx or c[2] != cz]
    return out


def _ground(b, use, x0, cy, z0, x1, z1, rng, dside, ctx, big):
    """The ground floor is the hall, and what the hall is for is the whole
    difference between one of these and the next."""
    far = INWARD[dside]
    cross = ["north", "south"] if dside in ("east", "west") else ["east", "west"]
    rng.shuffle(cross)
    if use == "moot":
        if big:
            x, z = x0, z0
            xx, zz = x1, z1
            if far == "north":
                zz = z0 + 1
            elif far == "south":
                z = z1 - 1
            elif far == "west":
                xx = x0 + 1
            else:
                x = x1 - 1
            strip = [(a, c) for a in range(x, xx + 1) for c in range(z, zz + 1)]
            bad = [q for q in strip
                   if q in ctx["avoid"] or q in ctx["placed"]]
            if not bad and len(ctx["cells"]) - len(strip) >= 12:
                b.dais(x, z, xx, zz, cy - 1, b.voice["frame"])
                for q in strip:
                    ctx["placed"].add(q)
                top = {"cells": set(strip), "placed": set(), "solid": set(),
                       "avoid": set(), "seed": None}
                _fit(b, "table", _line(x, cy + 1, z, xx, zz, far), INWARD[far],
                     top, mat=b.voice["frame"], room="hall")
        for s in cross:
            _fit(b, "bench", _line(x0, cy, z0, x1, z1, s), INWARD[s], ctx,
                 mat=b.voice["frame"], room="hall")
            _fit(b, "bench", _line(x0, cy, z0, x1, z1, s)[::-1], INWARD[s],
                 ctx, mat=b.voice["frame"], room="hall")
        _fit(b, "store", _line(x0, cy, z0, x1, z1, far)[::-1], INWARD[far],
             ctx, run=3, mat=b.voice["frame"], extent=2, room="hall")
    elif use == "market":
        _fit(b, "store", _line(x0, cy, z0, x1, z1, cross[0]), INWARD[cross[0]],
             ctx, run=4, mat=b.voice["frame"], extent=3, room="store")
        for s in cross:
            _fit(b, "table", _line(x0, cy, z0, x1, z1, s), INWARD[s], ctx,
                 mat=b.voice["frame"], room="store")
        _fit(b, "trough", _line(x0, cy, z0, x1, z1, far), INWARD[far], ctx,
             mat=b.voice["footing"], room="store")
        _fit(b, "shelf", _line(x0, cy, z0, x1, z1, far)[::-1], INWARD[far],
             ctx, mat=b.voice["frame"], room="store")
        _fit(b, "light", _mid(x0, cy, z0, x1, z1), "north", ctx, room="store")
        return
    else:
        h = _fit(b, "hearth", _line(x0, cy, z0, x1, z1, far), INWARD[far],
                 ctx, mat=b.voice["footing"], room="hall")
        if h is None:
            _fit(b, "oven", _line(x0, cy, z0, x1, z1, far)[::-1], INWARD[far],
                 ctx, mat=b.voice["footing"], room="hall")
        for s in cross:
            _fit(b, "table", _line(x0, cy, z0, x1, z1, s), INWARD[s], ctx,
                 mat=b.voice["frame"], room="hall")
            _fit(b, "bench", _line(x0, cy, z0, x1, z1, s)[::-1], INWARD[s],
                 ctx, mat=b.voice["frame"], room="hall")
        _fit(b, "store", _line(x0, cy, z0, x1, z1, cross[0]), INWARD[cross[0]],
             ctx, run=3, mat=b.voice["frame"], extent=2, room="hall")
    _fit(b, "light", _mid(x0, cy, z0, x1, z1), "north", ctx, room="hall")


def _upper(b, use, x0, cy, z0, x1, z1, rng, lvl, dside, ctx):
    """Over the hall, the chamber it needs: beds where people sleep over the
    work, shelves where they keep the record of it."""
    far = INWARD[dside]
    cross = ["north", "south"] if dside in ("east", "west") else ["east", "west"]
    rng.shuffle(cross)
    room = "house"
    if use == "moot":
        for s in cross:
            _fit(b, "bookshelf", _line(x0, cy, z0, x1, z1, s), INWARD[s],
                 ctx, mat=b.voice["frame"], room="house")
        _fit(b, "table", _mid(x0, cy, z0, x1, z1), "north", ctx,
             mat=b.voice["frame"], room="house")
        _fit(b, "bench", _line(x0, cy, z0, x1, z1, far), INWARD[far], ctx,
             mat=b.voice["frame"], room="house")
    elif use == "market":
        _fit(b, "store", _line(x0, cy, z0, x1, z1, cross[0]), INWARD[cross[0]],
             ctx, run=3, mat=b.voice["frame"], extent=2, room="store")
        _fit(b, "bed", _line(x0, cy, z0, x1, z1, far), INWARD[far], ctx,
             mat=b.voice["frame"], room="house")
        _fit(b, "shelf", _line(x0, cy, z0, x1, z1, cross[1]), INWARD[cross[1]],
             ctx, mat=b.voice["frame"], room="house")
        room = "store" if lvl > 1 else "house"
    else:
        for s in cross:
            _fit(b, "bed", _line(x0, cy, z0, x1, z1, s), INWARD[s], ctx,
                 mat=b.voice["frame"], room="house")
        _fit(b, "shelf", _line(x0, cy, z0, x1, z1, far), INWARD[far], ctx,
             mat=b.voice["frame"], room="house")
        _fit(b, "rug", _mid(x0, cy, z0, x1, z1), "north", ctx,
             mat=b.voice["frame"], room="house")
    _fit(b, "light", _mid(x0, cy, z0, x1, z1), "north", ctx, room=room)


def _read_room(b, x0, y, z0, x1, z1):
    """What is already in this room, read out of the world rather than
    guessed: where the floor is open to the storey below (a stairwell), and
    which cells something is already standing in. The cells around a stairwell
    are the landing, and nothing of mine goes on a landing."""
    holes = set()
    taken = set()
    treads = set()
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            under = b.get_block(x, y, z) or "air"
            here = b.get_block(x, y + 1, z) or "air"
            if "air" in under or "water" in under:
                holes.add((x, z))
            if "air" not in here:
                taken.add((x, z))
            if "stairs" in here or "stairs" in under or "slab" in here:
                treads.add((x, z))
    edge = set()
    for (x, z) in treads:
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, z + dz)
            if x0 <= q[0] <= x1 and z0 <= q[1] <= z1 and q not in treads:
                edge.add(q)
    for (x, z) in holes:
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, z + dz)
            if x0 <= q[0] <= x1 and z0 <= q[1] <= z1 and q not in holes:
                edge.add(q)
    return holes, taken, edge


def _floor_check(b, x0, z0, x1, z1, y, seeds, fill, block):
    """Every cell of this storey a person could stand in, and whether they could walk to it
    from the way in. What cannot be reached is not floor: carry the pier up through it
    rather than leave a pocket nobody can enter.
    """
    ok = set()
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            under = b.get_block(x, y, z) or "air"
            if "air" in under or "water" in under:
                continue
            if [k for k in KIT if k in under]:
                continue
            if not _thru(b.get_block(x, y + 1, z) or "air"):
                continue
            if not _thru(b.get_block(x, y + 2, z) or "air"):
                continue
            ok.add((x, z))
    start = [q for q in seeds if q in ok]
    seen = set(start)
    stack = list(start)
    while stack:
        (x, z) = stack.pop()
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, z + dz)
            if q in ok and q not in seen:
                seen.add(q)
                stack.append(q)
    left = sorted(ok - seen)
    if fill and len(left) <= 10:
        for (x, z) in left:
            for yy in (y + 1, y + 2, y + 3):
                b.place_block(x, yy, z, block)
    return left, len(ok)


def _thru(bid):
    return ("air" in bid) or ("door" in bid) or ("carpet" in bid)


KIT = ("lantern", "barrel", "bed", "carpet", "campfire", "furnace", "smoker",
       "chest", "table", "cauldron", "composter", "bookshelf", "torch",
       "candle", "anvil", "pot", "loom", "fence", "wall", "pane", "bars",
       "sign", "leaves", "rail", "grass", "fern", "flower", "snow")


def _stand_y(b, x, z, fy):
    for y in range(fy + 2, fy - 9, -1):
        bid = b.get_block(x, y, z) or "air"
        if "air" in bid or "water" in bid:
            continue
        if [k for k in KIT if k in bid]:
            return None
        if not _thru(b.get_block(x, y + 1, z) or "air"):
            continue
        if not _thru(b.get_block(x, y + 2, z) or "air"):
            continue
        return y
    return None


def _skirt(b, px0, pz0, px1, pz1, fy, dc, dside, block, cap, keep):
    """Ground inside the plot that the finished building has walled off from
    the way in is not ground anybody stands on: carry the footing course over
    it instead."""
    m = 6
    hh = {}
    half = set()
    for x in range(px0 - m, px1 + m + 1):
        for z in range(pz0 - m, pz1 + m + 1):
            y = _stand_y(b, x, z, fy)
            if y is not None:
                hh[(x, z)] = y
                bid = b.get_block(x, y, z) or ""
                if "slab" in bid or "stairs" in bid:
                    half.add((x, z))
    seeds = []
    nl = b.nearest_lane(dc[0], dc[2])
    if isinstance(nl, dict) and nl.get("x") is not None:
        if (int(nl["x"]), int(nl["z"])) in hh:
            seeds.append((int(nl["x"]), int(nl["z"])))
    if not seeds:
        ox, oz = STEP[dside]
        seeds = [q for q in ((dc[0] + ox, dc[2] + oz),
                             (dc[0] + 2 * ox, dc[2] + 2 * oz)) if q in hh]
    if not seeds:
        return [], hh, set()
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        (x, z) = stack.pop()
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (x + dx, z + dz)
            if q not in hh or q in seen:
                continue
            rise = hh[q] - hh[(x, z)]
            if rise > 1 or rise < -1 or (rise == 1 and q not in half):
                continue
            seen.add(q)
            stack.append(q)
    left = [q for q in sorted(hh)
            if px0 <= q[0] <= px1 and pz0 <= q[1] <= pz1 and q not in seen
            and not (keep[0] <= q[0] <= keep[2] and keep[1] <= q[1] <= keep[3])]
    if len(left) <= cap:
        for (x, z) in left:
            if hh[(x, z)] + 1 >= fy:
                b.place_block(x, hh[(x, z)] + 1, z, block)
                b.place_block(x, hh[(x, z)] + 2, z, "air")
    return left, hh, seen


def _door_cell(dr, door, fy):
    if isinstance(dr, dict):
        x, y, z = dr.get("x"), dr.get("y"), dr.get("z")
        if x is not None and z is not None:
            return (int(x), int(y if y is not None else fy + 1), int(z))
    if isinstance(dr, (list, tuple)):
        n = [v for v in dr if isinstance(v, (int, float))]
        if len(n) == 3:
            return (int(n[0]), int(n[1]), int(n[2]))
        if len(n) == 2:
            return (int(n[0]), fy + 1, int(n[1]))
    if door:
        return (int(door[0]), fy + 1, int(door[1]))
    return None


def build(b, part, seed, storeys=2, dormers=1, use=None):
    rng = random.Random(seed)
    px0, pz0 = int(part["x0"]), int(part["z0"])
    px1, pz1 = int(part["x1"]), int(part["z1"])
    fy = int(part["floor_y"])
    label = part.get("label")
    door = part.get("door") or (px0, pz0)
    facing = part.get("facing") or "south"
    dside = _door_side(px0, pz0, px1, pz1, door, facing)
    if use is None:
        use = ["moot", "market", "refectory"][seed % 3]
    st = 2
    nd = max(1, min(2, int(dormers or 1)))

    rect, ins, auto = _plan(px0, pz0, px1, pz1, dside)
    if auto:
        ins = _shrink(px0, pz0, px1, pz1, ins, dside, rng)
        rect = _rect(px0, pz0, px1, pz1, ins)
    else:
        st = 2
    w = rect[2] - rect[0] + 1
    d = rect[3] - rect[1] + 1
    roofspec = {"style": "gable", "axis": "z" if d <= w else "x",
                "pitch": rng.choice([(2, 1), (2, 1), (3, 1)]),
                "overhang": 2 if min(ins.values()) >= 1 else 1}
    if rng.random() < 0.34:
        roofspec["ends"] = rng.choice([("gable", "half-hip"),
                                       ("half-hip", "gable")])
    cl = _face_line(rect, INWARD[dside])
    stair = "auto" if auto else "none"
    base = {"mat": dict(b.voice), "openings": "rhythm", "stair": stair, "porch": True,
            "chimney": cl[len(cl) // 2], "dormers": nd, "brackets": True,
            "flashing": True, "oriel": (dside, 1 if st == 2 else
                                        rng.choice([1, 2]))}
    if min(ins.values()) >= 1:
        base["jetty"] = 1
    else:
        cross = [q for q in SIDES if q != dside and q != INWARD[dside]]
        base["oriel"] = (cross[seed % 2], 1 if st == 2 else rng.choice([1, 2]))
    plans = [(rect, dict(base))]
    for drop in ("jetty", "porch", "oriel", "dormers", "chimney"):
        k = dict(base)
        k.pop(drop, None)
        plans.append((rect, k))
    k = dict(base)
    for drop in ("jetty", "oriel", "brackets", "flashing", "dormers"):
        k.pop(drop, None)
    plans.append((rect, k))
    plans.append((rect, {"mat": dict(b.voice), "openings": "rhythm", "stair": stair}))
    plans.append(((px0, pz0, px1, pz1),
                  {"mat": dict(b.voice), "openings": "rhythm", "stair": "none"}))

    res, used, kw = None, rect, {}
    for i, (rc, kwi) in enumerate(plans):
        r = b.building(label, rc[0], rc[1], rc[2], rc[3], st, roofspec, **kwi)
        if r and r.get("ok"):
            res, used, kw = r, rc, kwi
            break
    if res is None:
        return {"ok": False, "reason": "no shell"}
    ex = res.get("extras") or {}

    rooms = [tuple(r) for r in (res.get("rooms") or [])]
    levels = sorted(set(r[1] for r in rooms))
    dc = _door_cell(res.get("door"), door, fy)
    avoid = set()
    solid = set()
    inner = None
    if dc:
        ux, uz = STEP[dside]
        for i in range(0, 3):
            avoid.add((dc[0] - ux * i, dc[2] - uz * i))
        avoid.add((dc[0] - ux - uz, dc[2] - uz - ux))
        avoid.add((dc[0] - ux + uz, dc[2] - uz + ux))
        inner = (dc[0] - ux, dc[2] - uz)
    for c in _cells_of(res.get("stairs"), []):
        avoid.add((c[0], c[2]))
        solid.add((c[0], c[2]))

    landing = None
    if kw.get("stair") == "none" and rooms:
        for r in rooms:
            if r[1] != levels[0]:
                continue
            ok, cells, landing = _dogleg(b, label, r[0], r[2], r[3], r[4],
                                         r[1], dc, dside)
            for q in cells:
                avoid.add(q)
                solid.add(q)
            solid.discard(landing)
            break

    o = ex.get("oriel")
    if (not kw.get("jetty")) and ("oriel" not in kw or (
            isinstance(o, dict) and not o.get("ok"))):
        _oriel(b, used, dside, door, fy + 4, rng.choice([2, 3]))

    for s in range(st):
        rc = used
        if kw.get("jetty") and s >= 1:
            rc = (used[0] - 1, used[1] - 1, used[2] + 1, used[3] + 1)
        top = fy + 4 * s + 3
        _frame(b, rc, fy + 4 * s + 1, top, top)

    if dc:
        _connect(b, dc, dside, used)

    big = (used[2] - used[0] - 1) * (used[3] - used[1] - 1) >= 40
    for r in rooms:
        lvl = levels.index(r[1])
        cells = set((x, z) for x in range(r[0], r[3] + 1)
                    for z in range(r[2], r[4] + 1))
        holes, taken, edge = _read_room(b, r[0], r[1], r[2], r[3], r[4])
        way = inner if lvl == 0 else landing
        if way not in cells or way in holes or way in taken:
            way = None
        for q in sorted(edge):
            if way is None and q not in taken:
                way = q
        blocked = set(q for q in solid if q in cells) | holes | taken
        keep = _reserve(cells - blocked, way, edge) if way else set()
        ctx = {"cells": cells, "placed": set(),
               "avoid": avoid | edge | holes | taken | keep,
               "solid": blocked, "seed": way}
        if lvl == 0:
            _ground(b, use, r[0], r[1] + 1, r[2], r[3], r[4], rng, dside,
                    ctx, big)
        else:
            _upper(b, use, r[0], r[1] + 1, r[2], r[3], r[4], rng, lvl, dside,
                   ctx)
        for (x, z) in keep:
            for yy in (r[1] + 1, r[1] + 2):
                if not _thru(b.get_block(x, yy, z) or "air"):
                    b.place_block(x, yy, z, "air")
        out = 1 if (kw.get("jetty") and lvl >= 1) else 0
        seeds = [inner] if lvl == 0 else sorted(edge)
        _floor_check(b, r[0] - out, r[2] - out, r[3] + out, r[4] + out,
                     r[1], seeds, True, _pier_b(b))

    if dc:
        b.check_door(dc[0], dc[1], dc[2])
    b.seal_voids(px0, pz0, px1, pz1, _wall_b(b))
    if dc:
        _skirt(b, px0, pz0, px1, pz1, fy, dc, dside, b.block(b.voice["footing"], "wall"), 8,
               used)
    _unfloat(b, b.check_attached(), px0, pz0, px1, pz1, avoid)
    b.check_walkable(label)
    return {"ok": True, "use": use, "rect": used, "storeys": st}
