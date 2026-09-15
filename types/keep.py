"""The keep of a walled town.

A tower with real thickness to its walls, a fighting top a person can walk up to, and --
where the ground is long enough -- a hall laid against it under a steep gable. This is a
**form** and every material in it comes from the settlement's palette on `b.voice`, so
the same keep stands in white render, in blackstone, or in ochre stone under green tile,
and is the same keep in all three.
"""

import math
import random

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

KIND = "plot"

PARAMS = {
    "storeys": ("int", 2, 3),
    "plan": ("choice", ["tower", "hall_and_tower"]),
}

# Its author declared (7, 7, 24, 24). `scripts/type_needs.py` measured (4, 4, 6, 6), and
# the measurement is what is on the file. What the sweep found is a real defect and it
# is worth stating. The keep is clean at **every even pad** it was swept at and dirty at
# the **odd** ones from 7 to 19: it leaves a sealed pocket of five to a hundred and
# fifty-six cells, which is E003 on its own plot. Its own checker never saw it, because
# the six registered fixtures are real ground and the sweep is a level plane at every
# size. Clean runs: 4-6, 8, 10, 12, 18, 20-22. The band is the one holding a registered
# fixture and it is the lowest.
NEEDS = {
    "footprint": (4, 4, 32, 32),
    "except": (7, 9, 11, 13, 14, 15, 16, 17, 19),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

# ---------------------------------------------------------------- palette. A keep is a
# form -- a tower, a hall beside it, a stair to the deck and a battlement -- and it is
# the same keep in ochre stone under green tile as in white render.

def _axial(block, axis):
    return (f"{block}[axis={axis}]"
            if block.endswith(("_log", "_pillar", "_wood")) else block)


def _wall(b):
    return b.block(b.voice["wall"])


def _wall2(b):
    return b.block(b.voice["wall"], "accent")


def _wall3(b):
    return b.block(b.voice["wall"], "fine")


def _pillar(b):
    return b.block(b.voice["wall"], "post")


def _qslab(b):
    return b.block(b.voice["wall"], "slab")


def _frame(b):
    return b.block(b.voice["frame"], "post")


def _trim_b(b):
    return b.block(b.voice["trim"], "bare")


def _plank(b):
    return b.block(b.voice["floor"])


def _roofmat(b):
    return b.voice["roof"]


def _roofb(b):
    return b.block(b.voice["roof"])


def _roofslab(b):
    return b.block(b.voice["roof"], "slab")


def _roofwall(b):
    """A battlement merlon. Not every family has a wall block; the slab is the
    library's own second answer and is never a silent substitution -- it is asked for
    only after `block()` has refused by name."""
    try:
        return b.block(b.voice["roof"], "wall")
    except ValueError:
        return b.block(b.voice["roof"], "slab")


def _foot(b):
    return b.block(b.voice["footing"])


def _footm(b):
    return b.block(b.voice["footing"], "accent")
PANE = "glass_pane"

OVERSAIL = 1

DIRV = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
SIDEDIR = {"n": "north", "s": "south", "e": "east", "w": "west"}


def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _log(b, axis):
    return _axial(_frame(b), axis)


def _trim(b, axis):
    return _axial(_trim_b(b), axis)


def _inner(rect, t):
    x0, z0, x1, z1 = rect
    return (x0 + t, z0 + t, x1 - t, z1 - t)


def _cells(rect):
    x0, z0, x1, z1 = rect
    out = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            out.append((x, z))
    return out


def _ring(b, rect, y, blk, inset=0):
    """One course of the perimeter of rect at height y."""
    x0, z0, x1, z1 = rect
    x0 += inset
    z0 += inset
    x1 -= inset
    z1 -= inset
    if x1 < x0 or z1 < z0:
        return
    b.place_cuboid(x0, y, z0, x1, y, z0, blk)
    b.place_cuboid(x0, y, z1, x1, y, z1, blk)
    b.place_cuboid(x0, y, z0, x0, y, z1, blk)
    b.place_cuboid(x1, y, z0, x1, y, z1, blk)


def _ring_band(b, rect, y, t, blk):
    """A band of thickness t around the perimeter, at height y."""
    for i in range(t):
        _ring(b, rect, y, blk, inset=i)


def _ring_logs(b, rect, y):
    """A girding course of stripped log, oriented along its own run."""
    x0, z0, x1, z1 = rect
    b.place_cuboid(x0 + 1, y, z0, x1 - 1, y, z0, _trim(b, "x"))
    b.place_cuboid(x0 + 1, y, z1, x1 - 1, y, z1, _trim(b, "x"))
    b.place_cuboid(x0, y, z0 + 1, x0, y, z1 - 1, _trim(b, "z"))
    b.place_cuboid(x1, y, z0 + 1, x1, y, z1 - 1, _trim(b, "z"))
    for (cx, cz) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        b.place_block(cx, y, cz, _log(b, "y"))


def _column(b, x, y0, y1, z, blk):
    if y1 < y0:
        return
    b.place_cuboid(x, y0, z, x, y1, z, blk)


def _face_runs(rect):
    """The four outer faces as (name, [(x, z), ...]) along the wall line."""
    x0, z0, x1, z1 = rect
    runs = []
    runs.append(("north", [(x, z0) for x in range(x0, x1 + 1)]))
    runs.append(("south", [(x, z1) for x in range(x0, x1 + 1)]))
    runs.append(("west", [(x0, z) for z in range(z0, z1 + 1)]))
    runs.append(("east", [(x1, z) for z in range(z0, z1 + 1)]))
    return runs


def _side_of(rect, x, z):
    x0, z0, x1, z1 = rect
    d = {"w": abs(x - x0), "e": abs(x - x1), "n": abs(z - z0), "s": abs(z - z1)}
    best = "w"
    for k in ("w", "e", "n", "s"):
        if d[k] < d[best]:
            best = k
    return best


# ------------------------------------------------------------------ shell --
def _tower_shell(b, rect, F, h, nf, t):
    """Solid mass, then the storeys carved out of it. Returns the deck y."""
    x0, z0, x1, z1 = rect
    top = F + nf * h
    b.place_cuboid(x0, F, z0, x1, top, z1, _wall(b))
    ix0, iz0, ix1, iz1 = _inner(rect, t)
    for s in range(nf):
        y = F + s * h
        b.place_cuboid(ix0, y + 1, iz0, ix1, y + h - 1, iz1, "air")
        b.place_cuboid(ix0, y, iz0, ix1, y, iz1, _plank(b) if s else _foot(b))
    return top


def _hall_shell(b, rect, F, h, hs, extra, t=1):
    """Hall walls to the eave; the upper room is open to its own rafters."""
    x0, z0, x1, z1 = rect
    top = F + (hs - 1) * h + h + extra
    b.place_cuboid(x0, F, z0, x1, top, z1, _wall(b))
    ix0, iz0, ix1, iz1 = _inner(rect, t)
    for s in range(hs):
        y = F + s * h
        # the top room is open to its own rafters: a shut roof space over it is a room
        # nobody can get into
        hi = top if s == hs - 1 else (y + h - 1)
        b.place_cuboid(ix0, y + 1, iz0, ix1, hi, iz1, "air")
        b.place_cuboid(ix0, y, iz0, ix1, y, iz1, _plank(b) if s else _foot(b))
    return top


def _frame_skin(b, rect, F, top, h, nf, foot, rng, gird=True):
    """Cobble footing, corner posts, girding courses, panel posts."""
    x0, z0, x1, z1 = rect
    for y in range(F, F + foot):
        _ring(b, rect, y, _foot(b) if (y - F) % 2 == 0 else _footm(b))
    corners = ((x0, z0), (x1, z0), (x0, z1), (x1, z1))
    for (cx, cz) in corners:
        _column(b, cx, F + foot, top - 1, cz, _log(b, "y"))
    if gird:
        for s in range(1, nf + 1):
            y = F + s * h
            if y >= top:
                break
            _ring_logs(b, rect, y)
    # a post at least every five blocks of wall run
    for (name, run) in _face_runs(rect):
        if name in ("north", "south"):
            coords = [x for x in range(x0 + 1, x1)]
            fixed = z0 if name == "north" else z1
        else:
            coords = [z for z in range(z0 + 1, z1)]
            fixed = x0 if name == "west" else x1
        if not coords:
            continue
        step = 4 if len(coords) > 5 else max(2, len(coords) - 1)
        picks = [c for i, c in enumerate(coords) if i % step == step - 1]
        if not picks and coords:
            picks = [coords[len(coords) // 2]]
        for c in picks:
            for s in range(nf):
                y0 = F + s * h + 1
                y1 = F + (s + 1) * h - 1
                if s == 0:
                    y0 = max(y0, F + foot)
                y1 = min(y1, top - 1)
                if name in ("north", "south"):
                    _column(b, c, y0, y1, fixed, _log(b, "y"))
                else:
                    _column(b, fixed, y0, y1, c, _log(b, "y"))


def _deck_and_parapet(b, rect, pad, y, rng, style):
    """The fighting top: a corbelled deck where there is room, then merlons."""
    x0, z0, x1, z1 = rect
    px0, pz0, px1, pz1 = pad
    px0 -= OVERSAIL
    pz0 -= OVERSAIL
    px1 += OVERSAIL
    pz1 += OVERSAIL
    dx0 = x0 - 1 if x0 - 1 >= px0 else x0
    dz0 = z0 - 1 if z0 - 1 >= pz0 else z0
    dx1 = x1 + 1 if x1 + 1 <= px1 else x1
    dz1 = z1 + 1 if z1 + 1 <= pz1 else z1
    deck = (dx0, dz0, dx1, dz1)
    if deck != rect:
        # the corbel table the overhang sits on, only where it actually hangs
        for (cx, cz) in _cells(deck):
            if x0 <= cx <= x1 and z0 <= cz <= z1:
                continue
            if (cx + cz) % 2 == 0:
                b.place_block(cx, y - 1, cz, _roofslab(b) + "[type=top]")
            else:
                b.place_block(cx, y - 1, cz, _roofwall(b))
    b.place_cuboid(dx0, y, dz0, dx1, y, dz1, _roofb(b))
    step = 2 if style == "merlon" else 3
    _ring(b, deck, y + 1, _wall2(b))
    per = []
    for x in range(dx0, dx1 + 1):
        per.append((x, dz0))
        per.append((x, dz1))
    for z in range(dz0 + 1, dz1):
        per.append((dx0, z))
        per.append((dx1, z))
    for (x, z) in per:
        if (x + z) % step == 0:
            b.place_block(x, y + 2, z, _wall2(b))
            b.place_block(x, y + 3, z, _roofslab(b))
        else:
            b.place_block(x, y + 2, z, _roofslab(b))
    for (cx, cz) in ((dx0, dz0), (dx1, dz0), (dx0, dz1), (dx1, dz1)):
        b.place_block(cx, y + 1, cz, _wall3(b))
        b.place_block(cx, y + 2, cz, _wall2(b))
        b.place_block(cx, y + 3, cz, _roofslab(b))
    return deck


# ------------------------------------------------------------------ stair --
def _straight_options(interior, h, rng, stand_out=None, land_out=None):
    """Straight flights that fit this room, the ones against a wall first.

    A flight down the middle of a room cuts the floor above it in two, so
    those come last.  `stand_out` are cells outside the room a flight may
    begin from -- a doorway, where you step straight onto the first tread.
    `land_out` are cells outside it a flight may land on -- the roof deck,
    which is laid right across the wall head."""
    ix0, iz0, ix1, iz1 = interior
    near = []
    far = []
    spec = []
    for (facing, sgn, axis) in (("east", 1, "x"), ("west", -1, "x"),
                                ("south", 1, "z"), ("north", -1, "z")):
        if axis == "x":
            lo, hi = ix0, ix1
            lines = range(iz0, iz1 + 1)
            edges = (iz0, iz1)
        else:
            lo, hi = iz0, iz1
            lines = range(ix0, ix1 + 1)
            edges = (ix0, ix1)
        for line in lines:
            for s0 in range(lo - 1, hi + 2):
                coords = [s0 + sgn * k for k in range(h + 2)]
                cells = []
                for c in coords:
                    cells.append((c, line) if axis == "x" else (line, c))
                bad = False
                for c in coords[1:h + 1]:
                    if c < lo or c > hi:
                        bad = True
                if bad:
                    continue
                special = False
                if not (lo <= coords[0] <= hi):
                    if not stand_out or cells[0] not in stand_out:
                        continue
                    special = True
                if not (lo <= coords[h + 1] <= hi):
                    if not land_out or cells[h + 1] not in land_out:
                        continue
                    special = True
                bag = spec if special else (near if line in edges else far)
                bag.append((facing, cells[1][0], cells[1][1], cells, special))
    rng.shuffle(near)
    rng.shuffle(far)
    rng.shuffle(spec)
    return near + far + spec


def _underpin(b, interior, y0, y1, blk):
    """Nothing stands on air: carry every tread down to the floor it left."""
    ix0, iz0, ix1, iz1 = interior
    for x in range(ix0, ix1 + 1):
        for z in range(iz0, iz1 + 1):
            for y in range(y0 + 2, y1 + 1):
                g = b.get_block(x, y, z)
                if isinstance(g, str) and g.endswith("stairs"):
                    b.place_cuboid(x, y0 + 1, z, x, y - 1, z, blk)
                    break


def _turn_cells(interior, h, corner, da, db):
    """Where a half-turn flight lands: (used, treads, mid, landing) or None."""
    ix0, iz0, ix1, iz1 = interior
    sx, sz = corner
    ax, az = DIRV[da]
    bx, bz = DIRV[db]
    treads = []
    for k in range(1, h):
        treads.append((sx + ax * k, sz + az * k))
    mid = (sx + ax * h, sz + az * h)
    tb = (mid[0] + bx, mid[1] + bz)
    land = (mid[0] + bx * 2, mid[1] + bz * 2)
    used = [(sx, sz)] + treads + [mid, tb, land]
    for (cx, cz) in used:
        if not (ix0 <= cx <= ix1 and iz0 <= cz <= iz1):
            return None
    return (used, treads, mid, tb, land)


def _turn_options(interior, h):
    """A stair that turns on a half landing, for a room too short to run one."""
    out = []
    ix0, iz0, ix1, iz1 = interior
    for cx in (ix0, ix1):
        for cz in (iz0, iz1):
            da_x = "east" if cx == ix0 else "west"
            db_z = "south" if cz == iz0 else "north"
            out.append((da_x, db_z, cx, cz))
            out.append((db_z, da_x, cx, cz))
    return out


def _reaches_all(interior, blocked, entry):
    """Is every cell of this floor that is not built on reachable from entry?"""
    ix0, iz0, ix1, iz1 = interior
    free = set()
    for x in range(ix0, ix1 + 1):
        for z in range(iz0, iz1 + 1):
            if (x, z) not in blocked:
                free.add((x, z))
    if entry is not None and entry not in free:
        return False
    start = entry
    if start is None:
        if not free:
            return True
        start = sorted(free)[0]
    seen = set([start])
    stack = [start]
    while stack:
        cx, cz = stack.pop()
        for (nx, nz) in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
            if (nx, nz) in free and (nx, nz) not in seen:
                seen.add((nx, nz))
                stack.append((nx, nz))
    return len(seen) == len(free)


def _pack_roof(b, rect, ceil_y, top_y):
    """Fill the space between a ceiling and the roof over it, column by
    column, so no shut loft is left behind."""
    for (cx, cz) in _cells(rect):
        hit = None
        yy = ceil_y + 1
        while yy <= top_y + 2:
            g = b.get_block(cx, yy, cz)
            if isinstance(g, str) and g != "air":
                hit = yy
                break
            yy += 1
        if hit is not None and hit > ceil_y + 1:
            b.place_cuboid(cx, ceil_y + 1, cz, cx, hit - 1, cz, _wall(b))


def _path(interior, blocked, start, goal):
    """The shortest way across a floor from one cell to another, or None."""
    ix0, iz0, ix1, iz1 = interior
    if start is None or goal is None:
        return None
    prev = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out
        cx, cz = cur
        for nxt in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
            if nxt in prev:
                continue
            if not (ix0 <= nxt[0] <= ix1 and iz0 <= nxt[1] <= iz1):
                continue
            if nxt in blocked and nxt != goal:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    return None


def _lay_turn(b, interior, y0, h, corner, da, db, landing_blk):
    """Two half flights round a newel, cut through the floor above."""
    got = _turn_cells(interior, h, corner, da, db)
    if not got:
        return None
    used, treads, mid, tb, land = got
    mx, mz = mid
    tx, tz = tb
    lx, lz = land
    tre = [(treads[k][0], y0 + k + 1, treads[k][1]) for k in range(len(treads))]
    # the newel the half landing stands on
    _column(b, mx, y0 + 1, y0 + h - 2, mz, _wall2(b))
    b.place_block(mx, y0 + h - 1, mz, _plank(b))
    # headroom over the whole flight, and the well cut through the floor above
    for (cx, cy, cz) in tre:
        b.place_cuboid(cx, cy + 1, cz, cx, cy + 3, cz, "air")
    b.place_cuboid(mx, y0 + h, mz, mx, y0 + h + 2, mz, "air")
    b.place_cuboid(tx, y0 + h + 1, tz, tx, y0 + h + 2, tz, "air")
    b.place_cuboid(lx, y0 + h + 1, lz, lx, y0 + h + 2, lz, "air")
    b.place_block(lx, y0 + h, lz, landing_blk or _plank(b))
    # say which way the flight climbs: the floor either side of it is level, and a tread
    # with level ground both sides has nothing else to go on
    b.steps(tre, b.block(b.voice["floor"], "stairs"),
            axis=("x" if da in ("east", "west") else "z"), prefer=da)
    b.steps([(tx, y0 + h, tz)], b.block(b.voice["floor"], "stairs"),
            axis=("x" if db in ("east", "west") else "z"), prefer=db)
    return used


def _lay_stair(b, label, interior, y0, h, rng, holes, entry, forbid,
               landing_blk=None, dbg=None, stand_out=None, land_out=None):
    """A walkable flight from the floor at y0 to the floor at y0 + h.

    A flight is a wall through the room at knee height: the room it leaves
    behind has to still be one room, walkable from the way in, or the flight
    goes somewhere else.  Returns (cells, landing) or None."""

    def ok(cells, blocked, exempt):
        for c in cells:
            if c in forbid and c not in exempt:
                return False
        bl = set(blocked) | set(holes)
        if entry is not None and entry in bl:
            return False
        return _reaches_all(interior, bl, entry)

    free = set(stand_out or []) | set(land_out or [])
    for (facing, fx, fz, cells, sp) in _straight_options(
            interior, h, rng, stand_out, land_out):
        # the first tread is a step up, the rest of the flight is a wall
        blocked = set(cells[2:h + 1])
        if not ok(cells, blocked, free if sp else set()):
            continue
        r = b.flight(label, fx, fz, y0, y0 + h, facing,
                     mat=b.voice["floor"])
        if isinstance(r, dict) and r.get("ok"):
            if dbg is not None:
                dbg.append(("flight", y0, facing, fx, fz, sp))
            return (cells, cells[h + 1])
    turns = _turn_options(interior, h)
    rng.shuffle(turns)
    for (da, db, cx, cz) in turns:
        got = _turn_cells(interior, h, (cx, cz), da, db)
        if not got:
            continue
        used, treads, mid, tb, land = got
        blocked = set(treads[1:]) | set([mid, tb])
        if not ok(used, blocked, set()):
            continue
        if _lay_turn(b, interior, y0, h, (cx, cz), da, db, landing_blk):
            if dbg is not None:
                dbg.append(("turn", y0, da, db, cx, cz))
            return (used, land)
    return None


# --------------------------------------------------------------- openings --
def _slits(b, rect, t, F, s, h, side, spacing, sill, head, skip):
    """Openings on a rhythm along one face, splayed through a thick wall."""
    x0, z0, x1, z1 = rect
    y = F + s * h
    if side == "north":
        ax0, az0, ax1, az1 = x0 + 1, z0, x1 - 1, z0
        inward = "south"
    elif side == "south":
        ax0, az0, ax1, az1 = x0 + 1, z1, x1 - 1, z1
        inward = "north"
    elif side == "west":
        ax0, az0, ax1, az1 = x0, z0 + 1, x0, z1 - 1
        inward = "east"
    else:
        ax0, az0, ax1, az1 = x1, z0 + 1, x1, z1 - 1
        inward = "west"
    if ax1 < ax0 or az1 < az0:
        return []
    got = b.openings(ax0, y, az0, ax1, az1, spacing=spacing, width=1,
                     sill=sill, head=head)
    out = []
    if not isinstance(got, (list, tuple)):
        return out
    dx, dz = DIRV[inward]
    for cell in got:
        if not (isinstance(cell, (list, tuple)) and len(cell) >= 3):
            continue
        ox, oy, oz = cell[0], cell[1], cell[2]
        if (ox, oz) in skip:
            b.place_cuboid(ox, y + sill, oz, ox, y + head, oz, _wall(b))
            continue
        out.append((ox, oy, oz))
        for k in range(1, t):
            b.place_cuboid(ox + dx * k, y + sill, oz + dz * k,
                           ox + dx * k, y + head, oz + dz * k, "air")
    return out


def _tunnel_path(dx, dz, inward, interior):
    """The cells a passage has to take from the doorstep to the room inside.

    The town reserves the doorstep; a keep's wall is two blocks thick and the
    doorstep can sit off the end of a wall, so the way in turns if it must."""
    ix0, iz0, ix1, iz1 = interior
    vx, vz = DIRV[inward]
    cx, cz = dx, dz
    cells = [(cx, cz)]
    for _ in range(48):
        if ix0 <= cx <= ix1 and iz0 <= cz <= iz1:
            break
        if vx and not (ix0 <= cx <= ix1):
            cx += vx
        elif vz and not (iz0 <= cz <= iz1):
            cz += vz
        elif cx < ix0:
            cx += 1
        elif cx > ix1:
            cx -= 1
        elif cz < iz0:
            cz += 1
        elif cz > iz1:
            cz -= 1
        else:
            break
        cells.append((cx, cz))
    return cells


def _oriel(b, rect, interior, pad, side, cx, cz, y, top, h, width, t):
    """The signature: a bay carried out on brackets over the entrance.

    It starts at the first floor and runs up under the corbel of the top,
    so it has no ledge of its own for the weather -- or for anyone -- to
    stand on."""
    dx, dz = DIRV[side]
    px0, pz0, px1, pz1 = pad
    px0 -= OVERSAIL
    pz0 -= OVERSAIL
    px1 += OVERSAIL
    pz1 += OVERSAIL
    ix0, iz0, ix1, iz1 = interior
    cells = []
    if side in ("north", "south"):
        span = [(cx + i, cz) for i in range(-(width // 2), width // 2 + 1)]
    else:
        span = [(cx, cz + i) for i in range(-(width // 2), width // 2 + 1)]
    for (sx, sz) in span:
        if not _on_face(rect, side, sx, sz):
            return False
        bx, bz = sx - dx * t, sz - dz * t
        if not (ix0 <= bx <= ix1 and iz0 <= bz <= iz1):
            return False
        ox, oz = sx + dx, sz + dz
        if not (px0 <= ox <= px1 and pz0 <= oz <= pz1):
            return False
        cells.append((sx, sz, ox, oz))
    axis = "x" if side in ("east", "west") else "z"
    mid = cells[len(cells) // 2]
    for (sx, sz, ox, oz) in cells:
        # the brackets it is carried on
        b.place_block(ox, y - 1, oz, _trim_b(b) + "[axis=" + axis + "]")
        b.place_cuboid(ox, y, oz, ox, top, oz, _wall2(b))
    for ys in range(y, top, h):
        for (sx, sz, ox, oz) in cells:
            b.place_block(ox, ys, oz, _plank(b))
            if (sx, sz) == (mid[0], mid[1]) or len(cells) < 3:
                b.place_cuboid(ox, ys + 1, oz, ox, ys + 2, oz, PANE)
            # open the wall behind it, right through
            for k in range(t):
                b.place_cuboid(sx - dx * k, ys + 1, sz - dz * k,
                               sx - dx * k, ys + 2, sz - dz * k, "air")
        for (sx, sz, ox, oz) in (cells[0], cells[-1]):
            b.place_block(ox, ys + 3, oz, _wall3(b))
    for (sx, sz, ox, oz) in cells:
        b.place_block(ox, top, oz, _wall2(b))
    return True


# -------------------------------------------------------------- furniture --
class Room(object):
    """One floor of one block: the cells in it and what is already spoken for."""

    def __init__(self, interior, y, taken):
        self.i = interior
        self.y = y
        self.taken = taken

    def free(self, rng, wall_only=False, inner_only=False):
        ix0, iz0, ix1, iz1 = self.i
        out = []
        for x in range(ix0, ix1 + 1):
            for z in range(iz0, iz1 + 1):
                if (x, z) in self.taken:
                    continue
                edge = (x == ix0 or x == ix1 or z == iz0 or z == iz1)
                if wall_only and not edge:
                    continue
                if inner_only and edge:
                    continue
                out.append((x, z))
        rng.shuffle(out)
        return out

    def facing_in(self, x, z):
        ix0, iz0, ix1, iz1 = self.i
        d = {"south": x - ix0, "north": x - ix0, "east": 0, "west": 0}
        best = None
        bestd = 99
        for (name, dist) in (("east", x - ix0), ("west", ix1 - x),
                             ("south", z - iz0), ("north", iz1 - z)):
            if dist < bestd:
                bestd = dist
                best = name
        return best or "north"


def _fit(b, rm, kind, rng, facing=None, wall_only=False, inner_only=False,
         tries=14, **kw):
    """Try a fitting round the room until one lands; claim what it uses."""
    n = 0
    for (x, z) in rm.free(rng, wall_only=wall_only, inner_only=inner_only):
        n += 1
        if n > tries:
            break
        f = facing or rm.facing_in(x, z)
        r = b.fitting(kind, x, rm.y + 1, z, f, **kw)
        if isinstance(r, dict) and r.get("ok"):
            got = r.get("cells")
            if isinstance(got, (list, tuple)):
                for c in got:
                    if isinstance(c, (list, tuple)) and len(c) >= 3:
                        rm.taken.add((c[0], c[-1]))
            rm.taken.add((x, z))
            return r
    return None


def _hearth(b, room, rng, ceiling, flue_top, mat, tag):
    """A hearth against a wall; if its flue will not go, the stack is built."""
    ix0, iz0, ix1, iz1 = room.i
    n = 0
    for (x, z) in room.free(rng, wall_only=True):
        n += 1
        if n > 12:
            break
        if x in (ix0, ix1) and z in (iz0, iz1):
            continue
        f = room.facing_in(x, z)
        r = b.fitting("hearth", x, room.y + 1, z, f, mat=mat, room=tag,
                      flue_to=flue_top)
        built = isinstance(r, dict) and r.get("ok")
        if not built:
            r = b.fitting("hearth", x, room.y + 1, z, f, mat=mat, room=tag)
            if not (isinstance(r, dict) and r.get("ok")):
                continue
            b.place_cuboid(x, ceiling, z, x, flue_top, z, mat)
            b.place_block(x, flue_top + 1, z, _roofslab(b))
        room.taken.add((x, z))
        for c in (r.get("cells") or []):
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                room.taken.add((c[0], c[-1]))
        return r
    return None


def _pitch_for(eave, span, ceiling):
    """The steepest roof that still leaves the tower standing over it."""
    for (rise, run) in ((2, 1), (3, 2), (1, 1), (1, 2), (1, 3)):
        top = eave + int(math.ceil((span / 2.0) * rise / float(run)))
        if top <= ceiling:
            return (rise, run)
    return (1, 3)


def _on_face(rect, side, x, z):
    """Is this cell on that outside face of this rectangle?"""
    x0, z0, x1, z1 = rect
    if side == "w":
        return x == x0 and z0 <= z <= z1
    if side == "e":
        return x == x1 and z0 <= z <= z1
    if side == "n":
        return z == z0 and x0 <= x <= x1
    return z == z1 and x0 <= x <= x1


def _read_part(part, b):
    px0 = min(int(part["x0"]), int(part["x1"]))
    px1 = max(int(part["x0"]), int(part["x1"]))
    pz0 = min(int(part["z0"]), int(part["z1"]))
    pz1 = max(int(part["z0"]), int(part["z1"]))
    pad = (px0, pz0, px1, pz1)
    F = int(part["floor_y"])
    label = part.get("label")
    dx = dz = None
    d = part.get("door")
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        dx, dz = int(d[0]), int(d[-1])
    facing = part.get("facing")
    fi = b.floor_from_threshold(label) if label is not None else None
    if isinstance(fi, dict):
        if dx is None and isinstance(fi.get("door"), (list, tuple)):
            fd = fi["door"]
            if len(fd) >= 2:
                dx, dz = int(fd[0]), int(fd[-1])
        if not facing:
            facing = fi.get("facing")
    if dx is None:
        dx, dz = px0, (pz0 + pz1) // 2
    if facing not in DIRV:
        facing = "east"
    return pad, F, label, dx, dz, facing


# ------------------------------------------------------------------ build --
def build(b, part, seed, **params):
    rng = random.Random((int(seed) * 2654435761 + 40503) & 0x7fffffff)
    storeys = _clamp(int(params.get("storeys", 2) or 2), 1, 3)
    if storeys < 2:
        storeys = 2
    plan = params.get("plan") or "tower"

    pad, F, label, dx, dz, facing_in = _read_part(part, b)
    px0, pz0, px1, pz1 = pad
    W = px1 - px0 + 1
    D = pz1 - pz0 + 1
    stand_y = F + 1
    # the town reserved the doorstep and which way you walk in off the lane; the door
    # goes in the face opposite that, and nowhere else
    side = None
    for k in ("n", "s", "e", "w"):
        if SIDEDIR[k] == OPP[facing_in]:
            side = k
    if side is None or not _on_face(pad, side, dx, dz):
        side = _side_of(pad, dx, dz)
        facing_in = OPP[SIDEDIR[side]]
    ddx, ddz = dx, dz
    ivx, ivz = DIRV[facing_in]
    inward = facing_in

    # the keep stands on the whole pad; only the eaves and the oriel oversail
    brect = pad
    bx0, bz0, bx1, bz1 = brect
    BW = bx1 - bx0 + 1
    BD = bz1 - bz0 + 1
    long_x = BW >= BD
    L = max(BW, BD)
    S = min(BW, BD)

    # ---- the massing: a tower, and a hall against it where the ground is long
    want_hall = (plan == "hall_and_tower" and L >= 13 and S >= 8)
    hrect = None
    yard = None
    if want_hall:
        T = _clamp(min(S, L // 2), 6, 9)
        if long_x:
            low = abs(dx - bx0) <= abs(dx - bx1)
            if low:
                trect = (bx1 - T + 1, bz0, bx1, bz1)
                hrect = (bx0, bz0, bx1 - T + 1, bz1)
            else:
                trect = (bx0, bz0, bx0 + T - 1, bz1)
                hrect = (bx0 + T - 1, bz0, bx1, bz1)
        else:
            low = abs(dz - bz0) <= abs(dz - bz1)
            if low:
                trect = (bx0, bz1 - T + 1, bx1, bz1)
                hrect = (bx0, bz0, bx1, bz1 - T + 1)
            else:
                trect = (bx0, bz0, bx1, bz0 + T - 1)
                hrect = (bx0, bz0 + T - 1, bx1, bz1)
    else:
        keep_len = min(L, S + 3 + rng.randint(0, 2))
        if L - keep_len >= 4:
            if long_x:
                if abs(dx - bx0) <= abs(dx - bx1):
                    trect = (bx0, bz0, bx0 + keep_len - 1, bz1)
                    yard = (bx0 + keep_len, bz0, bx1, bz1)
                else:
                    trect = (bx1 - keep_len + 1, bz0, bx1, bz1)
                    yard = (bx0, bz0, bx1 - keep_len, bz1)
            else:
                if abs(dz - bz0) <= abs(dz - bz1):
                    trect = (bx0, bz0, bx1, bz0 + keep_len - 1)
                    yard = (bx0, bz0 + keep_len, bx1, bz1)
                else:
                    trect = (bx0, bz1 - keep_len + 1, bx1, bz1)
                    yard = (bx0, bz0, bx1, bz1 - keep_len)
        else:
            trect = brect

    tx0, tz0, tx1, tz1 = trect
    TW = tx1 - tx0 + 1
    TD = tz1 - tz0 + 1
    t = 2 if min(TW, TD) >= 8 else 1
    tw = TW - 2 * t
    td = TD - 2 * t
    # A storey is four blocks, always: under a ceiling three above the floor a person
    # cannot get up the first tread, and the flight is a stair to nowhere. Where the
    # room cannot carry a flight at all the keep is one tall chamber with its fighting
    # top over it and a watch house on that.
    h = 4
    tiny = not (max(tw, td) >= h + 2 or
                (max(tw, td) >= h + 1 and min(tw, td) >= 3))
    nf = 1 if tiny else storeys + (1 if hrect else 0)
    deck_y = F + nf * h
    t_int = _inner(trect, t)

    # ---- the mass itself
    _tower_shell(b, trect, F, h, nf, t)

    hs = 0
    hall_top = F
    ridge = F
    if hrect:
        hs = 2 if storeys >= 3 else 1
        extra = 1 if hs == 1 else 0
        hall_top = _hall_shell(b, hrect, F, h, hs, extra, 1)
        hx0, hz0, hx1, hz1 = hrect
        hw = hx1 - hx0 + 1
        hd = hz1 - hz0 + 1
        if hw >= hd:
            axis = "z"
            span = hd
        else:
            axis = "x"
            span = hw
        pitch = _pitch_for(hall_top, span, deck_y - 1)
        got = b.roof(hx0, hz0, hx1, hz1, hall_top, _roofmat(b), style="gable",
                     axis=axis, pitch=pitch, overhang=OVERSAIL)
        ridge = got if isinstance(got, int) else hall_top + span
        # the ceiling is the eave course; everything between it and the underside of the
        # roof is packed solid, because a roof space with no way into it is a room
        # nobody can enter
        b.place_cuboid(hx0 + 1, hall_top, hz0 + 1, hx1 - 1, hall_top,
                       hz1 - 1, "air")
        _pack_roof(b, hrect, hall_top, ridge)
        # heal the tower where the hall roof lapped onto it
        b.place_cuboid(tx0, F, tz0, tx1, deck_y - 1, tz1, _wall(b))
        ix0, iz0, ix1, iz1 = t_int
        for s in range(nf):
            y = F + s * h
            b.place_cuboid(ix0, y + 1, iz0, ix1, y + h - 1, iz1, "air")
            b.place_cuboid(ix0, y, iz0, ix1, y, iz1, _plank(b) if s else _foot(b))

    deck = _deck_and_parapet(b, trect, pad, deck_y, rng,
                             rng.choice(["merlon", "merlon", "wide"]))
    _frame_skin(b, trect, F, deck_y, h, nf, 2, rng)
    if hrect:
        _frame_skin(b, hrect, F, hall_top + 1, h, hs + 1, 2, rng)

    # ---- which block the doorway goes in, and the way in from it
    dblock = trect
    if hrect and _on_face(hrect, side, ddx, ddz):
        dblock = hrect
    dt = t if dblock == trect else 1
    d_int = _inner(dblock, dt)
    tunnel = _tunnel_path(ddx, ddz, inward, d_int)

    # ---- the way through from hall to tower, before anything is laid in it
    avoid = {}
    nostair = {}
    for s in range(nf + 2):
        avoid[s] = set()
        nostair[s] = set()
    pierce_cells = []
    tower_entry = None
    dirn = None
    if hrect:
        hx0, hz0, hx1, hz1 = hrect
        ix0, iz0, ix1, iz1 = t_int
        if long_x:
            if hx1 == tx0:
                sx, dirn = tx0, "east"
            else:
                sx, dirn = tx1, "west"
            sz = (iz0 + iz1) // 2
            base = (sx, sz)
        else:
            if hz1 == tz0:
                sz, dirn = tz0, "south"
            else:
                sz, dirn = tz1, "north"
            sx = (ix0 + ix1) // 2
            base = (sx, sz)
        pvx, pvz = DIRV[dirn]
        for s in range(min(hs, nf)):
            y = F + s * h + 1
            for k in range(t):
                cx = base[0] + pvx * k
                cz = base[1] + pvz * k
                b.place_cuboid(cx, y, cz, cx, y + 1, cz, "air")
                b.place_block(cx, y + 2, cz, _trim_b(b) + "[axis=" +
                              ("x" if dirn in ("east", "west") else "z") + "]")
            inner = (base[0] + pvx * t, base[1] + pvz * t)
            avoid[s].add(inner)
            nostair[s].add(inner)
            pierce_cells.append((base[0], y, base[1]))
            if s == 0:
                tower_entry = inner

    # ---- the entry and its passage, kept clear of stairs and furniture
    for c in tunnel:
        avoid[0].add(c)
        nostair[0].add(c)
    avoid[0].add((tunnel[-1][0] + ivx, tunnel[-1][1] + ivz))

    # ---- the flights, floor to floor and out onto the top
    dbg = []
    entry = tunnel[-1] if dblock == trect else tower_entry
    holes = set()
    stair_log = []
    routes = []
    for s in range(nf):
        y0 = F + s * h
        land = _roofb(b) if s == nf - 1 else _plank(b)
        # a flight laid over the flight below buries its headroom
        forbid = set(nostair[s]) | set(holes)
        s_out = set(tunnel) if (s == 0 and dblock == trect) else None
        l_out = None
        if s == nf - 1:
            l_out = set(_cells((deck[0] + 1, deck[1] + 1, deck[2] - 1,
                                deck[3] - 1))) - set(_cells(t_int))
        got = _lay_stair(b, label, t_int, y0, h, rng, holes, entry, forbid,
                         land, dbg, s_out, l_out)
        if not got:
            got = _lay_stair(b, label, t_int, y0, h, rng, holes, None, forbid,
                             land, dbg, s_out, l_out)
        if not got:
            got = _lay_stair(b, label, t_int, y0, h, rng, set(), None,
                             set(holes), land, dbg, s_out, l_out)
        if got:
            stair_log.append((y0, got[0], got[1]))
            # the way from the way in to the foot of the flight is circulation
            blk = set(got[0][1:]) | set(holes)
            blk.discard(got[0][0])
            way = _path(t_int, blk, entry, got[0][0])
            if way:
                routes.append((way, y0))
                for c in way:
                    avoid[s].add(c)
            used, landing = got
            for c in used:
                avoid[s].add(c)
                avoid[s + 1].add(c)
            holes = set(used)
            holes.discard(landing)
            entry = landing
        else:
            holes = set()
            entry = None
    hall_keep = set()
    if hrect and dirn is not None:
        h_int = _inner(hrect, 1)
        pvx, pvz = DIRV[dirn]
        hall_side = (base[0] - pvx, base[1] - pvz)
        if dblock == hrect:
            way = _path(h_int, set(), tunnel[-1], hall_side)
            if way:
                routes.append((way, F))
                hall_keep = set(way)
        else:
            hall_keep = set([hall_side])
    b.resolve_steps()
    for s in range(nf):
        _underpin(b, t_int, F + s * h, F + (s + 1) * h, _wall2(b))

    # ---- openings: slits low, wider as the walls get thinner with height
    tshared = None
    hshared = None
    if hrect:
        tshared = OPP[dirn]
        hshared = dirn
    dface = SIDEDIR[side]
    dskip = set()
    for k in (-1, 0, 1):
        if side in ("w", "e"):
            dskip.add((ddx, ddz + k))
        else:
            dskip.add((ddx + k, ddz))
    for s in range(nf):
        sp = 5 if s == 0 else (4 if s + 1 < nf else 3)
        sill = 1
        head = h - 1
        for face in ("north", "south", "east", "west"):
            if face == tshared:
                continue
            sk = dskip if (s == 0 and face == dface and dblock == trect) else set()
            _slits(b, trect, t, F, s, h, face, sp, sill, head, sk)
    if hrect:
        for s in range(hs):
            for face in ("north", "south", "east", "west"):
                if face == hshared:
                    continue
                sk = dskip if (s == 0 and face == dface and
                               dblock == hrect) else set()
                _slits(b, hrect, 1, F, s, h, face, 4, 1, h - 1, sk)

    # ---- the doorway, and the passage driven through the thickness
    dres = b.doorway(ddx, stand_y, ddz, facing_in, b.voice["frame"],
                     leaf=b.joinery(b.voice, "door"), jamb="build", lintel=True)
    dbg.append(("door", ddx, stand_y, ddz, facing_in, dt, dres))
    # where the flight starts in the doorway itself, the head of the door is open above
    # it -- you cannot climb under your own lintel
    for (y0, used, landing) in stair_log:
        if y0 == F and used[0] in tunnel:
            b.place_cuboid(used[0][0], stand_y + 2, used[0][1],
                           used[0][0], stand_y + 2, used[0][1], "air")
    dix0, diz0, dix1, diz1 = d_int
    for (cx, cz) in tunnel[1:]:
        b.place_cuboid(cx, stand_y, cz, cx, stand_y + 1, cz, "air")
        b.place_block(cx, stand_y - 1, cz, _foot(b))
        if not (dix0 <= cx <= dix1 and diz0 <= cz <= diz1):
            b.place_block(cx, stand_y + 2, cz, _wall3(b))

    # ---- the signature: the oriel on brackets over the entrance
    ow = 3 if rng.random() < 0.7 else 1
    if nf < 2:
        ow = 0
    if side in ("w", "e"):
        ocx = ddx
        ocz = _clamp(ddz, diz0 + ow // 2, diz1 - ow // 2)
    else:
        ocx = _clamp(ddx, dix0 + ow // 2, dix1 - ow // 2)
        ocz = ddz
    o_top = deck_y - 2 if dblock == trect else hall_top - 1
    if ow and o_top > F + h:
        if not _oriel(b, dblock, d_int, pad, dface, ocx, ocz, F + h, o_top,
                      h, ow, dt):
            _oriel(b, dblock, d_int, pad, dface, ddx, ddz, F + h, o_top,
                   h, 1, dt)

    # ---- the bailey wall round the yard, where there is ground left over
    if yard:
        yx0, yz0, yx1, yz1 = yard
        gate = set()
        if yx1 - yx0 >= yz1 - yz0:
            gc = (yz0 + yz1) // 2
            gate = set([(yx0, gc), (yx0, gc + 1), (yx1, gc), (yx1, gc + 1)])
        else:
            gc = (yx0 + yx1) // 2
            gate = set([(gc, yz0), (gc + 1, yz0), (gc, yz1), (gc + 1, yz1)])
        for (cx, cz) in _cells(yard):
            if not (cx in (yx0, yx1) or cz in (yz0, yz1)):
                continue
            if tx0 <= cx <= tx1 and tz0 <= cz <= tz1:
                continue
            if (cx, cz) in gate:
                continue
            b.place_cuboid(cx, F + 1, cz, cx, F + 2, cz, _foot(b))
            if (cx + cz) % 2 == 0:
                b.place_block(cx, F + 3, cz, b.joinery(b.voice, "fence"))

    b.resolve_steps()

    # ---- furnished like somewhere people hold out
    parapet_top = deck_y + 3
    for s in range(nf):
        room = Room(t_int, F + s * h, set(avoid[s]))
        ceil_y = F + (s + 1) * h
        if s == 0:
            _fit(b, room, "store", rng, wall_only=True, mat=_foot(b),
                 extent=2 + rng.randint(0, 2), room="store")
            _fit(b, room, "store", rng, wall_only=True, mat=_foot(b),
                 extent=2, room="store")
            if rng.random() < 0.5:
                _fit(b, room, "workbench", rng, wall_only=True, room="store")
            else:
                _fit(b, room, "trough", rng, wall_only=True, mat=_foot(b),
                     room="store")
            _fit(b, room, "light", rng, wall_only=True, room="store")
        elif s == nf - 1:
            if nf == 1:
                _fit(b, room, "store", rng, wall_only=True, mat=_foot(b),
                     extent=2 + rng.randint(0, 1), room="store")
                _fit(b, room, "bed", rng, wall_only=True, room="house")
            _hearth(b, room, rng, ceil_y, parapet_top, _foot(b), "hall")
            _fit(b, room, "table", rng, inner_only=(tw > 3 and td > 3),
                 room="hall")
            _fit(b, room, "bench", rng, room="hall")
            _fit(b, room, "bench", rng, room="hall")
            if nf < 3:
                _fit(b, room, "bed", rng, wall_only=True, room="house")
            _fit(b, room, "light", rng, room="hall")
            _fit(b, room, "light", rng, wall_only=True, room="hall")
        else:
            _fit(b, room, "bed", rng, wall_only=True, room="house")
            _fit(b, room, "bed", rng, wall_only=True, room="house")
            _fit(b, room, "store", rng, wall_only=True, mat=_foot(b), extent=2,
                 room="store")
            _fit(b, room, "shelf", rng, wall_only=True, room="house")
            _fit(b, room, "light", rng, room="house")

    # the top: a brazier's worth of light behind the merlons
    dkx0, dkz0, dkx1, dkz1 = deck
    top_room = Room((dkx0 + 1, dkz0 + 1, dkx1 - 1, dkz1 - 1), deck_y,
                    set(avoid[nf]))
    _fit(b, top_room, "light", rng, room="hall")
    if (dkx1 - dkx0) > 4 and (dkz1 - dkz0) > 4:
        _fit(b, top_room, "store", rng, wall_only=True, mat=_foot(b), extent=2,
             room="store")

    if hrect:
        h_int = _inner(hrect, 1)
        for s in range(hs):
            room = Room(h_int, F + s * h, set(hall_keep))
            if s == 0:
                _fit(b, room, "table", rng, inner_only=True, room="hall")
                _fit(b, room, "bench", rng, room="hall")
                _fit(b, room, "bench", rng, room="hall")
                _fit(b, room, "store", rng, wall_only=True, mat=_foot(b),
                     extent=3, room="store")
                _fit(b, room, "light", rng, room="hall")
                _fit(b, room, "light", rng, room="hall")
            else:
                _fit(b, room, "bed", rng, wall_only=True, room="house")
                _fit(b, room, "bed", rng, wall_only=True, room="house")
                _fit(b, room, "store", rng, wall_only=True, mat=_foot(b),
                     extent=2, room="store")
                _fit(b, room, "light", rng, room="house")

    if yard:
        yx0, yz0, yx1, yz1 = yard
        yroom = Room((yx0 + 1, yz0 + 1, yx1 - 1, yz1 - 1), F, set())
        _fit(b, yroom, "well", rng, inner_only=False, mat=_foot(b), room="yard")
        _fit(b, yroom, "light", rng, room="yard")

    # ---- circulation is not furniture's to take: clear it again
    for (way, wy) in routes:
        for (cx, cz) in way:
            b.place_cuboid(cx, wy + 1, cz, cx, wy + 2, cz, "air")
    for (cx, cz) in tunnel[1:]:
        b.place_cuboid(cx, stand_y, cz, cx, stand_y + 1, cz, "air")
    tex, tez = tunnel[-1][0] + ivx, tunnel[-1][1] + ivz
    if dix0 <= tex <= dix1 and diz0 <= tez <= diz1:
        b.place_cuboid(tex, stand_y, tez, tex, stand_y + 1, tez, "air")
    for (y0, used, landing) in stair_log:
        sx, sz = used[0]
        b.place_cuboid(sx, y0 + 1, sz, sx, y0 + 2, sz, "air")
        lx, lz = landing
        b.place_cuboid(lx, y0 + h + 1, lz, lx, y0 + h + 2, lz, "air")
    for (cx, cy, cz) in pierce_cells:
        b.place_cuboid(cx, cy, cz, cx, cy + 1, cz, "air")

    # ---- read the work back
    b.resolve_steps()
    cd = b.check_door(ddx, stand_y, ddz)
    cw = b.check_walkable(label)
    sv = b.seal_voids(px0, pz0, px1, pz1, _wall2(b))
    ca = b.check_attached()
    # a chimney cap or a bracket the fitting left standing on nothing is a loose end,
    # not a form: take it away rather than prop it up
    if isinstance(ca, dict):
        for piece in (ca.get("floating") or []):
            bb = piece.get("bbox")
            if piece.get("natural") or not bb or len(bb) < 6:
                continue
            if piece.get("cells", 99) > 8:
                continue
            if bb[0] < px0 - 2 or bb[3] > px1 + 2:
                continue
            if bb[2] < pz0 - 2 or bb[5] > pz1 + 2:
                continue
            if bb[1] <= F:
                continue
            b.place_cuboid(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], "air")
        ca = b.check_attached()
    return {"tower": trect, "hall": hrect, "deck": deck_y, "ridge": ridge}
