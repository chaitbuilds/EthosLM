"""A temple hall, in whatever the settlement is made of.

A type, not a building: a hall standing back from the lip of its own platform, wall
panels between posts at close intervals carried through to the eave, a two-tiered
irimoya roof shallow below and steep at the ridge, its eaves turned up and carried out
past the posts on a bracket course. The panels, the posts and the tiles are roles on
`b.voice`, and this file names no material.

What the seed changes is form driven by purpose -- how much of the platform the hall
takes and how far back off its lip it stands, whether the ridge runs across the way in
or along it, whether the hall gains a side hall against one flank, how deep the
sanctuary is, and what the floor in front of it is arranged to do.

Two things about the library shape this file. Reading the world reads it as it was
before this pass began, so nothing here is worked out by looking at what has just been
built: the wall, the openings and the furniture are all placed from geometry this
program already knows and from what the library hands back. And the shell lays its own
openings, so it is asked for blank walls and the wall is put together here -- posts
first, then a light in the panel between one pair of posts and the next.
"""

import random

FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "civic"

PARAMS = {
    "storeys": ("int", 1, 3),
    "plan": ("choice", ["hall", "verandah", "cloister"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3x3 to 32x32 measured, clean at every size swept, once the shell's yard wall
    # opened where the wall stands and the court's basins kept its ways out. Measured by
    # scripts/type_needs.py; the band is rounds/type-needs.json. The pair is the pad
    # site() hands build(), after siting's inset.
    "footprint": (3, 3, 32, 32),
    "except": (),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

# One palette, by role, for every temple in the settlement.
def _post(b):
    """The upright of the frame. From the voice, not from this file."""
    p = b.block(b.voice["frame"], "post")
    return b.axial(p, "y")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


# ---------------------------------------------------------------- the pad

def _pins(part):
    """Which pad edges the reserved doorstep pins, and which are free.

        The threshold sits in one wall and a block off the corner beside it, so one x
        edge and one z edge are spoken for; the other two are mine to pull in.
        
    """
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    pin = {"x0": (dx - x0) <= 1, "x1": (x1 - dx) <= 1,
           "z0": (dz - z0) <= 1, "z1": (z1 - dz) <= 1}
    if not (pin["x0"] or pin["x1"]):
        pin["x0" if (dx - x0) <= (x1 - dx) else "x1"] = True
    if not (pin["z0"] or pin["z1"]):
        pin["z0" if (dz - z0) <= (z1 - dz) else "z1"] = True
    free_x = None if pin["x1"] and pin["x0"] else ("x1" if not pin["x1"] else "x0")
    free_z = None if pin["z1"] and pin["z0"] else ("z1" if not pin["z1"] else "z0")
    return pin, free_x, free_z


def _edge_cells(part, edge):
    """The line of ground one block outside a pad edge."""
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    if edge == "x0":
        return [(x0 - 1, z) for z in range(z0, z1 + 1)]
    if edge == "x1":
        return [(x1 + 1, z) for z in range(z0, z1 + 1)]
    if edge == "z0":
        return [(x, z0 - 1) for x in range(x0, x1 + 1)]
    return [(x, z1 + 1) for x in range(x0, x1 + 1)]


def _setback(b, part):
    """How far back off each lip of the platform the hall has to stand.

        Where the ground falls away past the pad, a wall built hard on the lip carries
        its own footing out over whatever is below -- and what is below, here, is the
        lane. Two courses of platform want two blocks of apron where a lane runs along
        the foot of them, one where the fall is private. A temple set back off its own
        edge is the right building anyway: the platform is meant to be walked on.
        
    """
    floor_y = part["floor_y"]
    back = {}
    for edge in ("x0", "x1", "z0", "z1"):
        cells = _edge_cells(part, edge)
        hs = sorted(b.get_height(x, z) for (x, z) in cells)
        drop = floor_y - hs[len(hs) // 2]
        near = 9
        for (mx, mz) in (cells[0], cells[len(cells) // 2], cells[-1]):
            lane = b.nearest_lane(mx, mz)
            if lane is not None and lane.get("distance") is not None:
                near = min(near, lane["distance"])
        if drop < 2:
            back[edge] = 0          # the ground out there is the platform's own top
        else:
            back[edge] = 3 if near <= 2 else 1
    # Never set back past the doorstep's own row: the door has to stay in a wall.
    dx, dz = part["door"]
    if part["facing"] in ("east", "west"):
        back["z0"] = max(0, min(back["z0"], (dz - 1) - part["z0"]))
        back["z1"] = max(0, min(back["z1"], part["z1"] - (dz + 1)))
    else:
        back["x0"] = max(0, min(back["x0"], (dx - 1) - part["x0"]))
        back["x1"] = max(0, min(back["x1"], part["x1"] - (dx + 1)))
    door_edge = {"east": "x0", "west": "x1", "south": "z0", "north": "z1"}[
        part["facing"]]
    for lo, hi, span in (("x0", "x1", part["x1"] - part["x0"] + 1),
                         ("z0", "z1", part["z1"] - part["z0"] + 1)):
        room = max(0, span - (6 if span >= 10 else 5))
        first, second = (lo, hi) if hi == door_edge else (hi, lo)
        while back[lo] + back[hi] > room:
            if back[first] > 0:
                back[first] -= 1
            elif back[second] > 0:
                back[second] -= 1
            else:
                break
    return back


# ---------------------------------------------------------------- massing

def _massing(b, part, rng, plan):
    """The footprint, and what the ground left over is spent on."""
    pin, free_x, free_z = _pins(part)
    back = _setback(b, part)
    x0 = part["x0"] + back["x0"]
    x1 = part["x1"] - back["x1"]
    z0 = part["z0"] + back["z0"]
    z1 = part["z1"] - back["z1"]
    pw, pd = x1 - x0 + 1, z1 - z0 + 1

    least = 6 if min(pw, pd) >= 8 else 5
    max_cut_x = max(0, min(pw - least, pw // 2)) if free_x else 0
    max_cut_z = max(0, min(pd - least, pd // 2)) if free_z else 0
    cut_x = rng.randint(0, max_cut_x)
    cut_z = rng.randint(0, max_cut_z)
    if plan == "cloister":                  # a walled precinct wants ground to wall
        cut_x = max(cut_x, min(2, max_cut_x))
        cut_z = max(cut_z, min(2, max_cut_z))
    elif plan == "verandah":                # a hall you can walk right round
        cut_x = max(cut_x, min(1, max_cut_x))
        cut_z = max(cut_z, min(1, max_cut_z))

    # A long pad reads as a hall with a side hall against it, not one long box.
    if max_cut_z >= 6 and rng.random() < 0.75:
        cut_z = rng.randint(6, max_cut_z)
    elif max_cut_x >= 6 and rng.random() < 0.75:
        cut_x = rng.randint(6, max_cut_x)

    if free_x == "x1":
        x1 -= cut_x
    elif free_x == "x0":
        x0 += cut_x
    if free_z == "z1":
        z1 -= cut_z
    elif free_z == "z0":
        z0 += cut_z

    wing = None
    if cut_z >= 6 and (x1 - x0) >= 5:
        wing = ((x0, part["z0"] + back["z0"], x1, z0 - 1) if free_z == "z0"
                else (x0, z1 + 1, x1, part["z1"] - back["z1"]))
    elif cut_x >= 6 and (z1 - z0) >= 5:
        wing = ((part["x0"] + back["x0"], z0, x0 - 1, z1) if free_x == "x0"
                else (x1 + 1, z0, part["x1"] - back["x1"], z1))
    return {"foot": (x0, z0, x1, z1), "wing": wing}


def _roof_spec(rng, w, d):
    """Two tiers of deepslate tile, shallow below and steep at the ridge.

        Anything shallower than (1,2) below runs the eave course so far out that the
        tier corners end up facing down their own slope, so (1,2) is the floor of it,
        and a third tier only buys another sealed loft nobody can reach.
        
    """
    if abs(w - d) <= 1:
        axis = rng.choice(["x", "z"])
    else:
        axis = "z" if w > d else "x"
    ends = ("irimoya", "irimoya")
    if rng.random() < 0.25:
        ends = ("irimoya", "half-hip") if rng.random() < 0.5 else ("half-hip", "irimoya")
    eave = "upturned" if rng.random() < 0.85 else "flared"
    upper = rng.choice([(2, 1), (2, 1), (3, 1)])
    return {"style": "hip", "axis": axis, "ends": ends, "eave": eave,
            "tiers": 2, "profile": [(1, 2), upper]}


def _storey_cap(w, d):
    inner = min(w, d) - 2
    if inner >= 6:
        return 3
    if inner >= 5:
        return 2
    return 1


def _attempts(b, part, rng, storeys, plan, mass):
    """The building, and what to give up first if the library refuses it."""
    x0, z0, x1, z1 = mass["foot"]
    w, d = x1 - x0 + 1, z1 - z0 + 1
    storeys = max(1, min(storeys, _storey_cap(w, d)))
    spec = _roof_spec(rng, w, d)

    extras = {"brackets": True}
    if mass["wing"] is not None:
        extras["wing"] = mass["wing"]
    if plan == "cloister":
        extras["yard"] = (part["x0"], part["z0"], part["x1"], part["z1"])
    if storeys >= 2 and min(w, d) >= 9 and rng.random() < 0.5:
        extras["dormers"] = 2

    base = dict(x0=x0, z0=z0, x1=x1, z1=z1, roof=spec, mat=dict(b.voice),
                openings="none")
    out = [dict(base, storeys=storeys, **extras)]
    if len(extras) > 1:
        out.append(dict(base, storeys=storeys, brackets=True))
    if storeys > 1:
        out.append(dict(base, storeys=1, brackets=True))
    simple = {"style": "hip", "ends": ("irimoya", "irimoya"), "eave": "upturned",
              "tiers": 2, "profile": [(1, 2), (2, 1)]}
    out.append(dict(base, storeys=1, roof=simple, brackets=True))
    out.append(dict(x0=part["x0"], z0=part["z0"], x1=part["x1"], z1=part["z1"],
                    roof=simple, mat=dict(b.voice), storeys=1, brackets=True,
                    openings="none"))
    return out


# ------------------------------------------- panels, posts and the openings

def _wall_cells(foot):
    """The perimeter columns of a footprint, side by side, in order along each
    of the four runs."""
    x0, z0, x1, z1 = foot
    return [[(x, z0) for x in range(x0, x1 + 1)],
            [(x, z1) for x in range(x0, x1 + 1)],
            [(x0, z) for z in range(z0, z1 + 1)],
            [(x1, z) for z in range(z0, z1 + 1)]]


def _fenestrate(b, foot, door, floor_y, storeys, eave, spacing):
    """White panels between dark posts, and a tall light in every other panel.

        The shell is asked for blank walls and the wall is put together here, in that
        order: the posts first, at a known rhythm and at every corner, carried from
        the floor through to the eave; then an opening cut in the panel between one
        pair of posts and the next, so a post can never land across a window and a
        window can never eat a post.  The doorway's own panel is left alone.
        
    """
    dx, dz = door
    posts = set()
    for run in _wall_cells(foot):
        for i, (x, z) in enumerate(run):
            if i == 0 or i == len(run) - 1 or i % spacing == 0:
                posts.add((x, z))
    posts.discard((dx, dz))
    for (x, z) in sorted(posts):
        for y in range(floor_y + 1, eave):
            b.place_block(x, y, z, _post(b))

    lights = 0
    for run in _wall_cells(foot):
        marks = [i for i, c in enumerate(run) if c in posts]
        for (a, c) in zip(marks, marks[1:]):
            if c - a < 3:                  # a single panel is left solid: one cell
                continue                   # is not a run of wall to open
            gap = run[a + 1:c]
            if any(cell == (dx, dz) for cell in gap):
                continue
            (gx0, gz0), (gx1, gz1) = gap[0], gap[-1]
            for i in range(storeys):
                b.openings(gx0, floor_y + 4 * i, gz0, gx1, gz1,
                           spacing=max(1, len(gap)), width=1, sill=1, head=3)
                lights += 1
    return lights


# ---------------------------------------------------------------- inside

def _frame(travel, rx0, rz0, rx1, rz1):
    """(u, v) -> (x, z), u counted inward from the wall the door is in."""
    if travel in ("east", "west"):
        length, cross = rx1 - rx0 + 1, rz1 - rz0 + 1
        base, step = (rx0, 1) if travel == "east" else (rx1, -1)
        return length, cross, (lambda u, v: (base + step * u, rz0 + v))
    length, cross = rz1 - rz0 + 1, rx1 - rx0 + 1
    base, step = (rz0, 1) if travel == "south" else (rz1, -1)
    return length, cross, (lambda u, v: (rx0 + v, base + step * u))


def _fit(b, kind, x, y, z, facing, **kw):
    r = b.fitting(kind, x, y, z, facing, **kw)
    return bool(r and r.get("ok"))


def _rows(L, first=2):
    """Rows furniture may stand in: every other one, counted back from the far
    wall, so the rows between are clear right across and nothing is walled in."""
    return [u for u in range(first, L) if (L - 1 - u) % 2 == 0]


def _furnish_nave(b, part, room, fy, rng, avoid=()):
    """The hall the temple is for: the sanctuary raised at the far end, the floor
    in front of it left to whoever comes in off the lane."""
    def put(kind, x, y, z, facing, **kw):
        if (x, z) in avoid:
            return False
        return _fit(b, kind, x, y, z, facing, **kw)

    rx0, rz0, rx1, rz1 = room
    travel = part["facing"]
    back = OPPOSITE[travel]
    L, C, at = _frame(travel, rx0, rz0, rx1, rz1)
    if L < 3 or C < 3:
        return 0
    centre = C // 2
    dx, dz = part["door"]
    v_door = (dz - rz0) if travel in ("east", "west") else (dx - rx0)
    v_door = max(0, min(C - 1, v_door))
    keep = {centre, v_door}
    rows = _rows(L)

    raised, depth = False, 0
    if L >= 8 and C >= 5:
        depth = 3 if (L >= 12 and rng.random() < 0.5) else 2
        a, c = at(L - 1, 0), at(L - depth, C - 1)
        r = b.dais(min(a[0], c[0]), min(a[1], c[1]), max(a[0], c[0]),
                   max(a[1], c[1]), fy - 1, b.voice["footing"])
        raised = bool(r and r.get("ok"))
        if not raised:
            depth = 0
    ay = fy + (1 if raised else 0)

    ax, az = at(L - 1, centre)
    put("table", ax, ay, az, back, mat=b.voice["frame"], room="shrine")
    for v in (centre - 2, centre + 2):
        if 0 <= v <= C - 1:
            lx, lz = at(L - 1, v)
            put("light", lx, ay, lz, back, mat=b.voice["frame"], room="shrine")
    shelf_row = L - 3 if (L - 3) in rows else None
    if shelf_row is not None and C >= 5:
        for v in (1, C - 2):
            if v in keep:
                continue
            sx, sz = at(shelf_row, v)
            yy = ay if (L - 1 - shelf_row) < depth else fy
            if not put("bookshelf", sx, yy, sz, back, mat=b.voice["frame"], room="shrine"):
                put("shelf", sx, yy, sz, back, mat=b.voice["frame"], room="shrine")

    seats = [v for v in range(1, C - 1) if v not in keep]
    open_rows = [u for u in rows if 2 <= u <= L - depth - 2]
    laid = 0
    for u in open_rows[-rng.choice([1, 2, 2, 3, 3]):] if open_rows else []:
        for v in seats:
            bx, bz = at(u, v)
            if put("bench", bx, fy, bz, travel, mat=b.voice["frame"], room="hall"):
                laid += 1
    if L >= 4:
        ux, uz = at(1, centre)
        put("rug", ux, fy, uz, travel, mat=b.voice["frame"], room="hall")
    edge_v = 0 if 0 not in keep else (C - 1 if (C - 1) not in keep else None)
    if edge_v is not None and open_rows:
        tx, tz = at(open_rows[0], edge_v)
        # one cell each: anything that runs two cells along a wall can pinch the corner
        # behind it off the rest of the floor.
        if not put("shelf", tx, fy, tz, travel, mat=b.voice["frame"], room="hall"):
            put("store", tx, fy, tz, travel, mat=b.voice["frame"], room="store", extent=1)
        lx, lz = at(open_rows[len(open_rows) // 2], edge_v)
        put("light", lx, fy, lz, travel, mat=b.voice["frame"], room="hall")
    return laid


def _furnish_side(b, part, room, fy, rng, kind, avoid=()):
    """Everything that is not the hall: the side hall, the floor above it, the
    loft under the upper tier."""
    def put(what, x, y, z, facing, **kw):
        if (x, z) in avoid:
            return False
        return _fit(b, what, x, y, z, facing, **kw)

    rx0, rz0, rx1, rz1 = room
    travel = part["facing"]
    L, C, at = _frame(travel, rx0, rz0, rx1, rz1)
    if L < 2 or C < 1:
        return 0
    centre = C // 2
    rows = _rows(L, first=1) or [L - 1]
    laid = 0
    spots = [(rows[-1], v) for v in (0, C - 1) if v != centre]
    if len(rows) > 1:
        spots.append((rows[0], 0 if centre != 0 else C - 1))
    kinds = ["bookshelf", "shelf", "store"] if kind != "house" else \
            ["bed", "shelf", "store"]
    for i, (u, v) in enumerate(spots):
        sx, sz = at(u, v)
        if put(kinds[i % len(kinds)], sx, fy, sz, OPPOSITE[travel], mat=b.voice["frame"],
               room=kind):
            laid += 1
        elif put("shelf", sx, fy, sz, OPPOSITE[travel], mat=b.voice["frame"], room=kind):
            laid += 1
    lx, lz = at(rows[len(rows) // 2], centre)
    if put("light", lx, fy, lz, travel, mat=b.voice["frame"], room=kind):
        laid += 1
    if L >= 4 and C >= 3:
        tx, tz = at(rows[0], centre)
        if put("table", tx, fy, tz, travel, mat=b.voice["frame"], room=kind):
            laid += 1
    return laid


def _furnish_court(b, part, foot, wing, rng):
    """The platform in front of the hall is half of what a temple is: lanterns
    round the edge of it and a basin to wash at, with the way in left clear."""
    fy = part["floor_y"] + 1
    dx, dz = part["door"]

    def taken(x, z):
        if foot[0] <= x <= foot[2] and foot[1] <= z <= foot[3]:
            return True
        return bool(wing) and (wing[0] <= x <= wing[2] and wing[1] <= z <= wing[3])

    free = [(x, z) for x in range(part["x0"], part["x1"] + 1)
            for z in range(part["z0"], part["z1"] + 1)
            if not taken(x, z) and max(abs(x - dx), abs(z - dz)) >= 2]
    if not free:
        return 0
    # The court a person can actually walk: air at standing height over a solid course.
    # The precinct wall of a cloister stands in plot cells too, and a walk counted
    # through it is a walk through a wall.
    have = {(x, z) for (x, z) in free
            if b.get_block(x, fy, z) == "air" and b.get_block(x, fy - 1, z) != "air"}

    def elbow(c):
        n = sum(1 for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (c[0] + d[0], c[1] + d[1]) in have)
        return n == 2                  # a cell in a one-block-wide run of court:
    free = [c for c in free if not elbow(c)]   # stand anything here and the far
    if not free:                               # end of the walk round is cut off
        return 0
    free.sort(key=lambda c: (abs(c[0] - dx) + abs(c[1] - dz)))
    want = max(2, min(8, len(free) // 5 + 2))
    step = max(1, len(free) // want)
    kinds = ["trough", "light", "light", "light", "trough", "light", "light", "light"]
    used = set()

    # Where the court is left by: a court cell beside standable ground outside the plot,
    # or beside the way in. A court with no way out of it is the hall's own doing, and
    # nothing stood in it should make it worse.
    def standable(x, z):
        return b.get_block(x, fy, z) == "air" and b.get_block(x, fy - 1, z) != "air"

    exits = {c for c in have
             if any(nb not in have and not taken(*nb) and standable(*nb)
                    for nb in ((c[0] + 1, c[1]), (c[0] - 1, c[1]),
                               (c[0], c[1] + 1), (c[0], c[1] - 1)))}

    def whole(taken_cells):
        """Can every cell of the court still be walked to from a way out of it with
        these cells stood on? A basin at the mouth of a one-block apron round the
        flank turns the apron into a room nobody can walk into, and the elbow rule
        above only sees the apron's own cells."""
        left = {c for c in have if c not in taken_cells}
        if not left:
            return True
        seeds = [c for c in exits if c in left]
        if not seeds:
            return False
        seen, stack = set(seeds), list(seeds)
        while stack:
            (cx, cz) = stack.pop()
            for nb in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
                if nb in left and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(left)

    for i in range(want):
        (x, z) = free[min(i * step, len(free) - 1)]
        kind = kinds[i % len(kinds)]
        mat = b.voice["footing"] if kind == "trough" else b.voice["frame"]
        if not whole(used | {(x, z)}):
            continue
        if _fit(b, kind, x, fy, z, part["facing"], mat=mat, room="shrine"):
            used.add((x, z))
            used.add((x + 1, z))       # a basin may run two cells: treat both as
            used.add((x, z + 1))       # standing on, when looking for pinches
    return used


def _mend_court(b, part, foot, wing, used, door):
    """A corner of the platform the building's own plan cuts off from the rest.

        The apron round a hall is sometimes an L a block wide, and a piece of it can
        end up reachable only across a diagonal, which is not walking.  Anything
        small enough to be that gets a pier of the platform's own stone standing on
        it, so it is masonry rather than floor nobody can reach.
        
    """
    fy = part["floor_y"]

    def taken(x, z):
        if foot[0] <= x <= foot[2] and foot[1] <= z <= foot[3]:
            return True
        return bool(wing) and (wing[0] <= x <= wing[2] and wing[1] <= z <= wing[3])

    free = {(x, z) for x in range(part["x0"], part["x1"] + 1)
            for z in range(part["z0"], part["z1"] + 1)
            if not taken(x, z) and (x, z) not in used}
    seen, groups = set(), []
    for cell in sorted(free):
        if cell in seen:
            continue
        stack, group = [cell], []
        seen.add(cell)
        while stack:
            (x, z) = stack.pop()
            group.append((x, z))
            for (nx, nz) in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if (nx, nz) in free and (nx, nz) not in seen:
                    seen.add((nx, nz))
                    stack.append((nx, nz))
        groups.append(group)
    if len(groups) < 2:
        return 0
    groups.sort(key=len, reverse=True)
    mended = 0
    for group in groups[1:]:
        if len(group) > 4:
            continue
        for (x, z) in group:
            b.place_block(x, fy + 1, z, b.block(b.voice["footing"]))
            mended += 1
    return mended


# ---------------------------------------------------------------- build

def build(b, part, seed, **params):
    rng = random.Random((int(seed) * 6364136223846793005 + 1442695040888963407)
                        % (2 ** 61))
    storeys = int(params.get("storeys", 1) or 1)
    plan = params.get("plan") or "hall"
    if plan not in ("hall", "verandah", "cloister"):
        plan = "hall"
    floor_y = part["floor_y"]
    label = part["label"]
    door = part["door"]

    mass = _massing(b, part, rng, plan)
    res, used = None, None
    for kw in _attempts(b, part, rng, storeys, plan, mass):
        r = b.building(label, **kw)
        if r and r.get("ok"):
            res, used = r, kw
            break
    if res is None:
        return {"ok": False, "reason": "no footprint the library would take"}

    foot = (used["x0"], used["z0"], used["x1"], used["z1"])
    wing = used.get("wing")
    n = max(1, int(used["storeys"]))
    eave = int(res.get("eave_y") or (floor_y + 4 * n))
    spacing = 3

    # Where the shell actually hung the door -- which is not the doorstep cell when the
    # hall stands back off the lip of its platform, and a post laid over a door leaf is
    # a building with no way in.
    d = res.get("door")
    if isinstance(d, dict):
        d = (d.get("x"), d.get("y"), d.get("z"))
    if isinstance(d, (list, tuple)) and len(d) == 3 and d[0] is not None:
        door = (int(d[0]), int(d[2]))

    # The wall: white panels, tall lights on a rhythm, a dark post either side of every
    # one of them and at every corner, carried through to the eave.
    _fenestrate(b, foot, door, floor_y, n, eave, spacing)
    if wing:
        _fenestrate(b, wing, door, floor_y, n, eave, spacing)

    # No loft to pack. The shell's roof is solid from its eave to its surface, and a
    # tiered roof is filled between its tiers, so there is no room over the ceiling for
    # anyone to be sealed out of; `seal_voids` below is the general answer to any pocket
    # that is left. This file used to pack "the loft" with a replace-air fill over the
    # roof's whole volume, and that fill read the world as it stood before the pass
    # began (demo-polish, 1a), so it overwrote the roof the shell had just laid. A type
    # does not pack a roof.

    inside = (foot[0] + 1, foot[1] + 1, foot[2] - 1, foot[3] - 1)
    _furnish_nave(b, part, inside, floor_y + 1, rng)
    for i in range(1, n):
        _furnish_side(b, part, inside, floor_y + 4 * i + 1, rng,
                      "house" if i == n - 1 else "hall")
    if wing:
        for i in range(n):
            _furnish_side(b, part, (wing[0] + 1, wing[1] + 1, wing[2] - 1,
                                    wing[3] - 1), floor_y + 4 * i + 1, rng, "hall")
    court = _furnish_court(b, part, foot, wing, rng)
    _mend_court(b, part, foot, wing, court, door)

    if isinstance(d, (list, tuple)) and len(d) == 3 and d[0] is not None:
        b.check_door(int(d[0]), int(d[1]), int(d[2]))

    lo_x, lo_z, hi_x, hi_z = foot
    if wing:
        lo_x, lo_z = min(lo_x, wing[0]), min(lo_z, wing[1])
        hi_x, hi_z = max(hi_x, wing[2]), max(hi_z, wing[3])
    b.seal_voids(lo_x, lo_z, hi_x, hi_z, b.block(b.voice["wall"]))
    b.seal_voids(part["x0"], part["z0"], part["x1"], part["z1"], b.block(b.voice["footing"]))
    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "footprint": foot, "storeys": n}
