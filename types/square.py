"""A paved market square: a well at the crossing, four stalls in the corners,
open to the lane on every edge.

The voice of the settlement at the scale of a booth: a footing course carrying a frame,
wall panels between the posts, a steep gable over the top, and -- the signature move,
shrunk to a stall -- a hood on brackets carried out over the counter, which is the way
in. Every one of those is a role on `b.voice`, so the square is the same square in any
palette.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "civic"

PARAMS = {
    "paving": ("choice", ["banded", "checker", "radial"]),
    "canopy": ("choice", ["gable", "hip", "mixed"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 12x12 and up measured; under twelve the paving and the canopies do not compose,
    # and at six and seven across it raises. Measured by scripts/type_needs.py; the band
    # is rounds/type-needs.json. The pair is the pad site() hands build(), after
    # siting's inset.
    "footprint": (5, 5, 40, 40),
    "except": (6, 7, 8, 9, 10, 11),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

# Eight module constants of block names, put to the voice instead. A market square is a
# form -- paving, booths, a canopy, a well -- and it is the same square in ochre stone
# under green tile as it is in cobble and dark oak.

def _foot(b):
    return b.block(b.voice["footing"])


def _foot2(b):
    return b.block(b.voice["footing"], "accent")


def _wall(b):
    return b.block(b.voice["wall"])


def _panel(b):
    return b.block(b.voice["wall"], "accent")


def _post(b):
    p = b.block(b.voice["frame"], "post")
    return b.axial(p, "y")


def _slab(b):
    return b.block(b.voice["floor"], "slab") + "[type=bottom]"


def _fence(b):
    return b.joinery(b.voice, "fence")


def _beam(b, axis):
    t = b.block(b.voice["trim"], "bare")
    return b.axial(t, axis)

OPP = {"n": "s", "s": "n", "e": "w", "w": "e"}
FACING = {"n": "north", "s": "south", "e": "east", "w": "west"}
DELTA = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}
RUN = {"n": "x", "s": "x", "e": "z", "w": "z"}


def _edge(rect, side, inset=1):
    """The cells along one edge of a rectangle, corners left out by default."""
    x0, z0, x1, z1 = rect
    if side == "n":
        return [(x, z0) for x in range(x0 + inset, x1 - inset + 1)]
    if side == "s":
        return [(x, z1) for x in range(x0 + inset, x1 - inset + 1)]
    if side == "w":
        return [(x0, z) for z in range(z0 + inset, z1 - inset + 1)]
    return [(x1, z) for z in range(z0 + inset, z1 - inset + 1)]


def _column(b, x, y0, y1, block):
    if y1 >= y0:
        b.place_cuboid(x[0], y0, x[1], x[0], y1, x[1], block)


def _pave(b, rect, y, pattern, cx, cz, rng):
    """The floor of the square: one field, read at the scale of a person."""
    x0, z0, x1, z1 = rect
    b.fill_region(x0, y, z0, x1, y, z1, _foot(b))
    phase = rng.randrange(2)
    if pattern == "checker":
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if ((x // 2) + (z // 2)) % 2 == phase:
                    b.place_block(x, y, z, _foot2(b))
    elif pattern == "banded":
        step = rng.choice([3, 4])
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if (x - x0 + phase) % step == 0 or (z - z0 + phase) % step == 0:
                    b.place_block(x, y, z, _foot2(b))
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if (x - x0 + phase) % step == 0 and (z - z0 + phase) % step == 0:
                    b.place_block(x, y, z, _panel(b))
    else:
        rmax = min(cx - x0, x1 - cx, cz - z0, z1 - cz)
        r = 2
        band = 0
        while r <= rmax:
            b.ring(cx, y, cz, r, _panel(b) if band % 2 == phase else _foot2(b))
            r += 2
            band += 1
    # the four ways in, marked in the paving rather than gated
    mx0, mx1 = cx - 1, cx + 1
    mz0, mz1 = cz - 1, cz + 1
    for x in range(mx0, mx1 + 1):
        b.place_block(x, y, z0, _panel(b))
        b.place_block(x, y, z1, _panel(b))
    for z in range(mz0, mz1 + 1):
        b.place_block(x0, y, z, _panel(b))
        b.place_block(x1, y, z, _panel(b))


def _stood_on(b, c, fy):
    """Can a person stand in this column: floor under, two cells clear."""
    return (b.get_block(c[0], fy, c[1]) != "air"
            and b.get_block(c[0], fy + 1, c[1]) == "air"
            and b.get_block(c[0], fy + 2, c[1]) == "air")


def _sweep(b, rect, fy, rng):
    """Walk the finished square the way a person would and stop up any cell
    left over: paving nobody can get to is a hole in a market, and a stall's
    goods are the honest way to fill one."""
    x0, z0, x1, z1 = rect
    for _ in range(4):
        free, through = set(), set()
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if _stood_on(b, (x, z), fy):
                    free.add((x, z))
                elif "_door" in b.get_block(x, fy + 1, z):
                    through.add((x, z))      # a hung door is a way, not a wall
        through |= free
        seen = set()
        edge = [c for c in free
                if c[0] in (x0, x1) or c[1] in (z0, z1)]
        while edge:
            c = edge.pop()
            if c in seen:
                continue
            seen.add(c)
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dz)
                if n in through and n not in seen:
                    edge.append(n)
        lost = sorted(free - seen)
        if not lost:
            return
        for c in lost:
            _crate(b, c, fy, 2 if rng.random() < 0.4 else 1)


def _in_the_open(b, c, fy):
    """True where this cell and everything orthogonally beside it is clear."""
    for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        if b.get_block(c[0] + dx, fy + 1, c[1] + dz) != "air":
            return False
    return b.get_block(c[0], fy + 2, c[1]) == "air"


SOFT = ("air", "slab", "stairs", "fence", "wall", "door", "lantern", "torch",
        "carpet", "pane", "chain", "sign", "pressure", "candle", "campfire",
        "bell", "flower", "sapling", "rail", "button", "lever", "water")


def _lid(b, rects, fy):
    """Under a roof, a waist-high block with two clear cells over it is a
    perch a person can see and never reach. Board it over."""
    for x0, z0, x1, z1 in rects:
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                for y in (fy + 1, fy + 2, fy + 3):
                    low = b.get_block(x, y, z)
                    if any(s in low for s in SOFT):
                        continue
                    if (b.get_block(x, y + 1, z) == "air"
                            and b.get_block(x, y + 2, z) == "air"):
                        b.place_block(x, y + 1, z, _slab(b))


def _crate(b, c, fy, high=1, facing="north"):
    """Goods stacked and lidded. The lid matters: a bare crate is a block
    with standing room on top of it that nobody can climb to."""
    b.place_block(c[0], fy + 1, c[1], "barrel[facing=up]")
    if high > 1:
        b.place_block(c[0], fy + 2, c[1], "barrel[facing=%s]" % facing)
    b.place_block(c[0], fy + high + 1, c[1], _slab(b))


def _prop(b, c, fy, kinds, mat, facing, room="market"):
    """One piece of market gear, or a crate if the library will not have it."""
    for kind in kinds:
        if b.fitting(kind, c[0], fy + 1, c[1], facing, mat=mat, extent=1,
                     room=room)["ok"]:
            return kind
    _crate(b, c, fy, 1, facing)
    return "crate"


def _lamp(b, x, z, fy):
    """A footed post with a lantern on it, at the corner of the square."""
    b.place_block(x, fy + 1, z, _foot(b))
    b.place_block(x, fy + 2, z, _fence(b))
    b.place_block(x, fy + 3, z, _fence(b))
    b.place_block(x, fy + 4, z, "lantern[hanging=false]")


def _stall(b, rect, fy, back, opts, rng):
    """One booth: footing, frame, panel, counter, hood, roof.

        `back` is the side the solid wall stands on; the counter faces the
        opposite way, into the square.
        
    """
    x0, z0, x1, z1 = rect
    front = OPP[back]
    foot = opts["foot"]
    lock = opts["lockup"]
    band = fy + 4                      # nothing walls this booth in below here
    plate_y = band + 2 + opts["tall"]
    eave_y = plate_y + 1
    dx, dz = DELTA[front]

    corners = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
    for c in corners:
        _column(b, c, fy + 1, fy + foot, _foot(b))
        _column(b, c, fy + foot + 1, plate_y, _post(b))

    for side in ("n", "s", "e", "w"):
        for c in _edge(rect, side):
            b.place_block(c[0], plate_y, c[1], _beam(b, RUN[side]))

    # Panel between the posts -- but carried on a rail at head height, not down to the
    # paving. A market booth you cannot walk round the back of is a shed, and a square
    # full of sheds is not a square.
    flanks = [s for s in ("n", "s", "e", "w") if s not in (back, front)]
    walled = [back] + (flanks if (opts["sides"] or lock) else [])
    for side in walled:
        for c in _edge(rect, side):
            if lock:
                _column(b, c, fy + 1, fy + foot, _foot(b))
                _column(b, c, fy + foot + 1, plate_y - 1, opts["panel"])
            else:
                b.place_block(c[0], band, c[1], _beam(b, RUN[side]))
                _column(b, c, band + 1, plate_y - 1, opts["panel"])

    run = _edge(rect, front)
    gapcell = None
    if lock:
        # the lock-up: one booth on the square shuts, and has a real door in its front
        # wall with the hood carried over it
        gapcell = run[rng.randrange(len(run))]
        for c in run:
            if c != gapcell:
                _column(b, c, fy + 1, fy + foot, _foot(b))
                _column(b, c, fy + foot + 1, plate_y - 1, opts["panel"])
        got = b.doorway(gapcell[0], fy + 1, gapcell[1], FACING[back],
                        b.voice["frame"], leaf=b.joinery(b.voice, "door"),
                        jamb="build", lintel=True)
        if not got["ok"]:
            b.fill_region(gapcell[0], fy + 1, gapcell[1],
                          gapcell[0], fy + 2, gapcell[1], "air")
    else:
        # the counter, with a gap left in it: the way through to the counter side, and
        # the only one, so it is never shut
        gap = rng.randrange(len(run)) if run else 0
        for i, c in enumerate(run):
            if i == gap:
                gapcell = c
                continue
            b.place_block(c[0], fy + 1, c[1], _foot(b))
            b.place_block(c[0], fy + 2, c[1], _slab(b))

    # the hood: a beam carried out over the counter on two brackets, which is the oriel
    # of this settlement done at the size of a booth
    hood = [(c[0] + dx, c[1] + dz) for c in _edge(rect, front, inset=0)]
    for c in hood:
        b.place_block(c[0], plate_y, c[1], _beam(b, RUN[front]))
    for c in (hood[0], hood[-1]):
        b.place_block(c[0], plate_y - 1, c[1],
                      b.joinery(b.voice, "trapdoor")
                      + "[facing=%s,half=top,open=true]" % FACING[front])
    mid = hood[len(hood) // 2]
    b.place_block(mid[0], plate_y - 1, mid[1], "lantern[hanging=true]")

    # The slope is the voice's (E015): `pitch=(2, 1)` stood here until the voice
    # contract, and `roof()` is handed the voice's profile over whatever is passed.
    b.roof(x0, z0, x1, z1, eave_y, b.voice["roof"], style=opts["style"],
           axis=("z" if back in ("n", "s") else "x"),
           overhang=1, eave=opts["eave"])

    # what the stall is selling. The cell behind the gap stays clear, or the rest of the
    # booth is reachable only across a corner, which is no way in.
    inside = [(x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, z1)]
    entry = (gapcell[0] - dx, gapcell[1] - dz) if gapcell else None
    cands = [c for c in inside if c != entry]
    rng.shuffle(cands)
    for c in cands[:max(1, len(cands) - 1)]:
        _prop(b, c, fy, opts["goods"], opts["goodsmat"], FACING[front])

    # and what stands out in front of it, under the hood: a bench for the queue at one
    # end, the day's crates at the other. Never in the middle -- that is the way to the
    # counter.
    ends = [hood[0], hood[-1]]
    rng.shuffle(ends)
    for c, high in zip(ends, opts["crates"]):
        if high:
            _crate(b, c, fy, high, FACING[front])


def _quadrant(rect, corner, wlen, dlen):
    """A stall footprint tucked into one corner, a block off the edge."""
    x0, z0, x1, z1 = rect
    if corner[0] == "w":
        sx0 = x0 + 1
    else:
        sx0 = x1 - wlen
    if corner[1] == "n":
        sz0 = z0 + 1
    else:
        sz0 = z1 - dlen
    return (sx0, sz0, sx0 + wlen - 1, sz0 + dlen - 1)


def build(b, part, seed, **params):
    rng = random.Random(seed)
    paving = params.get("paving", "banded")
    canopy = params.get("canopy", "gable")
    fp = part.get("footprint") or [part["x0"], part["z0"], part["x1"],
                                   part["z1"]]
    x0, z0, x1, z1 = fp
    fy = part["floor_y"]
    rect = (x0, z0, x1, z1)
    w, d = x1 - x0 + 1, z1 - z0 + 1
    cx, cz = x0 + (w - 1) // 2, z0 + (d - 1) // 2

    b.fill_region(x0, fy + 1, z0, x1, fy + 12, z1, "air")
    _pave(b, rect, fy, paving, cx, cz, rng)

    # the well at the crossing
    cells = [(cx, cz), (cx + 1, cz), (cx, cz + 1), (cx + 1, cz + 1)]
    rng.shuffle(cells)
    wellcell = None
    for c in cells:
        got = b.fitting("well", c[0], fy + 1, c[1],
                        FACING[rng.choice(["n", "s", "e", "w"])],
                        mat=_foot(b), room="market")
        if got["ok"]:
            wellcell = c
            break
    if wellcell:
        ap = min(wellcell[0] - x0, x1 - wellcell[0],
                 wellcell[1] - z0, z1 - wellcell[1])
        if ap >= 2:
            b.ring(wellcell[0], fy, wellcell[1], 2, _panel(b))

    # four stalls, one to a corner, no two of them the same booth
    covered = []
    half = min(w, d) // 2 - 1
    wlen, dlen = min(4, half), min(3, half)
    if half >= 2:
        styles = {"gable": ["gable"], "hip": ["hip", "gable"],
                  "mixed": ["gable", "gable", "hip", "gambrel"]}[canopy]
        walled_goods = [["store", "table"], ["store"], ["table"],
                        ["table", "store"]]
        open_goods = [["store"], ["table"], ["store", "table"],
                      ["table", "store"]]
        # Two booths broad and two deep, alternating round the square, so no two roofs
        # run into one another over an entrance.
        turns = rng.choice([(0, 1, 1, 0), (1, 0, 0, 1)])
        shut = rng.randrange(4)      # one booth of the four is a lock-up shop
        for i, (corner, turned) in enumerate(zip((("w", "n"), ("e", "n"),
                                                  ("w", "s"), ("e", "s")),
                                                 turns)):
            if turned:
                srect = _quadrant(rect, corner, dlen, wlen)
                back = corner[0]
            else:
                srect = _quadrant(rect, corner, wlen, dlen)
                back = corner[1]
            opts = {
                "foot": rng.choice([1, 1, 2]),
                "tall": rng.choice([0, 0, 1]),
                "panel": rng.choice([_wall(b), _wall(b), _panel(b)]),
                "sides": rng.choice([0, 1, 1]),
                "eave": rng.choice(["straight", "straight", "upturned"]),
                "style": rng.choice(styles),
                "goodsmat": rng.choice([b.voice["floor"],
                                        b.voice["footing"]]),
                "crates": rng.choice([(1, 0), (0, 1), (2, 0), (0, 2)]),
                "lockup": i == shut,
            }
            # a shelf wants a wall to hang on; an open booth has none
            opts["goods"] = rng.choice(
                walled_goods if (opts["sides"] or opts["lockup"])
                else open_goods)
            _stall(b, srect, fy, back, opts, rng)
            covered.append((max(x0, srect[0] - 1), max(z0, srect[1] - 1),
                            min(x1, srect[2] + 1), min(z1, srect[3] + 1)))

    # lamps at the corners of the square, benches out of the way of the crossing
    for c in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        _lamp(b, c[0], c[1], fy)

    seats = [(x0 + 1, cz), (x1 - 1, cz), (cx, z0 + 1), (cx, z1 - 1),
             (x0 + 1, cz + 1), (x1 - 1, cz - 1), (cx + 1, z0 + 1),
             (cx - 1, z1 - 1)]
    rng.shuffle(seats)
    want = rng.choice([2, 3, 3, 4])
    laid = 0
    for c in seats:
        if laid >= want:
            break
        # only where nothing stands against it: a bench is stairs, and a stair with a
        # wall behind it and the paving in front reads as a step down
        if not _in_the_open(b, c, fy):
            continue
        kind = rng.choice(["bench", "bench", "trough", "table"])
        got = b.fitting(kind, c[0], fy + 1, c[1],
                        FACING[rng.choice(["n", "s", "e", "w"])],
                        mat=rng.choice([b.voice["floor"],
                                        b.voice["footing"]]),
                        room="market")
        if got["ok"]:
            laid += 1

    _lid(b, covered, fy)
    _sweep(b, rect, fy, rng)
    _lid(b, covered, fy)
    return {"ok": True}
