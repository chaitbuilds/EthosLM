"""court_large -- a large courtyard house: four ranges round an open yard.

The plan is a ring. An outer wall on the edge of the pad, a band of rooms behind it, a
boarded veranda -- the engawa -- one cell wide facing the yard under the overhanging
eave, and the yard itself left as planted ground rather than paved through. The way in
is an entrance passage driven from the lane through one range into the yard, so the yard
and not a corridor is what every room opens onto. The north range carries the upper
storey; the other three are single and low, so the house reads as one tall roof among
three quiet ones.

Nothing is placed on trust: every post, partition, flight and stick of furniture
is offered to a flood fill over the floor first, and anything that would cut one
cell of floor off from the doorway is not placed at all.
"""

import random

KIND = "plot"
FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "urban"

#: What this type delivers, by name, so a requirement can ask for it and the assembled
#: world can be asked whether it is there. The yard is the point of the building.
FEATURES = ("courtyard",)

PARAMS = {
    "storeys": ("int", 1, 3),
    "yard": ("choice", ["garden", "well", "orchard"]),
}

NEEDS = {
    # The band `scripts/type_needs.py` measured, on a plane and on a bank: clean at
    # every size from 6x6 to the 32x32 the sweep reaches, with no broken size between.
    # It read 6-9 and 19-32 with nine sizes broken in the middle until the stair defects
    # below were closed; those nine were one defect and the `except` list is empty now.
    # rounds/type-needs.json is what this line is held to.
    "footprint": (6, 6, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

SIDES = ("north", "south", "west", "east")
INWARD = {"north": "south", "south": "north", "west": "east", "east": "west"}

#: Fittings a person stands on rather than beside: the cell stays floor, and the walk
#: model reads it as floor, so the house has to go on being able to reach it.
STAND_ON = ("rug",)


def _bands(size, rng):
    """Depths of the two ranges that face each other across the yard."""
    hi = min(size - 4, size // 3 + 2)
    lo = max(2, min(hi, size // 3))
    yard = rng.randint(lo, hi)
    tot = size - yard
    a = min(6, tot - tot // 2)
    c = min(6, tot - a)
    if c < 2:
        c = 2
        a = min(6, tot - 2)
    return a, c


def _ring_cells(x0, z0, x1, z1):
    """The cells of one rectangle's own edge, clockwise-ish."""
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


def _roof_kw(part, ends, overhang, eave):
    """The voice's silhouette, with this settlement's default where it is silent."""
    r = part.get("roof")
    if not isinstance(r, dict):
        r = {}
    return {
        "profile": r.get("profile") or [(1, 2), (2, 1)],
        "ends": r.get("ends") or ends,
        "eave": eave,
        "tiers": r.get("tiers") or 1,
        "overhang": overhang,
    }


def _band_of(x, z, x0, z0, x1, z1):
    """Which range an interior cell belongs to: its nearest outside wall."""
    d = ((z - z0, "north"), (z1 - z, "south"), (x - x0, "west"), (x1 - x, "east"))
    return min(d)[1]


def _walk(free, start):
    """Every cell of `free` a person can reach from `start`, four ways."""
    if start not in free:
        return set()
    seen = set([start])
    stack = [start]
    while stack:
        cx, cz = stack.pop()
        for n in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
            if n in free and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


def build(b, part, seed, **params):
    rng = random.Random((seed * 1103515245 + 12345) % 2147483647)
    v = part["voice"]
    x0, z0 = part["x0"], part["z0"]
    x1, z1 = part["x1"], part["z1"]
    fy = part["floor_y"]
    W, D = x1 - x0 + 1, z1 - z0 + 1

    wall_b = b.block(v["wall"], "full")
    frame_b = b.block(v["frame"], "full")
    foot_b = b.block(v["footing"], "full")
    trim_b = b.block(v["trim"], "full")
    floor_b = b.block(v["floor"], "full")
    screen = b.joinery(v, "fence")
    leaf = b.joinery(v, "door")

    dx, dz = part["door"][0], part["door"][1]
    if dx <= x0:
        gate = "west"
    elif dx >= x1:
        gate = "east"
    elif dz <= z0:
        gate = "north"
    else:
        gate = "south"

    # ---- the four ranges and the yard they stand round ---------------------
    dn, ds = _bands(D, rng)
    dw, de = _bands(W, rng)
    if rng.random() < 0.5:
        dw, de = de, dw
    yx0, yx1 = x0 + dw, x1 - de
    yz0, yz1 = z0 + dn, z1 - ds
    ex0, ex1 = yx0 - 1, yx1 + 1          # the engawa ring, one cell wide
    ez0, ez1 = yz0 - 1, yz1 + 1
    yard_w, yard_d = yx1 - yx0 + 1, yz1 - yz0 + 1

    want = max(1, min(3, int(params.get("storeys", 2) or 2)))
    storey = 4 if want > 1 else rng.choice([4, 5])
    wall_h = storey

    ring = _ring_cells(ex0, ez0, ex1, ez1)
    ringset = set(ring)
    yard = set((x, z) for x in range(yx0, yx1 + 1) for z in range(yz0, yz1 + 1))

    # ---- floors: boards through the ranges, ground left in the yard --------
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if (x, z) in yard:
                continue
            b.place_block(x, fy, z, trim_b if (x, z) in ringset else floor_b)

    # ---- the entrance passage, lane to yard through one range --------------
    passage = []
    if gate == "west":
        passage += [(x, dz) for x in range(x0, ex0 + 1)]
        if dz < yz0:
            passage += [(ex0, z) for z in range(dz, yz0 + 1)]
        elif dz > yz1:
            passage += [(ex0, z) for z in range(yz1, dz + 1)]
    elif gate == "east":
        passage += [(x, dz) for x in range(ex1, x1 + 1)]
        if dz < yz0:
            passage += [(ex1, z) for z in range(dz, yz0 + 1)]
        elif dz > yz1:
            passage += [(ex1, z) for z in range(yz1, dz + 1)]
    elif gate == "north":
        passage += [(dx, z) for z in range(z0, ez0 + 1)]
        if dx < yx0:
            passage += [(x, ez0) for x in range(dx, yx0 + 1)]
        elif dx > yx1:
            passage += [(x, ez0) for x in range(yx1, dx + 1)]
    else:
        passage += [(dx, z) for z in range(ez1, z1 + 1)]
        if dx < yx0:
            passage += [(x, ez1) for x in range(dx, yx0 + 1)]
        elif dx > yx1:
            passage += [(x, ez1) for x in range(yx1, dx + 1)]
    passage = set(passage)

    # ---- the floor a person can stand on, and what may be put on it -------
    floor_cells = set((x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, z1))
    blocked = set()      # every cell something of mine stands in
    solid = set()        # the structural part of that: walls, posts, treads
    #: Cells nothing may be put in, because somebody has to stand in them: the cell at
    #: the foot of each flight. Marking one `blocked` would be the opposite of what is
    #: meant -- a blocked cell is out of `would_shut`'s free set, so from then on
    #: nothing protects the way to it, and a chest set down beside a stair walled its
    #: foot into a one-cell pocket, which the dead-corner pass at the end then posted up
    #: solid. Two storeys nobody could reach, off one cell.
    keep = set()
    start = (yx0, yz0)

    def would_shut(cells):
        """Would taking these floor cells cut any of the rest off from the door?"""
        want_cells = [c for c in cells if c in floor_cells and c not in blocked]
        if not want_cells:
            return False
        if [c for c in want_cells if c in passage or c in keep]:
            return True
        free = floor_cells - blocked - set(want_cells)
        return len(_walk(free, start)) != len(free)

    def claim(cells):
        """Take these floor cells if the rest of the floor stays walkable."""
        if would_shut(cells):
            return False
        blocked.update(c for c in cells if c in floor_cells)
        return True

    def build_on(cells):
        """The same, for something you can never walk through."""
        if not claim(cells):
            return False
        solid.update(c for c in cells if c in floor_cells)
        return True

    # ---- the outer wall, boarded between a heavy frame ---------------------
    top = fy + wall_h
    spacing = rng.choice([3, 4, 5])
    for run in ((x0, z0, x1, z0), (x0, z1, x1, z1), (x0, z0, x0, z1),
                (x1, z0, x1, z1)):
        b.wall(run[0], fy + 1, run[1], run[2], top, run[3], wall_b, post=frame_b,
               spacing=spacing, base=foot_b, base_height=1, band=trim_b)

    # openings: wide rather than tall, screened rather than glazed
    ow = 3 if W >= 11 else 2
    od = 3 if D >= 11 else 2
    osp = rng.choice([4, 5, 6])
    if W > 4:
        b.openings(x0 + 1, fy, z0, x1 - 1, z0, spacing=osp, width=ow,
                   sill=2, head=3, block=screen)
        b.openings(x0 + 1, fy, z1, x1 - 1, z1, spacing=osp, width=ow,
                   sill=2, head=3, block=screen)
    if D > 4:
        b.openings(x0, fy, z0 + 1, x0, z1 - 1, spacing=osp, width=od,
                   sill=2, head=3, block=screen)
        b.openings(x1, fy, z0 + 1, x1, z1 - 1, spacing=osp, width=od,
                   sill=2, head=3, block=screen)

    # ---- partitions between the ranges, the veranda left to link them ------
    runs = []
    if ez0 - z0 >= 2:
        runs.append((ex0, z0 + 1, ez0 - 1))
        runs.append((ex1, z0 + 1, ez0 - 1))
    if z1 - ez1 >= 2:
        runs.append((ex0, ez1 + 1, z1 - 1))
        runs.append((ex1, ez1 + 1, z1 - 1))
    for (px, pa, pb) in runs:
        cells = [(px, z) for z in range(pa, pb + 1)]
        if not build_on(cells):
            continue
        for (cx, cz) in cells:
            for y in range(fy + 1, fy + wall_h):
                b.place_block(cx, y, cz, wall_b)
            b.place_block(cx, fy + wall_h, cz, frame_b)

    # ---- the veranda: a plate under the eave, posts standing clear ---------
    for (px, pz) in ring:
        b.place_block(px, fy + wall_h, pz, trim_b)
    psp = rng.choice([2, 3])
    posts = [c for c in ((ex0, ez0), (ex1, ez0), (ex0, ez1), (ex1, ez1))]
    if yard_w >= 4 and yard_d >= 4:
        for (px, pz) in _ring_cells(yx0, yz0, yx1, yz1):
            if (abs(px - yx0) + abs(pz - yz0)) % (psp + 1) == 0:
                posts.append((px, pz))
    for c in posts:
        if build_on([c]):
            for y in range(fy + 1, fy + wall_h):
                b.place_block(c[0], y, c[1], frame_b)

    # ---- how tall the north range stands ----------------------------------- **Where
    # the stair goes is decided on free cells, not on free floor.** A run of five treads
    # and the cell you set off from is six columns of one row, and `would_shut` says yes
    # to a run that crosses a partition: a cell something already stands in is simply
    # taken as far as furniture is concerned, and is filtered out of the test. A flight
    # cannot share one. `flight()` will not drive a stair through a wall -- it says so
    # and lays nothing -- and the storey above then stands with a floor, four walls and
    # no way up, which is what E003 and E011 are and what this type was reported for at
    # six of its nine parameter sets. Asking for free cells is also what keeps the whole
    # flight inside one compartment of the range, so the foot of it is in the same room
    # as the head.
    rows = list(range(z0 + 1, ez0))
    ns = 1
    flights = []
    if rows and W >= storey + 5:
        heading = rng.choice(["east", "west"])
        for s in range(1, want):
            got = None
            for z in rows:
                step = 1 if heading == "east" else -1
                span = range(x0 + 2, x1 - storey) if heading == "east" \
                    else range(x1 - 2, x0 + storey, -1)
                for fx in span:
                    cells = [(fx + step * i, z) for i in range(0, storey + 1)]
                    foot = (fx - step, z)
                    run = cells + [foot]
                    if [c for c in run if not (x0 + 1 <= c[0] <= x1 - 1)]:
                        continue
                    if [c for c in run if c in blocked or c in passage
                            or c in keep]:
                        continue
                    # The treads and the landing are built on; the cell you set off from
                    # is stood in, and is kept rather than taken -- so the flood fill
                    # goes on holding a way to it.
                    if build_on(cells):
                        keep.add(foot)
                        got = (fx, z, heading, cells, foot)
                        break
                if got:
                    break
            if not got:
                break
            flights.append(got)
            ns += 1
            heading = "west" if heading == "east" else "east"

    # **The stair before the storey.** A floor laid over a flight that was refused is a
    # sealed room, and the refusal is the library saying so in words. So each storey is
    # entered before it is built: the flight goes up, and only if it stands does the
    # floor, the walls and the windows follow. If it does not, the house stops at the
    # height it can be walked to.
    upper = []
    built = 1
    for s in range(1, ns):
        fys = fy + storey * s
        fx, fz, heading, cells, _foot = flights[s - 1]
        # The stairwell is a hole in the floor above it and stays one: the flight cuts
        # the cells over its treads, and laying the floor back across them would be the
        # ceiling `flight()` exists to open.
        well = set(cells[:-1])
        room = set((x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, ez0))
        # **And a storey a person cannot cross is not built at all.** Two stairs stand
        # in this floor: the well of the one arriving, which is a hole in it, and the
        # first three columns of the one leaving -- the tread, then the cell each tread
        # is carried on, which stands at head height one column along. A range two rows
        # deep with a well across one and a stair across the other is a storey in two
        # halves and the far half is what E003 and E011 report. So the floor is walked
        # here, before a block of it is laid, and a storey that does not come back whole
        # is the height the house stops at.
        stands_in, nxt = set(), set()
        if s < len(flights):
            stands_in = set(flights[s][3][:3])
            nxt = stands_in | {flights[s][4]}
        free = room - well - stands_in
        if not free or len(_walk(free, cells[-1])) != len(free):
            break
        r = b.flight(part["label"], fx, fz, fy + storey * (s - 1), fys, heading,
                     mat=v["floor"])
        if not (isinstance(r, dict) and r.get("ok")):
            break
        for x in range(x0, x1 + 1):
            for z in range(z0, ez0 + 1):
                if (x, z) not in well:
                    b.place_block(x, fys, z, floor_b)
        for run in ((x0, z0, x1, z0), (x0, ez0, x1, ez0), (x0, z0, x0, ez0),
                    (x1, z0, x1, ez0)):
            b.wall(run[0], fys + 1, run[1], run[2], fys + wall_h, run[3], wall_b,
                   post=frame_b, spacing=spacing, base=trim_b, base_height=1,
                   band=trim_b)
        if W > 4:
            b.openings(x0 + 1, fys, z0, x1 - 1, z0, spacing=osp, width=ow,
                       sill=2, head=3, block=screen)
            b.openings(x0 + 1, fys, ez0, x1 - 1, ez0, spacing=osp, width=ow,
                       sill=2, head=3, block=screen)
        # ...and what the storey is, is what a person can walk to from the head of its
        # own stair, less what the next one stands in: its treads and the cell somebody
        # sets off from are somewhere to walk over and nowhere to put a chest. A candle
        # stood in that one cell is a storey nobody reaches.
        reach = free - nxt
        upper.append((fys, reach, cells[-1]))
        built += 1
    ns = built

    # ---- roofs: the biggest thing about the building ----------------------- Four
    # roofs round a yard is four chances for one roof's overhang to land on the slope of
    # the next, which reads as a broken pitch and is not one. So every rect is pulled in
    # by the overhang it is going to throw, and each roof ends up covering its own
    # range, the lane outside it and nothing else: the long ranges hold the outward
    # eave, the side ranges the one over the yard.
    oh_x = 1 if yard_w >= 3 and yard_d >= 3 else 0
    kw_side = _roof_kw(part, ("hip", "hip"), oh_x, "upturned")
    kw_main = _roof_kw(part, ("irimoya", "irimoya"), 1, "upturned")
    rz0, rz1 = (yz0 + 1, yz1 - 1) if oh_x else (yz0, yz1)
    low = fy + wall_h + 1
    high = fy + storey * (ns - 1) + wall_h + 1
    ridge = {}
    ridge["west"] = b.roof(x0, rz0, ex0, rz1, low, v["roof"], axis="x", **kw_side)
    ridge["east"] = b.roof(ex1, rz0, x1, rz1, low, v["roof"], axis="x", **kw_side)
    ridge["south"] = b.roof(x0, ez1 + 1, x1, z1, low, v["roof"], axis="z", **kw_main)
    ridge["north"] = b.roof(x0, z0, x1, ez0 - 1, high, v["roof"], axis="z",
                            **kw_main)
    for k in SIDES:
        if not isinstance(ridge.get(k), int):
            ridge[k] = low + 3

    # ---- furniture, none of it in the way ---------------------------------- what
    # stands in a room, as against what is hung over it or fixed to its wall: a lit room
    # with nothing in it is still a room with nothing in it, so lanterns and wall
    # shelves do not count towards furnishing one
    standing = set()

    def _footprint(r, anchor):
        out = set([anchor])
        for c in (r.get("cells") or []):
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                out.add((c[0], c[2]))
            elif isinstance(c, (list, tuple)) and len(c) == 2:
                out.add((c[0], c[1]))
        return out

    def furnish(cells, kind, facing, mat, room=None, extent=1, flue=None,
                y=None, test=None, mark=None, limit=10):
        """One piece of equipment, offered to the flood fill before it is placed."""
        yy = fy + 1 if y is None else y
        tries = 0
        for c in list(cells):
            if tries >= limit:
                return None
            tries += 1
            kw = {"mat": mat, "extent": extent}
            if room is not None:
                kw["room"] = room
            if flue is not None:
                kw["flue_to"] = flue
            # **What a fitting takes is the library's answer, not this file's guess.**
            # It used to be the fitting's own cell and the two beside it -- a fair
            # picture of a chest against a wall and a wrong one of a well, which is a
            # 3x3 curb round an open shaft. The yard is the room every other room opens
            # onto, and one well tested on three cells walled it in two. `dry=True` lays
            # nothing and says which cells it would fill; the cell you use the piece
            # from is not among them, which is right, because that is the cell the room
            # is walked through.
            probe = b.fitting(kind, c[0], yy, c[1], facing, dry=True, **kw)
            near = list({(px, pz) for (px, _py, pz) in (probe.get("cells") or ())}
                        | {c})
            if test is not None and test(near):
                cells.remove(c)
                continue
            r = b.fitting(kind, c[0], yy, c[1], facing, **kw)
            if isinstance(r, dict) and r.get("ok"):
                took = _footprint(r, c)
                if kind in STAND_ON and y is None:
                    # **A rug is floor.** A carpet is laid where people walk and the
                    # walk model counts the cell under it as somewhere to stand, so
                    # taking it as `blocked` is the opposite of what happened: the cell
                    # leaves the flood fill's free set, nothing protects the way to it
                    # any more, and the next three things put down round it walled a rug
                    # into a one-cell room. Kept, so it stays reachable and nothing else
                    # is put on it.
                    keep.update(t for t in took if t in floor_cells)
                elif mark is not None:
                    mark(took)
                if kind not in ("light", "shelf"):
                    standing.update(took)
                for d in list(cells):
                    if d in took or abs(d[0] - c[0]) + abs(d[1] - c[1]) <= 1:
                        cells.remove(d)
                return r
        return None

    def mark_ground(took):
        blocked.update(t for t in took if t in floor_cells)

    # the passage is plugged while the furniture goes in, so that nothing standing
    # against a wall can reach a foot into the way in
    for (px, pz) in passage:
        b.place_block(px, fy + 1, pz, wall_b)

    bands = dict((s, []) for s in SIDES)
    for (x, z) in floor_cells:
        if (x, z) in yard or (x, z) in ringset or (x, z) in blocked:
            continue
        if (x, z) in passage:
            continue
        bands[_band_of(x, z, x0, z0, x1, z1)].append((x, z))
    for side in SIDES:
        if not bands[side]:
            bands[side] = [c for c in ring if c not in passage
                           and c not in blocked
                           and _band_of(c[0], c[1], x0, z0, x1, z1) == side]
        bands[side].sort(key=lambda c: (min(c[0] - x0, x1 - c[0],
                                            c[1] - z0, z1 - c[1]), rng.random()))

    programs = ["sleep", "store", "work"]
    rng.shuffle(programs)
    jobs = {"north": "hall"}
    for i, side in enumerate([s for s in SIDES if s != "north"]):
        jobs[side] = programs[i]
    for side in SIDES:
        job = jobs[side]
        cells = bands[side]
        face = INWARD[side]
        if job == "hall" and ns == 1:
            # an open hearth in the floor, no stack: a chimney driven up through a
            # thatch slope is what the eave stairs break against
            furnish(cells, "hearth", face, v["footing"], room="hall",
                    test=would_shut, mark=mark_ground)
        if job == "hall":
            furnish(cells, "table", face, v["trim"], room="hall", test=would_shut, mark=mark_ground)
            furnish(cells, "shelf", face, v["trim"], room="hall", test=would_shut, mark=mark_ground)
            furnish(cells, "light", face, v["trim"], room="hall", test=would_shut, mark=mark_ground)
        elif job == "sleep":
            furnish(cells, "bed", face, v["wall"], room="house", test=would_shut, mark=mark_ground)
            furnish(cells, "shelf", face, v["trim"], room="house", test=would_shut, mark=mark_ground)
            furnish(cells, "rug", face, v["trim"], room="house", test=would_shut, mark=mark_ground)
            furnish(cells, "light", face, v["trim"], room="house", test=would_shut, mark=mark_ground)
        elif job == "store":
            furnish(cells, "store", face, v["wall"], room="store",
                    extent=min(3, max(1, len(cells) // 3)), test=would_shut, mark=mark_ground)
            furnish(cells, "shelf", face, v["trim"], room="store", test=would_shut, mark=mark_ground)
            furnish(cells, "light", face, v["trim"], room="store", test=would_shut, mark=mark_ground)
        else:
            if ns > 1:
                furnish(cells, "oven", face, v["footing"], room="house",
                        test=would_shut, mark=mark_ground)
            furnish(cells, "workbench", face, v["trim"], room="house",
                    test=would_shut, mark=mark_ground)
            furnish(cells, "bookshelf", face, v["trim"], room="house",
                    test=would_shut, mark=mark_ground)
            furnish(cells, "rug", face, v["trim"], room="house", test=would_shut, mark=mark_ground)
            furnish(cells, "light", face, v["trim"], room="house", test=would_shut, mark=mark_ground)

    # ---- no stretch of floor left saying nothing -------------------------- A range
    # twenty cells long furnished only at one end is a room with nothing in it at the
    # other, so the house is swept for floor that is far from anything and given
    # something there.
    spread = [c for c in (floor_cells - yard - solid)
              if c not in passage and c not in blocked]
    spread.sort(key=lambda c: (min(c[0] - x0, x1 - c[0],
                                   c[1] - z0, z1 - c[1]), c))
    kinds = ["store", "shelf", "workbench", "bookshelf", "table"]
    n = 0
    for c in spread:
        if standing and min(abs(c[0] - f[0]) + abs(c[1] - f[1])
                            for f in standing) <= 4:
            continue
        side = _band_of(c[0], c[1], x0, z0, x1, z1)
        room = "store" if c in ringset else "house"
        if furnish([c], kinds[n % len(kinds)], INWARD[side], v["trim"],
                   room=room, test=would_shut, mark=mark_ground) is not None:
            n += 1
        else:
            furnish([c], "light", INWARD[side], v["trim"], room=room,
                    test=would_shut, mark=mark_ground)

    # lanterns down the veranda on a rhythm of their own, so that the length of it reads
    # as somewhere people sit rather than as a corridor
    kinds = ("table", "store", "rug", "bookshelf")
    for i, c in enumerate(sorted(ring)):
        if c in passage or c in blocked:
            continue
        side = _band_of(c[0], c[1], x0, z0, x1, z1)
        if (c[0] + c[1]) % 7 == 0:
            furnish([c], "light", INWARD[side], v["trim"], room="hall",
                    test=would_shut, mark=mark_ground)
        elif (c[0] + c[1]) % 5 == 2:
            for kind in kinds:
                if furnish([c], kind, INWARD[side], v["trim"], room="house",
                           test=would_shut, mark=mark_ground) is not None:
                    break

    # and again room by room: the veranda, which is one room the whole way round however
    # many posts stand in it, and then each room behind it
    rooms = [(set(ringset), ("table", "store", "rug", "bookshelf"))]
    left = floor_cells - yard - ringset - solid
    while left:
        comp = _walk(left, min(left))
        left = left - comp
        rooms.append((comp, ("shelf", "store", "table")))
    for (comp, kinds) in rooms:
        if len(comp) < 2 or [c for c in comp if c in standing]:
            continue
        cand = [c for c in comp if c not in blocked and c not in passage]
        cand.sort(key=lambda c: (min(c[0] - x0, x1 - c[0],
                                     c[1] - z0, z1 - c[1]), c))
        if not cand:
            continue
        side = _band_of(cand[0][0], cand[0][1], x0, z0, x1, z1)
        for kind in kinds:
            if not cand:
                break
            if furnish(cand, kind, INWARD[side], v["trim"], room="house",
                       test=would_shut, mark=mark_ground,
                       limit=40) is not None:
                break

    # ---- upstairs, over the north range ------------------------------------
    for (fys, room, landing) in upper:
        up_blocked = set()

        def up_shut(cells, room=room, landing=landing, up_blocked=up_blocked):
            want_cells = [c for c in cells if c in room and c not in up_blocked]
            if not want_cells:
                return False
            free = room - up_blocked - set(want_cells)
            return len(_walk(free, landing)) != len(free)

        def up_mark(took, room=room, up_blocked=up_blocked):
            up_blocked.update(t for t in took if t in room)

        cells = [c for c in room if c != landing]
        cells.sort(key=lambda c: (min(c[0] - x0, x1 - c[0], c[1] - z0),
                                  rng.random()))
        furnish(cells, "bed", "south", v["wall"], room="house", y=fys + 1,
                test=up_shut, mark=up_mark)
        furnish(cells, "shelf", "south", v["trim"], room="house", y=fys + 1,
                test=up_shut, mark=up_mark)
        furnish(cells, "bookshelf", "south", v["trim"], room="house",
                y=fys + 1, test=up_shut, mark=up_mark)
        furnish(cells, "light", "south", v["trim"], room="house", y=fys + 1,
                test=up_shut, mark=up_mark)

    # ---- the yard: planted ground, not a paved through-way ------------------
    plot = [c for c in yard if c not in blocked and c not in passage]
    plot.sort(key=lambda c: (abs(c[0] - (yx0 + yx1) / 2.0)
                             + abs(c[1] - (yz0 + yz1) / 2.0), rng.random()))
    kind = params.get("yard", "garden")
    if kind == "well":
        furnish(plot, "well", INWARD[gate], v["footing"], room="courtyard",
                test=would_shut, mark=mark_ground)
        furnish(plot, "trough", "north", v["trim"], room="courtyard",
                test=would_shut, mark=mark_ground)
    elif kind == "orchard":
        for i in range(0, 3):
            furnish(plot, "fodder", "north", v["trim"], room="courtyard",
                    test=would_shut, mark=mark_ground)
        furnish(plot, "trough", "east", v["trim"], room="courtyard",
                test=would_shut, mark=mark_ground)
    else:
        for i in range(0, 2):
            furnish(plot, "trough", "north", v["trim"], room="courtyard",
                    test=would_shut, mark=mark_ground)
        furnish(plot, "trough", INWARD[gate], v["trim"], room="courtyard",
                test=would_shut, mark=mark_ground)
    furnish(plot, "light", "north", v["trim"], room="courtyard",
            test=would_shut, mark=mark_ground)
    # a yard with nothing in it is a yard nobody gardens: if none of that took, work
    # down what a yard can hold until something does
    if not [c for c in yard if c in standing]:
        for kind in ("well", "trough", "fodder", "store", "table", "rug"):
            if furnish(plot, kind, INWARD[gate], v["footing"],
                       room="courtyard", test=would_shut, mark=mark_ground,
                       limit=40) is not None:
                break

    # ---- open the way in again and hang the door ---------------------------
    for (px, pz) in passage:
        b.place_block(px, fy, pz, foot_b)
        for y in range(fy + 1, fy + wall_h):
            b.place_block(px, y, pz, "air")
    b.doorway(dx, fy + 1, dz, gate, v["wall"], leaf=leaf, jamb="build",
              lintel=True)

    # ---- read the finished floor back and post up any dead corner ----------
    stand = set()
    for c in floor_cells:
        if (b.get_block(c[0], fy + 1, c[1]) == "air"
                and b.get_block(c[0], fy + 2, c[1]) == "air"):
            stand.add(c)
    doorstep = [c for c in passage if c in stand]
    if doorstep:
        # ...but never the foot of a stair: a dead corner is posted up because nothing
        # happens there, and the way up to a storey is not that.
        dead = (stand - _walk(stand, min(doorstep))) - keep
        if 0 < len(dead) <= 12:
            for c in dead:
                for y in range(fy + 1, fy + wall_h):
                    b.place_block(c[0], y, c[1], frame_b)

    b.dress_ground(yx0, yz0, yx1, yz1)
    b.seal_voids(x0, z0, x1, z1, wall_b)
    b.check_door(dx, fy + 1, dz)
    b.check_walkable(part["label"])
    b.check_attached()
    # **The court is declared, so the assembled world can be asked about it.** The
    # composition round: this type lays four ranges round a yard and published nothing
    # about it, so `usable.court_accessible` answered `unsupported` on every one of the
    # twenty-five courts the section built -- the registered relationship "courts that
    # are enclosed and entered" could not be measured at all, and an unmeasured
    # relationship holds nothing. `courtyard` is in `construction.OPEN_FEATURES`, so
    # what is verified is that the yard is *open* ground reachable from the house, which
    # is what a court is.
    return {"ranges": 4, "storeys": ns, "yard": (yx0, yz0, yx1, yz1),
            "emitted": {"storeys": int(ns),
                        "features": {"courtyard": True, "chimney": False},
                        "rects": {"main": [int(x0), int(z0), int(x1), int(z1)],
                                  "courtyard": [int(yx0), int(yz0), int(yx1),
                                                int(yz1)]}}}
