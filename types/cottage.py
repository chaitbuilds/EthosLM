"""A cottage: a low roof pitched hard against weather, a thick-shouldered mass, and a
lean-to outshot down one long side.

This is a **form** and not a voice. Every material comes from the settlement's palette
-- `b.voice[role]`, the same six roles `part["voice"]` carries -- so the same cottage
stands in dry field-stone, in ochre stone under green tile, or in blackstone, and is the
same building in all three.
"""

import random

FORM = "european_vernacular"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "rural"

PARAMS = {
    "storeys": ("int", 1, 3),
    "outshot": ("choice", ["byre", "store", "scullery"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3x3 to 10x10 measured; above ten its own upper floor comes back as a room the
    # stair does not reach. Measured by scripts/type_needs.py; the band is rounds/type-
    # needs.json. The pair is the pad site() hands build(), after siting's inset.
    "footprint": (3, 3, 24, 24),
    "except": (11, 12, 13, 14, 15, 16, 17, 18, 19, 21),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _door_side(part):
    """Which edge of the pad the reserved doorstep sits on."""
    x, z = part["door"][0], part["door"][-1]
    d = {"west": abs(x - part["x0"]), "east": abs(part["x1"] - x),
         "north": abs(z - part["z0"]), "south": abs(part["z1"] - z)}
    return min(d, key=lambda k: d[k])


def _shrink(rect, side, n):
    x0, z0, x1, z1 = rect
    if side == "west":
        x0 += n
    elif side == "east":
        x1 -= n
    elif side == "north":
        z0 += n
    else:
        z1 -= n
    return (x0, z0, x1, z1)


def _axis_len(rect, side):
    x0, z0, x1, z1 = rect
    if side in ("east", "west"):
        return x1 - x0 + 1
    return z1 - z0 + 1


def _pick(bits, salt, n):
    """One decision out of the seed. Different salts move independently, so
    two seeds differ in the massing and not only in the furniture."""
    h = (bits ^ (salt * 0x9E3779B1)) & 0xFFFFFFFF
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h % n


def _cap(rect, side, dx, dz):
    """How far that wall may come in and still leave the doorstep in it,
    with a block of wall beside the door."""
    x0, z0, x1, z1 = rect
    if side == "north":
        room = dz - z0
    elif side == "south":
        room = z1 - dz
    elif side == "west":
        room = dx - x0
    else:
        room = x1 - dx
    return max(0, room - 1)


def _face_to_center(x, z, rect):
    x0, _, z0, x1, z1 = rect
    cx = (x0 + x1) / 2.0
    cz = (z0 + z1) / 2.0
    if abs(cx - x) >= abs(cz - z):
        if cx > x:
            return "east"
        if cx < x:
            return "west"
    if cz > z:
        return "south"
    if cz < z:
        return "north"
    return "south"


def _collect(obj, out, depth=0):
    """Every (x, z) column named anywhere inside a result the library handed back."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        ints = [v for v in obj if isinstance(v, int) and not isinstance(v, bool)]
        if len(obj) == 3 and len(ints) == 3:
            out.add((obj[0], obj[2]))
            return
        if len(obj) == 2 and len(ints) == 2:
            out.add((obj[0], obj[1]))
            return
        for v in obj:
            _collect(v, out, depth + 1)


def _stand_y(b, x0, y, z0, x1, z1):
    """rooms[] gives a floor level; the cell a fitting stands in is the air above it."""
    probes = [(x0, z0), (x1, z1), (x0, z1), (x1, z0),
              ((x0 + x1) // 2, (z0 + z1) // 2)]
    for (x, z) in probes:
        if b.get_block(x, y, z) == "air":
            return y
    return y + 1


def _room_cells(x0, z0, x1, z1, blocked, rng, edge_first=True):
    edge = []
    mid = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if (x, z) in blocked:
                continue
            if x in (x0, x1) or z in (z0, z1):
                edge.append((x, z))
            else:
                mid.append((x, z))
    rng.shuffle(edge)
    rng.shuffle(mid)
    if edge_first:
        return edge + mid
    return mid + edge


def _put(b, kind, cells, y, rect, **kw):
    """Walk candidate cells until the room accepts the piece. Refusal is not
    failure: it is the room saying something is already standing there."""
    for (x, z) in cells:
        r = b.fitting(kind, x, y, z, _face_to_center(x, z, rect), **kw)
        if isinstance(r, dict) and r.get("ok"):
            return (x, z)
    return None


def _public(b, part):
    """Which edges of the pad have lane running along them. The eaves, the
    plinth skirt and the doorstep all have to keep off that ground."""
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    mx = (px0 + px1) // 2
    mz = (pz0 + pz1) // 2
    edges = {"north": [(px0, pz0), (mx, pz0), (px1, pz0)],
             "south": [(px0, pz1), (mx, pz1), (px1, pz1)],
             "west": [(px0, pz0), (px0, mz), (px0, pz1)],
             "east": [(px1, pz0), (px1, mz), (px1, pz1)]}
    ds = _door_side(part)
    fy = part["floor_y"]
    out = {}
    for s in ("north", "south", "east", "west"):
        stand = 1 if s == ds else 0
        for (x, z) in edges[s]:
            r = b.nearest_lane(x, z)
            if not isinstance(r, dict):
                continue
            d = r.get("distance")
            if not isinstance(d, (int, float)) or d > 2.5:
                continue
            ly = r.get("y")
            # a lane running well below the pad is reached by a battered skirt that
            # steps out as it goes down: stand further off it
            deep = isinstance(ly, int) and ly < fy - 1
            stand = max(stand, 2 if deep else 1)
        out[s] = stand
    return out


def _plan(part, rng, out_use, storeys, public, bits):
    """Where the house sits on the pad, which long side carries the outshot."""
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    plot = (px0, pz0, px1, pz1)
    ds = _door_side(part)
    dx, dz = part["door"][0], part["door"][-1]
    main = plot

    # keep the whole build -- skirt and eaves included -- off public ground
    for axis in (("west", "east"), ("north", "south")):
        want = [(s, public.get(s, 0)) for s in axis if public.get(s, 0)]
        want += [(s, 1) for s in axis if not public.get(s, 0)
                 and _axis_len(plot, s) >= 9]
        for (s, n) in want:
            lim = 5 if s == ds else 6
            for _ in range(n):
                if _axis_len(main, s) - 1 >= lim and (s == ds
                                                      or _cap(main, s, dx, dz) >= 1):
                    main = _shrink(main, s, 1)
    # and then how much of the rest of the pad this one takes: seed's business
    slack = [s for s in ("north", "south", "east", "west") if s != ds]
    rng.shuffle(slack)
    for i, s in enumerate(slack):
        n = [0, 0, 1, 2][_pick(bits, 10 + i, 4)]
        for _ in range(n):
            if _axis_len(main, s) - 1 >= 7 and _cap(main, s, dx, dz) >= 1:
                main = _shrink(main, s, 1)

    W = _axis_len(main, "east")
    D = _axis_len(main, "north")
    if W > D + 1:
        longs = ["north", "south"]
    elif D > W + 1:
        longs = ["east", "west"]
    else:
        longs = ["north", "south", "east", "west"]
    # a lean-to one block deep inside is a cupboard with no door: three or none
    cands = [s for s in longs if s != ds and _cap(main, s, dx, dz) >= 3
             and _axis_len(main, s) - 3 >= 5]
    if not cands:
        cands = [s for s in OPP if s != ds and _cap(main, s, dx, dz) >= 3
                 and _axis_len(main, s) - 3 >= 5]
    depth = 0
    out_side = ds
    if cands:
        cands = sorted(cands)
        out_side = cands[_pick(bits, 20, len(cands))]
        depth = 3
        # a byre wants stalls and a passage: deeper where the pad can carry it
        if (out_use == "byre" or _pick(bits, 21, 2)) \
                and _cap(main, out_side, dx, dz) >= 4 \
                and _axis_len(main, out_side) - 4 >= 6:
            depth = 4
        main = _shrink(main, out_side, depth)

    mw = main[2] - main[0] + 1
    md = main[3] - main[1] + 1
    inner = (mw - 2, md - 2)
    if min(inner) < 3 or max(inner) < 5:
        storeys = 1
    elif storeys > 2 and max(inner) < 7:
        storeys = 2
    axis = "z" if mw >= md else "x"
    # Plain gable ends keep the low eaves clear of uphill ground; vary the pitch and
    # footprint while retaining the cottage's weather roof.
    style = "gable"
    pitch = [(2, 1), (1, 1)][_pick(bits, 23, 2)]
    spec = {"style": style, "axis": axis, "pitch": pitch}
    if style == "gable" and _pick(bits, 24, 4) == 0:
        spec["ends"] = ("half-hip", "gable") if _pick(bits, 25, 2) else \
                       ("gable", "half-hip")
    # the stack goes up a gable end, never out of an eave
    gables = ["east", "west"] if axis == "z" else ["north", "south"]
    chim = [s for s in gables if s != ds and s != out_side]
    if not chim:
        chim = [s for s in gables if s != out_side] or gables
    chim = sorted(chim)
    return {"main": main, "out_side": out_side, "depth": depth,
            "storeys": storeys, "roof": spec, "door_side": ds,
            "chimney": chim[_pick(bits, 26, len(chim))],
            "dormers": (1 if md * mw < 90 else 2) if storeys >= 2 else 0,
            "plot": plot}


def _way_in(main, side, dx, dz):
    """The floor a person walks over coming in off the lane. Furniture on any
    of it is a room you can only enter by jumping over the table."""
    x0, z0, x1, z1 = main
    out = set()
    for k in range(0, 3):
        for d in (-1, 0, 1):
            if side in ("west", "east"):
                wx = (x0 + k) if side == "west" else (x1 - k)
                out.add((wx, dz + d))
            else:
                wz = (z0 + k) if side == "north" else (z1 - k)
                out.add((dx + d, wz))
    return out


def _inner_row(rect, main):
    """The row of an outshot that faces the house it leans on -- where the way
    through from the hall has to come out."""
    x0, _, z0, x1, z1 = rect
    mx = (main[0] + main[2]) / 2.0
    mz = (main[1] + main[3]) / 2.0
    out = set()
    if abs(mx - (x0 + x1) / 2.0) >= abs(mz - (z0 + z1) / 2.0):
        col = x0 if mx < x0 else x1
        for z in range(z0, z1 + 1):
            out.add((col, z))
    else:
        row = z0 if mz < z0 else z1
        for x in range(x0, x1 + 1):
            out.add((x, row))
    return out


def _side_cells(rect, side, blocked):
    """Cells along one wall of a room, middle of the wall first."""
    x0, y, z0, x1, z1 = rect
    if side == "north":
        cells = [(x, z0) for x in range(x0, x1 + 1)]
    elif side == "south":
        cells = [(x, z1) for x in range(x0, x1 + 1)]
    elif side == "west":
        cells = [(x0, z) for z in range(z0, z1 + 1)]
    else:
        cells = [(x1, z) for z in range(z0, z1 + 1)]
    mid = (len(cells) - 1) / 2.0
    pairs = sorted(enumerate(cells), key=lambda p: abs(p[0] - mid))
    return [c for _, c in pairs if c not in blocked]


def _furnish_hall(b, rect, y, blocked, rng, stack_side, flue):
    """The room you come into: the fire under the stack, a board to eat at, a
    bench, somewhere to put things."""
    x0, _, z0, x1, z1 = rect
    area = (x1 - x0 + 1) * (z1 - z0 + 1)
    edge = _room_cells(x0, z0, x1, z1, blocked, rng, True)
    mid = _room_cells(x0, z0, x1, z1, blocked, rng, False)
    kw = {"mat": b.voice["wall"]}
    if flue is not None:
        kw["flue_to"] = flue
    hearth = _put(b, "hearth", _side_cells(rect, stack_side, blocked) + edge, y,
                  rect, **kw)
    if hearth is None:
        _put(b, "oven", edge, y, rect, mat=b.voice["wall"])
    used = set()
    if hearth is not None:
        used.add(hearth)
    cells = [c for c in mid if c not in used] if area >= 12 else []
    tbl = _put(b, "table", cells, y, rect, mat=b.voice["floor"])
    if tbl is not None:
        used.add(tbl)
        near = [(tbl[0] + 1, tbl[1]), (tbl[0] - 1, tbl[1]),
                (tbl[0], tbl[1] + 1), (tbl[0], tbl[1] - 1)]
        near = [c for c in near if c not in used and c not in blocked
                and x0 < c[0] < x1 and z0 < c[1] < z1]
        bn = _put(b, "bench", near, y, rect, mat=b.voice["floor"])
        if bn is not None:
            used.add(bn)
    if area >= 14:
        st = _put(b, "store", [c for c in edge if c not in used], y, rect,
                  mat=b.voice["floor"], extent=2 if area >= 24 else 1)
        if st is not None:
            used.add(st)
    if area >= 30 and rng.random() < 0.7:
        w = _put(b, "workbench", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["floor"])
        if w is not None:
            used.add(w)
    room = "hall" if area >= 40 else "house"
    _put(b, "light", [c for c in edge if c not in used], y, rect, room=room,
         mat=b.voice["floor"])
    if area >= 20:
        _put(b, "rug", [c for c in mid if c not in used], y, rect, room=room,
             mat=b.voice["floor"])


def _furnish_chamber(b, rect, y, blocked, rng):
    """An upper floor: beds under the slope, a shelf, one light."""
    x0, _, z0, x1, z1 = rect
    area = (x1 - x0 + 1) * (z1 - z0 + 1)
    edge = _room_cells(x0, z0, x1, z1, blocked, rng, True)
    mid = _room_cells(x0, z0, x1, z1, blocked, rng, False)
    used = set()
    for _ in range(2 if area >= 30 else 1):
        bd = _put(b, "bed", [c for c in edge if c not in used], y, rect,
                  mat=b.voice["floor"])
        if bd is None:
            break
        used.add(bd)
    if rng.random() < 0.6:
        sh = _put(b, "bookshelf", [c for c in edge if c not in used], y, rect,
                  mat=b.voice["floor"])
    else:
        sh = _put(b, "shelf", [c for c in edge if c not in used], y, rect,
                  mat=b.voice["floor"])
    if sh is not None:
        used.add(sh)
    if area >= 24 and rng.random() < 0.5:
        ch = _put(b, "store", [c for c in edge if c not in used], y, rect,
                  mat=b.voice["floor"], extent=1)
        if ch is not None:
            used.add(ch)
    _put(b, "light", [c for c in edge if c not in used], y, rect, room="house",
         mat=b.voice["floor"])
    if area >= 18 and rng.random() < 0.5:
        _put(b, "rug", [c for c in mid if c not in used], y, rect, room="house",
             mat=b.voice["floor"])


def _furnish_outshot(b, rect, y, blocked, rng, use):
    """The lean-to, and what it is for."""
    x0, _, z0, x1, z1 = rect
    edge = _room_cells(x0, z0, x1, z1, blocked, rng, True)
    mid = _room_cells(x0, z0, x1, z1, blocked, rng, False)
    used = set()
    if use == "byre":
        for kind in ("trough", "fodder", "fodder"):
            c = _put(b, kind, [c for c in edge if c not in used], y, rect,
                     mat=b.voice["floor"])
            if c is not None:
                used.add(c)
    elif use == "scullery":
        c = _put(b, "oven", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["wall"])
        if c is not None:
            used.add(c)
        c = _put(b, "workbench", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["floor"])
        if c is not None:
            used.add(c)
        c = _put(b, "store", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["floor"], extent=1)
        if c is not None:
            used.add(c)
    else:
        c = _put(b, "store", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["floor"], extent=3)
        if c is None:
            c = _put(b, "store", [c for c in edge if c not in used], y, rect,
                     mat=b.voice["floor"], extent=1)
        if c is not None:
            used.add(c)
        c = _put(b, "shelf", [c for c in edge if c not in used], y, rect,
                 mat=b.voice["floor"])
        if c is not None:
            used.add(c)
    _put(b, "light", [c for c in edge + mid if c not in used], y, rect,
         room="store", mat=b.voice["floor"])


def build(b, part, seed, **params):
    rng = random.Random((int(seed) * 7919) ^ 0x5F3A9)
    storeys = params.get("storeys", 2)
    if not isinstance(storeys, int) or storeys < 1:
        storeys = 1
    storeys = min(3, storeys)
    use = params.get("outshot", "byre")
    if use not in ("byre", "store", "scullery"):
        use = "store"

    bits = (int(seed) * 2654435761 + 1013904223
            + (part["x0"] * 2246822519) ^ (part["z0"] * 3266489917)) % (2 ** 32)
    plan = _plan(part, rng, use, storeys, _public(b, part), bits)
    label = part["label"]
    main = plan["main"]
    fy = part["floor_y"]
    plot = plan["plot"]

    base = {"mat": dict(b.voice), "openings": "rhythm", "stair": "auto",
            "chimney": plan["chimney"], "flashing": True}
    if plan["dormers"]:
        base["dormers"] = plan["dormers"]
    if plan["depth"] >= 2:
        base["outshot"] = {"side": plan["out_side"], "depth": plan["depth"]}
    else:
        # too tight a pad for a lean-to: the door gets a porch instead
        base["porch"] = True

    lean = dict(base)
    lean.pop("dormers", None)
    bare = dict(lean)
    bare.pop("outshot", None)
    bare.pop("porch", None)
    plain = {"mat": dict(b.voice), "openings": "rhythm", "stair": "auto"}
    flat_roof = {"style": "gable", "axis": plan["roof"]["axis"], "pitch": (1, 1)}

    attempts = [
        (plan["storeys"], plan["roof"], base),
        (plan["storeys"], plan["roof"], lean),
        (plan["storeys"], plan["roof"], bare),
        (max(1, plan["storeys"] - 1), plan["roof"], dict(bare)),
        (1, flat_roof, plain),
    ]
    res = None
    won = None
    for (st, spec, kw) in attempts:
        r = b.building(label, main[0], main[1], main[2], main[3], st, spec, **kw)
        if isinstance(r, dict) and r.get("ok"):
            res = r
            won = kw
            break
    if res is None:
        return {"ok": False, "label": label}

    ridge = res.get("ridge_y")
    eave = res.get("eave_y")
    flue = None
    if "chimney" not in won:
        flue = ridge if isinstance(ridge, int) else None

    blocked = set()
    _collect(res.get("stairs"), blocked)
    _collect(res.get("extras"), blocked)
    door_cells = set()
    _collect(res.get("door"), door_cells)
    ft = b.floor_from_threshold(label)
    if isinstance(ft, dict):
        _collect(ft.get("door"), door_cells)
    door_cells.add((part["door"][0], part["door"][-1]))
    for (x, z) in sorted(door_cells):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                blocked.add((x + dx, z + dz))
    # nothing stands in the way in: three cells of clear floor off the door
    blocked |= _way_in(main, plan["door_side"], part["door"][0],
                       part["door"][-1])

    rooms = []
    for r in (res.get("rooms") or []):
        if not isinstance(r, (list, tuple)) or len(r) != 5:
            continue
        x0, y, z0, x1, z1 = r
        if x1 < x0:
            x0, x1 = x1, x0
        if z1 < z0:
            z0, z1 = z1, z0
        if x1 - x0 < 0 or z1 - z0 < 0:
            continue
        rooms.append((x0, y, z0, x1, z1))
    if rooms:
        ground = min(r[1] for r in rooms)
        lower = [r for r in rooms if r[1] <= ground]
        lower.sort(key=lambda r: (r[3] - r[0] + 1) * (r[4] - r[2] + 1),
                   reverse=True)
        upper = [r for r in rooms if r[1] > ground]
        for i, r in enumerate(lower):
            y = _stand_y(b, r[0], r[1], r[2], r[3], r[4])
            # Keep one side of each corner open: two fittings on adjacent walls
            # otherwise trap the bare floor between them.
            circulation = {(r[0] + 1, r[2]), (r[0] + 1, r[4]),
                           (r[3] - 1, r[2]), (r[3] - 1, r[4])}
            if i == 0:
                _furnish_hall(b, r, y, blocked | circulation, rng, plan["chimney"]
                              if isinstance(plan["chimney"], str) else "north",
                              flue)
            else:
                _furnish_outshot(b, r, y, blocked | circulation | _inner_row(r, main), rng,
                                 use)
        for r in upper:
            y = _stand_y(b, r[0], r[1], r[2], r[3], r[4])
            _furnish_chamber(b, r, y, blocked, rng)

    if isinstance(eave, int) and _pick(bits, 27, 2):
        top = min(eave, fy + 1 + 4 * max(1, plan["storeys"]))
        post = b.block(b.voice["trim"], "post")
        upright = post + "[axis=y]" if post.endswith(("_log", "_pillar")) else post
        for (cx, cz) in ((main[0], main[1]), (main[2], main[1]),
                         (main[0], main[3]), (main[2], main[3])):
            for y in range(fy + 1, top):
                b.place_block(cx, y, cz, upright)

    dr = res.get("door")
    if isinstance(dr, (list, tuple)) and len(dr) == 3:
        b.check_door(dr[0], dr[1], dr[2])
    elif isinstance(dr, (list, tuple)) and len(dr) == 2:
        b.check_door(dr[0], fy + 1, dr[1])

    b.seal_voids(plot[0], plot[1], plot[2], plot[3], b.block(b.voice["wall"]),
                 max_cells=400)
    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "label": label, "ridge_y": ridge}
