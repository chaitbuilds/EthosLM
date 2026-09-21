"""A town house: tall, narrow, jettied over the street.

A **form**, not a voice. Every material is `b.voice[role]`.
"""

import math
import random

FORM = "european_vernacular"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "urban"

#: **What this type is for.** The realization round: a `ROLE` says what work a building
#: is for and is satisfied by a hall, a barn or a temple alike; a sentence asking for
#: houses people live in is asking for a `dwelling`. Declared so that the function can
#: be checked rather than inferred from a label.
FUNCTION = "dwelling"


PARAMS = {
    "storeys": ("int", 2, 3),
    "jetty_side": ("choice", ["street", "left", "right", "both"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 11x11 and up measured. Below eleven the pad is too narrow for the flight and the
    # upper storey is a room nobody can walk into. Measured by scripts/type_needs.py;
    # the band is rounds/type-needs.json. The pair is the pad site() hands build(),
    # after siting's inset.
    "footprint": (8, 8, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

def _post(b, axis="y"):
    """An upright of the frame, turned along `axis`. The block comes from the voice, and
    the registry says whether that block takes an axis (`b.axial`).
    """
    p = b.block(b.voice["frame"], "post")
    return b.axial(p, axis)


def _band(b, axis):
    """The horizontal member at a floor line, laid along `axis`."""
    t = b.block(b.voice["trim"], "bare")
    return b.axial(t, axis)

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}
RIGHT = {"north": "east", "east": "south", "south": "west", "west": "north"}
STEP = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
SOLID_NO = ("air", "water", "lava", "cave_air", "void_air", "grass", "fern",
            "snow", "torch", "lantern", "candle", "vine", "seagrass", "kelp")


def _front_side(part):
    """Which edge of the pad the way in arrives on."""
    dx, dz = part["door"][0], part["door"][1]
    if dx <= part["x0"]:
        return "west"
    if dx >= part["x1"]:
        return "east"
    if dz <= part["z0"]:
        return "north"
    if dz >= part["z1"]:
        return "south"
    return OPP.get(part.get("facing"), "south")


def _is_render(b, block):
    """True for the wall's own panel material, and only for a full block of it."""
    if not block:
        return False
    head = block.split("[")[0]
    for bad in ("stairs", "slab", "wall", "pane", "fence", "door", "trapdoor"):
        if bad in head:
            return False
    return head in (b.block(b.voice["wall"]), b.block(b.voice["wall"], "accent"),
                    b.block(b.voice["wall"], "fine"))


def _solid(block):
    if not block:
        return False
    head = block.split("[")[0]
    return head not in SOLID_NO and "air" not in head


def _dress(b, x, y, z, block):
    """Turn a panel cell into frame, and leave anything else alone."""
    if _is_render(b, b.get_block(x, y, z)):
        b.place_block(x, y, z, block)
        return True
    return False


def _perimeter(x0, z0, x1, z1):
    cells = []
    for x in range(x0, x1 + 1):
        cells.append((x, z0, "x"))
        if z1 != z0:
            cells.append((x, z1, "x"))
    for z in range(z0 + 1, z1):
        cells.append((x0, z, "z"))
        if x1 != x0:
            cells.append((x1, z, "z"))
    return cells


def _post_lines(x0, z0, x1, z1):
    """Corner posts, and enough intermediate ones that no run exceeds five."""
    lines = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
    span_x, span_z = x1 - x0, z1 - z0
    n_x = max(1, int(math.ceil(span_x / 5.0)))
    n_z = max(1, int(math.ceil(span_z / 5.0)))
    for i in range(1, n_x):
        x = x0 + int(round(span_x * i / float(n_x)))
        lines.append((x, z0))
        lines.append((x, z1))
    for i in range(1, n_z):
        z = z0 + int(round(span_z * i / float(n_z)))
        lines.append((x0, z))
        lines.append((x1, z))
    out = []
    for c in lines:
        if c not in out:
            out.append(c)
    return out


def _rect(front, run_lo, run_hi, dep_lo, dep_hi):
    """Turn a (frontage run, depth) pair back into world corners."""
    if front in ("east", "west"):
        return (dep_lo, run_lo, dep_hi, run_hi)
    return (run_lo, dep_lo, run_hi, dep_hi)


def _plans(part, rng, front, seed=0):
    """Where on this pad the house stands: the full frontage, and how deep.

        A town house is built up to its neighbours on both sides -- the pad is its
        plot -- so the frontage is the whole of it and what varies is the depth,
        which leaves a forecourt of one to three blocks on the street. That strip
        is the doorstep the lane already comes to, so it is never shut in.
        
    """
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    if front in ("east", "west"):
        run_lo, run_hi, dep_lo, dep_hi = pz0, pz1, px0, px1
    else:
        run_lo, run_hi, dep_lo, dep_hi = px0, px1, pz0, pz1
    dep_avail = dep_hi - dep_lo + 1
    near = front in ("west", "north")

    fms = [f for f in (1, 2, 3) if dep_avail - f >= 6]
    if fms:
        k = seed % len(fms)
        fms = fms[k:] + fms[:k]
    fms.append(0)

    out = []
    for fm in fms:
        usable = dep_avail - fm
        if usable < 5:
            continue
        if near:
            d0, d1 = dep_lo + fm, dep_lo + fm + usable - 1
        else:
            d1, d0 = dep_hi - fm, dep_hi - fm - usable + 1
        rect = _rect(front, run_lo, run_hi, d0, d1)
        if (rect, fm) not in out:
            out.append((rect, fm))
    return out


def _roofspec(rng, main, front):
    """A steep gable, ridged across the street, and never twice the same."""
    ax = "z" if front in ("east", "west") else "x"
    if (main[2] - main[0]) < (main[3] - main[1]):
        ax = "x" if ax == "z" else "z"
    if rng.random() < 0.25:
        ax = "x" if ax == "z" else "z"
    spec = {"style": "gable", "axis": ax, "overhang": 2,
            "pitch": (2, 1) if rng.random() < 0.75 else (3, 1)}
    r = rng.random()
    if r < 0.2:
        spec["ends"] = ("gable", "half-hip")
    elif r < 0.3:
        spec["ends"] = ("half-hip", "gable")
    if rng.random() < 0.35:
        spec["eave"] = "flared"
    return spec


def _stack_at(main, ax, door):
    """The stack goes up the far gable, on the ridge line and in the wall.

        Through a slope it would have the roof's own treads running down away from
        it on one side, which is the one thing an eave cannot answer for; and
        standing in the room it takes the floor the flight needs.
        
    """
    x0, z0, x1, z1 = main
    dx, dz = door[0], door[1]
    if ax == "z":
        z = (z0 + z1) // 2
        return (x1 if abs(dx - x1) > abs(dx - x0) else x0, z)
    x = (x0 + x1) // 2
    return (x, z1 if abs(dz - z1) > abs(dz - z0) else z0)


def _house(b, part, rng, storeys, jetty_side, seed=0):
    """Try the massing the seed wants, then plainer ones, until one stands."""
    label = part["label"]
    front = _front_side(part)
    side_of = {"street": front, "left": LEFT[front], "right": RIGHT[front],
               "both": front}
    oriel_face = side_of.get(jetty_side, front)
    plans = _plans(part, rng, front, seed)

    for n in (storeys, max(2, storeys - 1), 2):
        for (main, fm) in plans:
            x0, z0, x1, z1 = main
            m = _margins(part, x0, z0, x1, z1)
            spec = _roofspec(rng, main, front)
            stack = _stack_at(main, spec["axis"], part["door"])
            base = {"mat": dict(b.voice), "openings": "rhythm", "stair": "auto",
                    "roof": spec, "chimney": stack, "flashing": True}
            extras = []
            if n >= 2:
                jet = dict(base)
                jet["jetty"] = n - 1
                if m.get(oriel_face, 0) >= 1:
                    jet["oriel"] = (oriel_face, 1)
                extras.append(jet)
                if m.get(oriel_face, 0) >= 1:
                    ori = dict(base)
                    ori["oriel"] = (oriel_face, 1)
                    extras.append(ori)
            extras.append(dict(base))
            for kw in extras:
                res = b.building(label, x0, z0, x1, z1, n, **kw)
                if res and res.get("ok"):
                    res["_rect"] = main
                    res["_front"] = front
                    res["_storeys"] = n
                    res["_oriel"] = oriel_face
                    res["_stack"] = stack
                    res["_asked"] = kw
                    return res
    return None


def _margins(part, x0, z0, x1, z1):
    return {
        "west": x0 - part["x0"],
        "east": part["x1"] - x1,
        "north": z0 - part["z0"],
        "south": part["z1"] - z1,
    }


def _levels(floors, fy, n):
    """The floor levels, as plain numbers, whatever shape they came back in."""
    out = []
    for f in _seq(floors):
        if isinstance(f, (int, float)):
            out.append(int(f))
        elif isinstance(f, dict) and isinstance(f.get("y"), (int, float)):
            out.append(int(f["y"]))
    out = sorted(set(out))
    return out if out else [fy + 4 * i for i in range(n)]


def _frame(b, res, fy, top_y, floors):
    """The dark frame over the pale panels: posts, and a band at every floor."""
    r = res["_rect"]
    rings = [r, (r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1)]
    for (a, c, d, e) in rings:
        for (px, pz) in _post_lines(a, c, d, e):
            for y in range(fy + 1, top_y + 1):
                _dress(b, px, y, pz, _post(b))
        bands = list(floors[1:])
        bands.append(top_y)
        for by in bands:
            for (cx, cz, ax) in _perimeter(a, c, d, e):
                _dress(b, cx, by, cz, _band(b, ax))


def _facing_in(x, z, x0, z0, x1, z1):
    """Which way you stand when you use something against this wall."""
    d = [(x - x0, "east"), (x1 - x, "west"), (z - z0, "south"), (z1 - z, "north")]
    d.sort()
    return d[0][1]


def _room_cells(rect, rng):
    x0, y, z0, x1, z1 = rect
    edge, mid = [], []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            f = _facing_in(x, z, x0, z0, x1, z1)
            if x in (x0, x1) or z in (z0, z1):
                edge.append((x, z, f))
            else:
                mid.append((x, z, f))
    rng.shuffle(edge)
    rng.shuffle(mid)
    return edge + mid


def _obstacles(b, res, floors, fy):
    """The treads, the landings and the doorways: circulation, not floor.

        A chair with its back to a stair is the one piece of furniture that costs
        you a whole storey, so nothing is set down in a cell a flight passes
        through or in the cell you step off it into.
        
    """
    taken = set()
    top = max(floors) + 2 if floors else fy + 4
    for rect in (res["_rect"],):
        x0, z0, x1, z1 = rect
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                for y in range(fy, top + 1):
                    head = b.get_block(x, y, z).split("[")[0]
                    if ("stairs" in head or "door" in head or "ladder" in head
                            or "trapdoor" in head):
                        for (ox, oz) in ((0, 0), (1, 0), (-1, 0), (0, 1),
                                         (0, -1)):
                            taken.add((x + ox, z + oz))
                        break
    return taken


def _snapshot(b, x, y, z, fy, r=3):
    out = []
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            for dy in range(-1, 4):
                if y + dy < fy:
                    continue
                out.append((x + dx, y + dy, z + dz,
                            b.get_block(x + dx, y + dy, z + dz)))
    return out


def _unreached(rep):
    if not isinstance(rep, dict):
        return 0
    bad = 0
    for r in _seq(rep.get("rooms")):
        if isinstance(r, dict):
            bad += max(0, int(r.get("cells") or 0) - int(r.get("walkable") or 0))
    return bad


def _fit(b, kind, rect, cells, taken, guard, **kw):
    """Put one piece of equipment in the first cell that will take it.

        Then walk the building again: a fitting that has shut a room off is lifted
        straight back out, because a chest is not worth a storey.
        
    """
    y = rect[1]
    fy = guard["fy"]
    for lift in (1, 0):
        for (x, z, facing) in cells:
            if (x, z) in taken:
                continue
            if not _solid(b.get_block(x, y + lift - 1, z)):
                continue
            if _solid(b.get_block(x, y + lift, z)):
                continue
            snap = _snapshot(b, x, y + lift, z, fy)
            r = b.fitting(kind, x, y + lift, z, facing, **kw)
            if not (r and r.get("ok")):
                continue
            bad = _unreached(b.check_walkable(guard["label"]))
            if bad > guard["bad"]:
                for (sx, sy, sz, blk) in snap:
                    if b.get_block(sx, sy, sz) == blk:
                        continue
                    if "stairs" in blk.split("[")[0]:
                        continue
                    b.place_block(sx, sy, sz, blk)
                taken.add((x, z))
                continue
            guard["bad"] = bad
            for c in _cell_list(r.get("cells")):
                taken.add((c[0], c[2]))
            taken.add((x, z))
            return r
    return None


def _furnish(b, res, rng, taken, guard):
    """What each floor of a town house is for, and so what it holds."""
    rooms = [r for r in _seq(res.get("rooms")) if len(r) >= 5]
    rooms.sort(key=lambda r: (r[1], r[0], r[2]))
    if not rooms:
        return
    levels = sorted(set(r[1] for r in rooms))
    stack = res.get("_stack")
    hearth = False
    for rect in rooms:
        idx = levels.index(rect[1])
        cells = _room_cells(rect, rng)
        area = (rect[3] - rect[0] + 1) * (rect[4] - rect[2] + 1)
        if idx == 0:
            if not hearth:
                near = cells
                if stack:
                    near = sorted(cells, key=lambda c: (abs(c[0] - stack[0]) +
                                                        abs(c[1] - stack[1])))
                hearth = bool(_fit(b, "hearth", rect, near[:8], taken, guard,
                                   mat=b.voice["footing"]))
            _fit(b, "table", rect, cells, taken, guard, mat=b.voice["frame"])
            _fit(b, "bench", rect, cells, taken, guard, mat=b.voice["frame"])
            if area >= 12:
                _fit(b, "store", rect, cells, taken, guard, extent=2)
            _fit(b, "light", rect, cells, taken, guard, room="hall")
        elif idx == 1:
            _fit(b, "bed", rect, cells, taken, guard, mat=b.voice["frame"])
            if area >= 14:
                _fit(b, "bed", rect, cells, taken, guard, mat=b.voice["frame"])
            _fit(b, "shelf", rect, cells, taken, guard, mat=b.voice["frame"])
            if area >= 15 and rng.random() < 0.6:
                _fit(b, "bookshelf", rect, cells, taken, guard, mat=b.voice["frame"])
            _fit(b, "light", rect, cells, taken, guard, room="house")
        else:
            _fit(b, "store", rect, cells, taken, guard, extent=2)
            if rng.random() < 0.5:
                _fit(b, "workbench", rect, cells, taken, guard, mat=b.voice["frame"])
            else:
                _fit(b, "bed", rect, cells, taken, guard, mat=b.voice["frame"])
            _fit(b, "light", rect, cells, taken, guard, room="store")


def _seq(v):
    """Only ever loop over something that really is a sequence."""
    return list(v) if isinstance(v, (list, tuple)) else []


def _cell_list(v):
    """The library answers in dicts; take only what is really a list of cells."""
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for c in v:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            out.append(c)
        elif isinstance(c, dict) and "y" in c:
            out.append((c.get("x"), c.get("y"), c.get("z")))
    return out


def _bbox5(room):
    bb = room.get("bbox") if isinstance(room, dict) else None
    if not bb or len(bb) < 4:
        return None
    if len(bb) >= 6:
        return (bb[0], bb[1], bb[2], bb[3], bb[5])
    if len(bb) == 5:
        return tuple(bb)
    return None


def _interiors(res):
    r = res["_rect"]
    if r[2] - r[0] < 2 or r[3] - r[1] < 2:
        return []
    return [(r[0] + 1, r[1] + 1, r[2] - 1, r[3] - 1)]


def _lay_flight(b, label, res, y0, y1, rng, rects=None):
    """A straight flight, wherever one will stand inside these walls."""
    rise = y1 - y0
    tries = []
    for (ix0, iz0, ix1, iz1) in (rects if rects is not None else _interiors(res)):
        for facing in ("north", "south", "east", "west"):
            sx, sz = STEP[facing]
            for x in range(ix0, ix1 + 1):
                for z in range(iz0, iz1 + 1):
                    ex, ez = x + sx * rise, z + sz * rise
                    if not (ix0 <= ex <= ix1 and iz0 <= ez <= iz1):
                        continue
                    edge = min(x - ix0, ix1 - x, z - iz0, iz1 - z)
                    tries.append((edge, rng.random(), x, z, facing))
    tries.sort()
    for (_, _, x, z, facing) in tries[:14]:
        r = b.flight(label, x, z, y0, y1, facing)
        if r and r.get("ok"):
            return r
    return None


def _has_flight(b, rect, y0, y1):
    """Is there a run of treads between these two floors, in this range?"""
    ix0, iz0, ix1, iz1 = rect
    for x in range(ix0, ix1 + 1):
        for z in range(iz0, iz1 + 1):
            for y in range(y0 + 1, y1 + 1):
                if "stairs" in b.get_block(x, y, z).split("[")[0]:
                    return True
    return False


def _ensure_flights(b, res, floors, rng, label):
    """A storey with no flight up to it is a storey nobody lives in.

        The library lays one between every pair of storeys, but a stack or a wall
        can leave a landing with nothing arriving at it, so the treads are read
        back out of the world rather than taken on trust.
        
    """
    for rect in _interiors(res):
        for i in range(1, len(floors)):
            if not _has_flight(b, rect, floors[i - 1], floors[i]):
                _lay_flight(b, label, res, floors[i - 1], floors[i], rng,
                            rects=[rect])


def _rescue(b, res, part, rng, floors):
    """Any storey you cannot walk up to gets a flight of its own."""
    fy = part["floor_y"]
    levels = list(floors)
    for _ in range(2):
        rep = b.check_walkable(part["label"])
        if not isinstance(rep, dict):
            return
        stuck = []
        for room in _seq(rep.get("rooms")):
            if not isinstance(room, dict):
                continue
            frac = room.get("fraction")
            walk = room.get("walkable")
            if not ((frac is not None and frac <= 0.02) or walk == 0):
                continue
            bb = _bbox5(room)
            if bb:
                stuck.append(int(bb[1]))
        if not stuck:
            return
        for y in stuck:
            below = [f for f in levels if f <= y]
            if not below:
                continue
            target = below[-1]
            i = levels.index(target)
            if i < 1 or target <= fy:
                continue
            _lay_flight(b, part["label"], res, levels[i - 1], target, rng)


def _cleanup(b, res, fy):
    """Nothing of mine is left hanging in the air.

        check_attached answers in bounding boxes, and the world around the plot has
        overhangs of its own in it, so only cells over my own footprint are lifted.
        
    """
    r = res["_rect"]
    mine = [(r[0] - 3, r[1] - 3, r[2] + 3, r[3] + 3)]
    for _ in range(3):
        rep = b.check_attached()
        if not isinstance(rep, dict) or rep.get("ok"):
            return
        cells = []
        for item in _seq(rep.get("floating")):
            if isinstance(item, dict):
                bb = item.get("bbox")
                got = _cell_list(item.get("cells"))
                if got:
                    cells.extend(got)
                elif isinstance(bb, (list, tuple)) and len(bb) >= 6:
                    if (bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2]) > 300:
                        continue
                    for x in range(int(bb[0]), int(bb[3]) + 1):
                        for y in range(int(bb[1]), int(bb[4]) + 1):
                            for z in range(int(bb[2]), int(bb[5]) + 1):
                                cells.append((x, y, z))
                elif "y" in item:
                    cells.append((item.get("x"), item.get("y"), item.get("z")))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                cells.append(item[:3])
        lifted = 0
        for c in cells:
            if not c or c[1] is None:
                continue
            x, y, z = int(c[0]), int(c[1]), int(c[2])
            if y <= fy:
                continue
            if not any(a0 <= x <= a1 and c0 <= z <= c1
                       for (a0, c0, a1, c1) in mine):
                continue
            if b.get_block(x, y, z).split("[")[0] in ("air", "cave_air"):
                continue
            b.place_block(x, y, z, "air")
            lifted += 1
        if not lifted:
            return


def _xz(spot, fallback):
    if spot is None:
        return fallback
    if isinstance(spot, dict):
        if "x" in spot and "z" in spot:
            return (spot["x"], spot["z"])
        return fallback
    if isinstance(spot, (list, tuple)):
        if len(spot) == 2:
            return (spot[0], spot[1])
        if len(spot) >= 3:
            return (spot[0], spot[2])
    return fallback


def build(b, part, seed, **params):
    rng = random.Random(seed)
    storeys = max(2, min(3, int(params.get("storeys") or 2)))
    jetty_side = params.get("jetty_side") or "street"
    fy = part["floor_y"]

    res = _house(b, part, rng, storeys, jetty_side, int(seed))
    if not res:
        return {"ok": False, "reason": "no massing stood on this pad",
                "emitted": {"requested": {"storeys": storeys, "jetty_side": jetty_side},
                            "storeys": 0, "attempt": None, "fallback": "no shell stood",
                            "omitted": ["storeys"], "features": {}}}

    n = res.get("_storeys") or storeys
    # **What survived, said by the type.** The closure round: `_house` walks the storeys
    # down from what was asked and the jetty and the oriel off the massing; the record
    # says what stood beside what `construction.outcome` measures.
    _kw = res.get("_asked") or {}
    res["emitted"] = {
        "requested": {"storeys": storeys, "jetty_side": jetty_side},
        "storeys": int(n), "attempt": None,
        "fallback": ("; ".join(
            ([f"storeys {storeys} -> {n}"] if n < storeys else [])
            + (["no jetty"] if n >= 2 and "jetty" not in _kw else [])) or None),
        "omitted": ([] if n >= storeys else ["storeys"])
                   + (["jetty"] if n >= 2 and "jetty" not in _kw else []),
        "features": {"jetty": "jetty" in _kw, "oriel": "oriel" in _kw,
                     "chimney": bool(_kw.get("chimney"))},
        "rects": {"main": list(res["_rect"])},
        "floors": list(res.get("floors") or []),
    }
    floors = _levels(res.get("floors"), fy, n)
    eave = int(res.get("eave_y") or (fy + 4 * n))
    _frame(b, res, fy, eave - 1, floors)
    _ensure_flights(b, res, floors, rng, part["label"])
    _rescue(b, res, part, rng, floors)

    taken = _obstacles(b, res, floors, fy)
    dx, dz = _xz(res.get("door"), part["door"])
    for ox in (-1, 0, 1):
        for oz in (-1, 0, 1):
            taken.add((dx + ox, dz + oz))
    guard = {"label": part["label"], "fy": fy,
             "bad": _unreached(b.check_walkable(part["label"]))}
    _furnish(b, res, rng, taken, guard)
    _cleanup(b, res, fy)

    x0, z0, x1, z1 = res["_rect"]
    b.seal_voids(x0, z0, x1, z1, b.block(b.voice["wall"]), max_cells=16)
    door_rep = b.check_door(dx, fy + 1, dz)
    walk = b.check_walkable(part["label"])
    att = b.check_attached()
    return res
