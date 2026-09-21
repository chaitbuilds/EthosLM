"""A farmstead: a low farmhouse with a boarded veranda under a deep eave,
standing inside a walled yard that holds a store, a byre or a stack, with a
gate in the wall on the side the lane arrives from."""

import math
import random

KIND = "plot"
FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "rural"

#: **What this type is for.** The realization round: a `ROLE` says what work a building
#: is for and is satisfied by a hall, a barn or a temple alike; a sentence asking for
#: houses people live in is asking for a `dwelling`. Declared so that the function can
#: be checked rather than inferred from a label.
FUNCTION = "dwelling"


PARAMS = {
    "storeys": ("int", 1, 2),
    "yard_use": ("choice", ["store", "byre", "stack"]),
}

NEEDS = {
    # Cut to the band `scripts/type_needs.py` measured -- measured 5x5 to 9x9 by the
    # sweep; declared to 24 by its author.
    "footprint": (5, 5, 32, 32),
    "except": (11, 12, 13, 14, 15, 17, 21, 22),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}


def _clamp(v, lo, hi):
    if hi < lo:
        return lo
    return max(lo, min(hi, v))


def _as_xz(d):
    if d is None:
        return None
    if isinstance(d, dict):
        if "x" in d and "z" in d:
            return (d["x"], d["z"])
        return None
    if isinstance(d, (list, tuple)):
        if len(d) == 2:
            return (d[0], d[1])
        if len(d) >= 3:
            return (d[0], d[2])
    return None


def _ridge_axis(x0, z0, x1, z1):
    """Ridge along the long side: axis names the way the slopes face."""
    return "z" if (x1 - x0) >= (z1 - z0) else "x"


def _facing_of(part):
    f = part.get("facing")
    if f in ("north", "south", "east", "west"):
        return f
    d = _as_xz(part.get("door"))
    if d is not None:
        if d[0] == part["x0"]:
            return "east"
        if d[0] == part["x1"]:
            return "west"
        if d[1] == part["z0"]:
            return "south"
    return "north"


def _make_frame(part):
    """Local (u, v): v runs inward from the edge the lane arrives at, u across it."""
    f = _facing_of(part)
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    w = x1 - x0 + 1
    d = z1 - z0 + 1
    if f == "east":
        return {"A": d, "P": w, "f": f,
                "to": (lambda u, v: (x0 + v, z0 + u)),
                "frm": (lambda x, z: (z - z0, x - x0))}
    if f == "west":
        return {"A": d, "P": w, "f": f,
                "to": (lambda u, v: (x1 - v, z0 + u)),
                "frm": (lambda x, z: (z - z0, x1 - x))}
    if f == "south":
        return {"A": w, "P": d, "f": f,
                "to": (lambda u, v: (x0 + u, z0 + v)),
                "frm": (lambda x, z: (x - x0, z - z0))}
    return {"A": w, "P": d, "f": f,
            "to": (lambda u, v: (x0 + u, z1 - v)),
            "frm": (lambda x, z: (x - x0, z1 - z))}


def _rect(fr, u0, v0, u1, v1):
    ax, az = fr["to"](u0, v0)
    bx, bz = fr["to"](u1, v1)
    return (min(ax, bx), min(az, bz), max(ax, bx), max(az, bz))


def _largest_rect(free, a, p, maxw, maxd):
    """Biggest clear rectangle in the yard, in local cells."""
    best = None
    heights = [0] * a
    for v in range(p):
        for u in range(a):
            heights[u] = heights[u] + 1 if (u, v) in free else 0
        for u in range(a):
            if heights[u] <= 0:
                continue
            mh = heights[u]
            for k in range(u, a):
                if heights[k] <= 0:
                    break
                if heights[k] < mh:
                    mh = heights[k]
                hh = min(mh, maxd)
                ww = min(k - u + 1, maxw)
                score = min(hh, 6) * min(ww, 6) * 10 + hh + ww
                if best is None or score > best[0]:
                    best = (score, u, v - hh + 1, u + ww - 1, v)
    return best


def _runs(cells, skip):
    out = []
    cur = []
    for c in cells:
        if c in skip:
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(c)
    if cur:
        out.append(cur)
    return out


def _ring_facings(x, z, x0, z0, x1, z1):
    """Which way something standing at (x, z) should face to look into the room."""
    out = []
    if z == z0:
        out.append("south")
    if z == z1:
        out.append("north")
    if x == x0:
        out.append("east")
    if x == x1:
        out.append("west")
    if not out:
        out = ["north", "south", "east", "west"]
    return out


def _facing_between(fx, fz, tx, tz):
    if tx > fx:
        return "east"
    if tx < fx:
        return "west"
    if tz > fz:
        return "south"
    return "north"


def _fit(b, kind, spots, y, **kw):
    """First spot that will take this piece of equipment."""
    for (x, z, f) in spots:
        r = b.fitting(kind, x, y, z, f, **kw)
        if r and r.get("ok"):
            return r
    return None


def _room_spots(rng, x0, z0, x1, z1, edge=True, avoid=(), ways=()):
    spots = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if (x, z) in avoid:
                continue
            on = (x == x0 or x == x1 or z == z0 or z == z1)
            if on != edge:
                continue
            for f in _ring_facings(x, z, x0, z0, x1, z1):
                spots.append((x, z, f))
    rng.shuffle(spots)
    if ways:
        spots.sort(key=lambda s: -min(abs(s[0] - w[0]) + abs(s[1] - w[1])
                                      for w in ways))
    return spots


def _aisle(rx0, rz0, rx1, rz1, entries):
    """A cross of floor through every way into the room, kept clear of
    furniture: the row and the column of each entry cell. Any two crosses
    meet, so everything the room opens onto stays walkable."""
    keep = set()
    for (ex, ez) in entries:
        for x in range(rx0, rx1 + 1):
            keep.add((x, ez))
        for z in range(rz0, rz1 + 1):
            keep.add((ex, z))
    return keep


def _ways_in(b, sky, rx0, rz0, rx1, rz1, sy):
    """Every cell of this room that stands in one of its own openings -- the
    front door, and the way through to the next room."""
    ways = set()
    for x in range(rx0, rx1 + 1):
        if b.get_block(x, sy, rz0 - 1) == sky:
            ways.add((x, rz0))
        if b.get_block(x, sy, rz1 + 1) == sky:
            ways.add((x, rz1))
    for z in range(rz0, rz1 + 1):
        if b.get_block(rx0 - 1, sy, z) == sky:
            ways.add((rx0, z))
        if b.get_block(rx1 + 1, sy, z) == sky:
            ways.add((rx1, z))
    return ways


def _collect_xz(obj, out, depth=0):
    """Every (x, z) hiding anywhere in a result the library handed back."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        p = _as_xz(obj)
        if p is not None:
            out.add(p)
        for v in obj.values():
            _collect_xz(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if len(obj) in (2, 3) and all(isinstance(v, int) for v in obj):
            p = _as_xz(obj)
            if p is not None:
                out.add(p)
        else:
            for v in obj:
                _collect_xz(v, out, depth + 1)


def _keep_clear(res, door_xz):
    """The doorway and every stair cell, plus one block round each: a fitting
    stood in a doorway is a room nobody can walk into."""
    seed_cells = set()
    if door_xz is not None:
        seed_cells.add((door_xz[0], door_xz[1]))
    _collect_xz(res.get("stairs"), seed_cells, 0)
    _collect_xz(res.get("door"), seed_cells, 0)
    grown = set()
    for (x, z) in seed_cells:
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                grown.add((x + dx, z + dz))
    return grown


def _massing(rng, A, P, u_door):
    """Farmhouse at the back, yard between it and the lane. Low and long."""
    min_yard = 2 if P <= 8 else 3
    hd_hi = min(9, P - min_yard)
    hd_lo = max(4, min(5, hd_hi))
    hd = rng.randint(hd_lo, hd_hi) if hd_hi > hd_lo else hd_hi
    # The side yard is never one block wide: a wall that close to the house stands under
    # its eave, and the top of it is a ledge nobody can reach.
    if A <= 6:
        side = 0
    elif A <= 8:
        side = rng.choice([0, 0, 2])
    elif A <= 10:
        side = rng.choice([0, 2, 3])
    else:
        side = rng.choice([0, 3, 4])
    ha = min(A - side, 13)
    if ha < 4:
        ha = min(A, 4)
    hv1 = P - 1
    hv0 = hv1 - hd + 1
    hu0 = 0 if u_door * 2 <= A - 1 else A - ha
    if not (hu0 <= u_door <= hu0 + ha - 1):
        hu0 = A - ha if hu0 == 0 else 0
    hu0 = _clamp(hu0, 0, A - ha)
    return hu0, hv0, hu0 + ha - 1, hv1


def _roof_spec(rng, part, hx0, hz0, hx1, hz1, span):
    spec = {
        "style": rng.choice(["hip", "hip", "gable"]),
        "axis": _ridge_axis(hx0, hz0, hx1, hz1),
        "pitch": (1, 2),
        "profile": [(1, 2), (2, 1)],
        "ends": rng.choice([("irimoya", "irimoya"), ("hip", "hip"),
                            ("irimoya", "hip"), ("half-hip", "half-hip")]),
        "eave": "upturned",
        "tiers": 1,
    }
    voiced = part.get("roof")
    if isinstance(voiced, dict):
        for k in ("profile", "ends", "eave", "tiers"):
            if voiced.get(k) is not None:
                spec[k] = voiced[k]
    if span >= 7 and rng.random() < 0.45:
        spec["tiers"] = 2
    return spec


def build(b, part, seed, **params):
    rng = random.Random((int(seed) * 7919) ^ 24007)
    voice = part["voice"]
    fy = part["floor_y"]
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    label = part["label"]

    storeys = int(params.get("storeys") or 1)
    asked_storeys = max(1, min(2, storeys))
    yard_use = params.get("yard_use") or "store"

    m_wall = b.block(voice["wall"], "full")
    m_frame = b.block(voice["frame"], "full")
    m_foot = b.block(voice["footing"], "full")
    m_trim_s = b.block(voice["trim"], "slab")
    m_deck = b.block(voice["trim"], "full")
    m_fence = b.joinery(voice, "fence")

    fr = _make_frame(part)
    A, P = fr["A"], fr["P"]
    dxz = _as_xz(part.get("door")) or fr["to"](A // 2, 0)
    u_door = _clamp(fr["frm"](dxz[0], dxz[1])[0], 0, A - 1)

    hu0, hv0, hu1, hv1 = _massing(rng, A, P, u_door)
    ha = hu1 - hu0 + 1
    hd = hv1 - hv0 + 1
    hx0, hz0, hx1, hz1 = _rect(fr, hu0, hv0, hu1, hv1)

    if max(ha, hd) < 7 or min(ha, hd) < 5:
        storeys = 1
    storeys = _clamp(storeys, 1, 2)
    gave_up = ([f"massing: storeys {asked_storeys} -> 1 for a {ha}x{hd} house"]
               if storeys < asked_storeys else [])

    spec = _roof_spec(rng, part, hx0, hz0, hx1, hz1, min(ha, hd))

    extras = {"brackets": True}

    res = b.building(label, hx0, hz0, hx1, hz1, storeys, spec,
                     mat=voice, openings="rhythm", stair="auto", **extras)
    if not res.get("ok") and extras:
        res = b.building(label, hx0, hz0, hx1, hz1, storeys, spec,
                         mat=voice, openings="rhythm", stair="auto")
    if not res.get("ok") and storeys > 1:
        storeys = 1
        gave_up.append("ladder: one storey")
        res = b.building(label, hx0, hz0, hx1, hz1, 1, spec,
                         mat=voice, openings="rhythm", stair="auto")
    if not res.get("ok"):
        spec2 = {"style": "hip", "axis": spec["axis"], "pitch": (1, 2)}
        gave_up.append("ladder: a plain hip roof")
        res = b.building(label, hx0, hz0, hx1, hz1, 1, spec2,
                         mat=voice, openings="rhythm", stair="none")
    if not res.get("ok"):
        return dict(res, emitted={"requested": {"storeys": asked_storeys},
                                  "storeys": 0, "attempt": None,
                                  "fallback": "no shell stood", "omitted": ["storeys"],
                                  "features": {}})
    # **What survived, said by the type.** The closure round.
    emitted = {"requested": {"storeys": asked_storeys, "yard_use": yard_use},
               "storeys": int(storeys), "attempt": len(gave_up),
               "fallback": "; ".join(gave_up) or None,
               "omitted": [] if storeys >= asked_storeys else ["storeys"],
               "features": {"brackets": bool(extras) and "brackets" in extras,
                            "engawa": True},
               "rects": {"main": [hx0, hz0, hx1, hz1]},
               "floors": list(res.get("floors") or [])}

    eave_y = res.get("eave_y") or (fy + 4)
    door_xz = _as_xz(res.get("door")) or fr["to"](u_door, hv0)
    u_dh = _clamp(fr["frm"](door_xz[0], door_xz[1])[0], hu0, hu1)

    # -- the engawa: boarded veranda the length of the house, under the eave --
    ew = 2 if (hv0 >= 4 and rng.random() < 0.6) else 1
    ev0 = hv0 - ew
    eng = set()
    for u in range(hu0, hu1 + 1):
        for v in range(ev0, hv0):
            eng.add((u, v))
    wu = None
    if hd >= 5 and rng.random() < 0.55:
        if hu0 - 1 >= 1:
            wu = hu0 - 1
        elif hu1 + 1 <= A - 2:
            wu = hu1 + 1
        if wu is not None:
            wend = min(hv1 + 1, P - 1, hv0 + rng.choice([3, 4, 5]))
            for v in range(ev0, wend):
                eng.add((wu, v))
    house_cells = set()
    for u in range(hu0, hu1 + 1):
        for v in range(hv0, hv1 + 1):
            house_cells.add((u, v))

    gw = 3 if A >= 7 else 2
    gu0 = _clamp(u_door - gw // 2, 1, max(1, A - 1 - gw))
    gu1 = _clamp(gu0 + gw - 1, gu0, A - 2)
    gate_cells = set((u, 0) for u in range(gu0, gu1 + 1))
    corridor = set()
    for u in range(gu0 - 1, gu1 + 2):
        for v in range(0, max(1, hv0)):
            corridor.add((u, v))

    def _yard_free():
        out = set()
        for u in range(1, A - 1):
            for v in range(1, P - 1):
                if (u, v) in house_cells or (u, v) in eng or (u, v) in corridor:
                    continue
                out.add((u, v))
        return out

    free = _yard_free()
    if not free and hu1 - hu0 >= 4:
        # a cottage yard with no room left in it: give the far end of the veranda back
        # to the yard so the farm has somewhere to keep things
        if u_door - hu0 <= hu1 - u_door:
            for v in range(ev0, hv0):
                eng.discard((hu1, v))
                eng.discard((hu1 - 1, v))
        else:
            for v in range(ev0, hv0):
                eng.discard((hu0, v))
                eng.discard((hu0 + 1, v))
        free = _yard_free()

    for (u, v) in sorted(eng):
        ex, ez = fr["to"](u, v)
        b.place_block(ex, fy, ez, m_deck)

    post_top = min(eave_y - 1, fy + 5)
    if ew >= 2 and post_top >= fy + 1:
        for u in range(hu0, hu1 + 1):
            if (u, ev0) not in eng:
                continue
            ex, ez = fr["to"](u, ev0)
            if abs(u - u_dh) <= 1:
                continue
            if (u - hu0) % 3 == 0 or u == hu1:
                b.place_cuboid(ex, fy + 1, ez, ex, post_top, ez, m_frame)
            else:
                b.place_block(ex, fy + 1, ez, m_fence)

    # ---------------- the yard wall, and the gate in it --------------------- No wall
    # course touching the house: it would stand under the eave and its top would be a
    # ledge with no way onto it.
    against = set()
    for (u, v) in house_cells:
        against.add((u + 1, v))
        against.add((u - 1, v))
        against.add((u, v + 1))
        against.add((u, v - 1))
    skip = set(house_cells) | eng | gate_cells | against
    # A second way out at the side, to the fields: without it the ground the library
    # laid outside the wall is ground nobody can ever stand on.
    for u_side in (0, A - 1):
        cand = [v for v in range(1, P - 1) if (u_side, v) not in skip]
        pairs = [v for v in cand if (v + 1) in cand]
        if pairs:
            v0 = pairs[len(pairs) // 2]
            skip.add((u_side, v0))
            skip.add((u_side, v0 + 1))
            break
    wall_h = rng.choice([2, 3, 3, 4])
    band = m_trim_s if wall_h >= 3 else None
    edges = [[(u, 0) for u in range(A)],
             [(u, P - 1) for u in range(A)],
             [(0, v) for v in range(P)],
             [(A - 1, v) for v in range(P)]]
    for edge in edges:
        for run in _runs(edge, skip):
            ax, az = fr["to"](run[0][0], run[0][1])
            bx, bz = fr["to"](run[-1][0], run[-1][1])
            b.wall(min(ax, bx), fy + 1, min(az, bz),
                   max(ax, bx), fy + wall_h, max(az, bz),
                   m_wall, post=m_frame, spacing=3,
                   base=m_foot, base_height=1, band=band)

    roofed_gate = hv0 >= 4 and rng.random() < 0.5
    pier_v = [0, 1] if roofed_gate else [0]
    for pu in (gu0 - 1, gu1 + 1):
        for pv in pier_v:
            gx, gz = fr["to"](pu, pv)
            b.place_cuboid(gx, fy + 1, gz, gx, fy + 3, gz, m_frame)
    for pv in pier_v:
        ax, az = fr["to"](gu0 - 1, pv)
        bx, bz = fr["to"](gu1 + 1, pv)
        b.place_cuboid(min(ax, bx), fy + 4, min(az, bz),
                       max(ax, bx), fy + 4, max(az, bz), m_frame)
    if roofed_gate:
        grx0, grz0, grx1, grz1 = _rect(fr, gu0 - 1, 0, gu1 + 1, 1)
        # The slope is the voice's (E015): `pitch=(1, 2)` stood here until the voice
        # contract, and `roof()` is handed the voice's profile over whatever is passed.
        b.roof(grx0, grz0, grx1, grz1, fy + 5, voice["roof"],
               style="gable", axis=_ridge_axis(grx0, grz0, grx1, grz1),
               overhang=0)
    else:
        ax, az = fr["to"](gu0 - 1, 0)
        bx, bz = fr["to"](gu1 + 1, 0)
        b.place_cuboid(min(ax, bx), fy + 5, min(az, bz),
                       max(ax, bx), fy + 5, max(az, bz), m_trim_s)

    # ------------------------- what the yard holds --------------------------
    best = _largest_rect(free, A, P, 6, 5)
    sheltered = None
    close_spots = []
    if best is not None and (best[3] - best[1]) >= 2 and (best[4] - best[2]) >= 2:
        su0, sv0, su1, sv1 = best[1], best[2], best[3], best[4]

        def _line_free(cs):
            return all(c in free for c in cs)

        def _line_shut(cs):
            return all(c not in free for c in cs)

        # A one-cell alley between the shed and whatever is beside it is a pocket nobody
        # can get to. Take it into the shed instead.
        vs = range(sv0, sv1 + 1)
        us = range(su0, su1 + 1)
        if _line_free([(su0 - 1, v) for v in vs]) and \
                _line_shut([(su0 - 2, v) for v in vs]):
            su0 -= 1
        if _line_free([(su1 + 1, v) for v in vs]) and \
                _line_shut([(su1 + 2, v) for v in vs]):
            su1 += 1
        if _line_free([(u, sv0 - 1) for u in us]) and \
                _line_shut([(u, sv0 - 2) for u in us]):
            sv0 -= 1
        if _line_free([(u, sv1 + 1) for u in us]) and \
                _line_shut([(u, sv1 + 2) for u in us]):
            sv1 += 1
        sx0, sz0, sx1, sz1 = _rect(fr, su0, sv0, su1, sv1)
        b.place_cuboid(sx0, fy, sz0, sx1, fy, sz1, m_foot)
        posts = [(sx0, sz0), (sx1, sz0), (sx0, sz1), (sx1, sz1)]
        if sx1 - sx0 >= 4:
            posts += [((sx0 + sx1) // 2, sz0), ((sx0 + sx1) // 2, sz1)]
        if sz1 - sz0 >= 4:
            posts += [(sx0, (sz0 + sz1) // 2), (sx1, (sz0 + sz1) // 2)]
        for (qx, qz) in posts:
            b.place_cuboid(qx, fy + 1, qz, qx, fy + 3, qz, m_frame)
        # The closed side of the lean-to only ever goes against something already solid,
        # so it shuts nothing off from the rest of the yard.
        closed = None
        if sv1 == P - 2 or sv1 + 1 == hv0:
            closed = ("v", sv1, -1)
        elif su0 == 1:
            closed = ("u", su0, 1)
        elif su1 == A - 2:
            closed = ("u", su1, -1)
        if closed is not None:
            if closed[0] == "v":
                face = [(u, closed[1]) for u in range(su0, su1 + 1)]
                back = [(u, closed[1] + closed[2]) for u in range(su0, su1 + 1)]
            else:
                face = [(closed[1], v) for v in range(sv0, sv1 + 1)]
                back = [(closed[1] + closed[2], v) for v in range(sv0, sv1 + 1)]
            wx0, wz0, wx1, wz1 = _rect(fr, face[0][0], face[0][1],
                                       face[-1][0], face[-1][1])
            b.place_cuboid(wx0, fy + 1, wz0, wx1, fy + 3, wz1, m_wall)
            for i in range(len(face)):
                wx, wz = fr["to"](face[i][0], face[i][1])
                ix, iz = fr["to"](back[i][0], back[i][1])
                close_spots.append((ix, iz, _facing_between(wx, wz, ix, iz)))
            rng.shuffle(close_spots)
        b.roof(sx0, sz0, sx1, sz1, fy + 4, voice["roof"],
               style=rng.choice(["hip", "gable"]),
               axis=_ridge_axis(sx0, sz0, sx1, sz1),
               overhang=0)          # pitch was (1, 2); the slope is the voice's (E015)
        sheltered = (sx0, sz0, sx1, sz1)

    if sheltered is not None:
        gx0, gz0, gx1, gz1 = sheltered
        # inside the shed, never on its open sides: the gaps between the posts are the
        # way in, and a shelf hung in one and a lantern in the other sealed a store in
        # the city
        ix0, iz0, ix1, iz1 = gx0 + 1, gz0 + 1, gx1 - 1, gz1 - 1
        if ix1 < ix0 or iz1 < iz0:
            ix0, iz0, ix1, iz1 = gx0, gz0, gx1, gz1
        spots = (close_spots
                 + _room_spots(rng, ix0, iz0, ix1, iz1, True)
                 + _room_spots(rng, ix0, iz0, ix1, iz1, False))
    else:
        cells = sorted(free, key=lambda c: -(abs(c[0] - u_door) + c[1]))
        spots = []
        for (u, v) in cells[:24]:
            cx, cz = fr["to"](u, v)
            for f in ("north", "south", "east", "west"):
                spots.append((cx, cz, f))

    laid = 0
    if spots:
        if yard_use == "store":
            wants = [("store", 3), ("store", 2), ("workbench", 0), ("shelf", 0)]
        elif yard_use == "byre":
            wants = [("trough", 0), ("fodder", 0), ("trough", 0), ("bench", 0)]
        else:
            wants = [("fodder", 0), ("fodder", 0), ("fodder", 0), ("store", 2)]
        for (kind, ext) in wants:
            mt = voice["footing"] if kind == "trough" else voice["wall"]
            if ext:
                got = _fit(b, kind, spots, fy + 1, mat=mt, extent=ext,
                           room=yard_use)
            else:
                got = _fit(b, kind, spots, fy + 1, mat=mt, room=yard_use)
            if got:
                laid += 1
        if _fit(b, "light", spots, fy + 1, mat=voice["wall"], room=yard_use):
            laid += 1
        if laid == 0:
            for kind in ("bench", "table", "shelf", "store", "light"):
                if _fit(b, kind, spots, fy + 1, mat=voice["wall"],
                        room=yard_use):
                    laid += 1
                    break

    if len(free) >= 16 and rng.random() < 0.6:
        cells = sorted(free, key=lambda c: abs(c[0] - u_door) + abs(c[1] - hv0))
        wspots = []
        for (u, v) in cells[:12]:
            cx, cz = fr["to"](u, v)
            for f in ("north", "south", "east", "west"):
                wspots.append((cx, cz, f))
        _fit(b, "well", wspots, fy + 1, mat=voice["footing"], room="yard")

    # ------------------------- inside the farmhouse -------------------------
    rooms = [r for r in (res.get("rooms") or []) if len(r) >= 5]
    base_clear = _keep_clear(res, door_xz)
    stair_xz = set()
    _collect_xz(res.get("stairs"), stair_xz, 0)
    sky = b.get_block(px0, min(fy + 50, 250), pz0)
    o0 = fr["to"](0, 0)
    o1 = fr["to"](0, 1)
    inside = (door_xz[0] + (o1[0] - o0[0]), door_xz[1] + (o1[1] - o0[1]))
    if rooms:
        ymin = min(r[1] for r in rooms)
        off = 1 if ymin == fy else 0
        for idx, r in enumerate(rooms):
            rx0, ry, rz0, rx1, rz1 = r[0], r[1], r[2], r[3], r[4]
            sy = ry + off
            ground = (ry == ymin)
            ways = _ways_in(b, sky, rx0, rz0, rx1, rz1, sy)
            entries = set(ways)
            if rx0 <= inside[0] <= rx1 and rz0 <= inside[1] <= rz1:
                entries.add(inside)
            for (cx, cz) in stair_xz:
                if rx0 <= cx <= rx1 and rz0 <= cz <= rz1:
                    entries.add((cx, cz))
            if not ground:
                # the stairwell is a hole in this floor: keep its head clear
                for x in range(rx0, rx1 + 1):
                    for z in range(rz0, rz1 + 1):
                        if b.get_block(x, sy - 1, z) == sky:
                            entries.add((x, z))
            near = sorted(entries, key=lambda e: abs(e[0] - inside[0])
                          + abs(e[1] - inside[1]))[:3]
            clear = set(base_clear) | _aisle(rx0, rz0, rx1, rz1, near)
            for (wx, wz) in entries:
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        clear.add((wx + dx, wz + dz))
            area = (rx1 - rx0 + 1) * (rz1 - rz0 + 1)
            budget = 1 if area <= 12 else max(1, area // 5)
            edge = _room_spots(rng, rx0, rz0, rx1, rz1, True, clear, entries)
            mid = _room_spots(rng, rx0, rz0, rx1, rz1, False, clear, entries)
            if not mid:
                mid = edge
            rname = "farmhouse" if ground else "house"
            if ground and idx == 0:
                todo = [("hearth", edge, {"mat": voice["footing"]}),
                        ("table", mid, {"mat": voice["wall"]}),
                        ("bench", edge, {"mat": voice["wall"]}),
                        ("store", edge, {"mat": voice["wall"], "extent": 2})]
                if storeys == 1:
                    todo.append(("bed", edge, {"mat": voice["wall"]}))
                todo.append(("rug", mid, {"mat": voice["trim"]}))
            elif ground:
                todo = [("store", edge, {"mat": voice["wall"], "extent": 3}),
                        ("workbench", edge, {"mat": voice["wall"]}),
                        ("bench", edge, {"mat": voice["wall"]})]
            else:
                todo = [("bed", edge, {"mat": voice["wall"]})]
                if area >= 12:
                    todo.append(("bed", edge, {"mat": voice["wall"]}))
                todo.append(("shelf", edge, {"mat": voice["wall"]}))
                todo.append(("store", edge, {"mat": voice["wall"],
                                             "extent": 2}))
            n = 0
            for (kind, sp, kw) in todo:
                if n >= budget:
                    break
                if _fit(b, kind, sp, sy, room=rname, **kw):
                    n += 1
            if _fit(b, "light", edge, sy, mat=voice["wall"], room=rname):
                n += 1
            if n == 0:
                # a corner is the one cell that can never cut a room in two
                loose = []
                for (cx, cz) in ((rx0, rz0), (rx1, rz0), (rx0, rz1),
                                 (rx1, rz1)):
                    if (cx, cz) in entries:
                        continue
                    for f in _ring_facings(cx, cz, rx0, rz0, rx1, rz1):
                        loose.append((cx, cz, f))
                rng.shuffle(loose)
                for kind in ("shelf", "bench", "store", "table", "light"):
                    if _fit(b, kind, loose, sy, mat=voice["wall"], room=rname):
                        break

    b.seal_voids(hx0, hz0, hx1, hz1, m_wall)
    b.seal_voids(px0, pz0, px1, pz1, m_foot, max_cells=96)
    res["emitted"] = emitted
    return res
