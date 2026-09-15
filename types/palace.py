import math
import random

KIND = "area"
FORM = "civic"
ROLE = "civic"

PARAMS = {
    "storeys": ("int", 1, 3),
    "court": ("choice", ["open", "garden", "colonnade"]),
}

NEEDS = {
    "footprint": (3, 3, 32, 32),
    "except": (18, 20, 21, 22, 24, 28),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
_SPANS = (3, 5, 7, 9, 11, 13, 15)
_HEADROOM = 28


def _est_roof(span):
    h = max(1, span // 2)
    return 2 * (1 + 2 * max(0, h - 1)) + 2


def _span_cap(eave_above):
    budget = _HEADROOM - eave_above
    best = 3
    for s in _SPANS:
        if _est_roof(s) <= budget:
            best = s
    return best


def _wall_h(storeys, extra):
    return 3 * (storeys - 1) + 5 + extra


def _max_depth(plat, storeys, extra=0):
    return _span_cap(plat + _wall_h(storeys, extra) + 1)


def _odds(lo, hi):
    lo = lo if lo % 2 else lo + 1
    return [v for v in range(lo, hi + 1, 2)]


def _frame_of(side, x0, z0, x1, z1):
    w = x1 - x0 + 1
    d = z1 - z0 + 1
    if side == "south":
        def f(u, v):
            return (x0 + u, z1 - v)
        return f, w, d, "north", "east", "west"
    if side == "west":
        def f(u, v):
            return (x0 + v, z0 + u)
        return f, d, w, "east", "south", "north"
    if side == "east":
        def f(u, v):
            return (x1 - v, z0 + u)
        return f, d, w, "west", "south", "north"

    def f(u, v):
        return (x0 + u, z0 + v)
    return f, w, d, "south", "east", "west"


def _r(f, u0, v0, u1, v1):
    ax, az = f(u0, v0)
    bx, bz = f(u1, v1)
    return min(ax, bx), min(az, bz), max(ax, bx), max(az, bz)


def _axis_u(side):
    return "z" if side in ("north", "south") else "x"


def _axis_v(side):
    return "x" if side in ("north", "south") else "z"


def _clamp(v, lo, hi):
    if hi < lo:
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _segs(lo, hi, skip):
    if hi < lo:
        return []
    if skip is None:
        return [(lo, hi)]
    out = []
    if skip - 2 >= lo:
        out.append((lo, skip - 2))
    if skip + 2 <= hi:
        out.append((skip + 2, hi))
    return out


def _open_run(b, C, ax, az, bx, bz, y, spacing, skip):
    mm = C["m"]
    if ax == bx:
        for a, c in _segs(min(az, bz), max(az, bz), skip):
            if c - a >= 2:
                b.openings(ax, y, a, ax, c, spacing=spacing, width=1, sill=2,
                           head=4, sill_block=mm["trim"], lintel=mm["frame"])
    else:
        for a, c in _segs(min(ax, bx), max(ax, bx), skip):
            if c - a >= 2:
                b.openings(a, y, az, c, az, spacing=spacing, width=1, sill=2,
                           head=4, sill_block=mm["trim"], lintel=mm["frame"])


def _wall_run(b, C, ax, ay, az, bx, by, bz):
    mm = C["m"]
    b.wall(ax, ay, az, bx, by, bz, mm["wall"], post=mm["frame"],
           spacing=C["sp"], base=mm["foot"], base_height=1, band=mm["trim"])


def _fence_run(b, C, ax, ay, az, bx, by, bz):
    mm = C["m"]
    b.wall(ax, ay, az, bx, by, bz, mm["foot"], post=mm["frame"],
           spacing=C["sp"] + 1, base=mm["foot"], base_height=1, band=mm["trim"])


def _inside(C, ax, az, bx, bz):
    px0, pz0, px1, pz1 = C["pad"]
    return ax >= px0 and az >= pz0 and bx <= px1 and bz <= pz1


def _hall(b, C, u0, v0, u1, v1, plat, storeys, door_face, door_at,
          style, ridge_axis, want_ov, room_kind, extra=0):
    """One roofed hall on the local grid. Returns its measurements."""
    f = C["f"]
    m = C["m"]
    voice = C["voice"]
    fy = C["fy"]
    wx0, wz0, wx1, wz1 = _r(f, u0, v0, u1, v1)
    if wx1 - wx0 < 2 or wz1 - wz0 < 2:
        return None
    ov = want_ov
    if not _inside(C, wx0 - ov, wz0 - ov, wx1 + ov, wz1 + ov):
        ov = 0
    floor_y = fy + plat
    iu = (u1 - u0) - 1
    iv = (v1 - v0) - 1
    wall_h = _wall_h(storeys, extra)
    top_y = floor_y + wall_h
    eave_y = top_y + 1

    b.place_cuboid(wx0, floor_y, wz0, wx1, floor_y, wz1, m["floor"])
    _wall_run(b, C, wx0, floor_y + 1, wz0, wx1, top_y, wz0)
    _wall_run(b, C, wx0, floor_y + 1, wz1, wx1, top_y, wz1)
    _wall_run(b, C, wx0, floor_y + 1, wz0, wx0, top_y, wz1)
    _wall_run(b, C, wx1, floor_y + 1, wz0, wx1, top_y, wz1)

    if door_face == "front":
        dwx, dwz = f(_clamp(door_at, u0 + 1, u1 - 1), v0)
        dfac = C["outward"]
    elif door_face == "back":
        dwx, dwz = f(_clamp(door_at, u0 + 1, u1 - 1), v1)
        dfac = C["inward"]
    elif door_face == "uminus":
        dwx, dwz = f(u0, _clamp(door_at, v0 + 1, v1 - 1))
        dfac = C["uminus"]
    else:
        dwx, dwz = f(u1, _clamp(door_at, v0 + 1, v1 - 1))
        dfac = C["uplus"]
    b.doorway(dwx, floor_y + 1, dwz, facing=dfac, mat=voice["frame"],
              leaf=b.joinery(voice, "door"), jamb="build", lintel=True)

    runs = [(wx0 + 1, wz0, wx1 - 1, wz0), (wx0 + 1, wz1, wx1 - 1, wz1),
            (wx0, wz0 + 1, wx0, wz1 - 1), (wx1, wz0 + 1, wx1, wz1 - 1)]
    for k in range(storeys):
        yk = floor_y + 4 * k
        if yk + 4 > top_y:
            break
        for run in runs:
            ax, az, bx, bz = run
            skip = None
            if k == 0:
                if ax == bx and dwx == ax and min(az, bz) <= dwz <= max(az, bz):
                    skip = dwz
                elif az == bz and dwz == az and min(ax, bx) <= dwx <= max(ax, bx):
                    skip = dwx
            _open_run(b, C, ax, az, bx, bz, yk, C["sp"], skip)

    b.roof(wx0, wz0, wx1, wz1, eave_y, mat=voice["roof"], style=style,
           axis=ridge_axis, overhang=ov)

    if ov >= 1:
        yb = eave_y - 1
        b.place_cuboid(wx0, yb, wz0 - 1, wx1, yb, wz0 - 1, m["rslab"])
        b.place_cuboid(wx0, yb, wz1 + 1, wx1, yb, wz1 + 1, m["rslab"])
        b.place_cuboid(wx0 - 1, yb, wz0, wx0 - 1, yb, wz1, m["rslab"])
        b.place_cuboid(wx1 + 1, yb, wz0, wx1 + 1, yb, wz1, m["rslab"])

    return {"rect": (wx0, wz0, wx1, wz1), "floor_y": floor_y, "eave_y": eave_y,
            "door": (dwx, floor_y + 1, dwz),
            "room": room_kind, "iu": iu, "iv": iv}


def _furnish(b, C, h):
    if h is None:
        return
    voice = C["voice"]
    wx0, wz0, wx1, wz1 = h["rect"]
    y = h["floor_y"] + 1
    kind = h["room"]
    ix0 = wx0 + 1
    iz0 = wz0 + 1
    ix1 = wx1 - 1
    iz1 = wz1 - 1
    if ix1 < ix0 or iz1 < iz0:
        return
    iw = ix1 - ix0 + 1
    idp = iz1 - iz0 + 1
    cx = (ix0 + ix1) // 2
    cz = (iz0 + iz1) // 2
    ddx, _dy, ddz = h["door"]

    def clear(xx, zz):
        if xx < ix0 or xx > ix1 or zz < iz0 or zz > iz1:
            return False
        return max(abs(xx - ddx), abs(zz - ddz)) >= 3

    # Nothing here may ever close a room off: a fitting only goes on the interior's
    # outer ring, and a pair of them across a run only where the run is wide enough that
    # a lane is left open past them.
    spots = [("light", ix0, iz0, None), ("light", ix1, iz1, None),
             ("light", ix1, iz0, None), ("light", ix0, iz1, None)]
    if iw >= 9 and idp >= 3:
        for xx in (ix0 + iw // 4, ix1 - iw // 4):
            spots.append(("light", xx, iz0, None))
            spots.append(("light", xx, iz1, None))
    if idp >= 9 and iw >= 3:
        for zz in (iz0 + idp // 4, iz1 - idp // 4):
            spots.append(("light", ix0, zz, None))
            spots.append(("light", ix1, zz, None))
    if iw >= 9 and idp >= 5:
        spots.append(("table", cx, cz, "south"))
        spots.append(("bench", cx - 2, cz, "east"))
        spots.append(("bench", cx + 2, cz, "west"))
    elif idp >= 9 and iw >= 5:
        spots.append(("table", cx, cz, "east"))
        spots.append(("bench", cx, cz - 2, "south"))
        spots.append(("bench", cx, cz + 2, "north"))
    for what, xx, zz, fac in spots:
        if not clear(xx, zz):
            continue
        if fac is None:
            b.fitting(what, xx, y, zz, room=kind, mat=voice["trim"])
        else:
            b.fitting(what, xx, y, zz, facing=fac, room=kind,
                      mat=voice["frame"])


def _platform(b, C, u0, v0, u1, v1, plat, step_u, step_w):
    """Two courses of masonry under a hall, with its flight on the court side."""
    if plat <= 0:
        return
    f = C["f"]
    m = C["m"]
    fy = C["fy"]
    voice = C["voice"]
    px0, pz0, px1, pz1 = C["pad"]
    ax, az, bx, bz = _r(f, u0 - 1, v0 - 1, u1 + 1, v1 + 1)
    ax = max(ax, px0)
    az = max(az, pz0)
    bx = min(bx, px1)
    bz = min(bz, pz1)
    b.place_cuboid(ax, fy + 1, az, bx, fy + plat, bz, m["foot"])
    lo = _clamp(step_u - step_w // 2, 1, max(1, C["W"] - 2))
    hi = _clamp(lo + step_w - 1, lo, C["W"] - 2)
    if plat >= 2 and v0 - 3 >= 1:
        rx0, rz0, rx1, rz1 = _r(f, lo, v0 - 2, hi, v0 - 2)
        b.place_cuboid(rx0, fy + 1, rz0, rx1, fy + 1, rz1, m["foot"])
        for uu in range(lo, hi + 1):
            ax1, az1 = f(uu, v0 - 3)
            ax2, az2 = f(uu, v0 - 2)
            b.steps([(ax1, fy + 1, az1), (ax2, fy + 2, az2)], voice["footing"])
    elif v0 - 2 >= 1:
        for uu in range(lo, hi + 1):
            ax1, az1 = f(uu, v0 - 2)
            b.steps([(ax1, fy + 1, az1)], voice["footing"])


def _perimeter(b, C, gate_u, gate_w, wall_h):
    """The courtyard wall round the whole compound, with its gate at the front."""
    f = C["f"]
    m = C["m"]
    voice = C["voice"]
    fy = C["fy"]
    W = C["W"]
    D = C["D"]
    y0 = fy + 1
    y1 = fy + wall_h
    ax, az, bx, bz = _r(f, 0, D - 1, W - 1, D - 1)
    _fence_run(b, C, ax, y0, az, bx, y1, bz)
    ax, az, bx, bz = _r(f, 0, 0, 0, D - 1)
    _fence_run(b, C, ax, y0, az, bx, y1, bz)
    ax, az, bx, bz = _r(f, W - 1, 0, W - 1, D - 1)
    _fence_run(b, C, ax, y0, az, bx, y1, bz)

    gl = _clamp(gate_u - gate_w // 2, 0, max(0, W - gate_w))
    gr = min(W - 1, gl + gate_w - 1)
    if gl - 1 >= 0:
        ax, az, bx, bz = _r(f, 0, 0, gl - 1, 0)
        _fence_run(b, C, ax, y0, az, bx, y1, bz)
    if gr + 1 <= W - 1:
        ax, az, bx, bz = _r(f, gr + 1, 0, W - 1, 0)
        _fence_run(b, C, ax, y0, az, bx, y1, bz)

    gd = min(3, D - 1)
    if gd < 1:
        return gl, gr
    py = fy + wall_h + 2
    piers = []
    if gl - 1 >= 0:
        piers.append(gl - 1)
    if gr + 1 <= W - 1:
        piers.append(gr + 1)
    for uu in piers:
        for vv in range(0, gd):
            pxx, pzz = f(uu, vv)
            b.place_cuboid(pxx, fy + 1, pzz, pxx, py, pzz, m["frame"])
    lu0 = max(0, gl - 1)
    lu1 = min(W - 1, gr + 1)
    lx0, lz0, lx1, lz1 = _r(f, lu0, 0, lu1, gd - 1)
    b.place_cuboid(lx0, py + 1, lz0, lx1, py + 1, lz1, m["trim"])
    b.roof(lx0, lz0, lx1, lz1, py + 2, mat=voice["roof"], style="hip",
           axis=_axis_u(C["side"]), overhang=0)
    gx, gz = f((gl + gr) // 2, max(0, gd - 2))
    b.fitting("light", gx, fy + 1, gz, room="hall", mat=voice["trim"])
    return gl, gr


def _pavilion(b, C, u0, v0, u1, v1, y_top, spacing, door_u, style, axis, ov):
    """Posts and a roof, open on every side: what a palace is when the pad is tiny."""
    f = C["f"]
    m = C["m"]
    voice = C["voice"]
    fy = C["fy"]
    wx0, wz0, wx1, wz1 = _r(f, u0, v0, u1, v1)
    if not _inside(C, wx0 - ov, wz0 - ov, wx1 + ov, wz1 + ov):
        ov = 0
    for uu in range(u0, u1 + 1):
        for vv in (v0, v1):
            if vv == v0 and uu == door_u:
                continue
            edge = (uu == u0 or uu == u1)
            if not edge and (uu - u0) % spacing != 0:
                continue
            pxx, pzz = f(uu, vv)
            b.place_cuboid(pxx, fy + 1, pzz, pxx, y_top, pzz, m["frame"])
    for vv in range(v0 + 1, v1):
        for uu in (u0, u1):
            if (vv - v0) % spacing != 0:
                continue
            pxx, pzz = f(uu, vv)
            b.place_cuboid(pxx, fy + 1, pzz, pxx, y_top, pzz, m["frame"])
    b.place_cuboid(wx0, y_top, wz0, wx1, y_top, wz0, m["trim"])
    b.place_cuboid(wx0, y_top, wz1, wx1, y_top, wz1, m["trim"])
    b.place_cuboid(wx0, y_top, wz0, wx0, y_top, wz1, m["trim"])
    b.place_cuboid(wx1, y_top, wz0, wx1, y_top, wz1, m["trim"])
    if ov >= 1:
        b.place_cuboid(wx0, y_top, wz0 - 1, wx1, y_top, wz0 - 1, m["rslab"])
        b.place_cuboid(wx0, y_top, wz1 + 1, wx1, y_top, wz1 + 1, m["rslab"])
        b.place_cuboid(wx0 - 1, y_top, wz0, wx0 - 1, y_top, wz1, m["rslab"])
        b.place_cuboid(wx1 + 1, y_top, wz0, wx1 + 1, y_top, wz1, m["rslab"])
    b.roof(wx0, wz0, wx1, wz1, y_top + 1, mat=voice["roof"], style=style,
           axis=axis, overhang=ov)
    return y_top + 1


def _court(b, C, u0, v0, u1, v1, axis_u, court, rnd):
    """What the open ground between the gate and the hall is made of."""
    f = C["f"]
    m = C["m"]
    voice = C["voice"]
    fy = C["fy"]
    if u1 - u0 < 2 or v1 - v0 < 2:
        return
    half = 1 if (u1 - u0) < 8 else 2
    ax, az, bx, bz = _r(f, _clamp(axis_u - half, u0, u1),
                        v0, _clamp(axis_u + half, u0, u1), v1)
    b.place_cuboid(ax, fy, az, bx, fy, bz, m["floor"])
    if court == "colonnade":
        for uu in (u0 + 1, u1 - 1):
            for vv in range(v0 + 1, v1, 3):
                pxx, pzz = f(uu, vv)
                b.place_cuboid(pxx, fy + 1, pzz, pxx, fy + 4, pzz, m["frame"])
                b.place_cuboid(pxx, fy + 5, pzz, pxx, fy + 5, pzz, m["rslab"])
    elif court == "garden":
        cu = (u0 + u1) // 2
        cv = (v0 + v1) // 2
        rw = min(3, (u1 - u0) // 3)
        rd = min(3, (v1 - v0) // 3)
        if rw >= 1 and rd >= 1:
            for du in (-1, 1):
                gu0 = _clamp(cu + du * (rw + 2) - rw, u0 + 1, u1 - 1)
                gu1 = _clamp(cu + du * (rw + 2) + rw, u0 + 1, u1 - 1)
                gv0 = _clamp(cv - rd, v0 + 1, v1 - 1)
                gv1 = _clamp(cv + rd, v0 + 1, v1 - 1)
                if gu1 - gu0 < 1 or gv1 - gv0 < 1:
                    continue
                ax, az, bx, bz = _r(f, gu0, gv0, gu1, gv1)
                b.place_cuboid(ax, fy + 1, az, bx, fy + 1, bz, m["rslab"])
                if bx - ax >= 2 and bz - az >= 2:
                    b.place_cuboid(ax + 1, fy + 1, az + 1, bx - 1, fy + 1,
                                   bz - 1, m["floor"])
                    b.fitting("light", (ax + bx) // 2, fy + 2, (az + bz) // 2,
                              room="hall", mat=voice["trim"])
    else:
        for du in (-1, 1):
            uu = _clamp(axis_u + du * (half + 3), u0 + 1, u1 - 1)
            for vv in range(v0 + 2, v1, 4):
                pxx, pzz = f(uu, vv)
                b.place_cuboid(pxx, fy + 1, pzz, pxx, fy + 2, pzz, m["foot"])
                b.place_cuboid(pxx, fy + 3, pzz, pxx, fy + 3, pzz, m["rslab"])


def build(b, part, seed, **params):
    voice = part["voice"]
    if "footprint" in part and part["footprint"]:
        x0, z0, x1, z1 = part["footprint"]
    else:
        x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    if x1 < x0:
        x0, x1 = x1, x0
    if z1 < z0:
        z0, z1 = z1, z0
    fy = part["floor_y"]
    label = part.get("label")
    rnd = random.Random(int(seed))

    storeys = int(params.get("storeys", 2) or 2)
    storeys = _clamp(storeys, 1, 3)
    court = params.get("court", "open")
    if court not in ("open", "garden", "colonnade"):
        court = "open"

    m = {
        "wall": b.block(voice["wall"], "full"),
        "foot": b.block(voice["footing"], "full"),
        "frame": b.block(voice["frame"], "full"),
        "trim": b.block(voice["trim"], "full"),
        "floor": b.block(voice["floor"], "full"),
        "rslab": b.block(voice["roof"], "slab"),
    }

    door = part.get("door")
    side = None
    dx = dz = None
    if door:
        dx = int(door[0])
        dz = int(door[1])
        cand = [("north", abs(dz - z0)), ("south", abs(z1 - dz)),
                ("west", abs(dx - x0)), ("east", abs(x1 - dx))]
        cand.sort(key=lambda t: t[1])
        side = cand[0][0]
    if side is None:
        fac = part.get("facing")
        side = fac if fac in _OPP else "north"

    f, W, D, inward, uplus, uminus = _frame_of(side, x0, z0, x1, z1)
    if dx is None:
        u_door = W // 2
    elif side in ("north", "south"):
        u_door = _clamp(dx - x0, 0, W - 1)
    else:
        u_door = _clamp(dz - z0, 0, W - 1)

    C = {"f": f, "W": W, "D": D, "side": side, "inward": inward,
         "outward": _OPP[inward], "uplus": uplus, "uminus": uminus,
         "m": m, "voice": voice, "fy": fy, "pad": (x0, z0, x1, z1),
         "sp": 3 + (abs(int(seed)) % 2), "label": label,
         "vs": abs(int(seed)) % 4}

    b.place_cuboid(x0, fy, z0, x1, fy, z1, m["foot"])
    if min(W, D) >= 5:
        b.place_cuboid(x0 + 1, fy, z0 + 1, x1 - 1, fy, z0 + 1, m["floor"])
        b.place_cuboid(x0 + 1, fy, z1 - 1, x1 - 1, fy, z1 - 1, m["floor"])
        b.place_cuboid(x0 + 1, fy, z0 + 1, x0 + 1, fy, z1 - 1, m["floor"])
        b.place_cuboid(x1 - 1, fy, z0 + 1, x1 - 1, fy, z1 - 1, m["floor"])

    small = min(W, D)
    au = _axis_u(side)
    av = _axis_v(side)
    halls = []

    if small <= 4:
        _small_palace(b, C, W, D, u_door, storeys, rnd)
    else:
        halls = _big_palace(b, C, W, D, u_door, storeys, court, rnd, au, av)

    for h in halls:
        _furnish(b, C, h)
        b.check_door(h["door"][0], h["door"][1], h["door"][2])

    b.seal_voids(x0, z0, x1, z1, m["wall"], max_cells=64)
    b.check_attached()
    b.check_walkable()
    return {"halls": len(halls)}


def _fit_roof(hw, hd, cap, au, av):
    """Run the ridge along the longer side and hold the span it crosses to cap."""
    if hw >= hd:
        return hw, min(hd, cap), au
    return min(hw, cap), hd, av


def _gateway(b, C, u_door, W, wall_h):
    """Two piers and a beam where there is no room for a wall to hang a gate in."""
    f = C["f"]
    m = C["m"]
    fy = C["fy"]
    lu = u_door - 2
    ru = u_door + 2
    if lu < 0 or ru > W - 1:
        lu = _clamp(u_door - 2, 0, W - 1)
        ru = _clamp(u_door + 2, 0, W - 1)
    if ru - lu < 2:
        return
    top = fy + wall_h
    for uu in (lu, ru):
        pxx, pzz = f(uu, 0)
        b.place_cuboid(pxx, fy + 1, pzz, pxx, top, pzz, m["frame"])
    ax, az, bx, bz = _r(f, lu, 0, ru, 0)
    b.place_cuboid(ax, top + 1, az, bx, top + 1, bz, m["trim"])
    b.place_cuboid(ax, top + 2, az, bx, top + 2, bz, m["rslab"])


def _small_palace(b, C, W, D, u_door, storeys, rnd):
    """Three or four across: one open hall, and the way through it left clear."""
    if min(W, D) < 3:
        return
    au = _axis_u(C["side"])
    av = _axis_v(C["side"])
    fy = C["fy"]
    m = C["m"]
    vs = C["vs"]
    y_top = fy + 3 + storeys + (vs % 3)
    length = min(D, [9, 11, 13, 17][vs] + storeys * 2)
    pv0 = D - length
    pv1 = D - 1
    style = ["hip", "gable", "gable", "hip"][vs]
    if abs(W - D) > 1 or length != D:
        style = "gable"
    if W >= (pv1 - pv0 + 1):
        axis = au
    else:
        axis = av
    door_u = u_door if pv0 == 0 else -1
    _pavilion(b, C, 0, pv0, W - 1, pv1, y_top, 2 + (vs % 2), door_u,
              style, axis, 0)
    f = C["f"]
    step = 3 + (vs % 2)
    for vv in range(1, max(1, pv0 - 1)):
        if (vv % step) != 1:
            continue
        for uu in (0, W - 1):
            pxx, pzz = f(uu, vv)
            b.place_cuboid(pxx, fy + 1, pzz, pxx, fy + 2, pzz, m["foot"])
            b.place_cuboid(pxx, fy + 3, pzz, pxx, fy + 3, pzz, m["rslab"])
    cx, cz = f(W // 2, (pv0 + pv1) // 2)
    b.fitting("light", cx, fy + 1, cz, room="hall", mat=C["voice"]["trim"])


def _big_palace(b, C, W, D, u_door, storeys, court, rnd, au, av):
    fy = C["fy"]
    small = min(W, D)
    halls = []
    vs = C["vs"]
    style = ["hip", "gable", "gable", "hip"][vs]
    extra = vs % 3

    if small < 10:
        inset = 1 if small >= 7 else 0
        au0 = inset
        au1 = W - 1 - inset
        av0 = inset
        av1 = D - 1 - inset
        cap = _max_depth(0, storeys, extra)
        hw = min(au1 - au0 + 1, 21)
        avail_d = av1 - av0 + 1
        hd = min(avail_d, 17)
        if avail_d >= 9:
            hd = min(hd, max(5, avail_d - 2 - (vs % 4)))
        hw, hd, ax_r = _fit_roof(hw, hd, cap, au, av)
        hu0 = au0 + ((au1 - au0 + 1) - hw) // 2
        hu1 = hu0 + hw - 1
        hv1 = av1
        hv0 = hv1 - hd + 1
        h = _hall(b, C, hu0, hv0, hu1, hv1, 0, storeys, "front",
                  _clamp(u_door, hu0 + 1, hu1 - 1), style, ax_r,
                  1 if inset >= 1 else 0, "hall", extra)
        if h is not None:
            halls.append(h)
        if hv0 - av0 >= 3 and W >= 7:
            _gateway(b, C, u_door, W, 4 + (vs % 2))
            _court(b, C, au0, av0, au1, hv0 - 2, u_door, court, rnd)
        return halls

    wall_h = 3 + (vs % 2)
    gate_w = 3 if W < 13 else (3 if vs < 2 else 5)
    _perimeter(b, C, u_door, gate_w, wall_h)

    cap = _max_depth(2, storeys, extra)
    hd_hi = min(cap, 17, max(4, D - 7))
    hd = max(4, hd_hi - (vs % 3))
    hv1 = D - 3
    hv0 = hv1 - hd + 1
    if hv0 < 4:
        hv0 = min(4, max(1, hv1 - 2))
        hd = hv1 - hv0 + 1
    plat = 1 if hv0 >= 3 else 0
    span_w = W - 4
    choices = _odds(5, min(span_w, 23))
    if not choices:
        choices = [max(3, min(span_w, 5))]
    pick = len(choices) - 1 - (vs % max(1, min(4, len(choices))))
    hw = choices[max(0, pick)]
    hw, hd2, ax_r = _fit_roof(hw, hd, cap, au, av)
    hd = hd2
    hv0 = hv1 - hd + 1
    if hv0 < 4:
        hv0 = 4
        hd = hv1 - hv0 + 1
    plat = 1 if hv0 >= 3 else 0
    hw = min(hw, max(9, hd * 3 + 1), span_w)
    if hw % 2 == 0:
        hw -= 1
    hw = max(5, min(hw, span_w))
    hu0 = 2 + (span_w - hw) // 2
    hu1 = hu0 + hw - 1
    axis_u = (hu0 + hu1) // 2

    _platform(b, C, hu0, hv0, hu1, hv1, plat, axis_u,
              _clamp(3 + 2 * (vs % 3), 1, max(1, hw - 2)))
    h = _hall(b, C, hu0, hv0, hu1, hv1, plat, storeys, "front", axis_u,
              style, ax_r, 1, "hall", extra)
    if h is not None:
        halls.append(h)

    cu0 = 2
    cu1 = W - 3
    cv0 = 3
    cv1 = max(cv0, hv0 - 2 - plat)

    if small >= 20:
        left_room = (hu0 - 1) - 2
        right_room = (W - 3) - (hu1 + 1)
        shw = min(9, left_room, right_room)
        if shw % 2 == 0:
            shw -= 1
        if shw >= 7:
            opts = _odds(5, shw)
            shw = opts[len(opts) - 1 - (vs % len(opts))]
        sv1 = hv0 - 3 - plat
        sv0 = max(3, sv1 - [17, 13, 9, 15][vs])
        if shw >= 5 and sv1 - sv0 >= 6:
            sst = max(1, storeys - 1)
            sextra = vs % 2
            scap = _max_depth(0, sst, sextra)
            sw2, sd2, sax = _fit_roof(sv1 - sv0 + 1, shw, scap, av, au)
            shw = sd2
            sv0 = sv1 - sw2 + 1
            lh = _hall(b, C, 2, sv0, 2 + shw - 1, sv1, 0, sst, "uplus",
                       (sv0 + sv1) // 2, style, sax, 1, "house", sextra)
            if lh is not None:
                halls.append(lh)
            rh = _hall(b, C, W - 3 - shw + 1, sv0, W - 3, sv1, 0, sst,
                       "uminus", (sv0 + sv1) // 2, style, sax, 1, "house",
                       sextra)
            if rh is not None:
                halls.append(rh)
            cu0 = 2 + shw + 1
            cu1 = W - 3 - shw - 1

    if cu1 - cu0 >= 2 and cv1 - cv0 >= 2:
        _court(b, C, cu0, cv0, cu1, cv1, axis_u, court, rnd)
    return halls
