"""A workshop: one working floor under a steep roof, with a walled yard and a
wide cart door opening onto it."""

import random

FORM = "european_vernacular"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "urban"

# a workshop is one working floor: storeys is declared, and there is one value of it
# this type will build.
PARAMS = {
    "storeys": ("int", 1, 1),
    "yard_side": ("choice", ["north", "east", "south", "west"]),
    "trade": ("choice", ["smithy", "joinery", "pottery", "dyeworks"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3x3 and up measured: one storey with a yard beside it stands on anything. Measured
    # by scripts/type_needs.py; the band is rounds/type-needs.json. The pair is the pad
    # site() hands build(), after siting's inset.
    "footprint": (3, 3, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

# Nine module constants of block names, put to the voice instead. A workshop is a form
# -- one working floor, a walled yard, a wide cart door -- and it is the same workshop
# in every palette.

def _axial(b, block, axis):
    return b.axial(block, axis)


def _wallb(b):
    return b.block(b.voice["wall"])


def _panels(b):
    """The three faces of the wall material, for the odd course that varies."""
    w = b.voice["wall"]
    return (b.block(w, "accent"), b.block(w, "fine"), b.block(w, "bare"))


def _wall_faces(b):
    """Every face of the wall material -- what "this cell is still render" means."""
    w = b.voice["wall"]
    return tuple(dict.fromkeys((b.block(w), b.block(w, "accent"), b.block(w, "fine"),
                                b.block(w, "bare"), b.block(w, "post"))))


def _post(b, axis="y"):
    return _axial(b, b.block(b.voice["frame"], "post"), axis)


def _trim(b, axis="y"):
    return _axial(b, b.block(b.voice["trim"], "bare"), axis)


def _foot(b):
    return b.block(b.voice["footing"])


def _foot2(b):
    return b.block(b.voice["footing"], "accent")


def _cap(b):
    return b.block(b.voice["roof"], "slab") + "[type=top]"


CW = ["north", "east", "south", "west"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
DELTA = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

# what each trade puts on its floor, best first: (kind, extent)
TRADES = {
    "smithy": [("forge", 1), ("anvil", 1), ("trough", 1), ("workbench", 1),
               ("store", 2), ("shelf", 1), ("bench", 1)],
    "joinery": [("workbench", 1), ("bench", 1), ("store", 3), ("table", 1),
                ("shelf", 1), ("rug", 1)],
    "pottery": [("oven", 1), ("table", 1), ("store", 2), ("shelf", 1),
                ("workbench", 1), ("bench", 1)],
    "dyeworks": [("trough", 1), ("hearth", 1), ("table", 1), ("store", 3),
                 ("shelf", 1), ("bench", 1)],
}
YARD_KIT = {
    "smithy": [("store", 2), ("trough", 1), ("bench", 1)],
    "joinery": [("store", 3), ("bench", 1), ("table", 1)],
    "pottery": [("store", 2), ("trough", 1), ("bench", 1)],
    "dyeworks": [("trough", 1), ("store", 2), ("fodder", 1)],
}
SMOKY = ("smithy", "pottery")


def _cell(side, fixed, along, out=0):
    """A cell on the wall line `fixed` of `side`, `along` it, `out` blocks out."""
    dx, dz = DELTA[side]
    if side in ("east", "west"):
        return (fixed + dx * out, along)
    return (along, fixed + dz * out)


def _wall_of(side, x0, z0, x1, z1):
    """(the wall's fixed coordinate, its low end along it, its high end)."""
    if side == "north":
        return z0, x0, x1
    if side == "south":
        return z1, x0, x1
    if side == "west":
        return x0, z0, z1
    return x1, z0, z1


def _along_axis(side):
    return "x" if side in ("north", "south") else "z"


def _out_axis(side):
    return "z" if side in ("north", "south") else "x"


def _ring(x0, z0, x1, z1):
    """Every cell of a rectangle's edge, with the way you face off that edge."""
    out = []
    for x in range(x0, x1 + 1):
        out.append((x, z0, "south"))
        out.append((x, z1, "north"))
    for z in range(z0 + 1, z1):
        out.append((x0, z, "east"))
        out.append((x1, z, "west"))
    return out


def _roof_spec(rng, axis, bw, bd):
    """Steep gable with a deep overhang, and never twice the same one."""
    r = {"style": "gable", "axis": axis, "pitch": (2, 1)}
    kind = rng.randrange(6)
    if kind == 1:
        r["profile"] = [(2, 1), (3, 1)]
    elif kind == 2:
        r["ends"] = ("half-hip", "gable")
    elif kind == 3:
        r["eave"] = "flared"
    elif kind == 4:
        r["pitch"] = (3, 1)
    elif kind == 5:
        # a second tier leaves a loft between the two that nobody can reach
        r["ends"] = ("gable", "half-hip") if min(bw, bd) >= 7 else (
            "half-hip", "half-hip")
    return r


def _pick_yard(rng, want, dedge, px0, pz0, px1, pz1, dx, dz):
    """Which side the yard takes, and how deep. Never the lane's own side, and
    never a strip the reserved doorstep stands in; and never the far side,
    where the only way round to the yard is off the plot."""
    idx = CW.index(want) if want in CW else 1
    for i in range(4):
        s = CW[(idx + i) % 4]
        if s == dedge or s == OPP[dedge]:
            continue
        dim = (px1 - px0 + 1) if s in ("east", "west") else (pz1 - pz0 + 1)
        d = min(rng.choice([2, 3, 3, 4]), dim - 4)
        if d < 2:
            continue
        if s == "north" and dz < pz0 + d:
            continue
        if s == "south" and dz > pz1 - d:
            continue
        if s == "west" and dx < px0 + d:
            continue
        if s == "east" and dx > px1 - d:
            continue
        return s, d
    return None, 0


def _floating(report):
    """Every cell check_attached() says is held up by nothing, whatever shape
    it hands them back in."""
    out = []
    stack = [report]
    while stack:
        it = stack.pop()
        if isinstance(it, dict):
            for k in ("floating", "cells", "cell", "blocks"):
                if k in it:
                    stack.append(it[k])
            for k in ("x", "y", "z"):
                if k in it and "x" in it and "y" in it and "z" in it:
                    out.append((it["x"], it["y"], it["z"]))
                    break
        elif isinstance(it, (list, tuple)):
            if len(it) == 3 and all(isinstance(v, int) for v in it):
                out.append(tuple(it))
            else:
                for k in it:
                    stack.append(k)
    return out


def _backed(b, x, z, facing, y):
    """True where the wall behind a cell is solid -- nothing stands in front of
    a door or an opening, or the room behind it cannot be walked into."""
    dx, dz = DELTA[OPP[facing]]
    for h in (y, y + 1):
        s = b.get_block(x + dx, h, z + dz)
        if s in ("air", "cave_air", "water") or "door" in s or "gate" in s:
            return False
    return True


def _fit(b, rng, kinds, cells, y, room, budget):
    """Try each fitting against the cells until one takes. Refusals are fine."""
    placed = []
    used = set()
    cells = [c for c in cells if _backed(b, c[0], c[1], c[2], y)]
    for kind, extent in kinds:
        if len(placed) >= budget:
            break
        for (x, z, facing) in cells:
            if (x, z) in used:
                continue
            r = b.fitting(kind, x, y, z, facing, mat=b.voice["footing"], extent=extent,
                          room=room)
            if r and r.get("ok"):
                placed.append(kind)
                for c in r.get("cells") or [(x, y, z)]:
                    used.add((c[0], c[2]))
                used.add((x, z))
                break
    return placed


def _hood(b, rng, side, fixed, lo, hi, top, lamp):
    """The signature move over the entrance: a bay carried out on brackets over
    the doorway -- brackets a course down, the projecting course over them, and
    a glazed cheek at each end so it reads as a bay and not a shelf."""
    oax = _out_axis(side)
    beam = rng.choice([b.block(b.voice["trim"], "bare"),
                       b.block(b.voice["frame"], "post")])
    deck = rng.choice([_cap(b), b.block(b.voice["floor"], "slab") + "[type=top]"])
    for a in (lo, hi):
        x, z = _cell(side, fixed, a, 1)
        b.place_block(x, top - 1, z, b.axial(beam, oax))
    for a in range(lo, hi + 1):
        x, z = _cell(side, fixed, a, 1)
        b.place_block(x, top, z, deck)
        if lo < a < hi and rng.random() < 0.7:
            b.place_block(x, top - 1, z, "glass_pane")
    if lamp:
        x, z = _cell(side, fixed, (lo + hi) // 2, 1)
        b.place_block(x, top - 1, z, "lantern[hanging=true]")


def _cart_door(b, rng, side, fixed, lo, hi, fy):
    """A wide door: two leaves in a timber frame under a glazed transom.
    Returns the span the frame occupies, posts included."""
    span = hi - lo + 1
    leaves = 2 if span >= 5 else 1
    grp = leaves + 2                      # leaves plus a jamb either side
    start = lo + (span - grp) // 2
    inward = OPP[side]
    door_a = [start + 1 + i for i in range(leaves)]
    for a in door_a:
        x, z = _cell(side, fixed, a, 0)
        b.fill_region(x, fy + 1, z, x, fy + 2, z, "air")
    ok = 0
    for a in door_a:
        x, z = _cell(side, fixed, a, 0)
        r = b.doorway(x, fy + 1, z, inward, b.voice["frame"], leaf=b.block(b.voice["frame"], "door"),
                      jamb="build", lintel=True)
        if r and r.get("ok"):
            ok += 1
    if not ok:                            # refused: leave it an open cart bay
        for a in door_a:
            x, z = _cell(side, fixed, a, 0)
            b.fill_region(x, fy + 1, z, x, fy + 2, z, "air")
    for a in door_a:                      # transom over the leaves
        x, z = _cell(side, fixed, a, 0)
        b.place_block(x, fy + 3, z, "glass")
    aax = _along_axis(side)
    p0, p1 = start, start + grp - 1
    for a in (p0, p1):
        if lo <= a <= hi:
            x, z = _cell(side, fixed, a, 0)
            for y in range(fy + 1, fy + 4):
                b.place_block(x, y, z, _trim(b))
    for a in range(p0, p1 + 1):
        if lo <= a <= hi:
            x, z = _cell(side, fixed, a, 0)
            b.place_block(x, fy + 4, z, _post(b, aax))
    return max(lo, p0), min(hi, p1)


def _yard_wall(b, rng, ys, open_side, yx0, yz0, yx1, yz1, fy, h):
    """A wall course round the working yard, open on one side for the way in."""
    outer = ys
    sides = ["north", "south"] if ys in ("east", "west") else ["east", "west"]
    for s in sides + [outer]:
        if s == open_side:
            continue
        fixed, a0, a1 = _wall_of(s, yx0, yz0, yx1, yz1)
        c0 = _cell(s, fixed, a0, 0)
        c1 = _cell(s, fixed, a1, 0)
        b.wall(c0[0], fy + 1, c0[1], c1[0], fy + h, c1[1], _foot(b),
               post=b.block(b.voice["frame"], "post"), spacing=4,
               band=b.joinery(b.voice, "fence"))
    # the way in, left open: one gatepost, on the corner the outer wall already holds,
    # and nothing at all across the gap
    fixed, a0, a1 = _wall_of(open_side, yx0, yz0, yx1, yz1)
    a = a0 if ys in ("north", "west") else a1
    x, z = _cell(open_side, fixed, a, 0)
    if yx0 <= x <= yx1 and yz0 <= z <= yz1:
        for y in range(fy + 1, fy + h + 2):
            b.place_block(x, y, z, _trim(b))


def build(b, part, seed, **params):
    rng = random.Random(seed * 131 + 7)
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    fy = part["floor_y"]
    label = part["label"]
    dcell = part["door"]
    dx, dz = dcell[0], dcell[1]
    trade = params.get("trade") or "smithy"
    if trade not in TRADES:
        trade = "smithy"

    # which edge of the pad the way in arrives on
    if dx <= px0:
        dedge = "west"
    elif dx >= px1:
        dedge = "east"
    elif dz <= pz0:
        dedge = "north"
    elif dz >= pz1:
        dedge = "south"
    else:
        dedge = OPP.get(part.get("facing"), "west")

    # the yard takes a strip off one side; never the side the lane is on, and never the
    # strip the doorstep stands in
    want = params.get("yard_side") or "east"
    ys, depth = _pick_yard(rng, want, dedge, px0, pz0, px1, pz1, dx, dz)

    bx0, bz0, bx1, bz1 = px0, pz0, px1, pz1
    yard = None
    if ys == "north":
        bz0 = pz0 + depth
        yard = (px0, pz0, px1, bz0 - 1)
    elif ys == "south":
        bz1 = pz1 - depth
        yard = (px0, bz1 + 1, px1, pz1)
    elif ys == "west":
        bx0 = px0 + depth
        yard = (px0, pz0, bx0 - 1, pz1)
    elif ys == "east":
        bx1 = px1 - depth
        yard = (bx1 + 1, pz0, px1, pz1)

    # where there is room, stand the front wall a block back off the lane so the oriel
    # over the entrance has somewhere to go
    inset = 0
    tw, td = bx1 - bx0 + 1, bz1 - bz0 + 1
    if rng.random() < 0.75:
        if dedge in ("east", "west") and tw - 1 >= 6 and td >= 6:
            inset = 1
            bx0, bx1 = (bx0 + 1, bx1) if dedge == "west" else (bx0, bx1 - 1)
        elif dedge in ("north", "south") and td - 1 >= 6 and tw >= 6:
            inset = 1
            bz0, bz1 = (bz0 + 1, bz1) if dedge == "north" else (bz0, bz1 - 1)

    kw = {"brackets": True}
    bw, bd = bx1 - bx0 + 1, bz1 - bz0 + 1
    n = 1                                 # a workshop is one working floor
    ax = "z" if bw >= bd else "x"
    rspec = _roof_spec(rng, ax, bw, bd)
    # a stack goes up a gable end, never through a slope: a chimney standing out of a
    # pitch leaves the courses beside it facing down-slope
    ends = ["east", "west"] if ax == "z" else ["north", "south"]
    smoke = [s for s in ends if s not in (ys, dedge)]
    if trade in SMOKY and smoke:
        rspec.pop("ends", None)
        kw["chimney"] = smoke[0]
        kw["flashing"] = True
    res = b.building(label, bx0, bz0, bx1, bz1, storeys=n, roof=rspec, mat=dict(b.voice),
                     openings="rhythm", stair="none", **kw)
    if not res.get("ok"):
        # The slope is the voice's (E015): a `"pitch": (2, 1)` stood in this spec and
        # the one below until the voice contract; `roof()` is handed the voice's.
        res = b.building(label, bx0, bz0, bx1, bz1, storeys=1, mat=dict(b.voice),
                         roof={"style": "gable", "axis": ax},
                         openings="rhythm", stair="none")
    rooms = res.get("rooms") or []
    if not res.get("ok"):
        b.place_cuboid(bx0, fy, bz0, bx1, fy, bz1, _foot(b))
        for (x, z, _f) in _ring(bx0, bz0, bx1, bz1):
            b.place_cuboid(x, fy + 1, z, x, fy + 4, z, _wallb(b))
        b.roof(bx0, bz0, bx1, bz1, fy + 5, b.voice["roof"], style="gable",
               axis=ax)
        fd, lod, hid = _wall_of(dedge, bx0, bz0, bx1, bz1)
        ad = dz if dedge in ("east", "west") else dx
        ad = max(lod + 1, min(hid - 1, ad))
        x, z = _cell(dedge, fd, ad, 0)
        b.fill_region(x, fy + 1, z, x, fy + 2, z, "air")
        b.doorway(x, fy + 1, z, OPP[dedge], b.voice["frame"], leaf=b.block(b.voice["frame"], "door"))
        rooms = [(bx0 + 1, fy, bz0 + 1, bx1 - 1, bz1 - 1)]
    return _dress(b, rng, part, res, rooms, label, fy, n, trade,
                  (bx0, bz0, bx1, bz1), yard, ys, dedge, inset, (dx, dz))


def _dress(b, rng, part, res, rooms, label, fy, n, trade, brect, yard, ys,
           dedge, inset, dcell):
    bx0, bz0, bx1, bz1 = brect
    dx, dz = dcell
    ey = res.get("eave_y") or (fy + 4 * n)
    if not (fy + 2 < ey < fy + 4 * n + 5):
        ey = fy + 4 * n

    # the frame: posts at the corners, braces under the eaves, and a stone footing
    # course carrying it clear of the wet, render panels between
    corners = set([(bx0, bz0), (bx1, bz0), (bx0, bz1), (bx1, bz1)])
    for (x, z) in corners:
        for y in range(fy + 1, ey + 1):
            b.place_block(x, y, z, _trim(b))
    foot_h = rng.choice([1, 1, 2])
    for (x, z, _f) in _ring(bx0, bz0, bx1, bz1):
        if (x, z) in corners:
            continue
        for y in range(fy + 1, ey + 1):
            if b.get_block(x, y, z) not in _wall_faces(b):
                continue
            if y <= fy + foot_h:
                b.place_block(x, y, z,
                              _foot(b) if rng.random() < 0.75 else _foot2(b))
            elif rng.random() < 0.10:
                b.place_block(x, y, z, rng.choice(_panels(b)))
    for s in CW:                          # knee braces down off the corner posts
        fixed, a0, a1 = _wall_of(s, bx0, bz0, bx1, bz1)
        aax = _along_axis(s)
        for base, step in ((a0, 1), (a1, -1)):
            for k in (1, 2):
                a = base + step * k
                if not (a0 < a < a1) or ey - k <= fy + foot_h:
                    continue
                x, z = _cell(s, fixed, a, 0)
                if b.get_block(x, ey - k, z) in _wall_faces(b):
                    b.place_block(x, ey - k, z, _trim(b, aax))

    # the wide door, onto the yard
    cs = ys or OPP[dedge]
    cf, clo, chi = _wall_of(cs, bx0, bz0, bx1, bz1)
    span = None
    if chi - clo >= 4:
        span = _cart_door(b, rng, cs, cf, clo + 1, chi - 1, fy)

    # the signature, over an entrance
    if inset:
        hs, hf, hlo, hhi = dedge, None, None, None
        hf, l0, l1 = _wall_of(dedge, bx0, bz0, bx1, bz1)
        ad = dz if dedge in ("east", "west") else dx
        ad = max(l0 + 1, min(l1 - 1, ad))
        hlo, hhi = max(l0, ad - 1), min(l1, ad + 1)
        _hood(b, rng, hs, hf, hlo, hhi, fy + 4, True)
    elif span:
        c0, c1 = span
        while c1 - c0 > 4:
            c0 += 1
            c1 -= 1
        _hood(b, rng, cs, cf, c0, c1, fy + 4, rng.random() < 0.6)
    if inset and span and ys:             # a hoist beam over the cart door
        a = (span[0] + span[1]) // 2
        x, z = _cell(cs, cf, a, 1)
        b.place_block(x, ey - 1, z, _trim(b, _out_axis(cs)))
        if ey - 3 >= fy + 4:              # never hung where a cart goes under
            b.place_block(x, ey - 2, z, "iron_chain")
            b.place_block(x, ey - 3, z, "lantern[hanging=true]")

    # the yard
    forbid = set()
    for side, a, wide in ((dedge, dz if dedge in ("east", "west") else dx, 1),):
        f2, l0, l1 = _wall_of(side, bx0, bz0, bx1, bz1)
        a = max(l0, min(l1, a))
        for out in (-1, -2):
            for da in (-1, 0, 1):
                forbid.add(_cell(side, f2, a + da, out))
    if span:
        for out in (-1, -2, 1, 2):
            for a in range(span[0] - 1, span[1] + 2):
                forbid.add(_cell(cs, cf, a, out))
    if yard:
        yx0, yz0, yx1, yz1 = yard
        perp = ["north", "south"] if ys in ("east", "west") else ["east", "west"]
        opens = dedge if dedge in perp else rng.choice(perp)
        _yard_wall(b, rng, ys, opens, yx0, yz0, yx1, yz1, fy,
                   rng.choice([2, 2, 3]))
        for x in range(yx0, yx1 + 1):
            for z in range(yz0, yz1 + 1):
                r = rng.random()
                if r < 0.5:
                    b.place_block(x, fy, z, _foot(b))
                elif r < 0.68:
                    b.place_block(x, fy, z, _foot2(b))
        # yard gear stands against the workshop wall, and only where there is still a
        # clear row to walk past it
        ydepth = (yz1 - yz0 + 1) if ys in ("north", "south") else (yx1 - yx0 + 1)
        if ydepth >= 3:
            ycells = []
            for a in range(clo, chi + 1):
                x, z = _cell(cs, cf, a, 1)
                if (x, z) in forbid:
                    continue
                if yx0 <= x <= yx1 and yz0 <= z <= yz1:
                    ycells.append((x, z, ys))
            rng.shuffle(ycells)
            _fit(b, rng, [("light", 1)] + YARD_KIT[trade], ycells, fy + 1,
                 "yard", rng.choice([2, 3, 3]))
        else:
            # one row wide: whatever stands here stands at the dead end, where it shuts
            # nothing off behind it
            alo, ahi = ((yx0, yx1) if ys in ("north", "south")
                        else (yz0, yz1))
            lowside = "west" if ys in ("north", "south") else "north"
            far = ahi - 1 if opens == lowside else alo + 1
            face = lowside if opens == lowside else OPP[lowside]
            x, z = _cell(cs, cf, far, 1)
            if (x, z) not in forbid and yx0 <= x <= yx1 and yz0 <= z <= yz1:
                _fit(b, rng, YARD_KIT[trade] + [("light", 1)],
                     [(x, z, face)], fy + 1, "yard", 1)

    # the floor of the shop, and whatever is over it
    ground_y = min([r[1] for r in rooms]) if rooms else fy
    for r in rooms:
        rx0, ry, rz0, rx1, rz1 = r[0], r[1], r[2], r[3], r[4]
        cells = [(x, z, f) for (x, z, f) in _ring(rx0, rz0, rx1, rz1)
                 if (x, z) not in forbid]
        rng.shuffle(cells)
        area = (rx1 - rx0 + 1) * (rz1 - rz0 + 1)
        if ry <= ground_y:
            kinds = list(TRADES[trade])
            room = "workshop"
        else:
            kinds = [("store", 3), ("shelf", 1), ("bookshelf", 1), ("table", 1),
                     ("bed", 1)]
            room = "store"
        rng.shuffle(kinds)
        kinds = [("light", 1)] + kinds
        budget = max(2, min(len(kinds), area // 4))
        if min(rx1 - rx0 + 1, rz1 - rz0 + 1) <= 2:
            budget = min(budget, 2)       # a narrow room keeps its walking room
        _fit(b, rng, kinds, cells, ry + 1, room, budget)

    b.seal_voids(bx0, bz0, bx1, bz1, _wallb(b))
    b.check_walkable(label)
    # anything left hanging gets the masonry under it that it should have had
    for (x, y, z) in _floating(b.check_attached()):
        if y - 1 >= fy + 1 and b.get_block(x, y - 1, z) in ("air", "cave_air"):
            b.place_block(x, y - 1, z, _wallb(b))
    return {"trade": trade, "storeys": n, "yard": ys}
