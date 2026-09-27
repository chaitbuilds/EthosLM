"""A round altar: concentric railed stone terraces with a crowning pavilion or altar.

A type, not a building. The circular terraced monument an ensemble sets beside or on its
axis -- a stepped round mound of stone, each tier coped and railed, climbed by four
flights on the cardinal axes, crowned by a round pavilion of posts under a conical roof
(or left open round a central altar stone). It is what the composition round needed for
the paired round monuments that flank the principal mass in the references it read, and
it is general: a round terrace, a stupa-like mound or an open-air altar is the same
composition at other parameters.

  tiers    how many stepped terraces, each two courses (the lowest three), stepping in
           evenly from the pad's inscribed circle
  top      `pavilion` (posts round a conical roof, a finial) or `open` (an altar stone
           on the top terrace)

Every material is a voice role: the terraces in the footing, their lips in its finer
stone, the rails in its wall shape, the pavilion's posts in the frame and its roof in the
roof role, which carries the rank's colour. The seed changes nothing structural.
"""

import math

FORM = "east_asian"
ROLE = "civic"

PARAMS = {
    "tiers": ("int", 1, 4),
    "top": ("choice", ["pavilion", "open"]),
}

#: Measured by `scripts/cf_forms.py altar` and a sweep of square pads 11..64: under 11
#: across three tiers do not step and a flight does not fit.
NEEDS = {
    "footprint": (11, 11, 72, 72),
    "except": (),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}


def _disc_cells(cx, cz, r):
    rr = r + 0.5
    n = int(math.ceil(r)) + 1
    return [(x, z) for x in range(int(cx) - n, int(cx) + n + 1)
            for z in range(int(cz) - n, int(cz) + n + 1)
            if (x - cx) ** 2 + (z - cz) ** 2 <= rr * rr]


def _ring_cells(cx, cz, r):
    inner = set(_disc_cells(cx, cz, r - 1))
    return [c for c in _disc_cells(cx, cz, r) if c not in inner]


def build(b, part, seed, **params):
    tiers = max(1, min(4, int(params.get("tiers") or 3)))
    top = params.get("top") or "pavilion"
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    fy = int(part["floor_y"])
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    # the lowest flight starts three rows out from the lowest lip: keep it on the pad
    R = min(x1 - x0, z1 - z0) // 2 - 4
    foot = b.block(b.voice["footing"])
    fine = b.block(b.voice["footing"], "fine")
    rail = b.block(b.voice["footing"], "wall")
    trim = b.block(b.voice["trim"])
    notes = []
    # the paved ground round it
    b.place_cuboid(x0, fy, z0, x1, fy, z1, fine)
    b.place_cuboid(x0, fy + 1, z0, x1, fy + 12, z1, "air")
    step = max(3, R // (tiers + 2))
    while tiers > 1 and R - (tiers - 1) * step < 5:
        tiers -= 1
        notes.append("tiers: fewer fit the pad")
    radii, tops = [], []
    y = fy
    for k in range(tiers):
        r = R - k * step
        h = 3 if k == 0 else 2
        cells = _disc_cells(cx, cz, r)
        for (x, z) in cells:
            b.place_cuboid(x, y + 1, z, x, y + h, z, foot)
        for (x, z) in _ring_cells(cx, cz, r):
            b.place_block(x, y + h, z, fine)
        y += h
        radii.append(r)
        tops.append(y)
    # the flights on the four axes: three wide, one course a row, the middle in trim
    treads = 0
    for (dx, dz, axis) in ((0, 1, "z"), (0, -1, "z"), (1, 0, "x"), (-1, 0, "x")):
        for w in (-1, 0, 1):
            cells = []
            yy = fy
            for k in range(tiers):
                r = radii[k]
                rn = radii[k + 1] if k + 1 < tiers else None
                h = tops[k] - (tops[k - 1] if k else fy)
                # the flight starts h rows outside the tier's lip and climbs onto it
                for j in range(h):
                    d = r + h - j
                    x = cx + dx * d + (w if dx == 0 else 0)
                    z = cz + dz * d + (w if dz == 0 else 0)
                    yy += 1
                    b.place_cuboid(x, (tops[k - 1] if k else fy) + 1, z, x, yy - 1, z, foot) \
                        if yy - 1 > (tops[k - 1] if k else fy) else None
                    b.place_cuboid(x, yy, z, x, yy + 3, z, "air")
                    cells.append((x, yy, z))
            mat = b.voice["trim"] if w == 0 else b.voice["footing"]
            b.steps(cells, mat, axis=axis)
            treads += len(cells)
    # the rails: on the lip of every tier, open where a flight arrives
    railed = 0
    for k, r in enumerate(radii):
        yr = tops[k] + 1
        for (x, z) in _ring_cells(cx, cz, r):
            if abs(x - cx) <= 2 or abs(z - cz) <= 2:
                continue
            b.place_block(x, yr, z, rail)
            railed += 1
    rt = radii[-1]
    yt = tops[-1]
    if top == "pavilion" and rt >= 5:
        rp = max(2, rt - 4)
        post = b.axial(b.block(b.voice["frame"], "post"), "y")
        H = 4 if rp < 5 else 5
        n_posts = 8 if rp < 5 else 12
        for i in range(n_posts):
            a = 2 * math.pi * i / n_posts
            px, pz = int(round(cx + rp * math.cos(a))), int(round(cz + rp * math.sin(a)))
            b.place_cuboid(px, yt + 1, pz, px, yt + H, pz, post)
        # the lintel ring and the roof
        for (x, z) in _ring_cells(cx, cz, rp):
            b.place_block(x, yt + H + 1, z, b.block(b.voice["frame"]))
        for (x, z) in _disc_cells(cx, cz, rp - 1):
            b.place_block(x, yt + H + 1, z, b.block(b.voice["wall"]))
        ridge = b.roof_cone(cx, yt + H + 2, cz, rp + 2, b.voice["roof"])
        b.place_block(cx, ridge + 1, cz, trim)
        b.place_block(cx, ridge + 2, cz, b.block(b.voice["trim"], "slab"))
        topy = ridge + 2
    else:
        b.place_cuboid(cx - 1, yt + 1, cz - 1, cx + 1, yt + 1, cz + 1, trim)
        topy = yt + 1
    b.check_walkable(part.get("label"))
    emitted = {"requested": {"tiers": int(params.get("tiers") or 3), "top": top},
               "tiers": tiers, "radii": radii, "top": top, "height": topy - fy,
               "treads": treads, "rail": railed, "fallback": "; ".join(notes) or None,
               "features": {"tiers": tiers, "treads": treads, "rail": railed}}
    return {"ok": True, "footprint": (x0, z0, x1, z1), "storeys": 1, "emitted": emitted}
