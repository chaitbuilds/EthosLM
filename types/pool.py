"""A pool: open water held in a stone basin, with a walk round it.

A well is a shaft and a lake is the land's; a pool is **made**, which is the kind of
water a village gathers round, a garden is laid out beside and a court is cooled by.
A type builds from the level `site()` left and never digs, so the basin is **raised**:
the walk round it is paved at the site's level, the basin wall stands on it to the
water's surface and a kerb of the voice's trim caps it -- the tank of a garden court or
an oasis, held rather than spilled. The kerb is open on the side the way in comes from,
and a few pads float on the water where it is big enough to take them.

The water is contained by construction, not by luck: every rim column is walled with
the footing course up to the surface and the bed is the paving itself, so water laid
here stays where it was put when the world ticks.
"""

import random

KIND = "area"
FORM = "civic"
ROLE = "civic"

#: The pool is water a person can reach: the walk round it and the steps into it.
FEATURES = ("water",)

PARAMS = {
    "shape": ("choice", ["round", "square"]),
    "kerb": ("choice", ["low", "flush"]),
    "depth": ("int", 1, 2),
}

NEEDS = {
    # a rim, a kerb and at least a 3x3 of water; 48 is the widest area the library lays
    "footprint": (7, 7, 48, 48),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}


def _rect(part):
    fp = part.get("footprint")
    if fp is not None:
        return int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3])
    return int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])


def _inside(shape, x, z, x0, z0, x1, z1):
    """Is (x, z) water: inside the basin, which is the rect less its walk ring."""
    ix0, iz0, ix1, iz1 = x0 + 2, z0 + 2, x1 - 2, z1 - 2
    if not (ix0 <= x <= ix1 and iz0 <= z <= iz1):
        return False
    if shape == "square":
        return True
    cx, cz = (ix0 + ix1) / 2.0, (iz0 + iz1) / 2.0
    rx, rz = (ix1 - ix0) / 2.0 + 0.5, (iz1 - iz0) / 2.0 + 0.5
    return ((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2 <= 1.0


def build(b, part, seed, shape=None, kerb=None, depth=None, **kw):
    rng = random.Random(int(seed) * 131 + 7)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    shape = shape if shape in ("round", "square") else rng.choice(["round", "square"])
    kerb = kerb if kerb in ("low", "flush") else "low"
    depth = max(1, min(2, int(depth) if isinstance(depth, int) else 1))
    voice = part["voice"]
    walk = b.block(voice["floor"])
    wall = b.block(voice["footing"])
    cap = b.block(voice["trim"], "slab")

    water = {(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)
             if _inside(shape, x, z, x0, z0, x1, z1)}
    if len(water) < 9:
        return {"ok": False, "reason": f"a {x1 - x0 + 1}x{z1 - z0 + 1} area leaves "
                                       f"{len(water)} cells of water inside its walk"}
    edge = {(x + dx, z + dz) for (x, z) in water for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)} - water
    b.fill_region(x0, y + 1, z0, x1, y + 4, z1, "air")
    # the walk round the basin and the basin's own bed, at the site's level
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b.place_block(x, y, z, walk)
    # the basin wall up to the surface, the kerb over it, the water held inside
    for (x, z) in edge:
        for yy in range(y + 1, y + depth + 1):
            b.place_block(x, yy, z, wall)
        if kerb == "low":
            b.place_block(x, y + depth + 1, z, cap)
    for (x, z) in water:
        for yy in range(y + 1, y + depth + 1):
            b.place_block(x, yy, z, "water")
    # the way in: where the reserved doorway is, the kerb is open and steps go down
    way = b.door_cell(x0, z0, x1, z1)
    steps = 0
    if way is not None:
        wx, wz = int(way[0]), int(way[1])
        best = min(edge, key=lambda c: abs(c[0] - wx) + abs(c[1] - wz))
        if kerb == "low":
            b.place_block(best[0], y + depth + 1, best[1], "air")
        steps = 1
    # lily pads, where the pool is big enough to be looked across
    pads = 0
    if len(water) >= 40:
        cells = sorted(water)
        rng.shuffle(cells)
        for (x, z) in cells[:max(1, len(water) // 30)]:
            if all(abs(x - ex) + abs(z - ez) > 1 for (ex, ez) in edge):
                b.place_block(x, y + depth + 1, z, "lily_pad")
                pads += 1
    b.check_attached()
    b.area_way_in(part["x0"], part["z0"], part["x1"], part["z1"], y)
    return {"ok": True, "shape": shape, "water_cells": len(water), "rim": len(edge),
            "steps": steps, "pads": pads, "cells": (x1 - x0 + 1) * (z1 - z0 + 1),
            "emitted": {"features": {"water": True},
                        "rects": {"water": [min(p[0] for p in water),
                                            min(p[1] for p in water),
                                            max(p[0] for p in water),
                                            max(p[1] for p in water)]}}}
