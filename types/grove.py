"""A grove: standing trees on the setting's own ground, in rows or as a copse.

The thing a place puts on ground it means to keep green -- an orchard behind a farm, a
stand at the head of an avenue, the planting a low quarter is mostly made of. Like the
garden it is planted in `voice["ground"]`, the surface `site()` read off the land beside
the part, so a grove is grass on a plain without this file naming a block.

A tree here is built rather than grown: a trunk of the voice's own frame timber and a
canopy of leaves over it, every cell of the canopy touching the trunk or another cell
that does, because half a canopy left hanging is the floating-leaves giveaway.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**. `civic`: a stand of trees belongs in a farm belt, a
#: garden quarter and a palace court alike.
ROLE = "civic"

PARAMS = {
    "planting": ("choice", ["rows", "copse", "avenue"]),
    "floor": ("choice", ["turf", "swept"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # Measured by scripts/type_needs.py; the band is rounds/type-needs.json. The pair is
    # the pad site() hands build(), after siting's inset.
    "footprint": (3, 3, 48, 48),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

#: The craft round, E6: the leaf is the **voice's**, matched to its frame timber by
#: `b.foliage()`, and this is only what a grove falls back to where nothing is handed
#: it. A belt of orchards in a Japanese voice had dark-oak trunks under oak leaves, and
#: every palette this project has read the same green.
_LEAF = "oak_leaves[persistent=true]"
_UNDER = ("fern", "poppy", "dandelion", "azure_bluet")

#: How tall a trunk is, and how far the canopy reaches. Small on purpose: a grove in a
#: walled quarter that overtops the wall is a wood, and the clearance a plan keeps round
#: an area is one block.
_TRUNK = (3, 4)
_REACH = 1


def _rect(part):
    fp = part.get("footprint")
    if fp is not None:
        return int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3])
    return int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])


def _spots(x0, z0, x1, z1, planting, rng, cx, cz):
    """Where the trunks stand: inside the rim by two, so no canopy leaves the part."""
    ax0, az0, ax1, az1 = x0 + 2, z0 + 2, x1 - 2, z1 - 2
    if ax1 < ax0 or az1 < az0:
        return []
    out = []
    if planting == "rows":
        step = rng.choice([3, 4])
        for x in range(ax0, ax1 + 1, step):
            for z in range(az0, az1 + 1, step):
                out.append((x, z))
    elif planting == "avenue":
        for z in range(az0, az1 + 1, 3):
            out.append((ax0, z))
            if ax1 != ax0:
                out.append((ax1, z))
    else:                                            # copse
        cells = [(x, z) for x in range(ax0, ax1 + 1) for z in range(az0, az1 + 1)]
        rng.shuffle(cells)
        want = max(1, len(cells) // 12)
        for c in cells:
            if len(out) >= want:
                break
            if any(abs(c[0] - o[0]) < 3 and abs(c[1] - o[1]) < 3 for o in out):
                continue
            out.append(c)
    return out


def _tree(b, x, z, y, height, trunk, rng, leaf=_LEAF):
    """A trunk with a canopy that touches it everywhere."""
    for i in range(height):
        b.place_block(x, y + 1 + i, z, trunk)
    top = y + height
    for dy in (0, 1):
        r = _REACH if dy == 0 else 0
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dy == 0 and dx == 0 and dz == 0:
                    continue
                b.place_block(x + dx, top + dy, z + dz, leaf)
    b.place_block(x, top + 1, z, leaf)


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    planting = params.get("planting", "rows")
    if planting not in ("rows", "copse", "avenue"):
        planting = "rows"
    floor = params.get("floor", "turf")
    if floor not in ("turf", "swept"):
        floor = "turf"

    ground = b.block(b.voice["ground"])
    swept = b.block(b.voice["footing"])
    trunk = b.axial(b.block(b.voice["frame"], "post"), "y")
    leaf = b.foliage(b.voice["frame"]) + "[persistent=true]"
    b.fill_region(x0, y + 1, z0, x1, y + 8, z1, "air")

    door = part.get("door")
    dx, dz = (int(door[0]), int(door[-1])) if door else (None, None)
    walk = set()
    if dx is not None and min(w, d) >= 5:
        # a swept way in, from the reserved doorway to the middle
        for t in range(0, max(abs(cx - dx), abs(cz - dz)) + 1):
            wx = dx + (1 if cx > dx else -1 if cx < dx else 0) * min(t, abs(cx - dx))
            wz = dz + (1 if cz > dz else -1 if cz < dz else 0) * min(t, abs(cz - dz))
            walk.add((wx, wz))

    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if (x, z) in walk:
                b.place_block(x, y, z, swept)
            else:
                b.place_block(x, y, z, swept if floor == "swept" else ground)
            b.place_block(x, y + 1, z, "air")

    spots = [c for c in _spots(x0, z0, x1, z1, planting, rng, cx, cz)
             if c not in walk]
    planted = 0
    for (tx, tz) in spots:
        _tree(b, tx, tz, y, rng.randint(*_TRUNK), trunk, rng, leaf)
        planted += 1

    # the undergrowth, where the floor is turf and nothing stands
    under = 0
    if floor == "turf":
        for x in range(x0 + 1, x1):
            for z in range(z0 + 1, z1):
                if (x, z) in walk or (x, z) in spots:
                    continue
                if (x * 7 + z * 13 + seed) % 11:
                    continue
                if b.get_block(x, y + 1, z).split("[")[0] != "air":
                    continue
                b.place_block(x, y + 1, z, rng.choice(_UNDER))
                under += 1

    b.check_attached()
    # the reserved doorway stays walkable, whatever the floor and the border did
    b.area_way_in(part["x0"], part["z0"], part["x1"], part["z1"], int(part["floor_y"]))
    return {"ok": True, "planting": planting, "floor": floor, "trees": planted,
            "undergrowth": under, "cells": w * d}
