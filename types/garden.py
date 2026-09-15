"""A garden: planted ground with a path through it, an edge, and something to sit on.

The ground a low-density quarter is mostly made of, and the thing a district planner
had no way to ask for. It is planted in the **setting's** own surface -- `voice["ground"]`,
which `site()` reads off the land beside the part before it lays a block -- so a garden
on a plain is grass and a garden on a shore is sand, with no block name in this file and
no voice that has to name one.

The path, the edging and the beds are the voice's floor, trim and footing; the flowers
are flowers, which are not a palette.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**. `civic` is admitted in a district of any role: a
#: garden is as much a thing in a farm belt as in a noble quarter, and a compound's
#: courts want them too.
ROLE = "civic"

PARAMS = {
    "layout": ("choice", ["cross", "border", "beds"]),
    "edge": ("choice", ["hedge", "rail", "kerb"]),
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

#: The small plants a bed is planted with. Flowers and not materials: a poppy is not a
#: role any voice names and E014 reads material positions, not every string.
_FLOWERS = ("poppy", "dandelion", "cornflower", "azure_bluet", "oxeye_daisy",
            "allium", "lily_of_the_valley")
_BUSH = "oak_leaves[persistent=true]"


def _rect(part):
    fp = part.get("footprint")
    if fp is not None:
        return int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3])
    return int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])


def _ring(x0, z0, x1, z1):
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


def _paths(x0, z0, x1, z1, layout, cx, cz):
    """The cells a person walks on, as a set."""
    out = set()
    if layout == "cross":
        for x in range(x0, x1 + 1):
            out.add((x, cz))
        for z in range(z0, z1 + 1):
            out.add((cx, z))
    elif layout == "border":
        for (x, z) in _ring(x0 + 1, z0 + 1, x1 - 1, z1 - 1):
            out.add((x, z))
    else:                                            # beds
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                if (x - x0) % 5 == 0 or (z - z0) % 5 == 0:
                    out.add((x, z))
    return out


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    layout = params.get("layout", "cross")
    if layout not in ("cross", "border", "beds"):
        layout = "cross"
    edge = params.get("edge", "hedge")
    if edge not in ("hedge", "rail", "kerb"):
        edge = "hedge"
    if min(w, d) < 5:
        layout = "cross"

    ground = b.block(b.voice["ground"])
    path = b.block(b.voice["floor"])
    kerb = b.block(b.voice["trim"], "slab")
    b.fill_region(x0, y + 1, z0, x1, y + 4, z1, "air")

    walk = _paths(x0, z0, x1, z1, layout, cx, cz)
    # **The rim cell where the way in is**, from the library: the reserved doorway where
    # the part has one and the threshold on the network where it does not, snapped to
    # this rectangle's own perimeter. A border closed over the way in is E008 and it is
    # what the first occupancy run found in this type.
    way = b.door_cell(x0, z0, x1, z1)
    dx, dz = way if way else (None, None)
    if dx is not None:
        walk.add((dx, dz))

    beds = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if (x, z) in walk:
                b.place_block(x, y, z, path)
            else:
                b.place_block(x, y, z, ground)
                beds.append((x, z))
            b.place_block(x, y + 1, z, "air")
            b.place_block(x, y + 2, z, "air")

    # The edge, broken where the lane comes in. A hedge is a course of leaf on the rim,
    # a rail is the voice's fence, a kerb is a slab of its trim.
    rim = _ring(x0, z0, x1, z1) if min(w, d) >= 5 else []
    rail = b.joinery(b.voice, "fence")
    for (x, z) in rim:
        if dx is not None and abs(x - dx) <= 1 and abs(z - dz) <= 1:
            continue
        if (x, z) in walk and layout == "cross":
            continue
        if edge == "hedge":
            b.place_block(x, y, z, ground)
            b.place_block(x, y + 1, z, _BUSH)
        elif edge == "rail":
            b.place_block(x, y + 1, z, rail)
        else:
            b.place_block(x, y + 1, z, kerb)

    # The planting: flowers in the beds at a rhythm the seed sets, and a bush here and
    # there where there is room for one to be seen.
    rng.shuffle(beds)
    density = rng.choice([4, 5, 7])
    flowers = 0
    bushes = 0
    for i, (x, z) in enumerate(beds):
        if (x, z) in walk:
            continue
        on_rim = x in (x0, x1) or z in (z0, z1)
        if on_rim and rim:
            continue
        if i % density == 0:
            b.place_block(x, y + 1, z, rng.choice(_FLOWERS))
            flowers += 1
        elif min(w, d) >= 9 and i % 23 == 1:
            b.place_block(x, y + 1, z, _BUSH)
            bushes += 1

    # ...and a bench or a lamp on the path, where a person would stop.
    spots = [(x, z) for (x, z) in sorted(walk)
             if x0 + 1 < x < x1 - 1 and z0 + 1 < z < z1 - 1]
    rng.shuffle(spots)
    laid = 0
    for (x, z) in spots[:3]:
        if dx is not None and abs(x - dx) <= 2 and abs(z - dz) <= 2:
            continue
        got = b.fitting(rng.choice(["bench", "light", "table"]), x, y + 1, z,
                        rng.choice(["north", "south", "east", "west"]),
                        mat=b.voice["footing"], room="garden")
        laid += 1 if got.get("ok") else 0

    b.check_attached()
    return {"ok": True, "layout": layout, "edge": edge, "flowers": flowers,
            "bushes": bushes, "fittings": laid, "cells": w * d}
