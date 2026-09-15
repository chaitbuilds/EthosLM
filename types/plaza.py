"""A paved open space: the ground a place leaves unbuilt on purpose.

A square is a market -- a well, four booths, goods on tables. A plaza is the other
thing a place does with ground it does not build on: it paves it, edges it, lights it
and leaves it open. At a street corner it is a widening with a lamp in it; at the head
of an avenue it is the forecourt of whatever stands behind it.

Every material is a role on `b.voice`, so it is the same plaza in any palette, and the
one block that is not is the setting's own ground, which `site()` writes as
`voice["ground"]` -- the verge round a paved court is the plain the city stands on.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `civic` is admitted in a district of any role: a place paves open ground in its
#: farmland as readily as in its streets.
ROLE = "civic"

PARAMS = {
    "paving": ("choice", ["banded", "framed", "quartered"]),
    "edge": ("choice", ["kerb", "verge", "bollards"]),
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

_FACING = ("north", "south", "east", "west")


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


def _pave(b, x0, z0, x1, z1, y, style, rng):
    """The floor of the plaza: the voice's floor, banded or framed in its trim."""
    floor = b.block(b.voice["floor"])
    trim = b.block(b.voice["trim"])
    foot = b.block(b.voice["footing"])
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if style == "banded":
                block = trim if (x - x0) % 4 == 0 else floor
            elif style == "quartered":
                block = trim if (x == cx or z == cz) else floor
            else:                                     # framed
                near = min(x - x0, x1 - x, z - z0, z1 - z)
                block = trim if near == 1 else floor
            b.place_block(x, y, z, foot if (x + z) % 17 == 0 else block)
            b.place_block(x, y + 1, z, "air")
            b.place_block(x, y + 2, z, "air")


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    style = params.get("paving", "banded")
    if style not in ("banded", "framed", "quartered"):
        style = "banded"
    edge = params.get("edge", "kerb")
    if edge not in ("kerb", "verge", "bollards"):
        edge = "kerb"
    if style == "quartered" and min(w, d) < 7:
        style = "banded"
    if style == "framed" and min(w, d) < 5:
        style = "banded"

    b.fill_region(x0, y + 1, z0, x1, y + 4, z1, "air")
    _pave(b, x0, z0, x1, z1, y, style, rng)

    # The edge, which is where a plaza stops being a plaza. A kerb is a course of trim
    # slab; a verge is the setting's own ground left round the paving; bollards are
    # posts of the frame timber at a rhythm, with the way in left between them. **The
    # rim cell where the way in is**, from the library: the reserved doorway where the
    # part has one and the threshold on the network where it does not, snapped to this
    # rectangle's own perimeter. A border closed over the way in is E008 and it is what
    # the first occupancy run found in this type.
    way = b.door_cell(x0, z0, x1, z1)
    dx, dz = way if way else (None, None)
    if min(w, d) >= 5:
        ring = _ring(x0, z0, x1, z1)
        if edge == "verge":
            ground = b.block(b.voice["ground"])
            for (x, z) in ring:
                b.place_block(x, y, z, ground)
        elif edge == "kerb":
            kerb = b.block(b.voice["trim"], "slab")
            for (x, z) in ring:
                b.place_block(x, y + 1, z, kerb)
        else:
            post = b.block(b.voice["frame"], "post")
            for (x, z) in ring:
                if (x - x0) % 5 or (z - z0) % 5:
                    continue
                if dx is not None and abs(x - dx) <= 1 and abs(z - dz) <= 1:
                    continue
                b.place_block(x, y + 1, z, post)

    # ...and the way in stays open whatever the edge is.
    if dx is not None:
        for yy in (y + 1, y + 2):
            b.place_block(dx, yy, dz, "air")

    # A lamp at each corner a lamp fits at, and a bench or two out of the crossing.
    lit = 0
    if min(w, d) >= 4:
        for (x, z) in ((x0 + 1, z0 + 1), (x1 - 1, z0 + 1),
                       (x0 + 1, z1 - 1), (x1 - 1, z1 - 1)):
            got = b.fitting("light", x, y + 1, z, rng.choice(_FACING),
                            mat=b.voice["footing"], room="plaza")
            lit += 1 if got.get("ok") else 0

    seats = [(x, z) for x in range(x0 + 2, x1 - 1) for z in range(z0 + 2, z1 - 1)
             if (x - x0) % 6 == 3 and (z - z0) % 6 == 3]
    rng.shuffle(seats)
    laid = 0
    for (x, z) in seats[:4]:
        if dx is not None and abs(x - dx) <= 2 and abs(z - dz) <= 2:
            continue
        got = b.fitting(rng.choice(["bench", "table", "trough"]), x, y + 1, z,
                        rng.choice(_FACING), mat=b.voice["footing"], room="plaza")
        laid += 1 if got.get("ok") else 0

    b.check_attached()
    return {"ok": True, "paving": style, "edge": edge, "lamps": lit, "seats": laid,
            "cells": w * d}
