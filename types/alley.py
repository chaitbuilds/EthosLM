"""An alley: a narrow paved cut between rows of houses, with a drain down it.

What a dense quarter has instead of gardens. A district of plots five apart has lanes
between them because the circulation puts them there; an alley is the ground a planner
*means* to leave between two rows -- paved, kerbed, lit at its ends, with the gutter and
the barrels and the woodpile that make a back street read as a back street rather than
as a gap.

Narrow by intent and square by accident: the pad it is given may be either, and what it
does is run its gutter down the long axis whichever that is.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**. `urban`: an alley is a street thing, and a farm belt
#: that asked for one would be asking for a street.
ROLE = "urban"

PARAMS = {
    "gutter": ("choice", ["channel", "kerbed", "none"]),
    "clutter": ("choice", ["none", "some", "lived_in"]),
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


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    along_x = w >= d
    gutter = params.get("gutter", "channel")
    if gutter not in ("channel", "kerbed", "none"):
        gutter = "channel"
    clutter = params.get("clutter", "some")
    if clutter not in ("none", "some", "lived_in"):
        clutter = "some"
    if min(w, d) < 3:
        gutter = "none"

    paving = b.block(b.voice["footing"])
    band = b.block(b.voice["trim"])
    kerb = b.block(b.voice["trim"], "slab")
    b.fill_region(x0, y + 1, z0, x1, y + 4, z1, "air")

    mid = ((z0 + z1) // 2) if along_x else ((x0 + x1) // 2)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            on_mid = (z == mid) if along_x else (x == mid)
            b.place_block(x, y, z, band if (on_mid and gutter != "none") else paving)
            b.place_block(x, y + 1, z, "air")
            b.place_block(x, y + 2, z, "air")

    # A kerbed gutter has a slab either side of its channel, which is what makes a paved
    # strip read as a street rather than as a floor.
    if gutter == "kerbed" and min(w, d) >= 5:
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                off = abs((z - mid) if along_x else (x - mid))
                if off == 1:
                    b.place_block(x, y + 1, z, kerb)

    # **The rim cell where the way in is**, from the library: the reserved doorway where
    # the part has one and the threshold on the network where it does not, snapped to
    # this rectangle's own perimeter. A border closed over the way in is E008 and it is
    # what the first occupancy run found in this type.
    way = b.door_cell(x0, z0, x1, z1)
    dx, dz = way if way else (None, None)
    if dx is not None:
        for yy in (y + 1, y + 2):
            b.place_block(dx, yy, dz, "air")

    # A lamp at each end of the run, on the wall side, and the clutter of a back street
    # where a barrel or a woodpile would actually be left.
    ends = ([(x0 + 1, mid), (x1 - 1, mid)] if along_x
            else [(mid, z0 + 1), (mid, z1 - 1)])
    lit = 0
    if min(w, d) >= 3:
        for (x, z) in ends:
            if not (x0 <= x <= x1 and z0 <= z <= z1):
                continue
            if dx is not None and abs(x - dx) <= 1 and abs(z - dz) <= 1:
                continue
            got = b.fitting("light", x, y + 1, z, rng.choice(_FACING),
                            mat=b.voice["footing"], room="street")
            lit += 1 if got.get("ok") else 0

    laid = 0
    if clutter != "none" and min(w, d) >= 4:
        want = 2 if clutter == "some" else 5
        side = [(x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, z1)
                if (abs((z - mid) if along_x else (x - mid)) >= 2)]
        rng.shuffle(side)
        for (x, z) in side:
            if laid >= want:
                break
            if dx is not None and abs(x - dx) <= 1 and abs(z - dz) <= 1:
                continue
            got = b.fitting(rng.choice(["store", "fodder", "trough", "shelf"]),
                            x, y + 1, z, rng.choice(_FACING),
                            mat=b.voice["trim"], room="street")
            laid += 1 if got.get("ok") else 0

    b.check_attached()
    return {"ok": True, "gutter": gutter, "clutter": clutter, "lamps": lit,
            "goods": laid, "axis": "x" if along_x else "z", "cells": w * d}
