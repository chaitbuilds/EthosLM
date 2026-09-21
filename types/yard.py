"""A yard: worked open ground behind or beside the houses, fenced and used.

Between a garden and an alley. A garden is planted and an alley is paved through; a yard
is where the work of a quarter happens outdoors -- a beaten floor, a fence round it with
the way in left open, a stack of something in a corner and a trough by the gate.

Its floor is the setting's own ground swept to earth in patches (`voice["ground"]`) with
a hard standing of the voice's footing where things are stacked, so a yard on a plain
reads as trodden grass and one on a shore as sand.
"""

import random

KIND = "area"

FORM = "civic"
#: What this building is **for**. `urban`: a yard is what a street quarter does with its
#: back ground. A farm's equivalent is a field and the farm belt has one.
ROLE = "urban"

PARAMS = {
    "fence": ("choice", ["rail", "wall", "open"]),
    "standing": ("choice", ["corner", "strip", "full"]),
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


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    fence = params.get("fence", "rail")
    if fence not in ("rail", "wall", "open"):
        fence = "rail"
    standing = params.get("standing", "corner")
    if standing not in ("corner", "strip", "full"):
        standing = "corner"
    if min(w, d) < 5:
        fence = "open"

    ground = b.block(b.voice["ground"])
    hard = b.block(b.voice["footing"])
    b.fill_region(x0, y + 1, z0, x1, y + 4, z1, "air")

    half_w = max(1, w // 3)
    half_d = max(1, d // 3)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if standing == "full":
                paved = True
            elif standing == "strip":
                paved = (x - x0) < half_w
            else:
                paved = (x - x0) < half_w and (z - z0) < half_d
            b.place_block(x, y, z, hard if paved else ground)
            b.place_block(x, y + 1, z, "air")
            b.place_block(x, y + 2, z, "air")

    # **The rim cell where the way in is**, from the library: the reserved doorway where
    # the part has one and the threshold on the network where it does not, snapped to
    # this rectangle's own perimeter. A border closed over the way in is E008 and it is
    # what the first occupancy run found in this type.
    way = b.door_cell(x0, z0, x1, z1)
    dx, dz = way if way else (None, None)
    if fence != "open":
        post = b.block(b.voice["frame"], "post")
        rail = b.joinery(b.voice, "fence")
        low = b.block(b.voice["footing"], "wall")
        for (x, z) in _ring(x0, z0, x1, z1):
            if dx is not None and abs(x - dx) <= 1 and abs(z - dz) <= 1:
                continue
            corner = x in (x0, x1) and z in (z0, z1)
            if fence == "rail":
                b.place_block(x, y + 1, z, post if corner else rail)
                if corner:
                    b.place_block(x, y + 2, z, post)
            else:
                b.place_block(x, y + 1, z, low)
                if corner:
                    b.place_block(x, y + 2, z, post)
    if dx is not None:
        for yy in (y + 1, y + 2):
            b.place_block(dx, yy, dz, "air")

    # What the yard is for: the gear, on the hard standing, clear of the way in.
    spots = [(x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, z1)]
    rng.shuffle(spots)
    want = 2 if min(w, d) < 9 else (4 if min(w, d) < 16 else 6)
    laid, taken = 0, []
    for (x, z) in spots:
        if laid >= want:
            break
        if dx is not None and abs(x - dx) <= 2 and abs(z - dz) <= 2:
            continue
        if any(abs(x - ox) < 3 and abs(z - oz) < 3 for (ox, oz) in taken):
            continue
        kind = rng.choice(["store", "fodder", "trough", "workbench", "light", "table"])
        got = b.fitting(kind, x, y + 1, z, rng.choice(_FACING),
                        mat=b.voice["trim"] if kind in ("store", "shelf", "table")
                        else b.voice["footing"],
                        extent=2 if kind == "store" and min(w, d) >= 9 else 1,
                        room="yard")
        if got.get("ok"):
            taken.append((x, z))
            laid += 1

    b.check_attached()
    # the reserved doorway stays walkable, whatever the floor and the border did
    b.area_way_in(part["x0"], part["z0"], part["x1"], part["z1"], int(part["floor_y"]))
    return {"ok": True, "fence": fence, "standing": standing, "gear": laid,
            "cells": w * d}
