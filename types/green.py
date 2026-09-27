"""A green: common grass at a village's heart, kept open for everyone's use.

Not a garden (nothing is planted in beds) and not a square (nothing is paved): the
setting's own turf, one thing at its middle that the place gathers round -- a well, a
great tree or a standing cross -- a few spreading trees at its edges for shade, benches
facing in, and a trough where stock is watered. What makes it a green is the buildings
that face it, which is the composition's decision, not this form's.

Its turf is `voice["ground"]`, the surface `site()` read off the land beside the part, so
a green on a plain is grass; a voice whose ground is made (dressed stone) still has soil
here, as a grove does.
"""

import random

KIND = "area"

FORM = "civic"
#: What this is **for**: the common ground a village shares.
ROLE = "civic"
FUNCTION = "gather"

PARAMS = {
    "centre": ("choice", ["well", "tree", "cross"]),
    "trees": ("int", 0, 4),
    "benches": ("int", 0, 4),
    "trough": ("choice", ["yes", "no"]),
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

SOIL = ("grass_block", "dirt", "coarse_dirt", "podzol", "rooted_dirt", "moss_block",
        "mud", "mycelium")
_FLOWERS = ("dandelion", "poppy", "oxeye_daisy", "azure_bluet")


def _rect(part):
    fp = part.get("footprint")
    if fp is not None:
        return int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3])
    return int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"])


def _spreading(b, x, z, y, trunk, leaf, rng, big=False):
    """A spreading tree: a trunk and a layered crown that is widest low and every leaf
    of which touches the trunk or another leaf (nothing hangs)."""
    h = rng.randint(5, 6) if big else rng.randint(4, 5)
    for i in range(h):
        b.place_block(x, y + 1 + i, z, trunk)
    top = y + h
    layers = ((top - 1, 2 if big else 1), (top, 2), (top + 1, 1), (top + 2, 0))
    for (ly, r) in layers:
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if r == 2 and abs(dx) == 2 and abs(dz) == 2:
                    continue
                if dx == 0 and dz == 0 and ly <= top:
                    continue
                b.place_block(x + dx, ly, z + dz, leaf)
    return h


def _cross(b, x, z, y):
    """A standing cross on a stepped base: the market cross of a green."""
    step = b.block(b.voice["footing"])
    # the footing is a stone family, and every stone family has a wall shape (as the
    # hall's skirt and the yard's low wall rely on); a trim of timber may not
    post = b.block(b.voice["footing"], "wall")
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            b.place_block(x + dx, y + 1, z + dz, step)
    for i in range(2, 6):
        b.place_block(x, y + i, z, post)
    b.place_block(x, y + 6, z, post)
    b.place_block(x - 1, y + 5, z, post)
    b.place_block(x + 1, y + 5, z, post)


def build(b, part, seed, **params):
    rng = random.Random(seed)
    x0, z0, x1, z1 = _rect(part)
    y = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    centre = params.get("centre", "well")
    if centre not in ("well", "tree", "cross"):
        centre = "well"
    n_trees = int(params.get("trees", 2))
    n_bench = int(params.get("benches", 2))
    trough = params.get("trough", "yes") == "yes"

    ground = b.block(b.voice["ground"])
    if ground.split("[")[0] not in SOIL:
        ground = "grass_block"
    trunk = b.axial(b.block(b.voice["frame"], "post"), "y")
    leaf = b.foliage(b.voice["frame"]) + "[persistent=true]"
    if not any(k in trunk for k in ("_log", "_wood", "_stem", "_hyphae")):
        trunk = b.axial("stripped_oak_log", "y")
    b.fill_region(x0, y + 1, z0, x1, y + 10, z1, "air")
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b.place_block(x, y, z, ground)

    reserved = b.door_cell(x0, z0, x1, z1)
    keep = set()
    if reserved is not None:
        for t in range(-1, 2):
            keep.add((int(reserved[0]) + t, int(reserved[1])))
            keep.add((int(reserved[0]), int(reserved[1]) + t))
    stood = []
    small = min(w, d) < 9
    # the heart of the green
    if small:
        centre = "cross" if centre == "tree" else centre
    if centre == "well" and min(w, d) >= 7:
        got = b.fitting("well", cx, y + 1, cz, "north", mat=b.voice["footing"])
        if got.get("ok"):
            stood.append("well")
    elif centre == "tree" and min(w, d) >= 11:
        _spreading(b, cx, cz, y, trunk, leaf, rng, big=True)
        stood.append("tree")
    else:
        _cross(b, cx, cz, y)
        stood.append("cross")
    heart = (cx, cz)
    # shade trees near the edge, clear of the heart and each other, never on the way in
    spots = []
    cands = [(x, z) for x in range(x0 + 3, x1 - 2) for z in range(z0 + 3, z1 - 2)
             if min(x - x0, x1 - x, z - z0, z1 - z) <= 4]
    rng.shuffle(cands)
    for c in cands:
        if len(spots) >= n_trees:
            break
        if max(abs(c[0] - heart[0]), abs(c[1] - heart[1])) < 6:
            continue
        if any(max(abs(c[0] - s[0]), abs(c[1] - s[1])) < 7 for s in spots):
            continue
        if any(abs(c[0] - k[0]) <= 3 and abs(c[1] - k[1]) <= 3 for k in keep):
            continue
        spots.append(c)
    for (tx, tz) in spots:
        _spreading(b, tx, tz, y, trunk, leaf, rng)
    # benches facing the heart
    benches = 0
    ring = [(cx, cz - 4, "south"), (cx, cz + 4, "north"), (cx - 4, cz, "east"),
            (cx + 4, cz, "west")]
    rng.shuffle(ring)
    for (bx, bz, face) in ring:
        if benches >= n_bench or small:
            break
        if not (x0 + 1 <= bx <= x1 - 1 and z0 + 1 <= bz <= z1 - 1):
            continue
        if any(max(abs(bx - s[0]), abs(bz - s[1])) < 3 for s in spots):
            continue
        got = b.fitting("bench", bx, y + 1, bz, face, mat=b.voice["trim"])
        if got.get("ok"):
            benches += 1
    # a trough where the stock is watered, by an edge
    troughs = 0
    if trough and not small:
        for (tx, tz) in ((x0 + 2, cz), (x1 - 2, cz), (cx, z0 + 2), (cx, z1 - 2)):
            if any(max(abs(tx - s[0]), abs(tz - s[1])) < 3 for s in spots):
                continue
            if any(abs(tx - k[0]) <= 2 and abs(tz - k[1]) <= 2 for k in keep):
                continue
            got = b.fitting("trough", tx, y + 1, tz, "north", mat=b.voice["footing"])
            if got.get("ok"):
                troughs = 1
                break
    # flowers in the turf, sparsely
    for x in range(x0 + 1, x1):
        for z in range(z0 + 1, z1):
            if (x * 5 + z * 11 + seed) % 41:
                continue
            if b.get_block(x, y + 1, z).split("[")[0] != "air":
                continue
            b.place_block(x, y + 1, z, rng.choice(_FLOWERS))
    b.check_attached()
    b.area_way_in(part["x0"], part["z0"], part["x1"], part["z1"], int(part["floor_y"]))
    return {"ok": True, "centre": stood, "trees": len(spots), "benches": benches,
            "trough": troughs, "cells": w * d}
