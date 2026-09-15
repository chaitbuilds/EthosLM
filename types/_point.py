"""The reference point type: the least a gate can be, written by hand.

A point is a cell and a facing -- where the plan says a thing goes and which way it is
turned -- and the one property everything downstream depends on is that the thing built
there is turned the way the plan turned it. So this builds the smallest object that has
an orientation you can read off the blocks: two posts across the facing axis, a lintel
over them, and the way through left open along it.

`passage=True` says a network may cross the wall here. It is a declaration of the type,
like `FORM` and `PARAMS`, because whether a part is a way through is a fact about what
it is and not about where it was put.
"""

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

KIND = "point"

PARAMS = {"height": ("int", 2, 5)}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3 and up measured: a post, a lintel and a light. Measured by
    # scripts/type_needs.py; the band is rounds/type-needs.json. The pair is the pad
    # site() hands build(), after siting's inset.
    "footprint": (3, 3, 16, 16),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

PASSAGE = True

_AHEAD = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def build(b, part, seed, height=3):
    post = b.block(part["voice"]["frame"], "post")
    lintel = b.block(part["voice"]["trim"])
    x, z = part["at"]
    y = part["floor_y"]
    ahead = _AHEAD[part["facing"]]
    across = (ahead[1], ahead[0])
    for side in (-1, 1):
        px, pz = x + across[0] * side, z + across[1] * side
        for dy in range(1, int(height) + 1):
            b.place_block(px, y + dy, pz, post)
    for side in (-1, 0, 1):
        b.place_block(x + across[0] * side, y + int(height) + 1,
                      z + across[1] * side, lintel)
    return {"ok": True, "cells": 2 * int(height) + 3,
            "reason": f"two posts across {part['facing']} with a lintel over them and "
                      f"the way through open along it"}
