"""The reference area type: the least a square can be, written by hand.

An area is all ground: `site()` brings the rectangle to one level and joins it to the
lane, and what a type does with it is paving and the one thing that says a square is a
square. This lays the paving at the level the library left and puts a well in the middle
through `fitting()`, and decides nothing else.
"""

FORM = "civic"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "civic"

KIND = "area"

PARAMS = {"inset": ("int", 0, 2)}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3x3 and up measured: it paves what it is given. Measured by scripts/type_needs.py;
    # the band is rounds/type-needs.json. The pair is the pad site() hands build(),
    # after siting's inset.
    "footprint": (3, 3, 48, 48),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}


def build(b, part, seed, inset=0):
    paving = b.block(part["voice"]["floor"])
    x0, z0, x1, z1 = part["footprint"]
    y = part["floor_y"]
    x0, z0, x1, z1 = x0 + inset, z0 + inset, x1 - inset, z1 - inset
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b.place_block(x, y, z, paving)
    well = b.fitting("well", (x0 + x1) // 2, y + 1, (z0 + z1) // 2)
    return {"ok": True, "cells": (x1 - x0 + 1) * (z1 - z0 + 1),
            "reason": f"paved to y={y}; well {well['reason']}"}
