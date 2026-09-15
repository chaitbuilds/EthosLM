"""The reference edge type: the least a wall can be, written by hand, not by a model.

An edge is a polyline with a width; `site()` grades it segment by segment and hands back
each segment with its own `floor_y` and its own columns. This file stands a course on
each of them and does nothing else -- no massing, no vocabulary, no decision. When an
instance of this has a gap at a vertex or a block hanging over nothing, the library is
what put it there.
"""

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

KIND = "edge"

PARAMS = {"height": ("int", 2, 5)}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # Width 1 to 3 and segments of 4 to 64 measured. Measured by scripts/type_needs.py;
    # the band is rounds/type-needs.json. The pair is (width, run of the longest
    # segment).
    "footprint": (1, 4, 3, 128),
    "frontage": "any",
    "ground": "any",
    "clearance": 0,
}


def build(b, part, seed, height=3):
    course = b.block(part["voice"]["wall"])
    laid = 0
    for seg in part["segments"]:
        y = seg["floor_y"]
        for (x, z) in seg["cells"]:
            for dy in range(1, int(height) + 1):
                b.place_block(x, y + dy, z, course)
            laid += 1
    return {"ok": True, "cells": laid,
            "reason": f"{len(part['segments'])} segment(s), {laid} columns of wall "
                      f"{height} high"}
