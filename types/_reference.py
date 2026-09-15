"""The reference type: the least a type can be, written by hand, not by a model.

Every defect this project has found about the ground was found by a model-authored type
on a new site, at a session and a million tokens apiece, and each time the argument had
to be made all over again that the fault was the library's and not the type's. This file
settles that argument in advance. When an instance of this is broken, the library is
broken.

`scripts/test_ground.py` stands it on all 36 fixtures of `rounds/terrain-bank.json`,
at two seeds and one and two storeys.
"""

FORM = "european_vernacular"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "civic"

PARAMS = {"storeys": ("int", 1, 2)}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 8x8 and up measured: two storeys need an inside a flight fits in, and it refuses
    # smaller by name. Measured by scripts/type_needs.py; the band is rounds/type-
    # needs.json. The pair is the pad site() hands build(), after siting's inset.
    "footprint": (8, 8, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}


def build(b, part, seed, storeys=1):
    x0, z0, x1, z1 = part["footprint"]
    res = b.building(part["label"], x0, z0, x1, z1, storeys, "gable",
                     mat=part["voice"])
    if not res["ok"]:
        return res
    fx0, y, fz0, fx1, fz1 = res["rooms"][0]
    b.fitting("light", (fx0 + fx1) // 2, y + 1, (fz0 + fz1) // 2)
    return res
