"""Is roof stair facing inverted? Build the same roof both ways and look at it.

Left  (west): what prims.roof() currently emits.
Right (east): the same profile with the facing flipped.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import world
from ethoslm.buildlib import Builder
from ethoslm.prims import material

X, Y, Z = 700, 92, 700
# A suite that writes blocks needs a backend; with none it says so and stops rather than
# ending in a connection traceback that reads like a broken suite.
_no_backend = world.serving()
if _no_backend:
    print(f"skipped: {_no_backend}")
    raise SystemExit(0)

ed = world.editor()
for ox in (-64, 0, 64):
    for oz in (-64, 0, 64):
        world.load_site(ed, X + ox - 32, Z + oz - 32, 64, 64)
site = world.load_site(ed, X - 96, Z - 96, 224, 224)
b = Builder(site)

W, D = 11, 13
MAT = "deepslate_tile"
full, stairs, slab = material(MAT)

# clean platform in the sky so nothing but the roofs is in frame
b.place_cuboid(X - 6, Y - 1, Z - 6, X + 40, Y - 1, Z + D + 6, "smooth_stone")
b.place_cuboid(X - 6, Y, Z - 6, X + 40, Y + 14, Z + D + 6, "air")

# --- A: current library output ---------------------------------------------
b.place_cuboid(X, Y, Z, X + W, Y + 3, Z + D, "white_concrete")
b.place_cuboid(X + 1, Y, Z + 1, X + W - 1, Y + 3, Z + D - 1, "air")
b.roof(X, Z, X + W, Z + D, Y + 4, MAT, style="gable", axis="z", pitch=(1, 1), overhang=1)

# --- B: identical, stair facing flipped ------------------------------------
OX = X + 22
b.place_cuboid(OX, Y, Z, OX + W, Y + 3, Z + D, "white_concrete")
b.place_cuboid(OX + 1, Y, Z + 1, OX + W - 1, Y + 3, Z + D - 1, "air")
rz0, rz1 = Z - 1, Z + D + 1
half = (rz1 - rz0 + 2) // 2
for x in range(OX - 1, OX + W + 2):
    for z in range(rz0, rz1 + 1):
        dz = min(z - rz0, rz1 - z)
        h = min(dz, half - 1)
        # flipped: tall side toward the ridge
        face = "south" if (z - rz0) <= (rz1 - z) else "north"
        b.place_block(x, Y + 4 + h, z, f"{stairs}[facing={face},half=bottom]")
        for yy in range(Y + 4, Y + 4 + h):
            b.place_block(x, yy, z, full)

res = b.flush()
print("placed", res["placed"], "failed", res["failed"], res["errors"])
print(f"CENTRE {X + 17} {Y + 4} {Z + D // 2}")
