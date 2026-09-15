"""Smoke test: build one box per roof style/pitch so the primitives can be eyeballed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import world
from ethoslm.buildlib import Builder
from ethoslm.measure import record

X, Z = 600, 600            # unused corner of the world
# A suite that writes blocks needs a backend; with none it says so and stops rather than
# ending in a connection traceback that reads like a broken suite.
_no_backend = world.serving()
if _no_backend:
    print(f"skipped: {_no_backend}")
    raise SystemExit(0)

ed = world.editor()
for ox in (-64, 0, 64, 128):
    for oz in (-64, 0, 64):
        world.load_site(ed, X + ox, Z + oz, 64, 64)
site = world.load_site(ed, X - 64, Z - 64, 256, 192)
b = Builder(site)

CASES = [
    ("gable  (1,1)", "gable", "z", (1, 1), "spruce"),
    ("gable  (1,2)", "gable", "z", (1, 2), "brick"),
    ("gable  (2,1)", "gable", "z", (2, 1), "deepslate_tile"),
    ("hip    (1,2)", "hip", "z", (1, 2), "sandstone"),
    ("gambrel", "gambrel", "z", (1, 1), "dark_oak"),
    ("mansard", "mansard", "z", (1, 1), "oxidized_copper"),
    ("shed south", "shed", "s", (1, 2), "cherry"),
    ("flat parapet", "flat", "z", (1, 1), "quartz"),
]

W, D, H = 11, 13, 5
GAP = 6
y = 70
for i, (label, style, axis, pitch, mat) in enumerate(CASES):
    x0 = X + i * (W + GAP)
    z0 = Z
    b.place_cuboid(x0 - 2, y - 6, z0 - 2, x0 + W + 1, y - 1, z0 + D + 1, "stone_bricks")
    b.place_cuboid(x0, y, z0, x0 + W, y + H, z0 + D, "air")
    # walls with depth: posts, base course, top band
    for (a0, c0, a1, c1) in ((x0, z0, x0 + W, z0), (x0, z0 + D, x0 + W, z0 + D),
                             (x0, z0, x0, z0 + D), (x0 + W, z0, x0 + W, z0 + D)):
        b.wall(a0, y, c0, a1, y + H - 1, c1, "white_terracotta",
               post="stripped_oak_log[axis=y]", spacing=5,
               base="cobblestone", base_height=2, band="oak_planks")
    ridge = b.roof(x0, z0, x0 + W, z0 + D, y + H, mat, style=style, axis=axis,
                   pitch=pitch, overhang=1)
    print(f"{label:14} ridge={ridge} height={ridge - (y + H)}")

# round tower with a proper cone
tcx, tcz = X + 30, Z + 60
b.cylinder(tcx, y, tcz, 5, 18, "stone_bricks", hollow=True)
b.roof_cone(tcx, y + 18, tcz, 7, "deepslate_tile", pitch=(3, 2))

# a path of even width
pts = [(X - 10, Z + 60), (X + 5, Z + 70), (X + 20, Z + 62), (X + 40, Z + 75), (X + 70, Z + 66)]
b.path(pts, 3, "gravel")

res = b.flush()
record("prims_smoke_test", placed=res["placed"], failed=res["failed"],
       errors=res["errors"], palette_size=res["palette_size"])
print(res["placed"], "placed,", res["failed"], "failed", res["errors"])
print("CENTRE", X + 4 * (W + GAP), y + 8, Z + D // 2)
