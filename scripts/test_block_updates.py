"""Does doBlockUpdates=True make connective blocks join up?

A human walking the town found every fence standing as an isolated post. We write with
doBlockUpdates=False (world.editor, buildlib.flush) so bulk placement does not trigger
gravity and water flow -- but that is also the flag that computes north/south/east/west
on a fence, up/height on a wall, and shape on a stair.

Two candidate fixes, and they are not equivalent:
  A. re-place connective blocks with updates ON and let the game compute the state
  B. compute the state ourselves and write it explicitly

A is a few lines; B is deterministic and cannot set off a landslide. Which one is even
possible depends on whether the game recomputes a block's *own* shape when it is placed
with updates on, or only its neighbours'. That is a fact, not a matter of taste, so
this measures it instead of arguing about it.

    bash scripts/mcrun.sh scripts/test_block_updates.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gdpc import Block  # noqa: E402
from gdpc.interface import placeBlocks  # noqa: E402
from gdpc.vector_tools import Rect  # noqa: E402

from ethoslm import observe, world  # noqa: E402
from ethoslm.measure import record  # noqa: E402

X, Z = -1280, 880          # clear of both the town and the nav course
Y = 120
# A suite that writes blocks needs a backend; with none it says so and stops rather than
# ending in a connection traceback that reads like a broken suite.
_no_backend = world.serving()
if _no_backend:
    print(f"skipped: {_no_backend}")
    raise SystemExit(0)


ed = world.editor()
ed.loadWorldSlice(Rect((X - 8, Z - 8), (40, 40)), cache=True)


def build(x0, updates: bool):
    """A fence run between two stone blocks, and a stair corner, placed one way."""
    items = []
    for dx in range(0, 7):
        for dy in range(0, 4):
            items.append(((x0 + dx, Y + dy, Z), Block("minecraft:air")))
            items.append(((x0 + dx, Y + dy, Z + 1), Block("minecraft:air")))
    for dx in range(0, 7):
        items.append(((x0 + dx, Y - 1, Z), Block("minecraft:stone")))
        items.append(((x0 + dx, Y - 1, Z + 1), Block("minecraft:stone")))
    # stone post, three fences, stone post
    items.append(((x0, Y, Z), Block("minecraft:stone")))
    for dx in (1, 2, 3):
        items.append(((x0 + dx, Y, Z), Block("minecraft:oak_fence")))
    items.append(((x0 + 4, Y, Z), Block("minecraft:stone")))
    # an L of stairs, which should mitre into an outer corner at the bend
    items.append(((x0 + 1, Y, Z + 1),
                  Block("minecraft:stone_brick_stairs[facing=east,half=bottom]")))
    items.append(((x0 + 2, Y, Z + 1),
                  Block("minecraft:stone_brick_stairs[facing=east,half=bottom]")))
    items.append(((x0 + 3, Y, Z + 1),
                  Block("minecraft:stone_brick_stairs[facing=south,half=bottom]")))
    placeBlocks(items, doBlockUpdates=updates, host=world.HOST)


build(X, False)
build(X + 12, True)

ed2 = world.editor()
ed2.loadWorldSlice(Rect((X - 8, Z - 8), (40, 40)), cache=True)
vol = observe.Volume.from_world_slice(ed2.worldSlice, X - 8, Z - 8, 40, 40, Y - 4, Y + 6)

rows = {}
for label, x0 in (("doBlockUpdates=False", X), ("doBlockUpdates=True", X + 12)):
    fences = [vol.state(x0 + dx, Y, Z) for dx in (1, 2, 3)]
    stairs = [vol.state(x0 + dx, Y, Z + 1) for dx in (1, 2, 3)]
    joined = sum(1 for f in fences
                 if any(f"{s}=true" in f for s in ("north", "south", "east", "west")))
    mitred = sum(1 for s in stairs if "shape=straight" not in s)
    rows[label] = {"fences_joined": joined, "of": len(fences),
                   "stairs_mitred": mitred, "fence_states": fences,
                   "stair_states": stairs}
    print(f"\n{label}")
    print(f"  fences joined to a neighbour: {joined}/{len(fences)}")
    for f in fences:
        print(f"    {f}")
    print(f"  stairs with a corner shape:   {mitred}/{len(stairs)}")
    for s in stairs:
        print(f"    {s}")

# --- and the same thing through Builder.flush(), which is the path builds use -------
from ethoslm.buildlib import Builder  # noqa: E402

site = world.load_site(ed2, X - 8, Z - 8, 40, 40)
bl = Builder(site)
BX = X + 24
bl.place_cuboid(BX, Y - 1, Z, BX + 6, Y - 1, Z + 1, "stone")
bl.place_cuboid(BX, Y, Z, BX + 6, Y + 3, Z + 1, "air")
bl.place_block(BX, Y, Z, "stone")
for dx in (1, 2, 3):
    bl.place_block(BX + dx, Y, Z, "oak_fence")
bl.place_block(BX + 4, Y, Z, "stone")
res = bl.flush()

ed3 = world.editor()
ed3.loadWorldSlice(Rect((X - 8, Z - 8), (40, 40)), cache=True)
vol3 = observe.Volume.from_world_slice(ed3.worldSlice, X - 8, Z - 8, 40, 40, Y - 4, Y + 6)
built = [vol3.state(BX + dx, Y, Z) for dx in (1, 2, 3)]
joined = sum(1 for f in built
             if any(f"{s}=true" in f for s in ("north", "south", "east", "west")))
print(f"\nvia Builder.flush()  ({res['connective_rejoined']} connective blocks re-placed)")
print(f"  fences joined to a neighbour: {joined}/3")
for f in built:
    print(f"    {f}")
rows["Builder.flush()"] = {"fences_joined": joined, "of": 3, "stairs_mitred": 0}
assert joined == 3, "Builder.flush() still leaves fences unjoined"

a = rows["doBlockUpdates=False"]
b = rows["doBlockUpdates=True"]
print("\nverdict:")
if b["fences_joined"] > a["fences_joined"] or b["stairs_mitred"] > a["stairs_mitred"]:
    print("  updates ON fixes it -- fix A (re-place connective blocks with updates)")
else:
    print("  updates ON changes nothing: the game does not recompute a block's own")
    print("  shape on placement, so the states must be computed by us -- fix B")
record("block_update_test", **{k: {kk: vv for kk, vv in v.items()
                                  if not kk.endswith("states")}
                              for k, v in rows.items()})
