"""Finish the seam between what was built and what was there.

    ETHOSLM_SETTLEMENT=<name> bash scripts/mcrun.sh scripts/finish_run.py

Runs last, after every structure stands. Rounds the lip of each cut face, fills the foot
of each fill face, and dresses the ground it touched -- bounded so it finishes edges
rather than regrading hills. Then lints, because a pass that moves ground can sever a
network like any other, and E009 exists for exactly that.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np  # noqa: E402
from gdpc.vector_tools import Rect  # noqa: E402

from ethoslm import finish, lint, observe, registry, settlement, world  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.measure import record  # noqa: E402

PAD = 24
s = settlement.site_info()
X, Z = s["origin"]
S = s["size"]
net = settlement.load_network()
plots = settlement.load_plots()

t0 = time.perf_counter()
ed = world.editor()
site = world.load_site(ed, X - PAD, Z - PAD, S + 2 * PAD, S + 2 * PAD)
hm = site.heights.astype(int) - 1
vol0 = observe.Volume.from_world_slice(ed.worldSlice, site.x, site.z, site.sx, site.sz,
                                       max(0, int(hm.min()) - 20), int(hm.max()) + 8)
ground, wet = observe.ground_heights(vol0)
nav0 = observe.Nav(vol0)
snap = lint.snapshot(nav0, net) if net else None
load_s = time.perf_counter() - t0
print(f"site loaded in {load_s:.1f}s; {len(plots)} plots, "
      f"{len(net.cells) if net else 0} lane cells")

# What we built and therefore own the edge of: the lane at its own height, and each
# plot's perimeter at the height it was graded to.
built = {}
if net:
    for (x, z), rec in net.cells.items():
        built[(x, z)] = rec["y"]
for p in plots:
    for x in range(p["x0"], p["x1"] + 1):
        for z in (p["z0"], p["z1"]):
            built[(x, z)] = int(ground[x - site.x, z - site.z])
    for z in range(p["z0"], p["z1"] + 1):
        for x in (p["x0"], p["x1"]):
            built[(x, z)] = int(ground[x - site.x, z - site.z])

# Keep off anything a person arrives on. The first run of this filled the doorstep of
# croft_north by three blocks and E002 caught it: "no standable threshold in the doorway
# at (1540,68,1278)". A doorway sits on a plot boundary, which is exactly where a seam
# band starts, so the band has to know where the doors are. Read from the world, not
# from the network, so it holds for a build with no circulation pass behind it.
keep_out = list(plots)
doors = vol0.find(lambda st: st.split("[")[0].endswith("_door")
                  or st.split("[")[0].endswith("_fence_gate"))
guarded = set()
for (x, _, z) in doors:
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            guarded.add((x + dx, z + dz))
if net:
    for t in net.thresholds:
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                guarded.add((t.x + dx, t.z + dz))
                guarded.add((t.door[0] + dx, t.door[2] + dz))

# Keep off the ground a tread was justified against, for the same reason. A stair's
# facing is decided from the finished ground on both sides of it; move that ground
# afterwards and a correct tread is suddenly facing down-slope. Only treads on the
# ground count -- a roof stair sits many blocks above its column, and guarding those
# would keep the seam pass off every eaves line in the town.
stairs = vol0.find(lambda st: st.split("[")[0].endswith("_stairs"))
on_ground = 0
for (x, y, z) in stairs:
    gx, gz = x - site.x, z - site.z
    if not (0 <= gx < ground.shape[0] and 0 <= gz < ground.shape[1]):
        continue
    if abs(y - int(ground[gx, gz])) > 1:
        continue
    on_ground += 1
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            guarded.add((x + dx, z + dz))

# Guard the path, not just the doorstep. The door at (1581,105,-1468) is served by a
# 48-column approach() path, and this pass dressed the ground along it. Two columns
# round the door was never the right radius -- the thing that has to be preserved is the
# whole flight that makes the door walkable, and the library now writes down every
# column it laid (settlement.load_paths). A dressed approach path and an undressed one
# are indistinguishable in the blocks, so this cannot be recovered by looking.
ways = settlement.load_paths()
way_cols = settlement.path_columns(ways)
guarded |= way_cols

keep_out += [{"x0": x, "z0": z, "x1": x, "z1": z} for (x, z) in sorted(guarded)]
print(f"keeping off {len(guarded)} columns around {len(doors)} doors, "
      f"{len(net.thresholds) if net else 0} thresholds and {on_ground} treads on the "
      f"ground (of {len(stairs)} stairs), including {len(way_cols)} columns of "
      f"approach and flight laid by {len(ways)} recorded ways in")

t0 = time.perf_counter()
targets = finish.plan_finish(built, ground, site.x, site.z, keep_out=keep_out)
plan_s = time.perf_counter() - t0
if targets:
    moves = [abs(y - int(ground[x - site.x, z - site.z])) for (x, z), y in targets.items()]
    print(f"planned in {plan_s:.1f}s: {len(targets)} columns to move, "
          f"{sum(moves)} blocks of movement, worst {max(moves)}")
else:
    print("nothing to finish")
    sys.exit(0)

b = Builder(site)
stats = finish.apply(b, targets)
bad = registry.check_all(sorted({v for v in b._pending.values()}))
if bad:
    print(f"!! invalid block states, nothing written: {bad}")
    sys.exit(2)
res = b.flush()
print(f"finish: {stats}")
print(f"placed {res['placed']} failed {res['failed']} in {res['seconds']}s")

# --- did moving the ground break anything -----------------------------------------
t0 = time.perf_counter()
ed2 = world.editor()
ed2.loadWorldSlice(Rect((X - PAD, Z - PAD), (S + 2 * PAD, S + 2 * PAD)), cache=True)
h = ed2.worldSlice.heightmaps[world.HEIGHTMAP].astype(int) - 1
inner = h[PAD:PAD + S, PAD:PAD + S]
vol = observe.Volume.from_world_slice(ed2.worldSlice, X - PAD, Z - PAD,
                                      S + 2 * PAD, S + 2 * PAD,
                                      max(0, int(inner.min()) - 8), int(h.max()) + 40)
ctx = lint.Context.build(vol, plots, network=net, before=snap,
                         region=(X, Z, X + S - 1, Z + S - 1))
report = lint.lint(ctx)
print(f"\n(lint in {time.perf_counter() - t0:.1f}s)")
print(report.summary())
pieces = ctx.lane_pieces()
print(f"lane in {len(pieces)} walkable piece(s)")

json.dump({"stats": stats, "columns": len(targets), "lint": report.to_json(),
           "pieces": [len(p) for p in pieces],
           "ways_in_guarded": {"rows": len(ways), "columns": len(way_cols),
                               "moved": sorted(set(targets) & way_cols)},
           "e002_seeds": ctx.outdoor_seeds},
          open(os.path.join(settlement.STATE, "finish.json"), "w"), indent=1)
record("finish_pass", name=settlement.NAME, **stats,
       errors=len(report.errors), warnings=len(report.warnings),
       counts=report.to_json()["counts"], pieces=len(pieces))
