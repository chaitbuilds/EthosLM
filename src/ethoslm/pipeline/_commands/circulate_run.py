"Build the settlement's circulation, before any structure stands on it.\n\n    ETHOSLM_SETTLEMENT=<name> bash scripts/mcrun.sh scripts/circulate_run.py [material]\n\nReads the plan's structure footprints, routes a walkable network between them, builds it\n-- cut, fill, stairs, retaining, rails where there is a drop -- reserves one threshold\nper structure, and then lints what it built on its own terms:\n\nNothing later can be judged until this passes, which is the point of running it first."
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np  # noqa: E402
from gdpc.vector_tools import Rect  # noqa: E402

from ethoslm import (circulate, lint, observe, pipeline, prims,  # noqa: E402
                   registry, settlement, world)
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.measure import record  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _type_declares_passage(name: str) -> bool:
    """Does the committed type called `name` say a network may cross it? A5."""
    p = os.path.join(ROOT, "types", f"{name}.py")
    try:
        return bool(pipeline.load_type(p)["passage"]) if os.path.exists(p) else False
    except Exception:                        # noqa: BLE001 -- a bad type is not a gate
        return False

MAT = sys.argv[1] if len(sys.argv) > 1 else None
PAD = 24                     # how far outside the site the lane may run

s = settlement.site_info()
X, Z = s["origin"]
S = s["size"]
plan = json.load(open(os.path.join(settlement.STATE, "plan.json")))
# A4/A5: the plan is a tree and a place has four kinds of part in it. `plan_parts`
# flattens it (an old flat plan reads as a list of plots, so every standing round routes
# the same way), and `parts_to_routing` says what each kind means to a network: a plot
# and an area are sites, an edge is an obstacle, and a point whose type declares
# `PASSAGE` is the one crossing of it.
_parts = pipeline.plan_parts(plan)
_passage = {n for n in {p.get("type") for p in _parts if p.get("type")}
            if _type_declares_passage(n)}
_routing = circulate.parts_to_routing(_parts, passage=_passage)
sites = _routing["sites"]
mat = MAT or plan.get("circulation_material") or "cobblestone"
if mat not in prims.MATERIALS:
    print(f"!! unknown material family {mat!r}, using cobblestone")
    mat = "cobblestone"

# ------------------------------------------------------------------ the ground
t0 = time.perf_counter()
ed = world.editor()
site = world.load_site(ed, X - PAD, Z - PAD, S + 2 * PAD, S + 2 * PAD)
hm = site.heights.astype(int) - 1
# Route on the ground, not on the canopy. Minecraft's heightmaps count tree trunks, so
# routing on them lays lanes over the forest; observe.ground_heights ignores vegetation
# and takes the water surface where a column is flooded.
vol0 = observe.Volume.from_world_slice(ed.worldSlice, site.x, site.z, site.sx, site.sz,
                                       max(0, int(hm.min()) - 20), int(hm.max()) + 8)
heights, wet = observe.ground_heights(vol0)
load_s = time.perf_counter() - t0
print(f"site {site.sx}x{site.sz} loaded in {load_s:.1f}s  {site.stats()}")
print(f"ground vs heightmap: {int((hm - heights).max())} blocks of vegetation at most, "
      f"{int(wet.sum())} columns under water")

t0 = time.perf_counter()
# Water is expensive to cross and a wall may not be crossed at all -- except at a gate,
# which `passable` carries through everything else. A5.
_avoid = np.where(wet, 40.0, 0.0)
# ...and, on a designed place, the band at the foot of every terrace
from ethoslm.pipeline.stages_plan import type_declarations as _type_declarations  # noqa: E402
_avoid = np.maximum(_avoid, circulate.designed_ground_costs(
    {**plan, "parts": parts}, _type_declarations(parts), heights.shape, site.x, site.z))
for (_wx, _wz) in _routing["obstacles"]:
    _i, _j = _wx - site.x, _wz - site.z
    if 0 <= _i < _avoid.shape[0] and 0 <= _j < _avoid.shape[1]:
        _avoid[_i, _j] = np.inf
# A4: **the arterials go down first.** They were routed at the place level, before any
# district was planned, and every district was drawn against them; laying them now and
# handing them to the router as ground it prefers is what makes the lanes join the road
# rather than run beside it.
_art_cells = circulate.arterial_cells(plan.get("arterials") or {}, heights, site.x, site.z)
if _art_cells:
    print(f"arterials: {len(_art_cells)} columns laid before the lanes")
net = circulate.plan_network(heights, site.x, site.z, sites,
                             centre=plan.get("centre"), max_step=3,
                             avoid_extra=_avoid, passable=_routing["passable"],
                             arterial=_art_cells)
plan_s = time.perf_counter() - t0
wc = circulate.walk_check(net)
print(f"planned {len(net.cells)} lane cells in {plan_s:.1f}s  {net.notes}")
print(f"walk check: {wc['reached']}/{wc['cells']} cells reachable in the plan, "
      f"{wc['unreachable_n']} not")
if wc["unreachable_n"]:
    print(f"  unreachable: {wc['unreachable'][:8]}")

# ------------------------------------------------------------------- the build
b = Builder(site)
t0 = time.perf_counter()
stats = circulate.emit(b, net, mat, ground=heights, x0=site.x, z0=site.z,
                       keep_off=settlement.path_columns())
emit_s = time.perf_counter() - t0

# The same cheap check preflight makes of a model program, made of ours: every id we are
# about to write, validated against the server's own registry before anything is placed.
bad = registry.check_all(sorted({v for v in b._pending.values()}))
if bad:
    print(f"!! invalid block states, nothing written: {bad}")
    record("circulation_reject", name=settlement.NAME, invalid=list(bad))
    sys.exit(2)

res = b.flush()
print(f"\nemitted {len(b._pending)} blocks in {emit_s:.1f}s, "
      f"placed {res['placed']} failed {res['failed']} in {res['seconds']}s "
      f"({res['blocks_per_sec']}/s), rejoined {res['connective_rejoined']} connective")
print(f"lane: {stats}")
net.notes["material"] = mat
net.notes["blocks"] = res["placed"]
net.save(settlement.network_path())
print(f"wrote {settlement.network_path()}")

# ------------------------------------------------------------------- the check
t0 = time.perf_counter()
ed2 = world.editor()
ed2.loadWorldSlice(Rect((X - PAD, Z - PAD), (S + 2 * PAD, S + 2 * PAD)), cache=True)
ys = [r["y"] for r in net.cells.values()]
y0, y1 = max(0, min(ys) - 24), max(ys) + 40
vol = observe.Volume.from_world_slice(ed2.worldSlice, X - PAD, Z - PAD,
                                      S + 2 * PAD, S + 2 * PAD, y0, y1)
ctx = lint.Context.build(vol, settlement.load_plots(), network=net, sites=sites,
                         region=(X, Z, X + S - 1, Z + S - 1))
report = lint.lint(ctx)
build_s = time.perf_counter() - t0

print(f"\ncontext {vol.shape} y {y0}..{y1} in {build_s:.1f}s; "
      f"{len(ctx.circulation)} stances on the derived network, "
      f"{len(net.cells)} lane cells declared")
print(report.summary())

pieces = ctx.lane_pieces()
print(f"\nlane is in {len(pieces)} walkable piece(s): "
      f"{sorted((len(p) for p in pieces), reverse=True)[:6]}")

out = {"site": [X, Z, S], "material": mat, "notes": net.notes, "lane": stats,
       "walk_check": wc, "flush": {k: v for k, v in res.items()
                                   if k not in ("palette", "material_families")},
       "lint": report.to_json(), "pieces": [len(p) for p in pieces],
       "seconds": {"load": round(load_s, 1), "plan": round(plan_s, 1),
                   "emit": round(emit_s, 1), "lint": round(build_s, 1)}}
json.dump(out, open(os.path.join(settlement.STATE, "circulation.json"), "w"), indent=1)
record("circulation_pass", name=settlement.NAME, cells=len(net.cells),
       steps=stats["steps"], blocks=res["placed"], pieces=len(pieces),
       thresholds=len(net.thresholds), sites=len(sites),
       errors=len(report.errors), warnings=len(report.warnings),
       counts=report.to_json()["counts"], seconds=out["seconds"])
sys.exit(0 if report.ok and len(pieces) <= 1 else 1)
