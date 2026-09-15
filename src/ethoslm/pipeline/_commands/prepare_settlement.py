'Pre-generate a settlement site and write the terrain briefing every pass reads.\n\n    bash scripts/mcrun.sh scripts/prepare_settlement.py [X Z SIZE]'
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
import numpy as np
from ethoslm import settlement, world
from ethoslm.measure import record

X, Z, S = -1536, 768, 192
if len(sys.argv) >= 4:
    X, Z, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ed = world.editor()

for ox in range(-64, S + 64, 96):
    for oz in range(-64, S + 64, 96):
        world.load_site(ed, X + ox, Z + oz, 96, 96)
site = world.load_site(ed, X - 32, Z - 32, S + 64, S + 64)
h = site.heights.astype(int) - 1
inner = h[32:32 + S, 32:32 + S]

# 12x12 grid of 16-block cells; also the max within each cell, so cliffs are visible.
# **Transposed on purpose.** `inner` is indexed [x][z], so reducing it gives a grid
# whose rows are x -- while the briefing that quotes it says, and has always said, that
# rows run north to south. The grid is stored the way its caption reads.
cell = S // 12
# **The grid is 12x12 whatever the site is.** `inner.reshape(12, cell, 12, cell)` needs
# `S` to be a multiple of twelve, and `spec.FOOTPRINT_STEP` exists to make it one --
# except that `CEILING["footprint"]` is 512, which is not, so the first place ever asked
# for at the ceiling crashed here. The summary is over the largest 12-divisible box
# inside the site; the remainder is at most eleven columns on two sides and is described
# by the cell beside it. A briefing is a summary, and a summary of 504 of 512 is one.
n = cell * 12
box = inner[:n, :n]
mean_g = box.reshape(12, cell, 12, cell).mean(axis=(1, 3)).round().astype(int).T
max_g = box.reshape(12, cell, 12, cell).max(axis=(1, 3)).astype(int).T
min_g = box.reshape(12, cell, 12, cell).min(axis=(1, 3)).astype(int).T
rough = (max_g - min_g)

# Say it out loud rather than trusting the reshape: a handful of real columns read back
# out of the world, against the cell the briefing puts them in.
checks = []
for (px, pz) in ((S // 6, S // 6), (S // 2, S // 6), (S // 6, S // 2), (5 * S // 6, S // 6)):
    checks.append({"x": X + px, "z": Z + pz, "world_y": int(inner[px, pz]),
                   "grid_cell": [pz // cell, px // cell],
                   "grid_y": int(mean_g[pz // cell][px // cell])})

surface = {}
for gx in range(0, S, 12):
    for gz in range(0, S, 12):
        b = ed.getBlock((X + gx, int(inner[gx, gz]), Z + gz)).id.split(":")[-1]
        surface[b] = surface.get(b, 0) + 1

brief = {
    "origin": [X, Z], "size": S,
    "stats": {"min": int(inner.min()), "max": int(inner.max()),
              "relief": int(inner.max() - inner.min()),
              "std": round(float(inner.std()), 1)},
    "mean_grid": mean_g.tolist(),          # [z-cell][x-cell]: rows north to south
    "roughness_grid": rough.tolist(),
    "grid": {"cell": int(cell), "summarised": int(n),
             "note": f"the 12x12 grid summarises the {n}x{n} inside this {S}x{S} site; "
                     f"{S - n} column(s) on the east and south edges are described by "
                     f"the cell beside them"},
    "surface_blocks": surface,
    "orientation_check": checks,
}
os.makedirs(settlement.STATE, exist_ok=True)
json.dump(brief, open(os.path.join(settlement.STATE, "site.json"), "w"), indent=1)

print(f"site ({X},{Z}) {S}x{S}  {brief['stats']}  -> {settlement.STATE}/site.json")
print("\nmean surface height, 16-block cells (rows N->S, cols W->E):")
for row in mean_g:
    print("  " + " ".join(f"{v:3d}" for v in row))
print("\nroughness within each cell (max-min) — high means steep or broken:")
for row in rough:
    print("  " + " ".join(f"{v:3d}" for v in row))
print("\nsurface:", surface)
print("\norientation check — a column read out of the world against the cell the grid "
      "puts it in:")
for c in checks:
    print(f"  ({c['x']},{c['z']}) world y={c['world_y']:>3}  "
          f"grid[{c['grid_cell'][0]}][{c['grid_cell'][1]}]={c['grid_y']:>3}  "
          f"{'ok' if abs(c['world_y'] - c['grid_y']) < 25 else 'MISMATCH'}")
record("settlement_site_prepared", name=settlement.NAME, origin=[X, Z], size=S,
       **brief["stats"])
