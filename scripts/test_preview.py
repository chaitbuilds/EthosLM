"""The previewer's contract, checked without a server or a render.

The previewer is about to become the eye of every stage decision, and the judge caches
on image *content* -- a renderer with a wobble would silently re-buy every judgement,
and a colour table with a hole would show a judge magenta where the roof is. So the
contract is tested, in the shape of the other offline suites:

  1. Pure function: byte-identical output across two calls, both projections.
     (Same-pixel depth ties are geometrically impossible in this projection -- equal
     px and equal depth force the same voxel -- so the stable sort is cross-platform
     insurance rather than a tie-breaker; determinism is still the tested property.)
  2. Clip: a cropped building does not sit on its column of subsurface stone. The
     discriminating half renders with clip off and shows the substrate is what the
     clip removes.
  3. The elevation is an elevation: a gable reads as a narrowing profile, a flat roof
     does not, and the two are distinguishable -- which a single isometric mass is not
     obliged to make true.
  4. The colour table covers every block state in all three standing towns' palettes.
     Magenta means "a state nobody has met before", not "a state nobody typed in".
  5. A building renders in under 20 ms, because the whole design depends on looking
     being too cheap to skip.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np                                   # noqa: E402

from ethoslm import observe, offline                   # noqa: E402
from ethoslm.preview import (block_colour, crop, elevation,  # noqa: E402
                     preview, UNKNOWN)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOWNS = [("site_g", "out/settlement"), ("site_a", "out/site_a"),
         ("site_b", "out/site_b")]


def hut_on_substrate(depth: int = 40):
    """A 7x5x7 hut whose ground sits on `depth` blocks of solid stone column."""
    blocks = {}
    for x in range(20):
        for z in range(20):
            for y in range(depth + 1):
                blocks[(x, y, z)] = "stone"          # substrate + surface at y=depth
    for x in range(6, 13):
        for z in range(6, 13):
            for y in range(depth + 1, depth + 5):
                if x in (6, 12) or z in (6, 12):
                    blocks[(x, y, z)] = "oak_planks"
    return observe.Volume.from_blocks(blocks, 0, 0, 0, 20, depth + 8, 20)


def gable(flat: bool = False):
    """A mass with a gabled (or flat) top, ridge along z, on thin ground."""
    blocks = {}
    for x in range(16):
        for z in range(16):
            blocks[(x, 0, z)] = "grass_block"
    for i in range(6):
        w0, w1 = (2, 13) if flat else (2 + i, 13 - i)
        for x in range(w0, w1 + 1):
            for z in range(4, 12):
                blocks[(x, 1 + i, z)] = "stone_bricks"
    return observe.Volume.from_blocks(blocks, 0, 0, 0, 16, 10, 16)


def profile_widths(img, bg=235):
    """Non-background pixel count per image row, top to bottom."""
    return [(row != bg).any(axis=1).sum() for row in img]


def main():
    cases = []

    # --- determinism ------------------------------------------------------
    vol = hut_on_substrate()
    a, b = preview(vol), preview(vol)
    cases.append(("preview byte-identical across two calls",
                  a.shape == b.shape and bool((a == b).all()), f"{a.shape}"))
    ea, eb = elevation(vol), elevation(vol)
    cases.append(("elevation byte-identical across two calls",
                  ea.shape == eb.shape and bool((ea == eb).all()), f"{ea.shape}"))

    # --- clip -------------------------------------------------------------
    clipped = preview(vol, scale=1)
    unclipped = preview(vol, scale=1, clip=False)
    # the hut is 4 tall on its ground; unclipped, 40 blocks of substrate render below
    cases.append(("clip: building does not sit on its subsurface column",
                  clipped.shape[0] + 20 < unclipped.shape[0],
                  f"clipped {clipped.shape[0]}px tall, unclipped {unclipped.shape[0]}"))
    cases.append(("clip: ...and the unclipped render really does show substrate",
                  unclipped.shape[0] > 40, f"{unclipped.shape[0]}px"))

    # --- the elevation is an elevation ------------------------------------
    g = elevation(gable(), scale=1, clip=False)
    f = elevation(gable(flat=True), scale=1, clip=False)
    gw, fw = profile_widths(g), profile_widths(f)
    cases.append(("elevation: a gable narrows toward its ridge",
                  gw[0] < gw[-2] and gw[0] <= 4, f"widths top..bottom {gw}"))
    cases.append(("elevation: ...and a flat roof does not",
                  fw[0] == fw[1] and fw[0] >= 10, f"widths {fw}"))
    cases.append(("elevation: gable and flat are distinguishable",
                  g.shape != f.shape or not (g == f).all(), ""))

    # --- colour table covers the standing towns ---------------------------
    for name, d in TOWNS:
        p = os.path.join(ROOT, d, "world_built.npz")
        if not os.path.exists(p):
            print(f"skip {name}: no cache at {d}/world_built.npz")
            continue
        pal = list(np.load(p, allow_pickle=True)["palette"])
        bad = sorted({s for s in pal if block_colour(s) == UNKNOWN})
        cases.append((f"palette: every {name} state resolves ({len(pal)} states)",
                      not bad, f"magenta: {bad[:6]}"))

    # --- speed ------------------------------------------------------------
    site_b = os.path.join(ROOT, "out/site_b/world_built.npz")
    if os.path.exists(site_b):
        town = offline.load_volume(site_b)
        plots = json.load(open(os.path.join(ROOT, "out/site_b/plots.json")))
        big = max(plots, key=lambda q: (q["x1"] - q["x0"]) * (q["z1"] - q["z0"]))
        bvol = crop(town, big["x0"], big["z0"], big["x1"], big["z1"])
        preview(bvol)                                   # warm the tables cache
        t0 = time.perf_counter()
        preview(bvol)
        ms = (time.perf_counter() - t0) * 1000
        cases.append((f"speed: {big['label']} renders in under 20 ms",
                      ms < 20, f"{ms:.1f} ms"))
        c, cc = preview(bvol), preview(bvol)
        cases.append(("preview byte-identical on a real building",
                      bool((c == cc).all()), ""))
    else:
        print("skip speed: no site_b cache")

    fails = 0
    for label, ok, detail in cases:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        fails += not ok
    print(f"\n{len(cases) - fails}/{len(cases)} preview cases pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
