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

    # --- v2 A6: the map and the instance sheet -------------------------------- Two
    # more drawings, held to the same contract as the two above: a pure function of
    # their input, byte-identical on the same input, and fast enough that looking is
    # never the reason not to.
    from ethoslm import pipeline, preview as preview_mod
    from ethoslm.circulate import Network

    ex = os.path.join(ROOT, "out", "example")
    if os.path.exists(os.path.join(ex, "plan.json")):
        plan = json.load(open(os.path.join(ex, "plan.json")))
        net = Network.load(os.path.join(ex, "network.json"))
        site = json.load(open(os.path.join(ex, "site.json")))
        where = {"origin": site["origin"], "size": site["size"]}
        m1 = preview_mod.plan_map(plan, net, where)
        t0 = time.perf_counter()
        m2 = preview_mod.plan_map(plan, net, where)
        map_s = time.perf_counter() - t0
        cases.append(("A6 map: byte-identical across two calls",
                      m1.shape == m2.shape and bool((m1 == m2).all()), f"{m1.shape}"))
        cases.append(("A6 map: the site at one pixel a column, and under a second",
                      m1.shape[0] == site["size"] + 16 and map_s < 1.0,
                      f"{m1.shape[0]}px for a site of {site['size']}, {map_s:.2f}s"))
        # ...and it draws what it says it draws: every named colour is on the map
        seen = {tuple(c) for c in np.unique(m1.reshape(-1, 3), axis=0)}
        want = {k: preview_mod.MAP_COLOURS[k]
                for k in ("ground", "lane", "plot", "area", "wall", "point", "door")}
        missing = sorted(k for k, c in want.items() if tuple(c) not in seen)
        cases.append(("A6 map: districts, lanes, plots, areas, walls, gates and doors "
                      "are all on it", not missing, f"missing: {missing}"))
        # a plan with no network still draws, which is what makes it a *plan* map
        cases.append(("A6 map: a plan alone is enough to draw one",
                      preview_mod.plan_map(plan, None, where).shape == m1.shape, ""))
    else:
        print("skip A6 map: no out/example/plan.json")

    types_cfg = os.path.join(ROOT, "rounds", "types_g.json")
    if os.path.exists(types_cfg):
        rnd = pipeline.Round.load(types_cfg)
        fixture = (rnd.types or {}).get("check_fixtures", [{}])[0]
        t0 = time.perf_counter()
        s1 = preview_mod.instances("cottage", fixture, seeds=(21, 22, 23, 24, 25))
        sheet_s = time.perf_counter() - t0
        s2 = preview_mod.instances("cottage", fixture, seeds=(21, 22, 23, 24, 25))
        cases.append(("A6 sheet: byte-identical across two calls",
                      s1.shape == s2.shape and bool((s1 == s2).all()), f"{s1.shape}"))
        cases.append(("A6 sheet: five buildings in seconds, with no server",
                      sheet_s < 10 and s1.shape[1] > 5 * 20,
                      f"{sheet_s:.1f}s, {s1.shape[1]}px wide"))
        # the five are not five copies: a type varies by seed, and a sheet is how you
        # see it
        one = preview_mod.instances("cottage", fixture, seeds=(21,))
        cases.append(("A6 sheet: the seeds differ, so the sheet shows something",
                      s1.shape[1] > 4 * one.shape[1], f"{s1.shape} against {one.shape}"))
    else:
        print("skip A6 sheet.")

    # --- the colour table covers every voice on disk, not three cached towns --- The
    # audit above reads the palettes of three towns built in 2025. The sheet's first
    # outing stood a wall in `white_render_dark_frame` and drew it magenta: quartz, the
    # material two committed voices render in, had no entry in the table at all. A
    # palette audit that only reads what has already been built cannot see the voice
    # nothing has been built in yet.
    from ethoslm import prims
    import glob
    magenta = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "voices", "*.json"))):
        roles = (json.load(open(p)).get("roles") or {})
        for role, family in roles.items():
            if not isinstance(family, str):
                continue
            for kind in prims.SHAPES:
                try:
                    block = prims.shape(family, kind)
                except Exception:                    # noqa: BLE001 -- not every shape
                    continue
                if not block:
                    continue
                name = str(block).split("[")[0].split(":")[-1]
                if block_colour(name) == UNKNOWN:
                    magenta.setdefault(name, set()).add(
                        f"{os.path.basename(p)[:-5]}:{role}")
    cases.append(("palette: every shape of every role of every voice on disk resolves",
                  not magenta,
                  f"magenta: {sorted(magenta)[:6]}"))
    # ...and every material family, whether a voice uses it or not: a family with no
    # reading is a family the card names without a colour and the previewer draws
    # magenta, and the two are the same hole.
    from ethoslm.prims import MATERIALS, solid
    families = sorted(f for f in MATERIALS if block_colour(solid(f)) == UNKNOWN)
    cases.append(("palette: every material family the library can shape has a colour",
                  not families, f"no reading for: {families}"))

    fails = 0
    for label, ok, detail in cases:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        fails += not ok
    print(f"\n{len(cases) - fails}/{len(cases)} preview cases pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
