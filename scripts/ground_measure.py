"""The ground under a built place, measured. Deterministic; no model call.

    $PY scripts/ground_measure.py <round>            # out/<round>/ground_measures.json
    $PY scripts/ground_measure.py <round> --no-lint  # everything but the place-wide lint

What a person sees of a place is mostly its ground: walls that step, houses below their
own gardens, fields on stilts. None of the nine bars reads any of it, so this reads it
off the record a round leaves behind -- `plan.json`'s layout, `parts.json`'s sited
records, the pre-build and built volumes -- and writes the numbers down:

  rings        per ring annulus of a concentric layout, on the pre-build ground: the
               share of columns under water, the median ground level, and the lowest
               ground in it (a hollow the wall runs over);
  walls        every edge part: the level of each segment it was sited at, the spread
               of those levels, and whether it stands at one level;
  parts        the census by ground class (plinth / platform / deck / footing), and
               every part whose floor is below the lowest ground of its own pad;
  thresholds   E008 place-wide: the reserved thresholds obstructed, against the count
               of thresholds, read by the same lint the bars read, over the whole site
               rather than a wave's own ground.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import observe, offline                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_GROUND = re.compile(r"over ground y=(-?\d+)\.\.(-?\d+)")
_LEVELS = re.compile(r"a level per segment at y=([-\d,]+)")
_ONE = re.compile(r"one level at y=(-?\d+)")


def ring_ground(vol, layout: dict) -> list:
    """Per ring annulus: water share, median and least ground on `vol`."""
    h, wet = observe.ground_heights(vol)
    cx, cz = layout["centre"]
    out = []
    for r in layout.get("rings") or []:
        a, b = int(r["inner"]), int(r["outer"])
        i0, i1 = cx - b - vol.x0, cx + b - vol.x0 + 1
        j0, j1 = cz - b - vol.z0, cz + b - vol.z0 + 1
        i0, j0 = max(0, i0), max(0, j0)
        hh = h[i0:i1, j0:j1]
        ww = wet[i0:i1, j0:j1]
        # the annulus: everything inside the outer half-side and outside the inner
        xs = np.arange(i0, i1) + vol.x0 - cx
        zs = np.arange(j0, j1) + vol.z0 - cz
        inner = (np.abs(xs)[:, None] <= a) & (np.abs(zs)[None, :] <= a)
        mask = ~inner
        land = hh[mask & ~ww]
        out.append({"ring": r["name"], "inner": a, "outer": b,
                    "columns": int(mask.sum()),
                    "water_pct": round(100.0 * float(ww[mask].mean()), 1),
                    "median_y": int(np.median(hh[mask])),
                    "land_median_y": int(np.median(land)) if land.size else None,
                    "min_y": int(hh[mask].min()), "max_y": int(hh[mask].max()),
                    "relief": int(hh[mask].max() - hh[mask].min())})
    return out


def wall_levels(rows: list) -> list:
    out = []
    for r in rows:
        if r.get("kind") != "edge":
            continue
        s = r.get("sited") or ""
        m = _LEVELS.search(s)
        levels = ([int(v) for v in m.group(1).split(",")] if m else
                  [int(_ONE.search(s).group(1))] if _ONE.search(s) else [])
        g = _GROUND.search(s)
        out.append({"part": r["part"], "type": r.get("type"),
                    "floor_y": r.get("floor_y"), "segments": len(levels),
                    "levels": levels,
                    "lowest": min(levels) if levels else None,
                    "highest": max(levels) if levels else None,
                    "spread": (max(levels) - min(levels)) if levels else None,
                    "one_level": len(set(levels)) == 1 if levels else None,
                    "ground": [int(g.group(1)), int(g.group(2))] if g else None})
    return out


def part_census(rows: list) -> dict:
    by: dict = {}
    sunk = []
    for r in rows:
        if r.get("status") != "built":
            continue
        by[r.get("ground")] = by.get(r.get("ground"), 0) + 1
        s = r.get("sited") or ""
        g = _GROUND.search(s)
        fy = r.get("floor_y")
        if g and fy is not None and int(fy) < int(g.group(1)):
            sunk.append({"part": r["part"], "type": r.get("type"), "floor_y": int(fy),
                         "ground": [int(g.group(1)), int(g.group(2))],
                         "below_by": int(g.group(1)) - int(fy)})
    return {"by_ground": by, "built": sum(by.values()),
            "below_own_ground": sorted(sunk, key=lambda d: -d["below_by"])}


def threshold_lint(state: str, log=print) -> dict:
    """E008 over the whole site, by the lint the bars read."""
    from ethoslm import lint, settlement
    from ethoslm.circulate import Network
    vb = os.path.join(state, "world_built.npz")
    net_p = os.path.join(state, "network.json")
    site_p = os.path.join(state, "site.json")
    if not (os.path.exists(vb) and os.path.exists(net_p) and os.path.exists(site_p)):
        return {"read": False, "why": "no world_built.npz, network.json or site.json"}
    t0 = time.perf_counter()
    vol = offline.load_volume(vb)
    plots = settlement.registry_with_floors(state)
    net = Network.load(net_p)
    s = json.load(open(site_p))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    base_p = os.path.join(state, "world.npz")
    base = offline.load_volume(base_p) if os.path.exists(base_p) else None
    ctx = lint.Context.build(vol, plots=plots, network=net,
                             region=(X, Z, X + S - 1, Z + S - 1), base=base)
    rep = lint.lint(ctx)
    counts: dict = {}
    for f in rep.findings:
        counts[f.code] = counts.get(f.code, 0) + 1
    e008 = [f for f in rep.findings if f.code == "E008"]
    doors = sum(1 for f in e008 if (f.detail or {}).get("door"))
    log(f"   lint over the whole site in {time.perf_counter() - t0:.0f}s: "
        f"{counts}")
    return {"read": True, "thresholds": len(net.thresholds), "e008": len(e008),
            "e008_doorway": doors, "e008_threshold": len(e008) - doors,
            "e008_pct": round(100.0 * len(e008) / max(1, len(net.thresholds)), 1),
            "errors_by_code": dict(sorted(counts.items())),
            "examples": [f.message for f in e008[:6]],
            "seconds": round(time.perf_counter() - t0, 1)}


def measure(name: str, *, do_lint: bool = True, log=print) -> dict:
    state = os.path.join(ROOT, "out", name)
    plan = json.load(open(os.path.join(state, "plan.json")))
    parts = json.load(open(os.path.join(state, "parts.json")))
    rows = [r for w in parts.get("waves", []) for r in w.get("parts", [])]
    layout = plan.get("layout") or {}
    pre = None
    for cand in ("world.before-plateau.npz", "world.npz"):
        p = os.path.join(state, cand)
        if os.path.exists(p):
            pre = cand
            break
    out = {"round": name, "generated_by": "scripts/ground_measure.py",
           "pre_build_volume": pre}
    if pre and layout.get("rings"):
        vol = offline.load_volume(os.path.join(state, pre))
        out["rings"] = ring_ground(vol, layout)
        for r in out["rings"]:
            log(f"   {r['ring']}: water {r['water_pct']}%, median y={r['median_y']}, "
                f"ground y={r['min_y']}..{r['max_y']}")
    out["walls"] = wall_levels(rows)
    for w in out["walls"]:
        log(f"   {w['part']}: {w['segments']} segment(s) at "
            f"{','.join(str(v) for v in w['levels'])}"
            + (" -- one level" if w["one_level"] else f" -- spread {w['spread']}"))
    out["parts"] = part_census(rows)
    log(f"   parts by ground: {out['parts']['by_ground']}; "
        f"{len(out['parts']['below_own_ground'])} below their own ground")
    for d in out["parts"]["below_own_ground"][:8]:
        log(f"      {d['part']} ({d['type']}) floor y={d['floor_y']} over ground "
            f"y={d['ground'][0]}..{d['ground'][1]}")
    if do_lint:
        out["thresholds"] = threshold_lint(state, log=log)
        t = out["thresholds"]
        if t.get("read"):
            log(f"   E008 place-wide: {t['e008']} of {t['thresholds']} thresholds "
                f"({t['e008_pct']}%), {t['e008_doorway']} of them the doorway")
    p = os.path.join(state, "ground_measures.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"-> {os.path.relpath(p, ROOT)}")
    return out


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    measure(argv[0], do_lint="--no-lint" not in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
