"""How much the buildings of a settlement differ from each other, as a number.

**Report this to a human and never to a builder.** A pass told its buildings are too
similar will add variation for its own sake, which is worse than the disease -- and an
aesthetic score is the one thing this project has refused to build for six rounds, on
the grounds that nine years of GDMC say it does not exist. Nothing here is a judgement.
Every row is a fact about geometry of the same kind as every other measurement in the
project: how tall, how long, how many floors, what shape the roof is, how many holes are
in the walls. What it is *for* is the risk that comes with the library owning more:
adding primitives is how this project got better and it is also how it could get worse,
and if a new primitive collapses a town toward one form, that should show up in a report
next to everything else rather than being discovered on foot three rounds later.

So it measures **spread**, not quality. A town of fourteen identical huts and a town of
fourteen identical cathedrals both score zero.
"""
from __future__ import annotations

import math

import numpy as np

from . import observe

#: Below this fraction of placed material a plot's mass is the hillside it stands on,
#: not a building. lint.Context.MADE, kept in step deliberately.
MADE = 0.25

#: Openings: the holes a builder puts in a wall. Deliberately not "decoration".
_OPENING = ("_pane", "_door", "_trapdoor", "_fence_gate", "iron_bars", "glass",
            "_bars", "lattice")

#: What a building is made of, as opposed to what it stands on. `observe._is_natural`
#: knows the ground's materials; anything else in a plot was placed by somebody.


def _built_mask(vol: observe.Volume) -> np.ndarray:
    """Solid, and not vegetation. Deliberately **material-blind**."""
    t = vol.tables()
    names = [s.split("[")[0] for s in vol.palette]
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))
    veg = np.array([observe._is_vegetation(n) for n in names], bool)
    return (solid & ~veg)[vol.codes]


def _placed_mask(vol: observe.Volume) -> np.ndarray:
    t = vol.tables()
    names = [s.split("[")[0] for s in vol.palette]
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))
    natural = np.array([observe._is_natural(n) or observe._is_vegetation(n)
                        for n in names], bool)
    return (solid & ~natural)[vol.codes]


def _opening_mask(vol: observe.Volume) -> np.ndarray:
    names = [s.split("[")[0] for s in vol.palette]
    op = np.array([any(k in n for k in _OPENING) for n in names], bool)
    # a door is two blocks and one opening
    upper = np.array([n.endswith("_door") and "half=upper" in s
                      for n, s in zip(names, vol.palette)], bool)
    return (op & ~upper)[vol.codes]


def _spread(values: list) -> dict:
    """min/max/mean/stdev, and the coefficient of variation where it means anything.

        Population standard deviation, because these are the whole settlement and not a
        sample of one. CV is reported alongside because a stdev of 3 means one thing on
        ridge heights of 5 and another on ridge heights of 40, and the towns being compared
        are not the same size.
        
    """
    vs = [v for v in values if v is not None]
    if not vs:
        return {"n": 0}
    m = sum(vs) / len(vs)
    sd = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))
    return {"n": len(vs), "min": round(min(vs), 2), "max": round(max(vs), 2),
            "mean": round(m, 2), "stdev": round(sd, 2),
            "cv": round(sd / m, 3) if m else None}


def _diversity(values: list) -> dict:
    """Distinct kinds, and Shannon entropy normalised to [0, 1] over what is present.

        1.0 is every plot a different roof; 0.0 is every plot the same one. Normalising by
        log(n_plots) rather than log(n_kinds) is deliberate -- five styles spread evenly
        over fourteen buildings should not score the same as fourteen.
        
    """
    vs = [v for v in values if v is not None]
    if not vs:
        return {"n": 0}
    counts: dict = {}
    for v in vs:
        counts[v] = counts.get(v, 0) + 1
    h = -sum((c / len(vs)) * math.log(c / len(vs)) for c in counts.values())
    return {"n": len(vs), "distinct": len(counts),
            "entropy": round(h / math.log(len(vs)), 3) if len(vs) > 1 else 0.0,
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


def _roof_form(tops: dict) -> tuple[str, float]:
    """Classify a roof from the top-of-building height per column, and measure its pitch.

        Geometry, not taxonomy: where the highest cells are, and how the surface falls away
        from them. "ridged" covers gable, gambrel and mansard, which are not distinguishable
        from a height field alone and are not claimed to be.
        
    """
    if not tops:
        return "none", 0.0
    hs = sorted(tops.values())
    lo = hs[0]
    # The ridge is not the highest block: a chimney, a lantern post or a finial is one
    # column, and taking the maximum made every building in both towns come out
    # "pyramid" with a pitch of up to 17. The ridge is the highest level a real part of
    # the roof reaches -- at least a twentieth of the columns, and never fewer than
    # three.
    need = max(3, len(hs) // 20)
    hi = hs[-need] if len(hs) >= need else hs[-1]
    if hi - lo <= 1:
        return "flat", 0.0
    ridge = [p for p, h in tops.items() if h >= hi - 0.5]
    xs = sorted({p[0] for p in ridge})
    zs = sorted({p[1] for p in ridge})
    allx = sorted({p[0] for p in tops})
    allz = sorted({p[1] for p in tops})
    spanx = (max(xs) - min(xs) + 1) / max(1, len(allx))
    spanz = (max(zs) - min(zs) + 1) / max(1, len(allz))

    # pitch: blocks of rise per block of run, from the ridge out to the lowest eave
    run = max(1, min(len(allx), len(allz)) / 2)
    pitch = round((hi - lo) / run, 2)

    edge_high = sum(1 for p in tops
                    if (p[0] in (allx[0], allx[-1]) or p[1] in (allz[0], allz[-1]))
                    and tops[p] >= hi - 1)
    if spanx > 0.6 and spanz > 0.6:
        return "stepped", pitch          # high all over: towers, terraces, massing
    if spanx > 0.6 or spanz > 0.6:
        # a ridge line across the building. If one end is also the highest edge it is a
        # shed (a single slope), otherwise it is a ridge with two.
        return ("shed" if edge_high > 0.4 * (len(allx) + len(allz)) else "ridged"), pitch
    return "pyramid", pitch              # the high ground is a point: hip, cone, spire


def measure_plot(vol: observe.Volume, built, placed, openings, rooms, plot: dict,
                 ground: np.ndarray) -> dict:
    """One structure, measured. Everything is read out of the world, nothing declared."""
    x0, z0 = max(plot["x0"], vol.x0), max(plot["z0"], vol.z0)
    x1 = min(plot["x1"], vol.x0 + vol.shape[0] - 1)
    z1 = min(plot["z1"], vol.z0 + vol.shape[2] - 1)
    if x1 < x0 or z1 < z0:
        return {"label": plot["label"], "built_columns": 0}

    sub = built[x0 - vol.x0:x1 - vol.x0 + 1, :, z0 - vol.z0:z1 - vol.z0 + 1]
    psub = placed[x0 - vol.x0:x1 - vol.x0 + 1, :, z0 - vol.z0:z1 - vol.z0 + 1]
    any_col = sub.any(axis=1)
    if not any_col.any():
        return {"label": plot["label"], "built_columns": 0}
    top = sub.shape[1] - 1 - np.argmax(sub[:, ::-1, :], axis=1) + vol.y0

    # The ground this stands in, taken from a ring outside the plot: inside it, the
    # "ground" is whatever the builder graded it to.
    ring = []
    for i in range(max(0, x0 - vol.x0 - 3), min(ground.shape[0], x1 - vol.x0 + 4)):
        for j in (max(0, z0 - vol.z0 - 3), min(ground.shape[1] - 1, z1 - vol.z0 + 3)):
            ring.append(int(ground[i, j]))
    for j in range(max(0, z0 - vol.z0 - 3), min(ground.shape[1], z1 - vol.z0 + 4)):
        for i in (max(0, x0 - vol.x0 - 3), min(ground.shape[0] - 1, x1 - vol.x0 + 3)):
            ring.append(int(ground[i, j]))
    base = sorted(ring)[len(ring) // 2] if ring else vol.y0

    # The structure is the mass standing above the ground around it, plus anything made
    # of placed material -- so a cistern sunk into a terrace counts and the hillside a
    # plot happens to include does not.
    tops = {(int(i), int(j)): int(top[i, j]) for i, j in np.argwhere(any_col)
            if int(top[i, j]) > base or psub[int(i), :, int(j)].any()}
    if not tops:
        return {"label": plot["label"], "built_columns": 0}
    ii = sorted({p[0] for p in tops})
    jj = sorted({p[1] for p in tops})
    w, d = ii[-1] - ii[0] + 1, jj[-1] - jj[0] + 1
    made = sum(1 for (i, j) in tops if psub[i, :, j].any()) / len(tops)

    form, pitch = _roof_form(tops)

    # storeys: how many distinct levels of interior this plot has, from the rooms the
    # observation layer already found. Levels within 2 blocks are one floor.
    levels = sorted(r["bbox"][1] for r in rooms
                    if r.get("plot") == plot["label"])
    storeys, last = 0, None
    for y in levels:
        if last is None or y - last > 2:
            storeys += 1
            last = y

    # openings against the wall they are in: exterior faces of the built mass.
    osub = openings[x0 - vol.x0:x1 - vol.x0 + 1, :, z0 - vol.z0:z1 - vol.z0 + 1]
    face = np.zeros_like(sub)
    for ax, sh in ((0, 1), (0, -1), (2, 1), (2, -1)):
        face |= sub & ~np.roll(sub, sh, axis=ax)
    wall = int(face.sum())
    n_open = int(osub.sum())

    return {"label": plot["label"], "built_columns": len(tops),
            "ridge_height": int(max(tops.values())) - base,
            "footprint": [w, d], "aspect": round(max(w, d) / max(1, min(w, d)), 2),
            "storeys": storeys, "roof": form, "pitch": pitch,
            "openings": n_open, "wall_cells": wall, "made": round(made, 2),
            "opening_density": round(100 * n_open / wall, 2) if wall else 0.0}


def measure(vol: observe.Volume, plots: list, rooms: list | None = None) -> dict:
    """Every plot, and the spread across them."""
    built = _built_mask(vol)
    placed = _placed_mask(vol)
    openings = _opening_mask(vol)
    ground, _wet = observe.ground_heights(vol)
    if rooms is None:
        nav = observe.Nav(vol)
        sky_open, _sealed = observe.shelter(vol)
        rooms = observe.rooms(nav, sky_open)
        for r in rooms:
            b = r["bbox"]
            cx, cz = (b[0] + b[3]) / 2, (b[2] + b[5]) / 2
            r["plot"] = next((p["label"] for p in plots
                              if p["x0"] <= cx <= p["x1"] and p["z0"] <= cz <= p["z1"]),
                             None)
    rows = [measure_plot(vol, built, placed, openings, rooms, p, ground)
            for p in plots]
    rows = [r for r in rows if r.get("built_columns")]
    # A plot whose mass is mostly what the hillside was already made of is a track, a
    # yard or a cliff, not a building, and averaging it in would measure the terrain.
    # Same threshold as lint.Context.MADE, and for the same reason.
    kept = [r for r in rows if r["made"] >= MADE]
    for r in rows:
        r["counted"] = r["made"] >= MADE
    rows, all_rows = kept, rows
    return {
        "structures": len(rows),
        "not_counted": [r["label"] for r in all_rows if not r["counted"]],
        "ridge_height": _spread([r["ridge_height"] for r in rows]),
        "aspect": _spread([r["aspect"] for r in rows]),
        "storeys": _spread([r["storeys"] for r in rows]),
        "opening_density": _spread([r["opening_density"] for r in rows]),
        "pitch": _spread([r["pitch"] for r in rows]),
        "roof": _diversity([r["roof"] for r in rows]),
        "plots": all_rows,
    }


def report(m: dict, title: str) -> str:
    out = [f"# {title}", "",
           f"{m['structures']} structures, measured out of the world.", ""]
    w = max([len(r["label"]) for r in m["plots"]] + [9])
    out.append(f"{'structure':{w}}  {'ridge':>5} {'w x d':>9} {'aspect':>6} "
               f"{'storeys':>7} {'roof':>8} {'pitch':>5} {'open/100':>8} "
               f"{'made':>5}")
    for r in sorted(m["plots"], key=lambda r: -r["ridge_height"]):
        out.append(f"{'' if r['counted'] else '('}{r['label']:{w}}  {r['ridge_height']:5d} "
                   f"{r['footprint'][0]:4d} x{r['footprint'][1]:3d} {r['aspect']:6.2f} "
                   f"{r['storeys']:7d} {r['roof']:>8} {r['pitch']:5.2f} "
                   f"{r['opening_density']:8.2f} {r['made']:5.2f}"
                   f"{'' if r['counted'] else ')'}")
    if m["not_counted"]:
        out += ["", f"  ({', '.join(m['not_counted'])} in brackets: less than "
                    f"{int(MADE * 100)}% placed material, so they are ground rather "
                    f"than building and are left out of the spread.)"]
    out += ["", "Spread across the settlement -- variance, not a score:", ""]
    for k in ("ridge_height", "aspect", "storeys", "pitch", "opening_density"):
        s = m[k]
        if not s.get("n"):
            continue
        out.append(f"  {k:16s} mean {s['mean']:7.2f}  stdev {s['stdev']:6.2f}  "
                   f"cv {s['cv'] if s['cv'] is not None else 0:5.3f}  "
                   f"range {s['min']} .. {s['max']}")
    r = m["roof"]
    out.append(f"  {'roof form':16s} {r['distinct']} distinct, entropy "
               f"{r['entropy']:.3f} of 1.0  {r['counts']}")
    return "\n".join(out)
