"""How much a building's walls stand out of their own plane, as a number.

**Report this to a human and never to a builder, and never target it.** It is here
because a person, shown two candidates from one brief, chose the one the judge did not
and wrote *"there is no depth in the structures in A"* -- and nothing in the measure set
was watching facades. This is that sentence turned into a geometric fact of the same
family as walk fraction: not "is it beautiful", but "does the surface deviate from a
plane, and by how much". A builder told this number exists would add noise to a wall,
which is worse than a flat one; see the same warning at the head of `variety.py`.

The measure, per exterior wall face:

    depth(u, y)   how far the outermost solid cell at that point sits behind the face's
                  own outer bounding plane, in blocks
    modal plane   the depth the face mostly is -- the mode over its samples
    articulation  the fraction of samples whose depth is not the modal one
    relief        the mean |depth - mode| over those samples, in blocks

A flat run scores 0. Pilasters, buttresses, recessed panels, jettied storeys and
set-backs score above it.

Two exclusions, and both are the difference between measuring depth and measuring
something else:

**Sloped surfaces are not sampled at all.** A pitched roof's outward face steps back
one block per course, so a measure that sampled it would report a gable as the most
articulated thing in the town and rank buildings by roof pitch. A sample is kept only
where the surface runs *vertically* -- the cell one above or one below it is solid at
the same depth -- which is what a wall is and what a slope is not. The cost is that a
one-course string ledge is dropped rather than counted, so this understates and never
inflates; `sloped` reports how much was set aside.

**What is more than `MAX_RECESS` blocks off the wall plane is not the wall.** In both
directions, and the symmetry is the point. Behind the plane it is a hole: look through
an unglazed window and the first solid cell is the far interior wall ten blocks back,
and a measure that counted that would score a barn door as depth. In front of it, it is
a wing, a porch or a tower -- a separate mass, whose *position* is massing and not
surface relief. The first version of this measure capped only the far side, so a
projecting wing counted fully as relief while an equally deep recess was discarded, and
on the four selection candidates it ranked the flat-walled one *highest*. Samples
outside the band are counted as `beyond` and reported.

One known property, so it is not a surprise in a later reading: **returns are sampled.**
Past a corner, a face still sees the side wall edge-on, and a side wall that is itself
articulated is genuinely set back from the front face's plane. That adds a few
deep-but-legitimate samples, which nudges `relief` up and leaves `articulation` -- the
fraction, and the headline -- where the wall puts it. `scripts/test_articulation.py`
pins both.
"""
from __future__ import annotations

import numpy as np

from . import observe
from .variety import MADE, _built_mask, _placed_mask, _spread

#: Half-width, in blocks, of the band around the modal plane that counts as the wall.
#: Three is deliberately generous -- a two-block reveal round a door is real depth, and
#: so is a two-block buttress -- and what falls outside it is reported rather than
#: dropped silently.
MAX_RECESS = 3

#: The four faces, as (axis, sign): +x is the face a viewer standing to the east sees.
FACES = (("east", 0, +1), ("west", 0, -1), ("south", 2, +1), ("north", 2, -1))


def _mode(values: np.ndarray) -> int:
    """The most common value, lowest wins a tie.

        Lowest rather than first-seen: the modal plane of a wall that is half flush and half
        set back one block is the flush half, and a tie broken by array order would make the
        answer depend on which corner the volume starts at.
        
    """
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _face_samples(solid: np.ndarray, axis: int, sign: int):
    """(depth, keep_vertical) for one face of a solid mask.

        `depth` is the distance from the face's outer bounding plane to the first solid
        cell along the view axis, per (u, y); `hit` marks the columns that have one at all.
        
    """
    s = solid if sign < 0 else np.flip(solid, axis=axis)
    hit = s.any(axis=axis)
    depth = np.argmax(s, axis=axis)
    return np.where(hit, depth, -1), hit


def measure_face(solid: np.ndarray, axis: int, sign: int) -> dict:
    """One exterior face of one structure. `solid` is (x, y, z) occupancy."""
    depth, hit = _face_samples(solid, axis, sign)
    # `depth` and `hit` are indexed by the two axes that survive the projection, in
    # their original order: (y, z) for an x-face, (x, y) for a z-face.
    if not hit.any():
        return {"samples": 0}

    # Vertical runs only: the sample above or below is at the same depth. y is axis 1 of
    # the volume, which is axis 0 of an x-face projection and axis 1 of a z-face one.
    yax = 0 if axis == 0 else 1
    up = np.zeros_like(hit)
    dn = np.zeros_like(hit)
    sl_lo = [slice(None), slice(None)]
    sl_hi = [slice(None), slice(None)]
    sl_lo[yax] = slice(0, -1)
    sl_hi[yax] = slice(1, None)
    same = hit[tuple(sl_lo)] & hit[tuple(sl_hi)] & (depth[tuple(sl_lo)]
                                                   == depth[tuple(sl_hi)])
    up[tuple(sl_lo)] = same          # a neighbour above at the same depth
    dn[tuple(sl_hi)] = same          # a neighbour below at the same depth
    vertical = hit & (up | dn)
    sloped = int(hit.sum() - vertical.sum())
    if not vertical.any():
        return {"samples": 0, "sloped": sloped}

    d = depth[vertical].astype(int)
    mode = _mode(d)
    beyond = np.abs(d - mode) > MAX_RECESS
    kept = d[~beyond]
    if not len(kept):
        return {"samples": 0, "sloped": sloped, "beyond": int(beyond.sum())}
    mode = _mode(kept)
    dev = kept != mode
    return {
        "samples": int(len(kept)),
        "sloped": sloped,
        "beyond": int(beyond.sum()),
        "mode_depth": mode,
        "deviating": int(dev.sum()),
        "articulation": round(float(dev.mean()), 4),
        "relief": round(float(np.abs(kept - mode).mean()), 3),
        "max_relief": int(np.abs(kept - mode).max()),
    }


def measure_plot(vol: observe.Volume, built: np.ndarray, placed: np.ndarray,
                 plot: dict, ground: np.ndarray) -> dict:
    """One structure, four faces. Everything read out of the world, nothing declared."""
    x0, z0 = max(plot["x0"], vol.x0), max(plot["z0"], vol.z0)
    x1 = min(plot["x1"], vol.x0 + vol.shape[0] - 1)
    z1 = min(plot["z1"], vol.z0 + vol.shape[2] - 1)
    row = {"label": plot["label"], "samples": 0}
    if x1 < x0 or z1 < z0:
        return row

    sub = built[x0 - vol.x0:x1 - vol.x0 + 1, :, z0 - vol.z0:z1 - vol.z0 + 1]
    psub = placed[x0 - vol.x0:x1 - vol.x0 + 1, :, z0 - vol.z0:z1 - vol.z0 + 1]
    if not sub.any():
        return row

    # The ground this stands in, from a ring outside the plot -- inside it, the "ground"
    # is whatever the builder graded it to. Identical to variety.measure_plot, and
    # deliberately the same rule: two measures that disagree about where a building
    # starts would not be comparable.
    ring = []
    for i in range(max(0, x0 - vol.x0 - 3), min(ground.shape[0], x1 - vol.x0 + 4)):
        for j in (max(0, z0 - vol.z0 - 3), min(ground.shape[1] - 1, z1 - vol.z0 + 3)):
            ring.append(int(ground[i, j]))
    for j in range(max(0, z0 - vol.z0 - 3), min(ground.shape[1], z1 - vol.z0 + 4)):
        for i in (max(0, x0 - vol.x0 - 3), min(ground.shape[0] - 1, x1 - vol.x0 + 3)):
            ring.append(int(ground[i, j]))
    base = sorted(ring)[len(ring) // 2] if ring else vol.y0

    # Walls, not foundations: a plinth buried in the hillside has no exterior face, and
    # the cut a builder made to seat it would read as relief.
    keep_y = (np.arange(sub.shape[1]) + vol.y0) > base
    mass = sub & keep_y[None, :, None]
    made = (psub & keep_y[None, :, None]).sum() / max(1, mass.sum())
    if not mass.any():
        return row
    # Crop to the structure's own bounding box, so the face's outer plane is the
    # building's rather than the plot's.
    ii = np.nonzero(mass.any(axis=(1, 2)))[0]
    kk = np.nonzero(mass.any(axis=(0, 1)))[0]
    jj = np.nonzero(mass.any(axis=(0, 2)))[0]
    mass = mass[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1, kk[0]:kk[-1] + 1]

    faces = {name: measure_face(mass, ax, sg) for name, ax, sg in FACES}
    n = sum(f.get("samples", 0) for f in faces.values())
    dev = sum(f.get("deviating", 0) for f in faces.values())
    rel = sum(f.get("relief", 0.0) * f.get("samples", 0) for f in faces.values())
    row.update(
        samples=n, deviating=dev,
        articulation=round(dev / n, 4) if n else None,
        relief=round(rel / n, 3) if n else None,
        max_relief=max((f.get("max_relief", 0) for f in faces.values()), default=0),
        sloped=sum(f.get("sloped", 0) for f in faces.values()),
        beyond=sum(f.get("beyond", 0) for f in faces.values()),
        made=round(float(made), 2), faces=faces)
    return row


def measure(vol: observe.Volume, plots: list) -> dict:
    """Every plot, and the spread across them."""
    built = _built_mask(vol)
    placed = _placed_mask(vol)
    ground, _wet = observe.ground_heights(vol)
    rows = [measure_plot(vol, built, placed, p, ground) for p in plots]
    rows = [r for r in rows if r.get("samples")]
    # A plot whose mass is mostly what the hillside was already made of is a terrace or
    # a cliff, not a building. Same threshold and same reason as variety.measure.
    for r in rows:
        r["counted"] = r.get("made", 0.0) >= MADE
    kept = [r for r in rows if r["counted"]]
    n = sum(r["samples"] for r in kept)
    dev = sum(r["deviating"] for r in kept)
    return {
        "structures": len(kept),
        "not_counted": [r["label"] for r in rows if not r["counted"]],
        "samples": n, "deviating": dev,
        # The headline: one number over every wall sample in the subject, so a big
        # building weighs more than a shed. The per-structure spread is beside it
        # because two buildings averaging 0.2 is not the same place as one at 0.4 and
        # one at 0.0.
        "articulation": round(dev / n, 4) if n else None,
        "relief": (round(sum(r["relief"] * r["samples"] for r in kept) / n, 3)
                   if n else None),
        "max_relief": max((r["max_relief"] for r in kept), default=0),
        "sloped": sum(r["sloped"] for r in kept),
        "beyond": sum(r["beyond"] for r in kept),
        "per_structure": _spread([r["articulation"] for r in kept]),
        "plots": rows,
    }


def report(m: dict, title: str) -> str:
    out = [f"# {title}", "",
           f"{m['structures']} structures, {m['samples']} wall samples.", ""]
    if not m["plots"]:
        return "\n".join(out + ["  nothing with an exterior wall face here."])
    w = max([len(r["label"]) for r in m["plots"]] + [9])
    out.append(f"{'structure':{w}}  {'artic':>6} {'relief':>6} {'max':>4} "
               f"{'samples':>7} {'sloped':>6} {'beyond':>6}   faces (e/w/s/n)")
    for r in sorted(m["plots"], key=lambda r: -(r["articulation"] or 0)):
        f = r["faces"]
        per = " ".join(f"{f[k].get('articulation', 0) or 0:.2f}"
                       for k in ("east", "west", "south", "north"))
        out.append(f"{'' if r['counted'] else '('}{r['label']:{w}}  "
                   f"{r['articulation']:6.3f} {r['relief']:6.2f} {r['max_relief']:4d} "
                   f"{r['samples']:7d} {r['sloped']:6d} {r['beyond']:6d}   {per}"
                   f"{'' if r['counted'] else ')'}")
    if m["not_counted"]:
        out += ["", f"  ({', '.join(m['not_counted'])} in brackets: less than "
                    f"{int(MADE * 100)}% placed material above ground, so ground "
                    f"rather than building, and left out of the total.)"]
    s = m["per_structure"]
    out += ["", f"  overall  articulation {m['articulation']}  relief "
                f"{m['relief']} blocks  max {m['max_relief']}",
            f"  spread   n {s.get('n', 0)}  mean {s.get('mean')}  stdev "
            f"{s.get('stdev')}  range {s.get('min')} .. {s.get('max')}",
            f"  set aside  {m['sloped']} sloped samples, {m['beyond']} beyond "
            f"the wall plane"]
    return "\n".join(out)
