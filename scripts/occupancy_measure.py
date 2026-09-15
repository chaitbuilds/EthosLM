"""How full a built place is, measured. Deterministic; no model call.

    $PY scripts/occupancy_measure.py <round>     # out/<round>/occupancy_measures.json

`ground_measure.py` reads the ground a place stands on. This reads what stands **on**
it, which is the other half of what a person sees from the air and what no bar reads:

  rings      per ring annulus of a concentric layout: how many structures stand in it,
             how many columns of ground there are per structure, and what share of the
             annulus is covered by plots, by areas, and by the two together;
  terraces   the cover block each ring's terrace was laid with, off the terrace record,
             and whether the rings agree on one block;
  floor      what the **open ground** of each ring is actually made of in the built
             world -- every column of the annulus that is not a leaf's rectangle and
             not a lane -- beside what the same columns read before a part was built.
             A terrace laid in grass that reads as footing afterwards says so here and
             nowhere else;
  centre     the compound's rectangle as a share of the whole footprint and of the
             innermost ring's square, beside the centre share the spec declared.

Every leaf is assigned to a ring by **geometry** -- the annulus its rectangle's centre
falls in -- so the reading does not depend on how a district was named. Rectangles are
clipped to the annulus, so a field drawn over a ring boundary is not counted twice.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _rect(p: dict):
    """A leaf's rectangle, from whichever of the three shapes it carries."""
    fp = p.get("footprint")
    if fp and len(fp) == 4:
        return (int(min(fp[0], fp[2])), int(min(fp[1], fp[3])),
                int(max(fp[0], fp[2])), int(max(fp[1], fp[3])))
    if all(k in p for k in ("x0", "z0", "x1", "z1")):
        return (int(min(p["x0"], p["x1"])), int(min(p["z0"], p["z1"])),
                int(max(p["x0"], p["x1"])), int(max(p["z0"], p["z1"])))
    at = p.get("at")
    if at and len(at) == 2:
        return (int(at[0]), int(at[1]), int(at[0]), int(at[1]))
    path = p.get("path")
    if path:
        xs = [int(c[0]) for c in path]
        zs = [int(c[1]) for c in path]
        return (min(xs), min(zs), max(xs), max(zs))
    return None


def _leaves(node: dict):
    """Every leaf of an assembled plan tree, with its quarter chain dropped."""
    kids = node.get("children")
    if kids:
        for c in kids:
            yield from _leaves(c)
        return
    if node.get("kind") in ("plot", "area", "edge", "point"):
        yield node


def _ring_of(layout: dict, x: int, z: int):
    """Which ring annulus a column is in: the first whose half-side reaches it.

        Chebyshev distance from the layout's own centre, which is how the rings are drawn.
        Returns the ring index, or None for the centre square and for outside the place.
        
    """
    cx, cz = layout["centre"]
    r = max(abs(x - cx), abs(z - cz))
    for k, ring in enumerate(layout.get("rings") or []):
        if r <= int(ring["outer"]):
            return None if r <= int(ring["inner"]) and k == 0 else k
    return None


def _clip(rect, cx, cz, inner, outer):
    """The columns of `rect` inside this annulus: the outer square less the inner."""
    x0, z0, x1, z1 = rect
    def area(half):
        ax0, az0 = max(x0, cx - half), max(z0, cz - half)
        ax1, az1 = min(x1, cx + half), min(z1, cz + half)
        return max(0, ax1 - ax0 + 1) * max(0, az1 - az0 + 1)
    return max(0, area(outer) - area(inner))


def rings(plan: dict, layout: dict) -> list:
    """Per ring: the structures in it, the ground per structure, and the cover."""
    cx, cz = layout["centre"]
    leaves = list(_leaves(plan["parts"][0])) if plan.get("parts") else []
    out = []
    for k, ring in enumerate(layout.get("rings") or []):
        inner, outer = int(ring["inner"]), int(ring["outer"])
        annulus = int(ring["annulus_columns"])
        plots = areas = 0
        plot_cols = area_cols = 0
        for leaf in leaves:
            rect = _rect(leaf)
            if rect is None:
                continue
            mx, mz = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
            if _ring_of(layout, mx, mz) != k:
                continue
            cols = _clip(rect, cx, cz, inner, outer)
            if leaf["kind"] == "plot":
                plots += 1
                plot_cols += cols
            elif leaf["kind"] == "area":
                areas += 1
                area_cols += cols
        out.append({
            "ring": ring["name"], "index": k, "walled": bool(ring.get("walled")),
            "density": ring.get("density"), "voice": ring.get("voice"),
            "inner": inner, "outer": outer, "annulus_columns": annulus,
            "district_columns": int(ring.get("district_columns") or 0),
            "district_coverage": ring.get("coverage"),
            "structures": plots, "areas": areas,
            "columns_per_structure": (round(annulus / plots, 1) if plots else None),
            "plot_columns": plot_cols, "area_columns": area_cols,
            "plot_cover": round(plot_cols / annulus, 4) if annulus else None,
            "area_cover": round(area_cols / annulus, 4) if annulus else None,
            "ground_cover": round((plot_cols + area_cols) / annulus, 4)
            if annulus else None,
        })
    return out


def terraces(state: str) -> dict:
    """The cover block each ring's terrace was laid with, off the terrace record."""
    p = os.path.join(state, "terraces.json")
    if not os.path.exists(p):
        return {"read": False, "why": "no terraces.json"}
    rec = json.load(open(p))
    by_ring = []
    covers: dict = {}
    for r in rec.get("rings") or []:
        pieces = ((r.get("terrace") or {}).get("piece_records")) or []
        got = sorted({str(pc.get("cover")) for pc in pieces if pc.get("cover")})
        for c in got:
            covers[c] = covers.get(c, 0) + 1
        by_ring.append({"ring": r.get("ring"), "level": r.get("level"),
                        "pieces": len(pieces), "covers": got})
    podium = (rec.get("terrace") or {}).get("podium")
    return {"read": True, "rings": by_ring, "podium_level": podium,
            "covers": dict(sorted(covers.items())),
            "one_cover": len(covers) == 1}


#: How many of the commonest blocks of a ring's open ground the record keeps.
FLOOR_TOP = 6


def floor(state: str, plan: dict, layout: dict, log=print) -> dict:
    """What the ground between the buildings is made of, per ring, in the built world.

        The open ground is the annulus less every leaf's rectangle grown by the clearance a
        plot keeps, and less the circulation's own cells: what is left is what a person
        walks over and looks at, and no instrument in this project had ever read it. Read on
        the built volume and again on the volume the ground work left behind, so a cover
        that was laid and then taken off is visible as the difference.
        
    """
    import numpy as np
    from ethoslm import observe, offline
    built = os.path.join(state, "world_built.npz")
    if not os.path.exists(built):
        return {"read": False, "why": "no world_built.npz"}
    before = os.path.join(state, "world.npz")
    cx, cz = layout["centre"]
    leaves = list(_leaves(plan["parts"][0])) if plan.get("parts") else []

    def surface(path):
        vol = offline.load_volume(path)
        h, _wet = observe.ground_heights(vol)
        nx, _ny, nz = vol.codes.shape
        names = np.array([b.split("[")[0] for b in vol.palette])
        top = names[vol.codes[np.arange(nx)[:, None], h - vol.y0,
                              np.arange(nz)[None, :]]]
        return vol, top

    vol, top = surface(built)
    nx, _ny, nz = vol.codes.shape
    xs = np.arange(nx) + vol.x0 - cx
    zs = np.arange(nz) + vol.z0 - cz
    cheb = np.maximum(np.abs(xs)[:, None], np.abs(zs)[None, :])
    # every leaf's own ground: its rectangle and the margin the plan keeps round it. An
    # edge is skipped -- a ring wall's bounding box is the city it encloses.
    occupied = np.zeros((nx, nz), bool)
    for leaf in leaves:
        if leaf.get("kind") == "edge":
            continue
        rect = _rect(leaf)
        if rect is None:
            continue
        i0 = max(0, rect[0] - 2 - vol.x0)
        i1 = min(nx, rect[2] + 3 - vol.x0)
        j0 = max(0, rect[1] - 2 - vol.z0)
        j1 = min(nz, rect[3] + 3 - vol.z0)
        occupied[i0:i1, j0:j1] = True
    net_p = os.path.join(state, "network.json")
    if os.path.exists(net_p):
        from ethoslm.circulate import Network
        for c in Network.load(net_p).cells:
            i, j = int(c[0]) - vol.x0, int(c[1]) - vol.z0
            if 0 <= i < nx and 0 <= j < nz:
                occupied[i, j] = True
    was = None
    if os.path.exists(before):
        _v, was = surface(before)

    def census(mask, grid):
        vals, counts = np.unique(grid[mask], return_counts=True)
        order = np.argsort(-counts)
        total = int(counts.sum()) or 1
        return [{"block": str(vals[i]), "pct": round(100.0 * int(counts[i]) / total, 1)}
                for i in order[:FLOOR_TOP]]

    out = []
    for ring in layout.get("rings") or []:
        m = (cheb > int(ring["inner"])) & (cheb <= int(ring["outer"])) & ~occupied
        if not m.any():
            continue
        row = {"ring": ring["name"], "open_columns": int(m.sum()),
               "built": census(m, top)}
        if was is not None:
            row["before_the_parts"] = census(m, was)
        out.append(row)
        log(f"   {ring['name']} open ground: "
            + ", ".join(f"{d['block']} {d['pct']}%" for d in row["built"][:4]))
    return {"read": True, "rings": out, "top": FLOOR_TOP}


def centre(plan: dict, layout: dict, site: dict) -> dict:
    """The compound's rectangle against the place and against its own ring."""
    rect = layout.get("compound_rect")
    S = int(site["size"])
    comps = plan.get("compounds") or []
    out = {"site_side": S, "site_columns": S * S,
           "centre_share_declared": layout.get("centre_share"),
           "centre_share_got": layout.get("centre_share_got"),
           "compound": (comps[0].get("name") if comps else None)}
    if rect:
        x0, z0, x1, z1 = [int(v) for v in rect]
        side = max(x1 - x0 + 1, z1 - z0 + 1)
        cols = (x1 - x0 + 1) * (z1 - z0 + 1)
        out.update({"compound_rect": [x0, z0, x1, z1], "compound_side": side,
                    "compound_columns": cols,
                    "share_of_place": round(cols / float(S * S), 4),
                    "side_share_of_place": round(side / float(S), 4)})
        inner = layout.get("rings") or []
        if inner:
            r0 = inner[0]
            square = (2 * int(r0["outer"]) + 1)
            out["innermost_ring_side"] = square
            out["side_share_of_innermost"] = round(side / float(square), 4)
    return out


def measure(name: str, log=print) -> dict:
    state = os.path.join(ROOT, "out", name)
    plan = json.load(open(os.path.join(state, "plan.json")))
    site = json.load(open(os.path.join(state, "site.json")))
    layout = plan.get("layout") or {}
    out = {"round": name, "generated_by": "scripts/occupancy_measure.py"}
    if layout.get("rings"):
        out["rings"] = rings(plan, layout)
        for r in out["rings"]:
            log(f"   {r['ring']}: {r['structures']} structures over "
                f"{r['annulus_columns']} columns = {r['columns_per_structure']} a "
                f"structure; plots {r['plot_cover']:.1%}, areas {r['area_cover']:.1%}, "
                f"ground {r['ground_cover']:.1%}")
    out["terraces"] = terraces(state)
    if out["terraces"].get("read"):
        for r in out["terraces"]["rings"]:
            log(f"   {r['ring']} terrace at y={r['level']}: cover "
                f"{', '.join(r['covers']) or 'none recorded'}")
    if layout.get("rings"):
        out["floor"] = floor(state, plan, layout, log=log)
    out["centre"] = centre(plan, layout, site)
    c = out["centre"]
    if c.get("compound_side"):
        log(f"   {c['compound']}: {c['compound_side']} square, "
            f"{c['share_of_place']:.2%} of the place, "
            f"{c['side_share_of_innermost']:.1%} of the innermost ring's width")
    p = os.path.join(state, "occupancy_measures.json")
    json.dump(out, open(p, "w"), indent=1)
    log(f"-> {os.path.relpath(p, ROOT)}")
    return out


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    measure(argv[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
