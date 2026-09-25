"""The controlled three-way material comparison, on production-owned surfaces.

    $PY scripts/material_compare.py --context farm=out/expr-farm --context city=out/expr-city
    $PY scripts/material_compare.py --context farm=out/expr-farm --controls-only
    $PY scripts/material_compare.py --context farm=out/closure-farm --reconstruct

**What this answers and what the expression round's version could not.** That run
judged `leave_off` and the review called the result inconclusive rather than negative,
for two reasons, and both are fixed here before anything is compared again:

  the cameras       `choose_group` picked a group by footprint adjacency and drew a
                    padded crop of it, so the farm's close views landed on a hillside
                    and a roof corner: terrain in front of the elevation is the first
                    solid the projection hits. Subjects are now derived from the built
                    record -- a named house's street face off its emitted footprint and
                    the plan's `front`, the same face again at eye height, the row it
                    stands in, a court from above, a wall's outer face -- and the crop
                    stops **at the facade plane**, so nothing in front of the subject
                    can occlude it.
  the display       flat shading gives cobblestone and andesite one colour nine units
                    apart, which is the substitution the farm's recipe makes on its
                    walls. Close views are drawn with `preview.elevation(texture=True)`
                    -- the game's own texels -- and a **positive control** measures
                    that the display registers both a gross substitution and that exact
                    pair before any arm is judged. `--controls-only` runs just that.

For each context (a round's state directory: `world_built.npz`, `world.npz`,
`parts.json`, `surfaces.json`) and each setting (`maintained`, `weathered`), the three
modes -- `none`, `random`, `contextual` -- are applied to the SAME built volume and
drawn with the SAME cameras and lighting. `random` is matched to `contextual` in how
much it changes of what, **per voice and per type** (`material.context_of`), so the
two differ only in where.

The surface record is reconciled against the assembled world first
(`surfaces.reconcile`): a cell a later part overwrote belongs to the later part, a
cell any part protected is protected, a cell whose block no longer stands is gone, a
cell with no air on any face is not worth editing, and exposure is recomputed from the
finished volume. The before/after counts are in the record.

Where a state has no `surfaces.json` (a candidate built before the emission record
existed), `--reconstruct` copies the state into the work directory and re-runs the
parts stage there on one worker with a wrapper that records every builder, then reports
whether the rebuilt world is byte-for-byte the state's. Nothing under the state
directory is written.

Determinism and the physical check are measured, not asserted: the pass is applied in
two orders and twice, and on its own output; the construction check (`ethoslm.lint`) runs
over the finished volume with the same plots, network and base as `stage_lint`, and the
solid occupancy of every cell is compared with the built world's.

The judgment is left `null`: a reader reads the images and writes the decision into
`out/des-material/comparison.json`. The pass stays off until that says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from ethoslm import lint, material as M, offline, preview, surfaces as S  # noqa: E402
from ethoslm.pipeline.stages_measure import LIVE_ONLY_CODES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODES = ("none", "random", "contextual")
SETTINGS = ("maintained", "weathered")
#: A close view gives a block the game's own 16 texels, doubled so a reader can see
#: them. `_tile` upscales nearest, so the texels stay square and countable.
TEXEL = 32
ROW_SCALE = 16
COURT_SCALE = 16
PLACE_SCALE = 2
#: How much of a facade an eye-height view takes in, and how far up it looks. Five
#: courses is a person's own view of a wall from the other side of the lane.
EYE_RUN = 24
EYE_RISE = 5
NEIGHBOUR_GAP = 8

#: The gross positive control: a substitution across a known face that no display may
#: miss. The second control is chosen per subject -- see `nearest_by_colour`.
GROSS = "blackstone"


def _rows(parts_rec: dict) -> list:
    return [r for w in (parts_rec.get("waves") or []) for r in (w.get("parts") or [])]


def _rect(row: dict, plan_parts: dict):
    """The rectangle a part actually stands on: the emitted footprint where emission
    measured one, the planned lot otherwise."""
    fp = (row.get("emitted") or {}).get("footprint")
    if fp and len(fp) == 4:
        return [int(v) for v in fp]
    p = plan_parts.get(row["part"])
    if p and p.get("x1") is not None:
        return [int(p["x0"]), int(p["z0"]), int(p["x1"]), int(p["z1"])]
    return None


def _gap(a, b) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dz = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dz)


# ----------------------------------------------------------------- the cameras A camera
# is derived from the built record and aimed at a named subject. The rule that makes an
# elevation of a building an elevation *of the building*: the crop stops at the facade
# plane, so the first solid the projection meets along the view axis is the face under
# judgement and not the hill, the lane or the neighbour in front of it.

def face_crop(vol, rect, facing: str, floor: int | None, *, side: int = 1,
              rise: int | None = None, run: int | None = None):
    """The sub-volume an elevation of `rect`'s `facing` face may draw.

        Bounded at the facade plane along the view axis; `side` blocks of slack across it
        so an eave that oversails the footprint is not clipped off. `floor` cuts the
        subsoil, `rise` limits the view to that many courses above the floor (an eye-height
        view of a wall) and `run` to that many blocks along the face, centred.
        
    """
    from ethoslm import observe
    x0, z0, x1, z1 = [int(v) for v in rect]
    if facing in ("north", "south"):
        ax0, ax1 = x0 - side, x1 + side
        az0, az1 = z0, z1
        if run:
            c = (ax0 + ax1) // 2
            ax0, ax1 = c - run // 2, c + run // 2
    else:
        az0, az1 = z0 - side, z1 + side
        ax0, ax1 = x0, x1
        if run:
            c = (az0 + az1) // 2
            az0, az1 = c - run // 2, c + run // 2
    sub = preview.crop(vol, ax0, az0, ax1, az1, pad=0)
    if floor is None:
        return sub
    pal = list(sub.palette)
    air = next((i for i, n in enumerate(pal)
                if str(n).split("[")[0].endswith("air")), None)
    if air is None:
        pal.append("air")
        air = len(pal) - 1
    codes = sub.codes.copy()
    codes[:, :max(0, int(floor) - 1 - sub.y0), :] = air
    if rise is not None:
        top = max(0, int(floor) + int(rise) - sub.y0)
        codes[:, top:, :] = air
    return observe.Volume(sub.x0, sub.y0, sub.z0, codes, pal)


def _clip_below(vol, y: int):
    """The volume with everything under `y` made air, so a place view is a place and
    not a bank of subsoil."""
    from ethoslm import observe
    pal = list(vol.palette)
    air = next((i for i, n in enumerate(pal) if str(n).split("[")[0].endswith("air")), None)
    if air is None:
        pal.append("air")
        air = len(pal) - 1
    codes = vol.codes.copy()
    codes[:, :max(0, int(y) - vol.y0), :] = air
    return observe.Volume(vol.x0, vol.y0, vol.z0, codes, pal)


def choose_row(plots: list, k: int = 3) -> tuple:
    """`k` buildings that front the same way side by side along their street."""
    if not plots:
        return [], None, None

    def beside(a, b, front) -> bool:
        if front in ("north", "south"):
            return a[1] <= b[3] and b[1] <= a[3] and _gap(a, b) <= NEIGHBOUR_GAP
        return a[0] <= b[2] and b[0] <= a[2] and _gap(a, b) <= NEIGHBOUR_GAP
    best, best_n = None, -1
    for name, rc, fr in plots:
        n = sum(1 for _m, o, f2 in plots if o is not rc and f2 == fr and beside(rc, o, fr))
        if n > best_n:
            best, best_n = (name, rc, fr), n
    fr = best[2]
    row = [p for p in plots if p[2] == fr and (p[1] is best[1] or beside(best[1], p[1], fr))]
    near = sorted(row, key=lambda p: _gap(best[1], p[1]))[:k]
    return ([n for n, _r, _f in near],
            [min(r[0] for _n, r, _f in near), min(r[1] for _n, r, _f in near),
             max(r[2] for _n, r, _f in near), max(r[3] for _n, r, _f in near)], fr)


def wall_run(plan_part: dict, centre, run: int = 40) -> tuple | None:
    """A straight run of a wall, and the side of it that faces out of the place.

        A wall is a swept polyline with no rectangle, so its camera comes off the path:
        the longest axis-aligned segment, `run` blocks of it about its midpoint, seen from
        whichever side is away from the centre of the place.
        
    """
    path = [tuple(int(v) for v in p) for p in (plan_part.get("path") or [])]
    if len(path) < 2 or centre is None:
        return None
    segs = [(a, b) for a, b in zip(path, path[1:]) if a[0] == b[0] or a[1] == b[1]]
    if not segs:
        return None
    a, b = max(segs, key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
    mx, mz = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    w = int(plan_part.get("width") or 3) + 2
    if a[1] == b[1]:                       # runs along x: its faces look north/south
        rect = [mx - run // 2, mz - w, mx + run // 2, mz + w]
        facing = "north" if mz < centre[1] else "south"
    else:
        rect = [mx - w, mz - run // 2, mx + w, mz + run // 2]
        facing = "west" if mx < centre[0] else "east"
    return rect, facing


def subjects(rows: list, plan_parts: dict, sdoc: dict, floors: dict,
             centre) -> list:
    """The subjects a material judgement is made on, off the built record.

        Chosen by what they are, not by a bounding box: the busiest street row and the
        house at its head, that house's own face at eye height, a courtyard type seen from
        above, the outer face of a wall, and the place itself for the composition.
        
    """
    built = [r for r in rows if r.get("status") == "built" and r.get("stood", True)]
    plots = [(r["part"], _rect(r, plan_parts),
              str((plan_parts.get(r["part"]) or {}).get("front") or "south"))
             for r in built if r.get("kind", "plot") == "plot" and _rect(r, plan_parts)]
    editable = {p.get("part"): sum(len(c) for c in (p.get("cells") or {}).values())
                for p in (sdoc.get("parts") or [])}
    out: list = []
    names, rect, front = choose_row(plots)
    if names:
        # the head of the row is the house that shows the most: a camera aimed at a part
        # that owns nothing after reconciliation answers nothing
        names = sorted(names, key=lambda n: -editable.get(n, 0))
        head = names[0]
        hrect = next(r for n, r, _f in plots if n == head)
        floor = floors.get(head)
        out.append({"view": "street_face", "why": "a named house's street face, "
                    "cropped at the facade plane so nothing in front occludes it",
                    "parts": [head], "rect": hrect, "facing": front, "floor": floor,
                    "kind": "elevation", "scale": TEXEL, "texture": True})
        out.append({"view": "eye", "why": f"the same wall's outer face, {EYE_RISE} "
                    f"courses above the floor and {EYE_RUN} blocks of it",
                    "parts": [head], "rect": hrect, "facing": front, "floor": floor,
                    "kind": "elevation", "scale": TEXEL, "texture": True,
                    "rise": EYE_RISE, "run": EYE_RUN})
        out.append({"view": "row", "why": "the row this house stands in, from its lane",
                    "parts": names, "rect": rect, "facing": front,
                    "floor": min((floors[n] for n in names if n in floors), default=None),
                    "kind": "elevation", "scale": ROW_SCALE, "texture": True})
    # a court, from above: the one view that shows an enclosed space and its ranges
    court = next((r for r in built if r.get("type") in
                  ("courtyard_house", "farmstead", "compound", "market", "square",
                   "yard")), None)
    if court is not None and _rect(court, plan_parts):
        out.append({"view": "court", "why": f"a {court['type']} from above -- the court "
                    "floor, its ranges and their roofs in one frame",
                    "parts": [court["part"]], "rect": _rect(court, plan_parts),
                    "facing": None, "floor": floors.get(court["part"]),
                    "kind": "top_down", "scale": COURT_SCALE, "texture": True})
    # a wall's outer face
    wall = next((r for r in built if r.get("kind") == "edge"
                 and (plan_parts.get(r["part"]) or {}).get("path")), None)
    if wall is not None:
        got = wall_run(plan_parts[wall["part"]], centre)
        if got:
            rect, facing = got
            out.append({"view": "wall_face", "why": "the outer face of a wall at eye "
                        "height, from the side away from the place",
                        "parts": [wall["part"]], "rect": rect, "facing": facing,
                        "floor": floors.get(wall["part"]), "kind": "elevation",
                        "scale": ROW_SCALE, "texture": True})
    rects = [_rect(r, plan_parts) for r in built]
    rects = [r for r in rects if r]
    if rects:
        bounds = [min(r[0] for r in rects) - 8, min(r[1] for r in rects) - 8,
                  max(r[2] for r in rects) + 8, max(r[3] for r in rects) + 8]
        low = min(floors.values()) if floors else None
        out.append({"view": "place_iso", "why": "the whole composition: does the "
                    "voice's colour and value hierarchy survive the pass",
                    "parts": [], "rect": bounds, "facing": None, "floor": low,
                    "kind": "iso", "scale": PLACE_SCALE, "texture": False, "drop": 6})
        out.append({"view": "place_roofs", "why": "the roofscape from above, where a "
                    "patch that reads as a stain rather than as a material shows",
                    "parts": [], "rect": bounds, "facing": None, "floor": low,
                    "kind": "top_down", "scale": PLACE_SCALE, "texture": False,
                    "drop": 6})
    return out


def draw(vol, subj: dict, out_dir: str) -> tuple:
    """One subject's image off one finished volume. Returns (path, image)."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    floor = subj.get("floor")
    if subj["kind"] == "elevation":
        sub = face_crop(vol, subj["rect"], subj["facing"], floor,
                        rise=subj.get("rise"), run=subj.get("run"))
        img = preview.elevation(sub, facing=subj["facing"], scale=subj["scale"],
                                clip=False, texture=subj["texture"])
    elif subj["kind"] == "top_down":
        sub = preview.crop(vol, *subj["rect"], pad=0)
        if floor is not None:
            sub = _clip_below(sub, floor - int(subj.get("drop") or 1))
        img = preview.top_down(sub, scale=subj["scale"], texture=subj["texture"])
    else:
        sub = preview.crop(vol, *subj["rect"], pad=0)
        if floor is not None:
            sub = _clip_below(sub, floor - int(subj.get("drop") or 1))
        img = preview.preview(sub, scale=subj["scale"])
    p = os.path.join(out_dir, f"{subj['view']}.png")
    cv2.imwrite(p, img[:, :, ::-1])
    return p, img


def image_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """How much two matched frames differ: the share of pixels that moved and by how
    much. Two frames of the same camera on the same geometry differ only where the
    pass wrote, so this is the display's own answer to `did it register`."""
    if a.shape != b.shape:
        return {"comparable": False, "why": f"{a.shape} vs {b.shape}"}
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    drawn = int((a != preview.BG).any(axis=2).sum())
    moved = int((d > 2).sum())
    return {"comparable": True, "changed_px": moved,
            "changed_share": round(moved / max(1, drawn), 4),
            "mean_delta_where_changed": round(float(d[d > 2].mean()) if moved else 0.0, 1),
            "max_delta": int(d.max())}


# --------------------------------------------------------- the positive control Before
# any arm is judged: does the display register the substitution under judgement? If it
# cannot, the display is the finding and the comparison is not evidence.

def control_recipe(role: str, family: str) -> dict:
    return {role: [{"family": family, "weight": 1.0, "when": "any"}]}


def one_part(sdoc: dict, name: str) -> dict:
    """The record narrowed to one part, so a control edits a known face and nothing
    else. Reconciliation is carried over, not redone."""
    return dict(sdoc, parts=[p for p in (sdoc.get("parts") or [])
                             if p.get("part") == name])


def nearest_by_colour(block: str) -> str | None:
    """The material family the **flat** display draws most nearly the same as `block`.

        This is how the fine control is chosen rather than guessed: the old display's own
        colour table names the pair it cannot separate. On the farm's cobblestone it
        returns andesite, which is exactly the substitution the expression round's recipe
        makes on its walls and exactly the one its pictures could not show.
        
    """
    from ethoslm import prims
    here = np.array(preview.block_colour(str(block)), float)
    base = str(block).split("[")[0].split(":")[-1]
    best, far = None, None
    for fam, (full, _st, _sl) in sorted(prims.MATERIALS.items()):
        # the gross control is excluded, or a dark timber would choose blackstone and
        # the fine control would quietly become the gross one again
        if str(full).split(":")[-1] == base or fam == GROSS:
            continue
        d = float(np.abs(np.array(preview.block_colour(str(full)), float) - here).max())
        if far is None or d < far:
            best, far = fam, d
    return best, far


def dominant(prec: dict, role: str | None = None) -> tuple:
    """The editable role a part shows most of, and the block it is laid in.

        `role` names one instead. The composition round: the dominant role of a courtyard
        house's street face is its roof, and the restrained recipe under judgement edits
        walls -- so the display is asked about the substitution that is actually being made
        as well as about the biggest one on the camera.
        
    """
    cells = prec.get("cells") or {}
    if not cells:
        return None, None
    if role:
        if not cells.get(role):
            return None, None
    else:
        role = max(cells, key=lambda r: len(cells[r]))
    states = prec.get("states") or []
    seen: dict = {}
    for c in cells[role]:
        s = states[c[3]]
        seen[s] = seen.get(s, 0) + 1
    return role, max(seen, key=seen.get) if seen else None


def controls(built, sdoc: dict, subj: dict, out_dir: str, seed: int,
             role_want: str | None = None) -> dict:
    """Two controls applied to the subject's own part, drawn on the same camera in
        both displays, and measured against the untouched frame.

        `gross` is a substitution no display may miss. `pair` is the two blocks the flat
        display draws alike, chosen by that display's own table. A display that registers
        `gross` and not `pair` cannot answer the question this comparison asks.
        
    """
    got: dict = {"subject": subj["view"], "role_asked_for": role_want,
                 "part": subj["parts"][0] if subj["parts"] else None, "cases": []}
    if not subj.get("parts"):
        return got
    doc = one_part(sdoc, subj["parts"][0])
    prec = (doc.get("parts") or [{}])[0]
    role, block = dominant(prec, role_want)
    if not role:
        got["error"] = (f"the subject's part owns no editable {role_want} cell after "
                        f"reconciliation" if role_want else
                        "the subject's part owns no editable cell after reconciliation")
        return got
    pair, apart = nearest_by_colour(block)
    got.update({"role": role, "block": block, "cells_in_part": len(prec["cells"][role]),
                "pair": pair, "flat_colours_apart_by": apart,
                "pair_chosen_because": f"of every material family the flat table draws "
                                       f"{pair} nearest to {block} -- {apart} units "
                                       f"apart on the widest channel, before shading"})
    cases = (("gross", GROSS), ("pair", pair))
    for texture, tag in ((True, "textured"), (False, "flat")):
        _p, ref = draw(built, {**subj, "view": "control_none", "texture": texture},
                       os.path.join(out_dir, tag))
        for name, family in cases:
            if not family:
                continue
            fin, rec = M.apply(built, doc, control_recipe(role, family),
                               {"condition": "weathered"}, seed, "contextual")
            path, img = draw(fin, {**subj, "view": f"control_{name}", "texture": texture},
                             os.path.join(out_dir, tag))
            got["cases"].append({"control": name, "display": tag, "role": role,
                                 "from": block, "to": family,
                                 "substituted": rec["substituted"],
                                 "image": path, "vs_none": image_diff(ref, img)})
    return got


# ------------------------------------------------------- what the arms look like Two
# numbers that separate the two arms without an opinion, computed off the plans rather
# than the pictures: whether a substitution landed on the condition it names, and
# whether the substitutions touch each other.

def coherence(p: dict) -> dict:
    """The share of a changed cell's six neighbours that changed to the same family.

        A coherent patch has neighbours; a uniform scatter at the same rate does not. This
        is the difference the contextual arm claims to make, as a number.
        
    """
    cells = p.get("cells") or {}
    if not cells:
        return {"changed": 0}
    fam = {c: v[1] for c, v in cells.items()}
    tot = same = 0
    for (x, y, z), f in fam.items():
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                           (0, 0, -1)):
            tot += 1
            if fam.get((x + dx, y + dy, z + dz)) == f:
                same += 1
    return {"changed": len(fam), "same_family_neighbour_share": round(same / max(1, tot), 4)}


def condition_fit(p: dict, sdoc: dict, recipe: dict) -> dict:
    """Of the substitutions a `damp` or `runoff` variant made, how many stand on a cell
    the record says is damp. The contextual arm's whole claim; the random arm's answer
    is the base rate."""
    F = S.FLAGS
    wet = F["ground_contact"] | F["water_near"] | F["base_course"] | F["under_opening"]
    flags = {}
    for prec in sdoc.get("parts") or []:
        for role, cells in (prec.get("cells") or {}).items():
            for c in cells:
                flags[(c[0], c[1], c[2])] = (role, c[4])
    conditioned = {}
    for _v, byrole in [("", recipe)] if "by_voice" not in recipe else \
            [(k, v) for k, v in (recipe.get("by_voice") or {}).items()]:
        for role, variants in (byrole or {}).items():
            for v in variants:
                if v["when"] in ("damp", "runoff"):
                    conditioned.setdefault(role, set()).add(v["family"])
    hit = tot = 0
    for c, (new, family, role, _old) in (p.get("cells") or {}).items():
        if family not in conditioned.get(role, ()):
            continue
        tot += 1
        if flags.get(c, ("", 0))[1] & wet:
            hit += 1
    # the base rate has to be read off the **same roles**: a drystone wall is mostly
    # base course, so the share of damp cells over every role is not what a scatter
    # would score on the roles these variants can reach
    pool = [f for _c, (r, f) in flags.items() if r in conditioned]
    base = sum(1 for f in pool if f & wet) / max(1, len(pool))
    return {"damp_variant_cells": tot, "on_a_damp_cell": hit,
            "share": round(hit / tot, 4) if tot else None,
            "roles": sorted(conditioned), "cells_in_those_roles": len(pool),
            "base_rate_of_damp_cells": round(base, 4)}


def _matches(state: str, copy: str) -> dict | None:
    orig = os.path.join(state, "world_built.npz")
    built_p = os.path.join(copy, "world_built.npz")
    if not (os.path.exists(orig) and os.path.exists(built_p)):
        return None
    a, b = offline.load_volume(orig), offline.load_volume(built_p)
    if a.shape != b.shape:
        return {"identical": False, "why": "shapes differ"}
    pa = np.array(list(a.palette), dtype=object)
    pb = np.array(list(b.palette), dtype=object)
    diff = int((pa[a.codes] != pb[b.codes]).sum())
    return {"identical": diff == 0, "differing_cells": diff}


def reconstruct(state: str, work: str, rebuild: bool = False) -> dict:
    """Re-run the parts stage in a copy of `state` with the emission record on, and
    write `surfaces.json` there. Returns what happened and whether the world matches."""
    from ethoslm import pipeline, deps as deps_mod
    from ethoslm.pipeline import stages_build
    name = os.path.basename(state.rstrip("/"))
    copy = os.path.join(work, name)
    done = all(os.path.exists(os.path.join(copy, f))
               for f in ("world_built.npz", "surfaces.json", "parts.json"))
    if done and not rebuild:
        sp = os.path.join(copy, "surfaces.json")
        own = S.read(sp)
        # ...and the copy is only reusable if its record carries what this comparison
        # reads. A copy made before `FLAGS["figure"]` existed is a stale record, not a
        # cache, and reusing it would silently answer with the design round's record.
        if own and S.owned_cells(own) and set(S.FLAGS) <= set(own.get("flags") or {}):
            return {"copy": copy, "reused": True, "surfaces_source": "driver",
                    "rebuilt_matches_state": _matches(state, copy)}
    if os.path.isdir(copy):
        shutil.rmtree(copy)
    os.makedirs(copy)
    for f in sorted(os.listdir(state)):
        s = os.path.join(state, f)
        if f in ("parts.json", "world_built.npz", "paths.json", "surfaces.json",
                 "world.quarters-from.npz") or f.startswith("stale."):
            continue
        if os.path.isdir(s):
            if f in ("parts", "inspection", "preview", "evidence", "acceptance"):
                continue
            shutil.copytree(s, os.path.join(copy, f), symlinks=True)
        else:
            shutil.copy2(s, os.path.join(copy, f))
    rp = os.path.join(ROOT, "rounds", f"{name}.json")
    rnd = pipeline.Round.load(rp)
    rnd.state_dir = copy
    rnd.flags = dict(rnd.flags, dry_run=True, workers=1)
    be = pipeline.OfflineBackend(rnd, dry_run=True)
    plan_parts = {p["name"]: p for p in pipeline.plan_parts(rnd.plan())}
    records: list = []
    real = offline.run_program

    def wrapped(path, vol, *a, **k):
        b = real(path, vol, *a, **k)
        label = os.path.splitext(os.path.basename(path))[0]
        sited = dict(b.parts[-1]) if getattr(b, "parts", None) else {}
        pp = plan_parts.get(label) or {}
        try:
            records.append(S.record(b, {**pp, **sited, "name": label,
                                        "type": pp.get("type"),
                                        "kind": pp.get("kind", "plot")}))
        except Exception as e:                    # noqa: BLE001 -- reported
            records.append({"part": label, "error": f"{type(e).__name__}: {e}",
                            "owned": 0, "cells": {}, "protected": [], "states": []})
        return b
    offline.run_program = wrapped
    stages_build.offline.run_program = wrapped
    t0 = time.perf_counter()
    try:
        res = stages_build.stage_parts(rnd, be, {})
        # `world_built.npz` is `stage_finish`'s, as in the driver
        fin = stages_build.stage_finish(rnd, be, {"parts": res})
    finally:
        offline.run_program = real
        stages_build.offline.run_program = real
    secs = round(time.perf_counter() - t0, 1)
    built_p = os.path.join(copy, "world_built.npz")
    sp = os.path.join(copy, "surfaces.json")
    # **The driver's own record where the stage wrote one** (the coordinator's wiring of
    # request C-1); the wrapper's record only where it did not.
    own = S.read(sp) if os.path.exists(sp) else None
    source = "driver"
    if not own or not S.owned_cells(own):
        source = "wrapper"
        pr = os.path.join(copy, "parts.json")
        rows = _rows(json.load(open(pr))) if os.path.exists(pr) else []
        voice_of = {r["part"]: r.get("voice") for r in rows}
        for rec in records:
            rec["voice_name"] = voice_of.get(rec.get("part"))
        S.write(sp, records, candidate=(res or {}).get("candidate"),
                built_digest=deps_mod.content_print(built_p) if os.path.exists(built_p) else None,
                note=f"reconstructed by scripts/material_compare.py from the stored "
                     f"programs of {name}; not the driver's own record")
    matches = _matches(state, copy)
    return {"copy": copy, "parts": len(records), "seconds": secs,
            "status": (res or {}).get("status"), "error": (res or {}).get("error"),
            "finish": {k: v for k, v in (fin or {}).items() if k in ("status", "error")},
            "surfaces_source": source, "rebuilt_matches_state": matches}


#: The waves `stage_parts` runs serially on the shared volume, before it snapshots it
#: and hands the quarters to workers. Everything else is a quarter wave.
SERIAL_WAVES = ("walls", "squares")


def re_record_state(state: str, work: str, *, rebuild: bool = False) -> dict:
    """Make a retained candidate's surface record again, off its **own stored programs**,
        without rebuilding it.

        Composition round. The figure declarations (`Primitives.figure`) are in the type
        programs, so a `surfaces.json` written before they existed carries no `figure` bit
        and cannot answer this comparison; and `--reconstruct`, which re-runs the parts
        *stage*, is refused on `out/des-city` because that candidate's own ground stamp is
        stale (`deps.check(rnd, "ground")` says so of the retained directory itself, not
        only of a copy). Neither is a reason to rebuild a city.

        So the programs are replayed exactly the way `stage_parts` ran them -- the `walls`
        and `squares` waves in order on `world.npz`, then each quarter wave on its own copy
        of `world.quarters-from.npz` and merged back in wave order -- and the record is made
        off each builder. **Nothing is built**: the replay's own volume is compared with the
        retained `world_built.npz` cell for cell and the difference is reported, so a record
        made this way is only usable if the replay reproduced the world. Declaring a figure
        places no block, so it should reproduce it exactly.

        The retained state directory is read and never written; everything goes to
        `work/<name>/`.
        
    """
    from ethoslm import ground as _ground, pipeline, stages, surfaces as SS
    from ethoslm.buildlib import max_blocks_for
    name = os.path.basename(state.rstrip("/"))
    out_dir = os.path.join(work, name)
    sp = os.path.join(out_dir, "surfaces.json")
    rp = os.path.join(out_dir, "re-record.json")
    if not rebuild and os.path.exists(sp) and os.path.exists(rp):
        doc = SS.read(sp)
        if doc and SS.owned_cells(doc) and set(SS.FLAGS) <= set(doc.get("flags") or {}):
            return {**json.load(open(rp)), "reused": True}
    os.makedirs(out_dir, exist_ok=True)
    rnd = pipeline.Round.load(os.path.join(ROOT, "rounds", f"{name}.json"))
    rnd.state_dir = state
    parts_rec = json.load(open(os.path.join(state, "parts.json")))
    plan_parts = {p["name"]: p for p in pipeline.plan_parts(rnd.plan())}
    registry = stages._Registry(state)
    net = rnd.network()
    gp = os.path.join(state, "ground.npz")
    resolved = _ground.load(gp) if os.path.exists(gp) else None
    site = (pipeline.settlement_site(rnd) or rnd.site or {})
    mb = max_blocks_for(site.get("size"))
    base = offline.load_volume(os.path.join(state, "world.npz"))
    snap_p = os.path.join(state, "world.quarters-from.npz")
    records: list = []
    problems: list = []
    t0 = time.perf_counter()

    # A stored program is the type's source **as it was when the part was built**,
    # followed by `_TYPE_HEADER` and the two driver lines that instantiate it
    # (`stages_build.instantiated_source`). The figure declarations are edits to the
    # type source, so the head is spliced for the current `types/<type>.py` and the
    # driver tail -- the plot, the seed, the params, the palette and the silhouette, all
    # of them the retained candidate's own -- is kept byte for byte.
    from ethoslm.pipeline.stages_build import _TYPE_HEADER
    spliced: dict = {}

    def one(label: str, vol):
        prog = os.path.join(state, "parts", f"{label}.py")
        if not os.path.exists(prog):
            problems.append({"part": label, "why": "no stored program"})
            return None
        stored = open(prog).read()
        src = stored
        ptype = (plan_parts.get(label) or {}).get("type")
        tf = os.path.join(ROOT, "types", f"{ptype}.py")
        if _TYPE_HEADER in stored and ptype and os.path.exists(tf):
            tail = stored.split(_TYPE_HEADER, 1)[1]
            tsrc = pipeline.load_type(tf)["src"]
            src = tsrc.rstrip("\n") + "\n" + _TYPE_HEADER + tail
            if src != stored:
                spliced[ptype] = spliced.get(ptype, 0) + 1
        else:
            problems.append({"part": label, "why": f"program has no type header, or "
                                                   f"types/{ptype}.py is not on disk"})
        b = offline.run_program(prog, vol, network=net, plots=registry, src=src,
                               max_blocks=mb, ground=resolved)
        pp = plan_parts.get(label) or {}
        sited = dict(b.parts[-1]) if getattr(b, "parts", None) else {}
        records.append(SS.record(b, {**pp, **sited, "name": label,
                                     "type": pp.get("type"),
                                     "kind": pp.get("kind", "plot")},
                                 voice_name=voice_of.get(label)))
        return dict(getattr(b, "_pending", {}) or {})

    rows = _rows(parts_rec)
    voice_of = {r["part"]: r.get("voice") for r in rows}
    built_rows = {r["part"] for r in rows if r.get("status") == "built"}
    vol = base
    for w in parts_rec.get("waves") or []:
        wave = w.get("wave")
        group = [r["part"] for r in (w.get("parts") or []) if r["part"] in built_rows]
        if wave in SERIAL_WAVES:
            for label in group:
                pend = one(label, vol)
                if pend:
                    vol = vol.overlay(pend)
        else:
            if not os.path.exists(snap_p):
                problems.append({"wave": wave, "why": "no world.quarters-from.npz"})
                continue
            snap = offline.load_volume(snap_p)
            merged: dict = {}
            wvol = snap
            for label in group:
                pend = one(label, wvol)
                if pend:
                    merged.update(pend)
                    wvol = wvol.overlay(pend)
            if merged:
                vol = vol.overlay(merged)
    secs = round(time.perf_counter() - t0, 1)
    built = offline.load_volume(os.path.join(state, "world_built.npz"))
    pa = np.array(list(vol.palette), dtype=object)
    pb = np.array(list(built.palette), dtype=object)
    same_shape = vol.shape == built.shape
    diff = int((pa[vol.codes] != pb[built.codes]).sum()) if same_shape else -1
    offline.save_volume(vol, os.path.join(out_dir, "world_replayed.npz"))
    SS.write(sp, records, candidate=parts_rec.get("candidate"),
             built_digest=None,
             note=f"re-recorded by scripts/material_compare.py from the stored programs "
                  f"of {name}; the world was not rebuilt and the retained "
                  f"world_built.npz is what every arm is drawn on")
    figs: dict = {}
    for r in records:
        for k, v in (r.get("figures") or {}).items():
            figs[k] = figs.get(k, 0) + int(v)
    rep = {"copy": out_dir, "surfaces": sp, "parts": len(records), "seconds": secs,
           "problems": problems[:12],
           "type_source_now_differs_from_the_stored_one": dict(sorted(spliced.items())),
           "recorded_editable": SS.editable_cells({"parts": records}),
           "figures": dict(sorted(figs.items())),
           "replay_matches_retained_world": {"same_shape": same_shape,
                                             "differing_cells": diff,
                                             "identical": bool(same_shape and diff == 0)},
           "how": ("the candidate's own stored programs replayed in stage_parts' order "
                   "-- the serial waves on world.npz, each quarter wave on its own copy "
                   "of world.quarters-from.npz, merged in wave order -- and the replayed "
                   "volume compared with the retained world_built.npz cell for cell")}
    json.dump(rep, open(rp, "w"), indent=1)
    return rep


def _solid_mask(vol) -> np.ndarray:
    t = vol.tables()
    return (t["lower"].astype(bool) | t["upper"].astype(bool))[vol.codes]


def physical(state: str, built, finished, parts_rec: dict) -> dict:
    """The construction check over the finished volume against the built one."""
    from ethoslm.circulate import Network
    plots_p = os.path.join(state, "plots.json")
    plots = json.load(open(plots_p)) if os.path.exists(plots_p) else []
    site_p = os.path.join(state, "site.json")
    region = None
    if os.path.exists(site_p):
        s = json.load(open(site_p))
        if s.get("origin") and s.get("size"):
            X, Z = s["origin"]
            region = (X, Z, X + s["size"] - 1, Z + s["size"] - 1)
    if parts_rec.get("sample"):
        names = {r["part"] for r in _rows(parts_rec)}
        plots = [p for p in plots if p.get("label") in names or p.get("name") in names]
        r = parts_rec["sample"].get("rect")
        if r:
            m = int(parts_rec["sample"].get("margin") or 0)
            region = (int(r[0]) - m, int(r[1]) - m, int(r[2]) + m, int(r[3]) + m)
    net_p = os.path.join(state, "network.json")
    net = Network.load(net_p) if os.path.exists(net_p) else None
    base_p = os.path.join(state, "world.npz")
    base = offline.load_volume(base_p) if os.path.exists(base_p) else None

    def codes(vol):
        ctx = lint.Context.build(vol, plots, network=net, region=region, base=base)
        rep = lint.lint(ctx)
        return sorted(f"{f.code}@{f.pos}" for f in rep.errors
                      if f.code not in LIVE_ONLY_CODES)
    t0 = time.perf_counter()
    before = codes(built)
    after = codes(finished)
    new = sorted(set(after) - set(before))
    occ = bool((_solid_mask(built) == _solid_mask(finished)).all())
    return {"clean": not new and occ, "errors_built": len(before),
            "errors_finished": len(after), "new_errors": new[:12],
            "occupancy_identical": occ,
            "how": ("ethoslm.lint over the finished volume with stage_lint's plots, "
                    "network, region and base; E005 (live-only) excluded; solid "
                    "occupancy of every cell compared with the built world"),
            "seconds": round(time.perf_counter() - t0, 1)}


def determinism(built, sdoc, recipe, settings, seed) -> dict:
    a, ra = M.apply(built, sdoc, recipe, settings, seed, "contextual")
    b, _ = M.apply(built, sdoc, recipe, settings, seed, "contextual")
    replay = M.volume_digest(a) == M.volume_digest(b)
    # a different order of the record's parts and cells
    rev = {**sdoc, "parts": list(reversed([
        {**p, "cells": {r: list(reversed(c)) for r, c in (p.get("cells") or {}).items()}}
        for p in sdoc.get("parts") or []]))}
    c, _ = M.apply(built, rev, recipe, settings, seed, "contextual")
    order = M.volume_digest(a) == M.volume_digest(c)
    # on its own output: nothing accumulates
    d, rd = M.apply(a, sdoc, recipe, settings, seed, "contextual", reconcile=False)
    idem = M.volume_digest(a) == M.volume_digest(d)
    e, _ = M.apply(built, sdoc, recipe, settings, seed + 1, "contextual")
    seeded = M.volume_digest(a) != M.volume_digest(e) if ra["substituted"] else True
    return {"replay_identical": replay, "order_independent": order,
            "idempotent_on_own_output": idem, "replay_substituted": rd["substituted"],
            "seed_moves_it": seeded, "substituted": ra["substituted"],
            "how": ("applied twice, applied with the record's parts and cells reversed, "
                    "applied to its own output, and at seed+1; volume digests compared")}


def narrow(recipe: dict, roles=None, whens=None) -> dict:
    """A recipe with only some roles or only some conditions left in it.

        Not part of the registered comparison: it is how a **diagnostic** run isolates
        which half of a recipe did the damage, after the registered run has been recorded.
        
    """
    def cut(rec):
        out = {}
        for role, variants in (rec or {}).items():
            if roles and role not in roles:
                continue
            keep = [v for v in variants if not whens or v["when"] in whens]
            if keep:
                out[role] = keep
        return out
    if "by_voice" in recipe:
        return {"by_voice": {v: cut(r) for v, r in (recipe["by_voice"] or {}).items()}}
    return cut(recipe)


def run_context(name: str, state: str, out_dir: str, work: str, seed: int,
                settings_list: list, do_reconstruct: bool, rebuild: bool = False,
                controls_only: bool = False, skip_physical: bool = False,
                roles=None, whens=None, re_record: bool = False) -> dict:
    ctx: dict = {"name": name, "state": state, "modes": list(MODES),
                 "settings": list(settings_list), "seed": seed, "images": {}}
    sp = os.path.join(state, "surfaces.json")
    src_state = state
    # **A record that predates a record change has to be made again.** Composition
    # round: the figure declarations (`Primitives.figure`) are in the type programs, so
    # a `surfaces.json` cached before they existed carries no `figure` bit and would
    # answer this comparison with the design round's record. `--re-record` re-runs the
    # stored programs in a copy, exactly as `--reconstruct` does for a candidate that
    # never had a record, and `rebuilt_matches_state` says whether the world came out
    # byte for byte the retained one -- which it must, since declaring a figure places
    # no block. The retained state directory is read and never written.
    if re_record:
        ctx["re_record"] = re_record_state(state, work, rebuild=rebuild)
        if not ctx["re_record"]["replay_matches_retained_world"]["identical"]:
            ctx["error"] = ("the replay did not reproduce the retained world "
                            f"({ctx['re_record']['replay_matches_retained_world']}); a "
                            "record made off a replay that built something else is not "
                            "a record of this candidate")
            return ctx
        sp = ctx["re_record"]["surfaces"]
    elif not os.path.exists(sp):
        if not do_reconstruct:
            ctx["error"] = f"{sp} is absent; pass --reconstruct to rebuild it in a copy"
            return ctx
        ctx["reconstruction"] = reconstruct(state, work, rebuild=rebuild)
        src_state = ctx["reconstruction"]["copy"]
        sp = os.path.join(src_state, "surfaces.json")
    raw = S.read(sp)
    built = offline.load_volume(os.path.join(src_state, "world_built.npz"))
    sdoc, rec_report = S.reconcile(raw, built)
    ctx["surfaces_from"] = sp
    ctx["owned_cells"] = S.owned_cells(raw)
    ctx["reconciliation"] = rec_report
    ctx["editable_cells"] = {"recorded": S.editable_cells(raw),
                             "after_reconciliation": S.editable_cells(sdoc)}
    # what the types declared they drew on purpose, and how much of it survived to the
    # record a pass reads: the composition round's Q2
    ctx["figures"] = {"declared": rec_report.get("figures") or {},
                      "cells_recorded": rec_report.get("figure_cells_recorded"),
                      "cells_kept": rec_report.get("figure_cells_kept"),
                      "by_part": {p.get("part"): p.get("figures")
                                  for p in (raw.get("parts") or []) if p.get("figures")}}
    parts_rec = json.load(open(os.path.join(src_state, "parts.json")))
    from ethoslm import pipeline
    plan_p = os.path.join(src_state, "plan.json")
    plan_parts = {p["name"]: p for p in pipeline.plan_parts(json.load(open(plan_p)))} \
        if os.path.exists(plan_p) else {}
    rows = _rows(parts_rec)
    voices = sorted({r.get("voice") for r in rows if r.get("voice")})
    recipe = {"by_voice": {v: M.recipe_for(v) for v in voices}}
    if roles or whens:
        recipe = narrow(recipe, roles, whens)
        ctx["narrowed"] = {"roles": sorted(roles or []), "when": sorted(whens or []),
                           "why": "a diagnostic arm, not the registered comparison"}
    ctx["voices"] = voices
    ctx["recipe"] = {v: {r: [(x["family"], x["weight"], x["when"]) for x in rows_]
                         for r, rows_ in rec.items()} for v, rec in recipe["by_voice"].items()}
    ctx["how"] = {}
    for p in raw.get("parts") or []:
        for k, v in (p.get("how") or {}).items():
            ctx["how"][k] = ctx["how"].get(k, 0) + v
    site_p = os.path.join(src_state, "site.json")
    centre = None
    if os.path.exists(site_p):
        s = json.load(open(site_p))
        if s.get("origin") and s.get("size"):
            centre = (s["origin"][0] + s["size"] // 2, s["origin"][1] + s["size"] // 2)
    floors = {r["part"]: r.get("floor_y") for r in rows if r.get("floor_y") is not None}
    subs = subjects(rows, plan_parts, sdoc, floors, centre)
    ctx["subjects"] = [{k: v for k, v in s.items() if k != "texture"} for s in subs]

    # the positive control, first: if the display cannot register a substitution the
    # comparison is not evidence about materials
    close = next((s for s in subs if s["view"] == "street_face"), None) or \
        next((s for s in subs if s["view"] == "eye"), None)
    if close is not None:
        ctx["positive_control"] = controls(built, sdoc, close,
                                           os.path.join(out_dir, name, "control"), seed)
        # ...and the same question about the role the recipe under judgement edits, not
        # only about the face's dominant one (composition round)
        roles_edited = sorted({r for rec in (recipe.get("by_voice") or {"": recipe}).values()
                               for r in (rec or {})})
        ctx["positive_control_by_role"] = [
            controls(built, sdoc, close, os.path.join(out_dir, name, "control", r), seed,
                     role_want=r) for r in roles_edited]
    if controls_only:
        return ctx
    ctx["records"] = {}
    ctx["frames"] = {}
    ctx["arms"] = {}
    for setting in settings_list:
        settings = {"condition": setting, "weather_from": M.WEATHER_FROM}
        refs: dict = {}
        for mode in MODES:
            fin, rec = M.apply(built, sdoc, recipe, settings, seed, mode)
            d = os.path.join(out_dir, name, setting, mode)
            imgs = {}
            for s in subs:
                path, img = draw(fin, s, d)
                imgs[s["view"]] = path
                if mode == "none":
                    refs[s["view"]] = img
                else:
                    ctx["frames"].setdefault(f"{setting}/{mode}", {})[s["view"]] = \
                        image_diff(refs[s["view"]], img)
            ctx["images"][f"{setting}/{mode}"] = imgs
            ctx["records"][f"{setting}/{mode}"] = {
                k: rec[k] for k in ("key", "substituted", "stale_skipped",
                                    "restated_from_the_standing_block", "figure_refused",
                                    "counts", "totals", "rates", "bucket_rates",
                                    "matched_rates", "matched_bucket_rates") if k in rec}
            if mode != "none":
                p = M.plan(sdoc, recipe, settings, seed, mode)
                ctx["arms"][f"{setting}/{mode}"] = {
                    "coherence": coherence(p),
                    "condition_fit": condition_fit(p, sdoc, recipe)}
                offline.save_volume(fin, os.path.join(d, "world_finished.npz"))
    weathered = {"condition": "weathered", "weather_from": M.WEATHER_FROM}
    ctx["determinism"] = determinism(built, sdoc, recipe, weathered, seed)
    if not skip_physical:
        fin, _ = M.apply(built, sdoc, recipe, weathered, seed, "contextual")
        ctx["physical"] = physical(src_state, built, fin, parts_rec)
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", action="append", default=[],
                    help="name=path/to/state, repeatable")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "des-material"))
    ap.add_argument("--work", default=os.path.join(ROOT, "out", "des-work", "C"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--reconstruct", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="with --reconstruct: rebuild even where the copy is finished")
    ap.add_argument("--controls-only", action="store_true",
                    help="run the display's positive control and stop")
    ap.add_argument("--re-record", action="store_true",
                    help="re-run the stored programs in a copy even where the state has "
                         "a surfaces.json, and use the new record: needed after a change "
                         "to what the record carries (the composition round's figures)")
    ap.add_argument("--no-physical", action="store_true",
                    help="skip the lint pass (it is the slow half on a city volume)")
    ap.add_argument("--json", default="comparison.json")
    ap.add_argument("--roles", default="",
                    help="diagnostic: keep only these surface roles in the recipe")
    ap.add_argument("--when", default="",
                    help="diagnostic: keep only these variant conditions")
    a = ap.parse_args()
    if not a.context:
        ap.error("at least one --context name=path")
    settings_list = [s.strip() for s in a.settings.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    doc = {"record": "material_comparison", "version": 2,
           "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": a.seed,
           "displays": {
               "close": f"preview.elevation(texture=True) at {TEXEL} texels a block, "
                        f"cropped at the facade plane",
               "row": f"preview.elevation(texture=True) at {ROW_SCALE}",
               "court": f"preview.top_down(texture=True) at {COURT_SCALE}",
               "place": f"preview.preview / top_down, flat, at {PLACE_SCALE}"},
           "contexts": [], "judgment": None, "integrated": False,
           "narrowed": {"roles": a.roles, "when": a.when} if (a.roles or a.when) else None}
    for spec in a.context:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--context {spec!r}: name=path")
        t0 = time.perf_counter()
        ctx = run_context(name, os.path.abspath(path), a.out, a.work, a.seed,
                          settings_list, a.reconstruct, a.rebuild,
                          controls_only=a.controls_only, skip_physical=a.no_physical,
                          roles=[r for r in a.roles.split(",") if r],
                          whens=[w for w in a.when.split(",") if w],
                          re_record=a.re_record)
        ctx["seconds"] = round(time.perf_counter() - t0, 1)
        doc["contexts"].append(ctx)
        print(f"== {name}: owned {ctx.get('owned_cells')} cells; editable "
              f"{json.dumps(ctx.get('editable_cells'))}; {ctx.get('error') or ''}",
              flush=True)
        if ctx.get("reconciliation"):
            print(f"   reconciled {json.dumps({k: v for k, v in ctx['reconciliation'].items() if k != 'how'})}")
        if ctx.get("figures"):
            print(f"   figures {json.dumps({k: v for k, v in ctx['figures'].items() if k != 'by_part'})}")
        for k, r in (ctx.get("records") or {}).items():
            if r.get("figure_refused"):
                print(f"   {k}: {r['figure_refused']} figure cells refused")
        for pc in [ctx.get("positive_control")] + list(ctx.get("positive_control_by_role") or []):
            if not pc:
                continue
            if pc.get("error"):
                print(f"   control role={pc.get('role_asked_for')}: {pc['error']}")
            for c in pc.get("cases") or []:
                print(f"   control {c['role']:8s} {c['control']:6s} {c['display']:8s} "
                      f"{c['from']:22s}-> {c['to']:22s} {c['substituted']:5d} cells -> "
                      f"{json.dumps(c['vs_none'])}")
        for k, r in (ctx.get("records") or {}).items():
            print(f"   {k}: {r.get('substituted')} substituted", flush=True)
        for k, r in (ctx.get("frames") or {}).items():
            print(f"   {k} frames {json.dumps({v: d.get('changed_share') for v, d in r.items()})}")
        for k, r in (ctx.get("arms") or {}).items():
            print(f"   {k} {json.dumps(r)}")
        if ctx.get("determinism"):
            print(f"   determinism {json.dumps({k: v for k, v in ctx['determinism'].items() if k != 'how'})}")
        if ctx.get("physical"):
            print(f"   physical {json.dumps({k: v for k, v in ctx['physical'].items() if k != 'how'})}")
    good = [c for c in doc["contexts"] if not c.get("error") and c.get("determinism")]
    doc["determinism"] = {
        "order_independent": bool(good) and all(c["determinism"]["order_independent"] for c in good),
        "replay_identical": bool(good) and all(c["determinism"]["replay_identical"]
                                               and c["determinism"]["idempotent_on_own_output"]
                                               for c in good),
        "how": "per context: see contexts[].determinism"}
    phys = [c for c in doc["contexts"] if c.get("physical")]
    doc["physical"] = {"clean": bool(phys) and all(c["physical"]["clean"] for c in phys),
                       "how": "per context: see contexts[].physical"}
    p = os.path.join(a.out, a.json)
    json.dump(doc, open(p, "w"), indent=1)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
