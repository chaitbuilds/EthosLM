"""Fixed cameras on a built section, so a rebuild can be photographed from the same place.

    $PY scripts/section_views.py --state out/comp-a --label before
    $PY scripts/section_views.py --state out/comp-a --label after --cameras out/comp-views/cameras.json

The composition round needs matched before/after views of the *same* subjects across a
rebuild that moves geometry. `pipeline/inspect.py`'s `_draw_built` cannot give that: it
recomputes its framing from the current plan every time (the bounding box of every plot,
the largest area for the street view), so the camera moves with the thing it is
photographing and the two frames are not comparable. `scripts/material_compare.py` has
the right model for cached geometry -- a camera is a plain record of
`(kind, rect, facing, floor, scale)` and drawing it on a second volume gives a matched
frame -- but it is a material experiment's harness, not a section reader's.

So: a camera here is a dict, resolved **once** from the plan and the registered section,
written to `cameras.json` beside the frames, and re-used verbatim afterwards. The frames
are drawn by `ethoslm.preview` off `world_built.npz` -- no server, no Chunky, no model
call, a second or two each -- and the manifest records the candidate and the built digest
each frame was taken of, so a frame can never be silently attributed to the wrong world.

`--cameras` re-uses a resolved set; without it the set is resolved from the state's own
plan and written. `--section x0,z0,x1,z1` bounds the overview to the registered section
rather than to everything the plan holds.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

from ethoslm import deps, observe, offline, preview                     # noqa: E402


#: Texels a block gets in a close textured view. `material_compare.TEXEL`, and the same
#: reason: below this the block's own texture is not what you are looking at.
TEXEL = 24

#: How far back a street elevation stands from the frontage it is about, in columns.
STREET_REACH = 26


def _above(vol, keep_below: int = 2):
    """The volume with the subsoil cut, so an elevation is a street and not a cliff.

        `pipeline/inspect.py:_above`, kept here rather than imported because that one is
        private to a stage and this script must not depend on a stage running.
        
    """
    heights = offline.surface_heights(vol)
    floor = int(np.percentile(heights, 5)) - keep_below
    y0 = max(floor - vol.y0, 0)
    return observe.Volume(vol.x0, vol.y0 + y0, vol.z0, vol.codes[:, y0:, :],
                          vol.palette)


def _rect_of(rec) -> list | None:
    for key in ("rect", "at_rect", "bounds"):
        r = rec.get(key)
        if isinstance(r, (list, tuple)) and len(r) == 4:
            return [int(v) for v in r]
    for a, b in (("x0", "z0"), ("X0", "Z0")):
        if a in rec and b in rec:
            return [int(rec["x0"]), int(rec["z0"]), int(rec["x1"]), int(rec["z1"])]
    return None


def _centre(rect) -> tuple:
    return ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)


def _inside(rect, section) -> bool:
    if not section:
        return True
    x0, z0, x1, z1 = rect
    sx0, sz0, sx1, sz1 = section
    return (min(x0, x1) >= min(sx0, sx1) and max(x0, x1) <= max(sx0, sx1)
            and min(z0, z1) >= min(sz0, sz1) and max(z0, z1) <= max(sz0, sz1))


def _walk(plan) -> list:
    """Every leaf of the plan with a rectangle, flattened: `(name, kind, type, rect)`."""
    out = []

    def visit(node, kind=None):
        if isinstance(node, list):
            for n in node:
                visit(n, kind)
            return
        if not isinstance(node, dict):
            return
        rect = _rect_of(node)
        name = node.get("name")
        if rect and name:
            out.append({"name": name, "kind": node.get("kind") or kind or "",
                        "type": node.get("type") or "", "rect": rect,
                        "district": node.get("district") or ""})
        for key in ("districts", "quarters", "plots", "areas", "leaves", "compounds",
                    "parts", "children"):
            if key in node:
                visit(node[key], node.get("kind"))
    visit(plan.get("districts") or [])
    visit(plan.get("compounds") or [])
    visit(plan.get("parts") or [])
    return out


#: How deep a band each side of the boundary a street elevation takes, in columns. A
#: city block in this plan is 40 (lower ring) to 48 (middle ring), so 34 is a little
#: under one block: the frontage on the near side of a street, with what stands behind
#: it reading as depth rather than as a second subject.
STREET_BAND = 34

#: How far along the boundary a gate view reaches each way. Twice a block: enough wall
#: on both sides for the gate to be an event in a wall rather than a lone tower.
GATE_REACH = 90

#: How far along the run a close isometric of one block reaches each way. A lower-ring
#: block is 40 columns, so this is about one and a half blocks: long enough that the
#: rhythm of a street is in the frame and short enough that the frame is of a street and
#: not of a map.
BLOCK_REACH = 30

#: How far past the gate and the anchor the axis view reaches, so the passage is an
#: event on a street rather than the edge of the picture.
AXIS_REACH = 24


def resolve(plan, section=None, sample=None, *,
            market_types=("market", "square", "plaza"),
            court_types=("court_large", "court_small", "courtyard_house", "yard",
                         "garden")) -> list:
    """The camera set for this plan and this registered section.

        Resolved from the plan once. Every camera is a plain record, so the same set drawn
        on a later volume is the same camera and the frames are comparable.
        
    """
    leaves = _walk(plan)
    cams = []
    if section:
        x0, z0, x1, z1 = section
        cams.append({"name": "section_iso", "kind": "iso", "rect": list(section),
                     "keep_below": 6, "scale": 2, "texture": False,
                     "why": "the whole registered section: both fabrics, the boundary "
                            "between them and the passage through it, in one frame"})
        cams.append({"name": "section_top", "kind": "top", "rect": list(section),
                     "scale": 2, "texture": False,
                     "why": "the grain of the two fabrics from above: lot size, "
                            "attachment, court share and street rhythm"})
        mid = (min(x0, x1) + max(x0, x1)) // 2
        cams.append({"name": "section_front", "kind": "elevation", "facing": "north",
                     "rect": [min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1)],
                     "keep_below": 2, "scale": 2, "texture": False,
                     "why": "the section's skyline from outside the outer wall: "
                            "boundary height against the fabric behind it"})
        del mid
    # **The two streets and the passage between them**, derived from the registered
    # boundary rather than from coordinates somebody typed: the boundary's own line is
    # the median of the clipped runs, the gate is the point part standing on it, and a
    # street view is a band one block deep on each side. A frontal elevation of a walled
    # city shows the wall and nothing else, which is true and useless; these are the
    # frames that show whether anybody lives here.
    runs = list((sample or {}).get("boundary_runs") or [])
    named = str(((sample or {}).get("registered") or {}).get("boundary") or "")
    zs = [p[1] for r in runs if not named or r.get("of") == named
          for p in r.get("path") or []]
    if zs and section:
        line = int(sorted(zs)[len(zs) // 2])
        x0, z0, x1, z1 = section
        lo, hi = min(x0, x1), max(x0, x1)
        gates = [p for p in _walk(plan)
                 if "gate" in str(p["name"]) and lo <= p["rect"][0] <= hi]
        gx = (_centre(gates[0]["rect"])[0] if gates else (lo + hi) // 2)
        cams.append({"name": "gate_from_crowded", "kind": "elevation", "facing": "north",
                     "rect": [int(gx) - GATE_REACH, line - STREET_BAND,
                              int(gx) + GATE_REACH, line + 6],
                     "keep_below": 2, "scale": 4, "texture": False,
                     "why": "the boundary and its gate seen from the street on the "
                            "crowded side: what a person walking to the passage sees"})
        for label, sign in (("crowded", -1), ("calm", 1)):
            near = line + sign * (STREET_BAND + 8)
            far = near + sign * STREET_BAND
            cams.append({"name": f"street_{label}", "kind": "elevation",
                         "facing": "north",
                         "rect": [lo, min(near, far), hi, max(near, far)],
                         "keep_below": 2, "scale": 4, "texture": False,
                         "why": f"the {label} side's street frontage, one block deep: "
                                f"rhythm, spacing, roof line and what stands between "
                                f"the houses"})
    # **The frames that show depth and enclosure**, the neighbourhood round's own
    # requirement: "keep old comparison cameras and add framing that shows depth and
    # enclosure; thin elevation panoramas alone are inadequate." An elevation is a
    # section through a street and it answers exactly one question -- what is the rhythm
    # and the roof line along this face. It cannot answer whether the street is a room:
    # how far back the far side stands, whether the lane is enclosed on both sides,
    # whether there is anything behind the frontage. An isometric of a short run, close
    # in, is the frame that does, and it is the same cheap draw. Every one of these is
    # derived from the registered boundary and the plan, like the elevations above, so
    # the set is reproducible and the frames are comparable.
    if zs and section:
        line = int(sorted(zs)[len(zs) // 2])
        x0, z0, x1, z1 = section
        lo, hi = min(x0, x1), max(x0, x1)
        gates = [p for p in _walk(plan)
                 if "gate" in str(p["name"]) and lo <= p["rect"][0] <= hi]
        gx = int(_centre(gates[0]["rect"])[0]) if gates else (lo + hi) // 2
        for label, sign in (("crowded", -1), ("calm", 1)):
            near = line + sign * (STREET_BAND + 8)
            far = near + sign * STREET_BAND
            # one block of fabric with the lane in front of it and the lane behind,
            # short enough along the run that a person is looking at a street and not at
            # a map of one
            cams.append({
                "name": f"block_{label}", "kind": "iso",
                "rect": [gx - BLOCK_REACH, min(near, far) - 8,
                         gx + BLOCK_REACH, max(near, far) + 8],
                "keep_below": 2, "scale": 5, "texture": False,
                "why": (f"one block of the {label} fabric close in, with the lane in "
                        f"front of it and the ground behind: whether the street is a "
                        f"room, how deep the frontage stands and what is between the "
                        f"houses -- the questions an elevation cannot answer")})
        # **the gate to the anchor**, along the axis the section is registered around:
        # the passage, the street it lands on and the market it leads to, in one frame
        mk = [l for l in leaves if l["type"] in market_types and _inside(l["rect"], section)]
        mk.sort(key=lambda l: -abs((l["rect"][2] - l["rect"][0])
                                   * (l["rect"][3] - l["rect"][1])))
        if mk:
            m = mk[0]
            mz = int(_centre(m["rect"])[1])
            cams.append({
                "name": "gate_to_market", "kind": "iso",
                "rect": [min(gx, int(m["rect"][0])) - AXIS_REACH,
                         min(line, mz) - 12,
                         max(gx, int(m["rect"][2])) + AXIS_REACH,
                         max(line, mz) + 12],
                "keep_below": 4, "scale": 4, "texture": False,
                "about": m["name"],
                "why": ("the passage through the boundary, the street it lands on and "
                        "the market it leads to, in one frame: whether the anchor "
                        "belongs to the neighbourhood around it or merely stands in it")})
    # the anchor: the largest market/square/plaza inside the section
    anchors = [l for l in leaves if l["type"] in market_types and _inside(l["rect"], section)]
    anchors.sort(key=lambda l: -abs((l["rect"][2] - l["rect"][0])
                                    * (l["rect"][3] - l["rect"][1])))
    if anchors:
        a = anchors[0]
        cams.append({"name": "anchor_top", "kind": "top", "rect": a["rect"], "pad": 6,
                     "scale": TEXEL // 2, "texture": True, "about": a["name"],
                     "why": f"the working anchor from above ({a['type']}): its floor, "
                            "its aisle and the way in from the street"})
        cams.append({"name": "anchor_face", "kind": "elevation", "facing": "north",
                     "rect": a["rect"], "pad": 4, "keep_below": 2, "scale": 6,
                     "texture": False, "about": a["name"],
                     "why": "the anchor and the frontage that encloses it"})
    # a court, for the enclosed-and-entered reading
    courts = [l for l in leaves if l["type"] in court_types and _inside(l["rect"], section)]
    courts.sort(key=lambda l: -abs((l["rect"][2] - l["rect"][0])
                                   * (l["rect"][3] - l["rect"][1])))
    if courts:
        c = courts[0]
        cams.append({"name": "court_top", "kind": "top", "rect": c["rect"], "pad": 4,
                     "scale": TEXEL // 2, "texture": True, "about": c["name"],
                     "why": f"one court ({c['type']}): enclosed by its ranges, entered "
                            "from the street, not spent to fit a lot"})
    return cams


def draw(vol, cam, out_dir: str) -> dict:
    """One camera, one PNG. The record of what was drawn, including the pixel size."""
    t0 = time.time()
    x0, z0, x1, z1 = cam["rect"]
    sub = preview.crop(vol, x0, z0, x1, z1, pad=int(cam.get("pad", 0)))
    kind = cam["kind"]
    if kind == "iso":
        sub = _above(sub, int(cam.get("keep_below", 6)))
        img = preview.preview(sub, scale=int(cam.get("scale", 2)),
                              grey=bool(cam.get("grey")), clip=False)
    elif kind == "top":
        img = preview.top_down(sub, scale=int(cam.get("scale", 2)),
                               texture=bool(cam.get("texture")))
    elif kind == "elevation":
        sub = _above(sub, int(cam.get("keep_below", 2)))
        img = preview.elevation(sub, facing=cam.get("facing", "south"),
                                scale=int(cam.get("scale", 2)), clip=False,
                                texture=bool(cam.get("texture")))
    else:
        raise ValueError(f"unknown camera kind {kind!r}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cam['name']}.png")
    cv2.imwrite(path, img[:, :, ::-1])
    return {"camera": cam["name"], "path": path, "px": [int(img.shape[1]),
                                                        int(img.shape[0])],
            "seconds": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="a state directory holding "
                                                   "world_built.npz and plan.json")
    ap.add_argument("--label", required=True, help="before | after | a candidate id")
    ap.add_argument("--out", default="", help="default out/comp-views/<label>")
    ap.add_argument("--cameras", default="", help="a resolved set to re-use verbatim")
    ap.add_argument("--section", default="", help="x0,z0,x1,z1 of the registered section")
    ap.add_argument("--volume", default="", help="a volume other than the artifact below")
    ap.add_argument("--artifact", default="",
                    help="world_built.npz | world_finished.npz. With neither this nor "
                         "--volume, the finished world is used where the round wrote "
                         "one; the manifest always records which was read.")
    a = ap.parse_args()

    state = os.path.abspath(a.state)
    out_dir = os.path.abspath(a.out or os.path.join("out", "comp-views", a.label))
    os.makedirs(out_dir, exist_ok=True)
    # **Which artifact is being delivered, chosen rather than defaulted.** The spatial
    # design round, and the neighbourhood review's last paragraph: "current default
    # views/checks still use `world_built.npz`, while material work writes
    # `world_finished.npz`. Select and inspect the actual delivered artifact when
    # finishing is used." `--artifact` names it; with none, the finished world is used
    # where the round wrote one and the structural world otherwise, and either way the
    # manifest says which, so a reader never has to infer it from a filename.
    vol_path = a.volume or os.path.join(state, a.artifact or "")
    if not a.volume and not a.artifact:
        finished = os.path.join(state, "world_finished.npz")
        vol_path = finished if os.path.exists(finished) \
            else os.path.join(state, "world_built.npz")
    artifact = os.path.basename(vol_path)
    vol = offline.load_volume(vol_path)
    plan = json.load(open(os.path.join(state, "plan.json")))
    section = [int(v) for v in a.section.split(",")] if a.section else None

    sample = None
    pp = os.path.join(state, "parts.json")
    if os.path.exists(pp):
        sample = (json.load(open(pp)) or {}).get("sample")
    if a.cameras:
        cams = json.load(open(a.cameras))["cameras"]
    else:
        cams = resolve(plan, section, sample)
        with open(os.path.join(out_dir, "cameras.json"), "w") as f:
            json.dump({"record": "section_cameras", "of": state, "section": section,
                       "cameras": cams}, f, indent=1)

    frames = [draw(vol, c, out_dir) for c in cams]
    man = {"record": "section_views", "label": a.label, "of": state,
           "volume": vol_path, "section": section,
           "artifact": artifact,
           "artifact_why": ("named on the command line" if (a.artifact or a.volume) else
                            ("the finished world: material finishing wrote one and it is "
                             "what this round delivers" if artifact == "world_finished.npz"
                             else "the structural world. "
                                  "artifact, and that is what is delivered")),
           "built_digest": deps.content_print(vol_path),
           "candidate": (json.load(open(os.path.join(state, "parts.json")))
                         .get("candidate") if os.path.exists(
                             os.path.join(state, "parts.json")) else None),
           "cameras": cams, "frames": frames, "t": time.time()}
    with open(os.path.join(out_dir, "views.json"), "w") as f:
        json.dump(man, f, indent=1)
    for fr in frames:
        print(f"  {fr['camera']:16s} {fr['px'][0]:5d}x{fr['px'][1]:<5d} "
              f"{fr['seconds']:5.2f}s  {fr['path']}")
    print(f"-> {os.path.join(out_dir, 'views.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
