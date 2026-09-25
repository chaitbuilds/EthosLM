#!/usr/bin/env python3
"""**Courtyard houses and shop houses joined wall to wall along one lane.**

    $PY scripts/test_attached_forms.py [name-fragment]

The fabric reset round, form worker. A hutong is "formed by lines of siheyuan" joined one
to another along a lane, and a market street is a continuous terrace of shop houses; both
types declare `ATTACHED` and this is the focused check that they are built that way, not
merely allowed to be. Offline, no cached world, no model call: each row is sited and
built through `Builder.site()` / `type_builder()` exactly as `scripts/type_needs.py`
stands an instance, on lots cut the way `district_compile` cuts a terraced run (the pad
inset `PAD_SITE_INSET` on free sides and 0 on attached ones, `site_pad_rect`), with the
compiled site's door in the middle of the pad's street edge and its landing on the lane.

    a1  four courtyard houses 15, 13, 17 and 15 wide by 17 deep, ends free, on a plane
    a2  the same row on a gentle bank falling along the lane
    a3  four shop houses 8, 7, 9 and 8 wide by 13 deep, asked 2, 2, 1 and 2 storeys and
        mixed trades, on a plane; each must stand the storeys asked and be open to the
        lane over its counter beside the door
    a4  the same shop row on the bank

(front north here; `build_row(front=...)` also stands a run on the other three fronts,
which the form worker swept with mixed widths, depths, seeds and parameters) and each asserts:
every part stands (no refusal, no `SiteRefused`); the build family of
the linter reports no error on the row's plots, with the lane as the network (so E002 is
"the door cannot be reached on foot from the lane"); no gap column between neighbours
below the eave, anywhere along the party wall and at the lane face; no block written by a
part's `build()` outside its own plot except the door approach in front of it; for a
courtyard house, the court is open to the sky and at least 5x5 on a 15x17 lot, and the
lane face is blank wall apart from the gate (no window or opening in it). Pictures of each row (isometric, top-down,
lane elevation) go to `out/fr-work-types/`.

**What fails here today is not the types.** The run is built in the production order --
`site()` then `build()`, lot after lot, as `instantiated_source` composes it -- and
`Builder._site_lay` lays its pad's one-column ledge on every side, attached ones
included: it fills the neighbour's column to the floor and cuts it to `floor + 5`, so
each lot's `site()` takes the party wall its already-built neighbour stood on the plot
line (a one-column slot through the lane face, the depth of the house; the same 72 cells
in a row of four `row_house` terraces). `--site-first` sites the whole run before
building any of it, which isolates the types from that: every case passes. The fix is
the library's (the ledge and its cut kept off the sides `part["attached"]` names) and is
reported, not made, by the form worker.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from ethoslm import lint, offline, pipeline, stages  # noqa: E402
from ethoslm.buildlib import Builder, SiteRefused, site_pad_rect  # noqa: E402
from ethoslm.circulate import Network, Threshold  # noqa: E402
from ethoslm.frontage import Frontage  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

OUT = os.path.join(ROOT, "out", "fr-work-types")
VOICE = "ochre_stone_green_tile"
GROUND = 64
SIZE = 96
PLOT_Z0 = 19
X_START = 14
#: the bank: a fall along the lane, so neighbours stand at different floors
BANK_FALL = 4

#: `--site-first` sites every lot of a run before building any of them: a diagnostic
#: only, which separates what a type builds from what a later neighbour's `site()` does
#: to it. The production call (`instantiated_source`) sites and builds one instance
#: after another, and that order is the default and what the cases assert.
SITE_FIRST = "--site-first" in sys.argv

CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the frame

#: For each street side: the flank on the low and the high side of the run, the axis the
#: run goes along, and the walk-in facing (from the lane into the lot).
FRONTS = {"north": ("west", "east", "x", "south"), "south": ("west", "east", "x", "north"),
          "west": ("north", "south", "z", "east"), "east": ("north", "south", "z", "west")}


def to_world(front: str, a: int, d: int, depth: int) -> tuple:
    """(x, z) of the cell `a` along the run and `d` into the lot from its street edge
    (negative `d` is the lane side)."""
    far = PLOT_Z0 + depth - 1
    if front == "north":
        return a, PLOT_Z0 + d
    if front == "south":
        return a, far - d
    if front == "west":
        return PLOT_Z0 + d, a
    return far - d, a


def rect_world(front, a0, d0, a1, d1, depth) -> tuple:
    p, q = to_world(front, a0, d0, depth), to_world(front, a1, d1, depth)
    return min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])


def along(front: str, rect) -> tuple:
    """(a0, a1) of a world rect along the run."""
    x0, z0, x1, z1 = rect
    return (x0, x1) if FRONTS[front][2] == "x" else (z0, z1)


def across(front: str, rect) -> tuple:
    """(lo, hi) of a world rect across the run, in world coordinates."""
    x0, z0, x1, z1 = rect
    return (z0, z1) if FRONTS[front][2] == "x" else (x0, x1)


def cell(front: str, a: int, c: int) -> tuple:
    """(x, z) of along-coordinate `a` and across (world) coordinate `c`."""
    return (a, c) if FRONTS[front][2] == "x" else (c, a)


# ------------------------------------------------------------------ the ground

def _ground(kind: str, front: str = "north") -> np.ndarray:
    """Ground height per column: level, or falling BANK_FALL blocks along the run
    (level either side of it)."""
    h = np.full((SIZE, SIZE), GROUND, np.int32)
    if kind == "bank":
        lo, hi = X_START, X_START + 62
        for a in range(SIZE):
            t = min(max((a - lo) / float(hi - lo), 0.0), 1.0)
            g = GROUND - int(round(BANK_FALL * t))
            if FRONTS[front][2] == "x":
                h[a, :] = g
            else:
                h[:, a] = g
    return h


def _volume(h: np.ndarray) -> Volume:
    y0 = GROUND - BANK_FALL - 14
    codes = np.zeros((SIZE, 76, SIZE), np.uint16)
    for x in range(SIZE):
        for z in range(SIZE):
            g = int(h[x, z]) - y0
            codes[x, :g, z] = 3
            codes[x, g, z] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


# ------------------------------------------------------------------ the row

def _row(type_name: str, widths, depth: int, h: np.ndarray, front: str = "north") -> list:
    """The leaves of one terraced run, lot against lot, fronting the lane on `front`,
    each with the compiled site `district_compile._leaf_site` would give it: the inset
    pad, the door in the middle of the pad's street edge, the landing on the lane."""
    lo_side, hi_side, _ax, _walk = FRONTS[front]
    leaves, a = [], X_START
    n = len(widths)
    for i, w in enumerate(widths):
        att = ([] if i == 0 else [lo_side]) + ([] if i == n - 1 else [hi_side])
        x0, z0, x1, z1 = rect_world(front, a, 0, a + w - 1, depth - 1, depth)
        lf = {"label": f"{type_name[:4]}_{i}", "kind": "plot", "x0": x0, "z0": z0,
              "x1": x1, "z1": z1, "front": front, "attached": list(att),
              "type": type_name}
        pad = list(site_pad_rect(x0, z0, x1, z1, att))
        pa0, pa1 = along(front, pad)
        da = (pa0 + pa1) // 2
        c_lo, c_hi = across(front, pad)
        edge = c_lo if front in ("north", "west") else c_hi
        land = to_world(front, da, -1, depth)
        door = cell(front, da, edge)
        floor = int(h[land[0], land[1]])
        lf["site"] = {"pad": pad, "floor": floor, "facing": front,
                      "door": [door[0], door[1]], "landing": [land[0], land[1]],
                      "attached": sorted(att), "why": "test_attached_forms"}
        leaves.append(lf)
        a += w
    return leaves


def _network(leaves, h, front: str = "north", depth: int = 17) -> Network:
    """The lane: two rows along the run's whole front and three columns past each end,
    and one threshold per leaf at its compiled landing with its compiled door."""
    cells, ths = {}, []
    a0 = min(along(front, (lf["x0"], lf["z0"], lf["x1"], lf["z1"]))[0] for lf in leaves)
    a1 = max(along(front, (lf["x0"], lf["z0"], lf["x1"], lf["z1"]))[1] for lf in leaves)
    for a in range(a0 - 3, a1 + 4):
        for d in (-2, -1):
            x, z = to_world(front, a, d, depth)
            cells[(x, z)] = {"y": int(h[x, z]), "rank": 1, "face": None}
    for lf in leaves:
        st = lf["site"]
        lx, lz = st["landing"]
        ths.append(Threshold(id=lf["label"], x=lx, z=lz, y=int(h[lx, lz]),
                             facing=FRONTS[front][3],
                             door=(st["door"][0], int(st["floor"]) + 1, st["door"][1])))
    return Network(cells=cells, thresholds=ths)


class _Registry:
    def __init__(self, leaves):
        self.plots = [{"label": lf["label"], "x0": lf["x0"], "z0": lf["z0"],
                       "x1": lf["x1"], "z1": lf["z1"]} for lf in leaves]
        self.claimed_this_pass = [dict(p) for p in self.plots]

    def plots_list(self):
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k):
        return True


def _decl(type_name: str) -> dict:
    path = os.path.join(ROOT, "types", f"{type_name}.py")
    ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
    exec(compile(open(path).read(), path, "exec"), ns)            # noqa: S102
    return ns


def build_row(type_name: str, widths, depth: int, ground: str, params_list,
              seeds=None, front: str = "north") -> dict:
    """Site and build one run. Returns everything the checks read."""
    h = _ground(ground, front)
    vol = _volume(h)
    leaves = _row(type_name, widths, depth, h, front)
    net = _network(leaves, h, front, depth)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net)
    b.registry = _Registry(leaves)
    mat, roof = pipeline.voice_palette(VOICE), pipeline.voice_roof(VOICE)
    ns = _decl(type_name)
    parts, results, writes, refused = [], [], [], []
    sited = {}

    def _site(i, lf):
        geo = {k: lf[k] for k in pipeline.PART_GEOMETRY if k in lf}
        try:
            sited[i] = b.site(dict(geo), mat=mat, roof=roof)
        except SiteRefused as e:
            sited[i] = None
            refused.append((lf["label"], f"SiteRefused: {e}"))

    if SITE_FIRST:
        for i, lf in enumerate(leaves):
            _site(i, lf)
    for i, lf in enumerate(leaves):
        if not SITE_FIRST:
            _site(i, lf)
        p = sited[i]
        if p is None:
            parts.append(None)
            results.append(None)
            writes.append({})
            continue
        before = dict(b._pending)
        seed = (seeds or [3 + i for i in range(len(leaves))])[i]
        res = ns["build"](b.type_builder(p, role=ns.get("ROLE")), p, seed,
                          **dict(params_list[i]))
        mine = {k: (before.get(k), v) for k, v in b._pending.items()
                if before.get(k) != v}
        if not (isinstance(res, dict) and res.get("ok", True) is not False):
            refused.append((lf["label"], (res or {}).get("reason")))
        parts.append(p)
        results.append(res)
        writes.append(mine)
    b.resolve_steps()
    built = stages.apply_pending(vol, b._pending)
    return {"type": type_name, "ground": ground, "h": h, "front": front, "vol": vol, "built": built,
            "leaves": leaves, "net": net, "parts": parts, "results": results,
            "writes": writes, "refused": refused, "b": b}


# ------------------------------------------------------------------ the checks

AIRS = ("air", "cave_air", "void_air")
#: what `building()`'s `clear_ground_cover` takes off the ground two columns round a
#: footprint (`prims.VEGETATION`): turf, not anybody's building
COVER = ("grass_block", "grass", "short_grass", "tall_grass", "fern", "dandelion",
         "poppy")
#: what fills an opening: a wall made of these is not blank
GLAZING = ("glass", "pane", "bars", "trapdoor", "fence")


def _name(state: str) -> str:
    return state.split("[")[0].split(":")[-1]


def _solid_at(vol: Volume, x: int, y: int, z: int) -> bool:
    if not vol.inside(x, y, z):
        return False
    return _name(vol.state(x, y, z)) not in AIRS


def lint_errors(row: dict) -> list:
    plots = [{"label": lf["label"], "x0": lf["x0"], "z0": lf["z0"], "x1": lf["x1"],
              "z1": lf["z1"]} for lf in row["leaves"]]
    x0 = min(p["x0"] for p in plots) - 6
    x1 = max(p["x1"] for p in plots) + 6
    z0 = min(p["z0"] for p in plots) - 6
    z1 = max(p["z1"] for p in plots) + 6
    ctx = lint.Context.build(row["built"], plots=plots, region=(x0, z0, x1, z1),
                             network=row["net"], fittings=row["b"].fitting_cells)
    rep = lint.lint(ctx, family=lint.BUILD).within(plots)
    return [f for f in rep.findings if f.code.startswith("E")]


def _eave(row: dict, i: int) -> int:
    res = row["results"][i] or {}
    em = res.get("emitted") or {}
    floors = em.get("floors") or [row["parts"][i]["floor_y"]]
    return int(max(floors)) + Builder.STOREY


def party_gaps(row: dict) -> list:
    """Every (x, y, z) between two neighbours' pads, or on the party line itself, that
    is open below the lower neighbour's eave, along the depth both houses share."""
    gaps = []
    parts, f = row["parts"], row["front"]
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if a is None or b is None:
            continue
        pa = (a["x0"], a["z0"], a["x1"], a["z1"])
        pb = (b["x0"], b["z0"], b["x1"], b["z1"])
        line = range(along(f, pa)[1], along(f, pb)[0] + 1)
        ra = ((row["results"][i] or {}).get("emitted") or {}).get("rects", {}).get("main")
        rb = ((row["results"][i + 1] or {}).get("emitted") or {}).get("rects", {}).get("main")
        c0 = max(across(f, ra or pa)[0], across(f, rb or pb)[0])
        c1 = min(across(f, ra or pa)[1], across(f, rb or pb)[1])
        top = min(_eave(row, i), _eave(row, i + 1))
        for c in range(c0, c1 + 1):
            for y in range(max(a["floor_y"], b["floor_y"]) + 1, top):
                # the party line is two columns (one each) or one: it must be closed in
                # every column between the two pads
                if not all(_solid_at(row["built"], *_xyz(f, al, y, c)) for al in line):
                    gaps.append(_xyz(f, min(line), y, c))
    return gaps


def _xyz(front: str, a: int, y: int, c: int) -> tuple:
    x, z = cell(front, a, c)
    return x, y, z


def strays(row: dict) -> list:
    """Non-air blocks a part's `build()` wrote outside its own plot, bar the strip in
    front of it (the door approach and the doorstep, which is the lane side)."""
    out = []

    def doorstep(x, y, z):
        """Ground-level work in front of **any** lot of the run: the approaches `site()`
        queues are resolved by whichever call next resolves the queue -- `shop_house`
        does so laying its own flight -- so a neighbour's treads can land in this
        part's writes. They are the library's, at the foot of a lot, never a roof."""
        for lf2, p2 in zip(row["leaves"], row["parts"]):
            f2 = lf2["front"]
            r2 = (lf2["x0"], lf2["z0"], lf2["x1"], lf2["z1"])
            a0, a1 = along(f2, r2)
            a_, c_ = (x, z) if FRONTS[f2][2] == "x" else (z, x)
            lo, hi = across(f2, r2)
            ahead = (c_ < lo or (p2 and c_ < across(f2, (p2["x0"], p2["z0"], p2["x1"],
                                                        p2["z1"]))[0])) \
                if f2 in ("north", "west") else \
                (c_ > hi or (p2 and c_ > across(f2, (p2["x0"], p2["z0"], p2["x1"],
                                                     p2["z1"]))[1]))
            if ahead and a0 <= a_ <= a1 and p2 and y <= p2["floor_y"] + 1:
                return True
        return False

    for lf, mine in zip(row["leaves"], row["writes"]):
        for (x, y, z), (was, st) in mine.items():
            if lf["x0"] <= x <= lf["x1"] and lf["z0"] <= z <= lf["z1"]:
                continue
            f = lf["front"]
            a0, a1 = along(f, (lf["x0"], lf["z0"], lf["x1"], lf["z1"]))
            a_, c_ = (x, z) if FRONTS[f][2] == "x" else (z, x)
            c_lo, c_hi = across(f, (lf["x0"], lf["z0"], lf["x1"], lf["z1"]))
            ahead = c_ < c_lo if f in ("north", "west") else c_ > c_hi
            if ahead and a0 <= a_ <= a1:
                continue                # in front: the approach to its own door
            if doorstep(x, y, z):
                continue
            if _name(st) in AIRS:
                prev = was if was is not None else (
                    row["vol"].state(x, y, z) if row["vol"].inside(x, y, z) else "air")
                if _name(prev) in AIRS or _name(prev) in COVER:
                    continue            # air over air, or the turf `building()` lifts
            out.append((lf["label"], (x, y, z), _name(st)))
    return out


def court_open(row: dict, i: int) -> tuple:
    """(w, d, closed cells) of the i-th courtyard house's court: every column of the
    emitted court must be open to the sky above its floor."""
    em = (row["results"][i] or {}).get("emitted") or {}
    yard = (em.get("court") or {}).get("yard")
    if not yard:
        return 0, 0, ["no court emitted"]
    x0, z0, x1, z1 = yard
    fy = row["parts"][i]["floor_y"]
    vol = row["built"]
    closed = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            for y in range(fy + Builder.STOREY, vol.y0 + vol.shape[1]):
                if _solid_at(vol, x, y, z):
                    closed.append((x, y, z))
                    break
    return x1 - x0 + 1, z1 - z0 + 1, closed


def lane_face_openings(row: dict) -> list:
    """Openings in the lane face of a courtyard run below its eave, other than a gate.
    The lane face is each house's front wall row; a door and its frame are the gate."""
    out = []
    for i, p in enumerate(row["parts"]):
        em = (row["results"][i] or {}).get("emitted") or {}
        main = (em.get("rects") or {}).get("main")
        gate = (em.get("rects") or {}).get("gate")
        if not main:
            continue
        f = row["front"]
        c_lo, c_hi = across(f, main)
        c = c_lo if f in ("north", "west") else c_hi
        a0, a1 = along(f, main)
        for a in range(a0, a1 + 1):
            x, z = cell(f, a, c)
            if gate and (x, z) == (gate[0], gate[1]):
                continue
            for y in range(p["floor_y"] + 1, _eave(row, i)):
                st = _name(row["built"].state(x, y, z))
                if st in AIRS or any(g in st for g in GLAZING):
                    out.append((p["label"], (x, y, z), st))
    return out


def shop_openings(row: dict) -> list:
    """(label, columns of the body's street face open over the counter): the shop is
    open to the lane, not only its door."""
    out = []
    for i, p in enumerate(row["parts"]):
        em = (row["results"][i] or {}).get("emitted") or {}
        main = (em.get("rects") or {}).get("main")
        if not (p and main):
            out.append((row["leaves"][i]["label"], 0))
            continue
        f = row["front"]
        c_lo, c_hi = across(f, main)
        c = c_lo if f in ("north", "west") else c_hi
        a0, a1 = along(f, main)
        door = tuple(row["leaves"][i]["site"]["door"])
        n = 0
        for a in range(a0, a1 + 1):
            x, z = cell(f, a, c)
            if (x, z) == door:
                continue
            st = _name(row["built"].state(x, p["floor_y"] + 2, z))
            if st in AIRS or any(k in st for k in ("lantern", "torch", "candle")):
                n += 1
        out.append((row["leaves"][i]["label"], n))
    return out


def door_leaves(row: dict) -> list:
    """(label, door cell) where the door leaf is on the lane face of its own pad."""
    got = []
    for lf, p in zip(row["leaves"], row["parts"]):
        dx, dz = lf["site"]["door"]
        st = row["built"].state(dx, p["floor_y"] + 1, dz) if p else "air"
        got.append((lf["label"], _name(st)))
    return got


def pictures(row: dict, tag: str) -> list:
    """An isometric, a top-down and the lane elevation of the run, cropped to it."""
    import cv2
    from ethoslm import preview
    os.makedirs(OUT, exist_ok=True)
    lv = row["leaves"]
    sub = preview.crop(row["built"], min(p["x0"] for p in lv), min(p["z0"] for p in lv),
                       max(p["x1"] for p in lv), max(p["z1"] for p in lv), pad=4)
    # from three courses under the lowest floor up: the ground under the fixture is not
    # the subject
    y_lo = min(p["floor_y"] for p in row["parts"] if p) - 3 - sub.y0
    sub = Volume(sub.x0, sub.y0 + y_lo, sub.z0, sub.codes[:, y_lo:, :], sub.palette)
    paths = []
    for name, img in (("iso", preview.preview(sub, scale=5)),
                      ("top", preview.top_down(sub, scale=6)),
                      ("lane", preview.elevation(sub, facing=row["front"], scale=5))):
        path = os.path.join(OUT, f"{tag}{'_sitefirst' if SITE_FIRST else ''}_{name}.png")
        cv2.imwrite(path, img[:, :, ::-1])
        paths.append(os.path.relpath(path, ROOT))
    return paths


def _assert_row(row: dict, tag: str) -> str:
    pics = pictures(row, tag)           # first, so a failing run still leaves its picture
    assert not row["refused"], row["refused"]
    errs = lint_errors(row)
    assert not errs, [f"{f.code} {f.message}" for f in errs[:6]] + [f"({len(errs)} in all)"]
    gaps = party_gaps(row)
    assert not gaps, (f"{len(gaps)} open cell(s) on a party line below the eave", gaps[:6])
    out = strays(row)
    assert not out, (f"{len(out)} block(s) outside their own plot", out[:8])
    doors = door_leaves(row)
    assert all(n.endswith("_door") for _, n in doors), doors
    em = [((r or {}).get("emitted") or {}) for r in row["results"]]
    return (f"{len(row['parts'])} stand, 0 lint errors, 0 party-line gaps, 0 strays; "
            f"storeys {[e.get('storeys') for e in em]}; pictures {pics[0]} (+top, lane)")


# ------------------------------------------------------------------ the cases

COURT_WIDTHS = (15, 13, 17, 15)
COURT_DEPTH = 17
COURT_PARAMS = [{"storeys": 1, "screen": s} for s in ("wall", "planted", "wall", "none")]
SHOP_WIDTHS = (8, 7, 9, 8)
SHOP_DEPTH = 13
SHOP_PARAMS = [{"storeys": 2, "trade": "tea"}, {"storeys": 2, "trade": "cloth"},
               {"storeys": 1, "trade": "grain"}, {"storeys": 2, "trade": "smith"}]


def _courts(ground: str) -> str:
    row = build_row("courtyard_house", COURT_WIDTHS, COURT_DEPTH, ground, COURT_PARAMS)
    said = _assert_row(row, f"court_row_{ground}")
    sizes = []
    for i, w in enumerate(COURT_WIDTHS):
        cw, cd, closed = court_open(row, i)
        assert not closed, (row["leaves"][i]["label"], "court roofed over", closed[:4])
        if (w, COURT_DEPTH) == (15, 17):
            assert cw >= 5 and cd >= 5, (row["leaves"][i]["label"], "court", cw, cd)
        sizes.append(f"{cw}x{cd}")
    opens = lane_face_openings(row)
    assert not opens, (f"{len(opens)} opening(s) in the lane face beside the gate",
                       opens[:6])
    feats = [sorted(k for k, v in ((r or {}).get("emitted") or {}).get("features", {}).items()
                    if v is True) for r in row["results"]]
    return said + f"; courts {sizes}; blank lane face but the gate; features {feats[0]}"


def _shops(ground: str) -> str:
    row = build_row("shop_house", SHOP_WIDTHS, SHOP_DEPTH, ground, SHOP_PARAMS)
    said = _assert_row(row, f"shop_row_{ground}")
    opens = shop_openings(row)
    assert all(n >= 1 for _, n in opens), ("a shop with no opening to the lane", opens)
    em = [((r or {}).get("emitted") or {}) for r in row["results"]]
    got = [e.get("storeys") for e in em]
    assert got == [p["storeys"] for p in SHOP_PARAMS], ("storeys asked, stood",
                                                        [p["storeys"] for p in SHOP_PARAMS], got)
    return said + (f"; shop openings over the counter {[n for _, n in opens]}; "
                   f"asked {[p['storeys'] for p in SHOP_PARAMS]} storeys, stood {got}")


@case
def a1_courtyard_houses_joined_along_a_lane_on_a_plane():
    return _courts("flat")


@case
def a2_courtyard_houses_joined_along_a_lane_on_a_bank():
    return _courts("bank")


@case
def a3_shop_houses_in_a_terrace_on_a_plane():
    return _shops("flat")


@case
def a4_shop_houses_in_a_terrace_on_a_bank():
    return _shops("bank")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    ok = fail = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__
        if only and only not in name:
            continue
        try:
            said = fn()
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        except Exception as e:                   # noqa: BLE001 -- reported
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} attached form cases pass ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
