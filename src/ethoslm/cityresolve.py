"""**The design compiler**: a `citydesign` record resolved into walls, streets, blocks,
lots, compounds, designed ground and construction regions.

The design synthesis round. The model designs organisation (`ethoslm.citydesign`); this
resolves it, deterministically and without reading prose:

1. **Boundary and rings.** One `boundary.Outline` at the extent's radius; each ring is
   the outline scaled to its `outer` fraction, so walls, owned land and ring roads are
   one geometry. Ring land is a raster of the outline's ratio, so a round city owns its
   round land to the column -- no square district crosses a round wall, no corner wedge
   is left unowned.
2. **Walls and gates.** `Outline.polyline` (axial and 45-degree runs, a straight run
   pinned at each gate) for every walled ring; the wall's footprint is rasterized with the
   same arithmetic as `Builder._decide_edge`, so the land a ring owns stops where its wall
   actually stands.
3. **Streets.** Ring roads along the rings' edges, radials on the gate bearings, and each
   grain's own grid (lanes and streets at the spacings its lots need, anchored on the
   centre so radials are grid lines). Streets are rasters with ranks.
4. **Blocks and lots.** Connected land between streets, packed by the grain's composer
   against the forms' own lot sizes (`formplan`): back-to-back courtyard rows on lanes,
   shop terraces on streets, stepped frontage along a curved ring road, estates composed
   as walled multi-court compounds. What cannot hold a lot is owned open ground
   (gardens, groves, yards), never an unowned remainder.
5. **The monument.** The axial sequence the design asked for, dimensioned into the
   ring's land with the wings composed by the same compound composer as the estates.
6. **Ground.** A designed level and treatment per column (`designground` codes): one
   terrace per terraced ring rising at its walls, a podium, graded or preserved land,
   ramps on the radials at the gates. Costs are estimated for comparison.
7. **Regions.** Every block, lot and wall piece is owned by one construction region, so a
   region is built alone with its neighbours as context and extended without rework.

Everything the compiler could not honour is a finding addressed to a design field.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

import numpy as np

from . import boundary as B
from . import citydesign as CD

# ---------------------------------------------------------------- raster codes
OUT, WALL, ROAD, LANE, LAND, LOT, OPEN, WATER, COURT, HALL, GATE = range(11)
USE_NAMES = ["out", "wall", "road", "lane", "land", "lot", "open", "water", "court",
             "hall", "gate"]
#: designground treatment codes (see `designground`); duplicated so a compile does not
#: depend on the module that applies it
KEEP, SOFT, HARD, STREET, WATER_T, WALL_T = 0, 1, 2, 3, 4, 5
NO_TARGET = -32768

#: how far a wall's apron reaches either side of its band (its towers' reach)
APRON = 7
#: construction region side, in blocks; a region is a unit of bounded build and resume
REGION = 192
#: how far outside the boundary the frame reaches (context for walls and views)
MARGIN = 24
#: a leftover piece of land smaller than this is paved into its street, not a garden
OPEN_LEAST = 30
#: the widest an area type is built (`types/garden.py` etc. declare 48)
AREA_MAX = 44

#: people per dwelling form, for the capacity table only (an estimate, stated as such)
OCCUPANTS = {"courtyard_house": 10, "court_large": 16, "court_small": 8, "row_house": 5,
             "shop_house": 6, "farmstead": 6, "ceremonial_hall": 0, "estate": 30}

_FRONT_OUT = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
_FLANKS = {"north": ("west", "east"), "south": ("west", "east"),
           "west": ("north", "south"), "east": ("north", "south")}
_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _rng(*key) -> random.Random:
    return random.Random(int(hashlib.sha256(repr(key).encode()).hexdigest()[:12], 16))


def _lineage(city) -> str:
    """The identity a design's random choices are seeded by: its `lineage` (the design
    it revises, so a revision keeps the accepted city's unchanged choices), else its id."""
    return city.design.get("lineage") or city.design.get("id")


def _band(v, default):
    if v is None:
        return tuple(default)
    if isinstance(v, (int, float)):
        return (int(v), int(v))
    return (int(v[0]), int(v[-1]))


class City:
    """The resolved design: rasters over the frame, leaves, findings, metrics."""

    def __init__(self, design: dict, centre, radius: int):
        self.design = design
        self.cx, self.cz = int(centre[0]), int(centre[1])
        self.R = int(radius)
        self.outline = B.Outline.from_record(design.get("boundary"), (self.cx + 0.5,
                                                                      self.cz + 0.5),
                                             self.R)
        self.X0 = self.cx - self.R - MARGIN
        self.Z0 = self.cz - self.R - MARGIN
        self.W = 2 * (self.R + MARGIN) + 1
        W = self.W
        self.use = np.zeros((W, W), np.uint8)
        self.ring = np.full((W, W), -1, np.int8)
        self.rank = np.zeros((W, W), np.uint8)       # street rank: 3 radial 2 road 1 lane
        self.block = np.full((W, W), -1, np.int32)
        self.target = np.full((W, W), NO_TARGET, np.int16)
        self.treat = np.zeros((W, W), np.uint8)
        self.pave = np.zeros((W, W), np.uint8)       # 0 none, 1 road, 2 lane, 3 court
        self.owner = np.full((W, W), -1, np.int32)
        self.leaves: list = []
        self.findings: list = []
        self.walls: list = []
        self.gates: list = []
        self.blocks: list = []
        self.levels: dict = {}
        self.metrics: dict = {}
        self.regions: dict = {}
        self.ground = None       # (found, water_surface, wet) over the frame

    # ------------------------------------------------------------ frame helpers
    def ix(self, x):
        return int(x) - self.X0

    def iz(self, z):
        return int(z) - self.Z0

    def inside(self, x, z) -> bool:
        return 0 <= x - self.X0 < self.W and 0 <= z - self.Z0 < self.W

    def find(self, severity: str, field: str, what: str, **kw) -> None:
        self.findings.append({"severity": severity, "field": field, "what": what, **kw})

    def rect_use(self, r) -> np.ndarray:
        x0, z0, x1, z1 = r
        return self.use[x0 - self.X0:x1 - self.X0 + 1, z0 - self.Z0:z1 - self.Z0 + 1]


# ================================================================ compile

def compile_design(design: dict, centre, *, ground=None, forms=None,
                   voices=None, prior=None) -> City:
    """Resolve `design` (read by `citydesign.read`) about `centre`.

    `ground` is `(found, water_surface, wet)` over the frame `[W, W]` (from `atlas.ground`
    or a volume), or None for flat ground at y=64 (a probe). `prior` is the block raster
    of an earlier resolution of this place on the same frame (`city.npz` `block`), or
    None: a block whose land is unchanged keeps the identity it had there, so a revision
    renames and reseeds only the blocks it actually changed. Returns a `City`."""
    d = CD.read(design)
    city = City(d, centre, d["extent"]["radius"])
    if isinstance(prior, dict):
        city.prior, city.prior_open = prior.get("block"), prior.get("open")
    else:
        city.prior, city.prior_open = prior, None
    W = city.W
    if ground is None:
        g = np.full((W, W), 64, np.int16)
        ground = (g, g.copy(), np.zeros((W, W), bool))
    city.ground = ground
    _rings(city)
    _levels(city)
    _walls(city)
    _roads(city)
    _landmarks(city)
    _monument(city)
    _grain_streets(city)
    _landmark_access(city)
    _squares(city)
    _blocks(city)
    _pack(city)
    _fit_needs(city)
    _shade_lanes(city)
    _retile_areas(city)
    _ground(city)
    _regions(city)
    _metrics(city)
    return city


# ---------------------------------------------------------------- 1. rings

def _rings(city: City) -> None:
    W = city.W
    xs = np.arange(city.X0, city.X0 + W, dtype=np.float64)[:, None] + 0.5
    zs = np.arange(city.Z0, city.Z0 + W, dtype=np.float64)[None, :] + 0.5
    ratio = city.outline.ratio(xs, zs)
    city.ratio = ratio.astype(np.float32)
    dx = xs - city.outline.centre[0]
    dz = zs - city.outline.centre[1]
    city.bearing = (np.degrees(np.arctan2(dx, -dz)) % 360.0).astype(np.float32)
    # blocks from the boundary inward, along the local radius
    city.radius_here = city.outline.radius_at(city.bearing).astype(np.float32)
    fr = [r["outer"] for r in city.design["rings"]]
    idx = np.searchsorted(np.array(fr), ratio, side="left")
    ring = np.where(ratio <= 1.0, idx, -1)
    city.ring[:] = ring.astype(np.int8)
    city.use[ring >= 0] = LAND
    # blocks inside the ring's outer edge / outside its inner edge
    city.d_out = np.zeros((W, W), np.float32)
    city.d_in = np.zeros((W, W), np.float32)
    for k, r in enumerate(city.design["rings"]):
        m = ring == k
        f_out = r["outer"]
        f_in = city.design["rings"][k - 1]["outer"] if k else 0.0
        city.d_out[m] = ((f_out - ratio[m]) * city.radius_here[m])
        city.d_in[m] = ((ratio[m] - f_in) * city.radius_here[m])
    # the ground as found: water is kept as water where the ring's policy keeps it
    found, wsurf, wet = city.ground
    city.wet = np.asarray(wet, bool)


# ---------------------------------------------------------------- 2. levels

def _levels(city: City) -> None:
    """One level per terraced ring (median dry ground, then each ring's `rise` over the
    ring outside it), a podium's level over its ring, none for graded/preserved rings."""
    found = np.asarray(city.ground[0], np.int32)
    rings = city.design["rings"]
    n = len(rings)
    med = {}
    for k in range(n):
        m = (city.ring == k) & ~city.wet
        med[k] = int(np.median(found[m])) if m.any() else 64
    lv = {}
    for k in reversed(range(n)):
        r = rings[k]
        pol = r["ground"]["policy"]
        rise = int(r["ground"].get("rise") or 0)
        if "level" in r["ground"]:
            lv[k] = int(r["ground"]["level"])
        elif k == n - 1 or (k + 1) not in lv or lv.get(k + 1) is None:
            lv[k] = med[k] + rise if pol in ("terrace", "podium") else None
            if pol in ("terrace", "podium") and k < n - 1:
                lv[k] = max(med[k], _ref_level(lv, k + 1, med)) + rise
        else:
            base = lv[k + 1]
            lv[k] = (base + rise) if pol in ("terrace", "podium") else None
            # a terrace follows its own ground where that is higher than the ring
            # outside
            if pol == "terrace" and not rise:
                lv[k] = max(base, med[k])
    city.levels = {rings[k]["name"]: lv[k] for k in range(n)}
    city.level_of = lv
    city.median_of = med


def _ref_level(lv, k, med):
    return lv[k] if lv.get(k) is not None else med[k]


# ---------------------------------------------------------------- 3. walls

def _edge_cells(path, width):
    """`Builder._decide_edge`'s footprint of a polyline, as a set of (x, z)."""
    from .buildlib import Builder
    cells = set()
    for a, b in zip(path, path[1:]):
        cells.update(Builder.edge_band(a, b, width))
    return cells


def _gate_rings(city: City, g: dict) -> list:
    names = [r["name"] for r in city.design["rings"]]
    rr = g.get("rings", "all")
    if rr == "all" or rr is None:
        return list(range(len(names)))
    return [names.index(n) for n in rr if n in names]


def _walls(city: City) -> None:
    d = city.design
    rings = d["rings"]
    city.gate_bearings = {}
    for k, r in enumerate(rings):
        wall = r.get("wall")
        if not wall:
            continue
        width = int(wall["width"])
        gates = [g["bearing"] for g in d["gates"] if k in _gate_rings(city, g)]
        mon = d.get("monument")
        if mon and mon.get("ring") == r["name"] and mon["enter"] not in gates:
            gates.append(mon["enter"])
        # gates snap to the nearest cardinal bearing where the grain is orthogonal
        gates = sorted({float(b) for b in gates})
        city.gate_bearings[k] = gates
        widths = [rd["width"] for rd in d["radials"]] or [7]
        # the straight run a gate stands on holds its pad (`ring_gate` NEEDS 15..16)
        gate_run = max(17, max(widths) + 8)
        centre_off = r["outer"] - (width / 2.0 + 0.5) / city.R
        ol = city.outline.scaled(centre_off)
        path = ol.polyline(step=8.0, gates=gates, gate_run=gate_run)
        bad = B.check_polyline(path)
        if bad:
            city.find("blocking", f"rings[{k}].wall", f"{len(bad)} wall segment(s) off the "
                      f"lattice's axes and diagonals")
        cells = _edge_cells(path, width)
        for (x, z) in cells:
            if city.inside(x, z):
                city.use[x - city.X0, z - city.Z0] = WALL
        # the wall's apron: the reach of its towers and stairs either side, owned by the
        # wall and graded to its level (`wall.occupied` publishes 6 out and 8 in)
        from scipy import ndimage
        band = np.zeros(city.use.shape, bool)
        for (x, z) in cells:
            if city.inside(x, z):
                band[x - city.X0, z - city.Z0] = True
        near = ndimage.binary_dilation(band, iterations=APRON) & ~band
        near &= (city.use == LAND)
        # inside, the apron is the street along the wall at the wall's own level;
        # outside it is the wall's berm, a ledge at the wall's foot retained down to the
        # ring
        inner = near & (city.ring == k)
        city.use[inner] = ROAD
        city.rank[inner] = np.maximum(city.rank[inner], 2)
        city.use[near & ~inner] = OPEN
        city.apron = getattr(city, "apron", np.zeros(city.use.shape, bool)) | near
        pts = ol.gate_points(gates)
        wtype = wall.get("type") or ("great_wall" if k == len(rings) - 1 and
                                     wall["height"] >= 24 else "wall")
        rec = {"ring": k, "name": f"wall_{r['name']}", "type": wtype, "path": path,
               "width": width, "height": int(wall["height"]),
               "crown": wall.get("crown"), "face": wall.get("face"),
               "gates": pts, "cells": len(cells),
               "length": round(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                                   for a, b in zip(path, path[1:])))}
        city.walls.append(rec)
        for gp in pts:
            city.gates.append({**gp, "ring": k, "wall": rec["name"],
                               "ring_name": r["name"], "width": gate_run})


# ---------------------------------------------------------------- 4. roads

def _band_along(city: City, bearing_deg: float, width: int, r0: float, r1: float):
    """Mask of a straight band of `width` along a bearing from radius r0 to r1 (blocks)."""
    W = city.W
    xs = np.arange(city.X0, city.X0 + W)[:, None] + 0.5 - city.outline.centre[0]
    zs = np.arange(city.Z0, city.Z0 + W)[None, :] + 0.5 - city.outline.centre[1]
    ux, uz = B.bearing_vec(bearing_deg)
    along = xs * ux + zs * uz
    across = -xs * uz + zs * ux
    # axis-aligned radials are exact bands of `width` columns
    if abs(ux) < 1e-9 or abs(uz) < 1e-9:
        lo = -(width // 2)
        hi = lo + width - 1
        a = np.floor(across).astype(int)
        return (a >= lo) & (a <= hi) & (along >= r0) & (along <= r1)
    return (np.abs(across) <= width / 2.0) & (along >= r0) & (along <= r1)


def _roads(city: City) -> None:
    d = city.design
    rings = d["rings"]
    land = city.use == LAND
    # ring roads along the rings' edges, inside their walls
    for k, r in enumerate(rings):
        roads = r.get("roads") or {}
        m = (city.ring == k) & land
        wall_out = (r.get("wall") or {}).get("width", 0)
        wall_in = (rings[k - 1].get("wall") or {}).get("width", 0) if k else 0
        w_out = int(roads.get("outer") or 0)
        w_in = int(roads.get("inner") or 0)
        if w_out:
            band = m & (city.d_out >= wall_out) & (city.d_out < wall_out + w_out)
            city.use[band] = ROAD
            city.rank[band] = np.maximum(city.rank[band], 2)
        if w_in and k:
            band = m & (city.d_in >= wall_in) & (city.d_in < wall_in + w_in)
            city.use[band] = ROAD
            city.rank[band] = np.maximum(city.rank[band], 2)
    # radials, from the monument's wall (or the centre) out to the boundary
    mon = d.get("monument")
    k_mon = [r["name"] for r in rings].index(mon["ring"]) if mon else None
    for rd in d["radials"]:
        # a radial starts inside the innermost ring from which every wall outward has a
        # gate on its bearing: a street does not run into a wall with no way through
        k0 = len(rings)
        for k in reversed(range(len(rings))):
            if rings[k].get("wall") and not any(
                    abs(((b - rd["bearing"] + 180) % 360) - 180) < 1.0
                    for b in city.gate_bearings.get(k, [])):
                break
            k0 = k
        if k_mon is not None:
            k0 = max(k0, k_mon + 1)
        r0 = (rings[k0 - 1]["outer"] * city.outline.radius_at(rd["bearing"]) + 0.5) \
            if k0 > 0 else 0
        if k0 >= len(rings):
            city.find("design", "radials", f"the radial at bearing {rd['bearing']} has "
                      f"no gate in the outermost wall", bearing=rd["bearing"])
            continue
        band = _band_along(city, rd["bearing"], rd["width"], r0, city.R * 1.02 + 4)
        band &= (city.use == LAND) | (city.use == ROAD) | (city.use == WALL)
        # the radial goes through a wall only at a gate
        through = band & (city.use == WALL)
        gate_ok = np.zeros_like(through)
        for gp in city.gates:
            if abs(((gp["bearing"] - rd["bearing"] + 180) % 360) - 180) < 1.0:
                x, z = city.ix(gp["x"]), city.iz(gp["z"])
                h = gp["width"] // 2 + 2
                gate_ok[max(0, x - h):x + h + 1, max(0, z - h):z + h + 1] = True
        city.use[band & ~(through & ~gate_ok)] = ROAD
        city.use[band & through & gate_ok] = GATE
        city.rank[band] = 3
        if (through & ~gate_ok).any():
            city.find("design", "radials", f"the radial at bearing {rd['bearing']} "
                      f"crosses a wall where no gate stands", bearing=rd["bearing"])


# ---------------------------------------------------------------- 5. landmarks

def _rect_at(city: City, bearing_deg: float, frac_r: float, w: int, h: int):
    cx, cz = city.outline.point_at(bearing_deg, frac_r)
    x0 = int(round(cx - w / 2))
    z0 = int(round(cz - h / 2))
    return (x0, z0, x0 + w - 1, z0 + h - 1)


#: the landmark forms (`citydesign.LANDMARK_FORMS`), each composed below
LANDMARK_FORMS = CD.LANDMARK_FORMS


def _span_rect(city: City, k: int, bearing: float, w: int):
    """A rect `w` across a cardinal bearing, from the ring's inner wall to its outer
    one: an approach space the whole depth of the ring."""
    d = city.design
    r = d["rings"][k]
    f_in = d["rings"][k - 1]["outer"] if k else 0.0
    wall_out = (r.get("wall") or {}).get("width", 0)
    R = city.outline.radius_at(bearing)
    # clear of the inner wall's berm and of the outer wall's street and gate ramp
    a = f_in * R + APRON + 2
    b = r["outer"] * R - wall_out - APRON - 2
    ux, uz = B.bearing_vec(bearing)
    ox, oz = city.outline.centre
    if abs(ux) > 1e-6 and abs(uz) > 1e-6:
        return None
    lo = -(w // 2)
    if abs(ux) < 1e-6:                     # north / south: along z
        z_a, z_b = oz + uz * a, oz + uz * b
        return (int(math.floor(ox)) + lo, int(round(min(z_a, z_b))),
                int(math.floor(ox)) + lo + w - 1, int(round(max(z_a, z_b))))
    x_a, x_b = ox + ux * a, ox + ux * b
    return (int(round(min(x_a, x_b))), int(math.floor(oz)) + lo,
            int(round(max(x_a, x_b))), int(math.floor(oz)) + lo + w - 1)


def _landmarks(city: City) -> None:
    d = city.design
    names = [r["name"] for r in d["rings"]]
    city.landmark_rects = []
    for i, lm in enumerate(d["landmarks"]):
        k = names.index(lm["ring"])
        r = d["rings"][k]
        f_in = d["rings"][k - 1]["outer"] if k else 0.0
        at = float(lm.get("at", 0.5))
        frac = f_in + (r["outer"] - f_in) * at
        w, h = (list(lm.get("size") or [32, 32]) + [32])[:2]
        form = lm.get("form", "plaza")
        span = lm.get("span") == "ring"
        if span:
            rect = _span_rect(city, k, lm["bearing"], int(w))
            if rect is None:
                city.find("design", f"landmarks[{i}].span", f"{lm.get('name')} spans its "
                          f"ring only on a cardinal bearing")
                continue
        else:
            rect = _rect_at(city, lm["bearing"], frac, int(w), int(h))
        # stand beside a radial on the same bearing, never over it
        side = lm.get("side")
        for rd in d["radials"]:
            if abs(((rd["bearing"] - lm["bearing"] + 180) % 360) - 180) < 5 and \
                    side != "on" and not span:
                ux, uz = B.bearing_vec(rd["bearing"])
                off = rd["width"] // 2 + 1 + max(w, h) // 2 + 1
                sgn = -1 if side == "west" or side == "north" else 1
                ox, oz = int(round(-uz * off * sgn)), int(round(ux * off * sgn))
                rect = (rect[0] + ox, rect[1] + oz, rect[2] + ox, rect[3] + oz)
        sub = city.rect_use(rect)
        ringsub = city.ring[rect[0] - city.X0:rect[2] - city.X0 + 1,
                            rect[1] - city.Z0:rect[3] - city.Z0 + 1]
        if sub.size == 0 or (ringsub != k).mean() > 0.02 or \
                ((sub == WALL).any() and not span):
            city.find("design", f"landmarks[{i}]", f"{lm.get('name')} ({w}x{h}) does not "
                      f"stand inside {lm['ring']} at bearing {lm['bearing']} and "
                      f"{at} of its width", rect=list(rect))
            continue
        nm = f"landmark_{lm.get('name', i)}"
        city.landmark_rects.append((rect, lm, k))
        if form == "compound":
            front = lm.get("front") or _cardinal((lm["bearing"] + 180) % 360)
            p = dict(lm.get("params") or {})
            compose_compound(city, rect, front, ring=k, block=None, name=nm, params=p,
                             rng=_rng("landmark", nm))
            continue
        if span:
            sub[np.isin(sub, (LAND, ROAD, OPEN))] = COURT
        else:
            sub[(sub == LAND) | (sub == ROAD)] = COURT
        if form == "park":
            _park(city, rect, nm, k, lm.get("params") or {})
            continue
        if form == "forecourt":
            _forecourt(city, rect, nm, k, lm, span)
            continue
        if form == "pool":
            # made water: one basin, never tiled -- a pool wider than an area type is
            # laid is held to that width and the design is told
            x0, z0, x1, z1 = rect
            if x1 - x0 + 1 > AREA_MAX or z1 - z0 + 1 > AREA_MAX:
                city.find("design", f"landmarks[{i}].size", f"{lm.get('name')} is "
                          f"{x1 - x0 + 1}x{z1 - z0 + 1}; a pool is at most {AREA_MAX} "
                          f"across and is laid at that", asked=[w, h], most=AREA_MAX)
                x1, z1 = min(x1, x0 + AREA_MAX - 1), min(z1, z0 + AREA_MAX - 1)
                rect = (x0, z0, x1, z1)
            sub = city.rect_use(rect)
            sub[sub == COURT] = WATER
            city.leaves.append({"kind": "area", "name": nm, "type": "pool",
                                "seed": _rng("seed", nm).randint(0, 99),
                                "params": dict(lm.get("params") or {}),
                                "x0": x0, "z0": z0, "x1": x1, "z1": z1, "ring": k,
                                "role": "landmark", "voice": _voice(city, k)})
            continue
        _area_leaves(city, rect, form if form in ("market", "plaza", "square", "garden",
                                                  "grove") else "plaza",
                     name=nm, ring=k, params=lm.get("params"), role="landmark")
        if form == "temple":
            city.leaves.pop()
            _hall_leaf(city, rect, "south", name=nm,
                       ring=k, params={"use": "principal", "storeys": 2, "eaves": 2,
                                       "platform": 2, "roof": "hip_gable"})


def _park(city: City, rect, nm: str, k: int, p: dict) -> None:
    """A public garden: planted ground round a pavilion at its centre, groves in its
    corners, open to the streets on every side."""
    x0, z0, x1, z1 = rect
    w, h = x1 - x0 + 1, z1 - z0 + 1
    sub = city.rect_use(rect)
    sub[sub == COURT] = OPEN
    ps = int(p.get("pavilion", 13 if min(w, h) >= 36 else 0))
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    if ps:
        pr = (cx - ps // 2, cz - ps // 2, cx - ps // 2 + ps - 1, cz - ps // 2 + ps - 1)
        _hall_leaf(city, pr, "south", name=f"{nm}_pavilion", ring=k,
                   params={"use": "pavilion", "storeys": 1, "eaves": 2, "platform": 1,
                           "roof": "pavilion"})
    g = max(8, min(w, h) // 4)
    for tag, r in (("nw", (x0, z0, x0 + g - 1, z0 + g - 1)),
                   ("ne", (x1 - g + 1, z0, x1, z0 + g - 1)),
                   ("sw", (x0, z1 - g + 1, x0 + g - 1, z1)),
                   ("se", (x1 - g + 1, z1 - g + 1, x1, z1))):
        city.leaves.append({"kind": "area", "name": f"{nm}_{tag}", "type": "grove",
                            "seed": _rng("seed", nm, tag).randint(0, 99),
                            "params": {"planting": "copse", "floor": "turf"},
                            "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3], "ring": k,
                            "role": "garden", "voice": _voice(city, k)})
    _area_leaves(city, rect, "garden", name=nm, ring=k,
                 params={"layout": p.get("layout", "cross"), "edge": p.get("edge", "rail")},
                 role="landmark")


def _forecourt(city: City, rect, nm: str, k: int, lm: dict, span: bool) -> None:
    """A processional forecourt: a paved open court on an approach, its long sides
    lined (trees or a low wall), the approach down its middle."""
    x0, z0, x1, z1 = rect
    p = dict(lm.get("params") or {})
    edge = p.get("edge", "grove")
    ew = int(p.get("edge_width", 5))
    # the approach's own direction, not the rect's proportions (a forecourt may be wider
    # than it is deep)
    ux, uz = B.bearing_vec(lm["bearing"])
    along_z = abs(uz) >= abs(ux)
    if edge == "grove" and ew >= 3:
        strips = ([(x0, z0, x0 + ew - 1, z1), (x1 - ew + 1, z0, x1, z1)] if along_z else
                  [(x0, z0, x1, z0 + ew - 1), (x0, z1 - ew + 1, x1, z1)])
        for j, r in enumerate(strips):
            sub = city.rect_use(r)
            sub[sub == COURT] = OPEN
            _area_leaves(city, r, "grove", name=f"{nm}_edge{j}", ring=k,
                         params={"planting": "avenue", "floor": "swept"}, role="garden",
                         skip_use=(WALL,))
        rect = ((x0 + ew, z0, x1 - ew, z1) if along_z else (x0, z0 + ew, x1, z1 - ew))
    # the approach runs down the forecourt's middle as the avenue it is: a road, so the
    # gates at either end are ramped to it like any street (`_gate_ramps`), with the
    # paved court either side
    aw = 0
    for rd in city.design.get("radials") or []:
        if abs(((rd["bearing"] - lm["bearing"] + 180) % 360) - 180) < 1.0:
            aw = int(rd["width"])
    if span and aw:
        rx0, rz0, rx1, rz1 = rect
        if along_z:
            c = int(math.floor(city.outline.centre[0]))
            a0, a1 = c - aw // 2, c - aw // 2 + aw - 1
            road = (a0, rz0, a1, rz1)
            sides = [(rx0, rz0, a0 - 1, rz1), (a1 + 1, rz0, rx1, rz1)]
        else:
            c = int(math.floor(city.outline.centre[1]))
            a0, a1 = c - aw // 2, c - aw // 2 + aw - 1
            road = (rx0, a0, rx1, a1)
            sides = [(rx0, rz0, rx1, a0 - 1), (rx0, a1 + 1, rx1, rz1)]
        sub = city.rect_use(road)
        m = sub == COURT
        sub[m] = ROAD
        rk = city.rank[road[0] - city.X0:road[2] - city.X0 + 1,
                       road[1] - city.Z0:road[3] - city.Z0 + 1]
        rk[m] = 3
        for j, r in enumerate(sides):
            _area_leaves(city, r, "plaza", name=f"{nm}_{'wn'[j]}", ring=k,
                         params={"paving": p.get("paving", "banded"), "edge": "kerb"},
                         role="landmark", skip_use=(WALL,))
        return
    _area_leaves(city, rect, "plaza", name=nm, ring=k,
                 params={"paving": p.get("paving", "banded"), "edge": "kerb"},
                 role="landmark", skip_use=(WALL,))


def _landmark_access(city: City) -> None:
    """Every landmark on a street: where a compound's gate or a park's side does not
    meet a road, lane or court, a lane is carried from its front to the nearest one."""
    for rect, lm, k in getattr(city, "landmark_rects", []):
        form = lm.get("form", "plaza")
        if form != "compound":
            continue
        fronts = [lm.get("front") or _cardinal((lm["bearing"] + 180) % 360)]
        x0, z0, x1, z1 = rect
        for f in fronts:
            if _fronts_street(city, rect, f, 0.2):
                continue
            ox, oz = _FRONT_OUT[f]
            mx, mz = (x0 + x1) // 2, (z0 + z1) // 2
            sx = x0 - 1 if f == "west" else (x1 + 1 if f == "east" else mx)
            sz = z0 - 1 if f == "north" else (z1 + 1 if f == "south" else mz)
            cells = []
            for n in range(0, 80):
                x, z = sx + ox * n, sz + oz * n
                if not city.inside(x, z):
                    cells = []
                    break
                u = city.use[x - city.X0, z - city.Z0]
                if u in (ROAD, LANE, COURT, GATE):
                    break
                if u not in (LAND,):
                    cells = []
                    break
                cells.append((x, z))
            half = 2
            for (x, z) in cells:
                for t in range(-half, half + 1):
                    xx, zz = (x + t, z) if f in ("north", "south") else (x, z + t)
                    if city.inside(xx, zz) and city.use[xx - city.X0, zz - city.Z0] == LAND:
                        city.use[xx - city.X0, zz - city.Z0] = LANE
                        city.rank[xx - city.X0, zz - city.Z0] = max(
                            1, city.rank[xx - city.X0, zz - city.Z0])
            if cells:
                city.find("info", f"landmarks.{lm.get('name')}", f"a lane of "
                          f"{len(cells)} carried from its {f} front to the street")


# ---------------------------------------------------------------- 6. monument

def _cardinal(bearing_deg: float) -> str:
    return ["north", "east", "south", "west"][int(((bearing_deg + 45) % 360) // 90)]


def _local(city, axis: str, u0: float, v0: float, u1: float, v1: float, origin):
    """World rect for a local axial rect: u along the axis inward from the gate,
    v across it (positive to the axis's right looking inward)."""
    ox, oz = origin
    if axis == "south":      # gate at south, walking north (-z)
        xs = (ox + v0, ox + v1)
        zs = (oz - u1, oz - u0)
    elif axis == "north":    # walking south (+z)
        xs = (ox - v1, ox - v0)
        zs = (oz + u0, oz + u1)
    elif axis == "east":     # gate at east, walking west (-x)
        xs = (ox - u1, ox - u0)
        zs = (oz - v1, oz - v0)
    else:                    # west: walking east (+x)
        xs = (ox + u0, ox + u1)
        zs = (oz + v0, oz + v1)
    return (int(round(min(xs))), int(round(min(zs))),
            int(round(max(xs))), int(round(max(zs))))


#: hall parameters a sequence element passes to its form as they are
_HALL_KEYS = ("storeys", "eaves", "platform", "roof", "tiers", "base", "entrance")


def _hall_params(s: dict, use: str) -> dict:
    p = {"use": use, "storeys": int(s.get("storeys", 1)), "eaves": int(s.get("eaves", 1)),
         "platform": int(s.get("platform", 1)),
         "roof": s.get("roof", "hip_gable" if use != "principal" else "hip")}
    for k in ("tiers", "base"):
        if s.get(k) is not None:
            p[k] = int(s[k])
    if s.get("entrance"):
        p["entrance"] = s["entrance"]
    return p


def _sequence_layout(seq: list) -> tuple:
    """The sequence laid along its axis at its asked sizes: `[(s, u0, depth, width)]`
    and the elements beside it `[(s, side, u0, depth, v_lo, v_hi)]`, in local axis
    coordinates (u inward from the gate, v across). Nothing is fitted to a ring here:
    this is the composition's own demand."""
    parts, beside = [], []
    u = 0
    for s in seq:
        dep = int(s.get("depth", 20))
        w = int(s.get("width") or 0)
        parts.append((s, u, dep, w))
        bs = s.get("beside")
        if bs:
            court = int(bs.get("court") or bs.get("size") or 41)
            gap = int(bs.get("gap", 8))
            half = (w or 0) // 2 + gap
            for side in (-1, 1):
                v_lo, v_hi = (half, half + court - 1) if side > 0 else \
                    (-half - court + 1, -half)
                uc = u + dep // 2
                beside.append((bs, side, uc - court // 2, court, v_lo, v_hi))
        u += dep
    return parts, beside, u


def _fits(r_in: float, parts, beside, margin: float = 5.0) -> list:
    """What does not stand inside a circle of radius `r_in` with the gate at u=0, each
    as `(what, how far over)`. The widest a court or garden may be (width 0) is taken
    as the chord where it stands, so only asked widths and the halls can fail."""
    over = []
    R = r_in - margin

    def worst(u0, u1, v0, v1):
        m = 0.0
        for uu in (u0, u1):
            for vv in (v0, v1):
                m = max(m, math.hypot(r_in - uu, vv) - R)
        return m

    for s, u0, dep, w in parts:
        if not w:
            continue
        if s["kind"] in ("forecourt", "court", "garden", "bridge_court"):
            # open ground may meet the curved wall at its corners: it is measured at its
            # middle, as the compiler lays it (the chord there, less a margin)
            mid = u0 + dep / 2
            half = math.sqrt(max(0.0, r_in * r_in - (r_in - mid) ** 2)) - 3
            o = w / 2 - half
        else:
            o = worst(u0, u0 + dep, -w / 2, w / 2)
        if o > 0:
            over.append((s["kind"], o))
    for bs, side, u0, dep, v0, v1 in beside:
        o = worst(u0, u0 + dep, v0, v1)
        if o > 0:
            over.append((f"beside {bs.get('form', 'round_altar')}", o))
    if parts:
        s, u0, dep, w = parts[-1]
        if u0 + dep > 2 * r_in - 2 * margin:
            over.append(("sequence depth", u0 + dep - (2 * r_in - 2 * margin)))
    return over


def monument_demand(design: dict, R: int) -> dict | None:
    """The land the monument's composition asks, against the ring it is given: the
    smallest interior radius that holds the whole sequence and what stands beside it at
    their asked sizes, and the ring `outer` fraction that gives it. The composition
    negotiates with its parent through this -- a finding naming the `outer` it needs --
    rather than being shortened to fit the ring it happened to get."""
    mon = design.get("monument")
    if not mon:
        return None
    names = [r["name"] for r in design["rings"]]
    k = names.index(mon["ring"])
    ring = design["rings"][k]
    wall_w = (ring.get("wall") or {}).get("width", 0)
    r_in = ring["outer"] * R - wall_w - 1
    parts, beside, depth = _sequence_layout(mon.get("sequence") or [])
    need = r_in
    while _fits(need, parts, beside) and need < 4 * R:
        need += 1
    return {"ring": k, "r_in": round(r_in, 1), "needs_r_in": round(need, 1),
            "needs_outer": round((need + wall_w + 1) / R, 4), "depth": depth,
            "over": _fits(r_in, parts, beside)}


def _monument(city: City) -> None:
    d = city.design
    mon = d.get("monument")
    city.monument = None
    if not mon:
        return
    names = [r["name"] for r in d["rings"]]
    k = names.index(mon["ring"])
    ring = d["rings"][k]
    axis = _cardinal(mon["enter"])
    if abs(((mon["enter"] - B.BEARINGS[axis] + 180) % 360) - 180) > 1:
        city.find("design", "monument.enter", f"the axis at bearing {mon['enter']} is laid "
                  f"on the nearest cardinal, {axis}: halls stand square to the lattice")
    wall_w = (ring.get("wall") or {}).get("width", 0)
    r_in = ring["outer"] * city.outline.radius_at(B.BEARINGS[axis]) - wall_w - 1
    ux, uz = B.bearing_vec(B.BEARINGS[axis])
    origin = (int(round(city.cx + ux * r_in)), int(round(city.cz + uz * r_in)))
    seq = mon.get("sequence") or []
    lay, beside, want = _sequence_layout(seq)
    # the composition's demand against the ring it was given: never compressed to fit
    dem = monument_demand(d, city.R)
    city.monument_demand = dem
    if dem and dem["over"]:
        what = ", ".join(f"{w} by {o:.0f}" for w, o in dem["over"][:6])
        city.find("blocking", f"rings[{k}].outer",
                  f"the monument's composition does not stand in {ring['name']} "
                  f"(interior radius {dem['r_in']:.0f}): {what} over. It needs an "
                  f"interior radius of {dem['needs_r_in']:.0f}, rings[{k}].outer >= "
                  f"{dem['needs_outer']} -- give the ring that land, or change the "
                  f"composition", needs_outer=dem["needs_outer"], asked=want)
    parts = []
    band_half = 0
    for i, (s, u, dep, w) in enumerate(lay):
        mid_u = u + dep / 2
        rr = abs(r_in - mid_u)
        chord = 2 * math.sqrt(max(0.0, r_in * r_in - rr * rr)) - 6
        if not w:
            w = int(min(chord, 0.55 * 2 * r_in))
        w = int(min(w, max(8, chord)))
        rect = _local(city, axis, u, -(w // 2), u + dep - 1, w - w // 2 - 1, origin)
        parts.append((s, rect, dep, w))
        band_half = max(band_half, w // 2 + 1)
    face_out = axis                       # halls face the entrance
    for i, (s, rect, dep, w) in enumerate(parts):
        sub = city.rect_use(rect)
        kind = s["kind"]
        nm = f"{mon.get('name', 'monument')}_{i:02d}_{kind}"
        if kind in ("forecourt", "court", "bridge_court"):
            sub[(sub == LAND)] = COURT
            fl = s.get("flanks")
            if fl and kind == "court":
                fw = int(fl.get("width", 17))
                fd = int(fl.get("depth", 11))
                n = int(fl.get("count", 2))
                gap = max(2, (dep - n * fw) // (n + 1))
                x0, z0, x1, z1 = rect
                for side in (-1, 1):
                    for j in range(n):
                        u0 = gap + j * (fw + gap)
                        # flank halls face across the court, toward the axis
                        if axis in ("south", "north"):
                            zz0 = (z1 - u0 - fw + 1) if axis == "south" else (z0 + u0)
                            hx0 = x0 + 1 if side < 0 else x1 - fd
                            hr = (hx0, zz0, hx0 + fd - 1, zz0 + fw - 1)
                            facing = "east" if side < 0 else "west"
                        else:
                            xx0 = (x1 - u0 - fw + 1) if axis == "east" else (x0 + u0)
                            hz0 = z0 + 1 if side < 0 else z1 - fd
                            hr = (xx0, hz0, xx0 + fw - 1, hz0 + fd - 1)
                            facing = "south" if side < 0 else "north"
                        _hall_leaf(city, hr, facing, name=f"{nm}_flank_{side}_{j}", ring=k,
                                   params={"use": "side", "storeys": int(fl.get("storeys", 1)),
                                           "eaves": 1, "platform": int(fl.get("platform", 1)),
                                           "roof": fl.get("roof", "hip_gable")})
            _area_leaves(city, rect, "plaza", name=nm, ring=k,
                         params={"paving": "framed", "edge": "kerb"}, role="court",
                         skip_use=(HALL,))
        elif kind == "garden":
            sub[(sub == LAND)] = OPEN
            _area_leaves(city, rect, "garden", name=nm, ring=k,
                         params={"layout": s.get("layout", "cross"),
                                 "edge": s.get("edge", "hedge")}, role="garden")
        else:
            use = "gate" if kind == "gate_hall" else (s.get("role") or "principal")
            _hall_leaf(city, rect, face_out, name=nm, ring=k, params=_hall_params(s, use),
                       passage=(kind == "gate_hall"))
    # what stands beside the axis: a pair of courts, each round its own monument
    for j, (bs, side, u0, dep, v0, v1) in enumerate(beside):
        court = _local(city, axis, u0, v0, u0 + dep - 1, v1, origin)
        if not (city.inside(court[0], court[1]) and city.inside(court[2], court[3])):
            continue
        sub = city.rect_use(court)
        sub[np.isin(sub, (LAND, ROAD))] = COURT
        # the court opens onto the axis: the land between it and the hall is court
        link = _local(city, axis, u0, min(0, v0), u0 + dep - 1, max(0, v1), origin)
        lsub = city.rect_use(link)
        lsub[lsub == LAND] = COURT
        size = int(bs.get("size") or max(11, dep - 12))
        cu, cv = u0 + dep // 2, (v0 + v1) // 2
        h = size // 2
        r = _local(city, axis, cu - h, cv - h, cu - h + size - 1, cv - h + size - 1, origin)
        form = bs.get("form", "round_altar")
        nm = f"{mon.get('name', 'monument')}_beside_{'w' if side < 0 else 'e'}{j // 2}"
        _area_leaves(city, court, "plaza", name=nm + "_court", ring=k,
                     params={"paving": "framed", "edge": "kerb"}, role="court",
                     skip_use=(HALL,))
        if form == "ceremonial_hall":
            _hall_leaf(city, r, face_out, name=nm, ring=k,
                       params=_hall_params(bs.get("params") or {}, "pavilion"))
        else:
            _claim(city, r, HALL)
            city.leaves.append({"kind": "plot", "name": nm, "type": form,
                                "seed": _rng("seed", nm).randint(0, 99),
                                "params": dict(bs.get("params") or {}),
                                "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3],
                                "front": face_out, "attached": [], "inset": 0, "ring": k,
                                "role": "civic",
                                "voice": _voice(city, k, "monument")})
        band_half = max(band_half, max(abs(v0), abs(v1)) + 1)
    # the axial band's full extent and the wings either side of it
    if parts:
        u_total = want
        city.monument = {"axis": axis, "origin": list(origin), "depth": int(u_total),
                         "band_half": band_half, "ring": k, "scale": 1.0,
                         "demand": dem,
                         "parts": [{"kind": s["kind"], "role": s.get("role"),
                                    "rect": list(r)} for s, r, _, _ in parts]}
        # the connective band is as wide as the halls it joins: courts and gardens keep
        # their own rects, and the land beside a narrower court is the wings'
        # (`monument.band: "halls"`; the default is the widest element, as it was)
        halls = [w for s, _, _, w in parts if s["kind"] in ("gate_hall", "hall")]
        if mon.get("band") == "halls" and halls:
            wide = max(halls) // 2 + 1
        else:
            wide = max(w for _, _, _, w in parts) // 2 + 1
        band = _local(city, axis, -2, -wide - 3, u_total + 2, wide + 3, origin)
        sub = city.rect_use(band)
        # the band between parts is court: the axis walk is continuous
        sub[sub == LAND] = COURT
        city.mon_band = band
        # the court leaves for the connective band
        _area_leaves(city, band, "plaza", name=f"{mon.get('name', 'monument')}_axis",
                     ring=k, params={"paving": "banded", "edge": "kerb"}, role="court",
                     only_use=COURT, skip_use=(HALL,), claimed=True)


# ---------------------------------------------------------------- 7. grain streets

def _grain_of(city: City, k: int, bearing_deg=None) -> tuple:
    r = city.design["rings"][k]
    if bearing_deg is not None:
        for wd in r.get("wards") or []:
            a, b = wd["from"], wd["to"]
            inside = (a <= bearing_deg <= b) if a <= b else (bearing_deg >= a or
                                                            bearing_deg <= b)
            if inside:
                return wd["grain"], {**(r.get("params") or {}), **(wd.get("params") or {})}
    return r.get("grain"), dict(r.get("params") or {})


def _ward_mask(city: City, k: int) -> list:
    """`[(grain, params, mask), ...]` for ring k: its wards, then the rest."""
    r = city.design["rings"][k]
    base = city.ring == k
    out = []
    taken = np.zeros_like(base)
    for wd in r.get("wards") or []:
        a, b = wd["from"], wd["to"]
        bm = ((city.bearing >= a) & (city.bearing <= b)) if a <= b else \
            ((city.bearing >= a) | (city.bearing <= b))
        m = base & bm & ~taken
        taken |= m
        out.append((wd["grain"], {**(r.get("params") or {}), **(wd.get("params") or {})},
                    m, wd))
    if r.get("role") != "monument" or True:
        out.append((r.get("grain"), dict(r.get("params") or {}), base & ~taken, None))
    return out


def grain_dims(grain: str, p: dict) -> dict:
    """The spacings a grain's grid needs, from its lots."""
    if grain == "courts":
        dep = _band(p.get("lot_depth"), (18, 21))
        wid = _band(p.get("lot_width"), (15, 19))
        lane = int(p.get("lane_width", 3))
        street = int(p.get("street_width", 7))
        every = int(p.get("street_every", 6))
        fd = int(p.get("front_depth", 12))
        strip = 2 * dep[1] + lane
        return {"strip": strip, "lane": lane, "street": street,
                "block_len": every * (wid[0] + wid[1]) // 2 + 2 * fd + street,
                "depth": dep, "width": wid, "front_depth": fd}
    if grain == "rows":
        dep = _band(p.get("lot_depth"), (9, 11))
        wid = _band(p.get("lot_width"), (5, 6))
        lane = int(p.get("lane_width", 3))
        street = int(p.get("street_width", 5))
        every = int(p.get("street_every", 10))
        fd = int(p.get("front_depth", 11))
        strip = 2 * dep[1] + lane
        return {"strip": strip, "lane": lane, "street": street,
                "block_len": every * (wid[0] + wid[1]) // 2 + 2 * fd + street,
                "depth": dep, "width": wid, "front_depth": fd}
    if grain == "estates":
        dep = _band(p.get("estate_depth"), (52, 70))
        wid = _band(p.get("estate_width"), (36, 48))
        street = int(p.get("street_width", 7))
        return {"strip": 2 * dep[1] + street, "lane": street, "street": street,
                "block_len": 3 * (wid[0] + wid[1]) // 2 + street,
                "depth": dep, "width": wid, "front_depth": 0}
    if grain == "clusters":
        dep = _lot_band(p, "lot_depth", (7, 11))
        wid = _lot_band(p, "lot_width", (6, 9))
        lane = int(p.get("lane_width", 3))
        return {"strip": 2 * dep[1] + lane, "lane": lane, "street": lane,
                "block_len": 0, "depth": dep, "width": wid, "front_depth": 0}
    if grain == "fields":
        fld = _band(p.get("field"), (32, 48))
        street = int(p.get("street_width", 5))
        return {"strip": 3 * fld[1] + street, "lane": street, "street": street,
                "block_len": 4 * fld[1] + street, "depth": fld, "width": fld,
                "front_depth": 0}
    return {"strip": 0, "lane": 0, "street": 0, "block_len": 0}


def _lines(c: int, lo: int, hi: int, period: int, width: int) -> list:
    """Band starts `[(a, a + width - 1)]` of a grid anchored so a band is centred on c."""
    if period <= width:
        return []
    out = []
    a0 = c - width // 2
    k0 = int(math.floor((lo - a0) / period)) - 1
    k1 = int(math.ceil((hi - a0) / period)) + 1
    for k in range(k0, k1 + 1):
        a = a0 + k * period
        if a + width - 1 >= lo and a <= hi:
            out.append((a, a + width - 1))
    return out


def _grain_streets(city: City) -> None:
    city.ward_grain = []
    for k, r in enumerate(city.design["rings"]):
        if r.get("role") == "monument":
            # the wings round the axial band are estates on lanes along the axis
            mon = city.design.get("monument") or {}
            wings = mon.get("wings") or {}
            grain = wings.get("grain", "estates")
            p = dict(wings.get("params") or {})
            axis = (city.monument or {}).get("axis", "south")
            p.setdefault("lane_dir", "north_south" if axis in ("north", "south")
                         else "east_west")
            m = (city.ring == k) & (city.use == LAND)
            city.ward_grain.append((k, grain, p, m))
            _grid(city, k, grain, p, m)
            continue
        for grain, p, m, wd in _ward_mask(city, k):
            city.ward_grain.append((k, grain, p, m))
            if grain in ("open",):
                continue
            if grain == "clusters":
                _cluster_lanes(city, k, p, m)
                continue
            _grid(city, k, grain, p, m)


def _grid(city: City, k: int, grain: str, p: dict, m: np.ndarray) -> None:
    dims = grain_dims(grain, p)
    if not dims["strip"]:
        return
    lane_dir = p.get("lane_dir", "east_west")
    ii, jj = np.nonzero(m)
    if not len(ii):
        return
    xlo, xhi = int(ii.min()) + city.X0, int(ii.max()) + city.X0
    zlo, zhi = int(jj.min()) + city.Z0, int(jj.max()) + city.Z0
    lanes = _lines(city.cz if lane_dir == "east_west" else city.cx,
                   zlo if lane_dir == "east_west" else xlo,
                   zhi if lane_dir == "east_west" else xhi,
                   dims["strip"], dims["lane"])
    streets = _lines(city.cx if lane_dir == "east_west" else city.cz,
                     xlo if lane_dir == "east_west" else zlo,
                     xhi if lane_dir == "east_west" else zhi,
                     dims["block_len"], dims["street"])
    city.grid_lines = getattr(city, "grid_lines", [])
    city.grid_lines.append({"ring": k, "grain": grain, "params": p, "lane_dir": lane_dir,
                            "lanes": lanes, "streets": streets, "mask": m})
    land = m & (city.use == LAND)
    for a, b in lanes:
        sl = slice(a - (city.Z0 if lane_dir == "east_west" else city.X0),
                   b + 1 - (city.Z0 if lane_dir == "east_west" else city.X0))
        if lane_dir == "east_west":
            sub = land[:, sl]
            city.use[:, sl][sub] = LANE if grain != "estates" else ROAD
            city.rank[:, sl][sub] = np.maximum(city.rank[:, sl][sub], 1)
        else:
            sub = land[sl, :]
            city.use[sl, :][sub] = LANE if grain != "estates" else ROAD
            city.rank[sl, :][sub] = np.maximum(city.rank[sl, :][sub], 1)
    land = m & (city.use == LAND)
    for a, b in streets:
        sl = slice(a - (city.X0 if lane_dir == "east_west" else city.Z0),
                   b + 1 - (city.X0 if lane_dir == "east_west" else city.Z0))
        if lane_dir == "east_west":
            sub = land[sl, :] | (m[sl, :] & (city.use[sl, :] == LANE))
            city.use[sl, :][sub] = ROAD
            city.rank[sl, :][sub] = np.maximum(city.rank[sl, :][sub], 2)
        else:
            sub = land[:, sl] | (m[:, sl] & (city.use[:, sl] == LANE))
            city.use[:, sl][sub] = ROAD
            city.rank[:, sl][sub] = np.maximum(city.rank[:, sl][sub], 2)


# ---------------------------------------------------------------- 7a. clusters

def _lot_band(p: dict, key: str, default) -> tuple:
    """A lot band from the design, clipped into the dwelling form's own measured
    footprint (the pad plus the inset on both sides): a cluster's lots are its form's."""
    band = _band(p.get(key), default)
    nd = _needs_of(p.get("dwelling")) if p.get("dwelling") else None
    if nd:
        fp = nd["footprint"]
        lo = min(fp[0], fp[1]) + 2
        hi = max(fp[2], fp[3]) + 2 if key == "lot_depth" else min(fp[2], fp[3]) + 2
        band = (max(lo, min(band[0], hi)), max(lo, min(band[1], hi)))
    return band


def _cluster_lanes(city: City, k: int, p: dict, m: np.ndarray) -> None:
    """The lanes of a cluster grain: concentric with the place's own outline every two
    lots' depth (so each lane has a lot fronting it on both sides), and `spokes` straight
    lanes out from the centre through the ring. Nothing here is a grid: the lanes follow
    the boundary the design drew, whatever its shape."""
    dims = grain_dims("clusters", p)
    lane = dims["lane"]
    dep = dims["depth"][1]
    ring = city.design["rings"][k]
    inner_road = int((ring.get("roads") or {}).get("inner") or 0) and k > 0
    # the first lane is one lot out from an inner road that lots already front, else one
    # lot out from the ring's inner edge
    first = (2 * dep if inner_road else dep) + (inner_road or 0)
    period = 2 * dep + lane
    land = m & (city.use == LAND)
    d = city.d_in
    dd = d - first
    band = land & (dd >= 0) & (np.mod(dd, period) < lane)
    # a lane that would leave less than one lot between it and the ring's outer edge is
    # not laid: that land is the last row's back
    band &= city.d_out >= dep * 0.6
    city.use[band] = LANE
    city.rank[band] = np.maximum(city.rank[band], 1)
    n = int(p.get("spokes", 0) or 0)
    if n > 0:
        f_in = city.design["rings"][k - 1]["outer"] if k else 0.0
        ax = (city.design.get("axis") or {}).get("bearing", 180)
        for i in range(n):
            b = (ax + 180.0 / n + i * 360.0 / n) % 360.0
            r0 = f_in * city.outline.radius_at(b) - 1
            sp = _band_along(city, b, lane, max(0.0, r0), city.R * 1.02 + 2)
            sp &= m & np.isin(city.use, (LAND, LANE))
            city.use[sp] = LANE
            city.rank[sp] = np.maximum(city.rank[sp], 1)
    city.grid_lines = getattr(city, "grid_lines", [])
    city.grid_lines.append({"ring": k, "grain": "clusters", "params": p,
                            "lane_dir": None, "lanes": [], "streets": [], "mask": m})


def _pack_clusters(city: City, blk: dict) -> None:
    """Lots along every street the block fronts -- the concentric lanes, the spokes, the
    ring roads -- each a rectangle square to the lattice, stepped round a curve, its
    front on the street; consecutive lots that touch along a flank stand in a run
    (their shared flank a party wall) up to `run` long, then a passage is left."""
    p = blk["params"]
    form = p.get("dwelling")
    if not form:
        city.find("design", f"rings[{blk['ring']}].params.dwelling", "a clusters grain "
                  "names its dwelling form (any dwelling in the forms table)")
        return
    dims = grain_dims("clusters", p)
    wl, wh = dims["width"]
    dl, dh = dims["depth"]
    rng = _rng("clusters", blk["id"], _lineage(city))
    x0, z0, x1, z1 = blk["x0"], blk["z0"], blk["x1"], blk["z1"]
    bid, k = blk["id"], blk["ring"]
    sub = (city.block[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] == bid) \
        & (city.use[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] == LAND)
    if sub.sum() < wl * dl:
        return
    ii, jj = np.nonzero(sub)
    order = sorted(zip(ii.tolist(), jj.tolist()),
                   key=lambda c: (city.bearing[c[0] + x0 - city.X0, c[1] + z0 - city.Z0],
                                  -city.d_in[c[0] + x0 - city.X0, c[1] + z0 - city.Z0]))
    placed = []
    run_most = int(p.get("run", 4) or 4)
    for (i, j) in order:
        x, z = x0 + i, z0 + j
        if city.use[x - city.X0, z - city.Z0] != LAND:
            continue
        done = False
        for front in ("north", "south", "west", "east"):
            ox, oz = _FRONT_OUT[front]
            if not city.inside(x + ox, z + oz) or city.use[x + ox - city.X0,
                                                            z + oz - city.Z0] not in (
                    ROAD, LANE, COURT):
                continue
            w = rng.randint(wl, wh)
            for dd in range(dh, dl - 1, -1):
                for ww in (w, wl):
                    for shift in range(0, ww):
                        a = (x if front in ("north", "south") else z) - shift
                        edge = z if front in ("north", "south") else x
                        rect = _lot_rect(front, a, ww, edge, dd)
                        if not (_free(city, rect) and _dry(city, rect)):
                            continue
                        gap = _front_gap(city, rect, front, 0.6)
                        if gap is None:
                            continue
                        _claim(city, rect)
                        for (gx, gz) in gap:
                            city.use[gx - city.X0, gz - city.Z0] = COURT
                        placed.append((rect, front))
                        done = True
                        break
                    if done:
                        break
                if done:
                    break
            if done:
                break
    # runs: lots facing the same way that share a whole flank stand attached, up to
    # `run` in a row; the lot that would make the run longer keeps its flank free
    att = {n: [] for n in range(len(placed))}
    if (_decl(form) or {}).get("attached"):
        runs = {}
        for a_i, (ra, fa) in enumerate(placed):
            for b_i, (rb, fb) in enumerate(placed):
                if b_i <= a_i or fa != fb:
                    continue
                f1, f2 = _FLANKS[fa]
                for lo, hi, lo_s, hi_s in ((ra, rb, f2, f1), (rb, ra, f2, f1)):
                    if not _touch(lo, hi, f1):
                        continue
                    # a party wall is a whole flank: the two lots' depths agree
                    if f1 == "west" and (lo[1], lo[3]) != (hi[1], hi[3]):
                        continue
                    if f1 == "north" and (lo[0], lo[2]) != (hi[0], hi[2]):
                        continue
                    ia = a_i if lo is ra else b_i
                    ib = b_i if lo is ra else a_i
                    ra_len = runs.get(ia, 1)
                    if ra_len >= run_most:
                        continue
                    att[ia].append(lo_s)
                    att[ib].append(hi_s)
                    runs[ib] = ra_len + 1
    for n, (rect, front) in enumerate(placed):
        params = _dwelling_params(city, "clusters", p, form, rng)
        _plot_leaf(city, rect, form, front, params, name=f"r{k}_b{bid}_h{n:02d}", ring=k,
                   attached=sorted(set(att[n])), block=bid)
    blk["lots"] = len(placed)


def _front_gap(city: City, rect, front: str, least: float, reach: int = 2):
    """Does `rect` front a street across a curve: the land cells between its front row
    and the street (at most `reach` deep) where at least `least` of its front columns
    reach a street, or None. A round lane is met by square lots; the few columns
    between a lot's front and the lane's curve are its paved forecourt."""
    x0, z0, x1, z1 = rect
    ox, oz = _FRONT_OUT[front]
    if front in ("north", "south"):
        cols = [(x, z0 - 1 if front == "north" else z1 + 1) for x in range(x0, x1 + 1)]
    else:
        cols = [(x0 - 1 if front == "west" else x1 + 1, z) for z in range(z0, z1 + 1)]
    gap, ok = [], 0
    for (x, z) in cols:
        run = []
        for n in range(reach + 1):
            cx, cz = x + ox * n, z + oz * n
            if not city.inside(cx, cz):
                break
            u = city.use[cx - city.X0, cz - city.Z0]
            if u in (ROAD, LANE, COURT, GATE):
                ok += 1
                gap.extend(run)
                break
            if u != LAND:
                break
            run.append((cx, cz))
    return gap if ok >= least * len(cols) else None


def _dry(city: City, rect) -> bool:
    """No lot on water as found, unless its ring makes the ground (a terrace or a
    podium fills it): a house is not stood in a pond."""
    x0, z0, x1, z1 = rect
    sl = (slice(x0 - city.X0, x1 - city.X0 + 1), slice(z0 - city.Z0, z1 - city.Z0 + 1))
    if not city.wet[sl].any():
        return True
    k = int(city.ring[x0 - city.X0, z0 - city.Z0])
    g = _ward_ground(city, k, (x0 + x1) / 2, (z0 + z1) / 2) if k >= 0 else {}
    return g.get("policy") in ("terrace", "podium")


_DECLS: dict = {}


def _decl(tname: str) -> dict | None:
    """A form's declaration (`pipeline.load_type`), cached; None where there is none."""
    if tname not in _DECLS:
        _DECLS[tname] = CD.library_form(tname)
    return _DECLS[tname]


def _shade_lanes(city: City) -> None:
    """A ring whose `shade` is `lanes`: every house fronting a narrow street of that
    ring, whose form offers a lane shade, is asked to carry its beams out over the
    street to its middle -- the reach measured here from the street's own width -- so
    the lane is roofed between the houses on either side of it."""
    rings = city.design["rings"]
    for lf in city.leaves:
        if lf["kind"] != "plot" or lf.get("ring") is None:
            continue
        r = rings[lf["ring"]] if 0 <= lf["ring"] < len(rings) else {}
        if r.get("shade", "none") != "lanes":
            continue
        d = _decl(lf["type"]) or {}
        opts = (d.get("params") or {}).get("shade")
        if not opts or opts[0] != "choice" or "lane" not in opts[1]:
            continue
        front = lf.get("front")
        if front not in _FRONT_OUT:
            continue
        ox, oz = _FRONT_OUT[front]
        x0, z0, x1, z1 = lf["x0"], lf["z0"], lf["x1"], lf["z1"]
        mx, mz = (x0 + x1) // 2, (z0 + z1) // 2
        sx = x0 - 1 if front == "west" else (x1 + 1 if front == "east" else mx)
        sz = z0 - 1 if front == "north" else (z1 + 1 if front == "south" else mz)
        width = 0
        while width < 12 and city.inside(sx + ox * width, sz + oz * width) and \
                city.use[sx + ox * width - city.X0, sz + oz * width - city.Z0] in (
                    LANE, ROAD, COURT):
            width += 1
        if not 1 <= width <= 5:
            continue                      # a street too wide to roof keeps its sky
        # the lot's own inset, then the lane to its middle (and the middle column)
        reach = int(lf.get("inset", 1)) + (width + 1) // 2
        lf["params"] = {**(lf.get("params") or {}), "shade": "lane",
                        "reach": max(1, min(4, reach))}


# ---------------------------------------------------------------- 7b. squares

def _squares(city: City) -> None:
    """A small square at every `square_every`-th crossing of a street and a lane: the
    four block corners round the crossing given up to paving, a well or a tree -- the
    places a crowded grain gathers, and what breaks a long lane into walkable pieces."""
    for gl in getattr(city, "grid_lines", []):
        n = int(gl["params"].get("square_every", 3 if gl["grain"] in ("rows", "courts")
                                 else 0))
        if not n or gl["grain"] not in ("rows", "courts"):
            continue
        side = 13 if gl["grain"] == "courts" else 12
        k = gl["ring"]
        cnt = 0
        for (sa, sb) in gl["streets"]:
            for (la, lb) in gl["lanes"]:
                cnt += 1
                h = _rng("sq", k, sa, la, _lineage(city)).random()
                if h > 1.0 / n:
                    continue
                if gl["lane_dir"] == "east_west":
                    cx, cz = (sa + sb) // 2, (la + lb) // 2
                else:
                    cx, cz = (la + lb) // 2, (sa + sb) // 2
                rect = (cx - side // 2, cz - side // 2, cx - side // 2 + side - 1,
                        cz - side // 2 + side - 1)
                if not (city.inside(rect[0], rect[1]) and city.inside(rect[2], rect[3])):
                    continue
                sub = city.rect_use(rect)
                ring = city.ring[rect[0] - city.X0:rect[2] - city.X0 + 1,
                                 rect[1] - city.Z0:rect[3] - city.Z0 + 1]
                if (ring != k).any() or np.isin(sub, (WALL, COURT, HALL, GATE)).any():
                    continue
                if not gl["mask"][cx - city.X0, cz - city.Z0]:
                    continue
                sub[np.isin(sub, (LAND, LANE, ROAD))] = COURT
                city.leaves.append({"kind": "area", "name": f"r{k}_sq{sa}_{la}",
                                    "type": "square",
                                    "seed": _rng("seed", "sq", sa, la).randint(0, 99),
                                    "params": {"paving": _rng("sqp", sa, la).choice(
                                        ["banded", "checker", "radial"]),
                                        "canopy": "hip"},
                                    "x0": rect[0], "z0": rect[1], "x1": rect[2],
                                    "z1": rect[3], "ring": k, "role": "square",
                                    "voice": _voice(city, k)})


# ---------------------------------------------------------------- 8. blocks

def _block_signature(mask: np.ndarray, sl) -> tuple:
    """A block's identity by its land: its frame rect and the exact cells it covers."""
    return (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop,
            hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()[:16])


def _stable_ids(lab: np.ndarray, n: int, objs, prior) -> np.ndarray:
    """`ids[label]`: each labelled block's id. Without a prior resolution, label order
    (the raster scan). With one, a block whose cells are exactly a prior block's keeps
    that block's id; every other block takes a fresh id past the prior's largest, in
    label order. Identity is the land's, not the scan position's, so a change in one
    place does not renumber -- and so rename and reseed -- every block after it."""
    ids = np.arange(-1, n, dtype=np.int64)          # ids[0] is the unlabelled ground
    if prior is None or prior.shape != lab.shape:
        return ids
    from scipy import ndimage
    pobjs = ndimage.find_objects(np.where(prior >= 0, prior + 1, 0))
    known = {}
    for pid, sl in enumerate(pobjs):
        if sl is None:
            continue
        known[_block_signature(prior[sl] == pid, sl)] = pid
    nxt = int(prior.max()) + 1 if prior.size else 0
    for b, sl in enumerate(objs):
        if sl is None:
            continue
        pid = known.get(_block_signature(lab[sl] == b + 1, sl))
        if pid is not None:
            ids[b + 1] = pid
        else:
            ids[b + 1] = nxt
            nxt += 1
    return ids


def _blocks(city: City) -> None:
    from scipy import ndimage
    land = city.use == LAND
    lab, n = ndimage.label(land, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    objs = ndimage.find_objects(lab)
    ids = _stable_ids(lab, n, objs, getattr(city, "prior", None))
    city.block[:] = ids[lab].astype(np.int32)
    grain_at = np.full(city.use.shape, -1, np.int16)
    gp = []
    for gi, (k, grain, p, m) in enumerate(city.ward_grain):
        grain_at[m & (grain_at < 0)] = gi
        gp.append((k, grain, p))
    city.blocks = []
    for b, sl in enumerate(objs):
        if sl is None:
            continue
        mm = lab[sl] == b + 1
        gi_vals = grain_at[sl][mm]
        gi = int(np.bincount(gi_vals[gi_vals >= 0]).argmax()) if (gi_vals >= 0).any() \
            else -1
        k, grain, p = gp[gi] if gi >= 0 else (int(city.ring[sl][mm][0]), "open", {})
        city.blocks.append({"id": int(ids[b + 1]), "x0": sl[0].start + city.X0,
                            "z0": sl[1].start + city.Z0,
                            "x1": sl[0].stop - 1 + city.X0, "z1": sl[1].stop - 1 + city.Z0,
                            "cells": int(mm.sum()), "ring": int(k), "grain": grain,
                            "params": p})


# ---------------------------------------------------------------- 9. lots

def _free(city: City, rect, allow=(LAND,)) -> bool:
    x0, z0, x1, z1 = rect
    if not (city.inside(x0, z0) and city.inside(x1, z1)):
        return False
    sub = city.use[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1]
    if not np.isin(sub, allow).all():
        return False
    # a lot keeps one column off a wall: its ledge is the ground contract's, and a
    # wall's body is not the ground a house's pad is levelled to
    ring = city.use[max(0, x0 - city.X0 - 1):x1 - city.X0 + 2,
                    max(0, z0 - city.Z0 - 1):z1 - city.Z0 + 2]
    return not (ring == WALL).any()


def _fronts_street(city: City, rect, front: str, least: float = 0.6) -> bool:
    x0, z0, x1, z1 = rect
    if front == "north":
        cells = [(x, z0 - 1) for x in range(x0, x1 + 1)]
    elif front == "south":
        cells = [(x, z1 + 1) for x in range(x0, x1 + 1)]
    elif front == "west":
        cells = [(x0 - 1, z) for z in range(z0, z1 + 1)]
    else:
        cells = [(x1 + 1, z) for z in range(z0, z1 + 1)]
    ok = 0
    for x, z in cells:
        if city.inside(x, z) and city.use[x - city.X0, z - city.Z0] in (ROAD, LANE, COURT,
                                                                        GATE):
            ok += 1
    return ok >= least * len(cells)


def _claim(city: City, rect, code=LOT) -> None:
    x0, z0, x1, z1 = rect
    city.use[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] = code


def _lot_rect(front: str, a: int, w: int, edge: int, d: int) -> tuple:
    """A lot `w` along its front and `d` deep whose front edge lies on line `edge`,
    starting at `a` along it."""
    if front == "north":
        return (a, edge, a + w - 1, edge + d - 1)
    if front == "south":
        return (a, edge - d + 1, a + w - 1, edge)
    if front == "west":
        return (edge, a, edge + d - 1, a + w - 1)
    return (edge - d + 1, a, edge, a + w - 1)


def _pack(city: City) -> None:
    """Every block packed by its grain's composer; leftovers owned as open ground."""
    for blk in city.blocks:
        grain = blk["grain"]
        if grain in ("courts", "rows"):
            _pack_rows(city, blk)
        elif grain == "estates":
            _pack_estates(city, blk)
        elif grain == "fields":
            _pack_fields(city, blk)
        elif grain == "clusters":
            _pack_clusters(city, blk)
    # frontage along the curved roads: stepped lots facing the ring road, in what is
    # left
    for blk in city.blocks:
        if blk["grain"] in ("courts", "rows"):
            _pack_curve(city, blk)
    _leftovers(city)


def _row_forms(grain: str, p: dict) -> tuple:
    if grain == "courts":
        return (p.get("dwelling", "courtyard_house"), p.get("front", "shop_house"))
    return (p.get("dwelling", "row_house"), p.get("front", "shop_house"))


def _dwelling_params(city, grain, p, form, rng, rank_front=False):
    if form == "courtyard_house":
        court = int(p.get("court", 7))
        return {"storeys": 1, "screen": rng.choice(["wall", "wall", "planted"]),
                "court": court}
    if form == "row_house":
        st = _band(p.get("storeys"), (1, 2))
        return {"storeys": rng.randint(st[0], st[1]),
                "front": rng.choice(["lattice", "screen", "open"])}
    if form == "shop_house":
        st = _band(p.get("front_storeys"), (2, 2))
        return {"storeys": rng.randint(st[0], st[1]),
                "trade": rng.choice(["grain", "cloth", "tea", "smith"])}
    if form == "court_large":
        return {"storeys": 1, "yard": rng.choice(["garden", "well", "orchard"])}
    if form == "court_small":
        return {"storeys": 1, "wings": rng.choice(["open", "screened", "closed"])}
    # any other form: its own declared parameters, the grain's storey band clipped into
    # the form's range, every choice the form's own
    decl = _decl(form) or {}
    out = {}
    for key, spec in (decl.get("params") or {}).items():
        if key in ("reach",):
            continue
        if spec[0] == "int":
            lo, hi = int(spec[1]), int(spec[2])
            if key == "storeys":
                st = _band(p.get("storeys"), (lo, hi))
                lo, hi = max(lo, st[0]), min(hi, max(st[0], st[1]))
                if lo > hi:
                    lo = hi
            out[key] = rng.randint(lo, hi)
        elif spec[0] == "choice":
            opts = list(spec[1])
            if key == "shade":
                # a shade carried over the street is the ring's decision (`shade:
                # lanes`, measured by `_shade_lanes`), never a lot's random draw
                opts = [o for o in opts if o != "lane"] or opts
            out[key] = rng.choice(opts)
    return out


def _pack_rows(city: City, blk: dict) -> None:
    """Back-to-back dwelling rows on the lanes, shop terraces on the streets."""
    grain, p = blk["grain"], blk["params"]
    dims = grain_dims(grain, p)
    dwell, front_form = _row_forms(grain, p)
    lane_dir = p.get("lane_dir", "east_west")
    rng = _rng("rows", blk["id"], _lineage(city))
    _x0, _z0, _x1, _z1 = blk["x0"], blk["z0"], blk["x1"], blk["z1"]
    k = blk["ring"]
    bid = blk["id"]
    row_seq = []
    # the street fronts first: shop terraces along the block's street sides
    fronts = (("west", "east") if lane_dir == "east_west" else ("north", "south"))
    lanes_f = (("north", "south") if lane_dir == "east_west" else ("west", "east"))
    fd = dims["front_depth"]
    shop_w = (7, 9) if grain == "courts" else (6, 8)
    for f in fronts:
        _pack_line(city, blk, f, fd, fd, shop_w, front_form, rng, want_street=ROAD,
                   tag="s", row_seq=row_seq)
    # the dwellings on the lanes, back to back
    for f in lanes_f:
        _pack_line(city, blk, f, dims["depth"][0], dims["depth"][1], dims["width"], dwell,
                   rng, want_street=None, tag="d", row_seq=row_seq)
    blk["lots"] = len(row_seq)


def _block_edges(city: City, blk: dict, front: str) -> dict:
    """For each position along the block's front direction, the first land column from
    that side: `{a: edge}` (a = x for north/south fronts, z for west/east)."""
    bid = blk["id"]
    x0, z0, x1, z1 = blk["x0"], blk["z0"], blk["x1"], blk["z1"]
    sub = city.block[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] == bid
    use = city.use[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1]
    sub = sub & (use == LAND)
    out = {}
    if front in ("north", "south"):
        for i in range(sub.shape[0]):
            col = np.nonzero(sub[i])[0]
            if len(col):
                out[x0 + i] = (z0 + int(col.min())) if front == "north" else \
                    (z0 + int(col.max()))
    else:
        for j in range(sub.shape[1]):
            row = np.nonzero(sub[:, j])[0]
            if len(row):
                out[z0 + j] = (x0 + int(row.min())) if front == "west" else \
                    (x0 + int(row.max()))
    return out


def _pack_line(city, blk, front, d_lo, d_hi, w_band, form, rng, *, want_street=None,
               tag="d", row_seq=None, least_front=0.6):
    """Lots along one side of a block, facing out of that side onto its street/lane."""
    edges = _block_edges(city, blk, front)
    if not edges:
        return
    nd = _needs_of(form)
    if nd:
        fp = nd["footprint"]
        hi_min = min(fp[2], fp[3])
        if int(w_band[1]) > hi_min:
            key = ("band", form, tuple(w_band))
            seen = getattr(city, "_band_seen", set())
            if key not in seen:
                seen.add(key)
                city._band_seen = seen
                city.find("design", f"rings[{blk['ring']}].params.lot_width",
                          f"{form} lots {list(w_band)} wide exceed the form's measured "
                          f"footprint ({hi_min} across at most); laid at "
                          f"{min(int(w_band[0]), hi_min)}..{hi_min}", form=form,
                          asked=list(w_band), most=hi_min)
            w_band = (min(int(w_band[0]), hi_min), hi_min)
    keys = sorted(edges)
    k = blk["ring"]
    placed = []
    a = keys[0]
    end = keys[-1]
    alley_every = int(blk["params"].get("alley_every", 6 if blk["grain"] == "rows" else 0))
    run = 0
    while a <= end:
        if alley_every and run >= alley_every and tag == "d":
            a += 2                         # a passage between premises, paved
            run = 0
            continue
        w = rng.randint(int(w_band[0]), int(w_band[1]))
        best = None
        for ww in sorted({w, int(w_band[0]), int(w_band[1])}, key=lambda v: abs(v - w)):
            if a + ww - 1 > end:
                continue
            span = [edges.get(q) for q in range(a, a + ww)]
            if any(s is None for s in span):
                continue
            # the front edge is the innermost of the span's edges, so the lot is inside
            edge = (max(span) if front in ("north", "west") else min(span))
            # a front set back more than 2 from the street is not on it
            if max(span) - min(span) > 2:
                continue
            for dd in range(int(d_hi), int(d_lo) - 1, -1):
                rect = _lot_rect(front, a, ww, edge, dd)
                if _free(city, rect) and _fronts_street(city, _front_probe(rect, front,
                                                                           edge, span),
                                                        front, least_front):
                    best = (rect, ww, dd)
                    break
            if best:
                break
        if not best:
            a += 1
            continue
        rect, ww, dd = best
        _claim(city, rect)
        # the grain's occasional larger household: a courtyard house among the rows, a
        # merchant's large court among the courtyard houses, at a corner of its row
        f_here = form
        mix = blk["params"].get("mix", 0.08)
        if tag == "d" and form in ("row_house",) and rng.random() < mix and \
                ww >= 6 and dd >= 9:
            f_here = "row_house"
        placed.append((rect, f_here))
        run += 1
        a += ww
    # attach neighbours: consecutive lots touching along the row share party walls
    n = len(placed)
    for i, (rect, form) in enumerate(placed):
        att = []
        f1, f2 = _FLANKS[front]
        if i > 0 and _touch(placed[i - 1][0], rect, f1):
            att.append(f1)
        if i < n - 1 and _touch(rect, placed[i + 1][0], f1):
            att.append(f2)
        params = _dwelling_params(city, blk["grain"], blk["params"], form, rng)
        if form == "shop_house" and _front_rank(city, rect, front) >= 3:
            params["storeys"] = rng.choice([2, 3, 3])     # the avenue stands taller
        name = f"r{k}_b{blk['id']}_{tag}{front[0]}{i:02d}"
        _plot_leaf(city, rect, form, front, params, name=name, ring=k, attached=att,
                   block=blk["id"])
        if row_seq is not None:
            row_seq.append(name)


def _front_rank(city: City, rect, front: str) -> int:
    x0, z0, x1, z1 = rect
    ox, oz = _FRONT_OUT[front]
    x = (x0 + x1) // 2 if front in ("north", "south") else (x0 - 1 if front == "west"
                                                            else x1 + 1)
    z = (z0 + z1) // 2 if front in ("west", "east") else (z0 - 1 if front == "north"
                                                          else z1 + 1)
    return int(city.rank[x - city.X0, z - city.Z0]) if city.inside(x, z) else 0


def _front_probe(rect, front, edge, span):
    return rect


def _touch(a, b, flank_low: str) -> bool:
    """Do lots `a` (before) and `b` (after) share an edge along their row?"""
    if flank_low == "west":           # rows along x
        return a[2] + 1 == b[0] and not (a[3] < b[1] or b[3] < a[1])
    return a[3] + 1 == b[1] and not (a[2] < b[0] or b[2] < a[0])


def _pack_curve(city: City, blk: dict) -> None:
    """Stepped frontage facing a curved ring road, in what the rows left: every lot a
    rectangle square to the lattice, its front on the road on the side nearest the
    road's direction, so a round street is lined rather than left as garden."""
    grain, p = blk["grain"], blk["params"]
    dims = grain_dims(grain, p)
    dwell, front_form = _row_forms(grain, p)
    rng = _rng("curve", blk["id"], _lineage(city))
    x0, z0, x1, z1 = blk["x0"], blk["z0"], blk["x1"], blk["z1"]
    bid = blk["id"]
    k = blk["ring"]
    sub = (city.block[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] == bid) \
        & (city.use[x0 - city.X0:x1 - city.X0 + 1, z0 - city.Z0:z1 - city.Z0 + 1] == LAND)
    if sub.sum() < 60:
        return
    form = front_form if grain == "courts" else dwell
    wl, wh = (7, 9) if form == "shop_house" else tuple(dims["width"])
    dl, dh = (10, dims["front_depth"]) if form == "shop_house" else tuple(dims["depth"])
    n = 0
    ii, jj = np.nonzero(sub)
    order = sorted(zip(ii.tolist(), jj.tolist()),
                   key=lambda c: city.bearing[c[0] + x0 - city.X0, c[1] + z0 - city.Z0])
    for (i, j) in order:
        x, z = x0 + i, z0 + j
        if city.use[x - city.X0, z - city.Z0] != LAND:
            continue
        for front in ("north", "south", "west", "east"):
            ox, oz = _FRONT_OUT[front]
            if not city.inside(x + ox, z + oz):
                continue
            if city.use[x + ox - city.X0, z + oz - city.Z0] not in (ROAD,):
                continue
            w = rng.randint(wl, wh)
            done = False
            for dd in range(dh, dl - 1, -1):
                for shift in range(0, w):
                    if front in ("north", "south"):
                        a = x - shift
                    else:
                        a = z - shift
                    edge = z if front in ("north", "south") else x
                    rect = _lot_rect(front, a, w, edge, dd)
                    if _free(city, rect) and _fronts_street(city, rect, front, 0.5):
                        _claim(city, rect)
                        params = _dwelling_params(city, grain, p, form, rng)
                        _plot_leaf(city, rect, form, front, params,
                                   name=f"r{k}_b{bid}_c{front[0]}{n:02d}", ring=k,
                                   attached=[], block=bid)
                        n += 1
                        done = True
                        break
                if done:
                    break
            if done:
                break
    blk["curve_lots"] = n


def _pack_estates(city: City, blk: dict) -> None:
    """Walled multi-court compounds along the block's street sides."""
    p = blk["params"]
    dims = grain_dims("estates", p)
    rng = _rng("estates", blk["id"], _lineage(city))
    lane_dir = p.get("lane_dir", "east_west")
    fronts = ("north", "south") if lane_dir == "east_west" else ("west", "east")
    k = blk["ring"]
    n = 0
    for f in fronts:
        edges = _block_edges(city, blk, f)
        keys = sorted(edges)
        if not keys:
            continue
        a, end = keys[0] + 1, keys[-1] - 1
        while a <= end:
            w = rng.randint(dims["width"][0], dims["width"][1])
            placed = False
            for ww in (w, dims["width"][0]):
                if a + ww - 1 > end:
                    continue
                span = [edges.get(q) for q in range(a, a + ww)]
                if any(s is None for s in span) or max(span) - min(span) > 2:
                    continue
                edge = max(span) if f in ("north", "west") else min(span)
                for dd in range(dims["depth"][1], dims["depth"][0] - 1, -2):
                    rect = _lot_rect(f, a, ww, edge, dd)
                    if _free(city, rect) and _fronts_street(city, rect, f, 0.6):
                        compose_compound(city, rect, f, ring=k, block=blk["id"],
                                         name=f"r{k}_b{blk['id']}_e{f[0]}{n:02d}",
                                         params=p, rng=rng)
                        n += 1
                        a += ww + 1
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                a += 2
    blk["estates"] = n
    if n == 0 and blk["cells"] > 400:
        # a block too small or too broken for an estate carries courtyard houses
        _pack_rows(city, {**blk, "grain": "courts",
                          "params": {"lane_dir": lane_dir, "court": 9,
                                     "lot_width": [17, 22], "lot_depth": [18, 22],
                                     "dwelling": "courtyard_house", "front": "court_large",
                                     "front_depth": 16}})


def _pack_fields(city: City, blk: dict) -> None:
    """The farmland composer: hamlets of farmsteads round the country road crossings,
    then the block parcelled into fields along its long side, each parcel given the use
    its ground can carry (`_ground_fit`): a field on level dry land, a graded field on a
    gentle slope, an orchard on a steeper one, the land left as found where it is too
    steep or wet -- so fields follow the land and never sit on water or a cliff."""
    p = blk["params"]
    dims = grain_dims("fields", p)
    rng = _rng("fields", blk["id"], _lineage(city))
    k = blk["ring"]
    bid = blk["id"]
    n = 0
    # hamlets: farmsteads at the block's corners that face two roads
    x0, z0, x1, z1 = blk["x0"], blk["z0"], blk["x1"], blk["z1"]
    fs = int(p.get("farmstead", 20))
    corners = [((x0, z0), ("north", "west")), ((x1 - fs + 1, z0), ("north", "east")),
               ((x0, z1 - fs + 1), ("south", "west")), ((x1 - fs + 1, z1 - fs + 1),
                                                         ("south", "east"))]
    hamlet = float(p.get("hamlet", 0.6))
    for (cx, cz), fronts in corners:
        if rng.random() > hamlet:
            continue
        for gx in range(0, 2):
            for gz in range(0, 2):
                ox = cx + (gx * (fs + 1) if fronts[1] == "west" else -gx * (fs + 1))
                oz = cz + (gz * (fs + 1) if fronts[0] == "north" else -gz * (fs + 1))
                rect = (ox, oz, ox + fs - 1, oz + fs - 1)
                f = fronts[0] if gz == 0 else fronts[1]
                if gx and gz:
                    continue
                if not _free(city, rect) or not _fronts_street(city, rect, f, 0.5):
                    continue
                fit = _ground_fit(city, rect, "farmstead")
                if fit is None:
                    continue
                _claim(city, rect)
                _plot_leaf(city, rect, "farmstead", f,
                           {"storeys": rng.randint(1, 2),
                            "yard_use": rng.choice(["store", "byre", "stack"])},
                           name=f"r{k}_b{bid}_h{n:02d}", ring=k, attached=[], block=bid)
                n += 1
    # the fields: parcels along the block's longer side
    fw = dims["width"]
    along_x = (x1 - x0) >= (z1 - z0)
    step_a = rng.randint(fw[0], fw[1])
    step_b = max(12, step_a // 2) if p.get("strips", True) else step_a
    nf = 0
    lanes = int(p.get("field_lane", 1))
    a0, a1 = (x0, x1) if along_x else (z0, z1)
    b0, b1 = (z0, z1) if along_x else (x0, x1)
    for u in range(a0, a1 + 1, step_a + lanes):
        for v in range(b0, b1 + 1, step_b + lanes):
            ua, ub = u, min(a1, u + step_a - 1)
            va, vb = v, min(b1, v + step_b - 1)
            rect = (ua, va, ub, vb) if along_x else (va, ua, vb, ub)
            if rect[2] - rect[0] < 5 or rect[3] - rect[1] < 5:
                continue
            sub = city.rect_use(rect)
            bl = city.block[rect[0] - city.X0:rect[2] - city.X0 + 1,
                            rect[1] - city.Z0:rect[3] - city.Z0 + 1]
            inside = (sub == LAND) & (bl == bid)
            if inside.mean() < 0.85:
                continue
            fit = _ground_fit(city, rect, "field")
            if fit is None:
                continue
            kind, graded = fit
            sub[inside] = OPEN
            params = ({"crop": rng.choice(["grain", "grain", "roots", "beet", "mixed"]),
                       "layout": rng.choice(["strips", "quarters", "plots"])}
                      if kind == "field" else
                      {"planting": "rows" if kind == "orchard" else "copse",
                       "floor": "turf"})
            city.leaves.append({"kind": "area", "name": f"r{k}_b{bid}_p{nf:03d}",
                                "type": "field" if kind == "field" else "grove",
                                "seed": _rng("seed", bid, nf).randint(0, 99),
                                "params": params, "x0": rect[0], "z0": rect[1],
                                "x1": rect[2], "z1": rect[3], "ring": k,
                                "role": "farmland", "use": kind,
                                "voice": _voice(city, k)})
            if graded:
                city.graded = getattr(city, "graded", [])
                city.graded.append(rect)
            nf += 1
    blk["farmsteads"] = n
    blk["fields"] = nf


#: what open ground can carry, by use: (most wet share, most relief as found for the use
#: as found, most relief it is graded to carry). Relief is p90 - p10 of the ground.
GROUND_FIT = {"field": (0.03, 3, 9), "farmstead": (0.0, 4, 10), "orchard": (0.08, 18, 18),
              "garden": (0.02, 3, 8), "yard": (0.02, 3, 8), "grove": (0.1, 24, 24),
              "plaza": (0.0, 2, 6), "square": (0.0, 2, 6), "market": (0.0, 2, 6)}


#: the steepest steady grade a use is laid on as found (blocks per column)
GRADE_MOST = {"field": 0.35, "orchard": 0.6, "grove": 0.8, "farmstead": 0.15}


def _ground_fit(city: City, rect, kind: str):
    """`(use, graded)` the ground under `rect` can carry for `kind`, or None.

    On designed ground (a terrace, a podium) every use fits: the ground is made for it.
    On ground kept as found, the ground decides: too wet or too steep for the use, it
    falls to the next use (a field to an orchard, an orchard to woodland) and at the
    end to landscape as found, never to a field laid over a lake or a cliff."""
    x0, z0, x1, z1 = rect
    sl = (slice(x0 - city.X0, x1 - city.X0 + 1), slice(z0 - city.Z0, z1 - city.Z0 + 1))
    if (city.treat[sl] == HARD).mean() > 0.5:
        return (kind, False)
    # ground a terrace or a podium will be made for (the treatment is laid after the
    # packing, from the ring's or ward's policy)
    rk = city.ring[sl]
    if rk.size and rk.max() >= 0:
        k = int(np.bincount(rk[rk >= 0].ravel()).argmax())
        g = _ward_ground(city, k, (x0 + x1) / 2, (z0 + z1) / 2)
        if g.get("policy") in ("terrace", "podium"):
            return (kind, False)
    found = np.asarray(city.ground[0])[sl].astype(np.int32)
    wet = city.wet[sl].mean()
    # a steady slope is land as much as a level is: judge the ground's roughness about
    # its own plane, and the plane's grade separately (a field climbs a gentle hill)
    ii, jj = np.meshgrid(np.arange(found.shape[0]), np.arange(found.shape[1]),
                         indexing="ij")
    A = np.stack([ii.ravel(), jj.ravel(), np.ones(ii.size)], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, found.ravel().astype(np.float64), rcond=None)
    resid = found.ravel() - A @ coef
    grade = float(np.hypot(coef[0], coef[1]))
    rel = int(np.percentile(resid, 90) - np.percentile(resid, 10))
    if grade > GRADE_MOST.get(kind, 0.35):
        rel = max(rel, int(np.percentile(found, 90) - np.percentile(found, 10)))
    order = {"field": ("field", "orchard", "grove"), "farmstead": ("farmstead",),
             "garden": ("garden", "grove"), "yard": ("yard", "grove"),
             "grove": ("grove",)}.get(kind, (kind,))
    for use in order:
        w_max, flat, graded = GROUND_FIT.get(use, (0.02, 3, 8))
        if wet > w_max:
            continue
        if rel <= flat:
            return (use, False)
        if rel <= graded:
            return (use, True)
    return None


def _leftovers(city: City) -> None:
    """What no lot took: owned open ground by the block's grain, never unowned."""
    from scipy import ndimage
    land = city.use == LAND
    lab, n = ndimage.label(land, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    objs = ndimage.find_objects(lab)
    # a leftover piece's identity is its land, as a block's is (`_stable_ids`)
    ids = _stable_ids(lab, n, objs, (getattr(city, "prior_open", None)))
    city.open_id = ids[lab].astype(np.int32)
    by_id = {b["id"]: b for b in city.blocks}
    kinds = {"courts": "garden", "rows": "yard", "estates": "garden", "fields": "field",
             "open": "grove", "clusters": "yard"}
    for li, sl in enumerate(objs):
        if sl is None:
            continue
        i = int(ids[li + 1])
        mm = lab[sl] == li + 1
        cells = int(mm.sum())
        k = int(city.ring[sl][mm][0])
        b = city.block[sl][mm]
        b = int(np.bincount(b[b >= 0]).argmax()) if (b >= 0).any() else -1
        grain = by_id[b]["grain"] if b in by_id else "open"
        if cells < OPEN_LEAST or min(mm.shape) < 4:
            # a sliver: paved into the street it opens on
            city.use[sl][mm] = COURT
            continue
        city.use[sl][mm] = OPEN
        x0, z0 = sl[0].start + city.X0, sl[1].start + city.Z0
        rect = (x0, z0, sl[0].stop - 1 + city.X0, sl[1].stop - 1 + city.Z0)
        kind = kinds.get(grain, "garden")
        gp = by_id[b]["params"] if b in by_id else {}
        if grain == "open":
            # the open grain's own cover: one word, or a list mixed piece by piece
            cover = _open_cover(city, k, gp, i)
            if cover == "as_found":
                continue
            kind = cover
        if grain == "fields":
            # the farmland composer has parcelled what fields can hold; what is left of
            # a farm block is its land as found (hedgerow, meadow, bank), not a field
            continue
        fit = _ground_fit(city, rect, kind)
        if fit is None:
            continue
        kind = {"orchard": "grove"}.get(fit[0], fit[0])
        if fit[1]:
            city.graded = getattr(city, "graded", [])
            city.graded.append(rect)
        rng = _rng("open", i, _lineage(city))
        params = {"garden": {"layout": rng.choice(["cross", "border", "beds"]),
                             "edge": rng.choice(["hedge", "rail", "kerb"])},
                  "yard": {"fence": rng.choice(["rail", "wall", "open"]),
                           "standing": rng.choice(["corner", "strip"])},
                  "field": {"crop": rng.choice(["grain", "roots", "beet", "mixed"]),
                            "layout": rng.choice(["strips", "quarters", "plots"])},
                  "grove": {"planting": rng.choice(["rows", "copse"]), "floor": "turf"}}[kind]
        trees = (gp or {}).get("trees") or _ring_param(city, k, "trees")
        if kind == "grove" and trees:
            params["crown"] = trees
        covers = _covers(city, k, gp)
        if grain == "clusters" or (grain == "open" and len(covers) > 1):
            # open ground round houses and a pool is laid in tiles that lie wholly on it
            # -- never over the lane beside it or the water inside it -- and a mixed
            # cover is mixed tile by tile, so gardens and palms stand in turn
            for n, t in enumerate(_open_tiles(city, sl, mm)):
                kd = covers[(n + i) % len(covers)] if grain == "open" else kind
                prm = _open_params(kd, _rng("open", i, n, _lineage(city)), trees)
                city.leaves.append({"kind": "area", "name": f"r{k}_o{i}_{n}", "type": kd,
                                    "seed": _rng("seed", "open", i, n).randint(0, 99),
                                    "params": prm, "x0": t[0], "z0": t[1], "x1": t[2],
                                    "z1": t[3], "ring": k, "role": "open",
                                    "voice": _voice(city, k)})
            continue
        _area_leaves(city, rect, kind, name=f"r{k}_o{i}", ring=k, params=params,
                     role="open", only_use=OPEN, mask=mm, claimed=True)


def _open_tiles(city: City, sl, mm, tile: int = 13, least: int = 5) -> list:
    """Rectangles lying wholly on one open piece (`mm` over the frame slice `sl`): the
    piece's bounding box cut into tiles, each shrunk off its edges until every cell is
    the piece's, kept where at least `least` a side remains."""
    x0, z0 = sl[0].start, sl[1].start
    H, Wd = mm.shape
    out = []
    for a in range(0, H, tile):
        for b in range(0, Wd, tile):
            i0, i1, j0, j1 = a, min(H, a + tile) - 1, b, min(Wd, b + tile) - 1
            for _ in range(2 * tile):
                sub = mm[i0:i1 + 1, j0:j1 + 1]
                if sub.size == 0 or sub.all():
                    break
                # drop the edge with the most cells off the piece
                bad = {"i0": (~sub[0, :]).sum(), "i1": (~sub[-1, :]).sum(),
                       "j0": (~sub[:, 0]).sum(), "j1": (~sub[:, -1]).sum()}
                e = max(bad, key=bad.get)
                if e == "i0":
                    i0 += 1
                elif e == "i1":
                    i1 -= 1
                elif e == "j0":
                    j0 += 1
                else:
                    j1 -= 1
                if i1 - i0 + 1 < least or j1 - j0 + 1 < least:
                    break
            if i1 - i0 + 1 >= least and j1 - j0 + 1 >= least and \
                    mm[i0:i1 + 1, j0:j1 + 1].all():
                out.append((x0 + i0 + city.X0, z0 + j0 + city.Z0,
                            x0 + i1 + city.X0, z0 + j1 + city.Z0))
    return out


def _ring_param(city: City, k: int, key: str):
    r = city.design["rings"][k] if 0 <= k < len(city.design["rings"]) else {}
    return (r.get("params") or {}).get(key)


def _covers(city: City, k: int, p: dict) -> list:
    c = (p or {}).get("cover") or _ring_param(city, k, "cover") or "grove"
    c = [c] if isinstance(c, str) else list(c)
    return [v for v in c if v in ("grove", "garden", "field", "as_found")] or ["grove"]


def _open_cover(city: City, k: int, p: dict, i: int) -> str:
    covers = _covers(city, k, p)
    return covers[0] if len(covers) == 1 else \
        covers[_rng("cover", k, i, _lineage(city)).randrange(len(covers))]


# ---------------------------------------------------------------- leaves

def _voice(city: City, k: int, role: str = "fabric") -> str | None:
    pal = city.design.get("palette") or {}
    rings = city.design["rings"]
    name = rings[k]["name"] if 0 <= k < len(rings) else None
    if role == "wall" and pal.get("walls"):
        return pal["walls"]
    mon = city.design.get("monument") or {}
    if role in ("monument",) and pal.get("monument"):
        return pal["monument"]
    return (pal.get("rings") or {}).get(name) or rings[k].get("voice")


def _is_monument_ring(city: City, k: int) -> bool:
    mon = city.design.get("monument") or {}
    rings = city.design["rings"]
    return 0 <= k < len(rings) and mon.get("ring") == rings[k]["name"]


def _site(city: City, rect, front: str, attached, level: int, inset: int = 1):
    from .buildlib import site_pad_rect
    pad = site_pad_rect(*rect, attached, inset)
    ox, oz = _FRONT_OUT[front]
    x0, z0, x1, z1 = rect
    px0, pz0, px1, pz1 = pad
    if front in ("north", "south"):
        mid = (px0 + px1) // 2
        cands = sorted(range(px0 + 1, px1), key=lambda a: abs(a - mid))
        edge = pz0 if front == "north" else pz1
        out = z0 - 1 if front == "north" else z1 + 1
        for a in cands:
            if city.inside(a, out) and city.use[a - city.X0, out - city.Z0] in (
                    ROAD, LANE, COURT, GATE):
                return {"pad": list(pad), "floor": int(level), "facing": front,
                        "door": [a, edge], "landing": [a, out], "attached": sorted(attached),
                        "why": "compiled by cityresolve: the lot's front on its street"}
    else:
        mid = (pz0 + pz1) // 2
        cands = sorted(range(pz0 + 1, pz1), key=lambda a: abs(a - mid))
        edge = px0 if front == "west" else px1
        out = x0 - 1 if front == "west" else x1 + 1
        for a in cands:
            if city.inside(out, a) and city.use[out - city.X0, a - city.Z0] in (
                    ROAD, LANE, COURT, GATE):
                return {"pad": list(pad), "floor": int(level), "facing": front,
                        "door": [edge, a], "landing": [out, a], "attached": sorted(attached),
                        "why": "compiled by cityresolve: the lot's front on its street"}
    return None


def _mixed_voice(city: City, ring: int, name: str) -> str | None:
    """The ring's voice, or -- where the design gives the ring a `mix` -- one of the mix's
    voices chosen by the lot's own name (weights as given), so a street of individual
    premises is built in related palettes rather than one."""
    pal = city.design.get("palette") or {}
    rings = city.design["rings"]
    rname = rings[ring]["name"] if 0 <= ring < len(rings) else None
    mix = (pal.get("mix") or {}).get(rname)
    if not mix:
        return _voice(city, ring)
    total = float(sum(mix.values()))
    h = _rng("mix", name).random() * total
    for v, w in sorted(mix.items()):
        h -= float(w)
        if h <= 0:
            return v
    return sorted(mix)[-1]


def _plot_leaf(city, rect, form, front, params, *, name, ring, attached, block=None,
               role="urban", extra=None):
    city.leaves.append({"kind": "plot", "name": name, "type": form,
                        "seed": _rng("seed", name).randint(0, 99), "params": params,
                        "x0": rect[0], "z0": rect[1], "x1": rect[2], "z1": rect[3],
                        "front": front, "attached": sorted(attached), "inset": 1,
                        "ring": ring, "block": block, "role": role,
                        "voice": _mixed_voice(city, ring, name), **(extra or {})})


def _hall_leaf(city, rect, facing, *, name, ring, params, passage=False):
    x0, z0, x1, z1 = rect
    _claim(city, rect, HALL)
    city.leaves.append({"kind": "plot", "name": name, "type": "ceremonial_hall",
                        "seed": _rng("seed", name).randint(0, 99),
                        "params": dict(params), "x0": x0, "z0": z0, "x1": x1, "z1": z1,
                        "front": facing, "attached": [], "inset": 0, "ring": ring,
                        "role": "civic", "passage": bool(passage),
                        # the monument's voice on the monument only: a hall in an estate
                        # is in its ring's colour, not the palace's (rank)
                        "voice": _voice(city, ring, "monument" if _is_monument_ring(
                            city, ring) else "fabric")})


def _area_leaves(city, rect, kind, *, name, ring, params=None, role="open", only_use=None,
                 skip_use=(), mask=None, claimed=False, trees=None):
    """Area leaves over a rect, split into tiles an area type can build. `kind` may be
    a list: each tile then takes one of them in turn (a mixed cover)."""
    kinds = list(kind) if isinstance(kind, (list, tuple)) else None
    x0, z0, x1, z1 = rect
    w, h = x1 - x0 + 1, z1 - z0 + 1
    nx, nz = max(1, math.ceil(w / AREA_MAX)), max(1, math.ceil(h / AREA_MAX))
    for i in range(nx):
        for j in range(nz):
            a0 = x0 + i * w // nx
            a1 = x0 + (i + 1) * w // nx - 1
            b0 = z0 + j * h // nz
            b1 = z0 + (j + 1) * h // nz - 1
            if a1 - a0 < 2 or b1 - b0 < 2:
                continue
            sub = city.use[a0 - city.X0:a1 - city.X0 + 1, b0 - city.Z0:b1 - city.Z0 + 1]
            if only_use is not None and not (sub == only_use).any():
                continue
            if skip_use and np.isin(sub, skip_use).mean() > 0.5:
                continue
            kd, prm = kind, dict(params or {})
            if kinds:
                kd = kinds[(i + j) % len(kinds)]
                prm = _open_params(kd, _rng("open", name, i, j), trees)
            city.leaves.append({"kind": "area", "name": f"{name}_{i}{j}", "type": kd,
                                "seed": _rng("seed", name, i, j).randint(0, 99),
                                "params": prm, "x0": a0, "z0": b0,
                                "x1": a1, "z1": b1, "ring": ring, "role": role,
                                "voice": _voice(city, ring)})


def _open_params(kind: str, rng, trees=None) -> dict:
    """An open tile's parameters, chosen as `_leftovers` chooses them."""
    if kind == "garden":
        return {"layout": rng.choice(["cross", "border", "beds"]),
                "edge": rng.choice(["hedge", "rail", "kerb"])}
    if kind == "field":
        return {"crop": rng.choice(["grain", "roots", "beet", "mixed"]),
                "layout": rng.choice(["strips", "quarters", "plots"])}
    if kind == "grove":
        return {"planting": rng.choice(["rows", "copse"]), "floor": "turf",
                **({"crown": trees} if trees else {})}
    return {}


# ---------------------------------------------------------------- compounds

def compose_compound(city: City, rect, front: str, *, ring: int, block, name: str,
                     params: dict, rng) -> None:
    """A walled multi-court compound in `rect`, entered from `front`: a gate hall in the
    compound wall, a front court, a main hall with side ranges, rear residential courts
    and a garden. The subordinate composition the estates and the palace wings share."""
    x0, z0, x1, z1 = rect
    along = x1 - x0 + 1 if front in ("north", "south") else z1 - z0 + 1
    deep = z1 - z0 + 1 if front in ("north", "south") else x1 - x0 + 1
    courts = int(params.get("courts", 2))
    hs = _band(params.get("hall_storeys"), (1, 2))
    garden = params.get("garden", "rear")
    gate_pos = params.get("gate", "centre")
    _claim(city, rect, COURT)

    def L(u0, v0, u1, v1):
        """Local (u deep from the front, v along from the low flank) -> world rect."""
        if front == "north":
            return (x0 + v0, z0 + u0, x0 + v1, z0 + u1)
        if front == "south":
            return (x1 - v1, z1 - u1, x1 - v0, z1 - u0)
        if front == "west":
            return (x0 + u0, z1 - v1, x0 + u1, z1 - v0)
        return (x1 - u1, z0 + v0, x1 - u0, z0 + v1)

    lvl_face = _OPP[front]          # buildings inside face back toward the gate
    side_face = {"north": ("east", "west"), "south": ("west", "east"),
                 "west": ("north", "south"), "east": ("south", "north")}[front]
    # the gate hall in the front wall, and the front range either side of it facing in
    # (the street sees a built front, not a blank wall)
    gw = 11 if along >= 36 else 9
    gd = 8
    gv0 = (along - gw) // 2 if gate_pos == "centre" else along - gw - 3
    _hall_leaf(city, L(0, gv0, gd - 1, gv0 + gw - 1), front, name=f"{name}_gate",
               ring=ring, params={"use": "gate", "storeys": 1, "eaves": 1, "platform": 0,
                                  "roof": "gable"}, passage=True)
    fr_d = 7
    for tag, v0, v1 in (("fl", 1, gv0 - 2), ("fh", gv0 + gw + 1, along - 2)):
        if v1 - v0 + 1 >= 9:
            _hall_leaf(city, L(1, v0, fr_d, v1), lvl_face, name=f"{name}_{tag}", ring=ring,
                       params={"use": "side", "storeys": 1, "eaves": 1, "platform": 0,
                               "roof": "gable"})
    # the compound wall: an open polyline from one flank of the gate round to the other
    corners_local = [(0, gv0 + gw), (0, along - 1), (deep - 1, along - 1), (deep - 1, 0),
                     (0, 0), (0, gv0 - 1)]
    path = []
    for (u, v) in corners_local:
        r = L(u, v, u, v)
        path.append([r[0], r[1]])
    city.leaves.append({"kind": "edge", "name": f"{name}_wall", "type": "wall",
                        "seed": _rng("seed", name, "wall").randint(0, 99),
                        "params": {"height": 5, "width": 1, "crown": "solid"},
                        "path": path, "width": 1, "ring": ring, "role": "compound",
                        "voice": _voice(city, ring)})
    for (a_, b_) in zip(path, path[1:]):
        for (x, z) in _edge_cells([tuple(a_), tuple(b_)], 1):
            if city.inside(x, z):
                city.use[x - city.X0, z - city.Z0] = WALL
    # the first court: side ranges down both flanks, the main hall across its far end, a
    # passage beside the hall at either end into the rear court
    side_w = 9 if along >= 40 else 7
    fc = max(12, min(20, deep // 4))
    mh_d = 11 if deep >= 60 else 9
    u0 = gd + 1
    u_hall = u0 + fc
    for tag, v0 in (("sl", 1), ("sh", along - side_w - 1)):
        r = L(u0, v0, u_hall - 2, v0 + side_w - 1)
        if (r[2] - r[0]) >= 6 and (r[3] - r[1]) >= 6:
            _hall_leaf(city, r, side_face[0 if tag == "sl" else 1], name=f"{name}_{tag}",
                       ring=ring, params={"use": "side", "storeys": 1, "eaves": 1,
                                          "platform": 0, "roof": "gable"})
    mh = L(u_hall, side_w + 3, u_hall + mh_d - 1, along - side_w - 4)
    hp = {"use": "rear", "storeys": rng.randint(hs[0], hs[1]), "eaves": 1,
          "platform": 1, "roof": "hip_gable"}
    # a compound whose purpose is its hall (a temple, a ministry) says so: its main
    # hall's rank, eaves and platform are the design's
    for key in ("use", "eaves", "platform", "roof", "tiers"):
        if params.get(f"hall_{key}") is not None:
            hp[key] = params[f"hall_{key}"]
    _hall_leaf(city, mh, lvl_face, name=f"{name}_hall", ring=ring, params=hp)
    u = u_hall + mh_d + 1
    # the rear courts: a residential court behind the hall, flanked by planted strips
    garden_d = 0 if garden != "rear" else max(0, min(16, (deep - u) // 3))
    rear_d = deep - 1 - u - garden_d
    n_courts = max(0, min(max(1, courts - 1), rear_d // 16))
    for c in range(n_courts):
        cd = rear_d // max(1, n_courts)
        cw = min(along - 5, 32)
        v0 = (along - cw) // 2
        r = L(u + 1, v0, u + min(cd, 22) - 1, v0 + cw - 1)
        if r[2] - r[0] >= 9 and r[3] - r[1] >= 9:
            city.leaves.append({"kind": "plot", "name": f"{name}_court{c}",
                                "type": "court_large",
                                "seed": _rng("seed", name, c).randint(0, 99),
                                "params": {"storeys": 1, "yard": rng.choice(
                                    ["garden", "well", "orchard"])},
                                "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3],
                                "front": front, "attached": [], "inset": 0, "ring": ring,
                                "role": "urban", "voice": _voice(city, ring),
                                "court_of": name})
            _claim(city, r, HALL)
        u += cd
    if garden == "rear" and garden_d >= 6:
        r = L(deep - garden_d - 1, 2, deep - 3, along - 3)
        city.leaves.append({"kind": "area", "name": f"{name}_garden", "type": "grove",
                            "seed": _rng("seed", name, "g").randint(0, 99),
                            "params": {"planting": rng.choice(["copse", "rows"]),
                                       "floor": "turf"},
                            "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3], "ring": ring,
                            "role": "garden", "voice": _voice(city, ring)})
        sub = city.rect_use(r)
        sub[sub == COURT] = OPEN
    city.compounds = getattr(city, "compounds", [])
    city.compounds.append({"name": name, "rect": list(rect), "front": front,
                           "ring": ring, "block": block})


# ---------------------------------------------------------------- 9b. the forms' needs

_NEEDS_CACHE: dict = {}


def _needs_of(tname: str):
    if tname not in _NEEDS_CACHE:
        from . import pipeline
        p = os.path.join(os.path.dirname(__file__), "..", "..", "types", f"{tname}.py")
        try:
            _NEEDS_CACHE[tname] = pipeline.load_type(p)["needs"]
        except Exception:                          # noqa: BLE001 -- no needs known
            _NEEDS_CACHE[tname] = None
    return _NEEDS_CACHE[tname]


def _fit_needs(city: City) -> None:
    """Every plot and area inside its form's measured footprint band: a lot a form cannot
    stand on is narrowed on a free side (the columns given to the open ground beside it),
    an area tile shrunk off a broken size; one that still cannot is dropped and its
    ground stays owned open ground. The forms' own measured limits, applied at compile
    time rather than discovered as refusals at construction."""
    from .pipeline import needs_footprint_failure
    keep = []
    dropped = 0
    narrowed = 0
    for lf in city.leaves:
        if lf["kind"] not in ("plot", "area"):
            keep.append(lf)
            continue
        nd = _needs_of(lf["type"])
        if not nd or not needs_footprint_failure(lf, nd):
            keep.append(lf)
            continue
        ok = False
        orig = (lf["x0"], lf["z0"], lf["x1"], lf["z1"])
        front = lf.get("front")
        att = set(lf.get("attached") or [])
        for n in range(1, 9):
            x0, z0, x1, z1 = orig
            if lf["kind"] == "area":
                cand = [(x0, z0, x1 - n, z1), (x0, z0, x1, z1 - n),
                        (x0, z0, x1 - n, z1 - n)]
            else:
                cand = []
                if front in ("north", "south"):
                    if "east" not in att:
                        cand.append((x0, z0, x1 - n, z1))
                    if "west" not in att:
                        cand.append((x0 + n, z0, x1, z1))
                    cand.append((x0, z0 + (n if front == "south" else 0), x1,
                                 z1 - (n if front == "north" else 0)))
                else:
                    if "south" not in att:
                        cand.append((x0, z0, x1, z1 - n))
                    if "north" not in att:
                        cand.append((x0, z0 + n, x1, z1))
                    cand.append((x0 + (n if front == "east" else 0), z0,
                                 x1 - (n if front == "west" else 0), z1))
            for r in cand:
                if r[2] - r[0] < 2 or r[3] - r[1] < 2:
                    continue
                trial = {**lf, "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]}
                if not needs_footprint_failure(trial, nd):
                    lf.update(x0=r[0], z0=r[1], x1=r[2], z1=r[3])
                    ok = True
                    break
            if ok:
                break
        if ok:
            narrowed += 1
            # the columns given up are the open ground beside the lot
            a = city.use[orig[0] - city.X0:orig[2] - city.X0 + 1,
                         orig[1] - city.Z0:orig[3] - city.Z0 + 1]
            inner = np.zeros(a.shape, bool)
            inner[lf["x0"] - orig[0]:lf["x1"] - orig[0] + 1,
                  lf["z0"] - orig[1]:lf["z1"] - orig[1] + 1] = True
            if lf["kind"] == "plot":
                a[(~inner) & (a == LOT)] = COURT
            keep.append(lf)
        else:
            dropped += 1
            city._dropped = getattr(city, "_dropped", {})
            city._dropped[lf["type"]] = city._dropped.get(lf["type"], 0) + 1
            if lf["kind"] == "plot":
                city.rect_use(orig)[city.rect_use(orig) == LOT] = OPEN
    city.leaves = keep
    if narrowed or dropped:
        city.find("info", "forms", f"{narrowed} leaf/leaves narrowed into their form's "
                  f"measured footprint and {dropped} dropped (their ground kept open)",
                  narrowed=narrowed, dropped=dropped,
                  by_type=getattr(city, "_dropped", {}))


def _retile_areas(city: City, tile: int = 14) -> None:
    """No area leaf over a building: an area type levels and dresses its whole rect, so a
    court drawn round the halls that stand in it would be laid over their platforms. An
    area that meets a plot is re-tiled into the pieces of it that are clear; its paving
    is the designed ground's either way."""
    blocked = np.zeros(city.use.shape, bool)
    for lf in city.leaves:
        if lf["kind"] == "plot":
            blocked[lf["x0"] - city.X0 - 1:lf["x1"] - city.X0 + 2,
                    lf["z0"] - city.Z0 - 1:lf["z1"] - city.Z0 + 2] = True
        elif lf["kind"] == "edge":
            for a, b in zip(lf["path"], lf["path"][1:]):
                for (x, z) in _edge_cells([tuple(a), tuple(b)], lf.get("width", 1)):
                    if city.inside(x, z):
                        blocked[x - city.X0, z - city.Z0] = True
    blocked |= np.isin(city.use, (WALL, GATE, OUT))
    # made water is the pool's own ground: nothing else is laid over it
    blocked |= city.use == WATER
    out = []
    for lf in city.leaves:
        if lf["kind"] != "area" or lf.get("type") == "pool":
            out.append(lf)
            continue
        sl = (slice(lf["x0"] - city.X0, lf["x1"] - city.X0 + 1),
              slice(lf["z0"] - city.Z0, lf["z1"] - city.Z0 + 1))
        if not blocked[sl].any():
            out.append(lf)
            continue
        n = 0
        for a0 in range(lf["x0"], lf["x1"] + 1, tile):
            for b0 in range(lf["z0"], lf["z1"] + 1, tile):
                a1, b1 = min(a0 + tile - 1, lf["x1"]), min(b0 + tile - 1, lf["z1"])
                if a1 - a0 < 4 or b1 - b0 < 4:
                    continue
                if blocked[a0 - city.X0:a1 - city.X0 + 1, b0 - city.Z0:b1 - city.Z0 + 1].any():
                    continue
                out.append({**lf, "name": f"{lf['name']}_t{n}", "x0": a0, "z0": b0,
                            "x1": a1, "z1": b1})
                n += 1
    city.leaves = out


# ---------------------------------------------------------------- 10. ground

def _ground(city: City) -> None:
    """Designed level and treatment per column, and a paving code."""
    found = np.asarray(city.ground[0], np.int32)
    rings = city.design["rings"]
    use = city.use
    for k, r in enumerate(rings):
        m = city.ring == k
        pol = r["ground"]["policy"]
        L = city.level_of.get(k)
        if pol == "terrace" and L is not None and int(r["ground"].get("step") or 0) > 0:
            _stepped_terrace(city, k, r, L, found)
        elif pol in ("terrace", "podium") and L is not None:
            city.target[m] = L
            city.treat[m] = HARD
            city.treat[m & np.isin(use, (ROAD, LANE, COURT, GATE))] = STREET
        elif pol == "graded":
            from scipy import ndimage
            sm = ndimage.uniform_filter(found.astype(np.float32), size=15)
            city.target[m] = np.round(sm[m]).astype(np.int16)
            city.treat[m] = SOFT
            city.treat[m & np.isin(use, (ROAD, LANE, COURT, GATE))] = STREET
        else:                                   # preserve
            city.treat[m] = KEEP
            st = m & np.isin(use, (ROAD, LANE, COURT, GATE))
            from scipy import ndimage
            sm = ndimage.uniform_filter(found.astype(np.float32), size=21)
            # a country road the ground is too steep to carry at a walkable grade is not
            # laid: its line stays field (a radial is always laid, and ramped)
            gy, gx = np.gradient(sm)
            steep = np.hypot(gx, gy) > 0.45
            drop = st & steep & (city.rank < 3)
            city.use[drop] = OPEN
            st &= ~drop
            city.target[st] = np.round(_lipschitz(sm, st, 0.5)[st]).astype(np.int16)
            city.treat[st] = STREET
            wet_keep = m & city.wet & ~st
            city.treat[wet_keep] = WATER_T
    # a podium stands above the terraces round it, however they step: its level is the
    # highest of the ring outside it along their common edge, plus its rise
    for k, r in enumerate(rings):
        if r["ground"]["policy"] != "podium" or k + 1 >= len(rings):
            continue
        from scipy import ndimage
        m = city.ring == k
        edge = ndimage.binary_dilation(m, iterations=40) & (city.ring == k + 1) & \
            (city.target != NO_TARGET)
        if edge.any():
            L = int(np.max(city.target[edge])) + int(r["ground"].get("rise") or 0)
            if L > int(city.level_of.get(k) or -999):
                city.level_of[k] = L
                city.levels[r["name"]] = L
                city.target[m] = L
    # walls stand on the higher of the two rings they part
    wm = use == WALL
    if wm.any():
        from scipy import ndimage
        # a wall stands at the level of the street along its inner face (its apron), so
        # its towers and stairs meet the street; where the ground beside it is kept it
        # climbs that ground, graded along its run
        st_t = np.where((city.treat == STREET) & (city.target != NO_TARGET),
                        city.target, -9999).astype(np.int32)
        hi_st = ndimage.maximum_filter(st_t, size=2 * APRON + 5)
        hi = ndimage.maximum_filter(np.where(city.target == NO_TARGET, -9999,
                                             city.target).astype(np.int32), size=7)
        base = np.where(hi_st > -9999, hi_st, hi)
        kept_side = ndimage.maximum_filter((city.treat == KEEP) & (city.ring >= 0) |
                                           (city.ring < 0), size=9) & wm
        sm = ndimage.uniform_filter(found.astype(np.float32), size=15)
        climb = _lipschitz(np.maximum(sm, np.where(base > -9999, base, -9999)), wm, 1.0)
        tw = np.where(base > -9999, base, found).astype(np.int16)
        city.target[wm] = tw[wm]
        city.target[kept_side] = np.round(climb[kept_side]).astype(np.int16)
        city.treat[wm] = WALL_T
        # the berm outside a wall at the wall's level: the nearest wall column's
        ap = getattr(city, "apron", None)
        if ap is not None and ap.any():
            berm = ap & (city.use == OPEN)
            idx = ndimage.distance_transform_edt(~wm, return_distances=False,
                                                 return_indices=True)
            near_t = city.target[idx[0], idx[1]]
            city.target[berm] = near_t[berm]
            city.treat[berm] = HARD
    # open ground graded to carry its use on kept land: a soft slope to its own median
    for rect in getattr(city, "graded", []):
        sl = (slice(rect[0] - city.X0, rect[2] - city.X0 + 1),
              slice(rect[1] - city.Z0, rect[3] - city.Z0 + 1))
        keep = (city.treat[sl] == KEEP) & (city.use[sl] == OPEN)
        if keep.any():
            med = int(np.median(found[sl][keep]))
            city.target[sl][keep] = med
            city.treat[sl][keep] = SOFT
    # building pads in preserved rings: their own level, the median of the ground under
    for lf in city.leaves:
        if lf["kind"] != "plot":
            continue
        x0, z0, x1, z1 = lf["x0"], lf["z0"], lf["x1"], lf["z1"]
        sl = (slice(x0 - city.X0, x1 - city.X0 + 1), slice(z0 - city.Z0, z1 - city.Z0 + 1))
        if (city.treat[sl] == KEEP).all() or (city.treat[sl] == SOFT).all():
            lvl = int(np.median(found[sl]))
            city.target[sl] = lvl
            city.treat[sl] = HARD
    _gate_ramps(city)
    city.pave[np.isin(use, (ROAD, GATE))] = 1
    city.pave[use == LANE] = 2
    city.pave[use == COURT] = 3
    # every area at the designed level of its ground (a compiled court floor)
    for lf in city.leaves:
        if lf["kind"] != "area":
            continue
        sl = (slice(lf["x0"] - city.X0, lf["x1"] - city.X0 + 1),
              slice(lf["z0"] - city.Z0, lf["z1"] - city.Z0 + 1))
        t = city.target[sl]
        if (t != NO_TARGET).mean() > 0.5 and (city.treat[sl] != KEEP).mean() > 0.5:
            lf["court_site"] = {"floor": int(np.median(t[t != NO_TARGET])),
                                "margin": [lf["x0"], lf["z0"], lf["x1"], lf["z1"]],
                                "why": "the designed ground's level under the area"}
    # every plot's compiled site, at the ground now designed
    for lf in city.leaves:
        if lf["kind"] != "plot":
            continue
        rect = (lf["x0"], lf["z0"], lf["x1"], lf["z1"])
        sl = (slice(rect[0] - city.X0, rect[2] - city.X0 + 1),
              slice(rect[1] - city.Z0, rect[3] - city.Z0 + 1))
        t = city.target[sl]
        lvl = int(np.median(t[t != NO_TARGET])) if (t != NO_TARGET).any() else 64
        if lf["type"] == "ceremonial_hall":
            lf["floor"] = lvl
            lf["site"] = _site(city, rect, lf["front"], [], lvl, inset=0) or \
                _hall_site(rect, lf["front"], lvl)
            continue
        st = _site(city, rect, lf["front"], lf.get("attached") or [], lvl,
                   int(lf.get("inset", 1)))
        lf["floor"] = lvl
        if st is None:
            city.find("info", f"lot {lf['name']}", "no street cell outside its front "
                      "for a door; built with the library's own siting")
        else:
            lf["site"] = st


def _ward_ground(city: City, k: int, x: float, z: float) -> dict:
    """The ring's ground policy, overridden by the ward the point stands in."""
    r = city.design["rings"][k]
    g = dict(r.get("ground") or {})
    b = B.bearing_of(x + 0.5 - city.outline.centre[0], z + 0.5 - city.outline.centre[1])
    for wd in r.get("wards") or []:
        a, c = wd["from"], wd["to"]
        if ((a <= b <= c) if a <= c else (b >= a or b <= c)) and wd.get("ground"):
            g.update(wd["ground"])
            break
    return g


_BROAD: dict = {}


def _broad(city: City, k: int, found, follow: float, L: int):
    key = (id(city), k, follow)
    if key not in _BROAD:
        from scipy import ndimage
        m = city.ring == k
        g = np.where(m & ~city.wet, found, np.nan).astype(np.float32)
        have = ~np.isnan(g)
        num = ndimage.gaussian_filter(np.where(have, g, 0), follow)
        den = ndimage.gaussian_filter(have.astype(np.float32), follow)
        _BROAD[key] = np.where(den > 1e-3, num / np.maximum(den, 1e-3), L)
    return _BROAD[key]


def _lipschitz(field, mask, slope: float, iters: int = 400):
    """`field` on `mask` held to at most `slope` blocks of rise per column between
    neighbouring mask columns: the mean of its upper and lower Lipschitz envelopes over
    the mask (a road or a wall graded along its own run)."""
    from scipy import ndimage
    f = field.astype(np.float32)
    big = np.float32(1e6)
    up = np.where(mask, f, big)
    lo = np.where(mask, f, -big)
    fp = np.ones((3, 3), bool)
    for _ in range(iters):
        u2 = np.minimum(up, ndimage.minimum_filter(up, footprint=fp) + slope)
        l2 = np.maximum(lo, ndimage.maximum_filter(lo, footprint=fp) - slope)
        u2 = np.where(mask, u2, big)
        l2 = np.where(mask, l2, -big)
        if np.array_equal(u2, up) and np.array_equal(l2, lo):
            break
        up, lo = u2, l2
    return np.where(mask, (up + lo) / 2, f)


def _stepped_terrace(city: City, k: int, r: dict, L: int, found) -> None:
    """A terraced ring on broken ground: every block a flat terrace at its own ground's
    level, quantized to the ring's `step` and held within `relief` of the ring's level;
    streets and lanes graded between the blocks they part. The blocks' edges against a
    lower street are retained (`designground` HARD edges), so a hillside is a stair of
    terraces along its streets rather than one cut."""
    from scipy import ndimage
    step = max(1, int(r["ground"].get("step") or 3))
    relief = int(r["ground"].get("relief") or 12)
    m = city.ring == k
    use = city.use
    blk = city.block
    lv = np.full(use.shape, np.nan, np.float32)
    # the broad lie of the land, not its bumps: the found ground (dry, within the ring)
    # smoothed over `follow` blocks, so neighbouring terraces step by a course or two
    follow = float(r["ground"].get("follow") or 28)
    g = np.where(m & ~city.wet, found, np.nan).astype(np.float32)
    have_g = ~np.isnan(g)
    num = ndimage.gaussian_filter(np.where(have_g, g, 0), follow)
    den = ndimage.gaussian_filter(have_g.astype(np.float32), follow)
    broad = np.where(den > 1e-3, num / np.maximum(den, 1e-3), L)
    for b in city.blocks:
        if b["ring"] != k:
            continue
        sl = (slice(b["x0"] - city.X0, b["x1"] - city.X0 + 1),
              slice(b["z0"] - city.Z0, b["z1"] - city.Z0 + 1))
        mm = blk[sl] == b["id"]
        med = float(np.median(broad[sl][mm]))
        # a ward may ask its own ground (a hill quarter following its hill)
        g_w = _ward_ground(city, k, (b["x0"] + b["x1"]) / 2, (b["z0"] + b["z1"]) / 2)
        st_b = max(1, int(g_w.get("step") or step))
        rl_b = int(g_w.get("relief") if g_w.get("relief") is not None else relief)
        if g_w.get("follow") and float(g_w["follow"]) != follow:
            med = float(np.median(_broad(city, k, found, float(g_w["follow"]), L)[sl][mm]))
        q = L + st_b * round((med - L) / st_b)
        q = int(max(L - rl_b, min(L + rl_b, q)))
        b["level"] = q
        # a block's land and everything the packer put on it (lots, halls, open ground)
        own = np.isin(use[sl], (LOT, HALL, OPEN, LAND, COURT)) & (blk[sl] == b["id"])
        lv[sl][own] = q
    # what no block claims -- squares, compounds' courts, the landmarks -- takes the
    # level of the nearest block
    have = ~np.isnan(lv) & m
    if not have.any():
        city.target[m] = L
        city.treat[m] = HARD
        return
    idx = ndimage.distance_transform_edt(~have, return_distances=False,
                                         return_indices=True)
    near = lv[idx[0], idx[1]]
    street = m & np.isin(use, (ROAD, LANE, COURT, GATE))
    # streets: the mean of the terraces either side, smoothed along their run
    sm = ndimage.uniform_filter(np.where(m, near, L).astype(np.float32), size=17)
    body = m & ~street
    city.target[body] = np.round(near[body]).astype(np.int16)
    city.treat[body] = HARD
    # ...held to one block per column along the street network, so a hill quarter's
    # streets are ramps (slab steps), never a stair of whole blocks
    sm = _lipschitz(sm, street, 0.9)
    city.target[street] = np.round(sm[street]).astype(np.int16)
    city.treat[street] = STREET


def _hall_site(rect, front, lvl):
    x0, z0, x1, z1 = rect
    if front in ("north", "south"):
        a = (x0 + x1) // 2
        edge, out = (z0, z0 - 1) if front == "north" else (z1, z1 + 1)
        return {"pad": list(rect), "floor": lvl, "facing": front, "door": [a, edge],
                "landing": [a, out], "attached": [], "why": "the hall's axis"}
    a = (z0 + z1) // 2
    edge, out = (x0, x0 - 1) if front == "west" else (x1, x1 + 1)
    return {"pad": list(rect), "floor": lvl, "facing": front, "door": [edge, a],
            "landing": [out, a], "attached": [], "why": "the hall's axis"}


def _gate_ramps(city: City) -> None:
    """Where a street passes a gate, grade it through: the passage at the wall's level
    and the street outside ramped from it to the level the street already has farther
    out, one block per `RAMP_RUN` columns, over the whole width of the street -- read
    off the designed ground itself, so a stepped terrace's own levels are met."""
    RAMP_RUN = 3
    for gp in city.gates:
        ux, uz = B.bearing_vec(gp["bearing"])
        half = gp["width"] // 2
        wt = None
        # the wall's level at the gate, and the passage through it
        for s_ in range(-6, 8):
            x = int(round(gp["x"] + ux * s_))
            z = int(round(gp["z"] + uz * s_))
            if city.inside(x, z) and city.use[x - city.X0, z - city.Z0] == WALL:
                wt = int(city.target[x - city.X0, z - city.Z0])
                break
        if wt is None:
            # a radial's gate: the cells on the axis are the passage, not the wall; the
            # wall's level is its body's beside the passage, and the gate form stands
            # its base across the wall's pad at that level
            wts = []
            for s_ in range(-4, 5):
                for t_ in range(-half - 8, half + 9):
                    x = int(round(gp["x"] + ux * s_ - uz * t_))
                    z = int(round(gp["z"] + uz * s_ + ux * t_))
                    if city.inside(x, z) and city.use[x - city.X0, z - city.Z0] == WALL:
                        wts.append(int(city.target[x - city.X0, z - city.Z0]))
            if wts:
                _gate_steps(city, gp, max(wts))
            continue
        # the first street column outside the wall and its apron, and its level farther
        # out
        def lvl(s_):
            x = int(round(gp["x"] + ux * s_))
            z = int(round(gp["z"] + uz * s_))
            if not city.inside(x, z):
                return None
            t = int(city.target[x - city.X0, z - city.Z0])
            return None if t == NO_TARGET else t
        for side, sgn in (("out", 1), ("in", -1)):
            s0 = APRON + 2 if side == "out" else 3
            far = None
            for s_ in range(s0 + 4, s0 + 60):
                v = lvl(sgn * s_)
                if v is not None and city.use[
                        int(round(gp["x"] + ux * sgn * s_)) - city.X0,
                        int(round(gp["z"] + uz * sgn * s_)) - city.Z0] in (ROAD, LANE):
                    far = (s_, v)
                    if abs(v - wt) * RAMP_RUN + 2 <= s_:
                        break
            if far is None:
                continue
            s_end, v_end = far
            dl = v_end - wt
            run = max(1, min(s_end, abs(dl) * RAMP_RUN + 2))
            for s_ in range(0, s_end + 1):
                frac = min(1.0, s_ / run)
                want = int(round(wt + dl * frac))
                for t_ in range(-half, half + 1):
                    x = int(round(gp["x"] + ux * sgn * s_ - uz * t_))
                    z = int(round(gp["z"] + uz * sgn * s_ + ux * t_))
                    if not city.inside(x, z):
                        continue
                    i, j = x - city.X0, z - city.Z0
                    u = city.use[i, j]
                    if u == WALL:
                        continue
                    if u in (ROAD, GATE, LANE, OPEN):
                        city.target[i, j] = want
                        city.treat[i, j] = STREET
                        if u == OPEN:
                            city.use[i, j] = ROAD


def _gate_steps(city: City, gp: dict, wt: int) -> None:
    """A street carried through a gate on its radial: the passage and the gate's pad
    at the wall's level, and each street outside the pad met by a flight graded one
    block a column (`designground` lays slab steps) over the street's own width, down
    or up to the level the street already has -- so no gate is a sheer step."""
    ux, uz = B.bearing_vec(gp["bearing"])
    pad = gp.get("pad", 15) // 2 + 1
    rw = 5
    for rd in city.design.get("radials") or []:
        if abs(((rd["bearing"] - gp["bearing"] + 180) % 360) - 180) < 1.0:
            rw = int(rd["width"])
    half = rw // 2 + 1
    for sgn in (1, -1):
        # the street's own level just beyond the pad
        far = None
        for s_ in range(pad + 1, pad + 40):
            x = int(round(gp["x"] + ux * sgn * s_))
            z = int(round(gp["z"] + uz * sgn * s_))
            if not city.inside(x, z):
                break
            i, j = x - city.X0, z - city.Z0
            if city.use[i, j] in (ROAD, LANE, COURT) and city.target[i, j] != NO_TARGET \
                    and not (getattr(city, "apron", None) is not None and city.apron[i, j]):
                far = (s_, int(city.target[i, j]))
                break
        if far is None:
            continue
        s_end, v_end = far
        dl = v_end - wt
        n = abs(dl)
        for s_ in range(0, s_end + n + 1):
            if s_ <= pad:
                want = wt
            else:
                k = s_ - pad
                want = wt + (dl if k >= n else (k if dl > 0 else -k))
            for t_ in range(-half, half + 1):
                x = int(round(gp["x"] + ux * sgn * s_ - uz * t_))
                z = int(round(gp["z"] + uz * sgn * s_ + ux * t_))
                if not city.inside(x, z):
                    continue
                i, j = x - city.X0, z - city.Z0
                u = city.use[i, j]
                if u == WALL:
                    continue
                if u in (ROAD, GATE, LANE, OPEN, COURT):
                    if s_ > pad and abs(int(city.target[i, j]) - want) == 0:
                        continue
                    city.target[i, j] = want
                    city.treat[i, j] = STREET
                    if u == OPEN:
                        city.use[i, j] = ROAD


# ---------------------------------------------------------------- 11. regions

def region_of(city: City, x: int, z: int) -> str:
    return f"{(x - city.X0) // REGION}_{(z - city.Z0) // REGION}"


def _regions(city: City) -> None:
    """Every column owned by one construction region: blocks whole (by centroid), streets
    and open ground by tile; leaves by what they stand on; walls cut at region edges."""
    W = city.W
    ti = (np.arange(W)[:, None] // REGION) * 1000 + (np.arange(W)[None, :] // REGION)
    owner = np.broadcast_to(ti, (W, W)).copy().astype(np.int32)
    # a block goes whole to the region of its centre
    for blk in city.blocks:
        cxb = (blk["x0"] + blk["x1"]) // 2 - city.X0
        czb = (blk["z0"] + blk["z1"]) // 2 - city.Z0
        rid = (cxb // REGION) * 1000 + czb // REGION
        sl = (slice(blk["x0"] - city.X0, blk["x1"] - city.X0 + 1),
              slice(blk["z0"] - city.Z0, blk["z1"] - city.Z0 + 1))
        owner[sl][city.block[sl] == blk["id"]] = rid
    city.owner[:] = owner
    city.owner[city.use == OUT] = -1
    # compounds and the monument band are owned whole with their block / tile
    for lf in city.leaves:
        if lf["kind"] == "edge":
            continue
        if lf["kind"] == "point":
            continue
        cxl = (lf["x0"] + lf["x1"]) // 2 - city.X0
        czl = (lf["z0"] + lf["z1"]) // 2 - city.Z0
        rid = int(owner[min(W - 1, max(0, cxl)), min(W - 1, max(0, czl))])
        lf["region"] = f"{rid // 1000}_{rid % 1000}"
    # monument parts go with the monument's own band region (one region for the axis)
    mon = getattr(city, "monument", None)
    if mon:
        o = mon["origin"]
        for lf in city.leaves:
            if lf["name"].startswith((city.design["monument"].get("name", "monument"),)):
                pass
    # wall leaves: each ring wall cut into runs by region, each carrying its whole ring
    # (path, per-segment floors on the designed ground, gates), so runs built apart
    # level their walk, space their towers and meet at their seams identically
    from . import designground as DG
    for w in city.walls:
        w["floors"] = DG.wall_floors(w["path"], w["width"], city.target, city.X0,
                                     city.Z0, np.asarray(city.ground[0]))
        cells = B.polyline_cells(w["path"])
        runs = []
        cur = None
        cur_pts = []
        for (x, z) in cells:
            rid = region_of(city, x, z)
            if rid != cur:
                if cur_pts:
                    runs.append((cur, cur_pts))
                cur, cur_pts = rid, ([cur_pts[-1]] if cur_pts else [])
            cur_pts.append((x, z))
        if cur_pts:
            runs.append((cur, cur_pts))
        for n, (rid, pts) in enumerate(runs):
            path = _compress(pts)
            if len(path) < 2:
                continue
            leaf = {"kind": "edge", "name": f"{w['name']}_{n:03d}", "type": w["type"],
                    "ring_path": [list(p) for p in w["path"]],
                    "ring_floors": w.get("floors"),
                    "ring_gates": [{"at": [g["x"], g["z"]], "size": 15}
                                   for g in w["gates"]],
                    "seed": _rng("seed", w["name"], n).randint(0, 99),
                    "params": _wall_params(w), "path": [list(p) for p in path],
                    "width": w["width"], "ring": w["ring"], "role": "defensive",
                    "voice": _voice(city, w["ring"], "wall"), "region": rid,
                    "wall": w["name"]}
            city.leaves.append(leaf)
    # gates as points, owned by the region they stand in
    for g in city.gates:
        k = g["ring"]
        is_mon = (city.design.get("monument") or {}).get("ring") == \
            city.design["rings"][k]["name"]
        city.leaves.append({"kind": "point", "name": f"gate_{g['ring_name']}_{int(g['bearing'])}",
                            "type": "ring_gate",
                            "seed": _rng("seed", "gate", k, g["bearing"]).randint(0, 99),
                            "params": {"storeys": 3 if is_mon else 2,
                                       "crown": "pavilion" if is_mon else "hip",
                                       "passage": _gate_passage(city, g)},
                            "at": [g["x"], g["z"]], "facing": g["facing"],
                            "ring": k, "role": "defensive",
                            "voice": _voice(city, k, "wall"),
                            "region": region_of(city, g["x"], g["z"]),
                            "on_wall": g["wall"]})
    regs = {}
    for lf in city.leaves:
        rid = lf.get("region")
        if rid is None:
            continue
        r = regs.setdefault(rid, {"id": rid, "leaves": 0, "plots": 0, "by_type": {}})
        r["leaves"] += 1
        r["plots"] += lf["kind"] == "plot"
        r["by_type"][lf["type"]] = r["by_type"].get(lf["type"], 0) + 1
    for rid, r in regs.items():
        i, j = (int(v) for v in rid.split("_"))
        r["rect"] = [city.X0 + i * REGION, city.Z0 + j * REGION,
                     city.X0 + (i + 1) * REGION - 1, city.Z0 + (j + 1) * REGION - 1]
    city.regions = regs


def _gate_passage(city: City, g: dict) -> int:
    """A gate's passage is at least as wide as the street it carries, as far as the
    gate form can carry it (its declared `passage` range); a street wider than that is
    a finding on the radial, not a silently narrowed gate."""
    w = 5
    for rd in city.design.get("radials") or []:
        if abs(((rd["bearing"] - g["bearing"] + 180) % 360) - 180) < 1.0:
            w = max(w, int(rd["width"]))
    lo, hi = 3, 11
    try:
        from . import pipeline
        spec = pipeline.load_type(os.path.join(os.path.dirname(__file__), "..", "..",
                                               "types", "ring_gate.py"))["params"]["passage"]
        lo, hi = int(spec[1]), int(spec[2])
    except Exception:                              # noqa: BLE001 -- the declared default
        pass
    if w > hi:
        city.find("design", "radials", f"the street through {g.get('wall')} at bearing "
                  f"{g['bearing']} is {w} wide and the gate form carries at most {hi}",
                  asked=w, most=hi)
    return max(lo, min(hi, w))


def _wall_params(w: dict) -> dict:
    if w["type"] == "great_wall":
        return {"height": max(24, min(48, w["height"])), "width": 3,
                "parapet": w.get("crown") if w.get("crown") in ("crenellated", "plain")
                else "crenellated",
                "face": w.get("face") if w.get("face") in ("framed", "banded", "plain",
                                                          "masonry") else "masonry"}
    return {"height": max(3, min(20, w["height"])), "width": max(1, min(5, w["width"])),
            "crown": w.get("crown") if w.get("crown") in ("crenellated", "machicolated",
                                                         "solid") else "crenellated"}


def _compress(pts: list) -> list:
    """A cell chain (8-connected, axial/diagonal steps) as its vertices."""
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        _a, b, c = out[-1], pts[i], pts[i + 1]
        d1 = (b[0] - pts[i - 1][0], b[1] - pts[i - 1][1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            out.append(b)
    out.append(pts[-1])
    return out


# ---------------------------------------------------------------- 12. metrics

def _metrics(city: City) -> None:
    use = city.use
    inside = city.ring >= 0
    by_type = {}
    dwell = 0
    people = 0
    for lf in city.leaves:
        by_type[lf["type"]] = by_type.get(lf["type"], 0) + 1
        if lf["kind"] == "plot" and lf["type"] in OCCUPANTS:
            if lf["type"] not in ("ceremonial_hall",):
                dwell += 1
            people += OCCUPANTS.get(lf["type"], 0)
        elif lf["kind"] == "plot" and (_decl(lf["type"]) or {}).get("function") == \
                "dwelling":
            dwell += 1
            people += 5
    comps = len(getattr(city, "compounds", []))
    people += comps * 8
    rings = []
    for k, r in enumerate(city.design["rings"]):
        m = city.ring == k
        n = int(m.sum())
        rings.append({"name": r["name"], "columns": n,
                      "width": round((r["outer"] - (city.design["rings"][k - 1]["outer"]
                                                    if k else 0)) * city.R),
                      "lots": int((use[m] == LOT).sum()),
                      "built_share": round(float(np.isin(use[m], (LOT, HALL)).mean()), 3)
                      if n else 0,
                      "street_share": round(float(np.isin(use[m], (ROAD, LANE)).mean()), 3)
                      if n else 0,
                      "open_share": round(float((use[m] == OPEN).mean()), 3) if n else 0,
                      "level": city.level_of.get(k)})
    found = np.asarray(city.ground[0], np.int32)
    t = city.target.astype(np.int32)
    ch = (t != NO_TARGET) & inside
    cut = int(np.clip(found - t, 0, None)[ch].sum())
    fill = int(np.clip(t - found, 0, None)[ch].sum())
    moved = int((ch & (np.abs(found - t) > 0)).sum())
    city.metrics = {
        "radius": city.R, "diameter": 2 * city.R, "area": int(inside.sum()),
        "boundary": city.outline.to_json(),
        "leaves": len(city.leaves), "by_type": dict(sorted(by_type.items())),
        "dwellings": dwell, "compounds": comps, "people_estimate": people,
        "rings": rings,
        "walls": [{"name": w["name"], "length": w["length"], "height": w["height"],
                   "gates": len(w["gates"])} for w in city.walls],
        "streets": {"road_columns": int((use == ROAD).sum()),
                    "lane_columns": int((use == LANE).sum())},
        "ground": {"cut": cut, "fill": fill, "columns_moved": moved,
                   "levels": {k: v for k, v in city.levels.items()}},
        "regions": len(city.regions),
        "findings": {s: sum(1 for f in city.findings if f["severity"] == s)
                     for s in ("blocking", "design", "info")},
        "hierarchy": hierarchy(city),
    }


def _est_height(p: dict) -> int:
    """A hall's height over its floor, from its parameters: an estimate for comparing
    ranks in the plan, not a measurement (construction measures the built one)."""
    base = int(p.get("base") or 0)
    plat = 2 * int(p.get("platform") or 0)
    storeys = int(p.get("storeys") or 1)
    tiers = max(int(p.get("tiers") or 0), int(p.get("eaves") or 1))
    post = 6 if p.get("use") == "principal" else 5
    return base + plat + post + 3 + 5 * (storeys - 1) + 4 * max(0, tiers - 1) + 8


def hierarchy(city: City) -> dict:
    """What the plan says about rank, for each measurable relation: the monument's
    precinct, principal hall footprint and estimated height against the largest other
    compound, hall and dwelling. Numbers support a visual judgment; they do not make
    one, and the design's own `hierarchy` statements say which relations it intends."""
    mon = getattr(city, "monument", None)
    k_mon = mon["ring"] if mon else None
    halls = [lf for lf in city.leaves if lf["kind"] == "plot" and lf["type"] in
             ("ceremonial_hall", "round_altar")]

    def fp(lf):
        return (lf["x1"] - lf["x0"] + 1) * (lf["z1"] - lf["z0"] + 1)
    principal = [lf for lf in halls if lf.get("ring") == k_mon and
                 (lf.get("params") or {}).get("use") == "principal"]
    others = [lf for lf in halls if lf.get("ring") != k_mon]
    dwell = [lf for lf in city.leaves if lf["kind"] == "plot" and lf["type"] in OCCUPANTS
             and lf["type"] != "ceremonial_hall"]
    out = {"statements": city.design.get("hierarchy") or []}
    if principal:
        p = max(principal, key=fp)
        out["principal"] = {"name": p["name"], "footprint": fp(p),
                            "est_height": _est_height(p.get("params") or {})}
    if others:
        o = max(others, key=lambda lf: _est_height(lf.get("params") or {}))
        out["tallest_other_hall"] = {"name": o["name"], "footprint": fp(o),
                                     "est_height": _est_height(o.get("params") or {})}
    if dwell:
        out["largest_dwelling"] = max(fp(lf) for lf in dwell)
    if mon:
        m = city.ring == k_mon
        out["precinct_area"] = int(m.sum())
    comps = getattr(city, "compounds", [])
    if comps:
        out["largest_compound"] = max((c["rect"][2] - c["rect"][0] + 1) *
                                      (c["rect"][3] - c["rect"][1] + 1) for c in comps
                                      if c.get("ring") != k_mon) if any(
            c.get("ring") != k_mon for c in comps) else None
    if "principal" in out and "tallest_other_hall" in out:
        out["principal_over_other_height"] = round(
            out["principal"]["est_height"] / max(1, out["tallest_other_hall"]["est_height"]), 2)
    if out.get("precinct_area") and out.get("largest_compound"):
        out["precinct_over_compound_area"] = round(out["precinct_area"] /
                                                   out["largest_compound"], 1)
    return out


# ================================================================ outputs

def plan_tree(city: City, place: str = "place") -> dict:
    """The resolved leaves as a plan tree the construction stage reads (`plan.json`)."""
    return {"intent": f"designed by {city.design.get('id')}: "
                      f"{(city.design.get('brief') or {}).get('silhouette', '')}",
            "centre": (city.design.get("monument") or {}).get("name"),
            "by": "cityresolve", "design": city.design.get("id"),
            "design_digest": CD.digest(city.design),
            "levels": city.levels,
            "parts": _by_region(city)}


def _by_region(city: City) -> list:
    """The leaves grouped by construction region, one `district` group each."""
    groups: dict = {}
    for lf in city.leaves:
        groups.setdefault(lf.get("region") or "none", []).append(lf)
    return [{"name": f"region_{rid}", "kind": "district", "region": rid, "children": lv}
            for rid, lv in sorted(groups.items())]


def network(city: City):
    """The designed streets and courts as a `circulate.Network`, with a threshold at
    every compiled door. Heights are the designed ground."""
    from .circulate import Network, Threshold
    cells = {}
    walk = np.isin(city.use, (ROAD, LANE, COURT, GATE))
    ii, jj = np.nonzero(walk)
    for i, j in zip(ii.tolist(), jj.tolist()):
        y = int(city.target[i, j])
        if y == NO_TARGET:
            y = int(city.ground[0][i, j])
        cells[(i + city.X0, j + city.Z0)] = {"y": y, "rank": int(city.rank[i, j]),
                                             "face": None}
    th = []
    walk_in = {"north": "south", "south": "north", "east": "west", "west": "east"}
    for lf in city.leaves:
        st = lf.get("site")
        if lf["kind"] != "plot" or not st:
            continue
        lx, lz = st["landing"]
        y = cells.get((lx, lz), {}).get("y", st["floor"])
        th.append(Threshold(lf["name"], lx, lz, y, walk_in[st["facing"]],
                            (st["door"][0], st["floor"] + 1, st["door"][1]),
                            step=int(st["floor"] - y)))
    return Network(cells, th, notes={"by": "cityresolve.network"})


def save(city: City, out_dir: str) -> dict:
    """Write the resolved city: `city.json` (record), `city.npz` (rasters), `plan.json`."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "city.npz"), use=city.use, ring=city.ring,
                        rank=city.rank, block=city.block,
                        open_id=getattr(city, "open_id", np.full_like(city.block, -1)),
                        target=city.target,
                        treat=city.treat, pave=city.pave, owner=city.owner,
                        origin=np.array([city.X0, city.Z0]))
    rec = {"design": city.design.get("id"), "design_digest": CD.digest(city.design),
           "centre": [city.cx, city.cz], "radius": city.R,
           "frame": [city.X0, city.Z0, city.W], "levels": city.levels,
           "walls": [{k: v for k, v in w.items() if k != "path"} | {"vertices": len(w["path"])}
                     for w in city.walls],
           "gates": city.gates, "monument": getattr(city, "monument", None),
           "blocks": len(city.blocks), "compounds": getattr(city, "compounds", []),
           "regions": city.regions, "metrics": city.metrics, "findings": city.findings}
    json.dump(rec, open(os.path.join(out_dir, "city.json"), "w"), indent=1)
    json.dump(plan_tree(city), open(os.path.join(out_dir, "plan.json"), "w"), indent=1)
    return rec


# ---------------------------------------------------------------- preview

COLOURS = {OUT: (70, 70, 70), WALL: (120, 30, 30), ROAD: (150, 150, 150),
           LANE: (190, 185, 170), LAND: (170, 150, 90), LOT: (160, 110, 80),
           OPEN: (90, 150, 70), WATER: (60, 100, 200), COURT: (225, 220, 205),
           HALL: (230, 190, 50), GATE: (200, 60, 60)}


def preview(city: City, path: str, *, scale: int = 1, shade: bool = True) -> str:
    """A plan map of the resolved city: land use over the designed ground's shading."""
    from PIL import Image
    use = city.use
    img = np.zeros(use.shape + (3,), np.float32)
    for code, c in COLOURS.items():
        img[use == code] = c
    # lots coloured by form
    for lf in city.leaves:
        if lf["kind"] == "area" and lf.get("role") == "farmland":
            x0, z0, x1, z1 = lf["x0"] - city.X0, lf["z0"] - city.Z0, lf["x1"] - city.X0, \
                lf["z1"] - city.Z0
            c = {"field": (190, 180, 90), "orchard": (70, 120, 50),
                 "grove": (45, 95, 40)}.get(lf.get("use"), (120, 160, 80))
            img[x0:x1 + 1, z0:z1 + 1] = c
            img[x0, z0:z1 + 1] *= 0.8
            img[x0:x1 + 1, z0] *= 0.8
    tint = {"courtyard_house": (150, 120, 95), "row_house": (140, 95, 70),
            "shop_house": (175, 90, 60), "court_large": (190, 150, 90),
            "farmstead": (150, 130, 80)}
    for lf in city.leaves:
        if lf["kind"] == "plot" and lf["type"] in tint:
            x0, z0, x1, z1 = lf["x0"] - city.X0, lf["z0"] - city.Z0, lf["x1"] - city.X0, \
                lf["z1"] - city.Z0
            img[x0:x1 + 1, z0:z1 + 1] = tint[lf["type"]]
            img[x0, z0:z1 + 1] *= 0.7
            img[x0:x1 + 1, z0] *= 0.7
    wet = city.wet & (use != WALL) & (city.treat == WATER_T)
    img[wet] = COLOURS[WATER]
    img[(city.ring < 0) & city.wet] = (50, 80, 160)
    if shade:
        found = np.asarray(city.ground[0], np.float32)
        t = np.where(city.target == NO_TARGET, found, city.target).astype(np.float32)
        gz = np.gradient(t, axis=1)
        gx = np.gradient(t, axis=0)
        s = np.clip(1.0 - 0.08 * (gx + gz), 0.6, 1.3)
        img *= s[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 0, 2)
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path)
    return path
