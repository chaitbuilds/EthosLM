"""The relation solver: the place level, placed by relation. v2, B2.

The spec's relation words existed and only `concentric` was resolved; every other
place was drawn freehand by a model as rectangles, and a walled town was never
checked to stand inside its own wall. This is the place-level planner now: the
defining parts **declare relations** -- `centre`, `perimeter`, `gateway`, `edge`,
`beside_the_centre`, `throughout`, `quarter`, `near`, `along`, `on`, and `concentric`
-- and the solver places them. The model never writes a coordinate at any level.

How it solves, in order:

  * candidates are **seeded from the relation**: a centre part at the plateau cut for
    it or at the site's centre; a perimeter wall as a closed loop about the centre,
    at every half-side from the least the districts need to the site's edge; a gate at
    the middle of each side of its wall; a part beside the centre on the four sides
    of it; a part at the edge just inside the wall; a part `near` another on the four
    sides of that one; a part `along` an edge in a strip beside its longest run; a
    part `on` an edge at a cell of its line;
  * the validators are **hard vetoes** (`VETOES`, in weight order: inside the site,
    no overlap or clearance breach, the type's footprint band, inside the perimeter
    wall, a compound on its plateau) and **soft costs** (`COSTS`: the relief and the
    water under the part, the distance from the relation's anchor, the relief along
    a wall's line, a wall shrunk from the site's edge);
  * **greedy insertion in declaration order**, each part taking the least-cost
    candidate that passes every veto, then a **bounded local improvement**: one pass
    over the parts, each tried at its next-best candidates with the rest fixed, kept
    where the whole costs less;
  * when nothing is feasible for a part, its vetoes are **demoted in weight order** --
    the lightest dropped first, the site's never -- and the least-violating placement
    is recorded with the reason;
  * the districts (`throughout`, `quarter`, `along` groups) tile the ground left over
    inside the wall round the centre, as the ring layout's strips do, and the
    structures the spec declared are spread over them by area, capped by the room
    each has;
  * a place with rings is **one solved case**: `concentric_layout`'s arithmetic is the
    candidate and the validator its veto.

Deterministic under the seed: candidates are ordered by cost and then by their own
coordinates, and the seed orders the ties. Every number is in `layout`.
"""
from __future__ import annotations

import math
import random

from . import pipeline, placeplan, spec as spec_mod, styles
from .placeplan import (_FACING, _SIDES, _answers, _edge_cells, _gate_at, _gate_type,
                        _largest_remainder, _octagon_path, _square_path, _top_params,
                        _wall_for, _wall_inset, _wall_types, compound_ground,
                        wall_face_for, wall_round_for, wall_stairs_for,
                        COMPOUND_MIN, COMPOUND_PLATEAU_SHARE, DISTRICT_FILL, LANE_GAP,
                        RING_COVERAGE, RING_EDGE_INSET, SECTOR_MAX)
from .placeread import inside

#: The hard vetoes, heaviest first. When a part has no feasible candidate they are
#: dropped from the lightest up; `site` is never dropped. Registered.
VETOES = (("site", 100), ("overlap", 90), ("footprint", 80), ("inside", 70),
          ("plateau", 60))

#: The soft costs. Registered.
COSTS = {"relief": 1.0,        # per block of relief under the part
         "water": 100.0,       # times the wet share of the part's ground
         "distance": 0.5,      # per column from the relation's anchor
         "line_relief": 2.0,   # per block of mean rise between neighbouring wall columns
         "line_water": 100.0,  # times the wet share of a wall's line
         "shrink": 0.1,        # per block a perimeter wall's half-side is under the edge
         "wall_distance": 0.1,  # per column a wall's centre is from the place's
         "axis": 10.0}         # a part beside the centre on the gate's side of it

#: How many candidates a part keeps for the improvement pass, and how many passes.
KEEP = 8
IMPROVE_PASSES = 1

#: How a perimeter wall's half-side is stepped between the least it can be and the
#: site's edge, and how many positions about the centre are tried at each.
WALL_STEP_BLOCKS = 8
WALL_SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

#: How far a part stands from the one it is beside or near: the larger clearance of the
#: two, and a lane.
BESIDE_GAP = LANE_GAP


class _Ground:
    """The ground the costs read: relief and water over a rectangle, off the volume."""

    def __init__(self, vol):
        self.h = self.wet = None
        self.x0 = self.z0 = 0
        if vol is None:
            return
        from . import observe
        self.h, self.wet = observe.ground_heights(vol)
        self.x0, self.z0 = int(vol.x0), int(vol.z0)

    def _win(self, rect):
        if self.h is None:
            return None
        x0, z0, x1, z1 = rect
        i0, i1 = x0 - self.x0, x1 - self.x0 + 1
        j0, j1 = z0 - self.z0, z1 - self.z0 + 1
        if i0 < 0 or j0 < 0 or i1 > self.h.shape[0] or j1 > self.h.shape[1] or i0 >= i1 \
                or j0 >= j1:
            return None
        return (slice(i0, i1), slice(j0, j1))

    def relief(self, rect) -> float:
        w = self._win(rect)
        if w is None:
            return 0.0
        win = self.h[w]
        return float(win.max() - win.min())

    def water(self, rect) -> float:
        w = self._win(rect)
        if w is None:
            return 0.0
        return float(self.wet[w].mean())

    def line(self, cells) -> tuple:
        """(mean rise between neighbouring columns, wet share) along a wall's cells."""
        if self.h is None or not cells:
            return 0.0, 0.0
        hs, wets = [], []
        for (x, z) in cells:
            i, j = x - self.x0, z - self.z0
            if 0 <= i < self.h.shape[0] and 0 <= j < self.h.shape[1]:
                hs.append(int(self.h[i, j]))
                wets.append(bool(self.wet[i, j]))
        if len(hs) < 2:
            return 0.0, 0.0
        rise = sum(abs(a - b) for a, b in zip(hs, hs[1:])) / float(len(hs) - 1)
        return rise, sum(wets) / float(len(wets))


def _clean_side(decl: dict) -> int:
    """The largest square pad this type admits, honouring `except`."""
    needs = decl.get("needs") or {}
    a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
    ex = set(needs.get("except") or ())
    clean = [v for v in range(min(a, b), max(c, e) + 1)
             if a <= v <= c and b <= v <= e and v not in ex]
    return max(clean) if clean else 0


def _type_for(family: str, kind: str, decls: dict) -> tuple | None:
    """(name, decl) of the committed type a defining part of this family is built as:
    one named for the family, else the largest of its kind."""
    named = sorted(n for n in decls if n == family or n.startswith(family + "_"))
    for n in named:
        if decls[n] is not None and decls[n].get("kind", "plot") == kind:
            return n, decls[n]
    if named and decls[named[0]] is not None:
        return named[0], decls[named[0]]
    pool = [(n, d) for n, d in decls.items()
            if d is not None and d.get("kind", "plot") == kind and not d.get("passage")]
    if not pool:
        return None
    return sorted(pool, key=lambda nd: (-_clean_side(nd[1]), nd[0]))[0]


def _rect_for(decl: dict, kind: str, cx: int, cz: int, cap: int | None = None) -> tuple:
    """A rectangle of this type's largest size, centred on (cx, cz)."""
    from .buildlib import Builder
    side = _clean_side(decl)
    i = 0 if kind in ("edge", "point", "area") else 2 * Builder.SITE_INSET
    w = side + i
    if cap is not None:
        w = min(w, int(cap))
    w = max(w, 3)
    x0, z0 = cx - w // 2, cz - w // 2
    return (x0, z0, x0 + w - 1, z0 + w - 1)


def _clearance(decls: dict, tname) -> int:
    return int(((decls.get(tname) or {}).get("needs")
                or pipeline.NEEDS_DEFAULT)["clearance"])


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _grow(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def _leaf_rects(leaf: dict) -> list:
    return pipeline.part_rects({**leaf, "name": leaf.get("name")})


class Solver:
    """One solve of one place. `solve()` is the whole of it; the rest is per relation."""

    def __init__(self, spec, site, plateau, decls, voice, vol=None, seed: int = 1):
        self.spec = spec
        self.site = site
        self.plateau = plateau or {}
        self.decls = {n: d for n, d in decls.items() if d is not None}
        self.voice = voice
        self.ground = _Ground(vol)
        self.rng = random.Random(int(seed))
        self.seed = int(seed)
        self.X, self.Z = int(site["origin"][0]), int(site["origin"][1])
        self.S = int(site["size"])
        from .buildlib import Builder
        self.dmin = int(2 * Builder.SITE_INSET + 24)
        self.core = spec_mod.core(spec)
        self.placed: list = []            # leaves, in order
        self.by_name: dict = {}
        self.compounds: list = []
        self.districts: list = []
        self.wall: dict | None = None     # the perimeter wall's record
        self.record: list = []
        self.demoted: list = []
        self.fails: list = []
        self.seq = 1
        # the centre: the plateau cut for the core, else the site's middle
        self.cx, self.cz = self.X + self.S // 2, self.Z + self.S // 2
        self.plateau_rect = None
        if self.plateau.get("rect") and self.core \
                and self.plateau.get("part") == self.core["name"]:
            px0, pz0, px1, pz1 = [int(v) for v in self.plateau["rect"]]
            self.plateau_rect = (px0, pz0, px1, pz1)
            self.cx, self.cz = (px0 + px1) // 2, (pz0 + pz1) // 2

    # --- the whole ---------------------------------------------------------------

    def solve(self) -> tuple:
        parts = list(self.spec["defining_parts"])
        groups = [p for p in parts if p["kind"] == "group" and not spec_mod.compound(p)]
        solid = [p for p in parts if p not in groups]
        # the perimeter before anything that has to stand inside it, the centre before
        # anything beside it: declaration order, with those two pulled forward
        order = sorted(solid, key=lambda p: (
            0 if p["relation"] == "centre" else 1 if p["relation"] == "perimeter"
            else 2 if p["relation"] == "gateway" else 3, parts.index(p)))
        for d in order:
            for k in range(int(d["count"])):
                name = d["name"] if d["count"] == 1 else f"{d['name']}_{k + 1}"
                self._place(d, name, k)
        self._improve()
        self._districts(groups)
        if self.fails:
            return None, self.fails
        return self._place_doc(), []

    # --- one part ------------------------------------------------------------------

    def _place(self, d: dict, name: str, k: int) -> None:
        rel = d["relation"]
        gen = {"centre": self._cands_centre, "perimeter": self._cands_perimeter,
               "gateway": self._cands_gateway, "edge": self._cands_edge,
               "beside_the_centre": self._cands_beside, "near": self._cands_near,
               "along": self._cands_along, "on": self._cands_on,
               "throughout": None, "quarter": None, "concentric": None}.get(rel)
        if gen is None:
            self.fails.append({"part": name, "check": "relation",
                               "why": f"{name}: a {d['kind']} of family {d['family']} "
                                      f"declares `{rel}`, which places a district or a "
                                      f"ring and not a {d['kind']}"})
            return
        cands, why = gen(d, name, k)
        if not cands:
            self.fails.append({"part": name, "check": "relation", "why": why or
                               f"{name}: nothing seeds a candidate for `{rel}`"})
            return
        chosen, dropped, vetoed = self._choose(cands, d, name)
        rec = {"part": name, "relation": rel, "family": d["family"], "kind": d["kind"],
               "candidates": len(cands), "vetoed": vetoed, "demoted": dropped,
               "chosen": None, "cost": None, "kept": []}
        if chosen is None:
            worst = min(cands, key=lambda c: (len(self._vetoes(c, d, name, set())), c["cost"]))
            rec["least_violating"] = {"leaf": _geom(worst["leaf"]),
                                      "vetoes": self._vetoes(worst, d, name, set())}
            self.fails.append({"part": name, "check": "vetoes",
                               "why": f"{name} ({rel}): no candidate of {len(cands)} "
                                      f"passes even with every veto but the site's "
                                      f"demoted; the least violating breaks "
                                      + ", ".join(sorted(rec["least_violating"]["vetoes"])),
                               "vetoes": vetoed})
            self.record.append(rec)
            return
        if dropped:
            self.demoted.append({"part": name, "dropped": dropped,
                                 "why": f"{name} had no feasible candidate under every "
                                        f"veto; {', '.join(dropped)} demoted in weight "
                                        f"order and the least-violating placement taken"})
        self._commit(chosen, d, name)
        rec.update(chosen=_geom(chosen["leaf"]), cost=round(chosen["cost"], 2),
                   kept=[{"leaf": _geom(c["leaf"]), "cost": round(c["cost"], 2)}
                         for c in self._feasible(cands, d, name, set(dropped))[:KEEP]])
        self.record.append(rec)

    def _commit(self, cand: dict, d: dict, name: str) -> None:
        leaf = dict(cand["leaf"])
        leaf["seed"] = self.seq
        self.seq += 1
        if cand.get("compound"):
            self.compounds.append(leaf)
        else:
            self.placed.append(leaf)
        self.by_name[name] = leaf
        if d["relation"] == "perimeter" and leaf.get("kind") == "edge":
            self.wall = {"name": name, "leaf": leaf, "rect": cand["rect"],
                         "half": cand["half"], "inset": cand["inset"],
                         "shape": cand.get("shape", "square")}

    def _order(self, cands: list) -> list:
        for c in cands:
            c.setdefault("tie", self.rng.random())
        return sorted(cands, key=lambda c: (c["cost"], c["key"], c["tie"]))

    def _feasible(self, cands, d, name, dropped: set) -> list:
        return [c for c in self._order(cands) if not self._vetoes(c, d, name, dropped)]

    def _choose(self, cands: list, d: dict, name: str) -> tuple:
        """The least-cost feasible candidate, demoting vetoes from the lightest up."""
        vetoed: dict = {}
        for c in cands:
            for v in self._vetoes(c, d, name, set()):
                vetoed[v] = vetoed.get(v, 0) + 1
        dropped: list = []
        order = [v for v, _w in sorted(VETOES, key=lambda vw: vw[1])]   # lightest first
        while True:
            ok = self._feasible(cands, d, name, set(dropped))
            if ok:
                return ok[0], dropped, vetoed
            nxt = [v for v in order if v not in dropped and v != "site"]
            if not nxt:
                return None, dropped, vetoed
            dropped.append(nxt[0])

    # --- the vetoes -----------------------------------------------------------------

    def _vetoes(self, cand: dict, d: dict, name: str, dropped: set) -> list:
        leaf = cand["leaf"]
        rects = [cand["rect"]] if cand.get("compound") else _leaf_rects(leaf)
        out = []
        X, Z, S = self.X, self.Z, self.S
        if "site" not in dropped:
            for r in rects:
                if not (X <= r[0] and r[2] < X + S and Z <= r[1] and r[3] < Z + S):
                    out.append("site")
                    break
        if "overlap" not in dropped:
            mine = _clearance(self.decls, leaf.get("type")) if not cand.get("compound") else 0
            passage = bool((self.decls.get(leaf.get("type")) or {}).get("passage"))
            for other in self.placed + self.compounds:
                if other.get("name") == name:
                    continue
                o_pass = bool((self.decls.get(other.get("type")) or {}).get("passage"))
                kinds = {leaf.get("kind", "plot"), other.get("kind", "plot")}
                if (passage or o_pass) and "edge" in kinds:
                    continue                  # a gate stands in its wall
                theirs = _clearance(self.decls, other.get("type")) \
                    if other in self.placed else 0
                m = max(mine, theirs)
                orects = _leaf_rects(other) if other in self.placed else \
                    [(other["x0"], other["z0"], other["x1"], other["z1"])]
                if any(_overlaps(a, _grow(b, m)) for a in rects for b in orects):
                    out.append("overlap")
                    break
        if "footprint" not in dropped and not cand.get("compound"):
            decl = self.decls.get(leaf.get("type"))
            if decl and pipeline.needs_footprint_failure(
                    {**leaf, "name": name}, decl["needs"]):
                out.append("footprint")
        if "inside" not in dropped and self.wall is not None \
                and d["relation"] not in ("perimeter", "gateway", "on") \
                and name != self.wall["name"]:
            path = self.wall["leaf"]["path"]
            for r in rects:
                corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]
                if not all(inside(path, c) for c in corners):
                    out.append("inside")
                    break
        if "plateau" not in dropped and cand.get("compound") and self.plateau_rect:
            px0, pz0, px1, pz1 = self.plateau_rect
            x0, z0, x1, z1 = cand["rect"]
            ok = px0 <= x0 and x1 <= px1 and pz0 <= z0 and z1 <= pz1
            share = ((x1 - x0 + 1) * (z1 - z0 + 1)
                     / float((px1 - px0 + 1) * (pz1 - pz0 + 1)))
            if not ok or share < COMPOUND_PLATEAU_SHARE:
                out.append("plateau")
        return out

    # --- the costs
    # ---------------------------------------------------------------------

    def _cost(self, rect, anchor=None) -> float:
        c = COSTS["relief"] * self.ground.relief(rect) \
            + COSTS["water"] * self.ground.water(rect)
        if anchor is not None:
            mx, mz = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
            c += COSTS["distance"] * math.hypot(mx - anchor[0], mz - anchor[1])
        return c

    # --- the candidate generators ----------------------------------------------------

    def _cands_centre(self, d, name, k) -> tuple:
        cx, cz = self.cx, self.cz
        if spec_mod.compound(d):
            if self.plateau_rect:
                rect = self.plateau_rect
            else:
                side = int(compound_ground(spec=self.spec, site_side=self.S)["side"])
                side = max(COMPOUND_MIN, side)
                rect = (cx - side // 2, cz - side // 2,
                        cx - side // 2 + side - 1, cz - side // 2 + side - 1)
            leaf = {"name": name, "defines": d["name"], "x0": rect[0], "z0": rect[1],
                    "x1": rect[2], "z1": rect[3],
                    "notes": (f"The {d['family']} at the centre of the place, over the "
                              f"whole of the ground levelled for it. " + (d.get("notes") or ""))}
            return [{"leaf": leaf, "rect": rect, "compound": True,
                     "cost": self._cost(rect), "key": rect}], None
        got = _type_for(d["family"], d["kind"], self.decls)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']}) at the centre")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        cap = None
        if self.plateau_rect:
            cap = min(self.plateau_rect[2] - self.plateau_rect[0] + 1,
                      self.plateau_rect[3] - self.plateau_rect[1] + 1)
        cands = []
        for dx, dz in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
            if kind == "point":
                leaf = {"name": name, "defines": d["name"], "kind": "point", "type": tname,
                        "params": {}, "at": [cx + dx, cz + dz], "facing": "north",
                        "notes": "the thing at the centre"}
                rect = pipeline.part_rect(leaf)
            else:
                rect = _rect_for(decl, kind, cx + dx, cz + dz, cap=cap)
                leaf = {"name": name, "defines": d["name"], "kind": kind, "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3], "notes": "the thing at the centre, at its largest"}
            cands.append({"leaf": leaf, "rect": rect, "key": rect,
                          "cost": self._cost(rect, (cx, cz))})
        return cands, None

    def _cands_perimeter(self, d, name, k) -> tuple:
        walls = _wall_types(self.decls)
        if not walls:
            return [], (f"{name}: no committed edge type of this place's form has a "
                        f"height parameter to build a wall with")
        # the type named for the family -- a town's `wall` is a wall, not the great wall
        # of a city's outermost ring -- at the top of its band; the tallest edge type
        # where nothing is named for it
        named = [w for w in walls if w[0] == d["family"]] or walls
        tname, tdecl, lo_h, hi_h = named[0]
        height = int(hi_h)
        max_run = int(tdecl["needs"]["footprint"][3])
        width = 3 if "width" in (tdecl.get("params") or {}) else 1
        inset = _wall_inset(tdecl, width)
        octagon = bool(wall_round_for(d, self.spec) and tdecl.get("diagonal"))
        # the least half-side: the centre's ground and every district's, inside the
        # inset, at the coverage the strips round a centre are held to
        need = self._ground_needed() / RING_COVERAGE
        h_min = max(int(math.ceil(math.sqrt(need) / 2.0)) + inset, self.dmin + inset)
        # the largest loop the site holds with the edge inset on every side
        h_max = (self.S - 1 - 2 * RING_EDGE_INSET) // 2
        if h_min > h_max:
            h_min = h_max
        cx, cz = self.cx, self.cz
        cands = []
        hs = list(range(h_max, h_min - 1, -WALL_STEP_BLOCKS))
        if hs[-1] != h_min:
            hs.append(h_min)
        for h in hs:
            for (sx, sz) in WALL_SHIFTS:
                ox, oz = cx + sx * WALL_STEP_BLOCKS, cz + sz * WALL_STEP_BLOCKS
                # clamped into the site less the edge inset, so the loop stands on the
                # place's ground: a wall centred on a plateau near the site's edge is a
                # wall moved in, not one refused
                lo_x, hi_x = self.X + RING_EDGE_INSET + h, self.X + self.S - 1 - RING_EDGE_INSET - h
                lo_z, hi_z = self.Z + RING_EDGE_INSET + h, self.Z + self.S - 1 - RING_EDGE_INSET - h
                if lo_x > hi_x or lo_z > hi_z:
                    continue
                ox, oz = min(max(ox, lo_x), hi_x), min(max(oz, lo_z), hi_z)
                rect = (ox - h, oz - h, ox + h, oz + h)
                if (rect[0], rect[1], rect[2], rect[3]) in [c["rect"] for c in cands]:
                    continue
                path = (_octagon_path(ox, oz, h, max_run) if octagon
                        else _square_path(ox, oz, h, max_run))
                params = {"height": height}
                if width == 3:
                    params["width"] = 3
                face = wall_face_for(d, self.spec)
                if "face" in (tdecl.get("params") or {}):
                    choices = list(tdecl["params"]["face"][1])
                    params["face"] = ("unbroken" if face == "plain"
                                      and wall_stairs_for(d, self.spec) == "sparse"
                                      and "unbroken" in choices else face)
                leaf = {"kind": "edge", "name": name, "defines": d["name"], "type": tname,
                        "params": params, "path": path, "width": width,
                        **({"shape": "octagon"} if octagon else {}),
                        **({"face": face} if face else {}),
                        "notes": f"the circuit of the place, {height} high as `{tname}`, "
                                 f"half-side {h} about ({ox},{oz}); solved"}
                rise, wet = self.ground.line(_edge_cells(leaf))
                cost = (COSTS["line_relief"] * rise + COSTS["line_water"] * wet
                        + COSTS["shrink"] * (h_max - h)
                        + COSTS["wall_distance"] * math.hypot(ox - cx, oz - cz))
                cands.append({"leaf": leaf, "rect": rect, "half": h, "inset": inset,
                              "shape": "octagon" if octagon else "square",
                              "cost": cost, "key": (h_max - h, rect)})
        return cands, None

    def _ground_needed(self) -> float:
        """The columns the districts and the centre need inside a perimeter wall."""
        cols = 0.0
        for p in self.spec["defining_parts"]:
            if p["kind"] == "group" and not spec_mod.compound(p):
                n = int(p.get("structures") or 0)
                cols += n * spec_mod.columns_per_plot(p) / DISTRICT_FILL
        if not cols:
            cols = int(self.spec.get("structures") or 0) * spec_mod.COLUMNS_PER_PLOT / DISTRICT_FILL
        centre = self.by_name.get((self.core or {}).get("name"))
        if centre is not None:
            r = pipeline.part_rect({**centre, "name": centre.get("name")})
            cols += (r[2] - r[0] + 1 + 2 * LANE_GAP) * (r[3] - r[1] + 1 + 2 * LANE_GAP)
        return cols

    def _cands_gateway(self, d, name, k) -> tuple:
        if self.wall is None:
            return [], f"{name}: a gateway is a way through a wall and this place has none"
        gate = _gate_type(self.decls, d.get("family", "gate"))
        if gate is None:
            return [], (f"{name}: no committed passage point type of this place's form "
                        f"builds a gate for {self.wall['name']}")
        w = self.wall
        ox, oz = (w["rect"][0] + w["rect"][2]) // 2, (w["rect"][1] + w["rect"][3]) // 2
        h = w["half"]
        from .buildlib import Builder
        pad = Builder.point_pad(int(w["leaf"]["params"].get("height") or 0),
                                gate[1]["needs"]["footprint"])
        taken = {tuple(p["at"]) for p in self.placed if p.get("kind") == "point"}
        cands = []
        for side in _SIDES:
            gx, gz = _gate_at(ox, oz, h, side)
            if (gx, gz) in taken:
                continue
            leaf = {"kind": "point", "name": name, "defines": d["name"], "type": gate[0],
                    "params": _top_params(gate[1]), "at": [gx, gz],
                    "facing": _FACING[side], "size": int(pad),
                    "notes": f"the way through {w['name']} on its {side} side; solved"}
            rect = pipeline.part_rect(leaf)
            cands.append({"leaf": leaf, "rect": rect, "key": rect,
                          "cost": self._cost(rect), "side": side})
        return cands, None

    def _bounds(self) -> tuple:
        """The rectangle everything inside the place stands in: inside the wall's
        inset, or the site less its edge inset."""
        if self.wall is not None:
            x0, z0, x1, z1 = self.wall["rect"]
            i = self.wall["inset"]
            return (x0 + i, z0 + i, x1 - i, z1 - i)
        e = RING_EDGE_INSET
        return (self.X + e, self.Z + e, self.X + self.S - 1 - e, self.Z + self.S - 1 - e)

    def _around(self, target_rect, d, name, tname, decl, kind, anchor_axis=None) -> list:
        """Candidates of this type on the four sides of a rectangle, a gap away."""
        tx0, tz0, tx1, tz1 = target_rect
        gap = max(_clearance(self.decls, tname), 0) + 1 + BESIDE_GAP
        side_len = _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4)
        cands = []
        for side in _SIDES:
            for along in (0, 1, -1):
                if side in ("north", "south"):
                    cx = (tx0 + tx1) // 2 + along * max(1, (tx1 - tx0) // 2)
                    cz = (tz0 - gap - side_len // 2 - 1) if side == "north" \
                        else (tz1 + gap + side_len // 2 + 1)
                else:
                    cz = (tz0 + tz1) // 2 + along * max(1, (tz1 - tz0) // 2)
                    cx = (tx0 - gap - side_len // 2 - 1) if side == "west" \
                        else (tx1 + gap + side_len // 2 + 1)
                if kind == "point":
                    leaf = {"kind": "point", "name": name, "defines": d["name"],
                            "type": tname, "params": {}, "at": [cx, cz],
                            "facing": _FACING[side], "notes": f"{d['relation']}; solved"}
                    rect = pipeline.part_rect(leaf)
                else:
                    rect = _rect_for(decl, kind, cx, cz)
                    leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                            "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                            "z1": rect[3], "notes": f"{d['relation']}, on the {side} "
                                                    f"side; solved"}
                cost = self._cost(rect, ((tx0 + tx1) / 2.0, (tz0 + tz1) / 2.0))
                if anchor_axis and side == anchor_axis:
                    cost += COSTS["axis"]
                cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": cost,
                              "side": side})
        return cands

    def _cands_beside(self, d, name, k) -> tuple:
        centre = self.by_name.get((self.core or {}).get("name"))
        if centre is None:
            return [], f"{name}: nothing stands at the centre for this part to be beside"
        got = _type_for(d["family"], d["kind"], self.decls)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        rect = pipeline.part_rect({**centre, "name": centre.get("name")}) \
            if "x0" in centre or "at" in centre or "path" in centre \
            else (centre["x0"], centre["z0"], centre["x1"], centre["z1"])
        gate_side = None
        for p in self.placed:
            if p.get("kind") == "point" and (self.decls.get(p.get("type")) or {}).get("passage"):
                gate_side = {"north": "south", "south": "north",
                             "east": "west", "west": "east"}.get(p.get("facing"))
                break
        return self._around(rect, d, name, tname, decl, kind, anchor_axis=gate_side), None

    def _cands_near(self, d, name, k) -> tuple:
        of = d.get("of") or (self.core or {}).get("name")
        target = self.by_name.get(of) or next(
            (c for c in self.compounds if c.get("name") == of), None)
        if target is None:
            return [], f"{name}: `near` names {of!r} and nothing of that name is placed"
        got = _type_for(d["family"], d["kind"], self.decls)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        rect = (pipeline.part_rect({**target, "name": target.get("name")})
                if target.get("kind") else
                (target["x0"], target["z0"], target["x1"], target["z1"]))
        return self._around(rect, d, name, tname, decl, kind), None

    def _cands_edge(self, d, name, k) -> tuple:
        got = _type_for(d["family"], d["kind"], self.decls)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        bx0, bz0, bx1, bz1 = self._bounds()
        side_len = _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4)
        half = side_len // 2
        cands = []
        spots = []
        for t in (0.5, 0.25, 0.75):
            spots += [(int(bx0 + (bx1 - bx0) * t), bz0 + half, "north"),
                      (int(bx0 + (bx1 - bx0) * t), bz1 - half, "south"),
                      (bx0 + half, int(bz0 + (bz1 - bz0) * t), "west"),
                      (bx1 - half, int(bz0 + (bz1 - bz0) * t), "east")]
        for cx, cz, side in spots:
            if kind == "point":
                leaf = {"kind": "point", "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "at": [cx, cz], "facing": _FACING[side],
                        "notes": "at the edge of the place; solved"}
                rect = pipeline.part_rect(leaf)
            else:
                rect = _rect_for(decl, kind, cx, cz)
                leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3], "notes": f"at the {side} edge of the place; solved"}
            cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": self._cost(rect),
                          "side": side})
        return cands, None

    def _target_edge(self, d, name):
        of = d.get("of") or (self.wall or {}).get("name")
        target = self.by_name.get(of)
        if target is None or target.get("kind") != "edge":
            return None, (f"{name}: `{d['relation']}` names {of!r} and no edge of that "
                          f"name is placed")
        return target, None

    def _cands_on(self, d, name, k) -> tuple:
        target, why = self._target_edge(d, name)
        if target is None:
            return [], why
        got = _type_for(d["family"], "point", self.decls)
        if got is None:
            return [], (f"{name}: no committed point type of this place's form builds a "
                        f"{d['family']} on {target['name']}")
        tname, decl = got
        taken = {tuple(p["at"]) for p in self.placed if p.get("kind") == "point"}
        cells = sorted(set(_edge_cells(target)))
        path = [(int(a[0]), int(a[1])) for a in target["path"]]
        # the corners of the run and the quarter points of every segment, on the line
        spots = list(path[:-1])
        for a, b in zip(path, path[1:]):
            for t in (0.25, 0.5, 0.75):
                spots.append((int(round(a[0] + (b[0] - a[0]) * t)),
                              int(round(a[1] + (b[1] - a[1]) * t))))
        cands = []
        for (x, z) in spots:
            if (x, z) in taken or (x, z) not in set(cells):
                continue
            leaf = {"kind": "point", "name": name, "defines": d["name"], "type": tname,
                    "params": {}, "at": [x, z], "facing": "north",
                    "notes": f"on {target['name']}; solved"}
            rect = pipeline.part_rect(leaf)
            cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": self._cost(rect)})
        return cands, None

    def _cands_along(self, d, name, k) -> tuple:
        target, why = self._target_edge(d, name)
        if target is None:
            return [], why
        got = _type_for(d["family"], d["kind"], self.decls)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']}) along {target['name']}")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        gap = _clearance(self.decls, target.get("type")) + int(target.get("width", 1)) // 2 + 1
        side_len = _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4)
        path = [(int(a[0]), int(a[1])) for a in target["path"]]
        bx0, bz0, bx1, bz1 = self._bounds()
        mid_in = ((bx0 + bx1) / 2.0, (bz0 + bz1) / 2.0)
        cands = []
        for a, b in zip(path, path[1:]):
            if a[0] != b[0] and a[1] != b[1]:
                continue
            for t in (0.5, 0.25, 0.75):
                mx = int(round(a[0] + (b[0] - a[0]) * t))
                mz = int(round(a[1] + (b[1] - a[1]) * t))
                # the inside of the run: toward the middle of the bounds
                if a[0] == b[0]:
                    sx = 1 if mid_in[0] > mx else -1
                    cx, cz = mx + sx * (gap + side_len // 2 + 1), mz
                else:
                    sz = 1 if mid_in[1] > mz else -1
                    cx, cz = mx, mz + sz * (gap + side_len // 2 + 1)
                rect = _rect_for(decl, kind, cx, cz)
                leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3], "notes": f"along {target['name']}; solved"}
                cands.append({"leaf": leaf, "rect": rect, "key": rect,
                              "cost": self._cost(rect, (mx, mz))})
        return cands, None

    # --- the improvement
    # ---------------------------------------------------------------

    def _improve(self) -> None:
        """One pass: each placed part tried at its next-best kept candidates with the
        rest fixed; kept where every veto still holds and the whole costs less."""
        for _ in range(IMPROVE_PASSES):
            for rec in self.record:
                if rec.get("chosen") is None or len(rec["kept"]) < 2:
                    continue
                name = rec["part"]
                d = next(p for p in self.spec["defining_parts"]
                         if p["name"] == name or name.startswith(p["name"] + "_"))
                if d["relation"] in ("perimeter", "gateway", "on"):
                    continue                  # a wall moves everything on and in it
                current = self.by_name[name]
                where = self.placed.index(current) if current in self.placed else None
                if where is None:
                    continue
                best = None
                for alt in rec["kept"][1:]:
                    leaf = dict(current)
                    leaf.update({k: v for k, v in alt["leaf"].items()})
                    cand = {"leaf": leaf, "rect": pipeline.part_rect(
                        {**leaf, "name": name}), "cost": alt["cost"]}
                    self.placed[where] = leaf
                    ok = not self._vetoes(cand, d, name, set()) and all(
                        not self._vetoes({"leaf": o, "rect": pipeline.part_rect(
                            {**o, "name": o["name"]}), "cost": 0.0},
                            next(p for p in self.spec["defining_parts"]
                                 if p["name"] == o["name"]
                                 or o["name"].startswith(p["name"] + "_")),
                            o["name"], set())
                        for o in self.placed if o is not leaf)
                    self.placed[where] = current
                    if ok and alt["cost"] < rec["cost"] and (best is None
                                                              or alt["cost"] < best["cost"]):
                        best = cand
                if best is not None:
                    was = rec["cost"]
                    self.placed[where] = best["leaf"]
                    self.by_name[name] = best["leaf"]
                    rec["improved"] = {"from": was, "to": round(best["cost"], 2)}
                    rec["chosen"] = _geom(best["leaf"])
                    rec["cost"] = round(best["cost"], 2)

    # --- the districts
    # --------------------------------------------------------------------

    def _districts(self, groups: list) -> None:
        if not groups:
            return
        bx0, bz0, bx1, bz1 = self._bounds()
        # the core box: the centre and everything beside it, a lane clear
        core_names = {(self.core or {}).get("name")}
        core_names |= {p["name"] for p in self.spec["defining_parts"]
                       if p["relation"] in ("beside_the_centre",)}
        core_rects = []
        for leaf in self.placed + self.compounds:
            base = next((p["name"] for p in self.spec["defining_parts"]
                         if _answers(leaf, p)), None)
            if base in core_names:
                core_rects.append(pipeline.part_rect({**leaf, "name": leaf.get("name")})
                                  if leaf.get("kind") else
                                  (leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"]))
        g = LANE_GAP
        if core_rects:
            cx0 = min(r[0] for r in core_rects) - g
            cz0 = min(r[1] for r in core_rects) - g
            cx1 = max(r[2] for r in core_rects) + g
            cz1 = max(r[3] for r in core_rects) + g
        else:
            cx0, cz0, cx1, cz1 = self.cx, self.cz, self.cx, self.cz
        strips = {"north": (bx0, bz0, bx1, cz0 - 1), "south": (bx0, cz1 + 1, bx1, bz1),
                  "west": (bx0, cz0, cx0 - 1, cz1), "east": (cx1 + 1, cz0, bx1, cz1)}
        # ...less every other placed part, trimmed on the axis that loses least
        others = [pipeline.part_rect({**leaf, "name": leaf.get("name")})
                  for leaf in self.placed
                  if next((p["name"] for p in self.spec["defining_parts"]
                           if _answers(leaf, p)), None) not in core_names
                  and leaf.get("kind") not in ("edge",)
                  and not (self.decls.get(leaf.get("type")) or {}).get("passage")]
        sectors = []
        for sname in _SIDES:
            rect = strips[sname]
            for o in others:
                rect = _trim(rect, _grow(o, g))
                if rect is None:
                    break
            if rect is None:
                continue
            x0, z0, x1, z1 = rect
            if x1 - x0 + 1 < self.dmin or z1 - z0 + 1 < self.dmin:
                continue
            along_x = (x1 - x0) >= (z1 - z0)
            length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
            pieces = max(1, int(math.ceil(length / float(SECTOR_MAX))))
            while pieces > 1 and (length - g * (pieces - 1)) // pieces < self.dmin:
                pieces -= 1
            plen = (length - g * (pieces - 1)) // pieces
            for i in range(pieces):
                start = (x0 if along_x else z0) + i * (plen + g)
                end = start + plen - 1 if i < pieces - 1 else (x1 if along_x else z1)
                r = ((start, z0, end, z1) if along_x else (x0, start, x1, end))
                label = sname if pieces == 1 else f"{sname}_{i + 1}"
                sectors.append((label, r))
        if not sectors:
            self.fails.append({"part": groups[0]["name"], "check": "district",
                               "why": f"no ground {self.dmin} on a side is left inside "
                                      f"the place for its districts"})
            return
        # which group each sector answers: `quarter` parts take one sector each in
        # order, `throughout` parts share the rest; an `along` group takes the strips
        # beside its edge
        assign: dict = {}
        remaining = list(sectors)
        for p in groups:
            if p["relation"] == "along":
                target = self.by_name.get(p.get("of") or (self.wall or {}).get("name"))
                if target is not None:
                    tr = pipeline.part_rect({**target, "name": target.get("name")})
                    mine = [s for s in remaining
                            if _overlaps(_grow(s[1], 2 * g), tr)]
                    for s in mine[:max(1, int(p["count"]))]:
                        assign[s[0]] = p
                        remaining.remove(s)
        for p in groups:
            if p["relation"] == "quarter":
                for _k in range(int(p["count"])):
                    if remaining:
                        s = remaining.pop(0)
                        assign[s[0]] = p
        through = [p for p in groups if p["relation"] == "throughout"]
        for i, s in enumerate(remaining):
            if through:
                assign[s[0]] = through[i % len(through)]
        # the counts: from the ground at each district's density, capped by the room it
        # has, and scaled to what the spec declared where the ground allows
        rows = []
        for label, r in sectors:
            p = assign.get(label)
            if p is None:
                continue
            area = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
            per = spec_mod.columns_per_plot(p)
            cap = int(area * DISTRICT_FILL // per)
            rows.append({"label": label, "rect": r, "part": p, "area": area, "per": per,
                         "cap": cap, "from_ground": min(cap, max(1, spec_mod.structures_for(area, p)))})
        declared = int(self.spec.get("structures") or 0)
        want = declared or sum(row["from_ground"] for row in rows)
        counts = _largest_remainder([row["area"] for row in rows], want,
                                    caps=[row["cap"] for row in rows])
        role_of = {}
        for row, n in zip(rows, counts):
            p = row["part"]
            r = row["rect"]
            role = p.get("role") or spec_mod.read_role(None, p, p["name"])
            role_of[p["name"]] = role
            self.districts.append({
                "name": f"{p['name']}_{row['label']}", "x0": r[0], "z0": r[1],
                "x1": r[2], "z1": r[3], "structures": int(max(1, min(n, row["cap"]))),
                "defines": p["name"],
                **({"voice": p["voice"]} if p.get("voice") else {}),
                "purpose": (f"{p.get('notes') or p['name']} A {p.get('density') or 'medium'}"
                            f", {role} district of the place; solved."),
                "notes": (f"The {row['label'].replace('_', ' ')} strip of the ground "
                          f"round the centre, inside "
                          f"{'the wall' if self.wall else 'the site'}; solved, not drawn.")})
        covered = sum(row["area"] for row in rows)
        inside_cols = (bx1 - bx0 + 1) * (bz1 - bz0 + 1)
        self.district_record = {
            "bounds": [bx0, bz0, bx1, bz1], "core_box": [cx0, cz0, cx1, cz1],
            "sectors": [{"label": l, "rect": list(r), "group": assign.get(l, {}).get("name")}
                        for l, r in sectors],
            "structures": {"declared": declared, "laid": sum(int(d["structures"])
                                                            for d in self.districts),
                           "caps": [row["cap"] for row in rows]},
            "coverage": round(covered / float(inside_cols), 4) if inside_cols else 0.0}

    # --- the document ----------------------------------------------------------------

    def _place_doc(self) -> dict:
        pal = None
        try:
            pal = (dict(styles.VOICES[self.voice]["palette"])
                   if self.voice in styles.VOICES else None)
        except Exception:                        # noqa: BLE001 -- no palette, no material
            pal = None
        core = self.core
        return {
            "intent": (f"A {self.spec['kind']} placed by relation: "
                       + "; ".join(f"{r['part']} {r['relation']}" for r in self.record)
                       + (f"; {len(self.districts)} district(s) round the centre"
                          if self.districts else "")
                       + ". Every part stands where its relation put it, inside the site"
                       + (" and inside the wall" if self.wall else "") + "."
                       + (f" {self.spec['invariants']}" if self.spec.get("invariants") else "")),
            "centre": core["name"] if core else None,
            "voice": self.voice, "palette": pal,
            "circulation_material": (pal or {}).get("floor") or "stone",
            "parts": [dict(p) for p in self.placed],
            "districts": list(self.districts),
            "compounds": [dict(c) for c in self.compounds],
            "layout": {"by": "placesolve.solve_place", "case": "relations",
                       "seed": self.seed, "centre": [self.cx, self.cz],
                       "plateau_rect": list(self.plateau_rect) if self.plateau_rect else None,
                       "wall": ({"name": self.wall["name"], "half": self.wall["half"],
                                 "rect": list(self.wall["rect"]), "inset": self.wall["inset"],
                                 "shape": self.wall["shape"]} if self.wall else None),
                       "solved": self.record, "demoted": self.demoted,
                       "districts": getattr(self, "district_record", None),
                       "registered": {"VETOES": [list(v) for v in VETOES],
                                      "COSTS": dict(COSTS), "KEEP": KEEP,
                                      "IMPROVE_PASSES": IMPROVE_PASSES,
                                      "WALL_STEP_BLOCKS": WALL_STEP_BLOCKS,
                                      "BESIDE_GAP": BESIDE_GAP,
                                      "RING_EDGE_INSET": RING_EDGE_INSET,
                                      "LANE_GAP": LANE_GAP, "SECTOR_MAX": SECTOR_MAX,
                                      "DISTRICT_FILL": DISTRICT_FILL,
                                      "RING_COVERAGE": RING_COVERAGE}}}


def _trim(rect, cut):
    """`rect` less `cut`: the larger remainder on the axis that loses least, or None."""
    if rect is None or not _overlaps(rect, cut):
        return rect
    x0, z0, x1, z1 = rect
    options = []
    if cut[0] > x0:
        options.append((x0, z0, cut[0] - 1, z1))
    if cut[2] < x1:
        options.append((cut[2] + 1, z0, x1, z1))
    if cut[1] > z0:
        options.append((x0, z0, x1, cut[1] - 1))
    if cut[3] < z1:
        options.append((x0, cut[3] + 1, x1, z1))
    if not options:
        return None
    return max(options, key=lambda r: ((r[2] - r[0] + 1) * (r[3] - r[1] + 1), r))


def _geom(leaf: dict) -> dict:
    return {k: v for k, v in leaf.items()
            if k in ("kind", "type", "x0", "z0", "x1", "z1", "at", "facing", "path",
                     "width", "size", "shape")}


def solve_place(spec: dict, site: dict, plateau: dict | None, decls: dict, voice: str,
                vol=None, seed: int = 1) -> tuple:
    """The place level, placed by relation. Returns `(place, fails)` in the shape
    `concentric_layout` returns, and a place with rings **is** that layout: the
    arithmetic is the candidate and `place_failures` the veto."""
    if spec_mod.rings(spec):
        place, fails = placeplan.concentric_layout(spec, site, plateau, decls, voice,
                                                   vol=vol)
        if place is not None:
            place["layout"]["solver"] = {
                "by": "placesolve.solve_place", "case": "concentric",
                "why": "a place with rings is one solved case: the ring arithmetic "
                       "seeds the one candidate and the place validator is its veto"}
        return place, fails
    return Solver(spec, site, plateau, decls, voice, vol=vol, seed=seed).solve()
