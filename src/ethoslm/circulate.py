"""Circulation, built before anything stands on it.

This module is Route C -- procedural, deterministic, no model call. It owns the ground
it crosses, because on a mesa building a lane *is* terrain work: cut, fill, stairs,
retaining. What it does not own is the ground inside a plot.

A cell cannot face two ways at once, so a cell that would be stepped up into from two
different directions is lowered into a landing until no cell needs two facings -- which
is what a real staircase does at a turn. `walk_check()` then re-derives reachability over
the lane graph under the movement rules, and scripts/test_circulation.py re-derives it
again from the emitted blocks with the real Nav model. The planner's own opinion of its
network is not evidence."""
from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .prims import material

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FACING = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}

#: Cost of a lane step over a terrain rise of `d` blocks, before it is cut or filled.
#: Quadratic, so the router prefers a switchback of three easy cells to one hard one --
#: which is how a track up a mesa is actually laid.
def _slope_cost(d: np.ndarray) -> np.ndarray:
    return d.astype(float) ** 2


@dataclass
class Threshold:
    """The few blocks where a lane meets a doorway. Owned by this pass.

        The spec's one addition to shared state, and deliberately the only one: the world is
        readable, so nothing that can be measured needs declaring. Intent cannot be measured.
        A later pass may not obstruct this, and lint E008 enforces it.
        
    """

    id: str
    x: int
    z: int
    y: int                  # lane surface block; you stand at y+1
    facing: str             # from the lane cell into the plot
    door: tuple             # where the building pass must put its door leaf

    def to_json(self):
        return {"id": self.id, "x": self.x, "z": self.z, "y": self.y,
                "facing": self.facing, "door": list(self.door)}


@dataclass
class Network:
    """The planned lane surface: one height and one rank per cell, plus thresholds."""

    cells: dict                     # (x, z) -> {"y": int, "rank": int, "face": str|None}
    thresholds: list                # Threshold
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    # --- queries ---------------------------------------------------------
    def stances(self) -> list:
        """Absolute half-heights a walker occupies on the lane: (x, z, s)."""
        return [(x, z, 2 * (rec["y"] + 1)) for (x, z), rec in self.cells.items()]

    def surface(self) -> list:
        return [(x, z, rec["y"]) for (x, z), rec in self.cells.items()]

    def threshold(self, label: str):
        for t in self.thresholds:
            if t.id == label:
                return t
        return None

    def nearest(self, x: int, z: int):
        """Closest lane cell to a point, as (x, z, y, chebyshev distance)."""
        best = None
        for (cx, cz), rec in self.cells.items():
            d = max(abs(cx - x), abs(cz - z))
            if best is None or d < best[3]:
                best = (cx, cz, rec["y"], d)
        return best

    # --- persistence -----------------------------------------------------
    def to_json(self) -> dict:
        return {
            "cells": [[x, z, r["y"], r["rank"], r["face"]]
                      for (x, z), r in sorted(self.cells.items())],
            "thresholds": [t.to_json() for t in self.thresholds],
            "nodes": self.nodes, "edges": self.edges, "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Network":
        cells = {(int(x), int(z)): {"y": int(y), "rank": int(r), "face": f}
                 for x, z, y, r, f in d["cells"]}
        th = [Threshold(t["id"], t["x"], t["z"], t["y"], t["facing"], tuple(t["door"]))
              for t in d["thresholds"]]
        return cls(cells, th, d.get("nodes", []), d.get("edges", []), d.get("notes", {}))

    def save(self, path: str) -> str:
        json.dump(self.to_json(), open(path, "w"))
        return path

    @classmethod
    def load(cls, path: str) -> "Network | None":
        if not os.path.exists(path):
            return None
        return cls.from_json(json.load(open(path)))


# --------------------------------------------------------------------- routing

def _roughness(h: np.ndarray) -> np.ndarray:
    """Local relief in the 3x3 around each column: how broken the ground is here."""
    p = np.pad(h.astype(int), 1, mode="edge")
    stack = np.stack([p[i:i + h.shape[0], j:j + h.shape[1]]
                      for i in range(3) for j in range(3)])
    return stack.max(axis=0) - stack.min(axis=0)


def _approach_candidates(h: np.ndarray, x0: int, z0: int, rect, per_site: int = 4,
                         front: str | None = None):
    """Where a lane could meet this plot: the best cell just outside each side.

        One candidate per side, so the router can choose which face of a building the town
        arrives at rather than being told. That choice is the whole point of going first.

        **...unless the plan says which face.** v2, C2: a leaf carrying `front` -- the side
        its street is on, which the district compiler writes and a row of party walls
        depends on, because its flanks have a neighbour against them -- offers that side
        and no other; the way in is held on the street. Where that side has no cell in
        the volume, every side, as before.
        
    """
    sx, sz = h.shape
    ax0, az0, ax1, az1 = rect
    rough = _roughness(h)
    out = []
    sides = {
        "north": [(x, az0 - 1) for x in range(ax0, ax1 + 1)],
        "south": [(x, az1 + 1) for x in range(ax0, ax1 + 1)],
        "west": [(ax0 - 1, z) for z in range(az0, az1 + 1)],
        "east": [(ax1 + 1, z) for z in range(az0, az1 + 1)],
    }
    into = {"north": "south", "south": "north", "west": "east", "east": "west"}
    if front in sides:
        only = [(x, z) for x, z in sides[front]
                if 0 <= x - x0 < sx and 0 <= z - z0 < sz]
        if only:
            sides = {front: sides[front]}
    for side, cells in sides.items():
        # **The middle of the side on a tie.** The smoothest cell wins; where every cell
        # of the side is as smooth as the next -- which is every side of every site on a
        # levelled terrace -- the first cell used to win, and the first cell of a side
        # is its corner. The doorway reserved one step in from a corner is the corner of
        # the site, where a field stands its post and a square its lamp. A lane arrives
        # at the middle of a face.
        mid = (len(cells) - 1) / 2.0
        best = None
        for k, (x, z) in enumerate(cells):
            i, j = x - x0, z - z0
            if not (0 <= i < sx and 0 <= j < sz):
                continue
            score = (int(rough[i, j]), abs(k - mid))
            if best is None or score < best[0]:
                best = (score, x, z)
        if best is not None:
            out.append({"x": best[1], "z": best[2], "face": into[side],
                        "rough": best[0][0]})
    out.sort(key=lambda c: c["rough"])
    return out[:per_site]


#: How many sources one Dijkstra call runs at once: sixty-four rows over an 864-square
#: is 380 MB, and the sum over batches is the same work as one call.
DIJKSTRA_BATCH = 64


def _source_distances(graph, src_idx: list, batch: int = DIJKSTRA_BATCH) -> np.ndarray:
    """The distance from every approach cell to every other, `(n, n)`, kept from
    Dijkstra rows that are otherwise discarded as each batch finishes."""
    from scipy.sparse.csgraph import dijkstra
    n = len(src_idx)
    out = np.full((n, n), np.inf)
    cols = np.asarray(src_idx)
    for i in range(0, n, batch):
        rows_ = dijkstra(graph, directed=False, indices=src_idx[i:i + batch])
        out[i:i + batch, :] = rows_[:, cols]
    return out


def _grid_graph(h: np.ndarray, avoid: np.ndarray, max_step: int):
    """4-connected grid over the site, weighted by gradient and by what to keep off.

        Steps steeper than `max_step` get no edge at all: the router must find a way round a
        cliff rather than price its way up one. Route is the scarce resource at 91 blocks of
        relief -- that is the spec's second reason for allocating it first.
        
    """
    from scipy.sparse import coo_matrix

    sx, sz = h.shape
    idx = np.arange(sx * sz).reshape(sx, sz)
    hi = h.astype(np.int32)
    rows, cols, vals = [], [], []
    for dx, dz in ((1, 0), (0, 1)):
        a = idx[:sx - dx, :sz - dz].ravel()
        b = idx[dx:, dz:].ravel()
        d = np.abs(hi[dx:, dz:] - hi[:sx - dx, :sz - dz]).ravel()
        pa = avoid[:sx - dx, :sz - dz].ravel()
        pb = avoid[dx:, dz:].ravel()
        w = 1.0 + _slope_cost(d) + 0.5 * (pa + pb)
        ok = (d <= max_step) & np.isfinite(pa) & np.isfinite(pb)
        rows.append(a[ok]); cols.append(b[ok]); vals.append(w[ok])
    return coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(sx * sz, sx * sz)).tocsr()


def _path(pred: np.ndarray, src: int, dst: int) -> list:
    out = [dst]
    cur = dst
    while cur != src:
        cur = int(pred[cur])
        if cur < 0:
            return []
        out.append(cur)
    out.reverse()
    return out


# ------------------------------------------------------------- the height field

def _lipschitz_from(seed: dict, adj: dict, sign: int) -> dict:
    """The extreme 1-Lipschitz function on the lane graph bounded by `seed`.

        sign=+1 gives the largest function <= seed (all cut, no fill); sign=-1 the smallest
        >= seed (all fill). Standard inf-convolution with graph distance, computed as a
        Dijkstra seeded with the terrain itself. Averaging the two gives a field that is
        1-Lipschitz -- the average of two 1-Lipschitz functions is 1-Lipschitz -- and splits
        the difference between cutting and filling, which is what earthwork does.
        
    """
    best = {c: sign * v for c, v in seed.items()}
    heap = [(sign * v, c) for c, v in seed.items()]
    heapq.heapify(heap)
    while heap:
        d, c = heapq.heappop(heap)
        if d > best[c]:
            continue
        for n in adj[c]:
            nd = d + 1
            if nd < best[n]:
                best[n] = nd
                heapq.heappush(heap, (nd, n))
    return {c: sign * v for c, v in best.items()}


def _solve_heights(terrain: dict, adj: dict) -> dict:
    up = _lipschitz_from(terrain, adj, +1)          # cut only
    down = _lipschitz_from(terrain, adj, -1)        # fill only
    return {c: (up[c] + down[c] + 1) // 2 for c in terrain}


def _needed_facings(c, y: dict, adj: dict) -> set:
    """Directions a walker steps up into `c` from. Two of them cannot both be built."""
    out = set()
    for n in adj[c]:
        if y[n] == y[c] - 1:
            out.add((c[0] - n[0], c[1] - n[1]))
    return out


def _landings(y: dict, adj: dict, rounds: int = 60) -> tuple[dict, int]:
    """Lower any cell that cannot honestly be a stair into a landing.

        Two ways at once: one block cannot face two directions, so a cell stepped up into
        from two sides is walkable from at most one of them. Flattening it to its lower
        neighbours is both the fix and the thing a stonemason would build -- a staircase
        that turns has a landing at the turn.

        A step to nowhere: a rising cell the lane does not continue past is the top step of
        a flight with nothing above it. It walks fine, but its raised quarter overhangs open
        ground, which is a stair facing down-slope by any reading and is what E004 exists to
        catch. Three of them turned up in the first mesa fixture. The top of a flight is a
        landing too.

        Only ever lowers, and the cut-side Lipschitz clamp is reapplied after, so the field
        stays 1-Lipschitz.
        
    """
    def unbuildable(c):
        f = _needed_facings(c, y, adj)
        if len(f) > 1:
            return True
        return bool(f) and (c[0] + next(iter(f))[0], c[1] + next(iter(f))[1]) not in y

    for i in range(rounds):
        bad = [c for c in y if unbuildable(c)]
        if not bad:
            return y, i
        for c in bad:
            y[c] -= 1
        y = _lipschitz_from(y, adj, +1)
    return y, rounds


# ------------------------------------------------------------------- the planner

def parts_to_routing(parts: list, passage=()) -> dict:
    """Until A4 a plan was a list of building footprints and the router's whole job was to
    join them up. A place has three more kinds of part in it and each means something
    different to a network:

      plot   a site, as always: the lane comes to it and reserves a doorstep.
      area   a site too. A square is joined to the lanes on every side -- it *is* a
             piece of circulation -- and the lane stops at its edge because the square's
             own type paves it.
      edge   an **obstacle**. A town wall is not something to route to; it is something
             the network may not cross, and the whole point of a walled district is
             that it is walled.
      point  a site, and if its type declares `PASSAGE` a **crossing**: the one cell of
             the wall the network is allowed through. That is what a gate is, and it is
             the only reason a walled district is not a sealed one.

    `passage` is the set of type names whose declaration says so -- read off the type
    files by the caller, because whether a part is a way through is a fact about the
    type and not about the plan.

    Returns {"sites", "obstacles", "passable"}: the footprints to route between, the
    columns to keep off, and the columns to cross anyway."""
    sites, obstacles, passable = [], set(), set()
    for p in parts:
        kind = p.get("kind", "plot")
        name = p.get("name") or p.get("id") or p.get("label")
        if kind == "edge":
            half = max(1, int(p.get("width", 1))) // 2
            path = [(int(a[0]), int(a[1])) for a in p["path"]]
            for a, b in zip(path, path[1:]):
                # **A 45-degree run is an obstacle too.** v2, A5. it is how a ring is
                # round rather than square -- and this read every segment as axial:
                # `step` came out (1, 0) for a chamfer, `n` was its Manhattan length,
                # and the columns marked were a line twice as long running due east from
                # one end of it. So an octagon's chamfers were not in `obstacles` at all
                # and a lane could be routed straight through the city wall. The frame
                # is the one the library sites in: the run's own direction, and the
                # width along the perpendicular lattice diagonal. A diagonal line of
                # cells is 8-connected, and `_grid_graph` is 4-connected, so the line
                # alone is a barrier -- nothing has to be thickened to seal it.
                if a[0] != b[0] and a[1] != b[1]:
                    sx = 1 if b[0] > a[0] else -1
                    sz = 1 if b[1] > a[1] else -1
                    line = [(a[0] + sx * i, a[1] + sz * i)
                            for i in range(abs(b[0] - a[0]) + 1)]
                    across = (-sz, sx)
                else:
                    step = (0, 1) if a[0] == b[0] else (1, 0)
                    n = abs(b[0] - a[0]) + abs(b[1] - a[1])
                    sign = 1 if (b[0] + b[1]) >= (a[0] + a[1]) else -1
                    line = [(a[0] + step[0] * i * sign, a[1] + step[1] * i * sign)
                            for i in range(n + 1)]
                    across = (step[1], step[0])
                for (cx, cz) in line:
                    for d in range(-half, half + 1):
                        obstacles.add((cx + across[0] * d, cz + across[1] * d))
            continue
        if kind == "point":
            at = p.get("at") or [p.get("x0"), p.get("z0")]
            ax, az = int(at[0]), int(at[-1])
            sites.append({"id": name, "x0": ax, "z0": az, "x1": ax, "z1": az})
            if p.get("type") in set(passage) or p.get("passage"):
                # The crossing is a *line* through the wall and not a hole in it: the
                # gate cell and one cell either side of it along the way you go through.
                # A three-by-three would let the router round the gate rather than
                # through it, which is a gate with a gap beside it.
                ah = {"north": (0, -1), "south": (0, 1),
                      "east": (1, 0), "west": (-1, 0)}[p.get("facing", "north")]
                for i in (-1, 0, 1):
                    passable.add((ax + ah[0] * i, az + ah[1] * i))
            continue
        site = {"id": name,
                "x0": min(p["x0"], p["x1"]), "z0": min(p["z0"], p["z1"]),
                "x1": max(p["x0"], p["x1"]), "z1": max(p["z0"], p["z1"])}
        if p.get("front"):
            site["front"] = p["front"]           # v2, C2: the way in is on this side
        sites.append(site)
    return {"sites": sites, "obstacles": obstacles - passable,
            "passable": passable}


#: How much dearer open ground is than a column already carrying an **arterial**, in the
#: units `_grid_graph` weighs an edge in: a step between two arterial columns costs `1.0
#: + slope` and a step between two ordinary ones `1.0 + slope + ARTERIAL_BONUS`. Big
#: enough that a lane joins the road and runs along it rather than beside it, and
#: applied this way round -- to everything else, rather than as a discount on the road
#: -- because a negative edge weight turns Dijkstra into the wrong algorithm.
ARTERIAL_BONUS = 1.6


def plan_network(heights: np.ndarray, x0: int, z0: int, sites: list, *,
                 centre: str | None = None, max_step: int = 3,
                 spine_width: int = 3, lane_width: int = 2, court_radius: int = 3,
                 shortcut_ratio: float = 2.5, max_shortcuts: int = 3,
                 avoid_extra: np.ndarray | None = None,
                 passable=None, arterial=None) -> Network:
    """Lay a walkable network between every planned structure site.

    `heights[x - x0, z - z0]` is the y of the topmost solid block. `sites` are the
    planner's footprints: {"id", "x0", "z0", "x1", "z1"}. Nothing is built here; this
    returns the surface to build, so it can be tested without a server.

    `arterial` is a network already laid -- the roads between the districts and the
    gates, routed before any district was planned. Its columns are never avoided, they
    are cheaper to walk than open ground by `ARTERIAL_BONUS`, and they come back in the
    answer at rank 0. That ordering is the whole of A4: a district whose lanes are
    routed with no knowledge of the road past it produces a place of cul-de-sacs that
    happen to touch, and a walk from one district to the next goes through somebody's
    front room or round the outside of the town."""
    from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

    h = heights.astype(np.int32)
    sx, sz = h.shape
    notes: dict = {"max_step": max_step}

    # Keep lanes off the plots themselves. A route through a footprint is a lane through
    # somebody's front room; the ground inside a plot belongs to whoever builds there.
    avoid = np.zeros((sx, sz), float)
    if avoid_extra is not None:
        avoid = avoid + avoid_extra
    for s in sites:
        i0, j0 = max(s["x0"] - x0, 0), max(s["z0"] - z0, 0)
        i1, j1 = min(s["x1"] - x0, sx - 1), min(s["z1"] - z0, sz - 1)
        if i0 <= i1 and j0 <= j1:
            avoid[i0:i1 + 1, j0:j1 + 1] = np.inf
    for (px, pz) in (passable or ()):
        i, j = px - x0, pz - z0
        if 0 <= i < sx and 0 <= j < sz:
            avoid[i, j] = 0.0
    notes["passable"] = len(passable or ())

    # **The arterials are already there**, so they are neither avoided nor re-planned:
    # they are the cheapest ground on the site and every lane the router lays is drawn
    # to them. A4.
    art_cells = (dict(arterial.cells) if isinstance(arterial, Network)
                 else dict(arterial or {}))
    if art_cells:
        # Everything *else* gets dearer, rather than the road getting cheaper: a
        # negative weight turns Dijkstra into the wrong algorithm, and scipy warned and
        # then ran out of memory before it said so plainly. Only the ordering matters,
        # and only where there is an arterial at all -- with none the field is untouched
        # and every round that ever ran routes exactly as it did.
        avoid = np.where(np.isfinite(avoid), avoid + ARTERIAL_BONUS, avoid)
        for (px, pz) in art_cells:
            i, j = px - x0, pz - z0
            if 0 <= i < sx and 0 <= j < sz:
                avoid[i, j] = 0.0
    notes["arterial_cells"] = len(art_cells)

    cand = {s["id"]: _approach_candidates(h, x0, z0, (s["x0"], s["z0"], s["x1"], s["z1"]),
                                          front=s.get("front"))
            for s in sites}
    ids = [s["id"] for s in sites if cand[s["id"]]]
    flat = [(sid, c) for sid in ids for c in cand[sid]]
    src_idx = [(c["x"] - x0) * sz + (c["z"] - z0) for _, c in flat]

    n = len(ids)
    rows = {sid: [k for k, (s, _) in enumerate(flat) if s == sid] for sid in ids}

    def _costs(dist):
        """Node-to-node cost is the cheapest pair of approach cells, so the router
        picks which side of a building the town arrives at. `dist[ka, kb]` is the
        distance between approach cells `ka` and `kb`."""
        cost = np.full((n, n), np.inf)
        pick = {}
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                best = None
                for ka in rows[ids[a]]:
                    for kb in rows[ids[b]]:
                        d = dist[ka, kb]
                        if np.isfinite(d) and (best is None or d < best[0]):
                            best = (d, ka, kb)
                if best:
                    cost[a, b] = best[0]
                    pick[(a, b)] = (best[1], best[2])
        return cost, pick

    step = max_step
    while True:
        graph = _grid_graph(h, avoid, step)
        # 1,756 sources over an 864-square is 9.8 GiB of float64 and as much again of
        # int32 -- and the router only ever reads the distances at the other approach
        # cells and the predecessors of the edges it chooses. The rows are the same rows
        # Dijkstra always gave, source by source.
        dist = _source_distances(graph, src_idx)
        cost, pick = _costs(dist)
        # one network or several is the first question the spec asks of this pass, and
        # it is answerable here, before anything is built
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in range(n):
                if v not in seen and np.isfinite(cost[u, v]):
                    seen.add(v)
                    stack.append(v)
        stranded = [ids[a] for a in range(n) if a not in seen]
        if not stranded or step >= 12:
            notes["stranded"] = stranded
            break
        # a cliff no switchback gets round: allow a deeper cut, and say so
        step += 3
        notes["raised_max_step"] = step

    finite = np.where(np.isfinite(cost), cost, 0)
    mst = minimum_spanning_tree(finite)
    edges = [(int(a), int(b)) for a, b in zip(*mst.nonzero())]

    # A tree is the smallest thing that connects everything, and a town of only dead-
    # ends is a bad town. Add a very few edges where the tree makes you walk more than
    # `shortcut_ratio` times the direct cost -- a rule, not a legibility score.
    tree = {a: [] for a in range(n)}
    for a, b in edges:
        tree[a].append(b); tree[b].append(a)
    tree_cost = _tree_costs(tree, cost, n)
    extra = []
    for a in range(n):
        for b in range(a + 1, n):
            if not np.isfinite(cost[a, b]) or (a, b) in edges or (b, a) in edges:
                continue
            # A stranded site has no tree distance to anything, and the loop above has
            # already said so in `notes["stranded"]` -- reading one out of `tree_cost`
            # raised `KeyError: 1` and lost the network that had been planned for
            # everything else.
            if b not in tree_cost[a]:
                continue
            if tree_cost[a][b] > shortcut_ratio * cost[a, b]:
                extra.append((cost[a, b], a, b))
    extra.sort()
    edges += [(a, b) for _, a, b in extra[:max_shortcuts]]

    # --- one approach cell per structure, or the network is not a network ----------
    # Choosing the cheapest pair of candidates per *edge* independently lets two routes
    # arrive at opposite sides of the same building and never meet: the first run of
    # this produced a flat site whose lanes fell into three disconnected pieces, and
    # only the lane-graph check saw it (the world flood walked round over open ground,
    # which is exactly the "reachable from anywhere" answer the spec says not to trust).
    # Fixing one approach per structure makes shared endpoints, and connectivity of the
    # tree then carries the network.
    nbrs = {a: [] for a in range(n)}
    for a, b in edges:
        nbrs[a].append(b); nbrs[b].append(a)
    choice = {a: (pick[(a, nbrs[a][0])][0] if nbrs[a] and (a, nbrs[a][0]) in pick
                  else rows[ids[a]][0]) for a in range(n)}
    for _ in range(3):
        for a in range(n):
            best = None
            for k in rows[ids[a]]:
                tot = sum(dist[k, choice[b]] for b in nbrs[a])
                if np.isfinite(tot) and (best is None or tot < best[0]):
                    best = (tot, k)
            if best:
                choice[a] = best[1]

    # --- rasterise the routes, and rank them by how much of the town they carry -----
    load = _edge_load(tree, n)
    cell_rank: dict = {}
    routes = []
    preds: dict = {}
    for (a, b) in edges:
        ka, kb = choice[a], choice[b]
        if ka not in preds:
            # the predecessor row of this one source, for the routes it carries
            _d, preds[ka] = dijkstra(graph, directed=False, indices=[src_idx[ka]],
                                     return_predecessors=True)
            preds[ka] = preds[ka][0]
        cells = _path(preds[ka], src_idx[ka], src_idx[kb])
        if not cells:
            continue
        pts = [(int(c // sz) + x0, int(c % sz) + z0) for c in cells]
        rank = 1 if load.get((min(a, b), max(a, b)), 0) >= max(2, n // 4) else 2
        routes.append({"a": ids[a], "b": ids[b], "rank": rank, "cells": len(pts)})
        for p in pts:
            cell_rank[p] = min(cell_rank.get(p, 9), rank)

    widened = _widen(cell_rank, spine_width, lane_width, avoid, x0, z0)
    # **The road is in the surface.** Every arterial column is a cell of the network at
    # rank 0 before the heights are solved, so the road and the lanes are one
    # 1-Lipschitz field on the ground as it stands. The road used to come back after the
    # solve at the level its own planning recorded, and where the ground the circulation
    # reads differs from the ground the road was planned on.
    for c in art_cells:
        i, j = c[0] - x0, c[1] - z0
        if 0 <= i < sx and 0 <= j < sz and np.isfinite(avoid[i, j]):
            widened[c] = 0

    # a court where the town's routes converge: the centre you can steer by
    hub = _hub(ids, edges, centre)
    if hub is not None and court_radius:
        hx = hz = None
        for a in range(n):
            if ids[a] == hub:
                hx, hz = flat[choice[a]][1]["x"], flat[choice[a]][1]["z"]
        if hx is not None:
            for dx in range(-court_radius, court_radius + 1):
                for dz in range(-court_radius, court_radius + 1):
                    if dx * dx + dz * dz > court_radius * court_radius:
                        continue
                    p = (hx + dx, hz + dz)
                    i, j = p[0] - x0, p[1] - z0
                    if 0 <= i < sx and 0 <= j < sz and np.isfinite(avoid[i, j]):
                        widened.setdefault(p, 0)
                        widened[p] = 0
            notes["court"] = [hx, hz, court_radius]
    notes["hub"] = hub

    # --- solve the surface ---------------------------------------------------------
    adj = {c: [(c[0] + dx, c[1] + dz) for dx, dz in DIRS
               if (c[0] + dx, c[1] + dz) in widened] for c in widened}
    terrain = {c: int(h[c[0] - x0, c[1] - z0]) for c in widened}
    y = _solve_heights(terrain, adj)
    y, rounds = _landings(y, adj)
    notes["landing_rounds"] = rounds
    notes["cut_max"] = max((terrain[c] - y[c] for c in y), default=0)
    notes["fill_max"] = max((y[c] - terrain[c] for c in y), default=0)

    cells = {}
    for c in widened:
        f = _needed_facings(c, y, adj)
        cells[c] = {"y": y[c], "rank": widened[c],
                    "face": FACING[next(iter(f))] if len(f) == 1 else None}
    # (The arterial columns are in `cells` at rank 0 with the height and the face the
    # one solve gave them; a recorded level or face on `arterial` is the plan's record
    # of the road and is not read here.)

    thresholds = []
    for a in range(n):
        chosen = [flat[choice[a]][1]] + [c for c in cand[ids[a]]
                                         if c is not flat[choice[a]][1]]
        t = _threshold_for(ids[a], chosen, cells)
        if t:
            thresholds.append(t)

    notes["sites"] = len(sites)
    notes["served"] = len(thresholds)
    return Network(cells, thresholds,
                   nodes=[{"id": ids[a]} for a in range(n)],
                   edges=[{"a": ids[a], "b": ids[b]} for a, b in edges], notes=notes)


def _tree_costs(tree: dict, cost: np.ndarray, n: int) -> dict:
    out = {}
    for a in range(n):
        d = {a: 0.0}
        stack = [a]
        while stack:
            u = stack.pop()
            for v in tree[u]:
                if v not in d:
                    d[v] = d[u] + cost[u, v]
                    stack.append(v)
        out[a] = d
    return out


def _edge_load(tree: dict, n: int) -> dict:
    """How many structures each tree edge carries: the town's own traffic, counted."""
    load = {}
    for a in range(n):
        for b in tree[a]:
            if a > b:
                continue
            # size of the component on b's side once (a,b) is cut
            seen = {a, b}
            stack = [b]
            size = 1
            while stack:
                u = stack.pop()
                for v in tree[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
                        size += 1
            load[(a, b)] = min(size, n - size)
    return load


def _widen(cell_rank: dict, spine_width: int, lane_width: int,
           avoid: np.ndarray, x0: int, z0: int) -> dict:
    """Thicken each centreline perpendicular to its own direction.

        Width is hierarchy made visible: a main way you can walk two abreast, branches you
        cannot. That is not a legibility *score* -- there is no such thing and the spec
        forbids inventing one -- it is the only thing a person walking has to steer by.
        
    """
    sx, sz = avoid.shape
    out = dict(cell_rank)
    pts = sorted(cell_rank)
    for (x, z) in pts:
        rank = cell_rank[(x, z)]
        width = spine_width if rank <= 1 else lane_width
        half = (width - 1) // 2
        extra = width - 1 - 2 * half            # even widths lean one way
        # perpendicular is whichever axis has fewer lane neighbours in line
        along_x = ((x + 1, z) in cell_rank) or ((x - 1, z) in cell_rank)
        offs = range(-half, half + 1 + extra)
        for k in offs:
            p = (x, z + k) if along_x else (x + k, z)
            i, j = p[0] - x0, p[1] - z0
            if 0 <= i < sx and 0 <= j < sz and np.isfinite(avoid[i, j]):
                out[p] = min(out.get(p, 9), rank)
    return out


def _hub(ids: list, edges: list, centre: str | None) -> str | None:
    if centre and centre in ids:
        return centre
    deg: dict = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    if not deg:
        return None
    return ids[max(deg, key=lambda k: deg[k])]


def _threshold_for(sid: str, cands: list, cells: dict):
    """The lane cell this structure is entered from, and which way its door faces."""
    for c in cands:
        p = (c["x"], c["z"])
        if p in cells:
            dx, dz = {v: k for k, v in FACING.items()}[c["face"]]
            y = cells[p]["y"]
            return Threshold(sid, p[0], p[1], y, c["face"],
                             (p[0] + dx, y + 1, p[1] + dz))
    return None


# ------------------------------------------------------------------ walk check

def designed_ground_costs(plan: dict, decls: dict, shape, x0: int, z0: int):
    """The cost field the lanes of a designed place carry beyond water and walls: the
    terrace-edge band (`placeplan.terrace_edge_band`), zero everywhere on a place that
    designs no ground. `decls` is {type: declaration} for the plan's types."""
    import numpy as np
    from . import placeplan
    lay = (plan or {}).get("layout") or {}
    if not placeplan.designed_terrace({"layout": lay}):
        return np.zeros(shape, float)
    parts = [p for p in (plan.get("parts") or []) if p.get("kind") == "point"
             and (decls.get(p.get("type")) or {}).get("passage")]
    base = placeplan.designed_heights(np.zeros(shape, int), x0, z0, lay, gates=parts,
                                      approaches=False)
    return placeplan.terrace_edge_band(shape, x0, z0, lay, gates=parts, designed=base)


def arterial_cells(art: dict, heights, x0: int, z0: int) -> dict:
    """The arterial record as `plan_network` takes it: every column of the road inside
    the heightmap at the level and with the stair face it was solved to, rank 0. The
    level falls back to the ground where a record carries none; the face to none."""
    levels = art.get("levels") or {}
    faces = art.get("faces") or {}
    out = {}
    for x, z in (art.get("cells") or []):
        x, z = int(x), int(z)
        i, j = x - x0, z - z0
        if not (0 <= i < heights.shape[0] and 0 <= j < heights.shape[1]):
            continue
        key = f"{x},{z}"
        out[(x, z)] = {"y": int(levels.get(key, heights[i, j])), "rank": 0,
                       "face": faces.get(key)}
    return out


def walk_check(net: Network) -> dict:
    """Re-derive reachability over the planned lane graph under the movement rules.

        Deliberately does not trust the construction. A rise of one block is passable only
        onto a stair facing the way you are going; anything else the planner emits is a
        defect, and this is where it shows before a single block is written.
        
    """
    y = {c: r["y"] for c, r in net.cells.items()}
    face = {c: r["face"] for c, r in net.cells.items()}
    inv = {v: k for k, v in FACING.items()}
    if not y:
        return {"cells": 0, "reached": 0, "components": 0, "unreachable": []}

    def moves(c):
        for dx, dz in DIRS:
            n = (c[0] + dx, c[1] + dz)
            if n not in y:
                continue
            rise = y[n] - y[c]
            if rise <= 0 and rise >= -3:
                yield n
            elif rise == 1 and face[n] and inv[face[n]] == (dx, dz):
                yield n

    start = min(y, key=lambda c: (net.cells[c]["rank"], c))
    seen = {start}
    stack = [start]
    while stack:
        c = stack.pop()
        for n in moves(c):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    unreached = sorted(set(y) - seen)
    return {"cells": len(y), "reached": len(seen),
            "unreachable": [list(c) for c in unreached[:20]],
            "unreachable_n": len(unreached)}


# ---------------------------------------------------------------------- building

#: Family -> the wall block that rails it. Only the families a lane is plausibly paved
#: in; anything else falls back to cobblestone, and registry validation catches a typo
#: before a block is written.
WALLS = {
    "cobblestone": "cobblestone_wall", "mossy_cobblestone": "mossy_cobblestone_wall",
    "stone_brick": "stone_brick_wall", "mossy_stone_brick": "mossy_stone_brick_wall",
    "deepslate_brick": "deepslate_brick_wall", "deepslate_tile": "deepslate_tile_wall",
    "brick": "brick_wall", "sandstone": "sandstone_wall",
    "red_sandstone": "red_sandstone_wall", "nether_brick": "nether_brick_wall",
    "blackstone": "blackstone_wall", "andesite": "andesite_wall",
    "granite": "granite_wall", "diorite": "diorite_wall",
    "mud_brick": "mud_brick_wall", "tuff_brick": "tuff_brick_wall",
    "end_stone_brick": "end_stone_brick_wall",
    "polished_blackstone_brick": "polished_blackstone_brick_wall",
    "prismarine": "prismarine_wall",
}


def emit(builder, net: Network, mat: str = "cobblestone", *,
         kerb: str | None = None, rail: str | None = None,
         lantern_every: int = 14, post_every: int = 12, max_fill: int = 24,
         max_cut: int = 24, ground=None, x0: int = 0, z0: int = 0,
         clear_veg: bool = True, keep_off=None) -> dict:
    """Write the network into the world through the model's own build API.

        Uses `builder` so the circulation pass is audited, preflighted and flushed exactly
        like a model pass -- including the second flush of connective blocks with updates on,
        without which every rail would stand as an isolated post.

        `ground` is the vegetation-free surface the planner routed on (observe.ground_heights).
        Filling up from `get_height` instead would measure from the top of whatever tree is
        standing there. The cut still uses `get_height`, because the tree is exactly what has
        to come out.
        
    """
    full, stairs, slab = material(mat)
    kerb = kerb or full
    rail = rail or WALLS.get(mat, "cobblestone_wall")
    stats = {"cells": len(net.cells), "steps": 0, "fill": 0, "cut": 0,
             "rails": 0, "lanterns": 0, "trees": 0}

    def grade(x, z):
        if ground is None:
            return builder.get_height(x, z)
        i, j = x - x0, z - z0
        if 0 <= i < ground.shape[0] and 0 <= j < ground.shape[1]:
            return int(ground[i, j])
        return builder.get_height(x, z)

    if clear_veg:
        # Whole trees, trunk and canopy, but only the ones actually standing in the
        # lane. Clearing the lane's bounding box instead would strip the site. In jungle
        # the canopies touch, so one trunk in the lane can pull down a grove:
        # `tree_columns` is how wide the swathe actually came out, which is the number
        # to look at before believing this is surgical.
        for (x, z) in sorted(net.cells):
            stats["trees"] += builder.clear_trees(x, z, x, z, margin=0)
        cleared = sorted({(p[0], p[2]) for p in builder._pending})
        stats["tree_columns"] = len(cleared)
        # Put the floor back. Taking the canopy and leaving bare dirt with leaf litter
        # on it is what a human called weird the first time they walked one of these.
        xs = [c[0] for c in cleared] or [0]
        zs = [c[1] for c in cleared] or [0]
        stats["dressed"] = builder.dress_ground(min(xs), min(zs), max(xs), max(zs),
                                                columns=cleared)

    for (x, z), rec in sorted(net.cells.items()):
        y, face = rec["y"], rec["face"]
        builder.place_block(x, y, z, f"{stairs}[facing={face},half=bottom]" if face
                            else full)
        if face:
            stats["steps"] += 1
        g = grade(x, z)
        if g < y:                                   # fill: nothing floats over a slope
            for yy in range(max(g + 1, y - max_fill), y):
                builder.place_block(x, yy, z, full)
            stats["fill"] += 1
        g = builder.get_height(x, z)
        # Headroom, and the cut that makes it. Clearing a fixed three blocks is not
        # enough where the lane is cut into a rise: what is left is a roof, the lane
        # stops being outdoors, and Nav.circulation() -- which is *outdoor* stances --
        # drops it out of the network entirely. A cutting is open to the sky.
        top = min(max(y + 3, g), y + max_cut)
        for yy in range(y + 1, top + 1):
            if builder.get_block(x, yy, z) != "air":
                stats["cut"] += 1
            builder.place_block(x, yy, z, "air")

    # The threshold itself: the doorstep, not just the lane it comes off. The spec gives
    # this pass the few blocks where a lane meets a doorway, and laying only the lane
    # cell is not that. a threshold that fails the test it exists to guarantee. The door
    # cell is levelled to the lane and cleared to head height, and the building pass
    # puts its floor at that level.
    for t in net.thresholds:
        dx, dz = {v: k for k, v in FACING.items()}[t.facing]
        px, pz = t.x + dx, t.z + dz
        builder.place_block(px, t.y, pz, full)
        g = builder.get_height(px, pz)
        for yy in range(max(g + 1, t.y - max_fill), t.y):
            builder.place_block(px, yy, pz, full)
        for yy in range(t.y + 1, min(max(t.y + 3, g), t.y + max_cut) + 1):
            builder.place_block(px, yy, pz, "air")
    stats["thresholds"] = len(net.thresholds)

    # What this pass has already reserved and may not build on: the doorstep of every
    # structure, the lane cell it comes off, and any way in an earlier pass wrote down.
    keep = set(keep_off or ())
    for t in net.thresholds:
        dx, dz = {v: k for k, v in FACING.items()}[t.facing]
        keep |= {(t.x, t.z), (t.x + dx, t.z + dz), (t.door[0], t.door[2])}

    # A rail is not decoration when it edges a drop. "Fences placed with no reason" is
    # its own row of the taxonomy; this is the version with a reason. **Never on a
    # reserved doorstep.** The rail fires on the *original* grade, and the threshold
    # pass above has already filled that column up to the lane -- so a doorstep on the
    # edge of a drop is solid ground with a fence standing in it, and E008 says the door
    # reserved there cannot be walked into off its own threshold. A later pass building
    # over what an earlier one reserved, because nothing told it.
    edge_cells = []
    for (x, z), rec in sorted(net.cells.items()):
        y = rec["y"]
        for dx, dz in DIRS:
            p = (x + dx, z + dz)
            if p in net.cells or p in keep:
                continue
            g = grade(p[0], p[1])
            if g <= y - 3:
                builder.place_block(p[0], y, p[1], kerb)
                builder.place_block(p[0], y + 1, p[1], rail)
                stats["rails"] += 1
                edge_cells.append((p[0], y, p[1]))
    for i, (x, y, z) in enumerate(edge_cells):
        if lantern_every and i % lantern_every == 0:
            builder.place_block(x, y + 2, z, "lantern")
            stats["lanterns"] += 1

    # Lamp posts along the lane itself. Rooms are lit, lanes are not. Every lantern in
    # the town was either inside a building or on a rail beside a drop, so a town on
    # level ground had none at all. This is the standing version: a post beside the lane
    # every `post_every` cells, on a column that is **not** lane -- so the lane cannot
    # lose a stance to it, which is the one thing a lamp post must not do -- and never
    # on a reserved threshold or a way in that has already been recorded.
    lit: list = []
    for i, (x, z) in enumerate(sorted(net.cells)):
        if not post_every or i % post_every:
            continue
        y = net.cells[(x, z)]["y"]
        if any(max(abs(x - lx), abs(z - lz)) < 6 for (lx, lz) in lit):
            continue                    # do not clump where the lane doubles back
        for dx, dz in DIRS:
            px, pz = x + dx, z + dz
            if (px, pz) in net.cells or (px, pz) in keep:
                continue
            g = grade(px, pz)
            if abs(g - y) > 2:          # not the shoulder of the lane: a bank or a void
                continue
            for yy in range(min(g, y), y):
                builder.place_block(px, yy, pz, full)
            builder.place_block(px, y, pz, kerb)
            builder.place_block(px, y + 1, pz, rail)
            builder.place_block(px, y + 2, pz, rail)
            builder.place_block(px, y + 3, pz, "lantern[hanging=false]")
            stats["lanterns"] += 1
            stats["posts"] = stats.get("posts", 0) + 1
            lit.append((x, z))
            break
    return stats
