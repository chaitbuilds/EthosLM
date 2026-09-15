"""The seam between what was built and what was there.

**It is the same solver as the lane, pointed sideways.** `circulate` makes a lane
walkable by solving a 1-Lipschitz height field over the lane cells; this makes an edge
finished by solving the same field over a band of ground beside it, anchored at the
built height on one side and at untouched terrain on the other. Ground that already sits
inside that envelope is not touched at all -- only what sticks up too close to the edge,
or drops away too fast from it, moves.

Two bounds keep it from becoming a regrade:

    slope     how fast the band may climb away from the edge (1 block per cell)
    max_move  how far any single column may be cut or filled (3 blocks)

So a five-block cut face does not become a five-cell ramp: its lip is rounded and its
foot is filled, and the face stays a face. That is the difference between finishing an
edge and flattening a hill, and the second one is what would sand off the character.

It touches nothing inside a plot -- that ground belongs to whoever built there -- and
nothing on a lane cell, which belongs to circulation."""
from __future__ import annotations

import heapq

import numpy as np


def _lipschitz(anchors: dict, adj: dict, sign: int, slope: int) -> dict:
    """Extreme `slope`-Lipschitz field on the band, bounded by the anchors."""
    best = {c: sign * v for c, v in anchors.items()}
    heap = [(sign * v, c) for c, v in anchors.items()]
    heapq.heapify(heap)
    while heap:
        d, c = heapq.heappop(heap)
        if d > best.get(c, 1 << 30):
            continue
        for n in adj.get(c, ()):
            nd = d + slope
            if nd < best.get(n, 1 << 30):
                best[n] = nd
                heapq.heappush(heap, (nd, n))
    return {c: sign * v for c, v in best.items()}


def plan_finish(built: dict, ground: np.ndarray, x0: int, z0: int, *,
                keep_out: list | None = None, width: int = 5, slope: int = 1,
                max_move: int = 3) -> dict:
    """Where the ground beside a built edge should sit.

        `built` maps (x, z) -> the finished height of ground we laid (lane cells, and the
        grade a plot was cut to). `ground` is the vegetation-free surface. Returns only the
        columns whose height should change, as (x, z) -> new height.
        
    """
    sx, sz = ground.shape

    def nat(c):
        return int(ground[c[0] - x0, c[1] - z0])

    blocked = set()
    for r in (keep_out or []):
        for x in range(r["x0"], r["x1"] + 1):
            for z in range(r["z0"], r["z1"] + 1):
                blocked.add((x, z))

    # the band: everything within `width` of a built column that is not itself built or
    # spoken for, found by a plain expanding front
    band: dict = {}
    front = list(built)
    seen = set(built)
    for step in range(1, width + 1):
        nxt = []
        for (x, z) in front:
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (x + dx, z + dz)
                if c in seen or not (0 <= c[0] - x0 < sx and 0 <= c[1] - z0 < sz):
                    continue
                seen.add(c)
                if c in blocked:
                    continue
                band[c] = step
                nxt.append(c)
        front = nxt
    if not band:
        return {}

    cells = set(band) | set(built)
    adj = {c: [(c[0] + dx, c[1] + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
               if (c[0] + dx, c[1] + dz) in cells] for c in cells}

    # anchored at the built edge on one side and at the untouched ground on the other:
    # the outer rim of the band is natural terrain and must stay where it is
    anchors = dict(built)
    for c, d in band.items():
        if d == width:
            anchors[c] = nat(c)
    up = _lipschitz(anchors, adj, +1, slope)       # nothing may stand higher than this
    down = _lipschitz(anchors, adj, -1, slope)     # nor lower than this

    out = {}
    for c, d in band.items():
        n = nat(c)
        target = min(max(n, down[c]), up[c])
        # How much a column may move **tapers to nothing at the far edge of the band**.
        # A flat `max_move` shaves the whole band by the same amount and what you get is
        # a bench cut along the hillside, not a rounded lip: on the test face every one
        # of five columns came back cut by exactly three. The taper is what makes this
        # finish an edge instead of regrading behind it.
        allow = max(0, round(max_move * (width - d + 1) / width))
        target = min(max(target, n - allow), n + allow)
        if target != n:
            out[c] = int(target)
    return out


def apply(builder, targets: dict, dress: bool = True) -> dict:
    """Cut and fill the columns `plan_finish` picked out, then put the skin back.

        Fill uses the column's own subsoil and caps with its own surface block, so a seam in
        a jungle finishes in dirt and grass and a seam in a mesa finishes in terracotta,
        without being told which site it is on.
        
    """
    stats = {"columns": len(targets), "cut": 0, "filled": 0, "moved": 0}
    for (x, z), y in sorted(targets.items()):
        g = builder.get_height(x, z)
        if y == g:
            continue
        skin = builder.get_block(x, g, z)
        sub = builder.get_block(x, g - 1, z)
        if sub in ("air", "water", "cave_air"):
            sub = skin
        if y < g:                                   # cut the lip back
            for yy in range(y + 1, g + 1):
                builder.place_block(x, yy, z, "air")
            builder.place_block(x, y, z, skin)
            stats["cut"] += 1
        else:                                       # fill the foot in
            for yy in range(g, y):
                builder.place_block(x, yy, z, sub)
            builder.place_block(x, y, z, skin)
            for yy in range(y + 1, y + 3):
                if builder.get_block(x, yy, z) != "air":
                    builder.place_block(x, yy, z, "air")
            stats["filled"] += 1
        stats["moved"] += abs(y - g)
    if dress and targets:
        xs = [c[0] for c in targets]
        zs = [c[1] for c in targets]
        stats["dressed"] = builder.dress_ground(min(xs), min(zs), max(xs), max(zs),
                                                columns=sorted(targets))
    return stats
