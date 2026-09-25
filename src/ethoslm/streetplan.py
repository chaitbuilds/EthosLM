"""Compose a district from its streets. The fabric reset round.

The block grid (`district_compile._grid_axis`) cuts a rectangle into blocks and then asks
each block what to hold, so the streets a quarter is lived on are whatever the grid
leaves: a one-row grid "has no principal street", the market's other edges are open
ground, and lanes exist only between blocks the grid happened to lay. This module
reverses the order. It decides, in world columns and on the ground the design prepares:

  1. the **principal streets** -- the arterial runs the district's own ground fronts,
     found from the routed road rather than assumed;
  2. the **anchor** (a required market) at the principal street nearest the point the
     quarter is entered from, and the **fronts that face it**;
  3. the **lanes**: parallel lanes along one axis, two rows of lots back to back between
     each pair, a cross lane where a lane would otherwise not reach a street, and the
     lanes pierce the principal street's frontage to meet the road;
  4. the **frontage runs** along every street face, and only then the **lots**, cut
     along each run at widths inside the type's band so that neighbours touch;
  5. the **open ground** the streets and lots leave: gardens and yards, named.

Then it **judges** the result on required relationships before preferences
(`judge`): a proposal that loses the market's edges, leaves a principal street without
fronts, or strands a lane is inadmissible whatever it would score. `compose` tries a
small family of materially different proposals -- lane axis, lot depth, how far shops
run along the principal street -- and returns every one with its verdict; the caller
adopts the best admissible one or reports that none is.

Nothing here names a place or a type: the caller says which lot is a dwelling and which
a shop, with the widths and depths their envelopes admit, and whether a lot's site can
be built is asked back through `site_ok` (the compiler's own `site_solve`).
"""
from __future__ import annotations

import itertools
import random

import numpy as np

#: A lane, the same width the compiler gives a street between blocks
#: (`placeplan.PLOT_LANE`).
LANE = 5
#: How far a run of lots may go before a cross lane breaks it: a lane with one way out
#: that is longer than this is a lane nobody can find their way round.
RUN_MAX = 72
#: How far a composed lot's building stands off its street and its back, in columns
#: (`buildlib.pad_insets`' free-side inset, which is 2 for a lot laid by the grid): a
#: lane is a lane between two walls, not between two aprons.
FRONT_INSET = 1
#: The walk left between a market floor and the fronts that face it.
MARKET_WALK = 2
#: How much of a principal street's run the district's side must front with doors or the
#: market for the street to be a street and not a road past a field.
PRINCIPAL_FRONTED = 0.6
#: ...and how much of it must be *active*: fronts facing the street, or the market.
PRINCIPAL_ACTIVE = 0.4
#: How much of an anchor's edge the fronts facing it must cover for the edge to count.
EDGE_FRONTED = 0.5
#: How much of a lane's length must be fronted on at least one side.
LANE_FRONTED = 0.5
#: The least a principal street's run is, in columns, before it is a street at all.
PRINCIPAL_LEAST = 16
#: How far a road may be from the first buildable column and still be the street it
#: fronts: the compiler's road verge (`district_compile.ROAD_VERGE`, 3) plus its edge
#: margin (`EDGE_MARGIN`, 2) and one.
ROAD_REACH = 6
#: How far a principal street may lie beyond the district's first buildable column when
#: every column between is outside the district (the sector gap a ring strip leaves
#: between its pieces, where the routed road may run on either side of it): the design
#: resolution round, where the gate street ran nine columns off the hill-toe piece and
#: the piece was composed as if the street were not there.
PRINCIPAL_REACH = 10
#: A principal street's run may be broken this many columns (a drain, a tree the mask
#: refuses) and still be one run of frontage.
RUN_GAP = 3
#: ...and it has to run along this share of the ground it fronts to organise it.
PRINCIPAL_SHARE = 0.4
#: The least dwellings a composed quarter holds before it is a quarter.
DWELLINGS_LEAST = 3

SIDES = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
FRONTS = ("north", "south", "west", "east")
#: The clearance two lots keep unless they are party walls on one street face: the
#: types' own `keeps clear` (2), which the plan validator holds every pair to
#: (`stages_plan` `overlap`), and the only touch it admits is flank to flank on one
#: frontage (`party_wall`).
CLEAR = 2
OPP = {"north": "south", "south": "north", "west": "east", "east": "west"}
#: Under a transpose of x and z, a world side becomes this one.
_T = {"north": "west", "west": "north", "south": "east", "east": "south"}


def _rect_cells(r):
    x0, z0, x1, z1 = r
    return (x1 - x0 + 1) * (z1 - z0 + 1)


class _Plan:
    """One proposal's geometry in a frame whose lanes run along x.

    `ok` is indexed `[x - X0, z - Z0]` and is the ground a building may stand on;
    `used` is what the proposal has already spent."""

    def __init__(self, X0, Z0, ok, road):
        self.X0, self.Z0 = X0, Z0
        self.ok = ok
        self.W, self.H = ok.shape
        self.road = road
        self.used = np.zeros_like(ok, dtype=bool)
        self.street = np.zeros_like(ok, dtype=bool)
        #: which way the lot on each column fronts (index into `FRONTS`), -1 for none:
        #: what the clearance between two lots is judged by
        self.front = np.full(ok.shape, -1, dtype=np.int8)
        self.streets: list = []
        self.lots: list = []
        self.open: list = []
        self.anchor = None
        self.notes: list = []

    # -- cells ------------------------------------------------------------------------
    def inside(self, x, z):
        return 0 <= x - self.X0 < self.W and 0 <= z - self.Z0 < self.H

    def free(self, x0, z0, x1, z1) -> bool:
        if x1 < x0 or z1 < z0:
            return False
        a0, a1, b0, b1 = x0 - self.X0, x1 - self.X0, z0 - self.Z0, z1 - self.Z0
        if a0 < 0 or b0 < 0 or a1 >= self.W or b1 >= self.H:
            return False
        return bool(self.ok[a0:a1 + 1, b0:b1 + 1].all()
                    and not self.used[a0:a1 + 1, b0:b1 + 1].any()
                    and not self.street[a0:a1 + 1, b0:b1 + 1].any())

    def take(self, r, street=False):
        x0, z0, x1, z1 = r
        a0, a1 = max(0, x0 - self.X0), min(self.W - 1, x1 - self.X0)
        b0, b1 = max(0, z0 - self.Z0), min(self.H - 1, z1 - self.Z0)
        if a1 < a0 or b1 < b0:
            return
        if street:
            self.street[a0:a1 + 1, b0:b1 + 1] = True
        else:
            self.used[a0:a1 + 1, b0:b1 + 1] = True

    def road_toward(self, x, z, side, reach=None) -> bool:
        """Is there road within `reach` of (x, z) toward `side`, over no buildable
        ground? The arterial is not part of `ok`, so the columns between are its verge."""
        dx, dz = SIDES[side]
        for k in range(1, (reach or ROAD_REACH) + 1):
            c = (x + dx * k, z + dz * k)
            if c in self.road:
                return True
            if self.inside(*c) and self.ok[c[0] - self.X0, c[1] - self.Z0]:
                return False
        return False


def principal_lines(P: _Plan) -> list:
    """The arterial runs the district's buildable ground fronts: for each side a road
    lies on, the line of ground columns that face it and the longest straight run of
    them. `[{"side", "line", "a0", "a1"}]`, where `line` is the constant coordinate (z
    for a north/south street, x for west/east) and `a0..a1` the run along the street."""
    out = []
    for side in SIDES:
        counts: dict = {}
        cells: dict = {}
        for a in range(P.W):
            for b in range(P.H):
                if not P.ok[a, b]:
                    continue
                x, z = a + P.X0, b + P.Z0
                if P.road_toward(x, z, side, reach=PRINCIPAL_REACH):
                    key = z if side in ("north", "south") else x
                    counts[key] = counts.get(key, 0) + 1
                    cells.setdefault(key, []).append(x if side in ("north", "south")
                                                     else z)
        if not counts:
            continue
        line = max(counts, key=lambda k: (counts[k], -abs(k)))
        along = sorted(cells[line])
        # the longest run on that line, bridging gaps of up to RUN_GAP columns
        best, cur = (along[0], along[0]), [along[0], along[0]]
        for v in along[1:]:
            if v <= cur[1] + 1 + RUN_GAP:
                cur[1] = v
            else:
                cur = [v, v]
            if cur[1] - cur[0] > best[1] - best[0]:
                best = (cur[0], cur[1])
        ground = np.where(P.ok.any(axis=1 if side in ("north", "south") else 0))[0]
        extent = (int(ground[-1]) - int(ground[0]) + 1) if len(ground) else 0
        # **a street the district fronts, not one that crosses it**: a principal street
        # bounds the ground (its line is at the ground's own edge on that side); a road
        # routed through the middle of a district is a road its lots keep clear of, and
        # its lanes may meet, but it does not organise the quarter
        across = np.where(P.ok.any(axis=0 if side in ("north", "south") else 1))[0]
        if len(across):
            lo_e = int(across[0]) + (P.Z0 if side in ("north", "south") else P.X0)
            hi_e = int(across[-1]) + (P.Z0 if side in ("north", "south") else P.X0)
            if side in ("north", "west") and line > lo_e + RUN_GAP:
                continue
            if side in ("south", "east") and line < hi_e - RUN_GAP:
                continue
        if best[1] - best[0] + 1 >= max(PRINCIPAL_LEAST, PRINCIPAL_SHARE * extent):
            out.append({"side": side, "line": int(line), "a0": int(best[0]),
                        "a1": int(best[1])})
    return out


def _widths(length: int, band: tuple, pref: int, rng: random.Random,
            spread: int = 1) -> list:
    """Widths inside `band` (lo, hi) that tile `length` exactly, near `pref` and varied
    by up to `spread`; [] where no tiling exists."""
    lo, hi = int(band[0]), int(band[1])
    if length < lo:
        return []
    n = max(1, int(round(length / float(pref))))
    for k in sorted({n, n - 1, n + 1} - {0}, key=lambda v: abs(v - n)):
        if not (k * lo <= length <= k * hi):
            continue
        ws = [length // k] * k
        for i in range(length - sum(ws)):
            ws[i] += 1
        # vary neighbours inside the band, keeping the sum
        for i in range(k - 1):
            d = rng.randint(-spread, spread)
            if lo <= ws[i] + d <= hi and lo <= ws[i + 1] - d <= hi:
                ws[i] += d
                ws[i + 1] -= d
        return ws
    return []


def cut_run(P: _Plan, run: dict, spec: dict, rng: random.Random, site_ok=None) -> list:
    """Lots along one frontage run, fronting `run["front"]`.

    `run` = {"axis": "x"|"z" (the direction the run goes along), "a0", "a1",
    "b0", "b1" (the depth across), "front", "use", "street"}. The run is split where the
    ground does not carry a lot's whole depth, each stretch is tiled with widths inside
    the use's band, and a lot whose site cannot be built (`site_ok`) is left out and the
    stretch goes on after it. Lots that touch are each other's party walls."""
    axis = run["axis"]
    a0, a1, b0, b1 = run["a0"], run["a1"], run["b0"], run["b1"]

    def rect(u0, u1):
        return ((u0, b0, u1, b1) if axis == "x" else (b0, u0, b1, u1))
    fcode = FRONTS.index(run["front"])

    def clear_of_others(u) -> bool:
        """No lot within CLEAR of this column of the run, except a lot on the same
        street face touching it flank to flank -- the one touch that is a party wall."""
        x0_, z0_, x1_, z1_ = rect(u, u)
        a0_ = max(0, x0_ - CLEAR - P.X0)
        a1_ = min(P.W - 1, x1_ + CLEAR - P.X0)
        c0_ = max(0, z0_ - CLEAR - P.Z0)
        c1_ = min(P.H - 1, z1_ + CLEAR - P.Z0)
        if a1_ < a0_ or c1_ < c0_:
            return True
        box = P.front[a0_:a1_ + 1, c0_:c1_ + 1]
        if not (box >= 0).any():
            return True
        for (ia, ic) in zip(*np.nonzero(box >= 0)):
            x, z = ia + a0_ + P.X0, ic + c0_ + P.Z0
            same = P.front[x - P.X0, z - P.Z0] == fcode
            if axis == "x":
                flank = same and z0_ <= z <= z1_ and x in (x0_ - 1, x1_ + 1)
            else:
                flank = same and x0_ <= x <= x1_ and z in (z0_ - 1, z1_ + 1)
            if not flank:
                return False
        return True
    # the stretches whose full depth is free
    stretches, cur = [], None
    for u in range(a0, a1 + 1):
        if P.free(*rect(u, u)) and clear_of_others(u):
            cur = [u, u] if cur is None else [cur[0], u]
        else:
            if cur:
                stretches.append(cur)
            cur = None
    if cur:
        stretches.append(cur)
    band, pref = spec["widths"], spec["pref"]
    # **A row's free ends carry their inset** (the design resolution round): the band is
    # what a lot between two party walls needs, and a lot at the end of a stretch keeps
    # the pad's inset on its free flank, so it is `end_extra` wider to hold the same
    # house (`ethoslm.formplan`, asked in both configurations by the compiler).
    e = int(spec.get("end_extra") or 0)

    def tile(n_cols):
        if e <= 0:
            return _widths(n_cols, band, pref, rng)
        if n_cols - 2 * e < band[0]:
            # one lot alone, both flanks free
            return ([n_cols] if band[0] + 2 * e <= n_cols <= band[1] + 2 * e else [])
        ws = _widths(n_cols - 2 * e, band, pref, rng)
        if not ws:
            return []
        if len(ws) == 1:
            return [ws[0] + 2 * e]
        ws[0] += e
        ws[-1] += e
        return ws
    got = []
    for s0, s1 in stretches:
        u = s0
        # the longest part of the stretch that tiles exactly; the rest is the run's
        # leftover, at the end away from where the run starts
        widths = []
        for n_cols in range(s1 - s0 + 1, int(band[0]) - 1, -1):
            widths = tile(n_cols)
            if widths:
                break
        if not widths:
            continue
        queue = list(widths)
        while queue and u <= s1:
            w = queue.pop(0)
            r = rect(u, u + w - 1)
            if site_ok is not None and not site_ok(r, run["front"], spec):
                # skip this width and re-tile the rest of the stretch after it
                u += w
                rest = s1 - u + 1
                queue = tile(rest) if rest >= band[0] else []
                continue
            lot = {"rect": list(r), "front": run["front"], "use": run["use"],
                   "street": run.get("street"), "run": run.get("id"), "attached": []}
            got.append(lot)
            P.take(r)
            P.front[r[0] - P.X0:r[2] - P.X0 + 1, r[1] - P.Z0:r[3] - P.Z0 + 1] = fcode
            u += w
    P.lots += got
    return got


def party_walls(lots: list) -> None:
    """Every pair of lots on one street face standing flank to flank shares a party
    wall -- inside one run or across two (a shop row meeting a row of houses)."""
    for lot in lots:
        lot["attached"] = []
    for i, a in enumerate(lots):
        for b in lots[i + 1:]:
            if a["front"] != b["front"]:
                continue
            ra, rb = a["rect"], b["rect"]
            if a["front"] in ("north", "south"):
                if not (ra[1] <= rb[3] and rb[1] <= ra[3]):
                    continue
                if ra[2] + 1 == rb[0]:
                    a["attached"].append("east")
                    b["attached"].append("west")
                elif rb[2] + 1 == ra[0]:
                    a["attached"].append("west")
                    b["attached"].append("east")
            else:
                if not (ra[0] <= rb[2] and rb[0] <= ra[2]):
                    continue
                if ra[3] + 1 == rb[1]:
                    a["attached"].append("south")
                    b["attached"].append("north")
                elif rb[3] + 1 == ra[1]:
                    a["attached"].append("north")
                    b["attached"].append("south")
    for lot in lots:
        lot["attached"] = sorted(set(lot["attached"]))


def _span(P: _Plan, b0: int, b1: int, a_start: int, step: int = 1) -> int:
    """How far the ground runs along x from `a_start` inside the band z `b0..b1`,
    bridging gaps of up to RUN_GAP columns: the last x with buildable ground."""
    last = a_start
    gap = 0
    a = a_start
    lo_b, hi_b = max(0, b0 - P.Z0), min(P.H - 1, b1 - P.Z0)
    while P.inside(a, P.Z0):
        col = P.ok[a - P.X0, lo_b:hi_b + 1] if hi_b >= lo_b else np.zeros(0, bool)
        if col.any():
            last, gap = a, 0
        else:
            gap += 1
            if gap > RUN_GAP:
                break
        a += step
    return last


def _propose(X0, Z0, ok, road, access, house, shop, anchor, fam, rng, site_ok):
    """One proposal, lanes along x (the caller transposes for lanes along z).

    The order is the design's: principal streets found; lanes decided across the ground;
    the anchor at the principal street nearest the way in, with the fronts that face it;
    the flank streets' fronts (shops near the way in); then the rows along each street
    face, back to back between lanes, cut into lots last."""
    P = _Plan(X0, Z0, ok, road)
    P.rows, P.lanes, P.lines = [], [], []
    D = int(fam["depth"])
    lines = principal_lines(P)
    P.lines = lines
    by_side = {ln["side"]: ln for ln in lines}
    xs = np.where(ok.any(axis=1))[0]
    zs = np.where(ok.any(axis=0))[0]
    if not len(xs) or not len(zs):
        return P
    gx0, gx1 = int(xs[0]) + X0, int(xs[-1]) + X0
    gz0, gz1 = int(zs[0]) + Z0, int(zs[-1]) + Z0
    west, east = by_side.get("west"), by_side.get("east")
    north, south = by_side.get("north"), by_side.get("south")
    Ds = int((shop or {}).get("depth") or 0)
    ax, az = access if access else (gx0, gz0)
    reach = int(fam.get("shop_reach") or 0)
    # the frontage bands along the flank streets: shops within `reach` of the way in
    # along the street, houses beyond (a house lot faces the street it stands on)
    flank_depth = {}
    for ln in (west, east):
        if ln is None:
            continue
        # how far the street's run is from the way in, along the street and across it
        if ln["side"] in ("west", "east"):
            d_acc = abs(ln["line"] - ax) + max(0, ln["a0"] - az, az - ln["a1"])
        else:
            d_acc = abs(ln["line"] - az) + max(0, ln["a0"] - ax, ax - ln["a1"])
        near = d_acc <= reach + D
        flank_depth[ln["side"]] = Ds if (shop and near) else D
    def x_limits(r0, r1):
        """Where a row along x runs: from the ground's edge, or from behind a flank
        street's fronts where that street runs past this row."""
        lo, hi = gx0, gx1
        if west and west["a0"] <= r1 and r0 <= west["a1"]:
            lo = west["line"] + flank_depth["west"]
        if east and east["a0"] <= r1 and r0 <= east["a1"]:
            hi = east["line"] - flank_depth["east"]
        return lo, hi
    x_lo = (west["line"] + flank_depth["west"]) if west else gx0
    # -- the lanes: a sweep across z from the principal street (or the ground's edge)
    z = north["line"] if north else gz0
    zend = south["line"] if south else gz1
    rows, lanes = [], []
    facing_lane = False
    if north:
        # the principal street's row is as deep as what fronts it: a run given wholly to
        # shops is a shop's depth, and the ground behind goes to the rows that need it
        d0 = D
        if shop and reach and reach >= (gx1 - gx0 + 1) - D:
            d0 = Ds
        rows.append((z, z + d0 - 1, "north"))
        z += d0
        facing_lane = True
    Dlo = int(house.get("depth_lo") or D)
    while True:
        left = zend - z + 1
        if facing_lane:
            # a row backing onto the one before it, facing the next lane (backs keep
            # their clearance: two rows back to back are CLEAR apart). **The pair takes
            # the depth there is**: where a whole pair at this lot depth does not fit,
            # both rows of it are made shallower -- down to the least the house stands
            # on -- before the lane is given up; and where only one row fits, it faces a
            # lane laid along the ground's own far edge.
            need_pair = CLEAR + LANE + (0 if south else 1) * 1
            D2 = D if left >= need_pair + D + (0 if south else D) else \
                (left - CLEAR - LANE) // (1 if south else 2)
            if D2 >= Dlo:
                D2 = min(D, D2)
                z += CLEAR
                rows.append((z, z + D2 - 1, "south"))
                lanes.append((z + D2, z + D2 + LANE - 1))
                z += D2 + LANE
                facing_lane = False
                continue
            if south and left >= CLEAR + Dlo:
                d3 = min(D, left - CLEAR)
                rows.append((zend - d3 + 1, zend, "south"))
            elif not south and left >= CLEAR + Dlo + LANE:
                d3 = min(D, left - CLEAR - LANE)
                z += CLEAR
                rows.append((z, z + d3 - 1, "south"))
                lanes.append((z + d3, z + d3 + LANE - 1))
            break
        if left >= Dlo and (lanes or north):
            d4 = min(D, left)
            rows.append((z, z + d4 - 1, "north"))
            z += d4
            facing_lane = True
            continue
        if not lanes and not north and left >= LANE + D:
            lanes.append((z, z + LANE - 1))
            z += LANE
            continue
        break
    P.rows, P.lanes = rows, lanes
    lane_ends = []
    for (l0, l1) in lanes:
        # a lane runs from the street it meets as far as its rows have ground
        xa = (west["line"] - ROAD_REACH) if west else gx0
        xb = _span(P, l0 - D, l1 + D, max(gx0, xa if west else gx0))
        if east and xb >= east["line"] - 1:
            xb = east["line"] + ROAD_REACH
        r = (max(X0, xa), l0, min(X0 + P.W - 1, xb), l1)
        P.take(r, street=True)
        P.streets.append({"kind": "lane", "rect": list(r), "axis": "x"})
        lane_ends.append((r, bool(west), bool(east and xb >= east["line"])))
    # cross lanes: where a lane meets no street at an end, and every RUN_MAX along it
    zc0 = (north["line"] - ROAD_REACH) if north else gz0
    cross = []
    for (r, w_ok, e_ok) in lane_ends:
        if not w_ok:
            cross.append((r[0], r[3]))
        if not e_ok and (r[2] - r[0] + 1) > RUN_MAX // 2:
            cross.append((r[2] - LANE + 1, r[3]))
        span = r[2] - x_lo + 1
        n_mid = max(0, int(span // RUN_MAX))
        for k in range(1, n_mid + 1):
            cross.append((int(x_lo + k * span / (n_mid + 1)) - LANE // 2, r[3]))
    done = set()
    for (cx, zc1) in sorted(cross):
        if any(abs(cx - d) < LANE + D for d in done):
            continue
        done.add(cx)
        r = (cx, max(Z0, zc0), cx + LANE - 1, min(Z0 + P.H - 1, zc1))
        P.take(r, street=True)
        P.streets.append({"kind": "cross", "rect": list(r), "axis": "z"})
    for ln in lines:
        P.streets.append({"kind": "principal", "side": ln["side"], "line": ln["line"],
                          "a0": ln["a0"], "a1": ln["a1"]})

    # -- the anchor: at the principal street nearest the way in, and its fronts
    if anchor:
        _place_anchor(P, anchor, by_side, (ax, az), house, shop, fam, rng, site_ok,
                      (gx0, gz0, gx1, gz1))

    # -- the flank streets' own fronts, pierced by the lanes that meet them
    for ln in (west, east):
        if ln is None:
            continue
        dep = flank_depth[ln["side"]]
        use = "shop" if (shop and dep == Ds) else "house"
        spec = shop if use == "shop" else house
        a, pieces = ln["a0"], []
        for (l0, l1) in sorted(lanes):
            if l0 - 1 >= a:
                pieces.append((a, l0 - 1))
            a = max(a, l1 + 1)
        if a <= ln["a1"]:
            pieces.append((a, ln["a1"]))
        for k, (p0, p1) in enumerate(pieces):
            for dep2 in _depths({**spec, "depth": dep}):
                b0, b1 = ((ln["line"], ln["line"] + dep2 - 1) if ln["side"] == "west"
                          else (ln["line"] - dep2 + 1, ln["line"]))
                cut_run(P, {"axis": "z", "a0": p0, "a1": p1, "b0": b0, "b1": b1,
                            "front": ln["side"], "use": use,
                            "street": f"principal_{ln['side']}",
                            "id": f"{ln['side']}_{k}"}, {**spec, "depth": dep2}, rng,
                        site_ok)
    # -- the rows along the north/south faces
    for (r0, r1, front) in rows:
        is_principal = bool((front == "north" and north and r0 == north["line"])
                            or (front == "south" and south and r1 == south["line"]))
        x_lo, x_hi = x_limits(r0, r1)
        segs = [(x_lo, x_hi, "house")]
        if is_principal and shop and reach:
            near_w = abs(x_lo - ax) <= abs(x_hi - ax)
            if near_w:
                segs = [(x_lo, x_lo + reach - 1, "shop"), (x_lo + reach, x_hi, "house")]
            else:
                segs = [(x_lo, x_hi - reach, "house"), (x_hi - reach + 1, x_hi, "shop")]
        for k, (s0, s1, use) in enumerate(segs):
            if s1 < s0:
                continue
            spec = shop if use == "shop" else house
            # the full depth first; where the back of the row is broken ground, the
            # stretches left are tried again shallower, down to the least the type
            # stands at, so a pond behind a row costs the lots over it and not the whole
            # run
            for dep in _depths(spec):
                if dep > r1 - r0 + 1:
                    continue
                b0, b1 = (r0, r0 + dep - 1) if front == "north" else (r1 - dep + 1, r1)
                cut_run(P, {"axis": "x", "a0": s0, "a1": s1, "b0": b0, "b1": b1,
                            "front": front, "use": use,
                            "street": "principal" if is_principal else "lane",
                            "id": f"row{r0}_{k}"}, {**spec, "depth": dep}, rng, site_ok)
    party_walls(P.lots)
    return P


def _depths(spec: dict) -> list:
    """The depths a run is tried at: its own, then shallower down to the least its type
    stands at (`depth_lo`)."""
    d = int(spec["depth"])
    lo = int(spec.get("depth_lo") or d)
    return list(range(d, min(d, lo) - 1, -1))


def _place_anchor(P, anchor, by_side, acc, house, shop, fam, rng, site_ok, g):
    """The anchor on the principal street nearest the way in -- at a corner of two where
    there is one -- the walk round it, and the fronts that face it across the walk."""
    gx0, gz0, gx1, gz1 = g
    ax, az = acc
    S = int(fam.get("anchor_side") or anchor["side"])
    cands = []
    for (sx, sz) in itertools.product(range(gx0, gx1 - S + 2), range(gz0, gz1 - S + 2)):
        r = (sx, sz, sx + S - 1, sz + S - 1)
        touches = [s for s, ln in by_side.items()
                   if (s == "north" and sz == ln["line"] and ln["a0"] <= sx + S // 2 <= ln["a1"])
                   or (s == "south" and sz + S - 1 == ln["line"]
                       and ln["a0"] <= sx + S // 2 <= ln["a1"])
                   or (s == "west" and sx == ln["line"] and ln["a0"] <= sz + S // 2 <= ln["a1"])
                   or (s == "east" and sx + S - 1 == ln["line"]
                       and ln["a0"] <= sz + S // 2 <= ln["a1"])]
        if not touches:
            continue
        # a market is open ground: it may be crossed by a lane's mouth but not by
        # buildable-ground refusals
        a0, a1, b0, b1 = sx - P.X0, sx + S - 1 - P.X0, sz - P.Z0, sz + S - 1 - P.Z0
        if a0 < 0 or b0 < 0 or a1 >= P.W or b1 >= P.H:
            continue
        if not P.ok[a0:a1 + 1, b0:b1 + 1].all() or P.used[a0:a1 + 1, b0:b1 + 1].any():
            continue
        d = abs(sx + S / 2.0 - ax) + abs(sz + S / 2.0 - az)
        if fam.get("anchor_at") == "street":
            # **on one street, with room for fronts on both its flanks**: a market a
            # frontage lot in from the corner, so the corner is a building and the
            # market's other three edges can all be faced
            if len(touches) != 1:
                continue
            fd = int((shop or house)["depth"])
            side = touches[0]
            ln = by_side[side]
            lo_a, hi_a = ((sx, sx + S - 1) if side in ("north", "south")
                          else (sz, sz + S - 1))
            if lo_a - ln["a0"] < fd + CLEAR or ln["a1"] - hi_a < fd + CLEAR:
                continue
            cands.append((0, d, sx, sz, touches))
            continue
        cands.append((-len(touches), d, sx, sz, touches))
    if not cands and not by_side:
        # no street reaches this ground yet (a piece whose road is routed after it is
        # cut): the anchor stands at the ground's edge nearest the way in, which is
        # where its road will arrive
        for (sx, sz) in itertools.product(range(gx0, gx1 - S + 2), range(gz0, gz1 - S + 2)):
            a0, a1, b0, b1 = sx - P.X0, sx + S - 1 - P.X0, sz - P.Z0, sz + S - 1 - P.Z0
            if a0 < 0 or b0 < 0 or a1 >= P.W or b1 >= P.H:
                continue
            if not P.ok[a0:a1 + 1, b0:b1 + 1].all() or P.used[a0:a1 + 1, b0:b1 + 1].any():
                continue
            d = abs(sx + S / 2.0 - ax) + abs(sz + S / 2.0 - az)
            cands.append((0, d, sx, sz, []))
    if not cands:
        P.notes.append("no square on a principal street holds the anchor")
        return
    cands.sort()
    _n, _d, sx, sz, touches = cands[int(fam.get("anchor_rank", 0)) % len(cands)]
    r = (sx, sz, sx + S - 1, sz + S - 1)
    # the anchor displaces any lane mouth it sits on: its floor is walkable ground
    a0, a1, b0, b1 = sx - P.X0, sx + S - 1 - P.X0, sz - P.Z0, sz + S - 1 - P.Z0
    P.street[a0:a1 + 1, b0:b1 + 1] = False
    P.take(r)

    def dist(s):
        cx = sx if s == "west" else (sx + S - 1 if s == "east" else sx + S / 2.0)
        cz = sz if s == "north" else (sz + S - 1 if s == "south" else sz + S / 2.0)
        return abs(cx - ax) + abs(cz - az)
    if not touches:
        # its front is the side facing the way in
        dx_, dz_ = ax - (sx + S / 2.0), az - (sz + S / 2.0)
        face = (("east" if dx_ > 0 else "west") if abs(dx_) >= abs(dz_)
                else ("south" if dz_ > 0 else "north"))
    else:
        face = min(touches, key=dist)
    P.anchor = {"rect": list(r), "street_sides": touches, "front": face}
    face_spec = shop if shop else house
    fd = int(face_spec["depth"])
    use = "shop" if shop else "house"
    W_ = MARKET_WALK
    for s in SIDES:
        if s in touches:
            continue
        if s == "east":
            walk = (sx + S, sz, sx + S + W_ - 1, sz + S - 1)
            run = {"axis": "z", "a0": sz, "a1": sz + S - 1 + (W_ if "south" not in touches else 0),
                   "b0": sx + S + W_, "b1": sx + S + W_ + fd - 1, "front": "west"}
        elif s == "west":
            walk = (sx - W_, sz, sx - 1, sz + S - 1)
            run = {"axis": "z", "a0": sz, "a1": sz + S - 1,
                   "b0": sx - W_ - fd, "b1": sx - W_ - 1, "front": "east"}
        elif s == "south":
            walk = (sx, sz + S, sx + S - 1 + (W_ if "east" not in touches else 0),
                    sz + S + W_ - 1)
            run = {"axis": "x", "a0": sx, "a1": sx + S - 1,
                   "b0": sz + S + W_, "b1": sz + S + W_ + fd - 1, "front": "north"}
        else:
            walk = (sx, sz - W_, sx + S - 1, sz - 1)
            run = {"axis": "x", "a0": sx, "a1": sx + S - 1,
                   "b0": sz - W_ - fd, "b1": sz - W_ - 1, "front": "south"}
        P.take(walk, street=True)
        P.streets.append({"kind": "walk", "rect": list(walk)})
        run.update({"use": use, "street": "anchor", "id": f"anchor_{s}"})
        # as deep as the ground behind the walk allows, down to the least the fronts'
        # type stands at: a lane behind a market edge is not a reason to leave it blank
        lo_d = int(face_spec.get("depth_lo") or fd)
        for dep in range(fd, lo_d - 1, -1):
            r2 = dict(run)
            if s in ("east", "south"):
                r2["b1"] = r2["b0"] + dep - 1
            else:
                r2["b0"] = r2["b1"] - dep + 1
            got_e = cut_run(P, r2, {**face_spec, "depth": dep}, rng, site_ok)
            if got_e:
                break


def _open_ground(P: _Plan) -> None:
    """Every free rectangle the streets and lots leave, largest first, named: a yard
    where it is behind lots, a garden where it is its own ground."""
    free = P.ok & ~P.used & ~P.street
    # open ground keeps the lots' clearance (the validator holds areas to it too)
    lotm = P.front >= 0
    if P.anchor is not None:
        ax0, az0, ax1, az1 = P.anchor["rect"]
        lotm[max(0, ax0 - P.X0):ax1 - P.X0 + 1, max(0, az0 - P.Z0):az1 - P.Z0 + 1] = True
    grown = lotm.copy()
    for da in range(-CLEAR, CLEAR + 1):
        for db in range(-CLEAR, CLEAR + 1):
            src = lotm[max(0, -da):P.W - max(0, da), max(0, -db):P.H - max(0, db)]
            grown[max(0, da):P.W - max(0, -da), max(0, db):P.H - max(0, -db)] |= src
    free &= ~grown
    for _k in range(200):
        best = None
        W, H = free.shape
        # largest rectangle of free cells: row-histogram method
        h = np.zeros(H, dtype=int)
        for a in range(W):
            h = np.where(free[a], h + 1, 0)
            stack = []
            for b in range(H + 1):
                cur = h[b] if b < H else 0
                start = b
                while stack and stack[-1][1] >= cur:
                    s, hh = stack.pop()
                    area = hh * (b - s)
                    if best is None or area > best[0]:
                        best = (area, a - hh + 1, a, s, b - 1)
                    start = s
                stack.append((start, cur))
        if best is None or best[0] < 12:
            break
        _area, a0, a1, b0, b1 = best
        if (a1 - a0 + 1) < 3 or (b1 - b0 + 1) < 3:
            free[a0:a1 + 1, b0:b1 + 1] = False
            continue
        r = [a0 + P.X0, b0 + P.Z0, a1 + P.X0, b1 + P.Z0]
        backs = sum(1 for lot in P.lots if _touch(lot["rect"], r))
        P.open.append({"rect": r, "use": "yard" if backs >= 2 and _rect_cells(r) < 150
                       else "garden"})
        free[max(0, a0 - CLEAR):a1 + CLEAR + 1, max(0, b0 - CLEAR):b1 + CLEAR + 1] = False


def _touch(a, b) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def pad_of(lot) -> list:
    """The building a lot will carry, as the builder sites it: the lot inset by
    `FRONT_INSET` on every side that is not a party wall (`buildlib.pad_insets` with
    the composed lot's own inset). Relationships a person sees are judged on this."""
    x0, z0, x1, z1 = lot["rect"]
    att = set(lot.get("attached") or ())
    i = FRONT_INSET
    return [x0 + (0 if "west" in att else i), z0 + (0 if "north" in att else i),
            x1 - (0 if "east" in att else i), z1 - (0 if "south" in att else i)]


def _faces(lot, rect=None):
    """The street edge of a lot -- or of the building on it where `rect` is its pad:
    the columns just outside its front, as a set."""
    x0, z0, x1, z1 = rect or lot["rect"]
    f = lot["front"]
    if f == "north":
        return {(x, z0 - 1) for x in range(x0, x1 + 1)}
    if f == "south":
        return {(x, z1 + 1) for x in range(x0, x1 + 1)}
    if f == "west":
        return {(x0 - 1, z) for z in range(z0, z1 + 1)}
    return {(x1 + 1, z) for z in range(z0, z1 + 1)}


def judge(P: _Plan, house_needed: bool = True, anchor_needed: bool = False,
          defer_roads: bool = False) -> dict:
    """**Required relationships first, preferences second.** Each relationship is
    measured on the proposal's own geometry -- the lots' fronts and the streets as
    reserved -- and a proposal is admissible only where every required one holds. The
    preference ranks admissible proposals and nothing else."""
    rel = []
    # **a lane's street may not be routed yet**: where no road comes within reach of
    # this district at all -- a piece of a strip whose arterial is routed after the cut
    # is adopted -- whether a lane meets a street is not known here, and it is recorded
    # as deferred rather than failed; the compile after routing asks it again
    near_road = any(
        (x, z) in P.road
        for x in range(P.X0 - ROAD_REACH, P.X0 + P.W + ROAD_REACH)
        for z in (P.Z0 - ROAD_REACH, P.Z0 + P.H - 1 + ROAD_REACH)) or any(
        (x, z) in P.road
        for z in range(P.Z0 - ROAD_REACH, P.Z0 + P.H + ROAD_REACH)
        for x in (P.X0 - ROAD_REACH, P.X0 + P.W - 1 + ROAD_REACH)) or any(
        (x, z) in P.road for x in range(P.X0, P.X0 + P.W) for z in range(P.Z0, P.Z0 + P.H))
    if defer_roads:
        # the caller knows this ground's road is routed only after it is adopted (a new
        # piece of a re-cut strip): what reaches a street is asked again then
        near_road = False
    # **walls, not lots**: an anchor's edge is fronted by the buildings that will stand
    # on its lots, which stand in from the lot line wherever they have no party wall.
    fronts = {}
    for lot in P.lots:
        for c in _faces(lot, pad_of(lot)):
            fronts.setdefault(c, []).append(lot)
    # the anchor at its street, and its other edges fronted
    if anchor_needed and P.anchor is None:
        rel.append({"name": "anchor_placed", "required": True, "held": False,
                    "measure": None,
                    "why": "the anchor this district carries has no square on a "
                           "principal street: " + ("; ".join(P.notes) or "none found")})
    if P.anchor is not None:
        x0, z0, x1, z1 = P.anchor["rect"]
        rel.append({"name": "anchor_on_street", "required": near_road,
                    "held": bool(P.anchor["street_sides"]),
                    "measure": P.anchor["street_sides"],
                    "why": ("the market stands on a principal street" if near_road else
                            "deferred: no road is routed to this district yet; the "
                            "market stands at the edge nearest the way in")})
        edges = {}
        for s in SIDES:
            if s in P.anchor["street_sides"]:
                continue
            if s in ("north", "south"):
                line = [(x, (z0 - 1 if s == "north" else z1 + 1)) for x in range(x0, x1 + 1)]
                step = (0, -1 if s == "north" else 1)
            else:
                line = [((x0 - 1 if s == "west" else x1 + 1), z) for z in range(z0, z1 + 1)]
                step = (-1 if s == "west" else 1, 0)
            hit = 0
            for (cx, cz) in line:
                # a front facing the market within its walk
                if any((cx + step[0] * k, cz + step[1] * k) in fronts
                       for k in range(0, MARKET_WALK + FRONT_INSET + 2)):
                    hit += 1
            edges[s] = round(hit / float(len(line)), 3)
        want = min(2, len(edges))
        got = sum(1 for v in edges.values() if v >= EDGE_FRONTED)
        # which of its edges are the street's is only known once its road arrives
        rel.append({"name": "anchor_edges_fronted", "required": near_road,
                    "held": got >= want, "measure": edges,
                    "why": (f"{got} of the market's {len(edges)} edges off the street are "
                            f"at least {EDGE_FRONTED:.0%} fronted by doors facing it "
                            f"across its walk; {want} are required")})
    # each principal street lined along the district's side, and active on it
    for ln in P.lines:
        side = ln["side"]
        marks = []                      # per column: "door", "wall", "open" or None
        for a in range(ln["a0"], ln["a1"] + 1):
            x, z = ((a, ln["line"]) if side in ("north", "south") else (ln["line"], a))
            if P.anchor and P.anchor["rect"][0] <= x <= P.anchor["rect"][2] \
                    and P.anchor["rect"][1] <= z <= P.anchor["rect"][3]:
                marks.append("door")    # the market's own open side is its front
                continue
            lot_here = next((lot for lot in P.lots
                             if lot["rect"][0] <= x <= lot["rect"][2]
                             and lot["rect"][1] <= z <= lot["rect"][3]), None)
            if lot_here is not None:
                # a front on this street -- or a front on the market that opens onto it:
                # the market and the shops facing it are one frontage on its street
                marks.append("door" if (lot_here["front"] == side
                                        or lot_here.get("street") == "anchor")
                             else "wall")
            elif P.inside(x, z) and P.street[x - P.X0, z - P.Z0]:
                marks.append("open")    # a lane's mouth
            else:
                marks.append(None)
        # a lot's clearance between two lots on the street is a passage, not a void
        for i, m in enumerate(marks):
            if m is None:
                lo = next((marks[j] for j in range(i - 1, max(-1, i - CLEAR - 1), -1)
                           if marks[j] is not None), None)
                hi = next((marks[j] for j in range(i + 1, min(len(marks), i + CLEAR + 1))
                           if marks[j] is not None), None)
                if lo in ("door", "wall") and hi in ("door", "wall"):
                    marks[i] = "gap"
        n = float(max(1, len(marks)))
        lined = sum(1 for m in marks if m is not None) / n
        doors = sum(1 for m in marks if m == "door") / n
        rel.append({"name": f"principal_{side}_fronted", "required": True,
                    "held": lined >= PRINCIPAL_FRONTED and doors >= PRINCIPAL_ACTIVE,
                    # `fronts`: the share of the run whose lot fronts the street (or the
                    # market on it) -- an oriented lot, not a count of openings
                    "measure": {"lined": round(lined, 3), "fronts": round(doors, 3)},
                    "why": (f"{lined:.0%} of the {len(marks)}-column run of the street on "
                            f"the district's {side} is lined by buildings, the market or "
                            f"a lane mouth ({PRINCIPAL_FRONTED:.0%} required), and "
                            f"{doors:.0%} of it by fronts facing the street or the "
                            f"market's open side ({PRINCIPAL_ACTIVE:.0%} required)")})
    # lanes fronted and connected
    reach_road = set()
    for i, st in enumerate(P.streets):
        if st["kind"] not in ("lane", "cross"):
            continue
        x0, z0, x1, z1 = st["rect"]
        ring = [(x, z) for x in range(x0 - 4, x1 + 5) for z in (z0 - 4, z1 + 4)] + \
               [(x, z) for z in range(z0 - 4, z1 + 5) for x in (x0 - 4, x1 + 4)]
        if any(c in P.road for c in ring) or any(
                c in P.road for c in [(x, z) for x in range(x0, x1 + 1)
                                      for z in range(z0, z1 + 1)]):
            reach_road.add(i)
    changed = True
    while changed:
        changed = False
        for i, st in enumerate(P.streets):
            if st["kind"] not in ("lane", "cross") or i in reach_road:
                continue
            if any(_touch(st["rect"], P.streets[j]["rect"]) for j in reach_road
                   if P.streets[j]["kind"] in ("lane", "cross")):
                reach_road.add(i)
                changed = True
    for i, st in enumerate(P.streets):
        if st["kind"] != "lane":
            continue
        x0, z0, x1, z1 = st["rect"]
        n_up = sum((min(x1, lot["rect"][2]) - max(x0, lot["rect"][0]) + 1)
                   for lot in P.lots
                   if lot["front"] == "south" and lot["rect"][3] + 1 == z0)
        n_dn = sum((min(x1, lot["rect"][2]) - max(x0, lot["rect"][0]) + 1)
                   for lot in P.lots
                   if lot["front"] == "north" and lot["rect"][1] - 1 == z1)
        length = max(1, x1 - x0 + 1)
        share = max(n_up, n_dn) / float(length)
        if n_up == 0 and n_dn == 0:
            # **no lot fronts it at all**: its rows did not stand (their ground was
            # refused), so it is a footway through the open ground and not a lane of the
            # quarter -- recorded, and not a relationship this composition claims
            rel.append({"name": f"lane_{i}_unfronted", "required": False, "held": True,
                        "measure": 0.0,
                        "why": "no lot fronts this way on either side: a footway "
                               "through open ground, not a lane the quarter claims"})
            continue
        rel.append({"name": f"lane_{i}_fronted", "required": True,
                    "held": share >= LANE_FRONTED, "measure": round(share, 3),
                    "why": f"{share:.0%} of the lane's length has doors on one side"})
        rel.append({"name": f"lane_{i}_reaches_street", "required": near_road,
                    "held": i in reach_road, "measure": i in reach_road,
                    "why": ("the lane meets a street, directly or by a cross lane"
                            if near_road else
                            "deferred: no road is routed to this district yet")})
    houses = sum(1 for lot in P.lots if lot["use"] == "house")
    shops = sum(1 for lot in P.lots if lot["use"] == "shop")
    if house_needed:
        # a shop house is a dwelling too (the room over the shop): a piece that anchors
        # the market street may be mostly shops
        rel.append({"name": "dwellings", "required": True,
                    "held": houses + shops >= DWELLINGS_LEAST,
                    "measure": {"houses": houses, "shops": shops},
                    "why": (f"{houses} courtyard house(s) and {shops} shop house(s); "
                            f"{DWELLINGS_LEAST} dwellings is a quarter")})
    attached = sum(1 for lot in P.lots if lot["attached"])
    joined = attached / float(max(1, len(P.lots)))
    frontage = [(r["measure"]["fronts"] if r["name"].startswith("principal_")
                 else r["measure"]) for r in rel
                if r["name"].endswith("_fronted") and not r["name"].startswith("anchor")
                and isinstance(r["measure"], (float, dict))]
    open_cols = sum(_rect_cells(o["rect"]) for o in P.open)
    pref = (houses + 0.5 * shops + 4.0 * joined
            + 4.0 * (sum(frontage) / float(max(1, len(frontage)))))
    ok = all(r["held"] for r in rel if r["required"])
    return {"admissible": bool(ok), "relationships": rel,
            "preference": round(pref, 3),
            "houses": houses, "shops": shops, "joined": round(joined, 3),
            "open_columns": int(open_cols),
            "failed": [r["name"] for r in rel if r["required"] and not r["held"]]}


def _transpose_lot(lot):
    x0, z0, x1, z1 = lot["rect"]
    return {**lot, "rect": [z0, x0, z1, x1], "front": _T[lot["front"]],
            "attached": [_T[s] for s in lot.get("attached") or []]}


def _transpose_rect(r):
    return [r[1], r[0], r[3], r[2]]


def compose(rect, ok, road, *, access=None, house=None, shop=None, anchor=None,
            seed=1, site_ok=None, families=None, defer_roads=False) -> dict:
    """Every proposal of the family, judged, and the one to adopt.

    `rect` is the district `(X0, Z0, X1, Z1)`; `ok` its buildable columns, `[x - X0,
    z - Z0]`; `road` the arterial cells; `access` the point it is entered from;
    `house`/`shop` `{"widths": (lo, hi), "pref", "depth"}` in plot columns; `anchor`
    `{"side"}` or None; `site_ok(rect, front, spec) -> bool` the compiler's own test that
    a lot's site can be built. Returns `{"adopted", "proposals": [...]}` where each
    proposal carries its geometry in world columns and its verdict."""
    X0, Z0, X1, Z1 = rect
    fams = families or default_families(house, shop, anchor)
    out = []
    for k, fam in enumerate(fams):
        rng = random.Random(f"{seed}/streetplan/{k}")
        if fam["axis"] == "x":
            P = _propose(X0, Z0, ok, road, access, house, shop, anchor, fam, rng, site_ok)
            _open_ground(P)
            verdict = judge(P, house_needed=house is not None,
                            anchor_needed=bool(anchor), defer_roads=defer_roads)
            lots = P.lots
            streets = P.streets
            anch = P.anchor
            opens = P.open
        else:
            okT = ok.T.copy()
            roadT = {(z, x) for (x, z) in road}
            accT = (access[1], access[0]) if access else None

            def site_T(r, front, spec):
                return site_ok(_transpose_rect(r), _T[front], spec) if site_ok else True
            P = _propose(Z0, X0, okT, roadT, accT, house, shop, anchor, fam, rng,
                         site_T if site_ok else None)
            _open_ground(P)
            verdict = judge(P, house_needed=house is not None,
                            anchor_needed=bool(anchor), defer_roads=defer_roads)
            lots = [_transpose_lot(lot) for lot in P.lots]
            streets = []
            for st in P.streets:
                if "rect" in st:
                    streets.append({**st, "rect": _transpose_rect(st["rect"]),
                                    "axis": "z" if st.get("axis") == "x" else "x"})
                else:
                    streets.append({**st, "side": _T[st["side"]]})
            anch = None
            if P.anchor:
                anch = {"rect": _transpose_rect(P.anchor["rect"]),
                        "street_sides": [_T[s] for s in P.anchor["street_sides"]],
                        "front": _T[P.anchor["front"]]}
            opens = [{**o, "rect": _transpose_rect(o["rect"])} for o in P.open]
        out.append({"family": fam, "lots": lots, "streets": streets, "anchor": anch,
                    "open": opens, **verdict})
    admissible = [p for p in out if p["admissible"]]
    adopted = max(admissible, key=lambda p: p["preference"]) if admissible else None
    return {"adopted": adopted, "proposals": out,
            "why": (f"{len(admissible)} of {len(out)} proposals hold every required "
                    f"relationship" + (f"; adopted the one preferred among them "
                                       f"({_fam_name(adopted['family'])})" if adopted else
                                       "; none does, so none is adopted"))}


def _fam_name(f) -> str:
    return (f"lanes along {f['axis']}, lots {f['depth']} deep, shops "
            f"{f.get('shop_reach') or 0} along the principal street"
            + (f", market {f['anchor_side']} across" if f.get("anchor_side") else "")
            + (", market a lot in from the corner" if f.get("anchor_at") == "street"
               else ""))


def default_families(house, shop, anchor=None) -> list:
    """The materially different proposals tried: the lane grain either way, the house lot
    at its preferred depth and one step either side, and the shop frontage short or
    long. Small on purpose: these are different organisations, not a parameter sweep."""
    d0 = int(house["depth"])
    depths = sorted({d0, max(house.get("depth_lo", d0), d0 - 2),
                     min(house.get("depth_hi", d0), d0 + 2)})
    reaches = [0] if not shop else [48, 96]
    # the anchor at the side it wants and at one between that and its least: a market
    # floor is sized inside its requirement's band, and a smaller one can be the one
    # whose edges its neighbours front
    sides = [None]
    if anchor and anchor.get("least") and int(anchor["least"]) < int(anchor["side"]):
        sides.append((int(anchor["side"]) + int(anchor["least"]) + 1) // 2)
        sides.append(int(anchor["least"]))
    # ...and at the corner of two principal streets or on one of them a lot in
    ats = ["corner"] + (["street"] if anchor else [])
    fams = []
    for axis in ("x", "z"):
        for d in depths:
            for r in reaches:
                for sd in sides:
                    for at in ats:
                        fams.append({"axis": axis, "depth": d, "shop_reach": r,
                                     **({"anchor_side": sd} if sd else {}),
                                     **({"anchor_at": at} if at != "corner" else {})})
    return fams


def rejudge(rect, ok, road, lots, streets, anchor, *, house_needed=True,
            anchor_needed=False, defer_roads=False) -> dict:
    """**The verdict on what was actually emitted**, not on the proposal. The compiler
    settles each lot after the proposal is chosen -- a lot whose site or storeys do not
    stand is refused and its neighbours lose a party wall -- so the relationships are
    judged again on the surviving lots, with the same rules. `lots` are
    `{"rect", "front", "use", "street", "attached"}` in world columns."""
    X0, Z0 = rect[0], rect[1]
    P = _Plan(X0, Z0, ok, road)
    P.lines = principal_lines(P)
    P.lots = [dict(lot) for lot in lots]
    for lot in P.lots:
        P.take(lot["rect"])
        r = lot["rect"]
        P.front[r[0] - X0:r[2] - X0 + 1, r[1] - Z0:r[3] - Z0 + 1] = FRONTS.index(lot["front"])
    for st in streets or []:
        if "rect" in st:
            P.take(st["rect"], street=True)
    P.streets = [dict(s) for s in (streets or [])]
    P.anchor = dict(anchor) if anchor else None
    return judge(P, house_needed=house_needed, anchor_needed=anchor_needed,
                 defer_roads=defer_roads)
