import math
import random

KIND = "edge"
FORM = "fortification"
ROLE = "defensive"
#: **What family of part this builds**, said outright rather than left to be read off
#: the filename. `capability._named_for` reads a family off the committed name, and
#: `great_wall` neither is `wall` nor starts with `wall_` -- so the capability record
#: would match a place's wall to `wall.py` while the ring layout, which needs a boundary
#: that can draw a diagonal run, builds this one, and the record and the layout would
#: disagree about what the place is made of. A type that is a wall says so.
FAMILY = "wall"
#: Its frame is a unit direction and a unit normal, and on a diagonal the direction is
#: (1,1) and the normal (-1,1), so the same arithmetic lays the body, the parapet and
#: the corner squares along a staircase of the wall's width. A diagonal run carries no
#: mural stair -- a tread has no diagonal facing -- so the ways down stand on the axial
#: runs.
DIAGONAL_RUNS = True
PARAMS = {
    "height": ("int", 24, 48),
    # The body's width is the part's own `width` (the swept line siting hands this
    # type); this parameter is held to one value so the sweep across every parameter
    # stays a size that can be run, and so a stored plan that passes it still checks.
    # **The mass is the part's**: a place declares what its wall is -- a screen, a
    # curtain, a rampart, a levee -- and `width` on the part is what that word means in
    # columns.
    "width": ("int", 3, 3),
    "parapet": ("choice", ["crenellated", "plain"]),
    # The face: `framed` is the grid of frame posts and bands this wall has always
    # carried; `banded` is masonry with a string course every few courses and no posts;
    # `plain` is one unbroken outer face under its cornice, an earthen or monolithic
    # wall; `masonry` is dressed stonework -- a plinth course at the foot, a string
    # course at `STRING_EVERY`, buttress piers at `PIER_EVERY` and a batter that steps
    # the face in as it rises, and the ways up at the corners and the gates only, where
    # a framed or banded face carries one every `STAIR_EVERY` columns and reads from the
    # air as timber cross-bracing. **`unbroken` is not among the choices and is still
    # accepted**: it means the plain face on both sides with sparse stairs, and masonry
    # is what an unbroken wall gets, so the word says nothing masonry does not. A stored
    # plan that passes it builds masonry. First is the default, so every stored program
    # builds the bytes it built.
    "face": ("choice", ["framed", "banded", "plain", "masonry"]),
}
NEEDS = {
    # The band scripts/type_needs.py measured. **The width reaches a rampart's**: a sweep
    # that tries only widths 1 to 3 certifies nothing thicker, and a wall forty-eight
    # high and three thick is a screen. `clearance` is the author's: the stair blocks
    # reach six lanes past the inner face where the wall is too thin to carry them
    # inside itself.
    "footprint": (1, 4, 12, 128),
    "frontage": "any",
    "ground": "any",
    "clearance": 8,
}

#: **A crown wide enough to be a road**. Below this the wall is a parapet and a ledge;
#: at it and above, the walk is a road between two parapets -- one on each edge --
#: which is what a person on top of a great wall is walking along.
CROWN_ROAD_MIN = 5

#: **A mural stair stands in the wall's own thickness** where the wall has this much to
#: spare over the lanes the flights need: a switchback of `nfl` lanes takes `nfl` of the
#: inner columns and what is left is still a walk two wide. Below it the stair is a
#: block against the inner face, as it has always been.
STAIR_INSIDE_SPARE = 2

#: The dressed face (`masonry`). Forty-eight blocks of one flat plane reads as a render
#: and the framed grid reads as lattice; between them is masonry. A plinth of
#: `PLINTH_H` courses at the foot in the footing family, a string course of trim every
#: `STRING_EVERY`, a buttress pier of the wall family every
#: `PIER_EVERY` columns standing one course proud at the crown, and a batter that steps
#: the outer lanes in by one every `BATTER_EVERY` courses of height.
PLINTH_H = 3
STRING_EVERY = 9
PIER_EVERY = 11
BATTER_EVERY = 14

# Longest single flight of the way down; more flights are added past this.
MAX_FLIGHT = 12
# A way down at least this often along the walk (the brief asks for 32).
STAIR_EVERY = 26.0
# The widest switchback `_place_stairs` will search for: `range(nfl0, 7)`.
MAX_LANES = 6
# Courses of parapet and merlon above `height`, which is the walk's level.
PARAPET_COURSES = 2


def occupied(part=None, **params):
    """**What this wall actually fills**, beyond the band the layout drew for it.

        The wall's piers and corbels extend beyond its nominal width, and more road
        clearance avoids one collision without giving planning, terrain, circulation,
        ownership and checking a common occupied envelope. This is that envelope,
        published by the type that knows it, so planning, ground, routing, emission and
        the checks read one number instead of four guesses.

        Measured by building this type on flat ground at widths 2, 3, 4, 5, 6 and 8 and
        heights 24, 36 and 48 and comparing the emitted columns with the swept band:

            width 2, H 24/36/48   2 / 3 / 4 columns past one face, 1 past the other
            width 3, H 24/36/48   1 past the outer face, 2 / 3 / 4 past the inner
            width 5, H 24/36/48   1 / 1 / 4 past the inner face
            width 8, H 48         1 either side

        Three things reach past the band and this is all three:

          * **the parapet's corbel course** -- one column outside the outer face at the
            walk's level, on every face style, and one inside the inner face as well where
            the crown is a road (`CROWN_ROAD_MIN`) and carries two parapets;
          * **the buttress piers** (`face="masonry"`, `PIER_EVERY`) -- the same one column,
            full height, so they add nothing the corbel does not already ask for;
          * **the switchback stair**, where the wall is too thin to carry it inside its own
            thickness (`STAIR_INSIDE_SPARE`). That block hangs past **one** face and which
            one is a fact about the ground (`_pick_side`) that this function has not seen,
            so it is published as `either` and reserved on both sides by a caller that
            cannot wait for the siting.

        The **batter** (`BATTER_EVERY`) steps the outer lanes *in* as the wall rises, so it
        never projects; it is named here because "the batter projects" is the wrong half of
        the true statement.

            {"band": the width the layout drew,
             "outer": columns of solid beyond the outer face, certain,
             "inner": columns of solid beyond the inner face, certain,
             "either": columns that hang past ONE of the two faces, side not yet decided,
             "above": courses of solid above `height`,
             "clearance": the free ground this type needs beside the band,
             "total": the columns a planner must reserve across the line,
             "why": str}

    """
    part = part or {}
    width = int(part.get("width") or params.get("width") or 3)
    _lo, hi = PARAMS["height"][1], PARAMS["height"][2]
    H = int(params.get("height") or hi)
    if _curved_path(part):
        # **a curved wall's envelope**: its towers project past the outer face and its
        # stair turrets stand against the inner one
        half = (width - 1) // 2
        t_r = max(4.5, half + 3.5)
        outer = int(math.ceil(1.5 + half * 0.5 + t_r - half))
        inner = 8
        return {"band": width, "outer": outer, "inner": inner, "either": 0,
                "above": PARAPET_COURSES + 8, "clearance": int(NEEDS["clearance"]),
                "total": width + outer + inner,
                "why": (f"a {width}-wide curved wall {H} high: towers every "
                        f"{TOWER_EVERY} blocks of arc project {outer} past the outer "
                        f"face, stair turrets stand {inner} into the inner side")}
    #: The lane counts `_place_stairs` may choose between, and the widest of them that
    #: does not fit inside the thickness. The solver prefers the fewest lanes and is
    #: free to take more where the ground refuses a placement -- a width-6 wall 48 high
    #: measured four columns of hanging stair though its cheapest switchback would have
    #: fitted -- so the published figure is the bound over the search and not the
    #: cheapest case. A wall thick enough for the widest switchback hangs nothing.
    nfl0 = max(2, int(math.ceil(H / float(MAX_FLIGHT))))
    hangs = [n for n in range(nfl0, MAX_LANES + 1)
             if width < n + STAIR_INSIDE_SPARE]
    either = max(hangs) if hangs else 0
    inner = 1 if width >= CROWN_ROAD_MIN else 0
    return {"band": width, "outer": 1, "inner": inner, "either": either,
            "above": PARAPET_COURSES, "clearance": int(NEEDS["clearance"]),
            "total": width + 1 + inner + either,
            "why": (f"a {width}-wide wall {H} high fills {width + 1 + inner + either} "
                    f"columns across the line: the band, one column of corbel outside"
                    + (", one inside under the crown road's inner parapet" if inner
                       else "")
                    + (f", and up to {either} columns of switchback stair hanging past "
                       f"one face (the side is chosen from the ground)" if either else
                       ", and a switchback that stands inside its own thickness")
                    + f". The batter steps in, not out. Clearance "
                      f"{NEEDS['clearance']}.")}


def _sgn(v):
    return (v > 0) - (v < 0)


def _xz(fr, t, o):
    return (fr["ax"] + t * fr["dx"] + o * fr["nx"],
            fr["az"] + t * fr["dz"] + o * fr["nz"])


def _to(fr, x, z):
    ex, ez = x - fr["ax"], z - fr["az"]
    # divided by the squared length of the frame's vectors, which is 1 on an axial run
    # and 2 on a diagonal one, so t and o are the cell's own steps either way
    dd = fr["dx"] * fr["dx"] + fr["dz"] * fr["dz"]
    nn = fr["nx"] * fr["nx"] + fr["nz"] * fr["nz"]
    return ((ex * fr["dx"] + ez * fr["dz"]) // dd, (ex * fr["nx"] + ez * fr["nz"]) // nn)


def _frames(segs, flip):
    """One frame per segment: origin a, along-direction d, inner normal n,
    and the perpendicular offsets its cells cover."""
    pts = [tuple(s["a"]) for s in segs] + [tuple(segs[-1]["b"])]
    cx = sum(p[0] for p in pts) / float(len(pts))
    cz = sum(p[1] for p in pts) / float(len(pts))
    out = []
    for s in segs:
        ax, az = s["a"]
        bx, bz = s["b"]
        dx, dz = _sgn(bx - ax), _sgn(bz - az)
        L = max(abs(bx - ax), abs(bz - az)) + 1
        nx, nz = -dz, dx
        mx, mz = (ax + bx) / 2.0, (az + bz) / 2.0
        if (cx - mx) * nx + (cz - mz) * nz < 0:
            nx, nz = -nx, -nz
        nx, nz = nx * flip, nz * flip
        cells = [tuple(c) for c in s["cells"]]
        offs = sorted(set((x - ax) * nx + (z - az) * nz for x, z in cells))
        out.append({"ax": ax, "az": az, "dx": dx, "dz": dz, "nx": nx, "nz": nz,
                    "L": L, "F": s["floor_y"], "axis": s["axis"], "offs": offs,
                    "outer": offs[0], "inner": offs[-1], "cells": cells})
    return out


def _pick_side(b, frs):
    """For an open line: the side whose ground sits nearest the footing level
    is the side the ways down should land on."""
    plus = minus = 0.0
    for fr in frs:
        for t in range(0, fr["L"], 4):
            x, z = _xz(fr, t, fr["inner"] + 3)
            plus += abs(b.get_height(x, z) - fr["F"])
            x, z = _xz(fr, t, fr["outer"] - 3)
            minus += abs(b.get_height(x, z) - fr["F"])
    return -1 if minus < plus * 0.8 else 1


def _rec(F, teff, par, face, t, o, seg, corner, post, corb, lane=0, shoulder=False):
    return {"F": F, "teff": teff, "par": par, "face": face, "t": t, "o": o,
            "seg": seg, "tread": False, "corner": corner, "gate": False,
            "jamb": False, "post": post, "corb": corb, "ramp": None,
            "lane": lane, "shoulder": shoulder}


def _batter_lanes(width, H, batter):
    """How many of the outer lanes step in as the wall rises, and by how much.

        A battered wall is a trapezoid, wide at the foot and narrow at the crown, and in a
        lattice that is the outer lanes stopping short. Never more than takes the crown
        below two columns, and never on a wall too low to show it.

    """
    if not batter or width < 3 or H < 2 * BATTER_EVERY:
        return 0
    return max(0, min(width - 2, (H - 1) // BATTER_EVERY - 1))


def _register_segments(frs, H, cells, road=False, batter=False):
    for i, fr in enumerate(frs):
        T = fr["F"] + H
        offs = fr["offs"]
        nb = _batter_lanes(len(offs), H, batter)
        for (x, z) in fr["cells"]:
            t, o = _to(fr, x, z)
            lane = offs.index(o) if o in offs else 0
            # a battered lane stops short of the crown, and the lane above it is what a
            # person standing outside sees next
            top = T - (nb - lane) * BATTER_EVERY if lane < nb else T
            crown = lane >= nb
            par = (lane == nb) or (road and o == fr["inner"])
            rec = cells.get((x, z))
            if rec is None:
                corb = []
                if o == fr["outer"]:
                    corb.append((x - fr["nx"], z - fr["nz"]))
                cells[(x, z)] = _rec(fr["F"], top, par,
                                     o in (fr["outer"], fr["inner"]) or not crown,
                                     t, o, i, False, False, corb,
                                     lane=lane, shoulder=not crown)
            else:
                rec["F"] = min(rec["F"], fr["F"])
                rec["teff"] = max(rec["teff"], top)


def _corner_square(frs, segs, i, j, H, cells):
    """The width-by-width block at the vertex where segment i meets segment j,
    at the higher of the two levels. Returns the square cells."""
    fi, fj = frs[i], frs[j]
    vx, vz = segs[i]["b"]
    Th = max(fi["F"], fj["F"]) + H
    Fl = min(fi["F"], fj["F"])
    sq = []
    for oa in fi["offs"]:
        for ob in fj["offs"]:
            x = vx + oa * fi["nx"] + ob * fj["nx"]
            z = vz + oa * fi["nz"] + ob * fj["nz"]
            par = (oa == fi["outer"]) or (ob == fj["outer"])
            corb = []
            if oa == fi["outer"]:
                corb.append((x - fi["nx"], z - fi["nz"]))
            if ob == fj["outer"]:
                corb.append((x - fj["nx"], z - fj["nz"]))
            face = par or oa == fi["inner"] or ob == fj["inner"]
            cells[(x, z)] = _rec(Fl, Th, par, face, 0, oa, i, True, True, corb)
            sq.append((x, z))
    return sq


def _corners(frs, segs, closed, H, cells):
    """Corner squares at every vertex, and where the two levels differ a tread
    ramp on the walkway of the lower segment climbing to the vertex. Returns
    per-segment forbidden t-ranges (corner squares plus ramps)."""
    zones = [[] for _ in frs]
    pairs = [(i, i + 1) for i in range(len(frs) - 1)]
    if closed:
        pairs.append((len(frs) - 1, 0))
    for (i, j) in pairs:
        fi, fj = frs[i], frs[j]
        sq = _corner_square(frs, segs, i, j, H, cells)
        ti = [_to(fi, x, z)[0] for x, z in sq]
        tj = [_to(fj, x, z)[0] for x, z in sq]
        zi = [min(ti), max(ti)]
        zj = [min(tj), max(tj)]
        Ti, Tj = fi["F"] + H, fj["F"] + H
        Th = max(Ti, Tj)
        D = abs(Ti - Tj)
        if D > 0:
            if Ti < Tj:
                low, t_edge, sign = i, min(ti), -1
                zi[0] -= D
            else:
                low, t_edge, sign = j, max(tj), 1
                zj[1] += D
            for (x, z), rec in cells.items():
                if rec["seg"] != low or rec["corner"]:
                    continue
                jj = (rec["t"] - t_edge) * sign
                if 1 <= jj <= D:
                    rec["teff"] = Th - jj + 1
                    rec["tread"] = not rec["par"]
                    rec["ramp"] = (i, j)
        zones[i].append(tuple(zi))
        zones[j].append(tuple(zj))
    return zones


def _flights_for(H, nfl):
    base, extra = divmod(H, nfl)
    return [base + (1 if q < extra else 0) for q in range(nfl)]


def _stair_offsets(fr, nfl, inside):
    """The perpendicular offsets the `nfl` lanes of a switchback occupy, bottom flight
    first. **Inside the wall's own thickness where the mass allows it**: a great wall
    carries its ways up in its body, and only a thin one hangs a block of stairs on its
    inner face."""
    n = fr["nx"] * 0 + 1                            # direction of `inner` from `outer`
    step = n if fr["inner"] > fr["outer"] else -n
    if inside:
        return [fr["inner"] - step * q for q in range(nfl)]
    return [fr["inner"] + step * (nfl - q) for q in range(nfl)]


def _stair_candidate(b, fr, i, nfl, run, p0, zones, wallset, occupied, desired,
                     nfl0, inside=False):
    L, F = fr["L"], fr["F"]
    t0, t1 = p0, p0 + run - 1
    if t0 < 0 or t1 > L - 1:
        return None
    for (zl, zh) in zones[i]:
        if t0 <= zh + 1 and t1 >= zl - 1:
            return None
    offs = _stair_offsets(fr, nfl, inside)
    for t in range(t0 - 1, t1 + 2):
        for o in offs:
            x, z = _xz(fr, t, o)
            if (x, z) in occupied or (not inside and (x, z) in wallset):
                return None
            if inside and (x, z) not in wallset:
                return None
    pen = 0.0
    for t in range(t0, t1 + 1, 3):
        for o in (offs[0], offs[-1]):
            x, z = _xz(fr, t, o)
            g = b.get_height(x, z)
            if g > F:
                pen += 6.0 * (g - F)
            else:
                pen += 0.4 * (F - g)
    return abs((t0 + t1) / 2.0 - desired) + pen + (nfl - nfl0) * 8.0


def _lay_stair(fr, H, nfl, ns, run, p0, s_top, ramp, flights, inside=False, cells=None):
    """A switchback stair: one flight per lane, the bottom flight furthest from the
    walk, the top flight landing beside it. Against the inner face where the wall is
    thin, and **in the wall's own thickness** where the mass allows -- in which case the
    lanes it takes come out of the body and the ramp lays them instead."""
    F = fr["F"]
    t0, t1 = p0, p0 + run - 1
    offs = _stair_offsets(fr, nfl, inside)
    Ls = F
    for q in range(nfl):
        n = ns[q]
        o = offs[q]
        s = s_top * (1 if (nfl - 1 - q) % 2 == 0 else -1)
        order = list(range(t0, t1 + 1)) if s > 0 else list(range(t1, t0 - 1, -1))
        fl = []
        for idx, t in enumerate(order):
            x, z = _xz(fr, t, o)
            if inside and cells is not None:
                cells.pop((x, z), None)
            if idx == 0:
                ramp[(x, z)] = {"F": F, "solid": Ls, "tread": None}
            elif idx <= n:
                y = Ls + idx
                ramp[(x, z)] = {"F": F, "solid": y - 1, "tread": y}
                fl.append((x, y, z))
            else:
                ramp[(x, z)] = {"F": F, "solid": Ls + n, "tread": None}
        flights.append((fl, fr["axis"]))
        Ls += n


def _desired(fr, i, nseg, closed, stairs, gate_ts, rng):
    """Where this segment wants its ways down, as positions along it.

    `every`: one every `STAIR_EVERY` columns, spread over the run, each nudged by the
    seed. `sparse`: one beside the corner the segment starts at (every corner of a
    closed loop, every vertex of an open line, plus the far end of an open line) and
    one beside each gate on the segment, on the side with more room -- and nothing
    else, so the face between reads as one surface."""
    L = fr["L"]
    if fr.get("axis") == "d":
        return []                                   # no tread faces a diagonal
    if stairs != "sparse":
        nst = max(1, int(math.ceil(L / STAIR_EVERY)))
        return [(k + 0.5) * L / float(nst) + rng.randint(-3, 3) for k in range(nst)]
    out = []
    if closed or i > 0:
        out.append(min(L - 1.0, CORNER_STAIR_AT))
    if not closed and i == nseg - 1:
        out.append(max(0.0, L - 1.0 - CORNER_STAIR_AT))
    for (t, half) in gate_ts:
        off = half + CORNER_STAIR_AT
        out.append(t - off if t >= L - t else t + off)
    return out


#: How far along a segment from its corner, or from a gate's pad, a sparse stair is
#: asked to stand: past the corner square and its ramp, close enough to read as the
#: corner's own way down.
CORNER_STAIR_AT = 10.0


def _place_stairs(b, frs, H, zones, cells, rng, s_top, ramp, flights,
                  stairs="every", closed=True, gates=None, width=3):
    wallset = set(cells.keys())
    occupied = set()
    ranges = [[] for _ in frs]
    nfl0 = max(2, int(math.ceil(H / float(MAX_FLIGHT))))
    for i, fr in enumerate(frs):
        L = fr["L"]
        gate_ts = []
        for g in (gates or []):
            gx, gz = int(g["at"][0]), int(g["at"][-1])
            if (gx, gz) in set(fr["cells"]):
                t, _o = _to(fr, gx, gz)
                gate_ts.append((float(t), int(g.get("size", 5)) // 2 + 3))
        for desired in _desired(fr, i, len(frs), closed, stairs, gate_ts, rng):
            best = None
            for nfl in range(nfl0, 7):
                # **In the wall's own thickness where the mass allows**: a switchback of
                # `nfl` lanes leaves a walk `STAIR_INSIDE_SPARE` wide. A thin wall hangs
                # its stair on its inner face as it always has.
                inside = width >= nfl + STAIR_INSIDE_SPARE
                ns = _flights_for(H, nfl)
                run = max(ns) + 2
                c = int(round(desired - run / 2.0))
                for p0 in range(c - 14, c + 15):
                    sc = _stair_candidate(b, fr, i, nfl, run, p0, zones, wallset,
                                          occupied, desired, nfl0, inside=inside)
                    if sc is not None and (best is None or sc < best[0]):
                        best = (sc, nfl, ns, run, p0, inside)
            if best is None:
                continue
            sc, nfl, ns, run, p0, inside = best
            _lay_stair(fr, H, nfl, ns, run, p0, s_top, ramp, flights,
                       inside=inside, cells=cells)
            for t in range(p0 - 1, p0 + run + 1):
                for o in _stair_offsets(fr, nfl + 1, inside):
                    occupied.add(_xz(fr, t, o))
            ranges[i].append((p0, p0 + run - 1))
    return ranges


def _gate(frs, zones, ranges, cells, rng):
    """One way through a closed loop, three wide and four high, clear of the
    corners and of every way down."""
    order = list(range(len(frs)))
    rng.shuffle(order)
    for i in order:
        fr = frs[i]
        L = fr["L"]
        mid = L / 2.0 + rng.randint(-6, 6)
        cands = sorted(range(3, L - 6), key=lambda t: abs(t - mid))
        for tg in cands:
            lo, hi = tg - 2, tg + 4
            bad = False
            for (zl, zh) in zones[i]:
                if lo <= zh + 1 and hi >= zl - 1:
                    bad = True
            for (r0, r1) in ranges[i]:
                if lo <= r1 + 2 and hi >= r0 - 2:
                    bad = True
            if bad:
                continue
            for (x, z), rec in cells.items():
                if rec["seg"] != i or rec["corner"]:
                    continue
                if tg <= rec["t"] <= tg + 2:
                    rec["gate"] = True
                elif rec["t"] in (tg - 1, tg + 3):
                    rec["jamb"] = True
            return True
    return False


# ======================================================================================
# THE CURVED WALL ENGINE -- one text, carried verbatim by `wall.py` and `great_wall.py`
# (a type file is a standalone program and cannot import its sibling;
# `scripts/test_round_walls.py` asserts the two copies are identical). A round ring is
# `boundary.Outline.polyline`: hundreds of short axial and 45-degree runs. Drawn segment
# by segment, as the square-ring code below does, every one of those joins is a corner
# square and a level change, and a diagonal run's swept cells are a checkerboard. So a
# path with a diagonal in it is drawn as **one field**: every column near the line is
# given its distance to the polyline and where along it it lies (`u`, in lattice steps),
# and the wall is the columns within its half-width -- a solid, 4-connected band on every
# run and round every join. What each column carries is read off `u`:
#   level -- the walk climbs the ground: each station's own segment floor plus the
#     wall's height, then an upper envelope rising or falling one HALF block a station
#     (a slab course), so the walk never steps more than half a block between neighbours
#     and never drops under its own segment's floor + height; a tower or a gate flattens
#     its reach to one level.
#   body -- from the column's own footing (the site's, or the ground where it is not a
#     site column) to the walk, never over air.
#   parapet -- the outer ring of the band, solid one course over the walk with a merlon
#     every `Builder.merlon(i)` of arc; a crown road carries one on each edge.
#   towers -- a round bastion every `every` blocks of ARC LENGTH, projecting past the
#     outer face, flat-topped at the walk with a drum and a crenellated roof; the walk
#     runs through the drum.
#   ways down -- a square stair turret against the inner face at every `turret_nth`
#     tower, a switchback climbing round its core from a door at the inner ground to the
#     walk.
# ======================================================================================

_CW_EPS = 1e-6
#: Stations past a run's own ends it draws as well, so the seam comes out whole
#: whichever run is built last (see `_cw_build`).
SEAM_REACH = 8


def _cw_sgn(v):
    return (v > 0) - (v < 0)


def _cw_band_k(half):
    """The widest perpendicular diagonal offset a band of half-width `half` covers."""
    return max(1, int(math.floor((half + 0.5) * math.sqrt(2.0) - _CW_EPS)))


def _cw_stations(segs, closed):
    st = []            # (x, z, seg, arc)
    seg_u0 = []
    arc = 0.0
    for si, s in enumerate(segs):
        ax, az = int(s["a"][0]), int(s["a"][1])
        bx, bz = int(s["b"][0]), int(s["b"][1])
        n = max(abs(bx - ax), abs(bz - az))
        dx, dz = _cw_sgn(bx - ax), _cw_sgn(bz - az)
        ls = math.sqrt(2.0) if (dx and dz) else 1.0
        seg_u0.append(len(st))
        for k in range(n):
            st.append((ax + dx * k, az + dz * k, si, arc))
            arc += ls
    if not closed:
        s = segs[-1]
        st.append((int(s["b"][0]), int(s["b"][1]), len(segs) - 1, arc))
    return st, seg_u0, arc


def _cw_field(segs, seg_u0, reach, inward_sign):
    """Every column within `reach` of the polyline: (x, z) -> [d, u, o]. `d` the
    distance to the line, `u` the station coordinate of the nearest point, `o` the
    signed offset, positive outward."""
    f = {}
    for si, s in enumerate(segs):
        ax, az = float(s["a"][0]), float(s["a"][1])
        bx, bz = float(s["b"][0]), float(s["b"][1])
        L = math.hypot(bx - ax, bz - az)
        if L < _CW_EPS:
            continue
        ex, ez = (bx - ax) / L, (bz - az) / L
        ls = math.sqrt(2.0) if (abs(bx - ax) > 0 and abs(bz - az) > 0) else 1.0
        nx, nz = -ez * inward_sign, ex * inward_sign      # inward normal
        R = int(math.ceil(reach)) + 1
        for x in range(int(min(ax, bx)) - R, int(max(ax, bx)) + R + 1):
            for z in range(int(min(az, bz)) - R, int(max(az, bz)) + R + 1):
                vx, vz = x - ax, z - az
                t = vx * ex + vz * ez
                if t < 0.0:
                    t = 0.0
                elif t > L:
                    t = L
                qx, qz = vx - t * ex, vz - t * ez
                d = math.hypot(qx, qz)
                if d > reach:
                    continue
                cur = f.get((x, z))
                if cur is not None and cur[0] <= d + _CW_EPS:
                    continue
                side = -(vx * nx + vz * nz)
                o = d if side >= 0 else -d
                f[(x, z)] = [d, seg_u0[si] + t / ls, o]
    return f


def _cw_envelope(p, closed, flats):
    """Raise `p` (half-blocks per station) to the envelope falling one a station, with
    each flat zone (a list of station indices) held at one even level."""
    n = len(p)
    for _round in range(3):
        for zone in flats:
            m = max(p[i] for i in zone)
            m += m & 1
            for i in zone:
                p[i] = m
        for _ in range(2 if closed else 1):
            for i in range(n):
                j = i - 1
                if j < 0:
                    if not closed:
                        continue
                    j = n - 1
                if p[j] - 1 > p[i]:
                    p[i] = p[j] - 1
            for i in range(n - 1, -1, -1):
                j = i + 1
                if j >= n:
                    if not closed:
                        continue
                    j = 0
                if p[j] - 1 > p[i]:
                    p[i] = p[j] - 1
    return p


def _cw_relax(level, cells):
    """No two 4-adjacent columns of `cells` differ by more than one half-block: the
    lower is raised. Returns the passes it took."""
    work = list(cells)
    passes = 0
    while work and passes < 64:
        passes += 1
        nxt = set()
        for (x, z) in work:
            v = level[(x, z)]
            for q in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                w = level.get(q)
                if w is not None and w < v - 1 and q in cells:
                    level[q] = v - 1
                    nxt.add(q)
        work = nxt
    return passes


def _cw_ring_segments(ring):
    """The segments of a closed vertex list, as `_decide_edge` names them (no cells)."""
    rp = [(int(p[0]), int(p[1])) for p in ring]
    if rp[0] != rp[-1]:
        rp.append(rp[0])
    out = []
    for a, c in zip(rp, rp[1:]):
        if a == c:
            continue
        axis = "d" if (a[0] != c[0] and a[1] != c[1]) else ("x" if a[0] != c[0] else "z")
        out.append({"a": list(a), "b": list(c), "axis": axis})
    return out


def _cw_pb(b, gf, x, y, z, blk):
    """`place_block`, never under the part's floor (the library refuses it)."""
    if y >= gf:
        b.place_block(x, y, z, blk)


def _cw_pc(b, gf, x0, y0, z0, x1, y1, z1, blk):
    """`place_cuboid`, clipped to the part's floor."""
    lo, hi = max(gf, min(y0, y1)), max(y0, y1)
    if hi >= lo:
        b.place_cuboid(x0, lo, z0, x1, hi, z1, blk)


def _cw_fr(b, gf, x0, y0, z0, x1, y1, z1, blk):
    """`fill_region`, clipped to the part's floor."""
    lo, hi = max(gf, min(y0, y1)), max(y0, y1)
    if hi >= lo:
        b.fill_region(x0, lo, z0, x1, hi, z1, blk)


def _cw_build(b, part, seed, H, cfg):
    """Draw a wall along an octilinear polyline as one field. See the banner above.
    `H` is the walk's height over its footing; `cfg` names the materials and the
    wall's word for its face, parapet, crown road, towers and ways down.

    **A run of a ring** (`part["ring_path"]`, the whole closed wall this open run is
    cut from): everything that decides the wall -- the walk's level, the towers, the
    turrets, which column belongs to which run -- is computed on the whole ring, and
    this run builds only the columns nearest its own stations. Two runs built in
    separate processes, in either order, meet in one wall. The ring's floors are
    `part["ring_floors"]` (one per ring segment) where the compiler states them, and
    otherwise the ground under the ring's band (`Builder.edge_band`) as this run reads
    it -- which agrees between runs only if the ground under the whole ring is settled
    before its first run is built. `part["ring_gates"]` ([[x, z], ...]) keeps towers
    off every gate of the ring, not only this run's."""
    own_segs = part["segments"]
    if not own_segs:
        return None
    own_closed = len(own_segs) > 1 and list(own_segs[0]["a"]) == list(own_segs[-1]["b"])
    GF = int(part["floor_y"])
    YMAX = GF + H + 60
    WALL, FOOT, TRIM, FRAME = cfg["wall"], cfg["foot"], cfg["trim"], cfg["frame"]
    PAVE, SLAB = cfg["pave"], cfg["slab"]
    width = max(1, int(cfg.get("width") or part.get("width") or 1))
    half = (width - 1) // 2
    thr = max(half + 0.5 - _CW_EPS, 0.72)
    road = bool(cfg.get("road"))
    par_out = bool(cfg.get("parapet_outside"))
    t_r = float(cfg.get("tower_r", 4.5))
    t_push = float(cfg.get("tower_push", 1.5))
    band_w = max(1, int(part.get("width") or width))

    # ---- the line this wall is part of: its own path, or the whole ring -----------
    mode = "own"
    note = None
    full, closed = own_segs, own_closed
    own_line, _u0, _a = _cw_stations(own_segs, False)
    own_cells = [(p[0], p[1]) for p in own_line]
    if part.get("ring_path") and not own_closed and len(own_cells) >= 2:
        ring = [(int(p[0]), int(p[1])) for p in part["ring_path"]]
        for attempt in (0, 1):
            full_try = _cw_ring_segments(ring)
            st_try, _s, _t = _cw_stations(full_try, True)
            idx = {}
            for i, q in enumerate(st_try):
                idx.setdefault((q[0], q[1]), i)
            i0 = idx.get(own_cells[0])
            i1 = idx.get(own_cells[1])
            if i0 is not None and i1 is not None and (i1 - i0) % len(st_try) == 1:
                full, closed, mode = full_try, True, "ring"
                break
            ring = ring[::-1]
        if mode != "ring":
            note = "ring_path does not run through this run's path; drawn on its own"
    st, seg_u0, arc_total = _cw_stations(full, closed)
    N = len(st)
    if N < 2:
        return None
    if mode == "ring":
        idx = {}
        for i, q in enumerate(st):
            idx.setdefault((q[0], q[1]), i)
        i0 = idx[own_cells[0]]
        n_own = len(own_cells) - 1
        own_st = set((i0 + k) % N for k in range(n_own))
        # **the seam is drawn by both runs**: a run's `site()` fills and cuts its own
        # band, which overlaps its neighbour's round the shared vertex, so each run
        # draws the columns nearest SEAM_REACH stations past its ends as well -- the
        # same blocks either way, since everything they are drawn from is the ring's --
        # and whichever run is built last leaves the seam as the other would have
        ext_st = set((i0 + k) % N for k in range(-SEAM_REACH, n_own + SEAM_REACH))
    else:
        i0 = 0
        n_own = N
        own_st = set(range(N))
        ext_st = own_st

    # which side is in: a closed line by its winding, an open one by its centroid
    if closed:
        area = 0.0
        for s in full:
            area += s["a"][0] * s["b"][1] - s["b"][0] * s["a"][1]
        inward_sign = 1 if area > 0 else -1
    else:
        cx = sum(p[0] for p in st) / float(N)
        cz = sum(p[1] for p in st) / float(N)
        s0 = full[len(full) // 2]
        ex, ez = _cw_sgn(s0["b"][0] - s0["a"][0]), _cw_sgn(s0["b"][1] - s0["a"][1])
        mx, mz = (s0["a"][0] + s0["b"][0]) / 2.0, (s0["a"][1] + s0["b"][1]) / 2.0
        inward_sign = 1 if ((cx - mx) * -ez + (cz - mz) * ex) >= 0 else -1

    def station_of(u):
        i = int(math.floor(u + 0.5))
        if closed:
            return i % N
        return min(N - 1, max(0, i))

    def sdist(i, j):
        d = abs(i - j)
        return min(d, N - d) if closed else d

    # ---- the window of the line this run has to know: its own stations and a reach
    reach_st = int(t_r) + 16
    if mode == "ring":
        win = set((i0 + k) % N for k in range(-reach_st, n_own + reach_st))
    else:
        win = set(range(N))
    win_segs = []
    for si, s in enumerate(full):
        n = max(abs(s["b"][0] - s["a"][0]), abs(s["b"][1] - s["a"][1]))
        if any(((seg_u0[si] + k) % N if closed else seg_u0[si] + k) in win
               for k in range(n + 1)):
            win_segs.append(si)
    f = _cw_field([full[si] for si in win_segs], [seg_u0[si] for si in win_segs],
                  thr + 2.0, inward_sign)

    def mine(c):
        v = f.get(c)
        return v is not None and station_of(v[1]) in ext_st

    if width == 1:
        # one column on an axial run; on a diagonal, the spine and the lattice line just
        # outside it, which is the thinnest band a person cannot see through
        body = set(c for c, v in f.items() if v[0] < 0.5 or 0.0 < v[2] <= 0.72)
    else:
        body = set(c for c, v in f.items() if v[0] <= thr)
    ring_out = set(c for c, v in f.items() if c not in body and v[2] > 0 and v[0] <= thr + 1.0)
    ring_in = set(c for c, v in f.items() if c not in body and v[2] < 0 and v[0] <= thr + 1.0)

    def nb4(c):
        x, z = c
        return ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1))

    face_out = set(c for c in body if any(q in ring_out for q in nb4(c)))
    face_in = set(c for c in body if any(q in ring_in for q in nb4(c)))

    def normal_at(i):
        """The inward unit normal at station i (from its segment)."""
        s = full[st[i][2]]
        ex, ez = _cw_sgn(s["b"][0] - s["a"][0]), _cw_sgn(s["b"][1] - s["a"][1])
        L = math.hypot(ex, ez) or 1.0
        return (-ez * inward_sign / L, ex * inward_sign / L)

    # ---- the floors: each station's segment ----------------------------------------
    bedc = {}

    def bed(x, z):
        v = bedc.get((x, z))
        if v is None:
            v = int(b.bed(x, z))
            bedc[(x, z)] = v
        return v

    if mode == "ring":
        rf = part.get("ring_floors")
        if rf is not None and len(rf) == len(full):
            seg_floor = [int(v) for v in rf]
        else:
            seg_floor = []
            for s in full:
                cells = b.edge_band(s["a"], s["b"], band_w)
                seg_floor.append(max(bed(c[0], c[1]) for c in cells))
    else:
        seg_floor = [int(s["floor_y"]) for s in full]
    F = [seg_floor[st[i][2]] for i in range(N)]
    if mode == "own" and not closed and N > 2 * (SEAM_REACH + 4):
        # **An open run's ends stand on its neighbour.** Without the ring, a run's end
        # segments are sited on whatever is there, and beside a run already built that
        # is the other run's wall: the site read a floor of 133 and 182 under a palace
        # wall laid at 83, and the walk climbed a half-block a station to meet it -- a
        # cliff of eighty blocks. The ends take the floor of the first station past
        # them; a ring's run (`ring_floors`) never needs this.
        k = SEAM_REACH + 4
        for i in range(k):
            F[i] = F[k]
            F[N - 1 - i] = F[N - 1 - k]

    # ---- gates, towers and turrets: where along the line ------------------------ **A
    # gate's pad is the gate's.** The wall leaves the pad of every gate it knows of
    # unbuilt -- `part["gates"]` (siting's `annotate_gates`) and `part["ring_gates"]`
    # (every gate of the ring, `[x, z]` or `{"at", "size"}`) -- so the gate opens the
    # wall whichever of the two is built first, and a run that draws its neighbour's
    # seam never closes a gate standing there. A pad's size where none is said is the
    # one `Builder.point_pad` gives a gate on a wall this high.
    gh = int(cfg.get("gate_height") or H)
    dflt = max(5, min(15, gh // 3))
    dflt -= 0 if dflt % 2 else 1
    gates_known = []
    for g in list(part.get("gates") or []) + list(part.get("ring_gates") or []):
        if isinstance(g, dict):
            gates_known.append(((int(g["at"][0]), int(g["at"][-1])),
                                int(g.get("size") or dflt)))
        else:
            gates_known.append(((int(g[0]), int(g[-1])), dflt))
    gate_cells = [q for q, _n in gates_known]
    gate_pads = set()
    for (gx, gz), n in gates_known:
        hh = n // 2
        for x in range(gx - hh, gx + hh + 1):
            for z in range(gz - hh, gz + hh + 1):
                gate_pads.add((x, z))
    gate_zone = set()
    for (gx, gz) in gate_cells:
        for i in range(N):
            if max(abs(st[i][0] - gx), abs(st[i][1] - gz)) <= 12:
                gate_zone.add(i)

    every = float(cfg.get("tower_every", 56))
    keep_off = int(t_r) + 6
    towers = []
    if every > 0 and arc_total > 2 * every:
        nt = max(3, int(round(arc_total / every))) if closed else \
            max(1, int(arc_total // every))
        stepa = arc_total / float(nt)
        off = stepa / 2.0
        arcs = [st[i][3] for i in range(N)]
        j = 0
        for k in range(nt):
            want = off + k * stepa
            while j < N - 1 and arcs[j] < want:
                j += 1
            i = j
            shift = 0
            while shift < N and any(((i + d) % N) in gate_zone
                                    for d in range(-keep_off, keep_off + 1)):
                shift += 1
                i = (j + shift) % N
            if not closed and (i < keep_off or i > N - 1 - keep_off):
                continue
            if towers and min(sdist(i, t) for t in towers) < every / 3.0:
                continue
            towers.append(i)
    nth = int(cfg.get("turret_nth", 2))
    turret_st = []
    if nth > 0:
        toff = int(t_r) + 5
        for ti, i in enumerate(towers):
            if ti % nth:
                continue
            j = (i + toff) % N if closed else min(N - 1, i + toff)
            if any(((j + d) % N) in gate_zone for d in range(-6, 7)):
                continue
            if not closed and (j < 6 or j > N - 7):
                continue
            turret_st.append(j)
    flats = []
    for i in towers:
        flats.append([k % N if closed else k for k in range(i - int(t_r) - 2, i + int(t_r) + 3)
                      if closed or 0 <= k < N])
    for j in turret_st:
        flats.append([k % N if closed else k for k in range(j - 5, j + 6)
                      if closed or 0 <= k < N])

    tower_disc = {}         # tower station -> (cx, cz, cells)
    tower_of = {}           # cell -> tower station
    # **A tower stands on ground.** It projects past the outer face where the ground
    # there is within a block of the wall's own floor; where it is lower -- a terrace
    # the wall stands at the top of, a ditch, a shore -- the tower is drawn back inward
    # until every column of it outside the band has ground under it (a column can be
    # built only down to the part's floor), so no bastion hangs over a drop.
    def disc_at(i, push):
        nx, nz = normal_at(i)
        cxf = st[i][0] - nx * push
        czf = st[i][1] - nz * push
        cells = set()
        R = int(math.ceil(t_r)) + 1
        for x in range(int(cxf) - R, int(cxf) + R + 1):
            for z in range(int(czf) - R, int(czf) + R + 1):
                if math.hypot(x - cxf, z - czf) <= t_r:
                    cells.add((x, z))
        return cxf, czf, cells

    def sound(i, cells):
        # the lowest floor of the wall about this tower: a column is laid down to the
        # wall's floor and no further, so its ground must be within one of it
        reach = int(t_r) + 4
        lo = min(F[(i + d) % N if closed else min(N - 1, max(0, i + d))]
                 for d in range(-reach, reach + 1))
        return all(c in body or bed(c[0], c[1]) >= lo - 1 for c in cells)
    towers_drawn_in = 0
    towers_unsound = []
    for i in list(towers):
        if i not in win:
            continue
        push = t_push
        cxf, czf, cells = disc_at(i, push)
        while not sound(i, cells) and push > -t_r:
            push -= 1.0
            cxf, czf, cells = disc_at(i, push)
        if not sound(i, cells):
            # nowhere across the wall is there ground for it: no tower here, and the
            # wall runs on through (the same answer in every run that draws it)
            towers_unsound.append(i)
            continue
        if push < t_push:
            towers_drawn_in += 1
        tower_disc[i] = (cxf, czf, cells)
        for c in cells:
            tower_of[c] = i
    ring_planned = len(towers)
    towers = [i for i in towers if i not in towers_unsound]

    def tower_cell_mine(c):
        if c in f:
            return mine(c)
        return tower_of[c] in ext_st

    turret_geo = {}
    for j in turret_st:
        if j in win:
            turret_geo[j] = _cw_turret_cells(st, j, normal_at(j), half)
    foreign_turret = set()
    for j, (cxy, cells) in turret_geo.items():
        if j not in own_st:
            foreign_turret.update(cells)

    # ---- the walk's level, in half-blocks -------------------------------------------
    p = [2 * (F[i] + H) for i in range(N)]
    p = _cw_envelope(p, closed, flats)
    level = {}
    walkcells = body | (ring_out if par_out else set())
    for c in walkcells:
        level[c] = p[station_of(f[c][1])]
    for i, (cxf, czf, cells) in tower_disc.items():
        m = p[i]
        for c in cells:
            level[c] = m
    relax_passes = _cw_relax(level, set(level))

    # ---- ground
    # -----------------------------------------------------------------------
    floor_of = {}
    for s in own_segs:
        for c in s.get("cells") or []:
            k = (int(c[0]), int(c[1]))
            floor_of[k] = max(floor_of.get(k, -1 << 30), int(s["floor_y"]))

    def base_of(c):
        if mode == "ring":
            # a run of a ring lays its courses from the ground block itself, so a
            # neighbour's `site()` stripping the turf off a seam column cannot leave a
            # hole under a course this run laid
            return max(GF, min(bed(c[0], c[1]), floor_of.get(c, 1 << 30) + 1))
        if c in floor_of:
            return floor_of[c] + 1
        return max(GF, bed(c[0], c[1]) + 1)

    # the clearing's ceiling reaches over any footing the site laid, however high
    YMAX = max(YMAX, max(floor_of.values(), default=YMAX) + 6)
    top_of = {}
    floating = [0]
    floating_at = {}

    def col_top(c, y):
        if y > top_of.get(c, -1 << 30):
            top_of[c] = y

    face = cfg.get("face", "framed")
    pier = int(cfg.get("pier_every", 5))
    band_every = int(cfg.get("band_every", 8))
    n_foot = int(cfg.get("foot_courses", 2))
    string_mid = bool(cfg.get("string_mid"))
    crenel = bool(cfg.get("crenel", True))

    def merlon_at(u):
        return (not crenel) or b.merlon(int(math.floor(u + 0.5)))

    def footing(c, W, upto):
        x, z = c
        bs = base_of(c)
        if bs > W:
            bs = W
        if c not in floor_of and bed(x, z) + 1 < GF:
            floating[0] += 1
            where = ("tower" if c in tower_of else "band" if c in body else
                     "parapet" if c in ring_out else "other")
            floating_at[where] = floating_at.get(where, 0) + 1
        ftop = min(bs + n_foot - 1, upto)
        if ftop >= bs:
            _cw_pc(b, GF, x, bs, z, x, ftop, z, FOOT)
        if ftop + 1 <= upto:
            _cw_pc(b, GF, x, ftop + 1, z, x, upto, z, WALL)
        return bs

    def body_column(c, lv, is_face, u):
        x, z = c
        W = lv // 2
        bs = footing(c, W, W - 1)
        if is_face and W - 1 > bs:
            i = int(math.floor(u + 0.5))
            if face == "framed":
                if i % pier == 0:
                    _cw_pc(b, GF, x, bs + n_foot, z, x, W - 2, z, FRAME)
                else:
                    y = W - 1 - band_every
                    while y > bs + n_foot:
                        _cw_pb(b, GF, x, y, z, FRAME)
                        y -= band_every
            elif face in ("banded", "masonry"):
                with b.figure("wall_string_course"):
                    y = W - 1 - band_every
                    while y > bs + n_foot:
                        _cw_pb(b, GF, x, y, z, TRIM)
                        y -= band_every
            elif face == "piers":
                if i % pier == 0:
                    _cw_pc(b, GF, x, bs + n_foot, z, x, W - 2, z, TRIM)
                elif string_mid:
                    y = W - 1 - max(3, (W - bs) // 2)
                    if y > bs + n_foot:
                        with b.figure("wall_string_course"):
                            _cw_pb(b, GF, x, y, z, TRIM)
            with b.figure("wall_cornice"):
                _cw_pb(b, GF, x, W - 1, z, TRIM)
        _cw_pb(b, GF, x, W, z, PAVE)
        col_top(c, W)
        return W

    def parapet_on(c, lv, u, y_from):
        x, z = c
        PT = (lv + 1) // 2 + 1
        if PT >= y_from:
            _cw_pc(b, GF, x, y_from, z, x, PT, z, WALL)
        col_top(c, PT)
        if merlon_at(u):
            _cw_pb(b, GF, x, PT + 1, z, WALL)
            col_top(c, PT + 1)

    def skip(c):
        return (not mine(c)) or c in tower_of or c in foreign_turret or c in gate_pads

    # ---- the band
    # --------------------------------------------------------------------- the walk:
    # the band less its parapets (towers' cells included -- the walk runs through a
    # tower's drum and its doors are where it crosses the rim)
    walk = set(c for c in body
               if not ((not par_out and c in face_out) or (road and c in face_in)))
    for c in body:
        if skip(c):
            continue
        d, u, o = f[c]
        lv = level[c]
        is_face = c in face_out or c in face_in
        W = body_column(c, lv, is_face, u)
        if c not in walk:
            parapet_on(c, lv, u, W + 1)
        elif lv & 1:
            _cw_pb(b, GF, c[0], W + 1, c[1], SLAB)
            col_top(c, W + 1)
    if par_out:
        for c in ring_out:
            if skip(c):
                continue
            d, u, o = f[c]
            lv = level[c]
            x, z = c
            W = lv // 2
            bs = footing(c, W, W)
            if face == "piers" and int(math.floor(u + 0.5)) % pier == 0 and W - 2 > bs:
                _cw_pc(b, GF, x, bs + n_foot, z, x, W - 2, z, TRIM)
            parapet_on(c, lv, u, W + 1)
    else:
        # the corbel course under the parapet's outer face, a block proud
        for c in ring_out:
            if skip(c) or c in body:
                continue
            d, u, o = f[c]
            W = p[station_of(u)] // 2
            if bed(c[0], c[1]) >= W - 1:
                continue
            _cw_pb(b, GF, c[0], W, c[1], TRIM)
            col_top(c, W)

    # ---- towers ----------------------------------------------------------------------
    drum = int(cfg.get("drum", 4))
    for i, (cxf, czf, cells) in tower_disc.items():
        lv = p[i]
        W = lv // 2
        rim = set(c for c in cells if any(q not in cells for q in nb4(c)))
        # the way the walk comes in: rim cells the walk crosses
        doors = set(c for c in rim if c in walk)
        for c in cells:
            if not tower_cell_mine(c) or c in foreign_turret or c in gate_pads:
                continue
            x, z = c
            footing(c, W, W - 1)
            if c in rim:
                with b.figure("wall_cornice"):
                    _cw_pb(b, GF, x, W - 1, z, TRIM)
            _cw_pb(b, GF, x, W, z, PAVE)
            col_top(c, W)
            if c in rim:
                if c in doors:
                    _cw_pc(b, GF, x, W + 1, z, x, W + 3, z, "air")
                    _cw_pc(b, GF, x, W + 4, z, x, W + drum, z, WALL)
                    _cw_pb(b, GF, x, W + 4, z, TRIM)
                else:
                    _cw_pc(b, GF, x, W + 1, z, x, W + drum, z, WALL)
                    ang = math.atan2(z - czf, x - cxf)
                    if int(round(ang * 4)) % 3 == 0 and drum >= 4:
                        _cw_pb(b, GF, x, W + 2, z, "air")      # a loop
            else:
                _cw_pc(b, GF, x, W + 1, z, x, W + drum, z, "air")
            _cw_pb(b, GF, x, W + drum + 1, z, PAVE if c not in rim else TRIM)
            col_top(c, W + drum + 1)
            if c in rim:
                ang = math.atan2(z - czf, x - cxf)
                if not crenel or int(math.floor((ang + math.pi) * t_r + 0.5)) % 3 != 2:
                    _cw_pb(b, GF, x, W + drum + 2, z, WALL)
                    col_top(c, W + drum + 2)

    # ---- ways down: a stair turret against the inner face, built by its run ---------
    turrets = []
    for j in turret_st:
        if j not in own_st:
            continue
        if mode == "ring" and min(sdist(j, i0), sdist(j, (i0 + n_own) % N)) < SEAM_REACH + 6:
            # a turret is one run's, and one close to a seam would stand in the
            # neighbour's band, which that run's `site()` clears
            turrets.append({"ok": False, "at": list(turret_geo[j][0]),
                            "reason": "too near a seam between runs"})
            continue
        turrets.append(_cw_turret(b, cfg, turret_geo[j], level, body, tower_of, walk,
                                  st, j, normal_at(j), p, bed, base_of, GF, col_top))

    # ---- a closed wall with no gate standing on it cuts itself one way through -------
    cut_gate = None
    if own_closed and not part.get("gates"):
        best = None
        for si, s in enumerate(full):
            if s["axis"] == "d":
                continue
            L = max(abs(s["b"][0] - s["a"][0]), abs(s["b"][1] - s["a"][1]))
            if L >= 9 and (best is None or L > best[0]):
                best = (L, si)
        if best is not None:
            si = best[1]
            ui = seg_u0[si] + best[0] // 2
            Fg = F[ui]
            Wg = p[ui] // 2
            if Wg - Fg >= 6:
                for c, v in f.items():
                    if abs(v[1] - ui) <= 1.01 and (c in body or c in ring_out
                                                   or c in ring_in):
                        _cw_pc(b, GF, c[0], Fg + 1, c[1], c[0], Fg + 4, c[1], "air")
                        _cw_pb(b, GF, c[0], Fg + 5, c[1], TRIM)
                cut_gate = [st[ui][0], st[ui][1]]

    # ---- the wall cuts what stood over it; nothing hangs --------------------------
    for c, y0 in top_of.items():
        x, z = c
        cr = int(b.crest(x, z))
        # ...but a column buried in a bank keeps the bank: only what stands on it goes.
        # A column `site()` laid a footing in is cut from this wall's own top: the
        # site's fill over it (a floor read off a neighbour's wall) is not a bank.
        ys = (y0 + 1) if c in floor_of else max(y0, bed(x, z)) + 1
        y1 = min(YMAX, max(cr, y0, floor_of.get(c, -1 << 30)) + 2)
        if y1 >= ys:
            _cw_fr(b, GF, x, ys, z, x, y1, z, "air")

    steps = 0
    for c in walk:
        v = f.get(c)
        if v is None or station_of(v[1]) not in own_st:
            continue
        for q in nb4(c):
            if q in walk:
                steps = max(steps, abs(level[c] - level[q]))
    own_towers = [i for i in towers if i in own_st]
    return {"curved": True, "mode": mode, "note": note, "columns": len(top_of),
            "body": len(body), "walk": len(walk), "towers": len(own_towers),
            "turrets": turrets,
            "tower_arc": [round(st[i][3], 1) for i in own_towers],
            "ring_towers": ring_planned,
            "arc": round(arc_total, 1), "stations": N, "own_stations": len(own_st),
            "segments": len(full),
            "max_walk_step_half": steps, "relax_passes": relax_passes,
            "floating_columns": floating[0], "top": max(top_of.values()) if top_of else None,
            "towers_drawn_in": towers_drawn_in, "floating_at": floating_at,
            "towers_without_ground": sum(1 for i in towers_unsound if i in own_st),
            "walk_levels": [[st[i][0], st[i][1], p[i]] for i in sorted(own_st)],
            "gate_cut": cut_gate, "closed": closed}


def _cw_turret_cells(st, i, nrm, half):
    """A stair turret's centre and 7x7 cells, against the inner face at station i."""
    nx, nz = nrm
    D = half + 4.0
    cx = int(round(st[i][0] + nx * D))
    cz = int(round(st[i][1] + nz * D))
    return (cx, cz), set((cx + a, cz + c) for a in range(-3, 4) for c in range(-3, 4))


def _cw_turret(b, cfg, geo, level, body, tower_of, walk, st, i, nrm, p, bed,
               base_of, GF, col_top):
    """A 7x7 stair turret against the inner face at station i: a switchback round a
    3x3 core, from a door at the inner ground to the walk."""
    WALL, FOOT, TRIM, PAVE = cfg["wall"], cfg["foot"], cfg["trim"], cfg["pave"]
    stepmat = cfg["stepmat"]
    nx, nz = nrm
    (cx, cz), cellset = geo
    cells = sorted(cellset)
    if any(c in tower_of for c in cells):
        return {"ok": False, "at": [cx, cz], "reason": "a tower stands there"}
    lv = p[i]
    lv += lv & 1
    W = lv // 2
    # the door: the middle of the side facing furthest inward
    sides = {(1, 0): nx, (-1, 0): -nx, (0, 1): nz, (0, -1): -nz}
    ddir = max(sides, key=lambda k: sides[k])
    ext = (cx + ddir[0] * 4, cz + ddir[1] * 4)
    y0 = max(GF, bed(ext[0], ext[1]))
    if W - y0 < 4:
        return {"ok": False, "at": [cx, cz], "reason": f"a drop of {W - y0} needs none"}
    # the ring round the core, in order, starting at the door side's middle
    ring = []
    for a in range(-2, 3):
        ring.append((a, -2))
    for c in range(-1, 3):
        ring.append((2, c))
    for a in range(1, -3, -1):
        ring.append((a, 2))
    for c in range(1, -2, -1):
        ring.append((-2, c))
    start = ring.index((ddir[0] * 2, ddir[1] * 2))
    ring = ring[start:] + ring[:start]
    corner = set([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    # levels round the spiral: flat at the corners, a rise of one on each side cell
    seq = []
    y = y0
    k = 0
    while True:
        a, c = ring[k % 16]
        if k > 0 and (a, c) not in corner:
            y += 1
        seq.append(((cx + a, cz + c), y, k > 0 and (a, c) not in corner))
        if y >= W:
            break
        k += 1
        if k > 16 * 8:
            return {"ok": False, "at": [cx, cz], "reason": "too tall a turret"}
    # the solid turret, to the walk
    for c in cells:
        x, z = c
        bs = base_of(c)
        if bs > W:
            bs = W
        _cw_pc(b, GF, x, bs, z, x, min(W - 1, bs + 1), z, FOOT)
        if bs + 2 <= W - 1:
            _cw_pc(b, GF, x, bs + 2, z, x, W - 1, z, WALL)
        _cw_pb(b, GF, x, W, z, PAVE)
        col_top(c, W)
        level[c] = 2 * W
    # the parapet round the turret's top, open where it meets the walk
    for c in cells:
        x, z = c
        if abs(x - cx) == 3 or abs(z - cz) == 3:
            if any(q in body or q in walk for q in ((x + 1, z), (x - 1, z),
                                                    (x, z + 1), (x, z - 1))):
                continue
            _cw_pb(b, GF, x, W + 1, z, WALL)
            col_top(c, W + 1)
            with b.figure("wall_cornice"):
                _cw_pb(b, GF, x, W - 1, z, TRIM)
    # the door and its passage through the shell
    for dd in (3,):
        x, z = cx + ddir[0] * dd, cz + ddir[1] * dd
        _cw_pb(b, GF, x, y0, z, FOOT)
        _cw_pc(b, GF, x, y0 + 1, z, x, y0 + 3, z, "air")
    # the stairwell: each cell of the spiral clear three over its own tread
    treads = []
    for (c, y, tread) in seq:
        x, z = c
        _cw_pc(b, GF, x, y + 1, z, x, min(y + 3, W + 3), z, "air")
    for idx, (c, y, tread) in enumerate(seq):
        x, z = c
        if tread and y <= W:
            treads.append((x, y, z))
        else:
            _cw_pb(b, GF, x, y, z, PAVE)
    # the flights: one steps() call per straight run, so each run faces up its own way
    run = []
    for (x, y, z) in treads:
        if run and (abs(x - run[-1][0]) + abs(z - run[-1][2]) != 1
                    or (len(run) >= 2 and (x - run[-1][0], z - run[-1][2])
                        != (run[-1][0] - run[-2][0], run[-1][2] - run[-2][2]))):
            b.steps(run, stepmat)
            run = []
        run.append((x, y, z))
    if run:
        b.steps(run, stepmat)
    return {"ok": True, "at": [cx, cz], "door": [cx + ddir[0] * 4, cz + ddir[1] * 4],
            "from": y0, "to": W, "treads": len(treads)}


#: A tower on a curved wall every this many blocks of arc (the curved engine).
TOWER_EVERY = 64

#: A path drawn by the curved engine: any 45-degree run, or more runs than a square
#: ring's handful (an octilinear outline's hundreds of short runs).
CURVED_FROM_SEGMENTS = 24


def _curved_path(part):
    """A part's path (before siting) with a 45-degree run in it, or its segments."""
    if part.get("segments"):
        return _curved(part)
    pts = part.get("path") or []
    runs = list(zip(pts, pts[1:]))
    return any(a[0] != b[0] and a[1] != b[1] for a, b in runs) or \
        len(runs) > CURVED_FROM_SEGMENTS


def _curved(part):
    segs = part.get("segments") or []
    return any(s.get("axis") == "d" for s in segs) or len(segs) > CURVED_FROM_SEGMENTS


def build(b, part, seed, **params):
    H = int(params.get("height", 36))
    H = max(24, min(48, H))
    crenel = params.get("parapet", "crenellated") == "crenellated"
    # The face is the parameter where the plan wrote one and the **part's** word
    # otherwise: the layout stamps `face` on every edge of a place whose walls the
    # sentence calls unbroken, at every level, and a wall inside such a place is one of
    # them whether the call that drew it thought to say so.
    face = params.get("face") or part.get("face") or "framed"
    if face not in ("framed", "banded", "plain", "masonry", "unbroken"):
        face = "framed"
    # **A plain face carries sparse stairs.** With the default `every`, stairs at a
    # house's scale read as red diagonal bracing painted on the wall rather than as a
    # way up. A face with no articulation on it has nothing for a switchback every 26
    # columns to belong to, so the ways down stand at the corners and beside the gates,
    # where a person actually climbs.
    stairs = "sparse" if face in ("unbroken", "plain", "masonry") else "every"
    if face == "unbroken":
        # **An unbroken wall is dressed masonry on both faces**: `plain` is forty-eight
        # blocks of one flat plane and reads as a render.
        face = "masonry"
    rng = random.Random(seed)
    v = part["voice"]
    WALL = b.block(v["wall"], "full")
    FOOT = b.block(v["footing"], "full")
    FRAME = b.block(v["frame"], "full")
    FLOOR = b.block(v["floor"], "full")
    TRIM = b.block(v["trim"], "full")
    foot_fam = v["footing"]

    segs = part["segments"]
    closed = len(segs) > 1 and list(segs[0]["a"]) == list(segs[-1]["b"])
    frs = _frames(segs, 1)
    if not closed and _pick_side(b, frs) < 0:
        frs = _frames(segs, -1)

    P = rng.choice([5, 6])
    phase = rng.randrange(P)
    s_top = rng.choice([1, -1])
    mer_phase = rng.randrange(2)
    band = rng.choice([7, 8, 9])
    if _curved(part):
        # **A round wall is one field**: see the curved engine above. The square-ring
        # code below serves a path of axial runs only.
        width = max(1, int(part.get("width") or 3))
        half = (width - 1) // 2
        return _cw_build(b, part, seed, H, {
            "wall": WALL, "foot": FOOT, "trim": TRIM, "frame": FRAME, "pave": FLOOR,
            "slab": b.block(v["floor"], "slab") + "[type=bottom]", "stepmat": foot_fam,
            "face": face, "pier_every": P,
            "band_every": STRING_EVERY if face == "masonry" else band,
            "foot_courses": PLINTH_H if face == "masonry" else 2,
            "crenel": crenel, "road": width >= CROWN_ROAD_MIN, "parapet_outside": False,
            "tower_every": TOWER_EVERY, "tower_r": max(4.5, half + 3.5),
            "tower_push": 1.5 + half * 0.5, "drum": 6, "turret_nth": 2,
            "gate_height": H})

    # **The mass the place declared.** The swept line siting hands this type is the
    # wall's own thickness, and what it buys is a crown wide enough to walk along with a
    # parapet on both edges and a stair in the wall's own body.
    width = max(1, int(part.get("width") or len(frs[0]["offs"]) if frs else 1))
    if frs:
        width = len(frs[0]["offs"])
    road = width >= CROWN_ROAD_MIN
    batter = face == "masonry"

    cells = {}
    _register_segments(frs, H, cells, road=road, batter=batter)
    zones = _corners(frs, segs, closed, H, cells)
    # A stair block never stands in a gate's pad: siting hands this wall the points on
    # it (`part["gates"]`), and a flight the gate's pad is then quarried through is
    # treads facing down a cut. Each gate is a forbidden run of the segment it is on.
    for g in (part.get("gates") or []):
        gx, gz = int(g["at"][0]), int(g["at"][-1])
        half = int(g.get("size", 5)) // 2 + 3
        for i, fr in enumerate(frs):
            if (gx, gz) in set(fr["cells"]):
                t, _o = _to(fr, gx, gz)
                zones[i].append((t - half, t + half))
                break
    ramp = {}
    flights = []
    ranges = _place_stairs(b, frs, H, zones, cells, rng, s_top, ramp, flights,
                           stairs=stairs, closed=closed, gates=part.get("gates"),
                           width=width)
    # A closed loop cuts itself one way through -- unless a gate stands on it, in which
    # case the gate's own pad is the opening and this wall cuts no second one. Siting
    # says which: `part["gates"]` is the points standing on this edge.
    if closed and not part.get("gates"):
        _gate(frs, zones, ranges, cells, rng)

    # The body of the wall, column by column.
    for (x, z), rec in cells.items():
        F, te = rec["F"], rec["teff"]
        solid = te - 1 if rec["tread"] else te
        if rec["gate"]:
            b.place_cuboid(x, F + 5, z, x, solid, z, WALL)
            b.place_block(x, F + 5, z, FRAME)
        else:
            b.place_cuboid(x, F + 1, z, x, F + 2, z, FOOT)
            b.place_cuboid(x, F + 3, z, x, solid, z, WALL)
            if rec["jamb"]:
                b.place_cuboid(x, F + 1, z, x, F + 5, z, FRAME)
        if rec["face"] and not rec["gate"]:
            if face == "framed":
                if rec["post"] or (rec["t"] + phase) % P == 0:
                    b.place_cuboid(x, F + 3, z, x, te - 2, z, FRAME)
                else:
                    y = F + 3 + band
                    while y <= te - 3:
                        b.place_block(x, y, z, FRAME)
                        y += band
            elif face == "banded":
                # coursed masonry: a string course in trim every `band` courses and at
                # the corners, no posts
                with b.figure("wall_string_course"):
                    y = F + 3 + band
                    while y <= te - 3:
                        b.place_block(x, y, z, TRIM)
                        y += band
            elif face == "masonry":
                # **dressed stonework**: a plinth at the foot in
                # the footing family, a string course of trim every `STRING_EVERY`, and
                # a buttress pier of the wall family standing proud of the face at
                # `PIER_EVERY`. The batter is in the cell's own top, above.
                b.place_cuboid(x, F + 1, z, x, min(F + PLINTH_H, te - 2), z, FOOT)
                # **The string course is a figure and the plinth is not.** A weathering
                # pass may streak this face, but a horizontal line drawn every
                # `STRING_EVERY` courses is the one thing on a monumental wall that must
                # stay unbroken, because it is what says the wall is dressed rather than
                # heaped. The plinth below it is deliberately left editable: an irregular
                # damp base course on the foot of a wall reads as weathering. So:
                # protect the line, let the mass age.
                with b.figure("wall_string_course"):
                    y = F + PLINTH_H + STRING_EVERY
                    while y <= te - 3:
                        b.place_block(x, y, z, TRIM)
                        y += STRING_EVERY
                # a buttress belongs on a mass: on a wall too thin to carry a walk a
                # pier standing proud of the face closes a pocket behind the parapet
                # that nothing can walk into (E003 at width 2, found by the sweep)
                if width >= CROWN_ROAD_MIN and not rec["corner"] \
                        and (rec["t"] + phase) % PIER_EVERY == 0:
                    for (px, pz) in rec["corb"]:
                        if (px, pz) not in cells:
                            b.place_cuboid(px, F + 1, pz, px, te - 2, pz, WALL)
            # plain: one unbroken face, the cornice alone -- and the cornice is the same
            # kind of line as the string course, so it is declared the same way
            with b.figure("wall_cornice"):
                b.place_block(x, te - 1, z, TRIM)
        if rec["par"]:
            b.place_block(x, te + 1, z, WALL)
            if crenel and b.merlon(rec["t"] + mer_phase):
                b.place_block(x, te + 2, z, WALL)
            for (cx, cz) in rec["corb"]:
                if (cx, cz) not in cells:
                    b.place_block(cx, te, cz, TRIM)
        elif not rec["tread"]:
            b.place_block(x, te, z, FLOOR)

    # Tread ramps on the walkway where a corner steps between two levels.
    groups = {}
    for (x, z), rec in cells.items():
        if rec["tread"]:
            key = (rec["ramp"], rec["seg"], rec["o"])
            groups.setdefault(key, []).append((rec["t"], x, rec["teff"], z))
    for key, lst in groups.items():
        lst.sort()
        if frs[key[1]]["axis"] == "d":
            # a ramp on a diagonal walk is laid solid to the tread's level rather than
            # as treads, which have no diagonal facing
            for (_, x, y, z) in lst:
                b.place_block(x, y, z, FLOOR)
            continue
        b.steps([(x, y, z) for (_, x, y, z) in lst], foot_fam,
                axis=frs[key[1]]["axis"])

    # The stair blocks against the inner face: solid to the tread, then treads.
    for (x, z), r in ramp.items():
        F, s = r["F"], r["solid"]
        b.place_cuboid(x, F, z, x, min(F + 2, s), z, FOOT)
        if s >= F + 3:
            b.place_cuboid(x, F + 3, z, x, s, z, WALL)
        if r["tread"] is None:
            b.place_block(x, s, z, FLOOR)
    for fl, axis in flights:
        if fl:
            b.steps(fl, foot_fam, axis=axis)
    return {"stairs": len(flights), "stair_blocks": sum(len(r) for r in ranges),
            "closed": closed}
