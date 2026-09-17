import math
import random

KIND = "edge"
FORM = "fortification"
ROLE = "defensive"
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
    # **The mass is the part's**, the craft round (E4): a place declares what its wall
    # is -- a screen, a curtain, a rampart, a levee -- and `width` on the part is what
    # that word means in columns.
    "width": ("int", 3, 3),
    "parapet": ("choice", ["crenellated", "plain"]),
    # The face: `framed` is the grid of frame posts and bands this wall has always
    # carried; `banded` is masonry with a string course every few courses and no posts;
    # `plain` is one unbroken outer face under its cornice, an earthen or monolithic
    # wall; `masonry` is dressed stonework -- a plinth course at the foot, a string
    # course at `STRING_EVERY`, buttress piers at `PIER_EVERY` and a batter that steps
    # the face in as it rises, and the ways up at the corners and the gates only, where
    # a framed or banded face carries one every `STAIR_EVERY` columns and reads from the
    # air as timber cross-bracing. **`unbroken` is gone from the choices and is still
    # accepted**, the craft round (E4): it meant the plain face on both sides with
    # sparse stairs, and masonry is what an unbroken wall gets now, so the word says
    # nothing masonry does not. A stored plan that passes it builds masonry. First is
    # the default, so every stored program builds the bytes it built.
    "face": ("choice", ["framed", "banded", "plain", "masonry"]),
}
NEEDS = {
    # The band scripts/type_needs.py measured. **The width reaches a rampart's**, the
    # craft round (E4): the sweep only ever tried 1, 2 and 3, so three was all that was
    # ever certified and a city's outer wall was forty-eight high and three thick -- a
    # screen. `clearance` is the author's: the stair blocks reach six lanes past the
    # inner face where the wall is too thin to carry them inside itself.
    "footprint": (1, 4, 12, 128),
    "frontage": "any",
    "ground": "any",
    "clearance": 8,
}

#: **A crown wide enough to be a road**, the craft round (E4). Below this the wall is a
#: parapet and a ledge; at it and above, the walk is a road between two parapets -- one
#: on each edge -- which is what a person on top of a great wall is walking along.
CROWN_ROAD_MIN = 5

#: **A mural stair stands in the wall's own thickness** where the wall has this much to
#: spare over the lanes the flights need: a switchback of `nfl` lanes takes `nfl` of the
#: inner columns and what is left is still a walk two wide. Below it the stair is a
#: block against the inner face, as it has always been.
STAIR_INSIDE_SPARE = 2

#: The dressed face (`masonry`), the craft round (E4). Forty-eight blocks of one flat
#: plane reads as a render and the grid before it read as lattice; between them is
#: masonry. A plinth of `PLINTH_H` courses at the foot in the footing family, a string
#: course of trim every `STRING_EVERY`, a buttress pier of the wall family every
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

        The craft round, E4: a battered wall is a trapezoid, wide at the foot and narrow at
        the crown, and in a lattice that is the outer lanes stopping short. Never more than
        takes the crown below two columns, and never on a wall too low to show it.
        
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
    first. **Inside the wall's own thickness where the mass allows it** (the craft
    round, E4): a great wall carries its ways up in its body, and only a thin one hangs
    a block of stairs on its inner face."""
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
    # **A plain face carries sparse stairs.** The ground look's third finding: the inner
    # rings' walls took the default `every`, which at a house's scale reads as red
    # diagonal bracing painted on the wall rather than as a way up. A face with no
    # articulation on it has nothing for a switchback every 26 columns to belong to, so
    # the ways down stand at the corners and beside the gates, where a person actually
    # climbs.
    stairs = "sparse" if face in ("unbroken", "plain", "masonry") else "every"
    if face == "unbroken":
        # **An unbroken wall is dressed masonry on both faces**, the craft round (E4):
        # `plain` is forty-eight blocks of one flat plane and reads as a render, which
        # is what every look since the first has said of it.
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
                y = F + 3 + band
                while y <= te - 3:
                    b.place_block(x, y, z, TRIM)
                    y += band
            elif face == "masonry":
                # **dressed stonework**, the craft round (E4): a plinth at the foot in
                # the footing family, a string course of trim every `STRING_EVERY`, and
                # a buttress pier of the wall family standing proud of the face at
                # `PIER_EVERY`. The batter is in the cell's own top, above.
                b.place_cuboid(x, F + 1, z, x, min(F + PLINTH_H, te - 2), z, FOOT)
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
            # plain: one unbroken face, the cornice alone
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
