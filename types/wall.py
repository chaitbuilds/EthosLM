import math
import random

KIND = "edge"
FORM = "fortification"
#: **This wall draws a 45-degree run**, the design synthesis round. A path of axial runs
#: only is drawn as it always was, segment by segment; a path with a diagonal in it --
#: `boundary.Outline.polyline`'s round ring, hundreds of short runs -- is drawn by the
#: curved engine below as one field: a solid band, a walk that climbs its ground in
#: half-block courses, towers by arc length and stair turrets for the ways down.
DIAGONAL_RUNS = True
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

PARAMS = {
    "height": ("int", 3, 20),
    "width": ("int", 1, 5),
    "crown": ("choice", ["crenellated", "machicolated", "solid"]),
}

#: `part["face"]` is `plain` for the flush face of a wall the spec calls unbroken,
#: written by the layout beside the part's level, and anything else is the buttress at
#: every pier and the string course between them this wall has carried. A part field
#: rather than a fourth parameter because the sweep is held to 27 points.
FACES = ("framed", "plain")

#: **A tall wall has a face.** Above this height the outer face carries a buttress a
#: block proud of the parapet line at every pier, three quarters of the way up and
#: capped in trim, and a string course at mid-height between them. Below it the wall
#: keeps its flush piers: a wall of eight is a wall of eight. The city's palace ring is
#: twenty high, and on the volume its flush red piers every five columns read as a blank
#: pale plane from fifty blocks off; relief is what reads at that distance. Registered
#: before the wall was re-swept.
FACE_FROM = 12

NEEDS = {
    # The band `scripts/type_needs.py` measured. **The width reaches a rampart's**, the
    # craft round (E4): the sweep only ever tried 1, 2 and 3, and the re-sweep over the
    # widths a place may declare as a wall's mass stands 2,520 instances of this type at
    # 1, 2, 3, 5, 7, 9 and 12 with none failing. A ring wall of a 512-block city takes a
    # vertex at least every 128 columns.
    "footprint": (1, 4, 12, 128),
    "frontage": "any",
    "ground": "any",
    "clearance": 4,
}


def _line(a, c, axis):
    """Every column from a to c inclusive, in order along the line."""
    x0, z0 = int(a[0]), int(a[1])
    x1, z1 = int(c[0]), int(c[1])
    out = []
    if axis == "x":
        st = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + st, st):
            out.append((x, z0))
    else:
        st = 1 if z1 >= z0 else -1
        for z in range(z0, z1 + st, st):
            out.append((x0, z))
    return out


def _terrain_bias(b, pts, axis):
    """Positive when the +perpendicular side of this run is the lower ground."""
    step = max(1, len(pts) // 8)
    hp = 0
    hm = 0
    c = 0
    for i in range(0, len(pts), step):
        x, z = pts[i]
        if axis == "x":
            hp += b.get_height(x, z + 5)
            hm += b.get_height(x, z - 5)
        else:
            hp += b.get_height(x + 5, z)
            hm += b.get_height(x - 5, z)
        c += 1
    if c == 0:
        return 1.0
    return float(hm - hp) + 0.001


# ======================================================================================
# THE CURVED WALL ENGINE -- one text, carried verbatim by `wall.py` and `great_wall.py`
# (a type file is a standalone program and cannot import its sibling;
# `scripts/test_round_walls.py` asserts the two copies are identical). The design
# synthesis round. A round ring is `boundary.Outline.polyline`: hundreds of short axial
# and 45-degree runs. Drawn segment by segment, as the square-ring code below does,
# every one of those joins is a corner square and a level change, and a diagonal run's
# swept cells are a checkerboard. So a path with a diagonal in it is drawn as **one
# field**: every column near the line is given its distance to the polyline and where
# along it it lies (`u`, in lattice steps), and the wall is the columns within its half-
# width -- a solid, 4-connected band on every run and round every join. What each column
# carries is read off `u`: level the walk climbs the ground: each station's own segment
# floor plus the wall's height, then an upper envelope rising or falling one HALF block
# a station (a slab course), so the walk never steps more than half a block between
# neighbours and never drops under its own segment's floor + height; a tower or a gate
# flattens its reach to one level. body from the column's own footing (the site's, or
# the ground where it is not a site column) to the walk -- never over air. parapet the
# outer ring of the band, solid one course over the walk with a merlon every
# `Builder.merlon(i)` of arc; a crown road carries one on each edge. towers a round
# bastion every `every` blocks of ARC LENGTH, projecting past the outer face, flat-
# topped at the walk with a drum and a crenellated roof; the walk runs through the drum.
# ways down: a square stair turret against the inner face at every `turret_nth` tower, a
# switchback climbing round its core from a door at the inner ground to the walk.
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
TOWER_EVERY = 48

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


def occupied(part=None, **params):
    """**What a curved wall fills** past its band (the design synthesis round): towers
    project past the outer face and stair turrets stand against the inner one. A square
    path publishes nothing, as before, and is held to its band."""
    part = part or {}
    if not _curved_path(part):
        return None
    width = max(1, min(5, int(params.get("width", 3))))
    height = int(params.get("height", 8))
    return {"band": int(part.get("width") or width), "outer": 6, "inner": 8, "either": 0,
            "above": 8, "clearance": int(NEEDS["clearance"]),
            "total": int(part.get("width") or width) + 14,
            "why": (f"a curved wall {height} high: towers every {TOWER_EVERY} blocks of "
                    f"arc project 6 past the outer face, stair turrets stand 8 into the "
                    f"inner side")}


def _curved(part):
    segs = part.get("segments") or []
    return any(s.get("axis") == "d" for s in segs) or len(segs) > CURVED_FROM_SEGMENTS


def build(b, part, seed, **params):
    rnd = random.Random(seed * 7919 + 13)
    height = max(3, min(20, int(params.get("height", 8))))
    width = max(1, min(5, int(params.get("width", 3))))
    crown = params.get("crown", "crenellated")
    if crown not in ("crenellated", "machicolated", "solid"):
        crown = "crenellated"
    face_kind = part.get("face") or "framed"
    if face_kind not in FACES:
        face_kind = "framed"

    voice = part["voice"]
    WALL = b.block(voice["wall"], "full")
    FOOT = b.block(voice["footing"], "full")
    TRIM = b.block(voice["trim"], "full")
    PAVE = b.block(voice["roof"], "full")
    STEPMAT = voice["roof"]
    GF = int(part["floor_y"])

    if _curved(part):
        # the walk two under the crown: this wall's own section, drawn along a curve
        # (see the curved engine above). The parapet stands outside the body where the
        # band the layout drew has room for it, and on the body's outer lane where the
        # body fills the band -- never past the band, where the ground is the next
        # ring's and a column cannot be built down to it.
        band = max(1, int(part.get("width") or width))
        par_out = width + 1 <= band
        return _cw_build(b, part, seed, height - 2, {
            "wall": WALL, "foot": FOOT, "trim": TRIM, "frame": TRIM, "pave": PAVE,
            "slab": b.block(voice["roof"], "slab") + "[type=bottom]", "stepmat": STEPMAT,
            "width": width if par_out else band,
            "face": "plain" if face_kind == "plain" else "piers",
            "pier_every": rnd.choice([4, 5]), "string_mid": height >= FACE_FROM,
            "foot_courses": 2, "crenel": crown != "solid", "road": False,
            "parapet_outside": par_out, "tower_every": TOWER_EVERY,
            "tower_r": 3.5 + (width - 1) // 2, "tower_push": 2.0, "drum": 4,
            "turret_nth": 2, "gate_height": height})

    out = width // 2
    inn = width - 1 - out
    M = out + 1

    # ---- the spine: one ordered run of columns down the whole line -----------
    segs = part["segments"]
    spine = []
    sfloor = []
    sidx = []
    junctions = []
    seg_pts = []
    for si, s in enumerate(segs):
        cells = _line(s["a"], s["b"], s["axis"])
        seg_pts.append(cells)
        if si > 0:
            junctions.append(len(spine))
            cells = cells[1:]
        for (x, z) in cells:
            spine.append((x, z))
            sfloor.append(int(s["floor_y"]))
            sidx.append(si)
    if not spine:
        return None
    closed = len(spine) > 3 and spine[0] == spine[-1]
    if closed:
        spine.pop()
        sfloor.pop()
        sidx.pop()
        junctions.append(0)
    n = len(spine)

    segstart = {}
    segend = {}
    for i in range(n):
        si = sidx[i]
        if si not in segstart:
            segstart[si] = i
        segend[si] = i

    # what an empty cell is called here, read off the sky rather than named
    probe = []
    for k in (0, n // 3, (2 * n) // 3):
        px0, pz0 = spine[k % n]
        probe.append(b.get_block(px0, GF + 58, pz0))
    AIR = probe[0] if (probe[0] == probe[1] and probe[1] == probe[2]) else None

    # ---- which way is out ----------------------------------------------------
    cx = sum(p[0] for p in spine) / float(n)
    cz = sum(p[1] for p in spine) / float(n)
    outv = []
    for si, s in enumerate(segs):
        if s["axis"] == "x":
            d = float(s["a"][1]) - cz
            if abs(d) < 1.0:
                d = _terrain_bias(b, seg_pts[si], "x")
            outv.append((0, 1 if d >= 0 else -1))
        else:
            d = float(s["a"][0]) - cx
            if abs(d) < 1.0:
                d = _terrain_bias(b, seg_pts[si], "z")
            outv.append((1 if d >= 0 else -1, 0))

    # ---- the height of the crown, column by column --------------------------- each
    # segment wants its own footing plus `height`; where two segments meet, the higher
    # of the two wins over a small plateau and the wall ramps a block a column back down
    # to its own level, so the walk never steps more than one and a person walks the
    # whole length without stopping.
    T = [sfloor[i] + height for i in range(n)]
    top = list(T)
    for j in junctions:
        v = max(T[(j - 1) % n], T[j], T[(j + 1) % n])
        for d in range(-M, M + 1):
            k = j + d
            if closed:
                k %= n
            elif k < 0 or k >= n:
                continue
            if v > top[k]:
                top[k] = v
    for _ in range(2):
        for i in range(n):
            k = i - 1
            if k < 0:
                if not closed:
                    continue
                k = n - 1
            if top[k] - 1 > top[i]:
                top[i] = top[k] - 1
        for i in range(n - 1, -1, -1):
            k = i + 1
            if k >= n:
                if not closed:
                    continue
                k = 0
            if top[k] - 1 > top[i]:
                top[i] = top[k] - 1
    YMAX = GF + 40
    for i in range(n):
        if top[i] > YMAX:
            top[i] = YMAX
    walk = [t - 2 for t in top]

    # ---- ground, and the record of what stands where --------------------------
    ghc = {}

    def gh(x, z):
        k = (x, z)
        v = ghc.get(k)
        if v is None:
            v = b.get_height(x, z)
            ghc[k] = v
        return v

    def base_at(x, z, sf):
        g = gh(x, z)
        v = sf + 1
        if g + 1 < v:
            v = g + 1
        if v < GF:
            v = GF
        return v

    # A column the library sited -- every column of every segment's swept width -- has a
    # footing laid to its segment's level, so it is always usable. A column outside that
    # width (a parapet's overhang, a buttress, a flight) is usable where its own ground
    # is within reach of the **local segment's** level, never the polyline's minimum:
    # the concentric run's ring walls omitted every column whose bed lay three below the
    # lowest segment of a ring, and read as walls that vanished into the earth over
    # every hollow.
    sited = set()
    for s in segs:
        for c in s.get("cells") or []:
            sited.add((int(c[0]), int(c[1])))

    def usable(x, z, sf=None):
        if (x, z) in sited:
            return True
        return gh(x, z) > (GF if sf is None else sf) - 3

    crest = {}
    placed = set()

    def mark(x, z, y):
        k = (x, z)
        v = crest.get(k)
        if v is None or y > v:
            crest[k] = y

    # ---- the columns the wall stands in --------------------------------------
    wcols = {}          # (x,z) -> [walk_y, seg_floor, spine_index, face]
    pcols = {}          # (x,z) -> [walk_y, seg_floor, spine_index, px, pz]
    for i in range(n):
        x, z = spine[i]
        px, pz = outv[sidx[i]]
        for o in range(-inn, out + 1):
            key = (x + px * o, z + pz * o)
            face = 0
            if o == out:
                face = 1
            elif o == -inn:
                face = -1
            cur = wcols.get(key)
            if cur is None or walk[i] > cur[0]:
                wcols[key] = [walk[i], sfloor[i], i, face]
        key = (x + px * (out + 1), z + pz * (out + 1))
        cur = pcols.get(key)
        if cur is None or walk[i] > cur[0]:
            pcols[key] = [walk[i], sfloor[i], i, px, pz]

    # the corner: a square bastion both segments run into, so the walk turns without a
    # gap and the parapet carries round the outside of it.
    for j in junctions:
        x, z = spine[j]
        p1 = outv[sidx[(j - 1) % n]]
        p2 = outv[sidx[j]]
        if p1[0] * p2[1] - p1[1] * p2[0] == 0:
            continue
        sf = max(sfloor[(j - 1) % n], sfloor[j])
        for a in range(-M, M + 1):
            for c in range(-M, M + 1):
                key = (x + p1[0] * a + p2[0] * c, z + p1[1] * a + p2[1] * c)
                if a == out + 1 or c == out + 1:
                    qx, qz = (p1 if a == out + 1 else p2)
                    cur = pcols.get(key)
                    if cur is None or walk[j] > cur[0]:
                        pcols[key] = [walk[j], sf, j, qx, qz]
                else:
                    cur = wcols.get(key)
                    if cur is None or walk[j] > cur[0]:
                        wcols[key] = [walk[j], sf, j, 0]
    for key in list(pcols.keys()):
        if key in wcols:
            del pcols[key]

    # ---- masonry -------------------------------------------------------------
    pierspace = rnd.choice([4, 5])
    talus = rnd.random() < 0.55
    for (x, z) in wcols:
        wy, sf, i, face = wcols[(x, z)]
        if not usable(x, z, sf):
            continue
        placed.add((x, z))
        bs = base_at(x, z, sf)
        if bs > wy:
            bs = wy
        ftop = bs + 1
        if ftop > wy - 1:
            ftop = wy - 1
        if ftop >= bs:
            b.place_cuboid(x, bs, z, x, ftop, z, FOOT)
            body0 = ftop + 1
        else:
            body0 = bs
        if body0 <= wy - 1:
            mat = WALL
            if face != 0 and (i % pierspace) == 0:
                mat = TRIM
            b.place_cuboid(x, body0, z, x, wy - 1, z, mat)
            if face != 0 and mat != TRIM:
                # the cornice under the walk: one continuous line, a figure
                with b.figure("wall_cornice"):
                    b.place_block(x, wy - 1, z, TRIM)
        b.place_block(x, wy, z, PAVE)
        mark(x, z, wy)

    # two walls of one city crowned differently and neither reading as a parapet.
    # `Builder.merlon()` is the rhythm, the wall family is what it is built of, and the
    # trim is left to the string course and the corbel, which is what a string course
    # is.
    brk = rnd.choice([2, 3])
    for (x, z) in pcols:
        wy, sf, i, px, pz = pcols[(x, z)]
        if not usable(x, z, sf):
            continue
        bs = base_at(x, z, sf)
        if bs > wy:
            bs = wy
        ftop = bs + 1
        if ftop > wy:
            ftop = wy
        if ftop >= bs:
            b.place_cuboid(x, bs, z, x, ftop, z, FOOT)
            body0 = ftop + 1
        else:
            body0 = bs
        if body0 <= wy:
            b.place_cuboid(x, body0, z, x, wy, z, WALL)
        b.place_block(x, wy + 1, z, WALL)
        if crown == "crenellated":
            cap = b.merlon(i)
        elif crown == "machicolated":
            cap = (i % 2) == 0
        else:
            cap = True
        if cap:
            b.place_block(x, wy + 2, z, WALL)
            mark(x, z, wy + 2)
        else:
            mark(x, z, wy + 1)
        # the overhang: a corbel bracket standing a block proud of the face
        if talus and (i % brk) == 0 and wy - 1 >= bs:
            kx, kz = x + px, z + pz
            if (kx, kz) not in wcols and (kx, kz) not in pcols \
                    and usable(kx, kz, sf):
                b.place_block(kx, wy - 1, kz, TRIM)
                mark(kx, kz, wy - 1)
        # the face, above FACE_FROM: a buttress at every pier, proud of the parapet
        # line, and a string course between the buttresses at mid-height -- unless the
        # spec called the wall unbroken, in which case the face is the face
        if height >= FACE_FROM and face_kind != "plain":
            kx, kz = x + px, z + pz
            if (kx, kz) not in wcols and (kx, kz) not in pcols and usable(kx, kz, sf):
                bb = base_at(kx, kz, sf)
                if (i % pierspace) == 0:
                    tb = bs + ((wy - bs) * 3) // 4
                    if tb >= bb:
                        b.place_cuboid(kx, bb, kz, kx, tb, kz, WALL)
                        # the cap of the pier: part of the same drawn line as the string
                        # course below, and declared a figure for the same reason
                        with b.figure("wall_string_course"):
                            b.place_block(kx, tb + 1, kz, b.block(voice["trim"], "slab")
                                          + "[type=bottom]")
                        mark(kx, kz, tb + 1)
                else:
                    sc = bs + (wy - bs) // 2
                    if sc > bb:
                        # **The string course is a figure.** Composition round. This one
                        # trim slab per bay, all at one height, is the horizontal line
                        # that says the wall is dressed; the design round's material
                        # pass put vertical stains up this face that cut straight
                        # through it (`out/des-material/comparison.json`, criterion 5,
                        # on `great_wall_upper_ring` -- which is this type). The wall
                        # body either side of the line stays editable, so the mass may
                        # age while the line does not break.
                        with b.figure("wall_string_course"):
                            b.place_block(kx, sc, kz, b.block(voice["trim"], "slab")
                                          + "[type=top]")
                        mark(kx, kz, sc)

    # ---- where the walk changes level, it changes on treads ------------------
    ramp = [False] * n
    for i in range(n):
        if i > 0:
            p = walk[i - 1]
        elif closed:
            p = walk[n - 1]
        else:
            p = walk[i]
        if i < n - 1:
            q = walk[i + 1]
        elif closed:
            q = walk[0]
        else:
            q = walk[i]
        ramp[i] = (walk[i] != p) or (walk[i] != q)
    i = 0
    while i < n:
        if not ramp[i]:
            i += 1
            continue
        j = i
        while j < n and ramp[j] and sidx[j] == sidx[i]:
            j += 1
        for o in range(-inn, out + 1):
            cells = []
            for k in range(i, j + 1):
                key = None
                if k < j:
                    x, z = spine[k]
                    px, pz = outv[sidx[k]]
                    key = (x + px * o, z + pz * o)
                if key is not None and key in placed:
                    cells.append((key[0], walk[k], key[1]))
                else:
                    if len(cells) > 1:
                        b.steps(cells, STEPMAT)
                    cells = []
        i = j

    # ---- getting off the wall ------------------------------------------------ a
    # flight running along the inner face, just outside it, on a solid ramp of its own,
    # doubling back once where the drop is too long for a single run.
    def lay_flight(cells, gbase):
        kept = []
        for (x, y, z) in cells:
            if not usable(x, z):
                break
            base = gh(x, z) + 1
            if base > gbase + 1:
                base = gbase + 1
            if base < GF:
                base = GF
            if base <= y - 1:
                b.place_cuboid(x, base, z, x, y - 1, z, FOOT)
            mark(x, z, y)
            kept.append((x, y, z))
        if kept:
            b.steps(kept, STEPMAT)
        return len(kept)

    # **A plain face carries sparse ways down**, the craft round. The layout stamps
    # `stairs` on the part beside its `face`; `sparse` puts a way down at each vertex
    # and beside each gate, where a person actually climbs, and nowhere else.
    sparse = str(part.get("stairs") or "") == "sparse"
    spacing = rnd.choice([14, 16, 18])
    if sparse:
        anchors = []
        for si in range(len(seg_pts)):
            a = segstart[si] + M + 2
            if segstart[si] <= a <= segend[si]:
                anchors.append(a)
    else:
        anchors = list(range(rnd.randrange(2, 8), n, spacing))
    if not anchors:
        anchors = [n // 2]
    # A way down never lands in a gate's pad: siting hands this wall the points standing
    # on it (`part["gates"]`, each with its anchor and pad), and a flight laid where a
    # gate is then sited is a flight the gate's pad cuts through -- the city's palace
    # gate had this wall's own treads facing down its quarried cut.
    keep_off = set()
    for g in (part.get("gates") or []):
        gx, gz = int(g["at"][0]), int(g["at"][-1])
        half = int(g.get("size", 5)) // 2 + M + 3
        near = [i for i in range(n) if abs(spine[i][0] - gx) + abs(spine[i][1] - gz) <= half]
        keep_off.update(near)
    anchors = [a for a in anchors if a not in keep_off] or anchors
    for ai in anchors:
        si = sidx[ai]
        lo = segstart[si]
        hi = segend[si]
        px, pz = outv[sidx[ai]]
        pts = seg_pts[si]
        if len(pts) < 2:
            continue
        if pts[1][0] != pts[0][0]:
            ux, uz = (1 if pts[1][0] > pts[0][0] else -1), 0
        else:
            ux, uz = 0, (1 if pts[1][1] > pts[0][1] else -1)
        if (hi - ai) >= (ai - lo):
            dx, dz = ux, uz
        else:
            dx, dz = -ux, -uz
        # nudge the flight to ground it can actually land on
        best = None
        besta = None
        for shift in (0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 13, -13):
            a = ai + shift
            if a - lo < M + 1:
                a = lo + M + 1
            if hi - a < M + 1:
                a = hi - M - 1
            if a < lo or a > hi:
                continue
            room = (hi - a) if (dx * ux + dz * uz) > 0 else (a - lo)
            lim = max(2, min(18, room - M))
            gmin = None
            for o in (inn + 1, inn + 2):
                for k in range(0, lim + 1, 2):
                    g = gh(spine[a][0] - px * o + dx * k,
                           spine[a][1] - pz * o + dz * k)
                    if gmin is None or g < gmin:
                        gmin = g
            if gmin is not None and (best is None or gmin > best):
                best = gmin
                besta = a
            if best is not None and best > GF - 3:
                break
        if besta is None:
            continue
        a = besta
        room = (hi - a) if (dx * ux + dz * uz) > 0 else (a - lo)
        limit = max(2, min(18, room - M))
        W = walk[a]
        sx, sz = spine[a]
        l1x = sx - px * (inn + 1)
        l1z = sz - pz * (inn + 1)
        l2x = sx - px * (inn + 2)
        l2z = sz - pz * (inn + 2)
        low = None
        for k in range(0, limit + 1):
            for (qx, qz) in ((l1x + dx * k, l1z + dz * k),
                             (l2x + dx * k, l2z + dz * k)):
                g = gh(qx, qz)
                if g <= GF - 3:
                    continue
                if low is None or g < low:
                    low = g
        if low is None or low < GF:
            low = GF
        need = W - low
        if need <= 0:
            continue
        n1 = need if need < limit else limit
        run1 = []
        for k in range(0, n1):
            run1.append((l1x + dx * k, W - k, l1z + dz * k))
        got = lay_flight(run1, low)
        if need <= n1 or got < n1:
            continue
        # a landing, then back the other way one column further in
        landy = W - n1
        n2 = need - n1
        if n2 > n1:
            n2 = n1
        stop = False
        for (bx, bz) in ((l1x, l1z), (l2x, l2z)):
            x = bx + dx * n1
            z = bz + dz * n1
            if not usable(x, z):
                stop = True
                continue
            base = gh(x, z) + 1
            if base > low + 1:
                base = low + 1
            if base < GF:
                base = GF
            if base <= landy:
                b.place_cuboid(x, base, z, x, landy, z, FOOT)
            mark(x, z, landy)
        if stop:
            continue
        run2 = []
        for k in range(0, n2):
            run2.append((l2x + dx * (n1 - 1 - k), landy - k,
                         l2z + dz * (n1 - 1 - k)))
        lay_flight(run2, low)

    # ---- the gate ------------------------------------------------------------ a wall
    # with no way through it is a wall a city cannot use. It goes only where the wall is
    # deep enough to carry the walk over the opening and the ground stands level on both
    # sides of it -- and only where no gate part stands on this wall: siting hands the
    # points standing on an edge back as `part["gates"]`, and a gate's pad is the way
    # through.
    if AIR is not None and closed and not part.get("gates"):
        best = None
        for si in range(len(segs)):
            ln = segend[si] - segstart[si]
            if best is None or ln > best[0]:
                best = (ln, si)
        gi = None
        if best is not None and best[0] > 2 * M + 6:
            si = best[1]
            mid = (segstart[si] + segend[si]) // 2
            for shift in (0, 3, -3, 6, -6, 9, -9, 12, -12):
                a = mid + shift
                if a - segstart[si] < M + 3 or segend[si] - a < M + 3:
                    continue
                base = sfloor[a] + 1
                if walk[a] - base < 4:
                    continue
                px, pz = outv[sidx[a]]
                sx, sz = spine[a]
                ox, oz = sx + px * (out + 2), sz + pz * (out + 2)
                ix, iz = sx - px * (inn + 2), sz - pz * (inn + 2)
                if gh(ox, oz) != base - 1 or gh(ix, iz) != base - 1:
                    continue
                gi = a
                break
        if gi is not None:
            a = gi
            base = sfloor[a] + 1
            px, pz = outv[sidx[a]]
            sx, sz = spine[a]
            for o in range(-(inn + 2), out + 3):
                b.fill_region(sx + px * o, base, sz + pz * o,
                              sx + px * o, base + 2, sz + pz * o, AIR)
            if px == 0:
                face = "south" if pz > 0 else "north"
            else:
                face = "east" if px > 0 else "west"
            b.doorway(sx, base, sz, face, voice["frame"],
                      leaf=b.joinery(voice, "door"), jamb="build", lintel=True)
        else:
            # too low for a gateway under the walk: a postern through the parapet, where
            # the ground outside stands level with the walk.
            cands = []
            near = []
            start = rnd.randrange(2, 7)
            for a in range(start, n, 2):
                atjoin = False
                for j in junctions:
                    if abs(a - j) <= M + 2:
                        atjoin = True
                if atjoin:
                    continue
                px, pz = outv[sidx[a]]
                sx, sz = spine[a]
                qx, qz = sx + px * (out + 1), sz + pz * (out + 1)
                if (qx, qz) not in pcols or not usable(qx, qz):
                    continue
                if (sx + px * out, sz + pz * out) not in placed:
                    continue
                wy = walk[a]
                ox, oz = sx + px * (out + 2), sz + pz * (out + 2)
                d0 = gh(ox, oz) - wy
                if abs(d0) > 1:
                    continue
                fx, fz = sx + px * (out + 3), sz + pz * (out + 3)
                if abs(gh(fx, fz) - wy) > 2:
                    continue
                if d0 == 0:
                    cands.append((a, qx, qz, wy, px, pz))
                elif len(near) < 8:
                    near.append((a, qx, qz, wy, px, pz))
                if len(cands) >= 10:
                    break
            chosen = None
            for (a, qx, qz, wy, px, pz) in cands + near:
                b.fill_region(qx, wy + 1, qz, qx, wy + 2, qz, AIR)
                r = b.check_door(qx, wy + 1, qz)
                if r is not None and r.get("ok"):
                    chosen = (a, qx, qz, wy, px, pz)
                    break
                cap = b.merlon(a) if crown == "crenellated" else (
                    (a % 2) == 0 if crown == "machicolated" else True)
                b.place_block(qx, wy + 1, qz, WALL)
                if cap:
                    b.place_block(qx, wy + 2, qz, WALL)
            if chosen is not None:
                (a, qx, qz, wy, px, pz) = chosen
                if px == 0:
                    face = "south" if pz > 0 else "north"
                else:
                    face = "east" if px > 0 else "west"
                b.doorway(qx, wy + 1, qz, face, voice["frame"],
                          leaf=b.joinery(voice, "door"), jamb="build",
                          lintel=False)
                mark(qx, qz, wy + 2)

    # ---- the wall cuts what stood in its columns; nothing is left hanging -----
    if AIR is not None:
        for (x, z) in crest:
            y0 = crest[(x, z)] + 1
            y1 = gh(x, z) + 4
            if y1 < y0 + 2:
                y1 = y0 + 2
            if y1 > YMAX:
                y1 = YMAX
            if y0 <= y1:
                b.fill_region(x, y0, z, x, y1, z, AIR)

    # ---- anything the mass roofed over that nobody can reach ------------------
    pad = M + 3
    for si in range(len(segs)):
        pts = seg_pts[si]
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        b.seal_voids(min(xs) - pad, min(zs) - pad, max(xs) + pad,
                     max(zs) + pad, FOOT, max_cells=900)

    return {"columns": len(wcols), "parapet": len(pcols), "top": max(top)}
