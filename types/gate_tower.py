import math
import random

FORM = "fortification"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "defensive"

KIND = "point"
PASSAGE = True

PARAMS = {
    "storeys": ("int", 1, 3),
    "crown": ("choice", ["gable", "hip", "gambrel"]),
}


#: What this type needs from the ground before it can stand.
NEEDS = {
    # 3 to 6 measured; above six its own chamber is a room the stair does not reach.
    # Measured by scripts/type_needs.py; the band is rounds/type-needs.json. The pair is
    # the pad site() hands build(), after siting's inset.
    "footprint": (3, 3, 16, 16),
    "except": (7, 8),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

# --- the settlement palette, by role. A gate tower is a form -- a passage, a stair, a
# chamber over it and a crown -- and it is the same tower in every palette.

def _axial(block, axis):
    return (f"{block}[axis={axis}]"
            if block.endswith(("_log", "_pillar", "_wood")) else block)


def _wall(b):
    return b.block(b.voice["wall"])


def _wall_alt(b):
    return b.block(b.voice["wall"], "accent")


def _wall_fine(b):
    return b.block(b.voice["wall"], "fine")


def _wall_chis(b):
    return b.block(b.voice["wall"], "accent")


def _frame(b):
    return b.block(b.voice["frame"], "post")


def _frame_plank(b):
    return b.block(b.voice["frame"])


def _trim(b):
    return b.block(b.voice["trim"], "bare")


def _foot(b):
    return b.block(b.voice["footing"])


def _foot_slab(b):
    return b.block(b.voice["footing"], "slab")


def _foot_mossy(b):
    return b.block(b.voice["footing"], "accent")


def _roof_family(b):
    return b.voice["roof"]


def _roof_block(b):
    return b.block(b.voice["roof"])


def _roof_slab(b):
    return b.block(b.voice["roof"], "slab")


def _roof_stair(b):
    return b.block(b.voice["roof"], "stairs")
GLASS = "glass"
PANE = "glass_pane"

STOREY = 4

#: A wall this many blocks over the gate's floor is one the chamber over the arch cannot
#: clear: the gate is built as a tower to the wall's crown -- the piers carried up as a
#: solid base with the arch through it, a switchback stair in the stair flank, the
#: chamber and its stages above the crown. A town wall of five keeps the gatehouse it
#: always had. The same bar `ring_gate` uses.
TALL_FROM = 12

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}


def _axis_of(facing):
    """'x' when the way in runs east-west, 'z' when it runs north-south."""
    return "x" if facing in ("east", "west") else "z"


def _post(b, axis):
    return _axial(_frame(b), axis)


def _stair(block, facing, half="bottom"):
    return block + "[facing=" + facing + ",half=" + half + "]"


class Frame(object):
    """The through-axis of the gate, in plain numbers.

        `t` runs along the way the gate faces (the road), `c` runs across it.
        Everything below is written in (t, c) and turned back into (x, z) here,
        so the same gatehouse can stand on either axis.
        
    """

    def __init__(self, part):
        self.part = part
        self.label = part["label"]
        self.facing = part.get("facing") or "north"
        self.axis = _axis_of(self.facing)
        at = part.get("at") or [
            (part["x0"] + part["x1"]) // 2,
            (part["z0"] + part["z1"]) // 2,
        ]
        self.cx, self.cz = int(at[0]), int(at[1])
        self.x0, self.x1 = min(part["x0"], part["x1"]), max(part["x0"], part["x1"])
        self.z0, self.z1 = min(part["z0"], part["z1"]), max(part["z0"], part["z1"])
        self.floor_y = int(part["floor_y"])
        if self.axis == "x":
            self.t0, self.t1, self.tc = self.x0, self.x1, self.cx
            self.c0, self.c1, self.cc = self.z0, self.z1, self.cz
        else:
            self.t0, self.t1, self.tc = self.z0, self.z1, self.cz
            self.c0, self.c1, self.cc = self.x0, self.x1, self.cx
        # keep the road centred on the cell the plan gave us
        self.cc = max(self.c0 + 1, min(self.c1 - 1, self.cc))
        self.tc = max(self.t0 + 1, min(self.t1 - 1, self.tc))

    def xz(self, t, c):
        return (t, c) if self.axis == "x" else (c, t)

    def inside(self, t, c):
        return self.t0 <= t <= self.t1 and self.c0 <= c <= self.c1

    # the four compass names, in gate terms
    def cross_name(self, sign):
        """The compass word for +1 (sign=1) or -1 along the cross axis."""
        if self.axis == "x":
            return "south" if sign > 0 else "north"
        return "east" if sign > 0 else "west"

    def along_name(self, sign):
        if self.axis == "x":
            return "east" if sign > 0 else "west"
        return "south" if sign > 0 else "north"


def _longest_run(ts):
    best, run = [], []
    for t in ts:
        if run and t == run[-1] + 1:
            run.append(t)
        else:
            run = [t]
        if len(run) > len(best):
            best = list(run)
    return best


def _plan(b, F, seed, storeys, crown):
    """Everything the seed decides, decided once."""
    rng = random.Random(seed * 7919 + 13)
    p = {}
    p["rng"] = rng
    p["storeys"] = max(1, min(3, int(storeys)))
    p["crown"] = crown if crown in ("gable", "hip", "gambrel") else "gable"

    # The lane is public ground. A gate is the one place it may be crossed, and it is
    # crossed over, not stood on: nothing of ours goes in a lane cell below the height a
    # person walks at.
    lane = set()
    for t in range(F.t0, F.t1 + 1):
        for c in range(F.c0, F.c1 + 1):
            x, z = F.xz(t, c)
            nl = b.nearest_lane(x, z)
            if nl and nl.get("distance") == 0:
                lane.add((t, c))
    p["lane"] = lane

    # which flank carries the stair, and which is solid masonry
    p["stair_side"] = 1 if rng.random() < 0.5 else -1
    p["stair_c"] = F.cc + p["stair_side"] * 2
    p["pier_c"] = F.cc - p["stair_side"] * 2
    if not (F.c0 <= p["stair_c"] <= F.c1):
        p["stair_side"] = -p["stair_side"]
        p["stair_c"] = F.cc + p["stair_side"] * 2
        p["pier_c"] = F.cc - p["stair_side"] * 2

    # the pier stands on the ground that is ours; the storey over it comes out on
    # brackets over the rest
    p["pier_ts"] = _longest_run(
        [t for t in range(F.t0, F.t1 + 1) if (t, p["pier_c"]) not in lane])
    if not p["pier_ts"]:
        p["pier_ts"] = [F.t0]

    # the flight climbs the other flank. A tread may pass over a lane cell once it is
    # high enough to leave a person room under it, never below.
    rise = STOREY
    best = None
    for s in (1, -1):
        start = F.t0 if s > 0 else F.t1
        end = start + s * rise
        if not (F.t0 <= end <= F.t1):
            continue
        cost = sum(1 for k in range(rise)
                   if (start + s * k, p["stair_c"]) in lane and k < 2)
        cost = cost * 10 + (0 if rng.random() < 0.5 else 1)
        if best is None or cost < best[0]:
            best = (cost, s, start, end)
    if best is None:
        best = (0, 1, F.t0, min(F.t1, F.t0 + rise))
    p["climb"] = best[1]
    p["stair_start"] = best[2]
    p["stair_top"] = best[3]

    # levels. The chamber floor is one storey over the road, which is the height a wall
    # walk comes in at; the stages above it are the tower rising, lit at each stage,
    # under one roof. **In a wall taller than that** -- siting names the wall's height
    # in `part["edge"]` -- the chamber floor is the wall's crown and the piers are
    # carried up to it as a base.
    p["head_y"] = F.floor_y + 3          # the arch springs here
    p["floors"] = [F.floor_y + STOREY]
    edge = F.part.get("edge") or {}
    p["tall"] = False
    if edge.get("height"):
        wf = edge.get("floor_y")
        crown = int(wf if wf is not None else F.floor_y) + int(edge["height"])
        wide = (F.c1 - F.c0 + 1) >= 9 and (F.t1 - F.t0 + 1) >= 9
        if crown - F.floor_y >= TALL_FROM and wide:
            p["tall"] = True
            p["floors"] = [crown]
    p["walk_y"] = p["floors"][0]         # a wall walk would come in here
    p["stage_h"] = 3
    p["wall_top"] = p["floors"][0] + p["stage_h"] * p["storeys"]

    # the chamber gives its outer flank to the stair, which climbs the whole height of
    # the gate in one run, open to the road under the roof; a tower's stair doubles back
    # in two lanes inside the flank, and the chamber keeps off both
    if p["tall"]:
        # the two lanes of the stair flank must be the tower's own: a road crossing the
        # pad on the skew leaves no flank to climb in, and then the gate keeps the
        # gatehouse form rather than standing a tower nobody can get up
        free = []
        for side in (p["stair_side"], -p["stair_side"]):
            lanes = [F.cc + side * 2, F.cc + side * 3]
            if all(F.c0 <= c <= F.c1 for c in lanes) and not any(
                    (t, c) in lane for t in range(F.t0, F.t1 + 1) for c in lanes):
                free.append(side)
        if free:
            p["stair_side"] = free[0]
            p["stair_c"] = F.cc + p["stair_side"] * 2
            p["pier_c"] = F.cc - p["stair_side"] * 2
        else:
            p["tall"] = False
            p["floors"] = [F.floor_y + STOREY]
            p["walk_y"] = p["floors"][0]
            p["wall_top"] = p["floors"][0] + p["stage_h"] * p["storeys"]
    if p["tall"]:
        p["lanes"] = [F.cc + p["stair_side"] * 2, F.cc + p["stair_side"] * 3]
        if p["stair_side"] > 0:
            p["room_c"] = (F.c0, p["lanes"][0] - 1)
        else:
            p["room_c"] = (p["lanes"][0] + 1, F.c1)
    elif p["stair_side"] > 0:
        p["room_c"] = (F.c0, F.c1 - 1)
    else:
        p["room_c"] = (F.c0 + 1, F.c1)

    # the way in is at the head of the flight, one back from the landing, so that the
    # wall it stands in has a jamb to either side of it
    p["door_c"] = p["room_c"][1] if p["stair_side"] > 0 else p["room_c"][0]
    p["door_t"] = p["stair_top"] - p["climb"]

    # the signature: an oriel over one mouth, or over both
    p["oriel_both"] = rng.random() < 0.35
    near = F.t0 if F.along_name(-1) == F.facing else F.t1
    far = F.t1 if near == F.t0 else F.t0
    p["oriel_ends"] = [near] + ([far] if p["oriel_both"] else [])
    cand = [c for c in range(F.cc - 1, F.cc + 2)
            if p["room_c"][0] < c < p["room_c"][1]]
    if rng.random() < 0.4 and cand:
        cand = [min(cand, key=lambda c: abs(c - F.cc))]
    p["oriel_cells"] = cand

    # the crown
    p["pitch"] = rng.choice([(2, 1), (2, 1), (3, 1)])
    p["ridge_axis"] = rng.choice(["t", "c"])
    p["eave"] = rng.choice(["straight", "flared", "upturned"])
    p["banded"] = rng.random() < 0.5     # a trim band under each storey
    p["top_inset"] = p["storeys"] >= 3 and rng.random() < 0.6
    p["lamp"] = rng.choice(["lantern", "lantern", "torch"])
    p["shutters"] = rng.random() < 0.5
    return p


def _bracket(b, x, y, z, out_dir, block=None):
    """A corbel under a jetty or an oriel: the end of a joist, showing.
    A timber bracket, not a tread -- so it is a log, laid on its side."""
    axis = "x" if out_dir in ("east", "west") else "z"
    b.place_block(x, y, z, _axial(block or _trim(b), axis))


def _clear(b, F, p):
    top = p["floors"][0] + STOREY * p["storeys"] + 14
    b.fill_region(F.x0, F.floor_y + 1, F.z0, F.x1, top, F.z1, "air")


def _base(b, F, p):
    """A tower's base: the piers carried up solid from the arch head to the
    crown, the road's three columns left open to the arch head, the corners
    posted. Nothing of it below the arch head, which `_ground` lays."""
    fy0 = p["floors"][0]
    head = p["head_y"]
    if fy0 - 1 <= head:
        return 0
    laid = 0
    for t in range(F.t0, F.t1 + 1):
        for c in range(F.c0, F.c1 + 1):
            if F.cc - 1 <= c <= F.cc + 1 and (t, c) not in p["lane"]:
                # over the road: solid from the course above the arch head
                lo = head + 1
            elif (t, c) in p["lane"]:
                lo = head + 1
            else:
                lo = head + 1
            x, z = F.xz(t, c)
            corner = t in (F.t0, F.t1) and c in (F.c0, F.c1)
            for y in range(lo, fy0):
                b.place_block(x, y, z, _post(b, "y") if corner
                              else _panel_block(b, p, t, c, y))
                laid += 1
    return laid


def _mural_flight(b, F, p):
    """A tower's stair: flights along the road's axis in the two lanes of the stair
    flank, each doubling back on the one below, from the road to the crown. Every
    flight is the library's `flight()`: cleared, carried and walked."""
    fy = F.floor_y
    y1 = p["floors"][0]
    lanes = p["lanes"]
    t_lo, t_hi = F.t0 + 1, F.t1 - 1
    r_max = (t_hi - t_lo + 1) - 2           # foot, treads, landing inside t_lo..t_hi
    need = y1 - fy
    n_fl = max(1, -(-need // r_max))
    base_r, extra = divmod(need, n_fl)
    runs = [base_r + (1 if q < extra else 0) for q in range(n_fl)]
    s = 1 if p["stair_start"] <= F.tc else -1
    foot_t, y = (t_lo if s > 0 else t_hi), fy
    reps = []
    for k, r in enumerate(runs):
        lane = lanes[k % 2]
        first = foot_t + s
        land_t = foot_t + s * (r + 1)
        if not (t_lo <= first <= t_hi and t_lo <= land_t <= t_hi):
            return {"ok": False, "reason": f"flight {k} of {r} does not fit the flank"}
        fx, fz = F.xz(foot_t, lane)
        b.place_block(fx, y, fz, _frame_plank(b))
        b.fill_region(fx, y + 1, fz, fx, y + 3, fz, "air")
        for i in range(r):
            tx, tz = F.xz(first + s * i, lane)
            b.place_block(tx, y + 1 + i, tz, "air")
        lx, lz = F.xz(land_t, lane)
        b.fill_region(lx, y + r, lz, lx, y + r + 3, lz, "air")
        sx, sz = F.xz(first, lane)
        rep_ = b.flight(F.label, sx, sz, y, y + r, F.along_name(s),
                        mat=b.voice["footing"])
        reps.append(rep_)
        if not rep_.get("ok"):
            return {"ok": False, "reason": f"flight {k}: {rep_.get('reason')}",
                    "flights": reps}
        other = lanes[(k + 1) % 2]
        ox, oz = F.xz(land_t, other)
        b.place_block(ox, y + r, oz, _frame_plank(b))
        b.fill_region(ox, y + r + 1, oz, ox, y + r + 3, oz, "air")
        foot_t, y, s = land_t, y + r, -s
    # the way into the chamber is beside the top landing
    p["stair_top"] = foot_t
    p["door_t"] = foot_t
    p["stair"] = {"ok": True, "flights": reps,
                  "reason": f"{len(runs)} flights of {runs} in the flank's two lanes"}
    return p["stair"]


def _ground(b, F, p):
    """The piers, the road through them, and the head of the arch."""
    fy = F.floor_y
    head = p["head_y"]
    foot_h = 1 + (p["rng"].random() < 0.5)
    p["foot_h"] = foot_h
    ts = p["pier_ts"]
    ta, tb = ts[0], ts[-1]
    pc = p["pier_c"]

    # the solid flank: cobblestone footing, then frame and render above it
    for t in ts:
        x, z = F.xz(t, pc)
        for y in range(fy + 1, fy + 1 + foot_h):
            b.place_block(x, y, z, _foot(b) if (t + y) % 5 else _foot_mossy(b))
        post = t in (ta, tb)
        for y in range(fy + 1 + foot_h, head + 1):
            if post:
                b.place_block(x, y, z, _post(b, "y"))
            else:
                b.place_block(x, y, z, _panel_block(b, p, t, pc, y))

    # the head of the arch: voussoirs shoulder the tunnel in from both edges
    span = [F.cc - 1, F.cc + 1]
    for t in ts:
        for c in span:
            if not F.inside(t, c):
                continue
            x, z = F.xz(t, c)
            b.place_block(x, head, z, _foot_slab(b) + "[type=top]")

    # where the gate oversails the lane there is no pier to stand on: the storey above
    # comes out on a lintel and a course of joist ends
    out = [t for t in range(F.t0, F.t1 + 1) if t not in ts]
    for t in out:
        near_pier = (t == ta - 1 or t == tb + 1)
        for c in range(F.c0, F.c1 + 1):
            if not (near_pier or c in (F.c0, F.c1)):
                continue
            x, z = F.xz(t, c)
            _bracket(b, x, head, z, F.along_name(1 if t > tb else -1))

    # keep the road itself open, three wide, all the way through
    for t in range(F.t0, F.t1 + 1):
        for c in range(F.cc - 1, F.cc + 2):
            x, z = F.xz(t, c)
            b.fill_region(x, fy + 1, z, x, head - 1, z, "air")
    for t in ts:
        xm, zm = F.xz(t, F.cc)
        b.place_block(xm, head, zm, "air")
    return p


def _deck(b, F, p, y, t_lo, t_hi, c_lo, c_hi):
    """A storey's floor, laid across the whole footprint of that storey."""
    for t in range(t_lo, t_hi + 1):
        for c in range(c_lo, c_hi + 1):
            x, z = F.xz(t, c)
            edge = t in (t_lo, t_hi) or c in (c_lo, c_hi)
            b.place_block(x, y, z, _trim(b) if edge and p["banded"] else _frame_plank(b))


def _flight(b, F, p):
    """The one long flight: off the road, up the flank, onto the walk."""
    fy = F.floor_y
    sc = p["stair_c"]
    s = p["climb"]
    start = p["stair_start"]
    y1 = p["floors"][0]
    x, z = F.xz(start, sc)
    r = b.flight(F.label, x, z, fy, y1, F.along_name(s), mat=b.voice["footing"])
    # the flight is carried, not floating: pack the wedge under the treads
    for k in range(y1 - fy):
        t = start + s * k
        if (t, sc) in p["lane"]:
            continue                      # the lane goes under, not through
        xx, zz = F.xz(t, sc)
        for y in range(fy + 1, fy + 1 + k):
            if b.get_block(xx, y, zz) == "air":
                b.place_block(xx, y, zz, _foot(b) if (t + y) % 4 else _foot_mossy(b))
    p["stair"] = r
    return r


def _posts_along(lo, hi):
    """Where the frame posts stand in a run: the ends, then every other cell."""
    n = hi - lo
    step = 2 if n <= 6 else 3
    cells = set([lo, hi])
    k = lo
    while k <= hi:
        cells.add(k)
        k += step
    cells.add(hi)
    return cells


def _shell(b, F, p):
    """The chamber over the arch and the stages of tower above it:
    dark frame, pale render between, a light in every panel of every stage."""
    fy = p["floors"][0]
    base, top = fy + 1, p["wall_top"]
    t_lo, t_hi = F.t0, F.t1
    c_lo, c_hi = p["room_c"]
    tposts = _posts_along(t_lo, t_hi)
    cposts = _posts_along(c_lo, c_hi)
    band_axis = "x" if F.axis == "x" else "z"
    door_c, door_t = p["door_c"], p["door_t"]
    for t in range(t_lo, t_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if not (t in (t_lo, t_hi) or c in (c_lo, c_hi)):
                continue
            end_face = t in (t_lo, t_hi)
            corner = end_face and c in (c_lo, c_hi)
            post = corner or (end_face and c in cposts) or (
                (not end_face) and t in tposts)
            x, z = F.xz(t, c)
            for y in range(base, top + 1):
                k = (y - base) % p["stage_h"]
                stage = (y - base) // p["stage_h"]
                if c == door_c and t == door_t and y in (base, base + 1):
                    b.place_block(x, y, z, "air")   # the way in off the stair
                    continue
                oriel_face = (t in p["oriel_ends"] and c_lo < c < c_hi
                              and stage == 0 and k in (0, 1))
                if post:
                    b.place_block(x, y, z, _post(b, "y"))
                elif oriel_face:
                    b.place_block(x, y, z, _panel_block(b, p, t, c, y))
                elif k == p["stage_h"] - 1:
                    ax = band_axis if not end_face else (
                        "z" if band_axis == "x" else "x")
                    b.place_block(x, y, z, _trim(b) + "[axis=" + ax + "]")
                elif k == 1:
                    b.place_block(x, y, z, PANE)
                else:
                    b.place_block(x, y, z, _panel_block(b, p, t, c, y))
    return {"base": base, "top": top, "c_lo": c_lo, "c_hi": c_hi}


def _panel_block(b, p, t, c, y):
    """The wall material, with a little grain so a wall is not one flat colour."""
    r = (t * 31 + c * 17 + y * 7) % 11
    if r == 0:
        return _wall_alt(b)
    if r == 1:
        return _wall_fine(b)
    return _wall(b)


def _oriel(b, F, p):
    """The signature: the first floor carried out over the entrance on
    brackets, and glazed into a bay."""
    base = p["floors"][0] + 1
    cells = p["oriel_cells"]
    laid = []
    for e in p["oriel_ends"]:
        if not cells:
            continue
        glaze = GLASS if len(cells) > 1 else PANE
        out = F.along_name(-1 if e == F.t0 else 1)
        for c in cells:
            x, z = F.xz(e, c)
            b.place_block(x, base, z, glaze)
            b.place_block(x, base + 1, z, glaze)
            if b.get_block(x, p["head_y"], z) == "air":
                _bracket(b, x, p["head_y"], z, out)
        laid.append((e, tuple(cells)))
    p["oriels"] = laid
    return laid


def _door(b, F, p):
    """A real leaf at the head of the stair: the one way into the chamber."""
    x, z = F.xz(p["door_t"], p["door_c"])
    stand = p["floors"][0] + 1
    facing = F.cross_name(-p["stair_side"])
    r = b.doorway(x, stand, z, facing, b.voice["frame"],
                  leaf=b.joinery(b.voice, "door"), jamb="build", lintel=True)
    p["door"] = r
    p["door_check"] = b.check_door(x, stand, z)
    return r


def _crown(b, F, p):
    """One roof over chamber, stair and all: steep, deep-eaved, busy."""
    eave_y = p["wall_top"] + 1
    if p["ridge_axis"] == "t":
        axis = "z" if F.axis == "x" else "x"
    else:
        axis = "x" if F.axis == "x" else "z"
    ridge_y = b.roof(F.x0 + 1, F.z0 + 1, F.x1 - 1, F.z1 - 1, eave_y,
                     _roof_family(b), style=p["crown"], axis=axis,
                     pitch=p["pitch"], overhang=1, eave=p["eave"])
    p["ridge_y"] = ridge_y
    return ridge_y


def _lights(b, F, p):
    """Torches on the pier, inside the arch, where the road is darkest."""
    side = 1 if p["pier_c"] < F.cc else -1
    face = F.cross_name(side)
    c = p["pier_c"] + side
    for t in (F.tc - 1, F.tc + 1):
        if not F.inside(t, c):
            continue
        x, z = F.xz(t, c)
        if b.get_block(x, F.floor_y + 2, z) == "air":
            b.place_block(x, F.floor_y + 2, z,
                          "wall_torch[facing=" + face + "]")


def _furnish(b, F, p):
    """The chamber over the arch is a guard room. Furnish it as one."""
    fy = p["floors"][0]
    y = fy + 1
    c_lo, c_hi = p["room_c"]
    rng = p["rng"]
    room = set((t, c) for t in range(F.t0 + 1, F.t1)
               for c in range(c_lo + 1, c_hi))
    inside = (p["door_t"], p["door_c"] - p["stair_side"])
    if inside not in room and room:
        inside = sorted(room)[0]
    taken = set([inside])            # the cell you land in is left clear
    order = sorted(room - taken)
    rng.shuffle(order)
    done = []

    def still_open(cell):
        """Would blocking this cell strand any of the floor that is left?"""
        left = room - taken - set([cell])
        seen, queue = set([inside]), [inside]
        while queue:
            t, c = queue.pop()
            for n in ((t + 1, c), (t - 1, c), (t, c + 1), (t, c - 1)):
                if n in left and n not in seen:
                    seen.add(n)
                    queue.append(n)
        return left <= seen

    def put(kind, facing=None, **kw):
        for cell in list(order):
            if cell in taken or not still_open(cell):
                continue
            t, c = cell
            x, z = F.xz(t, c)
            f = facing or F.cross_name(1 if c < F.cc else -1)
            r = b.fitting(kind, x, y, z, f, **kw)
            if r.get("ok"):
                taken.add(cell)
                order.remove(cell)
                for got in (r.get("cells") or []):
                    if len(got) >= 3:
                        for other in list(order):
                            ox, oz = F.xz(other[0], other[1])
                            if got[0] == ox and got[2] == oz:
                                taken.add(other)
                                order.remove(other)
                done.append(kind)
                return r
        return {"ok": False, "reason": "no cell for " + kind}

    put("light", room="hall")
    for kind in rng.sample(["table", "bench", "store", "shelf"], 2):
        put(kind, mat="dark_oak", room="hall")

    # and the gate passage itself is a room: it is lit, and it is watched
    y0 = F.floor_y + 1
    hung = 0
    for t in p["pier_ts"]:
        for c in (F.cc - 1, F.cc + 1):
            if hung >= 2 or (t, c) in p["lane"]:
                continue
            x, z = F.xz(t, c)
            if b.fitting("light", x, y0, z, F.facing, room="hall").get("ok"):
                hung += 1
                done.append("gate light")
    p["fittings"] = done
    return done


def build(b, part, seed, **params):
    F = Frame(part)
    p = _plan(b, F, seed, params.get("storeys", 2),
              params.get("crown", "gable"))

    _clear(b, F, p)
    _ground(b, F, p)
    if p["tall"]:
        _base(b, F, p)

    fy0 = p["floors"][0]
    c_lo, c_hi = p["room_c"]
    _deck(b, F, p, fy0, F.t0, F.t1, c_lo, c_hi)
    if p["tall"]:
        _mural_flight(b, F, p)
    else:
        lx, lz = F.xz(p["stair_top"], p["stair_c"])
        b.place_block(lx, fy0, lz, _frame_plank(b))
        _flight(b, F, p)
    _shell(b, F, p)
    _door(b, F, p)
    _oriel(b, F, p)
    _crown(b, F, p)
    _lights(b, F, p)
    _furnish(b, F, p)

    b.seal_voids(F.x0, F.z0, F.x1, F.z1, _wall(b))
    att = b.check_attached()
    walk = b.check_walkable(F.label)
    return {"stair": p.get("stair"), "ridge_y": p.get("ridge_y"),
            "door": p.get("door"), "door_check": p.get("door_check"),
            "attached": att.get("ok"), "walkable": walk.get("ok"),
            "fittings": p.get("fittings")}
