"""A small courtyard house: three ranges and a wall round a tight yard.

The ordinary house of a crowded city -- the same kind of thing as a great
courtyard house and a good deal less of it. The lane side is a wall with the
gate through it; the other three sides are ranges, timber-framed and boarded,
open to the yard under deep upturned eaves. Every room is entered off the yard,
so every room is reachable on foot from the gate.
"""

import random

KIND = "plot"
FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "urban"

PARAMS = {
    "storeys": ("int", 1, 2),
    "wings": ("choice", ["open", "screened", "closed"]),
}

NEEDS = {
    # Cut to the band `scripts/type_needs.py` measured -- measured 5x5 to 9x9 by the
    # sweep; declared to 24 by its author.
    "footprint": (4, 4, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

WALL_H = 3          # boarded wall: floor+1 .. floor+3
STOREY = 4          # floor to floor

# Local frame. f runs from the lane wall (f=0) inward; t runs across it. "in" is the way
# you face walking in off the lane, "out" is back at the lane.
DIRS = {
    "north": {"in": "north", "out": "south", "tp": "east", "tm": "west", "tx": True},
    "south": {"in": "south", "out": "north", "tp": "east", "tm": "west", "tx": True},
    "east": {"in": "east", "out": "west", "tp": "south", "tm": "north", "tx": False},
    "west": {"in": "west", "out": "east", "tp": "south", "tm": "north", "tx": False},
}


def _facing(part):
    """Which way you walk in off the lane, taken from the reserved doorstep."""
    dx, dz = part["door"][0], part["door"][1]
    cands = []
    if dx == part["x0"]:
        cands.append("east")
    if dx == part["x1"]:
        cands.append("west")
    if dz == part["z0"]:
        cands.append("south")
    if dz == part["z1"]:
        cands.append("north")
    said = part.get("facing")
    if said in cands:
        return said
    if cands:
        return cands[0]
    return said if said in DIRS else "north"


def _span(part, facing):
    """(across, deep) of the pad in the local frame."""
    w = part["x1"] - part["x0"] + 1
    d = part["z1"] - part["z0"] + 1
    if DIRS[facing]["tx"]:
        return w, d
    return d, w


def _wxz(part, facing, t, f):
    if facing == "north":
        return part["x0"] + t, part["z1"] - f
    if facing == "south":
        return part["x0"] + t, part["z0"] + f
    if facing == "east":
        return part["x0"] + f, part["z0"] + t
    return part["x1"] - f, part["z0"] + t


def _box(part, facing, t0, f0, t1, f1):
    ax, az = _wxz(part, facing, t0, f0)
    bx, bz = _wxz(part, facing, t1, f1)
    return min(ax, bx), min(az, bz), max(ax, bx), max(az, bz)


def _t_of(part, facing, x, z):
    if DIRS[facing]["tx"]:
        return x - part["x0"]
    return z - part["z0"]


def _f_of(part, facing, x, z):
    if facing == "north":
        return part["z1"] - z
    if facing == "south":
        return z - part["z0"]
    if facing == "east":
        return x - part["x0"]
    return part["x1"] - x


def _mats(b, voice):
    return {
        "wall": b.block(voice["wall"], "full"),
        "frame": b.block(voice["frame"], "full"),
        "foot": b.block(voice["footing"], "full"),
        "floor": b.block(voice["floor"], "full"),
        "trim": b.block(voice["trim"], "full"),
        "trim_slab": b.block(voice["trim"], "slab"),
        "fence": b.joinery(voice, "fence"),
        "leaf": b.joinery(voice, "door"),
    }


def _roof_spec(part):
    spec = part.get("roof") or {}
    if not isinstance(spec, dict):
        spec = {}
    return {
        "profile": spec.get("profile") or [(1, 2), (2, 1)],
        "ends": spec.get("ends") or ("irimoya", "irimoya"),
        "eave": spec.get("eave") or "upturned",
        "tiers": spec.get("tiers") or 1,
    }

def _flood(cells, start):
    """Which of `cells` you can walk to from `start`, four ways, on the flat."""
    seen, stack = set(), [start]
    while stack:
        c = stack.pop()
        if c in seen or c not in cells:
            continue
        seen.add(c)
        stack += [(c[0] + 1, c[1]), (c[0] - 1, c[1]),
                  (c[0], c[1] + 1), (c[0], c[1] - 1)]
    return seen


def build(b, part, seed, **params):
    rnd = random.Random(seed * 7919 + 17)
    voice = part["voice"]
    m = _mats(b, voice)
    fy = part["floor_y"]
    facing = _facing(part)
    d = DIRS[facing]
    W, D = _span(part, facing)
    rs = _roof_spec(part)

    storeys = int(params.get("storeys") or 1)
    storeys = max(1, min(2, storeys))
    wings_mode = params.get("wings") or "open"
    if wings_mode not in ("open", "screened", "closed"):
        wings_mode = "open"

    axis_t = "z" if d["tx"] else "x"      # ridge running across the front
    axis_f = "x" if d["tx"] else "z"      # ridge running front to back

    def P(t, f, y, block):
        x, z = _wxz(part, facing, t, f)
        b.place_block(x, y, z, block)

    def C(t0, f0, t1, f1, y0, y1, block):
        x0, z0, x1, z1 = _box(part, facing, t0, f0, t1, f1)
        b.place_cuboid(x0, y0, z0, x1, y1, z1, block)

    def cells_of(t0, f0, t1, f1):
        if t0 == t1:
            return [(t0, f) for f in range(min(f0, f1), max(f0, f1) + 1)]
        return [(t, f0) for t in range(min(t0, t1), max(t0, t1) + 1)]

    # ---- massing ----------------------------------------------------------
    db_lo = max(3, min(4, D // 5))                 # depth of the back range
    if storeys >= 2:
        db_lo = max(db_lo, 4)
    db_hi = max(db_lo, min(3 + (D - 6) // 3, D - 3, 7))
    db = rnd.randint(db_lo, db_hi)
    db = max(3, min(db, D - 2))
    fb0 = D - db                                   # its front line

    ws_hi = max(2, min(4, (W - 2) // 2))           # width of each wing
    ws = rnd.randint(2, ws_hi)
    while ws > 1 and W - 2 * ws < 2:
        ws -= 1

    td = _t_of(part, facing, part["door"][0], part["door"][1])
    td = max(0, min(W - 1, td))
    gate_low = td <= (W - 1) / 2.0

    # the way in: gate, then straight on and along to the yard. Nothing -- no post, no
    # rail, no lamp -- is ever put in these cells.
    ynear = ws if td < ws else (W - ws - 1 if td > W - ws - 1 else td)
    corridor = set((t, 1) for t in
                   range(min(td, ynear), max(td, ynear) + 1))
    keep_clear = set(corridor)
    keep_clear.add((td, 0))
    keep_clear.add((ynear, 2))

    bay_off = rnd.choice([-1, 0, 0, 1]) if W >= 11 else 0
    wing_from = rnd.choice([0, 1])          # does the wing roof take in the wall
    wing_pitch = rnd.choice([(1, 2), (1, 3)])
    hall_ends = rnd.choice([rs["ends"], rs["ends"], ("hip", "hip")])
    lane_spacing = rnd.choice([3, 4])
    wing_f0, wing_f1 = 1, fb0 - 1
    wing_mid = wing_f0 + (wing_f1 - wing_f0) // 2
    for wt in (ws - 1, W - ws):             # a way into each gallery, always
        corridor.add((wt, wing_mid))
        if wing_mid + 1 <= wing_f1:
            corridor.add((wt, wing_mid + 1))
    two = (storeys >= 2 and W >= 8 and db >= 4)
    hall_top = fy + WALL_H
    hall_eave = fy + STOREY + WALL_H if two else fy + WALL_H + 1
    wing_top = fy + WALL_H
    wing_eave = fy + WALL_H + 1

    # ---- the floor: boarded under the ranges, paved in the yard -----------
    b.place_cuboid(part["x0"], fy, part["z0"], part["x1"], fy, part["z1"], m["foot"])
    C(0, fb0, W - 1, D - 1, fy, fy, m["floor"])
    C(0, wing_f0, ws - 1, wing_f1, fy, fy, m["floor"])
    C(W - ws, wing_f0, W - 1, wing_f1, fy, fy, m["floor"])

    # the engawa: boards along the yard face of the hall, under its eave
    eng_t0, eng_t1 = max(0, ws - 1), min(W - 1, W - ws)
    C(eng_t0, fb0 - 1, eng_t1, fb0 - 1, fy, fy, m["trim"])
    eng_side = "low" if rnd.random() < 0.5 else "high"
    if wing_f1 >= wing_f0 + 1:
        if eng_side == "low":
            C(ws, wing_f0, ws, wing_f1, fy, fy, m["trim"])
        else:
            C(W - ws - 1, wing_f0, W - ws - 1, wing_f1, fy, fy, m["trim"])

    # ---- timber runs ------------------------------------------------------
    def timber_run(t0, f0, t1, f1, y0, ytop, mode="solid", gap=()):
        cs = cells_of(t0, f0, t1, f1)
        n = len(cs)
        gap = set(gap)
        for i, (t, f) in enumerate(cs):
            is_post = (i == 0 or i == n - 1 or i % 3 == 0)
            if (t, f) in corridor or (t, f) in gap:
                P(t, f, ytop, m["frame"])
                continue
            if is_post:
                C(t, f, t, f, y0, ytop, m["frame"])
            elif mode == "solid":
                C(t, f, t, f, y0, ytop - 1, m["wall"])
                P(t, f, ytop, m["frame"])
            elif mode == "rail":
                P(t, f, y0, m["fence"])
                P(t, f, ytop, m["frame"])
            else:
                P(t, f, ytop, m["frame"])

    def mid_gap(cs, n=2):
        if len(cs) <= 2:
            return []
        i = len(cs) // 2
        out = [cs[i]]
        if n > 1 and i - 1 >= 1:
            out.append(cs[i - 1])
        elif n > 1 and i + 1 <= len(cs) - 2:
            out.append(cs[i + 1])
        return out

    # ---- the lane wall, and the gate through it ---------------------------
    lx0, lz0, lx1, lz1 = _box(part, facing, 0, 0, W - 1, 0)
    b.wall(lx0, fy + 1, lz0, lx1, fy + WALL_H, lz1, m["wall"],
           post=m["frame"], spacing=lane_spacing, base=m["foot"], base_height=1,
           band=m["trim_slab"])
    gx, gz = _wxz(part, facing, td, 0)
    b.doorway(gx, fy + 1, gz, d["in"], voice["frame"], leaf=m["leaf"],
              jamb="build", lintel=True)

    # ---- the two wings ----------------------------------------------------
    gate_side = "low" if gate_low else "high"
    wing_specs = []
    for side in ("low", "high"):
        if side == "low":
            wt0, wt1, yard_t = 0, ws - 1, ws - 1
        else:
            wt0, wt1, yard_t = W - ws, W - 1, W - ws
        if side == gate_side or ws < 3:
            mode = "post"
        elif wings_mode == "screened":
            mode = "rail"
        elif wings_mode == "closed":
            mode = "solid"
        else:
            mode = "post"
        wing_specs.append((side, wt0, wt1, yard_t, mode))

    for side, wt0, wt1, yard_t, mode in wing_specs:
        outer_t = wt0 if side == "low" else wt1
        timber_run(outer_t, wing_f0, outer_t, wing_f1, fy + 1, wing_top, "solid")
        cs = cells_of(yard_t, wing_f0, yard_t, wing_f1)
        gap = mid_gap(cs, 2) if mode in ("rail", "solid") else []
        timber_run(yard_t, wing_f0, yard_t, wing_f1, fy + 1, wing_top, mode, gap)

    # ---- the hall across the back ----------------------------------------
    timber_run(0, D - 1, W - 1, D - 1, fy + 1, hall_top, "solid")
    timber_run(0, fb0, 0, D - 1, fy + 1, hall_top, "solid")
    timber_run(W - 1, fb0, W - 1, D - 1, fy + 1, hall_top, "solid")
    front_cs = cells_of(0, fb0, W - 1, fb0)
    bay_w = rnd.choice([2, 3]) if W >= 9 else 2
    i = len(front_cs) // 2 + bay_off
    bay = [front_cs[j] for j in range(i - bay_w // 2, i - bay_w // 2 + bay_w)
           if 1 <= j <= len(front_cs) - 2]
    if not bay:
        bay = mid_gap(front_cs, 2)
    timber_run(0, fb0, W - 1, fb0, fy + 1, hall_top, "solid", bay)
    for (bt, bf) in bay:                      # the way from the yard into it
        keep_clear.add((bt, bf))
        keep_clear.add((bt, bf - 1))
        keep_clear.add((bt, bf + 1))

    # ---- an upper storey over the hall, where the hall is long enough -----
    stair_run = set()
    land_cell = None
    foot_cell = None
    if two:
        C(0, fb0, W - 1, D - 1, fy + STOREY, fy + STOREY, m["floor"])
        laid = False
        for ts in (2, 3, W - 6, W - 7):
            if laid or ts < 2 or ts + 4 > W - 2:
                continue
            sx, sz = _wxz(part, facing, ts, D - 2)
            got = b.flight(part["label"], sx, sz, fy, fy + STOREY, d["tp"],
                           mat=voice["floor"])
            if got and got.get("ok"):
                laid = True
                land_cell = (ts + 4, D - 2)
                foot_cell = (ts - 1, D - 2)
                stair_run = set((st, D - 2) for st in range(ts, ts + 4))
                keep_clear.add(foot_cell)
                keep_clear.add(land_cell)
        if laid:
            up0, up1 = fy + STOREY + 1, fy + STOREY + WALL_H
            timber_run(0, D - 1, W - 1, D - 1, up0, up1, "solid")
            timber_run(0, fb0, 0, D - 1, up0, up1, "solid")
            timber_run(W - 1, fb0, W - 1, D - 1, up0, up1, "solid")
            timber_run(0, fb0, W - 1, fb0, up0, up1, "solid")
        else:
            # no flight would go: one storey under its own roof, not a chamber nobody
            # can reach.
            two = False
            hall_eave = fy + WALL_H + 1

    # ---- roofs: the biggest thing about the building ----------------------
    for side, wt0, wt1, yard_t, mode in wing_specs:
        rx0, rz0, rx1, rz1 = _box(part, facing, wt0, wing_from, wt1, fb0 - 1)
        b.roof(rx0, rz0, rx1, rz1, wing_eave, voice["roof"], style="hip",
               axis=axis_f, pitch=wing_pitch, overhang=1, eave=rs["eave"])

    tiers = 1
    if db >= 5 and W >= 9:
        tiers = 2 if ((rs["tiers"] or 1) > 1 or rnd.random() < 0.5) else 1
    hx0, hz0, hx1, hz1 = _box(part, facing, 0, fb0, W - 1, D - 1)
    b.roof(hx0, hz0, hx1, hz1, hall_eave, voice["roof"], style="hip",
           axis=axis_t, profile=rs["profile"], ends=hall_ends,
           eave=rs["eave"], tiers=tiers, overhang=1)

    if two:
        # where the wing roofs run into a two-storey hall they leave a gutter under its
        # eave: carry the boarding up and close it.
        for gt in list(range(0, ws + 1)) + list(range(W - ws - 1, W)):
            if 0 <= gt < W:
                C(gt, fb0 - 1, gt, fb0 - 1, wing_eave + 1, hall_eave - 1,
                  m["wall"])

    if ws <= td <= W - ws - 1:
        g0 = max(0, td - 1)
        g1 = min(W - 1, td + 1)
        qx0, qz0, qx1, qz1 = _box(part, facing, g0, 0, g1, 0)
        # The slope is the voice's (E015): `pitch` was (1, 2) here until the voice
        # contract, and `roof()` is handed the voice's profile over whatever is passed.
        b.roof(qx0, qz0, qx1, qz1, fy + WALL_H + 1, voice["roof"], style="gable",
               axis=axis_t, overhang=1, eave=rs["eave"])

    # ---- openings: wide, low and screened, on the outward faces ----------
    def face_openings(t0, f0, t1, f1, y, spacing=4):
        cs = cells_of(t0, f0, t1, f1)
        if len(cs) < 4:
            return
        x0, z0, x1, z1 = _box(part, facing, t0, f0, t1, f1)
        b.openings(x0, y, z0, x1, z1, spacing=spacing, width=2, sill=1,
                   head=2, block=m["fence"])

    face_openings(1, D - 1, W - 2, D - 1, fy)
    face_openings(0, wing_f0, 0, wing_f1, fy)
    face_openings(W - 1, wing_f0, W - 1, wing_f1, fy)
    face_openings(0, fb0 + 1, 0, D - 2, fy)
    face_openings(W - 1, fb0 + 1, W - 1, D - 2, fy)
    if two:
        up = fy + STOREY
        face_openings(1, D - 1, W - 2, D - 1, up, spacing=3)
        face_openings(0, fb0 + 1, 0, D - 2, up, spacing=3)
        face_openings(W - 1, fb0 + 1, W - 1, D - 2, up, spacing=3)

    # ---- the engawa's own posts, carrying the eave over the boards -------
    if eng_t1 > eng_t0:
        for et in (eng_t0, eng_t1):
            if (et, fb0 - 1) in corridor or (et, fb0 - 1) in keep_clear:
                continue
            P(et, fb0 - 1, fy + 1, m["frame"])
            P(et, fb0 - 1, fy + 2, m["frame"])
        if not two:      # under a two-storey hall the beam would be a ledge
            C(eng_t0, fb0 - 1, eng_t1, fb0 - 1, fy + WALL_H, fy + WALL_H,
              m["frame"])

    # ---- what the rooms are for ------------------------------------------
    stand = fy + 1
    din, dout = d["in"], d["out"]
    hall_t0, hall_t1 = 1, W - 2
    hall_f0, hall_f1 = fb0 + 1, D - 2
    room_hall = "hall" if (hall_t1 - hall_t0) >= 4 else "house"

    def fitf(kind, t, f, y, face, fam, room, extent=1):
        if not (0 <= t < W and 0 <= f < D):
            return None
        if (t, f) in keep_clear or (t, f) in corridor:
            return None
        x, z = _wxz(part, facing, t, f)
        return b.fitting(kind, x, y, z, face, mat=fam, extent=extent, room=room)

    # Where a piece may stand: never a cell whose loss would seal off the corner behind
    # it. The row nearest the yard stays clear, as the aisle.
    slots = []
    if hall_f1 > hall_f0:
        slots += [(t, hall_f1) for t in range(hall_t0, hall_t1 + 1)]
        for ff in range(hall_f0 + 1, hall_f1):
            slots += [(hall_t0, ff), (hall_t1, ff)]
    elif hall_t1 > hall_t0:
        slots = [(hall_t0, hall_f0), (hall_t1, hall_f0)]
    # nothing may stand where its own bulk could close the way to the stair or the bay:
    # a hearth is three cells wide and spreads from its anchor.
    guard = set()
    for c in [land_cell, foot_cell] + list(bay):
        if c is None:
            continue
        for dt in (-1, 0, 1):
            for df in (-1, 0, 1):
                guard.add((c[0] + dt, c[1] + df))
    slots = [c for c in slots
             if c not in keep_clear and c not in corridor and c not in guard]
    slots = slots[::2] if len(slots) > 3 else slots

    def slot_face(c):
        if c[1] == hall_f1:
            return dout
        return d["tp"] if c[0] == hall_t0 else d["tm"]

    floor_cells = set((t, f) for t in range(hall_t0, hall_t1 + 1)
                      for f in range(hall_f0, hall_f1 + 1))

    def furnish(plan, y, room, free, entry, where=None):
        """Put pieces on the slots, but never one that seals a cell off."""
        left = list(plan)
        for c in (slots if where is None else where):
            if not left or c not in free:
                continue
            trial = free - {c}
            if entry not in trial or _flood(trial, entry) != trial:
                continue
            kind, fam = left[0]
            r = fitf(kind, c[0], c[1], y, slot_face(c), fam, room)
            if r is None:
                continue
            if r.get("ok"):
                used = {c}
                for cc in (r.get("cells") or []):
                    if isinstance(cc, (list, tuple)) and len(cc) >= 3:
                        used.add((_t_of(part, facing, cc[0], cc[2]),
                                  _f_of(part, facing, cc[0], cc[2])))
                free = free - used
                left.pop(0)
        return free

    plan = [("hearth", voice["footing"]), ("table", voice["floor"]),
            ("light", voice["trim"])]
    if not two:
        plan.append(("bed", voice["floor"]))
    plan.append(("bench", voice["floor"]))
    ground_free = floor_cells - stair_run
    ground_entry = (bay[len(bay) // 2][0], hall_f0)
    if ground_entry in ground_free:
        rest = furnish(plan, stand, room_hall, ground_free, ground_entry)
        if rest == ground_free:      # a hall too tight for the slot rule
            spare = [c for c in sorted(floor_cells - stair_run)
                     if c not in keep_clear and c not in corridor]
            furnish([("light", voice["trim"]), ("bench", voice["floor"])],
                    stand, room_hall, ground_free, ground_entry, where=spare)
    if two and land_cell is not None:      # the chamber over the hall
        up_free = floor_cells - stair_run
        if land_cell in up_free:
            furnish([("bed", voice["floor"]), ("store", voice["floor"]),
                     ("light", voice["trim"])],
                    fy + STOREY + 1, "house", up_free, land_cell)

    for side, wt0, wt1, yard_t, mode in wing_specs:
        # a 1-wide gallery is a way through: one piece at the far end, no more
        inner_t = wt0 if side == "low" else wt1
        inner_t = inner_t + 1 if side == "low" else inner_t - 1
        inner_t = max(wt0, min(wt1, inner_t))
        deep = (ws >= 4)
        face = d["tp"] if side == "low" else d["tm"]
        if side == gate_side:
            fitf("bench", inner_t, wing_f1, stand, face, voice["floor"], "house")
            if deep:
                fitf("shelf", inner_t, wing_f0, stand, face, voice["floor"],
                     "house")
        else:
            fitf("store", inner_t, wing_f1, stand, face, voice["floor"],
                 "store", extent=2 if deep else 1)
            if deep:
                fitf("light", inner_t, wing_f0, stand, face, voice["trim"],
                     "store")

    # ---- the yard ---------------------------------------------------------
    yt0, yt1 = ws, W - ws - 1
    yf0, yf1 = wing_f0, fb0 - 2
    if yt1 - yt0 >= 4 and yf1 - yf0 >= 4:
        # room for a wellhead in the middle of the court with a way round it
        fitf("well", (yt0 + yt1) // 2, yf0 + 2 + (yf1 - yf0 - 4) // 2, stand,
             dout, voice["footing"], "yard")
    elif yt1 - yt0 >= 2 and yf1 - yf0 >= 2:
        # a stone lantern in the corner of the court, never in its middle
        fitf("light", yt1, yf1, stand, dout, voice["trim"], "yard")

    # ---- checking my own work --------------------------------------
    b.seal_voids(part["x0"], part["z0"], part["x1"], part["z1"], m["wall"],
                 max_cells=256)
    b.check_door(gx, fy + 1, gz)
    b.check_walkable(part["label"])
    b.check_attached()
    return {"kind": "court_small", "storeys": storeys, "wings": wings_mode}
