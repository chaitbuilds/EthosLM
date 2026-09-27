"""A flat-roofed masonry house: a cube of rooms whose roof is a terrace.

The dry-country dwelling found from the Maghreb to the Pueblos and along every warm
shore: thick walls, few and small openings, a flat roof behind a parapet that is used
as a floor -- for sleeping, drying and sitting out -- reached by a stair, and shade
made of timber rather than of roof. It stands free or in a run: the flanks carry no
eave and no overhang, so a neighbour can stand against either of them and the two
parapets meet.

What it lays, all of it measured on the result rather than claimed:

  - the shell through `building()` with a `flat` roof and no overhang, one or two
    storeys, furnished;
  - a **parapet** round the roof deck, one course of the wall, broken only where the
    stair lands;
  - the **way up**: an outside stair of solid masonry against the back wall where a
    rear court is long enough to take it (the voice's "external stair climbing the
    outside of the building"), otherwise an inside flight up through a hatch;
  - a **shade**: a timber pergola standing on the roof terrace, an awning of slats
    over the front door, or -- where the plan says the lane is shaded -- slatted beams
    carried out from the front wall at the head of the ground storey across the lane
    to its middle, so the houses either side of a narrow lane roof it between them;
  - a **rear court** walled to the height of a person where the lot is deep enough,
    planted or paved.

Everything the lot could not hold is dropped by name in `emitted`, never replaced with
something else. The roof terrace and the pergola publish their rectangles and the level
they are laid at, so `construction.outcome` checks them where they stand.
"""

import random

KIND = "plot"
FORM = "dryland_vernacular"
ROLE = "urban"

#: What this type is for: a house people live in.
FUNCTION = "dwelling"

#: The flanks are blank party walls with no eave, so the plan may stand this house in a
#: run with a neighbour against either flank.
ATTACHED = True

#: What this house can be asked for, and is checked against once built.
FEATURES = ("flat_roof", "roof_terrace", "canopy", "courtyard")

PARAMS = {
    "storeys": ("int", 1, 2),
    "court": ("choice", ["rear", "none"]),
    "shade": ("choice", ["pergola", "awning", "lane"]),
    # how far a `lane` shade reaches out from the front wall: the plan's word, from the
    # width of the lane the house fronts (to its middle, and one over it)
    "reach": ("int", 0, 4),
}

NEEDS = {
    # The pad `site()` hands build(), after the inset. Five across is the least that
    # holds a room beside a door; twenty deep is a house and its court. Measured by
    # `construction.probe_build` over 5..12 across x 5..20 deep, one and two storeys.
    "footprint": (5, 5, 12, 20),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

#: The least pad (across, deep) each storey count stands on: two storeys need a room
#: deep enough for the flight between them.
STOREY_PAD = {1: (5, 5), 2: (5, 8)}

_NAME = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
_FRONT_EDGE = {"north": ("z", 1), "south": ("z", -1), "west": ("x", 1), "east": ("x", -1)}
_FLANK_SIDES = {"z": ("west", "east"), "x": ("north", "south")}


def _front_edge(part):
    """`(axis, sign)` of the front: the plan's word where the doorstep is on it, else the
    pad edge the reserved doorstep is nearest (see `row_house._front_edge`)."""
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    said = _FRONT_EDGE.get(str(part.get("front") or "").strip().lower())
    near = {("z", 1): abs(dz - pz0), ("z", -1): abs(dz - pz1),
            ("x", 1): abs(dx - px0), ("x", -1): abs(dx - px1)}
    if said is not None and near[said] <= 1:
        return said
    runs = {"x": px1 - px0 + 1, "z": pz1 - pz0 + 1}
    return min(near, key=lambda k: (near[k], -runs[k[0]]))


def build(b, part, seed, storeys=None, court=None, shade=None, reach=None, **kw):
    rng = random.Random(int(seed) * 7907 + 4481)
    voice = part["voice"]
    fy = int(part["floor_y"])
    label = part["label"]
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    dx, dz = part["door"]
    ax, s = _front_edge(part)
    if ax == "x":
        pad_d, pad_w = px1 - px0 + 1, pz1 - pz0 + 1
        front_c, latlo, dlat = (px0 if s > 0 else px1), pz0, dz
    else:
        pad_d, pad_w = pz1 - pz0 + 1, px1 - px0 + 1
        front_c, latlo, dlat = (pz0 if s > 0 else pz1), px0, dx

    def cell(dd, tt):
        """World (x, z) of depth `dd` back from the front wall and lateral `tt`."""
        c = front_c + s * dd
        return (c, tt) if ax == "x" else (tt, c)

    def rect(d0, t0, d1, t1):
        p, q = cell(d0, t0), cell(d1, t1)
        return (min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1]))

    def lat_face(sign):
        return _NAME[(0, sign)] if ax == "x" else _NAME[(sign, 0)]

    inward = _NAME[(s, 0)] if ax == "x" else _NAME[(0, s)]

    if court not in ("rear", "none"):
        court = rng.choice(["rear", "rear", "none"])
    if shade not in ("pergola", "awning", "lane"):
        shade = rng.choice(["pergola", "pergola", "awning"])
    asked = int(storeys) if isinstance(storeys, int) else rng.choice([1, 1, 2])

    # the court takes the back of the lot where there is depth for a house and a court
    court_d = 0
    if court == "rear" and pad_d >= 10:
        court_d = max(3, min(6, pad_d // 3))
    house_d = pad_d - court_d
    w = pad_w
    top = 2 if (house_d >= STOREY_PAD[2][1] and w >= STOREY_PAD[2][0]) else 1
    st = max(1, min(asked, top))

    a = latlo
    lat_hi = latlo + w - 1
    attempts = [(house_d, st), (house_d, 1)] if st > 1 else [(house_d, 1)]
    if court_d and house_d < pad_d:
        attempts.append((pad_d, 1))              # no court: the house takes the lot
    res = None
    refused = []
    for hd, stt in attempts:
        x0, z0, x1, z1 = rect(0, a, hd - 1, lat_hi)
        res = b.building(label, x0, z0, x1, z1, stt, {"style": "flat"},
                         openings="rhythm", stair="auto", mat=voice, overhang=0)
        if res and res.get("ok"):
            if hd != house_d:
                court_d = 0
            house_d, st = hd, stt
            break
        refused.append(f"{stt} storey(s) on {hd} deep: {str((res or {}).get('reason'))[:120]}")
    if not (res and res.get("ok")):
        return dict(res or {"ok": False}, emitted={
            "requested": {"storeys": asked, "court": court, "shade": shade},
            "storeys": 0, "attempt": None, "fallback": "no shell stood",
            "omitted": ["storeys", "roof_terrace", "canopy"], "features": {}})

    eave = int(res.get("eave_y") or fy + 4 * st)
    deck = eave                     # the roof's own course; one stands on it at deck + 1
    wall_blk = b.block(voice["wall"], "full")
    trim_slab = b.block(voice["trim"], "slab")
    fence = b.joinery(voice, "fence")
    timber = fence[:-len("_fence")] if fence.endswith("_fence") else "oak"
    beam = b.block(timber, "slab")
    beam_top = beam + ("[type=top]" if "[" not in beam else "")
    omitted, fallback = [], []
    if st < asked:
        fallback.append(f"storeys {asked} -> {st}"
                        + (f" ({refused[0]})" if refused else " (the lot is too small)"))

    main = rect(0, a, house_d - 1, lat_hi)
    blocked = set()                 # roof cells the parapet, the stair and the shade keep

    # --- the parapet: one course of the wall round the deck ---
    rim = []
    for dd in range(house_d):
        for tt in (a, lat_hi):
            rim.append((dd, tt))
    for tt in range(a + 1, lat_hi):
        rim.append((0, tt))
        rim.append((house_d - 1, tt))
    for (dd, tt) in rim:
        X, Z = cell(dd, tt)
        b.place_block(X, deck + 1, Z, wall_blk)
        blocked.add((dd, tt))

    # --- the way up ---
    rise = deck - fy                # treads from the court's floor to the deck's level
    way_up = None
    if court_d >= 2 and w >= rise + 2:
        # an outside stair of solid masonry against the back wall, climbing along it
        sign = rng.choice([1, -1])
        t0 = a if sign > 0 else lat_hi
        treads = []
        for k in range(rise):
            tt = t0 + sign * k
            X, Z = cell(house_d, tt)
            y = fy + 1 + k
            for yy in range(fy + 1, y):
                b.place_block(X, yy, Z, wall_blk)
            b.place_block(X, y + 1, Z, "air")
            b.place_block(X, y + 2, Z, "air")
            treads.append((X, y, Z))
        b.steps(treads, voice["wall"], axis=("z" if ax == "x" else "x"))
        land = t0 + sign * (rise - 1)
        X, Z = cell(house_d - 1, land)
        b.place_block(X, deck + 1, Z, "air")
        b.place_block(X, deck + 2, Z, "air")
        blocked.add((house_d - 2, land))
        way_up = {"kind": "outside", "rect": list(rect(house_d, t0, house_d, land))}
    why_not = ""
    if way_up is None:
        # an inside flight from the top floor up through a hatch in the deck
        top_floor = fy + 4 * (st - 1)
        n = deck - top_floor
        why_not = "no run of the top floor is long enough for a flight"
        tries = []
        for sign in (1, -1):
            for dd in (house_d - 2, 1):
                t0 = (a + 1) if sign > 0 else (lat_hi - 1)
                if a + 1 <= t0 + sign * n <= lat_hi - 1:
                    tries.append(("lat", sign, dd, t0))
            for tt in (a + 1, lat_hi - 1):
                d0 = 1 if sign > 0 else house_d - 2
                if 1 <= d0 + sign * n <= house_d - 2:
                    tries.append(("dep", sign, tt, d0))
        for kind, sign, fixed, start in tries:
            if kind == "lat":
                X, Z = cell(fixed, start)
                facing = lat_face(sign)
                end = cell(fixed, start + sign * (n - 1))
                land = (fixed, start + sign * n)
            else:
                X, Z = cell(start, fixed)
                facing = inward if sign > 0 else _NAME[(-s, 0)] if ax == "x" \
                    else _NAME[(0, -s)]
                end = cell(start + sign * (n - 1), fixed)
                land = (start + sign * n, fixed)
            r = b.flight(label, X, Z, top_floor, deck, facing, mat=voice["floor"])
            why_not = str((r or {}).get("reason") or "")[:160]
            if isinstance(r, dict) and r.get("ok"):
                blocked.add(land)
                way_up = {"kind": "inside", "rect": [min(X, end[0]), min(Z, end[1]),
                                                      max(X, end[0]), max(Z, end[1])]}
                break
    if way_up is None:
        # a ladder in a corner of the top room, up through a hatch: the way a small
        # flat-roofed house of the dry country has always reached its roof
        top_floor = fy + 4 * (st - 1)
        for (dd, tt, face) in ((house_d - 2, a + 1, lat_face(1)),
                               (house_d - 2, lat_hi - 1, lat_face(-1)),
                               (1, a + 1, lat_face(1)), (1, lat_hi - 1, lat_face(-1))):
            if (dd, tt) in blocked or dd < 1 or not (a < tt < lat_hi):
                continue
            X, Z = cell(dd, tt)
            if any(b.get_block(X, y, Z) not in ("air", "cave_air")
                   for y in range(top_floor + 1, top_floor + 3)):
                continue
            for y in range(top_floor + 1, deck + 1):
                b.place_block(X, y, Z, f"ladder[facing={face}]")
            b.place_block(X, deck + 1, Z, "air")
            b.place_block(X, deck + 2, Z, "air")
            blocked.add((dd, tt))
            way_up = {"kind": "ladder", "rect": [X, Z, X, Z]}
            break
    if way_up is None:
        omitted.append("roof_terrace")
        fallback.append("no stair to the roof fits this lot"
                        + (f" ({why_not})" if why_not else ""))

    # --- the shade ---
    canopy = None
    if shade == "lane":
        # beams out from the front wall at the head of the ground storey, a slat on
        # every other column and a rail along their ends, over the lane to its middle
        n = max(1, min(4, int(reach) if isinstance(reach, int) else 2))
        y = fy + 4
        laid = 0
        for tt in range(a, lat_hi + 1):
            for k in range(1, n + 1):
                if (tt - a) % 2 == 0 or k == n:
                    X, Z = cell(-k, tt)
                    if b.get_block(X, y, Z) in ("air", "cave_air"):
                        b.place_block(X, y, Z, beam_top)
                        laid += 1
        if laid:
            # the rectangle the check reads is the first two columns out, which are
            # within the part's own window; the rest is the same beams carried on
            canopy = {"rect": list(rect(-min(2, n), a, -1, lat_hi)), "level": fy + 1,
                      "kind": "lane", "reach": n}
        else:
            fallback.append("the lane in front is already roofed; no beams laid")
    inner_d = list(range(1, house_d - 1))
    inner_t = list(range(a + 1, lat_hi))
    if shade == "pergola" and way_up is not None and len(inner_d) >= 3 and len(inner_t) >= 3:
        # a pergola over the part of the deck farthest from where the stair lands
        cand = []
        for d0 in range(1, house_d - 3):
            for t0 in range(a + 1, lat_hi - 2):
                d1 = min(house_d - 2, d0 + rng.choice([2, 3]))
                t1 = min(lat_hi - 1, t0 + rng.choice([2, 3]))
                cells = {(dd, tt) for dd in range(d0, d1 + 1) for tt in range(t0, t1 + 1)}
                if cells & blocked:
                    continue
                cand.append((len(cells), d0, t0, d1, t1))
        if cand:
            cand.sort(reverse=True)
            _, d0, t0, d1, t1 = cand[0]
            post = fence
            for (dd, tt) in ((d0, t0), (d0, t1), (d1, t0), (d1, t1)):
                X, Z = cell(dd, tt)
                b.place_block(X, deck + 1, Z, post)
                b.place_block(X, deck + 2, Z, post)
            for dd in range(d0, d1 + 1):
                for tt in range(t0, t1 + 1):
                    # slats on every other row, carried on beams along both sides that
                    # tie them to the posts
                    if (dd - d0) % 2 == 0 or dd == d1 or tt in (t0, t1):
                        X, Z = cell(dd, tt)
                        b.place_block(X, deck + 3, Z, beam_top)
            canopy = {"rect": list(rect(d0, t0, d1, t1)), "level": deck, "kind": "pergola",
                      "d": (d0, d1)}
    if canopy is None and shade in ("awning", "pergola", "lane"):
        # an awning of slats over the front door, one course over its head, on the wall
        span = [t for t in range(dlat - 1, dlat + 2) if a <= t <= lat_hi]
        made = 0
        for tt in span:
            X, Z = cell(-1, tt)
            if (X, Z) in _outside(part):
                continue
            b.place_block(X, fy + 4, Z, beam_top)
            made += 1
        if made:
            canopy = {"rect": list(rect(-1, span[0], -1, span[-1])), "level": fy + 1,
                      "kind": "awning"}
            if shade == "pergola":
                fallback.append("no room on the deck for a pergola; an awning at the door")
    if canopy is None:
        omitted.append("canopy")

    # --- the rear court: walled to a person's height, planted or paved ---
    court_rect = None
    if court_d:
        ground = b.block(voice.get("ground") or voice["floor"])
        for dd in range(house_d, pad_d):
            for tt in range(a, lat_hi + 1):
                X, Z = cell(dd, tt)
                edge = dd == pad_d - 1 or tt in (a, lat_hi)
                if edge:
                    if b.get_block(X, fy + 1, Z) in ("air", "cave_air"):
                        b.place_block(X, fy + 1, Z, wall_blk)
                        b.place_block(X, fy + 2, Z, trim_slab)
        c0 = house_d + (1 if way_up and way_up["kind"] == "outside" else 0)
        court_rect = list(rect(c0, a + 1, pad_d - 2, lat_hi - 1)) \
            if pad_d - 2 >= c0 and lat_hi - 1 >= a + 1 else None
        if court_rect:
            # a tree or a bed of green in the middle of the court, and the floor swept
            cx = (court_rect[0] + court_rect[2]) // 2
            cz = (court_rect[1] + court_rect[3]) // 2
            if b.get_block(cx, fy + 1, cz) in ("air", "cave_air"):
                b.place_block(cx, fy, cz, "grass_block")
                b.place_block(cx, fy + 1, cz, rng.choice(["azalea", "flowering_azalea",
                                                           "fern", "sweet_berry_bush"]))
    elif court == "rear":
        omitted.append("courtyard")
        fallback.append("the lot is too shallow for a court behind the house")

    # --- the rooms, furnished ---
    rooms = [r for r in (res.get("rooms") or []) if isinstance(r, (list, tuple))
             and len(r) == 5]
    kinds = ["bed", "table", "store", "light", "bench"]
    door_near = {(dx + i, dz + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
    for r in rooms:
        rx0, ry, rz0, rx1, rz1 = r
        rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
        rz0, rz1 = min(rz0, rz1), max(rz0, rz1)
        y = ry if b.get_block(rx0, ry, rz0) in ("air", "cave_air") else ry + 1
        edge = [(x, z) for x in range(rx0, rx1 + 1) for z in range(rz0, rz1 + 1)
                if (x in (rx0, rx1) or z in (rz0, rz1)) and (x, z) not in door_near]
        rng.shuffle(edge)
        for kind in kinds[:max(2, min(len(kinds), len(edge) // 4))]:
            for (x, z) in edge:
                face = ("east" if x == rx0 else "west" if x == rx1
                        else "south" if z == rz0 else "north")
                got = b.fitting(kind, x, y, z, face, mat=voice["floor"])
                if isinstance(got, dict) and got.get("ok"):
                    edge.remove((x, z))
                    break

    dr = res.get("door")
    if isinstance(dr, (list, tuple)) and len(dr) == 3:
        b.check_door(dr[0], dr[1], dr[2])
    b.check_walkable(label)
    b.check_attached()

    features = {"flat_roof": True, "roof_terrace": way_up is not None,
                "canopy": canopy is not None, "courtyard": court_rect is not None}
    rects = {"main": list(main), "flat_roof": list(main)}
    levels = {"flat_roof": deck}
    if way_up is not None and inner_d and inner_t:
        rows = inner_d
        if canopy and canopy["kind"] == "pergola":
            before = [dd for dd in inner_d if dd < canopy["d"][0]]
            after = [dd for dd in inner_d if dd > canopy["d"][1]]
            rows = before if len(before) >= len(after) else after
        if rows:
            p, q = cell(rows[0], inner_t[0]), cell(rows[-1], inner_t[-1])
            rects["roof_terrace"] = [min(p[0], q[0]), min(p[1], q[1]),
                                     max(p[0], q[0]), max(p[1], q[1])]
            levels["roof_terrace"] = deck
    if canopy is not None:
        rects["canopy"] = canopy["rect"]
        levels["canopy"] = canopy["level"]
    if court_rect:
        rects["courtyard"] = court_rect
    emitted = {
        "requested": {"storeys": asked, "court": court, "shade": shade},
        "storeys": int(st), "attempt": 0,
        "fallback": "; ".join(fallback) or None,
        "omitted": ([] if st >= asked else ["storeys"]) + omitted,
        "features": features, "rects": rects, "levels": levels,
        "floors": [fy + 4 * i for i in range(st)],
        "roof": {"style": "flat", "deck": deck, "parapet": deck + 1,
                 "way_up": way_up["kind"] if way_up else None},
        "shade": canopy["kind"] if canopy else None,
    }
    return {"ok": True, "label": label, "ridge_y": deck + 1, "emitted": emitted}


def _outside(part):
    """Cells a front awning may not stand over: none are known to the type itself; the
    builder refuses a write outside the lot and the awning is then simply shorter."""
    return set()
