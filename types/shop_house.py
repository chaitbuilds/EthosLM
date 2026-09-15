"""A shop-house on a city street.

The ground floor is the shop, open to the lane under a counter sill that runs the
width of the front; the way in is beside it. Rooms above, reached by a straight
flight that runs across the plan at the back of the shop. Where the plot is too
narrow to carry a flight, the shop takes the whole of it under the full height of
the roof. The eave runs out over the engawa -- along the flank where there is room
for it, along the back where the frontage is needed for the stair -- which is
where this settlement's signature move lands on a street plot.

What changes with the seed: the depth taken off a deep plot, which flank the
veranda goes on, where the flight runs and which way it climbs, the silhouette of
the roof, the bay rhythm of the frame, where the upper floor is divided, and what
the trade puts in the shop.
"""

import random

KIND = "plot"
FORM = "east_asian"
#: What this building is **for**, and so which part of a settlement it belongs in.
#: `FORM` is the tradition it is built in and this is a different question: a farmhouse
#: and a shop-house are both east Asian.
ROLE = "urban"

PARAMS = {
    "storeys": ("int", 2, 3),
    "trade": ("choice", ["grain", "cloth", "tea", "smith"]),
}

NEEDS = {
    # Cut to the band `scripts/type_needs.py` measured -- measured 10x10 to 14x14 by the
    # sweep; its author declared 6x8 because a 6x6 pad takes the shop with no floor over
    # it, and the sweep says that instance is not clean.
    "footprint": (3, 3, 17, 17),
    "except": (5, 6, 8, 9),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _key(d, k, dflt=None):
    """A value out of a dict-ish thing, with a default for missing or None."""
    if d is None:
        return dflt
    if k in d and d[k] is not None:
        return d[k]
    return dflt


def _frame(part):
    """A local frame: u runs along the street front, v runs into the plot.

        (u=0, v=0) is a front corner of the pad and v grows the way a person walks
        coming in off the lane, so v=0 is always the street face whichever way the
        pad is turned.
        
    """
    f = _key(part, "facing", "north")
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    if f == "east":
        o, du, dv = (x0, z0), (0, 1), (1, 0)
        w, d = z1 - z0 + 1, x1 - x0 + 1
        lo, hi = "north", "south"
    elif f == "west":
        o, du, dv = (x1, z0), (0, 1), (-1, 0)
        w, d = z1 - z0 + 1, x1 - x0 + 1
        lo, hi = "north", "south"
    elif f == "south":
        o, du, dv = (x0, z0), (1, 0), (0, 1)
        w, d = x1 - x0 + 1, z1 - z0 + 1
        lo, hi = "west", "east"
    else:
        o, du, dv = (x0, z1), (1, 0), (0, -1)
        w, d = x1 - x0 + 1, z1 - z0 + 1
        lo, hi = "west", "east"
    return {"o": o, "du": du, "dv": dv, "W": w, "D": d,
            "facing": f, "front": _OPP[f], "back": f, "lo": lo, "hi": hi}


def _dirname(fr, cu, cv):
    """The compass name of a local step direction."""
    if cv > 0:
        return fr["facing"]
    if cv < 0:
        return fr["front"]
    if cu > 0:
        return fr["hi"]
    return fr["lo"]


def _runs(lo, hi, keep_out):
    """The contiguous stretches of [lo, hi] left once keep_out is taken away."""
    out = []
    cur = []
    for u in range(lo, hi + 1):
        if u in keep_out:
            if cur:
                out.append((cur[0], cur[-1]))
            cur = []
        else:
            cur.append(u)
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def build(b, part, seed, **params):
    rnd = random.Random(seed)
    voice = part["voice"]
    fy = part["floor_y"]
    label = part["label"]

    storeys = 2
    if "storeys" in params and params["storeys"]:
        storeys = int(params["storeys"])
    storeys = max(2, min(3, storeys))
    trade = "grain"
    if "trade" in params and params["trade"]:
        trade = str(params["trade"])

    wall_full = b.block(voice["wall"], "full")
    frame_full = b.block(voice["frame"], "full")
    foot_full = b.block(voice["footing"], "full")
    trim_full = b.block(voice["trim"], "full")
    floor_full = b.block(voice["floor"], "full")
    floor_slab = b.block(voice["floor"], "slab")
    fence = b.joinery(voice, "fence")
    leaf = b.joinery(voice, "door")
    screen = b.joinery(voice, "trapdoor")

    fr = _frame(part)
    W, D = fr["W"], fr["D"]
    ox, oz = fr["o"]
    dux, duz = fr["du"]
    dvx, dvz = fr["dv"]

    def P(u, v):
        return (ox + dux * u + dvx * v, oz + duz * u + dvz * v)

    def box(u0, v0, u1, v1, y0, y1, blk):
        ax, az = P(u0, v0)
        cx, cz = P(u1, v1)
        b.place_cuboid(min(ax, cx), min(y0, y1), min(az, cz),
                       max(ax, cx), max(y0, y1), max(az, cz), blk)

    def put(u, v, y, blk):
        px, pz = P(u, v)
        b.place_block(px, y, pz, blk)

    def screen_of(side):
        return screen + "[open=true,facing=" + side + ",half=bottom]"

    # ---- where the way in lands, in local coordinates --------------------
    dxw, dzw = part["door"]
    ud = None
    for v in (0, 1):
        for u in range(W):
            px, pz = P(u, v)
            if px == dxw and pz == dzw:
                ud = u
                break
        if ud is not None:
            break
    if ud is None:
        ud = W // 2
    ud = max(1, min(W - 2, ud))

    # ---- massing --------------------------------------------------------- The veranda
    # takes a strip off the flank furthest from the door, so the doorway keeps a jamb on
    # both sides and the eave has something to cover.
    inset = 2 if W >= 12 else (1 if W >= 6 else 0)
    # The flight runs across the plan, from one flank wall to the other, and that wants
    # six clear cells between the walls. Where the veranda strip would take the sixth,
    # the veranda goes to the back of the plot instead and the building keeps the full
    # width of the frontage.
    eng_axis = "flank"
    if W - inset - 2 < 6 and W - 2 >= 6:
        inset = 0
        eng_axis = "rear"
    if ud <= (W - 1) / 2.0:
        eng_side = 1
        bu0, bu1 = 0, W - 1 - inset
    else:
        eng_side = -1
        bu0, bu1 = inset, W - 1
    if ud <= bu0:
        bu0 = max(0, ud - 1)
    if ud >= bu1:
        bu1 = min(W - 1, ud + 1)
    if eng_side > 0:
        eng_u0, eng_u1 = bu1 + 1, W - 1
    else:
        eng_u0, eng_u1 = 0, bu0 - 1
    if inset == 0:
        eng_u0, eng_u1 = 1, 0

    depth = D
    if D > 17:
        depth = 13 + (seed % 5)
    if eng_axis == "rear" and depth >= 8:
        depth -= 1
    elif eng_axis == "rear":
        eng_axis = None
    bv0, bv1 = 0, depth - 1
    eng_rear = bv1 + 1 if eng_axis == "rear" else None
    yard = (D - depth) >= 4 and eng_axis != "rear"

    iu0, iu1 = bu0 + 1, bu1 - 1
    iv0, iv1 = bv0 + 1, bv1 - 1
    iw = iu1 - iu0 + 1
    idp = iv1 - iv0 + 1

    # A storey is four high. Three leaves two blocks of headroom on the shop floor, and
    # a flight climbing out of a room that low is one the library will not certify as
    # walkable -- every one of them was refused.
    sh = 4

    part_v = None

    def inside(u, v):
        return iu0 <= u <= iu1 and iv0 <= v <= iv1

    def free(cells):
        """Cells a flight may have: inside, off the doorstep, and clear of the
        line the upper floor is divided on."""
        for (u, v) in cells:
            if not inside(u, v):
                return False
            if (u, v) == (ud, iv0) or (u, v) == (ud, iv0 + 1):
                return False
            if part_v is not None and v == part_v:
                return False
        return True

    def straight_options(n):
        """A straight flight wants n+2 cells in a line: a cell to stand in at
        the foot, n treads, and the landing on the floor above. Anything less
        and the library refuses it, and rightly: nobody could start up it."""
        out = []
        if iw >= n + 2:
            out.append((iu1 - n, iv1, 1, 0))
            out.append((iu0 + n, iv1, -1, 0))
            out.append((iu1 - n, iv0, 1, 0))
            out.append((iu0 + n, iv0, -1, 0))
        keep = []
        for o in out:
            su, sv, cu, cv = o
            if free([(su + cu * i, sv + cv * i) for i in range(-1, n + 1)]):
                keep.append(o)
        return keep

    # Where the upper floor is divided, on a line that still leaves the plan a place to
    # put the flight: a partition that takes the last of them away is a partition that
    # costs the building its stair.
    if idp >= 7 and iw >= 3:
        for cand_v in (iv0 + max(2, idp // 3), iv1 - 2, iv0 + 2, iv1 - 3):
            if iv0 < cand_v < iv1 - 1:
                part_v = cand_v
                if straight_options(sh):
                    break
                part_v = None

    # How many flights the plan will actually carry, decided before anything is built,
    # because it is what says how many storeys this plot can have.
    opts = straight_options(sh)
    rnd.shuffle(opts)
    picks = []
    taken = set()
    for o in opts:
        cells = set((o[0] + o[2] * i, o[1] + o[3] * i)
                    for i in range(-1, sh + 2))
        if cells & taken:
            continue
        picks.append(o)
        taken |= cells
        if len(picks) >= storeys - 1:
            break
    # Where no line inside is long enough for a flight, the stair starts out on the
    # veranda instead and climbs in through a tall opening in the flank -- which is what
    # a shop-house does when the shop takes the whole ground floor and there is nowhere
    # inside to begin a flight. A plot that cannot carry a flight cannot carry a floor
    # over the shop: it gets the shop alone, under the full height of the roof.
    if len(picks) < storeys - 1:
        storeys = len(picks) + 1

    eave_y = fy + storeys * sh
    top_y = min(fy + 56, eave_y + 12)

    # ---- clear the volume, lay the deck ----------------------------------
    box(bu0, bv0, bu1, bv1, fy + 1, top_y, "air")
    box(bu0, bv0, bu1, bv1, fy, fy, floor_full)
    for u in range(bu0, bu1 + 1):
        put(u, bv0, fy, foot_full)
        put(u, bv1, fy, foot_full)
    for v in range(bv0, bv1 + 1):
        put(bu0, v, fy, foot_full)
        put(bu1, v, fy, foot_full)

    # ---- the engawa: boards under the overhanging eave --------------------
    eng_walk = None
    if eng_rear is not None:
        for u in range(bu0, bu1 + 1):
            for v in range(eng_rear, D):
                px, pz = P(u, v)
                if b.get_height(px, pz) >= fy - 1:
                    b.place_block(px, fy, pz, trim_full)
        for u in (bu0, bu1):
            for v in range(eng_rear + 1, D):
                px, pz = P(u, v)
                if b.get_height(px, pz) >= fy:
                    b.place_block(px, fy + 1, pz, fence)
    if eng_u1 >= eng_u0:
        eng_walk = eng_u0 if eng_side > 0 else eng_u1
        rail_u = eng_u1 if eng_side > 0 else eng_u0
        for u in range(eng_u0, eng_u1 + 1):
            for v in range(bv0, bv1 + 1):
                px, pz = P(u, v)
                if b.get_height(px, pz) >= fy - 1:
                    b.place_block(px, fy, pz, trim_full)
        if rail_u != eng_walk:
            for v in range(bv0 + 3, bv1 - 1):
                px, pz = P(rail_u, v)
                if b.get_height(px, pz) >= fy:
                    b.place_block(px, fy + 1, pz, fence)

    # ---- shell: boarded walls between an exposed frame --------------------
    bay = 3 if (seed % 2 == 0) else 4

    def wall_run(u0, v0, u1, v1, y0, y1, band=None, base=None):
        ax, az = P(u0, v0)
        cx, cz = P(u1, v1)
        b.wall(min(ax, cx), y0, min(az, cz), max(ax, cx), y1, max(az, cz),
               wall_full, post=frame_full, spacing=bay, base=base,
               base_height=1, band=band)

    for s in range(storeys):
        y0 = fy + s * sh + 1
        y1 = fy + (s + 1) * sh - 1
        band = trim_full if (y1 - y0) >= 2 and s + 1 == storeys else None
        base = foot_full if s == 0 else None
        wall_run(bu0, bv0, bu1, bv0, y0, y1, band=band, base=base)
        wall_run(bu0, bv1, bu1, bv1, y0, y1, band=band, base=base)
        wall_run(bu0, bv0, bu0, bv1, y0, y1, band=band, base=base)
        wall_run(bu1, bv0, bu1, bv1, y0, y1, band=band, base=base)
        if s + 1 < storeys:
            yf = fy + (s + 1) * sh
            box(bu0, bv0, bu1, bv1, yf, yf, floor_full)
            for u in range(bu0, bu1 + 1):
                put(u, bv0, yf, trim_full)
                put(u, bv1, yf, trim_full)
            for v in range(bv0, bv1 + 1):
                put(bu0, v, yf, trim_full)
                put(bu1, v, yf, trim_full)

    for cu, cv in ((bu0, bv0), (bu1, bv0), (bu0, bv1), (bu1, bv1)):
        for y in range(fy + 1, eave_y):
            put(cu, cv, y, frame_full)

    # ---- the shop front: a wide opening onto the lane ---------------------
    keep = set([bu0, bu1, ud - 1, ud, ud + 1])
    shop_open = []
    for (a, c) in _runs(bu0, bu1, keep):
        posts = set()
        if c - a + 1 >= bay + 2:
            posts.add((a + c) // 2)
        for u in range(a, c + 1):
            if u in posts:
                for y in range(fy + 1, fy + sh):
                    put(u, bv0, y, frame_full)
            else:
                shop_open.append(u)
    for u in shop_open:
        for y in range(fy + 1, fy + sh):
            put(u, bv0, y, "air")
        put(u, bv0, fy + 1, trim_full)          # the counter, on the street

    # ---- the shop front is a shopfront and never a void ------------------- The ground
    # look's sixth finding: a two-storey tea house opens its ground floor to the lane as
    # a dark hole under the jetty, with nothing lit or built behind it. An opening is a
    # hole until something is *in* it, so the head of every run carries a shutter hung
    # out over the counter and a lamp hangs in the bay behind it, which is what a person
    # sees from the other side of the street.
    shutter = b.joinery(voice, "trapdoor")
    runs = []
    for u in sorted(shop_open):
        if runs and u == runs[-1][-1] + 1:
            runs[-1].append(u)
        else:
            runs.append([u])
    for run in runs:
        for u in run:
            put(u, bv0, fy + sh - 1,
                shutter + "[open=true,facing=" + fr["front"] + ",half=top]")
        # a lamp standing on the counter of each bay, which is the one thing a person on
        # the other side of the street can see inside a shop
        mid = run[len(run) // 2]
        mx, mz = P(mid, bv0)
        b.fitting("light", mx, fy + 2, mz, fr["back"],
                  mat=voice["trim"], room="market")

    # ---- the way in -------------------------------------------------------
    dpx, dpz = P(ud, bv0)
    b.doorway(dpx, fy + 1, dpz, fr["facing"], voice["wall"],
              leaf=leaf, jamb="build", lintel=True)


    # ---- upper openings: wide, low, screened ------------------------------
    def band_openings(u0, v0, u1, v1, y, side, width, spacing):
        ax, az = P(u0, v0)
        cx, cz = P(u1, v1)
        b.openings(min(ax, cx), y, min(az, cz), max(ax, cx), max(az, cz),
                   spacing=spacing, width=width, sill=1, head=2,
                   block=screen_of(side))

    for s in range(1, storeys):
        yy = fy + s * sh
        wide = 3 if (bu1 - bu0) >= 8 else 2
        if bu1 - bu0 >= 3:
            band_openings(bu0 + 1, bv0, bu1 - 1, bv0, yy, fr["front"], wide, 4)
            band_openings(bu0 + 1, bv1, bu1 - 1, bv1, yy, fr["back"], 2, 5)
        if bv1 - bv0 >= 3:
            band_openings(bu0, bv0 + 1, bu0, bv1 - 1, yy, fr["lo"], 2, 5)
            band_openings(bu1, bv0 + 1, bu1, bv1 - 1, yy, fr["hi"], 2, 5)

    # ---- the roof, by far the biggest thing about it ----------------------
    ax, az = P(bu0, bv0)
    cx, cz = P(bu1, bv1)
    rx0, rx1 = min(ax, cx), max(ax, cx)
    rz0, rz1 = min(az, cz), max(az, cz)
    axis = "z" if (rx1 - rx0) >= (rz1 - rz0) else "x"
    rk = _key(part, "roof")
    kw = {"overhang": 1}
    if isinstance(rk, dict):
        prof = rk.get("profile")
        if isinstance(prof, (list, tuple)) and len(prof) > 0:
            pp = [tuple(p) for p in prof
                  if isinstance(p, (list, tuple)) and len(p) == 2]
            if pp:
                kw["profile"] = pp
        ends = rk.get("ends")
        if isinstance(ends, str):
            kw["ends"] = (ends, ends)
        elif isinstance(ends, (list, tuple)) and len(ends) == 2:
            kw["ends"] = (ends[0], ends[1])
        if isinstance(rk.get("eave"), str):
            kw["eave"] = rk["eave"]
        if isinstance(rk.get("tiers"), int) and rk["tiers"] >= 1:
            kw["tiers"] = int(rk["tiers"])
    if "profile" not in kw:
        kw["profile"] = [(1, 2), (2, 1)]
    if "eave" not in kw:
        kw["eave"] = "upturned"
    if "ends" not in kw:
        kw["ends"] = rnd.choice([("irimoya", "irimoya"), ("hip", "hip"),
                                 ("irimoya", "hip"), ("half-hip", "irimoya")])
    if "tiers" not in kw:
        big = min(rx1 - rx0, rz1 - rz0) + 1 >= 9
        kw["tiers"] = 2 if (big and rnd.random() < 0.55) else 1
    ridge_y = b.roof(rx0, rz0, rx1, rz1, eave_y, voice["roof"],
                     style="hip", axis=axis, **kw)
    if not isinstance(ridge_y, int):
        ridge_y = eave_y + 4
    # The rooms are the rooms: whatever the roof brought down inside the top storey
    # comes back out, or the flight lands in a room half full of thatch.
    for s in range(storeys):
        box(iu0, iv0, iu1, iv1, fy + s * sh + 1, fy + (s + 1) * sh - 1, "air")

    # ---- getting upstairs -------------------------------------------------
    blocked = set()
    reserved = set()

    def note(res):
        if not isinstance(res, dict):
            return res
        cells = res.get("cells")
        if isinstance(cells, (list, tuple, set)):
            for c in cells:
                if isinstance(c, (list, tuple)) and len(c) >= 3:
                    blocked.add((c[0], c[2]))
        return res

    def clear_over(u, v, y):
        put(u, v, y + 1, "air")
        put(u, v, y + 2, "air")

    def lay_treads(cells):
        """Queue the treads, then read the flight back: one the queue would not
        justify leaves an empty cell, and an empty cell in a flight is a block
        to jump up. A slab on the packed wedge is the same half step."""
        wc = [(P(u, v)[0], y, P(u, v)[1]) for (u, v, y) in cells]
        b.steps(wc, voice["floor"])
        b.resolve_steps()
        for (px, py, pz) in wc:
            if b.get_block(px, py, pz) == "air":
                b.place_block(px, py, pz, floor_slab)

    stair_u = iu1
    spent = set()
    laid = 0
    for s in range(storeys - 1):
        y0 = fy + s * sh
        chosen = None
        for o in opts:
            line = set((o[0] + o[2] * i, o[1] + o[3] * i)
                       for i in range(-1, sh + 2))
            if line & spent:
                continue
            sx, sz = P(o[0], o[1])
            res = b.flight(label, sx, sz, y0, y0 + sh,
                           _dirname(fr, o[2], o[3]), mat=voice["floor"])
            if isinstance(res, dict) and res.get("ok"):
                note(res)
                chosen = o
                spent |= line
                laid += 1
                break
        if chosen is None:
            continue
        su, sv, cu, cv = chosen
        foot = (su - cu, sv - cv)
        land = (su + cu * sh, sv + cv * sh)
        for i in range(sh + 1):
            blocked.add(P(su + cu * i, sv + cv * i))
        reserved.add(land)
        reserved.add(foot)
        reserved.add((land[0] + cu, land[1] + cv))
        if s == 0:
            stair_u = land[0]
            for v in range(min(iv0, foot[1]), max(iv0, foot[1]) + 1):
                reserved.add((ud, v))
            for u in range(min(ud, foot[0]), max(ud, foot[0]) + 1):
                reserved.add((u, foot[1]))
        if part_v is not None:
            lo_v = min(land[1], part_v) - 1
            hi_v = max(land[1], part_v) + 1
            for v in range(lo_v, hi_v + 1):
                reserved.add((land[0], v))

    # A floor is only a floor if a flight reached it. Any storey above the last one a
    # flight climbed to has its floor taken out again, and the room under it runs up to
    # the roof instead: better an open shop two storeys high than a room over it that
    # nobody can walk into.
    reach = laid + 1
    if reach < storeys:
        for s in range(reach, storeys):
            box(iu0, iv0, iu1, iv1, fy + s * sh, fy + s * sh, "air")
    if reach < 2:
        part_v = None
    upper_ok = reach >= 2

    # ---- a way out onto the veranda, cut once the flight is in ------------
    if eng_walk is not None and idp >= 3:
        gu = bu1 if eng_side > 0 else bu0
        gv = max(iv0, min(iv1 - 1, iv0 + (idp // 2) - 1))
        for v in (gv, gv + 1):
            if bv0 < v < bv1 and P(gu, v) not in blocked:
                for y in range(fy + 1, fy + min(sh, 3)):
                    put(gu, v, y, "air")


    # A spine down the plan and a way across each part of the upper floor. Furniture
    # that fills the one cell joining two halves of a room walls the far half off, and a
    # room walled off by its own furniture is a room the walk cannot get into.
    for v in range(iv0, iv1 + 1):
        reserved.add((stair_u, v))
    if part_v is not None:
        for u in range(iu0, iu1 + 1):
            if part_v - 1 >= iv0:
                reserved.add((u, part_v - 1))
            if part_v + 1 <= iv1:
                reserved.add((u, part_v + 1))

    # keep the way in off the lane clear of furniture
    reserved.add((ud, iv0))
    if idp >= 3:
        reserved.add((ud, iv0 + 1))
    for (u, v) in list(reserved):
        if inside(u, v):
            blocked.add(P(u, v))

    # ---- the upper floor divided, where it is long enough -----------------
    if part_v is not None and upper_ok:
        for s in range(1, reach):
            yy = fy + s * sh
            for u in range(iu0, iu1 + 1):
                if abs(u - stair_u) <= 1 or (u, part_v) in reserved:
                    continue
                px, pz = P(u, part_v)
                if (px, pz) in blocked:
                    continue
                for y in range(yy + 1, yy + sh - 1):
                    put(u, part_v, y, wall_full)
                put(u, part_v, yy + sh - 1, trim_full)
                if (u - iu0) % 3 == 0:
                    for y in range(yy + 1, yy + sh):
                        put(u, part_v, y, frame_full)

    # ---- furniture -------------------------------------------------------- A block in
    # every cell that has to stay walkable, so a fitting wider than the cell it was
    # offered cannot spill across the way through. They come out again once the
    # furniture is in.
    marks = []
    for (u, v) in sorted(reserved):
        if not inside(u, v):
            continue
        px, pz = P(u, v)
        for s in range(storeys):
            yy = fy + s * sh + 1
            if b.get_block(px, yy, pz) == "air":
                b.place_block(px, yy, pz, wall_full)
                marks.append((px, yy, pz))

    def open_side(u, v):
        if u == iu1:
            return fr["lo"]
        if u == iu0:
            return fr["hi"]
        if v == iv1:
            return fr["front"]
        return fr["back"]

    def spots(s, wallside=True):
        y = fy + s * sh + 1
        out = []
        for v in range(iv0, iv1 + 1):
            for u in range(iu0, iu1 + 1):
                px, pz = P(u, v)
                if (px, pz) in blocked:
                    continue
                edge = (u == iu0 or u == iu1 or v == iv0 or v == iv1)
                if wallside != edge:
                    continue
                out.append((u, v, px, y, pz, open_side(u, v)))
        rnd.shuffle(out)
        return out

    def place_fit(kind, cand, **kw):
        for (u, v, px, py, pz, fc) in cand:
            r = b.fitting(kind, px, py, pz, fc, **kw)
            if isinstance(r, dict) and r.get("ok"):
                blocked.add((px, pz))
                note(r)
                return r
        return None

    front_cand = []
    for u in shop_open:
        px, pz = P(u, iv0)
        if (px, pz) not in blocked:
            front_cand.append((u, iv0, px, fy + 1, pz, fr["front"]))
    rnd.shuffle(front_cand)
    place_fit("table", front_cand, mat=voice["trim"], room="store")

    ws = spots(0, True)
    ms = spots(0, False)
    goods = max(2, min(4, iw - 1))
    if trade == "smith":
        place_fit("forge", ws, mat=voice["footing"], room="store")
        place_fit("anvil", ws + ms, mat=voice["footing"], room="store")
        place_fit("store", ws, mat=voice["trim"], extent=2, room="store")
    elif trade == "tea":
        place_fit("hearth", ws, mat=voice["footing"], room="store",
                  flue_to=ridge_y)
        place_fit("bench", ws, mat=voice["trim"], room="store")
        place_fit("store", ws, mat=voice["trim"], extent=2, room="store")
    elif trade == "cloth":
        place_fit("store", ws, mat=voice["trim"], extent=goods, room="store")
        place_fit("workbench", ws + ms, mat=voice["trim"], room="store")
        place_fit("bench", ws, mat=voice["trim"], room="store")
    else:
        place_fit("store", ws, mat=voice["trim"], extent=goods, room="store")
        place_fit("trough", ws, mat=voice["footing"], room="store")
        place_fit("workbench", ws + ms, mat=voice["trim"], room="store")
    place_fit("light", ws, mat=voice["trim"], room="store")

    for s in range(1, reach):
        uw = spots(s, True)
        um = spots(s, False)
        place_fit("bed", uw, mat=voice["floor"], room="house")
        place_fit("table", um + uw, mat=voice["trim"], room="house")
        place_fit("bookshelf", uw, mat=voice["trim"], room="house")
        if trade in ("cloth", "tea"):
            place_fit("store", uw, mat=voice["trim"], extent=2, room="house")
        if idp >= 7 or iw >= 7:
            place_fit("bed", uw, mat=voice["floor"], room="house")
            place_fit("rug", um, mat=voice["trim"], room="house")
        place_fit("light", uw, mat=voice["trim"], room="house")
        if part_v is not None:
            back = [c for c in spots(s, True) if c[1] > part_v]
            place_fit("light", back, mat=voice["trim"], room="house")
            place_fit("store", back, mat=voice["trim"], extent=2, room="house")

    for (px, yy, pz) in marks:
        b.place_block(px, yy, pz, "air")

    # ---- the yard behind, on a deep plot ----------------------------------
    if yard:
        yv0, yv1 = bv1 + 1, D - 1
        for v in range(yv0 + 2, yv1 + 1):
            for u in (bu0, bu1):
                px, pz = P(u, v)
                yy = b.get_height(px, pz) + 1
                if yy >= fy:
                    b.place_block(px, yy, pz, fence)
        for u in range(bu0, bu1 + 1):
            px, pz = P(u, yv1)
            yy = b.get_height(px, pz) + 1
            if yy >= fy:
                b.place_block(px, yy, pz, fence)
        cx2, cz2 = P((bu0 + bu1) // 2, (yv0 + yv1) // 2)
        cy = b.get_height(cx2, cz2) + 1
        if cy >= fy:
            b.fitting("well", cx2, cy, cz2, fr["front"], mat=voice["footing"],
                      room="yard")

    # ---- what a person can do with it -------------------------------------
    b.check_door(dpx, fy + 1, dpz)
    b.check_walkable(label)
    b.seal_voids(part["x0"], part["z0"], part["x1"], part["z1"], floor_full)

    att = b.check_attached()
    if isinstance(att, dict):
        for piece in att.get("floating", []) or []:
            if not isinstance(piece, dict):
                continue
            bb = piece.get("bbox")
            if piece.get("natural") or not isinstance(bb, (list, tuple)):
                continue
            if piece.get("cells", 99) > 6 or len(bb) != 6:
                continue
            if bb[1] <= fy:
                continue
            b.place_cuboid(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], "air")
    return {"ridge": ridge_y, "storeys": storeys, "trade": trade}
