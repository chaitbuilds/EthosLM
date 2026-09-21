"""A market: two rows of stalls facing each other across an open aisle.

The working space of a crowded quarter, as an **area** the layout can put on a lane like
a square. The claims it rests on: `lower-ring-is-crowded-upper-is-not` (the lower ring is
busy and crowded -- a market is where the crowd is) and `three-rings-ranked-by-class`
(a ring of traders and craftsmen), both from the closure-city reading; the form -- a
covered stall row on each side of a clear aisle, open at both ends to the lane -- is a
design decision the sources do not spell out, and the docstring says so. It is the same
market in any tradition because every block is a role on `b.voice`; the difference from
the corner-booth `square` is the organisation: a street of trade rather than a paved
crossing with booths at its corners.

What it **emits** is measured, not declared: `emitted.features.stalls` counts the
booths that stood and `emitted.rects.stalls` is the rectangle they occupy, which
`construction.outcome` verifies as solid; `rects.aisle` is the clear way through, verified
as open ground. A pad too narrow for two rows builds one row and says so in `fallback`;
one too narrow for any row paves the ground and reports `stalls` as omitted -- a paved
rectangle is not a market and is not reported as one.
"""

import random

KIND = "area"
FORM = "civic"
ROLE = "civic"
FUNCTION = "market"
#: The features this type can be asked to deliver, for `envelope.table`.
FEATURES = ("stalls", "aisle")

PARAMS = {
    "goods": ("choice", ["produce", "cloth", "wares"]),
    "canopy": ("choice", ["gable", "shed"]),
}

NEEDS = {
    # measured by scripts/type_needs.py; a row of stalls, an aisle and a margin
    "footprint": (3, 3, 48, 48),
    "frontage": "lane",
    "ground": "any",
    "clearance": 1,
}

#: **The lots this type needs, measured** by `$PY scripts/type_needs.py --envelope
#: --transcribe market`: each row is the least lot on which the parameters stand with
#: the features named, at one seed (`lot_min`) and at seeds [1, 2, 3] (`lot_pref`). Read
#: by `ethoslm.envelope.lot_for` before a lot is drawn; the outcome construction measures
#: afterwards remains authoritative.
ENVELOPE = [
 {
  "params": {
   "canopy": "gable",
   "goods": "produce"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "produce"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "produce"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "produce"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "cloth"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "cloth"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "cloth"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "cloth"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "wares"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "wares"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "wares"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "gable",
   "goods": "wares"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "produce"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "produce"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "produce"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "produce"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "cloth"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "cloth"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "cloth"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "cloth"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "wares"
  },
  "features": [],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "wares"
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   5,
   5
  ],
  "lot_pref": [
   5,
   5
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "wares"
  },
  "features": [
   "storeys",
   "stalls"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "canopy": "shed",
   "goods": "wares"
  },
  "features": [
   "storeys",
   "aisle"
  ],
  "lot_min": [
   14,
   14
  ],
  "lot_pref": [
   14,
   14
  ],
  "why": "probed at seeds [1, 2, 3]"
 }
]

STALL_DEPTH = 3
STALL_LEN = 4
#: the aisle: three clear cells, and a row either side under the stalls' hoods
AISLE = 5
AISLE_CLEAR = 3


def _foot(b):
    return b.block(b.voice["footing"])


def _foot2(b):
    return b.block(b.voice["footing"], "accent")


def _panel(b):
    return b.block(b.voice["wall"], "accent")


def _post(b):
    return b.axial(b.block(b.voice["frame"], "post"), "y")


def _slab(b):
    return b.block(b.voice["floor"], "slab") + "[type=bottom]"


def _beam(b, axis):
    return b.axial(b.block(b.voice["trim"], "bare"), axis)


def _crate(b, x, z, fy, high=1):
    b.place_block(x, fy + 1, z, "barrel[facing=up]")
    if high > 1:
        b.place_block(x, fy + 2, z, "barrel[facing=up]")
    b.place_block(x, fy + high + 1, z, _slab(b))


def _stall(b, rect, fy, front, goods, canopy, rng):
    """One booth: four posts, a back panel, a counter facing the aisle, a hood and a
    roof. `front` is the side the counter is on ("north"|"south"|"east"|"west")."""
    x0, z0, x1, z1 = rect
    plate = fy + 3
    for (cx, cz) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        b.place_block(cx, fy + 1, cz, _foot(b))
        b.place_cuboid(cx, fy + 2, cz, cx, plate, cz, _post(b))
    axis_x = front in ("north", "south")
    # the beams at plate height round the top
    for x in range(x0, x1 + 1):
        b.place_block(x, plate, z0, _beam(b, "x"))
        b.place_block(x, plate, z1, _beam(b, "x"))
    for z in range(z0, z1 + 1):
        b.place_block(x0, plate, z, _beam(b, "z"))
        b.place_block(x1, plate, z, _beam(b, "z"))
    # the back: a panel at head height, open below so the booth can be walked round
    if front == "north":
        back = [(x, z1) for x in range(x0 + 1, x1)]
        counter = [(x, z0) for x in range(x0 + 1, x1)]
        hood = [(x, z0 - 1) for x in range(x0, x1 + 1)]
    elif front == "south":
        back = [(x, z0) for x in range(x0 + 1, x1)]
        counter = [(x, z1) for x in range(x0 + 1, x1)]
        hood = [(x, z1 + 1) for x in range(x0, x1 + 1)]
    elif front == "west":
        back = [(x1, z) for z in range(z0 + 1, z1)]
        counter = [(x0, z) for z in range(z0 + 1, z1)]
        hood = [(x0 - 1, z) for z in range(z0, z1 + 1)]
    else:
        back = [(x0, z) for z in range(z0 + 1, z1)]
        counter = [(x1, z) for z in range(z0 + 1, z1)]
        hood = [(x1 + 1, z) for z in range(z0, z1 + 1)]
    for (x, z) in back:
        b.place_block(x, fy + 2, z, _panel(b))
    for (x, z) in counter:
        b.place_block(x, fy + 1, z, _foot(b))
        b.place_block(x, fy + 2, z, _slab(b))
    for (x, z) in hood:
        b.place_block(x, plate, z, _beam(b, "x" if axis_x else "z"))
    mid = hood[len(hood) // 2]
    b.place_block(mid[0], plate - 1, mid[1], "lantern[hanging=true]")
    if canopy == "shed":
        b.roof(x0, z0, x1, z1, plate + 1, b.voice["roof"], style="shed",
               axis=front[0], overhang=1)
    else:
        b.roof(x0, z0, x1, z1, plate + 1, b.voice["roof"], style="gable",
               axis=("z" if axis_x else "x"), overhang=1)
    # what it sells, behind the counter
    inside = [(x, z) for x in range(x0 + 1, x1) for z in range(z0 + 1, z1)]
    rng.shuffle(inside)
    wares = {"produce": ["store", "table"], "cloth": ["table", "shelf"],
             "wares": ["workbench", "store"]}[goods]
    for c in inside[:max(1, len(inside) - 1)]:
        placed = False
        for kind in wares:
            got = b.fitting(kind, c[0], fy + 1, c[1], front, mat=b.voice["floor"],
                            extent=1, room="market")
            if isinstance(got, dict) and got.get("ok"):
                placed = True
                break
        if not placed:
            _crate(b, c[0], c[1], fy, 1)


def _lid(b, rect, fy):
    """Under a roof, a waist-high block with two clear cells over it is a perch nobody
    can reach; board it over, as the square does."""
    x0, z0, x1, z1 = rect
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            for y in (fy + 1, fy + 2):
                low = b.get_block(x, y, z)
                if low == "air" or any(s in low for s in ("slab", "stairs", "fence", "lantern", "barrel", "door")):
                    continue
                if b.get_block(x, y + 1, z) == "air" and b.get_block(x, y + 2, z) == "air":
                    b.place_block(x, y + 1, z, _slab(b))


def build(b, part, seed, **params):
    rng = random.Random(int(seed) * 104729 + 0x77)
    goods = params.get("goods", "produce")
    if goods not in ("produce", "cloth", "wares"):
        goods = "produce"
    canopy = params.get("canopy", "gable")
    if canopy not in ("gable", "shed"):
        canopy = "gable"
    fp = part.get("footprint") or [part["x0"], part["z0"], part["x1"], part["z1"]]
    x0, z0, x1, z1 = [int(v) for v in fp]
    fy = int(part["floor_y"])
    w, d = x1 - x0 + 1, z1 - z0 + 1
    b.fill_region(x0, fy + 1, z0, x1, fy + 10, z1, "air")
    # **The market floor is a figure**, composition round: the ground course here and
    # the accent laid down the aisle below are one drawn thing -- the way through reads
    # on the ground only because the two contrast -- and the design round's material
    # pass replaced the pair with camouflage (`out/des-material/comparison.json`,
    # criterion 5: "the market floor's chequer of quartz and diorite is replaced by
    # camouflage"). Declared where it is laid, because nothing else knows.
    with b.figure("market_floor"):
        b.fill_region(x0, fy, z0, x1, fy, z1, _foot(b))
    # the aisle runs along the long axis
    along_x = w >= d
    across = d if along_x else w
    length = w if along_x else d
    rows = 2 if across >= 2 * STALL_DEPTH + AISLE + 2 else (
        1 if across >= STALL_DEPTH + AISLE + 2 else 0)
    stalls = []
    aisle = None
    if rows and length >= STALL_LEN + 2:
        # the aisle, centred across; stall rows either side of it
        if along_x:
            az0 = z0 + (d - AISLE) // 2
            aisle = (x0, az0, x1, az0 + AISLE - 1)
            fronts = [("north", az0 + AISLE, az0 + AISLE + STALL_DEPTH - 1),
                      ("south", az0 - STALL_DEPTH, az0 - 1)]
            if rows == 1:
                # one row: on the side with room, the aisle against the other edge
                az0 = z0 + 1
                aisle = (x0, az0, x1, az0 + AISLE - 1)
                fronts = [("north", az0 + AISLE, az0 + AISLE + STALL_DEPTH - 1)]
            n = (length - 2 + 1) // (STALL_LEN + 1)
            n = max(1, n)
            span = n * STALL_LEN + (n - 1)
            sx = x0 + 1 + (length - 2 - span) // 2
            for front, sz0, sz1 in fronts:
                if sz0 < z0 + 1 or sz1 > z1 - 1:
                    continue
                for i in range(n):
                    rx0 = sx + i * (STALL_LEN + 1)
                    stalls.append(((rx0, sz0, rx0 + STALL_LEN - 1, sz1), front))
        else:
            ax0 = x0 + (w - AISLE) // 2
            aisle = (ax0, z0, ax0 + AISLE - 1, z1)
            fronts = [("west", ax0 + AISLE, ax0 + AISLE + STALL_DEPTH - 1),
                      ("east", ax0 - STALL_DEPTH, ax0 - 1)]
            if rows == 1:
                ax0 = x0 + 1
                aisle = (ax0, z0, ax0 + AISLE - 1, z1)
                fronts = [("west", ax0 + AISLE, ax0 + AISLE + STALL_DEPTH - 1)]
            n = max(1, (length - 2 + 1) // (STALL_LEN + 1))
            span = n * STALL_LEN + (n - 1)
            sz = z0 + 1 + (length - 2 - span) // 2
            for front, sx0, sx1 in fronts:
                if sx0 < x0 + 1 or sx1 > x1 - 1:
                    continue
                for i in range(n):
                    rz0 = sz + i * (STALL_LEN + 1)
                    stalls.append(((sx0, rz0, sx1, rz0 + STALL_LEN - 1), front))
    # the clear way through is the aisle's middle; the outer rows stand under the hoods
    clear = None
    if aisle:
        inset = (AISLE - AISLE_CLEAR) // 2
        clear = ((aisle[0], aisle[1] + inset, aisle[2], aisle[3] - inset) if along_x
                 else (aisle[0] + inset, aisle[1], aisle[2] - inset, aisle[3]))
        if rows == 1:
            # one row: the hoods take one side only; the far side is clear to the edge
            clear = ((aisle[0], aisle[1], aisle[2], aisle[3] - inset) if along_x
                     else (aisle[0], aisle[1], aisle[2] - inset, aisle[3]))
    # paving: the aisle in the accent, so the way through reads on the ground -- the
    # other half of the `market_floor` figure declared above
    if clear:
        with b.figure("market_floor"):
            b.fill_region(clear[0], fy, clear[1], clear[2], fy, clear[3], _foot2(b))
    for rect, front in stalls:
        _stall(b, rect, fy, front, goods, canopy, rng)
    for rect, _front in stalls:
        _lid(b, rect, fy)
    # a well at the middle of the aisle where it is long enough to keep the way round it
    well = None
    if clear and (clear[2] - clear[0] + 1) * (clear[3] - clear[1] + 1) >= 3 * 12 and rows == 2:
        cx, cz = (clear[0] + clear[2]) // 2, (clear[1] + clear[3]) // 2
        got = b.fitting("well", cx, fy + 1, cz, "north", mat=_foot(b), room="market")
        if isinstance(got, dict) and got.get("ok"):
            well = [cx, cz, cx, cz]
    if not stalls:
        # too small for a row: a paved ground with a table and a lamp, and said so
        cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
        b.fitting("table", cx, fy + 1, cz, "north", mat=b.voice["floor"], room="market")
    b.area_way_in(part["x0"], part["z0"], part["x1"], part["z1"], fy)
    srect = None
    if stalls:
        srect = [min(r[0] for r, _f in stalls), min(r[1] for r, _f in stalls),
                 max(r[2] for r, _f in stalls), max(r[3] for r, _f in stalls)]
    emitted = {
        "requested": {"goods": goods, "canopy": canopy, "stalls": True},
        "storeys": 0, "attempt": 0 if rows == 2 else (1 if rows == 1 else 2),
        "fallback": (None if rows == 2 and stalls else
                     "one row of stalls: the pad is too narrow for two" if stalls else
                     "no row of stalls fits: paved ground with a table"),
        "omitted": [] if stalls else ["stalls"],
        "features": {"stalls": len(stalls), "aisle": bool(aisle), "rows": rows,
                     "well": bool(well)},
        "rects": {"main": [x0, z0, x1, z1],
                  **({"stalls": srect} if srect else {}),
                  **({"aisle": list(clear)} if clear else {}),
                  **({"well": well} if well else {})},
        "floors": [fy],
    }
    return {"ok": True, "stalls": len(stalls), "emitted": emitted}
