"""A chapel hall: one aisled room under a steep gable, a raised dais with an altar at the
end away from the door, benches facing it down an open middle aisle, and a bell hung on
the wall over the door.

The expression round's first type authored through the growth path for a **function**
gap: the orchard hamlet asks for worship (`function/worship`) and no committed type of
`european_vernacular` declared it. This is a form and not a voice -- every block is a
role on `b.voice` and the roof is the voice's profile -- and what it delivers is said in
`emitted` with a rectangle per feature, so `construction.outcome` verifies the dais, the
altar, the benches and the bell on the built blocks rather than taking `FUNCTION` on
trust. The lot it needs is small on purpose: a chapel of a hamlet is a room, not a nave.
"""

import random

KIND = "plot"
FORM = "european_vernacular"
ROLE = "civic"
FUNCTION = "worship"
#: The features this type can be asked to deliver, for `envelope.table`.
FEATURES = ("altar", "dais", "benches", "bell")

PARAMS = {
    "storeys": ("int", 1, 1),
    "bell": ("choice", ["hung", "none"]),
}

NEEDS = {
    # measured by scripts/type_needs.py; the pair is the pad site() hands build()
    "footprint": (3, 3, 32, 32),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

#: **The lots this type needs, measured** by `$PY scripts/type_needs.py --envelope
#: --transcribe worship`: each row is the least lot on which the parameters stand with
#: the features named, at one seed (`lot_min`) and at seeds [1, 2, 3] (`lot_pref`). Read
#: by `ethoslm.envelope.lot_for` before a lot is drawn; the outcome construction measures
#: afterwards remains authoritative.
ENVELOPE = [
 {
  "params": {
   "bell": "hung",
   "storeys": 1
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
   "bell": "hung",
   "storeys": 1
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
   "bell": "hung",
   "storeys": 1
  },
  "features": [
   "storeys",
   "altar"
  ],
  "lot_min": [
   7,
   8
  ],
  "lot_pref": [
   7,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "hung",
   "storeys": 1
  },
  "features": [
   "storeys",
   "dais"
  ],
  "lot_min": [
   7,
   8
  ],
  "lot_pref": [
   7,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "hung",
   "storeys": 1
  },
  "features": [
   "storeys",
   "benches"
  ],
  "lot_min": [
   8,
   8
  ],
  "lot_pref": [
   8,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "hung",
   "storeys": 1
  },
  "features": [
   "storeys",
   "bell"
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
   "bell": "none",
   "storeys": 1
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
   "bell": "none",
   "storeys": 1
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
   "bell": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "altar"
  ],
  "lot_min": [
   7,
   8
  ],
  "lot_pref": [
   7,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "dais"
  ],
  "lot_min": [
   7,
   8
  ],
  "lot_pref": [
   7,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "benches"
  ],
  "lot_min": [
   8,
   8
  ],
  "lot_pref": [
   8,
   8
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "bell": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "bell"
  ],
  "lot_min": None,
  "lot_pref": None,
  "why": "worship delivers no lot up to 44x44 with {'bell': 'none', 'storeys': 1} and ['storeys', 'bell']: ['bell'] never appeared"
 }
]

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
STEP = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


def _door_side(part):
    x, z = part["door"][0], part["door"][-1]
    d = {"west": abs(x - part["x0"]), "east": abs(part["x1"] - x),
         "north": abs(z - part["z0"]), "south": abs(part["z1"] - z)}
    return min(d, key=lambda k: d[k])


def _door_cell(res, part, fy):
    dr = res.get("door")
    if isinstance(dr, (list, tuple)) and len(dr) == 3:
        return int(dr[0]), int(dr[2])
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        return int(dr[0]), int(dr[1])
    return int(part["door"][0]), int(part["door"][-1])


def build(b, part, seed, **params):
    rng = random.Random(int(seed) * 7919 + 0xC4A)
    voice = b.voice
    label = part["label"]
    fy = int(part["floor_y"])
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    bell_want = params.get("bell", "hung")
    if bell_want not in ("hung", "none"):
        bell_want = "hung"
    asked = {"storeys": 1, "bell": bell_want}

    # the hall keeps a block off every edge of a pad that can spare it, so the eaves and
    # the skirt stay off the lane
    x0, z0, x1, z1 = px0, pz0, px1, pz1
    if x1 - x0 + 1 >= 9:
        x0, x1 = x0 + 1, x1 - 1
    if z1 - z0 + 1 >= 9:
        z0, z1 = z0 + 1, z1 - 1
    W, D = x1 - x0 + 1, z1 - z0 + 1
    axis = "z" if W >= D else "x"          # the ridge runs along the long side
    ds = _door_side(part)
    attempts = [
        (dict(style="gable", axis=axis, pitch=(2, 1)), dict(openings="rhythm")),
        (dict(style="gable", axis=axis, pitch=(1, 1)), dict(openings="rhythm")),
        (dict(style="gable", axis=axis), dict(openings="none")),
    ]
    res, rung = None, None
    for i, (roof, kw) in enumerate(attempts):
        r = b.building(label, x0, z0, x1, z1, 1, roof, mat=dict(voice), stair="none", **kw)
        if isinstance(r, dict) and r.get("ok"):
            res, rung = r, i
            break
    if res is None:
        return {"ok": False, "label": label, "reason": "no hall stood",
                "emitted": {"requested": asked, "storeys": 0, "attempt": None,
                            "fallback": "no shell stood",
                            "omitted": ["storeys", "altar", "dais", "benches"],
                            "features": {}}}
    rooms = [tuple(r) for r in (res.get("rooms") or []) if len(r) == 5]
    dx, dz = _door_cell(res, part, fy)
    features = {"altar": False, "dais": False, "benches": 0, "bell": False}
    rects = {"main": [x0, z0, x1, z1]}
    stand = fy + 1
    slab_top = b.block(voice["footing"], "slab") + "[type=bottom]"
    altar_blk = b.block(voice["trim"])
    if rooms:
        rx0, ry, rz0, rx1, rz1 = max(rooms, key=lambda r: (r[3] - r[0] + 1) * (r[4] - r[2] + 1))
        stand = ry + 1
        # the end away from the door, along the long axis of the room
        long_x = (rx1 - rx0) >= (rz1 - rz0)
        if long_x:
            far_east = abs(dx - rx1) >= abs(dx - rx0)
            far_col = rx1 if far_east else rx0
            inner = rx1 - 1 if far_east else rx0 + 1
            dais_cells = [(x, z) for x in (far_col, inner) for z in range(rz0, rz1 + 1)]
            altar_cells = [(far_col, z) for z in range(rz0, rz1 + 1)]
            face = "west" if far_east else "east"          # benches look this way
            aisle = {(x, (rz0 + rz1) // 2) for x in range(rx0, rx1 + 1)}
            rows = range(rx0 + (3 if not far_east else 0), rx1 + 1)
            bench_cells = [(x, z) for x in range(rx0, rx1 + 1) for z in range(rz0, rz1 + 1)
                           if (x, z) not in aisle and x not in (far_col, inner)
                           and (x - rx0) % 2 == (1 if far_east else 0)]
        else:
            far_south = abs(dz - rz1) >= abs(dz - rz0)
            far_row = rz1 if far_south else rz0
            inner = rz1 - 1 if far_south else rz0 + 1
            dais_cells = [(x, z) for z in (far_row, inner) for x in range(rx0, rx1 + 1)]
            altar_cells = [(x, far_row) for x in range(rx0, rx1 + 1)]
            face = "north" if far_south else "south"
            aisle = {((rx0 + rx1) // 2, z) for z in range(rz0, rz1 + 1)}
            bench_cells = [(x, z) for x in range(rx0, rx1 + 1) for z in range(rz0, rz1 + 1)
                           if (x, z) not in aisle and z not in (far_row, inner)
                           and (z - rz0) % 2 == (1 if far_south else 0)]
        # keep the door's own three cells clear
        keep = set()
        ox, oz = STEP[OPP[ds]]
        for k in range(0, 4):
            keep.add((dx + ox * k, dz + oz * k))
            keep.add((dx + ox * k + oz, dz + oz * k + ox))
            keep.add((dx + ox * k - oz, dz + oz * k - ox))
        room_w = (rz1 - rz0 + 1) if long_x else (rx1 - rx0 + 1)
        room_d = (rx1 - rx0 + 1) if long_x else (rz1 - rz0 + 1)
        if room_d >= 4 and room_w >= 3:
            # the dais: a half step up across the end, the altar on its far row
            dais = [c for c in dais_cells if c not in keep]
            for (x, z) in dais:
                b.place_block(x, ry + 1, z, slab_top)
            if dais:
                features["dais"] = True
                rects["dais"] = [min(c[0] for c in dais), min(c[1] for c in dais),
                                 max(c[0] for c in dais), max(c[1] for c in dais)]
            mid = altar_cells[len(altar_cells) // 2]
            altar = [mid] if room_w < 5 else [altar_cells[len(altar_cells) // 2 - 1], mid]
            altar = [c for c in altar if c not in keep]
            for (x, z) in altar:
                b.place_block(x, ry + 1, z, altar_blk)
                b.place_block(x, ry + 2, z, b.block(voice["footing"], "slab") + "[type=bottom]")
            if altar:
                features["altar"] = True
                rects["altar"] = [min(c[0] for c in altar), min(c[1] for c in altar),
                                  max(c[0] for c in altar), max(c[1] for c in altar)]
            # a light over the altar
            b.fitting("light", mid[0], ry + 1, mid[1] if long_x else mid[1], face,
                      mat=voice["trim"], room="shrine")
        # the benches, in rows facing the dais, the middle aisle open
        placed = []
        for (x, z) in bench_cells:
            if (x, z) in keep:
                continue
            got = b.fitting("bench", x, stand, z, face, mat=voice["floor"], room="hall")
            if isinstance(got, dict) and got.get("ok"):
                placed.append((x, z))
        features["benches"] = len(placed)
        if placed:
            rects["benches"] = [min(c[0] for c in placed), min(c[1] for c in placed),
                                max(c[0] for c in placed), max(c[1] for c in placed)]
    # the bell, hung on the wall over the door, on the outside
    if bell_want == "hung":
        ox, oz = STEP[ds]
        bx, bz = dx + ox, dz + oz
        by = fy + 3                         # over the head of a person on the doorstep
        if b.get_block(bx, by, bz) == "air" and b.get_block(bx, by - 1, bz) == "air" \
                and b.get_block(bx, by - 2, bz) == "air":
            b.place_block(bx, by, bz, f"bell[attachment=single_wall,facing={ds}]")
            features["bell"] = True
            rects["bell"] = [bx, bz, bx, bz]
    omitted = [f for f in ("altar", "dais", "benches") if not features[f]]
    if bell_want == "hung" and not features["bell"]:
        omitted.append("bell")
    gave = []
    if rung:
        gave.append(("ladder: a shallower roof", "ladder: blank walls under the voice's gable")[rung - 1])
    if omitted:
        gave.append("room: " + ", ".join(omitted) + " did not fit")
    emitted = {
        "requested": asked, "storeys": 1, "attempt": int(rung or 0),
        "fallback": "; ".join(gave) or None, "omitted": omitted,
        "features": features, "rects": rects,
        "floors": list(res.get("floors") or [fy]),
    }
    dr = res.get("door")
    if isinstance(dr, (list, tuple)) and len(dr) == 3:
        b.check_door(dr[0], dr[1], dr[2])
    b.seal_voids(px0, pz0, px1, pz1, b.block(voice["wall"]), max_cells=300)
    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "label": label, "ridge_y": res.get("ridge_y"), "emitted": emitted}
