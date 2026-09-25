"""**The form a lot holds, asked of the type before the lot is drawn.** The design
resolution round.

The fabric reset round answered "does this design work?" four times, in four places, with
four different estimates: the envelope certificate (does a shell stand on this rectangle),
the compiler's lot band (`court_least + 6 + 2 * FRONT_INSET`), `settle_storeys` (the
envelope again, with the flanks the row really has) and the type's own `build()` (which
thinned its ranges to a clear depth of one to reach the court it was asked for). All four
agreed that a 15-wide lot held a courtyard house with a court of seven; only the last knew
the rooms had become corridors, and it did not say so.

A type that knows its own geometry now publishes it: `form_plan(pad_w, pad_d, *, params,
front, attached)` returns the rooms, court and storeys a pad holds -- or a refusal naming
the pad it would need. This module is the one place a **lot** is turned into that pad
(the inset rule `buildlib.pad_insets` applies: `inset` on a free side, none on a party
wall) and the answer read back. The compiler admits and sizes lots with it
(`least_lot`, `lot_band`), the type's `build()` lays the same plan, and `measure` reads
the critical dimensions off the emitted blocks, so admission, emission and the built
reading can be compared and cannot silently disagree.

A type that publishes no `form_plan` is answered `None` everywhere here and the callers
keep the envelope path they had.
"""
from __future__ import annotations

import os

from .buildlib import pad_insets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: The frontage sides and, for each, the two flanks along the street (low, high).
FLANKS = {"north": ("west", "east"), "south": ("west", "east"),
          "west": ("north", "south"), "east": ("north", "south")}

_PLANS: dict = {}


def plan_fn(tname: str | None, decl: dict | None = None):
    """The type's `form_plan`, or None."""
    if decl is not None and callable(decl.get("form_plan")):
        return decl["form_plan"]
    if not tname:
        return None
    if tname in _PLANS:
        return _PLANS[tname]
    fn = None
    path = os.path.join(ROOT, "types", f"{tname}.py")
    if os.path.exists(path):
        from .pipeline.stages_plan import load_type
        try:
            fn = load_type(path).get("form_plan")
        except Exception:                        # noqa: BLE001 -- no plan, envelope path
            fn = None
    _PLANS[tname] = fn if callable(fn) else None
    return _PLANS[tname]


def pad_of(lot_w: int, lot_d: int, front: str, attached=(), inset: int = 1) -> tuple:
    """(pad along the street, pad from the street back) of a lot `lot_w` along its
    street and `lot_d` deep, with party walls on `attached` (world sides)."""
    front = front if front in FLANKS else "north"
    if front in ("north", "south"):
        W, D = lot_w, lot_d
    else:
        W, D = lot_d, lot_w
    iw, iN, ie, iS = pad_insets(W, D, list(attached or ()), inset)
    pw, pd = W - iw - ie, D - iN - iS
    return (pw, pd) if front in ("north", "south") else (pd, pw)


def plan(tname: str, lot_w: int, lot_d: int, *, params: dict | None = None,
         front: str = "north", attached=(), inset: int = 1, decl: dict | None = None):
    """The type's plan for this lot, or None where the type publishes none."""
    fn = plan_fn(tname, decl)
    if fn is None:
        return None
    pw, pd = pad_of(lot_w, lot_d, front, attached, inset)
    got = dict(fn(int(pw), int(pd), params=dict(params or {}), front=front,
                  attached=list(attached or ())))
    got.setdefault("pad", [int(pw), int(pd)])
    got["lot"] = [int(lot_w), int(lot_d)]
    got["flanks"] = sorted(attached or ())
    return got


def plan_for(tname: str, leaf: dict, params: dict | None = None, inset: int = 1):
    """`plan` for a plan leaf (its rectangle, `front` and `attached`)."""
    front = leaf.get("front") or "north"
    w = leaf["x1"] - leaf["x0"] + 1
    d = leaf["z1"] - leaf["z0"] + 1
    lot_w, lot_d = (w, d) if front in ("north", "south") else (d, w)
    return plan(tname, lot_w, lot_d, params=params if params is not None
                else leaf.get("params"), front=front,
                attached=leaf.get("attached") or (), inset=int(leaf.get("inset") or inset))


def configs(front: str = "north") -> list:
    """The four flank configurations a lot in a run can be in: middle, either end, alone."""
    lo, hi = FLANKS[front]
    return [(lo, hi), (lo,), (hi,), ()]


def least_lot(tname: str, params: dict | None, *, front: str = "north", attached=(),
              inset: int = 1, decl: dict | None = None, w_max: int = 40,
              d_max: int = 40) -> dict | None:
    """`{"lot": [w, d], "plan"}` -- the least lot (width first) this type's plan admits
    with these parameters in this flank configuration, or `{"lot": None, "why"}`."""
    fn = plan_fn(tname, decl)
    if fn is None:
        return None
    last = None
    for d in range(3, d_max + 1):
        for w in range(3, w_max + 1):
            got = plan(tname, w, d, params=params, front=front, attached=attached,
                       inset=inset, decl=decl)
            last = got
            if got.get("ok"):
                # the least width at this depth; then is any shallower depth enough at
                # this width? (scan order is depth-major, so this is the least depth)
                return {"lot": [w, d], "plan": got}
    return {"lot": None, "plan": last,
            "why": f"no lot up to {w_max}x{d_max} holds {tname} with {params}"}


def admits(tname: str, lot_w: int, lot_d: int, params: dict | None, *,
           front: str = "north", inset: int = 1, decl: dict | None = None,
           configs_=None) -> bool | None:
    """Does a lot of this size hold the plan in every flank configuration a run may
    leave it in (middle, either end)? None where the type has no plan."""
    if plan_fn(tname, decl) is None:
        return None
    for att in (configs_ if configs_ is not None else configs(front)):
        got = plan(tname, lot_w, lot_d, params=params, front=front, attached=att,
                   inset=inset, decl=decl)
        if not got.get("ok"):
            return False
    return True


# ------------------------------------------------------------------ the built reading

_AIR = ("air", "cave_air", "void_air")
#: what a person can stand in: air and the passable things a room is furnished with
_PASS = ("air", "torch", "lantern", "carpet", "button", "pressure_plate", "sign",
         "banner", "flower_pot", "candle", "rail", "door", "trapdoor")


def _name(st: str) -> str:
    return str(st).split("[")[0].split(":")[-1]


def _passable(vol, x, y, z) -> bool:
    if not vol.inside(x, y, z):
        return True
    n = _name(vol.state(x, y, z))
    return n in _AIR or any(k in n for k in _PASS[1:])


#: What a wall is built of (a voice's wall, frame and footing families), as opposed to
#: what stands in a room: a room's clear depth is wall to wall, whatever furniture is in
#: it -- a bed across the measuring line is still a room.
_STRUCTURE = ("planks", "log", "wood", "brick", "stone", "terracotta", "concrete",
              "sandstone", "quartz", "mud", "deepslate", "granite", "andesite", "diorite",
              "tuff", "calcite", "copper", "prismarine", "basalt", "blackstone", "clay",
              "wool", "glass", "pane", "bars", "fence", "wall")


def _structural(vol, x, y, z) -> bool:
    if not vol.inside(x, y, z):
        return False
    n = _name(vol.state(x, y, z))
    if n in _AIR or "door" in n or "stairs" in n or "slab" in n or "trapdoor" in n:
        return False
    return any(k in n for k in _STRUCTURE)


def _clear_run(vol, y, cells) -> int:
    """The longest run of cells along `cells` with no wall at foot or head height
    (furniture is in a room, not a wall of it)."""
    best = cur = 0
    for (x, z) in cells:
        if not _structural(vol, x, y, z) and not _structural(vol, x, y + 1, z):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def measure(vol, tname: str, part: dict, emitted: dict | None) -> dict:
    """The critical dimensions of what stands, read off the blocks.

    For a courtyard house: the court (open to the sky, measured) and, for every range,
    the clear depth between its outer and court walls, measured on the ground floor along
    three lines across it. For a shop house: the storeys whose floor a flight reaches and
    the ground-floor clear width. Anything the type does not publish is omitted."""
    em = emitted or {}
    fy = int(part["floor_y"])
    out: dict = {}
    if tname == "courtyard_house":
        court = (em.get("court") or {}).get("yard")
        ranges = ((em.get("rects") or {}))
        main = ranges.get("main") or [part["x0"], part["z0"], part["x1"], part["z1"]]
        if court:
            cx0, cz0, cx1, cz1 = court
            open_ = 0
            top = vol.y0 + vol.codes.shape[1]
            for x in range(cx0, cx1 + 1):
                for z in range(cz0, cz1 + 1):
                    if all(_name(vol.state(x, y, z)) in _AIR
                           for y in range(fy + 3, top) if vol.inside(x, y, z)):
                        open_ += 1
            out["court"] = [cx1 - cx0 + 1, cz1 - cz0 + 1]
            out["court_open"] = round(open_ / max(1, (cx1 - cx0 + 1) * (cz1 - cz0 + 1)), 2)
            x0, z0, x1, z1 = main
            ent = (em.get("court") or {}).get("entrance")
            hall = (em.get("court") or {}).get("hall")
            use = {ent: "gate", hall: "hall"}
            y = fy + 1
            depths = {}
            for side in ("north", "south", "west", "east"):
                runs = []
                if side in ("north", "south"):
                    zs = range(z0, cz0) if side == "north" else range(cz1 + 1, z1 + 1)
                    for x in (cx0 + 1, cx1 - 1):
                        runs.append(_clear_run(vol, y, [(x, z) for z in zs]))
                else:
                    xs = range(x0, cx0) if side == "west" else range(cx1 + 1, x1 + 1)
                    for z in (cz0 + 1, cz1 - 1):
                        runs.append(_clear_run(vol, y, [(x, z) for x in xs]))
                depths[use.get(side, f"wing_{side}")] = max(runs) if runs else 0
            out["clear"] = depths
    elif tname == "shop_house":
        out["storeys"] = em.get("storeys")
        main = (em.get("rects") or {}).get("main")
        if main:
            x0, z0, x1, z1 = main
            front = part.get("front") or part.get("facing") or "north"
            # the shop floor's clear width, across the plan, on the best line of its
            # front half (a flight or a partition behind the shop is not the shop)
            if front in ("north", "south"):
                zs = (range(z0 + 1, (z0 + z1) // 2 + 1) if front == "north"
                      else range((z0 + z1) // 2, z1))
                out["clear_width"] = max([_clear_run(vol, fy + 1, [(x, zm) for x in
                                                                   range(x0, x1 + 1)])
                                          for zm in zs] or [0])
            else:
                xs = (range(x0 + 1, (x0 + x1) // 2 + 1) if front == "west"
                      else range((x0 + x1) // 2, x1))
                out["clear_width"] = max([_clear_run(vol, fy + 1, [(xm, z) for z in
                                                                   range(z0, z1 + 1)])
                                          for xm in xs] or [0])
        stairs = 0
        x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                for y in range(fy + 1, fy + 5):
                    if vol.inside(x, y, z) and "stairs" in _name(vol.state(x, y, z)):
                        stairs += 1
        out["stair_blocks"] = stairs
    return out


def plan_image(vol, rect, fy: int, *, scale: int = 10, level: int = 1, floors=None):
    """A legible floor plan of `rect` at `fy + level`: walls dark, doors and openings
    orange, roofed floor (a room) light, floor open to the sky (a court or lane) green-
    grey, stairs blue, furniture and fittings purple, bare ground and outside grass.

    The top-down render of a cut volume shows wall tops and floor textures in one tone;
    this is the drawing a reader needs to say whether rooms are rooms."""
    import numpy as np
    x0, z0, x1, z1 = rect
    W, H = x1 - x0 + 1, z1 - z0 + 1
    img = np.zeros((H * scale, W * scale, 3), np.uint8)
    top = vol.y0 + vol.codes.shape[1]
    # `floors`: [(rect, floor_y)] -- each building drawn at its own floor, and ground
    # outside every rect at its own surface
    heights = None
    if floors is not None:
        from .observe import ground_heights
        heights, _w = ground_heights(vol)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            y = fy + level
            if floors is not None:
                hit = next((f for (r, f) in floors
                            if r[0] <= x <= r[2] and r[1] <= z <= r[3]), None)
                if hit is not None:
                    y = int(hit) + level
                elif vol.inside(x, vol.y0, z):
                    y = int(heights[x - vol.x0, z - vol.z0]) + 1
            n = _name(vol.state(x, y, z)) if vol.inside(x, y, z) else "air"
            below = _name(vol.state(x, y - 1, z)) if vol.inside(x, y - 1, z) else "air"
            roofed = any(vol.inside(x, yy, z) and _name(vol.state(x, yy, z)) not in _AIR
                         for yy in range(y + 2, min(top, y + 12)))
            if "door" in n and "trap" not in n:
                c = (230, 140, 40)
            elif "stairs" in n or "ladder" in n:
                c = (60, 110, 220)
            elif n in _AIR:
                if below in _AIR:
                    c = (40, 40, 40)
                elif below in ("grass_block", "dirt", "podzol", "coarse_dirt"):
                    c = (110, 160, 90) if not roofed else (200, 180, 140)
                else:
                    c = (235, 225, 200) if roofed else (160, 170, 160)
            elif any(k in n for k in ("glass", "pane")):
                c = (150, 210, 230)
            elif any(k in n for k in _PASS[1:]) or any(k in n for k in (
                    "bed", "chest", "barrel", "table", "lectern", "shelf", "cauldron",
                    "furnace", "anvil", "composter", "stall", "crafting", "loom",
                    "smoker", "well", "slab", "fence", "wall_", "leaves", "fern")):
                c = (170, 90, 170)
            else:
                c = (70, 55, 45)
            img[(z - z0) * scale:(z - z0 + 1) * scale, (x - x0) * scale:(x - x0 + 1) * scale] = c
    # a faint grid, one line per five blocks
    for k in range(0, W, 5):
        img[:, k * scale, :] = (img[:, k * scale, :] * 0.7).astype(np.uint8)
    for k in range(0, H, 5):
        img[k * scale, :, :] = (img[k * scale, :, :] * 0.7).astype(np.uint8)
    return img


def _side_of(door, main) -> str | None:
    if not door or not main:
        return None
    x0, z0, x1, z1 = main
    dx, dz = int(door[0]), int(door[-1])
    if dz <= z0:
        return "north"
    if dz >= z1:
        return "south"
    if dx <= x0:
        return "west"
    if dx >= x1:
        return "east"
    return None


_OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def measure_row(vol, row: dict) -> dict | None:
    """`measure` for a built row of `parts.json` (its construction outcome), on the
    assembled world. None where the row carries no geometry this can read."""
    em = row.get("emitted") or {}
    rects = em.get("rects") or {}
    main = rects.get("main") or row.get("footprint")
    if not main or row.get("floor_y") is None:
        return None
    part = {"x0": main[0], "z0": main[1], "x1": main[2], "z1": main[3],
            "floor_y": int(row["floor_y"])}
    tname = str(row.get("type"))
    if tname == "courtyard_house":
        yard = rects.get("courtyard")
        if not yard:
            return {"court": None}
        ent = _side_of(row.get("door"), main)
        e2 = {"court": {"yard": yard, "entrance": ent, "hall": _OPP.get(ent)},
              "rects": {"main": main}}
        return measure(vol, tname, part, e2)
    if tname == "shop_house":
        front = _side_of(row.get("door"), main) or "north"
        part["front"] = front
        return measure(vol, tname, part, {"storeys": em.get("storeys"),
                                          "rects": {"main": main}})
    return None


def row_meets(tname: str, got: dict | None, params: dict | None) -> tuple:
    """`(ok, why)` -- do the dimensions measured on the world meet what the form owes?
    A type without a form plan has nothing here to meet and answers `(None, ...)`."""
    params = dict(params or {})
    if got is None:
        return None, "no form measured"
    fn = plan_fn(tname)
    if fn is None:
        return None, f"{tname} publishes no form plan"
    if tname == "courtyard_house":
        mod = _module(tname)
        mins = dict(getattr(mod, "ROOM_CLEAR", {}) or {}) if mod else {}
        clear = got.get("clear") or {}
        short = {k: v for k, v in clear.items()
                 if v < int(mins.get("hall" if k == "hall" else "gate" if k == "gate"
                                     else "wing", 0))}
        court = got.get("court") or [0, 0]
        want = max(int(params.get("court") or 0), int(getattr(mod, "COURT_LEAST", 0) or 0))
        small = min(court) < want if court else True
        ok = not short and not small and (got.get("court_open") or 0) >= 0.8
        return ok, ("rooms " + ", ".join(f"{k} {v}" for k, v in sorted(clear.items()))
                    + f"; court {court[0]}x{court[1]} (owed {want}), "
                    f"open {got.get('court_open')}"
                    + (f"; short: {short}" if short else ""))
    if tname == "shop_house":
        want = int(params.get("storeys") or 1)
        st = int(got.get("storeys") or 0)
        mod = _module(tname)
        ok = st >= want and (got.get("clear_width") or 0) >= int(
            getattr(mod, "SHOP_CLEAR", 3) if mod else 3)
        return ok, (f"{st} storey(s) of {want} owed, shop {got.get('clear_width')} "
                    f"clear, {got.get('stair_blocks')} stair block(s)")
    return None, "unmeasured"


def _module(tname: str):
    import importlib.util
    path = os.path.join(ROOT, "types", f"{tname}.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(f"_ethoslm_type_{tname}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:                            # noqa: BLE001
        return None
    return mod
