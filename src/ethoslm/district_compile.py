"""The district compiler: a district's plan from its **character**, and no model asked.

v2, C1. A district's plan is read from one file at one seam
(`stages_plan.district_asks`: `plan.district.<name>.json`), and until this a model
drew that file -- every plot a rectangle a model wrote, five hundred times a city,
judged after the fact by `placeplan.district_failures`. Where the district's defining
part carries a **character** -- the density word and the role it always had, and now a
frontage rule, a block size, a lot depth, attached or detached, a courtyard share, an
open-ground share and fixed landmarks (`spec.CHARACTER_FIELDS`) -- this writes the file
instead, deterministically under the seed, and the validator, the assembler and the
circulation read it exactly as they read a model's. The model writes the character in
the district brief's place: a paragraph of words about what the district is like, not
four hundred rectangles.

What it lays, in order:

  1. **streets**, from the arterial's band down, at the block size: a lane's width
     (`placeplan.PLOT_LANE`) between blocks along the streets and across them, and the
     arterial routed before the district was asked is a street already;
  2. **blocks** between them, each two rows of lots back to back (`lot_depth` deep,
     a clearance between the backs), and each block one of: a **row** block, a
     **courtyard** block (the back row is a court the front row shares), an **open**
     block (fields, gardens, groves or a plaza, by the role and the density), or the
     **landmark** block nearest the district's middle where the character names one;
  3. **lots** along each street face at the frontage width the admitted types declare
     -- the lot the density word means (`placeplan.occupancy_shares`'s plot side)
     clamped into the house type's own envelope, never a pad its sweep found broken --
     detached by a clearance, or touching where the character says `attached`;
  4. **buildings** on the lots, the role's house type and every fourth one another of
     the role's types, seeds and parameters varied under the district's seed, each
     carrying `front`: the side its street is on, so the way in is on the street;
  5. **leftover ground assigned**: every column of the district is one of `LEDGER` --
     street, lot, court, garden, plaza, field, grove, verge -- or `undeveloped`, and
     the record says how many of each.

Nothing here reads the ground: the contract (`ethoslm.ground`) settles every lot's
level after the plan, and `pipeline.plan_ground` classifies it before. Nothing here
names a place, a type or a voice: the types are the table the district's role and
form admit.
"""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np

from . import pipeline, spec as spec_mod
from .buildlib import Builder

#: **Registered before it was read** (v2, C1): the least of a compiled district's
#: rectangle its **plots** cover, per density word. The validator's own floor --
#: `placeplan.DISTRICT_MIN_FRACTION` of the density's plot share
#: (`placeplan.occupancy_shares`) -- rounded to two places, written here as numbers so
#: the compiler's proof reads against a bar and not against the arithmetic it is being
#: tested with.
PLOT_COVER = {"sparse": 0.15, "low": 0.21, "medium": 0.25, "dense": 0.25}

#: What every column of a compiled district is, in the record. `undeveloped` is what no
#: rule assigned, and the run's bar is on it.
LEDGER = ("undeveloped", "street", "lot", "court", "garden", "plaza", "field",
          "grove", "verge")

#: The clearance between two things that are not a street apart: the larger of the two
#: `keeps clear` figures plus one, as the validator holds any pair to. Every committed
#: plot type keeps two clear and every area one, so `LOT_GAP` is what anything must
#: leave a **lot** and `AREA_GAP` what one piece of open ground must leave the next.
#: Holding open ground to a lot's clearance is how a compiled district left a fifth of
#: itself assigned to nothing.
LOT_GAP = 3
AREA_GAP = 2

#: How many lots long a block is, per density word, where the character does not say. A
#: loose fabric on a three-lot block spends a quarter of its ground on streets -- which
#: is a suburb, not farmland -- so the looser the density the longer the block. The
#: compiler lengthens it further where the ground is still short of its cover.
BLOCK_LOTS = {"sparse": 5, "low": 4, "medium": 3, "dense": 3}
BLOCK_LOTS_MAX = 8

#: How much of a block's length the leftover along a street may be before the lots are
#: widened to take it up: under this, the remainder is a wider last gap.
WIDEN_UP_TO = 0.5

#: Every fourth lot takes another of the role's types rather than the house, where the
#: role has more than one plot type that admits the lot.
OTHER_EVERY = 4


# ------------------------------------------------------------------- the character

def character_of(part: dict) -> dict:
    """The full character of a district part: the spec's words over
    `spec.CHARACTER_DEFAULTS` for its density word, with the numbers the defaults
    leave to the density (`block`, `lot_depth`) filled from the density's own lot."""
    density = part.get("density") or "medium"
    out = dict(spec_mod.CHARACTER_DEFAULTS[density])
    for k, v in (part.get("character") or {}).items():
        if v is not None:
            out[k] = v
    out["density"] = density
    out["role"] = part.get("role")
    return out


def _lot_side(density: str) -> int:
    """The side of the square lot a density word means: `occupancy_shares`' plot side."""
    from .placeplan import occupancy_shares
    return int(occupancy_shares()[density]["plot_side"])


def _plot_range(decl: dict) -> tuple:
    """(lo, hi) of a plot side this type admits, in plot columns (the pad plus the
    inset), and the pad sides its sweep found broken."""
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 2 * Builder.SITE_INSET
    ex = set(decl["needs"].get("except") or ())
    return (max(lo_w, lo_d) + i, min(hi_w, hi_d) + i, ex)


def _admits(decl: dict, w: int, d: int, attached=None) -> bool:
    """Does this plot type stand on a w x d plot, with these sides attached? Its
    envelope and its except list, read in the validator's own units
    (`needs_footprint_failure`, through `Builder.pad_extent`)."""
    part = {"kind": "plot", "x0": 0, "z0": 0, "x1": w - 1, "z1": d - 1}
    if attached:
        part["attached"] = list(attached)
    return pipeline.needs_footprint_failure(part, decl["needs"]) is None


def _attached_lot(decl: dict, side: int, flanks: tuple) -> tuple | None:
    """(w, d) of a lot in a row of party walls: the size nearest the density's own
    that this type admits **with both flanks attached, with one, and with none** --
    because a neighbour dropped for the road or a standing part frees a flank, and
    the pad's inset comes back on it (`Builder._insets`). None where no size does."""
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 2 * Builder.SITE_INSET
    ws = sorted(range(max(3, lo_w), hi_w + i + 1), key=lambda v: (abs(v - side), v))
    ds = sorted(range(max(3, lo_d), hi_d + i + 1), key=lambda v: (abs(v - side), v))
    for w in ws:
        for d in ds:
            if all(_admits(decl, w, d, att) for att in (flanks, flanks[:1], ())):
                return w, d
    return None


def _clamp_side(n: int, decl: dict) -> int:
    """The plot side nearest `n` this type admits: inside its envelope and off its
    except list, the nearer side first and the smaller on a tie."""
    lo, hi, _ex = _plot_range(decl)
    n = max(lo, min(hi, n))
    if _admits(decl, n, n):
        return n
    for k in range(1, hi - lo + 2):
        for cand in (n - k, n + k):
            if lo <= cand <= hi and _admits(decl, cand, cand):
                return cand
    return n


def _deepest(decl: dict, w: int, most: int, flanks=()) -> int:
    """The deepest lot of width `w` this type admits inside `most` columns -- checked
    at the width it will actually be built at, and in every configuration of its
    flanks where it is attached. 0 where none does. A block too shallow for two rows
    takes one deep one, and asking whether a *square* of that depth is admitted said
    yes to a 6x10 lot of a type written for a 6x6 pad."""
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 2 * Builder.SITE_INSET
    atts = [tuple(flanks), tuple(flanks[:1]), ()] if flanks else [()]
    for d in range(min(most, hi_d + i), max(3, lo_d) - 1, -1):
        if all(_admits(decl, w, d, a) for a in atts):
            return d
    return 0


def house_types(decls: dict, role: str | None, form: str | None = None) -> list:
    """The plot types a district of this role may build, the role's own first."""
    out = []
    for name, d in sorted(decls.items()):
        if d.get("kind", "plot") != "plot":
            continue
        if not pipeline.role_ok(d.get("role"), role):
            continue
        if form and not pipeline.form_ok(d.get("form"), form):
            continue
        out.append((0 if d.get("role") == role else 1, name, d))
    out.sort()
    return [(n, d) for _r, n, d in out]


def area_types(decls: dict, role: str | None, form: str | None = None) -> dict:
    """{name: declaration} of the area types this district may draw."""
    out = {}
    for name, d in sorted(decls.items()):
        if d.get("kind") != "area":
            continue
        if not pipeline.role_ok(d.get("role"), role):
            continue
        if form and not pipeline.form_ok(d.get("form"), form):
            continue
        out[name] = d
    return out


def _open_type(areas: dict, role: str | None, density: str) -> str | None:
    """What an open block is: fields for a rural district, a plaza for a dense one,
    gardens and groves for the rest -- the first the district's table has."""
    if role == "rural":
        order = ("field", "grove", "garden")
    elif density == "dense":
        order = ("plaza", "garden", "yard")
    else:
        order = ("garden", "grove", "plaza")
    return next((t for t in order if t in areas), next(iter(areas), None))


def _court_type(areas: dict, role: str | None) -> str | None:
    order = ("yard", "garden", "plaza") if role == "urban" else ("garden", "yard", "plaza")
    return next((t for t in order if t in areas), next(iter(areas), None))


def _verge_type(areas: dict, role: str | None) -> str | None:
    order = ("grove", "garden", "field") if role != "rural" else ("field", "grove", "garden")
    return next((t for t in order if t in areas), next(iter(areas), None))


def _area_tiles(u0: int, u1: int, v0: int, v1: int, decl: dict, gap: int) -> list:
    """Split a rectangle into tiles this area type admits: at most its largest side,
    off its except list, `gap` apart. (u0, u1, v0, v1) inclusive, in the block's frame."""
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    ex = set(decl["needs"].get("except") or ())
    hi = min(hi_w, hi_d)
    lo = max(lo_w, lo_d)

    def cuts(a, b):
        n = b - a + 1
        if n < lo:
            return []
        k = max(1, math.ceil((n + gap) / (hi + gap)))
        while True:
            side = (n - (k - 1) * gap) // k
            if side < lo:
                return []
            sizes = [side] * k
            extra = n - (k - 1) * gap - side * k
            for i in range(extra):
                sizes[i % k] += 1
            if all(s not in ex and lo <= s <= hi for s in sizes):
                break
            # a side the sweep found broken: one tile more, so every side shrinks
            k += 1
        out, at = [], a
        for s in sizes:
            out.append((at, at + s - 1))
            at += s + gap
        return out

    return [(ua, ub, va, vb) for ua, ub in cuts(u0, u1) for va, vb in cuts(v0, v1)]


# ---------------------------------------------------------------------- the grid

class _Frame:
    """The district's rectangle in (u, v): u runs along the streets, v across them.
    Streets run along the district's longer side."""

    def __init__(self, x0, z0, x1, z1):
        self.x0, self.z0, self.x1, self.z1 = x0, z0, x1, z1
        w, d = x1 - x0 + 1, z1 - z0 + 1
        self.u_is_x = w >= d
        self.U = w if self.u_is_x else d
        self.V = d if self.u_is_x else w

    def rect(self, u0, u1, v0, v1) -> tuple:
        """(x0, z0, x1, z1) of a (u, v) rectangle."""
        u0, u1, v0, v1 = int(u0), int(u1), int(v0), int(v1)
        if self.u_is_x:
            return (self.x0 + u0, self.z0 + v0, self.x0 + u1, self.z0 + v1)
        return (self.x0 + v0, self.z0 + u0, self.x0 + v1, self.z0 + u1)

    def front(self, toward_low_v: bool) -> str:
        """The side name of a street at lower (or higher) v."""
        if self.u_is_x:
            return "north" if toward_low_v else "south"
        return "west" if toward_low_v else "east"

    def flanks(self) -> tuple:
        """The side names of a lot's two flanks, at lower u and at higher u."""
        return ("west", "east") if self.u_is_x else ("north", "south")


def _runs(start: int, end: int, street: int, block: int, least: int,
          lead: bool = True, widen: bool = True) -> list:
    """[(from, to)] of the blocks along one axis inside [start, end]: a street, a
    block, a street... as many blocks of `block` as fit; a remainder of at least
    `least` after a street is one more, shorter block, and a smaller remainder
    **widens the blocks evenly**, so no strip at the end of a run is nobody's. A run
    shorter than a block is one block, down to `least`. `lead` false starts with a
    block: the interval begins at a street that is already there (the arterial's
    band)."""
    n = end - start + 1
    room = n - (street if lead else 0)
    if room < least:
        return []
    k = max(1, (room + street) // (block + street))
    sizes = [min(block, room)] * k
    left = room - (k * min(block, room) + (k - 1) * street)
    if left - street >= least:
        sizes.append(left - street)
    elif widen:
        for i in range(left):
            sizes[i % k] += 1
    out, at = [], start + (street if lead else 0)
    for s in sizes:
        out.append((at, at + s - 1))
        at += s + street
    return out


#: An arterial's band is a **street of the grid** where it runs along one axis of the
#: district for at least this share of that axis and is no deeper across it than this
#: many lanes: the blocks are laid on either side of it. A road that crosses the
#: district any other way is a band the lots keep off. **The width test earns its
#: place.** Dropping it -- on the argument that a road is a road whatever its width, and
#: the blocks should go beside it -- splits both axes at every band and fragments the
#: grid into pieces too short for a block: a district that laid 30 lots laid 6. A wide
#: swathe of road is an obstacle, not a street.
AXIAL_SHARE = 0.4
AXIAL_DEPTH = 3


def _intervals(present: np.ndarray) -> list:
    """[(from, to)] of the runs of False in a boolean array."""
    out, at = [], None
    for i, on in enumerate(present):
        if not on and at is None:
            at = i
        if on and at is not None:
            out.append((at, i - 1))
            at = None
    if at is not None:
        out.append((at, len(present) - 1))
    return out


def _grid_axis(band: np.ndarray, axis: int, total: int, street: int, block: int,
               least: int, lead: bool = True, widen: bool = True) -> tuple:
    """The block runs along one axis (0: u, 1: v), the arterial's band a street of
    the grid where it is axial, else a lead street at the district's edge -- unless
    `lead` is off: a strip too thin for a street and a lot fronts its own edge, where
    the ground beyond it (a road, the next district's street) is the way in.
    Returns (runs, axial)."""
    across = band.any(axis=1 - axis)          # True where the band reaches this line
    along = band.any(axis=axis)               # the band's extent along the other axis
    axial = bool(across.any()) and along.sum() >= AXIAL_SHARE * band.shape[1 - axis] \
        and across.sum() <= AXIAL_DEPTH * street
    if not axial:
        return _runs(0, total - 1, street, block, least, lead=lead, widen=widen), False
    out = []
    for k, (a, b) in enumerate(_intervals(across)):
        out += _runs(a, b, street, block, least, lead=(a == 0 and lead), widen=widen)
    return out, True


def _largest_free(free: np.ndarray) -> tuple | None:
    """The largest all-True axis-aligned rectangle of a boolean matrix, as
    (u0, u1, v0, v1), or None where there is none. The histogram method."""
    U, V = free.shape
    if not free.any():
        return None
    h = np.zeros(V, dtype=int)
    best, best_area = None, 0
    for u in range(U):
        h = np.where(free[u], h + 1, 0)
        stack: list = []
        for v in range(V + 1):
            cur = h[v] if v < V else 0
            start = v
            while stack and stack[-1][1] >= cur:
                sv, sh = stack.pop()
                area = sh * (v - sv)
                if area > best_area:
                    best_area, best = area, (u - sh + 1, u, sv, v - 1)
                start = sv
            stack.append((start, cur))
    return best


def _seeded(seed: int, *parts) -> float:
    h = hashlib.sha256(("|".join(str(p) for p in (seed, *parts))).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


# ------------------------------------------------------------------- the compiler

#: How far the compiler raises a district's courtyard share, then its open share, a step
#: at a time, when the character's own shares leave the ground under the cover the
#: density word asks (`placeplan.district_target`): a court behind a row and an open
#: block are the ground between the houses, and the character said how much of it there
#: is only approximately.
COVER_STEP = 0.1

#: How much shallower than the density's own lot the compiler will make a lot, a column
#: at a time, when the count the district was asked for is under its floor: a deep block
#: whose middle nobody owns is two more rows of slightly shallower lots.
DEPTH_GIVE = 4

#: ...and how much **bigger** than the lot the ask implies the compiler will make one, a
#: column at a time, where the houses it is allowed to lay do not cover the ground its
#: density asks to be built on. A district asked for three houses on seven thousand
#: columns covers its share with three big plots or not at all.
LOT_GROW_MAX = 10

#: A landmark's plot is at most this many of the density's lot sides across: a hall at
#: its largest admitted plot took a third of a small district for one building.
LANDMARK_SIDES = 2

#: The columns a compiled district keeps free inside every edge of its rectangle: the
#: place level tiles districts edge to edge, and two districts' lots each on their own
#: edge are within the clearance the assembled plan holds any pair to. Two a side is a
#: lane's worth between neighbours, and it is a street in the ledger.
EDGE_MARGIN = 2


def compile_district(district: dict, part: dict, place: dict, decls: dict, *,
                     spec: dict | None = None, seed: int = 1) -> tuple:
    """One district's plan file from its character. Returns (the file, the record).

        `district` is the place plan's rectangle (`x0..z1`, `structures`, `name`), `part`
        the defining part it answers (with `density`, `role` and `character`), `place` the
        place plan (its `arterials` and its standing `parts`), `decls` the types the
        district's role and form admit (`placeplan.types_card(..., role)`).

        Compiled once as the character says; where the plots and the areas cover less of
        the rectangle than the density's ground cover asks, compiled again with the
        courtyard share a `COVER_STEP` higher, then the open share, until it is met or
        both are 1 -- and the record says what it had to raise.
        
    """
    from .placeplan import district_target
    ch0 = character_of(part)
    t = district_target(district, part)
    tries = []
    ch = dict(ch0, _lead=True)

    # **The cover the validator asks of this district**, which for a rural one is
    # `RURAL_COVER` over its whole rectangle and for every other the density's own share
    # of it: the number the compiler works to and the number it is refused on are one
    # number.
    from .placeplan import RURAL_COVER
    want_ground = max(t["min_ground_columns"],
                      int(math.ceil(RURAL_COVER * t["columns"]))
                      if ch0.get("role") == "rural" else 0)

    def score(rec):
        plots = rec["plot_cover"] * rec["columns"]
        ground = rec["ground_cover"] * rec["columns"]
        return (rec["lots"] >= t["min_count"], plots >= t["min_plot_columns"],
                ground >= want_ground, rec["ground_cover"], rec["plot_cover"])

    best = None
    seen = set()
    while True:
        key = (round(ch["courtyard_share"], 2), round(ch["open_share"], 2),
               ch.get("lot_depth"), ch["_lead"], ch.get("_block_lots"),
               ch.get("_lot_grow"))
        if key in seen:
            break
        seen.add(key)
        got, rec = _compile_once(district, part, place, decls, ch, spec=spec, seed=seed)
        tries.append({"courtyard_share": ch["courtyard_share"],
                      "open_share": ch["open_share"], "lead_street": ch["_lead"],
                      "block_lots": ch.get("_block_lots"), "block": rec["block"],
                      "ground_cover": rec["ground_cover"],
                      "plot_cover": rec["plot_cover"], "lots": rec["lots"]})
        if best is None or score(rec) > score(best[1]):
            best = (got, rec, dict(ch))
        ok_lots, ok_plots, ok_ground = score(rec)[:3]
        if ok_lots and ok_plots and ok_ground:
            break
        # **The levers, in the order the fabric answers to them.** Every one of these is
        # a number the character gave approximately and the compiler is answerable for
        # exactly: the count and the cover of the district's own density. 1. the lead
        # street: a strip too thin for a street and two rows of lots gives its street up
        # and fronts the ground beyond its own edge
        if (not ok_plots or not ok_lots) and ch["_lead"]:
            ch["_lead"] = False
            continue
        # 2. the shares, **down**: a district short of houses has given whole blocks to
        # open ground its own density wants built on. This is the lever the cover
        # raises, run the other way -- a strip beside a place's centre came back four
        # houses in seven thousand columns, two thirds of it field, and nothing in the
        # loop could turn a field back into a house.
        if not ok_plots or not ok_lots:
            if ch["open_share"] > 0:
                ch["open_share"] = max(0.0, round(ch["open_share"] - COVER_STEP, 2))
                continue
            if ch["courtyard_share"] > 0:
                ch["courtyard_share"] = max(0.0,
                                            round(ch["courtyard_share"] - COVER_STEP, 2))
                continue
        # 3. the lot, **up**: the count is a number and not a floor, so where the houses
        # the district is allowed to lay do not cover the ground its density asks to be
        # built on, those houses are bigger -- deeper as well as wider, which is why any
        # depth the loop gave away below is given back here.
        if not ok_plots and not ch0.get("lot_depth") \
                and int(ch.get("_lot_grow") or 0) < LOT_GROW_MAX:
            ch["_lot_grow"] = int(ch.get("_lot_grow") or 0) + 1
            ch["lot_depth"] = None
            continue
        # 4. the lot, **down**: a shallower lot, a column at a time, down to
        # `DEPTH_GIVE` under the density's own, where a deep block's middle becomes two
        # more rows and the district is still short of houses
        if not ok_lots:
            deep = _lot_side(ch["density"])
            have = int(ch.get("lot_depth") or rec["lot"][1])
            if have > deep - DEPTH_GIVE and have > 3:
                ch["lot_depth"] = have - 1
                continue
            break
        # 5. the block: a longer block is fewer streets, and a loose fabric's ground
        # goes to streets before it goes anywhere else
        here = int(ch.get("_block_lots") or BLOCK_LOTS.get(ch["density"], 3))
        if not ok_ground and not ch0.get("block") and here < BLOCK_LOTS_MAX:
            ch["_block_lots"] = here + 1
            continue
        # 6. the shares, **up**: what is left of the cover is whole blocks of open
        # ground and courts behind the rows
        if ch["courtyard_share"] < 1.0:
            ch["courtyard_share"] = min(1.0, round(ch["courtyard_share"] + COVER_STEP, 2))
        elif ch["open_share"] < 1.0:
            ch["open_share"] = min(1.0, round(ch["open_share"] + COVER_STEP, 2))
        else:
            break
    got, rec, ch = best
    rec["tries"] = tries
    rec["raised"] = {k: [ch0.get(k), ch[k]]
                     for k in ("courtyard_share", "open_share", "lot_depth",
                               "_block_lots", "_lot_grow")
                     if ch.get(k) != ch0.get(k)}
    rec["want_ground_columns"] = int(want_ground)
    if not ch["_lead"]:
        rec["raised"]["lead_street"] = [True, False]
    return got, rec


def _compile_once(district: dict, part: dict, place: dict, decls: dict, ch: dict, *,
                  spec: dict | None = None, seed: int = 1) -> tuple:
    from .placeplan import PLOT_LANE, district_target
    target = district_target(district, part)
    density, role = ch["density"], ch.get("role")
    form = (spec or {}).get("form")
    houses = house_types(decls, role, form)
    areas = area_types(decls, role, form)
    if not houses:
        raise ValueError(f"{district['name']}: no committed type of the form "
                         f"{form or 'any'} builds a {role or 'district'} house")
    X0, X1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    Z0, Z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    W_all, D_all = X1 - X0 + 1, Z1 - Z0 + 1
    edge = EDGE_MARGIN if min(W_all, D_all) > 2 * EDGE_MARGIN + 3 else 0
    fr = _Frame(X0 + edge, Z0 + edge, X1 - edge, Z1 - edge)
    ledger = np.zeros((fr.U, fr.V), dtype=np.uint8)
    L = {n: i for i, n in enumerate(LEDGER)}

    # --- the lot ------------------------------------------------------------- **The
    # lot is what this district's own ask and its own ground imply**, not what the
    # density word means in the abstract. v2, C5: the compiler filled every district at
    # the density's lot size and laid 102 houses in a village asked for 26 -- four times
    # its kind's whole band -- because the count it was given was read as a floor and
    # never as the number. The count is the district's `structures`; the cover is
    # `district_target`'s share of its rectangle; a lot is the second divided by the
    # first, clamped into what the house type admits. Where a district is asked for
    # nothing, it lays no lots.
    want_lots = int(district.get("structures") or 0)
    side = _lot_side(density)
    if want_lots > 0:
        from_ask = int(math.ceil(math.sqrt(target["min_plot_columns"]
                                           / float(want_lots))))
        side = max(side, from_ask)
    side += int(ch.get("_lot_grow") or 0)
    attached = bool(ch.get("attached"))
    open_front = ch.get("frontage") == "open"
    attached_note = None
    house_name, house = houses[0]
    if attached:
        # v2, C2: a row of party walls is built of a type that declares `ATTACHED`, at a
        # lot every configuration of its flanks admits; where the role has no such type,
        # or none admits a lot, the row is detached and the record says
        terraced = [(n, d) for n, d in houses if d.get("attached")]
        pick = None
        for n, d in terraced:
            got = _attached_lot(d, side, ("west", "east") if fr.u_is_x
                                else ("north", "south"))
            if got:
                pick = (n, d, got)
                break
        if pick:
            house_name, house, (w, ld) = pick
            if ch.get("lot_depth"):
                d2 = int(ch["lot_depth"])
                fl = fr.flanks()
                if all(_admits(house, w, d2, att) for att in (fl, fl[:1], ())):
                    ld = d2
        else:
            attached = False
            attached_note = ("no committed type of this role declares ATTACHED at a "
                             "lot it admits; the row is detached")
    if not attached:
        w = _clamp_side(side, house)
        ld = _clamp_side(int(ch.get("lot_depth") or side), house)
    gap = 0 if attached else (PLOT_LANE if open_front else LOT_GAP)
    row_gap = PLOT_LANE if open_front else LOT_GAP
    # **...or what the cover needs at the lot this fabric admits, whichever is more.**
    # The ask is `area x plot_share / columns_per_plot` at the density's *own* lot, and
    # a fabric whose lot is smaller than that covers less ground with the same number of
    # houses: a row of party walls is six by eight where a dense lot is ten square, so
    # the district was asked for eight houses and eight of them covered fourteen per
    # cent of ground its density asks thirty-three of. The count is never under the ask;
    # where the lot is smaller, there are more of them.
    if want_lots > 0 and w * ld:
        want_lots = max(want_lots,
                        int(math.ceil(target["min_plot_columns"] / float(w * ld))))
    others = [(n, d) for n, d in houses if n != house_name and _admits(d, w, ld)
              and not attached]
    per_block = int(ch.get("_block_lots") or BLOCK_LOTS.get(density, 3))
    block = int(ch.get("block") or (per_block * w + (per_block - 1) * gap))
    block = max(block, w)
    bd = 2 * ld + row_gap
    street = PLOT_LANE

    # --- what the ground already holds: the arterial and the standing parts ----
    band = np.zeros((fr.U, fr.V), dtype=bool)
    for c in ((place.get("arterials") or {}).get("cells") or []):
        x, z = int(c[0]), int(c[1])
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                xx, zz = x + dx, z + dz
                if fr.x0 <= xx <= fr.x1 and fr.z0 <= zz <= fr.z1:
                    u, v = ((xx - fr.x0, zz - fr.z0) if fr.u_is_x
                            else (zz - fr.z0, xx - fr.x0))
                    band[u, v] = True
    taken = np.zeros((fr.U, fr.V), dtype=bool)
    from .placeplan import types_card
    _t, every = types_card()
    for p in (place.get("parts") or []):
        d = every.get(p.get("type")) or {}
        m = int((d.get("needs") or {}).get("clearance", 2)) + 1
        for (rx0, rz0, rx1, rz1) in pipeline.part_rects(p):
            for xx in range(max(fr.x0, rx0 - m), min(fr.x1, rx1 + m) + 1):
                for zz in range(max(fr.z0, rz0 - m), min(fr.z1, rz1 + m) + 1):
                    u, v = ((xx - fr.x0, zz - fr.z0) if fr.u_is_x
                            else (zz - fr.z0, xx - fr.x0))
                    taken[u, v] = True

    def free(u0, u1, v0, v1) -> bool:
        return not band[u0:u1 + 1, v0:v1 + 1].any() \
            and not taken[u0:u1 + 1, v0:v1 + 1].any()

    def mark(u0, u1, v0, v1, what):
        ledger[u0:u1 + 1, v0:v1 + 1] = L[what]

    # --- the grid: the arterial's band is a street of it where it is axial --------
    lead = bool(ch.get("_lead", True))
    # a row of party walls is lots of one width, so its blocks are not widened to a
    # remainder: what is left at the end of a run is a verge
    cols, axial_u = _grid_axis(band, 0, fr.U, street, block, w, lead=lead,
                               widen=not attached)
    rows, axial_v = _grid_axis(band, 1, fr.V, street, bd, ld, lead=lead)
    blocks = [(i, j) for j in range(len(rows)) for i in range(len(cols))]
    n_blocks = len(blocks)
    # the free block nearest the district's middle takes the landmark
    cu, cv = (fr.U - 1) / 2.0, (fr.V - 1) / 2.0
    order = sorted(blocks, key=lambda ij: (abs((cols[ij[0]][0] + cols[ij[0]][1]) / 2.0 - cu)
                                           + abs((rows[ij[1]][0] + rows[ij[1]][1]) / 2.0 - cv),
                                           ij))
    landmarks = list(ch.get("landmarks") or [])
    kind_of = {}
    if landmarks:
        for ij in order:
            if free(cols[ij[0]][0], cols[ij[0]][1], rows[ij[1]][0], rows[ij[1]][1]):
                kind_of[ij] = "landmark"
                break
    rest = [b for b in blocks if b not in kind_of]
    rest.sort(key=lambda ij: _seeded(seed, "block", ij[0], ij[1]))
    n_open = int(round(float(ch.get("open_share") or 0.0) * n_blocks))
    n_court = int(round(float(ch.get("courtyard_share") or 0.0) * n_blocks))
    for b in rest[:n_open]:
        kind_of[b] = "open"
    # a courtyard block is a block of two rows whose back row is the court: only a block
    # deep enough for two rows can be one
    deep = [b for b in rest[n_open:]
            if rows[b[1]][1] - rows[b[1]][0] + 1 >= 2 * ld + row_gap]
    for b in deep[:n_court]:
        kind_of[b] = "courtyard"
    for b in rest[n_open:]:
        kind_of.setdefault(b, "row")

    # streets: everything not in a block, and the arterial's band
    ledger[:, :] = L["street"]
    for (i, j) in blocks:
        u0, u1 = cols[i]
        v0, v1 = rows[j]
        mark(u0, u1, v0, v1, "undeveloped")

    # --- the leaves ----------------------------------------------------------
    quarters: dict = {}
    dropped = {"arterial": 0, "standing": 0}
    counts = {"lots": 0, "courts": 0, "open": 0, "verges": 0, "landmarks": 0,
              "party_walls": 0}
    n_leaf = [0]

    def leaf(kind, name, tname, decl, u0, u1, v0, v1, front=None, notes=""):
        x0, z0, x1, z1 = fr.rect(u0, u1, v0, v1)
        n_leaf[0] += 1
        k = n_leaf[0]
        rng = random.Random(f"{seed}/{district['name']}/{k}")
        params = {}
        for pn, spec_p in (decl.get("params") or {}).items():
            if not isinstance(spec_p, (list, tuple)) or not spec_p:
                continue
            if spec_p[0] == "int" and len(spec_p) >= 3:
                params[pn] = rng.randint(int(spec_p[1]), int(spec_p[2]))
            elif spec_p[0] == "choice" and len(spec_p) >= 2 and spec_p[1]:
                params[pn] = rng.choice(list(spec_p[1]))
        row = {"kind": kind, "name": name, "type": tname, "seed": 1 + (k % 89),
               "params": params, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
               "notes": notes}
        if front:
            row["front"] = front
        return row

    def lots_along(u0, u1, v0, v1, front, i, j, r, plots, depth=None):
        """A row of lots along one street face of a block, in [u0, u1] x [v0, v1]."""
        if counts["lots"] >= want_lots:
            return      # the district has the houses it was asked for, or asked none
        dd = ld if depth is None else depth
        n = u1 - u0 + 1
        k = (n + gap) // (w + gap)
        if k < 1:
            return
        lo, hi, _ex = _plot_range(house)
        widths = [w] * k
        left = n - (k * w + (k - 1) * gap)
        if 0 < left <= WIDEN_UP_TO * n and not attached:
            # widen the lots to take the remainder up, inside the house's envelope
            per = left // k
            for q in range(k):
                widths[q] = min(hi, w + per)
            for q in range(left - per * k):
                widths[q] = min(hi, widths[q] + 1)
            widths = [ww if _admits(house, ww, dd) else w for ww in widths]
        at = u0
        row = []
        for q, ww in enumerate(widths):
            ua, ub = at, at + ww - 1
            at += ww + gap
            if not free(ua, ub, v0, v1):
                dropped["arterial" if band[ua:ub + 1, v0:v1 + 1].any()
                        else "standing"] += 1
                continue
            tname, decl = house_name, house
            if others and (counts["lots"] % OTHER_EVERY) == OTHER_EVERY - 1:
                tname, decl = others[int(_seeded(seed, "other", i, j, r, q)
                                         * len(others)) % len(others)]
                if not _admits(decl, ww, dd):
                    tname, decl = house_name, house
            if counts["lots"] >= want_lots:
                break
            row.append((ua, ub, leaf("plot", f"b{i}_{j}_{r}{q}", tname, decl, ua, ub,
                                     v0, v1, front=None if open_front else front,
                                     notes=f"a {tname} fronting the street on its "
                                           f"{front} side")))
            plots.append(row[-1][2])
            mark(ua, ub, v0, v1, "lot")
            counts["lots"] += 1
        if attached:
            # v2, C2: the sides a lot's neighbour actually stands against, after the
            # drops -- a party wall to each; the pad reaches the plot's edge there
            low, high = fr.flanks()
            for q, (ua, ub, p) in enumerate(row):
                sides = []
                if q > 0 and row[q - 1][1] + 1 == ua:
                    sides.append(low)
                if q + 1 < len(row) and row[q + 1][0] == ub + 1:
                    sides.append(high)
                p["attached"] = sides
                counts["party_walls"] += len(sides)

    def areas_over(u0, u1, v0, v1, tname, what, plots, note, prefix):
        decl = areas.get(tname)
        if decl is None or u1 < u0 or v1 < v0:
            return 0
        n = 0
        for (ua, ub, va, vb) in _area_tiles(u0, u1, v0, v1, decl, AREA_GAP):
            if not free(ua, ub, va, vb):
                dropped["arterial" if band[ua:ub + 1, va:vb + 1].any()
                        else "standing"] += 1
                continue
            plots.append(leaf("area", f"{prefix}_{n}", tname, decl, ua, ub, va, vb,
                              notes=note))
            mark(ua, ub, va, vb, what)
            n += 1
        return n

    for (i, j) in blocks:
        u0, u1 = cols[i]
        v0, v1 = rows[j]
        kind = kind_of[(i, j)]
        q = quarters.setdefault(f"row_{j}", {"name": f"row_{j}", "notes":
                                             f"the blocks between the {j + 1}th street "
                                             f"and the next", "plots": []})
        plots = q["plots"]
        two_rows = (v1 - v0 + 1) >= 2 * ld + row_gap
        if kind == "open":
            t = _open_type(areas, role, density)
            what = t if t in LEDGER else "garden"
            counts["open"] += areas_over(u0, u1, v0, v1, t, what, plots,
                                         f"open ground: a {t} of the block", f"o{i}_{j}")
            continue
        if kind == "landmark":
            lm = landmarks[0]
            tname = lm.get("type")
            decl = decls.get(tname)
            if decl and decl.get("kind", "plot") == "plot":
                lo, hi, _ex = _plot_range(decl)
                s = min(hi, u1 - u0 + 1, v1 - v0 + 1, LANDMARK_SIDES * side)
                s = _clamp_side(s, decl)
                if s <= min(u1 - u0 + 1, v1 - v0 + 1) and lo <= s:
                    ua = u0 + ((u1 - u0 + 1) - s) // 2
                    if free(ua, ua + s - 1, v0, v0 + s - 1):
                        plots.append(leaf("plot", f"landmark_{tname}", tname, decl,
                                          ua, ua + s - 1, v0, v0 + s - 1,
                                          front=fr.front(True),
                                          notes=lm.get("notes") or
                                          f"the {tname} the character names, on the "
                                          f"block nearest the middle"))
                        mark(ua, ua + s - 1, v0, v0 + s - 1, "lot")
                        counts["landmarks"] += 1
                        pt = "plaza" if "plaza" in areas else _court_type(areas, role)
                        # the plaza before and beside it, where a tile fits
                        areas_over(u0, ua - LOT_GAP - 1, v0, v1, pt, "plaza", plots,
                                   f"the plaza beside the {tname}", f"p{i}_{j}w")
                        areas_over(ua + s + LOT_GAP, u1, v0, v1, pt, "plaza", plots,
                                   f"the plaza beside the {tname}", f"p{i}_{j}e")
                        areas_over(ua, ua + s - 1, v0 + s + LOT_GAP, v1, pt, "plaza",
                                   plots, f"the plaza behind the {tname}", f"p{i}_{j}s")
                        continue
            kind = "row"
        # a row block or a courtyard block: the front row fronts the street before it
        if not two_rows:
            # **One row takes the whole block.** A block too shallow for two rows of
            # lots back to back used to lay one row of the density's own depth and leave
            # the rest of its depth to the fill -- a strip beside a place's centre came
            # back three houses on a tenth of its ground, under the plot cover its own
            # density asks. A shallow block is one row of deep lots.
            deep = _deepest(house, w, v1 - v0 + 1,
                            fr.flanks() if attached else ())
            if deep:
                lots_along(u0, u1, v0, v0 + deep - 1, fr.front(True), i, j, 0, plots,
                           depth=deep)
            continue
        lots_along(u0, u1, v0, v0 + ld - 1, fr.front(True), i, j, 0, plots)
        if kind == "courtyard":
            t = _court_type(areas, role)
            counts["courts"] += areas_over(u0, u1, v0 + ld + row_gap, v1, t, "court",
                                           plots, f"the court the row shares",
                                           f"c{i}_{j}")
        else:
            lots_along(u0, u1, v1 - ld + 1, v1, fr.front(False), i, j, 1, plots)

    # --- the leftover, assigned: every free rectangle of a block, and the strips the
    # grid leaves at the district's edges, are verges of the role's ground type --
    # largest first, a clearance from everything laid, until none three across is left.
    # What is left after that is undeveloped, and the record says how much.
    verge_t = _verge_type(areas, role)
    if verge_t and areas.get(verge_t):
        vq = quarters.setdefault("verges", {"name": "verges", "notes":
                                            "the ground the blocks and the streets "
                                            "leave, assigned", "plots": []})
        # **Each pair's own clearance.** A lot keeps `LOT_GAP`; one piece of open ground
        # keeps `AREA_GAP` from the next, which is what the validator holds two areas
        # to. Growing everything by a lot's clearance left a fifth of a district
        # assigned to nothing and under the cover its own density asks for.
        lots_laid = (ledger == L["lot"])
        areas_laid = ((ledger != L["street"]) & (ledger != L["undeveloped"])
                      & ~lots_laid)

        def _grow(mask, k):
            out = mask.copy()
            for du in range(-k, k + 1):
                for dv in range(-k, k + 1):
                    if du == 0 and dv == 0:
                        continue
                    src = mask[max(0, -du):fr.U - max(0, du),
                               max(0, -dv):fr.V - max(0, dv)]
                    out[max(0, du):fr.U - max(0, -du),
                        max(0, dv):fr.V - max(0, -dv)] |= src
            return out

        grown = _grow(lots_laid | band | taken, LOT_GAP) | _grow(areas_laid, AREA_GAP)
        # ...and a lane's width kept clear along the streets of the grid, so the verges
        # do not close them: the street rows and columns stay free ground
        street_mask = np.ones((fr.U, fr.V), dtype=bool)
        for (i, j) in blocks:
            street_mask[cols[i][0]:cols[i][1] + 1, rows[j][0]:rows[j][1] + 1] = False
        # the edge strips beyond the last block are verge ground, not street
        if rows:
            street_mask[:, rows[-1][1] + street + 1:] = False
        if cols:
            street_mask[cols[-1][1] + street + 1:, :] = False
        room = ~grown & ~street_mask
        n_v = 0
        for _k in range(400):
            r = _largest_free(room)
            if r is None:
                break
            u0, u1, v0, v1 = (int(v) for v in r)
            if (u1 - u0 + 1) < 3 or (v1 - v0 + 1) < 3:
                break
            n_here = areas_over(u0, u1, v0, v1, verge_t,
                                verge_t if verge_t in LEDGER else "verge", vq["plots"],
                                f"a {verge_t} on ground the blocks left", f"v{n_v}")
            n_v += n_here
            # whatever was laid, and its clearance from the next piece of open ground,
            # is out of the room now
            room[max(0, u0 - AREA_GAP):u1 + AREA_GAP + 1,
                 max(0, v0 - AREA_GAP):v1 + AREA_GAP + 1] = False
        counts["verges"] = n_v
        if not vq["plots"]:
            del quarters["verges"]
    # the arterial's band is a street whatever block it crossed
    ledger[band] = L["street"]

    # --- the record ------------------------------------------------------------
    ch = {k: v for k, v in ch.items() if not k.startswith("_")}
    got = {"notes": (f"Compiled from the character of {part.get('name') or district['name']}: "
                     f"a {density} {role or ''} district, {'attached' if attached else 'detached'} "
                     f"lots of {w}x{ld} on blocks of {block}, streets {street} wide, "
                     f"{n_open} open block(s), {n_court} courtyard block(s)"
                     + (f", a {landmarks[0].get('type')} landmark" if landmarks else "")
                     + f"; {counts['lots']} lots"),
           "quarters": [q for q in quarters.values() if q["plots"]],
           "character": {k: v for k, v in ch.items()}}
    leaves = [p for q in got["quarters"] for p in q["plots"]]
    total = W_all * D_all
    assigned = {name: int((ledger == L[name]).sum()) for name in LEDGER}
    assigned["street"] += total - fr.U * fr.V        # the margin: a lane's worth
    plot_cols = sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1)
                    for p in leaves if p["kind"] == "plot")
    ground_cols = sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1) for p in leaves)
    record = {"district": district["name"], "part": part.get("name"), "seed": seed,
              "asked_for": want_lots,
              "character": ch, "house": house_name, "attached": attached,
              "attached_note": attached_note,
              "others": [n for n, _d in others], "lot": [w, ld], "gap": gap,
              "block": block, "block_depth": bd, "street": street,
              "columns": total, "blocks": n_blocks,
              "axial_arterial": [bool(axial_u), bool(axial_v)],
              "block_kinds": {k: sum(1 for b in blocks if kind_of[b] == k)
                              for k in ("row", "courtyard", "open", "landmark")},
              "grid": {"columns": len(cols), "rows": len(rows)},
              "leaves": len(leaves), **counts, "dropped": dropped,
              "assigned": assigned,
              "undeveloped_share": round(assigned["undeveloped"] / float(total), 4),
              "plot_cover": round(plot_cols / float(total), 4),
              "ground_cover": round(ground_cols / float(total), 4),
              "target": {"count": target["count"],
                         "min_count": target["min_count"],
                         "plot_share": target["plot_share"],
                         "min_plot_columns": target["min_plot_columns"],
                         "min_ground_columns": target["min_ground_columns"]},
              "registered_plot_cover": PLOT_COVER[density],
              "meets_registered_cover": plot_cols / float(total) >= PLOT_COVER[density]}
    return got, record
