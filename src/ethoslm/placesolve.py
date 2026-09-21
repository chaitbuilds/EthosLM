"""The relation solver: the place level, placed by relation. v2, B2.

The spec's relation words existed and only `concentric` was resolved; every other
place was drawn freehand by a model as rectangles, and a walled town was never
checked to stand inside its own wall. This is the place-level planner now: the
defining parts **declare relations** -- `centre`, `perimeter`, `gateway`, `edge`,
`beside_the_centre`, `throughout`, `quarter`, `near`, `along`, `on`, and `concentric`
-- and the solver places them. The model never writes a coordinate at any level.

How it solves, in order:

  * candidates are **seeded from the relation**: a centre part at the plateau cut for
    it or at the site's centre; a perimeter wall as a closed loop about the centre,
    at every half-side from the least the districts need to the site's edge; a gate at
    the middle of each side of its wall; a part beside the centre on the four sides
    of it; a part at the edge just inside the wall; a part `near` another on the four
    sides of that one; a part `along` an edge in a strip beside its longest run; a
    part `on` an edge at a cell of its line;
  * the validators are **hard vetoes** (`VETOES`, in weight order: inside the site,
    no overlap or clearance breach, the type's footprint band, inside the perimeter
    wall, a compound on its plateau) and **soft costs** (`COSTS`: the relief and the
    water under the part, the distance from the relation's anchor, the relief along
    a wall's line, a wall shrunk from the site's edge);
  * **greedy insertion in declaration order**, each part taking the least-cost
    candidate that passes every veto, then a **bounded local improvement**: one pass
    over the parts, each tried at its next-best candidates with the rest fixed, kept
    where the whole costs less;
  * when nothing is feasible for a part, its vetoes are **demoted in weight order** --
    the lightest dropped first, the site's never -- and the least-violating placement
    is recorded with the reason;
  * the districts (`throughout`, `quarter`, `along` groups) tile the ground left over
    inside the wall round the centre, as the ring layout's strips do, and the
    structures the spec declared are spread over them by area, capped by the room
    each has;
  * a place with rings is **one solved case**: `concentric_layout`'s arithmetic is the
    candidate and the validator its veto.

Deterministic under the seed: candidates are ordered by cost and then by their own
coordinates, and the seed orders the ties. Every number is in `layout`.
"""
from __future__ import annotations

import math
import random
import re

from . import pipeline, placeplan, placeregion, spec as spec_mod, styles
from .placeplan import (_FACING, _SIDES, _answers, _edge_cells, _gate_at, _gate_type,
                        _largest_remainder, _octagon_path, _square_path, _top_params,
                        _wall_for, _wall_inset, _wall_types, compound_ground,
                        wall_face_for, wall_round_for, wall_stairs_for,
                        COMPOUND_MIN, COMPOUND_PLATEAU_SHARE, DISTRICT_FILL, LANE_GAP,
                        RING_COVERAGE, RING_EDGE_INSET, SECTOR_MAX)
from .placeread import inside

#: **The least a sector may be asked for to be a district at all.** A division of a
#: place that holds houses holds more than one: a sliver asked for a single house is
#: left-over ground, and holding it to a district's count and cover refuses a whole
#: place for a corner of it. One number, read from `placeplan` so the ring layout and
#: the relation solver cannot drift apart on it.
DISTRICT_MIN_STRUCTURES = placeplan.DISTRICT_MIN_STRUCTURES

#: The hard vetoes, heaviest first. When a part has no feasible candidate they are
#: dropped from the lightest up; `site` is never dropped. Registered.
VETOES = (("site", 100), ("overlap", 90), ("footprint", 80), ("inside", 70),
          ("plateau", 60))

#: The soft costs. Registered.
COSTS = {"relief": 1.0,        # per block of relief under the part
         "water": 100.0,       # times the wet share of the part's ground
         "distance": 0.5,      # per column from the relation's anchor
         "line_relief": 2.0,   # per block of mean rise between neighbouring wall columns
         "line_water": 100.0,  # times the wet share of a wall's line
         "shrink": 0.1,        # per block a perimeter wall's half-side is under the edge
         "wall_distance": 0.1,  # per column a wall's centre is from the place's
         "axis": 10.0,
         # **a side round the centre lost to a satellite**, per side (the closure
         # round): a hall placed on the square's east side took the whole east strip
         # with it, and the cottages the sentence gathers *around* the square could
         # reach two of its four sides. Costed, not vetoed: a satellite has to stand
         # somewhere, and where every side is small the least-losing one is taken.
         "strips": 60.0,
}         # a part beside the centre on the gate's side of it

#: How many candidates a part keeps for the improvement pass, and how many passes.
KEEP = 8
IMPROVE_PASSES = 1

#: How a perimeter wall's half-side is stepped between the least it can be and the
#: site's edge, and how many positions about the centre are tried at each.
WALL_STEP_BLOCKS = 8
WALL_SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

#: How far a part stands from the one it is beside or near: the larger clearance of the
#: two, and a lane.
BESIDE_GAP = LANE_GAP


class _Ground:
    """The ground the costs read: relief and water over a rectangle, off the volume."""

    def __init__(self, vol):
        self.h = self.wet = None
        self.x0 = self.z0 = 0
        if vol is None:
            return
        from . import observe
        self.h, self.wet = observe.ground_heights(vol)
        self.x0, self.z0 = int(vol.x0), int(vol.z0)

    def _win(self, rect):
        if self.h is None:
            return None
        x0, z0, x1, z1 = rect
        i0, i1 = x0 - self.x0, x1 - self.x0 + 1
        j0, j1 = z0 - self.z0, z1 - self.z0 + 1
        if i0 < 0 or j0 < 0 or i1 > self.h.shape[0] or j1 > self.h.shape[1] or i0 >= i1 \
                or j0 >= j1:
            return None
        return (slice(i0, i1), slice(j0, j1))

    def relief(self, rect) -> float:
        w = self._win(rect)
        if w is None:
            return 0.0
        win = self.h[w]
        return float(win.max() - win.min())

    def water(self, rect) -> float:
        w = self._win(rect)
        if w is None:
            return 0.0
        return float(self.wet[w].mean())

    def line(self, cells) -> tuple:
        """(mean rise between neighbouring columns, wet share) along a wall's cells."""
        if self.h is None or not cells:
            return 0.0, 0.0
        hs, wets = [], []
        for (x, z) in cells:
            i, j = x - self.x0, z - self.z0
            if 0 <= i < self.h.shape[0] and 0 <= j < self.h.shape[1]:
                hs.append(int(self.h[i, j]))
                wets.append(bool(self.wet[i, j]))
        if len(hs) < 2:
            return 0.0, 0.0
        rise = sum(abs(a - b) for a, b in zip(hs, hs[1:])) / float(len(hs) - 1)
        return rise, sum(wets) / float(len(wets))


def _clean_side(decl: dict) -> int:
    """The largest square pad this type admits, honouring `except`."""
    needs = decl.get("needs") or {}
    a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
    ex = set(needs.get("except") or ())
    clean = [v for v in range(min(a, b), max(c, e) + 1)
             if a <= v <= c and b <= v <= e and v not in ex]
    return max(clean) if clean else 0


def _type_for(family: str, kind: str, decls: dict) -> tuple | None:
    """(name, decl) of the committed type a defining part of this family is built as:
    one named for the family, else the largest of its kind."""
    named = sorted(n for n in decls if n == family or n.startswith(family + "_"))
    for n in named:
        if decls[n] is not None and decls[n].get("kind", "plot") == kind:
            return n, decls[n]
    if named and decls[named[0]] is not None:
        return named[0], decls[named[0]]
    pool = [(n, d) for n, d in decls.items()
            if d is not None and d.get("kind", "plot") == kind and not d.get("passage")]
    if not pool:
        return None
    return sorted(pool, key=lambda nd: (-_clean_side(nd[1]), nd[0]))[0]


def _rect_for(decl: dict, kind: str, cx: int, cz: int, cap: int | None = None) -> tuple:
    """A rectangle of this type's largest size, centred on (cx, cz)."""
    from .buildlib import Builder
    side = _clean_side(decl)
    i = 0 if kind in ("edge", "point", "area") else 2 * Builder.SITE_INSET
    w = side + i
    if cap is not None:
        w = min(w, int(cap))
    w = max(w, 3)
    x0, z0 = cx - w // 2, cz - w // 2
    return (x0, z0, x0 + w - 1, z0 + w - 1)


def _clearance(decls: dict, tname) -> int:
    return int(((decls.get(tname) or {}).get("needs")
                or pipeline.NEEDS_DEFAULT)["clearance"])


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _grow(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def _leaf_rects(leaf: dict) -> list:
    return pipeline.part_rects({**leaf, "name": leaf.get("name")})


# ------------------------------------------------------------ allocation from purpose
# The expression round. The review's first remaining cause: `_rect_for` chose the
# largest rectangle a type admits for the thing at the centre and for every civic part
# beside it, and the strips round them were cut from the site's extent -- so a market
# square for sixteen cottages was forty columns a side because `square` admits forty,
# the hall beside it thirty-six because `hall` admits thirty-two, and the fields a
# farming village works were whatever strip was left. Nothing about the *purpose* of the
# place -- how many households, at what density, of what envelope, working what land --
# reached the dimension. What follows is the one sizing rule the relation solver and the
# ring layout both read: a parent allocation is derived from the programme, and the
# derivation is recorded as a `target` with its `from` list, so a reader can see why a
# square is the size it is and a built finding can move the input it cites.

#: **How much open anchor one counted household brings to the thing the place is
#: gathered about**, as a share of one fabric lot. Registered, with its derivation: a
#: village of sixteen cottages on lots of 12x10 brings 16 x 0.5 x 120 = 960 columns of
#: square, about 31 a side, which is a market square two cottage-lengths across -- the
#: scale of a small market place, and eight times a cottage's footprint rather than
#: thirteen. A dense town's plaza takes the same share of a smaller lot. A built finding
#: about the square's scale moves this share down toward `ANCHOR_SHARE_MIN` through
#: `reallocate`, and the programme overrides it through an `allocation` row.
ANCHOR_SHARE = {"square": 0.5, "plaza": 0.5, "market": 0.5, "green": 0.6,
                "court": 0.35, "garden": 0.5, "yard": 0.35}
ANCHOR_SHARE_DEFAULT = 0.5
ANCHOR_SHARE_MIN = 0.2
ANCHOR_SHARE_STEP = 0.7          # one bounded `shrink_anchor` multiplies the share by this

#: **How big a civic building beside the centre is, in fabric lots**, by the family the
#: spec names it as. Registered: a hall for a village is two cottages' ground and not
#: the thirty-two-square the type admits; a temple or a church three; a keep four; a
#: workshop the one lot of the house it works beside. Clamped into what the type admits
#: (`district_compile._clamp_side`), so a family whose type cannot be built that small
#: is built at the type's least.
CIVIC_LOTS = {"hall": 2.0, "moot": 2.0, "temple": 3.0, "church": 3.0, "chapel": 2.0,
              "shrine": 1.0, "keep": 4.0, "tower": 1.0, "workshop": 1.0, "smithy": 1.0,
              "forge": 1.0, "inn": 1.5, "tavern": 1.5, "mill": 1.5, "barn": 1.5,
              "market": 2.0, "school": 2.0, "manor": 4.0}
CIVIC_LOTS_DEFAULT = 1.5

#: **How much working land one household of the place needs**, in columns of the land
#: itself, by land use. Registered: a farming village's fields are 240 columns a
#: household -- two and a half field tiles of 10x10 -- which over `LAND_FILL` of a
#: district's ground (lanes and verges take the rest) is 400 columns of district a
#: household; sixteen cottages want 6,400 columns of fields, an eighty-square, and not
#: the belt of three tiles the closure round's judge called thin. An orchard is half a
#: field; pasture a third again more. A place's own explicit count is the number of
#: households, wherever its houses stand.
LAND_PER_HOUSE = {"farmland": 240, "orchard": 120, "pasture": 320, "garden": 60,
                  "industrial": 120, "civic": 40}
LAND_FILL = 0.6                  # `placeplan.RURAL_COVER`: the land's share of its district
LAND_STEP = 1.5                  # one bounded `grow_land` multiplies the need by this
#: **How many of a place's households live among its working land**, at the most, as a
#: share of the count: "the few farm cottages that work them". Registered: about one in
#: six, so a village of sixteen keeps three among its fields and the rest about its
#: square. A land district's houses were split by its ground before this, and a wide
#: fields quarter took five of sixteen.
LAND_HOUSE_SHARE = 0.16

#: The land-use families an `area` defining part is, and the house words a plot defining
#: part is counted by. The entity rule below reads these and the spec's own
#: declarations; it reads no place's name.
_LAND_FAMILIES = frozenset(("field", "fields", "farmland", "orchard", "pasture", "grove",
                            "garden", "paddy", "meadow", "wood", "woods", "forest"))
_AMENITY_FAMILIES = frozenset(("square", "plaza", "market", "green", "court", "yard"))
_HOUSE_FAMILIES = frozenset(("house", "houses", "home", "homes", "cottage", "cottages",
                             "dwelling", "dwellings", "hut", "huts", "farmhouse",
                             "townhouse", "villa", "hovel"))


def entity_of(part: dict, spec: dict | None = None) -> dict:
    """**What kind of thing a defining part is, for the count's sake.**

        `{"class": building|land|amenity|compound|boundary|point, "counted": bool,
        "unit": the count's subject class or None, "land_use": ...|None}`.

        The one rule: **the explicit count is spent only on parts whose class is
        `building` (a district of houses, or a house on a plot) with the count's unit**; land,
        amenities, compounds and boundaries never take a counted lot. A land district takes
        counted buildings only where the programme says buildings stand in it -- a character
        that declares a lot or a storey band, or a structure count the spec declared rather
        than inferred -- so a farm quarter "with the few cottages that work it" holds its
        cottages and an orchard beside the houses holds none.

        Prefers `spec.entity_of` where the meaning module states it (the coordinator's
        binding record); this is the same rule as the layout's own fallback.
        
    """
    fn = getattr(spec_mod, "entity_of", None)
    if fn is not None:
        try:
            names = getattr(getattr(fn, "__code__", None), "co_varnames", ())
            if "count" in names:
                got = fn(part, (spec or {}).get("explicit_count"))
            elif "spec" in names:
                got = fn(part, spec)
            else:
                got = fn(part)
            if isinstance(got, dict) and got.get("class"):
                return got
        except Exception:                    # noqa: BLE001 -- the fallback answers
            pass
    unit = None
    if spec and (spec.get("explicit_count") or {}).get("what"):
        unit = str(spec["explicit_count"]["what"])
    kind = str(part.get("kind") or "plot")
    fam = str(part.get("family") or "").lower()
    if spec_mod.compound(part):
        return {"class": "compound", "counted": False, "unit": None, "land_use": None}
    if kind == "group":
        use = spec_mod.land_use(part)
        if use != "settled":
            ch = part.get("character") or {}
            holds = bool(ch.get("lot_width") or ch.get("lot_depth") or ch.get("storeys")
                         or (int(part.get("structures") or 0)
                             and not part.get("structures_inferred")))
            return {"class": "land", "counted": holds, "unit": unit if holds else None,
                    "land_use": use}
        return {"class": "building", "counted": True, "unit": unit, "land_use": "settled"}
    if kind == "area":
        if fam in _LAND_FAMILIES:
            use = spec_mod.land_use(part)
            return {"class": "land", "counted": False, "unit": None,
                    "land_use": use if use != "settled" else fam}
        return {"class": "amenity", "counted": False, "unit": None, "land_use": None}
    if kind in ("edge", "point"):
        return {"class": "boundary" if kind == "edge" else "point", "counted": False,
                "unit": None, "land_use": None}
    role = str(part.get("role") or "")
    counted = (fam in _HOUSE_FAMILIES or str(part.get("function") or "") == "dwelling") \
        and role not in ("civic", "defensive")
    return {"class": "building" if counted else "amenity", "counted": bool(counted),
            "unit": unit if counted else None, "land_use": None}


def allocation_of(spec: dict | None, allocation: dict | None = None) -> dict:
    """The allocation overrides in force: every `allocation` row of the spec's
    `negotiated` list, in order, under the caller's own. `{}` where there are none.

    A reallocation is a revision of an inferred choice and is recorded where the scale
    owner records its own (`spec.negotiated`), which `spec.read_spec` preserves and the
    plan's fingerprint reads -- so a place laid out again from the spec is laid out
    with the same allocation, and nothing has to remember it."""
    out: dict = {}
    for row in (spec or {}).get("negotiated") or []:
        if isinstance(row, dict) and row.get("what") == "allocation":
            for k, v in (row.get("to") or {}).items():
                if isinstance(v, dict):
                    out.setdefault(k, {}).update(v)
                else:
                    out[k] = v
    for k, v in (allocation or {}).items():
        if isinstance(v, dict):
            out.setdefault(k, {}).update(v)
        else:
            out[k] = v
    return out


#: **Where a run's probed envelopes are kept**: `out/<round>/envelopes.json`, set for
#: the duration of a solve by `solve_place(envelope_cache=)` and written on the place as
#: `layout.envelope_cache`, which the compiler reads. A module global because the sizing
#: rule is called from module functions with no round in reach; one solve at a time.
ENVELOPE_CACHE: str | None = None


def _demand_module():
    """`ethoslm.demand` where it is on disk, else None. Imported defensively because the
    module lands separately; a caller with a **required** feature to carry refuses on
    None rather than asking a featureless question."""
    try:
        from . import demand as _demand
    except Exception:                        # noqa: BLE001 -- not landed yet
        return None
    return _demand


def _lot_min_for(tname: str | None, params: dict | None, features=(), *,
                 voice: str | None = None, seed: int = 1,
                 cache: str | None = None) -> tuple | None:
    """The least lot this type delivers these parameters **with these features** on,
        from the envelope module where it exists (worker B's `envelope.lot_for`), else None.

        **The features are not optional.** The expression review's first cause, measured on
        the imported code: `lot_for("cottage", {"storeys": 2})` answers a 5x5 minimum and
        `lot_for("cottage", {"storeys": 2}, features=("storeys",))` answers 15x17, because
        with an empty feature list `_stands` checks no requested feature and a standing
        shell answers a multi-storey query. Every caller here names what it is asking for.
        
    """
    if not tname:
        return None
    try:
        from . import envelope as _env
    except Exception:                        # noqa: BLE001 -- not landed yet
        return None
    fn = getattr(_env, "lot_for", None)
    if fn is None:
        return None
    try:
        got = fn(tname, dict(params or {}), features=tuple(features or ()), voice=voice,
                 seed=int(seed), cache=cache or ENVELOPE_CACHE)
    except Exception:                        # noqa: BLE001 -- an envelope that cannot answer
        return None
    lm = (got or {}).get("lot_min")
    return (int(lm[0]), int(lm[1])) if lm and len(lm) == 2 else None


def lots_from_constraints(constraints, group: str | None = None) -> dict:
    """`{type: [w, d]}` -- the largest `needs.lot_min` construction recorded per type
    among the emitted constraints the layout owns. What a second allocation enlarges
    lots to, so a storey the ground refused is not asked of the same lot twice."""
    out: dict = {}
    for c in constraints or []:
        if not isinstance(c, dict) or c.get("owner") != "layout":
            continue
        lm = (c.get("needs") or {}).get("lot_min")
        t = str(c.get("type") or "")
        if not lm or not t:
            continue
        if group and not str(c.get("part") or "").startswith(str(group)):
            continue
        w, d = int(lm[0]), int(lm[1])
        was = out.get(t)
        out[t] = [max(w, was[0]), max(d, was[1])] if was else [w, d]
    return out


class Refused(Exception):
    """**A demand nothing in the approved band delivers.**

        Raised where an envelope refuses, or where a required feature cannot be carried to
        the envelope at all. The design round's first contract: "a failed or unknown
        envelope becoming a smaller default" is the defect, so this is an exception and not
        a return value -- a caller has to decide between an honest stop and an alternative
        arrangement, and cannot accidentally carry on with a smaller lot.
        
    """

    def __init__(self, why: str, *, part=None, demand=None, lot=None):
        super().__init__(why)
        self.why, self.part, self.demand, self.lot = why, part, demand, lot


def fabric_lot(part: dict, decls: dict, spec: dict | None = None, *,
               allocation: dict | None = None, arrangement: dict | None = None,
               demand: dict | None = None, voice: str | None = None) -> tuple:
    """The lot one house of this district part stands on, and where it came from:
        `((w, d), [from])`. The fabric's own arithmetic at the part's density and character
        (`placeplan.fabric`), clamped into the house type the record approves, and never
        under the lot an `allocation.lots` override, the negotiated `arrangement`, or the
        **resolved demand** asks for.

        `demand` is the demand this part resolved before anything was sized
        (`demand.resolve`): the approved type pool, the required parameters and features,
        and the requirement ids that made them required. Where one is given, the lot comes
        from `demand.lot` -- with the required features in the envelope query -- and a
        refusal is raised (`Refused`), never quietly answered with a smaller lot.
        
    """
    from . import district_compile as dc
    ch = {**(part.get("character") or {}),
          **{k: v for k, v in (arrangement or {}).items() if v is not None}}
    density = part.get("density") or "medium"
    role = part.get("role") or placeplan.DENSITY_ROLE.get(density)
    f = placeplan.fabric(density, role,
                         {k: v for k, v in ch.items()
                          if k in ("lot_width", "lot_depth", "attached", "frontage")} or None)
    w, ld = int(f["lot"][0]), int(f["lot"][1])
    src = [f"fabric `{density}`/{role}: lot {w}x{ld}"
           + (" declared by the character" if (part.get("character") or {}).get("lot_width")
              else " chosen by the negotiated arrangement" if (arrangement or {}).get("lot_width")
              else "")]
    house = None
    pool = list((demand or {}).get("types") or part.get("fabric_types") or [])
    for n2, d in dc.house_types(decls, role, (spec or {}).get("form"),
                                approved=pool or None):
        house = (n2, d)
        break
    if house is not None and not ch.get("lot_width"):
        w = dc._clamp_side(w, house[1])
        ld = dc._clamp_side(ld, house[1])
    lots = (allocation or {}).get("lots") or {}
    over = lots.get(part.get("name")) or (lots.get(house[0]) if house else None)
    if over and len(over) == 2:
        w2, d2 = max(w, int(over[0])), max(ld, int(over[1]))
        if house is not None:
            # a minimum clamps up to the next side the type admits, never down
            w2, d2 = dc._side_at_least(house[1], w2), dc._side_at_least(house[1], d2)
        if (w2, d2) != (w, ld):
            src.append(f"allocation.lots {part.get('name')}: at least {over[0]}x{over[1]}")
            w, ld = w2, d2
    lm, why = _demand_lot(part, house, ch, demand, voice=voice)
    if why:
        src.append(why)
    if lm and house is not None and (lm[0] > w or lm[1] > ld):
        w = dc._side_at_least(house[1], max(w, lm[0]))
        ld = dc._side_at_least(house[1], max(ld, lm[1]))
    return (int(w), int(ld)), src


def _demand_lot(part: dict, house, ch: dict, demand: dict | None, *,
                voice: str | None = None) -> tuple:
    """`(lot_min, why)` for the resolved demand of this district's fabric.

        Three cases, and none of them is a smaller default:

          * a resolved demand and `ethoslm.demand` on disk -- the one path; a refusal raises;
          * a resolved demand with **required** features and no module -- refused, because
            asking the envelope without the features it requires proves nothing;
          * no resolved demand -- the pre-contract path, which now at least names what it
            is asking for (`features=("storeys",)`), so an empty feature set cannot answer
            a storey question.
        
    """
    if demand is not None:
        mod = _demand_module()
        if mod is None or not hasattr(mod, "lot"):
            if demand.get("required"):
                raise Refused(
                    f"{part.get('name')}: the request requires "
                    f"{', '.join(demand.get('required') or ())} of this fabric "
                    f"(requirement {', '.join(str(i) for i in demand.get('requirements') or ()) or '?'}) "
                    f"and `ethoslm.demand` is not available to say what lot delivers it",
                    part=part.get("name"), demand=demand)
            return None, None
        got = mod.lot(demand, cache=ENVELOPE_CACHE) or {}
        if got.get("refused"):
            # **`lot_min: None` is the answer and not a hint.** Where a *required* token
            # is what failed (`binding`), nothing smaller answers it and the caller gets
            # a refusal; where the ask was an inference, the refusal is on the record
            # and the fabric's own lot stands.
            if got.get("binding", True):
                raise Refused(
                    f"{part.get('name')}: no approved type delivers "
                    f"{', '.join(demand.get('required') or ()) or 'the resolved demand'}"
                    f" -- {got['refused']}",
                    part=part.get("name"), demand=demand)
            return None, (f"demand `{part.get('name')}`: refused and not binding -- "
                          f"{got['refused']}; the fabric's own lot stands")
        declared = (part.get("character") or {}).get("lot_width")
        if declared and hasattr(mod, "validate_lot"):
            want = [int(declared), int(ch.get("lot_depth") or declared)]
            ok = mod.validate_lot(demand, want, cache=ENVELOPE_CACHE) or {}
            if not ok.get("ok") and ok.get("binding", True):
                raise Refused(
                    f"{part.get('name')}: the character declares a lot "
                    f"{want[0]}x{want[1]} and it cannot hold what the request requires "
                    f"-- {ok.get('why')}",
                    part=part.get("name"), demand=demand, lot=want)
        lm = got.get("lot_min")
        if not lm or len(lm) != 2:
            return None, (f"demand {got.get('key') or ''}: no least lot "
                          f"({got.get('why') or 'the envelope did not answer'})")
        return ([int(lm[0]), int(lm[1])],
                f"demand `{part.get('name')}` on `{got.get('type')}` "
                f"({', '.join(got.get('asked') or demand.get('required') or ())
                   or 'no required feature'}): lot at least "
                f"{int(lm[0])}x{int(lm[1])} ({got.get('source')})")
    st = ch.get("storeys")
    if house is None or not st or ch.get("lot_width"):
        return None, None
    lm = _lot_min_for(house[0], {"storeys": int(st[0])}, features=("storeys",),
                      voice=voice)
    if not lm:
        return None, None
    return (list(lm), f"envelope {house[0]} storeys {st[0]} with the storeys feature "
                      f"asked for: lot at least {lm[0]}x{lm[1]}")


def lot_raised(src: list) -> bool:
    """Did an allocation, a resolved demand or an envelope raise this lot above the
    fabric's own? The one test, so the layout and the compiler cannot disagree on
    whether a district's `lot_min` is a floor construction has to honour."""
    return any(x.startswith(("allocation.lots", "envelope ", "demand "))
               for x in src or [])


#: The share of a district's rectangle the compiler can develop once the arterial's band
#: and the standing parts' clearances are taken. Registered from the expression city's
#: middle ring (5776 of 7560 columns); a rectangle sized for its houses over its whole
#: area lays them at the ceiling of their word.
LAND_DEVELOPABLE = 0.78


def land_need(part: dict, n: int, decls: dict, spec: dict | None = None, *,
              households: int | None = None, allocation: dict | None = None,
              arrangement: dict | None = None, demand: dict | None = None) -> dict:
    """**The ground a district part needs, from its purpose.** The one sizing rule.

        For a district of houses: `n` houses at the fabric's lot over the middle of the
        density word's band -- and never less than whole blocks and their lanes, because the
        compiler lays `BLOCK_LOTS` lots to a block behind a street. For a district of land
        (`land_use` not `settled`): the place's `households` times `LAND_PER_HOUSE` over
        `LAND_FILL`, plus whatever houses stand in it. Returns the columns, the cover, the lot,
        the least length and depth of a rectangle that lays it, and `from` -- every input by
        name, which is what the resolution's `target` records.
        
    """
    from . import district_compile as dc
    band = placeplan.density_target(part.get("density"), part.get("role"))
    lo, hi = float(band.get("lo") or 0.0), band.get("hi")
    mid = (lo + float(hi)) / 2.0 if hi is not None else max(lo, 0.35)
    (w, ld), src = fabric_lot(part, decls, spec, allocation=allocation,
                              arrangement=arrangement, demand=demand)
    ch = {**(part.get("character") or {}),
          **{k: v for k, v in (arrangement or {}).items() if v is not None}}
    lot = max(49, w * ld)
    n = max(0, int(n))
    src = [f"count {n} house(s) of {part.get('name')}",
           f"density `{part.get('density') or 'medium'}`: target cover {mid:.0%} "
           f"(the middle of the band {lo:.0%}-{hi if hi is None else format(hi, '.0%')})",
           *src]
    # over the ground the compiler can DEVELOP, not the rectangle: the arterial's band
    # and the standing parts' clearances take a share of every district (the city's
    # middle ring measured 5776 developable of 7560), and a rectangle sized at the
    # band's middle over its whole area laid its lots at the band's ceiling and was
    # refused by thirteen columns (the coordinator, the expression round's city)
    need = int(math.ceil(n * lot / max(mid * LAND_DEVELOPABLE, 0.02))) if n else 0
    # the least ground the word admits at all: at the top of the band (a `dense` word
    # has none and takes the fabric's own share); what a site is held to
    top = float(hi) if hi is not None else max(mid, 0.4)
    least = int(math.ceil(n * lot / max(top, 0.02))) if n else 0
    k = max(1, min(max(n, 1), dc.BLOCK_LOTS.get(part.get("density") or "medium", 3)))
    gap = (0 if ch.get("attached") else
           placeplan.PLOT_LANE if ch.get("frontage") == "open" else dc.LOT_GAP)
    min_len = 2 * dc.EDGE_MARGIN + placeplan.PLOT_LANE + k * w + (k - 1) * gap
    rows_needed = int(math.ceil(n / float(k))) if n else 0
    # **How deep a district of this fabric is, from the arrangement and not a constant**
    # (the design round's second contract). `district_compile.district_depth` is the one
    # arithmetic both sides read: a band of one row of lots is a district, and a parent
    # held to two rows is a parent whose least width had nothing to do with the houses
    # standing in it.
    rows = int((arrangement or {}).get("rows") or min(2, max(rows_needed, 1)))
    row_gap = placeplan.PLOT_LANE if ch.get("frontage") == "open" else dc.LOT_GAP
    min_dep = dc.district_depth(ld, rows, row_gap)
    if arrangement:
        src = list(src) + [f"arrangement: {rows} row(s) of {w}x{ld} lots, "
                           f"{ch.get('frontage') or 'street'} frontage -- a district "
                           f"{min_dep} deep"]
    if n:
        need = max(need, min_len * min_dep)
        least = max(least, min_len * min_dep)
    out = {"what": "lots", "columns": int(need), "columns_least": int(least),
           "target_cover": round(mid, 3), "least_cover": round(top, 3),
           "lot": [int(w), int(ld)], "min_len": int(min_len), "min_dep": int(min_dep),
           "houses": n, "from": src}
    ent = entity_of(part, spec)
    if ent.get("class") == "land":
        use = ent.get("land_use") or spec_mod.land_use(part)
        hh = int(households if households is not None else
                 (spec or {}).get("structures") or n)
        per = int(((allocation or {}).get("land") or {}).get("per_house")
                  or LAND_PER_HOUSE.get(use, LAND_PER_HOUSE["farmland"]))
        land_cols = int(math.ceil(hh * per / LAND_FILL))
        over = ((allocation or {}).get("land") or {}).get(part.get("name"))
        if isinstance(over, dict) and over.get("columns"):
            land_cols = max(land_cols, int(over["columns"]))
            src = src + [f"allocation.land {part.get('name')}: at least {over['columns']} columns"]
        out.update({"what": "land", "land_use": use, "households": hh,
                    "per_house": per, "land_columns": int(land_cols),
                    "columns": int(need + land_cols),
                    "columns_least": int(least + land_cols),
                    "from": [f"land use `{use}`", f"{hh} household(s) of the place",
                             f"LAND_PER_HOUSE[{use}] = {per} columns a household over "
                             f"LAND_FILL {LAND_FILL:g}", *src]})
    return out


def _side_in_band(decl: dict | None, kind: str, side: int) -> int:
    """`side` clamped into the drawn band this type admits (a plot's band is its pad
    plus the inset on both sides), and off its except list where it is a plot."""
    from . import district_compile as dc
    from .buildlib import Builder
    if not decl:
        return max(3, int(side))
    a, b, c, d = (decl.get("needs") or {}).get("footprint", (3, 3, 3, 3))
    if kind == "plot":
        return int(dc._clamp_side(int(side), decl))
    return int(max(min(a, b), min(min(c, d), int(side))))


def anchor_size(spec: dict, decls: dict, part: dict, decl: dict | None, kind: str, *,
                households: int | None = None, civic: list | None = None,
                allocation: dict | None = None, cap: int | None = None,
                least: tuple | None = None) -> dict:
    """**The thing the place is gathered about, sized from what gathers about it.**

        Area = households x `ANCHOR_SHARE[family]` x one fabric lot, plus a lane's depth of
        frontage for every civic part the spec puts `near` or `on` it; the side is the square
        root, clamped into the type's band and under `cap` (the plateau cut for it). Returns
        `{"side", "columns", "share", "from"}`; `allocation.anchor` overrides the share or
        the side outright, and the override is on the `from` list.
        
    """
    fam = str(part.get("family") or "").lower()
    alloc = (allocation or {}).get("anchor") or {}
    share = float(alloc.get("share") if alloc.get("share") is not None
                  else ANCHOR_SHARE.get(fam, ANCHOR_SHARE_DEFAULT))
    share = max(ANCHOR_SHARE_MIN, share)
    hh = int(households if households is not None else spec.get("structures") or 0)
    groups = [p for p in spec.get("defining_parts") or []
              if p["kind"] == "group" and entity_of(p, spec).get("class") == "building"]
    lot_cols, lot_src = 120, "no district of houses: a lot of 120 assumed"
    if groups:
        # **the lot the fabric's own resolved demand asks for** (the design round's
        # first contract): the anchor is sized in lots and a lot sized before the demand
        # is a lot of a type the district may not be built of
        (w, d), src = fabric_lot(groups[0], decls, spec, allocation=allocation,
                                 demand=groups[0].get("demand"))
        lot_cols, lot_src = w * d, src[0]
    area = hh * share * lot_cols
    front = 0
    for c in civic or []:
        cw = int(c.get("side") or 0)
        front += cw * placeplan.PLOT_LANE
    area += front
    side = int(math.ceil(math.sqrt(max(area, 9.0))))
    src = [f"{hh} household(s)", f"ANCHOR_SHARE[{fam}] = {share:g} of one lot"
           + (" (allocation override)" if alloc.get("share") is not None else ""),
           lot_src]
    if front:
        src.append(f"{len(civic or [])} civic part(s) on it: {front} columns of frontage")
    if least and int(least[0]) > side:
        # **what gathers about it has to reach it**: the relation's own reach is a
        # multiple of the anchor's half-diagonal (`intent.AROUND_REACH`), so a layout
        # whose districts stand where the ground puts them (a shore's ribbons) bounds
        # the anchor from below by the farthest of them
        side = int(least[0])
        src.append(str(least[1]))
    if alloc.get("side"):
        side = int(alloc["side"])
        src.append(f"allocation.anchor.side = {side}")
    side = _side_in_band(decl, kind, side)
    if cap is not None and side > int(cap):
        side = int(cap)
        src.append(f"capped at {cap} by the ground levelled for it")
    return {"side": int(side), "columns": int(side) * int(side), "share": share,
            "from": src}


def civic_size(spec: dict, decls: dict, part: dict, decl: dict | None, *,
               allocation: dict | None = None) -> dict:
    """A civic building beside the centre, sized in fabric lots (`CIVIC_LOTS`) and
    clamped into its type's band. `{"side", "columns", "lots", "from"}`."""
    fam = str(part.get("family") or "").lower()
    alloc = ((allocation or {}).get("civic") or {}).get(part.get("name")) or {}
    lots = float(alloc.get("lots") if alloc.get("lots") is not None
                 else CIVIC_LOTS.get(fam, CIVIC_LOTS_DEFAULT))
    groups = [p for p in spec.get("defining_parts") or []
              if p["kind"] == "group" and entity_of(p, spec).get("class") == "building"]
    lot_cols, lot_src = 120, "no district of houses: a lot of 120 assumed"
    if groups:
        (w, d), src = fabric_lot(groups[0], decls, spec, allocation=allocation,
                                 demand=groups[0].get("demand"))
        lot_cols, lot_src = w * d, src[0]
    from .buildlib import Builder
    side = int(math.ceil(math.sqrt(lots * lot_cols))) + 2 * Builder.SITE_INSET
    src = [f"CIVIC_LOTS[{fam}] = {lots:g} lot(s)"
           + (" (allocation override)" if alloc.get("lots") is not None else ""), lot_src,
           f"plus the siting inset of {Builder.SITE_INSET} on every side"]
    side = _side_in_band(decl, "plot", side)
    return {"side": int(side), "columns": int(side) * int(side), "lots": lots,
            "from": src}


class Solver:
    """One solve of one place. `solve()` is the whole of it; the rest is per relation."""

    def __init__(self, spec, site, plateau, decls, voice, vol=None, seed: int = 1,
                 caps: dict | None = None, intent: dict | None = None,
                 allocation: dict | None = None):
        self.spec = spec
        # **The allocation in force** (the expression round): the spec's recorded
        # `allocation` rows under the caller's, and every derived dimension goes on
        # `targets` with its inputs by name.
        self.allocation = allocation_of(spec, allocation)
        self.targets: list = []
        # **The sentence's relations, resolved to this spec's parts** (the closure
        # round). `around` between a group and a solid part is what `_districts` lays
        # the strips for; `beside`/`near` between a solid part and the centre is what
        # `_cands_near` keeps the strips clear for. A word neither the spec nor the
        # library can name resolves to nothing and the record says so.
        self.intent = intent
        self.relations = _relations_of(spec, intent)
        self.exact = bool((spec.get("explicit_count") or {}).get("n")
                          and not (spec.get("explicit_count") or {}).get("about"))
        self.site = site
        self.plateau = plateau or {}
        self.decls = {n: d for n, d in decls.items() if d is not None}
        self.voice = voice
        # **The type the capability record approved, per defining part.** The
        # integration round's third finding: `capability.match` wrote down which type
        # answers each part and then the solver went and picked one of its own, so the
        # record was a description of a decision nobody made. `_type_of` prefers what
        # matching approved and falls back to `_type_for` only where matching has
        # nothing to say -- a round with no capability record behaves exactly as before.
        self.caps = {k: v for k, v in (caps or {}).items() if v}
        self.ground = _Ground(vol)
        self.rng = random.Random(int(seed))
        self.seed = int(seed)
        self.X, self.Z = int(site["origin"][0]), int(site["origin"][1])
        self.S = int(site["size"])
        from .buildlib import Builder
        self.dmin = int(2 * Builder.SITE_INSET + 24)
        self.core = spec_mod.core(spec)
        self.placed: list = []            # leaves, in order
        self.by_name: dict = {}
        self.compounds: list = []
        self.left_over: list = []
        self.districts: list = []
        self.wall: dict | None = None     # the perimeter wall's record
        self.record: list = []
        self.demoted: list = []
        self.fails: list = []
        self.seq = 1
        # the centre: the plateau cut for the core, else the site's middle
        self.cx, self.cz = self.X + self.S // 2, self.Z + self.S // 2
        self.plateau_rect = None
        if self.plateau.get("rect") and self.core \
                and self.plateau.get("part") == self.core["name"]:
            px0, pz0, px1, pz1 = [int(v) for v in self.plateau["rect"]]
            self.plateau_rect = (px0, pz0, px1, pz1)
            self.cx, self.cz = (px0 + px1) // 2, (pz0 + pz1) // 2

    def _type_of(self, d: dict) -> tuple | None:
        """`(name, decl)` of the type this defining part is built as.

                The capability record's answer where it has one and the declarations back it:
                an approved type this checkout cannot load is not silently used, because a
                record naming a type that is not there is worse than no record.
                
        """
        chosen = self.caps.get(d.get("name"))
        if chosen and self.decls.get(chosen) is not None:
            return chosen, self.decls[chosen]
        return _type_for(d["family"], d["kind"], self.decls)

    # --- the whole ---------------------------------------------------------------

    def solve(self) -> tuple:
        parts = list(self.spec["defining_parts"])
        groups = [p for p in parts if p["kind"] == "group" and not spec_mod.compound(p)]
        solid = [p for p in parts if p not in groups]
        # the perimeter before anything that has to stand inside it, the centre before
        # anything beside it: declaration order, with those two pulled forward
        order = sorted(solid, key=lambda p: (
            0 if p["relation"] == "centre" else 1 if p["relation"] == "perimeter"
            else 2 if p["relation"] == "gateway" else 3, parts.index(p)))
        for d in order:
            for k in range(int(d["count"])):
                name = d["name"] if d["count"] == 1 else f"{d['name']}_{k + 1}"
                self._place(d, name, k)
        self._improve()
        self._districts(groups)
        if self.fails:
            return None, self.fails
        return self._place_doc(), []

    # --- one part ------------------------------------------------------------------

    def _place(self, d: dict, name: str, k: int) -> None:
        rel = d["relation"]
        gen = {"centre": self._cands_centre, "perimeter": self._cands_perimeter,
               "gateway": self._cands_gateway, "edge": self._cands_edge,
               "beside_the_centre": self._cands_beside, "near": self._cands_near,
               "along": self._cands_along, "on": self._cands_on,
               "throughout": None, "quarter": None, "concentric": None}.get(rel)
        if gen is None:
            self.fails.append({"part": name, "check": "relation",
                               "why": f"{name}: a {d['kind']} of family {d['family']} "
                                      f"declares `{rel}`, which places a district or a "
                                      f"ring and not a {d['kind']}"})
            return
        cands, why = gen(d, name, k)
        if not cands:
            self.fails.append({"part": name, "check": "relation", "why": why or
                               f"{name}: nothing seeds a candidate for `{rel}`"})
            return
        chosen, dropped, vetoed = self._choose(cands, d, name)
        rec = {"part": name, "relation": rel, "family": d["family"], "kind": d["kind"],
               "candidates": len(cands), "vetoed": vetoed, "demoted": dropped,
               "chosen": None, "cost": None, "kept": []}
        if chosen is None:
            worst = min(cands, key=lambda c: (len(self._vetoes(c, d, name, set())), c["cost"]))
            rec["least_violating"] = {"leaf": _geom(worst["leaf"]),
                                      "vetoes": self._vetoes(worst, d, name, set())}
            self.fails.append({"part": name, "check": "vetoes",
                               "why": f"{name} ({rel}): no candidate of {len(cands)} "
                                      f"passes even with every veto but the site's "
                                      f"demoted; the least violating breaks "
                                      + ", ".join(sorted(rec["least_violating"]["vetoes"])),
                               "vetoes": vetoed})
            self.record.append(rec)
            return
        if dropped:
            self.demoted.append({"part": name, "dropped": dropped,
                                 "why": f"{name} had no feasible candidate under every "
                                        f"veto; {', '.join(dropped)} demoted in weight "
                                        f"order and the least-violating placement taken"})
        self._commit(chosen, d, name)
        rec.update(chosen=_geom(chosen["leaf"]), cost=round(chosen["cost"], 2),
                   kept=[{"leaf": _geom(c["leaf"]), "cost": round(c["cost"], 2)}
                         for c in self._feasible(cands, d, name, set(dropped))[:KEEP]])
        self.record.append(rec)

    def _commit(self, cand: dict, d: dict, name: str) -> None:
        leaf = dict(cand["leaf"])
        leaf["seed"] = self.seq
        self.seq += 1
        if cand.get("compound"):
            self.compounds.append(leaf)
        else:
            self.placed.append(leaf)
        self.by_name[name] = leaf
        if d["relation"] == "perimeter" and leaf.get("kind") == "edge":
            self.wall = {"name": name, "leaf": leaf, "rect": cand["rect"],
                         "half": cand["half"], "inset": cand["inset"],
                         "shape": cand.get("shape", "square")}

    def _order(self, cands: list) -> list:
        for c in cands:
            c.setdefault("tie", self.rng.random())
        return sorted(cands, key=lambda c: (c["cost"], c["key"], c["tie"]))

    def _feasible(self, cands, d, name, dropped: set) -> list:
        return [c for c in self._order(cands) if not self._vetoes(c, d, name, dropped)]

    def _choose(self, cands: list, d: dict, name: str) -> tuple:
        """The least-cost feasible candidate, demoting vetoes from the lightest up."""
        vetoed: dict = {}
        for c in cands:
            for v in self._vetoes(c, d, name, set()):
                vetoed[v] = vetoed.get(v, 0) + 1
        dropped: list = []
        order = [v for v, _w in sorted(VETOES, key=lambda vw: vw[1])]   # lightest first
        while True:
            ok = self._feasible(cands, d, name, set(dropped))
            if ok:
                return ok[0], dropped, vetoed
            nxt = [v for v in order if v not in dropped and v != "site"]
            if not nxt:
                return None, dropped, vetoed
            dropped.append(nxt[0])

    # --- the vetoes -----------------------------------------------------------------

    def _vetoes(self, cand: dict, d: dict, name: str, dropped: set) -> list:
        leaf = cand["leaf"]
        rects = [cand["rect"]] if cand.get("compound") else _leaf_rects(leaf)
        out = []
        X, Z, S = self.X, self.Z, self.S
        if "site" not in dropped:
            for r in rects:
                if not (X <= r[0] and r[2] < X + S and Z <= r[1] and r[3] < Z + S):
                    out.append("site")
                    break
        if "overlap" not in dropped:
            mine = _clearance(self.decls, leaf.get("type")) if not cand.get("compound") else 0
            passage = bool((self.decls.get(leaf.get("type")) or {}).get("passage"))
            for other in self.placed + self.compounds:
                if other.get("name") == name:
                    continue
                o_pass = bool((self.decls.get(other.get("type")) or {}).get("passage"))
                kinds = {leaf.get("kind", "plot"), other.get("kind", "plot")}
                if (passage or o_pass) and "edge" in kinds:
                    continue                  # a gate stands in its wall
                theirs = _clearance(self.decls, other.get("type")) \
                    if other in self.placed else 0
                m = max(mine, theirs)
                orects = _leaf_rects(other) if other in self.placed else \
                    [(other["x0"], other["z0"], other["x1"], other["z1"])]
                if any(_overlaps(a, _grow(b, m)) for a in rects for b in orects):
                    out.append("overlap")
                    break
        if "footprint" not in dropped and not cand.get("compound"):
            decl = self.decls.get(leaf.get("type"))
            if decl and pipeline.needs_footprint_failure(
                    {**leaf, "name": name}, decl["needs"]):
                out.append("footprint")
        if "inside" not in dropped and self.wall is not None \
                and d["relation"] not in ("perimeter", "gateway", "on") \
                and name != self.wall["name"]:
            path = self.wall["leaf"]["path"]
            for r in rects:
                corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]
                if not all(inside(path, c) for c in corners):
                    out.append("inside")
                    break
        if "plateau" not in dropped and cand.get("compound") and self.plateau_rect:
            px0, pz0, px1, pz1 = self.plateau_rect
            x0, z0, x1, z1 = cand["rect"]
            ok = px0 <= x0 and x1 <= px1 and pz0 <= z0 and z1 <= pz1
            share = ((x1 - x0 + 1) * (z1 - z0 + 1)
                     / float((px1 - px0 + 1) * (pz1 - pz0 + 1)))
            if not ok or share < COMPOUND_PLATEAU_SHARE:
                out.append("plateau")
        return out

    # --- the costs
    # ---------------------------------------------------------------------

    def _cost(self, rect, anchor=None) -> float:
        c = COSTS["relief"] * self.ground.relief(rect) \
            + COSTS["water"] * self.ground.water(rect)
        if anchor is not None:
            mx, mz = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
            c += COSTS["distance"] * math.hypot(mx - anchor[0], mz - anchor[1])
        return c

    # --- the candidate generators ----------------------------------------------------

    def _cands_centre(self, d, name, k) -> tuple:
        cx, cz = self.cx, self.cz
        if spec_mod.compound(d):
            if self.plateau_rect:
                rect = self.plateau_rect
            else:
                side = int(compound_ground(spec=self.spec, site_side=self.S,
                                           part=d)["side"])
                side = max(COMPOUND_MIN, side)
                rect = (cx - side // 2, cz - side // 2,
                        cx - side // 2 + side - 1, cz - side // 2 + side - 1)
            leaf = {"name": name, "defines": d["name"], "x0": rect[0], "z0": rect[1],
                    "x1": rect[2], "z1": rect[3],
                    "notes": (f"The {d['family']} at the centre of the place, over the "
                              f"whole of the ground levelled for it. " + (d.get("notes") or ""))}
            return [{"leaf": leaf, "rect": rect, "compound": True,
                     "cost": self._cost(rect), "key": rect}], None
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']}) at the centre")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        cap = None
        if self.plateau_rect:
            cap = min(self.plateau_rect[2] - self.plateau_rect[0] + 1,
                      self.plateau_rect[3] - self.plateau_rect[1] + 1)
        # **The centre is sized from what gathers about it** (the expression round): an
        # area (a square, a plaza) from the households, the anchor share and the civic
        # parts the spec puts on it; a plot (a keep, a hall at the centre) in civic
        # lots. Never `_rect_for`'s largest admitted rectangle.
        sized = self._sized(d, decl, kind, cap=cap)
        cands = []
        for dx, dz in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
            if kind == "point":
                leaf = {"name": name, "defines": d["name"], "kind": "point", "type": tname,
                        "params": {}, "at": [cx + dx, cz + dz], "facing": "north",
                        "notes": "the thing at the centre"}
                rect = pipeline.part_rect(leaf)
            else:
                rect = self._rect_sized(sized, decl, kind, cx + dx, cz + dz, cap=cap)
                leaf = {"name": name, "defines": d["name"], "kind": kind, "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3],
                        "notes": (f"the thing at the centre, sized from the programme: "
                                  f"{'; '.join(sized['from'])}" if sized else
                                  "the thing at the centre, at its largest"),
                        **({"target": sized} if sized else {})}
            cands.append({"leaf": leaf, "rect": rect, "key": rect,
                          "cost": self._cost(rect, (cx, cz))})
        return cands, None

    def _households(self) -> int:
        """The households the place is sized for: the sentence's count, else the
        programme's **first** inferred target -- a negotiation of the promise
        (`negotiated: size_band`) does not move the ground it was measured on, which is
        the repair module's own rule for the site."""
        ec = self.spec.get("explicit_count") or {}
        if ec.get("n"):
            return int(ec["n"])
        for row in self.spec.get("negotiated") or []:
            if isinstance(row, dict) and row.get("what") == "size_band":
                first = (row.get("from") or {}).get("structures")
                if first:
                    return int(first)
        return int(self.spec.get("structures") or 0)

    def _civic_on(self, core_name: str | None) -> list:
        """The civic plot parts the spec puts `near`/`on`/`beside_the_centre` the
        centre, each with the side it will be drawn at, for the anchor's frontage."""
        out = []
        for p in self.spec["defining_parts"]:
            if p["kind"] != "plot" or p["relation"] not in ("near", "beside_the_centre", "on"):
                continue
            if p.get("of") and p["of"] != core_name:
                continue
            got = self._type_of(p)
            decl = got[1] if got else None
            cs = civic_size(self.spec, self.decls, p, decl, allocation=self.allocation)
            out.append({"part": p["name"], "side": cs["side"]})
        return out

    def _sized(self, d: dict, decl: dict | None, kind: str, *, cap=None) -> dict | None:
        """The programme-derived size of one solid defining part, or None where the
        part is not one this rule sizes (an edge, a point)."""
        if kind == "area":
            got = anchor_size(self.spec, self.decls, d, decl, kind,
                              households=self._households(),
                              civic=self._civic_on(d["name"]) if d["relation"] == "centre"
                              else [], allocation=self.allocation, cap=cap,
                              least=getattr(self, "_anchor_least", None)
                              if d["relation"] == "centre" else None)
            got = {"what": "anchor", "part": d["name"], **got}
        elif kind == "plot":
            got = civic_size(self.spec, self.decls, d, decl, allocation=self.allocation)
            got = {"what": "civic", "part": d["name"], **got}
        else:
            return None
        self.targets = [t for t in self.targets if t.get("part") != d["name"]] + [got]
        return got

    @staticmethod
    def _rect_sized(sized: dict | None, decl: dict, kind: str, cx: int, cz: int,
                    cap=None) -> tuple:
        if not sized:
            return _rect_for(decl, kind, cx, cz, cap=cap)
        w = max(3, int(sized["side"]))
        if cap is not None:
            w = min(w, int(cap))
        x0, z0 = cx - w // 2, cz - w // 2
        return (x0, z0, x0 + w - 1, z0 + w - 1)

    def _cands_perimeter(self, d, name, k) -> tuple:
        walls = _wall_types(self.decls)
        if not walls:
            return [], (f"{name}: no committed edge type of this place's form has a "
                        f"height parameter to build a wall with")
        # the type named for the family -- a town's `wall` is a wall, not the great wall
        # of a city's outermost ring -- at the top of its band; the tallest edge type
        # where nothing is named for it
        named = [w for w in walls if w[0] == d["family"]] or walls
        tname, tdecl, lo_h, hi_h = named[0]
        # **a town wall over the storeys of its fabric** (the expression round), not the
        # top of the type's band; `_write_hierarchy` records the kind and why
        kind, _ksrc = wall_kind_for(d, self.spec, outermost=True)
        if kind == "great":
            tname, tdecl, lo_h, hi_h = max(walls, key=lambda w: (w[3], w[0]))
        height, _hsrc = wall_height_for(kind, self.spec, tdecl,
                                        storeys=_fabric_storeys(self.spec))
        height = int(height)
        max_run = int(tdecl["needs"]["footprint"][3])
        width = 3 if "width" in (tdecl.get("params") or {}) else 1
        inset = _wall_inset(tdecl, width)
        octagon = bool(wall_round_for(d, self.spec) and tdecl.get("diagonal"))
        # the least half-side: the centre's ground and every district's, inside the
        # inset, at the coverage the strips round a centre are held to
        need = self._ground_needed() / RING_COVERAGE
        h_min = max(int(math.ceil(math.sqrt(need) / 2.0)) + inset, self.dmin + inset)
        # **...and a district's depth on every side of the core** (the expression
        # round): a gathered group stands on three sides of its centre, and a wall that
        # leaves the strips beside the core under `dmin` deep leaves one side of four
        gathered = any(r["relation"] == "around" and r.get("subject_groups")
                       and r.get("object_part") == (self.core or {}).get("name")
                       for r in self.relations)
        if gathered:
            h_min = max(h_min, self._core_half() + LANE_GAP + self.dmin + inset)
        # the largest loop the site holds with the edge inset on every side
        h_max = (self.S - 1 - 2 * RING_EDGE_INSET) // 2
        if h_min > h_max:
            h_min = h_max
        cx, cz = self.cx, self.cz
        cands = []
        hs = list(range(h_max, h_min - 1, -WALL_STEP_BLOCKS))
        if hs[-1] != h_min:
            hs.append(h_min)
        for h in hs:
            for (sx, sz) in WALL_SHIFTS:
                ox, oz = cx + sx * WALL_STEP_BLOCKS, cz + sz * WALL_STEP_BLOCKS
                # clamped into the site less the edge inset, so the loop stands on the
                # place's ground: a wall centred on a plateau near the site's edge is a
                # wall moved in, not one refused
                lo_x, hi_x = self.X + RING_EDGE_INSET + h, self.X + self.S - 1 - RING_EDGE_INSET - h
                lo_z, hi_z = self.Z + RING_EDGE_INSET + h, self.Z + self.S - 1 - RING_EDGE_INSET - h
                if lo_x > hi_x or lo_z > hi_z:
                    continue
                ox, oz = min(max(ox, lo_x), hi_x), min(max(oz, lo_z), hi_z)
                rect = (ox - h, oz - h, ox + h, oz + h)
                if (rect[0], rect[1], rect[2], rect[3]) in [c["rect"] for c in cands]:
                    continue
                path = (_octagon_path(ox, oz, h, max_run) if octagon
                        else _square_path(ox, oz, h, max_run))
                params = {"height": height}
                if width == 3:
                    params["width"] = 3
                face = wall_face_for(d, self.spec)
                if "face" in (tdecl.get("params") or {}):
                    choices = list(tdecl["params"]["face"][1])
                    params["face"] = ("unbroken" if face == "plain"
                                      and wall_stairs_for(d, self.spec) == "sparse"
                                      and "unbroken" in choices else face)
                leaf = {"kind": "edge", "name": name, "defines": d["name"], "type": tname,
                        "params": params, "path": path, "width": width,
                        **({"shape": "octagon"} if octagon else {}),
                        **({"face": face} if face else {}),
                        "notes": f"the circuit of the place, {height} high as `{tname}`, "
                                 f"half-side {h} about ({ox},{oz}); solved"}
                rise, wet = self.ground.line(_edge_cells(leaf))
                cost = (COSTS["line_relief"] * rise + COSTS["line_water"] * wet
                        + COSTS["shrink"] * (h_max - h)
                        + COSTS["wall_distance"] * math.hypot(ox - cx, oz - cz))
                cands.append({"leaf": leaf, "rect": rect, "half": h, "inset": inset,
                              "shape": "octagon" if octagon else "square",
                              "cost": cost, "key": (h_max - h, rect)})
        return cands, None

    def _ground_needed(self) -> float:
        """The columns the districts and the centre need inside a perimeter wall, by
        the one sizing rule (`land_need`: houses at their density's target cover, land
        by the households that work it) plus the core and the lanes round it. The
        expression round: the estimate before this charged a house its lot over
        `DISTRICT_FILL`, knew nothing of an orchard, and drew a hamlet's wall so tight
        that one strip of the four fit inside it."""
        cols = 0.0
        groups = [p for p in self.spec["defining_parts"]
                  if p["kind"] == "group" and not spec_mod.compound(p)]
        hh = self._households()
        for p in groups:
            ent = entity_of(p, self.spec)
            n = int(p.get("structures") or 0) if ent.get("counted") else 0
            if ent.get("class") == "land" and ent.get("counted"):
                n = min(n, max(1, int(math.ceil(hh * LAND_HOUSE_SHARE))))
            try:
                cols += float(land_need(p, n, self.decls, self.spec, households=hh,
                                        allocation=self.allocation)["columns"])
            except Exception:                # noqa: BLE001 -- the old estimate stands in
                cols += n * spec_mod.columns_per_plot(p) / DISTRICT_FILL
        if not cols:
            cols = int(self.spec.get("structures") or 0) * spec_mod.COLUMNS_PER_PLOT / DISTRICT_FILL
        centre = self.by_name.get((self.core or {}).get("name"))
        if centre is not None:
            r = pipeline.part_rect({**centre, "name": centre.get("name")})
            cols += (r[2] - r[0] + 1 + 2 * LANE_GAP) * (r[3] - r[1] + 1 + 2 * LANE_GAP)
        return cols

    def _core_half(self) -> int:
        """Half the core box's longer side, or 0 with nothing at the centre yet."""
        names = {(self.core or {}).get("name")}
        names |= {p["name"] for p in self.spec["defining_parts"]
                  if p["relation"] in ("beside_the_centre", "near")
                  and (not p.get("of") or p["of"] == (self.core or {}).get("name"))}
        rects = []
        for leaf in self.placed + self.compounds:
            base = next((p["name"] for p in self.spec["defining_parts"]
                         if _answers(leaf, p)), None)
            if base in names:
                rects.append(pipeline.part_rect({**leaf, "name": leaf.get("name")})
                             if leaf.get("kind") else
                             (leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"]))
        if not rects:
            return 0
        w = max(r[2] for r in rects) - min(r[0] for r in rects) + 1
        d = max(r[3] for r in rects) - min(r[1] for r in rects) + 1
        return int(math.ceil(max(w, d) / 2.0))

    def _cands_gateway(self, d, name, k) -> tuple:
        if self.wall is None:
            return [], f"{name}: a gateway is a way through a wall and this place has none"
        gate = _gate_type(self.decls, d.get("family", "gate"))
        if gate is None:
            return [], (f"{name}: no committed passage point type of this place's form "
                        f"builds a gate for {self.wall['name']}")
        w = self.wall
        ox, oz = (w["rect"][0] + w["rect"][2]) // 2, (w["rect"][1] + w["rect"][3]) // 2
        h = w["half"]
        from .buildlib import Builder
        pad = Builder.point_pad(int(w["leaf"]["params"].get("height") or 0),
                                gate[1]["needs"]["footprint"])
        taken = {tuple(p["at"]) for p in self.placed if p.get("kind") == "point"}
        cands = []
        for side in _SIDES:
            gx, gz = _gate_at(ox, oz, h, side)
            if (gx, gz) in taken:
                continue
            leaf = {"kind": "point", "name": name, "defines": d["name"], "type": gate[0],
                    "params": _top_params(gate[1]), "at": [gx, gz],
                    "facing": _FACING[side], "size": int(pad),
                    "notes": f"the way through {w['name']} on its {side} side; solved"}
            rect = pipeline.part_rect(leaf)
            cands.append({"leaf": leaf, "rect": rect, "key": rect,
                          "cost": self._cost(rect), "side": side})
        return cands, None

    def _bounds(self) -> tuple:
        """The rectangle everything inside the place stands in: inside the wall's
        inset, or the site less its edge inset."""
        if self.wall is not None:
            x0, z0, x1, z1 = self.wall["rect"]
            i = self.wall["inset"]
            return (x0 + i, z0 + i, x1 - i, z1 - i)
        e = RING_EDGE_INSET
        return (self.X + e, self.Z + e, self.X + self.S - 1 - e, self.Z + self.S - 1 - e)

    def _around(self, target_rect, d, name, tname, decl, kind, anchor_axis=None) -> list:
        """Candidates of this type on the four sides of a rectangle, a gap away."""
        tx0, tz0, tx1, tz1 = target_rect
        gap = max(_clearance(self.decls, tname), 0) + 1 + BESIDE_GAP
        # **A part beside the centre is sized from the programme too** (the expression
        # round): a hall on the square in civic lots, a green beside it by the anchor
        # share, and never at the type's largest.
        sized = self._sized(d, decl, kind)
        side_len = (int(sized["side"]) if sized else
                    _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4))
        cands = []
        for side in _SIDES:
            for along in (0, 1, -1):
                if side in ("north", "south"):
                    cx = (tx0 + tx1) // 2 + along * max(1, (tx1 - tx0) // 2)
                    cz = (tz0 - gap - side_len // 2 - 1) if side == "north" \
                        else (tz1 + gap + side_len // 2 + 1)
                else:
                    cz = (tz0 + tz1) // 2 + along * max(1, (tz1 - tz0) // 2)
                    cx = (tx0 - gap - side_len // 2 - 1) if side == "west" \
                        else (tx1 + gap + side_len // 2 + 1)
                if kind == "point":
                    leaf = {"kind": "point", "name": name, "defines": d["name"],
                            "type": tname, "params": {}, "at": [cx, cz],
                            "facing": _FACING[side], "notes": f"{d['relation']}; solved"}
                    rect = pipeline.part_rect(leaf)
                else:
                    rect = self._rect_sized(sized, decl, kind, cx, cz)
                    leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                            "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                            "z1": rect[3], "notes": f"{d['relation']}, on the {side} "
                                                    f"side; solved"
                            + (f"; sized from the programme: {'; '.join(sized['from'])}"
                               if sized else ""),
                            **({"target": sized} if sized else {})}
                cost = self._cost(rect, ((tx0 + tx1) / 2.0, (tz0 + tz1) / 2.0))
                if anchor_axis and side == anchor_axis:
                    cost += COSTS["axis"]
                cost += COSTS["strips"] * self._sides_lost(target_rect, rect)
                cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": cost,
                              "side": side})
        return cands

    def _sides_lost(self, core_rect, rect) -> int:
        """How many sides round the centre a satellite standing at `rect` would close
        to the districts: the strips cut round the core box alone, less the strips cut
        round the box grown to hold the satellite. Zero where the place has no district
        groups or where the satellite is not the core's."""
        groups = [p for p in self.spec["defining_parts"]
                  if p["kind"] == "group" and not spec_mod.compound(p)]
        core = self.by_name.get((self.core or {}).get("name"))
        if not groups or core is None:
            return 0
        crect = pipeline.part_rect({**core, "name": core.get("name")}) \
            if core.get("kind") else (core["x0"], core["z0"], core["x1"], core["z1"])
        if tuple(core_rect) != tuple(crect):
            return 0
        before = len(self._strips([crect]))
        after = len(self._strips([crect, tuple(rect)]))
        return max(0, before - after)

    def _strips(self, core_rects: list) -> dict:
        """The strips of the place's bounds round the box these rectangles span, a lane
        clear, each at least `dmin` on both sides; the one cut `_districts` sectors."""
        bx0, bz0, bx1, bz1 = self._bounds()
        g = LANE_GAP
        cx0 = min(r[0] for r in core_rects) - g
        cz0 = min(r[1] for r in core_rects) - g
        cx1 = max(r[2] for r in core_rects) + g
        cz1 = max(r[3] for r in core_rects) + g
        strips = {"north": (bx0, bz0, bx1, cz0 - 1), "south": (bx0, cz1 + 1, bx1, bz1),
                  "west": (bx0, cz0, cx0 - 1, cz1), "east": (cx1 + 1, cz0, bx1, cz1)}
        strips = {k: (max(bx0, r[0]), max(bz0, r[1]), min(bx1, r[2]), min(bz1, r[3]))
                  for k, r in strips.items()}
        return {k: r for k, r in strips.items()
                if r[0] <= r[2] and r[1] <= r[3]
                and r[2] - r[0] + 1 >= self.dmin and r[3] - r[1] + 1 >= self.dmin}

    def _cands_beside(self, d, name, k) -> tuple:
        centre = self.by_name.get((self.core or {}).get("name"))
        if centre is None:
            return [], f"{name}: nothing stands at the centre for this part to be beside"
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        rect = pipeline.part_rect({**centre, "name": centre.get("name")}) \
            if "x0" in centre or "at" in centre or "path" in centre \
            else (centre["x0"], centre["z0"], centre["x1"], centre["z1"])
        gate_side = None
        for p in self.placed:
            if p.get("kind") == "point" and (self.decls.get(p.get("type")) or {}).get("passage"):
                gate_side = {"north": "south", "south": "north",
                             "east": "west", "west": "east"}.get(p.get("facing"))
                break
        return self._around(rect, d, name, tname, decl, kind, anchor_axis=gate_side), None

    def _cands_near(self, d, name, k) -> tuple:
        of = d.get("of") or (self.core or {}).get("name")
        target = self.by_name.get(of) or next(
            (c for c in self.compounds if c.get("name") == of), None)
        if target is None:
            return [], f"{name}: `near` names {of!r} and nothing of that name is placed"
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        rect = (pipeline.part_rect({**target, "name": target.get("name")})
                if target.get("kind") else
                (target["x0"], target["z0"], target["x1"], target["z1"]))
        return self._around(rect, d, name, tname, decl, kind), None

    def _cands_edge(self, d, name, k) -> tuple:
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']})")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        bx0, bz0, bx1, bz1 = self._bounds()
        side_len = _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4)
        half = side_len // 2
        cands = []
        spots = []
        for t in (0.5, 0.25, 0.75):
            spots += [(int(bx0 + (bx1 - bx0) * t), bz0 + half, "north"),
                      (int(bx0 + (bx1 - bx0) * t), bz1 - half, "south"),
                      (bx0 + half, int(bz0 + (bz1 - bz0) * t), "west"),
                      (bx1 - half, int(bz0 + (bz1 - bz0) * t), "east")]
        for cx, cz, side in spots:
            if kind == "point":
                leaf = {"kind": "point", "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "at": [cx, cz], "facing": _FACING[side],
                        "notes": "at the edge of the place; solved"}
                rect = pipeline.part_rect(leaf)
            else:
                rect = _rect_for(decl, kind, cx, cz)
                leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3], "notes": f"at the {side} edge of the place; solved"}
            cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": self._cost(rect),
                          "side": side})
        return cands, None

    def _target_edge(self, d, name):
        of = d.get("of") or (self.wall or {}).get("name")
        target = self.by_name.get(of)
        if target is None or target.get("kind") != "edge":
            return None, (f"{name}: `{d['relation']}` names {of!r} and no edge of that "
                          f"name is placed")
        return target, None

    def _cands_on(self, d, name, k) -> tuple:
        target, why = self._target_edge(d, name)
        if target is None:
            return [], why
        got = self._type_of({**d, "kind": "point"})
        if got is None:
            return [], (f"{name}: no committed point type of this place's form builds a "
                        f"{d['family']} on {target['name']}")
        tname, decl = got
        taken = {tuple(p["at"]) for p in self.placed if p.get("kind") == "point"}
        cells = sorted(set(_edge_cells(target)))
        path = [(int(a[0]), int(a[1])) for a in target["path"]]
        # the corners of the run and the quarter points of every segment, on the line
        spots = list(path[:-1])
        for a, b in zip(path, path[1:]):
            for t in (0.25, 0.5, 0.75):
                spots.append((int(round(a[0] + (b[0] - a[0]) * t)),
                              int(round(a[1] + (b[1] - a[1]) * t))))
        cands = []
        for (x, z) in spots:
            if (x, z) in taken or (x, z) not in set(cells):
                continue
            leaf = {"kind": "point", "name": name, "defines": d["name"], "type": tname,
                    "params": {}, "at": [x, z], "facing": "north",
                    "notes": f"on {target['name']}; solved"}
            rect = pipeline.part_rect(leaf)
            cands.append({"leaf": leaf, "rect": rect, "key": rect, "cost": self._cost(rect)})
        return cands, None

    def _cands_along(self, d, name, k) -> tuple:
        target, why = self._target_edge(d, name)
        if target is None:
            return [], why
        got = self._type_of(d)
        if got is None:
            return [], (f"{name}: no committed type of this place's form builds a "
                        f"{d['family']} ({d['kind']}) along {target['name']}")
        tname, decl = got
        kind = decl.get("kind", d["kind"])
        gap = _clearance(self.decls, target.get("type")) + int(target.get("width", 1)) // 2 + 1
        side_len = _clean_side(decl) + (0 if kind in ("edge", "point", "area") else 4)
        path = [(int(a[0]), int(a[1])) for a in target["path"]]
        bx0, bz0, bx1, bz1 = self._bounds()
        mid_in = ((bx0 + bx1) / 2.0, (bz0 + bz1) / 2.0)
        cands = []
        for a, b in zip(path, path[1:]):
            if a[0] != b[0] and a[1] != b[1]:
                continue
            for t in (0.5, 0.25, 0.75):
                mx = int(round(a[0] + (b[0] - a[0]) * t))
                mz = int(round(a[1] + (b[1] - a[1]) * t))
                # the inside of the run: toward the middle of the bounds
                if a[0] == b[0]:
                    sx = 1 if mid_in[0] > mx else -1
                    cx, cz = mx + sx * (gap + side_len // 2 + 1), mz
                else:
                    sz = 1 if mid_in[1] > mz else -1
                    cx, cz = mx, mz + sz * (gap + side_len // 2 + 1)
                rect = _rect_for(decl, kind, cx, cz)
                leaf = {"kind": kind, "name": name, "defines": d["name"], "type": tname,
                        "params": {}, "x0": rect[0], "z0": rect[1], "x1": rect[2],
                        "z1": rect[3], "notes": f"along {target['name']}; solved"}
                cands.append({"leaf": leaf, "rect": rect, "key": rect,
                              "cost": self._cost(rect, (mx, mz))})
        return cands, None

    # --- the improvement
    # ---------------------------------------------------------------

    def _improve(self) -> None:
        """One pass: each placed part tried at its next-best kept candidates with the
        rest fixed; kept where every veto still holds and the whole costs less."""
        for _ in range(IMPROVE_PASSES):
            for rec in self.record:
                if rec.get("chosen") is None or len(rec["kept"]) < 2:
                    continue
                name = rec["part"]
                d = next(p for p in self.spec["defining_parts"]
                         if p["name"] == name or name.startswith(p["name"] + "_"))
                if d["relation"] in ("perimeter", "gateway", "on"):
                    continue                  # a wall moves everything on and in it
                current = self.by_name[name]
                where = self.placed.index(current) if current in self.placed else None
                if where is None:
                    continue
                best = None
                for alt in rec["kept"][1:]:
                    leaf = dict(current)
                    leaf.update({k: v for k, v in alt["leaf"].items()})
                    cand = {"leaf": leaf, "rect": pipeline.part_rect(
                        {**leaf, "name": name}), "cost": alt["cost"]}
                    self.placed[where] = leaf
                    ok = not self._vetoes(cand, d, name, set()) and all(
                        not self._vetoes({"leaf": o, "rect": pipeline.part_rect(
                            {**o, "name": o["name"]}), "cost": 0.0},
                            next(p for p in self.spec["defining_parts"]
                                 if p["name"] == o["name"]
                                 or o["name"].startswith(p["name"] + "_")),
                            o["name"], set())
                        for o in self.placed if o is not leaf)
                    self.placed[where] = current
                    if ok and alt["cost"] < rec["cost"] and (best is None
                                                              or alt["cost"] < best["cost"]):
                        best = cand
                if best is not None:
                    was = rec["cost"]
                    self.placed[where] = best["leaf"]
                    self.by_name[name] = best["leaf"]
                    rec["improved"] = {"from": was, "to": round(best["cost"], 2)}
                    rec["chosen"] = _geom(best["leaf"])
                    rec["cost"] = round(best["cost"], 2)

    # --- the districts
    # --------------------------------------------------------------------

    def _districts(self, groups: list) -> None:
        if not groups:
            return
        bx0, bz0, bx1, bz1 = self._bounds()
        # the core box: the centre and everything beside it, a lane clear
        core_names = {(self.core or {}).get("name")}
        core_names |= {p["name"] for p in self.spec["defining_parts"]
                       if p["relation"] in ("beside_the_centre",)}
        # **...and what stands `near` the centre by name** (the closure round): a hall
        # on the square is part of the middle of the place, and the strips are cut round
        # the two of them together rather than trimmed round the hall afterwards --
        # which took a whole side with it.
        core_names |= {p["name"] for p in self.spec["defining_parts"]
                       if p["relation"] == "near"
                       and p.get("of") == (self.core or {}).get("name")}
        core_rects = []
        for leaf in self.placed + self.compounds:
            base = next((p["name"] for p in self.spec["defining_parts"]
                         if _answers(leaf, p)), None)
            if base in core_names:
                core_rects.append(pipeline.part_rect({**leaf, "name": leaf.get("name")})
                                  if leaf.get("kind") else
                                  (leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"]))
        g = LANE_GAP
        if core_rects:
            cx0 = min(r[0] for r in core_rects) - g
            cz0 = min(r[1] for r in core_rects) - g
            cx1 = max(r[2] for r in core_rects) + g
            cz1 = max(r[3] for r in core_rects) + g
        else:
            cx0, cz0, cx1, cz1 = self.cx, self.cz, self.cx, self.cz
        strips = {"north": (bx0, bz0, bx1, cz0 - 1), "south": (bx0, cz1 + 1, bx1, bz1),
                  "west": (bx0, cz0, cx0 - 1, cz1), "east": (cx1 + 1, cz0, bx1, cz1)}
        # **Clamped to the place's own bounds.** v2, C5: the strips are cut from the
        # core's box, and a core part at the *edge* of the site -- a village's hard on
        # the water, anything placed `edge` where no part is at the centre -- puts the
        # box's margin outside the site, so the strips beside it reached two columns
        # past the ground the place stands on and the place level refused its own layout
        # by name. A strip is what is left of the site, never more than it.
        strips = {k: (max(bx0, r[0]), max(bz0, r[1]), min(bx1, r[2]), min(bz1, r[3]))
                  for k, r in strips.items()}
        strips = {k: r for k, r in strips.items() if r[0] <= r[2] and r[1] <= r[3]}
        # **A side too thin for a strip gives its corners to its neighbours** (the
        # expression round): a place whose centre stands near the site's south edge has
        # no south strip, and the west and east strips were cut to the core's own depth,
        # so the houses beside the centre could not reach past its axis **...and a side
        # too thin for the fabric that has to stand in it** (the design round's first
        # contract, at the place level). `dmin` is a constant about what a district
        # *is*; how deep a band has to be to hold a row of the houses this request
        # requires is a fact about the resolved demand. Found by running the farming
        # village whose two-storey cottages need a 24x24 lot: the west and east bands
        # were 28 deep, `_fabric_depth` says 34, the bands were cut anyway, the
        # allocator promised them houses and the compiler laid none -- fourteen of the
        # sentence's sixteen. A band no house of this demand can stand in is not a band
        # for houses; its ground goes to its neighbours, as a thin side's always has.
        depth_min = self._fabric_depth(groups)
        # the depth of a band is its **shorter side**, whichever axis that is: the side
        # a row of lots has to fit across. Naming the axis from the compass point is
        # wrong for the slivers a core box leaves -- the farming village's west band is
        # 83 along x and 28 across z, and the 28 is what a 24-deep cottage cannot fit
        # in.
        thin = {k for k, r in strips.items()
                if min(r[2] - r[0] + 1, r[3] - r[1] + 1) < depth_min}
        if thin and depth_min > self.dmin:
            # on the district record, not on `self.record` -- that one is the relation
            # placements and every row of it is a `(part, relation)`
            self._thin_for_fabric = {
                "sides": sorted(thin), "depth_min": int(depth_min),
                "dmin": int(self.dmin),
                "why": (f"a band holding one row of this request's own fabric is "
                        f"{depth_min} deep, not {self.dmin}: "
                        f"{', '.join(sorted(thin))} cannot hold a house of it and its "
                        f"ground goes to the sides that can")}
        for k in thin:
            strips.pop(k, None)
        for k in ("west", "east"):
            if k not in strips:
                continue
            x0s, z0s, x1s, z1s = strips[k]
            if "north" in thin or "north" not in strips:
                z0s = bz0
            if "south" in thin or "south" not in strips:
                z1s = bz1
            strips[k] = (x0s, z0s, x1s, z1s)
        for k in ("north", "south"):
            if k not in strips:
                continue
            x0s, z0s, x1s, z1s = strips[k]
            if "west" in thin or "west" not in strips:
                x0s = bx0
            if "east" in thin or "east" not in strips:
                x1s = bx1
            strips[k] = (x0s, z0s, x1s, z1s)
        # ...less every other placed part, trimmed on the axis that loses least
        others = [pipeline.part_rect({**leaf, "name": leaf.get("name")})
                  for leaf in self.placed
                  if next((p["name"] for p in self.spec["defining_parts"]
                           if _answers(leaf, p)), None) not in core_names
                  and leaf.get("kind") not in ("edge",)
                  and not (self.decls.get(leaf.get("type")) or {}).get("passage")]
        sectors = []
        for sname in _SIDES:
            rect = strips.get(sname)
            if rect is None:
                continue
            for o in others:
                rect = _trim(rect, _grow(o, g))
                if rect is None:
                    break
            if rect is None:
                continue
            x0, z0, x1, z1 = rect
            # **...and again after the standing parts have been trimmed out of it.** The
            # thin test above reads the bands as they were cut round the core; the hall
            # and the square are then taken out of them, and the farming village's west
            # band came out of that 28 deep -- under the 36 one row of its own 24x24
            # cottages needs. It was kept, the allocator promised it two of the sixteen,
            # and the compiler laid none.
            deep = min(x1 - x0 + 1, z1 - z0 + 1)
            if x1 - x0 + 1 < self.dmin or z1 - z0 + 1 < self.dmin or deep < depth_min:
                if deep < depth_min <= max(x1 - x0 + 1, z1 - z0 + 1):
                    self._thin_for_fabric = {
                        **(getattr(self, "_thin_for_fabric", None) or
                           {"depth_min": int(depth_min), "dmin": int(self.dmin)}),
                        "sides": sorted(set((getattr(self, "_thin_for_fabric", None)
                                             or {}).get("sides") or []) | {sname}),
                        "why": (f"a band holding one row of this request's own fabric "
                                f"is {depth_min} deep; once the parts standing in it "
                                f"were trimmed out, `{sname}` was {deep}, so it holds "
                                f"no house of this demand and is not asked for one")}
                continue
            along_x = (x1 - x0) >= (z1 - z0)
            length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
            pieces = max(1, int(math.ceil(length / float(SECTOR_MAX))))
            while pieces > 1 and (length - g * (pieces - 1)) // pieces < self.dmin:
                pieces -= 1
            plen = (length - g * (pieces - 1)) // pieces
            for i in range(pieces):
                start = (x0 if along_x else z0) + i * (plen + g)
                end = start + plen - 1 if i < pieces - 1 else (x1 if along_x else z1)
                r = ((start, z0, end, z1) if along_x else (x0, start, x1, end))
                label = sname if pieces == 1 else f"{sname}_{i + 1}"
                sectors.append((label, r))
        if not sectors:
            self.fails.append({"part": groups[0]["name"], "check": "district",
                               "why": f"no ground {self.dmin} on a side is left inside "
                                      f"the place for its districts"})
            return
        # which group each sector answers: `quarter` parts take one sector each in
        # order, `throughout` parts share the rest; an `along` group takes the strips
        # beside its edge
        assign: dict = {}
        remaining = list(sectors)
        # **A group the sentence gathers *around* the centre takes at least three sides
        # of it** (the closure round). The strips are one per side; where fewer than
        # three sides are left after the `quarter` groups have taken theirs, the largest
        # strip is split and the quarter takes one piece, so the gathered group keeps
        # every side. Where the ground itself leaves fewer than three sides, the record
        # says so and the checker's finding names the layout, which is right.
        gathered = [r for r in self.relations
                    if r["relation"] == "around" and r.get("object_part")
                    and r["object_part"] == (self.core or {}).get("name")
                    and r.get("subject_groups")]
        reserved: set = set()
        # the land quarters the sentence puts beside the gathered houses take a piece of
        # a gathered strip below (`adjoining`), so they need no side of their own
        subj_all = {n for r in self.relations if r["relation"] == "around"
                    for n in (r.get("subject_groups") or [])}
        adjoin_names = {p["name"] for p in groups
                        if p["relation"] == "quarter"
                        and entity_of(p, self.spec).get("class") == "land"
                        and any(r["relation"] in ("beside", "near")
                                and p["name"] in (r.get("subject_groups") or [])
                                and (set(r.get("object_groups") or []) & subj_all)
                                for r in self.relations)}
        if gathered:
            subj = {n for r in gathered for n in r["subject_groups"]}
            quarters_n = sum(int(p["count"]) for p in groups
                             if p["relation"] == "quarter" and p["name"] not in subj
                             and p["name"] not in adjoin_names)
            sides = {}
            for label, r in sectors:
                sides.setdefault(label.split("_")[0], []).append((label, r))
            if quarters_n and len(sides) - quarters_n < 3:
                # split the largest strip along its long axis so a quarter can take a
                # piece and the gathered group still stands on that side
                biggest = max(sectors, key=lambda s: (s[1][2] - s[1][0] + 1)
                              * (s[1][3] - s[1][1] + 1))
                label, (x0, z0, x1, z1) = biggest
                along_x = (x1 - x0) >= (z1 - z0)
                length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
                if (length - g) // 2 >= self.dmin:
                    plen = (length - g) // 2
                    a = (x0 if along_x else z0)
                    pieces = [((a, z0, a + plen - 1, z1) if along_x
                               else (x0, a, x1, a + plen - 1)),
                              ((a + plen + g, z0, x1, z1) if along_x
                               else (x0, a + plen + g, x1, z1))]
                    sectors = [sct for sct in sectors if sct[0] != label] + [
                        (f"{label}_1", pieces[0]), (f"{label}_2", pieces[1])]
                    remaining = list(sectors)
            # the gathered group keeps one sector on every side it can; quarters take
            # the rest, smallest side first, so the gathered group loses the least
            by_side = {}
            for label, r in sectors:
                by_side.setdefault(label.split("_")[0], []).append((label, r))
            for side_name, rows_here in by_side.items():
                keep = max(rows_here, key=lambda s: (s[1][2] - s[1][0] + 1)
                           * (s[1][3] - s[1][1] + 1))
                reserved.add(keep[0])
            self.relation_record = {"gathered": sorted(subj),
                                    "about": (self.core or {}).get("name"),
                                    "sides": sorted(by_side),
                                    "reserved": sorted(reserved)}
        else:
            # **Nothing gathered, and the record says why.** A relation whose subject or
            # object this spec's parts cannot name is not a place laid out without it in
            # silence: the layout states what it could not resolve, and the checker's
            # own clause for that relation then reads against a record rather than
            # against an absence.
            self.relation_record = {
                "gathered": [], "about": (self.core or {}).get("name"),
                "unresolved": [
                    {"id": r.get("id"), "relation": r.get("relation"),
                     "subject": r.get("subject"), "object": r.get("object"),
                     "why": ("this spec has no defining part the word "
                             + ("`%s` names" % r.get("subject")
                                if not r.get("subject_groups") else
                                "`%s` names" % r.get("object")))}
                    for r in self.relations
                    if r["relation"] == "around"
                    and (not r.get("subject_groups")
                         or not (r.get("object_part") or r.get("object_groups")))],
                "why": "no group of this place is gathered about its centre"}
        for p in groups:
            if p["relation"] == "along":
                target = self.by_name.get(p.get("of") or (self.wall or {}).get("name"))
                if target is not None:
                    tr = pipeline.part_rect({**target, "name": target.get("name")})
                    mine = [s for s in remaining
                            if _overlaps(_grow(s[1], 2 * g), tr)]
                    for s in mine[:max(1, int(p["count"]))]:
                        assign[s[0]] = p
                        remaining.remove(s)
        # **A land quarter the sentence puts beside the houses adjoins them** (the
        # expression round's orchard): the sector it takes is a piece of a strip a
        # gathered group holds, split along the strip, so `_compact` draws it beside the
        # gathered row on that side rather than in a strip of its own across the place.
        adjoining: dict = {}
        by_side = {}
        for label, r in sectors:
            by_side.setdefault(label.split("_")[0], []).append((label, r))
        for p in groups:
            if p["relation"] != "quarter" or entity_of(p, self.spec).get("class") != "land":
                continue
            # `subj_all`, not `subj`: the latter is bound only where this place has a
            # gathered group about its own core, and this loop runs whether or not it
            # has one. Found by running the held-out village, where the `around`
            # relation's subject resolved to nothing and this raised `UnboundLocalError`
            # instead of laying a place without gathering. `subj_all` is also the set
            # `adjoin_names` above tests against, so the two now ask one question.
            bound = [r for r in self.relations
                     if r["relation"] in ("beside", "near")
                     and p["name"] in (r.get("subject_groups") or [])
                     and (set(r.get("object_groups") or []) & subj_all)]
            if not bound:
                continue
            host = None
            for label, r in sorted([sct for sct in sectors if sct[0] in reserved
                                    and assign.get(sct[0]) is None],
                                   key=lambda sct: -((sct[1][2] - sct[1][0] + 1)
                                                     * (sct[1][3] - sct[1][1] + 1))):
                x0, z0, x1, z1 = r
                along_x = (x1 - x0) >= (z1 - z0)
                length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
                if (length - g) // 2 >= self.dmin:
                    host = (label, r, along_x, length)
                    break
            if host is None:
                continue
            label, (x0, z0, x1, z1), along_x, length = host
            # the gathered group keeps a piece centred on the core -- so its row reaches
            # both sides of the thing it gathers about -- and the land takes the larger
            # remainder; a remainder too small for a district is the houses' own
            lo_e, hi_e = (x0 if along_x else z0), (x1 if along_x else z1)
            mid = ((cx0 + cx1) / 2.0) if along_x else ((cz0 + cz1) / 2.0)
            plen = max(self.dmin, (length - g) // 2)
            h0 = int(round(mid - plen / 2.0))
            h0 = max(lo_e, min(h0, hi_e - plen + 1))
            h1 = h0 + plen - 1
            left, right = (h0 - g) - lo_e + 1, hi_e - (h1 + g) + 1
            if left < self.dmin and right < self.dmin:
                continue
            if left >= right:
                q0, q1 = lo_e, h0 - g - 1
                if right < self.dmin:
                    h1 = hi_e
            else:
                q0, q1 = h1 + g + 1, hi_e
                if left < self.dmin:
                    h0 = lo_e
            first = (h0, z0, h1, z1) if along_x else (x0, h0, x1, h1)
            second = (q0, z0, q1, z1) if along_x else (x0, q0, x1, q1)
            strip_of = getattr(self, "_strip_of", {})
            strip_of[f"{label}_1"] = strip_of[f"{label}_2"] = (x0, z0, x1, z1)
            self._strip_of = strip_of
            sectors = [sct for sct in sectors if sct[0] != label] + [
                (f"{label}_1", first), (f"{label}_2", second)]
            remaining = [sct for sct in remaining if sct[0] != label] + [
                (f"{label}_1", first)]
            reserved.discard(label)
            reserved.add(f"{label}_1")
            assign[f"{label}_2"] = p
            adjoining[p["name"]] = {"host": f"{label}_1", "sector": f"{label}_2",
                                    "relation": bound[0].get("id")}
        if adjoining:
            self.relation_record = {**(getattr(self, "relation_record", None) or {}),
                                    "adjoining": adjoining}
        for p in groups:
            if p["relation"] == "quarter":
                if p["name"] in adjoining:
                    continue
                for _k in range(int(p["count"])):
                    free = [q for q in remaining if q[0] not in reserved] or remaining
                    if free:
                        # the smallest free sector, so a gathered group keeps the most
                        pick = min(free, key=lambda q: (q[1][2] - q[1][0] + 1)
                                   * (q[1][3] - q[1][1] + 1)) if reserved else free[0]
                        remaining.remove(pick)
                        assign[pick[0]] = p
        through = [p for p in groups if p["relation"] == "throughout"]
        for i, s in enumerate(remaining):
            if through:
                assign[s[0]] = through[i % len(through)]
        # the counts: from the ground at each district's density, capped by the room it
        # has, and scaled to what the spec declared where the ground allows
        rows = []
        standing = {"parts": [dict(p) for p in self.placed], "arterials": {"cells": []}}
        for label, r in sectors:
            p = assign.get(label)
            if p is None:
                continue
            area = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
            per = spec_mod.columns_per_plot(p)
            cap = int(area * DISTRICT_FILL // per)
            # **How many houses this sector holds at its word**, from the one density
            # definition (`placeplan.count_band`): the band's floor, middle and ceiling
            # over the ground the sector can develop, at the fabric's own lot. The
            # ceiling is the most a district of that word may be asked for; the middle
            # is what it is asked for where the spec gives no number.
            band = placeplan.count_band({"name": label, "x0": r[0], "z0": r[1],
                                         "x1": r[2], "z1": r[3]}, p, standing,
                                        self.decls)
            rows.append({"label": label, "rect": r, "part": p, "area": area, "per": per,
                         "cap": cap, "band": band,
                         "from_ground": max(1, min(cap, band["mid"])),
                         "most": max(1, min(cap, band["hi"]))})
        # **The count is spent only on counted buildings** (the expression round). The
        # falsification pass's orchard: a land district is a group like any other, the
        # apportionment gives every district at least one, and a house was laid on the
        # orchard. `entity_of` says which groups hold counted buildings; a land district
        # the programme puts no houses in is asked for none, keeps its sector, and lays
        # its land.
        counted = {p["name"]: bool(entity_of(p, self.spec).get("counted")) for p in groups}
        land = {p["name"] for p in groups if entity_of(p, self.spec).get("class") == "land"}
        few = max(1, int(math.ceil(self._households() * LAND_HOUSE_SHARE)))
        for row in rows:
            if not counted.get(row["part"]["name"], True):
                row["from_ground"], row["most"], row["cap"] = 0, 0, 0
                row["uncounted"] = True
            elif row["part"]["name"] in land and self.exact:
                # a land district holds the few houses that work it, not what its ground
                # would fit at its word (`LAND_HOUSE_SHARE`) -- of a count the sentence
                # stated; an inferred count is the ground's to fill, as before
                row["from_ground"] = min(row["from_ground"], few)
                row["most"] = min(row["most"], few)
                row["few"] = few
        declared = int(self.spec.get("structures") or 0)
        want = declared or sum(row["from_ground"] for row in rows)
        # **Each group's own share of the count, then each sector's share of its
        # group's.** The spec apportions the place's structures over its groups
        # (`spec.apportion`); spreading the total over every sector by area threw that
        # away and put most of a village's cottages in its fields. A group's sectors
        # share its number by the middle of their bands, capped at their ceilings; what
        # the ceilings cannot hold is `short`, reported and never quietly laid denser.
        counts = [0] * len(rows)
        short_total = 0
        by_group: dict = {}
        for i, row in enumerate(rows):
            by_group.setdefault(row["part"]["name"], []).append(i)
        group_want = {}
        if declared:
            said = {p["name"]: int(p.get("structures") or 0) for p in groups}
            # **An even split is the apportionment's default and not a declaration.**
            # `spec.apportion` gives every district the same share where the spec
            # declared none, and a spec cannot say which of its numbers it meant. An
            # all-equal split over groups whose ground holds very different numbers is
            # read as inferred and re-split by what each holds at its own word; an
            # uneven split is the spec's and is kept. (Adapter request: `apportion`
            # should mark the shares it infers, so this needs no inference.)
            even = len(by_group) > 1 and len({said.get(n, 0) for n in by_group}) == 1
            # the coordinator's `apportion` now marks the shares it inferred; where the
            # marker is on the parts it decides, and the even-split inference is only
            # for a spec read before the marker existed
            marked = [p for p in groups if "structures_inferred" in p]
            if marked:
                even = any(p.get("structures_inferred") for p in marked)
            if sum(said.get(n, 0) for n in by_group) == declared and not even:
                group_want = {n: (said[n] if counted.get(n, True) else 0)
                              for n in by_group}
                lost = declared - sum(group_want.values())
                if lost > 0:
                    # a share the spec gave a district of land goes to the districts
                    # that hold counted buildings, by what each holds at its word
                    homes = [n for n in by_group if counted.get(n, True)]
                    if homes:
                        more = _largest_remainder(
                            [sum(rows[i]["most"] for i in by_group[n]) for n in homes],
                            lost, floor=0)
                        for n, c in zip(homes, more):
                            group_want[n] += int(c)
            else:
                # by what each group's ground holds at its word -- between the middle
                # and the ceiling of its bands, so a sparse quarter of fields keeps the
                # farm cottages that work it and a low quarter of homes takes the rest
                mids = {n: sum((rows[i]["from_ground"] + rows[i]["most"]) / 2.0
                               for i in ix)
                        for n, ix in by_group.items()}
                got = _largest_remainder([mids[n] for n in by_group], declared,
                                         caps=[sum(rows[i]["most"] for i in ix)
                                               for ix in by_group.values()])
                group_want = dict(zip(by_group, got))
        else:
            group_want = {n: sum(rows[i]["from_ground"] for i in ix)
                          for n, ix in by_group.items()}
        for n, ix in by_group.items():
            got = _largest_remainder([rows[i]["from_ground"] for i in ix],
                                     int(group_want[n]),
                                     caps=[rows[i]["most"] for i in ix])
            for i, c in zip(ix, got):
                counts[i] = int(c)
            short_total += max(0, int(group_want[n]) - sum(got))
        # **What one group's ground cannot hold goes to a group that can.** The split of
        # the count between the groups is the spec's *inferred* share where it declared
        # none (`spec.apportion` splits evenly), and a farmland quarter that holds two
        # sparse cottages was handed eight. The count is the sentence's; where the
        # cottages stand between the groups is the layout's, and only what no group can
        # hold is short.
        declared_split = bool(declared and any(int(p.get("structures") or 0)
                                               for p in groups)
                              and self.spec.get("explicit_count") is None)
        if short_total and not declared_split:
            room = [max(0, rows[i]["most"] - counts[i]) for i in range(len(rows))]
            if sum(room):
                more = _largest_remainder([rows[i]["from_ground"] for i in range(len(rows))],
                                          int(short_total), caps=room, floor=0)
                for i, c in enumerate(more):
                    counts[i] += int(c)
                short_total = max(0, short_total - sum(more))
        # **An exact count is laid whole** (see `placeshore`): what no district's
        # ceiling holds at its first rectangle goes on the districts by area, over the
        # ceiling, for the layout owner's extent action rather than as a shortfall of a
        # number the sentence stated.
        if self.exact and short_total and rows:
            more = _largest_remainder([0 if rows[i].get("uncounted") else rows[i]["area"]
                                       for i in range(len(rows))],
                                      int(short_total), floor=0)
            for i, c in enumerate(more):
                counts[i] += int(c)
            self.count_over_ceiling = True
            short_total = 0
        self.count_short = int(short_total)
        # **A sector the place cannot ask for `DISTRICT_MIN_STRUCTURES` in is not a
        # district.** v2, C5: a district is a division of the place that holds houses,
        # and the strips cut round a centre leave slivers -- a 35x46 corner beside a
        # village's hard, asked for one house -- which are then held to the same count
        # and cover as a quarter of a city and can meet neither: a sliver is lanes and
        # clearances almost all the way through. Left-over ground is what it is, and
        # nothing has ever asked for every column of a place to be in a district.
        # ...**but a `quarter` the spec named is a district at any count** (the closure
        # round): a quarter of fields is fields, and dropping it as a sliver because it
        # holds one farm cottage made the place lose the part the sentence named.
        if len(rows) > 1 and any(n >= DISTRICT_MIN_STRUCTURES for n in counts):
            small = [(row, n) for row, n in zip(rows, counts)
                     if n < DISTRICT_MIN_STRUCTURES
                     and row["part"].get("relation") != "quarter"
                     and not row.get("uncounted")]
            if small:
                self.left_over = [{"label": row["label"], "rect": list(row["rect"]),
                                   "area": row["area"], "structures": int(n),
                                   "why": f"under {DISTRICT_MIN_STRUCTURES} structures"}
                                  for row, n in small]
                kept = [(row, n) for row, n in zip(rows, counts)
                        if n >= DISTRICT_MIN_STRUCTURES
                        or row["part"].get("relation") == "quarter"
                        or row.get("uncounted")]
                rows = [row for row, _n in kept]
                # the houses a dropped sliver was to hold go to its own group's sectors
                lost: dict = {}
                for row, n in small:
                    lost[row["part"]["name"]] = lost.get(row["part"]["name"], 0) + n
                counts = [n for _row, n in kept]
                for gname, extra in lost.items():
                    ix = [i for i, row in enumerate(rows) if row["part"]["name"] == gname]
                    if not ix:
                        self.count_short += extra
                        continue
                    more = _largest_remainder(
                        [rows[i]["from_ground"] for i in ix], extra,
                        caps=[max(0, rows[i]["most"] - counts[i]) for i in ix],
                        floor=0)
                    for i, c in zip(ix, more):
                        counts[i] += int(c)
                    self.count_short += max(0, extra - sum(more))
        # **A gathered group's district is drawn to what its share of the count needs,
        # not to the strip it stands in** (the closure round's revision loop). Each
        # `around` district took the whole strip on its side of the square, so the
        # compiler spread five cottages over nine thousand columns, forty columns of
        # verge between them, and the judge read the village as scattered. The rectangle
        # now hugs the anchor's box on its side -- the strip's inner edge -- with the
        # depth and length its count needs at the density's target cover (the band's
        # middle) at the lot the compiler will lay, plus lanes; what is left of the
        # strip is unclaimed ground and the record says so. A `quarter` named beside the
        # gathered group on the same side (the fields) is sized to its own count at its
        # own word and drawn adjoining the gathered district, so `beside` holds on the
        # geometry. Only where the count is the sentence's own: an inferred count is the
        # ground's to fill. **...or where the layout owner's `regroup` action asked for
        # it.** The composition round's fifth change. Compaction was gated on the count
        # being the *sentence's* (`self.exact`) -- an inferred count "is the ground's to
        # fill" -- and that is right as a default and wrong as the only possibility: a
        # built reading that says the fabric is scattered is a finding, and the answer
        # to it is to gather the fabric, whatever kind of count asked for the houses.
        # `regroup` is that answer as a bounded allocation (`reallocate`), so the
        # decision is recorded where every other inferred dimension is and a place laid
        # out again from the spec is this place. It changes no count: `_compact` moves
        # rectangles and `_straddle` splits one in two with its count split by length.
        regroup = bool((self.allocation.get("regroup") or {}).get("gather"))
        if gathered and (self.exact or regroup):
            self._compact(rows, counts, (cx0, cz0, cx1, cz1), g)
            self._straddle(rows, counts, (cx0, cz0, cx1, cz1), g)
            if regroup and not self.exact:
                self.targets.append(
                    {"what": "regroup", "value": {"gathered": sorted(
                        {n for r in self.relations if r["relation"] == "around"
                         for n in (r.get("subject_groups") or [])})},
                     "from": ["allocation.regroup.gather: the built reading found the "
                              "fabric scattered over its strips, so the districts are "
                              "drawn to what their count needs and hugged to the anchor "
                              "and its streets rather than spread over the whole strip"]})
        # **Every district records what it was sized from** (the expression round),
        # whether or not it was compacted: the need its count and land use imply, by the
        # one sizing rule, with every input by name.
        for row, n in zip(rows, counts):
            if "need" not in row:
                row["need"] = land_need(row["part"], n, self.decls, self.spec,
                                        demand=row["part"].get("demand"),
                                        households=self._households(),
                                        allocation=self.allocation)
        role_of = {}
        for row, n in zip(rows, counts):
            p = row["part"]
            r = row["rect"]
            role = p.get("role") or spec_mod.read_role(None, p, p["name"])
            role_of[p["name"]] = role
            need = row["need"]
            target = {"what": need["what"], "value": {"columns": need["columns"],
                                                       "lot": need["lot"],
                                                       "cover": need["target_cover"],
                                                       "houses": int(n),
                                                       "got_columns": int(row["area"])},
                      "from": list(need["from"])}
            if need.get("what") == "land":
                target["value"].update({"land_use": need.get("land_use"),
                                        "land_columns": need.get("land_columns")})
            if need.get("short_columns"):
                target["short_columns"] = int(need["short_columns"])
                target["why_short"] = need.get("why_short")
            self.districts.append({
                "name": f"{p['name']}_{row['label']}", "x0": r[0], "z0": r[1],
                "x1": r[2], "z1": r[3],
                "structures": (int(min(n, row["cap"])) if row.get("uncounted")
                               else int(max(1, min(n, row["cap"])))),
                "target": target,
                # the least lot the compiler lays, where an allocation or the envelope
                # raised it above the fabric's own
                **({"lot_min": list(need["lot"])} if lot_raised(need.get("from"))
                   else {}),
                **({"surface": "open", "land_use": need.get("land_use")}
                   if row.get("uncounted") else {}),
                # the four columns of this region, fixed at resolution and not revised
                # by any later allocation (`placeregion.column_record`); a strip the
                # count did not reach is still this part's ground and is still inside
                # what its density is measured over
                **placeregion.column_record(
                    (r[0], r[1], r[2], r[3]),
                    scope_of=(p["name"] if row.get("uncounted") else None),
                    why=[f"the {row['label'].replace('_', ' ')} strip of `{p['name']}`, "
                         f"fixed when the place was solved"]),
                "defines": p["name"],
                # **the sentence's own number is laid exactly**: the compiler lays this
                # many and not the number its cover would like (`district_compile`)
                **({"exact": True} if self.exact else {}),
                "count_band": {k: row["band"][k] for k in ("lo", "mid", "hi", "usable")},
                **({"voice": p["voice"]} if p.get("voice") else {}),
                "purpose": (f"{p.get('notes') or p['name']} A {p.get('density') or 'medium'}"
                            f", {role} district of the place; solved."),
                "notes": (f"The {row['label'].replace('_', ' ')} strip of the ground "
                          f"round the centre, inside "
                          f"{'the wall' if self.wall else 'the site'}; solved, not drawn.")})
        covered = sum(row["area"] for row in rows)
        inside_cols = (bx1 - bx0 + 1) * (bz1 - bz0 + 1)
        self.district_record = {
            "bounds": [bx0, bz0, bx1, bz1], "core_box": [cx0, cz0, cx1, cz1],
            **({"thin_for_fabric": self._thin_for_fabric}
               if getattr(self, "_thin_for_fabric", None) else {}),
            "sectors": [{"label": row["label"], "rect": list(row["rect"]),
                         "group": row["part"]["name"],
                         **({"compacted": row["compacted"]} if row.get("compacted") else {}),
                         **({"straddle": row["straddle"]} if row.get("straddle") else {})}
                        for row in rows],
            "structures": {"declared": declared, "laid": sum(int(d["structures"])
                                                            for d in self.districts),
                           "caps": [row["cap"] for row in rows],
                           "bands": [row["band"] for row in rows],
                           "short": int(getattr(self, "count_short", 0)),
                           "laid_over_ceiling": bool(getattr(self, "count_over_ceiling", False)),
                           "exact": self.exact},
            "relations": self._relations_predicted(),
            "coverage": round(covered / float(inside_cols), 4) if inside_cols else 0.0,
            "left_over": self.left_over}

    def _fabric_depth(self, groups: list) -> int:
        """**The least depth a band of houses may be, from the fabric that stands in it**:
        one row of the lot the resolved demand asks for, its street, its edge margins
        and its clearance (`district_compile.district_depth`), over every counted group.
        Never under `dmin`, so a place whose fabric asks for less is laid exactly as it
        was.
        """
        from . import district_compile as dc
        want = int(self.dmin)
        for p in groups:
            if entity_of(p, self.spec).get("class") != "building":
                continue
            try:
                (_w, ld), _src = fabric_lot(p, self.decls, self.spec,
                                            allocation=self.allocation,
                                            demand=p.get("demand"), voice=self.voice)
            except (Refused, KeyError, ValueError):   # noqa: PERF203 -- the floor stands
                continue
            ch = p.get("character") or {}
            gap = placeplan.PLOT_LANE if ch.get("frontage") == "open" else dc.LOT_GAP
            want = max(want, dc.district_depth(int(ld), 1, gap))
        return int(want)

    def _compact(self, rows: list, counts: list, core_box: tuple, g: int) -> None:
        """Shrink each gathered (or adjoining quarter) row's rectangle to its count's
        need, hugging the core box; the remainder of the strip goes to `left_over`."""
        from . import district_compile as dc
        cx0, cz0, cx1, cz1 = core_box
        ax, az = (cx0 + cx1) / 2.0, (cz0 + cz1) / 2.0
        subj = {n for r in self.relations if r["relation"] == "around"
                for n in (r.get("subject_groups") or [])}
        placed_by_side: dict = {}

        def need_columns(row, part, n):
            """The one sizing rule (`land_need`), kept on the row for its `target`."""
            rec = land_need(part, n, self.decls, self.spec,
                            demand=part.get("demand"),
                            households=self._households(), allocation=self.allocation)
            row["need"] = rec
            return (rec["columns"], rec["target_cover"], rec["lot"],
                    (rec["min_len"], rec["min_dep"]))

        for row, n in zip(rows, counts):
            part = row["part"]
            side = row["label"].split("_")[0]
            if part["name"] not in subj or n <= 0:
                continue
            need, mid, lot, (min_len, min_dep) = need_columns(row, part, n)
            x0, z0, x1, z1 = row["rect"]
            row.setdefault("sector", list(row["rect"]))   # the strip, before compaction
            horiz = side in ("north", "south")          # the strip runs along x
            strip_len = (x1 - x0 + 1) if horiz else (z1 - z0 + 1)
            strip_dep = (z1 - z0 + 1) if horiz else (x1 - x0 + 1)
            dep = max(self.dmin, min_dep, min(strip_dep, int(math.ceil(math.sqrt(need)))))
            ln = max(self.dmin, min_len, min(strip_len, int(math.ceil(need / float(dep)))))
            # **and never longer or deeper than the strip it is cut from.** `min_len`
            # and `min_dep` are what a block of this fabric wants; they are a proposal,
            # and the strip is the ground there actually is. Found by running the
            # farming village whose two-storey cottages need 24-wide lots: a block of
            # two of them is 60 along the street, the west strip was 52, and this drew
            # the row 60 long -- eight columns into the south strip, which the place
            # validator refused as an overlap. A row that cannot hold a whole block of
            # its fabric is short, and `short_columns` below is where that is said; it
            # is not licence to draw over the neighbour.
            ln, dep = min(ln, strip_len), min(dep, strip_dep)
            while dep * ln < need and dep < strip_dep:
                dep = min(strip_dep, dep + 4)
                ln = max(self.dmin, min(strip_len, int(math.ceil(need / float(dep)))))
            if horiz:
                nz0, nz1 = ((cz1 + 1 + g, cz1 + g + dep) if side == "south"
                            else (cz0 - g - dep, cz0 - 1 - g))
                nz0, nz1 = max(z0, nz0), min(z1, nz1)
                if nz1 - nz0 + 1 < self.dmin:
                    nz0, nz1 = (z0, min(z1, z0 + dep - 1)) if side == "south" \
                        else (max(z0, z1 - dep + 1), z1)
                mx0 = int(round(ax - ln / 2.0))
                mx0 = max(x0, min(mx0, x1 - ln + 1))
                new = (mx0, nz0, mx0 + ln - 1, nz1)
            else:
                nx0, nx1 = ((cx1 + 1 + g, cx1 + g + dep) if side == "east"
                            else (cx0 - g - dep, cx0 - 1 - g))
                nx0, nx1 = max(x0, nx0), min(x1, nx1)
                if nx1 - nx0 + 1 < self.dmin:
                    nx0, nx1 = (x0, min(x1, x0 + dep - 1)) if side == "east" \
                        else (max(x0, x1 - dep + 1), x1)
                mz0 = int(round(az - ln / 2.0))
                mz0 = max(z0, min(mz0, z1 - ln + 1))
                new = (nx0, mz0, nx1, mz0 + ln - 1)
            if new != tuple(row["rect"]):
                row["compacted"] = {"from": list(row["rect"]), "need": need,
                                    "target_cover": round(mid, 3), "lot": lot, "count": n}
                self.left_over.append({"label": row["label"], "rect": list(row["rect"]),
                                       "kept": list(new), "structures": 0,
                                       "why": (f"the strip beyond what {n} house(s) of "
                                               f"{part['name']} need at {mid:.0%} lot "
                                               f"cover ({need} columns): unclaimed ground, "
                                               f"left so the houses gather about the "
                                               f"centre rather than spread over it")})
                row["rect"] = new
                row["area"] = (new[2] - new[0] + 1) * (new[3] - new[1] + 1)
            if row["area"] < need:
                row["need"]["short_columns"] = int(need - row["area"])
                row["need"]["why_short"] = (f"the {side} strip holds {row['area']} of the "
                                            f"{need} columns {n} house(s) of "
                                            f"{part['name']} need at their lot: the site "
                                            f"is the limit")
            placed_by_side.setdefault(side, []).append(row)
        # the adjoining quarter: sized to its own count, drawn beside the gathered
        # district on its side, on the side of it its own piece lay
        for row, n in zip(rows, counts):
            part = row["part"]
            side = row["label"].split("_")[0]
            if part["name"] in subj or part.get("relation") != "quarter" \
                    or side not in placed_by_side:
                continue
            need, mid, lot, (min_len, _md) = need_columns(row, part, n)
            x0, z0, x1, z1 = row["rect"]
            # **Beside the houses it is actually beside.** Found by running the farming
            # village on the ground its two-storey demand needs: a 344-column site cuts
            # the north strip into two gathered rows, this took
            # `placed_by_side[side][0]` whichever of them that was, drew the fields
            # against it, and clamped the result to the strip alone -- so the fields
            # landed on top of the *other* row of houses and the place validator refused
            # the plan. With one gathered row a side, which is every village in the
            # record until now, the first was the only one and the defect could not
            # bite.
            mx, mz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
            anchor_row = min(placed_by_side[side],
                             key=lambda r: (abs((r["rect"][0] + r["rect"][2]) / 2.0 - mx)
                                            + abs((r["rect"][1] + r["rect"][3]) / 2.0 - mz),
                                            r["label"]))
            a0, b0, a1, b1 = anchor_row["rect"]
            horiz = side in ("north", "south")
            dep = (b1 - b0 + 1) if horiz else (a1 - a0 + 1)
            ln = max(self.dmin, min_len, int(math.ceil(need / float(dep))))
            # a quarter of land the sentence puts beside the houses stands a lot's
            # clearance from their district, not a lane: `beside` is measured from the
            # land's own edge to the nearest house
            from . import district_compile as _dc
            g = (_dc.LOT_GAP if entity_of(part, self.spec).get("class") == "land"
                 else LANE_GAP)
            # the ground the quarter may take: its own piece and, where its piece was
            # cut from the gathered row's strip, the strip's free ground up to the row
            sx0, sz0, sx1, sz1 = getattr(self, "_strip_of", {}).get(row["label"], row["rect"])
            if horiz:
                after = (x0 + x1) / 2.0 >= (a0 + a1) / 2.0
                ln = min(ln, (sx1 - (a1 + g + 1) + 1) if after else ((a0 - g - 1) - sx0 + 1))
                if ln < self.dmin:
                    continue
                new = ((a1 + g + 1, b0, a1 + g + ln, b1) if after
                       else (a0 - g - ln, b0, a0 - g - 1, b1))
            else:
                after = (z0 + z1) / 2.0 >= (b0 + b1) / 2.0
                ln = min(ln, (sz1 - (b1 + g + 1) + 1) if after else ((b0 - g - 1) - sz0 + 1))
                if ln < self.dmin:
                    continue
                new = ((a0, b1 + g + 1, a1, b1 + g + ln) if after
                       else (a0, b0 - g - ln, a1, b0 - g - 1))
            new = (max(sx0, new[0]), max(sz0, new[1]), min(sx1, new[2]), min(sz1, new[3]))
            # ...and clear of every other district, not merely inside its own strip. The
            # quarter grows away from the district it adjoins, so the end that is free
            # is the far one; it is cut back until it touches nothing, and where the
            # free end cannot clear the obstacle the quarter stays where the tiling put
            # it and says nothing it cannot keep.
            others = [tuple(o["rect"]) for o in rows
                      if o is not row and o.get("rect")]
            new = _clear_of(new, others, horiz, free_high=after, gap=g)
            if new is None or new[2] - new[0] + 1 < self.dmin \
                    or new[3] - new[1] + 1 < self.dmin:
                continue
            if new != tuple(row["rect"]):
                row["compacted"] = {"from": list(row["rect"]), "need": need,
                                    "target_cover": round(mid, 3), "lot": lot,
                                    "count": n, "adjoins": anchor_row["label"]}
                self.left_over.append({"label": row["label"], "rect": list(row["rect"]),
                                       "kept": list(new), "structures": 0,
                                       "why": (f"the strip beyond what {n} house(s) of "
                                               f"{part['name']} need at {mid:.0%} lot "
                                               f"cover ({need} columns), drawn adjoining "
                                               f"{anchor_row['label']}: unclaimed ground")})
                row["rect"] = new
                row["area"] = (new[2] - new[0] + 1) * (new[3] - new[1] + 1)
            g = LANE_GAP
            if row["area"] < need:
                # the strip cannot hold what the purpose needs: recorded, for the layout
                # owner's `grow_land` and for the reader, never hidden
                row["need"]["short_columns"] = int(need - row["area"])
                row["need"]["why_short"] = (f"the {side} strip beside {anchor_row['label']} "
                                            f"holds {row['area']} of the {need} columns "
                                            f"{part['name']} needs")

    def _free_span(self, row: dict, rows: list, horiz: bool, s_lo: int,
                   s_hi: int) -> tuple:
        """`(lo, hi)` of the strip this row may move along: its own sector, narrowed
        clear of every other district that shares its band. A district may abut its
        neighbour -- the validator's rule is overlap, and the ring's sectors tile edge
        to edge -- but a lot on either edge keeps `LOT_GAP`, so that is what is left
        between them."""
        from . import district_compile as dc
        x0, z0, x1, z1 = row["rect"]
        cur_lo, cur_hi = (x0, x1) if horiz else (z0, z1)
        lo, hi = int(s_lo), int(s_hi)
        for o in rows:
            if o is row or not o.get("rect"):
                continue
            ox0, oz0, ox1, oz1 = o["rect"]
            if horiz:
                if oz1 < z0 or oz0 > z1:          # not in this row's band
                    continue
                a, b = ox0, ox1
            else:
                if ox1 < x0 or ox0 > x1:
                    continue
                a, b = oz0, oz1
            if b < cur_lo:
                lo = max(lo, b + 1 + dc.LOT_GAP)
            elif a > cur_hi:
                hi = min(hi, a - 1 - dc.LOT_GAP)
        return (lo, hi) if lo <= hi else (int(cur_lo), int(cur_hi))

    def _straddle(self, rows: list, counts: list, core_box: tuple, g: int) -> None:
        """**A gathered row stands on both sides of the axis it crosses, or wholly on
        the side that adds a quadrant** (the expression round). The checker reads
        `around` as bearings from the anchor's centre, and a row centred on that axis
        promises both quadrants while the compiler fills its blocks from one end: the
        orchard hamlet's eleven houses all landed north-west of the chapel. A row long
        enough is cut at the axis into a district each side, its count split by
        length; a short row is moved to the side of the axis whose quadrant no row has
        yet, and the record says so."""
        cx0, cz0, cx1, cz1 = core_box
        ax, az = (cx0 + cx1) / 2.0, (cz0 + cz1) / 2.0
        subj = {n for r in self.relations if r["relation"] == "around"
                for n in (r.get("subject_groups") or [])}
        if not subj:
            return
        covered: set = set()
        order = sorted(range(len(rows)), key=lambda i: -(
            (rows[i]["rect"][2] - rows[i]["rect"][0] + 1)
            * (rows[i]["rect"][3] - rows[i]["rect"][1] + 1)))
        out: dict = {}

        def quadrant(rect):
            mx, mz = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
            return (int(mx >= ax), int(mz >= az))

        for i in order:
            row, n = rows[i], counts[i]
            part = row["part"]
            side = row["label"].split("_")[0]
            if part["name"] not in subj or n <= 0:
                out[i] = [(row, n)]
                continue
            x0, z0, x1, z1 = row["rect"]
            horiz = side in ("north", "south")
            lo, hi = (x0, x1) if horiz else (z0, z1)
            mid = ax if horiz else az
            cut = int(math.floor(mid))
            a_len = (cut - g // 2) - lo + 1
            b_len = hi - (cut + g - g // 2) + 1
            if n >= 2 and a_len >= self.dmin and b_len >= self.dmin:
                ra = (lo, z0, cut - g // 2, z1) if horiz else (x0, lo, x1, cut - g // 2)
                rb = ((cut + g - g // 2, z0, hi, z1) if horiz
                      else (x0, cut + g - g // 2, x1, hi))
                na, nb = _largest_remainder([a_len, b_len], int(n), floor=1)
                pieces = []
                for tag, rr, nn in (("a", ra, na), ("b", rb, nb)):
                    if nn <= 0:
                        continue
                    r2 = dict(row)
                    r2["label"] = f"{row['label']}_{tag}"
                    r2["rect"] = rr
                    r2["area"] = (rr[2] - rr[0] + 1) * (rr[3] - rr[1] + 1)
                    r2["need"] = land_need(part, int(nn), self.decls, self.spec,
                                           demand=part.get("demand"),
                                           households=self._households(),
                                           allocation=self.allocation)
                    r2["straddle"] = {"from": list(row["rect"]), "cut_at": cut,
                                      "why": (f"the row crossed the anchor's axis; a "
                                              f"district each side so the houses stand "
                                              f"on both, {nn} of {n} here")}
                    covered.add(quadrant(rr))
                    pieces.append((r2, int(nn)))
                out[i] = pieces
                continue
            ln = hi - lo + 1
            q = quadrant(row["rect"])
            straddles = lo < mid < hi
            if straddles:
                # wholly on the side whose quadrant is not yet covered; the lower side
                # where neither is. The row keeps what its count needs at its word
                # (`need`), within the strip it was cut from, so moving it off the axis
                # does not put it over its density's ceiling
                q_lo = (int(lo >= ax), int(z0 >= az)) if horiz else (int(x0 >= ax), int(lo >= az))
                q_hi = (int(hi >= ax), int(z0 >= az)) if horiz else (int(x0 >= ax), int(hi >= az))
                want_hi = q_lo in covered and q_hi not in covered
                sec = row.get("sector") or list(row["rect"])
                s_lo, s_hi = (sec[0], sec[2]) if horiz else (sec[1], sec[3])
                # **...and only over ground no other district owns.** `sector` is the
                # whole strip this row was cut from, and by the time this runs
                # `_compact` has given part of that strip to the quarter drawn beside it
                # -- the fields the sentence puts beside the houses. Moving the row off
                # the axis inside its old strip therefore moved it onto them: found by
                # running the farming village on the ground its two-storey demand needs,
                # where `homes_north_1_1` shifted fourteen columns west onto
                # `fields_north_1_2` and the place validator refused the plan. The span
                # this may move over is the strip less what the neighbours hold.
                s_lo, s_hi = self._free_span(row, rows, horiz, s_lo, s_hi)
                dep = (z1 - z0 + 1) if horiz else (x1 - x0 + 1)
                need_cols = int((row.get("need") or {}).get("columns") or 0)
                want_len = max(ln, int(math.ceil(need_cols / float(max(dep, 1)))))
                # **The row is moved, never shortened past what a block of it needs.**
                # `dmin` is a constant about what a district is; the length one block of
                # *this* fabric takes is `need.min_len` -- two edge margins, a street,
                # and the lots of a block side by side. Found by running the farming
                # village whose two-storey cottages need 24-wide lots: the west row was
                # moved off the axis and cut to 28 along its street, where a block of
                # two of them is 60, so the compiler laid none of the two the allocator
                # had promised it and the place came to fourteen of sixteen. Where the
                # strip cannot give the row that length on one side of the axis, the row
                # stays where it is: standing across the axis is a recorded compromise,
                # and standing in a band no house fits is not.
                floor = min(max(int(self.dmin),
                                int((row.get("need") or {}).get("min_len") or 0)),
                            int(s_hi) - int(s_lo) + 1)
                if want_hi:
                    n_lo, n_hi = cut + 1, min(s_hi, cut + want_len)
                    if n_hi - n_lo + 1 < floor:
                        n_lo = max(s_lo, n_hi - floor + 1)
                else:
                    n_hi, n_lo = cut - 1, max(s_lo, cut - want_len)
                    if n_hi - n_lo + 1 < floor:
                        n_hi = min(s_hi, n_lo + floor - 1)
                if n_hi - n_lo + 1 >= floor:
                    # ...and where the strip's end caps the row short of its need, the
                    # row deepens away from the core, within its strip
                    have = (n_hi - n_lo + 1) * dep
                    ax0, az0, ax1, az1 = x0, z0, x1, z1
                    if need_cols and have < need_cols:
                        extra = int(math.ceil((need_cols - have) / float(n_hi - n_lo + 1)))
                        if horiz:
                            if side == "north":
                                az0 = max(sec[1], z0 - extra)
                            else:
                                az1 = min(sec[3], z1 + extra)
                        else:
                            if side == "west":
                                ax0 = max(sec[0], x0 - extra)
                            else:
                                ax1 = min(sec[2], x1 + extra)
                    rr = (n_lo, az0, n_hi, az1) if horiz else (ax0, n_lo, ax1, n_hi)
                    row["straddle"] = {"from": list(row["rect"]), "cut_at": cut,
                                       "why": (f"the row crossed the anchor's axis and is "
                                               f"too short to stand on both sides; moved "
                                               f"wholly to the side whose quadrant no row "
                                               f"had")}
                    row["rect"] = rr
                    row["area"] = (rr[2] - rr[0] + 1) * (rr[3] - rr[1] + 1)
                    q = quadrant(rr)
            covered.add(q)
            out[i] = [(row, n)]
        new_rows, new_counts = [], []
        for i in range(len(rows)):
            for r2, nn in out.get(i, [(rows[i], counts[i])]):
                new_rows.append(r2)
                new_counts.append(nn)
        rows[:] = new_rows
        counts[:] = new_counts

    def _relations_predicted(self) -> list:
        """Every resolved relation, scored on the districts as laid with the checker's
        own rule (`intent.relation_measure`) over pseudo-leaves -- the corners and the
        middle of each subject district, and the object part -- so the layout says what
        the checker will find before a lot is compiled. A prediction and not a pass."""
        out = []
        try:
            from . import intent as intent_mod
        except Exception:                        # noqa: BLE001 -- no prediction
            return out
        fn = getattr(intent_mod, "relation_measure", None)
        for rel in self.relations:
            row = {k: rel.get(k) for k in ("id", "relation", "subject", "object",
                                           "object_part", "subject_groups")}
            if fn is None or not (rel.get("object_part") or rel.get("object_groups")):
                row["predicted"] = "unresolved"
                out.append(row)
                continue
            pseudo = []
            for d in self.districts:
                if d.get("defines") not in ((rel.get("subject_groups") or [])
                                            + (rel.get("object_groups") or [])):
                    continue
                x0, z0, x1, z1 = d["x0"], d["z0"], d["x1"], d["z1"]
                part = next((p for p in self.spec["defining_parts"]
                             if p["name"] == d.get("defines")), {})
                # a land district's probes are its land (a field, an orchard, a
                # pasture), a settled district's its houses (the expression round)
                ent = entity_of(part, self.spec)
                use = ent.get("land_use") if ent.get("class") == "land" else None
                land = use if use in ("farmland", "orchard", "pasture") else None
                fam = {"farmland": "field", "orchard": "orchard", "pasture": "pasture"}.get(land)
                # **the probes stand where the leaves will** (the expression round): a
                # house lot keeps the compiler's edge margin and a lot's clearance from
                # its district's edge, an open block of land its margin only, so a
                # `beside` predicted here is the gap the checker measures on the
                # assembled plan and not the gap between two rectangles of ground
                from . import district_compile as _dc
                inset = _dc.EDGE_MARGIN if land else _dc.EDGE_MARGIN + _dc.LOT_GAP
                if d.get("surface") == "open":
                    inset = 0
                if x1 - x0 > 2 * inset + 2 and z1 - z0 > 2 * inset + 2:
                    x0, z0, x1, z1 = x0 + inset, z0 + inset, x1 - inset, z1 - inset
                for (x, z) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1),
                               ((x0 + x1) // 2, (z0 + z1) // 2)):
                    pseudo.append({"name": f"{d['name']}_probe",
                                   "kind": "area" if land else "plot",
                                   "family": fam if land else "house",
                                   "type": fam if land else "cottage",
                                   "in": [d["name"], d.get("defines") or ""],
                                   "defines": d.get("defines"),
                                   **({"land_use": use} if use else {}),
                                   "x0": x, "z0": z, "x1": x + 1, "z1": z + 1})
            for leaf in self.placed:
                if any(_answers(leaf, {"name": n}) for n in
                       [rel["object_part"], *(rel.get("subject_parts") or [])] if n):
                    r = pipeline.part_rect({**leaf, "name": leaf.get("name")})
                    # with the family of the defining part it answers, which is the word
                    # the checker selects it by on the assembled plan
                    d0 = next((p for p in self.spec["defining_parts"]
                               if _answers(leaf, p)), None)
                    pseudo.append({**leaf, "x0": r[0], "z0": r[1], "x1": r[2],
                                   "z1": r[3],
                                   **({"family": d0["family"], "answers": d0["name"]}
                                      if d0 else {})})
            try:
                status, why, ev = fn({"subject": rel["subject"],
                                      "relation": rel["relation"],
                                      "object": rel["object"]}, pseudo)
            except Exception as e:               # noqa: BLE001 -- reported
                status, why, ev = "unresolved", f"{type(e).__name__}: {e}", {}
            row.update(predicted=status, why=why, evidence=ev)
            out.append(row)
        return out

    # --- the document ----------------------------------------------------------------

    def _place_doc(self) -> dict:
        pal = None
        try:
            pal = (dict(styles.VOICES[self.voice]["palette"])
                   if self.voice in styles.VOICES else None)
        except Exception:                        # noqa: BLE001 -- no palette, no material
            pal = None
        core = self.core
        return {
            "intent": (f"A {self.spec['kind']} placed by relation: "
                       + "; ".join(f"{r['part']} {r['relation']}" for r in self.record)
                       + (f"; {len(self.districts)} district(s) round the centre"
                          if self.districts else "")
                       + ". Every part stands where its relation put it, inside the site"
                       + (" and inside the wall" if self.wall else "") + "."
                       + (f" {self.spec['invariants']}" if self.spec.get("invariants") else "")),
            "centre": core["name"] if core else None,
            "voice": self.voice, "palette": pal,
            "circulation_material": (pal or {}).get("floor") or "stone",
            "parts": [dict(p) for p in self.placed],
            "districts": list(self.districts),
            "compounds": [dict(c) for c in self.compounds],
            "layout": {"by": "placesolve.solve_place", "case": "relations",
                       "seed": self.seed, "centre": [self.cx, self.cz],
                       "plateau_rect": list(self.plateau_rect) if self.plateau_rect else None,
                       "wall": ({"name": self.wall["name"], "half": self.wall["half"],
                                 "rect": list(self.wall["rect"]), "inset": self.wall["inset"],
                                 "shape": self.wall["shape"]} if self.wall else None),
                       "solved": self.record, "demoted": self.demoted,
                       "districts": getattr(self, "district_record", None),
                       # **What every dimension was derived from** (the expression
                       # round): the anchor's and the civic parts' targets, and the
                       # allocation overrides in force, so a place re-solved from the
                       # spec is the same place and a reader can see why.
                       "targets": list(self.targets),
                       "allocation": dict(self.allocation),
                       "registered": {"VETOES": [list(v) for v in VETOES],
                                      "COSTS": dict(COSTS), "KEEP": KEEP,
                                      "IMPROVE_PASSES": IMPROVE_PASSES,
                                      "WALL_STEP_BLOCKS": WALL_STEP_BLOCKS,
                                      "BESIDE_GAP": BESIDE_GAP,
                                      "RING_EDGE_INSET": RING_EDGE_INSET,
                                      "LANE_GAP": LANE_GAP, "SECTOR_MAX": SECTOR_MAX,
                                      "DISTRICT_FILL": DISTRICT_FILL,
                                      "RING_COVERAGE": RING_COVERAGE}}}


def _parts_for_word(spec: dict, word: str) -> list:
    """The defining parts a sentence word names, by the same ladder the checker's
    selector walks: the plain words for a dwelling name the settled districts; the
    words for fields name the farmland districts; anything else is matched on the
    part's name, its family and the family `intent.FEATURES` builds that word as."""
    from . import intent as intent_mod
    w = intent_mod._slug(word)
    words = [x for x in w.split("_") if x and x not in ("the", "a", "an", "of", "its")]
    if not words:
        return []
    head = words[-1]
    parts = list(spec.get("defining_parts") or [])
    groups = [p for p in parts if p["kind"] == "group" and not spec_mod.compound(p)]
    # **The selector and the entity rule read one vocabulary.** `intent._HOUSE_WORDS` is
    # the checker's list and `_HOUSE_FAMILIES` is this module's; they are two lists of
    # the same thing and they disagreed. Found by running the held-out village:
    # "thirteen **townhouses** gathered around a keep" -- `townhouse` is in
    # `_HOUSE_FAMILIES` (so `entity_of` calls the district a building) and not in
    # `_HOUSE_WORDS` (so this returned no part at all), and the sentence's own relation
    # resolved to nothing with no one to say so.
    if head in intent_mod._HOUSE_WORDS or head in _HOUSE_FAMILIES \
            or head.rstrip("s") in _HOUSE_FAMILIES:
        # the districts of houses are the building entities (the expression round): an
        # orchard is land whatever word its land use is, and is not "the houses"
        got = [p for p in groups if entity_of(p, spec).get("class") == "building"]
        return got or groups
    if head in ("field", "fields", "farmland", "farm", "farms", "fieldland"):
        return [p for p in groups if spec_mod.land_use(p) == "farmland"]
    if head in ("orchard", "orchards", "pasture", "pastures", "grove", "groves"):
        use = {"orchards": "orchard", "pastures": "pasture", "groves": "grove"}.get(head, head)
        got = [p for p in groups if entity_of(p, spec).get("land_use") == use]
        if got:
            return got
    fams = {head} | {f for rid, f, _ph in intent_mod.FEATURES if rid == head and f}
    out = []
    for p in parts:
        names = {intent_mod._slug(p.get("name") or ""), intent_mod._slug(p.get("family") or "")}
        if names & fams or any(n.startswith(f + "_") for n in names for f in fams):
            out.append(p)
    return out


def _relations_of(spec: dict, intent: dict | None) -> list:
    """The `relation` requirements of an intent record, each resolved to the spec's
    parts: `object_part` (a solid defining part's name, or None) and `subject_groups`
    (the district groups the subject names). Empty without an intent record."""
    out = []
    for r in (intent or {}).get("requirements") or []:
        if r.get("kind") != "relation" or r.get("status") == "unsupported":
            continue
        w = r.get("wants") or {}
        subj, obj = str(w.get("subject") or ""), str(w.get("object") or "")
        objs = _parts_for_word(spec, obj)
        subs = _parts_for_word(spec, subj)
        solid = [p for p in objs if p["kind"] != "group"]
        out.append({"id": r.get("id"), "relation": str(w.get("relation") or ""),
                    "subject": subj, "object": obj,
                    "object_part": solid[0]["name"] if solid else None,
                    "object_groups": [p["name"] for p in objs if p["kind"] == "group"],
                    "subject_groups": [p["name"] for p in subs if p["kind"] == "group"],
                    "subject_parts": [p["name"] for p in subs if p["kind"] != "group"]})
    return out


def _clear_of(rect, others, horiz: bool, *, free_high: bool, gap: int):
    """`rect` cut back along its running axis until it touches nothing in `others`.

        A district drawn **beside** another one grows away from it, so one end of it is
        anchored and the other is free; this moves the free end and never the anchored one,
        because moving the anchored end is moving the thing `beside` is measured from.
        Returns None where the free end cannot clear the obstacle -- the caller then leaves
        the region where the tiling put it rather than promising ground it cannot keep.
        
    """
    x0, z0, x1, z1 = (int(v) for v in rect)
    for o in others:
        ox0, oz0, ox1, oz1 = (int(v) for v in o)
        if not (x0 <= ox1 and ox0 <= x1 and z0 <= oz1 and oz0 <= z1):
            continue
        if horiz:
            x1, x0 = ((min(x1, ox0 - 1 - gap), x0) if free_high
                      else (x1, max(x0, ox1 + 1 + gap)))
        else:
            z1, z0 = ((min(z1, oz0 - 1 - gap), z0) if free_high
                      else (z1, max(z0, oz1 + 1 + gap)))
        if x1 < x0 or z1 < z0:
            return None
    return (x0, z0, x1, z1)


def _trim(rect, cut):
    """`rect` less `cut`: the larger remainder on the axis that loses least, or None."""
    if rect is None or not _overlaps(rect, cut):
        return rect
    x0, z0, x1, z1 = rect
    options = []
    if cut[0] > x0:
        options.append((x0, z0, cut[0] - 1, z1))
    if cut[2] < x1:
        options.append((cut[2] + 1, z0, x1, z1))
    if cut[1] > z0:
        options.append((x0, z0, x1, cut[1] - 1))
    if cut[3] < z1:
        options.append((x0, cut[3] + 1, x1, z1))
    if not options:
        return None
    return max(options, key=lambda r: ((r[2] - r[0] + 1) * (r[3] - r[1] + 1), r))


def _geom(leaf: dict) -> dict:
    return {k: v for k, v in leaf.items()
            if k in ("kind", "type", "x0", "z0", "x1", "z1", "at", "facing", "path",
                     "width", "size", "shape")}


def policy_for(spec: dict) -> str:
    """Which whole-place layout policy this request asks for.

        One seam, three answers, and the order is the order of strength:

          `concentric`  the spec declares rings. Rings are a statement about the whole
                        place and nothing else can be the organisation at the same time.
          `shoreline`   the **sentence** asks for a place along the water
                        (`placeshore.wants_shoreline`, through `ethoslm.intent`, never a
                        place's name).
          `relations`   everything else: every defining part placed against the others.

        A policy is a **choice recorded on the plan**, which is what lets `intent.coverage`
        check "the sentence asked for a shoreline place" against what was actually laid out
        instead of against a word in a brief.
        
    """
    from . import placeshore
    if spec_mod.rings(spec):
        return "concentric"
    if placeshore.wants_shoreline(spec):
        return "shoreline"
    return "relations"


def approved_types(caps: dict | None) -> dict:
    """`{defining part: the type the capability record approved}`.

        The one translation between the capability record and the layouts, so that "which
        type is this part?" has a single answer and a reader can find where it was made.
        
    """
    out = {}
    for e in (caps or {}).get("entries") or []:
        want = e.get("wants") or {}
        if e.get("matched") and want.get("of") == "defining_part" and want.get("part"):
            out[want["part"]] = e.get("type")
    return out


def approved_fabric(caps: dict | None) -> dict:
    """`{defining part: the plot types the record approved for its fabric}`.

        The fabric want is the one the compiler answers, and it is answered out of a *pool*
        rather than one type -- so the entry's lead type and its runners-up together are the
        approval, which is exactly what `ALTERNATIVES` was recorded for.
        
    """
    out = {}
    for e in (caps or {}).get("entries") or []:
        want = e.get("wants") or {}
        if not e.get("matched") or want.get("of") != "fabric" or not want.get("part"):
            continue
        pool = [e.get("type"), *(e.get("alternatives") or [])]
        out[want["part"]] = [t for t in pool if t]
    return out


def approved_from_place(place: dict | None) -> tuple:
    """`({defining part: [approved types]}, {defining part: demand})` read off a place
        that has already been laid out.

        **A place is the record of what was approved for it.** The design round, found by
        running the farm's improve stage: `reallocate` re-solves through `solve_place`, and
        where its caller passes no capability record `approved_fabric({})` is empty -- so
        the re-solve resolved its demand against the *general library*, `house_types`
        admitted every type of the role, and the district that had been built of `cottage`
        came back built of `hall` and `worship`. The capability check then refused the
        reallocation for it, and with a hall's lot in place of a cottage's the same
        rectangles held ten of the sixteen the sentence states. One cause, both defects.

        So a revision does not depend on its caller to hand it the record again. The place
        it is revising carries `fabric_types` and `demand` on every district
        (`_write_fabric`, `_write_demand`), and that is where they are read from.
        
    """
    pools, demands = {}, {}
    for d in (place or {}).get("districts") or []:
        key = d.get("defines") or d.get("name")
        if not key:
            continue
        if d.get("fabric_types"):
            pools.setdefault(key, list(d["fabric_types"]))
        if d.get("demand"):
            demands.setdefault(key, d["demand"])
    for name, dem in ((place or {}).get("layout") or {}).get("demand", {}).items():
        demands.setdefault(name, dem)
    return pools, demands


def _write_fabric(place: dict | None, caps: dict | None,
                  approved: dict | None = None) -> None:
    """Tell each district which types the capability record approved for its fabric.

        **The one seam where the record reaches the compiler.** The review's first finding
        was that a capability record was written and then not consulted; for a district that
        is not a missing keyword argument but a missing field, because the compiler is
        reached one level down through the district document rather than through this call.
        
    """
    pools = {**(approved or {}), **approved_fabric(caps)}
    for d in (place or {}).get("districts") or []:
        got = pools.get(d.get("defines")) or pools.get(d.get("name"))
        if got:
            d["fabric_types"] = list(got)


def _write_demand(place: dict | None, spec: dict) -> None:
    """Carry each defining part's **resolved demand** onto the districts it defines.

        The compiler is reached one level down, through the district document, so a demand
        that stayed on the defining part would be a demand the envelope queries never saw
        -- which is the same shape of defect `_write_fabric` exists to close. A district
        whose part resolved none is untouched and behaves as it did.
        
    """
    by = {p.get("name"): p for p in (spec or {}).get("defining_parts") or []}
    for d in (place or {}).get("districts") or []:
        part = by.get(d.get("defines")) or by.get(d.get("name"))
        if part is not None and part.get("demand") and not d.get("demand"):
            d["demand"] = part["demand"]


def resolved_spec(spec: dict, caps: dict | None, intent: dict | None = None, *,
                  decls: dict | None = None, allocation: dict | None = None,
                  voice: str | None = None, approved: dict | None = None,
                  demands: dict | None = None) -> tuple:
    """**The spec with every district's demand resolved, before anything is sized.**

        The review's first cause, at its timing end: `_solve_place` applied `approved_fabric`
        through `_write_fabric` *after* the policy had already sized the place, so
        `fabric_lot` estimated a parent from a type the general library offered and the
        district was eventually built of another. A parent sized against a type the child
        does not use is a parent sized against nothing.

        So the approved pool and the resolved demand -- the type pool in its approved order,
        the required parameters and features, the requirement ids that made them required,
        and the scope -- go onto each defining part **before** the first dimension is
        derived from it. Returns `(spec, refusals)`: the spec to lay out, and the parts
        whose demand nothing in the approved band delivers, which is an honest stop and
        never a smaller default.

        A call with no capability record behaves exactly as it did.
        
    """
    # the capability record where the caller has one, and otherwise what the place being
    # revised already records as approved (`approved_from_place`): a re-solve never
    # falls back to the general library, because a pool nothing approved is how a
    # village of cottages comes back built of halls.
    pools = {**(approved or {}), **approved_fabric(caps)}
    mod = _demand_module()
    parts, refused = [], []
    for p in spec.get("defining_parts") or []:
        got = pools.get(p.get("name"))
        p2 = dict(p)
        if got:
            p2["fabric_types"] = list(got)
        elif (demands or {}).get(p.get("name"), {}).get("types"):
            p2["fabric_types"] = list(demands[p["name"]]["types"])
        if mod is not None and hasattr(mod, "resolve") and intent is not None:
            try:
                d = mod.resolve(spec, intent, p2, decls or {}, allocation=allocation,
                                voice=voice,
                                context={"capabilities": caps, "voice": voice})
            except Exception as e:           # noqa: BLE001 -- recorded, never defaulted
                refused.append({"part": p.get("name"), "why": str(e),
                                "owner": "capability"})
                d = None
            if d:
                p2["demand"] = d
                if d.get("types"):
                    p2["fabric_types"] = list(d["types"])
        if not p2.get("demand") and (demands or {}).get(p.get("name")):
            # the demand the place resolved before it was first sized: a revision is of
            # the same request and resolves the same demand, so carrying it is carrying
            # the answer rather than guessing it again with less in hand
            p2["demand"] = demands[p["name"]]
        parts.append(p2)
    return dict(spec, defining_parts=parts), refused


def solve_place(spec: dict, site: dict, plateau: dict | None, decls: dict, voice: str,
                vol=None, seed: int = 1, caps: dict | None = None,
                intent: dict | None = None, allocation: dict | None = None,
                envelope_cache: str | None = None, approved: dict | None = None,
                demands: dict | None = None) -> tuple:
    """The place level, by the policy the request asks for. Returns `(place, fails)`.

        Every policy answers in the same shape and is validated by the same
        `place_failures`, compiled by the same `district_compile` and built by the same
        parts stage. That is the whole of what makes this a layout *interface* rather than
        two pipelines behind a switch.

        `caps` is the capability record, and it is the **authority on which type each
        defining part is built as**. A call that passes none behaves as it always did.

        `intent` is the checked intent record (the closure round): its `relation`
        requirements reach the relation solver as constraints on where the districts stand
        -- "gathered around a market square" puts the houses on at least three sides of the
        square before any rectangle is filled -- and its explicit count is laid exactly. A
        call that passes none lays the place as it always did.
        
    """
    from . import placeshore
    global ENVELOPE_CACHE
    policy = policy_for(spec)
    chosen = approved_types(caps)
    alloc = allocation_of(spec, allocation)
    was_cache = ENVELOPE_CACHE
    ENVELOPE_CACHE = envelope_cache or was_cache
    # **Demand before size** (the design round's first contract): the approved fabric
    # pool and the resolved demand go onto the defining parts before the first dimension
    # is derived from one of them.
    spec, refused = resolved_spec(spec, caps, intent, decls=decls, allocation=alloc,
                                  voice=voice, approved=approved, demands=demands)
    try:
        place, fails = _solve_place(spec, site, plateau, decls, voice, vol, seed, caps,
                                    intent, alloc, chosen, policy, placeshore,
                                    approved=approved)
    finally:
        ENVELOPE_CACHE = was_cache
    if refused:
        # a demand nothing in the approved band delivers is a refusal the caller has to
        # answer, not a smaller default: it goes in `fails` beside the layout's own
        fails = list(fails or []) + [
            {"part": r["part"], "clause": "demand", "why": r["why"],
             "owner": r.get("owner") or "capability"} for r in refused]
    if place is not None:
        # the demand each part resolved before it was sized, on the record -- and only
        # where there is one, so a call with no capability record produces exactly the
        # layout document it always did
        got = {p["name"]: {k: p["demand"].get(k) for k in
                           ("types", "params", "required", "optional", "requirements",
                            "scope")}
               for p in spec.get("defining_parts") or [] if p.get("demand")}
        if got:
            place.setdefault("layout", {})["demand"] = got
        if refused:
            place.setdefault("layout", {})["demand_refused"] = refused
    return place, fails


def _solve_place(spec, site, plateau, decls, voice, vol, seed, caps, intent, alloc,
                 chosen, policy, placeshore, approved=None) -> tuple:
    if policy == "concentric":
        place, fails = placeplan.concentric_layout(spec, site, plateau, decls, voice,
                                                   vol=vol, caps=chosen,
                                                   allocation=alloc)
        if place is not None:
            place["layout"]["solver"] = {
                "by": "placesolve.solve_place", "case": "concentric",
                "why": "a place with rings is one solved case: the ring arithmetic "
                       "seeds the one candidate and the place validator is its veto"}
    elif policy == "shoreline":
        place, fails = placeshore.shoreline_layout(spec, site, plateau, decls, voice,
                                                   vol=vol, seed=seed, caps=chosen,
                                                   intent=intent)
        if place is not None:
            place["layout"].setdefault("allocation", dict(alloc))
    else:
        place, fails = Solver(spec, site, plateau, decls, voice, vol=vol, seed=seed,
                              caps=chosen, intent=intent, allocation=alloc).solve()
        if place is not None:
            place["layout"]["policy"] = "relations"
    _apply_structures(place, alloc, spec)
    _write_fabric(place, caps, approved)
    _write_demand(place, spec)
    if place is not None:
        _write_hierarchy(place, spec, decls)
        if ENVELOPE_CACHE:
            place.setdefault("layout", {})["envelope_cache"] = ENVELOPE_CACHE
    return place, fails


def _apply_structures(place, allocation: dict | None, spec: dict) -> None:
    """**`allocation.structures` -- the layout owner's `redistribute`, applied.**

        One override per district name, written by `reallocate`'s `redistribute` action. It is
        applied *here*, after whichever layout the policy chose has drawn its districts, so
        that the same action works on a concentric place, a shoreline and a relations solve
        without three copies of it.

        **Count-preserving reallocation is an invariant, and this is where it is held.** The
        total is compared before and after and an override that changes it is *not applied*:
        the record on the place says so by name. A district marked `exact` is never moved --
        `redistribute` refuses itself in that case, and this refuses again, because an
        invariant checked in one place is a convention.
        
    """
    over = ((allocation or {}).get("structures") or {})
    if place is None or not isinstance(over, dict) or not over:
        return
    ds = place.get("districts") or []
    was = {d.get("name"): int(d.get("structures") or 0) for d in ds}
    total = sum(was.values())
    moved, refused = {}, []
    for d in ds:
        name = d.get("name")
        if name not in over:
            continue
        if d.get("exact"):
            refused.append({"district": name, "why": "this district's count is exact"})
            continue
        band = d.get("count_band") or {}
        want = max(0, int(over[name]))
        hi = int(band.get("hi") or 0)
        if hi and want > hi:
            refused.append({"district": name,
                            "why": f"{want} is over this district's own band top of {hi}"})
            continue
        d["structures"] = want
        moved[name] = [was[name], want]
    now = sum(int(d.get("structures") or 0) for d in ds)
    rec = {"asked": dict(over), "applied": moved, "refused": refused,
           "total": {"was": total, "now": now}}
    if now != total:
        # the invariant, asserted in code: put it back and say so
        for d in ds:
            if d.get("name") in moved:
                d["structures"] = was[d["name"]]
        rec.update(applied={}, invariant="count",
                   why=(f"this redistribution would change the place's total count from "
                        f"{total} to {now}; a reallocation moves population between "
                        f"districts and never creates or destroys it, so it is refused "
                        f"whole and the districts stand as they were"),
                   total={"was": total, "now": total})
    elif moved:
        rec["why"] = (f"{len(moved)} district(s) re-counted by an `allocation.structures` "
                      f"row, the place's total unchanged at {total}")
    place.setdefault("layout", {})["structures_allocated"] = rec


# ------------------------------------------------------------ the owner's action

#: The bounded actions the layout owner has for a built finding, and nothing else. **The
#: four arrangement actions are the design round's.** The expression round's dense ring
#: had two owner actions. A district's depth in rows, its bay, its frontage and whether
#: its houses stand along a street or about a court are four more, they are the
#: compiler's own (`arrange.arrangements`), and each is certified by `district_compile`
#: before it is adopted. They are tried in this order, one per finding, so an action
#: that moved nothing leaves the next one available rather than ending the selection.
ARRANGEMENT_ACTIONS = ("row_depth", "bay_width", "frontage", "compound")

#: **The composition round's three additions, and why each is a layout action.**
#: enlarge_anchor the other direction of `shrink_anchor`. `shrink_anchor` clamped with
#: `min(new, share)` so it could only ever reduce, and a finding that says the square is
#: *too small for what gathers about it* -- which is what the checker's own `least`
#: bound in `anchor_size` exists for -- had no action at all. `resize_anchor` is the
#: same code taking a direction; both names route to it and `shrink_anchor` behaves
#: exactly as it did. regroup gather the fabric toward the anchor and its streets
#: (`Solver._compact`/`_straddle`), which until now ran only where the count was the
#: sentence's own. A scattered-fabric finding is a real finding about an inferred count
#: too. redistribute move **inferred** population between the districts of one part,
#: preserving the place's total. and it is refused **by name** where the count is
#: explicit or a district is marked exact.
RESIZE_ANCHOR_ACTIONS = ("shrink_anchor", "enlarge_anchor", "resize_anchor")
REALLOCATE_ACTIONS = (*RESIZE_ANCHOR_ACTIONS, "resize_ring", "grow_land", "enlarge_lots",
                      "regroup", "redistribute", "move_object", *ARRANGEMENT_ACTIONS)
RING_STEP = {"down": 0.7, "up": 1.4}

#: **One discoverable and executable action inventory, published from the side that
#: executes them.** The composition round's fifth change, and the review's words: the
#: inventory disagreed across layers. `pipeline/improve.py`'s `OWNER_ACTIONS` listed
#: `move_object` for `layout`, which `_reallocate` refuses, and omitted the four
#: `ARRANGEMENT_ACTIONS` the design round built -- so an action that existed was
#: undiscoverable and an action that was discoverable did not exist. A list kept in the
#: stage that dispatches and a list kept in the module that acts will drift; this is the
#: one the actions are in, so `improve` reads it rather than repeating it. `move_object`
#: stays on the layout's list on purpose: `_reallocate` refuses it *and* names the owner
#: that has it (`owner_of_action`) and, for a district's own fabric, the layout action
#: that does move it (`alternative`). A refusal that routes is worth more to the
#: dispatcher than an absence.
ACTIONS_BY_OWNER = {
    "layout": (*RESIZE_ANCHOR_ACTIONS, "resize_ring", "grow_land", "enlarge_lots",
               "regroup", "redistribute", "move_object", *ARRANGEMENT_ACTIONS),
    # a character revision is the fabric owner's; the lot is the one dimension both own
    "fabric": ("character", "enlarge_lots", *ARRANGEMENT_ACTIONS),
    "voice": ("voice",),
    "build": ("enlarge_lots",),
    # the scale owner sizes the place and its population; `redistribute` divides an
    # inferred population without changing it, which is this owner's decision as much as
    # the layout's
    "scale": (*RESIZE_ANCHOR_ACTIONS, "grow_land", "redistribute"),
    "capability": (),
}

#: One bounded `enlarge_anchor` multiplies the anchor's share by this, and this is the
#: ceiling it may reach. `ANCHOR_SHARE_STEP` is 0.7 down; up is its reciprocal, so an
#: enlargement and a shrink of the same subject are the same size of step in both
#: directions and a pair of them returns to where it started. The ceiling is two lots a
#: household: the largest `ANCHOR_SHARE` any family declares is 0.6 (`green`), and a
#: square at more than two lots a household is a field with houses round it.
ANCHOR_SHARE_UP = 1.0 / ANCHOR_SHARE_STEP
ANCHOR_SHARE_MAX = 2.0

#: How much of one district's count a bounded `redistribute` may move, as a share of
#: that district's own number. A fifth: enough to relieve a district the compiler could
#: not fill and small enough that the fabric of the place is recognisably the same
#: fabric.
REDISTRIBUTE_SHARE = 0.2


def _action_for(finding: dict, place: dict, spec: dict) -> tuple:
    """`(action, subject)` the finding asks of the layout, from its own `action` where
    it names one, else from its measure and subjects; `(None, why)` where none."""
    act = str(finding.get("action") or "")
    if isinstance(finding.get("action"), dict):
        return None, "the action is another owner's (a character or a voice revision)"
    subjects = [str(s) for s in (finding.get("subjects") or [])]
    measure = str(finding.get("measure") or "")
    about = str(finding.get("about") or "")
    says = str(finding.get("says") or "").lower()
    districts = {d.get("name"): d for d in place.get("districts") or []}
    parts = {p.get("name"): p for p in place.get("parts") or []}
    core = (spec_mod.core(spec) or {}).get("name")
    land = [p["name"] for p in spec.get("defining_parts") or []
            if p["kind"] == "group" and entity_of(p, spec).get("class") == "land"]
    rings = [r["name"] for r in spec_mod.rings(spec)]
    if act in REALLOCATE_ACTIONS:
        subj = subjects[0] if subjects else None
        if act in RESIZE_ANCHOR_ACTIONS:
            return act, (next((s for s in subjects if s in parts), None) or core)
        if act in ("regroup", "redistribute"):
            # both are about a **group of houses**: the district's defining part where
            # the finding names a district, else the part it names outright
            return act, (next((districts[s].get("defines") for s in subjects
                               if s in districts), None)
                         or next((s for s in subjects
                                  if s in {p.get("name")
                                           for p in spec.get("defining_parts") or []}),
                                 None) or subj)
        if act == "grow_land":
            return act, (next((districts[s].get("defines") for s in subjects
                               if s in districts and districts[s].get("defines") in land),
                              None) or next((s for s in subjects if s in land), None)
                         or (land[0] if land else None))
        if act == "resize_ring":
            return act, (next((districts[s].get("defines") for s in subjects
                               if s in districts and districts[s].get("defines") in rings),
                              None) or next((s for s in subjects if s in rings), None)
                         or (rings[0] if rings else None))
        if act == "enlarge_lots" or act in ARRANGEMENT_ACTIONS:
            return act, (next((districts[s].get("defines") for s in subjects
                               if s in districts), None) or subj)
        return act, subj
    if measure == "square_scale" or (about == "scale" and core and
                                     (not subjects or core in subjects)):
        return "shrink_anchor", core
    if measure == "open_to_built" and land and \
            (str((finding.get("target") or {}).get("direction") or "") == "up"
             or any(w in says for w in ("thin", "more", "working land", "fields"))):
        return "grow_land", next((districts[s].get("defines") for s in subjects
                                  if s in districts and districts[s].get("defines") in land),
                                 land[0])
    if any(w in says for w in ("storey", "storeys", "lot")) and about == "fabric":
        return "enlarge_lots", next((districts[s].get("defines") for s in subjects
                                     if s in districts), None)
    # **a density finding is an arrangement decision before it is a size decision.**
    # "the lots cover 9.0% of the districts' ground against the at least 30% this build
    # calls dense" is a statement about how the houses are arranged in the ground they
    # were given, and enlarging empty lots is exactly the answer that would raise the
    # number without making the place denser. The four arrangement actions are tried in
    # order; one per finding, so the next stays available.
    if measure in ("lot_cover", "density") or about == "density" \
            or any(w in says for w in ("cover", "dense", "sparse", "density")):
        subj = next((districts[s].get("defines") for s in subjects if s in districts),
                    None) or next((s for s in subjects
                                   if s in {p.get("name") for p in
                                            spec.get("defining_parts") or []}), None)
        if subj:
            done = {row.get("action") for row in spec.get("negotiated") or []
                    if isinstance(row, dict)
                    and str(row.get("action")) in ARRANGEMENT_ACTIONS
                    and subj in ((row.get("to") or {}).get("arrangement") or {})}
            nxt = next((a for a in ARRANGEMENT_ACTIONS if a not in done), None)
            if nxt:
                return nxt, subj
    if rings and any(s in rings or (s in districts and districts[s].get("defines") in rings)
                     for s in subjects):
        return "resize_ring", next((districts[s].get("defines") if s in districts else s
                                    for s in subjects
                                    if s in rings or (s in districts and
                                                      districts[s].get("defines") in rings)))
    return None, "this finding names no anchor, ring, land or lot the layout owner sizes"


def _rects_of(place: dict) -> dict:
    out = {}
    for key in ("parts", "districts", "compounds"):
        for r in place.get(key) or []:
            if r.get("x1") is not None:
                out[r.get("name")] = [int(r["x0"]), int(r["z0"]), int(r["x1"]), int(r["z1"])]
            elif r.get("path"):
                out[r.get("name")] = [list(a) for a in r["path"]][:2]
    return out


def reallocate(place: dict, spec: dict, finding: dict, *, site: dict | None = None,
               decls: dict | None = None, envelopes=None,
               constraints: list | None = None, intent: dict | None = None,
               plateau: dict | None = None, vol=None, seed: int | None = None,
               caps: dict | None = None, **_kw) -> tuple:
    """**The layout owner's one bounded action on a finding**, and the place laid out
        again from it. Returns `(place, record)`: the re-solved place where the action
        applied, the place handed in where it was refused, and a record either way --
        the action, the finding, what it changed (`from`/`to` rectangles), the allocation
        row it wrote, and why. Deterministic under the seed.

        The action is a revision of an **inferred** allocation -- the anchor's share, a
        ring's width, a land region's columns, a district's least lot -- recorded as an
        `allocation` row on the spec's `negotiated` list (the scale owner's channel, which
        `spec.read_spec` preserves and the plan's fingerprint reads), so the place laid
        out again from the spec is this place. It never touches an explicit requirement:
        the count, the relations and an absent feature are the sentence's, and the solver
        is what lays them, whatever the allocation says.
        
    """
    rec = {"finding": finding.get("id"), "owner": "layout", "applied": False,
           "changed": False, "action": None, "subject": None, "refused": None,
           "before": None, "after": None}
    got_place, rec = _reallocate(place, spec, finding, rec, site=site, decls=decls,
                                 constraints=constraints, intent=intent, plateau=plateau,
                                 vol=vol, seed=seed, caps=caps, envelopes=envelopes)
    # the record's contract: `refused` is the reason or None; `before`/`after` the
    # rectangles the action moved
    if rec.get("refused") is True:
        rec["refused"] = str(rec.get("why") or "refused")
    elif not rec.get("applied"):
        rec["refused"] = rec.get("refused") or str(rec.get("why") or "refused")
    else:
        rec["refused"] = None
    moved = rec.get("districts") or []
    rec["before"] = {m["district"]: m["from"] for m in moved} or None
    rec["after"] = {m["district"]: m["to"] for m in moved} or None
    return got_place, rec


def _reallocate(place: dict, spec: dict, finding: dict, rec: dict, *, site=None,
                decls=None, constraints=None, intent=None, plateau=None, vol=None,
                seed=None, caps=None, envelopes=None) -> tuple:
    action, subject = _action_for(finding, place, spec)
    if action is None:
        rec.update(refused=True, why=subject)
        return place, rec
    rec.update(action=action, subject=subject)
    if action == "move_object":
        # **A refusal names the owner that does have the action, and offers one.** The
        # composition round: `pipeline/improve.py`'s `OWNER_ACTIONS` lists `move_object`
        # for `layout` and this refused it unconditionally with a reason and nothing
        # else, so the finding reached a dead end that looked like an exhausted action
        # list. The owner that moves a part is the relation repair; where the subject is
        # a district's fabric rather than a standing part, the layout's own `regroup` is
        # the action that moves it, and the alternative says so by name.
        here = {d.get("name"): d for d in place.get("districts") or []}
        mine = (subject in here or any(d.get("defines") == subject
                                      for d in place.get("districts") or []))
        alt = ("regroup" if mine else None)
        rec.update(refused=True, owner_of_action="relation repair (`repair.apply`)",
                   alternative=alt,
                   why=("moving a standing part is the relation repair's action "
                        "(`repair.apply`), which measures where the subjects stand and "
                        "moves one of them; an allocation cannot place a part. "
                        + (f"For `{subject}`, which is this place's own fabric and not a "
                           f"standing part, the layout owner's action that moves it is "
                           f"`regroup`: it gathers the districts toward the anchor and "
                           f"its streets. Route the finding to `regroup`."
                           if alt else
                           f"`{subject}` is a standing part; hand the finding to the "
                           f"relation repair.")))
        return place, rec
    if not site:
        rec.update(refused=True, why="no site: the place cannot be laid out again")
        return place, rec
    if decls is None:
        _t, decls = placeplan.types_card(None, spec.get("form"))
    decls = decls or {}
    current = allocation_of(spec)
    to: dict = {}
    why = ""
    parts_by = {p["name"]: p for p in spec.get("defining_parts") or []}
    if action in RESIZE_ANCHOR_ACTIONS:
        # **The anchor's share moves in both directions**, the composition round's fifth
        # change. `shrink_anchor` clamped with `min(new, share)`, so a finding asking
        # for a *larger* square -- which is what `anchor_size`'s own `least` bound is
        # about, and what "the palace's processional sequence needs relationships" means
        # at the square's scale -- reached this branch and was told the anchor cannot be
        # smaller. One body, a direction, and `shrink_anchor` is unchanged where it is
        # asked to shrink. The direction is the finding's own where it states one, the
        # action's name where the name says it, and down by default (which is every
        # existing caller).
        part = parts_by.get(subject) or spec_mod.core(spec) or {}
        fam = str(part.get("family") or "").lower()
        share = float((current.get("anchor") or {}).get("share")
                      if (current.get("anchor") or {}).get("share") is not None
                      else ANCHOR_SHARE.get(fam, ANCHOR_SHARE_DEFAULT))
        tgt = finding.get("target") or {}
        said = str(tgt.get("direction") or "").lower()
        up = (action == "enlarge_anchor" or (said == "up" and action != "shrink_anchor"))
        step = ANCHOR_SHARE_UP if up else ANCHOR_SHARE_STEP
        new = share * step
        if tgt.get("value") is not None and finding.get("measured") is not None \
                and str(tgt.get("measure") or finding.get("measure")) == "square_scale":
            try:
                new = share * float(tgt["value"]) / float(finding["measured"])
            except (TypeError, ValueError, ZeroDivisionError):
                new = share * step
        if up:
            new = min(ANCHOR_SHARE_MAX, round(max(new, share * ANCHOR_SHARE_UP), 3))
        else:
            new = max(ANCHOR_SHARE_MIN, round(min(new, share), 3))
        at_stop = (share >= ANCHOR_SHARE_MAX - 1e-9 if up
                   else share <= ANCHOR_SHARE_MIN + 1e-9)
        if at_stop or (new >= share if not up else new <= share):
            rec.update(refused=True, direction="up" if up else "down",
                       why=(f"the anchor's share is {share:g} of a lot a household, at "
                            f"the registered "
                            + (f"ceiling of {ANCHOR_SHARE_MAX:g}; the square cannot be "
                               f"larger and still be a square the fabric encloses"
                               if up else
                               f"floor of {ANCHOR_SHARE_MIN:g}; the square cannot be "
                               f"smaller and still be the market the count gathers "
                               f"about")))
            return place, rec
        to = {"anchor": {"share": new}}
        why = (f"the anchor `{part.get('name')}` is sized at {share:g} of a lot a "
               f"household and the built reading finds it "
               f"{'too small for what gathers about it' if up else 'out of scale'}; the "
               f"share is {'raised' if up else 'reduced'} to {new:g} and the place laid "
               f"out again about it. The hard constraints this preserves: the place's "
               f"count, its relations and the anchor's own type band "
               f"(`anchor_size` clamps the side into it)")
        rec["direction"] = "up" if up else "down"
    elif action == "grow_land":
        part = parts_by.get(subject)
        if not part:
            rec.update(refused=True, why=f"no land part named `{subject}`")
            return place, rec
        d = next((d for d in place.get("districts") or [] if d.get("defines") == subject
                  and (d.get("target") or {}).get("what") == "land"), None)
        have = int((((d or {}).get("target") or {}).get("value") or {}).get("land_columns")
                   or (current.get("land") or {}).get(subject, {}).get("columns") or 0)
        if not have:
            hh = int((spec.get("explicit_count") or {}).get("n") or spec.get("structures") or 0)
            have = land_need(part, 0, decls, spec, households=hh,
                             allocation=current).get("land_columns") or 0
        tgt = finding.get("target") or {}
        new = int(math.ceil(have * LAND_STEP))
        if tgt.get("value") and str(tgt.get("measure") or "") == "land_columns":
            new = max(new, int(tgt["value"]))
        short = [(x.get("name"), (x.get("target") or {}).get("value") or {})
                 for x in place.get("districts") or [] if x.get("defines") == subject]
        if d is not None and int((d.get("target") or {}).get("short_columns") or 0) \
                or any(v.get("got_columns") and v.get("columns")
                       and int(v["got_columns"]) < int(v["columns"]) for _n, v in short):
            got_cols = sum(int(v.get("got_columns") or 0) for _n, v in short)
            need_cols = sum(int(v.get("columns") or 0) for _n, v in short)
            rec.update(refused=True, allocation={"land": {subject: {"columns": new}}},
                       why=(f"the land `{subject}` already holds {got_cols} of the "
                            f"{need_cols} columns its purpose needs and the strip it "
                            f"stands in is the site's limit: no adjoining ground is "
                            f"free, so asking for {new} lays the same ground. The site "
                            f"was sized for the count and not for the land; the site "
                            f"is the limiting constraint"))
            return place, rec
        to = {"land": {subject: {"columns": new}}}
        why = (f"the land `{subject}` is sized at {have} columns for the place's "
               f"households and the built reading finds it thin for its purpose; the "
               f"need is raised to {new} and the place laid out again")
    elif action == "enlarge_lots":
        lots = lots_from_constraints(constraints)
        group = subject
        wanted = None
        if lots:
            # the constrained parts' district names begin with the district's own
            names = {d.get("name"): d.get("defines") for d in place.get("districts") or []}
            by_group: dict = {}
            for c in constraints or []:
                if c.get("owner") != "layout" or not (c.get("needs") or {}).get("lot_min"):
                    continue
                g = next((names[n] for n in names if str(c.get("part") or "").startswith(n)),
                         None)
                if g:
                    lm = c["needs"]["lot_min"]
                    was = by_group.get(g)
                    by_group[g] = [max(int(lm[0]), was[0]), max(int(lm[1]), was[1])] \
                        if was else [int(lm[0]), int(lm[1])]
            if group and group in by_group:
                wanted = {group: by_group[group]}
            elif by_group:
                wanted = by_group
        if wanted is None and group and parts_by.get(group):
            st = (parts_by[group].get("character") or {}).get("storeys")
            (w0, d0), _s = fabric_lot(parts_by[group], decls, spec, allocation=current)
            from . import district_compile as dc
            house = next((n for n, _d in dc.house_types(
                decls, parts_by[group].get("role"), spec.get("form"),
                approved=parts_by[group].get("fabric_types") or None)), None)
            lm = _lot_min_for(house, {"storeys": int(st[-1])}) if (st and house) else None
            if lm and (lm[0] > w0 or lm[1] > d0):
                wanted = {group: [int(lm[0]), int(lm[1])]}
        if not wanted:
            rec.update(refused=True,
                       why="no emitted constraint names a lot construction needs and no "
                           "envelope says the storey band wants a larger one: nothing "
                           "to enlarge the lots to")
            return place, rec
        old = current.get("lots") or {}
        if all(old.get(g) and list(old[g]) >= list(v) for g, v in wanted.items()):
            rec.update(refused=True,
                       why=f"the lots are already at least what was asked: {old}")
            return place, rec
        to = {"lots": wanted}
        why = (f"construction recorded the lot its storeys need; the least lot of "
               f"{', '.join(f'{g} = {v[0]}x{v[1]}' for g, v in wanted.items())} is "
               f"written on the allocation and the districts are laid out again")
    elif action in ARRANGEMENT_ACTIONS:
        from . import arrange as _arrange
        part = parts_by.get(subject)
        if not part:
            rec.update(refused=True, why=f"no defining part named `{subject}`")
            return place, rec
        pool = next((d.get("fabric_types") for d in place.get("districts") or []
                     if d.get("defines") == subject and d.get("fabric_types")), None)
        opts = _arrange.arrangements(dict(part, **({"fabric_types": list(pool)}
                                                   if pool else {})),
                                     decls, spec=spec, allocation=current)
        here = [a for a in opts if a.get("action") == action]
        alive = [a for a in here if not a.get("refused")]
        if not alive:
            rec.update(refused=True,
                       why=(f"`{action}` is not a decision this fabric offers: "
                            + "; ".join(str(a.get("refused")) for a in here)
                            or f"no alternative of `{action}` for `{subject}`"))
            return place, rec
        # the alternative of this kind that puts most ground under buildings: fewest
        # rows for a depth decision, the largest lot for a bay decision, and the other
        # frontage or the court for the two that are a choice between two things
        pick = max(alive, key=lambda a: (a["lot"][0] * a["lot"][1], -int(a["rows"])))
        if action == "row_depth":
            pick = min(alive, key=lambda a: (int(a["rows"]), int(a["depth"])))
        have = (current.get("arrangement") or {}).get(subject) or {}
        want = dict(pick["arrangement"])
        if all(have.get(k) == v for k, v in want.items()):
            rec.update(refused=True,
                       why=f"`{action}` for `{subject}` is already what the allocation "
                           f"says: {have}")
            return place, rec
        to = {"arrangement": {subject: {**have, **want}}}
        why = (f"the built reading finds `{subject}` short of the density its own word "
               f"asks; `{action}` is the layout owner's bounded answer -- {pick['why']} "
               f"-- and the place is laid out again with it"
               + (f" (this revises the district brief's {', '.join(pick['revises'])})"
                  if pick.get("revises") else ""))
    elif action == "regroup":
        # **Gather the fabric toward the anchor and the street.** `Solver._compact`
        # draws each gathered district to what its own count needs at its density's
        # target cover and hugs it to the core box; `Solver._straddle` cuts a row long
        # enough at the axis so both quadrants are actually occupied. Both already
        # existed and both ran only where the count was the sentence's own; this asks
        # for them where it is not.
        gathered = sorted({n for r in _relations_of(spec, intent)
                           if r.get("relation") == "around"
                           for n in (r.get("subject_groups") or [])})
        if not gathered:
            gathered = sorted({n for r in ((place.get("layout") or {}).get("relations")
                                           or []) if r.get("relation") == "around"
                               for n in (r.get("subject_groups") or [])})
        if not gathered:
            rec.update(refused=True,
                       why=(f"nothing in this place is gathered about anything: "
                            f"`regroup` compacts the districts of an `around` relation "
                            f"toward the thing they gather about, and this sentence "
                            f"states no such relation. A concentric place's sectors are "
                            f"a ring's arithmetic and `resize_ring` is their action"))
            return place, rec
        if (current.get("regroup") or {}).get("gather"):
            rec.update(refused=True,
                       why=f"the fabric of {', '.join(gathered)} is already compacted "
                           f"toward the centre by an earlier `regroup`; there is no "
                           f"second step of this action")
            return place, rec
        to = {"regroup": {"gather": True}}
        why = (f"the built reading finds the fabric of {', '.join(gathered)} spread over "
               f"its strips rather than gathered about the centre; `regroup` draws each "
               f"district to the ground its own count needs at its density's target cover "
               f"and hugs it to the anchor and its streets, leaving the rest as unclaimed "
               f"ground on the record. The hard constraints it preserves: every count "
               f"(the rectangles move, the numbers do not), the `around` relation itself, "
               f"and every district's least side")
    elif action == "redistribute":
        # **Move inferred population between the districts of one part.** The rings case
        # did not add this and said why: that sentence stated its count, so moving
        # houses between its districts would be moving the sentence's own number.
        # `spec.structures_for` derived it from the site -- so the *distribution* of it
        # is the layout's to settle while the total is not. Refused by name where the
        # count is explicit, and where any district of the subject is marked `exact`.
        explicit = (spec.get("explicit_count") or {})
        mine = [d for d in place.get("districts") or []
                if d.get("defines") == subject or d.get("name") == subject]
        if explicit.get("n") and not explicit.get("about"):
            rec.update(refused=True, invariant="count",
                       why=(f"this place's count is the sentence's own "
                            f"({explicit.get('n')} exactly, `explicit_count`), so how it "
                            f"is divided between districts is not an inferred allocation: "
                            f"`redistribute` is refused by name. The layout's actions on "
                            f"an exact count are the four arrangement actions and "
                            f"`resize_ring`"))
            return place, rec
        if any(d.get("exact") for d in mine):
            rec.update(refused=True, invariant="count",
                       why=(f"{sum(1 for d in mine if d.get('exact'))} of "
                            f"{len(mine)} district(s) of `{subject}` carry an exact count, "
                            f"which the compiler lays as stated; `redistribute` is "
                            f"refused by name rather than moving a number somebody else "
                            f"declared"))
            return place, rec
        if len(mine) < 2:
            rec.update(refused=True,
                       why=(f"`{subject}` has {len(mine)} district(s): there is nowhere "
                            f"to move population to. A single district's number is a "
                            f"scale decision (`structures`), not a distribution"))
            return place, rec
        # from the district that could not hold what it was promised, to the one with
        # the most room left at its own density -- both read off the layout's own record
        def _short_of(d):
            got = (d.get("target") or {}).get("value") or {}
            return max(0, int(d.get("structures") or 0) - int(got.get("houses") or
                                                              d.get("structures") or 0))
        def _room(d):
            band = d.get("count_band") or {}
            return int(band.get("hi") or 0) - int(d.get("structures") or 0)
        donor = max(mine, key=lambda d: (_short_of(d), -_room(d),
                                         int(d.get("structures") or 0), d["name"]))
        taker = max((d for d in mine if d is not donor),
                    key=lambda d: (_room(d), -int(d.get("structures") or 0), d["name"]))
        move = max(1, int(round(REDISTRIBUTE_SHARE * int(donor.get("structures") or 0))))
        move = min(move, max(0, int(donor.get("structures") or 0) - 1),
                   max(0, _room(taker)))
        if move <= 0:
            rec.update(refused=True,
                       why=(f"no district of `{subject}` has room for more houses at its "
                            f"own density's ceiling: `{taker['name']}` is at "
                            f"{taker.get('structures')} of a band top of "
                            f"{(taker.get('count_band') or {}).get('hi')}. The count is "
                            f"the limit and not the distribution"))
            return place, rec
        was = {d["name"]: int(d.get("structures") or 0) for d in mine}
        to = {"structures": {**{k: v for k, v in
                                ((current.get("structures") or {}).items())},
                             donor["name"]: was[donor["name"]] - move,
                             taker["name"]: was[taker["name"]] + move}}
        why = (f"the built reading finds `{donor['name']}` unable to hold the "
               f"{was[donor['name']]} house(s) it was promised while `{taker['name']}` has "
               f"room to its density's ceiling of "
               f"{(taker.get('count_band') or {}).get('hi')}; {move} inferred house(s) "
               f"move from the first to the second. The hard constraints this preserves: "
               f"the place's **total** count stays "
               f"{sum(int(d.get('structures') or 0) for d in place.get('districts') or [])} "
               f"and no explicit or exact count is touched -- this action refuses itself "
               f"where one exists")
        rec["redistributed"] = {"from": donor["name"], "to": taker["name"],
                                "houses": int(move), "was": was}
    elif action == "resize_ring":
        ring = subject
        lay = next((r for r in ((place.get("layout") or {}).get("rings") or [])
                    if r.get("name") == ring), None)
        if not lay:
            rec.update(refused=True, why=f"no ring named `{ring}` on this place")
            return place, rec
        tgt = finding.get("target") or {}
        direction = str(tgt.get("direction") or "down")
        have = float((current.get("rings") or {}).get(ring, {}).get("width")
                     or lay.get("width") or 0)
        new = int(round(have * RING_STEP.get(direction, 0.7)))
        new = max(int(lay.get("min_width") or 0), new)
        if new == int(have):
            rec.update(refused=True,
                       why=f"ring `{ring}` is {int(have)} wide, at its least width of "
                           f"{lay.get('min_width')}: it cannot be narrower")
            return place, rec
        to = {"rings": {ring: {"width": new}}}
        why = (f"ring `{ring}` is {int(have)} wide and the built reading asks for it "
               f"{direction}; the width is set to {new} and the rings laid out again")
    # **A revision does not re-decide what it is not revising.** The composition round,
    # and it cost a whole improvement cycle before it was found: the layout measures
    # which side the city's gates stand on from the ground under them, this site is flat
    # plains so the four sides score within a whisker of each other, and a re-solve
    # whose ring widths moved by a few columns measured the other side. The palace's
    # gate then stood where no road arrived, the compound validator refused the place --
    # correctly -- and **every one of the layout owner's actions was refused with the
    # same message**, because the refusal had nothing to do with the action. The place
    # being revised is the record of what it has already decided; a choice this action
    # is not about is carried into the re-solve rather than measured again.
    carried = {k: v for k, v in (("axis_side",
                                  (place.get("layout") or {}).get("axis_side")),)
               if v is not None and current.get(k) is None and k not in to}
    # re-solve from the spec with the allocation, deterministically
    spec2 = dict(spec)
    spec2["negotiated"] = list(spec.get("negotiated") or []) + (
        [{"what": "allocation", "from": {}, "to": carried,
          "why": "carried from the place being revised: a revision does not re-decide "
                 "a layout choice it is not about",
          "finding": finding.get("id"), "action": action}] if carried else []) + [
        {"what": "allocation", "from": current, "to": to, "why": why,
         "finding": finding.get("id"), "action": action}]
    core = spec_mod.core(spec) or {}
    if plateau is None and (place.get("layout") or {}).get("plateau_rect"):
        plateau = {"part": core.get("name"), "rect": list(place["layout"]["plateau_rect"])}
    elif plateau is None and (place.get("layout") or {}).get("compound_rect") \
            and spec_mod.compound(core):
        plateau = {"part": core.get("name"), "rect": list(place["layout"]["compound_rect"])}
    # **A revision is laid out with what this place was approved for.** Read off the
    # place itself (`approved_from_place`) and not from the caller: the improve stage
    # has no capability record to hand in, and without one the re-solve resolved its
    # demand against the general library and came back with a village of cottages built
    # of halls -- and, at a hall's lot, with ten of the sentence's sixteen.
    approved, demands = approved_from_place(place)
    got, fails = solve_place(spec2, site, plateau, decls, place.get("voice") or "",
                             vol=vol, seed=int(seed if seed is not None else
                                                ((place.get("layout") or {}).get("seed") or 1)),
                             caps=caps, intent=intent, approved=approved,
                             demands=demands,
                             envelope_cache=(envelopes if isinstance(envelopes, str) else
                                             (place.get("layout") or {}).get("envelope_cache")))
    if got is None:
        rec.update(refused=True, allocation=to,
                   why=f"laid out again with {to}, the place fails its own layout: "
                       + "; ".join(f"{f.get('part')}: {f.get('why')}" for f in fails[:3]))
        return place, rec
    # **Count-preserving reallocation is an invariant, asserted here and not only in a
    # test.** The composition round's fifth change, last clause. Every action above is a
    # revision of an *inferred* dimension, and the re-solve derives counts from the
    # geometry it produces -- so a narrower ring or a bigger anchor can, in principle,
    # arrive with fewer counted structures than the sentence stated. Where the count is
    # the sentence's own that is not a trade the layout owner may make, and until now
    # nothing checked: the place came back, the rectangles had moved, and the number was
    # whatever the arithmetic gave. An action that would lose a counted structure under
    # an exact count is refused, by name, with both numbers.
    exact_count = bool((spec.get("explicit_count") or {}).get("n")
                       and not (spec.get("explicit_count") or {}).get("about"))
    was_n = sum(int(d.get("structures") or 0) for d in place.get("districts") or [])
    now_n = sum(int(d.get("structures") or 0) for d in got.get("districts") or [])
    if exact_count and now_n != was_n:
        rec.update(refused=True, invariant="count",
                   counts={"was": int(was_n), "now": int(now_n)}, allocation=to,
                   why=(f"laid out again with {to} the place holds {now_n} counted "
                        f"structure(s) against {was_n}, and this sentence states its count "
                        f"exactly ({(spec.get('explicit_count') or {}).get('n')}): a "
                        f"reallocation revises an inferred dimension and may not change a "
                        f"number the request made. `{action}` is refused and the place "
                        f"stands as it was"))
        return place, rec
    before, after = _rects_of(place), _rects_of(got)
    moved = {n: {"from": before.get(n), "to": after.get(n)}
             for n in sorted(set(before) | set(after)) if before.get(n) != after.get(n)}
    counted = {d.get("name"): int(d.get("structures") or 0)
               for d in got.get("districts") or []}
    was_by = {d.get("name"): int(d.get("structures") or 0)
              for d in place.get("districts") or []}
    recounted = {n: [was_by.get(n), counted[n]] for n in sorted(counted)
                 if was_by.get(n) != counted[n]}
    if not moved and not recounted:
        # **`changed: False`, honestly.** An action that ran and produced the same
        # geometry says so; it is not reported as a refusal of the action's premise and
        # it is not repeated.
        rec.update(refused=True, changed=False, allocation=to,
                   why=f"laid out again with {to} the place is the same geometry and the "
                       f"same counts: the action changed nothing and is not repeated")
        return place, rec
    spec.setdefault("negotiated", []).append(spec2["negotiated"][-1])
    # **Every action states the finding's subject, the direction it moved and the hard
    # constraints it preserved**, as fields and not only inside a `why` string -- the
    # round's requirement, and the difference between a record a reader can check and a
    # sentence they have to trust. `preserves` carries only what was **measured** across
    # the re-solve; a constraint this level cannot see is not listed.
    kept = [p.get("name") for p in place.get("parts") or []
            if p.get("name") and before.get(p["name"]) == after.get(p["name"])]
    rec.update(applied=True, changed=True, allocation=to, why=why,
               districts=[{"district": n, "from": m["from"], "to": m["to"]}
                          for n, m in moved.items()],
               moved=len(moved),
               counts={"was": int(was_n), "now": int(now_n), "exact": bool(exact_count),
                       "per_district": recounted or None},
               subject_of_finding=str((finding.get("subjects") or [subject])[0]),
               # the direction the finding asked for, where it named one; the anchor
               # action sets its own above and this does not overwrite it
               direction=(rec.get("direction")
                          or (str((finding.get("target") or {}).get("direction")).lower()
                              if (finding.get("target") or {}).get("direction") else None)),
               preserves={
                   "total_count": int(was_n),
                   "count_is_exact": bool(exact_count),
                   "exact_districts": sorted(
                       str(d.get("name")) for d in got.get("districts") or []
                       if d.get("exact")),
                   "parts_unmoved": sorted(str(n) for n in kept),
                   "why": (f"the place holds {now_n} counted structure(s), the same "
                           f"{was_n} it held; "
                           + (f"{len(kept)} standing part(s) are where they were; "
                              if kept else "")
                           + f"the action revised {to and list(to)} and nothing else")})
    return got, rec


# ------------------------------------------------------------ wall hierarchy, voice

#: **What kind of wall a boundary is, and how tall**, derived from the request. The
#: expression round: the place read held every concentric place's outermost wall to a
#: great-wall constant of thirty, and a small walled hill town -- whose sentence said
#: `walled` and nothing about a great wall -- failed for it. A wall is `great` only
#: where the request or the sourced claims call it one (`GREAT_WALL_WORDS` in the spec's
#: own words of it, or a claim about it); it is `compound` inside a compound; it is
#: `town` otherwise. A great wall stands `GREAT_OVER` times the tallest other ring wall
#: of its place, at the least, so hierarchy is relative and read off the design; a town
#: wall stands over the storeys of the fabric it encloses by a parapet
#: (`WALL_OVER_STOREYS`), clamped into its type's band. Every wall carries its
#: `hierarchy` record with the inputs by name, and the reader's clause is asked of that
#: record.
GREAT_WALL_WORDS = ("great wall", "greatest wall", "great outer wall", "largest structure",
                    "largest man-made", "world's largest", "greatest structure")
GREAT_OVER = 1.5
WALL_OVER_STOREYS = 4       # a parapet and a walk over the eaves of the tallest house
STOREY_PITCH = 4


def wall_kind_for(part: dict | None, spec: dict | None, *, outermost: bool = False,
                  claims: list | None = None) -> tuple:
    """`(kind, from)`: `great`, `town` or `compound` for this wall part, and why."""
    text = " ".join([str((part or {}).get("name") or "").replace("_", " "),
                     str((part or {}).get("family") or "").replace("_", " "),
                     str((part or {}).get("notes") or ""),
                     str((spec or {}).get("invariants") or ""),
                     str((spec or {}).get("sentence") or "")]).lower()
    # a clause that denies it ("not a great wall") is not a clause that asks for it
    said = []
    for clause in re.split(r"[.;,:]", text):
        if any(neg in f" {clause} " for neg in (" not ", " no ", " never ", " nothing ",
                                                 " rather than ", " unlike ")):
            continue
        said += [w for w in GREAT_WALL_WORDS if w in clause and w not in said]
    cited = [c.get("id") for c in (claims or [])
             if any(w in str(c.get("says") or "").lower() for w in GREAT_WALL_WORDS)]
    if outermost and (said or cited):
        return "great", ([f"the spec's words of it: {', '.join(said)}"] if said else []) \
            + [f"claim {c}" for c in cited]
    if (part or {}).get("compound"):
        return "compound", ["a wall of a compound"]
    return "town", ["the request asks for a walled place and no great wall: a town wall"]


def wall_height_for(kind: str, spec: dict | None, decl: dict | None, *,
                    others: list | None = None, storeys: int | None = None) -> tuple:
    """`(height, from)` for a wall of this kind out of this type's band."""
    h = (decl or {}).get("params", {}).get("height") or ("int", 3, 20)
    lo, hi = int(h[1]), int(h[2])
    if kind == "great":
        tallest = max([int(v) for v in (others or []) if v is not None] or [0])
        want = int(math.ceil(GREAT_OVER * tallest)) if tallest else lo
        want = max(want, lo)
        src = [f"GREAT_OVER {GREAT_OVER:g} x the tallest other ring wall ({tallest})"
               if tallest else f"no other wall to stand over: the least of the type's "
                               f"band, {lo}"]
        return max(lo, min(hi, want)), src + [f"clamped into `{h[1]}..{h[2]}`"]
    st = int(storeys or 2)
    want = st * STOREY_PITCH + WALL_OVER_STOREYS
    src = [f"{st} storey(s) of fabric x {STOREY_PITCH} + WALL_OVER_STOREYS "
           f"{WALL_OVER_STOREYS} = {want}", f"clamped into `{h[1]}..{h[2]}`"]
    return max(lo, min(hi, want)), src


def _fabric_storeys(spec: dict) -> int:
    top = 0
    for p in spec.get("defining_parts") or []:
        st = (p.get("character") or {}).get("storeys")
        if st:
            top = max(top, int(st[-1]))
    return top or 2


def _write_hierarchy(place: dict, spec: dict, decls: dict) -> None:
    """Every wall of the place level carries `hierarchy` -- kind, rank, height, its
    inputs -- and every wall and gate the voice of the ring or place it bounds unless
    the design records an override. The reader's wall clause reads this record."""
    parts = place.get("parts") or []
    walls = [p for p in parts if p.get("kind") == "edge"
             and (decls.get(p.get("type")) or {}).get("kind") == "edge"
             and not (decls.get(p.get("type")) or {}).get("passage")]
    if not walls:
        return
    # rank by the area each closes, outermost first
    def _area(p):
        path = [(int(a[0]), int(a[1])) for a in (p.get("path") or [])]
        if len(path) < 3:
            return 0.0
        s = 0.0
        for (x0, z0), (x1, z1) in zip(path, path[1:] + path[:1]):
            s += x0 * z1 - x1 * z0
        return abs(s) / 2.0
    ranked = sorted(walls, key=lambda p: -_area(p))
    spec_part = {p["name"]: p for p in spec.get("defining_parts") or []}
    for k, p in enumerate(ranked):
        if p.get("hierarchy"):
            continue
        d = spec_part.get(p.get("defines")) or {}
        kind, src = wall_kind_for(d, spec, outermost=(k == 0))
        h = (p.get("params") or {}).get("height")
        p["hierarchy"] = {"kind": kind, "rank": k, "of": len(ranked),
                          "height": int(h) if h is not None else None,
                          "from": src + [f"built at {h} as `{p.get('type')}`"]}
    place_voice = place.get("voice")
    by_name = {p.get("name"): p for p in parts}
    for p in parts:
        if p.get("kind") not in ("edge", "point"):
            continue
        if p.get("voice_override"):
            p["voice_from"] = "override"
            continue
        if p.get("voice"):
            p.setdefault("voice_from", "ring" if p.get("ring") is not None else "place")
            continue
        # a gate takes its wall's voice; a wall with none takes the place's
        host = None
        if p.get("kind") == "point":
            host = next((w for w in walls if w.get("ring") == p.get("ring")
                         and p.get("ring") is not None), None) \
                or next((w for w in walls
                         if str(p.get("notes") or "").find(str(w.get("name"))) >= 0), None)
        v = (host or {}).get("voice") or place_voice
        if v:
            p["voice"] = v
            p["voice_from"] = "wall" if host is not None and host.get("voice") else "place"
