"""This is the top of the plan layer. One model call with a fixed schema turns the sentence
into `place.json`, and from there everything is arithmetic:

    kind            hamlet | village | town | city | keep | monument
    size_band       how many structures, from the kind -- **fixed here**, and overridden
                    by an explicit number in the sentence
    defining_parts  what makes it that place and not a heap of buildings: three
                    concentric walls, a palace at the centre, a market square. Each names
                    a kind of part, a family of type, its multiplicity and its relation.
    voice           named if the sentence names one, **authored by the model** where it
                    asks for a palette no voice on disk is, otherwise chosen against the
                    ground
    form            which family of form the place is built in, and what the plan filters
                    its types by
    needs           the ground it wants: footprint, relief, water, forest

The model decides `kind`, `defining_parts`, `voice` and the explicit count if the
sentence carries one. **Every number derived from those is this module's**, for the
reason `Builder.pad_extent` exists: the arithmetic between two layers belongs to one of
them, and a model asked to do it will do it differently each time."""
from __future__ import annotations

import json
import math
import re

#: The kinds of place a sentence may ask for.
KINDS = ("hamlet", "village", "town", "city", "keep", "monument")

#: The four settlement kinds are the spec's own numbers and are not ours to move. a few
#: parts, not a settlement -- and a monument is one thing and its setting.
SIZE_BANDS = {
    "hamlet": (5, 12),
    "village": (12, 40),
    "town": (40, 120),
    "city": (120, 400),
    "keep": (3, 12),
    "monument": (1, 4),
}


def kind_order() -> list:
    """The kinds of place from the smallest to the largest, read off `SIZE_BANDS` --
    the one table -- and not declared again anywhere. v2, C0: the site search's last
    escape drops a place one band down, and it kept its own list of the order."""
    return sorted(SIZE_BANDS, key=lambda k: (SIZE_BANDS[k], k))


def kind_below(kind: str) -> str | None:
    """The kind one size band down from this one, or None at the bottom of the table
    (and for a kind the table does not have)."""
    order = kind_order()
    if kind not in order or order.index(kind) == 0:
        return None
    return order[order.index(kind) - 1]


#: What "about sixty" means, either side. The spec's own bar reads "about sixty: 48-72",
#: which is this fraction, and it is written here rather than in a round file so that a
#: sentence and the band it produces cannot drift apart.
ABOUT = 0.2

#: The kinds of part a defining part may be, which are the library's four plus `group`
#: for a defining part that is a division of the place rather than a thing in it -- "and
#: districts by ring" is a group, and it becomes areas at plan time.
PART_KINDS = ("plot", "edge", "point", "area", "group")

#: What a defining part may ask to be made of, as a family rather than a type name: the
#: spec is written before the site is chosen and before a type is authored, and a spec
#: that named `townhouse` would be a spec that could not ask for a keep.
FAMILIES = ("wall", "gate", "square", "keep", "palace", "hall", "house", "workshop",
            "temple", "tower", "district", "quarter", "bridge", "monument")

#: The families a `group` defining part is a **division of the place** for: the ground
#: left over, drawn as rectangles at the place level and filled with houses by a call
#: per district.
DISTRICT_FAMILIES = ("district", "quarter")

#: The families that are a compound by the word: a palace is "very nearly a city in
#: itself" -- an axis, an inner wall, gates, halls, courts -- and a monument is one
#: thing and the precinct that sets it. The place level draws its rectangle and a call
#: of its own plans the parts inside it out of committed types, exactly as a district is
#: planned. A castle, a monastery or a cathedral close is the same shape and is a
#: compound when the spec says `group` of its family -- a keep is a building unless the
#: sentence makes it a castle. See `compound`.
COMPOUND_FAMILIES = ("palace", "monument")

#: **What a compound of each family is made of, at the least -- by the word.** v2, C0.
#: Until this, every compound was palace-shaped: a closed wall of its own with a gate on
#: it, two halls and a court were the validator, the brief, the place read's clause and
#: the plateau arithmetic for any compound family, `monument` included, and every
#: compound was granted the defensive types. A monument is one thing and its setting; a
#: shrine precinct is a shrine and its court; a castle is a keep inside its own curtain
#: wall with a gatehouse and a bailey. Each row says whether the compound has a closed
#: wall of its own (`walled`), a gate on it (`gated`), the least enclosed parts
#: (`halls`: plots) and open parts (`courts`: areas) it holds, and the roles it admits
#: over its own (`admits`: a wall and a gate are defensive whatever the compound is for,
#: so a walled compound admits them). The count of halls is a **floor** where the row
#: `scales`: the compound's rectangle raises it at the compound's density
#: (`placeplan.compound_target`), so a palace on two hundred square is not two halls and
#: a court. A monument does not scale -- it is one thing whatever the size of its
#: setting. A wall drawn where none is asked for is still held to be closed and inset,
#: and a gate drawn is held to stand on it. A family not in the table is the default
#: row, which is the palace's. ...and **`axis`**, the craft round (E5): whether a
#: compound of this family is a **sequence** and not a set. A composition says what a
#: great thing holds and said nothing about the order, so a palace precinct a quarter of
#: a city wide came out as halls and courts arranged to fit rather than as an approach.
#: Where a family declares an axis the library lays it as one -- the gate, a forecourt,
#: a hall, an inner court and the greatest hall at the far end, the flanking ranges
#: paired down it -- and where it does not, the compound is drawn as it always was. A
#: palace, a shrine precinct and a castle are approaches; a monument is one thing and
#: its setting, and has none.
COMPOSITIONS = {
    "palace":   {"walled": True,  "gated": True,  "halls": 2, "courts": 1,
                 "scales": True, "admits": ("defensive",), "axis": True},
    "monument": {"walled": False, "gated": False, "halls": 1, "courts": 1,
                 "scales": False, "admits": (), "axis": False},
    "temple":   {"walled": False, "gated": False, "halls": 1, "courts": 1,
                 "scales": True, "admits": (), "axis": True},
    "keep":     {"walled": True,  "gated": True,  "halls": 1, "courts": 1,
                 "scales": True, "admits": ("defensive",), "axis": True},
    "tower":    {"walled": True,  "gated": True,  "halls": 1, "courts": 1,
                 "scales": True, "admits": ("defensive",), "axis": False},
    "hall":     {"walled": False, "gated": False, "halls": 2, "courts": 1,
                 "scales": True, "admits": (), "axis": True},
    "house":    {"walled": False, "gated": False, "halls": 2, "courts": 1,
                 "scales": True, "admits": (), "axis": False},
    "workshop": {"walled": False, "gated": False, "halls": 2, "courts": 1,
                 "scales": True, "admits": (), "axis": False},
}
COMPOSITION_DEFAULT = COMPOSITIONS["palace"]


def composition(part_or_family) -> dict:
    """What a compound of this family is made of, at the least: one `COMPOSITIONS` row,
    copied, with `family` on it. Takes a defining part or a family name; `None` and a
    family not in the table are the default row."""
    fam = part_or_family.get("family") if isinstance(part_or_family, dict) \
        else part_or_family
    row = COMPOSITIONS.get(fam) if fam else None
    out = dict(row or COMPOSITION_DEFAULT)
    out["admits"] = tuple(out["admits"])
    out["family"] = fam or "compound"
    return out

#: The families of form a place is built in. A type declares one of these and the plan
#: filters on it, so a place says which building tradition it is in and the palette is a
#: separate question answered against the ground. Two are regional and two are
#: functional -- see `pipeline.form_ok`.
FORMS = ("european_vernacular", "east_asian", "fortification", "civic")

#: How a defining part sits in the place. Free text is refused: a relation nothing can
#: act on is a comment, and A5 plans the place level off exactly these words.
RELATIONS = ("concentric", "centre", "edge", "perimeter", "gateway", "throughout",
             "quarter", "beside_the_centre",
             # v2, B2: the three that name another part -- `of` -- and the solver places
             # against it: `near` (on a side of it), `along` (a strip beside an edge's
             # run), `on` (a point at a cell of an edge's line). `near` with no `of` is
             # `beside_the_centre`.
             "near", "along", "on")

#: The relations that may name the part they are placed against, in `of`.
RELATIONS_OF = ("near", "along", "on")

#: How thickly a defining part that holds structures is built up. "large courtyard
#: houses, gardens, low density" and "small courtyard houses and workshops, dense" are
#: the same sentence with this word changed. A multiplier on `COLUMNS_PER_PLOT`, so a
#: sparse ring is drawn with more ground per structure and a dense one with less.
#: **Bounded by what a plan can actually be checked against.** `DISTRICT_FILL` is 0.55
#: and the circulation pass needs five blocks between footprints, so a density that made
#: the ground per structure smaller than a plot would produce districts that pass this
#: level and are handed back by the next one.
DENSITIES = {"sparse": 2.0, "low": 1.5, "medium": 1.0, "dense": 0.7}

#: **How much of a district's tiled ground goes to plots rather than to areas**, per
#: density word. `placeplan.occupancy_shares()` turns it into the share of a district
#: that is plots and the share that is plots **and** areas, using the library's own plot
#: and area sizes and the five blocks the circulation needs between footprints. This is
#: what a density word actually is. A dense quarter is mostly built on and its open
#: ground is cut between the rows; a sparse one is mostly not, and what it is not built
#: on is fields, groves and gardens. Neither statement is about *cover*: a dense
#: district ends up covered **less** than a sparse one, because at a house's scale the
#: five-block street between two footprints is a third of the ground and at a field's
#: scale it is a tenth. That is a fact about tiling and it is why these two numbers are
#: one derivation and not two opinions -- registered separately, `PLOT_SHARE` 0.42 and
#: `GROUND_COVER` 0.70 for `dense` asked a planner for 400 plots of ten and 168 areas of
#: twelve in a 300-square district, which needs 133,602 columns of a rectangle that has
#: 90,000. Nothing would have found that but drawing it.
PLOT_CELLS = {"sparse": 0.30, "low": 0.45, "medium": 0.60, "dense": 0.75}

#: What a defining part is **for**, which is a different question from what family of
#: form it is built in. Nothing in the stack could say that was wrong, because nothing
#: in the stack knew what a minka is for.
ROLES = ("urban", "rural", "civic", "defensive")

#: Which family of defining part carries which role where the spec does not say. A spec
#: call may write `role` and this is what is used when it does not.
_ROLE_OF_FAMILY = {"wall": "defensive", "gate": "defensive", "tower": "defensive",
                   "keep": "defensive", "house": "urban", "workshop": "urban"}

#: What a **defining part** may say about the ground it wants, over and above what the
#: place says. A place with three concentric rings does not want one number for its
#: ground. The palace compound is the plateau and wants a level square; the outer ring
#: is farmland on a hillside and wants nothing of the sort; the wall between them climbs
#: whatever is there. One `max_relief` over 512x512 cannot say any of that, and the
#: record says what happens when it tries -- `MAX_GRADIENT`'s own docstring is a filter
#: that admitted nothing. Every field is optional and `None` means "the place's own need
#: applies here".
PART_NEEDS_DEFAULT = {"max_relief": None,      # over this part's own ground
                      "plateau": None,         # the level square this part stands on
                      "max_water_pct": None,
                      "max_forest_pct": None}

#: A district part may say which ring of the place it is, counting outward from the
#: centre; what **share** of the place's whole area it is; whether a **wall** bounds it
#: on its outside; and which **voice** it is built in. From those four the layout is
#: arithmetic (`placeplan.concentric_layout`): the ring rectangles from the cumulative
#: shares, walls at the walled boundaries, one gate per walled ring on one axis, and
#: districts tiling every annulus. The demo's farm belt came out at 24% districts and
#: the rest forest, because nothing could say "half the place". Only a district part
#: carries these. `share` is a fraction of the whole footprint's area; the rings' shares
#: sum to at most one and **the remainder is the centre's**. `voice` takes the three
#: forms the place's voice takes -- a name, null for the place's, or an authored voice
#: -- and an authored one is validated by the voice validator exactly as the place's is.
RING_FIELDS = ("ring", "share", "walled", "voice")

#: **A district's character**, v2, C1: what a district is like, in words and a few
#: numbers, which the district compiler (`ethoslm.district_compile`) turns into streets,
#: blocks, lots and buildings with no model asked. The model writes this in the district
#: brief's place. Every field is optional; the density word's row of
#: `CHARACTER_DEFAULTS` fills what is not said. frontage `street`: lots front the
#: streets, doors on them, a clearance apart; `open`: freestanding buildings a lane
#: apart, the way in wherever the lanes arrive. block the block's length along its
#: street, in columns. lot_width how wide a lot's frontage is, in columns. Declared, it
#: is the width the lots are laid at -- clamped into what the type admits and never
#: grown by the cover lever, which is `lot_depth`'s rule. lot_depth how deep a lot runs
#: back from its street, in columns. attached true: lots in a row touch, party wall to
#: party wall. courtyard_share the share of blocks whose back row is a court the front
#: row shares, 0 to 1. open_share the share of blocks left as open ground -- fields,
#: gardens, groves, a plaza, by the role and the density -- 0 to 1. landmarks `[{"type":
#: name, "notes": ...}]`: a fixed landmark, placed on the block nearest the district's
#: middle with a plaza about it. variety how far a lot's width and depth may stray from
#: the density's own, 0 to 1 of the lot side; omit it and the frontage decides
#: (`district_compile.VARIETY`). storeys `[lo, hi]`: the band a lot's building takes its
#: storeys from, clamped into what each type declares. Omit it and the density's own
#: band is used. A run of roofs steps rather than lying flat. `lot_width` is the
#: realization round's, and it is the field an inspection asked for and could not have.
#: The shore village came back as six buildings twenty-four columns wide -- a barn's
#: frontage -- because the compiler grows the lot to make its district cover and a
#: character could say how *deep* a lot is and never how *wide*. The reading asked for
#: "more, smaller houses"; the only lever that existed made them shallower and the cover
#: fell, so the revision was rolled back for doing what it was asked. A declared width
#: is the compiler's to clamp into what the type admits and not to grow, which is the
#: rule `lot_depth` has always had.
CHARACTER_FIELDS = ("frontage", "block", "lot_width", "lot_depth", "attached",
                    "courtyard_share", "open_share", "landmarks", "variety", "storeys")

FRONTAGES = ("street", "open")

#: What a district of each density word is like when the spec says only the word.
#: `block` and `lot_depth` are None here because they are the density's own lot
#: (`placeplan.occupancy_shares`: three lots a block, a lot deep) and the compiler fills
#: them. `variety` is None here because the **frontage** decides it and a character may
#: override the frontage; `storeys` is the band a quarter of that density is built to,
#: clamped into what each type declares -- a farm belt is low and a dense quarter builds
#: upward, and either way a street of one height is a wall and not a skyline.
CHARACTER_DEFAULTS = {
    "sparse": {"frontage": "open", "block": None, "lot_width": None,
               "lot_depth": None, "attached": False,
               "courtyard_share": 0.0, "open_share": 0.5, "landmarks": [],
               "variety": None, "storeys": [1, 2]},
    "low":    {"frontage": "open", "block": None, "lot_width": None,
               "lot_depth": None, "attached": False,
               "courtyard_share": 0.0, "open_share": 0.3, "landmarks": [],
               "variety": None, "storeys": [1, 3]},
    "medium": {"frontage": "street", "block": None, "lot_width": None,
               "lot_depth": None, "attached": False, "courtyard_share": 0.15, "open_share": 0.15,
               "landmarks": [], "variety": None, "storeys": [2, 3]},
    "dense":  {"frontage": "street", "block": None, "lot_width": None,
               "lot_depth": None, "attached": False, "courtyard_share": 0.25, "open_share": 0.05,
               "landmarks": [], "variety": None, "storeys": [1, 3]},
}

#: The words `setting.relief` may be, and the **fall per block of footprint** each one
#: caps the whole footprint at. Registered from the record before any search ran on
#: them, and from nothing else: flat 0.25. The demo city stood on 178 over 512 (0.35)
#: and a reviewer called it stepped terraces and a hilly city on a plain; 0.25 refuses
#: it. rolling 0.55. any no cap beyond the place's own. The word is the model's; the
#: number is this module's; the cap is folded into the place's `needs.max_relief` where
#: it is tighter than what is there. steep no cap of its own -- see `RELIEF_FLOORS`. a
#: mountain village, a cliff keep -- could not ask for one. `steep` is a floor on relief
#: with a preference for high ground, and a mountain village, a cliff keep and a plain
#: capital are the same mechanism asked three ways.
RELIEF_WORDS = {"flat": 0.25, "rolling": 0.55, "steep": None, "any": None}

#: `steep` is `MAX_GRADIENT`'s own 0.55: the line `rolling` caps at is the line steep
#: ground starts at, so no ground is both. The floor is folded into the place's needs as
#: `min_relief`, and a candidate under it does not meet the needs.
RELIEF_FLOORS = {"steep": 0.55}

#: ...and the cap a `steep` footprint is held to, per block, in place of the place's own
#: `MAX_GRADIENT`: a number rather than none so every need stays a number, and 1.5 per
#: block over any footprint this project builds is more fall than the world's height
#: allows, so it refuses nothing the floor admits.
STEEP_CAP = 1.5

#: Two per cent, the same figure a "no water" setting caps a whole footprint at
#: (`WATER_NONE_PCT`): the core of an 83x83 compound is 6,889 columns and two per cent
#: of it is a pond of 138, which the plateau fills from the bed. The demo's site read
#: 15.3% water over the 83x83 at its centre and the search never asked.
CORE_WATER_MAX_PCT = 2.0


class SpecError(ValueError):
    """This is not a place spec, and here is the field that says so.

        `field` and `part` name what was refused where the refusal is about one field of
        one part (or one top-level field), so a hand-back can take that field alone from
        the next answer (`merge_hand_back`). None where the refusal is about the whole.
        
    """

    def __init__(self, msg: str, field: str | None = None, part: str | None = None):
        super().__init__(msg)
        self.field = field
        self.part = part


#: The fields of a defining part a hand-back may replace one at a time.
PART_FIELDS = ("kind", "family", "relation", "of", "count", "structures", "needs", "forms",
               "density", "role", "ring", "share", "walled", "voice", "character",
               "notes")


def merge_hand_back(first: dict, answer: dict, fields: list) -> dict:
    """The first answer with only the refused `fields` taken from the new `answer`."""
    import copy
    out = copy.deepcopy(first)
    if not fields:
        return copy.deepcopy(answer)
    by_name = {p.get("name"): p for p in (answer.get("defining_parts") or [])
               if isinstance(p, dict)}
    for f in fields:
        if f in PART_FIELDS:
            for p in out.get("defining_parts") or []:
                q = by_name.get(p.get("name"))
                if q is not None and f in q:
                    p[f] = copy.deepcopy(q[f])
                elif q is not None and f in p:
                    del p[f]
        elif f in answer:
            out[f] = copy.deepcopy(answer[f])
        elif f in out:
            del out[f]
    return out


# ---------------------------------------------------------------- the ceiling

#: A2. What one run of this pipeline may be. Hard limits, in one place, checked before a
#: site is looked for and applied to the spec rather than to the plan -- a place that is
#: scaled after its site is chosen has already spent the search. footprint the largest
#: square of world one place may occupy structures the most things one place may stand
#: up sessions how many server sessions a run may take. One: the sandbox gives every
#: shell its own network namespace, so a server dies with the call that started it, and
#: a place that needs two sessions cannot be built here.
CEILING = {"footprint": 512, "structures": 400, "sessions": 1}

#: A city at 768, every other kind at `CEILING["footprint"]`. The principal's decision
#: for the last run -- it goes for it all -- and a general fact: a city of four rings on
#: 512 had its belt squeezed to 0.44 of the site against 0.58 declared because the inner
#: rings' least widths took what the shares could not give; at 768 the shares fit. Every
#: reader asks `footprint_ceiling(kind)`; a spec read before this carried the old
#: ceiling's footprint and is read back as it was (`_read_needs`).
CEILING_FOOTPRINT_BY_KIND = {"city": 768}


def footprint_ceiling(kind: str | None) -> int:
    """The largest square of world a place of this kind may occupy."""
    return int(CEILING_FOOTPRINT_BY_KIND.get(kind or "", CEILING["footprint"]))


def structures_ceiling(kind: str | None) -> int:
    """The most things a place of this kind may stand up, **scaled by its ground and by
        the fabric it is built at**.

        The craft round, E1, and the same argument a third time: the other half of that
        statement is **how much ground one structure takes**, and 400 on 512 square is 655
        columns a structure. The compiler lays a medium district at 323. A ceiling left at
        the old fabric clips the count the new arithmetic gives -- a city of four rings on
        768 asks for about a thousand structures against a ceiling of 900 -- and a ceiling
        that clips an honest count is a ceiling deciding the answer.
        
    """
    return int(round(CEILING["structures"] * _ground_ratio(kind) * fabric_ratio()))


def _ground_ratio(kind: str | None) -> float:
    """How much more ground this kind is given than the ceiling it was registered on."""
    f = footprint_ceiling(kind)
    return (f * f) / float(CEILING["footprint"] ** 2)


#: The density word the size bands and `CEILING["structures"]` are re-expressed at: the
#: middle of the ladder, and the one `DENSITIES` is 1.0 at.
FABRIC_REFERENCE_DENSITY = "medium"


def fabric_reference() -> float:
    """The ground one structure takes in the fabric `CEILING` was registered at: 400
    structures on 512 square, all in."""
    return (CEILING["footprint"] ** 2) / float(CEILING["structures"])


def fabric_ratio() -> float:
    """How much denser the library's own fabric is than the one `SIZE_BANDS` and
        `CEILING["structures"]` were registered at. The craft round, E1.

        655 columns a structure is a freestanding house on a square plot with a lane on all
        four sides, which is the village this project measured a district on and the model
        every number here was written against. `placeplan.fabric` measures what the compiler
        actually lays -- a
        frontage, a depth and a share of the street -- and a medium district comes out at
        323. Read off the files, so a type added or a band closed moves it.
        
    """
    from .placeplan import fabric, DENSITY_ROLE
    w = FABRIC_REFERENCE_DENSITY
    per = float(fabric(w, DENSITY_ROLE.get(w))["columns_per_structure"])
    return fabric_reference() / per if per > 0 else 1.0


def size_band_for(kind: str | None) -> tuple:
    """How many structures this kind is, as a band, **on the ground it is given and at
        the fabric it is built at**.

        The craft round adds the fourth number in the same statement -- the ground one
        structure takes (`fabric_ratio`) -- for the same reason: a band written at 655
        columns a house refuses at 900 a city whose own arithmetic, at the fabric the
        compiler lays, asks for a thousand. Every kind's band moves under it, because the
        fabric is the library's and not the kind's.
        
    """
    lo, hi = SIZE_BANDS[kind]
    r = _ground_ratio(kind) * fabric_ratio()
    if r == 1.0:
        return (int(lo), int(hi))
    return (int(round(lo * r)), int(round(hi * r)))


# ----------------------------------------------------------------- the ground

#: How much ground one structure takes, all in: a 15x15 plot -- which is what a
#: townhouse turned out to need -- plus its share of the lane, the clearance every type
#: declares, and the open ground a place is made of. this is the figure for a site the
#: search has *chosen* rather than one a human picked.
COLUMNS_PER_STRUCTURE = 1024

#: ...and how much ground one structure's **plot** takes inside a district, which is a
#: smaller number and a different question. A townhouse needs an 11x11 pad, which is a
#: 15x15 plot, which is 225 columns. **The plot alone, and not the lane with it.** This
#: was 400 -- the plot plus its share of a five-block lane, 20x20 -- and
#: `placeplan.DISTRICT_FILL` then reserved a further 45% of the district for the lanes,
#: so the same street was counted twice and a district needed 727 columns a house. The
#: composition is `plot / DISTRICT_FILL`, and each of the two numbers has to be about
#: one thing.
COLUMNS_PER_PLOT = 225

#: ...and how much of a chosen site actually takes a plot. The rest is the lane network,
#: the squares, the wall's own run and the ground a search accepts because the rest of
#: the footprint is good.
BUILDABLE_FRACTION = 0.45

#: Footprints are a multiple of this, because `scripts/prepare_settlement.py` reduces
#: the site to a 12x12 grid of cells for the planner's briefing and a size that is not a
#: multiple of twelve does not reshape. Twelve, not "round to a hundred": the constraint
#: is real and this is where it is written down.
FOOTPRINT_STEP = 12

#: The smallest place worth looking for a site for. Below this the footprint is smaller
#: than one quarter and the search has nothing to discriminate on.
FOOTPRINT_MIN = 96


def footprint_for(structures: int, kind: str | None = None) -> int:
    """The square of ground a place of this many structures needs, in blocks, under
    the kind's ceiling (`footprint_ceiling`)."""
    n = max(1, int(structures))
    want = math.sqrt(n * COLUMNS_PER_STRUCTURE / BUILDABLE_FRACTION)
    step = FOOTPRINT_STEP * math.ceil(want / FOOTPRINT_STEP)
    return int(max(FOOTPRINT_MIN, min(footprint_ceiling(kind), step)))


#: What a place's `needs` may say about its ground, and what each defaults to. These are
#: the terrain bank's own measures (`scripts/terrain_bank.py`) so a site is scored on
#: the same numbers every fixture in this project is described by. **Relief is not
#: scale-free, and a fixed cap on it is a filter that admits nothing.** This was
#: `max_relief: 40` -- a flat number over the whole footprint -- and running A3's search
#: against it found **nothing within 1,536 blocks of the origin**: the flattest 372x372
#: anywhere in that disc has 49 blocks of relief, and water, forest and the plateau all
#: passed at every candidate. Then the record was read, and the number is refuted by it
#: outright: **every site this project has ever built a town on carries relief 46 to 99
#: over 192x192**. A cap of 40 would have refused all eight. Read what an instrument
#: *admits*, not only what it rejects. So the need is expressed the way the evidence is
#: -- as **fall per block of footprint** -- and the band comes from the record: those
#: eight sites span 0.24 to 0.52. `MAX_GRADIENT` is 0.55, which is "no steeper than
#: anything this project has built a town on", with a margin and no site fitted to. The
#: reading under the old number is kept.
MAX_GRADIENT = 0.55

#: Floats and not ints, so that reading a spec that has already been read gives the
#: identical spec: `float(40)` is `40.0` and `40 != 40.0` under `json.dumps`.
#: `max_relief` is `None` by default and is then derived from `MAX_GRADIENT` and the
#: footprint. A spec may still give one and it is then taken as written -- a sentence
#: that says "on a plain" is entitled to ask for a plain.
NEEDS_DEFAULT = {"footprint": None,      # filled from the structure count
                 "max_relief": None,     # ...and so is this: MAX_GRADIENT x footprint
                 "max_water_pct": 25.0,  # a place is not a lake
                 "max_forest_pct": 60.0,
                 "plateau": None}        # the flat square the innermost part wants


#: **The setting.** What a place stands *in*, as distinct from what it needs of the
#: ground's shape -- and the first thing a person sees of a rendered place. open
#: plains", and every word of that stayed prose: the needs list had relief, water,
#: canopy and a plateau in it and no word for what the ground is *made of*, so a scan of
#: 111 squares put the city on the one square that met those, which was red sand and
#: terracotta. A setting is two words the model may write and the library measures: the
#: **surface** the ground should predominantly be, in `groundread.SURFACES`' vocabulary,
#: and whether there should be **water** -- some, for a lake or a harbour; none, for a
#: desert fort; or no opinion. The model says which; every number below is this
#: module's.
SETTING_DEFAULT = {"surface": None, "water": None, "relief": None, "biome": None}

#: Half, as `SURFACE_SHARE` is, and for the same plain reading of "predominantly". Among
#: candidates that meet it, more of the list ranks higher, beside the surface.
BIOME_SHARE = 0.5

#: The words `setting.water` may be. "some" is a need for water in the footprint, "none"
#: tightens the place's water cap, and `None` says nothing.
WATER_WORDS = ("some", "none")

#: **Registered, and the threshold a setting's surface is held to.** The wanted class
#: must be at least this share of the footprint's **land** columns -- the columns that
#: are not water, because a lake in a plain does not make the plain less green. Half:
#: "predominantly" in its plain sense, and no site was measured before it was written. A
#: candidate under it does not meet the needs; among candidates that do, more of the
#: class ranks higher (`find_site.rank_key`).
SURFACE_SHARE = 0.5

#: **Registered.** "some water" is at least this much of the footprint under water: two
#: per cent of 512x512 is a lake seventy blocks across, and a place asking for a lake
#: that gets a puddle has not got one. "no water" caps the footprint at the same figure.
WATER_SOME_PCT = 2.0
WATER_NONE_PCT = 2.0


def read_setting(got, where: str = "setting") -> dict:
    """The place's setting, checked, or the empty one where the spec says nothing.

        Refuses by name: a surface word the search cannot measure is a comment, and the
        whole reason this field exists is that the setting was a comment before.
        
    """
    from .groundread import SURFACES
    out = dict(SETTING_DEFAULT, notes="")
    if got is None:
        return out
    if not isinstance(got, dict):
        raise SpecError(f"{where} is an object of {sorted(SETTING_DEFAULT)}, or null, "
                        f"not {type(got).__name__}")
    unknown = sorted(set(got) - set(SETTING_DEFAULT) - {"notes"})
    if unknown:
        raise SpecError(f"{where} has no field called {unknown[0]!r}; it declares "
                        f"{sorted(SETTING_DEFAULT)} and notes")
    surface = got.get("surface")
    if surface is not None and surface not in SURFACES:
        raise SpecError(f"{where}: surface is one of {list(SURFACES)}, or null for no "
                        f"preference, not {surface!r}")
    water = got.get("water")
    if water is not None and water not in WATER_WORDS:
        raise SpecError(f"{where}: water is one of {list(WATER_WORDS)}, or null for no "
                        f"opinion, not {water!r}")
    relief = got.get("relief")
    if relief is not None and relief not in RELIEF_WORDS:
        raise SpecError(f"{where}: relief is one of {sorted(RELIEF_WORDS)}, or null for "
                        f"no opinion, not {relief!r}")
    biome = got.get("biome")
    if biome is not None:
        from .groundread import BIOMES
        words = [biome] if isinstance(biome, str) else biome
        if not isinstance(words, list) or not words \
                or any(not isinstance(w, str) for w in words):
            raise SpecError(f"{where}: biome is a word or a list of words from "
                            f"{list(BIOMES)}, or null for no preference, not {biome!r}")
        bad = [w for w in words if w not in BIOMES]
        if bad:
            raise SpecError(f"{where}: biome is one or more of {list(BIOMES)}, or null "
                            f"for no preference, not {bad[0]!r}")
        biome = None if "any" in words else sorted(set(words))
    out["surface"] = str(surface) if surface else None
    out["water"] = str(water) if water else None
    out["relief"] = str(relief) if relief else None
    out["biome"] = biome or None
    out["notes"] = str(got.get("notes") or "")
    return out


def relief_cap(setting: dict | None, footprint: int) -> float | None:
    """The most relief the setting's word allows over this footprint, or None.

        `RELIEF_WORDS` is fall per block and this is that times the side, rounded the way
        `relief_for` rounds, so the two caps a place carries are in the same unit and the
        tighter one wins in `read_spec`. A `steep` word caps at `STEEP_CAP`, which is the
        world's height and not a preference.
        
    """
    word = (setting or {}).get("relief") or "any"
    g = RELIEF_WORDS.get(word)
    if g is None and word in RELIEF_FLOORS:
        return round(float(STEEP_CAP) * int(footprint), 1)
    return None if g is None else round(float(g) * int(footprint), 1)


def relief_floor(setting: dict | None, footprint: int) -> float | None:
    """The least relief the setting's word asks for over this footprint, or None."""
    g = RELIEF_FLOORS.get((setting or {}).get("relief") or "any")
    return None if g is None else round(float(g) * int(footprint), 1)


def relief_for(footprint: int) -> float:
    """How much fall a place of this footprint may have across it. See `MAX_GRADIENT`.

        The whole-site relief is not what makes a site unbuildable and never was: `site()`
        prepares each part's own ground, and what a *place* needs is a level centre -- which
        is the plateau measure -- and to be one place rather than the two sides of a
        mountain, which is this.
        
    """
    return round(MAX_GRADIENT * int(footprint), 1)


# --------------------------------------------------------------- the sentence

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
                 "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
                 "hundred": 100, "two hundred": 200, "three hundred": 300,
                 "four hundred": 400, "five hundred": 500, "a thousand": 1000}

# **Thirteen to ninety-nine, by rule.** The closure round's first retained failure: the
# table above had no row for `sixteen`, and a village of sixteen cottages was sized at
# fifty-two. The interpretation now carries the count (see
# `stages_plan.explicit_count_of`) and this table is the rules' cross-check, which has
# to be able to read the same words.
_NUMBER_WORDS.update({"thirteen": 13, "fourteen": 14, "sixteen": 16, "seventeen": 17,
                      "eighteen": 18, "nineteen": 19})
for _tens, _tv in (("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50),
                   ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90)):
    for _ones, _ov in (("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
                       ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)):
        _NUMBER_WORDS[f"{_tens}-{_ones}"] = _tv + _ov
        _NUMBER_WORDS[f"{_tens} {_ones}"] = _tv + _ov

_ABOUT_WORDS = ("about", "around", "roughly", "some", "approximately", "or so", "~")


def count_in(sentence: str) -> dict | None:
    """The explicit number of structures a sentence asks for, and whether it is `about`.

        Read here and not by the model, because "about sixty houses" has to become the same
        band every time it is asked. Returns `{"n": int, "about": bool, "phrase": str}` or
        None where the sentence names no count.
        
    """
    s = (sentence or "").lower()
    # ...of *structures*, and nothing else a place has numbers of. "three concentric
    # walls" is a defining part's multiplicity and is the model's to read; this is the
    # size of the settlement and it is attached to a word for a building.
    unit = r"(?:houses?|homes?|dwellings?|buildings?|structures?|cottages?|halls?)"
    words = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    # **Six digits, and thousands separators.** The architecture audit, found by asking
    # for one: `\d{1,4}` read "10000 houses" as no count at all, so a sentence naming a
    # number this system certainly cannot build was read as a sentence naming none --
    # and a place that cannot be built became a place nobody asked a question about. A
    # number too large is refused downstream, by name, with the number in the refusal; a
    # number unread is refused nowhere.
    m = re.search(rf"(\b(?:{'|'.join(_ABOUT_WORDS[:-1])})\s+)?"
                  rf"\b(\d{{1,3}}(?:,\d{{3}})+|\d{{1,6}}|{words})\b[\w\s-]{{0,20}}?"
                  rf"\b{unit}\b", s)
    if not m:
        return None
    raw = m.group(2).replace(",", "")
    n = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
    about = bool(m.group(1)) or "or so" in s or "~" in s
    return {"n": n, "about": about, "phrase": m.group(0).strip()}


def band_for(kind: str, count: dict | None) -> tuple:
    """The size band, from the kind and any explicit count. A1.

        The kind's band unless the sentence gave a number, in which case that number wins --
        `+/- ABOUT` where the sentence hedged and exactly where it did not. The kind's band
        is **not** re-applied afterwards: a sentence that asks for a town of twelve houses
        is asking for twelve houses and the number is what it said.
        
    """
    if kind not in SIZE_BANDS:
        raise SpecError(f"kind is one of {list(KINDS)}, not {kind!r}")
    if count is None:
        return size_band_for(kind)
    n = max(1, int(count["n"]))
    if not count.get("about"):
        return (n, n)
    return (max(1, int(round(n * (1 - ABOUT)))), int(round(n * (1 + ABOUT))))


def size_from(declared: int, lo: int, hi: int) -> int:
    """How big the place is, from what the districts declared and the kind's band.

        So the arithmetic is here, where A1 says it belongs, and the rule turns on **which
        side of the band** the sum falls:

          - at or above the band's floor, it is a count. A city declared at 1,200 has been
            given a size and it is too big; that is the **ceiling's** job, and the scaling
            case in `test_place` is what says so, unmoved.
          - below the band's floor, it is not a count at all: no city is thirteen buildings
            and no model believes one is. It is read as the proportions between the parts
            and the size comes from the middle of the band.

        Either way what the model is answerable for is the *ratio* between its districts,
        which is the thing it actually knows, and the size is the library's.
        
    """
    if not declared or int(declared) < int(lo):
        return int(round((lo + hi) / 2))
    return int(declared)


def apportion(parts: list, declared: int, target: int) -> list:
    """Spread `target` structures over the group parts in their declared proportions.

        Untouched where the declarations already sum to the target, so a spec that gave real
        counts keeps them exactly. Largest-remainder, so the parts add up to the target and
        the rounding goes to the parts that lost most of it -- and never to nothing, because
        a district that accounted for structures still accounts for at least one.
        
    """
    groups = [p for p in parts if district(p) and p["structures"]]
    if not groups:
        # "so that the count is computed from kind, share and density rather than stated
        # here"). The proportions are then the library's: each ring's share of the place
        # over the ground one of its structures takes at its density -- a sparse belt of
        # half the place and a dense ring of a fifth hold about the same number -- and
        # equal shares where the districts have none.
        groups = [p for p in parts if district(p)]
        if not groups or not target:
            return parts
        for p in groups:
            p["structures_inferred"] = True      # the closure round: an inferred share
        weight = {p["name"]: (float(p["share"]) / DENSITIES.get(p.get("density")
                                                                  or "medium", 1.0)
                              if p.get("share") is not None else 1.0) for p in groups}
        if any(p.get("share") is not None for p in groups):
            for p in groups:
                if p.get("share") is None:
                    weight[p["name"]] = 0.0
        total = sum(weight.values()) or 1.0
        shares = [(p, float(target) * weight[p["name"]] / total) for p in groups]
        for p, want in shares:
            p["structures"] = max(1, int(want))
        short = int(target) - sum(p["structures"] for p in groups)
        order = sorted(shares, key=lambda s: (-(s[1] - int(s[1])), s[0]["name"]))
        i = 0
        while short > 0 and order:
            order[i % len(order)][0]["structures"] += 1
            short -= 1
            i += 1
        while short < 0 and order:
            p = order[-(1 + (-short - 1) % len(order))][0]
            if p["structures"] > 1:
                p["structures"] -= 1
            short += 1
        return parts
    if not declared or int(declared) == int(target):
        return parts
    shares = [(p, p["structures"] * float(target) / float(declared)) for p in groups]
    for p, want in shares:
        p["structures"] = max(1, int(want))
    short = int(target) - sum(p["structures"] for p in groups)
    order = sorted(shares, key=lambda s: (-(s[1] - int(s[1])), s[0]["name"]))
    i = 0
    while short > 0 and order:
        order[i % len(order)][0]["structures"] += 1
        short -= 1
        i += 1
    while short < 0 and order:
        p = order[-(1 + (-short - 1) % len(order))][0]
        if p["structures"] > 1:
            p["structures"] -= 1
        short += 1
    return parts


# ------------------------------------------------------------------ the schema

def read_part(d: dict, where: str) -> dict:
    """One defining part, checked. Refuses by name."""
    if not isinstance(d, dict):
        raise SpecError(f"{where}: a defining part is an object, not "
                        f"{type(d).__name__}")
    name = str(d.get("name") or "").strip()
    if not re.fullmatch(r"[a-z0-9_]{2,40}", name):
        raise SpecError(f"{where}: a defining part's name is a short lower-case slug, "
                        f"not {d.get('name')!r}")
    kind = d.get("kind")
    if kind not in PART_KINDS:
        raise SpecError(f"{where} ({name}): kind is one of {list(PART_KINDS)}, "
                        f"not {kind!r}", field="kind", part=name)
    fam = d.get("family")
    if fam not in FAMILIES:
        raise SpecError(f"{where} ({name}): family is one of {list(FAMILIES)}, "
                        f"not {fam!r}", field="family", part=name)
    rel = d.get("relation")
    if rel not in RELATIONS:
        raise SpecError(f"{where} ({name}): relation is one of {list(RELATIONS)}, "
                        f"not {rel!r}", field="relation", part=name)
    count = d.get("count", 1)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise SpecError(f"{where} ({name}): count is a whole number of this part, "
                        f"one or more, not {count!r}", field="count", part=name)
    holds = d.get("structures", 0)
    if not isinstance(holds, int) or isinstance(holds, bool) or holds < 0:
        raise SpecError(f"{where} ({name}): structures is how many of the place's "
                        f"structures this part accounts for, not {holds!r}",
                        field="structures", part=name)
    out = {"name": name, "kind": kind, "family": fam, "relation": rel,
           "count": int(count), "structures": int(holds),
           "needs": read_part_needs(d.get("needs"), f"{where} ({name})"),
           "forms": read_part_forms(d.get("forms"), f"{where} ({name})"),
           "density": read_density(d.get("density"), f"{where} ({name})"),
           "notes": str(d.get("notes") or "")}
    out["role"] = read_role(d.get("role"), out, f"{where} ({name})")
    if d.get("structures_inferred"):
        out["structures_inferred"] = True        # a checked spec read back keeps it
    read_ring_fields(d, out, f"{where} ({name})")
    read_character(d.get("character"), out, f"{where} ({name})")
    of = d.get("of")
    if of is not None:
        if rel not in RELATIONS_OF:
            raise SpecError(f"{where} ({name}): `of` names the part a `near`, `along` "
                            f"or `on` relation is placed against, and this part's "
                            f"relation is `{rel}`", field="of", part=name)
        if not isinstance(of, str) or not re.fullmatch(r"[a-z0-9_]{2,40}", of):
            raise SpecError(f"{where} ({name}): `of` is another defining part's name, "
                            f"not {of!r}", field="of", part=name)
        out["of"] = of
    return out


def read_ring_fields(d: dict, out: dict, where: str) -> None:
    """`ring`, `share`, `walled` and `voice` off one defining part, checked, onto `out`.

        Refuses by name on a part that is not a district: a wall with a `share` is a
        comment, and the whole of why these fields exist is that a ring's proportion was
        a comment before. An authored voice lands on `out["authored_voice"]` beside the
        name, for `read_spec` to collect.
        
    """
    present = [k for k in RING_FIELDS if d.get(k) is not None]
    if not present:
        return
    if not district(out):
        raise SpecError(f"{where}: {present[0]!r} belongs to a district part -- a "
                        f"`group` of family {list(DISTRICT_FAMILIES)} -- and this is a "
                        f"{out['kind']} of family {out['family']}")
    ring = d.get("ring")
    if ring is not None:
        if not isinstance(ring, int) or isinstance(ring, bool) or ring < 0:
            raise SpecError(f"{where}: ring is a whole number, 0 at the centre and "
                            f"counting outward, not {ring!r}")
        out["ring"] = int(ring)
    share = d.get("share")
    if share is not None:
        if isinstance(share, bool) or not isinstance(share, (int, float)) \
                or not 0.0 < float(share) <= 1.0:
            raise SpecError(f"{where}: share is this ring's fraction of the place's "
                            f"whole area, over 0 and at most 1, not {share!r}")
        out["share"] = float(share)
    if "ring" in out and "share" not in out:
        raise SpecError(f"{where}: a ring carries a share -- its fraction of the "
                        f"place's area -- and ring {out['ring']} has none")
    if "share" in out and "ring" not in out:
        raise SpecError(f"{where}: a share belongs to a ring; say which ring this is "
                        f"(0 at the centre, counting outward)")
    walled = d.get("walled")
    if walled is not None:
        if not isinstance(walled, bool):
            raise SpecError(f"{where}: walled is true or false, not {walled!r}")
        if "ring" not in out:
            raise SpecError(f"{where}: walled says a wall bounds this ring on its "
                            f"outside, and this part names no ring")
        out["walled"] = bool(walled)
    elif "ring" in out:
        out["walled"] = False
    if d.get("voice") is not None:
        if "ring" not in out:
            raise SpecError(f"{where}: a voice of its own belongs to a ring; say which "
                            f"ring this is")
        name, authored = read_voice(d.get("voice"))
        out["voice"] = name
        if authored:
            out["authored_voice"] = authored


def read_character(got, out: dict, where: str) -> None:
    """A district part's `character`, checked, onto `out["character"]`. v2, C1.

        Absent means the district is planned as it always was; present -- even empty --
        means the compiler plans it. Refused by name on a part that is not a district, on
        a field that is not one of `CHARACTER_FIELDS`, and on a value out of its range.
        
    """
    if got is None:
        return
    if not district(out):
        raise SpecError(f"{where}: a character belongs to a district part -- a `group` "
                        f"of family {list(DISTRICT_FAMILIES)} -- and this is a "
                        f"{out['kind']} of family {out['family']}", field="character",
                        part=out["name"])
    if not isinstance(got, dict):
        raise SpecError(f"{where}: character is an object of {list(CHARACTER_FIELDS)}, "
                        f"not {type(got).__name__}", field="character", part=out["name"])
    ch = {}
    for k, v in got.items():
        if k not in CHARACTER_FIELDS:
            raise SpecError(f"{where}: character has no field {k!r}; it is one of "
                            f"{list(CHARACTER_FIELDS)}", field="character",
                            part=out["name"])
        if v is None:
            continue
        if k == "frontage":
            if v not in FRONTAGES:
                raise SpecError(f"{where}: frontage is one of {list(FRONTAGES)}, not "
                                f"{v!r}", field="character", part=out["name"])
        elif k in ("block", "lot_width", "lot_depth"):
            if isinstance(v, bool) or not isinstance(v, int) or not 3 <= v <= 256:
                raise SpecError(f"{where}: {k} is a whole number of columns, 3 to 256, "
                                f"not {v!r}", field="character", part=out["name"])
        elif k == "attached":
            if not isinstance(v, bool):
                raise SpecError(f"{where}: attached is true or false, not {v!r}",
                                field="character", part=out["name"])
        elif k in ("courtyard_share", "open_share"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not 0.0 <= float(v) <= 1.0:
                raise SpecError(f"{where}: {k} is a share, 0 to 1, not {v!r}",
                                field="character", part=out["name"])
            v = float(v)
        elif k == "variety":
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not 0.0 <= float(v) <= 1.0:
                raise SpecError(f"{where}: variety is a share of the lot's own side, "
                                f"0 to 1, not {v!r}", field="character",
                                part=out["name"])
            v = float(v)
        elif k == "storeys":
            ok = (isinstance(v, (list, tuple)) and len(v) == 2
                  and all(not isinstance(n, bool) and isinstance(n, int) and 1 <= n <= 8
                          for n in v) and v[0] <= v[1])
            if not ok:
                raise SpecError(f"{where}: storeys is a band [lo, hi] of whole numbers "
                                f"1 to 8 with lo <= hi, not {v!r}", field="character",
                                part=out["name"])
            v = [int(v[0]), int(v[1])]
        elif k == "landmarks":
            if not isinstance(v, list) or not all(
                    isinstance(m, dict) and isinstance(m.get("type"), str)
                    and re.fullmatch(r"[a-z0-9_]{2,40}", m["type"]) for m in v):
                raise SpecError(f"{where}: landmarks is a list of {{\"type\": a type "
                                f"name}}, not {v!r}", field="character",
                                part=out["name"])
            v = [{"type": m["type"], "notes": str(m.get("notes") or "")} for m in v]
        ch[k] = v
    out["character"] = ch


def character(part: dict) -> dict | None:
    """The character a district part carries, or None where it carries none and the
    district is a model's to plan."""
    if not district(part):
        return None
    return part.get("character")


def rings(spec: dict) -> list:
    """The ring parts of a spec, innermost first. Empty where the place has none."""
    got = [p for p in spec.get("defining_parts") or [] if p.get("ring") is not None]
    return sorted(got, key=lambda p: (int(p["ring"]), p["name"]))


def walled_rings(spec: dict) -> list:
    """The rings a wall bounds, innermost first."""
    return [p for p in rings(spec) if p.get("walled")]


def centre_share(spec: dict) -> float:
    """What is left of the place for its centre once the rings have their shares."""
    return max(0.0, 1.0 - sum(float(p["share"]) for p in rings(spec)))


def _check_rings(parts: list, where: str = "defining_parts") -> None:
    """The rings as a set: distinct indices, shares under one, and the wall and gate
    parts consistent with how many rings are walled. Refuses by name."""
    ringed = [p for p in parts if p.get("ring") is not None]
    if not ringed:
        return
    seen: dict = {}
    for p in ringed:
        if p["ring"] in seen:
            raise SpecError(f"{where}: {p['name']} and {seen[p['ring']]} are both "
                            f"ring {p['ring']}; each ring is one district part")
        seen[p["ring"]] = p["name"]
    total = sum(float(p["share"]) for p in ringed)
    if total > 1.0 + 1e-9:
        raise SpecError(f"{where}: the rings' shares sum to {total:.3f} and a place is "
                        f"one whole: they sum to at most 1, and what is left is the "
                        f"centre's")
    walled = sum(1 for p in ringed if p.get("walled"))
    for p in parts:
        if p["family"] == "wall" and p["relation"] == "concentric" \
                and int(p["count"]) != walled:
            raise SpecError(f"{where} ({p['name']}): the concentric wall's count is the "
                            f"number of walled rings, which is {walled}, and this says "
                            f"{p['count']}; mark the rings `walled` and the count "
                            f"follows")
        if p["family"] == "gate" and p["relation"] == "gateway" \
                and int(p["count"]) > max(walled, 0) and walled:
            raise SpecError(f"{where} ({p['name']}): one gate per walled ring is laid, "
                            f"which is {walled}, and this asks for {p['count']}")


def district_voice(spec: dict, district: dict) -> str | None:
    """The voice a place-level district is built in, off the ring it answers, or None
    for the place's own."""
    want = (district or {}).get("defines")
    for p in spec.get("defining_parts") or []:
        if p["name"] == want:
            return p.get("voice") or None
    return None


def read_role(got, part: dict, where: str) -> str:
    """What this defining part is for, checked, or derived where it does not say. A2."""
    if got is not None:
        if got not in ROLES:
            raise SpecError(f"{where}: role is one of {list(ROLES)}, or null to take "
                            f"the one this part's family implies, not {got!r}")
        return str(got)
    fam = part.get("family")
    if fam in DISTRICT_FAMILIES:
        return "rural" if part.get("density") == "sparse" else "urban"
    return _ROLE_OF_FAMILY.get(fam, "civic")


def read_part_needs(got, where: str) -> dict:
    """One defining part's own ground, checked. A1. Refuses by name."""
    out = dict(PART_NEEDS_DEFAULT)
    if got is None:
        return out
    if not isinstance(got, dict):
        raise SpecError(f"{where}: needs is an object of "
                        f"{sorted(PART_NEEDS_DEFAULT)}, not {type(got).__name__}")
    unknown = sorted(set(got) - set(PART_NEEDS_DEFAULT))
    if unknown:
        raise SpecError(f"{where}: a defining part's needs has no field called "
                        f"{unknown[0]!r}; it declares {sorted(PART_NEEDS_DEFAULT)}")
    for k in ("max_relief", "max_water_pct", "max_forest_pct"):
        if got.get(k) is not None:
            out[k] = float(got[k])
    if got.get("plateau") is not None:
        n = int(got["plateau"])
        if n < 1:
            raise SpecError(f"{where}: needs['plateau'] is the side of the level square "
                            f"this part stands on, one or more, not {n}")
        out["plateau"] = n
    return out


def read_part_forms(got, where: str) -> list | None:
    """The forms a defining part's own types may be, or None for the place's. A1."""
    if got is None:
        return None
    if isinstance(got, str):
        got = [got]
    if not isinstance(got, list) or not got:
        raise SpecError(f"{where}: forms is a non-empty list of {list(FORMS)}, or null "
                        f"to take the place's own form")
    bad = [f for f in got if f not in FORMS]
    if bad:
        raise SpecError(f"{where}: forms is drawn from {list(FORMS)}, not {bad[0]!r}")
    return sorted(set(got))


def read_density(got, where: str) -> str | None:
    """How thickly this part is built up, or None. A1."""
    if got is None:
        return None
    if got not in DENSITIES:
        raise SpecError(f"{where}: density is one of {sorted(DENSITIES)}, or null, "
                        f"not {got!r}")
    return str(got)


def columns_per_plot(part: dict) -> int:
    """The **lot** one structure of this defining part stands on, its density applied."""
    from .placeplan import density_lot, DENSITY_ROLE
    d = part.get("density") or "medium"
    return int(density_lot(d, part.get("role") or DENSITY_ROLE.get(d))["columns"])


def plot_share(part: dict) -> float:
    """How much of this defining part's district ground is plots.

        The craft round: the lots' share of a block and its streets, off the compiler's own
        fabric (`placeplan.fabric`), so `structures_for` below is `columns` over what one
        house of that fabric actually costs.
        
    """
    from .placeplan import fabric, DENSITY_ROLE
    d = part.get("density") or "medium"
    # **Of the fabric this part is actually built of.** `fabric` has always taken the
    # character and this caller has always dropped it, so a district that declared its
    # own lot was held to the cover of the density's *default* lot -- a target computed
    # for a fabric it had been told not to build. Found by running the shore village's
    # revision: the inspection asked for a twelve-column frontage, the district laid it,
    # and its validator refused it for covering 30% against a floor derived from a
    # twenty-four column one. A check whose evidence is a different decision than the
    # one that was made is the defect this whole round is about, one level down. **The
    # declared lot, and not the shares.** The lot is a thing a character *declares* and
    # the compiler honours; `open_share` and `courtyard_share` are the levers the
    # compiler moves to reach its count, and feeding those back into the count is a
    # circle: a character asking for open ground everywhere made `houses_per_block`
    # zero, so the district was asked for no houses, so there was nothing for the
    # compiler to lower the share for. Found by running `test_compile`'s own
    # `open_share: 1` case.
    ch = {k: v for k, v in (part.get("character") or {}).items()
          if k in ("lot_width", "lot_depth", "attached", "frontage")}
    return float(fabric(d, part.get("role") or DENSITY_ROLE.get(d),
                        ch or None)["plot_share"])


def structures_for(columns: float, part: dict, shape: tuple | None = None) -> int:
    """**The number of structures a piece of designed ground is asked for.**"""
    from .placeplan import fabric_fit, DENSITY_ROLE
    d = part.get("density") or "medium"
    role = part.get("role") or DENSITY_ROLE.get(d)
    per = columns_per_plot(part)
    by_ground = max(0, int(float(columns) * plot_share(part) // per))
    if shape is None:
        return by_ground
    # **...and never more than the grid that ground assumes can be laid on it.** The
    # craft round, E1: the divide above is right for a rectangle big enough to run a
    # grid on and wrong for a strip, and a district asked for more houses than its own
    # shape holds is a hand-back nobody can answer.
    return min(by_ground, fabric_fit(int(shape[0]), int(shape[1]), d, role,
                                     part.get("character")))


def read_spec(doc: dict, sentence: str | None = None, count: dict | None = None) -> dict:
    """A place spec off a model's answer: checked, completed, and scaled. A1 + A2.

        `count` is the explicit count the **interpretation** read from the sentence
        (`{"n", "about", "phrase", "what"}`), the closure round. Read before this the count
        came from `count_in` alone, a regex with a table of number words, so "sixteen low
        cottages" -- a number word the table lacks -- was read as no count at all and the
        village was sized at fifty-two from the kind's band while the interpretation beside
        it said sixteen, exactly. Meaning lost at the first consumer. The order now: the
        count handed in; else the count a **checked** spec already carries, so a spec read
        back from disk is the spec that was written; else `count_in`, for a round with no
        interpretation.

        What the model is answerable for is `kind`, `defining_parts` and `voice`. The band,
        the structure count, the footprint and the ceiling are computed here from the
        sentence and the kind, so the same sentence is the same place every time.
        
    """
    if not isinstance(doc, dict):
        raise SpecError(f"a place spec is an object, not {type(doc).__name__}")
    sentence = sentence if sentence is not None else doc.get("sentence")
    if not str(sentence or "").strip():
        raise SpecError("a place spec carries the sentence it was made from")
    kind = doc.get("kind")
    if kind not in KINDS:
        raise SpecError(f"kind is one of {list(KINDS)}, not {kind!r}")
    parts = doc.get("defining_parts")
    if not isinstance(parts, list) or not parts:
        raise SpecError("defining_parts is a non-empty list: what makes this place that "
                        "place and not a heap of buildings")
    seen: set = set()
    out_parts = []
    for i, p in enumerate(parts):
        row = read_part(p, f"defining_parts[{i}]")
        if row["name"] in seen:
            raise SpecError(f"two defining parts are called {row['name']!r}")
        seen.add(row["name"])
        out_parts.append(row)
    for row in out_parts:
        if row.get("of") is not None and row["of"] not in seen:
            raise SpecError(f"defining part {row['name']!r}: `of` names "
                            f"{row['of']!r} and no defining part is called that",
                            field="of", part=row["name"])
    form = doc.get("form")
    if form is not None and form not in FORMS:
        raise SpecError(f"form is one of {list(FORMS)}, or null to accept every "
                        f"committed type, not {form!r}")
    voice, authored = read_voice(doc.get("voice"))
    _check_rings(out_parts)
    # The voices the rings authored, collected off the parts: `stage_place_spec` writes
    # each to `voices/` beside the place's own, so a district's palette is on disk on
    # the same terms before anything is planned.
    ring_voices = {}
    for p in out_parts:
        if p.get("authored_voice"):
            ring_voices[p["voice"]] = p.pop("authored_voice")
    if authored and voice in ring_voices:
        raise SpecError(f"the voice {voice!r} is authored twice, by the place and by a "
                        f"ring; author it once and name it elsewhere")
    from .styles import VOICES
    known = set(VOICES) | set(ring_voices) | ({voice} if authored else set())
    for p in out_parts:
        if p.get("voice") and p["voice"] not in known:
            raise SpecError(f"defining_parts ({p['name']}): voice {p['voice']!r} is no "
                            f"voice this project has and none this spec authors; name "
                            f"one of {sorted(known)} or write one")
    invariants = doc.get("invariants")
    if invariants is not None and not isinstance(invariants, str):
        raise SpecError("invariants is a short paragraph of prose -- the named place's "
                        "defining spatial facts -- or null")

    if count is None and isinstance(doc.get("explicit_count"), dict) \
            and doc["explicit_count"].get("n"):
        count = dict(doc["explicit_count"])
    if count is None:
        count = count_in(sentence)
    lo, hi = band_for(kind, count)
    #: How big this place is, in the order the three answers are trusted: 1. the number
    #: the sentence gave -- "about sixty houses" is sixty; 2. what the **districts** say
    #: they hold, where they say. A district is the defining part that stands for
    #: ordinary buildings; three rings declaring twelve hundred between them is an ask,
    #: and it is the ask A2's ceiling scales. 3. the middle of the kind's band, where
    #: neither says. Only the groups, and that is not a detail. declining to give a
    #: number for the houses, which is what the brief tells it to do -- and counting the
    #: palace made a city of **one structure**. A defining plot is one building; it is
    #: never the size of the place. ...and only the **districts** among the groups. A
    #: compound -- a palace planned as a place of halls -- is a great thing and not the
    #: size of the place either, whatever its `structures` says; its halls are counted
    #: when they stand.
    declared = sum(p["structures"] for p in out_parts if district(p))
    target = count["n"] if count else size_from(declared, lo, hi)

    apportion(out_parts, declared, target)

    spec = {"sentence": str(sentence).strip(), "kind": kind,
            "size_band": [int(lo), int(hi)], "structures": int(target),
            "explicit_count": count, "defining_parts": out_parts,
            "voice": voice or None, "form": form or None,
            **({"authored_voice": authored} if authored else {}),
            "setting": read_setting(doc.get("setting")),
            "needs": _read_needs(doc.get("needs"), target, kind,
                                 least=(doc.get("footprint_from") or {}).get("footprint")
                                 or (doc.get("footprint_from") or {}).get("wanted_footprint")
                                 or (doc.get("footprint_from") or {}).get("least_footprint")),
            "ceiling": dict(CEILING, footprint=footprint_ceiling(kind),
                            structures=structures_ceiling(kind)),
            "notes": str(doc.get("notes") or "")}
    if invariants:
        spec["invariants"] = str(invariants).strip()
    if ring_voices:
        spec["authored_voices"] = ring_voices
    # **The least ground the parts need, before the count's footprint stands.** The
    # closure round's transfer case: an explicit count of twenty-four sized the
    # footprint at 240 and the ring arithmetic could not lay a 93-column compound and
    # two rings at their least widths in it. Extent, hierarchy and count are resolved
    # together: the footprint is the larger of what the count implies and what the
    # defining parts need, and the record says which governed.
    try:
        from . import placeplan as _placeplan
        least = _placeplan.least_footprint(spec)
    except Exception:                            # noqa: BLE001 -- no layout, no floor
        least = None
    recorded = doc.get("footprint_from") if isinstance(doc.get("footprint_from"), dict) else None
    if least and int(least) > int(spec["needs"]["footprint"]):
        spec["needs"]["footprint"] = int(least)
    if least and int(least) >= int(spec["needs"]["footprint"]):
        spec["footprint_from"] = {"least_footprint": int(least),
                                  "why": "the defining parts' least ground governs"}
    # **...and the ground the sentence's own count needs at its density words.** The
    # expression round (worker A): `least_footprint` is the side the ring layout refuses
    # below; `wanted_footprint` adds the ground each ring of the count needs at the
    # least cover its density word admits, so a sparse ring of nine farmhouses is not
    # squeezed onto a count-sized site and the farm's fields are not short of their
    # purpose.
    try:
        from . import placeplan as _placeplan
        wanted = _placeplan.wanted_footprint(spec)
    except Exception:                            # noqa: BLE001 -- no layout, no want
        wanted = None
    if wanted and int(wanted) > int(spec["needs"]["footprint"]):
        spec["needs"]["footprint"] = int(wanted)
        spec["footprint_from"] = {"wanted_footprint": int(wanted),
                                  **({"least_footprint": int(least)} if least else {}),
                                  "why": "the ground the count needs at its density words "
                                         "governs"}
    # **A record is read back as it was written.** A checked spec carries the footprint
    # it was sited on under `footprint_from`; a later change to the layout's arithmetic
    # does not resize a place that has already been laid out, and the record says which
    # number governed then.
    if recorded and recorded.get("footprint") is not None:
        spec["needs"]["footprint"] = int(recorded["footprint"])
        spec["footprint_from"] = dict(recorded)
    elif recorded and spec.get("footprint_from"):
        spec["footprint_from"]["footprint"] = int(spec["needs"]["footprint"])
    if spec.get("footprint_from"):
        spec["footprint_from"].setdefault("footprint", int(spec["needs"]["footprint"]))
    spec = scale_to_ceiling(spec)
    # The setting's relief word, folded into the place's need as a cap over the
    # footprint the place finally has: the tighter of the two caps stands, so a spec
    # that named a flat plain is scored against one, and a spec read twice reads the
    # same -- `min` of a cap with itself is itself.
    cap = relief_cap(spec["setting"], spec["needs"]["footprint"])
    if cap is not None:
        spec["needs"]["max_relief"] = min(float(spec["needs"]["max_relief"]), cap)
    # The cap it carries is the world's, so the floor is the need, and `min` of a floor
    # with itself is itself.
    floor = relief_floor(spec["setting"], spec["needs"]["footprint"])
    if floor is not None:
        spec["needs"]["max_relief"] = max(float(spec["needs"]["max_relief"]), cap)
        spec["needs"]["min_relief"] = floor
    elif "min_relief" in spec["needs"]:
        del spec["needs"]["min_relief"]
    # A spec that has already been scaled is read back with its parts already reduced,
    # so nothing here would scale it a second time -- and the record of the scaling
    # would vanish on the second read. It is kept.
    if doc.get("scaled_from") and "scaled_from" not in spec:
        spec["scaled_from"] = doc["scaled_from"]
    # ...and the same for the record of a requirement the ceiling could not meet. A spec
    # read back from disk is already under the ceiling, so nothing here would write
    # `unmet` a second time -- and an obligation that vanishes when the record is re-
    # read is the failure mode this whole round exists to close.
    if doc.get("unmet") and "unmet" not in spec:
        spec["unmet"] = doc["unmet"]
    if spec.get("unmet") and spec.get("explicit_count"):
        # The band on disk is the band the sentence set, and a second read restates it
        # rather than deriving a softer one from the reduced count.
        spec["size_band"] = [int(v) for v in spec["unmet"]["band"]]
    # **A negotiated target survives being read again.** The architecture round, found
    # by running the repair loop: `repair.apply` writes the revised band and count to
    # `place.checked.json`, and the next `place_spec()` re-derived both from the kind
    # and threw them away -- so the repair landed, the plan was laid out from the
    # programme it had before, and the finding it was meant to close stayed open. A
    # negotiation is a **decision about an inferred choice**, on the record with its
    # bound and its reason, and it is as much a part of the spec as the sentence's own
    # count. Only `size_band` and `structures` are restated, and never over an explicit
    # count: `repair` refuses to negotiate one and this could not restate it if it did.
    neg = doc.get("negotiated")
    if neg:
        spec["negotiated"] = list(neg)
        last = [n for n in neg if n.get("what") == "size_band"]
        if last and not spec.get("explicit_count"):
            to = last[-1]["to"]
            spec["size_band"] = [int(v) for v in to["size_band"]]
            spec["structures"] = int(to["structures"])
            apportion(out_parts, sum(p["structures"] for p in out_parts if district(p)),
                      spec["structures"])
    return spec


def read_voice(got) -> tuple:
    """The spec's `voice`: a name, `null`, or **a voice the model wrote**.

        Returns `(name, authored)` -- the name to build in, and the validated voice to write
        to `voices/` where the model made one, or `None` where it named one that exists.

        This is the whole of why voices became data. `blackstone_and_ash` was chosen for the
        first town built from a sentence because it was the darkest of seven literals in a
        Python file, not because a sentence asked for it; nothing here could ever have
        answered "ochre stone under green tile" except by somebody adding an eighth
        literal. A voice the model authors is validated here, before a plan exists and long
        before a block is placed -- so a palette naming a family with no stairs is a refusal
        about a JSON file rather than a chimney that quietly failed to be built sixty times.
        
    """
    from .voices import VoiceError, validate
    if got is None:
        return (None, None)
    if isinstance(got, str):
        return (got, None)
    if not isinstance(got, dict):
        raise SpecError("voice is the name of a voice, an object authoring a new one, "
                        "or null to choose one against the ground")
    name = got.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError("a voice the spec authors carries the name it will be known by: "
                        "\"voice\": {\"name\": \"ochre_and_green_tile\", \"roles\": ...}")
    try:
        authored = validate({k: v for k, v in got.items() if k != "name"} | {"name": name},
                            where=f"the voice {name!r} this spec authors")
    except VoiceError as e:
        raise SpecError(str(e)) from e
    return (name, authored)


def _read_needs(got, structures: int, kind: str | None = None,
                least: int | None = None) -> dict:
    """The ground the place wants, filled out. **The footprint is derived, never given.**

        A footprint that is present and does not match what the count implies is refused by
        name rather than obeyed: the whole of A1 is that the model reads and the library
        computes, and a spec that could hand back its own footprint is a spec a model can
        size a place with. A footprint that *does* match is accepted, because reading a spec
        that has already been read has to give the same spec -- `Round.place_spec()` is
        called by five stages.
        
    """
    out = dict(NEEDS_DEFAULT)
    want = footprint_for(structures, kind)
    if got is not None:
        if not isinstance(got, dict):
            raise SpecError(f"needs is an object of {sorted(NEEDS_DEFAULT)}, not "
                            f"{type(got).__name__}")
        # `min_relief` is never the model's: `read_spec` derives it from a `steep`
        # setting and a spec read back carries it, so it is accepted on the way in and
        # derived again rather than refused (as `value` is by the voice validator).
        unknown = sorted(set(got) - set(NEEDS_DEFAULT) - {"min_relief"})
        if unknown:
            raise SpecError(f"needs has no field called {unknown[0]!r}; it declares "
                            f"{sorted(NEEDS_DEFAULT)}")
        for k in ("max_relief", "max_water_pct", "max_forest_pct"):
            if got.get(k) is not None:
                out[k] = float(got[k])
        if got.get("plateau") is not None:
            out["plateau"] = int(got["plateau"])
        # A spec on the record is read back as the spec it was, and the record is not
        # resized. ...or the footprint the defining parts' least ground governed, which
        # a checked spec records beside its needs (`footprint_from`) -- the closure
        # round
        if got.get("footprint") is not None \
                and int(got["footprint"]) != want \
                and int(got["footprint"]) != min(want, footprint_ceiling(kind)) \
                and int(got["footprint"]) != min(want, CEILING["footprint"]) \
                and int(got["footprint"]) != (int(least) if least else None) \
                and not (least and int(got["footprint"]) >= want):
            raise SpecError(
                f"needs['footprint'] is derived from the structure count and is not "
                f"yours to give: {structures} structures is {want}x{want} and this "
                f"says {got['footprint']}")
        if got.get("footprint") is not None and int(got["footprint"]) != want:
            want = int(got["footprint"])
    out["footprint"] = want
    if out["max_relief"] is None:
        out["max_relief"] = relief_for(want)
    return out


# ------------------------------------------------------------------ A2, scaling

def scale_to_ceiling(spec: dict) -> dict:
    """Bring a spec under `CEILING`, and write down that it was. A2.

        The footprint is capped in the same act and re-derived from the reduced count, so a
        scaled spec asks for the ground it now needs and not the ground it wanted.
        
    """
    cap = structures_ceiling(spec.get("kind"))
    want = int(spec["structures"])
    fcap = footprint_ceiling(spec.get("kind"))
    if want <= cap and int(spec["needs"]["footprint"]) <= fcap:
        return spec
    factor = cap / want if want > cap else 1.0
    was = {"structures": want, "size_band": list(spec["size_band"]),
           "footprint": int(spec["needs"]["footprint"]),
           "per_part": {p["name"]: p["structures"] for p in spec["defining_parts"]},
           "factor": round(factor, 4)}
    for p in spec["defining_parts"]:
        if p["structures"]:
            # Down, and never to nothing: a defining part that accounted for structures
            # still accounts for at least one. Truncated rather than rounded so the
            # scaled parts cannot add up to more than the ceiling they were scaled to.
            p["structures"] = max(1, int(p["structures"] * factor))
    got = sum(p["structures"] for p in spec["defining_parts"])
    spec["structures"] = min(cap, got or min(want, cap))
    lo, hi = spec["size_band"]
    # **The band the sentence gave is not the band the ceiling may move.** The
    # architecture audit, and the sharpest thing it found: asked for *exactly 2,000
    # houses*, the ceiling reduced the place to 1,824 and moved the acceptance band down
    # to match, so `in_band` passed and the run reported success on a request it had not
    # met. A ceiling is a statement about this system's capacity. It is not a licence to
    # rewrite what was asked for. So the band moves only where it was the **library's
    # own inference** -- the kind's band, which is revisable by definition -- and stands
    # where the **sentence** set it. A spec whose explicit count cannot be reached keeps
    # the band that says so, carries `unmet` naming the limiting constraint, and fails
    # the count clause honestly.
    explicit = spec.get("explicit_count")
    if explicit:
        spec["unmet"] = {
            "requirement": "count/structures",
            "asked": int(explicit["n"]), "about": bool(explicit.get("about")),
            "band": [int(lo), int(hi)], "can_build": spec["structures"],
            "limit": "structures_ceiling", "limit_value": cap,
            "why": (f"the sentence asks for {explicit['n']} structures and this "
                    f"build's ceiling for a {spec.get('kind')} is {cap}; the place is "
                    f"planned at {spec['structures']} and the band the sentence set is "
                    f"left where it is, so the count clause fails rather than passing "
                    f"against a band moved to fit")}
    else:
        spec["size_band"] = [min(int(lo), spec["structures"]), min(int(hi), cap)]
    spec["needs"]["footprint"] = min(footprint_for(spec["structures"], spec.get("kind")),
                                     fcap, int(spec["needs"]["footprint"]))
    spec["scaled_from"] = {
        **was, "to": {"structures": spec["structures"],
                      "size_band": list(spec["size_band"]),
                      "footprint": spec["needs"]["footprint"],
                      "per_part": {p["name"]: p["structures"]
                                   for p in spec["defining_parts"]}},
        "ceiling": dict(CEILING, footprint=fcap, structures=cap),
        "band_held": bool(explicit),
        "why": ("the spec asked for more than the ceiling allows; counts inside each "
                "defining part were reduced proportionally and no defining part was "
                "dropped"
                + ("; the sentence's own band was NOT moved to fit the reduction"
                   if explicit else ""))}
    return spec


# ------------------------------------------------------------------- utilities

def walls(spec: dict) -> list:
    """The defining parts that are walls. What A6 reads "walled" off."""
    return [p for p in spec["defining_parts"] if p["family"] == "wall"]


def compound(part: dict) -> bool:
    """Is this defining part a great thing that is planned as a place inside the place?

        Yes for a `group` of any family that is not a district -- a castle, a monastery, a
        temple precinct, whatever the spec calls a division of the place that is made of
        parts -- and yes for a `COMPOUND_FAMILIES` family whatever kind the spec guessed,
        for `placeplan._kind_ok`'s reason: a spec is written before any type exists and
        cannot know that a palace is built as a compound of halls and not as a house.
        
    """
    fam = part.get("family")
    if fam in COMPOUND_FAMILIES:
        return True
    return part.get("kind") == "group" and fam not in DISTRICT_FAMILIES


def district(part: dict) -> bool:
    """Is this defining part a district: a division of the place that holds houses?"""
    return part.get("kind") == "group" and part.get("family") in DISTRICT_FAMILIES


#: What the ground between the buildings of a district **is**. Distinct from `role`,
#: which says what the buildings are for, and the integration review's fourth finding
#: turns on the two having been the same word: > its supplied programme uses `role:
#: rural`, which `placeplan.district_failures` > treats as a **farmland belt** requiring
#: 60% farms/fields coverage [...] "rural > means farmland" also imports the wrong
#: functional assumption. A fishing village is rural and is not a farm. A farm belt is a
#: farm because it is fields, and a spec that means fields can say so.
LAND_USES = ("settled", "farmland", "pasture", "orchard", "industrial", "civic")

#: The words a spec spends on a district when it means farmland, for the derivation
#: below. Read off the part's **own** name and prose, which is where a model that meant
#: a farm belt actually said so. Matched on word boundaries against the part's name and
#: prose with `_` read as a space, so `farm_belt` and "the belt of fields inside the
#: great wall" both answer and "a village of fisherfolk" does not.
_FARMLAND_WORDS = ("farm", "farms", "farmland", "farmlands", "farmstead", "farmsteads",
                   "field", "fields", "agrarian", "agricultural", "arable", "paddy",
                   "paddies", "cropland", "irrigation")


#: **The programme's entities, bound once.** The expression round. The falsification
#: pass's held-out sentence spent one of eleven houses on the orchard's lot and could
#: not select the orchard by word, because "what is a counted building, what is land"
#: was answered by each policy with its own noun table. A defining part is one of these
#: classes, read from what it declares -- kind, family, role, function, land use -- and
#: never from its name alone; the explicit count is spent only on `building` entities
#: whose unit is the count's subject. `land` families are areas of ground with a
#: purpose.
ENTITY_CLASSES = ("building", "land", "amenity", "compound", "boundary", "point")
_LAND_FAMILIES = ("field", "orchard", "pasture", "paddy", "grove", "garden", "yard",
                  "farmland", "meadow", "park", "cropland")
_AMENITY_FAMILIES = ("square", "plaza", "market", "court", "well", "fountain")
_LAND_USE_WORDS = {"orchard": ("orchard", "orchards", "fruit trees"),
                   "pasture": ("pasture", "pastures", "paddock", "grazing", "meadow"),
                   "farmland": tuple(_FARMLAND_WORDS),
                   "garden": ("garden", "gardens", "allotment"),
                   "grove": ("grove", "wood", "woods", "copse"),
                   "square": ("square", "plaza", "market", "marketplace")}
_HOUSE_FAMILIES = ("house", "houses", "home", "homes", "dwelling", "cottage", "cottages",
                   "minka", "townhouse", "row_house", "shop_house", "courtyard_house",
                   "farmstead", "residence")


def land_use_of(part: dict | None) -> str | None:
    """The land use a defining part names, or None where it names none.

    In order: the declaration; the part's own **name** (an `orchard` laid as a grove is
    an orchard); its family; its prose -- and prose can only name working land
    (`orchard`, `pasture`, `farmland`, `garden`, `grove`), never a square: a homes
    district "gathered about the market square" is not a square."""
    import re as _re
    p = part or {}
    said = str(p.get("land_use") or "").strip().lower()
    if said in LAND_USES:
        return said

    def in_text(text, words):
        return any(_re.search(rf"(?<![a-z]){_re.escape(w)}(?![a-z])", text) for w in words)
    name = str(p.get("name") or "").lower().replace("_", " ")
    working = ("orchard", "pasture", "farmland", "garden", "grove")
    for use in working:
        if in_text(name, _LAND_USE_WORDS[use]):
            return use
    fam = str(p.get("family") or "").lower()
    for use, words in _LAND_USE_WORDS.items():
        if fam == use or fam in words:
            return use
    # prose names a land use only for ground that is mostly open: a homes district whose
    # notes say "the orchard takes none of them" is not an orchard
    ch = p.get("character") if isinstance(p.get("character"), dict) else {}
    mostly_open = (int(p.get("structures") or 0) == 0
                   or float(ch.get("open_share") or 0) >= 0.5
                   or str(p.get("density") or "") == "sparse")
    if not mostly_open:
        return None
    blob = " ".join(str(p.get(k) or "") for k in ("notes", "purpose")).lower() \
        .replace("_", " ")
    for use in working:
        if in_text(blob, _LAND_USE_WORDS[use]):
            return use
    return None


def entity_of(part: dict | None, count: dict | None = None) -> dict:
    """`{"class", "counted", "unit", "land_use"}` for one defining part.

    `count` is the spec's explicit count (`{"n", "what", ...}`); a part is `counted` when
    it is a `building` entity whose unit is the count's subject class (house words are
    one class). A group of houses is a building entity too: its lots are what the count
    is spent on. A group whose land use is farmland/orchard/pasture is `land` even if
    the spec budgets a few farm cottages in it -- those cottages are counted, the ground
    is not."""
    p = part or {}
    kind = str(p.get("kind") or "plot")
    fam = str(p.get("family") or "").lower()
    # a land use is a fact about ground: an area or a group has one, a hall "on the
    # square" does not
    use = land_use_of(p) if kind in ("area", "group") or p.get("land_use") else None
    what = str(((count or {}) if isinstance(count, dict) else {}).get("what") or "").lower()
    house_unit = (not what) or what.rstrip("s") in [h.rstrip("s") for h in _HOUSE_FAMILIES] \
        or what in ("building", "buildings", "structure", "structures")
    if compound(p):
        cls, counted, unit = "compound", False, None
    elif kind == "edge":
        cls, counted, unit = "boundary", False, None
    elif kind == "point":
        cls, counted, unit = "point", False, None
    elif kind == "area":
        if fam in _LAND_FAMILIES or use in ("orchard", "pasture", "farmland", "garden",
                                             "grove"):
            cls, counted, unit = "land", False, None
        else:
            cls, counted, unit = "amenity", False, None
    elif kind == "group":
        if use in ("orchard", "pasture", "farmland", "grove", "garden") \
                and int(p.get("structures") or 0) == 0:
            cls, counted, unit = "land", False, None
        elif use in ("orchard", "pasture", "farmland", "grove", "garden"):
            # working land with a few buildings in it: the ground is land, the lots are
            # counted buildings of the fabric's unit
            cls, counted, unit = "land", bool(house_unit), ("house" if house_unit else None)
        else:
            cls, counted, unit = "building", bool(house_unit), ("house" if house_unit else None)
    else:
        # a plot: a house family is a counted building; a hall, temple, workshop is a
        # building of another unit and is counted only when the sentence counts it
        is_house = fam in _HOUSE_FAMILIES or str(p.get("function") or "") == "dwelling"
        unit = "house" if is_house else (fam or None)
        counted = bool(is_house and house_unit) or (bool(what) and what.rstrip("s") == fam.rstrip("s"))
        cls = "building"
    return {"class": cls, "counted": counted, "unit": unit, "land_use": use}


def entities(spec: dict) -> list:
    """Every defining part's entity binding, in spec order, with its name."""
    count = spec.get("explicit_count") if isinstance(spec.get("explicit_count"), dict) else None
    return [{"part": d.get("name"), "kind": d.get("kind"), "family": d.get("family"),
             **entity_of(d, count)} for d in spec.get("defining_parts") or []]


def land_use(part: dict | None) -> str:
    """What the ground of this district is given over to. `settled` where it does not say.

        Declared first -- `land_use` on the defining part, from `LAND_USES` -- then derived
        from the part's own name and notes, and `settled` otherwise. Derivation reads the
        part's prose because that is where a spec that means a farm belt says it does, and
        it reads nothing else: a *role* is what the buildings are for and never implies what
        lies between them.
        
    """
    said = str((part or {}).get("land_use") or "").strip().lower()
    if said in LAND_USES:
        return said
    import re as _re
    blob = " ".join(str((part or {}).get(k) or "")
                    for k in ("name", "notes", "purpose")).lower().replace("_", " ")
    if any(_re.search(rf"(?<![a-z]){_re.escape(w)}(?![a-z])", blob)
           for w in _FARMLAND_WORDS):
        return "farmland"
    return "settled"


def compounds(spec: dict) -> list:
    """The defining parts that are compounds, in spec order."""
    return [p for p in spec["defining_parts"] if compound(p)]


def core(spec: dict) -> dict | None:
    """The defining part at the middle: the one the ground is levelled for. A1.

        The same rule `find_site.innermost` has always applied -- the part whose
        relation is `centre`, and where two are, the one that is not a group -- lifted here
        because A1 gives that part a *say*: its own `needs` are what the core of the site is
        scored against, and the search reads them from this one answer.
        
    """
    at_centre = [p for p in spec["defining_parts"] if p["relation"] == "centre"]
    solid = [p for p in at_centre if p["kind"] != "group"]
    pool = solid or at_centre or spec["defining_parts"]
    return sorted(pool, key=lambda p: (p["kind"] == "group", p["name"]))[0] if pool \
        else None


def core_needs(spec: dict) -> dict:
    """The ground the **core** wants: the core part's own needs over the place's. A1.

        A place's `needs` describe the whole footprint and always did. What A1 adds is that
        the number that matters most -- how level the middle is -- belongs to the part that
        stands on it, and a spec that says so is a spec whose outer rings are free to be a
        hillside.
        
    """
    out = {k: v for k, v in spec["needs"].items()}
    p = core(spec) or {}
    for k, v in (p.get("needs") or {}).items():
        if v is not None:
            out[k] = v
    out["part"] = p.get("name")
    return out


def district_role(spec: dict, district: dict) -> str | None:
    """What one district of a place plan is for. A2.

        A place-level district carries `defines`: the name of the defining part it is part
        of. The role comes from there and from nowhere else, so a district is not asked to
        have an opinion about what it is -- the level that named it already said.

        None where a district defines nothing the spec knows, in which case no role check is
        made: a rule nobody can obey is not one to refuse a plan on.
        
    """
    if not district:
        return None
    want = district.get("defines")
    for p in (spec or {}).get("defining_parts") or []:
        if p["name"] == want:
            return p.get("role")
    return None


def outer_parts(spec: dict) -> list:
    """The defining parts that are not the core: the rings, the wall, the fields. A1."""
    c = core(spec)
    return [p for p in spec["defining_parts"] if not c or p["name"] != c["name"]]


def in_band(spec: dict, n: int) -> bool:
    lo, hi = spec["size_band"]
    return int(lo) <= int(n) <= int(hi)


def load(path: str) -> dict:
    return json.load(open(path))


def summary(spec: dict) -> str:
    """One line per fact, for a log and for the record."""
    lo, hi = spec["size_band"]
    rows = [f"{spec['kind']}: {spec['structures']} structures (band {lo}-{hi}), "
            f"footprint {spec['needs']['footprint']}x{spec['needs']['footprint']}, "
            f"voice {spec['voice'] or 'to be chosen against the ground'}"
            + (f" (authored)" if spec.get("authored_voice") else "")
            + f", form {spec.get('form') or 'any'}"]
    st = spec.get("setting") or {}
    if st.get("surface") or st.get("water") or st.get("relief") or st.get("biome"):
        rows.append(f"  setting: surface {st.get('surface') or 'any'}, water "
                    f"{st.get('water') or 'any'}, relief {st.get('relief') or 'any'}, "
                    f"biome {'/'.join(st.get('biome') or []) or 'any'}")
    if spec.get("invariants"):
        rows.append(f"  invariants: {spec['invariants']}")
    for p in spec["defining_parts"]:
        needs = {k: v for k, v in (p.get("needs") or {}).items() if v is not None}
        rows.append(f"  - {p['name']}: {p['count']} x {p['family']} as "
                    f"{'a compound' if compound(p) else p['kind']}, {p['relation']}"
                    + (f", holding {p['structures']} structures"
                       if p["structures"] else "")
                    + (f", {p['density']}" if p.get("density") else "")
                    + (f", forms {'/'.join(p['forms'])}" if p.get("forms") else "")
                    + (f", needs {needs}" if needs else "")
                    + (f", ring {p['ring']} share {p['share']:g}"
                       f"{' walled' if p.get('walled') else ''}"
                       f"{' voice ' + p['voice'] if p.get('voice') else ''}"
                       if p.get("ring") is not None else ""))
    if rings(spec):
        rows.append(f"  rings: {len(rings(spec))}, {len(walled_rings(spec))} walled, the "
                    f"centre's share {centre_share(spec):.2f}")
    if spec.get("scaled_from"):
        rows.append(f"  scaled from {spec['scaled_from']['structures']} by "
                    f"{spec['scaled_from']['factor']}")
    return "\n".join(rows)
