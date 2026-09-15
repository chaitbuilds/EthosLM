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

#: The families of form a place is built in. A type declares one of these and the plan
#: filters on it, so a place says which building tradition it is in and the palette is a
#: separate question answered against the ground. Two are regional and two are
#: functional -- see `pipeline.form_ok`.
FORMS = ("european_vernacular", "east_asian", "fortification", "civic")

#: How a defining part sits in the place. Free text is refused: a relation nothing can
#: act on is a comment, and A5 plans the place level off exactly these words.
RELATIONS = ("concentric", "centre", "edge", "perimeter", "gateway", "throughout",
             "quarter", "beside_the_centre")

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
PART_FIELDS = ("kind", "family", "relation", "count", "structures", "needs", "forms",
               "density", "role", "ring", "share", "walled", "voice", "notes")


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
    """The most things a place of this kind may stand up, **scaled by its ground**."""
    return int(round(CEILING["structures"] * _ground_ratio(kind)))


def _ground_ratio(kind: str | None) -> float:
    """How much more ground this kind is given than the ceiling it was registered on."""
    f = footprint_ceiling(kind)
    return (f * f) / float(CEILING["footprint"] ** 2)


def size_band_for(kind: str | None) -> tuple:
    """How many structures this kind is, as a band, **on the ground it is given**.

        A kind whose footprint ceiling is the registered one gets its band untouched, which
        is every kind but `city`.
        
    """
    lo, hi = SIZE_BANDS[kind]
    r = _ground_ratio(kind)
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
    m = re.search(rf"(\b(?:{'|'.join(_ABOUT_WORDS[:-1])})\s+)?"
                  rf"\b(\d{{1,4}}|{words})\b[\w\s-]{{0,20}}?\b{unit}\b", s)
    if not m:
        return None
    raw = m.group(2)
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
    read_ring_fields(d, out, f"{where} ({name})")
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
    """The ground one structure of this defining part takes, its density applied. A1."""
    from .placeplan import dense_plot
    d = part.get("density") or "medium"
    if d == "dense":
        return int(dense_plot()["columns"])
    return int(round(COLUMNS_PER_PLOT * DENSITIES.get(d, 1.0)))


def plot_share(part: dict) -> float:
    """How much of this defining part's district ground is plots."""
    from .placeplan import occupancy_shares
    return float(occupancy_shares()[part.get("density") or "medium"]["plot_share"])


def structures_for(columns: float, part: dict) -> int:
    """**The number of structures a piece of designed ground is asked for.**"""
    per = columns_per_plot(part)
    return max(0, int(float(columns) * plot_share(part) // per))


def read_spec(doc: dict, sentence: str | None = None) -> dict:
    """A place spec off a model's answer: checked, completed, and scaled. A1 + A2.

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
            "needs": _read_needs(doc.get("needs"), target, kind),
            "ceiling": dict(CEILING, footprint=footprint_ceiling(kind),
                            structures=structures_ceiling(kind)),
            "notes": str(doc.get("notes") or "")}
    if invariants:
        spec["invariants"] = str(invariants).strip()
    if ring_voices:
        spec["authored_voices"] = ring_voices
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


def _read_needs(got, structures: int, kind: str | None = None) -> dict:
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
        # resized.
        if got.get("footprint") is not None \
                and int(got["footprint"]) != want \
                and int(got["footprint"]) != min(want, footprint_ceiling(kind)) \
                and int(got["footprint"]) != min(want, CEILING["footprint"]):
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
    spec["size_band"] = [min(int(lo), spec["structures"]),
                         min(int(hi), cap)]
    spec["needs"]["footprint"] = min(footprint_for(spec["structures"], spec.get("kind")),
                                     fcap, int(spec["needs"]["footprint"]))
    spec["scaled_from"] = {
        **was, "to": {"structures": spec["structures"],
                      "size_band": list(spec["size_band"]),
                      "footprint": spec["needs"]["footprint"],
                      "per_part": {p["name"]: p["structures"]
                                   for p in spec["defining_parts"]}},
        "ceiling": dict(CEILING, footprint=fcap, structures=cap),
        "why": ("the spec asked for more than the ceiling allows; counts inside each "
                "defining part were reduced proportionally and no defining part was "
                "dropped")}
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
