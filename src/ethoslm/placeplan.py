"""So the plan is made **once per level**:

Each level is validated before the next is asked for, and a failing level goes back to
its own planner once. The levels are then assembled into the one tree
`pipeline.plan_parts` has read since A4, so nothing downstream knows or cares that the
plan was made in four calls rather than one.

The briefs are composed here rather than in `scripts/make_settlement_prompts.py` because
that script bakes `settlement.STATE` in at import from `$ETHOSLM_SETTLEMENT` and says so in
its own error message. These take their paths."""
from __future__ import annotations

import json
import math
import os

from . import circulate, pipeline, spec as spec_mod, styles
from .buildlib import Builder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: How much of a district's ground the plots in it may take, before the lanes between
#: them have nowhere to go. The circulation pass needs five blocks between footprints
#: and a district whose rectangles tile it has no lanes. Checked at the place level,
#: where the districts are drawn.
DISTRICT_FILL = 0.55

#: its farms and its fields together -- at the least. A farm belt is fields with farms
#: in them, not four farmsteads in a forest: the demo's agrarian districts were 24% of
#: their ring and the fields inside them a fraction of that. Three fifths, said in the
#: brief in columns and refused by name.
RURAL_COVER = 0.6

#: Three quarters, of the **count** and of the **cover** both. Two numbers about the
#: same ground, and a district may miss neither. A count alone does not say a district
#: is full -- twenty houses in one corner of a rectangle is twenty houses -- and a cover
#: alone does not say it is a quarter: one courtyard house drawn at its largest covers
#: as much as eight small ones. The brief states both, in the units this refuses in,
#: because a level is handed back at most once and a rule the planner cannot read is a
#: hand-back spent for nothing. Three quarters rather than all of it because the count
#: is derived from a rectangle and drawn on ground: the arterial runs through it, the
#: wall keeps its clearance, and a plot has to fit whole. nothing here was ever asked
#: for a fraction.
DISTRICT_MIN_FRACTION = 0.75

#: How big one area is, as a multiple of the district's own plot, per density word. A
#: dense quarter's open ground is cut between the rows -- an alley, a yard, a small
#: paved widening, each about the size of a house and a half -- and a sparse one's is a
#: grove or an orchard covering a dozen plots, capped at the largest square a committed
#: area type admits. The brief turns this into a count and a size in columns; nothing
#: here names a type, because which areas a district may draw is its role's table and
#: not a list in the library.
AREA_PLOTS = {"sparse": 12.0, "low": 6.0, "medium": 3.0, "dense": 1.5}

#: The one number the arithmetic below turns a size into a cover with.
PLOT_LANE = 5


def _tiled(side: int) -> float:
    """What share of the ground a grid of `side`-square rectangles covers.

        A rectangle laid out with `PLOT_LANE` between it and the next repeats every
        `side + PLOT_LANE`, so it covers `side^2 / (side + PLOT_LANE)^2` of the ground it
        sits on. This is the whole of why a dense district is covered **less** than a sparse
        one: at a house's scale the street is a third of the ground and at a field's scale
        it is a tenth.
        
    """
    s = max(1, int(side))
    return (s * s) / float((s + PLOT_LANE) ** 2)


#: Which role's plot types each density word's lot is measured against, where nothing
#: says. A `sparse` district is farmland and the rest are town quarters; a district that
#: carries its own role is measured against that and this is only the fallback for the
#: registered table, which is keyed by the word alone.
DENSITY_ROLE = {"sparse": "rural", "low": "urban", "medium": "urban", "dense": "urban"}


def occupancy_shares() -> dict:
    """E1).

        Three numbers per density word, from the lot the committed types declare
        (`density_lot`) and the ground the compiler lays round it (`fabric`):

            columns_per_plot  the lot itself, `density_lot`'s side squared
            plot_share        the lots' share of a block and its streets
            ground_cover      the lots, the courts and the open blocks' share of the same

        What this replaced, and why. The shares were `PLOT_CELLS x tiled(plot side)`, where
        `tiled` charges a **square** plot a five-block lane on all four sides -- a model of
        a village of freestanding cottages, and not what any compiled district is. A terrace
        house on a street needs its frontage, its depth and its share of the street: on the
        old arithmetic a dense house on a 10-square lot was charged 500 columns of its ring
        and the last city's dense ring came in **under** that at 451 and reads as detached
        houses on lawns in every frame of the look.

        `AREA_PLOTS` still says how big one piece of open ground is, capped at the largest
        square a committed area type admits; `PLOT_CELLS` is still what a spec call is shown
        as the word's place on the ladder, and is no longer what charges the ground.
        
    """
    big = largest_area()[0] or 48
    out = {}
    for word in spec_mod.PLOT_CELLS:
        fab = fabric(word, DENSITY_ROLE.get(word))
        per = int(fab["lot_columns"])
        p = int(fab["lot"][0])
        a = max(3, min(big, int(round((per * AREA_PLOTS[word]) ** 0.5))))
        out[word] = {
            "plot_cells": spec_mod.PLOT_CELLS[word], "plot_side": p, "area_side": a,
            "columns_per_plot": per, "area_columns": a * a,
            "columns_per_structure": fab["columns_per_structure"],
            "plot_share": fab["plot_share"], "ground_cover": fab["ground_cover"],
            "fabric": fab}
    return out


def fabric_fit(w: int, d: int, density: str, role: str | None = None,
               character: dict | None = None) -> int:
    """**How many structures the compiler's grid fits in a `w` by `d` rectangle.**
        The craft round, E1, found by a case.

        `spec.structures_for` divides ground by what one house of the fabric costs, which is
        right for a rectangle big enough to run a grid on and wrong for a strip: a district
        of 3,640 columns is eleven houses' worth of ground and holds four, because a block
        with its streets is 1,504 columns of it and the edges are most of the rest. A count
        about a rectangle has to be about that rectangle's **shape**, and asking a district
        for a number its own ground cannot lay is a hand-back nobody can answer -- which for
        a compiled district stops the run, because no model drew it.

        The same runs the compiler lays (`district_compile._runs`), the same lots along
        them, two rows to a block where it is deep enough for two, less the open blocks and
        the back rows the courtyard blocks give up.
        
    """
    from . import district_compile as dc
    f = fabric(density, role, character)
    U, V = max(int(w), int(d)), min(int(w), int(d))
    lw, ld = f["lot"]
    gap, row_gap, street = f["gap"], f["row_gap"], f["street"]
    cols = dc._runs(0, U - 1, street, f["block"], lw)
    rows = dc._runs(0, V - 1, street, f["block_depth"], ld)
    if not cols or not rows:
        return 0
    along = [max(0, (b - a + 1 + gap) // (lw + gap)) for a, b in cols]
    across = [2 if (b - a + 1) >= 2 * ld + row_gap else 1 for a, b in rows]
    # the lots of each block of the grid, the shortest runs included: the end of a run
    # is a block one lot long, and counting the grid as `lots x rows` credits it a full
    # one. Sorted, because the blocks the compiler takes for a landmark and for open
    # ground are whole blocks and the landmark's is the one nearest the middle.
    per = sorted((a * b for a in along for b in across), reverse=True)
    n = len(per)
    take = int(round(f["open_share"] * n)) + (1 if (character or {}).get("landmarks")
                                              else 0)
    per = per[min(take, n):]
    courts = int(round(f["courtyard_share"] * n))
    for i in range(min(courts, len(per))):
        per[i] = per[i] - per[i] // 2
    return max(0, int(sum(per)))


def ground_cover(density: str | None) -> float:
    """How much of a district of this density is plots and areas together."""
    return float(occupancy_shares()[density or "medium"]["ground_cover"])

#: **How much ground per standing structure a ring of each density may have**, as a
#: ceiling the readout reports against. Not a fourth table of taste: it is what one
#: house of that density's own fabric costs -- its frontage, its depth and its share of
#: the street (`fabric`) -- over `RING_COVERAGE`, which is the least of an annulus its
#: districts may cover. A ring laid out by `concentric_layout` and filled to its brief
#: comes in under it by however much better than the floor its coverage is.
def columns_per_structure_ceiling() -> dict:
    """The ceiling per density word, derived. See `COLUMNS_PER_STRUCTURE_CEILING`."""
    return {w: int(round(fabric(w, DENSITY_ROLE.get(w))["columns_per_structure"]
                         / RING_COVERAGE))
            for w in spec_mod.PLOT_CELLS}


#: phase 4 is what makes it true and this is where it is written down. 34% -- and from
#: the air it reads as a compound among compounds rather than as the thing four terraces
#: rise to. Half is the least a centre can be and still be the centre.
CENTRE_SHARE_MIN = 0.5

#: The levels, in the order they are planned. The place first, because a district cannot
#: be drawn until the wall it is inside has been; then the **compounds**.
LEVELS = ("place", "compounds", "districts")

#: The smallest a compound may be on a side. A compound holds a wall, a gate, two halls
#: and a court with the lanes between them, and the smallest plot two halls need with
#: their clearance and a wall of its own round them is about this.
COMPOUND_MIN = 24

#: the plateau was cut for the part and a compound in one corner of it has left the cut
#: for nothing.
COMPOUND_PLATEAU_SHARE = 0.75

#: **Registered.** The least halls (plots) a compound of the **default** composition
#: holds -- two, with a closed wall of its own, a gate on it and a court. A great thing
#: is a composition; one hall inside a fence is a house with a garden. v2, C0: what a
#: compound is made of is its **family's** row of `spec.COMPOSITIONS` -- a monument is
#: one thing and its setting, a castle a keep inside its wall -- and this is the floor
#: of the row a family not in that table gets, which is the palace's.
COMPOUND_MIN_HALLS = spec_mod.COMPOSITION_DEFAULT["halls"]

#: **How thickly a compound is built up**, as a density word, so its composition is the
#: same arithmetic a district's is. `COMPOUND_MIN_HALLS` is a *floor* -- two halls and a
#: court -- and nothing ever scaled it, so a palace precinct of 154 blocks square would
#: be laid out with the same three parts a precinct of 24 is: five buildings on a plaza,
#: which is what the ground look saw. `low` is what a great thing is: large parts with
#: ground between them, gardens and secondary courts rather than a street.
COMPOUND_DENSITY = "low"

#: How far inside its rectangle a compound's wall runs, at the least. A gate's pad
#: straddles the wall, so a wall on the rim puts the gate's pad outside the compound;
#: three is a gate's half-width plus one.
COMPOUND_WALL_INSET = 3

#: How near the road's arrival a compound's gate has to stand, in blocks: the arterial
#: is routed to the compound's edge before the compound is planned, and its gate is
#: where the place comes in.
COMPOUND_GATE_REACH = 8

#: The roles a compound of the **default** composition admits over its own part's role:
#: a walled compound has a wall and a gate whatever it is for, so the `defensive` types
#: stand in it; `pipeline.UNIVERSAL_ROLES` still applies. v2, C0: each family's row of
#: `spec.COMPOSITIONS` says what it admits (`compound_composition`), and an unwalled
#: precinct admits nothing over its own role.
COMPOUND_ROLES = spec_mod.COMPOSITION_DEFAULT["admits"]


def compound_composition(part: dict | None = None, *, spec: dict | None = None,
                         comp: dict | None = None) -> dict:
    """What this compound is made of, at the least: its family's row of
        `spec.COMPOSITIONS`. v2, C0.

        Found from the defining part where the caller has it, else from the plan's compound
        rectangle (`comp`, by `_answers`), else from the spec's compound at the centre, else
        the default row. `margin` is how far inside its rectangle the composition stands:
        the wall's inset and its clearance where the compound is walled, one plot clearance
        where it is not.
        
    """
    if part is None and comp is not None and spec:
        part = next((d for d in spec_mod.compounds(spec) if _answers(comp, d)), None)
    if part is None and spec:
        cs = spec_mod.compounds(spec)
        part = next((d for d in cs if d.get("relation") == "centre"), None) \
            or (cs[0] if cs else None)
    out = spec_mod.composition(part)
    plot_clear, wall_clear = _clearances()
    out["margin"] = (COMPOUND_WALL_INSET + wall_clear + 1) if out["walled"] \
        else plot_clear + 1
    return out


# ------------------------------------------------- how much ground a compound needs

def largest_plot(types: list | None = None) -> tuple:
    """(side, name): the largest square plot any committed plot type admits.

        Read off the files, honouring `NEEDS["except"]` -- a size the sweep found broken is
        not a size the library may size ground for. This is the biggest single building the
        place can hold anywhere, and so the thing a great thing has to be greater than.
        
    """
    d = os.path.join(ROOT, "types")
    best = (0, None)
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or name.startswith("_") \
                or (types is not None and name not in types):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if decl["kind"] != "plot":
            continue
        needs = decl.get("needs") or {}
        a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
        ex = set(needs.get("except") or ())
        clean = [v for v in range(min(a, b), max(c, e) + 1)
                 if a <= v <= c and b <= v <= e and v not in ex]
        if clean and max(clean) > best[0]:
            best = (max(clean), name)
    return best


#: The role whose plot types a dense quarter is made of. A dense ring is a **town**
#: quarter -- houses, shops, workshops -- and what a farm or a fortification needs is a
#: different question that `RURAL_COVER` and the defensive types answer.
DENSE_ROLE = "urban"


def dense_plot(types: list | None = None) -> dict:
    """The plot one house of a **dense** quarter stands on, read off the committed types.

        What a dense quarter is made of is the **smallest house the library can build a
        quarter of**: among the committed plot types that belong in an urban district, the
        one whose band *tops out* soonest, because a quarter of that type cannot be drawn
        any larger than its own ceiling and every other urban type can come down to meet it.
        That type at its largest admitted pad, plus `site()`'s inset on four sides, is the
        plot. `row_house` tops out at a 6-square pad, which is a 10-square plot: 100
        columns against 158.

        Returns the columns and every term of it, so the record says why rather than what.
        
    """
    d = os.path.join(ROOT, "types")
    best = None
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or name.startswith("_") \
                or (types is not None and name not in types):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if decl["kind"] != "plot" or decl.get("role") != DENSE_ROLE:
            continue
        needs = decl.get("needs") or {}
        a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
        ex = set(needs.get("except") or ())
        clean = [v for v in range(min(a, b), max(c, e) + 1)
                 if a <= v <= c and b <= v <= e and v not in ex]
        if not clean:
            continue
        if best is None or max(clean) < best[0]:
            best = (max(clean), name)
    if best is None:                      # no urban plot type: the village constant
        return {"columns": int(round(spec_mod.COLUMNS_PER_PLOT
                                     * spec_mod.DENSITIES["dense"])),
                "pad": None, "plot": None, "type": None,
                "why": "no committed plot type is urban; the village constant scaled by "
                       "the density word stands in"}
    pad, name = best
    plot = pad + 2 * Builder.SITE_INSET
    return {"columns": int(plot * plot), "pad": int(pad), "plot": int(plot),
            "type": name, "inset": int(Builder.SITE_INSET),
            "why": (f"{name} is the {DENSE_ROLE} plot type whose band tops out soonest, "
                    f"at a {pad}x{pad} pad; site()'s inset of {Builder.SITE_INSET} on "
                    f"four sides makes that a {plot}x{plot} plot")}


def largest_area(types: list | None = None) -> tuple:
    """(side, name): the largest square an `area` type admits, off the files.

        What `largest_plot` is for a building, for the ground between them: the biggest
        single piece of open ground the library can make. A brief that asked a sparse
        district for one area of 5,400 columns would be asking for a 73-block square and no
        committed type builds one.
        
    """
    d = os.path.join(ROOT, "types")
    best = (0, None)
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or name.startswith("_") \
                or (types is not None and name not in types):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if decl["kind"] != "area":
            continue
        needs = decl.get("needs") or {}
        a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
        ex = set(needs.get("except") or ())
        clean = [v for v in range(min(a, b), max(c, e) + 1)
                 if a <= v <= c and b <= v <= e and v not in ex]
        if clean and max(clean) > best[0]:
            best = (max(clean), name)
    return best


def admitted_plot_sides(role: str | None = None) -> list:
    """Every **plot** side a committed plot type of this role stands on, sorted.

        In plot columns -- the pad plus `site()`'s inset on four sides -- which is the unit
        a plan is drawn in and the unit `Builder.pad_extent` converts. `role` of `None` is
        every plot type on disk. The craft round, E1: a lot the library cannot build on is
        not a lot, and until this the only place on the ladder that was read off the files
        was its dense end (`dense_plot`).
        
    """
    d = os.path.join(ROOT, "types")
    out: set = set()
    i = 2 * Builder.SITE_INSET
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or name.startswith("_"):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if decl["kind"] != "plot":
            continue
        if role is not None and decl.get("role") != role:
            continue
        needs = decl.get("needs") or {}
        a, b, c, e = needs.get("footprint", (3, 3, 3, 3))
        ex = set(needs.get("except") or ())
        for v in range(min(a, b), max(c, e) + 1):
            if a <= v <= c and b <= v <= e and v not in ex:
                out.add(v + i)
    return sorted(out)


def density_lot(density: str, role: str | None = None) -> dict:
    """**The lot one structure of a district of this density stands on**, off the
        committed types. The craft round, E1.

        `dense_plot()` measured the dense end of this ladder off the files -- the urban plot
        type whose band tops out soonest, at its top, plus `site()`'s inset on four sides --
        and the other three words were `spec.COLUMNS_PER_PLOT`, a townhouse's 15x15 written
        for a village, times `spec.DENSITIES`. One end of the ladder was the
        library's and the other three were a constant nothing had ever measured.

        The whole ladder is the library's now: the dense lot is the **anchor**, a density
        word is its place on `spec.DENSITIES` **in area**, and the side that comes out is
        clamped to one the role's own plot types declare (`admitted_plot_sides`: nearest,
        the larger on a tie). A word asking for more ground never gets a smaller lot than
        one asking for less, because `DENSITIES` is monotone and the clamp is nearest.
        
    """
    anchor = dense_plot()
    base = float(anchor["columns"])
    dens = spec_mod.DENSITIES
    want = base * (dens.get(density, 1.0) / dens["dense"])
    side = max(3, int(round(want ** 0.5)))
    sides = admitted_plot_sides(role)
    got = min(sides, key=lambda v: (abs(v - side), -v)) if sides else side
    return {"density": density, "role": role, "target_side": side, "side": int(got),
            "columns": int(got) * int(got), "anchor_columns": int(anchor["columns"]),
            "anchor_type": anchor.get("type"),
            "why": (f"the dense lot is {anchor.get('plot')} square ({anchor['why']}); "
                    f"`{density}` is {dens.get(density, 1.0)}/{dens['dense']} of that in "
                    f"area, which is {side} a side, and the nearest side a "
                    f"{role or 'committed'} plot type admits is {got}")}


def fabric(density: str, role: str | None = None, character: dict | None = None) -> dict:
    """**What a district of this density is made of, and what one structure of it costs
        in ground.** The craft round, E1.

        A terrace house on a street needs its frontage, its depth and its share of the
        street, and nothing else. What charged it before was a **square** plot with a
        five-block lane on all four sides (`_tiled`) over `spec.PLOT_CELLS`: a dense house
        on a ten-square lot was charged 500 columns of its ring for 100 columns of plot, and
        the last city's dense ring came in at 451 a house -- **under** that ceiling -- and
        reads in every frame as detached houses on lawns.

        This is the compiler's own arithmetic (`ethoslm.district_compile`), which is what
        actually lays the ground: lots `w` by `ld` along a street, `gap` between flanks,
        two rows back to back with `row_gap` between their backs, `BLOCK_LOTS` lots to a
        block, a street of `PLOT_LANE` round every block, and the open and courtyard blocks
        the density's own character takes out of the houses. One block and its share of the
        streets round it is `(block + street) x (block_depth + street)`; the houses on it
        are `lots_per_block x (2 - 2 x open_share - courtyard_share)`, because an open
        block has none and a courtyard block's back row is the court its front row shares.

        `character` is a district's own where it has one -- a row of party walls is charged
        its frontage and no gap at all -- and the density's registered default otherwise,
        which is what `columns_per_structure_ceiling()` is the table of.
        
    """
    from . import district_compile as dc
    ch = dict(spec_mod.CHARACTER_DEFAULTS[density])
    for k, v in (character or {}).items():
        if v is not None and k in ch:
            ch[k] = v
    lot = density_lot(density, role)
    w = ld = int(lot["side"])
    attached = bool(ch.get("attached"))
    open_front = ch.get("frontage") == "open"
    house = None
    if attached:
        _t, every = types_card()
        for n, d in dc.house_types(every, role):
            if not d.get("attached"):
                continue
            got = dc._attached_lot(d, w, ("west", "east"))
            if got:
                house, (w, ld) = n, got
                break
        attached = house is not None
    gap = 0 if attached else (PLOT_LANE if open_front else dc.LOT_GAP)
    row_gap = PLOT_LANE if open_front else dc.LOT_GAP
    per_block = int(ch.get("block_lots") or dc.BLOCK_LOTS.get(density, 3))
    block = max(w, per_block * w + (per_block - 1) * gap)
    depth = 2 * ld + row_gap
    street = PLOT_LANE
    total = float((block + street) * (depth + street))
    op, cs = float(ch.get("open_share") or 0.0), float(ch.get("courtyard_share") or 0.0)
    houses = per_block * (2.0 - 2.0 * op - cs)
    per = total / houses if houses > 0 else total
    lots_area = houses * w * ld
    court_area = block * ld * cs
    open_area = block * depth * op
    return {"density": density, "role": role, "type": house, "attached": attached,
            "frontage": ch.get("frontage"), "lot": [int(w), int(ld)],
            "lot_columns": int(w * ld), "gap": int(gap), "row_gap": int(row_gap),
            "lots_per_block": int(per_block), "block": int(block),
            "block_depth": int(depth), "street": int(street),
            "open_share": op, "courtyard_share": cs,
            "houses_per_block": round(houses, 4),
            "block_columns": int(total),
            "columns_per_structure": round(per, 1),
            "plot_share": round(lots_area / total, 4),
            "ground_cover": round((lots_area + court_area + open_area) / total, 4),
            "why": (f"{int(per_block)} lots of {int(w)}x{int(ld)} a side of a block "
                    f"{int(block)} long and {int(depth)} deep, a street of {street} "
                    f"round it, {op:g} of the blocks open and {cs:g} courtyard: "
                    f"{int(total)} columns hold {houses:g} houses")}


def _clearances(types: list | None = None) -> tuple:
    """(the plot clearance a composition has to leave, the least a wall asks for)."""
    d = os.path.join(ROOT, "types")
    plot, wall = 0, None
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or name.startswith("_") \
                or (types is not None and name not in types):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        c = int((decl.get("needs") or {}).get("clearance", 0))
        if decl["kind"] == "plot":
            plot = max(plot, c)
        elif decl["kind"] == "edge":
            wall = c if wall is None else min(wall, c)
    return plot, (4 if wall is None else wall)


def centre_side(spec: dict | None, site_side: int | None) -> int:
    """The side of the square the spec's own **centre share** asks for, or 0."""
    if not spec or not site_side:
        return 0
    if not spec_mod.rings(spec):
        return 0
    return int(round(int(site_side) * math.sqrt(spec_mod.centre_share(spec))))


def compound_ground(types: list | None = None, *, spec: dict | None = None,
                    site_side: int | None = None, part: dict | None = None) -> dict:
    """How big the level square a **compound** stands on has to be, and why.

        So the number is the library's, and it is arithmetic over what a compound is
        registered to hold and what the committed types are measured to need -- never a
        model's guess and never a place's name:

          * the composition (its family's halls and courts, `spec.COMPOSITIONS`; v2, C0)
            laid out as squarely as it goes, every one of them at the largest plot the
            types admit, with the clearance the plan validator holds two plots to between
            them;
          * the wall's own inset (`COMPOUND_WALL_INSET`) and the clearance a wall asks of
            anything standing inside it, on all four sides -- or, where the family is not
            walled, one plot clearance;
          * and never below `COMPOUND_MIN`, nor below the square that satisfies the place
            read's own registered footprint margin over that same largest plot -- because a
            compound that cannot be monumental on the ground it is given is a compound the
            planner is being set up to fail.

        Returns the side and every term of it, so the record says why rather than what.
        
    """
    from .placeread import MONUMENT_FOOTPRINT_MARGIN
    side, biggest = largest_plot(types)
    plot_clear, wall_clear = _clearances(types)
    made_of = compound_composition(part, spec=spec)
    n = made_of["halls"] + made_of["courts"]      # the halls and the courts
    g = math.ceil(math.sqrt(n))
    grid = g * side + (g - 1) * (plot_clear + 1)
    margin = (COMPOUND_WALL_INSET + wall_clear + 1) if made_of["walled"] \
        else plot_clear + 1
    holds = grid + 2 * margin
    monumental = math.ceil(math.sqrt(MONUMENT_FOOTPRINT_MARGIN * side * side))
    # What a place's rings leave over is the centre's, and the thing four terraces rise
    # to should be that square and not the smallest one the committed types fit in.
    declared = centre_side(spec, site_side)
    want = max(holds, monumental, COMPOUND_MIN, declared)
    cap = Builder.plateau_max(site_side)
    return {"side": int(min(want, cap)),
            "monumental_columns": int(math.ceil(MONUMENT_FOOTPRINT_MARGIN * side * side)),
            "wanted": int(want), "capped": want > cap, "cap": int(cap),
            "declared": int(declared),
            "largest_plot": [int(side), biggest], "grid": [g, int(grid)],
            "holds": int(holds), "monumental": int(monumental),
            "parts": n, "plot_clearance": plot_clear, "wall_clearance": wall_clear,
            "family": made_of["family"], "walled": bool(made_of["walled"]),
            "why": (f"{n} parts of {side}x{side} ({biggest}) in a {g}x{g} grid with "
                    f"{plot_clear + 1} between them is {grid}"
                    f" (a {made_of['family']}: {made_of['halls']} hall(s) and "
                    f"{made_of['courts']} court(s))"
                    + (f"; a wall inset {COMPOUND_WALL_INSET} keeping {wall_clear} clear "
                       f"adds {margin} a side" if made_of["walled"] else
                       f"; unwalled, one plot clearance adds {margin} a side")
                    + f"; the place read's {MONUMENT_FOOTPRINT_MARGIN:g}x footprint "
                    f"margin over {side}x{side} wants {monumental}"
                    + (f"; the spec's centre share wants {declared}" if declared else "")
                    + f"; the bound on this site is {cap}")}


def _monumental_columns(types: list | None = None) -> dict:
    """What "materially larger than the largest single building" is, in columns.

        The place read's own `MONUMENT_FOOTPRINT_MARGIN` over the largest plot the committed
        types admit -- the same two numbers `compound_ground` sizes the ground from, so the
        plan-time refusal and the ground the search cut cannot disagree.
        
    """
    from .placeread import MONUMENT_FOOTPRINT_MARGIN
    side, name = largest_plot(types)
    columns = int(math.ceil(MONUMENT_FOOTPRINT_MARGIN * side * side))
    return {"columns": columns, "side": int(side), "type": name,
            "margin": MONUMENT_FOOTPRINT_MARGIN,
            "side_needed": int(math.ceil(math.sqrt(columns)))}


# ------------------------------------------------------------------- the ground

def site_grids(site: dict) -> str:
    """The terrain briefing, from a prepared `site.json`."""
    X, Z = site["origin"]
    S = site["size"]
    mean = "\n".join("    " + " ".join(f"{v:3d}" for v in row)
                     for row in site["mean_grid"])
    rough = "\n".join("    " + " ".join(f"{v:3d}" for v in row)
                      for row in site["roughness_grid"])
    surf = ", ".join(f"{k} ({v})" for k, v in
                     sorted(site["surface_blocks"].items(), key=lambda kv: -kv[1]))
    return f"""The site is x {X} to {X + S - 1}, z {Z} to {Z + S - 1} — {S} by {S} blocks.
Surface height runs from y={site['stats']['min']} to y={site['stats']['max']}. That is
{site['stats']['relief']} blocks of relief.

Mean surface height per {S // 12}-block cell. Rows run north to south (increasing z),
columns west to east (increasing x):

{mean}

Roughness within each of those cells (highest minus lowest block in the cell). A cell
above about 25 contains a cliff or a steep face; you cannot put a building flat on it:

{rough}

Surface materials: {surf}.
"""


def types_card(names=None, form=None, role=None, *, forms=None, roles=None) -> tuple:
    """(the table of types a plan may name, {name: declaration}).

        `forms` and `roles`, where given, are lists a type may match **any** of -- a
        compound admits the tradition of the place and the ones its own defining part
        names, and its part's role plus the defensive types -- and `form` and `role` are the
        one-word cases every district brief has always been composed with.

        Read off the committed files, because a second copy of a type's parameters is a
        second chance for a plan to ask for one that does not exist.

        Filtered by **form**, never by palette: a type says which family of form it is and
        nothing about materials, so the question a brief asks here is whether this building
        belongs in this kind of place, and the answer no longer depends on which of seven
        literals somebody put at the top of the file.
    """
    d = os.path.join(ROOT, "types")
    rows = ["| type | kind | form | role | params |", "|---|---|---|---|---|"]
    decls = {}
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or (names is not None and name not in names) \
                or (names is None and name.startswith("_")):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if forms is not None:
            if not any(pipeline.form_ok(decl["form"], f_) for f_ in (forms or [None])):
                continue
        elif not pipeline.form_ok(decl["form"], form):
            continue
        if roles is not None:
            if not any(pipeline.role_ok(decl.get("role"), r) for r in (roles or [None])):
                continue
        elif not pipeline.role_ok(decl.get("role"), role):
            continue
        decls[name] = decl
        kind = decl["kind"] + (", passage" if decl["passage"] else "")
        params = ", ".join(
            f"{k}: " + (f"{v[1]}..{v[2]}" if v[0] == "int" else "|".join(map(str, v[1])))
            for k, v in sorted(decl["params"].items())) or "none"
        rows.append(f"| `{name}` | {kind} | {decl['form']} | "
                    f"{decl.get('role') or '-'} | {params} |")
    return "\n".join(rows), decls


#: A spec names a **family** and not a type -- it is written before the site is chosen
#: and before a type is authored -- and the level that turns one into the other is the
#: place planner, which is given the whole table of committed types and picks. Nothing
#: here maps the two, on purpose: a table in this file would be a second opinion about
#: which type is a keep, and `types_card` already hands the planner the only one that
#: matters.


# ------------------------------------------------------------------ the briefs

PLACE_LEVEL = """# Plan a place: the level above the buildings

## What was asked for

> {sentence}

That sentence has already been read into a **place spec**, and you are not being asked to
re-read it. This is what it says, and it is what you are planning:

{spec}

## The site — chosen by the system for this place, on these needs

{grids}

{site_note}

{voice}

## What you are deciding, and what you are not

You are planning the **place**, not its buildings. Two things, and nothing else:

**1. The defining parts.** Every one of the defining parts above becomes a leaf of the
plan with real geometry: an `edge` for a wall, a `point` for a gate, an `area` for a
square, a `plot` for a keep or a hall. **Every one of them must be present.** They are
what makes this place the place the sentence asked for; a walled town without a closed
wall is not a walled town, and the finished place is checked against this spec by a pass
that does not run a model.

**2. The districts.** The ground left over is divided into districts: rectangles of land
that later calls will fill with houses. You are not placing a single house. A district is
a piece of ground with a name, a purpose, and how many structures it should hold; the
call that plans it is given its rectangle and nothing outside it.
{compounds}
## Rules the geometry is checked against, before anything is built

- Everything lies inside the site.
- A wall that the spec calls a wall is **one closed loop** — its path's last vertex is
  its first — and every gate stands **on** it, at a cell of its path.
- Every segment of an edge runs along x or along z. A corner is a vertex.
- No two districts overlap; no district overlaps a defining part; a district is at least
  {district_min} blocks on a side.
- A district holding *n* structures needs about **{per_structure} columns of ground for
  each of them** -- that is a {plot_side}x{plot_side} plot each, and the rest is the
  lanes between them. Give it the room: a district asked for twenty houses and drawn
  40x40 will be handed back to you.
{density_note}
- Leave the ground between things: the circulation pass routes the lanes and it needs
  five blocks between footprints, more where it is steep -- and between any two parts
  at least the larger of their two **keeps clear** figures in the table, plus one.
{concentric_rules}

## The types the buildings will be made of

Every leaf you place names one of these and a seed. The districts do not — the call that
plans each district chooses its own.

{types}

{needs}

## Output

Write a single JSON file to {out} with this shape:

{{
  "intent": "a few sentences: what this place is, why it is here, what should make it
             read as one place. Every later call reads this and nothing else about your
             reasoning.",
  "centre": "the name of the part this place is centred on",
  "voice": "{voice_name}",
  "palette": {{ "role": "material" }},
  "circulation_material": "one material family for the lanes",
  "parts": [
    {{"kind": "edge", "name": "town_wall", "type": "wall", "seed": 1, "params": {{}},
      "path": [[0,0],[0,0],[0,0],[0,0],[0,0]], "width": 3,
      "notes": "why it runs where it runs"}},
    {{"kind": "point", "name": "north_gate", "type": "gate_tower", "seed": 2,
      "params": {{}}, "at": [0,0], "facing": "north",
      "notes": "which way you go through it"}},
    {{"kind": "area", "name": "market", "type": "square", "seed": 3, "params": {{}},
      "x0": 0, "z0": 0, "x1": 0, "z1": 0, "notes": "..."}},
    {{"kind": "plot", "name": "keep", "type": "keep", "seed": 4, "params": {{}},
      "x0": 0, "z0": 0, "x1": 0, "z1": 0, "notes": "..."}}
  ],
  "districts": [
    {{"name": "short_slug", "x0": 0, "z0": 0, "x1": 0, "z1": 0,
      "structures": 0,
      "defines": "the name of the defining part this district is part of, where the
                  spec names one -- that is what fixes how much ground each of its
                  structures gets. Leave it out only for a district the spec does not
                  name.",
      "purpose": "what lives and happens here, in plain words — the call that plans it
                  reads this and turns it into what kinds of building go in it",
      "notes": "how it sits on its ground and against its neighbours"}}
  ],
  "compounds": [
    {{"name": "the defining part's own name", "defines": "the same name",
      "x0": 0, "z0": 0, "x1": 0, "z1": 0,
      "notes": "what this great thing is and how it sits -- the call that plans its
                halls, wall, gates and courts reads this"}}
  ]
}}

The `structures` across all your districts should come to about {structures}.

Use the Write tool. Output only the file; reply with the number of defining parts and
the number of districts.
"""


DISTRICT_LEVEL = """# Plan one district of a place

## The place this is part of

> {sentence}

{intent}

The place level has already been planned. **Centre:** {centre}. **Lanes:**
{lane_material}. These parts are already in the plan and their ground is taken:

{standing}

## The site

{grids}

{voice}

## Your district

**{name}** — x {x0}..{x1}, z {z0}..{z1} ({w} by {d} blocks).

{purpose}

{notes}

It should hold about **{structures}** structures.

{arterial}

## What you are deciding

The plots inside this district, and nothing else. Every plot names a type from the table
below and a seed, and is built by instantiating that type: there is no builder in this
run and no building is written by hand.

- Every plot lies **inside your district's rectangle** and nowhere else.
- No plot overlaps another, and none overlaps a part listed as already standing.
- **Nothing stands on the arterial.** The road through your district was routed before
  you were asked, it is how the rest of the place reaches you, and it is not yours to
  move: leave its columns clear and draw your plots off it.
- Leave at least five blocks between footprints for the lanes, more where the roughness
  grid is high — a route across a rise needs length to climb it — and between any two
  parts at least the larger of their two **keeps clear** figures in the table, plus
  one: the great wall keeps eight clear, so nothing stands within nine of it.
- **The plot you draw is not the ground the building gets.** Draw at least the smallest
  plot the type needs and at most the largest; the table below has the library's inset
  already added.
- `seed` is what makes two instances of one type different buildings. **No two adjacent
  plots may share both `type` and `seed`.**
- Group your plots into **quarters** — two or three of them — so the place has a grain
  rather than a grid. A quarter is a group with children and no geometry of its own.

{role_note}

{cover_note}

{types}

{needs}

## Output

Write a single JSON file to {out} with this shape:

{{
  "notes": "what this district is, in a sentence or two",
  "quarters": [
    {{"name": "short_slug", "notes": "what this quarter is for",
      "plots": [
        {{"kind": "plot", "name": "short_slug", "type": "<a type above>", "seed": 1,
          "params": {{"storeys": 2}},
          "x0": 0, "z0": 0, "x1": 0, "z1": 0,
          "notes": "what this one is for and how it sits on its ground"}}
      ]}}
  ]

`kind` is **the kind the type you name builds**, and the table above says which that is
for every one of them. Almost all of them are `plot` -- a building on a piece of ground.
One that says `area` is a rectangle brought to one level with no door and no rooms in it:
a field, a square, a yard. Name the type you want; the leaf's kind is taken from the
type, so a `field` or a `square` is an `area` whatever you write.
}}

Use the Write tool. Output only the file; reply with the number of plots.
"""


def place_brief(spec: dict, site: dict, out_path: str, types: list | None,
                voice: str, plateau: dict | None = None) -> str:
    """The place-level brief. One call, and the level never had."""
    table, decls = types_card(types, spec.get("form"))
    from .buildlib import Builder
    return PLACE_LEVEL.format(
        sentence=spec["sentence"], spec=spec_mod.summary(spec),
        grids=site_grids(site), site_note=_site_note(site, plateau),
        compounds=_compound_note(spec, plateau),
        voice=styles.voice_card(voice, site.get("surface_blocks")),
        voice_name=voice, types=table,
        needs=pipeline.needs_table(decls),
        district_min=int(2 * Builder.SITE_INSET + 24),
        # What the planner has to draw against is the **district** area a structure
        # needs, which is its plot divided by how much of a district may be plots. Two
        # numbers, one sentence, and quoting the plot alone understated it by 45%.
        per_structure=int(spec_mod.COLUMNS_PER_PLOT / DISTRICT_FILL),
        plot_side=int(round(spec_mod.COLUMNS_PER_PLOT ** 0.5)),
        density_note=_density_note(spec), concentric_rules=_concentric_rules(spec),
        structures=spec["structures"], out=out_path)


def _kind_ok(part: dict, d: dict, decls: dict) -> bool:
    """Does this leaf answer this defining part's kind? The spec's, **or its type's**.

        A spec names a *family* and not a type, because it is written before the site is
        chosen and before a type is authored -- and for exactly the same reason it cannot
        know what **kind** of part that family will turn out to be built as. Asked for Ba
        Sing Se, the spec call named the palace a `plot`, which is what a palace sounds
        like; the palace that was then written is an `area`, because a compound of three
        halls inside a courtyard wall is a rectangle of ground and not a building on a plot.
        Holding the plan to the spec's guess would have deadlocked it: a plot leaf is
        refused by the type check and an area leaf by this one.

        So the kind a leaf may be is the spec's or the one its committed type declares. What
        the spec is answerable for -- that a palace is here, at the centre, and there is one
        of it -- is untouched.
        
    """
    kind = part.get("kind", "plot")
    if kind == d["kind"]:
        return True
    decl = decls.get(part.get("type"))
    return bool(decl and decl.get("kind", "plot") == kind)


def _district_part(spec: dict, district: dict) -> dict:
    """The defining part a district answers, by its own `defines`, or a bare one. A1.

        A district says which ring it is; where it does not, it gets the place's own number,
        which is `DENSITIES["medium"]` and is what every round before this one used.
        
    """
    name = district.get("defines")
    for p in spec.get("defining_parts", []):
        if p["kind"] == "group" and (p["name"] == name
                                     or (not name and p["name"] == district.get("name"))):
            return p
    return {"density": None}


def _density_note(spec: dict) -> str:
    """What each defining part's own density costs in ground. A1.

        The number above is the place's; this is the part's, and it is here because a spec
        that says one ring is dense and another is farmland has said something the planner
        can only act on if somebody turns the word into columns. That arithmetic is the
        library's -- `spec.columns_per_plot` -- and this is where it is quoted.
        
    """
    rows = [p for p in spec["defining_parts"] if p.get("density")
            and p["kind"] == "group"]
    if not rows:
        return ""
    out = ["- **The rings are not all built alike, and the spec says how.** A district "
           "answering one of these is drawn to *its* number and not to the one above:"]
    for p in rows:
        out.append(f"    - `{p['name']}` is **{p['density']}**: about "
                   f"{int(spec_mod.columns_per_plot(p) / DISTRICT_FILL)} columns of "
                   f"district for each of its {p['structures']} structures"
                   + (f" — {p['notes']}" if p.get("notes") else ""))
    return "\n".join(out)


def _concentric_rules(spec: dict) -> str:
    """The rules a place of rings is checked against, stated. A2.

        Silent where nothing is concentric. A rule that is checked and not stated is a plan
        handed back once for free, and this level is handed back at most once in total.
        
    """
    from .placeread import GREAT_WALL_HEIGHT
    rings = [p for p in spec["defining_parts"] if p["relation"] == "concentric"
             and p["kind"] == "edge"]
    if not rings:
        return ""
    n = sum(p["count"] for p in rings)
    centre = spec_mod.core(spec)
    return "\n".join([
        f"- **The {n} ring walls are concentric, and that is checked.** Each one lies "
        f"wholly *inside* the one outside it -- every vertex of the inner ring is "
        f"within the outer ring's polygon -- and each is its own closed loop. Three "
        f"loops side by side is not this place.",
        f"- **Every ring carries a gate**, on its own path, and the arterial road runs "
        f"from the outermost gate to the centre crossing each ring **at its gate and "
        f"nowhere else**. Line the gates up so one road can do it.",
        f"- **{centre['name'] if centre else 'What the spec centres this place on'} "
        f"lies inside the innermost ring**, and every district lies **between** two "
        f"rings -- a district that straddles a ring wall is refused.",
        "- Give the rings room to be rings: the gap between one and the next is where "
        "a whole ring of districts goes, so a ring drawn just inside its neighbour "
        "leaves nothing to put in it.",
        f"- **The outermost ring is the place's great wall, and that is checked.** Build "
        f"it as the tallest wall type the table offers, at the top of that type's "
        f"`height` range -- a great wall is at least {GREAT_WALL_HEIGHT} high, and a "
        f"ring planned lower than that is not the wall this place is known for. The "
        f"inner rings may be lower."])


def district_brief(spec: dict, site: dict, district: dict, place: dict,
                   out_path: str, types: list | None, voice: str) -> str:
    """One district's brief: its rectangle, and everything already standing round it."""
    role = spec_mod.district_role(spec, district)
    table, decls = types_card(types, spec.get("form"), role)
    standing = "\n".join(
        f"  - **{p.get('name')}** ({p.get('kind', 'plot')}, `{p.get('type')}`): "
        + _geometry_line(p) + f"\n    {p.get('notes', '')}"
        for p in (place.get("parts") or [])) or "  (nothing yet)"
    others = [d for d in (place.get("districts") or [])
              if d["name"] != district["name"]]
    if others:
        standing += "\n" + "\n".join(
            f"  - **{d['name']}** (another district, not yours): "
            f"x {d['x0']}..{d['x1']}, z {d['z0']}..{d['z1']}" for d in others)
    for c in (place.get("compounds") or []):
        standing += (f"\n  - **{c['name']}** (a compound -- a great thing planned by a "
                     f"call of its own, not yours): x {c['x0']}..{c['x1']}, "
                     f"z {c['z0']}..{c['z1']}")
    x0, z0 = min(district["x0"], district["x1"]), min(district["z0"], district["z1"])
    x1, z1 = max(district["x0"], district["x1"]), max(district["z0"], district["z1"])
    # The planner reads the palette its buildings will be composed in.
    own = district.get("voice") or spec_mod.district_voice(spec, district) or voice
    return DISTRICT_LEVEL.format(
        sentence=spec["sentence"], intent=place.get("intent", ""),
        centre=place.get("centre", "not stated"),
        lane_material=place.get("circulation_material", "stone"),
        standing=standing, grids=site_grids(site),
        voice=styles.voice_card(own, site.get("surface_blocks"))
        + (f"\n(This district's own voice is `{own}`; the place's is `{voice}` and its "
           f"other rings may differ. Every building here is composed in `{own}`.)\n"
           if own != voice else ""),
        name=district["name"], x0=x0, x1=x1, z0=z0, z1=z1,
        w=x1 - x0 + 1, d=z1 - z0 + 1,
        purpose=district.get("purpose", ""), notes=district.get("notes", ""),
        structures=district.get("structures", 0), arterial=_arterial_note(place,
                                                                         district),
        role_note=_role_note(role),
        cover_note=_cover_note(district, role, _district_part(spec, district),
                               place, decls),
        types=table, needs=pipeline.needs_table(decls), out=out_path)


def developable_columns(district: dict, place: dict | None,
                        decls: dict | None = None) -> int:
    """How much of a district's rectangle anything can be put on: its columns less the
        arterial's band and less every standing part's rectangle with the clearance it keeps.

        The craft round, E1, found by a case. A district is asked for a cover, and the cover
        was over the whole rectangle -- so a road cut corner to corner across one, a band of
        558 columns, fragmented the grid, took a third of the rectangle out of it, and the
        district was still held to covering 46% of the whole. A floor asked of ground nobody
        may build on is a floor about somebody else's decision, and the district is the one
        that gets handed back for it.
        
    """
    x0, x1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    z0, z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    area = (x1 - x0 + 1) * (z1 - z0 + 1)
    if not place:
        return area
    from .district_compile import LOT_GAP
    gone: set = set()
    # the road's own cells, the band the compiler dilates them into, and the clearance a
    # lot has to keep off that band: ground no leaf may stand on
    k = 1 + LOT_GAP
    for c in ((place.get("arterials") or {}).get("cells") or []):
        x, z = int(c[0]), int(c[1])
        for dx in range(-k, k + 1):
            for dz in range(-k, k + 1):
                if x0 <= x + dx <= x1 and z0 <= z + dz <= z1:
                    gone.add((x + dx, z + dz))
    every = decls
    if every is None:
        try:
            _t, every = types_card()
        except Exception:                        # noqa: BLE001 -- no types on disk
            every = {}
    for p in (place.get("parts") or []):
        d = (every or {}).get(p.get("type")) or {}
        m = int((d.get("needs") or {}).get("clearance", 2)) + 1
        for (rx0, rz0, rx1, rz1) in pipeline.part_rects(p):
            for xx in range(max(x0, rx0 - m), min(x1, rx1 + m) + 1):
                for zz in range(max(z0, rz0 - m), min(z1, rz1 + m) + 1):
                    gone.add((xx, zz))
    return max(1, area - len(gone))


def district_target(district: dict, part: dict, place: dict | None = None,
                    decls: dict | None = None) -> dict:
    """What one district is asked for: a count **and** a cover, with the floors.

        The craft round: the cover is over the ground the district can **develop**
        (`developable_columns`) and not over its whole rectangle, where the caller knows the
        place well enough to say. The count is untouched: it is the layout's own ask and the
        compiler's retry is what answers it.
        
    """
    x0, x1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    z0, z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    area = (x1 - x0 + 1) * (z1 - z0 + 1)
    usable = developable_columns(district, place, decls)
    share = spec_mod.plot_share(part)
    count = int(district.get("structures") or 0)
    density = part.get("density") or "medium"
    per = spec_mod.columns_per_plot(part)
    cover = ground_cover(density)
    ground_columns = int(math.ceil(cover * usable))
    plot_columns = int(math.ceil(share * usable))
    area_columns = max(0, ground_columns - plot_columns)
    area_size = int(occupancy_shares()[density]["area_columns"])
    return {"columns": area, "usable_columns": int(usable), "density": density,
            "plot_share": share, "count": count,
            "plot_columns": plot_columns,
            "columns_per_plot": per,
            "min_count": int(math.ceil(DISTRICT_MIN_FRACTION * count)),
            "min_plot_columns": int(math.ceil(DISTRICT_MIN_FRACTION * share * usable)),
            "ground_cover": cover, "ground_columns": ground_columns,
            "min_ground_columns": int(math.ceil(DISTRICT_MIN_FRACTION * cover * usable)),
            "area_columns": area_columns, "area_size": area_size,
            "area_side": int(round(area_size ** 0.5)),
            "areas": max(0, area_columns // area_size),
            "fraction": DISTRICT_MIN_FRACTION}


def _cover_note(district: dict, role: str | None, part: dict | None = None,
                place: dict | None = None, decls: dict | None = None) -> str:
    """What a district is held to: a count and a cover, in columns."""
    x0, x1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    z0, z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    area = (x1 - x0 + 1) * (z1 - z0 + 1)
    out = ""
    if part is not None:
        t = district_target(district, part, place, decls)
        out += (
            f"**How full this district is, in two numbers, and it is handed back on "
            f"either.** It is a **{t['density']}** district, and a density is a fact "
            f"about ground: {int(t['plot_share'] * 100)}% of your rectangle is plots and "
            f"the rest is the lanes between them and the open ground of the place.\n\n"
            f"  - **The count.** {t['count']} structures, which is "
            f"{t['columns']} columns x {t['plot_share']:g} / {t['columns_per_plot']} "
            f"columns a plot. Fewer than **{t['min_count']}** is handed back.\n"
            f"  - **The cover.** Your plots together cover at least "
            f"**{t['min_plot_columns']} of your {t['columns']} columns** "
            f"({int(DISTRICT_MIN_FRACTION * t['plot_share'] * 100)}%). A count met by "
            f"twenty houses in one corner is not a district, and one courtyard house "
            f"drawn at its largest is not twelve.\n\n"
            f"Draw plots the size the type actually wants -- the table's smallest is as "
            f"real a building as its largest -- and put them on the ground evenly, off "
            f"the arterial, five apart.\n\n")
        if role != "rural":
            out += (
                f"**And the ground you do not build on is not nothing.** A quarter is "
                f"plots *and* the things between them: gardens, groves, paved widenings, "
                f"alleys cut between the rows, working yards. Every `area` type in the "
                f"table below is one of those, and an `area` leaf is a rectangle brought "
                f"to one level with no door and no rooms -- draw them the way you draw "
                f"plots, inside your rectangle, off the arterial, five apart.\n\n"
                f"  - **The ground cover.** Your plots and your areas together cover at "
                f"least **{t['min_ground_columns']} of your {t['columns']} columns** "
                f"({int(DISTRICT_MIN_FRACTION * t['ground_cover'] * 100)}%, of a target "
                f"of {int(t['ground_cover'] * 100)}%). The rest is the lanes, and they "
                f"need it.\n"
                f"  - **The size of them.** In a {t['density']} district an area is about "
                f"**{t['area_size']} columns** -- {t['area_side']} or so on a side -- so "
                f"about **{t['areas']}** of them fills the "
                f"{t['area_columns']} columns your plots leave. Fewer and larger, or more "
                f"and smaller, is yours; the cover is not.\n\n")
    if role != "rural":
        return out
    return out + (f"**This district is farmland, and farmland is covered.** The ground "
                  f"between the farmsteads is fields, not forest: draw `area` leaves of "
                  f"a field type edge to edge over it, each at most the largest plot the "
                  f"table allows and five apart for the lanes, so that the farms and the "
                  f"fields together cover at least "
                  f"**{int(RURAL_COVER * 100)}% of your rectangle -- "
                  f"{int(math.ceil(RURAL_COVER * area))} of its {area} columns**. A "
                  f"district under that is handed back with the number.")


def _role_note(role: str | None) -> str:
    """What this district is for, said in the brief and not only enforced after it. A2."""
    if not role:
        return ""
    return (f"**This is a {role} district**, and the table below is the types that "
            f"belong in one -- what is for a {role} part of a place, plus the "
            f"{' and '.join(pipeline.UNIVERSAL_ROLES)} buildings that stand anywhere. "
            f"There is no other table: a type that is not in it will be refused by "
            f"name, whatever else is right about the plot you drew for it.")


def _arterial_note(place: dict, district: dict) -> str:
    """Where the road already runs through this district, **column by column**. A4.

        Run-length encoded by row, because 909 columns written out one pair at a time is a
        brief nobody reads and a box is a brief nobody can obey.
        
    """
    art = place.get("arterials") or {}
    joins = (art.get("joins") or {}).get(district.get("name")) or []
    if not joins:
        return ""
    by_z: dict = {}
    for c in joins:
        by_z.setdefault(int(c[1]), set()).add(int(c[0]))
    rows = []
    for z in sorted(by_z):
        xs = sorted(by_z[z])
        runs, a, b = [], xs[0], xs[0]
        for x in xs[1:]:
            if x == b + 1:
                b = x
                continue
            runs.append((a, b))
            a = b = x
        runs.append((a, b))
        rows.append(f"  z={z}: " + ", ".join(f"x {p}..{q}" if p != q else f"x {p}"
                                             for (p, q) in runs))
    return ("**The arterial.** The road between the districts and the gates was routed "
            f"before this district was planned, and {len(joins)} of its columns are on "
            f"or beside your ground, {art.get('width', 4)} blocks wide. Your lanes will "
            "be routed to it. **Keep your plots off every one of these columns** -- a "
            "plot drawn on the road is a plot the rest of the place cannot walk to. "
            "Here is every one of them, by row:\n\n" + "\n".join(rows))


def _geometry_line(p: dict) -> str:
    k = p.get("kind", "plot")
    if k == "edge":
        return (f"a path of {len(p.get('path') or [])} vertices, width "
                f"{p.get('width', 1)}: "
                + " -> ".join(f"({a[0]},{a[1]})" for a in (p.get("path") or [])[:8]))
    if k == "point":
        at = p.get("at") or []
        return f"at ({at[0]},{at[-1]}) facing {p.get('facing')}"
    return f"x {p['x0']}..{p['x1']}, z {p['z0']}..{p['z1']}"


def _site_note(site: dict, plateau: dict | None = None) -> str:
    s = site.get("search") or {}
    if not s:
        return ""
    rows = ["**Why this ground.** The system scanned the world on a lattice and ranked "
            "every candidate against what this place needs. This one came first:", ""]
    for t in s.get("top", [])[:3]:
        rows.append(f"  {t['rank']}. ({t['x']}, {t['z']}) — relief {t['relief']}, "
                    f"water {t['water_pct']}%, forest {t['forest_pct']}%, "
                    f"the flattest {t['plateau']['size']}x{t['plateau']['size']} in it "
                    f"has {t['plateau']['relief']} of relief")
    if s.get("terraform"):
        rows += ["", f"The ground under **{s['terraform']['part']}** has been levelled "
                     f"to a plateau of {s['terraform']['plateau']} blocks before you "
                     f"were called: that part goes on it"
                     + (f" -- **x {plateau['rect'][0]}..{plateau['rect'][2]}, "
                        f"z {plateau['rect'][1]}..{plateau['rect'][3]}, level at "
                        f"y={plateau['y']}**, and it is checked."
                        if plateau and plateau.get("rect") else ".")]
    return "\n".join(rows)


def _compound_note(spec: dict, plateau: dict | None = None) -> str:
    """The great things of this place, each drawn as a compound and not a building.

        Silent where the spec has none. Stated rather than only refused, for the reason
        every rule in this brief is: the level is handed back at most once.
        
    """
    rows = spec_mod.compounds(spec)
    if not rows:
        return ""
    big = _monumental_columns()
    out = ["", "**3. The great things, as compounds.** A palace, a castle, a monastery "
           "is not one building: it is a place inside the place -- an inner wall with "
           "a gate, halls, courts and gardens on an axis -- and a later call plans "
           f"those parts out of the same types a district is made of, inside a "
           f"rectangle you draw. Draw each of these under `compounds` and **not** as "
           f"a leaf of `parts`; a compound is at least {COMPOUND_MIN} blocks on a side "
           f"and nothing else may overlap it:", "",
           f"  **And a great thing has to be great.** The largest single building any "
           f"district here can hold is {big['side']}x{big['side']} (`{big['type']}`), so "
           f"a compound covers at least {big['margin']:g}x that -- **{big['columns']} "
           f"columns, which is {big['side_needed']}x{big['side_needed']}** -- and is "
           f"refused below it. The ground levelled for a compound was cut to hold "
           f"exactly this; draw the compound over the whole of it.", ""]
    for p in rows:
        line = (f"  - **{p['name']}** ({p['family']}, {p['relation']}"
                + (f", {p['density']}" if p.get("density") else "") + ")"
                + (f": {p['notes']}" if p.get("notes") else ""))
        on = (plateau if plateau and plateau.get("rect")
              and plateau.get("part") == p["name"] else None)
        if on:
            r = on["rect"]
            line += (f" **It stands on the ground levelled for it: draw it inside "
                     f"x {r[0]}..{r[2]}, z {r[1]}..{r[3]}, covering at least "
                     f"{int(COMPOUND_PLATEAU_SHARE * 100)}% of that square** -- the "
                     f"whole of it is the right answer.")
        out.append(line)
    return "\n".join(out)


# -------------------------------------------------------------- the validation

def place_failures(place: dict, spec: dict, site: dict, decls: dict,
                   ground: dict | None = None, *, voice=None,
                   plateau: dict | None = None) -> list:
    """Everything wrong with a place-level plan, named. A5, and A6's other half.

        `plateau` is `stage_plateau`'s record -- `part`, `rect`, `y` -- where ground was
        levelled for a defining part, and the compound answering that part is held to it.
        
    """
    out = []
    X, Z = site["origin"]
    S = int(site["size"])
    parts = list(place.get("parts") or [])
    districts = list(place.get("districts") or [])
    compounds = list(place.get("compounds") or [])
    by_name = {p.get("name"): p for p in parts}

    def fail(name, check, why, **more):
        out.append({"part": name, "check": check, "why": why, **more})

    # 1. the spec's defining parts are all here
    for d in spec["defining_parts"]:
        if spec_mod.compound(d):
            # a rectangle a call of its own plans into a wall, gates, halls and courts
            # -- and never as one leaf. A leaf answering it is refused by name, because
            # a 12x12 palace on a plot is exactly the thing this exists to stop.
            got = [c for c in compounds if _answers(c, d)]
            if len(got) < d["count"]:
                fail(d["name"], "spec",
                     f"the spec asks for {d['count']} x {d['family']} as a compound "
                     f"({d['relation']}) -- a place inside the place -- and the plan "
                     f"draws {len(got)} under `compounds`: draw its rectangle there, "
                     f"named {d['name']}, and no leaf for it under `parts`",
                     wanted=d["count"], got=len(got))
            for p in parts:
                if _answers(p, d):
                    fail(p.get("name"), "compound",
                         f"{p.get('name')} answers {d['name']}, which is a "
                         f"{d['family']} and therefore a compound planned by a call of "
                         f"its own: draw it under `compounds` as a rectangle, not as a "
                         f"{p.get('kind', 'plot')} leaf of `parts`")
            continue
        if d["kind"] == "group":
            n = len(districts)
            if n < d["count"]:
                fail(d["name"], "spec", f"the spec asks for {d['count']} "
                                        f"{d['family']}(s) as districts and the plan "
                                        f"draws {n}")
            continue
        got = [p for p in parts if _kind_ok(p, d, decls)
               and (p.get("defines") == d["name"] or p.get("name") == d["name"]
                    or p.get("name", "").startswith(d["name"]))]
        if len(got) < d["count"]:
            fail(d["name"], "spec",
                 f"the spec asks for {d['count']} x {d['family']} as a "
                 f"{d['kind']} ({d['relation']}) and the plan has {len(got)}: name "
                 f"them {d['name']} (or {d['name']}_1, {d['name']}_2, ...) or carry "
                 f"\"defines\": \"{d['name']}\" on each",
                 wanted=d["count"], got=len(got))

    # 2. a wall the spec calls a wall is a closed loop, and its gates stand on it
    for d in spec_mod.walls(spec):
        for p in parts:
            if p.get("kind") != "edge":
                continue
            if not (p.get("defines") == d["name"] or p.get("name", "").startswith(
                    d["name"])):
                continue
            path = [(int(a[0]), int(a[1])) for a in (p.get("path") or [])]
            if len(path) < 4 or path[0] != path[-1]:
                fail(p["name"], "wall", "the spec calls this a wall, so its path is a "
                                        "closed loop: the last vertex is the first")
    gates = [p for p in parts if p.get("kind") == "point"
             and (decls.get(p.get("type")) or {}).get("passage")]
    walls = [p for p in parts if p.get("kind") == "edge"]
    for g in gates:
        at = g.get("at") or []
        cell = (int(at[0]), int(at[-1])) if len(at) >= 2 else None
        on = any(cell in set(_edge_cells(w)) for w in walls) if cell else False
        if walls and not on:
            fail(g["name"], "gate", "a gate is the one crossing of a wall and has to "
                                    "stand on it: this one is at "
                                    f"{list(at)} and no wall's path passes through it")

    # 3. the geometry
    for p in parts:
        r = pipeline.part_rect({**p, "name": p.get("name")})
        if not (X <= r[0] and r[2] < X + S and Z <= r[1] and r[3] < Z + S):
            fail(p.get("name"), "site", f"this part reaches x {r[0]}..{r[2]}, "
                                        f"z {r[1]}..{r[3]} and the site is "
                                        f"x {X}..{X + S - 1}, z {Z}..{Z + S - 1}")
    from .buildlib import Builder
    dmin = int(2 * Builder.SITE_INSET + 24)
    seen: set = set()
    for c in compounds:
        n = c.get("name")
        if n in seen:
            fail(n, "compound", "two compounds have this name")
        seen.add(n)
        x0, x1 = min(c["x0"], c["x1"]), max(c["x0"], c["x1"])
        z0, z1 = min(c["z0"], c["z1"]), max(c["z0"], c["z1"])
        if not (X <= x0 and x1 < X + S and Z <= z0 and z1 < Z + S):
            fail(n, "site", f"this compound reaches x {x0}..{x1}, z {z0}..{z1} and the "
                            f"site is x {X}..{X + S - 1}, z {Z}..{Z + S - 1}")
        if (x1 - x0 + 1) < COMPOUND_MIN or (z1 - z0 + 1) < COMPOUND_MIN:
            fail(n, "compound", f"a compound is at least {COMPOUND_MIN} blocks on a "
                                f"side -- a wall, a gate, two halls and a court with "
                                f"lanes between them -- and this one is "
                                f"{x1 - x0 + 1}x{z1 - z0 + 1}")
        # **And large enough to be monumental at all**, which is the place read's own
        # footprint margin asked at plan time, in columns, before the ground is touched.
        # a plan that cannot be the place asked for is refused before a block is placed
        # -- and the compound's scale clause was the one that was only ever read
        # afterwards.
        big = _monumental_columns()
        got = (x1 - x0 + 1) * (z1 - z0 + 1)
        if got < big["columns"]:
            fail(n, "scale",
                 f"this compound covers {got} columns and a great thing is at least "
                 f"{big['columns']} -- {big['margin']:g}x the largest single building "
                 f"the committed types admit, which is {big['side']}x{big['side']} "
                 f"({big['type']}) -- so it is {big['side_needed']} blocks on a side at "
                 f"the least; the place read holds the built compound to the same margin",
                 columns=got, needs=big["columns"])
        # **On its cut ground.** Where the site search levelled a plateau for the part
        # this compound answers, the compound stands on it: inside it, and covering most
        # of it.
        if plateau and plateau.get("rect") and plateau.get("part") \
                and (c.get("defines") == plateau["part"] or n == plateau["part"]):
            px0, pz0, px1, pz1 = plateau["rect"]
            inside = px0 <= x0 and x1 <= px1 and pz0 <= z0 and z1 <= pz1
            share = ((x1 - x0 + 1) * (z1 - z0 + 1)
                     / float((px1 - px0 + 1) * (pz1 - pz0 + 1)))
            if not inside:
                fail(n, "plateau",
                     f"the ground under {plateau['part']} was levelled at "
                     f"x {px0}..{px1}, z {pz0}..{pz1} (y={plateau.get('y')}) and this "
                     f"compound is drawn at x {x0}..{x1}, z {z0}..{z1}: a great thing "
                     f"stands on the ground cut for it, wholly inside that square",
                     plateau=list(plateau["rect"]))
            elif share < COMPOUND_PLATEAU_SHARE:
                fail(n, "plateau",
                     f"this compound covers {share:.0%} of the plateau levelled for "
                     f"{plateau['part']} (x {px0}..{px1}, z {pz0}..{pz1}) and it has "
                     f"to cover at least {COMPOUND_PLATEAU_SHARE:.0%}: the ground was "
                     f"cut for it, so draw it over the whole of it",
                     plateau=list(plateau["rect"]), share=round(share, 3))
    for d in districts:
        n = d.get("name")
        if n in seen:
            fail(n, "district", "two districts have this name")
        seen.add(n)
        x0, x1 = min(d["x0"], d["x1"]), max(d["x0"], d["x1"])
        z0, z1 = min(d["z0"], d["z1"]), max(d["z0"], d["z1"])
        if not (X <= x0 and x1 < X + S and Z <= z0 and z1 < Z + S):
            fail(n, "site", f"this district reaches x {x0}..{x1}, z {z0}..{z1} and the "
                            f"site is x {X}..{X + S - 1}, z {Z}..{Z + S - 1}")
        if (x1 - x0 + 1) < dmin or (z1 - z0 + 1) < dmin:
            fail(n, "district", f"a district is at least {dmin} blocks on a side and "
                                f"this one is {x1 - x0 + 1}x{z1 - z0 + 1}")
        want = int(d.get("structures") or 0)
        room = (x1 - x0 + 1) * (z1 - z0 + 1) * DISTRICT_FILL
        # **The density the spec gave this district, and not one number for all of
        # them.** A1: the brief quotes a per-ring figure -- 287 columns for the dense
        # ring, 818 for the farmland -- and this check quoted the flat 225. Two derived
        # numbers about the same thing, disagreeing.
        per = spec_mod.columns_per_plot(_district_part(spec, d))
        if want and want * per > room:
            fail(n, "district",
                 f"this district is asked to hold {want} structures, which needs about "
                 f"{want * per} columns of buildable ground, "
                 f"and {x1 - x0 + 1}x{z1 - z0 + 1} offers about {int(room)}: draw it "
                 f"bigger or ask it for fewer",
                 wanted=want, columns=int(room), per_structure=per,
                 defines=d.get("defines"))
    boxes = ([(p.get("name"), r) for p in parts for r in pipeline.part_rects(p)]
             + [(d.get("name"), (min(d["x0"], d["x1"]), min(d["z0"], d["z1"]),
                                 max(d["x0"], d["x1"]), max(d["z0"], d["z1"])))
                for d in districts + compounds])
    passage = {p.get("name") for p in parts
               if (decls.get(p.get("type")) or {}).get("passage")}
    for i, (an, a) in enumerate(boxes):
        for bn, b in boxes[i + 1:]:
            if an == bn or {an, bn} & passage:
                continue
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                # A district *inside* a ring wall is what a walled town is, so an edge
                # crossing a district is only a fault where the edge's own swept line
                # runs through it -- which is what `part_rects` gives and a bounding box
                # does not.
                fail(an, "overlap", f"{an} and {bn} overlap", other=bn)

    # 3b. **the rings nest, and the centre is inside the innermost.** A2, at plan time
    # rather than only in the place read. The read at the end asks five questions and
    # three of them are about what stood; these two are about geometry alone and can be
    # answered now.
    out += concentric_failures(place, spec, decls)

    # 4. the roads between the districts, where they have been routed. A4.
    if len(districts) > 1 and place.get("arterials"):
        out += arterial_failures(place, place["arterials"], decls)

    # 5.
    leaves = [{**p, "name": p.get("name"), "in": []} for p in parts]
    out += [{**f, "check": f"leaf/{f['check']}"}
            for f in pipeline.plan_failures(leaves, decls, ground=ground,
                                            form=spec.get("form"))]
    return out


def _answers(p: dict, d: dict) -> bool:
    """Does this plan entry -- a leaf, a district or a compound -- answer this defining
    part? By `defines`, by name, or by the name with a suffix, as the brief asks."""
    n = p.get("name") or ""
    return (p.get("defines") == d["name"] or n == d["name"]
            or n.startswith(d["name"] + "_"))


def compound_rects(place: dict) -> dict:
    """{compound name: (x0, z0, x1, z1)} off a place plan or an assembled plan."""
    out = {}
    for c in (place.get("compounds") or []):
        out[c["name"]] = (min(c["x0"], c["x1"]), min(c["z0"], c["z1"]),
                          max(c["x0"], c["x1"]), max(c["z0"], c["z1"]))
    return out


def concentric_failures(place: dict, spec: dict, decls: dict) -> list:
    """Everything wrong with a plan of rings, named, before anything is sited. A2.

        Three of the place read's five clauses are about geometry the plan already has --
        nesting, a gate per ring, and what is at the centre -- so they are asked here too.
        The other two need the arterial and what stood, and are the read's.
        
    """
    from .placeread import inside, rings
    parts = [{**p, "name": p.get("name"), "in": []}
             for p in (place.get("parts") or [])]
    wanted = [d for d in spec["defining_parts"] if d["relation"] == "concentric"
              and d["kind"] == "edge"]
    if not wanted:
        return []
    got = rings(spec, parts)
    out = []
    n_want = sum(d["count"] for d in wanted)
    if len(got) < n_want:
        out.append({"part": wanted[0]["name"], "check": "concentric",
                    "why": f"the spec asks for {n_want} concentric ring wall(s) and "
                           f"the plan draws {len(got)}: name them "
                           f"{wanted[0]['name']}_1, {wanted[0]['name']}_2, ... or "
                           f"carry \"defines\": \"{wanted[0]['name']}\" on each"})
        return out
    for outer, inner in zip(got, got[1:]):
        bad = [v for v in inner["path"] if not inside(outer["path"], v)]
        if bad:
            out.append({"part": inner["name"], "check": "concentric",
                        "why": f"{inner['name']} is not inside {outer['name']}: "
                               f"{len(bad)} of its vertices lie outside that ring, the "
                               f"first at {list(bad[0])}. Concentric means one wholly "
                               f"within the next", "other": outer["name"]})
    gates = [p for p in parts if p.get("kind") == "point"
             and (decls.get(p.get("type")) or {}).get("passage")]
    at = {(int((g.get("at") or [0, 0])[0]), int((g.get("at") or [0, 0])[-1]))
          for g in gates}
    for r in got:
        if not (at & set(_edge_cells(r["part"]))):
            out.append({"part": r["name"], "check": "concentric",
                        "why": "no gate stands on this ring: every ring carries one, "
                               "and the road crosses a ring at its gate or not at all"})
    innermost = got[-1] if got else None
    for d in spec["defining_parts"]:
        if d["relation"] != "centre" or innermost is None:
            continue
        rects = [(p["name"], pipeline.part_rect({**p, "name": p.get("name")}))
                 for p in parts
                 if (p.get("defines") == d["name"] or p.get("name") == d["name"]
                     or p.get("name", "").startswith(d["name"]))]
        # ...and a compound at the centre, by its rectangle: the whole of the great
        # thing lies inside the innermost ring, not only its halls.
        rects += [(n, r) for n, r in compound_rects(place).items()
                  if _answers(next(c for c in place["compounds"] if c["name"] == n), d)]
        for name, r in rects:
            corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]
            if not all(inside(innermost["path"], c) for c in corners):
                out.append({"part": name, "check": "concentric",
                            "why": f"{name} is what the spec centres this place "
                                   f"on and it is not wholly inside "
                                   f"{innermost['name']}, the innermost ring",
                            "other": innermost["name"]})
    return out


# ------------------------------------------------------------------ arterials A4.
# Between the place level and the district level there was nothing. Each district was
# handed a rectangle and asked for the plots in it; the circulation pass then routed
# lanes between every plot in the whole place at once, with no road drawn first -- so
# what joined one district to the next was whatever the router happened to lay through
# the gaps, and the gate had no road leading to it until after everything else existed.
# An **arterial** is the road between the districts and the gates, and it is planned
# before a district is. It is deterministic and costs no model call: the nodes are the
# gates and the district centres, the routing is the same one the lanes get, and the
# only thing this level knows that the district level cannot is where the other
# districts are.

#: How wide an arterial is, against a lane's two and a spine's three. A road between
#: districts carries the whole place through it.
ARTERIAL_WIDTH = 4


def arterial_nodes(place: dict, decls: dict) -> list:
    """The things an arterial joins: every gate, and the centre of every district."""
    out = []
    for p in (place.get("parts") or []):
        if p.get("kind") != "point":
            continue
        if not (decls.get(p.get("type")) or {}).get("passage"):
            continue
        at = p.get("at") or []
        if len(at) < 2:
            continue
        ax, az = int(at[0]), int(at[-1])
        out.append({"id": p["name"], "kind": "gate",
                    "x0": ax, "z0": az, "x1": ax, "z1": az})
    for d in (place.get("districts") or []):
        x0, x1 = min(d["x0"], d["x1"]), max(d["x0"], d["x1"])
        z0, z1 = min(d["z0"], d["z1"]), max(d["z0"], d["z1"])
        cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
        # The centre, as a small square rather than a cell: `_approach_candidates`
        # answers per side, and a one-cell node has one candidate and no choice about
        # which way the road comes in.
        out.append({"id": d["name"], "kind": "district",
                    "x0": cx - 1, "z0": cz - 1, "x1": cx + 1, "z1": cz + 1})
    # A compound is joined at its **edge** and never crossed: the road arrives where its
    # gate will be, and the compound's own call is told where that is. Its whole
    # rectangle is the node, so the router chooses the side, and `plan_arterials` keeps
    # the road out of the inside.
    for name, (x0, z0, x1, z1) in compound_rects(place).items():
        out.append({"id": name, "kind": "compound",
                    "x0": x0, "z0": z0, "x1": x1, "z1": z1})
    return out


def plan_arterials(place: dict, decls: dict, heights, x0: int, z0: int,
                   *, max_step: int = 3, extra_cost=None):
    """The roads between the districts and the gates. Deterministic, no model call.

        Routed by the same `plan_network` the lanes get, over the same ground, with the
        walls as obstacles and the gates as their one crossing -- so an arterial reaches the
        outside world through the gate and nowhere else, which is what a walled town means.
        
    """
    nodes = arterial_nodes(place, decls)
    if len(nodes) < 2:
        return None, nodes
    routing = circulate.parts_to_routing(
        [p for p in (place.get("parts") or []) if p.get("kind") in ("edge", "point")],
        passage={n for n, d in decls.items() if d.get("passage")})
    import numpy as np
    avoid = (np.array(extra_cost, dtype=float, copy=True) if extra_cost is not None
             else np.zeros(heights.shape, float))
    for (wx, wz) in routing["obstacles"]:
        i, j = wx - x0, wz - z0
        if 0 <= i < avoid.shape[0] and 0 <= j < avoid.shape[1]:
            avoid[i, j] = np.inf
    # A defining plot -- a keep, a palace -- is somebody's building and a road does not
    # go through it. A defining **area** is a square, and a road through a square is
    # what a square is for.
    for p in (place.get("parts") or []):
        if p.get("kind", "plot") != "plot":
            continue
        r = pipeline.part_rect({**p, "name": p.get("name")})
        avoid[max(0, r[0] - x0):max(0, r[2] - x0) + 1,
              max(0, r[1] - z0):max(0, r[3] - z0) + 1] = np.inf
    # ...and a compound is somebody's palace: the road stops at its edge.
    for r in compound_rects(place).values():
        avoid[max(0, r[0] - x0):max(0, r[2] - x0) + 1,
              max(0, r[1] - z0):max(0, r[3] - z0) + 1] = np.inf
    net = circulate.plan_network(
        heights, x0, z0, [{k: n[k] for k in ("id", "x0", "z0", "x1", "z1")}
                          for n in nodes],
        max_step=max_step, spine_width=ARTERIAL_WIDTH, lane_width=ARTERIAL_WIDTH,
        court_radius=0, avoid_extra=avoid, passable=routing["passable"])
    return net, nodes


def arterial_record(net, nodes, place: dict) -> dict:
    """What goes on the plan: the cells, and where each district joins the road.

        A district's **join** is a column of the arterial on or inside that district's
        rectangle. It is a fact about the two geometries and not a declaration, so it is
        computed here rather than asked for -- and a district with none of them is a
        district the road does not reach, which is what `place_failures` refuses.
        
    """
    cells = sorted(net.cells) if net is not None else []
    joins: dict = {}
    for d in (place.get("districts") or []):
        x0, x1 = min(d["x0"], d["x1"]), max(d["x0"], d["x1"])
        z0, z1 = min(d["z0"], d["z1"]), max(d["z0"], d["z1"])
        joins[d["name"]] = [[x, z] for (x, z) in cells
                            if x0 - 1 <= x <= x1 + 1 and z0 - 1 <= z <= z1 + 1]
    # A compound's join is where the road **arrives**: the columns of it against the
    # compound's edge, which is where the compound's own call is told to put its gate.
    for name, (x0, z0, x1, z1) in compound_rects(place).items():
        joins[name] = [[x, z] for (x, z) in cells
                       if x0 - 1 <= x <= x1 + 1 and z0 - 1 <= z <= z1 + 1]
    # **The faces too.** A rise of one block on the road is passable only onto a stair
    # facing the way you are going, and the router solved one for every rise.
    return {"cells": [[x, z] for (x, z) in cells],
            "levels": {f"{x},{z}": net.cells[(x, z)]["y"] for (x, z) in cells}
                      if net is not None else {},
            "faces": {f"{x},{z}": net.cells[(x, z)]["face"] for (x, z) in cells
                      if net.cells[(x, z)].get("face")} if net is not None else {},
            "nodes": [n["id"] for n in nodes],
            "edges": [[e["a"], e["b"]] for e in (net.edges if net is not None else [])],
            "joins": {k: v for k, v in joins.items()},
            "width": ARTERIAL_WIDTH,
            "notes": (net.notes if net is not None else {})}


def arterial_failures(place: dict, arterials: dict, decls: dict) -> list:
    """Everything wrong with the roads between the districts. A4.

        Three things, and each is a way a place stops being one place: a district the road
        does not reach, a road through somebody's building, and a road through a wall
        somewhere other than its gate.
        
    """
    out = []
    if not arterials or not arterials.get("cells"):
        return [{"part": "arterials", "check": "arterial",
                 "why": "no arterial was routed: a place of more than one district is "
                        "joined by roads planned before the districts are"}]
    cells = {(int(x), int(z)) for x, z in arterials["cells"]}
    for d in (place.get("districts") or []):
        if not arterials.get("joins", {}).get(d["name"]):
            out.append({"part": d["name"], "check": "arterial",
                        "why": "no arterial reaches this district: every district joins "
                               "the road network at one or more points, and its own "
                               "lanes are routed to that join"})
    for name, r in compound_rects(place).items():
        if not arterials.get("joins", {}).get(name):
            out.append({"part": name, "check": "arterial",
                        "why": "no arterial reaches this compound: the road arrives at "
                               "its edge, where its gate goes, or nobody arrives at all"})
        hit = [c for c in cells if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]]
        if hit:
            out.append({"part": name, "check": "arterial",
                        "why": f"an arterial runs through this compound at "
                               f"{sorted(hit)[:3]}: a road arrives at a great thing's "
                               f"gate and does not cross its ground"})
    for p in (place.get("parts") or []):
        kind = p.get("kind", "plot")
        if kind == "plot":
            r = pipeline.part_rect({**p, "name": p.get("name")})
            hit = [c for c in cells if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]]
            if hit:
                out.append({"part": p.get("name"), "check": "arterial",
                            "why": f"an arterial runs through this plot at "
                                   f"{sorted(hit)[:3]}: a road does not go through "
                                   f"somebody's building"})
        elif kind == "edge":
            passage = {(int((g.get("at") or [0, 0])[0]),
                        int((g.get("at") or [0, 0])[-1]))
                       for g in (place.get("parts") or [])
                       if g.get("kind") == "point"
                       and (decls.get(g.get("type")) or {}).get("passage")}
            near = {(px + i, pz + j) for (px, pz) in passage
                    for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)}
            hit = [c for c in cells & set(_edge_cells(p)) if c not in near]
            if hit:
                out.append({"part": p.get("name"), "check": "arterial",
                            "why": f"an arterial crosses this wall at {sorted(hit)[:3]}"
                                   f", and a wall is crossed at its gate or not at all"})
    return out


def _edge_cells(edge: dict) -> list:
    """Every column an edge's swept line covers. `Builder._edge_run` over the path."""
    half = max(1, int(edge.get("width", 1))) // 2
    path = [(int(a[0]), int(a[1])) for a in (edge.get("path") or [])]
    out = []
    for a, b in zip(path, path[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            if abs(b[0] - a[0]) != abs(b[1] - a[1]):
                continue
            sx, sz = (1 if b[0] > a[0] else -1), (1 if b[1] > a[1] else -1)
            for i in range(abs(b[0] - a[0]) + 1):
                cx, cz = a[0] + sx * i, a[1] + sz * i
                for e in range(-half, half + 1):
                    out.append((cx - sz * e, cz + sx * e))
            continue
        step = (0, 1) if a[0] == b[0] else (1, 0)
        n = abs(b[0] - a[0]) + abs(b[1] - a[1])
        sign = 1 if (b[0] + b[1]) >= (a[0] + a[1]) else -1
        for i in range(n + 1):
            cx, cz = a[0] + step[0] * i * sign, a[1] + step[1] * i * sign
            for e in range(-half, half + 1):
                out.append((cx + step[1] * e, cz + step[0] * e))
    return out


def occupancy_failures(district: dict, plots: list, part: dict,
                       role: str | None = None, place: dict | None = None,
                       decls: dict | None = None) -> list:
    """A district that came back under the count or under the cover it was asked for.

        The craft round: the cover floor is over the ground the district can develop
        (`developable_columns`), where the caller passes the place. The road across it is
        not the district's to cover.
        
    """
    t = district_target(district, part, place, decls)
    if not t["count"]:
        return []
    x0, x1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    z0, z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    houses = [p for p in plots if leaf_kind(p) == "plot"]
    cols = 0
    for p in houses:
        r = pipeline.part_rect(p)
        w = min(r[2], x1) - max(r[0], x0) + 1
        d = min(r[3], z1) - max(r[1], z0) + 1
        cols += max(0, w) * max(0, d)
    out = []
    if len(houses) < t["min_count"]:
        out.append({"part": district["name"], "type": None, "kind": "district",
                    "check": "count",
                    "why": f"this district was asked for {t['count']} structures and "
                           f"draws {len(houses)}: a {t['density']} district is "
                           f"{int(t['plot_share'] * 100)}% of its ground in plots, which "
                           f"over {t['columns']} columns at {t['columns_per_plot']} a "
                           f"plot is {t['count']}, and fewer than {t['min_count']} is "
                           f"handed back. Draw more plots, at the sizes the types "
                           f"actually admit",
                    "wanted": t["count"], "got": len(houses),
                    "floor": t["min_count"]})
    if cols < t["min_plot_columns"]:
        out.append({"part": district["name"], "type": None, "kind": "district",
                    "check": "cover",
                    "why": f"this district's plots cover {cols} of its {t['columns']} "
                           f"columns ({cols / float(t['columns']):.0%}) and a "
                           f"{t['density']} district is {int(t['plot_share'] * 100)}%: "
                           f"at least {t['min_plot_columns']} columns. Spread the plots "
                           f"over the whole rectangle rather than into one corner of it",
                    "covered": cols, "columns": t["columns"],
                    "floor": t["min_plot_columns"]})
    # **What the plots do not cover, the areas do.** A rural district has its own rule
    # for this and it is `RURAL_COVER`, registered a round earlier and unmoved; this is
    # every other district, where until now there was nothing to fill the ground with.
    if role != "rural":
        ground = cols
        for p in plots:
            if leaf_kind(p) != "area":
                continue
            r = pipeline.part_rect(p)
            w = min(r[2], x1) - max(r[0], x0) + 1
            d = min(r[3], z1) - max(r[1], z0) + 1
            ground += max(0, w) * max(0, d)
        if ground < t["min_ground_columns"]:
            out.append({"part": district["name"], "type": None, "kind": "district",
                        "check": "ground_cover",
                        "why": f"this district's plots and areas together cover {ground} "
                               f"of its {t['columns']} columns "
                               f"({ground / float(t['columns']):.0%}) and a "
                               f"{t['density']} district is "
                               f"{int(t['ground_cover'] * 100)}%: at least "
                               f"{t['min_ground_columns']} columns. The ground between "
                               f"the houses is gardens, groves, paved widenings, alleys "
                               f"and yards -- draw `area` leaves of the area types in "
                               f"your table over it, about {t['area_size']} columns each "
                               f"({t['area_side']} on a side), five apart for the lanes",
                        "covered": ground, "columns": t["columns"],
                        "floor": t["min_ground_columns"]})
    return out


def district_failures(district: dict, got: dict, place: dict, decls: dict,
                      ground: dict | None = None, *, form=None, role=None,
                      part: dict | None = None, spec: dict | None = None) -> list:
    """Everything wrong with one district's plots, named.

        `pipeline.plan_failures` over the plots, plus the two things only this level knows:
        a plot outside its own district, and a plot on top of something the place level
        already put there.
        
    """
    x0, x1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    z0, z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    plots = district_plots(got, role, spec)
    out = []
    for p in plots:
        r = pipeline.part_rect(p)
        if not (x0 <= r[0] and r[2] <= x1 and z0 <= r[1] and r[3] <= z1):
            out.append({"part": p["name"], "type": p.get("type"), "kind": "plot",
                        "check": "district",
                        "why": f"this plot is at x {r[0]}..{r[2]}, z {r[1]}..{r[3]} and "
                               f"your district is x {x0}..{x1}, z {z0}..{z1}"})
    # A4: nothing this district draws stands on the road that was routed to it.
    art = {(int(x), int(z)) for x, z in ((place.get("arterials") or {})
                                         .get("cells") or [])}
    if art:
        for p in plots:
            r = pipeline.part_rect(p)
            hit = [c for c in art if r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]]
            if hit:
                out.append({"part": p["name"], "type": p.get("type"), "kind": "plot",
                            "check": "arterial",
                            "why": f"this plot stands on the arterial at "
                                   f"{sorted(hit)[:3]}: the road through this district "
                                   f"was routed before the district was planned and is "
                                   f"how the rest of the place reaches it"})
    # Its farms and fields together cover at least `RURAL_COVER` of its rectangle, said
    # in the brief in the same columns and refused here by name. Counted as the leaves'
    # own rectangles, clipped to the district, so a field drawn over the edge is not
    # credit.
    if role == "rural" and plots:
        area = float(developable_columns(district, place, decls))
        covered = 0
        for p in plots:
            r = pipeline.part_rect(p)
            w = min(r[2], x1) - max(r[0], x0) + 1
            d = min(r[3], z1) - max(r[1], z0) + 1
            covered += max(0, w) * max(0, d)
        share = covered / area if area else 0.0
        if share < RURAL_COVER:
            out.append({"part": district["name"], "type": None, "kind": "district",
                        "check": "cover",
                        "why": f"this is a rural district and its farms and fields cover "
                               f"{share:.0%} of it ({covered} of {int(area)} columns) "
                               f"against {RURAL_COVER:.0%}: a belt of farmland is fields "
                               f"with farms in them, so draw `area` leaves of a field type "
                               f"edge to edge over the ground between the farms, five "
                               f"apart for the lanes, until at least "
                               f"{int(math.ceil(RURAL_COVER * area))} columns are drawn",
                        "covered": covered, "columns": int(area),
                        "share": round(share, 3)})
    if part is not None and plots:
        out += occupancy_failures(district, plots, part, role, place, decls)
    standing = [{**p, "name": p.get("name"), "in": []}
                for p in (place.get("parts") or [])]
    out += pipeline.plan_failures(plots + standing, decls, ground=ground, form=form)
    # A defining part is not this call's to fix, so a failure charged to one is dropped:
    # the district planner cannot move the town wall and telling it to would be asking
    # it to re-plan the level above it.
    mine = {p["name"] for p in plots} | {district["name"]}
    return [f for f in out if f["part"] in mine or f.get("other") in mine]


_KIND_CACHE: dict = {}


def leaf_kind(p: dict) -> str:
    """The kind a leaf is: **its type's**, where the type is on disk, else what it says."""
    t = p.get("type")
    if t:
        f = os.path.join(ROOT, "types", f"{t}.py")
        try:
            stamp = os.path.getmtime(f)
        except OSError:
            stamp = None
        if stamp is not None:
            key = (t, stamp)
            if key not in _KIND_CACHE:
                try:
                    _KIND_CACHE[key] = pipeline.load_type(f).get("kind", "plot")
                except Exception:            # noqa: BLE001 -- a bad type is refused later
                    _KIND_CACHE[key] = None
            if _KIND_CACHE[key]:
                return _KIND_CACHE[key]
    return p.get("kind", "plot")


def district_plots(got: dict, role: str | None = None,
                   spec: dict | None = None) -> list:
    """The leaves of one district's answer, flattened, each carrying its quarter.

        `role` is what the district is **for**, off the defining part it was drawn for, and
        it is stamped on every leaf so `pipeline.plan_failures` can put it beside the type's
        own `ROLE` without having to work out which district a leaf came from. A2.

        **The leaf's own kind, not `plot` for everything.** A district of farmland is fields
        and a field is an `area`: it is a rectangle of worked ground at one level, with no
        door and no rooms, and the type that builds one says so. This forced every leaf to
        `plot`, so a district that named `field` -- which is exactly what the agrarian ring's
        brief asks for -- had every one of its leaves refused by name for building the wrong
        kind of thing. The planner was right and the driver was not.

        A leaf still defaults to `plot`, because that is what almost every one of them is.
        
    """
    out = []
    for q in (got.get("quarters") or []):
        for p in (q.get("plots") or []):
            row = {**p, "kind": leaf_kind(p), "name": p.get("name"),
                   "quarter": q.get("name", "quarter"), "in": []}
            if role:
                row["role"] = role
            out.append(row)
    return _stamp_edges(out, spec)


# A great thing -- a palace, a castle, a monastery -- was a single type on a single
# leaf, and the largest enclosed thing this project could make was the largest pad one
# type stood clean on: 12x12. But the project builds *places* out of parts by
# construction and builds them well, and a palace is a place: an axis, an inner wall,
# gates, halls, courts. So a compound is planned the way a district is -- one call,
# given a rectangle and the types, validated leaf by leaf and handed back once -- and
# what comes out is an edge, points, plots and areas that every stage downstream already
# knows how to site, build, lint, walk and photograph.

COMPOUND_LEVEL = """# Plan one great thing of a place: a compound

## The place this is part of

> {sentence}

{intent}

The place level has already been planned. **Centre:** {centre}. **Lanes:**
{lane_material}. These parts are already in the plan and their ground is taken:

{standing}

## The site

{grids}

{voice}

## Your compound

**{name}** -- x {x0}..{x1}, z {z0}..{z1} ({w} by {d} blocks). {ground_note}

{what}

{notes}

{arrival}

## What you are deciding

**The parts inside this rectangle, and nothing else.** A great thing is a place inside
the place, not one building: what you are drawing is {made_of}, every one an instance
of a committed type from the table below, built by instantiating that type. There is
no builder in this run and no building is written by hand.

- **Everything lies inside your rectangle** and nowhere else.
{composition}
- **And it is filled.** Your halls and your areas together cover at least
  **{cover_columns} of the {inner_columns} columns inside {inside_what}** -- gardens,
  secondary courts, a paved precinct, about {area_size} columns each ({area_side} on a
  side), five apart for the lanes. A great thing is a composition and not five buildings
  on a plaza.
- **This is the greatest thing in the place, and it is measured.** The largest single
  building standing outside this compound may be up to {largest_side}x{largest_side}
  blocks, and when it is finished this compound is read against **{margin}x that by
  footprint and {margin}x it by the blocks it lays**. Your rectangle was cut to hold
  the composition at full size: draw your halls at or near the largest plot the table
  allows, not at the smallest that fits, and fill the rectangle. A handful of small
  pavilions in a large yard fails that reading and there is no second call.
- No plot overlaps another, and none overlaps the wall; leave at least five blocks
  between footprints for the lanes, and between any two parts at least the larger of
  their two **keeps clear** figures in the table, plus one -- a wall keeps four clear,
  so a hall stands five from the wall.
- **The plot you draw is not the ground the building gets.** Draw at least the smallest
  plot the type needs and at most the largest, and never a size the table says it is
  never at; the table has the library's inset already added.
- `seed` is what makes two instances of one type different buildings. No two adjacent
  plots may share both `type` and `seed`.

{role_note}

{types}

{needs}

## Output

Write a single JSON file to {out} with this shape:

{{
  "notes": "what this compound is, in a sentence or two",
  "axis": "north|south|east|west -- the side the gate is on",
  "parts": [
    {{"kind": "edge", "name": "short_slug", "type": "<an edge type above>", "seed": 1,
      "params": {{}}, "path": [[0,0],[0,0],[0,0],[0,0],[0,0]], "width": 1,
      "notes": "the compound wall"}},
    {{"kind": "point", "name": "short_slug", "type": "<a passage type above>",
      "seed": 2, "params": {{}}, "at": [0,0], "facing": "north",
      "notes": "the gate, on the wall, where the road arrives"}},
    {{"kind": "plot", "name": "short_slug", "type": "<a plot type above>", "seed": 3,
      "params": {{}}, "x0": 0, "z0": 0, "x1": 0, "z1": 0,
      "notes": "the great hall, at the far end of the axis"}},
    {{"kind": "area", "name": "short_slug", "type": "<an area type above>", "seed": 4,
      "params": {{}}, "x0": 0, "z0": 0, "x1": 0, "z1": 0,
      "notes": "the court the axis crosses"}}
  ]
}}

`kind` is **the kind the type you name builds**, and the table above says which that is
for every one of them.

Use the Write tool. Output only the file; reply with the number of parts.
"""


def compound_types(types, spec: dict, part: dict) -> tuple:
    """(the table of types a compound may name, {name: declaration}).

        The traditions of the place **and** the ones the defining part names, plus the
        universal forms; the part's own role **and** the defensive types, plus the
        universal roles -- a compound has a wall and a gate whatever it is for.
        
    """
    forms = sorted({f for f in [spec.get("form"), *(part.get("forms") or [])] if f})
    admits = compound_composition(part, spec=spec)["admits"]
    roles = sorted({r for r in [part.get("role"), *admits] if r})
    return types_card(types, forms=forms or [None], roles=roles or [None])


def _composition_bullets(t: dict) -> tuple:
    """The brief's lines for what this compound is made of -- its family's composition
    (v2, C0): (what it is drawing, the bullets, what "inside" is)."""
    walled, gated = t["walled"], t["gated"]
    halls, floor, courts = t["count"], t["min_count"], t["courts"]
    inside = "the wall" if walled else "the rectangle's margin"
    drawing = ", ".join([*(["its inner wall"] if walled else []),
                         *(["its gates"] if gated else []),
                         "its hall" if halls == 1 else "its halls",
                         "its court" if courts == 1 else "its courts"])
    lines = []
    if walled:
        lines.append(
            f"- **One closed wall of its own** -- an `edge` whose path's last vertex is "
            f"its first,\n  every segment along x or along z, a corner at every vertex "
            f"-- runs round the compound\n  **at least {COMPOUND_WALL_INSET} blocks "
            f"inside the rectangle on every side**, because a gate's\n  pad straddles "
            f"the wall and has to stay inside. Nothing stands on or outside it except\n"
            f"  its gates.")
    else:
        lines.append(
            f"- **No wall is asked of a {t['family']}.** Keep {t['margin']} blocks "
            f"inside the rectangle on every\n  side, for the clearance from what "
            f"stands outside; the ground between is the setting.")
    if gated:
        lines.append(
            f"- **A gate on that wall where the road arrives.** A gate is a `point` of a "
            f"type the\n  table marks `passage`, standing **on** a cell of the wall's "
            f"path, within\n  {COMPOUND_GATE_REACH} blocks of the arrival named above. "
            f"A second gate elsewhere is yours to\n  choose.")
    axis = (("the road comes in at the gate, crosses\n  a court, and arrives at the "
             "greatest hall at the far end. Side halls flank the axis.") if halls > 1
            else ("the road arrives at the court and the\n  one hall stands at the far "
                  "end of it, facing the way in."))
    number = ((f"That number is your rectangle's: {t['inner'][0]}x{t['inner'][1]} of "
               f"ground inside {inside},\n  at this compound's density, is {halls} "
               f"plot{'s' if halls != 1 else ''}, and fewer than\n  {floor} is handed "
               f"back.") if t.get("scales", True) else
              (f"A {t['family']} is one thing whatever the size of its setting: "
               f"{halls} plot{'s' if halls != 1 else ''}, and the\n  rest of the "
               f"{t['inner'][0]}x{t['inner'][1]} inside {inside} is the ground that "
               f"sets it."))
    lines.append(
        f"- **{halls} hall{'s' if halls != 1 else ''} inside {inside}** -- `plot` "
        f"leaves -- and **the court{'s' if courts != 1 else ''} and gardens\n  "
        f"{'between them' if halls > 1 else 'before it'}** -- `area` leaves -- on an "
        f"axis: {axis}\n  {number} At least {courts} court{'s' if courts != 1 else ''} "
        f"whatever the size.")
    return drawing, "\n".join(lines), inside


def compound_brief(spec: dict, site: dict, comp: dict, place: dict, out_path: str,
                   types: list | None, voice: str, plateau: dict | None = None) -> str:
    """One compound's brief: its rectangle, the road's arrival, and what it is."""
    part = next((d for d in spec_mod.compounds(spec) if _answers(comp, d)), None) \
        or {"family": "compound", "role": None, "notes": ""}
    table, decls = compound_types(types, spec, part)
    standing = "\n".join(
        f"  - **{p.get('name')}** ({p.get('kind', 'plot')}, `{p.get('type')}`): "
        + _geometry_line(p) + f"\n    {p.get('notes', '')}"
        for p in (place.get("parts") or [])) or "  (nothing yet)"
    for d in (place.get("districts") or []):
        standing += (f"\n  - **{d['name']}** (a district, not yours): "
                     f"x {d['x0']}..{d['x1']}, z {d['z0']}..{d['z1']}")
    x0, z0 = min(comp["x0"], comp["x1"]), min(comp["z0"], comp["z1"])
    x1, z1 = max(comp["x0"], comp["x1"]), max(comp["z0"], comp["z1"])
    on = (plateau if plateau and plateau.get("rect")
          and plateau.get("part") in (comp.get("defines"), comp.get("name")) else None)
    ground_note = (f"The ground here was levelled to y={on.get('y')} before you were "
                   f"called: it is flat, and it was cut for this." if on else
                   "The library prepares the ground under each part you draw.")
    t = compound_target(comp, spec)
    made_of, composition, inside_what = _composition_bullets(t)
    admits = compound_composition(part, spec=spec)["admits"]
    return COMPOUND_LEVEL.format(
        made_of=made_of, composition=composition, inside_what=inside_what,
        sentence=spec["sentence"], intent=place.get("intent", ""),
        centre=place.get("centre", "not stated"),
        lane_material=place.get("circulation_material", "stone"),
        standing=standing, grids=site_grids(site),
        voice=styles.voice_card(voice, site.get("surface_blocks")),
        name=comp["name"], x0=x0, x1=x1, z0=z0, z1=z1, w=x1 - x0 + 1, d=z1 - z0 + 1,
        ground_note=ground_note,
        what=(f"**What it is.** A {part.get('family')}"
              + (f": {part['notes']}" if part.get("notes") else ".")),
        notes=comp.get("notes", ""), arrival=_arrival_note(place, comp),
        inner_columns=t["columns"], cover_columns=t["min_ground_columns"],
        area_size=t["area_size"], area_side=t["area_side"],
        largest_side=_monumental_columns()["side"],
        margin=f"{_monumental_columns()['margin']:g}",
        role_note=(f"**This compound is {part.get('role') or 'civic'}**, and the table "
                   f"below is the types that belong in one -- what is for a "
                   f"{part.get('role') or 'civic'} part of a place, "
                   + (f"the {' and '.join(admits)} types a walled compound has, and "
                      if admits else "and ")
                   + f"the {' and '.join(pipeline.UNIVERSAL_ROLES)} buildings that stand "
                   f"anywhere. There is no other table."),
        types=table, needs=pipeline.needs_table(decls), out=out_path)


def _arrival_note(place: dict, comp: dict) -> str:
    """Where the road arrives at this compound, cell by cell. Its gate goes there."""
    art = place.get("arterials") or {}
    cells = (art.get("joins") or {}).get(comp.get("name")) or []
    if not cells:
        return ("**The road.** No arterial was routed to this compound; put the gate on "
                "the side that faces the centre of the place.")
    side = _arrival_side(comp, cells)
    return (f"**The road arrives on your {side} side.** The arterial between the "
            f"districts and the gates was routed before you were called, and it ends "
            f"against your rectangle at these columns: "
            + ", ".join(f"({x},{z})" for x, z in cells[:12])
            + (f" and {len(cells) - 12} more" if len(cells) > 12 else "")
            + f". **Your gate stands on your wall within {COMPOUND_GATE_REACH} blocks of "
              f"one of them**, facing out along the road, and nothing else stands on "
              f"them.")


def _arrival_side(comp: dict, cells: list) -> str:
    x0, z0 = min(comp["x0"], comp["x1"]), min(comp["z0"], comp["z1"])
    x1, z1 = max(comp["x0"], comp["x1"]), max(comp["z0"], comp["z1"])
    cx, cz = sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells)
    d = {"north": cz - z0, "south": z1 - cz, "west": cx - x0, "east": x1 - cx}
    return min(d, key=lambda k: abs(d[k]))


def place_wall_face(spec: dict | None) -> str | None:
    """**The face every wall of this place carries**, off the sentence's own words."""
    if not spec:
        return None
    part = next((p for p in spec.get("defining_parts", [])
                 if p.get("family") == "wall"), None)
    return wall_face_for(part, spec) if part else None


def _stamp_edges(rows: list, spec: dict | None) -> list:
    """Write the place's wall face onto every edge leaf that does not name one."""
    face = place_wall_face(spec)
    if not face:
        return rows
    for row in rows:
        if row.get("kind") == "edge" and not row.get("face") \
                and not (row.get("params") or {}).get("face"):
            row["face"] = face
    return rows


def compound_parts(got: dict, part: dict | None, name: str,
                   spec: dict | None = None) -> list:
    """The leaves of one compound's answer, each carrying what it is part of.

        `role` is the defining part's, so `pipeline.plan_failures` can put it beside the
        type's own; `compound` is the compound's name, which is what lets that check admit
        the defensive types a compound has whatever it is for.
        
    """
    out = []
    admits = compound_composition(part, spec=spec)["admits"] if part else None
    for p in (got.get("parts") or []):
        row = {**p, "kind": leaf_kind(p), "name": p.get("name"),
               "compound": name, "in": []}
        if part:
            row["defines"] = part["name"]
            if part.get("role"):
                row["role"] = part["role"]
            # v2, C0: what this compound admits over its own role is its family's
            # (`spec.COMPOSITIONS`), and the validator reads it off the leaf.
            row["admits"] = list(admits)
        out.append(row)
    return _stamp_edges(out, spec)


def compound_target(comp: dict, spec: dict | None = None) -> dict:
    """What one compound is asked to hold: halls, courts and gardens, from its rectangle."""
    x0, x1 = min(comp["x0"], comp["x1"]), max(comp["x0"], comp["x1"])
    z0, z1 = min(comp["z0"], comp["z1"]), max(comp["z0"], comp["z1"])
    # v2, C0: the floor, the wall and the margin are the family's.
    made_of = compound_composition(spec=spec, comp=comp)
    floor = int(made_of["halls"])
    margin = int(made_of["margin"])
    w = max(1, (x1 - x0 + 1) - 2 * margin)
    d = max(1, (z1 - z0 + 1) - 2 * margin)
    inner = {"name": comp.get("name"), "x0": 0, "z0": 0, "x1": w - 1, "z1": d - 1,
             "structures": 0}
    part = {"density": COMPOUND_DENSITY}
    inner["structures"] = spec_mod.structures_for(w * d, part)
    t = district_target(inner, part)
    scales = bool(made_of.get("scales", True))
    halls = max(floor, t["count"]) if scales else floor
    return {**t, "count": halls,
            "min_count": max(floor, t["min_count"]) if scales else floor,
            "scales": scales, "inner": [w, d], "margin": margin, "floor": floor,
            "courts": int(made_of["courts"]), "walled": bool(made_of["walled"]),
            "gated": bool(made_of["gated"]), "admits": list(made_of["admits"]),
            "family": made_of["family"], "rect": [x0, z0, x1, z1]}


def compound_failures(comp: dict, got: dict, place: dict, decls: dict,
                      ground: dict | None = None, *, spec: dict | None = None) -> list:
    """Everything wrong with one compound's plan, named.

        `pipeline.plan_failures` over its parts -- type, form, role, footprint, ground,
        overlap -- plus what only this level knows: a part outside its own rectangle, no
        closed wall of its own, no gate on it, a gate away from where the road arrives, a
        hall outside the wall, and too few parts to be a compound at all. v2, C0: which of
        those apply is the **family's composition** (`compound_composition`): a monument
        asks no wall and no gate of itself, a castle one keep inside its wall, a palace two
        halls; a wall drawn where none is asked is still held closed and inset, and a gate
        drawn is held to stand on a wall.
        
    """
    from .placeread import inside
    part = next((d for d in spec_mod.compounds(spec or {"defining_parts": []})
                 if _answers(comp, d)), None)
    name = comp["name"]
    parts = compound_parts(got, part, name)
    x0, z0 = min(comp["x0"], comp["x1"]), min(comp["z0"], comp["z1"])
    x1, z1 = max(comp["x0"], comp["x1"]), max(comp["z0"], comp["z1"])
    out = []

    def fail(p, check, why, **more):
        out.append({"part": p if isinstance(p, str) else p["name"],
                    "type": None if isinstance(p, str) else p.get("type"),
                    "kind": "compound" if isinstance(p, str) else p.get("kind", "plot"),
                    "check": check, "why": why, **more})

    for p in parts:
        for r in pipeline.part_rects(p):
            if not (x0 <= r[0] and r[2] <= x1 and z0 <= r[1] and r[3] <= z1):
                fail(p, "compound", f"this part reaches x {r[0]}..{r[2]}, z {r[1]}..{r[3]} "
                                    f"and your compound is x {x0}..{x1}, z {z0}..{z1}")
                break
    # The forms a compound admits are a list, which `plan_failures` does not take, so
    # the form check is made here and the role check there with `compound` set.
    forms = sorted({f for f in [(spec or {}).get("form"),
                                *((part or {}).get("forms") or [])] if f})
    for p in parts:
        decl = decls.get(p.get("type"))
        if decl and forms and not any(pipeline.form_ok(decl.get("form"), f)
                                      for f in forms):
            fail(p, "form", f"{p['name']}: type {p.get('type')} is a "
                            f"{decl.get('form')} and this compound is built in "
                            f"{' or '.join(forms)}, plus the "
                            f"{' and '.join(pipeline.UNIVERSAL_FORMS)} every place has",
                 allowed=[*forms, *pipeline.UNIVERSAL_FORMS])
    standing = [{**p, "name": p.get("name"), "in": []}
                for p in (place.get("parts") or [])]
    # Every type the plan names, on disk or not, and not only the table the brief
    # listed: a shop in a palace is refused for its **role**, by name, and not as a type
    # that does not exist.
    every = {**pipeline.type_declarations(parts + standing), **decls}
    out += pipeline.plan_failures(parts + standing, every, ground=ground)
    # The wall, the gate, the halls and the court: what a compound is -- by its family.
    t = compound_target(comp, spec)
    walls = [p for p in parts if p.get("kind") == "edge"]
    closed = []
    for w in walls:
        path = [(int(a[0]), int(a[1])) for a in (w.get("path") or [])]
        if len(path) >= 4 and path[0] == path[-1]:
            closed.append((w, path))
        else:
            fail(w, "wall", "a compound's wall is one closed loop: the last vertex is "
                            "the first")
        k = COMPOUND_WALL_INSET
        out_of = [v for v in path if not (x0 + k <= v[0] <= x1 - k
                                          and z0 + k <= v[1] <= z1 - k)]
        if out_of:
            fail(w, "inset", f"a compound's wall runs at least {k} blocks inside its "
                             f"rectangle on every side, so its gates' pads stay inside: "
                             f"{len(out_of)} of this wall's vertices are nearer the edge "
                             f"than that, the first at {list(out_of[0])} against "
                             f"x {x0}..{x1}, z {z0}..{z1}")
    if not closed and t["walled"]:
        fail(name, "wall", f"a {t['family']} has a closed wall of its own round it, and "
                           f"{name} draws none: an edge whose path ends where it began")
    gates = [p for p in parts if p.get("kind") == "point"
             and (decls.get(p.get("type")) or {}).get("passage")]
    on_wall = []
    for g in gates:
        at = g.get("at") or []
        cell = (int(at[0]), int(at[-1])) if len(at) >= 2 else None
        if cell and any(cell in set(_edge_cells(w)) for w, _ in closed):
            on_wall.append((g, cell))
        else:
            fail(g, "gate", f"a gate stands on the compound's wall: this one is at "
                            f"{list(at)} and no closed wall's path passes through it")
    if closed and not on_wall and t["gated"]:
        fail(name, "gate", f"no gate stands on the {t['family']}'s wall: a point of a "
                           f"passage type, on a cell of its path")
    arrive = [(int(c[0]), int(c[1])) for c in
              ((place.get("arterials") or {}).get("joins") or {}).get(name) or []]
    if arrive and on_wall and not any(
            min(max(abs(cx - ax), abs(cz - az)) for ax, az in arrive)
            <= COMPOUND_GATE_REACH for _g, (cx, cz) in on_wall):
        fail(on_wall[0][0], "gate",
             f"the road arrives at this compound at {[list(c) for c in arrive[:3]]} and "
             f"no gate stands within {COMPOUND_GATE_REACH} blocks of it: the gate is "
             f"where the place comes in")
    halls = [p for p in parts if p.get("kind", "plot") == "plot"]
    courts = [p for p in parts if p.get("kind") == "area"]
    for p in halls + courts:
        r = pipeline.part_rect(p)
        corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]
        if closed and not any(all(inside(path, c) for c in corners)
                              for _w, path in closed):
            fail(p, "inside", f"{p['name']} is not wholly inside the compound's wall: "
                              f"the halls and the courts stand within it")
    # The floor is its family's and a great thing on two hundred square is not two halls
    # and a court. The count and the cover are `district_target`'s arithmetic over the
    # ground inside the wall at `COMPOUND_DENSITY`.
    if len(halls) < t["min_count"] or len(courts) < t["courts"]:
        fail(name, "composed",
             f"a {t['family']} of {t['rect'][2] - t['rect'][0] + 1} blocks square holds "
             f"{t['count']} halls (plots) and the courts and gardens between them, and "
             f"{name} draws {len(halls)} plot(s) and {len(courts)} area(s): a great "
             f"thing is a composition, and at this size {t['min_count']} plot(s) is the "
             f"least it can be (the floor for a {t['family']} is {t['floor']} and "
             f"{t['courts']} court(s))")
    # **An axial compound is a sequence and not a set**, the craft round (E5): a family
    # whose composition declares an axis is an *approach* -- the gate, a forecourt, a
    # hall, an inner court and the greatest hall at the far end -- and a plan that holds
    # the right parts in the wrong order is five buildings on a plaza, which is what the
    # ground look saw. Refused by name so a compound a model drew is held to it too.
    if compound_composition(part, spec=spec).get("axis") and halls and courts:
        why = _axis_failure(comp, place, halls, courts)
        if why:
            fail(name, "axis", why)
    # ...and the cover is asked **where the arithmetic asks for more than the floor**.
    # the family's floor is binding on a small rectangle -- two halls and a court is all
    # a 64-square precinct holds at the sizes the types admit -- and asking that
    # rectangle for a density's share of cover as well is the same inconsistency
    # `occupancy_shares` exists to stop: two numbers about one piece of ground that
    # cannot both be met.
    covered = 0
    for p in halls + courts:
        r = pipeline.part_rect(p)
        covered += (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    if t["count"] > t["floor"] and covered < t["min_ground_columns"]:
        fail(name, "cover",
             f"this compound's halls, courts and gardens cover {covered} of the "
             f"{t['columns']} columns inside its wall "
             f"({covered / float(t['columns']):.0%}) and it has to cover "
             f"{int(t['ground_cover'] * 100)}%: at least {t['min_ground_columns']}. "
             f"Fill the ground between the halls with `area` leaves -- courts, gardens, "
             f"paved precincts -- about {t['area_size']} columns each "
             f"({t['area_side']} on a side), five apart for the lanes")
    mine = {p["name"] for p in parts} | {name}
    return [f for f in out if f["part"] in mine or f.get("other") in mine]


# ------------------------------------------------------------------ the assembly

def assemble(place: dict, districts: dict, spec: dict,
             compounds: dict | None = None) -> dict:
    """The one tree, from the levels. What `pipeline.plan_parts` reads.

        A district in the plan is a `quarter` group and the defining parts are a quarter of
        their own, so the whole place is one `district` node -- which is what `part_waves`
        turns into the build order: the walls and the gates, then the squares, then a
        quarter at a time.
        
    """
    children = []
    defining = [dict(p) for p in (place.get("parts") or [])]
    if defining:
        children.append({"kind": "quarter", "name": "defining",
                         "notes": "the parts the sentence named: what makes this place "
                                  "that place",
                         "children": defining})
    # **The assembly owns the uniqueness of a name**, because it is the only level that
    # can see every district. `gate_smithy` twice, `moot_hall` twice -- which is not a
    # mistake either of them made. A name identifies a part in `plots.json`, in the
    # circulation pass and in the readout, so a collision is qualified by the district
    # it is in and the rename goes on the record. Only the ones that collide: the
    # planner's own word for a building is what the notes and the walk ask call it.
    taken: dict = {p.get("name"): 1 for p in defining}
    for d in (place.get("districts") or []):
        for p in district_plots(districts.get(d["name"]) or {}):
            taken[p["name"]] = taken.get(p["name"], 0) + 1
    comp_specs = {c["name"]: c for c in (place.get("compounds") or [])}
    for cn in comp_specs:
        for p in compound_parts((compounds or {}).get(cn) or {}, None, cn):
            taken[p["name"]] = taken.get(p["name"], 0) + 1
    clashes = {n for n, k in taken.items() if k > 1}
    renamed: dict = {}
    # the defining part they answer -- and `compound`, so the place read, the render
    # budget and the role check can all find them, and nothing downstream has to know a
    # great thing was planned by a call of its own.
    for cn, c in comp_specs.items():
        part = next((d for d in spec_mod.compounds(spec) if _answers(c, d)), None)
        kids = []
        cvoice = c.get("voice")
        for p in compound_parts((compounds or {}).get(cn) or {}, part, cn, spec):
            p = {k: v for k, v in p.items() if k != "in"}
            if cvoice and not p.get("voice"):
                p["voice"] = cvoice
            if p.get("name") in clashes:
                was, p["name"] = p["name"], f"{cn}_{p['name']}"
                renamed[p["name"]] = {"was": was, "compound": cn}
            kids.append(p)
        if kids:
            children.append({"kind": "quarter", "name": cn, "compound": True,
                             "defines": c.get("defines") or cn,
                             "notes": c.get("notes", ""), "children": kids})
    for d in (place.get("districts") or []):
        got = districts.get(d["name"]) or {}
        # A2: and the district's role with it, for the same reason -- a leaf checked at
        # the district level against what its district is for has to carry that into the
        # assembled plan, or the whole-plan check asks a weaker question than the level
        # below it did. `district_plots` stamps it there; this is the other copy and the
        # two are the same line.
        role = spec_mod.district_role(spec, d)
        # the build resolves a part's palette from its own leaf and falls back to the
        # place's, so a district with none is built exactly as every district was
        # before.
        dvoice = d.get("voice") or spec_mod.district_voice(spec, d)
        for q in (got.get("quarters") or []):
            plots = []
            for p in (q.get("plots") or []):
                # The leaf's own kind, for `district_plots`' reason: a field is an area.
                # This forced `plot` too, so a district whose leaves were right
                # assembled into a tree whose leaves were wrong -- and the assembled
                # check refused what the district check had just passed.
                p = dict(p, kind=leaf_kind(p))
                _stamp_edges([p], spec)
                if role:
                    p["role"] = role
                if dvoice:
                    p["voice"] = dvoice
                if p.get("name") in clashes:
                    was, p["name"] = p["name"], f"{d['name']}_{p['name']}"
                    renamed[p["name"]] = {"was": was, "district": d["name"]}
                plots.append(p)
            if not plots:
                continue
            children.append({"kind": "quarter",
                             "name": f"{d['name']}_{q.get('name', 'q')}",
                             "notes": q.get("notes", d.get("purpose", "")),
                             "children": plots})
    return {"intent": place.get("intent", ""),
            "centre": place.get("centre"),
            # **The roads, on the assembled plan.** They are routed at the place level
            # and were kept only on `plan.place.json`, so every reader downstream of the
            # assembly -- the place read's two arterial clauses among them -- saw a
            # place with no roads in it and said so.
            "arterials": place.get("arterials"),
            "voice": place.get("voice"),
            "palette": place.get("palette"),
            "circulation_material": place.get("circulation_material"),
            "sentence": spec["sentence"],
            # The compounds' rectangles, on the assembled plan, for the same reason the
            # arterials are: the place read's compound clauses and the render read the
            # assembled plan and not the place level's file.
            "compounds": [dict(c) for c in (place.get("compounds") or [])],
            # The readout's `concentric` measure reads the shares, the coverage and the
            # centre off it.
            **({"layout": place["layout"]} if place.get("layout") else {}),
            "levels": {"place": len(defining),
                       "compounds": {cn: len((compounds or {}).get(cn, {})
                                             .get("parts") or [])
                                     for cn in comp_specs},
                       "districts": {d["name"]: len(district_plots(
                           districts.get(d["name"]) or {}))
                           for d in (place.get("districts") or [])}},
            "renamed": renamed,
            "parts": [{"kind": "district", "name": _slug(spec), "notes":
                       place.get("intent", "")[:200], "children": children}]}


def _slug(spec: dict) -> str:
    return spec["kind"] + "_" + str(len(spec["defining_parts"]))


def load(path: str) -> dict | None:
    return json.load(open(path)) if os.path.exists(path) else None


# A place of rings was a model's freehand drawing constrained by prose: the validator
# checked nesting, closure, a gate per ring and the centre inside, and the one sentence
# about ring spacing had no check. The demo's farm belt came out at 24% districts and
# the rest untouched forest; the palace slid ninety blocks off the site's centre and
# every ring was drawn round the wrong point; three walls where the place has more class
# boundaries than that; one palette for all of them. Every one of those is a number a
# model guessed, and the pattern this project has applied ten times is that a number a
# model guesses becomes a number the library computes. So where the spec declares rings
# (`spec.rings`: a district part with `ring`, `share`, `walled` and `voice`) the place
# level is **not a call**. The ring rectangles come from the cumulative shares, centred
# on the site's centre; a wall of the tallest admitted edge type stands at every walled
# boundary; one gate per walled ring on one axis; and every annulus is tiled into
# district rectangles that cover it. The place validator and the concentric clauses then
# run on the arithmetic's output, as the proof that it obeys them. A place with no rings
# is untouched: the freehand place planner remains for it.

#: **Registered.** How much of a ring's annulus its districts cover, at the least. The
#: assessment read 24% for the demo's belt; a ring whose districts are a fifth of it is
#: a ring of clearings in a forest. Three fifths, and the minimum width a ring is given
#: is derived from it, so a spec's shares are widened rather than a ring being refused.
RING_COVERAGE = 0.6

#: **Registered.** Each wall inside the outermost is this fraction of the height of the
#: one outside it, so the walls step down towards the centre and the great wall stays
#: the tallest thing in the place: 48 -> 36 -> 27 -> 20.
WALL_STEP = 0.75

#: A terrace step of four blocks. The outermost ring stands at the site's median ground,
#: each ring in is `TERRACE_STEP` above the one outside it, and the compound's podium a
#: further step above the innermost ring, so a city of four rings rises sixteen blocks
#: from its farm belt to its palace floor and every ring is one level. Four: a person
#: walks it as a slope over the six-wide feather every terrace edge carries (a block a
#: column), the retaining face under a wall reads as a plinth and not a cliff, and the
#: concentric run's frames -- where walls stepped 13 to 22 across a ring and the palace
#: stood level with its belt -- suggested three to six. Written before any terrace was
#: laid.
TERRACE_STEP = 4


def terrace_levels(median: int, n: int, step: int = TERRACE_STEP) -> dict:
    """The level of each of `n` rings and the podium, from the site's median ground.

        Ring 0 is the innermost. `{"rings": [level of ring 0, ...], "podium": y,
        "median": median, "step": step}`; the outermost ring's level is the median.
        
    """
    rings = [int(median) + (n - 1 - k) * int(step) for k in range(n)]
    return {"rings": rings, "podium": int(median) + n * int(step),
            "median": int(median), "step": int(step)}


#: The wall's swept half-width and one column more, so the wall stands on the terrace
#: and the retaining face is the column outside that. `stage_terraces` lays this reach
#: and `designed_heights` reads the same one.
TERRACE_REACH = 2


def designed_terrace(place: dict) -> dict | None:
    """The terrace record of a layout that designs its ground, or None: the layout's
    `terrace` where every ring carries a level."""
    lay = (place or {}).get("layout") or {}
    rings = lay.get("rings") or []
    if not rings or not lay.get("terrace") or any(r.get("level") is None for r in rings):
        return None
    return dict(lay["terrace"])


#: The ring's terrace, or the podium at a compound's gate. The climb from the ring
#: outside is made **before** the wall, over open ground, and the road runs level for
#: this many columns up to the gate -- so the threshold the circulation reserves in
#: front of the gate, and the doorstep it levels behind it, are on the flat and not on
#: the ramp. The second dry run put the climb at the wall line, where the terrace steps:
#: the doorstep's levelling then overwrote the top stair at the upper ring's gate and
#: cut away the lane at the lower ring's, and the upper ring and the palace were a piece
#: of 2,260 lane cells nobody could walk to.
GATE_APPROACH = 12
#: ...and half the width of that level approach across the road (the road is
#: `ARTERIAL_WIDTH` wide and may sit a column or two off the gate's axis).
GATE_APPROACH_HALF = 4
#: ...and past the level run the approach **ramps** down to the ground beyond, a block
#: every this many columns, so a road with a stair on every rise climbs it at a walker's
#: grade and the route planner sees a slope where the terrace steps.
GATE_RAMP_RUN = 2


def approach_points(layout: dict, gates=None) -> list:
    """Every point the ground ramps up to: the gates given, and for each ring with a
    level and no gate of its own -- an unwalled ring, whose terrace steps up all the
    same -- a **ramp** on the axis the gates stand on, at the ring's boundary, as a
    virtual point (`"virtual": True`) that nothing builds a gate at. Without it an
    unwalled ring is a terrace no road can climb at a walker's step, and the router
    raised its step to six to get in."""
    out = list(gates or [])
    rings = layout.get("rings") or []
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    side = layout.get("axis_side") or "south"
    have = {int(g["ring"]) for g in out if g.get("ring") is not None}
    for k, r in enumerate(rings):
        if r.get("level") is None or int(r.get("ring", k)) in have:
            continue
        gx, gz = _gate_at(cx, cz, int(r["outer"]), side)
        out.append({"name": f"ramp_{r['name']}", "at": [gx, gz], "ring": int(r.get("ring", k)),
                    "virtual": True})
    return out


def gate_outward(layout: dict, gate: dict) -> tuple:
    """The unit step away from the centre through `gate`, along the dominant axis."""
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    gx, gz = int(gate["at"][0]), int(gate["at"][-1])
    dx, dz = gx - cx, gz - cz
    if abs(dx) >= abs(dz):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dz > 0 else -1)


def gate_approach_pieces(h, x0: int, z0: int, layout: dict, gate: dict, *,
                         reach: int = TERRACE_REACH, approach: int = GATE_APPROACH,
                         approach_half: int = GATE_APPROACH_HALF,
                         run: int = GATE_RAMP_RUN) -> dict | None:
    """The ground outside one gate, as rectangles at levels: the level run -- the
    ground just inside the gate carried `reach + approach` columns outward,
    `approach_half` either side of the gate's axis -- and then a ramp of one block a
    `run` columns down to the ground the map `h` has at the ramp's foot. Read off
    whatever map is given: the designed one when the road is planned, the volume's
    when the terraces stage lays it; the two agree because the rings are at their
    levels on both. None for a gate off the map or with no `at`."""
    at = gate.get("at")
    if not at:
        return None
    gx, gz = int(at[0]), int(at[-1])
    ox, oz = gate_outward(layout, gate)
    n0, n1 = h.shape
    def inside_map(x, z):
        return 0 <= x - x0 < n0 and 0 <= z - z0 < n1
    ix, iz = gx - ox, gz - oz
    if not inside_map(ix, iz):
        return None
    inside = int(h[ix - x0, iz - z0])
    def strip(k0, k1, level):
        """The corridor from `k0` to `k1` columns out (inclusive), at `level`."""
        a = (gx + ox * k0, gz + oz * k0)
        b = (gx + ox * k1, gz + oz * k1)
        if ox:
            rect = (min(a[0], b[0]), gz - approach_half, max(a[0], b[0]), gz + approach_half)
        else:
            rect = (gx - approach_half, min(a[1], b[1]), gx + approach_half, max(a[1], b[1]))
        return {"rect": [int(v) for v in rect], "level": int(level),
                "from": int(k0), "to": int(k1)}
    far = reach + approach
    pieces = [strip(0, far, inside)]
    k = far + 1
    level = inside - 1
    foot = None
    while True:
        fx, fz = gx + ox * k, gz + oz * k
        if not inside_map(fx, fz):
            break
        below = int(h[fx - x0, fz - z0])
        if level <= below:
            foot = below
            break
        pieces.append(strip(k, k + run - 1, level))
        k += run
        level -= 1
    return {"gate": gate.get("name"), "at": [gx, gz], "outward": [ox, oz],
            "inside": inside, "foot": foot, "pieces": pieces,
            "registered": {"GATE_APPROACH": approach, "GATE_APPROACH_HALF": approach_half,
                           "GATE_RAMP_RUN": run, "TERRACE_REACH": reach}}


#: Outside every ring wall the ground steps down -- the retaining face at the terrace's
#: reach and the feather beyond it, a block a column or two -- and a road laid across
#: that band is a surface climbing across its own width, with rises no stair can face.
#: The router is charged this much on every column of the band (the cost a wet column
#: carries) so it crosses at a gate's level approach and nowhere else; `EDGE_BAND` is
#: how far outside the wall line the band reaches.
EDGE_BAND_COST = 40.0
EDGE_BAND = TERRACE_REACH + 2 * TERRACE_STEP


def terrace_edge_band(shape, x0: int, z0: int, layout: dict, gates=None,
                      designed=None, band: int = EDGE_BAND,
                      cost: float = EDGE_BAND_COST):
    """A cost field over `shape`: `cost` on every column from just outside a ring's
    wall line to `band` columns out, except inside a gate's approach (which is level
    ground by design); zero elsewhere. `designed` is the map the approaches are read
    off -- `designed_heights(..., approaches=False)`, the rings and the compound
    alone; with none they are read off a flat map, which places the level runs the
    same and the ramps not at all."""
    import numpy as np
    out = np.zeros(shape, float)
    rings = layout.get("rings") or []
    if not rings or any(r.get("level") is None for r in rings):
        return out
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    n0, n1 = shape
    xs = np.arange(n0) + x0 - cx
    zs = np.arange(n1) + z0 - cz
    cheb = np.maximum(np.abs(xs)[:, None], np.abs(zs)[None, :])
    for r in rings:
        o = int(r["outer"])
        out[(cheb > o) & (cheb <= o + band)] = cost
    h = designed if designed is not None else np.zeros(shape, int)
    for g in approach_points(layout, gates):
        ap = gate_approach_pieces(h, x0, z0, layout, g)
        for piece in (ap or {}).get("pieces", []):
            ax0, az0, ax1, az1 = piece["rect"]
            i0, i1 = max(0, ax0 - x0), min(n0, ax1 - x0 + 1)
            j0, j1 = max(0, az0 - z0), min(n1, az1 - z0 + 1)
            if i0 < i1 and j0 < j1:
                out[i0:i1, j0:j1] = 0.0
    return out


def designed_heights(heights, x0: int, z0: int, layout: dict, gates=None,
                     reach: int = TERRACE_REACH, approach: int = GATE_APPROACH,
                     approach_half: int = GATE_APPROACH_HALF, run: int = GATE_RAMP_RUN,
                     approaches: bool = True):
    """The ground as the terraces stage will leave it: `heights` with every ring's annulus
    at its level and the compound's rectangle at the podium, the rectangles
    `stage_terraces` lays (each ring's outer half-side plus `reach`), and -- given the
    `gates`, point parts with `at` -- every gate's approach (`gate_approach_pieces`):
    the ground inside the gate carried `reach + approach` columns outward and ramped
    down to the ground beyond a block every `run` columns, so the road climbs before the
    wall at a walker's grade and arrives level. Everything outside the outermost ring is
    the ground as found. With `approaches` False the rings and the compound alone: the
    map the approaches are read against. A copy; `heights` is untouched.
    """
    import numpy as np
    h = np.array(heights, copy=True)
    rings = layout.get("rings") or []
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    n0, n1 = h.shape
    def fill(rect, level):
        ax0, az0, ax1, az1 = rect
        i0, i1 = max(0, ax0 - x0), min(n0, ax1 - x0 + 1)
        j0, j1 = max(0, az0 - z0), min(n1, az1 - z0 + 1)
        if i0 < i1 and j0 < j1:
            h[i0:i1, j0:j1] = int(level)
    for k in sorted(range(len(rings)), key=lambda k: -k):      # outermost first
        oh = int(rings[k]["outer"]) + reach
        fill((cx - oh, cz - oh, cx + oh, cz + oh), rings[k]["level"])
    comp = layout.get("compound_rect")
    podium = (layout.get("terrace") or {}).get("podium")
    if comp and podium is not None:
        fill(tuple(int(v) for v in comp), podium)
    if not approaches:
        return h
    base = np.array(h, copy=True)                 # the approaches read the rings, not each other
    for g in approach_points(layout, gates):
        ap = gate_approach_pieces(base, x0, z0, layout, g, reach=reach, approach=approach,
                                  approach_half=approach_half, run=run)
        for piece in (ap or {}).get("pieces", []):
            fill(piece["rect"], piece["level"])
    return h


def site_median(vol, site: dict) -> int | None:
    """The median level of the site's **land** on `vol`, or None with no volume.

        Water stands at its own surface and a terrace laid at a pond's level is a terrace
        laid a block too low, so the median is of the columns that are not water.
        
    """
    if vol is None:
        return None
    import numpy as np
    from . import observe
    X, Z, S = int(site["origin"][0]), int(site["origin"][1]), int(site["size"])
    h, wet = observe.ground_heights(vol)
    i, j = X - vol.x0, Z - vol.z0
    win = h[max(0, i):i + S, max(0, j):j + S]
    damp = np.asarray(wet)[max(0, i):i + S, max(0, j):j + S]
    if win.size == 0:
        return None
    land = win[~damp] if (~damp).any() else win
    return int(np.median(land))


#: **Registered.** The most, in blocks, that the ground levelled for a concentric
#: place's centre may stand off the site's centre. The plateau is cut at the centre
#: (`stages_plan.compound_cut`), so this catches a record made by an older rule: the
#: demo's plateau stood 91.6 off.
CENTRED_TOLERANCE = 4

#: How far inside the site's edge the outermost boundary's wall line runs: the wall's
#: swept half-width plus the margin `place_failures` needs to see it inside the site.
#: **Four was a three-thick wall's**, the craft round (E4 and E7): a place now declares
#: the *mass* its wall carries and a rampart of nine reached a column outside the site,
#: which `place_failures` refuses by name. `ring_edge_inset()` is the half-width of the
#: mass the place actually declared plus this margin, so the constant is the margin and
#: the arithmetic is the wall's.
RING_EDGE_INSET = 4


def ring_edge_inset(wall_part: dict | None = None, spec: dict | None = None,
                    decl: dict | None = None) -> int:
    """How far inside the site's edge the outermost wall line runs, for the mass this
    place's wall actually carries."""
    mass = wall_width_for(wall_part, spec, decl) if wall_part is not None \
        else WALL_MASSES["screen"]
    return int(RING_EDGE_INSET + max(0, (int(mass) - 3 + 1) // 2))

#: The clear ground between two districts, and between a district and a boundary no wall
#: stands on: the five blocks the circulation pass needs between footprints, and one
#: more so two plots on either side of it never touch.
LANE_GAP = 6

#: The longest a district sector is on a side before it is cut in two. A district is one
#: planner call; the demo's longest was 283 and its planners managed, so this is about
#: the call reading its own rectangle rather than about the ground.
SECTOR_MAX = 256

#: The sides a ring's districts are cut into, in the order they are named, and the way a
#: gate faces on each -- the way a person walks through it, into the place.
_SIDES = ("north", "south", "west", "east")
_FACING = {"south": "north", "north": "south", "east": "west", "west": "east"}


def _wall_types(decls: dict) -> list:
    """The edge types a ring wall may be, tallest first, as (name, decl, lo, hi)."""
    out = []
    for n, d in decls.items():
        if d.get("kind") != "edge" or d.get("passage"):
            continue
        h = (d.get("params") or {}).get("height")
        if not h or h[0] != "int":
            continue
        out.append((n, d, int(h[1]), int(h[2])))
    return sorted(out, key=lambda r: (-r[3], r[0]))


def _wall_for(height: int, walls: list) -> tuple:
    """(type name, decl, the height it is built at): the tallest edge type that admits
    `height`, or the tallest type at its floor where nothing reaches down that far."""
    for n, d, lo, hi in walls:
        if lo <= height <= hi:
            return n, d, height
    below = [r for r in walls if r[2] > height]
    if below:
        n, d, lo, _hi = min(below, key=lambda r: r[2])
        return n, d, lo
    n, d, _lo, hi = walls[0]
    return n, d, min(height, hi)


def _wall_inset(decl: dict, width: int) -> int:
    """How far a district keeps from a wall's line: its clearance, its swept half-width
    and one, on the district's side of it."""
    return int((decl.get("needs") or pipeline.NEEDS_DEFAULT)["clearance"]) + width // 2 + 1


def _gate_type(decls: dict, family: str) -> tuple | None:
    """The passage point type a ring gate is: one named for the family, else the one
    with the widest footprint band."""
    gates = [(n, d) for n, d in decls.items()
             if d.get("kind") == "point" and d.get("passage")]
    if not gates:
        return None
    named = [g for g in gates if family and (g[0] == family or family in g[0])]
    pool = named or gates
    # Clean across its whole band first -- a type with sizes its sweep measured broken
    # (`NEEDS["except"]`) is one the wall's height may size a pad into -- then the
    # widest band, then the name.
    return sorted(pool, key=lambda g: (len(g[1]["needs"].get("except") or ()),
                                       -max(g[1]["needs"]["footprint"][2:]), g[0]))[0]


def _top_params(decl: dict) -> dict:
    """Every int parameter at the top of its range, every choice at its first: a ring
    gate is a landmark and is built at its type's full size."""
    out = {}
    for k, rule in sorted((decl.get("params") or {}).items()):
        out[k] = int(rule[2]) if rule[0] == "int" else list(rule[1])[0]
    return out


def _square_path(cx: int, cz: int, h: int, max_run: int) -> list:
    """A closed square path of half-side `h` about (cx, cz), every side cut into
    segments no longer than `max_run` columns, corners square."""
    corners = [(cx - h, cz - h), (cx + h, cz - h), (cx + h, cz + h), (cx - h, cz + h)]
    out = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        length = abs(b[0] - a[0]) + abs(b[1] - a[1])
        pieces = max(1, math.ceil(length / float(max_run - 1)))
        for i in range(pieces):
            t = i / float(pieces)
            out.append([int(round(a[0] + (b[0] - a[0]) * t)),
                        int(round(a[1] + (b[1] - a[1]) * t))])
    out.append(list(corners[0]))
    return out


#: An octagon whose diagonal sides are half the axial ones reads as round from the air
#: and costs the districts their corner pockets only. Every hand-built copy of the
#: demo's place on record draws its rings as circles; ours were squares because an
#: edge's segments ran along x or z. **A regular octagon is what a round ring should be
#: and is not what this draws**, the craft round (E4), measured and not delivered. A
#: regular octagon has its eight sides equal -- `c = 2R / (2 + sqrt(2))`, 0.5858 of the
#: half-side -- and reads as a curve where a quarter reads as a square with its corners
#: cut. What stops it is not the wall type: `great_wall` draws the diagonal run at any
#: depth and its sweep stands it. It is the **districts**, which tile a ring as
#: rectangles laid in the square annulus and know nothing about the diagonal: at 0.5858
#: the outermost ring's districts overlap their own wall and the ring falls to 57%
#: covered against `RING_COVERAGE` 0.6. The number of sides a lattice can carry is
#: eight; making them equal needs the ring's own tiling to follow the octagon, which is
#: a layout change and not a constant.
RING_CHAMFER = 0.25

#: The words in a spec's invariants or a wall part's notes that say the place's rings
#: are round, as patterns about **shape**: "round" alone is a preposition in "a wall
#: round the belt", which is what every fixture's wall says, and drew an octagon for a
#: square spec the first time this was a bare word.
WALL_ROUND_WORDS = (r"\bcircular\b", r"\bcircles?\b", r"\bcurved\b",
                    r"\bround (?:rings?|walls?|city|capital|place|plan|circuits?)\b",
                    r"\brings? (?:are|is) round\b", r"\bdrawn round\b")


def wall_round_for(part: dict, spec: dict | None = None) -> bool:
    """Do the spec's words say this place's rings are round? Never a place's name."""
    import re
    text = " ".join([str(part.get("notes") or ""),
                     str((spec or {}).get("invariants") or "")]).lower()
    return any(re.search(w, text) for w in WALL_ROUND_WORDS)


def _octagon_path(cx: int, cz: int, h: int, max_run: int,
                  chamfer: float = RING_CHAMFER) -> list:
    """A closed octagon of half-side `h` about (cx, cz): the square with each corner
    cut by `chamfer` of the half-side along both axes, so the four cut sides are
    45-degree runs; every axial side cut into segments no longer than `max_run`."""
    c = max(2, int(round(h * float(chamfer))))
    verts = [(cx - h + c, cz - h), (cx + h - c, cz - h), (cx + h, cz - h + c),
             (cx + h, cz + h - c), (cx + h - c, cz + h), (cx - h + c, cz + h),
             (cx - h, cz + h - c), (cx - h, cz - h + c)]
    out = []
    for a, b in zip(verts, verts[1:] + verts[:1]):
        if a[0] != b[0] and a[1] != b[1]:
            out.append(list(a))                       # a diagonal is one segment
            continue
        length = abs(b[0] - a[0]) + abs(b[1] - a[1])
        pieces = max(1, math.ceil(length / float(max_run - 1)))
        for i in range(pieces):
            t = i / float(pieces)
            out.append([int(round(a[0] + (b[0] - a[0]) * t)),
                        int(round(a[1] + (b[1] - a[1]) * t))])
    out.append(list(verts[0]))
    return out


def _largest_remainder(weights: list, total: int, caps: list | None = None,
                       floor: int = 1) -> list:
    """`total` spread over `weights` in proportion, each at least `floor` and at most
    its cap, the rounding going to the largest remainders. Deterministic."""
    n = len(weights)
    if n == 0:
        return []
    caps = list(caps) if caps is not None else [None] * n
    w = [max(0.0, float(v)) for v in weights]
    s = sum(w) or 1.0
    want = [total * v / s for v in w]
    got = [max(floor, int(v)) for v in want]
    for i in range(n):
        if caps[i] is not None:
            got[i] = min(got[i], max(caps[i], 0))
    order = sorted(range(n), key=lambda i: (-(want[i] - int(want[i])), i))
    short = total - sum(got)
    k = 0
    while short > 0 and any(caps[i] is None or got[i] < caps[i] for i in range(n)):
        i = order[k % n]
        if caps[i] is None or got[i] < caps[i]:
            got[i] += 1
            short -= 1
        k += 1
    k = 0
    while short < 0 and any(got[i] > floor for i in range(n)):
        i = order[-(1 + k % n)]
        if got[i] > floor:
            got[i] -= 1
            short += 1
        k += 1
    return got


def _axis_side(vol, cx: int, cz: int, gates: list, pad: int) -> str:
    """Which side of the place the gates stand on: the one whose ground under every
    gate is driest and flattest, off the volume where there is one; south otherwise
    and on a tie. `gates` is the half-side of each walled boundary."""
    if vol is None or not gates:
        return "south"
    from . import observe
    h, wet = observe.ground_heights(vol)
    best = None
    for side in ("south", "north", "east", "west"):
        score = 0.0
        for hh in gates:
            gx, gz = _gate_at(cx, cz, hh, side)
            i, j = gx - vol.x0 - pad // 2, gz - vol.z0 - pad // 2
            if i < 0 or j < 0 or i + pad > h.shape[0] or j + pad > h.shape[1]:
                score += 1e6
                continue
            win = h[i:i + pad, j:j + pad]
            score += float(win.max() - win.min()) \
                + 100.0 * float(wet[i:i + pad, j:j + pad].mean())
        if best is None or score < best[0]:
            best = (score, side)
    return best[1]


def _gate_at(cx: int, cz: int, h: int, side: str) -> tuple:
    return {"south": (cx, cz + h), "north": (cx, cz - h),
            "east": (cx + h, cz), "west": (cx - h, cz)}[side]


def wall_face_for(part: dict, spec: dict | None = None) -> str | None:
    """The face a ring wall is built with, from the words the spec used of it.

        `plain` where the spec's own description of the wall -- the wall part's notes and
        the spec's `invariants` paragraph -- says it is earthen, solid or monolithic: a
        word in the spec and never a place's name; `banded` where it says coursed masonry;
        `framed` (the grid) otherwise. See `WALL_FACE_WORDS`.
        
    """
    text = " ".join([str(part.get("notes") or ""),
                     str((spec or {}).get("invariants") or "")]).lower()
    for face, words in WALL_FACE_WORDS:
        if any(w in text for w in words):
            return face
    return "framed"


#: **Registered.** The words in a wall part's description that choose its face, in the
#: order they are tried. Earth and monolith before masonry: a rammed-earth wall with
#: courses in it is still earth.
WALL_FACE_WORDS = (("plain", ("earth", "rammed", "monolith", "solid", "unbroken",
                              "sheer", "seamless")),
                   ("banded", ("banded", "coursed", "ashlar", "masonry", "dressed")))


#: **The mass a wall carries**, the craft round (E4), registered before the numbers that
#: test it: the width in columns each word means. `great_wall` declared `width` as three
#: and only three because three is all its sweep had ever tried, so a city's outer wall
#: was forty-eight high and three thick -- a screen. What a curtain wall, a rampart and
#: a levee differ by is their mass, and it is the **place** that says which it has.
#: screen 3 -- a boundary, not a thing you walk along. curtain 5 --
#: `great_wall.CROWN_ROAD_MIN`: the crown is a road between two parapets, one on each
#: edge. rampart 9 -- an earthwork: wide enough that its ways up stand in its own
#: thickness (`great_wall.STAIR_INSIDE_SPARE`) and its face can be battered. levee 12 --
#: the greatest mass the sweep certifies, and the widest a ring wall may be before its
#: own footprint is a district.
WALL_MASSES = {"screen": 3, "curtain": 5, "rampart": 9, "levee": 12}

#: The words in a wall part's description that choose its mass, in the order they are
#: tried. The heaviest words first: a rampart described as a great wall is a rampart.
WALL_MASS_WORDS = (("levee", ("levee", "dyke", "dike", "embankment", "bund")),
                   ("rampart", ("rampart", "earthwork", "bulwark", "the largest wall",
                                "greatest wall", "mound", "berm")),
                   ("curtain", ("curtain", "battlement", "rampart walk", "wall walk",
                                "walk along", "road along")))


def wall_mass_for(part: dict, spec: dict | None = None) -> str:
    """The mass a ring wall carries, from the words the spec used of it, and `screen`
    where it used none. A word in the spec and never a place's name."""
    said = str((part or {}).get("mass") or "").strip().lower()
    if said in WALL_MASSES:
        return said
    text = " ".join([str((part or {}).get("notes") or ""),
                     str((spec or {}).get("invariants") or "")]).lower()
    for mass, words in WALL_MASS_WORDS:
        if any(w in text for w in words):
            return mass
    return "screen"


def wall_width_for(part: dict, spec: dict | None = None, decl: dict | None = None) -> int:
    """The mass in columns, held to what the type's own band admits."""
    want = WALL_MASSES[wall_mass_for(part, spec)]
    if decl:
        lo_w, _lo_d, hi_w, _hi_d = (decl.get("needs") or {}).get(
            "footprint", (1, 1, want, want))
        want = max(int(lo_w), min(int(hi_w), want))
    return int(want)


def wall_stairs_for(part: dict, spec: dict | None = None) -> str:
    """How often a ring wall carries a way down its inner face, from the spec's words."""
    text = " ".join([str(part.get("notes") or ""),
                     str((spec or {}).get("invariants") or "")]).lower()
    return "sparse" if any(w in text for w in WALL_STAIR_WORDS) else "every"


#: **Registered.** The words that say a wall's faces are one surface, both faces.
WALL_STAIR_WORDS = ("unbroken", "sheer", "seamless", "smooth")


def concentric_layout(spec: dict, site: dict, plateau: dict | None, decls: dict,
                      voice: str, vol=None) -> tuple:
    """The place level of a concentric place, by arithmetic. Returns `(place, fails)`.

        `place` is in the shape the place planner's answer takes -- `parts`, `districts`,
        `compounds`, `intent`, `centre`, `voice`, `palette`, `circulation_material` -- plus
        `layout`, the record of every number and why. `fails` is a list of named refusals
        where the spec's rings cannot be laid on this site, in `place_failures`' shape; a
        non-empty list is the round stopping, because there is nobody to hand the level
        back to and nothing here to guess.

        The arithmetic, in the order it is done:

          1. the **centre**: the site's centre, or the plateau's where one was cut for the
             centre part (and refused where that stands off the site's centre by more than
             `CENTRED_TOLERANCE`); the centre square is the compound's rectangle grown by a
             lane, or the centre's own share of the site, whichever is larger;
          2. the **boundaries**: ring k's outer half-side is `S * sqrt(cum_k) / 2` where
             `cum_k` is the centre's share plus the shares of rings 0..k, the outermost
             being the site's edge less `RING_EDGE_INSET`; every ring is at least the width
             its two insets and `RING_COVERAGE` need, and where the shares leave less the
             width is taken from the rings with slack, proportionally, and recorded;
          3. the **walls**: at every walled boundary, the tallest admitted edge type at the
             top of its range for the outermost and `WALL_STEP` of the wall outside it for
             each one in, in the face the spec's own words choose; a closed square path,
             corners square, every segment along x or z and no longer than the type's run;
          4. the **gates**: one per walled ring, on one axis through the centre, on the
             side whose ground is driest under them, facing in;
          5. the **districts**: each annulus cut into four strips inside its insets, a strip
             longer than `SECTOR_MAX` cut into sectors with a lane between, each carrying
             the ring's `defines`, density, role and voice, and the ring's structures spread
             over them by area and capped by the room each has at the ring's density;
          6. the **compound** at the centre over the whole of its plateau, or the centre
             part as one leaf where it is not a compound.
        
    """
    from .buildlib import Builder
    fails = []

    def fail(name, check, why, **more):
        fails.append({"part": name, "check": check, "why": why, **more})

    rings = spec_mod.rings(spec)
    place_voice = voice
    # **The outermost wall's own mass, before the boundaries are drawn.** The craft
    # round: a rampart of nine reaches four columns further than a screen of three and
    # the site's edge does not move.
    _wall_part = next((p for p in spec.get("defining_parts") or []
                       if p.get("family") == "wall"), None)
    edge_inset = ring_edge_inset(_wall_part, spec,
                                 (decls or {}).get("great_wall")
                                 or next((d for n, d in sorted((decls or {}).items())
                                          if d.get("kind") == "edge"), None))
    if not rings:
        fail("place", "rings", "the spec declares no rings; the freehand place planner "
                               "plans this place")
        return None, fails
    X, Z = int(site["origin"][0]), int(site["origin"][1])
    S = int(site["size"])
    core = spec_mod.core(spec)
    cx, cz = X + S // 2, Z + S // 2
    dmin = int(2 * Builder.SITE_INSET + 24)

    # 1. the centre
    comp_rect = None
    if plateau and plateau.get("rect") and core and plateau.get("part") == core["name"]:
        px0, pz0, px1, pz1 = [int(v) for v in plateau["rect"]]
        pcx, pcz = (px0 + px1) / 2.0, (pz0 + pz1) / 2.0
        off = max(abs(pcx - cx), abs(pcz - cz))
        if off > CENTRED_TOLERANCE:
            fail(core["name"], "centred",
                 f"the ground levelled for {core['name']} is centred at "
                 f"({pcx:g},{pcz:g}) and the site's centre is ({cx},{cz}): {off:g} "
                 f"off against {CENTRED_TOLERANCE}. A concentric place is drawn round "
                 f"its centre, and the plateau is cut there", offset=off)
            return None, fails
        comp_rect = (px0, pz0, px1, pz1)
        cx, cz = int(math.floor(pcx)), int(math.floor(pcz))
    centre_side = int(round(S * math.sqrt(spec_mod.centre_share(spec))))
    if comp_rect is None and core is not None:
        # No plateau: the centre square is the centre's share, at least what a compound
        # needs where the centre is one, at most the site.
        want = centre_side
        if spec_mod.compound(core):
            want = max(want, int(compound_ground(spec=spec, site_side=S)["side"]))
        want = max(COMPOUND_MIN, min(want, S - 2 * edge_inset - 2 * dmin))
        half = want // 2
        comp_rect = (cx - half, cz - half, cx - half + want - 1, cz - half + want - 1)
    comp_half_x = (comp_rect[2] - comp_rect[0]) // 2
    comp_half_z = (comp_rect[3] - comp_rect[1]) // 2
    comp_half = max(comp_half_x, comp_half_z)
    # Ring 0 begins at the compound's edge: the centre's share is what the spec left
    # over, the compound's size is the library's arithmetic, and where the share is the
    # larger the difference goes to ring 0 rather than to a band of nothing round the
    # palace. The boundaries below are still where the cumulative shares put them.
    hc = comp_half

    # 2. the boundaries
    walls = _wall_types(decls)
    if not walls and spec_mod.walled_rings(spec):
        fail("ring_walls", "type", "no committed edge type of this place's form has a "
                                   "height parameter to build a ring wall with")
        return None, fails
    h_last = S // 2 - edge_inset
    n = len(rings)
    # the wall at each walled boundary, outermost first, so heights step down inward
    heights: dict = {}
    if walls:
        outer_h = walls[0][3]
        step = 0
        for r in reversed(rings):
            if r.get("walled"):
                heights[r["name"]] = _wall_for(int(round(outer_h * WALL_STEP ** step)),
                                               walls)
                step += 1
    insets = []
    for k, r in enumerate(rings):
        inner_ring = rings[k - 1] if k else None
        if k == 0:
            inset_in = LANE_GAP
        elif inner_ring.get("walled"):
            _n, d, _h = heights[inner_ring["name"]]
            inset_in = _wall_inset(d, 3)
        else:
            inset_in = LANE_GAP // 2
        if r.get("walled"):
            _n, d, _h = heights[r["name"]]
            inset_out = _wall_inset(d, 3)
        else:
            inset_out = LANE_GAP // 2
        insets.append((inset_in, inset_out))
    mins = []
    for (i_in, i_out) in insets:
        total_i = i_in + i_out
        mins.append(max(total_i + dmin,
                        int(math.ceil(total_i / (1.0 - RING_COVERAGE))) + LANE_GAP))
    cum = spec_mod.centre_share(spec)
    targets = []
    for r in rings:
        cum += float(r["share"])
        targets.append(S * math.sqrt(min(1.0, cum)) / 2.0)
    # the outermost boundary is the outer wall's line, not the site's edge
    targets[-1] = float(h_last)
    prev = float(hc)
    widths = []
    for k, t in enumerate(targets):
        w = t - prev
        prev = max(prev, t)
        widths.append(w)
    widths = [max(w, float(m)) for w, m in zip(widths, mins)]
    room = float(h_last - hc)
    if sum(mins) > room:
        fail("rings", "shares",
             f"the rings cannot be laid on a {S}x{S} site: their least widths -- "
             + ", ".join(f"{r['name']} {m}" for r, m in zip(rings, mins))
             + f" -- sum to {sum(mins)} and the ground between the centre square "
               f"(half-side {hc}) and the outer wall (half-side {h_last}) is {int(room)}",
             widths=mins, room=int(room))
        return None, fails
    over = sum(widths) - room
    squeezed = []
    if over > 1e-9:
        slack = [w - m for w, m in zip(widths, mins)]
        total_slack = sum(slack)
        widths = [w - over * (s / total_slack) for w, s in zip(widths, slack)]
        squeezed = [r["name"] for r, s in zip(rings, slack) if s > 0]
    # integers: the boundary half-sides, the last forced to the outer wall line
    hs = []
    acc = float(hc)
    for w in widths:
        acc += w
        hs.append(int(round(acc)))
    hs[-1] = h_last
    for k in range(1, n):
        if hs[k] - hs[k - 1] < mins[k]:
            hs[k] = hs[k - 1] + mins[k]
    if hs[-1] != h_last:
        # rounding pushed the last boundary past the wall line: take it from the belt
        hs[-1] = h_last
        if n > 1 and hs[-1] - hs[-2] < mins[-1]:
            hs[-2] = hs[-1] - mins[-1]

    # a layout with no ground to read lays its walls on whatever is there, as it did.
    terrace = None
    if plateau and plateau.get("terrace"):
        terrace = dict(plateau["terrace"])
    else:
        med = site_median(vol, site)
        if med is not None:
            terrace = terrace_levels(med, n)
    levels = ([int(v) for v in terrace["rings"]] if terrace else [None] * n)

    # 3. the walls, 4. the gates
    wall_part = next((p for p in spec["defining_parts"]
                      if p["family"] == "wall" and p["relation"] == "concentric"), None)
    gate_part = next((p for p in spec["defining_parts"]
                      if p["family"] == "gate" and p["relation"] == "gateway"), None)
    walled_hs = [hs[k] for k, r in enumerate(rings) if r.get("walled")]
    gate_decl = _gate_type(decls, (gate_part or {}).get("family", "gate"))
    pad = Builder.point_pad(walls[0][3] if walls else None,
                            gate_decl[1]["needs"]["footprint"] if gate_decl else None)
    side = _axis_side(vol, cx, cz, walled_hs, pad)
    parts = []
    seed = 1
    # for a wall type that declares it draws a diagonal run, and a square for one that
    # does not, the miss named per ring.
    round_asked = bool(wall_part is not None and wall_round_for(wall_part, spec))
    round_rings: dict = {}
    chamfers: dict = {}
    for k, r in enumerate(rings):
        if not r.get("walled"):
            continue
        tname, tdecl, height = heights[r["name"]]
        max_run = int(tdecl["needs"]["footprint"][3])
        octagon = round_asked and bool(tdecl.get("diagonal"))
        if round_asked:
            round_rings[r["name"]] = ("octagon" if octagon else
                                      f"square: {tname} draws no diagonal run")
        if octagon:
            chamfers[k] = max(2, int(round(hs[k] * RING_CHAMFER)))
        # **The mass the place declared**, the craft round (E4). The part's `width` is
        # the swept line siting hands the type and is what the wall actually is; the
        # `width` *parameter* stays at the one value the sweep runs every combination
        # at.
        mass_word = wall_mass_for(wall_part, spec) if wall_part is not None else "screen"
        mass = (wall_width_for(wall_part, spec, tdecl) if wall_part is not None
                else WALL_MASSES["screen"])
        params = {"height": int(height)}
        if "width" in (tdecl.get("params") or {}):
            params["width"] = int((tdecl["params"]["width"] or ("int", 3, 3))[1])
        face_word = None
        if wall_part is not None:
            face_word = wall_face_for(wall_part, spec)
            if "face" in (tdecl.get("params") or {}):
                choices = list(tdecl["params"]["face"][1])
                # **A wall the spec calls unbroken is dressed masonry**, the craft round
                # (E4): `plain` is forty-eight blocks of one flat plane and reads as a
                # render, and `masonry` carries the ways up at the corners and the gates
                # as `unbroken` did. `unbroken` was the word for the two together and
                # says nothing this does not.
                params["face"] = ("masonry" if face_word == "plain"
                                  and wall_stairs_for(wall_part, spec) == "sparse"
                                  and "masonry" in choices else face_word)
        wname = f"{(wall_part or {}).get('name', 'ring_wall')}_{r['name']}"
        parts.append({"kind": "edge", "name": wname, "type": tname, "seed": seed,
                      "params": params,
                      "path": (_octagon_path(cx, cz, hs[k], max_run) if octagon
                               else _square_path(cx, cz, hs[k], max_run)),
                      **({"shape": "octagon"} if octagon else {}),
                      "width": mass, "mass": mass_word,
                      # **A ring's wall is in its ring's palette**, the craft round: a
                      # place whose rings differ in colour differs in colour at its
                      # walls too, and a wall left in the place's own voice is the one
                      # thing in a ring that is not the ring's.
                      "voice": r.get("voice") or place_voice,
                      "defines": (wall_part or {}).get("name"),
                      "ring": int(r["ring"]),
                      # the ring's terrace level: the wall stands on the terrace edge at
                      # one level, and `_site_edge` reads it
                      **({"level": levels[k]} if levels[k] is not None else {}),
                      # ...and the face the spec's words chose, for a type that reads it
                      # off the part rather than a parameter (`wall`), and with it how
                      # often that face carries a way down: the craft round (E6) gives
                      # `wall` the `stairs` word `great_wall` has had, so a low ring
                      # wall of an unbroken place stops zigzagging.
                      **({"face": face_word} if face_word else {}),
                      **({"stairs": wall_stairs_for(wall_part, spec)}
                         if wall_part is not None else {}),
                      "notes": f"the wall on the outside of ring {r['ring']} "
                               f"({r['name']}), {height} high and {mass} thick as "
                               f"`{tname}` -- a {mass_word} -- laid by "
                               f"arithmetic at half-side {hs[k]} about the centre"
                               + (f", on the ring's terrace at y={levels[k]}"
                                  if levels[k] is not None else "")})
        seed += 1
        if gate_decl is None:
            fail(wname, "gate", "no committed passage point type of this place's form "
                                "builds a gate for this wall")
            continue
        gx, gz = _gate_at(cx, cz, hs[k], side)
        gname = f"{(gate_part or {}).get('name', 'ring_gate')}_{r['name']}"
        parts.append({"kind": "point", "name": gname, "type": gate_decl[0], "seed": seed,
                      "params": _top_params(gate_decl[1]), "at": [gx, gz],
                      "facing": _FACING[side], "defines": (gate_part or {}).get("name"),
                      "ring": int(r["ring"]),
                      "notes": f"the one way through {wname}, on the {side} side, on "
                               f"the axis every ring's gate stands on"})
        seed += 1

    # 5. the districts
    districts = []
    layout_rings = []
    left_over = []
    for k, r in enumerate(rings):
        a = comp_half if k == 0 else hs[k - 1]
        i_in, i_out = insets[k]
        lo = a + i_in + 1
        hi = hs[k] - i_out
        g = LANE_GAP
        # under an octagonal wall the strips stop short of the chamfered corners
        cc = chamfers.get(k, 0)
        strips = {
            "north": (cx - hi + cc, cz - hi, cx + hi - cc, cz - lo),
            "south": (cx - hi + cc, cz + lo, cx + hi - cc, cz + hi),
            "west": (cx - hi, cz - lo + g, cx - lo, cz + lo - g),
            "east": (cx + lo, cz - lo + g, cx + hi, cz + lo - g),
        }
        sectors = []
        for sname in _SIDES:
            x0, z0, x1, z1 = strips[sname]
            if x1 - x0 + 1 < dmin or z1 - z0 + 1 < dmin:
                continue
            along_x = (x1 - x0) >= (z1 - z0)
            length = (x1 - x0 + 1) if along_x else (z1 - z0 + 1)
            pieces = max(1, int(math.ceil(length / float(SECTOR_MAX))))
            while pieces > 1 and (length - g * (pieces - 1)) // pieces < dmin:
                pieces -= 1
            plen = (length - g * (pieces - 1)) // pieces
            for i in range(pieces):
                start = (x0 if along_x else z0) + i * (plen + g)
                end = start + plen - 1 if i < pieces - 1 else (x1 if along_x else z1)
                rect = ((start, z0, end, z1) if along_x else (x0, start, x1, end))
                if pieces == 1:
                    label = sname
                elif pieces == 2:
                    label = sname + ("_west" if along_x else "_north") if i == 0 \
                        else sname + ("_east" if along_x else "_south")
                else:
                    label = f"{sname}_{i + 1}"
                sectors.append((label, rect))
        # `area x plot_share / columns_per_plot` -- capped by the room `DISTRICT_FILL`
        # leaves, and never nothing. What the spec's rings declared is kept beside it as
        # a declaration the readout reports against and nothing reads.
        per = spec_mod.columns_per_plot(r)
        areas = [(rc[2] - rc[0] + 1) * (rc[3] - rc[1] + 1) for _l, rc in sectors]
        shapes = [(rc[2] - rc[0] + 1, rc[3] - rc[1] + 1) for _l, rc in sectors]
        caps = [int(ar * DISTRICT_FILL // per) for ar in areas]
        counts = [min(cap, spec_mod.structures_for(ar, r, sh))
                  for ar, cap, sh in zip(areas, caps, shapes)] if sectors else []
        # **A sector the fabric fits no house in is asked for none**, the craft round,
        # found by running it: the count used to be floored at one, so a sliver of 4,257
        # columns in the corner of a ring was asked for a house its own shape cannot
        # hold, drew none and stopped the run. It stays a district -- its ground is the
        # ring's and the coverage clause counts it -- and what it holds is the open
        # ground a district asked for nothing lays.
        left_over.extend({"ring": r["name"], "label": lab, "rect": list(rc),
                          "why": "its own shape fits no house of this density, so it is "
                                 "asked for none and lays open ground"}
                         for (lab, rc), n in zip(sectors, counts) if n < 1)
        declared = int(r.get("structures") or 0)
        capped = ({"asked": declared, "room": sum(caps)}
                  if declared > sum(caps) else None)
        rvoice = r.get("voice") or place_voice
        role = r.get("role") or spec_mod.read_role(None, r, r["name"])
        for (label, rc), cnt in zip(sectors, counts):
            districts.append({
                "name": f"{r['name']}_{label}", "x0": rc[0], "z0": rc[1],
                "x1": rc[2], "z1": rc[3], "structures": int(cnt),
                "defines": r["name"], "ring": int(r["ring"]), "voice": rvoice,
                "purpose": (f"{r.get('notes') or r['name']} Ring {r['ring']} of "
                            f"{n}, counting from the centre: a {r.get('density') or 'medium'}"
                            f", {role} district"
                            + (f", built in the voice `{rvoice}`" if rvoice else "")
                            + "."),
                "notes": (f"The {label.replace('_', ' ')} sector of ring {r['ring']}, "
                          f"between the boundary at half-side {a} and the one at "
                          f"{hs[k]} from the centre"
                          + (f"; a wall stands on this ring's outside"
                             if r.get("walled") else "; no wall on this ring's outside")
                          + (f"; a wall stands on its inside"
                             if k and rings[k - 1].get("walled") else "")
                          + ". Laid out by arithmetic, not drawn."),
            })
        # the ground the ring encloses: the square less the corner triangles an
        # octagonal wall cuts off, on its own boundary and on the ring's inside
        c_out = chamfers.get(k, 0)
        c_in = chamfers.get(k - 1, 0) if k else 0
        annulus = ((2 * hs[k] + 1) ** 2 - 2 * c_out * c_out) - ((2 * a + 1) ** 2 - 2 * c_in * c_in)
        covered = sum(areas)
        cov = covered / float(annulus) if annulus else 0.0
        got_share = annulus / float(S * S)
        if cov < RING_COVERAGE:
            fail(r["name"], "coverage",
                 f"ring {r['ring']} ({r['name']}) is covered {cov:.0%} by its districts "
                 f"against {RING_COVERAGE:.0%}: {covered} of {annulus} columns",
                 coverage=round(cov, 3))
        layout_rings.append({
            "name": r["name"], "ring": int(r["ring"]), "walled": bool(r.get("walled")),
            "share_asked": float(r["share"]), "share_got": round(got_share, 4),
            "inner": int(a), "outer": int(hs[k]), "width": int(hs[k] - a),
            "min_width": int(mins[k]), "insets": [int(i_in), int(i_out)],
            "wall": (heights[r["name"]][0] if r.get("walled") else None),
            "wall_height": (heights[r["name"]][2] if r.get("walled") else None),
            "districts": [d["name"] for d in districts if d["defines"] == r["name"]],
            "structures": {"asked": int(sum(counts)), "declared": declared,
                           "laid": int(sum(counts)), "capped": capped,
                           "from_ground": {"plot_share": spec_mod.plot_share(r),
                                           "columns_per_plot": per,
                                           "district_columns": int(sum(areas))}},
            "annulus_columns": int(annulus), "district_columns": int(covered),
            "coverage": round(cov, 4), "voice": rvoice, "level": levels[k]})

    # 5b. **the ceiling, over the whole place.** The counts above are derived from each
    # sector's own ground and nothing in them knows how much ground there is in total,
    # so the one place-wide number a place is held to is applied here: every district's
    # count scaled by the same factor, floored at one, and recorded.
    # `spec.structures_ceiling` scales with the footprint ceiling by area.
    cap_total = spec_mod.structures_ceiling(spec.get("kind"))
    asked_total = sum(int(d["structures"]) for d in districts)
    ceiling_rec = {"ceiling": cap_total, "asked": asked_total, "applied": False}
    if asked_total > cap_total and asked_total:
        factor = cap_total / float(asked_total)
        for d in districts:
            d["structures"] = max(1, int(int(d["structures"]) * factor))
        for lr in layout_rings:
            lr["structures"]["laid"] = sum(int(d["structures"]) for d in districts
                                           if d["defines"] == lr["name"])
        ceiling_rec.update({"applied": True, "factor": round(factor, 4),
                            "got": sum(int(d["structures"]) for d in districts),
                            "why": "the ground asked for more than the place's ceiling; "
                                   "every district was scaled by the same factor and "
                                   "none went to nothing"})

    # 6. the centre
    compounds = []
    # **The centre is inside the innermost ring and carries its palette**, the craft
    # round, found by running it: a compound part may not name a voice -- a voice is a
    # ring's -- so a palace stood in the place's own voice while every ring round it was
    # in its class's, and the brightest end of the axis had nothing at its centre.
    centre_voice = next((r.get("voice") or place_voice
                         for r in rings if r.get("ring") == 0), place_voice)
    if core is not None and spec_mod.compound(core):
        compounds.append({
            "name": core["name"], "defines": core["name"], "voice": centre_voice,
            "x0": comp_rect[0], "z0": comp_rect[1], "x1": comp_rect[2], "z1": comp_rect[3],
            "notes": (f"The {core['family']} at the centre of the place, the fixed point "
                      f"every ring is set out around, over the whole of the ground "
                      f"levelled for it. "
                      + (core.get("notes") or "")
                      + f" The road arrives on its {side} side, on the axis every "
                        f"ring's gate stands on; its gate goes there and its greatest "
                        f"hall at the far end of that axis.")})
    elif core is not None:
        leaf = _centre_leaf(core, decls, comp_rect, seed)
        if leaf is None:
            fail(core["name"], "type", f"no committed type of this place's form builds "
                                       f"{core['name']} ({core['family']}) at the centre")
        else:
            parts.append(leaf)
            seed += 1

    pal = None
    try:
        pal = dict(styles.VOICES[voice]["palette"]) if voice in styles.VOICES else None
    except Exception:                        # noqa: BLE001 -- no palette, no material
        pal = None
    place = {
        "intent": (f"A {spec['kind']} of {n} rings laid out by arithmetic around "
                   f"{core['name'] if core else 'its centre'}: "
                   + "; ".join(f"ring {r['ring']} {r['name']}, {r.get('density') or 'medium'}"
                               f"{', walled' if r.get('walled') else ''}"
                               f"{', in ' + r['voice'] if r.get('voice') else ''}"
                               for r in rings)
                   + f". Every ring is a band of districts round the one before; the "
                     f"{len(walled_hs)} wall(s) stand where the spec says a ring is "
                     f"walled, the outermost of them the place's great wall; the gates "
                     f"stand on the {side} side on one axis and the road runs up it to "
                     f"the centre."
                   + (f" {spec['invariants']}" if spec.get("invariants") else "")),
        "centre": core["name"] if core else None,
        "voice": voice, "palette": pal,
        "circulation_material": (pal or {}).get("floor") or "stone",
        "parts": parts, "districts": districts, "compounds": compounds,
        "layout": {"by": "placeplan.concentric_layout", "site_centre": [X + S // 2, Z + S // 2],
                   "centre": [cx, cz], "centre_square_half": int(hc),
                   "centre_share_got": round(((2 * hc + 1) ** 2) / float(S * S), 4),
                   "compound_rect": list(comp_rect) if comp_rect else None,
                   "centre_share": round(spec_mod.centre_share(spec), 4),
                   "outer_half": int(h_last), "axis_side": side, "gate_pad": int(pad),
                   "squeezed": squeezed, "rings": layout_rings,
                   "structures": {**ceiling_rec, "laid": sum(int(d["structures"])
                                                             for d in districts),
                                  "declared": int(spec.get("structures") or 0),
                                  "size_band": list(spec.get("size_band") or []),
                                  "dense_plot": dense_plot(),
                                  "shares": occupancy_shares()},
                   "terrace": terrace,
                   "round": ({"asked": True, "rings": round_rings,
                              "chamfer": RING_CHAMFER} if round_asked else None),
                   "registered": {"PLOT_CELLS": dict(spec_mod.PLOT_CELLS),
                                  "AREA_PLOTS": dict(AREA_PLOTS),
                                  "PLOT_LANE": PLOT_LANE,
                                  "DISTRICT_FILL": DISTRICT_FILL,
                                  "RING_COVERAGE": RING_COVERAGE, "WALL_STEP": WALL_STEP,
                                  "CENTRED_TOLERANCE": CENTRED_TOLERANCE,
                                  "TERRACE_STEP": TERRACE_STEP,
                                  "RING_CHAMFER": RING_CHAMFER,
                                  "RING_EDGE_INSET": RING_EDGE_INSET,
                                  "ring_edge_inset": edge_inset,
                                  "left_over": left_over,
                                  "LANE_GAP": LANE_GAP, "SECTOR_MAX": SECTOR_MAX}},
    }
    return place, fails


def _centre_leaf(core: dict, decls: dict, rect: tuple, seed: int) -> dict | None:
    """The centre part as one leaf where it is not a compound: a type named for its
    family, at its largest footprint inside the centre square, centred."""
    from .buildlib import Builder
    cands = [(n, d) for n, d in decls.items()
             if n == core["family"] or n.startswith(core["family"] + "_")]
    if not cands:
        return None
    name, decl = sorted(cands)[0]
    kind = decl.get("kind", "plot")
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 0 if kind in ("edge", "point", "area") else 2 * Builder.SITE_INSET
    w = min(hi_w + i, rect[2] - rect[0] + 1)
    d = min(hi_d + i, rect[3] - rect[1] + 1)
    cx = (rect[0] + rect[2]) // 2
    cz = (rect[1] + rect[3]) // 2
    x0, z0 = cx - w // 2, cz - d // 2
    if kind == "point":
        return {"kind": "point", "name": core["name"], "type": name, "seed": seed,
                "params": {}, "at": [cx, cz], "facing": "north",
                "defines": core["name"], "notes": "the thing at the centre"}
    return {"kind": kind, "name": core["name"], "type": name, "seed": seed, "params": {},
            "x0": x0, "z0": z0, "x1": x0 + w - 1, "z1": z0 + d - 1,
            "defines": core["name"], "notes": "the thing at the centre, at its largest"}


# ------------------------------------------------ an axial compound, laid by the
# library

#: **How deep the forecourt and the inner court are**, as a share of the axial run each.
#: The craft round, E5: an approach is a sequence and the courts are the pauses in it.
#: The greatest hall takes what is left after them and the flanking ranges' band.
COMPOUND_COURT_SHARE = 0.22

#: How much of the axial run the greatest hall takes, at the least: it is the end of the
#: sequence and the thing the whole approach is for, so it is never a band like the
#: rest.
COMPOUND_GREAT_SHARE = 0.28


def compound_axial(comp: dict, part: dict | None, place: dict, decls: dict,
                   spec: dict | None = None, seed: int = 1) -> tuple:
    """**A compound of a family that declares an axis, laid as a sequence.** The craft
        round, E5, and the same seam the district compiler writes at.

        A composition says what a great thing *holds* -- so many halls, so many courts -- and
        said nothing about the order, so a palace precinct a quarter of a city wide came out
        as halls and courts arranged to fit and read as five buildings on a plaza. An
        approach is an order: the gate, then a forecourt, then a hall, then an inner court,
        then **the greatest hall at the far end**, with the flanking ranges paired down the
        axis either side of the inner court. The greatest hall is the largest plot inside the
        wall, which is what makes it the thing the sequence is for.

        Returns `(got, record)` in the shape a model's answer has, or `(None, why)` where the
        rectangle is too small to lay a sequence in and the compound is asked for as before.
        
    """
    import random
    made_of = compound_composition(part, spec=spec)
    if not made_of.get("axis"):
        return None, "this family's composition declares no axis"
    x0, z0 = min(comp["x0"], comp["x1"]), min(comp["z0"], comp["z1"])
    x1, z1 = max(comp["x0"], comp["x1"]), max(comp["z0"], comp["z1"])
    t = compound_target(comp, spec)
    margin = int(t["margin"])
    # the side the road arrives on is where the gate is, and the axis runs from it to
    # the far side: `_arrival_side` is the same answer the brief gives a model
    cells = ((place.get("arterials") or {}).get("joins") or {}).get(comp.get("name")) or []
    side = _arrival_side(comp, cells) if cells else "north"
    # (u, v): u runs along the axis away from the gate, v across it
    horiz = side in ("west", "east")
    U = (x1 - x0 + 1) if horiz else (z1 - z0 + 1)
    V = (z1 - z0 + 1) if horiz else (x1 - x0 + 1)
    flip = side in ("east", "south")

    def rect(u_a, u_b, v_a, v_b):
        """(x0, z0, x1, z1) of a (u, v) box, u measured from the gate."""
        if flip:
            u_a, u_b = U - 1 - u_b, U - 1 - u_a
        if horiz:
            return (x0 + u_a, z0 + v_a, x0 + u_b, z0 + v_b)
        return (x0 + v_a, z0 + u_a, x0 + v_b, z0 + u_b)

    # **inside the wall and clear of it**: the composition's own margin, which is the
    # wall's inset and the clearance it keeps, and no more -- a margin of thirteen on a
    # sixty-four-square precinct leaves 1,444 columns of 2,304 and no composition can
    # cover that.
    if made_of["walled"]:
        clear = int(((decls.get(_axial_type(decls, "edge", part, spec)) or {})
                     .get("needs") or {}).get("clearance", 4))
        margin = max(margin, COMPOUND_WALL_INSET + 2 + clear)
    inner_u = (margin, U - 1 - margin)
    inner_v = (margin, V - 1 - margin)
    run = inner_u[1] - inner_u[0] + 1
    span = inner_v[1] - inner_v[0] + 1
    if run < 3 * COMPOUND_MIN // 2 or span < COMPOUND_MIN:
        return None, (f"a {U}x{V} compound leaves {run}x{span} inside its wall, which is "
                      f"too little to lay a sequence in")

    plot_t = _axial_type(decls, "plot", part, spec)
    area_t = _axial_type(decls, "area", part, spec)
    if not plot_t or not area_t:
        return None, "this compound admits no plot type or no area type"
    gap = 3                                  # a plot keeps two clear and an area one

    # **The sequence.** Two pauses and an end: the forecourt inside the gate, the inner
    # court, and the greatest hall at the far side. The flanking ranges are what the
    # courts are flanked *by* -- the strips either side of each court band, paired
    # across the axis and divided along it until the composition's halls are all placed.
    fore = max(5, int(round(run * COMPOUND_COURT_SHARE)))
    inner_c = max(5, int(round(run * COMPOUND_COURT_SHARE)))
    great = max(COMPOUND_MIN // 2, int(round(run * COMPOUND_GREAT_SHARE)))
    gap = 3                                  # a plot keeps two clear and an area one
    spare = run - fore - inner_c - great - 2 * gap
    if spare > 0:                            # what is left deepens the two courts
        fore += spare // 2
        inner_c += spare - spare // 2
    elif spare < 0:
        take = -spare
        fore = max(5, fore - (take + 1) // 2)
        inner_c = max(5, inner_c - take // 2)
    bands, at = [], inner_u[0]
    for name, depth in (("forecourt", fore), ("inner_court", inner_c), ("great", great)):
        bands.append((name, at, min(inner_u[1], at + depth - 1)))
        at += depth + gap
    if bands[-1][1] > inner_u[1] or bands[-1][2] - bands[-1][1] + 1 < 5:
        return None, "the sequence's bands do not fit the rectangle"

    rng = random.Random(f"{seed}/{comp['name']}/axial")
    parts, order = [], []
    n_plot = [0]

    def plot(kind, tname, r, notes, name=None):
        n_plot[0] += 1
        nm = name or f"{comp['name']}_range_{n_plot[0]}"
        decl = decls.get(tname) or {}
        params = {}
        for pn, sp in (decl.get("params") or {}).items():
            if isinstance(sp, (list, tuple)) and sp and sp[0] == "int" and len(sp) >= 3:
                params[pn] = rng.randint(int(sp[1]), int(sp[2]))
            elif isinstance(sp, (list, tuple)) and sp and sp[0] == "choice" and sp[1]:
                params[pn] = rng.choice(list(sp[1]))
        row = {"kind": "area" if kind == "court" else "plot", "name": nm, "type": tname,
               "seed": 1 + (n_plot[0] % 89), "params": params,
               "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3], "notes": notes}
        parts.append(row)
        order.append(nm)
        return row

    lo_p, hi_p = _axial_band(decls.get(plot_t), "plot")
    lo_a, hi_a = _axial_band(decls.get(area_t), "area")
    want_halls = max(1, int(t["count"]))

    def admit(tname, d, w, lo):
        """The largest (depth, width) at or under this that the type actually admits.

                The band is a first cut and the **pad** is the answer: `Builder.pad_extent`
                insets a plot by one on every side where a side of it is under nine, so a lot 36
                by 7 gives a pad of 34 by 5 and a type written for at most 32 refuses it. Asked
                of the same check the plan validator asks, and shrunk a column at a time.
                
        """
        decl = decls.get(tname) or {}
        kind = decl.get("kind", "plot")
        for _k in range(160):
            if d < lo or w < lo:
                return None
            r = {"kind": kind, "x0": 0, "z0": 0, "x1": int(w) - 1, "z1": int(d) - 1}
            if pipeline.needs_footprint_failure(r, decl["needs"]) is None:
                return (int(d), int(w))
            if w >= d:
                w -= 1
            else:
                d -= 1
        return None

    # **The spine, and the wings either side of it.** The sequence is the middle strip:
    # the forecourt inside the gate, the inner court, and the greatest hall across the
    # far end. What a great thing holds beyond those is its **ranges**, and on a
    # precinct a hundred and eighty-eight square a composition asks for fifty of them --
    # so the ground either side of the spine is a grid of halls with lanes between,
    # paired across the axis, and not a single file. The craft round, E5 and E7. the
    # spine's own width: the court an area type will take, at most a third of the span,
    # so the wings either side of it are the greater part of the precinct
    spine_w = max(lo_a, min(hi_a, max(span // 3, lo_a)))
    wing_w = max(0, (span - spine_w - 2 * gap) // 2)
    court_w = span - 2 * (wing_w + gap) if wing_w >= lo_p else span
    court_w = max(lo_a, min(hi_a, court_w))
    wing = wing_w >= lo_p and want_halls > 1
    per_side = max(0, (want_halls - 1 + 1) // 2) if wing else 0

    # the spine's own bands
    bands, at = [], inner_u[0]
    for name, depth in (("forecourt", fore), ("middle", inner_c), ("great", great)):
        bands.append((name, at, min(inner_u[1], at + depth - 1)))
        at += depth + gap
    if bands[-1][1] > inner_u[1] or bands[-1][2] - bands[-1][1] + 1 < 5:
        return None, "the sequence's bands do not fit the rectangle"

    rng = random.Random(f"{seed}/{comp['name']}/axial")
    parts, order = [], []
    n_plot = [0]

    def plot(kind, tname, r, notes, name=None):
        n_plot[0] += 1
        nm = name or f"{comp['name']}_range_{n_plot[0]}"
        decl = decls.get(tname) or {}
        params = {}
        for pn, sp in (decl.get("params") or {}).items():
            if isinstance(sp, (list, tuple)) and sp and sp[0] == "int" and len(sp) >= 3:
                params[pn] = rng.randint(int(sp[1]), int(sp[2]))
            elif isinstance(sp, (list, tuple)) and sp and sp[0] == "choice" and sp[1]:
                params[pn] = rng.choice(list(sp[1]))
        row = {"kind": "area" if kind == "court" else "plot", "name": nm, "type": tname,
               "seed": 1 + (n_plot[0] % 89), "params": params,
               "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3], "notes": notes}
        parts.append(row)
        order.append(nm)
        return row

    # 1. the spine
    great_row = None
    for (name, u_a, u_b) in bands:
        depth = u_b - u_a + 1
        if name == "great":
            got = admit(plot_t, max(lo_p, min(hi_p, depth)),
                        max(lo_p, min(hi_p, court_w if wing else span)), lo_p)
            if got is None:
                return None, "no admitted plot type stands the greatest hall"
            d, w = got
            v_a = inner_v[0] + (span - w) // 2
            great_row = plot("great", plot_t, rect(u_b - d + 1, u_b, v_a, v_a + w - 1),
                             "the greatest hall, at the far end of the axis",
                             name=f"{comp['name']}_great_hall")
            continue
        got = admit(area_t, max(lo_a, min(hi_a, depth)), court_w, lo_a)
        if got is None:
            continue
        d, cw = got
        v_a = inner_v[0] + (span - cw) // 2
        plot("court", area_t, rect(u_a, u_a + d - 1, v_a, v_a + cw - 1),
             "the forecourt inside the gate" if name == "forecourt"
             else "the inner court, before the greatest hall",
             name=(f"{comp['name']}_forecourt" if name == "forecourt"
                   else f"{comp['name']}_inner_court"))

    # 2. the wings: a grid of ranges either side of the spine, paired across it
    laid = 0
    if wing and per_side:
        # **the wings stop where the greatest hall begins**: a range beyond it is a
        # range past the end of the approach, and the axis clause refuses it by name
        wing_u1 = bands[-1][1] - gap - 1
        run_w = max(0, wing_u1 - inner_u[0] + 1)
        cols_n = max(1, min(per_side, wing_w // (lo_p + gap)))
        rows_n = max(1, min(-(-per_side // cols_n), run_w // (lo_p + gap)))
        cell_w = (wing_w - (cols_n - 1) * gap) // cols_n
        cell_d = (run_w - (rows_n - 1) * gap) // rows_n
        got = admit(plot_t, cell_d, cell_w, lo_p)
        if got is not None:
            cell_d, cell_w = got
            for r_i in range(rows_n):
                uu = inner_u[0] + r_i * (cell_d + gap)
                if uu + cell_d - 1 > wing_u1:
                    break
                for c_i in range(cols_n):
                    # **a pair or nothing**: a range on one side of an approach with
                    # nothing opposite it is not a flanking range
                    left = inner_v[0] + c_i * (cell_w + gap)
                    right = inner_v[1] - c_i * (cell_w + gap) - cell_w + 1
                    if left + cell_w - 1 >= right:
                        break
                    for v_a in (left, right):
                        plot("range", plot_t,
                             rect(uu, uu + cell_d - 1, v_a, v_a + cell_w - 1),
                             "a flanking range, paired across the axis")
                        laid += 1

    # 3. **the courts and gardens between them**: every free rectangle inside the wall
    # that an area type will take, largest first, until none is left. A composition asks
    # a great thing for a cover as well as a count, and the ground the grid leaves
    # between its ranges is what the courts of a palace actually are.
    import numpy as _np
    from .district_compile import _largest_free as _free_rect
    UU, VV = inner_u[1] - inner_u[0] + 1, inner_v[1] - inner_v[0] + 1
    if UU > 0 and VV > 0:
        taken = _np.zeros((UU, VV), dtype=bool)
        for p_ in parts:
            if p_.get("x0") is None:
                continue
            for uu_ in range(UU):
                pass
            # mark the part's own box, grown by the clearance, in (u, v)
            r0 = (p_["x0"], p_["z0"], p_["x1"], p_["z1"])
            for xx in range(r0[0], r0[2] + 1):
                for zz in range(r0[1], r0[3] + 1):
                    uu_, vv_ = ((xx - x0, zz - z0) if horiz else (zz - z0, xx - x0))
                    if flip:
                        uu_ = U - 1 - uu_
                    uu_ -= inner_u[0]
                    vv_ -= inner_v[0]
                    # a lot keeps the plot clearance; one piece of open ground keeps
                    # only an area's, which is what the validator holds two areas to
                    k_ = gap if p_.get("kind") != "area" else 2
                    if 0 <= uu_ < UU and 0 <= vv_ < VV:
                        taken[max(0, uu_ - k_):uu_ + k_ + 1,
                              max(0, vv_ - k_):vv_ + k_ + 1] = True
        room = ~taken
        for _k in range(64):
            r_ = _free_rect(room)
            if r_ is None:
                break
            ua_, ub_, va_, vb_ = (int(v) for v in r_)
            got = admit(area_t, ub_ - ua_ + 1, vb_ - va_ + 1, lo_a)
            if got is None:
                room[ua_:ub_ + 1, va_:vb_ + 1] = False
                continue
            ad, aw = got
            plot("court", area_t,
                 rect(inner_u[0] + ua_, inner_u[0] + ua_ + ad - 1,
                      inner_v[0] + va_, inner_v[0] + va_ + aw - 1),
                 "a court between the ranges", name=f"{comp['name']}_court_{n_plot[0]}")
            room[max(0, ua_ - 2):ua_ + ad + 2, max(0, va_ - 2):va_ + aw + 2] = False

    great_row = next((p for p in parts if p["name"].endswith("_great_hall")), None)
    if great_row is None:
        return None, "the greatest hall did not fit"
    biggest = max(((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1))
                  for p in parts if p["kind"] == "plot")
    if (great_row["x1"] - great_row["x0"] + 1) \
            * (great_row["z1"] - great_row["z0"] + 1) < biggest:
        return None, "a flanking range is larger than the greatest hall"

    # the wall round it, and the gate where the road arrives
    wall_t = _axial_type(decls, "edge", part, spec)
    gate_t = _axial_type(decls, "point", part, spec)
    if made_of["walled"] and wall_t:
        i = COMPOUND_WALL_INSET
        wx0, wz0, wx1, wz1 = x0 + i, z0 + i, x1 - i, z1 - i
        # a vertex at least every `max_run`: the edge type declares how long a single
        # segment it stands, and a 188-square precinct's side is 182
        max_run = int((decls.get(wall_t) or {}).get("needs", {})
                      .get("footprint", (1, 4, 3, 128))[3])
        path = _square_path((wx0 + wx1) // 2, (wz0 + wz1) // 2,
                            (wx1 - wx0) // 2, max_run)
        parts.append({"kind": "edge", "name": f"{comp['name']}_wall", "type": wall_t,
                      "seed": 1, "params": {"height": 8, "width": 1}, "width": 1,
                      "path": path, "notes": "the wall round the precinct"})
        if made_of["gated"] and gate_t:
            # **on the wall, at the middle of the side the road arrives on**: the gate
            # is a point of the wall's own path and the validator reads it off that
            mid_x, mid_z = (wx0 + wx1) // 2, (wz0 + wz1) // 2
            want = {"north": (mid_x, wz0), "south": (mid_x, wz1),
                    "west": (wx0, mid_z), "east": (wx1, mid_z)}[side]
            on = min(_path_cells(path),
                     key=lambda c: abs(c[0] - want[0]) + abs(c[1] - want[1]))
            parts.append({"kind": "point", "name": f"{comp['name']}_gate",
                          "type": gate_t, "seed": 2, "params": {},
                          "at": [int(on[0]), int(on[1])], "facing": side, "size": 5,
                          "notes": "the gate, where the road arrives"})
    got = {"notes": (f"Laid as a sequence from the {side} gate: "
                     + " -> ".join(order)), "axis": side, "parts": parts}
    return got, {"compound": comp["name"], "axis": side, "bands": [b[0] for b in bands],
                 "order": order, "plot_type": plot_t, "area_type": area_t,
                 "great_hall": great_row["name"],
                 "great": [great_row["x1"] - great_row["x0"] + 1,
                           great_row["z1"] - great_row["z0"] + 1],
                 "parts": len(parts)}


def _path_cells(path: list) -> list:
    """Every column a path passes through, in order."""
    out = []
    for a, b in zip(path, path[1:]):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for i in range(n + 1):
            out.append((a[0] + (b[0] - a[0]) * i // max(1, n),
                        a[1] + (b[1] - a[1]) * i // max(1, n)))
    return out or [tuple(p) for p in path]


def _axis_failure(comp: dict, place: dict, halls: list, courts: list) -> str | None:
    """Why this compound's parts are not a sequence from its gate, or None.

        Two things, and both are what a person walking in would notice missing: the
        **greatest hall** -- the largest plot inside the wall -- is the **last** thing on the
        axis, and there is a **court** between the gate and it. The craft round, E5. The
        fuller sequence -- forecourt, ranges, inner court, the greatest hall -- is what
        `compound_axial` lays and what its own case holds it to; a compound a model drew is
        held to the two a person sees.
        
    """
    x0, z0 = min(comp["x0"], comp["x1"]), min(comp["z0"], comp["z1"])
    x1, z1 = max(comp["x0"], comp["x1"]), max(comp["z0"], comp["z1"])
    cells = ((place.get("arterials") or {}).get("joins") or {}).get(comp.get("name")) or []
    side = _arrival_side(comp, cells) if cells else "north"

    def along(p):
        """How far this part's centre is from the gate, along the axis."""
        r = pipeline.part_rect(p)
        cx, cz = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
        if side == "north":
            return cz - z0
        if side == "south":
            return z1 - cz
        if side == "west":
            return cx - x0
        return x1 - cx

    def area(p):
        r = pipeline.part_rect(p)
        return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)

    great = max(halls, key=lambda p: (area(p), p["name"]))
    d_great = along(great)
    behind = [p for p in halls if p is not great and along(p) > d_great]
    if behind:
        return (f"the greatest hall is the last thing on the axis and "
                f"{behind[0]['name']} stands beyond {great['name']}: this is a "
                f"{comp.get('family') or 'compound'} and a compound of its family is an "
                f"approach -- the gate, a forecourt, the ranges, an inner court, and the "
                f"greatest hall at the far end")
    if len(halls) > 1 and not [p for p in courts if along(p) < d_great]:
        return ("there is no court between the gate and the greatest hall: an approach "
                "is a sequence of pauses and this one is a wall of building")
    return None


def _axial_band(decl: dict | None, kind: str = "plot") -> tuple:
    """(smallest, largest) **drawn** side this type admits, off its declared footprint.

        A plot is drawn as a plot and built on the pad `site()`'s inset leaves of it, so its
        band is the footprint plus the inset on both sides -- the same conversion
        `district_compile._plot_range` makes and `Builder.pad_extent` is the one place of.
        An edge, a point and an area are drawn as they are built. The craft round: without
        it an axial compound on a 188-square podium divided its flanking strips into cells
        of 32x3, which the footprint check then refused one by one.
        
    """
    from .buildlib import Builder
    if not decl:
        return (3, 3)
    a, b, c, d = (decl.get("needs") or {}).get("footprint", (3, 3, 3, 3))
    i = 2 * Builder.SITE_INSET if kind == "plot" else 0
    return (max(a, b) + i, min(c, d) + i)


def _axial_type(decls: dict, kind: str, part: dict | None, spec: dict | None) -> str | None:
    """The type an axial compound builds this kind of part out of: the largest-banded
    one of the compound's own role first, so the greatest hall is as great as the
    library can make it."""
    best = None
    for name, d in sorted((decls or {}).items()):
        if d.get("kind", "plot") != kind:
            continue
        lo, hi = _axial_band(d)
        want = (d.get("role") == ((part or {}).get("role") or "civic"), hi, name)
        if best is None or want > best[0]:
            best = (want, name)
    return best[1] if best else None
