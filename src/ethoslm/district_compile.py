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

import contextlib
import hashlib
import math
import os
import random
import re

import numpy as np

from . import pipeline, spec as spec_mod
from .buildlib import Builder, WALL_ALT_SHARE, WALL_ALT_TOLERANCE

#: **Registered before it was read** (v2, C1): the least of a compiled district's
#: rectangle its **plots** cover, per density word. The validator's own floor --
#: `placeplan.DISTRICT_MIN_FRACTION` of the density's plot share
#: (`placeplan.occupancy_shares`) -- rounded to two places, written here as numbers so
#: the compiler's proof reads against a bar and not against the arithmetic it is being
#: tested with. **Re-registered by the craft round, E1**, before the numbers that test
#: it were read, because the plot share it is three quarters of is no longer a square
#: plot with a lane on four sides but the compiler's own fabric (`placeplan.fabric`):
#: 0.32 / 0.39 / 0.45 / 0.43 against the 0.20 / 0.28 / 0.34 / 0.33 it was. What it was:
#: sparse 0.15, low 0.21, medium 0.25, dense 0.25.
PLOT_COVER = {"sparse": 0.24, "low": 0.29, "medium": 0.33, "dense": 0.32}

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

#: How much of a rectangle the ground has to be able to carry before this compiler will
#: lay anything on it. The spatial-design round, and the coordinator's ground interface
#: (`ethoslm.feasible`): a district now carries a measured mask of the columns a building
#: may be founded on, and a plot **most of whose columns the mask refuses** is a plot on
#: the lake. Half, and not all, for a reason the round's own figures give: on
#: `middle_ring_north_west` 35.4% of the rectangle can carry a building, and a bar of
#: "every column" would lay nothing at all there and turn a district that should compile
#: smaller into a district that does not compile. `site()` cuts and fills a pad, and the
#: ground a plinth can absorb is exactly what this slack is for; below half the plinth
#: is a causeway. A block and a piece of open ground are held to the same bar, because
#: the question -- is there ground here to build on -- is the same one.
GROUND_FOUNDED = 0.5

#: **This compiler consults the feasibility mask.** Declared rather than assumed,
#: because `placeplan.district_failures`' `ground` clause and
#: `arrange.NEGOTIABLE_CHECKS` both switch on it. The flag is what lets the layout owner
#: turn the refusal on without either side guessing.
LAYS_ON_FEASIBLE_GROUND = True

#: How many lots long a block is, per density word, where the character does not say. A
#: loose fabric on a three-lot block spends a quarter of its ground on streets -- which
#: is a suburb, not farmland -- so the looser the density the longer the block. The
#: compiler lengthens it further where the ground is still short of its cover.
BLOCK_LOTS = {"sparse": 5, "low": 4, "medium": 3, "dense": 3}
BLOCK_LOTS_MAX = 8


def block_lots_for(density: str, lot_width: int | None = None,
                   gap: int | None = None) -> int:
    """How many lots long a block of this density's fabric is, at **this** lot width.

        **A block is a length of street, and a count of lots is only the same thing while
        every lot is the same width.** The spatial-design round, measured through
        `placeplan.fabric`: `BLOCK_LOTS["dense"]` is 3, and it was written when a dense lot
        was the density's own 10-column square -- 30 columns of frontage between two cross
        streets. A row of party walls is 5 or 6 columns wide, so the same 3 lots is 18
        columns, and a block that shrank with its lots while `PLOT_LANE` stayed at 5 spends
        *more* of the ring on street the narrower its houses get. That is backwards: a narrow
        frontage is the thing attachment is for, and it was being charged for itself.

        Measured on the retained dense fabric (`fabric("dense", "urban", {"attached": True})`,
        a 6x8 lot): 3 lots a block is 552 columns holding 4.95 houses, **111.5 columns a
        house**; the same lot on a block holding the density's own 30 columns of frontage is
        5 lots, 840 columns, 8.25 houses, **101.8**. Nine per cent of the crowded ring was
        going to cross streets that the square fabric this number was written for never paid
        for.

        So the count is the density's own block *length* divided by the width of the lot
        actually being laid, floored at the registered count and capped at `BLOCK_LOTS_MAX`
        (a block is still a block, and the cap is where `arrange.alternatives` already stops).
        A lot at or above the density's own side gets the registered number unchanged, which
        is every unattached fabric in the library: sparse 5, low 4, medium 3, dense 3.
        
    """
    base = int(BLOCK_LOTS.get(density, 3))
    if not lot_width or int(lot_width) <= 0:
        return base
    # `density_lot` and not `occupancy_shares`: the shares are derived from `fabric`,
    # which is one of this function's callers, and asking them here is a cycle.
    from .placeplan import density_lot, DENSITY_ROLE
    try:
        side = int(density_lot(density, DENSITY_ROLE.get(density))["side"])
    except (KeyError, TypeError, ValueError):
        return base
    g = 0 if gap is None else int(gap)
    # the frontage the registered count buys at the density's own square lot, and the
    # number of *these* lots that stands in it
    want = base * side + (base - 1) * g
    n = int(round((want + g) / float(int(lot_width) + g))) if (int(lot_width) + g) else base
    return max(base, min(BLOCK_LOTS_MAX, n))


#: The executed namespaces of the type files `storey_pad` has read, by path and size.
_STOREY_PAD: dict = {}


def storey_pad(decl: dict, storeys: int) -> tuple | None:
    """`(across, along)` of the least **pad** this type needs to stand `storeys` storeys,
        off its own `STOREY_PAD` declaration, or None where it declares nothing.

        **The other half of the question a lot asks.** The spatial-design round: `NEEDS`
        says what ground a type may stand on and nothing said what standing there *gets*,
        so the layout was reading a type's footprint ceiling as though it were a statement
        about storeys -- a capability limit standing in for a demand. A density word asks for
        an amount of ground and a shape; this is the type's answer about what it can do with
        it, declared by the file that has to build it and measured there.

        Optional, per the rule every other type declaration here follows: a file that says
        nothing gets None and every caller behaves exactly as it did.
        
    """
    try:
        n = int(storeys)
    except (TypeError, ValueError):
        return None
    if n <= 1 or not isinstance(decl, dict) or not decl.get("path"):
        return None
    key = (decl["path"], len(decl.get("src") or ""))
    got = _STOREY_PAD.get(key)
    if got is None:
        ns: dict = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
        try:
            exec(compile(decl["src"], decl["path"], "exec"), ns)       # noqa: S102
            table = ns.get("STOREY_PAD") or {}
            got = {int(k): (int(v[0]), int(v[1])) for k, v in dict(table).items()}
        except Exception:                        # noqa: BLE001 -- no table, no answer
            got = {}
        _STOREY_PAD[key] = got
    if not got:
        return None
    # the declared row for this many storeys, or the deepest row at or under it: a type
    # that declares 1, 2 and 3 answers a band of 2 with its 2 and never with its 3.
    rows = sorted(k for k in got if k <= n)
    return got[rows[-1]] if rows else None

#: How much of a block's length the leftover along a street may be before the lots are
#: widened to take it up: under this, the remainder is a wider last gap.
WIDEN_UP_TO = 0.5

#: Every fourth lot takes another of the role's types rather than the house, where the
#: role has more than one plot type that admits the lot. **The floor and not the rule**
#: since the craft round (E3): a lot draws its own type from everything the role admits
#: at its size, and this is what guarantees a second type on a short street where the
#: draw might not reach one. **Within the quarter's own uses**, since the composition
#: round -- see `use_mix`. It used to mean "another of the *pool's* types", and the pool
#: a capability record approves for a ring of traders includes the civic types the
#: ring's landmark needs, so every fourth lot of the middle ring drew a hall or a
#: temple: the built section came back with 5 temples and 4 halls against 1 shop house
#: in 22 buildings. Variety inside the intended uses is a street of dwellings and shops;
#: variety across uses is a civic precinct.
OTHER_EVERY = 4

#: **Retired by the neighbourhood round, and kept here with its reasoning.** The
#: composition round registered a tenth of a district's lots as the share it might spend
#: on a use other than the quarter's own, on the argument that "a tenth is one building
#: in a block of ten, which is a guild hall on a long street of houses". The number
#: worked as a brake on civic filler and it was still a **universal quota**: the same
#: tenth in a traders' quarter, a farm belt and an elite ring, in a project whose whole
#: claim is that a place's composition is derived from its own design. The neighbourhood
#: round's instruction is explicit -- "infer and record this neighbourhood's mix from
#: the design... avoid universal quotas". What replaced it (`use_mix`): the uses a
#: quarter holds are the ones its **own design names** -- its resolved demand's
#: requirements and the words of its own description -- and *where* each use stands is a
#: spatial decision (the street the quarter's anchor fronts) rather than a fraction. The
#: share is then **measured** off what was laid and recorded with its derivation, which
#: is the opposite direction of travel from this constant. Nothing reads it; it is here
#: so the abandoned rule and the reason it was abandoned stay legible beside the rule
#: that replaced it.
PROGRAMME_USE_SHARE = 0.1

#: **The one parameter key in this library that names a building's occupation rather
#: than its form.** `shop_house` and `workshop` declare `trade` as a choice of the
#: trades they house; no other type declares it, and no other parameter key in the
#: library names what goes on inside rather than what the building looks like. This is a
#: **convention and it is named as one**. The clean statement would be `FUNCTION =
#: "trade"` on those two types, of exactly the kind `FUNCTION = "dwelling"` already is
#: -- at which point this constant becomes dead and `FUNCTION_USE` carries the whole
#: answer. the request is on the record instead.
TRADE_PARAM = "trade"

#: **What a declared `FUNCTION` means as a use of a quarter.** A translation table and
#: not a vocabulary: every word on the left is a `FUNCTION` some type on disk declares,
#: and a function this table does not know is carried through under its own name rather
#: than being dropped (`type_use`).
FUNCTION_USE = {"dwelling": "home", "market": "market", "worship": "worship"}

#: The use a type of the district's own role is counted under when it declares no
#: function at all -- `court_small` and `court_large` are the library's two. It is an
#: **inference from the role** and never a declaration, and `type_use` says so in the
#: reason it returns, so a reader can tell the two apart on the record.
UNSTATED_USE = "unstated"

#: **How far a lot's width and depth may stray from the density's own**, as a fraction
#: of the lot side, by what the street is. The craft round, E3: the compiler divided a
#: street's frontage evenly, so every lot was one width and the building on it was one
#: building -- twenty-one cottages on identical pads in a village, and a district that
#: reads from the air as a comb. A terrace is *meant* to be regular, so an attached row
#: varies least; a street of detached houses varies more; freestanding buildings a lane
#: apart on open frontage vary most, because nothing lines them up.
VARIETY = {"open": 0.35, "street": 0.3, "attached": 0.0}

#: **Registered before the numbers that test them** (the craft round, E3), and read on
#: every compiled district's own record: shapes_per_hundred distinct building shapes --
#: the type, the width, the depth and the storeys, which is what a person sees of a
#: house from outside -- per hundred houses in the district. identical_run the longest
#: run of neighbours of one shape along one street face. A rhythm is not a comb. What
#: the compiler read before the phase, on the compile fixture: **30.2 per hundred and a
#: run of 3** for a medium detached district, **15.8 and 3** for a row of party walls.
#: Both numbers are set above that, so neither is met by standing still.
SHAPES_PER_HUNDRED = 35
IDENTICAL_RUN_MAX = 2


# ------------------------------------------------------------------- the character

#: **What the layout may settle about a district's own arrangement**, over the character
#: its defining part declares. The design round's second contract: a parent's dimensions
#: and its children's arrangement are negotiated together, so the ring layout has to be
#: able to say "this ring is one row of houses deep, in bays this wide" and have the
#: compiler lay exactly that -- and the compiler stays the authority on whether it
#: holds. These are decisions about *this rectangle*, so they live on the district
#: record and not on the defining part, whose character is a statement about the whole
#: fabric. A value the spec's own character declared is never overridden (`_declared`).
#: `perimeter` is the neighbourhood round's addition: whether a block is built on all
#: four of its faces with its court inside, rather than on its two long faces with the
#: back row given to the court. It is a decision about *this rectangle* like the rest of
#: them -- a block has to be deep enough and long enough to hold one -- and the compiler
#: stays the authority on whether it fits (`_compile_once`; `counts.perimeter_blocks`
#: says how many blocks actually became one).
ARRANGEMENT_FIELDS = ("rows", "lot_width", "lot_depth", "frontage", "attached",
                      "courtyard_share", "block", "perimeter")

#: How many rows of lots one block holds where nothing says. Two: a block is two rows
#: back to back, which is what every district in the record has been.
BLOCK_ROWS = 2


def character_of(part: dict, district: dict | None = None) -> dict:
    """The full character of a district part: the spec's words over
        `spec.CHARACTER_DEFAULTS` for its density word, with the numbers the defaults
        leave to the density (`block`, `lot_depth`) filled from the density's own lot,
        and the **arrangement** the layout negotiated for this rectangle over both.

        The arrangement never overrides a value the spec's own character declared: the
        declaration is the principal's and the arrangement is the layout owner's inference
        about one rectangle, in that order.
        
    """
    density = part.get("density") or "medium"
    out = dict(spec_mod.CHARACTER_DEFAULTS[density])
    declared = {k for k, v in (part.get("character") or {}).items() if v is not None}
    for k in declared:
        out[k] = part["character"][k]
    # **The arrangement stands over the character, and says that it does.** The
    # character is the district brief's author speaking about the fabric in general; the
    # arrangement is the layout owner's settled decision about *this rectangle*, taken
    # after it with the compiler's own capacity in hand and recorded as a negotiated
    # allocation row. Where it moves a value the character declared,
    # `arrangement_revises` names it, so a reader can see which words of the brief the
    # layout overruled and why.
    revised = []
    # **A district composed from its streets has no grid arrangement to adopt** (the
    # fabric reset round): an `arrangement` is the grid's settled lot, rows and block
    # form for this rectangle, and one left on a district by an earlier grid layout --
    # the quarter design round's `perimeter` of 13x13 lots -- would re-cut the streets'
    # lots to the old grid's size. The composition searches its own proposals.
    street = str(out.get("layout") or "") == "street"
    for k, v in ((district or {}).get("arrangement") or {}).items():
        if v is None or k not in ARRANGEMENT_FIELDS or street:
            continue
        if k in declared and out.get(k) != v:
            revised.append(k)
        out[k] = v
    if revised:
        out["arrangement_revises"] = sorted(revised)
    # **What the request requires is a floor under what the character prefers.** The
    # resolved demand (`demand.resolve`, written on the district before it was sized)
    # carries the parameters the requirement made required -- a second storey the
    # sentence asked for -- and a character band that admits one storey is a preference,
    # not permission to drop it. Raised, never lowered, and the district's record says
    # the demand raised it.
    params = ((district or {}).get("demand") or {}).get("params") or {}
    want = params.get("storeys")
    if want is not None:
        lo_w = int(want[0] if isinstance(want, (list, tuple)) else want)
        band = out.get("storeys")
        hi_w = max(lo_w, int(band[1]) if band else lo_w)
        out["storeys"] = [lo_w, hi_w]
    # **The parent's programme distribution stands over the character's landmarks.** The
    # quarter design round: where the layout re-cut a sector into pieces, the sector's
    # landmarks belong to its piece nearest the gate, and the others carry none
    # (`stages_plan._negotiate_sectors`), so a strip cut in four does not lay four
    # markets.
    if district is not None and district.get("landmarks") is not None:
        out["landmarks"] = list(district["landmarks"])
    out["density"] = density
    out["role"] = part.get("role")
    return out


def block_depth(lot_depth: int, rows: int, row_gap: int) -> int:
    """How deep a block of `rows` rows of `lot_depth`-deep lots is, back to back."""
    rows = max(1, int(rows))
    return rows * int(lot_depth) + (rows - 1) * int(row_gap)


def district_depth(lot_depth: int, rows: int, row_gap: int, *,
                   street: int | None = None, edge: int | None = None) -> int:
    """**The least depth a district of these blocks needs**, in columns: its two edge
        margins, the street its lots front on, one block, and the clearance behind it.

        The one arithmetic the ring layout and the compiler share on this question. The
        expression round's dead end was that the ring's least width carried a *constant*
        district depth of 28 whatever the fabric was -- so a ring whose count needed 22
        stood at 37 and the density was measured over the difference. A district one row of
        seven-deep lots thick is nineteen columns, and that is a decision this owner can
        take.
        
    """
    from .placeplan import PLOT_LANE
    st = PLOT_LANE if street is None else int(street)
    ed = EDGE_MARGIN if edge is None else int(edge)
    return int(2 * ed + st + block_depth(lot_depth, rows, row_gap) + LOT_GAP)


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


def _admits(decl: dict, w: int, d: int, attached=None, u_is_x: bool = True) -> bool:
    """Does this plot type stand on a w x d plot, with these sides attached? Its
        envelope and its except list, read in the validator's own units
        (`needs_footprint_failure`, through `Builder.pad_extent`).

        `w` and `d` are **in the district's frame** -- `w` along its streets, `d` across
        them -- and `attached` names sides of the **world**. Those two are the same thing
        only where the district's streets run along x, and `u_is_x` is what says whether
        they do.

        Found by running the city: a dense ring district whose streets run along z tested
        its party-walled lot as 8 across x and 6 along z with its flanks named north and
        south, so the attachment was applied to the axis the lot was *not* long on. The
        pad came out 6x5 and passed; the lot the compiler then drew was 6 by 8 with its
        south flank attached, whose pad is 4x7, and the plan validator refused it
        forty-seven times in one district. Without attachment the two orientations are the
        same question -- the inset is symmetric and the validator sorts the pair -- which is
        why this never bit until a type with party walls met a district on the other grain.
        
    """
    x_span, z_span = (w, d) if u_is_x else (d, w)
    part = {"kind": "plot", "x0": 0, "z0": 0, "x1": x_span - 1, "z1": z_span - 1}
    if attached:
        part["attached"] = list(attached)
    return pipeline.needs_footprint_failure(part, decl["needs"]) is None


def _attached_lot(decl: dict, side: int, flanks: tuple,
                  u_is_x: bool = True, area: int | None = None,
                  want: tuple | None = None,
                  shape: tuple | None = None) -> tuple | None:
    """**(w, d) of a lot in a row of party walls**: the lot nearest the **ground** a
        house of this density gets that this type admits with both flanks attached, with
        one, and with none -- because a neighbour dropped for the road or a standing part
        frees a flank, and the pad's inset comes back on it (`Builder._insets`). None where
        no size does.

        **Nearest by area, not by side**, the neighbourhood round, and the ranking it
        replaces is why the library's row house has been a 6x6 box:

            ws = sorted(..., key=lambda v: (abs(v - side), v))
            ds = sorted(..., key=lambda v: (abs(v - side), v))

        -- both axes ranked by distance from the *same* number, so the first admitted pair
        is the **squarest** one and a type whose declared sentence is "narrow to the lane,
        deep into the plot" can never be given its own shape. Worse, it made the type's band
        unwidenable: re-sweeping `row_house` to `(4, 4, 6, 24)` under the old rank moves the
        dense fabric lot from 6x8 to 10x10, which is bigger lots and fewer houses, so the
        correct measurement was being suppressed by a ranking rule.

        Area is the thing a density word actually means (`placeplan.occupancy_shares`' plot
        is a *number of columns*), and among lots of equal ground the narrower frontage is
        the one attachment is *for*: more front doors on a length of street is the whole
        point of a party wall. `area` is the ground a house of this density gets; `side`
        stands in for it where no caller passed one, so every existing call behaves as a
        square ask.

        **`shape` is the (frontage, depth) the density asked for**, the spatial-design round,
        and it is what makes ranking by area safe to widen a band under. Area alone is a
        circle round a number: with the depth ceiling at 6 the nearest lot to a hundred
        columns was 6x8 (48) because nothing deeper was admitted, and the moment the ceiling
        moved the *exact* hundred -- 10x10 -- became available and won, which is the 182
        columns a house the last round measured and refused to adopt. A density word that can
        only say "a hundred columns" cannot tell a narrow deep terrace from a large square
        plot; `placeplan.density_lot` says both now. Depth is ranked first and as a floor,
        because depth is what carries a stair and a lot one column short of it stands one
        storey however much ground it has; then frontage nearest the ask; then area; then,
        unchanged, the narrower and shallower lot, because among lots of equal ground more
        front doors on a length of street is what a party wall is for.
        
    """
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 2 * Builder.SITE_INSET

    def ok(w, d):
        return all(_admits(decl, w, d, att, u_is_x)
                   for att in (flanks, flanks[:1], flanks[1:], ()))

    if want and len(want) == 2 and int(want[0]) > 0 and int(want[1]) > 0 \
            and ok(int(want[0]), int(want[1])):
        return int(want[0]), int(want[1])
    goal = int(area or (int(side) * int(side)))
    sw = sd = None
    if shape and len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
        sw, sd = int(shape[0]), int(shape[1])
    cands = [(w, d)
             for w in range(max(3, lo_w), hi_w + i + 1)
             for d in range(max(3, lo_d), hi_d + i + 1)
             if ok(w, d)]
    if not cands:
        return None
    if sw is None:
        return min(cands, key=lambda g: (abs(g[0] * g[1] - goal), g[0], g[1]))
    return min(cands, key=lambda g: (0 if g[1] >= sd else 1, abs(g[1] - sd),
                                     abs(g[0] - sw), abs(g[0] * g[1] - goal),
                                     g[0], g[1]))


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


def _side_at_least(decl: dict, n: int) -> int:
    """The smallest plot side this type admits that is at least `n`, or the largest it
    admits where none is: what a **minimum** lot clamps to (a nearest side can be
    smaller than the minimum, which defeats it). The expression round."""
    lo, hi, _ex = _plot_range(decl)
    for cand in range(max(lo, n), hi + 1):
        if _admits(decl, cand, cand):
            return cand
    for cand in range(hi, lo - 1, -1):
        if _admits(decl, cand, cand):
            return cand
    return max(lo, min(hi, n))


def _side_within(decl: dict, n: int) -> int | None:
    """The largest plot side this type admits that is at most `n`, or None."""
    lo, hi, _ex = _plot_range(decl)
    for cand in range(min(hi, n), lo - 1, -1):
        if _admits(decl, cand, cand):
            return cand
    return None


def _deepest(decl: dict, w: int, most: int, flanks=(),
             u_is_x: bool = True) -> int:
    """The deepest lot of width `w` this type admits inside `most` columns -- checked
    at the width it will actually be built at, and in every configuration of its
    flanks where it is attached. 0 where none does. A block too shallow for two rows
    takes one deep one, and asking whether a *square* of that depth is admitted said
    yes to a 6x10 lot of a type written for a 6x6 pad."""
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    i = 2 * Builder.SITE_INSET
    atts = ([tuple(flanks), tuple(flanks[:1]), tuple(flanks[1:]), ()]
            if flanks else [()])
    for d in range(min(most, hi_d + i), max(3, lo_d) - 1, -1):
        if all(_admits(decl, w, d, a, u_is_x) for a in atts):
            return d
    return 0


def house_types(decls: dict, role: str | None, form: str | None = None,
                approved: list | None = None) -> list:
    """The plot types a district of this role may build, the role's own first.

        `approved` is the pool the **capability record** matched for this district's fabric.
        Where one is given it is the authority and this filters to it; where none is, every
        type of the role is admitted, as it always was.

        **The pool's order is the pool's, and losing it made a whole repair ineffective.**
        The review's second finding, measured: the layout owner's fabric action re-ordered
        the approved pool smallest-footprint first, and this function sorted by role and name
        before returning it -- so `[farmstead, cottage]` and `[cottage, farmstead]` produced
        the identical compiler choice, and the run's record of "the fabric alternative was
        tried and changed nothing" was not evidence about fabric at all. Restricting the
        pool's *membership* did reach the compiler, which is what proved that it was the
        ordering specifically that was being discarded. So where a pool is given, its order
        is preserved: the district's own first choice is the pool's first entry, and the
        role-then-name sort remains the rule only for the unrestricted library.

        **The review's first finding, at the level it actually bites.** The capability record
        matches a `fabric` want per district and names the type it chose and its runners-up;
        the compiler then re-derived its own pool from the whole library by role. Two rules,
        one library, two answers -- and the one the record wrote down was not the one that
        built the houses. A record nothing consults is a document, not a decision.
        
    """
    allow = {str(a) for a in approved} if approved is not None else None
    #: Where the pool was given, its position in it; the role-then-name rank otherwise.
    rank_in_pool = ({str(a): i for i, a in enumerate(approved)}
                    if approved is not None else {})
    out = []
    for name, d in sorted(decls.items()):
        if d.get("kind", "plot") != "plot":
            continue
        if allow is not None and name not in allow:
            continue
        if not pipeline.role_ok(d.get("role"), role):
            continue
        if form and not pipeline.form_ok(d.get("form"), form):
            continue
        out.append(((rank_in_pool[name],) if name in rank_in_pool
                    else (len(rank_in_pool) + (0 if d.get("role") == role else 1),),
                    name, d))
    out.sort()
    return [(n, d) for _r, n, d in out]


def terrace_order(pairs: list) -> list:
    """**Which attached type a density's arithmetic is charged for**, when nothing
    names one. The fabric reset round: once `courtyard_house` and `shop_house` declared
    `ATTACHED`, "the first attached type" in the table's alphabetical order made the
    crowded ring's party-walled terrace a row of courtyard houses in the density
    arithmetic (`placeplan.fabric`). A density word's own lot is a home's -- not a shop's
    -- and the least one the library lays, so homes come first, the smallest declared
    pad first among them; a district whose own design names its type still leads with it
    (`use_mix`)."""
    def key(nd):
        n, d = nd
        use = type_use(n, d)[0]
        fp = ((d or {}).get("needs") or {}).get("footprint") or (99, 99)
        return (0 if use in ("home", UNSTATED_USE) else 1,
                int(fp[0]) * int(fp[1]), str(n))
    return sorted(list(pairs), key=key)


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


# ------------------------------------------------- what the request *requires* here The
# composition round's first change: "markets, courts, streets and working land made
# necessary by the programme must reserve usable space before ordinary housing consumes
# it". To reserve space for a thing one must first know that the request asked for it,
# and the compiler had no way to tell a landmark the character *preferred* from one the
# sentence **requires** -- so the area branch below dropped an unplaceable market into
# `kind = "row"` and laid houses on the ground it was for, and not one field of the
# record said so. These three functions are how requiredness arrives, and none of them
# names a market, a place or a type: the fact comes off the district's resolved demand.


def _requirements_of(district: dict | None) -> list:
    """`[(kind, wants, id)]` -- the requirements this district resolved, by id.

        `demand.resolve` writes the requirement **ids** that made a district's features
        required onto `district["demand"]["requirements"]`; a caller that has the whole
        requirement records may put them on `district["requirements"]` instead, and both are
        read. An id is `kind/what` by this project's own convention throughout
        `intent.py` -- `function/market`, `feature/farmland`, `quality/density/dense@lower` --
        so where only the id is in hand, `wants` is reconstructed from it. That is a
        convention and it is named as one: a requirement record with its own `wants` is
        always preferred over the id it was written under.
        
    """
    out = []
    seen = set()
    src = list(((district or {}).get("demand") or {}).get("requirements") or ())
    src += list((district or {}).get("requirements") or ())
    for r in src:
        if isinstance(r, dict):
            rid = str(r.get("id") or "")
            kind = str(r.get("kind") or rid.partition("/")[0])
            wants = dict(r.get("wants") or {})
        else:
            rid = str(r)
            kind, _sep, what = rid.partition("/")
            wants = {kind: what.split("@")[0]} if what else {}
        if rid in seen:
            continue
        seen.add(rid)
        out.append((kind, wants, rid))
    return out


def requirement_for(tname: str | None, decl: dict | None,
                    district: dict | None) -> str | None:
    """**Which requirement of this district a leaf of this type answers**, or None."""
    if not tname:
        return None
    d = decl or {}
    mine = {str(tname).lower(), str(d.get("function") or "").lower(),
            str(d.get("family") or "").lower()}
    mine.discard("")
    for _kind, wants, rid in _requirements_of(district):
        for key, val in (wants or {}).items():
            v = str(val or "").strip().lower()
            if not v:
                continue
            # the requirement's own word against what this type declares under the same
            # key, and against the three ways a type says what it is
            if v in mine or str(d.get(key) or "").lower() == v:
                return rid
    return None


#: `{type: side}` -- what the envelope measured as the least side a landmark type needs
#: to deliver the features it declares. One probe per type per process;
#: `envelope.lot_for` keeps its own disk cache under the run.
_LANDMARK_LEAST: dict = {}


def landmark_least(tname: str | None, decl: dict | None) -> int:
    """**The least side a landmark of this type delivers its own features on.**

        `_plot_range`'s `lo` is the smallest plot the type's sweep *stood* on, which for
        `market` is 7 -- and a 7-column market paves its ground and reports `stalls` omitted,
        because the type says so in its own docstring. A reservation sized at that floor is
        ground claimed for a feature that will not be there, which is the same defect as not
        reserving it. So the floor is what the **envelope** measured for the features the
        type declares (`envelope.lot_for(..., features=FEATURES)`): 14 for `market`, against
        the 7 its footprint band admits.

        Falls back to `_plot_range`'s `lo` where no envelope can answer -- a missing module
        must not become a larger claim than anything measured.
        
    """
    if not tname or not decl:
        return 0
    if str(tname) in _LANDMARK_LEAST:
        return int(_LANDMARK_LEAST[str(tname)])
    try:
        lo, _hi, _ex = _plot_range(decl)
    except (KeyError, TypeError, ValueError):
        lo = 3
    got = int(lo)
    try:
        from . import envelope as _env
        feats = tuple(sorted(_env.declared_features(str(tname))))
        if feats:
            ans = _env.lot_for(str(tname), {}, features=feats) or {}
            lm = ans.get("lot_min")
            if lm and len(lm) == 2:
                got = max(got, int(max(lm[0], lm[1])))
    except Exception:                        # noqa: BLE001 -- no envelope: the band stands
        pass
    _LANDMARK_LEAST[str(tname)] = got
    return got


def reservations_of(district: dict | None, ch: dict, decls: dict, *,
                    lot: tuple = (0, 0), side: int = 0) -> list:
    """**The usable space this district must claim before it lays a house**, in order.

        One row per landmark the character names, each saying what the reservation is for,
        the requirement that makes it required (`requirement_for`), the least side that
        delivers its features (`landmark_least`) and the side it would like. The row is what
        `_compile_once` sizes its block grid against and what a shortfall is reported as.

        Nothing here places anything; it says what has to fit.
        
    """
    w, ld = int(lot[0] or 0), int(lot[1] or 0)
    out = []
    for lm in list(ch.get("landmarks") or []):
        tname = lm.get("type")
        decl = (decls or {}).get(tname)
        found = "the district's own table"
        if decl is None and tname:
            _t, every_t = _whole_table()
            decl = every_t.get(tname)
            found = "the whole type table" if decl is not None else "nowhere"
        rid = requirement_for(tname, decl, district)
        least = landmark_least(tname, decl) if decl is not None else 0
        kind = str((decl or {}).get("kind", "plot"))
        want = least
        if decl is not None:
            try:
                lo, hi, _ex = _plot_range(decl)
            except (KeyError, TypeError, ValueError):
                lo, hi = 3, 48
            least = max(int(least), int(lo))
            if kind == "area":
                want = max(least, min(int(hi), LANDMARK_AREA_SIDE))
            else:
                cap = min(int(hi), LANDMARK_SIDES * max(1, int(side)))
                if str((decl or {}).get("role") or "") != "civic":
                    cap = min(cap, max(w, ld) + 2)
                want = max(least, min(int(hi), cap))
        out.append({"subject": tname, "what": "landmark", "kind": kind,
                    "required": rid is not None, "requirement": rid,
                    "found": found, "least": int(least), "want": int(want),
                    "notes": lm.get("notes")})
    return out


def type_use(tname: str | None, decl: dict | None) -> tuple:
    """**What one building type is for**, as a single use word, and why.

        One word and not a set, deliberately. A shop-house is a dwelling over a shop and both
        are true of it; what a *quarter's programme* needs to know is which of the two puts
        it on a street, and for a building that declares the trades it houses that is the
        trade. The second reading is not lost -- it is in the reason string.

        Returns `(use, why)`. `unstated` is an honest answer and the caller decides what to do
        with it (`use_mix` counts an unstated type of the district's own role as the quarter's
        own fabric, and says on the record that it inferred that from the role).
        
    """
    d = decl or {}
    fn = str(d.get("function") or "").strip().lower()
    if TRADE_PARAM in (d.get("params") or {}):
        return ("trade",
                f"`{tname}` declares a `{TRADE_PARAM}` parameter -- the trades it houses"
                + (f" -- over its `FUNCTION = {fn}`" if fn else ""))
    if fn:
        return (FUNCTION_USE.get(fn, fn), f"`{tname}` declares `FUNCTION = {fn}`")
    if str(d.get("role") or "") == "civic":
        return ("civic", f"`{tname}` declares no function and its `ROLE` is `civic`")
    return (UNSTATED_USE, f"`{tname}` declares no function")


def _tokens(text) -> list:
    """The words of a piece of the design's own prose, lowercased."""
    return [w for w in re.split(r"[^a-z0-9]+", str(text or "").lower()) if w]


def _same_word(tok: str, word: str) -> bool:
    """Is `tok` the design's way of saying `word`? Plurals and the possessive, and a
    prefix where the declared word is long enough for one to mean anything.

    `traders` says `trade`; `houses` says `house`; `marke` says nothing."""
    if tok == word:
        return True
    if tok in (word + "s", word + "es", word + "'s"):
        return True
    return len(word) >= 5 and tok.startswith(word) and len(tok) - len(word) <= 3


def design_words(district: dict | None, ch: dict, part: dict | None = None) -> list:
    """**What the design says this quarter is**, in its own words, each with its source.

        `[(text, where)]`. The design's description of a neighbourhood is the only place the
        sources' reading survives by the time a rectangle reaches this compiler: the
        interpretation wrote the defining part's `notes` ("The traders' and craftsmen's ring,
        and the one that has to hold a working market"), the layout copied them onto the
        district as its `purpose`, and the character's landmark carries its own note. Nothing
        here is a place name and nothing here is a branch on one -- it is the district's own
        text, read against what the library's types declare (`named_types`).

        Deliberately **not** `demand.why`, which recites the whole approved pool ("built from
        ['court_large', 'court_small', ...]") and would make every type in the pool a type the
        design named.
        
    """
    out = []
    for key in ("purpose", "notes"):
        v = (district or {}).get(key)
        if v:
            out.append((str(v), f"the district's own `{key}`"))
    if part is not None and part.get("notes"):
        out.append((str(part["notes"]), f"the defining part `{part.get('name')}`'s notes"))
    for lm in list((ch or {}).get("landmarks") or []):
        if lm.get("notes"):
            out.append((str(lm["notes"]),
                        f"the note on the `{lm.get('type')}` the character names"))
    scope = ((district or {}).get("demand") or {}).get("scope")
    if scope:
        out.append((str(scope), "the scope the resolved demand carries"))
    return out


def named_types(text_rows: list, admitted: list) -> dict:
    """**Which of these types the design's own words name**: `{type: [phrase, ...]}`.

        A type is named where the design's prose contains

          * its **whole name** as consecutive words -- "courtyard houses" names
            `courtyard_house`, "row houses" names `row_house`, "market" names `market`;
          * a word it declares -- its `FUNCTION`, its `FAMILY`, or `trade` where it declares
            the trades it houses, so "the traders' and craftsmen's ring" names `shop_house`.

        **Not a segment of a name on its own**, and that restriction is the whole reason this
        is trustworthy: "house" alone names three of this library's five urban types, one of
        which is a shop, so an elite ring described as "courtyard houses on generous lots"
        would acquire a street of shops from the word `houses`. Measured on the retained
        city: with segment matching the upper ring names `trade`; without it, only the middle
        ring does, which is what the sources say.
        
    """
    toks = [(w, where) for text, where in (text_rows or []) for w in _tokens(text)]
    out: dict = {}

    def hit(name, phrase):
        out.setdefault(str(name), [])
        if phrase not in out[str(name)]:
            out[str(name)].append(phrase)

    for name, decl in admitted or []:
        segs = [s for s in str(name).lower().split("_") if s]
        # the whole name as consecutive words
        if segs:
            n = len(segs)
            for i in range(len(toks) - n + 1):
                if all(_same_word(toks[i + k][0], segs[k]) if k == n - 1
                       else toks[i + k][0] == segs[k] for k in range(n)):
                    hit(name, " ".join(t for t, _w in toks[i:i + n])
                        + f" ({toks[i][1]})")
        # the words the type itself declares
        words = {str((decl or {}).get("function") or "").lower(),
                 str((decl or {}).get("family") or "").lower()}
        if TRADE_PARAM in ((decl or {}).get("params") or {}):
            words.add(TRADE_PARAM)
        for w in {x for x in words if len(x) >= 4}:
            for tok, where in toks:
                if _same_word(tok, w):
                    hit(name, f"{tok} ({where})")
    return out


def _required_count(wants: dict | None) -> int:
    """How many of a thing a requirement asks for: its own number where it carries one,
    and otherwise one. A requirement is a statement that the place must hold the thing;
    it is not a statement that a share of the fabric must be it."""
    for key in ("count", "n", "structures"):
        v = (wants or {}).get(key)
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            pass
    return 1


def use_mix(district: dict | None, ch: dict, houses: list, role: str | None, *,
            part: dict | None = None, decls: dict | None = None) -> dict:
    """**This quarter's programme: which uses it holds, which types answer them, and
        where each one stands** -- inferred from this district's own design.

        The composition round put a use filter here, and the neighbourhood round's brief is
        what was wrong with it: "The current `use_mix` filter compares coarse `ROLE` values,
        imposes a fixed 10% secondary-use share, and falls back to other uses when none
        match. It removes civic filler but is insufficient as a general programme."

        All three are answered, and none of them by a constant:

          **1. The use is the type's own declaration, not its `ROLE`.** `type_use` reads
          `FUNCTION`, the `trade` parameter and -- only where a type declares neither -- the
          role, and says which of the three it read. A quarter of `urban` types is not one
          use: `shop_house` is a place of work, `row_house` and `courtyard_house` are homes,
          and `court_small` and `court_large` state nothing and are counted with the quarter's
          own fabric *by inference from their role*, which the record says.

          **2. What the quarter is for comes off its own design.** Two sources, both the
          district's:

            * the **requirements its demand resolved** (`requirement_for`): `function/market`
              scoped to the middle ring is the request asking for a market there, and a type
              that answers a requirement is laid **at the count the requirement asks** -- one,
              unless it says otherwise -- rather than at a share of the street;
            * the **words of its own description** (`design_words`, `named_types`): the
              middle ring's "traders' and craftsmen's ring... with the market laid as an open
              floor among the houses" names `trade` and `market` and `courtyard_house`, the
              lower ring's "attached row houses on a short block" names `row_house` and
              nothing else, and the upper ring's "courtyard houses on generous lots" names no
              work at all. That is the sources' own contrast arriving here as a programme.

          **3. There is no share.** Where a use other than the quarter's own is asked for,
          *where it stands* is a spatial decision taken by the compiler -- the street the
          quarter's anchor fronts (`_compile_once`'s principal face) -- and the resulting
          share is **measured off what was laid** and written on the record with its
          derivation. A quarter with a market gets a street of shops on the market's own
          street because that is where trade goes, and the number that comes out is a
          consequence rather than a target.

        Returns the split the compiler builds from -- `own` (the quarter's own fabric),
        `programme` (the other uses its design asks for, with how each was asked), `unasked`
        (in the pool, asked for by nothing, not drawn) -- plus `uses`, the whole inferred
        programme with its derivation, which is what the plan file and the record carry.

        A district whose pool holds only its own use -- the lower ring's single `row_house` --
        behaves exactly as it did.
        
    """
    want = str(role or "")
    admitted = list(houses or [])
    # the character's landmarks are part of this quarter's programme whatever table they
    # came from, so they are in the lexicon even though they are not fabric
    lm_types = []
    for lm in list((ch or {}).get("landmarks") or []):
        tn = lm.get("type")
        if not tn:
            continue
        dl = (decls or {}).get(tn)
        if dl is None:
            with contextlib.suppress(Exception):
                _t, every_t = _whole_table()
                dl = every_t.get(tn)
        if dl is not None:
            lm_types.append((str(tn), dl))
    uses_of = {str(n): type_use(n, d) for n, d in admitted}
    for n, d in lm_types:
        uses_of.setdefault(n, type_use(n, d))

    # --- what the quarter's own fabric is ------------------------------------------
    # The primary use is the use of the district's **own role's dwellings** where the
    # role has any, and otherwise the commonest use its own-role types declare. It is
    # not a word this module chose: a residential district's fabric is dwellings because
    # dwelling types are what its role admits.
    mine = [(n, d) for n, d in admitted
            if not want or str(d.get("role") or "") == want]
    kinds: dict = {}
    for n, _d in mine:
        kinds[uses_of[n][0]] = kinds.get(uses_of[n][0], 0) + 1
    if kinds.get("home"):
        primary, primary_why = "home", (
            f"{kinds['home']} of the {len(mine)} type(s) this district's `{want}` role "
            f"admits declare `FUNCTION = dwelling`, so the quarter's own fabric is homes")
    elif kinds:
        primary = max(sorted(kinds), key=lambda k: (kinds[k], k != UNSTATED_USE))
        primary_why = (f"no type of this district's `{want}` role declares a dwelling; "
                       f"the commonest use its own types declare is `{primary}`")
    else:
        primary, primary_why = UNSTATED_USE, "this district's role admits no type at all"

    # --- what else its own design asks for -----------------------------------------
    words = design_words(district, ch, part)
    named = named_types(words, admitted + lm_types)
    required: dict = {}
    for name, decl in admitted + lm_types:
        rid = requirement_for(name, decl, district)
        if rid:
            wants = next((w for _k, w, r in _requirements_of(district) if r == rid), {})
            required[str(name)] = (rid, _required_count(wants))

    own, programme, unasked, why = [], [], [], []
    why.append(primary_why)
    for name, decl in admitted:
        use, use_why = uses_of[name]
        if use == primary or (use == UNSTATED_USE
                              and (not want or str(decl.get("role") or "") == want)):
            own.append((name, decl))
            if use == UNSTATED_USE:
                why.append(f"{use_why}; its `ROLE` is this district's own, so it is "
                           f"counted with the quarter's `{primary}` fabric -- an "
                           f"inference from the role and not a declaration")
            continue
        if name in required:
            rid, n_req = required[name]
            programme.append((name, decl))
            why.append(f"{use_why}, and answers the requirement `{rid}` this district "
                       f"resolved: {n_req} of them, which is what the requirement asks "
                       f"for and not a share of the fabric")
        elif name in named:
            programme.append((name, decl))
            why.append(f"{use_why}, and this district's own design names it: "
                       f"{'; '.join(named[name][:2])}")
        else:
            unasked.append((name, decl))
    if unasked:
        why.append(f"{', '.join(n for n, _d in unasked)} "
                   f"({', '.join(sorted({uses_of[n][0] for n, _d in unasked}))}) are in "
                   f"the approved pool and nothing in this district's programme asks for "
                   f"them -- no requirement it resolved and no word of its own "
                   f"description -- so the fabric is not drawn from them")
    # **A word of the design that no type answers is a programme shortfall, recorded.**
    # "craftsmen" and "schoolteachers" name work this library has no type for; that is a
    # true fact about the library and it belongs on the record rather than in silence.
    if not own and admitted:
        # a pool with nothing of the quarter's own use: a capability question and not
        # the compiler's to answer. It lays what it was given, ranked by the pool's own
        # order, and the record says the mix is the pool's and not the programme's.
        own, programme, unasked = list(admitted), [], []
        why.append(f"no type of this pool answers this district's `{primary}` fabric; "
                   f"the fabric is laid from what the pool has, in the pool's own order, "
                   f"and this mix is the pool's rather than the programme's")
    # **A quarter is built of what its own design names, before what the pool happens to
    # rank first.** The neighbourhood round, found by building the section twice. The
    # compiler takes the head of `own` as the district's house, and `own`'s order is the
    # approved pool's -- which `arrange`'s fabric rung sorts **smallest footprint
    # first**, a rule whose purpose is *recovery* for a district short of its count and
    # not a statement about what the quarter is. So when the generator stream corrected
    # `court_large`'s declared floor from 6x6 to 9x9 and `court_small`'s from 4x4 to 8x9
    # -- honest corrections; at the old floors the types raised `empty range in randint`
    # and published 2x2 light wells as courtyards -- both slid to the back of the pool
    # behind `row_house`, and the traders' ring of a city whose sources say in as many
    # words that "its houses are courtyard houses" came out as thirty row houses.
    # `named_types` already reads the district's own `purpose` and notes against what
    # the library's types declare. Where the design names a type of the quarter's own
    # use, it leads the quarter's fabric; the rest keep the pool's order behind it.
    # Nothing here names a type, a role or a place: the words are the district's and the
    # match is the library's own declarations.
    lead = [x for x in own if x[0] in named]
    if lead and len(lead) < len(own):
        own = lead + [x for x in own if x[0] not in named]
        why.append(f"this district's own design names {', '.join(n for n, _d in lead)} "
                   f"({'; '.join(named[lead[0][0]][:2])}), so the quarter's fabric leads "
                   f"with it rather than with whatever the approved pool ranks first -- "
                   f"that order is `arrange`'s smallest-footprint recovery rung and is "
                   f"not a statement about what this quarter is")

    uses = [{"use": primary, "role": "fabric", "types": [n for n, _d in own],
             "asked_by": ["the district's own role and the types it admits"],
             "from": primary_why}]
    for name, _d in programme:
        use, use_why = uses_of[name]
        row = {"use": use, "role": "programme", "types": [name], "from": use_why}
        if name in required:
            rid, n_req = required[name]
            row.update({"asked_by": [f"the requirement `{rid}`"], "requirement": rid,
                        "count": int(n_req), "where": "the quarter's principal street"})
        else:
            row.update({"asked_by": named.get(name, []), "count": None,
                        "where": "the quarter's principal street"})
        uses.append(row)
    for name, _d in lm_types:
        use, use_why = uses_of[name]
        rid = required.get(str(name), (None, 1))[0]
        uses.append({"use": use, "role": "anchor", "types": [str(name)],
                     "asked_by": ([f"the requirement `{rid}`"] if rid else [])
                                 + named.get(str(name), [])
                                 + ["the landmark the character names"],
                     "requirement": rid, "count": 1,
                     "where": "reserved before the housing takes the ground",
                     "from": use_why})
    return {"own": own, "programme": programme, "unasked": unasked,
            "role": want, "primary": primary,
            "use_of": {n: u for n, (u, _w) in uses_of.items()},
            "required": {k: {"requirement": v[0], "count": v[1]}
                         for k, v in required.items()},
            "named": named, "uses": uses,
            "words": [{"text": t[:240], "from": w} for t, w in words],
            # **no share is imposed.** The key stays because every consumer of this
            # record reads it; it is `None` until `_compile_once` measures what was
            # laid.
            "share": None, "from": why}


def _whole_table() -> tuple:
    from .placeplan import types_card
    return types_card()


def _reserve_alternatives(r: dict, have_u: int, have_v: int, block: int, bd: int,
                          frame_u: int, frame_v: int, want_lots: int) -> list:
    """**What would make this reservation fit**, in the caller's own terms.

        The round's rule for an unmet demand: "Return the unmet demand to the owner with
        feasible alternatives." Each row names an owner's own lever and the number it would
        have to reach -- not a suggestion to try harder. `arrange` and `placeplan` carry
        these up to whoever can pull one.
        
    """
    from .placeplan import PLOT_LANE
    need = int(r.get("least") or 0)
    out = []
    # **the block the reservation was actually given**, where one was, and the block the
    # grid was cut to otherwise: the grid is cut to hold a required reservation
    # (`reserve_drove`) and the *runs* can still come back shorter than the block asked
    # for, which is the case this row is about
    have = ([int(have_u), int(have_v)] if (have_u and have_v)
            else [int(block), int(bd)])
    if need and min(have) < need:
        out.append({"what": "block", "owner": "layout",
                    "field": "arrangement.block", "need": int(need),
                    "have": have,
                    "why": (f"a block of at least {need}x{need} holds the "
                            f"{r['subject']}; the block it was given is "
                            f"{have[0]}x{have[1]} and this district's grid was cut to "
                            f"{block}x{bd}")})
    least_frame = need + 2 * EDGE_MARGIN + PLOT_LANE
    if need and (frame_u < least_frame or frame_v < least_frame):
        out.append({"what": "extent", "owner": "layout",
                    "field": "district rectangle", "need": int(least_frame),
                    "have": [int(frame_u), int(frame_v)],
                    "why": (f"a district of at least {least_frame} columns each way "
                            f"holds a {need}-column {r['subject']} with its street and "
                            f"its edge margins; this one is {frame_u}x{frame_v}")})
    if want_lots > 0:
        out.append({"what": "count", "owner": "scale",
                    "field": "structures", "need": None, "have": int(want_lots),
                    "why": (f"this district was asked for {want_lots} house(s); the "
                            f"reservation and the housing are competing for the same "
                            f"ground and the count is the other side of that trade")})
    if have_u or have_v:
        out.append({"what": "measured", "owner": None, "field": None,
                    "need": int(need), "have": [int(have_u), int(have_v)],
                    "why": (f"the block the reservation was given is "
                            f"{have_u}x{have_v} and it needs {need}x{need}")})
    return out


def _open_type(areas: dict, role: str | None, density: str,
               use: str = "settled") -> str | None:
    """What an open block is: fields where the ground is farmland, a plaza for a dense
        district, gardens and groves for the rest -- the first the district's table has.

        **Fields because the ground is farmland, not because the houses are rural.** A
        fishing village and a farm belt are both `rural` and only one of them is fields;
        that conflation is the review's fourth finding and this is the other half of it.
        
    """
    if use == "farmland":
        order = ("field", "grove", "garden")
    elif use in ("orchard", "wood", "woods", "forest"):
        # an orchard is trees in rows: the grove is the library's tree ground
        order = ("grove", "field", "garden")
    elif use == "pasture":
        order = ("field", "grove", "garden")
    elif use == "garden":
        order = ("garden", "grove", "yard")
    elif density == "dense":
        order = ("plaza", "garden", "yard")
    else:
        order = ("garden", "grove", "plaza")
    return next((t for t in order if t in areas), next(iter(areas), None))


def _court_type(areas: dict, role: str | None) -> str | None:
    order = ("yard", "garden", "plaza") if role == "urban" else ("garden", "yard", "plaza")
    return next((t for t in order if t in areas), next(iter(areas), None))


#: **What a court a block's own ranges enclose is made of.** The neighbourhood delivery
#: round, found by building the section and asking the court predicate about it.
#: `_court_type` answers "what is the open ground behind a row of houses", and for an
#: urban role that is a `yard` -- a working yard, which fences its own perimeter and
#: stands barrels in it. That is right for the ground behind a row and wrong for a court
#: four ranges already enclose: the enclosure is the buildings', and the yard's fence
#: closes the space a second time. Measured on this section's two composed courts, both
#: of which stood: `lower_ring_north_2_pc0_0_0` is an 8x5 tile with a spruce fence round
#: all of it and barrels inside, and `usable.court_accessible` answered *"is no longer
#: open paved ground in the assembled world"* -- correctly, of a court that is mostly
#: fence. A court is paved open ground with the sky over it, so the order is the paved
#: types first and the yard last, and the record says which was laid.
ENCLOSED_COURT_TYPES = ("plaza", "square", "garden", "yard")


def _enclosed_court_type(areas: dict, role: str | None) -> str | None:
    return next((t for t in ENCLOSED_COURT_TYPES if t in areas),
                _court_type(areas, role))


def _verge_type(areas: dict, role: str | None, use: str = "settled") -> str | None:
    if use in ("farmland", "pasture"):
        order = ("field", "grove", "garden")
    elif use in ("orchard", "wood", "woods", "forest", "garden"):
        order = ("grove", "garden", "field")
    else:
        order = ("grove", "garden", "field")
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


#: How far outside a district's frame a doorstep may be from the arterial that bounds it
#: and still be entered from it: the verge a lane crosses to the road.
ROAD_VERGE = 3


def _runs(start: int, end: int, street: int, block: int, least: int,
          lead: bool = True, widen: bool = True, trail: bool = False) -> list:
    """[(from, to)] of the blocks along one axis inside [start, end]: a street, a
    block, a street... as many blocks of `block` as fit; a remainder of at least
    `least` after a street is one more, shorter block, and a smaller remainder
    **widens the blocks evenly**, so no strip at the end of a run is nobody's. A run
    shorter than a block is one block, down to `least`. `lead` false starts with a
    block: the interval begins at a street that is already there (the arterial's
    band)."""
    n = end - start + 1
    # `trail`: the run also ends with a street, for a block every face of which is a
    # frontage (a perimeter block): its last block's far face is entered from a street
    # of its own and not from the district's edge
    room = n - (street if lead else 0) - (street if trail else 0)
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


#: An arterial's band is a **street of the grid** on the lines it *covers* for at least
#: this share of the axis it runs along, and where no more than `AXIAL_DEPTH` lanes'
#: worth of lines are covered: the blocks are laid on either side of it. A road that
#: crosses the district any other way covers no line of it and is a band the lots keep
#: off. **The width test earns its place.** Dropping it -- on the argument that a road
#: is a road whatever its width, and the blocks should go beside it -- splits both axes
#: at every band and fragments the grid into pieces too short for a block: a district
#: that laid 30 lots laid 6. A wide swathe of road is an obstacle, not a street.
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
               least: int, lead: bool = True, widen: bool = True,
               trail: bool = False) -> tuple:
    """The block runs along one axis (0: u, 1: v), the arterial's band a street of
    the grid where it is axial, else a lead street at the district's edge -- unless
    `lead` is off: a strip too thin for a street and a lot fronts its own edge, where
    the ground beyond it (a road, the next district's street) is the way in.
    Returns (runs, axial)."""
    # **A line is a street of this grid where the road covers it, and a road crossing it
    # is not one.** The block design round, found by asking why every block of
    # `lower_ring_north_2` had a road through its back range. `across = band.any(axis)`
    # is True for a line the band *touches*, and a district with one road along it and
    # two roads across it has every line touched: 65 of 65 here. `across.sum()` was then
    # 65 against a bar of 15, the band was declared not-axial, and `_runs` laid the grid
    # over the whole rectangle **ignoring the road entirely** -- so a block row ended
    # two columns inside the carriageway, the back range of every block in the district
    # was refused for the arterial, 19 lots were dropped and no court could be composed
    # anywhere in the crowded ring. The question is not whether a line is touched but
    # whether it is *covered*: a street running along this axis fills its own lines and
    # puts a few columns in everyone else's. The width test the comment above earns
    # stays, over the covered lines only, so a wide swathe of road is still an obstacle
    # and not a street.
    cover = band.mean(axis=1 - axis) if band.size else np.zeros(total, dtype=float)
    across = cover >= AXIAL_SHARE
    axial = bool(across.any()) and int(across.sum()) <= AXIAL_DEPTH * street
    if not axial:
        return _runs(0, total - 1, street, block, least, lead=lead, widen=widen,
                     trail=trail), False
    out = []
    parts = _intervals(across)
    for k, (a, b) in enumerate(parts):
        out += _runs(a, b, street, block, least, lead=(a == 0 and lead), widen=widen,
                     trail=trail and b == total - 1)
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


def _feasible_core(cum, U: int, V: int, bar: float,
                   least_u: int, least_v: int) -> tuple | None:
    """**The part of a rectangle a grid can actually be laid over**, or None.

        The neighbourhood delivery round. `_largest_free` answers "the largest rectangle
        every column of which is feasible", which is the right question for a standing part
        and the wrong one here: on `middle_ring_north_west`'s mesa the largest all-true
        rectangle is **391 columns of the 3,227** that can carry a building, because the
        feasible set is a scatter and not a shelf. A district's grid does not need every
        column -- `GROUND_FOUNDED` is the share a *lot* is held to -- it needs a rectangle
        most of which is buildable.

        Found by peeling: the border line (of the four) with the least feasible ground is
        dropped, and again, until the rectangle as a whole is over the bar or it has shrunk
        to the least block this fabric needs. Four O(1) lookups a step off the integral
        image, so this costs nothing next to a compile.
        
    """
    def good(u0, u1, v0, v1) -> int:
        return int(cum[u1 + 1, v1 + 1] - cum[u0, v1 + 1]
                   - cum[u1 + 1, v0] + cum[u0, v0])

    def share(u0, u1, v0, v1) -> float:
        n = (u1 - u0 + 1) * (v1 - v0 + 1)
        return good(u0, u1, v0, v1) / float(n) if n else 0.0

    u0, u1, v0, v1 = 0, int(U) - 1, 0, int(V) - 1
    for _ in range(int(U) + int(V)):
        if share(u0, u1, v0, v1) >= bar:
            return (u0, u1, v0, v1)
        cands = []
        if u1 - u0 + 1 > least_u:
            cands.append((share(u0, u0, v0, v1), "u0"))
            cands.append((share(u1, u1, v0, v1), "u1"))
        if v1 - v0 + 1 > least_v:
            cands.append((share(u0, u1, v0, v0), "v0"))
            cands.append((share(u0, u1, v1, v1), "v1"))
        if not cands:
            return None
        _s, which = min(cands)
        if which == "u0":
            u0 += 1
        elif which == "u1":
            u1 -= 1
        elif which == "v0":
            v0 += 1
        else:
            v1 -= 1
    return (u0, u1, v0, v1) if share(u0, u1, v0, v1) >= bar else None


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

#: **The court share a district's adopted form keeps, whatever the count costs.** The
#: neighbourhood delivery round, and it is the other half of the audit's fourth cause.
#: Lever 2 below spends the open share and then the *court* share back into housing when
#: a district is short of the count its density asks for, and it spends them to
#: **zero**. The composition round already called that out -- "courtyard/open shares can
#: be spent to fit housing" -- and answered it with a record (`shares_spent`) rather
#: than a floor, so the crowded ring's registered 0.25 was spent to 0.05 on the city and
#: to 0.0 on the compile fixture, `demand.court_obligation` read the remainder, and
#: `round(0.05 * 8)` laid no court at all. A fabric whose form is courts around which
#: houses stand is not the same fabric with the courts removed. The floor is
#: deliberately small -- one court block in a district of a dozen -- and it costs much
#: less than it used to, because a court block is now *composed* rather than subtracted:
#: `compose_court_block` lays a range of houses on all four sides of the court, where
#: the old back-row `courtyard` block gave the whole back row away. What is still
#: negotiable is how many courts; that there is one is what the form said.
COURT_SHARE_KEEP = 0.05

#: How much shallower than the density's own lot the compiler will make a lot, a column
#: at a time, when the count the district was asked for is under its floor: a deep block
#: whose middle nobody owns is two more rows of slightly shallower lots.
DEPTH_GIVE = 4

#: ...and how much **bigger** than the lot the ask implies the compiler will make one, a
#: column at a time, where the houses it is allowed to lay do not cover the ground its
#: density asks to be built on. A district asked for three houses on seven thousand
#: columns covers its share with three big plots or not at all.
LOT_GROW_MAX = 10

#: **When a district's grid is laid over its buildable core rather than its rectangle.**
#: The neighbourhood delivery round. Two bars, both registered before the numbers that
#: test them: * the rectangle has to be **mostly refused** -- a district that can build
#: on most of itself keeps its whole rectangle, because moving a grid off a corner costs
#: streets and frontage and buys nothing; * the core has to be **most of what is
#: feasible** -- a mask that refuses half a district in a checkerboard has no core to
#: move to, and shrinking the grid onto one patch of it would throw away the other half.
#: `lower_ring_south_1` is 16% feasible and coring it took it from 9 lots to **15**. So
#: the bar is "mostly refused" and not "not entirely buildable".
GROUND_CORE_WHOLE = 0.5
GROUND_CORE_LEAST = 0.45

#: The shares a core is tried at, tightest first. A core is only worth moving a grid
#: onto if the grid can then lay lots on it, and `GROUND_FOUNDED` is the share one *lot*
#: is held to -- a rectangle at exactly that share loses about half its lots to the
#: ground again. The tightest bar that still keeps `GROUND_CORE_LEAST` of the district's
#: buildable ground is the one taken.
GROUND_CORE_BARS = (0.9, 0.8, 0.7, 0.6, GROUND_FOUNDED)

#: How far outside a lot the ground it is entered from reaches: the column a doorstep is
#: reserved on and the one a platform's ledge is laid over, which are the same column.
SKIRT = 1

#: **How many columns of a lot's own front the design has to prepare for it to have a
#: way in.** The block design round. `GROUND_FOUNDED` is a *share* over a rectangle, and
#: a share cannot say that a door works: the audit's sentence is that "the critical
#: cells can lie in the rejected half of both rectangles". This is the hard condition
#: that goes beside it -- a run of named columns on the lot's street face, prepared
#: outside for the doorstep and inside for the floor it opens onto. Two, because a
#: person steps onto the doorstep and then through the door, and one column of luck is
#: not an entrance.
ENTRY_RUN = 2

#: **How far inside its plot the pad a building is handed begins.**
#: `buildlib.SITE_INSET`, the clearance `site()` leaves round a pad on a free side. A
#: plot's pad is what `build()` is given and what a type's declared footprint is
#: measured against, so it is the rectangle whose ground has to be ground this design
#: prepares. See `pad_founded`.
PAD_INSET = 2

#: **How far a lot may be deepened to carry a height its own band asks for.** The
#: neighbourhood delivery round. A density word says how much ground a house gets and a
#: storey band says how tall the fabric is; where the lot the first resolves admits only
#: the floor of the second, the two disagree by a column or two and the resolution is
#: the type's own envelope. Four columns, because beyond that the lot is no longer the
#: lot the density asked for and the honest answer is that this density does not carry
#: that band.
STOREY_DEPTH_REACH = 4

#: A landmark's plot is at most this many of the density's lot sides across: a hall at
#: its largest admitted plot took a third of a small district for one building.
LANDMARK_SIDES = 2

#: **The side an area landmark is drawn at where its block holds one.** A decision and
#: not a general rule, recorded as one: the number is the market type's own two-row
#: arithmetic -- `2 x STALL_DEPTH + AISLE + 2` pad columns across, which is 13, plus the
#: `2 x Builder.SITE_INSET` the plot carries -- so a landmark drawn at this side is a
#: market with a row of stalls either side of its aisle rather than one row or a paved
#: rectangle. It was written into the area branch as a bare `17` with no derivation and,
#: worse, as a **ceiling**: `max(lo, min(hi, 17, block_w, block_d))` clamps the side to
#: the block and the clamp is then tested against the block, so a block narrower than
#: `lo` silently defeated the fit and the leaf fell through to `kind = "row"`. It is now
#: the *preferred* side, under `landmark_least` as the floor, and where the demand
#: **requires** the landmark the block grid is sized to hold it (`_compile_once`).
LANDMARK_AREA_SIDE = 17

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
    ch0 = character_of(part, district)
    t = district_target(district, part, place, decls)
    tries = []
    ch = dict(ch0, _lead=True)

    # **The cover the validator asks of this district**, which for a rural one is
    # `RURAL_COVER` over its whole rectangle and for every other the density's own share
    # of it: the number the compiler works to and the number it is refused on are one
    # number.
    from .placeplan import RURAL_COVER
    # **The farmland cover is the farmland's, here as well as in the validator.** The
    # review's fourth finding reached this objective too, and it is the more damaging
    # place for it: keyed on `role == "rural"`, a fishing village's district was scored
    # against a 60% ground-cover target it had never been asked to meet, and the search
    # therefore preferred a one-lot configuration that met it over a two-lot one that
    # did not -- and the validator then refused the district for having too few houses.
    # The compiler chose the arrangement its own validator rejects. Found by running the
    # held-out village; the two now key on the same fact.
    want_ground = max(t["min_ground_columns"],
                      int(math.ceil(RURAL_COVER * t["usable_columns"]))
                      if spec_mod.land_use(part) == "farmland" else 0)

    def clauses(rec):
        """**The clauses the search chooses between, by name.** Returned as a dict because
        this tuple has been mis-sliced once already at real cost (see the note in the
        loop below), and because the composition round adds a clause *before* the count
        -- which would have shifted every index the loop reads.
        """
        plots = rec["plot_cover"] * rec["columns"]
        ground = rec["ground_cover"] * rec["columns"]
        # **A district with houses beats one without**, the craft round, found by
        # running it: a strip gave its lead street up, the single row of blocks that
        # left landed on the arterial's band, every lot was dropped -- and that try was
        # the one kept, because open ground covers more than nothing. Where the count
        # floor is missed, the number of houses is what is being chosen between.
        # **...and the number of houses outranks the ground clause while the count is
        # short.** Found by running the held-out village: an 84x35 strip had a two-lot
        # arrangement covering 40% of its ground and a one-lot arrangement covering 60%,
        # and this kept the one-lot one -- because `ground >= want` came before the
        # count and only the second arrangement met it. The validator then refused the
        # district for drawing one house where three is the floor. The compiler was
        # choosing, on purpose and by this line, the arrangement its own validator
        # rejects; a district is not refused for having less grass than it might have,
        # and it is refused for having too few houses. **...and no more lots than the
        # density's ceiling admits** (the closure round): a density word is a band,
        # `district_target` carries both ends of it, and an arrangement over the ceiling
        # is refused by the validator's own `cover_over` clause, so the search prefers
        # one under it. An exact count is the count whatever the ceiling says, and where
        # the two cannot both be met the record says so (`over_ceiling`) and the layout
        # owner gets the finding. **...and an arrangement that keeps the required
        # reservation beats one that loses it, before anything else.** The composition
        # round's first change: this tuple had no landmark term at all, so a retry that
        # dropped the market and laid two more houses on its block ranked *above* one
        # that kept it -- the search was selecting, by this line, the arrangement that
        # loses the thing the request requires. It ranks first because a required
        # reservation is the sentence's own and the count of these districts is
        # inferred; it is the reservation the **demand requires** and never one the
        # character merely prefers, so a district with no requirement on it scores
        # exactly as it did.
        ceiling = t.get("max_plot_columns")
        ok_ceiling = ceiling is None or plots <= ceiling
        return {
            "reserved": not [x for x in (rec.get("demand_short") or [])
                             if x.get("required")],
            "lots": rec["lots"] >= t["min_count"],
            "plots": plots >= t["min_plot_columns"],
            "count": min(rec["lots"], t["min_count"]),
            "ground": ground >= want_ground,
            "ceiling": ok_ceiling,
            "ground_cover": rec["ground_cover"], "plot_cover": rec["plot_cover"]}

    #: The order the clauses are compared in. Named here so that adding one is a change
    #: to this list and not a silent renumbering of the loop below.
    ORDER = ("reserved", "lots", "plots", "count", "ground", "ceiling",
             "ground_cover", "plot_cover")

    def score(rec):
        c = clauses(rec)
        return tuple(c[k] for k in ORDER)

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
        if rec.get("composition") is not None:
            # **a street-composed district searched its own proposals** and judged them
            # on their relationships (`ethoslm.streetplan`); the shares and lot levers
            # below belong to the grid and would re-lay the same composition
            best = (got, rec, dict(ch))
            tries.append({"layout": "street", "lots": rec["lots"],
                          "admissible": rec["composition"]["admissible"]})
            break
        tries.append({"courtyard_share": ch["courtyard_share"],
                      "open_share": ch["open_share"], "lead_street": ch["_lead"],
                      "block_lots": ch.get("_block_lots"), "block": rec["block"],
                      "ground_cover": rec["ground_cover"],
                      "plot_cover": rec["plot_cover"], "lots": rec["lots"]})
        if best is None or score(rec) > score(best[1]):
            best = (got, rec, dict(ch))
        # **The three clauses the search stops on, by name.** The review's second
        # finding, and it was an unpacking bug with a real cost: `score` gained a
        # numeric count as its third element and this still read its first three as
        # `ok_lots, ok_plots, ok_ground`, so a positive lot count *was* the ground
        # Boolean. A candidate with 200 ground columns against 800 required read as
        # meeting the ground clause and the search stopped on it without looking at the
        # next candidate that met the bar. Named rather than sliced, so the next field
        # added to the score cannot silently become one of these again -- which is what
        # `clauses` returning a dict now makes structural rather than a matter of care.
        c = clauses(rec)
        ok_lots, ok_plots = c["lots"], c["plots"]
        ok_ground, ok_ceiling = c["ground"], c["ceiling"]
        # **...and the search does not stop on an arrangement that lost a required
        # reservation.** The composition round: ranking it first is only half the fix --
        # the search *stopped* as soon as the count, the cover and the ceiling were met,
        # so the first arrangement that housed the market's ground satisfactorily ended
        # the search and the levers below never ran. Measured on a 200x22 middle-ring
        # band: the lead-street lever (1) turns a 13-deep block into an 18-deep one,
        # which holds the 14-column market, and the search had already stopped one try
        # earlier.
        met = ok_lots and ok_plots and ok_ground and ok_ceiling
        if met and c["reserved"]:
            break
        # **the reservation's own lever, and only one.** A required reservation that did
        # not fit is short of *depth across the block*, and the one lever that changes a
        # block's depth without touching the fabric is the lead street: a district that
        # gives it up starts its grid at a block instead of a street, which is `street`
        # more room across. Tried once. The numbers' levers below -- bigger lots, fewer
        # courts, longer blocks -- are not spent chasing a reservation, because each of
        # them is an answer to a different question and pulling them for this one would
        # trade the fabric the district was asked for against a market that may not fit
        # at any setting. Where the lever does not help, the shortfall is reported.
        if met and ch["_lead"]:
            ch["_lead"] = False
            continue
        if met:
            break
        # 0. over the ceiling with the count met: the lots stand on too much of the
        # ground for the density word. Smaller lots were tried up front
        # (`_compile_once`); what is left is more open ground, which is only right where
        # the count is a proposal and not the sentence's own number.
        if not ok_ceiling and ok_lots and not bool(district.get("exact")) \
                and ch["open_share"] < 1.0:
            ch["open_share"] = min(1.0, round(ch["open_share"] + COVER_STEP, 2))
            continue
        if not ok_ceiling and ok_lots:
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
            # ...and the court share down to the floor its form keeps, never to zero
            # where the character declared one (`COURT_SHARE_KEEP`)
            keep = (COURT_SHARE_KEEP if float(ch0.get("courtyard_share") or 0.0) > 0
                    else 0.0)
            if ch["courtyard_share"] > keep:
                ch["courtyard_share"] = max(keep,
                                            round(ch["courtyard_share"] - COVER_STEP, 2))
                continue
        # 3. the lot, **up**: the count is a number and not a floor, so where the houses
        # the district is allowed to lay do not cover the ground its density asks to be
        # built on, those houses are bigger -- deeper as well as wider, which is why any
        # depth the loop gave away below is given back here.
        if not ok_plots and not ch0.get("lot_depth") and not ch0.get("lot_width") \
                and rec["lots"] > 0 \
                and int(ch.get("_lot_grow") or 0) < LOT_GROW_MAX:
            ch["_lot_grow"] = int(ch.get("_lot_grow") or 0) + 1
            ch["lot_depth"] = None
            continue
        # 4. the lot, **down**: a shallower lot, a column at a time, down to
        # `DEPTH_GIVE` under the density's own, where a deep block's middle becomes two
        # more rows and the district is still short of houses. **A depth the character
        # named is not the compiler's to give away** -- lever 3 has said so since it was
        # written and this said nothing, so under the craft round's count a district
        # that asked for lots seventeen deep got fourteen and the record called it a
        # raise. A lever gives back what the density suggested, never what the character
        # declared.
        if not ok_lots and not ch0.get("lot_depth"):
            deep = _lot_side(ch["density"])
            have = int(ch.get("lot_depth") or rec["lot"][1])
            if have > deep - DEPTH_GIVE and have > 3 \
                    and _depth_keeps_storeys(rec, ch, decls, have - 1):
                ch["lot_depth"] = have - 1
                continue
            break
        # 5. the block: a longer block is fewer streets, and a loose fabric's ground
        # goes to streets before it goes anywhere else
        here = int(ch.get("_block_lots")
                   or block_lots_for(ch["density"], (rec.get("lot") or [0])[0]))
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
    # **A court share spent to meet a lot count is visible.** The composition round, and
    # the review's own words: "courtyard/open shares can be spent to fit housing". Lever
    # 2 above is the lever that does it -- a district short of houses gives its open
    # blocks and then its courts back to the fabric -- and the only trace was a `raised`
    # entry that reads like a raise. The court a character asked for is part of what the
    # fabric *is*; where it was given up to reach the count, the record says which
    # share, from what to what, how many courts are left and why. It is not a refusal:
    # an inferred court share is the compiler's to negotiate. It is a fact the reader
    # was owed.
    spent = {}
    for k in ("courtyard_share", "open_share"):
        was, now = float(ch0.get(k) or 0.0), float(ch.get(k) or 0.0)
        if now < was - 1e-9:
            spent[k] = [round(was, 3), round(now, 3)]
    rec["shares_spent"] = ({
        **spent,
        "courts": int(rec.get("courts") or 0),
        "open": int(rec.get("open") or 0),
        "lots": int(rec.get("lots") or 0),
        "asked_for": int(rec.get("asked_for") or 0),
        "why": (f"this district was short of the {t['min_count']} house(s) or the "
                f"{t['min_plot_columns']} columns of lot its density asks for, so the "
                f"ground its character gave to "
                f"{' and '.join(k.replace('_share', '') for k in spent)} went to housing "
                f"instead: "
                + "; ".join(f"{k} {v[0]:g} -> {v[1]:g}" for k, v in spent.items())
                + f". It holds {int(rec.get('courts') or 0)} court(s) and "
                  f"{int(rec.get('open') or 0)} open block(s)")}
        if spent else None)
    return got, rec


def _depth_keeps_storeys(rec: dict, ch: dict, decls: dict, depth: int) -> bool:
    """**A lever may not take a lot below the depth its own storey band needs.**

        The neighbourhood delivery round, found by the rhythm bar. The cover loop's fourth
        lever gives a lot depth back, a column at a time, when a district is short of houses
        -- and it gave it back past the one column that decides whether the house has a
        stair. Measured on `test_compile`'s dense attached fixture, through the same envelope
        the compiler asks:

            row_house, 6x13, both flanks attached -> 2 storeys
            row_house, 6x12, both flanks attached -> 1

        So the lever bought a few more houses and took the second storey off **every** house
        in the district, which is also why four neighbours of one shape stood in a row: with
        one admitted height there is no second height to alternate to, and the compiler has
        no other lever on an attached terrace whose frontage is fixed by its party walls.
        `_attached_lot`'s own docstring has said since the spatial design round that depth is
        ranked first *because* "a lot one column short of it stands one storey however much
        ground it has"; this is the same rule applied to the lever that undoes it.

        A district whose band asks for one storey is unaffected, and so is one whose type
        declares no envelope: the guard only bites where a storey is actually lost.
        
    """
    band = ch.get("storeys")
    if not band or int(band[-1]) <= 1:
        return True
    name = rec.get("house")
    decl = (decls or {}).get(str(name))
    if not decl:
        return True
    w = int((rec.get("lot") or [0, 0])[0]) or None
    if not w:
        return True
    flanks = 2 if ch.get("attached") else 0
    lo, hi = int(band[0]), int(band[-1])
    try:
        was = _storeys_fit(name, lo, hi, w, int(depth) + 1, flanks=flanks) or {}
        now = _storeys_fit(name, lo, hi, w, int(depth), flanks=flanks) or {}
    except Exception:                        # noqa: BLE001 -- no envelope, no rule
        return True
    a, b = was.get("storeys"), now.get("storeys")
    if a is None or b is None:
        return True
    return int(b) >= int(a)


def _compile_once(district: dict, part: dict, place: dict, decls: dict, ch: dict, *,
                  spec: dict | None = None, seed: int = 1) -> tuple:
    from .placeplan import PLOT_LANE, district_target
    target = district_target(district, part, place, decls)
    density, role = ch["density"], ch.get("role")
    form = (spec or {}).get("form")
    # **The fabric pool the capability record approved for this district**, written onto
    # it by `placesolve.solve_place`. None where no record reached the layout, which is
    # every call that predates the record and every direct call from a test. **The
    # demand this district resolved before it was sized** (the design round's first
    # contract), where the layout wrote one: the approved type pool, the required
    # parameters and features, and the requirement ids that made them required. It is
    # what the envelope questions below carry; a district with none behaves as it did.
    dem = district.get("demand") or None
    houses = house_types(decls, role, form, approved=district.get("fabric_types"))
    areas = area_types(decls, role, form)
    # what lies between the buildings, declared by the part or derived from its own
    # prose; a role is what the buildings are for and never decides this. **The
    # district's own land use first** (the expression round): the open remainder of a
    # ring is the ring's ground and gardens, whatever its houses' part says.
    use = district.get("land_use") or spec_mod.land_use(part)
    if not houses and district.get("fabric_types"):
        raise ValueError(f"{district['name']}: the capability record approved "
                         f"{district['fabric_types']} for this district's fabric and no "
                         f"committed type of the form {form or 'any'} for "
                         f"{role or 'any'} work is among them")
    if not houses:
        raise ValueError(f"{district['name']}: no committed type of the form "
                         f"{form or 'any'} builds a {role or 'district'} house")
    X0, X1 = min(district["x0"], district["x1"]), max(district["x0"], district["x1"])
    Z0, Z1 = min(district["z0"], district["z1"]), max(district["z0"], district["z1"])
    W_all, D_all = X1 - X0 + 1, Z1 - Z0 + 1
    edge = EDGE_MARGIN if min(W_all, D_all) > 2 * EDGE_MARGIN + 3 else 0
    # **Open land reaches its own edge** (the expression round): a district of land
    # asked for no house lays one block of its land use with no street to keep a margin
    # for, and `beside` is measured from that block's edge to the nearest house
    if int(district.get("structures") or 0) == 0 and district.get("surface") == "open":
        edge = 0
    fr = _Frame(X0 + edge, Z0 + edge, X1 - edge, Z1 - edge)
    # **Which way this district's lots front, where the resolved design says so.** One
    # of `north`/`south`/`west`/`east`, or None where the district has no frontage
    # obligation and a block may face both its streets. It is a side of the *world* and
    # not of the block, so a district whose streets run the other way simply ignores it
    # and the block lays as it always did -- a frontage obligation across the grain of a
    # district is a layout problem and not something a block can solve.
    faces_at = district.get("faces_anchor") or None
    faces_default = str(district.get("faces") or "").lower() or None
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
    # **A width the character declared is the width**, clamped into what the type admits
    # and not raised by the cover lever -- the same rule a declared `lot_depth` has
    # always had. Without it an inspection could ask for a narrower frontage and get a
    # wider one, because the lever that makes a district cover its ground is the lever
    # that widens the lot. **What the resolved demand says this district's fabric needs
    # of a lot**, before a side is chosen (the design round's first contract). A refusal
    # here is a refusal: nothing in the approved band delivers a feature the requirement
    # made required, and a smaller lot is not an answer to that.
    dem_lot, lot_refused = _demand_lot(dem, district)
    declared_w = ch.get("lot_width")
    if declared_w:
        side = int(declared_w)
        # **An explicit lot width is validated, not trusted.** A character may declare a
        # frontage the requirement's own features cannot stand on; the declaration is
        # the principal's inference and the envelope is what measured it. The refusal is
        # on the record and the demand's own least lot stands over the declaration.
        d_mod = _demand_module()
        if dem is not None and d_mod is not None and hasattr(d_mod, "validate_lot"):
            want_d = int(ch.get("lot_depth") or side)
            ok = d_mod.validate_lot(dem, [side, want_d]) or {}
            if not ok.get("ok"):
                lot_refused = {"declared": [side, want_d],
                               "why": str(ok.get("why") or "the envelope refuses this "
                                          "lot for the required features"),
                               "requirements": list(dem.get("requirements") or ())}
    else:
        side = _lot_side(density)
        if want_lots > 0:
            from_ask = int(math.ceil(math.sqrt(target["min_plot_columns"]
                                               / float(want_lots))))
            side = max(side, from_ask)
        side += int(ch.get("_lot_grow") or 0)
    attached = bool(ch.get("attached"))
    open_front = ch.get("frontage") == "open"
    attached_note = None
    lot_min_raised = None
    storey_depth_raised = None
    # **What this district is for decides what it is built of** (`use_mix`). The
    # quarter's own fabric is what the streets are drawn from; a building of another use
    # appears only where this district's own design asks for it -- a requirement its
    # demand resolved, or a word of its own description -- and *where* it stands is the
    # street its anchor fronts rather than a fraction of the lots. The primary house is
    # the first of the quarter's own uses in the pool's own order, so a capability
    # record that put a civic type first no longer makes it the district's house.
    mix = use_mix(district, ch, houses, role, part=part, decls=decls)
    kin = mix["own"]
    house_name, house = kin[0]
    if attached:
        # v2, C2: a row of party walls is built of a type that declares `ATTACHED`, at a
        # lot every configuration of its flanks admits; where the role has no such type,
        # or none admits a lot, the row is detached and the record says the type the
        # district's own design names first, and otherwise the library's terrace
        # (`terrace_order`: homes before shops, the least pad first) -- not the table's
        # alphabetical order, which made every attached fabric a row of courtyard houses
        # once that type could stand attached
        terraced = [(n, d) for n, d in kin if d.get("attached")]
        named_t = [x for x in terraced if x[0] in (mix.get("named") or {})]
        terraced = named_t + terrace_order([x for x in terraced if x not in named_t])
        pick = None
        # **The lot the character or the adopted arrangement asked for, exactly**, where
        # the type admits it. The neighbourhood round's first seam defect: an adopted
        # `lot_depth` reached `character_of` -- so the record said 6x6 -- and reached
        # the plots only when it happened to stand at the width `_attached_lot` had
        # already chosen, and when it did not, the depth was silently the other number
        # and nothing said so. An adopted arrangement is either laid or refused by name.
        want_lot = ((int(declared_w), int(ch["lot_depth"]))
                    if declared_w and ch.get("lot_depth") else None)
        want_area = (int(want_lot[0]) * int(want_lot[1])) if want_lot else None
        # **...and the shape the density asked for, where nobody declared one.** The
        # spatial design round, found by building the section and looking at its street.
        # `placeplan.fabric` resolves a dense attached fabric to a **6x13** terrace and
        # charges the ring's ground for one; this call passed `side` and an area and
        # nothing else, so `_attached_lot` answered with the nearest lot to a hundred
        # columns -- **10x9** -- and the compiler laid that. Two rules about one
        # question, again: the arithmetic that says what a house costs and the compiler
        # that lays it disagreed by four columns of frontage. What that looks like
        # built, on this section's crowded ring: 82 row houses on 10-wide lots, each one
        # **6 wide**, so every party wall stands four columns short of its neighbour and
        # a street of terraces reads from the air as detached houses on lawns -- which
        # is what the independent reader said of the round before this one, about the
        # same street, for the same reason. A preference and not a demand: `shape` ranks
        # the candidates and a type that cannot stand at that depth still gets the lot
        # its own band admits.
        shape = None
        if want_lot is None:
            with contextlib.suppress(Exception):
                from . import placeplan as _pp
                shape = tuple(int(v) for v in
                              _pp.fabric(density, role, ch)["lot"])[:2]
        for n, d in terraced:
            got = _attached_lot(d, side, fr.flanks(), fr.u_is_x,
                                area=want_area, want=want_lot, shape=shape)
            if got:
                pick = (n, d, got)
                break
        if pick:
            house_name, house, (w, ld) = pick
            if ch.get("lot_depth"):
                d2 = int(ch["lot_depth"])
                fl = fr.flanks()
                atts = (fl, fl[:1], fl[1:], ())
                if all(_admits(house, w, d2, att, fr.u_is_x) for att in atts):
                    ld = d2
                elif int(ld) != d2:
                    # **the depth this fabric was asked for and cannot stand at**, on
                    # the record rather than in silence. A width that carries it is
                    # preferred over the area's own answer; where no width does, the
                    # shortfall is named and `placeplan.arrangement_failures` refuses
                    # the district.
                    alt = next((ww for ww in range(max(3, house["needs"]["footprint"][0]),
                                                   house["needs"]["footprint"][2]
                                                   + 2 * Builder.SITE_INSET + 1)
                                if all(_admits(house, ww, d2, att, fr.u_is_x)
                                       for att in atts)), None)
                    if alt is not None:
                        w, ld = alt, d2
                    else:
                        attached_note = (
                            f"this district asked for {declared_w}x{d2} attached lots and "
                            f"`{house_name}` admits no lot {d2} deep with its flanks "
                            f"against its neighbours' at any width of its "
                            f"{house['needs']['footprint']} band; it lays {w}x{ld}")
            # **A lot that admits one height where the band asks for a range is one
            # column short of being this fabric's lot.** The neighbourhood delivery
            # round, found by the registered rhythm bar and not by a reading. The
            # detached branch below has applied `lot_min` -- the least lot the envelope
            # requires -- since the expression round; the attached branch never has. So
            # the density arithmetic resolved this fixture's dense terrace to **6x12**,
            # and measured through the same envelope the compiler asks: row_house, 6x12,
            # both flanks attached -> 1 storey row_house, 6x13, both flanks attached ->
            # 2 One column of depth is the difference between a terrace with a skyline
            # and a terrace at one height, and at one height there is no second height
            # for the compiler to alternate to: four neighbours of one shape stood in a
            # row on a street whose frontage is fixed by its own party walls, which is
            # what `IDENTICAL_RUN_MAX` refuses and what no other lever could reach.
            # `_attached_lot`'s own docstring has said since the spatial design round
            # that depth is ranked first *because* "a lot one column short of it stands
            # one storey however much ground it has". Bounded by `STOREY_DEPTH_REACH`
            # and taken only where the character declared neither dimension: a declared
            # lot is the principal's and is not the compiler's to deepen, which is the
            # rule lever 4 states in the other direction.
            if not ch.get("lot_depth") and not declared_w:
                lo_s, hi_s = _storeys_band(house, ch)
                fl = 2 if attached else 0
                fla = fr.flanks()
                atts = (fla, fla[:1], fla[1:], ())
                with contextlib.suppress(Exception):
                    if hi_s > lo_s:
                        got0 = _storeys_fit(house_name, lo_s, hi_s, w, ld,
                                            flanks=fl) or {}
                        if int(got0.get("storeys") or lo_s) <= lo_s:
                            for d2 in range(ld + 1, ld + 1 + STOREY_DEPTH_REACH):
                                if not all(_admits(house, w, d2, att, fr.u_is_x)
                                           for att in atts):
                                    continue
                                got2 = _storeys_fit(house_name, lo_s, hi_s, w, d2,
                                                    flanks=fl) or {}
                                if int(got2.get("storeys") or lo_s) > lo_s:
                                    storey_depth_raised = {
                                        "was": [w, ld], "now": [w, d2],
                                        "band": [lo_s, hi_s],
                                        "admitted_was": int(got0.get("storeys") or lo_s),
                                        "admitted_now": int(got2.get("storeys") or lo_s),
                                        "why": (f"a {w}x{ld} lot of `{house_name}` with "
                                                f"{fl} flank(s) attached admits "
                                                f"{got0.get('storeys')} storey(s) and "
                                                f"this fabric's band asks for "
                                                f"{lo_s}..{hi_s}; {w}x{d2} admits "
                                                f"{got2.get('storeys')}")}
                                    ld = d2
                                    break
        else:
            attached = False
            attached_note = ("no committed type of this role declares ATTACHED at a "
                             "lot it admits; the row is detached")
    if not attached:
        w = _clamp_side(side, house)
        ld = _clamp_side(int(ch.get("lot_depth") or side), house)
        # **A lot the layout says construction needs is the least lot** (the expression
        # round): `district.lot_min` carries the envelope a required storey band or an
        # emitted `needs.lot_min` asked for, and a lot under it is asked of the ground
        # twice. Clamped into what the type admits. It stands over a lot the character
        # declared: the declaration is the principal's inference and the minimum is what
        # construction measured or the envelope requires, and the record says so.
        lm = _least_lot(district.get("lot_min"), dem_lot)
        if lm and len(lm) == 2 and (int(lm[0]) > w or int(lm[1]) > ld):
            was = (w, ld)
            w = _side_at_least(house, max(w, int(lm[0])))
            ld = _side_at_least(house, max(ld, int(lm[1])))
            lot_min_raised = [list(was), [w, ld]]
        # **A lot is a rectangle and `_clamp_side` only ever asked about squares.**
        # Found by running the city once the repair loop let it get this far: the width
        # and the depth were each a side the type admits, and the *pair* was not -- a
        # 6x8 lot of a type written for a 6x6 pad, forty-seven times in one district,
        # refused by the plan validator at the end of the stage. `_deepest` is the
        # question that was meant: the deepest lot **of this width** the type admits.
        if not _admits(house, w, ld, None, fr.u_is_x):
            deep = _deepest(house, w, ld, (), fr.u_is_x)
            if deep:
                ld = deep
            else:
                # no depth at this width stands: fall back to the square the type
                # admits, which `_clamp_side` has already found
                ld = w
    gap = 0 if attached else (PLOT_LANE if open_front else LOT_GAP)
    row_gap = PLOT_LANE if open_front else LOT_GAP
    # **How far one lot may differ from the next**, in columns: the character's own
    # `variety` where it names one, and what the street is where it does not
    # (`VARIETY`). A terrace is meant to be regular and varies least; freestanding
    # buildings a lane apart vary most. The craft round, E3.
    var = ch.get("variety")
    if var is None:
        var = VARIETY["attached" if attached else ("open" if open_front else "street")]
    spread = int(round(float(var) * min(w, ld)))
    if attached:
        # **A row of party walls varies in what it can.** `_attached_lot` picks the one
        # size every configuration of a lot's flanks admits, and for the library's one
        # `ATTACHED` type that is a single width: widening a lot of it by a column is
        # not a rhythm, it is a lot the type refuses. A terrace is meant to be regular
        # anyway; what varies down a terrace is its height and, where the role has more
        # than one attached type, its type.
        spread = 0
    # **...or what the cover needs at the lot this fabric admits, whichever is more.**
    # The ask is `area x plot_share / columns_per_plot` at the density's *own* lot, and
    # a fabric whose lot is smaller than that covers less ground with the same number of
    # houses: a row of party walls is six by eight where a dense lot is ten square, so
    # the district was asked for eight houses and eight of them covered fourteen per
    # cent of ground its density asks thirty-three of. The count is never under the ask;
    # where the lot is smaller, there are more of them. **An exact count is the count.**
    # The closure round's retained failure: asked for seven, this laid eleven, because
    # the floor below raised the ask to what the cover needed at the fabric's lot -- and
    # the sentence had said sixteen, exactly. Where the place level marks a district
    # `exact`, nothing here moves its number; a cover the count cannot reach at this lot
    # is the layout owner's finding, not more houses.
    exact = bool(district.get("exact"))
    if want_lots > 0 and w * ld and not exact:
        want_lots = max(want_lots,
                        int(math.ceil(target["min_plot_columns"] / float(w * ld))))
    # **...and no more of the ground than the density's ceiling admits.** Where the ask
    # at this lot would put the lots over `max_plot_columns`, the lot is shrunk first --
    # the largest side the type admits that brings the count under the ceiling -- unless
    # the character declared the lot, in which case the declaration stands and the
    # record reports the overshoot.
    ceiling = target.get("max_plot_columns")
    if want_lots > 0 and ceiling is not None and w * ld * want_lots > ceiling \
            and not declared_w and not ch.get("lot_depth") and not attached \
            and not district.get("lot_min"):
        most = int(math.floor(math.sqrt(ceiling / float(want_lots))))
        # the largest admitted side under the ceiling, else the smallest the type admits
        # at all: an exact count is laid as far as the ground allows and the record says
        # it is over, rather than laid large and far short
        got_side = _side_within(house, most) or _plot_range(house)[0]
        if got_side and got_side < w:
            w = got_side
            ld = _deepest(house, w, ld, (), fr.u_is_x) or w
            ld = min(ld, w) if _admits(house, w, min(ld, w), None, fr.u_is_x) else ld
            spread = int(round(float(var) * min(w, ld)))
    # **A proposal the ceiling cannot hold is laid at the ceiling**, and the shortfall
    # is reported as one (`arrange` reads `lots < proposed` as `short`, the layout owner
    # moves the ask). The loop fixture asked a medium district for the fabric's own 39
    # and the validator refused the 39 it laid as over the word's ceiling, handing the
    # district to its character's author, who cannot change a count. A count the
    # sentence stated is never capped here: it is laid, and the record says it is over.
    # **...and no widening to a street's leftover where the ceiling binds.** The lots
    # were sized to sit under the ceiling and then widened to take up the run, which put
    # an exact district half a percent over its word. A binding ceiling means the lot is
    # the lot; the leftover is a wider last gap, as it is for a row of party walls.
    # ...for an **exact** count only: a proposal that overshoots is trimmed below
    # (`trimmed`), and a street's rhythm is worth more than a lot's exact width there.
    # **Whether or not the count is the sentence's.** The expression round's city: the
    # middle ring's eleven courtyard houses at their least lot fill 95% of the medium
    # ceiling, the count is inferred and not exact, so remainders were widened onto two
    # lots and one deep one and the district laid 26 columns over its word -- twice, and
    # stopped. A ceiling that the least lots nearly fill binds whatever kind of count
    # asked for them.
    ceiling_binds = bool(want_lots > 0 and ceiling is not None
                         and w * ld * want_lots > (0.8 if exact else 0.9) * ceiling)
    capped_to = None
    if want_lots > 0 and ceiling is not None and not exact and w * ld * want_lots > ceiling:
        capped_to = max(1, int(ceiling // (w * ld)))
        if capped_to < want_lots:
            want_lots = capped_to
    # the quarter's own alternatives, and -- kept apart -- the other uses its own design
    # asks for. Three lists because they are drawn by three different rules: the fabric
    # by the draw, the named work by the street it belongs on, and the required use at
    # the count its requirement asks for.
    others = [(n, d) for n, d in kin if n != house_name
              and _admits(d, w, ld, None, fr.u_is_x) and not attached]
    # **A building of another use is not held to the quarter's own lot.** This filter
    # was `_admits(d, w, ld)` -- the district's own lot, exactly -- and that is what
    # silenced the programme on the retained city: the middle ring's lot is 13x13,
    # `shop_house`'s sweep found a 9-column pad broken, so the one trade type the
    # traders' ring admits was dropped here and the ring came back with no shops and
    # nothing recorded. A shop street has the grain of a shop; what a type must fit is
    # the *run*, not the house next door. Admitted here at any lot at all, sized below
    # (`street_lot`), and the per-lot `stands()` is still the authority.
    asked = [(n, d) for n, d in mix["programme"]
             if not attached and d.get("kind", "plot") == "plot"]
    work = [(n, d) for n, d in asked if n not in (mix.get("required") or {})]
    #: One entry per building a requirement of this district actually asks for, in the
    #: order the requirements resolved: a requirement is a statement that the place must
    #: hold the thing, so it is laid that many times and no more. This is what replaced
    #: `PROGRAMME_USE_SHARE`.
    owed: list = []
    for n, d in asked:
        req = (mix.get("required") or {}).get(n)
        if req:
            owed += [(n, d, req["requirement"])] * max(1, int(req.get("count") or 1))
    #: **The lot the quarter's principal street is cut into**, where its design names
    #: work to put on it. A trade frontage is a count of doors along a length of street,
    #: so the work type takes the **narrowest** lot its own envelope admits at or above
    #: the least lot this district's required features need (`_side_at_least` over
    #: `lot_min`/the demand's own) -- and the deepest lot it admits at that width inside
    #: the district's own row depth, because the row's band is what there is. That is
    #: the one place in this file where a use is given a grain of its own, and it is
    #: why: a shop street with three doors on it is a street with three doors on it
    #: whatever the cover figure says. **The floor is the street type's own envelope and
    #: the layout's explicit minimum, and not the fabric's.** `_demand_lot` answers "a
    #: lot every approved type of this district stands on" -- 13x13 on the retained
    #: middle ring, which is `courtyard_house`'s floor -- and holding a shop to the
    #: largest minimum in the pool is holding it to a house. Where the demand carries a
    #: **required** feature the fabric's own least lot does stand over this, because a
    #: lot that cannot deliver what the request requires is the defect and a narrower
    #: street is not an answer to it. **The recurring use sizes the street, not the one-
    #: off.** A requirement asks for one building and the named work is the whole face,
    #: so a street cut to the requirement's type is a street cut for a building that
    #: appears once. Worse, `street_decl` is also the face's fallback where the drawn
    #: type does not stand on the lot. Measured, then fixed here.
    street_name, street_decl, street_lot = None, None, None
    _lm_here = district.get("lot_min") or [0, 0]
    if ((dem or {}).get("required")):
        _lm_here = _least_lot(_lm_here, dem_lot) or [0, 0]
    for _n, _d in work + ([(owed[0][0], owed[0][1])] if owed else []):
        try:
            _lo2 = _plot_range(_d)[0]
        except (KeyError, TypeError, ValueError):
            continue
        _w2 = _side_at_least(_d, max(int(_lo2), int(_lm_here[0] or 0)))
        _d2 = _deepest(_d, _w2, ld, (), fr.u_is_x)
        if _w2 and _d2 and _d2 >= int(_lm_here[1] or 0):
            street_name, street_decl, street_lot = _n, _d, (int(_w2), int(_d2))
            break
    per_block = int(ch.get("_block_lots") or block_lots_for(density, w, gap))
    block = int(ch.get("block") or (per_block * w + (per_block - 1) * gap))
    block = max(block, w)
    # **How many rows of lots a block holds.** Two, back to back, unless the layout
    # negotiated one for this rectangle (`district.arrangement.rows`): a ring whose
    # count needs a band twenty columns deep is one row of houses along a street, and
    # holding it to a block of two rows is what made the ring's least width the thing
    # its density was measured over. The compiler is still the authority -- it lays the
    # rows and the record says how many it got.
    rows_per_block = max(1, int(ch.get("rows") or BLOCK_ROWS))
    bd = block_depth(ld, rows_per_block, row_gap)
    street = PLOT_LANE
    # **Is a block built round its court?**.
    perimeter = bool(ch.get("perimeter"))

    # --- the reservations, **before the ground is cut into blocks** --------------- The
    # composition round's first change. The grid was cut from the fabric's own lot and
    # block, the landmark was then given whichever block came nearest the middle, and
    # its side was clamped to that block (`max(lo, min(hi, 17, block_w, block_d))`) --
    # so a block narrower than the landmark's own floor produced a side the same block
    # then refused, the leaf fell through to `kind = "row"`, and the ground the request
    # required for a market became a row of houses with nothing recorded. Measured on a
    # middle-ring sector laid one row deep: 19 houses, no market, `landmarks: 0`, and
    # every check silent. So the reservation is sized first, and where the request
    # **requires** it the block grid is sized to hold it rather than capping it. A
    # landmark the character merely prefers is left exactly as it was: its block is what
    # the fabric's own grid gives.
    reserve = reservations_of(district, ch, decls, lot=(w, ld), side=side)
    demand_short: list = []
    reserve_drove = None
    need_side = max((int(r["least"]) for r in reserve if r["required"]), default=0)
    if need_side and (need_side > block or need_side > bd):
        reserve_drove = {"least": int(need_side),
                         "block": [int(block), int(max(block, need_side))],
                         "block_depth": [int(bd), int(max(bd, need_side))],
                         "for": [r["subject"] for r in reserve if r["required"]],
                         "why": (f"a block of {block}x{bd} cannot hold the "
                                 f"{need_side}-column reservation this district's "
                                 f"requirement makes required; the grid is cut to hold "
                                 f"it before the housing takes the ground")}
        block = max(block, need_side)
        bd = max(bd, need_side)

    # --- what the ground already holds: the arterial and the standing parts ----
    band = np.zeros((fr.U, fr.V), dtype=bool)
    #: the road in world columns, for a doorstep that falls just outside this district
    _levels = ((place.get("arterials") or {}).get("levels") or {})
    _g_here = district.get("ground") or {}
    _lvl_here = _g_here.get("level") if _g_here.get("measured") else None
    # ...**at the level the lots stand at**: a road graded down into a pond in front of
    # a terrace is not a doorstep for the terrace's houses (the first built candidate:
    # landings at y=54 in front of floors at y=64, ten blocks of air between)
    road = {(int(c[0]), int(c[1]))
            for c in ((place.get("arterials") or {}).get("cells") or [])
            if _lvl_here is None or _levels.get(f"{int(c[0])},{int(c[1])}") is None
            or abs(int(_levels[f"{int(c[0])},{int(c[1])}"]) - int(_lvl_here)) <= 1}
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

    # --- ...and the ground a building may actually be founded on ------------------
    # **The third kind of unavailable ground.** The spatial-design round, and the
    # review's first finding at the level it bites: this module's own docstring says it
    # does not read terrain, `placeplan.developable_columns` now subtracts the ground
    # `ethoslm.feasible` refuses so a district's *count* is honest, and the grid was still
    # laid over the whole rectangle -- so the count was right and the lots were on the
    # lake. The arterial's band and a standing part's clearance are ground this compiler
    # has always known it may not build on; water and a grade the terrace cannot reach
    # are the same kind of fact, and they go in the same predicate. `feasible.mask_of`
    # answering None is **not** "all bad": a record with no bitmap is ground nobody
    # measured, and treating it as refused would empty every district compiled without a
    # volume, which is every fixture and every unit test. It is also not silently "all
    # good" -- `ground["measured"]` is the flag that says which, and a district that
    # carries a measured record gets the mask applied.
    ground_rec = district.get("ground") or {}
    #: The level this district's ground was designed at, carried onto every leaf so the
    #: builder sites a house on the terrace it was planned for rather than on a
    #: doorstep. `None` where no ground decision was measured, which is every fixture.
    _design_level = (int(ground_rec["level"])
                     if ground_rec.get("measured") and ground_rec.get("level") is not None
                     else None)
    founded = None
    try:
        from . import feasible as _feasible
        gmask = _feasible.mask_of(ground_rec) if ground_rec.get("measured") else None
    except Exception:                            # noqa: BLE001 -- no reading, no rule
        gmask = None
    if gmask is not None:
        ox, oz = (ground_rec.get("origin")
                  or (ground_rec.get("mask_bits") or {}).get("origin")
                  or [fr.x0, fr.z0])
        ok = np.zeros((fr.U, fr.V), dtype=bool)
        # the mask is indexed [x - origin_x, z - origin_z]; the frame is (u, v)
        xs = np.arange(fr.x0, fr.x1 + 1) - int(ox)
        zs = np.arange(fr.z0, fr.z1 + 1) - int(oz)
        vx = (xs >= 0) & (xs < gmask.shape[0])
        vz = (zs >= 0) & (zs < gmask.shape[1])
        sub = np.zeros((len(xs), len(zs)), dtype=bool)
        if vx.any() and vz.any():
            sub[np.ix_(vx, vz)] = gmask[np.ix_(xs[vx], zs[vz])]
        ok[:, :] = sub if fr.u_is_x else sub.T
        # an integral image, so asking about one lot is four lookups and not a slice
        cum = np.zeros((fr.U + 1, fr.V + 1), dtype=np.int32)
        cum[1:, 1:] = np.cumsum(np.cumsum(ok.astype(np.int32), axis=0), axis=1)

        def founded(u0, u1, v0, v1) -> bool:           # noqa: F811 -- the measured case
            # **...and the ground it is entered from.** The neighbourhood delivery
            # round, found by building the block once the earthwork was bounded. The
            # mask was asked about the lot and about nothing else, so a lot whose own
            # columns are mostly prepared could be entered across a column the design
            # had refused -- `middle_ring_north_east_b0_0_02` stood on a plinth at y=64
            # and the lane at its doorstep was left at its own bed of y=60, three blocks
            # below the plot's edge, and `E008` said the reserved doorway could not be
            # walked into off its own threshold. The plot was right, the lane was right,
            # and the seam between two correct decisions ran through the way in. A
            # doorstep is one column outside the plot (`SKIRT`), which is where
            # `circulate._threshold_for` reserves it and where `site()` lays the
            # platform's ledge. Asking the mask over the lot *and its skirt* refuses the
            # lot that cannot be reached instead of standing it and refusing its door.
            # **Two questions, not one bigger rectangle.** Asking the share over the lot
            # *and* its skirt as a single rectangle makes the test weaker, not stronger:
            # a 13x13 lot 49% of which the design refuses passes at 15x15 if the ring
            # round it is dry. Both have to hold.
            def share(a0, a1, b0, b1) -> bool:
                a0, b0 = max(0, int(a0)), max(0, int(b0))
                a1, b1 = min(fr.U - 1, int(a1)), min(fr.V - 1, int(b1))
                if a1 < a0 or b1 < b0:
                    return False
                n = (a1 - a0 + 1) * (b1 - b0 + 1)
                good = int(cum[a1 + 1, b1 + 1] - cum[a0, b1 + 1]
                           - cum[a1 + 1, b0] + cum[a0, b0])
                return good >= GROUND_FOUNDED * n
            return (share(u0, u1, v0, v1)
                    and share(u0 - SKIRT, u1 + SKIRT, v0 - SKIRT, v1 + SKIRT))

        def pad_founded(u0, u1, v0, v1, need=None) -> bool:  # noqa: F811 -- measured
            """**Is the ground this building will stand on ground the design prepares?**

            The block design round, found by building the section: `middle_ring_north_east
            _b0_1_01` is a `court_large` on a 13x13 lot whose ground runs y=64..73. The
            mask refuses those columns -- a nine-block cut is past
            `placeplan.DISTRICT_TERRACE_REACH` -- but they are a minority of the lot, so
            `founded`'s two share tests admitted it; `site()` then cut the flattest pad it
            could out of broken ground, handed the type **7x7**, and `court_large` refused
            it by name ("four ranges of 3 round a court of 3 need 9 on each axis"). What
            stood was a platform and an approach with nothing on them: `E010`, eight
            blocks held up by nothing.

            A share is the right test for "is this lot mostly buildable" and the wrong one
            for "can this building be built". A pad is the rectangle `site()` insets out of
            the plot (`buildlib.SITE_INSET`) and it is what `build()` is handed, so **every
            column of it** has to be ground this design prepares. A lot whose pad the
            design does not prepare is a lot for a building that will not stand.
            """
            a0, a1 = max(0, u0 + PAD_INSET), min(fr.U - 1, u1 - PAD_INSET)
            b0, b1 = max(0, v0 + PAD_INSET), min(fr.V - 1, v1 - PAD_INSET)
            if a1 < a0 or b1 < b0:
                return False
            if need is None:
                return bool(ok[a0:a1 + 1, b0:b1 + 1].all())
            # **the lot, not the inset plot.** `site()` drops the inset on an attached
            # flank, so a terrace lot's pad reaches its own edge on two sides and an
            # inset rectangle is not the ground the building gets: held to the inset,
            # `lower_ring_north_2` laid **no lot at all** -- `row_house` needs nine
            # columns of pad depth and a 10x10 plot inset by two leaves six.
            a0, a1 = max(0, u0), min(fr.U - 1, u1)
            b0, b1 = max(0, v0), min(fr.V - 1, v1)
            # **the pad this type needs, cut out of the ground this design prepares.**
            # Holding the *whole* inset plot to the mask is the honest rule and it is
            # far too strong: `site()` takes the flattest rectangle it can and a type
            # only needs its own declared footprint, so the whole-plot rule emptied the
            # middle ring (19 proposed, 1 realized) on ground its buildings would have
            # stood on. The question is whether a rectangle of the size this building
            # needs is all prepared, anywhere inside the plot's pad.
            nw, nd = int(need[0]), int(need[1])
            for (aw, ad) in ({(nw, nd), (nd, nw)} if nw != nd else {(nw, nd)}):
                if aw > a1 - a0 + 1 or ad > b1 - b0 + 1:
                    continue
                for ia in range(a0, a1 - aw + 2):
                    for ib in range(b0, b1 - ad + 2):
                        if ok[ia:ia + aw, ib:ib + ad].all():
                            return True
            return False

        def entrance_ok(u0, u1, v0, v1, front) -> bool:    # noqa: F811 -- measured case
            """**Is there a doorstep on this lot's own front that the design prepares?**

            The block design round, and the audit's first cause. `founded` above is two
            *share* tests, and a share does not establish an entrance: the audit's
            sentence is "the critical cells can lie in the rejected half of both
            rectangles". A lot 60% of which the design prepares passes both tests with
            its whole street face on ground the terrace will not reach -- and then the
            threshold pass reserves a doorstep on a lane at its own bed, siting follows
            the doorstep down, and a house stands three blocks inside a pit with a stair
            facing the wrong way. That is exactly `lower_ring_north_2_b1_0_00`.

            So the entrance is a **hard condition on named cells**, asked before the lot
            is drawn: the strip one column outside the lot's front (`SKIRT`, where
            `circulate._threshold_for` reserves the doorstep and where `site()` lays the
            platform's ledge) must hold a run of `ENTRY_RUN` columns the design prepares,
            each with the lot's own front column prepared beside it. A person steps from
            the street onto the doorstep and from the doorstep into the house, and both
            of those columns are ground this design has decided to make.

            An intentional step or ramp is not what this refuses: the terrace's own level
            is what the strip is prepared *to*, and `approach()` lays the flight. What it
            refuses is a lot whose way in was never part of the design.
            """
            if front is None:
                return True
            lo_u, hi_u = max(0, u0), min(fr.U - 1, u1)
            lo_v, hi_v = max(0, v0), min(fr.V - 1, v1)
            if hi_u < lo_u or hi_v < lo_v:
                return False
            if front == fr.front(True):
                step_v, in_v = v0 - SKIRT, lo_v
                cells = [(u, step_v, in_v) for u in range(lo_u, hi_u + 1)]
            elif front == fr.front(False):
                step_v, in_v = v1 + SKIRT, hi_v
                cells = [(u, step_v, in_v) for u in range(lo_u, hi_u + 1)]
            elif front == fr.flanks()[0]:
                step_u, in_u = u0 - SKIRT, lo_u
                cells = [(v, step_u, in_u) for v in range(lo_v, hi_v + 1)]
            elif front == fr.flanks()[1]:
                step_u, in_u = u1 + SKIRT, hi_u
                cells = [(v, step_u, in_u) for v in range(lo_v, hi_v + 1)]
            else:
                # an orientation that is not one of this frame's four sides: a lot on
                # open ground, which has no street face to be entered from
                return True
            along_v = front in (fr.front(True), fr.front(False))
            run = 0
            # **the arterial is a street a door may open onto.** The quarter design
            # round: a road cell is not in the ground mask (the road owns it), so a lot
            # fronting the gate street -- the one street every visitor arrives on -- was
            # refused its way in, and the end range of the court block beside it with
            # it. A doorstep on the carriageway's edge is a doorstep on a prepared
            # street. ...**and so is the road just outside the district**, where a lot
            # stands at the district's own edge against the arterial that bounds it.
            def _on_road(u, v) -> bool:
                # the doorstep on the road, or on the verge between this district's
                # frame and the road (`ROAD_VERGE` columns): the lane that joins them is
                # the circulation's to lay
                x_, z_, _x, _z = fr.rect(u, u, v, v)
                return any((int(x_) + dx, int(z_) + dz) in road
                           for dx in range(-ROAD_VERGE, ROAD_VERGE + 1)
                           for dz in range(-ROAD_VERGE, ROAD_VERGE + 1))
            for (n, outside, inside) in cells:
                if along_v:
                    out_ok = ((ok[n, outside] or (band[n, outside] and _on_road(n, outside)))
                              if 0 <= outside < fr.V else _on_road(n, outside))
                    good = bool(out_ok and ok[n, inside])
                else:
                    out_ok = ((ok[outside, n] or (band[outside, n] and _on_road(outside, n)))
                              if 0 <= outside < fr.U else _on_road(outside, n))
                    good = bool(out_ok and ok[inside, n])
                run = run + 1 if good else 0
                if run >= ENTRY_RUN:
                    return True
            return False

    def free(u0, u1, v0, v1, front=None, pad=False) -> bool:
        if band[u0:u1 + 1, v0:v1 + 1].any() or taken[u0:u1 + 1, v0:v1 + 1].any():
            return False
        if founded is None:
            return True
        return (founded(u0, u1, v0, v1) and entrance_ok(u0, u1, v0, v1, front)
                and (pad is False or pad_founded(u0, u1, v0, v1,
                                                 None if pad is True else pad)))

    def why_dropped(u0, u1, v0, v1, front=None, pad=False) -> str:
        """Which of the four unavailable grounds refused this rectangle. Asked in the
        order a reader cares about: the road, then a standing part, then the terrain,
        and last the **way in** -- a lot whose own ground the design prepares and whose
        street face it does not is a different finding with a different repair, and the
        block design round separates them."""
        if band[u0:u1 + 1, v0:v1 + 1].any():
            return "arterial"
        if taken[u0:u1 + 1, v0:v1 + 1].any():
            return "standing"
        if founded is not None and not founded(u0, u1, v0, v1):
            return "ground"
        if founded is not None and not entrance_ok(u0, u1, v0, v1, front):
            return "entrance"
        if founded is not None and pad is not False and not pad_founded(
                u0, u1, v0, v1, None if pad is True else pad):
            return "pad"
        return "ground"

    def mark(u0, u1, v0, v1, what):
        ledger[u0:u1 + 1, v0:v1 + 1] = L[what]

    # --- the site: one compatible pad, door and landing, chosen here -----------------
    # **A pad somewhere and an entrance somewhere are not a compatible pair.** The
    # quarter design round, and the block design audit's second cause. `pad_founded` and
    # `entrance_ok` answered "is there one" and returned a boolean; `Builder._site_pad`,
    # `circulate._approach_candidates`, `_threshold_for` and `_door_cell` then chose the
    # pad, the landing and the door again, each by its own rule. `site_solve` chooses
    # them **once**, on the prepared-ground mask the two predicates read, with the pad
    # inset by the one definition the builder uses (`buildlib.pad_insets`), and the leaf
    # carries the answer as `site`. Every consumer downstream reads it verbatim. Written
    # only where the district's ground decision has a level (`_design_level`): a site's
    # floor is that level, and a plan with no measured ground (every fixture) has no
    # floor to carry, so it keeps the old path unchanged.
    _mask_ok = ok if gmask is not None else None
    #: lots refused after they were drawn, by the ground they were refused on
    sites_refused = {"pad": 0, "entrance": 0, "storeys": 0}
    #: every lot whose type changed because its own envelope refused its storeys
    retyped: list = []
    #: every lot refused after it was drawn, with why -- unresolved demand, returned
    lots_refused: list = []
    _OUT = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}

    def prepared(x, z) -> bool:
        """Is world column (x, z) ground this design prepares? Everything, unmeasured."""
        if _mask_ok is None:
            return True
        u, v = ((x - fr.x0, z - fr.z0) if fr.u_is_x else (z - fr.z0, x - fr.x0))
        if 0 <= u < fr.U and 0 <= v < fr.V:
            # the arterial at the lots' level is prepared street (see `entrance_ok`)
            return bool(_mask_ok[u, v]) or bool(band[u, v] and (int(x), int(z)) in road)
        # outside the district: a landing on the road that bounds it, or its verge
        return any((int(x) + dx, int(z) + dz) in road
                   for dx in range(-ROAD_VERGE, ROAD_VERGE + 1)
                   for dz in range(-ROAD_VERGE, ROAD_VERGE + 1))

    def site_solve(x0, z0, x1, z1, front, attached_sides=(), need=None, band=None,
                   inset=None):
        """`(site, None)` or `(None, "pad" | "entrance")` for a plot in world columns.

                The pad is the plot inset by `buildlib.pad_insets` for this attachment. Where
                every column of it is prepared ground it is the pad; where it is not, the
                largest all-prepared rectangle inside it that holds the type's declared
                footprint (`need`, either way round, inside its band `band`), nearest the street
                on a tie -- and a pad set back from the street edge is only a pad where its door
                line across the set-back is prepared too. The door is a cell of the pad's
                street edge, off its corners, nearest its middle, inside a run of `ENTRY_RUN`
                edge columns whose whole line out to the **landing** -- the first column outside
                the plot on the street side -- is prepared. Nothing here reads the world: it is
                the plan's own ground decision, and construction consumes the answer.
                
        """
        from .buildlib import site_pad_rect
        if _design_level is None:
            return None, "unmeasured"
        att = [s for s in (attached_sides or ()) if s]
        sides = ([front] if front else
                 [fr.front(True), fr.front(False)] + list(fr.flanks()))
        sides = [s for s in sides if s in _OUT and s not in att]
        if not sides:
            return None, "entrance"
        # **a lot composed on its street may stand one column off it** (the fabric reset
        # round): `inset` is the leaf's own where it carries one (`p["inset"]`), and the
        # library's `PAD_SITE_INSET` otherwise
        ix0, iz0, ix1, iz1 = (site_pad_rect(int(x0), int(z0), int(x1), int(z1), att)
                              if inset is None else
                              site_pad_rect(int(x0), int(z0), int(x1), int(z1), att,
                                            int(inset)))
        if ix1 - ix0 < 2 or iz1 - iz0 < 2:
            return None, "pad"
        if band is None and need is not None and isinstance(need, dict):
            band = need
        if isinstance(need, dict):
            fp_n = (need.get("needs") or {}).get("footprint")
            need = (int(fp_n[0]), int(fp_n[1])) if fp_n else None
        nw, nd = (int(need[0]), int(need[1])) if need else (3, 3)
        why_last = "pad"
        for side in sides:
            ox, oz = _OUT[side]
            along_x = side in ("north", "south")
            # columns along the street edge, and how deep each is prepared from it
            lanes = (list(range(ix0, ix1 + 1)) if along_x else list(range(iz0, iz1 + 1)))
            depth_max = (iz1 - iz0 + 1) if along_x else (ix1 - ix0 + 1)

            def cell(a, k):
                """The pad cell `k` columns in from the street edge at lane `a`."""
                if side == "north":
                    return (a, iz0 + k)
                if side == "south":
                    return (a, iz1 - k)
                if side == "west":
                    return (ix0 + k, a)
                return (ix1 - k, a)
            grid = [[prepared(*cell(a, k)) for k in range(depth_max)] for a in lanes]
            # the candidate pads, largest first and nearest the street on a tie: the
            # whole inset plot where it is all prepared, else every all-prepared
            # rectangle that holds the footprint, set back `k0` from the street edge
            # (its door line then crosses the set-back, which has to be prepared too)
            cands = []
            if all(all(col) for col in grid):
                cands = [(0, len(lanes) - 1, depth_max, 0)]
            else:
                for k0 in range(0, depth_max - 2):
                    heights = []
                    for col in grid:
                        k = k0
                        while k < depth_max and col[k]:
                            k += 1
                        heights.append(k - k0)
                    for i0 in range(len(lanes)):
                        h = depth_max
                        for i1 in range(i0, len(lanes)):
                            h = min(h, heights[i1])
                            if h < 3:
                                break
                            wid = i1 - i0 + 1
                            if (wid >= nw and h >= nd) or (wid >= nd and h >= nw):
                                cands.append((i0, i1, h, k0))
                cands.sort(key=lambda c: (-(c[1] - c[0] + 1) * c[2], c[3], c[0]))
            if not cands:
                why_last = "pad"
                continue
            tried = 0
            for (i0, i1, h, k0) in cands:
                a0, a1 = lanes[i0], lanes[i1]
                if along_x:
                    pad = ((a0, iz0 + k0, a1, iz0 + k0 + h - 1) if side == "north"
                           else (a0, iz1 - k0 - h + 1, a1, iz1 - k0))
                else:
                    pad = ((ix0 + k0, a0, ix0 + k0 + h - 1, a1) if side == "west"
                           else (ix1 - k0 - h + 1, a0, ix1 - k0, a1))
                if not _pad_fits(band, pad[2] - pad[0] + 1, pad[3] - pad[1] + 1):
                    why_last = "pad"
                    continue
                tried += 1
                if tried > 12:
                    break
                # the door's line out: from the pad's edge cell to the landing outside
                # the plot, every column prepared
                edge = {"north": pad[1], "south": pad[3],
                        "west": pad[0], "east": pad[2]}[side]
                out_to = {"north": int(z0) - 1, "south": int(z1) + 1,
                          "west": int(x0) - 1, "east": int(x1) + 1}[side]

                def line(a):
                    if along_x:
                        zs = (range(edge, out_to - 1, -1) if oz < 0
                              else range(edge, out_to + 1))
                        return [(a, z) for z in zs]
                    xs = (range(edge, out_to - 1, -1) if ox < 0
                          else range(edge, out_to + 1))
                    return [(x, a) for x in xs]
                good = [all(prepared(*c) for c in line(a)) for a in range(a0, a1 + 1)]
                need_run = min(int(ENTRY_RUN), a1 - a0 + 1)
                in_run = [False] * len(good)
                n = 0
                for k, g in enumerate(good):
                    n = n + 1 if g else 0
                    if n >= need_run:
                        for q in range(k - n + 1, k + 1):
                            in_run[q] = True
                mid = (a0 + a1) // 2
                doors = [a0 + k for k, r in enumerate(in_run)
                         if r and a0 < a0 + k < a1]
                if not doors:
                    why_last = "entrance"
                    continue
                da = min(doors, key=lambda a: (abs(a - mid), a))
                door = (da, edge) if along_x else (edge, da)
                landing = (da, out_to) if along_x else (out_to, da)
                full = pad == (ix0, iz0, ix1, iz1)
                return ({"pad": [int(v) for v in pad], "floor": int(_design_level),
                         "facing": side, "door": [int(door[0]), int(door[1])],
                         "landing": [int(landing[0]), int(landing[1])],
                         "attached": sorted(att),
                         "why": (f"the {'inset plot' if full else 'prepared part of the inset plot'}"
                                 f" at the district level y={int(_design_level)}, its door "
                                 f"on the {side} edge over prepared ground to the street")},
                        None)
        if os.environ.get("ETHOSLM_STREETPLAN_DEBUG") == "2":
            import sys as _sys
            print(f"        site_solve {x0},{z0},{x1},{z1} {front} att={att} pad="
                  f"{(ix0, iz0, ix1, iz1)} need={need} -> {why_last}; prepared "
                  f"{sum(prepared(x, z) for x in range(ix0, ix1 + 1) for z in range(iz0, iz1 + 1))}"
                  f"/{(ix1 - ix0 + 1) * (iz1 - iz0 + 1)}", file=_sys.stdout)
        return None, why_last

    def _pad_fits(dcl, w_p, d_p) -> bool:
        """Is a `w_p` x `d_p` pad inside this type's declared footprint band and off its
        measured-broken sizes? `needs_footprint_failure`'s test, in pads. True where
        the type declares nothing."""
        fp = ((dcl or {}).get("needs") or {}).get("footprint")
        if not fp or len(fp) < 4:
            return True
        got = (min(w_p, d_p), max(w_p, d_p))
        lo = (min(fp[0], fp[1]), max(fp[0], fp[1]))
        hi = (min(fp[2], fp[3]), max(fp[2], fp[3]))
        if got[0] < lo[0] or got[1] < lo[1] or got[0] > hi[0] or got[1] > hi[1]:
            return False
        ex = tuple(((dcl or {}).get("needs") or {}).get("except") or ())
        return not any(g in ex for g in got)

    def site_lot(pad, attached_sides=()) -> tuple:
        """The lot a pad is equivalent to under `pad_insets`, `(w, d)` in world x, z:
        what the envelope is asked about when the pad is smaller than the inset plot."""
        from .buildlib import pad_insets
        w_p, d_p = pad[2] - pad[0] + 1, pad[3] - pad[1] + 1
        # the insets a lot of about this size would take on each side
        ins = pad_insets(w_p + 4, d_p + 4, attached_sides)
        return (w_p + ins[0] + ins[2], d_p + ins[1] + ins[3])

    # --- the ground this district's grid is laid over -------------------------------
    # **A quarter stands on the part of its rectangle that can carry it.** The
    # neighbourhood delivery round, and the audit's third cause in its architectural
    # half: `middle_ring_north_west` records 2,969 feasible columns of 11,400, proposed
    # nineteen houses and realized **zero**, and the frozen images show empty levelled
    # ground where a quarter should be. Nothing was wrong with the count or with the
    # mask; the grid was laid over the whole rectangle and then lot after lot was
    # refused for ground, because a regular grid over a mesa puts most of its blocks on
    # the mesa. A hillside with a shelf on it is not an unbuildable district: it is a
    # smaller district on the shelf, with the rest honestly open and its unmet demand
    # still owed (`demand_short` below). So where the mask leaves a materially smaller
    # core the grid is laid over the core and the rest of the rectangle is left as it
    # was found. The bar is deliberate: a district that loses only a corner keeps its
    # whole rectangle, because moving a grid costs streets and frontage and buys nothing
    # there. **The envelope is `placeplan`'s, because that is where the mask is
    # published.** The neighbourhood delivery round. Two searches for one answer existed
    # briefly -- this compiler's own peel and `placeplan.developable_envelope` -- and
    # the ground owner's is the authority: it lives beside `district_ground` and
    # `developable_columns` and it is measured against this compiler. It also answers a
    # better question: maximising the feasible ground a rectangle **contains** rather
    # than peeling to a cover, because a rectangle at 100% cover that a block does not
    # fit in lays nothing (`middle_ring_north_west`: the largest all-feasible rectangle
    # is 23x37 and lays **0** lots; a 57x60 at 66% cover lays 2). Measured over this
    # city's 34 districts, it narrows 7 of them and takes 61 lots to 67.
    core = None
    if founded is not None and gmask is not None:
        with contextlib.suppress(Exception):
            from .placeplan import developable_rect as _dev_rect
            got_r = _dev_rect(district, place, decls)
            if got_r:
                gx0, gz0, gx1, gz1 = (int(v) for v in got_r)
                # into the district's own (u, v) frame, and never outside it
                a0, b0 = (gx0 - fr.x0, gz0 - fr.z0) if fr.u_is_x else (gz0 - fr.z0,
                                                                        gx0 - fr.x0)
                a1, b1 = (gx1 - fr.x0, gz1 - fr.z0) if fr.u_is_x else (gz1 - fr.z0,
                                                                        gx1 - fr.x0)
                cu0, cu1 = max(0, min(a0, a1)), min(fr.U - 1, max(a0, a1))
                cv0, cv1 = max(0, min(b0, b1)), min(fr.V - 1, max(b0, b1))
                if (cu1 - cu0 + 1) >= block and (cv1 - cv0 + 1) >= bd \
                        and (cu0, cu1, cv0, cv1) != (0, fr.U - 1, 0, fr.V - 1):
                    core = (cu0, cu1, cv0, cv1)
    if core is None and founded is not None and gmask is not None:
        # the fallback, where the ground owner's envelope cannot answer: this compiler's
        # own peel, on the same mask
        whole = fr.U * fr.V
        here = int(cum[fr.U, fr.V])
        if here and here < GROUND_CORE_WHOLE * whole:
            # **the tightest core that still holds this district's ground.** Peeling to
            # `GROUND_FOUNDED` gives a rectangle exactly half of which is refusable, and
            # a grid laid on it loses about half its lots to the ground again -- which
            # is the defect one step smaller. Measured on `middle_ring_north_west`: the
            # core at a bar of 0.5 is 90x30 and 50% feasible, and at 0.9 it is 49x30 and
            # 91% feasible **holding the same 1,355 buildable columns**. The tighter
            # rectangle is strictly better there, so the bars are tried from the
            # tightest down and the first that keeps enough of the district's buildable
            # ground wins.
            for bar in GROUND_CORE_BARS:
                got = _feasible_core(cum, fr.U, fr.V, bar,
                                     least_u=int(block), least_v=int(bd))
                if not got:
                    continue
                cu0, cu1, cv0, cv1 = got
                kept = int(cum[cu1 + 1, cv1 + 1] - cum[cu0, cv1 + 1]
                           - cum[cu1 + 1, cv0] + cum[cu0, cv0])
                # the core has to hold a block of this fabric and most of what the
                # rectangle can build on; a mask that refuses half a district in a
                # checkerboard has no core to move to
                if (cu1 - cu0 + 1) >= block and (cv1 - cv0 + 1) >= bd \
                        and kept >= GROUND_CORE_LEAST * here \
                        and (cu0, cu1, cv0, cv1) != (0, fr.U - 1, 0, fr.V - 1):
                    core = (cu0, cu1, cv0, cv1)
                    break
    # --- the grid: the arterial's band is a street of it where it is axial --------
    lead = bool(ch.get("_lead", True))
    u_lo, u_hi = (core[0], core[1]) if core else (0, fr.U - 1)
    v_lo, v_hi = (core[2], core[3]) if core else (0, fr.V - 1)
    sub_band = band[u_lo:u_hi + 1, v_lo:v_hi + 1]
    # a row of party walls is lots of one width, so its blocks are not widened to a
    # remainder: what is left at the end of a run is a verge
    cols, axial_u = _grid_axis(sub_band, 0, u_hi - u_lo + 1, street, block, w, lead=lead,
                               widen=not attached and not ceiling_binds)
    # **A perimeter block is entered on all four faces**, the quarter design round: laid
    # against the district's back edge, its back range fronted ground no street reaches
    # and `entrance_ok` refused every lot of it, so no composition could close on a
    # district whose blocks are one row deep. Its rows end with a street as well.
    rows, axial_v = _grid_axis(sub_band, 1, v_hi - v_lo + 1, street, bd, ld, lead=lead,
                               trail=perimeter)
    if core:
        cols = [(a + u_lo, b + u_lo) for a, b in cols]
        rows = [(a + v_lo, b + v_lo) for a, b in rows]
    blocks = [(i, j) for j in range(len(rows)) for i in range(len(cols))]
    n_blocks = len(blocks)
    # **A land district asked for no houses is still laid.** The expression round's
    # orchard hamlet: the orchard district (0 structures, open land) took the land lot
    # of 24 as its block depth, its rectangle held no row of that depth, and the
    # compiler laid nothing -- no grove, no leaf, nothing for the relation to measure. A
    # district with no lots to grid is one open block of its land use.
    if not blocks and want_lots == 0:
        cols, rows = [(0, fr.U - 1)], [(0, fr.V - 1)]
        blocks = [(0, 0)]
        n_blocks = 1
    # the free block nearest the district's middle takes the landmark -- **or nearest
    # the point the quarter is entered from**, where the parent names one. The quarter
    # design round: the traders' market was put on the grid's middle block, seventy
    # columns from the gate the whole quarter is entered through, and nothing asked that
    # a market meet the street its visitors arrive on. `access` is the ring's gate
    # (`stages_plan._negotiate_sectors`), in world columns.
    cu, cv = (fr.U - 1) / 2.0, (fr.V - 1) / 2.0
    _acc = district.get("access")
    if _acc:
        ax, az = int(_acc[0]), int(_acc[-1])
        au, av = ((ax - fr.x0, az - fr.z0) if fr.u_is_x else (az - fr.z0, ax - fr.x0))
        cu, cv = min(max(au, 0), fr.U - 1), min(max(av, 0), fr.V - 1)
    order = sorted(blocks, key=lambda ij: (abs((cols[ij[0]][0] + cols[ij[0]][1]) / 2.0 - cu)
                                           + abs((rows[ij[1]][0] + rows[ij[1]][1]) / 2.0 - cv),
                                           ij))

    def _landmark_u(u0_, u1_, s_) -> int:
        """Where along its block a landmark of side `s_` stands: centred, as it always
        was -- **or at the end nearest the quarter's access**, where the parent names
        one. The quarter design round: centred on a 48-column block, a 17-column market
        left fifteen columns either side, too few for a lot, so the market's own block
        held no house and "a market among houses" stood alone on its frontage. At the
        end nearest the gate it meets the gate street and the rest of the block is one
        run of frontage its houses can stand on."""
        if not _acc:
            return u0_ + ((u1_ - u0_ + 1) - s_) // 2
        # one lot in from that end where the block has room for a lot either side, so
        # the corner is a house on the street the quarter is entered by and the market
        # stands between houses rather than at the end of the row
        one = int(w) + LOT_GAP
        room = (u1_ - u0_ + 1) - s_ - 2 * one
        if abs(cu - u0_) <= abs(cu - u1_):
            return u0_ + one if room >= 0 else u0_
        return u1_ - s_ + 1 - one if room >= 0 else u1_ - s_ + 1
    landmarks = list(ch.get("landmarks") or [])
    kind_of = {}
    if landmarks:
        # **the block nearest the middle that is free *and holds the reservation***, and
        # where none does, the nearest free block with the shortfall recorded. The old
        # rule was "free", full stop, so a required landmark could be given a block
        # narrower than its own floor and lose the argument with the block later, in
        # silence.
        def _holds(ij) -> bool:
            return (cols[ij[0]][1] - cols[ij[0]][0] + 1) >= need_side \
                and (rows[ij[1]][1] - rows[ij[1]][0] + 1) >= need_side
        free_blocks = [ij for ij in order
                       if free(cols[ij[0]][0], cols[ij[0]][1],
                               rows[ij[1]][0], rows[ij[1]][1])]
        pick = next((ij for ij in free_blocks if _holds(ij)), None) \
            or (free_blocks[0] if free_blocks else None)
        if pick is not None:
            kind_of[pick] = "landmark"
        else:
            for r in reserve:
                demand_short.append({
                    **r, "available": None, "available_columns": 0,
                    "needed": [r["least"], r["least"]],
                    "needed_columns": int(r["least"]) ** 2,
                    "why": (f"no block of this district's {n_blocks} is free of the "
                            f"arterial's band and the standing parts, so the "
                            f"{r['subject']} the character names has nowhere to stand"),
                    "alternatives": _reserve_alternatives(r, 0, 0, block, bd,
                                                          fr.U, fr.V, want_lots)})
    rest = [b for b in blocks if b not in kind_of]
    rest.sort(key=lambda ij: _seeded(seed, "block", ij[0], ij[1]))
    n_open = int(round(float(ch.get("open_share") or 0.0) * n_blocks))
    court_share = float(ch.get("courtyard_share") or 0.0)
    n_court = int(round(court_share * n_blocks))
    # **A form that owes a court lays one.** The neighbourhood delivery round, and the
    # audit's fourth cause. `demand.court_obligation` publishes the obligation off this
    # same share, so a district at 0.05 over eight blocks *owes* a court and
    # `round(0.4)` laid **none** -- which is the whole of "the crowded ring has no
    # court" on the delivered candidate, and no amount of re-ranking arrangements or
    # widening the housing pool reaches it. A share is a proportion of a fabric, not a
    # permission to round the obligation away: where the share is positive and the
    # district has a block to spend, at least one block is a court block.
    court_owed = bool(court_share > 0) and n_blocks > 0
    if court_owed:
        n_court = max(1, n_court)
    for b in rest[:n_open]:
        kind_of[b] = "open"
    # a courtyard block is a block of two rows whose back row is the court: only a block
    # deep enough for two rows can be one (named apart from the `deep` the block loop
    # below computes for a shallow block's one deep row: this one is the list of blocks
    # that could hold a court at all)
    court_deep = [b for b in rest[n_open:]
                  if rows[b[1]][1] - rows[b[1]][0] + 1 >= 2 * ld + row_gap]
    for b in court_deep[:n_court]:
        kind_of[b] = "courtyard"
    # **...and where no block of this district is deep enough for two rows, the court is
    # owed and unheld rather than silently absent.** Recorded for the layout owner, who
    # is the one that can deepen a block or a district; `demand_short` below is the
    # channel a shortfall belongs in.
    court_unheld = court_owed and not court_deep
    for b in rest[n_open:]:
        kind_of.setdefault(b, "row")

    # --- the quarter's principal street ----------------------------------------------
    # **Where the work of a quarter stands.** The neighbourhood round replaced
    # `PROGRAMME_USE_SHARE` -- a tenth of every district's lots, everywhere -- with a
    # spatial decision this district's own plan already contains: a trade belongs on the
    # street its market is on. So the principal street is the one the **anchor** fronts:
    # the block the reservation was given, and the block across the street from it.
    # Where nothing anchors this district it is the middle block of its own grid, which
    # is the block the landmark rule would have picked and the nearest thing to a main
    # street a bare grid has. **A street and not a block**, and both sides of it: the
    # low-v faces of the blocks in the anchor's own row and the high-v faces of the row
    # before it. A street is a length of frontage, which is what the trade on it is a
    # count of. The remaining rows are the quarter's houses, so the scale of the claim
    # is one street of this district's own grid: a half of a two-row sector, a third of
    # a three-row one, and the share is whatever those faces actually held. **A district
    # of one street has no principal street**, and that restriction is load bearing. On
    # a 200x17 band the grid cuts one block row, so the one street is the only street --
    # and a rule that gives the quarter's work the principal street gave it *every lot
    # the district has*: measured on the round's own thin fixture, three buildings, all
    # of them shops, in a quarter whose fabric is homes. A principal street is principal
    # relative to the back streets behind it; where there are none, the work stands on
    # whatever lots the quarter's own grain admits it on and the record says why it got
    # no street of its own.
    lm_block = next((ij for ij, k in kind_of.items() if k == "landmark"), None)
    principal_j = (int(lm_block[1]) if lm_block is not None
                   else (len(rows) // 2 if rows else None))
    if len(rows) < 2:
        principal_j = None
    principal_from = (
        (f"the street the `{reserve[0]['subject'] if reserve else 'anchor'}` this "
         f"district reserves fronts: block row {principal_j} of {len(rows)}, both sides")
        if lm_block is not None and principal_j is not None else
        (f"the middle street of this district's own grid (row {principal_j} of "
         f"{len(rows)}, both sides); nothing anchors this district, so its principal "
         f"street is the one the landmark rule would have used")
        if principal_j is not None else
        (f"this district's grid is {len(rows)} block row(s) deep, so it has no street to "
         f"give one use over another; every lot is drawn from the quarter's own fabric "
         f"and a building of another use stands only where that fabric's own lot admits "
         f"it"))

    def _principal_face(i, j, front) -> bool:
        if principal_j is None or (not work and not owed):
            return False
        return ((int(j) == principal_j and front == fr.front(True))
                or (int(j) == principal_j - 1 and front == fr.front(False)))

    # streets: everything not in a block, and the arterial's band
    ledger[:, :] = L["street"]
    for (i, j) in blocks:
        u0, u1 = cols[i]
        v0, v1 = rows[j]
        mark(u0, u1, v0, v1, "undeveloped")

    # --- the leaves ----------------------------------------------------------
    quarters: dict = {}
    # **Three grounds a lot can be refused on, counted apart.** The third is the terrain
    # (`why_dropped`): a reader who sees a district short of its count is owed which of
    # the road, a standing part and the ground it was short to, and they are three
    # different findings with three different repairs.
    dropped = {"arterial": 0, "standing": 0, "ground": 0, "entrance": 0, "pad": 0}
    counts = {"lots": 0, "courts": 0, "open": 0, "verges": 0, "landmarks": 0,
              "party_walls": 0}
    n_leaf = [0]
    #: The shape of every house, street face by street face, in the order it was laid:
    #: the type, the width, the depth and the storeys, which is what a person walking
    #: down the street sees of it. The craft round, E3, and the two numbers the phase
    #: registers are read off this.
    faces: list = []
    #: What the district still owes the voice's second wall material, in buildings.
    alt_debt = [0.0]
    #: The passages laid from a street into a court, in world coordinates: what makes an
    #: enclosed court a place somebody can walk into rather than a light well.
    court_entries: list = []

    #: **Houses this district has promised to a composition that has not been laid
    #: yet.** The block design round, and the audit's second cause. A court block is not
    #: a sequence of four independent rows: it is one composition whose court exists
    #: only because four ranges stand round it. `compose_court_block` called the
    #: ordinary row emitters in turn and each of them stopped at `counts["lots"] >=
    #: want_lots`, so a district whose count ran out on the third side laid two ranges,
    #: a paved tile and no enclosure -- recorded honestly as `perimeter_open`, and still
    #: standing in the world as a court that is not one. So the composition's children
    #: are **reserved before anything else spends them**. `reserved[0]` is the number of
    #: lots owed to compositions still to be laid; every ordinary emitter spends against
    #: `want_lots - reserved[0]` and therefore cannot take a court block's third range.
    #: A composition whose reservation the district cannot afford is not begun: it is
    #: refused by name, with the shortfall on the record, and the block falls through to
    #: the ordinary two-row block.
    reserved = [0]
    #: Compositions refused or rolled back, with the reason: an unmet obligation stays
    #: explicit rather than becoming a paved tile nobody asked about.
    compositions: list = []

    def budget_left() -> int:
        """Lots this emitter may still spend: the district's count, less what is owed to
        compositions that have not been laid."""
        return int(want_lots) - int(reserved[0]) - int(counts["lots"])

    #: **How many of a lot's flanks the next building stands against.** The
    #: neighbourhood delivery round. `Builder._insets` drops the pad's inset on an
    #: attached side, so a 6x13 plot with two party walls hands `build()` a 6x9 pad and
    #: the same plot standing free hands it 4x11 -- and `row_house`'s own declaration
    #: (`STOREY_PAD[2] = (5, 9)`, in **pad** columns) admits a second storey on the
    #: first and refuses it on the second. The envelope has to be asked the question the
    #: lot is actually in, and the two ends of a terrace are a different question from
    #: its middle: they have one free flank each. A fabric's *intended* configuration is
    #: what a lot is drawn under -- a terrace is drawn expecting both neighbours -- and
    #: the leaf's own count is settled below, after the drops, when the neighbours are
    #: known (`settle_storeys`).
    _FABRIC_FLANKS = 2 if attached else 0

    def _keeps_storeys(tname, decl, w_cols, deep, shallow) -> bool:
        """Does narrowing this type's lot from `deep` to `shallow` cost it a storey its
        own band asks for? The one question every depth lever in this compiler has to
        ask, asked in the flank configuration the fabric is in."""
        if int(shallow) >= int(deep):
            return True
        lo_p, hi_p = _storeys_band(decl, ch)
        if hi_p <= lo_p:
            return True
        try:
            a = (_storeys_fit(tname, lo_p, hi_p, int(w_cols), int(deep),
                              flanks=_FABRIC_FLANKS) or {}).get("storeys")
            b = (_storeys_fit(tname, lo_p, hi_p, int(w_cols), int(shallow),
                              flanks=_FABRIC_FLANKS) or {}).get("storeys")
        except Exception:                        # noqa: BLE001 -- no envelope, no rule
            return True
        return a is None or b is None or int(b) >= int(a)

    def _admits_storeys(tname, decl, w_cols, d_cols, flanks):
        """What the envelope says a `w_cols` x `d_cols` lot of this type admits, asked
        in the flank configuration the lot is actually in. `{}` where nothing can say."""
        lo_p, hi_p = _storeys_band(decl, ch)
        got = _storeys_fit(tname, lo_p, hi_p, int(w_cols), int(d_cols),
                           cache=(place.get("layout") or {}).get("envelope_cache")
                           if isinstance(place.get("layout"), dict) else None,
                           demand=(dem if dem and dem.get("types")
                                   and tname in (dem.get("types") or ())
                                   else None),
                           flanks=int(flanks)) or {}
        return lo_p, hi_p, got

    def _envelope_row(tname, lo_p, hi_p, got, w_cols, d_cols, flanks):
        """The `envelope` record a leaf carries: **what this lot admits of this type**.

                `flanks` is on it because the answer is about that configuration and a reader
                comparing the record with what stood needs to know which was asked. `type` is on
                it because the answer is about one building and the audit found records naming
                another (`court_large` on a `shop_house`).

                **Written for every leaf, not only the restricted ones.** The block design round,
                and the audit's third cause in its second half: this returned `None` wherever the
                lot admitted its whole band, so the delivered section recorded an envelope on 5 of
                83 leaves, all five restrictions -- and a check of the published records therefore
                could not establish that any *successful* leaf had been admitted at all. Absence
                was doing the work of an affirmative answer. A leaf now carries the answer either
                way and `restricts` says which kind it is, so agreement is a thing on disk.
                
        """
        fit = got.get("storeys")
        base = {"storeys_band": [lo_p, hi_p], "flanks_attached": int(flanks),
                "type": tname, "lot": [int(w_cols), int(d_cols)]}
        if got and got.get("selected") and got.get("selected") != tname:
            # a demand that refused to answer about the selected leaf: kept verbatim
            base["selected_refused"] = got.get("why")
        if fit is not None and fit < hi_p:
            return {**base, "storeys_admitted": int(fit), "restricts": True,
                    "why": got.get("why") or
                           f"the {w_cols}x{d_cols} lot admits {fit} storey(s) of "
                           f"{tname} by its envelope with {flanks} flank(s) attached"}
        if got and not got.get("holds", True):
            # **the floor itself does not fit.** The ask stands: the shortfall is
            # recorded so construction emits the constraint and the obligation ledger
            # owns it, rather than the parameter being quietly lowered here.
            return {**base, "storeys_admitted": None, "restricts": True,
                    "required": bool(got.get("required")),
                    "why": got.get("why") or
                           f"a {w_cols}x{d_cols} lot does not admit {lo_p} storey(s) "
                           f"of {tname} with {flanks} flank(s) attached"}
        if fit is not None:
            return {**base, "storeys_admitted": int(fit), "restricts": False,
                    "why": got.get("why") or
                           f"the {w_cols}x{d_cols} lot admits {fit} storey(s) of "
                           f"{tname}, the whole of its band, with {flanks} flank(s) "
                           f"attached"}
        # nothing could answer: recorded as that, and never as agreement
        return {**base, "storeys_admitted": None, "restricts": None,
                "why": (got.get("why") if got else None) or
                       f"no envelope could say what a {w_cols}x{d_cols} lot admits of "
                       f"{tname} with {flanks} flank(s) attached"}

    def leaf(kind, name, tname, decl, u0, u1, v0, v1, front=None, notes="",
             flanks=None):
        x0, z0, x1, z1 = fr.rect(u0, u1, v0, v1)
        n_leaf[0] += 1
        k = n_leaf[0]
        rng = random.Random(f"{seed}/{district['name']}/{k}")
        params = {}
        row_env = None
        fl = _FABRIC_FLANKS if flanks is None else int(flanks)
        for pn, spec_p in (decl.get("params") or {}).items():
            if not isinstance(spec_p, (list, tuple)) or not spec_p:
                continue
            if spec_p[0] == "int" and len(spec_p) >= 3:
                lo_p, hi_p = int(spec_p[1]), int(spec_p[2])
                if pn == "storeys":
                    # **Only the storeys this lot admits are asked** (the expression
                    # round): a cottage asked for two on a lot that holds one emitted
                    # one and the record called the loss a constraint. Where the
                    # envelope module can say what the lot admits, the band is capped to
                    # it and the cap is on the row; the ask and the outcome agree.
                    lo_p, hi_p, got = _admits_storeys(tname, decl, x1 - x0 + 1,
                                                      z1 - z0 + 1, fl)
                    row_env = _envelope_row(tname, lo_p, hi_p, got,
                                            x1 - x0 + 1, z1 - z0 + 1, fl)
                    if row_env and row_env.get("storeys_admitted") is not None:
                        hi_p = max(lo_p, int(row_env["storeys_admitted"]))
                params[pn] = rng.randint(lo_p, hi_p)
            elif spec_p[0] == "choice" and len(spec_p) >= 2 and spec_p[1]:
                params[pn] = rng.choice(list(spec_p[1]))
        # a parameter the character governs is the character's, never a draw
        for ch_key, pname in spec_mod.CHARACTER_PARAMS.items():
            if pname in params:
                params[pname] = int(ch.get(ch_key) or 0)
        row = {"kind": kind, "name": name, "type": tname, "seed": 1 + (k % 89),
               "params": params, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
               "notes": notes, **({"envelope": row_env} if row_env else {})}
        if front:
            row["front"] = front
        # **The level this leaf was designed to stand on.** The block design round, and
        # the audit's first cause. A district's ground decision has a level -- the
        # terrace `placeplan.district_ground` chose and `stage_terraces` cut to -- and
        # until now nothing carried it to the builder, so `Builder._decide_rect` took
        # its floor from whatever doorstep the circulation pass had reserved and
        # `_within_reach` let that doorstep pull the floor three blocks down. That is
        # `lower_ring_north_2_b1_0_00`: floor y=63 over ground y=66, its stair facing
        # down-slope, and neither preceding decision had refused it. The floor is the
        # **plot's own prepared ground**, and a doorstep may move it by a step and not
        # by a storey. Construction realizes the difference as a flight
        # (`Builder.approach`), which is what a step or a ramp is for.
        if kind == "plot" and _design_level is not None:
            row["floor"] = int(_design_level)
        return row

    def settle_storeys(row, face, ch_local=None) -> tuple:
        """**Admissibility governs the final parameters.** The neighbourhood delivery
                round, and the audit's first cause in its second half.

                A lot's storeys were drawn in `leaf()` under the fabric's intended attachment,
                and then `lots_along`'s "no two neighbours are the same building" step moved
                `params.storeys` *afterwards* -- outside the envelope that had just been asked.
                The delivered candidate records `storeys_admitted: 1` beside `params.storeys: 2`
                on the same leaf, which is the plan disagreeing with itself in writing.

                Two things are wrong with doing it there and both are fixed by doing it here.
                The flanks are not known until the row is complete: a lot dropped for the road
                or for a standing part frees its neighbour's flank, and an end lot of a terrace
                has one free flank whatever the fabric intended. And the variation has to move
                *inside* what the lot admits rather than over the top of it.

                So: for each leaf of a finished row, the envelope is asked again in the flank
                configuration that leaf is really in, the record is rewritten to that answer,
                the parameter is clamped into it, and only then is a neighbour that would
                otherwise be the same building stepped -- to another admitted height, or not at
                all. `face` is the shape list the rhythm measure reads and is kept in step.

                Returns `(moved, refused)`: how many leaves this changed, for the record, and
                the row items refused -- **the quarter design round**. Where the district's
                ground has a design level, each leaf is also **sited** here (`site_solve`, with
                the flanks it really has) and its envelope is asked about the pad it will be
                built on; a leaf whose own envelope admits fewer storeys than its type's floor
                is **not** clamped up to that floor and emitted -- it is re-typed to another
                type of the same use that its lot admits, or refused, and the refusal is
                unresolved demand on the district's record (`demand_short`, `lots_refused`).
                
        """
        moved = 0
        refused = []
        ch_use = ch_local if ch_local is not None else ch
        for q, item in enumerate(row):
            p = item[2]
            tname = item[3] if len(item) > 3 else p.get("type")
            decl = item[4] if len(item) > 4 else None
            if decl is None or p.get("kind", "plot") != "plot":
                continue
            ww = p["x1"] - p["x0"] + 1
            dd = p["z1"] - p["z0"] + 1
            ew, ed = ww, dd
            if _design_level is not None:
                st, why = site_solve(p["x0"], p["z0"], p["x1"], p["z1"], p.get("front"),
                                     p.get("attached") or [], pad_need(decl), decl,
                                     inset=p.get("inset"))
                if st is None:
                    refuse_lot(p, tname, why, refused, item)
                    continue
                p["site"] = st
                ew, ed = _envelope_lot(p, st)
            if p.get("form") and _form_plan_fn(tname, decl) is not None:
                # **the form this lot owes, asked of the pad it will be built on** with
                # the flanks it really has. A lot that cannot hold it is refused -- the
                # demand goes back to the layout -- never built lower or re-typed to a
                # type that owes less.
                fp_ = _form_of_pad(p, tname, decl)
                p["form_plan"] = {k_: fp_.get(k_) for k_ in
                                  ("ok", "storeys", "court", "clear", "need", "pad", "why")
                                  if fp_.get(k_) is not None}
                if not fp_.get("ok"):
                    refuse_lot(p, tname, "form", refused, item, env={"why": fp_.get("why"),
                                                                      "need": fp_.get("need")})
                    continue
                if "storeys" in (p.get("params") or {}):
                    st_n = int(fp_.get("storeys") or p["params"]["storeys"])
                    p["params"]["storeys"] = st_n
                    p["envelope"] = {"storeys_band": [st_n, st_n],
                                     "flanks_attached": len(p.get("attached") or ()),
                                     "type": tname, "lot": [int(ew), int(ed)],
                                     "storeys_admitted": st_n, "restricts": False,
                                     "by": "form_plan", "why": fp_.get("why")}
                if q < len(face):
                    face[q] = (tname, ww, dd, int((p.get("params") or {}).get("storeys", 1)))
                continue
            if "storeys" not in (p.get("params") or {}):
                continue
            fl = len(p.get("attached") or ())
            lo_s, hi_s, got = _admits_storeys(tname, decl, ew, ed, fl)
            env = _envelope_row(tname, lo_s, hi_s, got, ew, ed, fl)
            p["envelope"] = env
            if _design_level is not None and env.get("storeys_admitted") is not None \
                    and int(env["storeys_admitted"]) < lo_s:
                # **a two-storey minimum beside a one-storey admission is unresolved
                # demand, not a buildable leaf**
                alt = retype_lot(p, tname, decl, env)
                if alt is None:
                    refuse_lot(p, tname, "storeys", refused, item, env=env)
                    continue
                tname, decl, lo_s, hi_s, env = alt
                row[q] = (item[0], item[1], p, tname, decl)
            top = hi_s
            if env.get("storeys_admitted") is not None:
                top = max(lo_s, int(env["storeys_admitted"]))
            elif env.get("restricts"):
                # the floor does not stand here: the ask stays the floor and the
                # shortfall is the constraint, as `_envelope_row` records
                top = lo_s
            # `restricts is None` is "nothing could answer", which is not a restriction
            # and must not become a smaller default: the band stands as the band.
            was = int(p["params"]["storeys"])
            now = max(lo_s, min(was, top))
            # **and no two neighbours are the same building** -- the one lever a terrace
            # of a single attached type at a single width still has, taken inside the
            # admitted band and never outside it.
            shape = (tname, ww, dd, now)
            prev = face[q - 1] if 0 < q <= len(face) else None
            if prev is not None and prev == shape and top > lo_s:
                now = lo_s + ((now - lo_s + 1) % (top - lo_s + 1))
                shape = (tname, ww, dd, now)
            if now != was:
                p["params"]["storeys"] = now
                moved += 1
            if q < len(face):
                face[q] = shape
        return moved, refused

    def _envelope_lot(p, st) -> tuple:
        """The lot the envelope is asked about: the plot itself where the site's pad is
        its whole inset, else the lot that pad is equivalent to."""
        from .buildlib import site_pad_rect
        full = list(site_pad_rect(p["x0"], p["z0"], p["x1"], p["z1"],
                                  p.get("attached") or [],
                                  *([int(p["inset"])] if p.get("inset") is not None
                                    else [])))
        if list(st["pad"]) == full:
            return p["x1"] - p["x0"] + 1, p["z1"] - p["z0"] + 1
        return site_lot(st["pad"], p.get("attached") or [])

    def refuse_lot(p, tname, why, refused, item, env=None) -> None:
        """Record a drawn lot refused on its site or its storeys; the caller removes it."""
        sites_refused[why if why in sites_refused else "pad"] += 1
        row_r = {"name": p.get("name"), "type": tname,
                 "rect": [p["x0"], p["z0"], p["x1"], p["z1"]], "refused_on": why,
                 "attached": list(p.get("attached") or []),
                 "why": ({"pad": "no pad of this type's footprint is all ground this "
                                 "design prepares, touching its street edge",
                          "entrance": "no door on the pad's street edge has a prepared "
                                      "line out to the street",
                          "form": (f"its pad does not hold the form its use owes: "
                                   f"{(env or {}).get('why')}"),
                          "storeys": (f"its envelope admits "
                                      f"{(env or {}).get('storeys_admitted')} storey(s) "
                                      f"against the type's floor of "
                                      f"{((env or {}).get('storeys_band') or ['?'])[0]}, "
                                      f"and no other type of its use stands here")
                          }.get(why, str(why)))}
        if env:
            row_r["envelope"] = dict(env)
        lots_refused.append(row_r)
        if why == "form":
            got = next((x for x in demand_short if x.get("what") == "form"
                        and x.get("subject") == tname), None)
            if got is None:
                demand_short.append({
                    "subject": tname, "what": "form", "kind": "plot", "required": True,
                    "form": dict(p.get("form") or {}), "lots": 1,
                    "available": [p["x1"] - p["x0"] + 1, p["z1"] - p["z0"] + 1],
                    "needed_pad": (env or {}).get("need"),
                    "why": (f"a lot of {tname} could not hold the form its use owes "
                            f"({(env or {}).get('why')}); refused rather than built short"),
                    "alternatives": [{"what": "lot", "owner": "layout",
                                      "why": "a larger lot for this use, or another "
                                             "allocation"}]})
            else:
                got["lots"] = int(got.get("lots") or 1) + 1
        if why == "storeys":
            # **unresolved demand, returned to the district**, one row per type
            got = next((x for x in demand_short if x.get("what") == "storeys"
                        and x.get("subject") == tname), None)
            if got is None:
                demand_short.append({
                    "subject": tname, "what": "storeys", "kind": "plot",
                    "required": bool((env or {}).get("required")),
                    "requirement": (dem or {}).get("requirements") if dem else None,
                    "least": ((env or {}).get("storeys_band") or [None])[0],
                    "lots": 1, "available": [p["x1"] - p["x0"] + 1,
                                             p["z1"] - p["z0"] + 1],
                    "needed": None, "needed_columns": None,
                    "why": (f"a lot of {tname} admits "
                            f"{(env or {}).get('storeys_admitted')} storey(s) against "
                            f"its floor of "
                            f"{((env or {}).get('storeys_band') or ['?'])[0]} and no "
                            f"other type of its use stands on it: the lot was refused "
                            f"rather than emitted at storeys it does not admit"),
                    "alternatives": [{"what": "lot", "owner": "layout",
                                      "why": "a deeper or wider lot for this type, or "
                                             "another block for this use"}]})
            else:
                got["lots"] = int(got.get("lots") or 1) + 1
        refused.append(item)

    def retype_lot(p, tname, decl, env):
        """Another plot type **of the same use** that this lot admits at storeys its
        own floor allows, or None. Rewrites the leaf in place when it finds one and
        returns `(tname, decl, lo, hi, envelope)`."""
        def use_of(n, d):
            u = mix["use_of"].get(str(n)) or type_use(n, d)[0]
            return mix["primary"] if u == UNSTATED_USE else u
        want_use = use_of(tname, decl)
        a0, a1, b0, b1 = _uv_of(p)
        fl = len(p.get("attached") or ())
        cands = sorted(((n, d) for n, d in (decls or {}).items()
                        if n != tname and isinstance(d, dict)
                        and d.get("kind", "plot") == "plot"
                        and use_of(n, d) == want_use
                        and (not fl or d.get("attached"))),
                       key=lambda nd: (_seeded(seed, "retype", p.get("name"), nd[0]),
                                       nd[0]))
        for n2, d2 in cands:
            try:
                if not _admits(d2, a1 - a0 + 1, b1 - b0 + 1, p.get("attached") or None,
                               fr.u_is_x):
                    continue
            except Exception:                    # noqa: BLE001 -- no envelope, no rule
                continue
            st2, _w = site_solve(p["x0"], p["z0"], p["x1"], p["z1"], p.get("front"),
                                 p.get("attached") or [], pad_need(d2), d2,
                                 inset=p.get("inset"))
            if st2 is None:
                continue
            ew, ed = _envelope_lot(p, st2)
            try:
                lo2, hi2, got2 = _admits_storeys(n2, d2, ew, ed, fl)
            except Exception:                    # noqa: BLE001 -- cannot answer, skip
                continue
            env2 = _envelope_row(n2, lo2, hi2, got2, ew, ed, fl)
            if env2.get("storeys_admitted") is None or \
                    int(env2["storeys_admitted"]) < lo2:
                continue
            rng2 = random.Random(f"{seed}/{district['name']}/{p.get('name')}/{n2}")
            params2 = {}
            for pn, spec_p in (d2.get("params") or {}).items():
                if not isinstance(spec_p, (list, tuple)) or not spec_p:
                    continue
                if spec_p[0] == "int" and len(spec_p) >= 3:
                    if pn == "storeys":
                        params2[pn] = rng2.randint(lo2, max(lo2, min(
                            hi2, int(env2["storeys_admitted"]))))
                    else:
                        params2[pn] = rng2.randint(int(spec_p[1]), int(spec_p[2]))
                elif spec_p[0] == "choice" and len(spec_p) >= 2 and spec_p[1]:
                    params2[pn] = rng2.choice(list(spec_p[1]))
            for ch_key, pname in spec_mod.CHARACTER_PARAMS.items():
                if pname in params2:
                    params2[pname] = int(ch.get(ch_key) or 0)
            retyped.append({"name": p.get("name"), "from": tname, "to": n2,
                            "use": want_use, "was_envelope": dict(env),
                            "why": (f"{tname} admits {env.get('storeys_admitted')} "
                                    f"storey(s) here against its floor of "
                                    f"{(env.get('storeys_band') or ['?'])[0]}; {n2} is "
                                    f"the same use and admits "
                                    f"{env2['storeys_admitted']}")})
            p["type"] = n2
            p["params"] = params2
            p["envelope"] = env2
            p["site"] = st2
            p["retyped"] = {"from": tname, "why": retyped[-1]["why"]}
            p["notes"] = (str(p.get("notes") or "").replace(str(tname), str(n2))
                          + f" (re-typed from {tname}: its storeys did not stand here)")
            return n2, d2, lo2, hi2, env2
        return None

    #: **Does this type stand on a `w` x `d` lot here?** In the district's frame, and in
    #: every configuration of the flanks a party wall may find itself in -- both
    #: neighbours, either one, neither -- because a neighbour dropped for the road or
    #: for a standing part frees a flank and the pad's inset comes back on it. The emit
    #: path asked the unattached question only, and a row of party walls on a district
    #: whose streets run the other way drew fifty-one lots the plan validator refused.
    _FLANKS = fr.flanks()
    _ATTS = ([tuple(_FLANKS), _FLANKS[:1], _FLANKS[1:], ()] if attached else [()])

    def stands(dcl, a, b) -> bool:
        return all(_admits(dcl, a, b, att, fr.u_is_x) for att in _ATTS)

    def pad_need(dcl) -> tuple | None:
        """The smallest pad this type declares it can build on, or None.

                `needs.footprint` is `(lo_w, lo_d, hi_w, hi_d)` in **pad** columns -- what
                `build()` is handed -- and its first pair is the floor. `pad_founded` asks whether
                a rectangle that size is all prepared ground somewhere inside the plot's pad; a
                type that declares no footprint is not asked.
                
        """
        fp = ((dcl or {}).get("needs") or {}).get("footprint")
        if not fp or len(fp) < 2:
            return None
        try:
            return (int(fp[0]), int(fp[1]))
        except (TypeError, ValueError):
            return None

    def lots_along(u0, u1, v0, v1, front, i, j, r, plots, depth=None):
        """A row of lots along one street face of a block, in [u0, u1] x [v0, v1].

                **A street is a rhythm and not a comb**, the craft round (E3). The run used to
                be divided evenly, so every lot was one width, every building the same type and
                every roof the same height -- twenty-one cottages on identical pads. Each lot
                now draws its own width and its own depth inside the band the character sets
                (`spread`), and its own type from everything the role admits at that size; the
                widths still tile the run exactly, because the last lot takes what is left.

                **...and a street of another use has the grain of that use**, the neighbourhood
                round. The face that fronts the quarter's principal street is drawn from the work
                this district's design names (`_principal_face`), and a shop is not a courtyard
                house: it stands on its own lot inside its own envelope. Measured on the retained
                city, and the reason this exists at all -- the middle ring's lot is 13x13 and
                `shop_house`'s sweep found 13 broken, so a shop street held at the quarter's own
                lot came back with **no shops at all** and nothing said why.
                
        """
        if budget_left() <= 0:
            return      # the district has the houses it was asked for, or asked none,
            # or the rest are reserved for a composition that has not been laid
        # **which street this is, and what it is a street of**
        on_street = bool(street_lot) and _principal_face(i, j, front)
        lw, dd0 = (w, ld if depth is None else depth)
        face_decl = house
        if on_street:
            lw, td = street_lot
            face_decl = street_decl
            if depth is None:
                # **The principal street's depth is the street type's, unless that costs
                # the quarter's own house a storey it admits.** The neighbourhood
                # delivery round, found by the registered rhythm bar. The work a quarter
                # is for stands on its main street and is sized for its own type, and
                # narrowing the whole face to that depth took the row house from 13 deep
                # to 12 -- which is exactly the column between two storeys and one, so
                # the main street of the quarter came out at a single height while the
                # back streets had two. Where the street type stands at the deeper lot
                # as well, the face keeps it and both uses get what they admit.
                keep = (_keeps_storeys(house_name, house, w, dd0, min(dd0, td))
                        and stands(face_decl, lw, dd0))
                dd0 = dd0 if keep else min(dd0, td)
            else:
                dd0 = (_deepest(face_decl, lw, min(dd0, depth), (), fr.u_is_x) or td)
        n = u1 - u0 + 1
        k = (n + gap) // (lw + gap)
        if k < 1:
            return
        lo, hi, _ex = _plot_range(face_decl)
        # the street is at the low-v edge of this row where `front` is the low-v side; a
        # lot shallower than the row keeps its face on the street and gives its back
        at_low = (front == fr.front(True))
        #: **Which side this row's doorsteps go on**, for the hard entrance condition in
        #: `free`. A district of `open` frontage has no street edge to be entered from
        #: -- its lots stand apart in their own ground and a person walks up to
        #: whichever face the house puts its door on -- so it names no side and the
        #: condition does not arise. Everywhere else a door is on the street, and that
        #: is the face the design has to have prepared.
        _entry_side = None if open_front else front
        widths = [lw] * k
        left = n - (k * lw + (k - 1) * gap)
        if 0 < left <= WIDEN_UP_TO * n and not attached and not ceiling_binds:
            # widen the lots to take the remainder up, inside the face type's envelope
            per = left // k
            for q in range(k):
                widths[q] = min(hi, lw + per)
            for q in range(left - per * k):
                widths[q] = min(hi, widths[q] + 1)
            widths = [ww if stands(face_decl, ww, dd0) else lw for ww in widths]
        # ...and then each lot is given or takes back a few columns of its own, the run
        # coming out the same length: a rhythm, and the same houses.
        if spread and k > 1:
            room = sum(widths)
            for q in range(k):
                step = int(round((_seeded(seed, "wide", i, j, r, q) * 2 - 1) * spread))
                widths[q] = max(lo, min(hi, widths[q] + step))
            over = sum(widths) - room
            q = 0
            while over and q < 4 * k:              # give the drift back, evenly
                idx = q % k
                if over > 0 and widths[idx] > lo:
                    widths[idx] -= 1
                    over -= 1
                elif over < 0 and widths[idx] < hi:
                    widths[idx] += 1
                    over += 1
                q += 1
        at = u0
        row = []
        face: list = []
        faces.append(face)
        for q, ww in enumerate(widths):
            ua, ub = at, at + ww - 1
            at += ww + gap
            dd = dd0
            # **The depth varies where the ground behind a lot is open ground anyway.**
            # On a street the lots stand back to back and what a shallow one gives up is
            # the clearance between two rows -- ground no rule can assign, and a fifth
            # of a district went undeveloped the first time this varied everywhere. On
            # open frontage the leftover is verge ground the fill reaches.
            if spread and open_front:
                give = int(round(_seeded(seed, "deep", i, j, r, q) * spread))
                dd = max(3, dd0 - give)
            va, vb = (v0, v0 + dd - 1) if at_low else (v1 - dd + 1, v1)
            if not free(ua, ub, va, vb, front=_entry_side):
                dropped[why_dropped(ua, ub, va, vb, _entry_side)] += 1
                continue
            # **the type is the lot's own**: everything the role admits at this size,
            # the role's own house type weighted so a quarter still reads as its own,
            # and `OTHER_EVERY` as the floor that puts a second type on a short street
            admits = [(n2, d2) for n2, d2 in others if stands(d2, ww, dd)]
            # **Half the street is the district's own house type, the rest is the
            # quarter's other uses of the same kind.** A street of forty-five townhouses
            # is one building forty-five times; a street of forty-five that came back
            # with fifteen halls in it is a civic precinct. `others` is now the
            # quarter's own use only (`use_mix`), so the weighting that used to hold a
            # foreign role to half a vote is gone: everything in this pool is a building
            # the quarter is for, and what varies between them is which dwelling or shop
            # it is.
            pool = [(house_name, house)] * max(1, len(admits))
            for n2, d2 in admits:
                pool.append((n2, d2))
            if others and (counts["lots"] % OTHER_EVERY) == OTHER_EVERY - 1:
                pool = [(n2, d2) for n2, d2 in others if stands(d2, ww, dd)] \
                    or [(house_name, house)]
            # **...and the quarter's own work, on the street it belongs on.** This was
            # `PROGRAMME_USE_SHARE` of the district's lots, spread evenly, and the
            # neighbourhood round's brief says why that had to go: a fixed tenth is a
            # universal quota, and a quota is not a programme. What a quarter's
            # programme actually says is *what* its work is and *where* it goes -- and
            # where trade goes is the street its market is on. So the faces that front
            # the principal street (`_principal_face`) are drawn from the work this
            # district's own design names, and nothing else is. A district whose design
            # names no work never reaches this line, and the share it comes to is
            # measured off what was laid rather than set here. A use a **requirement**
            # asks for comes first and is laid at the count the requirement asks -- one
            # temple where the demand requires a temple, not a tenth of the street in
            # temples -- and takes the best lots of the principal street, which is where
            # a building the request requires belongs.
            if owed and on_street:
                here = [(n2, d2, rq) for n2, d2, rq in owed if stands(d2, ww, dd)]
                if here:
                    n2, d2, rq = here[0]
                    owed.remove((n2, d2, rq))
                    pool = [(n2, d2)]
                    counts["programme_uses"] = counts.get("programme_uses", 0) + 1
            elif work and on_street:
                here = [(n2, d2) for n2, d2 in work if stands(d2, ww, dd)]
                if here:
                    pool = here
                    counts["programme_uses"] = counts.get("programme_uses", 0) + 1
            tname, decl = pool[int(_seeded(seed, "type", i, j, r, q) * len(pool))
                               % len(pool)]
            if not stands(decl, ww, dd):
                # the quarter's house first, and the face's own type after it **only on
                # the face that is that use's street**. Offering the street type as a
                # fallback everywhere put three shops on a district whose grid is one
                # block row deep and therefore has no principal street at all: a
                # fallback is not a programme, and a use with no street of its own does
                # not get one through the back of a `stands()` check.
                for _n2, _d2 in ([(house_name, house)]
                                 + ([(street_name, street_decl)] if on_street else [])):
                    if _d2 is not None and stands(_d2, ww, dd):
                        tname, decl = _n2, _d2
                        break
            if not stands(decl, ww, dd):
                dd = dd0
                va, vb = (v0, v0 + dd - 1) if at_low else (v1 - dd + 1, v1)
                if not stands(decl, ww, dd) or not free(ua, ub, va, vb,
                                                       front=_entry_side):
                    continue
            # **and the ground can carry the pad this type needs.** The block design
            # round, found by building the section: `middle_ring_north_east_b0_1_01` is
            # a `court_large` on a 13x13 lot whose ground runs y=64..73, `founded`'s
            # share tests admitted it, `site()` cut the flattest 7x7 it could out of the
            # broken half and the type refused it by name -- leaving a platform and an
            # approach with nothing on them (`E010`, eight blocks held up by nothing).
            # The lot is admitted for the type and the *ground* is admitted for its pad.
            if _design_level is not None:
                # **the pad and the way in this lot will be built with**, chosen once
                # (`site_solve`) under the attachment the fabric intends; the leaf's own
                # is settled with its row (`settle_storeys`) and is a subset of this pad
                _st, _why = site_solve(*fr.rect(ua, ub, va, vb),
                                       (faces_side or None) if open_front else front,
                                       list(fr.flanks()) if attached else [],
                                       pad_need(decl), decl)
                if _st is None:
                    dropped["entrance" if _why == "entrance" else "pad"] += 1
                    continue
            elif not free(ua, ub, va, vb, front=_entry_side, pad=pad_need(decl)):
                dropped[why_dropped(ua, ub, va, vb, _entry_side,
                                    pad=pad_need(decl))] += 1
                continue
            if budget_left() <= 0:
                break
            # **An open-frontage lot records a front where the design demands one.**
            # `open` frontage means there is no street edge to build to, and it used to
            # mean the plan recorded no orientation at all -- so a district asked to
            # face the water laid houses whose facing nothing had decided and nothing
            # could check. Which way a house faces and whether it stands on a street are
            # two questions; a frontage obligation answers the first either way.
            _front = (faces_side or None) if open_front else front
            # the row carries the type and its declaration beside the leaf, because
            # `settle_storeys` below has to ask the envelope again once the flanks of
            # this lot are known and the pool means the type is the lot's own
            row.append((ua, ub, leaf("plot", f"b{i}_{j}_{r}{q}", tname, decl, ua, ub,
                                     va, vb, front=_front,
                                     notes=(f"a {tname} fronting the street on its "
                                            f"{front} side") if not open_front else
                                           (f"a {tname} on open ground, facing "
                                            f"{_front}" if _front else
                                            f"a {tname} on open ground")),
                        tname, decl))
            got_leaf = row[-1][2]
            plots.append(got_leaf)
            # the shape the rhythm measure reads; `settle_storeys` rewrites the storey
            # of each entry once the row's own attachment is settled
            face.append((tname, ww, dd,
                         int((got_leaf.get("params") or {}).get("storeys", 1))))
            mark(ua, ub, va, vb, "lot")
            counts["lots"] += 1
        # **The second stone, at the share the library registers, by the street.** The
        # craft round, E3: `TypeBuilder` decides one part at a time off a hash and a
        # street is a distribution, so the compiler -- which can see the whole face --
        # says which lots are faced in the voice's `wall_alt` and spreads them evenly,
        # because a quarter of a street clumped at one end is not a quarter of a street.
        if row:
            # the share is the **district's** and not the face's: a face of three lots
            # rounded on its own gives one, which is a third, and every face in the
            # district does the same. The debt carries over.
            alt_debt[0] += WALL_ALT_SHARE * len(row)
            n_alt = int(alt_debt[0])
            alt_debt[0] -= n_alt
            if n_alt:
                step = len(row) / float(n_alt)
                phase = _seeded(seed, "alt", i, j, r) * step
                for t_i in range(n_alt):
                    row[min(len(row) - 1, int(phase + t_i * step))][2]["wall_alt"] = True
            for item in row:
                item[2].setdefault("wall_alt", False)
        # **...and only now are the storeys settled.** The flanks are known, so the
        # envelope can be asked the question each lot is actually in -- an end of the
        # terrace has one free flank and admits what that pad admits -- and the
        # variation between neighbours moves inside the answer instead of over it.
        settle_row(row, face, plots, attach=bool(attached))

    def attach_row(row) -> None:
        """v2, C2: the sides a lot's neighbour actually stands against, after the
        drops -- a party wall to each; the pad reaches the plot's edge there."""
        low, high = fr.flanks()
        for q, item in enumerate(row):
            ua, ub, p = item[0], item[1], item[2]
            sides = []
            if q > 0 and row[q - 1][1] + 1 == ua:
                sides.append(low)
            if q + 1 < len(row) and row[q + 1][0] == ub + 1:
                sides.append(high)
            counts["party_walls"] += len(sides) - len(p.get("attached") or ())
            p["attached"] = sides

    def settle_row(row, face, plots, attach: bool) -> None:
        """Attach, settle and site a finished row; a lot refused on its own site or
        storeys leaves the row, its ground goes back, and its neighbours are settled
        again with the flank it freed. `face` is kept in step with `row`."""
        for _round in range(len(row) + 1):
            if attach:
                attach_row(row)
            moved, refused = settle_storeys(row, face)
            counts["storeys_settled"] = counts.get("storeys_settled", 0) + moved
            if not refused:
                return
            for item in refused:
                q = next(k for k, it in enumerate(row) if it[2] is item[2])
                del row[q]
                if q < len(face):
                    del face[q]
                p = item[2]
                if p in plots:
                    plots.remove(p)
                counts["party_walls"] -= len(p.get("attached") or ())
                counts["lots"] -= 1
                a0, a1, b0, b1 = _uv_of(p)
                ledger[a0:a1 + 1, b0:b1 + 1] = L["undeveloped"]

    def _uv_of(p) -> tuple:
        """A leaf's rectangle back in this district's (u, v) frame."""
        if fr.u_is_x:
            return (p["x0"] - fr.x0, p["x1"] - fr.x0, p["z0"] - fr.z0, p["z1"] - fr.z0)
        return (p["z0"] - fr.z0, p["z1"] - fr.z0, p["x0"] - fr.x0, p["x1"] - fr.x0)

    def lots_across(v0, v1, u0, u1, front, i, j, r, plots):
        """**A row of lots along one *cross* street of a block**: the run goes across the
                district's grain (along v) and the lots are `ld` deep into the block (along u).

                The neighbourhood round's `perimeter` arrangement needs this and nothing in this
                compiler has ever had it -- every lot ever laid here fronts one of the two long
                faces of its block, so the short faces of every block in every district this
                project has built are blank ground and the cross streets have nothing standing on
                them. A perimeter block is four faces and a court, and these are the other two.

                Deliberately the plain version of `lots_along`: an even division of the run, no
                `spread`, and **detached whatever the district is**, because a party wall is a
                statement about a lot's flanks along the *grain* and the end of a terrace is
                where the terrace stops. What it does share is the leaf, the clearance, the free
                test, the wall-alt share and the type draw, so a corner building is an ordinary
                building of this quarter.
                
        """
        if budget_left() <= 0 or v1 < v0 or u1 < u0:
            return
        at_low_u = (front == fr.flanks()[0])
        deep = min(ld, u1 - u0 + 1)
        n = v1 - v0 + 1
        # the cross face is detached: its own clearance between lots, and a lot of at
        # least the house's least side
        c_gap = LOT_GAP
        lo, hi, _ex = _plot_range(house)
        k = (n + c_gap) // (max(lo, min(hi, w)) + c_gap)
        if k < 1:
            return
        wid = (n - (k - 1) * c_gap) // k
        wid = max(lo, min(hi, wid))
        row = []
        face: list = []
        faces.append(face)
        at = v0
        for q in range(k):
            va, vb = at, at + wid - 1
            at += wid + c_gap
            if vb > v1:
                break
            dd = deep
            while dd >= 3 and not _admits(house, dd, wid, None, fr.u_is_x):
                dd -= 1
            if dd < 3:
                return
            ua, ub = (u0, u0 + dd - 1) if at_low_u else (u1 - dd + 1, u1)
            if not free(ua, ub, va, vb, front=front):
                dropped[why_dropped(ua, ub, va, vb, front)] += 1
                continue
            here = [(n2, d2) for n2, d2 in ([(house_name, house)] + others)
                    if _admits(d2, dd, wid, None, fr.u_is_x)]
            tname, decl = here[int(_seeded(seed, "cross", i, j, r, q) * len(here))
                               % len(here)] if here else (house_name, house)
            if _design_level is not None:
                _st, _why = site_solve(*fr.rect(ua, ub, va, vb), front, [],
                                       pad_need(decl), decl)
                if _st is None:
                    dropped["entrance" if _why == "entrance" else "pad"] += 1
                    continue
            elif not free(ua, ub, va, vb, front=front, pad=pad_need(decl)):
                dropped[why_dropped(ua, ub, va, vb, front, pad=pad_need(decl))] += 1
                continue
            if budget_left() <= 0:
                break
            # the cross face is detached whatever the district is (see the docstring),
            # so its lots are drawn and settled with no flank attached
            row.append((va, vb, leaf("plot", f"x{i}_{j}_{r}{q}", tname, decl,
                                     ua, ub, va, vb, front=front, flanks=0,
                                     notes=f"a {tname} on the cross street at this "
                                           f"block's {front} end"),
                        tname, decl))
            plots.append(row[-1][2])
            face.append((tname, wid, dd,
                         int((row[-1][2].get("params") or {}).get("storeys", 1))))
            mark(ua, ub, va, vb, "lot")
            counts["lots"] += 1
        if row:
            alt_debt[0] += WALL_ALT_SHARE * len(row)
            n_alt = int(alt_debt[0])
            alt_debt[0] -= n_alt
            if n_alt:
                step = len(row) / float(n_alt)
                phase = _seeded(seed, "alt", i, j, r) * step
                for t_i in range(n_alt):
                    row[min(len(row) - 1, int(phase + t_i * step))][2]["wall_alt"] = True
            for item in row:
                item[2].setdefault("wall_alt", False)
        settle_row(row, face, plots, attach=False)

    def areas_over(u0, u1, v0, v1, tname, what, plots, note, prefix):
        decl = areas.get(tname)
        if decl is None or u1 < u0 or v1 < v0:
            return 0
        n = 0
        for (ua, ub, va, vb) in _area_tiles(u0, u1, v0, v1, decl, AREA_GAP):
            if not free(ua, ub, va, vb):
                dropped[why_dropped(ua, ub, va, vb)] += 1
                continue
            # an area stands in its own ground: no flank of it is a party wall
            plots.append(leaf("area", f"{prefix}_{n}", tname, decl, ua, ub, va, vb,
                              notes=note, flanks=0))
            mark(ua, ub, va, vb, what)
            n += 1
        return n

    #: **How wide the way into a court is.** A court a person cannot walk into is a
    #: light well between somebody's back walls, and four ranges of houses round a
    #: rectangle leave no way in at all -- which is what `perimeter` built and what
    #: `usable.court_accessible` would have to answer `cut_off` for. One lot of the
    #: street range is given up to a passage: wide enough for the lane's own width and
    #: its clearance, narrow enough that the street front still reads as continuous.
    COURT_ENTRY = max(3, LOT_GAP + 2)

    #: The least a court may measure across and still be a court somebody uses rather
    #: than a light well between two ranges' back walls.
    COURT_LEAST = 5

    #: **How much of a court's face has to carry building for the face to be there.**
    #: The block design round, after an independent reader measured the delivered pair:
    #: the gap test alone -- "no unbroken run of uncovered columns wider than the
    #: fabric's own clearance" -- passes an eight-column face with two columns covered,
    #: in the pattern `..X...X.`. A hole test is the right question about a passage and
    #: the wrong one about a wall. Half, because a court whose faces are half building
    #: and half the clearance between two lots is the ordinary urban block and a bar
    #: above that would refuse every fabric that leaves its houses a gap.
    COURT_FACE_COVER = 0.5

    def court_range_depth(u0, u1, v0, v1) -> tuple:
        """**How deep the ranges round a court are**: `(rd_front, rd_back, cd)` -- the
                depth of the street range, of the back range, and of the two end ranges across
                the block's grain. `rd_front` is 0 where this block cannot hold the composition
                at all; `cd` is 0 where it can hold the court between two ranges but has no width
                for the ends.

                The neighbourhood round's `perimeter` arrangement asked for the district's own
                lot depth on all four sides, and therefore needed a block a whole third row
                deeper *and* wider than the fabric -- which no block of a two-row dense ring is,
                so on this city every court fell through to the back-row form the audit is
                about. A range round a court is a range of *this* type at a depth this type
                admits; it does not have to be the deepest lot the density would otherwise draw.
                Each depth is therefore the deepest its own axis leaves room for once the court
                and the clearances are taken out of it, capped at the fabric's own `ld`.

                A block wide enough for two end ranges gets a court enclosed on four sides; one
                that is not gets a court enclosed on two, open at its ends to the cross streets,
                which is a close rather than a courtyard block. Both are composed; the record
                says which was laid (`perimeter_shut`/`perimeter_open`).
                
        """
        V, U = v1 - v0 + 1, u1 - u0 + 1
        # **...and a range round a court still has to carry the height its band asks
        # for.** The neighbourhood delivery round, found by the rhythm bar: this branch
        # takes depth out of the ranges to make room for the court, and taking one
        # column too many takes the second storey off every house in the range -- which
        # on a terrace of one width is the whole of its variation. `row_house` on a
        # 6-wide attached lot admits 2 storeys at 13 deep and 1 at 12, so a court that
        # costs its ranges that column costs the street its skyline. The depth is chosen
        # as the deepest the block leaves room for **that keeps what `ld` admits**, and
        # only where no such depth exists does it take what stands.
        _band = _storeys_band(house, ch)
        _fl = 2 if attached else 0

        def _admits_here(d2) -> bool:
            # **The emitter's own test, and not a weaker one.** The block design round,
            # found by composing this city's first court block: this asked `_admits` in
            # the both-flanks-attached configuration only, and `lots_along` asks
            # `stands()` -- every configuration a party wall can be in, because the two
            # ends of a terrace have a free flank whatever the fabric intended. So a
            # 6x13/6x5 pair was chosen here (5 deep is admitted between two party walls)
            # and the back range then laid **no lot at all** (6x5 standing free insets
            # to 4x3, which `row_house` refuses). The composition came out four ranges
            # on paper, one range in the world, and was rolled back.
            return stands(house, w, d2)

        def _heights(d2):
            try:
                return int((_storeys_fit(house_name, _band[0], _band[1], w, int(d2),
                                         flanks=_fl) or {}).get("storeys") or _band[0])
            except Exception:                    # noqa: BLE001 -- no envelope, no rule
                return _band[0]
        _want_h = _heights(ld) if _band[1] > _band[0] else None

        def deepest(most):
            top = min(int(ld), int(most))
            keeps = [d2 for d2 in range(top, 2, -1) if _admits_here(d2)
                     and (_want_h is None or _heights(d2) >= _want_h)]
            if keeps:
                return int(keeps[0])
            d2 = top
            while d2 >= 3 and not _admits_here(d2):
                d2 -= 1
            return int(d2) if d2 >= 3 else 0

        # **The two ranges round a court need not be the same depth**, and on most
        # blocks they cannot both be the depth that carries a stair. The neighbourhood
        # delivery round: this block is 36 columns deep, two 13-deep ranges and their
        # clearances take 32 and leave 4 for the court, which is under the least a court
        # may measure; two 12-deep ranges leave 6 and take the second storey off **every
        # house of both ranges**, which is a street at one height and the run of
        # identical neighbours the rhythm bar refuses. One range at 13 and one at 12
        # leaves 5, which is a court, and gives the block two heights. A court block
        # whose two ranges differ in depth is the ordinary urban block and not a
        # compromise. So the pair is searched, best first: most ranges keeping the
        # height their band admits, then deepest, then most room left for the court.
        # **...and the band between them has to hold the end ranges too.** The block
        # design round, found by composing this city's first court block. The pair was
        # chosen for the court's own least measure (`COURT_LEAST`, 5) and the end ranges
        # were then drawn across that band by `lots_across`, which needs a run of at
        # least one lot's **width** -- 6 here. On a 29-deep block the deepest admissible
        # pair leaves 5, `lots_across` computed `k = (5 + 3) // (6 + 3) = 0` and
        # returned before laying anything, so the composition came out as a street
        # range, a back range and a court open at both ends: `perimeter_open`, every
        # time, on every block of this city. Two ranges of eight leave seven, which
        # holds a cross lot and closes the court. So the floor for the band is what the
        # block is actually going to put in it.
        lo_c, hi_c, _ex_c = _plot_range(house)
        cross_wid = max(int(lo_c), min(int(hi_c), int(w)))
        cd_want = deepest((U - 2 * LOT_GAP - COURT_LEAST) // 2)
        if cd_want and U < 2 * cd_want + 2 * LOT_GAP + COURT_LEAST:
            cd_want = 0

        def _pairs(band_least):
            return [(a, b) for a in range(3, int(ld) + 1)
                    for b in range(3, int(ld) + 1)
                    if V - a - b - 2 * row_gap >= band_least
                    and _admits_here(a) and _admits_here(b)]

        # the court enclosed on four sides first; a close open at its ends only where no
        # pair of ranges leaves room for the ends at all
        cd = cd_want
        pairs = _pairs(max(COURT_LEAST, cross_wid)) if cd_want else []
        if not pairs:
            cd = 0
            pairs = _pairs(COURT_LEAST)
        if not pairs:
            return 0, 0, 0
        rd_f, rd_b = max(pairs, key=lambda ab: (
            (0 if _want_h is None else
             (_heights(ab[0]) >= _want_h) + (_heights(ab[1]) >= _want_h)),
            ab[0] + ab[1], min(ab)))
        if not cd and U < COURT_LEAST:
            return 0, 0, 0
        return rd_f, rd_b, cd

    def court_entry_at(u0, u1, v0, mid0, side_least, cu0=None, cu1=None):
        """Where the passage through the street range goes, or None.

                **A third of the way along, and not the middle.** A passage on the block's
                centre line takes the one house that stands opposite the court's own centre, so
                a court enclosed on four ranges measures open to the north at the only column
                anything asks about (`test_neighbourhood_spatial.p6`'s `enclosed_in`, which reads
                the geometry and no field of the plan). An archway a third along a street front
                is the ordinary place for one and leaves the enclosure whole.

                **...and it has to be opposite the court.** The block design round, found by an
                independent reader of the delivered block: on a 40-column block whose end ranges
                are 13 deep the court runs from `u0+16` to `u0+23`, and a third of the way along
                the *block* is `u0+11` -- so the passage was cut through the front range opposite
                the **end range**, and neither of the two delivered passages shared a single
                column with the court it was recorded as the way into. A passage that does not
                open onto the court is a gap in a street front. `cu0`/`cu1` are the court's own
                run and the passage is searched inside it.

                Lifted out of `compose_court_block` so the reservation pass and the emission
                answer the same question about the same block: a need computed against one
                passage and a composition laid against another is two decisions again.
                
        """
        span = int(COURT_ENTRY)
        lo = u0 if cu0 is None else max(u0, int(cu0))
        hi = u1 if cu1 is None else min(u1, int(cu1))
        if hi - lo + 1 < span:
            lo, hi = u0, u1                 # no court run to be opposite: the block's
        centre = lo + (hi - lo + 1) // 3 - span // 2
        for shift in range(0, max(1, (hi - lo) // 2 + span), 2):
            for ea in sorted({centre - shift, centre + shift}):
                eb = ea + span - 1
                if ea <= u0 + int(side_least) or eb >= u1 - int(side_least):
                    continue
                if ea < lo or eb > hi:
                    continue
                if free(ea, eb, v0, mid0 - 1):
                    return (ea, eb)
        return None

    def court_need(u0, u1, v0, v1, side_least, rd, rd_b, cd) -> dict:
        """**What this composition will cost its district, before any of it is laid.**

                The block design round. A court block's four ranges are its children, and a
                composition is adopted -- or refused -- as a whole: this is the number of houses
                the district has to be able to afford for the court between them to be enclosed
                at all. `{"lots", "ranges": {...}, "entry": (ea, eb) | None}`.

                **It counts lots this block can actually carry**, and that is the half that
                makes it a reservation rather than an estimate. A run divided by a lot and a
                clearance says how many lots would fit on empty level ground; the back face of
                `lower_ring_north_2`'s first block is under the arterial road, and a composition
                reserved on the arithmetic laid its street range, found no ground for its back
                range and was rolled back -- having taught the district nothing except that the
                block before it should have had the houses. So each face is walked here with
                `free()`, the same predicate the emitter uses, under the same front: a face with
                no room is zero, and a composition with a zero face is not reserved at all.
                
        """
        mid0, mid1 = v0 + rd + row_gap, v1 - rd_b - row_gap
        cu0 = u0 + cd + LOT_GAP if cd else u0
        cu1 = u1 - cd - LOT_GAP if cd else u1
        entry = court_entry_at(u0, u1, v0, mid0, side_least, cu0, cu1)

        def along_k(a, b, va, vb, front) -> int:
            """How many lots of this fabric `free()` admits along [a, b]."""
            n = b - a + 1
            if n < int(side_least) or n < int(w):
                return 0
            at, got = a, 0
            while at + w - 1 <= b:
                if free(at, at + w - 1, va, vb, front=front, pad=pad_need(house)):
                    got += 1
                at += w + gap
            return got

        f_low, f_high = fr.front(True), fr.front(False)
        if entry:
            ea, eb = entry
            front = (along_k(u0, ea - 1, v0, v0 + rd - 1, f_low)
                     + along_k(eb + 1, u1, v0, v0 + rd - 1, f_low))
        else:
            front = along_k(u0, u1, v0, v0 + rd - 1, f_low)
        back = along_k(u0, u1, v1 - rd_b + 1, v1, f_high)
        ends = 0
        if cd:
            lo_c, hi_c, _ex = _plot_range(house)
            wid = max(int(lo_c), min(int(hi_c), int(w)))
            for (a0, a1, side) in ((u0, u0 + cd - 1, fr.flanks()[0]),
                                   (u1 - cd + 1, u1, fr.flanks()[1])):
                at = mid0
                while at + wid - 1 <= mid1:
                    if free(a0, a1, at, at + wid - 1, front=side, pad=pad_need(house)):
                        ends += 1
                    at += wid + LOT_GAP
        return {"lots": int(front + back + ends),
                "closes": bool(front and back and (ends >= 2 if cd else False)),
                "ranges": {"front": int(front), "back": int(back), "ends": int(ends)},
                "entry": entry}

    def compose_court_block(i, j, u0, u1, v0, v1, plots, side_least, rd, rd_b,
                            cd) -> int:
        """**Buildings and the space between them, composed together.**

                The neighbourhood delivery round's answer to the audit's fourth cause: "a court
                is not simply a type label or the area replacing a housing row". A `courtyard`
                block used to be a block whose *back row* was the court -- open along both short
                faces, the cross streets carrying nothing, and the enclosure its own name
                promises nowhere in the geometry. A `perimeter` block closed all four sides and
                then had no way in.

                What is laid here is the ordinary urban block of most cities that have one: the
                street range with a **passage** through it, the back range, a range at each end,
                and the court they enclose. The passage is ground the ledger owns as street, so
                the circulation pass can route a lane through it to the court's own threshold
                and `usable.court_accessible` has a stance to reach the court from.

                Returns the number of court tiles laid. Records enclosure and entry apart, as
                `perimeter_shut`/`perimeter_open` already keep them apart, because a court is
                enclosed because its four ranges stood and not because an arrangement said so.
                
        """
        # `rd` is the depth of each range round the court (`court_range_depth`), which
        # is the fabric's own `ld` where the block has room for it and shallower where
        # it does not -- a range round a court is a range of this type at a depth this
        # type admits, and a block that has to be a whole third row deeper than the
        # fabric is a block no dense two-row ring has.
        mid0, mid1 = v0 + rd + row_gap, v1 - rd_b - row_gap
        # the passage through the street range, near the middle of the run and moved off
        # anything the block is not free to build on (`court_entry_at`, which the
        # reservation pass asked first so both answers are about one passage)
        _cu0 = u0 + cd + LOT_GAP if cd else u0
        _cu1 = u1 - cd - LOT_GAP if cd else u1
        entry = court_entry_at(u0, u1, v0, mid0, side_least, _cu0, _cu1)
        sides = []
        if entry:
            ea, eb = entry
            # the passage is claimed before the ranges are drawn, so no lot is laid in
            # it and `free` refuses the columns to everything that comes after
            mark(ea, eb, v0, mid0 - 1, "street")
            runs = [(u0, ea - 1, 0), (eb + 1, u1, 4)]
        else:
            runs = [(u0, u1, 0)]
        _mine0 = len(plots)
        _n0 = counts["lots"]
        for (ra, rb, rk) in runs:
            if rb - ra + 1 >= int(side_least):
                lots_along(ra, rb, v0, v0 + rd - 1, fr.front(True), i, j, rk, plots,
                           depth=rd)
        sides.append(counts["lots"] - _n0)
        ends = ([lambda: lots_across(mid0, mid1, u0, u0 + cd - 1, fr.flanks()[0],
                                     i, j, 0, plots),
                 lambda: lots_across(mid0, mid1, u1 - cd + 1, u1, fr.flanks()[1],
                                     i, j, 1, plots)] if cd else [])
        for _fn in [lambda: lots_along(u0, u1, v1 - rd_b + 1, v1, fr.front(False),
                                       i, j, 1, plots, depth=rd_b)] + ends:
            _n0 = counts["lots"]
            _fn()
            sides.append(counts["lots"] - _n0)
        t = _enclosed_court_type(areas, role)
        cu0 = u0 + cd + LOT_GAP if cd else u0
        cu1 = u1 - cd - LOT_GAP if cd else u1
        n_here = areas_over(cu0, cu1, mid0, mid1,
                            t, "court", plots,
                            (f"the court the {sum(1 for n in sides if n)} range(s) of "
                             f"this block enclose"
                             + (", entered from the street through the passage in its "
                                "front range" if entry else "")),
                            f"pc{i}_{j}")
        # **The court's whole site, composed with it.** The quarter design round, and
        # the audit's second cause in its court half: the passage was a ledger mark and
        # a record-only rectangle the router never saw, and the clearance between the
        # paving and the ranges' lots was ground nobody owned -- four or five columns of
        # it between the paving and the facades. `court_site` carries all of it on the
        # leaf that is built: the paved court, the **margin** ring out to the ranges'
        # lot lines (owned by the court and laid by `site()` as a planted edge with its
        # way in paved), the **passage** through the front range sharing columns with
        # the court's own run, the street **landing** it opens from, the cell the court
        # is entered at, and the floor. The router takes the passage as the court's one
        # way in and the margin as ground no through-lane crosses.
        if n_here == 1 and entry and _design_level is not None:
            _ct = plots[-1]
            ea, eb = entry
            pc = (ea + eb) // 2
            mu0, mu1 = (u0 + cd, u1 - cd) if cd else (u0, u1)
            mv0, mv1 = v0 + rd, v1 - rd_b

            def _xz(a, b):
                x_, z_, _x, _z = fr.rect(a, a, b, b)
                return [int(x_), int(z_)]
            _ct["court_site"] = {
                "court": [int(_ct["x0"]), int(_ct["z0"]), int(_ct["x1"]), int(_ct["z1"])],
                "margin": [int(v) for v in fr.rect(mu0, mu1, mv0, mv1)],
                "passage": [int(v) for v in fr.rect(ea, eb, v0, mid0 - 1)],
                "landing": _xz(pc, v0 - 1),
                "approach": _xz(pc, mid0 - 1),
                "door": _xz(pc, mid0),
                "floor": int(_design_level),
                "street_side": fr.front(True),
                "ranges": [],                    # filled below, once they are laid
                "why": ("the court paved, its margin out to the ranges' lot lines "
                        "planted and owned by it, entered from the street through the "
                        "passage in its front range at the district level "
                        f"y={int(_design_level)}")}
        counts["courts"] += n_here
        counts["perimeter_blocks"] = counts.get("perimeter_blocks", 0) + 1
        # **How many of them actually closed, and how many can be walked into.** A block
        # whose fourth face was dropped for the arterial, or whose lots ran out against
        # the district's count, is a perimeter block with a hole in it -- which is an
        # ordinary courtyard block with extra steps. The numbers are kept apart so a
        # reader is never told a court is enclosed, or entered, because an arrangement
        # said `perimeter`. ...and a court closed on two sides is not closed: `cd` is 0
        # where the block had no width for its end ranges, which is a close and says so.
        # **Asked of the geometry, and not of the bookkeeping.** The delivery round:
        # counting four non-empty ranges says four ranges were laid, which is not the
        # same sentence as "this court has a building on each of its four sides". A face
        # whose lots were drawn short of the court's own centre line, or whose last lot
        # was dropped, leaves a court open at the one place a reader looks. So the claim
        # is made the way `test_neighbourhood_spatial.p6` measures it: for each court
        # tile, is there a plot of this block within reach on each of its four sides,
        # covering the tile's own centre.
        laid = [q for q in plots[_mine0:] if q.get("kind", "plot") == "plot"]
        courts_here = [q for q in plots[_mine0:]
                       if q.get("kind") == "area" and str(q.get("name", "")).startswith(
                           f"pc{i}_{j}")]
        reach = int(rd) + 2 * LOT_GAP + 1

        # **Enclosure is a property of the court, and it is measured as a gap.** The
        # block design round. The delivery round asked, of each court *tile*, whether a
        # plot covered the tile's own centre on each of its four sides -- and a court
        # laid as several tiles along its length then failed at the tiles the end ranges
        # do not reach, so a court enclosed by four ranges could report
        # `courts_enclosed: 0` and a court open on a whole face could report one tile
        # shut. Neither sentence is about the place. What makes a court a court is that
        # **a person standing in it has building on every side**: so the court's own
        # extent is taken whole, each of its four sides is walked column by column, and
        # the longest run of that side with no building of this block within reach is
        # its gap. A clearance between two lots is not a hole in a wall; the passage
        # into the court is the way in and is one hole by design. Anything wider is a
        # face that is not there.
        def side_gaps(rect) -> dict:
            cx0, cz0, cx1, cz1 = rect

            def run(cells) -> int:
                most = now = 0
                for ok in cells:
                    now = 0 if ok else now + 1
                    most = max(most, now)
                return most
            north = run(any(q["z1"] < cz0 and cz0 - q["z1"] <= reach
                            and q["x0"] <= x <= q["x1"] for q in laid)
                        for x in range(cx0, cx1 + 1))
            south = run(any(q["z0"] > cz1 and q["z0"] - cz1 <= reach
                            and q["x0"] <= x <= q["x1"] for q in laid)
                        for x in range(cx0, cx1 + 1))
            west = run(any(q["x1"] < cx0 and cx0 - q["x1"] <= reach
                           and q["z0"] <= z <= q["z1"] for q in laid)
                       for z in range(cz0, cz1 + 1))
            east = run(any(q["x0"] > cx1 and q["x0"] - cx1 <= reach
                           and q["z0"] <= z <= q["z1"] for q in laid)
                       for z in range(cz0, cz1 + 1))
            return {"north": north, "south": south, "west": west, "east": east}

        #: The widest hole a face of a court may have and still be a face: the clearance
        #: this fabric leaves between two lots. The street face may also have the
        #: passage, which is the way in and is the one hole a court is supposed to have.
        gap_ok = max(int(gap), int(LOT_GAP)) + 1
        front_is_low = (fr.front(True) in ("north", "west"))
        court_rect = None
        gaps = None
        if courts_here:
            court_rect = (min(c["x0"] for c in courts_here),
                          min(c["z0"] for c in courts_here),
                          max(c["x1"] for c in courts_here),
                          max(c["z1"] for c in courts_here))
            gaps = side_gaps(court_rect)
            street_side = (("north" if front_is_low else "south") if fr.u_is_x
                           else ("west" if front_is_low else "east"))
            bars = {s: (gap_ok + int(COURT_ENTRY) if s == street_side else gap_ok)
                    for s in gaps}
            open_faces = [s for s, g in gaps.items() if g > bars[s]]
        else:
            open_faces = ["(no court tile was laid)"]
        enclosed = [c["name"] for c in courts_here] if not open_faces else []
        shut_here = bool(n_here and cd and len(sides) == 4 and all(sides)
                         and not open_faces)
        # **The claim travels with the court, so the assembled world can be asked it.**
        # The block design round, and the audit's "`range_relation` ... is not asked of
        # these types at all". `range_relation` is about a *part's own* ranges round its
        # own yard, and a block's court is enclosed by other parts entirely -- so no
        # predicate in `usable` had anything to ask. The compiler writes what it claims
        # here: which ranges stand round this court, how far out they are, and how wide
        # a hole each face may have. `usable.court_enclosed` then reads the blocks and
        # says whether it is true of the world, which is a different sentence from this
        # one.
        for c in courts_here:
            c["enclosure"] = {
                "block": [int(i), int(j)],
                # the district, so a reader of the assembled world can find the ranges:
                # the plan names a leaf `b0_0_00` and the assembler stands it as
                # `<district>_b0_0_00`. `usable.court_enclosed` looked the plan's name
                # up in the built world's index and reported "0 of 12 ranges stood" for
                # every court in this project, which is a lookup that can never succeed.
                "district": str(district.get("name") or ""),
                # **the court, and not this tile of it.** `areas_over` lays a court as
                # several tiles with a clearance between them, and a tile is not the
                # place: the middle tile of a long court has its neighbouring tiles east
                # and west of it and no building for twenty columns, which is a true
                # statement about a tile and a false one about the court. Both the
                # compiler's own measure above and `usable.court_enclosed` answer about
                # this rectangle.
                "court": [int(v) for v in court_rect] if court_rect else None,
                "ranges": [q["name"] for q in laid],
                # **how far out the mass of a range actually starts**, which is the
                # clearance this fabric leaves between the court's paving and a lot
                # (`LOT_GAP`) plus the inset `site()` leaves round a pad (`PAD_INSET`).
                # The delivery round's `reach` was the range's whole depth plus its
                # clearances -- thirteen columns, wider than the court is deep -- and an
                # independent reader was right that "buildings on all four sides"
                # measured at that distance is a weaker sentence than it sounds.
                "reach": int(LOT_GAP + PAD_INSET + 1),
                "range_reach": int(reach),
                "cover_bar": COURT_FACE_COVER,
                "gap_bar": int(gap_ok),
                "entry_bar": int(COURT_ENTRY),
                "street_side": (("north" if front_is_low else "south") if fr.u_is_x
                                else ("west" if front_is_low else "east")),
                "planned_gaps": gaps,
                "entered": bool(entry),
                "claim": ("this court is enclosed by the ranges of its own block and "
                          "entered from the street through the passage in its front "
                          "range" if shut_here else
                          "this court's ranges did not close it: " + str(open_faces))}
            if c.get("court_site"):
                c["court_site"]["ranges"] = [q["name"] for q in laid]
                # the claim reads its margin and passage off the same record
                c["enclosure"]["margin"] = list(c["court_site"]["margin"])
                c["enclosure"]["passage"] = list(c["court_site"]["passage"])
        if shut_here:
            counts["perimeter_shut"] = counts.get("perimeter_shut", 0) + 1
        else:
            counts["perimeter_open"] = counts.get("perimeter_open", 0) + 1
        counts["courts_enclosed"] = counts.get("courts_enclosed", 0) + len(enclosed)
        if n_here and entry:
            counts["courts_entered"] = counts.get("courts_entered", 0) + 1
            x0e, z0e, x1e, z1e = fr.rect(entry[0], entry[1], v0, mid0 - 1)
            court_entries.append({"block": [int(i), int(j)],
                                  "rect": [x0e, z0e, x1e, z1e],
                                  "why": "the passage from the street into the court "
                                         "this block's ranges enclose"})
        elif n_here:
            counts["courts_unentered"] = counts.get("courts_unentered", 0) + 1
        return {"courts": int(n_here), "shut": shut_here, "entered": bool(entry),
                "enclosed": list(enclosed), "sides": [int(n) for n in sides],
                # which face laid how many lots at what depth, because "the third side
                # laid nothing" is the finding and a list of four numbers is not it
                "faces": {"street": {"lots": int(sides[0]), "depth": int(rd)},
                          "back": {"lots": int(sides[1]) if len(sides) > 1 else 0,
                                   "depth": int(rd_b)},
                          "ends": {"lots": sum(int(n) for n in sides[2:]),
                                   "depth": int(cd)}},
                "lots": int(len(laid)), "gaps": gaps,
                "court": [int(v) for v in court_rect] if court_rect else None,
                "why": ("" if shut_here else
                        (f"no court tile fits the {cu1 - cu0 + 1}x{mid1 - mid0 + 1} "
                         f"middle of this block" if not n_here else
                         "this block has no width for its end ranges" if not cd else
                         f"the ranges round the court laid {sides} lot(s): a face that "
                         f"laid none leaves the court open"
                         if not all(sides) or len(sides) != 4 else
                         f"the court's {', '.join(open_faces)} face(s) have a hole "
                         f"wider than this fabric's clearance: gaps {gaps}"))}

    def compose_court_block_if_whole(i, j, u0, u1, v0, v1, plots, side_least,
                                     rd, rd_b, cd) -> dict:
        """**Adopt the composition, or leave the block as it was.**

                The block design round, and the audit's second cause in its second half. A
                composition that half-succeeds is worse than one that was never begun: it spends
                the district's houses on two ranges, lays a paved tile between them, records
                `perimeter_open` and leaves a court in the world that no building encloses.

                So the composition is laid into a savepoint and kept **only if it closed**: the
                court exists, it has a building of this block on each of its four sides, and it
                is entered through the passage in its front range. Anything else is rolled back
                whole -- the leaves, the ledger, the counts, the rhythm faces and the wall-alt
                debt -- with the reason recorded, and the block falls through to the ordinary
                two-row block that this district can actually build. An unmet obligation stays
                an unmet obligation; it does not become a tile.
                
        """
        # the savepoint: everything this composition can move
        keep_ledger = ledger[u0:u1 + 1, v0:v1 + 1].copy()
        keep = {"plots": len(plots), "faces": len(faces), "counts": dict(counts),
                "dropped": dict(dropped), "alt": float(alt_debt[0]),
                "leaf": int(n_leaf[0]), "entries": len(court_entries)}
        got = compose_court_block(i, j, u0, u1, v0, v1, plots, side_least,
                                  rd, rd_b, cd)
        if got.get("shut") and got.get("entered"):
            compositions.append({"block": [int(i), int(j)], "adopted": True,
                                 "lots": got["lots"], "courts": got["courts"],
                                 "ranges": got["sides"]})
            return got
        del plots[keep["plots"]:]
        del faces[keep["faces"]:]
        del court_entries[keep["entries"]:]
        counts.clear()
        counts.update(keep["counts"])
        dropped.clear()
        dropped.update(keep["dropped"])
        alt_debt[0] = keep["alt"]
        n_leaf[0] = keep["leaf"]
        ledger[u0:u1 + 1, v0:v1 + 1] = keep_ledger
        counts["courts_refused"] = counts.get("courts_refused", 0) + 1
        compositions.append({"block": [int(i), int(j)], "adopted": False,
                             "rolled_back": int(got.get("lots") or 0),
                             "ranges": got.get("sides"), "faces": got.get("faces"),
                             "gaps": got.get("gaps"), "why": got.get("why")})
        return {**got, "shut": False, "adopted": False}

    # **The principal street is laid before the back streets.** A district is asked for
    # a number of houses and stops when it has them, and the blocks were filled in grid
    # order -- so on the retained city's middle ring the count ran out in the first
    # block row and the market's own street, which is the one street the quarter's
    # programme is about, got the leftovers: measured, **one shop** on a face that holds
    # five. The main street of a quarter is its primary structure and is laid first;
    # every district with nothing to put on a principal street keeps the grid order it
    # always had, which is why this is conditional rather than a reordering of the whole
    # compiler.
    order_blocks = list(blocks)
    if street_lot and principal_j is not None:
        first = [principal_j, principal_j - 1]
        order_blocks.sort(key=lambda ij: (first.index(ij[1]) if ij[1] in first
                                          else len(first), ij[1], ij[0]))

    # --- **the compositions are reserved before any of their children is spent** ------
    # The block design round, and the audit's second cause. Court blocks were composed
    # in grid order out of the same pot every ordinary row drew from, so whether a court
    # closed depended on how many houses the blocks before it had already taken. The
    # land and the programme of a complete composition are claimed **here**, before the
    # emission loop starts: what is left is what the ordinary rows may spend. A
    # composition the district cannot afford is not begun. Where the count is short of
    # the whole need, the composition is offered a **shorter block** first -- the same
    # four ranges about a smaller court, which is a reshape and not a broken perimeter
    # -- and only refused when even the shortest one it can draw does not fit. The
    # refusal is recorded with the number it needed and the number it had.
    court_plan: dict = {}
    if want_lots > 0 and (perimeter or n_court):
        _least = _plot_range(house)[0]
        #: how many compositions this district owes: every block where the arrangement
        #: is `perimeter`, else the share its form asks for
        want_courts = int(n_blocks if perimeter else n_court)
        # **and it may be any block of this district that can carry one.** The block
        # design round. `court_deep[:n_court]` picks the court blocks off a seeded
        # shuffle before anything knows where the road runs, and the first block of
        # `lower_ring_north_2` has the arterial along its back face -- so the one court
        # this district owed was reserved on the one block whose back range cannot be
        # built, and there was no second chance. The obligation is the district's and
        # not that block's: the marked blocks are tried first, then any other block deep
        # enough for two rows, until the district's courts are placed or it runs out of
        # blocks. **Deep enough for a composition, not deep enough for two rows.**
        # `court_deep` is the *old* courtyard block's test -- a block whose back row is
        # the court -- and it asks for two full lots of depth plus the gap between them.
        # A composed block is four ranges round a court and its ranges are as deep as
        # the block leaves room for: `lower_ring_north_2`'s second block row is 26
        # columns where two 13-deep rows need 29, and it holds two 7-deep ranges round a
        # 6-deep court perfectly well. `court_range_depth` is the test that knows that,
        # so it is the test, and the candidates are every block this district did not
        # give to a landmark or to open ground.
        cand_blocks = ([b for b in order_blocks if kind_of[b] == "courtyard"]
                       + [b for b in order_blocks if kind_of[b] == "row"])
        for (i, j) in cand_blocks:
            if len(court_plan) >= want_courts:
                break
            bu0, bu1 = cols[i]
            bv0, bv1 = rows[j]
            rdf, rdb, cdd = court_range_depth(bu0, bu1, bv0, bv1)
            if not rdf:
                continue
            free_lots = int(want_lots) - int(reserved[0])
            got = None
            # the whole block first, then the same composition on a shorter run: two
            # columns at a time, never below what still leaves a court and two end
            # ranges (`COURT_LEAST`, `side_least`), so a shrunk block is a block and not
            # a stub
            least_u = 2 * (int(cdd) + LOT_GAP) + COURT_LEAST if cdd else COURT_LEAST
            least_u = max(least_u, int(COURT_ENTRY) + 2 * int(_least) + 2)
            u_hi = bu1
            whole = None
            while u_hi - bu0 + 1 >= least_u:
                need = court_need(bu0, u_hi, bv0, bv1, _least, rdf, rdb, cdd)
                if whole is None:
                    whole = need
                # **and the composition has to close on this ground.** `court_need`
                # walks each face with `free()`, so a face the arterial or the terrain
                # takes comes back zero and the block is passed over here rather than
                # half-built and rolled back there.
                if need["closes"] and need["lots"] <= free_lots:
                    got = {"u1": int(u_hi), "need": int(need["lots"]),
                           "depths": [int(rdf), int(rdb), int(cdd)],
                           "shortened": int(bu1 - u_hi)}
                    break
                u_hi -= 2
            if got is None:
                whole = whole or court_need(bu0, bu1, bv0, bv1, _least, rdf, rdb, cdd)
                compositions.append({
                    "block": [int(i), int(j)], "adopted": False, "reserved": False,
                    "needed": int(whole["lots"]), "had": int(free_lots),
                    "closes": bool(whole["closes"]), "ranges_free": whole["ranges"],
                    "depths": [int(rdf), int(rdb), int(cdd)],
                    "why": ((f"this block's court needs {whole['lots']} house(s) round "
                             f"it ({whole['ranges']}) and the district has "
                             f"{free_lots} left of its {want_lots}: the composition is "
                             f"not begun rather than laid open")
                            if whole["closes"] else
                            (f"this block cannot carry a composition that closes: its "
                             f"faces admit {whole['ranges']} lot(s) on the ground the "
                             f"design prepares, with the road and what stands taken "
                             f"out. The court is not begun rather than laid open"))})
                continue
            court_plan[(i, j)] = got
            reserved[0] += got["need"]
    if court_plan:
        counts["courts_reserved"] = sum(g["need"] for g in court_plan.values())

    # --- **composed from its streets** (the fabric reset round) ----------------------
    # Where the character says `layout: street`, the block grid above is not what is
    # laid. `ethoslm.streetplan` decides the principal streets from the routed road, the
    # anchor at the street nearest the way in and the fronts that face it, the lanes and
    # the cross lanes, and only then cuts lots along each street face; it tries a small
    # family of materially different proposals and judges each on its required
    # relationships before any preference. The lots it adopts are emitted through this
    # compiler's own `leaf`, sited by `site_solve` and settled by `settle_storeys`, so
    # everything downstream reads a street-composed district exactly as it reads a grid.
    # Where no proposal holds its relationships, the best-preferred one is still laid --
    # a district of nothing is not better -- and the record says it is inadmissible and
    # why, for the layout owner (`composition.admissible`). a district asked for nothing
    # and anchoring nothing is open land, which the grid path already lays as one block
    # of its land use
    street_layout = (str(ch.get("layout") or "grid") == "street"
                     and (int(district.get("structures") or 0) > 0 or bool(landmarks)))
    composition = None
    street_plan_mask = None
    if street_layout:
        from . import streetplan as _sp
        order_blocks = []
        X0w, Z0w, X1w, Z1w = fr.x0, fr.z0, fr.x1, fr.z1

        def _to_uv(x, z):
            return ((x - fr.x0, z - fr.z0) if fr.u_is_x else (z - fr.z0, x - fr.x0))

        def _uv_rect(r):
            a0, b0 = _to_uv(r[0], r[1])
            a1, b1 = _to_uv(r[2], r[3])
            return min(a0, a1), max(a0, a1), min(b0, b1), max(b0, b1)
        okuv = (np.array(_mask_ok, dtype=bool) if _mask_ok is not None
                else np.ones((fr.U, fr.V), dtype=bool))
        okuv = okuv & ~band & ~taken
        okxz = okuv if fr.u_is_x else okuv.T
        # the anchor's side: the reservation the request requires, at the side it wants
        anchor_decl, anchor_name, anchor_spec = None, None, None
        if landmarks:
            anchor_name = landmarks[0].get("type")
            anchor_decl = decls.get(anchor_name)
            if anchor_decl is None and anchor_name:
                _t, _every = types_card()
                anchor_decl = _every.get(anchor_name)
            res0 = reserve[0] if reserve else {}
            s_want = max(int(res0.get("least") or 0),
                         int(res0.get("want") or LANDMARK_AREA_SIDE))
            anchor_spec = {"side": int(s_want),
                           "least": int(res0.get("least") or s_want)}
        # the dwelling and the shop: their lots, asked of their own envelopes in every
        # flank configuration a party wall can be in
        def _lot_band(dcl, pref_w, depth, lo_w=None, hi_w=None):
            if dcl is None:
                return None
            ok_w = []
            for ww in range(max(3, (lo_w or pref_w - 3)), (hi_w or pref_w + 3) + 1):
                if all(_admits(dcl, ww, depth, att, True)
                       for att in (("west", "east"), ("west",), ("east",), ())):
                    ok_w.append(ww)
            if not ok_w:
                return None
            # the longest run of admitted widths round the preferred one
            runs, cur = [], [ok_w[0], ok_w[0]]
            for v in ok_w[1:]:
                if v == cur[1] + 1:
                    cur[1] = v
                else:
                    runs.append(cur)
                    cur = [v, v]
            runs.append(cur)
            r = min(runs, key=lambda q: (0 if q[0] <= pref_w <= q[1] else 1,
                                         min(abs(q[0] - pref_w), abs(q[1] - pref_w))))
            return {"widths": (int(r[0]), int(r[1])),
                    "pref": int(min(max(pref_w, r[0]), r[1])), "depth": int(depth)}
        h_w = int(ch.get("lot_width") or w)
        h_d = int(ch.get("lot_depth") or ld)
        # **the court the design asks for is admitted, not hoped for**: with ranges of
        # at least 3 (the type's floor) a pad holds a court of `court_least` only where
        # it is that plus 6 across, and 2 x `FRONT_INSET` of the lot is not pad (1 at a
        # row's free end, where the flank keeps its inset)
        c_least = int(ch.get("court_least") or 0)
        # **What each use owes, resolved against the type selected for it, before any
        # lot is sized** (the design resolution round). `forms` is the character's
        # adopted minimum per use; the lot is then asked of the type's own form plan
        # (`ethoslm.formplan`) -- the rooms, court and storeys the builder will lay -- in
        # the flank configurations a run leaves its lots in, and a character lot that
        # cannot hold the form is enlarged to what can, openly, never the form shrunk.
        from . import formplan as _fp
        forms_ch = {u: dict(v) for u, v in (ch.get("forms") or {}).items()}
        dw_form = dict(forms_ch.get("dwelling") or {})
        if c_least and "court" not in dw_form:
            dw_form["court"] = c_least
        tr_form = dict(forms_ch.get("trade") or {})
        form_lots = {}
        house_spec = None
        if house is not None and _fp.plan_fn(house_name, house) is not None:
            _flo, _fhi = _fp.FLANKS["north"]
            mid = _fp.least_lot(house_name, dw_form, attached=(_flo, _fhi),
                                inset=_sp.FRONT_INSET, decl=house) or {}
            end = _fp.least_lot(house_name, dw_form, attached=(_flo,),
                                inset=_sp.FRONT_INSET, decl=house) or {}
            if mid.get("lot") and end.get("lot"):
                mw, md = (int(v) for v in mid["lot"])
                ew = int(end["lot"][0])
                house_spec = {"widths": (mw, mw + 2), "pref": int(min(max(mw, h_w), mw + 2)),
                              "depth": int(min(max(md, h_d), md + 2)), "depth_lo": md,
                              "depth_hi": md + 2, "end_extra": max(0, ew - mw),
                              "type": house_name}
            form_lots["dwelling"] = {
                "type": house_name, "params": dw_form,
                "character_lot": [h_w, h_d],
                "least_lot": {"middle": mid.get("lot"), "end": end.get("lot")},
                "plan": (mid.get("plan") or {}).get("why") or mid.get("why"),
                "why": (f"the character's {h_w}x{h_d} lot is "
                        + ("enough" if mid.get("lot") and h_w >= mid["lot"][0]
                           and h_d >= mid["lot"][1] else "raised")
                        + f" for {house_name} with {dw_form or 'its own proportion'}: "
                        f"{(mid.get('plan') or {}).get('why') or mid.get('why')}")}
        else:
            if c_least:
                h_d = max(h_d, c_least + 6 + 2 * _sp.FRONT_INSET)
            house_spec = _lot_band(house, h_w, h_d,
                                   max(h_w - 2, (c_least + 6 + _sp.FRONT_INSET)
                                       if c_least else 0), h_w + 2)
        if house_spec and "end_extra" not in house_spec:
            # shallower than asked where the ground behind a row is broken, down to what
            # the type stands on; never deeper than a step past the ask
            deps_ok = [d2 for d2 in range(max(9, h_d - 4, (c_least + 6 + 2 * _sp.FRONT_INSET)
                                                  if c_least else 0), h_d + 3)
                       if _lot_band(house, h_w, d2, h_w - 2, h_w + 2)]
            house_spec["depth_lo"] = min(deps_ok) if deps_ok else h_d
            house_spec["depth_hi"] = max(deps_ok) if deps_ok else h_d
            house_spec["type"] = house_name
        shop_name, shop_decl = None, None
        for _n, _d in list(mix["programme"]) + list(mix["own"]):
            if mix["use_of"].get(str(_n)) == "trade" or type_use(_n, _d)[0] == "trade":
                if _d.get("attached") or shop_decl is None:
                    shop_name, shop_decl = _n, _d
        shop_spec = None
        if shop_decl is not None and (mix["programme"] or mix["named"].get(shop_name)) \
                and _fp.plan_fn(shop_name, shop_decl) is not None:
            if "storeys" not in tr_form:
                tr_form["storeys"] = int(_storeys_band(shop_decl, ch)[0])
            _flo, _fhi = _fp.FLANKS["north"]
            for sd in (12, 13, 11, 14):
                ok_w = [ww for ww in range(7, 11)
                        if _fp.admits(shop_name, ww, sd, tr_form, decl=shop_decl,
                                      configs_=[(_flo, _fhi)])
                        and _lot_band(shop_decl, ww, sd, ww, ww)]
                if not ok_w:
                    continue
                end_w = next((ww for ww in range(ok_w[0], ok_w[0] + 4)
                              if _fp.admits(shop_name, ww, sd, tr_form, decl=shop_decl,
                                            configs_=[(_flo,), (_fhi,)])), ok_w[0] + 1)
                # ...and a lot standing alone (both flanks free) gets the extra twice
                alone_w = next((ww for ww in range(ok_w[0], ok_w[0] + 6)
                                if _fp.admits(shop_name, ww, sd, tr_form, decl=shop_decl,
                                              configs_=[()])), ok_w[0] + 2)
                end_w = max(end_w, ok_w[0] + (alone_w - ok_w[0] + 1) // 2)
                # a shallower shop where the ground asks, down to a pad of ten -- a shop
                # and a stair behind it -- and only where the least width holds the form
                # in every flank configuration its end of a run can leave it in
                sdeps = [d2 for d2 in range(12, sd + 1)
                         if _fp.admits(shop_name, ok_w[0], d2, tr_form, decl=shop_decl,
                                       configs_=[(_flo, _fhi)])
                         and _fp.admits(shop_name, end_w, d2, tr_form, decl=shop_decl,
                                        configs_=[(_flo,), (_fhi,)])]
                shop_spec = {"widths": (ok_w[0], ok_w[-1]),
                             "pref": int(min(max(8, ok_w[0]), ok_w[-1])), "depth": sd,
                             "depth_lo": min(sdeps) if sdeps else sd,
                             "end_extra": max(0, end_w - ok_w[0]), "type": shop_name}
                break
            form_lots["trade"] = {
                "type": shop_name, "params": tr_form,
                "lot": ([shop_spec["widths"], shop_spec["depth"]] if shop_spec else None),
                "why": (f"{shop_name} carries {tr_form} on lots "
                        f"{list(shop_spec['widths'])} wide (+{shop_spec['end_extra']} at a "
                        f"free end), {shop_spec['depth']} deep, by its form plan"
                        if shop_spec else
                        f"no lot 7-10 wide and 11-14 deep carries {shop_name} with "
                        f"{tr_form}")}
        elif shop_decl is not None and (mix["programme"] or mix["named"].get(shop_name)):
            for sd in (12, 13, 11, 14):
                shop_spec = _lot_band(shop_decl, 8, sd, 7, 10)
                if shop_spec:
                    break
            if shop_spec:
                shop_spec["type"] = shop_name
                # a shallow shop -- one storey, a counter on the street -- is still a
                # front: the least depth the type stands at is its floor here
                sdeps = [d2 for d2 in range(6, int(shop_spec["depth"]) + 1)
                         if _lot_band(shop_decl, 8, d2, 7, 10)]
                shop_spec["depth_lo"] = min(sdeps) if sdeps else shop_spec["depth"]
        _decl_of = {"house": (house_name, house), "shop": (shop_name, shop_decl)}

        def _site_ok(r, front, spec_):
            if _design_level is None:
                return True
            # the spec may be a copy at another depth: its type says which it is
            dcl = _decl_of["shop" if (spec_ or {}).get("type") == shop_name
                           and shop_name else "house"][1]
            flank = (("west", "east") if front in ("north", "south")
                     else ("north", "south"))
            st_, _why = site_solve(r[0], r[1], r[2], r[3], front, list(flank),
                                   pad_need(dcl), dcl, inset=_sp.FRONT_INSET)
            if st_ is None and os.environ.get("ETHOSLM_STREETPLAN_DEBUG"):
                print(f"      site refused {list(r)} front {front}: {_why}")
            return st_ is not None
        _acc2 = district.get("access")
        if not _acc2:
            pts = [p for p in (place.get("parts") or [])
                   if p.get("kind") == "point" and "gate" in str(p.get("type") or "")]
            cx_, cz_ = (X0w + X1w) / 2.0, (Z0w + Z1w) / 2.0
            if pts:
                g = min(pts, key=lambda p: abs(p["at"][0] - cx_) + abs(p["at"][-1] - cz_))
                _acc2 = [int(g["at"][0]), int(g["at"][-1])]
        # every routed road cell is a street a front may face; `road` above is only the
        # ones at the lots' level, which is the doorstep question and not this one
        _all_road = {(int(c[0]), int(c[1]))
                     for c in ((place.get("arterials") or {}).get("cells") or [])}
        _dump = os.environ.get("ETHOSLM_STREETPLAN_DUMP")
        if _dump:
            import pickle
            with open(_dump, "wb") as fh:
                pickle.dump({"rect": (X0w, Z0w, X1w, Z1w), "ok": okxz, "road": _all_road,
                             "access": _acc2, "house": house_spec, "shop": shop_spec,
                             "anchor": anchor_spec}, fh)
        got_c = _sp.compose((X0w, Z0w, X1w, Z1w), okxz, _all_road,
                            access=tuple(_acc2) if _acc2 else None,
                            house=house_spec, shop=shop_spec, anchor=anchor_spec,
                            seed=seed, site_ok=_site_ok,
                            defer_roads=bool(district.get("roads_pending")))
        pick_c = got_c["adopted"] or (max(got_c["proposals"],
                                          key=lambda p: p["preference"])
                                      if got_c["proposals"] else None)
        composition = {
            "by": "ethoslm.streetplan", "admissible": bool(got_c["adopted"]),
            "family": (_sp._fam_name(pick_c["family"]) if pick_c else None),
            "relationships": (pick_c["relationships"] if pick_c else []),
            "failed": (pick_c["failed"] if pick_c else []),
            "preference": (pick_c["preference"] if pick_c else 0),
            "houses": (pick_c["houses"] if pick_c else 0),
            "shops": (pick_c["shops"] if pick_c else 0),
            "house_lot": house_spec, "shop_lot": shop_spec, "anchor": anchor_spec,
            "streets": (pick_c["streets"] if pick_c else []),
            "tried": [{"family": _sp._fam_name(p["family"]),
                       "admissible": p["admissible"], "failed": p["failed"],
                       "preference": p["preference"], "houses": p["houses"],
                       "shops": p["shops"]} for p in got_c["proposals"]],
            "why": got_c["why"],
            # what each use owes and the lot its form plan needs (the design resolution
            # round): the lot is the form's, not the character's
            "forms": form_lots}
        ledger[:, :] = L["undeveloped"]
        street_plan_mask = np.zeros((fr.U, fr.V), dtype=bool)
        if pick_c:
            for st_ in pick_c["streets"]:
                if "rect" not in st_:
                    continue
                a0, a1, b0, b1 = _uv_rect(st_["rect"])
                a0, b0 = max(0, a0), max(0, b0)
                a1, b1 = min(fr.U - 1, a1), min(fr.V - 1, b1)
                if a1 >= a0 and b1 >= b0:
                    mark(a0, a1, b0, b1, "street")
                    street_plan_mask[a0:a1 + 1, b0:b1 + 1] = True
            # the anchor, as the leaf the grid would have laid
            if pick_c["anchor"] and anchor_decl is not None:
                ar = pick_c["anchor"]["rect"]
                a0, a1, b0, b1 = _uv_rect(ar)
                qa = quarters.setdefault("anchor", {"name": "anchor", "notes":
                                                    "the anchor at its street and the "
                                                    "fronts that face it", "plots": []})
                kind_a = "area" if anchor_decl.get("kind") == "area" else "plot"
                qa["plots"].append(leaf(kind_a, f"landmark_{anchor_name}", anchor_name,
                                        anchor_decl, a0, a1, b0, b1,
                                        front=pick_c["anchor"]["front"], flanks=0,
                                        notes=(landmarks[0].get("notes") or
                                               f"the {anchor_name} at the street the "
                                               f"quarter is entered by")))
                if kind_a == "plot" and _design_level is not None:
                    _st_a, _w = site_solve(ar[0], ar[1], ar[2], ar[3],
                                           pick_c["anchor"]["front"], [],
                                           pad_need(anchor_decl), anchor_decl)
                    if _st_a is not None:
                        qa["plots"][-1]["site"] = _st_a
                mark(a0, a1, b0, b1, "plaza" if kind_a == "area" else "lot")
                counts["landmarks"] += 1
            # the lots, run by run, each run settled as a row of party walls
            _lot_street: dict = {}
            runs: dict = {}
            for lot in pick_c["lots"]:
                runs.setdefault(str(lot.get("run")), []).append(lot)
            for rid, lots_r in runs.items():
                q = quarters.setdefault(f"street_{rid}", {
                    "name": f"street_{rid}",
                    "notes": (f"the {lots_r[0]['use']} fronts of one street face, "
                              f"fronting {lots_r[0]['front']}"), "plots": []})
                row, face = [], []
                for k, lot in enumerate(lots_r):
                    tname, decl = _decl_of[lot["use"]]
                    if decl is None:
                        continue
                    a0, a1, b0, b1 = _uv_rect(lot["rect"])
                    p = leaf("plot", f"s{len(quarters)}_{k:02d}", tname, decl,
                             a0, a1, b0, b1, front=lot["front"],
                             flanks=len(lot["attached"]),
                             notes=(f"a {tname} of the "
                                   f"{'principal street' if lot['street'] != 'lane' else 'lane'}"
                                   f"'s {lot['use']} front, facing {lot['front']}"
                                   if lot["street"] != "anchor" else
                                   f"a {tname} facing the {anchor_name} across its walk"))
                    p["attached"] = sorted(lot["attached"])
                    p["inset"] = int(_sp.FRONT_INSET)
                    # the use's adopted form: what this building owes, set on the leaf
                    # and not drawn (`settle_storeys` asks the form plan again with the
                    # flanks the lot really has)
                    f_use = "trade" if lot["use"] == "shop" else "dwelling"
                    f_par = (tr_form if f_use == "trade" else dw_form)
                    if _fp.plan_fn(tname, decl) is not None:
                        for k_, v_ in f_par.items():
                            if k_ in (decl.get("params") or {}):
                                p["params"][k_] = int(v_)
                        p["form"] = {"use": f_use, "params": dict(f_par)}
                    _lot_street[id(p)] = (lot.get("street"), lot["use"])
                    counts["party_walls"] += len(p["attached"])
                    row.append((a0, a1, p, tname, decl))
                    face.append((tname, a1 - a0 + 1, b1 - b0 + 1,
                                 int((p.get("params") or {}).get("storeys", 1))))
                    q["plots"].append(p)
                    mark(a0, a1, b0, b1, "lot")
                    counts["lots"] += 1
                # settle: site and storeys with the flanks each lot really has; a lot
                # refused frees its neighbours' flanks, which are settled again
                for _round in range(len(row) + 1):
                    moved, refused = settle_storeys(row, face)
                    counts["storeys_settled"] = counts.get("storeys_settled", 0) + moved
                    if not refused:
                        break
                    gone = {id(it[2]) for it in refused}
                    for it in refused:
                        pp_ = it[2]
                        if pp_ in q["plots"]:
                            q["plots"].remove(pp_)
                        counts["lots"] -= 1
                        counts["party_walls"] -= len(pp_.get("attached") or ())
                        a0_, a1_, b0_, b1_ = _uv_of(pp_)
                        ledger[a0_:a1_ + 1, b0_:b1_ + 1] = L["undeveloped"]
                    keep = [(it, f) for it, f in zip(row, face + [None] * len(row))
                            if id(it[2]) not in gone]
                    row = [it for it, _f in keep]
                    face = [f for _it, f in keep if f is not None]
                    # the flanks the survivors still have
                    rects = [it[2] for it in row]
                    for it in row:
                        pp_ = it[2]
                        was = list(pp_.get("attached") or [])
                        now = []
                        for s_ in was:
                            for o in rects:
                                if o is pp_:
                                    continue
                                if s_ == "east" and o["x0"] == pp_["x1"] + 1:
                                    now.append(s_)
                                elif s_ == "west" and o["x1"] + 1 == pp_["x0"]:
                                    now.append(s_)
                                elif s_ == "south" and o["z0"] == pp_["z1"] + 1:
                                    now.append(s_)
                                elif s_ == "north" and o["z1"] + 1 == pp_["z0"]:
                                    now.append(s_)
                        now = sorted(set(now))
                        if pp_.get("form") and set(was) - set(now):
                            # **a form's wall stays on its lot line** (the design
                            # resolution round): a neighbour refused on its ground frees
                            # this flank, and taking the pad's inset back there shrinks
                            # the house under the form it was admitted for -- one
                            # refusal then cascades down the row. The flank stands as a
                            # blank wall on open ground instead: still `attached` for
                            # the pad, recorded as having no neighbour.
                            pp_["blank_flanks"] = sorted(set(pp_.get("blank_flanks")
                                                             or ()) | (set(was) - set(now)))
                            counts["party_walls"] -= len(was) - len(now)
                            continue
                        counts["party_walls"] -= len(was) - len(now)
                        pp_["attached"] = now
                faces.append(face)
                if (lots_r[0]["use"] == "shop"):
                    counts["programme_uses"] = (counts.get("programme_uses", 0)
                                                + len(row))
            # the open ground it named, as the area types this district has
            gd_t = next((t for t in ("garden", "grove", "yard") if t in areas), None)
            yd_t = next((t for t in ("yard", "garden", "grove") if t in areas), None)
            qo = quarters.setdefault("open", {"name": "open", "notes":
                                              "the ground the streets and lots leave, "
                                              "named", "plots": []})
            for k, o in enumerate(pick_c["open"]):
                t_o = yd_t if o["use"] == "yard" else gd_t
                if not t_o:
                    continue
                a0, a1, b0, b1 = _uv_rect(o["rect"])
                counts["open"] += areas_over(a0, a1, b0, b1, t_o,
                                             t_o if t_o in LEDGER else "garden",
                                             qo["plots"],
                                             f"a {t_o}: {o['use']} ground the streets and "
                                             f"lots leave", f"o{k}")
            for _leaf in qo["plots"]:
                _leaf["open_ground"] = True
            if not qo["plots"]:
                del quarters["open"]
        # **judged again on what was emitted** (`streetplan.rejudge`): the proposal's
        # verdict was taken before each lot was settled, and a lot refused on its site
        # or storeys frees its neighbours' flanks and may open a street face
        if pick_c:
            final = []
            for qn, qq in quarters.items():
                if not str(qn).startswith("street_"):
                    continue
                for p_ in qq["plots"]:
                    if p_.get("kind", "plot") != "plot":
                        continue
                    st_, use_ = _lot_street.get(id(p_), (None, "house"))
                    final.append({"rect": [p_["x0"], p_["z0"], p_["x1"], p_["z1"]],
                                  "front": p_.get("front"), "use": use_,
                                  "street": st_, "attached": list(p_.get("attached") or [])})
            again = _sp.rejudge((X0w, Z0w, X1w, Z1w), okxz, _all_road, final,
                                pick_c["streets"], pick_c["anchor"],
                                house_needed=house_spec is not None,
                                anchor_needed=bool(anchor_spec),
                                defer_roads=bool(district.get("roads_pending")))
            composition["proposal_verdict"] = {
                "admissible": composition["admissible"],
                "failed": composition["failed"]}
            composition.update(admissible=bool(again["admissible"]),
                               relationships=again["relationships"],
                               failed=again["failed"], houses=again["houses"],
                               shops=again["shops"],
                               judged_on="the lots emitted after settling")
        if not composition["admissible"]:
            # a required anchor the composition could not front or place is the
            # request's reservation unmet -- refused by the validator and returned to
            # the layout owner -- and any other failed relationship is recorded as a
            # composition this district may not claim
            anchor_fail = [f for f in composition["failed"] if f.startswith("anchor")]
            req_anchor = any(r.get("required") for r in reserve)
            demand_short.append({
                "subject": (anchor_name if anchor_fail and req_anchor else "composition"),
                "what": "relationships", "kind": "district",
                "required": bool(anchor_fail and req_anchor),
                "requirement": ((reserve[0] or {}).get("requirement")
                                if anchor_fail and req_anchor else None),
                "least": None, "available": None, "needed": None, "needed_columns": None,
                "why": (f"the street composition laid here does not hold "
                        f"{', '.join(composition['failed'])} on the lots emitted; it is "
                        f"not a design this district may claim"),
                "alternatives": [{"what": "parent", "owner": "layout",
                                  "why": "a different sector, street pattern or ring "
                                         "allocation for this ground"}]})
        principal_from = ("composed from the routed road: "
                          + ", ".join(f"the {s_.get('side')} street"
                                      for s_ in (pick_c or {}).get("streets") or []
                                      if s_.get("kind") == "principal"))

    for (i, j) in order_blocks:
        u0, u1 = cols[i]
        v0, v1 = rows[j]
        kind = kind_of[(i, j)]
        # **The obligation is resolved per block, not per district.** Found by running
        # the shoreline village on the site the search chose: the shore curves, the
        # district carries one `faces` computed from its own middle, and the blocks at
        # its ends have the water in a different direction -- so two of nine homes
        # fronted away from the water beside them and the requirement failed on a place
        # that was doing its best. A district is a rectangle and a shore is not; the
        # block is the finest grain at which "which way do these lots front" is still
        # one answer, and it is where the question is asked.
        faces_side = faces_default
        if faces_at:
            from .placeshore import water_direction
            bx0, bz0, bx1, bz1 = fr.rect(u0, u1, v0, v1)
            got_side = water_direction(faces_at, (bx0 + bx1) // 2, (bz0 + bz1) // 2)
            if got_side:
                faces_side = got_side
        q = quarters.setdefault(f"row_{j}", {"name": f"row_{j}", "notes":
                                             f"the blocks between the {j + 1}th street "
                                             f"and the next", "plots": []})
        plots = q["plots"]
        # a block holds a back row where it is deep enough **and** the arrangement asks
        # for one: a negotiated one-row district that met a widened block would
        # otherwise lay the second row the ring's width was not budgeted for
        two_rows = rows_per_block >= 2 and (v1 - v0 + 1) >= 2 * ld + row_gap
        if kind == "open":
            t = _open_type(areas, role, density, use)
            what = t if t in LEDGER else "garden"
            counts["open"] += areas_over(u0, u1, v0, v1, t, what, plots,
                                         f"open ground: a {t} of the block", f"o{i}_{j}")
            continue
        if kind == "landmark":
            lm = landmarks[0]
            tname = lm.get("type")
            res = reserve[0] if reserve else {"subject": tname, "what": "landmark",
                                              "required": False, "requirement": None,
                                              "least": 0, "want": 0, "kind": "plot"}
            decl = decls.get(tname)
            # **A landmark the character names is laid whatever the district's role
            # table admits** (the closure round's held-out hamlet): a smithy is an urban
            # type and the hamlet's district is rural, so the workshop the character
            # asked for was silently not in `decls` and never laid, and the sentence's
            # "smithy beside them" had nothing to measure. The character's author chose
            # it by name; the whole table is where a named type is found.
            if decl is None and tname:
                _t, every_t = types_card()
                decl = every_t.get(tname)
                if decl is not None:
                    counts["landmark_from_whole_table"] = tname
            if decl is None:
                # **A named landmark no table has is an unmet demand and not a row of
                # houses.** It was silence: the branch simply fell through.
                demand_short.append({
                    **res, "available": [u1 - u0 + 1, v1 - v0 + 1],
                    "available_columns": (u1 - u0 + 1) * (v1 - v0 + 1),
                    "needed": None, "needed_columns": None,
                    "why": (f"the character names `{tname}` as this district's landmark "
                            f"and no committed type of that name is on disk"),
                    "alternatives": [{"what": "type", "owner": "capability",
                                      "field": "character.landmarks", "need": None,
                                      "have": None,
                                      "why": f"commit a type named `{tname}`, or name "
                                             f"one of this role's own"}]})
            if decl and decl.get("kind", "plot") == "plot":
                lo, hi, _ex = _plot_range(decl)
                s = min(hi, u1 - u0 + 1, v1 - v0 + 1, LANDMARK_SIDES * side)
                # a working landmark is a lot among lots, not a civic building on twice
                # the lot side: a 24x24 smithy among 12x10 cottages put a hamlet over
                # its density's ceiling on its own
                if str(decl.get("role") or "") != "civic":
                    s = min(s, max(w, ld) + 2)
                s = _clamp_side(s, decl)
                # **Where the request *requires* this landmark, the floor the envelope
                # measured is a floor**: a landmark drawn under the side its own
                # features need is ground claimed for something that will not be there
                # (`landmark_least`). A landmark the character merely prefers keeps the
                # lot-among-lots rule above, which the closure round's hamlet needed.
                if res.get("required") and int(res.get("least") or 0) > s:
                    s = _side_at_least(decl, int(res["least"]))
                if s <= min(u1 - u0 + 1, v1 - v0 + 1) and lo <= s:
                    ua = _landmark_u(u0, u1, s)
                    _lm_site = None
                    if _design_level is not None:
                        _lm_site, _w = site_solve(*fr.rect(ua, ua + s - 1, v0, v0 + s - 1),
                                                  fr.front(True), [], pad_need(decl), decl)
                    if free(ua, ua + s - 1, v0, v0 + s - 1) and (
                            _design_level is None or _lm_site is not None):
                        # a landmark stands on its own block, free on every side
                        plots.append(leaf("plot", f"landmark_{tname}", tname, decl,
                                          ua, ua + s - 1, v0, v0 + s - 1,
                                          front=fr.front(True), flanks=0,
                                          notes=lm.get("notes") or
                                          f"the {tname} the character names, on the "
                                          f"block nearest the middle"))
                        if _lm_site is not None:
                            plots[-1]["site"] = _lm_site
                        mark(ua, ua + s - 1, v0, v0 + s - 1, "lot")
                        # **a lot of its own**: the landmark's ground and its clearance
                        # are taken, so a house lot that would overlap it -- the back
                        # row of a block the landmark is deeper than -- is skipped by
                        # `free()` and laid elsewhere, never on top of it
                        taken[max(0, ua - LOT_GAP):min(fr.U, ua + s + LOT_GAP),
                              max(0, v0 - LOT_GAP):min(fr.V, v0 + s + LOT_GAP)] = True
                        counts["landmarks"] += 1
                        # **A working landmark stands among the houses; a civic one on
                        # its plaza** (the closure round's held-out hamlet): a smithy
                        # flanked by plazas stood twenty-three columns from the nearest
                        # house and "a smithy beside them" failed on its own block. Only
                        # a civic type gets the paved ground; the rest of a working
                        # landmark's block is houses either side of it.
                        if str(decl.get("role") or "") != "civic":
                            lots_along(u0, ua - LOT_GAP - 1, v0, v0 + ld - 1,
                                       fr.front(True), i, j, 0, plots)
                            lots_along(ua + s + LOT_GAP, u1, v0, v0 + ld - 1,
                                       fr.front(True), i, j, 2, plots)
                            if two_rows:
                                lots_along(u0, u1, v1 - ld + 1, v1, fr.front(False),
                                           i, j, 1, plots)
                            continue
                        pt = "plaza" if "plaza" in areas else _court_type(areas, role)
                        # the plaza before and beside it, where a tile fits
                        areas_over(u0, ua - LOT_GAP - 1, v0, v1, pt, "plaza", plots,
                                   f"the plaza beside the {tname}", f"p{i}_{j}w")
                        areas_over(ua + s + LOT_GAP, u1, v0, v1, pt, "plaza", plots,
                                   f"the plaza beside the {tname}", f"p{i}_{j}e")
                        areas_over(ua, ua + s - 1, v0 + s + LOT_GAP, v1, pt, "plaza",
                                   plots, f"the plaza behind the {tname}", f"p{i}_{j}s")
                        continue
            elif decl and decl.get("kind") == "area":
                # **An area landmark -- a market floor -- is laid as open ground among
                # the houses** (the expression round's city): the middle ring's
                # character named `market`, an area type, and this branch knew only
                # plots, so the landmark block stayed empty and the traders' ring had no
                # market. It is laid at the side its own features need, charged as open
                # ground (a market floor is not built cover), with houses along the rest
                # of the block so the market stands on the lane among them. **The side
                # is the reservation's, not the block's clamp.** The composition round:
                # this read `max(lo, min(hi, 17, block_w, block_d))` and then tested the
                # result against the same block -- so wherever the block was narrower
                # than the landmark's own floor the `max` put `s` above the block, the
                # test failed, and the leaf fell through to `kind = "row"` with nothing
                # recorded. `reservations_of` sized it before the grid was cut and the
                # grid was cut to hold it (`reserve_drove`); `LANDMARK_AREA_SIDE` is the
                # preferred side and `least` the floor the envelope measured.
                try:
                    lo, hi, _ex = _plot_range(decl)
                except Exception:                # noqa: BLE001 -- an area may say less
                    lo, hi = 3, 48
                least = max(int(lo), int(res.get("least") or 0))
                s = max(least, min(int(hi), int(res.get("want") or LANDMARK_AREA_SIDE),
                                   u1 - u0 + 1, v1 - v0 + 1))
                if s <= min(u1 - u0 + 1, v1 - v0 + 1) and lo <= s:
                    ua = _landmark_u(u0, u1, s)
                    if free(ua, ua + s - 1, v0, v0 + s - 1):
                        plots.append(leaf("area", f"landmark_{tname}", tname, decl,
                                          ua, ua + s - 1, v0, v0 + s - 1,
                                          front=fr.front(True), flanks=0,
                                          notes=lm.get("notes") or
                                          f"the {tname} the character names, on the "
                                          f"block nearest the middle"))
                        mark(ua, ua + s - 1, v0, v0 + s - 1, "plaza")
                        taken[max(0, ua - LOT_GAP):min(fr.U, ua + s + LOT_GAP),
                              max(0, v0 - LOT_GAP):min(fr.V, v0 + s + LOT_GAP)] = True
                        counts["landmarks"] += 1
                        lots_along(u0, ua - LOT_GAP - 1, v0, v0 + ld - 1,
                                   fr.front(True), i, j, 0, plots)
                        lots_along(ua + s + LOT_GAP, u1, v0, v0 + ld - 1,
                                   fr.front(True), i, j, 2, plots)
                        if two_rows:
                            # **the houses behind a market back onto it.** The quarter
                            # design round: in a block deeper than two rows the back row
                            # stood twenty columns behind the market with a strip of
                            # nobody's ground between, and the market stood alone on its
                            # own block. Behind the landmark the back row's lots run as
                            # deep as the block leaves (entered from the back street, as
                            # before), so the market is among houses on two sides.
                            deep0 = v0 + s + LOT_GAP
                            if v1 - deep0 + 1 > ld and ua - u0 >= 0:
                                a_, b_ = max(u0, ua - LOT_GAP), min(u1, ua + s + LOT_GAP - 1)
                                lots_along(u0, a_ - 1, v1 - ld + 1, v1, fr.front(False),
                                           i, j, 1, plots)
                                lots_along(a_, b_, deep0, v1, fr.front(False), i, j, 3,
                                           plots, depth=v1 - deep0 + 1)
                                lots_along(b_ + 1, u1, v1 - ld + 1, v1, fr.front(False),
                                           i, j, 4, plots)
                            else:
                                lots_along(u0, u1, v1 - ld + 1, v1, fr.front(False),
                                           i, j, 1, plots)
                        continue
            # **The reservation did not fit, so the demand goes back to its owner.** The
            # composition round's first change, and the one line of it that matters:
            # this was `kind = "row"` and nothing else -- the ground reserved for the
            # thing the request required became ordinary housing and no field of the
            # record, the plan file or the validator said a word. The block is still
            # filled (a block of nothing is not better than a block of houses), and the
            # unmet demand is now returned with what was needed, what was available and
            # what would make it fit; `compile_district.score` ranks an arrangement that
            # keeps it over one that loses it, and `placeplan.district_failures` refuses
            # the district where the demand was **required**.
            if decl is not None and not any(x.get("subject") == res.get("subject")
                                            for x in demand_short):
                need = max(int(res.get("least") or 0), 1)
                demand_short.append({
                    **res,
                    "available": [u1 - u0 + 1, v1 - v0 + 1],
                    "available_columns": (u1 - u0 + 1) * (v1 - v0 + 1),
                    "needed": [need, need], "needed_columns": need * need,
                    "why": (f"the block nearest this district's middle is "
                            f"{u1 - u0 + 1}x{v1 - v0 + 1} and the {res.get('subject')} "
                            f"the character names needs {need}x{need} to deliver what it "
                            f"declares; the block was filled with housing instead"),
                    "alternatives": _reserve_alternatives(
                        res, u1 - u0 + 1, v1 - v0 + 1, block, bd, fr.U, fr.V, want_lots)})
            kind = "row"
        # **A district may be told which way its lots front**, and where it is, a block
        # lays one row and not two. The integration round's second finding: the saved
        # shoreline village had nine of fifteen homes fronting *away* from the water it
        # was asked to face, and every check passed, because a two-row block puts one
        # row on each side of it and the checks were reading the district's aspect
        # ratio. A ribbon district's streets run along the shore either way; which side
        # of a street a door is on is a different decision and nothing was making it.
        # So: `district["faces"]` names a side, the row that fronts that side is laid,
        # and the rest of the block is the court or the open ground it would otherwise
        # have been. It costs the back row -- a place that must face one way holds fewer
        # houses per block than one that may face both, which is a true fact about the
        # request and not a defect -- and it is a general rule about frontage, not a
        # rule about water. ...and **only where the lots front a street**. A district of
        # `open` frontage has no street edge to build to: its lots stand apart in their
        # own ground, and `front` on one of them is an orientation and nothing else, so
        # both rows can simply be turned to face the water and no lot is lost. Where the
        # lots do front a street, a door has to be *on* one, and the back row of a two-
        # row block cannot be: there is the row's own back there and not a street. That
        # is when the block becomes one row, and the houses it costs are the price of
        # the request rather than a defect.
        if faces_side in (fr.front(True), fr.front(False)) and two_rows \
                and not open_front:
            at_low = faces_side == fr.front(True)
            if at_low:
                lots_along(u0, u1, v0, v0 + ld - 1, faces_side, i, j, 0, plots)
                rest = (v0 + ld + row_gap, v1)
            else:
                lots_along(u0, u1, v1 - ld + 1, v1, faces_side, i, j, 1, plots)
                rest = (v0, v1 - ld - row_gap)
            if rest[1] >= rest[0]:
                # **The role's own open ground, not a court.** Found by running it: a
                # rural district laid courts behind its single row and then failed its
                # own ground-cover clause at 34% against 60%, because what a rural
                # district's ground has to be is *fields*. The block behind a one-sided
                # row is the same open ground the rest of the district lays and is
                # chosen by the same rule (`_open_type`): fields for a farm belt, a
                # plaza for a dense quarter, gardens and groves for the rest.
                t = _open_type(areas, role, density, use)
                what = t if t in LEDGER else "garden"
                counts["open"] += areas_over(u0, u1, rest[0], rest[1], t, what, plots,
                                             f"the ground behind the row, which fronts "
                                             f"{faces_side} because the place is asked "
                                             f"to face that way",
                                             f"f{i}_{j}")
            continue
        # **A block composed about its court: built on all four faces, with the court
        # inside and a passage into it.** The neighbourhood round's `perimeter`
        # arrangement, and the delivery round's composition. A back-row `courtyard`
        # block is a block whose *back row* is the court, so the court is open along
        # both short faces, the cross streets carry nothing, and the enclosure the
        # arrangement's own name promises is not in the geometry -- which is what the
        # composition round's built section looks like from above. Here the front range
        # (with its passage), the back range and a range at each end of the middle band
        # are laid and what is left in the middle is the court. **A district whose
        # adopted form owes a court composes one whether or not its arrangement said
        # `perimeter`.** The delivery round: the obligation comes off the form
        # (`demand.court_obligation`), the composition is how it is realized, and a
        # block too small for it falls through to the two-row block below with the
        # shortfall on the record rather than in silence.
        _side_least = _plot_range(house)[0]
        _booked = court_plan.get((i, j))
        #: True where a composition was begun on this block and rolled back, or where
        #: the district owed a court here and its reservation pass refused this block.
        #: Such a block builds houses; it does not lay the open half of a court.
        _refused_here = any(not c.get("adopted") and c.get("block") == [int(i), int(j)]
                            for c in compositions)
        if _booked:
            # **the composition spends its own reservation and nothing else's.** It was
            # claimed before the loop began (`court_plan`); released here so its four
            # ranges can draw on it, and whatever it does not spend -- or gives back
            # when the composition is rolled back whole -- returns to the district.
            reserved[0] -= int(_booked["need"])
            _rdf, _rdb, _cd = _booked["depths"]
            _cu1 = int(_booked["u1"])
            _got = compose_court_block_if_whole(i, j, u0, _cu1, v0, v1, plots,
                                                _side_least, _rdf, _rdb, _cd)
            if _got.get("shut"):
                # the block's own remainder, where the composition was given a shorter
                # run than the block: ordinary street frontage, not leftover ground
                if _cu1 < u1 and (u1 - _cu1 - LOT_GAP) >= _side_least:
                    lots_along(_cu1 + LOT_GAP + 1, u1, v0, v0 + ld - 1,
                               fr.front(True), i, j, 2, plots)
                    lots_along(_cu1 + LOT_GAP + 1, u1, v1 - ld + 1, v1,
                               fr.front(False), i, j, 3, plots)
                continue
            # rolled back: this block is the ordinary two-row block below
            _refused_here = True
        # a row block or a courtyard block: the front row fronts the street before it
        if not two_rows:
            # **One row takes the whole block.** A block too shallow for two rows of
            # lots back to back used to lay one row of the density's own depth and leave
            # the rest of its depth to the fill -- a strip beside a place's centre came
            # back three houses on a tenth of its ground, under the plot cover its own
            # density asks. A shallow block is one row of deep lots.
            deep = _deepest(house, w, v1 - v0 + 1,
                            fr.flanks() if attached else (), fr.u_is_x)
            # a deep lot takes the whole block only where the density's ceiling leaves
            # room for it: sized under the ceiling, a lot stays the lot it was sized as
            if deep and ceiling_binds:
                deep = min(deep, ld)
            # **...and a depth the character or the adopted arrangement named is not the
            # compiler's to deepen.** The neighbourhood round, measured on the retained
            # crowded ring: `compact_bay` adopted 6x6 at one row to a block, this branch
            # deepened every lot to the 8 the block had room for, and the alternative
            # was then ranked on a 6x8 fabric under a 6x6 label. Lever 4 above has said
            # since the craft round that a named depth is not the compiler's to *give
            # away*; this is the same rule in the other direction. **Held to `ld` and
            # not to the named number**, which is the same thing where the lot was laid
            # as named and is the only safe form of it where it was not: a 200x17 band
            # adopted an 8-deep lot, `lot_min` raised the district's own to 13 because
            # the required features need it, and holding the block's one row to the
            # stale 8 laid **no house at all** -- the type does not stand on it.
            if deep and ch.get("lot_depth"):
                deep = min(deep, int(ld))
            # ...and never so deep that the count's lots exceed the ceiling: three farm
            # cottages deepened to 24 columns covered 720 of a sparse district's 430
            if deep and ceiling is not None and want_lots > 0:
                deep = max(min(deep, int(ceiling // (want_lots * max(1, w)))), min(deep, ld))
            if deep:
                one = (faces_side if faces_side in (fr.front(True), fr.front(False))
                       else fr.front(True))
                lots_along(u0, u1,
                           v0 if one == fr.front(True) else v1 - deep + 1,
                           v0 + deep - 1 if one == fr.front(True) else v1,
                           one, i, j, 0, plots, depth=deep)
            continue
        lots_along(u0, u1, v0, v0 + ld - 1, fr.front(True), i, j, 0, plots)
        # **A block whose composition was refused is a block of houses, not a back-row
        # court.** The block design round. `kind == "courtyard"` used to lay the block's
        # back row as open paved ground whatever had happened to the composition -- so
        # the block the round had just established *cannot* enclose a court laid the
        # open half of one anyway, which is the audit's fourth cause with the refusal
        # written down beside it. The obligation is unmet and the record says so
        # (`compositions`); the ground goes to the houses this district is short of.
        if kind == "courtyard" and not _refused_here:
            t = _court_type(areas, role)
            counts["courts"] += areas_over(u0, u1, v0 + ld + row_gap, v1, t, "court",
                                           plots, f"the court the row shares",
                                           f"c{i}_{j}")
        else:
            lots_along(u0, u1, v1 - ld + 1, v1, fr.front(False), i, j, 1, plots)

    # --- over the ceiling, trimmed. The lots are widened to take up a street's leftover
    # run and vary by the character's `variety`, so what was laid can cover more ground
    # than the count times the lot it was sized at. A district over its word's ceiling
    # gives up its last lots, from the end of the last street laid, until it is under;
    # the ground goes to the verges below. Never an exact count: that is laid whole and
    # the record says it is over, for the layout owner (`over_ceiling`).
    trimmed = 0
    narrowed = 0
    ceiling_cols = target.get("max_plot_columns")

    def _plot_cols():
        return sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1)
                   for q in quarters.values() for p in q["plots"]
                   if p["kind"] == "plot")

    # **An exact count over its ceiling is narrowed, not trimmed.** The block design
    # round, found by replanning the city after the grid learned to stand its blocks
    # beside a road rather than across it: `agrarian_belt_south_3` laid 1,566 columns of
    # lots against a sparse ceiling of 1,547 -- nineteen columns, 1.2% -- and the
    # validator stopped the whole plan twice for it. The count is the sentence's and is
    # not this compiler's to reduce (see `exact` above), and the loop below only knows
    # how to remove a lot; so an exact district gives the columns back off the
    # **widths** of its widest lots, down to the least side its type admits, and says it
    # is over only where narrowing cannot reach. A district a per cent over its word is
    # a district with slightly narrower houses, not a district with one house fewer.
    if ceiling_cols is not None and exact and _plot_cols() > ceiling_cols:
        _least_w = _plot_range(house)[0]
        while _plot_cols() > ceiling_cols:
            wide = None
            for q in quarters.values():
                for p in q["plots"]:
                    if p["kind"] != "plot":
                        continue
                    w_p = ((p["x1"] - p["x0"] + 1) if fr.u_is_x
                           else (p["z1"] - p["z0"] + 1))
                    if w_p > _least_w and (wide is None or w_p > wide[1]):
                        wide = (p, w_p)
            if wide is None:
                break
            if fr.u_is_x:
                wide[0]["x1"] -= 1
            else:
                wide[0]["z1"] -= 1
            narrowed += 1
    if ceiling_cols is not None and not exact:
        while _plot_cols() > ceiling_cols:
            victim = None
            for q in reversed(list(quarters.values())):
                lots_here = [p for p in q["plots"] if p["kind"] == "plot"]
                if lots_here:
                    victim = (q, lots_here[-1])
                    break
            if victim is None:
                break
            q, p = victim
            q["plots"].remove(p)
            # its ground back to the ledger, undeveloped, for the verge fill
            u0, v0 = ((p["x0"] - fr.x0, p["z0"] - fr.z0) if fr.u_is_x
                      else (p["z0"] - fr.z0, p["x0"] - fr.x0))
            u1, v1 = ((p["x1"] - fr.x0, p["z1"] - fr.z0) if fr.u_is_x
                      else (p["z1"] - fr.z0, p["x1"] - fr.x0))
            ledger[max(0, u0):u1 + 1, max(0, v0):v1 + 1] = L["undeveloped"]
            counts["lots"] -= 1
            trimmed += 1
    # --- the sites, against the plan as it finally stands ----------------------------
    # The quarter design round. Narrowing moves a lot's edge and trimming takes a lot
    # out of a terrace, freeing its neighbour's flank; a site chosen before either is a
    # site of a different plot. So every sited leaf is re-sited here with the flanks its
    # neighbours actually stand against. A leaf that no longer has a compatible pad and
    # way in is refused -- except in an exact district, whose count is the sentence's:
    # there it keeps no `site` (the old path) and the record names it.
    sites_missing: list = []
    if _design_level is not None:
        _all = [(qq, p) for qq in quarters.values() for p in qq["plots"]
                if p.get("kind", "plot") == "plot" and p.get("site")]

        def _abuts(p, side) -> bool:
            for _qq, o in _all:
                if o is p:
                    continue
                if side in ("west", "east"):
                    if not (o["z0"] <= p["z1"] and p["z0"] <= o["z1"]):
                        continue
                    if (side == "west" and o["x1"] + 1 == p["x0"]) or \
                            (side == "east" and o["x0"] == p["x1"] + 1):
                        return True
                else:
                    if not (o["x0"] <= p["x1"] and p["x0"] <= o["x1"]):
                        continue
                    if (side == "north" and o["z1"] + 1 == p["z0"]) or \
                            (side == "south" and o["z0"] == p["z1"] + 1):
                        return True
            return False
        for qq, p in list(_all):
            att = [s for s in (p.get("attached") or []) if _abuts(p, s)]
            if att != list(p.get("attached") or []):
                counts["party_walls"] -= len(p.get("attached") or ()) - len(att)
                p["attached"] = att
            st, why = site_solve(p["x0"], p["z0"], p["x1"], p["z1"], p.get("front"), att,
                                 pad_need(decls.get(p.get("type"))),
                                 decls.get(p.get("type")), inset=p.get("inset"))
            if st is not None:
                p["site"] = st
                continue
            if exact:
                p.pop("site", None)
                sites_missing.append({"name": p.get("name"), "refused_on": why})
                continue
            refuse_lot(p, p.get("type"), why, [], None)
            qq["plots"].remove(p)
            _all = [(a, b) for a, b in _all if b is not p]
            counts["lots"] -= 1
            a0, a1, b0, b1 = _uv_of(p)
            ledger[max(0, a0):a1 + 1, max(0, b0):b1 + 1] = L["undeveloped"]

    # --- the leftover, assigned: every free rectangle of a block, and the strips the
    # grid leaves at the district's edges, are verges of the role's ground type --
    # largest first, a clearance from everything laid, until none three across is left.
    # What is left after that is undeveloped, and the record says how much.
    verge_t = _verge_type(areas, role, use)
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
        # a street-composed district's streets are the ones it composed
        if street_plan_mask is not None:
            street_mask = street_plan_mask.copy()
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
    # **The programme, measured off what was laid.** `use_mix` says which uses this
    # quarter holds and where each one goes; this counts what actually stood, so the
    # share on the record is a consequence with a derivation beside it and never a
    # target that was aimed at. Every row says how it was asked for.
    laid_use: dict = {}
    for _q in quarters.values():
        for _p in _q["plots"]:
            if _p.get("kind", "plot") != "plot":
                continue
            u_here = (mix["use_of"].get(str(_p.get("type")))
                      or type_use(_p.get("type"), decls.get(_p.get("type")))[0])
            if u_here == UNSTATED_USE:
                u_here = mix["primary"]
            laid_use[u_here] = laid_use.get(u_here, 0) + 1
    n_laid = sum(laid_use.values())
    programme = {"primary": mix["primary"], "role": mix["role"],
                 "principal_street": principal_from,
                 # the block row the principal street is, so a reader can check *where*
                 # the quarter's work went rather than only how much of it there is. A
                 # leaf the compiler laid is `b<col>_<row>_<face><n>`.
                 "principal_row": principal_j,
                 "buildings": int(n_laid),
                 "uses": [], "unasked": [n for n, _d in mix["unasked"]],
                 "words": list(mix["words"]), "from": list(mix["from"])}
    for row in mix["uses"]:
        n_here = (int(counts["landmarks"]) if row["role"] == "anchor"
                  else sum(laid_use.get(u, 0) for u in (row["use"],)))
        programme["uses"].append({
            **row, "laid": int(n_here),
            "share": (round(n_here / float(n_laid), 4)
                      if n_laid and row["role"] != "anchor" else None),
            "share_from": (
                (f"{n_here} of the {n_laid} building(s) this district laid are "
                 f"`{row['use']}`; the share is measured off what was laid and was not "
                 f"aimed at -- the decision was {principal_from}")
                if row["role"] == "programme" else
                (f"{n_here} of the {n_laid} building(s) are the quarter's own fabric"
                 if row["role"] == "fabric" else None))})
    # **A use the design asked for and the ground did not hold is a shortfall, named.**
    # Not a refusal -- a quarter that could not seat its shops is still a quarter -- and
    # not silence either, which is what the composition round's filter gave when the one
    # trade type of a traders' ring fell out of the pool for being the wrong lot size.
    programme["short"] = [
        {"type": n, "requirement": rq, "why":
            f"`{rq}` requires a {n} of this district and no lot of its principal street "
            f"admitted one"} for n, _d, rq in owed]
    for row in programme["uses"]:
        if row["role"] == "programme" and not row["laid"] \
                and not any(s["type"] in row["types"] for s in programme["short"]):
            programme["short"].append({
                "type": row["types"][0], "requirement": row.get("requirement"),
                "why": (f"this district's own design asks for `{row['use']}` "
                        f"({'; '.join(row.get('asked_by') or []) or 'no source'}) and "
                        f"none was laid: "
                        + ("its fabric is attached, and a terrace is built of the one "
                           "type that declares party walls"
                           if attached else principal_from))})
    got = {"notes": (f"Compiled from the character of {part.get('name') or district['name']}: "
                     f"a {density} {role or ''} district, {'attached' if attached else 'detached'} "
                     f"lots of {w}x{ld} on blocks of {block}, streets {street} wide, "
                     f"{n_open} open block(s), {n_court} courtyard block(s)"
                     + (f", a {landmarks[0].get('type')} landmark" if landmarks else "")
                     + f"; {counts['lots']} lots"),
           "quarters": [q for q in quarters.values() if q["plots"]],
           "character": {k: v for k, v in ch.items()},
           # **The programme travels with the artifact that gets built.** The
           # neighbourhood round: "record the inferred mix on the compiled district so
           # it can be read downstream and the section record can report it".
           # `district_failures`, the assembler and the section record all read the
           # *file* and not the compile record, so a plan whose quarter is a traders'
           # quarter says so here.
           "programme": programme,
           # **The plan file says what it could not hold.** `district_failures` reads
           # the file and not the compile record, so the unmet demand travels with the
           # artifact that gets built: a plan whose required reservation was lost cannot
           # be validated as though it had one.
           **({"demand_short": [dict(x) for x in demand_short]} if demand_short else {}),
           **({"reservations": [dict(x) for x in reserve]} if reserve else {})}
    leaves = [p for q in got["quarters"] for p in q["plots"]]
    total = W_all * D_all
    assigned = {name: int((ledger == L[name]).sum()) for name in LEDGER}
    assigned["street"] += total - fr.U * fr.V        # the margin: a lane's worth
    plot_cols = sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1)
                    for p in leaves if p["kind"] == "plot")
    ground_cols = sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1) for p in leaves)
    # **A lot is ground promised to a building; it is not a building.** The design
    # round's second contract: `plot_cover` has always been the *lots'* share of the
    # rectangle, so a cover figure improved by drawing the same houses on bigger empty
    # lots read as denser construction. `footprint_columns` is the mass those lots admit
    # -- the pad `site()` will hand `build()`, the plan's own solid extent -- and the
    # two are reported apart so the difference is visible. **And it is an estimate,
    # which is now what it is called.** The neighbourhood round: "Alternative 'built
    # cover' is currently pad area (`district_compile.footprint_columns`), not
    # construction. Label estimates honestly." The number is the same number; what
    # changes is that `pad_columns` says in its name what it is, `footprint_estimate`
    # marks it, and `footprint_basis` says what would replace it. Nothing that stands
    # has been measured at this point in the run -- there is no world yet -- and a
    # reader who takes this for construction is taking a plan for a building.
    foot_cols = footprint_columns(leaves)
    record = {"district": district["name"], "part": part.get("name"), "seed": seed,
              **({"composition": composition} if composition is not None else {}),
              "asked_for": want_lots,
              "character": ch, "house": house_name, "attached": attached,
              "attached_note": attached_note,
              # **Three lots, and they are three different facts.** `lot` is the sizing
              # decision this compile worked to; `lot_asked` is what the character or
              # the adopted arrangement asked for, where it asked; `lot_laid` is the
              # **median lot actually drawn**, which is the one every downstream reader
              # has been taking `lot` for. They differ where the type clamps a side,
              # where a run's remainder widens the last lots, and.
              "others": [n for n, _d in others], "lot": [w, ld], "gap": gap,
              "lot_asked": ([int(ch["lot_width"]), int(ch["lot_depth"])]
                            if ch.get("lot_width") and ch.get("lot_depth") else None),
              "lot_laid": _median_lot(leaves, fr.u_is_x),
              "lot_min": _least_lot(district.get("lot_min"), dem_lot),
              "lot_min_raised": lot_min_raised,
              # the deepening that bought this fabric the height its own band asks for
              "storey_depth_raised": storey_depth_raised,
              "lot_refused": lot_refused,
              "demand": ({k: dem.get(k) for k in ("part", "types", "params", "required",
                                                  "optional", "requirements", "scope")}
                         if dem else None),
              "block": block, "block_depth": bd, "street": street,
              # **What this district reserved before it laid a house, and what it could
              # not.** `demand_short` is the unmet demand with feasible alternatives;
              # `reserve_drove` is where the reservation sized the block grid rather
              # than being clamped by it. Both empty for every district that names no
              # landmark, which is almost all of them. **what this district is for, and
              # what that let it be built of**: the quarter's own fabric, the other uses
              # its own design asks for and how each was asked, the types the pool held
              # that nothing asked for, and the share each use came to -- **measured**,
              # with its derivation, never set. The user's instruction for the
              # composition round and the number it was found at: 5 temples and 4 halls
              # against 1 shop house in 22 buildings of the middle ring; the
              # neighbourhood round's, and its number: 1 shop in 22 buildings of a ring
              # the sources call the traders' and craftsmen's.
              "use_mix": {"role": mix["role"], "primary": mix["primary"],
                          "own": [n for n, _d in mix["own"]],
                          "programme": [n for n, _d in mix["programme"]],
                          "unasked": [n for n, _d in mix["unasked"]],
                          "use_of": dict(mix["use_of"]),
                          "named": {k: list(v) for k, v in mix["named"].items()},
                          "required": dict(mix["required"]),
                          "share": next((u["share"] for u in programme["uses"]
                                         if u["role"] == "programme"), None),
                          "laid": int(counts.get("programme_uses") or 0),
                          "by_use": dict(laid_use),
                          "from": list(mix["from"])},
              "programme": programme,
              "reservations": [dict(x) for x in reserve],
              "reservations_kept": int(counts["landmarks"]),
              "reservations_required": sum(1 for x in reserve if x.get("required")),
              "demand_short": [dict(x) for x in demand_short],
              "reserve_drove": reserve_drove,
              "columns": total, "blocks": n_blocks,
              "axial_arterial": [bool(axial_u), bool(axial_v)],
              "block_kinds": {k: sum(1 for b in blocks if kind_of[b] == k)
                              for k in ("row", "courtyard", "open", "landmark")},
              # **The court this district's adopted form owes, and what became of it.**
              # The delivery round. `demand.court_obligation` publishes the obligation
              # off the same `courtyard_share`; this says what the compiler did with it,
              # so a reader can tell "no court was owed" from "a court was owed and the
              # blocks are too shallow to compose one" from "a court was composed and
              # entered". `courts` beside it is the number of court tiles laid. **Held
              # is "a composition closed", not "a block was deep enough for two rows".**
              # The block design round. `court_deep` is the old courtyard block's test
              # and a composed block is four ranges at the depth the court leaves them,
              # so `lower_ring_north_2` published `blocks_deep_enough: 0` and `unheld:
              # true` on a compile that had just laid an enclosed court. The obligation
              # is held by the compositions that were adopted.
              "court_obligation": ({"share": court_share,
                                    "blocks_asked": int(n_court),
                                    "blocks_considered": len(compositions),
                                    "blocks_composed": sum(1 for c in compositions
                                                           if c.get("adopted")),
                                    "blocks_two_rows_deep": len(court_deep),
                                    "entries": [dict(e) for e in court_entries],
                                    "unheld": bool(court_owed and not any(
                                        c.get("adopted") for c in compositions)),
                                    "why": ("this district's form owes a court at a "
                                            f"share of {court_share:g} of its "
                                            f"{n_blocks} block(s); "
                                            f"{sum(1 for c in compositions if c.get('adopted'))}"
                                            f" of {len(compositions)} block(s) it "
                                            f"considered carried a composition that "
                                            f"closed")}
                                   if court_owed else None),
              # **Every complete composition this district considered, and its
              # verdict.** The block design round. A court block is adopted as a whole
              # or not begun, and this is the ledger of both: the blocks whose
              # composition was reserved and closed, the blocks whose reservation the
              # count could not afford, and the blocks whose composition was laid and
              # **rolled back** because its court did not close. A reader can tell an
              # unowed court from an unaffordable one from one that was refused on the
              # geometry, which the old `perimeter_open` count could not say.
              "compositions": [dict(x) for x in compositions],
              "composition_reserved": int(counts.get("courts_reserved") or 0),
              "grid": {"columns": len(cols), "rows": len(rows)},
              "leaves": len(leaves), **counts, "dropped": dropped,
              # **What the ground refused, by name and in columns.** The spatial-design
              # round: `arrange.alternatives` ranks proposals and the acceptance runner
              # reads them, and neither can tell a district that was laid small from a
              # district whose ground would not carry what it was asked for unless the
              # compiler says so. `ground_measured` false means no mask was read and
              # this rule did not run -- which is not the same as a mask that refused
              # nothing. ...and the columns are the **district's own** infeasible
              # columns, not a sum over the rectangles this rule refused: `free()` is
              # asked about overlapping candidates and adding them up gave a district of
              # 11,400 columns a refusal of 13,010, which is not a measurement of
              # anything.
              "lots_refused_for_ground": int(dropped["ground"]),
              "ground_refused_columns": (
                  int(ground_rec.get("columns") or 0)
                  - int(ground_rec.get("feasible_columns") or 0)
                  if founded is not None else None),
              "ground_measured": founded is not None,
              "ground_feasible_share": (round(float(ground_rec.get("feasible_columns")
                                                    or 0)
                                              / float(ground_rec.get("columns") or 1), 4)
                                        if founded is not None else None),
              "ground_bar": GROUND_FOUNDED,
              "assigned": assigned,
              "undeveloped_share": round(assigned["undeveloped"] / float(total), 4),
              "plot_cover": round(plot_cols / float(total), 4),
              "ground_cover": round(ground_cols / float(total), 4),
              # the four columns of this region (`placeregion.COLUMN_FIELDS`), as far as
              # a plan can answer them: what was allocated and what mass it admits.
              # `built_columns` stays the emitted world's to fill.
              "allocated_columns": int(plot_cols),
              # **The pad estimate, under a name that says so.** `footprint_columns` is
              # kept because `placeplan.region_columns` and every retained record read
              # it; `pad_columns` is the same number under the name it earns, and the
              # two marker fields are what a reader needs to not mistake it for
              # construction. `region_columns` replaces it with the emitted extent as
              # soon as there is a parts record to read, and says which it read in
              # `built_from`.
              "footprint_columns": int(foot_cols),
              "pad_columns": int(foot_cols),
              "footprint_estimate": True,
              "footprint_basis": ("planned pads: the extent `site()` will hand `build()` "
                                  "for each plot of this plan, summed. Not construction "
                                  "-- nothing has been built at this point in the run. "
                                  "`placeplan.region_columns(..., parts_record=...)` "
                                  "answers with the emitted footprint once there is one"),
              "rows": int(rows_per_block),
              "target": {"count": target["count"],
                         "min_count": target["min_count"],
                         "plot_share": target["plot_share"],
                         "min_plot_columns": target["min_plot_columns"],
                         "max_plot_columns": target.get("max_plot_columns"),
                         "min_ground_columns": target["min_ground_columns"],
                         "density_target": target.get("density_target")},
              "exact": exact,
              "capped_to": capped_to, "trimmed": trimmed,
              "narrowed": narrowed,
              "lot_cover_usable": round(plot_cols / float(max(1, target["usable_columns"])), 4),
              "over_ceiling": bool(target.get("max_plot_columns") is not None
                                   and plot_cols > target["max_plot_columns"]),
              "registered_plot_cover": PLOT_COVER[density],
              "meets_registered_cover": plot_cols / float(total) >= PLOT_COVER[density],
              "variety": _variety(faces),
              "wall_alt": _wall_alt_record(leaves),
              # **The sites this plan carries, and the lots it refused for want of
              # one.** The quarter design round: a leaf's `site` is the one pad, floor,
              # door and landing construction builds; a lot with no compatible pad and
              # way in, or whose envelope refuses its type's storeys with no other type
              # of its use to stand there, is refused here and counted apart.
              "sites": {"sited": sum(1 for p in leaves if p.get("site")),
                        "courts_sited": sum(1 for p in leaves if p.get("court_site")),
                        "refused": dict(sites_refused),
                        "lots_refused": [dict(x) for x in lots_refused],
                        "retyped": [dict(x) for x in retyped],
                        "missing": [dict(x) for x in sites_missing],
                        "measured": _design_level is not None}}
    return got, record


def _median_lot(leaves: list, u_is_x: bool = True) -> list | None:
    """**The lot this district actually drew**, as the median width and the median depth
        over its plot leaves, **in the district's own frame** (`u` along its streets, `v`
        across them) -- which is the frame `lot`, `lot_width` and `lot_depth` are all in, so
        a district whose streets run along z is compared like for like. None where it drew
        no lot.

        Two medians rather than the median pair on purpose: a run whose last lot takes the
        remainder, a landmark on a lot of its own and a shop street at its own grain all make
        the *pair* rare, and what a reader wants from this number is "what size is a lot
        here". `placeplan.arrangement_failures` compares an adopted arrangement against it.
        
    """
    sizes = [((p["x1"] - p["x0"] + 1, p["z1"] - p["z0"] + 1) if u_is_x else
              (p["z1"] - p["z0"] + 1, p["x1"] - p["x0"] + 1))
             for p in leaves or [] if p.get("kind", "plot") == "plot"]
    if not sizes:
        return None
    ws = sorted(s[0] for s in sizes)
    ds = sorted(s[1] for s in sizes)
    return [int(ws[len(ws) // 2]), int(ds[len(ds) // 2])]


def pad_columns(leaves: list) -> int:
    """**The mass these leaves would stand as, estimated from the plan**: the pad
        `site()` will hand `build()` for each plot, summed -- not the lots they stand on, and
        **not what was built**.

        The design round's escape route E2, in one number. A district whose lots grew from
        6x7 to 10x10 covers more ground with the same fifteen houses, and a density figure
        read off the lots calls that denser construction. The pad is what a person walking
        the street *would* see standing; an empty lot is not a building.

        The neighbourhood round renamed it. It was `footprint_columns`, and every consumer
        read it into a field called `built_columns` and a figure called "built cover" -- so
        the project's whole history of built cover is this estimate under a name that says it
        was measured. The estimate is a good one and it is the only answer available before
        anything is built; what it is not is construction. `footprint_columns` remains as an
        alias because retained records and `placeplan.region_columns` use the old name.
        
    """
    from .buildlib import Builder
    out = 0
    for p in leaves or []:
        if p.get("kind", "plot") != "plot":
            continue
        try:
            w, d = Builder.pad_extent(p)
        except (KeyError, TypeError, ValueError):    # noqa: PERF203 -- no geometry
            continue
        out += max(0, int(w)) * max(0, int(d))
    return int(out)


#: The old name, kept so retained records, `placeplan.region_columns` and any consumer
#: written before the rename keep working. Deliberately an alias and not a copy: there
#: is one arithmetic and the honest name is the one above.
footprint_columns = pad_columns


def _wall_alt_record(leaves: list) -> dict:
    """The share of this district's houses faced in the voice's second wall material.

        The craft round, E3. `Builder.WALL_ALT_SHARE` is the library's number and the
        compiler is what makes it a **share** rather than a distribution, because it can see
        the whole street; `WALL_ALT_TOLERANCE` is how far a district may sit from it.
        
    """
    said = [p for p in leaves if isinstance(p.get("wall_alt"), bool)]
    alt = sum(1 for p in said if p["wall_alt"])
    share = (alt / len(said)) if said else None
    return {"houses": len(said), "faced": alt,
            "share": None if share is None else round(share, 4),
            "registered": WALL_ALT_SHARE, "tolerance": WALL_ALT_TOLERANCE,
            "holds": bool(share is None or len(said) < 20
                          or abs(share - WALL_ALT_SHARE) <= WALL_ALT_TOLERANCE)}


def _storeys_fit(tname: str, lo: int, hi: int, w: int, d: int,
                 cache: str | None = None, demand=None, flanks: int = 0) -> int | None:
    """The most storeys in `lo..hi` this type delivers on a `w`x`d` lot, or None where
        no envelope can answer.

        **`flanks` is how many of the lot's flanks the next building stands against**, and
        it is the neighbourhood delivery round's addition. `Builder._insets` drops the pad's
        inset on an attached side, so the same plot is a different pad with two party walls,
        with one, and with none -- and a type declares what it can do with a **pad**. Asking
        this question with free flanks of a lot in a terrace is asking about a different
        building: every 6x13 terrace lot of the delivered candidate answered "1 storey"
        because 6x13 free insets to 4x11, while the pad it was actually built on was 6x9,
        which `row_house`'s own `STOREY_PAD` admits two storeys on. Default 0, so every
        caller that does not name a configuration behaves exactly as it did.

        **And it is asked about `tname`, the type this leaf selected.** The block design
        round, and the audit's third cause: this forwarded the district's whole multi-type
        demand to `demand.storeys_admitted`, which took `next(iter(d['types']))`. A district
        whose pool is `{court_large, shop_house, row_house}` therefore certified every leaf
        against `court_large`, whatever stood there. The type is now named at the call and
        a demand that does not approve it refuses instead of answering about another
        building.

        **Through the resolved demand, and never through an ad-hoc cache of its own.** The
        expression review's first cause, at this call site: the question went to
        `envelope.lot_for(tname, {"storeys": st})` with no `features`, so `_stands` checked
        no requested feature and a standing shell answered a multi-storey query; and the
        answer was memoised on `(type, storeys)` alone, so an edited type file answered from
        an obsolete certificate. `demand.storeys_admitted` asks the same question carrying
        the required features, the voice and the tested context, and fingerprints the type
        and its primitives in its own key.

        `demand` is the district's resolved demand (`demand.resolve`). Where the district
        has one and the module is present, that is the only path. Where the district has
        **required features** and the module is not present, this refuses -- a missing
        module must not become a smaller default.
        
    """
    d_mod = _demand_module()
    if demand is not None:
        if d_mod is None or not hasattr(d_mod, "storeys_admitted"):
            raise ValueError(
                f"{tname}: this district resolved a demand "
                f"({', '.join(demand.get('required') or ()) or 'no required feature'}) "
                f"and `ethoslm.demand` cannot answer what a {w}x{d} lot admits; a storey "
                f"band cannot be settled without it")
        return d_mod.storeys_admitted(demand, int(w), int(d), cache=cache,
                                      flanks=int(flanks), type_name=tname)
    # No resolved demand reached this district: the pre-demand path, kept so a direct
    # call from a test or a plan written before the contract behaves as it did.
    try:
        from . import envelope as _env
    except Exception:                        # noqa: BLE001 -- not landed
        return None
    fn = getattr(_env, "lot_for", None)
    if fn is None:
        return None
    for st in range(int(hi), int(lo) - 1, -1):
        try:
            got = fn(tname, {"storeys": st}, features=("storeys",), cache=cache,
                     context={"attached": int(flanks)}) or {}
            lm = tuple(int(v) for v in (got.get("lot_min") or (0, 0)))
        except Exception:                    # noqa: BLE001 -- an envelope that cannot say
            lm = (0, 0)
        if not lm or not lm[0]:
            return None
        if (lm[0] <= w and lm[1] <= d) or (lm[0] <= d and lm[1] <= w):
            return {"storeys": int(st), "asked": int(lo), "band": [int(lo), int(hi)],
                    "holds": True, "required": False, "type": tname,
                    "why": f"the {w}x{d} lot admits {st} storey(s) of {tname}"}
    return {"storeys": int(lo), "asked": int(lo), "band": [int(lo), int(hi)],
            "holds": True, "required": False, "type": tname,
            "why": f"the {w}x{d} lot admits the band's floor of {lo}"}


def _demand_lot(dem: dict | None, district: dict | None = None) -> tuple:
    """`(lot_min, refused)` for a district's resolved demand.

        `refused` is a record and never a smaller lot: where nothing in the approved band
        delivers a required feature the compiler raises, because a district laid at a lot
        that cannot hold what the request requires is the defect, not the recovery.
        
    """
    if not dem:
        return None, None
    d_mod = _demand_module()
    if d_mod is None or not hasattr(d_mod, "lot"):
        if dem.get("required"):
            raise ValueError(
                f"{(district or {}).get('name') or dem.get('part')}: the request "
                f"requires {', '.join(dem.get('required') or ())} of this district's "
                f"fabric and `ethoslm.demand` is not available to say what lot delivers "
                f"it; a lot cannot be chosen without it")
        return None, None
    got = d_mod.lot(dem) or {}
    if got.get("refused"):
        raise ValueError(
            f"{(district or {}).get('name') or dem.get('part')}: no approved type "
            f"delivers {', '.join(dem.get('required') or ()) or 'the resolved demand'} "
            f"-- {got['refused']}. A smaller lot is not an answer to a refused envelope")
    lm = got.get("lot_min")
    return ([int(lm[0]), int(lm[1])] if lm and len(lm) == 2 else None), None


def _least_lot(a, b):
    """The larger of two `[w, d]` minimums, either of which may be None."""
    if not a:
        return list(b) if b else None
    if not b:
        return list(a)
    return [max(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))]


def _form_plan_fn(tname, decl):
    from . import formplan as _fp
    return _fp.plan_fn(tname, decl)


def _form_of_pad(p: dict, tname: str, decl: dict) -> dict:
    """The type's form plan on the pad this leaf's site will build (`p["site"]["pad"]`),
    or on the lot's inset pad where it has no site yet."""
    from . import formplan as _fp
    front = p.get("front") or "north"
    pad = (p.get("site") or {}).get("pad")
    if pad:
        W, D = pad[2] - pad[0] + 1, pad[3] - pad[1] + 1
        pw, pd = (W, D) if front in ("north", "south") else (D, W)
        got = dict(_fp.plan_fn(tname, decl)(int(pw), int(pd),
                                            params=dict(p.get("params") or {}),
                                            front=front,
                                            attached=list(p.get("attached") or ())))
        got["pad"] = [int(pw), int(pd)]
        return got
    return _fp.plan_for(tname, p) or {"ok": True}


def _demand_module():
    """`ethoslm.demand` where it is on disk, else None. Imported defensively because the
    module lands separately; every caller that has a **required** feature to carry
    refuses on None rather than falling back to a featureless query."""
    try:
        from . import demand as _demand
    except Exception:                        # noqa: BLE001 -- not landed yet
        return None
    return _demand


def _storeys_band(decl: dict, ch: dict) -> tuple:
    """**A street has a skyline**, the craft round (E3): the height of a building is the
    character's own band clamped into what the type declares, so a run of roofs steps
    rather than lying flat and a district that wants to be low is low in every type on
    it. Where the two bands do not meet, the type's own is what it can build.
    """
    spec_p = ((decl.get("params") or {}).get("storeys") or ())
    if not (isinstance(spec_p, (list, tuple)) and len(spec_p) >= 3
            and spec_p[0] == "int"):
        return (1, 1)
    lo, hi = int(spec_p[1]), int(spec_p[2])
    want = ch.get("storeys")
    if want and max(lo, int(want[0])) <= min(hi, int(want[1])):
        lo, hi = max(lo, int(want[0])), min(hi, int(want[1]))
    return (lo, hi)


def _variety(faces: list) -> dict:
    """**Is this street a rhythm or a comb?** The craft round, E3, read off the shapes
        the compiler laid rather than off the built world, because it is the plan that
        decides them and a plan is on disk in a fiftieth of a second.

        Two numbers, both registered before they were read: how many distinct building
        shapes there are per hundred houses (`SHAPES_PER_HUNDRED`), and the longest run of
        neighbours of one shape along one street face (`IDENTICAL_RUN_MAX`).
        
    """
    flat = [sh for face in faces for sh in face]
    kinds = {sh for sh in flat}
    longest = 0
    for face in faces:
        run = 1 if face else 0
        longest = max(longest, run)
        for a, b in zip(face, face[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
    per = 100.0 * len(kinds) / len(flat) if flat else 0.0
    # **...and which of the four things a shape is made of actually varies.** The craft
    # round's two numbers say a street is a comb and do not say *why*, and the reason is
    # almost always one of these four standing still: the neighbourhood round's reading
    # of the retained section's crowded side is "one house repeated ninety-six times",
    # and the cause is that its lot admits one storey of its type, not that the compiler
    # failed to vary anything. A reader who has these four does not have to open the
    # leaves to tell a layout limit from a generator limit.
    return {"lots": len(flat), "shapes": len(kinds), "faces": len(faces),
            "per_hundred": round(per, 1), "longest_run": int(longest),
            "types": len({sh[0] for sh in flat}),
            "widths": len({sh[1] for sh in flat}),
            "depths": len({sh[2] for sh in flat}),
            "heights": len({sh[3] for sh in flat}),
            "storeys": sorted({int(sh[3]) for sh in flat}),
            "registered": {"shapes_per_hundred": SHAPES_PER_HUNDRED,
                           "identical_run": IDENTICAL_RUN_MAX},
            "holds": bool(flat and per >= SHAPES_PER_HUNDRED
                          and longest <= IDENTICAL_RUN_MAX)}
