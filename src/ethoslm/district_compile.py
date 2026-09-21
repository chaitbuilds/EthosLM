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

#: **What share of a district's lots may be a building of a use other than the quarter's
#: own**, where its programme asks for that use at all -- and nothing where it does not.
#: Registered before the numbers that test it, with its reasoning: a tenth is one
#: building in a block of ten, which is a guild hall on a long street of houses; a third
#: -- which is what `OTHER_EVERY` over an undifferentiated pool produced -- is a civic
#: precinct with houses in it. A district asks for another use by naming it in the
#: demand the request resolved (`requirement_for`), not by having it in the pool: a pool
#: is what the fabric *may* be built of and a programme is what the quarter is *for*.
PROGRAMME_USE_SHARE = 0.1

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
ARRANGEMENT_FIELDS = ("rows", "lot_width", "lot_depth", "frontage", "attached",
                      "courtyard_share", "block")

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
    for k, v in ((district or {}).get("arrangement") or {}).items():
        if v is None or k not in ARRANGEMENT_FIELDS:
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
                  u_is_x: bool = True) -> tuple | None:
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
            if all(_admits(decl, w, d, att, u_is_x)
                   for att in (flanks, flanks[:1], flanks[1:], ())):
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


def _use_slot(laid: int, share: float) -> bool:
    """Is the `laid`-th lot of this district one of the programme's other-use share?

        By position and not by draw: `share` of the lots, spread evenly, so a tenth is a tenth
        of the district rather than a tenth in expectation that may clump at one end. The same
        rule the voice's second wall material is spread by (`WALL_ALT_SHARE`), for the same
        reason -- a quarter of a street clumped at one end is not a quarter of a street.
        
    """
    s = float(share or 0.0)
    if s <= 0:
        return False
    return int((int(laid) + 1) * s) > int(int(laid) * s)


def use_mix(district: dict | None, ch: dict, houses: list, role: str | None) -> dict:
    """**Which uses this district's fabric is drawn from, and in what share.**

        The composition round's second change, and the user's own instruction for it: "Make
        district uses govern type selection... Allow supported variation within the intended
        building uses, rather than substituting civic buildings for variety."

        What it replaces: the compiler drew every lot's type from *everything the pool admits*,
        weighted 2:1 toward the district's own role, with `OTHER_EVERY` forcing an unweighted
        draw over the whole pool every fourth lot. The pool is the capability record's approved
        fabric, and the middle ring's includes `hall` and `temple` because the ring's landmark
        programme needs civic types -- so the built section came back with **5 temples and 4
        halls against 1 shop house in 22 buildings**, in the ring the sources describe as
        traders, craftsmen and schoolteachers. A type pool is what a quarter may be *built of*;
        it is not what the quarter is *for*.

        So the admitted types are split by use:

          `own`        the quarter's own use: a type whose declared `ROLE` is the district's.
                       The fabric is drawn from these, and `variety` varies among them -- five
                       kinds of dwelling and shop on the middle ring, which is variation inside
                       the intended use.
          `programme`  a type of another use that this district's **programme** asks for: one
                       whose declaration answers a requirement the district's resolved demand
                       carries (`requirement_for`), or one the character names as a landmark.
                       Drawn at `PROGRAMME_USE_SHARE` and no more.
          `unasked`    every other type the pool admits. Not drawn at all. A civic building is
                       an exception the programme asks for, not every third lot.

        Nothing here names a role, a type or a place: `role` is the district's own and the
        requirement is the one the request resolved. A district whose pool holds only its own
        use -- the lower ring's single `row_house` -- behaves exactly as it did.
        
    """
    want = str(role or "")
    own, other = [], []
    for name, decl in houses or []:
        if not want or str(decl.get("role") or "") == want:
            own.append((name, decl))
        else:
            other.append((name, decl))
    named = {str(lm.get("type")) for lm in (ch.get("landmarks") or []) if lm.get("type")}
    programme, unasked, why = [], [], []
    for name, decl in other:
        rid = requirement_for(name, decl, district)
        if rid:
            programme.append((name, decl))
            why.append(f"`{name}` answers the requirement `{rid}` this district resolved")
        elif name in named:
            programme.append((name, decl))
            why.append(f"`{name}` is the landmark the character names")
        else:
            unasked.append((name, decl))
    if unasked:
        why.append(f"{', '.join(n for n, _d in unasked)} are of another use "
                   f"(`{'`, `'.join(sorted({str(d.get('role')) for _n, d in unasked}))}`) "
                   f"that nothing in this district's programme asks for, so the fabric is "
                   f"not drawn from them")
    if not own and other:
        # a pool with none of the district's own use is a capability question and not
        # the compiler's to answer: it lays what it was given and the record says so
        own, programme, unasked = other, [], []
        why.append(f"no type of this pool is of the district's own `{want}` use; the "
                   f"fabric is laid from what the pool has and the mix is the pool's")
    return {"own": own, "programme": programme, "unasked": unasked,
            "share": PROGRAMME_USE_SHARE if programme else 0.0,
            "role": want, "from": why}


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
            if ch["courtyard_share"] > 0:
                ch["courtyard_share"] = max(0.0,
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
    # **What this district is for decides what it is built of** (`use_mix`). The
    # quarter's own use is what the fabric is drawn from; a building of another use
    # appears only where the programme asks for it, at `PROGRAMME_USE_SHARE`. The
    # primary house is the first of the quarter's own uses in the pool's own order, so a
    # capability record that put a civic type first no longer makes it the district's
    # house.
    mix = use_mix(district, ch, houses, role)
    kin = mix["own"]
    house_name, house = kin[0]
    if attached:
        # v2, C2: a row of party walls is built of a type that declares `ATTACHED`, at a
        # lot every configuration of its flanks admits; where the role has no such type,
        # or none admits a lot, the row is detached and the record says
        terraced = [(n, d) for n, d in kin if d.get("attached")]
        pick = None
        for n, d in terraced:
            got = _attached_lot(d, side, fr.flanks(), fr.u_is_x)
            if got:
                pick = (n, d, got)
                break
        if pick:
            house_name, house, (w, ld) = pick
            if ch.get("lot_depth"):
                d2 = int(ch["lot_depth"])
                fl = fr.flanks()
                if all(_admits(house, w, d2, att, fr.u_is_x)
                       for att in (fl, fl[:1], fl[1:], ())):
                    ld = d2
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
    # the quarter's own alternatives, and -- kept apart -- the other use its programme
    # asked for. Two lists because they are drawn by two different rules.
    others = [(n, d) for n, d in kin if n != house_name
              and _admits(d, w, ld, None, fr.u_is_x) and not attached]
    asked = [(n, d) for n, d in mix["programme"]
             if _admits(d, w, ld, None, fr.u_is_x) and not attached
             and d.get("kind", "plot") == "plot"]
    per_block = int(ch.get("_block_lots") or BLOCK_LOTS.get(density, 3))
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
                               widen=not attached and not ceiling_binds)
    rows, axial_v = _grid_axis(band, 1, fr.V, street, bd, ld, lead=lead)
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
    # the free block nearest the district's middle takes the landmark
    cu, cv = (fr.U - 1) / 2.0, (fr.V - 1) / 2.0
    order = sorted(blocks, key=lambda ij: (abs((cols[ij[0]][0] + cols[ij[0]][1]) / 2.0 - cu)
                                           + abs((rows[ij[1]][0] + rows[ij[1]][1]) / 2.0 - cv),
                                           ij))
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
    #: The shape of every house, street face by street face, in the order it was laid:
    #: the type, the width, the depth and the storeys, which is what a person walking
    #: down the street sees of it. The craft round, E3, and the two numbers the phase
    #: registers are read off this.
    faces: list = []
    #: What the district still owes the voice's second wall material, in buildings.
    alt_debt = [0.0]

    def leaf(kind, name, tname, decl, u0, u1, v0, v1, front=None, notes=""):
        x0, z0, x1, z1 = fr.rect(u0, u1, v0, v1)
        n_leaf[0] += 1
        k = n_leaf[0]
        rng = random.Random(f"{seed}/{district['name']}/{k}")
        params = {}
        row_env = None
        for pn, spec_p in (decl.get("params") or {}).items():
            if not isinstance(spec_p, (list, tuple)) or not spec_p:
                continue
            if spec_p[0] == "int" and len(spec_p) >= 3:
                lo_p, hi_p = int(spec_p[1]), int(spec_p[2])
                if pn == "storeys":
                    lo_p, hi_p = _storeys_band(decl, ch)
                    # **Only the storeys this lot admits are asked** (the expression
                    # round): a cottage asked for two on a lot that holds one emitted
                    # one and the record called the loss a constraint. Where the
                    # envelope module can say what the lot admits, the band is capped to
                    # it and the cap is on the row; the ask and the outcome agree.
                    got = _storeys_fit(tname, lo_p, hi_p, x1 - x0 + 1, z1 - z0 + 1,
                                       cache=(place.get("layout") or {}).get("envelope_cache")
                                       if isinstance(place.get("layout"), dict) else None,
                                       demand=(dem if dem and dem.get("types")
                                               and tname in (dem.get("types") or ())
                                               else None)) or {}
                    fit = got.get("storeys")
                    row_env = None
                    if fit is not None and fit < hi_p:
                        row_env = {"storeys_band": [lo_p, hi_p],
                                   "storeys_admitted": int(fit),
                                   "why": got.get("why") or
                                          f"the {x1 - x0 + 1}x{z1 - z0 + 1} lot admits "
                                          f"{fit} storey(s) of {tname} by its envelope"}
                        hi_p = max(lo_p, fit)
                    elif got and not got.get("holds", True):
                        # **the floor itself does not fit.** The ask stands: the
                        # shortfall is recorded so construction emits the constraint and
                        # the obligation ledger owns it, rather than the parameter being
                        # quietly lowered here.
                        row_env = {"storeys_band": [lo_p, hi_p],
                                   "storeys_admitted": None,
                                   "required": bool(got.get("required")),
                                   "why": got.get("why") or
                                          f"a {x1 - x0 + 1}x{z1 - z0 + 1} lot does not "
                                          f"admit {lo_p} storey(s) of {tname}"}
                params[pn] = rng.randint(lo_p, hi_p)
            elif spec_p[0] == "choice" and len(spec_p) >= 2 and spec_p[1]:
                params[pn] = rng.choice(list(spec_p[1]))
        row = {"kind": kind, "name": name, "type": tname, "seed": 1 + (k % 89),
               "params": params, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
               "notes": notes, **({"envelope": row_env} if row_env else {})}
        if front:
            row["front"] = front
        return row

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

    def lots_along(u0, u1, v0, v1, front, i, j, r, plots, depth=None):
        """A row of lots along one street face of a block, in [u0, u1] x [v0, v1].

                **A street is a rhythm and not a comb**, the craft round (E3). The run used to
                be divided evenly, so every lot was one width, every building the same type and
                every roof the same height -- twenty-one cottages on identical pads. Each lot
                now draws its own width and its own depth inside the band the character sets
                (`spread`), and its own type from everything the role admits at that size; the
                widths still tile the run exactly, because the last lot takes what is left.
                
        """
        if counts["lots"] >= want_lots:
            return      # the district has the houses it was asked for, or asked none
        dd0 = ld if depth is None else depth
        n = u1 - u0 + 1
        k = (n + gap) // (w + gap)
        if k < 1:
            return
        lo, hi, _ex = _plot_range(house)
        # the street is at the low-v edge of this row where `front` is the low-v side; a
        # lot shallower than the row keeps its face on the street and gives its back
        at_low = (front == fr.front(True))
        widths = [w] * k
        left = n - (k * w + (k - 1) * gap)
        if 0 < left <= WIDEN_UP_TO * n and not attached and not ceiling_binds:
            # widen the lots to take the remainder up, inside the house's envelope
            per = left // k
            for q in range(k):
                widths[q] = min(hi, w + per)
            for q in range(left - per * k):
                widths[q] = min(hi, widths[q] + 1)
            widths = [ww if stands(house, ww, dd0) else w for ww in widths]
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
            if not free(ua, ub, va, vb):
                dropped["arterial" if band[ua:ub + 1, va:vb + 1].any()
                        else "standing"] += 1
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
            # **...and the other use the programme asked for, at its own share and
            # spread evenly down the street.** `PROGRAMME_USE_SHARE` of the district's
            # lots, by position rather than by draw, so a tenth is a tenth of the
            # district and not a tenth in expectation that clumps at one end -- the same
            # rule the voice's second wall material is spread by. A district whose
            # programme asked for no other use never reaches this line.
            if asked and _use_slot(counts["lots"], mix["share"]):
                here = [(n2, d2) for n2, d2 in asked if stands(d2, ww, dd)]
                if here:
                    pool = here
                    counts["programme_uses"] = counts.get("programme_uses", 0) + 1
            tname, decl = pool[int(_seeded(seed, "type", i, j, r, q) * len(pool))
                               % len(pool)]
            if not stands(decl, ww, dd):
                tname, decl = house_name, house
            if not stands(decl, ww, dd):
                dd = dd0
                va, vb = (v0, v0 + dd - 1) if at_low else (v1 - dd + 1, v1)
                if not stands(decl, ww, dd) or not free(ua, ub, va, vb):
                    continue
            if counts["lots"] >= want_lots:
                break
            # **An open-frontage lot records a front where the design demands one.**
            # `open` frontage means there is no street edge to build to, and it used to
            # mean the plan recorded no orientation at all -- so a district asked to
            # face the water laid houses whose facing nothing had decided and nothing
            # could check. Which way a house faces and whether it stands on a street are
            # two questions; a frontage obligation answers the first either way.
            _front = (faces_side or None) if open_front else front
            row.append((ua, ub, leaf("plot", f"b{i}_{j}_{r}{q}", tname, decl, ua, ub,
                                     va, vb, front=_front,
                                     notes=(f"a {tname} fronting the street on its "
                                            f"{front} side") if not open_front else
                                           (f"a {tname} on open ground, facing "
                                            f"{_front}" if _front else
                                            f"a {tname} on open ground"))))
            # **and no two neighbours are the same building.** Where the draw comes back
            # with the shape the lot before it has, and the type is one that can be
            # another height, it is stepped on: the one lever a terrace of a single
            # attached type at a single width still has.
            got_leaf = row[-1][2]
            lo_s, hi_s = _storeys_band(decl, ch)
            shape = (tname, ww, dd, int((got_leaf.get("params") or {})
                                        .get("storeys", 1)))
            if face and face[-1] == shape and hi_s > lo_s:
                nxt = lo_s + ((shape[3] - lo_s + 1) % (hi_s - lo_s + 1))
                got_leaf["params"]["storeys"] = nxt
                shape = (tname, ww, dd, nxt)
            plots.append(got_leaf)
            face.append(shape)
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
            for _ua, _ub, p_leaf in row:
                p_leaf.setdefault("wall_alt", False)
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
                    ua = u0 + ((u1 - u0 + 1) - s) // 2
                    if free(ua, ua + s - 1, v0, v0 + s - 1):
                        plots.append(leaf("plot", f"landmark_{tname}", tname, decl,
                                          ua, ua + s - 1, v0, v0 + s - 1,
                                          front=fr.front(True),
                                          notes=lm.get("notes") or
                                          f"the {tname} the character names, on the "
                                          f"block nearest the middle"))
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
                    ua = u0 + ((u1 - u0 + 1) - s) // 2
                    if free(ua, ua + s - 1, v0, v0 + s - 1):
                        plots.append(leaf("area", f"landmark_{tname}", tname, decl,
                                          ua, ua + s - 1, v0, v0 + s - 1,
                                          front=fr.front(True),
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
        if kind == "courtyard":
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
    ceiling_cols = target.get("max_plot_columns")
    if ceiling_cols is not None and not exact:
        def _plot_cols():
            return sum((p["x1"] - p["x0"] + 1) * (p["z1"] - p["z0"] + 1)
                       for q in quarters.values() for p in q["plots"]
                       if p["kind"] == "plot")
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
           "character": {k: v for k, v in ch.items()},
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
    # two are reported apart so the difference is visible.
    foot_cols = footprint_columns(leaves)
    record = {"district": district["name"], "part": part.get("name"), "seed": seed,
              "asked_for": want_lots,
              "character": ch, "house": house_name, "attached": attached,
              "attached_note": attached_note,
              "others": [n for n, _d in others], "lot": [w, ld], "gap": gap,
              "lot_min": _least_lot(district.get("lot_min"), dem_lot),
              "lot_min_raised": lot_min_raised,
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
              # what that let it be built of**: the quarter's own uses, the other use
              # its programme asked for and at what share, and the types the pool held
              # that nothing asked for. 5 temples and 4 halls against 1 shop house in 22
              # buildings of the middle ring.
              "use_mix": {"role": mix["role"],
                          "own": [n for n, _d in mix["own"]],
                          "programme": [n for n, _d in mix["programme"]],
                          "unasked": [n for n, _d in mix["unasked"]],
                          "share": mix["share"],
                          "laid": int(counts.get("programme_uses") or 0),
                          "from": list(mix["from"])},
              "reservations": [dict(x) for x in reserve],
              "reservations_kept": int(counts["landmarks"]),
              "reservations_required": sum(1 for x in reserve if x.get("required")),
              "demand_short": [dict(x) for x in demand_short],
              "reserve_drove": reserve_drove,
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
              # the four columns of this region (`placeregion.COLUMN_FIELDS`), as far as
              # a plan can answer them: what was allocated and what mass it admits.
              # `built_columns` stays the emitted world's to fill.
              "allocated_columns": int(plot_cols),
              "footprint_columns": int(foot_cols),
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
              "lot_cover_usable": round(plot_cols / float(max(1, target["usable_columns"])), 4),
              "over_ceiling": bool(target.get("max_plot_columns") is not None
                                   and plot_cols > target["max_plot_columns"]),
              "registered_plot_cover": PLOT_COVER[density],
              "meets_registered_cover": plot_cols / float(total) >= PLOT_COVER[density],
              "variety": _variety(faces),
              "wall_alt": _wall_alt_record(leaves)}
    return got, record


def footprint_columns(leaves: list) -> int:
    """**The mass these leaves stand as**, in columns: the pad `site()` will hand
        `build()` for each plot, summed -- not the lots they stand on.

        The design round's escape route E2, in one number. A district whose lots grew from
        6x7 to 10x10 covers more ground with the same fifteen houses, and a density figure
        read off the lots calls that denser construction. The pad is what a person walking
        the street sees standing; an empty lot is not a building.
        
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
                 cache: str | None = None, demand=None) -> int | None:
    """The most storeys in `lo..hi` this type delivers on a `w`x`d` lot, or None where
        no envelope can answer.

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
        return d_mod.storeys_admitted(demand, int(w), int(d), cache=cache)
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
            got = fn(tname, {"storeys": st}, features=("storeys",), cache=cache) or {}
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
    return {"lots": len(flat), "shapes": len(kinds), "faces": len(faces),
            "per_hundred": round(per, 1), "longest_run": int(longest),
            "registered": {"shapes_per_hundred": SHAPES_PER_HUNDRED,
                           "identical_run": IDENTICAL_RUN_MAX},
            "holds": bool(flat and per >= SHAPES_PER_HUNDRED
                          and longest <= IDENTICAL_RUN_MAX)}
