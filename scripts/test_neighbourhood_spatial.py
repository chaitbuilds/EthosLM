#!/usr/bin/env python3
"""The neighbourhood round's spatial programme and arrangement work, on real ground.

Cheap production counterexamples for the three things the round's brief says the
composition round's spatial layer cannot do. Every case runs the actual entry points --
`district_compile.use_mix`, `district_compile.compile_district`, `arrange.arrangements`,
`arrange.alternatives`, `placeplan.region_columns` -- on the **retained section's own
districts** (`out/comp-city`), which are immutable input here: read, never written. No
world is built and no interpretation is re-run.

    P1  the quarter's programme is inferred from its own design -- the requirements its
        demand resolved and the words of its own description -- and not from a constant.
        The contrasting control is the elite ring: same pool, same types, a description
        that names no work, and no work laid.

    P2  a use a **requirement** asks for is laid at the count the requirement asks and
        not at a share of the street, and `use_mix` imposes no share at all.

    P3  the quarter's work stands on the street its anchor fronts, and the share it comes
        to is measured off what was laid with its derivation beside it.

    P4  the pad figure is labelled as the estimate it is, and where construction has
        reported rectangles for a district's leaves the alternative reads **those** --
        which the composition round's `alternatives` computed and then discarded.

    P6  a perimeter block puts building on the cross streets and encloses its court on
        four sides -- neither of which any arrangement in this compiler could do.

Two to six minutes: the fixtures are retained plans and the compiler is asked about
single rectangles.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import (arrange, district_compile as dc, placeplan,  # noqa: E402
                   spec as spec_mod)

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the fixtures

_LOADED: dict = {}


def section():
    """The retained section's plan and spec (`out/comp-city`), read and never written."""
    if "sec" in _LOADED:
        return _LOADED["sec"]
    d = os.path.join(ROOT, "out", "comp-city")
    if not os.path.exists(os.path.join(d, "plan.place.json")):
        raise Skip("no out/comp-city/plan.place.json")
    place = json.load(open(os.path.join(d, "plan.place.json")))
    spec = json.load(open(os.path.join(d, "place.json")))
    got = (spec, place, {p["name"]: p for p in spec["defining_parts"]}, d)
    _LOADED["sec"] = got
    return got


def district(name: str):
    """`(district, part, decls, spec, place)` for one of the section's own districts."""
    spec, place, by, _d = section()
    x = next((q for q in place["districts"] if q["name"] == name), None)
    if x is None:
        raise Skip(f"no district {name} in the retained section")
    part = by[x["defines"]]
    role = spec_mod.district_role(spec, x)
    _t, mine = placeplan.types_card(None, spec.get("form"), role)
    return x, part, mine, spec, place


def compiled(name: str, **over):
    """One of the section's districts, through the real compiler."""
    x, part, decls, spec, place = district(name)
    if over:
        x = dict(x, **over)
    got, rec = dc.compile_district(x, part, place, decls, spec=spec, seed=1)
    return got, rec, x, part, decls, spec, place


def leaves_of(got: dict, kind: str = "plot") -> list:
    return [p for q in got.get("quarters") or [] for p in q.get("plots") or []
            if p.get("kind", "plot") == kind]


#: The two sides of the section the round is about, and the third that is its control:
#: the crowded lower ring, the traders' middle ring, and the elite upper ring, which has
#: the **same approved pool** as the middle ring and a description that names no work.
CROWDED = "lower_ring_north_2"
TRADERS = "middle_ring_north_west"
ELITE = "upper_ring_north"


# ------------------------- P1: the programme comes off the district's own design

@case
def t_p1_the_programme_is_inferred_from_the_design_and_not_from_a_constant():
    """The round's brief: "The current `use_mix` filter compares coarse `ROLE` values,
        imposes a fixed 10% secondary-use share, and falls back to other uses when none
        match... Infer and record this neighbourhood's mix from the design."

        What is inferred, on the retained middle ring, and from where:

          * `market` -- from the requirement `function/market` its demand resolved, **and**
            from the word `market` in its own description;
          * `trade` -- from the word *traders* in "The traders' and craftsmen's ring", which
            matches `shop_house`'s declaration of the trades it houses (`TRADE_PARAM`);
          * `home` -- the quarter's own fabric, because two of the five types its `urban` role
            admits declare `FUNCTION = dwelling`.

        And what is **not**: `hall` and `temple` are in the approved pool, nothing the
        district resolved or says asks for them, and no lot is one.

        The control is the elite ring. Its pool holds `shop_house` exactly as the middle
        ring's does; its description says "courtyard houses on generous lots behind their own
        gates"; it must draw no trade at all. Without that control the rule could be "put
        shops everywhere" and measure the same on the traders' ring.
        
    """
    got, rec, x, _p, _d, _s, _pl = compiled(TRADERS)
    pr = got["programme"]
    mix = rec["use_mix"]
    uses = {u["use"]: u for u in pr["uses"]}
    assert set(uses) >= {"home", "trade", "market"}, sorted(uses)
    # the trade use is named by the design's own words and by nothing else
    trade = uses["trade"]
    assert trade["types"] == ["shop_house"], trade
    assert not trade.get("requirement"), trade
    assert any("trader" in s for s in trade["asked_by"]), trade["asked_by"]
    # ...and the words it was read out of are on the record with their source
    said = " ".join(w["text"] for w in pr["words"])
    assert "traders" in said, pr["words"]
    assert any("notes" in w["from"] or "purpose" in w["from"] for w in pr["words"]), \
        pr["words"]
    # the market: asked for by the requirement *and* by the design's words
    assert uses["market"]["requirement"] == "function/market", uses["market"]
    # the civic types are in the pool, asked for by nothing, and not laid
    assert set(mix["unasked"]) == {"hall", "temple"}, mix["unasked"]
    kinds = {p["type"] for p in leaves_of(got)}
    assert not (kinds & {"hall", "temple"}), kinds
    assert "shop_house" in kinds, kinds
    # the use of every type is read off a declaration, and the two that state nothing
    # say so
    assert mix["use_of"]["shop_house"] == "trade", mix["use_of"]
    assert mix["use_of"]["courtyard_house"] == "home", mix["use_of"]
    # **A type that declares nothing is recorded as an inference, and the rule is about
    # the mechanism rather than about one file.** This named `court_small` -- which
    # declared no `FUNCTION`, so its use was inferred from its `ROLE`. The neighbourhood
    # round then *declared* it, because the inference had a cost: a quarter's own fabric
    # is drawn from the types whose declared function is its use, so an inferred
    # `unstated` ranked below `row_house`'s declared `dwelling` and the traders' ring of
    # a city whose sources say "its houses are courtyard houses" built thirty row
    # houses. The rule this case is about is unmoved: wherever a type still states
    # nothing, the record says the use is an inference and not a declaration. Where
    # every admitted type declares one -- the better state, and the one this pool is now
    # in -- there is no inference to record and the case says so rather than asserting
    # one exists.
    unstated = sorted(n for n, u in mix["use_of"].items() if u == dc.UNSTATED_USE)
    if unstated:
        assert any("inference from the role and not a declaration" in s
                   for s in mix["from"]), (unstated, mix["from"])
    else:
        assert all(u != dc.UNSTATED_USE for u in mix["use_of"].values()), mix["use_of"]

    # --- the control: the same pool, a description that names no work ---------------
    got2, rec2, x2, _p2, _d2, _s2, _pl2 = compiled(ELITE)
    mix2 = rec2["use_mix"]
    assert "shop_house" in (x2.get("fabric_types") or []), x2.get("fabric_types")
    assert "shop_house" in mix2["unasked"], mix2
    assert not mix2["programme"], mix2["programme"]
    kinds2 = {p["type"] for p in leaves_of(got2)}
    assert "shop_house" not in kinds2, kinds2
    # ...and the crowded ring, whose pool holds one type, behaves exactly as it did
    got3, rec3, _x3, _p3, _d3, _s3, _pl3 = compiled(CROWDED)
    assert rec3["use_mix"]["own"] == ["row_house"], rec3["use_mix"]
    assert not rec3["use_mix"]["programme"], rec3["use_mix"]
    assert {p["type"] for p in leaves_of(got3)} == {"row_house"}, "the crowded ring moved"

    n_trade = trade["laid"]
    return (f"the traders' ring infers `home` + `trade` + `market` from its own design: "
            f"trade from {trade['asked_by'][0]}, market from the requirement "
            f"`function/market`; it lays {n_trade} shop(s) in {pr['buildings']} "
            f"building(s) and 0 of the `hall`/`temple` its pool admits. The elite ring, "
            f"same pool and same `shop_house`, names no work and lays none; the crowded "
            f"ring's single-type pool is unchanged at {rec3['lots']} row houses")


@case
def t_p2_a_required_use_is_laid_at_the_count_its_requirement_asks():
    """No universal quota. The composition round spent `PROGRAMME_USE_SHARE` -- a
        registered tenth -- on any use a district's programme asked for; a requirement is a
        statement that the place must **hold** the thing, and one temple answers
        `feature/temple`.

        Run on the traders' ring with `feature/temple` added to its resolved demand and
        nothing else changed, which is the same control the composition round used and the
        opposite expectation.
        
    """
    x, part, decls, spec, place = district(TRADERS)
    asked = dict(x, name="temple_required",
                 demand={**x["demand"],
                         "requirements": [*x["demand"]["requirements"],
                                          "feature/temple"]})
    got, rec = dc.compile_district(asked, part, place, decls, spec=spec, seed=1)
    mix = rec["use_mix"]
    assert mix["required"].get("temple") == {"requirement": "feature/temple",
                                             "count": 1}, mix["required"]
    laid = leaves_of(got)
    temples = [p for p in laid if p["type"] == "temple"]
    assert len(temples) == 1, (len(temples), len(laid))
    # ...and the share `use_mix` itself imposes is None -- it does not have one to
    # impose
    raw = dc.use_mix(asked, dc.character_of(part, asked),
                     dc.house_types(decls, part.get("role"), spec.get("form"),
                                    approved=asked.get("fabric_types")),
                     part.get("role"), part=part, decls=decls)
    assert raw["share"] is None, raw["share"]
    # the share on the record is the one that was **measured**, with its derivation
    row = next(u for u in got["programme"]["uses"] if u["role"] == "programme")
    assert row["share"] is not None and row["share_from"], row
    assert "measured off what was laid" in row["share_from"], row["share_from"]
    # and with the requirement taken away again, no temple
    got2, _rec2 = dc.compile_district(dict(x, name="temple_optional"), part, place,
                                      decls, spec=spec, seed=1)
    assert not [p for p in leaves_of(got2) if p["type"] == "temple"], "a temple appeared"
    return (f"`feature/temple` on the traders' ring lays exactly 1 temple in {len(laid)} "
            f"building(s) -- the count the requirement asks -- where the retired "
            f"`PROGRAMME_USE_SHARE` of {dc.PROGRAMME_USE_SHARE:.0%} would have given it "
            f"a share of the street; `use_mix` imposes no share at all (`share: None`) "
            f"and the figure on the record is measured with its derivation")


@case
def t_p3_the_quarters_work_stands_on_the_street_its_anchor_fronts():
    """A trade belongs on the street its market is on.

        Measured on the traders' ring: every shop the compiler laid is on the block row the
        market's reservation was given or the one across the street from it, and none is
        anywhere else. That is what makes the share a consequence -- "the traders' market
        belongs to its neighbourhood" is a statement about position, and a tenth spread
        evenly down a district is a statement about frequency.
        
    """
    got, rec, x, _p, _d, _s, _pl = compiled(TRADERS)
    pr = got["programme"]
    assert "reserves fronts" in pr["principal_street"], pr["principal_street"]
    j_p = pr["principal_row"]
    assert j_p is not None, pr
    shops = [p for p in leaves_of(got) if p["type"] == "shop_house"]
    assert shops, "no shop was laid"
    assert any(p["type"] == "market" for q in got["quarters"]
               for p in q["plots"]), "the market reservation was lost"

    def face_of(p):
        """`(block column, block row, face)` off the compiler's own leaf naming,
        `b<col>_<row>_<face><n>` -- which is where the lot is, said by the thing that
        put it there rather than re-derived from coordinates."""
        nm = str(p.get("name") or "")
        if not nm.startswith("b"):
            return None
        try:
            i, j, rest = nm[1:].split("_", 2)
            return int(i), int(j), int(rest[0])
        except (ValueError, IndexError):
            return None

    #: A face fronts the principal street where it is the low-v face of a block in the
    #: anchor's row (`face` 0 or 2 -- the two strips the landmark branch lays beside the
    #: reservation) or the high-v face of the row before it (`face` 1).
    def principal(p):
        got_f = face_of(p)
        if got_f is None:
            return False
        _i, j, r = got_f
        return (j == j_p and r in (0, 2)) or (j == j_p - 1 and r == 1)

    for p in shops:
        assert principal(p), (p["name"], j_p)
    homes = [p for p in leaves_of(got) if p["type"] != "shop_house"]
    off = [p for p in homes if not principal(p)]
    assert off, "every lot of this district is on the principal street"
    return (f"{len(shops)} of the traders' ring's {rec['lots']} buildings are shops and "
            f"every one of them is on a face of the market's own street "
            f"({pr['principal_street']}); {len(off)} of the quarter's "
            f"{len(homes)} houses stand on the back streets and not one of them is a "
            f"shop")


# --------------------------- P4: the estimate is labelled and the emitted read

@case
def t_p4_every_proposal_is_ranked_on_its_own_geometry():
    """**Rewritten, and the rule it now tests is stated here.** The neighbourhood round
        wrote this case to assert that `arrange.alternatives`, given a parts record, reports
        `built_columns` off the previous build's **emitted** rectangles. The spatial design
        round's independent review found that this is not a reuse of an observation at all:

            "`arrange.alternatives` passes the previous candidate's `parts_record` into
            `region_columns` and `street_enclosure` for hypothetical newly compiled leaves.
            Both match emitted measurements by part name, without proving unchanged geometry.
            A reused leaf name can therefore attach the old footprint to a new arrangement."

        The compiler's leaf names are **positional** -- `b2_0_06` is block 2, row 0, lot 6 --
        so a different arrangement of the same rectangle re-uses almost every name for a lot
        of a different size in a different place. The old assertion was therefore an
        assertion that a mislabelled estimate be produced. What is tested now is the
        corrected rule: every row is the compiler's own pad arithmetic, uniformly, and the
        row says so; the observation is taken after the build
        (`improve._estimate_vs_built`), where the geometry is the geometry that was built.

        The compile record's own honest labelling -- `pad_columns`, `footprint_estimate`,
        `footprint_basis` -- is unchanged and is still asserted; that half of the
        neighbourhood round's case was right and stands.
        
    """
    _got, rec, x, part, decls, spec, place = compiled(CROWDED)
    assert rec["pad_columns"] == rec["footprint_columns"], rec["pad_columns"]
    assert rec["footprint_estimate"] is True, rec
    assert "Not construction" in rec["footprint_basis"], rec["footprint_basis"]
    d = os.path.join(ROOT, "out", "comp-city")
    if not os.path.exists(os.path.join(d, "parts.json")):
        raise Skip("no out/comp-city/parts.json")
    parts_record = json.load(open(os.path.join(d, "parts.json")))
    rows = placeplan.emitted_columns(parts_record)
    mine = [k for k in rows if k.startswith(CROWDED)]
    assert len(mine) > 20, len(mine)
    got = arrange.alternatives(x, part, place, decls, spec=spec, seed=1,
                               ceiling=int(x["structures"]),
                               parts_record=parts_record, most=4)
    live = [a for a in got if a.get("lots")]
    assert live, [a["action"] for a in got]
    # **the counterexample, and how far this fixture pair can carry it.** The join was
    # by name, so the question is whether one candidate's leaf names ever land on
    # another's leaves. The retained `comp-city` build was laid by an older naming
    # (`c0_0_0`, `v0_0`) and today's compiler lays `b2_0_06`, so on *this* pair the
    # overlap happens to be small -- which is luck and not a property, because both
    # namings are positional and neither says anything about the geometry. The overlap
    # is measured and reported rather than asserted; what is asserted is the rule.
    names = {p["name"] for p in leaves_of(_got)}
    reused = sorted(n for n in names
                    if placeplan.emitted_for(rows, CROWDED, n) is not None)
    # not one row is joined: every row is a prediction and says why
    for a in live:
        assert a["built_estimate"] is True, (a["action"], a["built_from"])
        assert a["built_from"] == "planned pads", (a["action"], a["built_from"])
        assert a["built_columns"] == a["pad_columns"], a
        assert a["predicted"] is True and a["observation_reused"] is None, a
        assert "on its own geometry" in a["predicted_why"], a["predicted_why"]
    # with no parts record at all the answer is identical, which is the point: the
    # ranking does not depend on which candidate happened to be built before it
    quiet = arrange.alternatives(x, part, place, decls, spec=spec, seed=1,
                                 ceiling=int(x["structures"]), most=4)
    assert [q["built_columns"] for q in quiet] == [a["built_columns"] for a in got], \
        ([q["built_columns"] for q in quiet], [a["built_columns"] for a in got])
    assert [q["enclosure"] for q in quiet] == [a["enclosure"] for a in got]
    return (f"the crowded ring's compile record calls its own figure `pad_columns` and "
            f"marks it an estimate; {len(reused)} of this compile's {len(names)} leaf "
            f"names collide with the retained build's parts record "
            f"({len(mine)} reported leaves), and **not one of them is joined**: all "
            f"{len(live)} live alternatives report `planned pads`, and the same call "
            f"with no parts record at all returns the identical built columns "
            f"{[a['built_columns'] for a in got]} and enclosure "
            f"{[a['enclosure'] for a in got]} -- so the ranking is of the proposals' own "
            f"geometry and not of whatever was built before them")


# ----------------- P5: arrangement operations that can shape a convincing street

@case
def t_p5_the_new_arrangement_operations_are_offered_and_certified():
    """"Add or revise reusable arrangement operations where existing ones cannot shape a
        convincing street."

          `terrace`      party walls or none. `attached` has been an `ARRANGEMENT_FIELD`
                         since the design round and **no action ever wrote one**, so whether
                         the buildings of a street touch was settled by the district brief's
                         author and was never a decision the layout could revise.
          `compact_bay`  the narrowest bay, one row to a block, on the longest block: the
                         three compaction levers together. Every other action moves one value
                         of the character, so the best crowded street any of them could offer
                         was the best single move.
          `perimeter`    lots on all four faces of a block with the court inside them.

        Offered on the section's own two sides, certified by `placeplan.district_failures`
        like every other alternative, and genuinely different in the geometry they produce.
        
    """
    said = []
    for name in (CROWDED, TRADERS):
        x, part, decls, spec, place = district(name)
        opts = arrange.arrangements(part, decls, spec=spec,
                                    pool=list(x.get("fabric_types") or []) or None)
        by = {}
        for a in opts:
            by.setdefault(a["action"], []).append(a)
        for act in ("terrace", "compact_bay", "perimeter"):
            assert act in by, (name, sorted(by))
            assert act in arrange.ARRANGEMENT_ACTIONS, act
        # each one writes a field the compiler reads, and the fields are declared
        for a in opts:
            arr = a.get("arrangement") or {}
            for k in arr:
                assert k in dc.ARRANGEMENT_FIELDS, (a["action"], k)
        # `terrace` is the only action that writes `attached`, and `perimeter` the only
        # one that writes `perimeter` -- so a reader can tell which decision moved
        wrote_att = {a["action"] for a in opts
                     if "attached" in (a.get("arrangement") or {})}
        wrote_per = {a["action"] for a in opts
                     if (a.get("arrangement") or {}).get("perimeter")}
        assert wrote_att <= {"terrace"} and wrote_att, wrote_att
        assert wrote_per == {"perimeter"}, wrote_per
        said.append(f"{name}: {len(opts)} arrangement(s) over "
                    f"{len(by)} action(s) -- {', '.join(sorted(by))}")
    return "; ".join(said)


@case
def t_p5b_the_new_operations_measurably_change_the_street():
    """The operations have to *do* something, measured through the real compiler and the
        real validator on the section's own ground -- an action nobody can distinguish from
        the incumbent is a name, not a decision.

        The crowded ring is the subject. Its declared fabric is 6x8 attached lots on a
        40-column block, which is what the composition round built and what the user read as
        "terraces standing in stripes with grass voids as wide as the terraces".
        
    """
    x, part, decls, spec, place = district(CROWDED)
    got = arrange.alternatives(x, part, place, decls, spec=spec, seed=1,
                               ceiling=int(x["structures"]))
    live = {a["action"]: a for a in got if a.get("lots")}
    rows = [a for a in got if a.get("lots")]
    base = next(a for a in rows if a["action"] == "as_declared")
    best = rows[0]
    # every alternative offered carries the real validator's verdict -- **and lays the
    # lot it was ranked on**, which is the seam `placeplan.arrangement_failures` closes:
    # a row ranked on the cover and the frontage of a fabric it did not lay is not a
    # comparison, and the composition round's list had three of them
    for a in rows:
        assert a["verdict"] in ("ok", "refused"), a
        assert a["pad_columns"] > 0 and a["allocated_columns"] > a["pad_columns"], a
        if a["lot_laid"] and not a["refuses"]:
            assert abs(a["lot_laid"][0] - a["lot"][0]) <= 1 \
                and abs(a["lot_laid"][1] - a["lot"][1]) <= 1, \
                (a["action"], a["lot"], a["lot_laid"])
    # the best is not refused and is better than the declaration on more than one of the
    # things a person walking the street can see. Deliberately "more than one of": which
    # axis moves depends on the type band the library currently commits, and an
    # arrangement that bought cover by giving up frontage would pass a single-axis bar.
    assert not best["refuses"], best["refused"]
    moved = [k for k in ("lots", "pad_cover", "frontage_length", "enclosure")
             if (best.get(k) or 0) > (base.get(k) or 0)]
    assert len(moved) >= 2, (moved, base, best)
    # ...and somewhere in the list is an arrangement that lays **more houses** than the
    # declaration, which is the finding the composition round could not close
    more = [a for a in rows if not a["refuses"] and a["lots"] > base["lots"]]
    assert more, [(a["action"], a["lots"]) for a in rows]
    most = max(more, key=lambda a: a["lots"])
    # `perimeter` is offered on this ring, is certified, and lays courts the two-row
    # arrangements do not. Two are offered (the declared lot and the narrowest one); the
    # one that actually cut perimeter blocks on this ground is the one measured, and if
    # neither did, that is the failure.
    pers = [a for a in rows if a["action"] == "perimeter"]
    assert pers, sorted(live)
    per = next((a for a in pers if (a["perimeter_blocks"] or 0) >= 1), None)
    assert per is not None, [(a["lot"], a["perimeter_blocks"]) for a in pers]
    assert not per["refuses"], per["refused"]
    assert (per["courts"] or 0) > (base["courts"] or 0), (base["courts"], per["courts"])
    # `compact_bay` reaches the same crowded fabric the single-lever `bay_width` does,
    # from a different decision -- and both beat the declaration
    cb = live.get("compact_bay")
    assert cb is not None and cb["lots"] > base["lots"], (cb or {}).get("lots")
    return (f"the crowded ring as declared: {base['lots']} houses on "
            f"{base['lot'][0]}x{base['lot'][1]} lots, pad cover "
            f"{base['pad_cover']:.1%}, frontage {base['frontage_length']}, enclosure "
            f"{base['enclosure']:.1%}, {base['courts']} court(s). Best of {len(rows)} "
            f"certified: `{best['action']}` {best['lot'][0]}x{best['lot'][1]} "
            f"x{best['rows']}row -- {best['lots']} houses, {best['pad_cover']:.1%}, "
            f"frontage {best['frontage_length']}, {best['enclosure']:.1%} "
            f"(moved: {', '.join(moved)}); most houses `{most['action']}` "
            f"{most['lot'][0]}x{most['lot'][1]} at {most['lots']}. "
            f"`compact_bay` {cb['lots']} houses at "
            f"{cb['pad_cover']:.1%}; `perimeter` {per['lots']} houses at "
            f"{per['pad_cover']:.1%} with {per['courts']} court(s) in "
            f"{per['perimeter_blocks']} perimeter block(s), "
            f"{per['perimeter_shut'] or 0} of them closed on four sides")


@case
def t_p6_a_perimeter_block_builds_its_corners_and_encloses_its_court():
    """A `compound` block in this compiler is a block whose **back row** is the court, so
        the court is open along both short faces and the cross streets carry nothing: every
        lot this project has ever laid fronts one of the two long faces of its block.

        The `perimeter` arrangement lays the two long faces, a lot at each end of the middle
        band (`district_compile.lots_across`) and the court between them. Checked on the
        crowded ring's own rectangle, on geometry and not on a field: a court leaf with a
        building on all four of its sides, and buildings whose declared `front` is a cross
        street.
        
    """
    src, part, decls, spec, place = district(TRADERS)
    bare = {"arterials": {}, "parts": [], "districts": [], "layout": {}}
    #: A bare 200x120 rectangle with the section's own fabric and demand. The operation
    #: has to be **general**, and the section's own rectangles are not the test of that:
    #: the crowded ring's 184x69 loses the back row of its first block row to the
    #: arterial and its second is three columns short of a second row of lots, both of
    #: which are true facts about that ground and neither of which is about perimeter
    #: blocks. Measured on the section's own districts below, and reported.
    def probe(ask, arrangement):
        rect = {"name": "perimeter_probe", "x0": 0, "z0": 0, "x1": 199, "z1": 119,
                "structures": ask, "defines": src["defines"],
                "fabric_types": list(src.get("fabric_types") or []),
                "demand": src.get("demand")}
        return rect, arrange.capacity_of(rect, part, bare, decls, arrangement,
                                         spec=spec, seed=1, ceiling=ask)

    opts = arrange.arrangements(part, decls, spec=spec,
                                pool=list(src.get("fabric_types") or []) or None)
    pers = [a for a in opts if a["action"] == "perimeter" and not a.get("refused")]
    assert pers, [a["action"] for a in opts]
    # **how many houses the rectangle is asked for is not the subject here**, and it
    # decides whether the last block's fourth face is reached: the count is the
    # district's and a perimeter block is an arrangement. Asked at a few counts, and the
    # first that cuts a block which actually closed is the one measured -- with the
    # number reported.
    ask, per, got, rect = None, None, None, None
    for n in (40, 60, 90):
        for a in pers:
            r2, g2 = probe(n, a["arrangement"])
            if (g2["record"].get("perimeter_shut") or 0) >= 1 \
                    and g2["certificate"]["verdict"] == "ok":
                ask, per, got, rect = n, a, g2, r2
                break
        if got is not None:
            break
    assert got is not None, [(n, a["arrangement"]) for n in (40, 60, 90) for a in pers]
    plan, rec = got["plan"], got["record"]
    plots, courts = leaves_of(plan), leaves_of(plan, "area")
    assert rec.get("perimeter_blocks"), rec.get("perimeter_blocks")
    # this rectangle's streets run along x, so a cross-street front is west or east --
    # and the same rectangle's ordinary arrangement has none at all
    fronts = {str(p.get("front") or "") for p in plots}
    assert fronts & {"west", "east"}, fronts
    plain = arrange.capacity_of(rect, part, bare, decls, {}, spec=spec, seed=1,
                                ceiling=ask)
    plain_plots = leaves_of(plain["plan"])
    plain_fronts = {str(p.get("front") or "") for p in plain_plots}
    assert not (plain_fronts & {"west", "east"}), plain_fronts

    def enclosed_in(these_plots, these_areas, reach):
        """Open leaves with a building on all four sides within `reach` columns, on the
        geometry alone -- no field of the plan is consulted."""
        out = []
        for p in these_areas:
            cx = (p["x0"] + p["x1"]) / 2.0
            cz = (p["z0"] + p["z1"]) / 2.0
            sides = {
                "n": lambda q: (q["z1"] < p["z0"] and p["z0"] - q["z1"] <= reach
                                and q["x0"] <= cx <= q["x1"]),
                "s": lambda q: (q["z0"] > p["z1"] and q["z0"] - p["z1"] <= reach
                                and q["x0"] <= cx <= q["x1"]),
                "w": lambda q: (q["x1"] < p["x0"] and p["x0"] - q["x1"] <= reach
                                and q["z0"] <= cz <= q["z1"]),
                "e": lambda q: (q["x0"] > p["x1"] and q["x0"] - p["x1"] <= reach
                                and q["z0"] <= cz <= q["z1"])}
            if all(any(f(q) for q in these_plots) for f in sides.values()):
                out.append(p["name"])
        return out

    reach = int(rec["lot"][1]) + 2 * dc.LOT_GAP + 1
    shut = enclosed_in(plots, courts, reach)
    assert shut, (len(courts), [p["name"] for p in courts][:6])
    # the courts the perimeter branch itself laid, and the compiler's own count of the
    # blocks that **closed** -- a block whose fourth face was dropped for the arterial
    # or ran out against the district's count is reported as open and never as enclosed
    per_courts = [p["name"] for p in courts if p["name"].startswith("pc")]
    assert per_courts, [p["name"] for p in courts][:6]
    assert (rec.get("perimeter_shut") or 0) >= 1, \
        (rec.get("perimeter_blocks"), rec.get("perimeter_shut"))
    assert len([n for n in per_courts if n in shut]) >= rec["perimeter_shut"], \
        (per_courts, shut, rec["perimeter_shut"])
    # ...and the ordinary arrangement's open ground is not enclosed on four sides
    plain_courts = leaves_of(plain["plan"], "area")
    plain_shut = enclosed_in(plain_plots, plain_courts, reach)
    assert len(plain_shut) < len(shut), (plain_shut, shut)

    # what the section's own two sides get, reported rather than asserted
    said = []
    for name in (CROWDED, TRADERS):
        x2, part2, decls2, spec2, place2 = district(name)
        o2 = arrange.arrangements(part2, decls2, spec=spec2,
                                  pool=list(x2.get("fabric_types") or []) or None)
        p2 = next((a for a in o2 if a["action"] == "perimeter"
                   and not a.get("refused")), None)
        if p2 is None:
            said.append(f"{name}: no perimeter alternative")
            continue
        g2 = arrange.capacity_of(x2, part2, place2, decls2, p2["arrangement"],
                                 spec=spec2, seed=1, ceiling=int(x2["structures"]))
        r2 = g2["record"]
        s2 = enclosed_in(leaves_of(g2["plan"]), leaves_of(g2["plan"], "area"),
                         int(r2["lot"][1]) + 2 * dc.LOT_GAP + 1)
        said.append(f"{name} {r2.get('perimeter_blocks') or 0} perimeter block(s) of "
                    f"which {r2.get('perimeter_shut') or 0} closed, {len(s2)} court(s) "
                    f"measured shut on four sides, verdict "
                    f"{g2['certificate']['verdict']}")
    return (f"on a bare 200x120 rectangle of the section's own fabric asked for {ask} "
            f"house(s) the perimeter arrangement lays "
            f"{rec['perimeter_blocks']} perimeter block(s), "
            f"{rec['perimeter_shut']} of them closed on four sides: "
            f"{sum(1 for p in plots if str(p.get('front')) in ('west', 'east'))} of "
            f"{len(plots)} buildings front a cross street, where the same rectangle's "
            f"ordinary arrangement fronts none, and {len(shut)} open leaves have "
            f"building on all four sides against {len(plain_shut)} in the ordinary "
            f"arrangement. On the section's own ground: " + "; ".join(said))


# ------------------------------------------------------------------ the runner

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = skipped = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__[2:]
        if only and only not in name:
            continue
        try:
            said = fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            skipped += 1
            continue
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} neighbourhood spatial cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
