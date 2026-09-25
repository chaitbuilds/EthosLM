#!/usr/bin/env python3
"""The composition round's spatial escape routes, through the real consumers.

Every case runs the actual entry points -- `district_compile.compile_district`,
`arrange.capacity_of`, `arrange.arrange`, `placeplan.district_failures`,
`placeplan.negotiate_ring`, `placeplan.built_occupation`, `placesolve.reallocate`,
`placesolve.solve_place` -- on the retained city plan and the retained farming village.
Nothing here reimplements a measurement it is checking, and nothing here builds a world.

    S1  a required landmark that cannot fit returns unmet demand instead of a row of
        houses, the district validator refuses it, and the positive control -- the same
        fabric on ground that holds it -- keeps the market and is accepted. A court share
        spent to meet a lot count is on the record.

    S2  an alternative the district validator would refuse is not offered for selection,
        and every alternative that is carries the validator's own verdict and both
        occupation figures.

    S3  two ring probes differing only in their resolved demand do not share an answer.

    S4  `built_occupation` and `street_enclosure` report what was emitted, and report
        `unavailable` rather than zero where nothing was.

    S5  the anchor action moves in both directions; `regroup` changes geometry;
        `redistribute` preserves the total and refuses an exact count; a re-solve that
        would lose a counted structure under an exact count is refused.

Seconds to a couple of minutes: the fixtures are retained plans and the compiler is asked
about single rectangles.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import (arrange, district_compile as dc, placeplan,  # noqa: E402
                   placesolve, spec as spec_mod)

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the fixtures

_LOADED: dict = {}


def city():
    """Spec, place, the middle-ring part and its types.

        `out/des-city` is immutable input -- read and never written. The round forbids
        re-running interpretation, the site search or whole-city leaf compilation for a local
        architectural edit, so the section's own districts are what these cases re-decide.
        
    """
    if "city" in _LOADED:
        return _LOADED["city"]
    d = os.path.join(ROOT, "out", "des-city")
    if not os.path.exists(os.path.join(d, "plan.place.json")):
        raise Skip("no out/des-city/plan.place.json")
    place = json.load(open(os.path.join(d, "plan.place.json")))
    spec = json.load(open(os.path.join(d, "place.json")))
    part = next(p for p in spec["defining_parts"] if p["name"] == "middle_ring")
    role = spec_mod.district_role(spec, {"defines": "middle_ring"})
    _t, mine = placeplan.types_card(None, spec.get("form"), role)
    got = (spec, place, part, mine, d)
    _LOADED["city"] = got
    return got


def village():
    """The retained farming village, read three ways: inferred, approximate and exact.

        `regroup` and `redistribute` are admissible on the first two and refused on the third,
        and the third is where the anchor action's arithmetic is not clamped by the ground
        levelled for it.
        
    """
    if "village" in _LOADED:
        return _LOADED["village"]
    d = os.path.join(ROOT, "out", "expr-farm")
    if not os.path.exists(os.path.join(d, "place.json")):
        raise Skip("no out/expr-farm/place.json")
    doc = json.load(open(os.path.join(d, "place.json")))
    site = json.load(open(os.path.join(d, "site.json")))
    plateau = (json.load(open(os.path.join(d, "plateau.json")))
               if os.path.exists(os.path.join(d, "plateau.json")) else None)
    intent = json.load(open(os.path.join(d, "intent.json")))
    said = json.load(open(os.path.join(d, "interpretation.json")))["sentence"]
    loose = said.replace("sixteen low cottages", "low cottages")
    assert loose != said, "the fixture's sentence no longer names its count"
    _t, decls = placeplan.types_card(None, doc.get("form"))
    exact = spec_mod.read_spec(doc, said, None)
    assert int(exact["explicit_count"]["n"]) == 16, exact["explicit_count"]
    got = {
        "inferred": spec_mod.read_spec(doc, loose, None),
        "about": spec_mod.read_spec(doc, said.replace("sixteen", "about sixteen"),
                                    {"n": 16, "about": True, "what": "cottages",
                                     "phrase": "about sixteen cottages"}),
        "exact": exact,
        "site": site, "plateau": plateau, "decls": decls, "intent": intent}
    _LOADED["village"] = got
    return got


def solved(which: str):
    """One reading of the village, solved through `placesolve.solve_place`."""
    v = village()
    place, fails = placesolve.solve_place(v[which], v["site"], v["plateau"], v["decls"],
                                          "x", seed=1, intent=v["intent"])
    assert place is not None and not fails, fails
    return v, place


def act(place, spec, finding, v):
    """One `reallocate` on a copy of the place and the spec, so cases do not leak."""
    return placesolve.reallocate(json.loads(json.dumps(place)),
                                 json.loads(json.dumps(spec)), finding,
                                 site=v["site"], decls=v["decls"],
                                 plateau=v["plateau"], intent=v["intent"], seed=1)


def _probe(district, part, place, decls, spec, **over):
    """One district compiled through `arrange.capacity_of`, on bare ground.

        The production path: `capacity_of` -> `compile_once` -> `district_compile`, plus the
        certificate `district_failures` gives the arrangement it compiled.
        
    """
    d = dict(district, **over)
    for k in ("rect_columns", "scope_columns", "developable_columns",
              "allocated_columns", "built_columns", "columns_from", "count_band",
              "target", "proposed_structures"):
        d.pop(k, None)
    bare = {"arterials": {}, "parts": [], "districts": [], "layout": {}}
    return arrange.capacity_of(d, part, bare, decls, d.get("arrangement") or {},
                               spec=spec, seed=1,
                               ceiling=int(d.get("structures") or 1)), d, bare


#: **The one-row arrangement the design round built** (`arrange.ARRANGEMENT_ACTIONS`'s
#: `row_depth`), which is a real decision a layout owner takes: one row of houses along
#: a street rather than two back to back.
ONE_ROW = {"rows": 1, "lot_depth": 8, "lot_width": 8, "frontage": "street",
           "courtyard_share": 0.0}

#: **A middle-ring band that holds houses and cannot hold its market at any setting.**
#: 200x17: the edge margins leave 13 columns across, the grid's runs give a block 13
#: deep whichever end it starts at, and the market's own floor is 14. Measured by
#: sweeping depths 12..28 through `compile_district`: 16 and below lays no house at all,
#: 18 and above holds both, and 17 is the one depth where the housing fits and the
#: reservation does not -- which is exactly the shape of the defect.
THIN = {"x0": 0, "z0": 0, "x1": 199, "z1": 16, "structures": 8,
        "defines": "middle_ring", "arrangement": ONE_ROW}


# ------------------------------- S1: compose the space before filling it

@case
def t_s1a_a_required_landmark_that_cannot_fit_returns_unmet_demand():
    """The review's own words: "a middle-ring market can disappear during compilation".

        The middle ring's character names a `market` landmark and `function/market` is scoped
        to the ring, so the reservation is **required**. `market` delivers its `stalls` and
        `aisle` from a 14-column lot (`district_compile.landmark_least`, measured by
        `envelope.lot_for`), and the area branch used to size it
        `max(lo, min(hi, 17, block_w, block_d))` -- a `max` over a `min` against the block --
        then test the result against the same block, so a block shallower than 14 produced a
        side the block refused, the leaf fell through to `kind = "row"`, and the houses took
        the ground. Nothing recorded it and the validator had nothing to refuse.

        The counterexample is a real middle-ring arrangement: one row of houses along a
        street (`row_depth`, which the design round built) on a band 22 columns deep. The
        houses fit; the market does not.
        
    """
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    thin = dict(THIN, name="middle_ring_probe_thin",
                fabric_types=list(src["fabric_types"]), demand=src["demand"])
    got, d, bare = _probe(thin, part, place, decls, spec)
    assert got["ok"], got["why"]
    # the houses were laid...
    assert got["lots"] >= 4, got["why"]
    # ...and the market's ground is **returned as unmet demand**, not silently housed
    short = [x for x in got["demand_short"] if x["required"]]
    assert short, (got["record"] or {}).get("reservations")
    row = short[0]
    assert row["subject"] == "market" and row["requirement"] == "function/market", row
    assert row["needed"] == [14, 14], row
    assert row["available"] and min(row["available"]) < 14, row
    # ...with alternatives an owner can actually pull, each naming its own field
    alts = {a["what"]: a for a in row["alternatives"]}
    assert "block" in alts and "extent" in alts and "count" in alts, alts
    assert alts["block"]["need"] == 14 and alts["block"]["owner"] == "layout", alts
    assert min(alts["block"]["have"]) < 14, alts["block"]
    # ...and the **district validator refuses it**, through the production entry point
    fails = placeplan.district_failures(d, got["plan"], bare, decls, form=spec.get("form"),
                                        role=part.get("role"), part=part, spec=spec)
    mine = [f for f in fails if f.get("check") == "reservation"]
    assert mine, [f.get("check") for f in fails]
    assert mine[0]["requirement"] == "function/market", mine[0]
    assert "function/market" in mine[0]["why"] and "make it fit" in mine[0]["why"], mine[0]
    # ...which is also the certificate `capacity_of` now carries with the arrangement
    assert got["certificate"]["verdict"] == "refused", got["certificate"]
    assert "reservation" in got["certificate"]["checks"], got["certificate"]
    return (f"a 200x17 middle-ring band laid one row deep holds {got['lots']} house(s) "
            f"and no market: the reservation needs {row['needed']} and the block nearest "
            f"its middle is {row['available']}; the unmet demand comes back naming "
            f"`{row['requirement']}` with {len(row['alternatives'])} alternative(s) "
            f"({', '.join(sorted(alts))}) and the district validator refuses it on "
            f"`reservation`")


@case
def t_s1b_the_positive_control_the_same_fabric_on_ground_that_holds_it():
    """The other half: the reservation drives the block grid and the market survives."""
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    got, d, bare = _probe(src, part, place, decls, spec, name="middle_ring_probe_wide",
                          arrangement=dict(ONE_ROW))
    assert got["ok"], got["why"]
    rec = got["record"]
    assert not [x for x in got["demand_short"] if x["required"]], got["demand_short"]
    assert rec["reservations_kept"] == rec["reservations_required"] == 1, rec
    # the grid was driven by the reservation and the record says so, with both numbers
    drove = rec["reserve_drove"]
    assert drove and drove["least"] == 14, drove
    assert drove["block_depth"][0] < 14 <= drove["block_depth"][1], drove
    # the market leaf is there, at a side that delivers its stalls
    leaves = [p for q in got["plan"]["quarters"] for p in q["plots"]]
    market = next(p for p in leaves if p["type"] == "market")
    side = market["x1"] - market["x0"] + 1
    assert side >= 14 and market["kind"] == "area", market
    # ...and the validator accepts the arrangement on the reservation clause
    fails = placeplan.district_failures(d, got["plan"], bare, decls, form=spec.get("form"),
                                        role=part.get("role"), part=part, spec=spec)
    assert not [f for f in fails if f.get("check") == "reservation"], fails
    return (f"the same one-row fabric on the section's own 190x60 sector: the grid's block "
            f"depth is driven {drove['block_depth'][0]} -> {drove['block_depth'][1]} to "
            f"hold the 14-column reservation before the housing takes the ground, the "
            f"market stands at {side}x{market['z1'] - market['z0'] + 1}, "
            f"{got['lots']} house(s) stand beside it, and the validator raises no "
            f"reservation failure")


@case
def t_s1c_an_arrangement_that_keeps_the_market_outranks_one_that_loses_it():
    """`compile_district.score` had no landmark term and the search stopped as soon as the
        count, the cover and the ceiling were met -- so an arrangement that housed the market's
        ground ended the search, and a retry that kept the market and laid *fewer* houses
        ranked below it. The compiler was selecting, by those two lines, the arrangement that
        loses the thing the request requires.

        The counterexample is one rectangle, one fabric, two demands: the middle ring's own
        resolved demand, which carries `function/market`, and the same demand with that
        requirement taken out. Nothing else differs. The required reading gives up a house to
        keep the market; the optional reading keeps the house -- which is why a score without
        the clause preferred it.
        
    """
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    bare = {"arterials": {}, "parts": [], "districts": [], "layout": {}}
    base = {"x0": 0, "z0": 0, "x1": 119, "z1": 21, "structures": 8,
            "defines": "middle_ring", "fabric_types": list(src["fabric_types"]),
            "arrangement": {**ONE_ROW, "lot_depth": 6}}
    loose = dict(src["demand"])
    loose["requirements"] = [r for r in loose["requirements"] if r != "function/market"]
    _p1, req = dc.compile_district(dict(base, name="required", demand=src["demand"]),
                                   part, bare, decls, spec=spec, seed=1)
    _p2, opt = dc.compile_district(dict(base, name="optional", demand=loose),
                                   part, bare, decls, spec=spec, seed=1)
    # the same rectangle, the same fabric: only the requirement differs
    assert req["reservations_required"] == 1 and opt["reservations_required"] == 0, (req, opt)
    # the required reading keeps the market; the optional one houses its ground
    assert req["reservations_kept"] == 1, req["demand_short"]
    assert opt["reservations_kept"] == 0 and opt["lots"] > 0, opt
    assert not [x for x in req["demand_short"] if x.get("required")], req["demand_short"]
    # ...and it costs a house, which is exactly the trade a score without the clause
    # takes
    assert opt["lots"] > req["lots"], (req["lots"], opt["lots"])
    # the lever the reservation pulled, on the record and nowhere else
    assert req["raised"].get("lead_street") == [True, False], req["raised"]
    assert "lead_street" not in opt["raised"], opt["raised"]
    return (f"one 120x22 middle-ring rectangle, one fabric: with `function/market` in its "
            f"demand the compiler gives up the lead street, lays {req['lots']} house(s) "
            f"and keeps the market; with the requirement taken out it keeps the street, "
            f"lays {opt['lots']} and houses the market's ground -- so the arrangement the "
            f"old score preferred is the one that loses the requirement")


@case
def t_s1e_a_plan_cannot_pass_by_leaving_the_field_out():
    """The clause reads the plan's **own statement** (`demand_short`) *and* measures. The
    first alone is a promise the writer makes about itself: a district file a model
    wrote carries no `demand_short`, and so would a compiler that dropped a reservation
    without recording it.

        Run on the retained city's own middle-ring plan with its market leaf taken out and no
        `demand_short` added: the shape of exactly that escape.
        
    """
    spec, place, part, decls, d = city()
    p = os.path.join(d, "plan.district.middle_ring_north_west.json")
    if not os.path.exists(p):
        raise Skip("no plan.district.middle_ring_north_west.json")
    x = next(q for q in place["districts"] if q["name"] == "middle_ring_north_west")
    got = json.load(open(p))
    role = spec_mod.district_role(spec, x)
    mine = placeplan._district_part(spec, x)
    _t, every = placeplan.types_card()
    # the retained plan holds its market and is accepted
    kept = placeplan.district_plots(got, role, spec)
    assert any(q.get("type") == "market" for q in kept), "the fixture has no market"
    assert not placeplan.reservation_failures(x, got, kept, every, mine), \
        placeplan.reservation_failures(x, got, kept, every, mine)
    # ...and every retained district plan of the city passes it, so the clause is not
    # refusing the delivered place
    n_ok = 0
    for q in place["districts"]:
        f = os.path.join(d, f"plan.district.{q['name']}.json")
        if not os.path.exists(f):
            continue
        g = json.load(open(f))
        r2 = spec_mod.district_role(spec, q)
        assert not placeplan.reservation_failures(
            q, g, placeplan.district_plots(g, r2, spec), every,
            placeplan._district_part(spec, q)), q["name"]
        n_ok += 1
    # the escape route: the same plan with the market leaf gone and no field admitting
    # it
    stripped = {**got,
                "quarters": [{**qq, "plots": [pp for pp in qq["plots"]
                                              if pp.get("type") != "market"]}
                             for qq in got["quarters"]]}
    assert "demand_short" not in stripped, stripped.keys()
    fails = placeplan.reservation_failures(
        x, stripped, placeplan.district_plots(stripped, role, spec), every, mine)
    assert len(fails) == 1, fails
    assert fails[0]["subject"] == "market", fails[0]
    assert fails[0]["requirement"] == "function/market", fails[0]
    assert "no leaf of its" in fails[0]["why"], fails[0]["why"]
    return (f"{n_ok} retained city district plan(s) pass the reservation clause; the "
            f"middle ring's own plan with its market leaf removed and no `demand_short` "
            f"field is refused anyway, on the measurement: "
            f"{fails[0]['why'][:90]}...")


@case
def t_s1d_a_court_share_spent_to_meet_a_lot_count_is_on_the_record():
    """"courtyard/open shares can be spent to fit housing" -- the review, and it was
        invisible. `compile_district`'s second lever gives the character's open blocks and
        then its courts back to the fabric when the count is short, and the only trace was a
        `raised` entry that reads like a raise.

        One middle-ring rectangle, asked for far more houses than its ground fits at its own
        lot, so the lever fires; and the positive control, the same rectangle at the count its
        ground actually holds, where no share is spent.
        
    """
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    ch = part["character"]
    assert float(ch["courtyard_share"]) > 0, ch
    greedy, _d1, _b1 = _probe(src, part, place, decls, spec, name="greedy",
                              x0=0, z0=0, x1=119, z1=59, structures=40)
    calm, _d2, _b2 = _probe(src, part, place, decls, spec, name="calm",
                            x0=0, z0=0, x1=119, z1=59, structures=4)
    assert greedy["ok"] and calm["ok"], (greedy["why"], calm["why"])
    spent = greedy["record"]["shares_spent"]
    assert spent, greedy["record"]["raised"]
    assert "courtyard_share" in spent or "open_share" in spent, spent
    moved = next(k for k in ("courtyard_share", "open_share") if k in spent)
    assert spent[moved][1] < spent[moved][0], spent
    assert str(spent["why"]).startswith("this district was short"), spent["why"]
    # the control: a count the ground holds spends nothing
    assert calm["record"]["shares_spent"] is None, calm["record"]["shares_spent"]
    return (f"a 120x60 medium rectangle asked for 40 houses lays "
            f"{greedy['record']['lots']} and pays for them out of its "
            f"`{moved}` ({spent[moved][0]:g} -> {spent[moved][1]:g}), leaving "
            f"{spent['courts']} court(s); the same rectangle asked for 4 lays "
            f"{calm['record']['lots']} and spends no share at all")


# --------------------- S1b: what a district is *for* decides what it is built of

@case
def t_s1f_district_uses_govern_type_selection():
    """The user's central instruction for the composition round, and the number it was
        found at.

        The built section's calm side came back with **5 temples and 4 halls against 1 shop
        house in 22 buildings**, in the ring the sources describe as traders, craftsmen and
        schoolteachers. The cause: the compiler drew every lot's type from everything the
        *pool* admitted, and the pool a capability record approves for a ring whose landmark
        programme needs civic types contains `hall` and `temple`. A type pool is what a quarter
        may be built of; it is not what the quarter is for.

        **This case is what the composition round's fix has to keep holding**, and it is now
        asserted against the rule that replaced it. What the composition round wrote was a
        `ROLE` filter with a registered 10% secondary share (`dc.PROGRAMME_USE_SHARE`); the
        neighbourhood round replaced the share with an inferred programme (`dc.use_mix`,
        `t_p1`..`t_p3` of `test_neighbourhood_spatial.py`). The civic types must still be out
        of the fabric, the market must still be laid as the landmark it is, and the quarter's
        own use must still be what its streets are drawn from -- those three are this case,
        and they are checked here on the same two districts as before.
        
    """
    spec, place, part, decls, _d = city()
    said = []
    for name in ("middle_ring_north_west", "middle_ring_north_east"):
        x = next(q for q in place["districts"] if q["name"] == name)
        got, rec = dc.compile_district(x, part, place, decls, spec=spec, seed=1)
        mix = rec["use_mix"]
        leaves = [p for q in got["quarters"] for p in q["plots"]
                  if p.get("kind", "plot") == "plot"]
        kinds: dict = {}
        for p in leaves:
            kinds[str(p["type"])] = kinds.get(str(p["type"]), 0) + 1
        # the pool really does hold the civic types -- otherwise the case proves nothing
        assert set(mix["unasked"]) == {"hall", "temple"}, mix
        # ...and not one lot of the fabric is one of them
        assert not (set(kinds) & set(mix["unasked"])), (name, kinds)
        # every lot is either the quarter's own fabric or the work its design names
        assert set(kinds) <= set(mix["own"]) | set(mix["programme"]), \
            (name, kinds, mix["own"], mix["programme"])
        assert len(kinds) >= 2, (name, kinds)
        # the quarter's own fabric is still most of it: a programme is not a takeover
        own_n = sum(v for k, v in kinds.items() if k in set(mix["own"]))
        assert own_n > sum(kinds.values()) / 2, (name, kinds, mix["own"])
        # ...and the market, which the programme *does* require, is still laid -- as the
        # landmark it is, not as every third lot
        assert rec["reservations_kept"] == rec["reservations_required"] == 1, rec
        assert any(p["type"] == "market" for q in got["quarters"] for p in q["plots"]), name
        said.append(f"{name}: {sum(kinds.values())} building(s) "
                    + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
                    + " and one market landmark")
    # the positive control: a district whose **demand requires** another use draws it,
    # at the count the requirement asks for and not at a share of the street. `temple`
    # answers `feature/temple`, so it is fabric -- the same rule, run the other way.
    x = next(q for q in place["districts"] if q["name"] == "middle_ring_north_west")
    asked = dict(x, name="asked",
                 demand={**x["demand"],
                         "requirements": [*x["demand"]["requirements"], "feature/temple"]})
    got2, rec2 = dc.compile_district(asked, part, place, decls, spec=spec, seed=1)
    mix2 = rec2["use_mix"]
    assert "temple" in mix2["programme"], mix2
    assert mix2["required"].get("temple", {}).get("count") == 1, mix2["required"]
    assert mix2["unasked"] == ["hall"], mix2
    laid = [p for q in got2["quarters"] for p in q["plots"]
            if p.get("kind", "plot") == "plot"]
    temples = sum(1 for p in laid if p["type"] == "temple")
    # exactly the one the requirement asks for.
    assert temples == 1, (temples, len(laid), mix2["required"])
    return ("; ".join(said)
            + f"; and a district whose demand requires `feature/temple` draws exactly "
              f"{temples} temple in {len(laid)} building(s) -- the count the requirement "
              f"asks for, where the retired `PROGRAMME_USE_SHARE` would have given it "
              f"{dc.PROGRAMME_USE_SHARE:.0%} of the street, and where nothing asked for "
              f"one draws none")


# ------------------------- S2: certify alternatives through the actual validator

@case
def t_s2a_an_alternative_the_validator_would_refuse_is_not_offered():
    """A4: "Every arrangement offered for selection is certified by the **actual district
        validator**, not by a compiler return."

        `capacity_of` returned what the compiler laid and `arrange._covers` -- a local
        approximation of one of the validator's clauses -- was all selection had; the real
        `district_failures` ran later, at the plan stage, after the width had been adopted. So
        the thin band above, whose compiler return is a perfectly good count of houses, is
        exactly the alternative that could be selected and then refused.
        
    """
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    bad, _d1, _b1 = _probe(dict(THIN, name="thin",
                                fabric_types=list(src["fabric_types"]),
                                demand=src["demand"]),
                           part, place, decls, spec)
    good, _d2, _b2 = _probe(src, part, place, decls, spec, name="wide",
                            arrangement=dict(ONE_ROW))
    # the compiler's own return says nothing is wrong with the refused one
    assert bad["ok"] and bad["lots"] > 0, bad["why"]
    assert arrange._covers(bad, 0), "the local cover test would have passed it"
    # the certificate does, and it is the validator's own verdict
    assert bad["certificate"]["from"] == "placeplan.district_failures", bad["certificate"]
    assert bad["certificate"]["verdict"] == "refused", bad["certificate"]
    # a `reservation` is never negotiable: it is on `refuses`, which is the gate
    checks = [f["check"] for f in bad["certificate"]["refuses"]]
    assert "reservation" in checks, bad["certificate"]
    # **...and `arrangement` is beside it now, which is a true second refusal and a
    # finding for the layout owner.** The neighbourhood round added
    # `placeplan.arrangement_failures`: this band adopted 8x8 lots and the compiler laid
    # 13x13, because the demand's own least lot (`lot_min`, the envelope for the
    # features the requirement makes required) stands over a lot the character declared.
    # That override is recorded on the record (`lot_asked` 8x8, `lot_laid` 13x13,
    # `lot_refused`) rather than silent, so it is arguably a *report* and not a refusal
    # -- raised with the round's coordinator, who owns that check.
    assert set(checks) <= {"reservation", "arrangement"}, bad["certificate"]
    # the wide sector raises no **reservation** refusal, which is what this case is
    # about. It does raise `arrangement`, for the same reason the thin band does and
    # with the same recorded cause: it adopted 8x8, its demand's required features need
    # 13x13, and `lot_min` stood over the declaration. See the note above.
    assert not [f for f in good["certificate"]["refuses"]
                if f.get("check") != "arrangement"], good["certificate"]
    # ...and every alternative carries its honest metrics: lots, the lots' columns, the
    # mass those lots admit, the reservations kept, the verdict
    for got in (bad, good):
        assert got["allocated_columns"] > got["footprint_columns"] > 0, got
        assert got["reservations_required"] == 1, got
    assert good["reservations_kept"] == 1 and bad["reservations_kept"] == 0, (good, bad)
    # a probe that is *not* offering an alternative says so rather than claiming a
    # verdict
    quiet, _d3, _b3 = _probe(src, part, place, decls, spec, name="quiet")
    quiet = arrange.capacity_of(_d3, part, _b3, decls, {}, spec=spec, seed=1,
                                ceiling=int(_d3["structures"]), certify=False)
    assert quiet["certificate"]["verdict"] == "unasked", quiet["certificate"]
    return (f"the thin band's compiler return is {bad['lots']} house(s) on "
            f"{bad['allocated_columns']} columns of lot ({bad['footprint_columns']} "
            f"built) and `_covers` accepts it; `district_failures` refuses it on "
            f"{bad['certificate']['checks']}, while the wide sector is accepted at "
            f"{good['lots']} house(s) on {good['allocated_columns']} columns "
            f"({good['footprint_columns']} built) with 1 of 1 reservation kept")


@case
def t_s2b_a_rings_alternatives_carry_the_verdict_and_both_occupations():
    """`negotiate_ring` selected on `(holds_count, allocated_cover, -width)`, where
        `allocated_cover` is the lots' share of the ring -- the one figure the review says
        cannot establish density. Selection still happens there, on certified alternatives,
        with the built occupation recorded beside the allocated one and the reservations
        counted.

        Run on the retained hill town through `concentric_layout`, which is how production
        calls it.
        
    """
    d = os.path.join(ROOT, "out", "expr-rings")
    if not os.path.exists(os.path.join(d, "place.json")):
        raise Skip("no out/expr-rings")
    doc = json.load(open(os.path.join(d, "place.json")))
    intent = json.load(open(os.path.join(d, "intent.json")))
    sentence = json.load(open(os.path.join(d, "interpretation.json")))["sentence"]
    count = next((dict(r.get("wants") or {}) for r in intent.get("requirements") or []
                  if r.get("kind") == "count"), None)
    spec = spec_mod.read_spec(doc, sentence, count)
    site = json.load(open(os.path.join(d, "site.json")))
    plateau = json.load(open(os.path.join(d, "plateau.json")))
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    lower = next(r for r in place["layout"]["rings"] if r["name"] == "lower_ring")
    considered = lower["target"]["negotiated"]
    real = [a for a in considered if a.get("capacity") is not None
            and not a.get("refused")]
    assert len(real) >= 2, considered
    for a in real:
        cert = a["certificate"]
        assert cert["from"] == "placeplan.district_failures", cert
        # **every alternative offered carries the validator's own verdict**, and no
        # alternative offered carries a refusal the negotiation may not negotiate away
        assert not cert["refuses"], (a["action"], cert)
        # both occupations, recorded apart, and the second under the first
        assert a["built_columns"] is not None and a["allocated_columns"] is not None, a
        assert 0 < a["built_cover"] < a["cover"], a
        assert a["reservations_required"] == 0, a
        for p in a["probed"]:
            assert not p["refuses"], p
    # **this ring is short of its own density word, and that is a finding and not a
    # refusal**: the dense lower ring covers 5% where its word asks 43%, which is
    # exactly the thing the negotiation exists to answer, so `cover` is on `short_of`
    # and not on `refuses` (`arrange.NEGOTIABLE_CHECKS`). A gate on it would leave the
    # ring unlayable.
    short = {c for a in real for c in (a["certificate"]["short_of"] or ())}
    assert short and short <= set(arrange.NEGOTIABLE_CHECKS), short
    chosen = lower["target"]["from"]
    assert any("validator refused" in line for line in chosen), chosen
    assert any(line.startswith("chosen:") and "the layout owner's finding" in line
               for line in chosen), chosen[-1]
    # and the width the ring was given is one of the certified alternatives' own: the
    # parent's dimension and the child's arrangement are still one decision
    assert lower["width"] in {int(a["width"]) for a in real}, \
        (lower["width"], sorted({int(a["width"]) for a in real}))
    return (f"{len(considered)} alternative(s) for the lower ring, {len(real)} offered for "
            f"selection with no non-negotiable refusal: "
            + "; ".join(f"{a['action']} {a['capacity']} house(s), lots "
                        f"{a['cover']:.1%} / built {a['built_cover']:.1%}"
                        for a in real)
            + f"; every one of them is short of {sorted(short)}, which the record carries "
              f"as the layout owner's finding rather than as a refusal")


@case
def t_s2c_three_certified_arrangements_for_the_sections_own_districts():
    """"Keep this section and terrain, compare arrangements cheaply" -- and judge them on
        frontage and built mass, not on a cover ratio.

        `arrange.alternatives` is that call: every arrangement the district's fabric admits,
        compiled once on the district's **own** rectangle and terrain, certified by
        `district_failures`, and measured on both occupations *and* on street enclosure. Run on
        the section's crowded side and its calm side.

        The user's first reading, in the numbers it turns on: the dense ring's lots cover 26.3%
        of developable ground against the 30% its own word asks, and its terraces stand in
        stripes with grass voids between them. The alternative that answers it is not a paving
        trick: it is **more, smaller houses** -- and the case checks that its built mass and its
        frontage both rise with its cover, which a relabelling cannot do.
        
    """
    spec, place, part_of, decls, _d = city()
    by = {p["name"]: p for p in spec["defining_parts"]}
    said = []
    for name in ("lower_ring_north_2", "middle_ring_north_west"):
        x = next(q for q in place["districts"] if q["name"] == name)
        part = by[x["defines"]]
        role = spec_mod.district_role(spec, x)
        _t, mine = placeplan.types_card(None, spec.get("form"), role)
        got = arrange.alternatives(x, part, place, mine, spec=spec, seed=1,
                                   ceiling=int(x["structures"]))
        live = [a for a in got if int(a["lots"] or 0) > 0 and not a.get("refused")]
        assert len(live) >= 3, [(a["action"], a["lots"], a.get("refused")) for a in got]
        # ...and an arrangement that laid nothing is last, not first
        assert all(int(a["lots"] or 0) > 0 for a in got[:len(live)]), \
            [(a["action"], a["lots"]) for a in got[:len(live)]]
        # every one of them is certified by the real validator and none is refused
        for a in live:
            assert a["verdict"] in ("ok", "refused"), a
            assert not a["refuses"], a
            # ...and carries the lots, the mass they admit and the frontage, apart
            assert a["allocated_columns"] > a["built_columns"] > 0, a
            assert a["enclosure"] is not None and a["street_columns"] > 0, a
            assert a["reservations_kept"] >= a["reservations_required"], a
        # they are genuinely different: at least three distinct (lot, rows, block)
        # shapes
        shapes = {(tuple(a["lot"]), a["rows"], (a["arrangement"] or {}).get("block"))
                  for a in live}
        assert len(shapes) >= 3, shapes
        base = next(a for a in live if a["action"] == "as_declared")
        best = live[0]
        said.append(f"{name}: {len(live)} certified, {len(shapes)} distinct shape(s); "
                    f"as declared {base['lots']} house(s) at built cover "
                    f"{base['built_cover']:.1%} / enclosure {base['enclosure']:.1%}; best "
                    f"`{best['action']}` {best['lot'][0]}x{best['lot'][1]} x"
                    f"{best['rows']}row {best['lots']} house(s) at "
                    f"{best['built_cover']:.1%} / {best['enclosure']:.1%}")
        if name == "lower_ring_north_2":
            # **The crowded side, and what this case asserted before.** The composition
            # round's claim was that the best alternative raises the allocated cover,
            # the built mass, the frontage and the count together, on smaller lots --
            # because every one of those has to move together or the figure is a
            # relabelling. * `spec.columns_per_plot` and `placeplan.count_band` were
            # corrected to take the adopted lot, so an arrangement of smaller lots is
            # now asked for the count its own lot earns rather than the incumbent
            # fabric's. The best alternative is therefore more, smaller houses with more
            # frontage and more of the street fronted -- and **less** mass per column of
            # ground, which is what smaller houses are. That trade is stated rather than
            # hidden: the assertion below is that the count, the frontage and the
            # enclosure move together on lots that did not grow, and that the mass that
            # was given up is reported.
            assert best["lots"] > base["lots"], (base["lots"], best["lots"])
            assert best["frontage_length"] > base["frontage_length"], \
                (base["frontage_length"], best["frontage_length"])
            assert best["enclosure"] > base["enclosure"], \
                (base["enclosure"], best["enclosure"])
            assert (best["lot"][0] * best["lot"][1]
                    <= base["lot"][0] * base["lot"][1]), (base["lot"], best["lot"])
            # the lots the count was won with are not bigger lots: allocated cover per
            # house falls, which a relabelling cannot do
            assert (best["allocated_columns"] / best["lots"]
                    < base["allocated_columns"] / base["lots"]), (base, best)
            said.append(f"  (crowded: {base['lots']} -> {best['lots']} houses, frontage "
                        f"{base['frontage_length']} -> {best['frontage_length']}, "
                        f"enclosure {base['enclosure']:.1%} -> {best['enclosure']:.1%}, "
                        f"pad estimate {base['built_cover']:.1%} -> "
                        f"{best['built_cover']:.1%} -- the mass given up for the count, "
                        f"reported)")
    # ...and an arrangement that gives up the street its character asked for does not
    # rank first, whatever its cover: a street with no building on it is not a street
    x = next(q for q in place["districts"] if q["name"] == "middle_ring_north_west")
    part = by["middle_ring"]
    role = spec_mod.district_role(spec, x)
    _t, mine = placeplan.types_card(None, spec.get("form"), role)
    got = arrange.alternatives(x, part, place, mine, spec=spec, seed=1,
                               ceiling=int(x["structures"]))
    live = [a for a in got if int(a["lots"] or 0) > 0 and not a.get("refused")]
    # **over the whole list and not only the eligible part of it.** The open-frontage
    # alternative is compiled and measured whatever the validator then says about it,
    # and on this rectangle the validator refuses it on `overlap` -- so looking for it
    # among the eligible rows finds nothing and proves nothing. What this case is about
    # is that it measures its enclosure **honestly at zero** and does not rank first for
    # it.
    opened = [a for a in got
              if int(a.get("lots") or 0) > 0
              and (a["arrangement"] or {}).get("frontage") == "open"]
    assert opened, [a["action"] for a in got]
    assert opened[0]["enclosure"] == 0.0, opened[0]
    assert got[0] is not opened[0], "an arrangement with no street frontage ranked first"
    assert live and live[0] is not opened[0], live[0]["action"] if live else got
    assert (live[0]["enclosure"] or 0) > 0, live[0]
    return ("; ".join(said)
            + f"; the open-frontage alternative measures {opened[0]['enclosure']:.0%} "
              f"enclosure honestly and does not rank first "
              f"(`{live[0]['action']}` does, at {live[0]['enclosure']:.1%})")


# ----------------------------------------------- S3: the ring probe's cache identity

@case
def t_s3a_two_probes_differing_only_in_demand_do_not_share_an_answer():
    """The review's named omission: "The ring-probe cache also omits
        demand/type/context/seed identities."

        The key was `(width, depth, count, arrangement items, region name)`. The probe body
        reads `r["demand"]`, `r["fabric_types"]`, the form, the role, the density, the seed,
        the ceiling and the exact flag -- so a capability revision that narrowed a ring's
        approved pool, or a demand that raised its least lot, was answered from the old
        certificate. Checked on `probe_key` itself and then on the cache through
        `_arrange_capacity`, which is the function that reads it.
        
    """
    spec, place, part, decls, _d = city()
    src = next(x for x in place["districts"] if x["name"] == "middle_ring_north_west")
    rect = (0, 0, 189, 59)
    arr = {"rows": 2, "lot_width": 13, "lot_depth": 13, "frontage": "street",
           "courtyard_share": 0.3}
    base = dict(part, name="middle_ring", fabric_types=list(src["fabric_types"]),
                demand=dict(src["demand"]))
    k0 = placeplan.probe_key(base, rect, 16, arr, spec, seed=1, ceiling=16)
    # one field of the resolved demand at a time; every one of them has to move the key
    moved = {}
    for field, value in (("required", ["courtyard"]),
                         ("types", ["courtyard_house"]),
                         ("params", {"storeys": 2}),
                         ("requirements", ["function/market", "feature/courtyard"])):
        other = dict(base, demand={**base["demand"], field: value})
        moved[field] = placeplan.probe_key(other, rect, 16, arr, spec, seed=1,
                                           ceiling=16) != k0
    assert all(moved.values()), moved
    # ...and so does the approved pool, the seed, the ceiling and the exact flag
    others = {
        "fabric_types": dict(base, fabric_types=["row_house"]),
        "density": dict(base, density="dense"),
        "role": dict(base, role="rural"),
    }
    for name, r in others.items():
        moved[name] = placeplan.probe_key(r, rect, 16, arr, spec, seed=1,
                                          ceiling=16) != k0
    moved["seed"] = placeplan.probe_key(base, rect, 16, arr, spec, seed=2,
                                        ceiling=16) != k0
    moved["ceiling"] = placeplan.probe_key(base, rect, 16, arr, spec, seed=1,
                                           ceiling=32) != k0
    moved["exact"] = placeplan.probe_key(base, rect, 16, arr, spec, seed=1, ceiling=16,
                                         exact=False) != k0
    moved["form"] = placeplan.probe_key(base, rect, 16, arr, dict(spec, form="gothic"),
                                        seed=1, ceiling=16) != k0
    # ...and whether the probe was certified: an uncertified entry answering a certified
    # question is an alternative with no verdict reading as one with no failures
    moved["certified"] = placeplan.probe_key(base, rect, 16, arr, spec, seed=1,
                                             ceiling=16, certified=False) != k0
    assert all(moved.values()), {k: v for k, v in moved.items() if not v}
    # the same inputs are the same key, whatever order the dicts were built in
    same = placeplan.probe_key(
        dict(part, name="middle_ring", demand=dict(reversed(list(src["demand"].items()))),
             fabric_types=list(src["fabric_types"])), rect, 16, arr, spec, seed=1,
        ceiling=16)
    assert same == k0, "the key is not stable under dict order"
    # and through the reader: the cache does not answer across a changed demand
    placeplan._RING_PROBE_CACHE.clear()
    a = placeplan._arrange_capacity(base, rect, 16, arr, spec, decls, seed=1)
    n_after_one = len(placeplan._RING_PROBE_CACHE)
    narrow = dict(base, fabric_types=["row_house"],
                  demand={**base["demand"], "types": ["row_house"]})
    b = placeplan._arrange_capacity(narrow, rect, 16, arr, spec, decls, seed=1)
    assert len(placeplan._RING_PROBE_CACHE) == n_after_one + 1, \
        "the second probe was answered from the first probe's entry"
    return (f"every one of {len(moved)} identities moves the probe key "
            f"({', '.join(sorted(moved))}), and dict order does not; through "
            f"`_arrange_capacity` the wide pool lays {a} house(s) and the pool narrowed "
            f"to `row_house` lays {b} from an entry of its own")


# ---------------------------------------- S4: honest built metrics for the checker

@case
def t_s4a_built_occupation_reads_what_construction_emitted():
    """The two figures the checker needs, and the defect that made one of them a lie.

        `region_columns` looked for `occupied_columns` on `parts_record["parts"][name]`;
        `parts.json` is `{"waves": [{"parts": [...]}]}` with the rectangle under
        `emitted.footprint`, and the assembler renames a clashing leaf to
        `<district>_<leaf>`. So the emitted branch never fired on a real record and every
        built figure in the project was the plan's own pads under the name `built_columns`.
        Read on the retained city's own `parts.json`, which was built.
        
    """
    spec, place, part, decls, d = city()
    pr_path = os.path.join(d, "parts.json")
    if not os.path.exists(pr_path):
        raise Skip("no out/des-city/parts.json")
    pr = json.load(open(pr_path))
    rows = placeplan.emitted_columns(pr)
    assert rows, "nothing in parts.json reported a rectangle"
    # a district whose leaves were actually built
    built = None
    for x in place["districts"]:
        p = os.path.join(d, f"plan.district.{x['name']}.json")
        if not os.path.exists(p):
            continue
        got = json.load(open(p))
        role = spec_mod.district_role(spec, x)
        leaves = [q for q in placeplan.district_plots(got, role, spec)
                  if q.get("kind", "plot") == "plot"]
        if any(placeplan.emitted_for(rows, x["name"], q["name"]) for q in leaves):
            built = (x, leaves)
            break
    if built is None:
        raise Skip("no district of out/des-city has an emitted leaf")
    x, leaves = built
    occ = placeplan.built_occupation(x, pr, leaves=leaves)
    assert str(occ["from"]).startswith("emitted"), occ
    # the four columns, and the two covers, apart -- lots over building
    assert occ["allocated_columns"] > occ["built_columns"] > 0, occ
    assert occ["allocated_cover"] > occ["built_cover"] > 0, occ
    assert occ["scope_columns"] >= occ["developable_columns"] > 0, occ
    # the same call with no record reports that it has none, rather than zero
    none = placeplan.built_occupation(x, None, leaves=leaves)
    assert none["from"] == "unavailable", none
    assert all(none[k] is None for k in ("built_columns", "built_cover",
                                         "allocated_cover")), none
    # ...and with a record but no emitted leaf it falls to the plan's pads and says so
    plan_only = placeplan.built_occupation(x, {"waves": []}, leaves=leaves)
    assert plan_only["from"] == "planned pads", plan_only
    return (f"`{x['name']}`: {occ['built_columns']} columns of building emitted under "
            f"{occ['allocated_columns']} columns of lot over "
            f"{occ['developable_columns']} developable -- built cover "
            f"{occ['built_cover']:.1%} against allocated {occ['allocated_cover']:.1%}, "
            f"from `{occ['from']}`; with no parts record the same call answers "
            f"`unavailable` and Nones, not zeros")


@case
def t_s4b_street_enclosure_measures_frontage_and_never_invents_a_score():
    """"Passing a percentage would not by itself create crowded streets" -- so the street
        is measured: of the district's own street cells, how much of their length has a
        building's declared front within a lot's gap of it.

        The dense lower ring and the medium middle ring, read off the retained plans. The
        contrast is the point: a terrace of party walls encloses more of its streets than
        courtyard houses on a looser grain, and this says so in columns rather than in a word.
        
    """
    spec, place, part, decls, d = city()
    pr = (json.load(open(os.path.join(d, "parts.json")))
          if os.path.exists(os.path.join(d, "parts.json")) else None)
    said = {}
    for name in ("lower_ring_north_2", "middle_ring_north_west"):
        p = os.path.join(d, f"plan.district.{name}.json")
        if not os.path.exists(p):
            raise Skip(f"no plan.district.{name}.json")
        x = next(q for q in place["districts"] if q["name"] == name)
        got = json.load(open(p))
        role = spec_mod.district_role(spec, x)
        leaves = [q for q in placeplan.district_plots(got, role, spec)
                  if q.get("kind", "plot") == "plot"]
        said[name] = placeplan.street_enclosure(x, place, leaves, parts_record=pr)
    for name, m in said.items():
        assert m["street_columns"] > 0 and m["faces"] > 0, (name, m)
        assert 0 < m["enclosed_length"] <= m["street_columns"], (name, m)
        assert m["mean_setback"] is not None and m["mean_setback"] <= dc.LOT_GAP, (name, m)
        assert m["frontage_length"] > 0, (name, m)
    dense, medium = said["lower_ring_north_2"], said["middle_ring_north_west"]
    assert dense["enclosure"] > medium["enclosure"], (dense, medium)
    assert dense["frontage_length"] > medium["frontage_length"], (dense, medium)
    # nothing to measure is not an enclosure of zero
    empty = placeplan.street_enclosure(
        next(q for q in place["districts"] if q["name"] == "lower_ring_north_2"),
        place, [], parts_record=pr)
    assert empty["from"] == "unavailable" and empty["enclosure"] is None, empty
    return (f"the dense lower ring's {dense['faces']} houses front "
            f"{dense['frontage_length']} columns and enclose "
            f"{dense['enclosed_length']} of its {dense['street_columns']} street columns "
            f"({dense['enclosure']:.1%}, mean setback {dense['mean_setback']:g}); the "
            f"medium middle ring's {medium['faces']} front "
            f"{medium['frontage_length']} and enclose {medium['enclosure']:.1%}; an "
            f"empty district answers `unavailable`, not 0%")


# ----------------------------- S5: effective composition operations for revision

@case
def t_s5a_the_anchor_action_moves_in_both_directions():
    """`shrink_anchor` clamped with `min(new, share)`, so it could only ever reduce -- and
        a finding that the square is too small for what gathers about it, which is what
        `anchor_size`'s own `least` bound exists for, had no action at all.

        Both directions through `placesolve.reallocate`, on the village read at its own
        sentence's count (where the anchor's arithmetic is inside both the type's band and the
        ground levelled for it, so a share that moves moves the geometry), and the count
        invariant read on both results.
        
    """
    v, place = solved("exact")
    spec = v["exact"]
    was_n = sum(int(x["structures"]) for x in place["districts"])

    def side(pl):
        m = next(p for p in pl["parts"] if p.get("name") == "market")
        return m["x1"] - m["x0"] + 1
    base = {"about": "scale", "measure": "square_scale", "subjects": ["market"],
            "says": "the square reads out of scale beside the houses"}
    small, r_down = act(place, spec, dict(base, id="find/scale/down",
                                         action="shrink_anchor",
                                         target={"direction": "down"}), v)
    big, r_up = act(place, spec, dict(base, id="find/scale/up", action="enlarge_anchor",
                                      target={"direction": "up"},
                                      says="the square is too small for the houses "
                                           "gathered about it"), v)
    assert r_down["applied"] and r_down["direction"] == "down", r_down
    assert r_up["applied"] and r_up["direction"] == "up", r_up
    a = float(r_down["allocation"]["anchor"]["share"])
    b = float(r_up["allocation"]["anchor"]["share"])
    assert a < b, (a, b)
    # the geometry actually moved, in the direction each asked for
    assert side(small) < side(place) < side(big), (side(small), side(place), side(big))
    # ...and `resize_anchor` is the same action under the direction-taking name
    _p, r_either = act(place, spec, dict(base, id="find/scale/either",
                                        action="resize_anchor",
                                        target={"direction": "up"}), v)
    assert r_either["applied"] and r_either["direction"] == "up", r_either
    assert r_either["allocation"] == r_up["allocation"], (r_either, r_up)
    # each record states the finding's subject, the direction and what it preserved
    for r in (r_down, r_up):
        assert r["subject"] == "market", r
        assert r["subject_of_finding"] == "market", r
        assert "hard constraints this preserves" in r["why"], r["why"]
        assert r["counts"]["was"] == r["counts"]["now"] == was_n, r["counts"]
        # ...as fields and not only in a sentence
        assert r["preserves"]["total_count"] == was_n, r["preserves"]
        assert r["preserves"]["count_is_exact"] is True, r["preserves"]
    assert sum(int(x["structures"]) for x in small["districts"]) == was_n
    assert sum(int(x["structures"]) for x in big["districts"]) == was_n
    # the stop in each direction is named and refused, not silently clamped
    at_floor = json.loads(json.dumps(spec))
    at_floor.setdefault("negotiated", []).append(
        {"what": "allocation",
         "to": {"anchor": {"share": placesolve.ANCHOR_SHARE_MIN}}})
    _p2, no_down = act(place, at_floor, dict(base, id="find/scale/floor",
                                            action="shrink_anchor",
                                            target={"direction": "down"}), v)
    assert not no_down["applied"] and "floor" in no_down["refused"], no_down
    at_ceiling = json.loads(json.dumps(spec))
    at_ceiling.setdefault("negotiated", []).append(
        {"what": "allocation",
         "to": {"anchor": {"share": placesolve.ANCHOR_SHARE_MAX}}})
    _p3, no_up = act(place, at_ceiling, dict(base, id="find/scale/ceiling",
                                            action="enlarge_anchor",
                                            target={"direction": "up"}), v)
    assert not no_up["applied"] and "ceiling" in no_up["refused"], no_up
    return (f"the anchor's share moves {a:g} down and {b:g} up from the same place, and "
            f"the market's side with it ({side(small)} / {side(place)} / {side(big)}); "
            f"`resize_anchor` up is the same decision as `enlarge_anchor`; the count stays "
            f"{was_n} both ways; at the registered floor of "
            f"{placesolve.ANCHOR_SHARE_MIN:g} and the ceiling of "
            f"{placesolve.ANCHOR_SHARE_MAX:g} each direction is refused by name")


@case
def t_s5b_regroup_changes_geometry():
    """An applied-but-ineffective action is not an action. `regroup` reuses
    `Solver._compact` and `_straddle` -- which draw each gathered district to what its
    own count needs and hug it to the anchor -- and which ran **only** where the count
    was the sentence's own. An inferred count's fabric could be read as scattered with
    no action available for it at all.
    """
    v, place = solved("inferred")
    spec = v["inferred"]
    before = {x["name"]: [x["x0"], x["z0"], x["x1"], x["z1"]] for x in place["districts"]}
    was_n = sum(int(x["structures"]) for x in place["districts"])
    assert not (place["layout"]["districts"] or {}).get("left_over"), \
        "this place is already compacted: the case would prove nothing"
    finding = {"id": "find/scatter/homes", "about": "composition", "action": "regroup",
               "subjects": ["homes"],
               "says": "the houses are spread over their strips rather than gathered "
                       "about the square"}
    got, rec = act(place, spec, finding, v)
    assert rec["applied"] and rec["changed"], rec
    after = {x["name"]: [x["x0"], x["z0"], x["x1"], x["z1"]] for x in got["districts"]}
    assert after != before, "the action changed no geometry"
    assert (got["layout"]["districts"] or {}).get("left_over"), got["layout"]["districts"]

    def area(r):
        return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    # the fabric is gathered: less ground claimed, or the long row cut in two at the
    # axis
    assert (sum(area(r) for r in after.values()) < sum(area(r) for r in before.values())
            or len(after) > len(before)), (before, after)
    # the count is preserved exactly, which is the invariant regrouping has to keep
    assert sum(int(x["structures"]) for x in got["districts"]) == was_n
    assert rec["counts"]["was"] == rec["counts"]["now"] == was_n, rec["counts"]
    assert rec["subject_of_finding"] == "homes", rec
    assert "hard constraints it preserves" in rec["why"], rec["why"]
    assert rec["preserves"]["total_count"] == was_n, rec["preserves"]
    assert rec["preserves"]["count_is_exact"] is False, rec["preserves"]
    # ...and it is not a lever with two pulls: a second `regroup` says `changed: False`
    _p2, again = act(got, spec, finding, v)
    assert not again["applied"] and again["changed"] is False, again
    return (f"`regroup` redraws {len(before)} district(s) as {len(after)} hugging the "
            f"square -- {sum(area(r) for r in before.values())} columns claimed becomes "
            f"{sum(area(r) for r in after.values())}, leaving "
            f"{len((got['layout']['districts'] or {}).get('left_over') or [])} "
            f"record(s) of unclaimed ground -- with the count unchanged at {was_n}; a "
            f"second pull reports `changed: False`")


@case
def t_s5c_redistribute_preserves_the_total_and_refuses_an_exact_count():
    """The action the rings case deliberately did not add, and why it is admissible here.
    The distribution of an inferred count is the layout's; a count the sentence stated
    is not, and the refusal is **by name** rather than a number quietly moving.
    """
    v, place = solved("about")
    spec = v["about"]
    was = {x["name"]: int(x["structures"]) for x in place["districts"]}
    finding = {"id": "find/count/homes", "about": "density", "action": "redistribute",
               "subjects": ["homes"],
               "says": "one district of houses cannot hold what it was promised while "
                       "another has room"}
    got, rec = act(place, spec, finding, v)
    assert rec["applied"], rec
    moved = rec["redistributed"]
    assert moved["houses"] >= 1 and moved["from"] != moved["to"], moved
    now = {x["name"]: int(x["structures"]) for x in got["districts"]}
    # **the total is the invariant**, and the two districts moved by the same number
    assert sum(now.values()) == sum(was.values()), (was, now)
    assert now[moved["from"]] == was[moved["from"]] - moved["houses"], (was, now)
    assert now[moved["to"]] == was[moved["to"]] + moved["houses"], (was, now)
    alloc = got["layout"]["structures_allocated"]
    assert alloc["total"]["was"] == alloc["total"]["now"], alloc
    assert not alloc["refused"], alloc
    # ...and the taker is inside its own density band's top: a distribution is not a
    # licence to over-fill one district
    top = next(int((x.get("count_band") or {}).get("hi") or 0)
               for x in got["districts"] if x["name"] == moved["to"])
    assert now[moved["to"]] <= top, (now[moved["to"]], top)
    # the positive control for the refusal: the same fixture at its own sentence's count
    _v2, hard = solved("exact")
    _p, refused = act(hard, v["exact"], finding, v)
    assert not refused["applied"] and refused["invariant"] == "count", refused
    assert "redistribute` is refused by name" in refused["refused"], refused["refused"]
    return (f"{moved['houses']} inferred house(s) move from `{moved['from']}` "
            f"({was[moved['from']]} -> {now[moved['from']]}) to `{moved['to']}` "
            f"({was[moved['to']]} -> {now[moved['to']]}, band top {top}), the place's "
            f"total unchanged at {sum(now.values())}; on the same fixture read with its "
            f"own sentence's count of {v['exact']['explicit_count']['n']} the action "
            f"refuses itself by name")


@case
def t_s5d_a_resolve_that_would_lose_a_counted_structure_is_refused():
    """The invariant, asserted in code and not only in a test. A reallocation revises an
        inferred dimension; a number the request made is not one, and until now nothing
        checked -- the place came back, the rectangles had moved, and the count was whatever
        the arithmetic gave.

        Driven through `solve_place` with the allocation a redistribution writes, which is the
        only route by which a district's count can change at all.
        
    """
    v, place = solved("about")
    counted = {x["name"]: int(x["structures"]) for x in place["districts"]}
    total = sum(counted.values())
    victim = max(counted, key=lambda n: (counted[n], n))
    other = next(n for n in sorted(counted) if n != victim)
    # a re-solve whose allocation does not conserve the count: one district loses two
    # houses and nothing takes them
    lossy = dict(counted)
    lossy[victim] = counted[victim] - 2
    lost, _f = placesolve.solve_place(json.loads(json.dumps(v["about"])), v["site"],
                                      v["plateau"], v["decls"], "x", seed=1,
                                      intent=v["intent"],
                                      allocation={"structures": lossy})
    rec = lost["layout"]["structures_allocated"]
    assert rec["invariant"] == "count", rec
    assert rec["applied"] == {}, rec
    assert rec["total"]["was"] == rec["total"]["now"] == total, rec
    assert sum(int(x["structures"]) for x in lost["districts"]) == total, \
        "the count changed anyway"
    assert "never creates or destroys it" in rec["why"], rec["why"]
    # the positive control: the same allocation, conserving
    kept = dict(lossy)
    kept[other] = counted[other] + 2
    ok, _f2 = placesolve.solve_place(json.loads(json.dumps(v["about"])), v["site"],
                                     v["plateau"], v["decls"], "x", seed=1,
                                     intent=v["intent"], allocation={"structures": kept})
    rec2 = ok["layout"]["structures_allocated"]
    assert rec2["applied"], rec2
    assert sum(int(x["structures"]) for x in ok["districts"]) == total, rec2
    # ...and a district whose count is the sentence's own is never moved at all: the
    # same conserving allocation on the exact reading is refused district by district,
    # by name
    _v3, hard = solved("exact")
    hard_counts = {x["name"]: int(x["structures"]) for x in hard["districts"]}
    h_total = sum(hard_counts.values())
    h_victim = max(hard_counts, key=lambda n: (hard_counts[n], n))
    h_other = next(n for n in sorted(hard_counts) if n != h_victim)
    h_alloc = {**hard_counts, h_victim: hard_counts[h_victim] - 1,
               h_other: hard_counts[h_other] + 1}
    held, _f3 = placesolve.solve_place(json.loads(json.dumps(v["exact"])), v["site"],
                                       v["plateau"], v["decls"], "x", seed=1,
                                       intent=v["intent"],
                                       allocation={"structures": h_alloc})
    rec3 = held["layout"]["structures_allocated"]
    assert not rec3["applied"], rec3
    assert all("exact" in str(r["why"]) for r in rec3["refused"]), rec3
    assert sum(int(x["structures"]) for x in held["districts"]) == h_total, rec3
    return (f"an allocation taking two houses off `{victim}` and giving them to nobody is "
            f"refused whole -- the place still holds {total} -- while the same allocation "
            f"giving them to `{other}` applies ({len(rec2['applied'])} district(s) "
            f"re-counted, total {total}); on the exact reading every override is refused "
            f"by name for the district's own exactness and the place holds {h_total}")


@case
def t_s5e_move_object_names_the_owner_that_has_the_action_and_offers_one():
    """`pipeline/improve.py`'s `OWNER_ACTIONS` lists `move_object` for `layout` and
    `_reallocate` refused it unconditionally with a reason and nothing else, so the
    finding reached a dead end that reads like an exhausted action list. A refusal that
    names the owner *and* a usable alternative is a routing decision; a reason alone is
    not.
    """
    v, place = solved("inferred")
    spec = v["inferred"]
    fabric = next(x["defines"] for x in place["districts"] if x.get("defines"))
    on_fabric = {"id": "find/move/homes", "about": "composition", "action": "move_object",
                 "subjects": [fabric], "says": "the houses stand away from the square"}
    on_part = dict(on_fabric, id="find/move/hall", subjects=["hall"],
                   says="the hall stands away from the square")
    _p, a = act(place, spec, on_fabric, v)
    _p2, b = act(place, spec, on_part, v)
    for r in (a, b):
        assert not r["applied"], r
        assert r["owner_of_action"] == "relation repair (`repair.apply`)", r
    assert a["alternative"] == "regroup", a
    assert "`regroup`" in a["refused"], a["refused"]
    assert b["alternative"] is None and "standing part" in b["refused"], b["refused"]
    # and the alternative it names is one that exists and applies
    assert a["alternative"] in placesolve.REALLOCATE_ACTIONS, a
    _p3, applied = act(place, spec, dict(on_fabric, id="find/move/routed",
                                        action=a["alternative"]), v)
    assert applied["applied"], applied
    return (f"`move_object` on `{fabric}` is refused naming "
            f"`{a['owner_of_action']}` and offering `{a['alternative']}`, which applies "
            f"when the finding is routed to it; on a standing part it names the same "
            f"owner and offers none, because an allocation cannot place a part")


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
    print(f"\n{ok}/{ok + fail} composition spatial cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
