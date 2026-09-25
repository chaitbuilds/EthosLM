"""The neighbourhood delivery round: one spatial decision, through every consumer.

    $PY scripts/test_delivery.py
    $PY scripts/test_delivery.py --case d1

The four seams the architectural audit. Every case states the rule it establishes,
carries a **positive control** beside its counterexample, and prints the measurement
that establishes it. Nothing here builds a city; what it builds is a compile, a
generated production program and a few records.

  d1  the plan's attachment, frontage and second stone reach `site()`, and the generated
      production call names them
  d2  admissibility governs the final parameters, and an end of a terrace is a different
      question from its middle
  d3  a court a district's form owes is composed and entered, not rounded to zero and not
      the area left where a housing row was removed
  d4  alternatives are generated from the adopted design, the incumbent is in the
      comparison, and a proposal that cannot move the finding's own measure is refused
  d5  a section's court denominator is the section's, and its route verdict is a
      traversal of the assembled world
  d6  promotion protects the explicit requirements and accounts for the tradeoffs,
      including qualities already failing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import arrange, district_compile as dc, pipeline, placeplan, section  # noqa: E402
from ethoslm.buildlib import Builder                                     # noqa: E402
from ethoslm.pipeline import promote                                     # noqa: E402

CASES: list = []


class Skip(Exception):
    """This case has no evidence to run on and says so rather than passing."""


def case(fn):
    CASES.append(fn)
    return fn


# --------------------------------------------------------------- a district to compile

FORM = "east_asian"
RECT = (0, 0, 119, 79)


def _decls(role="urban"):
    _t, decls = placeplan.types_card(None, FORM, role)
    return decls


def _part(density="dense", character=None, structures=40):
    return {"name": "houses", "kind": "defining_part", "density": density,
            "role": "urban", "structures": structures,
            "character": dict(character or {})}


def _spec(part):
    return {"sentence": "a quarter", "form": FORM, "structures": part["structures"],
            "defining_parts": [part], "voice": "japanese_minka"}


def _district(part, rect=RECT, arrangement=None):
    d = {"name": "houses_1", "defines": "houses", "kind": "district",
         "x0": rect[0], "z0": rect[1], "x1": rect[2], "z1": rect[3],
         "structures": part["structures"], "density": part["density"],
         "role": part["role"]}
    if arrangement:
        d["arrangement"] = dict(arrangement)
    return d


def _place(spec, district):
    return {"districts": [district], "layout": {"seed": 1},
            "structures": spec["structures"]}


def _compile(part, character=None, arrangement=None, rect=RECT):
    p = dict(part, character=dict(character or part.get("character") or {}))
    spec = _spec(p)
    d = _district(p, rect, arrangement)
    place = _place(spec, d)
    decls = _decls()
    ch = dc.character_of(p, d)
    return dc.compile_district(d, p, place, decls, spec=spec, seed=1), ch, d, place, decls


def _leaves(got):
    return [q for r in (got[1].get("plan") or got[0] or {}).get("quarters", [])
            for q in (r.get("plots") or [])] if isinstance(got, tuple) else [
        q for r in ((got or {}).get("quarters") or []) for q in (r.get("plots") or [])]


# ------------------------------------------------ d1: the decision reaches construction

@case
def d1_the_plans_attachment_and_frontage_reach_the_generated_production_call():
    """**`pipeline.PART_GEOMETRY` is the whole of what a leaf hands `site()`.**

        The audit's first cause. `stages_build.instantiated_source` filters every leaf
        through this tuple before it writes the program that builds it, and `settle_ground`
        filters it again before `declare()` reads the ground. A field the compiler decided
        and this tuple drops is a decision that never reached a block: `Builder._insets` then
        sees four free sides, a 6x13 attached lot hands `build()` a 4x11 pad, and the row
        house that stands on it is four columns wide with its party walls four columns short
        of its neighbours'.

        The counterexample and its control are the same lot, sited both ways.
        
    """
    for f in ("attached", "front", "wall_alt"):
        assert f in pipeline.PART_GEOMETRY, f"PART_GEOMETRY drops `{f}`"
    lot = {"label": "b0_0_01", "kind": "plot", "x0": 0, "z0": 0, "x1": 5, "z1": 12,
           "front": "north", "attached": ["west", "east"], "wall_alt": True,
           "type": "row_house", "seed": 3}
    free = Builder.pad_extent({k: v for k, v in lot.items() if k != "attached"})
    held = Builder.pad_extent(lot)
    assert held[0] > free[0], (free, held)
    # the generated production call, composed exactly as `instantiate_part` composes it
    src = pipeline.instantiated_source(
        "KIND='plot'\nNEEDS={'footprint': (4, 4, 6, 32)}\nPARAMS={}\n"
        "def build(b, part, seed, **kw):\n    return {'ok': True}\n",
        [({k: lot[k] for k in pipeline.PART_GEOMETRY if k in lot}, 3, {"storeys": 2})])
    for f in ("attached", "front", "wall_alt"):
        assert f"'{f}'" in src, f"the generated call drops `{f}`:\n{src[-400:]}"
    # ...and a leaf the compiler laid carries them
    got, ch, d, place, decls = _compile(_part(character={"attached": True}))
    plots = [p for p in _leaves(got[0] if isinstance(got, tuple) else got)
             if p.get("kind") == "plot"]
    att = [p for p in plots if p.get("attached")]
    assert att, "the compiler laid no attached lot to carry anything"
    assert all(p.get("front") for p in att), "an attached lot carries no front"
    return (f"`PART_GEOMETRY` carries attached, front and wall_alt; the same 6x13 lot "
            f"sites to a {free[0]}x{free[1]} pad with free flanks and "
            f"{held[0]}x{held[1]} with its party walls, so the house is "
            f"{held[0] - free[0]} column(s) wider for the attachment; the generated "
            f"production program names all three; the compiler laid {len(att)} attached "
            f"lot(s) of {len(plots)}, every one with a front")


# ------------------------------------------------- d2: admissibility governs the params

@case
def d2_no_leaf_asks_for_more_storeys_than_its_own_flanks_admit():
    """**Variation moves inside what the lot admits, not over the top of it.**

        The audit's first cause in its second half. The storeys were drawn in `leaf()`
        against the envelope, and `lots_along`'s "no two neighbours are the same building"
        step then moved `params.storeys` *afterwards* -- so the delivered candidate records
        `storeys_admitted: 1` beside `params.storeys: 2` on the same leaf, which is the plan
        disagreeing with itself in writing. And the flanks are not known when a lot is drawn:
        a lot dropped for the road frees its neighbour's flank and an end of a terrace has
        one free flank whatever the fabric intended, so the question has to be asked again
        once the row is complete (`district_compile.settle_storeys`).
        
    """
    got, ch, d, place, decls = _compile(_part(character={"attached": True}))
    rec = got[1] if isinstance(got, tuple) else {}
    plots = [p for p in _leaves(got[0] if isinstance(got, tuple) else got)
             if p.get("kind") == "plot"]
    over = [p["name"] for p in plots
            if (p.get("envelope") or {}).get("storeys_admitted") is not None
            and int((p.get("params") or {}).get("storeys", 1))
            > int(p["envelope"]["storeys_admitted"])]
    assert not over, f"{len(over)} leaf/leaves ask over their admitted band: {over[:5]}"
    # the record says which configuration each answer is about, and the ends differ
    flanks = {p["name"]: len(p.get("attached") or ()) for p in plots}
    asked = {p["name"]: (p.get("envelope") or {}).get("flanks_attached")
             for p in plots if p.get("envelope")}
    wrong = [n for n, v in asked.items() if v is not None and v != flanks.get(n)]
    assert not wrong, f"the envelope was asked the wrong flank configuration for {wrong[:5]}"
    ends = sorted(n for n, k in flanks.items() if k == 1)
    middles = sorted(n for n, k in flanks.items() if k == 2)
    assert ends and middles, f"no row with both ends and middles: {flanks}"
    return (f"{len(plots)} lot(s): {len(middles)} with both flanks against a neighbour "
            f"and {len(ends)} at an end of a row with one free; every recorded envelope "
            f"is an answer about that lot's own configuration and no leaf asks for more "
            f"storeys than its own admits; {rec.get('storeys_settled')} parameter(s) "
            f"were settled after the attachment was known")


# ---------------------------------------------------------------- d3: a composed court

@case
def d3_a_court_the_form_owes_is_composed_and_entered():
    """**A court is a realizable composition, not a share rounded to zero.**

        The audit's fourth cause. `demand.court_obligation` publishes the obligation off the
        district's `courtyard_share`, and the compiler laid `round(share * blocks)` courts --
        so a district at 0.05 over eight blocks *owed* a court and laid **none**, which is
        the whole of "the crowded ring has no court" on the delivered candidate. And the
        court it did lay when the share was large enough was the block's back row, open along
        both short faces, with the cross streets carrying nothing.

        The control is the same district compiled with the share at zero: nothing is owed and
        nothing is laid.
        
    """
    from ethoslm import demand as demand_mod
    got, ch, d, place, decls = _compile(
        _part(character={"attached": True, "courtyard_share": 0.05}))
    rec = got[1] if isinstance(got, tuple) else {}
    obl = demand_mod.court_obligation(
        {"name": "houses_1", "kind": "district", "density": "dense",
         "character": dict(ch)})
    assert obl, "the form owes no court to compose"
    assert int(rec.get("courts") or 0) >= 1, (
        f"the form owes a court ({obl}) and the compiler laid {rec.get('courts')}")
    assert rec.get("court_obligation"), "the record does not publish the obligation"
    zero, ch0, _d0, _p0, _dc0 = _compile(
        _part(character={"attached": True, "courtyard_share": 0.0}))
    rec0 = zero[1] if isinstance(zero, tuple) else {}
    assert rec0.get("court_obligation") is None, (
        f"a district whose form owes no court publishes an obligation: "
        f"{rec0.get('court_obligation')}")
    entered = int(rec.get("courts_entered") or 0)
    return (f"a dense district at a 0.05 court share over {rec.get('blocks')} block(s) "
            f"lays {rec.get('courts')} court(s) and publishes the obligation "
            f"({obl.get('from')}); {rec.get('perimeter_blocks') or 0} block(s) were "
            f"composed about a court, {rec.get('perimeter_shut') or 0} closed on four "
            f"ranges and {entered} entered through a passage from the street; the same "
            f"district at a share of 0 owes none and lays {rec0.get('courts')}")


# ------------------------------------------- d4: alternatives from the adopted design

@case
def d4_alternatives_are_generated_from_the_adopted_design():
    """**The comparison varies the fabric that is standing.**

        The audit's second cause. `arrangements` read `character_of(part)` with no district,
        so the base it varied was the *brief's* fabric and not the one the layout had
        negotiated for this rectangle -- and the `allocation` argument it already took was
        never read. Measured on the delivered candidate: the incumbent lays 6x13 and both
        `compound` and `row_depth` were offered at **10x10**, so a trial that named a row
        count changed the building as well as the number of rows.

        And a proposal has to be able to move the number the finding is about: naming an
        owner and an action does not establish that the action can affect the subject.
        
    """
    part = _part(character={"attached": True})
    decls = _decls()
    adopted = {"lot_width": 6, "lot_depth": 13, "rows": 2}
    d = _district(part, RECT, adopted)
    plain = arrange.arrangements(part, decls, spec=_spec(part))
    with_it = arrange.arrangements(part, decls, spec=_spec(part), district=d)
    base_plain = next((r["lot"] for r in plain if r.get("action") == "as_declared"), None)
    base_adopt = next((r["lot"] for r in with_it if r.get("action") == "as_declared"), None)
    assert base_adopt == [6, 13], (
        f"the base of the comparison is {base_adopt} and the adopted lot is 6x13")
    inc = [r for r in with_it if r.get("incumbent")]
    assert inc, "the incumbent is not in the comparison"
    off = [r["action"] for r in with_it
           if r.get("arrangement") and r["action"] in ("row_depth", "compound")
           and r["lot"][0] != 6]
    assert not off, f"these alternatives changed the bay as well as what they name: {off}"
    # the finding's own measure, and what a row estimates for it
    key, way = arrange.finding_measure(
        {"measure": "section.crowded.court_share",
         "target": {"measure": "section.crowded.court_share", "direction": "up"}})
    assert (key, way) == ("courts", "up"), (key, way)
    none = arrange.finding_measure({"says": "the street is dull"})
    assert none == (None, None), none
    return (f"with no district the comparison's base is {base_plain} and with the "
            f"adopted arrangement it is {base_adopt}; {len(inc)} row(s) are marked as "
            f"the incumbent; {len(with_it)} alternative(s) are one move from the fabric "
            f"that is standing and none of them changes the bay while naming something "
            f"else; a finding naming `section.crowded.court_share` resolves to the "
            f"`{key}` estimate going `{way}`, and one naming no measure resolves to "
            f"nothing rather than to a guess")


# ------------------------------------------------------- d5: the section's own rulers

@case
def d5_the_section_measures_its_own_scope_and_walks_its_own_world():
    """**A section is a section, and a route verdict is a walk.**

        Two of the audit's evidence limits. `_courts` took its denominator from every
        court-owing district of the *city*, so a section record failed on eighteen subjects
        of which five were in it; and `_route` read `circulate.walk_check`, which re-derives
        reachability over the **planned lane graph** -- the right check at planning time and
        a different question afterwards, because it cannot see a pad laid over a lane or a
        wall closed across a threshold.
        
    """
    inside = {"name": "a", "x0": 0, "z0": 0, "x1": 50, "z1": 50}
    outside = {"name": "b", "x0": 400, "z0": 400, "x1": 450, "z1": 450}
    sec = [0, 0, 100, 100]
    assert section._in_section(inside, sec) and not section._in_section(outside, sec)
    assert section._in_section(outside, None), "no section means everything is in scope"
    state = os.path.join(ROOT, "out", "sd-city")
    if not os.path.exists(os.path.join(state, "parts.json")):
        raise Skip("no retained out/sd-city to measure")
    got = section.record(state, of="sd-city")
    cts = next(r for r in got["relationships"] if r["id"] == "courts")
    m = cts["measured"]
    assert m.get("section_rect"), "the court record does not name the section it is of"
    assert m.get("out_of_section") is not None, "out-of-section districts are not named"
    rt = next(r for r in got["relationships"] if r["id"] == "route")
    walked = (rt.get("measured") or {}).get("walked")
    assert isinstance(walked, dict) and walked.get("artifact"), (
        "the route verdict carries no traversal of an assembled world")
    assert walked.get("reached"), "the traversal reached no threshold at all"
    # the two rulers, apart: the neighbour gap between buildings and between extents
    con = next(r for r in got["relationships"] if r["id"] == "contrast")
    sides = {s["side"]: s for s in con["measured"]["sides"]}
    for s in sides.values():
        assert s.get("median_neighbour_gap_by_extent") is not None, (
            "the old neighbour-gap figure is not retained beside the corrected one")
    return (f"the court denominator is the {m['subjects']} subject(s) the section's own "
            f"rectangle reaches, with {len(m['out_of_section'])} court-owing district(s) "
            f"of the place named as outside it rather than failed in it; the route "
            f"verdict is a walk on {walked['artifact']} reaching "
            f"{len(walked['reached'])} of {walked['with_stance']} built threshold(s) "
            f"without jumping, with the planned graph's own answer beside it; the "
            f"neighbour gap is measured between buildings "
            f"({sides['crowded']['median_neighbour_gap']}) with the old figure between "
            f"emitted extents retained beside it "
            f"({sides['crowded']['median_neighbour_gap_by_extent']})")


# ---------------------------------------------------------- d6: what a promotion owes

@case
def d6_promotion_protects_the_requirements_and_accounts_for_the_tradeoffs():
    """**Type survival and one improving scalar are too weak.**

        The audit's fifth cause. Promotion protected demonstrated relationship *labels* and
        the continued presence of each type; it did not protect an explicit requirement that
        was met and is not any more, and it could not see a quality worsening inside a
        relationship that was already failing -- so a trial could lose a required feature, or
        make an already-failed court verdict worse, and promote on one scalar.
        
    """
    def sec(features_held, features_owed, unreached, crowded_cover):
        return {"by": "ethoslm.section.record", "relationships": [
            {"id": "contrast", "status": "failed", "measured": {"sides": [
                {"side": "crowded", "built_cover": crowded_cover, "structures": 70,
                 "median_neighbour_gap": 1, "court_share": 0.0, "median_storeys": 1},
                {"side": "calm", "built_cover": 0.05, "structures": 9}]}},
            {"id": "anchor", "status": "demonstrated", "measured": {"stood": 1}},
            {"id": "courts", "status": "failed",
             "measured": {"asked": 3, "held": 3, "owed": [], "omitted": []}},
            {"id": "route", "status": "demonstrated",
             "measured": {"walked": {"unreached": ["g"] * unreached, "no_stance": []}}},
            {"id": "features", "status": "failed",
             "measured": {"held": features_held, "owed": features_owed}}]}
    was = section_quantities(sec(7, 0, 0, 0.26))
    lost = section_quantities(sec(5, 2, 0, 0.30))
    kept = section_quantities(sec(7, 0, 0, 0.24))
    assert promote._moved(was["features held"]["value"],
                          lost["features held"]["value"], "up") == "worse"
    assert promote._moved(was["crowded built cover"]["value"],
                          lost["crowded built cover"]["value"], "up") == "better"
    assert promote._moved(was["crowded built cover"]["value"],
                          kept["crowded built cover"]["value"], "up") == "worse"
    hard = {h[0] for h in promote.PROTECTED_HARD}
    assert "features held" in hard and "crowded built cover" not in hard
    # the correction path: a reading is corrected on the record, never with an editor
    rows = [{"of": "inspection/views.json",
             "path": "reading.findings[b1].owner", "was": "layout", "now": "capability",
             "why": "three trials established that the missing court is the approved pool"}]
    got = promote.apply_corrections(rows, [{"id": "b1", "owner": "layout",
                                            "says": "no court in the crowded ring"}])
    assert got[0]["owner"] == "capability" and got[0]["was"]["owner"] == "layout"
    assert got[0]["corrected"][0]["why"].startswith("three trials")
    return (f"{len(hard)} quantity(ies) are protected as explicit requirements and "
            f"{len(promote.PROTECTED_ACCOUNTED)} as composition qualities to account "
            f"for; a trial that raises crowded built cover from 0.26 to 0.30 while "
            f"losing two required features is a regression on `features held` and not a "
            f"promotion, and a trial that keeps every requirement while built cover "
            f"falls 0.26 -> 0.24 carries that fall as a recorded tradeoff; a correction "
            f"to a reading re-attributes `b1` from layout to capability through "
            f"`promote.correct`, with the original kept on the finding as `was`")


def section_quantities(sec: dict) -> dict:
    return promote._quantities(sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="")
    a = ap.parse_args()
    want = [c for c in CASES if not a.case or c.__name__.startswith(a.case)]
    ok = bad = skipped = 0
    for fn in want:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {fn.__name__}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {fn.__name__}: {says}")
    print(f"\n{ok}/{ok + bad} delivery cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
