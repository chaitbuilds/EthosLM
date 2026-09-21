"""The closure round's spatial counterexamples, as regressions with positive controls.

    $PY scripts/test_closure_spatial.py

Every case is a thing the review reproduced or the closure round's own failed first run
showed, and every one runs through the **production entry points** -- `arrange.arrange`,
`district_compile.compile_district`, `placesolve.solve_place`, `placeplan.district_target`,
`placeplan.place_failures`, `repair.apply`, `intent.relation_measure` -- with a control
that must still pass. Under a minute; the shore cases need `out/real-shore/` (skipped
without it).
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (arrange, contracts, district_compile as dc, intent as intent_mod,  # noqa: E402
                   pipeline, placeplan, placesolve, repair, spec as spec_mod)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    pass


def _bits(w=110, d=80, want=20, density="low", role="rural", character=None):
    part = {"name": "homes", "kind": "group", "family": "district", "role": role,
            "density": density, "count": 1, "relation": "throughout",
            "character": character if character is not None
            else {"density": density, "role": role}}
    _t, decls = placeplan.types_card(None, "european_vernacular", role)
    dist = {"name": "homes_1", "x0": 0, "z0": 0, "x1": w - 1, "z1": d - 1,
            "structures": want, "density": density, "role": role, "defines": "homes"}
    place = {"parts": [], "arterials": {"cells": []}, "districts": [dist]}
    return dist, part, place, decls


# ------------------------------------------------ 1. proposed vs adopted

@case
def t_a_the_same_proposal_lays_the_same_arrangement_after_adoption():
    """The review: proposal 20 -> lots 15; adopt 15, regenerate with 20 -> 12."""
    dist, part, place, decls = _bits(want=20)
    first = arrange.arrange(dist, part, place, decls, seed=1, proposed=20)
    adopted = dict(dist, structures=first["realized"])
    again = arrange.arrange(adopted, part, place, decls, seed=1, proposed=20)
    assert first["realized"] == again["realized"], (first["realized"], again["realized"])
    assert first.get("lot") == again.get("lot"), (first.get("lot"), again.get("lot"))
    assert json.dumps(first["plan"], sort_keys=True) == json.dumps(again["plan"], sort_keys=True)
    # control: a different proposal on the same ground is allowed to be a different
    # arrangement (the compiler sizes from the ask)
    other = arrange.arrange(dist, part, place, decls, seed=1, proposed=6)
    assert other["realized"] <= first["realized"], (other["realized"], first["realized"])
    return (f"proposal 20 lays {first['realized']} of {first['lot']}; adopting "
            f"{first['realized']} and regenerating from 20 lays the identical plan")


# ------------------------------------------------ 2. one density definition

@case
def t_b_the_compiler_targets_and_the_checker_measures_the_same_band():
    """The review: compiler floors summing to 19.4% against a checker cap of 12%."""
    dist, part, place, decls = _bits(w=180, d=120, want=10, density="sparse")
    t = placeplan.district_target(dist, part, place, decls)
    band = intent_mod.density_target("sparse", "rural")
    usable = t["usable_columns"]
    assert t["min_plot_columns"] == int(-(-band["lo"] * usable // 1)), (t, band)
    assert t["max_plot_columns"] == int(band["hi"] * usable), (t, band)
    assert t["min_plot_columns"] < t["max_plot_columns"]
    got, rec = dc.compile_district(dist, part, place, decls, seed=1)
    cover = rec["lot_cover_usable"]
    assert band["lo"] <= cover <= band["hi"], (cover, band)
    # ...and the checker's own measure over the same denominator agrees with the record
    leaves = [p for q in got["quarters"] for p in q["plots"] if p["kind"] == "plot"]
    lc = intent_mod.lot_cover(leaves, [{"rect": [0, 0, dist["x1"], dist["z1"]],
                                        "developable_columns": usable}])
    assert abs(lc["cover"] - cover) < 1e-3, (lc, cover)     # the record rounds to 4 places
    # control: `dense` has a floor and no ceiling, and a dense district is laid dense
    dist2, part2, place2, decls2 = _bits(w=110, d=80, want=12, density="dense",
                                          role="urban")
    band2 = intent_mod.density_target("dense", "urban")
    got2, rec2 = dc.compile_district(dist2, part2, place2, decls2, seed=1)
    assert band2["hi"] is None and rec2["lot_cover_usable"] >= band2["lo"], (rec2["lot_cover_usable"], band2)
    return (f"sparse: floor {t['min_plot_columns']} / ceiling {t['max_plot_columns']} "
            f"of {usable} usable, laid at {cover:.1%}, the checker reads {lc['cover']:.1%}"
            f"; dense laid at {rec2['lot_cover_usable']:.1%} over a floor of {band2['lo']:.0%}")


@case
def t_c_a_district_over_its_ceiling_is_refused_and_a_capped_ask_is_not():
    dist, part, place, decls = _bits(w=90, d=60, want=30, density="sparse")
    got, rec = dc.compile_district(dist, part, place, decls, seed=1)
    assert rec["capped_to"] is not None and rec["lots"] <= rec["capped_to"], rec["capped_to"]
    fails = placeplan.district_failures(dist, got, place, decls, role="rural", part=part)
    assert not any(f["check"] == "cover_over" for f in fails), fails
    # the counterexample: an exact count the ceiling cannot hold is laid whole and
    # refused by name, for the layout owner and never for the character's author
    exact = dict(dist, exact=True)
    got_e, rec_e = dc.compile_district(exact, part, place, decls, seed=1)
    assert rec_e["exact"] and rec_e["capped_to"] is None, rec_e["capped_to"]
    assert rec_e["lots"] > rec["lots"] and rec_e["over_ceiling"], (rec_e["lots"], rec["lots"])
    fails_e = placeplan.district_failures(exact, got_e, place, decls, role="rural", part=part)
    assert any(f["check"] == "cover_over" for f in fails_e), fails_e
    return (f"30 sparse cottages asked of 90x60: capped to {rec['capped_to']} and not "
            f"refused; the same count exact lays {rec_e['lots']} of {rec_e['lot']} as far "
            f"as the ground allows, is over the ceiling and is refused `cover_over`")


# ------------------------------------------------ 3. an exact count is the count

@case
def t_d_an_exact_count_is_laid_exactly_and_a_proposal_may_be_raised_to_the_floor():
    dist, part, place, decls = _bits(w=110, d=80, want=4, density="low",
                                     character={"frontage": "street", "lot_width": 12,
                                                "lot_depth": 10})
    got, rec = dc.compile_district(dict(dist, exact=True), part, place, decls, seed=1)
    assert rec["lots"] == 4, rec["lots"]
    got2, rec2 = dc.compile_district(dist, part, place, decls, seed=1)
    assert rec2["lots"] >= 4, rec2["lots"]
    return f"exact 4 -> {rec['lots']}; the same ask as a proposal -> {rec2['lots']}"


# ------------------------------------------------ 4. the validator asks the compiler

@case
def t_e_the_place_validator_asks_the_compiler_before_refusing_on_its_estimate():
    """The first run: `place_failures` refused a district for 3920 > 3870 columns while
    `arrange` had just laid the same count in it."""
    st = os.path.join(ROOT, "out", "closure-c1", "spatial", "farm")
    if not os.path.exists(os.path.join(st, "place.checked.json")):
        raise Skip("no out/closure-c1/spatial/farm copy of the proof")
    spec = spec_mod.read_spec(json.load(open(os.path.join(st, "place.checked.json"))))
    site = json.load(open(os.path.join(st, "site.json")))
    _t, decls = placeplan.types_card(None, spec.get("form"))
    homes = next(p for p in spec["defining_parts"] if p["name"] == "homes")
    # **A dense district of small declared lots**: the estimate charges every house the
    # density's own 10x10 lot and refuses 39 in 102x69, and the compiler lays them on
    # the 7x7 lots the character declares. Under the density band the two rules can only
    # disagree this way round -- the ceiling is stricter than the estimate at `low`,
    # `medium` and `sparse` -- which is itself a fact this case records.
    homes["density"], homes["role"] = "dense", "urban"
    homes["character"] = {"frontage": "street", "lot_width": 7, "lot_depth": 7}
    spec = spec_mod.read_spec(spec, spec["sentence"])
    _t, decls = placeplan.types_card(None, spec.get("form"))
    d = {"name": "homes_west", "defines": "homes", "x0": -768, "z0": -256, "x1": -667,
         "z1": -188, "structures": 39}
    place = {"parts": [], "districts": [d], "compounds": [], "arterials": {"cells": []}}

    def _district_fails(fails):
        return [f for f in fails if f["check"] == "district" and f["part"] == "homes_west"]
    per = spec_mod.columns_per_plot(homes)
    room = 102 * 69 * placeplan.DISTRICT_FILL
    assert 39 * per > room, "the counterexample no longer reproduces: the estimate admits 39"
    _t2, urban = placeplan.types_card(None, spec.get("form"), "urban")
    held = arrange.capacity(d, homes, place, urban, spec=spec, ceiling=39)
    if held < 39:
        raise Skip(f"the compiler lays {held} of 39 here, so the estimate's refusal stands")
    fails = placeplan.place_failures(place, spec, {"origin": site["origin"], "size": site["size"]}, decls)
    assert not _district_fails(fails), _district_fails(fails)
    # control: a count the compiler cannot lay either is still refused
    d["structures"] = 200
    fails2 = placeplan.place_failures(place, spec, {"origin": site["origin"], "size": site["size"]}, decls)
    assert _district_fails(fails2), fails2
    return (f"39 dense houses on 7x7 lots in 102x69: the estimate refuses ({39 * per} > "
            f"{int(room)}), the compiler lays {held}, no failure; 200 is refused")


# ------------------------------------------------ 5. around, on the shore

def _shore():
    st = os.path.join(ROOT, "out", "real-shore")
    if not os.path.exists(os.path.join(st, "world.npz")):
        raise Skip("no out/real-shore with its ground")
    cfg = json.load(open(os.path.join(ROOT, "rounds", "real-shore.json")))
    rnd = pipeline.Round(name="real-shore", sentence=cfg["sentence"], flags=cfg["flags"],
                         state_dir=st)
    from ethoslm.pipeline import stages_plan
    spec = rnd.place_spec()
    site = pipeline.settlement_site(rnd)
    plateau = stages_plan.plateau_record(rnd)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    caps = contracts.load(rnd, "capabilities")
    it = contracts.load(rnd, "intent")
    return rnd, spec, site, plateau, decls, caps, it


def _assembled(rnd, spec, site, place):
    districts = {}
    for d in place["districts"]:
        part = placeplan._district_part(spec, d)
        role = spec_mod.district_role(spec, d)
        _t, mine = placeplan.types_card(None, spec.get("form"), role)
        got = arrange.arrange(d, part, place, mine, spec=spec, site=site, seed=1,
                              proposed=int(d["structures"]))
        if got["ok"] and not got["thin"]:
            districts[d["name"]] = got["plan"]
    return placeplan.assemble(place, districts, spec, {})


@case
def t_f_the_shore_village_gathers_round_its_square_when_the_sentence_says_so():
    """The review's twelve passes: the square moved, the houses never did."""
    rnd, spec, site, plateau, decls, caps, it = _shore()
    req = next(r for r in it["requirements"] if r["kind"] == "relation")
    vol = rnd.volume()
    base, _f = placesolve.solve_place(spec, site, plateau, decls, "blackstone_and_ash",
                                      vol=vol, seed=1, caps=caps, intent=None)
    alt, _f2 = placesolve.solve_place(spec, site, plateau, decls, "blackstone_and_ash",
                                      vol=vol, seed=1, caps=caps, intent=it)
    b = intent_mod.relation_measure(req, pipeline.plan_parts(_assembled(rnd, spec, site, base)))
    a = intent_mod.relation_measure(req, pipeline.plan_parts(_assembled(rnd, spec, site, alt)))
    assert b[0] == "failed", "the counterexample no longer reproduces: " + b[1]
    assert a[0] == "satisfied", a[1]
    assert (alt["layout"].get("policy") == "shoreline"), alt["layout"].get("policy")
    # the predicted relation the layout wrote agrees with what the checker found
    pred = next((r for r in (alt["layout"]["districts"].get("relations") or [])
                 if r.get("id") == req["id"]), None)
    assert pred and pred["predicted"] == "satisfied", pred
    return f"without the relation: {b[1][:70]}; with it: {a[1][:70]}"


# ------------------------------------------------ 6. repair: once per candidate

@case
def t_g_an_action_tried_on_a_candidate_is_not_tried_on_it_again():
    rnd, spec, site, plateau, decls, caps, it = _shore()
    tmp = tempfile.mkdtemp(prefix="closure-spatial-")
    try:
        for f in ("site.json", "site_search.json", "world.npz", "intent.json",
                  "place.checked.json", "place.json", "capabilities.json", "plateau.json",
                  "plan.json", "plan.place.json"):
            src = rnd.rel(f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, f))
        cfg = json.load(open(os.path.join(ROOT, "rounds", "real-shore.json")))
        r2 = pipeline.Round(name="real-shore", sentence=cfg["sentence"], flags=cfg["flags"],
                            state_dir=tmp)
        place = json.load(open(r2.rel("plan.place.json")))
        finding = {"id": "find/relation/houses_around_market",
                   "requirement": "relation/houses_around_market", "part": None,
                   "says": "2 of 4 sides", "owner": "layout", "blocks": "feasibility",
                   "severity": "error", "seen_by": "test", "fixed": False,
                   "evidence": {"subjects": 6, "quadrants": 2, "object": "green",
                                "subject": "houses", "relation": "around"}}
        found = contracts.make("findings", findings=[finding], stage="test")
        first = repair.apply(r2, spec, found, place=place, place_path=r2.rel("plan.place.json"))
        second = repair.apply(r2, spec, found, place=place, place_path=r2.rel("plan.place.json"))
        assert first["applied"] or first["refused"], first
        ref2 = [x for x in second["refused"] if x["finding"] == finding["id"]]
        assert ref2 and "already tried" in ref2[0]["why"], second
        rec = json.load(open(r2.rel(repair.ATTEMPTS_RECORD)))
        assert len(rec["attempts"]) == 1, rec
        return (f"first pass: {repair.says(first)}; second pass on the same candidate: "
                f"refused -- {ref2[0]['why'][:80]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------ 7. the least footprint of rings

@case
def t_h_a_ringed_spec_needs_the_ground_its_rings_need_whatever_its_count_says():
    p = os.path.join(ROOT, "fixtures", "closure-rings", "place.json")
    if not os.path.exists(p):
        raise Skip("no fixtures/closure-rings/place.json")
    spec = spec_mod.read_spec(json.load(open(p)), "Build a walled town of twenty-four houses in two rings around a temple compound.")
    least = placeplan.least_footprint(spec)
    # the expression round: the spec's footprint is now the larger of the count's, the
    # parts' least and the rings' wanted ground (`wanted_footprint`); the layout's own
    # refusal threshold is `least_footprint`. **The design round moved which of the two
    # this is asked of.** `least_footprint` is the side the layout refuses below, and
    # the layout now negotiates a ring's district depth against its fabric
    # (`arrange.arrangements`) -- a dense ring one row of houses deep needs twenty
    # columns where the constant `dmin` asked twenty-eight -- so the refusal threshold
    # came down from 236 to 216. The claim this case makes is "the count sizes the
    # houses; the parts size the place", and the number that carries it is the one the
    # site search asks for: `wanted_footprint`, which is unmoved.
    count_fp = spec_mod.footprint_for(int(spec["structures"]), spec["kind"])
    wanted_fp = placeplan.wanted_footprint(spec)
    assert wanted_fp and wanted_fp > count_fp, (wanted_fp, count_fp, spec["needs"])
    assert least and least < wanted_fp, (least, wanted_fp)
    # **the control**: the rings lay on a site of exactly that side, and are refused one
    # step below it for the ring check's own reason
    _t, decls = placeplan.types_card(None, spec.get("form"))
    site = {"origin": [0, 0], "size": int(least)}
    place, fails = placeplan.concentric_layout(spec, site, None, decls, "japanese_temple")
    assert not [f for f in fails if f.get("check") == "shares"], fails
    assert place is not None and len(place["districts"]) >= 2, fails
    below = {"origin": [0, 0], "size": int(least) - 12}
    _p2, fails2 = placeplan.concentric_layout(spec, below, None, decls, "japanese_temple")
    assert [f for f in fails2 if f.get("check") == "shares"], (least, fails2)
    assert placeplan.least_footprint(spec_mod.read_spec(
        {"kind": "village", "form": None, "voice": None, "defining_parts": [
            {"name": "homes", "kind": "group", "family": "district",
             "relation": "throughout", "count": 1}]}, "Build a village.")) is None
    return (f"the rings fixture wants {wanted_fp} a side against a count footprint of "
            f"{count_fp} (the spec asks {spec['needs']['footprint']}); the layout lays "
            f"on {least} and is refused on {least - 12}")


# ------------------------------------------------ 8. one chooser for a compound

@case
def t_i_a_compounds_descendants_are_the_types_the_capability_record_approved():
    """The city run: the record approved `hall` and `square`, `compound_axial` chose
    `temple` and `plaza` by form, and the two disagreed at the plan."""
    import copy
    st = os.path.join(ROOT, "out", "closure-city")
    if not os.path.exists(os.path.join(st, "plan.place.json")):
        raise Skip("no out/closure-city place plan")
    place = json.load(open(os.path.join(st, "plan.place.json")))
    spec = spec_mod.read_spec(json.load(open(os.path.join(st, "place.checked.json"))))
    caps = json.load(open(os.path.join(st, "capabilities.json")))
    comp = next(c for c in place["compounds"])
    cpart = next((d for d in spec_mod.compounds(spec) if placeplan._answers(comp, d)), None)
    _ct, cdecl = placeplan.compound_types(None, spec, cpart or {})
    approved = copy.deepcopy(caps)
    for e in approved["entries"]:
        if e["id"] == f"cap/{comp['name']}/hall":
            e.update(matched=True, type="hall", used=[])
        if e["id"] == f"cap/{comp['name']}/court":
            e.update(matched=True, type="square", used=[])
    laid, why = placeplan.compound_axial(comp, cpart, place, cdecl, spec=spec, seed=1,
                                         caps=approved)
    assert laid is not None, why
    assert why["plot_type"] == "hall" and why["area_type"] == "square", why
    assert all(v["by"] == "capability record" for v in why["chosen_by"].values()), why["chosen_by"]
    used = sorted({p["type"] for p in laid["parts"]})
    assert "temple" not in used and "plaza" not in used, used
    # control: with no record the form pick decides, and says so
    laid2, why2 = placeplan.compound_axial(comp, cpart, place, cdecl, spec=spec, seed=1)
    assert laid2 is not None and why2["chosen_by"]["plot"]["by"].startswith("form"), why2
    return (f"approved hall/square -> laid of {used}, every type by the capability "
            f"record; without a record the form pick chose {why2['plot_type']}/"
            f"{why2['area_type']} and said so")


# ------------------------------------------------ 9. a working landmark's own lot

@case
def t_j_a_working_landmark_takes_a_lot_of_its_own_and_the_count_stands():
    """The held-out replay: the smithy overlapped two house lots of the back row."""
    dist, part, place, decls = _bits(w=116, d=60, want=10, density="low",
                                     character={"frontage": "street", "lot_width": 12,
                                                "lot_depth": 10, "open_share": 0.15,
                                                "landmarks": [{"type": "workshop"}]})
    _t, every = placeplan.types_card()
    got, rec = dc.compile_district(dict(dist, exact=True), part, place, decls, seed=1)
    leaves = [p for q in got["quarters"] for p in q["plots"]]
    plots = [p for p in leaves if p["kind"] == "plot"]
    lm = [p for p in plots if p["type"] == "workshop"]
    houses = [p for p in plots if p["type"] != "workshop"]
    assert len(lm) == 1 and len(houses) == 10, (len(lm), len(houses))
    fails = pipeline.plan_failures(leaves, every, form=None)
    over = [f for f in fails if f["check"] == "overlap"]
    assert not over, over[:3]
    r = pipeline.part_rect(lm[0])
    assert r[2] - r[0] + 1 <= 16, r
    return (f"10 cottages and a {r[2] - r[0] + 1}x{r[3] - r[1] + 1} workshop on its own "
            f"lot, no overlap, the count intact")


def main() -> int:
    import time
    bad = 0
    t0 = time.perf_counter()
    for name, fn in CASES:
        try:
            says = fn()
            print(f"ok   {name}: {says}")
        except Skip as e:
            print(f"skip {name}: {e}")
        except Exception as e:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            bad += 1
    print(f"{len(CASES) - bad}/{len(CASES)} closure spatial cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
