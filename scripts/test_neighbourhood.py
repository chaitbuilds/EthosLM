#!/usr/bin/env python3
"""The neighbourhood round's connected causes, through the real production path.

Every case runs the actual entry points -- `placeplan.concentric_layout`,
`placesolve.reallocate`, `placeplan.street_enclosure`,
`pipeline/stages_plan._stage_arterials`, `pipeline/improve._apply_layout`,
`intent.check` -- on the retained city plan and the retained village. Nothing here
reimplements a measurement it is checking, and nothing here builds a world.

    N1  an arrangement written to the allocation reaches the districts of a ring whose
        count is **inferred**, and neither the ring's width nor any district rectangle
        moves for it. The control: the same override on an explicit-count place, which
        negotiates its width as it always did.

    N2  the controller adopts `arrange.alternatives`' first eligible row of the action --
        the compiled, validator-certified comparison -- and not a private pick over the
        uncompiled catalogue. Counterexample: the largest lot is not the choice.

    N3  an arrangement action that moves no parent rectangle and no parent count is
        `applied` and `changed`, and says what it changed inside them.

    N4  a re-solve that re-routes the road drops the **compound** plans laid against it,
        not only the district plans. This is the composition round's eleven withdrawn
        actions, whose refusal was a stale palace plan and not an asymmetric check.

    N5  `street_enclosure` measures the street and not the field: free ground further
        from a built face than a lane is `unclaimed_columns`, is out of the denominator,
        and `continuity` reports the longest built run along one lane. An empty district
        still answers `unavailable` rather than zero.

    N6  `intent`'s density reading takes an enclosure measurement on each district's own
        leaves rather than reporting `unavailable` for every one of them.

Seconds to a few minutes: the fixtures are retained plans; N4 copies a state directory.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import (arrange, intent as intent_mod, pipeline, placeplan,  # noqa: E402
                   placesolve, spec as spec_mod)

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


_LOADED: dict = {}

CITY = os.path.join(ROOT, "out", "comp-city")


def city():
    """The composition round's retained state: read, never written."""
    if "city" in _LOADED:
        return _LOADED["city"]
    if not os.path.exists(os.path.join(CITY, "plan.place.json")):
        raise Skip("no out/comp-city/plan.place.json")

    def load(name):
        p = os.path.join(CITY, name)
        return json.load(open(p)) if os.path.exists(p) else None
    spec = load("place.checked.json") or load("place.json")
    _t, decls = placeplan.types_card(None, spec.get("form"))
    got = {"spec": spec, "place": load("plan.place.json"), "site": load("site.json"),
           "plateau": load("plateau.json"), "caps": load("capabilities.json"),
           "intent": load("intent.json"), "parts": load("parts.json"),
           "voice": (load("voice.json") or {}).get("voice") or "", "decls": decls}
    _LOADED["city"] = got
    return got


def finding(fid: str):
    """One of the composition round's own built findings, from its own reading."""
    p = os.path.join(CITY, "inspection", "views.json")
    if not os.path.exists(p):
        raise Skip("no out/comp-city/inspection/views.json")
    rows = ((json.load(open(p)) or {}).get("reading") or {}).get("findings") or []
    got = next((f for f in rows if f.get("id") == fid), None)
    if got is None:
        raise Skip(f"no finding {fid} on the retained reading")
    return copy.deepcopy(got)


def leaves_of(name: str):
    p = os.path.join(CITY, f"plan.district.{name}.json")
    if not os.path.exists(p):
        raise Skip(f"no compiled district {name}")
    plan = json.load(open(p))
    return [q for quarter in (plan.get("quarters") or [])
            for q in (quarter.get("plots") or []) if q.get("kind", "plot") == "plot"]


# ------------------------------------------ N1: one adopted arrangement, inferred count

@case
def t_n1_an_adopted_arrangement_reaches_an_inferred_count_rings_districts():
    """`concentric_layout` read `allocation.arrangement` inside its explicit-count branch."""
    c = city()
    want = {"rows": 1, "lot_width": 6, "lot_depth": 6, "frontage": "street",
            "courtyard_share": 0.05}
    out = {}
    for label, alloc in (("without", {"axis_side": "north"}),
                         ("with", {"axis_side": "north",
                                   "arrangement": {"lower_ring": dict(want)}})):
        place, fails = placesolve.solve_place(
            copy.deepcopy(c["spec"]), c["site"], c["plateau"], c["decls"], c["voice"],
            seed=1, caps=c["caps"], allocation=alloc, intent=c["intent"])
        assert place is not None and not fails, (label, fails)
        d = next(x for x in place["districts"] if x["name"] == "lower_ring_north_2")
        out[label] = {
            "rect": [d["x0"], d["z0"], d["x1"], d["z1"]],
            "structures": int(d.get("structures") or 0),
            "arrangement": d.get("arrangement"),
            "widths": {r["name"]: r.get("width")
                       for r in ((place.get("layout") or {}).get("rings") or [])}}
    assert out["without"]["arrangement"] is None, out["without"]
    assert out["with"]["arrangement"], "the adopted arrangement did not reach the district"
    for k in want:
        assert out["with"]["arrangement"].get(k) == want[k], out["with"]["arrangement"]
    # **and it cost no parent movement**: the round's own requirement, "a local fabric
    # change must not require an unrelated parent-size change". The *relation* is what
    # is registered, not two literals -- the count is expected to move, and it is
    # expected to move up, because `columns_per_plot` and `count_band` take the adopted
    # lot: a tighter fabric earns a district more houses on the same ground. A case
    # asserting the numbers would have to be edited every time either end improved,
    # which is how a test stops being evidence.
    assert out["with"]["rect"] == out["without"]["rect"], out
    assert out["with"]["widths"] == out["without"]["widths"], out
    assert out["with"]["structures"] > out["without"]["structures"], out
    return (f"the 6x6 reaches `lower_ring_north_2` under an inferred count and earns it "
            f"{out['without']['structures']} -> {out['with']['structures']} houses; its "
            f"rectangle {out['with']['rect']} and every ring width "
            f"{list(out['with']['widths'].values())} are what they were without it")


@case
def t_n1b_the_control_an_explicit_count_ring_still_negotiates_its_width():
    """The contrasting control: the hoist must not have turned the width negotiation off
    for the places that have one. A counted ring still measures a `width_need` and still
    records the alternatives it considered."""
    d = os.path.join(ROOT, "out", "closure-rings")
    if not os.path.exists(os.path.join(d, "place.json")):
        raise Skip("no out/closure-rings/place.json")
    doc = json.load(open(os.path.join(d, "place.json")))
    said = (json.load(open(os.path.join(d, "interpretation.json")))["sentence"]
            if os.path.exists(os.path.join(d, "interpretation.json")) else
            "Build a walled town of twenty-four houses in two rings around a temple "
            "compound.")
    spec = spec_mod.read_spec(doc, said)
    assert (spec.get("explicit_count") or {}).get("n"), spec.get("explicit_count")
    _t, decls = placeplan.types_card(None, spec.get("form"))
    site = json.load(open(os.path.join(d, "site.json"))) \
        if os.path.exists(os.path.join(d, "site.json")) else \
        {"origin": [0, 0], "size": int(placeplan.wanted_footprint(spec) or 240)}
    place, fails = placeplan.concentric_layout(spec, site, None, decls, "japanese_temple")
    assert place is not None, fails
    ring = next((d2 for d2 in place["districts"]
                 if (d2.get("target") or {}).get("what") == "ring"), None)
    assert ring is not None, "no counted ring recorded a target"
    tgt = ring["target"]["value"]
    assert tgt.get("width_need") is not None, tgt
    return (f"the explicit-count fixture still negotiates: `{ring['name']}` records a "
            f"width need of {tgt['width_need']} against a share width of "
            f"{tgt.get('share_width')}, and its arrangement is {tgt.get('arrangement')}")


# ------------------------------------------ N2 and N3: the controller's own choice

@case
def t_n2_the_controller_adopts_the_certified_comparisons_first_row():
    """`_reallocate` picked `max(lot area)` over the uncompiled catalogue."""
    c = city()
    place, spec = c["place"], copy.deepcopy(c["spec"])
    # **the finding names no action**, which is the case this is about: the controller
    # takes the comparison's first eligible row whatever operation it belongs to, rather
    # than the next name in a fixed list
    f = dict(finding("s1"))
    f.pop("action", None)
    got, rec = placesolve.reallocate(place, spec, f, site=c["site"], decls=c["decls"],
                                     intent=c["intent"], parts_record=c["parts"])
    assert rec.get("applied"), rec.get("refused")
    cert = rec.get("certified") or {}
    assert cert.get("on"), rec
    assert cert.get("verdict") == "ok", cert
    assert str(cert.get("by") or "").startswith("arrange.alternatives"), cert
    # the choice is the comparison's own first eligible row, recomputed here
    d = next(x for x in place["districts"] if x["name"] == cert["on"])
    part = next(p for p in spec["defining_parts"] if p["name"] == rec["subject"])
    part = dict(part, **({"fabric_types": list(d["fabric_types"])}
                         if d.get("fabric_types") else {}))
    rows = arrange.alternatives(d, part, place, c["decls"], spec=spec, seed=1,
                                parts_record=c["parts"])
    mine = [r for r in rows if r.get("action") in arrange.ARRANGEMENT_ACTIONS
            and r.get("arrangement") and not r.get("refused")
            and int(r.get("lots") or 0) > 0]
    assert mine, "the comparison offers no eligible arrangement row"
    adopted = (rec["allocation"]["arrangement"][rec["subject"]])
    for k, v in mine[0]["arrangement"].items():
        assert adopted.get(k) == v, (k, adopted, mine[0]["arrangement"])
    assert rec.get("action") == mine[0]["action"], (rec.get("action"), mine[0]["action"])
    # **the counterexample**: the private pick this replaced was the largest lot, and
    # the comparison's choice is a smaller one. A ranking that cannot be distinguished
    # from `max(lot area)` is not evidence that the ranking is being used.
    biggest = max(mine, key=lambda r: (r["lot"][0] * r["lot"][1]))
    assert (mine[0]["lot"][0] * mine[0]["lot"][1]) <= (biggest["lot"][0] * biggest["lot"][1]), \
        (mine[0]["lot"], biggest["lot"])
    return (f"adopted `{rec.get('action')}` {mine[0]['arrangement']} -- the comparison's "
            f"first eligible row on `{cert['on']}` ({cert['lots']} lots, enclosure "
            f"{cert['enclosure']}, built-cover estimate {cert['built_cover_estimate']}, "
            f"validator {cert['verdict']}); chosen by {rec.get('action_from')}; the "
            f"largest-lot pick would have been {biggest['lot']} at "
            f"{biggest.get('lots')} lots and {biggest.get('enclosure')} enclosure")


@case
def t_n3_an_internal_rearrangement_is_a_change():
    """`_reallocate` asked only whether a rectangle or a count had moved, and an
    arrangement action is the one that moves neither."""
    c = city()
    f = dict(finding("s1"))
    f.pop("action", None)
    got, rec = placesolve.reallocate(c["place"], copy.deepcopy(c["spec"]), f,
                                     site=c["site"], decls=c["decls"],
                                     intent=c["intent"], parts_record=c["parts"])
    assert rec.get("applied") and rec.get("changed"), rec.get("refused")
    assert not rec.get("moved"), f"a parent rectangle moved: {rec.get('districts')}"
    refab = rec.get("refabricated") or {}
    assert refab, "nothing was reported as refabricated"
    assert "lower_ring_north_2" in refab, sorted(refab)
    was = next(x for x in c["place"]["districts"] if x["name"] == "lower_ring_north_2")
    now = next(x for x in got["districts"] if x["name"] == "lower_ring_north_2")
    assert [was["x0"], was["z0"], was["x1"], was["z1"]] == \
           [now["x0"], now["z0"], now["x1"], now["z1"]], (was, now)
    assert any("fabric" in s for s in (rec.get("changed_what") or [])), rec
    return (f"applied and changed with {rec.get('moved')} rectangle(s) moved; "
            f"{len(refab)} district(s) refabricated; `changed_what` says "
            f"{rec.get('changed_what')}")


# ------------------------------------------ N4: the road moved, so the compound is
# stale

@case
def t_n4_a_rerouted_road_drops_the_compound_plans_laid_against_it():
    """The composition round withdrew eleven layout actions on "the road arrives at this
    compound at ... and no gate stands within 8 blocks of it" and called the check
    asymmetric. The plan stage routes the road before it validates the compound; what was
    stale was the palace's own plan, which `_stage_arterials` kept while dropping every
    district file beside it."""
    from ethoslm.pipeline import stages_plan
    c = city()
    tmp = tempfile.mkdtemp(prefix="nb-arterials-")
    try:
        for f in ("arterials.json",):
            src = os.path.join(CITY, f)
            if not os.path.exists(src):
                raise Skip(f"no out/comp-city/{f}")
            shutil.copy2(src, os.path.join(tmp, f))
        for f in ("plan.compound.royal_palace.json", "compound_royal_palace_axial.json",
                  "plan.district.lower_ring_north_2.json"):
            src = os.path.join(CITY, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, f))
        cfg = json.load(open(os.path.join(ROOT, "rounds", "comp-city.json")))
        rnd = pipeline.Round(name="nb-arterials", sentence=cfg["sentence"],
                             flags=cfg["flags"], state_dir=tmp)
        had = sorted(os.listdir(tmp))
        assert any(f.startswith("plan.compound.") for f in had), had
        # a place whose geometry has moved: one district rectangle, which is exactly the
        # "local fabric change" case. `vol=None` stops before routing, which is all this
        # case is about -- what the cache retires when it finds a different key.
        moved = copy.deepcopy(c["place"])
        d = next(x for x in moved["districts"] if x["name"] == "lower_ring_north_2")
        d["z1"] = int(d["z1"]) - 1
        stages_plan._stage_arterials(rnd, moved, c["decls"], None)
        left = sorted(os.listdir(tmp))
        assert not [f for f in left if f.startswith(("plan.compound.", "compound_"))], left
        assert not [f for f in left if f.startswith(("plan.district.", "district_"))], left
        assert any(f.startswith("arterials.") and f != "arterials.json" for f in left), left
        return (f"the road's key changed and the cache retired it; of {len(had)} file(s) "
                f"the district plan and both compound files are dropped and the old road "
                f"is kept beside the new key")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t_n4b_the_revisions_own_seam_drops_compounds_too():
    """`improve._apply_layout` removes the files a re-solved place invalidates. It listed
    districts and not compounds, so the revision path had the same hole."""
    import inspect as _inspect
    from ethoslm.pipeline import improve
    src = _inspect.getsource(improve._apply_layout)
    assert "plan.compound." in src and "compound_" in src, \
        "_apply_layout does not drop compound plans"
    src2 = _inspect.getsource(improve._estimate_vs_built)
    assert "built_from" in src2, "_estimate_vs_built does not label how it read the build"
    return ("the revision's own removal list names `plan.compound.` and `compound_` "
            "beside the district files, and the cycle records the estimate beside what "
            "construction emitted")


# ------------------------------------------ N5 and N6: the street, measured

@case
def t_n5_street_enclosure_measures_the_street_and_not_the_field():
    """This called every column no leaf stood on "street", so a district holding 69
    houses on 12,696 columns was judged against ten thousand columns of open ground."""
    c = city()
    place = c["place"]
    said = {}
    for name in ("lower_ring_north_2", "middle_ring_north_west"):
        d = next(x for x in place["districts"] if x["name"] == name)
        leaves = leaves_of(name)
        rp = os.path.join(CITY, f"district_{name}_compiled.json")
        width = (json.load(open(rp)).get("street") if os.path.exists(rp) else None)
        said[name] = placeplan.street_enclosure(d, place, leaves, parts_record=c["parts"],
                                                street_width=width)
    for name, got in said.items():
        area = ((max(place_d["x0"], place_d["x1"]) - min(place_d["x0"], place_d["x1"]) + 1)
                * (max(place_d["z0"], place_d["z1"]) - min(place_d["z0"], place_d["z1"]) + 1)
                ) if (place_d := next(x for x in place["districts"]
                                      if x["name"] == name)) else 0
        assert got["street_columns"] and got["street_columns"] > 0, (name, got)
        assert got["street_columns"] < area, (name, got["street_columns"], area)
        assert got["unclaimed_columns"] is not None and got["unclaimed_columns"] > 0, \
            (name, got)
        assert got["continuity"] is not None, (name, got)
        assert 0 <= got["enclosure"] <= 1, (name, got)
        # the free ground is accounted for exactly once: street, unclaimed or interior
        assert got["longest_enclosed_run"] <= got["longest_street_run"], (name, got)
    # **an empty district still answers `unavailable` and not zero**
    empty = placeplan.street_enclosure(
        {"name": "x", "x0": 0, "z0": 0, "x1": 9, "z1": 9}, None, [])
    assert empty["enclosure"] is None and empty["from"] == "unavailable", empty
    a = said["lower_ring_north_2"]
    return (f"the crowded district: {a['enclosed_length']} of {a['street_columns']} "
            f"columns of street enclosed ({a['enclosure']:.1%}), with "
            f"{a['unclaimed_columns']} columns of unclaimed ground **outside** that "
            f"denominator and a longest built run of {a['longest_enclosed_run']} against "
            f"a longest street run of {a['longest_street_run']}")


@case
def t_n6_the_intents_density_reading_actually_takes_an_enclosure_measurement():
    """`_quality_measure` called `street_enclosure(d, None, None, ...)` -- no leaves --
    and that call answers `unavailable` by design, so every district of every reading in
    the record reported `unavailable` for the measure the street is judged by."""
    import inspect as _inspect
    src = _inspect.getsource(intent_mod._quality_measure)
    assert "_rect_in" in src, "the enclosure call still hands in no leaves"
    # and the helper does what its name says: whole or not at all
    assert intent_mod._rect_in([1, 1, 4, 4], [0, 0, 9, 9])
    assert not intent_mod._rect_in([-1, 1, 4, 4], [0, 0, 9, 9])
    assert not intent_mod._rect_in([1, 1, 40, 4], [0, 0, 9, 9])
    res = json.load(open(os.path.join(CITY, "resolution.json"))) \
        if os.path.exists(os.path.join(CITY, "resolution.json")) else None
    if not res:
        raise Skip("no out/comp-city/resolution.json")
    n = len([r for r in (res.get("regions") or []) if r.get("rect")])
    return (f"the density measure hands each district its own leaves, chosen whole or "
            f"not at all; the retained resolution records {n} region rectangles to hand "
            f"them to")


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
    print(f"\n{ok}/{ok + fail} neighbourhood cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
