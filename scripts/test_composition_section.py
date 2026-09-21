"""The composition round's false-pass routes, through the real consumers.

    $PY scripts/test_composition_section.py

Small inputs, seconds not minutes, each with a positive control. Not a suite. The
coordinator's four, in the order of the acceptance record:

**The section is the registered extent** (`pipeline.stages_build.section_parts`):

  1. a plot that straddles the section boundary is **cut out and named**, never half
     built; a plot wholly inside is taken. The old sampler's rule -- keep an edge leaf
     whenever its *bounding* rectangle meets the sample -- bought the whole ring wall
     from one segment, so: an edge is **clipped** to the section, and a run too short to
     be a wall is dropped;
  2. a section whose cut falls next to a gate **refuses itself** rather than cutting
     through the passage and calling the broken access a sampling artifact;
  3. `intent.sample_scope` counts the plots the section actually included, so a partly
     taken quarter does not put leaves nobody attempted into the scope a clause about
     standing leaves is measured over.

**A relationship is measured, not asserted** (`ethoslm.section`):

  4. two fabrics with the same grain on the same side of the boundary do **not**
     demonstrate contrast, and the honest answer to "no boundary to measure across" is
     `unmeasured`; the positive control -- different grain, different spacing, opposite
     sides -- demonstrates it;
  5. an anchor that stood but whose predicate answered `declared` does not demonstrate
     its use; the same anchor with an `observed` answer does;
  6. a required feature with no affirmative evidence is **owed**, and owed is not passed.

**An undecided predicate is not a predicate that held** (`ethoslm.intent`):

  7. `_usable_verdict` counts a `declared`/`None` answer as owed, not as held, and
     `_function_measure` answers `unresolved` rather than `satisfied` with the words
     "1 final-world predicate(s) on them hold";
  8. a density word that passes on allocated lot cover **fails** when the emitted mass
     fills less of those lots than any type in this library builds; the same cover with
     real buildings on it passes.

**The action inventory is one list** (`pipeline.improve`):

  9. `move_object` is not offered to the `layout` owner, because that path refuses it;
     the arrangement actions the spatial layer implements are offered;
 10. a requirement's word reaches the token the constraint is about (`market` ->
     `stalls`), so a required feature's obligation is material.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import intent, section                                       # noqa: E402
from ethoslm.pipeline import improve                                      # noqa: E402
from ethoslm.pipeline.stages_build import (SECTION_GATE_CLEAR,            # noqa: E402
                                         SECTION_MIN_RUN, section_parts)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ fixtures

SECTION = {"id": "t", "rect": [0, 0, 100, 100],
           "sides": {"crowded": "north_", "calm": "south_"},
           "demonstrate": ["contrast", "anchor", "courts", "route", "features"]}


def _plot(name, x0, z0, x1, z1, *, type_="row_house", quarter="north_1_row_0"):
    return {"kind": "plot", "name": name, "type": type_, "x0": x0, "z0": z0,
            "x1": x1, "z1": z1, "in": ["place", quarter]}


def _wall(name, path, width=3):
    return {"kind": "edge", "name": name, "type": "great_wall", "path": path,
            "width": width, "params": {"height": 20}}


def _gate(name, at):
    return {"kind": "point", "name": name, "type": "ring_gate", "at": at, "size": 5}


# ------------------------------------------------------------------ 1-3, selection

@case
def t_a_leaf_is_taken_whole_or_cut_out():
    parts = [_plot("north_1_row_0_a", 10, 10, 20, 20),
             _plot("north_1_row_0_b", 95, 10, 115, 20),      # straddles x1 = 100
             _plot("far_away", 300, 300, 310, 310)]
    got, rec = section_parts(parts, SECTION)
    names = {p["name"] for p in got}
    assert names == {"north_1_row_0_a"}, names
    cut = {c["part"] for c in rec["cut_out"]}
    assert cut == {"north_1_row_0_b"}, cut
    assert "far_away" not in cut, "a leaf nowhere near the section is not a cut"
    return (f"1 of 3 leaves taken, the straddler named in cut_out, the distant leaf "
            f"neither taken nor claimed as a cut")


@case
def t_a_boundary_is_clipped_not_bought_entire():
    # a ring wall four times the section's width, of which one run crosses it
    path = [[-500, 50], [500, 50]]
    parts = [_wall("ring", path), _plot("north_1_row_0_a", 10, 10, 20, 20)]
    got, rec = section_parts(parts, SECTION)
    runs = rec["boundary_runs"]
    assert len(runs) == 1, runs
    assert runs[0]["columns"] == 101, runs[0]["columns"]
    assert runs[0]["of"] == "ring"
    edge = [p for p in got if p.get("kind") == "edge"][0]
    assert edge["path"] == [[0, 50], [100, 50]], edge["path"]
    assert edge["name"].endswith("@t") and edge["section_clip"]["of"] == "ring"
    # ...and a run too short to be a wall is not a part
    short = section_parts([_wall("stub", [[99, 50], [140, 50]])], SECTION)[1]
    assert not short["boundary_runs"], short["boundary_runs"]
    return (f"1001 columns of wall clipped to 101 inside the section; a "
            f"{SECTION_MIN_RUN}-column floor drops a 2-column stub")


@case
def t_a_cut_next_to_a_gate_refuses_the_section():
    # the gate sits 6 columns from where the section cuts the wall
    parts = [_wall("ring", [[-500, 50], [500, 50]]), _gate("ring_gate", [94, 50]),
             _plot("north_1_row_0_a", 10, 10, 20, 20)]
    _got, rec = section_parts(parts, SECTION)
    assert rec.get("refused"), "a cut beside a gate must refuse the section"
    assert "ring_gate" in rec["refused"] and "gate" in rec["refused"]
    # the control: the same gate in the middle of the run is fine
    ok = section_parts([_wall("ring", [[-500, 50], [500, 50]]), _gate("ring_gate", [50, 50]),
                        _plot("north_1_row_0_a", 10, 10, 20, 20)], SECTION)[1]
    assert not ok.get("refused"), ok.get("refused")
    return (f"a cut {SECTION_GATE_CLEAR - 10} columns from a gate refuses by name; the "
            f"same gate 50 columns in is built")


@case
def t_the_scope_is_the_plots_actually_included():
    parts = [_plot("north_1_row_0_a", 10, 10, 20, 20),
             _plot("north_1_row_0_b", 95, 10, 115, 20)]      # same quarter, cut out
    _got, rec = section_parts(parts, SECTION)
    pr = {"sample": rec, "waves": [{"parts": [{"part": "north_1_row_0_a"}]}]}
    scope = intent.sample_scope(pr, parts)
    assert "north_1_row_0_a" in scope
    assert "north_1_row_0_b" not in scope, ("a leaf the section cut out is not in the "
                                            "scope a clause is measured over")
    # the control: a record with no `included_plots` (the old whole-quarter sampler)
    old = {"quarters": ["north_1_row_0"], "rect": rec["rect"], "margin": 0,
           "joining_parts": []}
    both = intent.sample_scope({"sample": old, "waves": []}, parts)
    assert both == {"north_1_row_0_a", "north_1_row_0_b"}, both
    return "the section's scope is 1 plot; the whole-quarter sampler's is still 2"


# ------------------------------------------------------------------ 4-6, the record

def _built(rows, *, sample):
    return {"candidate": "c0", "waves": [{"wave": "w", "parts": rows}], "sample": sample}


def _row(name, type_, rect, *, stood=True, features=None, method="inferred"):
    return {"part": name, "type": type_, "kind": "area" if type_ in
            ("market", "square", "plaza") else "plot", "stood": stood,
            "emitted": {"footprint": rect, "storeys": 1,
                        "features": features or {},
                        "features_method": {k: method for k in (features or {})}}}


def _state(tmp, parts_record, *, usable=None, circulation=None, network=None, plan=None):
    os.makedirs(tmp, exist_ok=True)
    json.dump(parts_record, open(os.path.join(tmp, "parts.json"), "w"))
    json.dump(plan or {"districts": [], "parts": []},
              open(os.path.join(tmp, "plan.json"), "w"))
    if usable is not None:
        json.dump(usable, open(os.path.join(tmp, "usable.json"), "w"))
    if circulation is not None:
        json.dump(circulation, open(os.path.join(tmp, "circulation.json"), "w"))
    if network is not None:
        json.dump(network, open(os.path.join(tmp, "network.json"), "w"))
    return tmp


@case
def t_contrast_is_measured_on_built_geometry():
    import tempfile
    sample = {"section": "t", "rect": [0, 0, 100, 100], "registered": SECTION,
              "question": SECTION["demonstrate"], "quarters": [], "margin": 0,
              "included_plots": [], "joining_parts": [],
              "boundary_runs": [{"of": "ring", "part": "ring@t", "columns": 101,
                                 "path": [[0, 50], [100, 50]]}]}
    plan = {"districts": [
        {"kind": "district", "name": "north_1", "x0": 0, "z0": 0, "x1": 100, "z1": 48},
        {"kind": "district", "name": "south_1", "x0": 0, "z0": 52, "x1": 100,
         "z1": 100}], "parts": []}
    # the negative: the same grain on both sides, 8 a side
    same = []
    for i in range(8):
        same.append(_row(f"north_1_h{i}", "row_house", [2 + i * 12, 5, 9 + i * 12, 12]))
        same.append(_row(f"south_1_h{i}", "row_house", [2 + i * 12, 60, 9 + i * 12, 67]))
    with tempfile.TemporaryDirectory() as d:
        rec = section.record(_state(d, _built(same, sample=sample), plan=plan),
                            registered=SECTION)
    flat = [r for r in rec["relationships"] if r["id"] == "contrast"][0]
    assert flat["status"] == "failed", flat
    # the positive: attached 6x8 rows one side, detached 13x13 courts the other
    diff = []
    for i in range(8):
        diff.append(_row(f"north_1_h{i}", "row_house",
                         [2 + i * 6, 5, 7 + i * 6, 12]))         # touching, small
    for i in range(8):
        diff.append(_row(f"south_1_h{i}", "courtyard_house",
                         [2 + i * 20, 60, 14 + i * 20, 72]))     # apart, large
    with tempfile.TemporaryDirectory() as d:
        rec2 = section.record(_state(d, _built(diff, sample=sample), plan=plan),
                             registered=SECTION)
    good = [r for r in rec2["relationships"] if r["id"] == "contrast"][0]
    assert good["status"] == "demonstrated", good
    held = good["measured"]["held"]
    assert len(held) >= section.CONTRAST_MEASURES, held
    assert good["measured"]["across_boundary"] is True
    # ...and with no boundary to measure across, the answer is `unmeasured`
    with tempfile.TemporaryDirectory() as d:
        rec3 = section.record(
            _state(d, _built(diff, sample={**sample, "boundary_runs": []}), plan=plan),
            registered=SECTION)
    none = [r for r in rec3["relationships"] if r["id"] == "contrast"][0]
    assert none["status"] == "unmeasured", none
    return (f"same grain -> failed; {', '.join(held)} -> demonstrated; no boundary -> "
            f"unmeasured")


@case
def t_an_anchor_needs_an_observed_answer():
    import tempfile
    sample = {"section": "t", "rect": [0, 0, 100, 100], "registered": SECTION,
              "quarters": [], "margin": 0, "joining_parts": [], "boundary_runs": []}
    rows = [_row("south_1_landmark_market", "market", [40, 60, 56, 76],
                 features={"stalls": True, "aisle": True})]
    declared = {"checks": [{"part": "south_1_landmark_market",
                            "want": "equipment_reachable", "holds": None,
                            "method": "declared",
                            "why": "claims stalls and published no rectangle"}]}
    with tempfile.TemporaryDirectory() as d:
        rec = section.record(_state(d, _built(rows, sample=sample), usable=declared),
                            registered=SECTION)
    a = [r for r in rec["relationships"] if r["id"] == "anchor"][0]
    assert a["status"] == "failed", a
    observed = {"checks": [{"part": "south_1_landmark_market",
                            "want": "equipment_reachable", "holds": True,
                            "method": "observed", "why": "14 of 14 stall columns stand"}]}
    with tempfile.TemporaryDirectory() as d:
        rec2 = section.record(_state(d, _built(rows, sample=sample), usable=observed),
                             registered=SECTION)
    b = [r for r in rec2["relationships"] if r["id"] == "anchor"][0]
    assert b["status"] == "demonstrated", b
    return "a `declared` answer fails the anchor; an `observed` one demonstrates it"


@case
def t_a_required_feature_with_no_evidence_is_owed():
    import tempfile
    sample = {"section": "t", "rect": [0, 0, 100, 100], "registered": SECTION,
              "quarters": [], "margin": 0, "joining_parts": [], "boundary_runs": []}
    plan = {"districts": [{"kind": "district", "name": "south_1", "x0": 0, "z0": 52,
                           "x1": 100, "z1": 100,
                           "demand": {"required": ["courtyard"]}}], "parts": []}
    short = [_row("south_1_h0", "courtyard_house", [2, 60, 14, 72],
                 features={"courtyard": False})]
    with tempfile.TemporaryDirectory() as d:
        rec = section.record(_state(d, _built(short, sample=sample), plan=plan),
                            registered=SECTION)
    f = [r for r in rec["relationships"] if r["id"] == "features"][0]
    assert f["status"] == "failed", f
    assert f["measured"]["owed"] and f["measured"]["owed"][0]["feature"] == "courtyard"
    ok = [_row("south_1_h0", "courtyard_house", [2, 60, 14, 72],
               features={"courtyard": True})]
    with tempfile.TemporaryDirectory() as d:
        rec2 = section.record(_state(d, _built(ok, sample=sample), plan=plan),
                             registered=SECTION)
    g = [r for r in rec2["relationships"] if r["id"] == "features"][0]
    assert g["status"] == "demonstrated", g
    return "a required courtyard that construction did not emit is owed, not passed"


# ------------------------------------------------------------------ 7-8, intent

@case
def t_an_undecided_predicate_is_owed_not_held():
    parts = [{"name": "m", "type": "market"}]
    rec = {"waves": [{"parts": [{"part": "m", "emitted": {"usable": {
        "equipment_reachable": {"holds": None, "method": "declared",
                                "why": "claims stalls, published no rectangle"},
        "entrance_connected": {"holds": None, "method": "unsupported",
                               "why": "no door leaf"}}}}]}]}
    got = intent._usable_verdict(parts, {"market"}, rec)
    assert got["held"] == 0, got
    assert len(got["owed"]) == 1 and got["owed"][0][2] == "declared", got
    assert got["unsupported"] == 1, got
    assert got["ran"] == 0, ("`ran` must count only answers that decided something", got)
    caps = {"entries": [{"type": "market", "matched": True,
                         "wants": {"function": "market"}}]}
    status, why, ev = intent._function_measure({"function": "market"}, parts, caps, rec)
    assert status == "unresolved", (status, why)
    assert "undecided" in why, why
    # the control: an observed answer satisfies it and says how many held
    rec2 = {"waves": [{"parts": [{"part": "m", "emitted": {"usable": {
        "equipment_reachable": {"holds": True, "method": "observed", "why": "14 stand"}}}}]}]}
    s2, w2, _e2 = intent._function_measure({"function": "market"}, parts, caps, rec2)
    assert s2 == "satisfied" and "1 final-world predicate(s) on them hold" in w2, (s2, w2)
    del ev
    return ("a declared answer is owed and the function is unresolved; an observed one "
            "satisfies it")


@case
def t_density_is_not_bought_with_empty_lots():
    # 4 lots of 30x30 on 10,000 developable columns: 36% cover, `dense` by the band
    plots, rows = [], []
    for i in range(4):
        x0 = 2 + i * 32
        plots.append({"kind": "plot", "name": f"d_h{i}", "type": "row_house",
                      "x0": x0, "z0": 2, "x1": x0 + 29, "z1": 31,
                      "params": {"storeys": 1}, "in": ["place", "d"]})
        rows.append({"part": f"d_h{i}", "type": "row_house", "kind": "plot",
                     "stood": True,
                     "emitted": {"footprint": [x0, 2, x0 + 9, 11], "storeys": 1}})
    res = {"regions": [{"name": "d", "part": "d", "rect": [0, 0, 99, 99], "lots": 4,
                        "scope_columns": 10000, "developable_columns": 10000,
                        "allocated_columns": 3600, "built_columns": 400}]}
    dens = intent.density_target("dense")
    assert dens["lo"] == 0.30, dens
    st, why, ev = intent._quality_measure(
        "density", "dense", 0.30, plots, res, {}, None,
        {"waves": [{"parts": rows}]})
    assert ev["coverage"] >= 0.30, ev
    assert st == "failed", (st, why)
    assert ev["lot_fill"] is not None and ev["lot_fill"] < intent.MIN_LOT_FILL, ev
    assert "allocation and not fabric" in why, why
    # the control: the same lots and the same cover, with buildings that fill them.
    # `built_columns` is the resolution record's, which `placeplan.region_columns` fills
    # from what construction emitted -- so this is the same number production measures,
    # moved by building rather than by allocating.
    full = [{**r, "emitted": {"footprint": [p["x0"], p["z0"], p["x0"] + 24,
                                            p["z0"] + 24], "storeys": 1}}
            for r, p in zip(rows, plots)]
    res2 = {"regions": [{**res["regions"][0], "built_columns": 2500}]}
    st2, why2, ev2 = intent._quality_measure(
        "density", "dense", 0.30, plots, res2, {}, None,
        {"waves": [{"parts": full}]})
    assert st2 == "satisfied", (st2, why2, ev2)
    return (f"36% lot cover at {ev['lot_fill']:.0%} fill fails under the "
            f"{intent.MIN_LOT_FILL:.0%} floor; the same cover at {ev2['lot_fill']:.0%} "
            f"passes")


# ------------------------------------------------------------------ 9-10, the inventory

@case
def t_the_offered_actions_are_the_executable_ones():
    from ethoslm import placesolve
    got = improve.owner_actions("layout")
    assert "move_object" not in got, got
    assert "move_object" in placesolve.REALLOCATE_ACTIONS, (
        "the point of the check is that placesolve still knows the action and this "
        "stage does not offer it")
    arr = set(getattr(placesolve, "ARRANGEMENT_ACTIONS", ()))
    assert arr and arr <= set(got), (arr, got)
    assert all(a in placesolve.REALLOCATE_ACTIONS or a in ("character", "voice")
               for a in got), got
    return (f"layout is offered {len(got)} action(s), all executable, including the "
            f"{len(arr)} arrangement action(s); move_object is the relation repair's")


@case
def t_a_requirement_word_reaches_the_constraint_token():
    from ethoslm import obligation
    toks = improve._tokens_of("market")
    assert "stalls" in toks, toks
    required = improve._required_tokens({"requirements": [
        {"kind": "feature", "status": "supported", "wants": {"feature": "market"}}]})
    assert "stalls" in required, required
    row = obligation.from_constraint(
        {"what": "stalls", "where": "m", "requested": 4, "got": 0},
        required, part="m", source="parts.json")
    assert row["material"] is True, row
    # the control: a feature nobody asked for is the type's own business
    other = obligation.from_constraint(
        {"what": "chimney", "where": "m", "requested": 1, "got": 0},
        required, part="m", source="parts.json")
    assert other["material"] is False, other
    return "`feature/market` makes a `stalls` constraint material; a chimney stays the "\
           "type's own"


def main() -> int:
    bad = 0
    t0 = time.perf_counter()
    for fn in CASES:
        name = fn.__name__.split("_", 1)[1].replace("_", " ")
        try:
            note = fn()
            print(f"ok   {name}: {note}")
        except Exception as e:                       # noqa: BLE001 -- a failing case
            import traceback
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} composition section cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
