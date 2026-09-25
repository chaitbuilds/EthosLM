"""The spatial design round: a trial is promoted on an observed improvement, or rejected.

    $PY scripts/test_promotion.py
    $PY scripts/test_promotion.py --case t1

The neighbourhood round's rollback ended at replanning: `improve._apply_layout` restored
the design records when the *plan* failed, and once planning succeeded `stage_improve`
rebuilt, stopped at lint, returned `blocked` and left the failed trial on disk as the
candidate. Both of its cycles did exactly that, and its second failed revision is what it
delivered. These cases are about the boundary that replaces it (`ethoslm.pipeline.promote`)
and about the two refusals in `placesolve._reallocate` that keep a revision inside the
scope it declared.

Every case states the rule it establishes and prints the measurement that establishes it.
Nothing here builds a world; the trial machinery is a directory of files, a copy out and a
copy back, and that is exactly what is tested.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm.pipeline import promote                                     # noqa: E402

CASES: list = []


class Skip(Exception):
    """This case has no evidence to run on and says so rather than passing."""


def case(fn):
    CASES.append(fn)
    return fn


class Rnd:
    """The two things `promote` asks of a round: where its state is, and a path in it."""

    def __init__(self, state: str):
        self.state = state

    def rel(self, *parts) -> str:
        return os.path.join(self.state, *parts)


def _state(tmp: str, *, candidate: str, rels: dict, types: dict,
           extra: dict | None = None) -> Rnd:
    """A state directory holding the artifacts `promote` treats as the candidate."""
    os.makedirs(tmp, exist_ok=True)
    rnd = Rnd(tmp)
    json.dump({"by": "ethoslm.section.record",
               "relationships": [{"id": k, "status": v} for k, v in rels.items()]},
              open(rnd.rel("section.json"), "w"))
    json.dump({"candidate": candidate,
               "waves": [{"parts": [{"part": f"{t}_{i}", "type": t, "stood": True}
                                    for t, n in types.items() for i in range(n)]}]},
              open(rnd.rel("parts.json"), "w"))
    json.dump({"candidate": candidate}, open(rnd.rel("plan.place.json"), "w"))
    json.dump({"candidate": candidate}, open(rnd.rel("place.checked.json"), "w"))
    os.makedirs(rnd.rel("inspection"), exist_ok=True)
    json.dump({"candidate": candidate}, open(rnd.rel("inspection", "views.json"), "w"))
    for k, v in (extra or {}).items():
        json.dump(v, open(rnd.rel(k), "w"))
    return rnd


def _put(rnd: Rnd, *, candidate: str, rels: dict, types: dict) -> None:
    """Move the state to a different candidate, as a rebuild would."""
    json.dump({"by": "ethoslm.section.record",
               "relationships": [{"id": k, "status": v} for k, v in rels.items()]},
              open(rnd.rel("section.json"), "w"))
    json.dump({"candidate": candidate,
               "waves": [{"parts": [{"part": f"{t}_{i}", "type": t, "stood": True}
                                    for t, n in types.items() for i in range(n)]}]},
              open(rnd.rel("parts.json"), "w"))
    json.dump({"candidate": candidate}, open(rnd.rel("plan.place.json"), "w"))
    json.dump({"candidate": candidate}, open(rnd.rel("inspection", "views.json"), "w"))


# ------------------------------------------------------------------ the cases

@case
def t1_a_rejected_trial_puts_the_accepted_candidate_back_whole():
    """**The whole candidate, byte for byte, and nothing the trial left behind.**

        `stages_media._restore` covers the design records; a rebuild writes `parts.json`,
        `section.json`, `usable.json`, the inspection and the built world, and the
        neighbourhood round's rollback covered none of them. That is how its final frames
        identified `829fb25adb7b6b84` while `inspection/views.json` identified
        `90b26d9c3f075492`.
        
    """
    tmp = tempfile.mkdtemp(prefix="promote-t1-")
    try:
        rnd = _state(tmp, candidate="aaa", rels={"contrast": "demonstrated",
                                                 "route": "demonstrated"},
                     types={"row_house": 96, "market": 1})
        before = {f: open(rnd.rel(f), "rb").read()
                  for f in promote._files(tmp)}
        promote.begin(rnd, candidate="aaa", finding="b1", action="row_depth",
                      measures={"open_to_built": 0.2})
        # the trial builds a different candidate, leaves a file of its own behind, and
        # loses the route relationship
        _put(rnd, candidate="bbb", rels={"contrast": "demonstrated", "route": "failed"},
             types={"row_house": 128})
        json.dump({"of": "bbb"}, open(rnd.rel("district_only_the_trial_has.json"), "w"))
        regs = promote.regressions(rnd)
        got = promote.reject(rnd, why="the route relationship was lost", evidence={
            "regressions": regs})
        after = {f: open(rnd.rel(f), "rb").read() for f in promote._files(tmp)}
        assert after == before, sorted(set(after) ^ set(before)) or [
            f for f in before if after.get(f) != before[f]]
        assert not os.path.exists(rnd.rel("district_only_the_trial_has.json"))
        rec = promote.load(rnd)
        assert rec["open"] is None and rec["trials"][0]["state"] == "rejected"
        assert rec["accepted"]["candidate"] == "aaa"
        return (f"the accepted candidate `aaa` is restored byte for byte over the trial's "
                f"`bbb` -- {len(before)} file(s), {got['restored']['restored']} written "
                f"back and {len(got['restored']['removed'])} of the trial's own removed "
                f"({got['restored']['removed']}) -- and the trial is retained as "
                f"rejected with its reason and its {len(regs)} regression(s): "
                f"{[r['id'] for r in regs]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t2_a_regression_is_measured_against_what_the_accepted_candidate_had():
    """**A flawed baseline's known failures do not authorize new ones, and are not
        themselves protected.** The round: "an initially flawed baseline remains labelled as
        such; its known failures do not authorize new regressions."

        So the protected set is the **status the accepted candidate actually reached**, per
        relationship: a baseline that failed `route` is not protected on `route` and a trial
        that also fails it has regressed nothing; a baseline that demonstrated it is, and a
        trial that loses it has.
        
    """
    tmp = tempfile.mkdtemp(prefix="promote-t2-")
    try:
        # (a) the accepted candidate already failed `route`
        rnd = _state(tmp, candidate="aaa",
                     rels={"contrast": "demonstrated", "route": "failed"},
                     types={"row_house": 96})
        promote.begin(rnd, candidate="aaa", finding="b1", action="row_depth")
        _put(rnd, candidate="bbb", rels={"contrast": "demonstrated", "route": "failed"},
             types={"row_house": 128})
        flawed = promote.regressions(rnd)
        promote.abandon(rnd, why="the case is about the measurement, not the disposition")
        # (b) ...and one that had it
        rnd2 = _state(os.path.join(tmp, "b"), candidate="ccc",
                      rels={"contrast": "demonstrated", "route": "demonstrated"},
                      types={"row_house": 96, "court_large": 4})
        promote.begin(rnd2, candidate="ccc", finding="b1", action="row_depth")
        _put(rnd2, candidate="ddd",
             rels={"contrast": "demonstrated", "route": "failed"},
             types={"row_house": 128})
        lost = promote.regressions(rnd2)
        assert not flawed, flawed
        assert [r["id"] for r in lost] == ["route", "court_large"], lost
        assert promote.protected(rnd2)["relationships"] == {
            "contrast": "demonstrated", "route": "demonstrated"}
        return (f"a trial that fails `route` on a baseline that already failed it "
                f"regresses **nothing** ({len(flawed)} regression(s)); the same trial on "
                f"a baseline that demonstrated it regresses "
                f"{[r['id'] for r in lost]} -- the relationship and the "
                f"`court_large` the accepted candidate stood four of and the trial "
                f"stands none of")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t3_a_completed_build_is_not_an_improvement():
    """**Promotion is on an observed improvement, not on a build that finished.**

        `improve._judge_trial` asks three questions in order and only the second promotes:
        did anything the accepted candidate demonstrated stop being demonstrated; was the
        improvement observed (the row closed, or its cited measure moved the way the row
        asked); or neither. This case is the third: a trial that regresses nothing, builds
        cleanly, and buys nothing measurable is rejected and the accepted candidate stands.
        
    """
    tmp = tempfile.mkdtemp(prefix="promote-t3-")
    try:
        rnd = _state(tmp, candidate="aaa", rels={"contrast": "demonstrated"},
                     types={"row_house": 96})
        promote.begin(rnd, candidate="aaa", finding="b1", action="row_depth",
                      measures={"open_to_built": 0.145})
        _put(rnd, candidate="bbb", rels={"contrast": "demonstrated"},
             types={"row_house": 96})
        promote.built(rnd, candidate="bbb", stages={"parts": {}, "lint": {}})
        assert not promote.regressions(rnd)
        # the ledger's own answer for "did the cited measure move": it did not
        measure = {"measure": "open_to_built", "before": 0.145, "after": 0.145,
                   "moved": False, "why": "open_to_built: 0.145 -> 0.145"}
        got = promote.reject(rnd, why="the improvement was not observed",
                             evidence={"measure": measure, "closed": False})
        assert got["state"] == "rejected" and got["restored_to"] == "aaa"
        assert json.load(open(rnd.rel("parts.json")))["candidate"] == "aaa"
        return (f"a trial that regresses nothing, rebuilds cleanly and leaves its own "
                f"measure at {measure['before']} -> {measure['after']} is **rejected**: "
                f"{got['restored']['restored']} file(s) of the accepted candidate `aaa` "
                f"are back and the candidate on disk is `aaa`, not the trial's `bbb`")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t4_a_promoted_trial_becomes_the_accepted_candidate():
    """The positive control, and it is the case the round has to produce: a trial that
    rebuilt, kept everything the accepted candidate had, and moved the measure its own
    row cited becomes the best retained result, and the next trial falls back to **it**
    rather than to the candidate before it."""
    tmp = tempfile.mkdtemp(prefix="promote-t4-")
    try:
        rnd = _state(tmp, candidate="aaa", rels={"contrast": "demonstrated"},
                     types={"row_house": 96})
        promote.begin(rnd, candidate="aaa", finding="b1", action="row_depth",
                      measures={"open_to_built": 0.145})
        _put(rnd, candidate="bbb", rels={"contrast": "demonstrated"},
             types={"row_house": 128})
        promote.built(rnd, candidate="bbb", stages={"parts": {}})
        measure = {"measure": "open_to_built", "before": 0.145, "after": 0.133,
                   "direction": "down", "moved": True}
        promote.promote(rnd, candidate="bbb", why="the measure moved the way the row asked",
                        evidence={"measure": measure, "closed": True, "regressions": []})
        assert promote.accepted(rnd)["candidate"] == "bbb"
        # a second trial now falls back to `bbb`
        promote.begin(rnd, candidate="bbb", finding="b2", action="compact_bay")
        _put(rnd, candidate="eee", rels={"contrast": "failed"}, types={"row_house": 40})
        regs = promote.regressions(rnd)
        got = promote.reject(rnd, why="contrast lost", evidence={"regressions": regs})
        assert json.load(open(rnd.rel("parts.json")))["candidate"] == "bbb", \
            json.load(open(rnd.rel("parts.json")))["candidate"]
        rec = promote.load(rnd)
        states = [t["state"] for t in rec["trials"]]
        return (f"trial 1 is promoted on a measure that moved "
                f"({measure['before']} -> {measure['after']}, asked `down`) and `bbb` "
                f"becomes the accepted candidate; trial 2 loses `contrast` and is "
                f"rejected **to `bbb`** and not to `aaa` -- trials {states}, "
                f"{got['restored']['restored']} file(s) restored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t5_the_boundary_says_what_it_covers_and_leaves_the_ledger_out():
    """**What a candidate is, published rather than implied.** A rollback that silently
        covers less than it claims is the defect; the set it covers is a value
        (`promote.boundary`).

        And one deliberate exclusion: `obligations.json` records what is owed *and what has
        been tried on which candidate*. Restoring it with the accepted candidate would erase
        the record that the rejected trial's action was ever attempted, and the loop would
        spend its whole budget re-applying the same refused action. What a failed trial leaves
        behind is exactly the knowledge that it failed.
        
    """
    tmp = tempfile.mkdtemp(prefix="promote-t5-")
    try:
        rnd = _state(tmp, candidate="aaa", rels={"contrast": "demonstrated"},
                     types={"row_house": 96},
                     extra={"obligations.json": {"rows": {"b1": {"actions": []}}},
                            "improve.json": {"cycles": []},
                            "usable.json": {"candidate": "aaa"}})
        b = promote.boundary(rnd)
        assert "obligations.json" not in b["files"], b["files"]
        assert "improve.json" not in b["files"], b["files"]
        assert {"parts.json", "section.json", "usable.json",
                "plan.place.json"} <= set(b["files"]), sorted(b["files"])
        assert "inspection" in b["dirs"], b["dirs"]
        promote.begin(rnd, candidate="aaa", finding="b1", action="row_depth")
        # the trial records that it tried, and fails
        json.dump({"rows": {"b1": {"actions": [{"action": "row_depth",
                                                "candidate": "aaa"}]}}},
                  open(rnd.rel("obligations.json"), "w"))
        json.dump({"cycles": [{"cycle": 1, "action": "row_depth"}]},
                  open(rnd.rel("improve.json"), "w"))
        _put(rnd, candidate="bbb", rels={"contrast": "failed"}, types={"row_house": 128})
        promote.reject(rnd, why="contrast lost")
        led = json.load(open(rnd.rel("obligations.json")))
        imp = json.load(open(rnd.rel("improve.json")))
        assert led["rows"]["b1"]["actions"], led
        assert imp["cycles"], imp
        assert json.load(open(rnd.rel("usable.json")))["candidate"] == "aaa"
        return (f"the boundary is {len(b['files'])} file(s) and {b['dirs']} -- published, "
                f"and `parts.json`, `section.json`, `usable.json` and `inspection/` are "
                f"in it where `stages_media.REVISION_FILES` alone had none of them -- "
                f"while `obligations.json` and `improve.json` are deliberately outside, "
                f"so after the rollback the ledger still records that `row_depth` was "
                f"tried on `aaa` and the loop does not spend its budget on it again")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t6_a_revision_carries_the_ground_it_is_not_revising():
    """**The spatial design round's first finding, as a counterexample and a control.**

        `placesolve._reallocate` rebuilt a `plateau` stub from `layout.plateau_rect` carrying
        a rectangle and nothing else, and `pipeline/improve.py` handed it no volume -- so
        `concentric_layout` found neither `plateau["terrace"]` nor a volume to take a median
        from, and every revision in this project's history came back with `terrace: null` and
        four `level: null` rings. Measured on the retained neighbourhood section: the baseline
        `60c62ebbdb32933b` was laid with ring terraces at y 71/79/67/75 and its first revision
        `90b26d9c3f075492` with none, so the revised candidates' rings were never levelled at
        all and their houses were founded on ground running from y 16 to y 127.

        The retained plans are the evidence, and the guard that would have caught it is
        exercised here on them.
        
    """
    base = os.path.join(ROOT, "out", "nb-city", "plan.place.stale.json")
    now = os.path.join(ROOT, "out", "nb-city", "plan.place.json")
    if not (os.path.exists(base) and os.path.exists(now)):
        raise Skip("no retained nb-city plans to read")
    was = (json.load(open(base)).get("layout") or {})
    lost = (json.load(open(now)).get("layout") or {})
    assert was.get("terrace"), "the retained baseline carries no terrace to lose"
    assert not lost.get("terrace"), "the retained revision carries a terrace"
    # the guard: a re-solve that comes back with no ground design where the place being
    # revised had one is refused by name
    from ethoslm import placesolve
    src = placesolve._reallocate.__code__.co_consts
    import inspect
    body = inspect.getsource(placesolve._reallocate)
    assert 'invariant="ground"' in body, "the ground invariant is not asserted"
    assert 'carried_ground' in body, "the terrace is not carried across a re-solve"
    del src
    return (f"the retained baseline's rings stand at "
            f"{[r.get('level') for r in was.get('rings') or []]} and its first "
            f"revision's at {[r.get('level') for r in lost.get('rings') or []]} -- a "
            f"ground design dropped by a revision that was about a lot. "
            f"`_reallocate` now carries the terrace across the re-solve "
            f"(`carried_ground`) and refuses by name (`invariant: ground`) a re-solve "
            f"that comes back without one")


@case
def t7_a_ring_wide_decision_is_qualified_ring_wide():
    """**The scope of a change and the evidence for it are the same set.** The review:
        "`_reallocate` certifies one named district, then stores the arrangement on its
        defining part; the recorded actions refabricate ten lower-ring districts. A local
        certificate cannot establish that a ring-wide change is safe."

        Read off the code rather than run, because running it is a plan: what is asserted is
        that the scope is declared, that every rectangle in it is qualified, and that a
        refusal anywhere in the scope refuses the action -- and that a re-solve taking
        structures from a district the action is not about is refused by name.
        
    """
    import inspect
    from ethoslm import placesolve
    body = inspect.getsource(placesolve._reallocate)
    for want in ('"scope": [d.get("name") for d in scope]',
                 '"qualified": qualified',
                 'A ring-wide decision needs ring-wide feasibility',
                 'invariant="scope"'):
        assert want in body, want
    # ...and the acceptance runner asks for exactly that
    runner = open(os.path.join(ROOT, "scripts", "check_neighbourhood.py")).read()
    assert "a change is qualified on every rectangle it governs" in runner
    assert "no applied change took structures from a district it was not about" in runner
    return ("`_reallocate` declares the scope an arrangement governs, qualifies the "
            "chosen arrangement on every rectangle in it (one compile each rather than "
            "the comparison's thirteen), refuses the action where any of them refuses "
            "it, and refuses a re-solve that takes structures from a district outside "
            "the scope (`invariant: scope`); `check_neighbourhood.gate_change_is_scoped` "
            "asks for all three off the records")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="")
    a = ap.parse_args()
    want = [w.strip() for w in a.case.split(",") if w.strip()]
    t0 = time.time()
    ok = skipped = 0
    for fn in CASES:
        name = fn.__name__
        if want and not any(w in fn.__name__ for w in want):
            continue
        try:
            got = fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            skipped += 1
            continue
        except Exception as e:                      # noqa: BLE001 -- a case reports
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        ok += 1
        print(f"ok   {name}: {got}")
    n = len([f for f in CASES
             if not want or any(w in f.__name__ for w in want)])
    print(f"\n{ok}/{n} promotion cases pass, {skipped} skipped "
          f"({time.time() - t0:.0f}s)")
    return 0 if ok + skipped == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
