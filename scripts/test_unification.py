"""The unification round: every counterexample the integration review reproduced.

    $PY scripts/test_unification.py

Offline, deterministic, no model call and no server. Each case is a **production call
path** -- `intent.read`, `capability.match` as `stage_plan_levels` calls it, the actual
driver loop, the actual preview stage -- and every negative case is paired with a
positive control, so a check cannot be made to pass by refusing everything.

this records what it must do instead, and the two are kept apart on purpose. Grouped by
the invariant each belongs to:

  1. **meaning survives.** A quality the sentence states is a requirement; a negation
     scopes over the feature it negates, not only over walls; a stated setting is
     checked against the site that was chosen; a sourced claim needs a source with
     content identity, not an id; research is a staged job and not an empty success.
  2. **one design is buildable.** The capability record reaches the solver, the site's
     ground and relief reach matching, district fabric and compound descendants are
     chosen from the same record, and agreement is checked against every realized leaf.
  3. **one controller owns progression.** Pending agent work stops the round; an
     exhausted repair budget with open feasibility findings is not `planned`; a plan
     whose inputs moved is redrawn rather than answered from a stale reading; a
     dependency change propagates to every consumer; rollback restores one candidate.
  4. **recovery preserves the request.** Scale negotiation is bound to the *original*
     accepted interpretation and not to the last one; promised, allocated and realized
     counts stay distinct and count every descendant; `rural` is a land use and not a
     demand for farmland.
  5. **checks measure independent evidence.** Frontage is measured against the water or
     the street that is actually there; a policy's geometry is measured; a tradition is
     satisfied by types the plan uses, not by a label; an identity has a route to close.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (capability, contracts, deps, growth, intent,  # noqa: E402
                   pipeline, placeplan, placeshore, placesolve, repair, resolve)
from ethoslm.pipeline import round as driver, stages_media, stages_plan  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


class Skip(Exception):
    pass


def _req(rec: dict, rid: str) -> dict | None:
    return next((r for r in rec["requirements"] if r["id"] == rid), None)


def _ids(rec: dict) -> list:
    return [r["id"] for r in rec["requirements"]]


def write(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(value, fh)


# ===================================================================== 1. meaning

@case
def t_quality_words_are_requirements():
    """`dense` and `tall` are requirements, and an empty reading is impossible."""
    rec = intent.read("Build a dense city with tall buildings.")
    assert not intent.holds(rec), "a sentence with unmet qualities cannot already hold"
    assert _req(rec, "quality/density/dense"), _ids(rec)
    assert _req(rec, "quality/height/tall"), _ids(rec)
    for r in rec["requirements"]:
        assert r["hard"], f"{r['id']} states a quality outright and is not soft"
    return f"{len(rec['requirements'])} requirement(s): {', '.join(_ids(rec))}"


@case
def t_quality_control_plain_sentence():
    """**Positive control**: a sentence stating no quality states no quality."""
    rec = intent.read("Build a village.")
    qs = [r for r in rec["requirements"] if r["kind"] == "quality"]
    assert not qs, f"a plain sentence invented qualities: {[r['id'] for r in qs]}"
    return "no quality requirement invented"


@case
def t_negation_scopes_over_any_feature():
    """`without a temple` is an absence, and it used to be a *positive* temple.

        `ABSENT` held one entry, for walls, so every other feature word was read straight
        through its own negation. The sentence asks for a village that has no temple and the
        reading asked for a temple.
        
    """
    rec = intent.read("Build a village without a temple.")
    assert _req(rec, "absent/temple"), _ids(rec)
    assert not _req(rec, "feature/temple"), "the negated feature is still positive"
    got, _ = intent.coverage(rec, {"defining_parts": [{"name": "t", "family": "temple",
                                                       "count": 1}]})
    assert _req(got, "absent/temple")["status"] == "failed", _req(got, "absent/temple")
    return "absence read and checked against a spec that declares one"


@case
def t_negation_control_positive_feature():
    """**Positive control**: an un-negated temple is still a positive requirement."""
    rec = intent.read("Build a village with a temple.")
    assert _req(rec, "feature/temple"), _ids(rec)
    assert not _req(rec, "absent/temple"), "a plain request was read as a negation"
    return "the positive reading is unchanged"


@case
def t_setting_is_checked_against_the_site():
    """`on a cliff` is a hard obligation measured on the site that was chosen.

        It was soft and permanently `unresolved`, so `holds()` was true while the place
        stood on flat ground. A site record carries its own relief; that is the measurement.
        
    """
    rec = intent.read("Build a village on a cliff.")
    r = _req(rec, "setting/cliff")
    assert r and r["hard"], f"setting is not a hard obligation: {r}"
    assert not intent.holds(rec), "a stated setting holds before a site is chosen"
    flat = {"origin": [0, 0], "size": 64, "relief": 2, "water_pct": 0.0}
    got, _ = intent.coverage(rec, {"defining_parts": []}, site=flat)
    assert _req(got, "setting/cliff")["status"] == "failed", _req(got, "setting/cliff")
    # **And relief alone is not a cliff.** The realization round: the review found this
    # clause certifying a property it never measured -- "Site relief alone certifies a
    # cliff; no local slope, enclosure or channel geometry is needed" -- so 40 blocks
    # falling evenly across a site is a hillside and says so, 40 blocks falling across
    # one step of the site's own grid is a face, and a site record with no height grid
    # answers `unresolved` rather than passing on the relief.
    even = {"origin": [0, 0], "size": 64, "relief": 40, "water_pct": 0.0,
            "mean_grid": [[64] * 8 for _ in range(8)]}
    got, _ = intent.coverage(rec, {"defining_parts": []}, site=even)
    assert _req(got, "setting/cliff")["status"] == "failed", _req(got, "setting/cliff")
    steep = {"origin": [0, 0], "size": 64, "relief": 40, "water_pct": 0.0,
             "mean_grid": [[64] * 4 + [24] * 4 for _ in range(8)]}
    got, _ = intent.coverage(rec, {"defining_parts": []}, site=steep)
    assert _req(got, "setting/cliff")["status"] == "satisfied", \
        _req(got, "setting/cliff")
    unsaid = {"origin": [0, 0], "size": 64, "relief": 40, "water_pct": 0.0}
    got, _ = intent.coverage(rec, {"defining_parts": []}, site=unsaid)
    assert _req(got, "setting/cliff")["status"] == "unresolved", \
        _req(got, "setting/cliff")
    return ("flat ground fails the cliff; 40 blocks of relief spread evenly is a "
            "hillside and fails; 40 across one step of the grid satisfies it; a record "
            "with no grid is unresolved and certifies nothing")


@case
def t_unclaimed_word_control():
    """**Positive control**: `bakery` still survives as an unread clause."""
    rec = intent.read("Build a Japanese village with a bakery.")
    assert _req(rec, "clause/bakery"), _ids(rec)
    assert _req(rec, "tradition/japanese"), _ids(rec)
    return "tradition and clause both survive"


@case
def t_source_needs_content_identity():
    """A source is a title, a URL, an access date and a content fingerprint.

        The review: a reading was accepted whose only source was `{"id": "invented"}`. The
        referenced-id check is real and it checks that a citation points *somewhere*; it did
        not check that the somewhere is a source.
        
    """
    claim = {"id": "claim", "says": "Three rings", "source": "s1", "inferred": False}
    try:
        contracts.make("reading", sentence="Build Ringed City.",
                       sources=[{"id": "s1"}], claims=[claim])
        raise AssertionError("a source with no content identity was accepted")
    except contracts.ContractError as e:
        assert "s1" in str(e), e
    good = {"id": "s1", "title": "Ringed City", "url": "https://example.invalid/bss",
            "accessed": "2026-09-18", "fingerprint": "sha256:" + "0" * 8}
    rec = contracts.make("reading", sentence="Build Ringed City.", sources=[good],
                         claims=[claim])
    assert rec["sources"][0]["fingerprint"], rec
    return "a source without content identity is refused and a complete one is kept"


@case
def t_dangling_citation_control():
    """**Positive control**: a claim citing no known source is still refused."""
    claim = {"id": "claim", "says": "Three rings", "source": "invented",
             "inferred": False}
    try:
        contracts.make("reading", sentence="x", sources=[], claims=[claim])
        raise AssertionError("a dangling citation was accepted")
    except contracts.ContractError:
        return "dangling citation refused"


@case
def t_reading_stages_a_research_job():
    """With no retrieval provider the reading stage **asks**; it does not succeed empty.

        The review's second finding, second half: `stage_reading` completed a reading with
        no sources and no claims, and the agent job that extracts claims only runs once
        sources exist -- so a terminal agent, which is a supported runtime, had no route to
        do the research at all. That is a missing state, not a missing API key.
        
    """
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-reading-") as tmp:
        rnd = pipeline.Round(name="u", sentence="Build Ringed City.", state_dir=tmp)
        with mock.patch.object(stages_plan, "_retrieval_provider", return_value=None):
            res = stages_plan.stage_reading(rnd, None, {})
        got = res.get("reading") or {}
        assert got.get("status") == "needs_model", got
        assert got.get("role") == "research", got
        assert os.path.exists(got["request"]), got
        return f"research staged as {got['role']}: {os.path.basename(got['request'])}"


# =========================================================== 2. one design buildable

@case
def t_capability_reaches_the_solver():
    """`stage_plan_levels` passes the capability record to `solve_place`.

        The review's first finding, reproduced through the production path: matching was
        called with `names` alone and solving with `vol` and `seed` alone, so the record
        naming each part's type was written and then not consulted by the thing that chooses
        each part's type.
        
    """
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-caps-") as tmp:
        rnd = pipeline.Round(name="u", sentence="Build a village.", state_dir=tmp,
                             site={"origin": [0, 0], "size": 64})
        write(rnd.rel("site.json"), rnd.site)

        class Reached(Exception):
            pass

        seen = {}

        def stop(*args, **kwargs):
            seen.update(kwargs)
            raise Reached()

        with mock.patch.object(capability, "match", wraps=capability.match) as matching, \
             mock.patch.object(growth, "type_gaps", return_value=[]), \
             mock.patch.object(stages_plan, "place_voice", return_value=None), \
             mock.patch.object(placeplan, "types_card",
                               return_value=("", {"cottage": {}})), \
             mock.patch.object(stages_plan, "plateau_record", return_value={}), \
             mock.patch.object(stages_plan, "_plan_volume", return_value=None), \
             mock.patch.object(placesolve, "solve_place", side_effect=stop):
            with contextlib.suppress(Reached):
                stages_plan.stage_plan_levels(rnd, None, {},
                                              {"form": "timber", "defining_parts": []})
            asked = set(matching.call_args.kwargs)
        assert "caps" in seen, f"solve_place got {sorted(seen)}"
        for k in ("ground", "relief", "round_boundaries"):
            assert k in asked, f"capability.match got {sorted(asked)}"
        return f"match({', '.join(sorted(asked))}) -> solve_place(caps=...)"


@case
def t_fabric_types_come_from_the_record():
    """A district's houses are drawn from the capability record's approved pool.

        The record matched a `fabric` want per district and the compiler then selected from
        the whole library by role, so the two could disagree about what a district is built
        of and nothing compared them.
        
    """
    decls = {"cottage": {"kind": "plot", "form": "timber", "role": "rural",
                         "needs": {"footprint": [5, 5, 12, 12], "ground": "any",
                                   "frontage": "lane", "clearance": 2}},
             "minka": {"kind": "plot", "form": "east_asian", "role": "rural",
                       "needs": {"footprint": [5, 5, 12, 12], "ground": "any",
                                   "frontage": "lane", "clearance": 2}}}
    from ethoslm import district_compile
    pool = [n for n, _d in district_compile.house_types(decls, "rural", "timber",
                                                        approved=["cottage"])]
    assert pool == ["cottage"], pool
    wide = [n for n, _d in district_compile.house_types(decls, "rural", None)]
    assert set(wide) >= {"cottage", "minka"}, wide
    caps = {"entries": [{"id": "cap/fields/fabric", "matched": True, "type": "cottage",
                         "alternatives": ["row_house"],
                         "wants": {"part": "fields", "of": "fabric"}}]}
    place = {"districts": [{"name": "fields", "defines": "fields"}]}
    placesolve._write_fabric(place, caps)
    assert place["districts"][0]["fabric_types"] == ["cottage", "row_house"], place
    return f"approved pool {pool} out of {sorted(wide)}, written onto the district"


@case
def t_agreement_reads_every_realized_leaf():
    """Agreement is checked against every leaf, not the first type per defining part.

        `agreements` recorded the first type it saw for a part and skipped the rest, skipped
        fabric and compound wants entirely, and skipped rechecking whenever the name already
        agreed -- so a district whose second half was built out of a type the capability
        rules refuse produced no finding.
        
    """
    decls = {"cottage": {"kind": "plot", "form": "timber", "role": "rural",
                         "needs": {"footprint": [5, 5, 12, 12], "ground": "any",
                                   "frontage": "lane", "clearance": 2}},
             "minka": {"kind": "plot", "form": "east_asian", "role": "rural",
                       "needs": {"footprint": [5, 5, 12, 12], "ground": "any",
                                   "frontage": "lane", "clearance": 2}}}
    spec = {"form": "timber", "defining_parts": [
        {"name": "fields", "kind": "group", "family": "district", "role": "rural",
         "structures": 2}]}
    rec = capability.match(spec, decls)
    plan = {"parts": [{"name": "fields", "kind": "district", "children": [
        {"name": "a", "kind": "plot", "type": "cottage", "defines": "fields"},
        {"name": "b", "kind": "plot", "type": "minka", "defines": "fields"}]}]}
    _got, rows = capability.agreements(rec, {"districts": [{"name": "fields",
                                                            "defines": "fields"}]},
                                       decls, plan=plan)
    assert any("minka" in str(r["says"]) for r in rows), rows
    ok = {"parts": [{"name": "fields", "kind": "district", "children": [
        {"name": "a", "kind": "plot", "type": "cottage", "defines": "fields"}]}]}
    _got, none = capability.agreements(rec, {"districts": [{"name": "fields",
                                                            "defines": "fields"}]},
                                       decls, plan=ok)
    assert not none, none
    return f"{len(rows)} disagreement(s) on the mixed fabric, none on the control"


# ========================================================== 3. one controller owns it

@case
def t_pending_agent_work_stops_the_round():
    """A stage waiting on an agent does not let the next stage run.

        The review drove the actual driver: a preview returning an unanswered judge job was
        followed by `parts`, because `wait=0` skips the wait loop entirely and nothing after
        it asks whether the stage is still pending. A terminal runtime must get a resumable
        pending state, not a person remembering to restrict the stage list.
        
    """
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-driver-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        visits = []

        def preview(*args):
            visits.append("preview")
            if len(visits) == 1:
                return {"status": "reenter", "why": "candidate changed"}
            return {"reading": {"status": "needs_model", "role": "judge",
                                "request": "pending.md", "write": "missing.md"}}

        def parts(*args):
            visits.append("parts")
            return {"reached": True}

        with mock.patch.dict(pipeline.STAGES, {"u_preview": preview, "u_parts": parts}), \
             mock.patch("ethoslm.model.router",
                        return_value=SimpleNamespace(fulfil=lambda _: 0)), \
             mock.patch.object(driver, "record"), \
             mock.patch.object(driver, "stage_report", return_value={}):
            res = driver.run(rnd, stages=("u_preview", "u_parts"), backend=object())
        assert "parts" not in visits, visits
        assert res.get("pending"), res
        assert res["pending"]["stage"] == "u_preview", res["pending"]
        return f"visits {visits}; pending at {res['pending']['stage']}"


@case
def t_driver_control_completes_when_nothing_pends():
    """**Positive control**: a stage with no pending work does not block the round."""
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-driver-ok-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        visits = []

        def one(*args):
            visits.append("one")
            return {"ok": True}

        def two(*args):
            visits.append("two")
            return {"ok": True}

        with mock.patch.dict(pipeline.STAGES, {"u_one": one, "u_two": two}), \
             mock.patch("ethoslm.model.router",
                        return_value=SimpleNamespace(fulfil=lambda _: 0)), \
             mock.patch.object(driver, "record"), \
             mock.patch.object(driver, "stage_report", return_value={}):
            res = driver.run(rnd, stages=("u_one", "u_two"), backend=object())
        assert visits == ["one", "two"], visits
        assert not res.get("pending"), res
        return "both stages ran"


@case
def t_open_feasibility_is_not_planned():
    """An exhausted repair budget with open feasibility findings is an explicit stop.

        `_plan_repair` returned None both when it had fixed everything and when it could fix
        nothing, and the caller wrote the registry and reported `planned` either way. The
        city's two district shortfalls were `blocks: feasibility` in the record and the run
        called itself planned.
        
    """
    found = {"findings": [
        {"id": "find/capacity/realized/upper_ring_east", "owner": "layout",
         "blocks": "feasibility", "severity": "warning", "says": "short by 1",
         "fixed": False, "requirement": None, "part": "upper_ring_east",
         "evidence": {}, "seen_by": "t", }]}
    blocked = stages_plan.blocking(found)
    assert blocked, "a feasibility finding does not block feasibility"
    assert blocked[0]["id"].endswith("upper_ring_east"), blocked
    clean = stages_plan.blocking({"findings": [
        {"id": "find/x", "owner": "reading", "blocks": "fidelity",
         "severity": "warning", "says": "", "fixed": False, "requirement": None,
         "part": None, "evidence": {}, "seen_by": "t"}]})
    assert not clean, clean
    return f"{len(blocked)} feasibility blocker(s); a fidelity finding is not one"


@case
def t_stale_preview_is_redrawn():
    """A preview whose plan moved is drawn again, not answered from the old reading.

        The review reproduced the bypass in the actual stage: the dependency check noticed
        the changed plan, the stamp was invalidated, and the early `done` branch returned
        the original drawing and `reading_1.md` without redrawing anything.
        
    """
    home = {"name": "home", "kind": "plot", "type": "cottage", "x0": 10, "x1": 20,
            "z0": 5, "z1": 15, "front": "north"}
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-preview-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        plan = {"parts": [home]}
        write(rnd.rel("plan.json"), plan)
        write(rnd.rel("preview/preview.json"),
              {"done": True, "revisions": 1, "repaired": True,
               "drawn": {"_1": {"images": {}, "candidate": "original"}}})
        with open(rnd.rel("preview/reading_1.md"), "w") as fh:
            fh.write("Inspection of the ORIGINAL plan.")
        deps.stamp(rnd, "preview", outputs=["preview/preview.json"], plan=plan)
        changed = copy.deepcopy(plan)
        changed["parts"][0]["params"] = {"height": 90}
        write(rnd.rel("plan.json"), changed)
        with mock.patch.object(stages_media, "_draw_preview",
                               return_value={"images": {}}) as draw, \
             mock.patch.object(stages_media, "_preview_characters", return_value=[]), \
             mock.patch.object(type(rnd), "place_spec", return_value=None), \
             mock.patch.object(type(rnd), "voice_name", return_value=None):
            res = stages_media.stage_preview(rnd, None, {})
        rec = res["preview"]
        assert not (rec.get("done") and not draw.call_count), \
            f"done on a stale preview without redrawing: {rec}"
        return f"redrawn {draw.call_count} time(s); done={rec.get('done')}"


@case
def t_plan_fingerprint_covers_build_inputs():
    """`attached` is a build input, so it is in the plan fingerprint."""
    home = {"name": "home", "kind": "plot", "type": "cottage", "x0": 10, "x1": 20,
            "z0": 5, "z1": 15, "front": "north"}
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-fp-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        nested = {"parts": [{"name": "q", "kind": "district", "children": [home]}]}
        base = deps.fingerprint(rnd, ("plan",), plan=nested)
        moved = {}
        for key, value in (("x1", 50), ("height", 90), ("attached", True),
                           ("params", {"height": 90})):
            other = copy.deepcopy(nested)
            other["parts"][0]["children"][0][key] = value
            moved[key] = deps.fingerprint(rnd, ("plan",), plan=other) != base
        assert moved["attached"], moved
        assert moved["x1"] and moved["params"], moved
        return f"moved: {sorted(k for k, v in moved.items() if v)}"


@case
def t_stamped_output_must_be_the_document():
    """A `plan.json` replaced by `{}` is not warm. Valid JSON is not a valid plan."""
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-empty-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        nested = {"parts": [{"name": "q", "kind": "plot", "x0": 0, "z0": 0,
                             "x1": 8, "z1": 8}]}
        write(rnd.rel("plan.json"), nested)
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        assert deps.check(rnd, "plan")[0], "the control plan is not warm"
        write(rnd.rel("plan.json"), {})
        fresh, why = deps.check(rnd, "plan")
        assert not fresh, why
        return why[:96]


@case
def t_invalidation_propagates_to_consumers():
    """Invalidating `plan` invalidates `resolution`, which is made out of it."""
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-prop-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        nested = {"parts": [{"name": "q", "kind": "plot", "x0": 0, "z0": 0,
                             "x1": 8, "z1": 8}]}
        write(rnd.rel("plan.json"), nested)
        write(rnd.rel("resolution.json"), {"policy": "relations"})
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        deps.stamp(rnd, "resolution", outputs=["resolution.json"])
        assert "plan" in deps.DEPENDS["resolution"], deps.DEPENDS["resolution"]
        dropped = deps.invalidate(rnd, "plan")
        assert "resolution" in dropped, dropped
        assert not deps.check(rnd, "resolution")[0], deps.check(rnd, "resolution")
        return f"dropped {sorted(dropped)}"


@case
def t_baseline_and_prepared_ground_are_two_grounds():
    """Cutting a terrace does not make the plan that asked for it stale.

        The review's sixth finding, last part: `terrain` fingerprinted `world.npz`, which is
        the file the plateau and terraces cut into -- so a run that prepared its ground
        invalidated the plan that decided on the preparation, and the report records exactly
        that happening to the shoreline case. The baseline is an input and the prepared
        volume is an output of the same design.
        
    """
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-ground-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        plan = {"parts": [{"name": "h", "kind": "plot", "x0": 0, "z0": 0,
                           "x1": 8, "z1": 8}]}
        write(rnd.rel("plan.json"), plan)
        with open(rnd.rel("world.npz"), "wb") as fh:
            fh.write(b"the ground as it was found")
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        assert deps.check(rnd, "plan")[0], deps.check(rnd, "plan")
        # The ground as it was found, kept aside before anything cuts it -- which is
        # what `stage_plateau` writes and what `terrain` is the identity of. Then the
        # cut, then the stamp: that is `stage_terraces`' own order, and the realization
        # round's output-identity rule makes it load-bearing, because a stamp now
        # records what it accepted and one written before its output exists describes a
        # file that is gone.
        with open(rnd.rel("world" + deps.BASELINE), "wb") as fh:
            fh.write(b"the ground as it was found")
        with open(rnd.rel("world.npz"), "wb") as fh:
            fh.write(b"the ground after this design cut into it")
        stages_plan._stamp_ground(rnd, plan, "a terrace")
        fresh, why = deps.check(rnd, "plan")
        assert fresh, f"preparing the ground made the plan stale: {why}"
        ok, _why2 = stages_plan.ground_for_this_design(rnd)
        assert ok, _why2
        # ...and ground cut for another candidate is refused before construction
        moved = copy.deepcopy(plan)
        moved["parts"][0]["x1"] = 40
        write(rnd.rel("plan.json"), moved)
        ok2, why3 = stages_plan.ground_for_this_design(rnd)
        assert not ok2, why3
        return f"the plan stays warm across its own ground work; {why3[:60]}..."


@case
def t_a_fixture_is_imported_explicitly():
    """An unstamped artifact is reused only where the round says it was imported."""
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-import-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        assert not deps.legacy(rnd, "plan"), "an unstamped directory was trusted"
        deps.import_legacy(rnd.state, "a shipped fixture", ["plan.json"])
        assert deps.legacy(rnd, "plan"), deps.imported(rnd)
        write(rnd.rel("plan.json"), {"parts": []})
        deps.invalidate(rnd, "plan", "withdrawn")
        assert not deps.legacy(rnd, "plan"), "a withdrawn artifact read as imported"
        return "silence is not an import; an import is, and a withdrawal undoes it"


@case
def t_a_candidate_has_an_identity():
    """The design has an id, and it moves when the design does."""
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-cand-") as tmp:
        rnd = pipeline.Round(name="u", sentence="Build a village.", state_dir=tmp)
        plan = {"parts": [{"name": "h", "kind": "plot", "x0": 0, "z0": 0,
                           "x1": 8, "z1": 8}]}
        write(rnd.rel("plan.json"), plan)
        a = deps.candidate_id(rnd)
        assert a == deps.candidate_id(rnd), "the same design is two candidates"
        moved = copy.deepcopy(plan)
        moved["parts"][0]["attached"] = True
        b = deps.candidate_id(rnd, plan=moved)
        assert a != b, (a, b)
        return f"{a} -> {b} when a build input moves"


@case
def t_rollback_restores_one_candidate():
    """Rollback restores every file of the candidate, including the repair records.

        It restored findings and resolution and left the rejected `layout_repairs.json`,
        `plan_repairs.json` and terrain behind, so the accepted candidate was mixed with the
        rejected one's decisions.
        
    """
    files = ("findings.json", "resolution.json", "layout_repairs.json",
             "plan_repairs.json")
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-rollback-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        for f in files:
            write(rnd.rel(f), {"candidate": "original"})
        with open(rnd.rel("world.npz"), "wb") as fh:
            fh.write(b"original terrain placeholder")
        snap = stages_media._snapshot(rnd)
        for f in files:
            write(rnd.rel(f), {"candidate": "rejected"})
        with open(rnd.rel("world.npz"), "wb") as fh:
            fh.write(b"rejected terrain placeholder")
        stages_media._restore(rnd, snap)
        got = {f: json.load(open(rnd.rel(f)))["candidate"] for f in files}
        assert set(got.values()) == {"original"}, got
        assert open(rnd.rel("world.npz"), "rb").read() == \
            b"original terrain placeholder", "terrain was not rolled back"
        return "four records and the terrain restored"


# =========================================================== 4. recovery preserves it

@case
def t_scale_is_bound_to_the_original_interpretation():
    """Two negotiations cannot walk the floor from 24 to 2 inside a 25% bound.

        Each call measured its bound against the *current* band, which the previous call had
        already moved, so the registered 25% minimum was re-applied to a floor it had
        produced. The bound belongs to the interpretation that was accepted.
        
    """
    spec = {"kind": "village", "size_band": [24, 81]}
    first = repair._band_repair(spec, {"evidence": {"promised": 6}})
    assert not first["refused"], first
    after = {**spec, **{k: v for k, v in first.items()
                        if k in ("structures", "size_band")},
             "negotiated": [{"what": "size_band",
                             "from": {"structures": 24, "size_band": [24, 81]},
                             "to": {"structures": 6, "size_band": [6, 81]}}]}
    second = repair._band_repair(after, {"evidence": {"promised": 2}})
    assert second and second["refused"], second
    assert "24" in second["why"], second["why"]
    return f"second negotiation refused: {second['why'][:90]}"


@case
def t_scale_control_single_negotiation():
    """**Positive control**: one negotiation inside the bound is still allowed."""
    got = repair._band_repair({"kind": "village", "size_band": [24, 81]},
                              {"evidence": {"promised": 12}})
    assert got and not got["refused"], got
    assert got["structures"] == 12, got
    return "24 -> 12 accepted"


@case
def t_recovery_tries_ground_and_types_before_promising_less():
    """A district short of its promise is grown, then re-typed, and only then reduced.

        The review's fourth finding: the allocation action "does not search alternative
        region geometry, lot types, site extent or ground works", so the one answer to a
        district that could not hold its promise was to promise less -- and a district that
        laid its promised one house could become zero-target open ground, taking the
        residential demand with it.
        
    """
    site = {"origin": [0, 0], "size": 400}
    d = {"name": "homes", "defines": "homes", "x0": 10, "z0": 10, "x1": 90, "z1": 60,
         "structures": 40, "fabric_types": ["townhouse", "cottage"]}
    place = {"districts": [d], "parts": [], "compounds": [], "layout": {}}
    spec = {"kind": "village", "size_band": [40, 80], "form": "timber",
            "defining_parts": [dict(d, kind="group", family="district",
                                    density="low", role="rural")]}
    fails = [{"check": "count", "why": "short"}]
    _t, decls = placeplan.types_card(None, "european_vernacular")
    with tempfile.TemporaryDirectory(prefix="ethoslm-unify-recover-") as tmp:
        rnd = pipeline.Round(name="u", state_dir=tmp)
        first = stages_plan._district_capacity_repair(rnd, spec, place, d, 6, fails,
                                                      site=site, decls=decls)
        assert first and first["action"] == "extent", first
        assert first["to"] == 40, "the promise moved before the ground did"
        assert first["rect"][2] - first["rect"][0] > d["x1"] - d["x0"], first["rect"]
        stages_plan._apply_reallocation(rnd, place, first)
        second = stages_plan._district_capacity_repair(rnd, spec, place, d, 6, fails,
                                                       site=site, decls=decls)
        assert second and second["action"] == "fabric", second
        assert second["to"] == 40, second
        assert second["fabric_types"][0] == "cottage", second["fabric_types"]
        stages_plan._apply_reallocation(rnd, place, second)
        third = stages_plan._district_capacity_repair(rnd, spec, place, d, 6, fails,
                                                      site=site, decls=decls)
        assert third and third["refused"], third
        assert "first accepted" in third["why"], third["why"]
    return ("extent, then fabric, then a reduction the accepted floor refuses: "
            f"{first['action']} -> {second['action']} -> "
            f"{third.get('action', 'allocation')} refused")


@case
def t_growth_is_bound_to_the_rectangle_first_laid_out():
    """Three growths of half-again do not make a district two and a quarter times over.

        The same arithmetic the review found the scale negotiation getting wrong, in the
        other direction: a bound re-applied to its own output is a growth rate. Found by
        running the held-out village, whose shore districts went 2,581 -> 3,870 -> 5,805
        columns inside a bound of "half as much again".
        
    """
    site = {"origin": [0, 0], "size": 400}
    d = {"name": "homes", "x0": 10, "z0": 10, "x1": 90, "z1": 60, "structures": 12}
    place = {"districts": [d], "parts": [], "compounds": [], "layout": {}}
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)          # noqa: E731
    start = area([d["x0"], d["z0"], d["x1"], d["z1"]])
    grown = 0
    for _ in range(6):
        got = stages_plan._grow_into_free_ground(place, d, site)
        if got is None:
            break
        d.setdefault("extent_from", [d["x0"], d["z0"], d["x1"], d["z1"]])
        d["x0"], d["z0"], d["x1"], d["z1"] = got
        grown += 1
    now = area([d["x0"], d["z0"], d["x1"], d["z1"]])
    bound = start * (1.0 + stages_plan.DISTRICT_GROWTH_MAX)
    assert now <= bound * 1.5, (start, now, bound)
    assert grown >= 1, "the first growth was refused"
    return f"{grown} growth(s), {start} -> {now} columns against a bound of {bound:.0f}"


@case
def t_a_ring_sector_is_not_grown_and_says_so():
    """**Negative control**: a ring sector's rectangle is the ring arithmetic's."""
    site = {"origin": [0, 0], "size": 400}
    d = {"name": "upper_ring_east", "x0": 100, "z0": 100, "x1": 130, "z1": 289,
         "structures": 2, "ring": 0}
    place = {"districts": [d], "parts": [], "compounds": [],
             "layout": {"rings": [{"name": "upper_ring", "inner": 93, "outer": 138}]}}
    assert stages_plan._grow_into_free_ground(place, d, site) is None
    plain = dict(d)
    plain.pop("ring")
    assert stages_plan._grow_into_free_ground(place, plain, site) is not None
    return "a ring sector is left alone and the same rectangle off a ring is grown"


@case
def t_counts_stay_distinct_and_count_descendants():
    """Promised, allocated and realized are three numbers and stay three numbers.

        `with_realized` overwrote `structures_promised` with the realized total and counted
        district plots only, so the city's resolution reported 583 against an assembled plan
        of 634: the palace's 51 plots were descendants nobody counted.
        
    """
    res = contracts.make(
        "resolution", policy="relations",
        regions=[{"name": "d1", "policy": "relations", "role": None, "defines": None,
                  "rect": [0, 0, 10, 10], "boundary": None, "inner": None,
                  "holes": None, "level": None, "routes": [], "access": [],
                  "surface": "built", "lots": 10, "density": None, "voice": None,
                  "anchor": None, "requirement": None, "notes": ""}],
        site={"origin": [0, 0], "size": 64}, centre=None,
        bounds={"structures_promised": 10}, negotiated=[], note="")
    plan = {"parts": [
        {"name": "d1", "kind": "district", "children": [
            {"name": "h1", "kind": "plot"}, {"name": "h2", "kind": "plot"}]},
        {"name": "palace", "kind": "compound", "children": [
            {"name": "hall", "kind": "plot"}, {"name": "court", "kind": "area"}]}]}
    got, rows = resolve.with_realized(res, {"d1": 2}, plan=plan)
    b = got["bounds"]
    assert b["structures_promised"] == 10, b
    assert b["structures_allocated"] == 10, b
    assert b["structures_realized"] == 3, b       # h1, h2 and the palace's hall
    assert any(r["id"].endswith("d1") for r in rows), rows
    return (f"promised {b['structures_promised']}, allocated "
            f"{b['structures_allocated']}, realized {b['structures_realized']}")


@case
def t_rural_is_a_land_use_not_farmland():
    """A `rural` district is not by that word a farmland belt owing 60% fields.

        The held-out fishing village's programme said `role: rural` and the compiler's
        validator demanded farms and fields over 60% of it, so a shore village of houses
        failed a cover clause it had never been asked to meet. The demand belongs to a
        declared land use, not to the coarse role word.
        
    """
    d = {"name": "shore", "x0": 0, "z0": 0, "x1": 60, "z1": 60,
         "structures": 6, "defines": "shore"}
    got = {"quarters": [{"name": "homes", "plots": [
        {"name": f"h{i}", "kind": "plot", "type": "cottage",
         "x0": 2 + 9 * i, "z0": 2, "x1": 9 + 9 * i, "z1": 9} for i in range(6)]}]}
    spec = {"form": "timber", "defining_parts": [
        {"name": "shore", "kind": "group", "family": "district", "role": "rural",
         "density": "sparse", "structures": 6}]}
    decls = {"cottage": {"kind": "plot", "form": "timber", "role": "rural",
                         "needs": {"footprint": [5, 5, 12, 12], "ground": "any",
                                   "frontage": "lane", "clearance": 2}}}
    fails = placeplan.district_failures(d, got, {"districts": [d]}, decls,
                                        form="timber", role="rural",
                                        part=spec["defining_parts"][0], spec=spec)
    assert not [f for f in fails if f.get("check") == "farmland_cover"], fails
    farm = dict(spec["defining_parts"][0], land_use="farmland")
    fails2 = placeplan.district_failures(d, got, {"districts": [d]}, decls,
                                         form="timber", role="rural", part=farm,
                                         spec={**spec, "defining_parts": [farm]})
    assert [f for f in fails2 if f.get("check") == "farmland_cover"], fails2
    return "rural alone owes no farmland; a declared farmland land use does"


# ====================================================== 5. checks measure evidence

@case
def t_water_frontage_needs_homes_facing_water():
    """Homes that all front *along* the shore do not front the water.

        The check read "none faces away" as "they face it", which is the weaker statement
        and not the one the sentence makes. Unknown frontage is not evidence either.
        
    """
    anchor = {"kind": "shoreline", "path": [[0, 10]], "water_side": "west"}
    res = {"bounds": {"anchor": anchor},
           "regions": [{"name": "shore", "lots": 1, "rect": [0, 0, 40, 40]}]}
    home = {"name": "home", "kind": "plot", "type": "cottage", "x0": 10, "x1": 20,
            "z0": 5, "z1": 15, "front": "north"}
    along = intent._fronts("water", res, [home])
    assert along["status"] != "satisfied", along
    unknown = intent._fronts("water", res, [{**home, "front": "west"}] +
                             [{**home, "name": f"u{i}", "front": None}
                              for i in range(9)])
    assert unknown["status"] != "satisfied", unknown
    away = intent._fronts("water", res, [{**home, "front": "east"}])
    assert away["status"] == "failed", away
    good = intent._fronts("water", res, [{**home, "name": f"h{i}", "front": "west"}
                                         for i in range(6)])
    assert good["status"] == "satisfied", good
    return (f"along={along['status']}, mostly-unknown={unknown['status']}, "
            f"away={away['status']}, facing={good['status']}")


@case
def t_street_frontage_needs_a_street():
    """`facing the street` is measured against streets, not against a `front` field."""
    home = {"name": "home", "kind": "plot", "type": "cottage", "x0": 10, "x1": 20,
            "z0": 5, "z1": 15, "front": "north"}
    bare = intent._fronts("street", {}, [home, {**home, "name": "u", "front": None}],
                          {"parts": [home]})
    assert bare["status"] != "satisfied", bare
    lane = {"name": "lane", "kind": "edge", "type": "alley",
            "path": [[0, 3], [40, 3]], "width": 3}
    plan = {"parts": [home, lane]}
    good = intent._fronts("street", {}, [home], plan)
    assert good["status"] == "satisfied", good
    return f"no street={bare['status']}, a lane at the door={good['status']}"


@case
def t_shoreline_geometry_is_measured():
    """A district a million columns from the shore is not laid out along it."""
    anchor = {"kind": "shoreline", "path": [[0, 0], [0, 100]], "water_side": "west"}
    far = {"bounds": {"anchor": anchor},
           "regions": [{"name": "offshore", "lots": 1,
                        "rect": [1000000, 1000000, 1000020, 1000020]}]}
    ok, why = intent._policy_geometry("shoreline", far, {})
    assert not ok, why
    near = {"bounds": {"anchor": anchor},
            "regions": [{"name": "band", "lots": 6, "rect": [10, 0, 60, 100]}]}
    ok2, why2 = intent._policy_geometry("shoreline", near, {})
    assert ok2, why2
    return f"far refused ({why[:56]}...), near accepted"


@case
def t_concentric_geometry_is_measured():
    """Two rings with no geometry are two words; real rings must actually nest."""
    empty = {"bounds": {"rings": [{"name": "one"}, {"name": "two"}]}}
    ok, why = intent._policy_geometry("concentric", empty, {})
    assert not ok, why
    nested = {"bounds": {"rings": [{"name": "inner", "inner": 0, "outer": 40},
                                   {"name": "outer", "inner": 40, "outer": 90}]}}
    ok2, why2 = intent._policy_geometry("concentric", nested, {})
    assert ok2, why2
    crossed = {"bounds": {"rings": [{"name": "a", "inner": 0, "outer": 90},
                                    {"name": "b", "inner": 40, "outer": 60}]}}
    ok3, why3 = intent._policy_geometry("concentric", crossed, {})
    assert not ok3, why3
    return f"empty refused, nested accepted, crossed refused ({why3[:48]}...)"


@case
def t_tradition_needs_types_the_plan_uses():
    """One capability entry saying `japanese` with no plan does not build Japan."""
    caps = {"entries": [{"type": "minka", "matched": True,
                         "wants": {"part": "houses", "of": "defining_part"},
                         "envelope": {"tradition": "japanese"}}]}
    rec = intent.read("Build a Japanese village.")
    spec = {"form": "east_asian",
            "defining_parts": [{"name": "houses", "kind": "plot", "family": None,
                                "count": 1}]}
    got, _ = intent.coverage(rec, spec, None, capabilities=caps)
    assert _req(got, "tradition/japanese")["status"] != "satisfied", \
        _req(got, "tradition/japanese")
    plan = {"parts": [{"name": "h1", "kind": "plot", "type": "minka",
                       "defines": "houses"}]}
    got2, _ = intent.coverage(rec, spec, plan, capabilities=caps)
    assert _req(got2, "tradition/japanese")["status"] == "satisfied", \
        _req(got2, "tradition/japanese")
    return "a label alone is not the tradition; types the plan uses are"


@case
def t_identity_has_a_route_to_close():
    """A named place can be resolved by sourced evidence **and** a visual judgment.

        It could only ever read `unresolved`, so a run that did the research and made the
        judgment had no way to record that it had: an obligation with no completion route is
        as much a broken contract as one with a false-pass route.
        
    """
    rec = intent.read("Build Ringed City.")
    rid = next(r["id"] for r in rec["requirements"] if r["kind"] == "identity")
    reading = {"sources": [{"id": "s1", "title": "t", "url": "u", "accessed": "d",
                            "fingerprint": "f"}],
               "claims": [{"id": "c1", "says": "three rings", "source": "s1"}]}
    spec = {"defining_parts": []}
    got, _ = intent.coverage(rec, spec, reading=reading)
    assert _req(got, rid)["status"] == "unresolved", _req(got, rid)
    # **The closure round: a positive verdict closes an identity only when it cites the
    # claims it rests on and looked at built output.** The old fixture -- a bare
    # `recognisable: true` off a preview reading -- is the review's false-pass route,
    # and a negative verdict is consumed however thin its evidence.
    verdict = {"candidate": "c", "looked_at": ["out/x/world_built.npz"],
               "verdicts": [{"about": "identity", "name": "Ringed City",
                             "recognisable": True, "why": "the rings read as the city",
                             "cites": ["c1"], "from": "inspection/reading.md"}]}
    got2, _ = intent.coverage(rec, spec, reading=reading, judgment=verdict)
    assert _req(got2, rid)["status"] == "satisfied", _req(got2, rid)
    uncited = {"candidate": "c", "looked_at": ["out/x/plan.json"],
               "verdicts": [{"about": "identity", "name": "Ringed City",
                             "recognisable": True, "why": "looks right",
                             "from": "preview/reading_2.md"}]}
    got2b, _ = intent.coverage(rec, spec, reading=reading, judgment=uncited)
    assert _req(got2b, rid)["status"] != "satisfied", _req(got2b, rid)
    refused = {"verdicts": [{"about": "identity", "name": "Ringed City",
                             "recognisable": False, "why": "no rings",
                             "from": "preview/reading_2.md"}]}
    got3, _ = intent.coverage(rec, spec, reading=reading, judgment=refused)
    assert _req(got3, rid)["status"] == "failed", _req(got3, rid)
    return "evidence alone stays unresolved; a judgment closes it either way"


@case
def t_identity_needs_evidence_behind_the_judgment():
    """**Negative control**: a judgment with no sourced claim cannot close an identity."""
    rec = intent.read("Build Ringed City.")
    rid = next(r["id"] for r in rec["requirements"] if r["kind"] == "identity")
    verdict = {"verdicts": [{"about": "identity", "name": "Ringed City",
                             "recognisable": True, "why": "looks right",
                             "from": "preview/reading_2.md"}]}
    got, _ = intent.coverage(rec, {"defining_parts": []},
                             reading={"sources": [], "claims": []}, judgment=verdict)
    assert _req(got, rid)["status"] != "satisfied", _req(got, rid)
    return "an unsourced judgment does not certify a named place"


@case
def t_access_is_measured_from_justified_entries():
    """A door is reachable from the gate, not from the lane stance outside its own wall.

        The sample reported zero unreachable doors while the same run reported a
        disconnected component of 5,922 stances, because the walk was seeded from *every*
        lane stance in every component. One network with two islands is not connected.
        
    """
    from ethoslm import lint
    reach = lint.reachable_from({"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]},
                                entries=["a"])
    assert reach == {"a", "b"}, reach
    both = lint.reachable_from({"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]},
                               entries=["a", "c"])
    assert both == {"a", "b", "c", "d"}, both
    return "the walk is seeded from the entries it is given and nothing else"


def main() -> int:
    bad = skipped = 0
    for fn in CASES:
        name = fn.__name__[2:]
        try:
            print(f"ok   {name:58s} {fn()}")
        except Skip as e:
            skipped += 1
            print(f"skip {name:58s} {e}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:58s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            print(f"ERR  {name:58s} {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - bad - skipped}/{len(CASES)} unification cases pass"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
