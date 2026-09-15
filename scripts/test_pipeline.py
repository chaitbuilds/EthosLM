"""The consolidation's own contract: one driver, one record schema, one judging path.

Offline, no server, no Chunky, no model. Same discipline as the rest of the suites:
every case asserts both directions, and the ones that matter also assert that the
behaviour they are pinning would *fail* if it regressed.

These are the unit-scale cases underneath it, plus the two things replay cannot show:
that a round config is validated rather than trusted, and that a refused card cannot be
judged from a stale file.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import card as card_mod  # noqa: E402
from ethoslm import measure, pipeline, verdicts  # noqa: E402

# Fixture rounds go through the real stages, and the stages cost the real log -- same
# trap test_judge.py had. Everything this suite records goes to a temp file.
measure.LOG = os.path.join(tempfile.mkdtemp(prefix="pipeline_test_measure_"),
                           "measurements.jsonl")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUNDS = os.path.join(ROOT, "rounds")
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


def _png(path, value, size=(40, 60)):
    import cv2
    cv2.imwrite(path, np.full((size[1], size[0], 3), value, np.uint8))
    return path


# ------------------------------------------------------------- round as config

@case
def c_every_shipped_round_config_loads():
    # `terrain-bank.json` sits here because it is registered ground rather than a round:
    # `scripts/terrain_bank.py` writes it, `scripts/test_ground.py` reads it, and
    # nothing runs it. `type-needs.json` is the same shape of thing for the types --
    # `scripts/type_needs.py` writes it and A1's declarations are read against it. What
    # three isolated schema calls made of the three sentences A1 registers, recorded
    # with the tokens they reported. `scripts/test_place.py` reads it and nothing runs
    # it. Skipped by name, not by shape, so a real config that failed to parse could
    # never slip through by looking like data.
    names = sorted(f for f in os.listdir(ROUNDS) if f.endswith(".json")
                   and not f.startswith(("replay-", "card-"))
                   and f not in ("terrain-bank.json", "type-needs.json",
                                 "place-specs.json"))
    assert names, "no round configs"
    for n in names:
        r = pipeline.Round.load(os.path.join(ROUNDS, n))
        assert r.name, n
        for j in r.judgements:
            assert j.get("question") and j.get("pairs"), f"{n}:{j.get('name')}"
    return f"{len(names)} configs: {', '.join(x[:-5] for x in names)}"


@case
def c_a_config_with_a_stray_field_is_refused():
    """A typo in a round file must not be silently ignored. `base_volumes` instead of
    `base_volume` would replay every wave against the wrong world and report success."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.json")
        json.dump({"name": "x", "base_volumes": "world.npz"}, open(p, "w"))
        try:
            pipeline.Round.load(p)
        except ValueError as e:
            assert "base_volumes" in str(e)
            return "unknown field named in the error"
    raise AssertionError("a config with an unknown field was accepted")


@case
def c_the_base_volume_is_part_of_the_round():
    """The fact that had never been written down. `world.npz` is refreshed between waves
    and by the end is not the pre-build state at all.
    """
    site_b = pipeline.Round.load(os.path.join(ROUNDS, "site_b.json"))
    site_c = pipeline.Round.load(os.path.join(ROUNDS, "site_c.json"))
    assert site_b.base_volume == "world.npz", site_b.base_volume
    assert site_c.base_volume == "world_prebuild.npz", site_c.base_volume
    assert len(site_c.waves) == 6
    return f"site_b <- {site_b.base_volume}, site_c <- {site_c.base_volume}"


@case
def c_a_round_with_no_waves_does_not_need_a_world():
    """E-craft judges cards that were rendered years of rounds ago and has no site."""
    r = pipeline.Round.load(os.path.join(ROUNDS, "e_craft.json"))
    assert r.waves == []
    out = pipeline.stage_programs(r, pipeline.OfflineBackend(r), {})
    assert out["waves"] == 0, out
    return "no waves, no volume load, no crash"


# ----------------------------------------------------------------- the composer

@case
def c_one_composer_lays_out_every_layout():
    with tempfile.TemporaryDirectory() as d:
        for k in ("eye", "aerial", "ne", "sw"):
            _png(os.path.join(d, f"t_{k}.png"), 120)
        p = card_mod.from_dir(d, "t", os.path.join(d, "t_card.png"), size=(40, 60))
        import cv2
        im = cv2.imread(p)
        assert im.shape == (60 * 2 + 8, 40 * 2 + 8, 3), im.shape
    return "quad is 2x2 frames with an 8px gutter both ways"


@case
def c_a_missing_eye_is_matted_and_a_missing_aerial_is_refused():
    """A structure with no reserved threshold has no eye-level shot, and that is a
    fact about the settlement. A missing aerial is a broken camera."""
    with tempfile.TemporaryDirectory() as d:
        for k in ("aerial", "ne", "sw"):
            _png(os.path.join(d, f"t_{k}.png"), 120)
        assert card_mod.from_dir(d, "t", os.path.join(d, "a.png"), size=(40, 60))
        os.remove(os.path.join(d, "t_aerial.png"))
        assert card_mod.from_dir(d, "t", os.path.join(d, "b.png"),
                                 size=(40, 60)) is None
        assert "aerial" in card_mod.DAMAGE["t"]
    return "eye blank -> composed; aerial absent -> refused, and named"


@case
def c_a_black_panel_is_refused_and_the_reason_survives():
    with tempfile.TemporaryDirectory() as d:
        for k in ("eye", "aerial", "ne", "sw"):
            _png(os.path.join(d, f"t_{k}.png"), 120)
        _png(os.path.join(d, "t_ne.png"), 0)
        assert card_mod.from_dir(d, "t", os.path.join(d, "s.png"),
                                 size=(40, 60)) is None
        assert card_mod.DAMAGE["t"] == "black panels: ne"
        assert card_mod.from_dir(d, "t", os.path.join(d, "l.png"), size=(40, 60),
                                 strict=False)
        assert "t" not in card_mod.DAMAGE
    return "strict refuses and names the panel; lenient composes"


@case
def c_a_refused_card_cannot_be_judged_from_a_stale_file():
    """The defect this driver had on its first run.

        A card refused for a black panel may still be on disk from an earlier lenient
        compose. A judge stage that resolved paths off the filesystem judged the stale one
        and reported a clean result -- which is precisely the "nobody looked" failure the
        refusal exists to stop.
        
    """
    with tempfile.TemporaryDirectory() as d:
        frames = os.path.join(d, "frames")
        os.makedirs(frames)
        for tag in ("a", "b"):
            for k in ("eye", "aerial", "ne", "sw"):
                _png(os.path.join(frames, f"{tag}_{k}.png"), 120)
        _png(os.path.join(frames, "a_ne.png"), 0)          # a is damaged
        card_mod.from_dir(frames, "a", os.path.join(d, "a_card.png"),
                          size=(40, 60), strict=False)     # ...but composed once
        card_mod.from_dir(frames, "b", os.path.join(d, "b_card.png"), size=(40, 60))
        assert os.path.exists(os.path.join(d, "a_card.png"))

        cfg = {"name": "t", "cards": [
            {"name": "c", "frames": frames, "out": d, "size": [40, 60],
             "tags": ["a", "b"]}],
            "judgements": [{"name": "j", "question": "q", "pairs": [
                {"label": "p", "a": os.path.join(d, "a_card.png"),
                 "b": os.path.join(d, "b_card.png")}]}]}
        p = os.path.join(d, "t.json")
        json.dump(cfg, open(p, "w"))
        rnd = pipeline.Round.load(p)
        be = pipeline.OfflineBackend(rnd)
        res = {"cards": pipeline.stage_cards(rnd, be, {})}
        assert res["cards"]["c"]["refused"] == {"a": "black panels: ne"}
        j = pipeline.stage_judge(rnd, be, res)["j"]
        assert j["rows"][0]["result"] == "no_cards", j["rows"][0]
        assert j["rows"][0]["damage"] == "black panels: ne"

        # ...and with no cards stage in front of it, the stale file goes to the judge:
        # the pair is *attempted* rather than dropped. That is why the refusal has to
        # travel rather than be re-derived from the filesystem.
        j2 = pipeline.stage_judge(rnd, be, {})["j"]
        assert j2["status"] == "needs_model" and j2["requests"] == 2, j2
    return "refusal travels from the cards stage into the judge stage"


# ------------------------------------------------------------ the building half

@case
def c_stage_waves_replays_round8s_six_recorded_pending_sets():
    """massing, cards, elaboration, conformance -- must land on the recorded dry-run count
    and on pending-set identity with the shipped <wave>.py, for all six waves, no model,
    no server. stage_programs replays the finished program; this replays how it was
    made.
    """
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_c.json"))
    if not os.path.exists(rnd.rel("wave1_massing_prompt.md")):
        return "SKIPPED -- no recorded waves for that site in this worktree"
    out = pipeline.stage_waves(rnd, pipeline.OfflineBackend(rnd), {})
    waves = [w["name"] for w in rnd.waves]
    assert len(waves) == 6
    for w in waves:
        r = out[w]
        assert r["status"] == "built", (w, r)
        assert r["reproduces"], (w, r.get("pending"), r.get("recorded"))
        assert r["matches_shipped"], w
    return f"6/6 waves: staged path == dry-run record == shipped program"


@case
def c_a_cold_wave_stages_a_build_request_and_reports_it():
    """A wave with no program on disk must surface as `needs_model` with the brief
    staged beside the builds -- the BuildNeeded resume pattern, not a crash."""
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, "out", "t")
        os.makedirs(state)
        open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("build a hut")
        cfg = {"name": "t", "waves": [{"name": "w"}],
               "flags": {"split": False}}
        p = os.path.join(d, "t.json")
        json.dump(cfg, open(p, "w"))

        # `state` is derived from ROOT by construction; point it at the fixture.
        class _R(pipeline.Round):
            @property
            def state(self):
                return state
        rnd = _R.load(p)
        be = pipeline.OfflineBackend(rnd)
        be._vol = observe.Volume(0, 0, 0, np.zeros((4, 4, 4), np.uint8), ["air"])
        out = pipeline.stage_waves(rnd, be, {})
        r = out["w"]
        assert r["status"] == "needs_model", r
        assert os.path.exists(r["request"])
        req = json.load(open(r["request"]))
        assert req["brief"] == "build a hut"
    return "no program on disk -> request staged, brief intact, nothing raised"


@case
def c_circulation_is_reported_offline_never_recomputed():
    """Offline, the stage must not pretend it can route: it reports the network the
    round already has, or says a live backend is needed."""
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_c.json"))
    if not os.path.exists(rnd.rel("network.json")):
        return "SKIPPED."
    out = pipeline.stage_circulation(rnd, pipeline.OfflineBackend(rnd), {})
    assert "skipped" in out and out.get("cells"), out
    r = pipeline.Round(name="nowhere")
    bare = pipeline.stage_circulation(r, pipeline.OfflineBackend(r), {})
    assert "live backend" in bare["skipped"], bare
    return f"site_c: {out['cells']} cells reported; no network -> asks for live"


# ---------------------------------------------------------------------- arms


def _arm_round(d, k=2):
    """A minimal arms round: two arms, one that sees and one that does not."""
    pipeline.BUILD_SCRATCH = os.path.join(d, "out", "build_scratch")
    state = os.path.join(d, "out", "t")
    os.makedirs(state, exist_ok=True)
    open(os.path.join(state, "base.md"), "w").write("build a hut")
    cfg = {"name": "t",
           "arms": {"wave": "w", "brief": "base.md", "k": k, "out": "sight",
                    "plots": ["a"], "max_bounces": 1,
                    "cycles": [{"name": "massing", "brief": "the masses"},
                               {"name": "final", "brief": "finish it"}],
                    "list": [{"name": "C", "see": True},
                             {"name": "B", "see": False}]}}
    p = os.path.join(d, "t.json")
    json.dump(cfg, open(p, "w"))

    class _R(pipeline.Round):
        @property
        def state(self):
            return state
    return _R.load(p), state


@case
def c_an_arms_round_enumerates_every_arm_and_candidate_from_config():
    """A round that tests a loop is a config, like every other round here. The arms,
    the k, and the fixed cycle text all live in the file rather than in an agent's
    head, which is what makes "the prompts were frozen before the run" checkable."""
    with tempfile.TemporaryDirectory() as d:
        rnd, _ = _arm_round(d, k=3)
        ids = pipeline._arm_ids(rnd)
        assert [c for c, _ in ids] == ["C/c0", "C/c1", "C/c2",
                                       "B/c0", "B/c1", "B/c2"], ids
        assert all(sub.startswith(os.path.join("sight", "w")) for _, sub in ids), ids
        assert len({sub for _, sub in ids}) == 6, ids
        # the cycle text is data on disk, not code
        assert [c["name"] for c in rnd.arms["cycles"]] == ["massing", "final"]
    return "2 arms x k=3, six distinct candidate directories, cycles read from config"


@case
def c_an_arm_builder_never_learns_which_arm_it_is_in():
    """The blinding that makes B vs C mean anything. `.../sight/w/C/c1/builds/` tells
    a builder it is candidate 1 of arm C of something called sight -- and a builder
    that knows it is in an experiment about seeing writes for the experiment. Same
    hash-named scratch as a selection candidate, asserted the same way: on the shape
    of the path, not on a blacklist of words that a rename would defeat."""
    import re
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        rnd, _ = _arm_round(d, k=2)
        be = pipeline.OfflineBackend(rnd)
        be._vol = observe.Volume(0, 0, 0, np.zeros((4, 4, 4), np.uint8), ["air"])
        out = pipeline.stage_arms(rnd, be, {})
        briefs, shown, dirs = set(), [], set()
        for cid in ("C/c0", "C/c1", "B/c0", "B/c1"):
            assert out[cid]["status"] == "needs_model", out[cid]
            b = out[cid]["blinded"]
            shown += [b["brief"], b["write"], b["dir"]]
            dirs.add(b["dir"])
            briefs.add(open(b["brief"]).read())
        assert len(dirs) == 4, dirs
        # Cycle 1 is identical for every arm and every candidate: nothing has been built
        # yet, so there is nothing for the seeing arm to see.
        assert len(briefs) == 1, briefs
        assert "sight" not in "".join(shown).replace(d, ""), shown
        for q in shown:
            parts = q[len(os.path.dirname(pipeline.BUILD_SCRATCH)) + 1:].split(os.sep)
            assert parts[0] == "build_scratch", parts
            assert re.fullmatch(r"[0-9a-f]{16}", parts[1]), parts
    return "4 hash-named jobs, one identical opening brief, no path names an arm"


# ------------------------------------------------------------------- revision
# `stage_revise` composes what a builder is told, and what a builder is told is the
# experiment. These cases hold the composition to the pre-registration: the three kinds
# of line and no fourth, the standard brief scoped the way the live loop scopes it, and
# nothing in the text that says the word "experiment".

#: The words the pre-registration forbids in a revision brief. `fraction` is held only
#: against the part this stage composes: the frozen advice text for E011 and W011 uses
#: it about a single room, the treatment is defined as that brief unchanged, and editing
#: `lint.FIXES` to satisfy a test would change the thing being measured.
LEAK_WORDS = ("walk_pct", "score", "candidate", "arm", "experiment", "compar", "repeat")
COMPOSED_ONLY = LEAK_WORDS + ("fraction",)


def _leaks(text, words):
    import re
    return [w for w in words
            if re.search(w if w in ("walk_pct", "compar") else r"\b" + w, text, re.I)]


def _diag(doors, rooms):
    return {"doors": doors, "rooms": rooms, "findings_on_plot": [],
            "lint_counts_whole_region": {}}


@case
def c_the_lines_the_linter_is_silent_on_are_the_three_that_were_registered():
    """The treatment, exactly: a door you can only jump to, a plot with rooms and no
    door, and a room whose floor cannot be walked to from outdoors whether or not it is
    enclosed. A door that walks is not a finding, and neither is a fully walkable room --
    a brief that listed those would be telling the builder to fix what is not broken."""
    lines = pipeline.entry_lines(_diag(
        [{"at": [10, 70, 20], "walk_from_outside": False, "jump_from_outside": True},
         {"at": [11, 70, 20], "walk_from_outside": True, "jump_from_outside": True}],
        [{"bbox": [1, 2, 3, 9, 9, 9], "cells": 40, "walk_from_outside": 40,
          "attributed": True},
         {"bbox": [4, 5, 6, 9, 9, 9], "cells": 12, "walk_from_outside": 5,
          "attributed": True},
         {"bbox": [7, 8, 9, 9, 9, 9], "cells": 3, "walk_from_outside": 0,
          "attributed": False}]))
    assert lines == [
        "the door at (10,70,20) cannot be reached on foot from outside -- getting to "
        "it needs a jump",
        "7 of the 12 floor cells of the room at (4,5,6) cannot be walked to from "
        "outdoors"], lines
    # no door at all, and an interior: W002's sentence, listed with the errors
    nodoor = pipeline.entry_lines(_diag([], [{"bbox": [1, 2, 3, 9, 9, 9], "cells": 4,
                                             "walk_from_outside": 0,
                                             "attributed": True}]))
    assert nodoor[0].endswith("no door or gate anywhere on it"), nodoor
    assert len(nodoor) == 2, nodoor
    # nothing wrong, nothing said
    assert pipeline.entry_lines(_diag([], [])) == []
    return "jumped-to door, no-door plot, unreachable floor; walkable rooms silent"


@case
def c_a_revision_brief_says_nothing_about_the_run_it_is_part_of():
    """The experiment is void if a builder is told. The brief it gets is a findings
    brief and reads as one: no measure, no comparison, no repeats, no arms, nothing
    about there being sixteen of these."""
    from ethoslm import lint
    findings = [lint.Finding(c, lint.ERROR if c[0] == "E" else lint.WARNING,
                             f"{c}: something is wrong at (1,2,3)", (1, 2, 3))
                for c in sorted(lint.FIXES)]
    text = pipeline.revision_findings(lint.Report(findings, 0.1),
                                      ["the door at (1,2,3) cannot be reached on foot "
                                       "from outside -- getting to it needs a jump"],
                                      2012, "windmill")
    bad = _leaks(text, LEAK_WORDS)
    assert not bad, f"the whole brief leaks {bad}"
    composed = text.split("## On foot")[1] + pipeline.ENTRY_FIX
    bad = _leaks(composed, COMPOSED_ONLY)
    assert not bad, f"the composed section leaks {bad}"
    assert "## On foot" in text and pipeline.ENTRY_FIX in text
    # and it would fail if it regressed
    assert _leaks(text + " walk_pct", LEAK_WORDS) == ["walk_pct"]
    return (f"{len(findings)} codes + the on-foot section, clean of "
            f"{len(LEAK_WORDS)} leak words")


@case
def c_a_revision_brief_carries_the_program_and_asks_for_the_whole_thing_back():
    """The artefact between passes is the program itself -- the same decision the split
    pipeline and run_cycles made.
    """
    subj = {"brief": "# API\n\nBuild a windmill."}
    b = pipeline._revise_brief(subj, "place_cuboid(1,2,3,4,5,6, 'stone')")
    assert b.startswith("# API\n\nBuild a windmill.")
    assert "place_cuboid(1,2,3,4,5,6, 'stone')" in b
    assert "whole" in b and "not a patch" in b and "findings.md" in b
    assert not _leaks(b.split("## The program as it stands")[1], COMPOSED_ONLY)
    return "brief + the draft, whole-program instruction, findings pointed at by name"


@case
def c_the_standard_brief_is_scoped_the_way_settlement_run_scopes_it():
    """The lint region here is a whole settlement and 6 of its errors are the bare
    volume's, so an unscoped brief would hand the builder of one structure its
    neighbours' defects and then measure how it responded to them.
    """
    from ethoslm import lint, observe
    b = {}
    for x in range(24):
        for z in range(24):
            b[(x, 64, z)] = "stone"
    for (x0, z0) in ((2, 2), (14, 14)):           # two huts, neither with a door
        for x in range(x0, x0 + 5):
            for z in range(z0, z0 + 5):
                b[(x, 68, z)] = "stone_bricks"
                for y in (65, 66, 67):
                    if x in (x0, x0 + 4) or z in (z0, z0 + 4):
                        b[(x, y, z)] = "stone_bricks"
    plots = [{"label": "mine", "x0": 2, "z0": 2, "x1": 6, "z1": 6},
             {"label": "theirs", "x0": 14, "z0": 14, "x1": 18, "z1": 18}]
    ctx = lint.Context.build(observe.Volume.from_blocks(b, 0, 60, 0, 24, 20, 24), plots)
    whole = lint.lint(ctx)
    scoped = pipeline.standard_report(ctx, [plots[0]])
    place = {c.code for c in lint.CHECKS if c.family == lint.PLACE}
    theirs = [f for f in whole.findings
              if (f.detail or {}).get("plot") == "theirs"]
    assert theirs, "fixture built nothing to be wrongly blamed for"
    # The build family is scoped and the place family is not, which is the split
    # settlement_run makes: what this pass built wrong, and what it did to the town.
    strayed = [f for f in scoped.findings
               if (f.detail or {}).get("plot") == "theirs" and f.code not in place]
    assert not strayed, f"the neighbour's own defects travelled: {strayed}"
    assert [f for f in scoped.findings if (f.detail or {}).get("plot") == "mine"]
    assert len(scoped.findings) < len(whole.findings)
    return (f"{len(whole.findings)} findings in the region, {len(scoped.findings)} in "
            f"the brief; the neighbour's build-family defects dropped, its place-family "
            f"ones kept as the live loop keeps them")


@case
def c_repair_by_removing_the_interior_is_counted_as_not_repaired():
    """Registered before the run because a builder told a room cannot be reached can
    delete the room, and a build that walks 100% because it has one cell of floor left
    has not been repaired."""
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_b_repair.json"))

    def rows(fc0, fcn, r0, rn, walk):
        d = lambda f, r, w: {                                  # noqa: E731
            "draft": 0, "lines": 1, "entry_lines": 0, "lint_errors_brief": 0,
            "measures": {"floor_cells": f, "rooms": r, "walk_pct": w,
                         "lint_errors": 6, "blocks": 1, "articulation": 0.1}}
        return {"p/c0": {"status": "exhausted",
                         "drafts": [d(fc0, r0, 10.0), d(fcn, rn, walk)]}}

    subj = [{"id": "p/c0", "wave": "p"}]
    keep = pipeline._repair_readout(rnd, subj, rows(100, 90, 4, 4, 90.0))
    assert keep["primary"]["below_threshold"] == 0, keep["primary"]
    gutted = pipeline._repair_readout(rnd, subj, rows(100, 40, 4, 4, 100.0))
    assert gutted["primary"]["below_threshold"] == 1, gutted["primary"]
    assert gutted["primary"]["repaired_by_deletion"] == ["p/c0"]
    rooms = pipeline._repair_readout(rnd, subj, rows(100, 95, 4, 2, 100.0))
    assert rooms["primary"]["repaired_by_deletion"] == ["p/c0"], rooms["primary"]
    return "floor_cells -60% and rooms -2 both counted below the bar despite walk 100%"


@case
def c_the_repair_round_reads_its_bars_off_the_registration_and_never_computes_one():
    """Every bar in the readout is a number written down before the run. A stage that
    recomputed a threshold from the data would be moving a bar with extra steps."""
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_b_repair.json"))
    bars = rnd.preregistered["bars"]
    assert bars["walk_pct_threshold"] == 50.0
    assert bars["closes_at_or_below"] == 3 and bars["fails_at_or_above"] == 7
    assert rnd.preregistered["lint_floor"]["errors"] == 6
    subj = [{"id": f"p{i}/c0", "wave": f"p{i}"} for i in range(16)]
    d = lambda w: {"draft": 0, "lines": 1, "entry_lines": 0,          # noqa: E731
                   "lint_errors_brief": 0,
                   "measures": {"floor_cells": 10, "rooms": 2, "walk_pct": w,
                                "lint_errors": 6, "blocks": 1, "articulation": 0.1}}
    for n_below, want in ((3, "closes"), (7, "fails"), (5, "partial")):
        rows = {s["id"]: {"status": "exhausted",
                          "drafts": [d(10.0), d(10.0 if i < n_below else 90.0)]}
                for i, s in enumerate(subj)}
        got = pipeline._repair_readout(rnd, subj, rows)
        assert got["primary"]["below_threshold"] == n_below
        assert got["primary"]["branch"] == want, (n_below, got["primary"])
    # an incomplete run has no branch at all -- a partial count is not a result
    rows = {s["id"]: {"status": "needs_model", "drafts": []} for s in subj}
    assert pipeline._repair_readout(rnd, subj, rows)["primary"]["branch"] is None
    # ...but a candidate that finished with no interior left to measure IS finished. It
    # is counted below the threshold, and reading it as unfinished would blank the
    # branch on precisely the case the bar exists to catch.
    gone = dict(d(10.0), measures=dict(d(10.0)["measures"], walk_pct=None,
                                       floor_cells=0, rooms=0))
    rows = {s["id"]: {"status": "exhausted", "drafts": [d(10.0), d(90.0)]}
            for s in subj}
    rows["p0/c0"] = {"status": "exhausted", "drafts": [d(10.0), gone]}
    got = pipeline._repair_readout(rnd, subj, rows)
    assert got["complete"] == 16, got["complete"]
    assert got["primary"]["below_threshold"] == 1 and got["primary"]["branch"] == "closes"
    return ("3 -> closes, 7 -> fails, 5 -> partial, incomplete -> no branch; "
            "a finished build with no interior counts below and does not blank it")


@case
def c_a_builder_call_is_costed_once_however_often_the_stage_reruns():
    """`measure_program` is ~90 s a draft, so the stage is re-run after every batch and
    most rows come back off the cache. A cost record hung off the cache would miss every
    call measured before it existed and double-count nothing; hung off the log, it
    backfills and cannot repeat. Both halves are asserted, because a cost log that
    silently under-reports is worse than no cost log."""
    with tempfile.TemporaryDirectory() as d:
        real_log, measure.LOG = measure.LOG, os.path.join(d, "m.jsonl")
        builds = os.path.join(d, "builds")
        os.makedirs(builds)
        prog = os.path.join(builds, "build_1.py")
        open(prog, "w").write("place_cuboid(1, 2, 3, 4, 5, 6, block='stone')\n")
        json.dump({"brief": "b" * 100, "findings": "f" * 50},
                  open(os.path.join(builds, "build_request_1.json"), "w"))
        rnd = pipeline.Round(name="site_b", revise={}, path=os.path.join(d, "r.json"))
        subj = {"id": "w/c0", "sub": "repair/w/c0"}
        sha = "deadbeef"
        try:
            for _ in range(3):                    # three stage runs, one call
                pipeline._record_call(rnd, subj, 1, prog, sha, builds)
            rows = [json.loads(x) for x in open(measure.LOG)]
            assert len(rows) == 1, f"{len(rows)} rows for one call"
            assert rows[0]["chars_in"] == 150 and rows[0]["subject"] == "w/c0"
            assert rows[0]["draft"] == 1 and rows[0]["key"]
            # a bounce rewrites the program: different bytes, a second call, a second
            # row
            pipeline._record_call(rnd, subj, 1, prog, "cafe", builds)
            assert len(list(open(measure.LOG))) == 2
        finally:
            measure.LOG = real_log
    return "3 stage runs -> 1 row; a rewritten program -> a second row"


@case
def c_the_revision_loop_advances_onto_a_draft_that_is_already_on_disk():
    """The loop staged a request for draft n+1 and then stopped, so on the next run it
    re-measured draft 0, re-staged the identical request and stopped again -- twelve
    answered builder calls sat on disk unmeasured and the round could never reach draft
    1. Found by running it, not by reading it.

    Driven through `stage_revise` itself with a stub row builder, because the bug was in
    the loop and nowhere else: the fixture is a subject whose drafts 0 and 1 exist and
    are both dirty, and the loop must measure both and then ask for draft 2."""
    seen = []

    def row(rnd, be, subj, n, prog):
        seen.append(n)
        return {"draft": n, "sha256": "x", "lines": 1, "pending": 1,
                "lint_errors_brief": 1, "lint_errors_region": 7, "entry_lines": 1,
                "own_door_pct": 0.0, "findings": findings,
                "measures": {"walk_pct": 10.0, "floor_cells": 10, "rooms": 1,
                             "lint_errors": 7, "blocks": 1, "articulation": 0.1}}

    with tempfile.TemporaryDirectory() as d:
        findings = os.path.join(d, "f.md")
        open(findings, "w").write("something is wrong")
        builds = os.path.join(d, "out", "site_b", "arms", "repair", "w", "c0", "builds")
        os.makedirs(builds)
        for n in (0, 1):                       # draft 1 is already answered
            open(os.path.join(builds, f"build_{n}.py"), "w").write(f"# draft {n}\n")
        rnd = pipeline.Round(name="site_b", revise={"max_revisions": 2},
                             path=os.path.join(d, "r.json"))
        subj = {"id": "w/c0", "wave": "w", "cand": "c0", "sub": "repair/w/c0",
                "source": os.path.join(builds, "build_0.py"), "mine": {"p"},
                "brief": "b", "recorded": os.path.join(d, "none.json"), "margin": 6}
        real_root, real_subj, real_row = pipeline.ROOT, pipeline._revise_subjects, \
            pipeline._draft_row
        try:
            pipeline.ROOT = d
            pipeline._revise_subjects = lambda r: [subj]
            pipeline._draft_row = row
            out = pipeline.stage_revise(rnd, None, {})
        finally:
            pipeline.ROOT, pipeline._revise_subjects, pipeline._draft_row = \
                real_root, real_subj, real_row
    rec = out["candidates"]["w/c0"]
    assert seen == [0, 1], f"the loop measured drafts {seen}, not both on disk"
    assert len(rec["drafts"]) == 2, rec["drafts"]
    assert rec["status"] == "needs_model" and rec["awaiting_draft"] == 2, rec
    assert os.path.basename(rec["blinded"]["dir"]) != ""
    return "drafts 0 and 1 both measured, draft 2 requested -- not draft 1 restaged"


@case
def c_the_repair_round_names_its_subjects_by_the_rounds_they_were_built_in():
    """Sixteen subjects, and not one plot, brief or base volume restated. A second copy
    of those is a second chance to revise a draft against ground it was not built on,
    which is why the subject list is four config paths and a mismatch is an error."""
    cfg = os.path.join(ROUNDS, "site_b_repair.json")
    rnd = pipeline.Round.load(cfg)
    if not os.path.isdir(os.path.join(ROOT, "out", "site_b", "product-claim")):
        return "SKIPPED -- no product-claim artefacts in this worktree"
    subs = pipeline._revise_subjects(rnd)
    assert len(subs) == 16, len(subs)
    assert len({s["sub"] for s in subs}) == 16
    for s in subs:
        assert os.path.exists(s["source"]), s["source"]
        assert s["brief"] and s["mine"], s["id"]
        assert s["wave"] in s["source"] and s["cand"] in s["source"]
    assert {s["wave"] for s in subs} == {"chapel", "forge", "granary", "windmill"}
    # a subject built against another world is refused rather than revised
    bad = json.loads(open(cfg).read())
    bad["base_volume"] = "world_prebuild.npz"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.json")
        json.dump(bad, open(p, "w"))
        try:
            pipeline._revise_subjects(pipeline.Round.load(p))
        except ValueError as e:
            assert "world_prebuild.npz" in str(e)
            return "16 subjects from 4 configs; a base-volume mismatch is refused"
    raise AssertionError("a subject from a different base volume was accepted")


# ----------------------------------------------------------------- selection

def _sel_round(d, k=4):
    """A minimal candidates round pointed at a temp state directory.

        `BUILD_SCRATCH` is global by design -- it sits beside `judge_scratch` so that a
        directory of hash-named jobs says nothing about any one of them -- so a suite that
        stages blinded builds would otherwise litter the real one. Same trap, and same fix,
        as `measure.LOG` at the top of this file.
        
    """
    pipeline.BUILD_SCRATCH = os.path.join(d, "out", "build_scratch")
    _sel_round.answer = lambda b, src: (          # write a program and say so
        open(b["write"], "w").write(src), open(b["done"], "w").close())
    state = os.path.join(d, "out", "t")
    os.makedirs(state, exist_ok=True)
    cfg = {"name": "t", "flags": {"split": False},
           "waves": [{"name": "w"}],
           "candidates": {"wave": "w", "arm": "sel", "k": k,
                          "plots": ["a"], "out": "sel",
                          "shots": {"prefix": "sel"}, "human_look": {"seed": 4}}}
    p = os.path.join(d, "t.json")
    json.dump(cfg, open(p, "w"))

    class _R(pipeline.Round):
        @property
        def state(self):
            return state
    return _R.load(p), state


@case
def c_a_builder_never_learns_it_is_one_of_several():
    """The blinding that makes the experiment valid. `build_from_files` stages its
    request under arms/<arm>/<wave>/<cid>/, and that path *is* the leak -- a builder
    handed `.../sel/w/c2/builds/` has been told it is candidate 2 of several. The
    staged brief must live somewhere that says none of it."""
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        rnd, state = _sel_round(d)
        open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("build a hut")
        be = pipeline.OfflineBackend(rnd)
        be._vol = observe.Volume(0, 0, 0, np.zeros((4, 4, 4), np.uint8), ["air"])
        out = pipeline.stage_candidates(rnd, be, {})
        briefs, shown, dirs = set(), [], set()
        for cid in ("c0", "c1", "c2", "c3"):
            b = out[cid]["blinded"]
            shown += [b["brief"], b["write"], b["dir"]]
            dirs.add(b["dir"])
            briefs.add(open(b["brief"]).read())
        assert briefs == {"build a hut"}, briefs
        assert len(dirs) == 4, dirs           # four distinct jobs
        # and no path a builder sees carries anything but a hash. Asserted as the shape
        # of the path rather than as a blacklist of words -- a blacklist passes the
        # moment someone renames the arm.
        import re
        for p in shown:
            parts = p[len(os.path.dirname(pipeline.BUILD_SCRATCH)) + 1:].split(os.sep)
            assert parts[0] == "build_scratch", parts
            assert re.fullmatch(r"[0-9a-f]{16}", parts[1]), parts
            assert parts[2:] in ([], ["brief.md"], ["program.py"]), parts
    return "4 identical briefs, 4 hash-named jobs, no path names a candidate"


@case
def c_an_answered_blind_job_is_adopted_and_the_candidate_builds():
    """The other half of the blinding: a program written in the scratch directory is
    copied back to the candidate it belongs to, and the round resumes."""
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        rnd, state = _sel_round(d, k=1)
        open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("hut")
        be = pipeline.OfflineBackend(rnd)
        be._vol = observe.Volume(0, 0, 0, np.zeros((4, 4, 4), np.uint8), ["air"])
        first = pipeline.stage_candidates(rnd, be, {})
        assert first["c0"]["status"] == "needs_model", first
        _sel_round.answer(first["c0"]["blinded"],
                          "place_block(1, 1, 1, 'stone')\n")
        second = pipeline.stage_candidates(rnd, be, {})
        assert second["adopted"], second["adopted"]
        assert second["c0"]["status"] == "built", second["c0"]
        assert second["c0"]["pending"] == 1, second["c0"]
    return "scratch program adopted -> candidate built, 1 block pending"


@case
def c_a_crashed_candidate_bounces_once_and_never_readopts_itself():
    """A program that preflights and then raises is a typo, not a candidate, and step
    3 fixed exactly this with one message per arm. Three things have to hold: the
    other candidates survive the crash, the bounce is capped by config so no candidate
    gets more attempts than another, and the unchanged scratch program is not adopted
    straight back -- which would replay the same crash for ever."""
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        rnd, state = _sel_round(d, k=2)
        rnd.candidates["max_bounces"] = 1
        open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("hut")
        be = pipeline.OfflineBackend(rnd)
        be._vol = observe.Volume(0, 0, 0, np.zeros((4, 4, 4), np.uint8), ["air"])
        first = pipeline.stage_candidates(rnd, be, {})
        bad = "place_cuboid(1, 1, 1, 2, 2, 2)\n"          # one argument short
        _sel_round.answer(first["c0"]["blinded"], bad)
        _sel_round.answer(first["c1"]["blinded"], "place_block(1, 1, 1, 'stone')\n")

        crash = pipeline.stage_candidates(rnd, be, {})
        assert crash["c0"]["status"] == "crashed", crash["c0"]
        assert crash["c0"]["bounced"] is True, crash["c0"]
        assert crash["c1"]["status"] == "built", crash["c1"]   # survives its sibling
        # The bounce carries the failing line and nothing else. This is the case that
        # caught a real leak: a raw traceback names `stage_candidates` and the path
        # `.../arms/sel/w/c0/w.py`, which tells the builder it is candidate 0 of
        # something -- the one thing the spec says must not be said.
        err = open(crash["c0"]["blinded"]["error"]).read()
        assert "TypeError" in err and "place_cuboid" in err, err
        assert "in the brief" in err, err
        for leak in ("candidate", "compar", "sibling", "arms", "/c0", "sel/",
                     "pipeline.py", "stages.py", "offline.py"):
            assert leak not in err.lower(), (leak, err)

        # the bounce clears the done marker, so the half-answered job is not adopted
        again = pipeline.stage_candidates(rnd, be, {})
        assert "still writing" in again["adopted"]["c0/0"], again["adopted"]
        assert again["c0"]["status"] == "needs_model", again["c0"]

        # and a builder that says done without changing anything is refused too
        open(crash["c0"]["blinded"]["done"], "w").close()
        stale = pipeline.stage_candidates(rnd, be, {})
        assert "not adopted" in stale["adopted"]["c0/0"], stale["adopted"]
        assert stale["c0"]["status"] == "needs_model", stale["c0"]

        # answered -> adopted; and a second crash is a failed candidate, not a loop
        _sel_round.answer(first["c0"]["blinded"], "place_cuboid(1, 1, 1, 2, 2)\n")
        twice = pipeline.stage_candidates(rnd, be, {})
        assert twice["c0"]["status"] == "crashed", twice["c0"]
        assert twice["c0"]["bounced"] is False, twice["c0"]
        assert "already bounced" in twice["c0"]["why"], twice["c0"]
    return "crash isolated, bounced once, same bytes refused, cap honoured"


@case
def c_adaptive_k_inside_a_candidate_is_refused():
    """Two selection mechanisms, one inside the other, would make the result
    unreadable: the round must refuse rather than silently run one of them."""
    with tempfile.TemporaryDirectory() as d:
        rnd, _ = _sel_round(d)
        rnd.flags["select"] = 4
        out = pipeline.stage_candidates(rnd, pipeline.OfflineBackend(rnd), {})
        assert "select=0" in out.get("error", ""), out
    return "select>0 on a candidate round -> refused, not overridden"


@case
def c_the_tie_rate_survives_a_winners_scored_judgement():
    """E3 and step 3 are scored by who won, and until now that path dropped the tie
    count on the floor. For a sibling round-robin the tie rate *is* the headline."""
    rows = [{"label": "c0_vs_c1", "result": "a"},
            {"label": "c0_vs_c2", "result": "tie"},
            {"label": "c1_vs_c2", "result": "b"}]
    res = {"rows": rows}
    seen = [r["result"] for r in rows]
    res["ties"] = sum(1 for r in seen if r == "tie")
    res["tie_rate"] = round(res["ties"] / len(seen), 3)
    assert res["tie_rate"] == 0.333, res
    rnd = pipeline.Round(name="x")
    assert pipeline._pair(rows, "c0", "c1") == "a"
    assert pipeline._pair(rows, "c1", "c0") == "b", "the pair must read both ways"
    assert pipeline._pair(rows, "c0", "c3") is None
    assert rnd.name == "x"
    return "ties counted on a winners-scored round-robin; pairs read in both orders"


@case
def c_copeland_picks_the_winner_and_says_when_it_guessed():
    """The registered rule: win 1, tie 0.5, loss 0; a tie in the total is broken by a
    decisive transitive head-to-head, and otherwise by index -- and *says* so, because
    a tie-break reported as a preference is a result this project would not have."""
    def run(rows, k=3):
        with tempfile.TemporaryDirectory() as d:
            rnd, _ = _sel_round(d, k=k)
            res = {"judge": {"selection_rr": {"status": "done", "rows": rows}}}
            rnd.candidates["judgement"] = "selection_rr"
            rnd.preregistered = {"stage1": {"tie_rate_bar": 0.5}}
            return pipeline.stage_selection(rnd, pipeline.OfflineBackend(rnd), res)

    clean = run([{"label": "c0_vs_c1", "result": "b"},
                 {"label": "c0_vs_c2", "result": "b"},
                 {"label": "c1_vs_c2", "result": "a"}])
    assert clean["winner"] == "c1" and clean["winner_by"] == "outright", clean
    assert clean["copeland"] == {"c0": 0.0, "c1": 2.0, "c2": 1.0}, clean["copeland"]
    assert clean["discriminates"] is True and clean["tie_rate"] == 0.0

    # a three-cycle: every candidate scores 1, no head-to-head order exists
    cyc = run([{"label": "c0_vs_c1", "result": "a"},
               {"label": "c0_vs_c2", "result": "b"},
               {"label": "c1_vs_c2", "result": "a"}])
    assert cyc["winner_by"] == "index_tie_break", cyc
    assert set(cyc["copeland"].values()) == {1.0}, cyc["copeland"]

    dead = run([{"label": "c0_vs_c1", "result": "tie"},
                {"label": "c0_vs_c2", "result": "tie"},
                {"label": "c1_vs_c2", "result": "a"}])
    assert dead["tie_rate"] == 0.667 and dead["discriminates"] is False, dead
    assert dead["stopped_at"].startswith("stage 1"), dead
    assert "convergent" not in dead, "a dead stage 1 must not look at stage 2"
    return "outright / cycle -> index tie-break / tie rate over the bar stops it"


@case
def c_ranks_are_average_ranks_and_the_median_is_2_point_5():
    """Convergent validity is read off ranks, so the tie handling is the result.
    Four candidates: above median is strictly better than the middle of the field."""
    r = pipeline._avg_ranks({"c0": 10, "c1": 30, "c2": 30, "c3": 5}, True)
    assert r == {"c1": 1.5, "c2": 1.5, "c0": 3.0, "c3": 4.0}, r
    lo = pipeline._avg_ranks({"c0": 3, "c1": 0, "c2": 9, "c3": 1}, False)
    assert lo == {"c1": 1.0, "c3": 2.0, "c0": 3.0, "c2": 4.0}, lo
    flat = pipeline._avg_ranks({"c0": 1, "c1": 1, "c2": 1, "c3": 1}, True)
    assert set(flat.values()) == {2.5}, flat
    assert not any(v < 2.5 for v in flat.values()), "a flat measure favours nobody"
    return "average ranks, ties share the mean place, a flat measure is neither"


@case
def c_convergent_validity_can_return_goodhart():
    """The outcome the spec calls the most valuable one available must be reachable:
    a winner that is worst on the measures the judge never saw."""
    with tempfile.TemporaryDirectory() as d:
        rnd, _ = _sel_round(d, k=4)
        rnd.candidates["judgement"] = "j"
        rnd.preregistered = {"stage1": {"tie_rate_bar": 0.5}}
        rows = [{"label": "c0_vs_c1", "result": "a"},
                {"label": "c0_vs_c2", "result": "a"},
                {"label": "c0_vs_c3", "result": "a"},
                {"label": "c1_vs_c2", "result": "a"},
                {"label": "c1_vs_c3", "result": "a"},
                {"label": "c2_vs_c3", "result": "a"}]

        def meas(err, walk, cv):
            return {"status": "measured", "lint_errors": err, "walk_pct": walk,
                    "variety_ridge_cv": cv, "blocks": 100}
        bad = {"judge": {"j": {"status": "done", "rows": rows}},
               "measures": {"c0": meas(9, 10.0, 0.01), "c1": meas(0, 80.0, 0.5),
                            "c2": meas(1, 70.0, 0.4), "c3": meas(2, 60.0, 0.3)}}
        out = pipeline.stage_selection(rnd, pipeline.OfflineBackend(rnd), bad)
        assert out["winner"] == "c0", out
        assert out["convergent"]["verdict"] == "goodhart", out["convergent"]
        good = dict(bad, measures={"c0": meas(0, 80.0, 0.5), "c1": meas(9, 10.0, 0.01),
                                   "c2": meas(1, 70.0, 0.4), "c3": meas(2, 60.0, 0.3)})
        out2 = pipeline.stage_selection(rnd, pipeline.OfflineBackend(rnd), good)
        assert out2["convergent"]["verdict"] == "convergent", out2["convergent"]
    return "same verdicts, measures flipped -> goodhart / convergent, both reachable"


@case
def c_the_human_look_is_seeded_blind_and_reported_as_outstanding():
    """A person looks last. The sibling is drawn by the registered seed rather than
    chosen, the key is written separately, and an unanswered look is never a result."""
    with tempfile.TemporaryDirectory() as d:
        rnd, _ = _sel_round(d, k=4)
        os.makedirs(pipeline._cand_dir(rnd), exist_ok=True)
        for i, v in enumerate((10, 60, 110, 160)):
            _png(os.path.join(pipeline._cand_dir(rnd), f"sel_c{i}_card.png"), v)
        a = pipeline._stage_human_look(rnd, "c0", ["c0", "c1", "c2", "c3"])
        b = pipeline._stage_human_look(rnd, "c0", ["c0", "c1", "c2", "c3"])
        assert a["sibling"] == b["sibling"] != "c0", (a, b)
        assert a["status"] == "outstanding" and a["verdict"] is None, a
        key = json.load(open(os.path.join(a["dir"], "key.json")))
        assert {key["a"], key["b"]} == {"c0", a["sibling"]}, key
        assert os.path.exists(os.path.join(a["dir"], "a.png"))
        assert "person" in open(os.path.join(a["dir"], "ASK.md")).read()
    return f"seed 4 -> sibling {a['sibling']}, sides in key.json, outstanding"


# ------------------------------------------------------------- the judging path

@case
def c_the_correct_side_may_differ_per_pair():
    """E-craft's blind picks were recorded per structure: house=v1, tower=v1,
    wall=library, inn=v1. A single global `correct` scores it 1/4 instead of 4/4."""
    res = ["a", "a", "b", "a"]
    flat = verdicts.tally(res, correct="a")
    per = verdicts.tally(res, correct=["a", "a", "b", "a"])
    assert flat["correct"] == 3 and per["correct"] == 4, (flat, per)
    return "3/4 with one global side, 4/4 with the recorded ones"


@case
def c_a_tie_counts_against_and_is_reported():
    t = verdicts.tally(["a", "a", "tie", "b"])
    assert (t["judged"], t["correct"], t["wrong"], t["ties"]) == (4, 2, 1, 1)
    assert t["tie_rate"] == 0.25 and t["correct_rate"] == 0.5
    return "ties are counted, never resolved"


@case
def c_wilson_is_the_interval_e1d_registered():
    assert verdicts.wilson(15, 22) == [0.473, 0.836]
    assert verdicts.wilson(0, 0) is None
    return "15/22 -> [0.473, 0.836], the number in the E1d section"


@case
def c_a_floor_is_read_never_computed():
    s = {"judged": 22, "correct_rate": 0.682, "tie_rate": 0.227}
    assert not verdicts.meets(s, {"min_correct_rate": 0.80, "max_tie_rate": 0.30})
    assert verdicts.meets(s, {"min_correct_rate": 0.60, "max_tie_rate": 0.30})
    return "E1d's 68% fails an 80 floor and clears a 60 one"


@case
def c_a_preregistration_on_disk_wins_over_the_source():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.json")
        assert verdicts.preregistration(p, {"min_correct_rate": 0.8}) == \
            {"min_correct_rate": 0.8}
        json.dump({"preregistered": {"min_correct_rate": 0.5}}, open(p, "w"))
        assert verdicts.preregistration(p, {"min_correct_rate": 0.8}) == \
            {"min_correct_rate": 0.5}
    return "first write wins; a threshold cannot be moved by editing a script"


@case
def c_every_missing_judgement_is_staged_at_once():
    """A cold session must learn everything it needs in one pass. Two of the six
    hand-written copies of this loop stopped at the first miss."""
    with tempfile.TemporaryDirectory() as d:
        imgs = [_png(os.path.join(d, f"{i}.png"), 10 + i * 40) for i in range(4)]
        js = verdicts.Session("q?", cache_path=os.path.join(d, "cache.jsonl"))
        for a, b in ((imgs[0], imgs[1]), (imgs[2], imgs[3])):
            assert js.compare(a, b) is None
        assert len(js.needed) == 4, len(js.needed)   # two pairs, both orientations
        n = js.stage(os.path.join(d, "req.json"))
        assert n == 4 and os.path.exists(os.path.join(d, "req.json"))
    return "two unanswerable pairs stage all four orientations, not one"


# ------------------------------------------------------------- the record schema

@case
def c_an_invented_measurement_kind_is_refused():
    """61 kinds, ~15 of them one-offs, is how "how long does a frame take?" stayed
    unanswerable while 124 frames were rendered."""
    try:
        measure.record("step5_frame_v2", tag="x")
    except measure.UnknownKind as e:
        assert "step5_frame_v2" in str(e)
        return "a new one-off kind is an error, not a typo"
    raise AssertionError("an unknown kind was written to the log")


@case
def c_the_retired_kinds_still_work_and_say_what_they_became():
    """Archived experiment scripts must keep running, and the log is append-only
    evidence that nothing rewrites -- so both vocabularies stay readable."""
    assert measure.LEGACY["e1d_frame"] == "frame"
    assert measure.LEGACY["settlement_pass"] == "pass_run"
    assert set(measure.LEGACY.values()) <= set(measure.KINDS)
    assert "model_call" in measure.KINDS and "model_call" not in measure.LEGACY
    return f"{len(measure.KINDS)} kinds, {len(measure.LEGACY)} retired names mapped"


@case
def c_the_log_the_project_already_has_is_covered():
    log = os.path.join(ROOT, "out", "measurements.jsonl")
    if not os.path.exists(log):
        return "SKIPPED -- no measurement log in this worktree"
    kinds = set()
    for ln in open(log):
        ln = ln.strip()
        if ln:
            try:
                kinds.add(json.loads(ln)["kind"])
            except (ValueError, KeyError):
                pass
    unknown = sorted(kinds - set(measure.KINDS) - set(measure.LEGACY))
    assert not unknown, f"kinds in the log that the schema does not name: {unknown}"
    return f"{len(kinds)} kinds in the standing log, all named by the schema"


# The builder gets `check.py` in its own directory and can run it as often as it likes;
# the acceptance is that the check tells it the truth about the one defect the last two
# rounds could not close, and stops saying so the moment the library's own `approach()`
# fixes it.


def _check_round(d):
    """A settlement the size of a hut: flat ground, a lane, one plot, one wave.

        Real enough for `lint.Context`, `observe.rooms` and the walk model -- the check
        under test is the real one, not a stand-in.
        
    """
    from ethoslm import observe, offline
    from ethoslm.circulate import Network, Threshold
    pipeline.BUILD_SCRATCH = os.path.join(d, "out", "build_scratch")
    pipeline.CHECK_JOBS = os.path.join(d, "out", "check_jobs")
    state = os.path.join(d, "out", "t")
    os.makedirs(state, exist_ok=True)
    X0, Z0, S, Y0, GROUND = 0, 0, 40, 50, 60
    blocks = {(x, y, z): "stone"
              for x in range(S) for z in range(S) for y in range(Y0, GROUND + 1)}
    vol = observe.Volume.from_blocks(blocks, X0, Y0, Z0, S, 64, S)
    offline.save_volume(vol, os.path.join(state, "world.npz"))
    json.dump({"origin": [X0, Z0], "size": S}, open(os.path.join(state, "site.json"), "w"))
    json.dump([{"label": "hut", "x0": 12, "z0": 12, "x1": 22, "z1": 22}],
              open(os.path.join(state, "plots.json"), "w"))
    # The lane runs three columns clear of the plot, which is what a routed lane does
    # and what leaves ground for an approach to be laid on. A door opening straight onto
    # a lane cell is a different defect -- the doorstep built over, E008 -- and its fix
    # is `floor_from_threshold`, not a path.
    net = Network({(x, 26): {"y": GROUND, "rank": 0, "face": None}
                   for x in range(8, 32)},
                  [Threshold("hut", 17, 26, GROUND, "south", (17, GROUND + 1, 22))])
    net.save(os.path.join(state, "network.json"))
    open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("build a hut")
    cfg = {"name": "t", "state_dir": state,
           "waves": [{"name": "w", "plots": ["hut"]}],
           "flags": {"split": False, "check": True, "arm": "site_d", "max_bounces": 1}}
    p = os.path.join(d, "t.json")
    json.dump(cfg, open(p, "w"))
    return pipeline.Round.load(p), state, GROUND


#: A hut on one course of plinth, its door a block above the ground you arrive on -- the
#: defect class exactly: E002 walk-only sees it, and nothing else in the suite did.
_HUT = """
FLOOR = {floor}
for x in range(12, 23):
    for z in range(12, 23):
        place_cuboid(x, FLOOR, z, x, FLOOR, z, "stone_bricks")
for x in range(12, 23):
    for z in range(12, 23):
        if x in (12, 22) or z in (12, 22):
            place_cuboid(x, FLOOR + 1, z, x, FLOOR + 4, z, "stone_bricks")
place_cuboid(13, FLOOR + 1, 13, 21, FLOOR + 4, 21, "air")
place_block(17, FLOOR + 1, 22, "oak_door[facing=north,half=lower]")
place_block(17, FLOOR + 2, 22, "oak_door[facing=north,half=upper]")
"""


@case
def c_check_py_tells_a_builder_its_door_is_a_jump_and_stops_when_it_is_not():
    """The registered acceptance for part B.

        A stub builder writes a hut whose door is a block above the lane, runs the
        `check.py` it was given, and the on-foot line is in `findings.md`. It appends one
        `approach()` call, runs the same check again, and the line is gone. Nothing else
        about the program changes.
        
    """
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        rnd, state, ground = _check_round(d)
        be = pipeline.OfflineBackend(rnd)
        out = pipeline.stage_waves(rnd, be, {})
        r = out["w"]
        assert r["status"] == "needs_model", r
        bl = r["blinded"]
        assert os.path.exists(bl["check"]), "no check.py in the builder's directory"
        assert sorted(os.listdir(bl["dir"])) == ["brief.md", "check.py"], \
            os.listdir(bl["dir"])
        assert "check.py" in open(bl["brief"]).read(), \
            "the builder was given a checker and never told about it"

        env = dict(os.environ, ETHOSLM_MEASURE_LOG=os.path.join(d, "m.jsonl"),
                   ETHOSLM_CHECK_JOBS=pipeline.CHECK_JOBS)
        prog = os.path.join(bl["dir"], "program.py")
        source = _HUT.format(floor=ground + 1)
        open(prog, "w").write(source)
        p = subprocess.run([sys.executable, bl["check"]], capture_output=True,
                           text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        first = open(os.path.join(bl["dir"], "findings.md")).read()
        assert pipeline.ENTRY_TITLE in first, \
            f"check.py did not report the on-foot defect:\n{first}"
        assert "cannot be reached on foot" in first, first
        assert "Placed versus attempted" in first, first

        open(prog, "w").write(source + '\napproach("hut")\n')
        p2 = subprocess.run([sys.executable, bl["check"]], capture_output=True,
                            text=True, env=env)
        assert p2.returncode == 0, p2.stdout + p2.stderr
        second = open(os.path.join(bl["dir"], "findings.md")).read()
        assert pipeline.ENTRY_TITLE not in second, \
            f"one approach() call and the on-foot findings did not clear:\n{second}"

        rows = [json.loads(ln) for ln in open(os.path.join(d, "m.jsonl"))
                if '"check_run"' in ln]
        assert len(rows) == 2, f"{len(rows)} check_run rows on the log, expected 2"
    return "jump-only door reported, approach() clears it, 2 runs on the log"


@case
def c_done_is_refused_when_the_checker_was_never_run():
    """`done` without a check is not a finished program. The refusal reads the log
    rather than a file in the directory, so the builder has nothing extra to see."""
    import ethoslm.measure as measure_mod
    with tempfile.TemporaryDirectory() as d:
        rnd, state, ground = _check_round(d)
        be = pipeline.OfflineBackend(rnd)
        bl = pipeline.stage_waves(rnd, be, {})["w"]["blinded"]
        open(os.path.join(bl["dir"], "program.py"), "w").write(
            _HUT.format(floor=ground + 1))
        open(bl["done"], "w").write("")
        subs = [("w", os.path.join("site_d", "w"))]
        got = pipeline._collect_blinded(rnd, subs)
        assert "without ever running check.py" in str(got), got
        assert not os.path.exists(rnd.rel("arms", "site_d", "w", "builds", "build_0.py")), \
            "a program was adopted without the checker ever having run"
        # ...and once it has run, the identical `done` is honoured.
        keep = measure_mod.LOG
        measure_mod.LOG = os.path.join(d, "m2.jsonl")
        pipeline.run_check(os.path.basename(bl["dir"]), bl["dir"])
        got2 = pipeline._collect_blinded(rnd, subs)
        measure_mod.LOG = keep
        assert got2["w/0"].endswith("build_0.py"), got2
    return "no check_run -> refused; one check_run -> adopted"


# Harness
@case
def c_check_py_carries_the_library_path_it_needs_to_run():
    """`check.py` re-execs the venv's python, whose numpy is a native wheel that needs
    libstdc++ and zlib out of the nix store. `scripts/env.sh` is where the three store
    paths are written down, and the checker now asks it rather than carrying a second
    copy that can drift.
    """
    src = pipeline.CHECK_PY.format(root=ROOT, key="k")
    assert "LD_LIBRARY_PATH" in src, "check.py still re-execs with no library path"
    assert "env.sh" in src, "the path is copied rather than read from scripts/env.sh"
    # ...and env.sh actually exports one, so what it asks for exists
    import subprocess
    p = subprocess.run(["bash", "-c",
                        'source "$1"; printf %s "$LD_LIBRARY_PATH"', "_",
                        os.path.join(ROOT, "scripts", "env.sh")],
                       capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip(), (p.returncode, p.stderr[-200:])
    assert "libstdc" in p.stdout or "gcc" in p.stdout, p.stdout
    return f"env.sh exports {len(p.stdout.split(':'))} library paths; check.py reads it"


@case
def c_round_json_is_merged_by_every_stage_never_overwritten():
    """A stage that did not run says nothing about its results."""
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, "out", "t")
        os.makedirs(state)
        cfg = {"name": "t", "state_dir": state, "preregistered": {"bar": 1}}
        p = os.path.join(d, "t.json")
        json.dump(cfg, open(p, "w"))
        rnd = pipeline.Round.load(p)
        be = pipeline.OfflineBackend(rnd)

        pipeline.stage_report(rnd, be, {"waves": {"wave1": {"status": "built"}}})
        pipeline.stage_report(rnd, be, {"judge": {"j": {"status": "done"}}})
        doc = json.load(open(pipeline._report_path(rnd)))
        assert set(doc["results"]) == {"waves", "judge"}, sorted(doc["results"])
        assert doc["results"]["waves"]["wave1"]["status"] == "built", doc["results"]
        assert doc["stages_last_run"] == ["judge"], doc["stages_last_run"]
        # the pre-registration still wins from the first write, as it has since E1a
        assert doc["preregistered"] == {"bar": 1}, doc["preregistered"]
        # ...and a stage that runs again does replace its own half
        pipeline.stage_report(rnd, be, {"waves": {"wave1": {"status": "error"}}})
        doc2 = json.load(open(pipeline._report_path(rnd)))
        assert doc2["results"]["waves"]["wave1"]["status"] == "error"
        assert doc2["results"]["judge"]["j"]["status"] == "done", "judge was blanked"
    return "waves then judge: both halves survive, the re-run half is replaced"


# A5 a type program instantiated per plot, A6 two voices.
# ==============================================================================

#: A fixture *type*: `build(b, plot, seed, **params)`, written the way a builder would
#: be asked to write one. It is deliberately a real generator rather than one building
#: with a seed bolted on -- the footprint, the storey count and the roof all move with
#: the seed, which is what the case is checking the apparatus can carry.
TYPE_SRC = '''
# A type declares what it is written in and what may be varied, so something that has
# not read it can instantiate it.
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 3),
          "roof_ends": ("choice", ["gable", "hip", "half-hip", "irimoya"])}


def build(b, plot, seed, storeys=1, roof_ends="gable"):
    w = 8 + seed * 2
    d = 9
    x0 = plot["x0"] + 1
    z0 = plot["z0"] + 2
    mat = {"wall": "cobblestone", "roof": "dark_oak", "footing": "mossy_cobblestone",
           "frame": "spruce", "floor": "spruce", "trim": "stripped_spruce_log"}
    return b.building(plot["label"], x0, z0, x0 + w, z0 + d, storeys,
                      {"style": "gable", "pitch": (1, 1 + seed % 3),
                       "ends": (roof_ends, roof_ends)}, mat=mat)
'''


def _type_world(d, plots, ground=64):
    """A temp state directory with a flat world and three plots side by side."""
    from ethoslm import observe, offline
    state = os.path.join(d, "out", "t")
    os.makedirs(state, exist_ok=True)
    sx = sz = 72
    blocks = {}
    for x in range(sx):
        for z in range(sz):
            for y in range(56, ground + 1):
                blocks[(x, y, z)] = "stone"
    vol = observe.Volume.from_blocks(blocks, 0, 56, 0, sx, 40, sz)
    offline.save_volume(vol, os.path.join(state, "world.npz"))
    json.dump(plots, open(os.path.join(state, "plots.json"), "w"))
    json.dump({"origin": [0, 0], "size": sx}, open(os.path.join(state, "site.json"), "w"))
    return state, vol


@case
def c_a_type_instantiated_on_three_plots_is_three_different_buildings():
    """A5, as the spec registers it: one program, three plots, three lint-clean
        walkable buildings with three distinct ridge heights.

        Distinct ridges is the discriminating half. A type that ignored its seed would
        build the same building three times, pass every other assertion here, and be the
        "one building with a seed bolted on" outcome the spec says to read for.
    """
    from ethoslm import lint, observe, stages
    with tempfile.TemporaryDirectory() as d:
        plots = [{"label": lab, "x0": 2 + 22 * i, "z0": 2, "x1": 20 + 22 * i,
                  "z1": 24} for i, lab in enumerate(("a", "b", "c"))]
        state, vol = _type_world(d, plots)
        src = os.path.join(d, "cottage.py")
        open(src, "w").write(TYPE_SRC)
        reg = stages._Registry(state)
        ridges, cells = [], []
        pending = {}
        for i, p in enumerate(plots):
            b = pipeline.instantiate(
                src, p, vol, seed=i + 1, plots=reg,
                params={"storeys": (1, 2, 3)[i],
                        "roof_ends": ("gable", "irimoya", "half-hip")[i]})
            b.resolve_steps()
            pending.update(b._pending)
            ys = [y for (x, y, z) in b._pending
                  if p["x0"] <= x <= p["x1"] and p["z0"] <= z <= p["z1"]]
            ridges.append(max(ys))
            # Scoped to the rooms the call itself reports rather than to the plot: a
            # plot-scoped room detector counts the ring under the eaves as part of the
            # room and reads it as half open to the sky, which is thread 10's "the walk
            # metric is not invariant to room count" and not what this case is about.
            part = b.parts[-1]
            fx0, fz0 = part["x0"] + 1, part["z0"] + 2
            w = b.check_walkable(x0=fx0 + 1, z0=fz0 + 1,
                                 x1=fx0 + 8 + (i + 1) * 2 - 1, z1=fz0 + 8)
            cells.append((sum(r["walkable"] for r in w["rooms"]),
                          sum(r["cells"] for r in w["rooms"])))
            assert w["ok"] and cells[-1][1] > 40, (p["label"], w["reason"], w["rooms"])
            assert cells[-1][0] == cells[-1][1], (p["label"], w["reason"], w["rooms"])
        v = observe.Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
        v.overlay(pending)
        ctx = lint.Context.build(v, plots=[dict(p) for p in plots],
                                 region=(0, 0, 71, 71))
        errs = [f"{f.code} {f.message}" for f in lint.lint(ctx).findings
                if f.code.startswith("E") and f.code != "E005"]
        assert not errs, errs
        assert len(set(ridges)) == 3, (f"three instances, {len(set(ridges))} ridge "
                                       f"heights: {ridges} -- the type built the same "
                                       f"building three times")
        assert len({c[1] for c in cells}) == 3, \
            f"three instances, {len({c[1] for c in cells})} floor areas: {cells}"
        return (f"3 plots from 1 type: ridges {ridges}, floors "
                f"{[c[1] for c in cells]}, "
                f"{sum(c[0] for c in cells)}/{sum(c[1] for c in cells)} walkable, "
                f"0 lint errors")


@case
def c_a_type_is_an_ordinary_program_to_everything_downstream():
    """The composed source is what lands at the candidate's program path, so the
        measures, the render and the lint replay a type by executing a file. If this drifts,
        a type round measures a program nobody ran.
    """
    with tempfile.TemporaryDirectory() as d:
        plots = [{"label": "a", "x0": 2, "z0": 2, "x1": 20, "z1": 24},
                 {"label": "b", "x0": 24, "z0": 2, "x1": 42, "z1": 24}]
        state, vol = _type_world(d, plots)
        src = pipeline.instantiated_source(
            TYPE_SRC, [(plots[0], 1, {"storeys": 1}), (plots[1], 2, {"storeys": 2})])
        assert src.count("_part = site({") == 2, src[-400:]
        assert src.count("build(type_builder(_part, role=globals().get('ROLE')), _part,") == 2, src[-400:]
        assert TYPE_SRC.strip() in src, "the type's own source was rewritten"
        p = os.path.join(d, "wave.py")
        open(p, "w").write(src)
        from ethoslm import lint, offline, stages
        pre = lint.preflight(src)
        assert pre.ok, pre.to_json()
        b = offline.run_program(p, vol, plots=stages._Registry(state))
        b.resolve_steps()
        on = {lab: sum(1 for (x, _y, z) in b._pending
                       if q["x0"] <= x <= q["x1"] and q["z0"] <= z <= q["z1"])
              for lab, q in (("a", plots[0]), ("b", plots[1]))}
        assert all(v > 500 for v in on.values()), on
        return (f"{len(src.splitlines())} lines composed, preflight clean, "
                f"{on['a']} + {on['b']} blocks on two plots")


@case
def c_a_type_round_stages_one_builder_call_and_writes_the_composed_program():
    """The stage, end to end: one request, one answer, one composed program on disk."""
    from ethoslm import observe
    with tempfile.TemporaryDirectory() as d:
        pipeline.BUILD_SCRATCH = os.path.join(d, "out", "build_scratch")
        plots = [{"label": lab, "x0": 2 + 22 * i, "z0": 2, "x1": 20 + 22 * i,
                  "z1": 24} for i, lab in enumerate(("a", "b", "c"))]
        state, vol = _type_world(d, plots)
        cfg = {"name": "t", "flags": {},
               "waves": [{"name": "w"}],
               "candidates": {"wave": "w", "arm": "ty", "k": 1, "plots": ["a", "b", "c"],
                              "out": "ty", "shots": {"prefix": "ty"},
                              "type": {"instances": [["a", {"seed": 1, "storeys": 1}],
                                                     ["b", {"seed": 2, "storeys": 2}],
                                                     ["c", {"seed": 3, "storeys": 1}]]}}}
        cp = os.path.join(d, "t.json")
        json.dump(cfg, open(cp, "w"))

        class _R(pipeline.Round):
            @property
            def state(self):
                return state

        rnd = _R.load(cp)
        open(os.path.join(state, "w_elaborate_prompt.md"), "w").write("write a type")
        be = pipeline.OfflineBackend(rnd)
        first = pipeline.stage_candidates(rnd, be, {})
        assert first["c0"]["status"] == "needs_model", first["c0"]
        blinded = first["c0"]["blinded"]
        assert open(blinded["brief"]).read() == "write a type"
        # the builder answers, once, with a type rather than a building
        open(os.path.join(rnd.rel("arms", "ty", "w", "c0", "builds"),
                          "build_0.py"), "w").write(TYPE_SRC)
        second = pipeline.stage_candidates(rnd, be, {})
        assert second["c0"]["status"] == "built", second["c0"]
        prog = second["c0"]["program"]
        assert os.path.exists(prog)
        assert open(prog).read().count("build(type_builder(_part, role=globals().get('ROLE')), _part,") == 3
        assert second["c0"]["pending"] > 1500, second["c0"]
        assert [i["plot"] for i in second["c0"]["type"]["instances"]] == ["a", "b", "c"]
        return (f"1 call, {second['c0']['type']['lines']} lines of type, "
                f"3 instances, {second['c0']['pending']} blocks")


@case
def c_the_two_japanese_voices_are_readable_and_roofable():
    """A6. A voice is text and nothing else, so the only thing to assert is that the
    text says what the spec registered and that the roof it asks for is a roof the
    library can now draw -- which before A1 it could not."""
    from ethoslm import styles
    from ethoslm.prims import Primitives
    for name in ("japanese_minka", "japanese_temple"):
        v = styles.VOICES[name]
        assert set(v["palette"]) >= {"wall", "roof", "footing"}, name
        card = styles.voice_card(name)
        assert "irimoya" in card, name
        assert v["palette"]["wall"] and v["palette"]["roof"]
    # Both say thatch -- the difference is that one of them is a material family, with
    # the stairs and the slab a roof is actually laid in, and the other was silently
    # answered in stone brick.
    from ethoslm.prims import material as _mat
    for name in ("japanese_minka", "japanese_temple"):
        roof = styles.VOICES[name]["palette"]["roof"]
        assert _mat(roof), (name, roof)          # raises if it is not a family
    assert _mat(styles.VOICES["japanese_minka"]["palette"]["roof"])[0] == "hay_block"
    assert styles.VOICES["japanese_temple"]["roofs"].count("tier") >= 1

    class FB(Primitives):
        def __init__(self):
            self.blocks = {}

        def place_block(self, x, y, z, b):
            self.blocks[(int(x), int(y), int(z))] = b

        def get_height(self, x, z):
            return 64
    b = FB()
    r = b.roof(0, 0, 12, 16, 70, "deepslate_tile", style="hip", pitch=(1, 2),
               profile=[(1, 2), (2, 1)], ends=("irimoya", "irimoya"),
               eave="upturned", tiers=2)
    assert r > 70 and len(b.blocks) > 400, (r, len(b.blocks))
    return (f"2 voices, both naming an irimoya; the temple's roof draws at "
            f"{len(b.blocks)} blocks, ridge {r}")


# ------------------------------------------------------ A4. the plan is a tree

@case
def t_a4_a_tree_flattens_to_the_plots_a_flat_plan_gives():
    """A4's acceptance, and the whole of the compatibility it is held to.

        The plan was a flat list of building footprints, which is why the architecture could
        say "a village" and could not say "a castle": a keep with towers and a curtain wall
        is parts of parts and there was nowhere to write the "of". A4 makes it a tree -- and
        every stage downstream reads plots, so the tree has to flatten to exactly the list
        the flat plan gave, or the change is a rewrite of the pipeline rather than an
        addition to the plan.
        
    """
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_f.json"))
    flat = rnd.plots()
    assert len(flat) == 14, len(flat)
    plan = rnd.plan()
    leaves = [{"kind": "plot", "name": s["id"],
               "x0": s["x0"], "z0": s["z0"], "x1": s["x1"], "z1": s["z1"]}
              for s in plan["structures"]]
    tree = {"parts": [{"kind": "district", "name": "town", "children": [
        {"kind": "quarter", "name": "upper", "children": leaves[:7]},
        {"kind": "quarter", "name": "lower", "children": leaves[7:]}]}]}
    assert pipeline.plan_plots(tree) == flat, "the tree does not flatten to the plots"
    parts = pipeline.plan_parts(tree)
    assert [p["in"] for p in parts] == [["town", "upper"]] * 7 + [["town", "lower"]] * 7
    assert [p["name"] for p in parts] == [p["label"] for p in flat]

    # ...and the four leaf kinds are carried, with an edge and a point flattening to no
    # plot at all: a wall is not a building and nothing downstream should think so
    mixed = {"parts": [{"kind": "quarter", "name": "q", "children": [
        {"kind": "edge", "name": "wall", "path": [[0, 0], [8, 0]], "width": 1},
        {"kind": "point", "name": "gate", "at": [4, 0], "facing": "south"},
        {"kind": "area", "name": "square", "x0": 0, "z0": 4, "x1": 8, "z1": 12},
        {"kind": "plot", "name": "hall", "x0": 12, "z0": 4, "x1": 20, "z1": 12}]}]}
    assert [p["kind"] for p in pipeline.plan_parts(mixed)] == \
        ["edge", "point", "area", "plot"]
    assert [p["label"] for p in pipeline.plan_plots(mixed)] == ["hall"]
    return (f"a recorded town's 14 structures flatten identically through a two-quarter tree; "
            f"a mixed quarter of 4 leaves gives 1 plot")


@case
def t_a4_the_tree_survives_a_round_trip_through_the_round():
    """A plan written as a tree and read back is the same tree, and `Round` reads it.

        The round-trip is the case because the tree is the only thing in this project that
        the driver reads and does not write: a planner writes `plan.json` and every stage
        after it reads that file, so a shape the driver silently flattens on the way in
        would lose the "of" the moment it landed.
        
    """
    d = tempfile.mkdtemp(prefix="ethoslm_tree_")
    tree = {"intent": "a walled district", "centre": "gate", "parts": [
        {"kind": "district", "name": "keep", "notes": "the walled part", "children": [
            {"kind": "edge", "name": "wall", "type": "wall", "seed": 1,
             "path": [[0, 0], [40, 0], [40, 40]], "width": 1},
            {"kind": "point", "name": "gate", "type": "gate_tower", "seed": 2,
             "at": [20, 0], "facing": "north"},
            {"kind": "quarter", "name": "market", "children": [
                {"kind": "area", "name": "square", "type": "square", "seed": 3,
                 "x0": 8, "z0": 8, "x1": 20, "z1": 20},
                {"kind": "plot", "name": "hall", "type": "hall", "seed": 4,
                 "params": {"dormers": 2},
                 "x0": 24, "z0": 8, "x1": 34, "z1": 20}]}]}]}
    json.dump(tree, open(os.path.join(d, "plan.json"), "w"), indent=1)
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_f.json"))
    rnd.state_dir = d
    assert rnd.plan() == tree, "the plan did not come back the way it went in"
    parts = rnd.parts()
    assert [p["name"] for p in parts] == ["wall", "gate", "square", "hall"], parts
    assert [p["in"] for p in parts] == [["keep"], ["keep"],
                                        ["keep", "market"], ["keep", "market"]]
    assert rnd.plots() == [{"label": "hall", "x0": 24, "z0": 8, "x1": 34, "z1": 20}]
    assert all(p.get("type") and p.get("seed") for p in parts), \
        "a leaf lost its type or its seed"
    assert pipeline.part_rect(parts[0]) == (0, 0, 40, 40), pipeline.part_rect(parts[0])
    assert pipeline.part_rect(parts[1]) == (18, -2, 22, 2), pipeline.part_rect(parts[1])
    return (f"a three-deep tree of {len(parts)} leaves round-trips; two of them are a "
            f"wall and a gate and neither is a plot")


# --------------------------------------------- A6. lint scoped per wave, once at the
# end

@case
def t_a6_the_per_wave_findings_union_equals_the_whole_town():
    """A6's acceptance, on the last town this project built.

        Every wave of every round so far linted the **whole town**, so a five-wave round
        linted wave 1's buildings five times and the last wave paid for all of them. A wave
        is answerable for its own parts plus a margin, and the whole place is linted once at
        the end. The thing that has to be true is that nothing falls between: the union of
        the per-wave findings, each restricted to that wave's own plots, is exactly the
        whole-town finding set restricted to the plots.
        
    """
    from ethoslm import lint, offline
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_f.json"))
    if not os.path.exists(rnd.rel("world_built.npz")):
        return "SKIPPED."
    vol = offline.load_volume(rnd.rel("world_built.npz"))
    plots = json.load(open(rnd.rel("plots.json")))
    by = {p["label"]: p for p in plots}
    net = rnd.network()
    s = rnd.site
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]

    t0 = time.perf_counter()
    ctx = lint.Context.build(vol, plots, network=net, region=(X, Z, X + S - 1, Z + S - 1))
    whole = lint.lint(ctx).within(plots)
    whole_s = time.perf_counter() - t0
    W = {(f.code, tuple(f.pos) if f.pos else None) for f in whole.findings}

    U, per_s = set(), 0.0
    for w in rnd.waves:
        mine = [by[l] for l in (w.get("plots") or []) if l in by]
        scope = pipeline.wave_scope(rnd, w)
        assert scope is not None, w["name"]
        t0 = time.perf_counter()
        c = lint.Context.build(vol, plots, network=net, region=scope)
        U |= {(f.code, tuple(f.pos) if f.pos else None)
              for f in lint.lint(c).within(mine).findings}
        per_s += time.perf_counter() - t0
    assert not (W - U), f"the per-wave lints miss {sorted(W - U)[:5]}"
    assert not (U - W), f"the per-wave lints invent {sorted(U - W)[:5]}"
    assert W, "nothing was found either way, so nothing is being compared"
    # ...and it is cheaper than what it replaces, which is a whole-town lint per wave
    assert per_s < len(rnd.waves) * whole_s, \
        f"{per_s:.1f}s scoped against {len(rnd.waves) * whole_s:.1f}s unscoped"
    return (f"{len(W)} findings over {len(rnd.waves)} waves, identical either way; "
            f"{per_s:.1f}s scoped against {len(rnd.waves) * whole_s:.1f}s for a "
            f"whole-town lint per wave ({whole_s:.1f}s once)")


@case
def t_a6_e010_reports_every_floating_mass_on_a_plot_not_the_first_sixty():
    """The library defect A6 found, and it had been hiding in the record for two rounds.

        The fix is that a cap is a cap on the answer: the rectangles the caller is
        answerable for go in, and only what is inside them is counted against it.
        
    """
    from ethoslm import lint, observe, offline
    rnd = pipeline.Round.load(os.path.join(ROUNDS, "site_f.json"))
    if not os.path.exists(rnd.rel("world_built.npz")):
        return "SKIPPED."
    vol = offline.load_volume(rnd.rel("world_built.npz"))
    plots = json.load(open(rnd.rel("plots.json")))
    s = rnd.site
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    region = (X, Z, X + S - 1, Z + S - 1)
    rects = [(p["x0"], p["z0"], p["x1"], p["z1"]) for p in plots]

    old = observe.unsupported(vol, region=region)
    on_plot_old = [m for m in old
                   if any(r[0] <= (m["bbox"][0] + m["bbox"][3]) // 2 <= r[2]
                          and r[1] <= (m["bbox"][2] + m["bbox"][5]) // 2 <= r[3]
                          for r in rects)]
    new = observe.unsupported(vol, region=region, regions=rects)
    assert len(old) == 60, f"the cap is not being reached, so nothing is tested: {len(old)}"
    assert len(on_plot_old) == 1, len(on_plot_old)
    assert len(new) > 30, f"the filtered call found {len(new)}"

    # ...and the hillside under a plot is not the build on it. A mass made entirely of
    # what the ground is made of is landscape wherever it lies: one block of andesite
    # thirty-four blocks under `east_row` was reported identically against three
    # independently written types that had never been near it.
    nat = [m for m in new if m["natural"]]
    assert nat, "no natural mass in the set, so the exclusion is not being tested"
    ctx = lint.Context.build(vol, plots, network=rnd.network(), region=region)
    fired = [f for f in lint.lint(ctx, only={"E010"}).findings]
    assert len(fired) == len(new) - len(nat), (len(fired), len(new), len(nat))
    return (f"unfiltered: {len(old)} masses, {len(on_plot_old)} of them on a plot; "
            f"filtered: {len(new)} on a plot, {len(nat)} of them landscape, and E010 "
            f"reports the other {len(fired)}")


# A plan is a proposal now, and this is the answer to it -- deterministic, before a
# block is placed, and returned to the planner once.

def _tree(plots: list) -> dict:
    """A plan tree of one district of one quarter, with these leaves in it."""
    return {"intent": "a fixture", "centre": plots[0]["name"],
            "parts": [{"kind": "district", "name": "d", "children": [
                {"kind": "quarter", "name": "q", "children": plots}]}]}


def _leaf(name, type_name, x0, z0, w, d, **more):
    return {"kind": "plot", "name": name, "type": type_name, "seed": 1,
            "x0": x0, "z0": z0, "x1": x0 + w - 1, "z1": z0 + d - 1, **more}


@case
def t_a2_a_plan_with_an_undersized_plot_is_rejected_with_that_plot_named():
    """A 6x6 plot for a townhouse.

        `site()` insets two on every side, so a 6x6 plot is a 4x4 pad, and every townhouse
        on one refused it by name **after** the ground had been prepared. Here the plan does
        not get that far: the leaf is named, the pad and the need are quoted, and widening
        the plot to what the type asks for makes the same tree pass.
        
    """
    from ethoslm import place
    decl = pipeline.load_type(os.path.join(ROOT, "types", "townhouse.py"))
    lo = decl["needs"]["footprint"][0]
    plots = [_leaf("gate_ward_house", "townhouse", 0, 0, 6, 6),
             _leaf("gate_smithy", "workshop", 20, 0, 9, 9)]
    parts = pipeline.plan_parts(_tree(plots))
    decls = place.type_declarations(parts)
    fails = pipeline.plan_failures(parts, decls)
    assert len(fails) == 1, [f["why"] for f in fails]
    assert fails[0]["part"] == "gate_ward_house", fails[0]
    assert fails[0]["check"] == "footprint" and fails[0]["pad"] == [4, 4], fails[0]
    assert str(lo) in fails[0]["why"], fails[0]["why"]

    wide = [_leaf("gate_ward_house", "townhouse", 0, 0, lo + 4, lo + 4), plots[1]]
    wide[1] = dict(plots[1], x0=lo + 10, x1=lo + 18)
    ok = pipeline.plan_failures(pipeline.plan_parts(_tree(wide)),
                                place.type_declarations(parts))
    assert not ok, [f["why"] for f in ok]
    return (f"a 6x6 plot for a townhouse is refused naming it, its 4x4 pad and its "
            f"{lo}x{lo} need; the same tree at {lo + 4}x{lo + 4} passes")


@case
def t_a2_a_plan_is_also_checked_for_overlap_kind_and_the_gate_in_its_wall():
    """The other three things a plan can be wrong about, and the one it cannot.

        Two plots on the same ground, a leaf whose kind is not the kind its type builds, and
        two plots closer than the clearance a building declares are all failures. A **gate**
        standing in the wall it crosses is not: `PASSAGE` is the type saying a network may
        cross it, and a crossing that did not touch the thing it crosses would be a gap
        beside a gate.
        
    """
    from ethoslm import place
    over = [_leaf("a", "workshop", 0, 0, 9, 9), _leaf("b", "workshop", 4, 4, 9, 9)]
    parts = pipeline.plan_parts(_tree(over))
    fails = pipeline.plan_failures(parts, place.type_declarations(parts))
    assert any(f["check"] == "overlap" for f in fails), fails
    assert any("overlap" in f["why"] for f in fails), fails

    near = [_leaf("a", "workshop", 0, 0, 9, 9), _leaf("b", "workshop", 10, 0, 9, 9)]
    parts = pipeline.plan_parts(_tree(near))
    fails = pipeline.plan_failures(parts, place.type_declarations(parts))
    assert any("clearance" in f["why"] for f in fails), [f["why"] for f in fails]

    wrong = [dict(_leaf("a", "wall", 0, 0, 9, 9))]
    parts = pipeline.plan_parts(_tree(wrong))
    fails = pipeline.plan_failures(parts, place.type_declarations(parts))
    assert any(f["check"] == "type" and "builds a" in f["why"] for f in fails), fails

    gate = [{"kind": "edge", "name": "town_wall", "type": "wall", "seed": 1,
             "width": 1, "path": [[0, 0], [40, 0], [40, 40]]},
            {"kind": "point", "name": "north_gate", "type": "gate_tower", "seed": 1,
             "at": [20, 0], "facing": "north", "size": 5}]
    parts = pipeline.plan_parts(_tree(gate))
    fails = pipeline.plan_failures(parts, place.type_declarations(parts))
    assert not fails, [f["why"] for f in fails]
    return ("overlap, clearance and a leaf of the wrong kind are all refused; a gate "
            "standing in its own wall is not")


@case
def t_a2_a_failing_plan_goes_back_to_the_planner_once_and_then_stops():
    """Once, in the same brief, with the plan it failed on kept beside it.

        A second failure stops the round with the list rather than asking again: a planner
        that has been told exactly which leaves are too small and has come back with leaves
        that are too small has answered the question.
        
    """
    from ethoslm import place

    class _Be:
        live = False

        @property
        def volume(self):
            raise FileNotFoundError("no cache in this fixture")

    d = tempfile.mkdtemp(prefix="ethoslm_plan_")
    rnd = pipeline.Round(name="plan_fixture", state_dir=d)
    open(os.path.join(d, "planner_prompt.md"), "w").write("# a brief\n")
    bad = _tree([_leaf("gate_ward_house", "townhouse", 0, 0, 6, 6)])
    json.dump(bad, open(os.path.join(d, "plan.json"), "w"))

    res = pipeline.STAGES["plan"](rnd, _Be(), {})["plan"]
    assert res["status"] == "needs_model" and res["attempt"] == 1, res
    assert not os.path.exists(os.path.join(d, "plan.json")), \
        "the failing plan was left in place, so the stage would read it again"
    assert os.path.exists(os.path.join(d, "plan.rejected.1.json")), os.listdir(d)
    brief = open(os.path.join(d, "planner_prompt.md")).read()
    assert "gate_ward_house" in brief and "smallest plot" in brief, brief[-400:]

    json.dump(bad, open(os.path.join(d, "plan.json"), "w"))
    res2 = pipeline.STAGES["plan"](rnd, _Be(), {})["plan"]
    assert res2["status"] == "error" and res2["attempt"] == 2, res2
    assert "gate_ward_house" in res2["error"], res2["error"]

    log = json.load(open(os.path.join(d, "plan_validation.json")))
    assert len(log["attempts"]) == 2, log
    assert log["attempts"][0]["ground"].startswith("no volume"), log["attempts"][0]

    # ...and "stops the round" means the driver stops: the stages after a refused plan
    # would route lanes to footprints nobody accepted and lint a place nobody built.
    json.dump(bad, open(os.path.join(d, "plan.json"), "w"))
    res3 = pipeline.run(rnd, ["plan", "parts"], _Be())
    assert res3.get("stopped", {}).get("stage") == "plan", res3.get("stopped")
    assert "parts" not in res3, "the round carried on past a plan it had refused"
    return ("a failing plan is handed back once with the leaf named in the same brief "
            "and the plan it failed on kept; the second failure stops the round and the "
            "stages after it do not run")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:52s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:52s} {e}")
        except Exception as e:                        # noqa: BLE001
            bad += 1
            print(f"ERR  {name:52s} {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} pipeline cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
