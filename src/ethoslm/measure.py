"""Append-only measurement log. The project had no numbers at all before this.

**One schema.** `record` had grown 64 distinct kinds, roughly twenty of them invented
once for one experiment and never read again (`e1d_frame`, `step3_frame`,
`step4_frame`, `e1c_frame`, `e3_frame` are five names for "a frame was rendered"). A
log with a kind per script is a log nothing can be asked a question of across rounds,
which is how "how long does a frame take?" stayed unanswerable while 124 of them were
being rendered.

So there is now a fixed set of `KINDS`, each with the fields it is expected to carry,
and `record` rejects a kind that is not in it. The historical names are kept in
`LEGACY`, mapped to what they became: `out/measurements.jsonl` is 250-odd rows of
evidence and nothing rewrites it, so anything reading the log has to know both
vocabularies anyway. Passing a legacy name still works, still writes the legacy name,
and prints a note -- the archived experiment scripts must keep running.

`model_call` is untouched, deliberately. It is the only reason anything in this project
is costed.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

#: $ETHOSLM_MEASURE_LOG redirects the log. One use: a test that runs `check.py` as the
#: builder runs it -- a real subprocess, the real generated file -- without appending
#: fixture rows to the project's own evidence.
LOG = (os.environ.get("ETHOSLM_MEASURE_LOG")
       or os.path.join(os.path.dirname(__file__), "..", "..", "out",
                       "measurements.jsonl"))

#: The fixed vocabulary. Value is the fields a kind is expected to carry -- not
#: enforced, because a measurement with an extra field is still a measurement, but
#: written down so two callers of the same kind can be compared.
KINDS: dict[str, tuple] = {
    "model_call": ("stage", "chars_in", "chars_out", "tokens_in_est",
                   "tokens_out_est", "seconds"),
    "round": ("name", "stage", "seconds"),
    "stage_run": ("arm", "name", "pending", "seconds", "error"),
    "pass_run": ("name", "settlement", "placed", "failed", "seconds"),
    "frame": ("tag", "shot", "seconds", "mean", "status"),
    "judgement": ("name", "settlement", "judged", "correct", "wrong", "ties"),
    # A round-robin between siblings from one brief has no correct side, so it cannot be
    # a `judgement`: what is being measured is whether the judge can separate them at
    # all, and then which one it chose. One new kind, added as a decision.
    "selection": ("name", "settlement", "judged", "ties", "tie_rate", "winner"),
    "check": ("name", "settlement", "errors", "warnings", "seconds"),
    # One run of `check.py` by a builder, inside its own call. A distinct kind because
    # it answers a question no other row can: how many times a builder looked before it
    # finished. `check` is the harness checking a build; this is the build checking
    # itself, and reading them as one number would hide exactly the thing the round says
    # to read first.
    "check_run": ("key", "settlement", "wave", "sha256", "blocks", "writes",
                  "errors", "entry_lines", "crashed", "seconds"),
    "survey": ("name", "sites", "seconds"),
    "world_io": ("op", "blocks", "seconds"),
    "experiment": ("name", "settlement"),
}

#: What each retired kind became. Left as data rather than as a migration because the
#: log is append-only evidence and rewriting it would destroy the thing it is for.
LEGACY: dict[str, str] = {
    "e1a": "judgement", "e1c": "judgement", "e1d": "judgement", "e3": "judgement",
    "step3": "judgement", "step4_checks": "judgement", "model_bake_off": "judgement",
    "e1c_frame": "frame", "e1d_frame": "frame", "e3_frame": "frame",
    "step3_frame": "frame", "step4_frame": "frame", "render_build": "frame",
    "render_views": "frame", "render_plan": "frame", "preview": "frame",
    "settlement_pass": "pass_run", "circulation_pass": "pass_run",
    "finish_pass": "pass_run", "step4_wave": "pass_run",
    "step3_arm_rendered": "pass_run", "model_build": "pass_run",
    "lint_town": "check", "lint_settlement": "check", "lint_cached": "check",
    "observe_town": "check", "defect_scan": "check", "hole_scan": "check",
    "mutation_scan": "check", "step4_checks_lint": "check",
    "block_update_test": "check", "wall_join_sweeps": "check",
    "prims_smoke_test": "check", "nav_course": "check", "api_use": "check",
    "site_survey": "survey", "hard_site_survey": "survey",
    "sites_prepared": "survey", "settlement_site_prepared": "survey",
    "step4_site_probe": "survey",
    "place_blocks": "world_io", "place_blocks_bulk": "world_io",
    "worldslice_load": "world_io", "cache_world": "world_io", "cache_town": "world_io",
    "lint_scaling": "check", "observe_fix": "check", "palette_recount": "check",
    "regress_wave": "check", "s001_s002_dead_on_town": "check",
    "render_coverage": "frame", "render_settlement": "frame",
    "render_three_views": "frame",
    "local_model_bench": "experiment", "rewrite": "experiment",
    "e1a_mutant": "experiment", "e1c_mutant": "experiment", "e1d_mutant": "experiment",
    "e3_arrangement": "experiment", "step4_prereg": "experiment",
    "step4_abstractions": "experiment", "variety": "experiment",
    "preflight_reject": "experiment", "circulation_reject": "experiment",
}


class UnknownKind(ValueError):
    """A kind that is neither in `KINDS` nor a known legacy name. Deliberately an
    error: a new one-off kind is exactly the growth this schema exists to stop, and
    adding one should be a decision rather than a typo."""


def record(kind: str, **fields) -> None:
    if kind not in KINDS:
        if kind in LEGACY:
            print(f"[measure] note: '{kind}' is retired; it is now "
                  f"'{LEGACY[kind]}' -- writing the legacy name")
        else:
            raise UnknownKind(
                f"'{kind}' is not a measurement kind. Use one of "
                f"{sorted(KINDS)}, or add it to KINDS if the project really needs "
                f"a new one.")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **fields}
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[measure] {row}")


def model_call(stage: str, prompt: str, response: str, seconds: float,
               **fields) -> None:
    """One model call, costed. Chars are what we can actually measure -- there is no
    API key in this environment, so no usage object -- and the 4-chars/token figure is
    an estimate and is labelled as one. In 219 records across 30 kinds there was not
    one token count; every experiment from here on is costed."""
    record("model_call", stage=stage, chars_in=len(prompt), chars_out=len(response),
           tokens_in_est=len(prompt) // 4, tokens_out_est=len(response) // 4,
           seconds=round(seconds, 2), **fields)


@contextmanager
def timed(kind: str, **fields):
    t0 = time.perf_counter()
    box = {}
    try:
        yield box
    finally:
        record(kind, seconds=round(time.perf_counter() - t0, 3), **fields, **box)
