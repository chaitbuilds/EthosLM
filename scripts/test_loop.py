"""The loop before the city: a preview stage after the plan and before the parts. v2, C4.

    $PY scripts/test_loop.py

  1. **The stage on the example.** The map, the landmark and five representative
     buildings are drawn to disk from the recorded plan, under a minute and with no
     model call; the stage stops at the judge's reading -- a `needs_model` of role
     `judge` carrying the three pictures and a file to write -- and, the reading
     written, a plan with no spec has nothing to revise and the stage is done.
  2. **One bounded revision.** On a place planned from a spec with a compiled district:
     the reading answered, the stage asks the principal (role `spec`) for one revision
     of the characters or the voice, with the reading, the characters as written, the
     vocabulary and the voices on disk; a revision naming a district's character and a
     voice is applied -- the spec on disk changed, the district recompiled from the new
     character, the voice recorded, the plan laid out again and redrawn -- with what it
     refuses named; and a second revision is not asked for: the build follows.

Nothing here needs a model call or a cached world beyond `out/example`.
"""
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import model as model_mod, offline, pipeline, spec as spec_mod  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# ---------------------------------------------------------------- 1. the example

@case
def t_1_the_stage_on_the_example_draws_in_seconds_and_stops_at_the_reading():
    ex = os.path.join(ROOT, "out", "example")
    if not os.path.exists(os.path.join(ex, "plan.json")):
        raise Skip("no out/example/plan.json")
    rnd = pipeline.Round.load(os.path.join(ROOT, "rounds", "example.json"))
    be = pipeline.OfflineBackend(rnd, dry_run=True)
    shutil.rmtree(rnd.rel("preview"), ignore_errors=True)
    t0 = time.perf_counter()
    res = pipeline.STAGES["preview"](rnd, be, {})
    secs = time.perf_counter() - t0
    assert secs < 60, secs
    rec = res["preview"]
    drawn = rec["drawn"][""]
    for k in ("map", "landmark", "buildings"):
        assert os.path.exists(drawn["images"][k]), k
    assert os.path.exists(rnd.rel("preview", "preview.json"))
    assert drawn["landmark"]["type"] and drawn["landmark"]["name"]
    assert 1 <= len(drawn["buildings"]) <= 5
    assert drawn["buildings"] == sorted(drawn["buildings"],
                                        key=lambda b: (-b["leaves"], b["type"]))
    ask = res["reading"]
    assert ask["status"] == "needs_model" and ask["role"] == "judge", ask
    assert len(ask["images"]) == 3 and ask["write"].endswith("reading.md")
    assert list(model_mod.staged(res)) == [ask]
    prompt = open(ask["request"]).read()
    assert drawn["landmark"]["name"] in prompt and "what would they see wrong" in prompt
    assert "recorded, not planned from a spec" in prompt
    # re-entered: the same ask, and nothing redrawn
    res2 = pipeline.STAGES["preview"](rnd, be, {})
    assert res2["reading"]["request"] == ask["request"]
    assert res2["preview"]["drawn"][""]["seconds"] == drawn["seconds"]
    # the reading written: a recorded plan has nothing to revise, and the stage is done
    open(ask["write"], "w").write("A reading.\n")
    res3 = pipeline.STAGES["preview"](rnd, be, {})
    assert res3["preview"].get("done") and "revision" not in res3, res3
    assert list(model_mod.staged(res3)) == []
    return (f"drawn in {drawn['seconds']['all']}s (map {drawn['seconds']['map']}s, "
            f"landmark {drawn['landmark']['name']} ({drawn['landmark']['type']}) "
            f"{drawn['seconds']['landmark']}s, buildings "
            f"{[b['type'] for b in drawn['buildings']]} {drawn['seconds']['buildings']}s); "
            f"the stage {secs:.1f}s; stops at the judge's reading with three pictures; "
            f"re-entered it asks the same; read, it is done")


# ------------------------------------------------------------ 2. the revision

SENTENCE = "Build a town of street houses round a green."
SIZE = 200
GROUND = 64


def _flat(size=SIZE, y=GROUND):
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


def _doc():
    return {"kind": "town", "form": "european_vernacular", "voice": None,
            "defining_parts": [
                {"name": "green", "kind": "area", "family": "square",
                 "relation": "centre", "count": 1, "notes": "the green at the middle"},
                {"name": "houses", "kind": "group", "family": "district",
                 "relation": "throughout", "count": 1, "structures": 40,
                 "density": "medium",
                 "character": {"landmarks": [{"type": "hall"}]},
                 "notes": "the street houses"}]}


def _round(tmp):
    state = os.path.join(ROOT, "out", "loop_fixture")
    shutil.rmtree(state, ignore_errors=True)
    os.makedirs(state, exist_ok=True)
    offline.save_volume(_flat(), os.path.join(state, "world.npz"))
    site = {"origin": [0, 0], "size": SIZE}
    json.dump({**site, "stats": {"min": GROUND, "max": GROUND, "relief": 0, "std": 0.0},
               "mean_grid": [[GROUND] * 12 for _ in range(12)],
               "roughness_grid": [[0] * 12 for _ in range(12)],
               "surface_blocks": {"grass_block": 100}},
              open(os.path.join(state, "site.json"), "w"))
    json.dump(_doc(), open(os.path.join(state, "place.json"), "w"))
    cfg = os.path.join(tmp, "loop_fixture.json")
    json.dump({"name": "loop_fixture", "state_dir": state, "sentence": SENTENCE,
               "site": site, "flags": {"dry_run": True}}, open(cfg, "w"), indent=1)
    rnd = pipeline.Round.load(cfg)
    return rnd, pipeline.OfflineBackend(rnd, dry_run=True)


@case
def t_2_one_bounded_revision_of_the_characters_or_the_voice_and_then_the_build():
    from ethoslm.pipeline import stages_plan
    with tempfile.TemporaryDirectory() as tmp:
        rnd, be = _round(tmp)
        t0 = time.perf_counter()
        res = stages_plan.stage_plan(rnd, be, {})
        plan_s = time.perf_counter() - t0
        assert not list(model_mod.staged(res)) and not res.get("plan", {}).get("stop"), res
        assert os.path.exists(rnd.rel("plan.json"))

        def compiled():
            """The compiled records of every district drawn for `houses`, summed."""
            recs = [json.load(open(rnd.rel(f))) for f in sorted(os.listdir(rnd.state))
                    if f.startswith("district_houses") and f.endswith("_compiled.json")]
            assert recs and all(r["part"] == "houses" for r in recs), recs
            return {"districts": len(recs),
                    "courtyard_blocks": sum(r["block_kinds"]["courtyard"] for r in recs),
                    "courts": sum(r["courts"] for r in recs),
                    "lots": sum(r["lots"] for r in recs),
                    "courtyard_share": {r["character"]["courtyard_share"] for r in recs}}

        before = compiled()
        voice0 = rnd.voice_name()
        # the preview: drawn, and the reading asked
        t0 = time.perf_counter()
        res = pipeline.STAGES["preview"](rnd, be, {})
        prev_s = time.perf_counter() - t0
        ask = res["reading"]
        assert ask["role"] == "judge" and len(ask["images"]) == 3
        drawn = res["preview"]["drawn"][""]
        assert drawn["landmark"]["type"] == "hall", drawn["landmark"]
        prompt = open(ask["request"]).read()
        assert "compiled as" in prompt and "landmarks" in prompt, prompt[-1200:]
        # the reading answered: the principal is asked for one revision
        open(ask["write"], "w").write(
            "The blocks read as detached houses on lawns; the hall is right.\n")
        res = pipeline.STAGES["preview"](rnd, be, {})
        ask2 = res["revision"]
        assert ask2["status"] == "needs_model" and ask2["role"] == "spec", ask2
        assert ask2["write"].endswith("revision.json") and len(ask2["images"]) == 3
        prompt2 = open(ask2["request"]).read()
        assert "detached houses on lawns" in prompt2
        assert "courtyard_share" in prompt2 and "drystone_and_thatch" in prompt2
        assert f"`{voice0}`" in prompt2
        # re-entered before the answer: the same ask
        assert pipeline.STAGES["preview"](rnd, be, {})["revision"]["request"] == \
            ask2["request"]
        # the revision: a character and a voice, plus two things it may not do
        json.dump({"characters": {"houses": {"courtyard_share": 1.0, "open_share": 0.0,
                                             "landmarks": [{"type": "hall"}]},
                                  "nowhere": {"attached": True},
                                  "green": {"attached": True}},
                   "voice": "drystone_and_thatch",
                   "why": "courts behind every row, and the hillside's own stone"},
                  open(ask2["write"], "w"))
        t0 = time.perf_counter()
        res = pipeline.STAGES["preview"](rnd, be, {})
        apply_s = time.perf_counter() - t0
        rec = res["preview"]
        assert rec.get("done") and rec["revisions"] == 1, rec
        applied = rec["applied"]
        assert list(applied["characters"]) == ["houses"], applied
        assert applied["characters"]["houses"]["courtyard_share"] == 1.0
        assert applied["voice"] == "drystone_and_thatch"
        assert any("nowhere" in r for r in applied["refused"]) and \
            any("green" in r for r in applied["refused"]), applied["refused"]
        assert applied.get("plan", {}).get("status") not in ("error", "needs_model"), \
            applied.get("plan")
        # the spec on disk carries the character; the district was recompiled from it;
        # the voice is the place's; the plan is laid out again; the pictures drawn again
        spec = rnd.place_spec()
        houses = next(p for p in spec["defining_parts"] if p["name"] == "houses")
        assert houses["character"]["courtyard_share"] == 1.0, houses["character"]
        after = compiled()
        # the spec carries what the principal wrote; each district compiles from it and
        # may lower it again to hold its own count (v2, C5), so what is asserted here is
        # that the revision reached the districts and moved the fabric ...and under the
        # craft round's fabric (E1) a district's count is high enough that every one of
        # them gives some of the share back, so what is held here is that the
        # principal's number reached the compiler and moved it up, not that any district
        # kept the whole of it
        assert max(after["courtyard_share"]) > max(before["courtyard_share"]), \
            (before, after)
        assert after["courtyard_blocks"] > before["courtyard_blocks"], (before, after)
        assert after["courts"] > before["courts"], (before, after)
        assert rnd.voice_name() == "drystone_and_thatch"
        assert json.load(open(rnd.rel("plan.place.json")))["voice"] == "drystone_and_thatch"
        assert json.load(open(rnd.rel("voice.json")))["voice"] == "drystone_and_thatch"
        assert os.path.exists(rnd.rel("plan.json"))
        assert all(os.path.exists(rec["drawn"]["_1"]["images"][k])
                   for k in ("map", "landmark", "buildings"))
        assert not list(model_mod.staged(res))
        # ...and the loop is bounded: asked again, it is done and asks nothing
        res = pipeline.STAGES["preview"](rnd, be, {})
        assert res["preview"].get("done") and not list(model_mod.staged(res))
        assert res["preview"]["revisions"] == 1
    return (f"planned and compiled in {plan_s:.1f}s; drawn and read in {prev_s:.1f}s; "
            f"the principal asked once with the reading, the characters and the voices; "
            f"a revision applied in {apply_s:.1f}s -- houses' courtyard share 1.0 on "
            f"the spec and {sorted(after['courtyard_share'])} over "
            f"{after['districts']} districts after each held its own count (courtyard "
            f"blocks "
            f"{before['courtyard_blocks']} -> {after['courtyard_blocks']}, courts "
            f"{before['courts']} -> {after['courts']}), voice {voice0} -> "
            f"drystone_and_thatch; two things it "
            f"may not change refused by name; redrawn; asked again it is done")


def main():
    ok = bad = skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok} of {ok + bad} loop cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
