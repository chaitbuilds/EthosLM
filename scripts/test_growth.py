"""The library grows when the spec asks for a form it lacks. v2, C3.

    $PY scripts/test_growth.py

  1. **The gap is detected by rule.** A defining part of a leaf kind with no committed
     type of its kind named for its family, in the place's form, is a gap; a compound,
     a district, a family a type is named for (with a suffix) and every fixture spec
     the other suites plan are not.
  2. **A spec admitting no type yields the ask.** The plan stage on such a spec stops
     at a blinded authoring: a `needs_model` of role `type`, the brief on disk with
     the API, the voice, the contract for the kind, the default fixtures and the
     request composed from the part; the checker in the author's directory; the record
     in `growth.json`. Re-entered without an answer it asks the same thing once.
  3. **Capped per run.** A third gap in a run that has asked two is refused by name.
  4. **The authored type passes the gate, or does not.** The author's answer -- written
     blind, checked by the author's own `check.py`, marked done -- is adopted into
     `types/` when every instance is clean in both voices, the gap closes and the plan
     goes on; an answer that fails the gate is recorded with its findings, the run stops
     by name, and the model is not asked twice.

nothing here needs a model call.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import growth, offline, pipeline, spec as spec_mod  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# ------------------------------------------------------------------ the fixture

SENTENCE = "Build a village round a manor house, with a watchtower beside it."
SIZE = 192
GROUND = 64
VOICE = "white_render_dark_frame"

#: A committed plot type whose source stands in for the author's answer: the proof is of
#: the machinery -- the gap, the ask, the checker, the gate, the adoption -- and what
#: the author writes is the model's business. This one is clean on the default fixtures
#: in both voices, which is what the gate asks.
CLEAN_ANSWER = "temple"


def _doc(parts=None):
    return {"kind": "village", "form": "east_asian", "voice": None,
            "defining_parts": parts or [
                {"name": "manor", "kind": "plot", "family": "house", "relation": "centre",
                 "count": 1, "structures": 1, "notes": "the manor house at the middle"},
                {"name": "watchtower", "kind": "point", "family": "tower",
                 "relation": "beside_the_centre", "count": 1,
                 "notes": "a watchtower beside the manor"},
                {"name": "houses", "kind": "group", "family": "district",
                 "relation": "throughout", "count": 1, "structures": 12,
                 "character": {}}]}


def _spec(parts=None):
    return spec_mod.read_spec(json.loads(json.dumps(_doc(parts))), SENTENCE)


def _flat(size=SIZE, y=GROUND):
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


def _round(tmp, spec, flags=None):
    """A round with a config on disk (the checker's job names it) and a state under
    `out/`, on flat ground, with the spec written as its place spec."""
    state = os.path.join(ROOT, "out", "growth_fixture")
    shutil.rmtree(state, ignore_errors=True)
    os.makedirs(state, exist_ok=True)
    offline.save_volume(_flat(), os.path.join(state, "world.npz"))
    site = {"origin": [0, 0], "size": SIZE}
    json.dump({**site, "stats": {"min": GROUND, "max": GROUND, "relief": 0, "std": 0.0},
               "mean_grid": [[GROUND] * 12 for _ in range(12)],
               "roughness_grid": [[0] * 12 for _ in range(12)],
               "surface_blocks": {"grass_block": 100}},
              open(os.path.join(state, "site.json"), "w"))
    json.dump(spec, open(os.path.join(state, "place.json"), "w"))
    cfg = os.path.join(tmp, "growth_fixture.json")
    json.dump({"name": "growth_fixture", "state_dir": state, "sentence": SENTENCE,
               "site": site, "flags": {"dry_run": True, "growth_sweep": False,
                                       **(flags or {})}},
              open(cfg, "w"), indent=1)
    rnd = pipeline.Round.load(cfg)
    # the author's scratch directories are keyed by the round's name and are outside its
    # state, so a previous run's answer would be adopted as this run's
    from ethoslm.pipeline import blind
    for n in ("house", "tower", "bridge"):
        shutil.rmtree(blind._blind_dir(rnd, os.path.join("growth", n), 0),
                      ignore_errors=True)
    return rnd, pipeline.OfflineBackend(rnd, dry_run=True)


def _types_on_disk():
    return sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "types"))
                  if f.endswith(".py"))


def _forget(*names):
    for n in names:
        p = os.path.join(ROOT, "types", f"{n}.py")
        if os.path.exists(p):
            os.remove(p)


# -------------------------------------------------------------- 1. the gap

@case
def t_1_a_gap_is_a_leaf_part_no_committed_type_of_its_kind_is_named_for():
    s = _spec()
    gaps = growth.type_gaps(s)
    assert [p["name"] for p in gaps] == ["manor", "watchtower"], gaps
    assert [growth.type_name(p) for p in gaps] == ["house", "tower"]
    # a family a type is named for, with a suffix, is not a gap; nor an area a type is
    # named for; nor a compound; nor a district
    s2 = _spec([
        {"name": "wall", "kind": "edge", "family": "wall", "relation": "perimeter",
         "count": 1},
        {"name": "gate", "kind": "point", "family": "gate", "relation": "gateway",
         "count": 1},
        {"name": "green", "kind": "area", "family": "square",
         "relation": "beside_the_centre", "count": 1},
        {"name": "palace", "kind": "plot", "family": "palace", "relation": "centre",
         "count": 1, "structures": 1, "needs": {"plateau": 64}},
        {"name": "houses", "kind": "group", "family": "district",
         "relation": "throughout", "count": 1, "structures": 12}])
    assert growth.type_gaps(s2) == [], [p["name"] for p in growth.type_gaps(s2)]
    # ...and the same family as a point is a gap where only a plot is named for it: the
    # kind is the type's
    s3 = _spec([{"name": "moot", "kind": "point", "family": "hall",
                 "relation": "centre", "count": 1},
                {"name": "houses", "kind": "group", "family": "district",
                 "relation": "throughout", "count": 1, "structures": 12}])
    assert [p["name"] for p in growth.type_gaps(s3)] == ["moot"]
    # the fixture specs the other suites plan have no gap
    import test_compile
    import test_compounds
    assert growth.type_gaps(test_compounds._spec()) == []
    assert growth.type_gaps(test_compile._spec()) == []
    # a round that pins its types reads gaps against the pinned list
    assert [p["name"] for p in growth.type_gaps(s2, ["wall", "square"])] == ["gate"]
    # the request is composed from the part and the sentence
    req = growth.request_for(s, gaps[0])
    assert "types/house.py" in req and 'KIND = "plot"' in req and SENTENCE in req \
        and "the manor house at the middle" in req and "east_asian" in req
    req2 = growth.request_for(s, gaps[1])
    assert 'KIND = "point"' in req2 and "beside_the_centre" in req2
    return ("manor (house, plot) and watchtower (tower, point) are the gaps; a wall, a "
            "gate (gate_tower), a square, a palace compound and a district are not; a "
            "hall as a point is; the monument and compile fixtures have none; the "
            "request names the file, the kind, the form, the sentence and the notes")


# -------------------------------------------------------------- 2. the ask

@case
def t_2_a_spec_admitting_no_type_yields_the_ask_with_brief_checker_and_record():
    from ethoslm.pipeline import stages_plan
    if not all(os.path.isdir(os.path.join(ROOT, "out", r)) for r in ("site_b", "site_d",
                                                                    "site_e")):
        raise Skip("the default fixtures' rounds are not cached under out/")
    with tempfile.TemporaryDirectory() as tmp:
        spec = _spec()
        rnd, be = _round(tmp, spec)
        t0 = time.perf_counter()
        res = stages_plan.stage_plan(rnd, be, {})
        secs = time.perf_counter() - t0
        got = res["plan"]
        assert got["status"] == "needs_model" and got["role"] == "type", got
        assert got["level"] == "type/house", got["level"]
        bl = got["blinded"]
        assert bl.get("role") == "type", bl
        d = bl["dir"]
        assert os.path.exists(os.path.join(d, "brief.md"))
        assert os.path.exists(os.path.join(d, "check.py"))
        assert sorted(os.listdir(d)) == ["brief.md", "check.py"], sorted(os.listdir(d))
        brief = open(os.path.join(d, "brief.md")).read()
        assert "types/house.py" in brief and SENTENCE in brief
        assert "## The pieces of ground your function will be called on" in brief
        assert "check.py" in brief and "both" in brief
        rec = json.load(open(rnd.rel("growth.json")))
        assert rec["gaps"] == ["house", "tower"] and list(rec["asked"]) == ["house"]
        assert len(rec["asked"]["house"]["fixtures"]) == 6, rec["asked"]["house"]
        assert rec["asked"]["house"]["tspec"]["part"] == "plot"
        assert os.path.exists(rec["asked"]["house"]["brief"])
        # the checker's job carries the default fixtures and the two voices
        key = os.path.basename(d)
        from ethoslm.pipeline import blind
        job = json.load(open(os.path.join(blind.CHECK_JOBS, f"{key}.json")))
        assert job["kind"] == "type" and job["type"] == "house"
        assert len(job["fixtures"]) >= 6 and job["sweep"] is False, job
        assert [f"{f['round']}/{f['plot']}" for f in job["fixtures"][:6]] == \
            rec["asked"]["house"]["fixtures"], job["fixtures"][:6]
        assert all(f["round"].startswith("slope") for f in job["fixtures"][6:]), \
            job["fixtures"][6:]
        assert len(job["voices"]) == 2 and job["voices"][0] == job["voice"]
        # re-entered without an answer: the same ask, once
        res2 = stages_plan.stage_plan(rnd, be, {})
        assert res2["plan"]["level"] == "type/house" and res2["plan"]["blinded"]["dir"] == d
        rec2 = json.load(open(rnd.rel("growth.json")))
        assert list(rec2["asked"]) == ["house"]
        assert not os.path.exists(rnd.rel("voice.json")), "a voice was recorded early"
    return (f"the plan stage stops at a blinded authoring of `house` (role type) in "
            f"{secs:.1f}s: brief.md and check.py alone in the author's directory, the "
            f"brief carrying the request, the six default fixtures and the slopes in "
            f"the job with two voices; growth.json records the ask; re-entered, the "
            f"same ask once")


# -------------------------------------------------------------- 3. the cap

@case
def t_3_a_third_gap_in_a_run_that_asked_two_is_refused_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        spec = _spec([
            {"name": "manor", "kind": "plot", "family": "house", "relation": "centre",
             "count": 1, "structures": 1},
            {"name": "watchtower", "kind": "point", "family": "tower",
             "relation": "beside_the_centre", "count": 1},
            {"name": "crossing", "kind": "edge", "family": "bridge",
             "relation": "beside_the_centre", "count": 1},
            {"name": "houses", "kind": "group", "family": "district",
             "relation": "throughout", "count": 1, "structures": 12}])
        rnd, be = _round(tmp, spec)
        gaps = growth.type_gaps(spec)
        assert [growth.type_name(p) for p in gaps] == ["house", "tower", "bridge"]
        # a record that has asked its run's worth
        rec = growth.record_of(rnd)
        for n in ("house", "tower"):
            rec["asked"][n] = {"tspec": {"name": n}, "fixtures": []}
        growth._save(rnd, rec)
        res = growth.stage(rnd, be, spec, gaps[2:], site=json.load(open(rnd.rel("site.json"))),
                           voice=VOICE)
        assert res["plan"]["status"] == "error" and res["plan"]["stop"], res
        assert "bridge" in res["plan"]["error"] and str(growth.GROWTH_CAP) in \
            res["plan"]["error"], res["plan"]["error"]
        rec = json.load(open(rnd.rel("growth.json")))
        assert rec["capped"] == ["bridge"] and "bridge" not in rec["asked"]
    return (f"with {growth.GROWTH_CAP} types asked, a third gap (bridge) is refused by "
            f"name and recorded as capped, not asked")


# ------------------------------------------------------------ 4. the gate

@case
def t_4_the_authored_type_passes_the_gate_and_is_adopted_or_fails_and_is_not_asked_twice():
    from ethoslm.pipeline import stages_plan
    if not all(os.path.isdir(os.path.join(ROOT, "out", r)) for r in ("site_b", "site_d",
                                                                    "site_e")):
        raise Skip("the default fixtures' rounds are not cached under out/")
    before = _types_on_disk()
    assert "house" not in before and "tower" not in before, "the library already has them"
    said = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec()
            rnd, be = _round(tmp, spec)
            res = stages_plan.stage_plan(rnd, be, {})
            d = res["plan"]["blinded"]["dir"]
            # the author: a clean type's source, written blind, checked, marked done
            src = open(os.path.join(ROOT, "types", f"{CLEAN_ANSWER}.py")).read()
            open(os.path.join(d, "program.py"), "w").write(src)
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, os.path.join(d, "check.py")],
                               capture_output=True, text=True, cwd=d)
            check_s = time.perf_counter() - t0
            assert p.returncode == 0, (p.stdout[-800:], p.stderr[-800:])
            assert os.path.exists(os.path.join(d, "findings.md"))
            open(os.path.join(d, "done"), "w").write("done\n")
            # re-entered: adopted through the gate, the gap closed, planning goes on
            t0 = time.perf_counter()
            res = stages_plan.stage_plan(rnd, be, {})
            gate_s = time.perf_counter() - t0
            rec = json.load(open(rnd.rel("growth.json")))
            assert "house" in rec["adopted"], rec
            got = rec["adopted"]["house"]
            assert got["passes"] and got["file"] == "types/house.py", got
            assert os.path.exists(os.path.join(ROOT, "types", "house.py"))
            assert all(v["clean"] == v["instances"] > 0 for v in got["voices"].values()), \
                got["voices"]
            assert os.path.exists(got["findings"])
            assert [growth.type_name(q) for q in growth.type_gaps(spec)] == ["tower"]
            said.append(f"house: the author's answer ({CLEAN_ANSWER}'s source) checked "
                        f"by its own check.py in {check_s:.0f}s, adopted through the "
                        f"gate in {gate_s:.0f}s -- {got['instances']} instances clean "
                        f"in {list(got['voices'])} over {len(got['fixtures'])} "
                        f"fixtures, sweep {got['sweep']}")
            # ...and the next gap is asked in the same run: the tower
            assert res["plan"]["status"] == "needs_model" and \
                res["plan"]["level"] == "type/tower", res["plan"]
            d2 = res["plan"]["blinded"]["dir"]
            # an answer that fails the gate: a point type that builds nothing
            open(os.path.join(d2, "program.py"), "w").write(
                'KIND = "point"\nFORM = "fortification"\nROLE = "defensive"\n'
                'PARAMS = {}\nNEEDS = {"footprint": (3, 3, 15, 15), "frontage": "lane",'
                ' "ground": "any", "clearance": 1}\n\n\n'
                'def build(b, part, seed, **params):\n'
                '    raise RuntimeError("a tower nobody wrote")\n')
            p = subprocess.run([sys.executable, os.path.join(d2, "check.py")],
                               capture_output=True, text=True, cwd=d2)
            assert p.returncode == 0, (p.stdout[-800:], p.stderr[-800:])
            open(os.path.join(d2, "done"), "w").write("done\n")
            res = stages_plan.stage_plan(rnd, be, {})
            assert res["plan"]["status"] == "error" and res["plan"]["stop"], res["plan"]
            assert "tower" in res["plan"]["error"] and "gate" in res["plan"]["error"]
            rec = json.load(open(rnd.rel("growth.json")))
            assert "tower" in rec["failed"] and not rec["failed"]["tower"]["passes"]
            assert not os.path.exists(os.path.join(ROOT, "types", "tower.py"))
            # ...and it is not asked twice
            res = stages_plan.stage_plan(rnd, be, {})
            assert res["plan"]["status"] == "error" and "not asked twice" in \
                res["plan"]["error"], res["plan"]
            said.append(f"tower: an answer that crashes fails the gate "
                        f"({rec['failed']['tower']['why'][:60]}...), is not adopted, "
                        f"stops the run by name and is not asked twice")
    finally:
        _forget("house", "tower")
    assert _types_on_disk() == before, "the library was left changed"
    return "; ".join(said) + "; the library is as it was"


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
    print(f"\n{ok} of {ok + bad} growth cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
