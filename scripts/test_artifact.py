#!/usr/bin/env python3
"""**The delivered artifact is explicit** (`ethoslm.artifact`).

    $PY scripts/test_artifact.py [name-fragment]

Offline, a temporary state directory each, no server:

    finished    a real scene built through `palettesheet.build` in a voice with a
                variants recipe, its surfaces written; `stage_material` writes and adopts
                `world_finished.npz`; `stage_write` (live backend stubbed, the diff
                captured) writes that volume and records it; `ca_eye` given the state
                directory draws it and says so
    fallback    no adoption: the structural world, a newer finished world, an older
                finished world; `_adopt_structural` (stage_finish's) adopts the built
                world as structural
    mismatch    the adopted file changed after adoption is detected and reported, and
                a finished adoption goes stale when the built world changes under it
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from ethoslm import artifact, offline  # noqa: E402

CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


class _Rnd:
    def __init__(self, state, flags=None):
        self.state = state
        self.flags = dict(flags or {})
        self.shots = {}
        self.name = "test_artifact"
        self.site = {"origin": [0, 0], "size": 64}

    def rel(self, *parts):
        return os.path.join(self.state, *parts)


class _Be:
    live = True
    dry_run = False


def _tiny(value: int):
    from ethoslm.observe import Volume
    codes = np.zeros((8, 8, 8), np.uint16)
    codes[:, :3, :] = value
    return Volume(0, 60, 0, codes, ["air", "stone", "dirt"])


def _tmp() -> str:
    d = tempfile.mkdtemp(prefix="ethoslm-artifact-")
    return d


@case
def finished_adopted_is_written_and_viewed():
    from ethoslm import palettesheet, surfaces
    from ethoslm.pipeline import inspect as insp, stages_media
    from ethoslm import pipeline as _pipeline
    st = _tmp()
    try:
        # a real built world and its surface record, in a voice with a recipe
        from ethoslm import material
        voice = "hutong_grey_brick_tile"
        assert material.recipe_for(voice), "the test voice carries no recipe"
        res = palettesheet.build("mansion", voice, finish=False)
        built = res["built"]
        offline.save_volume(built, os.path.join(st, "world_built.npz"))
        offline.save_volume(palettesheet._flat(), os.path.join(st, "world.before-plateau.npz"))
        # the surface record, recorded the way `palettesheet.build` does it
        recs = _records(voice)
        assert recs, "no surface records"
        surfaces.write(os.path.join(st, "surfaces.json"), recs)
        rnd = _Rnd(st, {"material": True, "seed": 1})
        got = insp.stage_material(rnd, None, {})
        assert got.get("written"), got
        rec = json.load(open(os.path.join(st, "delivered.json")))
        assert rec["kind"] == "finished" and rec["path"] == "world_finished.npz", rec
        r = artifact.resolve(st)
        assert r["source"] == "adopted" and r["kind"] == "finished", r
        # the write reads the adopted artifact
        seen = {}

        def fake_diff(pre, vol, x0, z0, x1, z1):
            seen["codes"] = vol.codes.copy()
            seen["palette"] = list(vol.palette)
            return []
        old = _pipeline.region_diff
        _pipeline.region_diff = fake_diff
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                w = stages_media.stage_write(rnd, _Be(), {})
        finally:
            _pipeline.region_diff = old
        assert w["artifact"]["kind"] == "finished", w
        fin = offline.load_volume(os.path.join(st, "world_finished.npz"))
        assert np.array_equal(seen["codes"], fin.codes), "the write read another volume"
        assert [str(s) for s in seen["palette"]] == [str(s) for s in fin.palette]
        differs = not (np.array_equal(fin.codes, built.codes)
                       and list(fin.palette) == list(built.palette))
        assert differs, "the finished world is the built world; the case proves nothing"
        # the eye view, given the state directory
        import ca_eye
        out = io.StringIO()
        png = os.path.join(st, "eye.png")
        with contextlib.redirect_stdout(out):
            ca_eye.main([st, "--at", "10,80,10", "--look", "30,64,30", "--png", png,
                         "--size", "64x40"])
        said = out.getvalue()
        assert "world_finished.npz" in said and "adopted" in said, said
        return (f"stage_material adopted world_finished.npz ({got.get('substituted')} "
                f"substitutions); stage_write wrote it; ca_eye drew it and said so")
    finally:
        shutil.rmtree(st, ignore_errors=True)


def _records(voice):
    """The mansion scene's surface records (`palettesheet.build` keeps them inside)."""
    from ethoslm import palettesheet, surfaces
    got = []
    orig = surfaces.record

    def spy(*a, **k):
        r = orig(*a, **k)
        got.append(r)
        return r
    surfaces.record = spy
    try:
        palettesheet.build("mansion", voice, finish=False)
    finally:
        surfaces.record = orig
    return got


@case
def structural_fallback_and_stage_finish_adoption():
    from ethoslm.pipeline.stages_build import _adopt_structural
    st = _tmp()
    try:
        b = os.path.join(st, "world_built.npz")
        f = os.path.join(st, "world_finished.npz")
        offline.save_volume(_tiny(1), b)
        r = artifact.resolve(st)
        assert r["source"] == "fallback" and r["path"] == b and r["kind"] == "structural", r
        # a finished world newer than the built one, nothing adopted: the finished one
        time.sleep(0.02)
        offline.save_volume(_tiny(2), f)
        r = artifact.resolve(st)
        assert r["path"] == f and r["kind"] == "finished", r
        # ...and older than a rebuilt structural world: not used
        time.sleep(0.02)
        offline.save_volume(_tiny(1), b)
        os.utime(f, (os.path.getmtime(b) - 10, os.path.getmtime(b) - 10))
        r = artifact.resolve(st)
        assert r["path"] == b and "older" in r["note"], r
        # stage_finish's adoption: structural, explicit
        with contextlib.redirect_stdout(io.StringIO()):
            rec = _adopt_structural(_Rnd(st))
        assert rec["kind"] == "structural" and rec["path"] == "world_built.npz", rec
        r = artifact.resolve(st)
        assert r["source"] == "adopted" and r["path"] == b, r
        return "fallback rule holds three ways; stage_finish adopts world_built.npz"
    finally:
        shutil.rmtree(st, ignore_errors=True)


@case
def digest_mismatch_detected():
    st = _tmp()
    try:
        b = os.path.join(st, "world_built.npz")
        f = os.path.join(st, "world_finished.npz")
        offline.save_volume(_tiny(1), b)
        offline.save_volume(_tiny(2), f)
        artifact.adopt(st, f, kind="finished", why="test", candidate="c1")
        assert artifact.resolve(st)["source"] == "adopted"
        # the adopted file changes after adoption
        time.sleep(0.02)
        offline.save_volume(_tiny(1), f)
        r = artifact.resolve(st)
        assert r["mismatch"] and "changed after adoption" in r["mismatch"], r
        assert r["source"] == "fallback" and "ADOPTION MISMATCH" in r["note"], r
        # a finished adoption goes stale when the structural world changes under it
        offline.save_volume(_tiny(2), f)
        artifact.adopt(st, f, kind="finished", why="test again")
        time.sleep(0.02)
        offline.save_volume(_tiny(2), b)
        r = artifact.resolve(st)
        assert r["mismatch"] and "stale" in r["mismatch"], r
        # refusals by name
        for bad in ({"kind": "pretty"}, {"why": ""}):
            try:
                artifact.adopt(st, f, **{"kind": "finished", "why": "x", **bad})
            except artifact.ArtifactError:
                continue
            raise AssertionError(f"adopt accepted {bad}")
        return "a changed adopted file and a stale finished adoption are both reported"
    finally:
        shutil.rmtree(st, ignore_errors=True)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    ok = fail = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__
        if only and only not in name:
            continue
        try:
            said = fn()
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        except Exception as e:                   # noqa: BLE001 -- reported
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} artifact cases pass ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
