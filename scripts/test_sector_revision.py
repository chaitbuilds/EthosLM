"""The fabric reset round: a strip cut is a decision with an input identity, rolled back with
the plans it describes, and screened on admissibility before preference.

    $PY scripts/test_sector_revision.py
    $PY scripts/test_sector_revision.py --case t2

Offline and fast: the ground measurement (`placeplan.negotiate_strip`) and the district
compile (`stages_plan._screen_piece`) are stubbed, because what is tested is the decision
logic around them -- when `sectors.json` is reapplied, when it is stale and what that
withdraws, what a rollback puts back, and how compiled arrangements are ranked.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import placeplan                                            # noqa: E402
from ethoslm.pipeline import promote, stages_media                      # noqa: E402
from ethoslm.pipeline import stages_plan as sp                          # noqa: E402

CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


class Rnd:
    """What `_negotiate_sectors`, `_screen_by_compile` and `promote` ask of a round."""

    def __init__(self, state: str, spec: dict, flags: dict | None = None):
        self.state, self._spec, self.flags = state, spec, dict(flags or {})
        self.name, self.sentence = "t", "a test"

    def rel(self, *parts) -> str:
        return os.path.join(self.state, *parts)

    def place_spec(self):
        return self._spec


def _spec(character: dict | None = None, density: str = "medium") -> dict:
    return {"form": "east_asian", "defining_parts": [
        {"name": "middle_ring", "kind": "group", "family": "district",
         "density": density, "role": "urban", "notes": "prose",
         "character": dict(character or {})}]}


def _place() -> dict:
    ds = [{"name": "m_a", "defines": "middle_ring", "x0": 0, "z0": 0, "x1": 49, "z1": 39,
           "structures": 6, "proposed_structures": 6, "landmarks": ["market"]},
          {"name": "m_b", "defines": "middle_ring", "x0": 55, "z0": 0, "x1": 104,
           "z1": 39, "structures": 6, "proposed_structures": 6}]
    return {"districts": ds, "parts": [{"name": "ring_gate_middle_ring", "kind": "point",
                                        "at": [52, 0]}],
            "arterials": {"cells": [[52, z] for z in range(40)]},
            "layout": {"rings": [{"name": "middle_ring", "level": 70,
                                  "districts": ["m_a", "m_b"]}]}}


def _alt(name, pieces, cuts, least=0.3, founded=1000):
    return {"arrangement": name, "cuts": cuts, "least_share": least,
            "founded_columns": founded,
            "pieces": [{"from": f, "rect": r, "open": False, "feasible_columns": 500,
                        "columns": 800, "level": 70, "share": 0.8} for f, r in pieces]}


EQUAL = _alt("equal", [("m_a", [0, 0, 49, 39]), ("m_b", [55, 0, 104, 39])], 0)
SPLIT = _alt("ground_2", [("m_a", [0, 0, 23, 39]), ("m_a", [27, 0, 49, 39]),
                          ("m_b", [55, 0, 104, 39])], 1, founded=1400)


class Stubs:
    """`negotiate_strip` and `_screen_piece` replaced for one case, and counted."""

    def __init__(self, least=0.3, piece=None):
        self.least, self.piece, self.calls, self.forms = least, piece, 0, []

    def __enter__(self):
        self._ns, self._sp = placeplan.negotiate_strip, sp._screen_piece

        def ns(vol, ds, **kw):
            self.calls += 1
            eq = dict(EQUAL, least_share=self.least)
            return {"measured": True, "alternatives": [eq, SPLIT], "chosen": "ground_2",
                    "adopted": SPLIT, "gain": 400}

        def piece(rnd, spec, trial_place, d, form, ring_level, vol, routes, types):
            self.forms.append((d["name"], form))
            return (self.piece or _legacy_piece)(d, form)
        placeplan.negotiate_strip, sp._screen_piece = ns, piece
        return self

    def __exit__(self, *a):
        placeplan.negotiate_strip, sp._screen_piece = self._ns, self._sp


def _legacy_piece(d, form):
    """A grid district: no `composition`. The split's m_b takes the perimeter form."""
    n = {"m_a": 6, "m_b": 6, "m_a_1": 4, "m_a_2": 5}[d["name"]]
    arr = None
    if form == "perimeter":
        if d["name"] != "m_b" or d["x0"] != 55:
            return None
        n, arr = 7, {"block": 60}
    return {"district": d["name"], "ask": 6, "realized": n, "landmarks": 0,
            "compositions": 1 if arr else 0, "form": form, "belongs": [],
            "form_why": f"`{form}`" if form else None, "arrangement": arr,
            "score": n + (sp.SCREEN_COURT if arr else 0)}


def _run(rnd, place):
    return sp._negotiate_sectors(rnd, place, {}, vol=object())


def _tmp(tag):
    return tempfile.mkdtemp(prefix=f"sector-rev-{tag}-",
                            dir=os.path.join(ROOT, "out", "fr-work-rev")
                            if os.path.isdir(os.path.join(ROOT, "out", "fr-work-rev"))
                            else None)


@case
def t1_a_matching_identity_is_reapplied_without_negotiation():
    """A decision whose ring part, rectangles, level and inputs are unchanged is applied
    again to a re-solved place and to an applied one, and is not negotiated again; the
    plan laid for its form is kept because its marker says it was laid for it."""
    tmp = _tmp("t1")
    try:
        rnd = Rnd(tmp, _spec())
        with Stubs() as st:
            p1 = _place()
            assert _run(rnd, p1) is True
            rec = json.load(open(rnd.rel("sectors.json")))
            assert [s["arrangement"] for s in rec["adopted"]] == ["ground_2"], rec
            ident = rec["adopted"][0]["identity"]
            assert set(ident) >= {"part", "rects", "level", "inputs"}, ident
            assert set(ident["inputs"]) == {"types", "schema", "compiler"}, ident
            assert os.path.exists(rnd.rel("sectors.m_b.laid"))
            open(rnd.rel("plan.district.m_b.json"), "w").write("{}")
            # the same place, as a re-solve draws it again
            p2 = _place()
            assert _run(rnd, p2) is True
            assert [d["name"] for d in p2["districts"]] == ["m_a_1", "m_a_2", "m_b"]
            # the place already carrying it
            assert _run(rnd, p2) is False
            assert st.calls == 1, st.calls
            assert os.path.exists(rnd.rel("plan.district.m_b.json"))
            assert not json.load(open(rnd.rel("sectors.json")))["revised"]
        return (f"negotiated once ({st.calls} call), identity recorded "
                f"{sorted(ident)} with inputs {sorted(ident['inputs'])}; re-applied to a "
                f"re-solved place and a no-op on the applied one, plan kept")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t2_a_changed_ring_character_makes_it_stale():
    """Changing the ring part's character makes the decision stale: it is negotiated
    again, the markers and plans it caused are withdrawn, and why is recorded -- both for
    a re-solved place and for a place that still carries the old pieces."""
    tmp = _tmp("t2")
    try:
        rnd = Rnd(tmp, _spec())
        with Stubs() as st:
            _run(rnd, _place())
            applied = _place()
            _run(rnd, applied)
            for f in ("plan.district.m_b.json", "district_m_b_compiled.json",
                      "plan.json"):
                open(rnd.rel(f), "w").write("{}")
            rec0 = json.load(open(rnd.rel("sectors.json")))
            # the programme changes: attached, street-composed, now `dense`
            rnd._spec = _spec({"attached": True, "courtyard_share": 0,
                               "layout": "street"}, density="dense")
            st.least = 0.9                      # and the ground no longer asks for a cut
            drawn = _place()
            _run(rnd, drawn)
            rec = json.load(open(rnd.rel("sectors.json")))
            assert st.calls == 2, st.calls
            assert rec["adopted"] == [], rec["adopted"]
            rv = rec["revised"][-1]
            assert "part" in rv["changed"], rv
            assert rv["changed"]["part"]["was"]["density"] == "medium"
            assert rv["changed"]["part"]["now"]["density"] == "dense"
            assert rv["changed"]["part"]["now"]["character"]["layout"] == "street"
            for f in ("sectors.m_b.laid", "plan.district.m_b.json",
                      "district_m_b_compiled.json", "plan.json"):
                assert not os.path.exists(rnd.rel(f)), f
                assert f in rv["removed"], (f, rv["removed"])
            assert [d["name"] for d in drawn["districts"]] == ["m_a", "m_b"]
            # a place still carrying the old pieces is taken back to the strip as drawn
            json.dump(rec0, open(rnd.rel("sectors.json"), "w"))
            open(rnd.rel("sectors.m_b.laid"), "w").write("x")
            assert _run(rnd, applied) is True
            assert st.calls == 3, st.calls
            assert [d["name"] for d in applied["districts"]] == ["m_a", "m_b"]
            assert applied["layout"]["rings"][0]["districts"] == ["m_a", "m_b"]
            assert not any(d.get("sector") for d in applied["districts"])
            rv2 = json.load(open(rnd.rel("sectors.json")))["revised"][-1]
            assert rv2["state"] == "applied" and not os.path.exists(
                rnd.rel("sectors.m_b.laid"))
        return (f"stale on {sorted(rv['changed'])}: density {rv['changed']['part']['was']['density']}"
                f" -> {rv['changed']['part']['now']['density']}, negotiated again, "
                f"removed {rv['removed']}; an applied place was taken back to "
                f"{[d['name'] for d in applied['districts']]} and re-negotiated")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t2b_legacy_record_and_local_scope():
    """A legacy `sectors.json` (no identity) is stale because its ring part cannot be
    verified; under a local scope that does not meet the strip, it is kept applied and
    recorded stale instead of re-cut."""
    tmp = _tmp("t2b")
    try:
        rnd = Rnd(tmp, _spec())
        with Stubs() as st:
            _run(rnd, _place())
            rec = json.load(open(rnd.rel("sectors.json")))
            for s in rec["adopted"] + rec["considered"]:
                s.pop("identity", None)
                s.pop("sources", None)
            rec.pop("revised", None)
            json.dump(rec, open(rnd.rel("sectors.json"), "w"))
            rnd.flags["local"] = {"rect": [500, 500, 600, 600], "margin": 0}
            p = _place()
            _run(rnd, p)                        # drawn, outside the scope: kept
            got = json.load(open(rnd.rel("sectors.json")))
            assert st.calls == 1 and got["adopted"][0]["stale_kept"].startswith("outside")
            assert "identity" in got["revised"][-1]["changed"]
            rnd.flags.pop("local")
            json.dump(rec, open(rnd.rel("sectors.json"), "w"))
            _run(rnd, _place())                 # no scope: negotiated again
            assert st.calls == 2
            assert json.load(open(rnd.rel("sectors.json")))["adopted"][0].get("identity")
        return ("legacy record stale on `identity` (unverifiable); kept applied outside "
                "a local scope, re-negotiated without one and then given an identity")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t3_rollback_restores_the_decision_with_its_plans():
    """`promote.reject` and `stages_media._restore` put back `sectors.json` and every
    marker with the plan files they describe, and remove a trial's own markers; an
    accepted copy kept before the boundary covered them leaves them alone."""
    tmp = _tmp("t3")
    try:
        rnd = Rnd(tmp, _spec())
        with Stubs():
            _run(rnd, _place())
        for f in ("plan.json", "plan.district.m_b.json", "plan.place.json"):
            open(rnd.rel(f), "w").write('{"v": "accepted"}')
        names = ("sectors.json", "sectors.m_b.laid", "plan.district.m_b.json", "plan.json")
        before = {f: open(rnd.rel(f), "rb").read() for f in names}
        assert all(f in promote.boundary(rnd)["files"] for f in names)
        promote.begin(rnd, candidate="aaa", finding="f", action="a")
        # the trial re-cuts the strip: new record, new markers, its own plans
        json.dump({"adopted": [], "considered": []}, open(rnd.rel("sectors.json"), "w"))
        os.remove(rnd.rel("sectors.m_b.laid"))
        open(rnd.rel("sectors.m_a_1.laid"), "w").write("trial")
        open(rnd.rel("plan.district.m_b.json"), "w").write('{"v": "trial"}')
        t = promote.reject(rnd, why="worse")
        after = {f: open(rnd.rel(f), "rb").read() for f in names}
        assert after == before
        assert not os.path.exists(rnd.rel("sectors.m_a_1.laid"))
        assert "sectors.m_a_1.laid" in t["restored"]["removed"]
        # the in-process snapshot the improve stage takes round a re-solve
        snap = stages_media._snapshot(rnd)
        assert "sectors.json" in snap and "sectors.m_b.laid" in snap
        os.remove(rnd.rel("sectors.m_b.laid"))
        open(rnd.rel("sectors.m_a_2.laid"), "w").write("x")
        stages_media._restore(rnd, snap)
        assert os.path.exists(rnd.rel("sectors.m_b.laid"))
        assert not os.path.exists(rnd.rel("sectors.m_a_2.laid"))
        # an accepted copy from before: no `covers`, no sectors files in it
        acc = rnd.rel(promote.ACCEPTED)
        for f in os.listdir(acc):
            if f.startswith(promote.LATE_FILES):
                os.remove(os.path.join(acc, f))
        rec = promote.load(rnd)
        rec["accepted"]["boundary"].pop("covers")
        promote.save(rnd, rec)
        promote.begin(rnd, candidate="aaa", finding="f", action="b")
        t2 = promote.reject(rnd, why="worse")
        assert os.path.exists(rnd.rel("sectors.json"))
        assert "sectors.json" in t2["restored"]["left"]["files"]
        return (f"rejected trial: {len(names)} decision/plan files byte-identical, trial "
                f"marker removed; `_snapshot`/`_restore` likewise; a pre-boundary "
                f"accepted copy left {t2['restored']['left']['files']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _comp(ok, pref):
    return {"family": "street", "admissible": ok, "preference": pref, "tried": [1, 2],
            "relationships": [{"name": "market_edges", "required": True, "held": ok,
                               "measure": 2 if ok else 0,
                               "why": "fronts face the market" if ok else "no fronts"}]}


def _screen(rnd, st, spec):
    ds = copy.deepcopy(_place()["districts"])
    got = {"alternatives": [EQUAL, SPLIT]}
    return sp._screen_by_compile(rnd, spec, _place(), ds, got, [52, 0], 70, object())


@case
def t4_admissible_before_preference():
    """An admissible arrangement outranks a higher-scoring inadmissible one; with none
    admissible no re-cut is adopted and the strip stays as drawn with `no_admissible`.
    A street-composed piece is never offered `perimeter`."""
    tmp = _tmp("t4")
    try:
        spec = _spec({"layout": "street", "attached": True}, density="dense")
        rnd = Rnd(tmp, spec)

        def piece(d, form):                     # the uncut m_a: 20 houses, no market
            ok = d["name"] != "m_a"
            n = 5 if ok else 20
            return {"district": d["name"], "ask": 6, "realized": n, "landmarks": 1,
                    "compositions": 0, "form": form, "belongs": [], "form_why": None,
                    "arrangement": None, "score": n + 6,
                    "composition": _comp(ok, 3.0)}
        with Stubs(piece=piece) as st:
            got = _screen(rnd, st, spec)
            assert got["best"] == "ground_2", got["rows"]
            rows = {r["arrangement"]: r for r in got["rows"]}
            assert rows["equal"]["score"] > rows["ground_2"]["score"]
            assert rows["equal"]["admissible"] is False and rows["ground_2"]["admissible"]
            assert not any(f == "perimeter" for _n, f in st.forms), st.forms
        with Stubs(piece=lambda d, f: dict(piece(d, f), composition=_comp(False, 9.0))):
            got2 = _screen(rnd, None, spec)
            assert got2["best"] is None and got2["no_admissible"] is True
            assert got2["failed"]["equal"][0]["relationships"][0]["name"] == "market_edges"
        # ...and through the negotiation: the strip is kept as drawn, with the finding
        with Stubs(piece=lambda d, f: dict(piece(d, f), composition=_comp(False, 9.0))):
            p = _place()
            changed = _run(rnd, p)
            rec = json.load(open(rnd.rel("sectors.json")))
            assert changed is False and rec["adopted"] == []
            row = rec["considered"][0]
            assert row["no_admissible"] and row["failed"]["ground_2"]
            assert [d["name"] for d in p["districts"]] == ["m_a", "m_b"]
        return (f"`ground_2` (admissible, score {rows['ground_2']['score']}) over `equal` "
                f"(inadmissible, score {rows['equal']['score']}); none admissible -> "
                f"best None, no_admissible, kept as drawn; forms offered "
                f"{sorted({f for _n, f in st.forms}, key=str)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@case
def t5_legacy_districts_score_exactly_as_before():
    """Without `composition` the screen is the old one: per piece the higher score of
    the declared fabric and `perimeter` (first on a tie), per arrangement houses + 6 x
    landmarks + 4 x courts + 6 x markets among houses, the maximum adopted, ties to
    fewer cuts; no admissibility fields appear."""
    tmp = _tmp("t5")
    try:
        rnd = Rnd(tmp, _spec())
        with Stubs() as st:
            got = _screen(rnd, st, rnd._spec)
        rows = {r["arrangement"]: r for r in got["rows"]}
        # reference: the old formula over the stub's per-piece best
        assert rows["equal"]["houses"] == 6 + 7 and rows["equal"]["compositions"] == 1
        for r in rows.values():
            assert "composed" not in r and "admissible" not in r
            assert r["score"] == (r["houses"] + 6 * r["landmarks"] + 4 * r["compositions"]
                                  + 6 * r["markets_among_houses"])
        want = max(got["rows"], key=lambda r: (r["score"], -r["cuts"]))["arrangement"]
        assert got["best"] == want, (got["best"], want)
        assert ("m_b", "perimeter") in st.forms
        assert got["score"].startswith("houses + 6 x landmarks")
        return (f"scores {{{', '.join(f'{k}: {v['score']}' for k, v in rows.items())}}}, "
                f"best `{got['best']}` as the old max; perimeter still offered")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    a = ap.parse_args()
    fails = 0
    for fn in CASES:
        if a.case and not fn.__name__.startswith(a.case):
            continue
        try:
            print(f"PASS {fn.__name__}: {fn()}")
        except Exception:                        # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}")
    print(f"{'ok' if not fails else f'{fails} failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
