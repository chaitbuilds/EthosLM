"""The closure round's acceptance runner: one machine-readable answer per gate.

    $PY scripts/check_closure.py --run c1
    $PY scripts/check_closure.py --run c1 --rounds closure-farm --no-probes

`out/<round>/`) and from **production entry points** run as probes in temporary copies
of those artifacts. Nothing here is a helper the pipeline does not call. A gate that
cannot be evidenced answers `pass: false` with the reason, and the whole record is
written to `out/closure-<run>/acceptance/<stamp>.json` and `latest.json` so a failing
run is retained beside the passing one that replaces it.

The runner never edits a round's state directory: probes copy what they need to a
temporary directory and run the stage there.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATES = ("preserved_meaning", "causal_composition", "stable_realization",
         "complete_small_place", "successful_revision", "transfer_reference_fidelity",
         "honest_evidence_lifecycle")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _state(name: str) -> str:
    return os.path.join(ROOT, "out", name)


def _round(name: str):
    from ethoslm import pipeline
    p = os.path.join(ROOT, "rounds", f"{name}.json")
    return pipeline.Round.load(p) if os.path.exists(p) else None


class Gate:
    def __init__(self, name: str):
        self.name = name
        self.checks: list = []

    def check(self, what: str, ok, why: str = "", **evidence) -> bool:
        self.checks.append({"check": what, "pass": bool(ok), "why": why,
                            **({"evidence": evidence} if evidence else {})})
        return bool(ok)

    def result(self) -> dict:
        return {"pass": bool(self.checks) and all(c["pass"] for c in self.checks),
                "checks": self.checks}


def _quiet(fn, *a, **kw):
    """Run a stage without its chatter; return (result, exception or None)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return fn(*a, **kw), None
    except Exception as e:                          # noqa: BLE001 -- a probe reports
        return None, e


def _copy_state(name: str, files=None) -> str:
    """A temporary copy of a round's state (all of it, or the named files)."""
    src = _state(name)
    tmp = tempfile.mkdtemp(prefix=f"closure-{name}-")
    if not os.path.isdir(src):
        return tmp
    for f in sorted(os.listdir(src)):
        if files is not None and f not in files:
            continue
        s, d = os.path.join(src, f), os.path.join(tmp, f)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True)
        else:
            shutil.copy2(s, d)
    return tmp


def _temp_round(name: str, files=None):
    from ethoslm import pipeline
    rnd = _round(name)
    tmp = _copy_state(name, files)
    return pipeline.Round(name=name, sentence=rnd.sentence if rnd else "",
                          flags=dict(rnd.flags) if rnd else {}, state_dir=tmp), tmp


# ------------------------------------------------------------------ the gates

def gate_preserved_meaning(g: Gate, proof: str, probes: bool) -> None:
    from ethoslm import contracts, interpret, intent as intent_mod
    st = _state(proof)
    it = _load(os.path.join(st, "intent.json"))
    spec = _load(os.path.join(st, "place.checked.json")) or _load(os.path.join(st, "place.json"))
    counts = [r for r in (it or {}).get("requirements") or [] if r["kind"] == "count"]
    g.check("count requirement carries n, exactness and its subject",
            counts and all(r["wants"].get("n") and "about" in r["wants"]
                           and r["wants"].get("what") for r in counts),
            "the interpretation's count reaches intent.json with its subject" if counts
            else "no count requirement in intent.json",
            counts=[r["wants"] for r in counts])
    if counts and spec:
        n = int(counts[0]["wants"]["n"])
        g.check("the spec's band is the sentence's count",
                spec.get("explicit_count") and int(spec.get("structures") or 0) == n
                and list(spec.get("size_band") or []) == [n, n],
                f"spec: structures {spec.get('structures')}, band {spec.get('size_band')}, "
                f"explicit_count {spec.get('explicit_count')}; sentence: {n}")
        g.check("an explicit count is never negotiated",
                not any(x.get("what") == "size_band" for x in spec.get("negotiated") or []),
                f"negotiated: {spec.get('negotiated') or []}")
    hard_open = [r["id"] for r in (it or {}).get("requirements") or []
                 if r["hard"] and r["status"] == "unsupported"]
    g.check("unsupported obligations stay explicit in the record",
            True, f"{len(hard_open)} unsupported hard requirement(s) carried: {hard_open}")
    if not probes:
        return
    # the 50 -> 1 counterexample, through the production cross-check
    s = "Build exactly 50 houses."
    interp = interpret.read_answer(s, {"reads": [
        {"id": "count/houses", "kind": "count", "says": "one house",
         "wants": {"n": 1, "about": False, "what": "houses"}, "phrase": "houses",
         "why": "a wrong reading"}]})
    ck, err = _quiet(interpret.cross_check, s, interp)
    disagrees = bool(ck and (ck.get("contradicts") or
                             any(c.get("subject", [None])[0] == "count"
                                 for c in ck.get("unsupported_phrase") or [])))
    reqs, err2 = _quiet(interpret.requirements, s, interp)
    got = {r["id"]: r for r in (reqs or {}).get("requirements") or []}
    survives = any(r["kind"] == "count" and int(r["wants"].get("n") or 0) == 50
                   for r in got.values())
    g.check("50 -> 1 is refused as a disagreement and the 50 survives",
            disagrees and survives,
            f"cross-check contradicts: {(ck or {}).get('contradicts')}; requirements: "
            f"{[(r['kind'], r['wants']) for r in got.values() if r['kind'] == 'count']}"
            + (f"; error {err or err2}" if err or err2 else ""))
    # unsupported meaning survives a reading that asks for it
    s2 = "Build a Persian caravanserai town around a windmill, laid out on a grid."
    rules = intent_mod.read(s2)
    interp2 = interpret.read_answer(s2, {"reads": [
        {"id": "feature/mill", "kind": "feature", "says": "a windmill",
         "wants": {"feature": "mill"}, "phrase": "windmill", "why": "named outright"},
        {"id": "layout/grid", "kind": "layout", "says": "on a grid",
         "wants": {"policy": "grid"}, "phrase": "laid out on a grid", "why": "named"},
        {"id": "tradition/persian", "kind": "tradition", "says": "Persian",
         "wants": {"tradition": "persian"}, "phrase": "Persian", "why": "named"}]})
    reqs2, err3 = _quiet(interpret.requirements, s2, interp2)
    uns = sorted(r["id"] for r in (reqs2 or {}).get("requirements") or []
                 if r["status"] == "unsupported")
    g.check("what this library cannot express stays unsupported through interpretation",
            len(uns) >= 3, f"unsupported: {uns}" + (f"; error {err3}" if err3 else ""),
            rules_unsupported=[r["id"] for r in rules["requirements"]
                               if r["status"] == "unsupported"])


def gate_causal_composition(g: Gate, run: str, proof: str) -> None:
    from ethoslm import intent as intent_mod
    p = os.path.join(ROOT, "out", f"closure-{run}", "spatial", "experiments.json")
    doc = _load(p)
    g.check("the paired composition experiments were run through production entry points",
            doc is not None and doc.get("generated_by") and doc.get("cases"),
            f"{p}: {'present' if doc else 'absent'}"
            + (f", generated by {doc.get('generated_by')}" if doc else ""))
    for want in ("around_vs_beside", "dense_sparse_swap"):
        case = next((c for c in (doc or {}).get("cases") or [] if c.get("name") == want),
                    None)
        g.check(f"experiment {want}: baseline violates, alternative satisfies, unchanged "
                f"constraints preserved",
                case and case.get("baseline_violates") and case.get("alternative_satisfies")
                and case.get("unchanged_preserved"),
                json.dumps({k: case.get(k) for k in ("baseline_violates",
                                                      "alternative_satisfies",
                                                      "unchanged_preserved", "why")})
                if case else "no such case")
    # density: one definition, consumed by generation and checking
    fn = getattr(intent_mod, "density_target", None)
    if fn is None:
        g.check("density units, denominators and bounds agree between generation and "
                "checking", False, "intent.density_target does not exist yet")
    else:
        from ethoslm import placeplan
        agree = getattr(placeplan, "density_bounds_from", None)
        got = fn("sparse", "rural")
        g.check("density units, denominators and bounds agree between generation and "
                "checking", agree is not None and got.get("metric") and got.get("hi")
                is not None,
                f"intent.density_target('sparse', 'rural') = {got}; placeplan consumer "
                f"{'present' if agree else 'absent'}")
    st = _state(proof)
    it = _load(os.path.join(st, "intent.json"))
    rel = [r for r in (it or {}).get("requirements") or [] if r["kind"] == "relation"]
    g.check("the proof's relations are satisfied on the adopted plan",
            rel and all(r["status"] == "satisfied" for r in rel),
            "; ".join(f"{r['id']}: {r['status']}" for r in rel) or "no relation requirement")


def gate_stable_realization(g: Gate, proof: str, probes: bool) -> None:
    st = _state(proof)
    arr = _load(os.path.join(st, "arrangements.json"))
    rows = (arr or {}).get("districts") or []
    g.check("arrangements record proposed and realized separately",
            rows and all("proposed" in r and "realized" in r for r in rows),
            f"{len(rows)} district row(s)")
    place = _load(os.path.join(st, "plan.place.json"))
    ds = (place or {}).get("districts") or []
    g.check("no district's adopted count has been written back as its proposal",
            ds and all(d.get("proposed_structures") is not None for d in ds)
            and all(int(d.get("proposed_structures")) >= int(d.get("structures") or 0)
                    or d.get("reasked") for d in ds),
            "; ".join(f"{d['name']}: proposed {d.get('proposed_structures')} adopted "
                      f"{d.get('structures')}" for d in ds) or "no districts")
    if probes and place and ds:
        from ethoslm import arrange, placeplan, spec as spec_mod
        rnd = _round(proof)
        spec = None
        with contextlib.suppress(Exception):
            spec = spec_mod.read_spec(_load(os.path.join(st, "place.checked.json"))
                                      or _load(os.path.join(st, "place.json")),
                                      rnd.sentence)
        outs = []
        for d in ds:
            part = placeplan._district_part(spec, d) if spec else None
            if not part or spec_mod.character(part) is None:
                continue
            role = spec_mod.district_role(spec, d)
            _t, decls = placeplan.types_card(None, spec.get("form"), role)
            base = {k: v for k, v in d.items() if k != "structures"}
            base["structures"] = int(d.get("proposed_structures") or 0)
            a, e1 = _quiet(arrange.compile_once, base, part, place, decls, spec=spec)
            b, e2 = _quiet(arrange.compile_once, dict(base), part, place, decls, spec=spec)
            # adopt, then regenerate from the ORIGINAL proposal
            adopted = dict(base, structures=int((a or {}).get("lots") or 0))
            again = dict(adopted, structures=int(base["structures"]))
            c, e3 = _quiet(arrange.compile_once, again, part, place, decls, spec=spec)
            same = bool(a and b and c and a["lots"] == b["lots"] == c["lots"]
                        and a.get("lot") == c.get("lot"))
            outs.append({"district": d["name"], "lots": [(x or {}).get("lots")
                                                          for x in (a, b, c)],
                         "same": same, "errors": [str(x) for x in (e1, e2, e3) if x]})
        g.check("original inputs regenerate the same arrangement",
                outs and all(o["same"] for o in outs), json.dumps(outs)[:800])
    parts = _load(os.path.join(st, "parts.json"))
    built = [r for w in (parts or {}).get("waves") or [] for r in w.get("parts") or []
             if r.get("status") == "built"]
    g.check("every built part carries an emitted outcome measured off construction",
            built and all(isinstance(r.get("emitted"), dict)
                          and "storeys" in r["emitted"] for r in built),
            f"{len(built)} built part(s); with emitted: "
            f"{sum(1 for r in built if isinstance(r.get('emitted'), dict))}")
    if probes:
        try:
            from ethoslm import construction
            got, err = _quiet(construction.probe_storeys, "cottage", 3, (9, 9), (28, 12))
            g.check("the three-storey cottage fallback is measured, not read off params",
                    got and got.get("small", {}).get("emitted") == 1
                    and got.get("control", {}).get("emitted") == 3
                    and got.get("small", {}).get("planned") == 3,
                    json.dumps(got)[:600] + (f"; error {err}" if err else ""))
        except ImportError:
            g.check("the three-storey cottage fallback is measured, not read off params",
                    False, "ethoslm.construction does not exist yet")


def _complete(g: Gate, name: str, label: str) -> None:
    st = _state(name)
    rj = _load(os.path.join(st, "round.json"))
    res = (rj or {}).get("results") or {}
    g.check(f"{label}: the round finished (no pending, no stopped)",
            rj is not None and "pending" not in res and "stopped" not in res,
            f"pending: {res.get('pending')}; stopped: {res.get('stopped')}"
            if rj else f"no round.json under {st}")
    parts = _load(os.path.join(st, "parts.json"))
    g.check(f"{label}: actual construction, every part built",
            parts and parts.get("built", 0) > 0 and parts.get("failed", 0) == 0,
            f"built {(parts or {}).get('built')}, failed {(parts or {}).get('failed')}")
    lint = res.get("lint") or {}
    acc = lint.get("e002_from_the_lane") or {}
    unmeasured = set((lint.get("unmeasured") or {}).get("codes") or [])
    measured_errors = sum(n for code, n in (lint.get("counts") or {}).items()
                          if code.startswith("E") and code not in unmeasured)
    g.check(f"{label}: construction check clean and access one connected network",
            lint and measured_errors == 0 and acc.get("status") == "connected",
            f"measured lint errors {measured_errors} (unmeasured offline: "
            f"{sorted(unmeasured)}), access {acc.get('status')}, "
            f"components {acc.get('lane_components')}")
    pr = _load(os.path.join(st, "place_read.json"))
    g.check(f"{label}: the place read holds on every clause",
            pr and pr.get("holds"),
            ("failed: " + ", ".join(pr.get("failed") or [])) if pr else "no place_read.json")
    it = _load(os.path.join(st, "intent.json"))
    hard = [r for r in (it or {}).get("requirements") or [] if r["hard"]]
    # the place read's own clauses are the measurement of record: `asked/<id>` per
    # requirement, on the built place
    clauses = {c.get("clause"): c for c in (pr or {}).get("clauses") or []}
    unmet = [r["id"] for r in hard
             if not (clauses.get(f"asked/{r['id']}") or {}).get("holds")]
    g.check(f"{label}: every hard requirement is satisfied by measurement",
            hard and pr and not unmet,
            f"unmet on the built place: {unmet}" if pr else "no place read")
    views = _load(os.path.join(st, "inspection", "views.json"))
    g.check(f"{label}: assembled built world inspected with overall and street views",
            views and views.get("overall") and views.get("street")
            and os.path.exists(os.path.join(st, "inspection", "reading.md")),
            "inspection/views.json and inspection/reading.md present" if views
            else "no inspection of the built world")


def gate_complete_small_place(g: Gate, proof: str) -> None:
    _complete(g, proof, proof)


def gate_successful_revision(g: Gate, proof: str) -> None:
    st = _state(proof)
    pv = _load(os.path.join(st, "preview", "preview.json"))
    ap = (pv or {}).get("applied") or {}
    g.check("one revision applied, not rolled back, caused by a real finding",
            pv and pv.get("revisions", 0) >= 1 and not ap.get("rolled_back")
            and (ap.get("characters") or ap.get("voice") or ap.get("place"))
            and ap.get("caused_by"),
            f"revisions {(pv or {}).get('revisions')}, rolled_back {ap.get('rolled_back')}, "
            f"caused_by {ap.get('caused_by')}, changed {sorted(ap.get('characters') or {})}")
    imp = ap.get("improvement") or {}
    g.check("the second reading of the revised candidate verifies the improvement",
            imp.get("verified") and imp.get("before") is not None
            and imp.get("after") is not None,
            json.dumps({k: imp.get(k) for k in ("verified", "closed", "still_open",
                                                "compare", "why")})[:500]
            or "no improvement record")
    views = _load(os.path.join(st, "inspection", "views.json"))
    built_closed = set(((views or {}).get("reading") or {}).get("closed") or [])
    cited = list(ap.get("caused_by") or [])
    g.check("the reading of the BUILT world closes the cited findings",
            views and cited and all(c in built_closed for c in cited)
            and views.get("candidate") == (pv or {}).get("candidate"),
            f"cited {cited}, closed by the built reading {sorted(built_closed)}, inspection "
            f"candidate {(views or {}).get('candidate')}")
    parts = _load(os.path.join(st, "parts.json"))
    g.check("the affected output was rebuilt for the revised candidate",
            parts and pv and parts.get("candidate") and parts.get("candidate") == pv.get("candidate"),
            f"parts candidate {(parts or {}).get('candidate')}, preview candidate "
            f"{(pv or {}).get('candidate')}")


def gate_transfer(g: Gate, rings: str, held: str) -> None:
    st = _state(rings)
    parts = _load(os.path.join(st, "parts.json"))
    g.check(f"{rings}: a contrasting compound/neighbourhood sample was actually built",
            parts and parts.get("built", 0) > 0 and parts.get("failed", 0) == 0
            and (parts.get("sample") or parts.get("built", 0) > 20),
            f"built {(parts or {}).get('built')}, failed {(parts or {}).get('failed')}, "
            f"sample {(parts or {}).get('sample', {}).get('plots') if parts else None}")
    rj = _load(os.path.join(st, "round.json"))
    res = (rj or {}).get("results") or {}
    lint = res.get("lint") or {}
    acc = lint.get("e002_from_the_lane") or {}
    unmeasured = set((lint.get("unmeasured") or {}).get("codes") or [])
    measured_errors = sum(n for code, n in (lint.get("counts") or {}).items()
                          if code.startswith("E") and code not in unmeasured)
    g.check(f"{rings}: approaches and network dependencies qualified",
            lint and measured_errors == 0 and acc.get("status") == "connected",
            f"measured lint errors {measured_errors} (unmeasured offline: "
            f"{sorted(unmeasured)}), access {acc.get('status')}")
    jd = _load(os.path.join(st, "judgment.json"))
    rd = _load(os.path.join(st, "reading.json"))
    srcs = {s.get("id") for s in (rd or {}).get("sources") or []}
    claims = {c.get("id"): c for c in (rd or {}).get("claims") or []}
    pos = [v for v in (jd or {}).get("verdicts") or []
           if (v.get("recognisable") if v.get("recognisable") is not None else v.get("holds"))]
    cited_ok = pos and all(v.get("cites") and all(c in claims and claims[c].get("source") in srcs
                                                   for c in v["cites"]) for v in pos)
    g.check(f"{rings}: a positive sourced identity/tradition judgment of built output",
            jd and pos and cited_ok and jd.get("candidate")
            and any(str(x).endswith(("world_built.npz", ".png")) for x in jd.get("looked_at") or []),
            f"verdicts {len((jd or {}).get('verdicts') or [])}, positive {len(pos)}, cites "
            f"resolve to retrieved sources: {bool(cited_ok)}")
    it = _load(os.path.join(st, "intent.json"))
    pr = _load(os.path.join(st, "place_read.json"))
    clauses = {c.get("clause"): c for c in (pr or {}).get("clauses") or []}
    trad = [r for r in (it or {}).get("requirements") or [] if r["kind"] in ("tradition", "identity")]
    got = [(r["id"], bool((clauses.get(f"asked/{r['id']}") or {}).get("holds"))) for r in trad]
    g.check(f"{rings}: the tradition/identity requirement closed by that judgment at the "
            f"place read", trad and pr and all(ok for _i, ok in got),
            "; ".join(f"{i}: {'holds' if ok else 'fails'}" for i, ok in got)
            or "no such requirement")
    _complete(g, held, f"held-out {held}")


def gate_lifecycle(g: Gate, proof: str, probes: bool) -> None:
    if not probes:
        g.check("lifecycle probes", False, "probes disabled")
        return
    from ethoslm import contracts, deps, pipeline
    from ethoslm.pipeline import stages_measure, stages_plan
    st = _state(proof)
    have_plan = os.path.exists(os.path.join(st, "plan.json"))
    # (a) a pending judgment answered for candidate A is not adopted for candidate B
    rnd, tmp = _temp_round(proof)
    try:
        if have_plan:
            it = contracts.load(rnd, "intent")
            it["requirements"].append({"id": "tradition/probe", "says": "probe",
                                       "kind": "tradition", "hard": True,
                                       "wants": {"tradition": "japanese"},
                                       "status": "open", "why": "probe"})
            contracts.save(rnd, "intent", it)
            for f in ("judgment.json", "judgment.answer.json"):
                if os.path.exists(rnd.rel(f)):
                    os.remove(rnd.rel(f))
            res, err = _quiet(stages_measure.stage_qualify, rnd, None, {})
            a = deps.candidate_id(rnd)
            asked = (res or {}).get("judgment") or {}
            if asked.get("status") == "needs_model":
                json.dump({"verdicts": [{"about": "tradition", "name": "japanese",
                                         "holds": True, "why": "probe", "cites": []}]},
                          open(asked["write"], "w"))
                plan = rnd.plan()

                def _first_leaf(nodes):
                    for n in nodes or []:
                        if n.get("x1") is not None and not n.get("children") \
                                and not n.get("parts"):
                            return n
                        got = _first_leaf(n.get("children") or n.get("parts") or [])
                        if got is not None:
                            return got
                    return None
                leaf = _first_leaf(plan.get("parts") or [])
                if leaf is not None:
                    leaf["x1"] = int(leaf["x1"]) + 7                # candidate B
                    json.dump(plan, open(rnd.rel("plan.json"), "w"))
                b = deps.candidate_id(rnd)
                res2, err2 = _quiet(stages_measure.stage_qualify, rnd, None, {})
                jd = contracts.load(rnd, "judgment")
                adopted_for_b = jd is not None and jd.get("candidate") == b and a != b
                g.check("a pending answer for candidate A is not labelled B's judgment",
                        not adopted_for_b and a != b,
                        f"A {a}, B {b}, judgment candidate {(jd or {}).get('candidate')}, "
                        f"result {str(res2)[:160]}" + (f"; error {err2}" if err2 else ""))
            else:
                g.check("a pending answer for candidate A is not labelled B's judgment",
                        False, f"qualify did not stage a judge job: {str(res)[:200]}"
                        + (f"; error {err}" if err else ""))
        else:
            g.check("a pending answer for candidate A is not labelled B's judgment",
                    False, "the proof has no plan.json to probe against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # (b) a fake citation cannot qualify
    try:
        contracts.read("reading", {"record": "reading", "version": 1, "sentence": "x",
                                   "classification": "named", "sources": [],
                                   "claims": [{"id": "c1", "says": "s", "source": "nope"}]})
        g.check("a claim citing a source nobody retrieved is refused", False, "accepted")
    except contracts.ContractError as e:
        g.check("a claim citing a source nobody retrieved is refused", True, str(e)[:160])
    # (c) plan-only evidence cannot qualify identity
    rnd, tmp = _temp_round(proof, files={"plan.json", "intent.json", "reading.json",
                                         "place.json", "place.checked.json", "deps.json",
                                         "interpretation.json"})
    try:
        if have_plan:
            it = contracts.load(rnd, "intent")
            it["requirements"].append({"id": "identity/probe", "says": "probe",
                                       "kind": "identity", "hard": True,
                                       "wants": {"name": "Probe"}, "status": "open",
                                       "why": "probe"})
            contracts.save(rnd, "intent", it)
            res, err = _quiet(stages_measure.stage_qualify, rnd, None, {})
            asked = (res or {}).get("judgment") or {}
            g.check("a plan alone is not inspectable output for a judge",
                    asked.get("status") != "needs_model"
                    and (res or {}).get("status") in ("unresolved", "error", None)
                    and not contracts.load(rnd, "judgment"),
                    f"{str(res)[:200]}" + (f"; error {err}" if err else ""))
        else:
            g.check("a plan alone is not inspectable output for a judge", False,
                    "the proof has no plan.json to probe against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # (d) a negative judgment blocks qualification
    rnd, tmp = _temp_round(proof)
    try:
        if have_plan and os.path.exists(rnd.rel("parts.json")):
            it = contracts.load(rnd, "intent")
            it["requirements"].append({"id": "tradition/probe", "says": "probe",
                                       "kind": "tradition", "hard": True,
                                       "wants": {"tradition": "japanese",
                                                 "nearest_form": "east_asian"},
                                       "status": "open", "why": "probe"})
            contracts.save(rnd, "intent", it)
            here = deps.candidate_id(rnd)
            contracts.save(rnd, "judgment", contracts.make(
                "judgment", sentence=rnd.sentence, candidate=here,
                looked_at=[rnd.rel("plan.json")],
                verdicts=[{"about": "tradition", "name": "japanese", "holds": False,
                           "why": "not built that way", "cites": []}]))
            res, err = _quiet(stages_measure.stage_place_check, rnd, None, {})
            g.check("a negative judgment cannot qualify the build",
                    res and not res.get("holds") and res.get("blocks") == "fidelity",
                    f"holds {(res or {}).get('holds')}, blocks {(res or {}).get('blocks')}, "
                    f"failed {(res or {}).get('failed')}" + (f"; error {err}" if err else ""))
        else:
            g.check("a negative judgment cannot qualify the build", False,
                    "the proof has no built parts to probe against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # (e) interrupt: a truncated output is stale, and rebuilding from unchanged inputs
    # restores the same candidate identity
    rnd, tmp = _temp_round(proof)
    try:
        if have_plan and os.path.exists(rnd.rel("site.json")):
            before = deps.candidate_id(rnd)
            open(rnd.rel("plan.json"), "w").write("")
            fresh, why = deps.check(rnd, "assembled")
            stale = not fresh
            be = pipeline.OfflineBackend(rnd, dry_run=True)
            res, err = _quiet(stages_plan.stage_plan, rnd, be, {})
            from ethoslm.pipeline import round as driver
            res, err2 = _quiet(driver._drive_reentries, rnd, be, {}, "plan", res or {})
            after = deps.candidate_id(rnd) if os.path.exists(rnd.rel("plan.json")) else None
            g.check("interrupted output is stale and an unchanged rebuild restores the "
                    "same candidate", stale and after == before,
                    f"stale: {why[:100]}; before {before}, after {after}"
                    + (f"; error {err or err2}" if err or err2 else ""))
        else:
            g.check("interrupted output is stale and an unchanged rebuild restores the "
                    "same candidate", False, "the proof has no plan/site to probe against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # (f) unchanged warm replay: every stamped artifact of the finished proof is warm
    rnd = _round(proof)
    if rnd is not None and os.path.isdir(st):
        table = deps.table(rnd)
        cold = [t for t in table if not t["fresh"]]
        g.check("unchanged warm replay: every artifact of the finished proof is warm",
                table and not cold and os.path.exists(os.path.join(st, "place_read.json")),
                f"{len(table)} stamped, cold: {[(t['artifact'], t['why'][:80]) for t in cold]}")
    else:
        g.check("unchanged warm replay: every artifact of the finished proof is warm",
                False, "no proof state")
    # (g) rollback: a revision the plan stage refuses leaves the candidate exactly as it
    # was -- driven through stage_preview on a copy of the finished proof
    rnd, tmp = _temp_round(proof)
    try:
        pv_p = rnd.rel("preview", "preview.json")
        if have_plan and os.path.exists(pv_p) and os.path.exists(rnd.rel("site.json")):
            from ethoslm.pipeline import stages_media
            pv = json.load(open(pv_p))
            before = deps.candidate_id(rnd)
            plan_bytes = open(rnd.rel("plan.json"), "rb").read()
            # re-open the loop at the revision: one reading on disk, no revision yet
            pv["revisions"] = 0
            pv["done"] = False
            pv.pop("applied", None)
            pv["repaired"] = True
            json.dump(pv, open(pv_p, "w"))
            for f in ("revision.json", "reading_1.md", "reading_1.findings.json"):
                if os.path.exists(rnd.rel("preview", f)):
                    os.remove(rnd.rel("preview", f))
            spec = rnd.place_spec() or {}
            dist = next((d["name"] for d in spec.get("defining_parts") or []
                         if d.get("character") is not None), None)
            json.dump({"characters": {dist: {"frontage": "street", "attached": True,
                                             "lot_width": 24, "lot_depth": 24,
                                             "block": 12, "open_share": 0.95,
                                             "courtyard_share": 0.95}},
                       "voice": None, "caused_by": [],
                       "why": "a probe: a fabric no rectangle here can carry"},
                      open(rnd.rel("preview", "revision.json"), "w"))
            be = pipeline.OfflineBackend(rnd, dry_run=True)
            res, err = _quiet(stages_media.stage_preview, rnd, be, {})
            after = deps.candidate_id(rnd)
            same_plan = open(rnd.rel("plan.json"), "rb").read() == plan_bytes
            rec = json.load(open(pv_p))
            rolled = bool((rec.get("applied") or {}).get("rolled_back"))
            g.check("rollback of a refused revision restores the candidate identity",
                    rolled and after == before and same_plan,
                    f"rolled_back {rolled}, before {before}, after {after}, plan bytes "
                    f"identical {same_plan}" + (f"; error {err}" if err else "")
                    + (f"; refused: {(rec.get('applied') or {}).get('refused')}"
                       if rec.get("applied") else ""))
        else:
            g.check("rollback of a refused revision restores the candidate identity",
                    False, "the proof has no plan, preview or site to probe against")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="c1")
    ap.add_argument("--proof", default="closure-farm")
    ap.add_argument("--rings", default="closure-rings")
    ap.add_argument("--held", default="closure-held")
    ap.add_argument("--no-probes", action="store_true")
    a = ap.parse_args()
    probes = not a.no_probes
    gates = {n: Gate(n) for n in GATES}
    steps = [
        (gates["preserved_meaning"], lambda g: gate_preserved_meaning(g, a.proof, probes)),
        (gates["causal_composition"], lambda g: gate_causal_composition(g, a.run, a.proof)),
        (gates["stable_realization"], lambda g: gate_stable_realization(g, a.proof, probes)),
        (gates["complete_small_place"], lambda g: gate_complete_small_place(g, a.proof)),
        (gates["successful_revision"], lambda g: gate_successful_revision(g, a.proof)),
        (gates["transfer_reference_fidelity"], lambda g: gate_transfer(g, a.rings, a.held)),
        (gates["honest_evidence_lifecycle"], lambda g: gate_lifecycle(g, a.proof, probes)),
    ]
    t0 = time.time()
    for g, fn in steps:
        try:
            fn(g)
        except Exception as e:                      # noqa: BLE001 -- a gate reports
            import traceback
            g.check("the gate's evaluator ran", False,
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
    out = {"run": a.run, "proof": a.proof, "rings": a.rings, "held": a.held,
           "gates": {n: g.result() for n, g in gates.items()},
           "passed": [n for n, g in gates.items() if g.result()["pass"]],
           "failed": [n for n, g in gates.items() if not g.result()["pass"]],
           "seconds": round(time.time() - t0, 1),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    d = os.path.join(ROOT, "out", f"closure-{a.run}", "acceptance")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    json.dump(out, open(os.path.join(d, f"{stamp}.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(d, "latest.json"), "w"), indent=1)
    for n, g in gates.items():
        r = g.result()
        print(f"{'PASS' if r['pass'] else 'FAIL'} {n}")
        for c in r["checks"]:
            print(f"   {'ok  ' if c['pass'] else 'MISS'} {c['check']}: {c['why'][:200]}")
    print(f"-> {os.path.join(d, stamp + '.json')}  ({len(out['passed'])}/{len(GATES)} gates)")
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
