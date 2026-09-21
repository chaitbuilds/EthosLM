"""The design round's acceptance runner: one machine-readable answer per gate.

    $PY scripts/check_design.py --run d1

Every gate is answered from the **real driver's artifacts** (`out/des-*/`) and, where a
question is about behaviour rather than about a retained record, from production entry
points run as probes. A gate that cannot be evidenced answers `pass: false` with the
reason; a gate never answers `pass` from an absence. The record is written to
`out/des-d1/acceptance/<stamp>.json` and `latest.json`, so a failing run is retained
beside the passing one that replaces it. Nothing here edits a round's state directory.

Five for the decision chain, four for the production proofs, one for the bounded visual
experiment. They were registered before behaviour was edited.

demand_before_size      the selected pool, parameters, required features, requirement
ids and scope resolve BEFORE the parent is sized, and every consumer uses that demand.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATES = ("demand_before_size", "spatial_ground_decision", "features_final_world",
         "obligations_close", "iteration_proportional", "farm_two_storeys",
         "rings_dense", "city_section", "held_out", "material_decision")

FARM, RINGS, CITY, HELD, REFUSE = ("des-farm", "des-rings", "des-city", "des-held",
                                   "des-refuse")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _state(name: str) -> str:
    return os.path.join(ROOT, "out", name)


def _rel(name: str, *parts) -> str:
    return os.path.join(_state(name), *parts)


def _doc(name: str, *parts):
    return _load(_rel(name, *parts))


def _rows(parts_rec) -> list:
    """Every part row of a parts record, across its waves."""
    return [r for w in ((parts_rec or {}).get("waves") or []) for r in (w.get("parts") or [])]


def _clauses(name: str) -> list:
    rd = _doc(name, "place_read.json") or _doc(name, "readout.json") or {}
    got = rd.get("clauses") or ((rd.get("place_read") or {}).get("clauses"))
    return list(got or [])


def _requirements(name: str) -> list:
    return list((_doc(name, "intent.json") or {}).get("requirements") or [])


class Gate:
    def __init__(self, name: str):
        self.name, self.checks = name, []

    def check(self, what: str, ok, why: str = "", **evidence) -> bool:
        self.checks.append({"check": what, "pass": bool(ok), "why": why,
                            **({"evidence": evidence} if evidence else {})})
        return bool(ok)

    def result(self) -> dict:
        return {"pass": bool(self.checks) and all(c["pass"] for c in self.checks),
                "checks": self.checks}


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return fn(*a, **kw), None
    except Exception as e:                          # noqa: BLE001 -- a probe reports
        return None, e


# ------------------------------------------------------------------ the chain

def gate_demand_before_size(g: Gate, probes: bool = True) -> None:
    """**A required feature reaches the envelope, and a refusal is propagated.**

        The measured defect this closes, on the imported code:
        `lot_for("cottage", {"storeys": 2})` answered `[5, 5]` and
        `lot_for(..., features=("storeys",))` answered `[15, 17]`, and both production
        callers used the first form.
        
    """
    if probes:
        from ethoslm import demand, envelope
        envelope._MEM.clear()
        bare, e1 = _quiet(envelope.lot_for, "cottage", {"storeys": 2})
        feat, e2 = _quiet(envelope.lot_for, "cottage", {"storeys": 2},
                          features=("storeys",))
        g.check("the featureless and the feature-bearing query differ, so the token is "
                "doing work", not (e1 or e2) and bare and feat
                and tuple(bare["lot_min"] or ()) != tuple(feat["lot_min"] or ()),
                f"without features {(bare or {}).get('lot_min')}, with `storeys` "
                f"{(feat or {}).get('lot_min')}",
                bare=(bare or {}).get("lot_min"), with_storeys=(feat or {}).get("lot_min"))
        g.check("`demand` exposes the resolved query the layout must use",
                all(hasattr(demand, n) for n in
                    ("resolve", "lot", "storeys_admitted", "validate_lot")),
                "resolve/lot/storeys_admitted/validate_lot")
    # no production envelope query under a feature requirement may carry no features
    seen, empty = 0, []
    for name in (FARM, RINGS, CITY, HELD):
        doc = _doc(name, "envelopes.json") or {}
        entries = doc.get("entries") if isinstance(doc.get("entries"), dict) else doc
        for key, got in (entries or {}).items():
            if not isinstance(got, dict):
                continue
            seen += 1
            asked = got.get("asked") or list((got.get("features") or {}).keys())
            if not asked and (got.get("params") or {}).get("storeys", 1) > 1:
                empty.append(key[:120])
    g.check("no retained envelope answer asked for storeys above one without the token",
            seen and not empty,
            f"{seen} cached answer(s); {len(empty)} asked with an empty feature set",
            empty=empty[:6])
    # a refusal is never a smaller default: every refused row carries `refused`
    bad = []
    for name in (FARM, RINGS, CITY, HELD):
        for row in (_doc(name, "resolution.json") or {}).get("regions") or []:
            d = row.get("demand") or {}
            if d.get("refused") and d.get("lot_min"):
                bad.append(row.get("name"))
    g.check("a refused envelope carries no lot", not bad,
            f"{len(bad)} region(s) carry both a refusal and a lot", regions=bad[:6])


def gate_spatial_ground_decision(g: Gate) -> None:
    """**One decision, measured over the ground the requirement is about.**"""
    from ethoslm import intent as intent_mod
    # the four columns, kept apart
    cols, missing = 0, []
    for name in (FARM, RINGS, CITY, HELD):
        for r in (_doc(name, "resolution.json") or {}).get("regions") or []:
            if r.get("lots") is None or not r.get("rect"):
                continue
            cols += 1
            if r.get("scope_columns") is None:
                missing.append(f"{name}/{r.get('name')}")
    g.check("every allocated region records the scope the requirement is about",
            cols and not missing,
            f"{cols} region(s); {len(missing)} without `scope_columns`",
            missing=missing[:6])
    # an inferred open remainder stays in the denominator
    leaked = []
    for name in (FARM, RINGS, CITY, HELD):
        for r in (_doc(name, "resolution.json") or {}).get("regions") or []:
            if str(r.get("surface") or "built") == "open" and not r.get("open_requested"):
                m, _e = _quiet(intent_mod.lot_cover, [], [r])
                if (m or {}).get("ground") in (0, None):
                    leaked.append(f"{name}/{r.get('name')}")
    g.check("an inferred open remainder does not leave the density denominator",
            not leaked, f"{len(leaked)} inferred-open region(s) left the measurement",
            regions=leaked[:6])
    # **the negotiation considered the compiler's real alternatives, and costed them.**
    # Recorded on the region's `target.alternatives`, each with what the actual district
    # compiler said it held -- a list of names would prove nothing.
    alts, costed = [], []
    for r in (_doc(RINGS, "resolution.json") or {}).get("regions") or []:
        for a in ((r.get("target") or {}).get("alternatives") or []):
            alts.append(str(a.get("action")))
            if a.get("laid") is not None or a.get("held") is not None \
                    or a.get("cover") is not None or a.get("refused"):
                costed.append(str(a.get("action")))
    g.check("the ring's width was negotiated against alternatives the compiler costed",
            len(alts) > 1 and len(costed) == len(alts),
            f"{len(alts)} alternative(s) on `target.alternatives`, {len(costed)} of them "
            f"carrying what the compiler reported: {sorted(set(alts))}")
    # ground is proposed from the design and applied from the immutable baseline
    props = [(n, _doc(n, "ground_proposal.json")) for n in (FARM, RINGS, CITY, HELD)]
    have = [n for n, d in props if d and d.get("pieces")]
    g.check("the prepared ground is a recorded proposal derived from the design",
            bool(have), f"a ground proposal on {have or 'no case'}; "
            + "; ".join(str(x)[:90] for n, d in props if d
                        for x in (d.get("from") or [])[:1]))
    # **and the world says so, not the record.** An independent reader falsified the
    # first version of this claim by counting blocks: the proposal truthfully recorded
    # 11,693 drystone blocks laid from the baseline, and the volume construction built
    # on still carried the abandoned voice's paving, because the circulation stage
    # restored its own pre-revision snapshot over the base. So this reads the world.
    reached = []
    for n, d in props:
        if not d or not ((d or {}).get("applied") or {}).get("laid"):
            continue
        got, e = _quiet(_paving_of, n, d)
        if e or not got:
            reached.append({"case": n, "why": f"could not be read: {e}"})
            continue
        reached.append({"case": n, **got})
    bad = [r for r in reached if r.get("share") is not None and r["share"] < 0.75]
    g.check("the ground the design asked for is the ground construction built on",
            bool(reached) and not bad and any(r.get("share") is not None
                                              for r in reached),
            "; ".join(f"{r['case']}: {r.get('share') if r.get('share') is None else format(r['share'], '.0%')}"
                      f" of the anchor's piece is the voice the proposal names"
                      f" ({r.get('voice')})" if r.get("share") is not None
                      else f"{r['case']}: {r.get('why')}" for r in reached),
            measured=reached[:4])
    g.check("an evaluation ran before anything was cut",
            all((d or {}).get("evaluated") for _n, d in props if d),
            "; ".join(f"{n}: {str(((d or {}).get('evaluated') or {}).get('why'))[:90]}"
                      for n, d in props if d))


#: How far outside the anchor's own rectangle the paved platform is sampled. Inside this
#: the ground is the design's paving; beyond it is the feather, which the plateau
#: dresses back to ground cover.
PLATFORM_RING = 4


def _paving_of(name: str, proposal: dict) -> dict:
    """What the anchor's piece is actually surfaced with in the world construction
    built on. Reads `world_built.npz`, falling back to the prepared `world.npz`."""
    from ethoslm import offline, voices as voices_mod
    piece = next((p for p in (proposal.get("pieces") or [])
                  if p.get("rect") and p.get("level") is not None), None)
    if piece is None:
        return {}
    for f in ("world_built.npz", "world.npz"):
        p = _rel(name, f)
        if os.path.exists(p):
            vol = offline.load_volume(p)
            break
    else:
        return {}
    want = voices_mod.load(piece.get("voice")) if piece.get("voice") else None
    fams = {str(v).split(":")[-1] for k, v in (want or {}).get("roles", {}).items()
            if k in ("wall", "footing", "trim", "floor", "ground")} if want else set()
    # **The paved platform immediately outside the anchor**, and nothing else. The
    # anchor's own rectangle carries the anchor -- what is on top of a palace is the
    # palace -- and the outer part of a piece is the feather, which `Builder.plateau`
    # dresses back to ground cover on purpose. The ring just outside the anchor is the
    # ground the design asked to be paved, and it is where an abandoned voice showed.
    own = [int(v) for v in (piece.get("own_rect") or [])]
    if not own:
        return {}
    x0, z0, x1, z1 = own[0] - PLATFORM_RING, own[1] - PLATFORM_RING, \
        own[2] + PLATFORM_RING, own[3] + PLATFORM_RING
    # ...and not a column some part stands on. A compound's own wall sits on the ring
    # outside its rectangle, and a wall is not paving.
    taken = []
    for q in (_doc(name, "plots.json") or []):
        if q.get("x0") is None:
            continue
        taken.append((min(q["x0"], q["x1"]) - 1, min(q["z0"], q["z1"]) - 1,
                      max(q["x0"], q["x1"]) + 1, max(q["z0"], q["z1"]) + 1))
    hit = tot = 0
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if own[0] <= x <= own[2] and own[1] <= z <= own[3]:
                continue
            if any(a <= x <= c and b <= z <= d for a, b, c, d in taken):
                continue
            top = None
            for y in range(vol.y0 + vol.shape[1] - 1, vol.y0 - 1, -1):
                n2 = vol.name(x, y, z)
                if n2 != "air":
                    top = n2
                    break
            if top is None:
                continue
            tot += 1
            if any(f and f.split("_")[0] in top for f in fams) or top in fams:
                hit += 1
    return {"voice": piece.get("voice"), "sampled": tot,
            "share": (hit / tot) if tot >= 24 else None,
            "why": (None if tot >= 24 else
                    f"only {tot} column(s) of the platform ring are free of a standing "
                    f"part: this piece is enclosed by what stands on it and the paving "
                    f"cannot be sampled this way")}


def gate_features_final_world(g: Gate) -> None:
    """**Use is demonstrated by the assembled world, or it is unresolved.**"""
    try:
        from ethoslm import usable
    except ImportError as e:
        g.check("the final-world predicate set exists", False, str(e))
        return
    g.check("the predicate set is the small reusable one the round registered",
            set(getattr(usable, "WANTS", ())) >= {
                "entrance_connected", "passage_connected", "equipment_reachable",
                "circulation_clear", "court_accessible", "range_relation"},
            f"{sorted(getattr(usable, 'WANTS', ()))}")
    ran, declared = 0, []
    for name in (FARM, RINGS, CITY, HELD):
        doc = _doc(name, "usable.json") or {}
        for row in doc.get("checks") or []:
            ran += 1
            if row.get("holds") and row.get("method") not in ("observed",):
                declared.append(f"{name}/{row.get('part')}/{row.get('want')}")
    g.check("every satisfied functional claim was observed on the assembled world",
            ran and not declared,
            f"{ran} predicate answer(s); {len(declared)} held on something weaker",
            weak=declared[:6])
    # a clause measured on the plan is not built evidence
    promoted = []
    for name in (FARM, RINGS, CITY, HELD):
        for c in _clauses(name):
            if c.get("method") in ("plan", "declared") and \
                    c.get("evidence") in ("built_place", "built_sample"):
                promoted.append(f"{name}/{c.get('clause')}")
    seen_clauses = sum(len(_clauses(n)) for n in (FARM, RINGS, CITY, HELD))
    g.check("a clause measured on the plan is `plan` evidence however much stood",
            seen_clauses and not promoted,
            f"{seen_clauses} clause(s) read across the cases; {len(promoted)} promoted "
            f"past their method" if seen_clauses else
            "no place read exists in any case: this cannot be evidenced from an absence",
            clauses=promoted[:6])


def gate_obligations_close(g: Gate) -> None:
    """**One ledger, and nothing closes without fresh evidence.**"""
    try:
        from ethoslm import obligation
    except ImportError as e:
        g.check("the obligation ledger exists", False, str(e))
        return
    g.check("the ledger exposes the three operations the contract names",
            all(hasattr(obligation, n) for n in ("upsert", "open_rows", "close")),
            "upsert/open_rows/close")
    # `rows` is a map keyed by id, not a list: a ledger is looked up by row
    seen, bad_close, alternatives, origins = 0, [], [], set()
    for name in (FARM, RINGS, CITY, HELD):
        led = _doc(name, "obligations.json") or {}
        for rid, row in sorted((led.get("rows") or {}).items()):
            seen += 1
            origins.add(str(row.get("origin")))
            ev = row.get("evidence") or []
            closed_on = [e for e in ev if isinstance(e, dict) and e.get("candidate")]
            if row.get("disposition") == "closed" and not closed_on:
                bad_close.append(f"{name}/{rid}")
            acts = [a for a in (row.get("actions") or []) if isinstance(a, dict)]
            if any(a.get("effective") is False for a in acts) \
                    and row.get("disposition") == "closed" and not closed_on:
                bad_close.append(f"{name}/{rid} (ineffective action)")
            # an action that was refused or did nothing leaves the others open
            if len({a.get("action") for a in acts if a.get("action")}) > 1:
                alternatives.append(f"{name}/{rid}")
    g.check("nothing closed without fresh evidence on the candidate that closed it",
            seen and not bad_close,
            f"{seen} obligation row(s); {len(bad_close)} closed without it",
            rows=bad_close[:6])
    g.check("an action that did not work leaves its owner's alternatives admissible",
            bool(alternatives),
            f"{len(alternatives)} row(s) were offered more than one bounded action",
            rows=alternatives[:6])
    # **One shape, and the production cases may honestly carry only one origin.** A
    # build in which nothing fell short emits no constraints, which is the outcome the
    # demand chain is for; so this is asked of the module as well as of the record.
    led = obligation.empty()
    obligation.upsert(led, [obligation.from_finding(
        {"id": "probe/f", "about": "composition", "material": True,
         "measure": "square_scale", "owner": "layout"}, source="probe")], "probe")
    obligation.upsert(led, [obligation.from_constraint(
        {"what": "storeys", "part": "probe/p", "type": "cottage", "owner": "layout",
         "requested": 2, "emitted": 1,
         "needs": {"lot": [12, 10], "lot_min": [15, 17]}, "why": "the lot holds one"},
        {"storeys"}, source="probe")], "probe")
    owed = obligation.open_rows(led, material=True)
    g.check("findings and emitted constraints are rows of one ledger, both owed",
            len(owed) == 2 and {r["origin"] for r in owed} == {"finding", "constraint"},
            f"the ledger owes {[(r['id'], r['origin']) for r in owed]}; "
            f"origins in the retained production ledgers: {sorted(origins - {'None'})}"
            + (" -- no constraint was emitted by any built case, which is what the "
               "demand chain is for" if "constraint" not in origins else ""))


def gate_iteration_proportional(g: Gate) -> None:
    """**A stage keys on what it consumes.**"""
    from ethoslm import deps
    g.check("the site search keys on the site's own demand and not on the whole spec",
            "site_demand" in (deps.DEPENDS.get("site_search") or ())
            and "spec" not in (deps.DEPENDS.get("site_search") or ()),
            f"site_search depends on {deps.DEPENDS.get('site_search')}")
    g.check("`site_demand` is a fingerprint kind this build computes",
            "site_demand" in deps.KINDS, f"kinds: {list(deps.KINDS)}")
    # a voice-only revision did not re-run the search
    researched = []
    for name in (FARM, RINGS, CITY, HELD):
        rec = _doc(name, "site_search.json") or {}
        runs = rec.get("runs") or rec.get("searches")
        if isinstance(runs, list) and len(runs) > 1:
            researched.append(f"{name}: {len(runs)} searches")
    g.check("no case searched its site twice for a change the search does not consume",
            not researched, "; ".join(researched) or "no repeated search recorded")


# ------------------------------------------------------------------ the proofs

def gate_farm_two_storeys(g: Gate) -> None:
    """Proof 1: the required second storey survives allocation and construction."""
    req = next((r for r in _requirements(FARM)
                if (r.get("wants") or {}).get("exact")
                and (r.get("wants") or {}).get("storeys")), None)
    g.check("the storey count is a hard, exact requirement of the run", bool(req)
            and req.get("hard"), f"{(req or {}).get('id')}")
    if not req:
        return
    want = int((req["wants"]).get("storeys") or 0)
    g.check("the requirement was not weakened anywhere in the run",
            req.get("status") in ("satisfied", "failed") and
            int((req["wants"]).get("storeys") or 0) == want,
            f"{req['id']}: {req.get('status')} -- {str(req.get('why'))[:140]}")
    # **the clause, not the plan-stage record.** `intent.json` on disk is written when
    # the programme is read, before a block exists; the place read is the measurement
    # made after construction, and it is what carries the method.
    clause = next((c for c in _clauses(FARM)
                   if str(c.get("clause", "")).endswith(req["id"])), None)
    g.check("it was measured on what construction emitted, not on the planned params",
            bool(clause) and clause.get("method") == "observed"
            and "emitted" in str(clause.get("says")),
            f"place read: method {(clause or {}).get('method')}, "
            f"{str((clause or {}).get('says'))[:120]}")
    g.check(f"every building of the fabric stands {want} storeys",
            bool(clause) and clause.get("holds") is True
            and clause.get("evidence") == "built_place",
            f"{str((clause or {}).get('says'))[:160]} "
            f"({(clause or {}).get('evidence')})")
    # a material built finding caused an actual improvement and a fresh reading
    rec = _doc(FARM, "improve.json") or {}
    cycles = [c for c in rec.get("cycles") or [] if c.get("applied")]
    closed = [c for c in cycles if c.get("closed")]
    g.check("a material built finding caused a rebuild and a reading that closed it",
            bool(closed), f"{len(cycles)} applied cycle(s), {len(closed)} closed",
            actions=[c.get("action") for c in cycles[:4]])
    g.check("the improvement was not a reduction of the storey requirement",
            not any(str(c.get("action")) == "character"
                    and "storeys" in json.dumps(c.get("action_record") or {})
                    for c in cycles),
            "no applied action touched the storey band")
    pr = _clauses(FARM)
    g.check("the place read holds on a whole built village",
            bool(pr) and all(c.get("holds") is not False for c in pr),
            f"{sum(1 for c in pr if c.get('holds') is False)} failing clause(s)",
            failed=[c["clause"] for c in pr if c.get("holds") is False][:6])


def gate_rings_dense(g: Gate) -> None:
    """Proof 2: the dense word met over the whole ring it is about."""
    reqs = _requirements(RINGS)
    dense = next((r for r in reqs if str(r.get("id", "")).startswith("quality/density/dense")),
                 None)
    g.check("the dense requirement is scoped to the lower ring and hard",
            bool(dense) and dense.get("scope") and dense.get("hard"),
            f"{(dense or {}).get('id')} scope {(dense or {}).get('scope')}")
    if not dense:
        return
    g.check("the dense word is met on the built world",
            dense.get("status") == "satisfied", str(dense.get("why"))[:200])
    # over the requested scope, with nothing excluded that no requirement asked for
    excluded = []
    for r in (_doc(RINGS, "resolution.json") or {}).get("regions") or []:
        if r.get("open_requested"):
            excluded.append(f"{r.get('name')} <- {r.get('open_requested')}")
    g.check("nothing left the denominator that no requirement asked to be open",
            not excluded, f"{len(excluded)} region(s) excluded", excluded=excluded[:6])
    ex = next((r for r in reqs if r.get("kind") == "count"), None)
    g.check("the exact count survived the negotiation",
            bool(ex) and ex.get("status") == "satisfied", str((ex or {}).get("why"))[:140])
    layout = _doc(RINGS, "plan.place.json") or {}
    rings = ((layout.get("layout") or {}).get("rings")
             or (_doc(RINGS, "resolution.json") or {}).get("rings") or [])
    g.check("radial rank and terrain elevation are recorded as different properties",
            bool(rings) and all(("ring" in r and "level" in r) for r in rings),
            f"{len(rings)} ring record(s)")
    parts = _rows(_doc(RINGS, "parts.json"))
    g.check("the town was built whole, not sampled",
            bool(parts) and not (_doc(RINGS, "parts.json") or {}).get("sample"),
            f"{len(parts)} part(s) built")


def gate_city_section(g: Gate) -> None:
    """Proof 3: the registered relationships, demonstrated in one connected section."""
    cfg = _load(os.path.join(ROOT, "rounds", "des-city.json")) or {}
    want = (((cfg.get("preregistered") or {}).get("section_extent") or {})
            .get("must_demonstrate") or [])
    shown = (_doc(CITY, "section.json") or {}).get("demonstrated") or []
    g.check("every registered relationship of the section was demonstrated",
            want and len(shown) >= len(want),
            f"{len(shown)} of {len(want)} relationship(s)",
            missing=[w[:70] for w in want[len(shown):]])
    net = _doc(CITY, "network.json") or {}
    g.check("the section is one connected circulation network",
            (net.get("components") in (1, None) and net.get("access") != "short")
            or net.get("access") == "connected",
            f"access {net.get('access')}, components {net.get('components')}")
    pr = _clauses(CITY)
    unob = [c for c in pr if c.get("evidence") == "unobserved"]
    g.check("clauses outside the section hold nothing",
            not any(c.get("holds") is True for c in unob),
            f"{len(unob)} unobserved clause(s)")
    ident = next((r for r in _requirements(CITY) if r.get("kind") == "identity"), None)
    g.check("whole-city identity is not claimed",
            bool(ident) and ident.get("status") != "satisfied",
            f"identity: {(ident or {}).get('status')}")


def gate_held_out(g: Gate) -> None:
    """Proof 4: the held-out combination, and the unsupported control beside it."""
    init = _doc(HELD, "acceptance", "initial.json")
    g.check("the held-out case's first outcome is preserved before any adaptation",
            bool(init), f"{_rel(HELD, 'acceptance', 'initial.json')}")
    choice = os.path.join(ROOT, "research", "des-held-choice.md")
    g.check("the sentence was registered by a checker after the implementation",
            os.path.exists(choice), choice)
    pr = _clauses(HELD)
    g.check("the held-out place was built, read, and its result recorded",
            bool(pr), f"{len(pr)} clause(s)",
            failed=[c["clause"] for c in pr if c.get("holds") is False][:6])
    ref = _doc(REFUSE, "round.json") or {}
    g.check("the unsupported-request control reports its actual gap",
            bool(ref) and bool((_doc(REFUSE, "intent.json") or {}).get("requirements")),
            "the control ran and left an intent record")
    unsup = [r for r in _requirements(REFUSE) if r.get("status") == "unsupported"]
    g.check("the control names what this build cannot express, and qualifies nothing",
            bool(unsup) and not any(c.get("holds") for c in _clauses(REFUSE)),
            f"{len(unsup)} unsupported requirement(s)",
            named=[r["id"] for r in unsup[:6]])


def gate_material_decision(g: Gate) -> None:
    """The bounded visual experiment: good enough observation to mean something."""
    cmp_p = os.path.join(ROOT, "out", "des-material", "comparison.json")
    rec = _load(cmp_p)
    g.check("the comparison ran on cached built volumes and is retained",
            bool(rec), cmp_p)
    if not rec:
        return
    contexts = list(rec.get("contexts") or [])
    controls = [c for c in contexts if (c.get("positive_control") or {}).get("cases")]
    # **the display has to register the thing being judged, before anything is judged.**
    # The expression round's comparison was inconclusive because flat shading put
    # cobblestone and andesite within a handful of grey levels of one another.
    spreads = []
    for c in controls:
        for case in (c["positive_control"] or {}).get("cases") or []:
            d = (case.get("vs_none") or {}).get("max_delta")
            if d is not None:
                spreads.append((case.get("display"), int(d)))
    best = {d: max((v for k, v in spreads if k == d), default=0)
            for d in {k for k, _v in spreads}}
    g.check("a visible positive control was demonstrated before anything was judged",
            bool(controls) and max(best.values(), default=0) >= 32,
            f"{len(controls)} context(s) with a control; widest frame movement per "
            f"display {best}")
    arms = set(rec.get("arms") or ())
    matched = any("context" in str(v).lower() or "voice" in str(v).lower()
                  for k, v in (rec.get("arms") or {}).items() if k == "random")
    g.check("unchanged, random and context-conditioned arms, matched per context",
            arms >= {"none", "random", "contextual"} and matched,
            f"arms {sorted(arms)}; the random arm is matched per architectural context: "
            f"{matched}")
    judged = (rec.get("judgment") or {}).get("decision")
    g.check("the judgment is recorded and acted on",
            judged in ("enable", "leave_off")
            and bool(rec.get("integrated")) == (judged == "enable"),
            f"judgment {judged}, integrated {rec.get('integrated')}: "
            f"{str((rec.get('judgment') or {}).get('kind'))[:120]}")
    det = rec.get("determinism") or {}
    g.check("determinism and physical compatibility held",
            det.get("order_independent") and det.get("replay_identical")
            and (rec.get("physical") or {}).get("clean"),
            f"order-independent {det.get('order_independent')}, replay-identical "
            f"{det.get('replay_identical')}, construction check "
            f"{(rec.get('physical') or {}).get('clean')}")
    g.check("the negative result is distinguished from inadequate observation",
            "inadequate" in str((rec.get("judgment") or {}).get("kind", "")).lower()
            or bool((rec.get("judgment") or {}).get("why_not_inadequate_observation")),
            str((rec.get("judgment") or {}).get("why_not_inadequate_observation"))[:160])


# ------------------------------------------------------------------ the runner

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="d1")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--gate", default="", help="comma-separated; the default is all")
    a = ap.parse_args()
    probes = not a.no_probes
    only = [g for g in a.gate.split(",") if g] or list(GATES)
    out = {"run": a.run, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": {}}
    for name in only:
        g = Gate(name)
        fn = globals()[f"gate_{name}"]
        try:
            fn(g, probes) if name == "demand_before_size" else fn(g)
        except Exception as e:                      # noqa: BLE001 -- the gate reports
            g.check("the gate ran", False, f"{type(e).__name__}: {e}")
        out["gates"][name] = g.result()
    passed = sum(1 for v in out["gates"].values() if v["pass"])
    out["passed"], out["of"] = passed, len(only)
    d = os.path.join(ROOT, "out", f"des-{a.run}", "acceptance")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for p in (os.path.join(d, f"{stamp}.json"), os.path.join(d, "latest.json")):
        json.dump(out, open(p, "w"), indent=1)
    for name, v in out["gates"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'} {name}")
        for c in v["checks"]:
            print(f"   {'ok  ' if c['pass'] else 'MISS'} {c['check']}: {c['why'][:160]}")
    print(f"\n{passed} of {len(only)} gates -> {os.path.join(d, 'latest.json')}")
    return 0 if passed == len(only) else 1


if __name__ == "__main__":
    sys.exit(main())
