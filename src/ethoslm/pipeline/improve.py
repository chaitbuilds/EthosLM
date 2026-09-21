"""**Built findings govern completion.** The expression round's loop that matters.

The closure round could observe the built world (`stage_inspect`) and could not act on
what it saw: the farm finished with its square thirteen times its cottages, thin fields
and nine cottages a storey short, every one of them recorded and none of them anybody's
to fix. This stage sits between the inspection and the qualification and does three
things, in this order:

  1. **Every finding of the built reading gets a disposition** -- `inspection/
     dispositions.json` -- and so does every emitted construction constraint. A material
     finding (the reading says `material: true`, or names a contextual target) is
     something the place is not yet finished without; an optional one is a suggestion
     and is `accepted` or `deferred` with the reason, never silently dropped.
  2. **One material finding is routed to its owner's bounded action** and applied:
     `layout` findings to `placesolve.reallocate` (the parent allocation reconsidered
     from the programme, the affected districts recompiled through the district
     compiler), `fabric`/`voice` findings that carry a `character` or `voice` action to
     the same revision machinery the preview uses, `build` findings whose constraint
     names a lot the layout can give to `enlarge_lots`. The action is recorded with the
     input that justified it and the measure before it; an action already tried on this
     candidate is refused as an unchanged retry; the owner's **other** actions stay
     admissible, because an action that did nothing is not a reason to stop trying.
  3. **The affected output is rebuilt and read again.** The plan is laid out again from
     the changed allocation, the lanes re-routed, the parts rebuilt, the construction
     check run and the built world drawn and handed to the judge for a fresh reading.
     On the next pass the obligation is offered to `obligation.close`, which takes a
     measurement of the row's own measure on **this** candidate or refuses by name.

Bounded: `flags.improve.cycles_per_candidate` (1) and `cycles_per_lineage` (2).
Exhausted budget leaves the finding `open`, and `stage_place_check` blocks completion on
an open material finding. Nothing here weakens a check.

**The design round moved the bookkeeping into `ethoslm.obligation`.** This stage used to
keep two lists -- the judge's findings and the emitted constraints -- and select actions
from the first only, closing a cycle when the next reading's `closed` list happened to
name it. Four things followed, and all four are now the ledger's rules rather than this
stage's: a reading that omits an old finding does not close it; an applied action that
did not move the cited measure leaves the row open **and leaves the other actions
admissible**; a missing measurement closes nothing; and an emitted constraint is an
obligation of the same shape, so it reaches action selection like any other. This stage
is what *applies* an owner's action; what is owed is the ledger's.
"""
from __future__ import annotations

import contextlib
import json
import os
import time

from .. import pipeline as _pipeline

RECORD = "improve.json"
DISPOSITIONS = os.path.join("inspection", "dispositions.json")
DEFAULT_BOUNDS = {"cycles_per_candidate": 1, "cycles_per_lineage": 2}
#: The owners this stage can route to, and the action families each admits.
#: **Discoverable and executable are one list** (the composition round). This table and
#: `placesolve.REALLOCATE_ACTIONS` disagreed in both directions, and the design round
#: paid for both halves of the disagreement: `move_object` was offered to the `layout`
#: owner here and refused unconditionally by `placesolve._reallocate` -- *"moving a part
#: is the relation repair's action, not an allocation"* -- so one of the five
#: alternatives was guaranteed to burn a ledger attempt and do nothing; and the four
#: arrangement actions the same round had just built and certified through the real
#: compiler (`row_depth`, `bay_width`, `frontage`, `compound`) were absent here, so they
#: could never be offered, never be a retry, and never appear in the list an exhausted
#: owner is said to have tried. "Every bounded action the layout owner has" was fifteen
#: attempts over three rows, and the composition actions were not among them. So the
#: table below is the *static* statement of which owner is responsible for what, and
#: `owner_actions()` is what selection uses: the static list, restricted to actions the
#: spatial layer will actually attempt, plus the actions it exports and this table does
#: not know about yet. A list that cannot be executed is not an inventory.
OWNER_ACTIONS = {
    "layout": ("regroup", "resize_anchor", "shrink_anchor", "enlarge_anchor",
               "resize_ring", "grow_land", "enlarge_lots", "redistribute",
               "row_depth", "bay_width", "frontage", "compound"),
    "fabric": ("character", "enlarge_lots", "row_depth", "bay_width", "frontage"),
    "voice": ("voice",),
    "build": ("enlarge_lots",),
    "scale": ("regroup", "resize_anchor", "shrink_anchor", "enlarge_anchor",
              "grow_land", "redistribute", "compound"),
    "capability": (),
}

#: Actions this stage must not offer, with the owner that does own them. `move_object`
#: is the relation repair's (`repair.apply`, which measures where the subjects stand);
#: offering it here produced a refusal and a spent attempt, every time.
NOT_OURS = {"move_object": "the relation repair (`repair.apply`)"}


def owner_actions(owner: str) -> tuple:
    """The actions this owner may be offered: declared here **and** executable there.

        Reads `placesolve`'s own registry where it exports one, so a spatial action added or
        withdrawn changes what selection offers without a second table having to be edited
        in step. Falls back to the static table, which is still filtered against
        `placesolve.REALLOCATE_ACTIONS` and `NOT_OURS`.
        
    """
    from .. import placesolve
    static = tuple(OWNER_ACTIONS.get(owner, ()))
    by_owner = getattr(placesolve, "ACTIONS_BY_OWNER", None)
    if isinstance(by_owner, dict) and by_owner.get(owner):
        got = tuple(by_owner[owner])
    else:
        can = set(getattr(placesolve, "REALLOCATE_ACTIONS", ()) or ())
        # `character` and `voice` are this stage's own revision path, not placesolve's
        mine = {"character", "voice"}
        got = tuple(a for a in static if a in can or a in mine)
    return tuple(a for a in got if a not in NOT_OURS)


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _record(rnd) -> dict:
    return _load(rnd.rel(RECORD)) or {"cycles": [], "per_candidate": {}, "lineage": 1,
                                       "bounds": dict(DEFAULT_BOUNDS)}


def _save(rnd, rec: dict) -> None:
    json.dump(rec, open(rnd.rel(RECORD), "w"), indent=1)


def _bounds(rnd) -> dict:
    got = rnd.flags.get("improve")
    out = dict(DEFAULT_BOUNDS)
    if isinstance(got, dict):
        out.update({k: int(v) for k, v in got.items() if k in out})
    elif got is False or got == 0:
        out["cycles_per_candidate"] = 0
    return out


def is_material(f: dict) -> bool:
    """A finding the place is not finished without. The reading says so outright, or
    names a contextual target; a finding with neither is a suggestion.

    The same rule `obligation.from_finding` applies, kept here because the readout and
    the acceptance runners ask it of a raw reading that has not reached a ledger."""
    if f.get("material") is not None:
        return bool(f.get("material"))
    return bool(f.get("target"))


def _tokens_of(word: str) -> set:
    """The emitted-feature token(s) a requirement's own word is about.

        **A word is not a token.** The design review's `market`/`stalls` disagreement, and it
        is entirely this translation: the sentence's requirement is `feature/market`, so the
        raw word is `market`; the constraint construction emits is about `stalls`; and
        `obligation.from_constraint` computed `material = "stalls" in {"market"}` -> False, so
        the row was never material, never selected, and never acted on. `envelope`'s own
        vocabulary already maps one to the other (`FEATURE_WORDS`: `stalls` is asked for by
        `market`, `stalls`, `bazaar`); this reads that table in the direction the ledger needs.
        
    """
    from .. import envelope
    w = str(word or "").lower()
    if not w:
        return set()
    out = {w}
    for token, words in (getattr(envelope, "FEATURE_WORDS", {}) or {}).items():
        if w == str(token).lower() or w in {str(x).lower() for x in words}:
            out.add(str(token))
    return out


def _required_tokens(intent_rec: dict | None, rnd=None) -> set:
    """Every feature token the request requires of something, by token, not by word.

        An emitted constraint naming one of these is an obligation the place is not finished
        without; one naming anything else was the type's own choice on its lot.

        Where the plan carries the resolved **per-part demand binding** (`demand.resolve`
        writes `required` onto each district), those tokens are used as well: that binding is
        the authority on what is required of what, and re-deriving requirements from the raw
        sentence is how this function came to disagree with the layer it feeds.
        
    """
    out = set()
    for r in (intent_rec or {}).get("requirements") or []:
        if r.get("status") == "unsupported":
            continue
        w = r.get("wants") or {}
        if r.get("kind") == "quality" and w.get("axis") == "height":
            out.add("storeys")
        if r.get("kind") == "feature" and w.get("feature"):
            out |= _tokens_of(w["feature"])
        if r.get("kind") == "function" and (w.get("function") or w.get("what")):
            out |= _tokens_of(w.get("function") or w.get("what"))
    for part, toks in (required_by_part(rnd) or {}).items():
        out |= set(toks)
        del part
    return out


def required_by_part(rnd) -> dict:
    """`{part or district name: {token, ...}}` -- what the demand binding requires of what.

        The resolved demand is on the plan (`placesolve._write_demand` copies the defining
        part's demand onto each district row), which is why this can be read without
        re-resolving anything. Empty where the plan predates the binding, which reads as "not
        bound" and never as "nothing is required".
        
    """
    if rnd is None:
        return {}
    got: dict = {}
    with contextlib.suppress(Exception):
        plan = rnd.plan() or {}
        for d in plan.get("districts") or []:
            dem = d.get("demand") or {}
            toks = {str(t) for t in (dem.get("required") or ())}
            for t in list(toks):
                toks |= _tokens_of(t)
            if toks and d.get("name"):
                got[str(d["name"])] = toks
    return got


def _required_for(rnd, part: str, intent_rec: dict | None) -> set:
    """The tokens required **of this part**, falling back to the request's own set.

        A part-bound answer is the better one: a `stalls` constraint on a cottage is the
        cottage's own business, and the same constraint on the market floor is the request's.
        
    """
    by_part = required_by_part(rnd)
    mine = set()
    for name, toks in by_part.items():
        if str(part).startswith(name):
            mine |= toks
    return mine or _required_tokens(intent_rec, rnd)


def _constraint_rows(rnd, intent_rec: dict | None) -> list:
    """Every emitted construction constraint on the built candidate, as ledger rows."""
    from .. import obligation
    out = []
    for r in [r for w in ((_load(rnd.rel("parts.json")) or {}).get("waves") or [])
              for r in (w.get("parts") or [])]:
        c = r.get("constraint")
        if c:
            out.append(obligation.from_constraint(
                c, _required_for(rnd, str(r.get("part") or ""), intent_rec),
                part=r.get("part"), source="parts.json"))
    return out


def feature_evidence(rnd) -> dict:
    """`{"<part>/<feature>": answer}` -- post-build feature evidence, for closure.

        An `emitted.<feature>` obligation is about a feature of a part on the assembled
        world. Until now closure was offered `inspect.MEASURES` -- `clusters`, `square_scale`,
        `open_to_built` and three more, all of them whole-place composition numbers -- and a
        row citing `emitted.stalls` could never find its measure in that dict, so an emitted
        constraint was structurally unclosable while appearing to be offered a chance. This
        is the measure those rows are actually about: what construction emitted for that
        part, and what the final-world predicates said about it.
        
    """
    out: dict = {}
    rows = [r for w in ((_load(rnd.rel("parts.json")) or {}).get("waves") or [])
            for r in (w.get("parts") or [])]
    for r in rows:
        part = str(r.get("part") or "")
        em = r.get("emitted") or {}
        for feat, got in (em.get("features") or {}).items():
            out[f"{part}/{feat}"] = {
                "holds": bool(got) if got is not None else None,
                "method": (em.get("features_method") or {}).get(feat),
                "source": (em.get("features_source") or {}).get(feat),
                "part": part, "feature": str(feat),
                "rect": (em.get("rects") or {}).get(feat),
            }
        for want, a in (em.get("usable") or {}).items():
            if isinstance(a, dict):
                out[f"{part}/predicate/{want}"] = {
                    "holds": a.get("holds"), "method": a.get("method"),
                    "part": part, "feature": str(want), "why": str(a.get("why"))[:200]}
    return out


def _apply_layout(rnd, be, spec: dict, finding: dict, action_hint: str | None) -> dict:
    """The layout owner's bounded action, through `placesolve.reallocate`, and the plan
    laid out again from the changed allocation. Mirrors the preview's repair pass."""
    from .. import contracts, deps, placesolve
    from . import stages_media, stages_plan
    place_p = rnd.rel("plan.place.json")
    place = _load(place_p)
    if not place:
        return {"applied": False, "refused": "no plan.place.json to reallocate"}
    fn = getattr(placesolve, "reallocate", None)
    if fn is None:
        return {"applied": False, "refused": "placesolve.reallocate is not available yet"}
    site = _pipeline.settlement_site(rnd)
    decls = stages_plan._decls_for(rnd, spec)
    rows = [r for w in ((_load(rnd.rel("parts.json")) or {}).get("waves") or [])
            for r in (w.get("parts") or [])]
    constraints = [r["constraint"] for r in rows if r.get("constraint")]
    env_p = rnd.rel("envelopes.json")
    try:
        place2, act = fn(place, spec, dict(finding, action=action_hint or finding.get("action")),
                         site=site, decls=decls, envelopes=env_p, constraints=constraints,
                         intent=contracts.load(rnd, "intent"))
    except Exception as e:                       # noqa: BLE001 -- the owner reports
        return {"applied": False, "refused": f"reallocate raised {type(e).__name__}: {e}"}
    act = dict(act or {})
    if act.get("refused") or place2 is None:
        return {"applied": False, "refused": act.get("refused") or "no place returned",
                "action": act}
    snap = stages_media._snapshot(rnd)
    json.dump(place2, open(place_p, "w"), indent=1)
    # the allocation decision lives on the spec (`negotiated` rows), so a plan laid out
    # again from the spec reproduces the reallocated place; the spec on disk is the one
    # every later stage reads
    json.dump(spec, open(rnd.rel("place.checked.json"), "w"), indent=1)
    for f in ("plan.json", "plots.json", "network.json", "circulation.json"):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    for f in sorted(os.listdir(rnd.state)):
        if f.startswith(("plan.district.", "district_")) and os.path.isfile(rnd.rel(f)):
            os.remove(rnd.rel(f))
    with contextlib.suppress(ValueError):
        deps.stamp(rnd, "plan", outputs=["plan.place.json"],
                   note=f"kept across the improve stage's {act.get('action')}: "
                        f"{str(act.get('why'))[:120]}")
    again = stages_media._replan(rnd, be)
    p = (again.get("plan") or {}) if isinstance(again, dict) else {}
    ok = (not p.get("stop") and p.get("status") not in ("error", "needs_model")
          and os.path.exists(rnd.rel("plan.json")))
    if not ok:
        stages_media._restore(rnd, snap)
        return {"applied": False, "rolled_back": True, "action": act,
                "refused": (f"laid out again the place " + (f"fails at {p.get('level')}: "
                            f"{p.get('error')}" if p.get("error") else "did not plan")
                            + "; the allocation stands as it was")}
    circ = _pipeline.stage_circulation(rnd, be, {})
    return {"applied": True, "action": act, "plan": {k: p.get(k) for k in ("status", "level")},
            "circulation": {k: circ.get(k) for k in ("status", "error") if k in circ}}


def _apply_revision(rnd, be, spec: dict, doc: dict) -> dict:
    """A character or voice action through the preview's own revision machinery."""
    from . import stages_media
    got = stages_media._apply_revision(rnd, be, spec, doc)
    if got.get("rolled_back") or not (got.get("characters") or got.get("voice")):
        return {"applied": False, "rolled_back": bool(got.get("rolled_back")),
                "refused": "; ".join(got.get("refused") or []) or "nothing applied",
                "action": {"action": "character" if doc.get("characters") else "voice"}}
    return {"applied": True, "action": {"action": "character" if doc.get("characters")
                                        else "voice", **{k: got.get(k) for k in
                                                         ("characters", "voice", "lineage")}}}


def _rebuild(rnd, be, results: dict) -> dict:
    """Parts, finish, the construction check, the inspection: the affected output built
    and looked at again. Returns the first stop or pending, else the inspection result."""
    out = {}
    # **From the prepared ground, not from the world the first build left.** The farm's
    # first improve cycle rebuilt onto the backend's in-memory volume, which still held
    # the previous candidate's cottages, and the check found three doorways walled in.
    # The dry backend's volume is forgotten so the next read is off `world.npz`, and the
    # previous candidate's built world and parts record are set aside by name.
    if hasattr(be, "refresh"):
        be.refresh()
    for f in ("world_built.npz", "parts.json", "surfaces.json"):
        if os.path.exists(rnd.rel(f)):
            os.replace(rnd.rel(f), rnd.rel(f"stale.improve.{f}"))
    # **The ground is asked for again, from the baseline.** The design round: an action
    # that moved the anchor or the voice changes what the design asks of the ground, and
    # the expression round's held-out village kept its first voice's paving under a
    # chapel that had been re-sized. `stage_ground` re-proposes from
    # `deps.baseline_path` and re-applies from it, so the second cut of a design that
    # changed its mind is the first cut of the design it changed to; the terraces stage
    # then re-lays on it. Both run before construction, else the parts stage refuses
    # ground "cut for a different candidate".
    for name in ("ground", "terraces", "parts", "finish", "lint", "material",
                 "section", "inspect"):
        res = _pipeline.STAGES[name](rnd, be, results)
        from .round import _drive_reentries, _needs_model, _stopped
        res = _drive_reentries(rnd, be, results, name, res)
        out[name] = res
        results[name] = res
        if _stopped(res):
            return {"stopped": name, "why": _stopped(res), "stages": out}
        if _needs_model(res):
            return {"pending": name, "res": res, "stages": out}
    return {"stages": out}


def _settle_preview(rnd, finding_id) -> None:
    """The pre-build reading is not repeated for a candidate the improve stage moved
    after construction: the preview record is stamped done for the new plan, with the
    reason. The reading that qualifies this candidate is the built one."""
    from . import stages_media
    d = rnd.rel("preview")
    rec_p = os.path.join(d, "preview.json")
    rec = _load(rec_p) or {}
    rec["settled_by_improve"] = (f"the improve stage moved the candidate after construction "
                                 f"acting on {finding_id}; the plan is not read again before "
                                 f"the rebuild, the built world is")
    with contextlib.suppress(Exception):
        stages_media._finish_preview(rnd, rec, rec_p, rnd.plan())


def _findings(rnd) -> tuple:
    rec = _load(rnd.rel("inspection", "views.json")) or {}
    reading = rec.get("reading") or {}
    return rec, list(reading.get("findings") or []), list(reading.get("closed") or [])


def _why_of(r: dict) -> str:
    """Why a row stands where it does, for the `dispositions.json` projection."""
    acts = r.get("actions") or []
    if r.get("disposition") == "open" and acts:
        last = acts[-1]
        return (f"acted on with `{last.get('action')}`"
                + (f" and the measure did not move ({last.get('why')})"
                   if last.get("effective") is False else
                   f"; {last.get('why')}" if not last.get("applied") else
                   "; awaiting the reading of the rebuilt world"))
    if r.get("disposition") == "open" and not r.get("material"):
        return "optional and nobody has disposed of it"
    if r.get("disposition") == "open":
        return "material; not yet acted on"
    return r.get("why") or ""


def _act_name(f) -> str | None:
    """The action a finding or a ledger row asks for, however the reader wrote it."""
    h = f.get("action")
    if isinstance(h, dict):
        return h.get("type")
    if h:
        return str(h)
    acc = f.get("acceptance")
    if isinstance(acc, dict) and acc.get("action"):
        return str(acc["action"])
    return None


def _hint_of(findings: list, rid: str):
    """The reader's own action record for a row, where the row came from a finding."""
    for f in findings or []:
        if f.get("id") == rid:
            return f.get("action")
    return None


def stage_improve(rnd, be, results: dict) -> dict:
    from .. import contracts, deps, obligation
    plan = rnd.plan()
    if not plan or not rnd.sentence:
        return {"skipped": "no plan or no sentence: nothing built to improve"}
    views, findings, closed_ids = _findings(rnd)
    if not views.get("reading"):
        return {"skipped": "no reading of the built world yet"}
    here = deps.candidate_id(rnd, plan=plan)
    if views.get("candidate") != here:
        return {"skipped": f"the inspection on disk is of {views.get('candidate')} and "
                           f"this is {here}; nothing to act on until it is read"}
    rec = _record(rnd)
    bounds = _bounds(rnd)
    rec["bounds"] = bounds
    spec = rnd.place_spec() or {}
    it = contracts.load(rnd, "intent")
    measures = dict(views.get("measures") or {})

    # --- everything owed, in one ledger ---------------------------------------------
    # **Both producers, one shape.** A judged finding of the built reading and an
    # emitted construction constraint are the same kind of thing -- something the place
    # owes -- and the expression round kept them in two lists and acted on one.
    led = obligation.load(rnd.state)
    obligation.upsert(led, [obligation.from_finding(f, source=f"reading/{here}")
                            for f in findings],
                      f"reading/{here}", candidate=here)
    obligation.upsert(led, _constraint_rows(rnd, it), f"parts/{here}", candidate=here)

    # --- close what this reading actually settles ------------------------------------
    # A row is closed by **evidence on this candidate**, never by a reader's silence.
    # `omitted_by` on the ledger says which passes did not mention it; that is a fact
    # about the readings, not a disposition.
    feats = feature_evidence(rnd)
    closed_now, refused_close = [], []
    for c in rec["cycles"]:
        if not c.get("applied") or c.get("candidate") == here:
            continue
        rid = c["finding"]
        if rid not in led.get("rows", {}):
            continue
        got = obligation.effect(led, rid, c["candidate"], c.get("measures_before"),
                                measures)
        c["reinspected"], c["reinspected_candidate"] = True, here
        c["measure"] = got
        # **The evidence a row's own measure lives in.** `measures` is the whole place's
        # composition numbers (`inspect.MEASURES`); `features` is what construction
        # emitted per part and what the final-world predicates said about it, which is
        # where an `emitted.<feature>` row's measure actually is. Offering a row only
        # the first dict is why emitted constraints could never close.
        shut = obligation.close(led, rid, {"candidate": here, "measures": measures,
                                           "features": feats,
                                           "before": (c.get("measures_before") or {}).get(
                                               got.get("measure")),
                                           "observed": rid in closed_ids},
                                candidate=here, reading=views.get("reading"))
        c["closed"] = bool(shut["closed"])
        c["why_open"] = None if shut["closed"] else shut["why"]
        (closed_now if shut["closed"] else refused_close).append(rid)
        print(f"   improve: cycle {c['cycle']} ({rid} -> {c['action']}): "
              f"{'CLOSED' if c['closed'] else 'still open'}; {shut['why']}", flush=True)

    # --- one bounded action, chosen from everything still owed -----------------------
    # the backend's working volume holds this candidate's construction; every action
    # below replans and re-routes the lanes on the prepared ground, never on the build
    if hasattr(be, "refresh"):
        be.refresh()
    spent_here = int(rec["per_candidate"].get(here, 0))
    spent_lineage = sum(1 for c in rec["cycles"] if c.get("applied")
                        and c.get("lineage") == rec.get("lineage"))
    can = (spent_here < bounds["cycles_per_candidate"]
           and spent_lineage < bounds["cycles_per_lineage"])
    owed = obligation.open_rows(led, material=True)
    chosen = None
    if owed and can:
        for r in owed:
            owner = str(r.get("owner") or "")
            hint = _hint_of(findings, r["id"])
            act_name = _act_name({"action": hint, "acceptance": r.get("acceptance")}) \
                or _act_name(r)
            # **an ineffective action leaves the alternatives open.** Only the same
            # action on the same candidate is the unchanged retry this refuses.
            left = obligation.admissible(led, r["id"], here, owner_actions(owner))
            if act_name is None or obligation.tried_here(led, r["id"], here, act_name):
                act_name = left[0] if left else None
            # **The alternatives are tried, not merely admissible.** The round's rule is
            # that an action which was refused or did nothing leaves its owner's others
            # open; a pass offering one action per row and moving on made that true on
            # paper and false in the run, and an independent reader said so. `queue` is
            # this row's remaining actions in its owner's own order, bounded by that
            # list -- five at most -- and the first that applies wins.
            queue = [a for a in [act_name] + list(left) if a] or [None]
            if act_name is None:
                # **exhausted is a state, not an event.** Recorded once per candidate:
                # every later pass would otherwise append the same empty row and the
                # ledger would say twenty times over what it said the first time.
                if not obligation.tried_here(led, r["id"], here, None):
                    obligation.act(led, r["id"], None, here, applied=False,
                                   why=(f"owner `{owner}` has no bounded action left "
                                        f"for `{r.get('about')}` on this candidate; "
                                        f"tried "
                                        f"{[a.get('action') for a in r.get('actions') or [] if a.get('action')]}"))
                continue
            before = dict(measures)
            f = next((x for x in findings if x.get("id") == r["id"]),
                     {"id": r["id"], "says": r.get("says"), "owner": owner,
                      "subjects": r.get("subjects"), "target": r.get("target"),
                      "measure": r.get("measure"), "action": act_name,
                      "acceptance": r.get("acceptance")})
            got, tried_any = {"applied": False}, False
            for act_name in queue:
                if obligation.tried_here(led, r["id"], here, act_name):
                    continue
                tried_any = True
                if owner in ("layout", "scale", "build") or (
                        owner == "fabric" and act_name == "enlarge_lots"):
                    got = _apply_layout(rnd, be, spec, f, act_name)
                elif owner in ("fabric", "voice") and isinstance(hint, dict) and (
                        hint.get("characters") or hint.get("voice")):
                    got = _apply_revision(rnd, be, spec, {
                        "characters": hint.get("characters") or {},
                        "voice": hint.get("voice"), "caused_by": [r["id"]],
                        "why": f"the improve stage acting on {r['id']}: {r.get('says')}"})
                else:
                    got = {"applied": False,
                           "refused": f"owner `{owner}` has no bounded action for "
                                      f"`{act_name or r.get('about')}` in this build"}
                obligation.act(led, r["id"], act_name, here,
                               applied=bool(got.get("applied")),
                               why=str(got.get("refused") or "applied"),
                               record=got.get("action"))
                if got.get("applied"):
                    break
                print(f"   improve: {r['id']} ({owner}) -> {act_name}: refused -- "
                      f"{str(got.get('refused'))[:200]}", flush=True)
            if not tried_any:
                continue
            cyc = {"cycle": len(rec["cycles"]) + 1, "candidate": here,
                   "lineage": rec.get("lineage"), "finding": r["id"],
                   "origin": r.get("origin"), "says": r.get("says"), "owner": owner,
                   "action": (got.get("action") or {}).get("action") or act_name,
                   "hint": act_name,
                   "action_record": got.get("action"), "applied": bool(got.get("applied")),
                   "refused": got.get("refused"), "rolled_back": got.get("rolled_back"),
                   "target": r.get("target"), "measures_before": before,
                   "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            rec["cycles"].append(cyc)
            if got.get("applied"):
                rec["per_candidate"][here] = spent_here + 1
                chosen = cyc
                _settle_preview(rnd, r["id"])
                print(f"   improve: {r['id']} ({owner}) -> {cyc['action']} applied; "
                      f"the affected output is rebuilt and read again", flush=True)
                break
    elif owed and not can:
        print(f"   improve: {len(owed)} obligation(s) still owed and the budget is spent "
              f"({spent_here} of {bounds['cycles_per_candidate']} on this candidate, "
              f"{spent_lineage} of {bounds['cycles_per_lineage']} in this lineage); "
              f"incomplete, not qualified", flush=True)

    # --- the record ------------------------------------------------------------------
    obligation.save(rnd.state, led)
    os.makedirs(rnd.rel("inspection"), exist_ok=True)
    # `dispositions.json` stays, because the readout and the acceptance runners read it;
    # it is now a projection of the ledger rather than a second place where a decision
    # lives. the projection reads the ledger's own rows, not `obligation.table`'s
    # flattening: `why` has to be derived from the actions as records, and the table
    # renders those as strings for a readout
    rows = [{**{k: r.get(k) for k in ("id", "about", "says", "subjects", "measure",
                                       "owner", "target", "acceptance")},
             "origin": r.get("origin"), "material": bool(r.get("material")),
             "disposition": r.get("disposition"),
             "why": r.get("why") or _why_of(r),
             "actions": [a.get("action") for a in r.get("actions") or []],
             "omitted_by": r.get("omitted_by") or []}
            for _rid, r in sorted((led.get("rows") or {}).items())]
    disp = {"candidate": here, "built_digest": views.get("built_digest"),
            "findings": [r for r in rows if r["origin"] == "finding"],
            "constraints": [r for r in rows if r["origin"] == "constraint"],
            "ledger": rnd.rel(obligation.RECORD),
            "open_material": len(obligation.open_rows(led, material=True)),
            "undisposed_optional": len(obligation.undisposed(led)),
            "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(disp, open(rnd.rel(DISPOSITIONS), "w"), indent=1)
    _save(rnd, rec)
    if chosen is None:
        return {"candidate": here, "obligations": len(rows),
                "open_material": disp["open_material"],
                "undisposed_optional": disp["undisposed_optional"],
                "closed": closed_now, "still_open": refused_close,
                "cycles": len(rec["cycles"]),
                "dispositions": rnd.rel(DISPOSITIONS), "ledger": disp["ledger"],
                "written": rnd.rel(RECORD)}

    # --- rebuild and read again -----------------------------------------------------
    for f in ("world_built.npz",):
        pass                                     # the parts stage rebuilds a moved candidate
    got = _rebuild(rnd, be, results)
    chosen["rebuilt"] = "parts" in got.get("stages", {})
    chosen["rebuild_stopped"] = got.get("stopped")
    chosen["rebuilt_candidate"] = deps.candidate_id(rnd)
    _save(rnd, rec)
    if got.get("stopped"):
        return {"status": "blocked", "stop": True, "blocks": "construction",
                "error": f"after the improve stage's {chosen['action']} the rebuild stopped "
                         f"at {got['stopped']}: {got['why']}",
                "cycle": chosen}
    if got.get("pending"):
        return {**got["res"], "cycle": chosen,
                "note": (f"the rebuilt world is drawn and awaits its reading; the improve "
                         f"stage closes cycle {chosen['cycle']} against it")}
    return {"status": "reenter", "why": "the affected output was rebuilt and read; the "
                                         "cycle is closed against the new reading",
            "cycle": chosen}
