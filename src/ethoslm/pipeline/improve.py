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
import hashlib
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
               "row_depth", "bay_width", "frontage", "compound", "relevel"),
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
    # the parent's piece level is this stage's own revision path (`_apply_relevel`)
    if owner == "layout" and "relevel" not in got:
        got = got + ("relevel",)
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
    parts_rec = _load(rnd.rel("parts.json"))
    rows = [r for w in ((parts_rec or {}).get("waves") or [])
            for r in (w.get("parts") or [])]
    constraints = [r["constraint"] for r in rows if r.get("constraint")]
    env_p = rnd.rel("envelopes.json")
    try:
        # **The certified comparison is measured on what this candidate actually
        # built.** `arrange.alternatives` takes the parts record so `region_columns` and
        # `street_enclosure` read emitted footprints rather than the plan's pads where
        # there are any; the controller adopts that comparison's first eligible row, so
        # the record has to reach it. **An action this stage routed is not an action the
        # reading named.** The neighbourhood round: this stage overwrites `action` with
        # its own hint, so `_reallocate` could not tell a reader who asked for a depth
        # decision from a dispatcher walking the owner's list -- and it restricts the
        # certified comparison to the named action, correctly, for the first and wrongly
        # for the second. `action_routed` carries that difference, and `action_from` on
        # the record then says which rule chose.
        routed = bool(action_hint and not finding.get("action"))
        # **The re-solve is given the ground.** The spatial design round: this call
        # passed no volume, so `placesolve._reallocate` -> `concentric_layout` fell
        # through `plateau["terrace"]` (a stub with a rectangle and nothing else) to
        # `site_median(None, site)` -- and every revision of this project's history was
        # laid out by a layout that could not read the terrain it was laying out on. The
        # backend's volume is the prepared ground of the candidate being revised, which
        # is what the re-solve is a revision *of*.
        vol = None
        with contextlib.suppress(Exception):
            vol = be.volume
        place2, act = fn(place, spec,
                         dict(finding, action=action_hint or finding.get("action"),
                              action_routed=routed),
                         site=site, decls=decls, envelopes=env_p, constraints=constraints,
                         parts_record=parts_rec, vol=vol,
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
    from .. import local as _local_n
    for f in ("plan.json", "plots.json") + (
            ("network.json", "circulation.json") if _local_n.scope_of(rnd) is None
            else ()):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    # **A re-solved place regenerates everything the geometry it changed was planned
    # against**, the neighbourhood round: districts *and* compounds. A compound is
    # planned against the road that arrives at it (`compound_failures`' gate/road
    # check), and keeping the palace's plan across a re-solve is what refused every one
    # of the composition round's eleven layout actions with the same message.
    # `_stage_arterials` enforces the same rule on the ordinary path; this is the
    # revision's own.
    from .. import local as _local_r
    _local_r.retire_plans(rnd, compounds=True)
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


def _estimate_vs_built(rnd, cycle: dict) -> dict | None:
    """**What the certified comparison estimated, beside what construction emitted**, on
        the districts one applied action actually refabricated.

        Both figures over the same denominator (the district's developable ground), both
        named for what they are, and `unmeasured` where construction reported no footprint
        for that district's leaves. This is the round's "label estimates honestly and compare
        them with emitted geometry after building", made a record rather than a sentence.
        
    """
    act = cycle.get("action_record") or {}
    cert = act.get("certified") or {}
    touched = list((act.get("refabricated") or {}).keys())
    if not touched:
        # **An action with no fabric estimate says so, rather than carrying nothing.**
        # The neighbourhood delivery round: `certified.on` is set only for arrangement
        # actions, so an anchor or a ring action reached this function, returned `None`,
        # and the cycle carried no comparison at all -- which reads on the record as a
        # missing measurement rather than as an action of a kind that has no fabric
        # estimate to compare. It is an honest answer either way; it was not written
        # down.
        return {"on": cert.get("on"), "districts": [],
                "estimate": None,
                "why": (f"`{cycle.get('action')}` refabricated no district: it is not a "
                        f"fabric decision and `arrange.alternatives` certified no lot "
                        f"estimate for it, so there is no estimate to set beside what "
                        f"construction emitted. What it changed is on "
                        f"`action_record` itself"),
                "note": "an action of a kind that carries no fabric estimate"}
    from .. import placeplan
    place = _load(rnd.rel("plan.place.json")) or {}
    parts_rec = _load(rnd.rel("parts.json"))
    by_name = {d.get("name"): d for d in (place.get("districts") or [])}
    out = []
    for name in touched:
        d = by_name.get(name)
        if not d or d.get("x1") is None:
            continue
        plan = _load(rnd.rel(f"plan.district.{name}.json")) or {}
        leaves = [p for q in (plan.get("quarters") or [])
                  for p in (q.get("plots") or []) if p.get("kind", "plot") == "plot"]
        if not leaves:
            out.append({"district": name, "built": "unmeasured",
                        "why": "this district has no compiled leaves to measure"})
            continue
        cols = placeplan.region_columns(d, place, None, leaves=leaves,
                                        parts_record=parts_rec)
        over = float(cols.get("developable_columns") or cols.get("scope_columns") or 0)
        built = cols.get("built_columns")
        out.append({
            "district": name, "lots": len(leaves),
            "estimated_built_cover": (cert.get("built_cover_estimate")
                                      if cert.get("on") == name else None),
            "estimated_lots": cert.get("lots") if cert.get("on") == name else None,
            "built_columns": built, "built_from": cols.get("built_from"),
            "built_cover": (round(built / over, 4) if built is not None and over else None),
            "allocated_columns": cols.get("allocated_columns"),
            "developable_columns": int(over) if over else None})
    if not out:
        return None
    return {"on": cert.get("on"), "districts": out,
            "note": ("the estimate is `arrange.alternatives`' pad arithmetic on the "
                     "district the action was certified on; `built_cover` is what "
                     "construction emitted over the same ground, and `built_from` says "
                     "whether it was read off emitted geometry or off the plan's pads")}


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


def _apply_relevel(rnd, be, spec: dict, doc: dict) -> dict:
    """**The parent's level for a piece, revised** (the design resolution round).

        `{"type": "relevel", "levels": {district: level}}`. A ring strip's pieces are cut and
        levelled by the sector decision (`sectors.json`), compiled at their levels and adopted
        on what the compile realized; the level is then the laid level
        (`stages_plan._stage_district_ground`). A reading that finds two pieces meeting across
        a step their ground does not ask for -- the market piece eight blocks over the lane
        piece, with the ring street climbing and falling between them -- names the level it
        should stand at, and this is the one owner that can move it: the decision record is
        revised (the old level and why kept beside it), the pieces' plans are withdrawn, the
        plan is laid out again and the lanes re-routed. The rebuild re-cuts the ground from the
        baseline. Refused, with nothing changed, where a level is outside the ring's offered
        steps or the piece's compile at it is not admissible.
        
    """
    from .. import placeplan as pp_mod, local as _local, deps as _deps
    from . import stages_media
    levels = {str(k): int(v) for k, v in ((doc or {}).get("levels") or {}).items()}
    if not levels:
        return {"applied": False, "refused": "a relevel names {district: level}"}
    rec_p = rnd.rel("sectors.json")
    place_p = rnd.rel("plan.place.json")
    if not (os.path.exists(rec_p) and os.path.exists(place_p)):
        return {"applied": False, "refused": "no sector decision or place plan to revise"}
    sec = json.load(open(rec_p))
    place = json.load(open(place_p))
    rings = {str(r.get("name")): r for r in ((place.get("layout") or {}).get("rings") or [])}
    by_name = {d.get("name"): d for d in place.get("districts") or []}
    snap = stages_media._snapshot(rnd)
    changed = []
    for name, lvl in levels.items():
        d = by_name.get(name)
        strip = next((s for s in sec.get("adopted") or []
                      if any(x.get("name") == name for x in s.get("districts") or [])), None)
        if d is None or strip is None:
            return {"applied": False, "refused": f"{name}: not a negotiated piece"}
        ring_l = (rings.get(str(d.get("defines"))) or {}).get("level")
        step = pp_mod.TERRACE_STEP
        offered = sorted({int(ring_l) + k * (step // 2)
                          for k in range(-2 * pp_mod.DISTRICT_TERRACE_STEPS,
                                         2 * pp_mod.DISTRICT_TERRACE_STEPS + 1)}) \
            if ring_l is not None else []
        if lvl not in offered:
            return {"applied": False,
                    "refused": f"{name}: {lvl} is not one of the ring's levels {offered}"}
        was = int((d.get("sector") or {}).get("level") or d.get("level") or 0)
        if was == lvl:
            return {"applied": False, "refused": f"{name} already stands at {lvl}"}
        for x in strip["districts"]:
            if x.get("name") == name:
                x.setdefault("sector", {})["level"] = int(lvl)
                x["level"] = int(lvl)
        d.setdefault("sector", {})["level"] = int(lvl)
        d["level"] = int(lvl)
        strip.setdefault("revised_levels", []).append(
            {"district": name, "was": was, "now": int(lvl),
             "caused_by": list((doc or {}).get("caused_by") or []),
             "why": str((doc or {}).get("why") or "")[:400],
             "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
        changed.append({"district": name, "was": was, "now": int(lvl)})
        for f in (rnd.rel(f"plan.district.{name}.json"),
                  rnd.rel(f"district_{name}_compiled.json")):
            if os.path.exists(f):
                os.remove(f)
    json.dump(sec, open(rec_p, "w"), indent=1)
    json.dump(place, open(place_p, "w"), indent=1)
    for f in ("plan.json", "plots.json") + (
            ("network.json", "circulation.json") if _local.scope_of(rnd) is None else ()):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    _local.retire_plans(rnd)
    with contextlib.suppress(ValueError):
        _deps.stamp(rnd, "plan", outputs=["plan.place.json"],
                    note=f"kept across a parent relevel of "
                         f"{', '.join(c['district'] for c in changed)}")
    got = stages_media._replan(rnd, be)
    p = (got.get("plan") or {}) if isinstance(got, dict) else {}
    ok = (not p.get("stop") and p.get("status") not in ("error", "needs_model")
          and os.path.exists(rnd.rel("plan.json")))
    comp_bad = []
    for c in changed:
        rec_c = _load(rnd.rel(f"district_{c['district']}_compiled.json")) or {}
        comp = rec_c.get("composition") or {}
        if comp and not comp.get("admissible"):
            comp_bad.append(f"{c['district']}: {comp.get('failed')}")
    if not ok or comp_bad:
        stages_media._restore(rnd, snap)
        return {"applied": False, "rolled_back": True,
                "refused": ("the relevel was not applied and the plan stands as it was: "
                            + ("; ".join(comp_bad) if comp_bad else
                               f"the place laid out again did not come back planned "
                               f"({p.get('status')}: {p.get('error')})"))}
    circ = _pipeline.stage_circulation(rnd, be, {})
    return {"applied": True, "action": {"action": "relevel", "levels": levels,
                                        "changed": changed,
                                        "circulation": (circ or {}).get("status")}}


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
    # ground "cut for a different candidate". **...and the lanes are laid again on the
    # ground that was just re-cut.** The neighbourhood round, found by running the
    # loop's own rebuild. `_apply_layout` routes the roads and lays the lanes while it
    # re-plans, and this list then re-cuts the ground and re-lays the terraces *under*
    # them -- so construction met a world whose cobblestone was laid for the levels the
    # previous candidate had. The cost, on the first rebuild after an applied
    # arrangement action: **57 row houses refused**, each with the same sentence -- "the
    # cell this door opens onto is cobblestone -- something is standing in the way of
    # walking through it" -- and 89 `E002` doors unreachable on foot behind them. The
    # cold order is `ground, terraces, circulation, parts` (`round.PLACE_DRY`); a
    # rebuild that re-cuts ground has to be the same order or it is a different pipeline
    # that looks alike. This is the round's own rule about a road: "if geometry or
    # access changes, update affected roads, entrances, compounds, districts and ground
    # before qualifying the candidate".
    for name in ("ground", "terraces", "circulation", "parts", "finish", "lint",
                 "material", "section", "inspect"):
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
    """The reading of the built world, **with its recorded corrections applied**.

        The neighbourhood delivery round. A reading is an agent's judgement of a world, and
        a later trial can establish that one of its attributions was wrong -- the spatial
        design round spent three actions establishing that the crowded ring's missing court
        was its approved pool and not its arrangement. That correction was then made with an
        editor, inside a directory this project calls kept byte for byte, and the independent
        reader was right to name it. `promote.correct` records a correction; this applies it
        on the way in and leaves `inspection/views.json` exactly as the reader wrote it.
        
    """
    from . import promote
    rec = _load(rnd.rel("inspection", "views.json")) or {}
    reading = rec.get("reading") or {}
    found = list(reading.get("findings") or [])
    rows = [r for r in promote.corrections(rnd, of="inspection/views.json")
            if not r.get("candidate") or r.get("candidate") == rec.get("candidate")]
    if rows:
        found = promote.apply_corrections(rows, found)
    return rec, found, list(reading.get("closed") or [])


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


def _act_key(hint) -> str | None:
    """A fingerprint of **what an action carries**, or None where a name is all of it.

        The block design round. `obligation.tried_here` refuses the unchanged retry -- the
        same action on the same candidate -- and for an action that is a name and nothing
        else the name is the whole of it. A `character` or `voice` revision is a name and a
        document, and two different documents are two different actions. See
        `obligation.tried_here` for the measurement that made this necessary.
        
    """
    if not isinstance(hint, dict):
        return None
    body = {k: hint.get(k) for k in ("characters", "voice", "levels")
            if hint.get(k) is not None}
    if not body:
        return None
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _routed(rnd, spec: dict, row: dict, finding: dict | None) -> str | None:
    """**Which action this finding is for, asked of the module that owns the actions.**

        The neighbourhood round, found by running the loop on its own built reading. A
        finding that names no action fell through to `owner_actions(owner)[0]` -- the head of
        a *declaration* list, which for the layout owner is `shrink_anchor` -- so a reading
        that said "the crowded ring is six terraces on an empty floor, and the ground between
        them belongs to nobody" was answered by making the market square smaller. The list is
        an inventory of what the owner *can* do; it is not, and was never meant to be, an
        answer to what this finding *asks* for.

        `placesolve._action_for` is that answer and has been since the design round: it reads
        the finding's own measure, words and subjects and returns the action and the subject,
        walking `ARRANGEMENT_ACTIONS` one per finding so a refusal leaves the rest available.
        Two lists about one question, disagreeing, is the defect this whole round is about;
        the dispatcher asks the module that acts.

        Returns `None` where the router has no answer, and the caller falls back to the
        owner's inventory exactly as it did.
        
    """
    from .. import placesolve
    place = _load(rnd.rel("plan.place.json"))
    if not place or not spec:
        return None
    f = dict(finding or {})
    for k in ("id", "says", "subjects", "measure", "target", "about", "owner"):
        if f.get(k) is None and row.get(k) is not None:
            f[k] = row[k]
    with contextlib.suppress(Exception):
        act, _subject = placesolve._action_for(f, place, spec)
        if act:
            return str(act)
    return None


def _judge_trial(rnd, rec: dict, here: str) -> dict | None:
    """**The open trial, judged on the world it built.** The spatial design round.

        Called on the re-entry pass, where the rebuilt world has been read and the cycle has
        been offered to `obligation.close`. Three questions in this order, because the first
        two are about what the place already had and the third is about what the trial was
        for:

          1. did anything the accepted candidate demonstrated stop being demonstrated, or did
             a form it stood disappear? (`promote.regressions`)
          2. was the improvement **observed** -- the row closed on a measurement of its own
             measure, or the cited measure moved in the direction the row asked for?
          3. neither: the trial cost a build and bought nothing measurable.

        Only (2) promotes. (1) and (3) put the accepted candidate back and keep the reason.
        
    """
    from . import promote
    rows = promote.load(rnd)
    t = promote._open_trial(rows)
    if t is None or not t.get("candidate"):
        return None
    if t.get("candidate") != here:
        return None
    cyc = next((c for c in reversed(rec.get("cycles") or [])
                if c.get("rebuilt_candidate") == here and c.get("applied")), None)
    want = promote.protected(rnd)
    regs = promote.regressions(rnd, want)
    # **What this trial cost, beside what it bought.** The neighbourhood delivery round:
    # "account for composition tradeoffs -- including qualities already failing". The
    # regressions above reject; this is the whole account, both directions, and it is on
    # the promotion whether or not anything fell.
    trade = promote.tradeoffs(rnd, want)
    measure = (cyc or {}).get("measure") or {}
    observed = bool((cyc or {}).get("closed")) or measure.get("moved") is True
    ev = {"cycle": (cyc or {}).get("cycle"), "finding": t.get("finding"),
          "action": t.get("action"), "measure": measure,
          "closed": bool((cyc or {}).get("closed")),
          "protected": want, "regressions": regs, "tradeoffs": trade}
    if regs:
        got = promote.reject(rnd, evidence=ev, why=(
            f"the trial built, and it lost what the accepted candidate had: "
            + "; ".join(str(r.get("why")) for r in regs[:3])
            + ". A local gain does not authorize a regression elsewhere, so the best "
              "retained result is put back and this trial is kept as a rejected one"))
        print(f"   improve: trial {t['n']} ({t.get('action')}) REJECTED -- "
              f"{len(regs)} regression(s); the accepted candidate "
              f"{got.get('restored_to')} is restored", flush=True)
        return got
    if observed:
        got = promote.promote(rnd, candidate=here, evidence=ev, why=(
            f"the rebuild completed, every protected requirement of the accepted "
            f"candidate still holds, and the improvement was observed on the built "
            f"world: {measure.get('why') or 'the row closed on its own measure'}"
            + (f". Composition tradeoffs: {trade.get('why')}"
               if trade.get("measured") else "")))
        print(f"   improve: trial {t['n']} ({t.get('action')}) PROMOTED -- {here} is the "
              f"accepted candidate", flush=True)
        return got
    got = promote.reject(rnd, evidence=ev, why=(
        f"the trial built and preserved what the place had, and the improvement it was "
        f"applied for was not observed: "
        f"{measure.get('why') or 'the row cites no measure that was read on both worlds'}"
        f". A trial is promoted on an observed improvement, not on a completed build"))
    print(f"   improve: trial {t['n']} ({t.get('action')}) rejected -- no observed "
          f"improvement; the accepted candidate {got.get('restored_to')} is restored",
          flush=True)
    return got


def stage_improve(rnd, be, results: dict) -> dict:
    from .. import contracts, deps, obligation
    from . import promote
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
        # a relevel promoted at "7 -> 6" read back as "7 -> 2, moved away" two trials
        # later
        if c.get("reinspected") and c.get("reinspected_candidate") not in (None, here):
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

    # --- the open trial, judged on the world it built --------------------------------
    # **Before anything is accepted and before any new action is chosen.** A trial that
    # regressed a protected requirement, or that bought no observable improvement, puts
    # the best retained result back; the stage is then re-entered against the restored
    # candidate rather than continuing to reason about a world that is no longer on
    # disk.
    verdict = _judge_trial(rnd, rec, here)
    _save(rnd, rec)
    if verdict is not None and verdict.get("state") == "rejected":
        obligation.save(rnd.state, led)
        return {"status": "reenter", "trial": verdict,
                "why": (f"trial {verdict.get('n')} ({verdict.get('action')}) was rejected "
                        f"and the accepted candidate {verdict.get('restored_to')} is "
                        f"restored; the loop continues on it")}
    # **The best retained result exists from the first read candidate onward.** Without
    # this the first trial would have nothing to fall back to, which is the state the
    # neighbourhood round was in when its first rebuild stopped at lint.
    if not promote.accepted(rnd):
        promote.accept(rnd, here, "the first candidate of this lineage to be built, "
                                  "checked and read; the best retained result until a "
                                  "trial earns promotion",
                       evidence={"measures": measures,
                                 "open_material": len(obligation.open_rows(
                                     led, material=True))})

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
    # **A revision whose trial was rejected on a ruler since corrected may be tried
    # again, once** (the fabric reset round): see the design check below. Its action is
    # not the unchanged retry `tried_here` refuses, because what judged it has changed.
    _rej = set()
    for t_ in (promote.load(rnd).get("trials") or []):
        regs_ = ((t_.get("evidence") or {}).get("regressions") or [])
        if t_.get("state") == "rejected" and regs_ and all(
                r_.get("what") == "quantity" and r_.get("id") in promote.SUBJECT_LABELS
                for r_ in regs_):
            _rej.add(str(t_.get("candidate")))
    # **...and one whose rebuild did not realize the action** (the design resolution
    # round): the first relevel's rebuild kept the ground cut for the old level -- the
    # ground stage keyed nothing a relevel moves -- so the trial built shops on ground
    # it had not cut, measured no step and was rejected for it. That is the pipeline's
    # defect, not the revision's verdict; a trial the round records as unrealized, with
    # the correction that fixed it, may be tried once more
    # (`flags.improve.unrealized_trials`).
    _unreal: dict = {}
    for u_ in ((rnd.flags.get("improve") or {}).get("unrealized_trials") or []):
        if isinstance(u_, dict) and u_.get("candidate") and u_.get("fixed_by"):
            _rej.add(str(u_["candidate"]))
            # each recorded defect-and-fix buys one more build of the same design
            _unreal[str(u_["candidate"])] = _unreal.get(str(u_["candidate"]), 0) + 1
    _done = {str(c.get("rebuilt_candidate")) for c in rec["cycles"]
             if c.get("rejudged") and c.get("rebuilt")
             and sum(1 for c2 in rec["cycles"] if c2.get("rejudged") and c2.get("rebuilt")
                     and c2.get("rebuilt_candidate") == c.get("rebuilt_candidate"))
             >= max(1, _unreal.get(str(c.get("rebuilt_candidate")), 0))}
    rejudge_keys = {_act_key(c.get("action_record")) for c in rec["cycles"]
                    if str(c.get("rebuilt_candidate")) in _rej - _done
                    and _act_key(c.get("action_record"))}
    if owed and can:
        for r in owed:
            owner = str(r.get("owner") or "")
            hint = _hint_of(findings, r["id"])
            act_name = _act_name({"action": hint, "acceptance": r.get("acceptance")}) \
                or _act_name(r)
            # **an ineffective action leaves the alternatives open.** Only the same
            # action on the same candidate is the unchanged retry this refuses.
            act_key = _act_key(hint)
            left = obligation.admissible(led, r["id"], here, owner_actions(owner))
            # **What this finding asks for, before what the owner happens to list
            # first.** See `_routed`: the owner's inventory is a declaration of
            # capability and its head is not an answer. The router's choice is put in
            # front of the remaining list, not instead of it, so an action that is
            # refused still leaves the rest.
            want = _routed(rnd, spec, r, next((x for x in findings
                                               if x.get("id") == r["id"]), None))
            if want and want in left:
                left = [want] + [a for a in left if a != want]
            # **...and the rest of the queue is the same kind of decision as the
            # first.** The spatial design round. `_routed` above fixed which action a
            # finding's *first* attempt is, and left the fallback exactly as it was: the
            # owner's declaration list, in declaration order. So the second and third
            # attempts on a finding walk an inventory that has nothing to do with it.
            # two whole rebuilds, sixteen minutes, spent making the market square a
            # different size in answer to a question about courtyards, and each one duly
            # rejected for changing nothing. An inventory is a statement of what an
            # owner *can* do; it was never an answer to what this finding asks, which is
            # the argument `_routed` makes one function above. Where the router's answer
            # is an arrangement decision, the alternatives offered are the other
            # arrangement decisions; where it is not, the list is unchanged. An owner
            # with nothing else of the right kind is exhausted for this row, which is a
            # state the ledger already records.
            if want:
                from .. import arrange as _arrange
                fam = set(_arrange.ARRANGEMENT_ACTIONS)
                if want in fam:
                    left = [a for a in left if a in fam]
            if act_name is None or (obligation.tried_here(led, r["id"], here, act_name,
                                                          key=act_key)
                                    and act_key not in rejudge_keys):
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
            # **The accepted candidate is on disk before the first action touches the
            # plan.** It has to be a directory and not an in-memory snapshot: the pass
            # that judges this trial is a *later driver invocation* -- the rebuilt world
            # is read by an agent between the two -- so nothing held in this process
            # survives to roll it back. That is why the neighbourhood round's rollback
            # could only ever cover replanning.
            promote.begin(rnd, candidate=here, finding=r["id"],
                          action=str(queue[0] or act_name), measures=before,
                          why=f"acting on {r['id']}: {str(r.get('says'))[:120]}")
            for act_name in queue:
                # the key is the reader's document, and only an action that carries one
                # has one: every arrangement action in the queue is a bare name
                _k = act_key if act_name == _act_name({"action": hint}) else None
                if obligation.tried_here(led, r["id"], here, act_name, key=_k) \
                        and not (_k and _k in rejudge_keys):
                    continue
                tried_any = True
                # **What an owner is offered, that owner can execute.** The spatial
                # design round, and it is `owner_actions`' own rule -- *"a list that
                # cannot be executed is not an inventory"* -- broken one function later.
                # `OWNER_ACTIONS["fabric"]` lists the arrangement actions, every one of
                # them is in `placesolve.REALLOCATE_ACTIONS`, so
                # `owner_actions("fabric")` duly offers them; and this line then routed
                # a `fabric` owner to `_apply_layout` for `enlarge_lots` **only** and
                # answered everything else with *"owner `fabric` has no bounded action
                # for `terrace` in this build"*. `terrace` refused, `compact_bay`
                # refused, two attempts spent on a row whose owner had been told it
                # could do both. An arrangement **is** a fabric decision -- it is a
                # statement about one district's blocks and lots -- and
                # `placesolve.reallocate` is what executes one, whichever owner the
                # reading gave it to.
                from .. import arrange as _arr
                if act_name == "relevel":
                    # the reading's document, or -- where the row came from another
                    # reading than the one on disk -- the ledger row's own
                    _doc = hint if isinstance(hint, dict) else (
                        r.get("action") if isinstance(r.get("action"), dict) else
                        next((a.get("record") for a in reversed(r.get("actions") or [])
                              if isinstance(a.get("record"), dict)
                              and a["record"].get("levels")), None) or {})
                    got = _apply_relevel(rnd, be, spec, {
                        **(_doc if isinstance(_doc, dict) else {}),
                        "caused_by": [r["id"]],
                        "why": f"the improve stage acting on {r['id']}: {r.get('says')}"})
                elif owner in ("layout", "scale", "build") or (
                        owner == "fabric"
                        and act_name in ("enlarge_lots", *_arr.ARRANGEMENT_ACTIONS)):
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
                               record=got.get("action"), key=_k)
                if got.get("applied"):
                    break
                print(f"   improve: {r['id']} ({owner}) -> {act_name}: refused -- "
                      f"{str(got.get('refused'))[:200]}", flush=True)
            if not tried_any:
                promote.abandon(rnd, why=f"every action of `{r['id']}` had already been "
                                         f"tried on this candidate; nothing was applied")
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
            promote.abandon(rnd, why=f"no action of `{r['id']}` applied: "
                                     f"{str(got.get('refused'))[:160]}")
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

    # --- before the build: can this proposal have helped? -----------------------------
    # **A proposal that reproduces a design already built is refused before the build
    # budget is spent.** The neighbourhood delivery round, and the audit's fifth cause,
    # measured on the spatial design round's own `trials.json`: candidate
    # `8b0a5f3932466d30` is recorded for **four** applied trials. Each of those spent a
    # full production cycle -- reallocate, the whole plan again, ground, terraces,
    # circulation, the build, the check and the inspection, about eight minutes -- to
    # rebuild a world byte-identical in its design to the one already on disk, and each
    # was then rejected for changing nothing. Naming an owner and an action does not
    # establish that the action can affect the subject, and the cheapest place to find
    # out is here: the plan has been laid out again and the candidate identity is known,
    # and nothing has been built yet. The check is deliberately about the **design**,
    # not about the measures: two candidates with the same id are the same plan, so no
    # build could tell them apart. A trial refused here is a trial that ran -- its
    # action stays recorded as tried, so the loop does not re-offer it -- and the
    # accepted candidate comes back.
    replanned = deps.candidate_id(rnd)
    chosen["replanned_candidate"] = replanned
    tried_designs = {here: "the candidate this action was applied to"}
    # **A design rejected by a ruler since corrected is not a design already judged**
    # (the fabric reset round). The promotion guard compared held-court and feature
    # *totals*; it now compares subjects (`promote._subject_regressions`). A trial whose
    # only regressions were those totals was rejected on a question the guard no longer
    # asks, so its design may be built and judged again -- recorded on the cycle.
    rejudge = set()
    for t_ in (promote.load(rnd).get("trials") or []):
        regs_ = ((t_.get("evidence") or {}).get("regressions") or [])
        if t_.get("state") == "rejected" and regs_ and all(
                r_.get("what") == "quantity" and r_.get("id") in promote.SUBJECT_LABELS
                for r_ in regs_):
            rejudge.add(str(t_.get("candidate")))
    unrealized = {str(u_["candidate"]): u_ for u_ in
                  ((rnd.flags.get("improve") or {}).get("unrealized_trials") or [])
                  if isinstance(u_, dict) and u_.get("candidate") and u_.get("fixed_by")}
    rejudge |= set(unrealized)
    if replanned in unrealized:
        chosen["rejudged"] = (f"{replanned} was built by a rebuild that did not realize "
                              f"its action ({unrealized[replanned].get('why')}); fixed by "
                              f"{unrealized[replanned].get('fixed_by')}; it is built and "
                              f"judged again, once")
    elif replanned in rejudge:
        chosen["rejudged"] = (f"{replanned} was rejected only on the held court and "
                              f"feature totals, which the guard now compares subject by "
                              f"subject; it is built and judged again")
    for c in rec["cycles"]:
        if c is chosen or not c.get("applied"):
            continue
        if replanned in rejudge and replanned in (str(c.get("rebuilt_candidate")),
                                                  str(c.get("replanned_candidate"))):
            continue
        for k in ("rebuilt_candidate", "replanned_candidate"):
            if c.get(k):
                tried_designs.setdefault(
                    str(c[k]), f"the design cycle {c.get('cycle')} "
                               f"({c.get('action')}) produced")
    if replanned in tried_designs:
        rej = promote.reject(rnd, evidence={
            "cycle": chosen.get("cycle"), "replanned_candidate": replanned,
            "same_as": tried_designs[replanned],
            "saved": "the build, the check and the inspection of a design already on "
                     "record"},
            why=(f"the improve stage's {chosen['action']} laid the place out again and "
                 f"the design that came back is {replanned}, which is "
                 f"{tried_designs[replanned]}: this proposal cannot affect the finding "
                 f"it was applied for, and it is refused before it is built"))
        chosen["trial"] = rej
        chosen["ineffective"] = True
        _save(rnd, rec)
        print(f"   improve: {chosen['action']} reproduced {replanned} "
              f"({tried_designs[replanned]}); trial {(rej or {}).get('n')} rejected "
              f"without building", flush=True)
        obligation.save(rnd.state, led)
        return {"status": "reenter", "trial": rej,
                "why": (f"the proposal reproduced a design already on record "
                        f"({replanned}); it is kept as a rejected trial with its reason "
                        f"and nothing was built"),
                "cycle": chosen}

    # --- rebuild and read again -----------------------------------------------------
    for f in ("world_built.npz",):
        pass                                     # the parts stage rebuilds a moved candidate
    got = _rebuild(rnd, be, results)
    chosen["rebuilt"] = "parts" in got.get("stages", {})
    chosen["rebuild_stopped"] = got.get("stopped")
    chosen["rebuilt_candidate"] = deps.candidate_id(rnd)
    # **The estimate this action was chosen on, against what the build emitted.** The
    # neighbourhood round: `arrange.alternatives` ranks on the compiler's pad
    # arithmetic, which is an estimate of mass and is not construction, and the
    # controller adopts that ranking -- so the ledger has to carry both numbers on the
    # same district or the estimate is never held to anything. Written after the
    # rebuild, per district the action refabricated, and `unmeasured` where construction
    # reported nothing.
    chosen["estimate_vs_built"] = _estimate_vs_built(rnd, chosen)
    _save(rnd, rec)
    if got.get("stopped"):
        # **A failed trial is a rejected trial, not a failed round.** The neighbourhood
        # round returned `blocked` here, twice, and left the failed trial on disk as the
        # candidate -- so its second failed revision is what it delivered. The trial's
        # reason is kept, the trial's world is not, and the accepted candidate comes
        # back; the stage is re-entered on it with the action recorded as tried.
        rej = promote.reject(rnd, evidence={
            "cycle": chosen.get("cycle"), "stopped": got.get("stopped"),
            "stages": sorted((got.get("stages") or {}).keys())},
            why=(f"after the improve stage's {chosen['action']} the rebuild stopped at "
                 f"{got['stopped']}: {str(got.get('why'))[:400]}"))
        chosen["trial"] = rej
        _save(rnd, rec)
        print(f"   improve: the rebuild stopped at {got['stopped']}; trial "
              f"{(rej or {}).get('n')} rejected and the accepted candidate "
              f"{(rej or {}).get('restored_to')} restored", flush=True)
        return {"status": "reenter", "trial": rej,
                "why": (f"the trial's rebuild stopped at {got['stopped']}; it is kept as a "
                        f"rejected trial with its reason and the best retained result is "
                        f"back on disk"),
                "cycle": chosen}
    promote.built(rnd, candidate=chosen["rebuilt_candidate"],
                  stages=got.get("stages") or {})
    if got.get("pending"):
        return {**got["res"], "cycle": chosen,
                "note": (f"the rebuilt world is drawn and awaits its reading; the improve "
                         f"stage judges trial {chosen['cycle']} against it and promotes "
                         f"it only on an observed improvement")}
    return {"status": "reenter", "why": "the affected output was rebuilt and read; the "
                                         "trial is judged against the new reading",
            "cycle": chosen}
