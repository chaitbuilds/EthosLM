"""Findings go to the layer that can fix them, and what is fixed is checked again.

The loop before this one could change two things: a district's character and the place's
voice, once, and then it drew the plan again and marked itself done **without looking**.
So a finding about scale, site, layout or a missing capability had exactly one available
response -- different words -- and a repair that did land was never re-inspected.

What is here is small and deterministic on purpose:

  `route`    every finding grouped by the layer that owns it, in severity order.
  `apply`    the repairs this build can actually make, each inside a bound that is
             written down, each recorded with what it changed and why.
  `refused`  the findings whose owner cannot act -- and, more importantly, the ones a
             repair **may not** act on: an explicit requirement is never negotiated.

The rule the whole module turns on: **an inferred choice is revisable and an explicit
requirement is not.** A village's size band is the library's own guess at what "village"
means and a shore that holds eleven houses is a fact about the ground; reconciling those
two is repair. "Exactly two thousand houses" is not a guess, and a build that cannot
reach it is a build that reports a shortfall.

A repair does not mark itself done. `stage_preview` re-lays the plan from what was
changed and runs the checks again, and a repair whose recheck fails is rolled back by the
snapshot that stage already keeps.
"""
from __future__ import annotations

import contextlib

import json
import os

from . import contracts, spec as spec_mod

#: How many repairs one run may apply before it stops and reports what is left. Two: the
#: point is to close a diagnosis the run produced, not to search.
REPAIR_BUDGET = 2

#: **Registered before the first repair ran.** The least a negotiated structure target
#: may be, as a share of the band floor the kind inferred. Below this the place the
#: ground can hold is not the place that was asked for, and the run says so instead of
#: quietly building a hamlet where a village was wanted.
SCALE_NEGOTIATION_MIN = 0.25

#: How far from the middle of the parts that were to gather round something the layout
#: owner will look for clear ground to put it on, and the step it walks outward by. A
#: bound and not a target: a square that has to be moved a hundred and twenty columns to
#: find room is not the square that place is gathered around, and the action says so
#: rather than putting it anywhere it fits.
RELATION_SEARCH_MAX = 120
RELATION_SEARCH_STEP = 8

#: **Every attempted action is recorded against the candidate it was tried on**, and an
#: action already tried on that candidate is not tried again (the closure round). The
#: shore village's record: twelve passes, seven with no applied action, eight identical
#: refusals of the same move on the same design. A repair that changed nothing is
#: `changed: false` here, and the next pass escalates to a different decision instead of
#: repeating it. `REPAIR_ATTEMPTS_TOTAL` bounds the whole run's attempts whether or not
#: they were charged as applied changes.
ATTEMPTS_RECORD = "repair_attempts.json"
REPAIR_ATTEMPTS_TOTAL = 12


def _attempts(rnd) -> dict:
    p = rnd.rel(ATTEMPTS_RECORD) if rnd is not None else None
    if p and os.path.exists(p):
        return json.load(open(p))
    return {"attempts": []}


def _tried(rec: dict, candidate: str, finding: str, action: str) -> dict | None:
    for a in rec.get("attempts") or []:
        if a.get("candidate") == candidate and a.get("finding") == finding \
                and a.get("action") == action:
            return a
    return None


def _record_attempt(rnd, rec: dict, candidate: str, finding: str, action: str,
                    changed: bool, why: str) -> None:
    import time as _time
    rec.setdefault("attempts", []).append({
        "candidate": candidate, "finding": finding, "action": action,
        "changed": bool(changed), "why": str(why)[:300],
        "at": _time.strftime("%Y-%m-%dT%H:%M:%S")})
    if rnd is not None:
        os.makedirs(rnd.state, exist_ok=True)
        json.dump(rec, open(rnd.rel(ATTEMPTS_RECORD), "w"), indent=1)


def _candidate(rnd) -> str:
    try:
        from . import deps
        return deps.candidate_id(rnd)
    except Exception:                          # noqa: BLE001 -- no id is one id
        return "none"


#: The owners a repair in this build can actually act on. Everything else is routed,
#: reported and left -- which is a better answer than a change that looks like one.
#: **`layout` is the integration round's addition and it is the important one.** Before
#: it, the only repair this system could make was "want fewer houses": a finding about
#: geometry -- homes fronting away from the water they were asked to face, a residual
#: strip of ground promised more lots than it can hold -- had nowhere to go, so the run
#: either stopped or asked for different adjectives. Reducing an inferred count is not a
#: spatial repair, and a loop whose only lever is the target is a loop that closes every
#: finding the same way.
ACTS_ON = ("scale", "layout")

#: How many districts one frontage repair may reorient. A bound and not a target: the
#: action re-states an obligation the design already carries and re-lays from it, and a
#: place whose every district is wrong is a layout failure and not a repair.
FRONTAGE_DISTRICTS_MAX = 64


def route(findings: dict) -> dict:
    """`{owner: [findings]}`, errors before warnings, in the order they were found."""
    out: dict = {}
    for f in (findings or {}).get("findings") or []:
        out.setdefault(f["owner"], []).append(f)
    for k in out:
        out[k].sort(key=lambda f: (f["severity"] != "error", f["id"]))
    return out


def _band_repair(spec: dict, f: dict) -> dict | None:
    """Reconcile an inferred size band with the capacity the ground actually gave.

        Returns the change to make, or None where it may not be made. Refuses outright
        where the sentence named a count: `find/capacity/explicit` is a different finding
        and this is not its repair.
        
    """
    if spec.get("explicit_count"):
        return {"refused": True,
                "why": ("the sentence named this count. An explicit requirement is not "
                        "negotiated: the shortfall is reported, the limiting constraint "
                        "named, and nothing here moves the bar")}
    promised = int((f.get("evidence") or {}).get("promised") or 0)
    band = (f.get("evidence") or {}).get("band") or spec.get("size_band") or [0, 0]
    lo, hi = int(band[0]), int(band[1])
    if not lo or promised >= lo:
        return None
    # **The bound belongs to the interpretation that was accepted, not to the last one
    # this function produced.** The review walked a village's floor 24 -> 6 -> 2 with
    # both calls inside "the registered 25% minimum", because each call measured 25% of
    # the band the call before it had already moved. A limit re-applied to its own
    # output is not a limit; it is a decay rate. So the floor every negotiation is
    # measured against is the original one, recorded the first time the spec was
    # negotiated.
    first = accepted_band(spec)
    origin = int(first[0]) if first else lo
    floor = int(round(origin * SCALE_NEGOTIATION_MIN))
    if promised < max(1, floor):
        return {"refused": True,
                "why": (f"the resolved design holds {promised} structures against the "
                        f"floor of {origin} this build first inferred a "
                        f"{spec.get('kind')} to be"
                        + (f" (the band now reads {lo}-{hi} after "
                           f"{len(spec.get('negotiated') or [])} negotiation(s))"
                           if origin != lo else "")
                        + f"; the least a negotiated target may be is {floor} "
                          f"({SCALE_NEGOTIATION_MIN:.0%} of that original floor). This "
                          f"site cannot hold the place that was asked for, and moving "
                          f"the target further would be moving the question")}
    return {"refused": False, "structures": promised,
            "size_band": [promised, max(promised, hi)],
            "accepted_floor": origin,
            "why": (f"the size band {lo}-{hi} is what this build infers a "
                    f"{spec.get('kind')} to be; the ground the layout could claim holds "
                    f"{promised}. The band is an inferred choice and the count is "
                    f"revised to the measured capacity, inside {SCALE_NEGOTIATION_MIN:.0%} "
                    f"of the originally accepted floor of {origin}")}


def accepted_band(spec: dict) -> list | None:
    """The size band of the **originally accepted** interpretation, or None.

        The first `negotiated` entry records what the band was before anything moved it;
        with no negotiation on record the current band is the original one. This is the one
        reader of that history, so "what was this place agreed to be" has one answer.
        
    """
    for n in (spec or {}).get("negotiated") or []:
        if n.get("what") != "size_band":
            continue
        was = (n.get("from") or {}).get("size_band")
        if was:
            return [int(was[0]), int(was[1])]
    band = (spec or {}).get("size_band")
    return [int(band[0]), int(band[1])] if band else None


def _relation_repair(rnd, spec: dict, f: dict, place: dict | None) -> dict | None:
    """Move the thing a place was asked to be gathered around, to where it is gathered.

        **The spatial repair the relation requirement needs.** "small houses around a big
        temple", "gathered around a market square": the sentence says a place is arranged
        *about* something, `intent._relation_measure` measures it as bearings, and until this
        a design that put the square outside its houses had no owner who could move it. The
        decision is the layout's -- where a `centre` part stands is the place level's -- so
        this is the layout's action.

        The object is moved to the middle of the subjects that were supposed to surround it,
        keeping its own rectangle's size, staying inside the site, and touching no other
        region. Where no such position exists the action refuses by name: a village strung
        along a shore may genuinely have nowhere to put a square its houses can stand round,
        and that is a finding about the request and the ground rather than a repair.
        
    """
    from .pipeline import stages_plan
    from . import intent as intent_mod, pipeline as _pipeline
    if place is None or rnd is None:
        return None
    site = _round_site(rnd)
    if not site:
        return {"refused": True,
                "why": "this round records no site, so where a part may be moved to is "
                       "not something it can measure"}
    w = (f.get("evidence") or {})
    obj_name = str(w.get("object") or "")
    plan = rnd.plan()
    parts = _pipeline.plan_parts(plan) if plan else []
    # **Deferred until the evidence exists.** A relation between the houses and the
    # square is measured on the houses, and before the districts compile there are none:
    # acting on a measurement of two place-level parts moved the square to the middle of
    # a hall. Not an attempt, and not charged as one.
    if not parts or not int(w.get("subjects") or 0):
        return {"refused": True, "deferred": True,
                "why": ("this relation is measured on the assembled plan's leaves and "
                        "none exist yet; the action waits for the districts to compile")}
    # **The subjects first.** "gathered around a market square" is a fact about where
    # the houses stand, and the layout owner's only move used to be the square: on a
    # place whose districts stood on two sides of it, moving the square could never make
    # them reach round it, and the shore village spent twelve passes finding that out.
    # The relation solver now lays the districts of an `around` relation on three sides
    # of its object (`placesolve`), so the action is to lay the place level out again
    # with the sentence's relations as constraints -- and it is refused, by name, where
    # that produces the geometry already on the record.
    if str(w.get("relation") or "") == "around" and (place.get("districts") or []):
        again = _resector(rnd, spec, place)
        if again is not None:
            return again
    # the object at the place level -- the part whose rectangle this action moves
    target = next((p for p in (place.get("parts") or [])
                   if p.get("name") == obj_name or p.get("defines") == obj_name), None)
    if target is None or target.get("x1") is None:
        return {"refused": True,
                "why": (f"the place level holds no movable part named `{obj_name}`; the "
                        f"thing this place was asked to be gathered around is not one "
                        f"this action owns")}
    subs = [p for p in parts
            if intent_mod._word_matches(p, str(w.get("subject") or "houses"))
            and p.get("x1") is not None]
    if not subs:
        return {"refused": True,
                "why": (f"no leaf of the plan answers `{w.get('subject')}`, so there is "
                        f"no middle to move `{obj_name}` to")}
    cx = sum((float(p["x0"]) + float(p["x1"])) / 2 for p in subs) / len(subs)
    cz = sum((float(p["z0"]) + float(p["z1"])) / 2 for p in subs) / len(subs)
    x0, z0, x1, z1 = (int(target["x0"]), int(target["z0"]),
                      int(target["x1"]), int(target["z1"]))
    w_, d_ = x1 - x0, z1 - z0
    ox, oz, n = int(site["origin"][0]), int(site["origin"][1]), int(site["size"])
    # **The nearest free ground to that middle, and free is the validator's own word.**
    # A part may not overlap a district, a compound or another part -- the place level
    # refuses it and is right to -- so this does not drop the square on top of the
    # houses; it walks outward from their middle and takes the first position that
    # stands clear of everything. A settlement gathered round a square has the square in
    # the gap its own fabric leaves, which on a shore band is between the districts.
    others = []
    for key in ("districts", "compounds"):
        for r in place.get(key) or []:
            if r.get("x1") is not None:
                others.append([min(int(r["x0"]), int(r["x1"])),
                               min(int(r["z0"]), int(r["z1"])),
                               max(int(r["x0"]), int(r["x1"])),
                               max(int(r["z0"]), int(r["z1"]))])
    for p in (place.get("parts") or []):
        if p is target or p.get("x1") is None:
            continue
        with contextlib.suppress(Exception):
            others.append([int(v) for v in _pipeline.part_rect(p)])

    def clear(r):
        if not (ox <= r[0] and r[2] <= ox + n - 1 and oz <= r[1] and r[3] <= oz + n - 1):
            return False
        return not any(r[0] <= o[2] and o[0] <= r[2] and r[1] <= o[3] and o[1] <= r[3]
                       for o in others)

    rect = None
    for step in range(0, RELATION_SEARCH_MAX, RELATION_SEARCH_STEP):
        ring = [(step, 0), (-step, 0), (0, step), (0, -step),
                (step, step), (-step, -step), (step, -step), (-step, step)] \
            if step else [(0, 0)]
        for dx, dz in ring:
            ax = max(ox, min(ox + n - 1 - w_, int(cx - w_ / 2) + dx))
            az = max(oz, min(oz + n - 1 - d_, int(cz - d_ / 2) + dz))
            cand = [ax, az, ax + w_, az + d_]
            if clear(cand):
                rect = cand
                break
        if rect is not None:
            break
    if rect is None:
        return {"refused": True,
                "why": (f"no ground within {RELATION_SEARCH_MAX} columns of the middle "
                        f"of the {len(subs)} part(s) that were to gather round "
                        f"`{obj_name}` is clear of every other region and inside the "
                        f"site; where this place puts its fabric leaves nowhere for the "
                        f"thing it was asked to be gathered around")}
    if rect == [x0, z0, x1, z1]:
        return {"refused": True,
                "why": (f"`{obj_name}` already stands at the nearest clear ground to "
                        f"the middle of the {len(subs)} part(s) that were asked to be "
                        f"gathered round it, and they still do not reach round it: this "
                        f"is the shape of the place and not the position of the square")}
    target["x0"], target["z0"], target["x1"], target["z1"] = rect
    target["notes"] = (target.get("notes") or "") + (
        f" The layout owner moved this part to the middle of the {len(subs)} part(s) "
        f"the request says are gathered around it.")
    # the districts are laid out about the place level's parts, so they are compiled
    # again from the moved one
    stages_plan  # noqa: B018 -- imported for the module's own invalidation contract
    return {"refused": False,
            "districts": [{"district": obj_name, "from": [x0, z0, x1, z1],
                           "to": rect}],
            "place": place,
            "why": (f"`{obj_name}` stood at ({x0},{z0})-({x1},{z1}) and the "
                    f"{len(subs)} part(s) the sentence says are gathered around it "
                    f"reached {w.get('quadrants')} of its 4 sides; it is moved to "
                    f"({rect[0]},{rect[1]})-({rect[2]},{rect[3]}), the middle of those "
                    f"parts, and the place is laid out and measured again. What moves "
                    f"is where the thing stands, not what the sentence asked for")}


#: The judge's composition measures the layout owner has an action for.
COMPOSITION_MEASURES = ("undeveloped_share", "clusters", "open_to_built")
#: ...and the measures and actions that are an **allocation** (the expression round).
SIZING_MEASURES = ("square_scale",)


def _composition_measure(f: dict) -> str | None:
    ev = f.get("evidence") or {}
    m = ev.get("measure") or (ev.get("finding") or {}).get("measure")
    return str(m) if m in COMPOSITION_MEASURES else None


def _finding_of(f: dict) -> dict:
    """The judge's own finding row inside a routed finding, or the routed one."""
    ev = f.get("evidence") or {}
    inner = ev.get("finding") if isinstance(ev.get("finding"), dict) else None
    return inner or {**f, **{k: ev.get(k) for k in ("measure", "subjects", "target",
                                                     "action", "about", "measured")
                             if ev.get(k) is not None}}


def _sizing_action(f: dict) -> str | None:
    """Is this reading's finding one the allocation answers: a sizing measure, a
    contextual target that asks for more land, or a named allocation action?"""
    from . import placesolve
    row = _finding_of(f)
    act = row.get("action")
    if isinstance(act, str) and act in placesolve.REALLOCATE_ACTIONS \
            and act != "move_object":
        return act
    m = str(row.get("measure") or "")
    if m in SIZING_MEASURES:
        return "shrink_anchor"
    if m == "open_to_built" and str((row.get("target") or {}).get("direction") or "") == "up":
        return "grow_land"
    return None


def _reallocate_repair(rnd, spec: dict, f: dict, place: dict | None) -> dict | None:
    """`placesolve.reallocate` as a repair: the record in the shape `apply` writes."""
    from . import contracts as contracts_mod, placesolve, placeplan
    from .pipeline import stages_plan
    if place is None or rnd is None:
        return None
    site = _round_site(rnd)
    if not site:
        return {"refused": True, "why": "this round records no site, so the place "
                                        "cannot be laid out again"}
    row = dict(_finding_of(f))
    row.setdefault("id", f.get("id"))
    _t, decls = placeplan.types_card(rnd.flags.get("types"), spec.get("form"))
    constraints = []
    pr = rnd.rel("parts.json")
    if os.path.exists(pr):
        with contextlib.suppress(Exception):
            rec = json.load(open(pr))
            constraints = [r.get("constraint") for w in rec.get("waves") or []
                           for r in w.get("parts") or [] if r.get("constraint")]
    got, rec = placesolve.reallocate(
        place, spec, row, site=site, decls=decls, constraints=constraints,
        intent=contracts_mod.load(rnd, "intent"),
        plateau=stages_plan.plateau_record(rnd), seed=int(rnd.flags.get("seed") or 1),
        caps=contracts_mod.load(rnd, "capabilities"))
    if not rec.get("applied"):
        return {"refused": True, "why": rec.get("why") or "refused", "record": rec}
    # the re-solved place replaces the one in hand, field by field, so the caller's
    # reference is the new design
    place.clear()
    place.update(got)
    return {"refused": False, "districts": rec.get("districts") or [], "place": place,
            "why": rec.get("why"), "allocation": rec.get("allocation"),
            "spec_changed": True}


def _resector(rnd, spec: dict, place: dict) -> dict | None:
    """Lay the place level out again with the intent's relations, or None where the
    solver cannot be asked (no site, no intent, another policy). Returns a repair
    result: refused `changed: false` where the geometry came back the same."""
    from . import contracts, placeplan, placesolve
    from .pipeline import stages_plan
    intent = contracts.load(rnd, "intent")
    site = _round_site(rnd)
    if intent is None or site is None:
        return None
    if placesolve.policy_for(spec) != "relations":
        return None
    vol = None
    with contextlib.suppress(Exception):
        vol = rnd.volume()
    plateau = None
    with contextlib.suppress(Exception):
        plateau = stages_plan.plateau_record(rnd)
    _t, decls = placeplan.types_card(rnd.flags.get("types"), spec.get("form"))
    caps = contracts.load(rnd, "capabilities")
    got, fails = placesolve.solve_place(spec, site, plateau, decls,
                                        place.get("voice") or "", vol=vol,
                                        seed=int(rnd.flags.get("seed") or 1),
                                        caps=caps, intent=intent)
    if got is None:
        return {"refused": True, "changed": False,
                "why": ("laying the place out again with the sentence's relations "
                        "fails its own validator: "
                        + "; ".join(f"{q.get('part')}: {q['why']}" for q in fails[:3]))}
    def geom(doc):
        return sorted((str(d.get("name")), int(d["x0"]), int(d["z0"]), int(d["x1"]),
                       int(d["z1"])) for d in (doc.get("districts") or [])) + \
            sorted((str(p.get("name")), *[int(v) for v in _pipeline_rect(p)])
                   for p in (doc.get("parts") or []))
    if geom(got) == geom(place):
        return {"refused": True, "changed": False,
                "why": ("the relation solver lays this place's districts exactly as they "
                        "stand; re-sectoring changes nothing, so the shape of this site "
                        "and not the sectoring is what keeps the houses off a side")}
    predicted = [r for r in ((got.get("layout") or {}).get("districts") or {})
                 .get("relations") or []]
    changed = []
    for d in got.get("districts") or []:
        was = next((q for q in place.get("districts") or [] if q.get("name") == d["name"]),
                   None)
        changed.append({"district": d["name"],
                        "from": [was["x0"], was["z0"], was["x1"], was["z1"]] if was else None,
                        "to": [d["x0"], d["z0"], d["x1"], d["z1"]]})
    # the whole place level is replaced: districts, parts and the layout record, with
    # the proposals the solver made (the arrangement pass compiles from them)
    place["districts"] = got["districts"]
    place["parts"] = got["parts"]
    place["compounds"] = got.get("compounds") or []
    place["layout"] = got["layout"]
    place["intent"] = got.get("intent") or place.get("intent")
    return {"refused": False, "changed": True, "districts": changed, "place": place,
            "predicted": predicted,
            "why": (f"the place level is laid out again with the sentence's relations "
                    f"as constraints: {len(got['districts'])} district(s) now stand on "
                    f"{len({s['label'].split('_')[0] for s in ((got.get('layout') or {}).get('districts') or {}).get('sectors') or []})} "
                    f"side(s) of the centre, and the relation solver predicts "
                    + ", ".join(f"{r.get('id')}: {r.get('predicted')}" for r in predicted)
                    + ". What moves is where the districts stand, not what the sentence "
                      "asked for")}


def _pipeline_rect(p: dict) -> tuple:
    from . import pipeline as _pipeline
    try:
        return tuple(int(v) for v in _pipeline.part_rect({**p, "name": p.get("name")}))
    except Exception:                          # noqa: BLE001 -- no rectangle
        return (0, 0, 0, 0)


def _frontage_repair(rnd, spec: dict, f: dict, place: dict | None) -> dict | None:
    """Reorient the districts whose homes front away from what the sentence asked.

        **The spatial repair.** The finding is `facing/water` failing on actual geometry --
        N homes with their doors in the wall away from the water beside them -- and the
        decision that produced it is a *layout* decision: a two-row block puts one row on
        each side of its street, and nothing had told the compiler that this place's rows
        may only front one way. So the action writes that obligation onto each district
        (`faces`, a side of the world, computed locally from the shore path beside it) and
        the plan is laid out again from it. `district_compile` then lays one row per block
        on the side that fronts the water, and the ground behind it becomes the court it
        would otherwise have been.

        It costs houses -- a district that may face only one way holds fewer than one that
        may face both -- and that is a true consequence of the request and not a defect.
        Where it would cost more than the place can afford, the capacity finding that
        follows is the scale owner's, and the two are separate findings on purpose.

        Returns the change to make, or None where this build may not make it.
        
    """
    from . import placeshore
    if place is None:
        return None
    lay = (place.get("layout") or {})
    anchor = dict(lay.get("anchor") or {})
    if lay.get("anchor_path"):
        anchor["path"] = lay["anchor_path"]
    if not anchor.get("path"):
        return {"refused": True,
                "why": ("this plan records no shore path, so which way the water lies "
                        "from each district is not a question it can answer; the "
                        "frontage cannot be re-stated without inventing a direction")}
    districts = list(place.get("districts") or [])
    if not districts or len(districts) > FRONTAGE_DISTRICTS_MAX:
        return {"refused": True,
                "why": (f"{len(districts)} district(s) is outside the bound of "
                        f"{FRONTAGE_DISTRICTS_MAX} this action may reorient")}
    changed = []
    for d in districts:
        want = placeshore.water_direction(anchor, (d["x0"] + d["x1"]) // 2,
                                          (d["z0"] + d["z1"]) // 2)
        near = placeshore._path_near(anchor, (d["x0"], d["z0"], d["x1"], d["z1"]))
        if want and (d.get("faces") != want
                     or not (d.get("faces_anchor") or {}).get("path")):
            changed.append({"district": d["name"], "from": d.get("faces"), "to": want,
                            "anchor_points": len(near.get("path") or [])})
            d["faces"] = want
            # **and the shore beside it**, so the compiler can ask the question again
            # per block. A district is a rectangle and a shore is not: one direction per
            # district put two of nine homes the wrong way round on a curving bank.
            d["faces_anchor"] = near
    if not changed:
        return {"refused": True,
                "why": ("every district already carries the frontage the sentence asks "
                        "for; the homes facing away are not a frontage decision this "
                        "action can reach")}
    return {"refused": False, "districts": changed, "place": place,
            "why": (f"{len(changed)} district(s) are told which way their lots front, "
                    f"from the shore path beside each of them, and the plan is laid out "
                    f"again from that. The obligation is the sentence's and the "
                    f"direction is the ground's; neither is invented here")}


def _extent_repair(rnd, spec: dict, f: dict, place: dict | None) -> dict | None:
    """Claim the free ground beside a district that is short of what was asked for.

        **The spatial answer, taken before the promise is moved.** `resolve.unclaimed_ground`
        measures it and this makes it: the district's rectangle grows into ground that is
        inside the site and touches no other region. The programme is untouched -- nothing
        here changes what the place must hold, only how much ground it has to hold it in.
        
    """
    from .pipeline import stages_plan
    if place is None or rnd is None:
        return None
    site = _round_site(rnd)
    if not site:
        return {"refused": True,
                "why": "this round records no site, so the ground available beside a "
                       "district is not something it can measure"}
    name = f.get("part")
    d = next((q for q in place.get("districts") or [] if q.get("name") == name), None)
    if d is None:
        return None
    bigger = stages_plan._grow_into_free_ground(place, d, site)
    if bigger is not None:
        # a shoreline district grows inland and along the shore, never into the water
        from .placeshore import clamp_to_land
        was_rect = [int(d["x0"]), int(d["z0"]), int(d["x1"]), int(d["z1"])]
        clamped = clamp_to_land(place, bigger)
        if clamped is not None:
            bigger = [min(clamped[0], was_rect[0]), min(clamped[1], was_rect[1]),
                      max(clamped[2], was_rect[2]), max(clamped[3], was_rect[3])]
        if clamped is None or bigger == was_rect:
            bigger = None
    if bigger is None:
        return {"refused": True,
                "why": (f"{name} has no free ground beside it that this action may "
                        f"take: it is a ring sector, every side of it meets another "
                        f"region or the edge of the site, or what is beside it is water")}
    was = [int(d["x0"]), int(d["z0"]), int(d["x1"]), int(d["z1"])]
    # the rectangle this district was first laid out as, so the bound stays bound to it
    d.setdefault("extent_from", list(was))
    d["x0"], d["z0"], d["x1"], d["z1"] = [int(v) for v in bigger]
    # **And the houses the new ground holds.** Found by running it: growing a district
    # and leaving its allocation where it was made the cover clause *harder* -- the same
    # four lots over half again as much ground covered 23% against a floor of 38% -- so
    # the action that was supposed to answer a shortfall produced a district its own
    # validator refused. Claiming ground the place is short of means claiming what it
    # holds; the count is the compiler's own arithmetic on the new rectangle and not a
    # number chosen here.
    promised = int(d.get("proposed_structures") or d.get("structures") or 0)
    holds = _fits_in(spec, d, place)
    # **The ask rises by what the place is short of, and no more.** The proposal is a
    # layout decision and this action revises it on purpose -- recorded as such, which
    # is a different thing from an adopted count feeding back as the next proposal
    # (`arrange` compiles from the proposal; see its docstring). A district grown to
    # claim ground is asked for the houses the place lacks, not for everything the
    # bigger rectangle could hold: a sentence's sixteen stays sixteen.
    total = sum(int(q.get("proposed_structures") or q.get("structures") or 0)
                for q in (place.get("districts") or []))
    short = max(0, int((spec or {}).get("structures") or 0) - total)
    # **A place that is not short claims no ground.** Growing a district that lacks
    # nothing only spreads its houses over more verge -- the scattered village the judge
    # read -- and the finding it answers is about a shortfall that is not there.
    if not short:
        return {"refused": True,
                "why": (f"the place promises {total} against {spec.get('structures')} "
                        f"asked: it is not short, and growing {name} would only spread "
                        f"its {promised} house(s) over ground they do not need")}
    to = min(holds, promised + short)
    if to > promised:
        d["structures"] = to
        d["proposed_structures"] = to
        d["proposal_revised"] = {"by": "layout/extent", "from": promised, "to": to,
                                 "holds": holds, "place_short": short}
    else:
        to = promised
    d["notes"] = (d.get("notes") or "") + (
        f" The layout owner grew this district from {was} to {bigger} to claim free "
        f"ground the design had left beside it, and its ask with it "
        f"({promised} -> {to}, of the {holds} the ground now holds).")
    return {"refused": False,
            "districts": [{"district": name, "from": was, "to": list(bigger),
                           "structures": [promised, int(to)]}],
            "place": place,
            "why": (f"{name} was {was[2] - was[0] + 1}x{was[3] - was[1] + 1} columns and "
                    f"is now {bigger[2] - bigger[0] + 1}x{bigger[3] - bigger[1] + 1}, "
                    f"taking ground that is inside the site and touches no other "
                    f"region; its ask goes with it, {promised} -> "
                    f"{to}, which is the place's shortfall inside what the compiler "
                    f"lays in the new rectangle. The place is short of the size its kind is "
                    f"inferred to be and the ground was there; what changes is the "
                    f"ground and not the request")}


def _fits_in(spec: dict, d: dict, place: dict | None = None) -> int:
    """How many structures **the construction logic lays** in this district's rectangle.

        **The estimate is not a capacity, and using one here is what made the extent repair
        unsound.** The review: "Extent repair reuses this estimate, so it can enlarge a
        region and promise more of the same unrealizable housing." That is exactly what this
        did -- it grew a district and raised its allocation to `placeplan.fabric_fit`'s
        prediction for the new rectangle, and the compiler then laid whatever the ground
        really held, which is the disagreement the whole round exists to remove.

        So the question is put to `arrange`, which compiles the grown rectangle and returns
        the houses that stand in it. A repair that claims ground now claims what can be
        built on it.
        
    """
    from . import arrange as arrange_mod, placeplan
    part = next((p for p in (spec or {}).get("defining_parts") or []
                 if p.get("name") == d.get("defines")), {})
    try:
        _t, decls = placeplan.types_card(None, (spec or {}).get("form"),
                                         part.get("role"))
        return int(arrange_mod.capacity(d, part, place or {"parts": [],
                                                           "arterials": {"cells": []}},
                                        decls, spec=spec))
    except Exception:                          # noqa: BLE001 -- no number is no change
        return 0


def _round_site(rnd) -> dict | None:
    from .pipeline import settlement_site
    try:
        return settlement_site(rnd)
    except Exception:                          # noqa: BLE001 -- absent is an answer
        return None


def _allocation_repair(spec: dict, f: dict, place: dict | None) -> dict | None:
    """Reconcile a district's allocation with what its ground actually held.

        **The other half of the promise/realization disagreement.** `resolve.with_realized`
        writes a finding when a district lays fewer lots than the allocator budgeted, and
        it blocks feasibility -- correctly, because a design whose own record disagrees with
        itself is not feasible. Until this, the plan level had no action for it: the finding
        was routed to `layout`, `layout` had only a frontage action, and the run reported
        the plan planned with the disagreement open. The review found exactly that in the
        city, twice.

        The allocator made the promise from an area estimate and the compiler measured the
        ground; the measurement wins, and the *place's* obligation does not move -- what
        falls out is a capacity finding for the scale owner, who is bound by the originally
        accepted floor and refuses below it. Two owners, two bounds, one direction.
        
    """
    if place is None:
        return None
    name = f.get("part")
    got = int((f.get("evidence") or {}).get("realized") or 0)
    was = int((f.get("evidence") or {}).get("promised") or 0)
    d = next((q for q in place.get("districts") or [] if q.get("name") == name), None)
    if d is None or got >= was:
        return None
    total = sum(int(q.get("structures") or 0) for q in place.get("districts") or [])
    band = accepted_band(spec)
    floor = int(round((band[0] if band else 0) * SCALE_NEGOTIATION_MIN))
    after = total - was + got
    if floor and after < floor:
        return {"refused": True,
                "why": (f"reconciling {name} from {was} to {got} would leave the place "
                        f"allocating {after} structures against the {floor} that is "
                        f"{SCALE_NEGOTIATION_MIN:.0%} of the {band[0]} floor this build "
                        f"first accepted for a {spec.get('kind')}; the districts cannot "
                        f"hold what was promised and the place cannot afford to promise "
                        f"less, which is a layout failure and not an allocation")}
    d["structures"] = got
    d["notes"] = (d.get("notes") or "") + (
        f" The layout owner reconciled this district's allocation from {was} to the "
        f"{got} its own streets, lots and setbacks laid.")
    return {"refused": False, "districts": [{"district": name, "from": was, "to": got}],
            "place": place,
            "why": (f"{name} was allocated {was} structure(s) from an area estimate and "
                    f"its ground laid {got}; the allocation is the layout owner's and "
                    f"is moved to the measurement. The place's own obligation is "
                    f"unchanged and the shortfall is the scale owner's finding")}


def apply(rnd, spec: dict, findings: dict, *, budget: int = REPAIR_BUDGET,
          place: dict | None = None, place_path: str | None = None) -> dict:
    """Make the repairs this build can make. Returns the record; writes the spec.

        The spec on disk is rewritten -- `place.checked.json`, the one every stage after the
        spec stage reads -- because a change of scale is a change to the **programme**, and
        a repair that left the programme alone and patched the plan would be a repair
        nothing downstream could see.
        
    """
    by = route(findings)
    rec = {"budget": int(budget), "applied": [], "refused": [], "routed": {},
           "bounds": {"SCALE_NEGOTIATION_MIN": SCALE_NEGOTIATION_MIN,
                      "REPAIR_BUDGET": REPAIR_BUDGET}}
    for owner, rows in sorted(by.items()):
        rec["routed"][owner] = [f["id"] for f in rows]
    # **The layout owner acts first, and that order is the rule.** A place short of what
    # it was asked for has two kinds of answer: change the ground, the geometry or the
    # types it is built of, or change what was asked for. The second is always available
    # and is almost always the worse one, and running the scale owner first meant it was
    # also always the one that happened -- the review's "recovery changes promises more
    # readily than it resolves design conflicts", as a line of control flow. A promise
    # is moved only after the spatial actions have been tried. --- the layout owner: a
    # spatial decision changed, and the plan laid out again ---
    place_changed = False
    allocation_changed = False
    attempts = _attempts(rnd)
    here = _candidate(rnd)
    rec["candidate"] = here
    for f in by.get("layout", []):
        if len(rec["applied"]) >= budget:
            rec["refused"].append({"finding": f["id"], "owner": "layout",
                                   "why": f"the repair budget of {budget} is spent"})
            continue
        # `requirement` is present and null on every finding that answers no clause of
        # the sentence, so `.get(k, "")` returns None and not "" -- found by running the
        # held-out village, whose realized-capacity finding crashed this line.
        rid = str(f.get("requirement") or "")
        if rid.startswith("relation/") or f["id"].startswith("find/relation/"):
            action, fn = "relation", lambda: _relation_repair(rnd, spec, f, place)
        elif rid.startswith("facing/") or "facing" in f["id"]:
            action, fn = "frontage", lambda: _frontage_repair(rnd, spec, f, place)
        elif f["id"].startswith("find/capacity/realized/"):
            action, fn = "allocation", lambda: _allocation_repair(spec, f, place)
        elif f["id"].startswith("find/extent/unclaimed/"):
            action, fn = "extent", lambda: _extent_repair(rnd, spec, f, place)
        elif f["id"].startswith("find/reading/") and _sizing_action(f):
            # **A finding about a dimension is the layout owner's allocation** (the
            # expression round): the square's scale, thin working land, a lot the
            # storeys need, a ring's width -- `placesolve.reallocate` revises the
            # inferred allocation, records it on the spec, and lays the place out again
            action, fn = "reallocate", lambda: _reallocate_repair(rnd, spec, f, place)
        elif f["id"].startswith("find/reading/") and _composition_measure(f):
            # **A judge's composition finding is a layout action** (the closure round):
            # "scattered, not gathered" -- `undeveloped_share`, `clusters`,
            # `open_to_built` -- is answered by laying the place level out again with
            # the sentence's relations and the compact sizing they imply, and refused by
            # name where that reproduces the geometry on the record.
            action, fn = "resector", lambda: (_resector(rnd, spec, place)
                                               or {"refused": True,
                                                   "why": "the relation solver cannot "
                                                          "be asked for this place"})
        else:
            action, fn = None, None
        if fn is None:
            rec["refused"].append({
                "finding": f["id"], "owner": "layout",
                "why": ("this build has no layout action for this finding; it is "
                        "reported and the requirement stays open")})
            continue
        # **Once per candidate, and a bounded number of times per run.** An action
        # already tried on this design -- whether it changed nothing or was refused --
        # is not tried again on it; the finding escalates to the next owner instead.
        was = _tried(attempts, here, f["id"], action)
        if was is not None:
            rec["refused"].append({
                "finding": f["id"], "owner": "layout", "changed": False,
                "why": (f"already tried `{action}` on candidate {here}: "
                        f"{'it changed nothing' if not was.get('changed') else 'it was applied'}"
                        f" ({was.get('why', '')[:120]}); it is not repeated and the "
                        f"finding escalates")})
            continue
        if len(attempts.get("attempts") or []) >= REPAIR_ATTEMPTS_TOTAL:
            rec["refused"].append({
                "finding": f["id"], "owner": "layout",
                "why": (f"{len(attempts['attempts'])} actions have been attempted this "
                        f"run, which is the registered bound of {REPAIR_ATTEMPTS_TOTAL}; "
                        f"the layout is wrong at a level these actions cannot reach")})
            continue
        got = fn()
        if got is None:
            rec["refused"].append({
                "finding": f["id"], "owner": "layout",
                "why": ("this build has no layout action for this finding; it is "
                        "reported and the requirement stays open")})
            continue
        if got.get("refused"):
            if not got.get("deferred"):
                _record_attempt(rnd, attempts, here, f["id"], action, False, got["why"])
            rec["refused"].append({"finding": f["id"], "owner": "layout",
                                   "changed": False, "why": got["why"]})
            continue
        _record_attempt(rnd, attempts, here, f["id"], action, True, got["why"])
        rec["applied"].append({"finding": f["id"], "owner": "layout",
                               "from": {c.get("district", "?"): c.get("from")
                                        for c in got["districts"]},
                               "to": {c.get("district", "?"): c.get("to")
                                      for c in got["districts"]},
                               "districts": got["districts"], "why": got["why"],
                               **({"allocation": got["allocation"]}
                                  if got.get("allocation") else {})})
        place_changed = True
        if got.get("spec_changed"):
            allocation_changed = True
    spec_changed = False
    for f in by.get("scale", []):
        if len(rec["applied"]) >= budget:
            rec["refused"].append({"finding": f["id"],
                                   "why": f"the repair budget of {budget} is spent"})
            continue
        # **A promise is not moved in the same pass that moved the ground.** The layout
        # owner has just changed a district's rectangle, and the capacity this finding
        # was measured against is the capacity *before* that change: negotiating the
        # band down to it now is negotiating against a number that is already wrong.
        # Found by running the loop fixture, where the two owners acting together walked
        # the band 81, 58, 53, 52 over four passes -- every step a true measurement of a
        # design the previous step had already superseded. The plan is laid out and
        # measured again, and the scale owner answers the shortfall that survives that.
        if place_changed:
            rec["refused"].append({
                "finding": f["id"], "owner": "scale",
                "why": ("the layout owner changed the ground in this pass; what the "
                        "place holds is measured again before any promise moves")})
            continue
        if f["id"] == "find/capacity/explicit":
            rec["refused"].append({
                "finding": f["id"], "owner": "scale",
                "why": ("the sentence named this count. An explicit requirement is not "
                        "negotiated: the shortfall is reported and the limiting "
                        "constraint named, and nothing here moves the bar")})
            continue
        got = _band_repair(spec, f)
        if got is None:
            rec["refused"].append({"finding": f["id"], "owner": "scale",
                                   "why": "nothing in this finding is an inferred "
                                          "choice this build may revise"})
            continue
        if got["refused"]:
            rec["refused"].append({"finding": f["id"], "owner": "scale",
                                   "why": got["why"]})
            continue
        was = {"structures": int(spec["structures"]),
               "size_band": list(spec["size_band"])}
        spec["structures"] = int(got["structures"])
        spec["size_band"] = [int(v) for v in got["size_band"]]
        spec_mod.apportion(spec["defining_parts"], 0, spec["structures"])
        spec.setdefault("negotiated", []).append(
            {"what": "size_band", "from": was,
             "to": {"structures": spec["structures"],
                    "size_band": list(spec["size_band"])},
             "bound": {"SCALE_NEGOTIATION_MIN": SCALE_NEGOTIATION_MIN},
             "why": got["why"], "finding": f["id"]})
        rec["applied"].append({"finding": f["id"], "owner": "scale",
                               "from": was,
                               "to": {"structures": spec["structures"],
                                      "size_band": list(spec["size_band"])},
                               "why": got["why"]})
        spec_changed = True
    if place_changed and place_path:
        with open(place_path, "w") as fh:
            json.dump(place, fh, indent=1)
        rec.setdefault("wrote_place", os.path.basename(place_path))
    if allocation_changed and rnd is not None:
        # **An allocation row is a change to the programme's inferred choices and is
        # written where the scale owner writes its own** (the expression round): the
        # checked spec, so the plan laid out again from it carries the allocation
        p = rnd.rel("place.checked.json")
        doc = json.load(open(p)) if os.path.exists(p) else dict(spec)
        doc["negotiated"] = list(spec.get("negotiated") or [])
        with contextlib.suppress(spec_mod.SpecError):
            doc = spec_mod.read_spec(doc, doc.get("sentence"))
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as fh:
            json.dump(doc, fh, indent=1)
        rec["wrote"] = os.path.basename(p)
        rec["allocation"] = [r for r in spec.get("negotiated") or []
                             if r.get("what") == "allocation"]

    for owner, rows in sorted(by.items()):
        if owner in ACTS_ON:
            continue
        for f in rows:
            rec["refused"].append({
                "finding": f["id"], "owner": owner,
                "why": (f"routed to `{owner}`, which this build has no automatic repair "
                        f"for; it is reported and the requirement stays open")})
    if spec_changed and rnd is not None:
        p = rnd.rel("place.checked.json")
        # **A negotiation that cannot be written down did not happen.** Found by running
        # the loop fixture, which reaches this stage without a checked spec on disk: the
        # band moved in memory, the file the next entry of the stage reads was left
        # alone, the same finding came back, and the driver re-entered until its cap
        # stopped it. A repair whose only record is a local variable is a repair the run
        # repeats.
        doc = json.load(open(p)) if os.path.exists(p) else dict(spec)
        doc["structures"] = spec["structures"]
        doc["size_band"] = list(spec["size_band"])
        doc["negotiated"] = spec["negotiated"]
        for dp in doc.get("defining_parts") or []:
            for sp in spec["defining_parts"]:
                if sp["name"] == dp.get("name"):
                    dp["structures"] = sp["structures"]
        # **Written the way the stage that reads it will read it.** A negotiated spec
        # that has not been through `read_spec` is not the document the next invocation
        # of `stage_place_spec` produces from it: that stage re-derives the band, the
        # apportionment and the ceiling, so the fingerprint moved *once*, on the run
        # after the repair -- and the plan the repair had just stamped was called stale,
        # the place was solved again from the reduced target, and it came out a
        # different and worse design than the one whose capacity had been measured.
        # Found by running the shore village: a completed plan and its inspection were
        # thrown away by a replay that had changed nothing.
        with contextlib.suppress(spec_mod.SpecError):
            doc = spec_mod.read_spec(doc, doc.get("sentence"))
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as fh:
            json.dump(doc, fh, indent=1)
        rec["wrote"] = os.path.basename(p)
        # **The ground the negotiation was measured on is the ground it keeps.** A scale
        # repair moves the spec, the spec is what the site search is keyed on, and the
        # search therefore called itself stale and chose a *different* site -- on which
        # the capacity that had just been measured no longer meant anything, and from
        # which the whole run started over. Found by running the shore village, where it
        # threw away a completed plan and its inspection. The same argument
        # `stages_plan._plan_repair` makes for keeping the place level across a repair:
        # what the scale owner changed is the programme's inferred target, and the
        # geometry that measured it is evidence. Evidence is not re-derived. So the
        # search is stamped again, by name, and the site stands.
        from . import deps as _deps_r
        for artifact, outputs in (("site_search", ["site_search.json"]),
                                  ("plateau", ["plateau.json"])):
            if not _deps_r.recorded(rnd, artifact):
                continue
            with contextlib.suppress(ValueError):
                _deps_r.stamp(rnd, artifact, outputs=outputs,
                              note=(f"kept across a scale negotiation: the ground was "
                                    f"chosen for this request and is what the capacity "
                                    f"the band was negotiated to was measured on"))
    rec["changed_plan"] = bool(spec_changed or place_changed)
    rec["changed_spec"] = bool(spec_changed)
    rec["changed_place"] = bool(place_changed)
    return rec


def unresolved(findings: dict) -> list:
    """The findings still open after a repair pass: what an incomplete result is."""
    return [f for f in (findings or {}).get("findings") or [] if not f.get("fixed")]


def says(rec: dict) -> str:
    return (f"{len(rec.get('applied') or [])} repair(s) applied, "
            f"{len(rec.get('refused') or [])} refused; routed to "
            + ", ".join(f"{k} ({len(v)})" for k, v in
                        sorted((rec.get("routed") or {}).items())))
