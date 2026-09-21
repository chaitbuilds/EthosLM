"""The library grows when the spec asks for a form it lacks. v2, C3.

The library grows by refusal: it says what it cannot yet build, and that is the next
type. Until this, the refusal stopped the run -- `stage_plan_levels` said "no committed
type of this place's form builds a tower" and a person wrote one. Now the plan stage
**detects the gap** before it plans (`type_gaps`: a defining part of one of the four
leaf kinds with no committed type of its kind named for its family, in the place's
form), and **stages a blinded authoring** for it through the machinery every type has
been authored by -- the same brief (`stages_build.type_brief`: the API,
the voice card, the contract for its kind, the ground it will be checked on, the
request), the same checker in the author's directory, the same `done` marker and the
same adoption (`blind._collect_blinded`) -- with **default fixtures for a plan-time
author** (`default_fixtures`: the six plots and the parts the rules choose, because a
plan-time round has registered none), and **capped per run** (`GROWTH_CAP`). An
answer is adopted into `types/` only when it **passes the gate**: every instance the
checker stands, in the author's voice and the place's, clean. One that does not is
recorded with its findings and the run stops by name; the model is not asked twice.

The request is composed from the defining part and the sentence, never from a place's
name. The record is `growth.json` in the round's state.
"""
from __future__ import annotations

import json
import os
import shutil

from . import pipeline as _pipeline, spec as spec_mod, styles

#: How many types one run may author before it stops and says so. A place that lacks
#: three forms is a place the library is not ready for; two is a run's worth.
GROWTH_CAP = 2

#: The kinds a defining part may be that a type builds; a `group` is a district or a
#: compound and is planned from committed types.
LEAF_KINDS = ("plot", "edge", "point", "area")


def type_gaps(spec: dict, names=None, intent: dict | None = None) -> list:
    """The defining parts no committed **capability** answers, in spec order, and --
        the expression round -- the **functions** the request asks for that no type of the
        place's form declares (`function_gaps`).

        Was: a type of the right kind whose *filename* began with the family, among the
        types of the place's form. The architecture audit's finding is that a filename is
        not a capability -- the same rule admitted a type of the wrong form at
        `stage()` below -- so the question is now asked of what the type declares about
        itself: its kind, its form, its role, its envelope and its attachment. See
        `ethoslm.capability.fits`.

        `names` is the round's own list of types where it has one; `intent` the checked
        intent record, where the caller has one (without it only defining-part gaps open).
        
    """
    from . import capability
    out = list(capability.gaps(spec, names=names))
    for p in function_gaps(spec, intent, names=names):
        if p not in out:
            out.append(p)
    return out


#: Which functions are open ground rather than a building, so the gap is an `area`.
AREA_FUNCTIONS = ("market", "bazaar", "fair")


def _declares_function(decl: dict, fn: str, form: str | None) -> bool:
    got = decl.get("function") or (decl.get("needs") or {}).get("function")
    if not got or str(got).lower() != str(fn).lower():
        return False
    return _pipeline.form_ok(decl.get("form"), form)


def function_gaps(spec: dict, intent: dict | None, names=None) -> list:
    """The functions the request asks for by name that no committed type of the place's
        form declares, each as a **pseudo defining part** the growth path can author.

        A `function` requirement (`function/market`, `function/smithing`) or a `feature`
        requirement whose feature word is a function (`feature/market`) opens a gap when
        every type declaring that function is of another form. The part carries the
        requirement it answers (`answers`) and the function (`function`), which the brief
        names as the feature contract and the gate checks on the answer: a type adopted for
        a function gap declares the function, or it is not adopted. Unsupported
        requirements open nothing -- they were refused by name upstream and stay refused.
        
    """
    from . import envelope
    if not intent:
        return []
    from .placeplan import types_card
    _t, decls = types_card(names, forms=[None])
    form = spec.get("form")
    out = []
    seen = set()
    for r in intent.get("requirements") or []:
        if r.get("kind") not in ("function", "feature"):
            continue
        if r.get("status") in ("unsupported", "satisfied"):
            continue
        # (coordinator, the expression round's farm run) a soft reading of what the
        # whole place is for -- `function/farming` of a "farming village" -- is not a
        # capability gap: only a hard function of a buildable thing opens one
        w = r.get("wants") or {}
        if not r.get("hard") or str(w.get("what") or "").lower() in (
                "village", "town", "city", "hamlet", "place", "settlement", "ring",
                "district", "quarter"):
            continue
        fn = str(w.get("function") or "").lower()
        if not fn and r.get("kind") == "feature":
            feat = str(w.get("feature") or "").lower()
            fn = feat if any(feat in words for words in
                             (envelope.FEATURE_WORDS.get("stalls"),
                              envelope.FEATURE_WORDS.get("forge"))) else ""
        if not fn or fn in seen:
            continue
        if any(_declares_function(d, fn, form) for d in decls.values()):
            continue
        seen.add(fn)
        kind = "area" if fn in AREA_FUNCTIONS else "plot"
        tokens = [t for t, words in envelope.FEATURE_WORDS.items() if fn in words]
        out.append({"name": fn, "kind": kind, "family": fn, "relation": "throughout",
                    "count": 1, "structures": 0, "role": "civic" if kind == "area" else None,
                    "forms": [form] if form else None, "function": fn,
                    "features": tokens, "answers": [r["id"]],
                    "notes": (f"the request asks for {fn} ({r.get('phrase') or r['id']}) "
                              f"and no committed type of this place's form declares "
                              f"FUNCTION = {fn!r}"),
                    "of": "function"})
    return out


def type_name(part: dict) -> str:
    """The name the authored type is committed under: the family's, which is what
    the planner looks a defining part up by."""
    return part["family"]


def request_for(spec: dict, part: dict) -> str:
    """What the author is asked to build, composed from the defining part and the
    sentence: the family, the kind, the role, the form, and the spec's own notes."""
    name = type_name(part)
    kind = part["kind"]
    form = (part.get("forms") or [None])[0] or spec.get("form") or "civic"
    role = part.get("role") or "civic"
    what = {"plot": "a building on a plot of prepared ground",
            "edge": "a line on the ground with a width, graded segment by segment",
            "point": "a thing at one cell, facing one way, on a pad the library prepares",
            "area": "a rectangle of ground brought to one level, with no door and no rooms"}
    return (f"Write `types/{name}.py`: `build(b, part, seed, **params)` builds a "
            f"**{part['family']}** -- {what[kind]} -- for a place the sentence below asks "
            f"for, and for any other place that asks for one. It is a `{kind}`: declare "
            f"`KIND = \"{kind}\"`, `FORM = \"{form}\"` (the tradition it is built in), "
            f"`ROLE = \"{role}\"` (what it is for), `PARAMS` (what varies between "
            f"instances -- storeys, a plan, a crown), and `NEEDS` (the ground it wants, "
            f"as the contract above says). It names no material and no roof silhouette: "
            f"every block is `b.voice[role]` and the roof is `b.roof(...)` in the "
            f"voice's own profile.\n\n"
            f"> {spec.get('sentence', '').strip()}\n\n"
            f"**What the place spec says of it:** {part.get('notes') or 'nothing more'}; "
            f"{part['count']} of them, placed `{part['relation']}`"
            + (f" of `{part['of']}`" if part.get("of") else "")
            + f".\n\nA person walks up to it from the lane and into it on foot where it "
            f"has an inside; every room is reachable without jumping; it stands on the "
            f"pad the library prepares and lays nothing below `part['floor_y']`. Write "
            f"the file, run `check.py`, read `findings.md`, and finish when every "
            f"instance is clean in both voices.\n"
            + _feature_contract(part))


def _feature_contract(part: dict) -> str:
    """What a type authored for a **function** gap has to declare and emit, so the gate
    can check it and construction can measure it. Empty for a family gap."""
    fn = part.get("function")
    if not fn:
        return ""
    tokens = list(part.get("features") or [])
    return (f"\n## The feature contract\n\n"
            f"This type answers the request's `{fn}` function (requirement(s) "
            f"{part.get('answers') or []}). Declare `FUNCTION = {fn!r}` beside `FORM` and "
            f"`ROLE`, and `FEATURES = {tuple(tokens)!r}` naming the features it can be "
            f"asked to deliver. `build()` returns `{{\"ok\": True, \"emitted\": {{...}}}}` "
            f"where `emitted` carries `requested` (the parameters as asked), `storeys` "
            f"(what stood), `attempt`, `fallback` (what was given up and why, or None), "
            f"`omitted` (the requested things that did not stand), `features` "
            f"({{name: bool or count}} for {tokens or ['the function\'s equipment']}) and "
            f"`rects` ({{name: [x0, z0, x1, z1]}} for every feature named, so "
            f"`construction.outcome` verifies each one on the built blocks -- a declared "
            f"feature with no rectangle is not evidence). The adoption gate refuses a "
            f"type that does not declare the function. `ethoslm.envelope.lot_for` will "
            f"probe the type for the lot each feature needs; keep the smallest lot that "
            f"delivers the function small.\n")


def default_fixtures(kind: str) -> list:
    """The ground a plan-time author's type is checked on: the six plots the rules
    choose for a plot type, the part of its kind the rules choose for an edge, a point
    or an area -- because a plan-time round has registered no fixtures of its own."""
    from .pipeline import blind
    if kind == "plot":
        return blind.check_fixtures()
    return [dict(f) for f in blind.check_parts() if f.get("kind") == kind]


def _tspec(rnd, spec: dict, part: dict, voice: str, fixtures: list) -> dict:
    name = type_name(part)
    return {"name": name, "part": part["kind"],
            "file": os.path.join("types", f"{name}.py"), "voice": voice,
            "kind": f"a {part['family']}", "request": request_for(spec, part),
            "fixtures": fixtures,
            **({"function": part["function"], "form": spec.get("form"),
                "answers": list(part.get("answers") or [])} if part.get("function") else {}),
            "check_sweep": bool(rnd.flags.get("growth_sweep", True))}


def _record_path(rnd) -> str:
    return rnd.rel("growth.json")


def record_of(rnd) -> dict:
    p = _record_path(rnd)
    if os.path.exists(p):
        return json.load(open(p))
    return {"cap": GROWTH_CAP, "gaps": [], "asked": {}, "adopted": {}, "failed": {},
            "capped": []}


def _save(rnd, rec: dict) -> None:
    os.makedirs(rnd.state, exist_ok=True)
    json.dump(rec, open(_record_path(rnd), "w"), indent=1)


def gate(rnd, be, tspec: dict, src_path: str, place_voice: str | None) -> dict:
    """Does this authored type pass? Its own checker over its fixtures and seeds, in
    the author's voice and the place's: every instance clean in both, and none
    crashed. What it read is on the record whichever way it went."""
    from .pipeline import blind
    mine = dict(tspec, file=src_path)
    fixtures = blind._fixtures_for(rnd, mine)
    seeds = blind._seeds_for(rnd, mine)
    voices = _pipeline.check_voices(tspec.get("voice"), place_voice)
    res = _pipeline.check_type(rnd, be, src_path, [], seeds, fixtures=fixtures,
                               voice=tspec.get("voice"), voices=voices,
                               sweep=bool(tspec.get("check_sweep", True)))
    by = res.get("voices") or {}
    passes = (not res.get("crashed") and bool(by)
              and all(v["instances"] > 0 and v["clean"] == v["instances"]
                      for v in by.values()))
    # **A function gap is closed by a type that declares the function**, in the place's
    # form: clean instances of a hall do not make it a market.
    fn = tspec.get("function")
    why_fn = ""
    used = None
    if fn:
        try:
            decl = _pipeline.load_type(src_path)
        except Exception as e:                   # noqa: BLE001 -- the gate reports
            decl, why_fn = {}, f"does not load: {e}"
        if decl and not _declares_function(decl, fn, tspec.get("form")):
            why_fn = (f"declares FUNCTION={decl.get('function')!r} and FORM="
                      f"{decl.get('form')!r}; this gap is `{fn}` in "
                      f"{tspec.get('form') or 'any form'}")
        # **...and by a type whose function can be *used*.** The expression review's
        # fourth finding: "its feature contract is stronger in the authoring brief than
        # in the adoption gate". The brief tells the author that every feature carries a
        # rectangle so `construction.outcome` can verify it; the gate then checked the
        # declaration and the clean instances and never once asked whether a person
        # could walk up to the thing the function needs. `usable` asks, on the world one
        # probe build assembles.
        if not why_fn:
            used = _usable_for(tspec, src_path)
            if used.get("refused"):
                why_fn = used["refused"]
        if why_fn:
            passes = False
    return {"passes": bool(passes), "instances": res.get("instances"),
            **({"function": fn, "function_refused": why_fn,
                **({"usable": used} if used else {})} if fn else {}),
            "errors": res.get("errors"), "entry_lines": res.get("entry_lines"),
            "crashed": bool(res.get("crashed")), "voices": by,
            "fixtures": [f"{f['round']}/{f.get('plot') or f.get('part')}"
                         for f in fixtures], "seeds": seeds,
            "sweep": bool(tspec.get("check_sweep", True)),
            "seconds": res.get("seconds"), "text": res.get("text", "")}


#: The lot a gate's usability probe builds the candidate on: the middle of the band its
#: own `NEEDS.footprint` declares, so the answer is about the type and not about a pad
#: it was never meant to stand on. Bounded to one build: the gate already pays for a
#: whole checker sweep and this is the one question that sweep cannot ask.
USABLE_PROBE = 0.5


def _usable_for(tspec: dict, src_path: str) -> dict:
    """Can a person get in and use what this authored type builds? One probe build.

        The three predicates a function gap is about -- a way in, equipment that stands and
        can be reached, a court that is open and reachable -- asked of the assembled probe
        world through `ethoslm.usable`. An `unsupported` answer refuses nothing: a type with
        no court has no court to be inaccessible, and this gate is not the place to invent
        a requirement the brief did not make. What it refuses is a predicate that **fails**.
        
    """
    from . import construction, usable
    name = tspec.get("name") or os.path.splitext(os.path.basename(src_path))[0]
    try:
        ns = {"__name__": "__ethoslm_type__", "__file__": src_path}
        exec(compile(open(src_path).read(), src_path, "exec"), ns)   # noqa: S102
        lo_w, lo_d, hi_w, hi_d = (_pipeline.read_needs(ns, where=src_path)
                                  .get("footprint") or (9, 9, 24, 24))
        w = int(lo_w + USABLE_PROBE * (hi_w - lo_w))
        d = int(lo_d + USABLE_PROBE * (hi_d - lo_d))
        # the file being gated is not committed to `types/` -- that is what this gate
        # decides -- so the probe is given its source by name
        b, sited, res = construction.probe_build(name, w, d, {}, seed=1,
                                                 voice=tspec.get("voice"),
                                                 source=src_path)
        got = construction.outcome(b, sited, None, {})
    except Exception as e:                       # noqa: BLE001 -- the gate reports
        return {"asked": [], "refused": None,
                "why": f"the usability probe did not run: {type(e).__name__}: {e}"}
    if not res.get("ok"):
        return {"asked": [], "refused": None,
                "why": f"the usability probe's instance did not stand: {res.get('reason')}"}
    part = {**sited, "name": name, "kind": "plot", "type": name, "emitted": got}
    world = usable.World.of_builder(b, part)
    # **The same three the production pass asks, named once.** The composition round:
    # this tuple was `construction.CONFIRM_WANTS` written out a second time, so a change
    # to what construction confirms left the adoption gate asking the old set.
    answers = {w2: usable.check(world, name, w2) for w2 in construction.CONFIRM_WANTS}
    failed = sorted(k for k, a in answers.items() if a["holds"] is False)
    return {"asked": sorted(answers), "lot": [w, d],
            "answers": {k: {"holds": a["holds"], "method": a["method"], "why": a["why"]}
                        for k, a in answers.items()},
            "refused": (f"the authored type builds a {w}x{d} instance that fails "
                        + "; ".join(answers[k]["why"] for k in failed)) if failed else None,
            "why": usable.says(answers)}




def _refresh_capabilities(rnd, spec: dict, types, site) -> None:
    """Re-match the capability record after the library grew.

    The record is written **before** growth opens its gaps (it is what the gaps are
    read from), so a type adopted here answered its want on disk and not on the record,
    and the plan went on refusing the part -- "nothing on disk is a plot for a worship"
    with `types/worship.py` freshly adopted. The orchard hamlet found it. The record is
    re-derived with the same facts the plan stage used, so the adoption is a decision
    the rest of the run can read."""
    try:
        from . import capability, contracts
        from .pipeline.stages_plan import site_capability_facts
        facts = site_capability_facts(rnd, spec, site)
        caps = capability.match(spec, names=types, ground=facts["ground"],
                                relief=facts["relief"],
                                round_boundaries=facts["round_boundaries"],
                                intent=contracts.load(rnd, "intent"))
        contracts.save(rnd, "capabilities", caps)
    except Exception as e:                       # noqa: BLE001 -- reported, never fatal
        print(f"   growth: the capability record could not be refreshed: "
              f"{type(e).__name__}: {e}", flush=True)


def stage(rnd, be, spec: dict, gaps: list, *, types=None, site=None,
          voice: str | None = None) -> dict | None:
    """Grow the library by the parts in `gaps`, one ask at a time.

        Returns the plan stage's answer where the run has to wait or stop -- a
        `needs_model` for the author, an error where an answer fails the gate or the cap
        is reached -- and None where every gap is closed and planning may go on.
        
    """
    from .pipeline import blind
    from .pipeline.stages_build import type_brief
    rec = record_of(rnd)
    rec["gaps"] = [type_name(p) for p in gaps]
    if voice is None:
        from .pipeline.stages_plan import _choose_voice
        voice = (spec.get("voice") if spec.get("voice") in styles.VOICES
                 else (_choose_voice(spec, site, check_types=False) if site
                       else styles.silent_voice()))
    place_voice = rnd.voice_name() or voice
    for part in gaps:
        name = type_name(part)
        sub = os.path.join("growth", name)
        path = _pipeline._resolve(os.path.join("types", f"{name}.py"))
        if os.path.exists(path):
            # **A file with the right name is not the capability.** The architecture
            # audit reproduced the bypass this replaces: `type_gaps` opened a gap
            # because the only `tower` on disk was of the wrong form, and this line --
            # `if os.path.exists(path)` -- closed it again with that very file, so the
            # place was planned around a capability it did not have. The file is now a
            # *candidate*: it is loaded, its declarations are read, and it is adopted
            # only where they answer the gap that was opened.
            from . import capability
            want = next((w for w in capability.wants_of(spec)
                         if w.get("part") == part["name"]), None)
            if part.get("function"):
                try:
                    decl = _pipeline.load_type(path)
                    ok = _declares_function(decl, part["function"], spec.get("form"))
                    why = "" if ok else (f"the file already at types/{name}.py declares "
                                         f"FUNCTION={decl.get('function')!r}, FORM="
                                         f"{decl.get('form')!r} and this gap is "
                                         f"`{part['function']}` in {spec.get('form')!r}")
                except Exception as e:           # noqa: BLE001 -- reported, not raised
                    ok, why = False, f"types/{name}.py does not load as a type: {e}"
            else:
                ok, why = (capability.adoptable(path, want) if want
                           else (True, ""))
            if ok:
                rec["adopted"].setdefault(name, {"note": "already on disk, and its "
                                                         "declarations answer this gap",
                                                 **({"answers": list(part["answers"]),
                                                     "function": part["function"]}
                                                    if part.get("function") else {})})
                _save(rnd, rec)
                _refresh_capabilities(rnd, spec, types, site)
                continue
            rec["failed"][name] = {"why": why, "file": os.path.relpath(
                path, _pipeline.ROOT), "passes": False}
            _save(rnd, rec)
            return {"plan": {"status": "error", "stop": True, "level": f"type/{name}",
                             "error": f"the library cannot build {part['name']} "
                                      f"({part['family']}, {part['kind']}): {why}. "
                                      f"A type of the right name is not a type of the "
                                      f"right kind, form, role and envelope, and one "
                                      f"cannot be adopted over the other",
                             "growth": rec["failed"][name]}}
        if name in rec["failed"]:
            _save(rnd, rec)
            return {"plan": {"status": "error", "stop": True, "level": f"type/{name}",
                             "error": f"the library could not grow a {name}: the "
                                      f"authored type failed its gate, and the model "
                                      f"is not asked twice -- "
                                      f"{rec['failed'][name].get('why', '')}",
                             "growth": rec["failed"][name]}}
        builds = rnd.rel("arms", sub, "builds")
        os.makedirs(builds, exist_ok=True)
        src_path = os.path.join(builds, "build_0.py")
        req = os.path.join(builds, "build_request_0.json")
        asked = rec["asked"].get(name)
        if asked:
            blind._collect_blinded(rnd, [(name, sub)])
        if os.path.exists(src_path):
            tspec = asked["tspec"]
            got = gate(rnd, be, tspec, src_path, place_voice)
            fp = rnd.rel("growth", f"{name}.findings.md")
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(got.pop("text") or "")
            got["findings"] = fp
            if got["passes"]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.copyfile(src_path, path)
                got["file"] = os.path.relpath(path, _pipeline.ROOT)
                if tspec.get("function"):
                    got["answers"] = list(tspec.get("answers") or [])
                    got["function"] = tspec["function"]
                rec["adopted"][name] = got
                print(f"   growth: {name} adopted as {got['file']} -- "
                      f"{got['instances']} instances clean in "
                      f"{', '.join(got['voices'])}", flush=True)
                _save(rnd, rec)
                _refresh_capabilities(rnd, spec, types, site)
                continue
            got["why"] = (f"{got['errors']} error(s), {got['entry_lines']} way-in "
                          f"line(s), crashed {got['crashed']} over {got['instances']} "
                          f"instance(s); per voice "
                          + "; ".join(f"{v}: {d['clean']}/{d['instances']} clean"
                                      for v, d in got["voices"].items())
                          + (f"; {got['function_refused']}" if got.get("function_refused")
                             else ""))
            rec["failed"][name] = got
            _save(rnd, rec)
            return {"plan": {"status": "error", "stop": True, "level": f"type/{name}",
                             "error": f"the library could not grow a {name}: the "
                                      f"authored type fails its gate -- {got['why']}; "
                                      f"the findings are at {fp}",
                             "growth": got}}
        if not asked and len(rec["asked"]) >= rec["cap"]:
            rec["capped"].append(name)
            _save(rnd, rec)
            return {"plan": {"status": "error", "stop": True, "level": f"type/{name}",
                             "error": f"the library cannot grow a {name} in this run: "
                                      f"{rec['cap']} type(s) is the cap and "
                                      f"{sorted(rec['asked'])} were asked",
                             "growth": {"capped": rec["capped"]}}}
        if not asked:
            fixtures = default_fixtures(part["kind"])
            tspec = _tspec(rnd, spec, part, voice, fixtures)
            brief_path = rnd.rel("growth", "briefs", f"{name}.md")
            os.makedirs(os.path.dirname(brief_path), exist_ok=True)
            open(brief_path, "w").write(type_brief(rnd, tspec, fixtures))
            json.dump({"i": 0, "images": [], "brief": open(brief_path).read()},
                      open(req, "w"), indent=1)
            rec["asked"][name] = {"part": part["name"], "kind": part["kind"],
                                  "family": part["family"], "brief": brief_path,
                                  "request": req, "tspec": tspec,
                                  "fixtures": [f"{f['round']}/"
                                               f"{f.get('plot') or f.get('part')}"
                                               for f in fixtures]}
            _save(rnd, rec)
            asked = rec["asked"][name]
        blinded = blind._blind_build(rnd, sub, req, check=("type", asked["tspec"]))
        _save(rnd, rec)
        return {"plan": {"status": "needs_model", "role": "type", "request": req,
                         "write": src_path, "images": [], "blinded": blinded,
                         "level": f"type/{name}",
                         "note": f"the library lacks a {part['family']} ({part['kind']}) "
                                 f"for {part['name']}: one type authored blind, checked "
                                 f"on {len(asked['fixtures'])} fixture(s), adopted when "
                                 f"clean; {len(rec['asked'])} of {rec['cap']} this run"}}
    _save(rnd, rec)
    return None
