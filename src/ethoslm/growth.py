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


def type_gaps(spec: dict, names=None) -> list:
    """The defining parts no committed type builds, in spec order.

        A part of one of `LEAF_KINDS` (never a compound: its halls and walls are the
        committed types') with no type of its kind **named for its family** -- `tower` or
        `tower_*` -- among the types of the place's form (and the part's own), which is
        the rule `stages_plan._choose_voice` refuses on and `placesolve._type_for` places
        by. `names` is the round's own list of types where it has one.
        
    """
    from .placeplan import types_card
    out = []
    for p in spec.get("defining_parts") or []:
        if p.get("kind") not in LEAF_KINDS or spec_mod.compound(p):
            continue
        forms = [f for f in [spec.get("form"), *(p.get("forms") or [])] if f]
        _t, decls = types_card(names, forms=forms or [None])
        fam = p["family"]
        if any(d.get("kind", "plot") == p["kind"]
               and (n == fam or n.startswith(fam + "_")) for n, d in decls.items()):
            continue
        out.append(p)
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
            f"instance is clean in both voices.\n")


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
    return {"passes": bool(passes), "instances": res.get("instances"),
            "errors": res.get("errors"), "entry_lines": res.get("entry_lines"),
            "crashed": bool(res.get("crashed")), "voices": by,
            "fixtures": [f"{f['round']}/{f.get('plot') or f.get('part')}"
                         for f in fixtures], "seeds": seeds,
            "sweep": bool(tspec.get("check_sweep", True)),
            "seconds": res.get("seconds"), "text": res.get("text", "")}


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
            rec["adopted"].setdefault(name, {"note": "already on disk"})
            continue
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
                rec["adopted"][name] = got
                print(f"   growth: {name} adopted as {got['file']} -- "
                      f"{got['instances']} instances clean in "
                      f"{', '.join(got['voices'])}", flush=True)
                _save(rnd, rec)
                continue
            got["why"] = (f"{got['errors']} error(s), {got['entry_lines']} way-in "
                          f"line(s), crashed {got['crashed']} over {got['instances']} "
                          f"instance(s); per voice "
                          + "; ".join(f"{v}: {d['clean']}/{d['instances']} clean"
                                      for v, d in got["voices"].items()))
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
