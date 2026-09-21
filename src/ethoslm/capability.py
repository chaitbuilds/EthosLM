"""What the library can build, described by what it *is* rather than by what it is called.

The audit's finding, at `growth.py:39` and `growth.py:183`: a defining part was matched
to a type by **filename prefix**, and a gap the matcher had just opened was closed again
two hundred lines later by `if os.path.exists(path)` -- so a `tower` of the wrong form,
the wrong kind, or an envelope that cannot hold the part was adopted as the answer to
"this library has no tower". The wrong-form bypass was reproducible and this module is
where it stops.

A capability is five facts about a type, all of them already on disk and none of them
its name:

    kind        plot, edge, point or area -- what shape of part it builds
    form        the tradition it is built in (`pipeline.form_ok` decides compatibility)
    role        what it is for: urban, rural, civic, defensive
    envelope    the footprint band its own sweep was checked across, plus the ground
                classes it will stand on
    attachment  whether its flanks are party walls, and whether a network may pass
                through it

Matching asks all five. A **name** is used for one thing only -- ranking two otherwise
equal candidates, because a type called `keep` is a better answer to "a keep" than one
called `hall` -- and never to admit a candidate the five facts refuse.

What this module will not do is substitute. An uncovered requirement comes back as an
entry with `status: "uncovered"` and the reason; the caller may then grow the library
(`ethoslm.growth`) or stop. Renaming a hall to a keep is not a capability.
"""
from __future__ import annotations

import os

from . import contracts, spec as spec_mod

#: The part kinds a type builds. A `group` -- a district, a compound -- is not built by
#: a type: it is planned, out of the types below it.
LEAF_KINDS = ("plot", "edge", "point", "area")


#: The ground a type will stand on, as a set. Every committed type says `any` today, so
#: the *constraint* below is vacuous against this library and the **mechanism** is not:
#: a type that says `dry` is refused a wet want, and a want with no ground stated asks
#: nothing. `any` is the wildcard and is spelled out rather than being the empty set,
#: because "this type declares nothing" and "this type declares it stands anywhere" are
#: different facts and the second is the one that may be relied on.
GROUND_ANY = "any"


def _ground_set(v) -> list:
    """`needs.ground` as a list of ground classes, however it was written.

        Found by reading the cards this produced: `needs = {"ground": "any"}` is what all
        28 committed types declare, and `list("any")` is `['a', 'n', 'y']` -- a card whose
        ground classes were three letters. It never bit because nothing asked; the
        integration round asks.
        
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def card(name: str, decl: dict) -> dict:
    """One type's capability card: the five facts, off its own declaration."""
    needs = decl.get("needs") or {}
    fp = needs.get("footprint") or (1, 1, 512, 512)
    return {"type": name,
            "kind": decl.get("kind", "plot"),
            # what family of part it builds: its own word where it says one, and
            # otherwise the name it is committed under (`family_of`)
            "family": decl.get("family"),
            "form": decl.get("form"),
            "role": decl.get("role"),
            "envelope": {"footprint": [int(v) for v in fp],
                         "ground": _ground_set(needs.get("ground")),
                         "clearance": needs.get("clearance"),
                         "frontage": needs.get("frontage"),
                         # **The extension point for a tradition, and it is empty.**
                         # `FORM` is four coarse families and a request for German or
                         # Japanese construction is finer than any of them. A type that
                         # is actually built in a named tradition says `TRADITION`, and
                         # `intent.coverage` will then let the tradition requirement
                         # reach `satisfied`. No committed type says it today, which is
                         # why every such requirement is `unresolved` and the round
                         # reports it as unproven rather than as met.
                         "tradition": decl.get("tradition"),
                         # **What the type is for**, declared rather than inferred from
                         # its role. See `pipeline.load_type`: a `rural` role admits a
                         # hall, a barn and a temple, and a request for houses people
                         # live in is a request for a `dwelling`. `intent._function_
                         # measure` reads this and nothing else, so a function is
                         # fulfilled by a type that says it does, or by nothing.
                         "function": decl.get("function"),
                         "max_relief": needs.get("max_relief")},
            "attachment": {"attached": bool(decl.get("attached")),
                           "passage": bool(decl.get("passage")),
                           "diagonal": bool(decl.get("diagonal"))},
            "params": sorted((decl.get("params") or {})),
            "declares_needs": bool(decl.get("declares_needs"))}


def cards(decls: dict) -> dict:
    """`{type name: capability card}` for a table of declarations."""
    return {n: card(n, d) for n, d in sorted((decls or {}).items())}


def _named_for(name: str, family: str) -> bool:
    """Does this type answer the family, by the name it is committed under?

        The one thing a type file does **not** declare is what family of part it is: a
        `PARAMS` and a `NEEDS` say how big a thing may be and what ground it wants, and
        nothing in either says "this is a keep". So the family is read off the committed
        name -- `keep` or `keep_*` -- which is the rule the planner has always placed by
        and the rule this project's whole `types/` directory is arranged under. A type may
        say so itself (`FAMILY = "keep"`), and where it does its own word wins; `family_of`
        is where the two come together.

        This is a **necessary** condition and never a sufficient one. The audit's finding
        was not that names were consulted -- it was that a name was consulted *instead of*
        the declarations, so a `tower` of the wrong form was adopted as a tower. `fits`
        asks the declarations as well, every time.
        
    """
    return name == family or name.startswith(family + "_")


def family_of(c: dict, family: str) -> bool:
    """Does capability `c` answer `family`? Its own declaration, else its name."""
    said = c.get("family")
    if said:
        return str(said) == family
    return _named_for(c["type"], family)


def fits(c: dict, want: dict) -> tuple:
    """Does capability `c` answer requirement `want`? `(bool, why not)`.

        Every clause after the first is a fact the type declared about itself. The order is
        the order a reader wants the refusal in: the wrong family first, then the wrong
        shape of part, then the wrong tradition, then the wrong purpose, then a footprint
        that cannot hold the thing.
        
    """
    from . import pipeline as _pipeline
    # a type declaring the function the part is asked for answers the part whatever it
    # is named (the expression round, B's R7): a `worship` chapel answers a hall part
    # asked to be a chapel; the function clause below still holds it to the function
    fn_match = bool(want.get("function")) and str(
        (c.get("envelope") or {}).get("function") or "") == str(want.get("function"))
    if want.get("family") and not family_of(c, want["family"]) and not fn_match:
        return (False, f"{c['type']} is not a {want['family']}")
    if c["kind"] != want["kind"]:
        return (False, f"{c['type']} builds a {c['kind']} and a {want['name']} is a "
                       f"{want['kind']}")
    forms = [f for f in (want.get("forms") or []) if f] or [None]
    if not any(_pipeline.form_ok(c["form"], f) for f in forms):
        return (False, f"{c['type']} is `{c['form']}` and this place is "
                       f"{'/'.join(str(f) for f in forms)}")
    roles = [r for r in (want.get("roles") or []) if r] or [None]
    if not any(_pipeline.role_ok(c["role"], r) for r in roles):
        return (False, f"{c['type']} is for `{c['role']}` work and this part is "
                       f"{'/'.join(str(r) for r in roles)}")
    # **What the type is for**, where the request asked for a function by name. A type
    # that declares none does not fulfil one: the obligation stays visible rather than
    # being closed by whichever building the role happened to admit.
    if want.get("function"):
        got = (c.get("envelope") or {}).get("function")
        if not got:
            return (False, f"{c['type']} declares no function, and this part is asked to "
                           f"be a {want['function']}; a role is what a building is for "
                           f"in the coarse, and it does not make a hall a dwelling")
        if str(got) != str(want["function"]):
            return (False, f"{c['type']} is a {got} and this part is asked to be a "
                           f"{want['function']}")
    if want.get("passage") and not c["attachment"]["passage"]:
        return (False, f"{c['type']} is not a passage and a {want['name']} is walked "
                       f"through")
    if want.get("diagonal") and not c["attachment"]["diagonal"]:
        return (False, f"{c['type']} draws no diagonal run and this boundary is round")
    # **The three construction constraints the round found missing.** The review's sixth
    # finding was reproduced with one call: a detached, dry-ground, zero-relief cottage
    # answered a want for an attached home on wet sloping ground, because `fits` asked
    # five questions and none of them was about the ground the thing has to stand on. A
    # capability that does not include "will it stand here" is a filename with extra
    # steps, which is the defect this module was written to stop one level up.
    if want.get("attached") and not c["attachment"]["attached"]:
        return (False, f"{c['type']} builds free-standing and this part shares party "
                       f"walls with its neighbours")
    ground = want.get("ground")
    have = c["envelope"]["ground"]
    if ground and have and GROUND_ANY not in have and str(ground) not in have:
        return (False, f"{c['type']} stands on {'/'.join(have)} ground and this part "
                       f"stands on {ground}")
    # `relief` is the want's word for it; `max_relief` is accepted as well because that
    # is the word the type declaration uses and a caller reading one side of the
    # contract should not have to remember that the other side renamed it.
    relief = want.get("relief", want.get("max_relief"))
    most = c["envelope"].get("max_relief")
    if relief is not None and most is not None and float(relief) > float(most):
        return (False, f"{c['type']} was checked to {most} blocks of relief across its "
                       f"footprint and this part stands on {relief}")
    need = want.get("footprint")
    if need:
        lo_w, lo_d, hi_w, hi_d = c["envelope"]["footprint"]
        w, d = int(need[0]), int(need[1])
        if not (lo_w <= w <= hi_w and lo_d <= d <= hi_d):
            return (False, f"{c['type']} was checked from {lo_w}x{lo_d} to {hi_w}x{hi_d} "
                           f"and this part needs {w}x{d}")
    return (True, "")


def _ground_want(spec: dict, ground: str | None) -> str | None:
    """The ground class the place's parts have to stand on, where one is known."""
    if ground:
        return str(ground)
    setting = (spec or {}).get("setting") or {}
    if isinstance(setting, dict) and setting.get("ground"):
        return str(setting["ground"])
    return None


def _part_footprint(p: dict) -> list | None:
    """The footprint a defining part has actually been resolved to, where it has one.

        `needs.footprint` on a defining part is a *want*, two numbers, not the four-number
        band a type declares. Only a resolved pair is returned: a want with no footprint
        asks the library nothing about size, which is honest, and a want carrying a band
        would be comparing a type's range against another range and passing whatever
        overlapped -- which is how an envelope stops meaning anything.
        
    """
    need = (p.get("needs") or {}).get("footprint")
    if not need or len(list(need)) != 2:
        return None
    return [int(need[0]), int(need[1])]


def requirement_for(intent: dict | None, part: dict, family: str | None) -> str | None:
    """The sentence's requirement this defining part answers, where one names it.

        **The link the record was declaring and not carrying.** Every capability entry has
        a `requirement` field and `wants_of` never filled one, so "which type answers which
        clause of the sentence" was a column of nulls -- and the round's contract is that a
        requirement, the design choice that answers it and the evidence for it are one
        chain. `resolve._requirement_for` does the same job for a *region*; this does it for
        a capability, by the same rule and against the same record.
        
    """
    for r in (intent or {}).get("requirements") or []:
        w = r["wants"]
        if family and r["kind"] in ("feature", "absent") and w.get("family") == family:
            return r["id"]
        if r["kind"] == "tradition" and (part.get("forms") or part.get("tradition")):
            continue
    return None


def requirements_for(spec: dict | None, intent: dict | None, part: dict) -> dict:
    """**Which clauses of the sentence this part has to answer**, and what they demand.

            {"ids": [requirement_id],            # in the order the sentence states them
             "required": (feature token, ...),   # what its type must actually deliver
             "optional": (),                     # filled by the type, not by the sentence
             "params": {"storeys": 2},           # parameters a requirement fixes outright
             "why": {token or param: [ids]}}

        `envelope.required_features` answers a related question -- which tokens a set of
        requirements names -- and it filters only on `scope`. That is not enough to ask a
        part for: `feature/market` has no scope, so every cottage in a farming village was
        being asked to deliver `stalls`, and a cottage that cannot is a capability gap that
        does not exist. A requirement reaches a part three ways and no other:

          * it **names** the part, through a scope (`intent._in_scope`);
          * it asks for a **feature or a function** and this part is the one that answers it
            (the same rule `requirement_for` and `_function_wanted` already use for the
            capability record);
          * it is a **fabric quality** -- density, height -- and this part is fabric: a
            district or a building. Ground and open space have no storeys.

        This is the coordinator's half of the resolved-demand contract; `demand.resolve`
        calls it and hands what comes back to the envelope, and the ids travel with the
        answer so a lot that had to grow can say which clause grew it.
        
    """
    from . import envelope as envelope_mod
    from . import intent as intent_mod
    from . import spec as spec_mod
    p = part or {}
    ent = spec_mod.entity_of(p, (spec or {}).get("explicit_count"))
    fam = str(p.get("family") or "").lower()
    #: **A quality of the fabric is about the buildings, and a feature is about the
    #: thing that builds it.** A farming village's fields hold a few counted cottages
    #: and those cottages are fabric; the ground they stand on has no storeys. A market
    #: *square* answers `feature/market` by standing there -- it is open ground, and
    #: asking an area type to emit `stalls` invents a capability gap. A market
    #: *building* is a `building` entity and is asked.
    holds_buildings = ent.get("class") == "building" or (
        ent.get("class") == "land" and ent.get("counted"))
    #: **A district is planned; the things in it are built.** A feature requirement
    #: about a district -- `function/market` scoped to the middle ring -- is answered by
    #: something *standing in* it, which is `capability.wants_of`'s landmark want and
    #: the place read's feature clause. It is not a token every house of that district's
    #: fabric has to emit: asking a courtyard house for market stalls refused the whole
    #: city by name (`middle_ring: no approved type delivers stalls`). A quality is
    #: different -- `dense`, and the storeys -- because those *are* properties of the
    #: fabric, and they stay below.
    builds_features = ent.get("class") in ("building", "compound") \
        and p.get("kind") != "group"
    ids, required, params, why = [], [], {}, {}

    def take(rid, tokens=(), fixes=None):
        if rid not in ids:
            ids.append(rid)
        for t in tokens:
            if t not in required:
                required.append(t)
            why.setdefault(t, []).append(rid)
        for k, v in (fixes or {}).items():
            params[k] = v
            why.setdefault(k, []).append(rid)

    for r in (intent or {}).get("requirements") or []:
        if not isinstance(r, dict) or r.get("status") == "unsupported":
            continue
        kind, w, scope = r.get("kind"), r.get("wants") or {}, r.get("scope")
        fixes = ({"storeys": int(w["storeys"])}
                 if w.get("exact") and w.get("storeys") else None)
        if scope is not None:
            if not intent_mod._in_scope({"name": p.get("name"),
                                         "defines": p.get("defines") or p.get("name"),
                                         "in": p.get("in") or []}, scope):
                continue
            take(r["id"], envelope_mod.required_features([r], p) if builds_features
                 or kind == "quality" else (), fixes)
            continue
        if kind == "quality" and w.get("axis") == "height":
            if holds_buildings:
                take(r["id"], (envelope_mod.STOREYS,), fixes)
        elif kind == "quality":
            if holds_buildings:
                take(r["id"])
        elif kind in ("feature", "function"):
            said = str(w.get("feature") or w.get("function") or w.get("what") or "")
            # an **unscoped** function is about the place's ordinary fabric, which is
            # what a district is (`_function_wanted`). A market square is not a dwelling
            # because a village's houses are, and neither is the hall on it: the hall
            # answers `feature/hall`, and `function/dwelling` is the cottages' clause.
            is_fabric = p.get("kind") == "group" and holds_buildings
            answers = (requirement_for(intent, p, fam) == r["id"]
                       or ((is_fabric or r.get("scope")) and
                           _function_wanted(intent, p) == said)
                       or (w.get("family") and w["family"] == fam))
            if answers:
                take(r["id"], envelope_mod.required_features([r], p)
                     if builds_features else ())
    return {"ids": ids, "required": tuple(required), "optional": (),
            "params": params, "why": why}


def _function_wanted(intent: dict | None, part: dict) -> str | None:
    """The function the request asks this defining part's **fabric** to fulfil, or None.

        A `function` requirement with no scope is about the place's ordinary fabric, which is
        what a district is; one with a scope is about the part whose name it gives. Nothing
        is inferred from a role or a family here -- the whole point is that those are what
        could not tell a dwelling from a hall.
    """
    from . import spec as spec_mod
    fabric = part.get("kind") == "group"
    for r in (intent or {}).get("requirements") or []:
        if r.get("kind") != "function" or r.get("status") == "unsupported":
            continue
        w = r.get("wants") or {}
        fn = str(w.get("function") or w.get("what") or "")
        if not fn:
            continue
        what = str(w.get("what") or "").lower().rstrip("s")
        if fabric and what and what not in {h.rstrip("s") for h in
                                            spec_mod._HOUSE_FAMILIES} \
                and what not in ("house", "home", "dwelling", "building", "structure"):
            continue
        scope = r.get("scope")
        if not scope:
            return fn
        from . import intent as intent_mod
        if intent_mod._in_scope({"name": part.get("name"),
                                 "defines": part.get("name"),
                                 "in": [part.get("name")]}, scope):
            return fn
    return None


def wants_of(spec: dict, *, round_boundaries: bool = False,
             ground: str | None = None, relief=None,
             intent: dict | None = None) -> list:
    """What this spec's programme needs the library to be able to build.

        One entry per **leaf** defining part -- a wall, a gate, a keep -- and one per
        **compound component** the family's composition declares, because a palace that has
        no hall type is as uncovered as a place that has no wall type and the old matcher
        could not say so. District fabric is covered by the plot types the compiler selects
        and is asked for as a plot want per district role.

        `ground` and `relief` are the **site's** facts and they are passed through to every
        want, because whether a type will stand somewhere is a fact about the pair and not
        about the type. The review's finding was that production never passed them at all,
        so `fits` was answering a question with half its terms missing.
        
    """
    out = []
    forms_place = [spec.get("form")] if spec.get("form") else []
    on = _ground_want(spec, ground)

    def site_facts(w: dict) -> dict:
        if on:
            w["ground"] = on
        if relief is not None:
            w["relief"] = relief
        # the clause of the sentence this want answers, so the record is a chain from
        # the request to the type that was chosen for it
        part = next((q for q in (spec.get("defining_parts") or [])
                     if q.get("name") == w.get("part")), {})
        w["requirement"] = requirement_for(intent, part, w.get("family")
                                           or part.get("family"))
        return w

    for p in (spec.get("defining_parts") or []):
        forms = [f for f in [*(p.get("forms") or []), *forms_place] if f]
        if spec_mod.compound(p):
            made = spec_mod.composition(p)
            roles = [p.get("role") or "civic", *made["admits"]]
            if made["walled"]:
                out.append(site_facts(
                    {"id": f"cap/{p['name']}/wall", "name": f"{p['name']} wall",
                     "kind": "edge", "forms": forms, "roles": list(roles),
                     "part": p["name"], "family": "wall", "of": "compound",
                     **({"diagonal": True} if round_boundaries else {})}))
            if made["gated"]:
                out.append(site_facts(
                    {"id": f"cap/{p['name']}/gate", "name": f"{p['name']} gate",
                     "kind": "point", "forms": forms, "roles": list(roles),
                     "part": p["name"], "family": "gate", "passage": True,
                     "of": "compound"}))
            if made["halls"]:
                out.append(site_facts(
                    {"id": f"cap/{p['name']}/hall", "name": f"{p['name']} hall",
                     "kind": "plot", "forms": forms,
                     "roles": [p.get("role") or "civic"],
                     "part": p["name"], "family": "hall", "of": "compound"}))
            if made["courts"]:
                out.append(site_facts(
                    {"id": f"cap/{p['name']}/court", "name": f"{p['name']} court",
                     "kind": "area", "forms": forms,
                     "roles": [p.get("role") or "civic"],
                     "part": p["name"], "family": "square", "of": "compound"}))
            continue
        if spec_mod.district(p):
            # **A district's fabric names no family.** What fills a district is
            # "whatever plot types of this role this place's form admits", which is what
            # `district_compile.house_types` selects from; a family here would ask for a
            # `house.py` no library has ever had. It does, however, name an
            # **attachment**: a district whose character says its houses share party
            # walls needs plot types that build attached, and a library of free-standing
            # cottages cannot fill it. That is a capability gap and it used to be
            # discovered by the compiler, one district at a time.
            ch = p.get("character") or {}
            want = {"id": f"cap/{p['name']}/fabric", "name": f"{p['name']} fabric",
                    "kind": "plot", "forms": forms,
                    "roles": [p.get("role")
                              or spec_mod.read_role(None, p, p["name"])],
                    "part": p["name"], "family": None, "of": "fabric"}
            if ch.get("attached"):
                want["attached"] = True
            # **And what the fabric is FOR, where the request said.** The review's third
            # finding in its semantic form: "An allowed plot type and a coarse role do
            # not establish that a dwelling's function has been fulfilled." Run without
            # this, a sentence asking for a village of houses produced a district of
            # seven halls and two cottages -- every one of them admitted, because
            # `rural` is what a hall, a barn and a cottage all are. A function
            # requirement the interpretation read is the one thing that can tell them
            # apart, and this is where it reaches the choice instead of only the check.
            fn = _function_wanted(intent, p)
            if fn:
                want["function"] = fn
            out.append(site_facts(want))
            continue
        if p.get("kind") not in LEAF_KINDS:
            continue
        want = {"id": f"cap/{p['name']}", "name": p["name"], "kind": p["kind"],
                "forms": forms, "roles": [p.get("role")], "part": p["name"],
                "family": p["family"], "of": "defining_part"}
        # **a function the request asks of this named part** (B's R7): "a chapel hall"
        # reads as `function/worship` with `what: hall`, and the hall part carries it,
        # so the chapel is built as the type that declares worship and not as a hall
        # beside a separate chapel
        for r in (intent or {}).get("requirements") or []:
            if r.get("kind") != "function" or not r.get("hard") \
                    or r.get("status") == "unsupported":
                continue
            w = r.get("wants") or {}
            what = str(w.get("what") or "").lower().rstrip("s")
            if what and what in (str(p["family"]).lower(), str(p["name"]).lower()):
                want["function"] = str(w.get("function"))
                want["requirement_function"] = r.get("id")
        if p["family"] == "gate":
            want["passage"] = True
        if p["family"] == "wall" and round_boundaries:
            want["diagonal"] = True
        fp = _part_footprint(p)
        if fp:
            want["footprint"] = fp
        out.append(site_facts(want))
    # **What the request says the place is FOR, where no defining part carries it.** The
    # expression round (worker B's request): a `function/market` or a courtyard feature
    # the interpretation read and no part names never became a want, so the growth path
    # could not open a gap for it. One want per such requirement, `of: "function"`, so
    # `match` records it covered or uncovered and `gaps` can grow it.
    answered = {str(w.get("function") or "") for w in out} | {
        str(w.get("family") or "") for w in out}
    place_words = ("village", "town", "city", "hamlet", "place", "settlement", "ring",
                   "district", "quarter")
    for r in (intent or {}).get("requirements") or []:
        if r.get("kind") not in ("function", "feature") or r.get("status") == "unsupported":
            continue
        # **a hard function of a buildable thing.** The farm's `function/farming` is a
        # soft reading of "farming village" -- what the whole place is for -- and it
        # opened an authoring job for a `farming` type; a gap is a capability the
        # request requires of a building, not a word about the place
        if not r.get("hard"):
            continue
        w = r.get("wants") or {}
        if str(w.get("what") or "").strip().lower() in place_words:
            continue
        fn = str(w.get("function") or w.get("feature") or "").strip().lower()
        if not fn or fn in answered or fn in ("dwelling", "wall", "farmland", "farming"):
            continue
        if r.get("kind") == "feature" and w.get("family"):
            continue                      # a family want above answers it
        answered.add(fn)
        out.append(site_facts({"id": f"cap/function/{fn}", "name": fn,
                               "kind": "area" if fn in ("market", "stalls") else "plot",
                               "forms": list(forms_place), "roles": [None],
                               "part": None, "family": None, "of": "function",
                               "function": fn, "requirement": r.get("id")}))
    return out


#: How many runner-up types a covered want records. A district's fabric is built out of
#: a *pool* of plot types and not out of one, so a record naming one type and nothing
#: else cannot be compared with what the compiler did.
ALTERNATIVES = 8


def _rank(name: str, c: dict, want: dict) -> tuple:
    """The order candidates are preferred in. **Never admits** -- `fits` has already
        said yes to everything this sees.

        The order the module's docstring has always claimed and the code did not do: a file
        committed under the family's own name first (`wall` answers a wall before
        `great_wall` does), then one committed under a name derived from it, then -- for a
        district's fabric, which names no family -- exactly the order
        `district_compile.house_types` draws in, so the record's lead type is the type the
        compiler will actually reach for. Widest envelope breaks what is left.

        The previous order was widest-envelope-first for everything, and it is why the
        round's own report recorded a match of `wall` while the ring layout built
        `great_wall`: two rules, one library, two answers.
        
    """
    fam = want.get("family")
    if fam:
        exact = 0 if name == fam else (1 if _named_for(name, fam) else 2)
    else:
        # the fabric want: the role's own types first, as the compiler sorts them
        roles = [r for r in (want.get("roles") or []) if r]
        exact = 0 if (roles and c.get("role") == roles[0]) else 1
    fp = c["envelope"]["footprint"]
    return (exact, -(int(fp[2]) * int(fp[3])), name)


def match(spec: dict, decls: dict | None = None, *, names=None,
          round_boundaries: bool = False, ground: str | None = None,
          relief=None, intent: dict | None = None) -> dict:
    """The capabilities record: what is covered, by which type, at what envelope.

        `decls` is a table of type declarations; where none is given the committed types are
        read off disk. **The form filter is not applied to the table** -- it is applied per
        want by `fits`, so a refusal can say *which* fact refused it instead of the type
        simply being absent from a list.
        
    """
    from .placeplan import types_card
    if decls is None:
        _t, decls = types_card(names, forms=[None])
    cs = cards(decls)
    entries = []
    for want in wants_of(spec, round_boundaries=round_boundaries, ground=ground,
                         relief=relief, intent=intent):
        ok, refused = [], []
        for n, c in cs.items():
            good, why = fits(c, want)
            (ok if good else refused).append((n, c, why))
        if ok:
            ranked = sorted(ok, key=lambda r: _rank(r[0], r[1], want))
            n, c, _ = ranked[0]
            entries.append({"id": want["id"], "requirement": want.get("requirement"),
                            "wants": want, "kind": want["kind"],
                            "family": want["family"], "form": c["form"],
                            "role": c["role"], "matched": True, "type": n,
                            "alternatives": [r[0] for r in ranked[1:ALTERNATIVES]],
                            "envelope": c["envelope"], "checked": c["declares_needs"],
                            "status": "covered",
                            "why": f"{n} is a {c['kind']} of form {c['form']} for "
                                   f"{c['role'] or 'any'} work, checked from "
                                   f"{c['envelope']['footprint'][0]}x"
                                   f"{c['envelope']['footprint'][1]} to "
                                   f"{c['envelope']['footprint'][2]}x"
                                   f"{c['envelope']['footprint'][3]}"})
            continue
        # **The nearest miss, named.** "No tower" is much less useful than "the only
        # tower on disk is `european_vernacular` and this place is `east_asian`", and
        # the second is what tells a reader whether to grow the library or fix the form.
        near = [(n, why) for n, _c, why in refused
                if want["family"] and _named_for(n, want["family"])]
        entries.append({"id": want["id"], "requirement": want.get("requirement"),
                        "wants": want, "kind": want["kind"], "family": want["family"],
                        "form": None, "role": None, "matched": False, "type": None,
                        "alternatives": [],
                        "envelope": None, "checked": False, "status": "uncovered",
                        "why": ("no committed type builds this: "
                                + ("; ".join(f"{n} -- {w}" for n, w in near[:3])
                                   if near else
                                   f"nothing on disk is a {want['kind']} for a "
                                   f"{want['family'] or want['name']}"))})
    return contracts.make("capabilities", entries=entries,
                          note="matched on declared kind, form, role, envelope and "
                               "attachment; a filename ranks and never admits")


def uncovered(rec: dict) -> list:
    """The wants nothing on disk answers, in order."""
    return [e for e in (rec or {}).get("entries") or [] if not e.get("matched")]


def _declaration(name: str, decls: dict | None):
    """A type's declaration, off the table or off disk. None where it will not load."""
    from . import pipeline as _pipeline
    got = (decls or {}).get(name)
    if got is not None:
        return got
    try:
        return _pipeline.load_type(os.path.join(_pipeline.ROOT, "types", f"{name}.py"))
    except Exception:                          # noqa: BLE001 -- reported by the caller
        return None


def _types_used(place: dict, plan: dict | None) -> dict:
    """`{defining part: [every type the place was actually built out of]}`.

        **Every leaf, and every kind of want.** The review's account of the previous rule is
        exact: it recorded the *first* type it saw per defining part, took its parts from the
        place level only, and skipped compound components and district fabric entirely. So a
        district whose first house was the approved type and whose remaining forty were not
        produced no finding at all, and a palace's halls were never compared with anything.
        
    """
    from . import pipeline as _pipeline
    out: dict = {}

    def note(part, tname, leaf_kind="plot"):
        if part and tname:
            row = out.setdefault(str(part), {})
            got = row.setdefault(str(leaf_kind or "plot"), [])
            if str(tname) not in got:
                got.append(str(tname))

    place = place or {}
    for key in ("parts", "compounds", "districts"):
        for p in place.get(key) or []:
            note(p.get("defines"), p.get("type"), p.get("kind", "plot"))
    # `answers` is the defining part a leaf inherits from the group it is in -- see
    # `pipeline.plan_parts`. Without it a district's forty houses named no part and the
    # fabric the whole reconciliation exists to check was invisible. **Kept by leaf
    # kind, because a want is for one kind.** Found by running the shore village once
    # the inheritance was in: a district's *open ground* -- its groves and gardens,
    # which are `area` leaves -- inherited the district's defining part along with its
    # houses, and were then reconciled against the `fabric` want, which is for a `plot`.
    # The record said the place had built its homes out of `grove`. The ground between
    # the houses is not the fabric, and this is where the two are told apart.
    for p in _pipeline.plan_parts(plan or {}):
        # **A character's landmark is not the district's fabric.** The closure round's
        # held-out hamlet: the smithy the character declared as its landmark was
        # reconciled against the `fabric` want and refused as "workshop is for urban
        # work"; a landmark is the character author's own deliberate choice and is
        # recorded under its own kind, where the fabric want does not look.
        kind = p.get("kind", "plot")
        if str(p.get("name") or "").startswith("landmark_"):
            kind = "landmark"
        note(p.get("defines") or p.get("compound") or p.get("answers"), p.get("type"), kind)
    return out


def agreements(rec: dict, place: dict, decls: dict, plan: dict | None = None) -> tuple:
    """`(record, findings)` -- does the laid-out place use the types matching approved?

        **The record follows the place, and a disagreement the record cannot justify is a
        finding.** The review's third finding was that `capabilities.json` was written, then
        layout and compilation chose types of their own, and nothing ever compared the two:
        the round's own report recorded matching `wall` while the layout built `great_wall`.

        Selection is not forced to be identical -- a ring wall is also chosen by the height
        its ring needs, which matching does not know -- so what is enforced is the weaker
        and more useful invariant: **whatever the layout built has to pass the same
        capability test the record applied.** Where it does, the entry is updated to name
        the type that is actually there and says why it differs. Where it does not, the
        layout has used a type the capability rules refuse, and that is an error owned by
        `capability`.

        `plan` is the assembled tree where the run has one. Without it only the place level
        can be compared, which is what the review found: the fabric a district was compiled
        out of and the halls inside a compound were never reconciled with anything, and both
        are where the type choices actually are.
        
    """
    used_by = _types_used(place, plan)
    entries, findings = [], []
    for e in (rec or {}).get("entries") or []:
        e = dict(e)
        want = e.get("wants") or {}
        # only the leaves of the kind this want is for; see `_types_used`
        by_kind = used_by.get(str(want.get("part")), {})
        used = [u for u in by_kind.get(str(want.get("kind") or "plot"), []) if u]
        approved = {e.get("type"), *(e.get("alternatives") or [])} - {None}
        # everything the place built that is not already the entry's own type has to be
        # reconciled, so that the record names what is there. A type matching already
        # ranked as an alternative passed `fits` once and is not re-tested; it is still
        # a *difference* and the record still has to follow it.
        strangers = [u for u in used if u != e.get("type")]
        if not strangers:
            if used:
                e = {**e, "used": used}
            entries.append(e)
            continue
        good, bad = [], []
        for u in strangers:
            decl = _declaration(u, decls)
            if u in approved:
                good.append((u, "", decl))
                continue
            ok, why = ((False, f"{u} is not a type this checkout can load")
                       if decl is None else fits(card(u, decl), want))
            (good if ok else bad).append((u, why, decl))
        if not bad:
            first, _why, decl = good[0]
            c = card(first, decl)
            e.update({"type": first, "matched": True, "status": "covered",
                      "used": used, "form": c["form"], "role": c["role"],
                      "envelope": c["envelope"], "checked": c["declares_needs"],
                      "alternatives": sorted({*(e.get("alternatives") or []),
                                              *(u for u, _w, _d in good),
                                              *([e["type"]] if e.get("type") else [])}),
                      "why": (f"the place is built out of {', '.join(used)} here, and "
                              f"each answers this want on the same five facts; matching "
                              f"ranked {e.get('type')} first on name and envelope and "
                              f"the layout had a further constraint")})
            entries.append(e)
            continue
        e.update({"matched": False, "status": "disagreed", "used": used,
                  "type": bad[0][0],
                  "why": (f"the place built {want.get('part')} out of "
                          f"{', '.join(u for u, _w, _d in bad)}, which this want "
                          f"refuses: {bad[0][1]}. The capability record and the place "
                          f"disagree about what this part is")})
        entries.append(e)
        findings.append({
            "id": f"find/{e['id']}/disagreed", "requirement": e.get("requirement"),
            "part": want.get("part"), "says": e["why"],
            "evidence": {"approved": sorted(approved), "built": used,
                         "refused": [u for u, _w, _d in bad], "kind": e["kind"],
                         "family": e["family"], "of": want.get("of")},
            "owner": "capability", "blocks": "feasibility", "severity": "error",
            "seen_by": "capability.agreements", "fixed": False})
    return (contracts.make("capabilities", entries=entries, cap=(rec or {}).get("cap"),
                           note=(rec or {}).get("note", "") +
                           "; reconciled against every type the laid-out place uses"),
            findings)


def gaps(spec: dict, decls: dict | None = None, *, names=None) -> list:
    """The **defining parts** with no capability, for `growth`. Compound components and
    district fabric are reported by `match` and are not grown one type at a time."""
    rec = match(spec, decls, names=names)
    by_part = {p["name"]: p for p in spec.get("defining_parts") or []}
    out = []
    for e in uncovered(rec):
        if e["wants"].get("of") != "defining_part":
            continue
        p = by_part.get(e["wants"]["part"])
        if p is not None and p not in out:
            out.append(p)
    return out


def adoptable(path: str, want: dict) -> tuple:
    """May the file already at `path` be adopted as the answer to `want`? `(bool, why)`.

        The audit's `growth.py:183`: a gap was closed by the mere existence of
        `types/<family>.py`, **including the file whose wrong form opened the gap**. A file
        on disk is a candidate here and nothing more; it is loaded, its five facts are read,
        and it is adopted only if they answer the want.
        
    """
    from . import pipeline as _pipeline
    if not os.path.exists(path):
        return (False, f"{path} does not exist")
    try:
        decl = _pipeline.load_type(path)
    except Exception as e:                        # noqa: BLE001 -- reported, not raised
        return (False, f"{os.path.basename(path)} does not load as a type: {e}")
    name = os.path.splitext(os.path.basename(path))[0]
    ok, why = fits(card(name, decl), want)
    return (ok, "" if ok else f"the file already at {os.path.basename(path)} does not "
                              f"answer this gap: {why}")


def table(rec: dict) -> str:
    """The capability table, for a brief and for a report."""
    rows = ["| want | kind | family | type | envelope | status |",
            "|---|---|---|---|---|---|"]
    for e in (rec or {}).get("entries") or []:
        fp = e.get("envelope", {}).get("footprint") if e.get("envelope") else None
        rows.append(f"| `{e['id']}` | {e['kind']} | {e['family']} | "
                    f"{('`' + e['type'] + '`') if e.get('type') else '--'} | "
                    f"{(str(fp[0]) + 'x' + str(fp[1]) + '..' + str(fp[2]) + 'x' + str(fp[3])) if fp else '--'} "
                    f"| {e['status']} |")
    return "\n".join(rows)
