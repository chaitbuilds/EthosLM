"""This is that row, and **it runs no model**. Everything "a walled town with a market square
and a keep" claims is a fact about geometry and about what stands:

The report is a bar: `place_read` in `pipeline.MEASURES`, and it is `1` when every one
of those holds and `0` otherwise, with the failing clause named. A place that misses it
is not broken.
"""
from __future__ import annotations

import os

from . import pipeline, spec as spec_mod

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: A natural barrier only counts as a wall when the sentence asked for one. "On a cliff"
#: is a thing a place can be and "walled" is a thing a place can be, and a cliff
#: standing in for a wall is the substitution this clause exists to refuse.
CLIFF_WORDS = ("on a cliff", "cliff", "escarpment", "crag", "bluff")


def _stood(parts_record: dict) -> dict:
    """{part name: did the type say it stood}, off `parts.json`."""
    out = {}
    for w in (parts_record or {}).get("waves", []):
        for r in w.get("parts", []):
            out[r["part"]] = bool(r.get("stood", r.get("status") == "built"))
    return out


def _decls(parts: list) -> dict:
    from . import place
    return place.type_declarations(parts)


#: What "monumental" is held to.** A compound's rectangle is at least this many times
#: the footprint of the largest single plot leaf standing outside any compound, and the
#: blocks its standing parts laid are at least this many times the blocks of the largest
#: single such plot. Three.
MONUMENT_FOOTPRINT_MARGIN = 3.0
MONUMENT_BLOCKS_MARGIN = 3.0

#: **Registered.** The height a great wall is held to, in blocks, where the spec asks
#: for one: in a place of concentric rings the outermost ring is its great wall. Half
#: again.
GREAT_WALL_HEIGHT = 30


def requirement_clauses(spec: dict, plan: dict, parts_record: dict,
                        intent_rec: dict | None = None,
                        resolution: dict | None = None,
                        site: dict | None = None,
                        reading: dict | None = None,
                        capabilities: dict | None = None,
                        judgment: dict | None = None) -> list:
    """The `asked/*` clauses alone; see `requirement_read` for the findings beside them."""
    return requirement_read(spec, plan, parts_record, intent_rec, resolution, site,
                            reading=reading, capabilities=capabilities,
                            judgment=judgment)[0]


def requirement_read(spec: dict, plan: dict, parts_record: dict,
                     intent_rec: dict | None = None,
                     resolution: dict | None = None,
                     site: dict | None = None,
                     reading: dict | None = None,
                     capabilities: dict | None = None,
                     judgment: dict | None = None) -> tuple:
    """The **sentence's** own requirements, checked against what stands. One clause each.

        Every other clause in this file is generated from the spec, and the architecture
        audit's counterexample is why that is not enough: a sentence asking for a walled
        village whose spec omitted the wall passed every one of them, because a clause
        generated from a document cannot notice what the document left out. These come from
        `ethoslm.intent`, which reads the sentence and has never seen the spec.

        Silent where the sentence states nothing this build reads by rule -- "Build Ba Sing
        Se" states no requirement outright, and a clause that holds vacuously is a clause
        nobody can read.
        
    """
    from . import contracts, intent as intent_mod
    rec = intent_rec or intent_mod.read(spec.get("sentence") or "")
    if not rec.get("requirements"):
        return [], []
    # **Everything the requirements are answerable against, and three of them were
    # missing.** The review's fifth finding, in its second half: this is the place
    # reader that decides whether the finished place meets the sentence, and it supplied
    # neither the sourced reading, nor the capability record, nor the inspection's
    # verdicts. So a named place could never be more than "nothing has been read about
    # this name yet" however much research the run had done and however the inspection
    # had judged it -- an unreachable success route beside an isolated false-pass
    # interface.
    checked, found = intent_mod.coverage(rec, spec, plan=plan,
                                         parts_record=parts_record,
                                         resolution=resolution, site=site,
                                         reading=reading, capabilities=capabilities,
                                         judgment=judgment)
    out = []
    for r in checked["requirements"]:
        if not r["hard"]:
            continue
        # a requirement a construction sample left unbuilt is neither met nor missed by
        # the sample; `limits.sample` names it
        outside = intent_mod.OUTSIDE_SAMPLE in (r.get("evidence") or [])
        out.append({"clause": f"asked/{r['id']}",
                    "holds": r["status"] == "satisfied" or outside,
                    "outside_sample": outside,
                    "says": f"the sentence asks for {r['says']}: {r['status']}"
                            + (f" -- {r['why']}" if r["why"] else ""),
                    "status": r["status"], "phrase": r["phrase"],
                    "owner": r["owner"], "evidence": r["evidence"],
                    # how this requirement's status was established, which the clause
                    # labelling below keeps separate from how much of the place was
                    # built
                    "method": r.get("method") or "plan"})
    return out, list((found or {}).get("findings") or [])


def construction_limits(plan: dict, parts_record: dict, findings: list) -> list:
    """What construction delivered short of what was asked, as recorded constraints.

        Interface I3's reader. `intent.emitted_findings` writes a `find/emitted/<part>`
        finding for every part whose emitted outcome dropped a storey or a feature; the
        fresh checker found sixteen of them on the finished proof and nothing reading them.
        They are not failing clauses -- the `low` clause holds on what stands, and a lost
        lean-to is a constraint on the lot and not a refusal of the place -- so each becomes
        a record here: the part, what was lost, what was requested and emitted, the owner,
        and the constraint `construction.constraint` derives by probing the type
        (`needs.lot_min`).
        
    """
    rows = [f for f in findings or [] if str(f.get("id", "")).startswith("find/emitted/")]
    if not rows:
        return []
    from . import construction
    parts = {p["name"]: p for p in pipeline.plan_parts(plan)}
    decls = _decls(list(parts.values()))
    out = []
    for f in rows:
        part = parts.get(f.get("part")) or {}
        em = (f.get("evidence") or {}).get("emitted") or {}
        got = None
        try:
            got = construction.constraint(part, decls.get(str(part.get("type") or "")),
                                          em, params=part.get("params"),
                                          seed=part.get("seed"))
        except Exception as e:                    # noqa: BLE001 -- recorded, not raised
            got = {"why": f"the constraint could not be probed: {type(e).__name__}: {e}"}
        got = got or {}
        omitted = list(em.get("omitted") or [])
        out.append({"part": f.get("part"), "type": part.get("type"),
                    "what": got.get("what") or (omitted[0] if omitted else "storeys"),
                    "requested": got.get("requested",
                                         (part.get("params") or {}).get("storeys")),
                    "emitted": got.get("emitted", em.get("storeys")),
                    "omitted": omitted, "owner": got.get("owner") or f.get("owner"),
                    "needs": got.get("needs") or {"lot": None, "lot_min": None},
                    "fallback": em.get("fallback"),
                    "why": got.get("why") or f.get("says")})
    return out


#: The words a sentence can use for a ring, and what each one is a statement **about**.
#: `inner`/`outer` are radial and nothing else; `lower`/`upper` are elevation words that
#: a flat place can only mean radially, and a **hill** place means literally.
RING_WORDS = {"inner": ("radial", 0), "innermost": ("radial", 0),
              "outer": ("radial", 1), "outermost": ("radial", 1),
              "lower": ("elevation", 0), "upper": ("elevation", 1),
              "high": ("elevation", 1), "low": ("elevation", 0)}

#: The words a sentence uses for ground a place climbs. Where one of these is in the
#: request, its `lower` and `upper` rings are read as heights; where none is, and the
#: rings differ socially, they are read as ranks. See `ring_words`.
HILL_WORDS = ("hill", "hillside", "slope", "hilltop", "terraced", "on terraces",
              "mountain", "mountainside", "valley", "escarpment", "cliff")

#: How much the terraces of a place have to differ before its `lower`/`upper` words are
#: read as elevation at all. Below this every ring stands at the same height and the
#: words can only be radial, so there is nothing to check.
RING_RELIEF = 3


def ring_words(spec: dict, layout: dict | None) -> list:
    """**A ring's rank, its elevation and the word the sentence used, kept apart.**

        The expression round put a Japanese hill town's *dense lower ring* on the terrace
        four blocks **above** its sparse upper ring, and nothing noticed, because `lower`
        and `upper` had been taken as names for ring 0 and ring 1. They are not names. On a
        hill they are statements about height, and a place whose rings differ in elevation
        can be asked whether it honoured them.

        Returns one row per ring word the spec's part names carry:
        `{"part", "word", "means", "ring", "level", "agrees", "why"}`. `agrees` is None
        where the word is radial or the place is flat -- neither met nor missed.
        
    """
    rings = [r for r in ((layout or {}).get("rings") or []) if r.get("name")]
    if len(rings) < 2:
        return []
    # **Social rank is the third thing, and it is not elevation either.** The round's
    # own instruction: keep radial order, social rank and terrain elevation distinct. a
    # Lower Ring of the poor and an Upper Ring of the elite -- and reading `upper` there
    # as a statement about height asks a city to put its rich quarter on a hill because
    # of a word. Where the rings' own record says the order is social (they differ in
    # density, role or voice, which is what a class ranking is in this build), the word
    # is a rank and this clause has nothing to hold it to. A hill town whose rings
    # differ only in where they stand is the case that remains. ...and what decides
    # between the two readings is the **sentence**, not a rule about rings. `a small
    # Japanese HILL town ... a dense lower ring and a sparse upper ring` is about a
    # slope. A sentence that names the ground the place climbs means its words
    # literally; one that does not, and whose rings differ in density, role and voice,
    # is ranking them.
    said = str((spec or {}).get("sentence") or "").lower()
    on_a_slope = any(w in said for w in HILL_WORDS)
    social = [tuple(sorted(((r.get("order") or {}).get("social") or {}).items()))
              for r in rings]
    if not on_a_slope and len([x for x in social if x]) == len(rings) \
            and len(set(social)) == len(rings):
        return [{"part": r["name"],
                 "word": next((w for w in RING_WORDS
                               if w in str(r["name"]).lower().split("_")), None),
                 "means": "social", "ring": r.get("ring"), "level": r.get("level"),
                 "agrees": None,
                 "why": ("this place's rings are ranked socially -- they differ in "
                         "density, role and voice -- so `upper` and `lower` are rank "
                         "words here and not statements about height")}
                for r in rings
                if any(w in str(r["name"]).lower().split("_") for w in RING_WORDS)]
    levels = [r.get("level") for r in rings if isinstance(r.get("level"), (int, float))]
    relief = (max(levels) - min(levels)) if len(levels) == len(rings) else 0
    by_rank = sorted(rings, key=lambda r: int(r.get("ring") or 0))
    by_level = sorted(rings, key=lambda r: float(r.get("level") or 0))
    out = []
    for r in rings:
        name = str(r.get("name") or "").lower()
        word = next((w for w in RING_WORDS if w in name.split("_")), None)
        if word is None:
            continue
        means, end = RING_WORDS[word]
        if means == "radial" or relief < RING_RELIEF:
            out.append({"part": r["name"], "word": word, "means": means,
                        "ring": r.get("ring"), "level": r.get("level"),
                        "agrees": None,
                        "why": (f"`{word}` is a radial word" if means == "radial" else
                                f"the rings stand within {relief} of one another, so "
                                f"`{word}` can only be read as radial order here")})
            continue
        want = by_level[-1] if end else by_level[0]
        ok = want["name"] == r["name"]
        out.append({"part": r["name"], "word": word, "means": "elevation",
                    "ring": r.get("ring"), "level": r.get("level"),
                    "agrees": bool(ok),
                    "why": (f"`{word}` on a place whose rings differ by {relief} is a "
                            f"statement about height: "
                            + (f"{r['name']} stands at {r.get('level')}, which is the "
                               f"{'highest' if end else 'lowest'} of them"
                               if ok else
                               f"{r['name']} stands at {r.get('level')} and the "
                               f"{'highest' if end else 'lowest'} ring is "
                               f"{want['name']} at {want.get('level')}; the ring rank "
                               f"was substituted for the elevation"))})
    return out


def read(spec: dict, plan: dict, parts_record: dict, *, voice: str | None = None,
         site: dict | None = None, built=None, base=None,
         plateau: dict | None = None, intent_rec: dict | None = None,
         resolution: dict | None = None, reading: dict | None = None,
         capabilities: dict | None = None, judgment: dict | None = None,
         layout: dict | None = None) -> dict:
    """The whole read. No model, no server: the plan, what stood, and -- when the
    built world and the ground it was built on are given -- what it is made of.

    `built` and `base` are `observe.Volume`s: the place as the waves left it and the
    cached ground before the first part was laid. With both, the `palette/built` clause
    is asked (A2 of the voice contract); without them it is not, and a stage that has a
    built world always gives it."""
    from . import intent as intent_mod
    parts = pipeline.plan_parts(plan)
    by_name = {p["name"]: p for p in parts}
    decls = _decls(parts)
    stood = _stood(parts_record)
    links = group_links(plan)
    clauses = []
    # **A sample qualifies its constructed scope.** Where the parts record carries a
    # `sample`, every clause that counts standing leaves is measured over the sampled
    # leaves -- planned-in-sample against standing-in-sample -- and says so; clauses
    # about the plan (relations, hierarchy, layout, density) stay whole-place. The
    # transfer town's read judged 116 leaves by the 57 its sample built and failed the
    # whole town for the quarters it never attempted.
    scope = intent_mod.sample_scope(parts_record, parts)
    sparts = [p for p in parts if p["name"] in scope] if scope is not None else parts
    tag = f" (sample of {len(scope)} of {len(parts)} leaves)" if scope is not None else ""

    #: **What a clause was read off**, independent of how much of the place was built.
    #: See `intent.METHODS`. The default is `plan`, because the great majority of the
    #: clauses below are measured on the planned geometry, and a clause that reads the
    #: built world or the construction record says so by passing `method="observed"`.
    #: The expression round had no such field and promoted every clause to `built_place`
    #: the moment the last leaf stood.
    def clause(name, ok, says, method="plan", **more):
        clauses.append({"clause": name, "holds": bool(ok), "says": says,
                        "method": method, **more})

    def scoped(name, ok, says, method="plan", **more):
        clauses.append({"clause": name, "holds": bool(ok), "says": says + tag,
                        "method": method,
                        **({"sampled": True} if scope is not None else {}), **more})

    def not_sampled(name, says, **more):
        clauses.append({"clause": name, "holds": True,
                        "says": f"{says}: no leaf of it is in the construction sample, "
                                f"so it is neither met nor missed here{tag}",
                        "method": "unsupported",
                        "sampled": False, "outside_sample": True, **more})

    # --- present ---------------------------------------------------------
    for d in spec["defining_parts"]:
        if spec_mod.compound(d):
            # A great thing is present when what it is made of stands: its wall closed
            # with a gate on it, its halls and its court inside. See `compound_clauses`.
            continue
        if d["kind"] == "group":
            # A group defining part is districts, and a district is not built: what it
            # is answerable for is that **its own** quarters have plots in them, and
            # that those plots stand. **Its own, and standing.** The architecture
            # audit's second finding, with a probe that reproduced it: this counted the
            # *global* set of quarters holding planned plots, so a spec declaring
            # `homes` and `missing_district` passed both presence clauses on a plan
            # whose only plot was in `homes_row` -- and the plot did not have to stand.
            # A district nobody drew was present because a different district was, which
            # is the clearest possible case of a clause generated from the plan rather
            # than asked of it.
            linked = bool(links) or any(
                a == d["name"] or a.startswith(d["name"] + "_")
                for p in parts for a in (p.get("in") or []))
            # **A land district is present when its ground stands.** The expression
            # round's held-out village: the pasture, a district of no houses, holds only
            # area leaves (grazing strips, groves), and a clause counting plots read it
            # as absent. A district asked for no structures is answerable for its areas;
            # one asked for houses is answerable for its plots, as before.
            land = int(d.get("structures") or 0) == 0
            kinds = ("area", "plot") if land else ("plot",)
            mine, up = set(), set()
            for p in sparts:
                if p.get("kind", "plot") not in kinds or not p.get("in"):
                    continue
                if linked and not _in_group(p, d, links):
                    continue
                mine.add(p["in"][-1])
                if stood.get(p["name"], False):
                    up.add(p["in"][-1])
            if scope is not None and not mine:
                not_sampled(f"present/{d['name']}",
                            f"the spec asks for {d['count']} {d['family']}(s) called "
                            f"{d['name']}", wanted=d["count"], got=[], standing=[])
                continue
            want = min(d["count"], len(mine)) if scope is not None else d["count"]
            scoped(f"present/{d['name']}", len(up) >= want,
                   f"the spec asks for {d['count']} {d['family']}(s) called "
                   f"{d['name']} and the plan has {len(mine)} quarter(s) "
                   + ("of it " if linked else "(this plan carries no link from a "
                                              "quarter to the part it answers, so "
                                              "every quarter counts) ")
                   + f"with plots in them, {len(up)} of which have a plot standing",
                   method="observed",
                   wanted=want, got=sorted(mine), standing=sorted(up),
                   linked=linked)
            continue
        mine = _matching(sparts, d, decls)
        if scope is not None and not mine:
            not_sampled(f"present/{d['name']}",
                        f"the spec asks for {d['count']} x {d['family']} as a "
                        f"{d['kind']} ({d['relation']})", wanted=d["count"],
                        planned=[], stood=[])
            continue
        up = [p["name"] for p in mine if stood.get(p["name"], False)]
        want = min(d["count"], len(mine)) if scope is not None else d["count"]
        scoped(f"present/{d['name']}", len(up) >= want,
               f"the spec asks for {d['count']} x {d['family']} as a {d['kind']} "
               f"({d['relation']}); the plan has {len(mine)} and {len(up)} of them "
               f"stand",
               method="observed",
               wanted=want, planned=[p["name"] for p in mine], stood=up)

    # --- closed ----------------------------------------------------------
    asked_cliff = any(w in spec["sentence"].lower() for w in CLIFF_WORDS)
    for d in spec_mod.walls(spec):
        mine = [p for p in _matching(parts, d, decls) if p.get("kind") == "edge"]
        if scope is not None and mine and not any(p["name"] in scope for p in mine):
            not_sampled(f"closed/{d['name']}", "the spec calls this a wall")
            continue
        mine = [p for p in mine if scope is None or p["name"] in scope]
        if not mine:
            clause(f"closed/{d['name']}", False,
                   "the spec calls this a wall and there is no edge part for it"
                   + (" -- and the sentence does not say the place is on a cliff, so a "
                      "natural barrier does not stand in for one" if not asked_cliff
                      else ""))
            continue
        for p in mine:
            path = [(int(a[0]), int(a[1])) for a in (p.get("path") or [])]
            closed = len(path) >= 4 and path[0] == path[-1]
            up = stood.get(p["name"], False)
            gates = _gates_on(p, sparts, decls)
            standing_gates = [g for g in gates if stood.get(g, False)]
            ok = closed and up and bool(standing_gates)
            scoped(f"closed/{p['name']}", ok,
                   f"{len(path)} vertices, "
                   + ("a closed loop" if closed else "NOT a closed loop: it starts at "
                      f"{list(path[0]) if path else None} and ends at "
                      f"{list(path[-1]) if path else None}")
                   + f"; the wall {'stands' if up else 'DOES NOT STAND'}; "
                   + (f"{len(standing_gates)} of {len(gates)} gate(s) on it stand"
                      if gates else "no gate stands on it"),
                   closed=closed, stood=up, gates=gates,
                   gates_standing=standing_gates)

    # --- compounds -------------------------------------------------------
    comp = compound_clauses(spec, plan, sparts, decls, stood, parts_record,
                            plateau=plateau)
    if scope is not None:
        sampled_comp = {d["name"] for d in spec_mod.compounds(spec)
                        if any(p.get("compound") and (
                            p.get("defines") == d["name"] or p.get("compound") == d["name"]
                            or str(p.get("compound", "")).startswith(d["name"]))
                               for p in sparts)}
        for c in comp:
            name = c["clause"].split("/")[1]
            if name not in sampled_comp:
                c.update(holds=True, sampled=False, outside_sample=True,
                         says=f"{name}: no leaf of it is in the construction sample, "
                              f"so it is neither met nor missed here{tag}")
            else:
                c["says"] += tag
                c["sampled"] = True
    clauses += comp

    # --- concentric ------------------------------------------------------
    clauses += concentric_clauses(spec, plan, parts, decls, stood)
    for c in great_wall_clauses(spec, plan, sparts, decls, stood):
        if scope is not None:
            c["says"] += tag
            c["sampled"] = True
        clauses.append(c)

    # --- count ----------------------------------------------------------- **A count is
    # a count of the thing the sentence counted.** The closure round's proof: sixteen
    # cottages and a hall stood, the sentence's own clause held at 16, and this clause
    # failed at 17 because it counted every standing plot. Where the spec carries an
    # explicit count with a subject, `intent.select` -- the one rule for which leaves a
    # word names -- picks the plots counted; a spec with no explicit subject counts
    # every plot, as before.
    from . import intent as intent_mod
    what = str(((spec.get("explicit_count") or {}) if isinstance(
        spec.get("explicit_count"), dict) else {}).get("what") or "")
    if what and intent_mod._slug(what) in ("building", "buildings", "structure",
                                            "structures"):
        what = ""
    plots = [p for p in parts if p.get("kind", "plot") == "plot"]
    counted = ([p for p in intent_mod.select(parts, what)
                if p.get("kind", "plot") == "plot"] if what else plots)
    lo, hi = spec["size_band"]
    if scope is not None:
        in_s = [p for p in counted if p["name"] in scope]
        up = [p["name"] for p in in_s if stood.get(p["name"], False)]
        scoped("count", len(up) == len(in_s) and spec_mod.in_band(spec, len(counted)),
               f"{len(up)} of {len(in_s)} sampled {what or 'structure(s)'} stand; "
               f"{len(counted)} planned in the whole place"
               + (f" ({len(plots)} plot(s) in all)" if what else "")
               + f", against the band {lo}-{hi} the spec produced",
               method="observed",
               stood=len(up), planned=len(in_s), planned_whole=len(counted),
               band=[lo, hi], subject=what or "every plot")
    else:
        up = [p["name"] for p in counted if stood.get(p["name"], False)]
        clause("count", spec_mod.in_band(spec, len(up)),
               f"{len(up)} {what or 'structure(s)'} stand of {len(counted)} planned"
               + (f" ({len(plots)} plot(s) in all)" if what else "")
               + f", against the band {lo}-{hi} the spec produced",
               method="observed", stood=len(up), planned=len(counted),
               band=[lo, hi], subject=what or "every plot")

    # --- ring words ------------------------------------------------------ **A rank is
    # not a height.** Only asked where the sentence's own rings carry an elevation word
    # and the place's terraces actually differ; a flat place's `lower` ring is a radial
    # statement and there is nothing here to hold it to.
    words = [w for w in ring_words(spec, layout) if w["agrees"] is not None]
    if words:
        clause("rings/elevation", all(w["agrees"] for w in words),
               "; ".join(w["why"] for w in words), method="observed",
               rings=words,
               failed=[w["part"] for w in words if not w["agrees"]])

    # --- palette. One clause: it holds when every voice standing in the place holds.
    chosen = voice or plan.get("voice") or spec.get("voice")
    voices = sorted({p.get("voice") or chosen for p in parts if (p.get("voice") or chosen)},
                    key=lambda v: (v != chosen, v))
    reads = [(v, *_palette(v, site)) for v in (voices or [chosen])]
    clause("palette", all(ok for _v, ok, _s in reads),
           "; ".join(says for _v, _ok, says in reads),
           voices=[v for v, _ok, _s in reads],
           failed=[v for v, ok, _s in reads if not ok])
    # --- palette, as built ------------------------------------------------
    if built is not None:
        got_pal = built_palette(chosen, sparts, stood, built, base)
        if scope is not None:
            got_pal["says"] += tag
            got_pal["sampled"] = True
        clause("palette/built", method="observed", **got_pal)

    # --- what the sentence asked for -------------------------------------- Last,
    # because it is the only group of clauses that does not come from the spec, and a
    # reader should see the place checked against its own plan and then against the
    # request that produced it.
    asked, found = requirement_read(spec, plan, parts_record, intent_rec, resolution,
                                    site, reading=reading, capabilities=capabilities,
                                    judgment=judgment)
    clauses += asked

    # **Every clause says what evidence it rests on.** The expression round; the closure
    # review's fourth finding. A clause measured over the plan while a sample was built,
    # a clause measured on the sampled leaves, a clause on a place built whole, and a
    # clause whose subject was never built are four different statements, and they
    # collapsed into one `holds` flag. An outside-sample clause is `unobserved` and its
    # `holds` is None -- neither met nor missed -- and it never makes a read hold.
    # **...and standing leaves do not promote a plan measurement.** The design round;
    # the expression review's third finding. Coverage and method are two independent
    # questions, and `built_place` is the conjunction of both: the whole place stood
    # *and* this clause was read off what stands. A relation measured on the drawing is
    # `plan` evidence on a village built to the last thatch.
    every_leaf = [p for p in parts if p.get("kind", "plot") in ("plot", "edge", "point",
                                                                  "area")]
    all_stood = bool(every_leaf) and all(stood.get(p["name"], False) for p in every_leaf)
    for c in clauses:
        method = c.get("method") or "plan"
        if c.get("outside_sample"):
            c["evidence"] = "unobserved"
            c["holds"] = None
        elif method in ("plan", "declared"):
            c["evidence"] = "plan"
        elif method == "unsupported":
            c["evidence"] = "unobserved"
        # `observed`, `judged` and `site` are all readings of something that exists --
        # the built world, the inspection's view of it, the ground it stands on -- so
        # they take the coverage the place actually has.
        elif scope is not None:
            c["evidence"] = "built_sample" if c.get("sampled") else "plan"
        else:
            c["evidence"] = "built_place" if all_stood else "plan"
    holds = all(c["holds"] for c in clauses if c["holds"] is not None)
    # **What this read cannot say, said.** Unresolved and unsupported obligations are
    # not failures of the place and they are not passes either; a sample built of a
    # larger plan qualifies the sample. Both travel with the verdict so a reader of a
    # `holds: true` sees what it is a verdict about.
    limits = {"unobserved": [c["clause"] for c in clauses
                             if c.get("evidence") == "unobserved"],
              "unresolved": [c["clause"] for c in clauses
                             if c.get("status") == "unresolved"],
              "unsupported": [c["clause"] for c in clauses
                              if c.get("status") == "unsupported"],
              "sample": (parts_record or {}).get("sample"),
              # what construction delivered short of the ask: recorded constraints, not
              # failing clauses (interface I3's reader)
              "construction": construction_limits(plan, parts_record, found)}
    if limits["sample"]:
        unbuilt = [p["name"] for p in parts if p["name"] not in (scope or set())]
        limits["sample"] = dict(limits["sample"],
                                not_built={"leaves": len(unbuilt),
                                           "of": len(parts),
                                           "quarters": sorted({(p.get("in") or ["?"])[-1]
                                                               for p in parts
                                                               if p["name"] not in
                                                               (scope or set())
                                                               and p.get("in")}),
                                           "examples": unbuilt[:8]},
                                outside_sample=[c["clause"] for c in clauses
                                                if c.get("outside_sample")])
        limits["note"] = (f"a construction sample of {len(scope or [])} of {len(parts)} "
                          f"leaves: this read qualifies the sample and not the whole "
                          f"place; {len(unbuilt)} leaf/leaves were not built")
    return {"holds": bool(holds), "got": 1 if holds else 0,
            "clauses": clauses,
            "failed": [c["clause"] for c in clauses if c["holds"] is False],
            "evidence": {lvl: sum(1 for c in clauses if c.get("evidence") == lvl)
                         for lvl in ("built_place", "built_sample", "plan", "unobserved")},
            # **the other census**: what the clauses were read off, which coverage
            # cannot tell you. A read of 17 clauses at `built_place` over a whole built
            # village and a read of 17 clauses of which 4 were observed are different
            # statements, and the second is the true one for most places.
            "method": {m: sum(1 for c in clauses if (c.get("method") or "plan") == m)
                       for m in ("observed", "plan", "declared", "judged", "site",
                                 "unsupported")},
            "limits": limits,
            "sentence": spec["sentence"], "kind": spec["kind"],
            "band": list(spec["size_band"]),
            "note": "The built place against the sentence's own spec, "
                    "with no model call anywhere in it"}


# ------------------------------------------------------------- A2, concentric A place
# of one wall needed one question asked of it: is the wall a closed loop with a gate on
# it. A place of three needs five more, because "three concentric ring walls" is a
# statement about how they sit relative to each other and to everything inside them, and
# every one of those relations is a fact about geometry that no model call has to be
# spent on. A plan that drew three closed loops side by side would pass every clause the
# place read had before this one. It also closes open thread 17, which is the same
# defect a size smaller: nothing in this project has ever checked that a town is
# **inside** its own wall. they were vanilla worldgen, and no instrument could say so.
# `inside()` is the one point-in-polygon test that thread says closes it, and every plot
# is now put through it.

#: How near a gate an arterial may cross a ring and still be crossing it at the gate.
#: `placeplan.arterial_failures` uses two either way at plan time and this is the same
#: number, read against the built place rather than the drawn one.
GATE_SLACK = 2


def inside(path: list, point: tuple) -> bool:
    """Is `point` inside the closed polyline `path`? Ray cast, half-open, no library.

        Written here in nine lines rather than pulled in, because it has to agree with
        `_edge_cells` about what a wall's line is and because a dependency for a crossing
        count is a dependency for nothing. A point **on** the line is not inside it: a gate
        stands in its wall and is answered by the clause that asks about gates.
        
    """
    pts = [(float(a[0]), float(a[1])) for a in path]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return False
    x, z = float(point[0]), float(point[1])
    hit = False
    for (ax, az), (bx, bz) in zip(pts, pts[1:] + pts[:1]):
        if (az > z) != (bz > z):
            cut = ax + (z - az) * (bx - ax) / (bz - az)
            if cut > x:
                hit = not hit
    return hit


def ring_area(path: list) -> float:
    """The shoelace area of a closed polyline. What orders rings outermost first."""
    pts = [(float(a[0]), float(a[1])) for a in path]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0
    return abs(sum(ax * bz - bx * az
                   for (ax, az), (bx, bz) in zip(pts, pts[1:] + pts[:1]))) / 2.0


def rings(spec: dict, parts: list) -> list:
    """The concentric ring walls of a plan, outermost first. A2."""
    out = []
    for d in spec["defining_parts"]:
        if d["relation"] != "concentric" or d["kind"] != "edge":
            continue
        for p in _matching(parts, d):
            path = [(int(a[0]), int(a[1])) for a in (p.get("path") or [])]
            out.append({"name": p["name"], "defines": d["name"], "path": path,
                        "part": p, "area": ring_area(path)})
    return sorted(out, key=lambda r: (-r["area"], r["name"]))


def concentric_clauses(spec: dict, plan: dict, parts: list, decls: dict,
                       stood: dict) -> list:
    """The five things "concentric" claims, each as its own clause. A2.

        Silent where the spec asks for no concentric part: a hamlet round a green is not a
        place this has anything to say about, and a clause that holds vacuously on every
        place that is not a city is a clause nobody can read.
        
    """
    from .placeplan import _edge_cells
    wanted = [d for d in spec["defining_parts"] if d["relation"] == "concentric"]
    if not wanted:
        return []
    got = rings(spec, parts)
    out = []

    def clause(name, ok, says, **more):
        out.append({"clause": name, "holds": bool(ok), "says": says, **more})

    # 1. the rings are there, they are closed, and each lies inside the one outside it.
    n_want = sum(d["count"] for d in wanted if d["kind"] == "edge")
    open_rings = [r["name"] for r in got
                  if len(r["path"]) < 4 or r["path"][0] != r["path"][-1]]
    nested, why = [], []
    for outer, inner in zip(got, got[1:]):
        bad = [v for v in inner["path"] if not inside(outer["path"], v)]
        nested.append(not bad)
        if bad:
            why.append(f"{inner['name']} has {len(bad)} vertex/vertices outside "
                       f"{outer['name']}, the first at {list(bad[0])}")
    ok = (len(got) >= n_want and n_want >= 1 and not open_rings and all(nested))
    clause("concentric/nested", ok,
           f"{len(got)} ring wall(s) of {n_want} asked for, outermost first: "
           + ", ".join(f"{r['name']} (area {int(r['area'])})" for r in got)
           + ("; every ring lies inside the one outside it"
              if got and all(nested) and not open_rings else
              "; " + "; ".join(why + [f"{n} is not a closed loop"
                                      for n in open_rings]) or "; no ring is planned"),
           wanted=n_want, got=[r["name"] for r in got], open_rings=open_rings,
           areas=[int(r["area"]) for r in got])
    if not got:
        return out

    art = {(int(x), int(z)) for x, z in
           ((plan.get("arterials") or {}).get("cells") or [])}
    gates = [p for p in parts if p.get("kind") == "point"
             and (decls.get(p.get("type")) or {}).get("passage")]
    gate_at = {g["name"]: (int((g.get("at") or [0, 0])[0]),
                           int((g.get("at") or [0, 0])[-1])) for g in gates}
    near_gate = {(gx + i, gz + j) for (gx, gz) in gate_at.values()
                 for i in range(-GATE_SLACK, GATE_SLACK + 1)
                 for j in range(-GATE_SLACK, GATE_SLACK + 1)}

    # 2. every ring has a gate on it, and every gate is on the road.
    per_ring, bad_gate = {}, []
    for r in got:
        cells = set(_edge_cells(r["part"]))
        mine = [n for n, c in gate_at.items() if c in cells]
        per_ring[r["name"]] = mine
        if not mine:
            bad_gate.append(f"{r['name']} has no gate on it")
    # ...and every gate **on a ring** is on the road. A compound's gate stands on the
    # compound's own wall, on ground the arterial arrives at and does not cross, and is
    # judged by the compound's clause.
    on_rings = {n for mine in per_ring.values() for n in mine}
    off_road = ([n for n, c in gate_at.items()
                 if n in on_rings and not ({(c[0] + i, c[1] + j)
                                            for i in range(-GATE_SLACK, GATE_SLACK + 1)
                                            for j in range(-GATE_SLACK, GATE_SLACK + 1)}
                                           & art)]
                if art else [])
    clause("concentric/gates", not bad_gate and not off_road,
           "; ".join(f"{k}: {v or 'no gate'}" for k, v in per_ring.items())
           + ("; every gate stands on an arterial" if art and not off_road
              else f"; {off_road} stand on no arterial" if off_road
              else "; no arterial was routed, so no gate is checked against one"),
           per_ring=per_ring, off_arterial=off_road)

    # 3. an arterial crosses a ring at a gate or not at all.
    crossings = {}
    for r in got:
        hit = sorted((art & set(_edge_cells(r["part"]))) - near_gate)
        crossings[r["name"]] = [list(c) for c in hit[:6]]
    stray = {k: v for k, v in crossings.items() if v}
    clause("concentric/crossings", not stray,
           "every arterial crosses every ring at a gate" if not stray else
           "; ".join(f"an arterial crosses {k} away from any gate at {v}"
                     for k, v in stray.items()),
           crossings=crossings)

    # 4. what the place is centred on is inside the innermost ring.
    innermost = got[-1]
    centre = [d for d in spec["defining_parts"] if d["relation"] == "centre"]
    named, outside = [], []
    for d in centre:
        # **With the declarations**, so a defining part the spec called a plot and the
        # plan built as an area is still found. `present/` passes them and this did not,
        # so the palace stood inside the innermost ring and the clause reported that the
        # spec centres this place on nothing.
        for p in _matching(parts, d, decls):
            r = pipeline.part_rect({**p, "name": p.get("name")})
            corners = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]
            named.append(p["name"])
            if not all(inside(innermost["path"], c) for c in corners):
                outside.append(p["name"])
    clause("concentric/centre", bool(named) and not outside,
           f"{named or 'nothing'} is what the spec centres this place on; "
           + (f"it lies inside {innermost['name']}, the innermost ring"
              if named and not outside else
              f"{outside} is not wholly inside {innermost['name']}" if outside else
              "the spec names no part at the centre, so nothing is checked against the "
              "innermost ring"),
           innermost=innermost["name"], centre=named, outside=outside)

    # 5. every structure is inside the outermost ring, and a quarter is in one ring.
    # Open thread 17: one point-in-polygon test per plot, which nothing had.
    outermost = got[0]
    plots = [p for p in parts if p.get("kind", "plot") == "plot"]
    bands = {r["name"]: set() for r in got}
    beyond, straddle = [], []
    per_quarter: dict = {}
    for p in plots:
        r = pipeline.part_rect({**p, "name": p.get("name")})
        cx, cz = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
        if not inside(outermost["path"], (cx, cz)):
            beyond.append(p["name"])
            continue
        band = next((q["name"] for q in reversed(got)
                     if inside(q["path"], (cx, cz))), outermost["name"])
        bands[band].add(p["name"])
        q = (p.get("in") or ["-"])[-1]
        per_quarter.setdefault(q, set()).add(band)
    straddle = sorted(q for q, b in per_quarter.items() if len(b) > 1)
    clause("concentric/districts", not beyond and not straddle,
           f"{len(plots) - len(beyond)} of {len(plots)} structures stand inside "
           f"{outermost['name']}"
           + (f"; {len(beyond)} are outside it: {beyond[:6]}" if beyond else "")
           + "; " + ", ".join(f"{k}: {len(v)}" for k, v in bands.items())
           + (f"; {len(straddle)} quarter(s) span more than one ring: {straddle[:4]}"
              if straddle else "; no quarter spans two rings"),
           outside_the_walls=beyond, straddling=straddle,
           per_ring={k: len(v) for k, v in bands.items()},
           quarters={k: sorted(v) for k, v in per_quarter.items()})
    return out


def _blocks(parts_record: dict) -> dict:
    """{part name: blocks its program laid}, off `parts.json`."""
    out = {}
    for w in (parts_record or {}).get("waves", []):
        for r in w.get("parts", []):
            out[r["part"]] = int(r.get("blocks") or 0)
    return out


def _grounds(parts_record: dict) -> dict:
    """{part name: how the library sited it -- plinth, platform, deck, footing}."""
    out = {}
    for w in (parts_record or {}).get("waves", []):
        for r in w.get("parts", []):
            out[r["part"]] = r.get("ground")
    return out


def compound_clauses(spec: dict, plan: dict, parts: list, decls: dict, stood: dict,
                     parts_record: dict, *, plateau: dict | None = None) -> list:
    """What "a great thing" claims, each as its own clause.

        Silent where the spec has no compound. Three clauses per compound:

          present/<name>   the composition stands -- **its family's** (v2, C0;
                           `placeplan.compound_composition`): a closed wall of its own with
                           a standing gate on it where the family is walled and gated, and
                           at least the family's standing plots and areas inside the wall
                           (inside the rectangle, where there is none);
          compound/<name>/scale
                           it is materially larger than any single building outside it,
                           by the registered margins on footprint and on blocks laid;
          compound/<name>/ground
                           its parts stand on prepared ground and not on water, and where
                           the site search levelled ground for it, on that ground.
        
    """
    from .placeplan import (COMPOUND_PLATEAU_SHARE, _edge_cells, compound_composition,
                            compound_rects)
    out = []

    def clause(name, ok, says, **more):
        out.append({"clause": name, "holds": bool(ok), "says": says, **more})

    rects = compound_rects(plan)
    blocks = _blocks(parts_record)
    grounds = _grounds(parts_record)
    # The largest single building outside any compound, by plot and by blocks: the thing
    # a great thing has to be greater than.
    outside = [p for p in parts if p.get("kind", "plot") == "plot"
               and not p.get("compound") and stood.get(p["name"], False)]
    big_fp, big_fp_name = 0, None
    big_bl, big_bl_name = 0, None
    for p in outside:
        r = pipeline.part_rect(p)
        fp = (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
        if fp > big_fp:
            big_fp, big_fp_name = fp, p["name"]
        if blocks.get(p["name"], 0) > big_bl:
            big_bl, big_bl_name = blocks[p["name"]], p["name"]

    for d in spec_mod.compounds(spec):
        mine = [p for p in parts if p.get("compound") and (
            p.get("defines") == d["name"] or p.get("compound") == d["name"]
            or str(p.get("compound", "")).startswith(d["name"]))]
        names = sorted({p["compound"] for p in mine})
        if not mine:
            clause(f"present/{d['name']}", False,
                   f"the spec asks for {d['count']} x {d['family']} as a compound "
                   f"({d['relation']}) and the plan has no compound answering it",
                   wanted=d["count"], planned=[], stood=[])
            continue
        # 1. the composition stands
        walls = [p for p in mine if p.get("kind") == "edge"]
        closed = []
        for w in walls:
            path = [(int(a[0]), int(a[1])) for a in (w.get("path") or [])]
            if len(path) >= 4 and path[0] == path[-1] and stood.get(w["name"], False):
                closed.append((w, path))
        gates = [p for p in mine if p.get("kind") == "point"
                 and (decls.get(p.get("type")) or {}).get("passage")
                 and stood.get(p["name"], False)]
        gate_on = [g["name"] for g in gates
                   if any((int(g["at"][0]), int(g["at"][-1])) in set(_edge_cells(w))
                          for w, _ in closed)]
        made_of = compound_composition(d, spec=spec)
        mine_rects = [rects[n] for n in names if n in rects]

        def within(p):
            # inside a standing closed wall where the family is walled (or one is
            # drawn); inside the compound's own rectangle where none is
            if closed:
                return any(all(inside(path, c) for c in _corners(p)) for _w, path in closed)
            return any(all(r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]
                           for c in _corners(p)) for r in mine_rects)

        halls = [p for p in mine if p.get("kind", "plot") == "plot"
                 and stood.get(p["name"], False) and within(p)]
        courts = [p for p in mine if p.get("kind") == "area"
                  and stood.get(p["name"], False) and within(p)]
        up = [p["name"] for p in mine if stood.get(p["name"], False)]
        ok = (bool(closed) or not made_of["walled"]) \
            and (bool(gate_on) or not made_of["gated"]) \
            and len(halls) >= made_of["halls"] and len(courts) >= made_of["courts"]
        clause(f"present/{d['name']}", ok,
               f"the spec asks for {d['count']} x {d['family']} as a compound "
               f"({d['relation']}); {'/'.join(names)} is {len(mine)} part(s) of which "
               f"{len(up)} stand: "
               + (f"{len(closed)} closed wall(s) standing" if closed
                  else ("NO closed wall stands" if made_of["walled"]
                        else "no wall, and a " + d["family"] + " asks none"))
               + (f", gate(s) {gate_on} on it" if gate_on
                  else (", NO standing gate on it" if made_of["gated"] else ""))
               + f", {len(halls)} hall(s) and {len(courts)} court(s) standing inside "
                 f"{'it' if closed else 'its rectangle'} against {made_of['halls']} "
                 f"and {made_of['courts']}",
               wanted=d["count"], planned=[p["name"] for p in mine], stood=up,
               walls=[w["name"] for w, _ in closed], gates=gate_on,
               halls=[p["name"] for p in halls], courts=[p["name"] for p in courts])
        # 2. the scale
        fp = sum((r[2] - r[0] + 1) * (r[3] - r[1] + 1)
                 for n, r in rects.items() if n in names)
        laid = sum(blocks.get(p["name"], 0) for p in mine if stood.get(p["name"]))
        fp_ok = big_fp == 0 or fp >= MONUMENT_FOOTPRINT_MARGIN * big_fp
        bl_ok = big_bl == 0 or laid >= MONUMENT_BLOCKS_MARGIN * big_bl
        clause(f"compound/{d['name']}/scale", fp_ok and bl_ok,
               f"{'/'.join(names)} covers {fp} columns and laid {laid} blocks; the "
               f"largest single building outside it is {big_fp_name} at {big_fp} "
               f"columns and {big_bl_name} at {big_bl} blocks, so it is "
               f"{(fp / big_fp) if big_fp else 0:.1f}x by footprint and "
               f"{(laid / big_bl) if big_bl else 0:.1f}x by blocks against "
               f"{MONUMENT_FOOTPRINT_MARGIN:g}x and {MONUMENT_BLOCKS_MARGIN:g}x",
               footprint=fp, blocks=laid, largest_footprint=[big_fp_name, big_fp],
               largest_blocks=[big_bl_name, big_bl],
               margins=[MONUMENT_FOOTPRINT_MARGIN, MONUMENT_BLOCKS_MARGIN])
        # 3. the ground
        wet = [p["name"] for p in mine if stood.get(p["name"])
               and grounds.get(p["name"]) == "deck"]
        on = plateau if plateau and plateau.get("rect") \
            and plateau.get("part") in (d["name"], *names) else None
        off, share = [], None
        if on:
            px0, pz0, px1, pz1 = on["rect"]
            area = float((px1 - px0 + 1) * (pz1 - pz0 + 1))
            covered = 0
            for n in names:
                x0, z0, x1, z1 = rects[n]
                if not (px0 <= x0 and x1 <= px1 and pz0 <= z0 and z1 <= pz1):
                    off.append(n)
                covered += (min(x1, px1) - max(x0, px0) + 1) \
                    * (min(z1, pz1) - max(z0, pz0) + 1)
            share = covered / area if area else 0.0
        g_ok = not wet and not off and (share is None or share >= COMPOUND_PLATEAU_SHARE)
        clause(f"compound/{d['name']}/ground", g_ok,
               (f"no standing part of {'/'.join(names)} is on a deck over water"
                if not wet else f"{wet} stand on decks over water")
               + (f"; it stands on the ground levelled for {on['part']} "
                  f"(x {on['rect'][0]}..{on['rect'][2]}, z {on['rect'][1]}..{on['rect'][3]})"
                  f", covering {share:.0%} of it against {COMPOUND_PLATEAU_SHARE:.0%}"
                  if on and not off else
                  f"; {off} are drawn off the ground levelled for {on['part']}"
                  if on else "; no ground was levelled for it"),
               on_water=wet, off_plateau=off, plateau=(on or {}).get("rect"),
               share=round(share, 3) if share is not None else None)
    return out


def _corners(p: dict) -> list:
    r = pipeline.part_rect({**p, "name": p.get("name")})
    return [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]


def great_wall_clauses(spec: dict, plan: dict, parts: list, decls: dict,
                       stood: dict) -> list:
    """The outermost wall of a ringed place, held to **the hierarchy the design adopted**."""
    got = rings(spec, parts)
    if not got:
        return []
    outer = got[0]
    p = outer["part"]
    h = (p.get("params") or {}).get("height")
    up = stood.get(p["name"], False)
    hier = p.get("hierarchy") if isinstance(p.get("hierarchy"), dict) else None
    others = [(r["part"].get("params") or {}).get("height") for r in got[1:]]
    others = [int(v) for v in others if v is not None]
    sentence = str(spec.get("sentence") or "").lower()
    if hier:
        kind = str(hier.get("kind") or "town")
        want = hier.get("height")
        if kind == "great":
            ok = (up and h is not None and (want is None or int(h) >= int(want))
                  and all(int(h) > o for o in others))
            says = (f"{p['name']} is the outermost of {len(got)} rings and the design's "
                    f"great wall (from {hier.get('from')}): planned {h} high against its "
                    f"recorded {want}, the other ring walls {others or 'none'}"
                    + ("; it stands" if up else "; it DOES NOT STAND"))
        else:
            ok = up and h is not None and (want is None or int(h) == int(want))
            says = (f"{p['name']} is the outermost of {len(got)} rings and a {kind} wall "
                    f"by the design (from {hier.get('from')}): planned {h} high against "
                    f"its recorded {want}" + ("; it stands" if up else "; it DOES NOT STAND"))
        return [{"clause": f"great/{p['name']}", "holds": bool(ok), "says": says,
                 "height": h, "against": want, "kind": kind, "type": p.get("type"),
                 "stood": up, "from": hier.get("from")}]
    if "great wall" in sentence:
        ok = up and h is not None and int(h) >= GREAT_WALL_HEIGHT
        return [{"clause": f"great/{p['name']}", "holds": bool(ok),
                 "says": f"{p['name']} is the outermost of {len(got)} rings and the "
                         f"sentence asks for a great wall: planned "
                         f"{h if h is not None else 'no'} high as `{p.get('type')}`, "
                         f"against {GREAT_WALL_HEIGHT} (no hierarchy record on the plan)"
                         + ("; it stands" if up else "; it DOES NOT STAND"),
                 "height": h, "against": GREAT_WALL_HEIGHT, "type": p.get("type"),
                 "stood": up}]
    return [{"clause": f"great/{p['name']}", "holds": bool(up),
             "says": f"{p['name']} is the outermost of {len(got)} rings; the sentence "
                     f"asks for a wall and not a great one, and the plan records no "
                     f"hierarchy, so it is held to standing: planned {h} high"
                     + ("; it stands" if up else "; it DOES NOT STAND"),
             "height": h, "against": None, "type": p.get("type"), "stood": up}]


def group_links(plan: dict) -> dict:
    """`{group node name: the defining part it answers}` off the plan tree."""
    out: dict = {}

    def walk(nodes):
        for n in nodes:
            if n.get("children"):
                if n.get("defines"):
                    out[n.get("name")] = n["defines"]
                walk(n["children"])
    walk(plan.get("parts") or [])
    return out


def _in_group(part: dict, d: dict, links: dict | None = None) -> bool:
    """Is this leaf inside the group defining part `d`?

        By the plan's own `defines` link where it carries one, and otherwise by the name
        the place level gives a district -- `<defining part>_<sector>` -- which is the same
        link a step weaker. A leaf whose ancestry resolves to no defining part is nobody's
        and is counted for nobody.
        
    """
    name = d["name"]
    for a in part.get("in") or []:
        if (links or {}).get(a) == name:
            return True
        if a == name or a.startswith(name + "_"):
            return True
    return False


def _matching(parts: list, d: dict, decls: dict | None = None) -> list:
    """The plan leaves that answer one defining part.

        By the part's own `defines` field where the planner wrote one, and otherwise by name,
        which is what the brief asks for. Two ways rather than one because a planner that
        calls the north gate `north_gate` has answered the defining part called `gate` and
        refusing that would be refusing English.

        The **kind** is the spec's or the one the leaf's committed type declares, for the
        reason `placeplan._kind_ok` gives: a spec names a family, is written before any type
        exists, and cannot know that a palace will be built as an area.
        
    """
    out = []
    for p in parts:
        k = p.get("kind", "plot")
        if k != d["kind"] and not (
                (decls or {}).get(p.get("type"), {}).get("kind", "plot") == k
                and (p.get("defines") == d["name"]
                     or p.get("name", "").startswith(d["name"]))):
            continue
        n = p.get("name", "")
        if p.get("defines") == d["name"] or n == d["name"] \
                or n.startswith(d["name"] + "_") or d["name"] in n:
            out.append(p)
    return out


def _gates_on(wall: dict, parts: list, decls: dict) -> list:
    """The names of the passage points standing on this wall's swept line."""
    from .placeplan import _edge_cells
    cells = set(_edge_cells(wall))
    out = []
    for p in parts:
        if p.get("kind") != "point":
            continue
        if not (decls.get(p.get("type")) or {}).get("passage"):
            continue
        at = p.get("at") or []
        if len(at) >= 2 and (int(at[0]), int(at[-1])) in cells:
            out.append(p["name"])
    return out


#: A2 of the voice contract: the share of a part's family-bearing blocks that have to
#: belong to the voice's six families for the part to have been built in it. Registered
#: before the first reading. A build in the library's default palette -- cobblestone and
#: dark oak -- inside a voice that names neither reads near zero; one that shares a
#: timber with it reads whatever that timber's share is, a fifth or a third; a build in
#: the voice reads over nine tenths, the residue being a door hung in a fallback timber
#: or the natural skin `site()` puts back on a column it worked.
BUILT_SHARE = 0.9

#: **How many family-bearing cells a part has to lay before its share is a reading.**
#: The craft round, registered before the number that tests it. A part under this is
#: `unread` and not `failed`: a garden of worked earth and planting lays almost nothing
#: the palette can resolve, and one stray fence made it 0% in the voice.
BUILT_MIN_CELLS = 24


def built_palette(voice: str | None, parts: list, stood: dict, built, base) -> dict:
    """What every standing part is **made of**, against the voice it was meant to be in.

        Read over every part rather than a sample, because the cost is a slice of two
        arrays per part and a sample is one more place to be wrong about which parts.
        
    """
    from . import prims, styles
    from .lint import plot_rects
    if not voice or voice not in styles.VOICES:
        return {"ok": False, "says": f"no voice this project knows to read the blocks "
                                     f"against: {voice!r}"}
    # A leaf carries `voice` where its ring has one, and a ring built in its own palette
    # read against the place's would fail by name for being exactly what it should be.
    fams_of: dict = {}

    def families(v):
        if v not in fams_of:
            pal_v = styles.VOICES[v]["palette"] if v in styles.VOICES else {}
            fams_of[v] = sorted({f for f in (prims.family(m) for m in pal_v.values()) if f})
        return fams_of[v]
    pal = styles.VOICES[voice]["palette"]
    fams = families(voice)
    rows, failed, unread = [], [], []
    for p in parts:
        name = p["name"]
        if not stood.get(name, False):
            continue
        # **Open ground is not read against a building voice.** The expression round's
        # town: the temple compound's courts are paved in the podium's stone and read
        # 60% "in the voice" of the ring they stand in; a square, a court, a field or a
        # grove is laid in the ground's materials and is unread here, not failed.
        if p.get("kind") == "area":
            unread.append(name)
            continue
        own = p.get("voice") or voice
        if own not in styles.VOICES:
            return {"ok": False, "says": f"{name} is planned in a voice this project "
                                         f"does not know: {own!r}"}
        counts: dict = {}
        for (x0, z0, x1, z1) in plot_rects(pipeline.part_registry_row(p)):
            got = _built_census(built, base, x0, z0, x1, z1)
            for k, v in got.items():
                counts[k] = counts.get(k, 0) + v
        by_fam: dict = {}
        for block, n in counts.items():
            f = prims.family(block)
            if f:
                by_fam[f] = by_fam.get(f, 0) + n
        total = sum(by_fam.values())
        # **A part with almost nothing the palette can read is unread, not failed.** The
        # craft round, and the third time this project has had to write the rule
        # (`Context.ENCLOSED` at 0.85, `walk_fraction`'s seed): a garden is planted
        # ground, and since E6 its paths are the setting's own worked earth and its
        # planting the ground's -- none of which belongs to a material family. One stray
        # fence left it reading `oak x1, 0% in the voice` and failing the clause for
        # being exactly what a garden should be. A share of one block is not a share.
        if total < BUILT_MIN_CELLS:
            unread.append(name)
            continue
        mine = families(own)
        share = sum(n for f, n in by_fam.items() if f in mine) / total
        top = sorted(by_fam.items(), key=lambda kv: -kv[1])[:4]
        row = {"part": name, "type": p.get("type"), "kind": p.get("kind", "plot"),
               "share": round(share, 3), "cells": total, "voice": own,
               "top": [[f, n] for f, n in top]}
        rows.append(row)
        if share < BUILT_SHARE:
            failed.append(row)
    shares = sorted(r["share"] for r in rows)
    ok = bool(rows) and not failed
    voices = sorted({r["voice"] for r in rows}, key=lambda v: (v != voice, v))
    label = (f"voice {voice} ({', '.join(fams)})" if voices == [voice] else
             f"voices {', '.join(voices)}")
    if not rows:
        says = "no standing part laid a block of any material family, so there is nothing to read"
    elif failed:
        worst = failed[0]
        says = (f"{label}: {len(failed)} of {len(rows)} parts are NOT built in their "
                f"own voice -- {worst['part']} ({worst['type']}, {worst['voice']}) is "
                + ", ".join(f"{f} x{n}" for f, n in worst["top"])
                + f", {worst['share']:.0%} in the voice against {BUILT_SHARE:.0%}")
    else:
        says = (f"{label}: all {len(rows)} standing parts are built in their own voice, "
                f"the least at {shares[0]:.1%} of its family-bearing blocks and the "
                f"median at {shares[len(shares) // 2]:.1%}")
    return {"ok": ok, "says": says, "voice": voice, "families": fams, "voices": voices,
            "per_voice": {v: sum(1 for r in rows if r["voice"] == v) for v in voices},
            "threshold": BUILT_SHARE, "read": len(rows), "unread": unread,
            "share_min": shares[0] if shares else None,
            "share_median": shares[len(shares) // 2] if shares else None,
            "failed": [{k: r[k] for k in ("part", "type", "share", "top", "voice")}
                       for r in failed[:24]],
            "failed_count": len(failed)}


def _built_census(built, base, x0: int, z0: int, x1: int, z1: int) -> dict:
    """{block: cells} laid inside a rectangle: what `built` has where `base` differs.

    Without `base` every non-air block in the column band is counted, which reads the
    hillside as well as the house and is the reason a base is always given where one
    exists."""
    import numpy as np
    ax0, ax1 = max(x0, built.x0), min(x1, built.x0 + built.codes.shape[0] - 1)
    az0, az1 = max(z0, built.z0), min(z1, built.z0 + built.codes.shape[2] - 1)
    if ax1 < ax0 or az1 < az0:
        return {}
    sub = built.codes[ax0 - built.x0:ax1 - built.x0 + 1, :, az0 - built.z0:az1 - built.z0 + 1]
    if base is not None and (base.x0, base.y0, base.z0) == (built.x0, built.y0, built.z0) \
            and base.codes.shape == built.codes.shape:
        ref = base.codes[ax0 - built.x0:ax1 - built.x0 + 1, :,
                         az0 - built.z0:az1 - built.z0 + 1]
        # Compared by *name* and not by code: the two volumes grew their palettes
        # separately, so the same block can carry a different index in each.
        bn = np.array(built.palette, dtype=object)[sub]
        rn = np.array(base.palette, dtype=object)[ref]
        mask = bn != rn
        names, n = np.unique(bn[mask], return_counts=True)
    else:
        names, n = np.unique(np.array(built.palette, dtype=object)[sub], return_counts=True)
    return {str(k).split("[")[0]: int(v) for k, v in zip(names, n)
            if str(k).split("[")[0] != "air"}


def _palette(voice: str | None, site: dict | None) -> tuple:
    """The voice's materials against the ground's, as learned it."""
    from . import styles
    if not voice or voice not in styles.VOICES:
        return (False, f"the plan names no voice this project knows: {voice!r}")
    pal = styles.VOICES[voice]["palette"]
    if not site or not site.get("surface_blocks"):
        return (True, f"voice {voice}; no surface census on the site, so the contrast "
                      f"with the ground is not measured")
    from .observe import _is_vegetation
    ground = {k: v for k, v in site["surface_blocks"].items() if not _is_vegetation(k)}
    top = [k for k, _v in sorted((ground or site["surface_blocks"]).items(),
                                 key=lambda kv: -kv[1])[:3]]
    # A family, not a block id: `stone` and `stone_bricks` are the same colour at
    # settlement distance and that is the distance this rule is about.
    def fam(b):
        return (b.replace("_block", "").replace("smooth_", "").replace("cut_", "")
                 .replace("polished_", "").replace("mossy_", "").replace("cracked_", "")
                 .replace("_bricks", "").replace("_brick", "").replace("chiselled_", ""))
    clash = sorted({r for r, m in pal.items()
                    if r in ("wall", "roof") and fam(m) in {fam(g) for g in top}})
    return (not clash,
            f"voice {voice}: " + (
                f"its {', '.join(clash)} is the same material family as the ground "
                f"({', '.join(top)}), and a place the colour of its own hillside "
                f"disappears at distance" if clash else
                f"wall {pal.get('wall')} and roof {pal.get('roof')} against ground of "
                f"{', '.join(top)}"))
