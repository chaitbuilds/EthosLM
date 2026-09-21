"""The resolved design, as a record: what ground each part owns and which requirement it serves.

A place plan says where everything is. It has never said **why**, in the one sense that
matters to a repair: which requirement each region answers, and which stage would have to
change to move it. So a finding at the end of a run could be routed to "the model" and
nowhere else, and the only revision the loop could make was a change of words.

`resolution.json` is the missing link. One entry per region of the design -- a ring, a
shore district, a compound, a wall -- carrying its geometry, its policy, its anchor, its
promised lots and the **requirement id** it answers. `findings_for` then checks the whole
thing against the intent record and returns findings that name an owner, so a shortfall
in capacity goes to `scale`, a wall that was asked for and not drawn goes to `layout`,
and a harbour nothing can build goes to `capability`.

The three completion states stay apart here and nowhere else:

    feasibility   the plan cannot be laid out or cannot hold what was asked for
    construction  it can be laid out and something failed to stand
    fidelity      it stands and it is not the place that was asked for

A resolution with no open `feasibility` finding is a **feasible plan**. That is all it
is: `sample built` and `finished place` are further along and are other records' job.
"""
from __future__ import annotations

from . import contracts, intent as intent_mod, spec as spec_mod


def _pipeline_parts(place: dict | None) -> list:
    """Every leaf of the assembled place, or an empty list before it is assembled."""
    from . import pipeline as _pipeline
    try:
        return list(_pipeline.plan_parts(place or {}))
    except Exception:                            # noqa: BLE001 -- no tree yet
        return []


def _requirement_for(intent: dict, *, family: str | None = None,
                     policy: str | None = None) -> str | None:
    """The requirement id a region answers, where one names it."""
    for r in (intent or {}).get("requirements") or []:
        w = r["wants"]
        if family and r["kind"] == "feature" and w.get("family") == family:
            return r["id"]
        if policy and r["kind"] == "layout" and w.get("policy") == policy:
            return r["id"]
    return None


def resolution_of(spec: dict, place: dict, site: dict,
                  intent: dict | None = None, decls: dict | None = None,
                  parts_record: dict | None = None, plan: dict | None = None) -> dict:
    """The resolution record for a laid-out place.

        Reads the plan that exists rather than laying one out: this is the *record* of a
        resolved design, and the policies are what resolve it.

        **Every region carries the four columns apart**, the design round: the ground the
        requirement is about, the ground the compiler may develop, the lots it drew and the
        footprint that stands. `intent.lot_cover` measures over the first and reports the
        last beside it, so an inferred remainder cannot leave the question and a cover
        figure improved only by enlarging empty lots shows as what it is. See
        `placeplan.region_columns`, which answers all four in one call.
        
    """
    lay = (place or {}).get("layout") or {}
    policy = lay.get("policy") or ("concentric" if lay.get("rings") else "relations")
    regions_out = []
    by_part = {p["name"]: p for p in (spec or {}).get("defining_parts") or []}
    # the **assembled** tree where the run has one: a district's lots live in its own
    # compiled plan, not on the place level, so before assembly there are no leaves to
    # measure and `allocated_columns` is what the compiler recorded or nothing. A leaf
    # names the *row* it stands in (`homes_west_row_0`), not the district, which is the
    # same prefix rule `placeread._in_group` reads a group's membership by.
    leaves = _pipeline_parts(plan if plan else place)

    def _mine(name: str) -> list:
        return [p for p in leaves
                if any(a == name or a.startswith(name + "_")
                       for a in (p.get("in") or []))]

    from . import placeplan as _placeplan
    for d in place.get("districts") or []:
        part = by_part.get(d.get("defines")) or {}
        cols = {}
        try:
            cols = _placeplan.region_columns(
                d, place, decls, record=d.get("compiled") or d.get("record"),
                leaves=_mine(d["name"]) or None, parts_record=parts_record)
        except Exception:                        # noqa: BLE001 -- the clause says so
            cols = {}
        regions_out.append({
            "name": d["name"], "policy": policy,
            **cols,
            "role": part.get("role") or spec_mod.read_role(None, part, d["name"])
            if part else None,
            "defines": d.get("defines"),
            "rect": [d["x0"], d["z0"], d["x1"], d["z1"]],
            "boundary": None, "inner": None, "holes": None,
            "level": d.get("level"),
            "routes": [], "access": [],
            "surface": str(d.get("surface") or "built"),
            "lots": int(d.get("structures") or 0),
            # the expression round: what this region was sized to and from; the design
            # round: the arrangement the parent and the child negotiated to get there
            **{k: d[k] for k in ("target", "land_use", "lot_min", "arrangement")
               if k in d},
            # what the district was told to face, where the design decided; a region
            # with no frontage obligation carries None and the compiler lays as it did
            "faces": d.get("faces"),
            "density": part.get("density"),
            "voice": d.get("voice"),
            "anchor": (lay.get("anchor") if policy == "shoreline" else
                       {"kind": "centre", "at": lay.get("centre")}),
            "requirement": _requirement_for(intent, family=part.get("family")),
            "notes": d.get("notes", "")})

    from . import pipeline as _pipeline
    for p in place.get("parts") or []:
        part = by_part.get(p.get("defines")) or {}
        rect = _pipeline.part_rect({**p, "name": p.get("name")})
        regions_out.append({
            "name": p["name"], "policy": policy, "role": part.get("role"),
            "defines": p.get("defines"), "rect": [int(v) for v in rect],
            "boundary": ([[int(a[0]), int(a[1])] for a in p["path"]]
                         if p.get("path") else None),
            "inner": None, "holes": None, "level": p.get("level"),
            "routes": [], "access": ([[int(p["at"][0]), int(p["at"][-1])]]
                                     if p.get("at") else []),
            "surface": "structure", "lots": None,
            **{k: p[k] for k in ("target", "hierarchy") if k in p},
            "density": None, "voice": p.get("voice"),
            "anchor": None,
            "requirement": _requirement_for(intent, family=part.get("family")),
            "notes": p.get("notes", "")})

    for c in place.get("compounds") or []:
        part = by_part.get(c.get("defines")) or {}
        regions_out.append({
            "name": c["name"], "policy": policy, "role": part.get("role"),
            "defines": c.get("defines"),
            "rect": [c["x0"], c["z0"], c["x1"], c["z1"]],
            "boundary": None, "inner": None, "holes": None, "level": None,
            "routes": [], "access": [], "surface": "compound", "lots": None,
            "density": None, "voice": c.get("voice"), "anchor": None,
            "requirement": _requirement_for(intent, family=part.get("family")),
            "notes": c.get("notes", "")})

    arterials = (place.get("arterials") or {}).get("routes") or []
    bounds = {"site": [site["origin"][0], site["origin"][1], int(site["size"])],
              "structures_promised": sum(int(d.get("structures") or 0)
                                         for d in place.get("districts") or []),
              "size_band": list((spec or {}).get("size_band") or []),
              "explicit_count": (spec or {}).get("explicit_count"),
              "arterial_routes": len(arterials)}
    if policy == "concentric":
        # **The rings, in the record.** `_policy_geometry` asks whether a design that
        # says `concentric` actually is rings inside rings, and before this it had to go
        # looking for them on a plan document the resolution's readers do not have -- so
        # a perfectly good ring city read as a policy with no geometry. A record that a
        # check depends on is part of the record.
        bounds["rings"] = [{"name": r.get("name"), "ring": r.get("ring"),
                            "walled": bool(r.get("walled")),
                            "inner": r.get("inner"), "outer": r.get("outer"),
                            "half": r.get("outer"), "wall": r.get("wall")}
                           for r in (lay.get("rings") or [])]
    if policy == "shoreline":
        bounds["faces"] = lay.get("faces")
        bounds["band_depth"] = (lay.get("districts") or {}).get("band_depth")
        # **The shore's own path, carried into the record.** Without it every downstream
        # check of "which way does the water lie from here" has to fall back to the
        # anchor's one global `water_side`, which is a fact about the site's grain and
        # not about any house on it. The path is what makes the question local, and the
        # resolution is where the consumers look.
        if lay.get("anchor"):
            bounds["anchor"] = {**lay["anchor"], "path": lay.get("anchor_path")}
    negotiated = list(((lay.get("districts") or {}).get("negotiated")) or [])
    if lay.get("squeezed"):
        negotiated.append({"what": "ring_widths", "rings": lay["squeezed"],
                           "why": "the rings' least widths did not fit their shares and "
                                  "the slack was taken proportionally"})
    return contracts.make(
        "resolution", policy=policy, regions=regions_out, site=dict(bounds["site"] and
                                                                   {"origin": list(site["origin"]),
                                                                    "size": int(site["size"])}),
        centre=lay.get("centre"), bounds=bounds, negotiated=negotiated,
        note="one entry per region of the resolved design, each linked to the "
             "requirement it answers")


def plan_structures(plan: dict | None) -> int:
    """Every building in the plan, wherever in the tree it stands.

        **One accounting rule.** The review found the city's resolution reporting 583
        structures against an assembled plan of 634: the count was the sum of the districts'
        compiled lots, and the palace's 51 plots are descendants of a compound rather than
        of a district, so they were in the plan and in no total. A building is a building
        whichever parent laid it.
        
    """
    from . import pipeline as _pipeline
    return sum(1 for p in _pipeline.plan_parts(plan or {})
               if p.get("kind", "plot") == "plot")


def with_realized(resolution: dict, realized: dict, plan: dict | None = None,
                  subject: str | None = None) -> tuple:
    """`(resolution, findings)` -- the record, over what the compiler actually laid.

        **The promise and the realization in one record, so they can be compared.** The
        review's third finding, and the sentence it turns on: "Do not promise one capacity
        and independently discover a different one during compilation." Until now `lots` was
        the allocator's promise and the number of buildings in the plan was a fact nobody
        wrote next to it; the two lived in different files produced by different rules, and
        the city's run discovered the difference as a validator refusal 36 districts in.

        A district short of its promise is a `layout` finding blocking feasibility -- the
        allocator made the promise and the allocator is who can move it. A district that
        beat its promise is recorded and is not a finding: more houses than budgeted is a
        fact about the ground, and the count the sentence asked for is checked elsewhere and
        by a rule that does not read this one.
        
    """
    out, rows = dict(resolution), []
    regions, promised, laid = [], 0, 0
    for r in resolution.get("regions") or []:
        r = dict(r)
        if r.get("lots") is not None:
            got = int(realized.get(r["name"], 0))
            r["realized"] = got
            promised += int(r["lots"])
            laid += got
            if got < int(r["lots"]):
                rows.append({
                    "id": f"find/capacity/realized/{r['name']}",
                    "requirement": r.get("requirement"), "part": r["name"],
                    "says": (f"{r['name']} was promised {r['lots']} structure(s) and "
                             f"the compiler laid {got} in it; the promise came from an "
                             f"area estimate and the realization from the actual "
                             f"streets, lots and setbacks"),
                    "evidence": {"promised": int(r["lots"]), "realized": got,
                                 "rect": r.get("rect")},
                    "owner": "layout", "blocks": "feasibility", "severity": "warning",
                    "seen_by": "resolve.with_realized", "fixed": False})
        regions.append(r)
    out["regions"] = regions
    # **Three numbers, and they are three different facts.** `with_realized` used to
    # overwrite `structures_promised` with the realized total, so the obligation the
    # programme carried and the count the ground gave became one field and the
    # difference between them could not be read afterwards. The review's arithmetic: the
    # city reported 583 against an assembled plan of 634. promised what the programme
    # obliged this place to hold. Never revised here; only `repair`, at the scale owner
    # and inside a registered bound, moves it, and it records that it did. allocated
    # what the allocator budgeted across the districts. Revisable by the layout owner,
    # and the number a shortfall finding is about. realized what is actually in the plan
    # -- **every** building, including the descendants of a compound, by one rule
    # (`plan_structures`).
    was = resolution.get("bounds") or {}
    # **The count the sentence made is of the thing it named.** The closure round: a
    # village of sixteen cottages with a hall on the square realized "17" because every
    # plot was a structure, and the explicit count was refused for the hall. `subject`
    # is the count's own word (`cottages`), resolved by the one selector the checker
    # uses.
    realized_all = plan_structures(plan) if plan is not None else laid
    realized_subject = None
    if subject and plan is not None:
        from . import intent as _intent, pipeline as _pipeline
        with __import__("contextlib").suppress(Exception):
            realized_subject = len([p for p in _intent.select(
                _pipeline.plan_parts(plan), subject) if p.get("kind", "plot") == "plot"])
    out["bounds"] = {**was,
                     "structures_promised": int(was.get("structures_promised") or 0),
                     "structures_allocated": promised,
                     "structures_realized": (realized_subject if realized_subject
                                             is not None else realized_all),
                     "structures_realized_all": realized_all,
                     "structures_in_districts": laid,
                     **({"count_subject": subject} if subject else {})}
    return (contracts.read("resolution", out), rows)


def capacity_findings(spec: dict, place: dict, resolution: dict,
                      intent: dict | None = None) -> list:
    """Does the resolved design hold what the place was asked to hold?

        The allocator's promise is the sum of the districts' counts, and it is checked
        against the band the spec carries **and** against any explicit count. Short of an
        explicit count is a `fidelity` failure that no repair may negotiate away; short of
        the kind's own band is a `feasibility` finding routed to `scale`, because the band
        is the library's own inference and is the thing a repair is allowed to revise.
        
    """
    out = []
    # **What the place holds, once anything has been laid.** Before compilation the
    # allocator's promise is the only number there is; after it, the buildings that
    # exist are what the band and the sentence are checked against, and `realized`
    # counts every one of them including a compound's descendants.
    bounds = resolution["bounds"]
    promised = int(bounds.get("structures_realized")
                   if bounds.get("structures_realized") is not None
                   else bounds.get("structures_promised") or 0)
    lo, hi = (spec.get("size_band") or [0, 0])[:2] or (0, 0)
    explicit = spec.get("explicit_count")
    req = next((r["id"] for r in (intent or {}).get("requirements") or []
                if r["kind"] == "count"), None)
    if explicit:
        n, about = int(explicit["n"]), bool(explicit.get("about"))
        want_lo, want_hi = ((max(1, int(round(n * (1 - spec_mod.ABOUT)))),
                             int(round(n * (1 + spec_mod.ABOUT)))) if about else (n, n))
        if not (want_lo <= promised <= want_hi):
            out.append({
                "id": "find/capacity/explicit", "requirement": req, "part": None,
                "says": (f"the sentence asks for {n} structures and the resolved design "
                         f"promises {promised}; the ceiling for a {spec.get('kind')} is "
                         f"{(spec.get('ceiling') or {}).get('structures')}"),
                "evidence": {"promised": promised, "asked": [want_lo, want_hi],
                             "unmet": spec.get("unmet"),
                             "ceiling": spec.get("ceiling")},
                "owner": "scale", "blocks": "fidelity", "severity": "error",
                "seen_by": "resolve.capacity_findings", "fixed": False})
    elif lo and promised < int(lo):
        out.append({
            "id": "find/capacity/band", "requirement": req, "part": None,
            "says": (f"the resolved design promises {promised} structures against the "
                     f"{lo}-{hi} band a {spec.get('kind')} is inferred to be; the "
                     f"ground the layout could claim is the limiting constraint"),
            "evidence": {"promised": promised, "band": [int(lo), int(hi)],
                         "policy": resolution.get("policy"),
                         "negotiated": resolution.get("negotiated"),
                         "districts": [{"name": r["name"], "lots": r["lots"],
                                        "rect": r["rect"]}
                                       for r in resolution["regions"]
                                       if r.get("lots") is not None]},
            "owner": "scale", "blocks": "feasibility", "severity": "error",
            "seen_by": "resolve.capacity_findings", "fixed": False})
    return out


def unclaimed_ground(spec: dict, place: dict, site: dict, short: list) -> list:
    """A design short of its band, standing beside ground it never claimed.

        **The measurement that gives recovery a spatial move.** A capacity shortfall was a
        `scale` finding and nothing else, so the only available answer was "want fewer
        houses" -- the review's fourth finding, in one sentence: recovery "does not search
        alternative region geometry, lot types, site extent or ground works". Whether the
        districts could simply be bigger is a question about the *site*, it is answerable
        from the rectangles that are already on the record, and it belongs to `layout`.

        Emitted only when the place is actually short: a design that holds what it promised
        and leaves ground over has chosen to, and that is not a defect.
        
    """
    from .pipeline import stages_plan
    # **Short, and not merely off.** The closure round's proof: an explicit count of 16
    # with 18 promised is a capacity finding, and this read every capacity finding as a
    # shortfall, grew both districts into their free ground and raised their asks to 28
    # -- the repair for a place with too many houses was more ground for more houses.
    short = [f for f in short or []
             if int((f.get("evidence") or {}).get("promised") or 0)
             < int(((f.get("evidence") or {}).get("asked") or
                    (f.get("evidence") or {}).get("band") or [0])[0] or 0)]
    if not short:
        return []
    out = []
    for d in place.get("districts") or []:
        if not d.get("structures"):
            continue
        bigger = stages_plan._grow_into_free_ground(place, d, site)
        if bigger is None:
            continue
        was = (int(d["x1"]) - int(d["x0"]) + 1) * (int(d["z1"]) - int(d["z0"]) + 1)
        now = (bigger[2] - bigger[0] + 1) * (bigger[3] - bigger[1] + 1)
        out.append({
            "id": f"find/extent/unclaimed/{d['name']}", "requirement": None,
            "part": d["name"],
            "says": (f"{d['name']} covers {was} columns and there are {now - was} "
                     f"columns of free ground beside it, inside the site and touching "
                     f"no other region, while this place is short of the size its own "
                     f"kind is inferred to be. The ground is a layout decision and it "
                     f"has not been made"),
            "evidence": {"rect": [int(d["x0"]), int(d["z0"]), int(d["x1"]),
                                  int(d["z1"])],
                         "grown": bigger, "columns": was, "grown_columns": now},
            "owner": "layout", "blocks": "feasibility", "severity": "warning",
            "seen_by": "resolve.unclaimed_ground", "fixed": False})
    return out


def findings_for(spec: dict, place: dict, site: dict, intent: dict,
                 parts_record: dict | None = None,
                 capabilities: dict | None = None,
                 reading: dict | None = None,
                 plan: dict | None = None, decls: dict | None = None) -> tuple:
    """`(intent, resolution, findings)` for a laid-out place.

        One call, so that nothing downstream can check coverage against a resolution it did
        not read, or read a resolution whose coverage was never checked.

        `plan` is the **assembled** plan where the run has one -- every leaf of every
        compiled district, not only the place level's own parts. Coverage checks that read
        geometry (which way a home fronts, whether an undeclared wall was drawn) can only
        answer from the leaves that exist, and before the districts compile there are none;
        that is a true "not yet" and it is recorded as `open`, never as satisfied.
        
    """
    resolution = resolution_of(spec, place, site, intent, decls=decls,
                               parts_record=parts_record, plan=plan)
    # **Before the districts compile, the plan is not a plan.** The closure round's
    # retained failure: coverage was handed the place level's two leaves as if they were
    # the assembled tree, measured `gathered around` on a hall and a square, found one
    # cottage where sixteen were asked, found no type declaring a dwelling in use -- and
    # the repair pass acted on all of it, growing districts and re-asking allocations
    # against findings that described nothing. What is decidable before assembly is what
    # the spec declares, what the site is, what the allocator promised and what the
    # library can build; every measurement of leaves waits for the leaves.
    checked, found = intent_mod.coverage(intent, spec, plan=plan,
                                         parts_record=parts_record,
                                         resolution=resolution, site=site,
                                         reading=reading, capabilities=capabilities)
    rows = list(found["findings"])
    short = capacity_findings(spec, place, resolution, checked)
    rows += short
    # a spatial answer to a shortfall, where the ground allows one, before the scale
    # owner is asked to want less
    rows += unclaimed_ground(spec, place, site, short)
    for e in (capabilities or {}).get("entries") or []:
        if e.get("matched"):
            continue
        rows.append({
            "id": f"find/{e['id']}", "requirement": e.get("requirement"),
            "part": e["wants"].get("part"),
            "says": f"{e['wants']['name']}: {e['why']}",
            "evidence": {"kind": e["kind"], "family": e["family"]},
            "owner": "capability", "blocks": "feasibility", "severity": "error",
            "seen_by": "capability.match", "fixed": False})
    return (checked, resolution,
            contracts.make("findings", findings=rows, stage="resolve.findings_for",
                           note="requirement coverage, capacity and capability, each "
                                "routed to the layer that can repair it"))


def _as_plan_parts(place: dict) -> list:
    """The place level's own leaves and compounds, in `plan_parts` shape.

        The place level is not the assembled plan -- the districts have not been compiled --
        and coverage has to be checkable **before** that, or a wall omitted from the spec is
        found after four hundred buildings rather than before them.
        
    """
    out = [dict(p) for p in (place.get("parts") or [])]
    for c in place.get("compounds") or []:
        out.append({**c, "kind": "area", "compound": c["name"],
                    "name": c["name"]})
    return out


def says(resolution: dict, findings: dict) -> str:
    """One line, for a log."""
    by: dict = {}
    for f in findings.get("findings") or []:
        by[f["blocks"]] = by.get(f["blocks"], 0) + 1
    b = resolution["bounds"]
    counts = [f"{b.get('structures_promised')} promised"]
    for k, word in (("structures_allocated", "allocated"),
                    ("structures_realized", "realized")):
        if b.get(k) is not None:
            counts.append(f"{b[k]} {word}")
    return (f"{resolution['policy']}: {len(resolution['regions'])} region(s), "
            + ", ".join(counts) + "; "
            + (", ".join(f"{k} {v}" for k, v in sorted(by.items())) or "no findings"))
