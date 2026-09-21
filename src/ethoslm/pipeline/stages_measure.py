"""Deterministic measurements, findings and registered readouts."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import time
from dataclasses import dataclass, field

from .. import pipeline as _pipeline
from .. import card as card_mod
from .. import measure as measure_mod
from .. import offline, settlement, verdicts
from ..measure import record
from .round import Round


def stage_lint(rnd: Round, be, results: dict) -> dict:
    """The correctness floor over the settlement as it stands."""
    from .. import lint
    built = rnd.rel("world_built.npz")
    if not os.path.exists(built):
        return {"error": "no world_built.npz -- nothing has been built yet"}
    vol = offline.load_volume(built)
    s = rnd.site or (json.load(open(rnd.rel("site.json")))
                     if os.path.exists(rnd.rel("site.json")) else None)
    region = None
    if s:
        X, Z = s["origin"]
        region = (X, Z, X + s["size"] - 1, Z + s["size"] - 1)
    plots = json.load(open(rnd.rel("plots.json"))) \
        if os.path.exists(rnd.rel("plots.json")) else []
    # **A sample is linted over the sample.** The integration review's finding, and it
    # is a scoping error with a large and misleading output: `stage_parts` builds a
    # construction sample -- two adjoining quarters and what joins them -- and this
    # stage then read the whole place's plot registry and the whole place's region, so
    # every plot of the other thirty-four districts was reported as a room nobody can
    # walk into and a block held up by nothing. They were not built. A check whose scope
    # and whose construction disagree is measuring the disagreement.
    scope = None
    built_rec = rnd.rel("parts.json")
    if os.path.exists(built_rec):
        rec = json.load(open(built_rec))
        # **Every registry row carries the floor its part was sited at.** The expression
        # round's rings baseline: `plots.json` on disk had no `y0` on any row (the
        # construction stage's annotation does not reach the file on the parallel path),
        # so `room_owner` could not tell a natural cave under the temple court from the
        # great hall's interior and the whole town blocked on a 340-cell overhang the
        # plateau had re-skinned. The construction record is authoritative for where
        # each part stands; the registry the check reads is annotated from it here,
        # always.
        from .stages_build import floors_into
        plots = floors_into(plots, [r for w in rec.get("waves") or []
                                    for r in (w.get("parts") or [])])
        if rec.get("sample"):
            names = {r["part"] for w in rec.get("waves") or []
                     for r in w.get("parts") or []}
            plots = [p for p in plots if p.get("label") in names
                     or p.get("name") in names]
            r = rec["sample"].get("rect")
            if r:
                m = int(rec["sample"].get("margin") or 0)
                region = (int(r[0]) - m, int(r[1]) - m,
                          int(r[2]) + m, int(r[3]) + m)
            scope = {"sample": True, "quarters": rec["sample"].get("quarters"),
                     "parts": len(names), "plots": len(plots),
                     "region": list(region) if region else None,
                     "of_plan": rec["sample"].get("of_plan"),
                     "why": ("this round built a construction sample, so the registry "
                             "and the region are the sample's; the rest of the place "
                             "was not built and is not linted as though it had been")}
    t0 = time.perf_counter()
    net = rnd.network()
    ctx = lint.Context.build(vol, plots, network=net, region=region,
                             base=_prebuild(rnd))
    rep = lint.lint(ctx)
    secs = round(time.perf_counter() - t0, 1)
    counts = rep.to_json()["counts"]
    record("check", name="lint", settlement=rnd.name, errors=len(rep.errors),
           warnings=len(rep.warnings), seconds=secs)
    access = _doors_from_the_lane(ctx, net, plan_entries(rnd.plan()))
    out = {"errors": len(rep.errors), "warnings": len(rep.warnings),
           "counts": counts, "seconds": secs,
           **({"scope": scope} if scope else {}),
           "e002_from_the_lane": access,
           "findings": [{"code": f.code, "message": f.message, "pos": f.pos}
                        for f in rep.findings]}
    out["confirmed"] = _confirm_features(rnd, vol, ctx, net)
    # **...and the built artifact is stamped over the record as it now stands.** The
    # composition round, found by asking `deps.check(rnd, "built")` on a state that had
    # just finished: `stage_parts` stamps `built` over `parts.json` and then *this*
    # stage rewrites `parts.json` with the assembled world's answers, so the stamped
    # output digest never matched the file again and every later invocation rebuilt 139
    # parts to reproduce a world it already had. The confirmation is part of what a
    # built candidate is; the stamp is re-made here, after it, over the same outputs.
    with contextlib.suppress(Exception):
        from .. import deps as _deps_l
        _deps_l.stamp(rnd, "built",
                      outputs=["parts.json", "world_built.npz"]
                      + (["surfaces.json"]
                         if os.path.exists(rnd.rel("surfaces.json")) else []),
                      plan=rnd.plan(),
                      note="re-stamped after the assembled-world confirmation rewrote "
                           "the parts record")
    # **A construction check that fails is an outcome, not a number in a report.** The
    # review's fifth finding: "`stage_lint` returns errors and `stage_place_check`
    # returns `holds: false`, but the driver's stop protocol requires `stop`; these
    # outputs do not become corresponding controller outcomes." So a place whose
    # buildings do not stand went on to be read, judged and reported as a finished run
    # while its own physical check said otherwise. `blocks: construction` is the state
    # this is: the place is built and it is not soundly built, which is distinct from a
    # plan that is infeasible and from a place that is not the one asked for. **What
    # this backend cannot measure is unmeasured, not absent and not harmless.** E005 is
    # "connective blocks that never joined to their neighbours", and the join is
    # computed by the *server*, in `Builder.flush`'s second pass with block updates on
    # -- which `LiveBackend.publish` runs and a dry run, by construction, never does. So
    # offline every fence, wall and pane in the place reports a missed join, and neither
    # available answer was honest: blocking calls a place unsound for a pass nobody ran,
    # and ignoring it is the "declare all errors harmless" the review named. It is
    # reported by name, with the reason it is not measurable here, and the round says so
    # rather than deciding.
    unmeasured = [f for f in rep.errors if f.code in LIVE_ONLY_CODES] \
        if not getattr(be, "live", False) else []
    real = [f for f in rep.errors if f not in unmeasured]
    if unmeasured:
        out["unmeasured"] = {
            "codes": sorted({f.code for f in unmeasured}),
            "findings": len(unmeasured),
            "why": ("these checks read block states the server computes when it writes "
                    "the place -- the joins a fence, a wall and a pane make with their "
                    "neighbours -- and this run built into a cached volume and wrote "
                    "nothing. They are not measured here and are not evidence either "
                    "way; a live write-back is what answers them")}
    if real:
        out.update({"status": "blocked", "stop": True, "blocks": "construction",
                    "error": (f"the construction check reports {len(real)} "
                              f"error(s) over what was built"
                              + (" (a construction sample)" if scope else "")
                              + ": " + "; ".join(
                                  f"{f.code} {f.message}" for f in real[:4]))})
    elif access.get("status") == "short":
        out.update({"status": "blocked", "stop": True, "blocks": "construction",
                    "error": (f"the built place is not one connected network: "
                              + (access.get("why") or
                                 f"{access.get('not_walkable_from_the_lane')} door(s) "
                                 f"cannot be walked to from any entry the plan draws"))})
    return out


def _confirm_features(rnd: Round, vol, ctx, net) -> dict:
    """**Ask the assembled world what each part actually delivers**, and write it down.

        The design round's third boundary, at the one point in the pipeline where the whole
        place stands and the volume is already open. `construction.outcome` measures a
        part's own emission *before its neighbours are built*, so a courtyard a later wall
        filled in, a forge a terrace buried and a door the finishing pass paved over were
        all still recorded as delivered. `construction.confirm` re-reads them here through
        `ethoslm.usable`'s predicates, mutates `parts.json` where an answer moved, and leaves
        `usable.json` beside it so a reader can see *how* each answer was got and what was
        standing when it was read.

        Reported and never raised: a predicate that cannot run leaves the emission record
        exactly as it was, which is the weaker evidence it always had, and says so.
        
    """
    from .. import construction, contracts, demand, usable
    rec_p = rnd.rel("parts.json")
    if not os.path.exists(rec_p):
        return {"skipped": "no parts record: nothing was built to confirm"}
    rec = json.load(open(rec_p))
    # **The predicates this part owes, not a fixed three.** The composition round: the
    # default ran `entrance_connected`, `equipment_reachable` and `court_accessible` on
    # everything, so three of the six existed and no part was ever asked about the
    # features *it* was required to deliver. The demand binding says what is required of
    # what (`demand.required_by_part`), and `confirm` derives each part's predicate set
    # from it. With no binding the answer is what it was.
    binding = {}
    try:
        binding = demand.required_by_part(rnd.place_spec(),
                                          contracts.load(rnd, "intent"), rnd.plan())
    except Exception as e:                        # noqa: BLE001 -- reported, not raised
        binding = {}
        print(f"   confirm: no demand binding ({type(e).__name__}: {e}); the fixed "
              f"predicate set is asked and nothing is bound per part", flush=True)
    try:
        world = usable.World(ctx, [r for w in rec.get("waves") or []
                                   for r in (w.get("parts") or [])],
                             digest=_content_print(rnd.rel("world_built.npz")),
                             state=rnd.state, network=net, registry=ctx.plots)
        got = construction.confirm(world, rec, required=binding or None)
    except Exception as e:                        # noqa: BLE001 -- reported, not raised
        return {"status": "unconfirmed",
                "why": (f"the assembled-world predicates could not run "
                        f"({type(e).__name__}: {e}); every feature keeps the evidence "
                        f"its own emission gave it, which is weaker")}
    json.dump(rec, open(rec_p, "w"), indent=1)
    rows = []
    for w in rec.get("waves") or []:
        for r in w.get("parts") or []:
            for want, a in ((r.get("emitted") or {}).get("usable") or {}).items():
                rows.append({"part": r.get("part"), "type": r.get("type"),
                             "want": want, "holds": a.get("holds"),
                             "method": a.get("method"), "why": a.get("why"),
                             "subjects": a.get("subjects")})
    json.dump({"candidate": _candidate(rnd), "built_digest": world.digest,
               "provenance": world.provenance(), "checks": rows,
               "changed": got.get("changed"), "why": got.get("why")},
              open(rnd.rel("usable.json"), "w"), indent=1)
    print(f"   confirmed: {got['why']}", flush=True)
    return {k: got[k] for k in ("parts", "why")} | {"moved": len(got.get("changed") or []),
                                                    "written": rnd.rel("usable.json")}


def _content_print(path: str):
    from .. import deps
    return deps.content_print(path)


def _candidate(rnd: Round):
    from .. import deps
    try:
        return deps.candidate_id(rnd)
    except Exception:                             # noqa: BLE001 -- no plan, no candidate
        return None


#: How far from an entry's own column a lane stance may be and still be that entry's
#: lane. A gate is a few blocks deep and its threshold is the lane on the inside of it.
ENTRY_REACH = 12

#: The lint codes whose evidence only a **live write-back** produces. `E005` asks
#: whether a fence, a wall or a pane joined to its neighbours, and that state is
#: computed by the server in `Builder.flush`'s block-update pass -- which
#: `LiveBackend.publish` runs once at the end of a live round and an offline build never
#: does. Listed here, by code, so that "this backend cannot answer it" is a decision on
#: the record rather than a silence or a false failure.
LIVE_ONLY_CODES = ("E005",)


def plan_entries(plan: dict | None) -> list:
    """The columns a place is walked into by: every passage point the plan draws.

        A gate is what justifies a seed. Where a plan draws none there is no justified
        entry, and `_doors_from_the_lane` says so rather than seeding the whole lane.
        
    """
    from . import stages_media
    out = []
    for p in _pipeline.plan_parts(plan or {}):
        if p.get("kind") == "point" and stages_media._passage(p) and p.get("at"):
            out.append((int(p["at"][0]), int(p["at"][-1])))
    return sorted(set(out))


def entry_stances(seeds: list, entries) -> list:
    """The lane stances at the place's own entries: where a walk in actually starts."""
    out = []
    for (ex, ez) in entries or []:
        for (x, z, s) in seeds:
            if abs(x - ex) <= ENTRY_REACH and abs(z - ez) <= ENTRY_REACH:
                out.append((x, z, s))
    return sorted(set(out))


def _doors_from_the_lane(ctx, net, entries=None) -> dict:
    """E002's question asked from the place's **entries** instead of from everywhere.

        The lane is the outdoors that a settlement's doors actually front onto, it is
        walk-only end to end by construction, and it does not move when the cache does. So
        this is the same check seeded from it.

        **From the entries, and the whole lane reported beside it.** The review:

          > The sample's zero unreachable-door result does not establish one connected
          > network: `_doors_from_the_lane` seeds its walk from every lane stance, across
          > all components. E007 simultaneously reports a disconnected component of 5,922
          > stances.

        Seeding every stance asks each island whether it can reach itself. So the walk now
        starts at the gates the plan draws -- `entries`, as `(x, z)` columns -- and the
        number of lane components is reported whether or not a door is short, because a
        place whose lanes are in three pieces has not got one network however its doors
        come out. With no entry given, this says so and does not certify anything.
        
    """
    from .. import lint
    if not net:
        return {"note": "no circulation network in this round"}
    seeds = lint.lane_stances(ctx.nav, net)
    if not seeds:
        return {"note": "no lane cell can be stood on"}
    # the lane's own shape, whatever the doors do: one flood per unvisited stance
    left, pieces = set(seeds), []
    while left:
        start = next(iter(left))
        got = set(ctx.nav.flood([start], max_jumps=0)) & left
        pieces.append(len(got) or 1)
        left -= (got or {start})
    pieces.sort(reverse=True)
    at = entry_stances(seeds, entries)
    entries_from = "the plan's passage points"
    if not at and not entries:
        # **An unwalled place is entered where its lanes reach the outside.** The
        # closure round's proof: a village with no wall draws no gate, so this check had
        # no justified seed and answered `unresolved` for a place whose lanes were one
        # piece. The lane stances at the outer edge of the lane network -- within
        # `ENTRY_REACH` of its own bounding box -- are where a walk in begins, and the
        # record says that is the rule it used.
        xs = [x for x, _z, _s in seeds]
        zs = [z for _x, z, _s in seeds]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        at = sorted({(x, z, s) for (x, z, s) in seeds
                     if min(x - x0, x1 - x, z - z0, z1 - z) <= ENTRY_REACH})
        entries_from = "unwalled: the lane network's outermost stances"
    if not at:
        return {"doors": len(ctx.doors), "lane_stances": len(seeds),
                "lane_components": len(pieces), "component_sizes": pieces[:6],
                "status": "unresolved",
                "note": (f"this plan draws no entry within {ENTRY_REACH} columns of a "
                         f"standable lane cell, so where a walk into the place begins "
                         f"is not something it says; seeding every lane stance would "
                         f"only show that each of {len(pieces)} component(s) reaches "
                         f"itself"),
                "e002_seeds": ctx.outdoor_seeds}
    reach = set(ctx.nav.flood(at, max_jumps=0))
    bad = []
    for (x, y, z) in ctx.doors:
        s = ctx.door_stance(x, y, z)
        if s is None or (x, z, s) not in reach:
            bad.append([x, y, z])
    stranded = len([s for s in seeds if s not in reach])
    # **One network, not one network per gate.** The review: this returned `connected`
    # for two disconnected components whenever each of them happened to have an entry
    # near it -- the flood is seeded from every entry at once, so a place in two pieces
    # with a gate into each piece reaches every door and strands no stance, and the
    # number this check exists to produce says the place is joined up. It is not: a
    # walker cannot get from one piece to the other. So the lane's own connectedness is
    # a clause of the answer and not a figure printed beside it.
    one_network = len(pieces) <= 1
    return {"doors": len(ctx.doors), "not_walkable_from_the_lane": len(bad),
            "lane_stances": len(seeds), "entry_stances": len(at),
            "entries": len(entries or []), "entries_from": entries_from,
            "lane_components": len(pieces),
            "component_sizes": pieces[:6], "one_network": one_network,
            "lane_stances_not_reached_from_an_entry": stranded,
            "status": ("connected" if not bad and not stranded and one_network
                       else "short"),
            "why": ("" if one_network else
                    f"every door is reachable and the lanes are in {len(pieces)} "
                    f"pieces ({', '.join(str(n) for n in pieces[:4])} stances): each "
                    f"piece has an entry of its own, so a walk from one gate reaches "
                    f"that piece and no other. Connected access is one network"),
            "examples": bad[:8], "e002_seeds": ctx.outdoor_seeds}


def stage_measures(rnd: Round, be, results: dict) -> dict:
    """Every candidate, by the instruments that never see an image. Offline and free.

        This is the guard against the circularity the spec leads with. Asking the judge
        whether its own winner is better measures nothing, so each candidate is measured
        here -- lint errors over the region, the walk-only interior fraction of the wave's
        own rooms, how differentiated the wave's structures are from each other -- and the
        numbers are on disk *before* any verdict is read against them.

        `blocks` rides along without a direction registered, because more blocks is not
        better and pretending otherwise would smuggle in a fourth measure.
        
    """
    c = rnd.candidates
    if not c:
        return {"note": "this round has no candidates to measure"}
    mine = {p["label"] for p in _pipeline._cand_plots(rnd)}
    margin = int((c.get("measures") or {}).get("variety_margin", 6))
    out = {}
    for cid in _pipeline._cand_ids(rnd):
        prog = _pipeline._cand_program(rnd, cid)
        out[cid] = measure_program(rnd, be, prog, mine, margin=margin)
        if out[cid]["status"] != "measured":
            continue
        record("check", name=f"candidate_measures/{cid}", settlement=rnd.name,
               errors=out[cid]["lint_errors"], warnings=out[cid]["lint_warnings"],
               seconds=out[cid]["seconds"], blocks=out[cid]["blocks"],
               walk_pct=out[cid]["walk_pct"])
    path = os.path.join(_pipeline._cand_dir(rnd), "measures.json")
    os.makedirs(_pipeline._cand_dir(rnd), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return {**out, "written": path}


def measure_program(rnd: Round, be, prog: str, mine: set, margin: int = 6) -> dict:
    """One program, by every instrument that never sees an image.

        Lifted out of `stage_measures` unchanged so the arms of a sight round are measured
        by exactly the same code as the selection candidates -- two measurement paths that
        drifted apart would make the two rounds incomparable, which is the whole reason
        there is one measurement schema.

        `articulation` is the one addition, and it carries **no registered direction** in
        any round. It is reported beside `blocks` for that reason: it is a fact about wall
        surfaces, put on the record because a person named facade depth as the thing that
        separated two candidates and nothing here was watching it.
        
    """
    from .. import articulation as artic_mod
    from .. import lint, stages, variety as variety_mod
    if not os.path.exists(prog):
        return {"status": "no_program", "program": prog}
    vol = be.volume
    net = rnd.network()
    plots = json.load(open(rnd.rel("plots.json")))
    s = rnd.site or json.load(open(rnd.rel("site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    t0 = time.perf_counter()
    b = be.execute(prog, vol)
    pending = b._pending
    applied = stages.apply_pending(vol, pending)
    ctx = lint.Context.build(applied, plots=plots, network=net,
                             region=(X, Z, X + S - 1, Z + S - 1),
                             fittings=b.fitting_cells)
    rep = lint.lint(ctx)
    counts = rep.to_json()["counts"]

    walk = set(ctx.nav.flood(ctx.nav.perimeter_seeds(inset=2, step=3),
                             max_jumps=0))
    cells = reach = zero = n_rooms = 0
    for r in ctx.rooms:
        if r.get("plot") not in mine:
            continue
        # `floor_cells` and `walk_pct` are read off the room's floor -- the surface you
        # walk on -- and not off every stance in the component. See
        # `observe.floor_stances`. This is the one change to this function since it was
        # lifted out of `stage_measures`, and it moves recorded rows: they are re-read
        # and superseded with the cause in `scripts/test_types.py`.
        st = set(map(tuple, r["floor"]))
        n_rooms += 1
        cells += len(st)
        got = len(st & walk)
        reach += got
        zero += 0 if got else 1

    # variety over what this candidate built, exactly as scripts/variety.py --program
    # measures a pass: crop to the built mass, overlay the pending set, and keep only
    # the plots inside that crop.
    xs = [p[0] for p in pending]
    zs = [p[2] for p in pending]
    vsub = vol.sub(min(xs) - margin, min(zs) - margin,
                   max(xs) - min(xs) + 2 * margin + 1,
                   max(zs) - min(zs) + 2 * margin + 1)
    vsub.overlay(pending)
    vplots = [p for p in plots
              if p["x0"] >= min(xs) - margin and p["x1"] <= max(xs) + margin
              and p["z0"] >= min(zs) - margin and p["z1"] <= max(zs) + margin]
    var = variety_mod.measure(vsub, vplots)

    art = artic_mod.measure(vsub, vplots)
    per = [r for r in art["plots"] if r.get("counted")]

    return {
        "status": "measured", "program": os.path.relpath(prog, _pipeline.ROOT),
        "blocks": len(pending),
        "lint_errors": len([f for f in rep.findings if f.code.startswith("E")]),
        "lint_warnings": len([f for f in rep.findings
                              if f.code.startswith("W")]),
        "lint_clean": not [f for f in rep.findings if f.code.startswith("E")],
        "lint_counts": counts,
        "rooms": n_rooms, "floor_cells": cells, "walk_reachable": reach,
        "walk_pct": round(100 * reach / cells, 1) if cells else None,
        "rooms_zero_walkable": zero,
        "variety_structures": var["structures"],
        "variety_ridge_cv": var["ridge_height"].get("cv"),
        "variety_ridge_stdev": var["ridge_height"].get("stdev"),
        "variety_aspect_stdev": var["aspect"].get("stdev"),
        "variety_roof_distinct": var["roof"].get("distinct"),
        # Reported, never targeted, no direction registered. Both the whole-subject
        # aggregate and the mean over structures, because they can disagree -- a
        # candidate that barely built its second structure is weighted differently by
        # the two, and on the four selection candidates that is exactly what happens.
        "articulation": art["articulation"],
        "articulation_per_structure": (
            round(sum(r["articulation"] for r in per) / len(per), 4)
            if per else None),
        "articulation_relief": art["relief"],
        "articulation_samples": art["samples"],
        "seconds": round(time.perf_counter() - t0, 1),
    }


def build_context(rnd: Round, be, prog: str | None, pending: dict | None = None,
                  plots: list | None = None, fittings: set | None = None):
    """The lint context for one program on one round's ground: execute offline against
        the base volume, apply the pending set, build the Context over the round's plots,
        network and region. `prog=None` is the bare pre-build volume -- the lint floor every
        candidate on that ground shares.

        Shared by `stage_revise`, `check_program` and `scripts/entry_diagnosis.py` so the
        diagnosis that registered this experiment, the loop that runs it and the check the
        builder holds in its own hand all read the same world.

        `pending` is that program's block set when the caller has already executed it.
        Executing a settlement wave twice to answer one question is thirty seconds a builder
        pays for nothing.
        
    """
    from .. import lint, stages
    vol = be.volume
    if plots is None:
        p = rnd.rel("plots.json")
        plots = json.load(open(p)) if os.path.exists(p) else []
    s = rnd.site or json.load(open(rnd.rel("site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    if pending is None and prog:
        # `fittings` is the program's own furniture register. Where the caller has
        # already executed the program it hands it in beside `pending`; where this call
        # does the executing it takes it off the builder it made.
        b = be.execute(prog, vol)
        pending = b._pending
        if fittings is None:
            fittings = b.fitting_cells
    base = be.volume if pending is not None else None
    if pending is not None:
        vol = stages.apply_pending(vol, pending)
    # B3: the same question the round asks -- a door this program did not hang is not
    # this program's door. Here the "before" is the ground the program was executed
    # against, which is `be.volume` untouched by the pending set.
    return lint.Context.build(vol, plots=plots, network=rnd.network(),
                              region=(X, Z, X + S - 1, Z + S - 1), fittings=fittings,
                              base=base)


def diagnose_entry(ctx, mine: str) -> dict:
    """Why you cannot walk into it, room by room and door by door, for one plot.

        Lifted verbatim out of `scripts/entry_diagnosis.py`, which is where it was written
        and where step 0 of the pre-registration was run. It lives here now because the
        revision loop composes its brief out of exactly these rows, and a second
        implementation of "which rooms cannot be reached" would be a second answer.

        Classes, in the order they are tested:

            walkable         reachable on foot from outdoors
            unattributed     the room's centre is off the plot, or it reads as natural
            open_room        below Context.ENCLOSED, so E011 does not hold it
            no_door          the plot has no door leaf at all (W002, a warning)
            raised_entry     the door stance is reachable only by jumping (E002 allows jumps)
            floor_from_door  E011's own case: the door is fine, the floor is not
        
    """
    from .. import lint
    plot = next(p for p in ctx.plots if p["label"] == mine)
    a, b, c, d = (min(plot["x0"], plot["x1"]), min(plot["z0"], plot["z1"]),
                  max(plot["x0"], plot["x1"]), max(plot["z0"], plot["z1"]))
    # The one walk-only flood, from the Context, seeded from the lane where there is
    # one.
    walk_out = set(ctx.from_outdoors)
    jump_out = set(ctx.from_anywhere)
    doors = []
    for (x, y, z) in ctx.doors:
        if a - 1 <= x <= c + 1 and b - 1 <= z <= d + 1:
            s = ctx.door_stance(x, y, z)
            st = (x, z, s) if s is not None else None
            doors.append({"at": [x, y, z], "stance": s,
                          "walk_from_outside": bool(st and st in walk_out),
                          "jump_from_outside": bool(st and st in jump_out)})
    iw = {tuple(r["bbox"]): r for r in ctx.interior_walk() if r["plot"] == mine}
    rooms = []
    for r in ctx.rooms:
        bb = r["bbox"]
        cx, cz = (bb[0] + bb[3]) / 2, (bb[2] + bb[5]) / 2
        on_plot = a <= cx <= c and b <= cz <= d
        if r.get("plot") != mine and not on_plot:
            continue
        # The room's floor, not every stance in it: the cell on top of its own barrel is
        # not floor. `observe.floor_stances` is the definition and this reads it.
        st = set(map(tuple, r["floor"]))
        w = iw.get(tuple(bb))
        row = {"bbox": bb, "cells": len(st), "enclosure": r.get("enclosure"),
               "made": r.get("made"), "attributed": r.get("plot") == mine,
               "walk_from_outside": len(st & walk_out),
               "jump_from_outside": len(st & jump_out),
               "walk_from_own_door": (w or {}).get("walkable"),
               "fraction_from_own_door": (w or {}).get("fraction")}
        if row["walk_from_outside"]:
            cls = "walkable"
        elif not row["attributed"]:
            cls = "unattributed"
        elif (r.get("enclosure") or 1.0) < lint.Context.ENCLOSED:
            cls = "open_room"
        elif not doors:
            cls = "no_door"
        elif not any(x["walk_from_outside"] for x in doors):
            cls = "raised_entry"
        elif not (w or {}).get("walkable"):
            cls = "floor_from_door"
        else:
            cls = "door_walkable_room_not_from_outside"
        row["class"] = cls
        rooms.append(row)
    rep = lint.lint(ctx)
    plot_findings = [f.code for f in rep.findings
                     if (f.detail or {}).get("plot") == mine
                     or (f.code == "E002" and f.pos and a - 1 <= f.pos[0] <= c + 1
                         and b - 1 <= f.pos[2] <= d + 1)]
    return {"doors": doors, "rooms": rooms,
            "findings_on_plot": sorted(plot_findings),
            # **Is this plot on the lane network at all?** W003's own question, carried
            # out of the report so `entry_lines` can act on it. "Walkable from outdoors"
            # is a walk *from the lane*, and on a plot the lane does not reach it is not
            # a question about the building: every room reads unreachable, the number
            # reads 0.0%, and the answer looks exactly like a sealed house. read what an
            # instrument admits, not only what it rejects -- and this is the third time.
            "off_network": "W003" in plot_findings,
            "lint_counts_whole_region": rep.to_json()["counts"]}


ENTRY_TITLE = "getting in and moving about on foot"


ENTRY_FIX = (
    "Getting in is held to the same standard as the lanes outside: walking, no "
    "jumping. A door whose sill sits a block above the ground you arrive on is a door "
    "nobody can walk to, and a floor laid above the level of its own doorway sill is a "
    "floor nobody can step onto. floor_from_threshold(label) gives you the level the "
    "doorstep is at. Fix the approach, the sill and the steps between storeys so a "
    "person can walk in off the ground and get about once inside.")


def entry_lines(diag: dict) -> list:
    """The lines the linter is silent on, in the linter's own wording.

        Three kinds, exactly as registered, and nothing else:

          * a door reachable from outside only by jumping -- E002's sentence, walking only;
          * a plot with rooms and no door leaf -- W002's sentence, listed with the errors;
          * a room whose floor cannot be walked to from outdoors, enclosed or not.

        The third has no equivalent in the suite at all: E011 asks the same question from
        the building's *own* door and skips a room below `Context.ENCLOSED`.
        
    """
    out = []
    for d in diag["doors"]:
        if d["jump_from_outside"] and not d["walk_from_outside"]:
            x, y, z = d["at"]
            out.append(f"the door at ({x},{y},{z}) cannot be reached on foot from "
                       f"outside -- getting to it needs a jump")
    if not diag["doors"] and any(r["attributed"] for r in diag["rooms"]):
        out.append("this plot has an interior but no door or gate anywhere on it")
    if diag.get("off_network"):
        # One line saying which question cannot be asked here, instead of one line per
        # room saying the answer is no. The library's own `check_door` already words it
        # this way -- "no lane within 36 blocks of this doorway; that is a distance, not
        # a defect" -- and the checker did not.
        n = sum(max(0, r["cells"] - r["walk_from_outside"])
                for r in diag["rooms"] if r["attributed"])
        if n:
            out.append(f"this plot is not on the lane network, so 'walkable from "
                       f"outdoors' cannot be asked of it: {n} floor cell(s) are "
                       f"unreached because there is no lane to walk from, which is a "
                       f"distance and not a defect. The whole-place checks judge it")
        return out
    for r in diag["rooms"]:
        if not r["attributed"]:
            continue
        missed = r["cells"] - r["walk_from_outside"]
        if missed <= 0:
            continue
        b = r["bbox"]
        out.append(f"{missed} of the {r['cells']} floor cells of the room at "
                   f"({b[0]},{b[1]},{b[2]}) cannot be walked to from outdoors")
    return out


def standard_report(ctx, plots: list):
    """The report `settlement_run.py` hands a live pass, over a Context built offline.

    Not `lint.lint(ctx)`. The live loop asks two questions and keeps them apart -- what
    did *this pass* build wrong (the build family, on its own plots) and what did it do
    to *the town* (the place family, everywhere) -- and hands the concatenation to the
    builder. `settlement_run.py:137-147` is the original and this is that, verbatim."""
    from .. import lint
    mine = lint.lint(ctx, family=lint.BUILD).within(plots)
    town = lint.lint(ctx, family=lint.PLACE)
    return lint.Report(mine.findings + town.findings, mine.seconds + town.seconds)


def revision_findings(rep, lines: list, pending: int, name: str,
                      extra: str = "") -> str:
    """The standard findings brief, plus the lines above it, and nothing else.

        `lint.findings_brief` over **every** code, E011 and W011 included -- the brief
        `settlement_run.py` hands a live pass, unchanged, because the question is what the
        ordinary loop does. The header is `walkcheck_round8.cmd_findings`' header, which is
        the only place this loop has ever been driven before.

        Nothing here says how many drafts there are, what is measured, that anything is
        compared, or that a fraction across the build exists. The builder is being told what
        is wrong with the building in front of it, which is all a findings brief has ever
        said.
        
    """
    from .. import lint
    body = lint.findings_brief(
        rep, f"{name}: what the linter found",
        header=(f"Your program ran and placed {pending} blocks. Revise it to fix "
                f"what follows. Your revised program is re-run whole against the "
                f"same ground, so write a complete build, not a patch."))
    if lines:
        n = len(lines)
        tail = ["", f"## On foot -- {ENTRY_TITLE} "
                    f"({n} finding{'s' if n != 1 else ''})", "",
                ENTRY_FIX, ""] + [f"    {ln}" for ln in lines] + [""]
        body = body.rstrip("\n") + "\n" + "\n".join(tail)
    return (body.rstrip("\n") + "\n\n" + extra.strip() + "\n") if extra else body


def _sd(xs: list) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _repair_readout(rnd: Round, subjects: list, rows: dict) -> dict:
    """The registered bars, read off the round's own pre-registration, never recomputed.

        Primary is a count of builds below 50% walk_pct **after** the deletion guard is
        applied, because a builder told a room cannot be reached can delete the room, and a
        build repaired that way is registered as not repaired. Secondary is the pooled
        within-prompt SD. Own lint errors are the region count minus the floor the bare
        volume already carries -- every candidate on this ground shares it, and it is 6.
        
    """
    prereg = ((json.load(open(_pipeline._report_path(rnd))).get("preregistered")
               if os.path.exists(_pipeline._report_path(rnd)) else None) or rnd.preregistered)
    bars = prereg.get("bars") or {}
    floor = int((prereg.get("lint_floor") or {}).get("errors", 6))
    thr = float(bars.get("walk_pct_threshold", 50.0))
    table, by_prompt = [], {}
    for subj in subjects:
        rec = rows.get(subj["id"]) or {}
        drafts = rec.get("drafts") or []
        if not drafts:
            table.append({"id": subj["id"], "status": rec.get("status"),
                          "final": None})
            continue
        d0, dn = drafts[0], drafts[-1]
        m0, mn = d0["measures"], dn["measures"]
        fc0, fcn = m0.get("floor_cells") or 0, mn.get("floor_cells") or 0
        r0, rn = m0.get("rooms") or 0, mn.get("rooms") or 0
        deleted = ((fc0 and (fc0 - fcn) / fc0 > 0.25) or (r0 - rn) > 1)
        walk = mn.get("walk_pct")
        row = {
            "id": subj["id"], "prompt": subj["wave"], "status": rec.get("status"),
            "drafts": len(drafts), "final_draft": dn["draft"],
            "walk_pct": [d["measures"].get("walk_pct") for d in drafts],
            "own_door_pct": [d.get("own_door_pct") for d in drafts],
            "rooms": [d["measures"].get("rooms") for d in drafts],
            "floor_cells": [d["measures"].get("floor_cells") for d in drafts],
            # The registered count: the whole-region number minus the floor the bare
            # volume already carries. `own_lint_errors_scoped` is the same question
            # asked the way `settlement_run` asks it -- the build family on this plot --
            # and is the better number; it is reported beside the registered one and
            # never instead of it, because the bar was written against the subtraction.
            "own_lint_errors": [max(0, (d["measures"].get("lint_errors") or 0) - floor)
                                for d in drafts],
            "own_lint_errors_scoped": [d.get("lint_errors_brief") for d in drafts],
            "entry_lines": [d["entry_lines"] for d in drafts],
            "blocks": [d["measures"].get("blocks") for d in drafts],
            "program_lines": [d["lines"] for d in drafts],
            "articulation": [d["measures"].get("articulation") for d in drafts],
            "repaired_by_deletion": bool(deleted),
            "final": walk,
            # The guard, applied where it was registered to apply: in the primary count.
            # A build below the threshold is below it; a build above it that got there
            # by removing interior is counted as if it were below.
            "below_threshold": bool(walk is None or walk < thr or deleted),
        }
        table.append(row)
        if walk is not None:
            by_prompt.setdefault(subj["wave"], []).append(walk)
    # "Finished" is the loop's own stop rule -- clean, or out of revisions -- and not
    # "has a walk_pct". A candidate whose last draft leaves no room the linter will
    # attribute measures None and is *counted* by the primary as below the threshold;
    # reading it as unfinished would suppress the branch on exactly the candidates the
    # bar was written to catch. The bar itself is untouched: this decides when there is
    # a result to read, never what it says.
    done = [r for r in table if r.get("status") in ("clean", "exhausted")]
    below = [r for r in table if r.get("below_threshold")]
    resid = []
    for vals in by_prompt.values():
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            resid += [(v - m) ** 2 for v in vals]
    dof = sum(len(v) - 1 for v in by_prompt.values() if len(v) > 1)
    pooled = (sum(resid) / dof) ** 0.5 if dof else None
    lint_clean = [r for r in table
                  if r.get("own_lint_errors") and r["own_lint_errors"][-1] == 0]
    return {
        "threshold": thr, "lint_floor": floor,
        "complete": len(done), "of": len(table),
        "primary": {
            "below_threshold": len(below),
            "of": len(table),
            "ids": [r["id"] for r in below],
            "repaired_by_deletion": [r["id"] for r in table
                                     if r.get("repaired_by_deletion")],
            "bars": bars.get("primary"),
            "branch": (None if len(done) < len(table) else
                       "closes" if len(below) <= int(bars.get("closes_at_or_below", 3))
                       else "fails" if len(below) >= int(bars.get("fails_at_or_above", 7))
                       else "partial"),
        },
        "secondary": {
            "pooled_within_prompt_sd": round(pooled, 2) if pooled is not None else None,
            "per_prompt_sd": {k: (round(_sd(v), 2) if _sd(v) is not None else None)
                              for k, v in sorted(by_prompt.items())},
            "bars": bars.get("secondary"),
        },
        "own_lint_errors_zero": {"n": len(lint_clean), "of": len(table),
                                 "bar": bars.get("own_lint_errors_zero")},
        "table": table,
    }


def stage_arm_measures(rnd: Round, be, results: dict) -> dict:
    """Every arm candidate, by the instruments that never see an image.

        The same `measure_program` the selection candidates went through, so arm A -- which
        is those four candidates, reused rather than rebuilt -- is directly comparable
        without re-measuring it under a second definition.
        
    """
    a = rnd.arms
    if not a:
        return {"note": "this round has no arms to measure"}
    plots = json.load(open(rnd.rel("plots.json")))
    want = set(a["plots"])
    mine = {p["label"] for p in plots if p["label"] in want}
    margin = int((a.get("measures") or {}).get("variety_margin", 6))
    out = {}
    for cid, sub in _pipeline._arm_ids(rnd):
        out[cid] = measure_program(rnd, be, _pipeline._arm_program(rnd, sub), mine,
                                   margin=margin)
        if out[cid]["status"] != "measured":
            continue
        record("check", name=f"arm_measures/{cid}", settlement=rnd.name,
               errors=out[cid]["lint_errors"], warnings=out[cid]["lint_warnings"],
               seconds=out[cid]["seconds"], blocks=out[cid]["blocks"],
               walk_pct=out[cid]["walk_pct"])
    # Arm A is free: the four selection candidates are independent blind builds of this
    # brief. Read off the measures the selection round already wrote where they are
    # there, so nothing is rebuilt and nothing is re-measured under a second definition.
    for cid, prog in (a.get("control") or {}).items():
        out[f"A/{cid}"] = measure_program(rnd, be, os.path.join(_pipeline.ROOT, prog), mine,
                                          margin=margin)
    path = os.path.join(_pipeline._arm_dir(rnd), "measures.json")
    os.makedirs(_pipeline._arm_dir(rnd), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return {**out, "written": path}


def region_diff(pre, built, x0: int, z0: int, x1: int, z1: int) -> list:
    """[(x, y, z, pre_state, built_state)] where two cached volumes disagree.

        The undo log for a readout that has to borrow the standing world: the region is
        written to `pre_state` before the first candidate and back to `built_state` after
        the last, so the town ends as it began. Step 3's readout is where this comes from
        and `scripts/step3_render.py` still calls it.
        
    """
    import numpy as np
    names_p = np.array(pre.palette, dtype=object)
    names_b = np.array(built.palette, dtype=object)
    y0 = max(pre.y0, built.y0)
    y1 = min(pre.y0 + pre.shape[1], built.y0 + built.shape[1])
    sb = built.codes[x0 - built.x0:x1 + 1 - built.x0, y0 - built.y0:y1 - built.y0,
                     z0 - built.z0:z1 + 1 - built.z0]
    sp = pre.codes[x0 - pre.x0:x1 + 1 - pre.x0, y0 - pre.y0:y1 - pre.y0,
                   z0 - pre.z0:z1 + 1 - pre.z0]
    nb, np_ = names_b[sb], names_p[sp]
    out = []
    for (dx, dy, dz) in np.argwhere(nb != np_):
        out.append((x0 + int(dx), y0 + int(dy), z0 + int(dz)),)
    return [(x, y, z, str(np_[x - x0, y - y0, z - z0]),
             str(nb[x - x0, y - y0, z - z0])) for (x, y, z) in out]


def _bar_value(text) -> tuple:
    """(comparator, value) from a registered bar.

        A bar is written as a human writes one -- `">= 90"`, `"0 findings"`, `"<= 15%"`,
        `"<= 2 x waves = 10"` -- because that is what is on disk in the rounds that already
        ran and a pre-registration is not ours to rewrite. The comparator is the prefix; the
        value is the number after the last `=` where the text works up to one, and otherwise
        the first number in it. A bar given as `{"comparator": .., "value": ..}` is taken as
        written, which is what a new round should do.
        
    """
    import re
    if isinstance(text, dict) and "value" in text:
        return text.get("comparator", ">="), float(text["value"])
    s = str(text).strip()
    op = "=="
    for cand in (">=", "<=", "==", ">", "<", "="):
        if s.startswith(cand):
            op = ">=" if cand == "=" else cand
            s = s[len(cand):].strip()
            break
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.split("=")[-1] if "=" in s else s)
    if not nums:
        # which is the one thing the round exists not to do. The comparator is `in` and
        # the measure hands back the band it is read against, so the number is still on
        # the record and still was not ours.
        return "in", None
    return op, float(nums[0])


def _meets(op: str, got, bar) -> bool | None:
    if got is None:
        return None
    if op == "in":
        if not bar:
            return None
        lo, hi = bar
        return float(lo) <= float(got) <= float(hi)
    return {">=": got >= bar, "<=": got <= bar, ">": got > bar, "<": got < bar,
            "==": got == bar}[op]


MEASURE_ALIASES = {"e002_walk_only": "e002_from_the_lane",
                   "e002_walk_only_from_the_lane": "e002_from_the_lane",
                   # the measure is the same one.
                   "tokens_whole_effort": "tokens_whole_round"}


def _m_walk_from_outdoors_pct(rnd, be, results, bar) -> dict:
    """The interior floor a person can walk to from outdoors, over the finished town.

        `scripts/walk_fraction.py`'s own `measure`, imported rather than reimplemented:
        there is one definition of this number and it is the one every round since 7 has
        been read against.
        
    """
    from .. import walk_fraction as mod
    # A4: the context this readout has already built, rather than a fifth one over the
    # same volume. The two differed only in the region's last column and in whether the
    # pre-build cache was handed over, and neither reaches this number.
    row, _rep = mod.measure(rnd.name, context=_town_context(rnd, be))
    return {"got": row["from_outdoors_walking_pct"],
            "rooms": row["rooms_on_plots"], "floor_cells": row["floor_cells"],
            "rooms_at_zero": row["rooms_zero_walkable_from_outdoors"],
            "jumping_allowed_pct": row["from_outdoors_jumping_pct"],
            "from_own_door_pct": row["from_own_door_walking_pct"],
            "command": f"$PY -m ethoslm.walk_fraction {rnd.name}",
            # The ruler, beside the number.
            "walk_model": row["walk_model"],
            "fitting_registry": row["fitting_registry"],
            "model": f"observe.STAIR_CLEAR = {_stair_clear()} half-blocks"}


def _stair_clear() -> int:
    from .. import observe
    return observe.STAIR_CLEAR


#: The last town context built, by (state directory, the built volume's digest). One
#: entry: a readout reads one place, and holding two would hold two cities in memory.
_TOWN_CONTEXT: dict = {}


def _built_digest(rnd) -> str:
    """A digest of the volume the readout is being read off, or "" where there is none.

        The key a memo has to be keyed on. A stage that re-built the town between two
        measures -- or a `--measure` re-read against a volume that has since been written
        again -- must get a fresh context, and the file's own content is what says so.
        
    """
    import hashlib
    p = rnd.rel("world_built.npz")
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _town_context(rnd, be):
    """The whole place as the linter sees it, built once per readout. A4.

        **It was built four times, and `walk_fraction` built a fifth.** Every measure that
        asks a question about the finished town -- E002 from the lane, own lint errors, the
        place read -- built its own `lint.Context` over the same volume, and building one is
        the expensive half of a readout: a minute and a half on a walled town, where the
        measure that follows it is under a second. Nothing between them changes the volume,
        so this is one memo on the built volume's digest, and `walk_fraction.measure` takes
        the same object rather than making a fifth.
        
    """
    key = (rnd.state, _built_digest(rnd))
    got = _TOWN_CONTEXT.get(key)
    if got is not None:
        return got
    _TOWN_CONTEXT.clear()
    got = _TOWN_CONTEXT[key] = _build_town_context(rnd, be)
    return got


def _build_town_context(rnd, be):
    from .. import lint, offline, settlement
    from ..circulate import Network
    vol = offline.load_volume(rnd.rel("world_built.npz"))
    # A1: with `y0` per part, so E002, E003 and E011 are asked about the buildings and
    # not about the ground under them. See `lint.Context.room_owner`.
    plots = settlement.registry_with_floors(rnd.state)
    net = (Network.load(rnd.rel("network.json"))
           if os.path.exists(rnd.rel("network.json")) else None)
    s = rnd.site or json.load(open(rnd.rel("site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    return lint.Context.build(vol, plots=plots, network=net,
                              region=(X, Z, X + S - 1, Z + S - 1),
                              base=_prebuild(rnd)), net


def _prebuild(rnd):
    """B3: `lint.Context.build` takes this to tell the build's own
        doors from the ones that were already in the ground. It is the round's own base
        volume -- the cached world the run was planned against -- and it is missing only
        where a round has no cache, which is where the correction cannot be made and the
        old reading stands.
        
    """
    from .. import offline
    p = rnd.rel(rnd.base_volume)
    try:
        return offline.load_volume(p) if os.path.exists(p) else None
    except Exception:                    # noqa: BLE001 -- a reading, not a build
        return None


def _m_e002_from_the_lane(rnd, be, results, bar) -> dict:
    """E002 over the finished town: doors nobody can walk to from outdoors."""
    from .. import lint
    ctx, net = _town_context(rnd, be)
    rep = lint.lint(ctx)
    e002 = [f for f in rep.findings if f.code == "E002"]
    return {"got": len(e002), "findings": [f.message for f in e002][:8],
            "from_the_lane": _doors_from_the_lane(ctx, net,
                                                  plan_entries(rnd.plan()))}


def _m_own_lint_errors(rnd, be, results, bar) -> dict:
    """The build family on each wave's own plots -- how `settlement_run` scopes a pass.

        A place built from typed parts has no build waves in its config: its waves are the
        plan's own -- the walls, the squares, then a quarter at a time -- and they are in
        `parts.json`. Reading `rnd.waves` alone on such a round would report zero errors
        because there was nothing to iterate, which is the worst possible way for a measure
        to pass.
        
    """
    from .. import lint, place
    ctx, _net = _town_context(rnd, be)
    plots = json.load(open(rnd.rel("plots.json")))
    waves = [(w["name"], list(w.get("plots") or [])) for w in rnd.waves]
    if not waves:
        waves = [(name, [p["name"] for p in group])
                 for name, group in place.part_waves(rnd.parts())]
    # One pass of the build family over the whole context, then each wave's own plots
    # read off it: `within` is a filter and the findings are the same, and the
    # concentric run spent fifty minutes computing them sixty-seven times.
    build = lint.lint(ctx, family=lint.BUILD)
    per = {}
    for name, labels in waves:
        mine = [p for p in plots if p["label"] in set(labels)]
        got = build.within(mine)
        per[name] = sorted(f.code for f in got.findings if f.code.startswith("E"))
    # The bar was scoped to the waves' own ground and the reserved-threshold check is a
    # place check: the concentric run read 8 own errors while 219 of its 250 doorways
    # were obstructed. Read place-wide here, by the same lint the place read runs, and
    # added to the bar's number.
    place_rep = lint.lint(ctx, family=lint.PLACE)
    e008 = [f for f in place_rep.findings if f.code == "E008"]
    return {"got": sum(len(v) for v in per.values()) + len(e008),
            "per_wave": per, "own_ground": sum(len(v) for v in per.values()),
            "e008_place_wide": len(e008),
            "e008_examples": [f.message for f in e008[:6]],
            "scoped_by": ("config waves" if rnd.waves else "the plan's own waves")
                         + ", plus E008 place-wide"}


def _m_builder_calls(rnd, be, results, bar) -> dict:
    """`build_request_<i>.json` files on disk: one per call a builder was asked for."""
    arm = rnd.flags.get("arm", rnd.name)
    calls, bounces = {}, {}
    for w in rnd.waves:
        b = rnd.rel("arms", arm, w["name"], "builds")
        listing = os.listdir(b) if os.path.isdir(b) else []
        calls[w["name"]] = len([f for f in listing
                                if f.startswith("build_request_")])
        bounces[w["name"]] = len([f for f in listing if ".crashed" in f])
    return {"got": sum(calls.values()), "waves": len(rnd.waves),
            "per_wave": calls, "bounces": bounces}


def _m_program_lines(rnd, be, results, bar) -> dict:
    """Lines of each adopted wave program, meaned over the waves."""
    per = [len(open(rnd.rel(f"{w['name']}.py")).read().splitlines())
           for w in rnd.waves if os.path.exists(rnd.rel(f"{w['name']}.py"))]
    return {"got": round(sum(per) / len(per), 1) if per else None, "per_wave": per}


def _m_writes_covered_over(rnd, be, results, bar) -> dict:
    """(positions written - blocks standing) / positions written, over the programs."""
    import contextlib
    import io
    writes = standing = 0
    per = {}
    for w in rnd.waves:
        p = rnd.rel(f"{w['name']}.py")
        if not os.path.exists(p):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            b = be.execute(p, be.volume)
        writes += b.writes
        standing += len(b._pending)
        per[w["name"]] = {"writes": b.writes, "standing": len(b._pending)}
    return {"got": round(100 * (writes - standing) / writes, 1) if writes else None,
            "writes": writes, "standing": standing, "per_wave": per}


def _instance_rows(rnd, results) -> list:
    got = results.get("types")
    if not got:
        p = os.path.join(_pipeline._type_out(rnd), "instances.json")
        got = json.load(open(p)) if os.path.exists(p) else {}
    rows = []
    for name, rec in sorted(got.items()):
        if not isinstance(rec, dict) or "instances" not in rec:
            continue
        for iid, r in sorted(rec["instances"].items()):
            rows.append({"type": name, "id": iid, **r})
    return rows


def _m_instances_lint_zero(rnd, be, results, bar) -> dict:
    rows = _instance_rows(rnd, results)
    ok = [r["id"] for r in rows if r.get("own_lint_errors") == 0]
    return {"got": len(ok), "of": len(rows), "clean": ok,
            "not_clean": {r["id"]: r.get("own_lint_errors", r.get("status"))
                          for r in rows if r["id"] not in ok}}


def _m_instances_walkable(rnd, be, results, bar) -> dict:
    """Instances at or above the percentage the bar names in its own `pct`."""
    pct = float((bar or {}).get("pct", 90))
    rows = _instance_rows(rnd, results)
    ok = [r["id"] for r in rows
          if r.get("walk_pct") is not None and r["walk_pct"] >= pct]
    return {"got": len(ok), "of": len(rows), "at_or_above": pct,
            "walk_pct": {r["id"]: r.get("walk_pct") for r in rows}}


def _m_tokens_per_building(rnd, be, results, bar) -> dict:
    """True tokens per building: the builder calls' own totals over the instances."""
    p = os.path.join(_pipeline._type_out(rnd), (bar or {}).get("tokens", "tokens.json"))
    if not os.path.exists(p):
        return {"got": None, "note": f"no {p} -- the builders' reported totals are "
                                     f"the only source and nothing has written them"}
    doc = json.load(open(p))
    calls = {k: v for k, v in doc.items()
             if not k.startswith("_") and isinstance(v, (int, float))}
    rows = _instance_rows(rnd, results)
    total = sum(calls.values())
    return {"got": round(total / len(rows)) if rows else None,
            "call_tokens": calls, "total": total, "buildings": len(rows)}


def _parts_record(rnd, results) -> dict:
    got = results.get("parts")
    if not got:
        p = rnd.rel("parts.json")
        got = json.load(open(p)) if os.path.exists(p) else {}
    return got or {}


def _m_structures(rnd, be, results, bar) -> dict:
    """How many things stand here. Plot leaves, because a wall is not a structure."""
    parts = rnd.parts()
    kinds: dict = {}
    for p in parts:
        kinds[p.get("kind", "plot")] = kinds.get(p.get("kind", "plot"), 0) + 1
    rows = [r for w in _parts_record(rnd, results).get("waves", [])
            for r in w["parts"]]
    built = {r["part"] for r in rows if r["status"] == "built"}
    # ...and what actually stands. A type that refuses -- "no massing stood on this pad"
    # -- runs to the end and places the ground the library prepared, so `status` says
    # built and nothing stands there. Both numbers, always.
    stood = {r["part"] for r in rows if r.get("stood", r["status"] == "built")}
    # The band the spec produced from the sentence, so a bar registered as "within the
    # scaled band the spec produced" has a band to be read against and the number is on
    # the record beside the count.
    spec = rnd.place_spec()
    return {"got": sum(1 for p in parts if p.get("kind", "plot") == "plot"),
            "band": list(spec["size_band"]) if spec else None,
            "planned_for": spec["structures"] if spec else None,
            "leaves": len(parts), "kinds": kinds, "built": len(built),
            "stood": len(stood),
            "refused": sorted({r["part"]: r.get("type_said") for r in rows
                               if r["part"] not in stood}.items()),
            "not_built": sorted({p["name"] for p in parts} - built)}


def _m_instantiation(rnd, be, results, bar) -> dict:
    """A part refused *before* siting."""
    parts = rnd.parts()
    rows = [r for w in _parts_record(rnd, results).get("waves", []) for r in w["parts"]]
    stood = {r["part"] for r in rows if r.get("stood", r["status"] == "built")}
    refused = [{"part": r["part"], "type": r.get("type"), "status": r["status"],
                "why": r.get("type_said") or r.get("error", "")[:200]}
               for r in rows if r["part"] not in stood]
    out = {"got": round(100 * len(stood) / len(parts), 1) if parts else None,
           "stood": len(stood), "leaves": len(parts),
           "refused_before_siting": sum(1 for r in rows if r["status"] == "refused"),
           "refusals": refused,
           "never_reached": sorted({p["name"] for p in parts}
                                   - {r["part"] for r in rows})}
    if rows and not any("stood" in r for r in rows):
        # The driver threw the type's answer away and read 35 of 35 built where 13
        # stood. A record that cannot answer this question says so rather than answering
        # 100.
        out["got"] = None
        out["unreadable"] = ("this parts record carries no `stood`: it was written "
                             "before the driver kept what the type said, so what "
                             "stands cannot be read off it")
    return out


def _m_hand_programs(rnd, be, results, bar) -> dict:
    """Programs in this build that are not an instance of a committed type.

        Part C's whole claim, as a number. Every part is composed by
        `place.instantiate_part` from a file under `types/`; a part that could not be
        instantiated is reported and left empty, and this counts anything that was built
        any other way. It is zero by construction and is measured anyway, because "by
        construction" is what this project checks rather than asserts.
        
    """
    rec = _parts_record(rnd, results)
    rows = [r for w in rec.get("waves", []) for r in w["parts"]]
    hand = [r["part"] for r in rows
            if r["status"] == "built" and not r.get("type")]
    left = [{"part": r["part"], "why": r["status"]} for r in rows
            if r["status"] != "built"]
    return {"got": len(hand), "hand_built": hand, "instances": len(rows) - len(left),
            "left_empty": left}


def _m_enclosure(rnd, be, results, bar) -> dict:
    """Is it a walled district: one closed wall, one gate, everything inside reachable?

        Three questions, and each is answered off what is on disk rather than off the plan's
        intention. The wall's columns come from the edge parts; the gate from the point part
        whose type declares `PASSAGE`; the reachability from the same walk-only flood every
        from-outdoors number in this project uses, seeded from the lane.
        
    """
    from .. import circulate, lint, observe
    parts = rnd.parts()
    passage = {p.get("type") for p in parts
               if p.get("type")
               and os.path.exists(os.path.join(_pipeline.ROOT, "types", f"{p['type']}.py"))
               and _pipeline.load_type(os.path.join(_pipeline.ROOT, "types", f"{p['type']}.py"))["passage"]}
    routing = circulate.parts_to_routing(parts, passage=passage)
    wall, gate = routing["obstacles"], routing["passable"]
    if not wall:
        return {"got": None, "note": "this place has no edge part in it, so there is "
                                     "nothing enclosing anything"}
    built = rnd.rel("world_built.npz")
    vol = offline.load_volume(built) if os.path.exists(built) else be.volume
    net = rnd.network()
    nav = observe.Nav(vol)
    lane = lint.lane_stances(nav, net)
    outside = [s for s in lane if (s[0], s[1]) not in wall and (s[0], s[1]) not in gate]
    reach = set(nav.flood(outside, max_jumps=0)) if outside else set()
    inner = [p for p in parts if p.get("kind", "plot") == "plot"]
    got = []
    for p in inner:
        r = _pipeline.part_rect(p)
        cx, cz = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
        t = net.threshold(p["name"]) if net else None
        cell = (t.door[0], t.door[2]) if t else (cx, cz)
        s = nav.stance_near(cell[0], cell[1], (t.y + 1) if t else None, tol=3) \
            if t else None
        got.append({"part": p["name"],
                    "reachable": bool(s is not None and (cell[0], cell[1], s) in reach)})
    crossings = sorted({(x, z) for (x, z, _y) in (net.surface() if net else [])}
                       & wall)
    ok = (not crossings) and all(g["reachable"] for g in got) and bool(gate)
    # One and zero rather than "holds" and "fails": a bar carries a number, and this is
    # a predicate the driver has to be able to compare. The word is beside it.
    return {"got": 1 if ok else 0, "holds": bool(ok),
            "wall_columns": len(wall), "gate_cells": sorted(gate),
            "lane_on_the_wall": crossings[:8],
            "plots_reachable": sum(1 for g in got if g["reachable"]),
            "plots": len(got),
            "unreachable": [g["part"] for g in got if not g["reachable"]]}


def _m_tokens_whole_round(rnd, be, results, bar) -> dict:
    """There is no API key in this project, so the honest source is what each isolated
        subagent says it used. The file is named by the bar so the harness's own
        under-reading estimate cannot be quietly substituted for it.
        
    """
    p = os.path.join(_pipeline._type_out(rnd) if rnd.types else rnd.state,
                     (bar or {}).get("tokens", "tokens.json"))
    if not os.path.exists(p):
        return {"got": None, "note": f"no {p} -- the callers' reported totals are the "
                                     f"only source and nothing has written them"}
    doc = json.load(open(p))
    calls = {k: v for k, v in doc.items()
             if not k.startswith("_") and isinstance(v, (int, float))}
    return {"got": round(sum(calls.values()) / 1e6, 3), "unit": "millions of tokens",
            "calls": calls, "largest": max(calls.items(), key=lambda kv: kv[1])
            if calls else None}


def _m_whole_place_lint_seconds(rnd, be, results, bar) -> dict:
    """How long the one whole-place lint took. A6, reported and not barred here."""
    got = results.get("lint") or {}
    if not got:
        p = _pipeline._report_path(rnd)
        got = (json.load(open(p)).get("results", {}).get("lint") or {}) \
            if os.path.exists(p) else {}
    return {"got": got.get("seconds"), "errors": got.get("errors"),
            "warnings": got.get("warnings")}


def _m_within_type_variation(rnd, be, results, bar) -> dict:
    """Types whose instances differ in **massing** and not only in trim.

        The bar names two numbers because one of them is easy: a seed that changes the
        ridge and not the footprint has changed the roof pitch, and a seed that changes
        neither has decorated. Both spreads have to clear the margin the bar carries.
        
    """
    ridge_by = float((bar or {}).get("ridge", 2))
    foot_by = float((bar or {}).get("footprint", 2))
    rows = _instance_rows(rnd, results)
    per: dict = {}
    for r in rows:
        per.setdefault(r["type"], []).append(r)
    out, met = {}, []
    for name, rs in sorted(per.items()):
        ridges = [r.get("ridge_height") for r in rs if r.get("ridge_height")]
        foots = [r["footprint"][0] * r["footprint"][1] for r in rs
                 if r.get("footprint")]
        dr = (max(ridges) - min(ridges)) if len(ridges) > 1 else 0
        df = (max(foots) - min(foots)) if len(foots) > 1 else 0
        ok = dr >= ridge_by and df >= foot_by
        out[name] = {"instances": len(rs), "ridge_heights": ridges,
                     "ridge_spread": dr, "footprint_cells": foots,
                     "footprint_spread": df, "met": ok}
        met.append(ok)
    return {"got": sum(met), "of": len(per), "margins":
            {"ridge": ridge_by, "footprint": foot_by}, "per_type": out}


def _m_building_calls_per_type(rnd, be, results, bar) -> dict:
    """Types whose source calls `building()`.

        Reported as a bar and read as information: a type that composes a raised-floor hall
        out of primitives because the shell's parameters do not reach it has told us where
        the next primitive goes, which is worth more than the row.
        
    """
    import ast
    out = {}
    for spec in _pipeline._types(rnd):
        p = _pipeline._type_file(rnd, spec)
        if not os.path.exists(p):
            out[spec["name"]] = None
            continue
        n = 0
        for node in ast.walk(ast.parse(open(p).read())):
            if isinstance(node, ast.Call):
                f = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                n += f == "building"
        out[spec["name"]] = n
    return {"got": sum(1 for v in out.values() if v), "of": len(out),
            "calls_per_type": out}


def _m_place_read(rnd, be, results, bar) -> dict:
    """A6: the built place against the sentence's own spec. 1 or 0, and what failed."""
    from .. import placeread
    got = results.get("place_check")
    if not got or "clauses" not in got:
        p = rnd.rel("place_read.json")
        if os.path.exists(p):
            got = json.load(open(p))
        else:
            spec = rnd.place_spec()
            if spec is None:
                return {"got": None, "note": "this round has no place.json, so there "
                                             "is no spec to read the place against"}
            pr = rnd.rel("parts.json")
            built, base = built_volumes(rnd)
            got = placeread.read(
                spec, rnd.plan(),
                json.load(open(pr)) if os.path.exists(pr) else {},
                voice=rnd.voice_name() or None, site=_pipeline.settlement_site(rnd),
                built=built, base=base, plateau=_pipeline.plateau_record(rnd))
    return {"got": got["got"], "holds": got["holds"], "failed": got["failed"],
            "clauses": {c["clause"]: c["says"] for c in got["clauses"]},
            "band": got.get("band"), "sentence": got.get("sentence")}


def _m_site_chosen(rnd, be, results, bar) -> dict:
    """A3: was the ground chosen by the system, and is the record of why on disk?

        1 when a site search ran, chose, and recorded its top three with their scores, and
        the round's config named no site. 0 when a human wrote a coordinate. Registered as a
        bar because "no human chooses anything" is the round's claim and a claim nobody
        measures is a comment.
        
    """
    got = rnd.site_search()
    if not got:
        return {"got": 0, "note": "no site_search.json: nothing searched for this "
                                  "ground", "config_site": rnd.site or None}
    ok = bool(got.get("chosen")) and len(got.get("top") or []) >= 3 and not rnd.site
    return {"got": 1 if ok else 0,
            "chosen": got.get("chosen", {}).get("origin"),
            "size": got.get("chosen", {}).get("size"),
            "attempt": got.get("attempt"), "recorded": len(got.get("top") or []),
            "config_named_a_site": bool(rnd.site),
            "terraform": got.get("terraform"),
            "size_band_dropped": got.get("size_band_dropped"),
            "top": [{"rank": t["rank"], "origin": [t["x"], t["z"]],
                     "score": t["excess"]["total"], "relief": t["relief"],
                     "water_pct": t["water_pct"], "forest_pct": t["forest_pct"],
                     "gravity_pct": t["gravity_pct"],
                     "plateau_relief": t["plateau"]["relief"]}
                    for t in got.get("top", [])[:3]]}


#: Five per cent of the site. The layout takes least widths from the ring with slack,
#: and that ring is the belt.
BELT_SHARE_TOLERANCE = 0.05

#: **Registered.** How many distinct palettes must stand in a place whose spec gave its
#: rings their own voices for the place to read as quarters and not as one town.
PALETTES_STANDING_MIN = 3


def _m_concentric(rnd, be, results, bar) -> dict:
    """The five things a concentric place is held to beyond the nine bars, read off the
    plan, what stood and the built volume, each against its registered threshold.
    reported, not barred. `got` is 1 when every clause holds.

          belt        the outermost ring's share of the site against its declared share
                      less `BELT_SHARE_TOLERANCE`;
          coverage    every ring's districts cover at least `placeplan.RING_COVERAGE` of it;
          centred     the compound at the centre stands within `placeplan.CENTRED_TOLERANCE`
                      of the site's centre;
          palettes    at least `PALETTES_STANDING_MIN` distinct voices among the parts that
                      stood;
          core_water  no column of the compound's ground is water in the built volume.
        
    """
    from .. import observe, placeplan, spec as spec_mod
    spec = rnd.place_spec()
    plan = rnd.plan()
    if not spec or not plan:
        return {"got": None, "note": "no place spec or no plan to read"}
    layout = plan.get("layout") or {}
    site = _pipeline.settlement_site(rnd) or {}
    out: dict = {"registered": {"BELT_SHARE_TOLERANCE": BELT_SHARE_TOLERANCE,
                                "RING_COVERAGE": placeplan.RING_COVERAGE,
                                "CENTRED_TOLERANCE": placeplan.CENTRED_TOLERANCE,
                                "PALETTES_STANDING_MIN": PALETTES_STANDING_MIN,
                                "core_water_columns": 0}}
    holds = []
    rings = layout.get("rings") or []
    if not rings:
        out["note"] = "the plan carries no layout record: the place was not laid out by arithmetic"
        return {"got": None, **out}
    belt = rings[-1]
    out["belt"] = {"ring": belt["name"], "asked": belt["share_asked"],
                   "got": belt["share_got"],
                   "holds": belt["share_got"] >= belt["share_asked"] - BELT_SHARE_TOLERANCE}
    holds.append(out["belt"]["holds"])
    out["coverage"] = {r["name"]: {"got": r["coverage"],
                                   "holds": r["coverage"] >= placeplan.RING_COVERAGE}
                       for r in rings}
    holds.append(all(v["holds"] for v in out["coverage"].values()))
    comps = placeplan.compound_rects(plan)
    core = spec_mod.core(spec)
    if site and core and core["name"] in comps:
        X, Z, S = site["origin"][0], site["origin"][1], int(site["size"])
        r = comps[core["name"]]
        cx, cz = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
        off = max(abs(cx - (X + S // 2)), abs(cz - (Z + S // 2)))
        out["centred"] = {"compound": core["name"], "centre": [cx, cz],
                          "site_centre": [X + S // 2, Z + S // 2], "offset": off,
                          "holds": off <= placeplan.CENTRED_TOLERANCE}
    else:
        out["centred"] = {"holds": False, "note": "no compound at the centre to read"}
    holds.append(out["centred"]["holds"])
    rec = _parts_record(rnd, results)
    voices = sorted({str(row.get("voice")) for w in rec.get("waves", [])
                     for row in w.get("parts", [])
                     if row.get("stood", row.get("status") == "built") and row.get("voice")})
    out["palettes"] = {"standing": voices, "count": len(voices),
                       "holds": len(voices) >= PALETTES_STANDING_MIN}
    holds.append(out["palettes"]["holds"])
    built, _base = built_volumes(rnd)
    if built is not None and core and core["name"] in comps:
        import numpy as np
        r = comps[core["name"]]
        _h, wet = observe.ground_heights(built)
        i0, j0 = r[0] - built.x0, r[1] - built.z0
        i1, j1 = r[2] - built.x0 + 1, r[3] - built.z0 + 1
        w = np.asarray(wet)[max(0, i0):i1, max(0, j0):j1]
        n = int(w.sum())
        out["core_water"] = {"columns": n, "of": int(w.size), "holds": n == 0}
    else:
        out["core_water"] = {"holds": False, "note": "no built volume to read"}
    holds.append(out["core_water"]["holds"])
    out["got"] = 1 if all(holds) else 0
    out["failed"] = [k for k, v in (("belt", out["belt"]), ("centred", out["centred"]),
                                    ("palettes", out["palettes"]),
                                    ("core_water", out["core_water"]))
                     if not v["holds"]] + [f"coverage/{k}" for k, v in out["coverage"].items()
                                           if not v["holds"]]
    return out


def _podium_of(rnd):
    """The level the compound's plateau was cut at, off the plateau record in
    whichever shape it carries it, else the layout's terrace record."""
    rec = _pipeline.plateau_record(rnd) or {}
    y = rec.get("y")
    if y is None:
        y = (rec.get("plateau") or {}).get("y") if isinstance(rec.get("plateau"), dict) else None
    if y is None:
        terrace = rec.get("terrace") or ((rnd.plan() or {}).get("layout") or {}).get("terrace")
        y = (terrace or {}).get("podium")
    return None if y is None else int(y)


def _wall_levels_of(rnd) -> list:
    """Every edge part's sited level against its ring's, off parts.json and plan.json."""
    import re
    rec = _parts_record(rnd, {})
    plan = rnd.plan() or {}
    levels = {q["name"]: q.get("level") for q in _pipeline.plan_parts(plan)
              if q.get("kind") == "edge"}
    out = []
    for w in rec.get("waves", []):
        for row in w.get("parts", []):
            if row.get("kind") != "edge":
                continue
            reason = str(row.get("sited") or "")
            m = re.search(r"a level per segment at y=([-\d,]+)", reason)
            segs = ([int(v) for v in m.group(1).split(",")] if m else
                    [int(row["floor_y"])] if row.get("floor_y") is not None else [])
            one = len(set(segs)) == 1
            want = levels.get(row["part"])
            out.append({"part": row["part"], "type": row.get("type"),
                        "floor_y": row.get("floor_y"), "ring_level": want,
                        "segments": len(segs), "one_level": one,
                        "at_ring_level": (one and want is not None
                                          and segs and segs[0] == int(want))})
    return out


def _m_ground(rnd, be, results, bar) -> dict:
    """The five things designed ground is held to beyond the nine bars, each against its
    registered threshold. reported, not barred. `got` is 1 when every clause holds.

          biome       the chosen site's cells in the setting's preferred biomes at
                      `spec.BIOME_SHARE` or more (holds trivially where no biome was asked);
          decks       zero parts sited on decks where the setting's water is not `some`;
          walls       every ring wall sited at one level, and that level its ring's;
          podium      the compound's plateau a step above the innermost ring's level;
          e008        reserved thresholds obstructed place-wide, at 0.
        
    """
    from .. import lint, spec as spec_mod
    spec = rnd.place_spec() or {}
    setting = spec.get("setting") or {}
    out: dict = {"registered": {"BIOME_SHARE": spec_mod.BIOME_SHARE,
                                "TERRACE_STEP": __import__("ethoslm.placeplan",
                                                           fromlist=["x"]).TERRACE_STEP,
                                "decks": 0, "e008": 0}}
    holds = []
    # biome
    search = rnd.site_search() or {}
    chosen = (search.get("chosen") or {}).get("measures") or {}
    want = setting.get("biome")
    census = chosen.get("biome") or {}
    # The second run's first registered change: where the layout designs the ground the
    # search read the share over the core and the inner rings, and that census is on the
    # chosen site's record as `inner`. The measure reads what the search held the site
    # to; the whole-footprint census is reported beside it.
    over = "the whole footprint"
    whole = census
    if census.get("inner"):
        census = census["inner"]
        over = f"the core and the inner rings (a {2 * int(census.get('window', 0))}-square)"
    if want and census.get("read"):
        from .. import groundread
        share = groundread.biome_share(census, want)
        out["biome"] = {"wanted": want, "share": share, "classes": census.get("classes"),
                        "over": over,
                        "whole": (groundread.biome_share(whole, want)
                                  if whole is not census and whole.get("read") else None),
                        "holds": share >= spec_mod.BIOME_SHARE}
    elif want:
        out["biome"] = {"wanted": want, "holds": False, "note": "no biome census on the "
                                                               "chosen site's record"}
    else:
        out["biome"] = {"wanted": None, "holds": True, "note": "no biome asked for"}
    holds.append(out["biome"]["holds"])
    # decks
    rec = _parts_record(rnd, results)
    by: dict = {}
    for w in rec.get("waves", []):
        for row in w.get("parts", []):
            if row.get("status") == "built":
                by[row.get("ground")] = by.get(row.get("ground"), 0) + 1
    decks = int(by.get("deck", 0))
    out["decks"] = {"by_ground": by, "decks": decks, "water": setting.get("water"),
                    "holds": decks == 0 or setting.get("water") == "some"}
    holds.append(out["decks"]["holds"])
    # walls
    walls = _wall_levels_of(rnd)
    ring_walls = [w for w in walls if w["ring_level"] is not None]
    out["walls"] = {"parts": walls,
                    "holds": bool(ring_walls) and all(w["at_ring_level"] for w in ring_walls)}
    holds.append(out["walls"]["holds"])
    # podium
    lay = ((rnd.plan() or {}).get("layout") or {})
    levels = [r.get("level") for r in (lay.get("rings") or []) if r.get("level") is not None]
    podium = _podium_of(rnd)
    out["podium"] = {"podium": podium, "ring_levels": levels,
                     "holds": (podium is not None and bool(levels)
                               and int(podium) == max(levels) + int(out["registered"]["TERRACE_STEP"]))}
    holds.append(out["podium"]["holds"])
    # e008
    try:
        ctx, _net = _town_context(rnd, be)
        rep = lint.lint(ctx, family=lint.PLACE)
        e008 = [f for f in rep.findings if f.code == "E008"]
        out["e008"] = {"place_wide": len(e008), "thresholds": len(ctx.network.thresholds)
                       if ctx.network else None,
                       "examples": [f.message for f in e008[:6]], "holds": not e008}
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        out["e008"] = {"holds": False, "note": f"unread: {type(e).__name__}: {e}"}
    holds.append(out["e008"]["holds"])
    out["got"] = 1 if all(holds) else 0
    out["failed"] = [k for k in ("biome", "decks", "walls", "podium", "e008")
                     if not out[k]["holds"]]
    return out


def _m_occupancy(rnd, be, results, bar) -> dict:
    """Five clauses, each with its registered number and the miss named where it misses:

          structures   the columns of ground per standing structure in each ring annulus,
                       under the ceiling the ring's own density word registers;
          ground       the plots and areas of each ring over the fraction of its districts
                       `occupancy_shares()` asks for, the belt's own being `RURAL_COVER`;
          floor        the open ground of each ring dressed in the **setting's** surface
                       rather than in the footing family a terrace fills with;
          centre       the compound over `CENTRE_SHARE_MIN` of the innermost ring's width;
          cover        the cover every terrace was laid with, off the terrace record.

        Read by `scripts/occupancy_measure.py` off the same records the round leaves behind,
        so the number in the readout and the number a person can reproduce on the command
        line are the same number.
        
    """
    import importlib.util
    p = os.path.join(_pipeline.ROOT, "scripts", "occupancy_measure.py")
    if not os.path.exists(p):
        return {"read": False, "why": "no scripts/occupancy_measure.py"}
    sp = importlib.util.spec_from_file_location("occupancy_measure", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    try:
        got = m.measure(rnd.name, log=lambda *a, **k: None)
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        return {"read": False, "why": f"{type(e).__name__}: {e}"}
    from .. import placeplan, spec as spec_mod
    spec = rnd.place_spec() or {}
    rings = {r["name"]: r for r in spec_mod.rings(spec)}
    reg = (bar or {})
    per_ceiling = dict(reg.get("columns_per_structure")
                       or placeplan.columns_per_structure_ceiling())
    out = {"read": True, "registered": {
        "columns_per_structure": per_ceiling,
        "shares": placeplan.occupancy_shares(),
        "RURAL_COVER": placeplan.RURAL_COVER,
        "PLOT_CELLS": dict(spec_mod.PLOT_CELLS),
        "CENTRE_SHARE_MIN": placeplan.CENTRE_SHARE_MIN}}
    holds = []
    rows = []
    for r in got.get("rings") or []:
        part = rings.get(r["ring"]) or {}
        word = part.get("density") or "medium"
        role = part.get("role") or spec_mod.read_role(None, part, r["ring"])
        cap = float(per_ceiling.get(word, per_ceiling.get("medium", 1e9)))
        cover_want = (placeplan.RURAL_COVER if role == "rural"
                      else placeplan.ground_cover(word))
        # the cover is asked of the district ground, which is what a planner was given
        dshare = (float(r["district_columns"]) / r["annulus_columns"]
                  if r["annulus_columns"] else 0.0)
        got_cover = (r["ground_cover"] / dshare) if dshare else 0.0
        per = r["columns_per_structure"]
        rows.append({"ring": r["ring"], "density": word, "role": role,
                     "structures": r["structures"], "areas": r["areas"],
                     "columns_per_structure": per, "ceiling": cap,
                     "under_ceiling": bool(per is not None and per <= cap),
                     "ground_cover_of_districts": round(got_cover, 4),
                     "wanted": cover_want,
                     "covered": bool(got_cover >= cover_want)})
        holds.append(rows[-1]["under_ceiling"])
        holds.append(rows[-1]["covered"])
    out["rings"] = rows
    # the floor: the open ground of each ring against the footing family it used to be
    floor = got.get("floor") or {}
    fam = (rnd.plan() or {}).get("palette") or {}
    footing = str(fam.get("footing") or "")
    froows = []
    for r in (floor.get("rings") or []):
        top = (r.get("built") or [{}])[0]
        froows.append({"ring": r["ring"], "open_columns": r["open_columns"],
                       "commonest": top.get("block"), "pct": top.get("pct"),
                       "before_the_parts": (r.get("before_the_parts") or [{}])[0],
                       "not_the_footing": top.get("block") != footing})
        holds.append(froows[-1]["not_the_footing"])
    out["floor"] = {"footing": footing, "rings": froows}
    out["terraces"] = got.get("terraces")
    out["centre"] = got.get("centre")
    share = ((got.get("centre") or {}).get("side_share_of_innermost") or 0.0)
    out["centre_share"] = {"got": share, "wanted": placeplan.CENTRE_SHARE_MIN,
                           "holds": share >= placeplan.CENTRE_SHARE_MIN}
    holds.append(out["centre_share"]["holds"])
    out["got"] = 1 if holds and all(holds) else 0
    out["failed"] = ([f"{r['ring']}/structures" for r in rows if not r["under_ceiling"]]
                     + [f"{r['ring']}/ground" for r in rows if not r["covered"]]
                     + [f"{r['ring']}/floor" for r in froows if not r["not_the_footing"]]
                     + ([] if out["centre_share"]["holds"] else ["centre"]))
    return out


def _m_levels(rnd, be, results, bar) -> dict:
    """Every ring's terrace level and the podium's, and every wall's sited level."""
    lay = ((rnd.plan() or {}).get("layout") or {})
    return {"terrace": lay.get("terrace"), "podium": _podium_of(rnd),
            "rings": [{"ring": r["name"], "level": r.get("level"), "inner": r["inner"],
                       "outer": r["outer"]} for r in (lay.get("rings") or [])],
            "walls": _wall_levels_of(rnd)}


def _m_preflights(rnd, be, results, bar) -> dict:
    """What every preflight said before its stage spent anything, off the records."""
    out = {}
    search = rnd.site_search() or {}
    if search.get("preflight"):
        out["site_search"] = {"radii": search["preflight"], "refused": search.get("refused")}
    for name, key in (("terraces.json", "terraces"), ("parts.json", "parts")):
        p = rnd.rel(name)
        if os.path.exists(p):
            doc = json.load(open(p))
            if doc.get("preflight"):
                out[key] = doc["preflight"]
    rp = _pipeline._report_path(rnd)
    if os.path.exists(rp):
        res = (json.load(open(rp)).get("results") or {})
        rb = ((res.get("render") or {}).get("budget") or {}).get("preflight")
        if rb:
            out["render"] = rb
    out["refused"] = [k for k, v in out.items()
                      if isinstance(v, dict) and v.get("refused")]
    return out


def _m_search_cost(rnd, be, results, bar) -> dict:
    """Where the search's squares came from and what it cost, against the concentric
    run's 4 h 12 min."""
    search = rnd.site_search() or {}
    rp = _pipeline._report_path(rnd)
    secs = None
    if os.path.exists(rp):
        secs = ((json.load(open(rp)).get("results") or {}).get("site_search") or {}).get("seconds")
    return {"squares": search.get("squares"), "generated": search.get("generated"),
            "located": len(search.get("located") or []),
            "refused_by_biome": len(search.get("refused_by_biome") or []),
            "candidates": search.get("candidates"), "meeting": search.get("meeting"),
            "attempt": search.get("attempt"), "read_by": search.get("read_by"),
            "seconds": secs, "concentric_run_seconds": 4 * 3600 + 12 * 60}


#: **Registered before it was read** (v2, C5): the fabric of the compiled districts. The
#: least share of a compiled district's houses whose way in is on the front the compiler
#: named; the most of a compiled district's rectangle no rule assigned.
FRONTAGE_FLOOR = 0.9
UNDEVELOPED_MAX = 0.15


def _m_fabric(rnd, be, results, bar) -> dict:
    """**The fabric of the compiled districts.** v2, C5. Four clauses, each with its
        registered number and the miss named where it misses:

          columns_per_house  the columns of ground per house in the compiled districts of
                             each density word, under the ceiling the word registers
                             (`placeplan.columns_per_structure_ceiling`);
          frontage           the share of compiled houses whose reserved threshold is on
                             the side their leaf names as its front, over `FRONTAGE_FLOOR`;
          assigned           every compiled district's undeveloped share under
                             `UNDEVELOPED_MAX`: the leftover ground is assigned;
          attached           every compiled district whose character said `attached` has
                             party walls.

        Read off the compiler's own records (`district_<name>_compiled.json`), the plan's
        leaves and the network's thresholds, all of them on disk.
        
    """
    from .. import placeplan
    reg = bar or {}
    ceilings = dict(reg.get("columns_per_house") or placeplan.columns_per_structure_ceiling())
    floor = float(reg.get("frontage_floor", FRONTAGE_FLOOR))
    most = float(reg.get("undeveloped_max", UNDEVELOPED_MAX))
    recs = []
    if os.path.isdir(rnd.state):
        for f in sorted(os.listdir(rnd.state)):
            if f.startswith("district_") and f.endswith("_compiled.json"):
                recs.append(json.load(open(rnd.rel(f))))
    out = {"read": bool(recs), "registered": {"columns_per_house": ceilings,
                                              "frontage_floor": floor,
                                              "undeveloped_max": most},
           "districts": len(recs)}
    if not recs:
        out.update(got=None, why="no compiled district in this round")
        return out
    # 1. columns per house, per density word, over the districts of that word
    by: dict = {}
    for r in recs:
        word = (r.get("character") or {}).get("density") or "medium"
        b = by.setdefault(word, {"columns": 0, "houses": 0, "districts": 0})
        b["columns"] += int(r["columns"])
        b["houses"] += int(r["lots"])
        b["districts"] += 1
    words = {}
    for word, b in sorted(by.items()):
        per = (b["columns"] / b["houses"]) if b["houses"] else None
        cap = float(ceilings.get(word, ceilings.get("medium", 1e9)))
        words[word] = {**b, "columns_per_house": None if per is None else round(per, 1),
                       "ceiling": cap, "under_ceiling": bool(per is not None and per <= cap)}
    out["columns_per_house"] = words
    # 2. the frontage: every compiled house with a front, against its threshold
    net = rnd.network()
    into = {"north": "south", "south": "north", "east": "west", "west": "east"}
    fronted = on_front = 0
    off = []
    for p in rnd.parts():
        # only a compiled leaf carries a front: the side its street is on
        if p.get("kind", "plot") != "plot" or not p.get("front"):
            continue
        fronted += 1
        th = net.threshold(p["name"]) if net is not None else None
        if th is not None and th.facing == into[p["front"]]:
            on_front += 1
        else:
            off.append(p["name"])
    share = (on_front / fronted) if fronted else None
    out["frontage"] = {"houses": fronted, "on_front": on_front,
                       "share": None if share is None else round(share, 4),
                       "floor": floor, "holds": bool(share is not None and share >= floor),
                       "off": off[:12]}
    # 3. the leftover assigned
    left = {r["district"]: r["undeveloped_share"] for r in recs}
    out["assigned"] = {"undeveloped_share": left, "max": most,
                       "over": sorted(n for n, v in left.items() if v > most)}
    # 4b. **the rhythm**, the craft round (E3): distinct building shapes per hundred and
    # the longest run of identical neighbours on one street face, both off the
    # compiler's own record, and the second wall material's share of the street
    from .. import district_compile as dc
    rh = {r["district"]: r.get("variety") for r in recs if r.get("variety")}
    alt = {r["district"]: r.get("wall_alt") for r in recs if r.get("wall_alt")}
    out["rhythm"] = {
        "registered": {"shapes_per_hundred": dc.SHAPES_PER_HUNDRED,
                       "identical_run": dc.IDENTICAL_RUN_MAX},
        "districts": {n: {k: v[k] for k in ("lots", "shapes", "per_hundred",
                                            "longest_run")} for n, v in rh.items()},
        "combed": sorted(n for n, v in rh.items()
                         if v["longest_run"] > dc.IDENTICAL_RUN_MAX),
        "under_shapes": sorted(n for n, v in rh.items()
                               if v["per_hundred"] < dc.SHAPES_PER_HUNDRED)}
    out["wall_alt"] = {"districts": {n: v["share"] for n, v in alt.items()},
                       "registered": (list(alt.values()) or [{}])[0].get("registered"),
                       "off": sorted(n for n, v in alt.items() if not v["holds"])}
    # 4. attached where the character said attached
    att = {r["district"]: {"party_walls": r.get("party_walls", 0), "lots": r["lots"],
                           "note": r.get("attached_note")}
           for r in recs if (r.get("character") or {}).get("attached")}
    out["attached"] = {"districts": att,
                       "without": sorted(n for n, v in att.items()
                                         if v["lots"] >= 2 and not v["party_walls"])}
    failed = ([f"columns_per_house/{w}" for w, v in words.items() if not v["under_ceiling"]]
              + ([] if out["frontage"]["holds"] else ["frontage"])
              + [f"assigned/{n}" for n in out["assigned"]["over"]]
              + [f"attached/{n}" for n in out["attached"]["without"]]
              + [f"rhythm/{n}" for n in out["rhythm"]["combed"]]
              + [f"shapes/{n}" for n in out["rhythm"]["under_shapes"]]
              + [f"wall_alt/{n}" for n in out["wall_alt"]["off"]])
    out["failed"] = failed
    out["got"] = 0 if failed else 1
    return out


MEASURES = {
    "fabric": _m_fabric,
    "ground": _m_ground,
    "occupancy": _m_occupancy,
    "levels": _m_levels,
    "preflights": _m_preflights,
    "search_cost": _m_search_cost,
    "concentric": _m_concentric,
    "walk_from_outdoors_pct": _m_walk_from_outdoors_pct,
    "e002_from_the_lane": _m_e002_from_the_lane,
    "own_lint_errors": _m_own_lint_errors,
    "builder_calls": _m_builder_calls,
    "program_lines": _m_program_lines,
    "writes_covered_over": _m_writes_covered_over,
    "tokens_per_building": _m_tokens_per_building,
    "instances_lint_zero": _m_instances_lint_zero,
    "instances_walkable": _m_instances_walkable,
    # A bar that names no measure is a bar nobody is holding the round to, so the
    # registry carries these as well and the deviation is on the record rather than in a
    # row marked unreadable.
    "within_type_variation": _m_within_type_variation,
    "building_calls_per_type": _m_building_calls_per_type,
    # A place is not a set of buildings, so it has measures a settlement did not need.
    "structures": _m_structures,
    "instantiation": _m_instantiation,
    "hand_programs": _m_hand_programs,
    "enclosure": _m_enclosure,
    "tokens_whole_round": _m_tokens_whole_round,
    "whole_place_lint_seconds": _m_whole_place_lint_seconds,
    # The one measure in this registry that reads the request.
    "place_read": _m_place_read,
    "site_chosen": _m_site_chosen,
}


def stage_readout(rnd: Round, be, results: dict) -> dict:
    """The round's registered bars, computed and written to `readout.json`.

        Nothing here decides anything. The bars are read off the round's own
        pre-registration; each names a measure from `MEASURES` and carries a comparator and
        a value; the measure is run and the row says met or missed. A bar naming no measure
        is reported as unreadable rather than skipped, because a bar the driver cannot read
        is a bar nobody is holding the round to.
    """
    from .. import observe
    prereg = ((json.load(open(_pipeline._report_path(rnd))).get("preregistered")
               if os.path.exists(_pipeline._report_path(rnd)) else None) or rnd.preregistered)
    bars = prereg.get("bars") or {}
    # A re-read of named rows under the instruments as they now stand -- `round.py
    # --stage readout --measure own_lint_errors`. The recorded readout is a record and
    # is not written over; this goes beside it as `readout.reread.json`, and the row it
    # moves belongs in `scripts/test_types.py`'s SUPERSEDED with the cause named.
    only = list(rnd.flags.get("measures") or [])
    if only:
        missing = [m for m in only if m not in bars]
        if missing:
            return {"error": f"this round registered no bar called {missing}; its bars "
                             f"are {sorted(bars)}"}
        bars = {k: v for k, v in bars.items() if k in only}
    out = {"round": rnd.name, "site": rnd.site, "bars": bars,
           "walk_model": observe.WALK_MODEL, "results": {}}
    for key, bar in bars.items():
        name = MEASURE_ALIASES.get(key, key)
        spec = bar if isinstance(bar, dict) else {"bar": bar}
        row: dict = {"measure": name, "bar": spec.get("bar", bar)}
        if name not in MEASURES:
            row["unreadable"] = (f"no measure called {name!r}; the registry is "
                                 f"{sorted(MEASURES)}")
            out["results"][key] = row
            continue
        op, value = _bar_value(spec.get("bar", bar))
        t0 = time.perf_counter()
        try:
            got = MEASURES[name](rnd, be, results, spec)
        except Exception as e:               # noqa: BLE001 -- reported, not raised
            row["error"] = f"{type(e).__name__}: {e}"
            out["results"][key] = row
            continue
        row.update(got)
        row["comparator"] = op
        # A measure may supply the band it is read against; `_bar_value` leaves `value`
        # as None when the registered sentence names one instead of a number.
        row["value"] = row.get("band") if (op == "in" and value is None) else value
        row["met"] = _meets(op, row.get("got"), row["value"])
        row["seconds"] = round(time.perf_counter() - t0, 1)
        out["results"][key] = row
    for extra in (prereg.get("reported_no_bar") or {}):
        # Its `got` and its own clauses go on the readout with no met or miss, because
        # nothing registered a bar on it.
        if not only and extra in MEASURES:
            try:
                got = MEASURES[extra](rnd, be, results, {})
            except Exception as e:           # noqa: BLE001 -- reported, not raised
                got = {"error": f"{type(e).__name__}: {e}"}
            out.setdefault("reported_no_bar", {})[extra] = {
                "measure": extra, "barred": False, **got}
            continue
        out.setdefault("reported_no_bar", {})[extra] = "see the round file"
    if rnd.types:
        out["instances"] = _instance_rows(rnd, results)
    p = rnd.rel(rnd.types.get("out", "") if rnd.types else "",
                "readout.reread.json" if only else "readout.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if only:
        rec = rnd.rel(rnd.types.get("out", "") if rnd.types else "", "readout.json")
        out["reread_of"] = rec
        out["recorded"] = {k: (json.load(open(rec)).get("results") or {}).get(k, {}).get("got")
                           for k in bars} if os.path.exists(rec) else {}
        out["_by"] = "stage_readout --measure"
        json.dump(out, open(p, "w"), indent=1)
        for k, r in out["results"].items():
            print(f"  re-read  {k:26} recorded {out['recorded'].get(k)!s:>8}   "
                  f"now {r.get('got', r.get('error'))}", flush=True)
        return {**out, "written": p}
    # A readout this stage did not write is a record: rounds 9, 10, 11 and 12 each ended
    # with one made by a script that no longer exists, and `out/` is not in Git. Put it
    # beside rather than under.
    was = (json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {})
    if os.path.exists(p) and "_by" not in was:
        import shutil
        keep = os.path.join(os.path.dirname(p), "readout.superseded.json")
        if not os.path.exists(keep):
            shutil.copyfile(p, keep)
            out["superseded"] = keep
    # A readout written over one read under a different walk model is not an update, it
    # is a different question answered on the same town, and the file says so. Rows this
    # moves belong in `scripts/test_types.py`'s SUPERSEDED with a cause.
    if was and was.get("walk_model", 3) != observe.WALK_MODEL:
        out["walk_model_moved"] = {
            "from": was.get("walk_model", "unversioned (<= 3)"),
            "to": observe.WALK_MODEL,
            "not_comparable": sorted(set(was.get("results") or {}) & set(out["results"])),
            "note": "these rows were read under a different definition of a room's "
                    "floor; difference them only through the SUPERSEDED record in "
                    "scripts/test_types.py, which names the change that moved each one"}
        for k in out["walk_model_moved"]["not_comparable"]:
            out["results"][k]["walk_model_moved"] = out["walk_model_moved"]["from"]
    out["_by"] = "stage_readout"
    json.dump(out, open(p, "w"), indent=1)
    n_met = sum(1 for r in out["results"].values() if r.get("met"))
    print(f"  {n_met}/{len(out['results'])} bars met", flush=True)
    for k, r in out["results"].items():
        mark = ("MET " if r.get("met") else "MISS" if r.get("met") is False else "??  ")
        print(f"  {mark}  {k:26} bar {str(r.get('bar'))[:18]:>18}   "
              f"got {r.get('got', r.get('unreadable') or r.get('error'))}", flush=True)
    return {**out, "written": p}


def built_volumes(rnd) -> tuple:
    """(the built world, the ground before it) off the round's own state, or (None,
    None) where nothing has been built. A2 of the voice contract: the place read asks
    what the place is made of whenever there is a built world to ask it of."""
    from .. import offline
    bp = rnd.rel("world_built.npz")
    if not os.path.exists(bp):
        return None, None
    gp = rnd.rel(rnd.base_volume)
    return (offline.load_volume(bp),
            offline.load_volume(gp) if os.path.exists(gp) else None)


QUALIFY_BRIEF = """# Is what was built the place that was asked for?

> {sentence}

You are the **judge**, and this is the last question the run asks. Every physical fact
about this place has already been measured; what is left are the obligations no
measurement can close -- whether the result is recognisably the named place, and whether
it is actually built in the tradition that was asked for.

Answer from **what was built**, listed below. Do not answer from the sentence, from the
plan's intentions, or from what the place was supposed to be.

## What was built

{built}

## What was looked at

{looked}

## What the research established

{claims}

## The obligations open at this point

{open}

## Output

Write a single JSON file to {out}:

```json {{"verdicts".

`about` is `identity` or `tradition`; `name` is the name or tradition it is about. Use
`"holds"` instead of `"recognisable"` for anything that is not an identity. **A verdict
must cite the claims it rests on**: a judgement with no evidence behind it does not close
an obligation, and one that cites research the run never did is refused. Say `false` and
why, plainly, where the result does not carry it. Refusing is a legitimate outcome and it
is a more useful one than a generous pass.
"""


def _open_obligations(rnd) -> list:
    """The requirements only a judgment can close, as they stand now.

        `identity` and `tradition`: everything else in this build is measured, and an
        obligation that a measurement can close is not asked of a judge.
        
    """
    from .. import contracts
    rec = contracts.load(rnd, "intent") or {}
    return [r for r in rec.get("requirements") or []
            if r.get("kind") in ("identity", "tradition")
            and r.get("status") not in ("satisfied", "unsupported")]


def stage_qualify(rnd, be, results: dict) -> dict:
    """**The inspection that closes what no measurement can**, bound to this candidate.

        The review's fifth finding, both halves of it: this build had an identity evaluator
        that no production call ever supplied a judgment to, and a final place reader that
        supplied neither reading, capabilities nor judgment to the requirements -- an
        isolated false-pass interface beside an unreachable success route. This stage is the
        production path between them.

        It runs after construction and before the place is read, because the question is
        about what was built. Its answer is a `judgment` record naming the candidate it is
        of, the outputs it looked at and the claims each verdict rests on; `stage_place_check`
        consumes it only where that candidate is the one in hand, so an inspection cannot
        survive the design it was made of.

        A round with nothing for a judge to close -- no named place, no tradition asked for --
        skips, and says so.
        
    """
    from .. import contracts, deps
    if not rnd.sentence:
        return {"skipped": "this round carries no sentence"}
    plan = rnd.plan()
    if not plan:
        return {"skipped": "no plan.json: there is nothing built to judge"}
    open_rows = _open_obligations(rnd)
    if not open_rows:
        return {"skipped": "this request carries no identity or tradition obligation "
                           "that a measurement has left open"}
    here = deps.candidate_id(rnd, plan=plan)
    got = contracts.load(rnd, "judgment")
    if got is not None and got.get("candidate") == here:
        return {"skipped": "already judged, and of this candidate",
                "verdicts": len(got["verdicts"]), "candidate": here}
    answer_p = rnd.rel("judgment.answer.json")
    if got is not None and got.get("candidate") != here:
        # the candidate moved under a finished inspection: the verdicts are about a
        # design that is gone and are withdrawn rather than carried forward
        for f in ("judgment.json", "judgment.answer.json"):
            if os.path.exists(rnd.rel(f)):
                os.replace(rnd.rel(f), rnd.rel(f"stale.{here}.{f}"))
        deps.invalidate(rnd, "judgment",
                        f"the candidate moved to {here} after this inspection")
        print(f"   qualify: the judgment on disk is of candidate "
              f"{got.get('candidate')} and this is {here}; it is withdrawn and the "
              f"place is judged again", flush=True)
    job_p = rnd.rel("judgment.job.json")
    job = json.load(open(job_p)) if os.path.exists(job_p) else None
    if job is not None and job.get("candidate") != here:
        # **A pending answer is bound to the candidate it was asked of.** The review's
        # counterexample through this very stage: a job staged for candidate A, its
        # answer written, the plan replaced by B, the stage resumed -- and A's answer
        # was adopted as B's judgment. The job record carries the candidate; an answer
        # that arrives for a candidate that is gone is set aside by name and the
        # question is asked again of the design in hand.
        for f in ("judgment.answer.json", "judgment.job.json", "qualify_prompt.md"):
            if os.path.exists(rnd.rel(f)):
                os.replace(rnd.rel(f), rnd.rel(f"stale.{job.get('candidate')}.{f}"))
        print(f"   qualify: the judge job on disk was asked of candidate "
              f"{job.get('candidate')} and this is {here}; its answer, if any, is set "
              f"aside and the place is judged again", flush=True)
        job = None
    if not os.path.exists(answer_p):
        built = _built_says(rnd, plan, results)
        looked = _looked_at(rnd)
        # **A plan is not something built.** The review's counterexample through this
        # stage: `plan.json` alone counted as inspectable output and a judge was asked
        # about a place nobody had constructed. A judge looks at the built volume or at
        # pictures of it; with neither, the obligation stays open and says why.
        if not any(str(x).endswith("world_built.npz") or str(x).endswith(".png")
                   for x in looked):
            return {"status": "unresolved", "plan_only": True,
                    "error": ("nothing has been built or rendered for this candidate -- "
                              "a plan is not inspectable output -- so there is nothing "
                              "for a judge to look at; the identity and tradition "
                              "obligations stay open"),
                    "obligations": [r["id"] for r in open_rows]}
        reading = contracts.load(rnd, "reading") or {}
        claims = [c for c in reading.get("claims") or [] if c.get("says")]
        brief_p = rnd.rel("qualify_prompt.md")
        open(brief_p, "w").write(QUALIFY_BRIEF.format(
            sentence=rnd.sentence, built=built, out=answer_p,
            looked="\n".join(f"- `{os.path.basename(p)}` -- {p}" for p in looked),
            claims=("\n".join(f"- `{c.get('id')}` {c['says']}"
                              + ("" if c.get("source") else "  (inferred, no source)")
                              for c in claims[:24])
                    or "(nothing was sourced about this request)"),
            open="\n".join(f"- `{r['id']}` -- {r['says']}: {r.get('why') or ''}"
                           for r in open_rows)))
        json.dump({"candidate": here, "request": brief_p, "write": answer_p,
                   "obligations": [r["id"] for r in open_rows],
                   "looked_at": looked, "t": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(job_p, "w"), indent=1)
        return {"judgment": {
            "status": "needs_model", "role": "judge", "request": brief_p,
            "write": answer_p, "candidate": here,
            "images": [p for p in looked if p.endswith(".png")][:8],
            "note": (f"{len(open_rows)} obligation(s) no measurement can close, judged "
                     f"against what was built")}}
    doc = json.load(open(answer_p))
    rec = contracts.make("judgment", sentence=rnd.sentence, candidate=here,
                         looked_at=_looked_at(rnd),
                         verdicts=list(doc.get("verdicts") or []),
                         note=str(doc.get("note") or "")
                              or "an inspection of what was built, bound to the "
                                 "candidate it looked at")
    contracts.save(rnd, "judgment", rec)
    with contextlib.suppress(ValueError):
        deps.stamp(rnd, "judgment", outputs=["judgment.json"], plan=plan,
                   note=f"{len(rec['verdicts'])} verdict(s)")
    for v in rec["verdicts"]:
        said = v.get("recognisable") if v.get("recognisable") is not None else v.get("holds")
        print(f"   judged {v.get('about')}/{v.get('name') or '-'}: "
              f"{'yes' if said else 'NO'} -- {str(v.get('why'))[:110]}", flush=True)
    return {"verdicts": len(rec["verdicts"]), "candidate": here,
            "judged": [{"about": v.get("about"), "name": v.get("name"),
                        "holds": (v.get("recognisable") if v.get("recognisable")
                                  is not None else v.get("holds"))}
                       for v in rec["verdicts"]]}


def _looked_at(rnd) -> list:
    """Every output of this candidate a judge can actually look at, in order."""
    out = []
    # **The built world's own views first.** The fresh checker: the judge was handed the
    # preview's type cards and maps and never the inspection stage's pictures of what
    # was actually built.
    for rel in ("inspection", "renders", "frames", "preview"):
        d = rnd.rel(rel)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith(".png") and not f.startswith("stale.")]
    for f in ("world_built.npz", "parts.json", "plan.json"):
        if os.path.exists(rnd.rel(f)):
            out.append(rnd.rel(f))
    return out


def _built_says(rnd, plan: dict, results: dict) -> str:
    """What stands, in the judge's own terms: the parts, the lint and the access."""
    parts = _pipeline.plan_parts(plan)
    pr = rnd.rel("parts.json")
    rec = json.load(open(pr)) if os.path.exists(pr) else (results.get("parts") or {})
    rows = [r for w in (rec.get("waves") or []) for r in (w.get("parts") or [])]
    stood = sum(1 for r in rows if r.get("stood", r.get("status") == "built"))
    # **What stands, by type -- not what the plan names.** The fresh checker: the brief
    # said "row_house x15, farmstead x7, minka x2" for a sample in which three row
    # houses and no farmhouse had been built.
    by_type: dict = {}
    src = [{"kind": r.get("kind", "plot"), "type": r.get("type")}
           for r in rows if r.get("stood", r.get("status") == "built")] or parts
    for p in src:
        if p.get("kind", "plot") == "plot" and p.get("type"):
            by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    sample = rec.get("sample") or {}
    lint = results.get("lint") or {}
    lines = [f"- **{len(parts)} leaf/leaves planned**, {stood} standing in the built "
             f"world" + (f" -- a construction SAMPLE of {sample.get('plots')} plot(s) in "
                         f"{', '.join(sample.get('quarters') or [])}; the rest of the plan "
                         f"is not built and cannot be judged" if sample else "")
             if rec else f"- **{len(parts)} leaf/leaves planned**; no parts "
             f"record, so nothing is recorded as standing",
             "- standing, by type: " + (", ".join(f"`{t}` x{n}" for t, n in
                                                 sorted(by_type.items(), key=lambda kv: -kv[1])[:10])
                                       or "nothing")]
    if lint:
        lines.append(f"- the construction check reports {lint.get('errors', '?')} "
                     f"error(s) and {lint.get('warnings', '?')} warning(s)")
        acc = lint.get("e002_from_the_lane") or {}
        if acc.get("status"):
            lines.append(f"- access: {acc['status']} -- {acc.get('lane_components')} "
                         f"lane component(s), "
                         f"{acc.get('not_walkable_from_the_lane')} door(s) not "
                         f"reachable from an entry")
    return "\n".join(lines)


def stage_place_check(rnd, be, results: dict) -> dict:
    """`ethoslm.placeread.read`, as a stage, and it runs no model. Its answer is a bar."""
    from .. import contracts, pipeline, placeread
    spec = rnd.place_spec()
    if spec is None:
        return {"skipped": "this round has no place.json, so there is no spec to read "
                           "the place against"}
    plan = rnd.plan()
    if not plan:
        return {"error": "no plan.json: there is nothing to read"}
    pr = rnd.rel("parts.json")
    parts_rec = json.load(open(pr)) if os.path.exists(pr) else (
        results.get("parts") or {})
    built, base = built_volumes(rnd)
    # **The sentence's own requirements, against the resolved design.** The architecture
    # round: the `asked/...` clauses check what the request said outright, and two of
    # them -- the layout policy and what the buildings face -- are facts about the
    # *resolution* rather than about any one part, so the record that carries them is
    # read here and handed in. Without it those clauses answer `unresolved`, which is
    # honest and useless. **The judgment, where there is one of *this* candidate.** See
    # `stage_qualify`. A verdict about the design before a revision is evidence about a
    # place that is gone, so the identity it would close stays open and says why --
    # which is the difference between an inspection bound to a candidate and one merely
    # stored beside it.
    from .. import deps as deps_mod
    judgment = contracts.load(rnd, "judgment")
    here = deps_mod.candidate_id(rnd, plan=plan)
    if judgment is not None and judgment.get("candidate") != here:
        print(f"   place read: the judgment on disk is of candidate "
              f"{judgment.get('candidate')} and this place is {here}; it is not read "
              f"against this one", flush=True)
        judgment = None
    got = placeread.read(spec, plan, parts_rec,
                         voice=rnd.voice_name() or None,
                         site=pipeline.settlement_site(rnd), built=built, base=base,
                         plateau=pipeline.plateau_record(rnd),
                         intent_rec=contracts.load(rnd, "intent"),
                         resolution=contracts.load(rnd, "resolution"),
                         reading=contracts.load(rnd, "reading"),
                         capabilities=contracts.load(rnd, "capabilities"),
                         judgment=judgment,
                         # the solved place's own layout record: where the rings stand
                         # and how high, which is what `rings/elevation` is asked of
                         layout=(json.load(open(rnd.rel("plan.place.json"))).get("layout")
                                 if os.path.exists(rnd.rel("plan.place.json")) else None))
    # **Open material findings of the built world block completion.** The expression
    # round: the inspection's findings were a report appended after the fact, and the
    # farm finished with its square thirteen times its cottages still recorded. The
    # improve stage writes a disposition for every finding; a material one still open --
    # no owner action, a refused action, or a spent budget -- is a clause this read
    # fails on, at built evidence, and says which. Optional findings never block.
    # **...and the ledger is what is asked, not the last reading.** The design round: a
    # judged finding and an emitted construction constraint are obligations of one
    # shape, they survive a reading that omits them, and an applied action that moved
    # nothing leaves its row owed. `obligations.json` is the record; the dispositions
    # file beside it is a projection of it for readers that predate the ledger.
    from .. import obligation as obligation_mod
    led = obligation_mod.load(rnd.state)
    disp_p = rnd.rel("inspection", "dispositions.json")
    ran = bool(led.get("passes")) or os.path.exists(disp_p)
    if ran:
        owed = obligation_mod.open_rows(led, material=True)
        loose = obligation_mod.undisposed(led)
        got["clauses"].append({
            "clause": "built/findings", "holds": not owed,
            "method": "observed",
            "evidence": "built_sample" if (parts_rec or {}).get("sample") else "built_place",
            "says": (f"{len(led.get('rows') or {})} obligation(s) of the built world, "
                     f"{len(owed)} material and open: "
                     + "; ".join(f"{r.get('id')} ({r.get('owner')}, {r.get('origin')}) "
                                 f"{str(r.get('says'))[:80]}" for r in owed[:6])
                     if owed else
                     f"{len(led.get('rows') or {})} obligation(s) of the built world, "
                     f"none material and open"
                     + (f"; {len(loose)} optional row(s) nobody has disposed of"
                        if loose else "")),
            "open": [r.get("id") for r in owed],
            "undisposed": [r.get("id") for r in loose],
            "ledger": rnd.rel(obligation_mod.RECORD),
            "dispositions": {r.get("id"): r.get("disposition")
                             for r in obligation_mod.table(led)}})
        if owed:
            got["holds"] = False
            got["got"] = 0
            got["failed"] = list(got.get("failed") or []) + ["built/findings"]
    else:
        got["clauses"].append({
            "clause": "built/findings", "holds": None, "evidence": "unobserved",
            "method": "unsupported",
            "says": "no obligations of the built world were recorded for this "
                    "candidate (the improve stage did not run)"})
    # **Of this candidate and this built world.** The fresh checker: the place read
    # named neither, so nothing bound it to the parts record it judged.
    got["candidate"] = here
    got["built_digest"] = deps_mod.content_print(rnd.rel("world_built.npz"))
    got["parts_candidate"] = (parts_rec or {}).get("candidate")
    p = rnd.rel("place_read.json")
    json.dump(got, open(p, "w"), indent=1)
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "place_read", outputs=["place_read.json"], plan=plan,
                       note="holds" if got["holds"] else "fails")
    print(f"  place read: {'HOLDS' if got['holds'] else 'FAILS'}"
          + (f" on {', '.join(got['failed'])}" if got["failed"] else ""), flush=True)
    for c in got["clauses"]:
        print(f"   {'ok  ' if c['holds'] else 'MISS'} {c['clause']:28} {c['says']}",
              flush=True)
    out = {**got, "written": p}
    if not got["holds"]:
        # **And a place that is not the place asked for says so as an outcome.** The
        # same finding as `stage_lint` above, at the other end of the run: `holds:
        # false` was a field and the round went on to its readout reporting a finished
        # place. `blocks: fidelity` is what this is -- the buildings stand, and the
        # result is not what the sentence asked for -- and it is deliberately a
        # different state from an unsound construction and from an infeasible plan.
        out.update({"status": "blocked", "stop": True, "blocks": "fidelity",
                    "error": (f"the built place does not answer the request it was "
                              f"made from: "
                              + "; ".join(str(c["says"])[:120] for c in got["clauses"]
                                          if c["holds"] is False)[:600])})
    return out
