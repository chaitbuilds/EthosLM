"""Circulation, wave execution and typed-part construction."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import time
from dataclasses import dataclass, field

from .. import pipeline as _pipeline
from .. import card as card_mod
from .. import measure as measure_mod
from .. import offline, settlement, verdicts
from ..measure import record
from .round import Round


def stage_programs(rnd: Round, be, results: dict) -> dict:
    """Execute each wave's stored program against the round's base volume.

        The deterministic half of a build: no model, no server, no writes -- the same path
        `test_stages`' control arm asserts is identical to `stages.run` with every flag off.
        Reports the pending block set size per wave, which is what the round's own
        `dryrun_<wave>.json` recorded when it was built.
        
    """
    if not rnd.waves:
        return {"waves": 0, "note": "this round has no build waves"}
    vol = be.volume
    out = {}
    for w in rnd.waves:
        prog = rnd.rel(w.get("program", f"{w['name']}.py"))
        rec: dict = {"program": prog}
        if not os.path.exists(prog):
            rec["error"] = "no program on disk"
            out[w["name"]] = rec
            continue
        t0 = time.perf_counter()
        try:
            b = be.execute(prog, vol)
            rec["pending"] = len(b._pending)
        except Exception as e:                      # noqa: BLE001 - reported, not raised
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        ref = rnd.rel(f"dryrun_{w['name']}.json")
        if os.path.exists(ref):
            rec["recorded"] = json.load(open(ref)).get("blocks")
            rec["reproduces"] = rec.get("pending") == rec["recorded"]
        out[w["name"]] = rec
    return out


def _gate_threshold(rnd: Round, net):
    """The reserved doorstep of the part the place is entered by. A7.

        The plan says which one: `stages_media.arrival_gate` -- a passage point on the
        outermost ring where the spec names rings, else the first leaf in plan order whose
        type declares `PASSAGE`. One rule, shared with the flythrough (demo-polish, 1d).
        With no such part -- an open village, which is every round up to 11 -- there is no
        arrival to photograph and the skyline frame is absent.
        
    """
    if net is None:
        return None
    from .stages_media import arrival_gate
    plan = rnd.plan()
    g = arrival_gate(plan, rnd.place_spec(), rnd.parts()) if plan else None
    return net.threshold(g["name"]) if g is not None else None


def _dry_circulation(rnd: Round, be) -> dict:
    """The circulation pass, offline: routed and emitted **into the cached volume**.

        `_commands/circulate_run.py` in the shape a dry run can run it. Everything that
        decides anything is the same call -- `parts_to_routing`, `plan_network` with the
        arterials laid first, `circulate.emit` -- and what differs is where the blocks go
        and where the ground was read from. It is deliberately not a second implementation
        of the routing: a second answer to "where do the lanes go" is the thing this project
        has twice paid for.
        
    """
    import numpy as np
    from .. import circulate, lint, observe, prims, settlement
    from ..buildlib import Builder
    plan = rnd.plan()
    parts = _pipeline.plan_parts(plan)
    if not parts:
        return {"status": "error", "stop": True,
                "error": "no plan.json, or a plan with no leaves in it"}
    passage = {n for n in {p.get("type") for p in parts if p.get("type")}
               if os.path.exists(os.path.join(_pipeline.ROOT, "types", f"{n}.py"))
               and _pipeline.load_type(
                   os.path.join(_pipeline.ROOT, "types", f"{n}.py"))["passage"]}
    from .stages_plan import type_declarations as _tdecl
    _decls_c = _tdecl(parts)
    parts = [({**p, "clearance": ((_decls_c.get(p.get("type")) or {}).get("needs")
                                  or {}).get("clearance")}
              if p.get("kind") == "edge" and ((_decls_c.get(p.get("type")) or {})
                                              .get("needs") or {}).get("clearance")
              is not None else p) for p in parts]
    routing = circulate.parts_to_routing(parts, passage=passage)
    mat = plan.get("circulation_material") or "cobblestone"
    if mat not in prims.MATERIALS:
        mat = "cobblestone"
    # **The lanes are laid on the ground before any lane, every time.** The expression
    # round's farm: the improve stage replanned after construction and this stage --
    # which bakes the lanes into `world.npz` from the backend's working volume -- wrote
    # the first candidate's sixteen cottages and its square into the base ground, so the
    # rebuild stood on the previous build (three blocked doorways, then a market on a
    # hill of its own paving). The ground as the plateau and the terraces left it is
    # kept beside the base the first time lanes are routed, and every routing starts
    # from it: a replan's lanes replace the previous lanes instead of joining them.
    before_lanes = rnd.rel(rnd.base_volume.replace(".npz", ".before-lanes.npz"))
    base_p = rnd.rel(rnd.base_volume)
    if not os.path.exists(before_lanes) and os.path.exists(base_p):
        import shutil
        shutil.copyfile(base_p, before_lanes)
    # **...and locally, only the local lanes are taken back.** The block design round.
    # Under a declared scope the roads outside it are what this unit connects *to*: the
    # base keeps them, the scope's own columns go back to the ground before any lane,
    # and the routing below is asked about this unit's sites only.
    from .. import local as _local
    scope = _local.scope_of(rnd)
    scope_rec: dict = {}
    if scope is not None and os.path.exists(before_lanes) and os.path.exists(base_p):
        vol = offline.load_volume(base_p)
        took = _local.restore_rect(vol, offline.load_volume(before_lanes),
                                   scope["outer"])
        scope_rec["columns_taken_back"] = took
        be._vol = vol
    elif os.path.exists(before_lanes):
        vol = offline.load_volume(before_lanes)
        be._vol = vol
    else:
        vol = be.volume
    heights, wet = observe.ground_heights(vol)
    avoid = np.where(wet, 40.0, 0.0)
    # ...and, on a designed place, the band at the foot of every terrace
    from .stages_plan import type_declarations
    avoid = np.maximum(avoid, circulate.designed_ground_costs(
        {**plan, "parts": parts}, type_declarations(parts), heights.shape, vol.x0, vol.z0))
    for (wx, wz) in routing["obstacles"]:
        i, j = wx - vol.x0, wz - vol.z0
        if 0 <= i < avoid.shape[0] and 0 <= j < avoid.shape[1]:
            avoid[i, j] = np.inf
    art_cells = circulate.arterial_cells(plan.get("arterials") or {}, heights,
                                         vol.x0, vol.z0)
    t0 = time.perf_counter()
    # what the scope changes is which sites are *routed to* and how much ground the
    # solver walks to do it. The margin is where the connection to the rest of the city
    # happens, and `walk_check` below says whether it did.
    sites_here = routing["sites"]
    h_used, hx0, hz0 = heights, vol.x0, vol.z0
    avoid_used = avoid
    if scope is not None:
        sites_here = [s for s in routing["sites"]
                      if _local.meets(scope, (s["x0"], s["z0"], s["x1"], s["z1"]))]
        ox0, oz0, ox1, oz1 = scope["outer"]
        i0 = max(0, ox0 - vol.x0)
        j0 = max(0, oz0 - vol.z0)
        i1 = min(heights.shape[0] - 1, ox1 - vol.x0)
        j1 = min(heights.shape[1] - 1, oz1 - vol.z0)
        if i1 > i0 and j1 > j0:
            h_used = heights[i0:i1 + 1, j0:j1 + 1]
            avoid_used = avoid[i0:i1 + 1, j0:j1 + 1]
            hx0, hz0 = vol.x0 + i0, vol.z0 + j0
            art_cells = {c: v for c, v in art_cells.items()
                         if hx0 <= c[0] <= hx0 + h_used.shape[0] - 1
                         and hz0 <= c[1] <= hz0 + h_used.shape[1] - 1}
        scope_rec.update(sites=len(sites_here), of_sites=len(routing["sites"]),
                         ground=[int(h_used.shape[0]), int(h_used.shape[1])],
                         of_ground=[int(heights.shape[0]), int(heights.shape[1])])
    net = circulate.plan_network(h_used, hx0, hz0, sites_here,
                                 centre=plan.get("centre"), max_step=3,
                                 avoid_extra=avoid_used, passable=routing["passable"],
                                 arterial=art_cells)
    plan_s = round(time.perf_counter() - t0, 1)
    #: what `emit` lays: the lanes this solve decided, and never the kept ones. The
    #: ground outside the scope already carries them and re-laying it would be the whole
    #: city's earthwork again by another name.
    net_here = net
    if scope is not None:
        # what this solve *lays* is held to the scope's outer bound, as the terraces'
        # writes are: a lane cell the solve reached past it is ground the scope does not
        # own (the fabric reset round: the gate street's cells were laid at the
        # arterial's designed level on the retained side of the gate, floating in its
        # passage)
        _o = scope["outer"]
        net_here = circulate.Network(
            {c: r for c, r in net.cells.items()
             if _o[0] <= c[0] <= _o[2] and _o[1] <= c[1] <= _o[3]},
            list(net.thresholds), nodes=net.nodes, edges=net.edges, notes=dict(net.notes))
        # **The city's own network is the boundary condition and it is kept.** A local
        # solve answers about this unit's lanes; the roads and thresholds outside it
        # were laid by the seed and are not re-decided, so they are carried through
        # unchanged and the record says how many of each came from where. Replacing the
        # whole network with the local one would make a block round quietly delete the
        # city it is a block of.
        was = rnd.network()
        if was is not None:
            keep_cells = {c: rec for c, rec in was.cells.items()
                          if not _local.meets(scope, (c[0], c[1], c[0], c[1]))}
            here_ids = {s["id"] for s in sites_here}
            # ...and a threshold inside the scope is this solve's to make or not: a site
            # the revision removed leaves no doorstep standing in its lanes
            keep_th = [t for t in was.thresholds if t.id not in here_ids
                       and not _local.meets(scope, (t.x, t.z, t.x, t.z))]
            merged = dict(keep_cells)
            merged.update(net.cells)
            # **...and a retained house the section builds is given back its way in**
            # (the design resolution round). A kept plot with no reserved doorstep was
            # entered through the party-wall slot `site()` used to cut; with the slot
            # gone its door opened into its neighbour's wall and it did not stand
            # (`lower_ring_north_2_b3_1_04`, the fabric reset reader's site_a). Its doorstep
            # is reserved on its own front, off the kept lane in front of it.
            def _g(x, z):
                i, j = x - vol.x0, z - vol.z0
                if 0 <= i < heights.shape[0] and 0 <= j < heights.shape[1]:
                    return int(heights[i, j])
                return None
            added, spur = _front_thresholds(rnd, plan, scope, merged,
                                            {t.id for t in keep_th} | here_ids, ground=_g)
            keep_th += added
            if spur:
                # the short lane a restored doorstep needs, laid with this solve's lanes
                merged.update(spur)
                net_here.cells.update(spur)
            scope_rec.update(cells_kept=len(keep_cells), cells_here=len(net.cells),
                             thresholds_kept=len(keep_th),
                             thresholds_here=len(net.thresholds),
                             thresholds_restored=[t.id for t in added])
            net = circulate.Network(merged, keep_th + list(net.thresholds),
                                    nodes=net.nodes, edges=net.edges,
                                    notes={**net.notes, "local": dict(scope_rec)})
    wc = circulate.walk_check(net)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    stats = circulate.emit(b, net_here, mat, ground=heights, x0=vol.x0, z0=vol.z0,
                           keep_off=settlement.path_columns())
    placed = be.commit(b)
    net.notes["material"] = mat
    net.notes["blocks"] = placed.get("placed")
    net.notes["dry_run"] = True
    net.save(rnd.rel("network.json"))
    offline.save_volume(be.volume, rnd.rel(rnd.base_volume))
    # **The ground with its lanes is the ground the stamp accepts.** The ground stamp is
    # bound to the base file's digest; this save moved it, and every run so far was
    # rescued by the next invocation's terraces re-stamp (there was always a pending
    # reading in between). A run that re-routes and builds in one invocation -- the
    # expression city after its wall clearance changed -- refused its own ground.
    from .stages_plan import _stamp_ground
    from .. import deps as _deps_c
    if _deps_c.recorded(rnd, "ground"):
        _stamp_ground(rnd, rnd.plan(), "lanes laid into the prepared ground")
    ctx = lint.Context.build(be.volume, _plots_or_empty(rnd), network=net,
                             sites=routing["sites"])
    rep = lint.lint(ctx)
    out = {"dry_run": True, "cells": len(net.cells),
           "thresholds": len(net.thresholds), "arterial_columns": len(art_cells),
           "material": mat, "lane": stats, "walk_check": wc,
           "blocks": placed.get("placed"), "seconds": {"plan": plan_s},
           "lint": rep.to_json(), "pieces": [len(p) for p in ctx.lane_pieces()],
           "note": "routed and emitted into the cached volume; no block was written to "
                   "the world"}
    if scope is not None:
        out["scope_sites"] = _scope_sites(rnd)
        out["scope"] = scope_rec
    json.dump(out, open(rnd.rel("circulation.json"), "w"), indent=1)
    return out


def _plots_or_empty(rnd: Round) -> list:
    p = rnd.rel("plots.json")
    return json.load(open(p)) if os.path.exists(p) else []


def _scope_sites(rnd) -> str | None:
    """A digest of the sites a local scope's lanes are routed to -- every plot and area
    of the plan meeting the scope, by name and rectangle -- or None without a scope."""
    import hashlib
    from .. import local as _local
    scope = _local.scope_of(rnd)
    plan = rnd.plan()
    if scope is None or not plan:
        return None
    rows = []
    for p in _pipeline.plan_parts(plan):
        if p.get("kind") in ("plot", "area") and p.get("x1") is not None \
                and _local.meets(scope, (p["x0"], p["z0"], p["x1"], p["z1"])):
            rows.append([p.get("name"), p["x0"], p["z0"], p["x1"], p["z1"],
                         p.get("site"), p.get("court_site")])
    return hashlib.sha256(json.dumps(sorted(rows, key=str), sort_keys=True)
                          .encode()).hexdigest()[:16]


def _front_thresholds(rnd, plan: dict, scope: dict, cells: dict, have: set,
                      ground=None) -> tuple:
    """Doorsteps for kept plots in the registered section that carry none: on the plot's
    own front, at the pad edge, off a lane cell one column outside the lot -- and, where
    the kept lane stops short of the plot, the spur that brings it along the front row
    (at most `SPUR_MAX` cells, over ground a step from the lane's). Returns
    `(thresholds, {cell: lane record})`."""
    from .. import circulate, local as _local
    from ..buildlib import site_pad_rect
    sec = ((rnd.flags.get("section") or {}).get("rect")) or None
    if not sec or not plan:
        return [], {}
    out, spur = [], {}
    SPUR_MAX = 12

    def _spur(row, want):
        """Lane cells from the nearest lane cell on `row` to `want`, or None."""
        have_ = [c for c in row if c in cells]
        if not have_:
            return None
        src = min(have_, key=lambda c: abs(c[0] - want[0]) + abs(c[1] - want[1]))
        d = abs(src[0] - want[0]) + abs(src[1] - want[1])
        if d > SPUR_MAX:
            return None
        y = int(cells[src]["y"])
        path, cur = {}, src
        while cur != want:
            cur = (cur[0] + (want[0] > cur[0]) - (want[0] < cur[0]),
                   cur[1] + (want[1] > cur[1]) - (want[1] < cur[1]))
            g = ground(*cur) if ground else y
            if g is None or abs(int(g) - y) > 1:
                return None
            path[cur] = {"y": y, "rank": 3, "face": None}
        return path
    step = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
    into = {"north": "south", "south": "north", "west": "east", "east": "west"}
    for leaf in _pipeline_plan_parts(plan):
        if leaf.get("kind", "plot") != "plot" or leaf.get("name") in have:
            continue
        front = leaf.get("front")
        if front not in step or leaf.get("x1") is None:
            continue
        r = (leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"])
        if r[2] < sec[0] or r[0] > sec[2] or r[3] < sec[1] or r[1] > sec[3]:
            continue
        if _local.meets(scope, r):
            continue
        pad = site_pad_rect(*r, leaf.get("attached") or [],
                            *([int(leaf["inset"])] if leaf.get("inset") is not None else []))
        dx, dz = step[front]
        if front in ("north", "south"):
            zd = pad[1] if front == "north" else pad[3]
            zl = r[1] - 1 if front == "north" else r[3] + 1
            mid = (pad[0] + pad[2]) // 2
            opts = sorted(range(pad[0] + 1, pad[2]), key=lambda x: abs(x - mid))
            pick = next(((x, zl, x, zd) for x in opts if (x, zl) in cells), None)
            if not pick and opts:
                row = [(x, zl) for x in range(r[0] - SPUR_MAX, r[2] + SPUR_MAX + 1)]
                got_ = _spur(row, (opts[0], zl))
                if got_:
                    spur.update(got_)
                    cells = {**cells, **got_}
                    pick = (opts[0], zl, opts[0], zd)
            if pick:
                lx, lz, ddx, ddz = pick
        else:
            xd = pad[0] if front == "west" else pad[2]
            xl = r[0] - 1 if front == "west" else r[2] + 1
            mid = (pad[1] + pad[3]) // 2
            opts = sorted(range(pad[1] + 1, pad[3]), key=lambda z: abs(z - mid))
            pick = next(((xl, z, xd, z) for z in opts if (xl, z) in cells), None)
            if not pick and opts:
                row = [(xl, z) for z in range(r[1] - SPUR_MAX, r[3] + SPUR_MAX + 1)]
                got_ = _spur(row, (xl, opts[0]))
                if got_:
                    spur.update(got_)
                    cells = {**cells, **got_}
                    pick = (xl, opts[0], xd, opts[0])
            if pick:
                lx, lz, ddx, ddz = pick
        if not pick:
            continue
        y = int(cells[(lx, lz)]["y"])
        out.append(circulate.Threshold(str(leaf.get("name")), lx, lz, y, into[front],
                                       (ddx, y + 1, ddz), step=0))
    return out, spur


def _pipeline_plan_parts(plan):
    from .. import pipeline as _pl
    return _pl.plan_parts(plan or {})


def stage_circulation(rnd: Round, be, results: dict) -> dict:
    """Route C: procedural, deterministic, no model call. Live only -- it owns the
    ground it crosses, so it writes blocks."""
    net = rnd.network()
    if not be.live:
        # **Under a local scope the network on disk is the city's, not this plan's.**
        # The quarter design round: re-laying a district no longer deletes the lanes
        # (they are the boundary condition the scope merges into), so "a network exists"
        # stopped meaning "this plan was routed". What the scope was routed for is
        # recorded (`scope_sites`) and a plan whose sites in the scope differ is routed
        # again.
        stale_scope = None
        if net is not None and getattr(be, "dry_run", False):
            want = _scope_sites(rnd)
            if want is not None:
                got = None
                with contextlib.suppress(Exception):
                    got = json.load(open(rnd.rel("circulation.json"))).get("scope_sites")
                if got != want:
                    stale_scope = want
        if net is not None and stale_scope is None:
            return {"skipped": "offline backend; the network is already on disk",
                    "cells": len(net.cells), "thresholds": len(net.thresholds)}
        if getattr(be, "dry_run", False):
            return _dry_circulation(rnd, be)
        return {"skipped": "circulation writes blocks; run it on a live backend"}
    if net is not None:
        # Idempotence the cheap way: a lane regraded over a standing lane is at best
        # wasted writes and at worst E009 against its own thresholds.
        return {"skipped": "network.json already exists -- delete it (after a "
                           "snapshot) to re-route", "cells": len(net.cells)}
    import subprocess
    p = subprocess.run(
        [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
         os.path.join(_pipeline.ROOT, "src", "ethoslm", "pipeline", "_commands", "circulate_run.py")],
        env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name),
        capture_output=True, text=True)
    be.refresh()
    net = rnd.network()
    out = {"returncode": p.returncode, "log_tail": p.stdout[-2000:]}
    if net is not None:
        out.update(cells=len(net.cells), thresholds=len(net.thresholds))
    elif p.returncode == 0:
        out["error"] = "circulate_run.py exited 0 but wrote no network.json"
    return out


#: How far outside the site a cached volume reaches: the lanes, the terraces and the
#: gate approaches all read ground outside the footprint, and the server path has always
#: taken this much.
CACHE_PAD = 48

#: Where a round records the squares its site search chose and whose ground could not
#: then be read. The dry search excludes them, so a run walks down its own ranking
#: instead of stopping on the first square the save does not fully hold.
UNREADABLE_SITES = "sites_unreadable.json"


def stage_cache(rnd: Round, be, results: dict) -> dict:
    p = rnd.rel(rnd.base_volume)
    if os.path.exists(p):
        vol = offline.load_volume(p)
        # **Cached for the site that was chosen, or cached again.** The closure round's
        # transfer case: the spec grew its footprint, the search chose a larger square
        # at the same origin, and this stage handed back the volume cached for the
        # smaller one because the file existed. A dry run's base volume is the ground
        # under the chosen site plus its pad and nothing else; a volume for another site
        # is set aside by name.
        want = rnd.site or rnd.chosen_site() or {}
        if want and not rnd.site and not be.live:
            pad = int(rnd.flags.get("cache_pad", CACHE_PAD))
            X, Z, S = int(want["origin"][0]), int(want["origin"][1]), int(want["size"])
            if (int(vol.x0), int(vol.z0)) != (X - pad, Z - pad) \
                    or int(vol.shape[0]) != S + 2 * pad or int(vol.shape[2]) != S + 2 * pad:
                stale = rnd.rel(rnd.base_volume.replace(
                    ".npz", f".stale.{vol.x0 + pad}_{vol.z0 + pad}_{vol.shape[0] - 2 * pad}.npz"))
                os.replace(p, stale)
                for f in (rnd.base_volume.replace(".npz", ".before-plateau.npz"),
                          "site.json", "plateau.json", "ground.npz", "ground.json"):
                    if os.path.exists(rnd.rel(f)):
                        os.replace(rnd.rel(f), rnd.rel(f"stale.{os.path.basename(f)}"))
                from .. import deps as _deps_c
                for a in ("plateau", "ground", "plan"):
                    with contextlib.suppress(Exception):
                        _deps_c.invalidate(rnd, a, "the site search chose a different site")
                print(f"   cache: the base volume was cached for ({vol.x0 + pad},"
                      f"{vol.z0 + pad}) {vol.shape[0] - 2 * pad}x{vol.shape[0] - 2 * pad} "
                      f"and the search chose ({X},{Z}) {S}x{S}; it is set aside and the "
                      f"ground is read again", flush=True)
                vol = None
        if vol is not None:
            return {"skipped": "already cached -- the base volume is a fixture, not a "
                               "running state", "path": p, "shape": list(vol.shape)}
    if not be.live and getattr(be, "dry_run", False):
        # **A dry run caches its own ground, off the save's region files, with no
        # server.** v2, C5, and the gap a fresh site found: the search reads region
        # files directly (`find_site.field_from_region_files`) and chose this square
        # without a server, and then the run refused to go on until a server session
        # wrote the same ground into `world.npz` -- so "one sentence in, a finished
        # place out, offline" was true only for a site somebody had already cached.
        # `savedworld.SavedWorld.volume` is the same read at the same fidelity, a chunk
        # at a time; the vertical range is trimmed to the ground it holds, as the server
        # path trims it.
        s = rnd.site or rnd.chosen_site() or {}
        o = s.get("origin", ["?", "?"])
        if not s:
            return {"status": "error", "stop": True,
                    "error": "this round's config names no site and no site search has "
                             "chosen one"}
        from .. import observe, savedworld
        world_dir = os.path.join(_pipeline.ROOT, "run", "server", "world")
        if not os.path.isdir(os.path.join(world_dir, "region")):
            return {"status": "error", "stop": True,
                    "error": f"a dry run is scored against a cached volume and there is "
                             f"none at {os.path.relpath(p, _pipeline.ROOT)}, and this "
                             f"checkout has no region files under "
                             f"{os.path.relpath(world_dir, _pipeline.ROOT)} to read one "
                             f"from: the ground under ({o[0]},{o[1]}) "
                             f"{s.get('size')}x{s.get('size')} has to be read once, by a "
                             f"session that writes nothing --\n"
                             f"      bash scripts/mcrun.sh src/ethoslm/pipeline/_commands/"
                             f"cache_world.py   (ETHOSLM_SETTLEMENT={rnd.name})"}
        X, Z, S = int(o[0]), int(o[1]), int(s["size"])
        pad = int(rnd.flags.get("cache_pad", CACHE_PAD))
        t0 = time.perf_counter()
        try:
            vol = savedworld.SavedWorld(world_dir).volume(X - pad, Z - pad,
                                                          S + 2 * pad, S + 2 * pad)
        except Exception as e:                   # noqa: BLE001 -- reported by name
            # **A site whose ground cannot be read is the site owner's finding.** The
            # search ranks the squares `out/sites/` holds, and a square's *cached height
            # field* existing does not mean the save holds every generated chunk of the
            # padded volume this stage reads. Until this the run simply stopped, and a
            # request that could not be built for that reason reported a chunk number --
            # which is the failure the realization round's refusal case exists to
            # prevent, in a different place. So the square is written down as unreadable
            # and the site search is asked again: the next-ranked square is chosen on
            # the next invocation, and a run that exhausts them says that, by name,
            # instead of naming a chunk.
            from .. import deps as _deps_c
            ref_p = rnd.rel(UNREADABLE_SITES)
            rec = json.load(open(ref_p)) if os.path.exists(ref_p) else {"refused": []}
            here = {"origin": [X, Z], "size": S, "pad": pad,
                    "why": f"{type(e).__name__}: {e}"}
            if not any(r["origin"] == here["origin"] and r["size"] == S
                       for r in rec["refused"]):
                rec["refused"].append(here)
                json.dump(rec, open(ref_p, "w"), indent=1)
            for f in ("site_search.json", "site.json"):
                if os.path.exists(rnd.rel(f)):
                    os.replace(rnd.rel(f), rnd.rel(f"unreadable.{X}_{Z}.{f}"))
            _deps_c.invalidate(rnd, "site_search",
                               f"the ground under ({X},{Z}) cannot be read")
            print(f"   cache: the ground under ({X},{Z}) {S}x{S} cannot be read "
                  f"({type(e).__name__}); the square is refused and the site is "
                  f"searched again", flush=True)
            # the driver re-enters the stage that asked, and the stage that has to run
            # again is the one before it, so this stage runs it: the search excludes
            # every refused square (`_dry_site_search`) and the re-entry caches the next
            # one it ranks
            from .. import place as _place
            got = _place.stage_site_search(rnd, be, {})
            if got.get("stop"):
                return {**got, "refused_sites": rec["refused"],
                        "error": (f"{got.get('error')}. "
                                  f"{len(rec['refused'])} square(s) were chosen and "
                                  f"then refused because this save does not hold every "
                                  f"generated chunk of the padded ground they need")}
            return {"status": "reenter", "refused_site": here,
                    "refused_total": len(rec["refused"]),
                    "why": (f"the ground under ({X},{Z}) {S}x{S} cannot be read off "
                            f"this save's region files ({type(e).__name__}); it is "
                            f"recorded unreadable and the search runs again")}
        h, _wet = observe.ground_heights(vol)
        inner = h[pad:pad + S, pad:pad + S]
        y0 = max(vol.y0, int(inner.min()) - 8)
        y1 = min(vol.y0 + vol.codes.shape[1] - 1, int(h.max()) + 48)
        vol = observe.Volume(vol.x0, y0, vol.z0,
                             vol.codes[:, y0 - vol.y0:y1 - vol.y0 + 1, :], vol.palette)
        offline.save_volume(vol, p)
        be.refresh()
        secs = round(time.perf_counter() - t0, 1)
        mb = round(os.path.getsize(p) / 1e6, 1)
        record("cache_world", name=rnd.name, shape=list(vol.shape),
               states=len(vol.palette), mb=mb, seconds=secs, read="region files")
        print(f"   cache: {vol.shape} y {y0}..{y1} read off the region files with no "
              f"server, {mb} MB in {secs}s", flush=True)
        return {"path": p, "mb": mb, "shape": list(vol.shape), "y": [y0, y1],
                "pad": pad, "seconds": secs, "read": "region files, no server"}
    if not be.live:
        return {"error": "caching the world reads the server; run this stage --live"}
    import subprocess
    r = subprocess.run(
        [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
         os.path.join(_pipeline.ROOT, "src", "ethoslm", "pipeline", "_commands", "cache_world.py")],
        env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name), capture_output=True, text=True)
    out = {"returncode": r.returncode, "log_tail": r.stdout[-800:]}
    if os.path.exists(p):
        out["path"] = p
        out["mb"] = round(os.path.getsize(p) / 1e6, 1)
    else:
        out["error"] = "cache_world.py wrote no volume"
    return out


def stage_briefs(rnd: Round, be, results: dict) -> dict:
    """Compose each wave's massing and elaboration briefs from the plan.

        `scripts/make_settlement_prompts.py`, once per wave, in a subprocess because
        `settlement.STATE` is baked in at import from $ETHOSLM_SETTLEMENT. The wave's `plots`
        are its assignment: the structure ids the planner gave it, which are also the plot
        labels it will claim and the labels its `check.py` is scoped to. One list, three
        uses, so they cannot drift.

        Never overwrites: a brief on disk is what a builder was actually shown.
        
    """
    import subprocess
    out = {}
    for w in rnd.waves:
        wave = w["name"]
        ids = list(w.get("plots") or [])
        rec = {"plots": ids}
        if not ids:
            rec["error"] = "this wave names no plots, so it has no assignment"
            out[wave] = rec
            continue
        for kind, suffix in (("massing", "_massing_prompt.md"),
                             ("elaborate", "_elaborate_prompt.md")):
            p = rnd.rel(f"{wave}{suffix}")
            if os.path.exists(p):
                rec[kind] = "already written"
                continue
            r = subprocess.run(
                [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
                 os.path.join(_pipeline.ROOT, "scripts", "make_settlement_prompts.py"),
                 f"--{kind}", wave, *ids],
                env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name),
                capture_output=True, text=True)
            rec[kind] = (f"{len(open(p).read().split())} words" if os.path.exists(p)
                         else f"FAILED: {(r.stderr or r.stdout)[-300:]}")
        out[wave] = rec
    return out


def _reconcile_surfaces(rnd: Round, path: str, built_path: str) -> dict:
    """Rewrite `surfaces.json` as what the **assembled** world makes true of it.

        Reported and never raised: a record that could not be reconciled stays exactly as
        emission left it, which is the weaker evidence it always was, and says so. See
        `ethoslm.surfaces.reconcile` for what is taken out and why.
        
    """
    from .. import offline as offline_mod, surfaces as surfaces_mod
    try:
        doc = surfaces_mod.read(path) or {}
        if doc.get("reconciled"):
            return {"skipped": "already reconciled against this built world"}
        was = surfaces_mod.editable_cells(doc)
        doc, report = surfaces_mod.reconcile(
            doc, offline_mod.load_volume(built_path),
            note="reconciled by stage_finish against the assembled world")
        json.dump(doc, open(path, "w"), indent=1)
        now = surfaces_mod.editable_cells(doc)
        print(f"   surfaces: {was} -> {now} editable cell(s) after reconciliation "
              f"against the assembled world", flush=True)
        return {"editable_before": was, "editable_after": now, "report": report}
    except Exception as e:                        # noqa: BLE001 -- reported, not raised
        return {"status": "unreconciled",
                "why": (f"the surface record could not be reconciled against the "
                        f"assembled world ({type(e).__name__}: {e}); it stands as "
                        f"emission left it, which is weaker evidence")}


def stage_finish(rnd: Round, be, results: dict) -> dict:
    """The seam between what was built and what was there, then the built world cached.

        `scripts/finish_run.py` rounds the lip of every cut face and dresses the ground the
        round touched, and lints afterwards because a pass that moves ground can sever a
        network. Then `world_built.npz`, which is what `lint`, `render` and every card are
        read off.
        
    """
    import subprocess
    if not be.live and getattr(be, "dry_run", False):
        # **The seam pass is a ground pass and a dry run does not run it.** What it
        # does. What a dry run does need is the same file every measure reads off:
        # `world_built.npz`, which is the cached volume as the waves left it.
        built = rnd.rel("world_built.npz")
        if not ((results.get("parts") or {}).get("skipped") and os.path.exists(built)):
            be.save(built)
        # **The built world is stamped for the plan it was built from.** See
        # `stage_parts`: a rebuild that nothing asked for is not a replay.
        from .. import deps as _deps_f
        sp = rnd.rel("surfaces.json")
        if os.path.exists(sp):
            from .. import surfaces as surfaces_mod
            doc = surfaces_mod.read(sp) or {}
            if doc.get("built_digest") != _deps_f.content_print(built):
                surfaces_mod.write(sp, doc.get("parts") or [], candidate=doc.get("candidate"),
                                   built_digest=_deps_f.content_print(built),
                                   note=doc.get("note") or "")
            # **...and the record on disk says what is true of the assembled world.**
            # The design round, worker C's second request. Each part's surface record is
            # made inside one builder before its neighbours exist, so a cell a later
            # part overwrote, one another part protects, one that no longer stands and
            # one that nothing can see are all still in it. Between a fifth and a half
            # of the recorded editable cells are of those kinds on the cached candidates
            # (22,110 -> 12,688 on the farm; 262,618 -> 167,448 on the city). Half a
            # second on a city-sized volume, and it is the difference between a record
            # and a record that is right.
            _reconcile_surfaces(rnd, sp, built)
        with contextlib.suppress(ValueError):
            _deps_f.stamp(rnd, "built", outputs=["parts.json", "world_built.npz"]
                          + (["surfaces.json"] if os.path.exists(sp) else []),
                          plan=rnd.plan(), note="the dry run's built volume")
        rows = [r for w in (results.get("parts") or {}).get("waves", [])
                for r in w["parts"]]
        return {"dry_run": True, "world_built": built,
                "mb": round(os.path.getsize(built) / 1e6, 1),
                "parts_built": sum(1 for r in rows if r["status"] == "built"),
                "parts_not_built": [r["part"] for r in rows
                                    if r["status"] != "built"],
                "skipped": "the seam pass moves ground and is the live run's; the "
                           "built world is cached from the volume the waves left"}
    if not be.live:
        return {"error": "the finish pass moves ground; run this stage --live"}
    # **The one publish.** v2, A1: a live round builds its parts against the volume,
    # exactly as a dry run does, and the world is written here -- once, in
    # `Builder.flush`'s two passes, the second of which computes the connective states
    # only a running server knows. It is before the seam pass because that pass runs in
    # its own process against the world, and there is nothing for it to dress until the
    # buildings are in it.
    published = be.publish() if hasattr(be, "publish") else None
    if published and published.get("published"):
        print(f"  published {published['published']} blocks in "
              f"{published.get('seconds')}s", flush=True)
    # "Leave the world as found if anything fails before the finish stage." A finish
    # pass over a half-built town regrades the seams of buildings that are not there
    # yet, and there is no undo short of the snapshot.
    unbuilt = [k for k, v in (results.get("waves") or {}).items()
               if isinstance(v, dict) and "status" in v and v["status"] != "built"]
    # ...and the same rule for a place built from parts. A finish pass over a district
    # with a quarter missing regrades the seams of buildings that are not there.
    unbuilt += [r["part"] for w in (results.get("parts") or {}).get("waves", [])
                for r in w["parts"] if r["status"] != "built"]
    if unbuilt:
        # ...and the built world is cached anyway. Refusing to dress the seams is not a
        # reason to refuse to *record* what stands: `world_built.npz` is what `lint`,
        # `render`, the cards and the readout are all read off, and a round that ended
        # with one part refused would otherwise have measured nothing at all.
        from .. import render as _render
        _render.cache_built(rnd.state)
        built = rnd.rel("world_built.npz")
        return {"refused": "these waves are not built, so there is no seam to finish "
                           "and the world is left as it stands",
                "waves": unbuilt,
                "world_built": built if os.path.exists(built) else "NOT CACHED"}
    r = subprocess.run(
        [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
         os.path.join(_pipeline.ROOT, "src", "ethoslm", "pipeline", "_commands", "finish_run.py")],
        env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name), capture_output=True, text=True)
    be.refresh()
    out = {"returncode": r.returncode, "log_tail": r.stdout[-2000:]}
    if published is not None:
        out["published"] = published
    fin = rnd.rel("finish.json")
    if os.path.exists(fin):
        d = json.load(open(fin))
        out.update(columns=d.get("columns"), pieces=d.get("pieces"),
                   errors=len(d.get("lint", {}).get("findings", [])))
    from .. import render
    render.cache_built(rnd.state)
    built = rnd.rel("world_built.npz")
    out["world_built"] = built if os.path.exists(built) else "NOT CACHED"
    return out


def _wave_brief(rnd: Round, w: dict) -> dict:
    """The wave's briefs, from the files the round wrote when they were composed --
    <wave>_massing_prompt.md, <wave>_elaborate_prompt.md, optionally <wave>_spaces.md,
    overridable per wave in the config. In a replay the text is never shown to
    anything; cold, it is exactly what the builder sees, which is why a missing file
    is reported rather than silently defaulted."""
    wave = w["name"]
    parts = {}
    for key, default in (("massing", f"{wave}_massing_prompt.md"),
                         ("elaboration", f"{wave}_elaborate_prompt.md"),
                         ("spaces", f"{wave}_spaces.md")):
        p = rnd.rel(w.get(f"{key}_brief", default))
        if os.path.exists(p):
            parts[key] = open(p).read()
    return parts


def _wave_check_runs(rnd: Round, sub: str, builds: str) -> list:
    """Every `check.py` run this wave's builders made, in call order.

        Reported with no bar registered against it. It is the first number that says
        whether a builder that missed had looked once or looked ten times, and the spec
        says to read it before anything else if walkability misses.
        
    """
    out = []
    for f in sorted(os.listdir(builds)) if os.path.isdir(builds) else []:
        if not f.startswith("build_request_"):
            continue
        i = int(f[len("build_request_"):-len(".json")])
        for r in _pipeline.checked_runs(os.path.basename(_pipeline._blind_dir(rnd, sub, i))):
            out.append({"call": i, **{k: r.get(k) for k in
                                      ("sha256", "blocks", "writes", "errors",
                                       "entry_lines", "crashed", "seconds")}})
    return out


LINT_MARGIN = 8


def wave_scope(rnd: Round, wave: dict) -> tuple | None:
    """The rectangle one wave is lint-scoped to: its own parts plus `LINT_MARGIN`."""
    labels = set(wave.get("plots") or [])
    if not labels:
        return None
    rects = [p for p in rnd.plots() if p["label"] in labels]
    if not rects:
        return None
    return (min(p["x0"] for p in rects) - LINT_MARGIN,
            min(p["z0"] for p in rects) - LINT_MARGIN,
            max(p["x1"] for p in rects) + LINT_MARGIN,
            max(p["z1"] for p in rects) + LINT_MARGIN)


def stage_waves(rnd: Round, be, results: dict) -> dict:
    """the building half.

        The model seam is `stages.build_from_files` over arms/<arm>/<wave>/builds/, the
        pattern that drove step 3's sixteen builder calls: call i answers from
        build_<i>.py when it is on disk, and otherwise stages the brief and images as a
        request there and this wave reports `needs_model`, exactly like a cold
        judgement. Answer the request and re-run; everything before it replays.

        With every program on disk this is the replay of the *pipeline* rather than of
        the finished program (which is stage_programs): massing, its cards, elaboration,
        conformance and lint all re-run, no model, no server, and the result is held to
        the round's own records -- the dry-run block count, and pending-set identity with
        the shipped <wave>.py. Live, the shipped program is then committed through the
        backend's settlement_run path.
    """
    from .. import stages
    if not rnd.waves:
        return {"waves": 0, "note": "this round has no build waves"}
    flags = dict(rnd.flags)
    allow_try = bool(flags.pop("allow_try", False))
    allow_collide = bool(flags.pop("allow_collide", False))
    checker = bool(flags.pop("check", False))
    arm = flags.pop("arm", None) or stages.arm_name(**flags)
    vol = be.volume
    subs = [(w["name"], os.path.join(arm, w["name"])) for w in rnd.waves]
    out = {"adopted": _pipeline._collect_blinded(rnd, subs)} if checker else {}
    for w in rnd.waves:
        wave = w["name"]
        brief = _wave_brief(rnd, w)
        if flags.get("split") and not ("massing" in brief and "elaboration" in brief):
            out[wave] = {"status": "no_brief",
                         "error": f"split needs {wave}_massing_prompt.md and "
                                  f"{wave}_elaborate_prompt.md under {rnd.state}"}
            continue
        builds = rnd.rel("arms", arm, wave, "builds")
        try:
            res = stages.run(brief if flags.get("split") else
                             brief.get("elaboration", ""), rnd.state,
                             sight=flags.get("sight", False),
                             split=flags.get("split", False),
                             select=flags.get("select", 0),
                             spaces=flags.get("spaces", False),
                             build=stages.build_from_files(builds),
                             arm=os.path.join(arm, wave), name=wave, vol=vol,
                             allow_try=allow_try, allow_collide=allow_collide,
                             lint_scope=wave_scope(rnd, w))
        except stages.BuildNeeded as e:
            out[wave] = {"status": "needs_model", "request": e.request_path,
                         "images": e.images or []}
            if checker:
                # Not on the massing call. A massing is a silhouette in one grey
                # material -- it has no doors, no openings, no interiors, and its own
                # brief says so -- and handing it a findings brief full of rooms nobody
                # can walk into would push it to build the very things it is told not
                # to. That is a check deforming a design, which this project has now
                # named three times. The elaboration call is the one that decides
                # floors, doors, stairs and fittings, and it is the one that gets the
                # checker.
                i = int(os.path.basename(e.request_path)
                        [len("build_request_"):-len(".json")])
                out[wave]["blinded"] = _pipeline._blind_build(
                    rnd, os.path.join(arm, wave), e.request_path,
                    check=((w.get("plots") or [], wave)
                           if (i or not flags.get("split")) else None))
            continue
        except Exception as e:              # noqa: BLE001 -- reported, not raised
            # A program that preflights and then raises is a typo, not a wave. Bounced
            # once so the round does not lose the other five to it, exactly as a
            # candidate is.
            import traceback
            out[wave] = {"status": "crashed", "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-2000:],
                         **(_pipeline._bounce(rnd, os.path.join(arm, wave), builds,
                                    traceback.format_exc()) if checker else {})}
            continue
        rep = res["report"]
        if rep.get("error"):
            out[wave] = {"status": "error", "error": rep["error"],
                         "preflight": rep.get("preflight")}
            continue
        rec = {"status": "built", "pending": rep.get("pending"),
               "program": res["program"],
               "checks": (_wave_check_runs(rnd, os.path.join(arm, wave), builds)
                          if checker else None),
               "conformance": rep.get("conformance"), "lint": rep.get("lint")}
        ref = rnd.rel(f"dryrun_{wave}.json")
        if os.path.exists(ref):
            rec["recorded"] = json.load(open(ref)).get("blocks")
            rec["reproduces"] = rec["pending"] == rec["recorded"]
        shipped = rnd.rel(w.get("program", f"{wave}.py"))
        if os.path.exists(shipped):
            # The shipped program is the round's record of what the world got; the
            # staged path must land on the identical pending set, not a similar one.
            rec["matches_shipped"] = (
                res["builder"]._pending == be.execute(shipped, vol)._pending)
        else:
            import shutil
            shutil.copyfile(res["program"], shipped)
            rec["shipped"] = shipped
        if be.live:
            # Once, and only once. A `--wait` run re-enters this stage every few minutes
            # while the later waves are still being written, and committing an already-
            # committed pass a second time re-runs its program against a world that now
            # contains it and against a registry that has already given away its plot --
            # so `reserve()` comes back False and the program takes a branch nobody
            # wrote it to take.
            done = (json.load(open(rnd.rel("passes.json")))
                    if os.path.exists(rnd.rel("passes.json")) else {})
            if wave in done and not done[wave].get("error"):
                rec["commit"] = {"already": True,
                                 **{k: done[wave].get(k)
                                    for k in ("placed", "failed", "plots_claimed")}}
            else:
                rec["commit"] = be.run_pass(wave)
        out[wave] = rec
    return out


def _cand(rnd: Round) -> dict:
    c = rnd.candidates
    if not c:
        raise ValueError(f"{rnd.path}: this round has no `candidates` block")
    return c


def _cand_ids(rnd: Round) -> list:
    return [f"c{i}" for i in range(_cand(rnd)["k"])]


def _cand_arm(rnd: Round, cid: str) -> str:
    c = _cand(rnd)
    return os.path.join(c["arm"], c["wave"], cid)


def _cand_program(rnd: Round, cid: str) -> str:
    return rnd.rel("arms", _cand_arm(rnd, cid), f"{_cand(rnd)['wave']}.py")


def _cand_dir(rnd: Round) -> str:
    return rnd.rel(_cand(rnd).get("out", "selection"))


def _cand_plots(rnd: Round) -> list:
    """The wave's plots, from the settlement's own registry. Candidates build on the
    fixture's ground: the plots are not the experiment's to invent."""
    want = list(_cand(rnd)["plots"])
    plots = json.load(open(rnd.rel("plots.json")))
    return [p for p in plots if p["label"] in want]


_TYPE_HEADER = '''

# --------------------------------------------------------------------------- Everything
# above this line is the type program, exactly as it was written. Everything below it is
# the driver instantiating it: for each piece of ground, `site()` prepares it -- sounds
# the bed, decides between a deck, a platform and a plinth, lays the way in from the
# lane -- and hands back the part with its floor level, its sited footprint and its door
# cell on it. `build()` is then called with that part and `type_builder(part)`, which is
# the same library with the ground taken out of it: the terrain calls refuse by name and
# a write below the floor, or of a stair, raises.
# ---------------------------------------------------------------------------

'''


def instantiated_source(type_src: str, instances: list, mat=None, roof=None) -> str:
    """The type program plus the calls that instantiate it, as one runnable program.

        `instances` is [(plot, seed, params), ...] -- a plot dict, the seed for that
        instance, and whatever the type takes beyond the three fixed arguments. `mat` is
        the palette the ground work is faced in, which is the type's own voice: a stone
        platform under a building whose footing is stone is a plinth, and under one whose
        footing is something else it is somebody else's building.
        
    """
    out = [type_src.rstrip("\n"), _TYPE_HEADER]
    for plot, seed, params in instances:
        p = {k: plot[k] for k in _pipeline.PART_GEOMETRY if k in plot}
        kw = "".join(f", {k}={v!r}" for k, v in sorted((params or {}).items()))
        out.append(f"_part = site({p!r}, mat={mat!r}, roof={roof!r})")
        # The type's own answer, kept. `_part` is the very dict `site()` appended to
        # `Builder.parts`, so writing the result into it is how the driver finds out
        # what the type said -- and a type that **refuses** ("no massing stood on this
        # pad") is the one thing a round that instantiates 35 parts most needs to hear.
        # `role=`: the type's own `ROLE`, read off the module the type source just
        # defined, so `TypeBuilder` can hand a civic type the voice's civic silhouette
        # (demo-polish, 2a). A stored instance program written before this calls
        # `type_builder(_part)` and gets the voice's one roof, as it always did.
        out.append(f"_part['build'] = build(type_builder(_part, "
                   f"role=globals().get('ROLE')), _part, {int(seed)}{kw})")
    return "\n".join(out) + "\n"


def voice_palette(voice: str | None) -> dict | None:
    """The palette of a voice, as `site()` and `building()` take it.

        Read from `voices/<name>.json` through `styles.VOICES`, so a voice the place spec
        authored is here on the same terms as a hand-written one -- and all six roles are
        named, because the voice file is validated and a role that defaults to the wall is a
        decision nobody made.
        
    """
    from .. import styles
    v = styles.VOICES.get(voice or "")
    return dict(v["palette"]) if v else None


def voice_roof(voice: str | None) -> dict | None:
    """The silhouette of a voice, as `roof()`'s own four parameters -- and, under
    `"civic"`, the voice's civic silhouette where it names one (demo-polish, 2a).
    `TypeBuilder` takes the civic one for a type whose `ROLE` is `civic` and the
    four keys for every other; nothing else reads the extra key."""
    from .. import styles
    v = styles.VOICES.get(voice or "")
    if not (v and v.get("roof")):
        return None
    out = dict(v["roof"])
    if v.get("roof_civic"):
        out["civic"] = dict(v["roof_civic"])
    # `stage_parts` resolves it before a type is composed)
    out["chimney"] = v.get("chimney")
    return out


def check_voices(voice: str | None, place: str | None = None) -> list:
    """The two voices a type's checker stands it in: the one its author was given and
        **the place's own** -- the voice the round's place is in -- and never a voice
        named in this file. v2, C0.

        Two and not one, because a checker that saw one voice validated six of the eight
        types a city authored in a silhouette the city was never going to use. Until this
        the partner was the demo's authored voice, a library constant, so every future
        type of every other place was checked against one city's roof. The partner is the
        place's: where the author's voice **is** the place's, or there is no place, it is
        the voice on disk whose silhouette is least like the first's
        (`styles.partner_voice`) -- a voice with a **different explicit silhouette** and
        not a silent one, because a type whose own default silhouette is its author's
        stands in a silent voice exactly as it stands in that one, and a silent partner
        tests nothing of a type that carries its own roof. With no voice and no place the
        first is the plainest voice on disk (`styles.silent_voice`).
        
    """
    from .. import styles
    first = voice or place or styles.silent_voice()
    second = place if place and place != first else styles.partner_voice(first)
    return [first, second]


def instantiate(type_program: str, plot, vol, *, seed: int = 0, network=None,
                plots=None, params: dict | None = None, allow_collide: bool = False,
                check: bool = True, voice: str | None = None):
    """Execute a stored type program once, on one plot. Returns the Builder.

        The unit of the type layer: one type, one plot, one seed. A district is this over a
        list of plots, which is the only difference between a probe and a district.

        `check` runs the parameters past the file's own `PARAMS` first and refuses by name.
        It is on by default and off for the one caller that has already checked them --
        `stage_types`, which checks once per instance and reports the refusal rather than
        raising it.
        
    """
    from .. import offline
    if check:
        params = _pipeline.check_params(_pipeline.load_type(type_program)["params"], params,
                              where=os.path.basename(type_program))
    src = instantiated_source(open(type_program).read(),
                              [(plot, seed, dict(params or {}))],
                              mat=voice_palette(voice), roof=voice_roof(voice))
    return offline.run_program(type_program, vol, network=network, plots=plots,
                               allow_collide=allow_collide, src=src)


def _type_instances(rnd: Round, spec: dict) -> list:
    """[(plot, seed, params), ...] from a config's `type.instances`.

        An instance is `[plot_label, params]` and `seed` is a key of `params`, because a
        seed *is* a parameter -- the one the type is required to vary on.
        
    """
    plots = {p["label"]: p for p in json.load(open(rnd.rel("plots.json")))}
    out = []
    for row in spec.get("instances") or []:
        label, params = (row + [{}])[:2] if isinstance(row, list) else (row, {})
        if label not in plots:
            raise ValueError(f"{rnd.path}: no plot called {label!r} in plots.json")
        params = dict(params)
        out.append((plots[label], int(params.pop("seed", 0)), params))
    return out


def _types(rnd: Round) -> list:
    t = rnd.types
    if not t:
        return []
    return list(t.get("list") or [])


def _type_out(rnd: Round) -> str:
    return rnd.rel(rnd.types.get("out", "types"))


def _type_file(rnd: Round, spec: dict) -> str:
    return _pipeline._resolve(spec.get("file") or os.path.join("types", f"{spec['name']}.py"))


def _type_sub(rnd: Round, spec: dict) -> str:
    """The blinded arm path for this type's one builder call."""
    return os.path.join(rnd.types.get("arm", "types"), spec["name"])


def _type_rows(rnd: Round, spec: dict) -> list:
    """[(plot, seed, params, id)] for every instance of one type.

        The plots come from the settlement's own registry: a type's instances build on
        ground the round did not invent, exactly as a wave's plots are its assignment.
        
    """
    plots = {p["label"]: p for p in json.load(open(rnd.rel("plots.json")))}
    out = []
    for row in spec.get("instances") or []:
        label = row["plot"]
        if label not in plots:
            raise ValueError(f"{rnd.path}: no plot called {label!r} in plots.json")
        seed = int(row.get("seed", 0))
        out.append((plots[label], seed, dict(row.get("params") or {}),
                    f"{spec['name']}/{label}/{seed}"))
    return out


def _instance_program(rnd: Round, spec: dict, plot, seed: int) -> str:
    return os.path.join(_type_out(rnd), spec["name"],
                        f"{plot['label']}_{seed}.py")


TYPE_CONTRACT = """## What to write

Write a **type**, not a building -- one Python file, and nothing else:

    FORM = "european_vernacular"    # or east_asian, fortification, civic
    ROLE = "urban"                  # or rural, civic, defensive
    PARAMS = {"storeys": ("int", 1, 3), "<one more you pick>": ("choice", [...])}
    NEEDS = {"footprint": (7, 7, 20, 20), "frontage": "lane",
             "ground": "any", "clearance": 2}

    def build(b, part, seed, **params):
        ...

`build` is called once for each piece of ground below, with that ground's own
rectangle, its own seed and its own parameters.

    b            the library. Every call documented above is a method on it:
                 b.building(...), b.place_block(...), b.roof(...), b.fitting(...),
                 b.check_walkable(...), b.threshold(...). Use it exactly as you would
                 use the bare names.
    part         {"label", "x0", "z0", "x1", "z1", "floor_y", "door", "facing",
                 "voice"} -- a pad of ground that has **already been prepared for
                 you**, corners inclusive. Build on it and nowhere else.
    seed         an integer. **Two instances with different seeds have to be different
                 buildings** -- a different footprint, a different height, a different
                 roof, a different arrangement inside -- while still being the same
                 kind of building in the same voice. Use it: `random.Random(seed)` is
                 available, and so is arithmetic on it.
    params       whatever `PARAMS` declares, by keyword.

## The ground is not yours

The library sites every part before your `build()` is called. It sounds the bed under
the pad, decks it on piles where it stands in water, cuts and fills it to one level
where there is relief, plinths it to grade where there is not, and lays the way in from
the lane. So:

    part["floor_y"]   the y of the floor. **Build from it up and never below it.**
                      A person standing on the floor is at part["floor_y"] + 1, which
                      is where a door's lower leaf goes.
    part["door"]      (x, z): the cell the way in arrives at, in the wall on the
                      part["facing"] side. building() finds it for itself from the
                      reserved threshold; place a door by hand and this is where.

These calls are the library's and are **refused** if a type makes them, by name:
`foundation_to_grade`, `terrace`, `plinth`, `grade`, `clear_trees`,
`clear_ground_cover`, `approach`. So is any `place_block`, `place_cuboid` or
`fill_region` below `part["floor_y"]`, or of a stair -- treads come from `steps()`,
`flight()` and `roof()`, which decide a facing from the finished ground rather than
from a number in your program. `get_height` still reads, because looking at the ground
is not working it.

None of this is a restriction on the building. It is the whole of the ground work,
done, before you start.

## Say what ground you need

`NEEDS` is the other half of that. `PARAMS` says what may be varied; **`NEEDS` says what
your `build()` requires to stand at all**, and it is what stops a plan handing you a pad
you cannot use. A plan is validated against it before anything is built, and a part
whose pad is outside it is refused by name rather than built into nothing.

    footprint  (min_w, min_d, max_w, max_d) of `part`, the pad you are handed -- **not**
               of the plot somebody drew, which is bigger. Give the smallest pad your
               `build()` stands a whole building on and the largest it fills well. The
               pair is matched without regard to which way round it is, because you turn
               your building to its door side. An edge's pair is (width, run).
    frontage   "lane" if a person arrives at this from the street, "any" if not.
    ground     "any", or which of "dry", "steep" and "wet" you can stand on -- a plinth,
               a cut platform and a deck on piles over water. The library lays all three
               and none of them is your work; say so only if your building genuinely
               cannot be one of them.
    clearance  how many blocks outside your footprint have to belong to nobody else.
               Two for a building: the pad is laid a course wider than itself and the
               ground round it is cleared two further.

Declare it honestly. A minimum bigger than you need costs the place a plot; one smaller
than you need is a plan that is prepared and then refused, plot by plot.

`PARAMS` is what may be varied and within what, declared by you: a name mapped to
`("int", lo, hi)` or to `("choice", [...])`. Anything not declared there cannot be
asked for. Declare `storeys` and **one other parameter of your own choosing**, and
give `build` a sensible result for every value either of them can take -- each is
instantiated and checked.

Return whatever you like; nothing reads it.

## The palette is not yours either

`FORM` says which family of form this building belongs to -- `european_vernacular`,
`east_asian`, `fortification` or `civic` -- and that is the whole of what a type says
about itself. **It names no material.** The settlement's palette arrives on the part:

    part["voice"]     {"wall", "footing", "frame", "roof", "trim", "floor"} -- one
                      material family per role. Hand the whole dict to `building(mat=)`
                      and a single role to anything that takes one material.
    b.block(fam, kind)    the block of one shape of a family: "full", "stairs", "slab",
                      "wall", "post", "bare", "accent", "fine". Refuses by name where
                      the game has no such block -- there is no quartz fence.
    b.joinery(part["voice"], kind)   a "door", "fence", "trapdoor" or "gate", taken
                      from the first role of the voice that has one. Never refuses.
    part["roof"]      the voice's own silhouette -- `roof()`'s `profile`, `ends`,
                      `eave` and `tiers` -- or None where the voice names none.

**The roof's silhouette is not yours either.** `roof()` and `building()` are handed
the voice's `profile`, `ends`, `eave` and `tiers` from `part["roof"]` **whatever you
pass them**, and where the voice names none the library's own default stands; so do
not pass them, and do not build anything that depends on what they turn out to be.
`style` and `axis` are yours -- a lean-to is a shed by construction. Take the ridge
height from what `roof()` returns rather than from a slope you assumed, and keep
walls, floors and flights clear of wherever a steeper or shallower slope, an upturned
eave or a second tier would land. Your type is checked in two voices with two
different silhouettes and has to be clean in both.

A block name written into a type file is **refused at preflight as E014**, because a
type that names `cobblestone` is a building that can only ever be built in one voice
and would have to be written again for the next one. A pitch, a profile, a pair of
ends, an eave or a tier count written into it is **refused as E015**, for the same
reason about the roof.

## Say what it is for

`ROLE` is one word and it is a different question from `FORM`. `FORM` is the tradition
the building is built in; `ROLE` is which part of a settlement it belongs in:

    urban      a street building: it stands shoulder to shoulder with others on a lane
    rural      a building of the fields: a farmhouse, a byre, a steading
    civic      what a place holds at its middle, and what may stand anywhere: a temple,
               a square, a palace, a hall of a whole town
    defensive  a wall, a gate, a tower, a keep

A district is planned out of the types whose role it is for, plus the `civic` ones. A
farmhouse offered as the ordinary house of a city street is how a city ends up made of
farmhouses, which is a thing that has happened here: say which of the four this is.

**No top-level code that places blocks.** The file is executed to read `FORM` and
`PARAMS` and to define `build`, and the function is then called once per instance;
anything that built at import time would build once for every instance and once more
besides. Module-level constants and helper functions are fine.

Write the whole thing as one Python program, and nothing else.
"""


TYPE_CONTRACT_KINDS = {
    "plot": "",
    "edge": """
## This type builds an **edge**

An edge is a line on the ground with a width -- a town wall, a quay, a terrace edge --
and it is handed to you already graded, segment by segment:

    KIND = "edge"          declare this at the top of the file, beside FORM.

    part["segments"]  [{"a": [x, z], "b": [x, z], "axis": "x"|"z", "floor_y": int,
                       "cells": [[x, z], ...]}, ...] in order along the line. Each
                      segment's `cells` are every column it covers across its width,
                      and `floor_y` is the level the library graded that segment to:
                      **the top of the footing is at floor_y, so your first course goes
                      at floor_y + 1.** Two segments meet at a vertex, and the vertex
                      column is in both of them.
    part["vertices"]  the corners, [[x, z], ...]. Where two segments join.
    part["path"]      the centre line you were given, and part["width"] its width.
    part["floor_y"]   the lowest of the segment levels, and the level below which you
                      may not write anything.

Build **one segment at a time and join them at the vertices**. A segment may be graded
to a different level from the one beside it, because the ground under a long wall is
not one level; where they differ the wall steps at the vertex, and a step is not a gap.
""",
    "point": """
## This type builds a **point**

A point is one cell and a way it faces -- a gatehouse, a well head, a beacon:

    KIND = "point"         declare this at the top of the file, beside FORM.
    PASSAGE = True         declare this too if a lane may cross the part here. A gate
                           is the one place a wall may be crossed, and that is a fact
                           about the type rather than about where it was put.

    part["at"]        [x, z], the cell the plan put you at.
    part["facing"]    "north" | "south" | "east" | "west" -- the way the thing is
                      turned. Whatever you build has to be turned this way: a gate
                      faces along it and its posts stand across it.
    part["floor_y"]   the y of the prepared ground, as for a plot. Build from it up.
    part["x0"..."z1"] the pad the library prepared round the cell.
""",
    "area": """
## This type builds an **area**

An area is a rectangle of open ground -- a market square, a yard, a court. The library
has brought the whole of it to one level and joined it to the lane before you are
called, so what is left is what makes it a place rather than a slab:

    KIND = "area"          declare this at the top of the file, beside FORM.

    part["footprint"]  [x0, z0, x1, z1], corners inclusive.
    part["floor_y"]    the level of the ground. **Its top face is at floor_y**, so
                       paving replaces the course at floor_y and anything standing on
                       it starts at floor_y + 1.

An area has no walls and no rooms, so nothing here is measured on being able to walk
into it -- it is measured on being able to walk **across** it. Leave it open.
""",
}


def type_contract(kind: str = "plot") -> str:
    """The contract a type of this kind is written to. See `TYPE_CONTRACT_KINDS`."""
    return TYPE_CONTRACT + TYPE_CONTRACT_KINDS.get(kind, "")


def _plot_block(vol, p: dict) -> str:
    """The site section of a brief: bounds, the height of every column, what is on top."""
    from .. import observe
    h, _wet = observe.ground_heights(vol)
    x0, z0, x1, z1 = p["x0"], p["z0"], p["x1"], p["z1"]
    rows = ["    " + " ".join(f"{int(h[x - vol.x0, z - vol.z0]):3d}"
                              for x in range(x0, x1 + 1))
            for z in range(z0, z1 + 1)]
    hs = [int(h[x - vol.x0, z - vol.z0])
          for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
    surf = _surface_census(vol, p)
    top = ", ".join(f"{k} ({v})" for k, v in
                    sorted(surf.items(), key=lambda kv: -kv[1])[:6])
    return f"""A plot of open ground in a world that already exists. Build inside it and nowhere else.

x from {x0} to {x1}, z from {z0} to {z1}. That is {x1 - x0 + 1} by {z1 - z0 + 1} blocks.
Vertically you may use y from {min(hs) - 8} to {max(hs) + 60}.

The ground is not flat. Surface height across the plot ranges from y={min(hs)} to
y={max(hs)}.

Here is the surface height of every column in the plot. Rows run north to south
(increasing z), columns run west to east (increasing x):

{chr(10).join(rows)}

Blocks found on the surface here: {top}.

Use get_height(x, z) for the exact height at any column.
"""


def _surface_census(vol, p: dict) -> dict:
    from .. import observe
    h, _wet = observe.ground_heights(vol)
    out: dict = {}
    for x in range(p["x0"], p["x1"] + 1):
        for z in range(p["z0"], p["z1"] + 1):
            s = vol.state(x, int(h[x - vol.x0, z - vol.z0]), z).split("[")[0]
            out[s] = out.get(s, 0) + 1
    return out


def stage_type_briefs(rnd: Round, be, results: dict) -> dict:
    """One brief per type: the API, the voice, the contract, the ground, the kind.

        **And nothing else.** No measure, no bar, no mention that other types exist or that
        anything will be compared -- the rule every builder call in this project is staged
        under. Never overwrites: a brief on disk is what a builder was actually shown.
    """
    if not rnd.types:
        return {"note": "this round has no types"}
    out_dir = os.path.join(_type_out(rnd), "briefs")
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for spec in _types(rnd):
        p = os.path.join(out_dir, f"{spec['name']}.md")
        if os.path.exists(p):
            out[spec["name"]] = {"brief": p, "note": "already written",
                                 "words": len(open(p).read().split())}
            continue
        fixtures = _pipeline._fixtures_for(rnd, spec)
        text = type_brief(rnd, spec, fixtures)
        open(p, "w").write(text)
        out[spec["name"]] = {"brief": p, "words": len(text.split()),
                             "voice": spec["voice"], "part": spec.get("part", "plot"),
                             "fixtures": [f"{f['round']}/"
                                          f"{f.get('plot') or f.get('part')}"
                                          for f in fixtures]}
    return out


def type_brief(rnd: Round, spec: dict, fixtures: list | None = None) -> str:
    """One type's brief: the API, the voice, the contract for its kind, the ground it
    will be checked on as sited, and the request. What `stage_type_briefs` writes for
    a round's own types and what the library's growth (v2, C3) writes for a type the
    plan found missing -- one composition, so an author at plan time is shown exactly
    what an author in a types round is."""
    from .. import styles
    from ..buildlib import API_DOC
    # A3: a type is checked on the ground of its own kind, and told about that ground
    # and no other.
    if fixtures is None:
        fixtures = _pipeline._fixtures_for(rnd, spec)
    census: dict = {}
    blocks = []
    for f in fixtures:
        frnd, _fbe = ((rnd, None) if f["round"] == rnd.name
                      else _pipeline._fixture_round(f["round"]))
        fvol = frnd.volume()
        if f.get("kind") in ("edge", "point", "area"):
            q = _pipeline.fixture_part(f, {})
            r = _pipeline.part_rect(q)
            for k, v in _surface_census(
                    fvol, {"x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]}).items():
                census[k] = census.get(k, 0) + v
            blocks.append(f"### {q['label']}\n\n"
                          + _sited_block(frnd, fvol, q,
                                         voice_palette(spec.get("voice"))))
            continue
        q = {r["label"]: r for r in
             json.load(open(frnd.rel("plots.json")))}[f["plot"]]
        for k, v in _surface_census(fvol, q).items():
            census[k] = census.get(k, 0) + v
        blocks.append(f"### {q['label']}\n\n"
                      + _sited_block(frnd, fvol, q,
                                     voice_palette(spec.get("voice"))))
    return (API_DOC + "\n" + styles.voice_card(spec["voice"], census) + "\n"
            + type_contract(spec.get("part", "plot"))
            + "\n## The pieces of ground your function will be called on\n\n"
            + "\n\n".join(blocks)
            + f"\n## What to build\n\n{spec['request']}\n")


def _sited_block(rnd: Round, vol, p: dict, mat) -> str:
    """One fixture's ground, as the type will actually receive it."""
    from .. import stages
    from ..buildlib import Builder
    from ..frontage import Frontage
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, rnd.network()) if rnd.network() else None
    b.registry = stages._Registry(rnd.state)
    if p.get("kind") in ("edge", "point", "area"):
        return _sited_part_block(b, p, mat)
    part = b.site(dict(p, kind="plot"), mat=mat)
    x0, z0, x1, z1 = part["footprint"]
    ground = {"deck": "a deck on piles carried to the bed, because this ground is "
                      "under water",
              "platform": "a platform cut and filled to one level, because this "
                          "ground has relief in it",
              "plinth": "a plinth carried to grade, because this ground is level"}
    return f"""A pad of prepared ground in a world that already exists, with the way in
from the lane already laid. Build on it and nowhere else.

x from {x0} to {x1}, z from {z0} to {z1}. That is {x1 - x0 + 1} by {z1 - z0 + 1} blocks.
The floor is at y={part['floor_y']}: `part["floor_y"]`, the same number, every time.
A person standing on it is at y={part['floor_y'] + 1}.
Vertically you may use y from {part['floor_y']} to {part['floor_y'] + 60}.

The way in is from the {part['facing']}, at the cell ({part['door'][0]}, \
{part['door'][1]}) -- `part["door"]`.

The library prepared this pad by laying {ground.get(part['ground'], part['ground'])}.
"""


def _sited_part_block(b, p: dict, mat) -> str:
    """One edge, point or area fixture's ground, as the type will receive it. A3.

        The same job `_sited_block` does for a plot: run `site()` on the real cache and
        describe what came back. What differs is what there is to describe -- a wall gets
        segments and their levels, a gate gets a cell and a facing, a square gets a
        rectangle at one level -- and none of them gets a heightmap, for the reason none of
        them gets one: a heightmap in a brief is an invitation to level it.
        
    """
    part = b.site(_pipeline.fixture_part(p, {}), mat=mat)
    kind = part["kind"]
    if kind == "edge":
        rows = "\n".join(
            f"      {i + 1}. {s['a']} -> {s['b']}, along {s['axis']}, "
            f"{len(s['cells'])} columns, floor y={s['floor_y']}"
            for i, s in enumerate(part["segments"]))
        return f"""A line of prepared ground in a world that already exists: a footing
under every column, graded segment by segment, with nothing standing on it.

{len(part['segments'])} segment(s), in order along the line -- `part["segments"]`:

{rows}

The vertices, where two segments meet, are {part['vertices']} -- `part["vertices"]`.
The top of the footing is at each segment's own `floor_y`, so the first course of your
wall goes at `floor_y + 1`. The lowest of them is {part['floor_y']}, and
`part["floor_y"]` is that number: nothing may be written below it.
Vertically you may use y from {part['floor_y']} to {part['floor_y'] + 40}.

{part['sited']['reason']}
"""
    if kind == "point":
        return f"""A cell in a world that already exists, with a pad of prepared ground
round it and the way in from the lane already laid.

The cell is ({part['at'][0]}, {part['at'][1]}) -- `part["at"]` -- and it faces
{part['facing']} -- `part["facing"]`. Whatever you build there has to be turned that
way.

The pad round it runs x from {part['x0']} to {part['x1']}, z from {part['z0']} to \
{part['z1']}.
The floor is at y={part['floor_y']}: `part["floor_y"]`, the same number, every time.
A person standing on it is at y={part['floor_y'] + 1}.
Vertically you may use y from {part['floor_y']} to {part['floor_y'] + 40}.

{part['sited']['reason']}
"""
    x0, z0, x1, z1 = part["footprint"]
    return f"""A rectangle of prepared ground in a world that already exists, brought to
one level and joined to the lane. It is all floor.

x from {x0} to {x1}, z from {z0} to {z1}. That is {x1 - x0 + 1} by {z1 - z0 + 1} blocks.
The ground's top face is at y={part['floor_y']}: `part["floor_y"]`. A person standing on
it is at y={part['floor_y'] + 1}, and paving replaces the course at y={part['floor_y']}.
Vertically you may use y from {part['floor_y']} to {part['floor_y'] + 30}.

{part['sited']['reason']}
"""


def stage_types(rnd: Round, be, results: dict) -> dict:
    """Every type in the round: one builder call each, then every instance measured.

        The model seam is the file-backed, blinded one every arm here uses -- one call per
        type, answered as `build_0.py` in a hash-named scratch directory holding the brief,
        the checker and nothing else. The answer is *committed as a type file* rather than
        kept in the round's own state, because a type is meant to outlive the round that
        wrote it: that is the whole claim being tested.

        Each instance is then composed into an ordinary program and measured by
        `measure_program`, so the lint, the walk model, the render and the replay all see a
        program and none of them needs to know what a type is.
        
    """
    from .. import lint, stages
    if not rnd.types:
        return {"note": "this round has no types"}
    vol = be.volume
    out: dict = {"adopted": _pipeline._collect_blinded(
        rnd, [(s["name"], _type_sub(rnd, s)) for s in _types(rnd)])}
    for spec in _types(rnd):
        name = spec["name"]
        path = _type_file(rnd, spec)
        builds = rnd.rel("arms", _type_sub(rnd, spec), "builds")
        os.makedirs(builds, exist_ok=True)
        src_path = os.path.join(builds, "build_0.py")
        if not os.path.exists(path) and not os.path.exists(src_path):
            req = os.path.join(builds, "build_request_0.json")
            brief_path = os.path.join(_type_out(rnd), "briefs", f"{name}.md")
            if not os.path.exists(brief_path):
                out[name] = {"status": "no_brief", "error":
                             f"run --stage type_briefs first: {brief_path} is not there"}
                continue
            if not os.path.exists(req):
                json.dump({"i": 0, "images": [], "brief": open(brief_path).read()},
                          open(req, "w"), indent=1)
            out[name] = {"status": "needs_model", "request": req, "images": [],
                         "blinded": _pipeline._blind_build(rnd, _type_sub(rnd, spec), req,
                                                 check=("type", spec))}
            continue
        if not os.path.exists(path):
            # Adopt the builder's answer as the committed type file. Copy, never move:
            # the scratch directory stays as the record of what that builder was shown.
            import shutil
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copyfile(src_path, path)
        try:
            decl = _pipeline.load_type(path)
        except Exception as e:                  # noqa: BLE001 -- reported, not raised
            out[name] = {"status": "bad_type", "file": os.path.relpath(path, _pipeline.ROOT),
                         "error": f"{type(e).__name__}: {e}"}
            continue
        rec = {"status": "built", "file": os.path.relpath(path, _pipeline.ROOT),
               "form": decl["form"], "params": decl["params"],
               "lines": decl["lines"], "instances": {}}
        for (plot, seed, params, iid) in _type_rows(rnd, spec):
            rec["instances"][iid] = _measure_instance(
                rnd, be, spec, decl, plot, seed, params, vol, lint, stages)
        rec["checks"] = len(_pipeline.checked_runs(
            os.path.basename(_pipeline._blind_dir(rnd, _type_sub(rnd, spec), 0))))
        out[name] = rec
    path = os.path.join(_type_out(rnd), "instances.json")
    os.makedirs(_type_out(rnd), exist_ok=True)
    json.dump({k: v for k, v in out.items() if k != "adopted"},
              open(path, "w"), indent=1)
    out["written"] = path
    return out


def _measure_instance(rnd: Round, be, spec: dict, decl: dict, plot, seed: int,
                      params: dict, vol, lint, stages) -> dict:
    """One instance: compose, preflight, execute, measure. Refusals are reported."""
    prog = _instance_program(rnd, spec, plot, seed)
    os.makedirs(os.path.dirname(prog), exist_ok=True)
    try:
        kw = _pipeline.check_params(decl["params"], params, where=f"{spec['name']} on "
                                                        f"{plot['label']}")
    except ValueError as e:
        return {"status": "bad_params", "error": str(e)}
    src = instantiated_source(decl["src"], [(plot, seed, kw)],
                              mat=voice_palette(spec.get("voice")),
                              roof=voice_roof(spec.get("voice")))
    open(prog, "w").write(src)
    # E014 judges the type file; the composed program's own `site(..., mat={...})` names
    # the voice's materials and is the driver's line, not the type's.
    pal = lint.preflight(decl["src"], allow_try=True, palette=True)
    pre = lint.preflight(src, allow_try=bool(rnd.flags.get("allow_try")),
                         forbid=_pipeline.TYPE_FORBIDDEN)
    if not (pre.ok and pal.ok):
        return {"status": "error", "program": os.path.relpath(prog, _pipeline.ROOT),
                "params": kw, "seed": seed,
                "error": "preflight rejected it",
                "preflight": (pre if not pre.ok else pal).to_json()}
    try:
        m = _pipeline.measure_program(rnd, be, prog, {plot["label"]}, margin=6)
        with _watching_refusals() as refused:
            b = be.execute(prog, vol)
    except Exception as e:                      # noqa: BLE001 -- reported, not raised
        import traceback
        return {"status": "crashed", "program": os.path.relpath(prog, _pipeline.ROOT),
                "params": kw, "seed": seed, "error": f"{type(e).__name__}: {e}",
                "traceback": _pipeline.scrub_traceback(traceback.format_exc())[-1200:]}
    ctx = _pipeline.build_context(rnd, be, prog, pending=b._pending,
                        fittings=b.fitting_cells)
    scoped = _pipeline.standard_report(ctx, [p for p in ctx.plots
                                   if p["label"] == plot["label"]])
    entry = _pipeline.entry_lines(_pipeline.diagnose_entry(ctx, plot["label"]))
    ridge, foot = _instance_form(b, plot)
    # What the library made of the ground before the type built on it: which of the
    # three cases fired, where the floor ended up, and whether the way in was laid.
    # Reported, no bar -- but A0 came back with an instance sealed at 0.0% walkable and
    # three `approach()` refusals, and this is the row that would have said which part
    # of the ground was the reason.
    sited = dict(b.parts[-1]) if getattr(b, "parts", None) else {}
    return {"status": "measured", "program": os.path.relpath(prog, _pipeline.ROOT),
            "sited": {k: sited.get(k) for k in
                      ("ground", "floor_y", "footprint", "door", "facing")}
                     | {"ok": bool((sited.get("sited") or {}).get("ok")),
                        "reason": (sited.get("sited") or {}).get("reason")},
            # Reported, no bar. A refusal is the library saying its vocabulary does not
            # reach this shape, and that list is where the next primitive comes from.
            # Collected by wrapping the primitives for one execution, which is what the
            # builder was actually told while it had the pen.
            "refusals": {k: v for k, v in refused.items() if v},
            "plot": plot["label"], "seed": seed, "params": kw,
            # `own_lint_errors` is the build family on this instance's own plot, which
            # is how `settlement_run` scopes a pass and how every round since 9 has
            # counted a builder's own errors. The whole-region number rides along.
            "own_lint_errors": len([f for f in scoped.findings
                                    if f.code.startswith("E")]),
            "region_lint_errors": m["lint_errors"],
            "entry_lines": len(entry), "entry": entry[:8],
            "walk_pct": m["walk_pct"], "rooms": m["rooms"],
            "floor_cells": m["floor_cells"], "blocks": m["blocks"],
            "ridge_height": ridge, "footprint": foot,
            "roof_to_wall": (round(ridge / foot[2], 2)
                             if foot and foot[2] else None)}


_REFUSING_CALLS = ("building", "flight", "dais", "approach", "fitting", "doorway")


@contextlib.contextmanager
def _watching_refusals():
    """Every library call that refuses while this is open, with its reason.

        The primitives are wrapped rather than logged from inside, so the library stays a
        library and this stays a readout. Wrapped on `Builder`, which is the class every
        program actually calls, and unwrapped in a `finally` by putting the class dictionary
        back as it was -- a wrapper left behind would follow every later stage in the
        process, and an inherited method restored as an *own* attribute is a wrapper left
        behind that looks like a restore.
        
    """
    from .. import buildlib
    got: dict = {n: [] for n in _REFUSING_CALLS}
    got["extras"] = []
    own = {n: buildlib.Builder.__dict__.get(n) for n in _REFUSING_CALLS}

    def wrap(name, fn):
        def inner(self, *a, **k):
            res = fn(self, *a, **k)
            if isinstance(res, dict):
                if not res.get("ok", True):
                    got[name].append({"args": [str(x)[:20] for x in a[:5]],
                                      "reason": str(res.get("reason"))[:220]})
                for word, e in (res.get("extras") or {}).items():
                    if isinstance(e, dict) and not e.get("ok", True):
                        got["extras"].append({"word": word,
                                              "reason": str(e.get("reason"))[:220]})
            return res
        return inner

    for n in _REFUSING_CALLS:
        setattr(buildlib.Builder, n, wrap(n, getattr(buildlib.Builder, n)))
    try:
        yield got
    finally:
        for n, f in own.items():
            if f is None:
                delattr(buildlib.Builder, n)
            else:
                setattr(buildlib.Builder, n, f)


def _instance_form(b, plot) -> tuple:
    """(ridge height over the ground, (width, depth, wall height)) for one instance."""
    cells = [(x, y, z) for (x, y, z), s in b._pending.items()
             if s.split("[")[0] not in ("air", "cave_air", "void_air")]
    if not cells:
        return None, None
    xs = [c[0] for c in cells]
    zs = [c[2] for c in cells]
    ys = [c[1] for c in cells]
    ground = min(ys)
    ridge = max(ys) - ground
    # the wall height: how far up the mass still fills most of its own footprint. A roof
    # narrows; a wall does not.
    area = (max(xs) - min(xs) + 1) * (max(zs) - min(zs) + 1)
    per_y: dict = {}
    for (x, y, z) in cells:
        per_y.setdefault(y, set()).add((x, z))
    wall = 0
    for y in sorted(per_y):
        if len(per_y[y]) >= 0.5 * area:
            wall = y - ground + 1
    return ridge, (max(xs) - min(xs) + 1, max(zs) - min(zs) + 1, wall)


def stage_candidates(rnd: Round, be, results: dict) -> dict:
    """k independent candidates from one brief -- the half of selection that generates.

        What makes the later comparison mean anything is that the k runs are *independent*:
        each candidate is its own `stages.run` over its own builds directory, from the
        byte-identical brief, and **no builder is told it is one of several or that anything
        will be compared**. A builder that knows it is competing writes for the comparison,
        and the experiment is void.

        `select` is forced to 0 and a configured `select` is an error rather than a silent
        override: adaptive-k inside a candidate would put a second, different selection
        mechanism inside the thing being measured.

        The model seam is the file-backed one every arm here has used -- call i answers from
        `build_<i>.py` when it is on disk, otherwise stages the request and this candidate
        reports `needs_model`. Answer the requests, re-run, and everything already built
        replays without a model call.
        
    """
    from .. import stages
    c = rnd.candidates
    if not c:
        return {"note": "this round generates no candidates"}
    wave = c["wave"]
    w = next((x for x in rnd.waves if x["name"] == wave), {"name": wave})
    brief = _wave_brief(rnd, w)
    flags = dict(rnd.flags)
    flags.pop("arm", None)
    allow_try = bool(flags.pop("allow_try", False))
    if flags.get("select"):
        return {"error": "a candidate round runs select=0: per-massing adaptive-k "
                         "inside a candidate is a second selection mechanism inside "
                         "the one being measured"}
    if flags.get("split") and not ("massing" in brief and "elaboration" in brief):
        return {"error": f"split needs a massing and an elaboration brief for {wave} "
                         f"under {rnd.state}"}
    vol = be.volume
    out = {"adopted": _pipeline._collect_blinded(rnd)}
    for cid in _cand_ids(rnd):
        sub = _cand_arm(rnd, cid)
        builds = rnd.rel("arms", sub, "builds")
        if c.get("type"):
            out[cid] = _type_candidate(rnd, be, c, cid, sub, builds, brief, flags)
            continue
        try:
            res = stages.run(brief if flags.get("split")
                             else brief.get("elaboration", ""), rnd.state,
                             sight=flags.get("sight", False),
                             split=flags.get("split", False), select=0,
                             spaces=flags.get("spaces", False),
                             build=stages.build_from_files(builds),
                             arm=sub, name=wave, vol=vol, allow_try=allow_try)
        except stages.BuildNeeded as e:
            out[cid] = {"status": "needs_model", "request": e.request_path,
                        "images": e.images or [],
                        "blinded": _pipeline._blind_build(
                            rnd, _cand_arm(rnd, cid), e.request_path,
                            check=((c.get("plots") or []), wave)
                            if c.get("check") else None)}
            continue
        except Exception as e:              # noqa: BLE001 -- reported, not raised
            # A program that preflights and then dies at run time is one candidate's
            # problem. Letting it raise here would take the other k-1 with it, and a
            # round that loses three good candidates to one bad one has measured
            # nothing. Reported with the traceback, which is what a bounce is made of.
            import traceback
            out[cid] = {"status": "crashed", "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2000:],
                        **_pipeline._bounce(rnd, _cand_arm(rnd, cid), builds, traceback.format_exc())}
            continue
        rep = res["report"]
        if rep.get("error"):
            out[cid] = {"status": "error", "error": rep["error"],
                        "preflight": rep.get("preflight")}
            continue
        out[cid] = {"status": "built", "program": res["program"],
                    "pending": rep.get("pending"),
                    "conformance": rep.get("conformance"), "lint": rep.get("lint")}
    return out


def _type_candidate(rnd: Round, be, c: dict, cid: str, sub: str, builds: str,
                    brief, flags: dict) -> dict:
    """One candidate that is a *type* rather than a building.

        The model seam is the same file-backed one every arm here uses -- one builder call,
        answered as `build_0.py` -- and the only difference is what happens to the answer:
        it is composed with the calls that instantiate it and written to the candidate's own
        program path, so `stage_measures`, `stage_candidate_render`, the lint and the
        replay all see an ordinary program and none of them needs to know what a type is.
        
    """
    from .. import lint, stages
    os.makedirs(builds, exist_ok=True)
    spec = dict(c["type"])
    src_path = os.path.join(builds, "build_0.py")
    if not os.path.exists(src_path):
        req = os.path.join(builds, "build_request_0.json")
        if not os.path.exists(req):
            json.dump({"i": 0, "images": [],
                       "brief": brief if isinstance(brief, str)
                       else brief.get("elaboration", "")},
                      open(req, "w"), indent=1)
        return {"status": "needs_model", "request": req, "images": [],
                "blinded": _pipeline._blind_build(rnd, sub, req,
                                        check=((c.get("plots") or []), c["wave"])
                                        if c.get("check") else None)}
    instances = _type_instances(rnd, spec)
    src = instantiated_source(open(src_path).read(), instances,
                              mat=voice_palette(rnd.voice),
                              roof=voice_roof(rnd.voice))
    prog = _cand_program(rnd, cid)
    os.makedirs(os.path.dirname(prog), exist_ok=True)
    open(prog, "w").write(src)
    # The composed program carries the driver's own `site(..., mat={...})` line, which
    # names the voice's materials by definition; E014 is a rule about the *type file*,
    # so the palette check is run on that and the whole program on everything else.
    pal = lint.preflight(open(src_path).read(), allow_try=True, palette=True)
    if not pal.ok:
        return {"status": "error", "error": "preflight rejected the type program",
                "preflight": pal.to_json(), "program": prog}
    pre = lint.preflight(src, allow_try=bool(flags.get("allow_try")))
    if not pre.ok:
        return {"status": "error", "error": "preflight rejected the type program",
                "preflight": pre.to_json(), "program": prog}
    try:
        b = be.execute(prog, be.volume)
    except Exception as e:                      # noqa: BLE001 -- reported, not raised
        import traceback
        return {"status": "crashed", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:], "program": prog,
                **_pipeline._bounce(rnd, sub, builds, traceback.format_exc())}
    return {"status": "built", "program": prog, "pending": len(b._pending),
            "type": {"source": os.path.relpath(src_path, _pipeline.ROOT),
                     "lines": len(open(src_path).read().splitlines()),
                     "instances": [{"plot": p["label"], "seed": s, "params": kw}
                                   for (p, s, kw) in instances]}}


def _revise(rnd: Round) -> dict:
    r = rnd.revise
    if not r:
        raise ValueError(f"{rnd.path}: this round has no `revise` block")
    return r


def _revise_subjects(rnd: Round) -> list:
    """[(cid, sub, source_program, mine, brief, wave)] over every subject."""
    out = []
    for s in _revise(rnd)["subjects"]:
        src = _pipeline.Round.load(_pipeline._resolve(s["config"]))
        if src.name != rnd.name or src.base_volume != rnd.base_volume:
            raise ValueError(
                f"{s['config']}: subject round is {src.name}/{src.base_volume}, this "
                f"round is {rnd.name}/{rnd.base_volume} -- a draft revised against a "
                f"different world is not a revision of it")
        wave = src.candidates["wave"]
        w = next((x for x in src.waves if x["name"] == wave), {"name": wave})
        brief = _wave_brief(src, w).get("elaboration", "")
        mine = {p["label"] for p in _cand_plots(src)}
        for cid in _cand_ids(src):
            out.append({
                "id": f"{wave}/{cid}", "wave": wave, "cand": cid,
                "sub": os.path.join(_revise(rnd).get("arm", "repair"), wave, cid),
                "source": _cand_program(src, cid), "mine": mine, "brief": brief,
                "recorded": os.path.join(_cand_dir(src), "measures.json"),
                "margin": int((src.candidates.get("measures") or {})
                              .get("variety_margin", 6)),
            })
    return out


def _revise_dir(rnd: Round, subj: dict) -> str:
    return rnd.rel(_revise(rnd).get("out", "repair"), subj["wave"], subj["cand"])


REVISE_CARRY = (
    "## The program as it stands\n\nThis is the program that built what the findings "
    "describe. Revise it and return the **whole** program -- it is re-run whole "
    "against the same ground, so write a complete build and not a patch.\n\n"
    "```python\n{program}\n```\n\n"
    "The findings are in `findings.md` beside this brief. Fix what they name.\n")


def _revise_brief(subj: dict, source: str) -> str:
    return (subj["brief"].rstrip("\n") + "\n\n"
            + REVISE_CARRY.replace("{program}", source))


def _draft_row(rnd: Round, be, subj: dict, n: int, prog: str) -> dict:
    """Everything one draft is worth, computed once and cached against its own bytes.

        Executing the program and building the lint Context is ~45 s and `measure_program`
        is another ~45 s, so a candidate costs a minute and a half and the stage is re-run
        after every builder batch. Keyed by sha rather than by mtime: a bounced program that
        was overwritten with the same bytes is the same draft, and a re-adopted one is not.

        `measure_program` itself is untouched. Drafts 0, 1 and 2 go through identical code,
        which is the only reason the three columns can be put beside each other at all.
        
    """
    import hashlib
    from .. import lint
    sha = hashlib.sha256(open(prog, "rb").read()).hexdigest()
    if n:
        _record_call(rnd, subj, n, prog, sha, builds=os.path.dirname(prog))
    d = _revise_dir(rnd, subj)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"draft_{n}.json")
    if os.path.exists(p):
        row = json.load(open(p))
        if row.get("sha256") == sha and "entry_lines" in row:
            _check_recorded(subj, row, sha, n)
            json.dump(row, open(p, "w"), indent=1)
            return row
    pend = len(be.execute(prog, be.volume)._pending)
    ctx = _pipeline.build_context(rnd, be, prog)
    diag = _pipeline.diagnose_entry(ctx, sorted(subj["mine"])[0])
    mine = [q for q in ctx.plots if q["label"] in subj["mine"]]
    rep = _pipeline.standard_report(ctx, mine)
    region = lint.lint(ctx)
    lines = _pipeline.entry_lines(diag)
    text = _pipeline.revision_findings(rep, lines, pend, subj["wave"])
    open(os.path.join(d, f"findings_{n}.md"), "w").write(text)
    json.dump(diag, open(os.path.join(d, f"diagnosis_{n}.json"), "w"), indent=1)
    # The check's own question -- can somebody who came through the door get about
    # inside -- beside the from-outdoors number. Computed here rather than inside
    # `measure_program`, which stays byte-identical to the code that measured the first
    # drafts.
    iw = [r for r in ctx.interior_walk() if r.get("plot") in subj["mine"]]
    cells = sum(r["cells"] for r in iw)
    walk = sum(r["walkable"] for r in iw)
    row = {
        "draft": n, "sha256": sha, "program": os.path.relpath(prog, _pipeline.ROOT),
        "lines": len(open(prog).read().splitlines()), "pending": pend,
        # `lint_errors_brief` is what the builder was shown and what the stop rule
        # reads; `lint_errors_region` is the whole-region count `measure_program`
        # reports and the one the registered floor correction applies to. Both travel:
        # they are different numbers and the record has confused them once already.
        "lint_errors_brief": len([f for f in rep.findings if f.code.startswith("E")]),
        "lint_errors_region": len([f for f in region.findings
                                   if f.code.startswith("E")]),
        "entry_lines": len(lines),
        "own_door_pct": round(100 * walk / cells, 1) if cells else None,
        "findings": os.path.join(d, f"findings_{n}.md"),
        "measures": _pipeline.measure_program(rnd, be, prog, subj["mine"],
                                    margin=subj["margin"]),
    }
    _check_recorded(subj, row, sha, n)
    json.dump(row, open(p, "w"), indent=1)
    return row


def _record_call(rnd: Round, subj: dict, n: int, prog: str, sha: str,
                 builds: str) -> None:
    """One builder call, costed, once.

        There is no API key here, so a builder call is an out-of-process agent and the
        driver never holds the conversation -- it only knows what it put in the blinded
        directory and what came back. That is what is logged: `chars_in` is the brief plus
        the findings plus, on a bounce, the scrubbed error, and `chars_out` is the program
        that came back. `seconds` is 0 because wall clock is not the driver's to measure,
        exactly as the selection round's rows record it; the token figure is 4 chars a
        token and `measure.model_call` labels it as an estimate.

        Deduped against the log rather than against `_draft_row`'s cache, and deliberately.
        The cache is keyed on the program's bytes, so a stage re-run over a draft measured
        before this logging existed would record nothing at all and the round would report
        a cost far under what it spent. The log is the thing being appended to, so the log
        is the thing that says whether this call is already on it -- and a builder bounced
        and rewritten has different bytes and is correctly a second row.
        
    """
    req = os.path.join(builds, f"build_request_{n}.json")
    if not os.path.exists(req):
        return
    key = os.path.basename(_pipeline._blind_dir(rnd, subj["sub"], n))
    if os.path.exists(measure_mod.LOG):
        for line in open(measure_mod.LOG):
            if key not in line:
                continue
            row = json.loads(line)
            if row.get("kind") == "model_call" and row.get("key") == key \
                    and row.get("sha256") == sha:
                return
    r = json.load(open(req))
    prompt = (r.get("brief") or "") + (r.get("findings") or "")
    err = os.path.join(_pipeline._blind_dir(rnd, subj["sub"], n), "error.md")
    bounces = sum(1 for f in os.listdir(builds)
                  if f.startswith(f"build_{n}.crashed"))
    if bounces and os.path.exists(err):
        prompt += open(err).read()
    measure_mod.model_call(
        "repair", prompt, open(prog).read(), 0.0, key=key, sha256=sha,
        subject=subj["id"], draft=n, bounces=bounces,
        note="wall clock unmeasured: the builder is an out-of-process agent")


def _check_recorded(subj: dict, row: dict, sha: str, n: int) -> None:
    """Part A's acceptance, per draft-0 row: this stage re-measures a published build."""
    import hashlib
    if n != 0 or not os.path.exists(subj["recorded"]):
        return
    want = json.load(open(subj["recorded"])).get(subj["cand"]) or {}
    got = row.get("measures") or {}
    diff = {k: [want.get(k), got.get(k)] for k in want
            if k not in ("seconds", "program") and want.get(k) != got.get(k)}
    src = os.path.join(_pipeline.ROOT, want.get("program") or subj["source"])
    if not os.path.exists(src):
        diff["program_bytes"] = ["recorded program missing", src]
    elif hashlib.sha256(open(src, "rb").read()).hexdigest() != sha:
        diff["program_bytes"] = ["the copy is not the recorded program", src]
    row["reproduces_recorded"] = not diff
    row.pop("recorded_mismatch", None)
    if diff:
        row["recorded_mismatch"] = diff


def stage_revise(rnd: Round, be, results: dict) -> dict:
    """Every subject back through the standard findings loop, capped, offline.

        One candidate, one column of drafts. `build_0.py` is the first draft it already has,
        seeded from the round it was built in and never regenerated; every later draft is a
        builder call staged through the same blinded seam a selection candidate uses. The
        loop is:

            execute the draft -> lint context -> compose the findings -> if there is nothing
            error-level left and no line about getting in on foot, stop; if the cap is
            reached, stop; otherwise stage the next request and report `needs_model`

        A candidate clean at draft 0 gets no call at all and says so. A draft that preflights
        and then raises is bounced once, exactly as a candidate is -- a typo is not a
        revision, and the cap is in the config so nobody can bounce one subject more than
        another.

        Nothing that reaches a builder says there are sixteen of these, that drafts are
        compared, or what is measured; the scratch directory is hash-named for that reason.
        
    """
    import shutil
    cfg = rnd.revise
    if not cfg:
        return {"note": "this round revises nothing"}
    subjects = _revise_subjects(rnd)
    cap = int(cfg.get("max_revisions", 2))
    out = {"adopted": _pipeline._collect_blinded(rnd, [(s["id"], s["sub"]) for s in subjects]),
           "max_revisions": cap, "subjects": len(subjects)}
    rows = {}
    for subj in subjects:
        builds = rnd.rel("arms", subj["sub"], "builds")
        os.makedirs(builds, exist_ok=True)
        seed = os.path.join(builds, "build_0.py")
        if not os.path.exists(seed):
            if not os.path.exists(subj["source"]):
                rows[subj["id"]] = {"status": "no_program",
                                    "program": subj["source"]}
                continue
            shutil.copyfile(subj["source"], seed)
        rec = {"status": None, "drafts": []}
        for n in range(cap + 1):
            prog = os.path.join(builds, f"build_{n}.py")
            if not os.path.exists(prog):
                rec["status"] = "needs_model"
                rec["awaiting_draft"] = n
                rec["blinded"] = _pipeline._blind_build(
                    rnd, subj["sub"],
                    os.path.join(builds, f"build_request_{n}.json"),
                    check=(sorted(subj["mine"]), subj["wave"])
                    if cfg.get("check") else None, role="revise")
                break
            try:
                row = _draft_row(rnd, be, subj, n, prog)
            except Exception as e:              # noqa: BLE001 -- reported, not raised
                import traceback
                rec.update(status="crashed", error=f"{type(e).__name__}: {e}",
                           draft=n, traceback=traceback.format_exc()[-2000:],
                           **_pipeline._bounce(rnd, subj["sub"], builds,
                                     traceback.format_exc()))
                break
            rec["drafts"].append(row)
            print(f"  {subj['id']} draft {n}: walk "
                  f"{row['measures'].get('walk_pct')}%  "
                  f"{row['lint_errors_brief']} own errors  "
                  f"({row['lint_errors_region']} region)  "
                  f"{row['entry_lines']} on-foot lines", flush=True)
            if not row["lint_errors_brief"] and not row["entry_lines"]:
                rec["status"] = "clean"
                rec["clean_at"] = n
                break
            if n == cap:
                rec["status"] = "exhausted"
                break
            # Stage the next request and keep going. Not `break`: the draft that
            # answered this request may already be on disk from an earlier batch, and
            # stopping here would restage a job that has been done and never measure the
            # answer -- the loop would sit at draft 0 for ever. The top of the next
            # iteration is the one place that decides whether a builder is needed.
            json.dump({"i": n + 1, "images": [],
                       "brief": _revise_brief(subj, open(prog).read()),
                       "findings": open(row["findings"]).read()},
                      open(os.path.join(builds, f"build_request_{n + 1}.json"), "w"),
                      indent=1)
        rows[subj["id"]] = rec
    out["candidates"] = rows
    # Part A's free acceptance: draft 0 is the published first draft, re-measured by the
    # identical code. Reported as a count so a drift shows in the round file rather than
    # in a log nobody reads.
    checked = [d for r in rows.values() for d in (r.get("drafts") or [])
               if d["draft"] == 0 and "reproduces_recorded" in d]
    out["reproduces_recorded"] = {
        "checked": len(checked),
        "reproduce": sum(1 for d in checked if d["reproduces_recorded"]),
        "mismatched": {r_id: d.get("recorded_mismatch")
                       for r_id, r in rows.items()
                       for d in (r.get("drafts") or [])
                       if d["draft"] == 0 and not d.get("reproduces_recorded", True)},
    }
    out["readout"] = _pipeline._repair_readout(rnd, subjects, rows)
    return out


def _arms(rnd: Round) -> dict:
    a = rnd.arms
    if not a:
        raise ValueError(f"{rnd.path}: this round has no `arms` block")
    return a


def _arm_list(rnd: Round) -> list:
    return list(_arms(rnd)["list"])


def _arm_sub(rnd: Round, arm: str, i: int) -> str:
    a = _arms(rnd)
    return os.path.join(a.get("out", "sight"), a["wave"], arm, f"c{i}")


def _arm_ids(rnd: Round) -> list:
    """[(id, sub)] over every arm and every candidate in it, in config order."""
    a = _arms(rnd)
    return [(f"{arm['name']}/c{i}", _arm_sub(rnd, arm["name"], i))
            for arm in a["list"] for i in range(int(a["k"]))]


def _arm_dir(rnd: Round) -> str:
    return rnd.rel(_arms(rnd).get("out", "sight"))


def _arm_program(rnd: Round, sub: str) -> str:
    """The candidate's final program: the last cycle that produced one that runs.

        `run_cycles` writes one file per cycle rather than overwriting a single program,
        because the sequence *is* the evidence -- what the loop did on each pass is the
        thing under test, and a single file would keep only the answer.
        
    """
    d = rnd.rel("arms", sub)
    rep = os.path.join(d, "report.json")
    if os.path.exists(rep):
        p = json.load(open(rep)).get("program")
        if p and os.path.exists(p):
            return p
    return os.path.join(d, "no_program.py")


def stage_arms(rnd: Round, be, results: dict) -> dict:
    """Every arm, every candidate, through the fixed cycle loop. The generating half.

        The model seam is the file-backed one, blinded exactly as a selection candidate is:
        the request is copied into a hash-named scratch directory, because a builder handed
        `.../sight/wave1/C/c2/builds/build_3.py` has been told it is candidate 2 of arm C on
        pass 3 of something, and a builder that knows it is in an experiment about seeing
        writes for the experiment.

        Answer the staged requests, re-run, and every cycle already built replays with no
        model call.
        
    """
    from .. import stages
    a = rnd.arms
    if not a:
        return {"note": "this round has no arms"}
    wave = a["wave"]
    brief = open(rnd.rel(a["brief"])).read()
    cycles = a["cycles"]
    vol = be.volume
    ids = _arm_ids(rnd)
    out = {"adopted": _pipeline._collect_blinded(rnd, [(cid, sub) for cid, sub in ids])}
    for arm in _arm_list(rnd):
        for i in range(int(a["k"])):
            cid = f"{arm['name']}/c{i}"
            sub = _arm_sub(rnd, arm["name"], i)
            builds = rnd.rel("arms", sub, "builds")
            try:
                res = stages.run_cycles(
                    brief, rnd.state, cycles=cycles, see=bool(arm["see"]),
                    build=stages.build_from_files(builds), arm=sub, name=wave,
                    vol=vol, allow_try=bool(rnd.flags.get("allow_try")))
            except stages.BuildNeeded as e:
                out[cid] = {"status": "needs_model", "request": e.request_path,
                            "images": e.images or [],
                            "blinded": _pipeline._blind_build(
                                rnd, sub, e.request_path,
                                check=((a.get("plots") or []), wave)
                                if a.get("check") else None)}
                continue
            except Exception as e:            # noqa: BLE001 -- reported, not raised
                import traceback
                out[cid] = {"status": "crashed", "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-2000:],
                            **_pipeline._bounce(rnd, sub, builds,
                                      traceback.format_exc())}
                continue
            rep = res["report"]
            if rep.get("error"):
                out[cid] = {"status": "error", "error": rep["error"]}
                continue
            out[cid] = {"status": "built", "program": res["program"],
                        "pending": rep.get("pending"),
                        "cycles_landed": rep.get("cycles_landed"),
                        "saw": bool(arm["see"]), "lint": rep.get("lint")}
    return out


def part_waves(parts: list) -> list:
    """[(wave_name, [part, ...]), ...] in build order. See above."""
    walls = [p for p in parts if p.get("kind") in ("edge", "point")]
    areas = [p for p in parts if p.get("kind") == "area"]
    quarters: dict = {}
    for p in parts:
        if p.get("kind", "plot") != "plot":
            continue
        quarters.setdefault(p["in"][-1] if p.get("in") else "plots", []).append(p)
    out = []
    if walls:
        out.append(("walls", walls))
    if areas:
        out.append(("squares", areas))
    out += sorted(quarters.items())
    return out


def sample_parts(parts: list, sample: dict) -> tuple:
    """The leaves of a **construction sample**: adjoining quarters and what joins them.

        The architecture round. A place of four hundred leaves cannot be materialised to
        answer "can the plan's promised geometry actually be built", and a single house
        cannot answer it either -- what a sample has to span is the seam: two districts that
        touch, the lanes between them, and the wall, gate or square that stands on their
        boundary. So the rule is written down and run off the plan, not chosen by hand:

          1. the `quarters` plot-bearing quarters with the most plots, taking the first by
             count and then each next one **nearest an already chosen quarter**, so the
             sample is contiguous rather than the largest few scattered over the place;
          2. every non-plot leaf -- an edge, a point, an area -- whose own rectangle comes
             within `margin` of their union. That is the boundary and the way through it.

        `sample` is `{"quarters": n, "margin": m}` off the round's flags. Returns
        `(parts, record)`; the record is what the readout and the report quote.
        
    """
    from .. import pipeline as _pipeline
    n = max(1, int((sample or {}).get("quarters") or 2))
    margin = int((sample or {}).get("margin") or 24)
    by_q: dict = {}
    for p in parts:
        if p.get("kind", "plot") == "plot" and p.get("in"):
            by_q.setdefault(p["in"][-1], []).append(p)
    if not by_q:
        return list(parts), {"quarters": [], "why": "this plan has no plot quarters"}

    def box(rows):
        rs = [_pipeline.part_rect({**p, "name": p.get("name")}) for p in rows]
        return (min(r[0] for r in rs), min(r[1] for r in rs),
                max(r[2] for r in rs), max(r[3] for r in rs))

    boxes = {q: box(rows) for q, rows in by_q.items()}
    order = sorted(by_q, key=lambda q: (-len(by_q[q]), q))
    # **A sample chosen by its architectural questions** (the expression round). The
    # closure city built two lower-ring quarters and every ring wall -- 120 parts that
    # answered no question about form. With `market`, the first quarter is the one that
    # holds a market leaf (working space); with `compound`, the compound's own quarters
    # join it (a courtyard/compound); the quarter between them, nearest both, joins for
    # the transition; with `boundary`, only the walls whose LINE passes within `margin`
    # of a chosen quarter are built, with their gates -- not every ring.
    want_market = bool((sample or {}).get("market"))
    want_compound = bool((sample or {}).get("compound"))
    want_boundary = bool((sample or {}).get("boundary"))
    question = []
    chosen = []
    if want_market:
        with_market = sorted({(p.get("in") or [None])[-1] for p in parts
                              if str(p.get("type") or "") == "market" and p.get("in")
                              and (p.get("in") or [None])[-1] in by_q},
                             key=lambda q: (-len(by_q[q]), q))
        if with_market:
            chosen.append(with_market[0])
            question.append(f"working space: {with_market[0]} holds the market")
    if not chosen:
        chosen = [order[0]]
    if want_compound:
        comp_q = sorted({q for q in by_q if any(p.get("compound") for p in by_q[q])},
                        key=lambda q: (-len(by_q[q]), q))
        if comp_q:
            a = boxes[chosen[0]]
            def gap_to(q):
                b = boxes[q]
                return (max(0, max(b[0] - a[2], a[0] - b[2]))
                        + max(0, max(b[1] - a[3], a[1] - b[3])))
            cq = sorted(comp_q, key=lambda q: (gap_to(q), q))[0]
            chosen.append(cq)
            question.append(f"compound: {cq}")
            # the quarter between the two, nearest both, for the transition
            def near_both(q):
                out_ = 0
                for c in (chosen[0], cq):
                    b, r_ = boxes[c], boxes[q]
                    out_ += (max(0, max(b[0] - r_[2], r_[0] - b[2]))
                             + max(0, max(b[1] - r_[3], r_[1] - b[3])))
                return out_
            between = [q for q in order if q not in chosen
                       and not any(p.get("compound") for p in by_q[q])]
            if between:
                bq = sorted(between, key=lambda q: (near_both(q), -len(by_q[q]), q))[0]
                chosen.append(bq)
                question.append(f"transition: {bq} between them")
    while len(chosen) < n and len(chosen) < len(order):
        def near(q):
            a = boxes[q]
            return min(max(0, max(b[0] - a[2], a[0] - b[2]))
                       + max(0, max(b[1] - a[3], a[1] - b[3]))
                       for b in (boxes[c] for c in chosen))
        rest = [q for q in order if q not in chosen]
        chosen.append(sorted(rest, key=lambda q: (near(q), -len(by_q[q]), q))[0])
    x0 = min(boxes[q][0] for q in chosen) - margin
    z0 = min(boxes[q][1] for q in chosen) - margin
    x1 = max(boxes[q][2] for q in chosen) + margin
    z1 = max(boxes[q][3] for q in chosen) + margin
    grown = [(boxes[q][0] - margin, boxes[q][1] - margin, boxes[q][2] + margin,
              boxes[q][3] + margin) for q in chosen]

    def line_passes(p) -> bool:
        """Does any segment of this edge's line come within margin of a chosen quarter?"""
        for r in _pipeline.part_rects({**p, "name": p.get("name")}):
            for g in grown:
                if r[0] <= g[2] and g[0] <= r[2] and r[1] <= g[3] and g[1] <= r[3]:
                    return True
        return False
    out, joins = [], []
    for p in parts:
        if p.get("kind", "plot") == "plot":
            if (p.get("in") or [None])[-1] in chosen:
                out.append(p)
            continue
        r = _pipeline.part_rect({**p, "name": p.get("name")})
        if not (r[0] <= x1 and x0 <= r[2] and r[1] <= z1 and z0 <= r[3]):
            continue
        if want_boundary and p.get("kind") == "edge" and not line_passes(p):
            continue                    # a ring wall whose line is elsewhere
        if p.get("kind") == "area" and p.get("in") \
                and (p.get("in") or [None])[-1] not in chosen and want_boundary:
            continue                    # another quarter's open ground
        out.append(p)
        joins.append(p["name"])
    # **A wall in the sample brings its gates.** The closure round's transfer sample:
    # the town wall was within the margin and its ring gate was not, so the wall was
    # built through the gate's cell, over the threshold the circulation pass had
    # reserved for it, and the sample's own check refused the sample for a defect the
    # sampling rule had made. A point that stands on an included edge is included.
    edge_rects = [r for p in out if p.get("kind") == "edge"
                  for r in _pipeline.part_rects({**p, "name": p.get("name")})]
    for p in parts:
        if p.get("kind") == "point" and p not in out and p.get("at"):
            ax, az = int(p["at"][0]), int(p["at"][-1])
            if any(r[0] - 2 <= ax <= r[2] + 2 and r[1] - 2 <= az <= r[3] + 2
                   for r in edge_rects):
                out.append(p)
                joins.append(p["name"])
    rec = {"quarters": chosen, "margin": margin,
           "rect": [int(x0), int(z0), int(x1), int(z1)],
           "question": question,
           "plots": sum(len(by_q[q]) for q in chosen),
           "joining_parts": joins,
           "of_plan": {"quarters": len(by_q), "leaves": len(parts)},
           "why": (f"the {len(chosen)} adjoining quarter(s) with the most plots, and "
                   f"every edge, point and area within {margin} columns of their "
                   f"union: the seam between two districts and the way through it")}
    return out, rec


#: The least a clipped wall run may be, in columns, before it is not a wall: below this
#: a boundary reads as a stub and costs a part to say nothing. Registered here before
#: the section was selected.
SECTION_MIN_RUN = 12

#: How much clear wall a gate needs on either side of it inside the section before the
#: passage reads as a passage rather than as a gate standing on its own. A section cut
#: closer than this to a gate is refused, by name, rather than silently made.
SECTION_GATE_CLEAR = 16


def _clip_runs(path: list, rect: tuple, *, half: int = 0) -> list:
    """The parts of an edge's polyline that lie inside `rect`, as separate runs.

        **A section is bounded at a cut, not by paying for the whole circuit.** The design
        round's sampler kept an edge leaf whenever its *bounding* rectangle met the sample
        (`sample_parts`, and `part_rect` of an edge is the bbox of the whole ring), so one
        wall segment touching the sample bought the entire annulus and its gates: 48 parts
        of which the walls, the gates and a compound were most, and the inhabited fabric the
        section was chosen for was two short rows. Clipping the line means the boundary in
        the section is the boundary *of* the section.

        Axis-aligned runs are clipped to the rectangle. A diagonal run -- a ring's chamfer --
        is kept only where it lies wholly inside, because half a staircase is not a wall
        somebody would build. Consecutive pieces that still meet are one run, so a clipped
        corner stays one part rather than two abutting ones.
        
    """
    x0, z0, x1, z1 = (min(rect[0], rect[2]), min(rect[1], rect[3]),
                      max(rect[0], rect[2]), max(rect[1], rect[3]))
    pts = [(int(a[0]), int(a[1])) for a in (path or [])]
    pieces = []
    for a, b in zip(pts, pts[1:]):
        if a == b:
            continue
        inside = (lambda p: x0 <= p[0] <= x1 and z0 <= p[1] <= z1)
        if a[0] != b[0] and a[1] != b[1]:
            if inside(a) and inside(b):
                pieces.append((a, b))
            continue
        if a[0] == b[0]:                                  # runs along z
            if not (x0 <= a[0] <= x1):
                continue
            lo, hi = sorted((a[1], b[1]))
            lo, hi = max(lo, z0), min(hi, z1)
            if hi - lo + 1 < 2:
                continue
            piece = ((a[0], lo), (a[0], hi)) if b[1] >= a[1] else ((a[0], hi), (a[0], lo))
        else:                                             # runs along x
            if not (z0 <= a[1] <= z1):
                continue
            lo, hi = sorted((a[0], b[0]))
            lo, hi = max(lo, x0), min(hi, x1)
            if hi - lo + 1 < 2:
                continue
            piece = ((lo, a[1]), (hi, a[1])) if b[0] >= a[0] else ((hi, a[1]), (lo, a[1]))
        pieces.append(piece)
    runs: list = []
    for a, b in pieces:
        if runs and runs[-1][-1] == a:
            runs[-1].append(b)
        else:
            runs.append([a, b])
    out = []
    for r in runs:
        length = sum(abs(q[0] - p[0]) + abs(q[1] - p[1]) for p, q in zip(r, r[1:])) + 1
        if length >= SECTION_MIN_RUN:
            out.append({"path": [list(p) for p in r], "columns": int(length)})
    return out


#: A district leaf's name is `<district>_<code>` and the code carries the block the
#: compiler laid it in: `b1_0_00` is the first lot of row 0 of block (1, 0), `pc3_0_0`
#: the first court tile of block (3, 0), `x0_1_00` a cross-street lot, `v2_0` a verge.
#: The block is the unit a composition is made in, so it is the unit a section that
#: means to judge compositions selects by.
_BLOCK_CODE = re.compile(r"^(?P<district>.+)_(?P<kind>[a-z]{1,2})(?P<i>\d+)_(?P<j>\d+)"
                         r"[a-z]?(?:_.*)?$")


def _block_key(name: str | None):
    """`(district, i, j)` for a leaf the compiler laid in a block, else None."""
    m = _BLOCK_CODE.match(str(name or ""))
    if not m:
        return None
    return (m.group("district"), int(m.group("i")), int(m.group("j")))


def section_parts(parts: list, section: dict) -> tuple:
    """The leaves of a **registered section**: one connected piece of a larger place.

        The composition round. `sample_parts` above chooses its own extent from the plan --
        the quarter with a market, the nearest compound, the quarter between them -- which
        is a rule for finding *an* interesting seam and not a rule for answering a stated
        architectural question. This takes the extent as given, registered before any
        candidate was compiled, and selects against it:

          - a **plot or area leaf is taken whole or not at all**, so a section cut falls
            between buildings and never through one. A leaf that straddles the boundary is
            counted and named in `cut_out`, not half-built;
          - an **edge is clipped** to the section (`_clip_runs`), so a boundary costs the
            section only the length of boundary the section actually contains;
          - a **point stands on an included run**, with its whole pad inside, and a cut
            closer than `SECTION_GATE_CLEAR` to a gate **refuses the section by name**
            rather than cutting through the passage and then excusing the failure as
            sampling.

        `section` is the round file's `flags.section`:
        `{"id", "rect": [x0,z0,x1,z1], "boundary", "gate", "sides": {label: district_prefix},
          "demonstrate": [...]}`. Returns `(parts, record)`; the record goes into
        `parts.json` under `sample`, because every downstream reader of a partial build --
        `intent.sample_scope`, `placeread`'s `limits.sample`, the lint's region -- already
        keys on that name, and a section is exactly a partial build: **its clauses qualify
        its own scope and nothing outside it.**
        
    """
    from .. import pipeline as _pipeline
    rect = [int(v) for v in (section or {}).get("rect") or []]
    if len(rect) != 4:
        return list(parts), {"refused": "the section registers no rect", "quarters": []}
    x0, z0, x1, z1 = (min(rect[0], rect[2]), min(rect[1], rect[3]),
                      max(rect[0], rect[2]), max(rect[1], rect[3]))
    sid = str((section or {}).get("id") or "section")

    def whole_in(r) -> bool:
        return r[0] >= x0 and r[1] >= z0 and r[2] <= x1 and r[3] <= z1

    def meets(r) -> bool:
        return r[0] <= x1 and x0 <= r[2] and r[1] <= z1 and z0 <= r[3]

    # **A section may select whole blocks rather than a rectangle's worth of leaves.**
    # The block design round. `whole_in` takes a plot or an area whole or not at all, so
    # a cut falls between buildings and never through one -- and it still falls through
    # the *compositions* those buildings make. A court block whose front range is inside
    # the rectangle and whose back range is two columns outside it builds three ranges
    # of four, and the court in the middle of them is then read, photographed and
    # counted as open ground. That is exactly the "arbitrary crop through a court" the
    # round refuses: "Choose scope for the relationship being designed; an arbitrary
    # crop through a court or route is not a complete unit." With `unit: "block"` the
    # selection is closed under the compiler's own block: a leaf inside the rectangle
    # brings in every other leaf of its block, wherever that block reaches. The extent
    # grows to whole blocks and the record says by how much.
    unit = str((section or {}).get("unit") or "rect")
    block_of = _block_key if unit == "block" else (lambda _n: None)
    here_blocks = set()
    if unit == "block":
        for p in parts:
            if p.get("kind", "plot") not in ("plot", "area"):
                continue
            if whole_in(_pipeline.part_rect({**p, "name": p.get("name")})):
                k = block_of(p.get("name"))
                if k:
                    here_blocks.add(k)

    out, joins, cut_out, runs_rec = [], [], [], []
    grown: list = []
    quarters: dict = {}
    for p in parts:
        kind = p.get("kind", "plot")
        if kind in ("plot", "area"):
            r = _pipeline.part_rect({**p, "name": p.get("name")})
            in_block = bool(here_blocks) and block_of(p.get("name")) in here_blocks
            if whole_in(r) or in_block:
                if in_block and not whole_in(r):
                    grown.append({"part": p["name"], "kind": kind,
                                  "rect": [int(v) for v in r],
                                  "why": "a leaf of a block this section selects, "
                                         "outside the registered rectangle: a block is "
                                         "taken whole or not at all"})
                out.append(p)
                q = (p.get("in") or [None])[-1]
                if kind == "plot" and q:
                    quarters.setdefault(q, []).append(p["name"])
                elif kind == "area":
                    joins.append(p["name"])
            elif meets(r):
                cut_out.append({"part": p["name"], "type": p.get("type"),
                                "kind": kind, "rect": [int(v) for v in r]})
            continue
        if kind == "edge":
            got = _clip_runs(p.get("path") or [],
                             (x0, z0, x1, z1),
                             half=max(1, int(p.get("width", 1))) // 2)
            for i, run in enumerate(got):
                name = f"{p['name']}@{sid}" + (f"_{i}" if len(got) > 1 else "")
                out.append({**p, "name": name, "path": run["path"],
                            "section_clip": {"of": p["name"], "columns": run["columns"],
                                             "in": sid}})
                joins.append(name)
                runs_rec.append({"of": p["name"], "part": name,
                                 "columns": run["columns"], "path": run["path"]})
            continue
    # a point -- a gate, a well -- stands on an included run, whole pad inside.
    edge_cells = {}
    for p in out:
        if p.get("kind") == "edge":
            from ..placeplan import _edge_cells
            edge_cells[p["name"]] = set(_edge_cells(p))
    all_cells = set().union(*edge_cells.values()) if edge_cells else set()
    refused = []
    for p in parts:
        if p.get("kind") != "point" or not p.get("at"):
            continue
        r = _pipeline.part_rect({**p, "name": p.get("name")})
        ax, az = int(p["at"][0]), int(p["at"][-1])
        on_run = any((ax + dx, az + dz) in all_cells
                     for dx in range(-2, 3) for dz in range(-2, 3))
        if whole_in(r) and (on_run or not all_cells):
            out.append(p)
            joins.append(p["name"])
            # **A cut may not fall on a gate.** The round: do not cut through a gate or
            # a structure, break access, and then excuse the failure as sampling.
            for run in runs_rec:
                for end in (run["path"][0], run["path"][-1]):
                    d = abs(end[0] - ax) + abs(end[1] - az)
                    if d < SECTION_GATE_CLEAR:
                        refused.append(
                            f"the section's cut of {run['of']} ends {d} column(s) from "
                            f"{p['name']}, inside the {SECTION_GATE_CLEAR} this build "
                            f"calls clear: move the section boundary, do not cut a gate")
        elif meets(r) and (on_run or not all_cells):
            cut_out.append({"part": p["name"], "type": p.get("type"), "kind": "point",
                            "rect": [int(v) for v in r]})
    sides = dict((section or {}).get("sides") or {})
    by_side = {k: 0 for k in sides}
    for q, names in quarters.items():
        for label, prefix in sides.items():
            if str(q).startswith(str(prefix)):
                by_side[label] += len(names)
    types: dict = {}
    for p in out:
        types[str(p.get("type") or p.get("kind"))] = \
            types.get(str(p.get("type") or p.get("kind")), 0) + 1
    rec = {"section": sid, "quarters": sorted(quarters), "margin": 0,
           "rect": [x0, z0, x1, z1],
           "registered": {k: v for k, v in (section or {}).items() if k != "rect"},
           "question": list((section or {}).get("demonstrate") or []),
           "plots": sum(len(v) for v in quarters.values()),
           "included_plots": sorted(n for v in quarters.values() for n in v),
           "by_side": by_side, "types": types,
           "joining_parts": joins,
           "boundary_runs": runs_rec,
           "cut_out": cut_out,
           # the block design round: what the rectangle would have cut through and the
           # block selection kept whole, so "the extent is bigger than the registered
           # rectangle" is a thing on the record and not a surprise in an image
           "unit": unit,
           "grown_to_whole_blocks": grown,
           "blocks": sorted(f"{d}/{i},{j}" for (d, i, j) in here_blocks),
           "of_plan": {"quarters": len({(p.get("in") or [None])[-1] for p in parts
                                       if p.get("kind", "plot") == "plot"
                                       and p.get("in")}),
                       "leaves": len(parts)},
           "why": (f"the registered section {sid} {[x0, z0, x1, z1]}: every plot and "
                   f"area whole inside it, every boundary clipped to it, every gate on "
                   f"an included run -- a bounded connected extent, not a quarter "
                   f"count")}
    if refused:
        rec["refused"] = refused[0]
        rec["refusals"] = refused
    return out, rec


def annotate_gates(parts: list) -> list:
    """Which point stands on which edge, and what that means for both. Demo-polish, 2b.

        A point whose anchor is on an edge's swept line gets `part["edge"]` -- the edge's
        name, type, declared `height` and `width` -- and, where it names no `size` of its
        own, a pad sized from that height by `Builder.point_pad`, inside its type's band.
        The edge gets `part["gates"]`: the points on it, with their anchors and pads, so a
        wall type leaves the opening to the gate standing in it rather than cutting one of
        its own somewhere else. Neither the planner nor the type is told a height by hand;
        this is the layer that knows both. Returns the names annotated.
        
    """
    from ..buildlib import Builder
    from ..placeplan import _edge_cells
    edges = [p for p in parts if p.get("kind") == "edge" and p.get("path")]
    cells = {e["name"]: set(_edge_cells(e)) for e in edges}
    done = []
    for p in parts:
        if p.get("kind") != "point" or not p.get("at"):
            continue
        at = (int(p["at"][0]), int(p["at"][-1]))
        for e in edges:
            if at not in cells[e["name"]]:
                continue
            h = (e.get("params") or {}).get("height")
            band = None
            tf = os.path.join(_pipeline.ROOT, "types", f"{p.get('type')}.py")
            if p.get("type") and os.path.exists(tf):
                try:
                    band = _pipeline.load_type(tf)["needs"].get("footprint")
                except Exception:            # noqa: BLE001 -- a bad type is refused later
                    band = None
            p["edge"] = {"name": e["name"], "type": e.get("type"),
                         "height": (int(h) if h is not None else None),
                         "width": int(e.get("width", 1) or 1)}
            if h is not None and not p.get("size"):
                p["size"] = Builder.point_pad(int(h), band)
            e.setdefault("gates", []).append(
                {"name": p["name"], "at": list(at),
                 "size": int(p.get("size") or Builder.SITE_POINT)})
            done.append(p["name"])
            break
    return done


#: and the blocks one part lays, 1,762,545 over 315, 5,600 a part. The parts preflight
#: multiplies these by the plan's leaves, divides the time by the workers, prints both
#: before the first part, and refuses over `PARTS_BOUND_S`: six hours.
COST_PART_S = 31.0
COST_PART_BLOCKS = 5600
PARTS_BOUND_S = 6 * 3600.0


def preflight_parts(parts: list, voices: list, workers: int = 1,
                    bound: float = PARTS_BOUND_S, log=print) -> dict:
    """What the parts stage is about to do, before it does any of it.

        Every voice's six roles are resolved to a block in every shape the shell lays
        them in **before the first part**: the concentric run lost 73 minutes to a voice
        whose floor was no family, wave after wave, and a refusal about a JSON file is
        cheaper than a crash on the thirtieth part. Then the estimate: the leaves by kind,
        the blocks and the seconds they will cost by the registered per-part numbers,
        the seconds divided by the workers. Refused over `bound`, by name.
        
    """
    from .. import pipeline, voices as _voices
    from ..prims import shape, solid
    failures = []
    resolved = {}
    for v in voices:
        pal = pipeline.voice_palette(v)
        if not pal:
            failures.append(f"voice {v!r}: no palette on disk")
            continue
        roles = {}
        for role in _voices.SHAPED:
            for kind in ("full", "stairs", "slab"):
                try:
                    roles[f"{role}/{kind}"] = shape(pal[role], kind)
                except Exception as e:           # noqa: BLE001 -- named, refused
                    failures.append(f"voice {v!r}: {role} {pal[role]!r} has no {kind}: {e}")
        for role in _voices.SOLID:
            try:
                roles[f"{role}/full"] = solid(pal[role])
            except Exception as e:               # noqa: BLE001 -- named, refused
                failures.append(f"voice {v!r}: {role} {pal[role]!r}: {e}")
        resolved[v] = roles
    kinds: dict = {}
    for q in parts:
        k = q.get("kind", "plot")
        kinds[k] = kinds.get(k, 0) + 1
    n = len(parts)
    est_s = n * COST_PART_S / max(1, int(workers))
    out = {"parts": n, "by_kind": kinds, "voices": sorted(voices),
           "estimated_blocks": int(n * COST_PART_BLOCKS),
           "estimated_seconds": round(est_s), "workers": int(workers),
           "bound_seconds": float(bound), "failures": failures,
           "refused": bool(failures) or est_s > bound,
           "registered": {"COST_PART_S": COST_PART_S, "COST_PART_BLOCKS": COST_PART_BLOCKS,
                          "PARTS_BOUND_S": PARTS_BOUND_S}}
    log(f"   preflight: {n} parts {kinds} in {len(voices)} voice(s), about "
        f"{out['estimated_blocks']:,} blocks and {est_s / 60:.0f} min on {workers} "
        f"worker(s) against a bound of {bound / 60:.0f}"
        + (f" -- REFUSED: {'; '.join(failures[:3])}" if failures else
           " -- REFUSED: over the bound" if out["refused"] else ""))
    return out


def settle_ground(rnd, be, parts: list, mat_of) -> tuple:
    """The ground contract of a build, settled once before the first part. v2, B1.

        Every leaf declares what it needs of the ground -- its pad and level, an edge's
        footing along its run -- read off the volume **as it stands before any part is
        built**, and the network declares its lanes and doorsteps; `ground.Contract`
        settles every column by precedence, holds each platform within reach of its own
        ground and above any water under it, and derives the seams. What comes back is
        read-only: `site()` lays what was settled for its part, and the surface heightmap
        the whole build reads is the one fixed here. Saved beside the round's state as
        `ground.npz` for a wave's worker and `ground.json` for a reader.

        Returns `(resolved, record)`; `mat_of(part)` is the palette a part is composed in.
        
    """
    from .. import ground as _ground, pipeline, stages
    from ..buildlib import Builder
    from ..frontage import Frontage
    t0 = time.perf_counter()
    vol = be.volume
    net = rnd.network()
    found = _ground.Found(vol)
    b = Builder(offline.OfflineSite(vol, heights=found.surface.h))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net else None
    b.registry = stages._Registry(rnd.state)
    contract = _ground.Contract()
    decided = {}
    refused = 0
    for part in parts:
        geo = {k: part[k] for k in pipeline.PART_GEOMETRY if k in part}
        geo["label"] = part["name"]
        geo["kind"] = part.get("kind", "plot")
        geo["type"] = part.get("type")
        try:
            dec = b.declare(geo, mat=mat_of(part), contract=contract)
        except Exception as e:                       # noqa: BLE001 -- reported, not raised
            dec = {"ok": False, "kind": geo["kind"],
                   "reason": f"{type(e).__name__}: {e}"}
        refused += not dec.get("ok")
        decided[part["name"]] = {
            "ok": bool(dec.get("ok")), "kind": dec.get("kind"),
            "ground": dec.get("ground"), "level": dec.get("level"),
            "rect": dec.get("rect") or ([dec.get("x0"), dec.get("z0"), dec.get("x1"),
                                         dec.get("z1")] if dec.get("ok") else None),
            "reason": dec.get("reason")}
    contract.network(net)
    resolved = contract.resolve(found.bed, found.wet, relief=Builder.SITE_RELIEF,
                                surface=found.surface)
    resolved.save(rnd.rel("ground.npz"))
    rec = resolved.record()
    rec.update(parts=decided, declared=len(parts) - refused, refused=refused,
               network={"cells": len(net.cells) if net else 0,
                        "thresholds": len(net.thresholds) if net else 0},
               seconds=round(time.perf_counter() - t0, 2),
               saved=os.path.relpath(rnd.rel("ground.npz"), _pipeline.ROOT))
    json.dump(rec, open(rnd.rel("ground.json"), "w"), indent=1)
    return resolved, rec


def instantiate_part(rnd, be, part: dict, mat, roof=None, paths_sink=None,
                     ground=None) -> dict:
    """One leaf, from the type it names. Returns what happened; never raises.

        The whole of Part C's claim is in this function being the only way a part gets
        built. A type is a committed file, a part is a leaf of the plan, and this composes
        the one with the other and executes it. There is no branch here for "and if that
        does not work, write a program": a part that cannot be instantiated is reported.
        
    """
    from .. import pipeline, stages
    from ..lint import preflight
    name = part["name"]
    tf = os.path.join(_pipeline.ROOT, "types", f"{part.get('type')}.py")
    if not os.path.exists(tf):
        return {"part": name, "status": "no_type",
                "error": f"the plan names a type called {part.get('type')!r} and "
                         f"types/{part.get('type')}.py is not on disk"}
    t0 = time.perf_counter()
    try:
        decl = pipeline.load_type(tf)
        params = pipeline.check_params(decl["params"], part.get("params"),
                                       where=os.path.basename(tf))
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        return {"part": name, "status": "bad_type", "error": f"{type(e).__name__}: {e}"}
    pre = preflight(decl["src"], forbid=pipeline.TYPE_FORBIDDEN, palette=True)
    if not pre.ok:
        return {"part": name, "status": "rejected", "type": part.get("type"),
                "preflight": pre.to_json()}
    # cut the pads, laid the plinths, laid the ways in -- and then heard "no massing
    # stood on this pad" from a type that could have said so from the geometry alone. A
    # part whose pad is outside what its type declares is refused here, with the type,
    # the pad and the need named, and the world is not touched.
    why = pipeline.needs_footprint_failure(part, decl["needs"])
    if why:
        from ..buildlib import Builder
        return {"part": name, "status": "refused", "type": part.get("type"),
                "kind": part.get("kind", "plot"),
                "needs": list(decl["needs"]["footprint"]),
                "pad": list(Builder.pad_extent(part)),
                "error": f"{part.get('type')} declares NEEDS and this part does not "
                         f"meet them: {why}"}
    geo = {k: part[k] for k in pipeline.PART_GEOMETRY if k in part}
    geo["label"] = name
    geo["kind"] = part.get("kind", "plot")
    src = pipeline.instantiated_source(decl["src"], [(geo, int(part.get("seed", 0)),
                                                      params)], mat=mat, roof=roof)
    prog = rnd.rel("parts", f"{name}.py")
    os.makedirs(os.path.dirname(prog), exist_ok=True)
    open(prog, "w").write(src)
    try:
        from ..buildlib import max_blocks_for
        site = _pipeline.settlement_site(rnd) or rnd.site or {}
        b = offline.run_program(prog, be.volume, network=rnd.network(),
                                plots=stages._Registry(rnd.state),
                                allow_collide=bool(rnd.flags.get("allow_collide")),
                                src=src, max_blocks=max_blocks_for(site.get("size")),
                                ground=ground)
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        import traceback
        if type(e).__name__ == "SiteRefused":
            # the compiled site could not be built as compiled: a visible refusal, not a
            # crash and not a sunk house (`buildlib.SiteRefused`)
            return {"part": name, "status": "refused", "type": part.get("type"),
                    "kind": part.get("kind", "plot"), "site_refused": True,
                    "error": f"SiteRefused: {e}"}
        return {"part": name, "status": "crashed", "type": part.get("type"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": pipeline.scrub_traceback(traceback.format_exc())[-1200:]}
    # **What construction actually delivered, measured before the blocks are
    # committed.** The closure round's construction boundary: the height clause read the
    # planned `params.storeys` and called a one-storey cottage tall. The outcome is read
    # off the builder's own pending blocks, the surface context beside it, and a
    # constraint where a requested feature was lost -- so the checker measures the
    # building that stands and the recovery ladder has a lot size to act on.
    emitted = surfaces = limit = surface_record = None
    try:
        from .. import construction
        sited_now = dict(b.parts[-1]) if getattr(b, "parts", None) else {}
        leaf = {**part, "params": params, **({"footprint": sited_now["footprint"]}
                                             if sited_now.get("footprint") else {})}
        emitted = construction.outcome(b, {**leaf, **sited_now, "name": name,
                                           "kind": geo["kind"]}, decl, params)
        surfaces = construction.surfaces(b, {**leaf, **sited_now, "name": name})
        # **The owned surfaces, recorded at emission.** The expression round (worker C):
        # who laid every block and as what role, with its exposure and what is protected
        # -- the record a material pass may edit and nothing else may.
        from .. import surfaces as surfaces_mod
        surface_record = surfaces_mod.record(
            b, {**leaf, **sited_now, "name": name, "kind": geo["kind"],
                "type": part.get("type")},
            voice_name=part.get("voice") or rnd.voice_name() or None)
        limit = construction.constraint(leaf, decl, emitted, seed=int(part.get("seed", 0)),
                                        voice=part.get("voice") or rnd.voice_name() or None)
    except Exception as e:                       # noqa: BLE001 -- reported, never fatal
        emitted = {"measured": False, "why": f"the outcome could not be measured: "
                                             f"{type(e).__name__}: {e}"}
    placed = be.commit(b)
    # What this part laid as a way in, beside the plot it was laid for. The finishing
    # pass reads it and keeps off those columns. Written here because `stage_parts` is
    # the pass now, and `settlement.add_paths` bakes its directory in from
    # $ETHOSLM_SETTLEMENT at import. `stage_parts` hands a `paths_sink` -- its own record,
    # which it writes -- and so does a wave's worker; the append below is for a caller
    # with neither, and it is the shape that let a stage run twice record every way in
    # twice (v2, B0).
    if b.paths:
        if paths_sink is not None:
            paths_sink.extend(dict(r) for r in b.paths)
        else:
            pp = rnd.rel("paths.json")
            rows = json.load(open(pp)) if os.path.exists(pp) else []
            json.dump(rows + [dict(r) for r in b.paths], open(pp, "w"), indent=1)
    sited = dict(b.parts[-1]) if getattr(b, "parts", None) else {}
    return {"part": name, "status": "built", "type": part.get("type"),
            # transient: `stage_parts` strips it from the row into surfaces.json
            "surface_record": surface_record,
            # 2b: the pad the library sited a point on, for the registry.
            "footprint": (list(sited.get("footprint")) if sited.get("footprint")
                          and part.get("kind") == "point" else None),
            # What the type itself said. `status` is about the program running; this is
            # about the building standing.
            "stood": bool((sited.get("build") or {"ok": True}).get("ok", True)),
            "type_said": (sited.get("build") or {}).get("reason"),
            "kind": geo["kind"], "seed": int(part.get("seed", 0)), "params": params,
            "program": os.path.relpath(prog, _pipeline.ROOT),
            "blocks": len(b._pending), "writes": b.writes,
            "placed": placed.get("placed"), "failed": placed.get("failed"),
            "ground": sited.get("ground"), "floor_y": sited.get("floor_y"),
            "sited": (sited.get("sited") or {}).get("reason"),
            # **The way in that was actually laid.** The neighbourhood delivery round:
            # `site()` chooses the door cell on the pad it prepared and `approach()`
            # answers, against the world it has just changed, whether a person can walk
            # to it. Both were inside the sited dict and neither reached the record, so
            # nothing downstream could tell a structure entered somewhere other than its
            # reserved doorstep from one that cannot be entered at all.
            "door": (list(sited["door"]) if sited.get("door") else None),
            "way_in": {"ok": bool(((sited.get("sited") or {}).get("approach") or {})
                                  .get("ok")),
                       "why": str((((sited.get("sited") or {}).get("approach") or {})
                                   .get("reason")) or "")[:200]},
            # the calls in which the voice's footing stood in for a bare floor block
            # (`TypeBuilder._shapeable`), or nothing
            **({"voice_stood_in": sited["voice_stood_in"]}
               if sited.get("voice_stood_in") else {}),
            "refusals": getattr(b, "_type_refusals", None),
            "emitted": emitted, "surfaces": surfaces,
            **({"constraint": limit} if limit else {}),
            "seconds": round(time.perf_counter() - t0, 1)}


def floors_into(plots: list, rows: list) -> list:
    """`plots` with each built part's sited floor and pad written in, unwritten."""
    floors = {r["part"]: r.get("floor_y") for r in rows
              if r.get("status") == "built" and r.get("floor_y") is not None}
    pads = {r["part"]: r["footprint"] for r in rows
            if r.get("status") == "built" and r.get("footprint")}
    out = []
    for p in plots:
        q = dict(p)
        if q["label"] in floors:
            q["y0"] = int(floors[q["label"]])
        if q["label"] in pads and q.get("kind") == "point":
            x0, z0, x1, z1 = pads[q["label"]]
            q.update(x0=int(x0), z0=int(z0), x1=int(x1), z1=int(z1))
        out.append(q)
    return out


class _WaveBackend:
    """A worker's own dry-run backend: one volume, commits as overlays, and the
    blocks every part of the wave laid, in order, for the driver to merge."""
    live = False
    dry_run = True

    def __init__(self, vol):
        self._vol = vol
        self.blocks: dict = {}

    @property
    def volume(self):
        return self._vol

    def commit(self, builder):
        pending = dict(getattr(builder, "_pending", {}) or {})
        if pending:
            self._vol = self._vol.overlay(pending)
            self.blocks.update(pending)
        return {"placed": len(pending), "failed": 0,
                "note": "a wave's worker: written into its own volume"}


def _wave_worker(args: tuple) -> dict:
    """One quarter wave, in its own process, off the volume as the walls and the
    squares left it: every part instantiated in order against the worker's own copy,
    the wave's own lint read on it, and the blocks, the rows, the paths and the lint
    handed back for the driver to merge in wave order. **The unit `par_map`
    distributes.**"""
    from .. import ground as _ground, lint, pipeline
    (rnd, wave, group, snapshot, palettes, plots, base_path, scope_margin, default,
     ground_path) = args
    vol = offline.load_volume(snapshot)
    be = _WaveBackend(vol)
    base = offline.load_volume(base_path) if base_path and os.path.exists(base_path) else None
    # the build's settled ground, read back in this process (v2, B1)
    ground = (_ground.load(ground_path) if ground_path and os.path.exists(ground_path)
              else None)
    rows, paths = [], []
    t0 = time.perf_counter()
    for part in group:
        v = part.get("voice") or default
        mat, roof = palettes[v]
        rec = instantiate_part(rnd, be, part, mat, roof, paths_sink=paths, ground=ground)
        rec["voice"] = v
        rows.append(rec)
    plots2 = floors_into(plots, rows)
    by = {p["label"]: p for p in plots2}
    rects = [by[r["part"]] for r in rows if r["part"] in by]
    scope = None
    if rects:
        scope = (min(p["x0"] for p in rects) - scope_margin,
                 min(p["z0"] for p in rects) - scope_margin,
                 max(p["x1"] for p in rects) + scope_margin,
                 max(p["z1"] for p in rects) + scope_margin)
    ctx = lint.Context.build(be.volume, plots2, network=rnd.network(),
                             region=scope, base=base)
    rep = lint.lint(ctx, family=lint.BUILD).within(rects or plots2)
    return {"wave": wave, "rows": rows, "paths": paths, "blocks": be.blocks,
            "scope": list(scope) if scope else None,
            "errors": sorted(f.code for f in rep.findings if f.code.startswith("E")),
            "counts": rep.to_json()["counts"],
            "seconds": round(time.perf_counter() - t0, 1)}


def record_part_floors(rnd, rows: list, plots: list) -> list:
    """Write the floor level each built part was sited at into `plots.json`. A1.

        The registry is where the passes agree about ground, and this is one more thing
        they have to agree about: `lint.Context.room_owner` reads `y0` to tell the inside
        of a building from the cave under its plot, and the answer is the library's --
        `site()` sounds the bed and returns `floor_y`, and preflight's E013 refuses a type
        that writes below it.

        Re-read from disk rather than written from the caller's copy, because `reserve()`
        may have added rows while the wave ran. Returns the registry as it now stands.
        
    """
    path = rnd.rel("plots.json")
    if not os.path.exists(path):
        return plots
    on_disk = json.load(open(path))
    floors = {r["part"]: r.get("floor_y") for r in rows
              if r.get("status") == "built" and r.get("floor_y") is not None}
    # 2b: and the pad a point was sited on, which is the library's answer once the
    # point's wall has sized it -- the registry row was written at plan time from the
    # type's minimum, and the lint's "own plot" for a gate has to be the gate.
    pads = {r["part"]: r["footprint"] for r in rows
            if r.get("status") == "built" and r.get("footprint")}
    if not floors and not pads:
        return on_disk
    for p in on_disk:
        if p["label"] in floors:
            p["y0"] = int(floors[p["label"]])
        if p["label"] in pads and p.get("kind") == "point":
            x0, z0, x1, z1 = pads[p["label"]]
            p.update(x0=int(x0), z0=int(z0), x1=int(x1), z1=int(z1))
    json.dump(on_disk, open(path, "w"), indent=1)
    return on_disk


def stage_parts(rnd, be, results: dict) -> dict:
    """Every leaf of the plan, instantiated, in waves. A6 lints each wave's own ground.

        No builder call and no hand-written program: that is the bar, and it is a bar about
        this function's own source as much as about the run.
        
    """
    from .. import lint, pipeline, stages
    from . import stages_measure
    from .stages_plan import ground_for_this_design
    parts = rnd.parts()
    if not parts:
        return {"error": "no plan.json, or a plan with no leaves in it"}
    # **Built once per candidate.** The closure round: this stage rebuilt every part on
    # every invocation, so a replay that changed nothing re-laid the world, moved the
    # built volume's identity under the inspection bound to it, and asked the judge to
    # read the same place again. A built world is an artifact of the plan it was built
    # from, stamped like any other, and reused while nothing it was made from has moved.
    from .. import deps as _deps_w
    # **The ground this is about to be built on was prepared for this design.** The
    # review's sixth finding: preparation and planning were separate decisions and
    # nothing connected them, so a repaired plan could be built on the terraces cut for
    # the plan it replaced. `terrain` is the baseline and `ground` is the preparation,
    # stamped against the plan it was cut for; this is where the two meet. **Asked
    # before the warm return, not after it** (the composition round). This check stood
    # eight lines below the warm return, so a build could be handed back as fresh
    # without the question ever being put -- and `DEPENDS["built"]` did not carry ground
    # identity either, so the freshness check could not notice the cut had moved. Both
    # halves had to go: the dependency is real now (`deps`, evidence stream) and the
    # order is this way round. A warm build is a claim about the world *and the ground
    # under it*; making that claim without asking is how a stale certificate survives.
    ready, why = ground_for_this_design(rnd)
    if not ready:
        return {"status": "error", "stop": True, "ground": why,
                "error": (f"the prepared ground under this plan was cut for a different "
                          f"candidate: {why}. Run the ground stages again before "
                          f"building on it")}
    fresh, why = _deps_w.check(rnd, "built", plan=rnd.plan())
    if fresh and os.path.exists(rnd.rel("parts.json")) \
            and os.path.exists(rnd.rel("world_built.npz")):
        was = json.load(open(rnd.rel("parts.json")))
        # the built volume is what every later stage reads; put it in the backend
        with contextlib.suppress(Exception):
            be._vol = offline.load_volume(rnd.rel("world_built.npz"))
        return {**{k: was.get(k) for k in ("built", "failed", "sample", "candidate",
                                           "voice", "voices")},
                "skipped": why, "written": rnd.rel("parts.json"), "waves": was.get("waves")}
    # **A round may build a sample of its plan rather than all of it.** The architecture
    # round: the large case stops at planning and preview, and what answers "can this
    # plan's promised geometry be built" is a compact sample spanning two adjoining
    # districts and the boundary between them. The rule is in `sample_parts` and the
    # round file names its two numbers; a round with no `sample` flag builds every leaf,
    # exactly as every round before it did. ...and **a round may register the section it
    # means** (the composition round): a bounded connected extent, chosen for the
    # architectural questions it has to answer and written down before any candidate was
    # compiled. `section_parts` selects against that registration; `sample_parts` above
    # chooses its own.
    sample_rec = None
    if rnd.flags.get("section"):
        parts, sample_rec = section_parts(parts, rnd.flags["section"])
        if sample_rec.get("refused"):
            return {"status": "error", "stop": True, "sample": sample_rec,
                    "error": f"the registered section refuses its own selection: "
                             f"{sample_rec['refused']}"}
        print(f"   section {sample_rec['section']} {sample_rec['rect']}: "
              f"{sample_rec['plots']} plot(s) in {len(sample_rec['quarters'])} "
              f"quarter(s), {len(sample_rec['boundary_runs'])} clipped boundary run(s), "
              f"{len(sample_rec['joining_parts'])} joining part(s), "
              f"{len(sample_rec['cut_out'])} leaf/leaves cut out, of "
              f"{sample_rec['of_plan']['leaves']} leaves in the plan"
              + (f"; by side {sample_rec['by_side']}" if sample_rec.get("by_side")
                 else ""), flush=True)
    elif rnd.flags.get("sample"):
        parts, sample_rec = sample_parts(parts, rnd.flags["sample"])
        print(f"   sample: {sample_rec['plots']} plot(s) in "
              f"{len(sample_rec['quarters'])} adjoining quarter(s) "
              f"({', '.join(sample_rec['quarters'])}) and "
              f"{len(sample_rec['joining_parts'])} joining part(s), of "
              f"{sample_rec['of_plan']['leaves']} leaves in the plan", flush=True)
    # **The voice the place is in, not the config's field.** Voice contract, A1.
    # `Round.voice` is the *config's* voice, and a round that carries a sentence leaves
    # it empty on purpose -- "a coordinate in the config is a human having chosen the
    # ground", and so is a palette. `voice_name()` is the one line that answers the
    # question properly: the config's voice, or the one the place chose, or the one the
    # spec wrote. the library's default palette. Nothing measured a block against the
    # voice, so nothing said so; A2's `palette/built` clause is that measurement now,
    # and this line is what it holds a build to.
    voice = rnd.voice_name() or None
    # a leaf with none is the place's, exactly as every leaf was before. One resolution
    # per voice name, so a city of four rings resolves four times, not 280.
    palettes: dict = {}
    form = (rnd.place_spec() or {}).get("form")

    def voice_of(part):
        v = part.get("voice") or voice
        if v not in palettes:
            roof = pipeline.voice_roof(v)
            # None in the east Asian tradition, one where the type asks in the European
            # one.
            if roof is not None and roof.get("chimney") is None:
                from .. import voices as _voices
                roof = dict(roof, chimney=_voices.chimney_default(form))
            palettes[v] = (pipeline.voice_palette(v), roof)
        return v, palettes[v]
    plots = json.load(open(rnd.rel("plots.json"))) \
        if os.path.exists(rnd.rel("plots.json")) else []
    by = {p["label"]: p for p in plots}
    # See `lint.Context.build`.
    base = stages_measure._prebuild(rnd)
    # 2b: a gate is sized by its wall, and a wall knows its gates. Decided here, once,
    # before any of them is sited; the edge's sited floor reaches its gates as each edge
    # is built, since the walls wave builds an edge before the points on it.
    out: dict = {"waves": [], "built": 0, "failed": 0,
                 **({"sample": sample_rec} if sample_rec else {}),
                 "gates_on_edges": annotate_gates(parts),
                 "voice": voice,
                 "voices": sorted({p.get("voice") or voice for p in parts
                                   if (p.get("voice") or voice)})}
    # Every voice resolved in every shape, the cost estimated and printed, refused by
    # name over the bound.
    from ..parallel import workers as _workers
    dry = not be.live and getattr(be, "dry_run", False)
    n_workers = _workers(1 if rnd.flags.get("workers") is None
                         else int(rnd.flags["workers"])) if dry else 1
    if os.environ.get("ETHOSLM_WORKERS") and dry:
        n_workers = _workers()
    pre = preflight_parts(parts, out["voices"], workers=n_workers,
                          bound=float(rnd.flags.get("parts_bound_seconds")
                                      or PARTS_BOUND_S))
    out["preflight"] = pre
    if pre["refused"]:
        out.update(status="error", stop=True,
                   error="the parts preflight refused: "
                         + ("; ".join(pre["failures"][:4]) if pre["failures"] else
                            f"about {pre['estimated_seconds']}s against "
                            f"{pre['bound_seconds']}s"))
        json.dump(out, open(rnd.rel("parts.json"), "w"), indent=1)
        return out
    floors: dict = {}
    surface_records: list = []
    # **The ground, settled once, before the first part.** v2, B1: every leaf's pad and
    # level, every edge's footing, the lanes and the doorsteps, declared on the volume
    # as it stands now and resolved by one contract; each part then lays what was
    # settled for it, and no part's ground is what its neighbour left behind.
    resolved, ground_rec = settle_ground(rnd, be, parts, lambda p: voice_of(p)[1][0])
    out["ground"] = {k: ground_rec[k] for k in ("columns", "declared", "refused",
                                                 "owned_by_class", "seam_totals",
                                                 "seconds", "saved")}
    print(f"   ground: {ground_rec['columns']:,} columns settled for {ground_rec['declared']}"
          f" of {len(parts)} parts and the network in {ground_rec['seconds']}s; seams "
          f"{ground_rec['seam_totals']}", flush=True)
    # **The record of every way in this stage lays is this stage's, written and not
    # appended.** v2, B0. Each part's `approach()` and `flight()` paths went onto the
    # end of whatever `paths.json` the last run left, so a `parts` stage run again
    # duplicated every row -- the example's file read 332 rows for 32 after eight runs,
    # and the finish pass reads it to keep off those columns. The stage owns the file:
    # it starts empty and holds exactly what this run has laid so far, so a stage that
    # dies half-way leaves a true record of the half.
    laid: list = []
    paths_path = rnd.rel("paths.json")

    def write_paths():
        json.dump(laid, open(paths_path, "w"), indent=1)
    write_paths()

    def lint_wave(wave, rows, vol_for_lint):
        nonlocal plots
        # **what the library sited this part at, on the record, before the wave is
        # linted.** `room_owner` needs it to tell a building's own floor from the cave
        # under its plot, and the wave's own lint is the first reader to ask. Written
        # here rather than at plan time because it is not a decision the plan makes:
        # `site()` sounds the ground and answers.
        plots = record_part_floors(rnd, rows, plots)
        by = {p["label"]: p for p in plots}
        # A6: the wave's own parts plus a margin, not the whole place.
        rects = [by[r["part"]] for r in rows if r["part"] in by]
        scope = None
        if rects:
            scope = (min(p["x0"] for p in rects) - pipeline.LINT_MARGIN,
                     min(p["z0"] for p in rects) - pipeline.LINT_MARGIN,
                     max(p["x1"] for p in rects) + pipeline.LINT_MARGIN,
                     max(p["z1"] for p in rects) + pipeline.LINT_MARGIN)
        ctx = lint.Context.build(vol_for_lint, plots, network=rnd.network(),
                                 region=scope, base=base)
        rep = lint.lint(ctx, family=lint.BUILD).within(rects or plots)
        return {"wave": wave, "parts": rows, "scope": list(scope) if scope else None,
                "errors": sorted(f.code for f in rep.findings if f.code.startswith("E")),
                "counts": rep.to_json()["counts"]}

    def finish_wave(wave, rows, rec):
        out["waves"].append(rec)
        for r in rows:
            out["built" if r["status"] == "built" else "failed"] += 1
        record("pass_run", name=wave, settlement=rnd.name,
               placed=sum(r.get("placed") or 0 for r in rows),
               failed=sum(1 for r in rows if r["status"] != "built"),
               seconds=round(sum(r.get("seconds") or 0 for r in rows), 1),
               parts=len(rows),
               built=sum(1 for r in rows if r["status"] == "built"),
               errors=len(rec["errors"]))

    waves = part_waves(parts)
    # The walls and the squares first, in order, on the shared volume: a point stands on
    # its edge's sited floor and a quarter's lanes meet a square. waves are independent
    # districts by construction -- each merged back in wave order.
    serial = [(w, g) for (w, g) in waves if w in ("walls", "squares")]
    quarters = [(w, g) for (w, g) in waves if w not in ("walls", "squares")]
    if n_workers <= 1 or len(quarters) <= 1:
        serial, quarters = waves, []
    for wave, group in serial:
        rows = []
        for part in group:
            if part.get("edge") and part["edge"].get("name") in floors:
                part["edge"]["floor_y"] = floors[part["edge"]["name"]]
            v, (mat, roof) = voice_of(part)
            rec = instantiate_part(rnd, be, part, mat, roof, paths_sink=laid,
                                   ground=resolved)
            write_paths()
            rec["voice"] = v
            sr = rec.pop("surface_record", None)
            if sr:
                surface_records.append(sr)
            if rec.get("floor_y") is not None:
                floors[part["name"]] = int(rec["floor_y"])
            rows.append(rec)
            print(f"   {wave}/{part['name']}: {rec['status']}"
                  + (f" {rec.get('blocks')} blocks, {rec.get('ground')}"
                     if rec["status"] == "built" else f" -- {rec.get('error', '')[:120]}"),
                  flush=True)
        finish_wave(wave, rows, lint_wave(wave, rows, be.volume))
    if quarters:
        for part in (q for _w, g in quarters for q in g):
            voice_of(part)                                # resolve every voice once
        snapshot = rnd.rel("world.quarters-from.npz")
        offline.save_volume(be.volume, snapshot)
        base_path = rnd.rel(rnd.base_volume)
        print(f"   {len(quarters)} quarter wave(s) across {n_workers} workers off "
              f"{os.path.relpath(snapshot, _pipeline.ROOT)}", flush=True)
        from ..parallel import par_map
        jobs = [(rnd, w, g, snapshot, palettes, plots, base_path, pipeline.LINT_MARGIN,
                 voice, rnd.rel("ground.npz")) for (w, g) in quarters]
        got = par_map(_wave_worker, jobs, n=n_workers)
        for res in got:
            be._vol = be.volume.overlay(res["blocks"]) if res["blocks"] else be.volume
            if res["paths"]:
                laid.extend(res["paths"])
                write_paths()
            plots = record_part_floors(rnd, res["rows"], plots)
            for r in res["rows"]:
                sr = r.pop("surface_record", None)
                if sr:
                    surface_records.append(sr)
                print(f"   {res['wave']}/{r['part']}: {r['status']}"
                      + (f" {r.get('blocks')} blocks, {r.get('ground')}"
                         if r["status"] == "built" else f" -- {r.get('error', '')[:120]}"),
                      flush=True)
            finish_wave(res["wave"], res["rows"],
                        {"wave": res["wave"], "parts": res["rows"], "scope": res["scope"],
                         "errors": res["errors"], "counts": res["counts"],
                         "worker_seconds": res["seconds"]})
        out["parallel"] = {"workers": n_workers, "quarter_waves": len(quarters),
                           "snapshot": os.path.relpath(snapshot, _pipeline.ROOT)}
    # **Where a structure was actually entered, written back onto the network.** The
    # neighbourhood delivery round, and it is the rule the spatial design round set for
    # gates applied to every part: *"the doorstep wins -- a doorway that cannot be
    # walked into is not a way in -- and the network record is corrected to say what was
    # laid"* (`circulate.emit`). The circulation pass reserves a doorstep from the
    # ground it plans on; `site()` prepares the plot's own ground, chooses the door cell
    # on the pad it made, and then **measures**, against the world it has just changed,
    # whether a person can walk to it (`approach()`). Where a lane ramps past a plot the
    # two disagree: `middle_ring_north_east_b0_0_02` stands on its district's terrace at
    # y=64, the lane outside it is cut to y=60 for the ramp, and `E008` refused the
    # build for the reserved doorstep while the door `site()` laid four columns away was
    # measured walk-reachable from outside. So the record is corrected to the door that
    # exists, and **only** on a measured positive: `way_in.ok` is `approach()`'s own
    # answer re-read off the built world, not a claim. The reserved cell is kept beside
    # it as `was`, and a part whose way in was *not* measured reachable is left exactly
    # as it was, so `E008` still refuses a structure nobody can enter -- which is the
    # whole of what that check is for. **...and for a leaf with a compiled site this is
    # a check, not a correction.** The quarter design round: the router reserved the
    # compiled door at the compiled floor (`circulate.site_way`) and `site()` built
    # exactly that door, so the two agree by construction. A disagreement is recorded
    # (`sites_disagree`, which should be empty) and the network is left as planned --
    # moving it would hide the very defect the compiled site exists to remove.
    with contextlib.suppress(Exception):
        net = rnd.network()
        moved = []
        disagree = []
        by_part = {r.get("part"): r for w in out.get("waves") or []
                   for r in (w.get("parts") or [])}
        sited_names = set()
        with contextlib.suppress(Exception):
            sited_names = {str(p.get("name")) for p in _pipeline.plan_parts(rnd.plan())
                           if isinstance(p.get("site"), dict)
                           or isinstance(p.get("court_site"), dict)}
        n_checked = 0
        for t in (net.thresholds if net else []):
            r = by_part.get(str(t.id))
            if str(t.id) in sited_names:
                if not r or r.get("status") != "built":
                    continue
                n_checked += 1
                d = [int(v) for v in (r.get("door") or [])]
                laid = (d if len(d) == 3 else
                        [d[0], int(r["floor_y"]) + 1, d[1]]
                        if len(d) == 2 and r.get("floor_y") is not None else None)
                if laid is None or list(t.door) != laid:
                    disagree.append({"part": str(t.id), "planned": [int(v) for v in t.door],
                                     "laid": laid,
                                     "way_in": (r.get("way_in") or {}).get("ok")})
                continue
            if not r or not (r.get("way_in") or {}).get("ok") or not r.get("door"):
                continue
            # `site()` records the door as the pad cell `(x, z)`; its level is the floor
            # a person stands on, which is the floor block plus one -- the same
            # arithmetic `floor_from_threshold` and `building()` use.
            d = [int(v) for v in r["door"]]
            if len(d) == 3:
                laid = d
            elif len(d) == 2 and r.get("floor_y") is not None:
                laid = [d[0], int(r["floor_y"]) + 1, d[1]]
            else:
                continue
            if list(t.door) == laid:
                continue
            moved.append({"part": str(t.id), "was": [int(v) for v in t.door],
                          "now": laid, "why": (r.get("way_in") or {}).get("why")})
            t.door = tuple(laid)
        if moved:
            net.save(rnd.rel("network.json"))
            out["thresholds_corrected"] = {
                "moved": moved[:40], "n": len(moved),
                "why": ("the door each of these parts was entered by is the one `site()` "
                        "laid on the pad it prepared and `approach()` measured walkable "
                        "from outside; the reserved cell is kept on the row as `was`. "
                        "A part whose way in was not measured reachable is untouched")}
            print(f"   thresholds: {len(moved)} door(s) corrected to the way in that was "
                  f"laid and measured walkable", flush=True)
        out["sites_checked"] = {"n": n_checked, "disagree": len(disagree),
                                "rows": disagree[:40],
                                "why": ("parts with a compiled site: the network's "
                                        "reserved door against the door `site()` laid. "
                                        "Recorded, never moved; should be zero")}
        if disagree:
            print(f"   sites: {len(disagree)} of {n_checked} compiled door(s) disagree "
                  f"with the door laid", flush=True)
    # **Which candidate this construction is of.** The closure round: a parts record
    # that cannot say which design it built is a record a revision can inherit.
    from .. import deps as _deps_b
    with contextlib.suppress(Exception):
        out["candidate"] = _deps_b.candidate_id(rnd)
    p = rnd.rel("parts.json")
    json.dump(out, open(p, "w"), indent=1)
    out["written"] = p
    from .. import surfaces as surfaces_mod
    surfaces_mod.write(rnd.rel("surfaces.json"), surface_records,
                       candidate=out.get("candidate"),
                       note=f"{len(surface_records)} part(s) recorded at emission; the "
                            f"built digest is stamped by stage_finish")
    out["surfaces"] = rnd.rel("surfaces.json")
    # the built volume is saved by the driver's cache step after this stage; the stamp
    # is written by `stage_finish`/`stage_lint` once it exists (see `_stamp_built`)
    return out
