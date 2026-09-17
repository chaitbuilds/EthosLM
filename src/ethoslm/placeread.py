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


def read(spec: dict, plan: dict, parts_record: dict, *, voice: str | None = None,
         site: dict | None = None, built=None, base=None,
         plateau: dict | None = None) -> dict:
    """The whole read. No model, no server: the plan, what stood, and -- when the
    built world and the ground it was built on are given -- what it is made of.

    `built` and `base` are `observe.Volume`s: the place as the waves left it and the
    cached ground before the first part was laid. With both, the `palette/built` clause
    is asked (A2 of the voice contract); without them it is not, and a stage that has a
    built world always gives it."""
    parts = pipeline.plan_parts(plan)
    by_name = {p["name"]: p for p in parts}
    decls = _decls(parts)
    stood = _stood(parts_record)
    clauses = []

    def clause(name, ok, says, **more):
        clauses.append({"clause": name, "holds": bool(ok), "says": says, **more})

    # --- present ---------------------------------------------------------
    for d in spec["defining_parts"]:
        if spec_mod.compound(d):
            # A great thing is present when what it is made of stands: its wall closed
            # with a gate on it, its halls and its court inside. See `compound_clauses`.
            continue
        if d["kind"] == "group":
            # A group defining part is districts, and a district is not built: what it
            # is answerable for is that its quarters have plots in them.
            got = {p["in"][-1] for p in parts
                   if p.get("kind", "plot") == "plot" and p.get("in")}
            clause(f"present/{d['name']}", len(got) >= d["count"],
                   f"the spec asks for {d['count']} {d['family']}(s) and the plan has "
                   f"{len(got)} quarter(s) with plots in them",
                   wanted=d["count"], got=sorted(got))
            continue
        mine = _matching(parts, d, decls)
        up = [p["name"] for p in mine if stood.get(p["name"], False)]
        clause(f"present/{d['name']}", len(up) >= d["count"],
               f"the spec asks for {d['count']} x {d['family']} as a {d['kind']} "
               f"({d['relation']}); the plan has {len(mine)} and {len(up)} of them "
               f"stand",
               wanted=d["count"], planned=[p["name"] for p in mine], stood=up)

    # --- closed ----------------------------------------------------------
    asked_cliff = any(w in spec["sentence"].lower() for w in CLIFF_WORDS)
    for d in spec_mod.walls(spec):
        mine = [p for p in _matching(parts, d, decls) if p.get("kind") == "edge"]
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
            gates = _gates_on(p, parts, decls)
            standing_gates = [g for g in gates if stood.get(g, False)]
            ok = closed and up and bool(standing_gates)
            clause(f"closed/{p['name']}", ok,
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
    clauses += compound_clauses(spec, plan, parts, decls, stood, parts_record,
                                plateau=plateau)

    # --- concentric ------------------------------------------------------
    clauses += concentric_clauses(spec, plan, parts, decls, stood)
    clauses += great_wall_clauses(spec, plan, parts, decls, stood)

    # --- count -----------------------------------------------------------
    plots = [p for p in parts if p.get("kind", "plot") == "plot"]
    up = [p["name"] for p in plots if stood.get(p["name"], False)]
    lo, hi = spec["size_band"]
    clause("count", spec_mod.in_band(spec, len(up)),
           f"{len(up)} structure(s) stand of {len(plots)} planned, against the band "
           f"{lo}-{hi} the spec produced",
           stood=len(up), planned=len(plots), band=[lo, hi])

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
        clause("palette/built", **built_palette(chosen, parts, stood, built, base))

    holds = all(c["holds"] for c in clauses)
    return {"holds": bool(holds), "got": 1 if holds else 0,
            "clauses": clauses,
            "failed": [c["clause"] for c in clauses if not c["holds"]],
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
    """The great wall, where the spec asks for one.

        In a place of concentric rings the outermost ring is the great wall, and it is held
        to `GREAT_WALL_HEIGHT`: the height the plan asked its type for, and the wall stood.
        Silent where nothing is concentric -- a walled town of sixty houses asked for a
        wall and not a great one.
        
    """
    got = rings(spec, parts)
    if not got:
        return []
    outer = got[0]
    p = outer["part"]
    h = (p.get("params") or {}).get("height")
    up = stood.get(p["name"], False)
    ok = up and h is not None and int(h) >= GREAT_WALL_HEIGHT
    return [{"clause": f"great/{p['name']}", "holds": bool(ok),
             "says": f"{p['name']} is the outermost of {len(got)} rings and the "
                     f"place's great wall: planned {h if h is not None else 'no'} "
                     f"high as `{p.get('type')}`, against {GREAT_WALL_HEIGHT}"
                     + ("; it stands" if up else "; it DOES NOT STAND"),
             "height": h, "against": GREAT_WALL_HEIGHT, "type": p.get("type"),
             "stood": up}]


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
        if not total:
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
