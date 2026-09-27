#!/usr/bin/env python3
"""**Round walls**: a ring wall drawn along `boundary.Outline.polyline` reads round, is
whole along its length, climbs its ground and is passed through by its gates.

    $PY scripts/test_round_walls.py [name-fragment] [--show] [--terrain]

The design synthesis round. The coordinator states a ring's boundary as an `Outline`
and draws its wall as `Outline.polyline(step=8, gates=..., gate_run=13)`: hundreds of
short axial and 45-degree runs, a straight axial run pinned at each gate. `wall` and
`great_wall` draw such a path with the curved engine they both carry (one text,
asserted identical here). Offline, on a synthetic flat volume, through the production
calls -- `Builder.site()`, the type's `build()` through `type_builder()`, then
`resolve_steps()` -- exactly as `instantiate_part` composes a part:

    r1  a wall ring of radius 150: no gap along its length (every bearing, sampled every
        0.1 degree, meets the wall at its walk), no floating column, towers spaced by
        arc length, and the walk continuous on foot (walked with the ground taken away)
    r2  the same for `great_wall` at width 5 (a crown road between two parapets)
    r3  gates on a curved wall: `ring_gate` on the pinned runs at bearing 90 and 45,
        each walked through on foot from outside to inside
    r4  a wall over a hill: the walk climbs it in half-block courses, never under its
        own segment's footing plus its height, and taller than the ground beside it
    r0  both wall types and `ring_gate` pass the parts preflight (`lint.preflight` with
        `TYPE_FORBIDDEN` and the palette check): no try, no ground call
    r5  a square path still builds byte for byte what it built before (against the
        committed type at HEAD, when git has it)
    site_a  the engine text in `wall.py` and `great_wall.py` is one text
    site_b  a ring cut into runs the way the city compiler cuts it (open runs, each ending
        where the next begins, each carrying `ring_path`), each built on its own builder:
        composed in either order, one wall -- no gap, the walk whole, every tower once,
        the order changing only footing material and never a hole
    site_c  a gate on such a run opens the wall whether the gate or the wall is built first
    site_d  runs built one after another, each on the world the runs before it left and
        composed through `PART_GEOMETRY` (the city build's `wall_palace_003`), never
        climb their neighbours: every block within max(ring_floors) + height + 2, a
        tower's drum within its own courses more -- with the ring's context and without

A hill, a mountain flank and water) and renders it. `--cost R` builds a `great_wall`
ring of radius R on flat ground and prints the seconds and blocks.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from ethoslm import boundary, observe, offline, pipeline, stages  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

OUT = os.path.join(ROOT, "out", "ds-work", "ground")
VOICE = "ochre_stone_green_tile"
GROUND = 64
AIRS = ("air", "cave_air", "void_air")
CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the harness

def flat_volume(size: int, g: int = GROUND, height: int = 120, hill=None) -> Volume:
    """A plane of grass at `g` (top solid y), `size` square about the origin; `hill`
    is (cx, cz, radius, rise): a cone of dirt added to it."""
    y0 = g - 20
    codes = np.zeros((size, height, size), np.uint16)
    x0 = z0 = -size // 2
    h = np.full((size, size), g, np.int32)
    if hill:
        hx, hz, hr, rise = hill
        xs = np.arange(size)[:, None] + x0
        zs = np.arange(size)[None, :] + z0
        d = np.hypot(xs - hx, zs - hz)
        h = h + np.clip(np.round(rise * (1 - d / hr)), 0, None).astype(np.int32)
    ys = np.arange(height)[None, :, None] + y0
    hh = h[:, None, :]
    codes[ys < hh - 3] = 3
    codes[(ys >= hh - 3) & (ys < hh)] = 2
    codes[ys == hh] = 1
    return Volume(x0, y0, z0, codes, ["air", "grass_block", "dirt", "stone"])


def _decl(type_name: str) -> dict:
    path = os.path.join(ROOT, "types", f"{type_name}.py")
    ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
    exec(compile(open(path).read(), path, "exec"), ns)            # noqa: S102
    return ns


def _decl_src(src: str) -> dict:
    ns: dict = {"__name__": "__ethoslm_type__"}
    exec(compile(src, "<type>", "exec"), ns)                      # noqa: S102
    return ns


def build_edge(vol, type_name, path, width, params, *, seed=3, gates=(), ns=None,
               extra=None, voice=VOICE):
    """Site and build one edge part (and its gates, `ring_gate` on each point) through
    the production calls. Returns everything the checks read."""
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    mat, roof = pipeline.voice_palette(voice), pipeline.voice_roof(voice)
    ns = ns or _decl(type_name)
    H = int(params.get("height", 12))
    gparts = []
    for k, g in enumerate(gates):
        size = Builder.point_pad(H, _decl("ring_gate")["NEEDS"]["footprint"])
        gparts.append({"label": f"gate_{k}", "kind": "point", "at": [g["x"], g["z"]],
                       "facing": g["facing"], "size": size,
                       "edge": {"name": "ring", "type": type_name, "height": H,
                                "width": width}})
    part = {"label": "ring", "kind": "edge", "path": [list(p) for p in path],
            "width": width, **(extra or {})}
    if gparts:
        part["gates"] = [{"name": g["label"], "at": g["at"], "size": g["size"]}
                         for g in gparts]
    t0 = time.perf_counter()
    p = b.site(dict(part), mat=mat, roof=roof)
    t1 = time.perf_counter()
    res = ns["build"](b.type_builder(p, role=ns.get("ROLE")), p, seed, **params)
    t2 = time.perf_counter()
    gres = []
    if gparts:
        gns = _decl("ring_gate")
        for gp in gparts:
            sp = b.site(dict(gp), mat=mat, roof=roof)
            gres.append((sp, gns["build"](b.type_builder(sp, role=gns.get("ROLE")), sp,
                                          5, storeys=2, crown="hip")))
    b.resolve_steps()
    t3 = time.perf_counter()
    built = stages.apply_pending(vol, b._pending)
    return {"b": b, "part": p, "res": res, "gates": gres, "built": built, "vol": vol,
            "blocks": len(b._pending),
            "seconds": {"site": round(t1 - t0, 2), "build": round(t2 - t1, 2),
                        "gates_and_steps": round(t3 - t2, 2)}}


def ring_path(R, centre=(0, 0), gates=(), shape=None):
    o = boundary.Outline.from_record(shape or {"shape": "circle"}, centre, R)
    return o, o.polyline(step=8, gates=list(gates), gate_run=13)


def _solid(vol, x, y, z):
    return vol.state(x, y, z).split("[")[0] not in AIRS


def spine_cells(path):
    return boundary.polyline_cells(path)


def gaps_by_bearing(vol, o, walk_min, frac=1.0, step_deg=0.1, band=4.0):
    """Bearings at which no column within `band` of the outline has a block at or
    above `walk_min`: a gap in the wall seen from the centre."""
    cx, cz = o.centre
    miss = []
    b = 0.0
    while b < 360.0:
        r = float(o.radius_at(b)) * frac
        dx, dz = boundary.bearing_vec(b)
        hit = False
        for k in np.arange(-band, band + 0.01, 0.5):
            x = int(math.floor(cx + (r + k) * dx + 0.5))
            z = int(math.floor(cz + (r + k) * dz + 0.5))
            if _solid(vol, x, walk_min, z):
                hit = True
                break
        if not hit:
            miss.append(round(b, 1))
        b += step_deg
    return miss


def walk_continuity(built, path, y_cut, walk_y_of):
    """Walk the wall's top with the ground below `y_cut` taken away: from the first
    spine cell's stance, every 7th spine cell's stance must be reached on foot with no
    jump. Returns (reached, total, first missing)."""
    codes = built.codes.copy()
    codes[:, : max(0, y_cut - built.y0), :] = 0
    v = Volume(built.x0, built.y0, built.z0, codes, list(built.palette))
    nav = observe.Nav(v)
    cells = spine_cells(path)
    seeds = []
    for (x, z) in cells[:5]:
        s = nav.stance_near(x, z, walk_y_of(x, z) + 1, tol=3)
        if s is not None:
            seeds.append((x, z, s))
    dist = nav.flood(seeds, max_jumps=0)
    got = {}
    for (x, z, s) in dist:
        got.setdefault((x, z), set()).add(s)
    want = cells[::7]
    reached = [c for c in want if c in got]
    missing = [c for c in want if c not in got]
    return len(reached), len(want), missing[:5]


def top_solid(vol, x, z, y_hi=None):
    sy = vol.codes.shape[1]
    for y in range((y_hi if y_hi is not None else vol.y0 + sy - 1), vol.y0 - 1, -1):
        if _solid(vol, x, y, z):
            return y
    return None


def gate_walk(built, g, size, reach=None, y_near=None):
    """Walk from outside the gate to inside it, on foot, without a jump. `g` is the
    gate point record (`Outline.gate_points`), facing outward."""
    fx, fz = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[g["facing"]]
    # four columns past each mouth of the pad, on the ground outside and inside. the
    # turf it stripped was never put back -- and stepping out of one is a jump;
    # `_site_rect` now records its clearing and restores it.)
    reach = reach or (size // 2 + 4)
    ox, oz = g["x"] + fx * reach, g["z"] + fz * reach
    ix, iz = g["x"] - fx * reach, g["z"] - fz * reach
    x0, x1 = min(ox, ix) - size, max(ox, ix) + size
    z0, z1 = min(oz, iz) - size, max(oz, iz) + size
    sub = built.sub(x0, z0, x1 - x0 + 1, z1 - z0 + 1)
    nav = observe.Nav(sub)
    # the stance on the ground (not on a roof over it, not in a cave under it): the one
    # nearest the top of the ground below the gate's arch
    y_hi = y_near + 4 if y_near is not None else None

    def stance(x, z):
        t = top_solid(built, x, z, y_hi=y_hi)
        return None if t is None else nav.stance_near(x, z, t + 1, tol=2)
    so, si = stance(ox, oz), stance(ix, iz)
    if so is None or si is None:
        return None
    return nav.route([(ox, oz, so)], (ix, iz, si), max_jumps=0)


# ------------------------------------------------------------------ the cases

def _ring_checks(r, o, path, H_walk, tag, every):
    built, res = r["built"], r["res"]
    out = []
    assert res.get("curved"), f"{tag}: the curved engine did not draw it: {res}"
    walk_min = GROUND + H_walk - 1
    miss = gaps_by_bearing(built, o, walk_min)
    out.append(f"bearings with no wall at y>={walk_min}: {len(miss)}")
    assert not miss, f"{tag}: a gap in the wall at bearings {miss[:10]}"
    assert res["floating_columns"] == 0, f"{tag}: floating columns {res['floating_columns']}"
    assert res["max_walk_step_half"] <= 1, f"{tag}: walk steps {res['max_walk_step_half']}"
    arcs = res["tower_arc"]
    gaps = sorted(((arcs[(i + 1) % len(arcs)] - arcs[i]) % res["arc"]) for i in range(len(arcs)))
    out.append(f"{len(arcs)} towers on {res['arc']} blocks of arc, spacing "
               f"{gaps[0]:.0f}..{gaps[-1]:.0f} (asked {every})")
    assert len(arcs) >= 3 and gaps[0] >= every * 0.5 and gaps[-1] <= every * 1.6, \
        f"{tag}: tower spacing {gaps}"
    # never a tower at every vertex: far fewer towers than vertices
    assert len(arcs) * 4 < len(path), f"{tag}: {len(arcs)} towers on {len(path)} vertices"

    def walk_y(x, z):
        return GROUND + H_walk
    got, want, missing = walk_continuity(built, path, GROUND + H_walk - 3, walk_y)
    out.append(f"walk reached on foot at {got}/{want} spine stations sampled")
    assert got == want, f"{tag}: the walk breaks: {missing}"
    ok_t = [t for t in res["turrets"] if t.get("ok")]
    out.append(f"{len(ok_t)} stair turrets, {res['columns']} columns, {r['blocks']} blocks, "
               f"{r['seconds']}")
    return out


@case
def r0_the_types_pass_preflight():
    """What `instantiate_part` runs before a type is composed: no try, no ground call,
    no block literal outside the palette. A type that fails here is never built."""
    from ethoslm.lint import preflight
    out = []
    for t in ("wall", "great_wall", "ring_gate"):
        src = open(os.path.join(ROOT, "types", f"{t}.py")).read()
        r = preflight(src, forbid=pipeline.TYPE_FORBIDDEN, palette=True)
        assert r.ok, (t, r.to_json())
        out.append(f"{t} ok")
    return ", ".join(out)


@case
def r1_wall_ring_round_whole_and_walkable():
    o, path = ring_path(150)
    vol = flat_volume(380)
    r = build_edge(vol, "wall", path, 3, {"height": 12, "width": 3})
    lines = _ring_checks(r, o, path, 10, "wall", _decl("wall")["TOWER_EVERY"])
    _show(r, o, "r1_wall_ring") if SHOW else None
    return "; ".join(lines)


@case
def r2_great_wall_ring_with_a_crown_road():
    o, path = ring_path(150)
    vol = flat_volume(380, height=140)
    r = build_edge(vol, "great_wall", path, 5, {"height": 30, "face": "masonry"})
    lines = _ring_checks(r, o, path, 30, "great_wall", _decl("great_wall")["TOWER_EVERY"])
    _show(r, o, "r2_great_wall_ring") if SHOW else None
    return "; ".join(lines)


@case
def r3_gates_on_a_curved_wall_are_walked_through():
    gb = [90.0, 45.0]
    o, path = ring_path(150, gates=gb)
    gps = o.gate_points(gb)
    bad = boundary.check_polyline(path)
    assert not bad, f"the polyline has off-lattice runs {bad[:3]}"
    out = []
    for t, params, w in (("wall", {"height": 12, "width": 3}, 3),
                         ("great_wall", {"height": 30, "face": "masonry"}, 5)):
        vol = flat_volume(380, height=150)
        r = build_edge(vol, t, path, w, params, gates=gps)
        for g, (sp, gr) in zip(gps, r["gates"]):
            route = gate_walk(r["built"], g, int(sp.get("size") or 5))
            out.append(f"{t} bearing {g['bearing']:.0f} facing {g['facing']} "
                       f"(pad {sp.get('size')}, {'tower' if gr.get('tower') else 'gatehouse'}): "
                       + ("walked through in %d steps" % len(route) if route
                          else "NOT WALKABLE"))
            assert route, f"{t}: gate at bearing {g['bearing']} cannot be walked through: {gr}"
        _show(r, o, f"r3_gates_{t}", gate=gps[1]) if SHOW else None
    return "; ".join(out)


@case
def r4_a_wall_climbs_a_hill():
    o, path = ring_path(150)
    # a hill of 18 over the plane, centred on the ring's east side
    vol = flat_volume(400, height=140, hill=(150, 0, 60, 18))
    r = build_edge(vol, "wall", path, 3, {"height": 12, "width": 3})
    built, res, p = r["built"], r["res"], r["part"]
    assert res["floating_columns"] == 0, res["floating_columns"]
    assert res["max_walk_step_half"] <= 1, res["max_walk_step_half"]
    # along the spine: the walk never under its own segment's floor + 10, and over the
    # hill higher than the ground either side of it
    worst_margin = None
    over = 0
    for s in p["segments"]:
        for (x, z) in Builder._edge_run(s["a"], s["b"]):
            top = top_solid(built, x, z)
            if top is None:
                continue
            # the walk: the highest paving at or under the parapet of this column
            need = int(s["floor_y"]) + 10
            assert top >= need, f"walk at {(x, z)} is {top}, under {need}"
            side = max(top_solid(vol, x + dx, z + dz) for dx in (-6, 0, 6) for dz in (-6, 0, 6))
            m = top - side
            worst_margin = m if worst_margin is None else min(worst_margin, m)
            over += 1
    assert worst_margin is not None and worst_margin >= 4, worst_margin
    hill_top = int(top_solid(vol, 150, 0))
    wall_top = int(top_solid(built, 150, 0))

    def walk_y(x, z):
        return top_solid(built, x, z) - 1
    got, want, missing = walk_continuity(built, path, GROUND + 6, walk_y)
    assert got == want, f"the walk over the hill breaks: {missing}"
    _show(r, o, "r4_hill", street=(150, 0)) if SHOW else None
    return (f"hill top y={hill_top}, the wall's crown over it y={wall_top}; the walk at "
            f"least {worst_margin} over the ground 6 either side along {over} spine "
            f"columns; walk steps <= {res['max_walk_step_half']} half-block; walked "
            f"{got}/{want}")


def cut_runs(path, region=128, origin=(-1000, -1000)):
    """A closed ring polyline cut into open runs where it crosses a region edge, as the
    city compiler cuts it (`cityresolve`): each run's last vertex is the next's first."""
    cells = boundary.polyline_cells(path)
    runs, cur, pts = [], None, []
    for (x, z) in cells:
        rid = ((x - origin[0]) // region, (z - origin[1]) // region)
        if rid != cur:
            if pts:
                runs.append(pts)
            cur, pts = rid, ([pts[-1]] if pts else [])
        pts.append((x, z))
    if pts:
        runs.append(pts)
    out = []
    for pts in runs:
        v = [pts[0]]
        for i in range(1, len(pts) - 1):
            if (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) != \
                    (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]):
                v.append(pts[i])
        v.append(pts[-1])
        if len(v) >= 2:
            out.append([list(q) for q in v])
    return out


@case
def r7_runs_of_a_ring_built_apart_join_as_one_wall():
    o, path = ring_path(150)
    runs = cut_runs(path, region=96)
    out = []
    for t, params, w in (("wall", {"height": 12, "width": 3}, 3),
                         ("great_wall", {"height": 30, "face": "masonry"}, 5)):
        vol = flat_volume(380, height=140, hill=(150, 0, 60, 14))
        built = []
        for k, rp in enumerate(runs):
            r = build_edge(vol, t, rp, w, params, seed=11,
                           extra={"ring_path": [list(q) for q in path], "wall": "ring"})
            assert r["res"]["mode"] == "ring", r["res"]
            built.append(r)
        fwd, rev = dict(), dict()
        for r in built:
            fwd.update(r["b"]._pending)
        for r in built[::-1]:
            rev.update(r["b"]._pending)
        diff = [k for k in set(fwd) | set(rev) if fwd.get(k) != rev.get(k)]

        def solid(v):
            return v is not None and v.split("[")[0] not in AIRS
        # what the order may change: a footing course's material. Never a hole (air one
        # way and a block the other), and never anything within three of the top.
        holes = [k for k in diff if solid(fwd.get(k)) != solid(rev.get(k))
                 and solid(vol.state(*k)) != solid(fwd.get(k) or vol.state(*k))
                 or solid(fwd.get(k)) != solid(rev.get(k))]
        tops = {}
        for r in built:
            for (x, y, z), v in r["b"]._pending.items():
                if solid(v) and y > tops.get((x, z), -999):
                    tops[(x, z)] = y
        top_diff = holes + [k for k in diff if k not in holes
                            and k[1] >= tops.get((k[0], k[2]), 999) - 3]
        vf = stages.apply_pending(vol, fwd)
        H_walk = params["height"] - (2 if t == "wall" else 0)
        miss = gaps_by_bearing(vf, o, GROUND + H_walk - 1)
        assert not miss, f"{t}: gaps where runs meet, at bearings {miss[:10]}"

        def walk_y(x, z):
            return top_solid(vf, x, z) - 1
        got, want, missing = walk_continuity(vf, path, GROUND + 5 + (H_walk // 2), walk_y)
        assert got == want, f"{t}: the walk breaks where runs meet: {missing}"
        arcs = sorted(a for r in built for a in r["res"]["tower_arc"])
        ring_n = built[0]["res"]["ring_towers"]
        skipped = sum(r["res"]["towers_without_ground"] for r in built)
        assert len(set(arcs)) == len(arcs) and len(arcs) + skipped == ring_n, \
            f"{t}: {len(arcs)} towers built by the runs, {skipped} without ground, the " \
            f"ring has {ring_n}"
        steps = max(r["res"]["max_walk_step_half"] for r in built)
        assert steps <= 1, steps
        assert not top_diff, f"{t}: the order the runs were built in shows above the " \
                             f"footing at {len(top_diff)} cells, e.g. {top_diff[:4]}"
        out.append(f"{t}: {len(runs)} runs built apart; {len(arcs)} of the ring's {ring_n} "
                   f"towers each built once ({skipped} without ground); "
                   f"no gap at any bearing; walk whole ({got}/{want}); order of building "
                   f"changes {len(diff)} cells, every one a footing course's "
                   f"material, none a hole")
        if SHOW:
            _show({"built": vf, "res": built[0]["res"]}, o, f"r7_runs_{t}", street=(150, 0))
    return "; ".join(out)


def build_gate(vol, g, H, wall_type, width, voice=VOICE):
    """One `ring_gate` on its own builder, as a separate leaf of its region."""
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    mat, roof = pipeline.voice_palette(voice), pipeline.voice_roof(voice)
    size = Builder.point_pad(H, _decl("ring_gate")["NEEDS"]["footprint"])
    gp = {"label": f"gate_{int(g['bearing'])}", "kind": "point", "at": [g["x"], g["z"]],
          "facing": g["facing"], "size": size,
          "edge": {"name": "ring", "type": wall_type, "height": H, "width": width}}
    gns = _decl("ring_gate")
    sp = b.site(dict(gp), mat=mat, roof=roof)
    res = gns["build"](b.type_builder(sp, role=gns.get("ROLE")), sp, 5, storeys=2,
                       crown="hip")
    b.resolve_steps()
    return {"b": b, "part": sp, "res": res, "size": size}


@case
def r8_a_gate_on_a_cut_run_opens_the_wall_in_either_order():
    gb = [90.0, 45.0]
    o, path = ring_path(150, gates=gb)
    gps = o.gate_points(gb)
    runs = cut_runs(path, region=96)
    out = []
    for t, params, w in (("wall", {"height": 12, "width": 3}, 3),
                         ("great_wall", {"height": 30, "face": "masonry"}, 5)):
        vol = flat_volume(380, height=150)
        H = params["height"]
        size = Builder.point_pad(H, _decl("ring_gate")["NEEDS"]["footprint"])
        walls = []
        for rp in runs:
            band = set()
            for a, c in zip(rp, rp[1:]):
                band.update(Builder.edge_band(a, c, w))
            mine = [{"name": f"gate_{int(g['bearing'])}", "at": [g["x"], g["z"]],
                     "size": size} for g in gps if (g["x"], g["z"]) in band]
            extra = {"ring_path": [list(q) for q in path], "wall": "ring",
                     "ring_gates": [[g["x"], g["z"]] for g in gps]}
            if mine:
                extra["gates"] = mine
            walls.append(build_edge(vol, t, rp, w, params, seed=11, extra=extra))
        gates = [build_gate(vol, g, H, t, w) for g in gps]
        for order in ("gates first", "gates last"):
            pend = {}
            seq = (gates + walls) if order == "gates first" else (walls + gates)
            for r in seq:
                pend.update(r["b"]._pending)
            vf = stages.apply_pending(vol, pend)
            for g, gr in zip(gps, gates):
                route = gate_walk(vf, g, gr["size"])
                assert route, f"{t}, {order}: the gate at {g['bearing']} does not open"
            out.append(f"{t} {order}: both gates walked through")
    return "; ".join(out)


@case
def r9_runs_built_one_after_another_never_climb_their_neighbours():
    """The city build's regression (`wall_palace_003`, out/ds-city region 3_4): runs are
    built region after region, each on the world as its neighbours left it, and the
    leaf reaches the type through `PART_GEOMETRY`. Its end segments were sited on the
    neighbour's wall (floors of 133 and 182 under a wall at 83), the ring context was
    dropped on the way to the type, and the walk climbed half a block a station to
    meet the false floors: a crown ~72 over its floor. Here: a ring cut into runs, each
    built on the volume the runs before it left, the part composed through
    `PART_GEOMETRY`, with the ring's context and without it; every column's top stays
    within max(ring_floors) + height + 2, a tower's drum within its own courses more
    (without the ring: the spine's median within it, nothing past one wall height more)."""
    from ethoslm import designground as DG
    o, path = ring_path(150)
    runs = cut_runs(path, region=96)
    out = []
    for t, params, w in (("wall", {"height": 12, "width": 3}, 3),
                         ("great_wall", {"height": 30, "face": "masonry"}, 5)):
        base = flat_volume(380, height=150)
        target = np.full((380, 380), GROUND, np.int16)
        floors = DG.wall_floors(path, w, target, base.x0, base.z0)
        H = params["height"]
        drum = 4 if t == "wall" else 6
        for with_ring in (True, False):
            vol = base
            worst = worst_t = -999
            for k, rp in enumerate(runs):
                leaf = {"label": f"ring_{k:03d}", "kind": "edge", "path": rp, "width": w,
                        "wall": "ring"}
                if with_ring:
                    leaf.update({"ring_path": [list(q) for q in path],
                                 "ring_floors": floors, "ring_gates": []})
                geo = {q: leaf[q] for q in pipeline.PART_GEOMETRY if q in leaf}
                assert not with_ring or "ring_floors" in geo, "the ring is dropped"
                extra = {q: v for q, v in geo.items() if q not in ("path", "width", "kind",
                                                                     "label")}
                r = build_edge(vol, t, rp, w, params, seed=11 + k, extra=extra)
                bound = max(floors) + H + 2
                for (x, y, z), v in r["b"]._pending.items():
                    if v.split("[")[0] in AIRS:
                        continue
                    worst = max(worst, y - bound)
                vol = stages.apply_pending(vol, r["b"]._pending)
            # the finished ring: every column's top, towers allowed their drum
            tops = []
            for (x, z) in boundary.polyline_cells(path)[::3]:
                tops.append(top_solid(vol, x, z) - (max(floors) + H + 2))
            if with_ring:
                assert worst <= drum + 2, (t, with_ring, worst)
                assert max(tops) <= drum + 2 and np.median(tops) <= 0, (t, max(tops))
            else:
                # without the ring a short run whose every segment stands on its built
                # neighbour cannot know its floor: bounded (it used to climb ~72), not
                # exact -- which is why the leaf must carry `ring_floors`
                assert worst <= H + drum + 2 and np.median(tops) <= 0, (t, worst)
            out.append(f"{t} {'with' if with_ring else 'without'} ring context: highest "
                       f"block {worst:+d} against max(floors)+height+2 (tower drums "
                       f"allowed +{drum + 2}), spine median {int(np.median(tops)):+d}")
    return "; ".join(out)


def _head_source(rel):
    try:
        return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    except Exception:                                   # noqa: BLE001
        return None


@case
def r5_a_square_path_builds_as_before():
    sq = [[-40, -40], [40, -40], [40, 40], [-40, 40], [-40, -40]]
    out = []
    for t, params, w in (("wall", {"height": 12, "width": 3, "crown": "crenellated"}, 3),
                         ("great_wall", {"height": 30, "face": "framed"}, 3),
                         ("great_wall", {"height": 36, "face": "masonry"}, 5)):
        vol = flat_volume(140, height=140, hill=(40, 0, 30, 6))
        now = build_edge(vol, t, sq, w, params)
        assert not (now["res"] or {}).get("curved"), f"{t}: a square went to the curved engine"
        src = _head_source(f"types/{t}.py")
        if src is None:
            out.append(f"{t}: built, {now['blocks']} blocks (no git HEAD to compare)")
            continue
        was = build_edge(vol, t, sq, w, params, ns=_decl_src(src))
        same = was["b"]._pending == now["b"]._pending
        assert same, (f"{t}: the square path's blocks changed: "
                      f"{len(was['b']._pending)} -> {len(now['b']._pending)}")
        out.append(f"{t} {params}: identical to HEAD ({now['blocks']} blocks)")
    return "; ".join(out)


@case
def r6_one_engine_text():
    def engine(t):
        s = open(os.path.join(ROOT, "types", f"{t}.py")).read()
        a = s.index("# THE CURVED WALL ENGINE")
        return s[a:s.index("#: A tower on a curved wall every")]
    a, b = engine("wall"), engine("great_wall")
    assert a == b, "the curved engine differs between wall.py and great_wall.py"
    return f"{len(a.splitlines())} lines, identical"


# ------------------------------------------------------------------ pictures

SHOW = "--show" in sys.argv


def _show(r, o, tag, gate=None, street=None):
    import ca_eye
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    built = r["built"]
    cx, cz = o.centre
    R = float(o.radius_at(0))
    top = int(r["res"].get("top") or GROUND + 20)
    img = ca_eye.render(built, (cx - R * 1.15, GROUND + R * 0.85, cz + R * 1.35),
                        (cx, GROUND, cz), size=(960, 540), fov=60, far=4 * R)
    Image.fromarray(img).save(os.path.join(OUT, f"{tag}_aerial.png"))
    if gate is not None:
        fx, fz = {"north": (0, -1), "south": (0, 1), "east": (1, 0),
                  "west": (-1, 0)}[gate["facing"]]
        at = (gate["x"] + fx * 26 + fz * 6, GROUND + 3, gate["z"] + fz * 26 + fx * 6)
        look = (gate["x"], GROUND + 6, gate["z"])
    elif street is not None:
        sx, sz = street
        gy = top_solid(built, sx - 40, sz + 28) or GROUND
        at = (sx - 40, gy + 2.6, sz + 28)
        look = (sx, (top_solid(built, sx, sz) or GROUND) - 4, sz)
    else:
        b = 135.0
        px, pz = o.point_at(b, 0.82)
        qx, qz = o.point_at(b + 12, 1.0)
        at = (px, GROUND + 2.6, pz)
        look = (qx, top - 4, qz)
    img = ca_eye.render(built, at, look, size=(960, 540), fov=75, far=400)
    Image.fromarray(img).save(os.path.join(OUT, f"{tag}_street.png"))
    # and on the walk itself
    b0 = 200.0
    px, pz = o.point_at(b0, 1.0)
    qx, qz = o.point_at(b0 + 9, 1.0)
    wy = top_solid(built, int(round(px)), int(round(pz)), y_hi=top - 1) or top
    img = ca_eye.render(built, (px, wy + 2.6, pz), (qx, wy + 1.5, qz), size=(960, 540),
                        fov=75, far=300)
    Image.fromarray(img).save(os.path.join(OUT, f"{tag}_walk.png"))


def terrain():
    import ca_eye
    from PIL import Image
    src = os.path.join(ROOT, "out", "ca-city", "world.before-plateau.npz")
    whole = offline.load_volume(src)
    cxw, czw = whole.x0 + 330, whole.z0 + 480
    R = 200
    vol = whole.sub(cxw - R - 40, czw - R - 40, 2 * R + 80, 2 * R + 80)
    del whole
    gb = [0.0, 225.0]
    o, path = ring_path(R, centre=(cxw, czw), gates=gb)
    gps = o.gate_points(gb)
    out = {}
    for t, params, w in (("wall", {"height": 14, "width": 3}, 3),
                         ("great_wall", {"height": 30, "face": "masonry"}, 5)):
        r = build_edge(vol, t, path, w, params, gates=gps)
        res = r["res"]
        print(f"{t}: {r['seconds']} {r['blocks']} blocks; towers {res['towers']}, "
              f"turrets ok {sum(1 for q in res['turrets'] if q.get('ok'))}/"
              f"{len(res['turrets'])}, walk step <= {res['max_walk_step_half']} half, "
              f"floating {res['floating_columns']}, top {res['top']}")
        for g, (sp, gr) in zip(gps, r["gates"]):
            size = int(sp.get("size") or 5)
            # mouth to mouth: one column past each end of the pad (the ground beyond is
            # the site as found -- a cliff at bearing 0 -- which designed ground ramps)
            route = gate_walk(r["built"], g, size, reach=size // 2 + 1,
                              y_near=int(sp["floor_y"]))
            print(f"   gate {g['bearing']:.0f}: "
                  + (f"walked through mouth to mouth ({len(route)} steps)" if route
                     else "not walkable"))
        built = r["built"]
        # the ground and the wall's crown along the spine
        segs = r["part"]["segments"]
        lows = []
        for s in segs:
            for (x, z) in Builder._edge_run(s["a"], s["b"]):
                g = top_solid(vol, x, z)
                t_ = top_solid(built, x, z)
                if g is not None and t_ is not None:
                    lows.append(t_ - g)
        print(f"   crown over the ground as found along the spine: min {min(lows)}, "
              f"median {int(np.median(lows))}, max {max(lows)}")
        img = ca_eye.render(built, (cxw - R * 1.2, 150 + R * 0.6, czw + R * 1.3),
                            (cxw, 80, czw), size=(1280, 720), fov=60, far=5 * R)
        Image.fromarray(img).save(os.path.join(OUT, f"terrain_{t}_aerial.png"))
        # a street view at the hill the ring crosses (bearing ~ 0: x-index 330, z 280)
        hx, hz = o.point_at(0.0, 1.0)
        gy = top_solid(built, int(hx) + 30, int(hz) + 30) or 80
        img = ca_eye.render(built, (hx + 34, gy + 3, hz + 40), (hx, gy + 12, hz),
                            size=(1280, 720), fov=75, far=500)
        Image.fromarray(img).save(os.path.join(OUT, f"terrain_{t}_street.png"))
        out[t] = r["seconds"]
    return out


def cost(R, t="great_wall"):
    o, path = ring_path(R)
    vol = flat_volume(2 * R + 60, height=140)
    params = {"height": 36, "face": "masonry"} if t == "great_wall" else {"height": 16, "width": 3}
    r = build_edge(vol, t, path, 5 if t == "great_wall" else 3, params)
    res = r["res"]
    print(f"{t} ring R={R}: {len(path) - 1} runs, {res['stations']} stations, "
          f"{res['arc']} blocks of arc; site {r['seconds']['site']} s, build "
          f"{r['seconds']['build']} s, steps {r['seconds']['gates_and_steps']} s; "
          f"{r['blocks']} blocks; {res['towers']} towers; "
          f"{sum(1 for q in res['turrets'] if q.get('ok'))} turrets")


def main() -> int:
    if "--cost" in sys.argv:
        R = int(sys.argv[sys.argv.index("--cost") + 1])
        t = sys.argv[sys.argv.index("--cost") + 2] if len(sys.argv) > sys.argv.index("--cost") + 2 else "great_wall"
        cost(R, t)
        return 0
    if "--terrain" in sys.argv:
        terrain()
        return 0
    frag = next((a for a in sys.argv[1:] if not a.startswith("--")), "")
    failed = 0
    for fn in CASES:
        if frag and frag not in fn.__name__:
            continue
        t0 = time.perf_counter()
        try:
            msg = fn()
            print(f"PASS {fn.__name__} ({time.perf_counter() - t0:.1f}s): {msg}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
