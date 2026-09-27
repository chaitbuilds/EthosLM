#!/usr/bin/env python3
"""**The city gate** (`types/ring_gate.py`): a wide vaulted passage through the wall
body, a timber gate tower on the wall's top, the gate joined to its wall.

    $PY scripts/test_ring_gate.py [name-fragment] [--show]

Offline, on a synthetic plane, through the production calls (`Builder.site()`, the
type's `build()` through `type_builder()`, `resolve_steps()`), on a round wall drawn
the way the city compiler draws it (`Outline.polyline` with a straight axial run
pinned at each gate) -- the harness of `scripts/test_round_walls.py`:

    g0  the type passes the parts preflight (no try, no ground call, palette)
    g1  on walls 10 and 44 high, 3 and 9 wide, a cardinal gate and a 45-degree gate:
        each walked through on foot from outside to inside with no jump, through a
        passage at least `passage` wide and six high; the wall's walk walked across the
        gate's deck from one side to the other with no jump and the ground taken away;
        no E-code from the linter on the gate's plot; each gate built in under five
        seconds
    g2  rank: the great wall's gate has three eaves and the wide passage; a lane
        through the pad widens the passage to the lane; a nine-high precinct wall
        gets a smaller tower than a twelve-high one
    g3  no E-code from the linter on flat ground (and on the needs sweep's bank), at
        the pads `NEEDS` declares, over every storey count and crown and three
        passages, standing on its own (`scripts/type_needs.py`'s instance)

`--show` renders the gates -- outside on the axis, oblique, and inside -- to
`out/ds-work/gates/`.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


from ethoslm import boundary, observe, offline, pipeline, stages  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

import test_round_walls as T  # noqa: E402

OUT = os.path.join(ROOT, "out", "ds-work", "gates")
VOICE = "hutong_grey_brick_tile"
GROUND = T.GROUND
CASES: list = []
SHOW = "--show" in sys.argv
DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def case(fn):
    CASES.append(fn)
    return fn


def gate_decl():
    return T._decl("ring_gate")


def build_ring_with_gates(wall_type, H, width, bearings, gparams, *, R=150,
                          voice=VOICE, gate_run=None, vol=None, road=0):
    """A round wall and a `ring_gate` on each gate point, as `instantiate_part` composes
    them: the wall's leaf first, then each gate on the world the wall left. `road`, a
    width: a lane that wide runs through each gate along the way it faces."""
    o = boundary.Outline.from_record({"shape": "circle"}, (0, 0), R)
    path = o.polyline(step=8, gates=list(bearings),
                      gate_run=gate_run or max(13, width + 6))
    gps = o.gate_points(list(bearings))
    vol = vol or T.flat_volume(2 * R + 80, height=175)
    mat, roof = pipeline.voice_palette(voice), pipeline.voice_roof(voice)
    gns = gate_decl()
    size = Builder.point_pad(H, gns["NEEDS"]["footprint"])
    params = ({"height": H, "width": min(5, width)} if wall_type == "wall"
              else {"height": H, "face": "masonry"})
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    edge = {"label": "ring", "kind": "edge", "path": [list(p) for p in path],
            "width": width,
            "gates": [{"name": f"g{k}", "at": [g["x"], g["z"]], "size": size}
                      for k, g in enumerate(gps)]}
    p = b.site(dict(edge), mat=mat, roof=roof)
    wns = T._decl(wall_type)
    wns["build"](b.type_builder(p, role=wns.get("ROLE")), p, 3, **params)
    b.resolve_steps()
    vol = stages.apply_pending(vol, b._pending)
    gates = []
    for k, g in enumerate(gps):
        if road:
            from ethoslm.circulate import Network
            from ethoslm.frontage import Frontage
            fx, fz = DIRS[g["facing"]]
            cells = {}
            for t in range(-24, 25):
                for k in range(-(road // 2), road - road // 2):
                    x = g["x"] + fx * t + (0 if fx else k)
                    z = g["z"] + fz * t + (k if fx else 0)
                    cells[(x, z)] = {"y": GROUND, "rank": 0, "face": None}
            vol = vol.overlay({(x, GROUND, z): "cobblestone" for (x, z) in cells
                               if T.top_solid(vol, x, z) == GROUND})
        gb = Builder(offline.OfflineSite(vol))
        gb._vol = vol
        if road:
            gb.frontage = Frontage(vol, Network(cells, []))
        gp = {"label": f"gate_{k}", "kind": "point", "at": [g["x"], g["z"]],
              "facing": g["facing"], "size": size,
              "edge": {"name": "ring", "type": wall_type, "height": H, "width": width,
                       "floor_y": int(p["floor_y"])}}
        t0 = time.perf_counter()
        sp = gb.site(dict(gp), mat=mat, roof=roof)
        res = gns["build"](gb.type_builder(sp, role=gns.get("ROLE")), sp, 5 + k,
                           **gparams)
        gb.resolve_steps()
        secs = time.perf_counter() - t0
        vol = stages.apply_pending(vol, gb._pending)
        gates.append({"g": g, "part": sp, "res": res, "seconds": round(secs, 2),
                      "b": gb})
    return {"o": o, "path": path, "vol": vol, "gates": gates, "wall": p, "size": size}


def through(vol, g, size):
    """Outside to inside on foot, no jump: `test_round_walls.gate_walk`."""
    return T.gate_walk(vol, g, size)


def passage_clear(vol, g, part):
    """The passage's clear width and height at the gate's centre line: the run of
    columns across the road open from the floor to six up, and the air over the
    centre."""
    fx, fz = DIRS[g["facing"]]
    F = int(part["floor_y"])
    ax, az = g["x"], g["z"]
    vx, vz = (0, 1) if fx else (1, 0)
    wid = 0
    for k in range(-8, 9):
        x, z = ax + vx * k, az + vz * k
        if all(not T._solid(vol, x, y, z) for y in range(F + 1, F + 4)):
            wid += 1
    h = 0
    while not T._solid(vol, ax, F + 1 + h, az) and h < 60:
        h += 1
    return wid, h


def along_walk(vol, path, g, walk_y, pad):
    """The wall's walk across the gate: from a spine cell past one end of the pad to
    one past the other, on foot, no jump, with the ground under the walk taken away."""
    cells = boundary.polyline_cells(path)
    fx, fz = DIRS[g["facing"]]
    ax, az = g["x"], g["z"]

    def along(c):
        return (c[1] - az) if fx else (c[0] - ax)

    def across(c):
        return (c[0] - ax) if fx else (c[1] - az)
    reach = pad // 2 + 4
    ends = []
    for s in (-1, 1):
        cand = [c for c in cells if along(c) * s >= reach and along(c) * s <= reach + 3
                and abs(across(c)) <= reach]
        if not cand:
            return None, "no spine cell past the pad"
        ends.append(min(cand, key=lambda c: (abs(across(c)), abs(along(c)))))
    x0 = min(e[0] for e in ends) - 12
    z0 = min(e[1] for e in ends) - 12
    x1 = max(e[0] for e in ends) + 12
    z1 = max(e[1] for e in ends) + 12
    sub = vol.sub(x0, z0, x1 - x0 + 1, z1 - z0 + 1)
    codes = sub.codes.copy()
    cut = walk_y - 3 - sub.y0
    codes[:, :max(0, cut), :] = 0
    v = Volume(sub.x0, sub.y0, sub.z0, codes, list(sub.palette))
    nav = observe.Nav(v)
    st = []
    for (x, z) in ends:
        top = T.top_solid(v, x, z, y_hi=walk_y + 1)
        s = nav.stance_near(x, z, (top or walk_y) + 1, tol=3)
        if s is None:
            return None, f"no stance on the walk at {(x, z)}"
        st.append((x, z, s))
    route = nav.route([st[0]], st[1], max_jumps=0)
    return route, f"{ends[0]} -> {ends[1]}"


def gate_lint(vol, gr):
    """The build family's E-codes against the gate's own plot, on the world the wall
    and the gate left."""
    from ethoslm import lint
    sp, b = gr["part"], gr["b"]
    plot = {"label": sp["label"], "x0": sp["x0"], "z0": sp["z0"], "x1": sp["x1"],
            "z1": sp["z1"]}
    region = (sp["x0"] - 6, sp["z0"] - 6, sp["x1"] + 6, sp["z1"] + 6)
    ctx = lint.Context.build(vol, plots=[plot], region=region, fittings=b.fitting_cells)
    rep = pipeline.standard_report(ctx, [plot])
    return [f"{f.code} {f.message[:100]}" for f in rep.findings if f.code.startswith("E")]


WALLS = (("wall", 10, 3), ("wall", 10, 9), ("great_wall", 44, 3), ("great_wall", 44, 9))
BEARINGS = (90.0, 45.0)


@case
def g0_preflight():
    from ethoslm.lint import preflight
    src = open(os.path.join(ROOT, "types", "ring_gate.py")).read()
    r = preflight(src, forbid=pipeline.TYPE_FORBIDDEN, palette=True)
    assert r.ok, r.to_json()
    return "ring_gate ok"


@case
def g1_walked_through_and_walked_over():
    out = []
    worst = 0.0
    for wt, H, w in WALLS:
        r = build_ring_with_gates(wt, H, w, BEARINGS, {"storeys": 2, "crown": "hip"})
        for gr in r["gates"]:
            g, sp, res = gr["g"], gr["part"], gr["res"]
            tag = f"{wt} h{H} w{w} bearing {g['bearing']:.0f} ({g['facing']})"
            route = through(r["vol"], g, r["size"])
            assert route, f"{tag}: cannot be walked through: {res}"
            wid, hh = passage_clear(r["vol"], g, sp)
            assert wid >= res["passage"] and res["passage"] >= 5, (tag, wid, res)
            assert hh >= 6, (tag, hh, res)
            wr, why = along_walk(r["vol"], r["path"], g, res["walk"][0], r["size"])
            assert wr, f"{tag}: the wall's walk does not cross the gate ({why}): {res}"
            errs = gate_lint(r["vol"], gr)
            assert not errs, f"{tag}: lint errors on the gate's plot: {errs[:3]}"
            worst = max(worst, gr["seconds"])
            assert gr["seconds"] < 5.0, (tag, gr["seconds"])
            out.append(f"{tag}: pad {r['size']}, passage {res['passage']} (clear {wid}) x "
                       f"{res['arch']} high, deck {res['deck'] - int(sp['floor_y'])}, "
                       f"{res['tiers']} eaves, through in {len(route)} steps, walk "
                       f"across in {len(wr)}, stair {res['stairs'][0].get('ok')}, "
                       f"{gr['seconds']} s")
        if SHOW:
            for gr in r["gates"]:
                show(r["vol"], gr["g"], gr["part"], gr["res"],
                     f"{wt}_h{H}_w{w}_b{int(gr['g']['bearing'])}")
    return f"slowest gate {worst:.2f} s\n    " + "\n    ".join(out)


@case
def g2_rank_shows():
    out = []
    # a great wall: three eaves and the wide passage whatever `storeys` says
    r = build_ring_with_gates("great_wall", 44, 7, (90.0,), {"storeys": 1, "crown": "hip"})
    res = r["gates"][0]["res"]
    assert res["tiers"] == 3 and res["passage"] >= 7, res
    out.append(f"great wall 44: {res['tiers']} eaves, passage {res['passage']}, arch "
               f"{res['arch']}")
    # storeys are eave tiers on a town wall, and the crown is the crown asked
    tops = {}
    for st in (1, 2, 3):
        r = build_ring_with_gates("wall", 12, 3, (90.0,), {"storeys": st,
                                                           "crown": "pavilion"})
        res = r["gates"][0]["res"]
        assert res["tiers"] == st, (st, res)
        tops[st] = res["ridge"]
    assert tops[1] < tops[2] < tops[3], tops
    out.append(f"wall 12, storeys 1/2/3: ridges {tops[1]}/{tops[2]}/{tops[3]}")
    # a precinct wall's tower is smaller than a town wall's
    small = build_ring_with_gates("wall", 9, 3, (90.0,), {"storeys": 3, "crown": "pavilion"})
    big = build_ring_with_gates("wall", 12, 3, (90.0,), {"storeys": 3, "crown": "pavilion"})
    s_res, b_res = small["gates"][0]["res"], big["gates"][0]["res"]
    assert s_res["ridge"] - s_res["deck"] < b_res["ridge"] - b_res["deck"], (s_res, b_res)
    out.append(f"palace wall 9: tower {s_res['ridge'] - s_res['deck']} over its deck, "
               f"town wall 12: {b_res['ridge'] - b_res['deck']}")
    # the axis road through it, eleven wide: the passage is the road's width. The city's
    # `gate_middle_ring_180`: a twelve-high town wall three wide, the gate on the south
    # axis facing south, storeys 2, a hip crown, in its own voice.
    r = build_ring_with_gates("wall", 12, 3, (180.0,), {"storeys": 2, "crown": "hip"},
                              road=11)
    res = r["gates"][0]["res"]
    wid, hh = passage_clear(r["vol"], r["gates"][0]["g"], r["gates"][0]["part"])
    assert res["passage"] >= 11 and wid >= 11, (res, wid)
    assert through(r["vol"], r["gates"][0]["g"], r["size"]), res
    out.append(f"an 11-wide road through a 12 wall: passage {res['passage']}, {wid} "
               f"clear, {hh} high")
    if SHOW:
        gr = r["gates"][0]
        show(r["vol"], gr["g"], gr["part"], gr["res"], "wall_h12_road11_b180")
    # a param widens it
    r = build_ring_with_gates("wall", 12, 3, (90.0,), {"storeys": 2, "crown": "hip",
                                                       "passage": 9})
    res = r["gates"][0]["res"]
    wid, hh = passage_clear(r["vol"], r["gates"][0]["g"], r["gates"][0]["part"])
    assert res["passage"] == 9 and wid >= 9, (res, wid)
    out.append(f"passage=9 on a 12 wall: {wid} clear, {hh} high")
    return "; ".join(out)


@case
def g3_no_lint_error_on_flat_ground():
    import type_needs as TN
    decl = gate_decl()
    lo, hi = decl["NEEDS"]["footprint"][0], decl["NEEDS"]["footprint"][2]
    bad, n = [], 0
    for size in range(lo, hi + 1):
        for ground in ("flat", "slope"):
            for st in (1, 2, 3):
                for crown in ("gable", "hip", "pavilion"):
                    for passage in (3, 7, 11):
                        seed = 1 + (st + passage) % 2
                        r = TN.instance("ring_gate", "point", (size, size), seed,
                                        {"storeys": st, "crown": crown,
                                         "passage": passage}, ground=ground)
                        n += 1
                        if not r["pass"]:
                            bad.append((size, ground, st, crown, passage, r["why"]))
    assert not bad, f"{len(bad)} of {n}: {bad[:3]}"
    return f"{n} instances at pads {lo}-{hi} on the plane and the bank, no E-code"


# ------------------------------------------------------------------ pictures

def show(vol, g, part, res, tag):
    import ca_eye
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    fx, fz = DIRS[g["facing"]]
    F = int(part["floor_y"])
    top = int(res.get("ridge") or res["deck"] + 10)
    tall = top - F
    d = max(36, int(tall * 1.6))
    look = (g["x"] + 0.5, F + tall * 0.45, g["z"] + 0.5)
    views = {
        "outside": ((g["x"] + fx * d + 0.5, F + 2.6, g["z"] + fz * d + 0.5), look),
        "oblique": ((g["x"] + fx * d * 0.8 + fz * d * 0.6 + 0.5, F + tall * 0.9,
                     g["z"] + fz * d * 0.8 + fx * d * 0.6 + 0.5), look),
        "inside": ((g["x"] - fx * d * 0.8 - fz * 4 + 0.5, F + 2.6,
                    g["z"] - fz * d * 0.8 - fx * 4 + 0.5), look),
        "passage": ((g["x"] + fx * 14 + 0.5, F + 2.6, g["z"] + fz * 14 + 0.5),
                    (g["x"] - fx * 10 + 0.5, F + 3, g["z"] - fz * 10 + 0.5)),
    }
    for name, (at, lk) in views.items():
        img = ca_eye.render(vol, at, lk, size=(960, 540), fov=70, far=400)
        Image.fromarray(img).save(os.path.join(OUT, f"{tag}_{name}.png"))


def main() -> int:
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
