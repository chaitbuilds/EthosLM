#!/usr/bin/env python3
"""**The ceremonial hall realizes hierarchy, measurably.**

    $PY scripts/test_ceremonial_hall.py [name-fragment] [--needs]

Offline, no cached world, no model call. Every instance is sited and built through
`Builder.site()` / `type_builder()` on flat synthetic ground, the way
`scripts/test_attached_forms.py` stands a lot (the pad inset by siting, the door in the
middle of the pad's street edge, the landing on a lane), and read back off the emitted
geometry, never off the parameters that were handed in.

    c1  storeys 1/2/3 on one principal footprint: `construction.measure` counts 1, 2, 3
        floors, the height rises with every storey, and the eave tiers counted off the
        blocks on the axis are storeys - 1 + eaves
    c2  a 3-storey principal hall dominates a 1-storey side hall on the same pad
    c3  a gate hall is walked straight through on the axis: from the doorstep, up the
        front flight, through the centre opening and down the back flight to the back
        of the pad, on foot, inside a strip five columns wide round the axis
    c4  platform 0..3 stands that many stepped tiers, read off the ground-to-top
        profile beside the flight
    c5  every use, both test voices, all four fronts: the build family of the linter
        reports no E-code on the plot, and no block the type wrote lies outside its pad
    c6  preflight (E012-E015) is clean on the file; the same seed builds the same blocks;
        another seed keeps the hierarchy (heights, floors, eave tiers)
    c7  a 60x40 principal hall (3 storeys, double eave, 3 tiers) builds in well under a
        minute

`--needs` sweeps pad sizes from 7x5 to 64x48 over every use and prints the sizes at
which any instance refuses or lints dirty (the NEEDS measurement).
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import test_attached_forms as taf  # noqa: E402
from ethoslm import construction, lint, pipeline  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

TYPE = "ceremonial_hall"
VOICES = ("vermilion_and_yellow_glaze", "ochre_stone_green_tile")
INSET = 2                                  # siting's pad inset on a free lot, each side

# a taller and wider world than the attached-forms rows: a 64x48 pad with three tiers,
# three storeys and four eaves stands ~45 blocks
taf.SIZE = 112


def _volume(h: np.ndarray) -> Volume:
    y0 = taf.GROUND - 14
    codes = np.zeros((taf.SIZE, 100, taf.SIZE), np.uint16)
    g = int(h[0, 0]) - y0
    codes[:, :g, :] = 3
    codes[:, g, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


taf._volume = _volume
CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


def stand(pad_w: int, pad_d: int, params: dict, voice=VOICES[0], front="north", seed=3):
    """One instance on a pad of `pad_w` x `pad_d` (across the front x back from it)."""
    taf.VOICE = voice
    t0 = time.time()
    row = taf.build_row(TYPE, [pad_w + 2 * INSET], pad_d + 2 * INSET, "flat", [params],
                        seeds=[seed], front=front)
    row["seconds"] = time.time() - t0
    return row


def emitted(row) -> dict:
    return ((row["results"][0] or {}).get("emitted") or {})


def _frame(row):
    """(f(u, v) -> (x, z), W, D, uc) of the part, the front at v=0, from its door."""
    p = row["parts"][0]
    x0, z0, x1, z1 = p["x0"], p["z0"], p["x1"], p["z1"]
    front = emitted(row).get("front") or row["front"]
    if front == "north":
        return (lambda u, v: (x0 + u, z0 + v)), x1 - x0 + 1, z1 - z0 + 1
    if front == "south":
        return (lambda u, v: (x1 - u, z1 - v)), x1 - x0 + 1, z1 - z0 + 1
    if front == "west":
        return (lambda u, v: (x0 + v, z1 - u)), z1 - z0 + 1, x1 - x0 + 1
    return (lambda u, v: (x1 - v, z0 + u)), z1 - z0 + 1, x1 - x0 + 1


def _axis_u(row, f, W) -> int:
    dx, dz = tuple(row["parts"][0]["door"])[0], tuple(row["parts"][0]["door"])[-1]
    for u in range(W):
        if f(u, 0) == (int(dx), int(dz)):
            return u
    return W // 2


def _name(s: str) -> str:
    return s.split("[")[0].split(":")[-1]


def roof_names(row) -> set:
    b = row["b"]
    fam = pipeline.voice_palette(taf.VOICE)["roof"]
    return {_name(b.block(fam, k)) for k in ("full", "stairs", "slab")}


def eave_tiers(row) -> int:
    """Eave tiers counted off the blocks: up the axis column, the frontmost roof block of
    each course; a tier starts wherever that front jumps outward again."""
    vol = row["built"]
    f, W, D = _frame(row)
    uc = _axis_u(row, f, W) + 1                  # one off the axis: clear of ridge ends
    names = roof_names(row)
    fy = row["parts"][0]["floor_y"]
    tiers, prev = 0, None
    for y in range(fy + 1, fy + 90):
        front = None
        for v in range(D):
            x, z = f(uc, v)
            if vol.inside(x, y, z) and _name(vol.state(x, y, z)) in names:
                front = v
                break
        if front is not None and (prev is None or front < prev):
            tiers += 1
        prev = front
    return tiers


def top_y(row) -> int:
    vol = row["built"]
    p = row["parts"][0]
    best = p["floor_y"]
    for x in range(p["x0"], p["x1"] + 1):
        for z in range(p["z0"], p["z1"] + 1):
            col = vol.codes[x - vol.x0, :, z - vol.z0]
            nz = np.nonzero(col)[0]
            if len(nz):
                best = max(best, vol.y0 + int(nz[-1]))
    return best


def outside_pad(row) -> list:
    """Non-air blocks the type's own build() wrote outside the pad it was handed."""
    p = row["parts"][0]
    out = []
    # site()'s own approach treads -- the way in, off the pad in front of the door at
    # ground level -- are the library's and allowed, whichever call resolved them (a
    # flight() inside build() resolves the queue site() left)
    dx, dz = tuple(p["door"])[0], tuple(p["door"])[-1]
    for (x, y, z), (was, st) in row["writes"][0].items():
        if p["x0"] <= x <= p["x1"] and p["z0"] <= z <= p["z1"]:
            continue
        if _name(st) in taf.AIRS:
            continue
        if abs(x - dx) + abs(z - dz) <= 3 and y <= p["floor_y"] + 1 \
                and _name(st).endswith(("_stairs", "_slab")):
            continue
        out.append(((x, y, z), _name(st)))
    q = row["b"]._step_queue() if hasattr(row["b"], "_step_queue") else []
    for t in q:
        if abs(t["x"] - dx) + abs(t["z"] - dz) <= 3 and t["y"] <= p["floor_y"] + 1:
            continue
        if not (p["x0"] <= t["x"] <= p["x1"] and p["z0"] <= t["z"] <= p["z1"]):
            out.append(((t["x"], t["y"], t["z"]), "tread"))
    return out


def clean(row, tag):
    assert not row["refused"], (tag, row["refused"])
    errs = taf.lint_errors(row)
    assert not errs, (tag, [f"{e.code} {e.message}" for e in errs[:5]])
    strays = outside_pad(row)
    assert not strays, (tag, f"{len(strays)} block(s) outside the pad", strays[:6])


# ------------------------------------------------------------------ the cases

PRINCIPAL = (52, 34)


@case
def c1_storeys_are_realized_and_counted():
    got = []
    for eaves in (1, 2):
        heights = []
        for s in (1, 2, 3):
            row = stand(*PRINCIPAL, {"use": "principal", "roof": "hip", "storeys": s,
                                     "eaves": eaves, "platform": 3})
            clean(row, f"principal s{s} e{eaves}")
            m = construction.measure(row["b"], row["parts"][0])
            et = eave_tiers(row)
            em = emitted(row)
            assert m["storeys"] == s, ("measured storeys", s, m["storeys"], m["levels"])
            assert em["storeys"] == s and em["eave_tiers"] == s - 1 + eaves, em
            assert et == s - 1 + eaves, ("eave tiers off the blocks", s, eaves, et)
            heights.append(top_y(row) - row["parts"][0]["floor_y"])
            got.append(f"s{s}e{eaves}:{m['storeys']}fl/{et}eaves/h{heights[-1]}")
        assert heights[0] < heights[1] < heights[2], ("height must rise", heights)
    return "; ".join(got)


@case
def c2_principal_dominates_a_side_hall():
    big = stand(*PRINCIPAL, {"use": "principal", "roof": "hip", "storeys": 3,
                             "eaves": 2, "platform": 3})
    small = stand(*PRINCIPAL, {"use": "side", "roof": "hip_gable", "storeys": 1,
                               "eaves": 1, "platform": 1})
    clean(big, "principal")
    clean(small, "side")
    hb = top_y(big) - big["parts"][0]["floor_y"]
    hs = top_y(small) - small["parts"][0]["floor_y"]
    fb, fs = emitted(big)["floors"][0], emitted(small)["floors"][0]
    assert hb >= 1.6 * hs, ("principal", hb, "side", hs)
    assert fb > fs, ("principal floor stands higher", fb, fs)
    return f"principal {hb} high, floor +{fb - big['parts'][0]['floor_y']}; " \
           f"side {hs} high, floor +{fs - small['parts'][0]['floor_y']}"


def _walk_through(row) -> tuple:
    """(reached, n) -- is the back of the pad on the axis reached on foot from the
    doorstep, confined to five columns round the axis?"""
    plots = [{"label": lf["label"], "x0": lf["x0"], "z0": lf["z0"], "x1": lf["x1"],
              "z1": lf["z1"]} for lf in row["leaves"]]
    p = row["parts"][0]
    ctx = lint.Context.build(row["built"], plots=plots,
                             region=(p["x0"] - 6, p["z0"] - 6, p["x1"] + 6, p["z1"] + 6),
                             network=row["net"], fittings=row["b"].fitting_cells)
    f, W, D = _frame(row)
    uc = _axis_u(row, f, W)
    a, c = f(uc - 2, 0), f(uc + 2, D - 1)
    bounds = (min(a[0], c[0]), min(a[1], c[1]), max(a[0], c[0]), max(a[1], c[1]))
    fy = p["floor_y"]
    x, z = f(uc, 0)
    s = ctx.nav.stance_near(x, z, fy + 1, tol=2)
    assert s is not None, "no stance on the doorstep"
    got = ctx.nav.flood([(x, z, s)], max_jumps=0, bounds=bounds)
    bx, bz = f(uc, D - 1)
    reached = any((k[0], k[1]) == (bx, bz) for k in got)
    # and the axis is walked over the hall's floor, not round it
    F1 = emitted(row)["floors"][0]
    cv = (emitted(row)["rects"]["colonnade"])
    mx, mz = f(uc, D // 2)
    on_floor = any((k[0], k[1]) == (mx, mz) and k[2] >= F1 for k in got)
    return reached and on_floor, len(got)


@case
def c3_gate_is_walked_through_on_the_axis():
    said = []
    for (w, d, P, front) in ((30, 16, 2, "north"), (24, 14, 1, "east"), (9, 6, 1, "south"),
                             (11, 8, 0, "west"), (34, 18, 3, "north")):
        row = stand(w, d, {"use": "gate", "roof": "hip_gable", "storeys": 1, "eaves": 1,
                           "platform": P}, front=front)
        clean(row, f"gate {w}x{d}")
        ok, n = _walk_through(row)
        assert ok, (f"gate {w}x{d} P{P} {front}: the back of the pad is not reached on "
                    f"the axis on foot", n)
        said.append(f"{w}x{d}/P{P}/{front}: through, openings "
                    f"{emitted(row)['features']['gate_openings']}")
    return "; ".join(said)


def _tier_steps(row) -> int:
    """Stepped courses under the hall, read beside the flight: the top of the solid
    column at each v from the front, counting the rises before the hall floor."""
    vol = row["built"]
    f, W, D = _frame(row)
    uc = _axis_u(row, f, W)
    fy = row["parts"][0]["floor_y"]
    u = uc + emitted(row)["features"].get("stair_half", 0) + 6
    F1 = emitted(row)["floors"][0]
    rises, prev = 0, fy
    for v in range(D // 2):
        x, z = f(u, v)
        t = fy
        for y in range(F1, fy, -1):
            n = _name(vol.state(x, y, z))
            if n not in taf.AIRS and not n.endswith("_wall"):
                t = y
                break
        if t > prev:
            rises += 1
            prev = t
        if t >= F1:
            break
    return rises


@case
def c4_platform_tiers_are_stood():
    got = []
    for P in (0, 1, 2, 3):
        row = stand(40, 26, {"use": "rear", "roof": "hip_gable", "storeys": 1, "eaves": 1,
                             "platform": P})
        clean(row, f"platform {P}")
        n = _tier_steps(row)
        em = emitted(row)
        assert em["platform"] == P and n == P, ("tiers", P, n, em["platform"])
        if P >= 2:
            assert em["features"]["balustrade"], "no balustrade on a high terrace"
        got.append(f"P{P}:{n} tiers")
    return ", ".join(got)


USE_CASES = [
    ("principal", 52, 34, {"roof": "hip", "storeys": 3, "eaves": 2, "platform": 3}),
    ("principal", 44, 28, {"roof": "hip", "storeys": 1, "eaves": 2, "platform": 3}),
    ("gate", 30, 16, {"roof": "hip_gable", "storeys": 1, "eaves": 2, "platform": 2}),
    ("gate", 22, 13, {"roof": "hip_gable", "storeys": 2, "eaves": 1, "platform": 1}),
    ("rear", 38, 20, {"roof": "hip_gable", "storeys": 2, "eaves": 1, "platform": 2}),
    ("side", 21, 11, {"roof": "hip_gable", "storeys": 1, "eaves": 1, "platform": 1}),
    ("side", 16, 9, {"roof": "gable", "storeys": 2, "eaves": 2, "platform": 0}),
    ("pavilion", 11, 11, {"roof": "pavilion", "storeys": 1, "eaves": 2, "platform": 1}),
    ("pavilion", 7, 7, {"roof": "pavilion", "storeys": 1, "eaves": 1, "platform": 0}),
    ("gate", 9, 6, {"roof": "gable", "storeys": 1, "eaves": 1, "platform": 1}),
    ("gate", 7, 5, {"roof": "hip", "storeys": 1, "eaves": 1, "platform": 1}),
]


@case
def c5_every_use_lints_clean_inside_its_pad():
    n = 0
    fronts = ("north", "east", "south", "west")
    for vi, voice in enumerate(VOICES):
        for i, (use, w, d, p) in enumerate(USE_CASES):
            front = fronts[(i + vi) % 4]
            row = stand(w, d, dict(p, use=use), voice=voice, front=front, seed=5 + i)
            clean(row, f"{use} {w}x{d} {voice} {front}")
            n += 1
    return f"{n} instances, 0 E-codes, 0 blocks outside the pad"


@case
def c6_preflight_and_determinism():
    src = open(os.path.join(ROOT, "types", f"{TYPE}.py")).read()
    rep = lint.preflight(src, forbid=pipeline.TYPE_FORBIDDEN, palette=True)
    errs = [f for f in rep.findings if f.code.startswith("E")]
    assert not errs, [f"{f.code} {f.message}" for f in errs]
    p = {"use": "principal", "roof": "hip", "storeys": 2, "eaves": 2, "platform": 2}
    a, b_, c = stand(44, 28, p, seed=7), stand(44, 28, p, seed=7), stand(44, 28, p, seed=8)
    wa = {k: v[1] for k, v in a["writes"][0].items()}
    wb = {k: v[1] for k, v in b_["writes"][0].items()}
    assert wa == wb, "the same seed built different blocks"
    for key in ("floors", "eave_ys", "ridge_y", "eave_tiers", "platform"):
        assert emitted(a)[key] == emitted(c)[key], ("hierarchy moved with the seed", key)
    return "preflight clean; seed 7 twice identical; seed 8 keeps floors/eaves/ridge"


@case
def c7_a_sixty_by_forty_principal_is_quick():
    row = stand(60, 40, {"use": "principal", "roof": "hip", "storeys": 3, "eaves": 2,
                         "platform": 3})
    clean(row, "principal 60x40")
    assert row["seconds"] < 30, row["seconds"]
    return f"{row['seconds']:.1f}s site+build, {len(row['writes'][0])} cells written"


# ------------------------------------------------------------------ NEEDS

def needs_sweep() -> int:
    sizes = [(7, 5), (8, 6), (9, 6), (11, 8), (13, 9), (15, 11), (17, 12), (21, 13),
             (25, 16), (30, 18), (35, 22), (40, 26), (46, 30), (52, 34), (58, 40),
             (64, 48)]
    params = [
        ("principal", {"roof": "hip", "storeys": 3, "eaves": 2, "platform": 3}),
        ("principal", {"roof": "hip", "storeys": 1, "eaves": 1, "platform": 0}),
        ("gate", {"roof": "hip_gable", "storeys": 1, "eaves": 2, "platform": 2}),
        ("gate", {"roof": "gable", "storeys": 2, "eaves": 1, "platform": 1}),
        ("rear", {"roof": "hip_gable", "storeys": 2, "eaves": 1, "platform": 2}),
        ("side", {"roof": "gable", "storeys": 1, "eaves": 1, "platform": 1}),
        ("pavilion", {"roof": "pavilion", "storeys": 3, "eaves": 2, "platform": 3}),
    ]
    bad = []
    for (w, d) in sizes:
        worst = 0.0
        for use, p in params:
            for (ww, dd) in ((w, d), (d, w)) if w != d else ((w, d),):
                row = stand(ww, dd, dict(p, use=use))
                worst = max(worst, row["seconds"])
                errs = taf.lint_errors(row)
                if row["refused"] or errs or outside_pad(row):
                    bad.append(((ww, dd), use, row["refused"],
                                [e.code for e in errs][:4]))
        print(f"{w}x{d}: worst {worst:.1f}s", flush=True)
    for b in bad:
        print("DIRTY", b)
    print(f"{len(bad)} dirty instance(s)")
    return 1 if bad else 0


def main() -> int:
    if "--needs" in sys.argv:
        return needs_sweep()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    ok = fail = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__
        if only and only not in name:
            continue
        try:
            said = fn()
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        except Exception as e:                   # noqa: BLE001 -- reported
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} ceremonial hall cases pass ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
