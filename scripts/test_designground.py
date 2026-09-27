#!/usr/bin/env python3
"""**Designed ground**: `ethoslm.designground` turns a designed level and treatment per
column into finished ground, with its costs and bounds on the record.

    $PY scripts/test_designground.py [name-fragment] [--show]

Offline, on synthetic volumes (a heightmap of stone under dirt under grass, water where
asked, a tree where asked). Each case has its positive control -- the thing that must
happen -- beside the thing that must not:

    d1  a flat site under KEEP is untouched, block for block
    d2  a platform on a slope: a faced retaining wall on its low side, a faced cut on
        its high side, a cap on both, no raw face measured, the tree over it felled
        with its leaves; cut and fill on the record as `estimate` predicted them
    d3  a SOFT band between two terraces never steps more than one per `soft_grade`
        columns, is grass all over, and reports nothing unresolved -- and a band too
        narrow for the rise reports its steps (and faces them)
    d4  a street across two terraces reports its step; the same street with a ramp band
        in its target resolves into slab ramps a person walks up without a jump
    d5  a platform over a pond: the platform's edge on the water is a bank (quay) from
        the pond's bed up, the rest of the pond keeps its water, the pond's far shore
        (KEEP) is untouched, and no water is left inside the platform
    d6  a platform sixteen above its ground is over `max_retain`: reported, and faced
    d7  applied in two halves by an `only` mask, in either order, the ground is the
        ground applied at once -- a seam comes out the same whichever side is built
        first (faces belong to the higher column, ramps to the lower)
    d8  `estimate` is cheap and agrees with `plan`'s record
    d9  handed the ground as found (`base`), a neighbour's build standing over these
        columns is left standing through the cut

`--show` writes each case's `preview()` and an oblique render to `out/ds-work/ground/`.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from ethoslm import designground as DG  # noqa: E402
from ethoslm import observe  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

OUT = os.path.join(ROOT, "out", "ds-work", "ground")
SHOW = "--show" in sys.argv
MATS = {"fill": "dirt", "cover": "grass_block", "retain": "stone_bricks",
        "cap": "stone_brick_slab", "bank": "stone_bricks", "path": "dirt_path",
        "pave": {1: "polished_andesite", 2: "packed_mud", 3: "smooth_stone"}}
Y0 = 40
HGT = 70
CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


def volume(h: np.ndarray, *, water=None, trees=()) -> Volume:
    """Stone under three of dirt under grass at `h` (top solid y) per column; `water`
    a `[w, h]` int array of water surface y (or -1) over the bed; `trees` (x, z)."""
    w, d = h.shape
    pal = ["air", "grass_block", "dirt", "stone", "water", "oak_log",
           "oak_leaves[distance=1,persistent=false]", "sand"]
    codes = np.zeros((w, HGT, d), np.uint16)
    ys = np.arange(HGT)[None, :, None] + Y0
    hh = h[:, None, :]
    codes[ys < hh - 3] = 3
    codes[(ys >= hh - 3) & (ys < hh)] = 2
    codes[ys == hh] = 1
    if water is not None:
        ws = water[:, None, :]
        wetc = (water >= 0)[:, None, :] & (ys > hh) & (ys <= ws)
        codes[wetc] = 4
        codes[(water >= 0)[:, None, :] & (ys == hh)] = 7         # a sand bed
    for (x, z) in trees:
        g = int(h[x, z])
        codes[x, g + 1 - Y0: g + 6 - Y0, z] = 5
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in (4, 5, 6):
                    xx, zz, yy = x + dx, z + dz, g + dy - Y0
                    if 0 <= xx < w and 0 <= zz < d and codes[xx, yy, zz] == 0:
                        codes[xx, yy, zz] = 6
    return Volume(0, Y0, 0, codes, pal)


def names(vol, x, z):
    return [vol.palette[int(c)].split("[")[0] for c in vol.codes[x, :, z]]


def top(vol, x, z):
    col = names(vol, x, z)
    for i in range(len(col) - 1, -1, -1):
        if col[i] not in ("air", "water") and not col[i].endswith(("_leaves", "_log")):
            return i + Y0
    return None


def at(vol, x, y, z):
    return vol.palette[int(vol.codes[x, y - Y0, z])].split("[")[0]


def show(tag, gp, vol2):
    if not SHOW:
        return
    os.makedirs(OUT, exist_ok=True)
    DG.preview(gp, os.path.join(OUT, f"{tag}_plan.png"))
    import ca_eye
    from PIL import Image
    w, _, d = vol2.codes.shape
    img = ca_eye.render(vol2, (-w * 0.35, 64 + w * 0.55, d * 1.25), (w / 2, 62, d / 2),
                        size=(960, 540), fov=55, far=4 * w)
    Image.fromarray(img).save(os.path.join(OUT, f"{tag}_view.png"))


# ------------------------------------------------------------------ cases

@case
def d1_flat_keep_is_untouched():
    h = np.full((40, 40), 64, np.int32)
    vol = volume(h, trees=[(20, 20)])
    tg = np.full(h.shape, 64, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr)
    v2, rec = DG.apply(vol, 0, 0, gp, materials=MATS)
    same = np.array_equal(np.asarray([v2.palette[c] for c in v2.codes.ravel()[::1]]),
                          np.asarray([vol.palette[c] for c in vol.codes.ravel()]))
    assert same, "KEEP changed the ground"
    assert rec["cut_blocks_placed"] == 0 and rec["fill_blocks_placed"] == 0, rec
    # positive control: the same site under HARD one above is filled
    tr2 = np.full(h.shape, DG.HARD, np.uint8)
    gp2 = DG.plan(h, np.zeros(h.shape, bool), tg + 1, tr2)
    _, rec2 = DG.apply(vol, 0, 0, gp2, materials=MATS)
    # every column filled a course (the tree's column fills over its felled trunk)
    assert rec2["fill_blocks_placed"] >= 40 * 40 - 1, rec2["fill_blocks_placed"]
    return (f"KEEP: 0 blocks changed, tree standing; control HARD+1 filled "
            f"{rec2['fill_blocks_placed']} blocks")


@case
def d2_platform_on_a_slope_is_retained_and_cut():
    W = 60
    h = np.fromfunction(lambda x, z: 58 + x // 4, (W, 40)).astype(np.int32)   # 58..72
    vol = volume(h, trees=[(30, 20)])
    tg = np.full(h.shape, DG.NO_TARGET, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    tg[20:40, 10:30] = 65
    tr[20:40, 10:30] = DG.HARD
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr)
    est = DG.estimate(h, tg, tr)
    v2, rec = DG.apply(vol, 0, 0, gp, materials=MATS)
    # the low side (x=20): the platform's own column faced down to the ground outside
    low = [at(v2, 20, y, 20) for y in range(int(h[19, 20]) + 1, 66)]
    assert all(b == "stone_bricks" for b in low), f"low side face {low}"
    drop = 65 - int(h[19, 20])
    if drop > DG.PARAPET_OVER:
        assert at(v2, 20, 66, 20) == "stone_bricks", "no parapet over a drop of 3"
        assert at(v2, 20, 67, 20) == "stone_brick_slab", "no cap on the parapet"
    else:
        assert at(v2, 20, 66, 20) == "stone_brick_slab", "no cap on the retaining wall"
    # the high side (x=40): the KEEP column faced from the platform up to its own top
    hi = [at(v2, 40, y, 20) for y in range(66, int(h[40, 20]) + 1)]
    assert all(b == "stone_bricks" for b in hi), f"high side cut {hi}"
    ht = int(h[40, 20])
    cap_y = ht + 1 + (1 if ht - 65 > DG.PARAPET_OVER else 0)
    if cap_y > ht + 1:
        assert at(v2, 40, ht + 1, 20) == "stone_bricks", "no parapet on a cut of 3"
    assert at(v2, 40, cap_y, 20) == "stone_brick_slab", "no cap on the cut"
    assert top(v2, 30, 20) == 65 and at(v2, 30, 65, 20) == "grass_block"
    logs = sum(1 for x in range(W) for z in range(40) for n in names(v2, x, z)
               if n == "oak_log")
    leaves = sum(1 for x in range(W) for z in range(40) for n in names(v2, x, z)
                 if n == "oak_leaves")
    assert logs == 0 and leaves == 0, f"the felled tree left {logs} logs, {leaves} leaves"
    assert rec["raw_faces"] == 0 and rec["raw_faces_measured"] == 0, rec
    assert rec["cut_blocks"] == est["cut_blocks"] and rec["fill_blocks"] == est["fill_blocks"]
    # the planned cut and fill are the blocks the volume moved (the tree apart) (the
    # placed count adds the caps and parapet courses standing over the faces)
    assert rec["fill_blocks_placed"] == est["fill_blocks"] + rec["caps"] + rec["parapets"], \
        (rec["fill_blocks_placed"], est["fill_blocks"], rec["caps"], rec["parapets"])
    show("d2_platform", gp, v2)
    return (f"cut {rec['cut_blocks']} / fill {rec['fill_blocks']} blocks "
            f"(estimate {est['cut_blocks']} / {est['fill_blocks']}); retained "
            f"{rec['retained_length']} edges, {rec['retained_area']} face blocks, by "
            f"height {rec['retained_length_by_height']}; max {rec['max_retain']}; "
            f"parapets {rec['parapets']}; raw faces 0 (plan) / 0 (measured)")


@case
def d3_soft_band_grades_and_a_narrow_one_reports():
    out = []
    for grade in (1.0, 2.0):
        W = 60
        h = np.full((W, 30), 64, np.int32)
        vol = volume(h)
        tg = np.full(h.shape, 64, np.int16)
        tr = np.full(h.shape, DG.SOFT, np.uint8)
        tg[:15], tr[:15] = 64, DG.HARD
        tg[45:], tr[45:] = 72, DG.HARD
        tg[15:30] = 64
        tg[30:45] = 72                       # the SOFT band's target itself is a step
        gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr, soft_grade=grade)
        v2, rec = DG.apply(vol, 0, 0, gp, materials=MATS)
        f = gp.fin[:, 15]
        k = int(np.ceil(grade))
        worst = max(abs(int(f[x + k]) - int(f[x])) for x in range(14, 46 - k))
        assert worst <= 1, f"soft_grade {grade}: a step of {worst} over {k} columns: {f}"
        assert not gp.soft_unresolved, gp.soft_unresolved[:4]
        grassy = all(at(v2, x, top(v2, x, 15), 15) == "grass_block" for x in range(15, 45))
        assert grassy, "a SOFT column not covered"
        out.append(f"grade {grade}: steepest {worst} per {k} column(s), all grass")
    # the negative control: a band two wide for a rise of eight
    h = np.full((30, 10), 64, np.int32)
    tg = np.full(h.shape, 64, np.int16)
    tr = np.full(h.shape, DG.HARD, np.uint8)
    tg[16:] = 72
    tr[14:16] = DG.SOFT
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr)
    v2, rec = DG.apply(volume(h), 0, 0, gp, materials=MATS)
    assert rec["soft_unresolved"] > 0, "a band too narrow reported nothing"
    assert rec["raw_faces_measured"] == 0, rec["raw_faces_measured"]
    out.append(f"a 2-wide band for a rise of 8: {rec['soft_unresolved']} unresolved "
               f"steps reported, faced (raw 0)")
    return "; ".join(out)


def _walk(vol, a, b):
    nav = observe.Nav(vol)
    sa = nav.ground_stance(*a)
    sb = nav.ground_stance(*b)
    r = nav.route([(a[0], a[1], sa)], (b[0], b[1], sb), max_jumps=0)
    return r


@case
def d4_street_across_terraces_reports_its_step_unless_ramped():
    W = 70
    h = np.full((W, 21), 64, np.int32)
    h[35:] = 68
    vol = volume(h)
    tg = np.where(np.arange(W)[:, None] < 35, 64, 68).astype(np.int16) * np.ones((1, 21), np.int16)
    tr = np.full(h.shape, DG.HARD, np.uint8)
    pave = np.zeros(h.shape, np.int16)
    tr[:, 8:13] = DG.STREET
    pave[:, 8:13] = 1
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr, pave=pave)
    v2, rec = DG.apply(vol, 0, 0, gp, materials=MATS)
    assert rec["street_steps_unresolved"] == 5 and rec["max_unresolved_step"] == 4, rec
    assert _walk(v2, (5, 10), (65, 10)) is None, "a four-block street step walked"
    # the ramp band: the street's target climbs one block every two columns
    tg2 = tg.copy()
    for x in range(28, 36):
        tg2[x, 8:13] = 64 + (x - 27) // 2
    gp2 = DG.plan(h, np.zeros(h.shape, bool), tg2, tr, pave=pave)
    v3, rec2 = DG.apply(vol, 0, 0, gp2, materials=MATS)
    assert rec2["street_steps_unresolved"] == 0, rec2["street_steps_sample"]
    assert rec2["street_ramps"] > 0
    r = _walk(v3, (5, 10), (65, 10))
    assert r is not None, "the ramped street cannot be walked without a jump"
    assert at(v3, 30, 65, 10) == "polished_andesite", at(v3, 30, 65, 10)
    show("d4_street_ramped", gp2, v3)
    return (f"unramped: {rec['street_steps_unresolved']} steps of "
            f"{rec['max_unresolved_step']} reported, not walkable; ramped: 0 unresolved, "
            f"{rec2['street_ramps']} slab ramps, walked end to end in {len(r)} steps")


@case
def d5_platform_over_a_pond_banks_its_water():
    W = 50
    h = np.full((W, 40), 64, np.int32)
    water = np.full(h.shape, -1, np.int32)
    h[10:40, 10:30] = 60
    water[10:40, 10:30] = 63
    vol = volume(h, water=water)
    wet = water >= 0
    tg = np.full(h.shape, DG.NO_TARGET, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    tr[wet] = DG.WATER
    tg[5:25, 5:35] = 65
    tr[5:25, 5:35] = DG.HARD
    gp = DG.plan(h, wet, tg, tr)
    v2, rec = DG.apply(vol, 0, 0, gp, materials=MATS)
    # the platform's edge on the water (x=24, over the pond) is a bank from the bed up
    bank = [at(v2, 24, y, 20) for y in range(61, 66)]
    assert all(b == "stone_bricks" for b in bank), f"no quay: {bank}"
    # the rest of the pond keeps its water
    kept = sum(1 for x in range(25, 40) for z in range(10, 30)
               if at(v2, x, 63, z) == "water")
    assert kept == 15 * 20, f"the pond lost water: {kept} of 300"
    # the far shore (KEEP, x=40..) is untouched
    assert all(names(v2, x, 20) == names(vol, x, 20) for x in range(40, 50))
    # no water inside the platform
    inside = sum(1 for x in range(10, 24) for z in range(10, 30)
                 for n in names(v2, x, z) if n == "water")
    assert inside == 0, inside
    assert rec["raw_faces_measured"] == 0, rec["raw_faces_measured"]
    show("d5_pond", gp, v2)
    return (f"quay of stone from the bed (y=61) to the platform (65) on the water side; "
            f"300/300 water cells kept; far shore untouched; {rec['water_removed']} water "
            f"blocks filled under the platform")


@case
def d6_over_bound_retain_is_reported_and_faced():
    h = np.full((40, 40), 64, np.int32)
    tg = np.full(h.shape, DG.NO_TARGET, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    tg[10:30, 10:30] = 80
    tr[10:30, 10:30] = DG.HARD
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr, max_retain=12)
    v2, rec = DG.apply(volume(h), 0, 0, gp, materials=MATS)
    # the platform's 76 rim columns, each facing a drop of 16
    assert rec["over_bound_count"] == 76, rec["over_bound_count"]
    assert rec["raw_faces_measured"] == 0
    assert at(v2, 10, 80, 20) == "stone_bricks" and at(v2, 10, 81, 20) == "stone_bricks"
    assert at(v2, 10, 82, 20) == "stone_brick_slab", "no parapet and cap over a drop of 16"
    # the control: at 72 the same platform is within bound
    tg[10:30, 10:30] = 72
    rec2 = DG.estimate(h, tg, tr, max_retain=12)
    assert rec2["over_bound_count"] == 0
    return (f"16 over its ground: {rec['over_bound_count']} face columns over the bound "
            f"of 12 reported, all faced, parapet and cap; at 8: none")


@case
def d7_a_seam_applied_either_way_is_one_ground():
    W = 64
    h = np.fromfunction(lambda x, z: 60 + (x + z) // 6, (W, W)).astype(np.int32)
    vol = volume(h, trees=[(31, 20), (40, 40)])
    rng = np.random.default_rng(3)
    tg = np.full(h.shape, DG.NO_TARGET, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    tg[8:56, 8:56] = 66
    tr[8:56, 8:56] = DG.SOFT
    tg[20:44, 20:44] = rng.integers(64, 70)
    tr[20:44, 20:44] = DG.HARD
    tr[30:34, :] = DG.STREET
    tg[30:34, :] = np.clip(h[30:34, :], 62, 70)
    pave = np.zeros(h.shape, np.int16)
    pave[30:34, :] = 2
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr, pave=pave)
    whole, _ = DG.apply(vol, 0, 0, gp, materials=MATS)
    left = np.zeros(h.shape, bool)
    left[:32] = True                      # the seam runs through the street and platform
    a, _ = DG.apply(vol, 0, 0, gp, materials=MATS, only=left)
    a, _ = DG.apply(a, 0, 0, gp, materials=MATS, only=~left)
    b, _ = DG.apply(vol, 0, 0, gp, materials=MATS, only=~left)
    b, _ = DG.apply(b, 0, 0, gp, materials=MATS, only=left)

    def states(v):
        return np.asarray(v.palette, object)[v.codes]
    sw, sa, sb = states(whole), states(a), states(b)
    da, db = int((sw != sa).sum()), int((sw != sb).sum())
    assert da == 0 and db == 0, f"the seam differs: {da} / {db} cells"
    return (f"left-then-right and right-then-left both equal the whole, cell for cell "
            f"({sw.size} cells); faces {len(gp.faces)}, ramps {int(gp.ramp.sum())}")


@case
def d9_a_neighbours_build_over_these_columns_is_left_standing():
    """Construction is per region, so a neighbour's wall may already stand over
    columns this region grades. Handed the ground as found (`base`), `apply` cuts,
    fills and faces round what is not the found ground and leaves it standing."""
    h = np.full((30, 30), 64, np.int32)
    vol = volume(h)
    base_codes = vol.codes.copy()
    # a neighbour's pier of stone bricks standing on these columns, 64..75
    pal = list(vol.palette) + ["stone_bricks"]
    codes = vol.codes.copy()
    codes[10, 65 - Y0: 76 - Y0, 10] = len(pal) - 1
    ctx = Volume(0, Y0, 0, codes, pal)
    tg = np.full(h.shape, 60, np.int16)
    tr = np.full(h.shape, DG.HARD, np.uint8)
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr)
    cut, rec = DG.apply(ctx, 0, 0, gp, materials=MATS, base=base_codes)
    kept = [at(cut, 10, y, 10) for y in range(65, 76)]
    assert all(b == "stone_bricks" for b in kept), kept
    under = [at(cut, 10, y, 10) for y in range(61, 65)]
    assert "air" not in under, f"the pier was undermined: {under}"
    assert at(cut, 11, 62, 10) == "air" and at(cut, 11, 60, 10) == "grass_block"
    bare, rec2 = DG.apply(ctx, 0, 0, gp, materials=MATS)
    assert at(bare, 10, 70, 10) == "air", "the control: without base the pier is cut"
    return (f"the pier stands through a cut of 4 round it ({rec['built_blocks_left_standing']}"
            f" blocks left standing, its ground under it kept); without `base` the cut "
            f"takes it")


@case
def d8_estimate_is_cheap_and_agrees():
    W = 600
    xs = np.arange(W)[:, None]
    zs = np.arange(W)[None, :]
    h = (64 + 10 * np.sin(xs / 37.0) + 8 * np.cos(zs / 53.0)).astype(np.int32)
    tg = np.full(h.shape, DG.NO_TARGET, np.int16)
    tr = np.full(h.shape, DG.KEEP, np.uint8)
    r = np.hypot(xs - 300, zs - 300)
    tg[r < 250] = 66
    tr[r < 250] = DG.SOFT
    tg[r < 150] = 70
    tr[r < 150] = DG.HARD
    tr[(r < 250) & (np.abs(xs - 300) < 3)] = DG.STREET
    t0 = time.perf_counter()
    est = DG.estimate(h, tg, tr)
    dt = time.perf_counter() - t0
    gp = DG.plan(h, np.zeros(h.shape, bool), tg, tr)
    s = gp.summary()
    for k in ("cut_blocks", "fill_blocks", "retained_area", "street_steps_unresolved"):
        assert est[k] == s[k], (k, est[k], s[k])
    assert dt < 60, dt
    if SHOW:
        os.makedirs(OUT, exist_ok=True)
        DG.preview(gp, os.path.join(OUT, "d8_estimate_plan.png"))
    return (f"{W}x{W} columns estimated in {dt:.1f}s: cut {est['cut_blocks']:,}, fill "
            f"{est['fill_blocks']:,}, retained {est['retained_length']} edges / "
            f"{est['retained_area']} blocks (max {est['max_retain']}), "
            f"{est['street_steps_unresolved']} street steps (max "
            f"{est['max_unresolved_step']}), soft unresolved {est['soft_unresolved']}")


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
