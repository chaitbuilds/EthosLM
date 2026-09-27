"""**Designed ground**: a level and a treatment per column, turned into finished ground.

The design compiler (`cityresolve`) decides, for every column of a region, a designed
level (`target`, the top solid y, or `NO_TARGET`) and a treatment. This module turns
that into ground a person would call finished -- terraces with faced retaining walls,
landscape graded in natural slopes, paved streets that are walkable across their own
level changes, banks where land meets water -- with its costs and bounds stated rather
than discovered in a picture.

    plan(found_h, found_wet, target, treatment, *, pave=None, water_surface=None,
         soft_grade=1.0, max_retain=12) -> GroundPlan
    estimate(found_h, target, treatment, *, found_wet=None, pave=None, soft_grade=1.0,
             max_retain=12) -> dict                     (no volume; for comparing designs)
    apply(vol, x0, z0, gp, *, materials, only=None, base=None) -> (vol2, record)
    preview(gp, path, *, scale=None) -> path            (a height-shaded map, faces drawn)
    wall_floors(path, width, target, x0, z0, found=None) -> [int per segment]

Arrays are indexed `[x - x0, z - z0]`. `found_h` is the top solid (bed) y of each column
as found, `found_wet` whether water stands on it.

Treatments and their rules
--------------------------
    KEEP 0    The ground as found stays, vegetation and all. Its edge against a changed
              neighbour is finished: where a changed column lies two or more below it,
              its exposed side is a faced **cut wall** (retain block, a cap on top).
    SOFT 1    Landscape: graded toward `target` in natural slopes. The finished level is
              the target clipped to the Lipschitz envelope of every fixed column round it
              (no step steeper than one block per `soft_grade` columns, a diagonal
              sqrt 2 as far) and then relaxed against its SOFT neighbours, so a SOFT band
              between two levels is a slope, not a step. Covered in the cover block; a
              cut SOFT column is re-covered, never left as raw stone or dirt. A step of
              two or more the band is too narrow to grade is `soft_unresolved` and faced.
    HARD 2    A built platform or terrace exactly at `target`. Every edge against lower
              ground is a faced **retaining wall** (retain block from the lower level up
              through the platform's top course; where the drop is two or more a cap
              slab on it, and where it is more than two a parapet course under the cap).
              An edge against higher KEEP ground is that KEEP column's cut wall.
    STREET 3  Paved at `target` in the `pave` code's block (the path block where none).
              Two adjacent STREET columns one apart are a **slab ramp** (a bottom slab on
              the lower one: a half step either way, walkable). More than one apart is an
              `unresolved street step` -- the compiler must ramp it (a target stepping
              one block at a time is a ramp band) -- and is faced meanwhile. A street's
              edge against lower non-street ground is retained like a platform's.
    WATER 4   Water kept at its found surface; the column is not touched. Where a changed
              land column meets it, that column's side is a **bank** (quay) of the bank
              block from the water's bed up, and where the land is lower than the water
              the bank is carried up to the waterline (a quay wall that holds it).
    WALL 5    Graded to `target` to receive a wall; the top is fill, since the wall type
              lays its own footing on it. Its edges are faced (no cap: the wall covers
              them).

A column with `NO_TARGET` is KEEP whatever its treatment says; WATER on a dry column is
KEEP.

Cut and fill. Every changed column is cleared above its finished level (terrain, trees
and water) and filled from its found bed up to it (fill, the top course by treatment),
so a filled column is solid down to the ground as found. Trees cut off by the clearing
lose their orphaned leaves (natural, non-persistent leaves more than four from any
standing log) in the columns being applied.

**Who owns a face.** Every transition is decided from the full `target` / `treatment`
arrays, so it comes out the same whichever side of a region seam is built first: a
retaining or cut face, its cap and its parapet belong to the **higher** column of the
pair; a street ramp's slab to the **lower** STREET column; a bank to the **land** column.
`apply(only=mask)` writes only the columns of `mask` and so builds exactly its share.

Bounds. A face taller than `max_retain` is still faced (never a raw cliff) and is
reported in `over_bound` so the compiler changes the design; nothing over the bound is
silent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NO_TARGET = -32768
KEEP, SOFT, HARD, STREET, WATER, WALL = 0, 1, 2, 3, 4, 5
NAMES = ("keep", "soft", "hard", "street", "water", "wall")

#: face kinds, per column
F_NONE, F_RETAIN, F_CUT, F_BANK = 0, 1, 2, 3
#: height bands a face's length and area are reported in
BANDS = ((1, 1), (2, 3), (4, 6), (7, 12), (13, 10 ** 6))
#: a face this many blocks tall or more carries a parapet course under its cap
PARAPET_OVER = 2
#: leaves further than this from any standing log, in a column being cleared, are
#: orphans of a felled tree
LEAF_REACH = 4

_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class GroundPlan:
    """What `plan` decided, per column (`[w, h]` arrays) and in summary."""
    found: np.ndarray            # int32 bed as found
    wet: np.ndarray              # bool water as found
    target: np.ndarray           # int32 designed level, NO_TARGET kept
    treat: np.ndarray            # uint8 effective treatment (NO_TARGET -> KEEP)
    fin: np.ndarray              # int32 finished top solid y
    changed: np.ndarray          # bool: this column is worked
    pave: np.ndarray             # int16 paving code (0 none)
    face: np.ndarray             # uint8 F_* kind of this column's face
    face_lo: np.ndarray          # int32 lowest y of the face (fin+1 = none)
    face_h: np.ndarray           # int32 tallest step this column faces
    cap: np.ndarray              # bool a cap slab on top
    parapet: np.ndarray          # bool a parapet course under the cap
    ramp: np.ndarray             # bool a street ramp slab on top
    bank_top: np.ndarray         # int32 the bank's top where a quay holds water
    water_surface: np.ndarray | None
    soft_grade: float = 1.0
    max_retain: int = 12
    street_steps: list = field(default_factory=list)     # (i, j, di, dj, step)
    soft_unresolved: list = field(default_factory=list)  # (i, j, step)
    over_bound: list = field(default_factory=list)       # (i, j, height)
    faces: list = field(default_factory=list)            # (i, j, di, dj, height, kind)

    @property
    def shape(self):
        return self.fin.shape

    def summary(self) -> dict:
        return _summary(self)


# ------------------------------------------------------------------------ helpers

def _shift(a: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """`out[i, j] = a[i + di, j + dj]`, `fill` off the edge."""
    out = np.full_like(a, fill)
    w, h = a.shape
    si = slice(max(0, -di), w - max(0, di))
    sj = slice(max(0, -dj), h - max(0, dj))
    ti = slice(max(0, di), w - max(0, -di))
    tj = slice(max(0, dj), h - max(0, -dj))
    out[si, sj] = a[ti, tj]
    return out


_N8 = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
       (-1, -1, math.sqrt(2)))


def _grade_soft(fin: np.ndarray, soft: np.ndarray, s: float, iters: int = 400):
    """SOFT columns graded: the target clipped to the envelope of the fixed columns
    (no steeper than `s` a column), then relaxed against its neighbours. Float."""
    f = fin.astype(np.float64)
    if not soft.any():
        return f
    fixed = ~soft
    big = 1e9
    up = np.where(fixed, f, big)
    lo = np.where(fixed, f, -big)
    for _ in range(iters):
        nu, nl = up.copy(), lo.copy()
        for di, dj, d in _N8:
            nu = np.minimum(nu, _shift(up, di, dj, big) + s * d)
            nl = np.maximum(nl, _shift(lo, di, dj, -big) - s * d)
        nu = np.where(fixed, f, nu)
        nl = np.where(fixed, f, nl)
        if np.array_equal(nu, up) and np.array_equal(nl, lo):
            break
        up, lo = nu, nl
    g = f.copy()
    ok = soft & (up < big / 2) & (lo > -big / 2)
    clip = np.where(lo <= up, np.clip(f, lo, up), (lo + up) / 2.0)
    g[ok] = clip[ok]
    # relax: a SOFT column within `s` of every neighbour, where it can be
    for _ in range(iters):
        mx = np.full_like(g, -big)
        mn = np.full_like(g, big)
        for di, dj, d in _N8:
            mx = np.maximum(mx, _shift(g, di, dj, -big) - s * d)
            mn = np.minimum(mn, _shift(g, di, dj, big) + s * d)
        want = np.where(mx <= mn, np.clip(g, mx, mn), (mx + mn) / 2.0)
        nxt = np.where(soft, want, g)
        if np.max(np.abs(nxt - g)) < 1e-3:
            g = nxt
            break
        g = nxt
    return g


def _effective(found, wet, target, treatment):
    tgt = np.asarray(target, np.int32)
    tr = np.asarray(treatment, np.uint8).copy()
    tr[tgt == NO_TARGET] = KEEP
    tr[(tr == WATER) & ~np.asarray(wet, bool)] = KEEP
    tr[tr > WALL] = KEEP
    return tgt, tr


# ------------------------------------------------------------------------ plan

def plan(found_h, found_wet, target, treatment, *, pave=None, water_surface=None,
         soft_grade: float = 1.0, max_retain: int = 12) -> GroundPlan:
    """Decide the finished ground. Pure arithmetic on the arrays; see the module
    docstring for the rules."""
    found = np.asarray(found_h, np.int32)
    wet = np.asarray(found_wet, bool) if found_wet is not None else np.zeros(found.shape, bool)
    tgt, tr = _effective(found, wet, target, treatment)
    pv = (np.asarray(pave, np.int16) if pave is not None
          else np.zeros(found.shape, np.int16))
    changed = (tr != KEEP) & (tr != WATER)
    fin = found.copy()
    fixed_set = changed & (tr != SOFT)
    fin[fixed_set] = tgt[fixed_set]
    soft = tr == SOFT
    fin[soft] = tgt[soft]
    s = 1.0 / max(1e-6, float(soft_grade))
    g = _grade_soft(fin, soft, s)
    fin = np.where(soft, np.round(g).astype(np.int32), fin)

    w, h = found.shape
    face = np.zeros((w, h), np.uint8)
    face_lo = fin + 1
    face_h = np.zeros((w, h), np.int32)
    cap = np.zeros((w, h), bool)
    parapet = np.zeros((w, h), bool)
    ramp = np.zeros((w, h), bool)
    bank_top = np.full((w, h), -1 << 20, np.int32)
    ws = (np.asarray(water_surface, np.int32) if water_surface is not None else None)
    gp = GroundPlan(found=found, wet=wet, target=tgt, treat=tr, fin=fin, changed=changed,
                    pave=pv, face=face, face_lo=face_lo, face_h=face_h, cap=cap,
                    parapet=parapet, ramp=ramp, bank_top=bank_top, water_surface=ws,
                    soft_grade=float(soft_grade), max_retain=int(max_retain))
    water_col = tr == WATER
    for di, dj in _N4:
        nf = _shift(fin, di, dj, 1 << 20)            # off the edge: never lower
        nt = _shift(tr.astype(np.int16), di, dj, -1)
        nch = _shift(changed, di, dj, False)
        nwater = _shift(water_col, di, dj, False)
        d = fin - nf                                  # >0: this column is the higher
        # a bank: a changed land column beside kept water
        bank = changed & nwater
        if bank.any():
            nbed = _shift(found, di, dj, 0)
            lo = nbed + 1
            face[bank] = F_BANK
            face_lo[bank] = np.minimum(face_lo[bank], np.minimum(lo[bank], fin[bank]))
            if ws is not None:
                nws = _shift(ws, di, dj, -1 << 20)
                bank_top[bank] = np.maximum(bank_top[bank], nws[bank])
            for i, j in zip(*np.nonzero(bank)):
                gp.faces.append((int(i), int(j), di, dj, int(fin[i, j] - lo[i, j] + 1),
                                 "bank"))
        land = ~nwater & (nt >= 0)
        # a street ramp: two STREET columns one apart -> a slab on the lower
        both_st = (tr == STREET) & (nt == STREET) & land
        ramp_lo = both_st & (d == -1)
        ramp |= ramp_lo
        st_step = both_st & (d >= 2)
        for i, j in zip(*np.nonzero(st_step)):
            gp.street_steps.append((int(i), int(j), di, dj, int(d[i, j])))
        # what faces
        hard_like = np.isin(tr, (HARD, WALL)) | ((tr == STREET) & (nt != STREET))
        need = land & (d >= 1) & (
            hard_like
            | st_step
            | ((tr == SOFT) & (d >= 2))
            | ((tr == KEEP) & nch & (d >= 2)))
        soft_bad = (tr == SOFT) & land & (d >= 2)
        for i, j in zip(*np.nonzero(soft_bad)):
            gp.soft_unresolved.append((int(i), int(j), int(d[i, j])))
        if need.any():
            kind = np.where(tr == KEEP, F_CUT, F_RETAIN).astype(np.uint8)
            upd = need & (face != F_BANK)
            face[upd] = np.maximum(face[upd], kind[upd])
            face_lo[need] = np.minimum(face_lo[need], nf[need] + 1)
            face_h[need] = np.maximum(face_h[need], d[need])
            # a face of one is a stone edge flush with the ground either side; a cap
            # (and over PARAPET_OVER a parapet) is for a face a person could fall from
            capme = need & (tr != WALL) & (d >= 2)
            cap |= capme
            par = capme & (d > PARAPET_OVER) & ~st_step
            parapet |= par
            for i, j in zip(*np.nonzero(need)):
                gp.faces.append((int(i), int(j), di, dj, int(d[i, j]),
                                 "cut" if tr[i, j] == KEEP else "retain"))
    # a ramp slab never sits on a column that is itself faced and capped
    ramp &= ~cap
    for (i, j, di, dj, hgt, kind) in gp.faces:
        if hgt > gp.max_retain:
            gp.over_bound.append((i, j, hgt))
    gp.over_bound = sorted(set(gp.over_bound))
    return gp


def _band_of(hgt: int) -> str:
    for lo, hi in BANDS:
        if lo <= hgt <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return "0"


def _summary(gp: GroundPlan, mask=None) -> dict:
    m = np.ones(gp.shape, bool) if mask is None else np.asarray(mask, bool)
    ch = gp.changed & m
    dz = gp.fin - gp.found
    cut_cols = ch & (dz < 0)
    fill_cols = ch & (dz > 0)
    by_band_len: dict = {}
    by_band_area: dict = {}
    max_ret = 0
    for (i, j, di, dj, hgt, kind) in gp.faces:
        if not m[i, j]:
            continue
        b = _band_of(hgt)
        by_band_len[b] = by_band_len.get(b, 0) + 1
        by_band_area[b] = by_band_area.get(b, 0) + int(hgt)
        if kind != "bank":
            max_ret = max(max_ret, int(hgt))
    steps = [s for s in gp.street_steps if m[s[0], s[1]]]
    cols = {NAMES[k]: int(((gp.treat == k) & m).sum()) for k in range(6)}
    return {"cut_columns": int(cut_cols.sum()),
            "cut_blocks": int((-dz[cut_cols]).sum()),
            "fill_columns": int(fill_cols.sum()),
            "fill_blocks": int(dz[fill_cols].sum()),
            "retained_length_by_height": by_band_len,
            "retained_area_by_height": by_band_area,
            "retained_length": int(sum(by_band_len.values())),
            "retained_area": int(sum(by_band_area.values())),
            "max_retain": max_ret,
            "street_steps_unresolved": len(steps),
            "max_unresolved_step": max([s[4] for s in steps], default=0),
            "street_steps_sample": [[int(s[0]), int(s[1]), int(s[4])] for s in steps[:12]],
            "street_ramps": int((gp.ramp & m).sum()),
            "soft_unresolved": sum(1 for s in gp.soft_unresolved if m[s[0], s[1]]),
            "over_bound": [[int(i), int(j), int(hh)] for (i, j, hh) in gp.over_bound
                           if m[i, j]][:40],
            "over_bound_count": sum(1 for (i, j, hh) in gp.over_bound if m[i, j]),
            "columns": cols,
            "columns_kept": cols["keep"],
            "parapets": int((gp.parapet & m).sum()),
            "caps": int((gp.cap & m).sum()),
            "raw_faces": _raw_faces(gp, m)}


def _raw_faces(gp: GroundPlan, m) -> int:
    """Pairs with a step of two or more beside a changed column that nothing faces:
    zero by construction; counted so the record says so rather than assumes it."""
    n = 0
    for di, dj in _N4:
        nf = _shift(gp.fin, di, dj, 1 << 20)
        nch = _shift(gp.changed, di, dj, False)
        nw = _shift(gp.treat == WATER, di, dj, False)
        d = gp.fin - nf
        bad = m & (d >= 2) & (gp.changed | nch) & ~nw & (gp.face == F_NONE)
        n += int(bad.sum())
    return n


# ------------------------------------------------------------------------ estimate

def estimate(found_h, target, treatment, *, found_wet=None, pave=None,
             soft_grade: float = 1.0, max_retain: int = 12) -> dict:
    """What a design would cost, without a volume: cut and fill (columns and blocks),
    retained face length and area by height band, the tallest face, unresolved street
    steps and the largest, SOFT steps it could not grade, faces over `max_retain`, and
    the columns kept. For comparing design proposals."""
    wet = found_wet if found_wet is not None else np.zeros(np.shape(found_h), bool)
    gp = plan(found_h, wet, target, treatment, pave=pave, soft_grade=soft_grade,
              max_retain=max_retain)
    out = _summary(gp)
    out["columns_total"] = int(np.size(gp.fin))
    return out


# ------------------------------------------------------------------------ apply

def _slab(block: str, fallback: str) -> str:
    try:
        from .buildlib import Builder
        return Builder.block(block, "slab")
    except Exception:                                  # noqa: BLE001 -- no slab shape
        return fallback


def apply(vol, x0: int, z0: int, gp: GroundPlan, *, materials: dict, only=None,
          base=None):
    """Lay the plan into a copy of `vol` (an `observe.Volume`), writing only the columns
    of `only` (`[w, h]` bool, default all). Returns `(vol2, record)`.

    `base` is the ground as found, as a `Volume` or its `codes` array, the same box as
    `vol` (its palette a prefix of `vol`'s, as `apply_diff` leaves it): where it is
    given, **a block that differs from the found ground is somebody's build** -- a
    neighbouring region's wall run or eave over these columns -- and is left exactly
    as it stands; the ground is cut, filled and faced round it.

    `materials`: `fill`, `cover`, `retain`, `cap` (a slab), `bank`, `path`, `pave`
    ({code: block}); optional `ramp` (the slab a street ramp is laid in where the pave
    block has no slab shape)."""
    from .observe import Volume
    w, h = gp.shape
    m = np.ones((w, h), bool) if only is None else np.asarray(only, bool).copy()
    codes = vol.codes.copy()
    pal = list(vol.palette)
    index = {s: i for i, s in enumerate(pal)}
    sx, sy, sz = codes.shape
    # the plan's window inside the volume
    ox, oz = x0 - vol.x0, z0 - vol.z0
    ix0, iz0 = max(0, -ox), max(0, -oz)
    ix1, iz1 = min(w, sx - ox), min(h, sz - oz)
    if ix1 <= ix0 or iz1 <= iz0:
        return Volume(vol.x0, vol.y0, vol.z0, codes, pal), {"applied_columns": 0}
    win = (slice(ix0, ix1), slice(iz0, iz1))
    vwin = (slice(ix0 + ox, ix1 + ox), slice(iz0 + oz, iz1 + oz))
    m = m[win]

    def code(state: str) -> int:
        c = index.get(state)
        if c is None:
            c = len(pal)
            pal.append(state)
            index[state] = c
        return c

    names = [p.split("[")[0].split(":")[-1] for p in pal]
    AIRN = ("air", "cave_air", "void_air")
    air = code("air")
    fill_c = code(materials.get("fill", "dirt"))
    cover_c = code(materials.get("cover", "grass_block"))
    retain_c = code(materials.get("retain", "stone_bricks"))
    cap_b = materials.get("cap", "stone_brick_slab")
    cap_c = code(cap_b if "[" in cap_b else cap_b + "[type=bottom]")
    bank_c = code(materials.get("bank", "stone_bricks"))
    path_c = code(materials.get("path", "dirt_path"))
    pave_map = {int(k): v for k, v in (materials.get("pave") or {}).items()}
    pave_c = {k: code(v) for k, v in pave_map.items()}
    ramp_default = materials.get("ramp") or cap_b.split("[")[0]
    ramp_c = {k: code(_slab(v, ramp_default) + "[type=bottom]") for k, v in pave_map.items()}
    ramp_c0 = code(_slab(materials.get("path", "dirt_path"), ramp_default) + "[type=bottom]")

    C = codes[vwin[0], :, vwin[1]]                    # (w', sy, h') view, a copy below
    C = C.copy()
    protected = None
    if base is not None:
        bc = base.codes if hasattr(base, "codes") else np.asarray(base)
        if bc.shape == codes.shape:
            protected = C != bc[vwin[0], :, vwin[1]]
    fin = gp.fin[win] - vol.y0
    found = gp.found[win] - vol.y0
    tr = gp.treat[win]
    ch = gp.changed[win] & m
    pv = gp.pave[win]
    face = gp.face[win]
    flo = gp.face_lo[win] - vol.y0
    capm = gp.cap[win] & m
    parm = gp.parapet[win] & m
    rampm = gp.ramp[win] & m
    btop = gp.bank_top[win] - vol.y0
    Y = np.arange(sy)[None, :, None]
    before = C.copy()
    water_i = [i for i, s_ in enumerate(pal) if s_.split("[")[0] == "water"]
    # the waterline a bank holds, where the plan was not told it: the highest water
    # surface among each bank column's kept-water neighbours, read off the volume
    if gp.water_surface is None and (face == F_BANK).any():
        wmask = np.isin(before, water_i)
        wtop = np.where(wmask.any(axis=1),
                        sy - 1 - np.argmax(wmask[:, ::-1, :], axis=1), -1 << 20)
        wkeep = tr == WATER
        btop = np.full(fin.shape, -1 << 20, np.int32)
        for di, dj in _N4:
            nws = _shift(np.where(wkeep, wtop, -1 << 20), di, dj, -1 << 20)
            btop = np.where(face == F_BANK, np.maximum(btop, nws), btop)

    # surface block by treatment
    top_code = np.full(fin.shape, cover_c, np.int32)
    top_code[tr == WALL] = fill_c
    st = tr == STREET
    top_code[st] = path_c
    for k, c in pave_c.items():
        top_code[(pv == k) & np.isin(tr, (STREET, HARD))] = c
    fin3 = fin[:, None, :]
    ch3 = ch[:, None, :]
    # cut: everything above the finished level in a worked column
    C = np.where(ch3 & (Y > fin3), air, C)
    # fill: from the found bed up to the finished level (water and air alike), fill
    lo3 = np.minimum(found, fin)[:, None, :]
    C = np.where(ch3 & (Y > lo3) & (Y < fin3), fill_c, C)
    # the top course
    C = np.where(ch3 & (Y == fin3), top_code[:, None, :], C)
    # faces: retain / cut / bank, from face_lo through the top
    fm = (face > 0) & m
    fcode = np.where(face == F_BANK, bank_c, retain_c)
    flo3 = flo[:, None, :]
    ftop = np.maximum(fin, np.where(face == F_BANK, btop, fin))
    ftop3 = ftop[:, None, :]
    C = np.where(fm[:, None, :] & (Y >= flo3) & (Y <= ftop3), fcode[:, None, :], C)
    # a KEEP column's cut face stops under its own turf: the face's top course is the
    # retain block too (a coping under the cap), which is what makes it a wall caps and
    # parapets
    ctop = np.where(parm, ftop + 1, ftop)
    C = np.where(parm[:, None, :] & (Y == (ftop + 1)[:, None, :]), retain_c, C)
    C = np.where(capm[:, None, :] & (Y == (ctop + 1)[:, None, :]), cap_c, C)
    # street ramps: a slab on the lower street column
    rc = np.full(fin.shape, ramp_c0, np.int32)
    for k, c in ramp_c.items():
        rc[pv == k] = c
    C = np.where(rampm[:, None, :] & (Y == fin3 + 1), rc[:, None, :], C)
    # orphaned leaves over the worked columns
    leaf_codes = [i for i, s in enumerate(pal) if s.split("[")[0].endswith("_leaves")
                  and "persistent=true" not in s]
    log_codes = [i for i, s in enumerate(pal)
                 if s.split("[")[0].endswith(("_log", "_wood")) and "stripped" not in s]
    leaves_cut = 0
    if leaf_codes and ch.any():
        from scipy import ndimage
        logs = np.isin(C, log_codes)
        near = ndimage.binary_dilation(logs, iterations=LEAF_REACH) if logs.any() \
            else np.zeros_like(logs)
        zone = ndimage.binary_dilation(ch, iterations=LEAF_REACH + 2) & m
        orphan = np.isin(C, leaf_codes) & ~near & zone[:, None, :]
        leaves_cut = int(orphan.sum())
        C[orphan] = air
    n_protected = 0
    if protected is not None:
        n_protected = int((protected & (C != before)).sum())
        C = np.where(protected, before, C)
        # ...and never undermined: under a block somebody built, nothing is cut away
        over = np.flip(np.logical_or.accumulate(np.flip(protected, 1), axis=1), 1)
        C = np.where(over & (C == air) & (before != air), before, C)
    codes[vwin[0], :, vwin[1]] = C
    out = Volume(vol.x0, vol.y0, vol.z0, codes, pal)
    # ---- the record
    ia = np.isin(before, np.nonzero(np.isin(np.arange(len(pal)),
                                            [index[s] for s in pal if s.split("[")[0]
                                             .split(":")[-1] in AIRN]))[0])
    ib = np.isin(C, [index[s] for s in pal if s.split("[")[0].split(":")[-1] in AIRN])
    was_water = np.isin(before, water_i)
    now_solid = ~ib & ~np.isin(C, water_i)
    rec = _summary(gp, _full_mask(gp.shape, win, m))
    rec.update({
        "applied_columns": int(m.sum()),
        "cut_blocks_placed": int((~ia & ~was_water & ib).sum()),
        "water_removed": int((was_water & ~np.isin(C, water_i)).sum()),
        "fill_blocks_placed": int(((ia | was_water) & now_solid).sum()),
        "leaves_orphaned_removed": leaves_cut,
        "built_blocks_left_standing": n_protected,
        "raw_faces_measured": _raw_faces_volume(C, fin, found, face, ch, tr, m, pal,
                                                retain_c, bank_c, cap_c),
    })
    return out, rec


def _full_mask(shape, win, m):
    full = np.zeros(shape, bool)
    full[win] = m
    return full


def _raw_faces_volume(C, fin, found, face, ch, tr, m, pal, retain_c, bank_c, cap_c):
    """Exposed side faces, read off the laid columns: for each applied column that is
    two or more above a 4-neighbour, with either of the two worked, the exposed blocks
    (neighbour's top + 1 .. own top) that are neither the retain nor the bank block.
    Zero is the bar."""
    from .observe import _is_vegetation
    w, sy, h = C.shape
    # the ground's top: not air, water, a plant or a tree
    solid = np.array([p.split("[")[0] not in ("air", "cave_air", "void_air", "water")
                      and not _is_vegetation(p)
                      and not p.split("[")[0].endswith(("_log", "_wood", "_leaves"))
                      for p in pal])
    tops = np.full((w, h), -1, np.int32)
    slab = np.array(["_slab" in p.split("[")[0] for p in pal])
    s3 = solid[C]
    anyc = s3.any(axis=1)
    tops[anyc] = sy - 1 - np.argmax(s3[:, ::-1, :], axis=1)[anyc]
    n = 0
    for di, dj in _N4:
        nt = _shift(tops, di, dj, 1 << 20)
        nch = _shift(ch, di, dj, False)
        d = tops - nt
        cand = m & (d >= 2) & (ch | nch)
        for i, j in zip(*np.nonzero(cand)):
            col = C[i, nt[i, j] + 1: tops[i, j] + 1, j]
            # a parapet's own cap and its retain course are part of the face, and a slab
            # on top (a street ramp's tread over a face of one) is a finished tread
            n += int((~(np.isin(col, (retain_c, bank_c, cap_c)) | slab[col])).sum())
    return n


# ------------------------------------------------------------------------ preview

def preview(gp: GroundPlan, path: str, *, scale: int | None = None) -> str:
    """A height-shaded map of the finished ground: hillshade on the level, tinted by
    treatment, retaining and cut faces in dark red (the higher column), banks in blue,
    street ramps in white, unresolved street steps in magenta, over-bound faces in
    yellow. For comparing design proposals by eye."""
    from PIL import Image
    fin = gp.fin.astype(np.float64)
    w, h = fin.shape
    gx = np.gradient(fin, axis=0)
    gz = np.gradient(fin, axis=1)
    shade = np.clip(0.75 + 0.18 * (-gx - gz), 0.35, 1.15)
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99)
    t = np.clip((fin - lo) / max(1.0, hi - lo), 0, 1)
    base = np.stack([90 + 120 * t, 110 + 100 * t, 70 + 90 * t], -1)
    tint = {KEEP: (0.85, 1.0, 0.8), SOFT: (0.8, 1.1, 0.7), HARD: (1.1, 1.0, 0.85),
            STREET: (1.05, 1.05, 1.05), WATER: (0.4, 0.6, 1.3), WALL: (1.2, 0.8, 0.7)}
    img = base.copy()
    for k, (r, g, b) in tint.items():
        mm = gp.treat == k
        img[mm] = img[mm] * np.array([r, g, b])
    street = gp.treat == STREET
    img[street] = 0.6 * img[street] + 0.4 * 200
    img = img * shade[..., None]
    img[gp.face == F_RETAIN] = (120, 30, 30)
    img[gp.face == F_CUT] = (150, 60, 40)
    img[gp.face == F_BANK] = (40, 70, 160)
    img[gp.ramp] = (235, 235, 235)
    for (i, j, di, dj, s) in gp.street_steps:
        img[i, j] = (230, 30, 230)
    for (i, j, hh) in gp.over_bound:
        img[i, j] = (250, 230, 30)
    arr = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 0, 2)   # rows = z
    sc = scale or max(1, int(600 // max(w, h)))
    im = Image.fromarray(arr).resize((w * sc, h * sc), Image.NEAREST)
    im.save(path)
    return path


# ------------------------------------------------------------------------ walls

def wall_floors(path, width: int, target, x0: int, z0: int, found=None) -> list:
    """The floor of each segment of a wall polyline under designed ground: the highest
    designed level under the band `Builder.edge_band` sweeps along it (the found ground
    where a column has no target). What a ring wall cut into runs is handed as
    `part["ring_floors"]` so every run levels its walk from the same numbers."""
    from .buildlib import Builder
    tg = np.asarray(target)
    fd = np.asarray(found) if found is not None else None
    w, h = tg.shape
    out = []
    for a, b in zip(path, path[1:]):
        best = None
        for (x, z) in Builder.edge_band(a, b, width):
            i, j = x - x0, z - z0
            if not (0 <= i < w and 0 <= j < h):
                continue
            v = int(tg[i, j])
            if v == NO_TARGET:
                if fd is None:
                    continue
                v = int(fd[i, j])
            best = v if best is None else max(best, v)
        out.append(best if best is not None else (int(fd.max()) if fd is not None else 64))
    return out
