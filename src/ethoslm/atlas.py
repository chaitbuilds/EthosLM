"""**The terrain atlas**: ground height and water for the whole generated world, cheaply.

The design synthesis round. The site search read terrain as cached squares of the
footprint it was asked for (`out/sites/ground_<x>_<z>_<n>x<n>.npz`), so a city larger
than any cached square had no ground to be judged on, and "no suitable site" meant "no
square of this size in the cache". Designing a larger place needs the ground first and
cheaply: its relief, its water and where the flat land lies, over tens of kilometres.

Every chunk's saved `Heightmaps` already hold it: `OCEAN_FLOOR` is the highest solid
block and `MOTION_BLOCKING_NO_LEAVES` is the surface including water, so their difference
is the water's depth. This reads those two arrays per chunk, straight from the region
files (read-only, no server, no block decoding), one process per region file, and keeps
one small file per region under `out/ds-atlas/`. A window of any size is then a stitch of
those files. Unfinished or missing chunks are `NODATA`.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORLD = os.path.join(ROOT, "run", "server", "world")
CACHE = os.path.join(ROOT, "out", "ds-atlas")
NODATA = -32768
MIN_Y = -64


def _unpack(longs, bits: int = 9, n: int = 256) -> np.ndarray:
    per = 64 // bits
    mask = (1 << bits) - 1
    out = np.empty(n, np.int32)
    for i in range(n):
        v = int(longs[i // per]) & 0xFFFFFFFFFFFFFFFF
        out[i] = (v >> ((i % per) * bits)) & mask
    return out


def _region(args) -> tuple:
    """One region file -> `(rx, rz, floor[512,512], surface[512,512])`, indexed [x, z],
    as the y of the top block; NODATA where a chunk is absent or unfinished."""
    path, rx, rz, out_dir = args
    dst = os.path.join(out_dir, f"r.{rx}.{rz}.npz")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(path):
        return rx, rz, "cached"
    from nbt.region import RegionFile
    floor = np.full((512, 512), NODATA, np.int16)
    surf = np.full((512, 512), NODATA, np.int16)
    try:
        with open(path, "rb") as f:
            reg = RegionFile(fileobj=f)
            for m in reg.get_metadata():
                try:
                    tag = reg.get_nbt(m.x, m.z)
                except Exception:                 # noqa: BLE001 -- a bad chunk is NODATA
                    continue
                if str(tag["Status"].value).split(":")[-1] != "full":
                    continue
                hm = tag.get("Heightmaps")
                if hm is None or "OCEAN_FLOOR" not in hm:
                    continue
                of = _unpack(hm["OCEAN_FLOOR"].value) + MIN_Y - 1
                mb = _unpack(hm["MOTION_BLOCKING_NO_LEAVES"].value) + MIN_Y - 1
                # index i = z * 16 + x within the chunk
                floor[m.x * 16:(m.x + 1) * 16, m.z * 16:(m.z + 1) * 16] = \
                    of.reshape(16, 16).T
                surf[m.x * 16:(m.x + 1) * 16, m.z * 16:(m.z + 1) * 16] = \
                    mb.reshape(16, 16).T
    except Exception as e:                        # noqa: BLE001 -- recorded
        return rx, rz, f"error {type(e).__name__}: {e}"
    np.savez_compressed(dst, floor=floor, surface=surf)
    return rx, rz, "read"


# ----------------------------------------------------------------- biomes The transfer
# round. A place spec's `setting` says what land a place stands in -- "an oasis" is dry
# sandy country -- and the design path's site candidates were read off heights and water
# alone, so a desert village was offered a snowy plain. The biome is already in every
# saved chunk, one entry per 4x4x4 cell, so the atlas keeps it beside the heights: the
# class (`groundread.BIOMES`) of the cell the ground stands in, one per 4x4 column, in a
# file of its own so an atlas read before this keeps its heights.

#: biome classes by index in the biome layer; 0 is unclassed (ocean, river, beach, cave)
BIOME_CLASSES = ("", "plains", "savanna", "forest", "jungle", "desert", "badlands",
                 "swamp", "snowy", "mountain")


def _biome_codes(sec_biomes) -> np.ndarray:
    """One section's 64 biome cells as class indices, index `(y * 4 + z) * 4 + x`."""
    from .groundread import biome_class
    pal = [BIOME_CLASSES.index(biome_class(str(v.value)) or "")
           for v in sec_biomes["palette"]]
    if len(pal) == 1 or "data" not in sec_biomes:
        return np.full(64, pal[0], np.uint8)
    bits = max(1, (len(pal) - 1).bit_length())
    per = 64 // bits
    mask = (1 << bits) - 1
    longs = sec_biomes["data"].value
    out = np.empty(64, np.uint8)
    for i in range(64):
        v = int(longs[i // per]) & 0xFFFFFFFFFFFFFFFF
        out[i] = pal[min(len(pal) - 1, (v >> ((i % per) * bits)) & mask)]
    return out


def _region_biomes(args) -> tuple:
    """One region file -> `r.<rx>.<rz>.biome.npz`: `[128, 128]` class indices, [x, z],
    each the biome of the cell a 4x4 column's ground stands in (255 where unread)."""
    path, rx, rz, out_dir = args
    dst = os.path.join(out_dir, f"r.{rx}.{rz}.biome.npz")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(path):
        return rx, rz, "cached"
    from nbt.region import RegionFile
    codes = np.full((128, 128), 255, np.uint8)
    try:
        with open(path, "rb") as f:
            reg = RegionFile(fileobj=f)
            for m in reg.get_metadata():
                try:
                    tag = reg.get_nbt(m.x, m.z)
                except Exception:                 # noqa: BLE001 -- a bad chunk is unread
                    continue
                if str(tag["Status"].value).split(":")[-1] != "full":
                    continue
                hm = tag.get("Heightmaps")
                if hm is None or "OCEAN_FLOOR" not in hm:
                    continue
                top = (_unpack(hm["OCEAN_FLOOR"].value) + MIN_Y - 1).reshape(16, 16)
                secs = {}
                for sec in tag["sections"]:
                    if "biomes" in sec:
                        secs[int(sec["Y"].value)] = sec["biomes"]
                for cx4 in range(4):
                    for cz4 in range(4):
                        y = int(top[cz4 * 4 + 2, cx4 * 4 + 2])
                        sy = y >> 4
                        sb = secs.get(sy)
                        if sb is None:
                            continue
                        cells = _biome_codes(sb)
                        yy = (y & 15) >> 2
                        codes[m.x * 4 + cx4, m.z * 4 + cz4] = cells[(yy * 4 + cz4) * 4
                                                                    + cx4]
    except Exception as e:                        # noqa: BLE001 -- recorded
        return rx, rz, f"error {type(e).__name__}: {e}"
    np.savez_compressed(dst, biome=codes)
    return rx, rz, "read"


def has_biomes(out_dir: str = CACHE) -> bool:
    """Does this atlas carry the biome layer for every region it has heights of?"""
    if not os.path.isdir(out_dir):
        return False
    hs = {f[:-4] for f in os.listdir(out_dir) if f.startswith("r.") and f.endswith(".npz")
          and not f.endswith(".biome.npz")}
    bs = {f[:-10] for f in os.listdir(out_dir) if f.endswith(".biome.npz")}
    return bool(hs) and hs <= bs


def build(world: str = WORLD, out_dir: str = CACHE, workers: int = 8,
          biomes: bool = True) -> dict:
    """Read every region file's heightmaps (and biomes) into the atlas cache.
    Incremental: a region already read is not read again."""
    from multiprocessing import Pool
    os.makedirs(out_dir, exist_ok=True)
    reg_dir = os.path.join(world, "region")
    jobs = []
    for f in sorted(os.listdir(reg_dir)):
        if not f.endswith(".mca"):
            continue
        _, rx, rz, _ = f.split(".")
        jobs.append((os.path.join(reg_dir, f), int(rx), int(rz), out_dir))
    with Pool(max(1, int(workers))) as p:
        got = p.map(_region, jobs, chunksize=1)
        bio = p.map(_region_biomes, jobs, chunksize=1) if biomes else []
    rec = {"regions": len(got), "read": sum(1 for g in got if g[2] == "read"),
           "cached": sum(1 for g in got if g[2] == "cached"),
           "errors": [g for g in got if str(g[2]).startswith("error")],
           "world": os.path.relpath(world, ROOT),
           "biomes": bool(biomes) and not any(str(g[2]).startswith("error") for g in bio)}
    rec["errors"] += [g for g in bio if str(g[2]).startswith("error")]
    xs = [g[0] for g in got]
    zs = [g[1] for g in got]
    if xs:
        rec["x"] = [min(xs) * 512, max(xs) * 512 + 511]
        rec["z"] = [min(zs) * 512, max(zs) * 512 + 511]
    json.dump(rec, open(os.path.join(out_dir, "atlas.json"), "w"), indent=1)
    return rec


def window(x0: int, z0: int, w: int, h: int, out_dir: str = CACHE) -> tuple:
    """`(floor, surface)` int16 `[w, h]` for the window, NODATA where unread."""
    floor = np.full((w, h), NODATA, np.int16)
    surf = np.full((w, h), NODATA, np.int16)
    for rx in range(x0 // 512, (x0 + w - 1) // 512 + 1):
        for rz in range(z0 // 512, (z0 + h - 1) // 512 + 1):
            p = os.path.join(out_dir, f"r.{rx}.{rz}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            ax0, az0 = max(x0, rx * 512), max(z0, rz * 512)
            ax1, az1 = min(x0 + w, rx * 512 + 512), min(z0 + h, rz * 512 + 512)
            if ax0 >= ax1 or az0 >= az1:
                continue
            floor[ax0 - x0:ax1 - x0, az0 - z0:az1 - z0] = \
                d["floor"][ax0 - rx * 512:ax1 - rx * 512, az0 - rz * 512:az1 - rz * 512]
            surf[ax0 - x0:ax1 - x0, az0 - z0:az1 - z0] = \
                d["surface"][ax0 - rx * 512:ax1 - rx * 512, az0 - rz * 512:az1 - rz * 512]
    return floor, surf


def ground(x0: int, z0: int, w: int, h: int, out_dir: str = CACHE) -> tuple:
    """`(ground, water_surface, wet)`: the top solid block under any canopy, the water's
    surface where there is water (else the ground), and the wet mask. `[w, h]`."""
    floor, surf = window(x0, z0, w, h, out_dir)
    wet = (surf > floor) & (floor != NODATA)
    g = np.where(wet, floor, np.minimum(floor, surf)).astype(np.int16)
    g[floor == NODATA] = NODATA
    return g, np.where(wet, surf, g).astype(np.int16), wet


def coverage(out_dir: str = CACHE) -> dict:
    p = os.path.join(out_dir, "atlas.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def coarse(step: int = 8, out_dir: str = CACHE) -> dict:
    """The whole atlas at `step` blocks per cell: median floor, water share, relief
    (max - min floor) per cell, and `x0, z0` of cell [0, 0]. For site search maps."""
    rec = coverage(out_dir)
    if not rec:
        raise RuntimeError("no atlas: run `atlas.build()` first")
    x0, x1 = rec["x"]
    z0, z1 = rec["z"]
    nx, nz = (x1 - x0 + 1) // step, (z1 - z0 + 1) // step
    med = np.full((nx, nz), NODATA, np.int16)
    lo = np.full((nx, nz), NODATA, np.int16)
    hi = np.full((nx, nz), NODATA, np.int16)
    wet = np.zeros((nx, nz), np.float32)
    biome = np.full((nx, nz), 255, np.uint8)
    per = 512 // step
    for f in os.listdir(out_dir):
        if not (f.startswith("r.") and f.endswith(".npz")) or f.endswith(".biome.npz"):
            continue
        _, rx, rz, _ = f.split(".")
        rx, rz = int(rx), int(rz)
        d = np.load(os.path.join(out_dir, f))
        fl0 = d["floor"].astype(np.int32)
        sf = d["surface"].astype(np.int32)
        fl = np.where(sf > fl0, fl0, np.minimum(fl0, sf))
        ok = fl0 != NODATA
        b = fl.reshape(per, step, per, step).transpose(0, 2, 1, 3).reshape(per, per, -1)
        s = sf.reshape(per, step, per, step).transpose(0, 2, 1, 3).reshape(per, per, -1)
        okb = ok.reshape(per, step, per, step).transpose(0, 2, 1, 3).reshape(per, per, -1)
        full = okb.all(axis=2)
        i0, j0 = (rx * 512 - x0) // step, (rz * 512 - z0) // step
        sl = (slice(i0, i0 + per), slice(j0, j0 + per))
        med[sl] = np.where(full, np.median(b, axis=2), NODATA).astype(np.int16)
        lo[sl] = np.where(full, b.min(axis=2), NODATA).astype(np.int16)
        hi[sl] = np.where(full, b.max(axis=2), NODATA).astype(np.int16)
        wet[sl] = np.where(full, (s > b).mean(axis=2), 0.0)
        # (s > b) is water: under a canopy s is below the leaf-counting floor
        bp = os.path.join(out_dir, f"r.{rx}.{rz}.biome.npz")
        if os.path.exists(bp):
            bc = np.load(bp)["biome"]              # [128, 128], 4 blocks a cell
            k = max(1, step // 4)
            biome[sl] = bc[k // 2::k, k // 2::k][:per, :per]
    return {"step": step, "x0": x0, "z0": z0, "median": med, "low": lo, "high": hi,
            "wet": wet, "biome": biome, "biomes": has_biomes(out_dir)}


def biome_window(x0: int, z0: int, w: int, h: int, out_dir: str = CACHE) -> np.ndarray:
    """Biome class names `[w, h]` over a window ('' unclassed, None unread): the land
    a place stands in, for its maps and its finishing."""
    got = np.full((w, h), 255, np.uint8)
    for rx in range(x0 // 512, (x0 + w - 1) // 512 + 1):
        for rz in range(z0 // 512, (z0 + h - 1) // 512 + 1):
            p = os.path.join(out_dir, f"r.{rx}.{rz}.biome.npz")
            if not os.path.exists(p):
                continue
            bc = np.repeat(np.repeat(np.load(p)["biome"], 4, 0), 4, 1)
            ax0, az0 = max(x0, rx * 512), max(z0, rz * 512)
            ax1, az1 = min(x0 + w, rx * 512 + 512), min(z0 + h, rz * 512 + 512)
            if ax0 >= ax1 or az0 >= az1:
                continue
            got[ax0 - x0:ax1 - x0, az0 - z0:az1 - z0] = \
                bc[ax0 - rx * 512:ax1 - rx * 512, az0 - rz * 512:az1 - rz * 512]
    return got


if __name__ == "__main__":
    import sys
    print(json.dumps(build(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8), indent=1))


# ----------------------------------------------------------------- site scan

def site_measures(co: dict, cx: float, cz: float, radius: float,
                  outline=None) -> dict | None:
    """What the ground inside a circle (or `outline`) of `radius` about (cx, cz) is,
    off the coarse atlas: broad fall (p95 - p5 of cell medians), roughness (cells whose
    own relief exceeds `ROUGH`), water share, and the usable share (dry and smooth).
    None where the circle leaves the atlas or holds unread ground."""
    step, x0, z0 = co["step"], co["x0"], co["z0"]
    i0, i1 = int((cx - radius - x0) // step), int((cx + radius - x0) // step) + 1
    j0, j1 = int((cz - radius - z0) // step), int((cz + radius - z0) // step) + 1
    med = co["median"]
    if i0 < 0 or j0 < 0 or i1 > med.shape[0] or j1 > med.shape[1]:
        return None
    ii = np.arange(i0, i1)[:, None]
    jj = np.arange(j0, j1)[None, :]
    px = x0 + (ii + 0.5) * step
    pz = z0 + (jj + 0.5) * step
    if outline is not None:
        inside = outline.ratio(px, pz) <= 1.0
    else:
        inside = np.hypot(px - cx, pz - cz) <= radius
    m = med[i0:i1, j0:j1]
    if (m[inside] == NODATA).any():
        return None
    hi, lo = co["high"][i0:i1, j0:j1], co["low"][i0:i1, j0:j1]
    wet = co["wet"][i0:i1, j0:j1]
    rel = (hi.astype(int) - lo)[inside]
    mm = m[inside].astype(float)
    w = wet[inside]
    dry = w < 0.5
    smooth = rel <= ROUGH
    biomes = None
    if co.get("biomes") and "biome" in co:
        bc = co["biome"][i0:i1, j0:j1][inside]
        read = bc != 255
        if read.any():
            biomes = {BIOME_CLASSES[c]: round(float((bc[read] == c).mean()), 3)
                      for c in np.unique(bc[read]) if BIOME_CLASSES[c]}
    return {"centre": [int(cx), int(cz)], "radius": int(radius), "biomes": biomes,
            "fall": int(np.percentile(mm[dry], 95) - np.percentile(mm[dry], 5))
            if dry.any() else None,
            "median": int(np.median(mm[dry])) if dry.any() else None,
            "water": round(float(w.mean()), 3),
            "rough": round(float((~smooth & dry).mean()), 3),
            "usable": round(float((smooth & dry).mean()), 3),
            "cells": int(inside.sum())}


#: a coarse cell whose own relief exceeds this is rough ground (a slope of more than one
#: block in two across an 8-block cell)
ROUGH = 4


def scan_sites(radius: float, *, co: dict | None = None, stride: int = 128,
               water=(0.01, 0.2), top: int = 8, excluded=None, biomes=None,
               least: float = 0.5) -> list:
    """Candidate centres for a place of `radius`, best first: the circle entirely on
    read ground, clear of every reserved site, water inside the band, ranked by broad
    fall and roughness together (`fall + 400 * rough`, both measured, both reported).

    `biomes` is the setting's list of biome classes (`groundread.BIOMES`), or None: a
    circle with less than `least` of its cells in them is not a candidate, and among
    those that are, more of the list ranks higher. An atlas with no biome layer cannot
    answer and the list is ignored (the caller says so).

    The ranking is arithmetic about ground, not a decision: the design job reads the
    candidates with their maps and chooses."""

    co = co or coarse()
    want = [b for b in (biomes or []) if b and b != "any"] if co.get("biomes") else []
    step, x0, z0 = co["step"], co["x0"], co["z0"]
    X1 = x0 + co["median"].shape[0] * step
    Z1 = z0 + co["median"].shape[1] * step
    out = []
    for cx in range(int(x0 + radius), int(X1 - radius), stride):
        for cz in range(int(z0 + radius), int(Z1 - radius), stride):
            if excluded is not None and excluded(int(cx - radius), int(cz - radius),
                                                 int(2 * radius), int(2 * radius)):
                continue
            m = site_measures(co, cx, cz, radius)
            if not m or m["fall"] is None:
                continue
            if not (water[0] <= m["water"] <= water[1]):
                continue
            share = None
            if want:
                share = round(sum((m.get("biomes") or {}).get(b, 0.0) for b in want), 3)
                if share < least:
                    continue
                m["setting_share"] = share
            m["score"] = round(m["fall"] + 400 * m["rough"]
                               + (40 * (1.0 - share) if share is not None else 0.0), 1)
            out.append(m)
    out.sort(key=lambda r: r["score"])
    # keep candidates apart: a centre within a radius of a better one is the same site
    kept = []
    for r in out:
        if all(math.hypot(r["centre"][0] - k["centre"][0],
                          r["centre"][1] - k["centre"][1]) > radius for k in kept):
            kept.append(r)
        if len(kept) >= top:
            break
    return kept
