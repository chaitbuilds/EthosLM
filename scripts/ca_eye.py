#!/usr/bin/env python3
"""**A perspective view of a built volume**, from a camera anywhere -- on foot in a lane,
on a wall, or high above the city -- with no server and no renderer.

    $PY scripts/ca_eye.py out/<run>/world_built.npz --at X,Y,Z --look X,Y,Z \
        --png eye.png [--size 960x540] [--fov 70] [--far 420]

`ethoslm.preview` draws isometric and orthographic frames, and Chunky renders only a
saved world, so without this a built candidate that has not been written can be seen
from above and never from the street. This marches a ray per pixel through the
volume's voxels (Amanatides-Woo, vectorised over the whole frame), colours each hit with
`preview.block_colour`, shades it by the face it entered and by distance, and paints the
sky. It is a diagnostic view, not a render: no textures, no transparency except water,
no light. `--at` and `--look` are world coordinates; `--eye` puts the camera 1.6 above
the ground at `--at`'s column.

`volume` may be a state directory, in which case the state's **adopted artifact** is
drawn (`ethoslm.artifact.adopted`) and the script prints which volume that is.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

AIR = ("air", "cave_air", "void_air", "light", "structure_void", "barrier")
#: block names that let the ray through: plants and thin things are drawn only where
#: nothing solid is behind them within a block (they are sampled, not skipped)
THIN = ("short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "seagrass",
        "tall_seagrass", "kelp", "kelp_plant", "vine", "torch", "wall_torch", "snow")


def load(path):
    from ethoslm import offline
    return offline.load_volume(path)


def textures(vol):
    """`(n, 16, 16, 3)` texels per palette entry, the flat colour where there is none."""
    from ethoslm import preview
    pal = [str(s) for s in vol.palette]
    tex = np.zeros((len(pal), 16, 16, 3), np.float32)
    for i, s in enumerate(pal):
        t = None
        try:
            t = preview.texture_of(s)
        except Exception:                                  # noqa: BLE001
            t = None
        if t is not None and t.shape[0] >= 16:
            tex[i] = t[:16, :16, :3].astype(np.float32)
            name = s.split("[")[0].split(":")[-1]
            # the atlas's grass and foliage are grey, tinted by biome in the game
            if any(k in name for k in ("grass", "leaves", "fern", "vine", "lily_pad")) \
                    and float(np.ptp(tex[i].mean(axis=(0, 1)))) < 25:
                tex[i] = tex[i] * np.array([0.55, 0.85, 0.42], np.float32)
        else:
            tex[i] = np.array(preview.block_colour(s)[:3], np.float32)
    return tex


def tables(vol):
    from ethoslm import preview
    pal = [str(s) for s in vol.palette]
    n = len(pal)
    col = np.zeros((n, 3), np.float32)
    solid = np.zeros(n, bool)
    water = np.zeros(n, bool)
    for i, s in enumerate(pal):
        name = s.split("[")[0].split(":")[-1]
        if name in AIR:
            continue
        c = preview.block_colour(s)
        col[i] = np.array(c[:3], np.float32)
        if name in ("water", "bubble_column") or "waterlogged=true" in s and name == "water":
            water[i] = True
            continue
        if name in THIN:
            continue
        solid[i] = True
    return col, solid, water


def render(vol, at, look, *, size=(960, 540), fov=70.0, far=420.0, sun=(0.45, 0.8, 0.35),
           texture=True, hits=None):
    """An (H, W, 3) uint8 frame. `hits`, if a dict, receives `"code"`: the (H, W)
    palette index of the block each pixel hit, -1 for sky."""
    col, solid, water = tables(vol)
    tex = textures(vol) if texture else None
    hit_code = None
    codes = vol.codes
    SX, SY, SZ = codes.shape
    W, H = size
    at = np.array(at, np.float64)
    fwd = np.array(look, np.float64) - at
    fwd /= np.linalg.norm(fwd)
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up0)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = np.tan(np.radians(fov) / 2.0)
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = 1 - (np.arange(H) + 0.5) / H * 2
    gx, gy = np.meshgrid(xs * t, ys * t * H / W)
    d = (fwd[None, None, :] + gx[..., None] * right + gy[..., None] * up).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    N = d.shape[0]
    # into volume index space
    o = at - np.array([vol.x0, vol.y0, vol.z0], np.float64)
    pos = np.repeat(o[None, :], N, 0)
    # start each ray where it enters the volume's box, so a camera above or beside the
    # volume sees into it; a ray that misses the box is sky
    lo_b = np.zeros(3)
    hi_b = np.array([SX, SY, SZ], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo_b[None, :] - pos) / d
        t2 = (hi_b[None, :] - pos) / d
    tn = np.nanmax(np.where(np.isfinite(np.minimum(t1, t2)), np.minimum(t1, t2), -np.inf), axis=1)
    tf = np.nanmin(np.where(np.isfinite(np.maximum(t1, t2)), np.maximum(t1, t2), np.inf), axis=1)
    miss = tf < np.maximum(tn, 0)
    t0 = np.clip(tn, 0, None) + 1e-4
    t0[miss] = 0.0
    pos0 = pos.copy()
    pos = pos + d * t0[:, None]
    cell = np.floor(pos).astype(np.int64)
    np.clip(cell, [0, 0, 0], [SX - 1, SY - 1, SZ - 1], out=cell)
    step = np.where(d >= 0, 1, -1).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        tdelta = np.where(d != 0, np.abs(1.0 / d), np.inf)
        nxt = np.where(step > 0, cell + 1 - pos, pos - cell)
        tmax = np.where(d != 0, nxt * tdelta, np.inf)
    out = np.zeros((N, 3), np.float32)
    hit = np.zeros(N, bool)
    face = np.zeros(N, np.int8)
    dist = np.full(N, far, np.float64)
    wet = np.zeros(N, np.float32)          # accumulated water depth along the ray
    wet_col = np.zeros((N, 3), np.float32)
    alive = np.nonzero(~miss)[0]
    tcur = t0.copy()
    last_axis = np.full(N, 1, np.int8)
    jit = np.zeros(N, np.float32)          # a per-block brightness, so courses read
    for _it in range(int(far * 3.2)):
        if not len(alive):
            break
        c = cell[alive]
        inside = ((c[:, 0] >= 0) & (c[:, 0] < SX) & (c[:, 1] >= 0) & (c[:, 1] < SY)
                  & (c[:, 2] >= 0) & (c[:, 2] < SZ))
        # a ray outside the volume above it may come back down: keep it while it is
        # within the horizontal bounds; drop it once it is out sideways or below
        gone = ~inside
        idx = np.zeros(len(alive), np.int64)
        ci = c[inside]
        idx[inside] = codes[ci[:, 0], ci[:, 1], ci[:, 2]]
        is_solid = inside & solid[idx]
        is_water = inside & water[idx]
        if is_water.any():
            a_w = alive[is_water]
            wet[a_w] += 1.0
            wet_col[a_w] = col[idx[is_water]]
        if is_solid.any():
            a_h = alive[is_solid]
            hit[a_h] = True
            out[a_h] = col[idx[is_solid]]
            if hit_code is None:
                hit_code = np.zeros(N, np.int64)
            hit_code[a_h] = idx[is_solid]
            face[a_h] = last_axis[a_h]
            dist[a_h] = tcur[a_h]
            hc = cell[a_h]
            jit[a_h] = ((hc[:, 0] * 73856093 ^ hc[:, 1] * 19349663 ^ hc[:, 2] * 83492791)
                        % 97) / 97.0
        stop = is_solid | gone | (tcur[alive] > far)
        alive = alive[~stop]
        if not len(alive):
            break
        tm = tmax[alive]
        ax = np.argmin(tm, axis=1)
        rows = np.arange(len(alive))
        tcur[alive] = tm[rows, ax]
        cell[alive, ax] += step[alive, ax]
        tmax[alive, ax] += tdelta[alive, ax]
        last_axis[alive] = ax
    # shading: the face the ray entered, lit by a fixed sun; distance haze
    sunv = np.array(sun, np.float64)
    sunv /= np.linalg.norm(sunv)
    normal = np.zeros((N, 3))
    for ax in range(3):
        m = face == ax
        normal[m, ax] = -step[m, ax]
    lam = np.clip((normal * sunv).sum(1), 0, 1)
    if tex is not None and hit_code is not None and hit.any():
        hp = pos0 + d * dist[:, None]
        fr = hp - np.floor(hp)
        u = np.where(face == 0, fr[:, 2], fr[:, 0])
        v = np.where(face == 1, fr[:, 2], 1.0 - fr[:, 1])
        iu = np.clip((u * 16).astype(np.int64), 0, 15)
        iv = np.clip((v * 16).astype(np.int64), 0, 15)
        hs = np.nonzero(hit)[0]
        out[hs] = tex[hit_code[hs], iv[hs], iu[hs]]
    shade = (0.55 + 0.45 * lam) * ((0.9 + 0.12 * jit) if tex is None else 1.0)
    img = out * shade[:, None]
    sky_t = np.clip((d[:, 1] + 0.1) / 0.8, 0, 1)
    sky = (np.array([200, 218, 235])[None, :] * (1 - sky_t[:, None])
           + np.array([120, 160, 215])[None, :] * sky_t[:, None])
    haze = np.clip(dist / far, 0, 1) ** 1.6
    img = np.where(hit[:, None], img * (1 - haze[:, None] * 0.75) + sky * haze[:, None] * 0.75,
                   sky)
    # water in front of whatever the ray reached: blue by its depth
    wa = np.clip(wet / 6.0, 0, 0.85)[:, None]
    img = img * (1 - wa) + np.array([50, 90, 170])[None, :] * wa
    if isinstance(hits, dict):
        hc = np.full(N, -1, np.int64)
        if hit_code is not None:
            hc[hit] = hit_code[hit]
        hits["code"] = hc.reshape(H, W)
    return np.clip(img, 0, 255).astype(np.uint8).reshape(H, W, 3)


def ground_y(vol, x, z):
    col, solid, water = tables(vol)
    i, k = int(x) - vol.x0, int(z) - vol.z0
    stack = vol.codes[i, :, k]
    ys = np.nonzero(solid[stack] | water[stack])[0]
    return int(vol.y0 + ys.max()) if len(ys) else vol.y0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("volume", help="a .npz volume, or a state directory (its adopted "
                                   "artifact is drawn)")
    ap.add_argument("--at", required=True, help="x,y,z (y may be 'eye')")
    ap.add_argument("--look", required=True, help="x,y,z")
    ap.add_argument("--png", required=True)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--far", type=float, default=420.0)
    a = ap.parse_args(argv)
    path = a.volume
    if os.path.isdir(path):
        from ethoslm import artifact as artifact_mod
        path = artifact_mod.adopted(path)
    print(f"volume {path}")
    vol = load(path)
    ax_, ay_, az_ = a.at.split(",")
    lx, ly, lz = a.look.split(",")
    x, z = float(ax_), float(az_)
    y = (ground_y(vol, x, z) + 2.6) if ay_ == "eye" else float(ay_)
    lyv = (ground_y(vol, float(lx), float(lz)) + 2.0) if ly == "eye" else float(ly)
    W, H = (int(v) for v in a.size.split("x"))
    img = render(vol, (x + 0.5, y, z + 0.5), (float(lx) + 0.5, lyv, float(lz) + 0.5),
                 size=(W, H), fov=a.fov, far=a.far)
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(a.png)), exist_ok=True)
    Image.fromarray(img).save(a.png)
    print(f"drew {a.png} from ({x:.0f},{y:.1f},{z:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
