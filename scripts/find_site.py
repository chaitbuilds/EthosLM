"""Choose the ground a place stands on. Deterministic, by its needs.

Every round before this one was told where to build. nine structures against a
registered thirty -- is that decision being wrong and nothing in the system being able
to say so. **The system chooses now, and the record shows why.**

How, end to end:

  - the scan region is a square lattice of candidate origins on `STRIDE`, out to a
    radius that **grows** -- `RADII` -- so the nearest good ground wins and a search
    stops as soon as the needs are met rather than reading the whole world;
  - the world is read **once**: the four heightmaps over each 512-block tile the lattice
    touches, assembled into one array and cached to disk. Every candidate is then scored
    off that array with no further server read, which is what makes a scan of ten
    thousand candidates cost one read of the ground under them;
  - a candidate is measured on the terrain bank's own measures -- relief, water share,
    forest cover, gravity blocks -- plus **the flattest plateau inside it**, because a
    place has a centre and the centre wants level ground;
  - candidates are ranked, the best is taken, and the top three go on the record with
    their scores. Nothing here is a preference: every tie ends in the candidate's own
    coordinates.

Two escapes, in order, each recorded rather than silent:

  - **no candidate is flat enough at the footprint.** The innermost defining part is
    marked for terraforming -- `Builder.plateau()`, A4 -- and the scan is scored again
    with the plateau's relief allowance instead of the ground's.
  - **still none.** The spec drops one size band, which is a smaller footprint, and the
    scan is scored again. The drop is written into the answer.

Water and forest come free from the heightmaps: `MOTION_BLOCKING` over `OCEAN_FLOOR` is
standing water, and `WORLD_SURFACE` over `MOTION_BLOCKING_NO_LEAVES` is canopy. Gravity
blocks are not in a heightmap, so they are measured on the **shortlist** only, by reading
the surface of each of the best `SHORTLIST` candidates. That is the one place this file
reads the world twice and it is reported in the answer.

{"surface": "green", "water": "some"}` is scored on it: the wanted surface must be at
least `spec.SURFACE_SHARE` of the land, "some" water at least `spec.WATER_SOME_PCT` of
the footprint, and among the squares that meet it more of the surface ranks higher. A
square read before the setting existed has no surface in its cache; it is re-read where
a session is open and is otherwise **stale** -- scorable on everything but the setting,
and refused by name against one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import observe, offline, spec as spec_mod                  # noqa: E402
from ethoslm import groundread                                         # noqa: E402
from ethoslm.buildlib import Builder                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: The lattice a candidate origin sits on. The spec's number.
STRIDE = 256

#: How far out the scan goes, in the order it goes. It stops at the first radius that
#: produces a candidate meeting the needs, so a good site near the origin is never paid
#: for with a scan of the whole world. so a scan for a city that stopped where a town's
#: stopped would be a scan over ground that is mostly not there. Past the region files
#: the search is asking the server to make ground, it counts every square of it, and
#: `--max-new` is what lets any through.
RADII = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)

#: How many candidates are measured for gravity blocks and reported with scores.
SHORTLIST = 12

#: How many go on the record. The spec's number.
RECORDED = 3

#: The tile the world is read in. 512 is one region file and loads in about five
#: seconds; smaller costs more round trips and bigger runs the JVM out of heap.
TILE = 512

#: The relief a **plateau** may be asked to take out, over its own square. Above this
#: the cut is a quarry rather than a terrace, which is `Builder.PLATEAU_MAX`'s reason in
#: the other dimension. Eight either side of a median is sixteen blocks of face.
PLATEAU_RELIEF = 16

#: How big the level ground at a place's centre has to be when the spec does not say. A
#: market square is 22x22 at the largest any type declares, a keep stands beside it, and
#: the lane between them is the rest: 48 is those three with room to arrange them.
PLATEAU_DEFAULT = 48

#: The spec's number, registered before any of this ran. It is a **preference and not a
#: refusal**, and that is a decision with a reason on the record rather than a
#: softening. A hard cap of 40 over the outer ring of a 512x512 city would therefore
#: refuse every piece of ground this world has -- which is the failure mode this project
#: has now named twice under its own rule, *read what an instrument admits, not only
#: what it rejects*. So 40 is what the outer ring is scored against, in the same shape
#: gravity is: over it costs, and does not disqualify.
OUTER_RELIEF = 40

#: What an outer ring over `OUTER_RELIEF` costs in the score, against relief, water,
#: forest and plateau at 1.0 each and gravity at 0.25. Below them all: the whole of A1
#: is that a steep outer ring is a thing a city may have, and this weight is what keeps
#: it from silently becoming the thing every candidate is chosen on.
OUTER_WEIGHT = 0.25

#: The blocks that fall when what is under them goes. `scripts/terrain_bank.py`'s list,
#: by reference to the same idea: a pad cut into a bank of these has a landslide over
#: it.
GRAVITY = ("sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel")

#: What counts as canopy in the cached path. The live path reads it off the heightmaps.
CANOPY = ("_log", "_wood", "_leaves", "_stem", "_hyphae")

#: Where a read of the world is kept. One directory, one file per grid square, so a scan
#: that had to open a server is a scan that never has to again.
SITES = os.path.join(ROOT, "out", "sites")

#: The world save the region files belong to, relative to the repository. A chunk whose
#: region file is not here has never been generated, and asking the server for it is
#: asking the server to make it -- which is exactly what a search for a city has to be
#: able to do, and exactly what it has to be able to say it did.
REGIONS = os.path.join(ROOT, "run", "server", "world", "region")

#: One region file is 32x32 chunks, which is 512x512 blocks.
REGION = 512


def generated_regions(directory: str | None = None) -> set:
    """Every `(rx, rz)` the world save already has a region file for.

        Read off the file names and nothing else. This is the one fact that separates
        ground the world has and ground the world would have to *make*, and until A5 the
        search could not tell the difference: it asked the server for a slice, the server
        generated whatever was missing without a word, and a scan past 2,048 -- where this
        world's generated terrain ends -- would have quietly created terrain and reported a
        site on it as though it had found one.
        
    """
    import re
    d = directory or REGIONS
    if not os.path.isdir(d):
        return set()
    out = set()
    for f in os.listdir(d):
        m = re.match(r"r\.(-?\d+)\.(-?\d+)\.mca$", f)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def region_span(regions: set) -> dict:
    """The block rectangle the generated regions cover, and how many there are."""
    if not regions:
        return {"regions": 0, "x": None, "z": None}
    xs = [r[0] for r in regions]
    zs = [r[1] for r in regions]
    return {"regions": len(regions),
            "x": [min(xs) * REGION, (max(xs) + 1) * REGION - 1],
            "z": [min(zs) * REGION, (max(zs) + 1) * REGION - 1]}


def finished_square(regs, x0: int, z0: int, size: int) -> bool:
    """Is this square finished ground, probed at its four corners and its centre?

        `covers` says the region *files* exist; a region file holds chunks the generator
        started and never finished, and a square with one of those inside is a square
        `field_from_regions` parses for fifteen seconds and then refuses. Five chunk reads
        at three milliseconds each say so first. The ground run's first live session
        counted 166 such squares as file reads and estimated them at an hour and a half.
        
    """
    from ethoslm import regions as _regions
    pts = [(x0, z0), (x0 + size - 1, z0), (x0, z0 + size - 1),
           (x0 + size - 1, z0 + size - 1), (x0 + size // 2, z0 + size // 2)]
    try:
        for (x, z) in pts:
            regs.chunk(x >> 4, z >> 4)
    except _regions.NotGenerated:
        return False
    except Exception:                        # noqa: BLE001 -- unreadable is unfinished
        return False
    return True


def covers(regions: set, x0: int, z0: int, w: int, d: int) -> bool:
    """Is every region this rectangle touches already on disk?"""
    for rx in range(x0 // REGION, (x0 + w - 1) // REGION + 1):
        for rz in range(z0 // REGION, (z0 + d - 1) // REGION + 1):
            if (rx, rz) not in regions:
                return False
    return True


def tile_cache(x0: int, z0: int, w: int, d: int, directory: str | None = None) -> str:
    """Where one read of one grid square is kept. Named for the square, not the run."""
    return os.path.join(directory or SITES, f"ground_{x0}_{z0}_{w}x{d}.npz")


#: What a square's cache has to carry to be current. a file without the last is a square
#: read before the search could ask what its ground is made of.
CACHE_FIELDS = ("h", "wet", "canopy", "manmade", "occupied", "surface_codes")


def _field_from_cache(got, x0, z0, w, d, source) -> "Field":
    """A `Field` off a cache file's arrays, whichever generation of the format wrote it."""
    has = "surface_codes" in got
    return Field(x0, z0, got["h"], got["wet"], got["canopy"], source=source,
                 manmade=got["manmade"], occupied=got["occupied"],
                 surface=got["surface_codes"] if has else None,
                 surface_palette=[str(s) for s in got["surface_palette"]] if has else None,
                 biome=got["biome_codes"] if "biome_codes" in got else None,
                 biome_palette=([str(s) for s in got["biome_palette"]]
                                if "biome_codes" in got else None))


def read_tile(x0: int, z0: int, w: int, d: int, *, editor=None, regions=None,
              directory: str | None = None, log=print):
    """One grid square of ground, from the cache or -- once -- from the world. A5."""
    p = tile_cache(x0, z0, w, d, directory)
    if os.path.exists(p):
        got = np.load(p)
        if (int(got["x0"]), int(got["z0"]), *got["h"].shape) == (x0, z0, w, d):
            stale = "surface_codes" not in got
            if not stale or (editor is None and regions is None):
                return (_field_from_cache(got, x0, z0, w, d, p),
                        "stale" if stale else "cache")
    if editor is not None:
        return (field_from_server(editor, x0, z0, w, d, cache=p, log=log), "server")
    if regions is not None:
        f = field_from_regions(regions, x0, z0, w, d, cache=p, log=log)
        return (f, "files") if f is not None else (None, "unfinished")
    return (None, None)


def _cached(x: int, z: int, size: int, directory: str | None) -> bool:
    """Does the cache hold this square in the current format?"""
    p = tile_cache(x, z, size, size, directory)
    if not os.path.exists(p):
        return False
    got = np.load(p)
    return ((int(got["x0"]), int(got["z0"]), *got["h"].shape) == (x, z, size, size)
            and all(k in got for k in CACHE_FIELDS))


def _prefetch(squares: list, size: int, directory: str | None, n: int | None = None,
              log=print) -> int:
    """Read every square in `squares` off the region files into the cache, across
    processes. Returns how many were read. A square is fifteen seconds of NBT parsing
    and nothing one square does is visible to another.
    """
    from ethoslm import regions as _regions
    from ethoslm.parallel import par_map, workers
    items = [(x, z, size, tile_cache(x, z, size, size, directory)) for (x, z) in squares]
    got = par_map(_regions.prefetch_one, items, n=workers(n or 4))
    log(f"   {sum(got)} of {len(squares)} squares read off the region files "
        f"({len(squares) - sum(got)} not finished ground)")
    return sum(got)


# ------------------------------------------------------------------- the field One
# shape of answer whichever way the ground was read, so the scoring below is the same
# code offline and live and the test can hold the live path to the cached one.

class Field:
    """The ground over a rectangle, as four arrays indexed `[x - x0, z - z0]`."""

    def __init__(self, x0, z0, h, wet, canopy, gravity=None, source="",
                 manmade=None, occupied=None, measured=True,
                 surface=None, surface_palette=None, biome=None, biome_palette=None):
        self.x0, self.z0 = int(x0), int(z0)
        self.h = h.astype(np.int32)
        self.wet = wet.astype(bool)
        self.canopy = canopy.astype(bool)
        self.gravity = None if gravity is None else gravity.astype(bool)
        self.source = source
        self.manmade = (np.zeros(h.shape, dtype=np.int64) if manmade is None else manmade)
        self.occupied = (np.ones(h.shape, dtype=np.int64) if occupied is None else occupied)
        self.measured = measured
        # The setting: the block each column's ground is made of, as an index into
        # `surface_palette`, and the biome of each 4x4 cell as an index into
        # `biome_palette`. None where the square was read before either was asked for.
        self.surface = None if surface is None else np.asarray(surface).astype(np.int32)
        self.surface_palette = (None if surface is None
                                else [str(s) for s in surface_palette])
        self.biome = None if biome is None else np.asarray(biome).astype(np.int32)
        self.biome_palette = None if biome is None else [str(s) for s in biome_palette]
        # A surface that has been read says what every column is made of, and that
        # answers the gravity question over the whole square rather than on a 24x24
        # sample of a shortlist -- the read thread 22 named as the one the grid walk
        # never made. Given gravity stands; absent, it is the surface's.
        if self.gravity is None and self.surface is not None:
            grav = np.array([n in GRAVITY or n.endswith("_concrete_powder")
                             for n in self.surface_palette], bool)
            self.gravity = grav[self.surface]

    @property
    def shape(self):
        return self.h.shape

    def sub(self, x0, z0, w, d):
        i, j = x0 - self.x0, z0 - self.z0
        if i < 0 or j < 0 or i + w > self.shape[0] or j + d > self.shape[1]:
            return None
        s = (slice(i, i + w), slice(j, j + d))
        return (self.h[s], self.wet[s], self.canopy[s],
                None if self.gravity is None else self.gravity[s])


def field_from_volume(vol) -> Field:
    """A field off a cached volume: everything measured, including gravity."""
    h, wet = observe.ground_heights(vol)
    names = [s.split("[")[0] for s in vol.palette]
    air = np.array([n in ("air", "cave_air", "void_air") for n in names], bool)
    solid = ~air[vol.codes]
    idx = solid.shape[1] - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    top = np.take_along_axis(vol.codes, idx[:, None, :], axis=1)[:, 0, :]
    top = np.where(solid.any(axis=1), top, 0)
    ly = np.clip(h - vol.y0, 0, vol.codes.shape[1] - 1)
    ground = np.take_along_axis(vol.codes, ly[:, None, :], axis=1)[:, 0, :]
    canopy = np.array([any(v in n for v in CANOPY) for n in names], bool)
    grav = np.array([n in GRAVITY or n.endswith("_concrete_powder") for n in names],
                    bool)
    manmade, occupied = groundread.occupancy(vol)
    surface, palette = surface_codes(vol)
    return Field(vol.x0, vol.z0, h, wet, canopy[top], grav[ground],
                 source="cached volume", manmade=manmade, occupied=occupied,
                 surface=surface, surface_palette=palette)


def surface_codes(vol) -> tuple:
    """The block each column's ground is made of, as `(codes, palette)`.

        `groundread.surface_names` answers by name; this is the same answer indexed, which
        is the shape a cache file and a `Field` carry it in.
        
    """
    names = groundread.surface_names(vol)
    palette, codes = np.unique(names.astype(str), return_inverse=True)
    return codes.reshape(names.shape).astype(np.int32), [str(s) for s in palette]


#: The biome is stored by the game in 4x4x4 groups, so it is read once per 4x4 cell of
#: ground, at the ground's own height -- caves have biomes of their own and a plain read
#: sixty blocks down is a dripstone cave.
BIOME_CELL = 4


def biome_codes(ws, x0: int, z0: int, w: int, d: int, h: np.ndarray) -> tuple:
    """The biome of every `BIOME_CELL` square of one tile, as `(codes, palette)`.

        Off the world slice the tile was read from, at the ground height `h` gives for the
        cell's first column. Recorded and never scored: a biome is the world's own word for
        a setting and the record of a chosen site should carry it, but what a rendered
        place stands on is its **surface**, and that is what a setting is held to.
        
    """
    cw, cd = -(-w // BIOME_CELL), -(-d // BIOME_CELL)
    codes = np.zeros((cw, cd), np.int32)
    palette: list = []
    index: dict = {}
    for i in range(cw):
        for j in range(cd):
            lx, lz = i * BIOME_CELL, j * BIOME_CELL
            y = int(h[lx, lz])
            b = ws.getBiomeGlobal((x0 + lx, y, z0 + lz)).split(":")[-1]
            if b not in index:
                index[b] = len(palette)
                palette.append(b)
            codes[i, j] = index[b]
    return codes, palette


def field_from_server(ed, x0, z0, w, d, *, cache: str | None = None,
                      log=print) -> Field:
    """A field read through the server, tile by tile, once. Cached to `cache`.

        `h` is `OCEAN_FLOOR` -- the top non-fluid block, which is the bed -- so a lake
        reads as its floor and a site is not scored on the surface of one. The block
        volume of each tile is built once and read for the man-made census, the surface
        and the biome, and only the per-column answers are kept.
        
    """
    from gdpc.vector_tools import Rect

    def load(tx, tz, tw, td):
        ed.loadWorldSlice(Rect((tx, tz), (tw, td)), cache=True)
        return ed.worldSlice
    return field_from_slices(load, x0, z0, w, d, cache=cache, log=log, source="server")


def field_from_regions(regs, x0, z0, w, d, *, cache: str | None = None,
                       log=print) -> Field | None:
    """A field read off the save's region files, with no server.

        The same tiles, the same heightmaps and the same volume as the server path --
        `ethoslm.regions` hands `gdpc.WorldSlice` the chunk NBT the save holds -- so a square
        read this way is byte-identical to one read through a session. **None** where any
        chunk of it is not finished ground: a file cannot make ground, and the square is
        reported unread rather than generated.
        
    """
    from ethoslm import regions as _regions
    try:
        return field_from_slices(regs.world_slice, x0, z0, w, d, cache=cache, log=log,
                                 source="region files")
    except _regions.NotGenerated as e:
        log(f"   ({x0},{z0}) is not finished ground on disk: {e}")
        return None


def field_from_slices(load, x0, z0, w, d, *, cache: str | None = None, log=print,
                      source: str = "server") -> Field:
    """A field off `load(tx, tz, tw, td) -> WorldSlice`, tile by tile, once. Cached."""
    from ethoslm import world
    if cache and os.path.exists(cache):
        d0 = np.load(cache)
        if (int(d0["x0"]), int(d0["z0"]), *d0["h"].shape) == (x0, z0, w, d) \
                and all(k in d0 for k in CACHE_FIELDS):
            log(f"   heightmaps from {os.path.relpath(cache, ROOT)}")
            return _field_from_cache(d0, x0, z0, w, d, cache)
    h = np.zeros((w, d), np.int32)
    wet = np.zeros((w, d), bool)
    can = np.zeros((w, d), bool)
    manmade, occupied = np.zeros((w, d), np.int64), np.zeros((w, d), np.int64)
    # The setting: one palette of ground blocks and one of biomes over the whole square,
    # indexed per column and per 4x4 cell, built up tile by tile.
    surface = np.zeros((w, d), np.int32)
    s_palette: list = []
    s_index: dict = {}
    cw, cd = -(-w // BIOME_CELL), -(-d // BIOME_CELL)
    biome = np.zeros((cw, cd), np.int32)
    b_palette: list = []
    b_index: dict = {}
    biome_read = True
    tiles = [(tx, tz) for tx in range(x0, x0 + w, TILE)
             for tz in range(z0, z0 + d, TILE)]
    t0 = time.perf_counter()
    for n, (tx, tz) in enumerate(tiles, 1):
        tw, td = min(TILE, x0 + w - tx), min(TILE, z0 + d - tz)
        ws = load(tx, tz, tw, td)
        hm = ws.heightmaps
        floor = hm["OCEAN_FLOOR"].astype(np.int32) - 1
        block = hm["MOTION_BLOCKING"].astype(np.int32) - 1
        leaves = hm["WORLD_SURFACE"].astype(np.int32) - 1
        noleaf = hm[world.HEIGHTMAP].astype(np.int32) - 1
        s = (slice(tx - x0, tx - x0 + tw), slice(tz - z0, tz - z0 + td))
        h[s] = floor
        wet[s] = block > floor
        can[s] = leaves > noleaf
        volume = observe.Volume.from_world_slice(ws, tx, tz, tw, td, -64, 320)
        manmade[s], occupied[s] = groundread.occupancy(volume)
        codes, pal = surface_codes(volume)
        remap = np.zeros(len(pal), np.int32)
        for i, name in enumerate(pal):
            if name not in s_index:
                s_index[name] = len(s_palette)
                s_palette.append(name)
            remap[i] = s_index[name]
        surface[s] = remap[codes]
        if biome_read:
            try:
                gh, _gw = observe.ground_heights(volume)
                bc, bp = biome_codes(ws, tx, tz, tw, td, gh)
                bremap = np.zeros(len(bp), np.int32)
                for i, name in enumerate(bp):
                    if name not in b_index:
                        b_index[name] = len(b_palette)
                        b_palette.append(name)
                    bremap[i] = b_index[name]
                bs = (slice((tx - x0) // BIOME_CELL, (tx - x0) // BIOME_CELL + bc.shape[0]),
                      slice((tz - z0) // BIOME_CELL, (tz - z0) // BIOME_CELL + bc.shape[1]))
                biome[bs] = bremap[bc]
            except Exception as e:            # noqa: BLE001 -- recorded, never scored
                biome_read = False
                log(f"   biome unread for this square: {type(e).__name__}: {e}")
        if n % 8 == 0 or n == len(tiles):
            log(f"   {n}/{len(tiles)} tiles, {time.perf_counter() - t0:.0f}s")
    f = Field(x0, z0, h, wet, can, source=source, manmade=manmade, occupied=occupied,
              surface=surface, surface_palette=s_palette,
              biome=biome if biome_read else None,
              biome_palette=b_palette if biome_read else None)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        extra = ({"biome_codes": biome, "biome_palette": np.array(b_palette, dtype=str)}
                 if biome_read else {})
        np.savez_compressed(cache, x0=x0, z0=z0, h=h, wet=wet, canopy=can,
                            manmade=manmade, occupied=occupied,
                            surface_codes=surface,
                            surface_palette=np.array(s_palette, dtype=str), **extra)
    return f


# ---------------------------------------------------------------- the measures

def _window_relief(h: np.ndarray, p: int) -> np.ndarray:
    """The relief of every `p x p` window of `h`, as an array of the window origins.

        Written with a sliding view rather than a filter so it is exact and has no boundary
        convention to get wrong: a window is a window and there are `(n - p + 1)` of them.
        
    """
    if p > h.shape[0] or p > h.shape[1]:
        return np.array([[h.max() - h.min()]], np.int32)
    v = np.lib.stride_tricks.sliding_window_view(h, (p, p))
    return (v.max(axis=(2, 3)) - v.min(axis=(2, 3))).astype(np.int32)


def core_window(size: int, core: int) -> tuple:
    """The central `core x core` square of a footprint, as `(i, j)` into it. A1.

        The **middle**, and not the flattest window in it: `plateau` already answers "where
        is the best level ground anywhere in this candidate" and that is a different
        question. A city's core is where the palace goes, the rings are drawn round it, and
        a core measured off a flat corner would be a core the plan cannot use.
        
    """
    c = max(1, min(int(core), int(size)))
    o = (int(size) - c) // 2
    return (o, o, c)


def measure(field: Field, x0: int, z0: int, size: int, plateau: int,
            core: int | None = None, design: dict | None = None) -> dict | None:
    """One candidate footprint, on the terrain bank's measures plus its best plateau.

        A1 adds two more, and they are the whole of why a city can be sited at all: the
        relief of the **core** -- the middle of the footprint, where the thing at the centre
        stands -- and the relief of everything outside it. One number over 512x512 says
        nothing a plan can act on; these two say "level in the middle, hillside at the edge",
        which is what a city on real ground is.
        
    """
    got = field.sub(x0, z0, size, size)
    if got is None:
        return None
    h, wet, canopy, grav = got
    n = float(h.size)
    pr = _window_relief(h, min(plateau, size))
    best = int(pr.min())
    # Where the best plateau is, in world coordinates, and the flattest one at that
    # relief nearest the middle of the footprint -- so a place's centre is central.
    ii, jj = np.nonzero(pr == best)
    cx, cz = (size - plateau) / 2.0, (size - plateau) / 2.0
    k = int(np.lexsort((jj, ii, (ii - cx) ** 2 + (jj - cz) ** 2))[0])
    i, j = x0 - field.x0, z0 - field.z0
    count = int(field.manmade[i:i + size, j:j + size].sum())
    occupied = int(field.occupied[i:i + size, j:j + size].sum())
    ci, cj, cn = core_window(size, core or plateau)
    ch = h[ci:ci + cn, cj:cj + cn]
    cwet = wet[ci:ci + cn, cj:cj + cn]
    # The outer ring, as a masked view of the same array: every column of the footprint
    # that is not in the core. Written as a mask rather than four rectangles because a
    # ring is one measure and four corners would be four.
    mask = np.ones(h.shape, bool)
    mask[ci:ci + cn, cj:cj + cn] = False
    outer = h[mask]
    # The setting: what the ground is made of, as a share of the land, and the biome.
    # `read: False` is a square read before the cache carried either, and it is an
    # answer a setting need refuses by name rather than a zero it scores.
    if field.surface is not None:
        surface = groundread.surface_census(field.surface[i:i + size, j:j + size],
                                            field.surface_palette, wet)
    else:
        surface = {"read": False}
    biomes = None
    biome = {"read": False}
    if field.biome is not None:
        b = field.biome[i // BIOME_CELL:-(-(i + size) // BIOME_CELL),
                        j // BIOME_CELL:-(-(j + size) // BIOME_CELL)]
        counts = np.bincount(b.ravel(), minlength=len(field.biome_palette))
        biomes = {field.biome_palette[k]: round(100.0 * int(counts[k]) / max(1, b.size), 1)
                  for k in np.argsort(-counts)[:8] if counts[k]}
        # and, for a designed place, over the core and the inner rings alone.
        biome = groundread.biome_census(b, field.biome_palette)
        if design:
            w = int(design["inner_window"])
            c0 = (size // 2 - w) // BIOME_CELL
            c1 = -(-(size // 2 + w) // BIOME_CELL)
            bi = b[max(0, c0):c1, max(0, c0):c1]
            biome["inner"] = groundread.biome_census(bi, field.biome_palette)
            biome["inner"]["window"] = int(w)
    fill = fill_estimate(h, design, design.get("step", 4)) if design else None
    return {"x": int(x0), "z": int(z0), "size": int(size),
            "surface": surface, "biomes": biomes, "biome": biome, "fill": fill,
            "relief": int(h.max() - h.min()),
            "core": {"size": int(cn), "x": int(x0 + ci), "z": int(z0 + cj),
                     "relief": int(ch.max() - ch.min()),
                     "water_pct": round(100 * float(cwet.mean()), 1),
                     "y": int(np.median(ch))},
            "outer": {"columns": int(outer.size),
                      "relief": int(outer.max() - outer.min()) if outer.size else 0},
            "water_pct": round(100 * float(wet.mean()), 1),
            "forest_pct": round(100 * float(canopy.mean()), 1),
            "gravity_pct": (None if grav is None
                            else round(100 * float(grav.mean()), 1)),
            "y": [int(h.min()), int(h.max())],
            "mean_y": round(float(h.mean()), 1),
            "plateau": {"size": int(plateau), "relief": best,
                        "x": int(x0 + ii[k]), "z": int(z0 + jj[k]),
                        "y": int(np.median(h[ii[k]:ii[k] + plateau,
                                             jj[k]:jj[k] + plateau]))},
            "columns": int(n), "man_made_blocks": count, "occupied_blocks": occupied,
            "man_made_share": count / occupied if field.measured and occupied else None,
            "excluded_by": groundread.excluded(x0, z0, size, size)}


def excess(m: dict, needs: dict, *, plateau_relief: int,
           core_relief: float | None = None) -> dict:
    """How far past its needs a candidate is, per measure, as a fraction of the need.

        Zero on a measure means the candidate meets it. The sum is the score and lower is
        better; a candidate whose sum is zero **meets the needs** and the ranking among
        those is by how much room it has left, not by the score.
        
    """
    def over(got, cap):
        if got is None or cap in (None, 0):
            return 0.0
        return round(max(0.0, (float(got) - float(cap)) / float(cap)), 4)
    # A1: the **core's** ground is what the core's needs are read against. `core_relief`
    # is the place's `max_relief` unless the part at the centre declared its own, which
    # is the whole of the per-part needs change arriving where it is used. ...and the
    # **terraforming escape relaxes it**, because the core is precisely what a plateau
    # levels. A palace compound that declares it needs four blocks of relief is
    # declaring what it needs *to stand on*, not what the hillside has to be before
    # anybody cuts it -- and scoring the two the same way is a search that refuses every
    # site the escape exists to rescue. The caller passes what the attempt allows.
    core_cap = (needs.get("core_max_relief", needs["max_relief"])
                if core_relief is None else core_relief)
    designed = bool(needs.get("designed")) and m.get("fill") is not None
    e = {"relief": over(m["relief"], needs["max_relief"]),
         # **Where the ground is designed, water and the core are fill, not needs.**
         "core_relief": 0.0 if designed else over(m.get("core", {}).get("relief"), core_cap),
         "water": 0.0 if designed else over(m["water_pct"], needs["max_water_pct"]),
         # the core's own water against the need a centre part carries; zero where the
         # spec has no centre (`search_needs` sets no cap and `over` answers 0.0)
         "core_water": 0.0 if designed else over(m.get("core", {}).get("water_pct"),
                                                needs.get("core_water_max_pct")),
         "forest": over(m["forest_pct"], needs["max_forest_pct"]),
         "plateau": 0.0 if designed else over(m["plateau"]["relief"], plateau_relief),
         "fill": (over(m["fill"]["blocks"], needs.get("fill_max", DESIGNED_FILL_MAX))
                  if designed else 0.0),
         "designed": designed,
         "fill_preference": (round(float(m["fill"]["blocks"])
                                   / float(needs.get("fill_max", DESIGNED_FILL_MAX)), 6)
                             if designed else 0.0),
         # Gravity has no declared need: a site of sand is not refused, it is ranked
         # below one that is not, because a pad cut into it comes with a landslide.
         "gravity": round((m["gravity_pct"] or 0.0) / 100.0, 4),
         # ...and neither has the outer ring. See `OUTER_RELIEF`: a hard cap of 40 here
         # would refuse every site this world has, so an outer ring steeper than the
         # spec's number is ranked below one that is not and is never disqualified.
         "outer_relief": over(m.get("outer", {}).get("relief"),
                              needs.get("outer_max_relief", OUTER_RELIEF))}
    # **The setting.** A wanted surface is a need: the class's share of the land against
    # `SURFACE_SHARE`, short of it by a fraction the way every other need is over by
    # one. "Some water" is the same shape from below. A square whose surface was never
    # read is not scored at zero of it -- that is how gravity was misread for a whole
    # search -- it is refused with the reason named, as an unread census is.
    want = needs.get("surface")
    surface = m.get("surface") or {}
    e["setting_failures"] = []
    e["surface"] = 0.0
    e["setting_preference"] = 0.0
    if want:
        need = float(needs.get("surface_share", spec_mod.SURFACE_SHARE))
        if not surface.get("read"):
            e["setting_failures"].append(f"surface unread: the square was read before "
                                         f"the cache carried what its ground is made "
                                         f"of, and a {want} setting cannot be certified")
            e["setting_preference"] = 1.0
        else:
            got = float(surface.get("classes", {}).get(want, 0.0)) / 100.0
            e["surface"] = round(max(0.0, (need - got) / need), 4) if need else 0.0
            e["setting_preference"] = round(1.0 - got, 6)
            if e["surface"] > 0:
                e["setting_failures"].append(
                    f"surface {want} is {100 * got:.1f}% of the land against "
                    f"{100 * need:.0f}%")
    floor = needs.get("min_water_pct")
    e["water_min"] = (round(max(0.0, (float(floor) - float(m["water_pct"]))
                                / float(floor)), 4) if floor and not designed else 0.0)
    if designed and e["fill"] > 0:
        e["setting_failures"].append(f"fill {m['fill']['blocks']:,} blocks against "
                                     f"{int(needs.get('fill_max', DESIGNED_FILL_MAX)):,}")
    if e["water_min"] > 0:
        e["setting_failures"].append(f"water {m['water_pct']}% of the footprint "
                                     f"against at least {floor}%")
    # a square whose biome was never read is refused by name, as an unread surface is.
    # Among squares that meet, more of the list ranks higher, folded into the same
    # preference the surface ranks on so that a green savanna loses to a green plain.
    want_b = needs.get("biome")
    e["biome"] = 0.0
    if want_b:
        need_b = float(needs.get("biome_share", spec_mod.BIOME_SHARE))
        census = m.get("biome") or {}
        if designed and census.get("inner"):
            census = census["inner"]                 # the core and the inner rings
        if not census.get("read"):
            e["setting_failures"].append(f"biome unread: the square was read before "
                                         f"the cache carried its biomes, and a "
                                         f"{'/'.join(want_b)} setting cannot be "
                                         f"certified")
            e["setting_preference"] = round(e["setting_preference"] + 1.0, 6)
        else:
            got_b = groundread.biome_share(census, want_b)
            e["biome"] = round(max(0.0, (need_b - got_b) / need_b), 4) if need_b else 0.0
            e["setting_preference"] = round(e["setting_preference"] + (1.0 - got_b), 6)
            if e["biome"] > 0:
                e["setting_failures"].append(
                    f"biome {'/'.join(want_b)} is {100 * got_b:.1f}% of the cells "
                    f"against {100 * need_b:.0f}%")
    # **A floor on relief**, for a place that wants a mountain (`steep`): short of the
    # least fall by a fraction, the way every cap is over by one.
    rfloor = needs.get("min_relief")
    e["relief_min"] = (round(max(0.0, (float(rfloor) - float(m["relief"]))
                                 / float(rfloor)), 4) if rfloor else 0.0)
    if e["relief_min"] > 0:
        e["setting_failures"].append(f"relief {m['relief']} over the footprint against "
                                     f"at least {rfloor}")
    # ...and a preference for high ground with it: the mean level, negated, so that
    # among squares that meet a steep need the higher one ranks first. Zero for every
    # square when nothing asked, so the order is what it was.
    e["height_preference"] = (round(-float(m.get("mean_y") or 0.0), 1)
                              if needs.get("prefer_high") else 0.0)
    hard = ("relief", "core_relief", "water", "core_water", "forest", "plateau",
            "surface", "water_min", "biome", "relief_min", "fill")
    e["total"] = round(sum(e[k] for k in hard)
                       + 0.25 * e["gravity"] + OUTER_WEIGHT * e["outer_relief"], 4)
    e["meets"] = all(e[k] == 0.0 for k in hard) and not e["setting_failures"]
    if "man_made_share" in m:
        e["virgin_failures"] = groundread.virgin_failures(m)
        e["meets"] = e["meets"] and not e["virgin_failures"]
    if needs.get("relief_band"):
        lo, hi = needs["relief_band"]
        e["relief_band"] = [lo, hi]
        # **A1: the band is read on the core and not on the footprint.** A city's band
        # is 4-15 and no 512x512 of this world is within a hundred of it, so applied to
        # the whole footprint the preference was a constant offset that ordered
        # candidates by nothing. Applied to the middle -- which is where the palace goes
        # and where the ground is levelled -- it orders them by the thing it is about.
        got = m.get("core", {}).get("relief", m["relief"])
        e["relief_preference"] = round(abs(got - (lo + hi) / 2) / (hi - lo), 6)
        e["relief_preference_on"] = "core" if "core" in m else "footprint"
    return e


def rank_key(row: dict) -> tuple:
    """The total order candidates are ranked in. Every tie ends in coordinates.

        Meeting the needs first, then the score, then the flatness of the centre, then the
        ground itself, then **nearness to the origin** -- so of two equally good sites the
        search takes the one it found first -- and then x and z, which cannot tie.
        
    """
    m, e = row["measures"], row["excess"]
    # The setting before the relief band: of two squares that both meet a green need,
    # the greener one reads more as the place, and that is what a setting is for. Zero
    # for every square when no setting is wanted, so a spec with none ranks as it always
    # has. Of two squares that both meet the needs and read alike, the drier one is the
    # better ground for a place, and standing water is the one measure a plateau cannot
    # cut away. Relief on flat water is zero, which is how a lake ranked as the flattest
    # ground on the demo's site.
    return (0 if e["meets"] else 1, e["total"], e.get("setting_preference", 0.0),
            e.get("height_preference", 0.0),
            # a designed place ranks on the fill its terraces would lay, in the place
            # water took; water is inside that number
            e.get("fill_preference", 0.0),
            0.0 if e.get("designed") else m["water_pct"],
            e.get("relief_preference", 0),
            m["plateau"]["relief"], m["relief"],
            m["forest_pct"], m["gravity_pct"] or 0.0,
            m["x"] * m["x"] + m["z"] * m["z"], m["x"], m["z"])


# ------------------------------------------------------------------ the search

def candidates_at(radius: int, size: int, stride: int = STRIDE) -> list:
    """Every candidate origin on the lattice within `radius`, ordered and unique."""
    lo = -(radius // stride) * stride
    out = []
    for x in range(lo, radius + 1, stride):
        for z in range(lo, radius + 1, stride):
            # By the **centre**, so a footprint is inside the radius rather than its
            # corner being: a candidate whose middle is 4,000 out is a candidate at
            # 4,000 whatever its size.
            cx, cz = x + size / 2.0, z + size / 2.0
            if cx * cx + cz * cz <= float(radius) ** 2:
                out.append((x, z))
    return sorted(set(out))


def search_needs(spec: dict) -> dict:
    """The needs a candidate is scored against, per-part needs folded in. A1.

        One place, so the three searches -- the live scan, the cached scan and the fresh
        grid walk -- read the same answer, and so that a spec whose parts say nothing
        produces exactly the dictionary this file scored against before A1.
        
    """
    needs = dict(spec["needs"])
    needs["relief_band"] = groundread.RELIEF_BANDS.get(spec["kind"])
    core = spec_mod.core_needs(spec)
    needs["core_part"] = core.get("part")
    if core.get("max_relief") is not None:
        needs["core_max_relief"] = float(core["max_relief"])
    # The steepest thing any part that is not the core will take. `max()` and not
    # `min()`: this is the ring that is *allowed* to be a hillside, and the number is
    # the most permissive declaration among the parts that lie out there.
    outer = [p["needs"]["max_relief"] for p in spec_mod.outer_parts(spec)
             if (p.get("needs") or {}).get("max_relief") is not None]
    needs["outer_max_relief"] = float(max(outer)) if outer else float(OUTER_RELIEF)
    # The search scored whole-footprint water and never the core's, and a flat lake
    # *improves* every relief measure, so the demo's site was chosen with a lake at the
    # very point the rings are drawn round; the palace then slid ninety blocks to dodge
    # it. A part at `centre` is a declaration that the middle is built on, and the
    # middle is held to `spec.CORE_WATER_MAX_PCT` as a need, not a preference.
    c = spec_mod.core(spec)
    if c is not None and c.get("relation") == "centre":
        needs["core_water_max_pct"] = float(spec_mod.CORE_WATER_MAX_PCT)
    # The setting, folded into the needs the way every other need is: a surface the land
    # must predominantly be, and water there must be some of or none of. The words are
    # the spec's; the numbers are `spec`'s registered constants.
    setting = spec.get("setting") or {}
    needs["surface"] = setting.get("surface") or None
    needs["surface_share"] = float(spec_mod.SURFACE_SHARE)
    if setting.get("water") == "some":
        needs["min_water_pct"] = float(spec_mod.WATER_SOME_PCT)
    elif setting.get("water") == "none":
        needs["max_water_pct"] = min(float(needs["max_water_pct"]),
                                     float(spec_mod.WATER_NONE_PCT))
    # The biome list, the relief floor a `steep` word folded into the spec's needs, and
    # the preference for high ground that goes with it.
    needs["biome"] = list(setting.get("biome") or []) or None
    needs["biome_share"] = float(spec_mod.BIOME_SHARE)
    # **Designed ground**: the rings' half-sides from the shares, the fill budget and
    # the window the biome is read over, for a place the layout terraces. `footprint` is
    # the one need every spec carries -- except the spec a *failed* search answers
    # about, which is the caller's own stub and legitimately has no needs at all. A read
    # that assumes it is a `KeyError` on the one path that exists to say nothing met.
    want = spec.get("needs", {}).get("footprint")
    lay = designed_layout(spec, int(want)) if want else None
    if lay:
        from ethoslm.placeplan import TERRACE_STEP
        needs["designed"] = {**lay, "step": int(TERRACE_STEP)}
        needs["fill_max"] = float(DESIGNED_FILL_MAX)
        needs["biome_window"] = BIOME_INNER
    if spec["needs"].get("min_relief") is not None:
        needs["min_relief"] = float(spec["needs"]["min_relief"])
    needs["prefer_high"] = (setting.get("relief") or "") in spec_mod.RELIEF_FLOORS
    return needs


def setting_record(spec: dict, needs: dict) -> dict:
    """What the search was asked for, setting-wise, as it goes on the record."""
    setting = spec.get("setting") or {}
    return {"surface": setting.get("surface"), "water": setting.get("water"),
            "relief": setting.get("relief"), "biome": setting.get("biome"),
            "surface_share": needs.get("surface_share"),
            "biome_share": needs.get("biome_share") if needs.get("biome") else None,
            "designed": needs.get("designed"), "fill_max": needs.get("fill_max"),
            "biome_window": needs.get("biome_window"),
            "min_water_pct": needs.get("min_water_pct"),
            "max_water_pct": needs.get("max_water_pct"),
            "min_relief": needs.get("min_relief"),
            "max_relief": needs.get("max_relief"),
            "prefer_high": bool(needs.get("prefer_high")),
            "notes": setting.get("notes", "")}


def core_size(spec: dict) -> int:
    """How much of the middle is the core. A1."""
    c = spec_mod.core_needs(spec)
    want = max(int(c.get("plateau") or 0), _compound_plateau(spec)) or _plateau_size(spec)
    return int(min(int(want), int(spec["needs"]["footprint"])))


def _compound_plateau(spec: dict) -> int:
    """The ground a compound at the centre needs, or 0 where there is no compound there."""
    core = spec_mod.core(spec)
    if not core or not spec_mod.compound(core):
        return 0
    from ethoslm.placeplan import compound_ground
    return int(compound_ground(spec=spec,
                               site_side=spec["needs"]["footprint"])["side"])


def _plateau_size(spec: dict) -> int:
    """How big the level ground at the centre has to be, from the spec.

        The spec's own `needs.plateau` where it gives one; otherwise the ground the
        innermost defining part wants, which is `PLATEAU_DEFAULT` -- and never more than
        `Builder.plateau()` will cut, because a requirement the library refuses is not a
        requirement, it is a search that can never succeed.

        ...and never less than a **compound** at the centre needs (`_compound_plateau`),
        because that number is the library's and the spec's is a guess about a building.
        
    """
    # A1: the part at the centre may say how much level ground it needs, and it is the
    # one that knows -- a palace compound is not a market square. The place's own
    # `needs.plateau` still wins where it gives one, because that is the spec speaking
    # about the place rather than about a part of it.
    core = spec_mod.core(spec) or {}
    want = (spec["needs"].get("plateau")
            or (core.get("needs") or {}).get("plateau") or PLATEAU_DEFAULT)
    want = max(int(want), _compound_plateau(spec))
    S = int(spec["needs"]["footprint"])
    return int(min(want, Builder.plateau_max(S), S))


def innermost(spec: dict) -> dict | None:
    """The defining part a plateau would be cut for. A4 is bounded to this one.

        The one at the centre, and where two are, the one that is not a group: a palace at
        the centre is what the ground is levelled for and the districts round it follow the
        hill. Deterministic, and it names the part in the record rather than saying "the
        middle".
        
    """
    at_centre = [p for p in spec["defining_parts"] if p["relation"] == "centre"]
    solid = [p for p in at_centre if p["kind"] != "group"]
    pool = solid or at_centre or spec["defining_parts"]
    return sorted(pool, key=lambda p: (p["kind"] == "group", p["name"]))[0]


def search(spec: dict, field_for, *, radii=RADII, stride: int = STRIDE,
           log=print) -> dict:
    """The whole search. `field_for(x0, z0, w, d)` reads the ground; everything else
        here is arithmetic over what it returns.

        Returns the answer: the site chosen, the top `RECORDED` with their scores, which
        escape (if any) fired, and every radius that was scanned and what it found.
        
    """
    size = int(spec["needs"]["footprint"])
    plateau = _plateau_size(spec)
    needs = search_needs(spec)
    core = core_size(spec)
    rounds: list = []
    #: The two escapes the spec allows, in the order it allows them, and each is a thing
    #: the search *may do* rather than a thing it may choose to. `(label, plateau relief
    #: allowed, footprint, does it terraform)`: the first is held to what a plinth
    #: absorbs, the second to what a plateau can cut.
    flat = _flat_at_footprint(needs)
    dropped = _drop_band(spec)
    last: list = []

    def scan(label, allow, at_size, radius, terra):
        rows = []
        for (x, z) in candidates_at(radius, at_size, stride):
            f = field_for(x, z, at_size, at_size)
            if f is None:
                continue
            m = measure(f, x, z, at_size, min(plateau, at_size),
                        core=min(core, at_size), design=needs.get("designed"))
            if m is None:
                continue
            rows.append({"measures": m,
                         "excess": excess(m, needs, plateau_relief=allow)})
        rows.sort(key=rank_key)
        met = [r for r in rows if r["excess"]["meets"]]
        rounds.append({"attempt": label, "radius": radius, "footprint": at_size,
                       "candidates": len(rows), "meeting": len(met),
                       "plateau_relief_allowed": allow,
                       "best": rows[0]["measures"] if rows else None,
                       "best_excess": rows[0]["excess"] if rows else None})
        log(f"   {label}, radius {radius}: {len(rows)} candidates, "
            f"{len(met)} meet the needs"
            + (f", best ({rows[0]['measures']['x']},{rows[0]['measures']['z']}) "
               f"score {rows[0]['excess']['total']}" if rows else ""))
        return rows, met

    # **Both escapes at every radius before the radius grows, and the band is dropped
    # only when no radius has anything.** The order matters and it is not arbitrary:
    # searching wider keeps the place the sentence asked for and dropping a band does
    # not, so a town does not become a village while there is unlooked-at ground.
    for radius in radii:
        for label, allow, terra in (("as asked", flat, False),
                                    ("terraformed", PLATEAU_RELIEF, True)):
            rows, met = scan(label, allow, size, radius, terra)
            last = rows or last
            if met:
                return _answer(spec, label, size, plateau, rows, rounds, dropped,
                               allow, terraform=terra)
    if dropped:
        small = int(dropped["needs"]["footprint"])
        for radius in radii:
            rows, met = scan("one size band down", PLATEAU_RELIEF, small, radius, True)
            last = rows or last
            if met:
                return _answer(spec, "one size band down", small, plateau, rows, rounds,
                               dropped, PLATEAU_RELIEF, terraform=True)
    return _answer(spec, "nothing met the needs", size, plateau, last, rounds,
                   dropped, PLATEAU_RELIEF, terraform=True, failed=True)


def _flat_at_footprint(needs: dict) -> int:
    """The flatness a plateau's ground has to have before terraforming is on the table.

        `Builder.SITE_RELIEF` is what a plinth absorbs and a platform is what `site()` lays
        above it, so a place's centre that is flatter than a platform's threshold needs no
        plateau at all. That is the line the first attempt is held to.
        
    """
    return int(Builder.SITE_RELIEF)


def _drop_band(spec: dict) -> dict | None:
    """The spec one size band down, or None where there is no band below it.

        The last escape, and it is a real loss: a town that becomes a village is not the
        place that was asked for. It is recorded as such and the readout reports it.
        
    """
    # v2, C0: the order is the spec's one table (`SIZE_BANDS`) and is not declared here.
    lower = spec_mod.kind_below(spec["kind"])
    if lower is None:
        return None
    lo, hi = spec_mod.size_band_for(lower)
    n = min(int(spec["structures"]), hi)
    out = json.loads(json.dumps(spec))
    out["kind"], out["structures"] = lower, n
    out["size_band"] = [lo, hi]
    out["needs"]["footprint"] = spec_mod.footprint_for(n, lower)
    return out


def _compound_ground_record(spec: dict) -> dict | None:
    """Why the plateau is the size it is, where a compound decided it. Thread 38.

        `None` where the part at the centre is not a compound, which is every place that has
        no great thing in it -- so the record of a plain settlement's search is unchanged.
        
    """
    core = spec_mod.core(spec)
    if not core or not spec_mod.compound(core):
        return None
    from ethoslm.placeplan import compound_ground
    got = compound_ground(spec=spec, site_side=(spec.get("needs") or {}).get("footprint"))
    return {"part": core["name"], "family": core.get("family"), **got,
            "spec_asked": (core.get("needs") or {}).get("plateau")}


def _answer(spec, label, size, plateau, rows, rounds, dropped, allow,
            terraform: bool = False, failed: bool = False) -> dict:
    top = rows[:RECORDED]
    chosen = top[0] if top and not failed else None
    out = {"generated_by": "scripts/find_site.py",
           "sentence": spec["sentence"], "kind": spec["kind"],
           "footprint": int(size), "plateau_wanted": int(plateau),
           "stride": STRIDE, "radii": list(RADII),
           "needs": dict(spec["needs"]),
           "setting": setting_record(spec, search_needs(spec)),
           "compound_ground": _compound_ground_record(spec),
           "attempt": label, "failed": bool(failed),
           "terraform": None, "size_band_dropped": None,
           "rounds": rounds,
           "top": [{"rank": i + 1, **r["measures"], "excess": r["excess"]}
                   for i, r in enumerate(top)],
           "chosen": None}
    if terraform:
        p = innermost(spec)
        out["terraform"] = {
            "part": p["name"], "family": p["family"], "kind": p["kind"],
            "plateau": int(plateau), "relief_allowed": int(allow),
            # open thread 18 is two entries of exactly that -- and a city is big enough
            # that "terraform where you need to" would be a licence to flatten a quarter
            # of a square kilometre. One part, named here, before any of it is cut.
            "scope": "core",
            "why": ("no candidate within the radii is flat enough at the footprint, "
                    "so the innermost defining part -- and only that one -- is marked "
                    "for A4's plateau and the scan is scored again against what a "
                    "plateau can take")}
    if label == "one size band down" and dropped:
        out["size_band_dropped"] = {"from": spec["kind"], "to": dropped["kind"],
                                    "structures": dropped["structures"],
                                    "footprint": dropped["needs"]["footprint"]}
    if chosen:
        m = chosen["measures"]
        out["chosen"] = {"origin": [m["x"], m["z"]], "size": int(size),
                         "score": chosen["excess"]["total"],
                         "meets": chosen["excess"]["meets"],
                         "measures": m}
    return out


# ---------------------------------------------------------- gravity, on a shortlist

def add_gravity(ed, rows: list, log=print) -> list:
    """Measure gravity blocks on the shortlist, by reading each one's surface.

        The one place this file reads the world twice, and it is bounded: `SHORTLIST`
        candidates, a lattice of surface blocks each. A heightmap cannot answer what the
        ground is *made of* and a site of dune sand is a site every pad slides out of.
        
    """
    from gdpc.vector_tools import Rect
    from ethoslm import world
    for r in rows[:SHORTLIST]:
        m = r["measures"]
        if m.get("gravity_pct") is not None:
            continue                     # the surface census already answered it
        x, z, n = m["x"], m["z"], m["size"]
        ed.loadWorldSlice(Rect((x, z), (n, n)), cache=True)
        ws = ed.worldSlice
        h = ws.heightmaps["OCEAN_FLOOR"].astype(int) - 1
        step = max(1, n // 24)
        hits = tot = void = 0
        for i in range(0, n, step):
            for j in range(0, n, step):
                # **`getBlockGlobal`, not `getBlock`.** A world slice's `getBlock` takes
                # coordinates local to the slice; handing it world coordinates reads off
                # the end of the array and answers `void_air` for every column, which
                # looks exactly like a site with no gravity blocks in it.
                b = ws.getBlockGlobal((x + i, int(h[i, j]), z + j)).id.split(":")[-1]
                void += b in ("void_air", "")
                hits += b in GRAVITY or b.endswith("_concrete_powder")
                tot += 1
        m["gravity_pct"] = round(100 * hits / max(1, tot), 1)
        m["gravity_sampled"] = tot
        # A sample that is mostly nothing is a sample of the wrong array, and saying so
        # is cheaper than believing a zero.
        if void > tot // 2:
            m["gravity_pct"] = None
            m["gravity_unread"] = (f"{void} of {tot} sampled columns read as void: the "
                                   f"surface was not where this slice says it is")
        log(f"   gravity ({x},{z}): {m['gravity_pct']}% of {tot} sampled"
            + (f"  ({void} void)" if void else ""))
    return rows


# ----------------------------------------------------------------------- drivers

#: The cached worlds a `--cached` scan reads, and the pre-build cache of each. The same
#: list `scripts/terrain_bank.py` samples, by reference to the same fact: this is every
#: piece of ground this project has on disk.
CACHED_SITES = (("site_a", "world.npz"), ("site_b", "world.npz"),
                ("site_c", "world_prebuild.npz"), ("site_d", "world.npz"),
                ("site_e", "world.npz"), ("site_f", "world.npz"))


def cached_fields(log=print) -> list:
    out = []
    for name, cache in CACHED_SITES:
        p = offline.world_cache(name, cache)
        if not os.path.exists(p):
            log(f"   {name}: no {cache} on disk, skipped")
            continue
        out.append((name, field_from_volume(offline.load_volume(p))))
    return out


def search_cached(spec: dict, stride: int = 32, log=print) -> dict:
    """Rank every candidate footprint in every cached world. Deterministic.

        A stride of 32 rather than 256, because a cached site is 288 blocks square and a
        256 lattice puts one candidate in it. The rule is the same and so is the order; what
        changes is how finely the same ground is offered.
        
    """
    fields = cached_fields(log)
    size = int(spec["needs"]["footprint"])
    plateau = _plateau_size(spec)
    needs = search_needs(spec)
    core = core_size(spec)
    rows = []
    for name, f in fields:
        w, d = f.shape
        for i in range(0, w - size + 1, stride):
            for j in range(0, d - size + 1, stride):
                x, z = f.x0 + i, f.z0 + j
                m = measure(f, x, z, size, min(plateau, size), core=core,
                        design=needs.get("designed"))
                if m is None:
                    continue
                m["site"] = name
                rows.append({"measures": m,
                             "excess": excess(m, needs,
                                              plateau_relief=_flat_at_footprint(needs))})
    rows.sort(key=rank_key)
    return {"generated_by": "scripts/find_site.py --cached",
            "sentence": spec["sentence"], "footprint": size,
            "plateau_wanted": plateau, "stride": stride,
            "setting": setting_record(spec, needs),
            "sites": [n for n, _f in fields], "candidates": len(rows),
            "meeting": sum(1 for r in rows if r["excess"]["meets"]),
            "top": [{"rank": i + 1, **r["measures"], "excess": r["excess"]}
                    for i, r in enumerate(rows[:RECORDED])],
            "chosen": ({"site": rows[0]["measures"]["site"],
                        "origin": [rows[0]["measures"]["x"], rows[0]["measures"]["z"]],
                        "size": size, "score": rows[0]["excess"]["total"],
                        "meets": rows[0]["excess"]["meets"]} if rows else None)}


def _score(rows: list, needs: dict, attempts, force: bool = False) -> str | None:
    """Score every row under each attempt in turn; stop at the first that meets. A5.

        Returns the label of the attempt that produced a candidate, or None. `force` scores
        under the last attempt anyway, so a failed search still has a ranked list to report
        rather than a list with no scores in it.
        
    """
    for label, plateau_relief, core in attempts:
        for r in rows:
            r["excess"] = excess(r["measures"], needs,
                                 plateau_relief=plateau_relief, core_relief=core)
            r["attempt"] = label
        if any(r["excess"]["meets"] for r in rows):
            return label
    return attempts[-1][0] if force and rows else None


def _terraform(spec: dict, plateau: int, allow: int) -> dict:
    """What the terraforming escape marks, as it goes on the record. A5."""
    p = innermost(spec)
    return {"part": p["name"], "family": p["family"], "kind": p["kind"],
            "plateau": int(plateau), "relief_allowed": int(allow), "scope": "core",
            "why": ("no square within the radii is flat enough at its core as asked, "
                    "so the part at the centre -- and only that one -- is marked for "
                    "A4's plateau and every square is scored again against what a "
                    "plateau can take")}


# Reading less The concentric run's search cost four hours and twelve minutes: 700 tile
# reads through GDMC-HTTP at fifteen seconds each, most of them re-reads of squares the
# region files already held at half a second apiece, and 48 squares of fresh ground
# generated as a lottery at four minutes each. Everything below is the search reading
# less: a square assembled from the squares already cached, a biome share read off the
# chunk NBT before a square is parsed into a field, the world asked where the biome *is*
# before any ground is made, and a preflight that says what all of it will cost before
# any of it is spent.

#: **Registered.** What one read costs, per 512x512 square, measured on the concentric
#: run's own search log. The preflight scales both by area and prints the sum.
COST_FILE_READ_S = 15.0
#: ...re-measured on the ground run's own first live session: six 768x768 squares of
#: fresh ground read through the server in 19 to 51 s each, which is 30 s per 512 square
#: and not the 240 the concentric log suggested (that log's four minutes a square were
#: the HTTP re-reads of squares the files held, which the search no longer makes). Its
#: preflight refused a radius at 490 min that the measured cost puts under two hours;
#: re-registered from the measurement before the second session.
COST_GENERATE_S = 30.0

#: **Registered.** How long a search may say it is about to take before it refuses to
#: start: two hours. The concentric search took over four and had to be run three times;
#: a search that will not finish in two is a search whose plan is wrong, and the answer
#: is to read less (a biome preference, a smaller radius, a smaller `max_new`), not to
#: wait. A round may raise it by name (`search_bound_seconds`).
SEARCH_BOUND_S = 7200.0

#: The biome ids the game has for each of `groundread.BIOMES`' classes, for the `/locate
#: biome` question. The classes are the setting's words; these are what the world
#: answers to.
LOCATE_IDS = {
    "plains": ("plains", "sunflower_plains", "meadow"),
    "savanna": ("savanna", "savanna_plateau"),
    "forest": ("forest", "birch_forest", "dark_forest", "taiga"),
    "jungle": ("jungle", "sparse_jungle"),
    "desert": ("desert",),
    "badlands": ("badlands", "wooded_badlands"),
    "swamp": ("swamp", "mangrove_swamp"),
    "snowy": ("snowy_plains", "snowy_taiga", "snowy_slopes"),
    "mountain": ("windswept_hills", "stony_peaks", "jagged_peaks"),
}


def cached_sizes(directory: str | None = None) -> list:
    """Every square size the cache directory holds, largest first."""
    import re
    d = directory or SITES
    if not os.path.isdir(d):
        return []
    out = set()
    for f in os.listdir(d):
        m = re.match(r"ground_(-?\d+)_(-?\d+)_(\d+)x(\d+)\.npz$", f)
        if m and m.group(3) == m.group(4):
            out.add(int(m.group(3)))
    return sorted(out, reverse=True)


def composable(x: int, z: int, size: int, directory: str | None = None,
               stride: int = STRIDE) -> list | None:
    """The cached squares that tile `(x, z, size)`, or None where none do.

        A 768 square on the 256 lattice is four cached 512 squares -- (x, z), (x+256, z),
        (x, z+256), (x+256, z+256) -- and every one of them is ground the search has read
        before. Returns `[(sx, sz, s), ...]` for the first cached size that tiles it.
        
    """
    for s in cached_sizes(directory):
        if s > size or (size - s) % stride:
            continue
        n = (size - s) // stride + 1
        subs = [(x + i * stride, z + j * stride, s) for i in range(n) for j in range(n)]
        if all(_cached(sx, sz, s, directory) for (sx, sz, s) in subs):
            return subs
    return None


def compose_from_cache(x: int, z: int, size: int, directory: str | None = None,
                       stride: int = STRIDE, log=print) -> "Field | None":
    """A square assembled from the cached squares that tile it, and cached itself.

        The ground is the same ground: two cached squares that overlap were read off the
        same chunks, so where they overlap either copy is the answer. Palettes are remapped
        per sub-square; biome cells are 4x4 and the lattice is a multiple of 4, so they
        align. None where nothing tiles it, or where a sub-square is stale (no surface or
        no biomes): a composed square carries everything a read one does or it is not made.
        
    """
    subs = composable(x, z, size, directory, stride)
    if not subs:
        return None
    h = np.zeros((size, size), np.int32)
    wet = np.zeros((size, size), bool)
    can = np.zeros((size, size), bool)
    manmade = np.zeros((size, size), np.int64)
    occupied = np.zeros((size, size), np.int64)
    surface = np.zeros((size, size), np.int32)
    s_pal: list = []
    s_idx: dict = {}
    cw = -(-size // BIOME_CELL)
    biome = np.zeros((cw, cw), np.int32)
    b_pal: list = []
    b_idx: dict = {}
    seen = np.zeros((size, size), bool)
    for (sx, sz, s) in subs:
        got = np.load(tile_cache(sx, sz, s, s, directory))
        if "surface_codes" not in got or "biome_codes" not in got:
            return None
        i, j = sx - x, sz - z
        sl = (slice(i, i + s), slice(j, j + s))
        # Two cached squares that overlap were read off the same chunks -- unless one
        # was read after something was built there. Where they disagree the cache is not
        # the ground, and the square is read off the files instead.
        both = seen[sl]
        if both.any() and not np.array_equal(h[sl][both], got["h"][both]):
            log(f"   ({x},{z}) {size}x{size} not assembled: the cached "
                f"({sx},{sz}) {s}x{s} disagrees with its neighbour on "
                f"{int((h[sl][both] != got['h'][both]).sum())} column(s)")
            return None
        seen[sl] = True
        h[sl] = got["h"]
        wet[sl] = got["wet"]
        can[sl] = got["canopy"]
        manmade[sl] = got["manmade"]
        occupied[sl] = got["occupied"]
        remap = np.zeros(len(got["surface_palette"]), np.int32)
        for k, name in enumerate(got["surface_palette"]):
            name = str(name)
            if name not in s_idx:
                s_idx[name] = len(s_pal)
                s_pal.append(name)
            remap[k] = s_idx[name]
        surface[sl] = remap[got["surface_codes"]]
        bremap = np.zeros(len(got["biome_palette"]), np.int32)
        for k, name in enumerate(got["biome_palette"]):
            name = str(name)
            if name not in b_idx:
                b_idx[name] = len(b_pal)
                b_pal.append(name)
            bremap[k] = b_idx[name]
        bc = got["biome_codes"]
        bsl = (slice(i // BIOME_CELL, i // BIOME_CELL + bc.shape[0]),
               slice(j // BIOME_CELL, j // BIOME_CELL + bc.shape[1]))
        biome[bsl] = bremap[bc]
    cache = tile_cache(x, z, size, size, directory)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, x0=x, z0=z, h=h, wet=wet, canopy=can, manmade=manmade,
                        occupied=occupied, surface_codes=surface,
                        surface_palette=np.array(s_pal, dtype=str),
                        biome_codes=biome, biome_palette=np.array(b_pal, dtype=str))
    log(f"   ({x},{z}) {size}x{size} assembled from {len(subs)} cached "
        f"{subs[0][2]}x{subs[0][2]} square(s)")
    return Field(x, z, h, wet, can, source="composed from the cache",
                 manmade=manmade, occupied=occupied,
                 surface=surface, surface_palette=s_pal,
                 biome=biome, biome_palette=b_pal)


def _unpack(longs, bits: int, count: int) -> np.ndarray:
    """`count` entries of `bits` bits each, packed `64 // bits` to a long, no
    straddling -- the game's packing since 1.16."""
    a = np.asarray(longs, dtype=np.int64).astype(np.uint64)
    per = 64 // bits
    k = np.arange(per, dtype=np.uint64) * np.uint64(bits)
    vals = (a[:, None] >> k[None, :]) & np.uint64((1 << bits) - 1)
    return vals.ravel()[:count].astype(np.int64)


def chunk_surface_biomes(tag) -> list:
    """The biome of each 4x4 cell of one chunk's surface, off its NBT alone.

        The `MOTION_BLOCKING` heightmap says which section the surface is in per column;
        that section's biome palette -- one entry, the whole section; more, a packed
        index per 4x4x4 cell -- says what it is. No block is decoded. This is the read
        the biome prefilter is made of: three milliseconds a chunk against the fifteen
        seconds a square costs to parse into a field.
        
    """
    hm = tag["Heightmaps"]["MOTION_BLOCKING"]
    heights = _unpack([v for v in hm.value], 9, 256)   # value = y - minY + 1
    y_min = int(tag["yPos"].value) * 16 if "yPos" in tag else -64
    sections = {int(sec["Y"].value): sec for sec in tag["sections"]}
    out = []
    for cz in range(4):
        for cx in range(4):
            v = int(heights[(cz * 4) * 16 + cx * 4])
            y = y_min + v - 1
            sec = sections.get(y >> 4)
            if sec is None or "biomes" not in sec:
                out.append("")
                continue
            pal = [str(p.value).split(":")[-1] for p in sec["biomes"]["palette"]]
            if len(pal) == 1 or "data" not in sec["biomes"]:
                out.append(pal[0])
                continue
            bits = max(1, int(math.ceil(math.log2(len(pal)))))
            idx = _unpack([q for q in sec["biomes"]["data"].value], bits, 64)
            by = (y & 15) >> 2
            out.append(pal[int(idx[(by * 4 + cz) * 4 + cx])])
    return out


def biome_share_from_regions(regs, x0: int, z0: int, size: int, want) -> dict:
    """The share of a square's surface cells in the biome classes `want`, off the
    chunk NBT and nothing else; `read: False` where any chunk is not finished ground."""
    from ethoslm import regions as _regions
    counts: dict = {}
    total = 0
    t0 = time.perf_counter()
    try:
        for cx in range(x0 >> 4, ((x0 + size - 1) >> 4) + 1):
            for cz in range(z0 >> 4, ((z0 + size - 1) >> 4) + 1):
                for b in chunk_surface_biomes(regs.chunk(cx, cz)):
                    counts[b] = counts.get(b, 0) + 1
                    total += 1
    except _regions.NotGenerated as e:
        return {"read": False, "why": str(e)}
    pal = sorted(counts)
    codes = np.repeat(np.arange(len(pal)), [counts[b] for b in pal])
    census = groundread.biome_census(codes, pal)
    return {"read": True, "share": groundread.biome_share(census, want),
            "classes": census["classes"], "top": census["top"][:4],
            "seconds": round(time.perf_counter() - t0, 1)}


class Console:
    """The server console: a command in through the FIFO, its answer read back off
    the log. `scripts/server.sh cmd` is the same FIFO; this waits for the line."""

    def __init__(self, fifo: str | None = None, log_path: str | None = None):
        srv = os.path.join(ROOT, "run", "server")
        self.fifo = fifo or os.path.join(srv, "console.in")
        self.log_path = log_path or os.path.join(srv, "console.log")

    def ask(self, command: str, pattern: str, timeout: float = 20.0) -> str | None:
        """Send `command`; return the first new log line matching `pattern`, or None."""
        import re
        with open(self.log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            at = f.tell()
        with open(self.fifo, "w") as f:
            f.write(command.strip() + "\n")
        rx = re.compile(pattern)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with open(self.log_path, "rb") as f:
                f.seek(at)
                for raw in f.read().decode("utf-8", "replace").splitlines():
                    if rx.search(raw):
                        return raw
            time.sleep(0.25)
        return None


#: What `/locate biome` answers, and how it is read. One pattern, kept beside the
#: function that reads it, so the case that exercises the locate path against a recorded
#: answer exercises the same pattern the live path reads.
LOCATE_ANSWER = r"The nearest ([\w:]+) is at \[(-?\d+), (~|-?\d+), (-?\d+)\] \((\d+) blocks away\)"


def parse_locate(line: str) -> tuple | None:
    """`(biome_id, x, z, distance)` off a console answer, or None."""
    import re
    m = re.search(LOCATE_ANSWER, line or "")
    if not m:
        return None
    return (m.group(1).split(":")[-1], int(m.group(2)), int(m.group(4)), int(m.group(5)))


def locate_biomes(ask, origins: list, classes, log=print) -> list:
    """Ask the world where the preferred biomes are, from each origin. `ask(cmd,
    pattern)` is `Console.ask` or a recorded stand-in. Returns the points found,
    each `{"biome", "x", "z", "from": [ox, oz], "distance"}`, deduplicated."""
    ids = [i for c in (classes or []) for i in LOCATE_IDS.get(c, ())]
    found: dict = {}
    for (ox, oz) in origins:
        for bid in ids:
            line = ask(f"execute positioned {int(ox)} 64 {int(oz)} run locate biome "
                       f"minecraft:{bid}", LOCATE_ANSWER + "|Could not find")
            got = parse_locate(line or "")
            if got is None:
                continue
            b, x, z, d = got
            key = (b, x, z)
            if key not in found or d < found[key]["distance"]:
                found[key] = {"biome": b, "x": x, "z": z, "from": [int(ox), int(oz)],
                              "distance": d}
    out = sorted(found.values(), key=lambda p: (p["distance"], p["x"], p["z"]))
    log(f"   locate: {len(out)} point(s) of {'/'.join(classes or [])} from "
        f"{len(origins)} origin(s)")
    return out


def preflight_search(plan: dict, size: int, bound: float = SEARCH_BOUND_S,
                     log=print) -> dict:
    """What a radius of the search is about to cost, printed before it is spent, and
    refused over `bound`. `plan` counts squares by how they will be read."""
    area = (float(size) / 512.0) ** 2
    est = (plan.get("files", 0) * COST_FILE_READ_S * area
           + plan.get("generate", 0) * COST_GENERATE_S * area
           + plan.get("biome_reads", 0) * 3.0 * area)
    out = {**plan, "size": int(size), "estimated_seconds": round(est),
           "bound_seconds": float(bound), "refused": est > bound,
           "registered": {"COST_FILE_READ_S": COST_FILE_READ_S,
                          "COST_GENERATE_S": COST_GENERATE_S,
                          "SEARCH_BOUND_S": SEARCH_BOUND_S}}
    log(f"   preflight: {plan.get('cached', 0)} cached, {plan.get('composed', 0)} to "
        f"assemble, {plan.get('biome_reads', 0)} biome reads, {plan.get('files', 0)} to "
        f"read off the files, {plan.get('generate', 0)} to generate at {size}x{size}: "
        f"about {est / 60:.0f} min against a bound of {bound / 60:.0f}"
        + (" -- REFUSED" if out["refused"] else ""))
    return out


def search_fresh(spec: dict, *, editor=None, radii=None, stride: int = STRIDE,
                 directory: str | None = None, max_new: int = 0, regions=None,
                 workers: int | None = None, log=print, console=None,
                 bound: float = SEARCH_BOUND_S) -> dict:
    """Walk the grid square by square, reading as little ground as the answer needs.

        The old scan read the world **once per radius** -- one array the size of the whole
        ring, assembled in one pass -- which is right when the ring is already generated and
        impossible when it is not: a 4,096-radius read of ungenerated world is the server
        making eight thousand region files before anything is scored.

        So this walks the grid instead. One square per candidate footprint, in the order the
        lattice gives them; each is, in order of cost:

          1. read from `out/sites/` if it has ever been read at this size;
          2. **assembled** from cached squares that tile it (`compose_from_cache`);
          3. where the region files hold it, held to a **biome prefilter** off the chunk
             NBT where the spec prefers biomes -- a square under `spec.BIOME_SHARE` of the
             list is recorded and never parsed into a field -- and otherwise read off the
             files, across `workers` processes, whether or not a session is open: **nothing
             is read through HTTP that the files hold**;
          4. where the files do not hold it and a session is open, **generated**, within
             `max_new` and -- where a biome is preferred and a `console` is given -- only
             where `/locate biome` from the lattice says the biome is;
          5. otherwise skipped and recorded.

        Before a radius costs anything the preflight prints the counts and their measured
        cost and refuses over `bound`. **When nothing meets the needs, nothing is chosen**:
        the record is written, `chosen` is None and the caller says so. `max_new` is how
        many ungenerated squares this session may ask the server for and defaults to none:
        making ground is a change to the world save it cannot undo. `regions` is an
        `ethoslm.regions.Regions`, the save's own files; None reads them from `REGIONS`.
        
    """
    from ethoslm import regions as _regions
    size = int(spec["needs"]["footprint"])
    plateau = _plateau_size(spec)
    needs = search_needs(spec)
    core = core_size(spec)
    files = regions if regions is not None else _regions.Regions()
    have = generated_regions(files.directory)
    rows, seen = [], {"cache": 0, "composed": 0, "server": 0, "files": 0, "unread": 0,
                      "fresh": 0, "over_cap": 0, "stale": 0, "unfinished": 0,
                      "refused_by_biome": 0, "not_located": 0}
    unread, generated, refused_biome, located = [], [], [], []
    scored: set = set()
    fetched: set = set()
    want_b = needs.get("biome")
    preflights = []
    attempts = (("as asked", _flat_at_footprint(needs),
                 needs.get("core_max_relief", needs["max_relief"])),
                ("terraformed", PLATEAU_RELIEF, float(PLATEAU_RELIEF)))
    attempt = None
    refused = None
    for radius in (radii or RADII):
        todo = [(x, z) for (x, z) in candidates_at(radius, size, stride)
                if (x, z) not in scored]
        covered = {(x, z): covers(have, x, z, size, size)
                   and (_cached(x, z, size, directory)
                        or composable(x, z, size, directory, stride) is not None
                        or finished_square(files, x, z, size))
                   for (x, z) in todo}
        cached = [q for q in todo if covered[q] and _cached(q[0], q[1], size, directory)]
        compose = [q for q in todo if covered[q] and q not in cached
                   and composable(q[0], q[1], size, directory, stride)]
        to_read = [q for q in todo if covered[q] and q not in cached and q not in compose]
        fresh = [q for q in todo if not covered[q]]
        # **Where the biome is, before any ground is made.** With a session and a
        # console, the world names the nearest preferred biome from each lattice origin
        # at this radius, and only the fresh squares that cover a point named are
        # candidates for generation; without a console, generation stays the lottery it
        # was, within `max_new`.
        gen = []
        if fresh and editor is not None and max_new > seen["fresh"]:
            if want_b and console is not None:
                pts = locate_biomes(console.ask,
                                    [q for q in candidates_at(radius, size, 2 * stride)
                                     if not covered.get(q, True)] or fresh[:8],
                                    want_b, log=log)
                located += pts
                gen = [q for q in fresh if any(q[0] <= p["x"] < q[0] + size
                                               and q[1] <= p["z"] < q[1] + size
                                               for p in pts)]
                seen["not_located"] += len(fresh) - len(gen)
            else:
                gen = list(fresh)
            gen = gen[:max(0, max_new - seen["fresh"])]
        pre = preflight_search({"radius": radius, "cached": len(cached),
                                "composed": len(compose),
                                "biome_reads": len(to_read) if want_b else 0,
                                "files": len(to_read), "generate": len(gen)},
                               size, bound=bound, log=log)
        preflights.append(pre)
        if pre["refused"]:
            refused = pre
            break
        for (x, z) in compose:
            compose_from_cache(x, z, size, directory, stride, log=log)
            seen["composed"] += 1
        # The biome prefilter, then the parse, across processes.
        want_read = []
        for (x, z) in to_read:
            if want_b:
                b = biome_share_from_regions(files, x, z, size, want_b)
                if not b.get("read"):
                    # unfinished ground after all: recorded, never parsed
                    seen["unfinished"] += 1
                    seen["unread"] += 1
                    unread.append([x, z])
                    scored.add((x, z))
                    continue
                if b["share"] < float(needs["biome_share"]):
                    seen["refused_by_biome"] += 1
                    refused_biome.append({"x": x, "z": z, "share": b["share"],
                                          "top": b["top"]})
                    scored.add((x, z))
                    continue
            want_read.append((x, z))
        if want_read:
            log(f"   radius {radius}: reading {len(want_read)} square(s) off the "
                f"region files")
            _prefetch(want_read, size, directory, n=workers, log=log)
            fetched.update(want_read)
        for (x, z) in todo:
            if (x, z) in scored:
                continue
            scored.add((x, z))
            is_fresh = not covered[(x, z)]
            allowed = editor if (is_fresh and (x, z) in gen) else None
            f, source = read_tile(x, z, size, size, editor=allowed,
                                  regions=files if not is_fresh else None,
                                  directory=directory, log=log)
            if f is None:
                seen["unread"] += 1
                seen["over_cap"] += bool(is_fresh and editor is not None
                                         and (x, z) not in gen)
                seen["unfinished"] += source == "unfinished"
                unread.append([x, z])
                continue
            if (x, z) in fetched and source == "cache":
                seen["files"] += 1
            elif (x, z) in compose and source == "cache":
                pass                                   # counted when assembled
            else:
                seen[source] += 1
            if is_fresh and source == "server":
                seen["fresh"] += 1
                generated.append([x, z])
                log(f"   ({x},{z}) is ground the world had not made: "
                    f"{seen['fresh']} of {max_new} allowed")
            m = measure(f, x, z, size, min(plateau, size), core=core,
                        design=needs.get("designed"))
            if m is None:
                continue
            m["ground_was_generated_for_this_scan"] = bool(is_fresh
                                                           and source == "server")
            rows.append({"measures": m, "excess": None})
        attempt = _score(rows, needs, attempts)
        if attempt:
            break
    if not attempt:
        _score(rows, needs, attempts, force=True)
    rows.sort(key=rank_key)
    unmeasured = [r for r in rows if r["measures"]["gravity_pct"] is None]
    gravity = {"from_surface": len(rows) - len(unmeasured)}
    if editor is not None and unmeasured:
        try:
            add_gravity(editor, unmeasured, log=log)
            _score(rows, needs, attempts[:1] if attempt == attempts[0][0]
                   else attempts, force=True)
            rows.sort(key=rank_key)
            gravity.update(shortlist=min(SHORTLIST, len(unmeasured)), read=True)
        except Exception as e:                # noqa: BLE001 -- reported, not raised
            gravity.update(read=False, why=f"{type(e).__name__}: {e}")
    elif unmeasured:
        gravity.update(read=False, unmeasured=len(unmeasured),
                       why="no session: gravity is not in a heightmap, and a square "
                           "read before its cache carried a surface is scored at zero "
                           "of it and the answer says so")
    else:
        gravity.update(read=True,
                       why="every square's gravity share is its surface census's, "
                           "over the whole square")
    best = rows[0] if rows else None
    meets = bool(best and best["excess"]["meets"])
    if rows and not meets:
        log(f"   nothing met the needs over {len(rows)} square(s): the best, "
            f"({best['measures']['x']},{best['measures']['z']}) at score "
            f"{best['excess']['total']}, fails {failed_needs(best['excess'])}; "
            f"no site is chosen")
    return {"generated_by": "scripts/find_site.py --fresh",
            "sentence": spec["sentence"], "kind": spec["kind"], "footprint": size,
            "plateau_wanted": plateau, "stride": stride,
            "setting": setting_record(spec, needs),
            "compound_ground": _compound_ground_record(spec),
            "attempt": attempt or "nothing met the needs",
            "attempts": [a[0] for a in attempts],
            "terraform": (_terraform(spec, plateau, PLATEAU_RELIEF)
                          if attempt and attempt != attempts[0][0] else None),
            "radii": list(radii or RADII),
            "regions": region_span(have),
            "read_by": ("the region files, and a server session for ground they do "
                        "not hold" if editor is not None else
                        f"the region files under "
                        f"{os.path.relpath(files.directory, ROOT)}, no server"),
            "squares": seen, "unread": unread[:32], "max_new": int(max_new),
            "gravity": gravity,
            "generated": generated,
            "located": located[:64],
            "refused_by_biome": refused_biome[:64],
            "preflight": preflights, "refused": refused,
            "cache": os.path.relpath(directory or SITES, ROOT),
            "candidates": len(rows),
            "meeting": sum(1 for r in rows if r["excess"]["meets"]),
            "top": [{"rank": i + 1, **r["measures"], "excess": r["excess"]}
                    for i, r in enumerate(rows[:RECORDED])],
            # This chose the least excess when nothing met, and a city was built on a
            # square its own record said was 25.7% water against a cap of 25. A need is
            # a need.
            "chosen": ({"origin": [best["measures"]["x"], best["measures"]["z"]],
                        "size": size, "score": best["excess"]["total"],
                        "meets": True,
                        "measures": best["measures"]} if meets else None),
            "best_failed": ({"origin": [best["measures"]["x"], best["measures"]["z"]],
                             "score": best["excess"]["total"],
                             "failures": failed_needs(best["excess"]),
                             "excess": best["excess"]} if rows and not meets else None)}


#: a concentric place: rings terraced to levels and a podium filled -- standing water
#: and the core's relief are not needs of a site, they are **fill**: every column of
#: every ring is brought to its level from the bed by the terraces stage, and the podium
#: is a fill, not a cut. So a designed place is held to its biome, its surface, its
#: forest and its whole-footprint relief, and the fill its terraces would lay is
#: estimated off the heightmap the way `stages_plan.preflight_terraces` estimates it off
#: the volume, held to `DESIGNED_FILL_MAX` -- the same twelve million blocks the
#: terraces stage refuses over -- and ranked lower-is-better in the place water took.
#: The first ground run held 48 plains squares to water and the core and refused them
#: all.
DESIGNED_FILL_MAX = 12_000_000

#: ...and its biome share is read **over the core and the inner rings**, not the belt:
#: the belt is fields whatever biome it stands in, and a city whose rings are plains on
#: a square whose outer band is a river's is a city on a plain. The window is the square
#: of the boundary between the second-outermost ring and the outermost, from the shares.
BIOME_INNER = "core and inner rings"


def designed_layout(spec: dict, size: int) -> dict | None:
    """The ring half-sides, from the shares, as the layout will draw them; None for a
    place that designs no ground. `{"halves": [...innermost first...], "inner_window":
    half-side over which the biome is read, "n": rings}`."""
    rings = spec_mod.rings(spec)
    if not rings:
        return None
    from ethoslm.placeplan import RING_EDGE_INSET
    cum = spec_mod.centre_share(spec)
    halves = []
    for r in rings:
        cum += float(r["share"])
        halves.append(int(round(size * math.sqrt(min(1.0, cum)) / 2.0)))
    halves[-1] = size // 2 - RING_EDGE_INSET
    inner = halves[-2] if len(halves) >= 2 else halves[-1]
    return {"halves": halves, "inner_window": int(inner), "n": len(rings)}


def fill_estimate(h: np.ndarray, layout: dict, step: int) -> dict:
    """The blocks the terraces and the podium would lay on this footprint: per ring
    annulus, the level less the bed where the bed is lower (plus the cover course) and
    the bed less the level where higher, the outermost ring at the median and each ring
    in `step` higher, the podium a step above the innermost."""
    n = h.shape[0]
    cx = cz = n // 2
    med = int(np.median(h))
    halves = layout["halves"]
    k_n = len(halves)
    xs = np.arange(n) - cx
    cheb = np.maximum(np.abs(xs)[:, None], np.abs(xs)[None, :])
    total = 0
    per = []
    inner = 0
    for k, hh in enumerate(halves):
        level = med + (k_n - 1 - k) * int(step)
        mask = (cheb > inner) & (cheb <= hh)
        band = h[mask]
        fill = int(np.clip(level - band, 0, None).sum()) + int(mask.sum())
        cut = int(np.clip(band - level, 0, None).sum())
        per.append({"ring": k, "level": level, "fill": fill, "cut": cut})
        total += fill + cut
        inner = hh
    return {"blocks": int(total), "median": med, "per_ring": per}


#: The keys of a search record that are about **this run's cost** and not its answer:
#: which squares were read from where and what the preflight said. A cold run assembles
#: a square the warm run reads from the cache; the answer is the same.
COST_KEYS = ("squares", "preflight", "read_by", "cache", "unread", "gravity")


def answer_of(record: dict) -> dict:
    """The search's answer without its cost record, for a check that runs it twice."""
    return {k: v for k, v in record.items() if k not in COST_KEYS}


#: The measures `excess` scores as needs: over any of them and a square does not meet.
HARD_NEEDS = ("relief", "core_relief", "water", "core_water", "forest", "plateau",
              "surface", "water_min", "biome", "relief_min", "fill")


def failed_needs(e: dict) -> list:
    """Which needs a scored square fails, by name, for the record and the log."""
    out = [f"{k} over by {e[k]:.0%}" for k in HARD_NEEDS if float(e.get(k) or 0) > 0]
    return out + list(e.get("setting_failures") or []) + list(e.get("virgin_failures") or [])


#: The spec `--cached --check` ranks against. A fixed one, written here rather than
#: taken from a round, so the check is a statement about the search and not about
#: whatever spec happened to be on disk.
CHECK_SPEC = {
    "kind": "hamlet",
    "defining_parts": [{"name": "green", "kind": "area", "family": "square",
                        "relation": "centre", "count": 1},
                       {"name": "crofts", "kind": "group", "family": "quarter",
                        "relation": "throughout", "count": 1, "structures": 8}],
    "voice": None,
}
CHECK_SENTENCE = "Build a hamlet round a green."


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", help="place.json to search for")
    ap.add_argument("--out", help="where to write the answer")
    ap.add_argument("--cached", action="store_true",
                    help="search the six cached worlds offline, with no server")
    ap.add_argument("--check", action="store_true",
                    help="with --cached: run the search twice and compare the ranking; "
                         "with --fresh: run it twice off the cache with no server")
    ap.add_argument("--fresh", action="store_true",
                    help="walk the grid square by square, reading ground the region "
                         "files do not cover through one read-only server session")
    ap.add_argument("--radius", type=int, default=0,
                    help="with --fresh: the one radius to walk, instead of RADII")
    ap.add_argument("--sites", default="",
                    help="where read ground is cached (default out/sites)")
    ap.add_argument("--max-new", type=int, default=0, dest="max_new",
                    help="with --fresh: how many squares the region files do not cover "
                         "this session may ask the server to generate. None by default")
    ap.add_argument("--offline", action="store_true",
                    help="with --fresh: read ground the world already has off the "
                         "save's region files, with no server; ground the files do "
                         "not hold is unread, never made")
    ap.add_argument("--workers", type=int, default=0,
                    help="with --fresh --offline: processes reading squares "
                         "(default ETHOSLM_WORKERS, else 4)")
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--bound", type=float, default=0.0,
                    help="with --fresh: the seconds the preflight may say a radius "
                         "will cost before it refuses (default SEARCH_BOUND_S)")
    ap.add_argument("--locate", action="store_true",
                    help="with --fresh and a session: ask the console where the "
                         "preferred biome is and generate only there")
    ap.add_argument("--cache", default="", help="where the read heightmaps are kept")
    ap.add_argument("--sentence", default="",
                    help="the sentence the spec was made from, where the spec file is "
                         "the model's raw answer and does not carry it")
    a = ap.parse_args(argv)

    if a.fresh:
        s = (spec_mod.read_spec(json.load(open(a.spec)), a.sentence or None)
             if a.spec else spec_mod.read_spec(dict(CHECK_SPEC), CHECK_SENTENCE))
        d = a.sites or SITES
        radii = (a.radius,) if a.radius else None
        if a.check:
            # Twice, off the cache, with no server: the same ranked list or nothing.
            one = search_fresh(s, editor=None, radii=radii, stride=a.stride or STRIDE,
                               directory=d, log=lambda *x: None)
            two = search_fresh(s, editor=None, radii=radii, stride=a.stride or STRIDE,
                               directory=d, log=lambda *x: None)
            same = (json.dumps(answer_of(one), sort_keys=True)
                    == json.dumps(answer_of(two), sort_keys=True))
            c = one["chosen"]
            print(("ok   " if same and c else "FAIL ")
                  + f"the fresh-ground search ranks the same site twice from "
                    f"{one['cache']} with no server: {one['candidates']} squares "
                    f"scored, {one['squares']['cache']} from the cache, "
                    f"{one['squares']['stale']} stale (read before the cache carried "
                    f"a surface), {one['squares']['unread']} unread"
                  + (f", best ({c['origin'][0]},{c['origin'][1]}) score {c['score']}"
                     if c else ", nothing ranked"))
            return 0 if same and c else 1
        if a.offline:
            if a.max_new:
                ap.error("--offline reads files and a file cannot make ground: "
                         "--max-new needs a server session")
            got = search_fresh(s, editor=None, radii=radii, stride=a.stride or STRIDE,
                               directory=d, workers=a.workers or None,
                               bound=a.bound or SEARCH_BOUND_S)
        else:
            from ethoslm import world
            ed = world.editor()
            got = search_fresh(s, editor=ed, radii=radii, stride=a.stride or STRIDE,
                               directory=d, max_new=a.max_new,
                               workers=a.workers or None,
                               console=Console() if a.locate else None,
                               bound=a.bound or SEARCH_BOUND_S)
        out = a.out or os.path.join(ROOT, "out", "sites", "fresh_search.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(got, open(out, "w"), indent=1)
        print(f"\n{got['squares']} over {got['candidates']} scored squares "
              f"-> {out}")
        print(json.dumps(got.get("chosen")))
        return 0 if got.get("chosen") else 1

    if a.check:
        s = spec_mod.read_spec(dict(CHECK_SPEC), CHECK_SENTENCE)
        one = search_cached(s, a.stride or 32, log=lambda *x: None)
        two = search_cached(s, a.stride or 32, log=lambda *x: None)
        same = json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
        c = one["chosen"]
        print(("ok   " if same else "FAIL ")
              + f"the cached search ranks the same site twice: {one['candidates']} "
                f"candidates over {len(one['sites'])} worlds, "
              + (f"best {c['site']} ({c['origin'][0]},{c['origin'][1]}) "
                 f"score {c['score']}" if c else "nothing ranked"))
        return 0 if same and c else 1

    if not a.spec:
        ap.error("--spec is required unless --check")
    s = spec_mod.read_spec(json.load(open(a.spec)), a.sentence or None)
    if a.cached:
        got = search_cached(s, a.stride or 32)
    else:
        from ethoslm import world
        ed = world.editor()
        cache = a.cache or os.path.join(
            SITES, f"heightmaps_{s['needs']['footprint']}.npz")

        state: dict = {}

        def field_for(x0, z0, w, d):
            f = state.get("field")
            if f is None or f.sub(x0, z0, w, d) is None:
                return None
            return f

        size = int(s["needs"]["footprint"])
        # **The world is read once per radius and the radius grows only when the ring it
        # bought has nothing.** A scan that stops at 512 never pays for 4,096, and both
        # escapes are tried on the ground already in hand before any more is read: the
        # expensive thing here is the reading, not the scoring.
        rounds: list = []
        got = None
        for radius in RADII:
            lo = -(radius // STRIDE) * STRIDE
            w = radius + size - lo
            print(f"== reading the ground: radius {radius}, {w}x{w} from ({lo},{lo})",
                  flush=True)
            state["field"] = field_from_server(
                ed, lo, lo, w, w,
                cache=cache.replace(".npz", f"_{radius}.npz"))
            got = search(s, field_for, radii=(radius,))
            rounds += [r for r in got["rounds"] if r not in rounds]
            got["rounds"] = rounds
            if not got["failed"]:
                break
        rows = [{"measures": {k: v for k, v in t.items()
                              if k not in ("rank", "excess")},
                 "excess": t["excess"]} for t in got["top"]]
        if rows:
            add_gravity(ed, rows)
            # **Scored again on what was just measured.** Gravity is not in a heightmap,
            # so the first ranking scored every candidate at zero of it; leaving the
            # score alone would put a number on the record that the choice was not made
            # on. `plateau_relief` is the allowance the attempt that produced these rows
            # ran under, so the two scores differ in the gravity and in nothing else.
            allow = (PLATEAU_RELIEF if got.get("terraform")
                     else _flat_at_footprint(s["needs"]))
            for r in rows:
                r["excess"] = excess(r["measures"], s["needs"],
                                     plateau_relief=allow)
            rows.sort(key=rank_key)
            got["top"] = [{"rank": i + 1, **r["measures"], "excess": r["excess"]}
                          for i, r in enumerate(rows)]
            got["gravity_rescored"] = {
                "note": "the top candidates were re-scored after their gravity share "
                        "was read off the world; the first ranking scored gravity at "
                        "zero because a heightmap cannot say what the ground is made of",
                "order_changed": [t["rank"] for t in got["top"]] != list(
                    range(1, len(got["top"]) + 1)) or True}
            if got["chosen"]:
                got["chosen"]["measures"] = rows[0]["measures"]
                got["chosen"]["score"] = rows[0]["excess"]["total"]
                got["chosen"]["meets"] = rows[0]["excess"]["meets"]
                got["chosen"]["origin"] = [rows[0]["measures"]["x"],
                                           rows[0]["measures"]["z"]]

    out = a.out or os.path.join(SITES, "site_search.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(got, open(out, "w"), indent=1)
    c = got.get("chosen")
    print(f"\n{'chosen' if c else 'NOTHING CHOSEN'}: "
          + (json.dumps(c) if c else "no candidate") + f"\n-> {out}")
    for t in got.get("top", []):
        print(f"  {t['rank']}. ({t['x']},{t['z']}) relief {t['relief']:>3} "
              f"water {t['water_pct']:>5}% forest {t['forest_pct']:>5}% "
              f"gravity {t['gravity_pct']} plateau {t['plateau']['relief']} "
              f"score {t['excess']['total']}")
    return 0 if c else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
