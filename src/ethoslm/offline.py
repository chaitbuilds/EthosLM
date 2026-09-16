"""The world without a server, so a build pass can test itself before it submits.

A `Volume` is already a complete dense read of the region, so it is everything a program
needs: `get_height` and `get_block` come off it, writes go to the same pending dict a
real Builder uses, and the result can be linted without a single block reaching the
world. The gap is block updates -- fence and wall connection states are computed by the
server, so E005 is the one check a dry run cannot make. It is also the one check that is
now always zero, because `Builder.flush` handles it."""
from __future__ import annotations

import os

import numpy as np

from . import observe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Read-only ground the repository ships, as against `out/`, which is where a run
#: writes. The deterministic instruments -- the conformance suite, the type checker, the
#: needs bank -- stand their subjects on a handful of cached worlds, and those are small
#: enough to ship where the record of a run is not. A checkout that has both prefers
#: `out/`, so a round that re-cached its own ground is what its own suite reads; a
#: checkout that has only `fixtures/` needs nothing else to run them.
FIXTURES = os.path.join(ROOT, "fixtures")


def fixture_path(site: str, *parts) -> str:
    """One file of a fixture site: what a run of it left under `out/`, or what the
    repository ships under `fixtures/`. `FIXTURES`."""
    out = os.path.join(ROOT, "out", site, *parts)
    if os.path.exists(out):
        return out
    shipped = os.path.join(FIXTURES, site, *parts)
    return shipped if os.path.exists(shipped) else out


def world_cache(site: str, cache: str = "world.npz") -> str:
    """The path of one fixture site's cached world. `fixture_path`."""
    return fixture_path(site, cache)


class _Block:
    __slots__ = ("id", "states")

    def __init__(self, state: str):
        self.id = "minecraft:" + state.split("[")[0]
        self.states = observe.parse_props(state)


class _Slice:
    """Enough of GDPC's WorldSlice for the build API to run against."""

    def __init__(self, vol: observe.Volume, heights: np.ndarray):
        self.vol = vol
        self.heightmaps = {"MOTION_BLOCKING_NO_LEAVES": heights + 1,
                           "MOTION_BLOCKING": heights + 1,
                           "WORLD_SURFACE": heights + 1,
                           "OCEAN_FLOOR": heights + 1}

    def getBlockGlobal(self, p):
        return _Block(self.vol.state(int(p[0]), int(p[1]), int(p[2])))


class _Editor:
    def __init__(self, slice_):
        self.worldSlice = slice_

    def getBlock(self, p):
        return self.worldSlice.getBlockGlobal(p)


class OfflineSite:
    """A `world.Site` backed by a cached Volume instead of a live server."""

    def __init__(self, vol: observe.Volume, heights=None):
        self.vol = vol
        sx, _, sz = vol.shape
        self.x, self.z, self.sx, self.sz = vol.x0, vol.z0, sx, sz
        # `heights` is the surface heightmap the ground contract fixed at resolution
        # (v2, B1): a build reads one heightmap for the whole of it rather than one
        # recomputed from the volume as each part left it. Taken only where it is the
        # same shape over the same origin; else read off the volume, as always.
        self.heights = (np.asarray(heights) if heights is not None
                        and np.shape(heights) == (sx, sz) else surface_heights(vol))
        self.editor = _Editor(_Slice(vol, self.heights))

    def height(self, wx: int, wz: int) -> int:
        lx = min(max(int(wx) - self.x, 0), self.sx - 1)
        lz = min(max(int(wz) - self.z, 0), self.sz - 1)
        return int(self.heights[lx, lz])


def surface_heights(vol: observe.Volume) -> np.ndarray:
    """The y of the topmost block that is not air and not a leaf, per column.

        Matches MOTION_BLOCKING_NO_LEAVES, which is what `Site.height` reads from the server
        -- including tree trunks, because that heightmap includes them and a dry run that
        disagreed with the live one would be worse than useless.
        
    """
    t = vol.tables()
    c = vol.codes
    leafy = np.array([s.split("[")[0].endswith("_leaves") for s in vol.palette], bool)
    solid = ((t["lower"].astype(bool) | t["upper"].astype(bool)) & ~leafy)[c]
    idx = solid.shape[1] - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    return np.where(solid.any(axis=1), idx + vol.y0, vol.y0).astype(int)


#: The names a model program is given. One list, because there were three -- in
#: dry_run.py, settlement_run.py and step2_run.py -- and they had already drifted apart,
#: so a primitive could exist in a dry run and not in the pass that followed it.
BUILD_API = (
    "place_block", "place_cuboid", "fill_region", "get_height", "get_block",
    "disc", "ring", "cylinder", "sphere", "dome", "line", "path",
    "roof", "roof_cone", "dormer",
    "clear_trees", "clear_ground_cover", "foundation_to_grade", "terrace",
    "flattest_rect", "wall", "window", "doorway", "fitting", "dress_ground",
    "plinth", "openings", "storey_steps", "steps", "step", "resolve_steps",
    "floor_from_threshold", "seal_voids", "approach", "flight", "building", "dais",
    "check_door", "nearest_lane", "threshold", "check_attached", "check_walkable",
)


def run_program(path: str, vol: observe.Volume, network=None, plots=None,
                allow_collide: bool = False, src: str | None = None,
                max_blocks: int | None = None, ground=None):
    """Execute a build program against a cached world. Nothing is written anywhere.

        Returns the Builder, with its queued treads already resolved -- which is what
        `Builder.flush` does first, so the blocks judged afterwards are the blocks the world
        would have got.

        `src` runs that source instead of the file's, under the file's name. It exists for
        one caller: `pipeline.instantiate`, which composes a *type* program's source with
        the calls that instantiate it. Composing and then executing through this one
        function is what keeps a type and an ordinary program the same thing to everything
        downstream -- the measures, the render and the lint all replay a type by executing
        a file, because that is all it is.
        
    """
    from .buildlib import Builder
    from .frontage import Frontage
    # `ground` is the build's resolved ground contract (`ground.Resolved`), v2 B1:
    # `site()` lays what it settled for this part, `grade()` reads its levels, and the
    # heightmap the program reads is the one the resolution fixed, where it holds one
    # over this volume's origin.
    heights = None
    if ground is not None and getattr(ground, "surface", None) is not None:
        s = ground.surface
        if (int(s.x0), int(s.z0)) == (int(vol.x0), int(vol.z0)):
            heights = s.h
    site = OfflineSite(vol, heights=heights)
    b = Builder(site)
    b._vol = vol
    b.ground = ground
    b.allow_collide = bool(allow_collide)
    if max_blocks:
        b.max_blocks = int(max_blocks)      # the guard scaled by the site's area
    b.frontage = Frontage(vol, network) if network else None
    env = {n: getattr(b, n) for n in BUILD_API}
    # The two names the *driver* half of a composed type program uses and a type does
    # not: `site()` prepares the ground a part stands on and `type_builder()` hands back
    # the library with the ground taken out of it. Kept out of `BUILD_API` on purpose --
    # they are what instantiates a type, not what a type is written against, and a type
    # that could call either could undo the contract it is held to.
    env["site"] = b.site
    env["type_builder"] = b.type_builder
    if plots is not None:
        env["reserve"] = plots.reserve
        env["plots"] = plots.plots_list
        # ...and the Builder gets it too, so `check_walkable()` can scope itself to the
        # plot the program reserved rather than to the box round everything it placed.
        b.registry = plots
    env["__name__"] = "__build__"
    env["__builtins__"] = __builtins__
    if src is None:
        src = open(path).read()
    exec(compile(src, path, "exec"), env)
    b.resolve_steps()
    return b


def save_volume(vol: observe.Volume, path: str) -> str:
    np.savez_compressed(path, codes=vol.codes, palette=np.array(vol.palette, dtype=object),
                        origin=np.array([vol.x0, vol.y0, vol.z0]))
    return path


def load_volume(path: str) -> observe.Volume:
    d = np.load(path, allow_pickle=True)
    x0, y0, z0 = (int(v) for v in d["origin"])
    return observe.Volume(x0, y0, z0, d["codes"], list(d["palette"]))
