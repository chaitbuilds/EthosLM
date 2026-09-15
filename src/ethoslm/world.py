"""World connection and site handling.

Note on build area: GDMC-HTTP's build area is set by the in-game `/setbuildarea`
command, which needs a player. This runs headless, so sites are declared in code
instead of read back from the mod.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
from gdpc import Editor
from gdpc.vector_tools import Rect

HOST = "http://localhost:9000"
HEIGHTMAP = "MOTION_BLOCKING_NO_LEAVES"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORLD_DIR = os.path.join(ROOT, "run", "server", "world")
SNAP_DIR = os.path.join(ROOT, "run", "snapshots")


def serving() -> str | None:
    """Why there is no backend to build against, or None if there is one.

        Four of this project's suites write blocks and so need a running server. With
        nothing listening they each ended in a `gdpc` connection traceback, which reads
        like four broken suites to anybody who has not been told that a backend is a
        separate thing you start. A suite that needs one asks here first and says it is
        skipping.
        
    """
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(f"{HOST}/blocks?x=0&y=0&z=0", timeout=2).read()
        return None
    except urllib.error.HTTPError:
        return None                      # it answered; the request is not the point
    except Exception as e:               # noqa: BLE001 -- any failure is "no backend"
        return (f"no GDMC HTTP backend at {HOST} ({type(e).__name__}). This suite "
                f"writes blocks into a world, so it needs a Minecraft server with the "
                f"GDMC HTTP mod running; the offline suites do not.")


def editor(buffering: bool = True, limit: int = 8192) -> Editor:
    ed = Editor(buffering=buffering, bufferLimit=limit, caching=True, host=HOST)
    ed.doBlockUpdates = False  # bulk placement; we are not relying on water or gravity
    return ed


@dataclass
class Site:
    """A rectangular patch of world with its surface heightmap already loaded."""

    x: int
    z: int
    sx: int
    sz: int
    heights: np.ndarray  # [x][z], local indices
    editor: Editor | None = None

    @property
    def rect(self) -> Rect:
        return Rect((self.x, self.z), (self.sx, self.sz))

    def height(self, wx: int, wz: int) -> int:
        """Surface y at world (x, z): the y of the topmost solid block."""
        lx = min(max(wx - self.x, 0), self.sx - 1)
        lz = min(max(wz - self.z, 0), self.sz - 1)
        return int(self.heights[lx, lz]) - 1

    def stats(self) -> dict:
        h = self.heights.astype(int) - 1
        return {
            "min": int(h.min()), "max": int(h.max()),
            "mean": round(float(h.mean()), 1),
            "std": round(float(h.std()), 2),
            "relief": int(h.max() - h.min()),
        }


def load_site(ed: Editor, x: int, z: int, sx: int, sz: int) -> Site:
    rect = Rect((x, z), (sx, sz))
    ed.loadWorldSlice(rect, cache=True)
    return Site(x, z, sx, sz, ed.worldSlice.heightmaps[HEIGHTMAP], ed)


def snapshot(tag: str) -> str:
    """Copy the world folder. This is the undo mechanism; take one before every run."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    dest = os.path.join(SNAP_DIR, f"{tag}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    t0 = time.perf_counter()
    shutil.copytree(WORLD_DIR, dest)
    return f"{dest} ({time.perf_counter() - t0:.1f}s)"


def restore(tag: str) -> None:
    src = os.path.join(SNAP_DIR, tag)
    shutil.rmtree(WORLD_DIR, ignore_errors=True)
    shutil.copytree(src, WORLD_DIR)
