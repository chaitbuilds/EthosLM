"""The world save's region files, read directly: a `WorldSlice` with no server.

The site search reads ground through one read-only server run, and a server run is a
thing this project has exactly one of. The ground the search wants is already on disk in
`run/server/world/region/`, in the same chunk NBT the GDMC-HTTP interface serves
(`ChunkSerializer` writes both), so the honest instrument is to read the files.

What this can and cannot do, and both are the point:

  - a chunk the save holds as `minecraft:full` is read exactly as the server would
    serve it -- the four heightmaps, every section's block states and biomes -- and
    `gdpc.WorldSlice` parses it, so every reader downstream is unchanged;
  - a chunk the save does not hold, or holds unfinished, is **`NotGenerated`**. A file
    cannot make ground. That is what `find_site.py --max-new` exists to gate on the
    server path, and here it cannot happen at all.

The fresh-ground search's `generated_regions()` already treats the region files as the fact of what
ground the world has; this reads what that fact points at.
"""
from __future__ import annotations

from io import BytesIO
import os

from gdpc import world_slice as _ws
from gdpc.vector_tools import Rect
from nbt import nbt, region

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Where the save keeps its region files. One file is 32x32 chunks, 512x512 blocks.
REGIONS = os.path.join(ROOT, "run", "server", "world", "region")

#: A chunk the generator has finished. Anything else -- a proto-chunk carved but not
#: decorated, a chunk with structures started and nothing else -- has no heightmaps
#: worth the name and is not ground the world has.
FULL = "minecraft:full"


class NotGenerated(Exception):
    """This chunk is not on disk as finished ground. The chunk is named."""


class Regions:
    """One directory of region files, with the files it has opened kept open."""

    def __init__(self, directory: str | None = None):
        self.directory = directory or REGIONS
        self._open: dict = {}

    def path(self, rx: int, rz: int) -> str:
        return os.path.join(self.directory, f"r.{rx}.{rz}.mca")

    def close(self) -> None:
        for f in self._open.values():
            try:
                f.close()
            except Exception:                 # noqa: BLE001 -- closing is best effort
                pass
        self._open.clear()

    def _file(self, rx: int, rz: int):
        key = (rx, rz)
        if key not in self._open:
            p = self.path(rx, rz)
            if not os.path.exists(p):
                raise NotGenerated(f"no region file r.{rx}.{rz}.mca")
            self._open[key] = region.RegionFile(p)
        return self._open[key]

    def chunk(self, cx: int, cz: int) -> nbt.NBTFile:
        """The chunk's NBT root, or `NotGenerated`."""
        f = self._file(cx >> 5, cz >> 5)
        try:
            tag = f.get_nbt(cx & 31, cz & 31)
        except region.InconceivedChunk as e:
            raise NotGenerated(f"chunk ({cx},{cz}) is not in its region file") from e
        status = tag["Status"].value if "Status" in tag else ""
        if status != FULL:
            raise NotGenerated(f"chunk ({cx},{cz}) is {status or 'unfinished'}, "
                               f"not {FULL}")
        return tag

    def chunks_nbt(self, chunk_rect: Rect) -> bytes:
        """The `Chunks` list the GDMC-HTTP `/chunks` endpoint answers with, as bytes.

                In the order `WorldSlice` indexes it -- `x + z * size.x` -- and uncompressed,
                which is what `asBytes=True` returns from the interface.
                
        """
        root = nbt.NBTFile()
        root.name = ""
        chunks = nbt.TAG_List(name="Chunks", type=nbt.TAG_Compound)
        ox, oz = int(chunk_rect.offset.x), int(chunk_rect.offset.y)
        sx, sz = int(chunk_rect.size.x), int(chunk_rect.size.y)
        for j in range(sz):
            for i in range(sx):
                tag = self.chunk(ox + i, oz + j)
                comp = nbt.TAG_Compound()
                comp.tags = list(tag.tags)
                chunks.append(comp)
        root.tags.append(chunks)
        buf = BytesIO()
        root.write_file(buffer=buf)
        return buf.getvalue()

    def world_slice(self, x0: int, z0: int, w: int, d: int) -> _ws.WorldSlice:
        """A `gdpc.WorldSlice` over this rectangle, off the files. `NotGenerated` if
        any chunk it needs is not finished ground."""
        rect = Rect((int(x0), int(z0)), (int(w), int(d)))
        chunk_rect = Rect(rect.offset >> 4, ((rect.last) >> 4) - (rect.offset >> 4) + 1)
        data = self.chunks_nbt(chunk_rect)
        # `WorldSlice` fetches through `interface.getChunks` and parses in one
        # constructor; handing it these bytes through that one seam is what keeps the
        # parse -- heightmaps, sections, biomes, block entities -- gdpc's own.
        orig = _ws.interface.getChunks
        _ws.interface.getChunks = lambda *a, **k: data
        try:
            return _ws.WorldSlice(rect)
        finally:
            _ws.interface.getChunks = orig


# ------------------------------------------------------------ reading in parallel

_FIND_SITE = None


def _find_site():
    """`scripts/find_site.py` as a module, loaded once per process. It is a script,
    and a worker function has to live somewhere a pool can name."""
    global _FIND_SITE
    if _FIND_SITE is None:
        import importlib.util
        p = os.path.join(ROOT, "scripts", "find_site.py")
        s = importlib.util.spec_from_file_location("find_site_regions_worker", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        _FIND_SITE = m
    return _FIND_SITE


def prefetch_one(item: tuple) -> bool:
    """Read one square off the files into its cache file. `(x, z, size, cache_path)`.

        The unit of work `ethoslm.parallel.par_map` distributes: one process, one square,
        its own open region files, nothing shared. True where the square was finished
        ground and is now cached; False where the files do not hold it.
        
    """
    x, z, size, cache = item
    regs = Regions()
    try:
        f = _find_site().field_from_regions(regs, x, z, size, size, cache=cache,
                                            log=lambda *a: None)
    finally:
        regs.close()
    return f is not None
