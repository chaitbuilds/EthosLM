"""Read generated chunk data from region files opened read-only."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from nbt.region import RegionFile
import numpy as np

from . import observe


class SavedWorld:
    """A bounded cache of decoded chunks; missing or unfinished chunks are refused."""

    def __init__(self, directory, cache_chunks=256):
        self.directory = Path(directory)
        self.cache_chunks = cache_chunks
        self.cache = OrderedDict()
        self.field_cache = OrderedDict()

    def chunk(self, cx, cz):
        key = cx, cz
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.directory / "region" / f"r.{cx // 32}.{cz // 32}.mca"
        with path.open("rb") as stream:
            tag = RegionFile(fileobj=stream).get_nbt(cx % 32, cz % 32)
        if str(tag["Status"].value).split(":")[-1] != "full":
            raise ValueError(f"chunk {cx},{cz} is not fully generated")
        codes = np.zeros((16, 384, 16), np.uint16)
        palette, index = ["air"], {"air": 0}
        for section in tag["sections"]:
            if "block_states" not in section:
                continue
            y = int(section["Y"].value) * 16
            if y < -64 or y >= 320:
                continue
            states = section["block_states"]
            local = []
            for item in states["palette"]:
                name = observe._state_str(item)
                if name not in index:
                    index[name] = len(palette)
                    palette.append(name)
                local.append(index[name])
            bits = max(4, (len(local) - 1).bit_length())
            data = (SimpleNamespace(_bitsPerEntry=bits, longArray=states["data"].value)
                    if "data" in states else None)
            sec = SimpleNamespace(blockStatesBitArray=data)
            decoded = np.asarray(local, np.uint16)[observe._decode_section(sec)]
            codes[:, y + 64:y + 80, :] = decoded.reshape(16, 16, 16).transpose(2, 0, 1)
        volume = observe.Volume(cx * 16, -64, cz * 16, codes, palette)
        self.cache[key] = volume
        while len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return volume

    def volume(self, x, z, width, depth):
        """A requested footprint, including the full generated vertical range."""
        codes = np.zeros((width, 384, depth), np.uint16)
        palette, index = ["air"], {"air": 0}
        for cx in range(x // 16, (x + width - 1) // 16 + 1):
            for cz in range(z // 16, (z + depth - 1) // 16 + 1):
                chunk = self.chunk(cx, cz)
                remap = []
                for name in chunk.palette:
                    if name not in index:
                        index[name] = len(palette)
                        palette.append(name)
                    remap.append(index[name])
                a, b = max(x, cx * 16), max(z, cz * 16)
                c, d = min(x + width, cx * 16 + 16), min(z + depth, cz * 16 + 16)
                block = chunk.codes[a - cx * 16:c - cx * 16, :, b - cz * 16:d - cz * 16]
                codes[a-x:c-x, :, b-z:d-z] = np.asarray(remap, np.uint16)[block]
        return observe.Volume(x, -64, z, codes, palette)

    def field(self, x, z, width, depth, field_module):
        """Read a field a chunk at a time, without allocating a radius-sized volume."""
        from nbt.region import InconceivedChunk
        h = np.zeros((width, depth), np.int32)
        wet, canopy, gravity = (np.zeros((width, depth), bool) for _ in range(3))
        manmade, occupied = (np.zeros((width, depth), np.int64) for _ in range(2))
        for cx in range(x // 16, (x + width - 1) // 16 + 1):
            for cz in range(z // 16, (z + depth - 1) // 16 + 1):
                try:
                    key = cx, cz
                    if key not in self.field_cache:
                        self.field_cache[key] = field_module.field_from_volume(self.chunk(cx, cz))
                    self.field_cache.move_to_end(key)
                    f = self.field_cache[key]
                    while len(self.field_cache) > 32768:
                        self.field_cache.popitem(last=False)
                except (FileNotFoundError, InconceivedChunk, ValueError):
                    return None
                a, b = max(x, cx * 16), max(z, cz * 16)
                c, d = min(x + width, cx * 16 + 16), min(z + depth, cz * 16 + 16)
                src = (slice(a-cx*16, c-cx*16), slice(b-cz*16, d-cz*16))
                dst = (slice(a-x, c-x), slice(b-z, d-z))
                for target, original in ((h, f.h), (wet, f.wet), (canopy, f.canopy),
                                         (gravity, f.gravity), (manmade, f.manmade),
                                         (occupied, f.occupied)):
                    target[dst] = original[src]
        return field_module.Field(x, z, h, wet, canopy, gravity,
            source="generated region files, read-only", manmade=manmade, occupied=occupied)
