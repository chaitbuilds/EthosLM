"""Dump the world to disk so build passes can be tested without a server.

    ETHOSLM_SETTLEMENT=<name> bash scripts/mcrun.sh scripts/cache_world.py

Run once after the circulation pass, and again after each wave if later passes should
see what earlier ones built. Writes out/<name>/world.npz -- a few tens of megabytes,
gitignored, and the only thing scripts/dry_run.py needs.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from gdpc.vector_tools import Rect  # noqa: E402

from ethoslm import observe, offline, settlement, world  # noqa: E402
from ethoslm.measure import record  # noqa: E402

PAD = 48
s = settlement.site_info()
X, Z = s["origin"]
S = s["size"]

t0 = time.perf_counter()
ed = world.editor()
ed.loadWorldSlice(Rect((X - PAD, Z - PAD), (S + 2 * PAD, S + 2 * PAD)), cache=True)
h = ed.worldSlice.heightmaps[world.HEIGHTMAP].astype(int) - 1
inner = h[PAD:PAD + S, PAD:PAD + S]
y0, y1 = max(0, int(inner.min()) - 8), int(h.max()) + 48
vol = observe.Volume.from_world_slice(ed.worldSlice, X - PAD, Z - PAD,
                                      S + 2 * PAD, S + 2 * PAD, y0, y1)
path = os.path.join(settlement.STATE, "world.npz")
offline.save_volume(vol, path)
secs = time.perf_counter() - t0
mb = os.path.getsize(path) / 1e6
print(f"cached {vol.shape} y {y0}..{y1} ({len(vol.palette)} states) "
      f"-> {path}  {mb:.1f} MB in {secs:.1f}s")
record("cache_world", name=settlement.NAME, shape=list(vol.shape),
       states=len(vol.palette), mb=round(mb, 1), seconds=round(secs, 1))
