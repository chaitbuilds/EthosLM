"""Does re-placing connective blocks twice actually join the last one in a run?

That was a hypothesis, and the evidence it was changed on (18 missed wall joins becoming
10) turned out to be confounded -- most of that residue was a check artefact, not a
build defect.

So measure it instead of arguing: the same run of walls flushed with one sweep and with
two, read back out of the world.

    bash scripts/mcrun.sh scripts/test_wall_joins.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gdpc.vector_tools import Rect  # noqa: E402

from ethoslm import observe, world  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.measure import record  # noqa: E402

X, Z, Y = 3040, 3040, 100          # empty ground far from anything we care about
# A suite that writes blocks needs a backend; with none it says so and stops rather than
# ending in a connection traceback that reads like a broken suite.
_no_backend = world.serving()
if _no_backend:
    print(f"skipped: {_no_backend}")
    raise SystemExit(0)

ed = world.editor()
site = world.load_site(ed, X - 8, Z - 8, 48, 48)


def run(x0, passes):
    """A wall run of four between two posts, plus a fence run, flushed `passes` times."""
    b = Builder(site)
    for k in range(6):
        b.place_block(x0 + k, Y, Z, "stone_bricks")
        b.place_block(x0 + k, Y + 4, Z + 4, "stone_bricks")
    for k in range(1, 5):
        b.place_block(x0 + k, Y + 1, Z, "cobblestone_wall")
        b.place_block(x0 + k, Y + 5, Z + 4, "oak_fence")
    b.flush(conn_passes=passes)
    return b


run(X, 1)
run(X + 12, 2)

ed2 = world.editor()
ed2.loadWorldSlice(Rect((X - 8, Z - 8), (48, 48)), cache=True)
vol = observe.Volume.from_world_slice(ed2.worldSlice, X - 8, Z - 8, 48, 48,
                                      Y - 2, Y + 10)
out = {}
for name, x0 in (("one sweep", X), ("two sweeps", X + 12)):
    j = observe.unconnected_joins(vol, region=(x0, Z - 1, x0 + 5, Z + 5))
    out[name] = {k: j[k] for k in ("fence", "wall")}
    print(f"{name:11s} wall {j['wall']}  fence {j['fence']}")
    for k in range(1, 5):
        print(f"    {vol.state(x0 + k, Y + 1, Z):58s} | "
              f"{vol.state(x0 + k, Y + 5, Z + 4)}")

record("wall_join_sweeps", **{k.replace(" ", "_"): v for k, v in out.items()})
same = all(out["one sweep"][k]["missed_joins"] == out["two sweeps"][k]["missed_joins"]
           for k in ("fence", "wall"))
print("\ntwo sweeps make no difference" if same else
      "\ntwo sweeps join what one sweep leaves")
