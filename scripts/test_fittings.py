"""Fittings, offline: the two halves of a bed must be where Minecraft expects them.

    "$PY" scripts/test_fittings.py

A bed's `facing` is the direction its head points, so the head block lies in that
direction *from* the foot. a person walking the town found it. This holds the rule for
all four facings.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm.prims import DIRS, Primitives  # noqa: E402

PASS, FAIL = [], []


def case(name, ok, note=""):
    (PASS if ok else FAIL).append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:44s} {note}")


class Flat(Primitives):
    """A floor at y=63 and open air above it."""

    def __init__(self):
        self.blocks = {}

    def place_block(self, x, y, z, block):
        self.blocks[(int(x), int(y), int(z))] = block

    def get_height(self, x, z):
        return 63

    def get_block(self, x, y, z):
        p = (int(x), int(y), int(z))
        if p in self.blocks:
            return self.blocks[p]
        return "stone" if int(y) <= 63 else "air"


for facing, (dx, dz) in DIRS.items():
    b = Flat()
    b.fitting("bed", 10, 64, 10, facing)
    head = [p for p, s in b.blocks.items() if "part=head" in s]
    foot = [p for p, s in b.blocks.items() if "part=foot" in s]
    want_foot = (10 - dx, 64, 10 - dz)          # behind the head, opposite `facing`
    case(f"bed facing {facing}: foot lies behind the head",
         head == [(10, 64, 10)] and foot == [want_foot],
         f"head {head}, foot {foot}, want {want_foot}")
    case(f"bed facing {facing}: both halves share the facing",
         all(f"facing={facing}" in s for s in b.blocks.values()))

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} fitting cases pass")
sys.exit(1 if FAIL else 0)
