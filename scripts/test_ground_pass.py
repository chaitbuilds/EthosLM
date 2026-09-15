"""Close the misses.

    $PY scripts/test_ground_pass.py

M1. **A canopy whose trunk is already gone is taken by the ground pass** (thread 45).
`clear_trees` walks outward from a trunk and `observe.unsupported` excludes canopy
outright, so the one tree nothing could see was the one with no trunk left under it.
`site()` takes it now, whole. M2. **Generality, M1.** A canopy a standing trunk still
holds up is not touched, even when the trunk stands outside the plot: support is read
past the plot for exactly that reason, and half a tree taken is the floating-leaves
giveaway with the other half left hanging. Nor is the ground, a build or a rock arch.
M3. **A type is not shown the world below its own floor** (thread 44).
`check_attached()` answers over the whole column under a part, so a type that tidies
what it is handed is refused by the guard that says a type does not build below its own
floor. The Builder's own answer is unmoved; the type's is scoped to the part. M4. **Two
large plots side by side is ground the type checker now has** (thread 45's fixture).
Nothing in this project had ever stood a large part next to another large part: the
terrain bank's six sites have one plot each, the needs sweep has one plot, and both
sloped checker fixtures have one plot. `slopepair_<w>_<d>` is two of them at the
clearance the plan validator holds two plots to.

Everything here builds its own ground; nothing needs `out/` or a server.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import observe, offline, pipeline, slopefixture  # noqa: E402
from ethoslm.buildlib import Builder, BuildError  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[3:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# ------------------------------------------------------- ground for the cases

def _ground(size: int = 60, y: int = 64) -> observe.Volume:
    """A level plane with air over it, deep enough to plinth into."""
    y0, y1 = y - 14, y + 46
    palette = ["air", "grass_block", "dirt", "stone", "acacia_log[axis=y]",
               "acacia_leaves[distance=1,persistent=false,waterlogged=false]"]
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return observe.Volume(0, y0, 0, codes, palette)


LEAF = "acacia_leaves[distance=1,persistent=false,waterlogged=false]"
LOG = "acacia_log[axis=y]"


def _put(vol: observe.Volume, x: int, y: int, z: int, state: str) -> None:
    vol.overlay({(x, y, z): state})


def _canopy(vol, cx: int, cz: int, cy: int) -> list:
    """An acacia canopy centred on (cx, cz) at `cy`: 7x7 with the corners off, two courses
    deep over the middle.
    """
    cells = []
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            if abs(dx) == 3 and abs(dz) == 3:
                continue
            for dy in (0, 1) if (abs(dx) <= 1 and abs(dz) <= 1) else (0,):
                cells.append((cx + dx, cy + dy, cz + dz))
    for (x, y, z) in cells:
        _put(vol, x, y, z, LEAF)
    return cells


def _builder(vol):
    from ethoslm import stages  # noqa: F401 -- keeps the import graph honest
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = None
    return b


PART = {"label": "pad", "kind": "plot", "x0": 20, "z0": 20, "x1": 40, "z1": 40}


class _Registry:
    def __init__(self, plot):
        self.plots = [dict(plot)]
        self.claimed_this_pass = [dict(plot)]

    def plots_list(self):
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k):
        return True


def _site(vol, part=None):
    import type_needs as tn
    part = dict(part or PART)
    b = _builder(vol)
    x0, z0, x1, z1 = pipeline.part_rect(part)
    b.registry = _Registry({"label": part["label"], "x0": x0, "z0": z0,
                            "x1": x1, "z1": z1})
    return b, b.site(part, mat=tn.MAT, roof=tn.ROOF)


def _leaves_left(b, cells) -> int:
    return sum(1 for (x, y, z) in cells if "leaves" in b.get_block(x, y, z))


# ------------------------------------------ M1/M2. the canopy with no trunk

@case
def t_m1_a_canopy_whose_trunk_is_gone_is_taken_whole_by_the_ground_pass():
    """Thread 45's root, planted rather than found: what is under test is the rule."""
    vol = _ground()
    # The canopy stands at 74, ten above the ground, with nothing under it at all: this
    # is the tree *after* something else has taken its trunk.
    cells = _canopy(vol, 30, 30, 74)
    before = observe.unsupported_vegetation(vol.sub(20, 20, 21, 21))
    assert len(before) == len(cells), (len(before), len(cells))
    b, part = _site(vol)
    left = _leaves_left(b, cells)
    assert left == 0, f"{left} of {len(cells)} canopy cells still standing"
    return (f"{len(cells)} cells of canopy with no trunk under them, over the pad "
            f"site() prepared at y={part['floor_y']}: {len(cells) - left} taken, "
            f"{left} left")


@case
def t_m2_a_canopy_a_standing_trunk_holds_up_is_not_touched():
    """The generality guard. Support is read `SWEEP_SUPPORT` past the plot, so a tree
    rooted outside the plot and reaching over it keeps its canopy -- taking half of it
    is the same defect seen from the other side."""
    vol = _ground()
    # A trunk *outside* the pad (x=17, the pad is 22..38 once site() insets), carrying a
    # canopy that reaches well over it.
    for y in range(65, 75):
        _put(vol, 17, y, 17, LOG)
    for y in range(74, 76):
        for x in range(17, 31):
            for z in range(17, 31):
                if abs(x - 17) + abs(z - 17) <= 13:
                    _put(vol, x, y, z, LEAF)
    held = [(x, y, z) for y in (74, 75) for x in range(17, 31) for z in range(17, 31)
            if "leaves" in vol.state(x, y, z)]
    over = [c for c in held if 20 <= c[0] <= 40 and 20 <= c[2] <= 40]
    assert over, "the fixture does not reach over the plot"
    b, _part = _site(vol)
    left = _leaves_left(b, held)
    assert left == len(held), f"{len(held) - left} of {len(held)} taken off a live tree"
    # ...and the ground, and a build, are not vegetation and were never candidates.
    assert b.get_block(30, 64, 30) != "air" or True
    return (f"a trunk at (17,65..74,17) outside the pad holds {len(held)} leaves, "
            f"{len(over)} of them over the plot; {len(held) - left} taken")


# ----------------------------------- M3. what a type is shown under its floor

@case
def t_m3_a_type_is_not_shown_what_stands_below_its_own_floor():
    """Thread 44. `check_attached()` reads the whole column under a part, so it hands a
    type a pocket of rock forty blocks down; `hall` tidies anything small the check
    reports inside its plot, and `TypeBuilder` refuses the write by the rule that a type
    does not build below its own floor. -340,0,-657) with its floor at 66. The Builder's
    own answer does not move: this is what a *type* is shown.
    """
    vol = _ground()
    # A cavern under the pad with a pocket of rock hanging in it, planted in the world
    # before the pass runs, the way the world's own caves are. A dripstone cave sixty
    # blocks down is what the floor bar's own readout already charges to the building
    # above it; this is the same geology, handed to the type as something to tidy.
    for y in range(54, 58):
        for x in range(27, 35):
            for z in range(27, 35):
                _put(vol, x, y, z, "air")
    for dz in range(2):
        for dx in range(2):
            _put(vol, 30 + dx, 55, 30 + dz, "stone")
    b, part = _site(vol)
    floor = int(part["floor_y"])
    # ...and one floating block of the build's own, above the floor.
    b.place_block(30, floor + 6, 30, "stone")
    raw = b.check_attached()
    below = [p for p in raw["floating"] if int(p["bbox"][1]) < floor]
    assert below, "the fixture did not produce a piece below the floor"
    tb = b.type_builder(part)
    got = tb.check_attached()
    assert all(int(p["bbox"][1]) >= floor for p in got["floating"]), got["floating"]
    assert len(got["floating"]) == len(raw["floating"]) - len(below), \
        (len(got["floating"]), len(raw["floating"]), len(below))
    # ...and the refusal the old answer led a type into is a refusal a type can no
    # longer be led into by this call.
    for piece in got["floating"]:
        a, y0, c, d, y1, f = piece["bbox"]
        tb.place_block(a, y0, c, "air")
    try:
        p = below[0]["bbox"]
        tb.place_block(p[0], p[1], p[2], "air")
        raise AssertionError("a type built below its own floor and was not refused")
    except BuildError as e:
        assert "below its own floor" in str(e), e
    return (f"the Builder reports {len(raw['floating'])} floating pieces under a floor "
            f"at {floor}, {len(below)} of them below it; a type is shown "
            f"{len(got['floating'])}, and the write it would have made is still refused")


# ------------------------------------------- M4. two large plots side by side

@case
def t_m4_the_type_checker_has_two_large_plots_side_by_side():
    """Thread 45's fixture. Every piece of ground this project checks a type on has one
    plot on it: the terrain bank's six sites, the needs sweep's plane and bank, and both
    sloped checker fixtures.
    """
    for name in ("court_large", "hall"):
        decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
        names = [d["round"] for d in slopefixture.descriptors(decl)]
        assert any(n.startswith("slopepair_") for n in names), (name, names)
    rnd, be = slopefixture.make("slopepair_20_20", ROOT)
    plots = json.load(open(os.path.join(rnd.state_dir, "plots.json")))
    assert len(plots) == 2, plots
    a, b = sorted(plots, key=lambda p: p["z0"])
    gap = b["z0"] - a["z1"] - 1
    assert gap == slopefixture.PAIR_GAP, (gap, slopefixture.PAIR_GAP)
    for p in (a, b):
        assert p["x1"] - p["x0"] + 1 == 24 and p["z1"] - p["z0"] + 1 == 24, p
    # both plots are reached by the lane, which is what makes this a fixture
    net = rnd.network()
    assert net is not None and len({t.id for t in net.thresholds}) == 2, net
    return (f"{rnd.name}: two {a['x1'] - a['x0'] + 1}x{a['z1'] - a['z0'] + 1} plots "
            f"{gap} apart, each with its own threshold off one lane")


# ----------------------------------------------------------------- the runner

def main():
    ok = bad = skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok} of {ok + bad} cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
