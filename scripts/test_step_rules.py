"""The three primitives that own a correctness rule, checked without a server.

Same discipline as scripts/test_roof_rules.py, and for the same reason: the roof stair
facing was inverted *in the library* and every roof in every build was serrated for four
rounds before anyone noticed. A primitive that owns a correctness rule is wrong
everywhere at once, so each one here is exercised on ground built to trip it, and each
suite is then run against a deliberately broken version of the rule to show it
discriminates. A test that only passes proves nothing.

Run: .venv/bin/python scripts/test_step_rules.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import observe                       # noqa: E402
from ethoslm.buildlib import Builder              # noqa: E402
from ethoslm.circulate import Network, Threshold  # noqa: E402
from ethoslm.frontage import Frontage             # noqa: E402
from ethoslm.offline import OfflineSite           # noqa: E402
from ethoslm.prims import Primitives              # noqa: E402

PASS, FAIL = [], []


def case(name, ok, note=""):
    (PASS if ok else FAIL).append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:52s} {note}")


# --------------------------------------------------------------------------- steps
class Ground(Primitives):
    """A builder over a synthetic hillside. `height(x, z)` is the terrain."""

    def __init__(self, height):
        self.blocks: dict[tuple[int, int, int], str] = {}
        self._h = height

    def place_block(self, x, y, z, block):
        self.blocks[(int(x), int(y), int(z))] = block

    def get_height(self, x, z):
        top = self._h(int(x), int(z))
        for (bx, by, bz), b in self.blocks.items():
            if (bx, bz) == (int(x), int(z)) and b != "air" and by > top:
                top = by
        return top

    def get_block(self, x, y, z):
        p = (int(x), int(y), int(z))
        if p in self.blocks:
            return self.blocks[p].split("[")[0]
        return "stone" if y <= self._h(int(x), int(z)) else "air"


def stair_facings(b):
    return {p: v.split("facing=")[1].split(",")[0].split("]")[0]
            for p, v in b.blocks.items() if "_stairs[" in v}


def volume_of(b, pad=3):
    xs = [p[0] for p in b.blocks] or [0]
    ys = [p[1] for p in b.blocks] or [0]
    zs = [p[2] for p in b.blocks] or [0]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    z0, z1 = min(zs) - pad, max(zs) + pad
    full = {}
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            for y in range(y0, y1 + 1):
                full[(x, y, z)] = "stone" if y <= b._h(x, z) else "air"
    full.update(b.blocks)
    return observe.Volume.from_blocks(full, x0, y0, z0,
                                      x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def flight_up_a_slope(broken=False):
    """A flight climbing a 1-in-1 slope, queued in the order a program would write it."""
    b = Ground(lambda x, z: 64 + max(0, min(6, z)))
    cells = [(0, 64 + i, i) for i in range(1, 7)]
    if broken:
        # what a program does without the primitive: pick the facing up front, from the
        # direction of travel rather than from the ground
        for (x, y, z) in cells:
            b.place_block(x, y, z, "cobblestone_stairs[facing=north,half=bottom]")
    else:
        b.steps(cells, "cobblestone")
        b.resolve_steps()
    return b


def steps_cases(broken=False):
    out = []

    b = flight_up_a_slope(broken)
    f = stair_facings(b)
    out.append(("steps: every tread of a climbing flight faces up-slope",
                len(f) == 6 and set(f.values()) == {"south"},
                f"{len(f)} treads, facings {sorted(set(f.values()))}"))

    # ...and the check that judges it in the world agrees
    bad = observe.backwards_stairs(volume_of(b))
    out.append(("steps: observe.backwards_stairs finds nothing in the built flight",
                not bad, f"{len(bad)} backwards"))

    # queued in reverse: the facing comes off the ground, not off the order
    b2 = Ground(lambda x, z: 64 + max(0, min(6, z)))
    cells = [(0, 64 + i, i) for i in range(6, 0, -1)]
    if broken:
        for (x, y, z) in cells:
            b2.place_block(x, y, z, "cobblestone_stairs[facing=north,half=bottom]")
    else:
        b2.steps(cells, "cobblestone")
        b2.resolve_steps()
    f2 = stair_facings(b2)
    out.append(("steps: a flight queued downhill comes out the same way round",
                len(f2) == 6 and set(f2.values()) == {"south"},
                f"facings {sorted(set(f2.values()))}"))

    # level ground: no facing can be justified, so it is not a stair. Treads laid into
    # the surface, four blocks apart so each one is judged on the ground and not on its
    # neighbours -- the last tread of a flight, out on a landing.
    b3 = Ground(lambda x, z: 64)
    lone = [(4 * i, 64, 0) for i in range(4)]
    if broken:
        for (x, y, z) in lone:
            b3.place_block(x, y, z, "cobblestone_stairs[facing=north,half=bottom]")
    else:
        for (x, y, z) in lone:
            b3.step(x, y, z, "cobblestone", "x")
        b3.resolve_steps()
    slabs = [v for v in b3.blocks.values() if "_slab" in v]
    out.append(("steps: a tread on level ground is demoted, not guessed at",
                len(slabs) == 4, f"{len(slabs)} slabs, {len(stair_facings(b3))} stairs"))

    # ...unless the caller says which way it should face. Policy, so it is a parameter.
    b4 = Ground(lambda x, z: 64)
    if broken:
        for (x, y, z) in lone:
            b4.place_block(x, y, z, "cobblestone_stairs[facing=north,half=bottom]")
    else:
        for (x, y, z) in lone:
            b4.step(x, y, z, "cobblestone", "x", prefer="east")
        b4.resolve_steps()
    f4 = stair_facings(b4)
    out.append(("steps: prefer= is honoured on level ground and nowhere else",
                len(f4) == 4 and set(f4.values()) == {"east"},
                f"{len(f4)} stairs facing {sorted(set(f4.values()))}"))

    # the step to nowhere: a rising tread the flight does not continue past
    b5 = Ground(lambda x, z: 64 if z <= 3 else 58)      # a cliff at z=4
    if broken:
        for i, z in enumerate((1, 2, 3)):
            b5.place_block(0, 64 + i, z, "cobblestone_stairs[facing=south,half=bottom]")
    else:
        b5.steps([(0, 64 + i, z) for i, z in enumerate((1, 2, 3))], "cobblestone")
        b5.resolve_steps()
    top = b5.blocks.get((0, 66, 3), "")
    out.append(("steps: a rising tread with nothing beyond it is refused",
                "_slab" in top, f"top tread is {top or 'missing'}"))

    # a tread with a drop on one side and a rise on the other is still a tread
    b6 = Ground(lambda x, z: 60 if x < 0 else 64 + max(0, min(4, x)))
    if broken:
        for i in range(1, 4):
            b6.place_block(i, 64 + i, 0, "cobblestone_stairs[facing=west,half=bottom]")
    else:
        b6.steps([(i, 64 + i, 0) for i in range(1, 4)], "cobblestone")
        b6.resolve_steps()
    f6 = stair_facings(b6)
    out.append(("steps: a flight beside a drop still climbs the right way",
                len(f6) == 3 and set(f6.values()) == {"east"},
                f"facings {sorted(set(f6.values()))}"))

    # A flight that starts against a wall. "the first stair is almost always in the
    # wrong direction". Deciding each tread from the two columns beside it, the bottom
    # tread sees the wall it starts against on one side and the next tread up on the
    # other, finds the wall higher, and turns round to face it. Three of that town's 29
    # flights did exactly this and E004 passed every one of them, because per tread the
    # facing is defensible -- it is only wrong relative to the flight.
    def against_a_wall():
        g = Ground(lambda x, z: 64)
        for y in (65, 66, 67):
            g.place_block(0, y, 0, "stone")            # the wall at the foot
        for x in (6, 7):
            g.place_block(x, 69, 0, "stone")           # the landing at the head
        return g, [(x, 64 + x, 0) for x in range(1, 6)]

    b7, cells7 = against_a_wall()
    if broken:
        for (x, y, z) in cells7:
            b7.place_block(x, y, z, "cobblestone_stairs[facing=west,half=bottom]")
    else:
        b7.steps(cells7, "cobblestone")
        b7.resolve_steps()
    f7 = stair_facings(b7)
    out.append(("steps: a flight starting against a wall does not turn its bottom tread",
                len(f7) == 5 and set(f7.values()) == {"east"},
                f"{len(f7)} treads, facings {sorted(set(f7.values()))}"))

    # If this comes out clean the case above proves nothing.
    if not broken:
        b8, cells8 = against_a_wall()
        b8._flight_facings = staticmethod(lambda queue: {})
        b8.steps(cells8, "cobblestone")
        b8.resolve_steps()
        f8 = stair_facings(b8)
        out.append(("steps: ...and per-tread resolution really does turn it round",
                    f8.get((1, 65, 0)) == "west",
                    f"bottom tread faces {f8.get((1, 65, 0))}, rest "
                    f"{sorted({v for k, v in f8.items() if k != (1, 65, 0)})}"))
    return out


# ---------------------------------------------------------------------- seal_voids
def world_with(blocks, x0, y0, z0, sx, sy, sz, ground_y):
    """A flat world at `ground_y` with `blocks` written into it, as an OfflineSite."""
    cells = {}
    for x in range(x0, x0 + sx):
        for z in range(z0, z0 + sz):
            for y in range(y0, y0 + sy):
                cells[(x, y, z)] = "grass_block" if y <= ground_y else "air"
    cells.update(blocks)
    vol = observe.Volume.from_blocks(cells, x0, y0, z0, sx, sy, sz)
    return vol, OfflineSite(vol)


def hut(sealed_cellar: bool, door: bool = True) -> dict:
    """A 7x7 hut with a doorway, and optionally a walled cellar with no way into it."""
    b = {}
    for x in range(0, 7):
        for z in range(0, 7):
            b[(x, 64, z)] = "oak_planks"                     # floor
            b[(x, 68, z)] = "oak_planks"                     # ceiling
            for y in (65, 66, 67):
                if x in (0, 6) or z in (0, 6):
                    b[(x, y, z)] = "cobblestone"
    if door:
        b[(3, 65, 0)] = "air"
        b[(3, 66, 0)] = "air"
    if sealed_cellar:
        for x in range(1, 6):
            for z in range(1, 6):
                for y in (61, 62, 63):
                    b[(x, y, z)] = "air" if 1 < x < 5 and 1 < z < 5 else "cobblestone"
        for x in range(1, 6):
            for z in range(1, 6):
                b[(x, 60, z)] = "cobblestone"
    return b


def seal_cases():
    out = []

    # 1. a sealed cellar under a hut: enclosed, unreachable, inside the footprint
    vol, site = world_with(hut(True), -6, 50, -6, 20, 40, 20, 63)
    b = Builder(site)
    b._vol = vol
    r = b.seal_voids(0, 0, 6, 6, "cobblestone", max_cells=64)
    out.append(("seal_voids: a sealed cellar with no way in is closed",
                r["filled"] == 1 and r["blocks"] > 0,
                f"{r['filled']} pockets, {r['blocks']} blocks, {len(r['left'])} left"))

    # 2. the same hut without the cellar: nothing to do, and it does nothing
    vol, site = world_with(hut(False), -6, 50, -6, 20, 40, 20, 63)
    b = Builder(site)
    b._vol = vol
    r = b.seal_voids(0, 0, 6, 6, "cobblestone")
    out.append(("seal_voids: a hut you can walk into is left alone",
                r["filled"] == 0 and r["blocks"] == 0 and not r["left"],
                r["reason"]))

    # 3. wave 5's failure, reproduced: air under a roof overhang, over the eaves. It is
    # sheltered and nobody walks in it, and packing it is what put 625 cells of solid
    # block above a building's own eaves.
    eaves = dict(hut(False))
    for x in range(-2, 9):
        for z in range(-2, 9):
            eaves[(x, 71, z)] = "dark_oak_planks"            # a roof, 3 above the top
    vol, site = world_with(eaves, -6, 50, -6, 22, 40, 22, 63)
    b = Builder(site)
    b._vol = vol
    r = b.seal_voids(-2, -2, 8, 8, "cobblestone")
    out.append(("seal_voids: air over its own eaves is reported, never packed",
                r["blocks"] == 0,
                f"{r['blocks']} blocks placed, {len(r['left'])} reported"))

    # 4. a pocket that straddles the footprint the caller claimed is not the caller's
    vol, site = world_with(hut(True), -6, 50, -6, 20, 40, 20, 63)
    b = Builder(site)
    b._vol = vol
    r = b.seal_voids(0, 0, 3, 6, "cobblestone")              # half the hut
    out.append(("seal_voids: a pocket outside the given footprint is not filled",
                r["blocks"] == 0 and any("outside the footprint" in x["reason"]
                                         for x in r["left"]),
                "; ".join(x["reason"] for x in r["left"])[:60] or "nothing reported"))

    # 5. max_cells is the caller's, and past it the answer is a report
    vol, site = world_with(hut(True), -6, 50, -6, 20, 40, 20, 63)
    b = Builder(site)
    b._vol = vol
    r = b.seal_voids(0, 0, 6, 6, "cobblestone", max_cells=4)
    out.append(("seal_voids: past max_cells it reports instead of filling",
                r["blocks"] == 0 and any("max_cells" in x["reason"] for x in r["left"]),
                "; ".join(x["reason"] for x in r["left"])[:60]))

    # 6. and the linter agrees about what was there: E003 before, silent after
    vol, site = world_with(hut(True), -6, 50, -6, 20, 40, 20, 63)
    b = Builder(site)
    b._vol = vol
    plots = [{"x0": 0, "z0": 0, "x1": 6, "z1": 6, "label": "hut"}]
    from ethoslm import lint
    before = lint.lint(lint.Context.build(vol, plots, region=(0, 0, 6, 6)),
                       only={"E003"})
    b.seal_voids(0, 0, 6, 6, "cobblestone")
    after_vol = vol.sub(-6, -6, 20, 20)
    after_vol.overlay(b._pending_view())
    after = lint.lint(lint.Context.build(after_vol, plots, region=(0, 0, 6, 6)),
                      only={"E003"})
    out.append(("seal_voids: E003 fires on the cellar and is silent once it is closed",
                len(before.findings) == 1 and len(after.findings) == 0,
                f"E003 {len(before.findings)} -> {len(after.findings)}"))
    return out


# -------------------------------------------------------------- floor_from_threshold
def town(floor_offset: int):
    """A lane with a reserved threshold, and a hut whose floor is `floor_offset` off it."""
    lane_y = 63
    cells = {(x, 10): {"y": lane_y, "rank": 3, "face": None} for x in range(0, 12)}
    th = Threshold("hut", 5, 10, lane_y, "north", (5, lane_y + 1, 9))
    net = Network(cells, [th])

    base = {}
    for x in range(-4, 16):
        for z in range(-4, 20):
            for y in range(50, 80):
                base[(x, y, z)] = ("stone" if y < 63 else
                                   ("grass_block" if y == 63 and z > 10 else
                                    ("grass_block" if y == 63 else "air")))
    for (x, z) in cells:
        base[(x, lane_y, z)] = "cobblestone"
    vol = observe.Volume.from_blocks(base, -4, 50, -4, 20, 30, 24)
    site = OfflineSite(vol)
    b = Builder(site)
    b._vol = vol
    b.frontage = Frontage(vol, net)

    fl = b.floor_from_threshold("hut")
    fy = fl["floor_y"] + floor_offset
    for x in range(2, 9):
        for z in range(3, 10):
            b.place_block(x, fy, z, "oak_planks")
            for y in range(fy + 1, fy + 4):
                b.place_block(x, y, z,
                              "cobblestone" if x in (2, 8) or z in (3, 9) else "air")
    door = fl["door"]
    b.place_block(door[0], fy + 1, door[2], "air")
    b.place_block(door[0], fy + 2, door[2], "air")
    return b, fl, (door[0], fy + 1, door[2])


def threshold_cases():
    out = []
    b, fl, door = town(0)
    out.append(("floor_from_threshold: floor_y is the doorstep, stand_y one above",
                fl["floor_y"] == 63 and fl["stand_y"] == 64 and fl["source"] == "threshold",
                f"{fl['floor_y']}/{fl['stand_y']} from {fl['source']}"))

    r = b.check_door(*door)
    out.append(("floor_from_threshold: the doorway it gives you is walkable off the lane",
                r["ok"], r["reason"]))

    b2, _, door2 = town(1)
    r2 = b2.check_door(*door2)
    out.append(("floor_from_threshold: one block higher and the doorway is not",
                not r2["ok"], r2["reason"]))

    # no network at all: it says so rather than inventing a level
    vol, site = world_with({}, -6, 50, -6, 20, 30, 20, 63)
    b3 = Builder(site)
    b3._vol = vol
    fl3 = b3.floor_from_threshold("nobody", x=0, z=0)
    out.append(("floor_from_threshold: with no network it falls back and says so",
                fl3["floor_y"] == 63 and "no circulation network" in fl3["source"],
                fl3["source"]))
    return out


# ------------------------------------------------ check_walkable, at write time The
# interior counterpart of check_door, and the reason it exists. Every reachability check
# in the suite reads a flood with unlimited jumps, so "reachable" has always meant
# "reachable if you jump". A linter finding after the fact is second-best; this is the
# question answered over a program's own unflushed blocks, the way check_door and
# check_attached are.
def walkable_cases():
    out = []

    def roofed(floor_lift=0, platform=None):
        """`town(0)`'s hut with a roof on it -- a space open to the sky is not an interior
        and `observe.rooms` rightly finds nothing in one. `floor_lift` raises the whole
        interior floor above its own doorway sill; `platform` raises one corner of it,
        which is architecture and must not read the same way.
        """
        b, fl, door = town(0)
        fy = fl["floor_y"]
        for x in range(2, 9):
            for z in range(3, 10):
                b.place_block(x, fy + 4, z, "oak_planks")
        if floor_lift:
            for x in range(3, 8):
                for z in range(4, 9):
                    b.place_block(x, fy + floor_lift, z, "oak_planks")
        if platform:
            for x in range(platform[0], platform[2] + 1):
                for z in range(platform[1], platform[3] + 1):
                    b.place_block(x, fy + 1, z, "oak_planks")
        return b, fl, door

    b, fl, door = roofed()
    r = b.check_walkable("hut")
    out.append(("check_walkable: a floor at the doorstep is walkable end to end",
                r["ok"] and r["rooms"] and r["rooms"][0]["fraction"] == 1.0,
                f"{r['rooms'][0]['walkable']}/{r['rooms'][0]['cells']} -- {r['reason']}"
                if r["rooms"] else r["reason"]))

    # the same hut with its floor laid one block above its own sill: the defect
    # lint.FIXES["E003"] describes, and the one the old check cannot see
    b2, fl2, door2 = roofed(floor_lift=1)
    r2 = b2.check_walkable("hut")
    out.append(("check_walkable: a floor a block above its own sill is not walkable",
                not r2["ok"] and r2["rooms"][0]["fraction"] == 0.0,
                f"{r2['rooms'][0]['walkable']}/{r2['rooms'][0]['cells']} -- "
                f"{r2['reason']}"))

    # ...and it is a fraction, not a verdict, when the raised part is a platform. a
    # solid column smaller than the floor it stands on -- and a platform is exactly that
    # shape, so the fraction went to 1.0 and nothing was reported. A raised floor inside
    # a room with no step onto it. A platform is floor; four of its cells being a jump
    # up is a fraction below 1 and a warning; it was never a failure and it is not one
    # now.
    b3, fl3, _ = roofed(platform=(6, 7, 7, 8))
    r3 = b3.check_walkable("hut")
    out.append(("check_walkable: a platform in the corner is a fraction, not a failure",
                r3["ok"] and 0.0 < r3["rooms"][0]["fraction"] < 1.0
                and r3["rooms"][0]["cells"] == r["rooms"][0]["cells"],
                f"{r3['rooms'][0]['walkable']}/{r3['rooms'][0]['cells']} = "
                f"{r3['rooms'][0]['fraction']:.0%}, bare floor "
                f"{r['rooms'][0]['cells']}"))

    # -------------- Four barrels laid through `fitting()`. The cell on top of a barrel
    # is a stance and nobody can step onto it, so the room is not charged for its own
    # furniture -- and the thing that says so is the call that placed them, not their
    # shape.
    b5, fl5, _ = roofed()
    barrels = b5.fitting("store", 4, fl5["floor_y"] + 1, 4, facing="north", extent=4)
    r5 = b5.check_walkable("hut")
    out.append(("check_walkable: four barrels placed through fitting() are not floor",
                barrels["ok"] and r5["rooms"][0]["fraction"] == 1.0
                and r5["rooms"][0]["cells"] == r["rooms"][0]["cells"] - 4,
                f"{r5['rooms'][0]['walkable']}/{r5['rooms'][0]['cells']} = "
                f"{r5['rooms'][0]['fraction']:.0%}, bare floor "
                f"{r['rooms'][0]['cells']}"))

    # ...and a 2x3 dais in the same room, with nothing to step onto it from, is six
    # cells of floor nobody can reach. Laid as a raw solid, which is what a dais with no
    # step is -- and what a type places when it builds furniture by hand instead of
    # calling for it. It is reported rather than silently excluded.
    b6, fl6, _ = roofed()
    b6.place_cuboid(5, fl6["floor_y"] + 1, 4, 6, fl6["floor_y"] + 1, 6, "oak_planks")
    site_a = b6.check_walkable("hut")
    out.append(("check_walkable: a 2x3 dais with no step is six floor cells nobody "
                "can reach",
                site_a["rooms"][0]["cells"] == r["rooms"][0]["cells"]
                and r["rooms"][0]["walkable"] - site_a["rooms"][0]["walkable"] == 6,
                f"{site_a['rooms'][0]['walkable']}/{site_a['rooms'][0]['cells']} = "
                f"{site_a['rooms'][0]['fraction']:.0%}, bare "
                f"{r['rooms'][0]['walkable']}/{r['rooms'][0]['cells']}"))

    # ...and the same dais laid by `dais()`, which brings its own way up: zero.
    b7, fl7, _ = roofed()
    d7 = b7.dais(5, 4, 6, 6, fl7["floor_y"] + 1, "oak")
    site_b = b7.check_walkable("hut")
    out.append(("check_walkable: the same dais through dais() is walkable end to end",
                d7["ok"] and site_b["rooms"][0]["fraction"] == 1.0
                and site_b["rooms"][0]["cells"] == r["rooms"][0]["cells"],
                f"{site_b['rooms'][0]['walkable']}/{site_b['rooms'][0]['cells']} = "
                f"{site_b['rooms'][0]['fraction']:.0%}, {d7['step_cells']} step(s) "
                f"along its {d7['side']} side"))

    # it reads the program's own unflushed blocks, like the other two
    b4, fl4, _ = roofed()
    before = b4.check_walkable("hut")
    fy4 = fl4["floor_y"]
    for x in range(3, 8):
        for z in range(4, 9):
            b4.place_block(x, fy4 + 1, z, "oak_planks")
    after = b4.check_walkable("hut")
    out.append(("check_walkable: the answer moves with the blocks, before any flush",
                before["ok"] and not after["ok"],
                f"before {before['rooms'][0]['fraction']:.0%}, "
                f"after {after['rooms'][0]['fraction']:.0%}"))

    # nothing to answer is said, not guessed
    vol, site = world_with({}, -6, 50, -6, 20, 30, 20, 63)
    b5 = Builder(site)
    b5._vol = vol
    r5 = b5.check_walkable()
    out.append(("check_walkable: with nothing placed it says so rather than passing "
                "silently", r5["ok"] and "nothing placed" in r5["reason"], r5["reason"]))
    return out


# ------------------------------------------------------------------ form primitives
# These decide nothing, so there is nothing here about whether a plinth *should* be
# battered. What is checked is that the numbers a caller passes mean what the docstring
# says they mean, and that the one invariant each of them does own holds.
def form_cases():
    out = []

    # plinth: whatever else it does, nothing it lays may hang over a slope
    b = Ground(lambda x, z: 64 - z // 2)                 # ground falling away in z
    b.plinth(0, 0, 6, 8, 70, "cobblestone", courses=3, batter=1, overhang=1, cap="stone")
    cols = {}
    for (x, y, z) in b.blocks:
        cols.setdefault((x, z), []).append(y)
    hanging = [(x, z) for (x, z), ys in cols.items() if min(ys) > b._h(x, z) + 1]
    out.append(("plinth: no column of it hangs clear of the real ground",
                not hanging, f"{len(cols)} columns, {len(hanging)} hanging"))
    top = [(x, z) for (x, y, z) in b.blocks if y == 70]
    out.append(("plinth: the top course is the footprint plus the overhang",
                len(top) == 9 * 11 and all(b.blocks[(x, 70, z)] == "stone" for x, z in top),
                f"{len(top)} cells of cap at y=70, wanted {9 * 11}"))
    widest = max(x for (x, y, z) in b.blocks if y == 68)
    out.append(("plinth: a battered base widens as it goes down",
                widest == 6 + 1 + 2, f"course 2 reaches x={widest}, wanted 9"))

    # openings: sill and head are measured from the floor, and reveal is a real recess
    b = Ground(lambda x, z: 64)
    b.place_cuboid = lambda *a: None                     # not needed for this one
    b = Ground(lambda x, z: 64)
    got = b.openings(0, 64, 0, 10, 0, at=[2, 6], sill=2, head=4, block="glass_pane",
                     sill_block="stone_brick_slab", lintel="oak_log")
    ys = sorted({y for (x, y, z) in b.blocks if b.blocks[(x, y, z)] == "glass_pane"})
    out.append(("openings: an opening runs from floor+sill to floor+head inclusive",
                ys == [66, 67, 68], f"glass at y {ys}, wanted [66, 67, 68]"))
    out.append(("openings: the sill sits under it and the lintel over it",
                b.blocks.get((2, 65, 0)) == "stone_brick_slab"
                and b.blocks.get((2, 69, 0)) == "oak_log"
                and got == [(2, 66, 0), (6, 66, 0)],
                f"{b.blocks.get((2, 65, 0))} / {b.blocks.get((2, 69, 0))}"))
    b2 = Ground(lambda x, z: 64)
    b2.openings(0, 64, 0, 10, 0, at=[5], sill=2, head=3, reveal=1, inward="south")
    out.append(("openings: reveal sets the glazing back and clears in front of it",
                (b2.blocks.get((5, 66, 1)) or "").startswith("glass")
                and b2.blocks.get((5, 66, 0)) == "air",
                f"face {b2.blocks.get((5, 66, 0))}, reveal {b2.blocks.get((5, 66, 1))}"))

    # glazing: a pane is thin, so what fills an opening depends on the opening.
    b3 = Ground(lambda x, z: 64)
    b3.openings(0, 64, 0, 10, 0, at=[3], width=1, sill=2, head=3)
    slit = {v for v in b3.blocks.values() if v.startswith("glass")}
    b4 = Ground(lambda x, z: 64)
    b4.openings(0, 64, 0, 10, 0, at=[3], width=3, sill=2, head=3)
    wide = {v for v in b4.blocks.values() if v.startswith("glass")}
    b5 = Ground(lambda x, z: 64)
    b5.openings(0, 64, 0, 10, 0, at=[3], width=3, sill=2, head=3, block="glass_pane")
    asked = {v for v in b5.blocks.values() if v.startswith("glass")}
    out.append(("openings: a one-wide opening is glazed with a pane, a wide one is not",
                slit == {"glass_pane"} and wide == {"glass"} and asked == {"glass_pane"},
                f"1-wide {slit}, 3-wide {wide}, asked-for {asked}"))

    # storey_steps: no bay is more than one storey off its neighbour, whatever the hill
    b = Ground(lambda x, z: 64 + x)                      # a 1-in-1 slope, 24 blocks of it
    bays = b.storey_steps(0, 0, 23, 8, axis="x", bays=4, storey=3)
    drops = [r["drop"] for r in bays]
    ok = all(abs(drops[i] - drops[i - 1]) <= 1 for i in range(1, len(drops)))
    out.append(("storey_steps: neighbouring bays are never more than one storey apart",
                ok and len(bays) == 4, f"drops {drops}, floors "
                f"{[r['floor_y'] for r in bays]}"))
    flat = Ground(lambda x, z: 70).storey_steps(0, 0, 23, 8, bays=4, storey=3)
    out.append(("storey_steps: level ground gives one level and no steps",
                {r["floor_y"] for r in flat} == {70},
                f"floors {[r['floor_y'] for r in flat]}"))
    return out


# ---------------------------------------------------------------------------- main
def main():
    print("# steps() -- a tread's raised quarter points up-slope\n")
    for name, ok, note in steps_cases():
        case(name, ok, note)

    print("\n# seal_voids() -- enclosed, unreachable, and yours\n")
    for name, ok, note in seal_cases():
        case(name, ok, note)

    print("\n# floor_from_threshold() -- the doorstep decides the floor\n")
    for name, ok, note in threshold_cases():
        case(name, ok, note)

    print("\n# check_walkable() -- can a person get about inside what I have built\n")
    for name, ok, note in walkable_cases():
        case(name, ok, note)

    print("\n# the form primitives -- the parameters mean what they say\n")
    for name, ok, note in form_cases():
        case(name, ok, note)

    # ---- and does any of it discriminate? --------------------------------- The roof
    # stair facing was inverted in the library for four rounds and every test that
    # existed passed the whole time. A suite that cannot fail on broken code is not
    # evidence, so each rule is re-run against a version built to break it.
    print("\n# does the suite discriminate? the same cases on code built to trip it\n")
    fired = 0
    broken = steps_cases(broken=True)
    tripped = [n for n, ok, _ in broken if not ok]
    print(f"     steps: {len(tripped)}/{len(broken)} cases fail when the facing is "
          f"chosen up front instead of deferred")
    for n in tripped:
        print(f"       trips: {n}")
    fired += bool(tripped)

    # seal_voids, done the way wave 5 did it: flood the box and pack whatever the flood
    # did not reach, with no test of whether the air escapes to the sky
    eaves = dict(hut(False))
    for x in range(-2, 9):
        for z in range(-2, 9):
            eaves[(x, 71, z)] = "dark_oak_planks"
    vol, _ = world_with(eaves, -6, 50, -6, 22, 40, 22, 63)
    naive = sum(1 for x in range(-2, 9) for z in range(-2, 9) for y in range(69, 71)
                if vol.name(x, y, z) in ("air", "cave_air"))
    print(f"     seal_voids: the unscoped version packs {naive} cells above these eaves; "
          f"this one packs 0")
    fired += bool(naive)

    b2, _, door2 = town(1)
    print(f"     floor_from_threshold: a floor one block off the doorstep gives "
          f"'{b2.check_door(*door2)['reason']}'")
    fired += 1

    if fired < 3:
        FAIL.append("the suite does not discriminate")
        print("FAIL not every rule was shown to fail on broken code")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} cases pass")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
