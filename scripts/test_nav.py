"""Every rule in observe.RULES, as a case the walk model must get right.

These are synthetic worlds built in memory -- no server, no world, sub-second. They
check that the model does what it claims to do. They cannot check that what it claims
matches Minecraft; that is what scripts/nav_course.py builds in-world for a human.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import observe  # noqa: E402
from ethoslm.observe import Nav, Volume  # noqa: E402

SX = SY = SZ = 24
X0, Z0 = 0, 0
# The volume deliberately does NOT start at y=0. With y0=0 an absolute half-height and
# an offset into the array are the same number, and a whole class of bug -- one that
# reached the town results -- is invisible.
Y0 = 50
GROUND = 58     # a floor of stone up to y=58, so stances there are at half-height 118


def world(extra: dict, floor_to=None) -> Volume:
    b = {}
    for x in range(SX):
        for z in range(SZ):
            for y in range(Y0, GROUND + 1):
                b[(x, y, z)] = "stone"
    if floor_to:
        for (x, z) in floor_to:
            b.pop((x, GROUND, z), None)
    b.update(extra)
    return Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ)


def nav(extra: dict, **kw) -> Nav:
    return Nav(world(extra), **kw)


def step(n: Nav, a, b) -> str:
    """How you get from stance a to stance b in one move: walk / jump / no."""
    for (nx, nz, ns, jump) in n.neighbours(*a):
        if (nx, nz, ns) == tuple(b):
            return "jump" if jump else "walk"
    return "no"


G = 2 * (GROUND + 1)      # absolute half-height of the ground surface = 118
CASES = []


def case(name, fn):
    CASES.append((name, fn))


# --------------------------------------------------------------- standing & headroom
def t_flat():
    n = nav({})
    assert n.can_stand(5, 5, G), "cannot stand on flat ground"
    assert not n.can_stand(5, 5, G + 1), "standing in mid-air"
    return f"ground stance at half-height {n.ground_stance(5, 5)}"


def t_headroom():
    # a ceiling 2 blocks above the floor is fine; 1 block above is not
    ok = nav({(5, GROUND + 3, 5): "stone"})
    tight = nav({(5, GROUND + 2, 5): "stone"})
    assert ok.can_stand(5, 5, G), "2 blocks of headroom rejected"
    assert not tight.can_stand(5, 5, G), "1 block of headroom accepted"
    return "2 blocks needed, 1 refused"


def t_slab_headroom():
    # a bottom slab at head height leaves only 1.5 blocks: not standable
    n = nav({(5, GROUND + 2, 5): "stone_slab[type=bottom]"})
    assert not n.can_stand(5, 5, G)
    return "slab ceiling at 1.5 refused"


# ------------------------------------------------------------------------ step & jump
def t_step_slab():
    n = nav({(6, GROUND + 1, 5): "stone_slab[type=bottom]"})
    assert step(n, (5, 5, G), (6, 5, G + 1)) == "walk", "slab step needed a jump"
    return "0.5 step walks"


def t_jump_block():
    n = nav({(6, GROUND + 1, 5): "stone"})
    assert step(n, (5, 5, G), (6, 5, G + 2)) == "jump", "1.0 step was not a jump"
    return "1.0 step jumps"


def t_slab_on_block_impossible():
    n = nav({(6, GROUND + 1, 5): "stone",
             (6, GROUND + 2, 5): "stone_slab[type=bottom]"})
    assert step(n, (5, 5, G), (6, 5, G + 3)) == "no", "1.5 step was allowed"
    return "1.5 step refused"


def t_jump_needs_headroom():
    # ceiling 2 above the source: you cannot jump, only walk
    n = nav({(6, GROUND + 1, 5): "stone", (5, GROUND + 3, 5): "stone"})
    assert step(n, (5, 5, G), (6, 5, G + 2)) == "no", "jumped into a ceiling"
    return "jump under a low ceiling refused"


def t_fall():
    n = nav({}, )
    v = world({})
    # dig a 3-deep pit at x=6
    b = {(6, y, 5): "air" for y in range(GROUND - 2, GROUND + 1)}
    n = nav(b)
    assert step(n, (5, 5, G), (6, 5, G - 6)) == "walk", "3-block fall refused"
    return "3-block fall allowed"


def t_deep_fall_refused():
    b = {(6, y, 5): "air" for y in range(GROUND - 4, GROUND + 1)}
    n = nav(b)
    got = [s for (nx, nz, s, _) in n.neighbours(5, 5, G) if (nx, nz) == (6, 5)]
    assert not got, f"fell further than 3 blocks: {got}"
    return "5-block fall refused"


# ---------------------------------------------------------------------------- stairs
def t_stairs_forward():
    """A staircase rising east, correctly oriented (facing = direction of travel)."""
    b = {}
    for i in range(1, 5):
        b[(5 + i, GROUND + i, 5)] = "stone_brick_stairs[facing=east,half=bottom]"
        for y in range(GROUND + 1, GROUND + i):
            b[(5 + i, y, 5)] = "stone_bricks"
    n = nav(b)
    moves = [step(n, (5 + i, 5, G + 2 * i), (6 + i, 5, G + 2 * (i + 1)))
             for i in range(0, 4)]
    assert all(m == "walk" for m in moves), f"correct staircase not walkable: {moves}"
    return "walked up 4 steps, 0 jumps"


def t_stairs_backwards():
    """The same staircase with facing inverted."""
    b = {}
    for i in range(1, 5):
        b[(5 + i, GROUND + i, 5)] = "stone_brick_stairs[facing=west,half=bottom]"
        for y in range(GROUND + 1, GROUND + i):
            b[(5 + i, y, 5)] = "stone_bricks"
    n = nav(b)
    moves = [step(n, (5 + i, 5, G + 2 * i), (6 + i, 5, G + 2 * (i + 1)))
             for i in range(0, 4)]
    assert all(m == "jump" for m in moves), f"backwards staircase read as walkable: {moves}"
    return "same staircase reversed: 4 jumps, 0 walks"


# ----------------------------------------------------------------------- obstructions
def t_fence_blocks():
    n = nav({(6, GROUND + 1, 5): "oak_fence"})
    assert step(n, (5, 5, G), (6, 5, G)) == "no", "walked through a fence"
    assert step(n, (5, 5, G), (6, 5, G + 3)) == "no", "vaulted a fence"
    return "fence blocks at 1.5"


def t_gate_open_closed():
    # A gate is passable either way -- a player opens a shut one. A fence beside it is
    # not, which is the distinction that matters for a walled yard with a gate in it.
    shut = nav({(6, GROUND + 1, 5): "oak_fence_gate[open=false]"})
    open_ = nav({(6, GROUND + 1, 5): "oak_fence_gate[open=true]"})
    assert step(shut, (5, 5, G), (6, 5, G)) == "walk", "shut gate treated as a wall"
    assert step(open_, (5, 5, G), (6, 5, G)) == "walk", "open gate blocked"
    return "gate passable shut and open; fence beside it still blocks"


def t_doors():
    wood = nav({(6, GROUND + 1, 5): "oak_door[half=lower]",
                (6, GROUND + 2, 5): "oak_door[half=upper]"})
    iron = nav({(6, GROUND + 1, 5): "iron_door[half=lower]",
                (6, GROUND + 2, 5): "iron_door[half=upper]"})
    assert step(wood, (5, 5, G), (6, 5, G)) == "walk", "wooden door blocked"
    assert step(iron, (5, 5, G), (6, 5, G)) == "no", "walked through an iron door"
    return "wooden door passable, iron door not"


def t_water():
    # one block of water in the ground: you wade in (dropping to the bed) only if
    # swimming is allowed. On foot the column offers no stance at all.
    b = {(6, GROUND, 5): "water"}
    dry = nav(b)
    swim = nav(b, allow_swim=True)
    assert dry.stances_in_column(6, 5) == [], "found a stance in water"
    assert step(dry, (5, 5, G), (6, 5, G - 2)) == "no", "waded on foot"
    assert step(swim, (5, 5, G), (6, 5, G - 2)) == "walk", "could not wade with allow_swim"
    return "water gives no stance on foot, waded with allow_swim"


def t_lava():
    n = nav({(6, GROUND, 5): "lava"})
    assert step(n, (5, 5, G), (6, 5, G)) == "no", "walked into lava"
    return "lava never traversable"


def t_ladder():
    b = {}
    for y in range(GROUND + 1, GROUND + 6):
        b[(5, y, 5)] = "ladder[facing=east]"
    n = nav(b)
    ups = [s for (nx, nz, s, _) in n.neighbours(5, 5, G) if (nx, nz) == (5, 5)]
    assert ups, "ladder gave no vertical move"
    return f"ladder climbs to {ups}"


# ----------------------------------------------------------------- rooms & enclosure
def t_sealed_room_unreachable():
    """A 5x5 hut with walls and a roof and no door: interior must be unreachable."""
    b = {}
    for x in range(4, 9):
        for z in range(4, 9):
            b[(x, GROUND + 4, z)] = "stone_bricks"          # roof
            for y in range(GROUND + 1, GROUND + 4):
                if x in (4, 8) or z in (4, 8):
                    b[(x, y, z)] = "stone_bricks"           # walls
    n = nav(b)
    inside = (6, 6, G)
    assert n.can_stand(*inside), "no floor inside the hut"
    reach = n.flood([(1, 1, G)])
    assert inside not in reach, "walked into a sealed hut"
    sky, sealed = observe.shelter(n.vol)
    assert sealed[6 - X0, GROUND + 1 - Y0, 6 - Z0], "sealed interior not detected"
    rs = observe.rooms(n, sky, region=(3, 3, 10, 10))
    assert rs and rs[0]["cells"] >= 9, f"interior not found as a room: {rs}"
    return f"sealed hut: interior found ({rs[0]['cells']} cells), correctly unreachable"


def t_door_makes_it_reachable():
    """The same hut with a doorway: interior must now be reachable."""
    b = {}
    for x in range(4, 9):
        for z in range(4, 9):
            b[(x, GROUND + 4, z)] = "stone_bricks"
            for y in range(GROUND + 1, GROUND + 4):
                if x in (4, 8) or z in (4, 8):
                    b[(x, y, z)] = "stone_bricks"
    b[(6, GROUND + 1, 4)] = "oak_door[half=lower]"
    b[(6, GROUND + 2, 4)] = "oak_door[half=upper]"
    n = nav(b)
    reach = n.flood([(1, 1, G)])
    assert (6, 6, G) in reach, "hut with a door still unreachable"
    return f"door opens it: {len(reach)} stances reachable"


def t_door_onto_a_wall():
    """The headline defect: a door that opens onto a blank face you cannot reach."""
    b = {}
    for x in range(4, 9):
        for z in range(4, 9):
            b[(x, GROUND + 4, z)] = "stone_bricks"
            for y in range(GROUND + 1, GROUND + 4):
                if x in (4, 8) or z in (4, 8):
                    b[(x, y, z)] = "stone_bricks"
    b[(6, GROUND + 1, 4)] = "oak_door[half=lower]"
    b[(6, GROUND + 2, 4)] = "oak_door[half=upper]"
    # wall the approach off: a cliff face right outside the door
    for y in range(GROUND + 1, GROUND + 6):
        for z in (2, 3):
            for x in range(3, 10):
                b[(x, y, z)] = "stone"
    n = nav(b)
    reach = n.flood([(1, 1, G)])
    assert (6, 6, G) not in reach, "reached a door that opens onto a blank face"
    return "door blocked by terrain: interior correctly unreachable"


def t_floor_is_the_surface_not_what_stands_on_it():
    """`floor_stances`, on the three things it has to tell apart in one world.

        A room with a barrel the library placed as a fitting, a two-block masonry counter
        the library did not place, and a second storey four blocks up on a plank deck.
    """
    b = {}
    for x in range(4, 12):
        for z in range(4, 12):
            b[(x, GROUND + 9, z)] = "stone_bricks"
            for y in range(GROUND + 1, GROUND + 9):
                if x in (4, 11) or z in (4, 11):
                    b[(x, y, z)] = "stone_bricks"
    for x in range(5, 11):                       # the upper storey's deck
        for z in range(5, 11):
            b[(x, GROUND + 4, z)] = "oak_planks"
    b[(6, GROUND + 1, 6)] = "barrel"
    b[(8, GROUND + 1, 6)] = "cobblestone"        # a counter, in a plain building block
    b[(9, GROUND + 1, 6)] = "cobblestone"
    n = nav(b)
    comp = {(x, z, s) for x in range(5, 11) for z in range(5, 11)
            for s in n.stances_in_column(x, z) if G <= s <= G + 12}
    barrel = {(6, 6, G + 2)}
    counter = {(8, 6, G + 2), (9, 6, G + 2)}
    upper = {(x, z, G + 8) for x in range(5, 11) for z in range(5, 11)}
    assert (barrel | counter) <= comp, \
        f"nothing stands on the barrel or the counter; nothing is tested: {barrel}"

    # the library's own record: the one cell it laid through `fitting()`
    floor = set(observe.floor_stances(n, comp, fittings={(6, GROUND + 1, 6)}))
    assert not (barrel & floor), "the cell on top of a registered fitting is floor"
    assert counter <= floor, \
        f"a solid the library did not place is not floor: {counter - floor}"
    assert upper <= floor, \
        f"the upper storey is not floor: {len(upper & floor)} of {len(upper)} cells"
    assert len(floor) == len(comp) - len(barrel), (len(floor), len(comp))

    # ...and with no record, nothing is furniture
    none = set(observe.floor_stances(n, comp))
    assert none == comp, \
        f"with no register {len(comp) - len(none)} stances were called furniture anyway"
    return (f"{len(comp)} stances -> {len(floor)} floor: the registered barrel top is "
            f"out, the two counter cells and all {len(upper)} of the storey above are "
            f"in; with no register all {len(none)} are")


# ------------------------------------------------------- the palette tables, v2 A2
def t_tables_updated_not_rebuilt():
    """An overlay adds palette entries; it never changes what an entry means.

        So the tables are extended by the rows the write added rather than dropped and
        recomputed over the whole palette -- and the test of that is the two answers being
        the same array, not merely the faster one arriving. A settlement's palette only
        grows, so the old behaviour re-classified every entry once per part.
        
    """
    v = world({})
    before = v.tables()
    n_before = len(v.palette)
    classified = {"entries": 0}
    real = observe._classify_palette

    def counting(palette):
        classified["entries"] += len(palette)
        return real(palette)

    observe._classify_palette = counting
    try:
        v.overlay({(3, GROUND + 1, 3): "oak_fence",
                   (4, GROUND + 1, 4): "water",
                   (5, GROUND + 1, 5): "oak_door[facing=north,half=lower,open=false]",
                   (6, GROUND + 1, 6): "stone_brick_stairs[facing=east]",
                   (7, GROUND + 1, 7): "minecraft:lantern",
                   (3, GROUND + 2, 3): "oak_fence"})          # a repeat, not a new row
        after = v.tables()
    finally:
        observe._classify_palette = real
    added = len(v.palette) - n_before
    assert after is before, "the tables were rebuilt, not extended"
    assert classified["entries"] == added, \
        f"{classified['entries']} entries classified for {added} new palette entries"
    fresh = Volume(v.x0, v.y0, v.z0, v.codes.copy(), list(v.palette)).tables()
    for k in sorted(fresh):
        assert len(after[k]) == len(v.palette), \
            f"{k} has {len(after[k])} rows for {len(v.palette)} palette entries"
        assert (after[k] == fresh[k]).all(), \
            f"{k} differs from a table built over the whole palette"
    # ...and the rows are right, not merely equal to each other
    i = v.palette.index("oak_fence")
    j = v.palette.index("water")
    assert after["tall"][i] and not after["liquid"][i], "a fence is tall and dry"
    assert after["liquid"][j], "water is not liquid"
    return (f"{added} entries added to a palette of {n_before}, {added} classified, "
            f"{len(fresh)} columns identical to a full rebuild")


case("tables_updated_not_rebuilt", t_tables_updated_not_rebuilt)


for fn in (t_flat, t_headroom, t_slab_headroom, t_step_slab, t_jump_block,
           t_slab_on_block_impossible, t_jump_needs_headroom, t_fall,
           t_deep_fall_refused, t_stairs_forward, t_stairs_backwards, t_fence_blocks,
           t_gate_open_closed, t_doors, t_water, t_lava, t_ladder,
           t_sealed_room_unreachable, t_door_makes_it_reachable, t_door_onto_a_wall,
           t_floor_is_the_surface_not_what_stands_on_it):
    case(fn.__name__[2:], fn)


def main():
    bad = 0
    for name, fn in CASES:
        try:
            note = fn()
            print(f"ok   {name:28s} {note}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:28s} {e}")
        except Exception as e:
            bad += 1
            print(f"ERR  {name:28s} {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} movement cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
