"""Each linter check, against a world built to trip it -- and one built not to.

A check that never fires is decoration; a check that always fires is noise. Every case
here asserts both directions where it can: the defect world reports the code, the clean
world does not. Synthetic volumes in memory, no server, sub-second.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import lint  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

SX = SY = SZ = 28
X0, Y0, Z0 = 0, 50, 0
GROUND = 58
PLOT = [{"x0": 3, "z0": 3, "x1": 12, "z1": 12, "label": "hut"}]


def base() -> dict:
    b = {}
    for x in range(SX):
        for z in range(SZ):
            for y in range(Y0, GROUND + 1):
                b[(x, y, z)] = "stone"
    return b


def hut(b, door=True, lit=True, x0=4, z0=4, x1=9, z1=9):
    """A walled, roofed hut. Optionally with a doorway and a lantern."""
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b[(x, GROUND + 4, z)] = "stone_bricks"
            for y in range(GROUND + 1, GROUND + 4):
                if x in (x0, x1) or z in (z0, z1):
                    b[(x, y, z)] = "stone_bricks"
    if door:
        cx = (x0 + x1) // 2
        b[(cx, GROUND + 1, z0)] = "oak_door[facing=north,half=lower]"
        b[(cx, GROUND + 2, z0)] = "oak_door[facing=north,half=upper]"
    if lit:
        b[(x0 + 2, GROUND + 3, z0 + 2)] = "lantern[hanging=true]"
    return b


def ctx_for(blocks, plots=PLOT, fittings=None):
    """`fittings` is the library's record of what it laid as furniture. These worlds are
    block dictionaries with no builder behind them, so the register is handed in by the
    cases that are about it, and is None everywhere else, which is what a cached town
    whose builder is long gone reads as.
    """
    return lint.Context.build(
        Volume.from_blocks(blocks, X0, Y0, Z0, SX, SY, SZ), plots, fittings=fittings)


def codes(report):
    return {f.code for f in report.findings}


CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


@case
def t_clean_hut_is_clean():
    """The control. A lit hut with a door on open ground must report no errors."""
    r = lint.lint(ctx_for(hut(base())))
    assert not r.errors, f"clean hut reported errors: {[f.message for f in r.errors]}"
    return f"no errors, {len(r.warnings)} warnings"


@case
def t_e001_invalid_block():
    b = hut(base())
    b[(6, GROUND + 1, 6)] = "cauldron[level=3]"
    r = lint.lint(ctx_for(b), only={"E001"})
    assert "E001" in codes(r), "invalid block state not caught"
    return r.findings[0].message


@case
def t_e002_no_door_at_all():
    """A sealed hut: its doorway does not exist, so nothing can be entered."""
    r = lint.lint(ctx_for(hut(base(), door=False)), only={"E002", "E003"})
    assert "E003" in codes(r), "sealed interior not reported unreachable"
    return f"{len(r.findings)} findings; sealed interior caught by E003"


@case
def t_e002_door_walled_off():
    """The headline defect: a door that opens onto a face nobody can reach."""
    b = hut(base())
    # A face across the whole approach, hard against the door. Walling only part of the
    # width does not isolate anything -- the first version of this left the strip in
    # front of the door connected round both ends of the hut, and nothing fired.
    for y in range(GROUND + 1, GROUND + 7):
        for x in range(SX):
            b[(x, y, 3)] = "stone"
    r = lint.lint(ctx_for(b), only={"E002"})
    assert "E002" in codes(r), "door onto a blank face not caught"
    return r.findings[0].message


@case
def t_e003_clean_hut_room_reachable():
    r = lint.lint(ctx_for(hut(base())), only={"E003"})
    assert not r.findings, f"reachable room reported unreachable: {codes(r)}"
    return "room with a door not reported"


# ----------------------------------------------- E011/W011. They were right, and no
# check could see it: every interior reachability check reads `Context.from_anywhere`,
# which is `flood(max_jumps=None)`, so "reachable" has always meant "reachable if you
# jump". Each case below asserts the defect fires, the clean version is silent, **and**
# that E003 -- the check that was supposed to cover this -- stays quiet throughout. That
# last assertion is the discrimination: without it these cases would pass on the code
# that has the bug.

TALL_PLOT = [{"x0": 3, "z0": 3, "x1": 14, "z1": 14, "label": "hall"}]


def tall_hut(b, floor=0, x0=4, z0=4, x1=11, z1=11, head=7, door_head=0):
    """A hut with room to stand on a raised floor, and a door in its north wall.

        `floor` raises the interior floor that many blocks above the doorway sill, which is
        the defect `lint.FIXES["E003"]` describes and cannot detect. `door_head` blocks the
        doorway at head height.

        The doorway opening is deliberately **three** blocks tall. A jump needs
        `JUMP_CLEAR` = 3 blocks of headroom at the cell you jump *from*, so a two-block
        doorway makes a raised floor unreachable by any means -- which E003 does catch, and
        which is therefore the wrong world to test the new check against. This hut is the
        one where the old check is satisfied and the building is still unusable.
        
    """
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b[(x, GROUND + head, z)] = "stone_bricks"
            for y in range(GROUND + 1, GROUND + head):
                if x in (x0, x1) or z in (z0, z1):
                    b[(x, y, z)] = "stone_bricks"
    for x in range(x0 + 1, x1):
        for z in range(z0 + 1, z1):
            for y in range(GROUND + 1, GROUND + 1 + floor):
                b[(x, y, z)] = "stone_bricks"
    cx = (x0 + x1) // 2
    b[(cx, GROUND + 1, z0)] = "oak_door[facing=north,half=lower]"
    b[(cx, GROUND + 2, z0)] = "oak_door[facing=north,half=upper]"
    b.pop((cx, GROUND + 3, z0), None)                   # headroom to jump from
    if door_head:
        b[(cx, GROUND + 2, z0)] = "stone_bricks"
    b[(x0 + 2, GROUND + head - 1, z0 + 2)] = "lantern[hanging=true]"
    return b


def walk_rows(b, plots=TALL_PLOT, fittings=None):
    ctx = ctx_for(b, plots, fittings=fittings)
    return ctx, ctx.interior_walk(), lint.lint(ctx, only={"E003", "E011", "W011"})


@case
def t_e011_floor_a_block_above_its_own_sill():
    """The case the E003 advice text describes and E003 cannot see. Same hut with the
    floor at sill level must be silent, and E003 must be silent in both."""
    ctx, bad, r = walk_rows(tall_hut(base(), floor=1))
    assert "E011" in codes(r), f"a floor above its own sill was not caught: {codes(r)}"
    assert max(x["fraction"] for x in bad) == 0.0, bad
    assert "E003" not in codes(r), \
        "E003 fired, so this case does not discriminate -- it was already covered"

    ctx2, good, r2 = walk_rows(tall_hut(base(), floor=0))
    assert not r2.findings, f"a floor at sill level was reported: {codes(r2)}"
    assert min(x["fraction"] for x in good) == 1.0, good
    return (f"raised floor {bad[0]['walkable']}/{bad[0]['cells']} walkable -> E011; "
            f"level floor {good[0]['walkable']}/{good[0]['cells']} -> silent")


@case
def t_e011_is_not_reachability_with_jumping():
    """The discrimination, stated directly. The *only* difference between the two
    measures is `max_jumps`, and on the raised-floor hut they disagree completely: the
    old one says the whole room is reachable, the new one says none of it is."""
    ctx, rows, _ = walk_rows(tall_hut(base(), floor=1))
    room = next(r for r in ctx.rooms if r.get("plot"))
    jumping = set(map(tuple, room["stances"])) & set(ctx.from_anywhere)
    assert len(jumping) == len(room["stances"]), \
        "the jump-allowed flood does not reach this room, so nothing is being contrasted"
    assert rows[0]["walkable"] == 0, rows
    return (f"jump-allowed reaches {len(jumping)}/{len(room['stances'])}, "
            f"walk-only reaches 0")


@case
def t_w011_a_raised_platform_is_architecture_not_an_error():
    """Severity has to be earned. A sleeping platform in the corner is a design, and a
        check that made it an error would be the fourth check in this project to deform a
        design. It comes back as a fraction and as a warning, or as nothing at all.
    """
    bare_rows = walk_rows(tall_hut(base(), floor=0))[1]
    b = tall_hut(base(), floor=0)
    for x in range(8, 11):                      # a one-block dais in the far corner
        for z in range(8, 11):
            b[(x, GROUND + 1, z)] = "stone_bricks"
    ctx, rows, r = walk_rows(b)
    assert "E011" not in codes(r), \
        f"a raised platform was called an error: {[f.message for f in r.errors]}"
    assert rows[0]["fraction"] > 0.5, rows
    assert rows[0]["fraction"] < 1.0, "the platform is not raised; nothing is measured"
    assert rows[0]["cells"] == bare_rows[0]["cells"], \
        (f"a platform is floor, so the count does not move: "
         f"{bare_rows[0]['cells']} -> {rows[0]['cells']}")
    return (f"platform: {rows[0]['walkable']}/{rows[0]['cells']} = "
            f"{rows[0]['fraction']:.0%} walkable, reported as a fraction, no error")


@case
def t_a0i_furniture_is_what_the_library_placed_as_furniture():
    """Four barrels and a hay bale standing on a room's own floor. Every cell on top of
        them is a stance -- a barrel is a full cube -- and none of them can be stepped onto,
        so the room used to be charged for its own furniture: five cells of floor nobody
        could reach, on a build with nothing wrong with it. It is where every remaining
        walkability miss in the record came from, on three independently written types and
        on all 144 instances of the conformance suite.
    """
    bare = walk_rows(tall_hut(base(), floor=0))[1][0]
    b = tall_hut(base(), floor=0)
    placed = [(5, 5, "barrel"), (9, 5, "barrel"), (5, 9, "barrel"),
              (9, 10, "barrel"), (7, 10, "hay_block")]
    for (x, z, blk) in placed:
        b[(x, GROUND + 1, z)] = blk
    register = {(x, GROUND + 1, z) for (x, z, _blk) in placed}
    ctx, rows, r = walk_rows(b, fittings=register)
    tops = [(x, z) for (x, z, _blk) in placed]
    room = next(rm for rm in ctx.rooms if rm.get("plot"))
    floor = {(x, z) for (x, z, _s) in room["floor"]}
    assert not any(c in floor for c in tops), \
        f"a cell on top of a registered fitting is still floor: {sorted(set(tops) & floor)}"
    assert rows[0]["cells"] == bare["cells"] - 5, \
        (f"five fittings stand on five floor cells, so the floor should fall by five: "
         f"{bare['cells']} -> {rows[0]['cells']}")
    assert rows[0]["fraction"] == bare["fraction"] == 1.0, (rows[0], bare)
    assert not r.findings, f"a furnished room was reported: {codes(r)}"

    # ...and the same world with no register: the same five cells are floor, they are
    # unreachable, and the room is charged for them.
    _c2, rows2, r2 = walk_rows(b)
    assert rows2[0]["cells"] == bare["cells"], (rows2[0], bare)
    assert rows2[0]["walkable"] == bare["walkable"] - 5, (rows2[0], bare)
    assert rows2[0]["fraction"] < 1.0, rows2
    assert not r2.errors, f"five cells of furniture is not an error: {codes(r2)}"
    return (f"bare {bare['walkable']}/{bare['cells']} = {bare['fraction']:.0%}; "
            f"furnished, registered {rows[0]['walkable']}/{rows[0]['cells']} = "
            f"{rows[0]['fraction']:.0%}; furnished, no register "
            f"{rows2[0]['walkable']}/{rows2[0]['cells']} = {rows2[0]['fraction']:.0%}")


@case
def t_a0g_a_loft_you_can_only_jump_to_is_still_floor():
    """The case the rule must not hide, and the reason it is not "whatever you can walk
    to and back from".

    A mezzanine laid across the top of a room, reachable only by jumping, is floor: the
    ceiling of the room below it is air, not a solid stack up from its floor. Charging
    for it is the whole job of E011 and W011, and a floor definition that quietly
    dropped an unreachable storey would report a sealed loft as a perfect building."""
    b = tall_hut(base(), floor=0)
    for x in range(6, 11):                      # a mezzanine over the far half
        for z in range(7, 11):
            b[(x, GROUND + 3, z)] = "oak_planks"
    ctx, rows, r = walk_rows(b)
    up = [(x, z, s) for rm in ctx.rooms if rm.get("plot")
          for (x, z, s) in rm["floor"] if (s - 1) // 2 == GROUND + 3]
    assert len(up) == 20, f"the mezzanine is not floor: {len(up)} of 20 cells"
    loft = next(rw for rw in rows if rw["bbox"][1] > GROUND + 1)
    assert loft["cells"] == 20 and loft["walkable"] == 0, loft
    assert "E011" in codes(r) or "E003" in codes(r), \
        f"an unreachable mezzanine was not reported at all: {codes(r)}"
    return (f"the mezzanine's {len(up)} cells are floor and unreachable: "
            f"{loft['walkable']}/{loft['cells']}, {sorted(codes(r))}")


@case
def t_a0g_a_fern_is_not_a_wall_and_roots_are_ground():
    """What is *not* the building's interior.

        Built here as the thing it is: a patch of swamp floor under a canopy, on a plot,
        with a building beside it.
    """
    b = hut(base())
    for x in range(11, 14):                     # a patch of swamp inside the plot
        for z in range(11, 14):
            b[(x, GROUND, z)] = "mud"
            b[(x, GROUND + 4, z)] = "mangrove_leaves"
    b[(11, GROUND + 1, 11)] = "fern"
    b[(13, GROUND + 1, 13)] = "short_grass"
    for (x, z) in ((10, 12), (14, 12), (12, 10), (12, 14)):
        for y in range(GROUND + 1, GROUND + 4):
            b[(x, y, z)] = "mangrove_roots"
    ctx = ctx_for(b)
    swamp = [r for r in ctx.rooms
             if r["bbox"][0] >= 10 and r["bbox"][3] <= 14 and r["bbox"][1] <= GROUND + 2]
    assert swamp, "the swamp patch is not a sheltered component at all; nothing is tested"
    assert all(r["made"] < lint.Context.MADE for r in swamp), \
        f"a patch of swamp reads as somebody's room: made={[r['made'] for r in swamp]}"
    assert all(r.get("plot") is None for r in swamp), \
        f"a patch of swamp is attributed to a plot: {[r.get('plot') for r in swamp]}"
    return (f"{len(swamp)} swamp component(s) walled by roots, a fern and a tuft of "
            f"grass: made={[r['made'] for r in swamp]} < {lint.Context.MADE}, "
            f"attributed to no plot")


@case
def t_e011_doorway_obstructed_at_head_height():
    """A doorway you cannot get your head through. The sill is fine, the floor is fine,
    and you still cannot walk in.

    Unlike the cases above this one is *not* exclusive to the new check: an opening
    blocked at head height cannot be jumped through either, so E003 catches it too.
    It is here because the spec asks for it and because E011 is the finding that says
    which room and what fraction. Discrimination is carried by the raised floor, the
    platform and the furniture."""
    ctx, rows, r = walk_rows(tall_hut(base(), floor=0, door_head=1))
    assert "E011" in codes(r), f"a blocked doorway head was not caught: {codes(r)}"
    assert rows[0]["walkable"] == 0, rows
    clean = walk_rows(tall_hut(base(), floor=0))[2]
    assert not clean.findings, "the unobstructed doorway is not silent"
    return (f"head-blocked doorway: {rows[0]['walkable']}/{rows[0]['cells']} walkable "
            f"(E003 fires here too)")


@case
def t_w011_furniture_across_the_only_route():
    """`fitting()` puts furniture in a room. A barrel in the one gap in a partition is
    a wall you have to hop, so half the room stops being somewhere you can walk to --
    and E003 stays silent throughout, because you *can* hop it."""
    b = tall_hut(base(), floor=0)
    # Full height to the roof: a partition that stops short has a standable top, which
    # `observe.rooms` correctly reports as its own little unreachable room and which
    # would make this case pass for the wrong reason.
    for x in range(5, 10):                      # a partition with one gap, at x=10
        for y in range(GROUND + 1, GROUND + 7):
            b[(x, y, 6)] = "stone_bricks"
    open_rows, open_r = walk_rows(dict(b))[1:]
    assert not open_r.findings, \
        f"the partition alone already trips the check: {codes(open_r)}"

    b[(10, GROUND + 1, 6)] = "barrel"           # what fitting() puts in a room
    ctx, rows, r = walk_rows(b)
    shut = min(x["fraction"] for x in rows)
    assert "W011" in codes(r), \
        f"furniture across the only route was not reported: {codes(r)}"
    assert "E003" not in codes(r) and "E011" not in codes(r), \
        f"a hoppable barrel was called unreachable: {codes(r)}"
    assert shut < min(x["fraction"] for x in open_rows), \
        "the fraction did not drop when the route was blocked"
    return (f"route open {min(x['fraction'] for x in open_rows):.0%} -> "
            f"blocked by furniture {shut:.0%}, reported as a warning")


@case
def t_e011_says_nothing_about_a_porch():
    """The guard every interior check in here needs. A roofed but open-sided porch is
    not somewhere you can be shut out of, and on a badlands site the alternative is
    dozens of findings that are all cliff overhangs."""
    b = base()
    for x in range(4, 10):                      # a roof on four posts, open all round
        for z in range(4, 10):
            b[(x, GROUND + 4, z)] = "stone_bricks"
    for (x, z) in ((4, 4), (4, 9), (9, 4), (9, 9)):
        for y in range(GROUND + 1, GROUND + 4):
            b[(x, y, z)] = "stone_bricks"
    r = lint.lint(ctx_for(b), only={"E011", "W011"})
    assert not r.findings, f"a porch was reported: {[f.message for f in r.findings]}"
    return "open porch reports nothing"


@case
def t_e004_backwards_stair():
    b = hut(base())
    for i in range(1, 4):                          # a staircase facing the wrong way
        b[(14 + i, GROUND + i, 14)] = "stone_brick_stairs[facing=west,half=bottom]"
        for y in range(GROUND + 1, GROUND + i):
            b[(14 + i, y, 14)] = "stone_bricks"
    r = lint.lint(ctx_for(b), only={"E004"})
    assert "E004" in codes(r), "backwards stairs not caught"
    return f"{len(r.findings)} backwards stairs"


@case
def t_e005_unconnected_fence():
    b = hut(base())
    b[(16, GROUND + 1, 16)] = "stone"
    for dx in (1, 2, 3):
        b[(16 + dx, GROUND + 1, 16)] = ("oak_fence[east=false,north=false,"
                                        "south=false,west=false]")
    r = lint.lint(ctx_for(b), only={"E005"})
    assert "E005" in codes(r), "unjoined fence not caught"
    return r.findings[0].message


@case
def t_e005_connected_fence_is_clean():
    b = hut(base())
    b[(16, GROUND + 1, 16)] = "stone"
    b[(17, GROUND + 1, 16)] = "oak_fence[east=true,north=false,south=false,west=true]"
    b[(18, GROUND + 1, 16)] = "stone"
    r = lint.lint(ctx_for(b), only={"E005"})
    assert not r.findings, f"properly joined fence reported: {r.findings[0].message}"
    return "joined fence not reported"


@case
def t_e005_leaves_are_not_a_missed_join():
    """A rail beside a canopy is not an unjoined fence. Leaves have no solid side face and
    a fence ignores them.
    """
    b = hut(base())
    b[(16, GROUND + 1, 16)] = "jungle_leaves[persistent=false]"
    b[(17, GROUND + 1, 16)] = ("cobblestone_wall[east=none,north=none,"
                               "south=none,west=none,up=true]")
    b[(18, GROUND + 1, 16)] = "oak_stairs[facing=east,half=bottom]"
    r = lint.lint(ctx_for(b), only={"E005"})
    assert not r.findings, f"leaves/stairs counted as a join: {r.findings[0].message}"
    return "a wall between leaves and a stair reports nothing"


@case
def t_e006_overlapping_plots():
    plots = PLOT + [{"x0": 8, "z0": 8, "x1": 16, "z1": 16, "label": "other"}]
    r = lint.lint(ctx_for(hut(base()), plots), only={"E006"})
    assert "E006" in codes(r), "overlapping plots not caught"
    return r.findings[0].message


@case
def t_a3_an_edge_is_its_swept_line_and_not_its_bounding_box():
    """A wall round a house does not overlap the house."""
    from ethoslm import pipeline
    wall = {"kind": "edge", "name": "ring", "width": 1,
            "path": [[2, 2], [20, 2], [20, 20], [2, 20], [2, 2]]}
    row = pipeline.part_registry_row(wall)
    assert row["rects"] and len(row["rects"]) == 4, row
    plots = PLOT + [row]
    r = lint.lint(ctx_for(hut(base()), plots), only={"E006"})
    assert not r.findings, [f.message for f in r.findings]
    # ...and a plot the wall actually runs through is still an overlap
    on_it = plots + [{"x0": 18, "z0": 10, "x1": 22, "z1": 14, "label": "on_the_wall"}]
    r2 = lint.lint(ctx_for(hut(base()), on_it), only={"E006"})
    assert "E006" in codes(r2), "a plot the wall runs through was not caught"
    # ...and the wall no longer owns the rooms of the town it encloses
    ctx = ctx_for(hut(base()), plots)
    assert ctx.plot_at(6, 6) == "hut", ctx.plot_at(6, 6)
    return (f"a 4-segment ring round a hut: 0 E006, {len(row['rects'])} rectangles; "
            f"a plot on the line is still caught; plot_at(6,6) is the hut")


@case
def t_a4_interior_walk_applies_the_enclosed_filter():
    """A porch is not charged as a room by the third reader either."""
    b = base()
    for x in range(4, 11):                      # a roof on four posts, open all round
        for z in range(4, 11):
            b[(x, GROUND + 4, z)] = "stone_bricks"
    for (x, z) in ((4, 4), (10, 4), (4, 10), (10, 10)):
        for y in range(GROUND + 1, GROUND + 4):
            b[(x, y, z)] = "oak_log"
    b[(7, GROUND + 1, 4)] = "oak_door[facing=north,half=lower]"
    b[(7, GROUND + 2, 4)] = "oak_door[facing=north,half=upper]"
    ctx = ctx_for(b)
    porches = [r for r in ctx.rooms if r["plot"]]
    assert porches, "the covered space was not found at all"
    assert all(r["enclosure"] < lint.Context.ENCLOSED for r in porches), \
        [r["enclosure"] for r in porches]
    assert not ctx.interior_walk(), \
        f"a porch is still charged as a room: {ctx.interior_walk()}"
    # ...and a walled hut, which is what the filter is meant to keep, is still counted
    hut_ctx = ctx_for(hut(base()))
    assert [r for r in hut_ctx.rooms if r["plot"]], "the hut has no room at all"
    return (f"a porch at enclosure {porches[0]['enclosure']} with a door in it is not "
            f"a row of interior_walk; the threshold is {lint.Context.ENCLOSED}")


@case
def t_a1_a_cave_under_a_plot_is_not_the_building_above_it():
    """Interior floor is the building's floor.

        The patch is a hut on ten blocks of rock with a cavern hollowed out under it and a
        wall running past it. Three rooms are found; one of them is somebody's interior.
    """
    b = hut(base())
    # ...a cavern under the hut's own plot: air in the rock, roofed and one wall lined
    # by the pad's own fill, which is what puts its `made` over the threshold.
    for x in range(5, 11):
        for z in range(5, 11):
            for y in range(52, 55):
                b.pop((x, y, z), None)
    for y in range(52, 55):
        for i in range(4, 12):
            b[(4, y, i)] = "stone_bricks"
            b[(i, y, 4)] = "stone_bricks"
    wall = {"kind": "edge", "name": "ring", "width": 1,
            "path": [[16, 2], [16, 24]]}
    from ethoslm import pipeline
    row = pipeline.part_registry_row(wall)
    # ...and a covered space on the wall's own line, which is masonry and not an inside
    for z in range(6, 14):
        for y in range(GROUND + 1, GROUND + 5):
            for x in (15, 17):
                b[(x, y, z)] = "stone_bricks"
        b[(16, GROUND + 4, z)] = "stone_bricks"

    # As the registry looked before this: a wall is a row like any other and nothing
    # says at what level the hut was sited.
    was = [dict(PLOT[0]), dict(row, kind="plot")]
    ctx = ctx_for(b, was)
    charged = [r for r in ctx.rooms if r["plot"]]
    cave = [r for r in ctx.rooms if r["bbox"][4] < GROUND]
    assert cave, "the cavern was not found at all"
    assert cave[0]["made"] >= lint.Context.MADE, \
        f"the cavern reads made {cave[0]['made']}, so `made` alone would have caught it"
    assert cave[0]["plot"] == "hut", \
        "the cavern is not charged to the hut, so this case proves nothing"
    assert any(r["plot"] == "ring" for r in charged), \
        "the wall's covered line is not charged to it either, so nor does that"

    # ...and as it looks now: a wall is an edge, and the hut carries the level the
    # library sited it at.
    now = [dict(PLOT[0], y0=GROUND + 1), row]
    ctx2 = ctx_for(b, now)
    got = [r for r in ctx2.rooms if r["plot"]]
    assert {r["plot"] for r in got} == {"hut"}, \
        f"attributed to {sorted({r['plot'] for r in got})}, not the hut alone"
    assert all(r["bbox"][4] >= GROUND for r in got), "a room under the hut survived"
    walk = set(ctx2.from_outdoors)
    cells = sum(len(r["floor"]) for r in got)
    reach = sum(len(set(map(tuple, r["floor"])) & walk) for r in got)
    dirty = sum(len(r["floor"]) for r in charged)
    assert reach == cells, f"the hut itself reads {reach}/{cells}"
    assert cells < dirty, f"nothing was excluded: {cells} of {dirty}"
    # ...and the vertical rule is waived, not guessed at, where nothing recorded a level
    silent = [r for r in ctx_for(b, [dict(PLOT[0]), row]).rooms if r["plot"]]
    assert len(silent) > len(got), "a round with no y0 on its registry moved"
    return (f"3 rooms, 1 interior: the hut reads {reach}/{cells} = 100%, where the "
            f"patch read {reach}/{dirty} = {round(100 * reach / dirty)}%; the cavern "
            f"at made {cave[0]['made']} and the wall's line are landscape")


@case
def t_w002_interior_without_a_door():
    r = lint.lint(ctx_for(hut(base(), door=False)), only={"W002"})
    assert "W002" in codes(r), "interior with no door not warned"
    return r.findings[0].message


@case
def t_w004_dark_room():
    r = lint.lint(ctx_for(hut(base(), lit=False)), only={"W004"})
    assert "W004" in codes(r), "unlit room not warned"
    lit = lint.lint(ctx_for(hut(base(), lit=True)), only={"W004"})
    assert not lit.findings, "lit room warned as dark"
    return f"unlit warned, lit not: {r.findings[0].message}"


@case
def t_preflight_catches_bad_ids():
    src = ('place_block(1, 2, 3, "cut_red_sandstone_stairs")\n'
           'place_cuboid(0, 0, 0, 1, 1, 1, "chain")\n'
           'wall(0, 0, 0, 4, 3, 0, "stone_bricks", post="oak_log")\n'
           'roof(0, 0, 8, 8, 70, "spruce", style="gable")\n'      # mat, not a block id
           'label = "chain"\n')                                   # not a block argument
    r = lint.preflight(src)
    got = {f.detail["state"] for f in r.findings}
    assert got == {"cut_red_sandstone_stairs", "chain"}, f"got {got}"
    ok = lint.preflight('place_block(1, 2, 3, "stone_bricks")')
    assert not ok.findings, "valid id flagged"
    bad = lint.preflight("place_block(1, 2, 3,")
    assert bad.findings and "does not parse" in bad.findings[0].message
    return "2 bad ids caught; style/material/non-argument strings not flagged"


@case
def t_s001_only_with_a_pass_palette():
    """Fires on what a pass placed; silent on the landscape it placed it in."""
    vol = Volume.from_blocks(hut(base()), X0, Y0, Z0, SX, SY, SZ)
    wide = {f"{m}_planks": 1 for m in
            ("oak", "spruce", "birch", "jungle", "acacia", "cherry", "mangrove")}
    fired = lint.lint(lint.Context.build(vol, PLOT, placed=wide), only={"S001"})
    narrow = lint.lint(lint.Context.build(vol, PLOT,
                                          placed={"oak_planks": 1, "stone_bricks": 1}),
                       only={"S001"})
    world_only = lint.lint(lint.Context.build(vol, PLOT), only={"S001"})
    assert fired.findings, "wide palette not reported"
    assert not narrow.findings, "narrow palette reported"
    assert not world_only.findings, "fired with no pass palette -- that is landscape"
    return fired.findings[0].message


@case
def t_s001_measures_against_the_declared_voice():
    """With a voice declared, palette drift stops being a count and becomes a contract."""
    vol = Volume.from_blocks(hut(base()), X0, Y0, Z0, SX, SY, SZ)
    voice = {"wall": "cobblestone", "roof": "dark_oak", "trim": "spruce"}
    on = {"cobblestone": 400, "cobblestone_stairs": 40, "mossy_cobblestone": 30,
          "dark_oak_stairs": 90, "spruce_slab": 12}          # all inside the voice
    off = dict(on, quartz_block=60, prismarine=8)            # two that are not
    clean = lint.lint(lint.Context.build(vol, PLOT, placed=on, voice=voice), only={"S001"})
    assert not clean.findings, f"voice-compliant palette reported: {clean.findings[0].message}"
    r = lint.lint(lint.Context.build(vol, PLOT, placed=off, voice=voice), only={"S001"})
    assert "S001" in codes(r), "materials outside the voice not reported"
    assert set(r.findings[0].detail["extra"]) == {"quartz", "prismarine"}, \
        r.findings[0].detail["extra"]
    return r.findings[0].message


@case
def t_s002_camouflage():
    """A town the colour of its own hillside. Not a beauty judgement -- material identity
    between the build and the ground it stands on.
    """
    b = base()
    for x in range(SX):                     # a landscape made of sandstone
        for z in range(SZ):
            b[(x, GROUND, z)] = "sandstone"
    vol = Volume.from_blocks(hut(b), X0, Y0, Z0, SX, SY, SZ)
    same = lint.lint(lint.Context.build(vol, PLOT, placed={"sandstone": 900,
                                                           "dark_oak_stairs": 80}),
                     only={"S002"})
    assert "S002" in codes(same), "a build made of the ground it stands on not reported"
    diff = lint.lint(lint.Context.build(vol, PLOT, placed={"dark_oak_planks": 900,
                                                           "sandstone": 80}),
                     only={"S002"})
    assert not diff.findings, f"a contrasting build reported: {diff.findings[0].message}"
    return same.findings[0].message


@case
def t_enclosure_excludes_a_porch():
    """A roof on posts is covered but open. It is not a room you can be shut out of."""
    b = base()
    for x in range(4, 11):
        for z in range(4, 11):
            b[(x, GROUND + 4, z)] = "stone_bricks"          # roof, no walls
    for (x, z) in ((4, 4), (10, 4), (4, 10), (10, 10)):
        for y in range(GROUND + 1, GROUND + 4):
            b[(x, y, z)] = "oak_log"
    ctx = ctx_for(b)
    porches = [r for r in ctx.rooms if r["plot"]]
    assert porches, "the covered space was not found at all"
    assert all(r["enclosure"] < lint.Context.ENCLOSED for r in porches), \
        f"a porch scored as enclosed: {[r['enclosure'] for r in porches]}"
    assert not lint.lint(ctx, only={"E003"}).findings, "porch reported unenterable"
    return f"porch enclosure {porches[0]['enclosure']}, below {lint.Context.ENCLOSED}"


# --- circulation: the checks the ordering change is for -----------------------------

def lane(b, cells, y=GROUND, thresholds=()):
    """Lay a lane on the flat and declare it, the way the circulation pass would."""
    from ethoslm import circulate
    for (x, z) in cells:
        b[(x, y, z)] = "cobblestone"
    net = circulate.Network(
        {(x, z): {"y": y, "rank": 1, "face": None} for x, z in cells},
        [circulate.Threshold(t[0], t[1], t[2], y, t[3],
                             (t[1], y + 1, t[2] - 1)) for t in thresholds])
    return net


LANE = [(x, 15) for x in range(2, 26)]


@case
def t_e007_one_network_is_clean():
    b = base()
    net = lane(b, LANE)
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     PLOT, network=net), only={"E007"})
    assert not r.findings, f"a single lane reported split: {[f.message for f in r.findings]}"
    return "24-cell lane reported as one network"


@case
def t_e007_split_network():
    """Two runs of lane with a wall between them. Each run is perfectly walkable and
    every other check would call this town fine."""
    b = base()
    net = lane(b, [c for c in LANE if c[0] < 13 or c[0] > 15])
    for y in range(GROUND + 1, GROUND + 5):        # a wall across the whole world
        for z in range(SZ):
            b[(14, y, z)] = "stone"
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     PLOT, network=net), only={"E007"})
    assert "E007" in codes(r), "a lane cut in two was not reported"
    return r.findings[0].message


@case
def t_e007_lane_built_over():
    b = base()
    net = lane(b, LANE)
    for x in range(8, 12):                          # somebody's wall, on the lane
        for y in range(GROUND + 1, GROUND + 4):
            b[(x, y, 15)] = "stone_bricks"
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     PLOT, network=net), only={"E007"})
    assert "E007" in codes(r), "lane cells built over were not reported"
    return r.findings[0].message


@case
def t_e008_threshold_obstructed():
    """The one piece of shared state, and the check that gives it teeth."""
    b = base()
    net = lane(b, LANE, thresholds=[("hut", 8, 15, "north")])
    clean = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                         PLOT, network=net), only={"E008"})
    assert not clean.findings, f"clean threshold reported: {clean.findings[0].message}"
    for y in range(GROUND + 1, GROUND + 4):
        b[(8, y, 15)] = "stone_bricks"
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     PLOT, network=net), only={"E008"})
    assert "E008" in codes(r), "a threshold built over was not reported"
    return r.findings[0].message


@case
def t_e009_pass_severs_the_network():
    """Snapshot, build, diff. A pass that walls off ground it did not build on has
    disconnected something that worked before it ran."""
    b = base()
    net = lane(b, LANE)
    before_ctx = lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                    PLOT, network=net)
    snap = lint.snapshot(before_ctx.nav, net)
    assert len(snap["reachable"]) > 100, "nothing was reachable to begin with"

    def wall_off(x0, z0, x1, z1):
        after = dict(b)
        for y in range(GROUND + 1, GROUND + 5):
            for x in range(x0, x1 + 1):
                after[(x, y, z0)] = "stone"
                after[(x, y, z1)] = "stone"
            for z in range(z0, z1 + 1):
                after[(x0, y, z)] = "stone"
                after[(x1, y, z)] = "stone"
        return lint.lint(lint.Context.build(
            Volume.from_blocks(after, X0, Y0, Z0, SX, SY, SZ),
            PLOT, network=net, before=snap), only={"E009"})

    # ground inside somebody's plot: the town lost something it needs
    severe = wall_off(4, 4, 11, 11)
    assert "E009" in codes(severe), f"a severed plot was not an error: {codes(severe)}"
    # open hillside outside every plot: reported, but not the same failure
    benign = wall_off(18, 20, 24, 26)
    assert "W006" in codes(benign), f"a severed pocket was not reported: {codes(benign)}"
    assert "E009" not in codes(benign), "open ground outside every plot raised an error"

    clean = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                         PLOT, network=net, before=snap), only={"E009"})
    assert not clean.findings, "an unchanged world reported a regression"
    return f"{severe.findings[0].message[:60]}... | benign: {benign.findings[0].message[:50]}"


@case
def t_w005_planned_site_unserved():
    b = base()
    net = lane(b, LANE)
    # the far site sits on a shelf five blocks up: reachable by scrambling from open
    # air, not reachable on foot from the network, which is the whole distinction
    for x in range(0, 10):
        for z in range(0, 10):
            for y in range(GROUND + 1, GROUND + 6):
                b[(x, y, z)] = "stone"
    sites = [{"id": "on_it", "x0": 6, "z0": 16, "x1": 10, "z1": 20},
             {"id": "miles_away", "x0": 3, "z0": 3, "x1": 6, "z1": 6}]
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     PLOT, network=net, sites=sites), only={"W005"})
    got = {f.detail["id"] for f in r.findings}
    assert got == {"miles_away"}, f"expected only the far site, got {got}"
    return r.findings[0].message


@case
def t_e010_floating_mass():
    """A roof one block too high: the walls stop, a band of air, then the roof."""
    b = hut(base())
    for x in range(14, 22):                        # a roof slab hanging over nothing
        for z in range(14, 22):
            b[(x, GROUND + 6, z)] = "oak_planks"
    r = lint.lint(ctx_for(b), only={"E010"})
    assert not r.findings, "reported a mass off the plots -- that is landscape"
    plots = PLOT + [{"x0": 13, "z0": 13, "x1": 23, "z1": 23, "label": "shed"}]
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     plots, region=(0, 0, SX - 1, SZ - 1)),
                  only={"E010"})
    assert "E010" in codes(r), "a floating roof on a plot was not reported"
    clean = lint.lint(lint.Context.build(Volume.from_blocks(hut(base()), X0, Y0, Z0,
                                                            SX, SY, SZ), plots,
                                         region=(0, 0, SX - 1, SZ - 1)), only={"E010"})
    assert not clean.findings, f"a normal hut reported: {clean.findings[0].message}"
    return r.findings[0].message


@case
def t_w007_void_band_needs_a_band():
    """One block of air between wall top and roof, right across the footprint."""
    b = base()
    for x in range(4, 14):
        for z in range(4, 14):
            for y in range(GROUND + 1, GROUND + 4):
                if x in (4, 13) or z in (4, 13):
                    b[(x, y, z)] = "stone_bricks"
            b[(x, GROUND + 5, z)] = "stone_bricks"      # roof, with y=GROUND+4 empty
    plots = [{"x0": 3, "z0": 3, "x1": 14, "z1": 14, "label": "hut"}]
    r = lint.lint(lint.Context.build(Volume.from_blocks(b, X0, Y0, Z0, SX, SY, SZ),
                                     plots), only={"W007"})
    assert "W007" in codes(r), "a roof standing off its walls was not reported"
    ok = lint.lint(ctx_for(hut(base())), only={"W007"})
    assert not ok.findings, "a hut whose roof sits on its walls was reported"
    return r.findings[0].message


@case
def t_families_split_town_from_build():
    """A castle is not a town. The build family has to be runnable on its own."""
    ctx = ctx_for(hut(base()))
    b = lint.lint(ctx, family=lint.BUILD)
    p = lint.lint(ctx, family=lint.PLACE)
    fams = {c.code: c.family for c in lint.CHECKS}
    assert all(fams[f.code] == lint.BUILD for f in b.findings)
    assert all(fams[f.code] == lint.PLACE for f in p.findings)
    assert {c.family for c in lint.CHECKS} == {lint.BUILD, lint.PLACE}
    return (f"{sum(1 for c in lint.CHECKS if c.family == lint.BUILD)} build checks, "
            f"{sum(1 for c in lint.CHECKS if c.family == lint.PLACE)} place")


@case
def t_a_cave_under_a_plot_is_not_a_room():
    """`plot_at` is two-dimensional, so anything hollow under a building was being
    charged to it. What tells a cellar from a cave is what the walls are made of."""
    b = hut(base())
    for x in range(5, 10):                  # a cavity in the rock under the hut
        for z in range(5, 10):
            for y in range(GROUND - 6, GROUND - 3):
                b.pop((x, y, z), None)
    ctx = ctx_for(b)
    deep = [r for r in ctx.rooms if r["bbox"][1] < GROUND - 2]
    assert deep, "the cavity was not found at all"
    assert all(r["made"] < lint.Context.MADE for r in deep), \
        f"a cave read as built: {[r['made'] for r in deep]}"
    assert all(r["plot"] is None for r in deep), "a cave was charged to a plot"
    inside = [r for r in ctx.rooms if r["bbox"][1] >= GROUND]
    assert inside and all(r["made"] >= lint.Context.MADE for r in inside), \
        f"the hut's own room read as a cave: {[r['made'] for r in inside]}"
    return (f"cave made={deep[0]['made']}, room made={inside[0]['made']}, "
            f"threshold {lint.Context.MADE}")


@case
def t_severity_ordering_and_summary():
    r = lint.lint(ctx_for(hut(base(), door=False, lit=False)))
    sev = [f.severity for f in r.findings]
    assert sev == sorted(sev, key=lambda s: lint._ORDER[s]), "not sorted by severity"
    assert not r.ok, "report with errors claims ok"
    text = r.summary()
    assert "errors" in text and len(text.splitlines()) < 40, "summary is not compact"
    return f"{len(r.findings)} findings summarise in {len(text.splitlines())} lines"


@case
def t_bss3_b3_a_door_that_was_already_there_is_not_this_builds_door():
    """E002 judges the doors this build hung."""
    before = base()
    # A hut nobody built, with a door in it, standing on the ground the run was planned
    # against.
    for x in range(20, 25):
        for z in range(20, 25):
            for y in range(GROUND + 1, GROUND + 5):
                if x in (20, 24) or z in (20, 24):
                    before[(x, y, z)] = "cobblestone"
    before[(22, GROUND + 1, 20)] = "jungle_door[facing=north,half=lower]"
    before[(22, GROUND + 2, 20)] = "jungle_door[facing=north,half=upper]"
    world = Volume.from_blocks(before, X0, Y0, Z0, SX, SY, SZ)
    # ...and what this build did: its own hut, and a wall through the village's door.
    mine = hut(dict(), door=True)
    for y in range(GROUND + 1, GROUND + 5):
        mine[(22, y, 19)] = "stone_bricks"
    got = Volume.from_blocks({**before, **mine}, X0, Y0, Z0, SX, SY, SZ)
    blind = lint.Context.build(got, PLOT)
    seeing = lint.Context.build(got, PLOT, base=world)
    assert len(blind.doors) == len(seeing.doors) + 1, \
        f"{len(blind.doors)} doors without the before, {len(seeing.doors)} with it"
    assert (22, GROUND + 1, 20) in blind.doors, "the village door was never counted"
    assert (22, GROUND + 1, 20) not in seeing.doors, \
        "a door that was already in the ground is still being judged"
    assert "E002" in codes(lint.lint(blind, only={"E002"})), \
        "the sealed village door was not an E002 before the correction"
    assert "E002" not in codes(lint.lint(seeing, only={"E002"})), \
        "the correction left an E002 on a door nobody here built"
    return (f"{len(blind.doors)} doors in the region, {len(seeing.doors)} of them this "
            f"build's; the village's sealed door is no longer an error")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:34s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:34s} {e}")
        except Exception as e:
            bad += 1
            print(f"ERR  {name:34s} {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} lint cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
