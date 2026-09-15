"""The library owns the stair, the instruments are seeded from the lane.

    $PY scripts/test_flight.py

A1. **flight() makes a loft walkable.** A two-storey hut with a door: before the call
the loft is 0% walkable from the doorway, after it is 100%, and a flight aimed into a
wall refuses and leaves the hut byte-identical. A2. from the lane, 16 of 16. A3. Scoped
to the two plots it reserved, the report names its own rooms. A4. **The finishing pass
keeps off a recorded way in.** A dress-ground pass over the columns an approach() laid
leaves them byte-identical.

the pre-build cache, the network and wave5's program. They are the cases the spec's
acceptance is written against, so a missing fixture is a failure and says so loudly
rather than skipping.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from ethoslm import finish, lint, observe, offline, settlement, stages  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.circulate import Network, Threshold  # noqa: E402
from ethoslm.frontage import Frontage  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R9 = os.path.join(ROOT, "out", "site_d")

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


# ------------------------------------------------------------------- fixtures
SX = SZ = 40
Y0, GROUND = 50, 60


def world(extra: dict | None = None) -> Volume:
    b = {}
    for x in range(SX):
        for z in range(SZ):
            for y in range(Y0, GROUND + 1):
                b[(x, y, z)] = "stone"
    b.update(extra or {})
    return Volume.from_blocks(b, 0, Y0, 0, SX, 64, SZ)


def builder(vol: Volume, net: Network | None = None) -> Builder:
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net else None
    return b


#: A two-storey hut, built through the Builder so every block in it is the program's own
#: -- which is what makes the refusal case mean anything. Ground floor at GROUND, loft
#: floor three blocks up, a door in the south wall, and *no way up*.
X0, Z0, X1, Z1 = 12, 12, 22, 22
FLOOR = GROUND
LOFT = GROUND + 4
DOOR = (17, FLOOR + 1, Z1)


def two_storey_hut(b: Builder) -> None:
    for x in range(X0, X1 + 1):
        for z in range(Z0, Z1 + 1):
            b.place_block(x, FLOOR, z, "stone_bricks")             # ground floor
            edge = x in (X0, X1) or z in (Z0, Z1)
            for y in range(FLOOR + 1, LOFT + 5):
                if edge:
                    b.place_block(x, y, z, "stone_bricks")         # the shell
                else:
                    b.place_block(x, y, z, "air")                  # hollow it out
    for x in range(X0 + 1, X1):
        for z in range(Z0 + 1, Z1):
            b.place_block(x, LOFT, z, "oak_planks")                # the loft floor
    b.place_block(DOOR[0], DOOR[1], DOOR[2], "oak_door[facing=south,half=lower]")
    b.place_block(DOOR[0], DOOR[1] + 1, DOOR[2], "oak_door[facing=south,half=upper]")
    for y in range(LOFT + 1, LOFT + 5):                            # roof over the loft
        pass
    for x in range(X0, X1 + 1):
        for z in range(Z0, Z1 + 1):
            b.place_block(x, LOFT + 4, z, "stone_bricks")


def rooms_by_level(res: dict) -> dict:
    """{floor level: fraction walkable} for the hut's two storeys.

        Only meaningful while they *are* two: once a flight cuts the stairwell the storeys
        are one connected component and `observe.rooms` rightly reports one room. That is
        why the acceptance is stated on totals -- see `walked`.
        
    """
    return {r["bbox"][1] - 1: r["fraction"] for r in res["rooms"]}


def walked(res: dict) -> tuple[int, int]:
    """(floor cells you can walk to, floor cells there are) over every room."""
    return (sum(r["walkable"] for r in res["rooms"]),
            sum(r["cells"] for r in res["rooms"]))


# ------------------------------------------------- A1. the library owns the stair
@case
def t_a1_a_loft_with_no_stair_cannot_be_walked_into():
    """The fixture has to exhibit the defect or the next case proves nothing."""
    vol = world()
    b = builder(vol)
    two_storey_hut(b)
    res = b.check_walkable(x0=X0, z0=Z0, x1=X1, z1=Z1)
    by = rooms_by_level(res)
    assert LOFT in by, f"no room found at the loft floor {LOFT}: {res['rooms']}"
    assert by[LOFT] == 0.0, f"the loft is already {by[LOFT]:.0%} walkable"
    assert by[FLOOR] > 0.9, f"the ground floor is only {by[FLOOR]:.0%} walkable"
    got, all_ = walked(res)
    return f"loft {by[LOFT]:.0%}, ground floor {by[FLOOR]:.0%}, {got}/{all_} overall"


@case
def t_a1_flight_makes_the_loft_walkable():
    vol = world()
    b = builder(vol)
    two_storey_hut(b)
    was_got, was_all = walked(b.check_walkable(x0=X0, z0=Z0, x1=X1, z1=Z1))
    # up the east side, climbing north from just inside the south wall: four treads and
    # a landing, cutting its own hole through the loft floor on the way
    res = b.flight("hut", 20, Z1 - 2, FLOOR, LOFT, "north", mat="oak")
    assert res["ok"], f"flight refused or did not connect: {res['reason']}"
    assert res["cells"] == (LOFT - FLOOR) + 1, res
    assert res["removed"] > 0, "no stairwell was cut through the loft floor"
    b.resolve_steps()
    after = b.check_walkable(x0=X0, z0=Z0, x1=X1, z1=Z1)
    got, all_ = walked(after)
    # One room now, not two: the stairwell joins the storeys, which is the whole point.
    assert len(after["rooms"]) == 1, [r["bbox"] for r in after["rooms"]]
    assert got == all_, f"only {got} of {all_} floor cells are walkable after the flight"
    assert got > was_got, f"{was_got} walkable before, {got} after"
    return (f"{res['cells']} cells, {res['removed']} cleared, "
            f"{was_got}/{was_all} walkable -> {got}/{all_}")


@case
def t_a1_a_flight_into_a_wall_refuses_and_places_nothing():
    """A refusal is the library saying what it cannot do. It may cut a floor -- that is
    what a stairwell is -- and it may not cut a wall, because a stair through a wall is
    a hole in the building."""
    vol = world()
    b = builder(vol)
    two_storey_hut(b)
    before = dict(b._pending)
    # first tread one block in from the west wall, climbing west: straight into it
    res = b.flight("hut", X0 + 1, 17, FLOOR, LOFT, "west", mat="oak")
    assert not res["ok"], "flight claimed to climb through the wall"
    assert res["cells"] == 0 and res["removed"] == 0, res
    assert "stone_bricks" in res["reason"] and "not floor or air" in res["reason"], \
        res["reason"]
    assert b._pending == before, \
        f"a refused flight changed {len(set(b._pending) ^ set(before))} cells"
    assert not b._step_queue(), "a refused flight left treads on the queue"
    return res["reason"][res["reason"].index("replace"):][:52]


@case
def t_a1_a_flight_is_walked_before_it_is_reported():
    """`ok` is answered against the world the call has just changed, not asserted from
        its own arithmetic.

        Here the treads and the landing are all fine -- every one of them lands in the
        hut's own air or its own loft floor, so nothing is refused -- and the flight is
        still useless, because it starts in the west wall and nobody can stand at the foot
        of it. A primitive that reported its own arithmetic would say `ok`.
    """
    vol = world()
    b = builder(vol)
    two_storey_hut(b)
    before = dict(b._pending)
    res = b.flight("hut", X0 + 1, 17, FLOOR, LOFT, "east", mat="oak")
    assert not res["ok"], f"flight reported ok starting inside the wall: {res['reason']}"
    assert "stand at the foot" in res["reason"], res["reason"]
    assert res["cells"] == 0 and res["removed"] == 0, res
    assert b._pending == before, \
        f"a refused flight left {len(set(b._pending) ^ set(before))} cells standing"
    assert not b._step_queue(), "a refused flight left treads on the queue"
    assert not b.paths, f"a refused flight wrote itself down as a way up: {b.paths}"
    return "laid nothing: " + res["reason"][res["reason"].index("nobody"):][:44]


# --------------------------------------------- A2. seeded from the lane, not the edge
def _r9(*parts):
    p = offline.fixture_path("site_d", *parts)
    if not os.path.exists(p):
        raise AssertionError(
            f"COULD NOT RUN: {os.path.relpath(p, ROOT)} is missing -- this is a case "
            f"the spec's acceptance is written against, so it fails rather than skips")
    return p


@case
def t_a2_the_pre_build_cache_is_reachable_from_the_lane_and_not_from_the_edge():
    vol = offline.load_volume(_r9("world.npz"))
    net = Network.load(_r9("network.json"))
    nav = observe.Nav(vol)
    lanes = lint.lane_stances(nav, net)
    assert len(lanes) == len(net.cells), \
        f"{len(lanes)} of {len(net.cells)} lane cells can be stood on"

    def doorsteps(reach):
        n = 0
        for t in net.thresholds:
            dx, dy, dz = t.door
            s = nav.stance_near(dx, dz, dy, tol=2)
            n += s is not None and (dx, dz, s) in reach
        return n

    peri = set(nav.flood(nav.perimeter_seeds(inset=2, step=3), max_jumps=0))
    lane = set(nav.flood(lanes, max_jumps=0))
    was, now = doorsteps(peri), doorsteps(lane)
    assert was == 0, f"the fixture no longer exhibits the defect: {was}/16 from the edge"
    assert now == len(net.thresholds), \
        f"only {now} of {len(net.thresholds)} doorsteps are walk-reachable from the lane"
    assert sum(1 for c in lanes if c in peri) == 0, "the lane is reachable from the edge"
    return f"doorsteps from the edge {was}/16, from the lane {now}/16"


@case
def t_a2_the_context_takes_its_outdoor_seeds_from_the_lane():
    """...and E002, W001 and `diagnose_entry` read that one flood, so none of them can
    ask the perimeter's question on a settlement again."""
    vol = offline.load_volume(_r9("world.npz"))
    net = Network.load(_r9("network.json"))
    site = json.load(open(_r9("site.json")))
    X, Z, S = site["origin"][0], site["origin"][1], site["size"]
    ctx = lint.Context.build(vol, [], network=net, region=(X, Z, X + S - 1, Z + S - 1))
    assert ctx.outdoor_seeds == "lane", ctx.outdoor_seeds
    bad = 0
    for t in net.thresholds:
        dx, dy, dz = t.door
        s = ctx.nav.stance_near(dx, dz, dy, tol=2)
        bad += not (s is not None and (dx, dz, s) in ctx.from_outdoors)
    assert bad == 0, f"{bad} of {len(net.thresholds)} reserved doorsteps are still " \
                     f"unreachable in ctx.from_outdoors"
    # ...and with no network there is nothing to seed from but the edge
    bare = lint.Context.build(vol, [], region=(X, Z, X + S - 1, Z + S - 1))
    assert bare.outdoor_seeds == "perimeter", bare.outdoor_seeds
    return f"lane seeds, 0 of {len(net.thresholds)} doorsteps a false E002"


# ------------------------------------------------- A3. scoped to the plot, not the box
@case
def t_a3_check_walkable_scopes_to_the_plot_not_the_bounding_box():
    """Two structures sixty blocks apart. The bounding box of everything the program placed
        is full of the hillside between them, `observe.rooms` finds the caves in it, and the
        builder was told fifteen rooms could not be walked into at all. It said so in its
        reply and discounted the report; it then shipped the tower that carries three of the
        town's four errors.
        
    """
    from ethoslm import pipeline
    rnd = pipeline.Round.load(os.path.join(ROOT, "rounds", "site_d.json"))
    be = pipeline.OfflineBackend(rnd)
    prog = _r9("wave5.py")
    b = be.execute(prog, be.volume)
    labels = ["ridge_lookout", "threshing_barn"]
    rects = [p for p in json.load(open(_r9("plots.json")))
             if p["label"] in labels]
    assert len(rects) == 2, [p["label"] for p in rects]

    was_shut, was_rooms = _round9_shut(b)
    scoped = {lab: b.check_walkable(lab) for lab in labels}
    assert all(r.get("scope", "").endswith(lab) for lab, r in scoped.items()), scoped

    mine = sum(len(r["rooms"]) for r in scoped.values())
    shut = sum(1 for r in scoped.values() for rm in r["rooms"] if rm["walkable"] == 0)
    # observe.STAIR_CLEAR) and two more of those hillside caves stopped being enterable.
    # The defect the case exists to show -- that the bounding box hands a builder the
    # hillside -- is unchanged and is what is asserted.
    assert was_shut >= 15, \
        f"the fixture no longer exhibits the defect: the scoping rule reports " \
        f"{was_shut} rooms nobody can walk into, and this builder was told fifteen"
    # Corrected to three (observe.STAIR_ CLEAR) the stair no longer carries anybody and
    # the shaft is its own shut room. Every one of them is still this builder's own,
    # which is what the case is for.
    assert mine == 3, f"the two plots should name three rooms, not {mine}"
    assert shut == 3, f"{shut} of the 3 rooms on its own plots are shut"
    # every room the scoped calls name is inside the plot it was asked about
    for lab, r in scoped.items():
        p = next(q for q in rects if q["label"] == lab)
        a, c = min(p["x0"], p["x1"]), max(p["x0"], p["x1"])
        d, e = min(p["z0"], p["z1"]), max(p["z0"], p["z1"])
        for rm in r["rooms"]:
            bb = rm["bbox"]
            cx, cz = (bb[0] + bb[3]) / 2, (bb[2] + bb[5]) / 2
            assert a - 4 <= cx <= c + 4 and d - 4 <= cz <= e + 4, \
                f"{lab}: a room centred at ({cx},{cz}) is not on its plot {p}"
    return (f"unscoped: {was_shut} of {was_rooms} rooms shut; scoped: "
            + ", ".join(f"{lab} {len(r['rooms'])}"
                        for lab, r in sorted(scoped.items()))
            + f", {shut} shut and all of them its own")


def _round9_shut(b: Builder) -> tuple[int, int]:
    """(rooms nobody can walk into, rooms)."""
    pending = b._pending_view()
    xs = [p[0] for p in pending]
    zs = [p[2] for p in pending]
    a, d, c, e = min(xs), min(zs), max(xs), max(zs)
    m = 4
    sub = b._world_volume().sub(a - m, d - m, c - a + 1 + 2 * m, e - d + 1 + 2 * m)
    sub.overlay(pending)
    nav = observe.Nav(sub)
    sky_open, _ = observe.shelter(sub)
    rooms = [r for r in observe.rooms(nav, sky_open, region=(a, d, c + 1, e + 1))
             if r.get("enclosure", 1.0) >= 0.85]
    seeds = []
    for (x, y, z) in sub.find(lambda s: s.split("[")[0].endswith("_door")
                              or s.split("[")[0].endswith("_fence_gate")):
        if observe.parse_props(sub.state(x, y, z)).get("half") == "upper":
            continue
        if not (a - 1 <= x <= c + 1 and d - 1 <= z <= e + 1):
            continue
        s = nav.stance_near(x, z, y, tol=2)
        if s is not None:
            seeds.append((x, z, s))
    reach = set(nav.flood(seeds, max_jumps=0)) if seeds else set()
    shut = sum(1 for r in rooms if not (set(map(tuple, r["stances"])) & reach))
    return shut, len(rooms)


@case
def t_a3_no_label_reports_each_plot_of_the_wave():
    """A wave of three structures is three answers, not one box round all of them."""
    class Reg:
        plots = [{"x0": X0, "z0": Z0, "x1": X1, "z1": Z1, "label": "hut"},
                 {"x0": 26, "z0": 26, "x1": 34, "z1": 34, "label": "shed"}]
        claimed_this_pass = plots

        def plots_list(self):
            return [dict(p) for p in self.plots]

    vol = world()
    b = builder(vol)
    b.registry = Reg()
    two_storey_hut(b)
    for x in range(26, 35):                       # a second, single-storey structure
        for z in range(26, 35):
            b.place_block(x, GROUND, z, "stone_bricks")
            edge = x in (26, 34) or z in (26, 34)
            for y in range(GROUND + 1, GROUND + 4):
                b.place_block(x, y, z, "stone_bricks" if edge else "air")
            b.place_block(x, GROUND + 4, z, "stone_bricks")
    b.place_block(30, GROUND + 1, 34, "oak_door[facing=south,half=lower]")
    b.place_block(30, GROUND + 2, 34, "oak_door[facing=south,half=upper]")

    res = b.check_walkable()
    assert "plots" in res and len(res["plots"]) == 2, res.get("reason")
    assert {p["plot"] for p in res["plots"]} == {"hut", "shed"}, res["plots"]
    assert all("plot" in r for r in res["rooms"]), res["rooms"][:2]
    one = b.check_walkable("shed")
    assert "plots" not in one and one["scope"].endswith("shed"), one
    return f"{len(res['plots'])} plots reported separately, {len(res['rooms'])} rooms"


# ---------------------------------------- A4. the finishing pass keeps off the way in
@case
def t_a4_a_recorded_way_in_is_left_byte_identical():
    """The door it fired on is served by a 48-column approach() path and the pass dressed
    the ground along it. Two columns round the doorstep was never the right radius.
    """
    # a flat shelf with a bank falling away to the south, and a lane along the top
    ground = np.full((40, 40), 64, int)
    ground[:, 13:] = 58
    built = {(x, 10): 64 for x in range(5, 35)}          # the lane
    path = [(20, z) for z in range(11, 21)]              # the way in, down the bank
    rows = [{"label": "hut", "kind": "approach",
             "cells": [[x, z] for (x, z) in path]}]
    cols = settlement.path_columns(rows)
    assert cols == set(path)

    loose = finish.plan_finish(built, ground, 0, 0)
    on_path = sorted(set(loose) & cols)
    assert on_path, ("the fixture no longer exhibits the defect: the seam pass moves "
                     "nothing on the path even without being told")

    keep_out = [{"x0": x, "z0": z, "x1": x, "z1": z} for (x, z) in sorted(cols)]
    guarded = finish.plan_finish(built, ground, 0, 0, keep_out=keep_out)
    assert not (set(guarded) & cols), \
        f"{len(set(guarded) & cols)} path columns are still planned to move"

    # ...and applying that plan writes nothing in any of them
    vol = Volume.from_blocks({(x, y, z): "stone"
                              for x in range(40) for z in range(40)
                              for y in range(50, 65)}, 0, 50, 0, 40, 20, 40)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    finish.apply(b, guarded)
    touched = sorted({(p[0], p[2]) for p in b._pending} & cols)
    assert not touched, f"the finishing pass wrote in {len(touched)} path columns"
    return (f"{len(on_path)} of {len(cols)} path columns moved unguarded, 0 guarded, "
            f"0 written")


@case
def t_a4_approach_and_flight_write_down_what_they_laid():
    """The pass cannot keep off a path it is not told about: a dressed approach and an
        undressed one are indistinguishable in the blocks.
    """
    extra = {}
    for x in range(14, 23):
        for z in range(14, 23):
            for y in range(GROUND + 1, GROUND + 4):
                extra[(x, y, z)] = "stone_bricks"
    vol = world(extra)
    b = builder(vol)
    for x in range(14, 23):
        for z in range(14, 23):
            edge = x in (14, 22) or z in (14, 22)
            for y in range(GROUND + 4, GROUND + 11):
                b.place_block(x, y, z, "stone_bricks" if edge else "air")
    for x in range(15, 22):                                  # the upper floor
        for z in range(15, 22):
            b.place_block(x, GROUND + 7, z, "oak_planks")
    door = (17, GROUND + 4, 22)
    b.place_block(*door, "oak_door[facing=south,half=lower]")
    b.place_block(door[0], door[1] + 1, door[2], "oak_door[facing=south,half=upper]")
    net = Network({(17, z): {"y": GROUND, "rank": 0, "face": None}
                   for z in range(26, 32)},
                  [Threshold("hut", 17, GROUND, 23, "north", tuple(door))])
    b.frontage = Frontage(vol, net)
    res = b.approach("hut", *door)
    assert res["ok"] and res["cells"] > 0, res["reason"]
    assert len(b.paths) == 1 and b.paths[0]["kind"] == "approach", b.paths
    assert len(b.paths[0]["cells"]) == res["cells"], \
        f"{len(b.paths[0]['cells'])} columns recorded for {res['cells']} laid"
    assert b.paths[0]["label"] == "hut"

    f = b.flight("hut", 17, 20, GROUND + 3, GROUND + 7, "north", mat="cobblestone")
    assert f["ok"], f["reason"]
    kinds = [p["kind"] for p in b.paths]
    assert kinds == ["approach", "flight"], kinds
    assert len(b.paths[1]["cells"]) == f["cells"], (b.paths[1], f)
    return (f"approach {res['cells']} columns and flight {f['cells']} cells recorded "
            f"against 'hut'")


@case
def r_bss3_b2_flight_carries_its_own_stair():
    """A flight lays what holds its treads up."""
    v = world()
    b = builder(v)
    # A room with a floor and nothing else in it, so a flight up the middle has nothing
    # beside it to be held by.
    b.place_cuboid(X0, FLOOR, Z0, X1, FLOOR, Z1, "stone_bricks")
    b.place_cuboid(X0 + 1, FLOOR + 1, Z0 + 1, X1 - 1, FLOOR + 6, Z1 - 1, "air")
    f = b.flight("mid", 17, 15, FLOOR, FLOOR + 4, "south", mat="cobblestone")
    assert f["ok"], f["reason"]
    got = stages.apply_pending(v, b._pending)
    under = [(17, FLOOR + y, 15 + y - 1) for y in range(1, 5)]
    empty = [c for c in under if got.state(*c).split("[")[0] in ("air", "cave_air")]
    assert not empty, f"nothing holds these treads up: {empty}"
    float_pieces = observe.unsupported(got)
    treads = [p for p in float_pieces
              if p["bbox"][0] <= 17 <= p["bbox"][3] and "stairs" in p["example"]]
    assert not treads, f"a tread is still reported as floating: {treads}"
    return f"{f['cells']} cells laid, {len(under)} treads carried, nothing floating"


@case
def r_bss3_b2_a_type_may_not_build_on_its_own_doorstep():
    """The way in is the library's.

        `site()` levels the doorstep, lays the path to it and measures that a person can
        walk in; a type that then puts a block in it seals the building. Four halls in a
        city did exactly that -- a footing course over ground their own flood could not
        reach, doorstep included -- and it read as four unreachable doors.
        
    """
    from ethoslm.buildlib import TypeBuilder
    v = world()
    b = builder(v)
    part = {"label": "hut", "x0": X0, "z0": Z0, "x1": X1, "z1": Z1,
            "floor_y": FLOOR, "door": [17, Z1 + 1],
            "way": [[17, FLOOR + 1, Z1 + 2], [17, FLOOR + 2, Z1 + 2]]}
    tb = TypeBuilder(b, part)
    tb.place_block(17, FLOOR + 1, Z1 + 1, "stone_bricks")       # the doorstep
    tb.place_cuboid(15, FLOOR + 1, Z1 + 2, 19, FLOOR + 1, Z1 + 2, "cobblestone_wall")
    got = stages.apply_pending(v, b._pending)
    for cell in ((17, FLOOR + 1, Z1 + 1), (17, FLOOR + 1, Z1 + 2)):
        assert got.state(*cell).split("[")[0] in ("air", "cave_air"), \
            f"the way in is blocked at {cell}: {got.state(*cell)}"
    assert got.state(15, FLOOR + 1, Z1 + 2).split("[")[0] == "cobblestone_wall", \
        "the rest of the write was refused too, and it should not have been"
    assert len(tb.refused) == 2, [r["reason"] for r in tb.refused]
    assert all("doorstep" in r["reason"] for r in tb.refused), tb.refused
    return f"2 cells held open, {len(tb.refused)} refusals by name, the rest laid"


@case
def r_bss3_b2_the_ground_pass_leaves_no_column_lower():
    """`clear_ground_cover` takes the turf, and the finishing pass puts the level back."""
    turf = {}
    for x in range(SX):
        for z in range(SZ):
            turf[(x, GROUND, z)] = "grass_block"
    turf[(17, GROUND + 1, 30)] = "short_grass"
    v = world(turf)
    b = builder(v)
    before = set(b._pending)
    # As `site()` clears it: the register is the library's own ground pass and nothing
    # else's, so a stored program that clears its own cover still places what it placed.
    b._library_ground = True
    b.clear_ground_cover(14, 28, 20, 32)
    b._library_ground = False
    assert b.stripped, "nothing was recorded as stripped"
    b.clear_ground_cover(14, 34, 20, 36)
    assert not [c for c in b.stripped if c[1] > 33], \
        "a program's own clearing was recorded as the library's"
    b.place_block(17, GROUND + 1, 28, "stone_bricks")     # something of ours, one column
    b._dress_worked(before, (14, 28, 20, 32))
    got = stages.apply_pending(v, b._pending)
    assert got.state(17, GROUND, 30).split("[")[0] == "grass_block", \
        f"the ground is still a block down: {got.state(17, GROUND, 30)}"
    assert got.state(17, GROUND + 1, 30).split("[")[0] in ("air", "cave_air"), \
        "the cover was not taken"
    assert got.state(17, GROUND + 1, 28).split("[")[0] == "stone_bricks", \
        "a column we built in was restored over"
    return ("turf back at the level it was taken from, the cover still gone, and a "
            "program's own clearing not recorded at all")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            note = fn()
            print(f"ok   {name:56s} {note}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:56s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:56s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} flight cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
