"""The shell moves into the library, and the walk model grows a head.

    $PY scripts/test_building.py

  A1. **`building()` lays a whole shell.** One call on flat ground and one on a
      six-block slope: no lint error, every floor walkable from the door under the
      corrected model, the chimney attached with no hole in the roof beside it, and the
      sloped one's door walk-reachable. Plus the two refusals the spec names, and the
      third the walkability bar demands.
  A2. **Headroom.** A flight under a ceiling two cells above its treads is unwalkable
      to `Nav`, and `flight()` clears the third cell so that the same flight built
      through the library is walkable.
  A3. **`dais()`** is walkable from the floor beside it; a bare raised floor is not.
  A4. **Fittings refuse.** A bed into a lantern refuses; a bed into a wall refuses;
      both place nothing.
  A5. **Six words**, each placed without error in six material families.
  A6. **Lanes lit**, and no lane cell loses its stance to a lamp post.

E005 -- unjoined panes -- is excluded from the lint assertions and only from those:
connection states are computed by the server's block updates, which `Builder.flush()`
does and a dry run cannot. It is the one check offline is structurally unable to make
(see `offline.py`), and every window `building()` lays is a pane.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

import json  # noqa: E402

from ethoslm import circulate, lint, observe, offline, registry, stages  # noqa: E402
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.slopefixture import make  # noqa: E402
from ethoslm.circulate import Network, Threshold  # noqa: E402
from ethoslm.frontage import Frontage  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402
from ethoslm.prims import MATERIALS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


# ------------------------------------------------------------------- fixtures
SX = SZ = 56
Y0, GROUND = 40, 60

#: The palette every case builds in: five roles, five families, one voice.
MAT = {"wall": "cobblestone", "roof": "dark_oak", "footing": "mossy_cobblestone",
       "frame": "spruce", "floor": "spruce", "trim": "stripped_spruce_log"}

#: The footprint. Odd on both sides so a door has a true centre block.
BOX = (14, 14, 26, 24)

#: ...and the one A3's courtyard needs. 21x21 of plot, because `site()` insets four
#: before the type sees it: the pad is 17x17, which holds a 5x5 yard with six-deep
#: ranges, and a range six deep has a room four deep in it.
COURT_BOX = (12, 10, 32, 30)
PLOT = {"x0": 10, "z0": 10, "x1": 34, "z1": 30, "label": "hut"}


def world(slope: int = 0, axis: str = "z") -> Volume:
    """Flat ground at y=60, or ground falling away `slope` blocks toward the south.

        The fall starts at z=12, two blocks north of the footprint, so a building on it is
        cut into the bank uphill and stands clear of it downhill -- which is the situation
        every door on a real site is in and the one that breaks a way in.

        `axis="x"` falls to the west instead, which leaves the *north* face level. That is
        the fixture a call with no lane needs: with no circulation pass to reserve a
        doorstep, `building()` puts its door in the north wall, and a door in the uphill
        wall of a bank is a door into the bank. A round never has that problem, because a
        round has lanes; a fixture without one has to put the fall where the door is not.
        
    """
    b = {}
    for x in range(SX):
        for z in range(SZ):
            t = x if axis == "x" else z
            g = GROUND - min(slope, max(0, (t - 12) // 2))
            for y in range(Y0, g + 1):
                b[(x, y, z)] = "stone"
    return Volume.from_blocks(b, 0, Y0, 0, SX, 80, SZ)


def builder(vol: Volume, net: Network | None = None) -> Builder:
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net else None
    return b


def errors(vol: Volume, pending: dict, plots=(PLOT,)) -> list:
    """This build's lint errors, over the world as it would stand."""
    v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(pending)
    ctx = lint.Context.build(v, plots=[dict(p) for p in plots],
                             region=(0, 0, SX - 1, SZ - 1))
    return [(f.code, f.message) for f in lint.lint(ctx).findings
            if f.code.startswith("E") and f.code != "E005"]


def door_reachable(vol: Volume, pending: dict, door) -> bool:
    """Can a person walk to this door from anywhere outdoors, over the finished world?"""
    v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(pending)
    nav = observe.Nav(v)
    s = nav.stance_near(door[0], door[2], door[1], tol=2)
    if s is None:
        return False
    return (door[0], door[2], s) in set(
        nav.flood(nav.perimeter_seeds(inset=1, step=2), max_jumps=0))


def walked(b: Builder, res: dict) -> tuple[int, int]:
    """(floor cells you can walk to, floor cells there are) over every room it made."""
    xs = [r[0] for r in res["rooms"]] + [r[3] for r in res["rooms"]]
    zs = [r[2] for r in res["rooms"]] + [r[4] for r in res["rooms"]]
    w = b.check_walkable(x0=min(xs) - 1, z0=min(zs) - 1,
                         x1=max(xs) + 1, z1=max(zs) + 1)
    return (sum(r["walkable"] for r in w["rooms"]),
            sum(r["cells"] for r in w["rooms"]))


def a_building(slope: int = 0, **kw):
    """One `building()` call on `slope`, resolved. Returns (volume, builder, result)."""
    vol = world(slope)
    b = builder(vol)
    res = b.building(kw.pop("label", None), *kw.pop("box", BOX),
                     kw.pop("storeys", 2), kw.pop("roof", "gable"), mat=MAT, **kw)
    if res["ok"]:
        b.resolve_steps()
    return vol, b, res


# --------------------------------------------- A1. one call lays the whole shell
@case
def t_a1_one_call_on_flat_ground_is_a_finished_building():
    vol, b, res = a_building(0, chimney=True, porch=True)
    assert res["ok"], res["reason"]
    assert res["floors"] == [GROUND, GROUND + 4], res["floors"]
    assert len(res["rooms"]) == 2, res["rooms"]
    got, all_ = walked(b, res)
    assert got == all_, f"only {got} of {all_} floor cells are walkable"
    errs = errors(vol, b._pending)
    assert not errs, errs
    assert b.check_attached()["ok"], b.check_attached()["reason"]
    assert res["stairs"] and all(s["ok"] for s in res["stairs"]), res["stairs"]
    assert res["approach"]["ok"], res["approach"]["reason"]
    return (f"{res['cells']} positions, {len(res['rooms'])} rooms, {got}/{all_} "
            f"walkable, ridge {res['ridge_y']}, 0 errors")


@case
def t_a1_the_same_call_on_a_six_block_slope_is_still_enterable():
    """The case the spec names, and the one that found three defects.

        On a bank the ground uphill is level with the wall head, so a building without its
        roof on is an open box a person walks into over the top; `approach()` correctly
        reported the door reachable and laid nothing, and the roof then shut the lid. The
        other two: an outshot's roof overhang drove solid blocks through the room above,
        and a lone tread on a descending path was demoted to a slab and left the door a
        two-block climb away.
        
    """
    vol, b, res = a_building(6, chimney=True, outshot={"side": "west", "depth": 3})
    assert res["ok"], res["reason"]
    got, all_ = walked(b, res)
    assert got == all_, f"only {got} of {all_} floor cells are walkable"
    errs = errors(vol, b._pending)
    assert not errs, errs
    assert door_reachable(vol, b._pending, res["door"]), \
        f"the door at {res['door']} cannot be walked to: {res['approach']['reason']}"
    assert b.check_attached()["ok"], b.check_attached()["reason"]
    return (f"door {res['door']} walkable, {got}/{all_} floor, "
            f"{res['approach']['cells']} columns of path, 0 errors")


@case
def t_a1_the_chimney_goes_through_the_roof_and_leaves_no_hole():
    """This one is masonry from the ground, in the wall line, capped two blocks past the
    ridge -- and the roof either side of it is untouched.
    """
    vol, b, res = a_building(0, chimney=True)
    c = res["chimney"]
    assert c and c["top"] > res["ridge_y"], c
    p = b._pending
    for y in range(res["floors"][0], c["top"]):
        assert (c["x"], y, c["z"]) in p, f"the stack has a gap at y={y}"
    assert "slab" in p[(c["x"], c["top"], c["z"])], p[(c["x"], c["top"], c["z"])]
    # ...and no hole beside it: every roof column next to the stack still carries the
    # roof it carried, at the height the roof puts it.
    holes = []
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        col = [y for (x, y, z) in p if (x, z) == (c["x"] + dx, c["z"] + dz)]
        if not col or max(col) < res["eave_y"]:
            holes.append((c["x"] + dx, c["z"] + dz))
    assert not holes, f"the roof is missing beside the stack at {holes}"
    assert b.check_attached()["ok"], "the chimney is attached to nothing"
    return (f"stack {c['x']},{c['z']} from y={c['from_y']} to {c['top']}, ridge "
            f"{res['ridge_y']}, no hole on any of four sides")


@case
def t_a1_a_footprint_off_the_plot_refuses_and_places_nothing():
    vol = world()
    b = builder(vol)

    class Reg:
        def plots_list(self):
            return [{"x0": 14, "z0": 14, "x1": 20, "z1": 24, "label": "hut"}]
        claimed_this_pass = ()

    b.registry = Reg()
    before = dict(b._pending)
    res = b.building("hut", *BOX, 2, "gable", mat=MAT)
    assert not res["ok"], "a footprint hanging off its own plot was built"
    assert res["cells"] == 0 and b._pending == before, \
        f"a refused building placed {len(set(b._pending) ^ set(before))} cells"
    assert "leaves the plot" in res["reason"], res["reason"]
    return res["reason"][:64]


@case
def t_a1_a_wing_over_the_door_refuses_and_places_nothing():
    vol = world()
    b = builder(vol)
    before = dict(b._pending)
    # With no threshold the door goes in the south wall; a wing along it takes the wall.
    res = b.building(None, *BOX, 2, "gable", mat=MAT, wing=(14, 24, 26, 30))
    assert not res["ok"], "a wing standing on the door wall was built"
    assert res["cells"] == 0 and b._pending == before
    assert "wing or outshot stands on it" in res["reason"], res["reason"]
    return res["reason"][:70]


@case
def t_a1_a_footprint_too_small_for_its_stair_refuses():
    """Not one of the two refusals the spec names, and it is here for the bar the spec
    registers. It is decided from the rectangle, so nothing is laid.
    """
    vol = world()
    b = builder(vol)
    before = dict(b._pending)
    res = b.building(None, 14, 14, 17, 17, 2, "gable", mat=MAT)
    assert not res["ok"], "a 4x4 two-storey building was built with no way upstairs"
    assert res["cells"] == 0 and b._pending == before
    assert "needs 6 columns along it and three across" in res["reason"], res["reason"]
    # ...and the same footprint at one storey is fine, because it needs no flight.
    one = b.building(None, 14, 14, 17, 17, 1, "shed", mat=MAT)
    assert one["ok"], one["reason"]
    return "2 storeys refused, 1 storey built"


@case
def t_a1_every_block_it_places_is_a_real_block():
    """Six palettes, six forms."""
    seen, bad = set(), {}
    forms = (dict(storeys=1, roof="shed"), dict(storeys=2, roof="gable"),
             dict(storeys=3, roof={"style": "hip", "pitch": (1, 2)}),
             dict(storeys=2, roof="gambrel", wing=(26, 16, 32, 22)),
             dict(storeys=2, roof="mansard", outshot=3, porch=True),
             dict(storeys=2, roof="flat", chimney=True))
    for fam, kw in zip(("cobblestone", "quartz", "sandstone", "blackstone",
                        "stone_brick", "brick"), forms):
        mat = dict(MAT, wall=fam, footing=fam)
        vol = world()
        b = builder(vol)
        res = b.building(None, *BOX, kw.pop("storeys"), kw.pop("roof"), mat=mat, **kw)
        assert res["ok"], (fam, res["reason"])
        b.resolve_steps()
        seen |= set(b._pending.values())
    bad = registry.check_all(sorted(seen))
    assert not bad, bad
    return f"{len(seen)} distinct block states over 6 palettes and 6 forms, all valid"


# ------------------------------------------------------ A2. three cells, not two
LOFT_X0, LOFT_Z0, LOFT_X1, LOFT_Z1 = 12, 12, 24, 22


def two_storey_shell(b: Builder, ceiling: int) -> None:
    """A room with a floor at GROUND, a ceiling `ceiling` blocks above it, and a door.

        Deliberately hand-built rather than built with `building()`: the case is about what
        the walk model says, and a fixture made by the thing under test proves nothing.
        
    """
    for x in range(LOFT_X0, LOFT_X1 + 1):
        for z in range(LOFT_Z0, LOFT_Z1 + 1):
            b.place_block(x, GROUND, z, "stone_bricks")
            edge = x in (LOFT_X0, LOFT_X1) or z in (LOFT_Z0, LOFT_Z1)
            for y in range(GROUND + 1, GROUND + ceiling):
                b.place_block(x, y, z, "stone_bricks" if edge else "air")
            b.place_block(x, GROUND + ceiling, z, "stone_bricks")


@case
def t_a2_a_stair_under_a_two_cell_ceiling_is_not_walkable():
    vol = world()
    b = builder(vol)
    two_storey_shell(b, ceiling=3)          # cells GROUND+1, +2 open; +3 is the ceiling
    for i in range(2):                      # a flight by hand, two cells cleared
        b.place_block(16 + i, GROUND + 1 + i, 16,
                      f"cobblestone_stairs[facing=east,half=bottom]")
    v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(b._pending)
    nav = observe.Nav(v)
    foot = nav.stance_near(15, 16, GROUND + 1, tol=2)
    up = nav.stance_near(17, 16, GROUND + 3, tol=2)
    assert foot is not None and up is not None, (foot, up)
    reach = nav.flood([(15, 16, foot)], max_jumps=0)
    assert (17, 16, up) not in reach, \
        "the walk model still walks up a flight with two cells over its treads"
    assert observe.STAIR_CLEAR == 6, observe.STAIR_CLEAR
    return f"foot {foot}, top {up}: unreachable on foot with 2 cells of headroom"


@case
def t_a2_flight_clears_the_third_cell_and_the_same_stair_is_walkable():
    vol = world()
    b = builder(vol)
    two_storey_shell(b, ceiling=4)
    res = b.flight("hut", 16, 16, GROUND, GROUND + 3, "east", mat="cobblestone")
    assert res["ok"], res["reason"]
    b.resolve_steps()
    thin = []
    for i in range(3):
        col = (16 + i, GROUND + 1 + i, 16)
        for k in (1, 2, 3):
            got = b._pending.get((col[0], col[1] + k, col[2]))
            if got is not None and got.split("[")[0] not in ("air",):
                thin.append((col, k, got))
    assert not thin, thin
    return f"{res['cells']} cells, {res['removed']} cleared, three over every tread"


# ------------------------------------------------- A3. the step inside the room
@case
def t_a3_a_dais_brings_its_own_way_up():
    vol = world()
    b = builder(vol)
    two_storey_shell(b, ceiling=4)
    b.place_block(18, GROUND + 1, LOFT_Z1,
                  "oak_door[facing=north,half=lower]")
    b.place_block(18, GROUND + 2, LOFT_Z1,
                  "oak_door[facing=north,half=upper]")
    bare = dict(b._pending)

    # the raised floor with no tread onto it.
    for x in range(14, 19):
        for z in range(14, 18):
            bare[(x, GROUND + 1, z)] = "spruce_planks"
    before = _reach_on(vol, bare, (14, 18, 14, 17), GROUND + 2)
    assert before == 0, f"{before} cells of the bare platform are already walkable"

    res = b.dais(14, 14, 18, 17, GROUND + 1, "spruce")
    assert res["ok"], res["reason"]
    b.resolve_steps()
    after = _reach_on(vol, b._pending, (14, 18, 14, 17), GROUND + 2)
    assert after == 20, f"only {after} of the dais's 20 cells can be walked to"
    return (f"bare platform {before}/20 walkable, dais {after}/20 with "
            f"{res['step_cells']} slab steps on its {res['side']} side")


def _reach_on(vol, pending, rect, y) -> int:
    """How many cells of a rectangle at level `y` a person can walk to from the door."""
    v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(pending)
    nav = observe.Nav(v)
    s = nav.stance_near(18, LOFT_Z1, GROUND + 1, tol=2)
    reach = set(nav.flood([(18, LOFT_Z1, s)], max_jumps=0)) if s is not None else set()
    n = 0
    for x in range(rect[0], rect[1] + 1):
        for z in range(rect[2], rect[3] + 1):
            st = nav.stance_near(x, z, y, tol=1)
            n += st is not None and (x, z, st) in reach
    return n


# ------------------------------------------------------- A4. fittings refuse
class Room:
    """A floor at y=63, open above it, and whatever has been placed since."""

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


class Fitted(Room, Builder):
    def __init__(self):
        Room.__init__(self)


def _room() -> Fitted:
    return Fitted()


@case
def t_a4_a_bed_into_a_lantern_refuses_and_places_nothing():
    """Neither the linter nor the judge could see it."""
    b = _room()
    lit = b.fitting("light", 10, 64, 10, "north", mat="cobblestone")
    assert lit["ok"], lit["reason"]
    before = dict(b.blocks)
    res = b.fitting("bed", 10, 64, 11, "south", mat="spruce")   # foot lands on (10,10)
    assert not res["ok"], "a bed was laid through a lantern"
    assert res["cell"] == [10, 64, 10], res["cell"]
    assert b.blocks == before, "a refused bed placed blocks"
    return res["reason"][:74]


@case
def t_a4_a_bed_into_a_wall_refuses_and_places_nothing():
    b = _room()
    for y in (64, 65):
        b.place_block(10, y, 12, "cobblestone")                 # the wall
    before = dict(b.blocks)
    res = b.fitting("bed", 10, 64, 11, "north", mat="spruce")   # foot lands on (10,12)
    assert not res["ok"], "a bed was laid into a wall"
    assert res["cell"] == [10, 64, 12], res["cell"]
    assert b.blocks == before
    ok = b.fitting("bed", 10, 64, 11, "south", mat="spruce")    # the other way round
    assert ok["ok"], ok["reason"]
    return res["reason"][:74]


@case
def t_a4_a_light_is_chosen_by_what_the_room_is_for():
    """"lanterns everywhere".

        Each room is given what its light actually needs -- a ceiling to hang from, a wall
        to sit on -- because a light that cannot be hung falls back to the one that only
        needs a floor, which is deliberate: a room with no light in it is worse than a room
        lit the ordinary way.
        
    """
    lights = {"lantern", "candle", "wall_torch", "glowstone", "torch"}
    got = {}
    for room, ceiling, wall in (("moot hall", True, False), ("house", False, False),
                                ("granary", False, True), ("shrine", False, False),
                                ("smithy", False, False), (None, False, False)):
        b = _room()
        if ceiling:
            b.place_block(10, 65, 10, "cobblestone")
        if wall:
            b.place_block(10, 64, 11, "cobblestone")            # behind, facing north
        r = b.fitting("light", 10, 64, 10, "north", mat="cobblestone", room=room)
        assert r["ok"], (room, r["reason"])
        got[room] = next(s for s in b.blocks.values()
                         if s.split("[")[0] in lights)
    assert got[None].startswith("lantern[hanging=false"), got
    assert got["moot hall"].startswith("lantern[hanging=true"), got
    assert got["house"].startswith("candle"), got
    assert got["granary"].startswith("wall_torch"), got
    assert got["shrine"] == "glowstone", got
    assert len({v.split("[")[0] for v in got.values()}) == 4, got
    return ", ".join(f"{k or 'unsaid'}={v.split('[')[0]}" for k, v in got.items())


# --------------------------------------------------------- A5. six more words
@case
def t_a5_the_six_new_words_place_without_error():
    seen, refused = set(), []
    for fam in ("cobblestone", "spruce", "quartz", "stone_brick", "sandstone",
                "blackstone"):
        for kind in ("bookshelf", "bench", "shelf", "rug", "oven", "well"):
            b = _room()
            r = b.fitting(kind, 10, 64, 10, "north", mat=fam, extent=2)
            if not r["ok"]:
                refused.append((fam, kind, r["reason"]))
                continue
            seen |= set(b.blocks.values())
    assert not refused, refused
    bad = registry.check_all(sorted(seen))
    assert not bad, bad
    have = set(Fitted.FITTING_BLOCKS)
    assert {"bookshelf", "bench", "shelf", "rug", "oven", "well"} <= have, sorted(have)
    return f"6 words x 6 palettes, {len(seen)} distinct states, all valid"


# ------------------------------------------------------------- A6. lanes lit
@case
def t_a6_a_lamp_post_never_costs_a_lane_cell_its_stance():
    """A lamp post that took a stance off the lane would be worse than a dark lane, so that
    is what is asserted.
    """
    ground = np.full((SX, SZ), GROUND, int)
    cells = {(x, 20): {"y": GROUND, "rank": 0, "face": None} for x in range(8, 48)}
    net = Network(cells, [Threshold("hut", 20, GROUND, 21, "south",
                                    (20, GROUND + 1, 22))])
    vol = world()
    b = builder(vol, net)
    dark = circulate.emit(b, net, "cobblestone", ground=ground, x0=0, z0=0,
                          clear_veg=False, post_every=0)
    b.resolve_steps()
    unlit = dict(b._pending)

    b2 = builder(world(), net)
    lit = circulate.emit(b2, net, "cobblestone", ground=ground, x0=0, z0=0,
                         clear_veg=False, post_every=12)
    b2.resolve_steps()
    assert lit.get("posts", 0) >= 3, lit
    assert lit["lanterns"] > dark["lanterns"], (dark, lit)

    def stances(pending):
        v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
        v.overlay(pending)
        nav = observe.Nav(v)
        return {c: nav.stance_near(c[0], c[1], GROUND + 1, tol=1) for c in cells}

    was, now = stances(unlit), stances(b2._pending)
    lost = [c for c in cells if was[c] is not None and now[c] != was[c]]
    assert not lost, f"{len(lost)} lane cells lost their stance to a lamp post: {lost[:5]}"
    # ...and no post stands on the lane, on the threshold or on its doorstep
    on_lane = [p for p in b2._pending if (p[0], p[2]) in cells
               and p not in unlit and p[1] > GROUND]
    assert not on_lane, on_lane[:5]
    return (f"{lit['posts']} posts, {lit['lanterns']} lanterns against "
            f"{dark['lanterns']}, {len(cells)} lane cells all still standable")


@case
def t_a6_nothing_the_lane_pass_builds_stands_on_a_reserved_doorstep():
    ground = np.full((SX, SZ), GROUND, int)
    ground[:, 21:] = GROUND - 6              # a bank falling away south of the lane
    cells = {(x, 20): {"y": GROUND, "rank": 0, "face": None} for x in range(8, 48)}
    thresholds = [Threshold(f"hut{i}", x, GROUND, 20, "south",
                            (x, GROUND + 1, 21))
                  for i, x in enumerate((14, 22, 30))]
    b = builder(world(), Network(cells, thresholds))
    circulate.emit(b, Network(cells, thresholds), "cobblestone", ground=ground,
                   x0=0, z0=0, clear_veg=False)
    b.resolve_steps()
    fenced = [t.id for t in thresholds
              if b._pending.get((t.door[0], t.door[1], t.door[2]), "air")
              .split("[")[0].endswith("_wall")]
    assert not fenced, f"the lane pass fenced the doorstep of {fenced}"
    # ...and the rail is still built where there is no doorstep, which is the point of
    # it
    rails = [p for p, s in b._pending.items()
             if s.split("[")[0].endswith("_wall") and p[2] == 21]
    assert len(rails) > 10, f"only {len(rails)} rails on a 40-cell lane above a bank"
    return f"{len(rails)} rails laid along the drop, 0 on any of 3 reserved doorsteps"


# A2 the between-vocabulary as parameters, A3 water is not ground, A4 a door checks the
# cell it opens onto.
# ==============================================================================


def a_wet_world(depth: int = 2) -> Volume:
    """Flat ground at y=60, with a lake `depth` blocks deep south of z=26."""
    b = {}
    for x in range(SX):
        for z in range(SZ):
            wet = z >= 26
            g = GROUND - depth if wet else GROUND
            for y in range(Y0, g + 1):
                b[(x, y, z)] = "stone"
            if wet:
                for y in range(g + 1, GROUND + 1):
                    b[(x, y, z)] = "water"
    return Volume.from_blocks(b, 0, Y0, 0, SX, 80, SZ)


class LiveSite(offline.OfflineSite):
    """An `OfflineSite` whose `height()` counts water, as the live server's does.

        `MOTION_BLOCKING_NO_LEAVES` is motion blocking, and water blocks motion, so over a
        lake `Site.height` is the water surface. The offline heightmap is computed from what
        a *body collides with* instead, and a body does not collide with water -- so a dry
        run has never been able to exhibit this defect and a fixture that used one would
        prove nothing. This is the live rule, in a fixture.
        
    """

    def height(self, wx: int, wz: int) -> int:
        x = min(max(int(wx) - self.x, 0), self.sx - 1)
        z = min(max(int(wz) - self.z, 0), self.sz - 1)
        top = self.vol.y0
        for y in range(self.vol.y0 + self.vol.shape[1] - 1, self.vol.y0 - 1, -1):
            if self.vol.name(int(wx), y, int(wz)) not in ("air", "cave_air", "void_air"):
                top = y
                break
        return top


def live_builder(vol: Volume, net: Network | None = None) -> Builder:
    b = Builder(LiveSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net else None
    return b


@case
def t_a3_a_plinth_over_water_is_carried_to_the_bed_not_to_the_waterline():
    """`plinth(to_grade=True)` filled to `get_height`, which over water is the surface, and
    every instrument said the building met the ground -- because it did, the ground it
    was told about.
    """
    vol = a_wet_world(2)
    b = live_builder(vol)
    # what the live heightmap says, and what is actually there
    assert b.get_height(20, 30) == GROUND, b.get_height(20, 30)
    assert b.grade(20, 30) == GROUND - 2, b.grade(20, 30)
    assert b.wet(20, 30) == GROUND, b.wet(20, 30)
    assert b.grade(20, 20) == GROUND and b.wet(20, 20) is None
    b.plinth(16, 28, 22, 34, GROUND + 1, "stone_bricks", courses=1, to_grade=True)
    water = [(x, y, z) for x in range(16, 23) for z in range(28, 35)
             for y in range(GROUND - 1, GROUND + 1)
             if b._pending.get((x, y, z), vol.name(x, y, z)) == "water"]
    assert not water, f"{len(water)} columns under the plinth are still water: {water[:4]}"
    return (f"live height {GROUND}, bed {GROUND - 2}: "
            f"{len(b._pending)} blocks, 0 water left under a 7x7 plinth")


@case
def t_a3_the_sounding_reads_the_world_and_not_the_program_own_air():
    """The one way this could have broken thirteen replaying wave programs: every
    stored program clears its ground before it plinths, and clearing writes `air` at the
    old surface. A sounding that could see its own work would walk down through it."""
    vol = world()
    b = builder(vol)
    b.place_block(20, GROUND, 20, "air")          # a tree cleared, as every program does
    assert b.get_block(20, GROUND, 20) == "air"
    assert b.grade(20, 20) == GROUND, \
        f"the sounding followed the program's own air down to {b.grade(20, 20)}"
    return f"cleared column reads air, grade still {GROUND}"


@case
def t_a4_a_fence_in_front_of_a_door_is_refused():
    """A fence standing in the cell a door opens onto. E002 was satisfied by another route,
    the lint was silent, and nothing in the stack asks what is in front of a door.
    """
    vol = world()
    b = builder(vol)
    b.place_cuboid(14, GROUND + 1, 18, 14, GROUND + 2, 18, "cobblestone")
    b.place_block(14, GROUND + 1, 20, "cobblestone")
    b.place_block(15, GROUND + 1, 19, "oak_fence")          # in front of the door
    before = dict(b._pending)
    res = b.doorway(14, GROUND + 1, 19, "west", "cobblestone")
    assert not res["ok"], "a door was hung facing a fence"
    assert b._pending == before, "a refused doorway placed blocks"
    assert "oak_fence" in res["reason"], res["reason"]
    # ...and with the fence gone the same call hangs the door
    b.place_block(15, GROUND + 1, 19, "air")
    ok = b.doorway(14, GROUND + 1, 19, "west", "cobblestone")
    assert ok["ok"], ok["reason"]
    return res["reason"][:72]


@case
def t_a4_a_door_over_nothing_to_stand_on_is_refused():
    vol = world()
    b = builder(vol)
    b.place_cuboid(14, GROUND + 1, 18, 14, GROUND + 2, 18, "cobblestone")
    b.place_block(14, GROUND + 1, 20, "cobblestone")
    b.place_block(15, GROUND, 19, "air")                    # the doorstep dug away
    res = b.doorway(14, GROUND + 1, 19, "west", "cobblestone")
    assert not res["ok"], "a door was hung over a hole"
    assert "not ground a person can stand on" in res["reason"], res["reason"]
    # front="ignore" is for a caller that means it -- a door onto a deck it will build
    ok = b.doorway(14, GROUND + 1, 19, "west", "cobblestone", front="ignore")
    assert ok["ok"], ok["reason"]
    return res["reason"][:72]


@case
def t_a4_the_lane_pass_leaves_every_doorstep_on_round_11_clear():
    """The measurement the spec names, on real ground rather than a fixture."""
    state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                         "site_f")
    if not os.path.exists(os.path.join(state, "world.npz")):
        return "skipped."
    v = offline.load_volume(state + "/world.npz")
    net = Network.load(state + "/network.json")
    # `world.npz` is cached after the circulation stage -- so re-emitting on it lays no
    # rail at all and would assert nothing. This is the same fourteen real thresholds
    # under maximum pressure instead: every column that is not lane is six blocks below
    # it, so every off-lane neighbour of every lane cell is a rail candidate and the
    # keep set is the only thing standing between the pass and a fence on all fourteen
    # doorsteps.
    lane_y = min(r["y"] for r in net.cells.values())
    ground = np.full((v.shape[0], v.shape[2]), lane_y - 6, int)
    for (x, z), rec in net.cells.items():
        if 0 <= x - v.x0 < v.shape[0] and 0 <= z - v.z0 < v.shape[2]:
            ground[x - v.x0, z - v.z0] = rec["y"]
    b = Builder(offline.OfflineSite(v))
    b._vol = v
    b.frontage = Frontage(v, net)
    stats = circulate.emit(b, net, "cobblestone", ground=ground, x0=v.x0, z0=v.z0,
                           clear_veg=False)
    b.resolve_steps()
    assert stats["rails"] > 200, f"only {stats['rails']} rails: the case has no pressure"
    fenced = []
    for t in net.thresholds:
        dx, dz = {"north": (0, -1), "south": (0, 1), "east": (1, 0),
                  "west": (-1, 0)}[t.facing]
        for cell in ((t.x + dx, t.z + dz), (t.door[0], t.door[2])):
            for y in (t.y + 1, t.y + 2, t.y + 3):
                got = b._pending.get((cell[0], y, cell[1]), "air").split("[")[0]
                if got not in ("air", "cave_air"):
                    fenced.append((t.id, cell, y, got))
    assert not fenced, f"{len(fenced)} doorstep cells built on: {fenced[:4]}"
    return (f"{len(net.thresholds)} of {len(net.thresholds)} doorsteps clear with "
            f"{stats['rails']} rails laid round them on {len(net.cells)} lane cells")


def a_lane(z: int = 14, wall_z: int = 16, x_from: int = 8, x_to: int = 48):
    """A straight lane at `z`, and one threshold reserved into the wall at `wall_z`.

        `Threshold` is (id, x, z, y, facing, door): the lane cell, its surface, the way you
        face walking in off it, and where the building pass must hang the leaf.
        
    """
    mid = 20
    cells = {(x, z): {"y": GROUND, "rank": 0, "face": None}
             for x in range(x_from, x_to)}
    return Network(cells, [Threshold("hut", mid, z, GROUND, "south",
                                     (mid, GROUND + 1, wall_z))])


@case
def t_a2_every_word_of_the_between_vocabulary_at_once():
    """The case the spec registers: one `building()` with every new parameter set, and
        it has to lint clean and be walkable from its own door.

        All seven together, because the failure mode of a vocabulary is not one word -- it
        is the jetty clearing the wall the oriel projects from, the yard wall standing in
        front of the door `doorway()` has just refused to hang against a fence, the flashing
        laid where the dormer already is.
        
    """
    net = a_lane()
    vol = a_wet_world(2)
    b = live_builder(vol, net)

    class Reg:
        def plots_list(self):
            return [dict(PLOT, x0=8, z0=13, x1=40, z1=36)]
        claimed_this_pass = ()

    b.registry = Reg()
    res = b.building("hut", 14, 16, 26, 24, 2, "gable", mat=MAT,
                     chimney=True, dormers=2, jetty=1, oriel=("east", 1),
                     brackets=True, flashing=True, yard=(8, 15, 13, 25), deck=True)
    assert res["ok"], res["reason"]
    b.resolve_steps()
    ex = res["extras"]
    refused = {k: v["reason"] for k, v in ex.items() if not v["ok"]}
    assert not refused, refused
    got, all_ = walked(b, res)
    assert got == all_, f"only {got} of {all_} floor cells are walkable: {ex.keys()}"
    errs = errors(vol, b._pending, plots=[dict(PLOT, x0=8, z0=13, x1=40, z1=36)])
    assert not errs, errs
    assert door_reachable(vol, b._pending, res["door"]), res["approach"]["reason"]
    assert b.check_attached()["ok"], b.check_attached()["reason"]
    bad = registry.check_all(sorted(set(b._pending.values())))
    assert not bad, bad
    return (", ".join(f"{k} {v['cells']}" for k, v in ex.items())
            + f"; {got}/{all_} walkable, 0 errors")


@case
def t_a2_each_word_refuses_by_name_rather_than_half_building_itself():
    """A refusal is the library saying its vocabulary does not reach *this* shape. It is
    per word: a jetty that cannot be built does not cost you the dormers."""
    reasons, checks = {}, 0
    for kw, want in ((dict(jetty=0), "storey 0 cannot jetty"),
                     (dict(jetty=9), "storeys 0 to 1"),
                     (dict(oriel=("up", 1)), "north, south, east or west"),
                     (dict(oriel=("south", 0)), "the wall the door is in"),
                     (dict(flashing=True), "no chimney on this building"),
                     (dict(deck=True), "is on dry ground"),
                     (dict(yard=(0, 0, 2, 2)), "leaves the plot"),
                     (dict(dormers=2, roof="flat"), "no slope to put a dormer in")):
        vol = world()
        b = builder(vol)

        class Reg:
            def plots_list(self):
                return [dict(PLOT)]
            claimed_this_pass = ()

        b.registry = Reg()
        roof = kw.pop("roof", "gable")
        res = b.building("hut", *BOX, 2, roof, mat=MAT, **kw)
        assert res["ok"], res["reason"]          # the building still stands
        name = next(iter(kw))
        e = res["extras"][name]
        assert not e["ok"], f"{name}={kw[name]!r} was built: {e['reason']}"
        assert want in e["reason"], (name, e["reason"])
        reasons[name] = e["reason"]
        checks += 1
    return (f"{checks} refusals over {len(reasons)} words, each naming what it could "
            f"not do and none of them costing the building")


# the courtyard

@case
def t_r19_a3_a_courtyard_is_four_ranges_round_a_yard_and_the_yard_is_not_a_room():
    """`building(courtyard=(w, d))`, on flat ground and on a slope.

        The one form the shell could not make. It knew a block, a wing, an outshot and a
        porch, and every one of them is a mass with something added to it; a courtyard house
        is a mass with something *taken out*, and the difference is not decoration. Ba Sing
        Se is courtyard houses, so a system that turns a sentence into a place could not turn
        that sentence into a place.

        What is asserted is the whole of what a courtyard has to be:

          - four ranges stand, each of them a room;
          - the **yard is not one of them** -- it is paved, open to the sky, and reads as
            outdoors to the same instrument the walk model reads;
          - every range is walkable in full, from the lane, in through the range the lane
            arrives at and across the yard;
          - and none of it costs a lint error, on flat ground or on a slope.

        Plus the two refusals, because a refusal a caller can act on is worth more than a
        courtyard house with a two-block light well in it.
        
    """
    got = []
    for ground in ("flat", "relief"):
        if ground == "flat":
            vol = world()
            b = builder(vol)

            class Reg:
                def plots_list(self):
                    return [dict(PLOT)]
                claimed_this_pass = ()

            b.registry = Reg()
            box, plot, relief = COURT_BOX, dict(PLOT), 0
        else:
            # The registered relief fixture: a 1:2 fall across the short axis, with a
            # lane routed to it and a doorstep reserved, and `site()` run over it -- so
            # the courtyard is asked to stand on a platform cut out of a bank, which is
            # what a type is ever actually handed.
            rnd, be = make("slope_21_21", str(ROOT))
            plot = json.load(open(os.path.join(rnd.state, "plots.json")))[0]
            vol = be.volume
            b = Builder(offline.OfflineSite(vol))
            b._vol = vol
            b.registry = stages._Registry(rnd.state)
            b.frontage = Frontage(vol, rnd.network())
            h, _w = observe.ground_heights(vol)
            sub = h[plot["x0"]:plot["x1"] + 1, plot["z0"]:plot["z1"] + 1]
            relief = int(sub.max() - sub.min())
            part = b.site({"label": plot["label"], "kind": "plot",
                           **{k: plot[k] for k in ("x0", "z0", "x1", "z1")}}, mat=MAT)
            assert part["ground"] == "platform", part["sited"]["reason"]
            box = (part["x0"], part["z0"], part["x1"], part["z1"])
        res = b.building(plot["label"], *box, 1, "gable", mat=MAT, courtyard=(5, 5))
        assert res["ok"], (ground, res["reason"])
        court = res["courtyard"]
        assert court["ok"] and court["paved"] == 25, court
        assert len(res["rooms"]) == 4, res["rooms"]
        b.resolve_steps()

        v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
        v.overlay(b._pending)
        region = (plot["x0"] - 6, plot["z0"] - 6, plot["x1"] + 6, plot["z1"] + 6)
        ctx = lint.Context.build(v, plots=[plot], region=region,
                                 fittings=b.fitting_cells)
        errs = [(f.code, f.message) for f in lint.lint(ctx).findings
                if f.code.startswith("E") and f.code != "E005"]
        assert not errs, (ground, errs)

        # every range walkable in full, and the yard is not a room
        rows = ctx.interior_walk()
        assert rows and all(r["fraction"] == 1.0 for r in rows), \
            [(r["bbox"], r["fraction"]) for r in rows]
        yx0, yz0, yx1, yz1 = court["yard"]
        inside = [r for r in rows
                  if yx0 <= r["bbox"][0] and r["bbox"][3] <= yx1
                  and yz0 <= r["bbox"][2] and r["bbox"][5] <= yz1]
        assert not inside, f"the yard is being counted as a room: {inside}"
        # ...and the sky is over it: every column of the yard is open above the eaves
        for x in range(yx0, yx1 + 1):
            for z in range(yz0, yz1 + 1):
                for y in range(res["floors"][0] + 1, res["eave_y"] + 2):
                    here = b._pending.get((x, y, z), "air")
                    assert here == "air", f"the yard is roofed at {(x, y, z)}: {here}"
        got.append((ground, relief, len(rows), sum(r["cells"] for r in rows),
                    court["entrance"], len(court["corridor"])))

    # ...and it refuses rather than building a light well or a range with no room in it
    vol = world()
    b = builder(vol)

    class Reg2:
        def plots_list(self):
            return [dict(PLOT)]
        claimed_this_pass = ()

    b.registry = Reg2()
    small = b.building("hut", *COURT_BOX, 1, "gable", mat=MAT, courtyard=(2, 2))
    assert not small["ok"] and "3x3" in small["reason"], small["reason"]
    tight = b.building("hut", 14, 14, 22, 22, 1, "gable", mat=MAT, courtyard=(5, 5))
    assert not tight["ok"] and "range is at least 3" in tight["reason"], tight["reason"]
    mixed = b.building("hut", *COURT_BOX, 1, "gable", mat=MAT, courtyard=(5, 5),
                       outshot=3)
    assert not mixed["ok"] and "different building" in mixed["reason"], mixed["reason"]
    return (f"4 ranges round a 5x5 yard on flat ground and on {got[1][1]} of relief: "
            f"{got[0][3]} and {got[1][3]} floor cells, every one of them walkable, "
            f"entered through the {got[0][4]} and {got[1][4]} range by a passage of "
            f"{got[0][5]} and {got[1][5]} cells, 0 lint errors either way, and the "
            f"yard is not a room; a 2x2 yard, a 9x9 footprint and a courtyard with an "
            f"outshot all refused by name")


# --------------------------------------- demo-polish, phase 1a: a fill sees its own
# work
@case
def t_dp1a_a_replace_fill_reads_the_pending_set_and_leaves_a_roof_standing():
    """The library defect under the demo's blank throne hall (demo-polish spec, 1a).

        `fill_region(replace="air")` tested `replace` against the **pre-build world** and
        not against the pending set, so a fill queued after a roof read every roof cell as
        the air that stood there before the pass began and overwrote it. `get_block` reads
        pending writes first; `fill_region` now has the same precedence. The case is the
        temple's own sequence in miniature: a roof laid, then a replace-air fill over the
        roof's whole volume -- every block the roof placed is byte-identical after, and the
        fill still filled the air the roof left.

        Against the old semantics the roof is overwritten and this fails, which is what a
        discriminating case is.
        
    """
    vol = world()
    b = builder(vol)
    x0, z0, x1, z1 = 14, 14, 26, 24
    eave = GROUND + 5
    ridge = b.roof(x0, z0, x1, z1, eave, "dark_oak", style="hip", pitch=(1, 1))
    roof = {p: blk for p, blk in b._pending.items()}
    assert roof, "the roof placed nothing; the case has no subject"
    # the fill the temple makes: the roof's volume and more, air only
    b.fill_region(x0, eave, z0, x1, eave + 9, z1, "cobblestone", replace="air")
    kept = sum(1 for p, blk in roof.items() if b._pending.get(p) == blk)
    lost = [p for p, blk in roof.items() if b._pending.get(p) != blk]
    assert not lost, (f"{len(lost)} of {len(roof)} roof cells were overwritten by a "
                      f"replace-air fill, e.g. {lost[0]} -> {b._pending.get(lost[0])}")
    filled = [p for p, blk in b._pending.items() if blk == "cobblestone"]
    assert filled, "the fill placed nothing at all: it cannot tell air from a roof"
    assert all(p not in roof for p in filled)
    # ...and a fill on a cell the world holds solid is still refused: the world is still
    # read where the program has not written
    b2 = builder(vol)
    b2.fill_region(20, GROUND, 20, 20, GROUND, 20, "cobblestone", replace="air")
    assert (20, GROUND, 20) not in b2._pending, "a replace-air fill wrote over stone"
    b2.fill_region(20, GROUND, 20, 20, GROUND, 20, "cobblestone", replace="stone")
    assert b2._pending.get((20, GROUND, 20)) == "cobblestone"
    return (f"roof of {len(roof)} cells to ridge {ridge} intact under a replace-air fill "
            f"that still filled {len(filled)} air cells; the world is read where nothing "
            f"is pending")


@case
def t_dp1c_a_fitting_into_a_flight_refuses_and_the_flight_stays_walkable():
    """The library defect under the temple's sealed top storey (demo-polish, 1c).

        A type furnishes its upper storeys without knowing where its own flight is, and
        the temple stood a barrel on the foot of its stair and another over the well it
        climbs into: the storey above was reachable only by jumping, E011 on real ground.
        A flight's cells -- treads, landing, foot and the headroom over each -- are the
        flight's, held by `flight()` and refused by `fitting()` by name. Against the old
        code the barrel is placed and this fails.
        
    """
    vol, b, res = a_building(storeys=2)
    assert res["ok"], res["reason"]
    fy = res["floors"][0]
    # a flight of our own, on the ground floor, clear of the shell's
    x, z = BOX[0] + 3, BOX[3] - 2
    fl = b.flight("hut", x, z, fy, fy + 4, "east", mat="cobblestone")
    assert fl["ok"], fl["reason"]
    assert b.flight_cells, "the flight recorded no cells"
    foot = (x - 1, fy + 1, z)
    assert foot in b.flight_cells
    r = b.fitting("store", foot[0], foot[1], foot[2], "north", mat="spruce")
    assert not r["ok"], "a barrel was stood on the foot of the flight"
    assert "flight" in r["reason"], r["reason"]
    assert foot not in b._pending or "barrel" not in b._pending[foot]
    over = (x + 1, fy + 3, z)                     # the headroom over the second tread
    assert over in b.flight_cells
    r2 = b.fitting("light", over[0], over[1], over[2], "north", mat="spruce")
    assert not r2["ok"], "a lantern was hung in the headroom over a tread"
    # ...and the floor between the storey's door and the foot of the shell's own flight
    # is held the same way -- against a *write*, which is what walls a corridor in.
    # Furniture is deliberately not refused on it: a refusal is something the type reads
    # and acts on, and one committed type acts on it by building something else.
    assert b.flight_way, "building() held no way to the foot of its flight"
    wx, wy, wz = sorted(b.flight_way)[0]
    r4 = b.fitting("store", wx, wy, wz, "north", mat="spruce")
    assert r4["ok"], f"furniture on the way was refused: {r4['reason']}"
    # ...and off the flight the same fitting is placed, so the refusal is the flight's
    # (the shell laid a flight of its own too; take a floor cell nothing holds)
    free = [(cx, cz) for cx in range(BOX[0] + 1, BOX[2]) for cz in range(BOX[1] + 1, BOX[3])
            if (cx, fy + 1, cz) not in b.flight_cells
            and (cx, fy + 1, cz) not in b.flight_way
            and b.get_block(cx, fy + 1, cz) == "air" and b.get_block(cx, fy + 2, cz) == "air"
            and b.get_block(cx, fy, cz) != "air"]
    r3 = next(b.fitting("store", cx, fy + 1, cz, "north", mat="spruce") for cx, cz in free)
    assert r3["ok"], r3["reason"]
    # a refused building winds the register back with everything else
    mark = b._mark()
    assert "flights" in mark and mark["flights"] == b.flight_cells
    assert mark.get("flight_way") == b.flight_way
    return (f"{len(b.flight_cells)} cells held for a 4-tread flight and "
            f"{len(b.flight_way)} for the way to a foot; a store on its foot and a "
            f"light in its headroom refused by name, the same store placed off them "
            f"and on the way, which is held against writes and not against furniture")


@case
def t_dp1c_a_yard_wall_opens_where_the_wall_would_stand_not_where_the_building_does():
    """The library defect under the temple's sealed cloister (demo-polish, 1c).

        `building(yard=)` put its gap at the perimeter column nearest the doorstep. With a
        hall pinned to the plot's edge and its door in that edge, the nearest perimeter is
        the hall's own wall, which the ring skips anyway -- so the yard behind the hall
        was walled with no gap at all, a court nobody could walk into, E003 on real
        ground. The gap goes on a perimeter column the wall would actually stand in.
        
    """
    vol = world()
    b = builder(vol)

    class Reg:
        def plots_list(self):
            return [dict(PLOT)]
        claimed_this_pass = ()
    b.registry = Reg()
    # the hall on the plot's west edge; the yard is the whole plot, behind it
    box = (PLOT["x0"], 14, PLOT["x0"] + 10, 22)
    yard = (PLOT["x0"], PLOT["z0"], PLOT["x1"], PLOT["z1"])
    res = b.building("hut", *box, 1, "gable", mat=MAT, yard=yard)
    assert res["ok"], res["reason"]
    y = res["extras"]["yard"]
    assert y["ok"], y["reason"]
    inside = [(x, z) for (x, z) in y["gap"]
              if box[0] <= x <= box[2] and box[1] <= z <= box[3]]
    assert not inside, f"the yard's gap lands on the building itself: {y['gap']}"
    # ...and the gap is really open: no wall block was laid in any of its cells
    fam = MAT["footing"]
    for (x, z) in y["gap"]:
        g = b.grade(x, z)
        blk = b.get_block(x, g + 1, z)
        assert "wall" not in blk, f"a wall block stands in the gap at {(x, z)}: {blk}"
    return f"gap {sorted(y['gap'])} on free perimeter beside a hall on the plot's edge"


@case
def t_dp3a_the_way_in_is_held_open_even_where_nothing_had_to_be_laid():
    """Demo-polish, 3a. Two of the city's doors were walkable when their parts were
    sited and not when they were built: the approach had found the door reachable
    over ground as it stood, laid nothing, recorded nothing, and the type then built
    its footing across the very columns the walk used. The walk is written down as the
    way in now, and `TypeBuilder` holds it open against the type's own writes."""
    net = a_lane()
    vol = world()
    b = builder(vol, net)

    plot = dict(PLOT, z0=16)                 # the plot south of the lane at z=14

    class Reg:
        def plots_list(self):
            return [dict(plot)]
        claimed_this_pass = ()
    b.registry = Reg()
    # the reserved doorstep is on the lane at z=14; the pad's door is inset from z=16
    part = b.site({"label": "hut", "kind": "plot", "x0": plot["x0"], "z0": plot["z0"],
                   "x1": plot["x1"], "z1": plot["z1"]}, mat=MAT)
    assert part["sited"]["approach"]["ok"], part["sited"]["approach"]
    ap = part["sited"]["approach"]
    assert ap["cells"] == 0, "the fixture should need no path laid; it is level ground"
    walked = [(c[0], c[2]) for c in part["way"]]
    assert ap.get("walked", 0) > 0, ap
    assert len(set(walked)) >= 2, part["way"]
    door = (int(part["door"][0]), int(part["door"][1]))
    assert door in set(walked)
    between = [c for c in set(walked) if c != door]
    assert between, "no column between the lane and the door was recorded"
    assert len(set(walked)) <= 8, ("the walk recorded is not the shortest", sorted(set(walked)))
    # a type covering a walked column has it put back
    tb = b.type_builder(part)
    cx, cz = between[0]
    y = part["floor_y"] + 1
    tb.place_block(cx, y, cz, "cobblestone")
    assert b._pending.get((cx, y, cz)) != "cobblestone", \
        "a type built on the way in and the library let it stand"
    assert tb.refused and "doorstep" in tb.refused[-1]["reason"], tb.refused
    return (f"door at {door}, {len(set(walked))} columns walked from the lane held open; "
            f"a block on one of them put back and refused by name")



def main():
    bad = 0
    for name, fn in CASES:
        try:
            note = fn()
            print(f"ok   {name:60s} {note}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:60s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:60s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} building cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
