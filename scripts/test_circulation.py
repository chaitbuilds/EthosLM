"""The circulation pass, judged at the cheapest observation that can exhibit its defect.

No server, no world, no render: synthetic terrain, the planner, the real block emitter,
and then the **real Nav model** re-deriving reachability from the blocks that came out.
The planner's own opinion of its network (`circulate.walk_check`) is checked too, but it
is not the evidence -- it shares the planner's assumptions. Nav does not.

What each case exists to catch:

flat            no stairs where the ground does not rise; a lane that stairs on the
level is a lane that fights the walk model for nothing slope           every rise is a
stair *facing up-slope*. One network, zero jumps, every site served cliff           a
route round an obstacle rather than up it, because a lane that needs a jump is not a
lane landings        a cell that would need to face two ways is flattened, not shipped
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from ethoslm import circulate, observe  # noqa: E402

FAILURES = []
CASES = 0


def check(name, cond, detail=""):
    global CASES
    CASES += 1
    if not cond:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")
    else:
        print(f"  ok    {name}  {detail}")


# ------------------------------------------------------------------ the fixtures

class FakeBuilder:
    """The build API without a world: exactly what Builder gives a model program."""

    def __init__(self, heights, x0, z0):
        self.h = heights
        self.x0, self.z0 = x0, z0
        self.blocks = {}
        self._pending = self.blocks

    def place_block(self, x, y, z, block):
        self.blocks[(int(x), int(y), int(z))] = block

    def get_height(self, x, z):
        i, j = int(x) - self.x0, int(z) - self.z0
        i = min(max(i, 0), self.h.shape[0] - 1)
        j = min(max(j, 0), self.h.shape[1] - 1)
        return int(self.h[i, j])

    def clear_trees(self, x0, z0, x1, z1, margin=0):
        return 0                      # the fixtures are bare ground; nothing to clear

    def dress_ground(self, *a, **k):
        return 0

    def get_block(self, x, y, z):
        p = (int(x), int(y), int(z))
        if p in self.blocks:
            return self.blocks[p].split("[")[0]
        return "stone" if y <= self.get_height(x, z) else "air"


def volume_of(fb, x0, z0, sx, sz, y0, y1):
    """The world as it would stand: terrain, with the emitted blocks written into it."""
    blocks = {}
    for i in range(sx):
        for j in range(sz):
            g = int(fb.h[i, j])
            for y in range(y0, min(g, y1 - 1) + 1):
                blocks[(x0 + i, y, z0 + j)] = "stone"
    for p, b in fb.blocks.items():
        if b == "air":
            blocks.pop(p, None)
        else:
            blocks[p] = b
    return observe.Volume.from_blocks(blocks, x0, y0, z0, sx, y1 - y0, sz)


def terrain(kind, n=64):
    """Deterministic ground. No randomness: a test that varies is not a test."""
    x = np.arange(n)[:, None] * np.ones((1, n))
    z = np.ones((n, 1)) * np.arange(n)[None, :]
    if kind == "flat":
        h = np.full((n, n), 70.0)
    elif kind == "slope":
        h = 66 + x * 0.35
    elif kind == "cliff":
        h = np.where(z < n // 2, 68.0, 68.0)
        h[:, n // 2 - 2:n // 2 + 2] = 96.0          # a wall across the middle
        h[n - 8:, n // 2 - 2:n // 2 + 2] = 68.0     # with one way round the end
    else:  # mesa
        h = (72
             + 14 * (np.sin(x / 11.0) * np.cos(z / 13.0))
             + 10 * (x > 40) * (z > 34)
             + 6 * ((x - 20) ** 2 + (z - 44) ** 2 < 90))
        h[26:32, 8:40] = 96                          # a butte with steep sides
    return np.round(h).astype(int)


def sites_for(kind, n=64):
    if kind == "cliff":
        return [{"id": "north", "x0": 8, "z0": 6, "x1": 16, "z1": 14},
                {"id": "south", "x0": 8, "z0": n - 16, "x1": 16, "z1": n - 8}]
    return [
        {"id": "hall", "x0": 10, "z0": 10, "x1": 20, "z1": 20},
        {"id": "granary", "x0": 34, "z0": 8, "x1": 42, "z1": 16},
        {"id": "well", "x0": 24, "z0": 30, "x1": 30, "z1": 36},
        {"id": "houses", "x0": 44, "z0": 40, "x1": 54, "z1": 50},
        {"id": "watch", "x0": 8, "z0": 46, "x1": 14, "z1": 52},
    ]


# ---------------------------------------------------------------------- the run

def run(kind, n=64, x0=0, z0=0, y0=56, y1=120, expect_stairs=None, verbose=True):
    h = terrain(kind, n)
    sites = sites_for(kind, n)
    net = circulate.plan_network(h, x0, z0, sites, max_step=3)
    wc = circulate.walk_check(net)

    fb = FakeBuilder(h, x0, z0)
    stats = circulate.emit(fb, net, "cobblestone")
    vol = volume_of(fb, x0, z0, n, n, y0, y1)
    nav = observe.Nav(vol)

    # every lane cell, as the walker stands on it
    want = []
    for (x, z), rec in net.cells.items():
        s = nav.stance_near(x, z, rec["y"] + 1, tol=2)
        want.append(((x, z), s))
    missing = [c for c, s in want if s is None]
    seeds = [(c[0], c[1], s) for c, s in want if s is not None]

    reach = nav.flood(seeds[:1], max_jumps=0) if seeds else {}
    off = [c for c, s in want if s is not None and (c[0], c[1], s) not in reach]

    back = [b for b in observe.backwards_stairs(vol)
            if (b["pos"][0], b["pos"][2]) in net.cells]

    if verbose:
        print(f"\n{kind}: {len(net.cells)} lane cells, {stats['steps']} steps, "
              f"cut<= {net.notes['cut_max']} fill<= {net.notes['fill_max']}, "
              f"landing rounds {net.notes['landing_rounds']}, "
              f"{len(net.thresholds)}/{len(sites)} sites served")

    check(f"{kind}: planner says one network",
          wc["unreachable_n"] == 0, f"{wc['unreachable_n']} cells unreachable in plan")
    check(f"{kind}: every lane cell is standable in the world",
          not missing, f"{len(missing)} cells with no stance")
    check(f"{kind}: one network under the real walk model, no jumps",
          not off, f"{len(off)} of {len(want)} lane stances off the network")
    check(f"{kind}: no stair on the lane faces down-slope",
          not back, f"{len(back)} backwards stairs")
    check(f"{kind}: every site has a reserved threshold",
          len(net.thresholds) == len(sites),
          f"{len(net.thresholds)}/{len(sites)}")
    if expect_stairs is not None:
        check(f"{kind}: stairs only where the ground rises",
              (stats["steps"] > 0) == expect_stairs,
              f"{stats['steps']} step cells")

    # a threshold has to be a place you can stand, on the network, with the door leaf
    # one block along the approach -- that is the contract a building pass is held to
    bad_th = []
    for t in net.thresholds:
        s = nav.stance_near(t.x, t.z, t.y + 1, tol=2)
        if s is None or (t.x, t.z, s) not in reach:
            bad_th.append(t.id)
        dx, dz = {v: k for k, v in circulate.FACING.items()}[t.facing]
        if (t.door[0], t.door[2]) != (t.x + dx, t.z + dz):
            bad_th.append(t.id + ":door")
    check(f"{kind}: thresholds stand on the network", not bad_th, str(bad_th))
    return net, nav, reach


print("=== circulation ===")
run("flat", expect_stairs=False)
run("slope", expect_stairs=True)
run("mesa", expect_stairs=True)

net, nav, reach = run("cliff", expect_stairs=None)
check("cliff: the route goes round the wall, not up it",
      max(r["y"] for r in net.cells.values()) < 90,
      f"highest lane cell y={max(r['y'] for r in net.cells.values())}")

# --- the invariant the whole design rests on --------------------------------------
# Adjacent lane cells differ by at most one block, and a cell is a stair for at most one
# direction. Everything above is downstream of these two.
for kind in ("mesa", "slope"):
    h = terrain(kind)
    net = circulate.plan_network(h, 0, 0, sites_for(kind), max_step=3)
    y = {c: r["y"] for c, r in net.cells.items()}
    adj = {c: [(c[0] + dx, c[1] + dz) for dx, dz in circulate.DIRS
               if (c[0] + dx, c[1] + dz) in y] for c in y}
    worst = max((abs(y[a] - y[b]) for a in y for b in adj[a]), default=0)
    check(f"{kind}: lane surface is 1-Lipschitz", worst <= 1, f"worst step {worst}")
    twoway = [c for c in y if len(circulate._needed_facings(c, y, adj)) > 1]
    check(f"{kind}: no cell has to face two ways", not twoway, f"{len(twoway)} cells")

# --- frontage: the check a build pass makes on itself, mid-run ----------------------
# The whole reason for building circulation first. Same machinery a model program gets:
# the world as it stands, plus the blocks the program has decided on and not yet
# flushed.
print("\nfrontage")
from ethoslm.frontage import Frontage  # noqa: E402

h = terrain("mesa")
sites = sites_for("mesa")
net = circulate.plan_network(h, 0, 0, sites, max_step=3)
fb = FakeBuilder(h, 0, 0)
circulate.emit(fb, net, "cobblestone")
vol = volume_of(fb, 0, 0, 64, 64, 56, 120)
front = Frontage(vol, net)

t = front.threshold("hall")
check("frontage: the reserved threshold is handed to the pass", t is not None, str(t))

# a wall on the plot edge with its door on the reserved threshold, as instructed
pending = {}
site = [s for s in sites if s["id"] == "hall"][0]
dx, dz = {v: k for k, v in circulate.FACING.items()}[t["facing"]]
door = tuple(t["door"])
for k in range(-3, 4):                      # the wall the door sits in
    wx, wz = (door[0] + (k if dz else 0), door[2] + (k if dx else 0))
    for y in range(t["stand_y"], t["stand_y"] + 3):
        pending[(wx, y, wz)] = "stone_bricks"
pending[door] = f"oak_door[facing={t['facing']},half=lower]"
pending[(door[0], door[1] + 1, door[2])] = f"oak_door[facing={t['facing']},half=upper]"
r_ok = front.check_door(*door, pending=pending)
check("frontage: a door on the reserved threshold passes", r_ok["ok"], r_ok["reason"])

# A door on a plinth above its own approach
plinth_y = t["stand_y"] + 4
px, pz = site["x1"] + 1, site["z1"] + 1
bad = dict(pending)
for ax in range(px - 4, px + 1):
    for az in range(pz - 4, pz + 1):
        for y in range(56, plinth_y):
            bad[(ax, y, az)] = "stone_bricks"
for ax in range(px - 4, px + 1):
    for y in range(plinth_y, plinth_y + 3):
        bad[(ax, y, pz)] = "stone_bricks"
bad_door = (px - 2, plinth_y, pz)
bad[bad_door] = "oak_door[facing=south,half=lower]"
bad[(bad_door[0], bad_door[1] + 1, bad_door[2])] = "oak_door[facing=south,half=upper]"
r_bad = front.check_door(*bad_door, pending=bad)
check("frontage: a door up a plinth off the lane fails", not r_bad["ok"],
      r_bad["reason"])
check("frontage: the pass is told how far off it is",
      r_bad["nearest_lane"] is not None, str(r_bad["nearest_lane"]))


# --- the control ------------------------------------------------------------------- A
# suite that only ever passes proves nothing. Invert the stair facing.
print("\ncontrol: stair facing inverted on purpose")
h = terrain("mesa")
net = circulate.plan_network(h, 0, 0, sites_for("mesa"), max_step=3)
flip = {"north": "south", "south": "north", "east": "west", "west": "east"}
for rec in net.cells.values():
    if rec["face"]:
        rec["face"] = flip[rec["face"]]
fb = FakeBuilder(h, 0, 0)
circulate.emit(fb, net, "cobblestone")
vol = volume_of(fb, 0, 0, 64, 64, 56, 120)
nav = observe.Nav(vol)
want = [((x, z), nav.stance_near(x, z, r["y"] + 1, tol=2)) for (x, z), r in net.cells.items()]
seeds = [(c[0], c[1], s) for c, s in want if s is not None]
reach = nav.flood(seeds[:1], max_jumps=0)
off = [c for c, s in want if s is not None and (c[0], c[1], s) not in reach]
back = [b for b in observe.backwards_stairs(vol) if (b["pos"][0], b["pos"][2]) in net.cells]
check("control: inverted stairs break the network", len(off) > 20, f"{len(off)} off-network")
check("control: inverted stairs are caught as backwards", len(back) > 20,
      f"{len(back)} backwards stairs")




# --- the seam ----------------------------------------------------------------------
# The finishing pass, judged the same way: synthetic ground, the real solver, and then
# the invariants read back off what it decided. It has to round the lip of a cut face
# without regrading the hill behind it, and it has to leave owned ground alone.
print("\nseam finishing")
from ethoslm import finish  # noqa: E402

n = 48
g = np.full((n, n), 70)
g[:, 24:] = 82                                  # a 12-block face across the middle
lane = {(x, 22): 70 for x in range(4, 44)}      # a lane running along the foot of it
plots = [{"x0": 30, "z0": 30, "x1": 40, "z1": 40, "label": "hall"}]
targets = finish.plan_finish(lane, g, 0, 0, keep_out=plots, width=5, slope=1, max_move=3)

moved = {c: abs(y - int(g[c[0], c[1]])) for c, y in targets.items()}
check("seam: it moves something", bool(targets), f"{len(targets)} columns")
check("seam: no column moves further than max_move",
      max(moved.values(), default=0) <= 3, f"worst {max(moved.values(), default=0)}")
check("seam: it does not touch the lane itself",
      not (set(targets) & set(lane)), f"{len(set(targets) & set(lane))} lane cells")
check("seam: it does not touch ground somebody owns",
      not [c for c in targets if 30 <= c[0] <= 40 and 30 <= c[1] <= 40],
      f"{len([c for c in targets if 30 <= c[0] <= 40 and 30 <= c[1] <= 40])} in a plot")
check("seam: the hill behind the face is left alone",
      not [c for c in targets if c[1] > 30], "nothing beyond the band")

after = g.copy()
for (x, z), y in targets.items():
    after[x, z] = y
lip = [(after[x, 23] - after[x, 22]) for x in range(6, 42)]
check("seam: the lip of the cut face is rounded, not squared",
      max(lip) < int(g[0, 24] - g[0, 22]),
      f"face was {int(g[0, 24] - g[0, 22])} blocks, first step now {max(lip)}")
check("seam: the face is still a face, not a ramp",
      after[20, 26] >= 80, f"ground 2 cells back is still y={after[20, 26]}")

flat = finish.plan_finish({(x, 22): 70 for x in range(4, 44)}, np.full((n, n), 70),
                          0, 0, width=5, slope=1, max_move=3)
check("seam: flat ground beside a flat lane is left entirely alone",
      not flat, f"{len(flat)} columns moved on the level")

# ---------------------------------------------- A5. a wall, a gate, and one way in The
# thing a walled district is: an edge the network may not cross, a point that is the one
# place it may, and plots inside it that a person can still get to on foot. Until A5 the
# router knew one kind of part -- a building footprint to join up -- so a wall was
# either a plot (and the lanes went round it, which is a wall with no district in it) or
# nothing at all (and the lanes went straight through it).
print("\n# A5: circulation through a gate")

WALL = [[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]]     # a closed loop
GATE = {"kind": "point", "name": "gate", "type": "gate_tower",
        "at": [25, 10], "facing": "north", "passage": True}
TREE = {"parts": [{"kind": "district", "name": "keep", "children": [
    {"kind": "edge", "name": "wall", "type": "wall", "path": WALL, "width": 1},
    GATE,
    {"kind": "plot", "name": "inner_hall", "type": "hall",
     "x0": 16, "z0": 16, "x1": 24, "z1": 24},
    {"kind": "plot", "name": "inner_row", "type": "townhouse",
     "x0": 28, "z0": 28, "x1": 36, "z1": 36},
    {"kind": "area", "name": "market", "type": "square",
     "x0": 16, "z0": 28, "x1": 24, "z1": 36},
    {"kind": "plot", "name": "outer_barn", "type": "workshop",
     "x0": 44, "z0": 20, "x1": 52, "z1": 28},
]}]}

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import pipeline as _pipeline  # noqa: E402

parts = _pipeline.plan_parts(TREE)
check("a5: the tree flattens to six leaves inside one district",
      [p["name"] for p in parts] == ["wall", "gate", "inner_hall", "inner_row",
                                     "market", "outer_barn"]
      and all(p["in"] == ["keep"] for p in parts),
      f"{[p['kind'] for p in parts]}")

routing = circulate.parts_to_routing(parts, passage={"gate_tower"})
wall_cells = routing["obstacles"] | routing["passable"]
check("a5: the wall is an obstacle and the gate is the hole in it",
      len(routing["sites"]) == 5 and len(routing["passable"]) == 3
      and (25, 10) in routing["passable"] and (24, 10) in routing["obstacles"],
      f"{len(routing['sites'])} sites, {len(routing['obstacles'])} wall columns, "
      f"{sorted(routing['passable'])} passable")

n = 64
ground = np.full((n, n), 70, dtype=np.int32)
avoid = np.zeros((n, n))
for (wx, wz) in routing["obstacles"]:
    avoid[wx, wz] = np.inf
net5 = circulate.plan_network(ground, 0, 0, routing["sites"], max_step=3,
                             avoid_extra=avoid, passable=routing["passable"])

crossings = sorted(set(net5.cells) & (wall_cells - routing["passable"]))
gate_used = sorted(set(net5.cells) & routing["passable"])
check("a5: the lane crosses the wall nowhere but at the gate",
      not crossings and gate_used,
      f"{len(crossings)} cells of lane on the wall, {len(gate_used)} at the gate")

# ...and it is one network, so everything inside is reachable from everything outside
walk = circulate.walk_check(net5)
check("a5: it is still one walkable network, inside and out",
      walk["unreachable_n"] == 0 and walk["reached"] == walk["cells"],
      f"{walk['reached']}/{walk['cells']} lane cells reachable, "
      f"{walk['unreachable_n']} not")
inside = [t for t in net5.thresholds if t.id in ("inner_hall", "inner_row", "market")]
check("a5: every part inside the wall got a doorstep off that one network",
      len(inside) == 3, f"{sorted(t.id for t in net5.thresholds)}")

# and the discrimination: with the gate shut, the district is sealed and the router says
# so rather than routing through the wall anyway
shut = circulate.plan_network(ground, 0, 0, routing["sites"], max_step=3,
                              avoid_extra=avoid, passable=set())
sealed_walk = circulate.walk_check(shut)
check("a5: with no gate the same wall is a sealed district, and it shows",
      sealed_walk["unreachable_n"] > 0
      or not (set(shut.cells) & (wall_cells - routing["passable"])) and
      len([t for t in shut.thresholds]) < len(net5.thresholds),
      f"{sealed_walk['unreachable_n']} unreachable lane cells, "
      f"{len(shut.thresholds)} thresholds against {len(net5.thresholds)}")


# the arterials

# A place of two districts inside one wall with one gate, and nothing else drawn but the
# defining parts. This is the smallest thing that can exhibit the defect A4 exists for:
# with no road planned first, what joins one district to the other is whatever the lane
# router happens to lay between two sets of plots it sees at the same instant, and the
# gate has no road leading to it until everything else exists.
A4_WALL = [[8, 8], [104, 8], [104, 104], [8, 104], [8, 8]]
A4_GATE = {"kind": "point", "name": "south_gate", "type": "gate_tower",
           "at": [56, 104], "facing": "south", "passage": True}
A4_PLACE = {
    "centre": "market",
    "parts": [
        {"kind": "edge", "name": "town_wall", "type": "wall", "defines": "wall",
         "path": A4_WALL, "width": 3},
        A4_GATE,
        {"kind": "area", "name": "market", "type": "square",
         "x0": 50, "z0": 50, "x1": 62, "z1": 62},
        {"kind": "plot", "name": "keep", "type": "keep",
         "x0": 20, "z0": 20, "x1": 30, "z1": 30},
    ],
    "districts": [
        {"name": "north_quarter", "x0": 14, "z0": 36, "x1": 46, "z1": 68,
         "structures": 4},
        {"name": "east_quarter", "x0": 66, "z0": 36, "x1": 98, "z1": 68,
         "structures": 4},
    ],
}


def _a4_decls():
    return {"wall": {"kind": "edge", "passage": False},
            "gate_tower": {"kind": "point", "passage": True},
            "square": {"kind": "area", "passage": False},
            "keep": {"kind": "plot", "passage": False}}


from ethoslm import placeplan as _placeplan  # noqa: E402

_a4_ground = np.full((112, 112), 70, dtype=np.int32)
_a4_net, _a4_nodes = _placeplan.plan_arterials(A4_PLACE, _a4_decls(), _a4_ground, 0, 0)
_a4_rec = _placeplan.arterial_record(_a4_net, _a4_nodes, A4_PLACE)
A4_PLACE["arterials"] = _a4_rec

check("a4: the arterials join every district and every gate, before a plot exists",
      _a4_net is not None and sorted(_a4_rec["nodes"]) ==
      ["east_quarter", "north_quarter", "south_gate"],
      f"{len(_a4_rec['cells'])} columns over {sorted(_a4_rec['nodes'])}")

check("a4: every district is joined to one",
      all(_a4_rec["joins"][d["name"]] for d in A4_PLACE["districts"]),
      ", ".join(f"{d['name']}: {len(_a4_rec['joins'][d['name']])} columns"
                for d in A4_PLACE["districts"]))

_a4_fails = _placeplan.arterial_failures(A4_PLACE, _a4_rec, _a4_decls())
check("a4: no arterial crosses a plot, and none crosses the wall but at its gate",
      not _a4_fails, "; ".join(f["why"][:70] for f in _a4_fails))

# ...and the discrimination: a district the road cannot reach is named, and so is a plot
# drawn on top of one.
_a4_stranded = dict(A4_PLACE)
_a4_stranded = {**A4_PLACE, "districts": A4_PLACE["districts"] + [
    {"name": "unreached", "x0": 200, "z0": 200, "x1": 232, "z1": 232}]}
_a4_bad = _placeplan.arterial_failures(_a4_stranded, _a4_rec, _a4_decls())
check("a4: a district no arterial reaches is refused by name",
      [f["part"] for f in _a4_bad] == ["unreached"],
      "; ".join(f"{f['part']}: {f['why'][:60]}" for f in _a4_bad))

_a4_on_road = sorted(set(map(tuple, _a4_rec["cells"])))[len(_a4_rec["cells"]) // 2]
_a4_over = {**A4_PLACE, "parts": A4_PLACE["parts"] + [
    {"kind": "plot", "name": "squatter", "type": "keep",
     "x0": _a4_on_road[0] - 1, "z0": _a4_on_road[1] - 1,
     "x1": _a4_on_road[0] + 1, "z1": _a4_on_road[1] + 1}]}
_a4_blocked = _placeplan.arterial_failures(_a4_over, _a4_rec, _a4_decls())
check("a4: a plot standing on the road is refused by name",
      [f["part"] for f in _a4_blocked] == ["squatter"],
      "; ".join(f"{f['part']}: {f['why'][:60]}" for f in _a4_blocked))

# --- and the lanes, routed second, join the road rather than running beside it
_a4_plots = [{"kind": "plot", "name": f"n{i}", "type": "cottage",
              "x0": 18 + 8 * i, "z0": 40, "x1": 24 + 8 * i, "z1": 46}
             for i in range(3)]
_a4_plots += [{"kind": "plot", "name": f"e{i}", "type": "cottage",
               "x0": 70 + 8 * i, "z0": 40, "x1": 76 + 8 * i, "z1": 46}
              for i in range(3)]
#: One site **outside** the wall, so the lane pass has somewhere to arrive from. Without
#: it the network never leaves the enclosure and "reachable from outside the gate" is a
#: question about an empty set.
_a4_plots += [{"kind": "plot", "name": "arrival", "type": "cottage",
               "x0": 52, "z0": 107, "x1": 60, "z1": 110}]
_a4_leaves = _placeplan.pipeline.plan_parts(
    {"parts": A4_PLACE["parts"] + _a4_plots})
_a4_routing = circulate.parts_to_routing(_a4_leaves, passage={"gate_tower"})
_a4_avoid = np.zeros((112, 112))
for (wx, wz) in _a4_routing["obstacles"]:
    _a4_avoid[wx, wz] = np.inf
_a4_cells = {(int(x), int(z)): {"y": 70, "rank": 0, "face": None}
             for x, z in _a4_rec["cells"]}
_a4_full = circulate.plan_network(_a4_ground, 0, 0, _a4_routing["sites"], max_step=3,
                                  avoid_extra=_a4_avoid,
                                  passable=_a4_routing["passable"],
                                  arterial=_a4_cells)
_a4_walk = circulate.walk_check(_a4_full)
check("a4: the lanes come back one walkable network with the road inside it",
      _a4_walk["unreachable_n"] == 0
      and set(_a4_cells) <= set(_a4_full.cells),
      f"{_a4_walk['reached']}/{_a4_walk['cells']} lane cells reachable, "
      f"{len(set(_a4_cells) - set(_a4_full.cells))} arterial columns lost")

_a4_served = {t.id for t in _a4_full.thresholds}
check("a4: every plot in both districts has a doorstep off it, and so does the gate",
      {p["name"] for p in _a4_plots} <= _a4_served | {"arrival"}
      and "south_gate" in _a4_served,
      f"{len(_a4_served)} of {len(_a4_routing['sites'])} sites served: "
      f"{sorted(_a4_served - {p['name'] for p in _a4_plots})}")

# ...and every plot is reachable on foot from outside the gate, which is the whole ask
_a4_out = [c for c in _a4_full.cells if c[1] > 104]
_a4_seen, _a4_stack = set(_a4_out), list(_a4_out)
while _a4_stack:
    _c = _a4_stack.pop()
    for _d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _q = (_c[0] + _d[0], _c[1] + _d[1])
        if _q in _a4_full.cells and _q not in _a4_seen \
                and abs(_a4_full.cells[_q]["y"] - _a4_full.cells[_c]["y"]) <= 1:
            _a4_seen.add(_q)
            _a4_stack.append(_q)
_a4_unreached = [t.id for t in _a4_full.thresholds
                 if (t.x, t.z) not in _a4_seen]
check("a4: every plot is reachable on foot from outside the gate",
      _a4_out and not _a4_unreached,
      f"{len(_a4_seen)} of {len(_a4_full.cells)} lane cells reached from "
      f"{len(_a4_out)} columns outside the wall; unreached: {_a4_unreached}")


# it is how a ring is round rather than square -- and `parts_to_routing` read every
# segment as axial. A chamfer came out as a line running due east from one end of it,
# twice as long as the chamfer and nowhere near it, so an octagon's four cut corners
# were not obstacles at all. The router is 4-connected and a diagonal line of cells is
# 8-connected, so the chamfer's own columns are all it takes to seal it.
_oct_c, _oct_h = 56, 40
_oct_path = _placeplan._octagon_path(_oct_c, _oct_c, _oct_h, max_run=64)
_oct_diag = [(a, b) for a, b in zip(_oct_path, _oct_path[1:])
             if a[0] != b[0] and a[1] != b[1]]
_oct_parts = [
    {"kind": "edge", "name": "ring", "type": "wall", "path": _oct_path, "width": 1},
    {"kind": "point", "name": "ring_gate", "type": "gate_tower",
     "at": [_oct_c, _oct_c - _oct_h], "facing": "north", "passage": True},
    {"kind": "plot", "name": "keep", "type": "hall",
     "x0": _oct_c - 4, "z0": _oct_c - 4, "x1": _oct_c + 4, "z1": _oct_c + 4},
    {"kind": "plot", "name": "outside", "type": "cottage",
     "x0": _oct_c - 4, "z0": _oct_c + _oct_h + 6, "x1": _oct_c + 4,
     "z1": _oct_c + _oct_h + 12},
]
_oct_routing = circulate.parts_to_routing(_oct_parts, passage={"gate_tower"})
_oct_want = {(int(a[0]) + (1 if b[0] > a[0] else -1) * i,
              int(a[1]) + (1 if b[1] > a[1] else -1) * i)
             for a, b in _oct_diag for i in range(abs(b[0] - a[0]) + 1)}
_oct_missing = sorted(_oct_want - _oct_routing["obstacles"])
check("a5: every column of an octagon's four chamfers is an obstacle",
      len(_oct_diag) == 4 and _oct_want and not _oct_missing,
      f"{len(_oct_diag)} diagonal segments, {len(_oct_want)} chamfer columns, "
      f"{len(_oct_missing)} of them missing from obstacles")

_oct_ground = np.full((128, 128), 70, dtype=np.int32)
_oct_avoid = np.zeros((128, 128))
for (wx, wz) in _oct_routing["obstacles"]:
    _oct_avoid[wx, wz] = np.inf
_oct_net = circulate.plan_network(_oct_ground, 0, 0, _oct_routing["sites"], max_step=3,
                                  avoid_extra=_oct_avoid,
                                  passable=_oct_routing["passable"])
_oct_wall = _oct_routing["obstacles"] | _oct_routing["passable"]
_oct_cross = sorted(set(_oct_net.cells) & (_oct_wall - _oct_routing["passable"]))
_oct_on_chamfer = [c for c in _oct_cross if c in _oct_want]
check("a5: the lane to the keep crosses the ring at its gate and not through a chamfer",
      not _oct_cross and set(_oct_net.cells) & _oct_routing["passable"],
      f"{len(_oct_cross)} cells of lane on the ring, {len(_oct_on_chamfer)} of them "
      f"on a chamfer")


print(f"\n{CASES - len(FAILURES)}/{CASES} cases pass")
if FAILURES:
    print("\n".join(FAILURES))
sys.exit(1 if FAILURES else 0)
