"The walk-only interior number, for every standing settlement. Read-only, no server.\n\nIs this new, or has every town had it? That is the only question here, and it is\nanswerable off the cached volumes with no rebuild: three towns, one number each.\n\nTwo measures, deliberately both, because they answer different questions and the\ndifference between them is itself informative:\n\n    from the door    what `Context.interior_walk` reports and what E011 fires on:\n                     walking only, from this building's own doorway, how much of each\n                     room's floor can you reach. This is the builder's question.\n    from outdoors    `flood(perimeter, max_jumps=0)`: walking only, from anywhere\n                     outside the town. This is the town's question, and it is the\n                     measure the defect was first reproduced with.\n\nAlongside each, the same fractions computed the way the linter has always computed\nreachability -- `max_jumps=None`, unlimited jumping -- because the gap between the two\ncolumns *is* the blind spot, expressed as a number."
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ethoslm import lint, observe, offline, settlement  # noqa: E402
from ethoslm.circulate import Network  # noqa: E402

ROOT = settlement.ROOT
TOWNS = ["site_a", "site_b", "site_c"]


def load(name):
    state = os.path.join(ROOT, "out", name)
    vol = offline.load_volume(os.path.join(state, "world_built.npz"))
    # A1: the registry carrying the floor level each part was sited at, so a room under
    # a plot is not read as the floor of the building above it.
    plots = settlement.registry_with_floors(state)
    netp = os.path.join(state, "network.json")
    net = Network.load(netp) if os.path.exists(netp) else None
    site = json.load(open(os.path.join(state, "site.json")))
    X, Z, S = site["origin"][0], site["origin"][1], site["size"]
    return lint.Context.build(vol, plots=plots, network=net,
                              region=(X, Z, X + S, Z + S)), net


def measure(name, show_rooms=False, context=None):
    """`context` is `(lint.Context, Network)` already built over this town's volume --
    A4, so that a readout builds one and not five. With none, this builds its own, which
    is what the command line does."""
    t0 = time.perf_counter()
    ctx, net = context if context is not None else load(name)
    rows = ctx.interior_walk()
    on_plot = [r for r in ctx.rooms if r.get("plot")]

    # on a cut volume the perimeter reaches nothing -- and this, the third reader of the
    # same question, never got it. It cost nothing for six rounds because **no wall this
    # project built was ever closed**. The town read 0.0% walkable and 94.7% with a jump
    # allowed, which is not a town nobody can enter, it is a seed that never gets
    # through the gate. Seeded from the lane too it reads 88.2%. Both readings are on
    # the record and **both miss the bar of 90**, so nothing here turns a miss into a
    # pass.
    walk = set(ctx.nav.flood(
        list(ctx.nav.perimeter_seeds(inset=2, step=3)) + lint.lane_stances(ctx.nav, net),
        max_jumps=0))
    jump = set(ctx.from_anywhere)
    cells = wo = jo = 0
    for r in on_plot:
        # the room's floor, as `observe.floor_stances` defines it.
        st = set(map(tuple, r["floor"]))
        cells += len(st)
        wo += len(st & walk)
        jo += len(st & jump)

    door_cells = sum(r["cells"] for r in rows)
    door_walk = sum(r["walkable"] for r in rows)
    rep = lint.lint(ctx, only={"E011", "W011"})
    out = {
        "settlement": name,
        # Which ruler this was read with, beside the number, always. A standing town is
        # read off `world_built.npz` and there is no record of what its builder laid as
        # furniture, so `fitting_registry` is False and every solid inside it is floor
        # -- see `observe.floor_stances`. A figure read with the register and one read
        # without it are not comparable and the readout says so rather than differencing
        # them.
        "walk_model": observe.WALK_MODEL,
        "fitting_registry": ctx.fittings is not None,
        "stair_clear": observe.STAIR_CLEAR,
        "rooms_on_plots": len(on_plot),
        "floor_cells": cells,
        "from_outdoors_walking": wo,
        "from_outdoors_walking_pct": round(100 * wo / cells, 1) if cells else None,
        "from_outdoors_jumping": jo,
        "from_outdoors_jumping_pct": round(100 * jo / cells, 1) if cells else None,
        "rooms_zero_walkable_from_outdoors":
            sum(1 for r in on_plot if not set(map(tuple, r["floor"])) & walk),
        "from_own_door_cells": door_cells,
        "from_own_door_walking": door_walk,
        "from_own_door_walking_pct":
            round(100 * door_walk / door_cells, 1) if door_cells else None,
        "E011": len([f for f in rep.findings if f.code == "E011"]),
        "W011": len([f for f in rep.findings if f.code == "W011"]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if show_rooms:
        out["rooms"] = sorted(rows, key=lambda r: r["fraction"])
    return out, rep


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    towns = args or TOWNS
    show = "--rooms" in sys.argv
    results = []
    for name in towns:
        r, rep = measure(name, show)
        results.append(r)
        print(f"\n=== {name}  ({r['seconds']}s)")
        print(f"  {r['rooms_on_plots']} rooms on plots, {r['floor_cells']} floor cells")
        print(f"  walking, from anywhere outdoors : "
              f"{r['from_outdoors_walking']:>5}/{r['floor_cells']} = "
              f"{r['from_outdoors_walking_pct']:>5}%   "
              f"({r['rooms_zero_walkable_from_outdoors']} rooms at 0%)")
        print(f"  jumping allowed, as the linter has always measured it: "
              f"{r['from_outdoors_jumping']:>5}/{r['floor_cells']} = "
              f"{r['from_outdoors_jumping_pct']:>5}%")
        print(f"  walking, from each building's own door: "
              f"{r['from_own_door_walking']:>5}/{r['from_own_door_cells']} = "
              f"{r['from_own_door_walking_pct']:>5}%")
        print(f"  new check: {r['E011']} errors (E011), {r['W011']} warnings (W011)")
        if show:
            for x in r["rooms"]:
                print(f"      {x['plot']:<18} {x['walkable']:>4}/{x['cells']:<4} "
                      f"{x['fraction']:>6.0%}  at {tuple(x['bbox'][:3])}")
    path = os.path.join(ROOT, "out", "walk_fraction.json")
    json.dump(results, open(path, "w"), indent=1)
    print(f"\n{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
