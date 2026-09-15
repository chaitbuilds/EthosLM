"""The terrain fixture bank, sampled by rule from the caches.

    $PY scripts/terrain_bank.py            # write rounds/terrain-bank.json
    $PY scripts/terrain_bank.py --check    # regenerate and compare, byte for byte

Rounds 12, 13 and 14 each spent a session and a million tokens to find one or two
library defects about the ground -- water counted as ground, decks that drain a lake,
propagules that float -- because the only instrument that ever met new ground was a
model-authored type instantiated on a new site. This is the cheap instrument: every
piece of ground already on disk, tiled, measured, and reduced by rule to thirty-six
fixtures a hand-written reference type can be stood on in minutes.

The rule, end to end:

  - tile each of the six cached sites into 40x40 patches on a 32-block stride (64 a
    site, 384 in all);
  - measure each patch: relief, water share, tree cover, gravity-block share, and
    whether the round's own circulation network runs through it;
  - keep the patches no lane runs through, so the only lane on a fixture is the one
    this file lays;
  - take six per site by six total orders -- flattest-driest, steepest, wettest, most
    treed, most gravity, and one by a fixed seed -- each from the patches the earlier
    rules have not taken and the router can lay a lane into, so six rules give six
    different patches and the bank is 36;
  - place a 13x15 plot in each by `flattest_rect`, and route a lane to it from a gate
    on the patch edge nearest it, using `circulate.plan_network` -- the round's own
    router -- so the reserved doorstep is where a circulation pass would put it.

Every tie-break ends in the patch's own coordinates, so there is nothing here to take
on trust and `--check` is the cheapest statement of that.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import circulate, observe, offline                         # noqa: E402
from ethoslm.buildlib import Builder                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "rounds", "terrain-bank.json")

#: Every cached world this project has, and the file the *pre-build* ground is in. the
#: other five have one npz and it is the ground as it was found.
SITES = (("site_a", "world.npz"), ("site_b", "world.npz"),
         ("site_c", "world_prebuild.npz"), ("site_d", "world.npz"),
         ("site_e", "world.npz"), ("site_f", "world.npz"))

PATCH = 40          # a patch is this square
STRIDE = 32         # ...and they overlap by eight, so nothing falls between two
PLOT = (13, 15)     # the plot placed in one, w x d
MARGIN = 4          # ...never closer than this to the patch edge, which is the lane

#: The id of the arrival point at the patch edge: where the town is, as far as one
#: fixture is concerned. Not a structure, and its own doorstep is dropped.
GATE = "lane_head"

#: The blocks that fall when what is under them goes. A pad cut into a bank of these is
#: a pad with a landslide over it, and no fixture in rounds 13 or 14 had any.
GRAVITY = ("sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel")

#: What counts as a tree for `tree_pct`: the column's top is wood or leaf.
CANOPY = ("_log", "_wood", "_leaves", "_stem", "_hyphae")

#: The six rules, in the order they take. Each is a key minimised over the patches no
#: earlier rule has taken; `seeded` is the one that is not a superlative, so that the
#: bank contains ordinary ground as well as the extremes of it.
RULES = ("flattest_driest", "steepest", "wettest", "most_treed", "most_gravity",
         "seeded")


# ----------------------------------------------------------------- measuring

def _column_tables(vol: observe.Volume):
    """(ground height, wet, top-of-column block name, ground block name) per column."""
    h, wet = observe.ground_heights(vol)
    names = [s.split("[")[0] for s in vol.palette]
    air = np.array([n in ("air", "cave_air", "void_air") for n in names], bool)
    solid = ~air[vol.codes]
    idx = solid.shape[1] - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    top_code = np.take_along_axis(vol.codes, idx[:, None, :], axis=1)[:, 0, :]
    top_code = np.where(solid.any(axis=1), top_code, 0)
    ly = np.clip(h - vol.y0, 0, vol.codes.shape[1] - 1)
    ground_code = np.take_along_axis(vol.codes, ly[:, None, :], axis=1)[:, 0, :]
    return h, wet, top_code, ground_code, names


def _patches(name: str, cache: str) -> tuple:
    """Every 40x40 patch of one cached site, measured. (volume, [patch, ...])."""
    vol = offline.load_volume(offline.world_cache(name, cache))
    h, wet, top_code, ground_code, names = _column_tables(vol)
    canopy = np.array([any(v in n for v in CANOPY) for n in names], bool)
    gravity = np.array([n in GRAVITY or n.endswith("_concrete_powder")
                        for n in names], bool)
    net = circulate.Network.load(os.path.join(ROOT, "out", name, "network.json"))
    lanes = set(net.cells) if net else set()
    sx, _sy, sz = vol.codes.shape
    out = []
    for i in range(0, sx - PATCH + 1, STRIDE):
        for j in range(0, sz - PATCH + 1, STRIDE):
            x0, z0 = vol.x0 + i, vol.z0 + j
            sl = (slice(i, i + PATCH), slice(j, j + PATCH))
            hs, ws = h[sl], wet[sl]
            out.append({
                "round": name, "x0": int(x0), "z0": int(z0), "size": PATCH,
                "relief": int(hs.max() - hs.min()),
                "water_pct": round(100 * float(ws.mean()), 1),
                "tree_pct": round(100 * float(canopy[top_code[sl]].mean()), 1),
                "gravity_pct": round(100 * float(gravity[ground_code[sl]].mean()), 1),
                "lane_free": not any(x0 <= lx < x0 + PATCH and z0 <= lz < z0 + PATCH
                                     for (lx, lz) in lanes),
            })
    return vol, out


# ------------------------------------------------------------------ choosing

def _seed(name: str) -> int:
    """A stable integer from a site's name. `hash()` is salted per process and
    `random.seed(str)` has changed its digest before; this cannot."""
    v = 2166136261
    for ch in name.encode():
        v = ((v ^ ch) * 16777619) & 0xFFFFFFFF
    return v


class Unroutable(Exception):
    """This patch cannot carry a fixture: no lane reaches the plot in it."""


def _ranked(pool: list, rule: str, round_name: str) -> list:
    """Every patch of a site in the order this rule wants them."""
    keys = {
        "flattest_driest": lambda p: (p["water_pct"], p["relief"], p["x0"], p["z0"]),
        "steepest": lambda p: (-p["relief"], p["water_pct"], p["x0"], p["z0"]),
        "wettest": lambda p: (-p["water_pct"], -p["relief"], p["x0"], p["z0"]),
        "most_treed": lambda p: (-p["tree_pct"], p["relief"], p["x0"], p["z0"]),
        "most_gravity": lambda p: (-p["gravity_pct"], p["relief"], p["x0"], p["z0"]),
    }
    if rule != "seeded":
        return sorted(pool, key=keys[rule])
    out = sorted(pool, key=lambda p: (p["x0"], p["z0"]))
    random.Random(_seed(f"{round_name}/terrain-bank")).shuffle(out)
    return out


# ---------------------------------------------------------- plot, and the lane

def _plot(b: Builder, patch: dict) -> dict:
    """The 13x15 the library will be asked to site, by `flattest_rect` in the patch.

        Never closer than `MARGIN` to a patch edge, because the edge is where the lane
        goes and a building laid over its own lane is a fixture asking the wrong question.
        
    """
    w, d = PLOT
    x0, z0 = patch["x0"] + MARGIN, patch["z0"] + MARGIN
    x1 = patch["x0"] + patch["size"] - 1 - MARGIN
    z1 = patch["z0"] + patch["size"] - 1 - MARGIN
    got = b.flattest_rect(x0, z0, x1, z1, w, d, step=1)
    if got is None:
        raise SystemExit(f"{patch['round']} ({x0},{z0}): no {w}x{d} fits the patch")
    return {"label": f"{patch['round']}_{patch['rule']}",
            "x0": int(got[0]), "z0": int(got[1]),
            "x1": int(got[0]) + w - 1, "z1": int(got[1]) + d - 1}


def _nearest_edge(patch: dict, plot: dict) -> str:
    """Which side of the patch the lane runs along: the one the plot is nearest."""
    px1 = patch["x0"] + patch["size"] - 1
    pz1 = patch["z0"] + patch["size"] - 1
    d = (("west", plot["x0"] - patch["x0"]), ("east", px1 - plot["x1"]),
         ("north", plot["z0"] - patch["z0"]), ("south", pz1 - plot["z1"]))
    return min(d, key=lambda kv: (kv[1], kv[0]))[0]


def _lane(vol: observe.Volume, h, wet, patch: dict, plot: dict) -> dict:
    """The lane, routed to the plot by the circulation pass itself.

        Not a line drawn along the patch edge. A round does not hand a builder a street
        forty blocks away and a doorstep at whatever height that street happens to be at:
        `circulate.plan_network` routes a lane **to** each plot, over the ground, at a
        slope a person can walk, and reserves the few blocks where that lane meets the
        doorway. Doing anything less here produced a fixture whose reserved doorstep was
        fifteen blocks below the plot it served, and `site()` -- which is right to follow
        the doorstep -- dutifully quarried a fifteen-block pit out of a dune to reach it.

        So the fixture is a settlement of one: the plot, and a gate at the middle of the
        patch edge nearest it, which is the lane the town arrives on. The router lays the
        way between them and picks which side of the plot the town arrives at; the gate is
        an arrival point rather than a structure, so its own threshold is dropped and only
        the plot's is kept.

        Returns `Network.to_json()`'s shape, so what the bank stores is a circulation
        network and `test_ground.py` hands the library the object a round hands it.
        
    """
    size = patch["size"]
    i0, j0 = patch["x0"] - vol.x0, patch["z0"] - vol.z0
    hp = h[i0:i0 + size, j0:j0 + size]
    wp = wet[i0:i0 + size, j0:j0 + size]
    edge = _nearest_edge(patch, plot)
    px1, pz1 = patch["x0"] + size - 1, patch["z0"] + size - 1
    cx, cz = (plot["x0"] + plot["x1"]) // 2, (plot["z0"] + plot["z1"]) // 2
    gx, gz = {"west": (patch["x0"] + 2, cz), "east": (px1 - 2, cz),
              "north": (cx, patch["z0"] + 2), "south": (cx, pz1 - 2)}[edge]
    sites = [{"id": plot["label"], "x0": plot["x0"], "z0": plot["z0"],
              "x1": plot["x1"], "z1": plot["z1"]},
             {"id": GATE, "x0": gx - 1, "z0": gz - 1, "x1": gx + 1, "z1": gz + 1}]
    net = circulate.plan_network(hp, patch["x0"], patch["z0"], sites,
                                 centre=plot["label"], max_step=3,
                                 avoid_extra=np.where(wp, 40.0, 0.0))
    if net.notes.get("stranded") or not net.cells:
        raise Unroutable(f"{plot['label']}: {net.notes.get('stranded')} unreachable")
    net.thresholds = [t for t in net.thresholds if t.id == plot["label"]]
    if not net.thresholds:
        raise Unroutable(f"{plot['label']}: the router reserved no doorstep")
    wc = circulate.walk_check(net)
    if wc["unreachable_n"]:
        raise Unroutable(f"{plot['label']}: {wc['unreachable_n']} lane cells of "
                         f"{wc['cells']} cannot be walked to in the plan")
    d = net.to_json()
    d["notes"] = {"material": "cobblestone", "gate": [int(gx), int(gz)], "edge": edge}
    d["edge"] = edge
    return d


# ------------------------------------------------------------------- the bank

def bank() -> dict:
    """The whole file, as a dict. One command, no arguments, the same answer."""
    fixtures = []
    for name, cache in SITES:
        vol, patches = _patches(name, cache)
        h, wet = observe.ground_heights(vol)
        b = Builder(offline.OfflineSite(vol))
        b._vol = vol
        pool = [p for p in patches if p["lane_free"]]
        taken: set = set()
        for rule in RULES:
            for patch in _ranked(pool, rule, name):
                if (patch["x0"], patch["z0"]) in taken:
                    continue
                plot = _plot(b, dict(patch, rule=rule))
                try:
                    lane = _lane(vol, h, wet, patch, plot)
                except Unroutable:
                    continue
                taken.add((patch["x0"], patch["z0"]))
                fixtures.append({**patch, "rule": rule, "cache": cache,
                                 "plot": plot, "lane": lane})
                break
            else:
                raise SystemExit(f"{name}: no routable patch left for {rule!r}")
    return {"generated_by": "scripts/terrain_bank.py",
            "patch": PATCH, "stride": STRIDE, "plot": list(PLOT), "margin": MARGIN,
            "rules": list(RULES), "sites": [s for s, _c in SITES],
            "fixtures": fixtures}


def text() -> str:
    return json.dumps(bank(), indent=1, sort_keys=True) + "\n"


def load() -> dict:
    """The bank as written. Every reader goes through here."""
    return json.load(open(OUT))


def main(argv: list) -> int:
    if "--check" in argv:
        if not os.path.exists(OUT):
            print(f"FAIL {os.path.relpath(OUT, ROOT)} has not been written")
            return 1
        want, got = open(OUT).read(), text()
        if want != got:
            print(f"FAIL {os.path.relpath(OUT, ROOT)} does not reproduce: "
                  f"{len(want)} bytes on disk, {len(got)} regenerated")
            return 1
        d = json.loads(got)
        print(f"ok   {os.path.relpath(OUT, ROOT)} reproduces byte for byte: "
              f"{len(d['fixtures'])} fixtures over {len(d['sites'])} sites")
        return 0
    s = text()
    open(OUT, "w").write(s)
    d = json.loads(s)
    for f in d["fixtures"]:
        print(f"{f['plot']['label']:26s} relief {f['relief']:3d}  water "
              f"{f['water_pct']:5.1f}%  trees {f['tree_pct']:5.1f}%  gravity "
              f"{f['gravity_pct']:5.1f}%  lane {f['lane']['edge']}")
    print(f"\n{len(d['fixtures'])} fixtures -> {os.path.relpath(OUT, ROOT)} "
          f"({len(s)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
