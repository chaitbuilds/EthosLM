"""Sloped ground for a type's own checker: a small pad on a bank, a large one, and two
large ones side by side.

Three fixtures now, and each was a piece of ground nothing in this project had.

`slope_<w>_<d>` is the original: a 1:2 fall across the shorter axis of a pad near 9x11
-- a house on a hillside, which is what every fixture in the terrain bank happens to be.
The same idea at **the top of the type's declared band**, falling `LARGE_FALL` blocks
across the pad, because the terrain bank samples six real sites and none of them has a
large plot on a slope and `type_needs.py` swept a plane. A city gave `court_large` a
30x30 plot on 122 of relief and nine rooms came back with no walkable floor at sizes
both instruments called clean.

`slopepair_<w>_<d>` is the close-misses round's, and it is the one dimension every other
fixture in this project is missing: **there is one plot on all of them**. The terrain
bank's six sites cut one plot each, the needs sweep stands one part on a plane and one
on a bank, and both fixtures above are a single pad in an empty working area. A district
is not like that. a `court_large` at 23x23 with a `hall` at 23x15 five columns away on
the same cut terrace -- and until this, a type that reached into its neighbour's ground
could only find out in a city. Two pads at the top of the declared band, `PAIR_GAP`
apart, on the same bank, off one lane.
"""
from pathlib import Path
import json
import os

import numpy as np

from . import circulate, observe, offline, stages
from .buildlib import Builder

#: How far the ground falls across a **large** sloped pad. A fall and not a gradient,
#: for `type_needs.SLOPE_FALL`'s reason -- a platform of any size cuts the same amount
#: of face -- and eight because it is the most `Builder.site()` takes across every size
#: swept without coming back with a smaller pad than it was asked for.
LARGE_FALL = 8

#: How many free columns stand between the two plots of `slopepair_<w>_<d>`. Two, which
#: is **the tightest the plan validator admits**: `stages_plan._validate` grows one
#: rectangle by the larger of the two declared clearances and refuses an overlap, and
#: every committed plot type declares `clearance: 2`, so a gap of one is refused and a
#: gap of two is a plan a district may hand the library. A fixture whose neighbours
#: stood further apart than the planner allows would be a fixture that cannot fail.
PAIR_GAP = 2


def _atomic(path, write) -> None:
    """Write a fixture file through a private temporary and rename it into place.

        Every worker that reads one of these fixtures **builds it again** -- the ground is
        arithmetic and the bytes are identical, so rebuilding is cheap and the cache is a
        convenience rather than a source of truth. What is not identical is the moment in
        between: a process that opened `plots.json` while another had it truncated read an
        empty file and raised `JSONDecodeError`, which is how a whole swept check run dies
        at eight jobs and survives at one. A rename is atomic on one filesystem; a truncate
        is not, and a checker that fails on how many processes it was given is not a checker.
        
    """
    # The suffix is kept: `np.savez_compressed` appends `.npz` to a path that has not
    # got one, and would write somewhere this call never renames.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _clean(lo: int, hi: int, ex: set, near: int) -> int:
    """The legal size nearest `near`, never one the sweep found broken, low side of a tie."""
    return min(range(lo, hi + 1), key=lambda v: (v in ex, abs(v - near), v))


def descriptors(declaration) -> list:
    """The sloped fixtures this type is checked on, without consulting any outcome.

        Three: a small pad on a steep bank, the largest pad the type declares on a bank that
        falls `LARGE_FALL` across it, and **two of that largest pad side by side** on one
        bank, `PAIR_GAP` apart. Where the band admits only one size the first two are the
        same fixture and the list is deduplicated, which is right -- there is one piece of
        ground to test -- but the pair is never a duplicate of either: what it tests is the
        neighbour.
        
    """
    needs = declaration.get("needs", {})
    a, b, c, d = needs.get("footprint", (3, 3, 64, 64))
    ex = set(needs.get("except") or ())
    # ...and never a size the sweep found broken: the nearest clean one inside the
    # envelope instead, and the low side of a tie, because these fixtures are about the
    # slope and not about the size.
    small = {"round": f"slope_{_clean(a, c, ex, 9)}_{_clean(b, d, ex, 11)}",
             "plot": "short_axis_slope",
             "rule": "1:2 across the shorter axis; pad clamped to NEEDS near 9x11"}
    big = {"round": f"slopebig_{_clean(a, c, ex, c)}_{_clean(b, d, ex, d)}",
           "plot": "short_axis_slope",
           "rule": f"{LARGE_FALL} blocks of fall across the pad, at the top of NEEDS"}
    out = [small] if big["round"].split("_", 1)[1] == small["round"].split("_", 1)[1] \
        else [small, big]
    # The neighbour. One descriptor per plot, because `_fixtures_for` and
    # `_check_type_job` select the plots of a fixture round by name and a fixture named
    # once would stand the type on one of the two and leave the other empty ground --
    # which is the fixture we already have.
    pair = f"slopepair_{_clean(a, c, ex, c)}_{_clean(b, d, ex, d)}"
    rule = (f"two pads {PAIR_GAP} apart on one bank falling {LARGE_FALL} across the "
            f"pair, at the top of NEEDS: the neighbour's ground")
    return out + [{"round": pair, "plot": "pair_uphill", "rule": rule},
                  {"round": pair, "plot": "pair_downhill", "rule": rule}]


def descriptor(declaration):
    """The small sloped fixture alone. Kept because it is what the record names."""
    return descriptors(declaration)[0]


def make(name, root):
    """Construct synthetic ground in memory and the small registry the checker reads."""
    from .pipeline import Round, OfflineBackend
    kind, width, depth = name.split("_")
    if kind == "slopepair":
        return _make_pair(name, root, int(width) + 4, int(depth) + 4)
    width, depth = int(width) + 4, int(depth) + 4
    size = max(width, depth) + 32
    x, z = np.indices((size, size))
    across = x if width <= depth else z
    if kind == "slope":
        h = 64 + (across - 12) // 2
    else:
        # A fall of `LARGE_FALL` across the pad itself, level above it and level below
        # it: what is under test is a large part on relief, and a gradient carried to
        # the edge of the working area would put the far end of its approach forty
        # blocks down a mountain and measure the fixture.
        run = max(1, min(width, depth) - 1)
        t = np.clip((across - 12) / run, 0.0, 1.0)
        h = 64 + np.rint(LARGE_FALL * (t - 1.0)).astype(int)
    ys = np.arange(40, 128)[None, :, None]
    codes = np.where(ys <= h[:, None, :], 1, 0).astype(np.uint16)
    vol = observe.Volume(0, 40, 0, codes, ["air", "stone"])
    state = Path(root) / "out" / "checker_fixtures" / name
    state.mkdir(parents=True, exist_ok=True)
    plot = dict(label="short_axis_slope", x0=12, z0=12,
                x1=11 + width, z1=11 + depth)
    _atomic(state / "plots.json",
            lambda p: p.write_text(json.dumps([plot], indent=2) + "\n"))
    # Route and pave the public approach before testing a type, just as the terrain bank
    # does. A reserved doorstep without its approach is an invalid fixture.
    if width <= depth:
        mid = (plot["z0"] + plot["z1"]) // 2
        gx, gz = 3, mid
    else:
        mid = (plot["x0"] + plot["x1"]) // 2
        gx, gz = mid, 3
    net = circulate.plan_network(h, 0, 0,
        [dict(plot, id=plot["label"]), dict(id="arrival", x0=gx-1, z0=gz-1,
                                          x1=gx+1, z1=gz+1)],
        centre=plot["label"], max_step=3)
    net.thresholds = [t for t in net.thresholds if t.id == plot["label"]]
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    circulate.emit(b, net, ground=h, x0=0, z0=0, lantern_every=0, post_every=0)
    b.resolve_steps()
    vol = stages.apply_pending(vol, b._pending)
    _atomic(state / "world.npz", lambda p: offline.save_volume(vol, str(p)))
    _atomic(state / "network.json", lambda p: net.save(str(p)))
    rnd = Round(name=name, state_dir=str(state), site={"origin": [0, 0], "size": size})
    be = OfflineBackend(rnd)
    be._vol = vol
    return rnd, be


def _make_pair(name, root, width, depth):
    """Two plots of the same size on one bank, `PAIR_GAP` apart, off one lane."""
    from .pipeline import Round, OfflineBackend
    span = 2 * depth + PAIR_GAP
    size = max(width, span) + 32
    _x, z = np.indices((size, size))
    run = max(1, span - 1)
    frac = np.clip((z - 12) / run, 0.0, 1.0)
    h = 64 + np.rint(LARGE_FALL * (frac - 1.0)).astype(int)
    ys = np.arange(40, 128)[None, :, None]
    codes = np.where(ys <= h[:, None, :], 1, 0).astype(np.uint16)
    vol = observe.Volume(0, 40, 0, codes, ["air", "stone"])
    state = Path(root) / "out" / "checker_fixtures" / name
    state.mkdir(parents=True, exist_ok=True)
    plots = [dict(label="pair_uphill", x0=12, z0=12,
                  x1=11 + width, z1=11 + depth),
             dict(label="pair_downhill", x0=12, z0=12 + depth + PAIR_GAP,
                  x1=11 + width, z1=11 + 2 * depth + PAIR_GAP)]
    _atomic(state / "plots.json",
            lambda p: p.write_text(json.dumps(plots, indent=2) + "\n"))
    # The lane comes in on the west side, level with the gap between the two, so neither
    # plot is the one the lane happens to end at.
    gx, gz = 3, (plots[0]["z1"] + plots[1]["z0"]) // 2
    net = circulate.plan_network(h, 0, 0,
        [dict(p, id=p["label"]) for p in plots]
        + [dict(id="arrival", x0=gx - 1, z0=gz - 1, x1=gx + 1, z1=gz + 1)],
        centre=plots[0]["label"], max_step=3)
    keep = {p["label"] for p in plots}
    net.thresholds = [t for t in net.thresholds if t.id in keep]
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    circulate.emit(b, net, ground=h, x0=0, z0=0, lantern_every=0, post_every=0)
    b.resolve_steps()
    vol = stages.apply_pending(vol, b._pending)
    _atomic(state / "world.npz", lambda p: offline.save_volume(vol, str(p)))
    _atomic(state / "network.json", lambda p: net.save(str(p)))
    rnd = Round(name=name, state_dir=str(state), site={"origin": [0, 0], "size": size})
    be = OfflineBackend(rnd)
    be._vol = vol
    return rnd, be
