"""What ground each committed type needs, measured.

    $PY scripts/type_needs.py            # sweep and write rounds/type-needs.json
    $PY scripts/type_needs.py --check    # re-sweep and assert it reproduces
    $PY scripts/type_needs.py townhouse  # one type, printed
    $PY scripts/type_needs.py --transcribe   # write the measured bands onto the files
    $PY scripts/type_needs.py --merge temple ring_gate   # re-sweep these, write their rows

`NEEDS` is a type's own declaration and A1 says every type file carries one. This is the
instrument that says whether a declaration is true, and it is the same shape of answer
as `scripts/terrain_bank.py`: one command, off nothing but the committed files, writing
a registered JSON that reproduces byte for byte.

**Synthetic ground on purpose.** `rounds/terrain-bank.json` and `scripts/test_ground.py`
ask whether the library can prepare ground; this asks the different question of what size
of prepared ground a type can use, and the two must not be confounded. So the world here
is made rather than sampled, the pad is cut to an exact size through `Builder.site()`, and
what varies is the size, the seed and the type's own declared parameters -- every
combination of them, which is what makes "the type is clean from 11 up" a statement
about the type rather than about the four plots its checker happened to run on.

A plane and a bank** (`GROUNDS`, `SLOPE_FALL`). A size passes only when every instance
of it passes on both. The plane alone was the whole sweep for four rounds and it made
every declared band a statement about level ground, which is not the ground a city is
on. See `GROUNDS`.

An instance **passes** when `build()` stands (returns no refusal, raises nothing) and
the build family of the linter reports no error on its plot. A run is an unbroken
stretch of sizes every instance of which passes, and a type can have more than one with
a broken stretch between them -- `square` is clean at 5 and again from 12 up,
`gate_tower` to 6 and again from 11, `palace` at 8-12 and again at 19-28.

the one holding the fixture the type was registered to be checked on, else the lowest --
and a declaration was held to that one run, so `palace` declared 12 as its ceiling on
ground it stands clean on to 28: a reliability measurement was deciding architectural
scale. The band is now the **envelope** from the smallest clean size to the largest,
plus, by name, every size between them the sweep stood the type at and found dirty
(`except`). A type declares both; the plan refuses a pad at a named size on either axis
and admits any other size inside the envelope. So the sweep is a floor of reliability
inside the range a type reaches and never a ceiling on it -- and a size that genuinely
breaks is still refused, because it is named. Where the envelope reaches the end of what
was swept, the declared maximum is that end and is a bound on the measurement rather
than on the type.
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import (lint, offline, observe, parallel,               # noqa: E402
                   pipeline, stages)
from ethoslm.buildlib import Builder                                    # noqa: E402
from ethoslm.frontage import Frontage                                   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "rounds", "type-needs.json")

#: One palette for every instance, so what changes between sizes is the size.
MAT = pipeline.voice_palette("white_render_dark_frame")
ROOF = pipeline.voice_roof("white_render_dark_frame")

SEEDS = (1, 2)

#: What is swept, per kind of part. A plot and an area are square pads; a point is a pad
#: size; an edge is a width and a run of centre line. and it stands clean at 26, 30, 36
#: and 44 on the same plane. A great hall in a palace compound is a plot, so a plot is
#: swept past a house's sizes at an area's step.
PLOT_SIZES = tuple(range(3, 23)) + (24, 28, 32)

#: **An area is not a plot and does not stop at 22.** A market square is 12x12 and a
#: palace compound is 48x48; sweeping an area type to a plot's ceiling measures a band
#: that cannot contain the ground the type exists for, and the band a type declares is
#: the one holding the fixture it is checked on. Beyond 22 the step widens: the
#: interesting sizes for an area are the ones a plan actually draws, and sweeping every
#: integer to 48 is four times the work for the same answer.
AREA_SIZES = PLOT_SIZES + (40, 48)
POINT_SIZES = tuple(range(3, 17))
#: **A wall's mass reaches a rampart's**, the craft round (E4): these were 1, 2 and 3,
#: so three was all that was ever certified and a city's outer wall was forty-eight high
#: and three thick -- a screen. The widths a place may declare are
#: `placeplan.WALL_MASSES`, and the sweep tries them and the ones between.
EDGE_WIDTHS = (1, 2, 3, 5, 7, 9, 12)
#: An edge's runs reach a district's scale on purpose.
EDGE_RUNS = (4, 8, 16, 32, 64, 96, 128)

#: The ground the sweep stands on: one level plane, deep enough to plinth into.
FLAT_Y = 64

#: and a size passes only if it passes on both. The plane alone was the whole sweep for
#: four rounds and it was measuring half the question. `site()` gives a pad with little
#: relief a **plinth** and one with real relief a **platform**, and those are two
#: different pieces of ground for everything that stands on the edge of them: the way
#: in, the doorstep, what the pad's rim looks like. rooms with no walkable floor -- in
#: `court_large` and `hall` at 24 to 31 across, every one of them clean at that size on
#: this sweep's plane; the cause was in the library and is fixed
#: (`TypeBuilder._clear_doorstep`), and the reason it reached a city is that **no
#: instrument in this project ever stood a large plot on a slope**. The terrain bank
#: asks whether the library can prepare ground and the fixtures it samples are six
#: pieces of real world with no large plot on any of them; this asks what size of
#: prepared ground a type can use and asked it on a plane. Neither could see it. A
#: **fall across the part** and not a gradient: a platform of any size cuts the same
#: amount of face, so one gradient would make a 3x3 pad flat and a 32x32 pad a cliff and
#: the sweep would be measuring the fixture. Eight is the most `site()` takes across
#: every size swept without shrinking the pad it was asked for -- at ten it comes back
#: 13x13 -- and it is above the plinth/platform line at every one of them.
SLOPE_FALL = 8
GROUNDS = ("flat", "slope")


def flat(size: int, y: int = FLAT_Y) -> observe.Volume:
    y0, y1 = y - 14, y + 46
    palette = ["air", "grass_block", "dirt", "stone"]
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return observe.Volume(0, y0, 0, codes, palette)


def slope(size: int, x0: int, x1: int, y: int = FLAT_Y,
          fall: int = SLOPE_FALL) -> observe.Volume:
    """A bank falling `fall` blocks from x0 to x1, level above it and level below it.

        Level either side on purpose: what is under test is the **part** on relief, and a
        gradient carried across the whole working area would put the far end of a 128-block
        edge's approach a hundred blocks down a mountain and measure the volume's depth.
        
    """
    y0, y1 = y - fall - 14, y + 46
    palette = ["air", "grass_block", "dirt", "stone"]
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    n = max(1, x1 - x0)
    for x in range(size):
        t = min(max((x - x0) / n, 0.0), 1.0)
        g = y - int(round(fall * t))
        codes[x, :g - y0, :] = 3
        codes[x, g - y0, :] = 1
    return observe.Volume(0, y0, 0, codes, palette)


#: One bank and its lane per (working area, part rectangle), built once and copied.
#: `par_map` forks a worker per size, so this is warm for the sixty-odd instances of one
#: size and cold for the next -- which is the right grain: the ground is a fact about
#: the size, the instances are what vary.
_BANKS: dict = {}


def bank(span: int, plot: dict) -> tuple:
    """(a bank across this part, the lane laid to it). A copy each time; nothing shared.

        **A fixture is not a fixture until it is joined to the rest of the place**, which is
        a rule this project earned twice before this and paid for a third time here. The
        plane needs no lane because every column of it is at grade: the perimeter of the
        working area *is* the lane, `approach()` seeds from it, and a doorstep anywhere on
        the pad opens onto ground at its own level. A **bank** has no such property -- a
        platform stands proud of the ground on its low side -- and a first cut of this arm
        with no lane in it reported nine of the eleven plot types as standing at no size at
        all: `building()` refused every door because the cell it opened onto was the
        platform's own skirt. Every one of them stands at every size once the lane is there.

        So the bank carries what `slopefixture` has carried since it was written: the plot,
        an arrival on the **downhill** side, a network planned between them and emitted, and
        the thresholds cut down to this part's own. An edge and a point get no lane, because
        a wall is not entered and a gate is a hole in one.
        
    """
    key = (int(span), plot["x0"], plot["z0"], plot["x1"], plot["z1"])
    if key not in _BANKS:
        from ethoslm import circulate
        # One at a time: a worker takes one size at a time and the ground is a fact
        # about the size, so holding every size a worker has seen is a gigabyte of
        # volumes for nothing.
        _BANKS.clear()
        vol = slope(span, plot["x0"], plot["x1"])
        h, _wet = observe.ground_heights(vol)
        h = h.astype(int)
        mid = (plot["z0"] + plot["z1"]) // 2
        gx, gz = min(span - 2, plot["x1"] + 8), mid
        net = circulate.plan_network(
            h, 0, 0,
            [dict(plot, id=plot["label"]),
             dict(id="arrival", x0=gx - 1, z0=gz - 1, x1=gx + 1, z1=gz + 1)],
            centre=plot["label"], max_step=3)
        net.thresholds = [t for t in net.thresholds if t.id == plot["label"]]
        b = Builder(offline.OfflineSite(vol))
        b._vol = vol
        circulate.emit(b, net, ground=h, x0=0, z0=0, lantern_every=0, post_every=0)
        b.resolve_steps()
        _BANKS[key] = (stages.apply_pending(vol, b._pending), net)
    vol, net = _BANKS[key]
    return (observe.Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(),
                           list(vol.palette)), net)


class _Registry:
    """The one plot this instance owns, and nothing else."""

    def __init__(self, plot: dict):
        self.plots = [dict(plot)]
        self.claimed_this_pass = [dict(plot)]

    def plots_list(self) -> list:
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k) -> bool:
        return True


_DECL: dict = {}


def declaration(name: str) -> dict:
    """The type file's namespace, executed once. `pipeline.load_type` validates it."""
    if name not in _DECL:
        path = os.path.join(ROOT, "types", f"{name}.py")
        ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
        exec(compile(open(path).read(), path, "exec"), ns)            # noqa: S102
        pipeline.load_type(path)
        _DECL[name] = ns
    return _DECL[name]


def param_combinations(name: str) -> list:
    """Every value of every declared parameter, crossed. See `PARAMS`.

        `pipeline.param_combinations` is the one enumeration, so the needs sweep and the
        type checker cross the same space -- the arithmetic between two instruments belongs
        to one of them. The sweep asks for the whole of an int range whatever its width,
        which is what it has always done and what a *size* sweep can afford; the checker
        samples a long one, and says so.
        
    """
    return pipeline.param_combinations(declaration(name)["PARAMS"], most=1 << 30)


def _plot_for(pad: tuple) -> tuple:
    """A plot whose sited pad is exactly `pad`. The inverse of `Builder.pad_extent`."""
    pw, pd = pad
    return (pw + 2, pd + 2) if min(pw, pd) < 7 else (pw + 4, pd + 4)


def _part(name: str, kind: str, size: tuple) -> tuple:
    """(part, registry plot, world size) for one instance of one size."""
    if kind == "plot":
        w, d = _plot_for(size)
        part = {"label": "needs", "kind": "plot",
                "x0": 20, "z0": 20, "x1": 19 + w, "z1": 19 + d}
        span = max(w, d)
    elif kind == "area":
        part = {"label": "needs", "kind": "area",
                "x0": 20, "z0": 20, "x1": 19 + size[0], "z1": 19 + size[1]}
        span = max(size)
    elif kind == "point":
        part = {"label": "needs", "kind": "point", "at": [30, 30],
                "facing": "north", "size": size[0]}
        # A passage point is also swept **in a wall**: siting sizes a gate's pad from
        # its wall's height (`Builder.point_pad`), so the pad this size answers to is
        # the wall three times as high, and the type builds to that crown. Demo-polish,
        # 2b. Below nine there is no tower form and the wall is a low one.
        if _passage(name) and size[0] >= 9:
            part["edge"] = {"name": "a_wall", "type": "great_wall",
                            "height": Builder.POINT_PAD_PER_HEIGHT * size[0], "width": 3}
        span = size[0] + 20
    else:
        width, run = size
        part = {"label": "needs", "kind": "edge", "width": width,
                "path": [[20, 20], [19 + run, 20]]}
        span = run + 10
    x0, z0, x1, z1 = pipeline.part_rect(part)
    return part, {"label": "needs", "x0": x0, "z0": z0, "x1": x1, "z1": z1}, span + 45


def instance(name: str, kind: str, size: tuple, seed: int, params: dict,
             ground: str = "flat") -> dict:
    """One type, one size, one seed, one set of parameters, on one ground. Nothing is
        written anywhere.

        Returns {"pass", "why"}: `pass` is the type standing and the build family of the
        linter reporting no error against the plot it stands on. `ground` is one of
        `GROUNDS` -- the plane, or the bank that falls `SLOPE_FALL` across the part.
        
    """
    part, plot, span = _part(name, kind, size)
    net = None
    if ground == "flat":
        vol = flat(span)
    elif kind in ("plot", "area"):
        vol, net = bank(span, plot)
    else:
        vol = slope(span, plot["x0"], plot["x1"])
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net is not None else None
    b.registry = _Registry(plot)
    p = b.site(dict(part), mat=MAT, roof=ROOF)
    if p.get("ground") == "unsited":
        return {"pass": False, "why": f"unsited: {p['sited']['reason']}"}
    if kind == "plot":
        got = (p["x1"] - p["x0"] + 1, p["z1"] - p["z0"] + 1)
        if got != tuple(size):
            return {"pass": False, "why": f"the pad came back {got[0]}x{got[1]}, not "
                                          f"{size[0]}x{size[1]}"}
    try:
        res = declaration(name)["build"](b.type_builder(p), p, seed, **params)
    except Exception as e:                       # noqa: BLE001 -- this is the result
        return {"pass": False, "why": f"raised {type(e).__name__}: {e}"}
    if isinstance(res, dict) and res.get("ok") is False:
        return {"pass": False, "why": f"refused: {res.get('reason')}"}
    if not res:
        return {"pass": False, "why": "build() returned nothing"}
    b.resolve_steps()
    built = stages.apply_pending(vol, b._pending)
    region = (min(plot["x0"], p["x0"]) - 6, min(plot["z0"], p["z0"]) - 6,
              max(plot["x1"], p["x1"]) + 6, max(plot["z1"], p["z1"]) + 6)
    ctx = lint.Context.build(built, plots=[plot], region=region,
                             fittings=b.fitting_cells)
    rep = pipeline.standard_report(ctx, [plot])
    errs = [f for f in rep.findings if f.code.startswith("E")]
    if errs:
        return {"pass": False,
                "why": f"{len(errs)} own lint error(s): "
                       + "; ".join(f"{f.code} {f.message}" for f in errs[:2])}
    return {"pass": True, "why": ""}


@functools.lru_cache(maxsize=None)
def _passage(name: str) -> bool:
    try:
        return bool(pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))["passage"])
    except Exception:                          # noqa: BLE001 -- not a gate, then
        return False


def sizes_for(kind: str) -> list:
    if kind == "edge":
        return [(w, r) for w in EDGE_WIDTHS for r in EDGE_RUNS]
    if kind == "point":
        return [(n, n) for n in POINT_SIZES]
    if kind == "area":
        return [(n, n) for n in AREA_SIZES]
    return [(n, n) for n in PLOT_SIZES]


def _sweep_size(job) -> dict:
    """One size of one type, over both seeds and every parameter combination.

        The unit of work, and it shares nothing with any other: its own synthetic ground,
        its own builder, its own registry. A19's `par_map` submits these in order and
        collects them in order, so the rows are the rows a serial loop produces.
    """
    name, kind, size, combos = job
    bad = []
    for ground in GROUNDS:
        for seed in SEEDS:
            for params in combos:
                r = instance(name, kind, size, seed, params, ground=ground)
                if not r["pass"]:
                    bad.append({"ground": ground, "seed": seed, "params": params,
                                "why": r["why"]})
    return {"size": list(size), "of": len(GROUNDS) * len(SEEDS) * len(combos),
            "failed": len(bad), "grounds": list(GROUNDS),
            "on": sorted({b["ground"] for b in bad}),
            "first": bad[0] if bad else None}


def sweep_jobs(name: str) -> tuple:
    """(kind, parameter combinations, one job per size) for one type."""
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
    kind = decl["kind"]
    combos = param_combinations(name)
    return kind, combos, [(name, kind, size, combos) for size in sizes_for(kind)]


def assemble(name: str, kind: str, combos: list, rows: list,
             verbose: bool = False) -> dict:
    """One type's answer, from its rows. No work; the rows are the work."""
    if verbose:
        for row in rows:
            print(f"  {name:11} {str(tuple(row['size'])):9} "
                  f"{row['of'] - row['failed']:3}/{row['of']:3} pass"
                  + (f"   {row['first']['why'][:90]}" if row["first"] else ""),
                  flush=True)
    return {"type": name, "kind": kind, "combinations": len(combos),
            "seeds": list(SEEDS), "sizes": rows, "band": band(kind, rows)}


def sweep(name: str, verbose: bool = False) -> dict:
    """Every size, seed and parameter combination of one type."""
    kind, combos, jobs = sweep_jobs(name)
    return assemble(name, kind, combos, parallel.par_map(_sweep_size, jobs), verbose)


def band(kind: str, rows: list) -> dict:
    """The envelope of clean sizes, and by name every size inside it that failed.

        Every run the sweep found is kept -- `runs` -- and the declaration a type is held
        to is the envelope of all of them with the failing sizes between them named. A run
        of one size counts: it passed at both seeds and every parameter combination, which
        is the same evidence any other size has.
        
    """
    if kind == "edge":
        # An edge's band is a **box** in (width x run) and every pair inside it has to
        # pass, which a min-and-max over the two axes taken separately does not give
        # you: the wall is clean at width 1 on every run swept and carries one floating
        # block at widths 2 and 3 on two of them, and a band of "width 1 to 3, run 4 to
        # 128" would claim those two. Anchored at the **smallest** run that passes, and
        # then as wide as it will go in run and in width: what a plan has to know is
        # what a type can be relied on for from its smallest size up.
        runs = sorted({r["size"][1] for r in rows})
        widths = sorted({r["size"][0] for r in rows})
        ok = {(r["size"][0], r["size"][1]): not r["failed"] for r in rows}
        passing = [i for i, r in enumerate(runs) if any(ok.get((w, r)) for w in widths)]
        if not passing:
            return {"footprint": None, "runs": [], "except": [],
                    "note": "no size passed"}
        first = passing[0]
        best = None
        for i in [first]:
            for j in range(i, len(runs)):
                w = 0
                for width in widths:
                    if all(ok.get((width, runs[k]), False) for k in range(i, j + 1)):
                        w = width
                    else:
                        break
                if not w:
                    continue
                key = (j - i + 1, w)
                if best is None or key > best[0]:
                    best = (key, [widths[0], runs[i], w, runs[j]])
        if best is None:
            return {"footprint": None, "runs": [], "except": [],
                    "note": "no size passed"}
        return {"footprint": best[1], "runs": [], "except": [],
                "note": ("an edge's pair is (width, run of the longest segment); the "
                         "band is the widest box every pair in which passed")}
    swept = (AREA_SIZES if kind == "area" else
             POINT_SIZES if kind == "point" else PLOT_SIZES)
    ok = [r["size"][0] for r in rows if not r["failed"]]
    bad = sorted(r["size"][0] for r in rows if r["failed"])
    runs = _bands(ok, swept)
    if not runs:
        return {"footprint": None, "runs": [], "except": [], "note": "no size passed"}
    lo, hi = runs[0][0], runs[-1][-1]
    dirty = [v for v in bad if lo < v < hi]
    top = max(s["size"][0] for s in rows)
    note = ("the maximum is the largest size swept" if hi == top
            else "the envelope closes below what was swept")
    if len(runs) > 1:
        note += (f"; clean over {len(runs)} runs of sizes "
                 + ", ".join(f"{b[0]}-{b[-1]}" for b in runs)
                 + f", and broken at {', '.join(map(str, dirty))} between them")
    return {"footprint": [lo, lo, hi, hi], "runs": [[b[0], b[-1]] for b in runs],
            "except": dirty, "note": note}


def _bands(values: list, swept: tuple | None = None) -> list:
    """Every unbroken run of values, in order, **as the sweep offered them**.

        "Unbroken" used to mean consecutive integers, which is right while the sweep steps
        by one and wrong the moment it does not. `AREA_SIZES` widens past 22 -- 24, 28, 32,
        40, 48 -- and `field`, which is clean at every size that was swept, came back as six
        separate runs (3-22, 24-24, 28-28, 32-32, 40-40, 48-48) because 23 and 25 are not in
        the list and never were. Its declared maximum was then 22, on ground it stands clean
        on to 48.

        A run is now a maximal stretch of **adjacent entries of the swept sequence**, so a
        gap in the band means a size that was tried and failed rather than one nobody asked
        about.
        
    """
    order = list(swept) if swept else None
    out: list = []
    prev = None
    for v in sorted(values):
        step = (order.index(v) == order.index(prev) + 1) if (order and prev is not None
                                                             and v in order
                                                             and prev in order) \
            else (prev is not None and v == prev + 1)
        if out and step:
            out[-1].append(v)
        else:
            out.append([v])
        prev = v
    return out


#: Every type on disk, in the order they are read. `_reference`, `_edge`, `_point` and
#: `_area` are the hand-written types the conformance suites stand on and are swept for
#: the same reason the six are: a declaration nobody measured is a comment.
def type_names() -> list:
    return sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "types"))
                  if f.endswith(".py"))


def build_bank(names=None, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    out = {
        "what": ("the pad each committed type stands clean on, swept on level ground "
                 "and on a bank"),
        "how": ("scripts/type_needs.py: every size, both seeds, every combination "
                "of the type's own PARAMS and **both grounds** -- one synthetic level "
                "plane and one bank falling SLOPE_FALL blocks across the part -- "
                "through Builder.site() so the pad is exactly the size asked for. An "
                "instance passes when build() stands and the build family of the "
                "linter reports no error on its plot, and a size passes only when "
                "every instance of it on both grounds does. The band is the envelope "
                "of every clean size with the broken sizes inside it named (`except`), "
                "so the sweep is a floor of reliability inside the range a type "
                "reaches and never a ceiling on it."),
        "swept": {"plot_and_area": list(PLOT_SIZES), "area": list(AREA_SIZES),
                  "point": list(POINT_SIZES),
                  "edge_widths": list(EDGE_WIDTHS), "edge_runs": list(EDGE_RUNS),
                  "seeds": list(SEEDS), "grounds": list(GROUNDS),
                  "slope_fall": SLOPE_FALL},
        "types": {},
    }
    # **One pool over every type**, not one per type. A6, and the reason is measured:
    # `wall`'s longest single job -- a 128-column run three wide -- is a large fraction
    # of the whole sweep, so parallelising *within* a type leaves that type's wall-clock
    # exactly where it was and the whole bank barely moves. Flattened, the long job runs
    # beside every other type's short ones.
    order = list(names or type_names())
    plan = {name: sweep_jobs(name) for name in order}
    flat = [job for name in order for job in plan[name][2]]
    done = parallel.par_map(_sweep_size, flat)
    at = 0
    for name in order:
        kind, combos, jobs = plan[name]
        if verbose:
            print(f"=== {name}", flush=True)
        out["types"][name] = assemble(name, kind, combos,
                                      done[at:at + len(jobs)], verbose)
        at += len(jobs)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def transcribe(bank: dict, names=None) -> dict:
    """Write each type's measured band onto its own file: the envelope as `footprint`
        and the broken sizes inside it as `except`. Returns {type: (was, now)} for every
        declaration that moved, so the record can name them all.

        The harness's act and not an author's: the declaration is the measurement, and
        the two lines this rewrites are the two the sweep is the instrument for. A type
        whose file does not carry a `"footprint":` line inside `NEEDS` is left alone and
        named.
        
    """
    import re
    moved: dict = {}
    for name, row in bank["types"].items():
        if names and name not in names:
            continue
        band = row["band"]
        if not band.get("footprint"):
            continue
        path = os.path.join(ROOT, "types", f"{name}.py")
        src = open(path).read()
        decl = pipeline.load_type(path)["needs"]
        want_fp = tuple(int(v) for v in band["footprint"])
        want_ex = tuple(int(v) for v in (band.get("except") or []))
        if tuple(decl["footprint"]) == want_fp and tuple(decl.get("except") or ()) == want_ex:
            continue
        m = re.search(r'^(\s*)"footprint":\s*\([^)]*\),?[^\n]*$', src, re.M)
        if not m:
            moved[name] = ("no footprint line to rewrite", None)
            continue
        indent = m.group(1)
        line = f"{indent}\"footprint\": {want_fp},"
        if want_ex:
            line += f"\n{indent}\"except\": {want_ex},"
        src = src[:m.start()] + line + src[m.end():]
        # an `except` line already there is replaced by the one above or dropped
        src = re.sub(r'^\s*"except":\s*\([^)]*\),?[^\n]*\n', "",
                     src[:m.start()] + src[m.start():].replace(line, "\x00", 1), count=0,
                     flags=re.M).replace("\x00", line, 1)
        open(path, "w").write(src)
        moved[name] = ({"footprint": list(decl["footprint"]),
                        "except": list(decl.get("except") or ())},
                       {"footprint": list(want_fp), "except": list(want_ex)})
    return moved


def main(argv: list) -> int:
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")] or None
    if "--merge" in argv:
        # Re-sweep the named types and write their rows into the registered bank, the
        # rest of it unmoved: a round that changes two types does not wait an hour and a
        # half for the twenty-one it did not touch. `--check` still reproduces the whole
        # file, so a merged row that does not reproduce is found the same way.
        if not names or not os.path.exists(OUT):
            print("--merge takes type names, and the bank has to exist")
            return 1
        got = build_bank(names, verbose=True)
        bank = json.load(open(OUT))
        for name in names:
            bank["types"][name] = got["types"][name]
        bank["swept"] = got["swept"]
        bank["merged"] = dict(bank.get("merged") or {},
                              **{n: {"seconds": got["seconds"]} for n in names})
        json.dump(bank, open(OUT, "w"), indent=1)
        for name in names:
            print(f"  {name:12} band {got['types'][name]['band']}")
        print(f"{len(names)} row(s) merged into {os.path.relpath(OUT, ROOT)} "
              f"({got['seconds']}s)")
        return 0
    if "--transcribe" in argv:
        if not os.path.exists(OUT):
            print(f"FAIL {os.path.relpath(OUT, ROOT)} does not exist")
            return 1
        moved = transcribe(json.load(open(OUT)), names)
        for name, (was, now) in moved.items():
            print(f"  {name:12} {was} -> {now}")
        print(f"{len(moved)} declaration(s) moved onto their files")
        return 0
    got = build_bank(names, verbose=not check)
    body = {k: v for k, v in got.items() if k != "seconds"}
    if check:
        if not os.path.exists(OUT):
            print(f"FAIL {os.path.relpath(OUT, ROOT)} does not exist")
            return 1
        was = {k: v for k, v in json.load(open(OUT)).items() if k != "seconds"}
        if json.dumps(was, sort_keys=True) != json.dumps(body, sort_keys=True):
            print(f"FAIL {os.path.relpath(OUT, ROOT)} does not reproduce")
            for name, row in body["types"].items():
                if was.get("types", {}).get(name) != row:
                    print(f"  moved: {name}")
            return 1
        print(f"ok   {os.path.relpath(OUT, ROOT)} reproduces: "
              f"{len(body['types'])} types in {got['seconds']}s")
        return 0
    if names:
        for name in names:
            print(f"{name}: band {got['types'][name]['band']}")
        return 0
    json.dump(got, open(OUT, "w"), indent=1)
    print(f"\n{OUT}  ({got['seconds']}s)")
    for name, row in got["types"].items():
        print(f"  {name:12} {row['kind']:6} band {row['band']['footprint']}"
              + (f" except {row['band']['except']}" if row["band"].get("except")
                 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
