"""The conformance suite. Can the library prepare ground anywhere?

    $PY scripts/test_ground.py                 # all 144 instances
    $PY scripts/test_ground.py round9_wettest  # one fixture, or any substring of one

`types/_reference.py` -- one `building()` call and one `fitting()`, written by hand --
stood on every fixture of `rounds/terrain-bank.json` at seeds 1 and 2 and one and two
storeys, and held to five assertions per instance. 36 x 2 x 2 = **144**.

The type decides nothing, so a failure is the library's. That is the whole design:
rounds 12, 13 and 14 each found one or two ground defects with a model-authored type on
a new site, at a session and a million tokens apiece, and each time the first argument
was about whose fault it was. Here there is nobody to blame.

The suite is **binary**: it is done when every instance passes. A defect it exposes is
fixed in the library and its fixture kept -- the rule against fitting to a test set is
about the model's side, never about the library.

What each instance is held to:

  1. own lint errors 0, over and above what the bare ground and its lane already carry;
  2. >= 90% of its interior floor walkable from the lane, and its door walk-
     reachable from the lane;
  3. no room whose floor is below `part["floor_y"]` -- the sealed void under a deck;
  4. every water column of the patch outside the pad and the way in is still water;
  5. nothing solid placed outside the plot except the way in.
"""
from __future__ import annotations

import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import (circulate, lint, observe, offline, parallel,        # noqa: E402
                   pipeline, stages)
from ethoslm.buildlib import AIR, Builder                                # noqa: E402
from ethoslm.frontage import Frontage                                    # noqa: E402

import terrain_bank                                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCE = os.path.join(ROOT, "types", "_reference.py")

#: How far outside the patch the working volume reaches. Wide enough for `approach()`'s
#: 24-block radius off a lane on the far edge and for a walk model that has somewhere to
#: come from; narrow enough that a Context costs a second rather than fifteen.
PAD = 18

SEEDS = (1, 2)
STOREYS = (1, 2)

#: One palette for every instance, so what changes between fixtures is the ground and
#: nothing else. The same voice `test_siting.py` sites its three cases in.
MAT = pipeline.voice_palette("white_render_dark_frame")

WATER = ("water", "bubble_column")


# ------------------------------------------------------------------ the ground

class _Registry:
    """The plot registry for one fixture: its own plot and nothing else."""

    def __init__(self, plot: dict):
        self.plots = [dict(plot)]
        self.claimed_this_pass = [dict(plot)]

    def plots_list(self) -> list:
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k) -> bool:
        return True


def _bed_heights(vol: observe.Volume) -> np.ndarray:
    """The topmost firm block per column: the bed, never the waterline, never a leaf.

        `observe.ground_heights` answers with the water surface over a lake, because a lane
        across a pond is a causeway at water level and that is what the router needs. A
        causeway also has to be *filled* to something, and filling to the surface is a road
        floating on a lake -- so the fill in this harness is measured from here.
        
    """
    t = vol.tables()
    c = vol.codes
    firm = (t["lower"][c].astype(bool) & t["upper"][c].astype(bool)
            & ~np.array([observe._is_vegetation(s) for s in vol.palette], bool)[c])
    idx = firm.shape[1] - 1 - np.argmax(firm[:, ::-1, :], axis=1)
    return np.where(firm.any(axis=1), idx + vol.y0, vol.y0).astype(int)


@functools.lru_cache(maxsize=8)
def _world(round_name: str, cache: str) -> observe.Volume:
    return offline.load_volume(offline.world_cache(round_name, cache))


@functools.lru_cache(maxsize=64)
def _ground(key: int) -> tuple:
    """(volume with the fixture's lane built into it, network, plot) for one fixture.

        The lane is laid **before** the builder under test exists, exactly as a round lays
        it: the circulation stage runs and flushes, and the build waves then read a world
        that already has a street in it. So it is `circulate.emit`, the same call a round
        makes, and what it writes is world and not the instance's own work.

        Lamp posts and lanterns are off. They are the one thing `emit` places that is
        lighting rather than circulation, and a fixture bank is not the place to measure
        whether a lantern is in the way.
        
    """
    fx = terrain_bank.load()["fixtures"][key]
    full = _world(fx["round"], fx["cache"])
    net = circulate.Network.from_json(fx["lane"])
    sub = full.sub(fx["x0"] - PAD, fx["z0"] - PAD,
                   fx["size"] + 2 * PAD, fx["size"] + 2 * PAD)
    b = Builder(offline.OfflineSite(sub))
    b._vol = sub
    circulate.emit(b, net, mat=net.notes.get("material", "cobblestone"),
                   ground=_bed_heights(sub), x0=sub.x0, z0=sub.z0,
                   lantern_every=0, post_every=0)
    b.resolve_steps()
    return stages.apply_pending(sub, b._pending), net, dict(fx["plot"])


@functools.lru_cache(maxsize=4)
def instance(key: int, seed: int, storeys: int) -> dict:
    """Site one fixture and stand the reference type on it. Nothing is written anywhere.

        Returns the builder, the part, the volume the instance stands in and the world it
        stood in before it -- which is what makes "the instance's own writes" a set rather
        than an argument.
        
    """
    base, net, plot = _ground(key)
    b = Builder(offline.OfflineSite(base))
    b._vol = base
    b.frontage = Frontage(base, net)
    b.registry = _Registry(plot)
    part = b.site(dict(plot, kind="plot"), mat=MAT)
    decl = _declaration()
    res = decl["build"](b.type_builder(part), part, seed, storeys=storeys)
    b.resolve_steps()
    return {"base": base, "net": net, "plot": plot, "b": b, "part": part,
            "built": res, "vol": stages.apply_pending(base, b._pending)}


@functools.lru_cache(maxsize=1)
def _declaration() -> dict:
    ns: dict = {"__name__": "__ethoslm_type__", "__file__": REFERENCE}
    exec(compile(open(REFERENCE).read(), REFERENCE, "exec"), ns)      # noqa: S102
    pipeline.load_type(REFERENCE)                # ...and it is a type by the contract
    return ns


# ------------------------------------------------------------- what is asserted

def _region(fx: dict) -> tuple:
    return (fx["x0"], fx["z0"], fx["x0"] + fx["size"] - 1, fx["z0"] + fx["size"] - 1)


def _key(f) -> tuple:
    """A finding, as something two reports can be differenced on."""
    d = f.detail if isinstance(f.detail, dict) else {}
    return (f.code, str(d.get("at") or d.get("pos") or d.get("bbox") or ""), f.message)


@functools.lru_cache(maxsize=64)
def _floor(key: int) -> tuple:
    """(findings, room bboxes) the fixture already carries with nothing built on it."""
    base, net, plot = _ground(key)
    fx = terrain_bank.load()["fixtures"][key]
    ctx = lint.Context.build(base, plots=[plot], network=net, region=_region(fx))
    rep = pipeline.standard_report(ctx, [plot])
    return (frozenset(_key(f) for f in rep.findings),
            tuple(tuple(r["bbox"]) for r in ctx.rooms))


def _top_block(vol: observe.Volume, x: int, z: int) -> str:
    """The name of the topmost block a body could stand on in this column."""
    t = vol.tables()
    ix, iz = x - vol.x0, z - vol.z0
    if not (0 <= ix < vol.codes.shape[0] and 0 <= iz < vol.codes.shape[2]):
        return "air"
    col = vol.codes[ix, :, iz]
    firm = (t["lower"][col].astype(bool) & t["upper"][col].astype(bool)
            & ~np.array([observe._is_vegetation(s) for s in vol.palette], bool)[col])
    if not firm.any():
        return "air"
    return vol.palette[col[int(np.argwhere(firm).max())]].split("[")[0]


def _surround(base: observe.Volume, plot: dict) -> str:
    """What the undisturbed ground just outside a plot is made of. A6.

        The commonest top block on the ring four columns out, which is past everything
        `site()` clears and is the answer `dress_ground` gives itself when nobody names a
        cover. Where that ring is itself bare subsoil there is nothing to put back and the
        assertion does not apply.
        
    """
    x0, z0 = min(plot["x0"], plot["x1"]), min(plot["z0"], plot["z1"])
    x1, z1 = max(plot["x0"], plot["x1"]), max(plot["z0"], plot["z1"])
    ring: dict = {}
    for x in range(x0 - 4, x1 + 5):
        for z in (z0 - 4, z1 + 4):
            ring[_top_block(base, x, z)] = ring.get(_top_block(base, x, z), 0) + 1
    for z in range(z0 - 4, z1 + 5):
        for x in (x0 - 4, x1 + 4):
            ring[_top_block(base, x, z)] = ring.get(_top_block(base, x, z), 0) + 1
    ring.pop("air", None)
    ring.pop("water", None)
    return max(ring, key=lambda k: ring[k]) if ring else "air"


def _overlaps(a, b) -> bool:
    return (a[0] <= b[3] and b[0] <= a[3] and a[1] <= b[4] and b[1] <= a[4]
            and a[2] <= b[5] and b[2] <= a[5])


def _inside(pos, box) -> bool:
    return (box[0] <= pos[0] <= box[3] and box[1] <= pos[1] <= box[4]
            and box[2] <= pos[2] <= box[5])


def assertions(key: int, seed: int, storeys: int) -> dict:
    """One instance, read: `bad` is every failure named, and empty `bad` is a pass."""
    fx = terrain_bank.load()["fixtures"][key]
    got = instance(key, seed, storeys)
    b, part, plot, base = got["b"], got["part"], got["plot"], got["base"]
    bad, note = [], ""
    if not (got["built"] or {}).get("ok"):
        return {"bad": [f"the reference type built nothing: "
                        f"{(got['built'] or {}).get('reason', 'build() returned nothing')}"],
                "note": ""}

    ctx = lint.Context.build(got["vol"], plots=[plot], network=got["net"],
                             region=_region(fx))
    rep = pipeline.standard_report(ctx, [plot])
    floor, was = _floor(key)
    # ...and not what was already there. A badlands mesa is hollow. The instance did not
    # put it there and cannot be answerable for it.
    mine = [f for f in rep.findings
            if f.code.startswith("E") and _key(f) not in floor
            and not (f.pos and any(_inside(f.pos, w) for w in was))]
    if mine:
        bad.append(f"1. {len(mine)} own lint error(s): "
                   + "; ".join(f"{f.code} {f.message}" for f in mine[:3]))

    # **The interior is the one `building()` says it built.** `Context.interior_walk`
    # measures rooms found by flooding sheltered space, and on a cliff the sheltered
    # ground under an overhang joins the interior through its own doorway. That is the
    # walk metric's own open thread and it is not a fact about the ground, so the floor
    # counted here is the rectangle per storey the shell returned, walked from the lane
    # by the flood every other from-outdoors measurement in this project uses.
    cells, walk = 0, 0
    for (rx0, ry, rz0, rx1, rz1) in (got["built"] or {}).get("rooms", []):
        for x in range(rx0, rx1 + 1):
            for z in range(rz0, rz1 + 1):
                st = ctx.nav.stance_near(x, z, ry + 1, tol=1)
                cells += 1
                walk += int(st is not None and (x, z, st) in ctx.from_outdoors)
    pct = round(100 * walk / cells, 1) if cells else 0.0
    if not cells:
        bad.append("2. the instance has no interior at all: nothing to walk in")
    elif pct < 90:
        bad.append(f"2. {pct}% of {cells} interior floor cells walkable from the lane")
    door = (got["built"] or {}).get("door")
    st = ctx.door_stance(*door) if door else None
    if st is None or (door[0], door[2], st) not in ctx.from_outdoors:
        bad.append(f"2. the door at {tuple(door) if door else None} cannot be walked "
                   f"to from the lane"
                   + ("" if st is not None else " -- nothing can stand in it"))

    floor_y = part["floor_y"]
    x0, z0, x1, z1 = part["footprint"]
    under = [r for r in ctx.rooms
             if r["bbox"][0] <= x1 + 1 and x0 - 1 <= r["bbox"][3]
             and r["bbox"][2] <= z1 + 1 and z0 - 1 <= r["bbox"][5]
             and r["bbox"][1] < floor_y
             and not any(_overlaps(r["bbox"], w) for w in was)]
    if under:
        r = under[0]
        bad.append(f"3. a room of {r['cells']} cells at y={r['bbox'][1]} under the "
                   f"floor at y={floor_y}: bbox {r['bbox']}")

    spared = {(x, z) for x in range(x0 - 1, x1 + 2) for z in range(z0 - 1, z1 + 2)}
    spared |= {(c[0], c[1]) for p in b.paths for c in p["cells"]}
    drained = sorted({(p[0], p[1], p[2]) for p, blk in b._pending.items()
                      if (p[0], p[2]) not in spared
                      and base.name(*p) in WATER
                      and blk.split("[")[0].split(":")[-1] not in WATER})
    if drained:
        bad.append(f"4. {len(drained)} water cell(s) written outside the pad and the "
                   f"way in, the first at {drained[0]}")

    px0, pz0 = min(plot["x0"], plot["x1"]), min(plot["z0"], plot["z1"])
    px1, pz1 = max(plot["x0"], plot["x1"]), max(plot["z0"], plot["z1"])
    path = {(c[0], c[1]) for p in b.paths for c in p["cells"]}
    # ...and the columns the ground pass put the skin back on. `clear_trees` reaches
    # past the plot by however far a tree rooted outside it leaned in, and leaving the
    # dirt it stood on bare is the defect A6 exists for, so putting grass back on it is
    # ground work rather than somebody building off their plot. By record --
    # `Builder.dressed` -- and not by rectangle, for the reason the fitting register is
    # by record.
    outside = sorted({(p[0], p[2]) for p, blk in b._pending.items()
                      if blk.split("[")[0].split(":")[-1] not in AIR
                      and not (px0 <= p[0] <= px1 and pz0 <= p[2] <= pz1)}
                     - path - b.dressed)
    if outside:
        bad.append(f"5. {len(outside)} column(s) of solid work outside the plot "
                   f"x {px0}..{px1} z {pz0}..{pz1}, the first at {outside[0]}")

    # 6. Worked ground has its skin back. `site()` and `approach()` finish by dressing
    # every column they changed, so a column the pass touched may not be left as the
    # subsoil the cut exposed while the ground around the plot is something else. Read
    # against the *undisturbed* surface round the patch rather than against a named
    # block, because a fixture bank runs over grass, sand, podzol and terracotta.
    around = _surround(base, plot)
    if around not in Builder.BARE_GROUND:
        left = sorted({(x, z) for (x, z) in
                       {(p[0], p[2]) for p in b._pending}
                       if _top_block(got["vol"], x, z) in Builder.BARE_GROUND
                       and _top_block(base, x, z) not in Builder.BARE_GROUND})
        if left:
            bad.append(f"6. {len(left)} column(s) the pass worked are left as bare "
                       f"subsoil where the ground round the plot is {around}, the "
                       f"first at {left[0]}")

    # Reported, never asserted: `entry_lines` holds a room to *every* floor cell being
    # walk-reachable. `observe.floor_stances`) and **132 of the 144 now read zero**. The
    # twelve that remain are three fixtures -- `round8_seeded`, `round11_steepest`,
    # `round11_most_treed` -- and none of them is furniture or ground: they are the
    # merged-room case this function's own note above describes, sheltered natural space
    # that joins the interior or sits under the plot and is attributed to it by
    # `plot_at`. That is the walk metric's remaining open thread, and it is still not a
    # fact about the ground. See STATE.md.
    lines = pipeline.entry_lines(pipeline.diagnose_entry(ctx, plot["label"]))
    note = f"{pct:5.1f}% of {cells:3d} walk, {len(lines)} entry line(s)"
    return {"bad": bad, "note": note}


def _one(work) -> dict:
    """One instance, read and reported, in whatever process this is. A6.

        Everything that crosses a process boundary is a string or a list of strings: the
        builder, the volume and the lint report stay in the worker, because the answer this
        suite gives is `bad` and the line it prints, and nothing downstream reads the rest.
        
    """
    key, seed, storeys = work
    fixtures = terrain_bank.load()["fixtures"]
    try:
        read = assertions(key, seed, storeys)
        got = instance(key, seed, storeys)
    except Exception as e:                        # noqa: BLE001 -- this is a result
        import traceback
        return {"bad": [f"{type(e).__name__}: {e}"], "crashed": traceback.format_exc(),
                "line": (f"FAIL {fixtures[key]['plot']['label']} s{seed} {storeys}st "
                         f"crashed: {type(e).__name__}: {e}")}
    return {"bad": read["bad"], "crashed": None,
            "line": one_line(key, seed, storeys, got, read)}


def one_line(key: int, seed: int, storeys: int, got: dict, read: dict) -> str:
    fx = terrain_bank.load()["fixtures"][key]
    p = got["part"]
    return (f"{fx['plot']['label']:24s} s{seed} {storeys}st {p['ground']:8s} "
            f"y={p['floor_y']:3d} {read['note']:24s} "
            + ("ok" if not read["bad"] else " | ".join(read["bad"])))


# --------------------------------------------------- the contract clarification

DOOR_TYPE = '''"""A type that puts its door somewhere else."""
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2)}


def build(b, part, seed, storeys=1):
    x0, z0, x1, z1 = part["footprint"]
    return b.doorway(x0, part["floor_y"] + 1, z1, "north", "cobblestone")
'''


def case_the_door_is_the_one_site_reserved() -> str:
    """A type asking for a door anywhere but `part['door']` is refused, cell named."""
    got = instance(0, 1, 1)
    tb = got["b"].type_builder(got["part"])
    ns: dict = {}
    exec(compile(DOOR_TYPE, "door_type", "exec"), ns)                 # noqa: S102
    r = ns["build"](tb, got["part"], 1)
    want = tuple(got["part"]["door"])
    assert r["ok"] is False, f"the door was allowed at a cell site() did not reserve: {r}"
    assert "site()" in r["reason"] and str(want[0]) in r["reason"], r["reason"]
    assert tb.refused and tb.refused[-1]["call"] == "doorway", tb.refused

    # ...and the cell site() did reserve is accepted through the same call
    ok = tb.doorway(want[0], got["part"]["floor_y"] + 1, want[1],
                    got["part"]["facing"], "cobblestone", front="ignore")
    assert "ok" in ok and "site()" not in str(ok.get("reason", "")), ok
    return (f"a door at a cell site() did not reserve is refused naming {want}; "
            f"the reserved cell is not")


#: A mangrove seed pod hanging from leaves that `clear_trees` took, left with air above
#: it and air below it.
PROPAGULE = (1634, 70, 414)


def case_a_propagule_is_vegetation() -> str:
    vol = _world("site_f", "world.npz")
    x, y, z = PROPAGULE
    assert "propagule" in vol.state(x, y, z), \
        f"the cache no longer has a propagule at {PROPAGULE}: {vol.state(x, y, z)}"
    assert observe._is_vegetation(vol.state(x, y, z)), "a propagule is not vegetation"
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    assert b.get_height(x, z) == y, "the propagule is not the top of its own column"
    # `bed()`, which is what `site()` sounds with. `grade()` is left answering with the
    # first thing you could stand on.
    assert b.bed(x, z) < y, f"bed() stands a building on the propagule at y={y}"
    b.clear_ground_cover(x - 1, z - 1, x + 1, z + 1)
    got = b._pending.get((x, y, z))
    assert got == "air", f"the ground sweep left {got!r} hanging at {PROPAGULE}"
    return (f"the propagule at {PROPAGULE} is vegetation: swept by the ground pass, "
            f"and the bed under it is y={b.bed(x, z)}, not y={y}")


def case_nothing_is_left_hanging_in_the_jungle() -> str:
    """After the ground pass, no plant on the patch holds nothing."""
    fixtures = terrain_bank.load()["fixtures"]
    keys = [i for i, f in enumerate(fixtures) if f["round"] == "site_a"]
    assert keys, "the terrain bank has no fixtures of that site"
    took = was = 0
    for key in keys:
        got = instance(key, 1, 1)
        b = got["b"]
        took += (got["part"].get("sited") or {}).get("swept") or 0
        cols = {(p[0], p[2]) for p in b._pending}
        m = Builder.SWEEP_MARGIN
        region = (min(c[0] for c in cols) - m, min(c[1] for c in cols) - m,
                  max(c[0] for c in cols) + m, max(c[1] for c in cols) + m)
        already = set(observe.unsupported_vegetation(got["base"], region=region))
        was += len(already)
        left = [p for p in observe.unsupported_vegetation(got["vol"], region=region)
                if p not in already]
        assert not left, (f"{fixtures[key]['plot']['label']}: {len(left)} plant(s) this "
                          f"pass left hanging from nothing, the first at {left[0]} "
                          f"({got['vol'].state(*left[0])})")
    return (f"{len(keys)} fixtures of one site, {took} plant(s) swept, none newly hanging; "
            f"the patches carried {was} the world had already left holding nothing")


# ------------------------------------------------------------------------ main

def main(argv: list) -> int:
    fixtures = terrain_bank.load()["fixtures"]
    want = argv[0] if argv else None
    keys = [i for i, f in enumerate(fixtures)
            if not want or want in f["plot"]["label"]]
    if not keys:
        print(f"no fixture matches {want!r}")
        return 1

    t0 = time.perf_counter()
    bad = 0
    cases = (("the door is the one site() reserved",
              case_the_door_is_the_one_site_reserved),
             ("a propagule is vegetation", case_a_propagule_is_vegetation),
             ("nothing is left hanging in the jungle",
              case_nothing_is_left_hanging_in_the_jungle))
    for name, fn in cases:
        try:
            print(f"ok   {name:52s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:52s} {e}")

    # A6: 144 instances that share nothing, run across processes and read back in the
    # order a serial loop produces them. `ETHOSLM_WORKERS=1` is the serial arm.
    work = [(key, seed, storeys) for key in keys for seed in SEEDS
            for storeys in STOREYS]
    n = len(work)
    for (key, seed, storeys), row in zip(work, parallel.par_map(_one, work)):
        if row.get("crashed"):
            bad += 1
            print(row["line"])
            print(row["crashed"])
            continue
        bad += bool(row["bad"])
        print(("ok   " if not row["bad"] else "FAIL ") + row["line"])
    total = n + len(cases)
    print(f"\n{total - bad}/{total} ground cases pass "
          f"({n} instances over {len(keys)} fixtures, "
          f"{time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
