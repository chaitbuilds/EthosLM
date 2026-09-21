"""The type becomes a first-class thing.

    $PY scripts/test_types.py

  A1. **A type is a file.** `types/<name>.py` with `FORM`, `PARAMS` and
      `build(b, plot, seed, **params)`; `instantiate` checks the parameters against the
      file's own declaration and runs it. Two seeds, two lint-clean walkable buildings
      that are not the same building; a parameter the type did not declare is refused
      by name.
  A2. **The type's checker.** `check.py` in a type builder's directory instantiates its
      `build()` on four plots x two seeds and writes `findings.md` worst first, with a
      table at the top. A type that omits its door reports eight instances and an entry
      line on every one.
  A3. **The shell owns its refusals.** A blocked front cell makes `building()` place
      nothing and say which cell.
  A4. **`roof(tiers=2)` fills the void between its tiers**: no room and no W007.
  A5. **`material()` refuses.** `material("hay")` raises with the family named;
      `material("thatch")` returns the triple; `bamboo` is a family and its stairs and
      slabs are visible to the walk model.
  A6. **`building(platform=N)`**: a hall on a podium whose door and floor a person can
      walk to from the lane.
  A7. **The readout is a stage.** `stage_readout` on rounds 10 and 11 reproduces their
      recorded bar rows -- and names the two the instruments have superseded since.
  A8. **The shot list moved home.** The three experiment scripts are gone and nothing
      imports them; `render` owns what `pipeline` was reaching through them for.

The cases that need `out/` -- A2 and A7 -- skip on a worktree without the caches rather
than fail, exactly as `test_camera`'s standing-panel cases do.
"""
import glob
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (lint, observe, offline, parallel, pipeline,  # noqa: E402
                   prims, registry, spec as spec_mod, voices)
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

# The fixture world, the palette, the plot and the lane are `test_building`'s. One
# definition of "flat ground with a lane and a reserved doorstep", not two.
from test_building import (  # noqa: E402
    GROUND, MAT, PLOT, a_lane, builder, door_reachable, errors, world,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Reg:
    """A plot registry holding one rectangle, as a program that reserved it would."""

    claimed_this_pass = ()

    def __init__(self, plot=PLOT):
        self._p = dict(plot)

    def plots_list(self):
        return [self._p]

    def reserve(self, label, *a, **k):
        return label == self._p["label"]


def write_type(src: str, name: str = "fixture") -> str:
    d = tempfile.mkdtemp(prefix="ethoslm_type_")
    p = os.path.join(d, f"{name}.py")
    open(p, "w").write(src)
    return p


# --------------------------------------------------------------- A1. a type is a file

A1_TYPE = '''"""A cottage, as a type. The smallest thing that satisfies the contract."""
import random

FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2), "roof_ends": ("choice", ["gable", "hip"])}


def build(b, plot, seed, storeys=1, roof_ends="gable"):
    r = random.Random(seed)
    cx = (plot["x0"] + plot["x1"]) // 2
    cz = (plot["z0"] + plot["z1"]) // 2
    w = 5 + r.randrange(0, 3)
    d = 5 + r.randrange(0, 3)
    res = b.building(plot["label"], cx - w, cz - d, cx + w, cz + d, storeys,
                     {"style": "gable", "ends": roof_ends}, mat=b.voice)
    if res["ok"]:
        for (fx, fy, fz, fx1, fz1) in res["rooms"][:1]:
            b.fitting("table", fx + 1, fy + 1, fz + 1, "north",
                      mat=b.voice["floor"])
    return res
'''


@case
def t_a1_a_type_file_declares_itself_and_two_seeds_are_two_buildings():
    """The contract, end to end: read the declarations without building anything, then
        instantiate twice and get two buildings that differ, both clean and both walkable.
    """
    p = write_type(A1_TYPE)
    decl = pipeline.load_type(p)
    assert decl["form"] == "european_vernacular", decl["form"]
    assert set(decl["params"]) == {"storeys", "roof_ends"}, decl["params"]

    got = {}
    for seed in (1, 2):
        vol = world()
        b = pipeline.instantiate(p, dict(PLOT), vol, seed=seed, plots=Reg(),
                                 params={"storeys": 2, "roof_ends": "hip"})
        errs = errors(vol, b._pending)
        assert not errs, (seed, errs)
        # walkable the way the round measures it: every floor cell on the plot, on foot,
        # from outdoors -- not from the building's own door
        v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
        v.overlay(b._pending)
        ctx = lint.Context.build(v, plots=[dict(PLOT)], region=(0, 0, 55, 55))
        reach = set(ctx.nav.flood(ctx.nav.perimeter_seeds(inset=2, step=3),
                                  max_jumps=0))
        mine = [r for r in ctx.rooms if r.get("plot") == PLOT["label"]]
        cells = sum(len(set(map(tuple, r["stances"]))) for r in mine)
        walk = sum(len(set(map(tuple, r["stances"])) & reach) for r in mine)
        assert cells and walk == cells, (seed, walk, cells)
        # The *building*, not the ground under it. Everything above the floor is the
        # type's own work.
        floor_y = b.parts[-1]["floor_y"]
        mass = [(x, y, z) for (x, y, z) in b._pending if y > floor_y]
        xs = [x for (x, _y, _z) in mass]
        zs = [z for (_x, _y, z) in mass]
        ys = [y for (_x, y, _z) in mass]
        got[seed] = {"blocks": len(b._pending), "walk": f"{walk}/{cells}",
                     "footprint": (max(xs) - min(xs) + 1, max(zs) - min(zs) + 1),
                     "ridge": max(ys), "cells": frozenset(b._pending)}

    assert got[1]["cells"] != got[2]["cells"], "two seeds built the identical building"
    assert (got[1]["footprint"], got[1]["ridge"]) != \
           (got[2]["footprint"], got[2]["ridge"]), \
        "the seed moved the trim and not the massing"
    return ("; ".join(f"seed {s}: {g['blocks']} blocks, {g['footprint'][0]}x"
                      f"{g['footprint'][1]}, ridge {g['ridge']}, {g['walk']} walkable"
                      for s, g in got.items()) + "; both lint-clean")


@case
def t_a1_a_parameter_the_type_did_not_declare_is_refused_by_name():
    """And so is one out of the range the type declared, and one off its own list.

        Refused *by name*: a config that asks a cottage for `storeys=9` should be told
        which parameter and what the type actually said, not handed a building that
        quietly ignored it.
        
    """
    p = write_type(A1_TYPE)
    vol = world()
    bad = {}
    for params, want in (({"chimney": True}, "no parameter called 'chimney'"),
                         ({"storeys": 9}, "whole number from 1 to 2"),
                         ({"roof_ends": "irimoya"}, "not one of this type's choices")):
        try:
            pipeline.instantiate(p, dict(PLOT), vol, seed=1, plots=Reg(),
                                 params=params)
        except ValueError as e:
            bad[next(iter(params))] = str(e)
            assert want in str(e), (params, str(e))
            continue
        raise AssertionError(f"{params} was accepted")
    # ...and a parameter the config leaves out is filled from the declaration itself
    filled = pipeline.check_params(pipeline.load_type(p)["params"], {})
    assert filled == {"storeys": 1, "roof_ends": "gable"}, filled
    return f"{len(bad)} refusals, each naming the parameter; unset -> {filled}"


# ------------------------------------------------------- A2. the type's own checker

#: A shed with no way into it. The defect a type checker exists to find. so it stood on
#: ground the library had raised under it, had no interior at all, and the checker had
#: nothing to report. It builds from `part["floor_y"]` up, like every type does now, and
#: is sealed for the same reason it always was: no door.
A2_TYPE = '''"""A shed with no way into it. The defect a type checker exists to find."""
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 1)}


def build(b, part, seed, storeys=1):
    x0, z0 = part["x0"] + 1, part["z0"] + 1
    x1, z1 = x0 + 5, z0 + 5
    y = part["floor_y"]
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            for dy in range(1, 4):
                b.place_block(x, y + dy, z, "cobblestone" if edge else "air")
    b.roof(x0, z0, x1, z1, y + 4, "dark_oak", style="gable")
'''


def _round7():
    if not os.path.exists(offline.world_cache("site_b")):
        return None
    return pipeline.Round.load(os.path.join(ROOT, "rounds", "site_b.json"))


FIXTURE_PLOTS = ("byre", "granary", "quarry_shed", "watch_spur")


@case
def t_a2_a_type_with_no_door_reports_eight_instances_worst_first():
    """Its one type call wrote a program it could not run anywhere, and two of the three
        buildings that came out of it were broken -- 0, 2 and 4 own errors, one of them with
        no door at all. This is the same failure, on purpose, and the check has to see it on
        every plot and every seed rather than on the one it happened to be pointed at.
        
    """
    rnd = _round7()
    if rnd is None:
        return "SKIPPED."
    p = write_type(A2_TYPE)
    res = pipeline.check_type(rnd, pipeline.OfflineBackend(rnd), p,
                              list(FIXTURE_PLOTS), [1, 2])
    rows = res["rows"]
    assert len(rows) == 16, f"{len(rows)} instances, not 4 plots x 2 seeds x 2 voices"
    assert {r["plot"] for r in rows} == set(FIXTURE_PLOTS), rows
    assert {r["seed"] for r in rows} == {1, 2}, rows
    # B1: every instance stood in two voices, and a sealed shed is sealed in both; v2,
    # C0: the pair is the place's own and its derived partner, not a constant
    pair = set(pipeline.check_voices(None, rnd.voice_name()))
    assert rnd.voice_name() in pair and len(pair) == 2, pair
    assert {r["voice"] for r in rows} == pair, rows
    assert set(res["voices"]) == pair, res["voices"]
    assert all(d["entry_lines"] >= 8 and d["clean"] == 0 for d in res["voices"].values()), \
        res["voices"]
    no_entry = [f"{r['plot']}/{r['seed']}" for r in rows if not r["entry_lines"]]
    assert not no_entry, f"a sealed shed reported no way-in finding on {no_entry}"
    assert all("no door or gate anywhere on it" in " ".join(r["lines"]) for r in rows), \
        [r["lines"][:1] for r in rows]

    # worst first, by what is wrong with it and then by how little you can walk in
    score = [(r["errors"] + r["entry_lines"]) for r in rows]
    assert score == sorted(score, reverse=True), score

    text = res["text"]
    head = text.split("\n---\n")[0]
    assert head.count("\n|") >= 8, f"no per-instance table at the top:\n{head[:300]}"
    for r in rows:
        assert f"# {r['plot']}, seed {r['seed']}, voice {r['voice']}" in text, \
            (r["plot"], r["seed"], r["voice"])
    order = [ln[2:] for ln in text.splitlines()
             if ln.startswith("# ") and ", seed " in ln]
    assert order == [f"{r['plot']}, seed {r['seed']}, voice {r['voice']}" for r in rows], \
        order
    return (f"16 instances in {res['seconds']}s, {res['entry_lines']} on-foot findings, "
            f"worst first ({rows[0]['plot']}/{rows[0]['seed']}), "
            f"{len(text.splitlines())} lines of findings")


@case
def t_a2_done_is_refused_without_a_run():
    """The rule that makes the checker a loop rather than a file in a directory."""
    import inspect
    assert "A `done` written\nwithout it is not accepted." in pipeline.TYPE_CHECK_LINES
    assert pipeline.checked_runs("no-such-builder-key") == []
    # and the adopter refuses on exactly that emptiness. Quoted from the source so a
    # change to the refusal is visible here rather than only in a round that lost it.
    src = inspect.getsource(pipeline._collect_blinded)
    assert "`done` without ever running check.py" in src, "the refusal has moved"
    assert 'os.path.exists(os.path.join(d, "check.py"))' in src
    # the type checker is the same file with a different job behind it, so the same
    # `check_run` key, and therefore the same refusal
    assert "CHECK_PY.format(root=ROOT, key=key)" in inspect.getsource(
        pipeline.stage_type_check)
    return "no check_run rows -> not adopted; the type checker uses the same key"


# ------------------------------------------------- A3. the shell owns its refusals

@case
def t_a3_a_blocked_front_cell_makes_building_place_nothing():
    net = a_lane()
    vol = world()
    b = builder(vol, net)
    b.registry = Reg(dict(PLOT, x0=8, z0=13, x1=40, z1=36))
    # the lane's own rail, standing in the cell the reserved door opens onto
    b.place_block(20, GROUND + 1, 15, "cobblestone_wall")
    before = dict(b._pending)
    res = b.building("hut", 14, 16, 26, 24, 2, "gable", mat=MAT)
    assert not res["ok"], "a building with no way into it was built"
    assert res["cells"] == 0, res["cells"]
    assert b._pending == before, \
        f"a refused building left {len(set(b._pending) ^ set(before))} cells standing"
    assert not b._step_queue(), "a refused building left treads queued"
    assert "cobblestone_wall" in res["reason"], res["reason"]
    assert res["front"] == [20, GROUND + 1, 15], res.get("front")
    assert f"({20},{GROUND + 1},{15})" in res["reason"], res["reason"]

    # ...and with the rail gone the identical call builds
    b2 = builder(world(), net)
    b2.registry = Reg(dict(PLOT, x0=8, z0=13, x1=40, z1=36))
    ok = b2.building("hut", 14, 16, 26, 24, 2, "gable", mat=MAT)
    assert ok["ok"], ok["reason"]
    return res["reason"][:96]


# ------------------------------------------ A4. a two-tier roof is not a lantern

def _cavities(pending: dict, xs, zs, band) -> list:
    """Cells of air *inside* a mass, column by column: anything not placed below the
    topmost thing that is. Air above the roof surface is sky and is not a cavity, which
    is the whole difference between reading this right and reading the eaves as a hole.
    """
    out = []
    for x in xs:
        for z in zs:
            col = [y for y in band
                   if pending.get((x, y, z), "air").split("[")[0]
                   not in ("air", "cave_air", "void_air")]
            if not col:
                continue
            out += [(x, y, z) for y in range(min(band), max(col))
                    if pending.get((x, y, z), "air").split("[")[0]
                    in ("air", "cave_air", "void_air")]
    return out


@case
def t_a4_two_tiers_leave_no_room_and_no_void_between_them():
    """`roof(tiers=2)` used to hand back a cavity between its tiers.

        A tier drawn short of its own eave places nothing inside the cut, so the rectangle
        between the lower roof and the upper one was air with a wall course round it: W007
        called it a void and the room finder called it a room. It is the inside of a roof.
        
    """
    vol = world()
    b = builder(vol)
    b.place_cuboid(14, GROUND + 1, 14, 26, GROUND + 4, 24, "stone_bricks")
    b.place_cuboid(15, GROUND + 1, 15, 25, GROUND + 4, 23, "air")
    ridge = b.roof(14, 14, 26, 24, GROUND + 5, "deepslate_tile", style="hip",
                   pitch=(1, 2), ends=("irimoya", "irimoya"), eave="upturned", tiers=2)
    v = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(b._pending)
    ctx = lint.Context.build(v, plots=[dict(PLOT)], region=(0, 0, 55, 55))
    rep = lint.lint(ctx)

    # The geometric statement, which is the one that discriminates: the rectangle the
    # upper tier stands on has no air under it. Before the fix the lower tier placed
    # nothing inside its own cut and this column was open from the eave to the tier
    # above it, all the way round.
    band = range(GROUND + 6, ridge + 1)
    air = _cavities(b._pending, range(16, 25), range(16, 23), band)
    assert not air, f"{len(air)} cells of cavity between the tiers, e.g. {air[:3]}"

    # ...and neither instrument that would have reported it does
    rooms = [r for r in ctx.rooms if r["bbox"][1] in band
             and (r.get("enclosure") or 0) >= lint.Context.ENCLOSED]
    voids = [f for f in rep.findings if f.code == "W007" and f.pos
             and f.pos[1] in band]
    assert not rooms, f"{len(rooms)} room(s) between the tiers: {rooms[:1]}"
    assert not voids, f"{len(voids)} W007 void(s) between the tiers: {voids[:1]}"
    assert not observe.unsupported(v, region=(10, 10, 34, 30)), "the roof floats"

    # ...and the same roof with solid_fill off is still allowed to be hollow, which is
    # what makes the case above about the fix rather than about `roof()` filling
    # everything it touches
    b2 = builder(world())
    b2.roof(14, 14, 26, 24, GROUND + 5, "deepslate_tile", style="hip", tiers=2,
            pitch=(1, 2), ends=("irimoya", "irimoya"), eave="upturned",
            solid_fill=False)
    hollow = _cavities(b2._pending, range(16, 25), range(16, 23), band)
    assert hollow, "solid_fill=False filled it anyway; the case discriminates nothing"
    return (f"ridge {ridge}, {len(b._pending)} blocks, no cavity in the "
            f"{len(band)} courses between the tiers; hollow leaves {len(hollow)}")


# ------------------------------------------------------------ A5. material refuses

@case
def t_a5_an_unknown_family_raises_and_thatch_is_a_family():
    try:
        prims.material("hay")
    except ValueError as e:
        assert "'hay'" in str(e) and "not a material family" in str(e), str(e)
    else:
        raise AssertionError("material('hay') did not refuse")
    try:
        prims.material("hay_block")
    except ValueError as e:
        assert "'hay_block'" in str(e), str(e)
    else:
        raise AssertionError("material('hay_block') did not refuse")

    assert prims.material("thatch") == ("hay_block", "bamboo_stairs", "bamboo_slab")
    assert prims.material("bamboo") == ("bamboo_planks", "bamboo_stairs", "bamboo_slab")
    # a block id is its own material said differently, and is not an error
    assert prims.material("stripped_spruce_log") == prims.material("spruce")
    assert prims.material("stone_bricks") == prims.material("stone_brick")
    assert prims.family("dark_oak_planks") == "dark_oak", "it stripped on to oak"
    # ...and where only a cube is wanted, a plain block id is allowed through
    assert prims.solid("white_concrete_powder") == "white_concrete_powder"
    assert prims.solid("thatch") == "hay_block"

    # the family has to be visible to the instruments, or it is not a family: bamboo
    # stairs and slabs used to read as bamboo shoots and a thatched roof was invisible
    st = observe._classify("bamboo_stairs", {"facing": "east", "half": "bottom"})
    sl = observe._classify("bamboo_slab", {"type": "bottom"})
    assert st[3] == "stairs" and sl[3] == "slab", (st, sl)
    assert observe._classify("bamboo", {})[3] == "passable", "a shoot became a block"

    from ethoslm import styles
    assert styles.VOICES["japanese_minka"]["palette"]["roof"] == "thatch"
    return "hay refused, thatch and bamboo are families, and the checker can see them"


# --------------------------------------------------------------- A6. the podium

@case
def t_a6_a_hall_on_a_podium_is_walk_reachable_from_the_lane():
    """`platform=N` is the library saying it, with the steps down included."""
    net = a_lane()
    vol = world()
    b = builder(vol, net)
    b.registry = Reg(dict(PLOT, x0=8, z0=13, x1=40, z1=36))
    res = b.building("hut", 14, 17, 26, 25, 1, {"style": "hip", "ends": "irimoya"},
                     mat=MAT, platform=2)
    assert res["ok"], res["reason"]
    b.resolve_steps()
    pf = res["extras"]["platform"]
    assert pf["ok"] and pf["courses"] == 2, pf
    assert pf["steps"]["ok"], pf["steps"]["reason"]
    assert res["floors"][0] == pf["grade_y"] + 2, (res["floors"], pf)

    assert not errors(vol, b._pending, plots=[dict(PLOT, x0=8, z0=13, x1=40, z1=36)]), \
        errors(vol, b._pending, plots=[dict(PLOT, x0=8, z0=13, x1=40, z1=36)])
    assert door_reachable(vol, b._pending, res["door"]), res["approach"]["reason"]
    w = b.check_walkable(x0=14, z0=17, x1=26, z1=25)
    cells = sum(r["cells"] for r in w["rooms"])
    walk = sum(r["walkable"] for r in w["rooms"])
    assert cells and walk == cells, f"only {walk} of {cells} floor cells are walkable"

    # ...and the same building without the podium sits on the ground, so the case is
    # about the podium and not about the plot
    flat = builder(world(), net)
    flat.registry = Reg(dict(PLOT, x0=8, z0=13, x1=40, z1=36))
    fr = flat.building("hut", 14, 17, 26, 25, 1,
                       {"style": "hip", "ends": "irimoya"}, mat=MAT)
    assert fr["floors"][0] == res["floors"][0] - 2, (fr["floors"], res["floors"])
    return (f"podium {pf['courses']} high, floor y={res['floors'][0]} over ground "
            f"y={pf['grade_y']}, {pf['steps']['cells']} treads down, "
            f"{walk}/{cells} walkable, door reachable")


# ------------------------------------------------------------- A7. readout as a stage

#: The bar rows rounds 10 and 11 recorded, from their own `readout.json` as the round
#: was closed. `stage_readout` has to reproduce them.
RECORDED = {
    "site_e": {"walk_from_outdoors_pct": 84.3, "e002_walk_only_from_the_lane": 0,
                "own_lint_errors": 0, "builder_calls": 10},
    "site_f": {"walk_from_outdoors_pct": 87.3, "e002_walk_only": 0,
                "own_lint_errors": 1, "builder_calls": 10, "program_lines": 500.8,
                "writes_covered_over": 23.8},
}

#: Rows the instruments have superseded since they were recorded, each with the value
#: the current instrument gives and the change that moved it. **None of them is a bar
#: moving.** A bar is a threshold; every threshold here is the one that was registered,
#: and what changed is the ruler. Every solid inside it is floor and the number is 87.3
#: again, which is exactly what it read under walk model 3. It went with us and it is
#: gone; the two that go against us are still here.
SUPERSEDED = {
    ("superseded by a named correction", "walk_from_outdoors_pct"): (
        71.8, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        8, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        37, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        10, "superseded by a named correction"),
    ("superseded by a named correction", "walk_from_outdoors_pct"): (
        98.1, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        10, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        29, "superseded by a named correction"),
    ("superseded by a named correction", "own_lint_errors"): (
        33, "superseded by a named correction"),
    ("superseded by a named correction", "walk_from_outdoors_pct"): (
        92.9, "superseded by a named correction"),
    ("superseded by a named correction", "writes_covered_over"): (
        22.5, "superseded by a named correction"),
}


def _shadow_state(name: str) -> str:
    """A round's state directory, symlinked entry by entry into a temp directory.

        The readout is *written*, and rounds 9 to 12 each ended with one made by a script
        that no longer exists and that `out/` does not keep in Git. So this case reads the
        real caches and writes nowhere near them.
        
    """
    src = os.path.join(ROOT, "out", name)
    d = tempfile.mkdtemp(prefix=f"ethoslm_readout_{name}_")
    for f in os.listdir(src):
        os.symlink(os.path.join(src, f), os.path.join(d, f))
    return d


@case
def t_a7_the_readout_stage_reproduces_rounds_10_and_11():
    """Four readout scripts became one stage. It has to say what they said.

        The bars are read off each round's own pre-registration, which was written before
        the first block was placed; each names a measure from the fixed registry; the driver
        computes it. The rows that are superseded are named above with the change that moved
        them -- reported, never quietly re-fitted.
    """
    from ethoslm import observe
    got, moved, superseded = {}, [], []
    for name, want in RECORDED.items():
        if not os.path.exists(os.path.join(ROOT, "out", name, "world_built.npz")):
            return f"SKIPPED -- no out/{name} in this worktree"
        rnd = pipeline.Round.load(os.path.join(ROOT, "rounds", f"{name}.json"))
        rnd.state_dir = _shadow_state(name)
        res = pipeline.stage_readout(rnd, pipeline.OfflineBackend(rnd), {})
        assert set(res["results"]) == set(want), (name, sorted(res["results"]))
        assert res["walk_model"] == observe.WALK_MODEL, res.get("walk_model")
        for key, recorded in want.items():
            row = res["results"][key]
            assert "unreadable" not in row and "error" not in row, (name, key, row)
            assert row["measure"] in pipeline.MEASURES, (name, key, row["measure"])
            if row["measure"] == "walk_from_outdoors_pct":
                assert row["walk_model"] == observe.WALK_MODEL, row
                assert row["fitting_registry"] is False, \
                    ("a standing town has no furniture register on disk; a row that "
                     "claims one is reading something else", name, row)
            now, why = SUPERSEDED.get((name, key), (recorded, None))
            if row["got"] != now:
                moved.append(f"{name}.{key}: {row['got']} != {now}")
            if why:
                superseded.append(f"{name}.{key} {recorded} -> {now}")
            got[f"{name}.{key}"] = row["got"]
        assert os.path.exists(res["written"]), res["written"]
        shutil.rmtree(rnd.state_dir, ignore_errors=True)
    assert not moved, f"rows the stage no longer reproduces: {moved}"
    return (f"{len(got)} bar rows over two rounds, {len(got) - len(superseded)} "
            f"identical to the record and {len(superseded)} superseded by a named "
            f"change ({'; '.join(superseded)})")


@case
def t_a7_a_bar_names_a_measure_and_a_number_and_both_are_read():
    """The registry, and the parse that reads a bar written the way a person writes
    one. Every form on disk in the rounds that already ran, plus the explicit form a
    new round should use."""
    want = {">= 90": (">=", 90.0), "0 findings": ("==", 0.0), "0": ("==", 0.0),
            "<= 2 x waves = 10": ("<=", 10.0), "<= 200 mean per wave": ("<=", 200.0),
            "<= 15%": ("<=", 15.0), "<= 25,000": ("<=", 25.0),
            ">= 11 of 12": (">=", 11.0)}
    for text, expect in want.items():
        assert pipeline._bar_value(text) == expect, (text, pipeline._bar_value(text))
    assert pipeline._bar_value({"comparator": ">=", "value": 11}) == (">=", 11.0)
    assert pipeline._meets(">=", 11, 11.0) and not pipeline._meets(">=", 10, 11.0)
    assert pipeline._meets("<=", 24999, 25000.0)
    assert pipeline._meets(">=", None, 1.0) is None, "a missing number is not a pass"
    # Five bars were registered and the seven-name registry answers three of them.
    assert set(pipeline.MEASURES) == {
        "walk_from_outdoors_pct", "e002_from_the_lane", "own_lint_errors",
        "builder_calls", "program_lines", "writes_covered_over",
        "tokens_per_building", "instances_lint_zero", "instances_walkable",
        "within_type_variation", "building_calls_per_type",
        # The five a *place* needs and a settlement did not.
        "structures", "hand_programs", "enclosure", "tokens_whole_round",
        # Designed ground's five clauses, the levels, what every preflight said and what
        # the search cost, reported beside the bars
        "ground", "levels", "preflights", "search_cost",
        "whole_place_lint_seconds",
        "instantiation",
        "place_read", "site_chosen",
        # belt share, ring coverage, a centred compound, palettes standing, a dry core
        # -- read off the layout, the parts record and the built volume, reported and
        # not barred.
        "concentric",
        # v2, C5: the fabric of the compiled districts -- columns of ground per house
        # per density word, the share of houses whose way in is on the front their leaf
        # names, the leftover ground assigned, and party walls where the character said
        # attached.
        "fabric",
        # ground per structure under the density word's ceiling, plots and areas over
        # the registered cover, the open ground the setting's rather than the footing's,
        # and the centre over its share of the innermost ring.
        "occupancy"}, \
        sorted(pipeline.MEASURES)
    return (f"{len(want)} bar forms parsed, {len(pipeline.MEASURES)} measures in the "
            f"registry")


# ------------------------------------------------------- A8. the shot list moves home

@case
def t_a8_the_three_experiment_scripts_are_gone_and_nothing_imports_them():
    """`pipeline` reached the shot list through `scripts/step4_render.py`, which reached
    it through `e1d_shotlist.py`, which reached `audit_mutants.py` -- so running a round
    imported three finished experiments. What it wanted lives in `render` now."""
    import ast
    from ethoslm import render
    scripts = os.path.join(ROOT, "scripts")
    gone = ("step4_render.py", "e1d_shotlist.py", "audit_mutants.py")
    left = [f for f in gone if os.path.exists(os.path.join(scripts, f))]
    assert not left, f"still on disk: {left}"

    for fn in ("merged_plots", "built_cells", "mid_y", "cache_built", "town_aerials"):
        assert callable(getattr(render, fn, None)), fn

    offenders = []
    for d in (os.path.join(ROOT, "src", "ethoslm"), scripts):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(d, f)).read())
            for n in ast.walk(tree):
                names = ([a.name for a in n.names] if isinstance(n, ast.Import)
                         else [n.module or ""] if isinstance(n, ast.ImportFrom)
                         else [])
                for nm in names:
                    if nm.split(".")[0] in {g[:-3] for g in gone}:
                        offenders.append(f"{f}:{n.lineno} {nm}")
    assert not offenders, f"still imported: {offenders}"

    # and the geometry did not move: the four functions still frame the same subject
    state = os.path.join(ROOT, "out", "site_b")
    if not os.path.exists(os.path.join(state, "world_built.npz")):
        return "3 scripts gone, 5 functions in render; SKIPPED the geometry check"
    built = offline.load_volume(os.path.join(state, "world_built.npz"))
    ps = render.merged_plots(state)
    ys = {p["label"]: render.mid_y(built, p) for p in ps}
    assert len(ps) == 14, f"{len(ps)} merged plots, not 14"
    assert all(isinstance(v, float) for v in ys.values()), ys
    return (f"3 scripts gone, nothing imports them, {len(ps)} merged plots and "
            f"mid_y on all of them (see test_camera for the 120 unmoved panels)")


# A type says what it needs `PARAMS` says what may be varied and nothing said what a
# `build()` requires to stand at all. The planner drew twenty-nine 6x6 plots, `site()`
# insets two on every side, and all twenty-two `townhouse` instances on the resulting
# 4x4 pads refused by name after the ground had been prepared for them.

@case
def t_a1_every_type_declares_needs_and_it_is_the_measured_band():
    """Every file under `types/` declares `NEEDS`, and its footprint is what was measured.

        The declaration is not a claim somebody made up: `scripts/type_needs.py` sweeps every
        size, both seeds and every combination of the type's own `PARAMS` on flat ground and
        writes the band to `rounds/type-needs.json`. This is the assertion that the file and
        the measurement have not drifted apart -- re-run the sweep and it says which moved.
        
    """
    bank = json.load(open(os.path.join(ROOT, "rounds", "type-needs.json")))
    d = os.path.join(ROOT, "types")
    files = sorted(f for f in os.listdir(d) if f.endswith(".py"))
    assert files, "no types on disk"
    for f in files:
        name = f[:-3]
        decl = pipeline.load_type(os.path.join(d, f))
        assert decl["declares_needs"], f"types/{f} declares no NEEDS"
        assert name in bank["types"], f"types/{f} is not in rounds/type-needs.json"
        band = bank["types"][name]["band"]
        want = band["footprint"]
        got = list(decl["needs"]["footprint"])
        # Inside the measured envelope, not necessarily equal to it: a type is allowed
        # to ask for less ground than it was measured to stand on, and claiming more is
        # the only thing that can be false.
        assert want, f"types/{f}: no size passed the sweep at all"
        assert got[0] >= want[0] and got[1] >= want[1] \
            and got[2] <= want[2] and got[3] <= want[3], \
            (f"types/{f} declares a footprint of {got} and the measured band is {want}: "
             f"it claims ground the sweep did not stand it on")
        # ...and every size inside what it declares that the sweep found broken is named
        # on the file, so the plan refuses it. The declaration is the whole of the
        # measurement, an envelope and its exceptions, rather than the one clean run
        # that held a fixture.
        lo, hi = min(got[0], got[1]), max(got[2], got[3])
        dirty = sorted(v for v in (band.get("except") or []) if lo <= v <= hi)
        named = sorted(decl["needs"].get("except") or ())
        missing = [v for v in dirty if v not in named]
        assert not missing, \
            (f"types/{f} declares {got} and names {named} as broken; the sweep found "
             f"it broken at {dirty} inside that, and {missing} is not named")
        # an edge is the one thing a lane is never routed to
        assert (decl["needs"]["frontage"] == "any") == (decl["kind"] == "edge"), \
            f"types/{f}: frontage {decl['needs']['frontage']} for a {decl['kind']}"
    return (f"{len(files)} types declare NEEDS, all of them the band "
            f"scripts/type_needs.py measured ({bank['swept']['seeds']} seeds, every "
            f"parameter combination)")


@case
def t_r2_stone_is_a_material_family_and_a_bare_block_has_a_cube():
    """The eleventh defect that is one missing name.

        A family is a material this game has in all three shapes, and `stone` is one:
        `stone_stairs` and `stone_slab` have existed since 1.14 and `MATERIALS` never listed
        it. Nothing noticed because no hand-written voice ever asked -- but the one voice a
        model has ever written did. `ochre_stone_green_tile` names `smooth_stone` for its
        floor, `smooth_stone` is the smooth face of stone, and `family()` stripped the
        prefix and found nothing. Five committed types pass `part["voice"]["floor"]` to
        `material()` through `fitting()`, `steps()` and `b.block()`, so **the city could not
        be built in the palette it wrote for itself**, and nothing ever tried.

        The second half is the role the voice contract already documents: `roles.floor`
        "need not be a family, because it is only ever laid as a cube". `b.block(x, "full")`
        refused every such block, which made the exemption unusable. It asks the game now
        and it still refuses by name -- `white_concrete_powder` has a cube and no slab.
        
    """
    assert "stone" in prims.MATERIALS, "stone is still not a family"
    assert prims.family("smooth_stone") == "stone", prims.family("smooth_stone")
    assert prims.shape("smooth_stone", "fine") == "smooth_stone"
    assert prims.shape("smooth_stone", "slab") == "stone_slab"
    assert prims.shape("stone", "stairs") == "stone_stairs"
    # ...and the older names still resolve where they always did
    assert prims.family("stone_bricks") == "stone_brick"
    assert prims.family("smooth_stone_slab") == "stone"
    assert prims.family("hay_block") is None, "hay grew stairs"
    # a bare block that belongs to no family: a cube, and a refusal for anything else
    assert prims.shape("white_concrete_powder", "full") == "white_concrete_powder"
    for kind in ("slab", "stairs", "fence"):
        try:
            prims.shape("white_concrete_powder", kind)
        except ValueError as e:
            assert "white_concrete_powder" in str(e) and kind in str(e), e
        else:
            raise AssertionError(f"a {kind} of white_concrete_powder was invented")
    # ...and the voice that found it now loads with every role shapeable
    v = voices.load("ochre_stone_green_tile")
    for role, fam in v["roles"].items():
        assert prims.shape(fam, "full"), (role, fam)
    return (f"{len(prims.MATERIALS)} families, stone among them; smooth_stone is the "
            f"fine face of stone and its slab is stone_slab; a bare block has a cube "
            f"and no slab; all six roles of ochre_stone_green_tile are shapeable")


@case
def t_r2a1_every_type_declares_what_it_is_for():
    """Every file under `types/` declares `ROLE`, and it is one of four words. A2.

        `load_type` leaves it optional for `NEEDS`' reason -- a type written before A2 says
        nothing and is admissible anywhere, which is what the plan did before A2 existed --
        so this is what holds the committed files to declaring one. A role that is only a
        default is a role nobody chose.
        
    """
    d = os.path.join(ROOT, "types")
    files = sorted(f for f in os.listdir(d) if f.endswith(".py"))
    got = {}
    for f in files:
        decl = pipeline.load_type(os.path.join(d, f))
        assert decl.get("role"), f"types/{f} declares no ROLE"
        assert decl["role"] in pipeline.ROLES, f"types/{f}: ROLE {decl['role']!r}"
        got[f[:-3]] = decl["role"]
    # ...and a word that is not one of the four is refused by name, not defaulted
    try:
        pipeline.read_role({"ROLE": "residential"}, where="a fixture")
    except ValueError as e:
        assert "residential" in str(e) and "urban" in str(e), e
    else:
        raise AssertionError("an undeclared role word was accepted")
    # ...and the filter is a filter: a rural type is refused in an urban district and a
    # civic one stands in either.
    assert not pipeline.role_ok("rural", "urban"), "a farmhouse passed as urban"
    assert pipeline.role_ok("civic", "urban") and pipeline.role_ok("civic", "rural"), \
        "a temple was refused somewhere"
    assert pipeline.role_ok("rural", None), "a district with no role refused something"
    by = {}
    for name, role in got.items():
        by.setdefault(role, []).append(name)
    return (f"{len(files)} types declare ROLE: "
            + "; ".join(f"{r} {sorted(v)}" for r, v in sorted(by.items())))


@case
def t_c0_the_checkers_partner_voice_is_the_places_own_and_never_a_named_constant():
    """v2, C0. The second voice a type's checker stands it in is the voice of the place
    the round is in; where the author's voice is the place's, or there is no place, it
    is the voice on disk whose silhouette is least like the first's -- derived from
    the directory, so a voice added to it changes the answer and no file names one."""
    from ethoslm import styles
    names = sorted(styles.VOICES)
    # the place's voice is the partner, whatever the author was given
    for author in names:
        for place in names:
            if author == place:
                continue
            got = pipeline.check_voices(author, place)
            assert got == [author, place], (author, place, got)
    # the author's voice is the place's: the partner is derived, explicit, and unlike
    for v in names:
        got = pipeline.check_voices(v, v)
        assert got[0] == v and got[1] != v, got
        assert styles.explicit_silhouette(got[1]), got
        if styles.explicit_silhouette(v):
            assert styles.silhouette_distance(v, got[1]) >= 1, got
        assert got == pipeline.check_voices(v, None), (v, got)
    # no voice at all: the plainest voice on disk and its partner
    first, second = pipeline.check_voices(None, None)
    assert first == styles.silent_voice() and not styles.explicit_silhouette(first)
    assert styles.explicit_silhouette(second)
    # no voice, a place: the place's and its partner
    assert pipeline.check_voices(None, names[0]) == [names[0],
                                                     styles.partner_voice(names[0])]
    # the answer moves with the directory: a voice added with a silhouette unlike every
    # other becomes the partner of the ones it is least like
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="voices-")
    try:
        for n in names:
            shutil.copy(os.path.join(ROOT, "voices", f"{n}.json"), d)
        src = json.load(open(os.path.join(ROOT, "voices", "japanese_temple.json")))
        src["name"] = "zz_pagoda"
        src["roof"] = {"ends": "gable", "eave": "straight", "profile": [[2, 1]],
                       "tiers": 3}
        json.dump(src, open(os.path.join(d, "zz_pagoda.json"), "w"))
        table = styles._VoiceTable(d)
        old = styles.VOICES
        styles.VOICES = table
        try:
            # **The base voice is found, not named.** It used to be `japanese_temple`.
            # The composition round gave `packed_earth_and_dark_tile` a straight eave --
            # a crowded ring cannot afford a turned-up one -- which put it at the
            # *maximum* silhouette distance from the temple as well, so on that base the
            # added voice only ties and the tie is broken by name. The claim being made
            # is that a voice added to the directory becomes the partner of the ones it
            # is least like, so the base is a voice `zz_pagoda` is uniquely furthest
            # from, and the failure names which one was tried.
            def _farthest(v):
                pool = [o for o in sorted(table)
                        if o != v and styles.explicit_silhouette(o)]
                top = max(styles.silhouette_distance(v, o) for o in pool)
                return [o for o in pool if styles.silhouette_distance(v, o) == top]
            base = next(v for v in sorted(table)
                        if v != "zz_pagoda" and styles.explicit_silhouette(v)
                        and _farthest(v) == ["zz_pagoda"])
            got = styles.partner_voice(base)
            assert got == "zz_pagoda", (base, got)
        finally:
            styles.VOICES = old
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # ...and nothing in the pipeline names a voice as the partner
    assert not hasattr(pipeline, "SILHOUETTE_PAIR") and \
        not hasattr(pipeline, "SILHOUETTE_ALT")
    return (f"{len(names)} voices: the place's is the partner of any other; "
            f"the derived partners are "
            + ", ".join(f"{v}->{styles.partner_voice(v)}" for v in names
                        if styles.explicit_silhouette(v))
            + f"; no voice at all is {first} and {second}")


@case
def t_a1_a_pad_too_small_is_refused_before_the_ground_is_touched():
    """A plot two cells too small on each axis: refused by name, world untouched."""
    from ethoslm import place
    from ethoslm.buildlib import Builder

    class _Be:
        live = False
        volume = None

        def commit(self, b):
            raise AssertionError("a refused part reached the world")

    decl = pipeline.load_type(os.path.join(ROOT, "types", "townhouse.py"))
    lo_w, lo_d = decl["needs"]["footprint"][:2]
    # the plot that gives a pad two cells short on each axis
    small = lo_w + 2 * Builder.SITE_INSET - 2
    part = {"name": "too_small", "kind": "plot", "type": "townhouse", "seed": 1,
            "x0": 0, "z0": 0, "x1": small - 1, "z1": small - 1}
    big = {**part, "name": "big_enough",
           "x1": lo_w + 2 * Builder.SITE_INSET - 1,
           "z1": lo_d + 2 * Builder.SITE_INSET - 1}

    d = tempfile.mkdtemp(prefix="ethoslm_needs_")
    rnd = pipeline.Round(name="needs", state_dir=d)
    rec = place.instantiate_part(rnd, _Be(), part, None)
    inset = 2 * Builder.SITE_INSET
    assert rec["status"] == "refused", rec
    assert "townhouse" in rec["error"], rec["error"]
    # the prose names the plot the planner drew and the plot the type needs at least.
    assert f"{lo_w + inset}x{lo_d + inset}" in rec["error"] \
        and f"{small}x{small}" in rec["error"], rec["error"]
    assert "pad of" in rec["error"], rec["error"]
    # ...and the record is in pads, which is what the library sites
    assert rec["pad"] == [lo_w - 2, lo_d - 2], rec["pad"]
    assert rec["needs"] == list(decl["needs"]["footprint"]), rec["needs"]
    assert not os.path.exists(os.path.join(d, "parts")), \
        "a refused part wrote a program"
    # ...and the same type on the smallest plot its NEEDS allow is not refused here
    assert pipeline.needs_footprint_failure(big, decl["needs"]) is None, \
        pipeline.needs_footprint_failure(big, decl["needs"])
    return (f"a {small}x{small} plot gives a {lo_w - 2}x{lo_d - 2} pad and townhouse "
            f"needs {lo_w}x{lo_d}: refused before site(), nothing written, the reason "
            f"in plots ({lo_w + inset}x{lo_d + inset}) and the record in pads; "
            f"{lo_w + inset}x{lo_d + inset} passes")


# Forms, voices, parallel

def _stand(name: str, voice: str, size: int, seed: int = 1, params=None,
           part_over: dict | None = None) -> dict:
    """One instance of one committed type, in one voice, on flat ground.

        Everything is fixed except the voice: the same pad, the same seed, the same
        parameters. What comes back is the set of columns written and the blocks written in
        them, which is exactly the two halves A1 separates.
        
    """
    import collections
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
    kind = decl["kind"]
    mat = pipeline.voice_palette(voice)
    roof = pipeline.voice_roof(voice)
    span = size + 45
    if kind == "plot":
        w = size + (2 if size < 7 else 4)
        part = {"label": "t", "kind": "plot", "x0": 20, "z0": 20,
                "x1": 19 + w, "z1": 19 + w}
    elif kind == "area":
        part = {"label": "t", "kind": "area", "x0": 20, "z0": 20,
                "x1": 19 + size, "z1": 19 + size}
    elif kind == "point":
        part = {"label": "t", "kind": "point", "at": [30, 30], "facing": "north",
                "size": size}
        span = size + 65
    else:
        part = {"label": "t", "kind": "edge", "width": 1,
                "path": [[20, 20], [19 + size, 20]]}
        span = size + 55
    # `part_over` is what the **plan** would have written on the part rather than what
    # the type is parameterised with -- an edge's own width, a point's wall. The craft
    # round, E4: a wall's mass is the part's and never a parameter.
    part.update(part_over or {})
    x0, z0, x1, z1 = pipeline.part_rect(part)
    vol = _flat_world(span)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = None
    b.registry = _OnePlot({"label": "t", "x0": x0, "z0": z0, "x1": x1, "z1": z1})
    sited = b.site(dict(part), mat=mat, roof=roof)
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)                  # noqa: S102
    res = ns["build"](b.type_builder(sited), sited, seed, **(params or {}))
    b.resolve_steps()
    return {"ok": bool(res and res.get("ok", True)),
            "shape": tuple(sorted(b._pending)),
            "blocks": collections.Counter(b._pending.values()),
            "bad": registry.check_all(set(b._pending.values()))}


def _flat_world(size: int, y: int = 64):
    y0, y1 = y - 14, y + 56
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return observe.Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


class _OnePlot:
    def __init__(self, plot):
        self.plots = [dict(plot)]
        self.claimed_this_pass = [dict(plot)]

    def plots_list(self):
        return [dict(p) for p in self.plots]

    def reserve(self, *_a, **_k):
        return True


def every_type() -> list:
    """Every file under `types/`, by name. **Not a list kept here**. An instrument with a
    hard-coded list of subjects goes blind the moment somebody adds one.
    """
    import glob
    return sorted(os.path.basename(f)[:-3]
                  for f in glob.glob(os.path.join(ROOT, "types", "*.py")))


def size_for(name: str) -> int:
    """A size inside the band the type's own `NEEDS` declares, by rule, per kind: a
    plot's pad near 9, an area near 12, a point near 5, an edge's run at 40 or as long
    as it may be. The band is the file's, so a type that moves its band moves its
    fixture with it and nothing here has to be told."""
    decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
    lo_w, lo_d, hi_w, hi_d = decl["needs"]["footprint"]
    kind = decl["kind"]
    if kind == "edge":
        return min(hi_d, 40)
    want = {"plot": 9, "area": 12, "point": 5}[kind]
    return min(hi_w, max(lo_w, want))


@case
def t_r19_a1_a_type_is_a_form_and_the_same_building_stands_in_any_voice():
    """A1's whole claim, on every file under `types/`: the palette is the settlement's
        and, since the voice contract, so is the roof's silhouette.

        Each type is stood three times on the same pad at the same seed:

        Plus E014 and E015, the static halves: a type that names a material or a
        silhouette where the build API takes one is refused before it runs, and every
        committed file is clean of both.
        
    """
    rows, moved, same_roof, off = [], [], [], []
    for name in every_type():
        size = size_for(name)
        a = _stand(name, "drystone_and_thatch", size)
        b = _stand(name, "white_render_dark_frame", size)
        c = _stand(name, "japanese_temple", size)
        for v, r in (("drystone_and_thatch", a), ("white_render_dark_frame", b),
                     ("japanese_temple", c)):
            assert r["ok"], (name, v, size)
            assert not r["bad"], (name, v, r["bad"])
        if a["shape"] != b["shape"]:
            moved.append((name, len(set(a["shape"]) ^ set(b["shape"]))))
        assert a["blocks"] != b["blocks"], \
            f"{name} places the same blocks in both silent voices: it is not reading the voice"
        cols_a = {(x, z) for (x, _y, z) in a["shape"]}
        cols_c = {(x, z) for (x, _y, z) in c["shape"]}
        if cols_a != cols_c:
            off.append((name, len(cols_a ^ cols_c)))
        if c["shape"] == a["shape"]:
            same_roof.append(name)
        rows.append((name, size, len(a["shape"]),
                     len(set(a["blocks"]) ^ set(b["blocks"]))))
    assert not moved, f"these types change shape between two silent voices: {moved}"
    assert not off, f"these types move their footprint with the silhouette: {off}"
    # A type with no roof at all -- a field, a square's paving, a wall -- is the same
    # shape under any silhouette, legitimately. What this asserts is that the silhouette
    # reaches every type that has a roof: the reference type, which is one `building()`
    # call, is the proof that `TypeBuilder` hands it over.
    assert "_reference" not in same_roof, "the voice's silhouette did not reach building()"

    # ...and the static half, on the same files
    static = {}
    for name in every_type():
        src = open(os.path.join(ROOT, "types", f"{name}.py")).read()
        found = [f for f in lint.preflight(src, allow_try=True, palette=True).findings
                 if f.code in ("E014", "E015")]
        if found:
            static[name] = [f"{f.code} line {f.detail['line']}" for f in found]
    assert not static, f"a committed type names a material or a silhouette: {static}"
    # ...which fires on one that does
    welded = lint.preflight(
        'FORM = "civic"\nPARAMS = {}\n\n'
        'def build(b, part, seed):\n'
        '    b.place_block(1, 2, 3, "cobblestone")\n'
        '    b.roof(0, 0, 4, 4, 5, "deepslate_tile")\n',
        allow_try=True, palette=True)
    codes = [f.code for f in welded.findings]
    fams = sorted(f.detail["family"] for f in welded.findings if f.code == "E014")
    assert codes.count("E014") == 2 and fams == ["cobblestone", "deepslate_tile"], \
        [f.message for f in welded.findings]
    return (f"{len(rows)} files under types/, each stood in three voices on the same "
            f"pad at the same seed: {sum(r[2] for r in rows)} columns, byte-identical "
            f"in shape between the two silent voices in all {len(rows)}, on the same "
            f"footprint under the temple silhouette in all {len(rows)} and a different "
            f"roof in {len(rows) - len(same_roof)}; 0 E014 and 0 E015 across types/")


@case
def t_vc_b2_a_type_that_names_a_silhouette_is_refused_at_preflight():
    """B2 of the voice contract: E015, judged by position like E014.

        A type passing `ends="irimoya"`, `eave="upturned"`, `tiers=2`, a `pitch` or a
        `profile` to `roof()` -- or the same keys in the dict `building(roof=...)` takes --
        is refused before it runs, because the silhouette is the voice's and `TypeBuilder`
        hands it to those two calls from `part['roof']`. `style` and `axis` are the type's:
        a lean-to is a shed by construction, and the voice's ends and profile override a
        style's own wherever the voice has them. One that omits them, taking the voice's,
        passes. `ends=None` is `roof()`'s own "the style's ends" and is not a literal.
        
    """
    def pre(body):
        return lint.preflight('FORM = "civic"\nPARAMS = {}\n\ndef build(b, part, seed):\n'
                              + body, allow_try=True, palette=True)
    welded = pre('    b.roof(0, 0, 8, 8, 5, part["voice"]["roof"], ends="irimoya", '
                 'eave="upturned", tiers=2, profile=[(1, 2), (2, 1)])\n'
                 '    b.building("h", 0, 0, 8, 8, 1, {"style": "gable", "axis": "x", '
                 '"pitch": (2, 1)}, mat=part["voice"])\n'
                 '    b.roof(0, 0, 8, 8, 5, part["voice"]["roof"], "gable", "z", (2, 1))\n')
    keys = sorted(f.detail["key"] for f in welded.findings if f.code == "E015")
    assert keys == ["eave", "ends", "pitch", "pitch", "profile", "tiers"], \
        [f.message for f in welded.findings]
    assert not [f for f in welded.findings if f.code != "E015"], welded.findings
    clean = pre('    r = part.get("roof") or {}\n'
                '    b.roof(0, 0, 8, 8, 5, part["voice"]["roof"], style="hip", axis="x", '
                'ends=None, overhang=1)\n'
                '    b.building("h", 0, 0, 8, 8, 1, {"style": "gable", "axis": "z"}, '
                'mat=part["voice"])\n'
                '    b.roof(0, 0, 8, 8, 5, part["voice"]["roof"], ends=r.get("ends"))\n')
    assert clean.ok, [f.message for f in clean.findings]
    # ...and the by-construction half: whatever a type passes, the voice's silhouette is
    # what roof() and building() are handed
    from ethoslm.buildlib import TypeBuilder
    class _Spy:
        def __init__(self):
            self.calls = []
        def roof(self, *a, **k):
            self.calls.append(("roof", a, k))
            return 0
        def building(self, *a, **k):
            self.calls.append(("building", a, k))
            return {"ok": True}
    spy = _Spy()
    t = TypeBuilder(spy, {"floor_y": 64, "roof": pipeline.voice_roof("japanese_temple")})
    t.roof(0, 0, 8, 8, 70, "deepslate_tile", style="gable", pitch=(2, 1), ends="gable")
    t.building("h", 0, 0, 8, 8, 1, "gable", mat={"wall": "quartz"})
    t.building("h", 0, 0, 8, 8, 1, roof={"style": "hip", "pitch": (1, 2)})
    _, _a, k = spy.calls[0]
    assert k["ends"] == "irimoya" and k["tiers"] == 2 and k["eave"] == "upturned" \
        and "pitch" not in k and k["profile"] and k["style"] == "gable", k
    assert spy.calls[1][1][6] == {"style": "gable", "ends": "irimoya", "eave": "upturned",
                                  "tiers": 2, "profile": k["profile"]}, spy.calls[1]
    assert spy.calls[2][2]["roof"]["ends"] == "irimoya" and "pitch" not in spy.calls[2][2]["roof"]
    # ...and a silent voice hands over nothing, so the type's own words stand
    spy2 = _Spy()
    TypeBuilder(spy2, {"floor_y": 64, "roof": None}).roof(0, 0, 8, 8, 70, "x", pitch=(2, 1))
    assert spy2.calls[0][2] == {"pitch": (2, 1)}, spy2.calls
    return ("six silhouette literals refused across roof() and building(roof=...), "
            "style/axis/ends=None accepted; TypeBuilder hands roof() and building() the "
            "temple's irimoya, two tiers and upturned eave over whatever the type passed, "
            "and nothing in a silent voice")


@case
def t_r19_a2_a_voice_is_a_file_the_model_may_write_and_it_is_validated():
    """A2: `voices/<name>.json`, refused by role and family, or authored and kept.

        The refusal is the case. A voice naming a family the game has no stairs of is not a
        voice: the roof is laid in stairs and slabs, `material()` says so, and until this
        round the answer to being handed one was that the chimney of every building in the
        place quietly failed to be built. That is not hypothetical -- `blackstone_and_ash`,
        the voice a whole town was built in, declared `basalt` as its footing, and basalt
        has no stairs and no slab. The validator found it the day it was written.
        
    """
    from ethoslm import voices
    names = voices.names()
    assert len(names) >= 5, names
    for n in names:
        v = voices.load(n)
        # The six the shell is made of, always; `voices.OPTIONAL`.
        assert sorted(set(v["roles"]) - set(voices.OPTIONAL)) == sorted(voices.ROLES), \
            (n, sorted(v["roles"]))
        for role in voices.SHAPED:
            assert prims.family(v["roles"][role]), (n, role, v["roles"][role])
        for role in voices.OPTIONAL_SHAPED:
            if v["roles"].get(role):
                assert prims.family(v["roles"][role]), (n, role, v["roles"][role])
        assert 0.0 <= v["value"]["spread"] <= 1.0, v["value"]

    # the six roles are the library's six, and not a second opinion about them
    from ethoslm.buildlib import _MAT_ROLES
    assert sorted(voices.ROLES) == sorted(_MAT_ROLES), (voices.ROLES, _MAT_ROLES)

    ochre = {"name": "ochre_and_green_tile",
             "roles": {"wall": "sandstone", "footing": "cut_sandstone",
                       "frame": "stripped_spruce_log", "roof": "prismarine",
                       "trim": "smooth_sandstone", "floor": "terracotta"},
             "roof": {"eave": "upturned", "tiers": 2},
             "notes": {"blurb": "Ochre stone under green tile."}}
    got = voices.validate(ochre, where="the authored voice")
    assert got["roles"]["roof"] == "prismarine" and got["roof"]["tiers"] == 2, got
    assert got["value"]["spread"] > 0, got["value"]

    refusals = {}
    for role, mat, want in (("roof", "hay_block", "no hay_block stairs"),
                            ("footing", "basalt", "no basalt stairs"),
                            ("wall", "not_a_material", "not a material family")):
        bad = json.loads(json.dumps(ochre))
        bad["roles"][role] = mat
        try:
            voices.validate(bad, where="a voice")
            raise AssertionError(f"roles.{role} = {mat!r} was accepted")
        except voices.VoiceError as e:
            assert f"roles.{role}" in str(e) and mat in str(e), str(e)
            refusals[role] = str(e)[:60]
    # ...and `floor` alone may be a block with no shapes at all, which is the whole
    # reason the two lists are different lists
    fine = json.loads(json.dumps(ochre))
    fine["roles"]["floor"] = "white_concrete_powder"
    assert voices.validate(fine, where="a voice")["roles"]["floor"] \
        == "white_concrete_powder"
    # ...and a role the library does not have, and a roof end the library cannot draw
    for field, value, word in (("roles", {**ochre["roles"], "shade": "spruce"}, "shade"),
                               ("roof", {"ends": "onion"}, "onion")):
        bad = {**json.loads(json.dumps(ochre)), field: value}
        try:
            voices.validate(bad, where="a voice")
            raise AssertionError(f"{field}={value!r} was accepted")
        except voices.VoiceError as e:
            assert word in str(e), str(e)

    # ...and the place spec may author one, validated before anything is planned
    doc = {"kind": "city", "voice": ochre,
           "defining_parts": [{"name": "ring_walls", "kind": "edge", "family": "wall",
                               "relation": "concentric", "count": 3, "structures": 0}]}
    read = spec_mod.read_spec(doc, "Build Ringed City.")
    assert read["voice"] == "ochre_and_green_tile", read["voice"]
    assert read["authored_voice"]["roles"]["wall"] == "sandstone", read
    broken = json.loads(json.dumps(doc))
    broken["voice"]["roles"]["roof"] = "hay_block"
    try:
        spec_mod.read_spec(broken, "Build Ringed City.")
        raise AssertionError("a spec authoring an unbuildable voice was accepted")
    except spec_mod.SpecError as e:
        assert "roles.roof" in str(e) and "hay_block" in str(e), str(e)
    return (f"{len(names)} voices on disk, every one of them naming all six roles in "
            f"families the shell can shape; an authored ochre-and-green voice passes "
            f"and is kept; {len(refusals)} roles refused by name "
            f"(roof/hay_block, footing/basalt, wall/not_a_material), plus an unknown "
            f"role and an unknown roof end")


@case
def t_r19_a6_the_checkers_run_across_processes_and_answer_the_same():
    """A6: the same rows, serial and parallel.

        The unit of work in `check_type` is one composed program -- one fixture round at one
        seed -- and nothing one of them does is visible to another. The assertion is not
        that it is faster; it is that the answer does not move, field for field and finding
        for finding, which is the only thing that makes a harness change safe to make in the
        same round as a measurement.
        
    """
    import time
    was = os.environ.get("ETHOSLM_WORKERS")
    rnd = pipeline.Round.load(os.path.join(ROOT, "rounds", "types_d.json"))
    be = pipeline.OfflineBackend(rnd)
    fixtures = pipeline._fixtures(rnd)[:3]
    prog = os.path.join(ROOT, "types", "cottage.py")

    def run(n):
        os.environ["ETHOSLM_WORKERS"] = str(n)
        t = time.perf_counter()
        got = pipeline.check_type(rnd, be, prog, [], [1, 2], fixtures=fixtures,
                                  voice="white_render_dark_frame")
        return got, time.perf_counter() - t

    try:
        serial, ts = run(1)
        par, tp = run(4)
    finally:
        if was is None:
            os.environ.pop("ETHOSLM_WORKERS", None)
        else:
            os.environ["ETHOSLM_WORKERS"] = was

    def key(res):
        return [(r["round"], r["plot"], r["kind"], r["seed"], r["errors"],
                 r["entry_lines"], r["lines"], r["walk_pct"], r["rooms"], r["shut"],
                 r["crashed"], r["pending"], r["writes"],
                 sorted(f"{f.code}@{f.pos}" for f in r["report"].findings))
                for r in res["rows"]]
    assert serial["instances"] == par["instances"] > 0, (serial["instances"],
                                                         par["instances"])
    assert key(serial) == key(par), "the rows moved between serial and parallel"
    assert serial["text"] == par["text"], "the findings text moved"
    for k in ("errors", "entry_lines", "blocks", "writes", "crashed"):
        assert serial[k] == par[k], (k, serial[k], par[k])
    # ...and the map itself, on something with no world in it at all
    assert parallel.par_map(abs, [-3, 1, -2]) == [3, 1, 2]
    assert parallel.workers(1) == 1
    return (f"{serial['instances']} instances over {len(fixtures)} fixtures x 2 seeds: "
            f"identical rows, findings and totals, {ts:.0f}s serial against "
            f"{tp:.0f}s on {parallel.workers()} processes")


@case
def r_bss3_b1_the_sweep_crosses_every_parameter_a_type_declares():
    """A checker stands a type where the plan may ask for it.

        `check_params` fills a missing parameter from the low end of an int range and the
        first of a choice, so a checker given no parameters stood a type at exactly one
        point of its own declared space. `shop_house` declares `storeys` 2 to 3 and was
        only ever checked at two; the city asked for three. `wall` declares `height` 3 to
        20 and `width` 1 to 5 and was only ever checked at three by one; the city asked for
        twenty by three. Every value of a choice, every value of a short int range, and the
        ends and the middle of a long one -- because eighteen heights crossed with five
        widths and three crowns is an instrument nobody would run.
        
    """
    got = pipeline.param_combinations({"storeys": ("int", 2, 3),
                                       "trade": ("choice", ["grain", "cloth"])})
    assert got == [{"storeys": 2, "trade": "grain"}, {"storeys": 2, "trade": "cloth"},
                   {"storeys": 3, "trade": "grain"}, {"storeys": 3, "trade": "cloth"}], got
    wide = pipeline.param_combinations({"height": ("int", 3, 20)})
    assert [d["height"] for d in wide] == [3, 11, 20], wide
    # ...and every committed type's declared space is crossed by it, at a size a sweep
    # can actually be run at.
    worst = None
    for f in sorted(glob.glob(os.path.join(ROOT, "types", "*.py"))):
        decl = pipeline.load_type(f)
        n = len(pipeline.param_combinations(decl["params"]))
        assert n >= 1, f
        if worst is None or n > worst[1]:
            worst = (os.path.basename(f), n)
    assert worst[1] <= 27, f"{worst[0]} would be swept at {worst[1]} points"
    return (f"4 combinations of 2x2, a range of 18 sampled at 3, and the widest type "
            f"is {worst[0]} at {worst[1]}")


@case
def t_dp2a_a_civic_type_stands_under_the_voice_civic_silhouette_and_a_house_does_not():
    """Demo-polish, 2a. A voice may carry a second silhouette, `roof_civic`; a type
    whose `ROLE` is `civic` is roofed with it and every other type with the voice's
    one roof. No type names any of it. Stood on the same pad at the same seed:
    the temple (civic) changes its roof between a voice with and without the civic
    silhouette, and the cottage (urban) does not move a block."""
    import collections
    from ethoslm.buildlib import Builder
    ochre = pipeline.voice_roof("ochre_stone_green_tile")
    assert ochre and ochre.get("civic"), "the demo voice names no civic silhouette"
    plain = {k: v for k, v in ochre.items() if k not in ("civic", "chimney")}

    def stand(name, roof, size=12, seed=1):
        decl = pipeline.load_type(os.path.join(ROOT, "types", f"{name}.py"))
        mat = pipeline.voice_palette("ochre_stone_green_tile")
        w = size + 4
        part = {"label": "t", "kind": "plot", "x0": 20, "z0": 20, "x1": 19 + w, "z1": 19 + w}
        x0, z0, x1, z1 = pipeline.part_rect(part)
        vol = _flat_world(size + 45)
        b = Builder(offline.OfflineSite(vol))
        b._vol = vol
        b.frontage = None
        b.registry = _OnePlot({"label": "t", "x0": x0, "z0": z0, "x1": x1, "z1": z1})
        sited = b.site(dict(part), mat=mat, roof=roof)
        ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
        exec(compile(decl["src"], decl["path"], "exec"), ns)                  # noqa: S102
        tb = b.type_builder(sited, role=decl["role"])
        res = ns["build"](tb, sited, seed, storeys=2)
        b.resolve_steps()
        assert res and res.get("ok", True), (name, res)
        return tb.roof_spec, tuple(sorted(b._pending.items()))

    spec_c, temple_c = stand("temple", ochre)
    spec_p, temple_p = stand("temple", plain)
    assert spec_c == ochre["civic"] and spec_p == plain, (spec_c, spec_p)
    assert temple_c != temple_p, "the civic silhouette did not reach the temple's roof"
    cols_c = {(x, z) for (x, _y, z), _b in temple_c}
    cols_p = {(x, z) for (x, _y, z), _b in temple_p}
    assert cols_c == cols_p, "the temple moved its footprint with the silhouette"
    top_c = max(y for (_x, y, _z), _b in temple_c)
    top_p = max(y for (_x, y, _z), _b in temple_p)
    assert top_c > top_p, (top_c, top_p, "two tiers steep at the ridge should stand higher")
    spec_h, house_c = stand("cottage", ochre)
    _spec, house_p = stand("cottage", plain)
    assert spec_h == plain, spec_h
    assert house_c == house_p, "a house's roof moved with the civic silhouette"
    # the schema: the civic roof is validated like the roof, and refused by name
    v = voices.load("ochre_stone_green_tile")
    assert v["roof_civic"]["tiers"] == 2, v["roof_civic"]
    bad = dict(json.load(open(os.path.join(ROOT, "voices", "ochre_stone_green_tile.json"))))
    bad["roof_civic"] = {"tiers": 9}
    try:
        voices.validate(bad, where="a test voice")
        raise AssertionError("a civic roof of nine tiers was accepted")
    except voices.VoiceError as e:
        assert "roof_civic" in str(e) and "tiers" in str(e), str(e)
    # a type that names the silhouette is still refused; the voice file is not a type
    return (f"temple roofed to y={top_c} under the civic silhouette against {top_p} "
            f"under the house roof, on the same {len(cols_c)} columns; cottage "
            f"byte-identical either way; a nine-tier civic roof refused by name")



@case
def t_craft_a_block_takes_an_axis_where_the_registry_says_so_and_every_type_stands_in_a_new_voice():
    """**The craft round, E2**, and a defect the new voice found the day it existed.

        Nine committed types carried a private `_axial` that decided by the block's **name**
        -- `endswith(("_log", "_pillar", "_wood"))` -- and four of them composed `[axis=...]`
        at other call sites without going through even that. A voice whose trim is `purpur`
        lays `bare` as `purpur_block`, a cube with no axis, and `purpur_block[axis=z]` is a
        block state the game does not have: `workshop`'s bay beam, `gate_tower`'s lintel and
        `keep`'s two wrote it, and the block registry is what found it. `prims.axial` asks
        the registry and every type asks `prims.axial`.

        The sweep is the second half: every committed type at the bottom, the middle and the
        top of its own band, in the new voice and in two of the axis's other rungs, at zero
        unknown block states -- which is the check that caught this one.
        
    """
    from ethoslm import placeread, prims, styles
    assert prims.axial("purpur_block", "z") == "purpur_block"
    assert prims.axial("purpur_pillar", "z") == "purpur_pillar[axis=z]"
    assert prims.axial("oak_log[axis=y]", "x") == "oak_log[axis=x]"
    assert prims.axial("diorite", "y") == "diorite"
    for f in sorted(os.listdir(os.path.join(ROOT, "types"))):
        if f.endswith(".py") and not f.startswith("_"):
            src = open(os.path.join(ROOT, "types", f)).read()
            assert "[axis=" not in src, f"{f} composes an axis by hand"
            assert '_pillar", "_wood"' not in src, f"{f} decides an axis by name"
    said = []
    for voice in ("pale_quartz_and_gilt", "ochre_stone_green_tile",
                  "packed_earth_and_dark_tile"):
        fams = {x for x in (prims.family(m) for m in
                            styles.VOICES[voice]["palette"].values()) if x}
        n, bad = 0, []
        for f in sorted(os.listdir(os.path.join(ROOT, "types"))):
            if not f.endswith(".py") or f.startswith("_"):
                continue
            name = f[:-3]
            decl = pipeline.load_type(os.path.join(ROOT, "types", f))
            nd = decl.get("needs") or {}
            a, b_, c, e = nd.get("footprint", (3, 3, 3, 3))
            ex = set(nd.get("except") or ())
            clean = [v for v in range(max(a, b_), min(c, e) + 1) if v not in ex]
            sizes = [v for v in sorted({clean[0], clean[len(clean) // 2], clean[-1]})
                     if v <= 20] if clean else []
            for size in sizes:
                got = _stand(name, voice, size)
                n += 1
                if not got["ok"] or got["bad"]:
                    bad.append((name, size, got["ok"], sorted(got["bad"])[:2]))
                    continue
                # an `area` is the ground between the buildings and its planting is the
                # setting's, not the voice's, so only a built thing is read against the
                # palette the way `placeread.built_palette` reads one
                if decl["kind"] == "area":
                    continue
                classed = {k: v for k, v in got["blocks"].items() if prims.family(k)}
                share = (sum(v for k, v in classed.items() if prims.family(k) in fams)
                         / max(1, sum(classed.values())))
                if share < placeread.BUILT_SHARE:
                    bad.append((name, size, round(share, 3)))
        assert not bad, (voice, bad[:6])
        said.append(f"{voice} {n}")
    return ("purpur_block takes no axis and purpur_pillar does, off the registry; no "
            "type under types/ composes one by hand or decides it by name; every "
            "committed type stands at the bottom, middle and top of its band in three "
            "voices at 0 unknown block states -- " + ", ".join(said) + " instances")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:62s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:62s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:62s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} type cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
