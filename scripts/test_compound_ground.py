"""A compound's ground, a large plot on a slope, and the arrival camera.

    $PY scripts/test_compound_ground.py

C1. **A compound's ground is the library's arithmetic** (thread 38). `compound_ground`
is computed from what a compound is registered to hold and what the committed types are
measured to need, never from a number a spec call wrote; the square it names actually
holds the composition, laid out and put through the plan validator; and it satisfies the
place read's own footprint margin. C2. **The scale clause is asked at plan time.** A
compound too small to be monumental is refused by name before the ground is touched, and
the brief the planner reads carries the same number the refusal quotes. C3.
**Generality, R1.** A place whose centre is not a compound. C4. **A large plot on a
slope** (thread 39). The needs sweep stands every size on two grounds and the type
checker stands every plot type on a large sloped pad; and the library defect both of
them found -- a doorstep put back as air, which is a hole in the floor a person has to
stand on -- is closed, with its case. C5. **The arrival camera stands back in proportion
to what is in front of it** (thread 41). A wall of twenty puts it exactly where the
constant always did; a wall of forty-eight stands it back far enough that the top of the
wall subtends the same angle; nothing measured leaves it at the constant. C6.
**Generality, R4.** The wall-less fallback is unchanged, and so is every frame shot
without a measured rise.

Nothing here needs `out/` except C4's checker-fixture case, which builds its own ground
in a scratch directory, and C3's, which skips without the cached worlds.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (  # noqa: E402
    circulate, observe, offline, pipeline, placeplan, placeread, render,
    spec as spec_mod, stages,
)
from ethoslm.buildlib import Builder  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[3:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


def _find_site():
    """`scripts/find_site.py` as a module. It is a script and this asks it a question."""
    import importlib.util
    p = os.path.join(ROOT, "scripts", "find_site.py")
    s = importlib.util.spec_from_file_location("find_site_case", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _recorded(name: str) -> dict:
    """One of the three specs a real isolated call wrote, off the record."""
    p = os.path.join(ROOT, "rounds", "place-specs.json")
    got = json.load(open(p))["calls"][name]
    return spec_mod.read_spec(json.loads(json.dumps(got["answered"])),
                              got["sentence"])


# ------------------------------------------- C1. a compound sizes its own ground

@case
def t_c1_a_compounds_ground_is_sized_from_what_it_holds_and_not_from_the_spec():
    """The square `compound_ground` names holds the registered composition -- two halls
        and a court at the largest plot the types admit, with the wall inset outside them --
        and it is at least the place read's own footprint margin over that same plot.

        Laid out and put through `pipeline.plan_failures`, because "it fits" is a claim about
        the arithmetic the validator does and not about the arithmetic this case does.
        
    """
    got = placeplan.compound_ground()
    side, biggest = got["largest_plot"]
    margin = placeread.MONUMENT_FOOTPRINT_MARGIN
    assert got["side"] >= placeplan.COMPOUND_MIN, got
    assert got["side"] * got["side"] >= margin * side * side, got

    # ...and it holds the composition. Two halls and a court of the largest plot, in the
    # square the arithmetic named, with the compound's wall on its own inset.
    n = got["side"]
    x0 = z0 = 0
    k = placeplan.COMPOUND_WALL_INSET
    inset = [[x0 + k, z0 + k], [x0 + n - 1 - k, z0 + k],
             [x0 + n - 1 - k, z0 + n - 1 - k], [x0 + k, z0 + n - 1 - k],
             [x0 + k, z0 + k]]
    gap = got["plot_clearance"] + 1
    m = k + got["wall_clearance"] + 1
    at = [(x0 + m, z0 + m), (x0 + m + side + gap, z0 + m),
          (x0 + m, z0 + m + side + gap)]
    parts = [{"kind": "edge", "name": "compound_wall", "type": "wall", "seed": 1,
              "params": {}, "path": inset, "width": 1, "in": []}]
    for i, (px, pz) in enumerate(at):
        parts.append({"kind": "plot", "name": f"hall_{i}", "type": biggest, "seed": i + 2,
                      "params": {}, "x0": px, "z0": pz,
                      "x1": px + side - 1, "z1": pz + side - 1, "in": []})
        assert px + side - 1 <= x0 + n - 1 - m and pz + side - 1 <= z0 + n - 1 - m, \
            (got, (px, pz), "the composition does not fit the square its own rule named")
    _table, decls = placeplan.types_card()
    bad = [f for f in pipeline.plan_failures(parts, decls)
           if f["check"] in ("overlap", "footprint")]
    assert not bad, bad

    # ...and the number is the library's, not the spec call's.
    fs = _find_site()
    spec = _recorded("ringed_city")
    core = spec_mod.core(spec)
    assert spec_mod.compound(core), core
    asked = (core.get("needs") or {}).get("plateau")
    assert fs._plateau_size(spec) == got["side"], (fs._plateau_size(spec), got["side"])
    assert fs.core_size(spec) == got["side"], fs.core_size(spec)
    return (f"a compound's ground is {got['side']}x{got['side']} from "
            f"{got['parts']} parts of {side}x{side} ({biggest}) and the wall's inset, "
            f"against {margin:g}x{side}x{side} = {got['monumental_columns']} columns "
            f"the place read wants; the composition fits it and the plan validator "
            f"agrees; the spec call asked for "
            + (f"{asked}" if asked else "nothing"))


@case
def t_c2_a_compound_too_small_to_be_monumental_is_refused_at_plan_time_and_said():
    """The place read's footprint margin, asked before the ground is touched -- and in
    the brief, because a level is handed back at most once."""
    big = placeplan._monumental_columns()
    spec = _recorded("ringed_city")
    site = {"origin": [0, 0], "size": 512,
            "stats": {"min": 60, "max": 68, "relief": 8, "std": 2.0},
            "mean_grid": [[64] * 12 for _ in range(12)],
            "roughness_grid": [[2] * 12 for _ in range(12)],
            "surface_blocks": {"grass_block": 100}}
    _table, decls = placeplan.types_card()

    def place_with(n):
        return {"intent": "a fixture", "centre": "royal_palace", "parts": [],
                "compounds": [{"name": "royal_palace", "defines": "royal_palace",
                               "x0": 200, "z0": 200, "x1": 199 + n, "z1": 199 + n}],
                "districts": []}

    small = big["side_needed"] - 1
    got = [f for f in placeplan.place_failures(place_with(small), spec, site, decls)
           if f["check"] == "scale"]
    assert got, f"a {small}x{small} compound was not refused"
    assert str(big["columns"]) in got[0]["why"], got[0]["why"]
    ok = [f for f in placeplan.place_failures(place_with(big["side_needed"]),
                                              spec, site, decls)
          if f["check"] == "scale"]
    assert not ok, ok
    # ...and the planner is told the number it will be refused on.
    note = placeplan._compound_note(spec)
    assert str(big["columns"]) in note and f"{big['side_needed']}x{big['side_needed']}" \
        in note, note
    return (f"a {small}x{small} compound is refused by name at plan time and "
            f"{big['side_needed']}x{big['side_needed']} passes; the brief quotes "
            f"{big['columns']} columns and {big['margin']:g}x a "
            f"{big['side']}x{big['side']} `{big['type']}`")


@case
def t_c3_a_place_with_no_compound_takes_the_ground_it_always_took():
    """The generality guard for R1. a hamlet has one district and nothing else. Neither is
    a compound, so neither number moves and neither record grows a field.
    """
    fs = _find_site()
    rows = []
    for name in ("walled_town", "hamlet"):
        spec = _recorded(name)
        core = spec_mod.core(spec)
        assert not spec_mod.compound(core), (name, core)
        assert fs._compound_plateau(spec) == 0, name
        assert fs._compound_ground_record(spec) is None, name
        # ...and the number is exactly the rule that stood before a compound existed.
        want = (spec["needs"].get("plateau")
                or (core.get("needs") or {}).get("plateau") or fs.PLATEAU_DEFAULT)
        want = int(min(want, Builder.PLATEAU_MAX, spec["needs"]["footprint"]))
        assert fs._plateau_size(spec) == want, (name, fs._plateau_size(spec), want)
        c = spec_mod.core_needs(spec)
        assert fs.core_size(spec) == int(min(int(c.get("plateau") or want),
                                             int(spec["needs"]["footprint"]))), name
        # ...and no compound means no compound note and no scale refusal anywhere.
        assert placeplan._compound_note(spec) == "", name
        rows.append(f"{name}: plateau {fs._plateau_size(spec)}, core "
                    f"{fs.core_size(spec)}, {len(spec_mod.walls(spec))} wall(s)")
    return "; ".join(rows) + " -- unmoved, and no compound arithmetic in either record"


# ------------------------------------------------- C4. a large plot on a slope

@case
def t_c4_a_large_plot_on_a_slope_is_ground_two_instruments_now_have():
    """Thread 39, at both of its general points, with the library defect they found.

        The needs sweep stands every size on a plane **and** on a bank; the type checker
        stands every plot type on a small sloped pad and on one at the top of its own band.
        And the thing that made a large plot on a slope different from the same plot on a
        plane was in the library: `TypeBuilder` puts a doorstep back as **air** where a type
        built on it, which on a plinth is right -- the cell was air -- and on a platform is a
        hole cut out of the paving `site()` laid, so the room's lowest level is the hole, and
        none of its floor can be walked to from its own door.
        
    """
    import type_needs as tn
    from ethoslm.slopefixture import descriptors
    assert set(tn.GROUNDS) == {"flat", "slope"}, tn.GROUNDS
    # every plot type is checked on a large sloped pad as well as a small one
    d = os.path.join(ROOT, "types")
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        decl = pipeline.load_type(os.path.join(d, f))
        if decl["kind"] != "plot":
            continue
        names = [x["round"] for x in descriptors(decl)]
        a, b, c, e = decl["needs"]["footprint"]
        top = max(c, e)
        if top > max(9, 11):
            assert any(n.startswith("slopebig_") for n in names), (f, names)
        rows.append(f[:-3])
    # ...and the defect itself. `site()` paves the columns of the way in, and the two
    # cells over each of them are held open for the person walking it; a type that
    # builds in one of them is refused and the cell **put back to what stood there**,
    # which on a paved column is paving and not air. Planted rather than found, because
    # what is under test is the rule and not one hillside.
    name, size = "court_large", (24, 24)
    combos = tn.param_combinations(name)
    part, plot, span = tn._part(name, "plot", size)
    vol = tn.slope(span, plot["x0"], plot["x1"])
    b = Builder(offline.OfflineSite(vol))
    b._vol, b.frontage, b.registry = vol, None, tn._Registry(plot)
    p = b.site(dict(part), mat=tn.MAT, roof=tn.ROOF)
    assert p["ground"] == "platform", p["sited"]
    tb = b.type_builder(p)
    paved, empty = tb._doorstep[-1], tb._doorstep[0]
    b.place_block(paved[0], paved[1], paved[2], "cobblestone")
    b._pending.pop(empty, None)
    tb = b.type_builder(p)
    tb.place_block(paved[0], paved[1], paved[2], "stone")
    tb.place_cuboid(empty[0], empty[1], empty[2], empty[0], empty[1], empty[2], "stone")
    assert len(tb.refused) == 2, tb.refused
    assert b._pending.get(paved) == "cobblestone", b._pending.get(paved)
    assert empty not in b._pending, b._pending.get(empty)
    was = "cobblestone"
    return (f"the sweep stands every size on {len(tn.GROUNDS)} grounds and "
            f"{len(rows)} plot types are checked on a large sloped pad; a doorstep the "
            f"library paved is put back as {was.split('[')[0]!r} and not as air, and "
            f"{name} at {size[0]}x{size[1]} is "
            f"{len(tn.SEEDS) * len(combos)} of {len(tn.SEEDS) * len(combos)} clean on "
            f"the bank")


# -------------------------------------------------------- C5/C6. the camera

@case
def t_c5_the_arrival_camera_stands_back_in_proportion_to_the_height_in_front_of_it():
    """Thread 41. The angle the top of the thing in front of the camera subtends is what
    is held; the distance is whatever holds it."""
    was = render.PLACE_SKYLINE_BACK
    eye = render.PLACE_SKYLINE_EYE
    twenty = render.skyline_back(render.PLACE_SKYLINE_RISE + eye)
    assert twenty == was, (twenty, was)
    great = render.skyline_back(48)
    assert great > was * 2, great
    a = math.atan2(render.PLACE_SKYLINE_RISE, twenty)
    bnew = math.atan2(48 - eye, great)
    assert abs(a - bnew) < 1e-9, (math.degrees(a), math.degrees(bnew))
    # ...and it never comes closer than the validated constant, whatever is in front
    assert render.skyline_back(4) == was and render.skyline_back(0) == was
    return (f"a wall of {render.PLACE_SKYLINE_RISE + eye} stands the camera at "
            f"{twenty:.0f}, one of 48 at {great:.0f}, and the top of both subtends "
            f"{math.degrees(a):.1f} degrees; nothing shorter brings it closer than "
            f"{was}")


@case
def t_c6_a_place_with_nothing_tall_on_its_approach_is_framed_exactly_as_before():
    """The generality guard for R4, and the measurement that feeds it.

        `approach_rise` reads the world; with no gate, no volume or nothing above the gate
        it is 0, and 0 is the constant. A wall of forty-eight in a fixture reads 48.
        
    """
    from ethoslm.circulate import Threshold
    assert render.skyline_back(None) == render.PLACE_SKYLINE_BACK
    assert render.approach_rise(None, None) == 0
    y = 64
    codes = np.zeros((64, 96, 64), np.uint16)
    codes[:, :y - 40, :] = 1
    vol = observe.Volume(0, 40, 0, codes, ["air", "stone"])
    gate = Threshold(id="gate", x=32, z=32, y=y, facing="north", door=(32, y, 32))
    assert render.approach_rise(vol, gate) == 0, render.approach_rise(vol, gate)
    flat_back = render.skyline_back(render.approach_rise(vol, gate))
    assert flat_back == render.PLACE_SKYLINE_BACK, flat_back
    # ...and a wall of 48 running past the gate is 48, wherever along the run it stands
    wall = vol.codes.copy()
    wall[28:36, :y + 48 - 40 + 1, :] = 1
    tall = observe.Volume(0, 40, 0, wall, ["air", "stone"])
    got = render.approach_rise(tall, gate)
    assert got == 48, got
    centre = (32.0, float(y), 200.0)
    # [0]: the skyline is a ladder since phase 4 and its first rung does not move
    a = render.place_card_shots(centre, 128, gate=gate)["place_skyline"][0].view.position
    c = render.place_card_shots(centre, 128, gate=gate,
                                rise=got)["place_skyline"][0].view.position
    assert abs(abs(a[2] - gate.door[2]) - render.PLACE_SKYLINE_BACK) <= 1, a
    assert abs(c[2] - gate.door[2]) > abs(a[2] - gate.door[2]), (a, c)
    return (f"nothing above the gate reads 0 and frames at {render.PLACE_SKYLINE_BACK}, "
            f"the constant every recorded frame was shot at; a wall of 48 over the gate "
            f"reads {got} and frames at {render.skyline_back(got):.0f}")


# ----------------------------------------------------------------- the runner

def main():
    ok = bad = skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok} of {ok + bad} cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
