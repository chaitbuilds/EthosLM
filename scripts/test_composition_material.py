"""The composition round's material cases: exact block state, and declared figures.

    $PY scripts/test_composition_material.py            # these cases
    $PY scripts/test_composition_material.py --only c   # one of them

**This file extends `scripts/test_design_material.py`; it does not supersede it.** That
file (and, through it, the expression round's) covers ownership at the write,
protection, shape preservation, the reconciliation rules, the per-context matched rate
and the textured display's positive control, and it still runs unchanged.

  exact state    freshness was decided on the bare block **name** in both places that
                 decide it (`surfaces.reconcile` and `material.apply`), so a cell whose
                 stair had been re-faced, whose slab had moved to the top half or whose
                 log had changed axis since the record was made still read `fresh` --
                 and `material.substitute` then re-emitted the *record's* suffix over
                 it and turned the block back round. Now `surfaces.same_state` decides,
                 and the substituted block takes its state from the block that stands.
  figures        the record said who owned a cell and what role it had, and nothing
                 said the cell was part of a pattern somebody drew, so the design
                 round's pass replaced both market floors' chequers, the palace courts'
                 laid paving and the great wall's string course with noise.
                 `Primitives.figure` declares it at the write, `surfaces.FLAGS["figure"]`
                 carries it, and `material.plan` refuses it -- in the random arm too, so
                 the control stays matched.

Small and fast: hand-made six-block worlds, one probed square and one probed wall,
under a minute, no server and no cached world.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from ethoslm import construction as C, material as M, prims  # noqa: E402
from ethoslm import surfaces as S  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


# ------------------------------------------------------------------ fixtures A hand-
# made world and a hand-made record, so a state change can be put in a position no
# probed building happens to produce. The palette carries two states of one block on
# purpose: that pair is the whole question.

_PAL = ["air", "cobblestone", "andesite", "oak_planks",
        "cobblestone_stairs[facing=north,half=bottom]",
        "cobblestone_stairs[facing=south,half=bottom]",
        "cobblestone_stairs[facing=north,half=bottom,shape=inner_left,waterlogged=false]",
        "cobblestone_slab[type=bottom]", "cobblestone_slab[type=top]"]


def _world(cells: dict, size: int = 8) -> Volume:
    codes = np.zeros((size, size, size), np.uint16)
    for (x, y, z), name in cells.items():
        codes[x, y, z] = _PAL.index(name)
    return Volume(0, 0, 0, codes, list(_PAL))


def _part(name: str, cells: dict, *, protected=(), voice="drystone_and_thatch",
          type_="cottage", states=("cobblestone",), figures=None) -> dict:
    """One part's record: `cells` is {role: [(x, y, z, state_index, flags), ...]}."""
    return {"part": name, "type": type_, "kind": "plot", "floor_y": 0, "voice": {},
            "voice_name": voice, "states": list(states),
            "cells": {r: [list(c) for c in v] for r, v in cells.items()},
            "protected": [list(p) for p in protected],
            "counts": {r: len(v) for r, v in cells.items()},
            "figures": dict(figures or {}), "how": {},
            "owned": sum(len(v) for v in cells.values()) + len(protected)}


def _doc(parts: list) -> dict:
    return {"record": "surfaces", "version": 1, "flags": S.FLAGS,
            "editable": list(prims.EDITABLE_ROLES), "parts": parts}


def _kept(doc: dict) -> set:
    return {(c[0], c[1], c[2]) for p in doc["parts"]
            for cells in p["cells"].values() for c in cells}


#: A recipe that will substitute anything it is allowed to, so nothing that survives a
#: case below survived because the recipe declined to reach it.
_ANY = {"wall": [{"family": "andesite", "weight": 1.0, "when": "any"}],
        "footing": [{"family": "andesite", "weight": 1.0, "when": "any"}]}
_DAMP = {"wall": [{"family": "andesite", "weight": 1.0, "when": "damp"}]}


# ------------------------------------------------------- Q1: the exact block state

@case
def t_a_a_cell_whose_state_moved_is_stale_and_is_left_alone():
    """The defect, with its own positive control beside it.

        Two cobblestone stairs, recorded facing north. One of them is facing south in the
        assembled world -- a later pass turned it. The name is the same, so the old
        name-only test called both fresh and the pass wrote `andesite_stairs[facing=north]`
        over the turned one, keeping the family *and putting the tread back the way the
        record remembered it*. Orientation is physics here: that is a different block.
        
    """
    up = S.FLAGS["open_up"]
    rec = _part("a", {"wall": [(2, 1, 2, 0, up), (2, 1, 3, 0, up)]},
                states=("cobblestone_stairs[facing=north,half=bottom]",))
    world = _world({(2, 1, 2): "cobblestone_stairs[facing=north,half=bottom]",
                    (2, 1, 3): "cobblestone_stairs[facing=south,half=bottom]"})
    out, rep = S.reconcile(_doc([rec]), world)
    assert (2, 1, 2) in _kept(out), "the unmoved stair was dropped: the control fails"
    assert (2, 1, 3) not in _kept(out), "the turned stair is still in the record"
    assert rep["stale_block_no_longer_stands"] == 1, rep
    # ...and `apply` refuses it on its own, without reconciliation, because it is the
    # second of the two places that decide freshness
    fin, mrec = M.apply(world, _doc([rec]), _ANY, {}, 1, "contextual", reconcile=False)
    assert mrec["stale_skipped"] == 1, mrec
    assert mrec["substituted"] == 1, mrec
    assert fin.state(2, 1, 3) == "cobblestone_stairs[facing=south,half=bottom]", \
        f"the turned stair was overwritten: {fin.state(2, 1, 3)}"
    assert fin.state(2, 1, 2).startswith("polished_andesite_stairs"), \
        fin.state(2, 1, 2)
    return (f"the turned stair is stale in both deciders ({rep['stale_block_no_longer_stands']} "
            f"in reconcile, {mrec['stale_skipped']} in apply) and the unturned one is "
            f"still edited ({mrec['substituted']} substitution)")


@case
def t_b_a_slab_that_moved_to_the_top_half_is_stale_too():
    """Not only facing: any property the record named. A slab flipped from bottom to
    top is a different cell to walk on, and the old test could not see it."""
    up = S.FLAGS["open_up"]
    rec = _part("a", {"floor": [(2, 1, 2, 0, up), (2, 1, 3, 0, up)]},
                states=("cobblestone_slab[type=bottom]",))
    world = _world({(2, 1, 2): "cobblestone_slab[type=bottom]",
                    (2, 1, 3): "cobblestone_slab[type=top]"})
    out, rep = S.reconcile(_doc([rec]), world)
    assert rep["stale_block_no_longer_stands"] == 1, rep
    assert (2, 1, 2) in _kept(out) and (2, 1, 3) not in _kept(out)
    return f"{rep['stale_block_no_longer_stands']} of 2 slabs stale; the flipped one"


@case
def t_c_a_world_that_names_more_properties_than_the_record_is_not_stale():
    """The rule that keeps the fix from switching the pass off by accident."""
    up = S.FLAGS["open_up"]
    rec = _part("a", {"wall": [(2, 1, 2, 0, up)]},
                states=("cobblestone_stairs[facing=north,half=bottom]",))
    world = _world({(2, 1, 2): "cobblestone_stairs[facing=north,half=bottom,"
                               "shape=inner_left,waterlogged=false]"})
    out, rep = S.reconcile(_doc([rec]), world)
    assert rep["stale_block_no_longer_stands"] == 0, rep
    assert (2, 1, 2) in _kept(out)
    assert S.same_state("cobblestone_stairs[facing=north,half=bottom]",
                        "cobblestone_stairs[facing=north,half=bottom,shape=inner_left,"
                        "waterlogged=false]")
    assert not S.same_state("cobblestone_stairs[facing=north,half=bottom]",
                            "cobblestone_stairs[facing=west,half=bottom,shape=straight,"
                            "waterlogged=false]")
    # and the substitution takes its suffix from the block that **stands**, so the
    # normalised shape is not thrown away by re-emitting the record's abbreviation
    fin, mrec = M.apply(world, _doc([rec]), _ANY, {}, 1, "contextual", reconcile=False)
    got = fin.state(2, 1, 2)
    assert got.startswith("polished_andesite_stairs["), got
    assert "shape=inner_left" in got, f"the world's own shape was dropped: {got}"
    assert mrec["restated_from_the_standing_block"] == 1, mrec
    return (f"a normalised world state is fresh, a moved one is not, and the "
            f"substitute keeps the world's own suffix: {got}")


# ----------------------------------------------------------- Q2: declared figures

@case
def t_d_a_declared_figure_survives_a_pass_an_identical_ordinary_cell_does_not():
    """The case the design round's result asked for, as an A/B on one world.

        Two cells of the **same role**, the **same block**, the **same flags** and the same
        part. One is declared part of a figure and one is not. The recipe would take both.
        
    """
    up = S.FLAGS["open_up"]
    fig = up | S.FLAGS["figure"]
    rec = _part("a", {"wall": [(2, 1, 2, 0, up), (2, 1, 3, 0, fig)]},
                figures={"chequer": 1})
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone"})
    fin, mrec = M.apply(world, _doc([rec]), _ANY, {}, 1, "contextual", reconcile=False)
    assert fin.state(2, 1, 2) == "andesite", fin.state(2, 1, 2)
    assert fin.state(2, 1, 3) == "cobblestone", \
        f"a declared figure was repainted: {fin.state(2, 1, 3)}"
    assert mrec["substituted"] == 1 and mrec["figure_refused"] == 1, mrec
    # the control arm refuses it for the same reason, or it is matched on a different
    # pool of cells than the arm it is a control for
    _f2, r2 = M.apply(world, _doc([rec]), _ANY, {}, 1, "random", reconcile=False)
    assert r2["figure_refused"] == 1, r2
    # ...and the figure is out of the denominator, so the matched rate is of the cells a
    # recipe may actually reach
    assert mrec["totals"]["wall"] == 1, mrec["totals"]
    return (f"the ordinary cell became andesite and the identical declared one did not; "
            f"{mrec['figure_refused']} refused in the contextual arm and "
            f"{r2['figure_refused']} in the matched random one; the recipe's denominator "
            f"is {mrec['totals']['wall']}, not 2")


@case
def t_e_the_figure_bit_survives_reconciliation_and_is_counted():
    """A neighbour built later cannot make a chequer stop being a chequer: the bit is
    carried the way ground contact is, and the report says how much of it is left."""
    up = S.FLAGS["open_up"]
    fig = up | S.FLAGS["figure"]
    a = _part("a", {"wall": [(2, 1, 2, 0, fig), (2, 1, 3, 0, fig), (2, 2, 2, 0, fig)]},
              figures={"chequer": 3})
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone",
                    (2, 2, 2): "cobblestone"})
    out, rep = S.reconcile(_doc([a]), world)
    kept = [c for p in out["parts"] for cells in p["cells"].values() for c in cells]
    # (2,1,2) is buried under (2,2,2) on its up face but still has open sides
    assert rep["figure_cells_recorded"] == 3, rep
    assert rep["figure_cells_kept"] == len(kept), rep
    assert all(c[4] & S.FLAGS["figure"] for c in kept), kept
    assert rep["figures"] == {"chequer": 3}, rep
    # and a cell a later part overwrote is dropped whole, so the bit cannot protect
    # somebody else's block
    b = _part("b", {"wall": [(2, 1, 3, 0, up)]}, voice="japanese_minka")
    out2, rep2 = S.reconcile(_doc([a, b]), world)
    later = next(p for p in out2["parts"] if p["part"] == "b")
    mine = [c for cells in later["cells"].values() for c in cells]
    assert mine and not (mine[0][4] & S.FLAGS["figure"]), mine
    return (f"{rep['figure_cells_kept']} of {rep['figure_cells_recorded']} figure cells "
            f"kept with the bit on, named {rep['figures']}; a later part's claim on the "
            f"same cell carries no figure")


@case
def t_f_the_types_that_lay_the_named_patterns_declare_them():
    """The three patterns the design round's result named, declared by the types that
    draw them -- read off a real build, not off the source."""
    got = {}
    b, sited, res = C.probe_build("square", 24, 24, {"paving": "checker"}, seed=3,
                                  voice="pale_quartz_and_gilt")
    assert res.get("ok"), res
    r = S.record(b, {**sited, "name": "square", "type": "square", "kind": "area"})
    got["square"] = dict(r["figures"])
    b2, sited2, res2 = C.probe_build("market", 26, 20, {}, seed=2,
                                     voice="ochre_stone_green_tile")
    assert res2.get("ok"), res2
    r2 = S.record(b2, {**sited2, "name": "market", "type": "market", "kind": "area"})
    got["market"] = dict(r2["figures"])
    assert any(k.startswith("square_paving") for k in got["square"]), got
    assert "market_floor" in got["market"], got
    for name, rec in (("square", r), ("market", r2)):
        cells = [c for cs in rec["cells"].values() for c in cs]
        n = sum(1 for c in cells if c[4] & S.FLAGS["figure"])
        assert n == sum(rec["figures"].values()), (name, n, rec["figures"])
    return f"declared on a built probe: {got}"


@case
def t_g_the_figure_register_is_of_the_last_write_and_a_refusal_gives_it_back():
    """Two rules `_owner` already had, and the figure register has to share or it would
    protect a cell that is gone: an ordinary write over a figure cell takes it back
    out, air takes it out, and a primitive that refuses winds the register back."""
    from ethoslm import buildlib as BL

    class _Site:
        def __init__(self):
            self.editor = None

        def height(self, x, z):
            return 0
    b = BL.Builder(_Site())
    with b.figure("chequer"):
        b.place_block(1, 1, 1, "quartz_block")
        b.place_block(1, 1, 2, "quartz_block")
    assert set(b.figure_cells) == {(1, 1, 1), (1, 1, 2)}, b.figure_cells
    b.place_block(1, 1, 1, "cobblestone")            # an ordinary write over a figure
    assert (1, 1, 1) not in b.figure_cells, b.figure_cells
    with b.figure("chequer"):
        b.place_block(1, 1, 3, "quartz_block")
    b.place_block(1, 1, 3, "air")                    # cleared
    assert (1, 1, 3) not in b.figure_cells, b.figure_cells
    mark = b._mark()
    with b.figure("string_course"):
        b.place_block(2, 2, 2, "purpur_block")
    assert (2, 2, 2) in b.figure_cells
    b._rollback(mark)
    assert (2, 2, 2) not in b.figure_cells, "a wound-back call left a figure behind"
    assert set(b.figure_cells) == {(1, 1, 2)}, b.figure_cells
    return ("last write wins, air clears it, and a rollback gives it back: "
            f"{sorted(b.figure_cells)}")


# ------------------------------------- Q3: the conditions, and what did not change

@case
def t_h_damp_wants_a_side_face_so_a_paved_floor_is_not_damp():
    """The design round's cause 2. A floor is in ground contact by definition, so
    `damp` selected every cell of a paved court at the variant's full weight. Damp is
    what wicks out of the ground into a vertical surface, so it wants a side face."""
    F = S.FLAGS
    inside = F["open_up"] | F["ground_contact"]              # paving, seen from above
    edge = F["open_up"] | F["open_north"] | F["ground_contact"]
    wall = F["open_north"] | F["base_course"]
    rec = _part("a", {"wall": [(2, 1, 2, 0, inside), (2, 1, 3, 0, edge),
                               (2, 1, 4, 0, wall)]})
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone",
                    (2, 1, 4): "cobblestone"})
    fin, mrec = M.apply(world, _doc([rec]), _DAMP, {"condition": "weathered"}, 1,
                        "contextual", reconcile=False)
    assert fin.state(2, 1, 2) == "cobblestone", "the inside of the paving went damp"
    assert fin.state(2, 1, 3) == "andesite", "the edge of the paving should be damp"
    assert fin.state(2, 1, 4) == "andesite", "a wall base course should be damp"
    assert mrec["substituted"] == 2, mrec
    return ("ground contact with no side face open is not damp; the edge of the paving "
            "and the wall's base course still are (2 of 3)")


@case
def t_i_reconciliation_still_only_removes_and_is_still_a_fixed_point():
    F = S.FLAGS
    up, fig = F["open_up"], F["open_up"] | F["figure"]
    a = _part("a", {"wall": [(2, 1, 2, 0, up), (2, 1, 3, 0, fig), (5, 5, 5, 0, up)],
                    "floor": [(2, 0, 2, 0, up)]}, figures={"paving": 1})
    b = _part("b", {"wall": [(2, 1, 3, 0, up), (3, 1, 2, 0, up)]},
              voice="japanese_minka")
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone",
                    (2, 0, 2): "cobblestone", (3, 1, 2): "andesite"})
    doc = _doc([a, b])
    before = _kept(doc)
    out, rep = S.reconcile(doc, world)
    kept = _kept(out)
    assert kept <= before, sorted(kept - before)
    out2, rep2 = S.reconcile(out, world)
    assert _kept(out2) == kept, "a second pass removed more"
    assert {k: v for k, v in rep2.items() if k != "how"} == \
           {**{k: v for k, v in rep.items() if k != "how"},
            "recorded_editable": rep["kept"], "overwritten_by_a_later_part": 0,
            "protected_by_another_part": 0, "stale_block_no_longer_stands": 0,
            "unexposed_in_the_assembled_world": 0, "exposure_recomputed_on_kept": 0,
            "figure_cells_recorded": rep["figure_cells_kept"]}, (rep, rep2)
    # nothing the record never owned is reachable
    assert (4, 4, 4) not in kept
    return (f"{rep['recorded_editable']} recorded -> {rep['kept']} kept, a subset; the "
            f"second pass is a fixed point at {rep2['kept']}")


@case
def t_j_the_pass_is_still_deterministic_replayable_and_accumulates_nothing():
    """On a real build with a real recipe, after both edits: two applications agree,
    the record's order does not matter, the seed does, and finishing an already
    finished world from the same record substitutes nothing."""
    b, sited, res = C.probe_build("square", 24, 24, {"paving": "checker"}, seed=3,
                                  voice="ochre_stone_green_tile")
    assert res.get("ok"), res
    vol = b._vol
    world = Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(),
                   list(vol.palette)).overlay(dict(b._pending))
    doc = {"parts": [S.record(b, {**sited, "name": "square", "type": "square",
                                  "kind": "area"},
                              voice_name="ochre_stone_green_tile")]}
    sdoc, _rep = S.reconcile(doc, world)
    recipe = M.recipe_for("ochre_stone_green_tile")
    st = {"condition": "weathered"}
    a1, r1 = M.apply(world, sdoc, recipe, st, 1, "contextual", reconcile=False)
    a2, _ = M.apply(world, sdoc, recipe, st, 1, "contextual", reconcile=False)
    assert M.volume_digest(a1) == M.volume_digest(a2), "a replay differs"
    rev = {**sdoc, "parts": [{**p, "cells": {r: list(reversed(c))
                                             for r, c in (p.get("cells") or {}).items()}}
                             for p in sdoc["parts"]]}
    a3, _ = M.apply(world, rev, recipe, st, 1, "contextual", reconcile=False)
    assert M.volume_digest(a1) == M.volume_digest(a3), "the record's order mattered"
    a4, r4 = M.apply(a1, sdoc, recipe, st, 1, "contextual", reconcile=False)
    assert M.volume_digest(a1) == M.volume_digest(a4), "a second pass moved it"
    assert r4["substituted"] == 0, r4
    a5, _ = M.apply(world, sdoc, recipe, st, 2, "contextual", reconcile=False)
    if r1["substituted"]:
        assert M.volume_digest(a1) != M.volume_digest(a5), "the seed does nothing"
    # the figures the square declared are untouched in every one of those worlds
    figs = {(c[0], c[1], c[2]) for p in sdoc["parts"]
            for cells in p["cells"].values() for c in cells
            if c[4] & S.FLAGS["figure"]}
    moved = [c for c in figs if a1.state(*c) != world.state(*c)]
    assert not moved, moved[:4]
    return (f"{r1['substituted']} substituted; replay, reversed order and own-output "
            f"identical (the replay substitutes {r4['substituted']}); the seed moves it; "
            f"{len(figs)} declared figure cells untouched in all of them")


def main():
    args = list(sys.argv[1:])
    only = ""
    if "--only" in args:
        only = args[args.index("--only") + 1]
    elif args and not args[0].startswith("-"):
        only = args[0]
    run = [(n, f) for n, f in CASES if not only or only in n]
    ok = 0
    for name, fn in run:
        try:
            why = fn()
            ok += 1
            print(f"ok   {name} {why}")
        except Exception as e:                       # noqa: BLE001 -- the suite reports
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(run)} composition material cases pass")
    return 0 if ok == len(run) else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
