"""The design round's material cases: the record against the assembled world, a
control the random arm is actually matched against, and a display that can show the
substitution under judgement.

    $PY scripts/test_design_material.py            # these cases and the expression's
    $PY scripts/test_design_material.py --only b   # one of them
    $PY scripts/test_design_material.py --no-inherit

**This file extends `scripts/test_expression_material.py`; it does not supersede it.**
That file already covers ownership at the write, protection, shape-and-state
preservation, determinism, the maintained/weathered control and the recipe refusals, and
its eight cases run here by import so one command answers for both.
`surfaces.reconcile`, the per-context matched rate, and `preview`'s textured display
with its positive control.

Small and fast: one probed cottage and a few six-block volumes, under a minute, no
server and no cached world.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from ethoslm import construction as C, lint, material as M, prims, preview  # noqa: E402
from ethoslm import surfaces as S  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


# ------------------------------------------------------------------ fixtures

def _cottage(voice="drystone_and_thatch", seed=1):
    b, sited, res = C.probe_build("cottage", 20, 14, {"storeys": 2, "outshot": "byre"},
                                  seed=seed, voice=voice)
    assert res.get("ok"), res
    return b, sited


def _volume_of(b) -> Volume:
    vol = b._vol
    return Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette)).overlay(
        dict(b._pending))


#: A hand-made world and a hand-made record, so the reconciliation rules can be put in a
#: position no probed cottage happens to produce: two parts over one cell, a neighbour
#: built later against an open face, a block that was swept away.
_PAL = ["air", "cobblestone", "andesite", "oak_planks"]


def _world(cells: dict, size: int = 8) -> Volume:
    codes = np.zeros((size, size, size), np.uint16)
    for (x, y, z), name in cells.items():
        codes[x, y, z] = _PAL.index(name)
    return Volume(0, 0, 0, codes, list(_PAL))


def _part(name: str, cells: dict, *, protected=(), voice="drystone_and_thatch",
          type_="cottage", states=("cobblestone",)) -> dict:
    """One part's record: `cells` is {role: [(x, y, z, state_index, flags), ...]}."""
    return {"part": name, "type": type_, "kind": "plot", "floor_y": 0, "voice": {},
            "voice_name": voice, "states": list(states),
            "cells": {r: [list(c) for c in v] for r, v in cells.items()},
            "protected": [list(p) for p in protected],
            "counts": {r: len(v) for r, v in cells.items()}, "how": {},
            "owned": sum(len(v) for v in cells.values()) + len(protected)}


def _doc(parts: list) -> dict:
    return {"record": "surfaces", "version": 1, "flags": S.FLAGS,
            "editable": list(prims.EDITABLE_ROLES), "parts": parts}


def _kept(doc: dict) -> set:
    return {(c[0], c[1], c[2]) for p in doc["parts"]
            for cells in p["cells"].values() for c in cells}


def _kept_by(doc: dict, part: str) -> set:
    p = next(q for q in doc["parts"] if q["part"] == part)
    return {(c[0], c[1], c[2]) for cells in p["cells"].values() for c in cells}


# --------------------------------------------------- the record against the world

@case
def t_a_a_cell_a_later_part_overwrote_belongs_to_the_later_part():
    # both parts recorded (2, 1, 2); B built after A, so B's is what stands
    F = S.FLAGS["open_up"]
    a = _part("a", {"wall": [(2, 1, 2, 0, F), (2, 1, 3, 0, F)]})
    b = _part("b", {"wall": [(2, 1, 2, 0, F), (3, 1, 2, 0, F)]}, voice="japanese_minka")
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone",
                    (3, 1, 2): "cobblestone"})
    out, rep = S.reconcile(_doc([a, b]), world)
    assert (2, 1, 2) not in _kept_by(out, "a"), "the earlier part kept a cell it lost"
    assert (2, 1, 2) in _kept_by(out, "b")
    assert rep["overwritten_by_a_later_part"] == 1, rep
    assert _kept(out) == {(2, 1, 2), (2, 1, 3), (3, 1, 2)}
    return (f"{rep['overwritten_by_a_later_part']} claim(s) moved to the later part; "
            f"{rep['kept']} cells kept of {rep['recorded_editable']}")


@case
def t_b_another_parts_protected_cell_is_not_editable():
    F = S.FLAGS["open_up"]
    a = _part("a", {"wall": [(2, 1, 2, 0, F), (2, 1, 3, 0, F)]})
    b = _part("b", {"wall": [(3, 1, 2, 0, F)]},
              protected=[(2, 1, 3, 0, "step")], voice="japanese_minka")
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 3): "cobblestone",
                    (3, 1, 2): "cobblestone"})
    out, rep = S.reconcile(_doc([a, b]), world)
    assert (2, 1, 3) not in _kept(out), "a tread of the next part stayed editable"
    assert rep["protected_by_another_part"] == 1, rep
    return f"{rep['protected_by_another_part']} cell(s) held by another part's protection"


@case
def t_c_a_block_that_no_longer_stands_is_refused_and_so_is_a_changed_one():
    F = S.FLAGS["open_up"]
    a = _part("a", {"wall": [(2, 1, 2, 0, F), (2, 1, 3, 0, F), (2, 1, 4, 0, F)]})
    # (2,1,3) was swept to air; (2,1,4) is a different block now
    world = _world({(2, 1, 2): "cobblestone", (2, 1, 4): "andesite"})
    out, rep = S.reconcile(_doc([a]), world)
    assert _kept(out) == {(2, 1, 2)}, _kept(out)
    assert rep["stale_block_no_longer_stands"] == 2, rep
    return f"{rep['stale_block_no_longer_stands']} recorded blocks no longer stand"


@case
def t_d_exposure_is_recomputed_and_a_buried_cell_is_not_worth_editing():
    F = S.FLAGS
    # recorded with a south face open; the neighbour built later fills it. The second
    # cell is walled in on all six sides by that later work.
    a = _part("a", {"wall": [(2, 1, 2, 0, F["open_south"] | F["open_up"]),
                             (4, 1, 4, 0, F["open_south"])]})
    solid = {(2, 1, 2): "cobblestone", (2, 1, 3): "oak_planks", (4, 1, 4): "cobblestone"}
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        solid[(4 + d[0], 1 + d[1], 4 + d[2])] = "oak_planks"
    world = _world(solid)
    out, rep = S.reconcile(_doc([a]), world)
    assert (4, 1, 4) not in _kept(out), "a cell nothing can see stayed editable"
    assert rep["unexposed_in_the_assembled_world"] == 1, rep
    got = next(c for cells in out["parts"][0]["cells"].values() for c in cells)
    assert not got[4] & F["open_south"], "the face the neighbour closed is still open"
    assert got[4] & F["open_up"], "the face that is still open was dropped"
    return (f"{rep['unexposed_in_the_assembled_world']} buried; "
            f"{rep['exposure_recomputed_on_kept']} kept cell(s) changed exposure")


@case
def t_e_reconciliation_only_ever_removes_and_never_reaches_unowned_ground():
    b, sited = _cottage()
    vol = _volume_of(b)
    raw = {"parts": [S.record(b, sited, voice_name="drystone_and_thatch")]}
    out, rep = S.reconcile(raw, vol)
    before, after = _kept(raw), _kept(out)
    assert after <= before, "reconciliation invented a cell"
    assert rep["kept"] == len(after) == S.editable_cells(out)
    # the ground the library laid is protected, and it is not in the editable set
    ground = {(c[0], c[1], c[2]) for c in raw["parts"][0]["protected"] if c[4] == "ground"}
    assert ground and not (ground & after)
    # reconciling a reconciled record is a fixed point
    again, rep2 = S.reconcile(out, vol)
    assert _kept(again) == after and rep2["kept"] == rep["kept"]
    return (f"{len(before)} recorded -> {len(after)} editable ("
            f"{rep['unexposed_in_the_assembled_world']} unseen, "
            f"{rep['protected_by_another_part']} protected); reconciling twice is a "
            f"fixed point; {len(ground)} ground cells never reachable")


@case
def t_f_the_pass_is_computed_from_the_structural_base_and_never_accumulates():
    b, sited = _cottage()
    vol = _volume_of(b)
    raw = {"parts": [S.record(b, sited, voice_name="drystone_and_thatch")]}
    recipe = M.recipe_for("drystone_and_thatch")
    st = {"condition": "weathered"}
    fin, r1 = M.apply(vol, raw, recipe, st, 1, "contextual")
    assert r1["substituted"] > 0 and r1["reconciled"]["kept"] < r1["reconciled"]["recorded_editable"]
    # the plan is a function of the record and the recipe alone: the same plan whether
    # it is about to be written into the base or into an already finished world
    doc, _ = S.reconcile(raw, vol)
    p1 = M.plan(doc, recipe, st, 1, "contextual")
    p2 = M.plan(doc, recipe, st, 1, "contextual")
    assert p1["cells"] == p2["cells"]
    again, r2 = M.apply(fin, doc, recipe, st, 1, "contextual", reconcile=False)
    assert r2["substituted"] == 0, "a replay wrote over its own output"
    assert M.volume_digest(again) == M.volume_digest(fin)
    # and the base still produces the same world, in any order, at any worker count
    rev = {**doc, "parts": [{**p, "cells": {k: list(reversed(v))
                                            for k, v in p["cells"].items()}}
                            for p in doc["parts"]]}
    other, _ = M.apply(vol, rev, recipe, st, 1, "contextual")
    assert M.volume_digest(other) == M.volume_digest(fin)
    return (f"{r1['substituted']} substituted from the base; a replay over the finished "
            f"world substitutes {r2['substituted']}; order-independent")


@case
def t_g_solid_occupancy_is_identical_and_the_check_finds_nothing_new():
    b, sited = _cottage()
    vol = _volume_of(b)
    raw = {"parts": [S.record(b, sited, voice_name="drystone_and_thatch")]}
    fin, r = M.apply(vol, raw, M.recipe_for("drystone_and_thatch"),
                     {"condition": "weathered"}, 1, "contextual")

    def solid(v):
        t = v.tables()
        return (t["lower"].astype(bool) | t["upper"].astype(bool))[v.codes]
    assert (solid(vol) == solid(fin)).all(), "the pass moved a solid"
    before = sorted(f"{x.code}@{x.pos}" for x in lint.lint(lint.Context.build(vol, [])).errors)
    after = sorted(f"{x.code}@{x.pos}" for x in lint.lint(lint.Context.build(fin, [])).errors)
    new = sorted(set(after) - set(before))
    assert not new, new[:6]
    return (f"{r['substituted']} substitutions; occupancy identical; "
            f"{len(before)} construction error(s) before and {len(after)} after")


# --------------------------------------------------- the control the random arm gets

@case
def t_h_random_is_matched_per_voice_and_per_type_not_pooled_by_role():
    # two voices over the same geometry: only the first has a recipe for `wall`
    F = S.FLAGS["open_up"] | S.FLAGS["open_south"]
    cells = {"wall": [(x, 1, z, 0, F) for x in range(2, 7) for z in range(2, 7)]}
    a = _part("a", cells, voice="drystone_and_thatch")
    b = _part("b", {"wall": [(x, 2, z, 0, F) for x in range(2, 7) for z in range(2, 7)]},
              voice="dark_timber_and_tile", type_="row_house")
    world = _world({(x, y, z): "cobblestone" for x in range(2, 7) for z in range(2, 7)
                    for y in (1, 2)}, size=10)
    doc, _ = S.reconcile(_doc([a, b]), world)
    recipe = {"by_voice": {"drystone_and_thatch": M.recipe_for("drystone_and_thatch"),
                           "dark_timber_and_tile": M.recipe_for("dark_timber_and_tile")}}
    assert recipe["by_voice"]["dark_timber_and_tile"] == {}, "the control voice has a recipe"
    st = {"condition": "weathered"}
    pc = M.plan(doc, recipe, st, 1, "contextual")
    pr = M.plan(doc, recipe, st, 1, "random")
    for p, name in ((pc, "contextual"), (pr, "random")):
        on_b = [c for c in p["cells"] if c[1] == 2]
        assert not on_b, f"{name} put {len(on_b)} variants on a voice with no recipe"
    # and the rate is matched inside the context, not across the place
    for key, fams in pc["bucket_rates"].items():
        for fam, rate in fams.items():
            got = (pr["matched_bucket_rates"].get(key) or {}).get(fam, 0.0)
            assert abs(got - rate) <= max(0.08, 0.5 * rate), (key, fam, rate, got)
    assert M.context_of(a) != M.context_of(b)
    return (f"contextual {json_short(pc['bucket_rates'])}; random matched "
            f"{json_short(pr['matched_bucket_rates'])}; the second voice is untouched "
            f"in both arms")


@case
def t_i_the_contextual_arm_is_the_more_coherent_at_the_same_rate():
    b, sited = _cottage()
    vol = _volume_of(b)
    doc, _ = S.reconcile({"parts": [S.record(b, sited,
                                             voice_name="drystone_and_thatch")]}, vol)
    recipe = M.recipe_for("drystone_and_thatch")
    st = {"condition": "weathered"}
    pc = M.plan(doc, recipe, st, 1, "contextual")
    pr = M.plan(doc, recipe, st, 1, "random")

    def coherence(p):
        fam = {c: v[1] for c, v in p["cells"].items()}
        tot = same = 0
        for (x, y, z), f in fam.items():
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                tot += 1
                same += fam.get((x + d[0], y + d[1], z + d[2])) == f
        return same / max(1, tot)
    kc, kr = coherence(pc), coherence(pr)
    assert kc > kr, (kc, kr)
    assert abs(len(pc["cells"]) - len(pr["cells"])) <= 0.25 * len(pc["cells"]), \
        (len(pc["cells"]), len(pr["cells"]))
    return (f"same-family neighbour share: contextual {kc:.3f} vs random {kr:.3f} on "
            f"{len(pc['cells'])} and {len(pr['cells'])} substitutions")


# ------------------------------------------------------------------ the display

@case
def t_j_every_material_family_resolves_to_a_texture():
    if not preview.texture_names():
        return "no client jar on this host; the textured display falls back to flat"
    miss = [(fam, b) for fam, shapes in prims.MATERIALS.items() for b in shapes
            if b and preview.texture_name(str(b)) is None]
    assert not miss, miss[:8]
    # a block the atlas has no file for falls back to exactly the flat colour, so a
    # textured frame is never *less* complete than a flat one
    t = preview._tile("decorated_pot", 4)
    assert t.shape == (4, 4, 3)
    assert (t == np.array(preview.block_colour("decorated_pot"), np.float32)).all()
    return (f"{len(prims.MATERIALS)} families x 3 shapes resolve; "
            f"{len(preview.texture_names())} textures in the atlas; an unfiled block "
            f"falls back to its flat colour")


@case
def t_k_the_textured_display_registers_the_pair_the_flat_one_collapses():
    """The positive control in miniature: a wall of cobblestone against the same wall
    in andesite. The flat table puts them nine units apart; a person does not."""
    if not preview.texture_names():
        return "no client jar on this host; the control cannot run"
    a = _world({(x, y, 4): "cobblestone" for x in range(1, 7) for y in range(1, 5)})
    b = _world({(x, y, 4): "andesite" for x in range(1, 7) for y in range(1, 5)})
    got = {}
    for texture, tag in ((False, "flat"), (True, "textured")):
        ia = preview.elevation(a, facing="south", scale=16, clip=False, texture=texture)
        ib = preview.elevation(b, facing="south", scale=16, clip=False, texture=texture)
        assert ia.shape == ib.shape
        got[tag] = int(np.abs(ia.astype(int) - ib.astype(int)).max())
        assert (ia == preview.elevation(a, facing="south", scale=16, clip=False,
                                        texture=texture)).all(), "the display wobbles"
    assert got["flat"] <= 12, got          # the defect the review named, as a number
    assert got["textured"] >= 40, got      # and the display that does not have it
    return (f"cobblestone against andesite: the flat display differs by at most "
            f"{got['flat']}/255, the textured one by {got['textured']}/255")


@case
def t_l_the_flat_display_is_unchanged_by_the_textured_path():
    """A flat frame is still one colour a block off `block_colour`, shaded -- the judge
    caches on image content and a renderer that quietly moved would re-buy every
    judgement it has."""
    v = _world({**{(x, y, 4): "cobblestone" for x in range(1, 5) for y in range(1, 3)},
                **{(x, 3, 2): "cobblestone" for x in range(1, 5)}})   # a course set back
    img = preview.elevation(v, facing="south", scale=1, clip=False)
    seen = {tuple(p) for row in img for p in row}
    want = np.array(preview.block_colour("cobblestone"), float)
    for p in seen:
        if p == (preview.BG,) * 3:
            continue
        ratio = np.array(p, float) / np.maximum(want, 1)
        assert abs(ratio.max() - ratio.min()) < 0.02, (p, want)
    top = preview.top_down(v, scale=1)
    assert top.shape[2] == 3 and (top != preview.UNKNOWN).any()
    return f"{len(seen)} colours in a flat frame, each a shade of the block's own"


# --------------------------------------------- the clearer against a laid frame
# `prims.py` is C's, and `scripts/test_prims.py` needs a live backend, so the regression
# for the design round's `clear_trees` fix lives here with the other prims-level cases.
# `Builder.site()` calls `clear_trees` for every part with a margin, so the box reaches
# into whatever a neighbour already built.

def _scene(b):
    """A worldgen oak with a canopy, and a type's frame post standing under it."""
    v = b._vol
    X, Z = v.x0 + 6, v.z0 + 6
    g = b.get_height(X, Z)
    for y in range(g + 1, g + 5):
        b.place_block(X, y, Z, "oak_log")
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            b.place_block(X + dx, g + 5, Z + dz, "oak_leaves")
    post = (X + 2, Z)
    for y in range(g + 1, g + 6):
        b.place_block(post[0], y, post[1], prims.shape("spruce", "post"))
    return X, Z, g, post


@case
def t_m_a_frame_post_is_not_a_trunk_and_the_tree_still_comes_out_whole():
    assert prims.shape("spruce", "post") == "stripped_spruce_log"
    b, _sited = _cottage()
    X, Z, g, post = _scene(b)
    # the box does not contain the post; the flood reaches it from the canopy
    removed = b.clear_trees(X - 1, Z - 1, X + 1, Z + 1, margin=1)
    standing = [b.get_block(post[0], y, post[1]) for y in range(g + 1, g + 6)]
    assert all(s == prims.shape("spruce", "post") for s in standing), standing
    assert b.get_block(X, g + 2, Z) == "air", "the trunk stayed"
    assert b.get_block(X + 2, g + 5, Z + 2) == "air", "a floating leaf was left"
    # the control: the same scene with the exclusion disabled is the reported defect
    was = prims.WORKED_TIMBER
    prims.WORKED_TIMBER = "\x00-no-such-prefix"
    try:
        c, _s = _cottage()
        X2, Z2, g2, p2 = _scene(c)
        c.clear_trees(X2 - 1, Z2 - 1, X2 + 1, Z2 + 1, margin=1)
        gone = [c.get_block(p2[0], y, p2[1]) for y in range(g2 + 1, g2 + 6)]
    finally:
        prims.WORKED_TIMBER = was
    assert all(s == "air" for s in gone), gone
    return (f"{removed} tree blocks out, post 5/5 standing; with the exclusion off the "
            f"same call takes all five posts")


@case
def t_n_a_frame_post_inside_the_box_is_not_seeded_either():
    b, _sited = _cottage()
    v = b._vol
    X, Z = v.x0 + 6, v.z0 + 6
    g = b.get_height(X, Z)
    for y in range(g + 1, g + 6):                    # a lone post, no tree anywhere
        b.place_block(X, y, Z, prims.shape("dark_oak", "post"))
    removed = b.clear_trees(X - 3, Z - 3, X + 3, Z + 3, margin=2)
    assert removed == 0, removed
    assert b.get_block(X, g + 3, Z) == prims.shape("dark_oak", "post")
    # not one voice's problem: every post and bare face the library can lay is either a
    # stripped log the clearer now excludes, or a block it never reads as tree at all
    flood = ("_log", "_wood", "_leaves", "_stem", "vine", "shroomlight", "_hyphae",
             "_propagule")
    laid = {}
    for fam in sorted(prims.MATERIALS):
        for kind in ("post", "bare"):
            try:
                blk = str(prims.shape(fam, kind)).split("[")[0].split(":")[-1]
            except ValueError:
                continue
            laid[(fam, kind)] = blk
            reads_as_tree = (blk.endswith(("_log", "_stem", "_wood"))
                             or any(v in blk for v in flood))
            assert blk.startswith(prims.WORKED_TIMBER) or not reads_as_tree, (fam, kind, blk)
    stripped = sorted({b for b in laid.values() if b.startswith(prims.WORKED_TIMBER)})
    return (f"a lone frame post seeds nothing ({removed} removed); of "
            f"{len(laid)} post/bare faces the library lays, {len(stripped)} are stripped "
            f"logs and the rest read as no kind of tree")


# ------------------------------------------- a fitting that lost its own piece Also
# `prims.py` and `buildlib.py`, both C's. `fitting` reports which of its cells **are**
# the piece as against the stone it lays round itself, and `TypeBuilder`'s flight-way
# rule refuses the piece rather than emptying it and answering `ok`.

@case
def t_o_a_fitting_says_which_cells_are_the_piece_and_which_it_lays_round_itself():
    b, _sited = _cottage()
    v = b._vol
    X, Y, Z = v.x0 + 40, 70, v.z0 + 40
    got = {}
    for kind in sorted(prims.Primitives.FITTING_BLOCKS):
        r = b.fitting(kind, X, Y, Z, "north", mat="cobblestone", extent=2,
                      room="hall", dry=True)
        assert "defining" in r and r["defining"], (kind, r)
        assert set(map(tuple, r["defining"])) <= set(map(tuple, r["cells"])), kind
        got[kind] = len(r["defining"])
    # the hearth is the case that cost the round: one cell of six is the fire
    h = b.fitting("hearth", X, Y, Z, "north", mat="cobblestone", dry=True)
    assert len(h["cells"]) == 6 and len(h["defining"]) == 1, h
    assert tuple(h["defining"][0]) == (X, Y, Z)
    return (f"all {len(got)} kinds name their own cells; a hearth is "
            f"{len(h['defining'])} of {len(h['cells'])} -- the fire, not the surround")


@case
def t_p_the_flight_way_refuses_the_piece_instead_of_emptying_it():
    """A hearth whose campfire lands on the way to a flight used to come back `ok`
    with the fire quietly removed, so `cottage` recorded a hearth and stopped looking.
    Now it is refused and the caller walks on to the next cell."""
    b, sited = _cottage()
    tb = b.type_builder(dict(sited))          # the rule lives on the type's library
    v = b._vol
    X, Y, Z = v.x0 + 40, 70, v.z0 + 40
    dry = tb.fitting("hearth", X, Y, Z, "north", mat="cobblestone", dry=True)
    assert dry.get("ok"), dry
    # stand the flight's way exactly on the fire, then on the surround instead
    fire = tuple(dry["defining"][0])
    surround = next(tuple(c) for c in dry["cells"] if tuple(c) != fire)
    b.flight_way = {fire}
    r = tb.fitting("hearth", X, Y, Z, "north", mat="cobblestone")
    assert r["ok"] is False and r["defining_refused"] is True, r
    assert list(r["cell"]) == list(fire) and not r["cells"], r
    assert b.get_block(*fire).split("[")[0] != "campfire", "it placed it anyway"
    assert "the piece itself" in r["reason"], r["reason"]
    assert tb.refused and tb.refused[-1]["call"] == "fitting"
    # a dry offer agrees with the real call, so a caller cannot be told yes and then no
    assert tb.fitting("hearth", X, Y, Z, "north", mat="cobblestone",
                      dry=True)["ok"] is False
    b.flight_way = {surround}
    r2 = tb.fitting("hearth", X, Y, Z, "north", mat="cobblestone")
    assert r2["ok"] is True, r2               # masonry is still taken back quietly
    assert b.get_block(*fire).split("[")[0] == "campfire", "the fire went missing"
    assert b.get_block(*surround).split("[")[0] != "cobblestone"
    return ("the fire on the flight's way is a refusal naming the cell, dry and wet "
            "alike; the surround on it is still taken back quietly and the fire stands")


@case
def t_r_a_fitting_does_not_shut_a_room_off_from_its_own_door():
    """The other side of the refusal above: with the fire going in again, two of
    `des-farm`'s cottages put it in the cell their upper room was reached through."""
    from ethoslm import buildlib as BL
    b, sited = _cottage()
    tb = b.type_builder(dict(sited))
    rect = tb._fitting_rect()
    assert rect and tuple(rect) == (min(sited["x0"], sited["x1"]),
                                    min(sited["z0"], sited["z1"]),
                                    max(sited["x0"], sited["x1"]),
                                    max(sited["z0"], sited["z1"]))
    # the guard blames only what this fitting did: a walk that was already shut is not
    # turned into a refusal
    calls = []
    real = BL.Builder.check_walkable

    def shut(self, *a, **k):
        calls.append(1)
        return {"ok": False, "rooms": [{"bbox": [0, 0, 0, 1, 1, 1], "cells": 9,
                                        "walkable": 0}], "reason": "shut"}
    v = b._vol
    X, Y, Z = v.x0 + 40, 70, v.z0 + 40
    for dx in range(0, 8):                   # a floor for the fixture to stand on
        b.place_block(X + dx, Y - 1, Z, "cobblestone")
    BL.Builder.check_walkable = shut
    try:
        r = tb.fitting("store", X, Y, Z, "north", mat="cobblestone")
        assert r["ok"] is True, r          # before is shut too, so nothing is blamed
    finally:
        BL.Builder.check_walkable = real
    assert calls, "the guard never asked"
    # and it does refuse where the walk was open and this fitting shut it
    seq = [{"ok": True, "rooms": []},
           {"ok": False, "rooms": [{"bbox": [1, 2, 3, 4, 5, 6], "cells": 30,
                                    "walkable": 0}], "reason": "shut"}]
    BL.Builder.check_walkable = lambda self, *a, **k: seq.pop(0) if seq else \
        {"ok": True, "rooms": []}
    try:
        X = v.x0 + 44
        dry = tb.fitting("store", X, Y, Z, "north", mat="cobblestone", dry=True)
        assert dry["ok"], dry
        r = tb.fitting("store", X, Y, Z, "north", mat="cobblestone")
        assert r["ok"] is False and r["defining_refused"] is True, r
        assert "shut a room off from its own door" in r["reason"], r["reason"]
        assert "[1, 2, 3, 4, 5, 6]" in r["reason"], r["reason"]
        for c in dry["cells"]:               # and it took every cell back out
            assert b._pending.get(tuple(c)) in (None, "air"), (c, b._pending.get(tuple(c)))
        assert tb.refused and "room" in tb.refused[-1]["reason"]
    finally:
        BL.Builder.check_walkable = real
    return ("a walk already shut is not blamed on the fitting; a walk this fitting "
            "shuts is a refusal naming the room, with every cell taken back out")


@case
def t_q_every_probe_cottage_lays_a_fire():
    """The functional number this fix bought, measured on the built blocks rather than
    on the certificate: 12 seeds at each of one, two and three storeys."""
    got = {}
    for storeys in (1, 2, 3):
        n = 0
        for seed in range(1, 5):             # four seeds a storey here; the full
            b, _s, res = C.probe_build(      # twelve is out/des-work/C/fire_sweep.py
                "cottage", 20, 14, {"storeys": storeys}, seed=seed,
                voice="drystone_and_thatch")
            assert res.get("ok"), (storeys, seed, res)
            n += any(s.split("[")[0] == "campfire" for s in b._pending.values())
        got[storeys] = n
        assert n == 4, (storeys, n)
    return f"a fire in every one: {got} of 4 seeds at one, two and three storeys"


def json_short(d: dict, n: int = 110) -> str:
    import json
    return json.dumps(d)[:n]


def main():
    args = [a for a in sys.argv[1:]]
    inherit = "--no-inherit" not in args
    only = ""
    if "--only" in args:
        only = args[args.index("--only") + 1]
    elif args and not args[0].startswith("-"):
        only = args[0]
    cases = list(CASES)
    if inherit:
        import test_expression_material as X
        cases = [(f"x_{n}", f) for n, f in X.CASES] + cases
    ok = 0
    run = [(n, f) for n, f in cases if not only or only in n]
    for name, fn in run:
        try:
            why = fn()
            ok += 1
            print(f"ok   {name} {why}")
        except Exception as e:                       # noqa: BLE001 -- the suite reports
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(run)} design material cases pass"
          f"{' (with the expression round''s, inherited)' if inherit else ''}")
    return 0 if ok == len(run) else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
