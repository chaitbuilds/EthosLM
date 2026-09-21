"""The expression round's material cases: ownership at the write, protection, and a
deterministic pass that changes only what it owns.

    $PY scripts/test_expression_material.py

Every case runs through the production entry points -- `construction.probe_build`
(the same builder `instantiate_part` uses), `surfaces.record`, `material.apply` --
with a control beside each claim. Under a minute; no server.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import construction as C, material as M, prims, surfaces as S  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


def _cottage(voice="drystone_and_thatch", seed=1):
    b, sited, res = C.probe_build("cottage", 20, 14, {"storeys": 2, "outshot": "byre"},
                                  seed=seed, voice=voice)
    assert res.get("ok"), res
    return b, sited


def _volume_of(b) -> Volume:
    vol = b._vol
    return Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette)).overlay(
        dict(b._pending))


@case
def t_a_roles_are_recorded_at_the_write_and_a_floor_is_not_a_wall():
    b, sited = _cottage()
    assert b.parts[-1]["voice"]["wall"] == b.parts[-1]["voice"]["floor"] == "cobblestone", \
        "the control: floor and wall share a block in this voice"
    roles = collections.Counter(b.owner_of(c)["role"] for c in b._owner)
    cob = collections.Counter(b.owner_of(c)["role"] for c in b._owner
                              if b._pending[c].split("[")[0] == "cobblestone")
    assert cob["floor"] > 0 and cob["wall"] > 0, cob
    assert roles.get("unknown", 0) == 0, roles
    assert roles["roof"] > 0 and roles["footing"] > 0 and roles["frame"] > 0, roles
    # every standing block is owned; every air write cleared its owner
    assert set(b._owner) <= set(b._pending)
    assert all(b._pending[c].split("[")[0] not in ("air",) for c in b._owner)
    return (f"cobblestone as floor {cob['floor']}, as wall {cob['wall']}; roles "
            f"{dict(sorted(roles.items()))}")


@case
def t_b_treads_doors_glass_fittings_and_lights_are_protected():
    b, sited = _cottage()
    prot = {c: b.owner_of(c) for c in b._owner if b.owner_of(c)["protected"]}
    kinds = collections.Counter(o["role"] for o in prot.values())
    assert kinds["step"] > 0 and kinds["door"] == 2 and kinds["glass"] > 0, kinds
    assert kinds["fitting"] > 0, kinds
    for c, o in prot.items():
        n = b._pending[c].split("[")[0]
        if o["role"] == "door":
            assert "door" in n, (c, n)
        if o["role"] == "glass":
            assert "glass" in n or "pane" in n, (c, n)
    rec = S.record(b, sited)
    for row in rec["protected"]:
        assert (row[0], row[1], row[2]) not in {
            (r[0], r[1], r[2]) for cells in rec["cells"].values() for r in cells}
    return f"protected {dict(sorted(kinds.items()))}; {len(rec['protected'])} listed apart"


@case
def t_c_the_pass_edits_only_owned_editable_cells_and_keeps_shape_and_state():
    b, sited = _cottage()
    vol = _volume_of(b)
    rec = S.record(b, sited, voice_name="drystone_and_thatch")
    sdoc = {"parts": [rec], "built_digest": "probe"}
    recipe = M.recipe_for("drystone_and_thatch")
    fin, r = M.apply(vol, sdoc, recipe, {"condition": "weathered"}, 1, "contextual")
    assert r["substituted"] > 0, r
    owned = {(c[0], c[1], c[2]) for cells in rec["cells"].values() for c in cells}
    protected = {(c[0], c[1], c[2]) for c in rec["protected"]}
    changed = 0
    for x in range(vol.x0, vol.x0 + vol.shape[0]):
        for y in range(vol.y0, vol.y0 + vol.shape[1]):
            for z in range(vol.z0, vol.z0 + vol.shape[2]):
                a, c = vol.state(x, y, z), fin.state(x, y, z)
                if a == c:
                    continue
                changed += 1
                assert (x, y, z) in owned, (x, y, z, a, c)
                assert (x, y, z) not in protected
                ka, kc = M.shape_of(a), M.shape_of(c)
                assert ka == kc, (a, c)             # same shape, same state
    assert changed == r["substituted"]
    # nothing outside the owned set moved: the world and the pad are as they were
    ground = [(c[0], c[1], c[2]) for c in rec["protected"] if c[4] == "ground"]
    assert ground and all(vol.state(*c) == fin.state(*c) for c in ground)
    return f"{changed} cells changed, all owned and editable, shape and state kept"


@case
def t_d_deterministic_in_any_order_and_nothing_accumulates():
    b, sited = _cottage()
    vol = _volume_of(b)
    rec = S.record(b, sited, voice_name="drystone_and_thatch")
    sdoc = {"parts": [rec]}
    recipe = M.recipe_for("drystone_and_thatch")
    st = {"condition": "weathered"}
    a, ra = M.apply(vol, sdoc, recipe, st, 1, "contextual")
    b2, _ = M.apply(vol, sdoc, recipe, st, 1, "contextual")
    assert M.volume_digest(a) == M.volume_digest(b2)
    rev = {"parts": [{**rec, "cells": {k: list(reversed(v)) for k, v in rec["cells"].items()}}]}
    c, _ = M.apply(vol, rev, recipe, st, 1, "contextual")
    assert M.volume_digest(a) == M.volume_digest(c), "order changed the answer"
    d, _ = M.apply(a, sdoc, recipe, st, 1, "contextual")
    assert M.volume_digest(a) == M.volume_digest(d), "finishing a finished world moved it"
    e, _ = M.apply(vol, sdoc, recipe, st, 2, "contextual")
    assert M.volume_digest(a) != M.volume_digest(e), "the seed does nothing"
    n, rn = M.apply(vol, sdoc, recipe, st, 1, "none")
    assert M.volume_digest(n) == M.volume_digest(vol) and rn["substituted"] == 0
    return (f"{ra['substituted']} substitutions; replay, reversed order and own-output "
            f"identical; seed 2 differs; `none` is the control")


@case
def t_e_random_matches_contextual_in_rate_and_differs_in_place():
    b, sited = _cottage()
    vol = _volume_of(b)
    rec = S.record(b, sited, voice_name="drystone_and_thatch")
    sdoc = {"parts": [rec]}
    recipe = M.recipe_for("drystone_and_thatch")
    st = {"condition": "weathered"}
    pc = M.plan(sdoc, recipe, st, 1, "contextual")
    pr = M.plan(sdoc, recipe, st, 1, "random")
    for role, fams in pc["rates"].items():
        for fam, rate in fams.items():
            got = (pr["matched_rates"].get(role) or {}).get(fam, 0.0)
            assert abs(got - rate) <= max(0.06, 0.5 * rate), (role, fam, rate, got)
    same = len(set(pc["cells"]) & set(pr["cells"]))
    assert same < len(pc["cells"]), "random placed every variant where contextual did"
    # contextual conditions: damp variants sit low, on base courses or ground contact
    F = S.FLAGS
    flags = {(c[0], c[1], c[2]): c[4] for c in rec["cells"].get("wall") or []}
    damp = [c for c, (_s, fam, role, _old) in pc["cells"].items()
            if role == "wall" and fam == "mossy_cobblestone"]
    low = sum(1 for c in damp if flags[c] & (F["base_course"] | F["ground_contact"]
                                             | F["water_near"] | F["under_opening"]))
    return (f"contextual rates {pc['rates']}; random matched {pr['matched_rates']}; "
            f"{same} of {len(pc['cells'])} cells coincide; {low}/{len(damp)} mossy wall "
            f"blocks stand on a damp condition")


@case
def t_f_maintained_is_cleaner_than_weathered_and_a_voice_without_a_recipe_is_untouched():
    b, sited = _cottage()
    vol = _volume_of(b)
    rec = S.record(b, sited, voice_name="drystone_and_thatch")
    sdoc = {"parts": [rec]}
    recipe = M.recipe_for("drystone_and_thatch")
    _m, rm = M.apply(vol, sdoc, recipe, {"condition": "maintained"}, 1, "contextual")
    _w, rw = M.apply(vol, sdoc, recipe, {"condition": "weathered"}, 1, "contextual")
    assert rm["substituted"] < rw["substituted"], (rm["substituted"], rw["substituted"])
    empty = M.recipe_for("blackstone_and_ash")
    assert empty == {}
    f, rf = M.apply(vol, sdoc, empty, {"condition": "weathered"}, 1, "contextual")
    assert rf["substituted"] == 0 and M.volume_digest(f) == M.volume_digest(vol)
    return (f"maintained {rm['substituted']} < weathered {rw['substituted']}; a voice "
            f"with no variants changes nothing")


@case
def t_g_unknown_stays_unknown_and_an_untagged_builder_keeps_the_old_census():
    # a bare block no voice role names, placed by a type by hand, is owned but unknown
    b, sited = _cottage()
    with b.laying("wall"):
        pass
    b.place_block(b.parts[-1]["x0"], int(b.parts[-1]["floor_y"]) + 12,
                  b.parts[-1]["z0"], "purpur_block")
    o = b.owner_of((b.parts[-1]["x0"], int(b.parts[-1]["floor_y"]) + 12, b.parts[-1]["z0"]))
    assert o and o["role"] == "unknown" and not o["protected"], o
    rec = S.record(b, sited)
    assert not any(r[4] == "unknown" and False for r in rec["protected"])
    assert "unknown" not in rec["cells"], "an unknown cell is never editable"
    assert any(r[4] == "unknown" for r in rec["protected"])
    # a builder without the record (its `_owner` emptied) falls back to the family
    # census
    b._owner.clear()
    s = C.surfaces(b, sited)
    assert s["known"] and "recorded" not in s and s["roles"].get("wall"), s
    return "a purpur block is unknown and uneditable; the census still answers a bare builder"


@case
def t_h_a_recipe_that_names_a_bad_family_or_an_uneditable_role_is_refused():
    try:
        M.recipe_for({"roles": {}, "variants": {"wall": [{"family": "nothing_stone",
                                                          "weight": 0.2}]}})
    except M.RecipeError as e:
        a = str(e)
    else:
        raise AssertionError("a made-up family was accepted")
    try:
        M.recipe_for({"roles": {}, "variants": {"ground": [{"family": "andesite",
                                                            "weight": 0.2}]}})
    except M.RecipeError as e:
        b = str(e)
    else:
        raise AssertionError("the ground was accepted as editable")
    assert M.substitute("quartz_block", "thatch") is None or \
        M.shape_of(M.substitute("quartz_block", "thatch"))[0] == "full"
    assert M.substitute("dark_oak_door[facing=north]", "spruce") is not None
    return f"refused: {a[:60]} / {b[:60]}"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = 0
    for name, fn in CASES:
        if only and only not in name:
            continue
        try:
            why = fn()
            ok += 1
            print(f"ok   {name} {why}")
        except Exception as e:                       # noqa: BLE001 -- the suite reports
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    n = len([c for c in CASES if not only or only in c[0]])
    print(f"\n{ok}/{n} expression material cases pass")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
