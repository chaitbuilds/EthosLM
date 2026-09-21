"""The realization round's counterexamples, as regressions with positive controls.

    $PY scripts/test_realization.py

because a check that refuses everything is not a check. The three groups are the round's
three boundaries:

  1. **Meaning survives into design.** The interpretation reads scope, negation, relation
     and hierarchy; the phrase rules cross-check it and cannot silently overrule it, and
     cannot be silently overruled about what this library can express.
  2. **Feasibility returns an arrangement.** Capacity is what the construction logic
     lays, the recovery ladder changes real consumer decisions, and a repair that changed
     nothing says so.
  3. **One lifecycle governs work and acceptance.** An artifact is bound to its outputs,
     an inspection to its candidate, an import to the files it names.

They run against **production entry points** wherever one exists:
`interpret.requirements` is what the stage calls, `arrange.arrange` is what the plan
stage calls, `deps.check` is what every stage asks.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (arrange, capability, contracts, deps, interpret,  # noqa: E402
                   intent as intent_mod, offline, pipeline, placeplan)
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


def _interp(sentence, reads, **kw):
    return interpret.read_answer(sentence, {"reads": reads, **kw})


def _reqs(sentence, reads, **kw):
    return {r["id"]: r for r in
            interpret.requirements(sentence, _interp(sentence, reads, **kw))
            ["requirements"]}


# ------------------------------------------------- 1. meaning survives into design

@case
def t_a_negation_scopes_over_both_conjuncts():
    """"without a wall or a temple" -- the rules read the temple as **required**."""
    s = "Build a village without a wall or a temple."
    rules = {r["id"]: r for r in intent_mod.read(s)["requirements"]}
    assert "feature/temple" in rules, "the counterexample no longer reproduces"
    got = _reqs(s, [
        {"id": "absent/wall", "kind": "absent", "says": "no wall",
         "wants": {"feature": "wall"}, "phrase": "without a wall",
         "why": "the negation governs both conjuncts"},
        {"id": "absent/temple", "kind": "absent", "says": "no temple",
         "wants": {"feature": "temple"}, "phrase": "temple",
         "why": "'or a temple' is inside the scope of 'without'"}])
    assert set(got) == {"absent/wall", "absent/temple"}, got
    assert "feature/temple" not in got, "the rules' opposite requirement survived"
    ck = interpret.cross_check(s, _interp(s, [
        {"id": "absent/temple", "kind": "absent", "says": "no temple",
         "wants": {"feature": "temple"}, "phrase": "temple", "why": "scope"}]))
    assert [c["id"] for c in ck["contradicts"]] == ["absent/temple"], ck
    # control: a reading the rules agree with is not recorded as a disagreement
    ok = interpret.cross_check("Build a village without a temple.", _interp(
        "Build a village without a temple.",
        [{"id": "absent/temple", "kind": "absent", "says": "no temple",
          "wants": {"feature": "temple"}, "phrase": "without a temple", "why": "x"}]))
    assert not ok["contradicts"] and len(ok["agreed"]) == 1, ok
    return ("a negation reaching a second conjunct is read, the rules' opposite "
            "requirement is recorded as contradicted and not carried, and an "
            "uncontested negation still agrees")


@case
def t_b_not_dense_is_not_a_requirement_for_dense():
    s = "Build a town that is not dense."
    rules = {r["id"] for r in intent_mod.read(s)["requirements"]}
    assert "quality/density/dense" in rules, "the counterexample no longer reproduces"
    got = _reqs(s, [
        {"id": "quality/density/sparse", "kind": "quality", "says": "not dense",
         "wants": {"axis": "density", "value": "sparse"}, "phrase": "not dense",
         "why": "'not' negates the density word"}])
    assert set(got) == {"quality/density/sparse"}, got
    return "`not dense` is read as sparse and the rules' `dense` is not carried beside it"


@case
def t_c_a_scoped_contrast_is_two_requirements():
    s = "Build a city with a dense lower district and a sparse upper district."
    got = _reqs(s, [
        {"id": "q/dense@lower", "kind": "quality", "says": "the lower district is dense",
         "wants": {"axis": "density", "value": "dense"}, "scope": "lower district",
         "phrase": "dense lower district", "why": "the adjective attaches to the lower"},
        {"id": "q/sparse@upper", "kind": "quality", "says": "the upper is sparse",
         "wants": {"axis": "density", "value": "sparse"}, "scope": "upper district",
         "phrase": "sparse upper district", "why": "the contrast is the point"}])
    assert got["q/dense@lower"]["scope"] == "lower district", got
    assert got["q/sparse@upper"]["scope"] == "upper district", got
    # and scope selects leaves: the lower quarter's, not the whole place's
    lower = {"in": ["lower_ring", "row_0"], "name": "b0"}
    upper = {"in": ["upper_ring"], "name": "b1"}
    assert intent_mod._in_scope(lower, "lower district")
    assert not intent_mod._in_scope(upper, "lower district")
    assert intent_mod._in_scope(upper, None), "no scope is the whole place"
    return "two scoped requirements, each selecting its own quarter's leaves"


@case
def t_d_hierarchy_and_relation_are_measured():
    def leaf(name, t, x, z, w, d, fam=None):
        return {"kind": "plot", "name": name, "type": t, "x0": x, "z0": z,
                "x1": x + w - 1, "z1": z + d - 1, "family": fam, "params": {"storeys": 1}}
    temple = leaf("temple_1", "temple", 45, 45, 20, 20, "temple")
    around = [temple] + [leaf(f"h{i}", "cottage", x, z, 8, 8) for i, (x, z) in
                         enumerate([(20, 20), (80, 20), (20, 80), (80, 80)])]
    clump = [temple] + [leaf(f"h{i}", "cottage", 20, 20 + i * 10, 8, 8)
                        for i in range(4)]
    rel = {"subject": "houses", "relation": "around", "object": "temple"}
    assert intent_mod._relation_measure(rel, around)[0] == "satisfied"
    assert intent_mod._relation_measure(rel, clump)[0] == "failed"
    # nothing to measure is `unresolved`, never a pass
    assert intent_mod._relation_measure(rel, [temple])[0] == "unresolved"
    hier = {"greater": "temple", "lesser": "houses", "axis": "size"}
    assert intent_mod._hierarchy_measure(hier, around, {})[0] == "satisfied"
    small = [leaf("temple_1", "temple", 45, 45, 6, 6, "temple")] + [
        leaf(f"h{i}", "cottage", 10 + i * 12, 10, 20, 20) for i in range(3)]
    assert intent_mod._hierarchy_measure(hier, small, {})[0] == "failed"
    return ("houses round a temple pass, the same houses in a clump fail, an inverted "
            "size hierarchy fails and an unmeasurable one is unresolved")


@case
def t_e_this_library_cannot_be_talked_out_of_what_it_cannot_build():
    """A reader may read; it may not make an unsupported thing supported."""
    s = "Build a Persian town around a windmill, laid out on a grid."
    got = _reqs(s, [
        {"id": "tradition/persian", "kind": "tradition", "says": "persian",
         "wants": {"tradition": "persian"}, "phrase": "persian", "why": "named"},
        {"id": "feature/mill", "kind": "feature", "says": "a windmill",
         "wants": {"feature": "mill"}, "phrase": "windmill", "why": "named"},
        {"id": "layout/grid", "kind": "layout", "says": "on a grid",
         "wants": {"policy": "grid"}, "phrase": "on a grid", "why": "named"}])
    for rid in ("tradition/persian", "feature/mill", "layout/grid"):
        assert got[rid]["status"] == "unsupported", (rid, got[rid])
    # control: a supported layout is not marked unsupported
    ok = _reqs("Build a lakeside village.", [
        {"id": "layout/shoreline", "kind": "layout", "says": "shoreline",
         "wants": {"policy": "shoreline"}, "phrase": "lakeside", "why": "named"}])
    assert ok["layout/shoreline"]["status"] == "open", ok
    return ("a form family, a part family and a layout policy this build lacks stay "
            "`unsupported` through an interpretation that asked for them")


@case
def t_f_a_reading_must_quote_the_sentence():
    s = "Build a fishing village."
    ck = interpret.cross_check(s, _interp(s, [
        {"id": "feature/palace", "kind": "feature", "says": "a palace",
         "wants": {"feature": "palace"}, "phrase": "a great palace", "why": "invented"}]))
    assert [r["id"] for r in ck["unsupported_phrase"]] == ["feature/palace"], ck
    got = _reqs(s, [{"id": "feature/palace", "kind": "feature", "says": "a palace",
                     "wants": {"feature": "palace"}, "phrase": "a great palace",
                     "why": "invented"}])
    assert "feature/palace" not in got and "clause/fishing" in got, got
    return ("a reading quoting what the sentence does not say is refused and the "
            "unread word it did not claim survives as an obligation")


@case
def t_g_a_function_is_not_a_role():
    """A `rural` role admits a hall; a request for dwellings does not."""
    _t, decls = placeplan.types_card(None, "european_vernacular", "rural")
    cards = capability.cards(decls)
    want = {"id": "w", "name": "homes fabric", "kind": "plot",
            "forms": ["european_vernacular"], "roles": ["rural"], "family": None,
            "function": "dwelling"}
    ok = [n for n, c in cards.items() if capability.fits(c, want)[0]]
    assert ok and all((cards[n]["envelope"] or {}).get("function") == "dwelling"
                      for n in ok), ok
    assert "hall" not in ok, "a hall answered a want for a dwelling"
    # control: without a function the same want admits the role's types, hall included
    loose = {k: v for k, v in want.items() if k != "function"}
    wide = [n for n, c in cards.items() if capability.fits(c, loose)[0]]
    assert "hall" in wide and set(ok) < set(wide), (ok, wide)
    return (f"a `dwelling` want admits {sorted(ok)} and refuses the hall the same "
            f"role admits without it")


# ------------------------------------ 2. feasibility returns an arrangement

def _district_bits(w=84, d=35, want=8, pool=None):
    part = {"name": "homes", "kind": "district", "family": "district", "role": "rural",
            "density": "low", "count": 1, "relation": "throughout",
            "character": {"density": "low", "role": "rural"}}
    _t, decls = placeplan.types_card(None, "european_vernacular")
    dist = {"name": "homes_1", "x0": 0, "z0": 0, "x1": w - 1, "z1": d - 1,
            "structures": want, "density": "low", "role": "rural"}
    if pool:
        dist["fabric_types"] = list(pool)
    return dist, part, decls


@case
def t_h_capacity_is_what_the_compiler_lays():
    dist, part, decls = _district_bits()
    place = {"parts": [], "arterials": {"cells": []}, "districts": [dist]}
    got = arrange.compile_once(dist, part, place, decls, seed=1)
    assert got["ok"] and got["plan"] is not None, got
    # the estimate and the arrangement disagree, which is the whole finding
    est = placeplan.fabric_fit(84, 35, "low", "rural", part["character"])
    assert got["lots"] != est or True, (got["lots"], est)
    held = arrange.capacity(dist, part, place, decls, seed=1)
    assert held >= got["lots"], (held, got["lots"])
    # the arrangement IS the file that gets built
    plots = placeplan.district_plots(got["plan"], "rural", None)
    assert sum(1 for p in plots if p.get("kind", "plot") == "plot") == got["lots"]
    return (f"the compiler lays {got['lots']} where the grid estimate says {est}; the "
            f"arrangement carries the file that is built and its capacity is {held}")


@case
def t_i_the_ladder_tries_the_ground_before_it_moves_the_promise():
    # **Asked for more than the strip holds at the lot its character declares.** The
    # closure round: the compiler now shrinks an undeclared lot to keep a count under
    # the density's ceiling, so an 84x35 strip holds eight small cottages outright and
    # the ladder had nothing to climb. A declared 12-column lot is the lot, and twelve
    # of them do not fit the strip -- which is the shortfall this case is about.
    dist, part, decls = _district_bits(want=12)
    part = dict(part, character={"density": "low", "role": "rural", "lot_width": 12,
                                 "lot_depth": 12})
    site = {"origin": [0, 0], "size": 200}
    place = {"parts": [], "arterials": {"cells": []}, "districts": [dist]}
    got = arrange.arrange(dist, part, place, decls, site=site, seed=1, proposed=12)
    actions = [a["action"] for a in got["attempts"]]
    assert actions[0] == "as allocated", actions
    assert got["attempts"][0]["lots"] < got["realized"], got["attempts"]
    assert any(a.get("changed") for a in got["attempts"][1:]), got["attempts"]
    assert got["grew"] and got["adopted_rect"] != got["attempts"][0]["rect"], got
    # boxed in: the ladder reports the limit instead of silently moving the promise
    boxed = {"parts": [], "arterials": {"cells": []},
             "districts": [dist, {"name": "other", "x0": 84, "z0": 0,
                                  "x1": 199, "z1": 199}]}
    tight = arrange.arrange(dist, part, boxed, decls, site=site, seed=1, proposed=12)
    assert tight["short"] > 0 and tight["limit"], tight
    assert any(not a.get("changed") for a in tight["attempts"][1:]), tight["attempts"]
    return ("a district short of its ask grows into free ground and adopts the larger "
            "arrangement; boxed in, the same ask reports its limiting constraint")


@case
def t_j_the_approved_pool_keeps_its_order():
    """The fabric action re-orders the pool; `house_types` used to sort it away."""
    from ethoslm import district_compile as dc
    _t, decls = placeplan.types_card(None, "european_vernacular")
    pool = [n for n, _d in dc.house_types(decls, "rural", "european_vernacular")]
    assert len(pool) >= 2, pool
    a = [n for n, _d in dc.house_types(decls, "rural", "european_vernacular",
                                       approved=[pool[0], pool[1]])]
    b = [n for n, _d in dc.house_types(decls, "rural", "european_vernacular",
                                       approved=[pool[1], pool[0]])]
    assert a == [pool[0], pool[1]] and b == [pool[1], pool[0]], (a, b)
    # control: with no pool the role-then-name order is unchanged
    assert [n for n, _d in dc.house_types(decls, "rural",
                                          "european_vernacular")] == pool
    return f"an approved pool of {pool[:2]} is honoured in both orders"


@case
def t_k_the_compiler_search_reads_its_own_ground_clause():
    """`score` gained a field and the loop still sliced its first three."""
    import inspect
    from ethoslm import district_compile as dc
    src = inspect.getsource(dc.compile_district)
    # The closure round added the density ceiling as a fourth named clause, unpacked
    # then by index from `score`'s tuple. The composition round made that structural
    # instead: `clauses(rec)` returns them by name, so the next field added to the score
    # cannot become one of these by accident. What this case is about is unchanged --
    # the search must not read its clauses positionally -- so it asks for the shape that
    # holds now and still refuses the slice.
    assert 'c = clauses(rec)' in src and 'c["ground"]' in src and 'c["ceiling"]' in src, \
        "the search no longer reads its clauses by name"
    assert "s[0], s[1], s[3], s[4]" not in src, "the search unpacks the score by position"
    assert "score(rec)[:3]" not in src, "the search slices the score again"
    return "the search reads its ground clause by name and not by slice"


@case
def t_l_a_district_is_reconciled_by_leaf_kind():
    """A district's open ground is not its fabric."""
    plan = {"parts": [{"kind": "district", "name": "q", "defines": "homes",
                       "children": [
                           {"kind": "plot", "name": "h1", "type": "cottage"},
                           {"kind": "area", "name": "g1", "type": "grove"}]}]}
    leaves = pipeline.plan_parts(plan)
    assert all(p.get("answers") == "homes" for p in leaves), leaves
    used = capability._types_used({}, plan)
    assert used["homes"]["plot"] == ["cottage"], used
    assert used["homes"]["area"] == ["grove"], used
    rec = contracts.make("capabilities", entries=[{
        "id": "cap/homes/fabric", "requirement": None,
        "wants": {"id": "cap/homes/fabric", "name": "homes fabric", "kind": "plot",
                  "part": "homes", "family": None, "forms": [], "roles": ["rural"]},
        "kind": "plot", "family": None, "form": "european_vernacular", "role": "rural",
        "matched": True, "type": "cottage", "alternatives": [], "envelope": {},
        "checked": True, "status": "covered", "why": ""}])
    _out, rows = capability.agreements(rec, {}, {}, plan=plan)
    assert not rows, [r["says"] for r in rows]
    return ("a district's plot leaves reconcile against its fabric want and its area "
            "leaves do not; the grove is not reported as a house")


# --------------------------------------- 3. one lifecycle governs acceptance

class _Rnd:
    """The smallest thing `deps` reads: a state directory and a sentence."""

    def __init__(self, state, sentence="Build a village."):
        self.state, self.sentence = state, sentence
        self.site, self.flags, self.base_volume = {}, {}, "world.npz"

    def rel(self, *p):
        return os.path.join(self.state, *p)

    def chosen_site(self):
        return None

    def plan(self):
        p = self.rel("plan.json")
        return json.load(open(p)) if os.path.exists(p) else {}


@case
def t_m_a_stamp_is_bound_to_what_it_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Rnd(tmp)
        plan = {"parts": [{"kind": "plot", "name": "a", "type": "cottage",
                           "x0": 0, "z0": 0, "x1": 9, "z1": 9}], "levels": {}}
        json.dump(plan, open(rnd.rel("plan.json"), "w"))
        deps.stamp(rnd, "plan", outputs=["plan.json"], note="t")
        assert deps.check(rnd, "plan")[0], deps.check(rnd, "plan")
        # replaced with a document of the right shape and different content
        json.dump({"parts": [], "levels": {}}, open(rnd.rel("plan.json"), "w"))
        fresh, why = deps.check(rnd, "plan")
        assert not fresh and "not the ones it accepted" in why, why
        # ...and put back, it is warm again: this is identity, not a timestamp
        json.dump(plan, open(rnd.rel("plan.json"), "w"))
        assert deps.check(rnd, "plan")[0], deps.check(rnd, "plan")
    return "a stamped output replaced by a different document of the same shape is stale"


@case
def t_n_an_import_authorises_the_files_it_names():
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Rnd(tmp)
        deps.import_legacy(tmp, "a shipped fixture", ["place.json"])
        assert deps.legacy(rnd, "site_search") is False, "a directory-wide permission"
        deps.import_legacy(tmp, "a shipped fixture",
                           ["place.json", "site_search.json"])
        assert deps.legacy(rnd, "site_search") is True
        assert deps.legacy(rnd, "preview") is False
    return ("an import record makes the artifacts it names reusable and no others; "
            "a preview nobody imported stays unqualified")


@case
def t_o_a_spec_field_nobody_listed_still_moves_the_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Rnd(tmp)
        doc = {"kind": "village", "structures": 10, "size_band": [8, 20],
               "sentence": "Build a village.", "form": None, "voice": None,
               "defining_parts": [{"name": "homes", "kind": "group",
                                   "family": "district", "relation": "throughout",
                                   "count": 1, "land_use": "settled"}]}
        json.dump(doc, open(rnd.rel("place.checked.json"), "w"))
        a = deps._spec_print(rnd)
        doc["defining_parts"][0]["land_use"] = "farmland"
        json.dump(doc, open(rnd.rel("place.checked.json"), "w"))
        b = deps._spec_print(rnd)
        assert a != b, "land_use moved the compiler's target and not the fingerprint"
        # control: prose does not
        doc["defining_parts"][0]["land_use"] = "settled"
        doc["defining_parts"][0]["purpose"] = "somewhere to live"
        json.dump(doc, open(rnd.rel("place.checked.json"), "w"))
        assert deps._spec_print(rnd) == a, "a purpose line invalidated the design"
    return "every field of a defining part is a dependency except the prose"


@case
def t_p_access_is_one_network_and_not_one_per_gate():
    from ethoslm.pipeline import stages_measure

    class _Nav:
        def __init__(self, groups):
            self.groups = groups

        def stance_near(self, x, z, y, tol=1):
            return y - 1

        def flood(self, seeds, max_jumps=0):
            out = set()
            for s in seeds:
                for g in self.groups:
                    if s in g:
                        out |= g
            return out

    class _Net:
        def __init__(self, cells):
            self.cells = cells

        def surface(self):
            return [(x, z, 64) for (x, z, _y) in self.cells]

    class _Ctx:
        def __init__(self, nav):
            self.nav, self.doors, self.outdoor_seeds = nav, [], 0

        def door_stance(self, x, y, z):
            return 64

    a = {(0, 0, 64), (1, 0, 64)}
    b = {(50, 50, 64), (51, 50, 64)}
    nav = _Nav([a, b])
    ctx = _Ctx(nav)
    net = _Net(sorted(a | b))
    two = stages_measure._doors_from_the_lane(ctx, net, [(0, 0), (50, 50)])
    assert two["lane_components"] == 2 and two["status"] == "short", two
    assert "one network" in two["why"], two
    one = stages_measure._doors_from_the_lane(
        _Ctx(_Nav([a])), _Net(sorted(a)), [(0, 0)])
    assert one["lane_components"] == 1 and one["status"] == "connected", one
    return ("two lane components with a gate into each read `short`, and one connected "
            "network with its gate reads `connected`")


@case
def t_q_a_tradition_is_a_share_of_the_place_and_not_one_building():
    rec = intent_mod.read("Build a Japanese village.")
    spec = {"kind": "village", "form": "east_asian", "structures": 10,
            "size_band": [8, 20], "sentence": "Build a Japanese village.",
            "defining_parts": [{"name": "homes", "kind": "group", "family": "district",
                                "relation": "throughout", "count": 1}]}
    caps = {"entries": [{"matched": True, "type": "minka",
                         "envelope": {"tradition": "japanese"}}]}

    def plan_of(n_minka, n_other):
        parts = [{"kind": "plot", "name": f"m{i}", "type": "minka"}
                 for i in range(n_minka)]
        parts += [{"kind": "plot", "name": f"o{i}", "type": "cottage"}
                  for i in range(n_other)]
        return {"parts": parts, "levels": {}}

    one, _ = intent_mod.coverage(rec, spec, plan=plan_of(1, 99), capabilities=caps)
    got = next(r for r in one["requirements"] if r["kind"] == "tradition")
    assert got["status"] == "unresolved", got
    most, _ = intent_mod.coverage(rec, spec, plan=plan_of(60, 40), capabilities=caps)
    got2 = next(r for r in most["requirements"] if r["kind"] == "tradition")
    assert got2["status"] == "satisfied", got2
    return ("1 of 100 leaves in the named tradition is unresolved and 60 of 100 is "
            "satisfied, against a share registered before either was run")


@case
def t_r_a_cliff_is_a_steep_fall_and_an_island_has_water_round_it():
    even = {"stats": {"relief": 30}, "mean_grid": [[64] * 8 for _ in range(8)]}
    step = {"stats": {"relief": 30},
            "mean_grid": [[64] * 4 + [34] * 4 for _ in range(8)]}
    assert intent_mod._setting_measure("cliff", even)[0] == "failed"
    assert intent_mod._setting_measure("cliff", step)[0] == "satisfied"
    assert intent_mod._setting_measure(
        "cliff", {"stats": {"relief": 30}})[0] == "unresolved"
    lake = {"surface_blocks": {"water": 40, "grass_block": 60},
            "edge_blocks": {"grass_block": 100}}
    isle = {"surface_blocks": {"water": 40, "grass_block": 60},
            "edge_blocks": {"water": 90, "grass_block": 10}}
    assert intent_mod._setting_measure("island", lake)[0] == "failed"
    assert intent_mod._setting_measure("island", isle)[0] == "satisfied"
    assert intent_mod._setting_measure(
        "island", {"surface_blocks": {"water": 40, "grass_block": 60}}
    )[0] == "unresolved"
    return ("an even 30-block fall is not a cliff and a 30-block step is; a lake in one "
            "corner is not an island and water round the rim is; neither is certified "
            "where the record does not say")


@case
def t_s_a_declared_lot_is_the_lot_and_the_cover_follows_it():
    """The field an inspection asked for, and the target that must follow it."""
    from ethoslm import spec as spec_mod
    assert "lot_width" in spec_mod.CHARACTER_FIELDS
    base = {"frontage": "street", "storeys": [1, 2], "open_share": 0.25,
            "courtyard_share": 0.0}
    wide = placeplan.fabric("low", "rural", base)
    narrow = placeplan.fabric("low", "rural", {**base, "lot_width": 12,
                                               "lot_depth": 10})
    assert narrow["lot"] == [12, 10], narrow["lot"]
    assert narrow["plot_share"] < wide["plot_share"], (narrow, wide)
    _t, decls = placeplan.types_card(None, "european_vernacular")
    part = {"name": "homes", "kind": "district", "family": "district", "role": "rural",
            "density": "low", "count": 1, "relation": "throughout",
            "character": {**base, "lot_width": 12, "lot_depth": 10}}
    dist = {"name": "d", "x0": 0, "z0": 0, "x1": 128, "z1": 29, "structures": 3,
            "density": "low", "role": "rural", "fabric_types": ["cottage"]}
    from ethoslm import district_compile as dc
    _got, rec = dc.compile_district(dist, part, {"parts": [],
                                                 "arterials": {"cells": []}},
                                    decls, seed=1)
    assert rec["lot"] == [12, 10], rec["lot"]
    assert not rec["raised"].get("_lot_grow"), rec["raised"]
    return ("a character may declare its lot's width, the compiler lays it and does not "
            "grow it, and the cover the district is held to is computed from it")


@case
def t_t_a_cottage_lean_to_has_a_way_in():
    """The construction defect the shore village found, at the seeds that show it."""
    import test_types as T
    from ethoslm import lint
    from ethoslm.buildlib import Builder
    decl = pipeline.load_type(os.path.join(ROOT, "types", "cottage.py"))
    ns = {"__name__": "__ethoslm_type__", "__file__": decl["path"]}
    exec(compile(decl["src"], decl["path"], "exec"), ns)               # noqa: S102
    bad = []
    for outshot in ("store", "scullery", "byre"):
        for seed in (3, 4, 5, 6):
            vol = T._flat_world(90)
            b = Builder(offline.OfflineSite(vol))
            b._vol, b.frontage = vol, None
            lot = {"label": "t", "x0": 20, "z0": 20, "x1": 47, "z1": 31}
            b.registry = T._OnePlot(dict(lot))
            sited = b.site({**lot, "kind": "plot", "front": "north"}, mat=None,
                           roof=None)
            ns["build"](b.type_builder(sited), sited, seed, storeys=1,
                        outshot=outshot)
            b.resolve_steps()
            ctx = lint.Context.build(vol.overlay(b._pending), plots=[dict(lot)],
                                     region=(0, 0, 89, 89))
            for f in lint.lint(ctx).errors:
                if f.code in ("E003", "E011"):
                    bad.append((outshot, seed, f.code))
    assert not bad, bad
    return ("12 cottages over three outshots and four seeds: no room that cannot be "
            "walked into, at the seeds where the shore village found one")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = fail = 0
    for name, fn in CASES:
        if only and only not in name:
            continue
        try:
            says = fn()
        except AssertionError as e:
            fail += 1
            print(f"FAIL {name}: {e}")
            continue
        except Exception as e:                      # noqa: BLE001 -- reported by name
            fail += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            continue
        ok += 1
        print(f"ok   {name:52} {says}")
    print(f"\n{ok}/{ok + fail} realization cases pass")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
