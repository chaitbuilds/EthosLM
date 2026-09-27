#!/usr/bin/env python3
"""Planning by strategy: repeated fabric, individual composition, or both, one contract.

    $PY scripts/test_composition_routing.py

Every case builds its own input (a spec, a design, flat or sloped synthetic ground) and
reads no run's state, so it runs the same in the public tree:

1. the planning strategy follows the spec's own statement of how each part is made and
   what the library can express: a village planned part by part is composed, a small
   planned settlement whose cottages have a character is fabric, a city of character
   rings with a hall of its own is mixed -- the size of the place decides none of them;
2. the choice is recorded before the design, kept while its inputs are unchanged, and
   when it changes the design made under the old one is retired, not compiled;
3. a composition compiles into the shared contract: leaves with their use, form and
   parameters, doors carried to a path or a walked space, a green addressed by the
   houses that face it, the unplaced land left as found and owned by a region;
4. what a composition asks that cannot be built comes back named: a form of another use,
   a parameter out of the form's range, a lot too small for its form, two things on
   one piece of ground, a door that reaches nothing, a field on ground too steep for it;
5. a composed ward stands inside a fabric ring: the fabric packs its own land and never
   the ward's, and both are leaves of one resolution with one region tiling;
6. a design must follow the recorded strategy, and a programme must answer the request;
7. the green is a form of its own: turf, a heart, trees and benches, built clean;
8. the site search holds a relief word: hill country is offered sloping ground;
9. a parameter the design chose stands in the blocks or is handed back with a lot that
   delivers it: a byre outshot asked of a cottage on a lot too small for one;
10. a revision answered at resolution survives the resume that re-runs the comparison;
    a revision that breaks the design stays handed back until it is answered, and an
    unchanged resolution is kept rather than recompiled.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from ethoslm import atlas, citydesign as CD, cityresolve as C, planning  # noqa: E402

RESULTS = []


def case(fn):
    def run():
        try:
            why = fn()
            RESULTS.append((fn.__name__, True, why or ""))
        except AssertionError as e:
            RESULTS.append((fn.__name__, False, str(e)))
        except Exception as e:                     # noqa: BLE001 -- reported
            RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {e}"))
    run.__name__ = fn.__name__
    return run


def _part(name, kind, family, relation="throughout", structures=0, character=None, **kw):
    p = {"name": name, "kind": kind, "family": family, "relation": relation,
         "count": 1, "structures": structures, **kw}
    if character:
        p["character"] = character
    return p


VILLAGE_SPEC = {"kind": "village", "defining_parts": [
    _part("green", "area", "square", "centre"),
    _part("homes", "group", "district", structures=12),
    _part("smithy", "plot", "workshop", "near", of="green"),
    _part("hall", "plot", "hall", "near", of="green")]}
PLANNED_SPEC = {"kind": "village", "defining_parts": [
    _part("terraces", "group", "district", structures=30,
          character={"frontage": "street", "attached": True, "block": 60})]}
CITY_SPEC = {"kind": "city", "defining_parts": [
    _part("palace", "group", "palace", "centre"),
    _part("inner", "group", "district", "concentric", 300, {"frontage": "street"}),
    _part("outer", "group", "district", "concentric", 600, {"frontage": "street"}),
    _part("guild_hall", "plot", "hall", "near", of="inner"),
    _part("wall", "edge", "wall", "concentric")]}
CAPITAL_SPEC = {"kind": "city", "defining_parts": [
    p for p in CITY_SPEC["defining_parts"] if p["name"] != "guild_hall"]}


@case
def c1_the_strategy_is_the_specs_not_the_size():
    got = {n: planning.choose(s, ROOT) for n, s in (
        ("village", VILLAGE_SPEC), ("planned", PLANNED_SPEC), ("city", CITY_SPEC),
        ("capital", CAPITAL_SPEC))}
    assert got["village"]["strategy"] == "composed", got["village"]
    assert got["planned"]["strategy"] == "fabric", \
        "a small settlement of like lots in a character is fabric: " + got["planned"]["why"]
    assert got["city"]["strategy"] == "mixed", got["city"]
    assert got["capital"]["strategy"] == "fabric", got["capital"]
    smithy = next(r for r in got["village"]["parts"] if r["part"] == "smithy")
    assert "workshop" in smithy["forms"] and not got["village"]["unsupported"], smithy
    # a use the library has no form for is said, not swapped
    odd = {"kind": "village", "defining_parts": [_part("mill", "plot", "tower")]}
    rec = planning.choose(odd, ROOT)
    assert rec["unsupported"] and rec["unsupported"][0]["part"] == "mill", rec
    return ", ".join(f"{k} {v['strategy']}" for k, v in got.items())


class _Round:
    """The part of a round the planning and design stages read."""

    def __init__(self, tmp, spec):
        self.state = tmp
        self.flags = {"design": {}}
        self.sentence = "a test"
        self.spec = spec

    def rel(self, *p):
        return os.path.join(self.state, *p)

    def place_spec(self):
        return self.spec


@case
def c2_the_choice_is_kept_and_its_design_retired_when_it_changes():
    from ethoslm.pipeline import stages_design as SD
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Round(tmp, copy.deepcopy(VILLAGE_SPEC))
        a = SD.stage_planning(rnd, None, {})
        assert a["strategy"] == "composed" and os.path.exists(rnd.rel("planning.json"))
        for f in ("design.proposals.json", "design.adopt.json", "design.json"):
            json.dump({"id": "x"}, open(rnd.rel(f), "w"))
        b = SD.stage_planning(rnd, None, {})
        assert b.get("kept") and os.path.exists(rnd.rel("design.json")), b
        # a changed note is not a changed input
        rnd.spec["defining_parts"][0]["notes"] = "the green, now described"
        assert SD.stage_planning(rnd, None, {}).get("kept")
        # a changed input that leaves the choice where it was keeps the design
        rnd.spec["defining_parts"].append(_part("well", "point", "gate"))
        c = SD.stage_planning(rnd, None, {})
        assert not c.get("kept") and not c["retired"] and \
            os.path.exists(rnd.rel("design.json")), c
        # the homes given a character, the smithy and hall gone: fabric, the design
        # retired
        rnd.spec = copy.deepcopy(PLANNED_SPEC)
        d = SD.stage_planning(rnd, None, {})
        assert d["strategy"] == "fabric" and sorted(d["retired"]) == sorted(
            ["design.proposals.json", "design.adopt.json", "design.json"]), d
        assert not os.path.exists(rnd.rel("design.json"))
        rec = json.load(open(rnd.rel("planning.json")))
        assert rec["replaces"]["strategy"] == "composed", rec
        # a design adopted under the old choice is refused at resolution
        json.dump({"id": "old", "planning": {"strategy": "composed", "digest": "stale"}},
                  open(rnd.rel("design.json"), "w"))
        r = SD.stage_design_resolve(rnd, None, {})
        assert r.get("stop") and "planned" in r["error"], r
    return "kept on resume and on a changed note; retired when the choice changed"


def _ground(W, slope=0.0, axis=0):
    i = np.arange(W, dtype=np.float64)
    g = 64 + slope * (i[:, None] if axis == 0 else i[None, :]) + 0 * i[None, :]
    g = np.broadcast_to(g, (W, W)).astype(np.int16).copy()
    return g, g.copy(), np.zeros((W, W), bool)


def _compile(design, slope=0.0):
    d = CD.read(copy.deepcopy(design))
    W = 2 * (d["extent"]["radius"] + C.MARGIN) + 1
    return C.compile_design(d, (0, 0), ground=_ground(W, slope))


COMPOSED = {
    "schema": "ethoslm.design/1", "id": "probe-composed", "extent": {"radius": 60},
    "site": {"candidate": "given", "centre": [0, 0]},
    "rings": [{"name": "village", "outer": 1.0, "grain": "composed",
               "ground": {"policy": "preserve"}}],
    "palette": {"rings": {"village": "drystone_and_thatch"}},
    "composition": {
        "programme": [
            {"use": "dwelling", "what": "households", "count": 3, "source": "request",
             "reads": ["homes"]},
            {"use": "work", "what": "a smithy by the road", "count": 1,
             "source": "inferred"},
            {"use": "civic", "what": "the moot hall on the green", "count": 1,
             "source": "inferred"}],
        "spaces": [{"id": "green", "kind": "green", "at": [0, 0], "size": [24, 20],
                    "params": {"centre": "well"}},
                   {"id": "barley", "kind": "field", "at": [-4, 42], "size": [30, 12]}],
        "groups": [{"id": "green_row", "purpose": "homes facing the green",
                    "around": "green"}],
        "buildings": [
            {"id": "h1", "use": "dwelling", "form": "cottage", "at": [-6, -19],
             "size": [9, 9], "faces": "green", "group": "green_row",
             "params": {"storeys": 1, "outshot": "byre"}},
            {"id": "h2", "use": "dwelling", "form": "cottage", "at": [6, -19],
             "size": [9, 9], "faces": "green", "group": "green_row"},
            {"id": "h3", "use": "dwelling", "form": "cottage", "at": [-20, 0],
             "size": [9, 11], "faces": "green", "group": "green_row"},
            {"id": "hall", "use": "civic", "form": "hall", "at": [0, 18],
             "size": [13, 9], "faces": "green", "params": {"use": "moot"}},
            {"id": "smithy", "use": "work", "form": "workshop", "at": [34, -14],
             "size": [11, 11], "faces": "street", "params": {"trade": "smithy"}}],
        "paths": [
            {"id": "street", "rank": "lane", "width": 3, "points": [[27, -62], [27, 62]]},
            {"id": "walk", "rank": "path", "width": 2,
             "points": [[-13, -12], [13, -12], [13, 12], [-13, 12], [-13, -12]]},
            {"id": "link", "rank": "path", "width": 2, "points": [[14, 0], [27, 0]]}]}}


def _blocking(city):
    return [f for f in city.findings if f["severity"] == "blocking"]


@case
def c3_a_composition_resolves_into_the_shared_contract():
    city = _compile(COMPOSED)
    assert not _blocking(city), _blocking(city)
    json.dumps(C.plan_tree(city))           # the construction inputs as written to disk
    json.dumps(city.composition)
    plots = {lf["element"]: lf for lf in city.leaves if lf.get("composed") and
             lf["kind"] == "plot"}
    assert sorted(plots) == ["h1", "h2", "h3", "hall", "smithy"], sorted(plots)
    assert plots["smithy"]["type"] == "workshop" and \
        plots["smithy"]["params"]["trade"] == "smithy" and plots["smithy"]["use"] == "work"
    assert plots["h1"]["params"]["outshot"] == "byre" and plots["h1"]["chosen"] == [
        "outshot", "storeys"]
    assert plots["hall"]["front"] == "north" and plots["h1"]["front"] == "south", \
        (plots["hall"]["front"], plots["h1"]["front"])
    assert all(lf.get("site") and lf.get("region") for lf in plots.values()), \
        "every composed building has a compiled door and a region"
    rep = city.composition
    green = next(s for s in rep["spaces"] if s["id"] == "green")
    assert set(green["addressed_by"]) >= {"h1", "h2", "h3", "hall"}, green
    assert rep["access"]["doors"] == 5 and rep["access"]["reach_boundary"] == 5 and \
        rep["access"]["networks"] == 1, rep["access"]
    assert rep["uses"] == {"dwelling": 3, "work": 1, "civic": 1}, rep["uses"]
    # the land nobody placed anything on is kept as found and owned, never packed
    land = city.comp["land"]
    kept = land & (city.use == C.OPEN) & (city.treat == C.KEEP)
    assert kept.sum() > 0.5 * land.sum(), "the unplaced land is not kept as found"
    assert not [lf for lf in city.leaves if not lf.get("composed") and
                lf["kind"] in ("plot", "area")], "something was packed into composed land"
    assert (city.owner[land] >= 0).all(), "composed land left unowned"
    return (f"{len(plots)} buildings of {len(rep['forms'])} forms, green addressed by "
            f"{len(green['addressed_by'])}, one network to the boundary")


def _finding(design, slope=0.0):
    try:
        city = _compile(design, slope)
    except CD.DesignError as e:
        return f"refused: {e}"
    return " | ".join(f["what"] for f in _blocking(city)) or None


@case
def c4_what_cannot_be_built_is_named():
    got = []
    cases = [
        (("buildings", 4, "form"), "cottage", "is a dwelling form"),
        (("buildings", 4, "params"), {"trade": "bakery"}, "trade one of"),
        (("buildings", 0, "size"), [4, 4], "needs at least"),
        # held at the pad construction builds on (the lot less one column each side):
        # 9x14 is a 7x12 pad, which the cottage's own sweep found broken
        (("buildings", 2, "size"), [9, 14], "measured broken"),
        (("buildings", 1, "at"), [0, 0], "stands on"),
        (("buildings", 4, "faces"), "east", "meets no path"),
    ]
    for (k, i, f), v, want in cases:
        d = copy.deepcopy(COMPOSED)
        d["composition"][k][i][f] = v
        why = _finding(d)
        assert why and want in why, f"{k}[{i}].{f}={v!r}: {why}"
        got.append(want)
    # a field on ground too steep for it: returned, not laid as something else
    d = copy.deepcopy(COMPOSED)
    d["composition"]["paths"] = [p for p in d["composition"]["paths"]]
    why = _finding(d, slope=1.2)
    assert why and "barley" in why and "field" in why, why
    got.append("steep field")
    return "; ".join(got)


FABRIC_WITH_WARD = {
    "schema": "ethoslm.design/1", "id": "probe-mixed", "extent": {"radius": 110},
    "site": {"candidate": "given", "centre": [0, 0]},
    "rings": [{"name": "town", "outer": 1.0, "grain": "rows",
               "params": {"lane_dir": "east_west", "street_every": 8},
               "wards": [{"from": 60, "to": 120, "grain": "composed",
                          "why": "the guild quarter, composed"}],
               "ground": {"policy": "terrace"}}],
    "composition": {
        "spaces": [{"id": "yard", "kind": "square", "at": [70, 0], "size": [16, 16]}],
        "buildings": [{"id": "guild", "use": "civic", "form": "hall", "at": [70, -16],
                       "size": [13, 9], "faces": "yard", "params": {"use": "market"}},
                      {"id": "forge", "use": "work", "form": "workshop", "at": [70, 16],
                       "size": [11, 11], "faces": "yard"}],
        "paths": [{"id": "in", "rank": "lane", "width": 3, "points": [[40, 0], [62, 0]]}]}}


@case
def c5_a_composed_ward_in_a_fabric_ring():
    city = _compile(FABRIC_WITH_WARD)
    assert not _blocking(city), _blocking(city)
    land = city.comp["land"]
    comp = [lf for lf in city.leaves if lf.get("composed")]
    fabric = [lf for lf in city.leaves if lf["kind"] == "plot" and not lf.get("composed")]
    assert {lf["element"] for lf in comp if lf["kind"] == "plot"} == {"guild", "forge"}
    assert len(fabric) > 50, f"the fabric packed only {len(fabric)} lots"
    for lf in fabric:
        sl = (slice(lf["x0"] - city.X0, lf["x1"] - city.X0 + 1),
              slice(lf["z0"] - city.Z0, lf["z1"] - city.Z0 + 1))
        assert not land[sl].any(), f"fabric lot {lf['name']} stands in the composed ward"
    rids = {lf.get("region") for lf in city.leaves if lf["kind"] != "edge"}
    assert None not in rids and set(city.regions) >= rids
    assert city.composition["access"]["on_network"] == 2
    return (f"{len(fabric)} fabric lots round a ward of {len(comp)} composed leaves, "
            f"{len(city.regions)} regions")


@case
def c6_a_design_follows_its_strategy_and_answers_the_request():
    assert planning.expects("composed", COMPOSED) is None
    clusters = {"rings": [{"name": "v", "outer": 1.0, "grain": "clusters"}]}
    assert "composed" in planning.expects("composed", clusters)
    assert "no `composed`" in planning.expects("fabric", COMPOSED)
    assert planning.expects("mixed", FABRIC_WITH_WARD) is None
    assert planning.expects("mixed", COMPOSED)
    from ethoslm.pipeline import stages_design as SD
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Round(tmp, VILLAGE_SPEC)
        json.dump({"reads": [
            {"id": "homes", "kind": "function", "hard": True, "says": "homes",
             "wants": {"function": "dwelling"}},
            {"id": "work", "kind": "function", "hard": True, "says": "places to work",
             "wants": {"function": "work"}},
            {"id": "green", "kind": "feature", "hard": True, "says": "a shared green",
             "wants": {"feature": "green"}}]}, open(rnd.rel("interpretation.json"), "w"))
        errs = SD.check_programme(rnd, COMPOSED)
        assert len(errs) == 1 and "`work`" in errs[0], errs
        d = copy.deepcopy(COMPOSED)
        d["composition"]["programme"][1]["reads"] = ["work"]
        assert SD.check_programme(rnd, d) == []
    return "strategies held; an uncited request read is named"


@case
def c7_the_green_is_a_form_of_its_own():
    from ethoslm import construction, lint
    # construction preflights a type's source before it builds anything (no `try`, no
    # forbidden calls): a form that fails it is refused in a real region, whatever a
    # probe on a flat lot says
    from ethoslm import pipeline
    src = open(os.path.join(ROOT, "types", "green.py")).read()
    for pre in (lint.preflight(src, forbid=pipeline.TYPE_FORBIDDEN),
                lint.preflight(src, allow_try=True, palette=True)):
        assert pre.ok, pre.to_json()
    rows = []
    for w, d, prm in ((24, 20, {"centre": "well", "trees": 3, "benches": 2}),
                      (14, 12, {"centre": "tree", "trees": 1, "benches": 4}),
                      (7, 7, {"centre": "cross", "trees": 0})):
        b, sited, res = construction.probe_build("green", w, d, prm, seed=3,
                                                 voice="dark_timber_and_tile",
                                                 front="south")
        assert res.get("ok"), f"{w}x{d}: {res}"
        assert res["centre"], res
        rows.append(f"{w}x{d}:{res['centre'][0]}/{res['trees']}t/{res['benches']}b")
    return ", ".join(rows)


@case
def c8_hill_country_is_offered_sloping_ground():
    n = 64
    med = np.full((n, n), 64, np.int16)
    # the east half is a steady hillside rising one block in five
    for i in range(n // 2, n):
        med[i, :] = 64 + int((i - n // 2) * 8 / 5)
    co = {"step": 8, "x0": 0, "z0": 0, "median": med, "low": med.copy(),
          "high": med.copy() + 1, "wet": np.zeros((n, n), np.float32),
          "biome": np.full((n, n), 255, np.uint8), "biomes": False}
    flat = atlas.scan_sites(64, co=co, stride=32, water=(0, 0.2), top=3, relief="flat")
    hill = atlas.scan_sites(64, co=co, stride=32, water=(0, 0.2), top=3, relief="rolling")
    assert flat[0]["fall"] <= 2, flat[0]
    assert hill[0]["fall"] >= 16 and hill[0]["centre"][0] > 8 * n // 2, hill[0]
    return f"flat first fall {flat[0]['fall']}, rolling first fall {hill[0]['fall']}"


@case
def c9_a_chosen_parameter_stands_in_the_blocks_or_is_handed_back():
    from ethoslm.pipeline import stages_design as SD
    d = copy.deepcopy(COMPOSED)
    d["composition"]["buildings"] = [b for b in d["composition"]["buildings"]
                                     if b["id"] in ("h1", "smithy")]
    d["composition"]["programme"] = []
    city = _compile(d)
    with tempfile.TemporaryDirectory() as tmp:
        got = SD.realization(_Round(tmp, VILLAGE_SPEC), city)
        # a cottage's byre outshot does not stand on a 9x9 lot at this seed; the finding
        # names the building's size field and a lot on which the probe delivers it
        h1 = [f for f in got if f["element"] == "h1"]
        assert h1 and h1[0]["severity"] == "blocking" and "outshot" in h1[0]["missed"] \
            and h1[0]["field"].endswith(".size") and h1[0]["delivers_at"], got
        assert not [f for f in got if f["element"] == "smithy"], got
        w, dd = h1[0]["delivers_at"]
        d["composition"]["buildings"][0]["size"] = [w, dd]
        d["composition"]["buildings"][0]["at"] = h1[0]["at"]
        again = SD.realization(_Round(tmp, VILLAGE_SPEC), _compile(d))
        assert not [f for f in again if f["element"] == "h1"], again
        assert os.path.exists(os.path.join(tmp, "design", "probes.json"))
    return f"h1's byre outshot handed back at 9x9, delivered at {w}x{dd}"


@case
def c10_a_revision_survives_resume_and_nothing_stale_is_kept():
    from ethoslm.pipeline import stages_design as SD
    was = SD.site_ground
    SD.site_ground = lambda rnd, frame: _ground(frame[2])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rnd = _Round(tmp, copy.deepcopy(VILLAGE_SPEC))
            SD.stage_planning(rnd, None, {})
            d = copy.deepcopy(COMPOSED)
            d["site"] = {"candidate": "given", "centre": [0, 0]}
            d["composition"]["buildings"] = [b for b in d["composition"]["buildings"]
                                             if b["id"] != "h1"]
            d["composition"]["programme"] = []
            json.dump({"proposals": [d]}, open(rnd.rel("design.proposals.json"), "w"))
            json.dump({"adopt": d["id"], "set": {}, "why": "the one proposal"},
                      open(rnd.rel("design.adopt.json"), "w"))
            SD.stage_design_compare(rnd, None, {})
            a = SD.stage_design_resolve(rnd, None, {})
            assert not a.get("status") and a["design"] == d["id"], a
            # a revision that breaks the design is handed back, and stays handed back on
            # resume until it is answered
            json.dump({"set": {"composition.buildings[2].at": [0, 26]},
                       "why": "the hall further from the green"},
                      open(rnd.rel("design.revision.json"), "w"))
            bad = SD.stage_design_resolve(rnd, None, {})
            assert bad.get("status") == "needs_model" and "hall" in json.dumps(
                bad["findings"]), bad
            again = SD.stage_design_resolve(rnd, None, {})
            assert again.get("status") == "needs_model", again
            # answered: the hall one block back from the green, its door still on it
            json.dump({"set": {"composition.buildings[2].at": [0, 19]},
                       "why": "the hall one block back, its door still on the green"},
                      open(rnd.rel("design.revision.json"), "w"))
            b = SD.stage_design_resolve(rnd, None, {})
            assert b.get("design_digest") and b["design_digest"] != a["design_digest"], b
            # resume: the comparison runs again and re-adopts; the revision is replayed
            SD.stage_design_compare(rnd, None, {})
            got = json.load(open(rnd.rel("design.json")))
            assert got["composition"]["buildings"][2]["at"] == [0, 19] and \
                len(got["revisions"]) == 2, got.get("revisions")
            c = SD.stage_design_resolve(rnd, None, {})
            assert c.get("skipped") and c["design_digest"] == b["design_digest"], c
    finally:
        SD.site_ground = was
    return "revised, resumed, re-adopted with the revision replayed, resolution kept"


def main() -> int:
    for fn in (c1_the_strategy_is_the_specs_not_the_size,
               c2_the_choice_is_kept_and_its_design_retired_when_it_changes,
               c3_a_composition_resolves_into_the_shared_contract,
               c4_what_cannot_be_built_is_named, c5_a_composed_ward_in_a_fabric_ring,
               c6_a_design_follows_its_strategy_and_answers_the_request,
               c7_the_green_is_a_form_of_its_own, c8_hill_country_is_offered_sloping_ground,
               c9_a_chosen_parameter_stands_in_the_blocks_or_is_handed_back,
               c10_a_revision_survives_resume_and_nothing_stale_is_kept):
        fn()
    for name, ok, why in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {why}" if why else ""))
    n = sum(ok for _, ok, _ in RESULTS)
    print(f"{n}/{len(RESULTS)}")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
