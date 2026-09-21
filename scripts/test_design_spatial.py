#!/usr/bin/env python3
"""The design round's spatial escape routes, E2 and E3, through the real consumers.

Small inputs, each with a positive control, and every one of them run through the
production entry points -- `placeplan.concentric_layout`, `arrange.arrangements`,
`arrange.capacity_of` (which is `district_compile`), `intent.lot_cover`,
`placesolve.reallocate`, `ground.propose/evaluate/apply`. Nothing here reimplements a
measurement it is checking.

    E2  relabelling unchanged ground `open` cannot satisfy the same density
        requirement; a genuinely improved arrangement can move the figure; and
        allocated lot area alone cannot prove built massing.

    E3  an anchor or voice revision regenerates owned ground from the baseline and
        preserves population and access; emitted solids fit the approved occupation,
        including the great wall's projection.

Seconds, not minutes: the fixtures are the retained expression-round plans and the
compiler is asked about single rectangles.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import (arrange, ground, intent as intent_mod, placeplan,  # noqa: E402
                   placeregion, placesolve, spec as spec_mod)

CASES = []


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the fixture

_LOADED: dict = {}


def rings_fixture():
    """The expression round's hill town: spec, site, plateau, types, intent.

        Retained state, read and never written. `out/expr-rings` is immutable input.
        
    """
    if "rings" in _LOADED:
        return _LOADED["rings"]
    d = os.path.join(ROOT, "out", "expr-rings")
    if not os.path.exists(os.path.join(d, "place.json")):
        raise Skip("no out/expr-rings/place.json")
    doc = json.load(open(os.path.join(d, "place.json")))
    intent = json.load(open(os.path.join(d, "intent.json")))
    sentence = json.load(open(os.path.join(d, "interpretation.json")))["sentence"]
    count = next((dict(r.get("wants") or {}) for r in intent.get("requirements") or []
                  if r.get("kind") == "count"), None)
    spec = spec_mod.read_spec(doc, sentence, count)
    site = json.load(open(os.path.join(d, "site.json")))
    plateau = json.load(open(os.path.join(d, "plateau.json")))
    _t, decls = placeplan.types_card(None, spec.get("form"))
    got = (spec, site, plateau, decls, intent)
    _LOADED["rings"] = got
    return got


def _compile_ring(place, spec, decls, name="lower_ring"):
    """Compile every sector of a ring through `arrange` and return
    `(plots, regions)` in the shape `intent.lot_cover` is given in production."""
    by = {p["name"]: p for p in spec["defining_parts"]}
    part = by[name]
    role = spec_mod.district_role(spec, {"defines": name})
    _t, mine = placeplan.types_card(None, spec.get("form"), role)
    plots, regions = [], []
    for d in place["districts"]:
        if d.get("defines") != name:
            continue
        got = arrange.arrange(d, part, place, mine or decls, spec=spec, seed=1,
                              proposed=int(d.get("structures") or 0))
        leaves = ([p for p in placeplan.district_plots(got["plan"] or {},
                                                       part.get("role"), spec)
                   if p.get("kind", "plot") == "plot"] if got.get("plan") else [])
        plots += leaves
        regions.append({**placeplan.region_columns(d, place, decls,
                                                   record=got.get("record"),
                                                   leaves=leaves),
                        "name": d["name"], "rect": [d["x0"], d["z0"], d["x1"], d["z1"]],
                        "surface": str(d.get("surface") or "built"),
                        "lots": int(d.get("structures") or 0)})
    return plots, regions


# ---------------------------------------------- E2: the denominator and the massing

@case
def t_e2a_relabelling_ground_open_cannot_satisfy_the_density_requirement():
    """The expression round measured a `dense` ring at 27.5% over the part of it that
    happened to receive lots. Naming the rest `open` is an allocation decision."""
    spec, site, plateau, decls, _intent = rings_fixture()
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    plots, regions = _compile_ring(place, spec, decls)
    opened = [r for r in regions if r["surface"] == "open"]
    assert opened, "the ring was not compacted: nothing to relabel"
    # every remainder is **inferred** -- no requirement asked for it -- so it names the
    # region it was cut from and keeps its whole rectangle inside the scope
    for r in opened:
        assert r["open_requested"] is None, r
        assert r["scope_columns"] == r["rect_columns"], r
        assert r["scope_of"], r
    honest = intent_mod.lot_cover(plots, regions)
    # the escape route: relabel the same ground and measure again. Nothing about the
    # place changed, so nothing about the figure may change either.
    relabelled = [dict(r, surface="open") if r["lots"] == 0 else dict(r)
                  for r in regions]
    after = intent_mod.lot_cover(plots, relabelled)
    assert after["cover"] == honest["cover"], (honest, after)
    assert after["ground"] == honest["ground"], (honest, after)
    # the positive control: ground an **explicit requirement** asks to be open does
    # leave the scope, and says whose authority it left on
    asked = [dict(r, **placeregion.column_record(r["rect"], open_requested="req/park/1"))
             if r["lots"] == 0 else dict(r) for r in regions]
    control = intent_mod.lot_cover(plots, asked)
    assert control["ground"] < honest["ground"], (honest, control)
    assert control["cover"] > honest["cover"], (honest, control)
    return (f"the ring's {len(regions)} region(s) hold {honest['built']} columns of lot "
            f"over {honest['ground']} ({honest['cover']:.1%}, denominator "
            f"`{honest['denominator']}`); relabelling the {len(opened)} inferred "
            f"remainder(s) `open` leaves it at {after['cover']:.1%} and an explicit "
            f"open requirement moves it to {control['cover']:.1%} over "
            f"{control['ground']}")


@case
def t_e2b_a_better_arrangement_moves_the_figure_and_the_compiler_says_so():
    """A genuinely improved arrangement can. Every alternative is sized to the ring
    width it needs and laid by the actual district compiler."""
    spec, site, plateau, decls, _intent = rings_fixture()
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    lower = next(r for r in place["layout"]["rings"] if r["name"] == "lower_ring")
    considered = lower["target"].get("negotiated") or []
    assert considered, "the ring recorded no alternatives"
    real = [a for a in considered if a.get("capacity") is not None
            and not a.get("refused")]
    assert len(real) >= 2, considered
    # the exact count survives every alternative that was chosen between
    assert all(int(a["capacity"]) >= 15 for a in real), \
        [(a["action"], a["capacity"]) for a in real]
    declared = next(a for a in real if a["action"] == "as_declared")
    chosen = max(real, key=lambda a: a["cover"])
    assert chosen["cover"] > declared["cover"], (declared, chosen)
    assert lower["width"] == chosen["width"], (lower["width"], chosen)
    # ...and the refusals are refusals with a reason, not silence
    refused = [a for a in considered if a.get("refused")]
    assert any("bay" in str(a.get("refused")) for a in refused), refused
    return (f"{len(considered)} alternative(s), {len(real)} the compiler could lay: "
            + "; ".join(f"{a['action']} {a['lot'][0]}x{a['lot'][1]}x{a['rows']}row "
                        f"ring {a['width']} -> {a['capacity']}/15 at "
                        f"{a['cover']:.1%}" for a in real)
            + f"; chosen {chosen['action']} against the declared "
              f"{declared['cover']:.1%}")


@case
def t_e2c_allocated_area_alone_cannot_prove_built_massing():
    """Larger lots are not denser construction, and the two are two fields.

        `plot_cover` has always been the **lots'** share of a district's rectangle. On one
        rectangle, with one type and one ask, declaring a bigger lot raises that figure
        three-fold and leaves the district holding *fewer houses*: the number a density
        clause reads went up while the place emptied. So `allocated_columns` and
        `built_columns` are reported apart and `intent.lot_cover` carries both.
        
    """
    spec, _site, _plateau, decls, _intent = rings_fixture()
    by = {p["name"]: p for p in spec["defining_parts"]}
    part = dict(by["lower_ring"], fabric_types=["court_small"])
    _t, mine = placeplan.types_card(None, spec.get("form"), part.get("role"))
    if "court_small" not in (mine or {}):
        raise Skip("no court_small type")
    bare = {"arterials": {}, "parts": [], "districts": [], "layout": {}}
    d = {"name": "probe", "x0": 0, "z0": 0, "x1": 119, "z1": 33, "structures": 6,
         "exact": True, "defines": "lower_ring", "fabric_types": ["court_small"]}

    def lay(width, ask=6):
        return arrange.capacity_of(dict(d, structures=int(ask)), part, bare, mine,
                                   {"rows": 1, "attached": False, "lot_width": width,
                                    "lot_depth": width, "frontage": "street",
                                    "courtyard_share": 0.0, "open_share": 0.0},
                                   spec=spec, ceiling=int(ask))
    small, big = lay(10), lay(16)
    assert small["ok"] and big["ok"], (small["why"], big["why"])
    # the escape route, measured: the lots cover three times the ground and the district
    # holds fewer houses than it did
    assert big["allocated_columns"] > 2 * small["allocated_columns"], (small, big)
    assert big["lots"] < small["lots"], (small["lots"], big["lots"])
    # ...and the two columns are never one column: a lot is ground promised to a
    # building and the pad is the building
    for got in (small, big):
        assert got["footprint_columns"] < got["allocated_columns"], got
    # the positive control: more houses on the same lot is more of both, and the count
    # moves with them
    more = lay(10, ask=12)
    assert more["lots"] > small["lots"], (small["lots"], more["lots"])
    assert more["footprint_columns"] > small["footprint_columns"], (small, more)
    # and the measurement carries both, so a reader cannot mistake one for the other
    region = {"name": "probe", "rect": [0, 0, 119, 33],
              "scope_columns": 120 * 34, "developable_columns": 120 * 34,
              "allocated_columns": big["allocated_columns"],
              "built_columns": big["footprint_columns"], "open_requested": None,
              "lots": big["lots"]}
    leaves = [p for q in (big["plan"] or {}).get("quarters") or [] for p in q["plots"]
              if p.get("kind", "plot") == "plot"]
    m = intent_mod.lot_cover(leaves, [region])
    assert m.get("built_cover") is not None and m["built_cover"] < m["cover"], m
    return (f"one 120x34 rectangle, one type, one ask: 10-column lots hold "
            f"{small['lots']} houses on {small['allocated_columns']} columns of lot "
            f"({small['footprint_columns']} built); 16-column lots hold "
            f"{big['lots']} on {big['allocated_columns']} "
            f"({big['footprint_columns']} built) -- the lot figure rises "
            f"x{big['allocated_columns'] / small['allocated_columns']:.1f} while the "
            f"count falls; the measurement reports cover {m['cover']:.1%} and built "
            f"cover {m['built_cover']:.1%} apart; control: asking for 12 lays "
            f"{more['lots']} and {more['footprint_columns']} built")


@case
def t_e2d_a_density_finding_reaches_a_bounded_arrangement_action():
    """The dense-word finding the expression round could not act on."""
    spec, site, plateau, decls, _intent = rings_fixture()
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    place["voice"] = "x"
    finding = {"id": "find/density/lower_ring", "about": "density",
               "measure": "lot_cover", "subjects": ["lower_ring"],
               "says": ("the lots cover 9.0% of the districts' developable ground "
                        "against the at least 30% this build calls dense"),
               "target": {"value": 0.3, "direction": "up"}}
    act, subj = placesolve._action_for(finding, place, spec)
    assert act in placesolve.ARRANGEMENT_ACTIONS and subj == "lower_ring", (act, subj)
    spec2 = json.loads(json.dumps(spec))
    got, rec = placesolve.reallocate(place, spec2, finding, site=site, decls=decls,
                                     plateau=plateau, seed=1)
    assert rec["applied"] and rec["action"] in placesolve.ARRANGEMENT_ACTIONS, rec
    arr = (rec["allocation"] or {}).get("arrangement", {}).get("lower_ring")
    assert arr and arr.get("rows") == 1, rec
    # the exact count is invariant under the action
    was = sum(int(d["structures"]) for d in place["districts"])
    now = sum(int(d["structures"]) for d in got["districts"])
    assert was == now == 24, (was, now)
    # ...and a second finding on the same subject takes the **next** bounded action
    act2, _s2 = placesolve._action_for(finding, got, spec2)
    assert act2 in placesolve.ARRANGEMENT_ACTIONS and act2 != rec["action"], (act2, rec)
    return (f"the density finding routes to `{rec['action']}` on lower_ring -> "
            f"{arr}; the count stays {now}; a second pass takes `{act2}`")


# --------------------------------------- E3: owned ground, regenerated from baseline

@case
def t_e3a_an_anchor_or_voice_revision_regenerates_the_ground_from_the_baseline():
    """`stage_plateau` cut 48 columns of black paving round a chapel and no owner
    re-sized or re-paved it. The proposal is the design's, so both move with it."""
    spec, site, plateau, decls, _intent = rings_fixture()
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    base = {"path": "out/expr-rings/world.before-plateau.npz"}
    p1 = ground.propose(spec, site, place, base["path"], decls=decls)
    podium = next(p for p in p1["pieces"] if p["what"] == "podium")
    comp = next(c for c in place["compounds"] if c["name"] == podium["part"])
    side = max(comp["x1"] - comp["x0"] + 1, comp["z1"] - comp["z0"] + 1)
    cut = podium["rect"][2] - podium["rect"][0] + 1
    assert cut == side + 2 * podium["apron"], (cut, side, podium["apron"])
    assert podium["apron"] <= ground.APRON_MAX, podium
    assert str(podium["voice"] or "") == str(place["compounds"][0].get("voice")
                                             or place.get("voice")), podium
    # **a voice revision re-paves**: the same ground, a different palette, from the same
    # baseline
    place2 = json.loads(json.dumps(place))
    place2["voice"] = "other_voice"
    for c in place2["compounds"]:
        c["voice"] = "other_voice"
    p2 = ground.propose(spec, site, place2, base["path"], decls=decls)
    pod2 = next(p for p in p2["pieces"] if p["what"] == "podium")
    assert pod2["voice"] == "other_voice" and pod2["rect"] == podium["rect"], pod2
    assert p2["print"] != p1["print"], (p1["print"], p2["print"])
    # **an anchor revision re-cuts**, and the old proposal is caught before it is
    # applied rather than imposed
    place3 = json.loads(json.dumps(place))
    for c in place3["compounds"]:
        c["x0"] += 12
        c["z0"] += 12
    ev_stale = ground.evaluate(p1, place3)
    assert not ev_stale["ok"] and any(c["what"] == "anchor"
                                      for c in ev_stale["conflicts"]), ev_stale
    p3 = ground.propose(spec, site, place3, base["path"], decls=decls)
    assert ground.evaluate(p3, place3)["ok"], ground.evaluate(p3, place3)
    pod3 = next(p for p in p3["pieces"] if p["what"] == "podium")
    assert pod3["rect"] != podium["rect"], pod3
    # every proposal is of the same baseline: a revision regenerates the whole of it
    assert p3["baseline"] == p1["baseline"] == p2["baseline"], (p1, p3)
    # ...and the population and the ways in are untouched by ground work
    assert sum(int(d["structures"]) for d in place["districts"]) == 24
    assert not [c for c in ground.evaluate(p1, place)["conflicts"]
                if c["what"] == "protected"], ground.evaluate(p1, place)
    # ...and the case that named this failure: the held-out village's chapel stood on
    # `_plateau_size`'s default of 48 columns, paved in the voice the plan had when the
    # ground stage ran. The proposal is the chapel's own.
    said_held = ""
    held = os.path.join(ROOT, "out", "expr-held")
    if os.path.exists(os.path.join(held, "plan.place.json")):
        hplace = json.load(open(os.path.join(held, "plan.place.json")))
        hspec = json.load(open(os.path.join(held, "place.json")))
        hsite = json.load(open(os.path.join(held, "site.json")))
        hplat = json.load(open(os.path.join(held, "plateau.json")))
        hp = ground.propose(hspec, hsite, hplace, os.path.join(
            held, "world.before-plateau.npz"), decls=decls)
        piece = next(x for x in hp["pieces"] if x["what"] == "podium")
        was = hplat["rect"]
        was_cols = (was[2] - was[0] + 1) * (was[3] - was[1] + 1)
        now_cols = ((piece["rect"][2] - piece["rect"][0] + 1)
                    * (piece["rect"][3] - piece["rect"][1] + 1))
        assert now_cols < was_cols / 2, (was_cols, now_cols)
        assert piece["voice"] != hplat.get("voice"), (piece, hplat.get("voice"))
        assert ground.evaluate(hp, hplace)["ok"], ground.evaluate(hp, hplace)
        said_held = (f"; the held-out chapel's ground is {now_cols} columns in "
                     f"`{piece['voice']}` against the cut {was_cols} in "
                     f"`{hplat.get('voice')}`")
    return (f"the anchor is {side} a side and its ground is {cut} -- an apron of "
            f"{podium['apron']} and not a default of 48; a voice revision re-paves "
            f"({podium['voice']} -> {pod2['voice']}, same rectangle, new print); an "
            f"anchor revision is refused as stale ({ev_stale['why'][0][:60]}...) and "
            f"re-proposed at {pod3['rect']} from the same baseline" + said_held)


@case
def t_e3b_the_occupied_envelope_carries_the_great_walls_projection():
    """One envelope, four consumers. The wall's corbel, its piers and its hanging
    switchback reach past the band the layout drew; the batter steps in."""
    _t, decls = placeplan.types_card()
    if "great_wall" not in decls:
        raise Skip("no great_wall type")
    wall = {"name": "outer_wall", "kind": "edge", "type": "great_wall", "width": 3,
            "params": {"height": 48},
            "path": [[0, 0], [60, 0], [60, 60], [0, 60], [0, 0]]}
    env = ground.occupied_envelope(wall, decls)
    assert env["extends"], "the type published no occupation"
    assert env["projects"] >= 1, env
    band = env["rects"]
    solid = env["solid"]
    assert all(s[0] <= b[0] and s[2] >= b[2] for s, b in zip(solid, band)), env
    grown = env["projects"] + env["clearance"]
    assert env["envelope"][0] <= min(r[0] for r in band) - grown + env["projects"], env
    # the positive control: a type that publishes nothing occupies its band and its
    # declared clearance, and not a column more
    plain = {"name": "house", "kind": "plot", "type": "cottage",
             "x0": 0, "z0": 0, "x1": 11, "z1": 9}
    if "cottage" in decls:
        p = ground.occupied_envelope(plain, decls)
        assert p["projects"] == 0 and p["extends"] is None, p
        assert p["envelope"] == [0 - p["clearance"], 0 - p["clearance"],
                                 11 + p["clearance"], 9 + p["clearance"]], p
    # a proposal that cuts into that envelope is refused before it is applied
    proposal = {"site": {"origin": [-10, -10], "size": 200}, "pieces": [
        {"label": "podium", "part": "core", "what": "podium",
         "rect": [55, 55, 75, 75], "level": 70, "own_rect": [55, 55, 75, 75]}],
        "protected": [], "occupied": {"outer_wall": env}, "baseline": {}}
    got = ground.evaluate(proposal, {"compounds": [
        {"name": "core", "x0": 55, "z0": 55, "x1": 75, "z1": 75}]})
    assert not got["ok"] and any(c["what"] == "occupation"
                                 for c in got["conflicts"]), got
    return (f"great_wall width 3 at height 48: band {band[0]}, solid grown "
            f"{env['projects']} column(s) either side, clearance {env['clearance']}, "
            f"envelope {env['envelope']}; a podium reaching into it is refused "
            f"({got['conflicts'][-1]['what']})")


@case
def t_e3d_designed_ground_is_laid_around_a_protected_route_not_over_it():
    """A cut cannot be taken back, so ground that would destroy a route must stop the run
    by name -- and a podium whose apron merely reaches a road must not.
    """
    route = [[x, 40] for x in range(0, 120)]          # a road along z = 40
    occupied = {}
    # the apron reaches the road; the anchor itself does not
    apron = {"site": {"origin": [-20, -20], "size": 200},
             "baseline": {}, "occupied": occupied,
             "protected": [{"what": "arterial", "cells": route, "why": "the way in"}],
             "pieces": [{"label": "palace", "part": "palace", "what": "podium",
                         "rect": [0, 0, 60, 45], "own_rect": [0, 0, 60, 30],
                         "level": 70, "apron": 15}]}
    got = ground.evaluate(apron, {"compounds": [
        {"name": "palace", "x0": 0, "z0": 0, "x1": 60, "z1": 30}]})
    assert got["ok"], got["why"]
    assert got["around"] and got["around"][0]["cells"] > 0, got
    assert not [c for c in got["conflicts"] if c["what"] == "protected"], got
    # the positive control: the anchor itself stands on the road -- the ground it needs
    # cannot be laid around it, and the run stops by name
    on_it = json.loads(json.dumps(apron))
    on_it["pieces"][0]["own_rect"] = [0, 0, 60, 45]
    bad = ground.evaluate(on_it, {"compounds": [
        {"name": "palace", "x0": 0, "z0": 0, "x1": 60, "z1": 45}]})
    assert not bad["ok"] and any(c["what"] == "protected"
                                 for c in bad["conflicts"]), bad
    return (f"an apron over {got['around'][0]['cells']} cell(s) of the arterial is laid "
            f"around them; an anchor drawn over the same road is refused: "
            f"{bad['why'][0][:80]}...")


@case
def t_e3c_apply_always_starts_from_the_baseline():
    """A revision regenerates the whole proposal from the baseline; cuts never
    accumulate. Run on the retained volume, which is read and never written."""
    import numpy as np
    from ethoslm import offline
    d = os.path.join(ROOT, "out", "expr-rings")
    p = os.path.join(d, "world.before-plateau.npz")
    if not os.path.exists(p):
        raise Skip("no out/expr-rings/world.before-plateau.npz")
    spec, site, plateau, decls, _intent = rings_fixture()
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, "x")
    assert not fails, fails
    prop = ground.propose(spec, site, place, p, decls=decls)
    podium = next(x for x in prop["pieces"] if x["what"] == "podium")
    # two applications of the same proposal, each from its own load of the baseline
    first = json.loads(json.dumps(prop))
    one = ground.apply(first, offline.load_volume(p))
    two = ground.apply(json.loads(json.dumps(prop)), offline.load_volume(p))
    h1, h2 = ground.bed_heights(one), ground.bed_heights(two)
    assert np.array_equal(h1, h2), "the same proposal from the baseline twice differs"
    # ...and a *revised* proposal applied from the baseline is not the first cut plus
    # the second: it is the second alone
    place2 = json.loads(json.dumps(place))
    for c in place2["compounds"]:
        c["x1"] -= 10
        c["z1"] -= 10
    prop2 = ground.propose(spec, site, place2, p, decls=decls)
    three = ground.apply(prop2, offline.load_volume(p))
    h3 = ground.bed_heights(three)
    assert not np.array_equal(h1, h3), "the revision changed no ground"
    stacked = ground.apply(json.loads(json.dumps(prop2)), one)
    hs = ground.bed_heights(stacked)
    assert not np.array_equal(h3, hs), \
        "applying the revision on top of the first cut gives the same ground as " \
        "applying it from the baseline, so this check proves nothing"
    base_h = ground.bed_heights(offline.load_volume(p))
    moved_fresh = int((h3 != base_h).sum())
    moved_stacked = int((hs != base_h).sum())
    assert moved_stacked > moved_fresh, (moved_fresh, moved_stacked)
    return (f"the podium {podium['rect']} at y={podium['level']} cuts "
            f"{(first.get('applied') or {}).get('blocks')} block(s) from the baseline, "
            f"the same twice; the revised anchor moves {moved_fresh} column(s) from "
            f"the baseline against {moved_stacked} if the first cut were left standing "
            f"under it")


# ------------------------------------------------------------------ the runner

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = skipped = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__[2:]
        if only and only not in name:
            continue
        try:
            said = fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            skipped += 1
            continue
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} design spatial cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
