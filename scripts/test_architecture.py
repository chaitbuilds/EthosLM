"""The architecture round: the contracts, and every negative case they exist for.

    $PY scripts/test_architecture.py

Offline, deterministic, no model call and no server. Each case is one of the failures
the architecture audit reproduced, written as a check that the failure is now refused.

  1. **the records refuse what they do not know.** An unknown field is an error, a
     version this build does not read is an error, a claim with no source and no
     `inferred` flag is an error, and a finding routed nowhere is an error.
  2. **the omitted wall.** A sentence asking for a walled village whose spec drops the
     wall fails coverage by name -- and the same spec with the wall passes.
  3. **the district that was present because another one was.** A spec declaring two
     districts, with plots in only one of them, fails the presence clause for the one
     nobody drew; and an unbuilt plot is not a standing one.
  4. **the exact count that was reduced.** "Exactly 2,000 houses" past the ceiling
     keeps the band the sentence set, carries `unmet`, fails coverage, and is refused a
     repair. Five- and six-digit counts are read at all.
  5. **the capability that was a filename.** A type of the wrong form, wrong kind or
     wrong envelope does not close a gap, and neither does a file already on disk.
     A harbour is `unsupported` and is never answered with a square.
  6. **the geometry that could not be laid.** A shoreline request on a dry site is
     refused by name; the ring layout budgets districts against the wall's real mass;
     a round ring's districts follow its chamfer.
  7. **the stale artifact.** A site chosen for a village is not reused for a city; an
     interrupted run's `done` marker is not completion; unchanged inputs stay warm.
  8. **the repair.** A capacity shortfall against an *inferred* band is negotiated
     inside a registered bound, rechecked, and closes; the same shortfall against an
     **explicit** count is refused and stays open.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (capability, contracts, deps, evidence, intent, offline,  # noqa: E402
                   observe, pipeline, placeplan, placeread, placeshore, placesolve,
                   placeregion as regions, repair, resolve,
                   spec as spec_mod)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


class Skip(Exception):
    pass


# ------------------------------------------------------------------- fixtures

WALLED_VILLAGE = "Build a walled village of about twenty houses."

VILLAGE_SPEC = {
    "sentence": WALLED_VILLAGE, "kind": "village",
    "defining_parts": [
        {"name": "village_wall", "kind": "edge", "family": "wall",
         "relation": "perimeter", "count": 1, "structures": 0,
         "notes": "A circuit of wall round the whole place."},
        {"name": "village_gate", "kind": "point", "family": "gate",
         "relation": "gateway", "count": 1, "structures": 0,
         "notes": "One way through it."},
        {"name": "homes", "kind": "group", "family": "district",
         "relation": "throughout", "count": 1, "structures": 20,
         "notes": "The houses."}],
    "voice": None, "form": "european_vernacular"}


def _spec(doc=None, sentence=WALLED_VILLAGE):
    return spec_mod.read_spec(copy.deepcopy(doc or VILLAGE_SPEC), sentence)


def _no_wall_spec():
    doc = copy.deepcopy(VILLAGE_SPEC)
    doc["defining_parts"] = [p for p in doc["defining_parts"]
                             if p["family"] != "wall"]
    return spec_mod.read_spec(doc, WALLED_VILLAGE)


def shore_volume(size=192, y0=40, sea_y=63, land_y=66, sea_at=96, water=True):
    """A synthetic site: water to the north of `sea_at`, land to the south.

        Synthetic on purpose, and labelled so in every readout that uses it: this is a unit
        fixture and it is not a site the search chose on recorded terrain.
        
    """
    H = 40
    codes = np.zeros((size, H, size), np.int16)
    palette = ["air", "stone", "water", "grass_block"]
    for x in range(size):
        for z in range(size):
            if water and z < sea_at:
                codes[x, :sea_y - y0 + 1, z] = 1
                codes[x, sea_y - y0 - 3:sea_y - y0 + 1, z] = 2
            else:
                codes[x, :land_y - y0 + 1, z] = 1
                codes[x, land_y - y0, z] = 3
    return observe.Volume(0, y0, 0, codes, palette)


SHORE_SENTENCE = ("Build an unwalled village following the shoreline, with homes "
                  "facing the water.")

SHORE_SPEC = {
    "sentence": SHORE_SENTENCE, "kind": "village",
    "defining_parts": [
        {"name": "green", "kind": "area", "family": "square",
         "relation": "beside_the_centre", "count": 1, "structures": 0,
         "notes": "Open ground where the boats are drawn up."},
        {"name": "homes", "kind": "group", "family": "district",
         "relation": "throughout", "count": 1, "structures": 0, "density": "low",
         "notes": "The houses, strung along the water."}],
    "voice": None, "form": "european_vernacular"}


def _plan_of(parts, districts=()):
    """A plan tree in `plan_parts` shape, with the district link on every quarter."""
    children = [{"kind": "quarter", "name": "defining", "children": list(parts)}]
    for name, defines, plots in districts:
        children.append({"kind": "quarter", "name": name, "defines": defines,
                         "children": list(plots)})
    return {"parts": [{"kind": "district", "name": "place", "children": children}]}


def _stood(plan, standing=True):
    return {"waves": [{"wave": "all", "parts": [
        {"part": p["name"], "status": "built", "stood": bool(standing)}
        for p in pipeline.plan_parts(plan)]}]}


# ------------------------------------------------- 1. the records refuse

@case
def t_1_a_record_refuses_a_field_a_version_and_a_claim_it_does_not_know():
    # an unknown field is an error and not a shrug
    try:
        contracts.read("intent", {"record": "intent", "version": 1,
                                  "sentence": "x", "requirements": [],
                                  "organisation": "rings"})
        raise AssertionError("an unknown top-level field was accepted")
    except contracts.ContractError as e:
        assert "organisation" in str(e), e
    # ...and inside a row, too
    try:
        contracts.read("intent", {"record": "intent", "version": 1, "sentence": "x",
                                  "requirements": [{"id": "a", "says": "b",
                                                    "kind": "feature", "wants": {},
                                                    "architectural_features": []}]})
        raise AssertionError("an unknown row field was accepted")
    except contracts.ContractError as e:
        assert "architectural_features" in str(e), e
    # a version this build does not read
    try:
        contracts.read("intent", {"record": "intent", "version": 99, "sentence": "x",
                                  "requirements": []})
        raise AssertionError("a future version was accepted")
    except contracts.ContractError as e:
        assert "version" in str(e), e
    # a claim with no source and no `inferred` flag: a recollection is not evidence
    try:
        contracts.make("reading", sentence="x", classification="named_reference",
                       claims=[{"id": "c1", "says": "it has three walls"}])
        raise AssertionError("an unsourced claim was accepted")
    except contracts.ContractError as e:
        assert "inferred" in str(e), e
    ok = contracts.make("reading", sentence="x", classification="named_reference",
                        claims=[{"id": "c1", "says": "three walls", "inferred": True}])
    assert ok["claims"][0]["inferred"]
    # a finding routed nowhere, and one blocking nothing
    for bad, word in (({"owner": "somebody"}, "owner"), ({"blocks": "later"}, "blocks")):
        try:
            contracts.make("findings", findings=[{
                "id": "f", "says": "s", "owner": "layout", "blocks": "fidelity",
                **bad}])
            raise AssertionError(f"a finding with a bad {word} was accepted")
        except contracts.ContractError as e:
            assert word in str(e), e
    # a legacy document is adapted **explicitly** and says it was
    legacy = contracts.read("reading", {"sentence": "x", "claims": [{"id": "c",
                                                                    "says": "y"}]})
    assert legacy["note"].startswith("adapted from"), legacy["note"]
    assert legacy["claims"][0]["inferred"] is True, legacy["claims"]
    assert legacy["provider"] == "legacy"
    return ("unknown fields at both levels, a future version, an unsourced claim, a "
            "finding with no owner and one blocking nothing are each refused by name; "
            "a legacy reading is adapted, labelled, and its claims marked inferred")


# ------------------------------------------------- 2. the omitted wall

@case
def t_2_a_sentence_that_asks_for_a_wall_fails_a_spec_that_drops_it():
    it = intent.read(WALLED_VILLAGE)
    ids = [r["id"] for r in it["requirements"]]
    assert "feature/wall" in ids and "count/structures" in ids, ids
    # the spec that drops the wall
    checked, found = intent.coverage(it, _no_wall_spec())
    wall = next(r for r in checked["requirements"] if r["id"] == "feature/wall")
    assert wall["status"] == "failed", wall
    assert wall["owner"] == "reading", wall
    assert any(f["requirement"] == "feature/wall" and f["blocks"] == "fidelity"
               for f in found["findings"]), found["findings"]
    assert not intent.holds(checked)
    # ...and the spec that keeps it does not fail for the wall
    checked2, _f2 = intent.coverage(intent.read(WALLED_VILLAGE), _spec())
    wall2 = next(r for r in checked2["requirements"] if r["id"] == "feature/wall")
    assert wall2["status"] == "open", wall2
    assert "nothing is planned yet" in wall2["why"], wall2
    # a drawn wall is still not a standing one, and a standing one satisfies it
    parts = [{"kind": "edge", "name": "village_wall", "defines": "village_wall",
              "type": "wall", "path": [[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]],
              "width": 1}]
    plan = _plan_of(parts)
    checked3, _ = intent.coverage(intent.read(WALLED_VILLAGE), _spec(), plan=plan)
    assert next(r for r in checked3["requirements"]
                if r["id"] == "feature/wall")["status"] == "open"
    checked4, _ = intent.coverage(intent.read(WALLED_VILLAGE), _spec(), plan=plan,
                                  parts_record=_stood(plan))
    assert next(r for r in checked4["requirements"]
                if r["id"] == "feature/wall")["status"] == "satisfied"
    checked5, found5 = intent.coverage(intent.read(WALLED_VILLAGE), _spec(), plan=plan,
                                       parts_record=_stood(plan, standing=False))
    w5 = next(r for r in checked5["requirements"] if r["id"] == "feature/wall")
    assert w5["status"] == "failed" and w5["owner"] == "build", w5
    # an `unwalled` sentence is the same rule the other way round
    un = intent.read("Build an unwalled village following the shoreline.")
    assert [r["id"] for r in un["requirements"]][:1] == ["absent/wall"], un
    bad, found_b = intent.coverage(un, _spec())
    assert next(r for r in bad["requirements"]
                if r["id"] == "absent/wall")["status"] == "failed"
    return ("a dropped wall fails `feature/wall` at the reading; a drawn one is open, a "
            "standing one satisfied and a fallen one failed at the build; `unwalled` "
            "fails on a spec that declares one")


# ------------------------------------------------- 3. the district nobody drew

@case
def t_3_a_district_is_present_only_when_its_own_plots_stand():
    doc = copy.deepcopy(VILLAGE_SPEC)
    doc["defining_parts"].append({"name": "missing_district", "kind": "group",
                                  "family": "district", "relation": "quarter",
                                  "count": 1, "structures": 4,
                                  "notes": "a district nobody draws"})
    spec = spec_mod.read_spec(doc, WALLED_VILLAGE)
    plots = [{"kind": "plot", "name": "h1", "type": "cottage",
              "x0": 20, "z0": 20, "x1": 28, "z1": 28, "seed": 1, "params": {}}]
    plan = _plan_of([], districts=[("homes_row", "homes", plots)])
    got = placeread.read(spec, plan, _stood(plan), voice="drystone_and_thatch")
    assert "present/missing_district" in got["failed"], got["failed"]
    assert "present/homes" not in got["failed"], got["failed"]
    # ...and a plot that does not stand does not make its own district present
    got2 = placeread.read(spec, plan, _stood(plan, standing=False),
                          voice="drystone_and_thatch")
    assert "present/homes" in got2["failed"], got2["failed"]
    # a plan with no link at all.
    plan_old = {"parts": [{"kind": "district", "name": "place", "children": [
        {"kind": "quarter", "name": "north_quarter", "children": plots}]}]}
    got3 = placeread.read(spec, plan_old, _stood(plan_old),
                          voice="drystone_and_thatch")
    row = next(c for c in got3["clauses"] if c["clause"] == "present/homes")
    assert row["holds"] and row["linked"] is False, row
    return ("a district with no plots of its own fails its presence clause while its "
            "neighbour passes; an unbuilt plot fails it too; a plan carrying no "
            "quarter-to-part link reads exactly as it did")


# ------------------------------------------------- 4. the count that was reduced

@case
def t_4_an_exact_count_is_never_weakened_to_make_it_pass():
    assert spec_mod.count_in("a town of 10000 houses")["n"] == 10000
    assert spec_mod.count_in("a city of 12,000 houses")["n"] == 12000
    assert spec_mod.count_in("about sixty houses") == {"n": 60, "about": True,
                                                       "phrase": "about sixty houses"}
    sentence = "Build a city of exactly 2000 houses inside a great wall."
    doc = {"sentence": sentence, "kind": "city",
           "defining_parts": [
               {"name": "city_wall", "kind": "edge", "family": "wall",
                "relation": "perimeter", "count": 1, "structures": 0,
                "notes": "the circuit"},
               {"name": "city_gate", "kind": "point", "family": "gate",
                "relation": "gateway", "count": 1, "structures": 0, "notes": "a gate"},
               {"name": "quarters", "kind": "group", "family": "district",
                "relation": "throughout", "count": 3, "structures": 2000,
                "notes": "the districts"}],
           "voice": None, "form": "european_vernacular"}
    s = spec_mod.read_spec(copy.deepcopy(doc), sentence)
    cap = spec_mod.structures_ceiling("city")
    assert s["structures"] <= cap, s["structures"]
    assert s["scaled_from"], "2000 was not scaled"
    # **the band the sentence set is where it was**
    assert s["size_band"] == [2000, 2000], s["size_band"]
    assert s["scaled_from"]["band_held"] is True
    assert s["unmet"]["asked"] == 2000 and s["unmet"]["limit"] == "structures_ceiling"
    assert not spec_mod.in_band(s, s["structures"]), "the reduced count passed the band"
    # read again off disk: the record is the record
    again = spec_mod.read_spec(json.loads(json.dumps(s)), sentence)
    assert again["size_band"] == [2000, 2000] and again["unmet"], again.get("unmet")
    # coverage fails, and the repair refuses to negotiate it
    it, found = intent.coverage(intent.read(sentence), s)
    row = next(r for r in it["requirements"] if r["id"] == "count/structures")
    assert row["status"] == "failed" and row["owner"] == "scale", row
    rec = repair.apply(None, s, found)
    assert not rec["applied"], rec["applied"]
    assert any("explicit" in r["why"] or "named this count" in r["why"]
               for r in rec["refused"]), rec["refused"]
    assert s["size_band"] == [2000, 2000], s["size_band"]
    # ...while a kind's *inferred* band is still scaled, as it always was
    plain = {"sentence": "Build Ringed City.", "kind": "city",
             "defining_parts": copy.deepcopy(doc["defining_parts"])}
    for p in plain["defining_parts"]:
        if p["family"] == "district":
            p["structures"] = 3 * cap
    ps = spec_mod.read_spec(plain, "Build Ringed City.")
    assert ps["scaled_from"] and ps["scaled_from"]["band_held"] is False
    assert spec_mod.in_band(ps, ps["structures"])
    return (f"2000 scaled to {s['structures']} with the band left at 2000-2000, `unmet` "
            f"naming the ceiling, coverage failing and the repair refusing it; an "
            f"inferred band still scales; 10000 and 12,000 are read at all")


# ------------------------------------------------- 5. the capability

@case
def t_5_a_capability_is_declarations_and_not_a_filename():
    plot = {"kind": "plot", "form": "european_vernacular", "role": "urban",
            "needs": {"footprint": (4, 4, 16, 16), "clearance": 2}, "params": {},
            "declares_needs": True}
    want = {"id": "cap/x", "name": "watchtower", "kind": "point", "family": "tower",
            "forms": ["east_asian"], "roles": ["defensive"]}
    # the wrong kind, the wrong form, the wrong role and the wrong envelope, each named
    assert not capability.fits(capability.card("tower", plot), want)[0]
    assert "builds a plot" in capability.fits(capability.card("tower", plot), want)[1]
    point = dict(plot, kind="point", role="defensive")
    said = capability.fits(capability.card("tower", point), want)
    assert not said[0] and "european_vernacular" in said[1], said
    right = dict(point, form="east_asian")
    assert capability.fits(capability.card("tower", right), want)[0]
    # a file of the right name and the wrong form does not close the gap
    assert not capability.fits(capability.card("tower", dict(right, role="rural")),
                               want)[0]
    # ...and a name alone never admits: `gate_tower` is not a tower
    assert not capability.fits(capability.card("gate_tower", right), want)[0]
    # ...unless the file says what family it is
    assert capability.fits(capability.card("gate_tower", dict(right, family="tower")),
                           want)[0]
    # an envelope that cannot hold the part
    small = dict(right, needs={"footprint": (3, 3, 5, 5), "clearance": 0})
    said = capability.fits(capability.card("tower", small), dict(want, footprint=(9, 9)))
    assert not said[0] and "9x9" in said[1], said
    # `adoptable`: a real file on disk, checked against a real want
    path = os.path.join(ROOT, "types", "gate_tower.py")
    ok, why = capability.adoptable(path, want)
    assert not ok and "does not answer this gap" in why, (ok, why)
    ok2, _ = capability.adoptable(path, {**want, "family": "gate", "forms": [None],
                                         "roles": [None], "passage": True})
    assert ok2, "the real gate_tower does not answer a gate want"
    # an unsupported family is unsupported and is never answered with a square
    it = intent.read("Build a fishing village with a harbour and about sixty houses.")
    harbour = next(r for r in it["requirements"] if r["id"] == "feature/harbour")
    assert harbour["status"] == "unsupported" and harbour["hard"], harbour
    _c, found = intent.coverage(it, _spec())
    row = next(f for f in found["findings"] if f["requirement"] == "feature/harbour")
    assert row["owner"] == "capability" and row["blocks"] == "fidelity", row
    return ("kind, form, role, envelope and family each refuse a candidate by name; a "
            "file of the right name and wrong form is not adoptable; a harbour is "
            "`unsupported` and never substituted")


# ------------------------------------------------- 6. the geometry

@case
def t_6_the_layouts_share_one_region_interface_and_refuse_what_they_cannot_lay():
    # a shoreline request on a dry site is refused by name
    dry = shore_volume(water=False)
    spec = spec_mod.read_spec(copy.deepcopy(SHORE_SPEC), SHORE_SENTENCE)
    assert placesolve.policy_for(spec) == "shoreline"
    _t, decls = placeplan.types_card(None, spec.get("form"))
    site = {"origin": [0, 0], "size": 192}
    place, fails = placesolve.solve_place(spec, site, None, decls,
                                          "drystone_and_thatch", vol=dry)
    assert place is None and fails[0]["check"] == "shoreline", fails
    assert "no shoreline can be derived" in fails[0]["why"], fails[0]["why"]
    # ...and on a site with water it is laid against the water the ground has
    wet = shore_volume()
    place, fails = placesolve.solve_place(spec, site, None, decls,
                                          "drystone_and_thatch", vol=wet)
    assert not fails, fails
    assert place["layout"]["policy"] == "shoreline"
    anchor = place["layout"]["anchor"]
    assert anchor["water_side"] == "north" and anchor["along"] == "x", anchor
    ok, why = placeshore.faces_water(place)
    assert ok, why
    # the same validator the ring layout answers to
    ground = pipeline.plan_ground([{**p, "name": p.get("name")}
                                   for p in place["parts"]], wet)
    assert placeplan.place_failures(place, spec, site, decls, ground=ground,
                                    voice="drystone_and_thatch") == [], "the shoreline " \
        "layout fails the place validator"
    # no hidden centre: nothing is declared at one and none is invented
    assert place["centre"] is None, place["centre"]
    # the districts are ribbons: every one longer along the shore than across it
    for d in place["districts"]:
        assert (d["x1"] - d["x0"]) >= (d["z1"] - d["z0"]), d
    # **the ring layout budgets districts against the wall's real mass.**
    import test_rings
    s = test_rings.three_ring_spec()
    for p in s["defining_parts"]:
        if p["family"] == "wall":
            p["notes"] = (p.get("notes") or "") + " A rampart, an earthwork of great mass."
    screen, fails_s = test_rings._layout(test_rings.three_ring_spec())[:2]
    rampart, fails_r = test_rings._layout(s)[:2]
    assert not fails_s and not fails_r, (fails_s, fails_r)
    w_screen = [r["insets"] for r in screen["layout"]["rings"]]
    w_ramp = [r["insets"] for r in rampart["layout"]["rings"]]
    assert w_ramp != w_screen, "a rampart budgeted the same ground as a screen"
    assert max(sum(i) for i in w_ramp) > max(sum(i) for i in w_screen), (w_ramp, w_screen)
    # **a round ring's districts follow its chamfer.** The generic tiler covers more of
    # a chamfered annulus than the four strips clipped at their ends ever could.
    cx = cz = 384
    for chamfer in (0, 60, 105):
        mask = regions.annulus_mask(0, 0, 768, cx, cz, 120, 300,
                                    chamfer_out=chamfer,
                                    chamfer_in=int(120 * chamfer / 300.0))
        reg = regions.region("ring", mask, 0, 0)
        rects = regions.tile(reg, most=16)
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert not (a[0] <= b[2] and b[0] <= a[2]
                            and a[1] <= b[3] and b[1] <= a[3]), (a, b)
        cover = sum((a[2] - a[0] + 1) * (a[3] - a[1] + 1) for a in rects) \
            / float(reg["columns"])
        assert cover >= placeplan.RING_COVERAGE, (chamfer, cover)
        if chamfer:
            strips = regions.ring_sectors(cx, cz, 120, 300, 126, 300, chamfer=chamfer)
            got = sum(int(mask[max(0, a):c + 1, max(0, b):d + 1].sum())
                      for _l, (a, b, c, d) in strips) / float(mask.sum())
            assert cover > got, (chamfer, cover, got)
    return ("a dry site refuses a shoreline by name; a wet one is laid against the "
            "water's own line, faces it, keeps no centre and passes the ring layout's "
            "own validator; a rampart is budgeted wider than a screen; a chamfered "
            "annulus tiles above the coverage bar and above what strips reach")


# ------------------------------------------------- 7. the stale artifact

def _tmp_round(tmp, sentence, spec_doc, size=192):
    state = os.path.join(tmp, "state")
    os.makedirs(state, exist_ok=True)
    offline.save_volume(shore_volume(size=size), os.path.join(state, "world.npz"))
    json.dump({"origin": [0, 0], "size": size},
              open(os.path.join(state, "site.json"), "w"))
    json.dump(spec_doc, open(os.path.join(state, "place.checked.json"), "w"))
    cfg = os.path.join(tmp, "r.json")
    json.dump({"name": "arch_fixture", "state_dir": state, "sentence": sentence,
               "flags": {"dry_run": True}}, open(cfg, "w"), indent=1)
    return pipeline.Round.load(cfg)


@case
def t_7_an_artifact_is_reused_only_where_what_it_was_made_from_has_not_moved():
    with tempfile.TemporaryDirectory() as tmp:
        doc = spec_mod.read_spec(copy.deepcopy(VILLAGE_SPEC), WALLED_VILLAGE)
        rnd = _tmp_round(tmp, WALLED_VILLAGE, doc)
        json.dump({"chosen": {"origin": [0, 0], "size": 192}},
                  open(rnd.rel("site_search.json"), "w"))
        contracts.save(rnd, "intent", intent.read(WALLED_VILLAGE))
        # unstamped: reused exactly as it always was
        fresh, why = deps.check(rnd, "site_search")
        assert not fresh and "carries no dependency stamp" in why, why
        deps.stamp(rnd, "site_search", outputs=["site_search.json"])
        fresh, why = deps.check(rnd, "site_search")
        assert fresh, why
        # **a change the search does not consume keeps it warm.** The design round split
        # `site_search` off the whole spec print onto `site_demand` -- footprint, needs,
        # setting, explicit count, kind -- because the expression round's city re-ran a
        # twenty-five-minute search over cached squares after a *voice* revision, twice,
        # and chose the same square both times. A voice is not a fact about the ground.
        voiced = copy.deepcopy(doc)
        voiced["voice"] = "japanese_minka"
        json.dump(voiced, open(rnd.rel("place.checked.json"), "w"))
        fresh, why = deps.check(rnd, "site_search")
        assert fresh, f"a voice-only change invalidated the site search: {why}"
        # **the spec becomes a city's: the village's site is stale**
        big = copy.deepcopy(doc)
        big["kind"] = "city"
        big["structures"] = 300
        json.dump(big, open(rnd.rel("place.checked.json"), "w"))
        fresh, why = deps.check(rnd, "site_search")
        assert not fresh and "site_demand" in why, why
        # an interrupted run: the stamp names an output that is not there
        json.dump(doc, open(rnd.rel("place.checked.json"), "w"))
        assert deps.check(rnd, "site_search")[0]
        os.remove(rnd.rel("site_search.json"))
        fresh, why = deps.check(rnd, "site_search")
        assert not fresh and "interrupted" in why, why
        # ...and a stamp is refused before its outputs exist
        try:
            deps.stamp(rnd, "site_search", outputs=["site_search.json"])
            raise AssertionError("a stamp was written before its output")
        except ValueError as e:
            assert "before its outputs exist" in str(e), e
        # invalidation reaches the artifacts downstream
        json.dump({"chosen": {"origin": [0, 0], "size": 192}},
                  open(rnd.rel("site_search.json"), "w"))
        deps.stamp(rnd, "site_search", outputs=["site_search.json"])
        json.dump({"revisions": 0, "drawn": {}, "done": True},
                  open(rnd.rel("preview.json"), "w"))
        deps.stamp(rnd, "preview", outputs=["preview.json"])
        dropped = deps.invalidate(rnd, "site")
        assert "preview" in dropped and "plan" not in dropped, dropped
    return ("an unstamped artifact is reused as it always was; a site search survives a "
            "voice change and goes stale when the site's own demand moves; an output "
            "that is not there is an interrupted run and not a finished one; a stamp "
            "before its output is refused; invalidating a kind drops every artifact "
            "that depends on it")


# ------------------------------------------------- 8. the repair

@case
def t_8_an_inferred_target_is_negotiated_inside_a_bound_and_rechecked():
    spec = spec_mod.read_spec(copy.deepcopy(SHORE_SPEC), SHORE_SENTENCE)
    site = {"origin": [0, 0], "size": 192}
    vol = shore_volume()
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placesolve.solve_place(spec, site, None, decls,
                                          "drystone_and_thatch", vol=vol)
    assert not fails, fails
    it = intent.read(SHORE_SENTENCE)
    caps = capability.match(spec)
    checked, res, found = resolve.findings_for(spec, place, site, it, capabilities=caps)
    short = [f for f in found["findings"] if f["id"] == "find/capacity/band"]
    assert short, [f["id"] for f in found["findings"]]
    assert short[0]["owner"] == "scale" and short[0]["blocks"] == "feasibility"
    # the depth of the band was already negotiated inside its own registered bound
    assert any(n["what"] == "band_depth" for n in res["negotiated"]), res["negotiated"]
    assert all("bound" in n for n in res["negotiated"] if n["what"] == "band_depth")
    was = list(spec["size_band"])
    rec = repair.apply(None, spec, found)
    assert rec["applied"] and rec["changed_plan"], rec
    assert rec["bounds"]["SCALE_NEGOTIATION_MIN"] == repair.SCALE_NEGOTIATION_MIN
    assert spec["size_band"] != was, spec["size_band"]
    assert spec["negotiated"][-1]["what"] == "size_band"
    # **rechecked**: the plan is laid out again and the finding is closed
    place2, fails2 = placesolve.solve_place(spec, site, None, decls,
                                            "drystone_and_thatch", vol=vol)
    assert not fails2, fails2
    _c2, _r2, found2 = resolve.findings_for(spec, place2, site, intent.read(
        SHORE_SENTENCE), capabilities=caps)
    assert not [f for f in found2["findings"] if f["id"] == "find/capacity/band"], \
        found2["findings"]
    # ...and the negotiated target survives being read off disk
    again = spec_mod.read_spec(json.loads(json.dumps(spec)), SHORE_SENTENCE)
    assert again["size_band"] == spec["size_band"], (again["size_band"],
                                                     spec["size_band"])
    # a shortfall past the bound is refused, not negotiated further
    far = copy.deepcopy(spec)
    far["size_band"] = [400, 800]
    far.pop("negotiated", None)
    bad = contracts.make("findings", findings=[{
        "id": "find/capacity/band", "says": "short", "owner": "scale",
        "blocks": "feasibility",
        "evidence": {"promised": 11, "band": [400, 800]}}])
    rec2 = repair.apply(None, far, bad)
    assert not rec2["applied"] and rec2["refused"], rec2
    assert "least a negotiated target may be" in rec2["refused"][0]["why"], rec2
    assert far["size_band"] == [400, 800]
    return (f"a shore band holding {res['bounds']['structures_promised']} against "
            f"{was} is negotiated to {spec['size_band']} inside "
            f"{repair.SCALE_NEGOTIATION_MIN:.0%} of the floor, the plan laid out again "
            f"and the finding closed; past the bound it is refused and the band stands")


# ------------------------------------------------- 9. the evidence

@case
def t_9_evidence_is_retrieved_recorded_and_never_invented():
    assert evidence.classify("Build Ringed City.")["kind"] == "named_reference"
    assert evidence.classify("Build Ringed City.")["name"] == "Ringed City"
    assert evidence.classify("Build a Japanese village.")["tradition"] == "japanese"
    assert evidence.classify(SHORE_SENTENCE)["kind"] == "self_contained"
    assert evidence.queries(SHORE_SENTENCE) == []
    assert len(evidence.queries("Build Ringed City.")) == 3
    assert evidence.needs_evidence("Build Ringed City.")
    assert not evidence.needs_evidence(SHORE_SENTENCE)
    # with no provider configured the gathering is empty and says why
    with tempfile.TemporaryDirectory() as tmp:
        got = evidence.gather("Build Ringed City.", os.path.join(tmp, "e"),
                              prov=evidence.NoProvider())
        assert got["sources"] == [] and got["provider"] == "none"
        rec = evidence.reading_of("Build Ringed City.", got)
        assert rec["claims"] == [] and rec["uncertainty"], rec
        assert "no source" in evidence.brief(rec)
        # a cassette: the same path, deterministic, and labelled `recorded`
        root = os.path.join(tmp, "cassette")
        os.makedirs(root)
        open(os.path.join(root, "p1.html"),
             "w").write("<html><body><p>Three concentric walls.</p></body></html>")
        json.dump({"queries": {evidence.queries("Build Ringed City.")[0]:
                               [{"title": "A page", "url": "http://x/1",
                                 "snippet": "walls"}]},
                   "pages": {"http://x/1": {"file": "p1.html", "media": "text/html",
                                            "title": "A page"}}},
                  open(os.path.join(root, "index.json"), "w"))
        prov = evidence.RecordedProvider(root)
        got2 = evidence.gather("Build Ringed City.", os.path.join(tmp, "e2"), prov=prov)
        assert len(got2["sources"]) == 1, got2
        s = got2["sources"][0]
        assert s["url"] == "http://x/1" and s["fingerprint"].startswith("sha256:")
        assert s["provider"] == "recorded" and s["accessed"]
        text = open(os.path.join(tmp, "e2", s["path"])).read()
        assert "Three concentric walls." in text and "<p>" not in text, text
        rec2 = evidence.reading_of("Build Ringed City.", got2, claims=[
            {"id": "claim/1", "says": "three concentric walls", "source": "src/1",
             "about": "layout"}])
        assert rec2["sources"][0]["fingerprint"] == s["fingerprint"]
        assert "src/1" in evidence.brief(rec2)
        # a claim naming no source is still refused, cassette or not
        try:
            evidence.reading_of("Build Ringed City.", got2,
                                claims=[{"id": "c", "says": "it is red"}])
            raise AssertionError("an unsourced claim was accepted")
        except contracts.ContractError:
            pass
    return ("a named place, a tradition and a description are told apart and only the "
            "first two are searched for; with no provider the reading is empty, "
            "uncertain and says so; a cassette produces a fingerprinted source and its "
            "stripped text; an unsourced claim is refused either way")


# ------------------------------------------------- 10. the construction sample

@case
def t_10_a_construction_sample_spans_two_adjoining_quarters_and_what_joins_them():
    """A sample is a seam, and the rule that picks it is on the record.

        Not the largest few quarters scattered over a place, and not one house: the two
        quarters with the most plots that **touch**, plus every edge, point and area within
        the margin of their union -- the boundary and the way through it.
        
    """
    from ethoslm.pipeline import stages_build
    plots_a = [{"kind": "plot", "name": f"a{i}", "type": "cottage", "x0": 10 * i,
                "z0": 0, "x1": 10 * i + 8, "z1": 8, "in": ["place", "qa"]}
               for i in range(5)]
    plots_b = [{"kind": "plot", "name": f"b{i}", "type": "cottage", "x0": 10 * i,
                "z0": 40, "x1": 10 * i + 8, "z1": 48, "in": ["place", "qb"]}
               for i in range(4)]
    far = [{"kind": "plot", "name": f"c{i}", "type": "cottage", "x0": 600 + 10 * i,
            "z0": 600, "x1": 608 + 10 * i, "z1": 608, "in": ["place", "qc"]}
           for i in range(6)]
    between = {"kind": "area", "name": "green", "type": "square",
               "x0": 0, "z0": 20, "x1": 40, "z1": 30, "in": ["place", "defining"]}
    away = {"kind": "area", "name": "elsewhere", "type": "square",
            "x0": 900, "z0": 900, "x1": 940, "z1": 940, "in": ["place", "defining"]}
    parts = plots_a + plots_b + far + [between, away]
    got, rec = stages_build.sample_parts(parts, {"quarters": 2, "margin": 24})
    names = {p["name"] for p in got}
    # the biggest quarter, and then the one **nearest it** -- not the next biggest
    assert rec["quarters"] == ["qc", "qb"] or rec["quarters"][0] == "qc", rec
    got2, rec2 = stages_build.sample_parts(
        [p for p in parts if (p.get("in") or [None])[-1] != "qc"],
        {"quarters": 2, "margin": 24})
    assert sorted(rec2["quarters"]) == ["qa", "qb"], rec2
    names2 = {p["name"] for p in got2}
    assert "green" in names2, "the area on the seam was not taken"
    assert "elsewhere" not in names2, "an area far from the sample was taken"
    assert rec2["plots"] == 9 and rec2["of_plan"]["quarters"] == 2, rec2
    # deterministic
    assert stages_build.sample_parts(parts, {"quarters": 2, "margin": 24})[1] == rec
    return (f"the sample is {rec2['plots']} plots in {rec2['quarters']} plus the area "
            f"on their seam; an area 900 columns away is not in it; the same plan gives "
            f"the same sample twice")


# ---------------------------------- 11. the sentence that read as nothing

@case
def t_11_a_sentence_this_reader_cannot_parse_is_never_read_as_asking_for_nothing():
    """The integration review's first finding, as the assertion it should have been."""
    named = intent.read("Build Ringed City.")
    ids = [r["id"] for r in named["requirements"]]
    assert "identity/ringed_city" in ids, ids
    assert not intent.holds(named), "a named city still holds before anything is planned"
    trad = intent.read("Build a Japanese village.")
    row = next(r for r in trad["requirements"] if r["id"] == "tradition/japanese")
    assert row["hard"] and row["wants"]["nearest_form"] == "east_asian", row
    assert not intent.holds(trad)
    # the combined request: the tradition is read and the two words nothing reads are
    # carried as clauses rather than dropped
    both = intent.read("Build a German village with a bakery and an aqueduct.")
    got = [r["id"] for r in both["requirements"]]
    assert "tradition/german" in got, got
    assert "clause/bakery" in got and "clause/aqueduct" in got, got
    for r in both["requirements"]:
        if r["kind"] == "clause":
            assert r["status"] == "unresolved" and r["hard"], r
    # ...and a tradition this library has no form family for is `unsupported`, not
    # approximated by the nearest thing on disk
    moor = intent.read("Build a Moorish town.")
    row = next(r for r in moor["requirements"] if r["id"] == "tradition/moorish")
    assert row["status"] == "unsupported", row
    # the independent check does not fire on the words a sentence spends on grammar
    plain = intent.read("Build a walled village of about twenty houses.")
    assert not [r for r in plain["requirements"] if r["kind"] == "clause"], \
        [r["id"] for r in plain["requirements"]]
    # a tradition is never `satisfied` by the nearest form family
    spec = _spec()
    spec["form"] = "east_asian"
    checked, found = intent.coverage(trad, spec)
    row = next(r for r in checked["requirements"] if r["id"] == "tradition/japanese")
    assert row["status"] == "unresolved" and row["owner"] == "capability", row
    assert "nearest form family" in row["why"], row["why"]
    # ...and the wrong form family is a failure routed to the reading
    spec2 = _spec()
    spec2["form"] = "european_vernacular"
    checked2, _ = intent.coverage(trad, spec2)
    row2 = next(r for r in checked2["requirements"] if r["id"] == "tradition/japanese")
    assert row2["status"] == "failed" and row2["owner"] == "reading", row2
    # an identity with no sourced claim is unresolved and says what is missing
    checked3, found3 = intent.coverage(named, _spec(), reading=contracts.make(
        "reading", sentence="Build Ringed City.", classification="named_reference"))
    row3 = next(r for r in checked3["requirements"] if r["id"].startswith("identity/"))
    assert row3["status"] == "unresolved" and row3["owner"] == "reading", row3
    assert any(f["requirement"] == row3["id"] and f["blocks"] == "fidelity"
               for f in found3["findings"]), found3["findings"]
    return ("a named city, a tradition and two unread clauses each become a hard "
            "requirement; none of the four sentences holds before it is planned; a "
            "tradition with no form family is unsupported and one with a nearest family "
            "is unresolved, never satisfied")


# ------------------------------- 12. the check that read a declaration

@case
def t_12_an_absence_is_checked_against_the_leaves_and_a_frontage_against_the_doors():
    """Both halves of the review's second finding, as refusals.

        A spec declaring nothing and a plan holding a wall passed an `unwalled` request,
        because the absence check enumerated the families the *spec* declared. And a
        frontage requirement passed on `resolution.bounds.faces == "water"` -- a word --
        while nine of the place's fifteen homes had their doors in the wall away from it.
        
    """
    empty = {"defining_parts": [], "sentence": "Build an unwalled village."}
    plan = _plan_of([{"name": "unexpected_wall", "kind": "edge", "type": "wall",
                      "path": [[0, 0], [20, 0]], "width": 3}])
    checked, found = intent.coverage(intent.read("Build an unwalled village."), empty,
                                     plan)
    row = next(r for r in checked["requirements"] if r["id"] == "absent/wall")
    assert row["status"] == "failed" and row["owner"] == "layout", row
    assert any(f["requirement"] == "absent/wall" for f in found["findings"])
    # ...and a plan with no wall in it still passes, on the same rule
    ok_plan = _plan_of([{"name": "green", "kind": "area", "type": "garden",
                         "x0": 0, "z0": 0, "x1": 8, "z1": 8}])
    checked2, _ = intent.coverage(intent.read("Build an unwalled village."), empty,
                                  ok_plan)
    assert next(r for r in checked2["requirements"]
                if r["id"] == "absent/wall")["status"] == "satisfied"
    # the label alone no longer certifies a frontage
    facing = intent.read("Build homes facing the water.")
    one = _plan_of([{"name": "home", "kind": "plot", "type": "cottage", "front": "west",
                     "x0": 0, "z0": 0, "x1": 8, "z1": 8}])
    checked3, _ = intent.coverage(facing, empty, one,
                                  resolution={"bounds": {"faces": "water"}})
    row3 = next(r for r in checked3["requirements"] if r["id"] == "facing/water")
    assert row3["status"] == "unresolved", row3
    assert "no shore path" in row3["why"], row3["why"]
    # ...and with a shore path it is the doors that answer
    anchor = {"kind": "shoreline", "along": "x", "water_side": "north",
              "path": [[0, 0], [8, 0], [16, 0]]}
    res = {"bounds": {"faces": "water", "anchor": anchor}}
    away = _plan_of([{"name": "h1", "kind": "plot", "type": "cottage", "front": "south",
                      "x0": 0, "z0": 20, "x1": 8, "z1": 28}])
    checked4, found4 = intent.coverage(facing, empty, away, resolution=res)
    row4 = next(r for r in checked4["requirements"] if r["id"] == "facing/water")
    assert row4["status"] == "failed" and row4["owner"] == "layout", row4
    assert any(f["owner"] == "layout" and f["requirement"] == "facing/water"
               for f in found4["findings"])
    toward = _plan_of([{"name": "h1", "kind": "plot", "type": "cottage",
                        "front": "north", "x0": 0, "z0": 20, "x1": 8, "z1": 28}])
    checked5, _ = intent.coverage(facing, empty, toward, resolution=res)
    assert next(r for r in checked5["requirements"]
                if r["id"] == "facing/water")["status"] == "satisfied"
    # the water's direction is **local**: the same front is right on one arm of a bay
    # and wrong on the other
    bay = {"kind": "shoreline", "along": "x", "water_side": "north",
           "path": [[0, 0], [20, 0], [40, 20], [40, 40]]}
    assert placeshore.water_direction(bay, 10, 20) == "north", "north arm"
    assert placeshore.water_direction(bay, 60, 40) == "west", "east arm"
    # a policy name is not geometry either
    lay = intent.read("Build a village following the shoreline.")
    checked6, _ = intent.coverage(lay, empty, one,
                                  resolution={"policy": "shoreline", "regions": []})
    row6 = next(r for r in checked6["requirements"] if r["id"] == "layout/shoreline")
    assert row6["status"] == "failed" and "does not carry it" in row6["why"], row6
    return ("an undeclared wall fails an `unwalled` request against the plan's own "
            "leaves; a frontage label certifies nothing and a door pointing away from "
            "the water beside it fails; the water's direction is read locally from the "
            "shore path; a `shoreline` policy with no shore path fails")


# --------------------------- 13. the capability that did not ask about ground

@case
def t_13_a_capability_asks_about_attachment_ground_and_relief_and_follows_the_place():
    c = capability.card("cottage", {"kind": "plot", "form": None, "role": None,
                                    "attached": False, "declares_needs": True,
                                    "needs": {"footprint": [5, 5, 20, 20],
                                              "ground": ["dry"], "max_relief": 0}})
    assert c["envelope"]["ground"] == ["dry"], c["envelope"]
    want = {"name": "attached wet slope home", "kind": "plot", "attached": True,
            "ground": "wet", "max_relief": 10, "footprint": [10, 10]}
    ok, why = capability.fits(c, want)
    assert not ok and "party walls" in why, (ok, why)
    ok, why = capability.fits(c, {**want, "attached": False})
    assert not ok and "wet" in why, why
    ok, why = capability.fits(c, {**want, "attached": False, "ground": None})
    assert not ok and "relief" in why, why
    assert capability.fits(c, {**want, "attached": False, "ground": None,
                               "max_relief": None})[0]
    # every committed type says it stands on `any` ground, and `any` is a wildcard and
    # not three letters
    _t, decls = placeplan.types_card(None, forms=[None])
    for n, card in capability.cards(decls).items():
        assert card["envelope"]["ground"] in ([], ["any"]), (n, card["envelope"])
    # **the record follows the place.** A round boundary needs a wall that draws a
    # diagonal run, and the ring layout builds `great_wall` for one: matching has to
    # reach the same answer, or the record describes a place nobody built.
    spec = _spec()
    plain = capability.match(spec)
    assert next(e for e in plain["entries"]
                if e["id"] == "cap/village_wall")["type"] == "wall"
    round_ = capability.match(spec, round_boundaries=True)
    assert next(e for e in round_["entries"]
                if e["id"] == "cap/village_wall")["type"] == "great_wall"
    # ...and the fabric want's lead type is the one the compiler reaches for first
    from ethoslm import district_compile
    _t2, d2 = placeplan.types_card(None, spec.get("form"))
    lead = [n for n, _d in district_compile.house_types(d2, "urban", spec.get("form"))]
    fabric = next(e for e in plain["entries"] if e["id"].endswith("/fabric"))
    assert fabric["type"] == lead[0], (fabric["type"], lead)
    assert fabric["alternatives"] == lead[1:len(fabric["alternatives"]) + 1], fabric
    # a place built out of a type the want refuses is a finding, not a silence
    place = {"parts": [{"name": "village_wall", "defines": "village_wall",
                        "type": "gate_tower", "kind": "edge"}], "compounds": []}
    _rec, rows = capability.agreements(plain, place, d2)
    assert rows and rows[0]["owner"] == "capability", rows
    assert rows[0]["blocks"] == "feasibility", rows[0]
    # ...and one the want admits is reconciled onto the record instead
    place2 = {"parts": [{"name": "village_wall", "defines": "village_wall",
                         "type": "great_wall", "kind": "edge"}], "compounds": []}
    rec2, rows2 = capability.agreements(plain, place2, d2)
    assert not rows2, rows2
    assert next(e for e in rec2["entries"]
                if e["id"] == "cap/village_wall")["type"] == "great_wall"
    return ("attachment, ground and relief each refuse a type by name; every committed "
            "type's ground reads as `any` and not as three letters; matching picks "
            "`wall` for a square place and `great_wall` for a round one, as the layout "
            "does; a type the want refuses is a `capability` finding")


# ------------------------- 14. the identities that were sizes

@case
def t_14_an_artifact_is_identified_by_its_content_and_an_invalidation_is_remembered():
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _tmp_round(tmp, WALLED_VILLAGE,
                         spec_mod.read_spec(copy.deepcopy(VILLAGE_SPEC),
                                            WALLED_VILLAGE))
        contracts.save(rnd, "intent", intent.read(WALLED_VILLAGE))
        # **a nested leaf is part of the plan.** A plan is a tree and the fingerprint
        # read its first level: moving a plot two levels down from 10 columns to 50 left
        # it unchanged.
        nested = {"parts": [{"name": "quarter", "kind": "district", "children": [
            {"name": "home", "kind": "plot", "type": "cottage",
             "x0": 0, "z0": 0, "x1": 10, "z1": 10}]}]}
        moved = copy.deepcopy(nested)
        moved["parts"][0]["children"][0]["x1"] = 50
        assert deps.fingerprint(rnd, ("plan",), plan=nested) != \
            deps.fingerprint(rnd, ("plan",), plan=moved), "a nested lot moved unseen"
        # **a type is its bytes.** `HEIGHT = 10` and `HEIGHT = 90` are the same length.
        root = os.path.join(tmp, "library")
        os.makedirs(os.path.join(root, "types"))
        f = os.path.join(root, "types", "home.py")
        open(f, "w").write("HEIGHT = 10\n")
        with mock.patch.object(pipeline, "ROOT", root):
            before = deps.fingerprint(rnd, ("types",))
            open(f, "w").write("HEIGHT = 90\n")
            after = deps.fingerprint(rnd, ("types",))
        assert before != after, (before, after)
        # **an output that is there is not an output.** A truncated file is what an
        # interrupted run leaves and it used to read as warm.
        json.dump(nested, open(rnd.rel("plan.json"), "w"))
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        assert deps.check(rnd, "plan", plan=nested)[0]
        open(rnd.rel("plan.json"), "w").write("")
        fresh, why = deps.check(rnd, "plan", plan=nested)
        assert not fresh and "zero bytes" in why, why
        open(rnd.rel("plan.json"), "w").write("{not json")
        fresh, why = deps.check(rnd, "plan", plan=nested)
        assert not fresh and "JSON" in why, why
        # **an invalidation is remembered.** Dropping a stamp used to make the artifact
        # qualify as a document from before the contract -- so invalidating it made it
        # reusable.
        json.dump(nested, open(rnd.rel("plan.json"), "w"))
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        assert not deps.legacy(rnd, "plan")
        deps.invalidate(rnd, "plan", "a test")
        assert not deps.legacy(rnd, "plan"), "an invalidated artifact read as legacy"
        fresh, why = deps.check(rnd, "plan")
        assert not fresh and "invalidated" in why, why
        # ...and an unstamped artifact is **not** a legacy fixture until the round says
        # its documents were imported. The unification round: a missing stamp and a
        # crashed stage look identical, so silence stopped meaning "adopt this".
        assert not deps.legacy(rnd, "preview"), "silence read as an import"
        deps.import_legacy(rnd.state, "a shipped fixture", ["preview/preview.json"])
        assert deps.legacy(rnd, "preview"), deps.imported(rnd)
        # making it again clears the tombstone
        deps.stamp(rnd, "plan", outputs=["plan.json"])
        assert deps.check(rnd, "plan", plan=nested)[0]
    return ("a nested lot, a same-length type file and a truncated output each move or "
            "refuse a fingerprint; an invalidated artifact is not a legacy one, an "
            "unstamped one is a fixture only once the round says it was imported, and "
            "making it again clears the tombstone")


# --------------------------- 15. the re-entry that was a sentence

@case
def t_15_the_driver_executes_a_re_entry_and_a_rollback_restores_the_candidate():
    from ethoslm.pipeline import round as driver, stages_media
    visits = []

    def preview(*_a):
        visits.append("preview")
        if len(visits) < 3:
            # exactly the shape production returns after a repair changes the plan
            return {"preview": {"done": False}, "status": "reenter",
                    "why": "a repair changed a planning decision"}
        return {"preview": {"done": True}}

    def parts(*_a):
        visits.append("parts")
        return {"reached": True}

    with tempfile.TemporaryDirectory() as tmp:
        rnd = pipeline.Round(name="reentry", state_dir=tmp)
        with mock.patch.dict(pipeline.STAGES,
                             {"t_preview": preview, "t_parts": parts}), \
                mock.patch("ethoslm.model.router",
                           return_value=SimpleNamespace(fulfil=lambda _r: 0)), \
                mock.patch.object(driver, "record"), \
                mock.patch.object(driver, "stage_report", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            got = driver.run(rnd, stages=("t_preview", "t_parts"), backend=object())
        assert visits == ["preview", "preview", "preview", "parts"], visits
        assert len(got["t_preview"]["reentries"]) == 2, got["t_preview"]

    # **the whole candidate is put back, not the plan files.** A rejected repair used to
    # leave its own `resolution.json` and `findings.json` beside the restored plan.
    with tempfile.TemporaryDirectory() as tmp:
        rnd = pipeline.Round(name="rollback", state_dir=tmp)
        os.makedirs(rnd.state, exist_ok=True)
        for f, doc in (("plan.json", {"candidate": "original"}),
                       ("findings.json", {"candidate": "original"}),
                       ("resolution.json", {"candidate": "original"}),
                       ("capabilities.json", {"candidate": "original"}),
                       ("deps.json", {"plan": {"candidate": "original"}})):
            json.dump(doc, open(rnd.rel(f), "w"))
        os.makedirs(rnd.rel("preview"), exist_ok=True)
        open(rnd.rel("preview", "reading.md"), "w").write("the original reading")
        snap = stages_media._snapshot(rnd)
        for f in ("plan.json", "findings.json", "resolution.json", "capabilities.json",
                  "deps.json"):
            json.dump({"candidate": "rejected"}, open(rnd.rel(f), "w"))
        open(rnd.rel("preview", "reading_1.md"), "w").write("the rejected reading")
        stages_media._restore(rnd, snap)
        for f in ("plan.json", "findings.json", "resolution.json", "capabilities.json"):
            assert json.load(open(rnd.rel(f))) == {"candidate": "original"}, f
        assert json.load(open(rnd.rel("deps.json"))) == {"plan": {"candidate": "original"}}
        assert not os.path.exists(rnd.rel("preview", "reading_1.md")), \
            "the rejected candidate's reading survived the rollback"
        assert os.path.exists(rnd.rel("preview", "reading.md"))
    return ("the driver re-enters a stage that asks to be re-entered and records every "
            "pass; a rollback restores the design records, the dependency stamps and "
            "the preview as well as the plan")


# ------------------- 16. the frontage obligation the compiler obeys

@case
def t_16_a_district_told_which_way_to_face_lays_its_lots_that_way():
    """The spatial repair's mechanism, on one district and no whole run.

        A ribbon district's streets run along the shore either way; which side of a street
        a door is on is a separate decision, and before this nothing made it. `faces` is
        that decision and `district_compile` is where it lands.
        
    """
    from ethoslm import district_compile
    spec = spec_mod.read_spec(copy.deepcopy(SHORE_SPEC), SHORE_SENTENCE)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    site = {"origin": [0, 0], "size": 192}
    vol = shore_volume()
    caps = capability.match(spec)
    place, fails = placesolve.solve_place(spec, site, None, decls,
                                          "drystone_and_thatch", vol=vol, caps=caps)
    assert not fails, fails
    anchor = dict(place["layout"]["anchor"])
    anchor["path"] = place["layout"]["anchor_path"]
    # the solver read `facing the water` off the sentence and wrote a side on every
    # district, from the shore path beside each of them
    assert place["districts"], "no district was laid"
    assert all(d.get("faces") for d in place["districts"]), \
        [d.get("faces") for d in place["districts"]]
    # the shore beside each district travels with it, so the compiler can ask which way
    # the water lies from **each block**: a district is a rectangle and a shore is not,
    # and one direction per district put two of the shoreline village's nine homes the
    # wrong way round on the site the search actually chose.
    assert all((d.get("faces_anchor") or {}).get("path") for d in place["districts"])
    by = {p["name"]: p for p in spec["defining_parts"]}
    one = place["districts"][0]
    got, _rec = district_compile.compile_district(one, by[one["defines"]], place,
                                                  decls, spec=spec, seed=1)
    lots = [l for q in got["quarters"] for l in q.get("plots", [])
            if l.get("kind") == "plot"]
    assert lots, "the district compiled no lots"
    faced = placeshore.fronts_facing_water(anchor, lots)
    assert faced["away"] == 0 and faced["unknown"] == 0, faced
    assert faced["toward"] == len(lots), (faced, len(lots))
    # ...and with the obligation removed the same district lays lots whose facing
    # nothing decided, which is the state the saved village was in
    bare = {k: v for k, v in one.items() if k not in ("faces", "faces_anchor")}
    got2, _r2 = district_compile.compile_district(bare, by[one["defines"]], place,
                                                  decls, spec=spec, seed=1)
    lots2 = [l for q in got2["quarters"] for l in q.get("plots", [])
             if l.get("kind") == "plot"]
    faced2 = placeshore.fronts_facing_water(anchor, lots2)
    assert faced2["toward"] == 0, faced2
    return (f"a district carrying `faces` lays all {faced['toward']} of its lots "
            f"fronting the water beside it and none away from it; the same district "
            f"without the obligation records no facing at all")


# ------------------ 17. the research answer that outlived its question

@case
def t_17_a_changed_sentence_does_not_keep_the_old_sentences_research():
    """The review's fifth finding, last part, and the most quietly wrong of them.

        **And so is a terminal agent's.** The unification round: with no retrieval provider
        the stage now asks the agent driving the round to do the research rather than
        completing an empty reading, so there is a second answer on disk that could outlive
        its question. It is keyed the same way (`evidence/asked.json`) and set aside by the
        same rule, which is what the second half of this case checks.
        
    """
    from ethoslm.pipeline import stages_plan
    with tempfile.TemporaryDirectory() as tmp:
        rnd = pipeline.Round(name="reading_fixture", sentence="Build Ringed City.",
                             state_dir=tmp)
        os.makedirs(rnd.state, exist_ok=True)
        json.dump({"classification": {"kind": "named_reference", "name": "Ringed City",
                                      "why": "a named city"},
                   "queries": ["Ringed City city layout"], "sources": [],
                   "provider": "none"},
                  open(rnd.rel("gathering.json"), "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            # a named place with no provider is a research job, not an empty success
            ask = stages_plan.stage_reading(rnd, None, {})["reading"]
            assert ask["status"] == "needs_model" and ask["role"] == "research", ask
            assert contracts.load(rnd, "reading") is None, "an unanswered ask read"
            # the agent answers it: honestly, having found nothing
            json.dump({"sources": [], "note": "no source was reachable offline"},
                      open(ask["write"], "w"))
            stages_plan.stage_reading(rnd, None, {})
            first = contracts.load(rnd, "reading")
            rnd.sentence = "Build an unwalled shoreline village."
            stages_plan.stage_reading(rnd, None, {})
        rec = contracts.load(rnd, "reading")
        assert first["classification"] == "named_reference", first
        assert rec["sentence"] == "Build an unwalled shoreline village.", rec["sentence"]
        assert rec["classification"] == "self_contained", rec["classification"]
        assert rec["queries"] == [], rec["queries"]
        assert not any("Ringed City" in str(x) for x in rec["inferred"]), rec["inferred"]
        # ...and the answer to the old question is set aside rather than deleted -- the
        # gathering, and the research the agent wrote for that sentence with it
        assert os.path.exists(rnd.rel("gathering.stale.json"))
        assert os.path.exists(rnd.rel("evidence/stale.sources.json")), \
            sorted(os.listdir(rnd.rel("evidence")))
        assert not os.path.exists(rnd.rel("evidence/sources.json"))
        # the intent record moved with it, and the old requirements are gone
        it = contracts.load(rnd, "intent")
        ids = [r["id"] for r in it["requirements"]]
        assert "absent/wall" in ids and "layout/shoreline" in ids, ids
        assert not [i for i in ids if i.startswith("identity/")], ids
    return ("a changed sentence sets its predecessor's gathering aside by name, "
            "re-classifies the request, drops the old queries and the old named city, "
            "and rewrites the intent record; the old gathering is kept beside it")


def main() -> int:
    bad = skipped = 0
    for fn in CASES:
        name = fn.__name__[2:]
        try:
            print(f"ok   {name:66s} {fn()}")
        except Skip as e:
            skipped += 1
            print(f"skip {name:66s} {e}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:66s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:66s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad - skipped}/{len(CASES)} architecture cases pass"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
