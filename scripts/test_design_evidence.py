"""The design round's escape-route checks E1 and E4, through the real consumers.

    $PY scripts/test_design_evidence.py

Small inputs, seconds not minutes, each with a positive control. This is not a suite and
does not try to be one: it is the set of measurements that would go green if somebody
quietly undid the four things the round is for.

**E1 -- demand before size** (`ethoslm.demand`, `ethoslm.envelope`):

  1. a **required** second storey reaches the envelope and forces a lot that delivers it,
     or refuses honestly. The measurement that opened the round is the control:
     `lot_for("cottage", {"storeys": 2})` is `[5, 5]` and the same question asked with
     the feature is `[15, 17]`, and no production query goes through the first form;
  2. the **optional-choice control**: an inferred storey band may still be revised down
     (`demand.relax`), and a required one may not;
  3. an explicit `character.lot_width` is validated rather than trusted;
  4. a **changed generator cannot reuse an obsolete certificate**: editing a type file's
     bytes moves the envelope key, and a cache document of another version is ignored.

**E4 -- evidence and obligations** (`ethoslm.usable`, `ethoslm.obligation`):

  5. remove the equipment, or wall off the way to it, with the FUNCTION declaration and
     `emitted.features` intact: the functional check fails. The unmodified build is the
     positive control and passes;
  6. a reading that omits an open finding does not close it; an applied action that did
     not move the measure leaves the row open **and** leaves other actions admissible;
     a missing measurement does not close it; an emitted construction constraint is a
     row of the same ledger and is closed by the lot it asked for.

**E3's emitted-solid half** (`types/great_wall.occupied`), because it is a type file and
types are this worker's:

  7. the great wall's published occupied envelope covers what it actually emits, at the
     widths and heights the review's collision came from.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (construction, demand, envelope, obligation,            # noqa: E402
                   offline, placeplan, usable)
from ethoslm.buildlib import Builder                                        # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


#: Every demand below is resolved against the real declaration map.
_DECLS = None


def decls():
    global _DECLS
    if _DECLS is None:
        _DECLS = placeplan.types_card(forms=[None])[1]
    return _DECLS


def _part(name="houses", **over):
    got = {"name": name, "defines": name, "kind": "group", "family": "district",
           "density": "medium", "role": "rural", "fabric_types": ["cottage"],
           "character": {"storeys": [2, 2]}}
    got["character"] = {**got["character"], **(over.pop("character", None) or {})}
    return {**got, **over}


def _intent(*reqs):
    return {"requirements": list(reqs)}


TALL = {"id": "quality/height/tall", "kind": "quality", "hard": True,
        "wants": {"axis": "height", "value": "tall", "bound": 0.62}, "scope": None}

SPEC = {"form": None, "voice": None, "kind": "village"}


# ------------------------------------------------------------- E1. demand before size

@case
def t1_required_storey_reaches_the_envelope():
    """A required second storey forces a lot that delivers it. The round's opening
    measurement is the control: the same question with an empty feature set is answered
    by a 5x5 lot, and nothing in `demand` asks it that way."""
    bare = envelope.lot_for("cottage", {"storeys": 2})
    feat = envelope.lot_for("cottage", {"storeys": 2}, features=("storeys",))
    assert bare["lot_min"] == [5, 5], bare
    assert feat["lot_min"] and feat["lot_min"][0] >= 15, feat

    d = demand.resolve(SPEC, _intent(TALL), _part(), decls())
    assert d["required"] == ("storeys",), d
    assert d["requirements"] == ["quality/height/tall"], d
    assert d["params"]["storeys"] == 2, d
    assert d["pool_from"].startswith("the part's approved"), d
    assert demand.features_asked(d) == ("storeys",), d

    got = demand.lot(d)
    assert got["asked"] == ["storeys"], got
    assert got["lot_min"] == feat["lot_min"], (got, feat)
    assert got["refused"] is None and got["binding"] is True, got
    # ...and a lot the fabric would have drawn does not hold it
    bad = demand.validate_lot(d, (12, 10))
    assert bad["ok"] is False and bad["binding"] is True, bad
    return (f"required storeys 2 -> lot_min {got['lot_min']} (the featureless query "
            f"answers {bare['lot_min']}); a 12x10 lot is refused, binding")


@case
def t2_optional_band_revises_down_and_required_does_not():
    """The control the farm-low case is. An **inferred** band may be relaxed and a
    **required** floor may not, and the answer says which."""
    opt = demand.resolve(SPEC, _intent(), _part("low"), decls())
    assert opt["required"] == () and "storeys" in opt["optional"], opt
    high = demand.lot(opt)
    assert high["binding"] is False, high

    down = demand.relax(opt)
    assert down is not None and down["params"]["storeys"] == 1, down
    low = demand.lot(down)
    assert low["lot_min"] == [5, 5], low

    req = demand.resolve(SPEC, _intent(TALL), _part(), decls())
    assert demand.relax(req) is None, "a required floor was relaxed"
    return (f"inferred band {opt['storeys_band']} relaxes to 1 storey and "
            f"{high['lot_min']} -> {low['lot_min']}; the required floor will not relax")


@case
def t3_a_lot_that_cannot_deliver_refuses_rather_than_shrinking():
    """A demand nothing in the approved pool answers is `lot_min: None` with `refused`
    set. Nothing substitutes a smaller default, and the reason names the requirement."""
    forge = {"id": "feature/forge", "kind": "feature", "hard": True,
             "wants": {"feature": "smithy", "family": "workshop", "count": 1},
             "scope": None}
    smithy = {"name": "smithy", "defines": "smithy", "kind": "plot",
              "family": "workshop", "count": 1, "structures": 0, "role": "urban",
              "fabric_types": ["market"]}
    # `market` declares FEATURES = ("stalls", "aisle"), so its vocabulary is stated and
    # a forge is outside it: a capability gap, answered without a probe
    d = demand.resolve(SPEC, _intent(forge), smithy, decls())
    assert d["required"] == ("forge",) and d["unsupported"] == ("forge",), d
    got = demand.lot(d)
    assert got["lot_min"] is None and got["lot_pref"] is None, got
    assert got["refused"] and got["binding"] is True, got
    assert "feature/forge" in got["why"], got
    return f"refused, lot_min None: {got['why'][:90]}"


@case
def t3b_an_unscoped_feature_does_not_reach_every_part():
    """The coordinator's measured defect, as a regression check. `feature/market` has no
    scope and its word maps to `stalls`, and `envelope.required_by` filtered on scope
    alone -- so every cottage, every field and the hall of a farming village were asked
    to emit market stalls and `demand.lot` refused all four for a capability gap that
    does not exist. `demand.resolve` binds requirements to parts through
    `capability.requirements_for`, which asks whether the part is the one that answers
    the clause."""
    market = {"id": "feature/market", "kind": "feature", "hard": True,
              "wants": {"feature": "market", "family": "square", "count": 1},
              "scope": None}
    storeys = {"id": "quality/height/storeys/2", "kind": "quality", "hard": True,
               "wants": {"axis": "height", "value": "storeys", "storeys": 2,
                         "exact": True}, "scope": None}
    homes = _part("homes", count=1, structures=16, density="low",
                  character={"storeys": [2, 2]})
    d = demand.resolve(SPEC, _intent(market, storeys), homes, decls())
    assert "stalls" not in d["required"], d
    assert d["required"] == ("storeys",), d
    assert "feature/market" not in d["requirements"], d
    # ...and an exact requirement wins over the character's band, which is an inference
    loose = _part("homes", count=1, structures=16, density="low",
                  character={"storeys": [1, 2]})
    d2 = demand.resolve(SPEC, _intent(storeys), loose, decls())
    assert d2["params"]["storeys"] == 2 and d2["storeys_band"] == [2, 2], d2
    assert d2["fixed"] == {"storeys": 2}, d2
    assert demand.relax(d2) is None, "a requirement-fixed floor was relaxed"
    got = demand.lot(d)
    assert got["lot_min"] and got["lot_min"][0] >= 15, got
    return (f"an unscoped market does not reach the cottages (required "
            f"{d['required']}); an exact storeys requirement beats a [1, 2] band and "
            f"the lot is {got['lot_min']}")


@case
def t4_storeys_admitted_never_answers_below_a_required_floor():
    """`demand.storeys_admitted` replaces `district_compile._storeys_fit`. Where the
    required floor does not fit it answers None with `holds: False`, so the ask stands
    and the shortfall becomes a constraint rather than a quietly lowered parameter."""
    d = demand.resolve(SPEC, _intent(TALL), _part(), decls())
    small = demand.storeys_admitted(d, 5, 5)
    big = demand.storeys_admitted(d, 18, 18)
    assert small["storeys"] is None and small["holds"] is False, small
    assert small["required"] is True and small["asked"] == 2, small
    assert big["storeys"] == 2 and big["holds"] is True, big
    return (f"a 5x5 lot admits {small['storeys']} (holds {small['holds']}, required "
            f"{small['required']}); an 18x18 admits {big['storeys']}")


@case
def t5_a_changed_generator_cannot_answer_from_an_old_certificate():
    """The envelope key carries the type file's content digest, the builder modules'
    and the voice's. An edited generator is a different question, and a cache document
    written under another version is ignored rather than trusted."""
    p = os.path.join(ROOT, "types", "cottage.py")
    was = envelope._key("cottage", {"storeys": 2}, ("storeys",), None, 1, None)
    src = open(p).read()
    try:
        open(p, "w").write(src + "\n# a comment that changes nothing but the bytes\n")
        envelope._DEPS.clear()
        now = envelope._key("cottage", {"storeys": 2}, ("storeys",), None, 1, None)
    finally:
        open(p, "w").write(src)
        envelope._DEPS.clear()
    assert now != was, "an edited type file answered to the same envelope key"
    back = envelope._key("cottage", {"storeys": 2}, ("storeys",), None, 1, None)
    assert back == was, "restoring the file did not restore the key"
    # context is keyed too: a flat-ground answer is not an answer about a slope
    assert envelope._key("cottage", {"storeys": 2}, ("storeys",), None, 1,
                         {"ground": "slope"}) != was

    # ...and a cache file of another version is read and dropped
    d = tempfile.mkdtemp(prefix="des-env-")
    try:
        old = os.path.join(d, "envelopes.json")
        json.dump({was: {"lot_min": [1, 1], "source": "a lie"}}, open(old, "w"))
        assert envelope._load_cache(old) == {}, "a pre-version cache was trusted"
        envelope._save_cache(old, {was: {"lot_min": [15, 17]}})
        doc = json.load(open(old))
        assert doc["version"] == envelope.CACHE_VERSION, doc
        assert envelope._load_cache(old)[was]["lot_min"] == [15, 17]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return (f"the key moved on an edited type file and came back; a cache of another "
            f"version answers nothing ({envelope.CACHE_VERSION})")


# --------------------------------------------------------- E4. the world, and the
# ledger

def _probe(type_name, w, d, params=None):
    b, sited, res = construction.probe_build(type_name, w, d, params or {}, seed=1)
    assert res.get("ok"), res
    got = construction.outcome(b, sited, None, params or {})
    part = {**sited, "name": type_name, "kind": "plot", "type": type_name,
            "emitted": got}
    return b, part, got


@case
def t6_a_declaration_does_not_survive_the_equipment_going_away():
    """The functional check, on the assembled world. The build is the positive control;
    the same build with its stalls taken out, and with its stalls walled off, fails --
    and `emitted.features` still says `stalls: True` in all three."""
    b, part, got = _probe("market", 24, 24)
    assert got["features"].get("stalls") is True, got["features"]
    assert got["rects"].get("stalls"), "the outcome carries no rectangle to look for"
    x0, z0, x1, z1 = got["rects"]["stalls"]
    fy = int(part["floor_y"])

    def ask(vol=None):
        w = usable.World.of_builder(b, part, volume=vol)
        return usable.check(w, "market", "equipment_reachable")

    control = ask()
    assert control["holds"] is True and control["method"] == "observed", control

    gone = usable.World.assembled(b)
    gone.overlay({(x, y, z): "air" for x in range(x0, x1 + 1)
                  for z in range(z0, z1 + 1) for y in range(fy + 1, fy + 9)})
    a_gone = ask(gone)

    shut = usable.World.assembled(b)
    shut.overlay({(x, y, z): "stone"
                  for x in range(x0 - usable.REACH - 1, x1 + usable.REACH + 2)
                  for z in range(z0 - usable.REACH - 1, z1 + usable.REACH + 2)
                  for y in range(fy + 1, fy + 7)
                  if not (x0 <= x <= x1 and z0 <= z <= z1)})
    a_shut = ask(shut)

    assert a_gone["holds"] is False and a_gone["evidence"]["gone"] == ["stalls"], a_gone
    assert a_shut["holds"] is False and a_shut["evidence"]["unreachable"] == ["stalls"], \
        a_shut
    assert got["features"]["stalls"] is True, "the declaration was edited by the check"
    # ...and the answer can never be `holds` on a declaration
    try:
        usable.answer(True, "declared", "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a declaration established a predicate")
    return ("the unmodified market holds; equipment removed -> gone; equipment walled "
            "off -> unreachable; the FUNCTION declaration is untouched throughout")


@case
def t6b_a_feature_in_four_places_is_four_rectangles():
    """The `des-farm` defect. `types/square.py` published `rects.stalls` as the bounding
    box of its four corner booths -- on a 40x40 square, a 38x38 rectangle with the whole
    open market inside it -- and `construction._verify_rect` asks whether half of a
    claimed rectangle carries something. Four corner booths gave it three per cent, so
    every market square this build has ever laid recorded `stalls: claimed_not_found`
    however well they stood, and a square whose booths a neighbour really had razed read
    exactly the same as a perfect one."""
    b, part, got = _probe("square", 40, 40, {"paving": "banded", "canopy": "gable"})
    rects = got["rects"]["stalls"]
    assert isinstance(rects[0], list) and len(rects) == 4, rects
    assert got["features"]["stalls"] is True, got["features_source"]
    assert got["features_source"]["stalls"] == "verified", got["features_source"]
    assert got["features_rects"]["stalls"] == [4, 4], got["features_rects"]
    # the bounding box the old record published covers almost the whole square
    bbox = [min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects)]
    assert (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) > 20 * sum(
        (r[2] - r[0] + 1) * (r[3] - r[1] + 1) for r in rects), bbox

    fy = int(part["floor_y"])

    def ask(vol=None):
        w = usable.World.of_builder(b, part, volume=vol)
        return usable.check(w, "square", "equipment_reachable")

    control = ask()
    assert control["holds"] is True, control
    assert control["evidence"]["features"][0]["standing"] == 4, control["evidence"]
    # ...and one booth razed, as a neighbour's tree-clearing pass razed two on the farm
    razed = usable.World.assembled(b)
    r = rects[0]
    razed.overlay({(x, y, z): "air" for x in range(r[0], r[2] + 1)
                   for z in range(r[1], r[3] + 1) for y in range(fy + 1, fy + 7)})
    gone = ask(razed)
    assert gone["holds"] is False, gone
    assert gone["evidence"]["features"][0]["standing"] == 3, gone["evidence"]
    return (f"four booths are four rectangles ({sum((x[2]-x[0]+1)*(x[3]-x[1]+1) for x in rects)} "
            f"columns, against {(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)} for the box round "
            f"them); 4 of 4 stand and the type's claim verifies; raze one and it is 3 of 4 "
            f"and the claim fails")


@case
def t6c_a_feature_is_not_certified_by_the_floor_above_it():
    """The `des-farm` place read's defect. `construction._verify_rect` asked
    `y >= fy + 1` with no ceiling, so a 1x1 `rects.hearth` was answered `verified` by the
    **second storey's floor** four courses up the hearth's own column -- and seven of the
    farm's seventeen cottages certified a fire that is not in the world. The emission
    check and `usable` now read the same window, and `types/cottage.py` publishes no
    rectangle for a fire that is not standing when the house is finished."""
    assert construction.FEATURE_COURSES == construction.STOREY_PITCH - 1
    with_fire = without = None
    for st in (1, 2):
        b, part, got = _probe("cottage", 20, 20, {"storeys": st})
        fire = [p for p, v in b._pending.items() if "campfire" in v]
        claim = got["features"].get("hearth")
        rect = got["rects"].get("hearth")
        assert bool(fire) == bool(claim) == bool(rect), (st, len(fire), claim, rect)
        if fire:
            with_fire = (b, part, got)
            assert got["features_source"]["hearth"] == "verified", got["features_source"]
        else:
            without = (b, part, got)
            assert "hearth" in got["omitted"] and got["fallback"], got
    # **A cottage without a fire is no longer the ordinary case, and that is the
    # point.** This case was written while `prims.fitting` reported success for a
    # campfire the flight-way rule had quietly taken back out again: eleven of thirty-
    # six probe cottages had no fire and every one of them certified `verified`. With
    # the window bounded the absence became visible, and with `fitting` reporting its
    # own defining cell the fires go in -- thirty-six of thirty-six. So the case asserts
    # what it is about (the window, and the honest answer for a part that claims
    # nothing) and not a population it would now have to break the build to produce.
    assert with_fire, "no probe cottage laid a fire: the fitting path is broken"
    b, part, got = with_fire
    a = usable.check(usable.World.of_builder(b, part), "cottage", "equipment_reachable")
    assert a["holds"] is True and a["method"] == "observed", a
    if without is not None:
        b2, part2, _g2 = without
        a2 = usable.check(usable.World.of_builder(b2, part2), "cottage",
                          "equipment_reachable")
        assert a2["holds"] is None and a2["method"] == "unsupported", a2
    else:
        # nothing claimed, nothing to reach: the honest answer, asked of a part whose
        # emission record carries no equipment at all
        bare = dict(part, emitted={"features": {}, "rects": {}})
        a2 = usable.check(usable.World.of_builder(b, bare), "cottage",
                          "equipment_reachable")
        assert a2["holds"] is None and a2["method"] == "unsupported", a2

    # ...and the floor above no longer certifies anything: a rect with nothing in its
    # own storey and a slab four courses up reads as not found
    cols = {(0, 0): {0: "stone", 4: "oak_planks"}}
    assert construction._verify_rect(cols, [0, 0, 0, 0], 0) is False
    cols[(0, 0)][1] = "campfire"
    assert construction._verify_rect(cols, [0, 0, 0, 0], 0) is True
    return ("a cottage with a fire verifies it and reads `observed`; one without claims "
            "no hearth, carries it in `omitted`, and reads `unsupported`; a block four "
            "courses above a rectangle no longer certifies it")


@case
def t7_the_six_predicates_decide_a_courtyard_house():
    """The positive control for the whole predicate set, and the two predicates only a
    court has. All six are `observed` or `inferred`, none `declared`."""
    b, part, got = _probe("courtyard_house", 26, 26, {"storeys": 1})
    w = usable.World.of_builder(b, part)
    ans = usable.features_for(w, "courtyard_house")
    assert set(ans) == set(usable.WANTS), sorted(ans)
    weak = [k for k, a in ans.items() if a["method"] in ("declared", "unsupported")]
    assert not weak, {k: ans[k]["why"] for k in weak}
    assert all(a["holds"] for a in ans.values()), usable.says(ans)
    court = ans["court_accessible"]["evidence"]["courts"][0]
    rel = ans["range_relation"]["evidence"]["courts"][0]
    assert rel["ranged"] >= usable.COURT_SIDES and rel["rooms_on_court"] > 0, rel
    return (f"all six hold on the built courtyard house: the court is {court['open']}/"
            f"{court['cells']} open and ranged on {rel['ranged']} of four sides by "
            f"{rel['rooms_on_court']} room(s)")


@case
def t8_an_omitted_finding_stays_open():
    """A reading that does not mention an open row moves `last_seen` and nothing else.
    An ineffective action leaves the row open and leaves other actions admissible. A
    missing measurement closes nothing."""
    d = tempfile.mkdtemp(prefix="des-ob-")
    here = "candidate-1"
    try:
        led = obligation.load(d)
        f = {"id": "find/square/generous", "about": "square", "owner": "layout",
             "says": "the square is thirteen times its cottages", "material": True,
             "target": {"measure": "square_ratio", "value": 4.0, "direction": "down"}}
        obligation.upsert(led, [obligation.from_finding(f, source="reading-1")],
                          "reading-1", candidate=here)
        assert [r["id"] for r in obligation.open_rows(led)] == ["find/square/generous"]

        # the next reading omits it
        obligation.upsert(led, [], "reading-2", candidate=here)
        row = led["rows"]["find/square/generous"]
        assert row["disposition"] == "open", row
        assert row["omitted_by"] == ["reading-2"], row
        assert row["last_seen"] >= row["first_seen"], row

        # an applied action that did not move the measure
        obligation.act(led, f["id"], "shrink_anchor", here, applied=True, why="tried")
        eff = obligation.effect(led, f["id"], here, {"square_ratio": 13.0},
                                {"square_ratio": 13.0})
        assert eff["moved"] is False, eff
        assert led["rows"][f["id"]]["disposition"] == "open"
        rest = obligation.admissible(led, f["id"], here,
                                     ("shrink_anchor", "resize_ring", "grow_land"))
        assert rest == ["resize_ring", "grow_land"], rest
        assert obligation.tried_here(led, f["id"], here, "shrink_anchor")

        # a missing measurement, a measurement of another candidate, and one that moved
        # toward the target without reaching it
        for ev, why in (({"candidate": here, "measures": {}}, "missing"),
                        ({"candidate": "candidate-2",
                          "measures": {"square_ratio": 3.0}}, "another candidate"),
                        ({"candidate": here, "before": 13.0,
                          "measures": {"square_ratio": 11.0}}, "not reached")):
            got = obligation.close(led, f["id"], ev, candidate=here)
            assert got["closed"] is False, (why, got)
        good = obligation.close(led, f["id"], {"candidate": here, "before": 13.0,
                                               "measures": {"square_ratio": 3.6}},
                                candidate=here)
        assert good["closed"] is True, good
        obligation.save(d, led)
        assert obligation.load(d)["rows"][f["id"]]["disposition"] == "closed"
        return ("an omitted finding stays open; an ineffective action leaves two "
                "alternatives admissible; a missing measure, another candidate's and "
                "one short of the target all refuse closure; 3.6 closes it")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def t9_an_emitted_constraint_is_a_row_of_the_same_ledger():
    """The fourth escape: a constraint was dispositioned into its own list and no action
    was ever selected from it. Here it is a row like any other, material because the
    request requires the token, closed by the lot it asked for and by nothing else."""
    d = tempfile.mkdtemp(prefix="des-ob2-")
    here = "candidate-1"
    try:
        led = obligation.load(d)
        c = {"what": "storeys", "part": "b1_2_00", "type": "cottage", "owner": "layout",
             "requested": 2, "emitted": 1,
             "needs": {"lot": [12, 10], "lot_min": [15, 17]},
             "why": "cottage asked for 2 storeys on a 12x10 lot and emitted 1"}
        rid = "constraint/storeys/b1_2_00"
        obligation.upsert(led, [obligation.from_constraint(c, required=("storeys",),
                                                           source="parts-1")],
                          "parts-1", candidate=here)
        row = led["rows"][rid]
        assert row["origin"] == "constraint" and row["material"] is True, row
        assert row["acceptance"]["lot_min"] == [15, 17], row
        assert obligation.open_rows(led, material=True), "not owed"
        # ...and a material row cannot be accepted away
        try:
            obligation.dispose(led, rid, "accepted", "never mind")
        except ValueError:
            pass
        else:
            raise AssertionError("a material obligation was accepted")
        short = obligation.close(led, rid, {"candidate": here, "lot": [12, 10],
                                            "measures": {"emitted.storeys": 1}},
                                 candidate=here)
        assert short["closed"] is False, short
        ok = obligation.close(led, rid, {"candidate": here, "lot": [16, 18],
                                         "measures": {"emitted.storeys": 2}},
                              candidate=here)
        assert ok["closed"] is True, ok
        # an optional row still needs somebody to decide
        obligation.upsert(led, [obligation.from_finding(
            {"id": "find/groups/fragmented", "about": "groups", "says": "three groups",
             "owner": "layout", "material": False}, source="reading-1")],
            "reading-1", candidate=here)
        assert [r["id"] for r in obligation.undisposed(led)] == ["find/groups/fragmented"]
        obligation.dispose(led, "find/groups/fragmented", "accepted", "a suggestion")
        assert not obligation.undisposed(led)
        return ("a constraint and a finding are one ledger; the constraint is material, "
                "refuses a 12x10 lot and closes on 16x18; an optional row is undisposed "
                "until somebody disposes of it")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- E3's emitted-solid half (a type
# file)

@case
def ta_the_great_wall_publishes_what_it_actually_fills():
    """The review's second finding: "the great wall's piers and batter extend beyond its
    nominal width". `types/great_wall.occupied` publishes the real extent; this builds
    the wall and checks the published envelope covers what was emitted."""
    ns = construction._type_ns("great_wall")
    occ = ns["occupied"]
    rows = []
    for width, H in ((3, 48), (5, 48), (8, 48)):
        vol = construction._flat_volume(140)
        b = Builder(offline.OfflineSite(vol))
        b._vol, b.frontage = vol, None
        part = {"label": "gw", "kind": "edge", "path": [[20, 70], [110, 70]],
                "width": width}
        b.registry = construction._OnePlot(dict(part))
        sited = b.site(dict(part))
        ns["build"](b.type_builder(sited, role=ns.get("ROLE")), sited, 1, height=H,
                    width=3, parapet="crenellated", face="masonry")
        zs = [z for (_x, _y, z) in b._pending]
        ys = [y for (_x, y, _z) in b._pending]
        band = [c[1] for s in sited["segments"] for c in s["cells"]]
        out, inn = min(band) - min(zs), max(zs) - max(band)
        g = occ(sited, height=H)
        assert out + inn + width <= g["total"], (width, H, out, inn, g)
        assert out <= g["outer"] + g["either"], (width, H, out, g)
        assert inn <= g["inner"] + g["either"], (width, H, inn, g)
        above = max(ys) - (sited["segments"][0]["floor_y"] + H)
        assert above <= g["above"], (width, H, above, g)
        rows.append(f"{width}x{H}: emitted {out + inn + width}, published {g['total']}")
    return "; ".join(rows)


def main() -> int:
    bad = 0
    t0 = time.perf_counter()
    for fn in CASES:
        name = fn.__name__.split("_", 1)[1].replace("_", " ")
        try:
            note = fn()
            print(f"ok   {name}: {note}")
        except Exception as e:                       # noqa: BLE001 -- a failing case
            import traceback
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} design evidence cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
