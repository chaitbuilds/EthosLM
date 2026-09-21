"""The composition round's false-pass routes, through the real consumers.

    $PY scripts/test_composition_evidence.py

Cheap production counterexamples, each with a positive control, in the style of
`scripts/test_design_evidence.py`. Not a suite: this is the set of measurements that
would go green again if somebody quietly undid the six things the round's evidence
connections are for.

**E1 -- a function needs affirmative evidence for its applicable required predicates**
(`construction.confirm`, `construction.evidence_for`, `construction.wants_for`):

  1. a required feature with no affirmative evidence cannot report observed success
     because another predicate ran. The old counting rule -- `intent._usable_verdict`
     incrementing `ran` for every answer that is not `unsupported` and calling the result
     `observed` -- is computed alongside and shown to disagree;
  2. an `unsupported` or `declared` answer stays `holds: None` and is reported as **owed**,
     with which of `OWED_REASONS` it is.

**E2 -- physical feature identity** (`usable.FEATURE_BLOCKS`, `construction._verify_feature`):

  3. a hearth rectangle full of the voice's own masonry does not certify a hearth, at
     emission and on the assembled world. The real fire is the control, and the control is
     what `types/cottage.py` actually lays.

**E3/E4 -- the ledger** (`ethoslm.obligation`):

  4. a finding that recurs on a new candidate reopens; a closed row on an unchanged
     candidate stays closed, and a pass that merely omits it does not reopen it;
  5. a measurement about a different subject cannot close a row; the row's own subject can;
  6. an `emitted.<feature>` row closes from post-build feature evidence and **not** from a
     view measure -- `inspect.MEASURES` carries no `emitted.*` key, which is why these rows
     could not be closed by any evidence that existed.

**E5 -- the per-part binding** (`demand.required_by_part`, `envelope.feature_token`):

  7. a `market` want and a `stalls` constraint agree about being required, and the binding
     asks the market leaf for stalls without asking the houses beside it.

**E6 -- prepared ground in reuse** (`ethoslm.deps`):

  8. a changed prepared ground invalidates the built artifact; an unchanged one stays warm.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (construction, demand, deps, envelope,                   # noqa: E402
                   obligation, prims, usable)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ----------------------------------------------------------------- shared probes

_BUILT: dict = {}


def built(type_name: str, w: int, d: int, params=None):
    """One probe-built type, cached: `(builder, part, outcome)`. The same build every
    case of this file asks about, so eight cases cost the builds of three types."""
    key = (type_name, w, d, tuple(sorted((params or {}).items())))
    got = _BUILT.get(key)
    if got is None:
        b, part, res = construction.probe_build(type_name, w, d, dict(params or {}),
                                               seed=1)
        assert res.get("ok"), (type_name, res)
        em = construction.outcome(b, part, None, dict(params or {}))
        got = _BUILT[key] = (b, {**part, "name": type_name, "kind": "plot",
                                 "type": type_name, "emitted": em}, em)
    return got


def record(part: dict) -> dict:
    """One part, as the `parts.json` the pipeline writes."""
    return {"waves": [{"parts": [{"part": part["name"], "type": part.get("type"),
                                  "kind": "plot", "status": "built", "stood": True,
                                  "x0": part.get("x0"), "z0": part.get("z0"),
                                  "x1": part.get("x1"), "z1": part.get("z1"),
                                  "floor_y": part.get("floor_y"),
                                  "footprint": part.get("footprint"),
                                  "emitted": json.loads(json.dumps(part["emitted"]))}]}]}


def old_verdict(rec: dict) -> dict:
    """`intent._usable_verdict`'s rule as it stood: **count the predicates that ran**.

        Reproduced here rather than imported, because the coordinator is fixing the original
        and the point of this file is to show what the rule it replaces would have said.
        
    """
    ran, failed = 0, []
    for w in rec.get("waves") or []:
        for r in w.get("parts") or []:
            for want, a in ((r.get("emitted") or {}).get("usable") or {}).items():
                if not isinstance(a, dict) or a.get("method") == "unsupported":
                    continue
                ran += 1
                if a.get("holds") is False:
                    failed.append((r.get("part"), want))
    return {"ran": ran, "failed": failed, "method": "observed" if ran else "declared"}


# --------------------------------------- E1. affirmative evidence, or an owed record

@case
def t1_a_required_feature_with_no_evidence_is_not_an_observed_success():
    """The round's first evidence connection. `construction.CONFIRM_WANTS` asked a fixed
    three predicates of every part, and `intent._usable_verdict` then counted every answer
    that was not `unsupported` as "a predicate that ran", labelled the result `observed`
    and let `intent._function_measure` report `satisfied` on that count. So a market whose
    stalls are not in the assembled world passed on the strength of its front door.

    Two subjects. A courtyard house required to hold a hearth: it is a complete building,
    every predicate asked of it holds, and the fire is not there. And a market square whose
    booths a neighbour razed, where the feature was in the emission record and is not in
    the world."""
    # a house required to hold a hearth, which this instance does not lay
    b, part, em = built("courtyard_house", 26, 26, {"storeys": 1})
    rec = record(part)
    got = construction.confirm(usable.World.of_builder(b, part), rec,
                               required={"courtyard_house": ("courtyard", "hearth")})
    answers = (rec["waves"][0]["parts"][0].get("emitted") or {}).get("usable") or {}
    was = old_verdict(rec)
    # **the old counting rule calls this a satisfied function.** Several predicates ran,
    # none of them failed, and the label on the result was `observed`.
    assert was["ran"] >= 3 and was["failed"] == [] and was["method"] == "observed", was
    assert answers["entrance_connected"]["holds"] is True
    assert answers["equipment_reachable"]["holds"] is True
    # ...and the required hearth has no affirmative evidence of any kind
    assert [o["feature"] for o in got["owed"]] == ["hearth"], got["owed"]
    owed = got["owed"][0]
    assert owed["reason"] in construction.OWED_REASONS, owed
    assert owed["holds"] is not True and owed["method"] != "observed" or \
        owed["holds"] is False, owed
    court = construction.evidence_for(rec, "courtyard_house", "courtyard")
    assert court["holds"] is True and court["owed"] is False, court
    hearth = construction.evidence_for(rec, "courtyard_house", "hearth")
    assert hearth["owed"] is True and hearth["want"] == "equipment_reachable", hearth

    # a market square whose booths are gone from the assembled world
    b2, part2, em2 = built("square", 40, 40, {"paving": "banded", "canopy": "gable"})
    rects = construction.rects_of(em2["rects"]["stalls"])
    fy = int(part2["floor_y"])
    good = record(part2)
    ok = construction.confirm(usable.World.of_builder(b2, part2), good,
                              required={"square": ("stalls",)})
    assert ok["owed"] == [], ok["owed"]
    a = construction.evidence_for(good, "square", "stalls")
    assert a["holds"] is True and a["method"] == "observed" and not a["owed"], a

    razed = usable.World.assembled(b2)
    for r in rects:
        razed.overlay({(x, y, z): "air" for x in range(r[0], r[2] + 1)
                       for z in range(r[1], r[3] + 1) for y in range(fy + 1, fy + 9)})
    bad = record(part2)
    gone = construction.confirm(usable.World.of_builder(b2, part2, volume=razed), bad,
                                required={"square": ("stalls",)})
    row = bad["waves"][0]["parts"][0]
    assert [o["feature"] for o in gone["owed"]] == ["stalls"], gone["owed"]
    assert gone["owed"][0]["reason"] == "not_in_world", gone["owed"]
    assert row["emitted"]["features"]["stalls"] is False, row["emitted"]
    assert row["emitted"]["features_source"]["stalls"] == "not_in_world"
    assert "stalls" in row["emitted"]["omitted"], row["emitted"]
    return (f"the courtyard house answers {was['ran']} predicate(s), fails none and is "
            f"`observed` by the counting rule, while the hearth it is required to hold is "
            f"owed by `{owed['reason']}`; the razed square owes stalls by "
            f"`{gone['owed'][0]['reason']}` and the intact one owes nothing")


@case
def t2_an_unknown_answer_stays_unresolved_and_is_owed():
    """`usable` answers `holds: None` wherever the question cannot be decided, and the
    two ways that happens are a feature declared with no rectangle to look for and a part
    that is not standing at all. Neither may become a satisfied requirement, and neither
    may disappear either: an unresolved answer is **owed**, with its reason."""
    b, part, em = built("cottage", 20, 20, {"storeys": 1})
    world = usable.World.of_builder(b, part)

    # a declaration with no rectangle
    said = json.loads(json.dumps(em))
    said["features"]["hearth"] = True
    said["rects"].pop("hearth", None)
    rec = record({**part, "emitted": said})
    got = construction.confirm(world, rec, required={"cottage": ("hearth",)})
    owed = got["owed"]
    assert [o["feature"] for o in owed] == ["hearth"], owed
    assert owed[0]["holds"] is None and owed[0]["reason"] == "declared", owed
    assert rec["waves"][0]["parts"][0]["emitted"]["usable"][
        "equipment_reachable"]["method"] == "declared"

    # a part that is not standing: nothing about its use can be observed
    down = record(part)
    down["waves"][0]["parts"][0].update(stood=False, status="failed")
    world2 = usable.World(world.ctx, [down["waves"][0]["parts"][0]],
                          registry=world.registry)
    got2 = construction.confirm(world2, down, required={"cottage": ("hearth",)})
    assert [o["reason"] for o in got2["owed"]] == ["unsupported"], got2["owed"]
    assert got2["owed"][0]["holds"] is None, got2["owed"]

    # and a token no assembled-world predicate decides says so rather than guessing
    none = construction.evidence_for(record(part), "cottage", "bell")
    assert none["holds"] is None and none["method"] == "unsupported", none
    assert none["owed"] is True, none
    return ("a hearth declared with no rectangle is `holds: None` by `declared` and owed; "
            "a part that did not stand is owed by `unsupported`; a feature no predicate "
            "decides answers `unsupported` instead of passing")


@case
def t2b_the_predicates_asked_follow_what_the_part_requires():
    """`CONFIRM_WANTS` was a fixed three of the six and was written out a second time in
    `growth.py`. A part required to hold a court is now asked whether its ranges enclose
    one; a part required to hold nothing is asked exactly what it was asked before, so no
    caller loses a check by this changing."""
    assert construction.wants_for({}, ()) == construction.CONFIRM_WANTS
    court = construction.wants_for({"emitted": {"features": {"courtyard": True}}},
                                   ("courtyard",))
    assert "range_relation" in court and "circulation_clear" in court, court
    stalls = construction.wants_for({}, ("stalls",))
    assert "passage_connected" in stalls, stalls
    # the growth gate asks the constant and not a copy of it
    from ethoslm import growth
    src = open(os.path.join(ROOT, "src", "ethoslm", "growth.py")).read()
    assert "construction.CONFIRM_WANTS" in src, "growth repeats the want list"
    assert '"entrance_connected", "equipment_reachable",\n' not in src
    assert growth is not None
    b, part, _em = built("courtyard_house", 26, 26, {"storeys": 1})
    rec = record(part)
    got = construction.confirm(usable.World.of_builder(b, part), rec,
                              required={"courtyard_house": ("courtyard",)})
    asked = sorted((rec["waves"][0]["parts"][0]["emitted"]["usable"] or {}))
    assert "range_relation" in asked, asked
    assert got["owed"] == [], got["owed"]
    return (f"a part owing nothing is asked {list(construction.CONFIRM_WANTS)}; a part "
            f"owing a court is asked {asked} and the courtyard house answers all of them")


# ------------------------------------------------- E2. a feature is not an occupied box

@case
def t3_rubble_in_a_hearth_rectangle_is_not_a_hearth():
    """`usable._stands_in` counted any non-air block within `FEATURE_COURSES` of the floor
    and `construction._verify_rect` did the same, so half a rectangle of the voice's own
    masonry answered exactly as a fire did -- which this module's own header admitted. A
    hearth is a fire: `prims.Primitives.FITTING_BLOCKS["hearth"]` lays `campfire[lit=true]`
    on a stone base, and that is the block the check looks for.

    The control is the cottage as `types/cottage.py` builds it, and it has to keep
    passing: the rectangle it publishes is published only where the fire is standing."""
    # the vocabulary this build actually lays, so the table cannot drift from the
    # library
    assert prims.Primitives.FITTING_BLOCKS["hearth"][0] == "campfire"
    assert "campfire" in usable.identifies("hearth"), usable.FEATURE_BLOCKS
    assert usable.identifies("forge"), usable.FEATURE_BLOCKS
    assert usable.identifies("bell") == (), "a bell has no identifying block in this build"

    b, part, em = built("cottage", 20, 20, {"storeys": 1})
    rect = construction.rects_of(em["rects"]["hearth"])[0]
    fy = int(part["floor_y"])
    hx, hz = rect[0], rect[1]
    fire = [p for p, v in b._pending.items() if "campfire" in v]
    assert fire, "the probe cottage laid no fire"

    control = usable.check(usable.World.of_builder(b, part), "cottage",
                          "equipment_reachable")
    assert control["holds"] is True, control
    at = [x for x in control["evidence"]["features"] if x["feature"] == "hearth"]
    assert at and at[0]["at"][0]["identified"] >= 1, at

    # the same rectangle, with rubble in it instead of a fire
    rubble = usable.World.assembled(b)
    rubble.overlay({(hx, y, hz): "cobblestone" for y in (fy + 1, fy + 2, fy + 3)})
    got = usable.check(usable.World.of_builder(b, part, volume=rubble), "cottage",
                       "equipment_reachable")
    assert got["holds"] is False, got
    assert got["evidence"]["unidentified"] == ["hearth"], got["evidence"]
    assert got["evidence"]["gone"] == [], got["evidence"]
    row = [x for x in got["evidence"]["features"] if x["feature"] == "hearth"][0]
    assert row["at"][0]["up"] is True and row["at"][0]["identified"] == 0, row

    # ...and the same at emission, where the type's own blocks are all there is
    cols = {(0, 0): {0: "stone", 1: "cobblestone", 2: "cobblestone"}}
    bad = construction._verify_feature(cols, "hearth", [0, 0, 0, 0], 0)
    assert bad[0] is False and bad[3] == "not_identified", bad
    cols[(0, 0)][1] = "campfire"
    good = construction._verify_feature(cols, "hearth", [0, 0, 0, 0], 0)
    assert good[0] is True and good[3] == "verified", good
    # a feature this build lays no identifying block for keeps the mass bar and says so
    mass = construction._verify_feature({(0, 0): {1: "cobblestone"}}, "bell",
                                        [0, 0, 0, 0], 0)
    assert mass[0] is True and mass[3] == "verified", mass
    ev = construction.evidence_for(record(part), "cottage", "bell")
    assert ev["method"] == "unsupported", ev

    # a court is a floor and the sky over it, not a filled rectangle
    b2, part2, em2 = built("courtyard_house", 26, 26, {"storeys": 1})
    court = construction.rects_of(em2["rects"]["courtyard"])[0]
    fy2 = int(part2["floor_y"])
    open_now = usable.check(usable.World.of_builder(b2, part2), "courtyard_house",
                            "court_accessible")
    assert open_now["holds"] is True, open_now
    sky = open_now["evidence"]["courts"][0]
    assert sky["sky"] >= construction.OPEN_STANDS * sky["cells"], sky
    lid = usable.World.assembled(b2)
    lid.overlay({(x, fy2 + 6, z): "cobblestone"
                 for x in range(court[0], court[2] + 1)
                 for z in range(court[1], court[3] + 1)})
    roofed = usable.check(usable.World.of_builder(b2, part2, volume=lid),
                          "courtyard_house", "court_accessible")
    assert roofed["holds"] is False, roofed
    assert roofed["evidence"]["roofed"] == ["courtyard"], roofed["evidence"]
    return (f"the cottage's fire at ({hx}, {hz}) identifies its hearth ("
            f"{at[0]['at'][0]['identified']} block(s)); the same rectangle filled with "
            f"cobblestone reads `unidentified` at emission and on the assembled world; "
            f"a lid {6} courses over a {sky['cells']}-cell court makes it a room")


# ------------------------------------------------------------ E3. a recurrence reopens

@case
def t4_a_finding_that_recurs_on_a_new_candidate_reopens():
    """`upsert` refreshed a closed row's wording and left its disposition at `closed`, and
    wrote `candidate` only on first insert. So a finding closed against candidate A that
    the reading of candidate B finds again stayed closed, kept A's name, was invisible to
    `open_rows(material=True)` and got no `omitted_by` trace either. A repeated failure
    cannot remain closed."""
    d = tempfile.mkdtemp(prefix="comp-ob-")
    a, bcand = "cand-A", "cand-B"
    try:
        led = obligation.load(d)
        f = {"id": "find/court/spent", "about": "court", "owner": "layout",
             "says": "the court was spent to fit a lot count", "material": True,
             "subjects": ["middle_ring_north_west"],
             "target": {"measure": "open_to_built", "value": 0.25, "direction": "up"}}
        obligation.upsert(led, [obligation.from_finding(f, source="reading-A")],
                          "reading-A", candidate=a)
        shut = obligation.close(led, f["id"], {"candidate": a, "before": 0.10,
                                               "measures": {"open_to_built": 0.30}},
                               candidate=a)
        assert shut["closed"] is True, shut
        assert obligation.closed_against(led["rows"][f["id"]]) == a

        # the control: the same candidate reports it again, and the closure stands
        obligation.upsert(led, [obligation.from_finding(f, source="reading-A2")],
                          "reading-A2", candidate=a)
        assert led["rows"][f["id"]]["disposition"] == "closed", led["rows"][f["id"]]
        # ...and a pass that merely omits it does not reopen it either
        obligation.upsert(led, [], "reading-A3", candidate=a)
        row = led["rows"][f["id"]]
        assert row["disposition"] == "closed" and row["omitted_by"] == ["reading-A3"], row
        assert not obligation.open_rows(led, material=True)

        # the finding recurs on the revised design
        obligation.upsert(led, [obligation.from_finding(f, source="reading-B")],
                          "reading-B", candidate=bcand)
        row = led["rows"][f["id"]]
        assert row["disposition"] == "open", row
        assert [r["id"] for r in obligation.open_rows(led, material=True)] == [f["id"]]
        assert [r["id"] for r in obligation.open_rows(led, material=True,
                                                      candidate=bcand)] == [f["id"]]
        # the history is kept, not overwritten
        assert len(row["reopened"]) == 1, row["reopened"]
        hist = row["reopened"][0]
        assert hist["was"] == "closed" and hist["closed_against"] == a, hist
        assert hist["recurred_on"] == bcand and hist["closed_why"], hist
        assert "0.3" in str(hist["closed_why"]) or "0.30" in str(hist["closed_why"]), hist
        assert obligation.closed_against(row) is None, row
        assert row["seen_on"] == [a, bcand], row

        # and it closes again only on the new candidate's own evidence
        no = obligation.close(led, f["id"], {"candidate": bcand, "before": 0.10,
                                             "measures": {"open_to_built": 0.11}},
                              candidate=bcand)
        assert no["closed"] is False, no
        yes = obligation.close(led, f["id"], {"candidate": bcand, "before": 0.10,
                                              "measures": {"open_to_built": 0.28}},
                               candidate=bcand)
        assert yes["closed"] is True and led["rows"][f["id"]][
            "closed_against"] == bcand, yes
        obligation.save(d, led)
        again = obligation.load(d)["rows"][f["id"]]
        assert again["disposition"] == "closed" and len(again["reopened"]) == 1
        return ("closed on cand-A at open_to_built 0.30; the same candidate reporting it "
                "again and a pass omitting it both leave it closed; cand-B reporting it "
                "reopens it with the whole closure on `reopened`, and 0.11 will not "
                "close it again while 0.28 does")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- E4. the row's measure, on its
# subject

@case
def t5_a_measurement_of_another_subject_cannot_close_a_row():
    """`close` decided relevance by one string lookup of `measure` in a flat global
    measures dict, so a finding about one district closed on a measure of the whole place
    -- the name matched and nothing asked what the number was a measurement *of*."""
    d = tempfile.mkdtemp(prefix="comp-ob2-")
    here = "cand-1"
    try:
        led = obligation.load(d)
        f = {"id": "find/undeveloped/middle_ring_north_west", "about": "undeveloped_share",
             "owner": "layout", "material": True,
             "subjects": ["middle_ring_north_west"],
             "says": "a third of this district's ground is nobody's",
             "target": {"measure": "undeveloped_share", "value": 0.10,
                        "direction": "down"}}
        obligation.upsert(led, [obligation.from_finding(f, source="reading-1")],
                          "reading-1", candidate=here)
        assert obligation.subjects_of(led["rows"][f["id"]]) == \
            ["middle_ring_north_west"]

        # the neighbouring district improved, and this row is not about it
        other = obligation.close(led, f["id"], {
            "candidate": here, "measures": {"undeveloped_share": 0.04},
            "by_subject": {"lower_ring_north_2": {"undeveloped_share": 0.04}}},
            candidate=here)
        assert other["closed"] is False, other
        assert "another subject" in other["why"], other["why"]
        # ...and a measurement that says outright what it is of
        said = obligation.close(led, f["id"], {
            "candidate": here, "subject": "lower_ring_north_2",
            "measures": {"undeveloped_share": 0.04}}, candidate=here)
        assert said["closed"] is False and "another subject" in said["why"], said

        # the control: the row's own subject, measured on the current candidate
        mine = obligation.close(led, f["id"], {
            "candidate": here, "before": 0.33,
            "measures": {"undeveloped_share": 0.19},
            "by_subject": {"middle_ring_north_west": {"undeveloped_share": 0.07},
                           "lower_ring_north_2": {"undeveloped_share": 0.04}}},
            candidate=here)
        assert mine["closed"] is True, mine
        ev = led["rows"][f["id"]]["evidence"][-1]
        assert ev["about"] == "middle_ring_north_west" and ev["value"] == 0.07, ev
        # the place's 0.19 would not have reached the target; the district's 0.07 does
        assert "0.07" in mine["why"], mine["why"]
        return ("a 0.04 measured on the neighbouring district refuses closure by name, "
                "and so does evidence that says it is of another subject; the row's own "
                "district at 0.07 closes it, while the whole place's 0.19 would not have "
                "reached the 0.10 target")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def t6_an_emitted_feature_row_closes_from_feature_evidence_only():
    """`improve.py:318` only ever offers a row `inspect.MEASURES` -- `clusters`,
    `cluster_gap`, `square_scale`, `open_to_built`, `undeveloped_share`, `plots` -- which
    carries no `emitted.*` key. So `constraint/stalls/<part>` could not be closed by any
    evidence that existed, and `_accepted`'s `lot_min` branch was unreachable because no
    caller passed `evidence["lot"]`. What settles an emitted feature is whether the
    assembled world carries it."""
    from ethoslm.pipeline import inspect as inspect_mod
    assert not [k for k in inspect_mod.MEASURES if k.startswith("emitted.")], \
        sorted(inspect_mod.MEASURES)
    d = tempfile.mkdtemp(prefix="comp-ob3-")
    here = "cand-1"
    try:
        led = obligation.load(d)
        c = {"what": "stalls", "part": "market_north_1", "type": "market",
             "owner": "build", "requested": True, "emitted": False,
             "needs": {"lot": [30, 18], "lot_min": None},
             "why": "market was asked for `stalls` and did not deliver them"}
        rid = "constraint/stalls/market_north_1"
        obligation.upsert(led, [obligation.from_constraint(
            c, {"market_north_1": {"stalls": ["function/market"]}},
            part="market_north_1", source="parts-1")], "parts-1", candidate=here)
        row = led["rows"][rid]
        assert row["material"] is True and row["measure"] == "emitted.stalls", row
        assert row["requirement"] == ["function/market"], row
        assert [r["id"] for r in obligation.open_rows(led, material=True)] == [rid]

        # a view measure does not close it: the whole of `inspect.MEASURES`, offered
        view = obligation.close(led, rid, {"candidate": here,
                                           "measures": {k: 1 for k in
                                                        inspect_mod.MEASURES}},
                                candidate=here)
        assert view["closed"] is False and "carries no emitted.stalls" in view["why"], view

        # nor does an unknown post-build answer
        unknown = obligation.close(led, rid, {
            "candidate": here,
            "features": {"market_north_1/stalls":
                         {"holds": None, "method": "declared", "reason": "declared",
                          "why": "the type says it emitted stalls and published no "
                                 "rectangle"}}}, candidate=here)
        assert unknown["closed"] is False, unknown
        assert "not evidence that the feature is there" in unknown["why"], unknown

        # nor does an affirmative answer about another part
        elsewhere = obligation.close(led, rid, {
            "candidate": here,
            "features": {"square_south_1/stalls": {"holds": True, "method": "observed",
                                                   "why": "stands"}}}, candidate=here)
        assert elsewhere["closed"] is False, elsewhere

        # the control: the assembled world carries them, on this part
        good = obligation.close(led, rid, {
            "candidate": here,
            "measures": {k: 1 for k in inspect_mod.MEASURES},
            "features": {"market_north_1/stalls":
                         {"holds": True, "method": "observed", "reason": None,
                          "why": "`equipment_reachable` holds for `stalls` on "
                                 "market_north_1: 1 place(s), 1 standing"}}},
            candidate=here)
        assert good["closed"] is True, good
        ev = led["rows"][rid]["evidence"][-1]
        assert ev["about"] == "feature" and ev["feature"]["method"] == "observed", ev
        assert not obligation.open_rows(led, material=True)
        return ("the six view measures of `inspect.MEASURES` do not close "
                "`emitted.stalls`; nor does a `declared` post-build answer, nor an "
                "affirmative one about another part; the part's own `observed` answer does")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------ E5. words against tokens

@case
def t7_a_market_want_and_a_stalls_constraint_agree():
    """The defect, in one string comparison: `improve._required_tokens` collects the
    sentence's own words, so `feature/market` gives `{"market"}`; the construction
    constraint about the same requirement emits `what = "stalls"`; and
    `obligation.from_constraint` computed `material = "stalls" in {"market"}` -> False. On
    a requirement the sentence makes hard, the row was never material and was never
    selected."""
    assert envelope.feature_token("market") == "stalls"
    assert envelope.feature_token("stalls") == "stalls"
    assert envelope.feature_token("smithy") == "forge"
    assert envelope.feature_token("storeys") == "storeys"
    assert envelope.feature_token("granary") is None

    c = {"what": "stalls", "part": "market_north_1", "type": "market", "owner": "build",
         "requested": True, "emitted": False, "needs": {"lot": [30, 18]},
         "why": "asked for stalls and did not deliver them"}
    # the raw word list `_required_tokens` builds today
    words = obligation.from_constraint(c, {"market", "storeys"}, part="market_north_1")
    assert words["material"] is True, words["evidence"]
    # ...and something nothing requires is still the type's own choice
    opt = obligation.from_constraint({**c, "what": "dormers"}, {"market"},
                                     part="market_north_1")
    assert opt["material"] is False, opt["evidence"]

    # the binding, off the real requirement record and the real type vocabulary
    market_req = {"id": "function/market", "kind": "function", "hard": True,
                  "wants": {"function": "market", "what": "market"},
                  "scope": "middle ring"}
    place = {"kind": "city", "defining_parts": [
        {"name": "middle_ring", "defines": "middle_ring", "kind": "group",
         "family": "district", "density": "medium", "count": 3}],
        "parts": [
            {"name": "middle_ring", "kind": "group", "in": [], "children": [
                {"name": "market_north_1", "kind": "plot", "type": "market",
                 "in": ["middle_ring"], "x0": 0, "z0": 0, "x1": 29, "z1": 17},
                {"name": "house_north_1", "kind": "plot", "type": "courtyard_house",
                 "in": ["middle_ring"], "x0": 40, "z0": 0, "x1": 65, "z1": 25},
            ]}]}
    got = demand.required_by_part(place, {"requirements": [market_req]}, place)
    assert got.get("market_north_1") == {"stalls": ["function/market"]}, got
    assert "house_north_1" not in got, got
    assert demand.required_for(got, "market_north_1") == ("stalls",)
    assert demand.required_for(got, "house_north_1") == ()

    # and the binding drives materiality per part: the same constraint on the house is
    # not
    bound = obligation.from_constraint(c, got, part="market_north_1")
    assert bound["material"] is True, bound["evidence"]
    assert bound["requirement"] == ["function/market"], bound
    house = obligation.from_constraint({**c, "part": "house_north_1"}, got,
                                       part="house_north_1")
    assert house["material"] is False, house["evidence"]
    return ("`market` and `stalls` are one token; the constraint is material from the raw "
            "word list and from the binding; the binding asks the market leaf for stalls "
            "and asks the courtyard house beside it for nothing")


# --------------------------------------------------- E6. prepared ground, and warm
# reuse

class _Rnd:
    """The smallest thing `deps` reads. `scripts/test_realization.py`'s shim."""

    def __init__(self, state, sentence="Build a walled city."):
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
def t8_a_changed_prepared_ground_invalidates_the_built_world():
    """`deps.GROUND_PROPOSAL_FILE` named `ground.json` while `stage_ground` writes
    `ground_proposal.json`, and of the seven keys it hashed only `levels` exists in what
    `ground.propose` writes -- so the fingerprint fell through to the plan every time, and
    `ground.json` is a live, different artifact (`settle_ground`'s record). Meanwhile
    `DEPENDS["built"]` named `terrain`, the ground as it was *found*, and nothing at all
    for the ground this candidate cut. A warm build could be returned across a ground
    change."""
    assert deps.GROUND_PROPOSAL_FILE == "ground_proposal.json"
    assert deps.GROUND_DECLARED_FILE == "ground.json"
    assert "prepared" in deps.DEPENDS["built"], deps.DEPENDS["built"]
    assert "prepared" in deps.KINDS, deps.KINDS
    assert "ground_proposal.json" in deps.ARTIFACT_FILES["ground"]
    assert "ground.json" in deps.ARTIFACT_FILES["ground"]
    with tempfile.TemporaryDirectory() as tmp:
        rnd = _Rnd(tmp)
        open(rnd.rel("world.npz"), "wb").write(b"baseline-ground")
        json.dump({"parts": [{"kind": "plot", "name": "a", "type": "cottage",
                              "x0": 0, "z0": 0, "x1": 9, "z1": 9}], "levels": {}},
                  open(rnd.rel("plan.json"), "w"))

        def proposal(print_, level):
            json.dump({"record": "ground_proposal", "version": 1, "print": print_,
                       "levels": {"podium": level},
                       "pieces": [{"label": "anchor", "rect": [0, 0, 20, 20],
                                   "level": level}]},
                      open(rnd.rel("ground_proposal.json"), "w"))

        proposal("cut-1", 64)
        was = deps.fingerprint(rnd, ("prepared", "ground_proposal"))
        # the proposal's own print is what the kind is keyed on
        proposal("cut-1", 99)
        same = deps.fingerprint(rnd, ("prepared", "ground_proposal"))
        assert same == was, (was, same)
        proposal("cut-1", 64)

        json.dump({"waves": [{"parts": [{"part": "a"}]}], "built": 1},
                  open(rnd.rel("parts.json"), "w"))
        open(rnd.rel("world_built.npz"), "wb").write(b"built-world")
        deps.stamp(rnd, "built", outputs=["parts.json", "world_built.npz"],
                   note="one cottage")
        fresh, why = deps.check(rnd, "built")
        assert fresh, why
        assert "prepared" in deps._load(rnd)["built"]["inputs"], deps._load(rnd)["built"]

        # the control: the ground stage runs again and proposes the same cut
        proposal("cut-1", 64)
        warm, why2 = deps.check(rnd, "built")
        assert warm, why2

        # ...and a different cut withdraws the build
        proposal("cut-2", 64)
        stale, why3 = deps.check(rnd, "built")
        assert not stale and "prepared" in why3, why3

        # so does the baseline the cut was made from
        proposal("cut-1", 64)
        assert deps.check(rnd, "built")[0]
        open(rnd.rel("world.npz"), "wb").write(b"a different baseline")
        moved, why4 = deps.check(rnd, "built")
        assert not moved and "prepared" in why4, why4

        # withdrawing the ground withdraws the build that stands on it
        proposal("cut-1", 64)
        open(rnd.rel("world.npz"), "wb").write(b"baseline-ground")
        assert deps.check(rnd, "built")[0]
        dropped = deps.invalidate(rnd, "ground", "re-cut for the revised anchor")
        assert "built" in dropped, dropped
        gone, why5 = deps.check(rnd, "built")
        assert not gone and "invalidated" in why5, why5
        return (f"the proposal's own print keys the cut (a level change under the same "
                f"print is warm); a `cut-2` proposal, a changed baseline and an "
                f"invalidated ground each withdraw the built world -- {why3[:70]}...")


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
    print(f"\n{len(CASES) - bad}/{len(CASES)} composition evidence cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
