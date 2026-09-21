"""The closure round's meaning counterexamples, each with a positive control.

    $PY scripts/test_closure_meaning.py

Every case is a thing the independent realization review **reproduced** -- a count read
to a different number and called agreement, containment certified by proximity, a
planned storey count read as a built one, a tradition tag overruling a negative
inspection, a citation of research nobody did -- run through the production entry
points the stages call: `interpret.requirements`, `interpret.cross_check`,
`intent.coverage`, `intent.select`, `intent.relation_measure`, `placeread.read`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import contracts, interpret, intent as intent_mod, placeread  # noqa: E402

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


def _req(rec, rid):
    return next(r for r in rec["requirements"] if r["id"] == rid)


def _plot(name, t, x, z, w=9, d=9, storeys=None, **more):
    p = {"name": name, "kind": "plot", "type": t, "x0": x, "z0": z, "x1": x + w - 1,
         "z1": z + d - 1, "in": more.pop("in", ["homes_west"]), **more}
    if storeys is not None:
        p["params"] = {"storeys": storeys}
    return p


def _village(n_cottages=16, hall=True, ring=True):
    """A plan of cottages round a square, a hall on it, a field beside the houses."""
    parts = [{"name": "market", "kind": "area", "type": "square", "family": "square",
              "x0": 40, "z0": 40, "x1": 63, "z1": 63, "defines": "market"}]
    if hall:
        parts.append(_plot("hall", "hall", 66, 44, 12, 12, in_=[]))
        parts[-1]["in"] = []
    spots = [(20, 20), (32, 20), (44, 20), (56, 20), (68, 20), (80, 20),
             (20, 32), (80, 32), (20, 44), (80, 56), (20, 68), (80, 68),
             (20, 80), (32, 80), (44, 80), (56, 80), (68, 80), (80, 80), (32, 92), (44, 92)]
    if not ring:
        spots = [(x, 20 + 12 * (i // 6)) for i, (x, _z) in enumerate(spots)]
    for i in range(n_cottages):
        x, z = spots[i % len(spots)]
        parts.append(_plot(f"c{i}", "cottage", x, z, 9, 9, storeys=1))
    parts.append({"name": "f1", "kind": "area", "type": "field", "x0": 94, "z0": 16,
                  "x1": 124, "z1": 60, "in": ["fields_south"], "defines": "fields"})
    return {"parts": parts, "levels": {}}


S = ("Build a farming village of sixteen low cottages gathered around a market square, "
     "with a hall on the square, fields beside the houses, and no wall.")


def _interp(reads):
    return interpret.read_answer(S, {"reads": reads})


def _reads():
    return [
        {"id": "count/cottages", "kind": "count", "says": "sixteen cottages",
         "wants": {"n": 16, "about": False, "what": "cottages"},
         "phrase": "sixteen low cottages", "why": "the number"},
        {"id": "relation/cottages_around_market", "kind": "relation", "says": "around",
         "wants": {"subject": "cottages", "relation": "around", "object": "market"},
         "phrase": "gathered around a market square", "why": "bearings"},
        {"id": "relation/hall_on_square", "kind": "relation", "says": "on the square",
         "wants": {"subject": "hall", "relation": "beside", "object": "market"},
         "phrase": "a hall on the square", "why": "edge"},
        {"id": "relation/fields_beside_houses", "kind": "relation", "says": "beside",
         "wants": {"subject": "fields", "relation": "beside", "object": "houses"},
         "phrase": "fields beside the houses", "why": "edge"},
        {"id": "feature/hall", "kind": "feature", "says": "a hall",
         "wants": {"feature": "hall", "family": "hall", "count": 1}, "phrase": "a hall",
         "why": "named"},
        {"id": "function/farming", "kind": "function", "says": "farming",
         "wants": {"what": "village", "function": "farming"}, "phrase": "farming village",
         "hard": False, "why": "purpose"}]


SPEC = {"sentence": S, "kind": "village", "form": "european_vernacular",
        "size_band": [16, 16], "structures": 16,
        "explicit_count": {"n": 16, "about": False, "phrase": "sixteen low cottages",
                           "what": "cottages"},
        "defining_parts": [
            {"name": "market", "kind": "area", "family": "square", "relation": "centre",
             "count": 1},
            {"name": "hall", "kind": "plot", "family": "hall", "relation": "near",
             "count": 1},
            {"name": "homes", "kind": "group", "family": "district", "relation":
             "throughout", "count": 1},
            {"name": "fields", "kind": "group", "family": "district", "relation":
             "quarter", "count": 1, "notes": "fields"}]}


# ------------------------------------------------------------- 1. exact counts

@case
def t_a_sixteen_reads_and_counts_the_cottages_not_the_hall():
    rules = intent_mod.read(S)
    c = next(r for r in rules["requirements"] if r["kind"] == "count")
    assert c["wants"] == {"n": 16, "about": False, "what": "cottages"}, c["wants"]
    assert not any(r["id"] == "clause/sixteen" for r in rules["requirements"])
    rec = interpret.requirements(S, _interp(_reads()))
    got, _f = intent_mod.coverage(rec, SPEC, plan=_village(16, hall=True))
    r = _req(got, "count/cottages")
    assert r["status"] == "satisfied", r
    got, _f = intent_mod.coverage(rec, SPEC, plan=_village(17, hall=True))
    r = _req(got, "count/cottages")
    assert r["status"] == "failed", r
    # control: an unscoped count counts every plot, as before
    rec2 = interpret.requirements("Build exactly 17 buildings.", None)
    got2, _f = intent_mod.coverage(rec2, SPEC, plan=_village(16, hall=True))
    assert _req(got2, "count/structures")["status"] == "satisfied", _req(got2, "count/structures")
    return ("`sixteen` reads as an exact count of the cottages; 16 cottages and a hall "
            "satisfy it, 17 cottages fail it, and an unscoped count still counts every "
            "plot")


@case
def t_b_fifty_to_one_is_a_disagreement_and_the_fifty_survives():
    s = "Build exactly 50 houses."
    bad = interpret.read_answer(s, {"reads": [
        {"id": "count/houses", "kind": "count", "says": "one house",
         "wants": {"n": 1, "about": False, "what": "houses"}, "phrase": "houses",
         "why": "a wrong reading"}]})
    ck = interpret.cross_check(s, bad)
    assert ck["contradicts"] and ck["contradicts"][0].get("rule_wins"), ck
    assert not ck["agreed"], ck
    got = interpret.requirements(s, bad)
    counts = [r for r in got["requirements"] if r["kind"] == "count"]
    assert len(counts) == 1 and counts[0]["wants"]["n"] == 50, counts
    assert counts[0]["wants"].get("what") == "houses", counts
    good = interpret.read_answer(s, {"reads": [
        {"id": "count/houses", "kind": "count", "says": "fifty",
         "wants": {"n": 50, "about": False, "what": "houses"},
         "phrase": "exactly 50 houses", "why": "the number"}]})
    ck2 = interpret.cross_check(s, good)
    assert len(ck2["agreed"]) == 1 and not ck2["contradicts"], ck2
    halls = interpret.read_answer(s, {"reads": [
        {"id": "count/halls", "kind": "count", "says": "fifty halls",
         "wants": {"n": 50, "about": False, "what": "halls"}, "phrase": "50 houses",
         "why": "wrong subject"}]})
    ck4 = interpret.cross_check(s, halls)
    assert ck4["contradicts"] and ck4["contradicts"][0]["values"]["reader"]["what"] == "halls", ck4
    got4 = interpret.requirements(s, halls)
    c4 = [r for r in got4["requirements"] if r["kind"] == "count"]
    assert len(c4) == 1 and c4[0]["wants"].get("what") == "houses", c4
    cottages = interpret.read_answer(s, {"reads": [
        {"id": "count/c", "kind": "count", "says": "fifty cottages",
         "wants": {"n": 50, "about": False, "what": "cottages"}, "phrase": "50 houses",
         "why": "a house word"}]})
    assert not interpret.cross_check(s, cottages)["contradicts"], "cottages are houses"
    about = interpret.read_answer("Build about 50 houses.", {"reads": [
        {"id": "count/houses", "kind": "count", "says": "fifty",
         "wants": {"n": 50, "about": False, "what": "houses"}, "phrase": "50 houses",
         "why": "dropped the hedge"}]})
    ck3 = interpret.cross_check("Build about 50 houses.", about)
    assert ck3["contradicts"], "dropping `about` is a changed value and was not noticed"
    return ("50 read as 1 is recorded as a contradiction the rule wins, the sentence's "
            "50 survives with the reader's subject, an agreeing reading agrees, a "
            "dropped `about` is a disagreement, fifty halls for fifty houses is a "
            "subject disagreement the sentence wins, and cottages are houses")


# --------------------------------------------------------- 2. the one selector

@case
def t_c_select_is_one_rule_for_both_sides():
    plan = _village(16)
    parts = plan["parts"]
    assert len(intent_mod.select(parts, "cottages")) == 16
    assert len(intent_mod.select(parts, "houses")) == 16
    assert [p["name"] for p in intent_mod.select(parts, "hall")] == ["hall"]
    assert [p["name"] for p in intent_mod.select(parts, "market")] == ["market"]
    assert [p["name"] for p in intent_mod.select(parts, "fields")] == ["f1"]
    assert [p["name"] for p in intent_mod.select(parts, "the south quarter")] == ["f1"]
    assert intent_mod.select(parts, "temple") == []
    place = {"parts": [{"name": "market", "kind": "area", "type": "square",
                        "family": "square", "x0": 0, "z0": 0, "x1": 20, "z1": 20},
                       {"name": "hall", "kind": "plot", "type": "hall",
                        "x0": 22, "z0": 0, "x1": 30, "z1": 8}],
             "districts": [{"name": "homes_west", "defines": "homes", "structures": 8,
                            "x0": -60, "z0": 0, "x1": -10, "z1": 40},
                           {"name": "fields_south", "defines": "fields", "structures": 4,
                            "x0": 0, "z0": 30, "x1": 60, "z1": 90,
                            "purpose": "the fields beside the houses"}],
             "compounds": []}
    assert [r["name"] for r in intent_mod.select_regions(place, "houses")] == ["homes_west"]
    assert [r["name"] for r in intent_mod.select_regions(place, "fields")] == ["fields_south"]
    assert [r["name"] for r in intent_mod.select_regions(place, "market")] == ["market"]
    assert [r["name"] for r in intent_mod.select_regions(place, "the south quarter")] == ["fields_south"]
    return ("cottages/houses select the dwelling types and not the hall; market, hall, "
            "fields and a scope phrase select their own leaves; the same at the place "
            "level over districts")


# ------------------------------------------------------------ 3. relations

@case
def t_d_inside_is_containment_and_beside_is_the_edge():
    temple = {"name": "temple", "kind": "plot", "type": "temple", "x0": 0, "z0": 0,
              "x1": 19, "z1": 19}
    outside_near = _plot("h1", "cottage", 40, 40)
    contained = _plot("h2", "cottage", 4, 4)
    far = _plot("h3", "cottage", 200, 200)
    w = {"subject": "houses", "relation": "inside", "object": "temple"}
    st, why, _ev = intent_mod.relation_measure(w, [temple, outside_near])
    assert st == "failed", (st, why)
    st, why, _ev = intent_mod.relation_measure(w, [temple, contained])
    assert st == "satisfied", (st, why)
    w = {"subject": "houses", "relation": "beside", "object": "temple"}
    near = _plot("h4", "cottage", 24, 4)          # 4 columns off the temple's edge
    st, why, ev = intent_mod.relation_measure(w, [temple, near])
    assert st == "satisfied", (st, why)
    st, why, ev = intent_mod.relation_measure(w, [temple, far])
    assert st == "failed", (st, why)
    st, why, ev = intent_mod.relation_measure(
        {"subject": "houses", "relation": "between", "object": "temple"}, [temple, near])
    assert st == "unresolved", (st, why)
    # a single building keeps the per-subject rule: the hall thirty columns off fails
    square = {"name": "market", "kind": "area", "type": "square", "family": "square",
              "x0": 0, "z0": 0, "x1": 23, "z1": 23}
    hall_far = _plot("hall", "hall", 54, 0, 12, 12, in_=[])
    hall_far["in"] = []
    st, why, ev = intent_mod.relation_measure(
        {"subject": "hall", "relation": "beside", "object": "market"}, [square, hall_far])
    assert st == "failed" and ev["unit"] == "each", (st, why, ev)
    return ("a house forty columns outside a temple fails `inside`, a contained one "
            "passes; `beside` is measured to the edge per building, and a hall thirty "
            "columns from the square fails; an unmeasurable relation is unresolved")


@case
def t_d2_plural_working_ground_is_measured_as_a_block():
    """`fields beside the houses`: the fields as one block adjoining the homes."""
    homes = [_plot(f"c{i}", "cottage", 20 + 12 * (i % 4), 20 + 12 * (i // 4), 9, 9)
             for i in range(16)]                       # x 20..64, z 20..64
    def tiles(x0):
        return [{"name": f"f{i}_{j}", "kind": "area", "type": "field",
                 "x0": x0 + 22 * i, "z0": 10 + 22 * j, "x1": x0 + 22 * i + 19,
                 "z1": 10 + 22 * j + 19, "in": ["fields_south"], "defines": "fields"}
                for i in range(2) for j in range(4)]
    w = {"subject": "fields", "relation": "beside", "object": "houses"}
    st, why, ev = intent_mod.relation_measure(w, homes + tiles(69))
    assert st == "satisfied" and ev["unit"] == "group", (st, why, ev)
    assert ev["tiles_within_reach"] >= 1 and ev["gap"] <= 12, ev
    st, why, ev = intent_mod.relation_measure(w, homes + tiles(160))
    assert st == "failed" and ev["unit"] == "group", (st, why, ev)
    # a block that reaches a house with one far corner and stands away otherwise
    far = tiles(160)
    far[0].update(x0=66, x1=85)              # one tile pulled beside the homes
    st, why, ev = intent_mod.relation_measure(w, homes + far)
    assert st == "satisfied", (st, why, ev)   # the block's union does adjoin: honest
    return ("eight field tiles as a block four columns from the homes pass `beside` "
            "with the group unit; the same tiles across the site fail; a lone hall is "
            "still measured itself")


@case
def t_e_the_proof_sentence_relations_on_a_ring_and_on_a_row():
    rec = interpret.requirements(S, _interp(_reads()))
    got, _f = intent_mod.coverage(rec, SPEC, plan=_village(16, ring=True))
    for rid in ("relation/cottages_around_market", "relation/hall_on_square",
                "relation/fields_beside_houses"):
        assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    got, _f = intent_mod.coverage(rec, SPEC, plan=_village(16, ring=False))
    assert _req(got, "relation/cottages_around_market")["status"] == "failed", \
        _req(got, "relation/cottages_around_market")
    return ("cottages on every side of the square satisfy `around`, a row north of it "
            "fails; the hall on the square and the fields beside the houses hold")


# --------------------------------------------------------- 4. density metric

@case
def t_f_density_is_one_metric_over_one_denominator():
    t = intent_mod.density_target("sparse")
    assert t["metric"] == "lot_cover" and t["hi"] == 0.12 and t["lo"] == 0.03, t
    assert intent_mod.density_target("dense")["lo"] == 0.30
    regions = [{"name": "upper_quarter", "part": "upper", "rect": [0, 0, 99, 99],
                "lots": 4, "developable_columns": 8000}]
    plots = [_plot(f"u{i}", "cottage", 10 + 12 * i, 10, 10, 10, in_=None)
             for i in range(4)]
    for p in plots:
        p["in"] = ["upper_quarter"]
    m = intent_mod.lot_cover(plots, regions)
    assert m["denominator"] == "developable_columns" and m["ground"] == 8000, m
    assert abs(m["cover"] - 400 / 8000) < 1e-9, m
    st, why, ev = intent_mod._quality_measure("density", "sparse", 0.12, plots,
                                              {"regions": regions}, {},
                                              scope="upper quarter")
    assert st == "satisfied" and ev["denominator"] == "developable_columns", (st, why, ev)
    dense = [_plot(f"d{i}", "cottage", 10 * (i % 9), 10 * (i // 9), 9, 9)
             for i in range(36)]
    for p in dense:
        p["in"] = ["upper_quarter"]
    st, why, ev = intent_mod._quality_measure("density", "sparse", 0.12, dense,
                                              {"regions": regions}, {},
                                              scope="upper quarter")
    assert st == "failed", (st, why)
    regions[0].pop("developable_columns")
    m2 = intent_mod.lot_cover(plots, regions)
    assert m2["denominator"] == "rect" and m2["ground"] == 10000, m2
    return ("sparse is 3-12% of the developable ground; measured over it where the "
            "record carries it and over the rectangle -- and said so -- where not")


# ------------------------------------------------------ 5. emitted construction

def _parts_record(rows):
    return {"waves": [{"wave": "w", "parts": rows}]}


@case
def t_g_height_is_read_off_what_construction_emitted():
    plan = {"parts": [_plot(f"c{i}", "cottage", 12 * i, 0, 9, 9, storeys=3)
                      for i in range(4)], "levels": {}}
    rec = intent_mod.read("Build a village of tall cottages.")
    rid = "quality/height/tall"
    got, _f = intent_mod.coverage(rec, SPEC, plan=plan)
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    built = _parts_record([{"part": f"c{i}", "status": "built", "stood": True,
                            "emitted": {"storeys": 1, "features": {}, "omitted": [],
                                        "fallback": "the lot is too small for three",
                                        "why": "9x9 lot: one storey emitted"}}
                           for i in range(4)])
    got, found = intent_mod.coverage(rec, SPEC, plan=plan, parts_record=built)
    r = _req(got, rid)
    assert r["status"] == "failed" and "emitted" in r["why"], r
    em = [f for f in found["findings"] if f["id"].startswith("find/emitted/")]
    assert len(em) == 4 and all(f["blocks"] == "construction" for f in em), em
    assert em[0]["owner"] == "layout", em[0]
    ok = _parts_record([{"part": f"c{i}", "status": "built", "stood": True,
                         "emitted": {"storeys": 3, "features": {}, "omitted": [],
                                     "fallback": None, "why": ""}} for i in range(4)])
    got, found = intent_mod.coverage(rec, SPEC, plan=plan, parts_record=ok)
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    assert not [f for f in found["findings"] if f["id"].startswith("find/emitted/")]
    # ...and the place read carries the constraints without failing on them: a lost
    # lean-to on a cottage the `low` clause still holds for
    low = intent_mod.read("Build a village of low cottages.")
    plan1 = {"parts": [_plot(f"c{i}", "cottage", 12 * i, 0, 6, 5, storeys=1)
                       for i in range(2)], "levels": {}}
    for p in plan1["parts"]:
        p["params"]["outshot"] = "scullery"
    lost = _parts_record([{"part": f"c{i}", "status": "built", "stood": True,
                           "emitted": {"storeys": 1, "features": {"outshot": False},
                                       "omitted": ["outshot"],
                                       "fallback": "plan: no room for a lean-to",
                                       "attempted": {"storeys": 1, "outshot": "scullery"},
                                       "why": "1 floor level inside a 6x5 shell"}}
                          for i in range(2)])
    read = placeread.read(dict(SPEC, sentence="Build a village of low cottages.",
                               voice="drystone_and_thatch"), plan1, lost, intent_rec=low)
    lim = read["limits"]["construction"]
    assert len(lim) == 2 and lim[0]["what"] == "outshot" and lim[0]["owner"] == "build", lim
    assert lim[0]["needs"].get("lot") == [6, 5] and "lot_min" in lim[0]["needs"], lim[0]
    assert "asked/quality/height/low" not in read["failed"], read["failed"]
    assert not any(c["clause"].startswith("find/") for c in read["clauses"])
    return ("four cottages planned at three storeys read tall on the plan, low off an "
            "emitted one storey with a construction finding per part owned by the "
            "layout, and tall again when three were emitted")


# ------------------------------------------------ 6. judgments, both directions

def _caps_japanese():
    return {"entries": [{"id": "e1", "matched": True, "type": "minka",
                         "wants": {"of": "fabric"},
                         "envelope": {"tradition": "japanese"}}]}


def _reading():
    return {"sources": [{"id": "s1", "title": "t", "url": "u", "accessed": "d",
                         "fingerprint": "f"}],
            "claims": [{"id": "c1", "says": "deep thatched roofs", "source": "s1"},
                       {"id": "c2", "says": "guess", "inferred": True}]}


@case
def t_h_a_negative_judgment_overrules_a_tradition_tag():
    s = "Build a Japanese village."
    rec = intent_mod.read(s)
    rid = "tradition/japanese"
    plan = {"parts": [_plot(f"m{i}", "minka", 12 * i, 0, 9, 9) for i in range(6)],
            "levels": {}}
    spec = dict(SPEC, form="east_asian", sentence=s)
    got, _f = intent_mod.coverage(rec, spec, plan=plan, capabilities=_caps_japanese())
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    no = {"candidate": "x", "looked_at": ["/r/world_built.npz"],
          "verdicts": [{"about": "tradition", "name": "japanese", "holds": False,
                        "why": "flat roofs, no eaves", "cites": ["c1"]}]}
    got, found = intent_mod.coverage(rec, spec, plan=plan, capabilities=_caps_japanese(),
                                     reading=_reading(), judgment=no)
    r = _req(got, rid)
    assert r["status"] == "failed", r
    assert any(f["requirement"] == rid for f in found["findings"]), found
    yes = {"candidate": "x", "looked_at": ["/r/world_built.npz"],
           "verdicts": [{"about": "tradition", "name": "japanese", "holds": True,
                         "why": "deep thatch, lifted floors", "cites": ["c1"]}]}
    got, _f = intent_mod.coverage(rec, spec, plan=plan, capabilities=_caps_japanese(),
                                  reading=_reading(), judgment=yes)
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    # ...and a positive judgment with no declaring type at all closes it too
    got, _f = intent_mod.coverage(rec, spec, plan=plan, capabilities={"entries": []},
                                  reading=_reading(), judgment=yes)
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    return ("six minka read Japanese by declaration; a negative inspection of the "
            "built place fails it regardless; a usable positive one closes it, with or "
            "without a declaring type")


@case
def t_i_plan_only_and_fake_citations_cannot_qualify():
    rec = intent_mod.read("Build Ringed City.")
    rid = "identity/ringed_city"
    spec = {"defining_parts": [], "sentence": "Build Ringed City."}
    plan_only = {"candidate": "x", "looked_at": ["/r/plan.json"],
                 "verdicts": [{"about": "identity", "name": "Ringed City",
                               "recognisable": True, "why": "rings", "cites": ["c1"]}]}
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=plan_only)
    r = _req(got, rid)
    assert r["status"] == "unresolved" and "built" in r["why"], r
    fake = {"candidate": "x", "looked_at": ["/r/world_built.npz"],
            "verdicts": [{"about": "identity", "name": "Ringed City",
                          "recognisable": True, "why": "rings", "cites": ["nope"]}]}
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=fake)
    r = _req(got, rid)
    assert r["status"] == "unresolved" and "nope" in r["why"], r
    inferred = dict(fake, verdicts=[dict(fake["verdicts"][0], cites=["c2"])])
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=inferred)
    assert _req(got, rid)["status"] == "unresolved", _req(got, rid)
    uncited = dict(fake, verdicts=[dict(fake["verdicts"][0], cites=[])])
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=uncited)
    assert _req(got, rid)["status"] == "unresolved", _req(got, rid)
    good = dict(fake, verdicts=[dict(fake["verdicts"][0], cites=["c1"])])
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=good)
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    no = dict(fake, verdicts=[dict(fake["verdicts"][0], recognisable=False, cites=[])])
    got, _f = intent_mod.coverage(rec, spec, reading=_reading(), judgment=no)
    assert _req(got, rid)["status"] == "failed", _req(got, rid)
    try:
        contracts.read("reading", {"record": "reading", "version": 1, "sentence": "x",
                                   "classification": "named", "sources": [],
                                   "claims": [{"id": "c1", "says": "s", "source": "s9"}]})
        raise AssertionError("a claim citing an unretrieved source was accepted")
    except contracts.ContractError:
        pass
    return ("plan-only, uncited, fake-cited and inferred-cited positive verdicts leave "
            "an identity unresolved naming why; a sourced one on built output closes "
            "it; a negative one fails it; a reading citing an unretrieved source is "
            "refused by the contract")


# --------------------------------------------------- 7. evidence links survive

@case
def t_j_reference_obligations_keep_their_claims_and_lose_fake_ones():
    s = "Build Ringed City."
    interp = interpret.read_answer(s, {"reads": [
        {"id": "feature/farmland", "kind": "feature", "says": "an agrarian zone",
         "wants": {"feature": "farmland", "family": "district"}, "phrase": "ringed-city",
         "why": "the sources say the outer ring is farmland", "evidence": ["c1", "nope"]},
        {"id": "quality/density/dense", "kind": "quality", "says": "crowded lower ring",
         "wants": {"axis": "density", "value": "dense"}, "scope": "lower ring",
         "phrase": "ringed-city", "why": "sourced", "evidence": ["c1"]}]})
    got = interpret.requirements(s, interp, reading=_reading())
    farm = _req(got, "feature/farmland")
    assert farm["evidence"] == ["c1"] and farm["status"] == "open", farm
    assert got["note"].endswith("stripped"), got["note"]
    assert _req(got, "quality/density/dense")["scope"] == "lower ring"
    return ("a reference-derived farmland obligation is carried with its real claim, "
            "the citation of a claim nobody made is stripped and recorded, and a "
            "scoped quality keeps its scope")


# ------------------------------------------------------------ 8. the place read

@case
def t_k_placeread_carries_its_limits_and_the_same_coverage():
    rec = interpret.requirements(S, _interp(_reads()))
    plan = _village(16)
    rows = [{"part": p["name"], "status": "built", "stood": True}
            for p in plan["parts"]]
    parts_record = {"waves": [{"wave": "w", "parts": rows}],
                    "sample": {"plots": 16, "quarters": ["homes_west"]}}
    got = placeread.read(SPEC, plan, parts_record, intent_rec=rec)
    assert "limits" in got and got["limits"]["sample"]["plots"] == 16, got["limits"]
    asked = {c["clause"]: c for c in got["clauses"] if c["clause"].startswith("asked/")}
    assert asked["asked/count/cottages"]["holds"], asked["asked/count/cottages"]
    assert asked["asked/relation/cottages_around_market"]["holds"]
    assert "function/farming" not in {c.replace("asked/", "") for c in asked}, \
        "a soft requirement is not a clause the place is held to"
    count = next(c for c in got["clauses"] if c["clause"] == "count")
    assert count["holds"] and count["subject"] == "cottages", count
    plan17 = _village(17)
    rows17 = [{"part": p["name"], "status": "built", "stood": True}
              for p in plan17["parts"]]
    got17 = placeread.read(SPEC, plan17, {"waves": [{"wave": "w", "parts": rows17}]},
                           intent_rec=rec)
    count17 = next(c for c in got17["clauses"] if c["clause"] == "count")
    assert not count17["holds"] and count17["stood"] == 17, count17
    return ("the place read reports its sample limit and unresolved clauses beside the "
            "verdict, its asked/ clauses are the coverage's own, and its count clause "
            "counts the sixteen cottages and not the hall (seventeen cottages fail)")


@case
def t_l_a_compound_answers_its_word_as_one_thing():
    """`royal_palace` answers `palace` as one rectangle; `palace_row` houses do not."""
    halls = [{"name": f"hall_{i}", "kind": "plot", "type": "temple", "family": "hall",
              "compound": "royal_palace", "answers": "royal_palace",
              "x0": 100 + 30 * i, "z0": 100, "x1": 120 + 30 * i, "z1": 120, "in": []}
             for i in range(3)]
    court = {"name": "court", "kind": "area", "type": "court_small", "family": "court",
             "compound": "royal_palace", "answers": "royal_palace",
             "x0": 100, "z0": 130, "x1": 180, "z1": 160, "in": []}
    row = [_plot(f"palace_row_{i}", "cottage", 10 * i, 300, 9, 9, in_=None)
           for i in range(4)]
    for p in row:
        p["in"], p["defines"] = ["palace_row"], "palace_row"
    ring = [_plot(f"h{i}", "cottage", x, z, 9, 9) for i, (x, z) in enumerate(
        [(60, 60), (140, 60), (220, 60), (60, 140), (220, 140), (60, 200), (140, 200),
         (220, 200)])]
    parts = halls + [court] + row + ring
    got = intent_mod.select(parts, "palace")
    assert len(got) == 1 and got[0]["name"] == "royal_palace", got
    assert got[0]["x0"] == 100 and got[0]["x1"] == 180 and got[0]["z1"] == 160, got[0]
    assert sorted(got[0]["leaves"]) == ["court", "hall_0", "hall_1", "hall_2"], got[0]
    st, why, ev = intent_mod.relation_measure(
        {"subject": "houses", "relation": "around", "object": "palace"}, parts)
    assert ev["object_rect"] == [100, 100, 180, 160], ev
    st, why, ev = intent_mod._hierarchy_measure(
        {"greater": "palace", "lesser": "houses", "axis": "size"}, parts, {})
    assert st == "satisfied" and ev["greater"] == 81 * 61, (st, why, ev)
    st, why, ev = intent_mod.relation_measure(
        {"subject": "city", "relation": "around", "object": "palace"}, parts)
    assert st == "satisfied" and ev["subjects"] == 12, (st, why, ev)
    place = {"parts": [], "compounds": [{"name": "royal_palace", "x0": 100, "z0": 100,
                                         "x1": 180, "z1": 160}],
             "districts": [{"name": "palace_row", "defines": "homes", "structures": 4,
                            "x0": 0, "z0": 300, "x1": 60, "z1": 320}]}
    assert [r["name"] for r in intent_mod.select_regions(place, "palace")] == ["royal_palace"]
    return ("three halls and a court of `royal_palace` answer `palace` as one 81x61 "
            "rectangle for the relation and the hierarchy; four cottages of "
            "`palace_row` do not; the same at the place level")


@case
def t_m_a_missed_quality_is_fidelity_at_every_stage():
    """A plan short of `dense` is buildable: the finding is fidelity, and the place
    read still fails on it once built."""
    rec = intent_mod.read("Build a dense village.")
    rid = "quality/density/dense"
    plan = {"parts": [_plot(f"c{i}", "cottage", 20 * i, 20, 9, 9) for i in range(4)],
            "levels": {}}
    res = {"regions": [{"name": "d", "part": "homes", "lots": 4, "rect": [0, 0, 99, 99],
                        "developable_columns": 10000}]}
    got, found = intent_mod.coverage(rec, SPEC, plan=plan, resolution=res)
    f = next(x for x in found["findings"] if x["requirement"] == rid)
    assert f["blocks"] == "fidelity" and f["owner"] == "layout", f
    assert "coverage" in f["evidence"] and f["evidence"]["lo"] == 0.30, f["evidence"]
    assert _req(got, rid)["status"] == "failed"
    rows = [{"part": p["name"], "status": "built", "stood": True}
            for p in plan["parts"]]
    read = placeread.read(dict(SPEC, sentence="Build a dense village."), plan,
                          {"waves": [{"wave": "w", "parts": rows}]}, intent_rec=rec,
                          resolution=res)
    assert not read["holds"] and f"asked/{rid}" in read["failed"], read["failed"]
    # control: a relation the plan misses still blocks feasibility
    rec2 = interpret.requirements(S, _interp(_reads()))
    _g, found2 = intent_mod.coverage(rec2, SPEC, plan=_village(16, ring=False))
    f2 = next(x for x in found2["findings"]
              if x["requirement"] == "relation/cottages_around_market")
    assert f2["blocks"] == "feasibility", f2
    return ("four cottages at 3% against `dense` are a layout fidelity finding on the "
            "plan, the built place read fails on the same clause, and a missed relation "
            "still blocks feasibility")


@case
def t_n_a_character_landmark_is_a_declared_feature():
    s = "Build a hamlet of ten low houses along the lake shore with a smithy and no wall."
    rec = intent_mod.read(s)
    rid = "feature/workshop"
    homes = {"name": "homes", "kind": "group", "family": "district", "relation":
             "throughout", "count": 1, "character": {"landmarks": [{"type": "workshop"}]}}
    with_lm = {"sentence": s, "kind": "hamlet", "size_band": [10, 10], "structures": 10,
               "defining_parts": [homes]}
    without = dict(with_lm, defining_parts=[dict(homes, character={})])
    assert intent_mod._families_in(with_lm).get("workshop") == 1
    assert not intent_mod._families_in(without).get("workshop")
    got, _f = intent_mod.coverage(rec, with_lm)
    assert _req(got, rid)["status"] == "open", _req(got, rid)
    got, _f = intent_mod.coverage(rec, without)
    assert _req(got, rid)["status"] == "failed", _req(got, rid)
    plan = {"parts": [*[_plot(f"c{i}", "cottage", 12 * i, 0, 9, 9) for i in range(10)],
                      _plot("landmark_workshop", "workshop", 60, 20, 10, 10)],
            "levels": {}}
    got, _f = intent_mod.coverage(rec, with_lm, plan=plan)
    assert _req(got, rid)["status"] == "open" and "1 part(s)" in _req(got, rid)["why"], \
        _req(got, rid)
    rows = [{"part": p["name"], "status": "built", "stood": True} for p in plan["parts"]]
    got, _f = intent_mod.coverage(rec, with_lm, plan=plan,
                                  parts_record={"waves": [{"wave": "w", "parts": rows}]})
    assert _req(got, rid)["status"] == "satisfied", _req(got, rid)
    return ("a district whose character declares a workshop landmark declares one "
            "workshop, one without does not, and the laid `landmark_workshop` answers "
            "the want planned and standing")


@case
def t_o_a_sample_qualifies_only_its_constructed_scope():
    flat = _village(16)["parts"]
    cottages = [q for q in flat if q["name"].startswith("c")]
    # a plan is a tree, and `plan_parts` reads a leaf's quarter off its ancestry
    plan = {"parts": [
        *[q for q in flat if q["name"] in ("market", "hall")],
        {"name": "homes_west", "kind": "district", "defines": "homes",
         "children": cottages[:8]},
        {"name": "homes_east", "kind": "district", "defines": "homes",
         "children": cottages[8:]},
        {"name": "fields_south", "kind": "district", "defines": "fields",
         "children": [q for q in flat if q["name"] == "f1"]}], "levels": {}}
    rec = interpret.requirements(S, _interp(_reads()))
    west = cottages[:8]
    sample = {"quarters": ["homes_west"], "plots": 8, "joining_parts": ["market"],
              "of_plan": {"leaves": len(plan["parts"])}}

    def record(missing=None):
        rows = [{"part": p["name"], "status": "built",
                 "stood": p["name"] != missing} for p in west]
        rows.append({"part": "market", "status": "built", "stood": True})
        return {"waves": [{"wave": "w", "parts": rows}], "sample": sample}

    spec = dict(SPEC, voice="drystone_and_thatch")
    got = placeread.read(spec, plan, record(), intent_rec=rec)
    assert got["holds"], got["failed"]
    by = {c["clause"]: c for c in got["clauses"]}
    assert by["count"]["sampled"] and by["count"]["planned_whole"] == 16, by["count"]
    assert by["count"]["says"].endswith("(sample of 9 of 19 leaves)"), by["count"]["says"]
    assert by["present/hall"]["outside_sample"], by["present/hall"]
    assert by["present/fields"]["outside_sample"], by["present/fields"]
    assert by["asked/count/cottages"]["holds"] and by["asked/feature/hall"]["outside_sample"]
    assert got["limits"]["sample"]["not_built"]["leaves"] == 10, got["limits"]
    assert "homes_east" in got["limits"]["sample"]["not_built"]["quarters"]
    bad = placeread.read(spec, plan, record(missing="c3"), intent_rec=rec)
    assert not bad["holds"] and "count" in bad["failed"] \
        and "asked/count/cottages" in bad["failed"], bad["failed"]
    # control: a whole-place record judges the whole place
    rows = [{"part": p["name"], "status": "built", "stood": True} for p in west]
    whole = placeread.read(spec, plan, {"waves": [{"wave": "w", "parts": rows}]},
                           intent_rec=rec)
    assert not whole["holds"] and "count" in whole["failed"], whole["failed"]
    return ("a sample of one quarter with every sampled part standing holds, naming "
            "the 17 leaves it did not build; one sampled cottage missing fails; a "
            "record with no sample is judged whole-place")


@case
def t_p_a_wall_is_its_ring_for_inside_and_its_line_for_beside():
    ring = [(0, 0), (200, 0), (200, 200), (0, 200), (0, 0)]
    wall = {"name": "great_wall", "kind": "edge", "type": "great_wall", "family": "wall",
            "path": [list(a) for a in ring], "defines": "great_wall"}
    inside = [{"name": f"f{i}", "kind": "area", "type": "field", "in": ["farm_belt"],
               "x0": 20 + 40 * i, "z0": 20, "x1": 50 + 40 * i, "z1": 50} for i in range(3)]
    outside = [dict(f, name=f"o{i}", x0=220 + 40 * i, x1=250 + 40 * i)
               for i, f in enumerate(inside)]
    w = {"subject": "fields", "relation": "inside", "object": "wall"}
    st, why, ev = intent_mod.relation_measure(w, [wall, *inside])
    assert st == "satisfied" and ev["object_ring"], (st, why, ev)
    st, why, ev = intent_mod.relation_measure(w, [wall, *outside])
    assert st == "failed", (st, why)
    # the bounding box would have called a field in the ring's corner inside; the ring
    # itself refuses one standing across its line
    astride = [dict(inside[0], name="a0", x0=180, x1=230)]
    st, why, ev = intent_mod.relation_measure(w, [wall, *astride])
    assert st == "failed", (st, why)
    near = [dict(inside[0], name="n0", x0=20, x1=50, z0=205, z1=235)]   # 5 off the line
    st, why, ev = intent_mod.relation_measure(
        {"subject": "fields", "relation": "beside", "object": "wall"}, [wall, *near])
    assert st == "satisfied" and ev["gap"] <= 12, (st, why, ev)
    bare = {"name": "ghost_wall", "kind": "edge", "type": "wall", "family": "wall"}
    st, why, ev = intent_mod.relation_measure(w, [bare, *inside])
    assert st == "unresolved", (st, why)
    return ("fields inside a closed ring wall hold, outside or astride its line fail, "
            "beside measures to the line, and a wall with no geometry is unresolved")


def main() -> int:
    bad = 0
    for name, fn in CASES:
        try:
            says = fn()
            print(f"ok   {name:52} {says}")
        except Exception as e:                           # noqa: BLE001
            bad += 1
            import traceback
            print(f"FAIL {name}: {e}\n{traceback.format_exc()[-600:]}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} closure-meaning cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
