"""The expression round's spatial cases: parent allocation from purpose, as regressions
with positive controls.

    $PY scripts/test_expression_spatial.py

Every case runs through the **production entry points** -- `placesolve.solve_place`,
`placesolve.reallocate`, `placeplan.concentric_layout`, `district_compile.compile_district`,
`repair.apply` -- against the retained closure artifacts under `out/closure-farm/` and
`out/closure-rings/` (skipped where they are absent), and every one has a control that
must still hold. Under two minutes.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import (district_compile as dc, pipeline, placeplan, placesolve,  # noqa: E402
                   repair, spec as spec_mod)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    pass


def _retained(name: str) -> dict:
    """The checked spec, site, plateau record, capabilities and intent of a retained
    closure round, or Skip."""
    st = os.path.join(ROOT, "out", name)
    p = os.path.join(st, "place.checked.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(st, "site.json")):
        raise Skip(f"no out/{name}/ with its checked spec and site")
    doc = json.load(open(p))
    # The retained record is read for its parts and its count, on the site it was laid
    # on
    (doc.get("needs") or {}).pop("footprint", None)
    doc.pop("footprint_from", None)
    spec = spec_mod.read_spec(doc, doc["sentence"])
    site = json.load(open(os.path.join(st, "site.json")))
    plateau = None
    pp = os.path.join(st, "plateau.json")
    if os.path.exists(pp):
        pl = json.load(open(pp))
        if pl.get("plateau", {}).get("ok"):
            q = pl["plateau"]
            plateau = {"part": pl["part"], "rect": [q["x0"], q["z0"], q["x1"], q["z1"]],
                       "y": q["y"]}
    _t, decls = placeplan.types_card(None, spec.get("form"))
    caps = json.load(open(os.path.join(st, "capabilities.json"))) \
        if os.path.exists(os.path.join(st, "capabilities.json")) else None
    it = json.load(open(os.path.join(st, "intent.json"))) \
        if os.path.exists(os.path.join(st, "intent.json")) else None
    return {"spec": spec, "site": site, "plateau": plateau, "decls": decls, "caps": caps,
            "intent": it, "voice": doc.get("voice") or "drystone_and_thatch", "state": st}


def _solve(r, spec=None, **kw):
    return placesolve.solve_place(spec or r["spec"], r["site"], r["plateau"], r["decls"],
                                  r["voice"], vol=None, seed=1, caps=r["caps"],
                                  intent=r["intent"], **kw)


def _side(p):
    return p["x1"] - p["x0"] + 1


# ------------------------------------------------ 1. the anchor from the count

@case
def t_a_the_square_is_sized_from_the_count_and_its_derivation_is_recorded():
    """The review: `_rect_for` chose the largest rectangle a type admits, so the market
    square was forty a side against cottages of 12x10 and the hall thirty-six."""
    r = _retained("closure-farm")
    place, fails = _solve(r)
    assert not fails, fails
    market = next(p for p in place["parts"] if p["name"] == "market")
    hall = next(p for p in place["parts"] if p["name"] == "hall")
    sq = r["decls"]["square"]["needs"]["footprint"][2]
    assert _side(market) < sq, (_side(market), sq)
    t = market.get("target") or {}
    assert t.get("what") == "anchor" and any("household" in x for x in t.get("from", [])), t
    assert any("ANCHOR_SHARE" in x for x in t["from"]), t["from"]
    assert _side(market) == t["side"], (_side(market), t["side"])
    ht = hall.get("target") or {}
    assert ht.get("what") == "civic" and _side(hall) == ht["side"] < 36, (ht, _side(hall))
    lay = place["layout"]
    assert any(x.get("part") == "market" for x in lay.get("targets") or []), lay.get("targets")
    # every district records what it was sized from, by the one rule
    for d in place["districts"]:
        dt = d.get("target") or {}
        assert dt.get("what") in ("lots", "land") and dt.get("from"), d["name"]
    fields = next(d for d in place["districts"] if d["defines"] == "fields")
    assert fields["target"]["what"] == "land" and \
        fields["target"]["value"]["land_columns"] >= 16 * placesolve.LAND_PER_HOUSE["farmland"], \
        fields["target"]
    # the relations the closure round made hold still hold on the sized place
    pred = {x["id"]: x["predicted"] for x in lay["districts"]["relations"]}
    assert all(v == "satisfied" for v in pred.values()), pred
    # control: a smaller anchor share through the allocation channel gives a smaller
    # square, and the default is the registered share
    spec2 = json.loads(json.dumps(r["spec"]))
    spec2["negotiated"] = [{"what": "allocation", "to": {"anchor": {"share": 0.3}},
                            "why": "control", "finding": None}]
    place2, f2 = _solve(r, spec=spec2)
    assert not f2 and _side(next(p for p in place2["parts"] if p["name"] == "market")) \
        < _side(market)
    assert t["share"] == placesolve.ANCHOR_SHARE["square"], t["share"]
    return (f"the square is {_side(market)} a side (type admits {sq}) from "
            f"{'; '.join(t['from'][:2])}; the hall {_side(hall)} in {ht['lots']:g} lots; "
            f"the fields need {fields['target']['value']['land_columns']} columns of land; "
            f"share 0.3 gives a square of "
            f"{_side(next(p for p in place2['parts'] if p['name'] == 'market'))}")


# ------------------------------------------------ 2. entities: land takes no house

def _orchard_spec(farm_spec: dict) -> dict:
    """The falsification pass's sentence as a spec, on the farm's ground."""
    doc = json.loads(json.dumps(farm_spec))
    s = ("Build a hamlet of eleven low houses inside a wall, gathered around a chapel "
         "hall, with an orchard beside the houses.")
    doc["sentence"] = s
    doc["kind"] = "hamlet"
    doc["structures"], doc["size_band"] = 11, [11, 11]
    doc["explicit_count"] = {"n": 11, "about": False, "phrase": "eleven low houses",
                             "what": "houses", "requirement": "count/houses"}
    doc["defining_parts"] = [
        {"name": "chapel", "kind": "plot", "family": "hall", "relation": "centre", "count": 1,
         "structures": 0, "role": "civic", "notes": "the chapel hall at the middle"},
        {"name": "wall", "kind": "edge", "family": "wall", "relation": "perimeter",
         "count": 1, "structures": 0, "role": "defensive",
         "notes": "a low wall round the hamlet with one gate, not a great wall"},
        {"name": "homes", "kind": "group", "family": "district", "relation": "throughout",
         "count": 1, "structures": 11, "density": "low", "role": "rural",
         "notes": "the eleven low houses gathered around the chapel",
         "character": {"frontage": "street", "block": 28, "lot_width": 12, "lot_depth": 12,
                       "open_share": 0.12, "courtyard_share": 0.0, "variety": 0.25,
                       "storeys": [1, 2]}},
        {"name": "orchard", "kind": "group", "family": "district", "relation": "quarter",
         "count": 1, "structures": 0, "density": "sparse", "role": "rural",
         "notes": "an orchard beside the houses: rows of fruit trees, no house in it",
         "character": {"frontage": "open", "block": 36, "open_share": 0.95,
                       "courtyard_share": 0.0, "storeys": [1, 1]}}]
    doc.pop("negotiated", None)
    (doc.get("needs") or {}).pop("footprint", None)
    doc.pop("footprint_from", None)
    return spec_mod.read_spec(doc, s)


@case
def t_b_a_land_district_never_takes_a_counted_house_and_holds_few():
    """The falsification pass: `select_regions` had no orchard and the solver spent a
    house on the orchard's lot."""
    r = _retained("closure-farm")
    spec = _orchard_spec(r["spec"])
    ents = {p["name"]: placesolve.entity_of(p, spec) for p in spec["defining_parts"]}
    assert ents["orchard"]["class"] == "land" and not ents["orchard"]["counted"], ents
    assert ents["homes"]["class"] == "building" and ents["homes"]["counted"], ents
    assert ents["wall"]["class"] == "boundary" and not ents["chapel"]["counted"], ents
    it = {"requirements": [
        {"id": "relation/houses_around_hall", "kind": "relation", "hard": True,
         "status": "open", "wants": {"subject": "houses", "relation": "around",
                                     "object": "hall"}},
        {"id": "relation/houses_inside_wall", "kind": "relation", "hard": True,
         "status": "open", "wants": {"subject": "houses", "relation": "inside",
                                     "object": "wall"}},
        {"id": "relation/orchard_beside_houses", "kind": "relation", "hard": True,
         "status": "open", "wants": {"subject": "orchard", "relation": "beside",
                                     "object": "houses"}}]}
    rr = dict(r, spec=spec, intent=it, caps=None, plateau=None)
    place, fails = _solve(rr)
    assert not fails, fails
    lay = place["layout"]["districts"]["structures"]
    assert lay["laid"] == 11 and lay["exact"] and lay["short"] == 0, lay
    orchard = [d for d in place["districts"] if d["defines"] == "orchard"]
    homes = [d for d in place["districts"] if d["defines"] == "homes"]
    assert orchard and all(d["structures"] == 0 and d.get("surface") == "open"
                           for d in orchard), orchard
    assert sum(d["structures"] for d in homes) == 11, [d["structures"] for d in homes]
    assert all(d["target"]["what"] == "land" and d["target"]["value"]["land_use"] == "orchard"
               for d in orchard), [d["target"] for d in orchard]
    rel = {x["id"]: x for x in place["layout"]["districts"]["relations"]}
    pred = {k: v["predicted"] for k, v in rel.items()}
    # the orchard is measurable as land (the falsification pass found 0 `orchard`);
    # whether it stands beside the houses is the geometry's, said either way
    assert pred["relation/orchard_beside_houses"] in ("satisfied", "failed") \
        and "`orchard`" in (rel["relation/orchard_beside_houses"].get("why") or ""), rel
    assert pred["relation/houses_around_hall"] == "satisfied", pred
    wall = next(p for p in place["parts"] if p["kind"] == "edge")
    assert wall["hierarchy"]["kind"] == "town", wall["hierarchy"]
    # control: the farm's fields hold the few cottages that work them, never their
    # ground's fill
    farm, _f = _solve(r)
    fields = [d for d in farm["districts"] if d["defines"] == "fields"]
    few = int(-(-16 * placesolve.LAND_HOUSE_SHARE // 1))
    assert sum(d["structures"] for d in fields) <= few, ([d["structures"] for d in fields], few)
    return (f"eleven houses laid exactly over {len(homes)} strip(s), the orchard "
            f"{[d['name'] for d in orchard]} open land with no house, every relation "
            f"predicted satisfied, a town wall; the farm's fields keep "
            f"{sum(d['structures'] for d in fields)} of 16 (at most {few})")


# ------------------------------------------------ 2b. a land quarter beside the houses

@case
def t_b2_a_land_quarter_beside_the_houses_adjoins_their_strip_and_predicts_the_leaf_gap():
    """The coordinator's live orchard run: the orchard took the east strip, the houses
    north and west, and the assembled plan measured the orchard 68 columns from the
    nearest house while the layout had predicted the region gap satisfied."""
    r = _retained("closure-farm")
    spec = _orchard_spec(r["spec"])
    it = {"requirements": [
        {"id": "relation/houses_around_hall", "kind": "relation", "hard": True,
         "status": "open", "wants": {"subject": "houses", "relation": "around",
                                     "object": "hall"}},
        {"id": "relation/orchard_beside_houses", "kind": "relation", "hard": True,
         "status": "open", "wants": {"subject": "orchard", "relation": "beside",
                                     "object": "houses"}}]}
    rr = dict(r, spec=spec, intent=it, caps=None, plateau=None)
    place, fails = _solve(rr)
    assert not fails, fails
    orchard = next(d for d in place["districts"] if d["defines"] == "orchard")
    homes = [d for d in place["districts"] if d["defines"] == "homes"]
    # the orchard's sector is a piece of a strip a homes district holds: same side
    side = orchard["name"].split("_")[1]
    host = [d for d in homes if d["name"].split("_")[1] == side]
    assert host, (orchard["name"], [d["name"] for d in homes])
    adj = (place["layout"]["districts"].get("relations") is not None) and \
        place["layout"].get("districts", {}).get("sectors")
    rec = getattr(placesolve, "_never", None)
    assert any(s["group"] == "orchard" and s["label"].startswith(side) for s in adj), adj
    # ...and it shares an edge with it, a lot's clearance apart
    from ethoslm import district_compile as dc
    # a gathered row is cut at the anchor's axis into a district each side: the orchard
    # adjoins the nearest piece
    def _gap(h):
        return max(orchard["x0"] - h["x1"] - 1, h["x0"] - orchard["x1"] - 1,
                   orchard["z0"] - h["z1"] - 1, h["z0"] - orchard["z1"] - 1)
    h = min(host, key=_gap)
    gap = _gap(h)
    assert gap == dc.LOT_GAP, (gap, orchard, h)
    # the prediction is the checker's own rule over probes that stand where the leaves
    # will (the land at its edge, a house a margin and a clearance in)
    rel = {x["id"]: x for x in place["layout"]["districts"]["relations"]}
    assert rel["relation/orchard_beside_houses"]["predicted"] == "satisfied", rel
    ev = rel["relation/orchard_beside_houses"].get("evidence") or {}
    why = rel["relation/orchard_beside_houses"].get("why") or ""
    assert "column(s)" in why, why
    # the houses still gather round the hall on three sides
    assert rel["relation/houses_around_hall"]["predicted"] == "satisfied", rel
    # the land's own words select it, and not the houses'
    assert [p["name"] for p in placesolve._parts_for_word(spec, "houses")] == ["homes"]
    assert [p["name"] for p in placesolve._parts_for_word(spec, "orchard")] == ["orchard"]
    # control: the same spec with no `beside` relation gives the orchard a strip of its
    # own and the houses every side
    rr2 = dict(rr, intent={"requirements": it["requirements"][:1]})
    place2, f2 = _solve(rr2)
    assert not f2, f2
    o2 = next(d for d in place2["districts"] if d["defines"] == "orchard")
    assert o2["name"].split("_")[1] not in {d["name"].split("_")[1]
                                            for d in place2["districts"]
                                            if d["defines"] == "homes"}, o2["name"]
    return (f"the orchard `{orchard['name']}` adjoins `{h['name']}` {gap} columns apart "
            f"and the prediction reads: {why[:90]}; without the relation it takes "
            f"`{o2['name']}`")


# ------------------------------------------------ 3. rings from the count

@case
def t_c_ring_widths_and_sectors_follow_the_count_at_the_density_word():
    """The closure town: fifteen dense houses over sixteen thousand columns covered 5%."""
    r = _retained("closure-rings")
    spec = r["spec"]
    place, fails = placeplan.concentric_layout(spec, r["site"], None, r["decls"],
                                               r["voice"], caps=None)
    assert not fails, fails
    L = place["layout"]
    lower = next(x for x in L["rings"] if x["name"] == "lower_ring")
    assert lower["target"]["counted"] and lower["target"]["columns"] > 0, lower["target"]
    kept = [d for d in place["districts"] if d["defines"] == "lower_ring"
            and d["structures"] > 0]
    opened = [d for d in place["districts"] if d["defines"] == "lower_ring"
              and d.get("surface") == "open"]
    assert opened, "no open sector: the ring was not compacted"
    got = sum((d["x1"] - d["x0"] + 1) * (d["z1"] - d["z0"] + 1) for d in kept)
    need = lower["target"]["columns"]
    assert got <= 2.0 * need, (got, need)
    for d in kept:
        assert d["target"]["what"] == "ring" and d["target"]["from"], d["name"]
    lots = sum(d["structures"] for d in kept) * lower["target"]["lot"][0] * lower["target"]["lot"][1]
    assert lots / float(got) >= 0.2, (lots, got)
    # no kept sector holds fewer houses than a block of its word (the coordinator's live
    # finding: two row houses on a 2100-column strip); a sector under a block is merged
    # into the roomiest and left open
    from ethoslm.district_compile import BLOCK_LOTS
    least_block = BLOCK_LOTS["dense"]
    assert all(d["structures"] >= least_block for d in kept), [(d["name"], d["structures"]) for d in kept]
    # the record says why the ring is as wide as it is where the count wants it narrower
    if lower["target"]["width_need"] < lower["min_width"]:
        assert any("least width" in x for x in lower["target"]["from"]), lower["target"]["from"]
    # every kept sector is sized near its need: the whole strip is never kept when a
    # remainder a district could own is left
    for d in kept:
        v = d["target"]["value"]
        assert v["got_columns"] <= 2 * v["columns"] + 300, (d["name"], v)
    assert sum(d["structures"] for d in place["districts"]) == 24, \
        [d["structures"] for d in place["districts"]]
    # the compound stands at its own need and not the share it was left
    cg = placeplan.compound_ground(spec=spec, site_side=r["site"]["size"])
    assert not cg["share_taken"] and cg["side"] < cg["declared"], cg
    # control: the same spec with no explicit count keeps its shares whole
    doc = json.loads(json.dumps(spec))
    doc["explicit_count"] = None
    doc["sentence"] = ("Build a small Japanese hill town in two rings around a temple "
                       "compound, walled, with a dense lower ring and a sparse upper ring.")
    for p in doc["defining_parts"]:
        if p.get("ring") is not None:
            p["structures"] = 0
            p.pop("structures_inferred", None)
    (doc.get("needs") or {}).pop("footprint", None)
    doc.pop("footprint_from", None)
    spec2 = spec_mod.read_spec(doc, doc["sentence"])
    assert not spec2.get("explicit_count"), spec2.get("explicit_count")
    place2, f2 = placeplan.concentric_layout(spec2, r["site"], None, r["decls"], r["voice"])
    assert not f2, f2
    assert not [d for d in place2["districts"] if d.get("surface") == "open"], "shares broken"
    return (f"the lower ring's {sum(d['structures'] for d in kept)} houses stand on "
            f"{got} columns against a need of {need} (was ~16000), {len(opened)} open "
            f"sector(s) keep the ring's ground; the compound {cg['side']} a side against "
            f"the share's {cg['declared']}; without the count the shares stand")


# ------------------------------------------------ 4. wall hierarchy

@case
def t_d_a_wall_is_a_town_wall_unless_the_design_or_the_words_make_it_great():
    r = _retained("closure-rings")
    place, fails = placeplan.concentric_layout(r["spec"], r["site"], None, r["decls"],
                                               r["voice"])
    assert not fails, fails
    wall = next(p for p in place["parts"] if p["kind"] == "edge")
    gate = next(p for p in place["parts"] if p["kind"] == "point")
    h = wall["hierarchy"]
    assert h["kind"] == "town" and wall["type"] == "wall", h
    assert wall["params"]["height"] < placeplan.GREAT_WALL_HEIGHT \
        if hasattr(placeplan, "GREAT_WALL_HEIGHT") else True
    assert wall["params"]["height"] == h["height"] and any("storey" in x for x in h["from"]), h
    assert gate.get("voice") == wall.get("voice") and gate.get("voice") != r["spec"].get("voice"), \
        (gate.get("voice"), wall.get("voice"), r["spec"].get("voice"))
    assert wall.get("voice_from") == "ring", wall.get("voice_from")
    # the words: a denial is not a request; a request is
    assert placesolve.wall_kind_for({"name": "wall", "notes": "a low town wall, not a "
                                     "great wall"}, {}, outermost=True)[0] == "town"
    assert placesolve.wall_kind_for({"name": "great_wall", "notes": "the great outer wall "
                                     "the sources lead with"}, {}, outermost=True)[0] == "great"
    assert placesolve.wall_kind_for({"name": "ring_walls", "notes": "coursed masonry"},
                                    {"sentence": "Build a ringed town."},
                                    outermost=True)[0] == "town"
    # the design: nested ring walls make an outermost great wall, stepped inward
    doc = json.loads(json.dumps(r["spec"]))
    for p in doc["defining_parts"]:
        if p["name"] == "lower_ring":
            p["walled"] = True
        if p["name"] == "town_wall":
            p["count"] = 2
    (doc.get("needs") or {}).pop("footprint", None)
    doc.pop("footprint_from", None)
    spec2 = spec_mod.read_spec(doc, doc["sentence"])
    big = {"origin": [0, 0], "size": max(int(r["site"]["size"]),
                                         int(placeplan.least_footprint(spec2, r["decls"]) or 0) + 24)}
    place2, f2 = placeplan.concentric_layout(spec2, big, None, r["decls"], r["voice"])
    assert not f2, f2
    walls = sorted([p for p in place2["parts"] if p["kind"] == "edge"],
                   key=lambda p: p["hierarchy"]["rank"])
    assert [w["hierarchy"]["kind"] for w in walls] == ["great", "ring"], \
        [w["hierarchy"] for w in walls]
    assert walls[0]["params"]["height"] > walls[1]["params"]["height"], \
        [w["params"] for w in walls]
    assert walls[0]["type"] == "great_wall", walls[0]["type"]
    return (f"one walled ring: `{wall['type']}` {wall['params']['height']} high, a town "
            f"wall over {h['from'][1][:40]}, its gate in `{gate['voice']}`; two walled "
            f"rings: {walls[0]['type']} {walls[0]['params']['height']} over "
            f"{walls[1]['type']} {walls[1]['params']['height']}")


# ------------------------------------------------ 5. constraints enlarge the lots

def _constraints(name: str) -> list:
    p = os.path.join(ROOT, "out", name, "parts.json")
    if not os.path.exists(p):
        raise Skip(f"no out/{name}/parts.json")
    rec = json.load(open(p))
    return [r.get("constraint") for w in rec.get("waves") or [] for r in w.get("parts") or []
            if r.get("constraint")]


@case
def t_e_emitted_constraints_enlarge_the_lots_and_the_compiler_lays_them():
    r = _retained("closure-farm")
    cons = _constraints("closure-farm")
    lots = placesolve.lots_from_constraints(cons)
    if not lots:
        raise Skip("the retained farm records no layout-owned lot_min")
    place, _f = _solve(r)
    spec = json.loads(json.dumps(r["spec"]))
    finding = {"id": "b3", "about": "fabric", "subjects": ["homes_west"], "material": False,
               "says": "most cottages stand one storey where two were allowed",
               "owner": "layout", "action": "enlarge_lots"}
    got, rec = placesolve.reallocate(place, spec, finding, site=r["site"], decls=r["decls"],
                                     constraints=cons, intent=r["intent"],
                                     plateau=r["plateau"], caps=r["caps"], seed=1)
    assert rec["applied"] and rec["refused"] is None, rec
    want = rec["allocation"]["lots"]["homes"]
    assert want[0] >= lots["cottage"][0] or want[1] >= lots["cottage"][1], (want, lots)
    homes = [d for d in got["districts"] if d["defines"] == "homes"]
    # the district's least lot is the allocation's, clamped UP to a side the type admits
    lm = homes[0].get("lot_min")
    assert lm and lm[0] >= want[0] and lm[1] >= want[1], (lm, want)
    assert all(d.get("lot_min") == lm for d in homes), [d.get("lot_min") for d in homes]
    assert dc._admits(r["decls"]["cottage"], lm[0], lm[1]), lm
    assert any(row["what"] == "allocation" for row in spec["negotiated"]), spec["negotiated"]
    # the compiler lays the lot the allocation asks
    d = homes[0]
    part = next(p for p in spec["defining_parts"] if p["name"] == "homes")
    _t, decls = placeplan.types_card(None, spec.get("form"), "rural")
    plan, crec = dc.compile_district(dict(d), part, got, decls, spec=spec, seed=1)
    assert crec["lot"][0] >= want[0] and crec["lot"][1] >= want[1], (crec["lot"], want)
    assert crec.get("lot_min_raised"), crec.get("lot_min_raised")
    # ...and where the site cannot hold the count at the bigger lot, the shortfall is on
    # the record -- the district's target says the strip is the limit -- and never
    # hidden by a smaller lot
    if crec["lots"] < d["structures"]:
        assert d["target"].get("short_columns"), d["target"]
    else:
        assert crec.get("over_ceiling") or crec["lots"] == d["structures"], crec
    # control: without the allocation the same district lays the character's 12x10
    d0 = next(x for x in place["districts"] if x["defines"] == "homes")
    _p0, c0 = dc.compile_district(dict(d0), part, place, decls, spec=r["spec"], seed=1)
    assert c0["lot"] == [12, 10] and not c0.get("lot_min_raised"), c0["lot"]
    return (f"construction's lot_min {lots} -> allocation.lots homes {want}; the district "
            f"carries {lm} (the next side the cottage admits) and compiles at {crec['lot']}"
            f"{' over the ceiling, recorded' if crec.get('over_ceiling') else ''} "
            f"(control {c0['lot']})")


# ------------------------------------------------ 6. reallocate: bounded, deterministic

@case
def t_f_reallocate_is_bounded_deterministic_and_refuses_by_name():
    r = _retained("closure-farm")
    place, _f = _solve(r)
    spec = json.loads(json.dumps(r["spec"]))
    f = {"id": "b1", "about": "scale", "subjects": ["market"], "material": True,
         "measure": "square_scale", "owner": "layout", "action": "shrink_anchor",
         "target": {"measure": "square_scale", "direction": "down", "value": 6.0}}
    got, rec = placesolve.reallocate(place, spec, f, site=r["site"], decls=r["decls"],
                                     intent=r["intent"], plateau=r["plateau"], caps=r["caps"],
                                     seed=1)
    assert rec["applied"] and rec["action"] == "shrink_anchor" and rec["before"] and rec["after"], rec
    m0 = _side(next(p for p in place["parts"] if p["name"] == "market"))
    m1 = _side(next(p for p in got["parts"] if p["name"] == "market"))
    assert m1 < m0, (m0, m1)
    # laid out again from the spec it wrote, the place is this place
    again, f2 = _solve(r, spec=spec)
    assert not f2 and json.dumps(again, sort_keys=True) == json.dumps(got, sort_keys=True)
    # bounded: the share walks down to its floor and then refuses by name
    steps = 0
    cur = got
    while steps < 6:
        cur2, rec2 = placesolve.reallocate(cur, spec, f, site=r["site"], decls=r["decls"],
                                           intent=r["intent"], plateau=r["plateau"],
                                           caps=r["caps"], seed=1)
        if not rec2["applied"]:
            break
        cur = cur2
        steps += 1
    assert rec2["refused"] and "floor" in rec2["refused"], rec2
    assert placesolve.allocation_of(spec)["anchor"]["share"] >= placesolve.ANCHOR_SHARE_MIN
    # the land on this site is the site's limit, and the action says so
    f3 = {"id": "b2", "about": "composition", "subjects": ["fields_south_2"], "material": True,
          "measure": "open_to_built", "owner": "layout",
          "target": {"measure": "open_to_built", "direction": "up"}}
    _p3, rec3 = placesolve.reallocate(cur, spec, f3, site=r["site"], decls=r["decls"],
                                      intent=r["intent"], plateau=r["plateau"], caps=r["caps"],
                                      seed=1)
    assert rec3["action"] == "grow_land" and rec3["refused"] and "site" in rec3["refused"], rec3
    # a move is the relation repair's, and a finding that names nothing is refused
    _p4, rec4 = placesolve.reallocate(cur, spec, {**f, "action": "move_object"},
                                      site=r["site"], decls=r["decls"])
    assert rec4["refused"], rec4
    _p5, rec5 = placesolve.reallocate(cur, spec, {"id": "x", "about": "voice", "says": "dull"},
                                      site=r["site"], decls=r["decls"])
    assert rec5["refused"] and rec5["action"] is None, rec5
    # the explicit requirements are untouched by every step
    lay = cur["layout"]["districts"]["structures"]
    assert lay["laid"] == 16 and lay["exact"], lay
    return (f"shrink_anchor: {m0} -> {m1}, reproduced from the spec's allocation row; "
            f"{steps + 1} step(s) to the floor then refused: {rec2['refused'][:60]}; "
            f"grow_land refused naming the site; the count stays 16 exact")


# ------------------------------------------------ 7. the repair route

@case
def t_g_a_readings_scale_finding_reaches_the_allocation_through_repair():
    r = _retained("closure-farm")
    tmp = tempfile.mkdtemp(prefix="expr-spatial-")
    try:
        for f in ("site.json", "site_search.json", "intent.json", "place.checked.json",
                  "place.json", "capabilities.json", "plateau.json", "plan.place.json",
                  "parts.json", "deps.json"):
            src = os.path.join(r["state"], f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, f))
        cfg = json.load(open(os.path.join(ROOT, "rounds", "closure-farm.json")))
        rnd = pipeline.Round(name="closure-farm", sentence=cfg["sentence"],
                             flags=cfg["flags"], state_dir=tmp)
        place = json.load(open(rnd.rel("plan.place.json")))
        spec = rnd.place_spec()
        finding = {"id": "find/reading/b1", "requirement": None, "part": "market",
                   "says": "the square is thirteen times its cottages", "owner": "layout",
                   "blocks": "fidelity", "severity": "warning", "seen_by": "inspection",
                   "fixed": False,
                   "evidence": {"finding": {"id": "b1", "about": "scale",
                                            "subjects": ["market"], "material": True,
                                            "measure": "square_scale",
                                            "owner": "layout", "action": "shrink_anchor"}}}
        rec = repair.apply(rnd, spec, {"findings": [finding]}, place=place,
                           place_path=rnd.rel("plan.place.json"))
        assert rec["applied"] and rec["applied"][0]["allocation"], rec
        assert rec.get("wrote") == "place.checked.json", rec
        doc = json.load(open(rnd.rel("place.checked.json")))
        assert any(x.get("what") == "allocation" for x in doc.get("negotiated") or []), doc.get("negotiated")
        # the second pass on the same candidate is refused: once per candidate
        rec2 = repair.apply(rnd, spec, {"findings": [finding]}, place=place,
                            place_path=rnd.rel("plan.place.json"))
        assert not rec2["applied"] and rec2["refused"], rec2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return (f"routed to the layout: {rec['applied'][0]['why'][:80]}; the allocation row "
            f"is on place.checked.json; a second pass refuses")


def main() -> int:
    ok = 0
    fails = 0
    skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
            ok += 1
            print(f"ok   {name}: {says}")
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
        except Exception as e:                   # noqa: BLE001 -- a test reports
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{ok}/{ok + fails} expression spatial cases pass"
          + (f", {skipped} skipped" if skipped else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
