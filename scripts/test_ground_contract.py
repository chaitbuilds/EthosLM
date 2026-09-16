"""v2, B1: the ground contract. Every part declares what it needs of the ground, one
resolver settles every column before a block is placed, the seams are derived from
the drop, and the ground is read-only afterwards.

    $PY scripts/test_ground_contract.py

What is proved here, each on a fixture and all of it in a few minutes:

1. precedence is total: a column two declarations claim goes to the higher class, then
to the earlier declaration, and every column is owned once; 2. seams are derived from
the drop and nothing can declare one; 3. the resolution is read-only; 4. 5. a bare
`site()` and a `site()` on the build's contract lay byte-identical pending sets, on the
terrain bank's fixtures -- one resolver, two ways in; 6. a parts stage settles the
ground once, before its first part, and every part stands at the level the contract
settled; 7. the ring fixture's terraces and the podium are declarations of the same
contract, laid at the levels it settled, with the step between rings a face.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                                    # noqa: E402

from ethoslm import ground, offline, pipeline                           # noqa: E402
from ethoslm.buildlib import Builder                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


MAT = {"wall": "sandstone", "footing": "granite", "frame": "dark_oak",
       "roof": "oxidized_copper", "trim": "red_sandstone", "floor": "smooth_stone"}


# ------------------------------------------------------------------ 1. precedence

@case
def t_1_precedence_is_total_and_every_column_is_owned_once():
    c = ground.Contract()
    flat = lambda x, z: 60                                            # noqa: E731
    dry = lambda x, z: None                                           # noqa: E731
    # a verge declared first, a street over the same columns, a footprint over part of
    # both, and a field over everything: the order of declaration is not the order of
    # precedence
    c.platform("verge", (0, 0, 9, 9), 61, cls="verge")
    c.profile("lane", {(x, 5): 60 for x in range(0, 10)}, cls="street")
    c.platform("house", (2, 2, 6, 6), 62, cls="footprint")
    c.platform("meadow", (0, 0, 9, 9), 60, cls="field")
    c.platform("plaza", (5, 5, 9, 9), 61, cls="designed")
    r = c.resolve(flat, dry)
    assert ground.PRECEDENCE == ("water", "footprint", "designed", "street", "doorstep",
                                 "field", "verge")
    assert r.owner(3, 3) == "house" and r.level(3, 3) == 62
    assert r.owner(3, 5) == "house", r.owner(3, 5)                    # footprint over street
    assert r.owner(8, 5) == "plaza" and r.level(8, 5) == 61           # designed over street
    assert r.owner(1, 5) == "lane" and r.level(1, 5) == 60            # street over field, verge
    assert r.owner(8, 8) == "plaza" and r.level(8, 8) == 61           # designed over field, verge
    assert r.owner(0, 0) == "meadow" and r.level(0, 0) == 60          # field over verge
    assert r.owner(6, 6) == "house"                                   # earlier footprint
    owned = {(x, z) for x in range(10) for z in range(10)}
    assert len(r) == 100 and all(r.owner(*c) for c in owned)
    counts = {}
    for c_ in owned:
        counts[r.owner(*c_)] = counts.get(r.owner(*c_), 0) + 1
    assert counts == {"house": 25, "lane": 2, "plaza": 21, "meadow": 52}, counts
    assert sum(len(r.columns_of(n)) for n in counts) == 100
    rec = r.record()
    assert rec["owned_by_class"] == {"footprint": 25, "street": 2, "designed": 21,
                                     "field": 52}, rec["owned_by_class"]
    assert [d["label"] for d in rec["declarations"]] == ["verge", "lane", "house",
                                                          "meadow", "plaza"]
    assert rec["declarations"][0]["won"] == 0 and rec["registered"]["PRECEDENCE"]
    # a class or a kind outside the vocabulary is refused by name
    for bad in (lambda: c.platform("x", (0, 0, 1, 1), 60, cls="lawn"),
                lambda: ground.Declaration("x", "seam", "verge", {})):
        try:
            bad()
        except ValueError as e:
            assert "one of" in str(e), e
        else:
            raise AssertionError("an unknown class or kind was accepted")
    return ("five declarations over 100 columns: the footprint takes the street's "
            "columns, the plaza takes the street's, the street the field's; owned by "
            f"class {rec['owned_by_class']}")


# ------------------------------------------------------------------ 2. seams

@case
def t_2_seams_are_derived_from_the_drop_and_nothing_declares_one():
    c = ground.Contract()
    flat = lambda x, z: 60                                            # noqa: E731
    c.platform("kerb", (0, 0, 3, 3), 61, cls="designed")              # one over the ground
    c.platform("bank", (10, 0, 13, 3), 63, cls="designed")            # three over
    c.platform("face", (20, 0, 23, 3), 70, cls="designed")            # ten over
    c.platform("upper", (24, 0, 27, 3), 74, cls="designed")           # four over `face`
    r = c.resolve(flat, None)
    s = {tuple(k): dict(v) for k, v in r.seams.items()}
    assert s[("kerb", "ground")] == {"kerb": 16, "bank": 0, "face": 0}, s
    assert s[("bank", "ground")] == {"kerb": 0, "bank": 16, "face": 0}, s
    assert s[("face", "ground")] == {"kerb": 0, "bank": 0, "face": 12}, s
    assert s[("face", "upper")] == {"kerb": 0, "bank": 0, "face": 4}, s
    assert s[("upper", "ground")]["face"] == 12
    assert ground.seam_kind(0) is None and ground.seam_kind(1) == "kerb"
    assert ground.seam_kind(-3) == "bank" and ground.seam_kind(4) == "face"
    assert (ground.SEAM_KERB, ground.SEAM_BANK) == (1, 3)
    # nothing declares a seam: the contract has no such call and a declaration of that
    # kind is refused by name
    assert not hasattr(c, "seam") and "seam" not in ground.KINDS
    rec = r.record()
    assert rec["seam_totals"] == {"kerb": 16, "bank": 16, "face": 28}, rec["seam_totals"]
    return (f"four platforms over flat ground: a rise of one is 16 kerbs, of three 16 "
            f"banks, of ten 12 faces, and the step between two platforms is a face; "
            f"totals {rec['seam_totals']}")


# ------------------------------------------------------------------ 3. read-only

@case
def t_3_the_resolution_is_read_only():
    c = ground.Contract()
    d = c.platform("pad", (0, 0, 2, 2), 61, cls="footprint")
    r = c.resolve(lambda x, z: 60, None)
    refused = 0
    for attempt in (lambda: setattr(r, "relief", 9),
                    lambda: setattr(r, "surface", None),
                    lambda: delattr(r, "relief"),
                    lambda: r.seams.__setitem__(("a", "b"), {}),
                    lambda: d.columns.__setitem__((0, 0), 99),
                    lambda: setattr(d, "level", 99)):
        try:
            attempt()
        except (AttributeError, TypeError):
            refused += 1
    assert refused == 6, refused
    cols = r.columns_of("pad")
    cols[(0, 0)] = 99                                                 # a copy
    assert r.level(0, 0) == 61
    # ...and it survives a process boundary and the disk as the same thing
    import pickle
    r2 = pickle.loads(pickle.dumps(r))
    with tempfile.TemporaryDirectory() as tmp:
        p = r.save(os.path.join(tmp, "g.npz"))
        r3 = ground.load(p)
    for q in (r2, r3):
        assert len(q) == 9 and q.owner(1, 1) == "pad" and q.level_of("pad") == 61
        assert q.record() == r.record()
    return ("six writes refused; the columns a reader gets are a copy; pickled and "
            "saved, the resolution reads back identical")


# ------------------------------------------------------------------ 4. the court at
# y=38

def _slope(X, Z, S, *, base=56, relief=6, margin=8):
    import test_site_needs
    return test_site_needs._volume(X, Z, S, base=base, relief=relief, margin=margin)


def _builder(vol, net=None):
    from ethoslm.frontage import Frontage
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    if net is not None:
        b.frontage = Frontage(vol, net)
    return b


@case
def t_4_a_platform_is_held_within_reach_of_its_own_ground_whoever_declared_it():
    from ethoslm.circulate import Network, Threshold
    X, Z, S = 1000, 2000, 64
    vol = _slope(X, Z, S)                                             # ground 56..62
    cells = {(x, Z + 40): {"y": 38, "rank": 0, "face": None} for x in range(X + 10, X + 50)}
    net = Network(cells, [Threshold("court", X + 30, Z + 40, 38, "north",
                                    (X + 30, 39, Z + 38)),
                          Threshold("green", X + 20, Z + 40, 38, "north",
                                    (X + 20, 39, Z + 38)),
                          Threshold("gate", X + 40, Z + 40, 38, "north",
                                    (X + 40, 39, Z + 38))])
    b = _builder(vol, net)
    # (a) the library's own declaration is within reach for a plot, an area and a point
    c = ground.Contract()
    plot = b.declare({"label": "court", "kind": "plot", "x0": X + 20, "z0": Z + 20,
                      "x1": X + 40, "z1": Z + 38}, mat=MAT, contract=c)
    area = b.declare({"label": "green", "kind": "area", "type": "garden", "x0": X + 8,
                      "z0": Z + 24, "x1": X + 17, "z1": Z + 36}, mat=MAT, contract=c)
    point = b.declare({"label": "gate", "kind": "point", "at": [X + 46, Z + 30],
                       "size": 5}, mat=MAT, contract=c)
    for dec in (plot, area, point):
        lo, hi = dec["grade"]
        assert dec["want"] == 38 and lo <= dec["level"] <= hi, dec
    assert c.declarations[1].cls == "field" and c.declarations[2].cls == "footprint"
    found = ground.Found(vol)
    r = c.resolve(found.bed, found.wet, relief=Builder.SITE_RELIEF)
    for name, dec in (("court", plot), ("green", area), ("gate", point)):
        assert r.level_of(name) == dec["level"] and r.clamped(name) is None, name
    # (b) a declaration that lies -- the recorded y=38 handed straight to the contract
    # -- is held to the ground under it by the resolver, by construction
    lie = ground.Contract()
    lie.platform("court", tuple(plot["rect"]), 38, cls="footprint")
    lie.platform("green", tuple(area["rect"]), 38, cls="field")
    lie.platform("gate", tuple(point["rect"]), 90, cls="footprint")
    held = lie.resolve(found.bed, found.wet, relief=Builder.SITE_RELIEF)
    for name, dec in (("court", plot), ("green", area), ("gate", point)):
        lo, hi = dec["grade"]
        got = held.level_of(name)
        assert lo <= got <= hi, (name, got, dec["grade"])
        note = held.clamped(name)
        assert note and note["asked"] in (38, 90) and note["got"] == got, note
    # (c) ...and a designed platform is not held: a terrace stands where it was designed
    des = ground.Contract()
    des.platform("terrace", (X + 20, Z + 20, X + 40, Z + 38), 80, cls="designed")
    assert des.resolve(found.bed, found.wet).level_of("terrace") == 80
    # (d) the resolution of a part does not depend on its neighbour's: the same
    # declaration resolves to the same level with and without a platform at y=38 beside
    # it, and with one at y=90
    alone = ground.Contract()
    b.declare({"label": "court", "kind": "plot", "x0": X + 20, "z0": Z + 20,
               "x1": X + 40, "z1": Z + 38}, mat=MAT, contract=alone)
    l_alone = alone.resolve(found.bed, found.wet).level_of("court")
    for ny in (38, 90):
        with_n = ground.Contract()
        with_n.platform("neighbour", (X + 8, Z + 20, X + 17, Z + 38), ny, cls="designed")
        b.declare({"label": "court", "kind": "plot", "x0": X + 20, "z0": Z + 20,
                   "x1": X + 40, "z1": Z + 38}, mat=MAT, contract=with_n)
        rr = with_n.resolve(found.bed, found.wet)
        assert rr.level_of("court") == l_alone, (ny, rr.level_of("court"), l_alone)
        assert rr.level_of("neighbour") == ny
    return (f"a doorstep recorded at y=38 under ground {plot['grade']}: the plot, the "
            f"area and the point declare {plot['level']}, {area['level']} and "
            f"{point['level']}; the same declared at 38, 38 and 90 are held to "
            f"{held.level_of('court')}, {held.level_of('green')} and "
            f"{held.level_of('gate')} by the resolver; a neighbour at 38 or 90 moves "
            f"the plot's level by 0")


# ------------------------------------------------------------------ 5. one resolver,
# two ways in

@case
def t_5_a_bare_site_and_a_contract_site_lay_byte_identical_pending_sets():
    import test_ground
    from ethoslm.frontage import Frontage
    bank = test_ground.terrain_bank.load()["fixtures"]
    picks = [i for i, f in enumerate(bank)
             if any(w in f["plot"]["label"] for w in ("round9_wettest", "round11_steepest",
                                                        "round7_flattest", "round8_seeded"))]
    assert len(picks) >= 3, picks
    same = 0
    lines = []
    t0 = time.perf_counter()
    for key in picks:
        base, net, plot = test_ground._ground(key)
        # the bare way: `site()` declares this one part and resolves it
        b1 = Builder(offline.OfflineSite(base))
        b1._vol = base
        b1.frontage = Frontage(base, net)
        b1.registry = test_ground._Registry(plot)
        p1 = b1.site(dict(plot, kind="plot"), mat=test_ground.MAT)
        # the build's way: the contract settled before, handed to the builder
        c = ground.Contract()
        b0 = Builder(offline.OfflineSite(base))
        b0._vol = base
        b0.frontage = Frontage(base, net)
        b0.declare(dict(plot, kind="plot", label=plot["label"]), mat=test_ground.MAT,
                   contract=c)
        c.network(net)
        found = ground.Found(base)
        res = c.resolve(found.bed, found.wet, relief=Builder.SITE_RELIEF,
                        surface=found.surface)
        b2 = Builder(offline.OfflineSite(base, heights=found.surface.h))
        b2._vol = base
        b2.frontage = Frontage(base, net)
        b2.registry = test_ground._Registry(plot)
        b2.ground = res
        p2 = b2.site(dict(plot, kind="plot"), mat=test_ground.MAT)
        assert p1["sited"]["contract"]["settled_by"].startswith("a contract of this one")
        assert p2["sited"]["contract"]["settled_by"] == "the build's contract"
        assert (p1["floor_y"], p1["footprint"], p1["ground"]) == \
            (p2["floor_y"], p2["footprint"], p2["ground"]), (p1["sited"], p2["sited"])
        assert b1._pending == b2._pending, plot["label"]
        assert res.owner(*p2["door"]) in (plot["label"], f"{plot['label']}/doorstep",
                                          "lanes"), res.owner(*p2["door"])
        same += 1
        lines.append(f"{plot['label']} {p2['ground']} y={p2['floor_y']} "
                     f"{len(b2._pending)} blocks")
    return (f"{same} fixtures, both ways byte-identical in {time.perf_counter() - t0:.0f}s: "
            + "; ".join(lines))


# ------------------------------------------------------------------ 6. the parts stage

@case
def t_6_a_parts_stage_settles_the_ground_once_and_every_part_stands_on_it():
    import test_site_needs
    from ethoslm.pipeline import stages_build
    with tempfile.TemporaryDirectory() as tmp:
        rnd, be = test_site_needs._two_district_state(tmp, 1)
        t0 = time.perf_counter()
        res = stages_build.stage_parts(rnd, be, {})
        secs = time.perf_counter() - t0
        assert res.get("built") == 8 and not res.get("failed"), res.get("failed")
        assert os.path.exists(os.path.join(tmp, "ground.json"))
        assert os.path.exists(os.path.join(tmp, "ground.npz"))
        rec = json.load(open(os.path.join(tmp, "ground.json")))
        settled = ground.load(os.path.join(tmp, "ground.npz"))
        assert rec["declared"] == 8 and rec["refused"] == 0, rec
        assert res["ground"]["columns"] == rec["columns"] == len(settled)
        rows = [r for w in res["waves"] for r in w["parts"]]
        for r in rows:
            d = settled.declaration(r["part"])
            assert d is not None and d.cls == "footprint" and d.kind == "platform"
            assert r["floor_y"] == settled.level_of(r["part"]), (r["part"], r["floor_y"])
            assert r["sited"].startswith("plinth") or r["sited"].startswith("platform")
        # the record says every part's decision, and the classes the columns went to
        assert set(rec["parts"]) == {r["part"] for r in rows}
        assert set(rec["owned_by_class"]) == {"footprint"}, rec["owned_by_class"]
        # ...and settled again on the same ground it is the same resolution
        rec2 = stages_build.settle_ground(rnd, pipeline.OfflineBackend(rnd, dry_run=True),
                                          rnd.parts(), lambda p: pipeline.voice_palette(
                                              "white_render_dark_frame"))[1]
        assert rec2["declarations"] == rec["declarations"]
    return (f"8 cottages in two quarters: {rec['columns']} columns settled in "
            f"{rec['seconds']}s before the first part, every part at its settled level, "
            f"the stage {secs:.0f}s; seams {rec['seam_totals']}")


# ------------------------------------------------------------------ 7. terraces and the
# podium

@case
def t_7_the_ring_fixtures_terraces_and_podium_are_declarations_of_the_contract():
    import test_rings
    import test_site_needs
    from ethoslm import placeplan
    from ethoslm.pipeline import stages_plan
    s = test_rings.three_ring_spec()
    site = test_rings._site()
    X, Z, S = site["origin"][0], site["origin"][1], 512
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        vol = test_site_needs._volume(X, Z, S, base=50, relief=40)
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        json.dump(json.loads(json.dumps(test_rings.THREE_RING)),
                  open(os.path.join(tmp, "place.json"), "w"))
        json.dump({**site, "mean_grid": [[70] * 12] * 12, "roughness_grid": [[3] * 12] * 12,
                   "stats": {"min": 50, "max": 90, "relief": 40}},
                  open(os.path.join(tmp, "site.json"), "w"))
        json.dump({"chosen": None, "terraform": None, "attempt": "as asked"},
                  open(os.path.join(tmp, "site_search.json"), "w"))
        rnd = pipeline.Round(name="contract_terraces", sentence="Build a ringed town.",
                             state_dir=tmp, voice="ochre_stone_green_tile",
                             flags={"dry_run": True})
        be = pipeline.OfflineBackend(rnd, dry_run=True)
        med = placeplan.site_median(rnd.volume(), site)
        pl = stages_plan.stage_plateau(rnd, be, {})
        assert pl.get("plateau", {}).get("ok"), pl
        # the podium: one declaration of designed ground at the level the arithmetic
        # chose, and the cut laid at the level it settled
        assert pl["contract"]["pieces"] == [{"label": pl["part"], "rect": pl["rect"],
                                             "level": med + 12, "what": "podium"}], pl["contract"]
        assert pl["plateau"]["y"] == med + 12
        assert pl["contract"]["seam_totals"]["face"] > 0
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(
            s, site, json.load(open(os.path.join(tmp, "plateau.json"))), decls,
            "ochre_stone_green_tile", vol=rnd.volume())
        assert not fails, fails
        json.dump(place, open(os.path.join(tmp, "plan.json"), "w"))
        got = stages_plan.stage_terraces(rnd, be, {})
        assert got.get("placed") and all(r["terrace"]["ok"] for r in got["rings"]), got
        # every ring's level is the contract's, and every piece the stage laid is a
        # piece the contract settled, at that level
        con = got["contract"]
        by_label = {p["label"]: p for p in con["pieces"]}
        assert [r["level"] for r in got["rings"]] == [med, med + 4, med + 8]
        laid = 0
        for r in got["rings"]:
            for q in r["terrace"]["piece_records"]:
                assert q["ok"] and q["y"] == r["level"]
                laid += 1
            mine = [p for p in con["pieces"] if p["label"].startswith(r["ring"] + "/")]
            assert mine and all(p["level"] == r["level"] and p["what"] == "terrace"
                                for p in mine), mine
        # ...the gates' approaches with them, at the levels they were read at
        for ap in got["approaches"]:
            for i, piece in enumerate(ap["pieces"]):
                p = by_label[f"{ap['gate']}_approach/{i}"]
                assert p["level"] == piece["level"] and p["rect"] == piece["rect"]
                assert p["what"] == "gate approach"
        # the step between two rings is a retaining face by the drop, derived and never
        # declared: TERRACE_STEP is 4 and a drop of 4 is a face
        assert placeplan.TERRACE_STEP == 4 and ground.seam_kind(placeplan.TERRACE_STEP) == "face"
        assert con["seam_totals"]["face"] > 0, con["seam_totals"]
        assert con["settled"].startswith("the rings before a block was laid")
        # A ring one level
        from ethoslm import observe
        v2 = rnd.volume()
        h, _wet = observe.ground_heights(v2)
        cx, cz = place["layout"]["centre"]
        r0 = place["layout"]["rings"][-1]
        a, b_ = r0["inner"] + 12, r0["outer"] - 4
        # a strip of the belt on a side with no gate on it, clear of the feathers
        side = -1 if place["layout"]["axis_side"] == "east" else 1
        x = cx + side * (a + b_) // 2
        sample = [int(h[x - v2.x0, cz + d - v2.z0]) for d in range(-a // 2, a // 2, 5)]
        assert all(v == r0["level"] for v in sample), (sample, r0["level"])
    return (f"the podium and {len(con['pieces'])} terrace pieces settled by the contract "
            f"and laid at their levels ({laid} bounded calls); rings at "
            f"{[r['level'] for r in got['rings']]}, seams {con['seam_totals']}; "
            f"{time.perf_counter() - t0:.0f}s")


# ------------------------------------------------------------------ 8. refusals

@case
def t_8_what_the_contract_refuses_site_refuses_the_same_way():
    X, Z, S = 1000, 2000, 64
    vol = _slope(X, Z, S)
    b = _builder(vol)
    bad = {"label": "skew", "kind": "edge", "path": [[X + 10, Z + 10], [X + 30, Z + 15]],
           "width": 1}
    c = ground.Contract()
    dec = b.declare(bad, mat=MAT, contract=c)
    assert not dec["ok"] and "neither along x" in dec["reason"] and len(c) == 0
    got = b.site(dict(bad), mat=MAT)
    assert got["ground"] == "unsited" and got["sited"]["reason"] == dec["reason"]
    assert not b._pending
    unknown = b.declare({"label": "blob", "kind": "cloud"}, mat=MAT, contract=c)
    assert not unknown["ok"] and "cloud" in unknown["reason"] and len(c) == 0
    return "a skew edge and an unknown kind are refused by the same words in both"


def main() -> int:
    bad = 0
    t0 = time.perf_counter()
    for fn in CASES:
        name = fn.__name__[2:].replace("_", " ")
        try:
            note = fn()
            print(f"ok   {name}: {note}")
        except Exception as e:                       # noqa: BLE001 -- a failing case
            import traceback
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} ground contract cases pass "
          f"({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
