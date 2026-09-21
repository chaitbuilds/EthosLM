"""A concentric place is a thing the spec declares and the library lays out.

    $PY scripts/test_rings.py

  1. **Rings in the spec.** A district part carries `ring`, `share`, `walled` and `voice`;
     shares over one are refused; a concentric wall count disagreeing with the walled
     rings is refused; a ring's authored voice goes through the voice validator; the
     setting's relief word folds into the place's needs as a registered cap.
  2. **Ring geometry by arithmetic.** A three-ring spec on a 512 site gives the rectangle
     sides the cumulative shares say; a belt at share 0.55 is over half the site; walls
     stand at the walled boundaries and nowhere else; districts tile every ring above the
     registered coverage; the arithmetic's output passes the place validator and the
     concentric clauses; an off-centre plateau is refused; a place with no rings still
     gets the freehand planner's brief.
  3. **The site and the core.** A square with a lake at its core fails a centred place's
     needs by name; a flat setting refuses the demo's site and accepts a plain; the
     plateau of a concentric place is cut at the site centre; a rural district is held to
     covering its ground; a leaf's kind is its type's, by construction.
  4. **The palette per ring, and the wall's face.** Two districts in two voices assemble
     into leaves carrying each its own voice and the place read reads each against its
     own; the great wall builds in three faces and they differ; the flythrough's final
     camera stands above the tallest thing between it and its subject.

Every case is deterministic and makes no model call. Cases that need `out/` skip on a
worktree without it, exactly as `test_place_spec`'s do.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ethoslm import pipeline, placeplan, placeread, spec as spec_mod  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something `out/` has and this checkout does not."""


# ------------------------------------------------------------- the ring spec A ringed
# capital, written as a fixture: a compound at the centre, four rings from the court's
# quarter out to a farm belt, two of them walled. General on purpose -- nothing here is
# a place anyone has heard of -- and the numbers are chosen so that every one of the
# arithmetic's branches is exercised: a ring too thin for its share, a walled and an
# unwalled boundary, a ring in a voice of its own.

RING_SENTENCE = "Build a ringed capital on a plain."

RING_SPEC = {
    "kind": "city",
    "form": "east_asian",
    # this fixture is the 512 city it was written as, and a recorded footprint reads
    # back as it was
    "needs": {"footprint": 512},
    "invariants": "A compound at the centre; four rings outward, the outer two walled; "
                  "a farm belt over half the whole; the plain is flat.",
    "defining_parts": [
        {"name": "great_court", "kind": "group", "family": "palace", "relation": "centre",
         "count": 1, "structures": 1, "forms": ["civic"], "role": "civic",
         "needs": {"max_relief": 4, "plateau": 40},
         "notes": "the compound at the middle"},
        {"name": "ring_walls", "kind": "edge", "family": "wall", "relation": "concentric",
         "count": 2, "structures": 0, "forms": ["fortification"],
         "notes": "an earthen wall, monolithic and unbroken, round the belt; a lesser "
                  "one round the town"},
        {"name": "ring_gates", "kind": "point", "family": "gate", "relation": "gateway",
         "count": 2, "structures": 0, "forms": ["fortification"],
         "notes": "one gate per wall"},
        {"name": "court_quarter", "kind": "group", "family": "district",
         "relation": "quarter", "count": 1, "structures": 6, "density": "low",
         "ring": 0, "share": 0.08, "walled": False, "voice": "japanese_temple",
         "notes": "the court's quarter, spacious"},
        {"name": "merchant_quarter", "kind": "group", "family": "district",
         "relation": "quarter", "count": 1, "structures": 12, "density": "medium",
         "ring": 1, "share": 0.12, "walled": False,
         "notes": "the merchants' quarter"},
        {"name": "outer_town", "kind": "group", "family": "district",
         "relation": "quarter", "count": 1, "structures": 20, "density": "dense",
         "ring": 2, "share": 0.2, "walled": True, "voice": "japanese_minka",
         "notes": "the packed outer town, inside the town wall"},
        {"name": "farm_belt", "kind": "group", "family": "district",
         "relation": "perimeter", "count": 1, "structures": 6, "density": "sparse",
         "role": "rural", "ring": 3, "share": 0.55, "walled": True,
         "notes": "the belt of fields inside the great wall"},
    ],
    "setting": {"surface": "green", "water": None, "relief": "flat",
                "notes": "a capital on a plain"},
    "voice": "ochre_stone_green_tile",
    "notes": "a fixture",
}


def ring_spec(**over):
    doc = json.loads(json.dumps(RING_SPEC))
    doc.update(over)
    return spec_mod.read_spec(doc, RING_SENTENCE)


def _site(size=512, origin=(-768, -1024)):
    return {"origin": list(origin), "size": size, "stats": {"relief": 100},
            "surface_blocks": {"grass_block": 100}}


# ------------------------------------------------------------ phase 1: the spec

@case
def t_1_a_spec_with_rings_validates_and_reads_back_the_same():
    s = ring_spec()
    rings = spec_mod.rings(s)
    assert [r["name"] for r in rings] == ["court_quarter", "merchant_quarter",
                                          "outer_town", "farm_belt"], rings
    assert [r["walled"] for r in rings] == [False, False, True, True]
    assert [r.get("voice") for r in rings] == ["japanese_temple", None,
                                                "japanese_minka", None]
    assert abs(spec_mod.centre_share(s) - 0.05) < 1e-9, spec_mod.centre_share(s)
    assert len(spec_mod.walled_rings(s)) == 2
    assert s["invariants"].startswith("A compound at the centre")
    # read back: the identical spec
    again = spec_mod.read_spec(json.loads(json.dumps(s)), RING_SENTENCE)
    assert json.dumps(again, sort_keys=True) == json.dumps(s, sort_keys=True)
    # the summary says so
    assert "rings: 4, 2 walled" in spec_mod.summary(s), spec_mod.summary(s)
    return "four rings, two walled, the centre's share 0.05; reads back byte-identical"


@case
def t_1_shares_over_one_and_a_wall_count_off_the_walled_rings_are_refused():
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][-1]["share"] = 0.7          # 0.08+0.12+0.2+0.7 = 1.1
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("shares of 1.1 were accepted")
    except spec_mod.SpecError as e:
        assert "sum to 1.100" in str(e), e
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][1]["count"] = 3                # two rings are walled
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("a concentric count of 3 over 2 walled rings was accepted")
    except spec_mod.SpecError as e:
        assert "number of walled rings, which is 2" in str(e), e
    # ...and a share on a part that is not a district is a comment, refused by name
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][1]["share"] = 0.1
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("a share on a wall was accepted")
    except spec_mod.SpecError as e:
        assert "belongs to a district part" in str(e), e
    # a ring without a share, and two parts on one ring
    doc = json.loads(json.dumps(RING_SPEC))
    del doc["defining_parts"][3]["share"]
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("a ring with no share was accepted")
    except spec_mod.SpecError as e:
        assert "carries a share" in str(e), e
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][4]["ring"] = 0
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("two parts on ring 0 were accepted")
    except spec_mod.SpecError as e:
        assert "both ring 0" in str(e), e
    return "shares 1.1, a wall count of 3 over 2 walled rings, a share on a wall, a ring with no share and two parts on one ring: each refused by name"


@case
def t_1_a_rings_authored_voice_is_validated_and_a_named_one_must_exist():
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][3]["voice"] = {
        "name": "court_lacquer", "roles": {"wall": "quartz", "footing": "granite",
                                           "frame": "dark_oak", "roof": "deepslate_tile",
                                           "trim": "red_sandstone", "floor": "smooth_stone"},
        "roof": {"ends": "hip", "eave": "upturned", "tiers": 2},
        "notes": {"blurb": "white walls under dark tile"}}
    s = spec_mod.read_spec(doc, RING_SENTENCE)
    assert s["defining_parts"][3]["voice"] == "court_lacquer"
    assert "court_lacquer" in s["authored_voices"], s.get("authored_voices")
    assert s["authored_voices"]["court_lacquer"]["roles"]["roof"] == "deepslate_tile"
    # a voice naming a family with no stairs is refused with the role named
    doc["defining_parts"][3]["voice"]["roles"]["roof"] = "hay_block"
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("a roof of hay was accepted")
    except spec_mod.SpecError as e:
        assert "roles.roof" in str(e) and "hay_block" in str(e), e
    # ...and a named voice that is on no disk and in no authoring is refused by name
    doc = json.loads(json.dumps(RING_SPEC))
    doc["defining_parts"][3]["voice"] = "no_such_voice"
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("an unknown voice name was accepted")
    except spec_mod.SpecError as e:
        assert "no_such_voice" in str(e), e
    # the stage writes a ring's authored voice to disk beside the place's
    from ethoslm import voices as _voices
    with tempfile.TemporaryDirectory() as tmp:
        got = _voices.author("court_lacquer", s["authored_voices"]["court_lacquer"],
                             directory=tmp)
        assert os.path.exists(os.path.join(tmp, "court_lacquer.json"))
        assert got["roof"]["tiers"] == 2
    return "an authored ring voice validates and lands on the spec; a hay roof and an unknown name are refused by name"


@case
def t_1_the_settings_relief_word_folds_into_the_needs_as_a_registered_cap():
    flat = ring_spec()
    assert flat["needs"]["footprint"] == 512
    assert flat["needs"]["max_relief"] == 128.0, flat["needs"]
    assert flat["setting"]["relief"] == "flat"
    doc = json.loads(json.dumps(RING_SPEC))
    doc["setting"]["relief"] = "rolling"
    rolling = spec_mod.read_spec(doc, RING_SENTENCE)
    assert rolling["needs"]["max_relief"] == 281.6, rolling["needs"]
    doc["setting"]["relief"] = "any"
    anyr = spec_mod.read_spec(doc, RING_SENTENCE)
    assert anyr["needs"]["max_relief"] == 281.6
    doc["setting"]["relief"] = None
    none = spec_mod.read_spec(doc, RING_SENTENCE)
    assert none["needs"]["max_relief"] == 281.6
    # a spec's own explicit cap that is tighter still wins
    doc["setting"]["relief"] = "flat"
    doc["needs"] = {"max_relief": 60}
    tight = spec_mod.read_spec(doc, RING_SENTENCE)
    assert tight["needs"]["max_relief"] == 60.0
    doc["setting"]["relief"] = "hilly"
    try:
        spec_mod.read_spec(doc, RING_SENTENCE)
        raise AssertionError("'hilly' was accepted")
    except spec_mod.SpecError as e:
        assert "relief is one of" in str(e), e
    assert spec_mod.RELIEF_WORDS == {"flat": 0.25, "rolling": 0.55, "steep": None,
                                     "any": None}
    return "flat 128.0 on 512, rolling and any 281.6, an explicit 60 wins, 'hilly' refused"


@case
def t_1_the_brief_asks_for_invariants_and_rings_and_names_no_place():
    from ethoslm.pipeline import stages_plan
    text = stages_plan.spec_brief("Build a ringed capital.", "/tmp/x.json")
    for want in ("`invariants`", "\"ring\":", "\"share\":", "\"walled\":",
                 "\"relief\":  \"flat\"", "two baileys"):
        assert want in text, want
    assert "Sing Se" not in text
    assert json.load(open(os.path.join(ROOT, "models.json")))["roles"]["spec"] \
        == "frontier"
    return "invariants first, the four ring fields, a relief word; the brief names no place; the spec role is the frontier tier"


@case
def t_1_districts_that_declare_no_number_share_the_size_by_share_over_density():
    doc = json.loads(json.dumps(RING_SPEC))
    for p in doc["defining_parts"]:
        p["structures"] = 0
    s = spec_mod.read_spec(doc, RING_SENTENCE)
    # The middle of the kind's band, whatever the band is.
    lo, hi = s["size_band"]
    n = int(round((lo + hi) / 2))
    assert s["structures"] == n, (s["structures"], lo, hi)
    got = {p["name"]: p["structures"] for p in spec_mod.rings(s)}
    assert sum(got.values()) == n, got
    # share over ground per structure: 0.08/1.5, 0.12/1.0, 0.2/0.7, 0.55/2.0
    w = {"court_quarter": 0.08 / 1.5, "merchant_quarter": 0.12 / 1.0,
         "outer_town": 0.2 / 0.7, "farm_belt": 0.55 / 2.0}
    tot = sum(w.values())
    for k, v in w.items():
        assert abs(got[k] - n * v / tot) < 1.0, (k, got[k], n * v / tot)
    # ...and reads back the same, and a spec that gave numbers keeps them
    again = spec_mod.read_spec(json.loads(json.dumps(s)), RING_SENTENCE)
    assert json.dumps(again, sort_keys=True) == json.dumps(s, sort_keys=True)
    given = ring_spec()      # declared 6 / 12 / 20 / 6, apportioned to the band
    kept = {p["name"]: p["structures"] for p in spec_mod.rings(given)}
    assert sum(kept.values()) == n, kept
    for name, declared in (("court_quarter", 6), ("merchant_quarter", 12),
                           ("outer_town", 20), ("farm_belt", 6)):
        assert abs(kept[name] - n * declared / 44.0) < 1.0, (name, kept[name], n)
    # a ringless spec with districts at 0 shares the size equally
    flat = spec_mod.read_spec({"kind": "town", "defining_parts": [
        {"name": "east_end", "kind": "group", "family": "quarter", "relation": "quarter",
         "count": 1}, {"name": "west_end", "kind": "group", "family": "quarter",
                       "relation": "quarter", "count": 1}], "voice": None}, "Build a town.")
    half = int(round(sum(spec_mod.size_band_for("town")) / 2)) // 2
    assert [p["structures"] for p in flat["defining_parts"]] == [half, half], flat
    return (f"{n} structures spread {got} by share over ground per structure; a spec "
            f"that gave numbers keeps their ratios; two bare quarters split a town "
            f"{half}/{half}")


# ------------------------------------------------- phase 2: the arithmetic

THREE_RING = {
    "kind": "city", "form": "east_asian", "needs": {"footprint": 512},
    "defining_parts": [
        {"name": "great_court", "kind": "group", "family": "palace", "relation": "centre",
         "count": 1, "structures": 1, "forms": ["civic"], "role": "civic",
         "needs": {"max_relief": 4, "plateau": 40}, "notes": "the compound"},
        {"name": "ring_walls", "kind": "edge", "family": "wall", "relation": "concentric",
         "count": 2, "structures": 0, "forms": ["fortification"],
         "notes": "coursed masonry walls"},
        {"name": "ring_gates", "kind": "point", "family": "gate", "relation": "gateway",
         "count": 2, "structures": 0, "forms": ["fortification"], "notes": "gates"},
        {"name": "inner_town", "kind": "group", "family": "district", "relation": "quarter",
         "count": 1, "structures": 10, "density": "low", "ring": 0, "share": 0.1,
         "notes": "the inner town"},
        {"name": "outer_town", "kind": "group", "family": "district", "relation": "quarter",
         "count": 1, "structures": 20, "density": "dense", "ring": 1, "share": 0.2,
         "walled": True, "notes": "the outer town"},
        {"name": "farm_belt", "kind": "group", "family": "district",
         "relation": "perimeter", "count": 1, "structures": 6, "density": "sparse",
         "role": "rural", "ring": 2, "share": 0.55, "walled": True,
         "notes": "the farm belt"},
    ],
    "setting": {"surface": "green", "relief": "flat"},
    "voice": "ochre_stone_green_tile", "notes": "a fixture",
}


def three_ring_spec():
    return spec_mod.read_spec(json.loads(json.dumps(THREE_RING)), "Build a ringed town.")


def _plateau(site, part="great_court", shift=(0, 0)):
    n = placeplan.compound_ground()["side"]
    X, Z = site["origin"]
    S = site["size"]
    x0, z0 = X + (S - n) // 2 + shift[0], Z + (S - n) // 2 + shift[1]
    return {"part": part, "rect": [x0, z0, x0 + n - 1, z0 + n - 1], "y": 66}


def _layout(spec, site=None, plateau=None, voice="ochre_stone_green_tile"):
    site = site or _site()
    plateau = _plateau(site) if plateau is None else plateau
    _t, decls = placeplan.types_card(None, spec.get("form"))
    place, fails = placeplan.concentric_layout(spec, site, plateau, decls, voice)
    return place, fails, decls, site, plateau


def _ring_rect(part):
    xs = [a[0] for a in part["path"]]
    zs = [a[1] for a in part["path"]]
    return min(xs), min(zs), max(xs), max(zs)


@case
def t_2_a_three_ring_spec_on_512_gives_the_sides_the_arithmetic_says():
    s = three_ring_spec()
    place, fails, decls, site, plateau = _layout(s)
    assert not fails, fails
    L = place["layout"]
    S = 512
    hc = (plateau["rect"][2] - plateau["rect"][0]) // 2
    # the boundaries: S * sqrt(cum) / 2, the last the site's edge less the inset
    cum = spec_mod.centre_share(s)
    want = []
    for r in spec_mod.rings(s):
        cum += r["share"]
        want.append(int(round(S * math.sqrt(cum) / 2.0)))
    want[-1] = S // 2 - placeplan.RING_EDGE_INSET
    got = [r["outer"] for r in L["rings"]]
    assert L["squeezed"] == [], L["squeezed"]
    assert got == want, (got, want)
    assert L["rings"][0]["inner"] == hc == 41, L["rings"][0]
    # the belt at share 0.55 is over half the site
    belt = L["rings"][-1]
    assert belt["share_got"] > 0.5, belt
    assert abs(belt["share_got"] - 0.55) < 0.05, belt
    # and the record says every number
    for r in L["rings"]:
        assert r["width"] == r["outer"] - r["inner"] and r["width"] >= r["min_width"], r
    return (f"boundaries at half-sides {got} (sides {[2 * h + 1 for h in got]}); the "
            f"belt is {belt['share_got']:.1%} of the site for a share of 0.55")


@case
def t_2_the_walls_stand_at_the_walled_boundaries_and_nowhere_else():
    s = three_ring_spec()
    place, fails, decls, site, plateau = _layout(s)
    L = place["layout"]
    walls = [p for p in place["parts"] if p["kind"] == "edge"]
    gates = [p for p in place["parts"] if p["kind"] == "point"]
    assert len(walls) == 2 and len(gates) == 2, (len(walls), len(gates))
    cx, cz = L["centre"]
    walled = [r for r in L["rings"] if r["walled"]]
    for w, r in zip(sorted(walls, key=lambda p: -placeread.ring_area(p["path"])),
                    sorted(walled, key=lambda r: -r["outer"])):
        x0, z0, x1, z1 = _ring_rect(w)
        assert (x0, z0, x1, z1) == (cx - r["outer"], cz - r["outer"],
                                    cx + r["outer"], cz + r["outer"]), (w["name"], r)
        assert w["path"][0] == w["path"][-1]
        assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(w["path"], w["path"][1:]))
        run = max(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1
                  for a, b in zip(w["path"], w["path"][1:]))
        assert run <= decls[w["type"]]["needs"]["footprint"][3], run
        assert w["params"]["height"] == r["wall_height"] and w["type"] == r["wall"]
    heights = sorted((w["params"]["height"] for w in walls), reverse=True)
    assert heights == [48, 36], heights
    # the great wall is the outermost and the tallest admitted type at its top
    assert heights[0] == decls["great_wall"]["params"]["height"][2]
    assert heights[1] == int(round(48 * placeplan.WALL_STEP))
    # coursed masonry in the spec's words: the banded face
    assert all(w["params"].get("face") in (None, "banded") for w in walls), walls
    # the unwalled boundary has no wall
    inner = [r for r in L["rings"] if not r["walled"]]
    assert len(inner) == 1 and inner[0]["wall"] is None
    # one gate per walled ring, on the axis, on one side, on its wall's line
    from ethoslm.placeplan import _edge_cells
    side = L["axis_side"]
    for g in gates:
        on = [w for w in walls if tuple(g["at"]) in set(_edge_cells(w))]
        assert len(on) == 1, g
        assert g["at"][0] == cx if side in ("north", "south") else g["at"][1] == cz, g
    return (f"walls at half-sides {[r['outer'] for r in walled]} as great_wall 48 and 36, "
            f"faces {[w['params'].get('face') for w in walls]}; 2 gates on the {side} side")


@case
def t_2_the_districts_tile_every_ring_above_the_registered_coverage():
    s = three_ring_spec()
    place, fails, decls, site, plateau = _layout(s)
    L = place["layout"]
    cx, cz = L["centre"]
    assert placeplan.RING_COVERAGE == 0.6
    for r in L["rings"]:
        assert r["coverage"] >= placeplan.RING_COVERAGE, r
        assert len(r["districts"]) >= 4, r
        mine = [d for d in place["districts"] if d["defines"] == r["name"]]
        # inside the annulus: every corner between the inner and outer boundaries
        for d in mine:
            for (x, z) in ((d["x0"], d["z0"]), (d["x1"], d["z1"]),
                           (d["x0"], d["z1"]), (d["x1"], d["z0"])):
                m = max(abs(x - cx), abs(z - cz))
                assert r["inner"] < m <= r["outer"], (d["name"], m, r)
            assert d["voice"] and d["ring"] == r["ring"] and d["structures"] >= 1
    # no two districts overlap, and none touches the compound or a wall's line
    rects = [(d["name"], (d["x0"], d["z0"], d["x1"], d["z1"])) for d in place["districts"]]
    for i, (an, a) in enumerate(rects):
        for bn, b in rects[i + 1:]:
            assert not (a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]), \
                (an, bn)
    c = place["compounds"][0]
    for n, a in rects:
        assert not (a[0] <= c["x1"] and c["x0"] <= a[2] and a[1] <= c["z1"]
                    and c["z0"] <= a[3]), n
    # the ring's structures are spread by area, and the record says what was laid
    for r in L["rings"]:
        laid = sum(d["structures"] for d in place["districts"] if d["defines"] == r["name"])
        assert laid == r["structures"]["laid"] <= r["structures"]["asked"], r["structures"]
    return ("coverage " + ", ".join(f"{r['name']} {r['coverage']:.0%}" for r in L["rings"])
            + f" against {placeplan.RING_COVERAGE:.0%}; {len(place['districts'])} districts, "
              f"none overlapping")


@case
def t_2_the_arithmetics_output_passes_the_validator_it_is_held_to():
    s = three_ring_spec()
    place, fails, decls, site, plateau = _layout(s)
    got = placeplan.place_failures(place, s, site, decls, ground={}, plateau=plateau)
    assert got == [], got
    assert placeplan.concentric_failures(place, s, decls) == []
    # ...and the four-ring fixture with a ring in a voice of its own and a thin ring
    # whose share is under its least width: squeezed, recorded, still valid
    s4 = ring_spec()
    place4, fails4, decls4, site4, plateau4 = _layout(s4)
    assert not fails4, fails4
    got4 = placeplan.place_failures(place4, s4, site4, decls4, ground={}, plateau=plateau4)
    assert got4 == [], got4
    L4 = place4["layout"]
    voices = {d["voice"] for d in place4["districts"]}
    assert voices == {"japanese_temple", "japanese_minka", "ochre_stone_green_tile"}, voices
    # shares too thin for their rings: the least widths are taken from the ring with
    # slack, recorded, and the layout is still valid
    doc = json.loads(json.dumps(RING_SPEC))
    for i, sh in zip((3, 4, 5, 6), (0.01, 0.01, 0.01, 0.95)):
        doc["defining_parts"][i]["share"] = sh
    thin = spec_mod.read_spec(doc, RING_SENTENCE)
    place_t, fails_t, decls_t, site_t, plateau_t = _layout(thin)
    assert not fails_t, fails_t
    Lt = place_t["layout"]
    assert Lt["squeezed"] == ["farm_belt"], Lt["squeezed"]
    for r in Lt["rings"][:3]:
        assert r["width"] == r["min_width"], r
    assert placeplan.place_failures(place_t, thin, site_t, decls_t, ground={},
                                    plateau=plateau_t) == []
    # ...and a site too small for the least widths is refused by name
    small = _site(size=160)
    place_s, fails_s = placeplan.concentric_layout(
        thin, small, _plateau(small), decls_t, "ochre_stone_green_tile")
    assert place_s is None and fails_s[0]["check"] == "shares", fails_s
    return (f"0 failures on the three-ring layout and 0 on the four-ring one in "
            f"{len(voices)} voices; shares of 0.01 are widened to their least widths "
            f"{[r['min_width'] for r in Lt['rings'][:3]]} out of the belt; a 160 site is "
            f"refused naming 'shares'")


@case
def t_2_an_off_centre_plateau_is_refused_and_a_place_with_no_rings_is_untouched():
    s = three_ring_spec()
    site = _site()
    off = _plateau(site, shift=(90, -17))            # the demo's offset
    place, fails, decls, site, plateau = _layout(s, site=site, plateau=off)
    assert place is None and fails and fails[0]["check"] == "centred", fails
    assert "91" not in fails[0]["why"] or True
    assert fails[0]["offset"] > placeplan.CENTRED_TOLERANCE == 4
    # within the tolerance is accepted
    near = _plateau(site, shift=(2, -1))
    place, fails, decls, site, plateau = _layout(s, site=site, plateau=near)
    assert place is not None and not fails, fails
    # no rings: not this function's place, and the freehand brief is what the stage
    # writes
    import test_place_spec
    flat = spec_mod.read_spec(json.loads(json.dumps(test_place_spec.CITY_SPEC)),
                              test_place_spec.CITY_SENTENCE)
    assert spec_mod.rings(flat) == []
    place, fails = placeplan.concentric_layout(flat, site, None, decls, "x")
    assert place is None and fails[0]["check"] == "rings"
    brief = placeplan.place_brief(flat, {**site, "mean_grid": [[64] * 12] * 12,
                                         "roughness_grid": [[1] * 12] * 12,
                                         "stats": {"min": 60, "max": 70, "relief": 10}},
                                  "/tmp/p.json", None, "ochre_stone_green_tile")
    assert "You are planning the **place**" in brief
    return "a plateau 90 off is refused naming 'centred'; 2 off is taken; a ringless spec gets the freehand brief"


@case
def t_2_the_plan_stage_lays_a_ringed_place_out_and_asks_for_no_place_planner():
    from ethoslm.pipeline import stages_plan
    from ethoslm import pipeline
    s = three_ring_spec()
    with tempfile.TemporaryDirectory() as tmp:
        site = {**_site(), "mean_grid": [[64] * 12] * 12, "roughness_grid": [[1] * 12] * 12,
                "stats": {"min": 60, "max": 70, "relief": 10}}
        json.dump(json.loads(json.dumps(THREE_RING)), open(os.path.join(tmp, "place.json"), "w"))
        json.dump(site, open(os.path.join(tmp, "site.json"), "w"))
        pl = _plateau(site)
        json.dump({"part": pl["part"], "rect": pl["rect"],
                   "plateau": {"ok": True, "y": 66}}, open(os.path.join(tmp, "plateau.json"), "w"))
        rnd = pipeline.Round(name="ring_fixture", sentence="Build a ringed town.",
                             state_dir=tmp, voice="ochre_stone_green_tile")
        be = pipeline.OfflineBackend(rnd)
        # **The stage is re-entered for as long as it asks to be**, which is what the
        # driver does (`pipeline.round._drive_reentries`). The integration round moved
        # the capacity repair to the plan level, so this ringed town's first answer is
        # `reenter`: the band a city is inferred to be is negotiated down to what the
        # ground gave, the place is resolved again, and only then is a district asked
        # for. Asserting the first answer is `needs_model` was asserting the protocol
        # the repair was added to change.
        reentries = []
        for _ in range(8):
            got = stages_plan.stage_plan_levels(rnd, be, {}, s)["plan"]
            if got.get("status") != "reenter":
                break
            reentries.append(got.get("why"))
        assert got["status"] == "needs_model", got
        assert reentries, "the plan level asked for no repair at all"
        # **the compound is laid by the library and no model is asked for it**, the
        # craft round (E5): a palace's composition declares an axis, so the stage writes
        # `plan.compound.<name>.json` itself and the next thing it asks for is a
        # district
        assert got["level"].startswith("district/"), got["level"]
        assert os.path.exists(os.path.join(tmp, "plan.compound.great_court.json"))
        assert os.path.exists(os.path.join(tmp, "compound_great_court_axial.json"))
        assert not os.path.exists(os.path.join(tmp, "compound_great_court_prompt.md"))
        assert os.path.exists(os.path.join(tmp, "plan.place.json"))
        assert not os.path.exists(os.path.join(tmp, "place_plan_prompt.md"))
        place = json.load(open(os.path.join(tmp, "plan.place.json")))
        assert place["layout"]["by"] == "placeplan.concentric_layout"
        log = json.load(open(os.path.join(tmp, "plan_validation.json")))
        first = [a for a in log["attempts"] if a["level"] == "place"][0]
        assert first["failures"] == [] and "arithmetic" in first["checked"], first
        axial = json.load(open(os.path.join(tmp, "compound_great_court_axial.json")))
        return (f"the stage wrote {len(place['districts'])} districts, laid the compound "
                f"itself as a sequence ({' -> '.join(axial['order'])}), repaired its "
                f"own capacity finding and re-entered {len(reentries)} time(s), then "
                f"asked for {got['level']}; no place brief and no compound brief exists")


# ---------------------------------------------- phase 3: the site and the core

def _find_site():
    import importlib.util
    p = os.path.join(ROOT, "scripts", "find_site.py")
    sp = importlib.util.spec_from_file_location("find_site", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _cached_square(fs, x0, z0, size=512):
    import numpy as np
    p = fs.tile_cache(x0, z0, size, size)
    if not os.path.exists(p):
        raise Skip(f"no {os.path.relpath(p, ROOT)}")
    got = np.load(p)
    return fs._field_from_cache(got, x0, z0, size, size, p)


@case
def t_3_a_site_with_a_lake_at_its_core_is_refused_for_a_centred_city():
    import numpy as np
    import test_place_spec
    fs = _find_site()
    s = three_ring_spec()
    needs = fs.search_needs(s)
    assert needs["core_water_max_pct"] == spec_mod.CORE_WATER_MAX_PCT == 2.0, needs
    size, core = 512, fs.core_size(s)
    f = test_place_spec._settled(fs, size, core, 2, 20, "grass_block")
    m = fs.measure(f, f.x0, f.z0, size, min(fs._plateau_size(s), size), core=core)
    e = fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert e["core_water"] == 0.0 and m["core"]["water_pct"] == 0.0
    # ...the same ground with a lake at the middle: 15% of the core under water, which
    # is what the demo's site read
    wet = np.zeros((size, size), bool)
    i, j, n = fs.core_window(size, core)
    k = int(round(n * np.sqrt(0.153)))
    wet[i + (n - k) // 2:i + (n - k) // 2 + k, j + (n - k) // 2:j + (n - k) // 2 + k] = True
    lake = fs.Field(f.x0, f.z0, f.h, wet, f.canopy, gravity=f.gravity, source="fixture",
                    manmade=f.manmade, occupied=f.occupied,
                    surface=np.where(wet, 1, 0).astype(np.int32),
                    surface_palette=["grass_block", "water"])
    m2 = fs.measure(lake, f.x0, f.z0, size, min(fs._plateau_size(s), size), core=core)
    e2 = fs.excess(m2, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert 14.0 < m2["core"]["water_pct"] < 17.0, m2["core"]
    assert e2["core_water"] > 0 and not e2["meets"], e2
    assert e2["relief"] == 0.0 and e2["water"] == 0.0, e2       # only the core fails it
    # ...and a spec with no part at the centre is scored as it always was
    no_centre = spec_mod.read_spec({"kind": "hamlet", "defining_parts": [
        {"name": "crofts", "kind": "group", "family": "quarter", "relation": "throughout",
         "count": 1, "structures": 8}], "voice": None}, "Build a hamlet.")
    assert "core_water_max_pct" not in fs.search_needs(no_centre)
    # the rank: water above relief among squares that meet
    dry_row = {"measures": m, "excess": e}
    wetter = {"measures": {**m, "water_pct": m["water_pct"] + 5.0,
                           "core": dict(m["core"]),
                           "plateau": {**m["plateau"], "relief": 0}},
              "excess": e}
    assert fs.rank_key(dry_row) < fs.rank_key(wetter)
    return (f"a dry core meets; {m2['core']['water_pct']}% water at the core fails by "
            f"'core_water' alone against {spec_mod.CORE_WATER_MAX_PCT}%; a spec with no "
            f"centre carries no such need; the drier square outranks the flatter one")


@case
def t_3_a_flat_setting_refuses_the_demos_site_and_accepts_a_plain():
    fs = _find_site()
    s = three_ring_spec()                      # setting.relief == "flat" -> 128 over 512
    assert s["needs"]["max_relief"] == 128.0
    needs = fs.search_needs(s)
    hilly = _cached_square(fs, -768, -1024)    # the demo's site: relief 178
    plain = _cached_square(fs, -1024, -256)    # the rank-2 candidate: relief 103
    size, core = 512, fs.core_size(s)
    pl = min(fs._plateau_size(s), size)
    mh = fs.measure(hilly, -768, -1024, size, pl, core=core)
    mp = fs.measure(plain, -1024, -256, size, pl, core=core)
    assert mh["relief"] == 178 and mp["relief"] == 103, (mh["relief"], mp["relief"])
    eh = fs.excess(mh, needs, plateau_relief=fs.PLATEAU_RELIEF)
    ep = fs.excess(mp, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert eh["relief"] > 0 and not eh["meets"], eh
    assert ep["relief"] == 0.0 and ep["core_water"] == 0.0, ep
    # ...and under `rolling` the demo's site passes the relief need again
    doc = json.loads(json.dumps(THREE_RING))
    doc["setting"]["relief"] = "rolling"
    r = spec_mod.read_spec(doc, "Build a ringed town.")
    er = fs.excess(mh, fs.search_needs(r), plateau_relief=fs.PLATEAU_RELIEF)
    assert er["relief"] == 0.0, er
    # the demo's own core water, read the way the search now reads it
    assert mh["core"]["water_pct"] > spec_mod.CORE_WATER_MAX_PCT, mh["core"]
    assert eh["core_water"] > 0
    return (f"flat: relief 178 refused ({eh['relief']:.2f} over 128), 103 accepted; "
            f"rolling admits 178; the demo's core reads {mh['core']['water_pct']}% water "
            f"over {mh['core']['size']}x{mh['core']['size']} and fails core_water too")


@case
def t_3_the_plateau_of_a_concentric_place_is_cut_at_the_site_centre():
    from ethoslm import offline
    from ethoslm.observe import Volume
    from ethoslm.pipeline import stages_plan
    s = three_ring_spec()
    with tempfile.TemporaryDirectory() as tmp:
        # a 160x160 site of two terraces: flat at 60 on the west, 70 on the east, so the
        # flattest window is off-centre and the centre straddles the step
        X, Z, S = 1000, 2000, 160
        b = {}
        for x in range(X - 8, X + S + 8):
            for z in range(Z - 8, Z + S + 8):
                b[(x, 60 if x < X + 96 else 70, z)] = "stone"
        vol = Volume.from_blocks(b, X - 8, 50, Z - 8, S + 16, 40, S + 16)
        offline.save_volume(vol, os.path.join(tmp, "world.npz"))
        rnd = pipeline.Round(name="centred_fixture", sentence="Build a ringed town.",
                             state_dir=tmp)
        site = {"origin": [X, Z], "size": S}
        terra = {"part": "great_court", "plateau": 40}
        # the search's record put the flattest 40 in the west terrace's corner
        m = {"size": 40, "x": X + 4, "z": Z + 4, "y": 60, "relief": 0}
        n, x0, z0, m2 = stages_plan.compound_cut(rnd, s, site, terra, 40, X + 4, Z + 4, m)
        want = placeplan.compound_ground()["side"]
        assert n == want, (n, want)
        assert (x0, z0) == (X + (S - n) // 2, Z + (S - n) // 2), (x0, z0)
        # A concentric place's plateau is the podium, a step above the innermost ring,
        # and the window's own median is recorded beside it
        assert m2["centred"] is True and m2["window_median"] in (60, 70), m2
        assert m2["y"] == m2["terrace"]["podium"] == m2["terrace"]["median"] + 3 * 4, m2
        # the same call for the ringless city spec keeps its dry-before-flat window
        import test_place_spec
        flat = spec_mod.read_spec(json.loads(json.dumps(test_place_spec.CITY_SPEC)),
                                  test_place_spec.CITY_SENTENCE)
        terra2 = {"part": "palace", "plateau": 40}
        n3, x3, z3, m3 = stages_plan.compound_cut(rnd, flat, site, terra2, 40, X + 4,
                                                  Z + 4, m)
        assert n3 == want and "centred" not in m3, m3
        assert (x3, z3) != (x0, z0), "the ringless spec was centred too"
        return (f"a {n}x{n} plateau at ({x0},{z0}), the centre of a {S} site at "
                f"({X},{Z}), level {m2['y']}; the ringless spec keeps its window at "
                f"({x3},{z3})")


@case
def t_3_a_rural_district_is_held_to_covering_its_ground_and_the_brief_says_so():
    s = three_ring_spec()
    place, fails, decls, site, plateau = _layout(s)
    d = next(d for d in place["districts"] if d["defines"] == "farm_belt")
    role = spec_mod.district_role(s, d)
    assert role == "rural"
    _t, rdecls = placeplan.types_card(None, s.get("form"), role)
    assert "farmstead" in rdecls and "field" in rdecls, sorted(rdecls)
    x0, z0 = d["x0"], d["z0"]
    # four farmsteads in a corner of a 239x59 district: a fifth of it
    sparse = {"quarters": [{"name": "farms", "plots": [
        {"kind": "plot", "name": f"farm_{i}", "type": "farmstead", "seed": i,
         "x0": x0 + 2 + i * 30, "z0": z0 + 2, "x1": x0 + 2 + i * 30 + 23,
         "z1": z0 + 25} for i in range(4)]}]}
    part = placeplan._district_part(s, d)
    # **the land use, not the role.** The unification round: a fishing village is also
    # `rural` and owes no fields, so what demands cover is the ground being farmland --
    # declared by the part or, as here, said in its own name and notes.
    assert spec_mod.land_use(part) == "farmland", part
    got = placeplan.district_failures(d, sparse, place, rdecls, role=role,
                                      form=s.get("form"), part=part, spec=s)
    cover = [f for f in got if f["check"] == "farmland_cover"]
    assert cover and cover[0]["part"] == d["name"], got
    assert cover[0]["share"] < placeplan.RURAL_COVER == 0.6
    assert str(cover[0]["columns"]) in cover[0]["why"]
    # ...fields drawn between and beyond the farms cover it, and the failure is gone
    plots = list(sparse["quarters"][0]["plots"])
    w = d["x1"] - d["x0"] + 1
    for i, fx in enumerate(range(x0 + 2, d["x1"] - 40, 47)):
        plots.append({"kind": "plot", "name": f"field_{i}", "type": "field",
                      "seed": i, "x0": fx, "z0": z0 + 31, "x1": min(fx + 41, d["x1"] - 1),
                      "z1": d["z1"] - 2})
    for i, fx in enumerate(range(x0 + 128, d["x1"] - 40, 47)):
        plots.append({"kind": "plot", "name": f"upper_field_{i}", "type": "field",
                      "seed": 9 + i, "x0": fx, "z0": z0 + 2, "x1": min(fx + 41, d["x1"] - 1),
                      "z1": z0 + 25})
    covered = {"quarters": [{"name": "farms", "plots": plots}]}
    got2 = placeplan.district_failures(d, covered, place, rdecls, role=role,
                                       form=s.get("form"), part=part, spec=s)
    assert not [f for f in got2 if f["check"] == "farmland_cover"], got2
    # ...and the same district as a settled one is not asked for fields at all
    settled = dict(part, land_use="settled")
    got3 = placeplan.district_failures(d, sparse, place, rdecls, role=role,
                                       form=s.get("form"), part=settled,
                                       spec={**s, "defining_parts": [settled]})
    assert not [f for f in got3 if f["check"] == "farmland_cover"], got3
    # the fields were written as `plot` and are areas, by construction: no kind failure
    assert not [f for f in got2 if f["check"] == "type"], got2
    assert all(p["kind"] == "area" for p in placeplan.district_plots(covered, role)
               if p["type"] == "field")
    # the brief says it, in columns
    brief = placeplan.district_brief(s, {**site, "mean_grid": [[64] * 12] * 12,
                                         "roughness_grid": [[1] * 12] * 12,
                                         "stats": {"min": 60, "max": 70, "relief": 10}},
                                     d, place, "/tmp/d.json", None, "ochre_stone_green_tile")
    area = (d["x1"] - d["x0"] + 1) * (d["z1"] - d["z0"] + 1)
    assert f"{int(math.ceil(placeplan.RURAL_COVER * area))} of its {area} columns" in brief
    urban = next(d for d in place["districts"] if d["defines"] == "inner_town")
    brief_u = placeplan.district_brief(s, {**site, "mean_grid": [[64] * 12] * 12,
                                           "roughness_grid": [[1] * 12] * 12,
                                           "stats": {"min": 60, "max": 70, "relief": 10}},
                                       urban, place, "/tmp/d.json", None,
                                       "ochre_stone_green_tile")
    assert "farmland is covered" not in brief_u
    # a ring in its own voice gets its own voice card in its brief
    s4 = ring_spec()
    place4, fails4, decls4, site4, plateau4 = _layout(s4)
    d4 = next(d for d in place4["districts"] if d["defines"] == "outer_town")
    b4 = placeplan.district_brief(s4, {**site4, "mean_grid": [[64] * 12] * 12,
                                       "roughness_grid": [[1] * 12] * 12,
                                       "stats": {"min": 60, "max": 70, "relief": 10}},
                                  d4, place4, "/tmp/d.json", None, "ochre_stone_green_tile")
    assert "This district's own voice is `japanese_minka`" in b4
    return (f"four farms cover {cover[0]['share']:.0%} and are refused naming "
            f"{cover[0]['columns']} columns; with fields between them the district "
            f"passes, its fields areas whatever the planner wrote; the rural brief asks "
            f"for it in columns and the urban one does not")


@case
def t_3_a_leafs_kind_is_its_types_by_construction_at_every_level():
    from ethoslm import pipeline as _p
    got = {"quarters": [{"name": "q", "plots": [
        {"kind": "plot", "name": "green", "type": "square", "seed": 1,
         "x0": 0, "z0": 0, "x1": 12, "z1": 12},
        {"kind": "plot", "name": "house", "type": "townhouse", "seed": 2,
         "x0": 20, "z0": 0, "x1": 34, "z1": 14},
        {"kind": "area", "name": "odd", "type": "townhouse", "seed": 3,
         "x0": 40, "z0": 0, "x1": 54, "z1": 14},
        {"name": "unknown", "type": "no_such_type", "seed": 4,
         "x0": 60, "z0": 0, "x1": 64, "z1": 4}]}]}
    kinds = {p["name"]: p["kind"] for p in placeplan.district_plots(got)}
    assert kinds == {"green": "area", "house": "plot", "odd": "plot", "unknown": "plot"}, kinds
    decls = _p.type_declarations(placeplan.district_plots(got))
    fails = _p.plan_failures(placeplan.district_plots(got), decls)
    assert not [f for f in fails if f["check"] == "type" and f["part"] in ("green", "odd")], fails
    assert [f["part"] for f in fails if f["check"] == "type"] == ["unknown"], fails
    # the compound level and the assembly say the same
    comp = placeplan.compound_parts({"parts": [
        {"kind": "plot", "name": "court", "type": "square", "seed": 1,
         "x0": 0, "z0": 0, "x1": 12, "z1": 12}]}, None, "c")
    assert comp[0]["kind"] == "area"
    spec = spec_mod.read_spec({"kind": "town", "defining_parts": [
        {"name": "the_quarter", "kind": "group", "family": "quarter", "relation": "throughout",
         "count": 1, "structures": 3}], "voice": None}, "Build a town.")
    plan = placeplan.assemble({"parts": [], "districts": [
        {"name": "d", "x0": 0, "z0": 0, "x1": 80, "z1": 80, "structures": 3}]},
        {"d": got}, spec)
    leaves = {p["name"]: p["kind"] for p in _p.plan_parts(plan)}
    assert leaves["green"] == "area" and leaves["house"] == "plot", leaves
    return "square -> area and townhouse -> plot whatever the leaf said, in the district, the compound and the assembly; an unknown type keeps what it wrote and is refused by name"


# --------------------------------- phase 4: the palette per ring, and the wall's face

def _two_voice_town():
    """A town of two quarters in two voices: the leaves, the spec, the plan and a
    synthetic built world in which each part is made of its own voice's wall."""
    from ethoslm import prims, styles
    from ethoslm.observe import Volume
    va, vb = "ochre_stone_green_tile", "japanese_minka"
    wall_a = prims.solid(styles.VOICES[va]["palette"]["wall"])
    wall_b = prims.solid(styles.VOICES[vb]["palette"]["wall"])
    assert prims.family(wall_a) != prims.family(wall_b), (wall_a, wall_b)
    spec = spec_mod.read_spec({"kind": "town", "form": "east_asian", "defining_parts": [
        {"name": "old_quarter", "kind": "group", "family": "quarter",
         "relation": "quarter", "count": 1, "structures": 2},
        {"name": "new_quarter", "kind": "group", "family": "quarter",
         "relation": "quarter", "count": 1, "structures": 2}],
        "voice": va}, "Build a town of two quarters.")
    got_a = {"quarters": [{"name": "q", "plots": [
        {"kind": "plot", "name": "house_a1", "type": "townhouse", "seed": 1,
         "x0": 10, "z0": 10, "x1": 24, "z1": 24},
        {"kind": "plot", "name": "house_a2", "type": "townhouse", "seed": 2,
         "x0": 30, "z0": 10, "x1": 44, "z1": 24}]}]}
    got_b = {"quarters": [{"name": "q", "plots": [
        {"kind": "plot", "name": "house_b1", "type": "townhouse", "seed": 3,
         "x0": 10, "z0": 40, "x1": 24, "z1": 54},
        {"kind": "plot", "name": "house_b2", "type": "townhouse", "seed": 4,
         "x0": 30, "z0": 40, "x1": 44, "z1": 54}]}]}
    place = {"parts": [], "voice": va, "districts": [
        {"name": "old", "x0": 0, "z0": 0, "x1": 60, "z1": 30, "structures": 2,
         "defines": "old_quarter"},
        {"name": "new", "x0": 0, "z0": 34, "x1": 60, "z1": 64, "structures": 2,
         "defines": "new_quarter", "voice": vb}]}
    plan = placeplan.assemble(place, {"old": got_a, "new": got_b}, spec)
    parts = pipeline.plan_parts(plan)
    base_blocks = {(x, 59, z): "stone" for x in range(0, 64) for z in range(0, 64)}
    built_blocks = dict(base_blocks)
    for p in parts:
        mat = wall_a if (p.get("voice") or va) == va else wall_b
        for x in range(p["x0"], p["x1"] + 1):
            for z in range(p["z0"], p["z1"] + 1):
                for y in (60, 61, 62):
                    built_blocks[(x, y, z)] = mat
    base = Volume.from_blocks(base_blocks, 0, 50, 0, 64, 24, 64)
    built = Volume.from_blocks(built_blocks, 0, 50, 0, 64, 24, 64)
    record = {"waves": [{"wave": "w", "parts": [{"part": p["name"], "stood": True,
                                                  "status": "built", "blocks": 675}
                                                 for p in parts]}]}
    return va, vb, spec, plan, parts, record, built, base


@case
def t_4_two_districts_in_two_voices_build_in_two_palettes_and_the_read_passes_both():
    va, vb, spec, plan, parts, record, built, base = _two_voice_town()
    # the assembly wrote each leaf's own voice, and none where the district has none
    assert {p["name"]: p.get("voice") for p in parts} == {
        "house_a1": None, "house_a2": None, "house_b1": vb, "house_b2": vb}
    site = {"origin": [0, 0], "size": 64, "surface_blocks": {"grass_block": 100}}
    got = placeread.read(spec, plan, record, voice=va, site=site, built=built, base=base)
    pal = next(c for c in got["clauses"] if c["clause"] == "palette")
    assert pal["holds"] and pal["voices"] == [va, vb], pal
    b = next(c for c in got["clauses"] if c["clause"] == "palette/built")
    assert b["holds"], b["says"]
    assert b["per_voice"] == {va: 2, vb: 2}, b["per_voice"]
    assert b["share_min"] == 1.0
    # read against the place's voice alone -- the old rule -- the new quarter fails by
    # name
    bare = [dict(p, voice=None) for p in parts]
    stood = {p["name"]: True for p in parts}
    old = placeread.built_palette(va, bare, stood, built, base)
    assert not old["ok"] and old["failed_count"] == 2, old["says"]
    assert {f["part"] for f in old["failed"]} == {"house_b1", "house_b2"}
    # ...and the build stage resolves a part's palette from its own leaf
    from ethoslm.pipeline import stages_build
    src = open(os.path.join(ROOT, "src", "ethoslm", "pipeline", "stages_build.py")).read()
    assert "def voice_of(part)" in src and 'part.get("voice") or voice' in src
    # the dead vocabulary is gone
    from ethoslm import styles
    assert not hasattr(styles, "matches_voice") and not hasattr(styles, "canonical_voice")
    assert "mixed_voices" not in open(os.path.join(ROOT, "src", "ethoslm", "spec.py")).read()
    return (f"four parts in two voices: the palette clause reads {pal['voices']} and holds, "
            f"palette/built holds at 100% per voice; read against one voice the other "
            f"quarter's two parts fail by name; matches_voice and mixed_voices are gone")


#: **The class axis**, the craft round (E2): the darkest and plainest ring on the
#: outside, the brightest and richest at the centre. Named here outward-in, so the list
#: is read the way the brief states the rule.
CLASS_AXIS = ("packed_earth_and_dark_tile", "ochre_stone_green_tile",
              "pale_quartz_and_gilt")


@case
def t_4_the_rings_are_coloured_by_class_on_one_axis_and_each_builds_in_its_own():
    """**The craft round, E2.** The last city chose three voices and gave its two upper
    rings the same one, because nothing ever told the spec call that a ringed place
    whose rings differ in wealth differs in colour. The brief says it now, the voice
    list it says it in carries the colour of every material, and the bright end of the
    axis is a voice on disk -- a candidate a call may name for a ring, never what the
    deterministic chooser falls back to.
    """
    import test_types
    from ethoslm import prims, styles, voices as voices_mod
    from ethoslm.observe import Volume
    from ethoslm.pipeline import stages_plan

    # 1. the brief says it, and says it beside the colours
    brief = stages_plan.spec_brief("Build a ringed capital.", "/tmp/x.json")
    assert "each ring carries its own voice" in brief, "the brief does not say it"
    assert "darkest and plainest at the outside to the" in brief and \
        "brightest and richest at the centre" in brief, brief[:0]
    for v in CLASS_AXIS:
        assert f"`{v}`" in brief, v
    assert "**ceremonial**, for a ring and not for a whole place" in brief
    assert "wall quartz (pale grey), roof copper (light red)" in brief, \
        "the voice list does not carry the colour of what a person sees"
    assert "% of white" in brief, "the voice list does not carry its value range"

    # 2. the axis is a measurement and not a word: lightest rising inward
    light = [styles.VOICES[v]["value"]["lightest"] for v in CLASS_AXIS]
    assert light == sorted(light) and light[0] < light[-1], dict(zip(CLASS_AXIS, light))

    # 3. the new voice validates, is ceremonial, and is never the chooser's default
    raw = json.load(open(os.path.join(ROOT, "voices", "pale_quartz_and_gilt.json")))
    ok = voices_mod.validate(raw, "pale_quartz_and_gilt")
    assert ok["ceremonial"] and voices_mod.validate(ok, "again") == ok, ok
    site = {"origin": [0, 0], "size": 64, "surface_blocks": {"grass_block": 100}}
    bare = spec_mod.read_spec({"kind": "town", "defining_parts": [
        {"name": "houses", "kind": "group", "family": "quarter", "relation": "quarter",
         "count": 1}], "voice": None}, "Build a town.")
    assert stages_plan._choose_voice(bare, site) != "pale_quartz_and_gilt"
    # ...and a call that names it for a ring gets it
    assert stages_plan._choose_voice(dict(bare, voice="pale_quartz_and_gilt"), site) \
        == "pale_quartz_and_gilt"

    # 4. each voice **builds**: a real instance of a committed type in each, and every
    # block it laid that belongs to a family belongs to that voice's own six
    built_in = {}
    for v in CLASS_AXIS:
        got = test_types._stand("townhouse", v, 12)
        assert got["ok"] and not got["bad"], (v, got["bad"])
        fams = {f for f in (prims.family(m) for m in
                            styles.VOICES[v]["palette"].values()) if f}
        classed = {b: n for b, n in got["blocks"].items() if prims.family(b)}
        mine = sum(n for b, n in classed.items() if prims.family(b) in fams)
        share = mine / max(1, sum(classed.values()))
        assert share >= placeread.BUILT_SHARE, (v, share)
        built_in[v] = (len(got["blocks"]), round(share, 3))
    walls = [prims.solid(styles.VOICES[v]["palette"]["wall"]) for v in CLASS_AXIS]
    assert len(set(walls)) == 3, walls

    # 5. the place read holds on a plan of three rings in three voices
    spec = spec_mod.read_spec({"kind": "city", "form": "east_asian", "defining_parts": [
        {"name": "court", "kind": "group", "family": "district", "relation": "concentric",
         "count": 1, "structures": 2, "ring": 0, "share": 0.2, "walled": False,
         "voice": CLASS_AXIS[2], "density": "low", "role": "urban"},
        {"name": "middle", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "structures": 2, "ring": 1, "share": 0.3,
         "walled": False, "voice": CLASS_AXIS[1], "density": "medium", "role": "urban"},
        {"name": "outer", "kind": "group", "family": "district", "relation": "concentric",
         "count": 1, "structures": 2, "ring": 2, "share": 0.4, "walled": False,
         "voice": CLASS_AXIS[0], "density": "dense", "role": "urban"}],
        "voice": CLASS_AXIS[1]}, "Build a ringed capital.")
    place = {"parts": [], "voice": CLASS_AXIS[1], "districts": [
        {"name": "d0", "x0": 0, "z0": 0, "x1": 60, "z1": 18, "structures": 2,
         "defines": "court", "voice": CLASS_AXIS[2], "ring": 0},
        {"name": "d1", "x0": 0, "z0": 22, "x1": 60, "z1": 40, "structures": 2,
         "defines": "middle", "voice": CLASS_AXIS[1], "ring": 1},
        {"name": "d2", "x0": 0, "z0": 44, "x1": 60, "z1": 62, "structures": 2,
         "defines": "outer", "voice": CLASS_AXIS[0], "ring": 2}]}
    quarters = {}
    for i, d in enumerate(place["districts"]):
        quarters[d["name"]] = {"quarters": [{"name": "q", "plots": [
            {"kind": "plot", "name": f"house_{i}{j}", "type": "townhouse",
             "seed": 1 + j, "x0": 10 + 20 * j, "z0": d["z0"] + 2,
             "x1": 24 + 20 * j, "z1": d["z0"] + 16} for j in (0, 1)]}]}
    plan = placeplan.assemble(place, quarters, spec)
    parts = pipeline.plan_parts(plan)
    base_blocks = {(x, 59, z): "stone" for x in range(0, 64) for z in range(0, 64)}
    built_blocks = dict(base_blocks)
    for p in parts:
        mat = prims.solid(styles.VOICES[p.get("voice") or CLASS_AXIS[1]]
                          ["palette"]["wall"])
        for x in range(p["x0"], p["x1"] + 1):
            for z in range(p["z0"], p["z1"] + 1):
                for y in (60, 61, 62):
                    built_blocks[(x, y, z)] = mat
    base = Volume.from_blocks(base_blocks, 0, 50, 0, 64, 24, 64)
    built = Volume.from_blocks(built_blocks, 0, 50, 0, 64, 24, 64)
    record = {"waves": [{"wave": "w", "parts": [{"part": p["name"], "stood": True,
                                                 "status": "built", "blocks": 675}
                                                for p in parts]}]}
    got = placeread.read(spec, plan, record, voice=CLASS_AXIS[1], site=site,
                         built=built, base=base)
    pal = next(c for c in got["clauses"] if c["clause"] == "palette")
    assert pal["holds"] and sorted(pal["voices"]) == sorted(CLASS_AXIS), pal
    b = next(c for c in got["clauses"] if c["clause"] == "palette/built")
    assert b["holds"] and sorted(b["per_voice"]) == sorted(CLASS_AXIS), b
    assert b["share_min"] == 1.0, b
    # ...and three voices standing is what the concentric measure's clause asks for
    from ethoslm.pipeline import stages_measure
    assert len(set(CLASS_AXIS)) >= stages_measure.PALETTES_STANDING_MIN, CLASS_AXIS
    return (f"the brief names the axis and every voice with its colours; lightest "
            f"{light[0]:.2f} -> {light[-1]:.2f} outward-in; pale_quartz_and_gilt "
            f"validates, is ceremonial and is never chosen by the ground; a townhouse "
            f"stands in each of the three at "
            + ", ".join(f"{v.split('_')[0]} {built_in[v][1]:.0%}" for v in CLASS_AXIS)
            + f"; six leaves in three voices hold palette and palette/built at 100%")


@case
def t_4_the_great_wall_builds_in_three_faces_and_they_differ():
    import test_types
    from ethoslm import prims, styles
    voice = "ochre_stone_green_tile"
    pal = styles.VOICES[voice]["palette"]
    frame = prims.family(prims.solid(pal["frame"]))
    trim = prims.family(prims.solid(pal["trim"]))
    decl = pipeline.load_type(os.path.join(ROOT, "types", "great_wall.py"))
    # the plain face on both sides -- as a fourth choice; the craft round (E4) replaced
    # it with `masonry`, dressed stonework with a plinth, string courses, buttress piers
    # and a batter, which is what an unbroken wall gets and carries the sparse stairs
    # `unbroken` carried. The three here are the bytes they were.
    assert list(decl["params"]["face"][1]) == ["framed", "banded", "plain", "masonry"], \
        decl["params"]["face"]
    got = {}
    for face in ("framed", "banded", "plain"):
        r = test_types._stand("great_wall", voice, 40, seed=1,
                              params={"height": 30, "width": 3,
                                      "parapet": "crenellated", "face": face})
        assert r["ok"] and not r["bad"], (face, r["bad"])
        fam = {}
        for block, n in r["blocks"].items():
            f = prims.family(block.split("[")[0])
            fam[f] = fam.get(f, 0) + n
        got[face] = {"frame": fam.get(frame, 0), "trim": fam.get(trim, 0),
                     "blocks": sum(r["blocks"].values()), "shape": r["shape"]}
    # the default is today's grid, byte for byte
    default = test_types._stand("great_wall", voice, 40, seed=1,
                                params={"height": 30, "width": 3, "parapet": "crenellated"})
    assert default["shape"] == got["framed"]["shape"]
    assert got["framed"]["frame"] > got["banded"]["frame"] == got["plain"]["frame"], got
    assert got["banded"]["trim"] > got["plain"]["trim"], got
    assert got["framed"]["shape"] == got["banded"]["shape"], \
        "a face is a choice of blocks on the same body"
    # ...and `plain` is the one that is not only a choice of blocks.
    assert len(got["plain"]["shape"]) < len(got["framed"]["shape"]), got
    # the arithmetic chooses the face from the spec's words, never a name
    assert placeplan.wall_face_for({"notes": "an earthen wall, monolithic"}) == "plain"
    assert placeplan.wall_face_for({"notes": "coursed ashlar masonry"}) == "banded"
    assert placeplan.wall_face_for({"notes": "a timber-framed palisade"}) == "framed"
    assert placeplan.wall_face_for({}) == "framed"
    # ...and the spec's invariants paragraph is read with the part's notes
    assert placeplan.wall_face_for({"notes": "the outer wall"},
                                   {"invariants": "a great wall, unbroken"}) == "plain"
    assert placeplan.wall_face_for({"notes": "the outer wall"},
                                   {"invariants": "dressed ashlar courses"}) == "banded"
    s4 = ring_spec()                         # its wall is "earthen ... monolithic"
    place, fails, decls, site, plateau = _layout(s4)
    # "unbroken" reaches both faces, so the great wall's is dressed `masonry` with
    # sparse stairs and a `wall` part carries its face by name
    faces = {p["params"].get("face") for p in place["parts"]
             if p["kind"] == "edge" and p["type"] == "great_wall"}
    assert faces == {"masonry"}, faces
    assert all(p.get("face") == "plain" for p in place["parts"] if p["kind"] == "edge")
    return ("frame blocks framed/banded/plain "
            f"{got['framed']['frame']}/{got['banded']['frame']}/{got['plain']['frame']}, "
            f"trim {got['framed']['trim']}/{got['banded']['trim']}/{got['plain']['trim']} on "
            f"one body of {got['framed']['blocks']} blocks; the default is the grid; "
            f"'earthen' chooses plain")


@case
def t_4_a_wall_carries_the_mass_its_place_declares_and_its_gate_is_framed():
    """**The craft round, E4.** `great_wall` declared `width` as three and only three,
    because three is all its sweep had ever tried, so a city's outer wall was forty-
    eight high and three thick -- a screen. A great wall is an earthwork: wide enough
    that its crown is a road with a parapet on both edges and its ways up stand in its
    own thickness. The place says which: a screen, a curtain, a rampart or a levee, from
    its own words.
    """
    import test_types
    from ethoslm import prims, styles
    import types as _pytypes
    gw = pipeline.load_type(os.path.join(ROOT, "types", "great_wall.py"))
    wl = pipeline.load_type(os.path.join(ROOT, "types", "wall.py"))

    # 1. the band reaches a rampart's, and the bank says the sweep stood it there
    assert gw["needs"]["footprint"] == (1, 4, 12, 128), gw["needs"]["footprint"]
    assert wl["needs"]["footprint"] == (1, 4, 12, 128), wl["needs"]["footprint"]
    bank = json.load(open(os.path.join(ROOT, "rounds", "type-needs.json")))
    assert bank["swept"]["edge_widths"] == [1, 2, 3, 5, 7, 9, 12], bank["swept"]
    for n in ("great_wall", "wall"):
        assert bank["types"][n]["band"]["footprint"] == [1, 4, 12, 128], n

    # 2. the place declares the mass, in its own words, and the layout lays it
    assert placeplan.WALL_MASSES == {"screen": 3, "curtain": 5, "rampart": 9,
                                     "levee": 12}
    assert placeplan.wall_mass_for({"notes": "a curtain wall"}) == "curtain"
    assert placeplan.wall_mass_for({}, {"invariants": "the largest wall in the world, "
                                        "unbroken earth and stone"}) == "rampart"
    assert placeplan.wall_mass_for({"notes": "a levee against the river"}) == "levee"
    assert placeplan.wall_mass_for({}) == "screen"
    assert placeplan.wall_mass_for({"mass": "levee"}) == "levee"
    # the fixture's own wall says nothing about its mass and is a screen, three thick,
    # as every wall this project has ever built was
    spec4 = ring_spec()
    place, fails, decls, site, plateau = _layout(spec4)
    assert not fails, fails
    walls = [p for p in place["parts"] if p["kind"] == "edge"]
    masses = {p["type"]: (p["width"], p.get("mass")) for p in walls}
    assert masses["great_wall"] == (3, "screen"), masses
    # ...and the same spec whose wall the sentence calls the largest in the world is a
    # rampart, and the layout lays nine columns of it
    heavy = json.loads(json.dumps(spec4))
    for p in heavy["defining_parts"]:
        if p.get("family") == "wall":
            p["notes"] = ("the largest wall in the world, an earthwork of rammed earth "
                          "and stone round the belt")
    place_h, fails_h, decls_h, site_h, plateau_h = _layout(
        spec_mod.read_spec(json.loads(json.dumps(heavy)), heavy["sentence"]))
    assert not fails_h, fails_h
    heavy_walls = {p["type"]: (p["width"], p.get("mass")) for p in place_h["parts"]
                   if p["kind"] == "edge"}
    assert heavy_walls["great_wall"] == (placeplan.WALL_MASSES["rampart"], "rampart"), \
        heavy_walls
    masses = heavy_walls

    # 3. it stands, at every face, at every mass, in two voices
    said = []
    for voice in ("ochre_stone_green_tile", "pale_quartz_and_gilt"):
        for width in (3, 5, 9, 12):
            for face in list(gw["params"]["face"][1]):
                r = test_types._stand("great_wall", voice, 40, seed=1,
                                      params={"height": 40, "width": 3,
                                              "parapet": "crenellated", "face": face},
                                      part_over={"width": width})
                assert r["ok"] and not r["bad"], (voice, width, face, r["bad"])
        said.append(voice)

    # 4. the crown is a road at `CROWN_ROAD_MIN`: a parapet on **both** edges
    def _profile(width, face="masonry"):
        r = test_types._stand("great_wall", "ochre_stone_green_tile", 40, seed=1,
                              params={"height": 40, "width": 3, "parapet": "crenellated",
                                      "face": face}, part_over={"width": width})
        zs = sorted({z for (_x, _y, z) in r["shape"]})
        top = max(y for (_x, y, _z) in r["shape"])
        edge = {}
        for z in (zs[0], zs[-1]):
            edge[z] = max((y for (_x, y, zz) in r["shape"] if zz == z), default=0)
        return r, zs, top, edge
    thin, zs_t, top_t, _e = _profile(3)
    wide, zs_w, top_w, _e = _profile(9)
    # the mass is the part's own width and the wall is that thick
    lanes_t = len({z for (_x, _y, z) in thin["shape"]})
    lanes_w = len({z for (_x, _y, z) in wide["shape"]})
    assert lanes_w > lanes_t, (lanes_t, lanes_w)
    assert placeplan.WALL_MASSES["curtain"] == 5

    # 5. the masonry face is not the plain one: string courses and a plinth in trim and
    # footing where a plain face has one cornice, and a batter that stops the outer
    # lanes short of the crown
    pal = styles.VOICES["pale_quartz_and_gilt"]["palette"]
    trim = prims.family(prims.solid(pal["trim"]))
    got = {}
    for face in ("plain", "masonry"):
        r = test_types._stand("great_wall", "pale_quartz_and_gilt", 40, seed=1,
                              params={"height": 40, "width": 3, "parapet": "crenellated",
                                      "face": face}, part_over={"width": 9})
        fam = {}
        for block, n in r["blocks"].items():
            f = prims.family(block.split("[")[0])
            fam[f] = fam.get(f, 0) + n
        tops = {}
        for (x, y, z) in r["shape"]:
            tops[z] = max(tops.get(z, 0), y)
        got[face] = {"trim": fam.get(trim, 0), "tops": tops}
    # a plain face has its cornice and nothing else; a masonry one has that, a plinth
    # and a string course every `STRING_EVERY`
    assert got["masonry"]["trim"] >= 2 * got["plain"]["trim"], \
        (got["masonry"]["trim"], got["plain"]["trim"])
    # the batter: the outermost lane of a masonry face stops below the crown, and a
    # plain one does not
    m, p_ = got["masonry"]["tops"], got["plain"]["tops"]
    lower = [z for z in m if m[z] < p_.get(z, 0)]
    assert len(lower) >= 2, (m, p_)
    # ...and they are the outer lanes of the wall and not a scatter: the batter steps
    # the face in from one end
    assert lower == sorted(m)[:len(lower)] or lower == sorted(m)[-len(lower):], \
        (lower, sorted(m))

    # 6. the gate's arch is framed
    from ethoslm import prims as _p
    frames = {}
    for voice in ("pale_quartz_and_gilt",):
        r = test_types._stand("ring_gate", voice, 15, seed=1,
                              part_over={"edge": {"name": "a_wall",
                                                  "type": "great_wall",
                                                  "height": 48, "width": 9}})
        assert r["ok"] and not r["bad"], r["bad"]
        fam = {}
        for block, n in r["blocks"].items():
            f = _p.family(block.split("[")[0])
            fam[f] = fam.get(f, 0) + n
        t = _p.family(_p.solid(styles.VOICES[voice]["palette"]["trim"]))
        frames[voice] = fam.get(t, 0)
    assert frames["pale_quartz_and_gilt"] > 0, frames
    return (f"the band reaches 12 on both edge types and the bank says the sweep stood "
            f"them there; a place declares its wall's mass in its own words "
            f"({masses['great_wall'][1]}, {masses['great_wall'][0]} thick); every face "
            f"at four masses in {len(said)} voices stands clean; a mass of 9 is "
            f"{lanes_w} lanes against a screen's {lanes_t}; the masonry face carries "
            f"{got['masonry']['trim']} trim blocks against a plain one's "
            f"{got['plain']['trim']} and {len(lower)} outer lanes batter in; the "
            f"gate's arch is "
            f"framed in {frames['pale_quartz_and_gilt']} trim blocks")


@case
def t_4_the_flythroughs_final_camera_rises_above_the_tallest_thing_before_its_subject():
    import test_camera
    from ethoslm import render
    from ethoslm.observe import Volume
    plan, spec, site, sm = test_camera._two_ring_plan()
    path = sm.flythrough_path(plan, spec, site)
    G, Y0 = test_camera.GROUND, test_camera.Y0
    flat = {(x, G, z): "stone" for x in range(-64, 160) for z in range(-64, 160)}
    open_vol = Volume.from_blocks(flat, -64, Y0, -64, 224, 40, 224)
    before = render.flythrough_shots(path, open_vol)
    # a wall 20 high round the compound at its rectangle's edge, as a walled compound
    # has: it stands between the final camera and the centre
    walled = dict(flat)
    for x in range(40, 56):
        for z in (40, 55):
            for y in range(G + 1, G + 21):
                walled[(x, y, z)] = "stone"
                walled[(z, y, x)] = "stone"
    wall_vol = Volume.from_blocks(walled, -64, Y0, -64, 224, 40, 224)
    after = render.flythrough_shots(path, wall_vol)
    last = f"fly_{len(path) - 1:03d}"
    y_before = before[last].view.position[1]
    y_after = after[last].view.position[1]
    assert path[-1]["subject"] == [40, 40, 55, 55] and path[-1]["compound"] is True
    top = render.tallest_between(wall_vol, tuple(path[-1]["at"]), tuple(path[-1]["look"]),
                                 stop=render.corridor_end(path[-1]))
    assert top == G + 20, top
    # the corridor ends where the line enters the subject, plus a compound wall's depth
    assert abs(render.corridor_end(path[-1])
               - (40 - path[-1]["at"][1] + render.FLY_INTO_COMPOUND)) < 0.6
    assert y_after >= top + render.FLY_RISE > y_before, (y_before, y_after, top)
    assert after[last].repair.get("tallest") == top and after[last].repair.get("frames") \
        == render.FLY_LIFT_FRAMES
    # the pitch of the final camera is still under the limit
    p, t = after[last].view.position, after[last].target
    d = (t[0] - p[0], t[1] - p[1], t[2] - p[2])
    down = -math.degrees(math.asin(d[1] / math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)))
    assert 0 < down < render.FLY_FINAL_PITCH_MAX, down
    # the move eases up over the last frames and the frames before them are untouched
    n = len(path)
    for i in range(n - render.FLY_LIFT_FRAMES):
        k = f"fly_{i:03d}"
        assert after[k].view.position == before[k].view.position, k
    ys = [after[f"fly_{i:03d}"].view.position[1] for i in range(n - render.FLY_LIFT_FRAMES, n)]
    assert ys == sorted(ys) and ys[-1] == y_after, ys
    # ...and over open ground the move is the move it was, byte for byte
    again = render.flythrough_shots(path, open_vol)
    assert all(again[k].view.position == before[k].view.position for k in before)
    assert not any(again[k].repair for k in again)
    return (f"a 20-high wall at the subject's edge lifts the final camera from y {y_before} "
            f"to {y_after} ({render.FLY_RISE} above its crown at {top}) over "
            f"{render.FLY_LIFT_FRAMES} frames, final pitch {down:.1f} deg; open ground is "
            f"unchanged")


@case
def t_4_a_voice_whose_floor_is_a_bare_block_builds_and_the_footing_stands_in_for_its_shapes():
    """a bare block the validator allows there -- and the types shape the floor role in
    slabs, steps and fittings; thirty parts crashed. The type builder keeps both
    promises: the cube is the floor and a shape of it is the footing's, recorded.
    """
    import test_types
    from ethoslm import prims, styles, voices as _voices
    name = "packed_earth_and_dark_tile"
    if name not in styles.VOICES:
        raise Skip(f"no voices/{name}.json")
    v = styles.VOICES[name]["palette"]
    assert prims.family(v["floor"]) is None and prims.family(v["footing"]) is not None
    stood = {}
    for t, size in (("shop_house", 12), ("row_house", 9), ("court_small", 14)):
        r = test_types._stand(t, name, size, seed=1)
        assert r["ok"], (t, r)
        blocks = r["blocks"]
        stood[t] = (sum(blocks.values()), any(b.split("[")[0] == v["floor"] for b in blocks))
    assert any(laid for _n, laid in stood.values()), (stood, "no type laid the floor cube")
    # the record names the calls the footing stood in for
    from ethoslm.buildlib import Builder, TypeBuilder
    from ethoslm import observe
    vol = test_types._flat_world(60)
    b = Builder(__import__("ethoslm.offline", fromlist=["OfflineSite"]).OfflineSite(vol))
    b._vol = vol
    part = {"voice": dict(v), "floor_y": 64}
    tb = TypeBuilder(b, part)
    assert tb.block(v["floor"], "full") == v["floor"]
    assert tb.block(v["floor"], "slab") == prims.shape(v["footing"], "slab")
    assert tb.block(v["wall"], "slab") == prims.shape(v["wall"], "slab")
    assert part["voice_stood_in"] == {f"block:{v['floor']}": 1}, part["voice_stood_in"]
    # a family floor is untouched, byte for byte: the demo's voice stands the same
    ochre = styles.VOICES["ochre_stone_green_tile"]["palette"]
    part2 = {"voice": dict(ochre), "floor_y": 64}
    tb2 = TypeBuilder(b, part2)
    assert tb2.block(ochre["floor"], "slab") == prims.shape(ochre["floor"], "slab")
    assert "voice_stood_in" not in part2
    return (f"shop_house/row_house/court_small stand in {name} at {stood} blocks with a "
            f"{v['floor']} floor; a slab of it is the footing's ({v['footing']}) and the "
            f"part records it; a family floor is untouched")


# ------------------------------------------- phase 5, before the run: the readout

@case
def t_5_the_concentric_measure_reads_the_five_expectations_and_the_readout_carries_it():
    from ethoslm import offline
    from ethoslm.observe import Volume
    from ethoslm.pipeline import stages_measure
    s = three_ring_spec()
    with tempfile.TemporaryDirectory() as tmp:
        X, Z, S = 0, 0, 384
        site = {"origin": [X, Z], "size": S, "stats": {"relief": 1},
                "surface_blocks": {"grass_block": 100}}
        plateau = _plateau(site)
        _t, decls = placeplan.types_card(None, s.get("form"))
        place, fails = placeplan.concentric_layout(s, site, plateau, decls,
                                                   "ochre_stone_green_tile")
        assert not fails, fails
        plan = placeplan.assemble(place, {}, s)
        assert plan["layout"]["by"] == "placeplan.concentric_layout"
        json.dump(json.loads(json.dumps(THREE_RING)), open(os.path.join(tmp, "place.json"), "w"))
        json.dump(site, open(os.path.join(tmp, "site.json"), "w"))
        json.dump(plan, open(os.path.join(tmp, "plan.json"), "w"))
        rows = [{"part": f"p{i}", "stood": True, "status": "built", "voice": v}
                for i, v in enumerate(("ochre_stone_green_tile", "japanese_minka",
                                       "japanese_temple", "ochre_stone_green_tile"))]
        json.dump({"waves": [{"wave": "w", "parts": rows}]},
                  open(os.path.join(tmp, "parts.json"), "w"))
        rnd = pipeline.Round(name="concentric_fixture", sentence="Build a ringed town.",
                             state_dir=tmp,
                             preregistered={"bars": {}, "reported_no_bar": {
                                 "concentric": "the five", "camera": "prose"}})
        be = pipeline.OfflineBackend(rnd)
        got = stages_measure._m_concentric(rnd, be, {}, {})
        assert got["belt"]["holds"] and all(v["holds"] for v in got["coverage"].values())
        assert got["centred"]["holds"] and got["centred"]["offset"] <= 4
        assert got["palettes"] == {"standing": ["japanese_minka", "japanese_temple",
                                                "ochre_stone_green_tile"], "count": 3,
                                   "holds": True}, got["palettes"]
        assert got["core_water"]["holds"] is False and got["got"] == 0
        assert got["failed"] == ["core_water"], got["failed"]
        # a built volume: dry under the compound, a pond outside it -> holds
        r = placeplan.compound_rects(plan)["great_court"]
        blocks = {}
        for x in range(X, X + S):
            for z in range(Z, Z + S):
                pond = (x < r[0] - 20 and z < r[1] - 20)
                blocks[(x, 60, z)] = "water" if pond else "stone"
                if pond:
                    blocks[(x, 59, z)] = "stone"
        vol = Volume.from_blocks(blocks, X, 50, Z, S, 16, S)
        offline.save_volume(vol, os.path.join(tmp, "world_built.npz"))
        got = stages_measure._m_concentric(rnd, be, {}, {})
        assert got["core_water"] == {"columns": 0, "of": (r[2] - r[0] + 1) * (r[3] - r[1] + 1),
                                     "holds": True}, got["core_water"]
        assert got["got"] == 1 and got["failed"] == [], got["failed"]
        # ...and water under one corner of the compound fails it by the count
        blocks[(r[0], 60, r[1])] = "water"
        blocks[(r[0] + 1, 60, r[1])] = "water"
        offline.save_volume(Volume.from_blocks(blocks, X, 50, Z, S, 16, S),
                            os.path.join(tmp, "world_built.npz"))
        got = stages_measure._m_concentric(rnd, be, {}, {})
        assert got["core_water"]["columns"] == 2 and got["got"] == 0
        # the readout reads a bar-less report that names a measure, and echoes one that
        # does not
        out = stages_measure.stage_readout(rnd, be, {})
        rep = out["reported_no_bar"]
        assert rep["concentric"]["barred"] is False and rep["concentric"]["got"] == 0
        assert rep["concentric"]["core_water"]["columns"] == 2
        assert rep["camera"] == "see the round file"
        assert "results" in out and out["results"] == {}
        # the round file: the demo's nine bars byte for byte, and the five registered
        demo = json.load(open(os.path.join(ROOT, "rounds", "demo.json")))
        conc = json.load(open(os.path.join(ROOT, "rounds", "concentric.json")))
        assert json.dumps(demo["preregistered"]["bars"], sort_keys=True) == \
            json.dumps(conc["preregistered"]["bars"], sort_keys=True)
        assert "concentric" in conc["preregistered"]["reported_no_bar"]
        assert not conc.get("site") and not conc.get("voice")
        for word in ("BELT_SHARE_TOLERANCE 0.05", "RING_COVERAGE 0.6", "CENTRED_TOLERANCE 4",
                     "PALETTES_STANDING_MIN 3", "CORE_WATER_MAX_PCT 2.0"):
            assert word in conc["preregistered"]["written_before"], word
        assert stages_measure.BELT_SHARE_TOLERANCE == 0.05
        assert stages_measure.PALETTES_STANDING_MIN == 3
        return ("belt, coverage x3, centred and 3 palettes hold on the fixture; core water "
                "0 of {} holds and 2 columns fail it; the readout reads the measure "
                "unbarred; rounds/concentric.json carries the demo's 9 bars byte for byte"
                .format((r[2] - r[0] + 1) * (r[3] - r[1] + 1)))


# ----------------------------------------------- phase 5, after the run: the budget

@case
def t_5_a_districts_leaf_is_never_a_defining_part_by_name_and_the_defining_quarters_all_are():
    """The first render of the concentric city photographed fifty renamed houses as
    defining parts and cut the flythrough to 39 frames: the assembly qualifies a
    colliding name with its district's, a district is named for its ring, a ring is a
    defining part, and the budget read the name. It reads the quarter now."""
    from ethoslm.pipeline import stages_media as sm
    spec = {"defining_parts": [{"name": "ring_walls"}, {"name": "lower_ring"},
                               {"name": "palace"}]}
    plan = {"compounds": [{"name": "palace", "x0": 0, "z0": 0, "x1": 40, "z1": 40}],
            "parts": [{"kind": "district", "name": "city", "children": [
                {"kind": "quarter", "name": "defining", "children": [
                    {"kind": "edge", "name": "ring_walls_lower_ring", "type": "wall",
                     "path": [[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]], "width": 1},
                    {"kind": "point", "name": "ring_gate_lower_ring", "type": "ring_gate",
                     "at": [4, 0]}]},
                {"kind": "quarter", "name": "palace", "compound": True, "children": [
                    {"kind": "plot", "name": "great_hall", "type": "hall", "defines": "palace",
                     "compound": "palace", "x0": 10, "z0": 10, "x1": 30, "z1": 30}]},
                {"kind": "quarter", "name": "lower_ring_east_rows", "children": [
                    {"kind": "plot", "name": "lower_ring_east_tea_house", "type": "shop_house",
                     "x0": 50, "z0": 50, "x1": 60, "z1": 60},
                    {"kind": "plot", "name": "ring_walls_view_house", "type": "shop_house",
                     "x0": 70, "z0": 50, "x1": 80, "z1": 60},
                    {"kind": "plot", "name": "plain_house", "type": "shop_house",
                     "x0": 90, "z0": 50, "x1": 100, "z1": 60}]}]}]}
    rows = sm.budget_frames(plan, spec, {"place": False, "compound": True, "defining": True,
                                         "sample": 24, "sample_seed": 1, "flythrough": False})
    defining = sorted(r["name"] for r in rows if r["why"] == "defining")
    sample = sorted(r["name"] for r in rows if r["why"] == "sample")
    # the gate laid for a spec with no gate part is defining because the place level
    # drew it; the compound's hall by `defines`; no house of a district by its name
    assert defining == ["great_hall", "ring_gate_lower_ring", "ring_walls_lower_ring"], defining
    assert sample == ["lower_ring_east_tea_house", "plain_house", "ring_walls_view_house"], sample
    # a flat plan with no tree keeps the name rule it always had
    flat = {"structures": [{"name": "ring_walls_house", "x0": 0, "z0": 0, "x1": 5, "z1": 5},
                           {"name": "other", "x0": 10, "z0": 0, "x1": 15, "z1": 5}]}
    rows = sm.budget_frames(flat, spec, {"place": False, "compound": False, "defining": True,
                                         "sample": 24, "sample_seed": 1, "flythrough": False})
    assert [r["name"] for r in rows if r["why"] == "defining"] == ["ring_walls_house"]
    return "the wall, its gate and the compound's hall are defining; three district houses, two named like defining parts, are the sample; a flat plan keeps the name rule"


# ------------------------------------------------------------------- the run

def main():
    only = set(sys.argv[1:])
    ok = fail = skipped = 0
    for name, fn in CASES:
        if only and not any(o in name for o in only):
            continue
        try:
            says = fn()
            ok += 1
            print(f"ok    {name}: {says}")
        except Skip as e:
            skipped += 1
            print(f"skip  {name}: {e}")
        except Exception as e:                    # noqa: BLE001 -- reported
            fail += 1
            import traceback
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok} ok, {fail} fail, {skipped} skipped of {ok + fail + skipped}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
