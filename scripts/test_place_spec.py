"""What a *city* needs squared off first. One case each.

    $PY scripts/test_place_spec.py

A1. **Needs per defining part.** A defining part carries its own `needs`, its own
allowed `forms` and its own `density`; the search reads the relief band off the **core**
rather than off the whole footprint, and a candidate whose middle is level and whose
outer ring is a hillside is accepted. A2. **Concentric, in the place read.** Ring walls
are nested closed loops, every ring has a gate on it and every gate is on an arterial,
an arterial crosses a ring at a gate or not at all, what the place is centred on is
inside the innermost ring, and every structure is inside the outermost one. a three-ring
fixture passes. A3. **The wall grows.** The committed `wall` takes `height` to 20 and
`width` to 5 and stands clean on a 160-column loop at relief 12 and on a 600-column
loop. A4. **The render budget.** At 400 structures a card per building is six hours, so
the shot list is the two whole-place frames, the defining parts, a seeded sample of 24,
and a flythrough written as frames. A5. **Search radius and terraform for a city.** The
grid reaches 8,192, a city may terraform its core and nothing else, and a search over a
fixed spec answers with a 512 footprint or with a named reason.

The cases that need `out/`.
"""
import importlib.util
import json
import shutil
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import offline, pipeline, placeplan, placeread, spec as spec_mod  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something `out/` has and this checkout does not."""


def _find_site():
    p = os.path.join(ROOT, "scripts", "find_site.py")
    s = importlib.util.spec_from_file_location("find_site", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# A case whose input is a model call is a case that measures the model.

CITY_SENTENCE = "Build Ringed City."

CITY_SPEC = {
    "kind": "city",
    "form": "east_asian",
    "defining_parts": [
        {"name": "palace", "kind": "area", "family": "palace", "relation": "centre",
         "count": 1, "structures": 1, "density": "sparse",
         "forms": ["civic"],
         "needs": {"max_relief": 8, "plateau": 96},
         "notes": "the compound at the middle; it stands on the plateau"},
        {"name": "ring_wall", "kind": "edge", "family": "wall",
         "relation": "concentric", "count": 3, "structures": 0,
         "forms": ["fortification"],
         "needs": {"max_relief": 200},
         "notes": "three concentric circuits; a wall climbs what is there"},
        {"name": "ring_gate", "kind": "point", "family": "gate", "relation": "gateway",
         "count": 3, "structures": 0, "forms": ["fortification"],
         "notes": "one gate on each ring, on the arterial"},
        {"name": "inner_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "structures": 40, "density": "sparse",
         "needs": {"max_relief": 20},
         "notes": "large courtyard houses, gardens, low density"},
        {"name": "middle_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "structures": 120, "density": "medium",
         "needs": {"max_relief": 30},
         "notes": "courtyard houses and shop-houses"},
        {"name": "lower_ring", "kind": "group", "family": "district",
         "relation": "concentric", "count": 1, "structures": 200, "density": "dense",
         "needs": {"max_relief": 40},
         "notes": "small courtyard houses and workshops, dense"},
        {"name": "agrarian_ring", "kind": "group", "family": "quarter",
         "relation": "concentric", "count": 1, "structures": 40, "density": "sparse",
         "needs": {"max_relief": 40},
         "notes": "farmhouses and fields, sparse"},
    ],
    "voice": None,
    "notes": "a fixture",
}


# --------------------------------------------------- A1. needs per defining part

@case
def t_a1_a_defining_part_carries_its_own_needs_forms_and_density():
    """The schema half of A1: read, completed, and refused by name where it is wrong."""
    s = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    by = {p["name"]: p for p in s["defining_parts"]}
    assert by["palace"]["needs"]["max_relief"] == 8.0, by["palace"]
    assert by["palace"]["needs"]["plateau"] == 96, by["palace"]
    assert by["palace"]["forms"] == ["civic"], by["palace"]
    assert by["lower_ring"]["density"] == "dense", by["lower_ring"]
    # A part that says nothing still reads, and says nothing: `None` everywhere, which
    # is what makes A1 a thing a spec may use rather than a thing every spec must.
    assert by["ring_gate"]["needs"] == dict(spec_mod.PART_NEEDS_DEFAULT)
    bare = spec_mod.read_part({"name": "well", "kind": "point", "family": "gate",
                               "relation": "gateway"}, "defining_parts[0]")
    assert bare["needs"] == dict(spec_mod.PART_NEEDS_DEFAULT), bare
    assert bare["density"] is None and bare["forms"] is None, bare
    # Reading a spec that has already been read gives the identical spec: five stages
    # call `Round.place_spec()` and a per-part need that changed on the second read
    # would be a need the plan and the search disagreed about.
    assert json.dumps(spec_mod.read_spec(json.loads(json.dumps(s)), CITY_SENTENCE),
                      sort_keys=True) == json.dumps(s, sort_keys=True)
    # ...and each of the three is refused by name rather than dropped.
    for bad, want in (({"needs": {"max_reliefs": 8}}, "no field called"),
                      ({"forms": ["gothic"]}, "forms is drawn from"),
                      ({"density": "packed"}, "density is one of")):
        p = dict(CITY_SPEC["defining_parts"][0]) | bad
        try:
            spec_mod.read_part(p, "defining_parts[0]")
            raise AssertionError(f"{bad} was accepted")
        except spec_mod.SpecError as e:
            assert want in str(e), (bad, str(e))
    # The density is arithmetic and it is this module's, not the model's.
    assert spec_mod.columns_per_plot(by["lower_ring"]) < spec_mod.COLUMNS_PER_PLOT \
        < spec_mod.columns_per_plot(by["inner_ring"])
    return (f"7 defining parts read; palace needs {by['palace']['needs']['max_relief']} "
            f"of relief on a {by['palace']['needs']['plateau']} plateau, the rings run "
            f"{by['inner_ring']['density']}/{by['middle_ring']['density']}/"
            f"{by['lower_ring']['density']} at "
            f"{spec_mod.columns_per_plot(by['inner_ring'])}/"
            f"{spec_mod.columns_per_plot(by['middle_ring'])}/"
            f"{spec_mod.columns_per_plot(by['lower_ring'])} columns a structure")


@case
def t_a1_a_districts_structures_is_a_share_and_a_count_only_inside_the_band():
    """A library defect found by running it, with the case kept."""
    doc = json.loads(json.dumps(CITY_SPEC))
    for p, n in zip([q for q in doc["defining_parts"] if q["kind"] == "group"],
                    (1, 6, 4, 2)):
        p["structures"] = n
    s = spec_mod.read_spec(doc, CITY_SENTENCE)
    lo, hi = s["size_band"]
    # The middle of the band, whatever the band is.
    assert s["structures"] == int(round((lo + hi) / 2)), (s["structures"], lo, hi)
    assert spec_mod.in_band(s, s["structures"]), s
    groups = {p["name"]: p["structures"] for p in s["defining_parts"]
              if p["kind"] == "group"}
    assert sum(groups.values()) == s["structures"], groups
    # The **ratios** are the model's and they survive: six-to-one between the ring it
    # declared at 6 and the ring it declared at 1 is what it actually said, and it is
    # what comes out.
    assert groups["middle_ring"] == 6 * groups["inner_ring"], groups
    assert groups["lower_ring"] == 4 * groups["inner_ring"], groups
    assert min(groups.values()) >= 1, groups
    # A sum **at or above** the floor is a count, and too big a count is the ceiling's
    # job. 1,200 still scales to 400.
    big = json.loads(json.dumps(CITY_SPEC))
    for p in big["defining_parts"]:
        if p["kind"] == "group":
            p["structures"] = 300
    b = spec_mod.read_spec(big, CITY_SENTENCE)
    assert b["scaled_from"], "a count over the ceiling was not scaled"
    assert b["structures"] <= spec_mod.structures_ceiling(b["kind"]), b["structures"]
    # ...and a spec whose groups already sum inside the band is untouched.
    exact = json.loads(json.dumps(CITY_SPEC))
    e = spec_mod.read_spec(exact, CITY_SENTENCE)
    assert e["structures"] == 400 and "scaled_from" not in e, e["structures"]
    assert {p["name"]: p["structures"] for p in e["defining_parts"]
            if p["kind"] == "group"} == {"inner_ring": 40, "middle_ring": 120,
                                         "lower_ring": 200, "agrarian_ring": 40}
    return (f"shares 1/6/4/2 -> {groups} summing to {s['structures']} in "
            f"{s['size_band']}; a count of 1,200 still scales to {b['structures']} by "
            f"the ceiling; a sum already in band is untouched at {e['structures']}")


@case
def t_a1_the_voice_validator_reads_what_it_writes():
    """A library defect found by running it, with the case kept.

        The first voice a place spec ever authored went `read_spec` -> `validate` ->
        `stage_place_spec` -> `author` -> `validate`, and the second pass **refused it by
        name for a field the model had never written**: `value`, the luma spread `validate`
        itself measures off the roles and adds to its own output. A validator that will not
        read what it just wrote stops the round at the first stage.
        
    """
    from ethoslm import voices
    doc = {"name": "t", "roles": {"wall": "stone_bricks", "footing": "cobblestone",
                                  "frame": "oak", "roof": "deepslate_tiles",
                                  "trim": "andesite", "floor": "smooth_stone"},
           "roof": {}, "notes": {"blurb": "a fixture"}}
    once = voices.validate(json.loads(json.dumps(doc)), where="a fixture")
    assert "value" in once and "spread" in once["value"], once
    twice = voices.validate(json.loads(json.dumps(once)), where="a fixture, again")
    assert json.dumps(twice, sort_keys=True) == json.dumps(once, sort_keys=True)
    # ...and a field that really is unknown is still refused by name.
    try:
        voices.validate(dict(once, colour="green"), where="a fixture")
        raise AssertionError("an unknown field was accepted")
    except voices.VoiceError as e:
        assert "colour" in str(e), str(e)
    return (f"validate(validate(v)) is validate(v) byte for byte, spread "
            f"{once['value']['spread']}; an unknown field is still refused by name")


@case
def t_a1_a_stage_that_stops_the_round_stops_it_from_its_own_answer():
    """A library defect found by running it, with the case kept."""
    from ethoslm.pipeline import round as rmod
    assert rmod._stopped({"status": "error", "stop": True, "error": "the reason"}) \
        == "the reason"
    assert rmod._stopped({"plan": {"stop": True, "error": "nested"}}) == "nested"
    assert rmod._stopped({"status": "read", "spec": {"kind": "city"}}) is None
    assert rmod._stopped({}) is None
    return ("a stage stops the round from its own answer and from its parts'; a stage "
            "that says nothing about stopping does not stop it")


#: Where the synthetic candidates stand. Far from anything: `groundread.excluded` reads
#: the durable reservations.
FIX_ORIGIN = 200_000


def _field(fs, size, core, core_relief, outer_relief):
    """One synthetic candidate: a level core in a footprint with a hillside round it."""
    h = np.zeros((size, size), np.int32)
    i, j, n = fs.core_window(size, core)
    # The outer ring falls away from the core, linearly, to `outer_relief`.
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    d = np.maximum(np.abs(xx - size // 2), np.abs(yy - size // 2)).astype(np.float64)
    d = np.clip((d - n / 2) / max(1.0, (size - n) / 2), 0.0, 1.0)
    h = (d * outer_relief).astype(np.int32)
    # ...and the core has a little relief of its own, so the band has something to read.
    h[i:i + n, j:j + n] += (np.arange(n) % (core_relief + 1)).astype(np.int32)[:, None]
    return fs.Field(FIX_ORIGIN, FIX_ORIGIN, h,
                    np.zeros((size, size), bool), np.zeros((size, size), bool),
                    gravity=np.zeros((size, size), bool), source="fixture",
                    manmade=np.zeros((size, size), np.int64),
                    occupied=np.ones((size, size), np.int64))


@case
def t_a1_the_relief_band_is_read_on_the_core_and_a_steep_outer_ring_is_accepted():
    """The search half of A1, and the whole reason it exists.

        A city's relief band is 4-15 (`groundread.RELIEF_BANDS`). Applied to 512x512 of
        this world that is a number no candidate anywhere comes within a hundred of, so it
        ordered candidates by nothing; applied to the middle -- where the palace goes and
        where the ground is levelled -- it orders them by the thing it is about. And the
        outer ring, which is where a city's farmland is, is allowed to be a hillside.
        
    """
    fs = _find_site()
    s = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    size = int(s["needs"]["footprint"])
    core = fs.core_size(s)
    assert core == 96, f"the palace's own plateau is the core: {core}"
    needs = fs.search_needs(s)
    assert needs["core_max_relief"] == 8.0, needs
    assert needs["outer_max_relief"] == 200.0, needs   # the wall's, the most permissive
    f = _field(fs, size, core, core_relief=6, outer_relief=120)
    m = fs.measure(f, FIX_ORIGIN, FIX_ORIGIN, size, min(48, size), core=core)
    assert m["core"]["size"] == core and m["core"]["relief"] <= 8, m["core"]
    assert m["outer"]["relief"] >= 100, m["outer"]
    assert m["relief"] >= 100, m["relief"]
    e = fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)
    assert e["meets"], e
    assert e["core_relief"] == 0.0 and e["outer_relief"] == 0.0, e
    assert e["relief_preference_on"] == "core", e
    # The band read on the core against the band read on the footprint.
    lo, hi = needs["relief_band"]
    on_core = e["relief_preference"]
    on_footprint = round(abs(m["relief"] - (lo + hi) / 2) / (hi - lo), 6)
    assert on_core < on_footprint / 10, (on_core, on_footprint)
    # ...and a steeper outer ring than any part declared costs, and does not disqualify.
    tight = dict(needs, outer_max_relief=float(fs.OUTER_RELIEF))
    e2 = fs.excess(m, tight, plateau_relief=fs.PLATEAU_RELIEF)
    assert e2["meets"], "an outer ring over the allowance must rank down, never refuse"
    assert e2["outer_relief"] > 0 and e2["total"] > e["total"], (e2, e)
    # ...while the core going out of band *does* refuse, because that is the ground the
    # place actually stands on.
    steep = _field(fs, size, core, core_relief=40, outer_relief=120)
    m3 = fs.measure(steep, FIX_ORIGIN, FIX_ORIGIN, size, min(48, size),
                    core=core)
    assert not fs.excess(m3, needs, plateau_relief=fs.PLATEAU_RELIEF)["meets"]
    return (f"core {core}x{core} at relief {m['core']['relief']} inside a footprint at "
            f"relief {m['relief']}: accepted, band preference {on_core} on the core "
            f"against {on_footprint} on the footprint; an outer ring at "
            f"{m['outer']['relief']} over an allowance of {fs.OUTER_RELIEF} costs "
            f"{round(e2['total'] - e['total'], 4)} and refuses nothing; a core at "
            f"{m3['core']['relief']} is refused")


# ------------------------------------------------- A2. concentric, in the place read

def _ring(x0, z0, x1, z1, step=120):
    """A closed rectangular ring, with a vertex at least every `step` blocks.

        `rounds/type-needs.json` measures `wall` at 1x4 to 1x128, so a 512-block circuit
        drawn as four segments is a circuit no committed type will stand on. The planner has
        to break it and so does the fixture.
        
    """
    def run(a, b):
        n = abs(b - a)
        k = max(1, -(-n // step))
        return [a + round(n * i / k) * (1 if b > a else -1) for i in range(1, k + 1)]
    path = [[x0, z0]]
    path += [[x, z0] for x in run(x0, x1)]
    path += [[x1, z] for z in run(z0, z1)]
    path += [[x, z1] for x in run(x1, x0)]
    path += [[x0, z] for z in run(z1, z0)]
    return path


def _three_ring_place():
    """A place-level plan of three nested circuits, three gates and a palace.

        One arterial runs south to north up x=256 and crosses each ring at its own gate,
        which is the geometry the concentric clause is about: a road that crossed a ring
        anywhere else would be a hole in it.
        
    """
    rings = [("ring_wall_outer", 16, 16, 496, 496),
             ("ring_wall_middle", 96, 96, 416, 416),
             ("ring_wall_inner", 176, 176, 336, 336)]
    parts = [{"kind": "edge", "name": n, "defines": "ring_wall", "type": "wall",
              "seed": i + 1, "params": {}, "width": 1,
              "path": _ring(a, b, c, d), "notes": "a circuit"}
             for i, (n, a, b, c, d) in enumerate(rings)]
    parts += [{"kind": "point", "name": f"ring_gate_{k}", "defines": "ring_gate",
               "type": "gate_tower", "seed": 10 + i, "params": {},
               "at": [256, z], "facing": "north", "notes": "the way through"}
              for i, (k, z) in enumerate([("outer", 16), ("middle", 96),
                                          ("inner", 176)])]
    parts.append({"kind": "area", "name": "palace", "defines": "palace",
                  "type": "square", "seed": 20, "params": {},
                  "x0": 240, "z0": 240, "x1": 271, "z1": 271, "notes": "the compound"})
    return {
        "intent": "a fixture", "centre": "palace", "voice": "white_render_dark_frame",
        "circulation_material": "cobblestone", "parts": parts,
        # The road: one column of cells up the middle, which crosses every ring exactly
        # at its gate. Four wide, as `placeplan.ARTERIAL_WIDTH` says.
        "arterials": {"cells": [[x, z] for z in range(8, 260)
                                for x in range(254, 258)],
                      "joins": {}, "width": 4, "nodes": [], "edges": []},
        "districts": [
            {"name": "inner_ring_d", "x0": 190, "z0": 190, "x1": 322, "z1": 232,
             "structures": 2, "purpose": "gardens", "notes": ""},
            {"name": "middle_ring_d", "x0": 110, "z0": 110, "x1": 402, "z1": 160,
             "structures": 2, "purpose": "shop-houses", "notes": ""},
            {"name": "lower_ring_d", "x0": 30, "z0": 30, "x1": 482, "z1": 80,
             "structures": 2, "purpose": "workshops", "notes": ""},
        ]}


def _district(name, z0, n=2, x0=200):
    return {"notes": name, "quarters": [{"name": "q", "notes": "", "plots": [
        {"kind": "plot", "name": f"{name}_{i}", "type": "cottage", "seed": i + 1,
         "params": {}, "x0": x0 + 30 * i, "z0": z0, "x1": x0 + 30 * i + 14,
         "z1": z0 + 14, "notes": ""} for i in range(n)]}]}


def _three_ring_plan(spec):
    place = _three_ring_place()
    districts = {"inner_ring_d": _district("inner", 200, x0=200),
                 "middle_ring_d": _district("middle", 120, x0=120),
                 "lower_ring_d": _district("lower", 40, x0=40)}
    plan = placeplan.assemble(place, districts, spec)
    plan["arterials"] = place["arterials"]
    return plan


def _all_stood(plan):
    return {"waves": [{"wave": "all", "parts": [
        {"part": p["name"], "status": "built", "stood": True}
        for p in pipeline.plan_parts(plan)]}]}


@case
def t_a2_a_three_ring_fixture_holds_every_concentric_clause():
    """The half that has to pass, or the clause is `Context.ENCLOSED` again."""
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    plan = _three_ring_plan(spec)
    parts = pipeline.plan_parts(plan)
    decls = placeread._decls(parts)
    got = placeread.concentric_clauses(spec, plan, parts, decls,
                                       {p["name"]: True for p in parts})
    failed = [c["clause"] for c in got if not c["holds"]]
    assert len(got) == 5, [c["clause"] for c in got]
    assert not failed, [c for c in got if not c["holds"]]
    nest = [c for c in got if c["clause"] == "concentric/nested"][0]
    assert nest["areas"] == sorted(nest["areas"], reverse=True), nest
    return ("5 clauses hold on three nested circuits: "
            + ", ".join(c["clause"].split("/")[1] for c in got)
            + f"; areas {nest['areas']}")


@case
def t_a2_each_concentric_clause_fails_by_name_on_the_thing_it_is_about():
    """Five clauses, five ways to break them, each naming the part it is about."""
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    base = _three_ring_place()
    said = {}

    def read(place, districts=None):
        d = districts or {"inner_ring_d": _district("inner", 200, x0=200),
                          "middle_ring_d": _district("middle", 120, x0=120),
                          "lower_ring_d": _district("lower", 40, x0=40)}
        plan = placeplan.assemble(place, d, spec)
        plan["arterials"] = place["arterials"]
        parts = pipeline.plan_parts(plan)
        return placeread.concentric_clauses(
            spec, plan, parts, placeread._decls(parts),
            {p["name"]: True for p in parts})

    def broken(clause, place=None, districts=None):
        got = read(place or base, districts)
        row = [c for c in got if c["clause"] == clause]
        assert row and not row[0]["holds"], (clause, got)
        said[clause] = row[0]["says"]

    # 1. two rings side by side are not nested.
    p = json.loads(json.dumps(base))
    p["parts"][2]["path"] = _ring(360, 360, 490, 490)
    broken("concentric/nested", p)
    # 2. a ring with no gate on it.
    p = json.loads(json.dumps(base))
    p["parts"] = [x for x in p["parts"] if x.get("name") != "ring_gate_middle"]
    broken("concentric/gates", p)
    # 3. an arterial through a ring away from its gate.
    p = json.loads(json.dumps(base))
    p["arterials"]["cells"] += [[100, 96], [101, 96]]
    broken("concentric/crossings", p)
    # 4. a palace outside the innermost ring.
    p = json.loads(json.dumps(base))
    for x in p["parts"]:
        if x.get("name") == "palace":
            x.update(x0=120, z0=120, x1=151, z1=151)
    broken("concentric/centre", p)
    # 5. a house outside the city wall -- open thread 17, and nothing had ever asked.
    broken("concentric/districts", None,
           {"inner_ring_d": _district("inner", 200, x0=200),
            "middle_ring_d": _district("middle", 120, x0=120),
            "lower_ring_d": _district("lower", 4, x0=4)})
    assert "outside" in said["concentric/nested"], said["concentric/nested"]
    assert "no gate" in said["concentric/gates"], said["concentric/gates"]
    assert "ring_wall_middle" in said["concentric/crossings"], said
    assert "palace" in said["concentric/centre"], said["concentric/centre"]
    assert "outside it" in said["concentric/districts"], said["concentric/districts"]
    return "; ".join(f"{k.split('/')[1]}: {v[:64]}" for k, v in said.items())


@case
def t_a2_a_plan_of_rings_is_refused_before_the_ground_is_touched():
    """A plan that cannot be the place asked for is refused **before anything is sited**,
        not discovered by the read at the end -- and three of the read's five clauses are
        about geometry the plan already has, so they can be asked now. The two that are not
        (where the road crosses, and what stood) are the read's.
        
    """
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    _t, decls = placeplan.types_card(None, spec.get("form"))
    base = _three_ring_place()
    assert not placeplan.concentric_failures(base, spec, decls), \
        placeplan.concentric_failures(base, spec, decls)
    said = {}
    for label, mut in (
            ("nested", lambda p: p["parts"][2].update(path=_ring(360, 360, 490, 490))),
            ("gates", lambda p: p.__setitem__(
                "parts", [x for x in p["parts"]
                          if x.get("name") != "ring_gate_middle"])),
            ("centre", lambda p: [x.update(x0=120, z0=120, x1=151, z1=151)
                                  for x in p["parts"] if x.get("name") == "palace"]),
            ("count", lambda p: p.__setitem__(
                "parts", [x for x in p["parts"]
                          if x.get("name") != "ring_wall_inner"]))):
        p = json.loads(json.dumps(base))
        mut(p)
        got = placeplan.concentric_failures(p, spec, decls)
        assert got and all(f["check"] == "concentric" for f in got), (label, got)
        said[label] = got[0]["why"]
    assert "not inside" in said["nested"], said["nested"]
    assert "no gate stands" in said["gates"], said["gates"]
    assert "innermost ring" in said["centre"], said["centre"]
    assert "and the plan draws 2" in said["count"], said["count"]
    # ...and the brief states every rule this refuses on, because a rule that is checked
    # and not stated is a hand-back spent for nothing -- and this level is handed back
    # at most once in total.
    note = placeplan._concentric_rules(spec)
    for word in ("concentric", "inside", "gate", "between", "closed loop"):
        assert word in note, (word, note[:200])
    dens = placeplan._density_note(spec)
    assert "dense" in dens and "sparse" in dens and "columns of district" in dens, dens
    return ("; ".join(f"{k}: {v[:52]}" for k, v in said.items())
            + f"; the brief states all four rules in {len(note.split())} words and the "
              f"densities in {len(dens.split())}")


@case
def t_a2_round_17s_plan_fails_the_nesting_clause_of_a_three_ring_spec():
    """The spec's own case."""
    p = offline.fixture_path("town", "plan.json")
    if not os.path.exists(p):
        raise Skip("no recorded town plan on this checkout")
    plan = json.load(open(p))
    pr = offline.fixture_path("town", "parts.json")
    parts_rec = json.load(open(pr)) if os.path.exists(pr) else {}
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    got = placeread.read(spec, plan, parts_rec, voice="blackstone_and_ash")
    assert not got["holds"], got
    assert "concentric/nested" in got["failed"], got["failed"]
    nest = [c for c in got["clauses"] if c["clause"] == "concentric/nested"][0]
    assert nest["wanted"] == 3 and len(nest["got"]) < 3, nest
    own = spec_mod.read_spec(
        json.load(open(offline.fixture_path("town", "place.json"))),
        "Build a walled town of about sixty houses with a market square and a keep.")
    site = offline.fixture_path("town", "site.json")
    mine = placeread.read(own, plan, parts_rec, voice="blackstone_and_ash",
                          site=json.load(open(site)) if os.path.exists(site) else None)
    assert mine["holds"], mine["failed"]
    assert not any(c["clause"].startswith("concentric/") for c in mine["clauses"])
    return (f"the recorded town's plan against a three-ring spec: {got['failed']} -- "
            f"{nest['says'][:90]}; against its own spec it still holds all "
            f"{len(mine['clauses'])} clauses with no concentric clause raised")


# ------------------------------------------------------- A2. the role

@case
def t_r2a2_a_district_is_built_out_of_what_it_is_for():
    """A farmhouse is not the default house of a city street."""
    # It is the only plan this project has that is big enough to have the defect in it.
    d = os.path.join(ROOT, "out", "city_a")
    if not os.path.isdir(d):
        d = os.path.join(ROOT, "out", "city_b")
    if not os.path.exists(os.path.join(d, "plan.place.json")):
        raise Skip("no out/city_a/plan.place.json")
    spec = spec_mod.read_spec(json.load(open(os.path.join(d, "place.json"))),
                              CITY_SENTENCE)
    roles = {p["name"]: p["role"] for p in spec["defining_parts"]}
    assert roles["agrarian_ring"] == "rural", roles
    assert roles["lower_ring"] == roles["middle_ring"] == "urban", roles
    assert roles["concentric_walls"] == "defensive", roles

    place = json.load(open(os.path.join(d, "plan.place.json")))
    districts = {}
    for x in place.get("districts") or []:
        p = os.path.join(d, f"plan.district.{x['name']}.json")
        if not os.path.exists(p):
            raise Skip(f"no {p}")
        districts[x["name"]] = json.load(open(p))
    plan = placeplan.assemble(place, districts, spec)
    parts = pipeline.plan_parts(plan)
    decls = {p.get("type"): pipeline.load_type(
        os.path.join(ROOT, "types", f"{p.get('type')}.py")) for p in parts}

    def role_fails(ps):
        return [f for f in pipeline.plan_failures(ps, decls, form=spec.get("form"))
                if f["check"] == "role"]

    bad = role_fails(parts)
    named = sorted({f["type"] for f in bad})
    assert named == ["farmstead", "keep", "minka"], named
    minkas = [f for f in bad if f["type"] == "minka"]
    assert len(minkas) >= 100, len(minkas)
    assert "rural" in minkas[0]["why"] and "urban" in minkas[0]["why"], minkas[0]

    # ...and the same plan with its urban houses typed as urban houses has none.
    urban = {"minka": "court_small", "farmstead": "court_small", "keep": "hall"}
    fixed = [dict(p, type=urban.get(p["type"], p["type"]))
             if p.get("role") in ("urban",) else p for p in parts]
    assert not role_fails(fixed), role_fails(fixed)[:3]
    # ...and a rural district still takes a farmhouse, so this is a filter and not a ban
    rural = [p for p in parts if p.get("role") == "rural"]
    assert any(p.get("type") in ("minka", "farmstead") for p in rural), "no rural leaf"
    assert not role_fails(rural), role_fails(rural)[:3]
    # ...and the brief a district planner is given lists only what it may build
    card, _ = placeplan.types_card(None, "east_asian", "urban")
    assert "minka" not in card and "shop_house" in card and "temple" in card, card
    return (f"the recorded city's plan: {len(bad)} of {len(parts)} leaves refused by role "
            f"({len(minkas)} minka, {len(bad) - len(minkas)} other); re-typed to urban "
            f"houses, 0; the rural ring keeps its farmhouses; the urban card lists "
            f"{card.count('| `')} types and no minka")


# ------------------------------------------- voice contract, A1 and A2: the voice
# reaches a block

def _voiced_fixture(voice: str, tmp: str):
    """A round carrying a sentence and a plan that chose `voice`, with one plot leaf on
    the slope fixture's own ground, in a scratch state directory of its own."""
    import shutil
    from ethoslm import slopefixture
    src, _be = slopefixture.make("slope_9_11", ROOT)
    for f in ("world.npz", "plots.json", "network.json"):
        shutil.copyfile(src.rel(f), os.path.join(tmp, f))
    plot = json.load(open(src.rel("plots.json")))[0]
    # The leaf carries the fixture plot's own label, because that is the name the
    # fixture's network reserved a doorstep under and `site()` looks the door up by it.
    plan = {"intent": "a fixture", "centre": "short_axis_slope", "voice": voice,
            "circulation_material": "cobblestone",
            "parts": [{"name": "short_axis_slope", "kind": "plot", "type": "_reference",
                       "x0": plot["x0"], "z0": plot["z0"], "x1": plot["x1"],
                       "z1": plot["z1"], "seed": 3, "params": {"storeys": 1},
                       "in": ["fixture"]}]}
    json.dump(plan, open(os.path.join(tmp, "plan.json"), "w"), indent=1)
    json.dump(src.site, open(os.path.join(tmp, "site.json"), "w"))
    rnd = pipeline.Round(name="voice-fixture", state_dir=tmp, sentence="Build a house.",
                         site=dict(src.site))
    return rnd, plan


@case
def t_vc_a1_the_build_reads_the_voice_the_place_chose():
    """A1: `stage_parts` composes every part in `rnd.voice_name()`, not `rnd.voice`."""
    from ethoslm import placeread
    tmp = tempfile.mkdtemp(prefix="ethoslm-vc-a1-")
    try:
        rnd, plan = _voiced_fixture("ochre_stone_green_tile", tmp)
        assert rnd.voice == "" and rnd.voice_name() == "ochre_stone_green_tile"
        be = pipeline.OfflineBackend(rnd, dry_run=True)
        got = pipeline.stage_parts(rnd, be, {})
        rec = got["waves"][0]["parts"][0]
        assert rec["status"] == "built" and rec["stood"], rec
        src = open(rnd.rel("parts", "short_axis_slope.py")).read()
        assert "mat=None" not in src and "'wall': 'sandstone'" in src \
            and "'ends': 'irimoya'" in src, src[-400:]
        be.save(rnd.rel("world_built.npz"))
        built = offline.load_volume(rnd.rel("world_built.npz"))
        base = offline.load_volume(rnd.rel("world.npz"))
        parts = pipeline.plan_parts(plan)
        stood = {"short_axis_slope": True}
        ok = placeread.built_palette("ochre_stone_green_tile", parts, stood, built, base)
        assert ok["ok"] and ok["read"] == 1, ok["says"]
        top = dict(_top(built, base, parts[0]))
        assert top.get("sandstone") and top.get("oxidized_copper") \
            and top.get("granite") and not top.get("cobblestone"), top
        # ...and read against no voice at all, which is what a plan that chose none
        # hands the clause, it fails by name rather than passing on an empty field
        no = placeread.built_palette(None, parts, stood, built, base)
        assert not no["ok"] and "no voice" in no["says"], no["says"]
        return (f"one part composed with the plan's voice on the record -- "
                f"{', '.join(f'{k} x{v}' for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:3])} "
                f"-- at {ok['share_min']:.1%} in the voice; a plan that chose none is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _top(built, base, part) -> list:
    from ethoslm import placeread, prims
    from ethoslm.lint import plot_rects
    counts: dict = {}
    for (x0, z0, x1, z1) in plot_rects(pipeline.part_registry_row(part)):
        for k, v in placeread._built_census(built, base, x0, z0, x1, z1).items():
            f = prims.family(k)
            if f:
                counts[f] = counts.get(f, 0) + v
    return sorted(counts.items(), key=lambda kv: -kv[1])


@case
def t_vc_a2_the_palette_clause_reads_blocks_and_round_1_fails_it_by_name():
    """A2."""
    from ethoslm import placeread
    d = os.path.join(ROOT, "out", "city_a")
    if not os.path.exists(os.path.join(d, "world_built.npz")):
        raise Skip("no out/city_a/world_built.npz")
    rnd = pipeline.Round(name="city_a", sentence="Build Ringed City.")
    assert rnd.voice_name() == "ochre_stone_green_tile", rnd.voice_name()
    spec, plan = rnd.place_spec(), rnd.plan()
    parts_rec = json.load(open(rnd.rel("parts.json")))
    built = offline.load_volume(rnd.rel("world_built.npz"))
    base = offline.load_volume(rnd.rel("world.npz"))
    # the declaration alone still passes, which is the whole defect...
    decl = placeread.read(spec, plan, parts_rec, voice=rnd.voice_name(),
                          site=pipeline.settlement_site(rnd))
    assert [c["clause"] for c in decl["clauses"] if c["clause"].startswith("palette")] \
        == ["palette"]
    assert next(c for c in decl["clauses"] if c["clause"] == "palette")["holds"]
    # ...and the blocks do not
    got = placeread.read(spec, plan, parts_rec, voice=rnd.voice_name(),
                         site=pipeline.settlement_site(rnd), built=built, base=base)
    c = next(c for c in got["clauses"] if c["clause"] == "palette/built")
    assert not c["holds"] and not got["holds"] and "palette/built" in got["failed"]
    assert c["failed_count"] == c["read"] >= 260, (c["failed_count"], c["read"])
    assert c["share_median"] < 0.5 and c["threshold"] == placeread.BUILT_SHARE == 0.9
    worst = c["failed"][0]
    assert worst["top"][0][0] == "cobblestone", worst
    return (f"the recorded city: the declaration passes and the blocks fail -- "
            f"{c['failed_count']} of {c['read']} parts not in {c['voice']}, median "
            f"{c['share_median']:.1%} in the voice against {c['threshold']:.0%}, "
            f"{worst['part']} is {worst['top'][0][0]} x{worst['top'][0][1]}")


# ---------------------------------------------------------- A4. the render budget

@case
def t_a4_the_render_budget_is_a_config_field_and_bounded_at_four_hundred():
    """A card per building at 400 structures is six hours of Chunky. So it is not that.

        The budget is read off the plan and the config, here, so that what a live run will
        render is a number this case can put a bound on before the run rather than a
        surprise in the middle of one.
        
    """
    from ethoslm.pipeline import stages_media
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    plan = _three_ring_plan(spec)
    budget = stages_media.render_budget(plan, spec, {})
    frames = stages_media.budget_frames(plan, spec, budget)
    assert budget["sample"] == stages_media.RENDER_BUDGET["sample"] == 24
    kinds = {f["why"] for f in frames}
    assert {"place", "defining", "sample", "flythrough"} <= kinds, kinds
    # Every defining part is photographed; the ordinary buildings are sampled, seeded,
    # and the same seed gives the same sample.
    plots = [p for p in pipeline.plan_parts(plan) if p.get("kind", "plot") == "plot"]
    sample = [f for f in frames if f["why"] == "sample"]
    assert len(sample) == min(len(plots), 24), (len(sample), len(plots))
    again = stages_media.budget_frames(plan, spec, budget)
    assert [f["name"] for f in again] == [f["name"] for f in frames]
    # ...and a place of four hundred structures is bounded by the sample, not by itself.
    big = json.loads(json.dumps(plan))
    q = big["parts"][0]["children"][-1]
    q["children"] = [dict(q["children"][0], name=f"h{i}", x0=40 + i, x1=45 + i)
                     for i in range(400)]
    n = len(stages_media.budget_frames(big, spec, budget))
    assert n <= stages_media.MAX_FRAMES, n
    assert n < 400, n
    return (f"{len(frames)} frames on the fixture ({sorted(kinds)}); a plan of 400 "
            f"structures renders {n} against a cap of {stages_media.MAX_FRAMES}, and "
            f"the sample of {budget['sample']} reproduces from its seed")


# ------------------------------------------------ the setting: ground that reads


def _settled(fs, size, core, core_relief, outer_relief, surface, water_pct=0.0,
             origin=FIX_ORIGIN):
    """A synthetic candidate with a surface: `surface` is the block every land column's
    ground is made of, and `water_pct` of the footprint is a lake in one corner."""
    f = _field(fs, size, core, core_relief, outer_relief)
    n = int(round(size * np.sqrt(water_pct / 100.0))) if water_pct else 0
    wet = np.zeros((size, size), bool)
    if n:
        wet[:n, :n] = True
    palette = [surface, "water"]
    codes = np.where(wet, 1, 0).astype(np.int32)
    return fs.Field(origin, origin, f.h, wet, f.canopy, gravity=f.gravity,
                    source="fixture", manmade=f.manmade, occupied=f.occupied,
                    surface=codes, surface_palette=palette)


@case
def t_setting_a_place_that_wants_green_ground_is_not_put_on_badlands():
    fs = _find_site()
    size, core = 512, 96
    base = json.loads(json.dumps(CITY_SPEC))
    plain = spec_mod.read_spec(json.loads(json.dumps(base)), CITY_SENTENCE)
    green = spec_mod.read_spec(dict(base, setting={"surface": "green", "water": "some",
                                                    "notes": "plains and a lake"}),
                               CITY_SENTENCE)
    desert = spec_mod.read_spec(dict(base, setting={"surface": "badlands",
                                                     "water": "none"}), CITY_SENTENCE)
    assert green["setting"]["surface"] == "green" and green["setting"]["water"] == "some"
    # The band is 4-15 and the preference is distance from its middle, so a core at 8 --
    # the most the palace's own need admits -- is the one the search as it stood prefers
    # over a core at 4.
    bad = _settled(fs, size, core, core_relief=8, outer_relief=20, surface="red_sand")
    good = _settled(fs, size, core, core_relief=4, outer_relief=20,
                    surface="grass_block", water_pct=6.0, origin=FIX_ORIGIN + 4096)

    def rank(spec):
        needs = fs.search_needs(spec)
        rows = []
        for f in (bad, good):
            m = fs.measure(f, f.x0, f.z0, size, 48, core=core)
            rows.append({"measures": m,
                         "excess": fs.excess(m, needs, plateau_relief=fs.PLATEAU_RELIEF)})
        rows.sort(key=fs.rank_key)
        return rows

    # As the search stood: both meet, the flatter badlands square wins.
    before = rank(plain)
    assert all(r["excess"]["meets"] for r in before), [r["excess"] for r in before]
    assert before[0]["measures"]["surface"]["classes"]["badlands"] == 100.0, before[0]
    assert before[0]["excess"]["setting_preference"] == 0.0

    # Wanting green ground and a lake: the badlands square is refused by name.
    after = rank(green)
    top, second = after[0]["measures"], after[1]["excess"]
    assert top["surface"]["classes"]["green"] >= 100.0 * spec_mod.SURFACE_SHARE, top
    assert top["water_pct"] >= spec_mod.WATER_SOME_PCT, top["water_pct"]
    assert after[0]["excess"]["meets"], after[0]["excess"]
    assert not second["meets"] and second["surface"] > 0, second
    assert any("surface green is 0.0%" in w for w in second["setting_failures"]), second
    assert any("water 0.0%" in w for w in second["setting_failures"]), second
    census = top["surface"]
    assert census["read"] and census["top"][0][0] == "grass_block", census
    assert census["classes"]["badlands"] == 0.0 and census["land_pct"] < 100.0, census

    # ...and a desert fort wants the badlands: the green square, with its lake, fails.
    fort = rank(desert)
    assert fort[0]["measures"]["surface"]["classes"]["badlands"] == 100.0, fort[0]
    assert fort[0]["excess"]["meets"] and not fort[1]["excess"]["meets"], \
        [r["excess"] for r in fort]
    assert fort[1]["excess"]["water"] > 0, fort[1]["excess"]

    # The words the spec may use are the census's, and no other.
    for word in ("plains", "grass", "desert"):
        try:
            spec_mod.read_spec(dict(base, setting={"surface": word}), CITY_SENTENCE)
            raise AssertionError(f"a setting of {word!r} was accepted")
        except spec_mod.SpecError as e:
            assert "surface" in str(e), str(e)
    return (f"no setting: badlands first (core relief 8 against 4); green wanted: "
            f"green first at {census['classes']['green']}% green, {top['water_pct']}% "
            f"water, badlands refused for {second['setting_failures'][0]!r}; badlands "
            f"wanted: badlands first, green refused")


# ------------------------------------ A5. search radius and terraform for a city

@case
def t_a5_the_grid_reaches_8192_and_a_city_terraforms_its_core_and_nothing_else():
    fs = _find_site()
    assert fs.RADII[-1] == 8192, fs.RADII
    assert 4096 in fs.RADII, fs.RADII
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    # The part the ground may be moved for is the core and it is named in the answer.
    p = fs.innermost(spec)
    assert p["name"] == "palace" == spec_mod.core(spec)["name"], p
    # ...and a plateau is bounded by what `Builder.plateau()` will actually cut, so a
    # 96-block compound asks for what the library can give and says so.
    assert fs.core_size(spec) == 96
    assert fs._plateau_size(spec) <= fs.Builder.PLATEAU_MAX
    ans = fs._answer(spec, "terraformed", 512, fs._plateau_size(spec), [], [], None,
                     fs.PLATEAU_RELIEF, terraform=True)
    assert ans["terraform"]["part"] == "palace", ans["terraform"]
    assert ans["terraform"]["scope"] == "core", ans["terraform"]
    return (f"radii {list(fs.RADII)}; the one part a city may level is "
            f"{ans['terraform']['part']} ({ans['terraform']['family']}), "
            f"{ans['terraform']['plateau']} blocks, scope {ans['terraform']['scope']}")


@case
def t_a5_a_city_search_over_the_cache_answers_with_a_site_or_with_a_named_reason():
    """The spec's own case: a fixed spec, twice, with no server, and an answer either way.

        A city's footprint is 512 and this world's cached squares are 144, so the honest
        answer here is very likely "nothing is readable" -- and *that is an answer*. What
        this case refuses is the third thing: a search that returns a site it did not read
        the ground for.
        
    """
    fs = _find_site()
    spec = spec_mod.read_spec(json.loads(json.dumps(CITY_SPEC)), CITY_SENTENCE)
    assert int(spec["needs"]["footprint"]) == spec_mod.footprint_ceiling("city") == 768, \
        spec["needs"]
    one = fs.search_fresh(spec, editor=None, radii=(1024,), log=lambda *a: None)
    two = fs.search_fresh(spec, editor=None, radii=(1024,), log=lambda *a: None)
    # The **answer** is the same twice; the cost record.
    assert json.dumps(fs.answer_of(one), sort_keys=True) == \
        json.dumps(fs.answer_of(two), sort_keys=True)
    assert one["footprint"] == 768
    assert one["max_new"] == 0, "a search that generates ground by default is a search "
    if one["chosen"]:
        assert one["candidates"] >= 1
        why = (f"chose ({one['chosen']['origin'][0]},{one['chosen']['origin'][1]}) "
               f"score {one['chosen']['score']} from {one['candidates']} squares")
    elif one["candidates"]:
        # A search that scored squares and chose none is a search nothing met, and the
        # record names the best square and what it failed.
        assert one["attempt"] == "nothing met the needs", one["attempt"]
        assert one["best_failed"] and one["best_failed"]["failures"], one["best_failed"]
        why = (f"no site: nothing of {one['candidates']} squares met the needs; the "
               f"best, {one['best_failed']['origin']}, fails "
               f"{one['best_failed']['failures'][0]!r}")
    else:
        assert one["squares"]["unread"] >= 1, one["squares"]
        why = (f"no site: {one['squares']['unread']} of "
               f"{one['squares']['unread'] + one['candidates']} squares are ground "
               f"nothing has read at 768 and no session is open")
    return f"768x768, twice, no server, identical: {why}"


@case
def t_dp2b_a_gate_is_sized_by_its_wall_and_a_wall_knows_its_gates():
    """Demo-polish, 2b. Siting hands a point on an edge the edge's type and height,
    sizes its pad from that height by `Builder.point_pad` inside the type's band, and
    hands the edge the points standing on it. Neither the planner nor the type is
    told a height by hand."""
    from ethoslm.buildlib import Builder
    # the registered rule, at the heights the demo's three rings and its precinct wall
    # have, and at a town wall's
    assert Builder.POINT_PAD_PER_HEIGHT == 3
    band = pipeline.load_type(os.path.join(ROOT, "types", "ring_gate.py"))["needs"]["footprint"]
    got = {h: Builder.point_pad(h, band) for h in (48, 36, 20, 16, 5, None)}
    assert got == {48: 15, 36: 11, 20: 5, 16: 5, 5: 5, None: 5}, got
    assert Builder.point_pad(96, band) == 15, "the pad left the type's band"
    assert Builder.point_pad(48, (3, 3, 16, 16)) == 15 and Builder.point_pad(48, None) == 15
    # on a plan: three rings with heights, a gate on each, one gate off any wall
    place = _three_ring_place()
    for p, h in zip([q for q in place["parts"] if q["kind"] == "edge"], (48, 36, 20)):
        p["params"] = {"height": h}
    place["parts"].append({"kind": "point", "name": "well_head", "type": "gate_tower",
                           "seed": 99, "params": {}, "at": [300, 300], "facing": "north"})
    parts = pipeline.plan_parts({"parts": place["parts"]})
    done = pipeline.annotate_gates(parts)
    assert set(done) == {"ring_gate_outer", "ring_gate_middle", "ring_gate_inner"}, done
    by = {p["name"]: p for p in parts}
    assert by["ring_gate_outer"]["edge"] == {"name": "ring_wall_outer", "type": "wall",
                                             "height": 48, "width": 1}
    assert by["ring_gate_outer"]["size"] == 15 and by["ring_gate_middle"]["size"] == 11 \
        and by["ring_gate_inner"]["size"] == 5, [by[n].get("size") for n in done]
    assert "edge" not in by["well_head"] and "size" not in by["well_head"]
    assert [g["name"] for g in by["ring_wall_outer"]["gates"]] == ["ring_gate_outer"]
    assert by["ring_wall_outer"]["gates"][0]["size"] == 15
    assert "gates" not in by["ring_wall_middle"] or \
        [g["name"] for g in by["ring_wall_middle"]["gates"]] == ["ring_gate_middle"]
    # ...and the pad reaches siting and the registry: `pad_extent` and `part_rect` read
    # the size the rule wrote
    assert Builder.pad_extent(by["ring_gate_outer"]) == (15, 15)
    x0, z0, x1, z1 = pipeline.part_rect(by["ring_gate_outer"])
    assert (x1 - x0 + 1, z1 - z0 + 1) == (15, 15) and x0 <= 256 <= x1, (x0, z0, x1, z1)
    # a part with a size of its own keeps it
    by["ring_gate_outer"]["size"] = 7
    del by["ring_gate_outer"]["edge"]
    pipeline.annotate_gates(parts)
    assert by["ring_gate_outer"]["size"] == 7
    return ("48 -> 15, 36 -> 11, 20 -> 5 inside ring_gate's band of 5..16; three gates "
            "annotated with their walls, the well head with none, each wall with its gate")



def main():
    bad = skipped = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:70s} {fn()}")
        except Skip as e:
            skipped += 1
            print(f"skip {name:70s} {e}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:70s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:70s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad - skipped}/{len(CASES)} city cases pass"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
