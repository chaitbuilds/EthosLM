"""The ground is the library's, and the fixtures span it.

    $PY scripts/test_siting.py

Each is the smallest fixture that would fail against the code as it stood when A0 fired.

**Siting is the library's.** `site(part)` sounds the bed, lays a deck on piles where the
ground is water, a platform where it has relief and a plinth where it has neither, and
lays the way in from the lane -- and the floor it hands back is walk-reachable from the
lane **before any type has built anything**. Three fixtures, one per case, off three
rounds' own caches. **The type contract forbids ground work.** A type that names a
terrain call is refused at preflight by E013; one that places a stair directly is
refused as it runs, with the cell named; and `get_height` still reads.

the pre-build caches, the networks and the plot registries. They are the fixtures the
spec's acceptance is written against, so a missing one fails rather than skips.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import lint, observe, offline, pipeline, stages  # noqa: E402
from ethoslm.buildlib import BuildError, Builder  # noqa: E402
from ethoslm.frontage import Frontage  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


#: One palette, so what changes between the three cases is the ground and nothing else.
MAT = pipeline.voice_palette("white_render_dark_frame")


def _voice_block(role, kind="full"):
    """What a reference type places for one role of the fixture voice.

        A type names no material now, so a test that expects `stone_bricks` is a test that
        would pass for the wrong reason: it has to ask the same voice the type was handed.
        
    """
    from ethoslm.prims import shape
    return shape(MAT[role], kind)


def _cache(name: str):
    p = offline.world_cache(name)
    if not os.path.exists(p):
        raise AssertionError(
            f"COULD NOT RUN: {os.path.relpath(p, ROOT)} is missing -- this is a case "
            f"the spec's acceptance is written against, so it fails rather than skips")
    return pipeline._fixture_round(name)


def sited(round_name: str, label: str):
    """(volume, builder, part) for one plot of one round's pre-build cache, sited.

        Nothing is written anywhere: the backend is the cached world, the treads are
        resolved so the walk model can see them, and the standing settlement is untouched.
        
    """
    rnd, _be = _cache(round_name)
    vol = rnd.volume()
    plot = {p["label"]: p for p in
            json.load(open(rnd.rel("plots.json")))}[label]
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    net = rnd.network()
    b.frontage = Frontage(vol, net) if net else None
    b.registry = stages._Registry(rnd.state)
    part = b.site(dict(plot, kind="plot"), mat=MAT)
    b.resolve_steps()
    return rnd, vol, b, part


def walkable_from_the_lane(rnd, vol, b, part) -> tuple:
    """(reached, columns) of the sited floor, on foot from the lane, no jumping.

        The same flood every from-outdoors measurement in this project uses, seeded from
        the declared circulation network by `lint.lane_stances`, over the cached world with
        everything `site()` decided overlaid.
        
    """
    v = observe.Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(b._pending)
    nav = observe.Nav(v)
    seeds = lint.lane_stances(nav, rnd.network())
    assert seeds, "no lane to walk from: this fixture cannot answer the question"
    reach = set(nav.flood(seeds, max_jumps=0))
    x0, z0, x1, z1 = part["footprint"]
    got = 0
    cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
    for (x, z) in cols:
        s = nav.stance_near(x, z, part["floor_y"] + 1, tol=1)
        got += int(s is not None and (x, z, s) in reach)
    return got, len(cols)


# siting is the library's The three cases the measurement found, each the extreme of it
# in the three rounds a type is checked on.
GROUND_CASES = (
    ("site_e", "fold_barn", "plinth", "dry and flat: relief 2 over 285 columns"),
    ("site_d", "south_gate", "platform", "dry with relief 21, the steepest plot in "
                                         "rounds 7, 9 and 10"),
    ("site_d", "tarn_landing", "deck", "99.4% standing water"),
)


@case
def t_a0a_the_three_kinds_of_ground_are_sited_and_the_floor_is_reachable():
    got = {}
    for (round_name, label, expect, why) in GROUND_CASES:
        rnd, vol, b, part = sited(round_name, label)
        assert part["ground"] == expect, \
            f"{label} ({why}) was sited as {part['ground']!r}, not {expect!r}"
        assert b._pending, f"{label}: siting placed nothing at all"
        walk, cols = walkable_from_the_lane(rnd, vol, b, part)
        assert walk == cols, \
            (f"{label}: {walk} of {cols} columns of the sited floor are walk-reachable "
             f"from the lane -- {part['sited']['reason']}")
        x0, z0, x1, z1 = part["footprint"]
        got[label] = (f"{label} {part['ground']} {x1 - x0 + 1}x{z1 - z0 + 1} floor "
                      f"y={part['floor_y']} {walk}/{cols} on foot")
    return "; ".join(got.values())


@case
def t_a0a_the_part_carries_the_floor_the_footprint_and_the_door():
    """What `build()` is handed. A type that has these three does not need the ground."""
    rnd, vol, b, part = sited("site_d", "south_gate")
    plot = {p["label"]: p for p in
            json.load(open(rnd.rel("plots.json")))}["south_gate"]
    x0, z0, x1, z1 = part["footprint"]
    assert (part["x0"], part["z0"], part["x1"], part["z1"]) == (x0, z0, x1, z1), part
    assert min(plot["x0"], plot["x1"]) <= x0 and x1 <= max(plot["x0"], plot["x1"]), part
    assert min(plot["z0"], plot["z1"]) <= z0 and z1 <= max(plot["z0"], plot["z1"]), part
    assert isinstance(part["floor_y"], int), part
    assert part["facing"] in ("north", "south", "east", "west"), part
    dx, dz = part["door"]
    assert x0 <= dx <= x1 and z0 <= dz <= z1, (part["door"], part["footprint"])
    assert dx in (x0, x1) or dz in (z0, z1), "the door is not in a wall of the pad"

    # ...and the floor is a floor: every column of the pad has its top face at floor_y
    tops = {b.get_block(x, part["floor_y"], z).split("[")[0]
            for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)}
    assert "air" not in tops, f"the pad has holes in it: {sorted(tops)}"
    return (f"floor y={part['floor_y']}, footprint {x1 - x0 + 1}x{z1 - z0 + 1} inside "
            f"the plot, door at ({dx},{dz}) facing {part['facing']}, "
            f"{len(tops)} material(s) in the pad")


@case
def t_a0a_a_pad_is_laid_inside_its_plot_and_never_on_the_lane():
    rnd, vol, b, part = sited("site_b", "court_wellhouse")
    plot = {p["label"]: p for p in
            json.load(open(rnd.rel("plots.json")))}["court_wellhouse"]
    px0, pz0 = min(plot["x0"], plot["x1"]), min(plot["z0"], plot["z1"])
    px1, pz1 = max(plot["x0"], plot["x1"]), max(plot["z0"], plot["z1"])
    net = rnd.network()
    lanes = {(x, z) for (x, z) in net.cells}

    # Solid work only.
    laid = {(x, z) for (x, _y, z), blk in b._pending.items()
            if blk.split("[")[0] not in ("air", "cave_air", "void_air")}
    on_lane = sorted(laid & lanes)
    assert not on_lane, f"the pad was laid on {len(on_lane)} lane cells: {on_lane[:4]}"
    # ...and the pad itself, which is everything laid that is not the way in. An
    # `approach()` crosses public ground by definition -- that is what it is for -- and
    # it writes down every column it laid, so the two are separable. past the plot by
    # however far a tree rooted outside it leaned in. Leaving the dirt that tree stood
    # on bare is the defect A6 exists for, so putting grass back on it is ground work
    # rather than pad. By record, `Builder.dressed`, for the reason the fitting register
    # is by record.
    path = {tuple(c) for p in b.paths for c in p["cells"]}
    outside = sorted(q for q in laid - path - b.dressed
                     if not (px0 <= q[0] <= px1 and pz0 <= q[1] <= pz1))
    assert not outside, \
        f"{len(outside)} columns of pad lie outside the plot: {outside[:4]}"

    # ...and the doorstep the pass reserved is still standable, at the level it was
    t = net.threshold("court_wellhouse")
    v = observe.Volume(vol.x0, vol.y0, vol.z0, vol.codes.copy(), list(vol.palette))
    v.overlay(b._pending)
    nav = observe.Nav(v)
    assert nav.stance_near(t.x, t.z, t.y + 1, tol=1) is not None, \
        f"the reserved threshold at ({t.x},{t.y},{t.z}) has been built over"
    assert len(lint.lane_stances(nav, net)) == len(net.cells), \
        (f"{len(net.cells) - len(lint.lane_stances(nav, net))} lane cells cannot be "
         f"stood on after siting")
    walk, cols = walkable_from_the_lane(rnd, vol, b, part)
    assert walk == cols, f"{walk} of {cols} columns of pad reachable"
    return (f"pad {part['footprint']} inside the plot, 0 of {len(lanes)} lane cells "
            f"touched, the reserved doorstep still stands, {walk}/{cols} on foot")


# the type contract forbids ground work

GROUNDWORK_TYPE = '''"""A type that levels its own ground. What A0 measured."""
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2)}


def build(b, part, seed, storeys=1):
    y = b.get_height(part["x0"], part["z0"])
    b.clear_trees(part["x0"], part["z0"], part["x1"], part["z1"])
    b.foundation_to_grade(part["x0"], part["z0"], part["x1"], part["z1"], y,
                          "cobblestone")
    b.approach(part["label"])
'''

TREAD_TYPE = '''"""A type that lays its own treads. `minka/byre/24` carried four."""
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2)}


def build(b, part, seed, storeys=1):
    x, z, y = part["x0"] + 1, part["z0"] + 1, part["floor_y"]
    b.place_block(x, y + 1, z, "bamboo_stairs[facing=north,half=bottom]")
'''

CELLAR_TYPE = '''"""A type that digs. The floor is the floor."""
FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2)}


def build(b, part, seed, storeys=1):
    x, z = part["x0"] + 1, part["z0"] + 1
    b.place_cuboid(x, part["floor_y"] - 4, z, x + 1, part["floor_y"] - 1, z + 1,
                   "stone_bricks")
'''


@case
def t_a0b_a_type_that_names_a_terrain_call_is_refused_at_preflight():
    """E013, before a block is placed. Preflight only: the frozen suite stays frozen.

        Three of the seven names, one call each, and the fix text names the part field that
        replaces them -- because a refusal a builder cannot act on costs a round.
        
    """
    rep = lint.preflight(GROUNDWORK_TYPE, forbid=pipeline.TYPE_FORBIDDEN)
    assert not rep.ok, "a type that levels its own ground passed preflight"
    codes = {f.code for f in rep.findings}
    assert codes == {"E013"}, codes
    named = {f.detail["call"] for f in rep.findings}
    assert named == {"clear_trees", "foundation_to_grade", "approach"}, named
    assert all("part['floor_y']" in f.message for f in rep.findings), \
        [f.message for f in rep.findings]
    assert "E013" in lint.FIXES and "floor_y" in lint.FIXES["E013"]

    # ...and E013 is not a check: the suite is unchanged and the catalogue does not
    # grow. do not add checks instead of quality -- still holds.
    assert "E013" not in {c.code for c in lint.CHECKS}, "E013 got into the suite"
    assert "E013" not in lint.catalogue()

    # ...and the same file with the ground work taken out passes
    ok = lint.preflight(TREAD_TYPE, forbid=pipeline.TYPE_FORBIDDEN)
    assert ok.ok, [f.message for f in ok.findings]
    return (f"{len(rep.findings)} E013s naming {sorted(named)}; the suite still has "
            f"{len(lint.CHECKS)} checks and none of them is E013")


@case
def t_a0b_a_type_that_places_a_stair_or_digs_is_refused_as_it_runs():
    """The half preflight cannot see: a block id built at run time, and a y computed
        from `floor_y`. Both name the cell.

        `get_height` is left readable on purpose -- a type has to be able to look at the
        ground it stands over; it may not work it.
        
    """
    _rnd, _vol, b, part = sited("site_e", "fold_barn")
    tb = b.type_builder(part)
    assert tb.get_height(part["x0"], part["z0"]) == b.get_height(part["x0"],
                                                                part["z0"]), \
        "get_height stopped reading through the type builder"

    bad = {}
    for name, src, want in (("stair", TREAD_TYPE, "does not place stairs"),
                            ("cellar", CELLAR_TYPE, "below its own floor")):
        ns = {}
        exec(compile(src, name, "exec"), ns)                        # noqa: S102
        try:
            ns["build"](tb, part, 1)
        except BuildError as e:
            bad[name] = str(e)
            assert want in str(e), (name, str(e))
            continue
        raise AssertionError(f"the {name} type was allowed to build")
    y = part["floor_y"]
    assert f",{y - 4}," in bad["cellar"] and f"which is {y}" in bad["cellar"], \
        bad["cellar"]
    assert "bamboo_stairs" in bad["stair"], bad["stair"]

    # ...and the seven names refuse by value rather than raising, because a refusal is a
    # thing the library hands back and this one has to read like every other
    r = tb.plinth(0, 0, 1, 1, y, "cobblestone")
    assert r["ok"] is False and "site()" in r["reason"], r
    assert tb.refused and tb.refused[0]["call"] == "plinth", tb.refused
    return "; ".join(f"{k}: {v[:58]}" for k, v in bad.items())


@case
def t_a0b_the_library_still_places_what_the_type_asks_it_for():
    """The contract is about what a *type program* writes, not about what the library does
    on its behalf. `building()` lays treads, cuts ground and calls `approach()` every
    time it is asked, and it must go on doing so through a `TypeBuilder` -- or the
    refusal has taken the shell away with the shovel.
    """
    _rnd, vol, b, part = sited("site_e", "fold_barn")
    tb = b.type_builder(part)
    x0, z0, x1, z1 = part["footprint"]
    before = len(b._pending)
    res = tb.building(part["label"], x0 + 1, z0 + 1, x1 - 1, z1 - 1, 2, "gable",
                      mat=MAT)
    assert res["ok"], res["reason"]
    assert len(b._pending) > before, "building() through a TypeBuilder placed nothing"
    b.resolve_steps()
    stairs = [p for p, blk in b._pending.items() if "_stairs[" in blk]
    assert stairs, "the shell laid no treads at all: the case discriminates nothing"
    assert not tb.refused, tb.refused
    return (f"{len(b._pending) - before} cells and {len(stairs)} treads placed by "
            f"building() through a builder that refuses both")


# the checker's fixtures span the ground

@case
def t_a0c_the_fixture_list_is_one_command_and_reproduces_itself():
    """Six plots, six rules, three rounds' caches, and the same six every time.

        The rules are total orders over measured numbers with the round and the label as
        the last tie-break, so there is nothing here to take on trust -- and running it
        twice is the cheapest statement of that.
        
    """
    a = pipeline.check_fixtures()
    b = pipeline.check_fixtures()
    assert a == b, "the fixture rule does not reproduce itself"
    assert len(a) == 6, a
    assert [f["rule"] for f in a] == [r for r, _w in pipeline.FIXTURE_RULES], a
    assert len({f["plot"] for f in a}) == 6, "two fixtures share a label"
    assert pipeline.FIXTURE_HELD_BACK not in {f["round"] for f in a}, \
        "a fixture came off a site the recorded readouts are measured on"
    return "; ".join(f"{f['round'].replace('round', 'r')}/{f['plot']} "
                     f"({f['rule']}: {f['columns']} cols, relief {f['relief']}, "
                     f"{f['water_pct']}% wet)" for f in a)


@case
def t_a0c_the_fixtures_span_dry_steep_and_wet():
    """The property A0 says the old fixture set did not have."""
    fx = pipeline.check_fixtures()
    assert max(f["water_pct"] for f in fx) > 90, \
        f"nothing in the fixture set is a lake: {[f['water_pct'] for f in fx]}"
    assert sum(f["water_pct"] > 0 for f in fx) >= 2, \
        f"only one fixture has any water in it: {[f['water_pct'] for f in fx]}"
    assert max(f["relief"] for f in fx) >= 20, \
        f"nothing in the fixture set is steep: {[f['relief'] for f in fx]}"
    assert min(f["relief"] for f in fx) <= 2, \
        f"nothing in the fixture set is flat: {[f['relief'] for f in fx]}"
    assert max(f["columns"] for f in fx) >= 4 * min(f["columns"] for f in fx), \
        f"the fixtures are all one size: {[f['columns'] for f in fx]}"

    # the old set, measured on the same ruler, has none of it
    old = [r for r in pipeline.plot_ground("site_b")
           if r["plot"] in ("byre", "granary", "quarry_shed", "watch_spur")]
    assert len(old) == 4, old
    assert max(r["water_pct"] for r in old) == 0.0, old
    assert max(r["relief"] for r in old) <= 11, old
    return (f"new: relief {min(f['relief'] for f in fx)}-"
            f"{max(f['relief'] for f in fx)}, water up to "
            f"{max(f['water_pct'] for f in fx)}%, {min(f['columns'] for f in fx)}-"
            f"{max(f['columns'] for f in fx)} columns; the four earlier ones: relief "
            f"{min(r['relief'] for r in old)}-{max(r['relief'] for r in old)}, "
            f"0.0% water")


@case
def t_a0c_a_check_run_executes_on_every_fixture_round():
    """`check_type` used to read one round's `plots.json`. A fixture is now
        `{round, plot}` and the run has to reach three caches, three networks and three
        plot registries -- and come back with one row per fixture per seed.

        The type is the smallest thing that satisfies the contract, because what is being
        checked here is the *checker*.
        
    """
    src = '''FORM = "european_vernacular"
PARAMS = {"storeys": ("int", 1, 2)}


def build(b, part, seed, storeys=1):
    x0, z0, x1, z1 = part["x0"], part["z0"], part["x1"], part["z1"]
    b.building(part["label"], x0 + 1, z0 + 1, x1 - 1, z1 - 1, storeys, "gable")
'''
    d = tempfile.mkdtemp(prefix="ethoslm_fixture_")
    p = os.path.join(d, "program.py")
    open(p, "w").write(src)
    fx = pipeline.check_fixtures()
    rnd, be = _cache("site_b")
    res = pipeline.check_type(rnd, be, p, [], [1], fixtures=fx,
                              voice="white_render_dark_frame")
    assert not res["crashed"], res["text"][:600]
    # ...and, since the voice contract's B1, one row per fixture per seed **per voice**:
    # the author's and the one whose silhouette is least like it.
    assert len(res["rows"]) == 12, [(r["plot"], r["voice"]) for r in res["rows"]]
    assert {r["voice"] for r in res["rows"]} == set(pipeline.check_voices(
        "white_render_dark_frame")), res["rows"]
    assert {r["plot"] for r in res["rows"]} == {f["plot"] for f in fx}, res["rows"]
    assert {r["round"] for r in res["rows"]} == {f["round"] for f in fx}, res["rows"]
    assert all(r["rooms"] for r in res["rows"]), \
        [(r["plot"], r["rooms"]) for r in res["rows"]]
    return (f"12 instances over {len({r['round'] for r in res['rows']})} rounds and "
            f"2 voices in {res['seconds']}s, {res['errors']} own errors, "
            + ", ".join(f"{r['plot']}/{r['voice'][:5]} {r['walk_pct']}%"
                        for r in res["rows"]))


# --------------------------------------------- A3. three new kinds of part A place is
# not a list of buildings. A wall is an **edge**, a gate is a **point**, a market square
# is an **area**, and until A3 `site()` knew one shape of ground -- a rectangle with a
# door on it -- so a type could only ever be a building on a plot. When an instance is
# broken the library is broken.

def _part_fixture(kind: str, part: str | None = None) -> dict:
    """One fixture of a kind, by name where there is more than one."""
    fx = [f for f in pipeline.check_parts() if f["kind"] == kind
          and (part is None or f["part"] == part)]
    assert fx, f"check_parts() produced no {kind} fixture called {part!r}"
    f = dict(fx[0])
    f["label"] = f.pop("part")
    for k in ("round", "rule", "why", "relief", "columns", "gate"):
        f.pop(k, None)
    return f


def _built(kind: str, type_file: str, params=None, part: str | None = None):
    """(round, builder, part) for one reference part type on its own fixture."""
    rnd, be = _cache(pipeline.PART_FIXTURE_ROUND)
    f = _part_fixture(kind, part)
    path = os.path.join(ROOT, type_file)
    src = pipeline.instantiated_source(open(path).read(), [(f, 1, dict(params or {}))],
                                       mat=MAT)
    b = pipeline._run_src(rnd, be, path, src)
    return rnd, b, b.parts[-1]


@case
def t_a3_the_part_fixtures_are_one_command_and_reproduce_themselves():
    a = pipeline.check_parts()
    b = pipeline.check_parts()
    assert a == b, "the part fixture rule does not reproduce itself"
    assert [f["kind"] for f in a] == ["edge", "edge", "edge", "point", "area"], \
        [f["part"] for f in a]
    edge = a[0]
    assert len(edge["path"]) == 3, edge
    assert (abs(edge["path"][1][0] - edge["path"][0][0])
            + abs(edge["path"][2][1] - edge["path"][1][1])
            == pipeline.PART_EDGE_CELLS - 1), edge
    area = a[4]
    assert area["x1"] - area["x0"] + 1 == pipeline.PART_AREA_SIZE, area

    # And the second edge fixture, which is a wall. The L is chosen for the least relief
    # it can find and reads 1.
    ring = a[1]
    assert ring["part"] == "wall_ring" and len(ring["path"]) == 5, ring
    assert ring["path"][0] == ring["path"][-1], f"the loop is not closed: {ring['path']}"
    assert ring["columns"] == 4 * (pipeline.PART_EDGE_LOOP - 1), ring
    assert ring["relief"] == pipeline.PART_EDGE_RELIEF, ring
    gx, gz = ring["gate"]["at"]
    xs = [p[0] for p in ring["path"]]
    zs = [p[1] for p in ring["path"]]
    assert gx in (min(xs), max(xs)) or gz in (min(zs), max(zs)), ring["gate"]

    # And the third, which is where a wall's ground **stops**. Both the others are
    # chosen for the ground the wall runs along; neither says what a wall does at an
    # escarpment, which is the case a walled place on real terrain meets wherever the
    # shelf it stands on ends.
    cliff = a[2]
    assert cliff["part"] == "wall_cliff" and len(cliff["path"]) == 2, cliff
    (ax, az), (bx, bz) = cliff["path"]
    assert abs(bx - ax) + abs(bz - az) == pipeline.PART_EDGE_CELLS - 1, cliff
    assert cliff["drop"] >= pipeline.PART_EDGE_CLIFF, cliff
    assert cliff["drop"] > ring["relief"], \
        "the cliff fixture is not steeper at its end than the loop is along its line"
    return "; ".join(f"{f['kind']} {f['part']} {f.get('rule')}" for f in a)


@case
def t_a3_an_edge_is_graded_segment_by_segment_and_the_wall_has_no_gap():
    """A3's acceptance for an edge: a continuous wall along an L, joined at the vertex.

        The vertex is the case. A wall is built one segment at a time -- that is what an
        edge type is handed -- so the only place a wall can come apart is where two
        segments meet, and a library that grades the two segments independently and does
        not tell the type where they join hands back two walls that miss each other.
        
    """
    rnd, b, part = _built("edge", "types/_edge.py", {"height": 3}, part="wall_L")
    assert part["ground"] == "footing", part
    assert len(part["segments"]) == 2, part["segments"]
    vert = tuple(part["vertices"][0])

    line = []
    for seg in part["segments"]:
        run = Builder._edge_run(tuple(seg["a"]), tuple(seg["b"]))
        line += run if not line else run[1:]
    ys = {}
    for (x, y, z), blk in b._pending.items():
        if (x, z) in set(line) and blk == _voice_block("wall"):
            ys.setdefault((x, z), set()).add(y)
    missing = [c for c in line if len(ys.get(c, ())) < 3]
    assert not missing, \
        f"{len(missing)} of {len(line)} columns of the centre line carry no wall: {missing[:5]}"
    gaps = []
    for a, c in zip(line, line[1:]):
        lo1, hi1 = min(ys[a]), max(ys[a])
        lo2, hi2 = min(ys[c]), max(ys[c])
        if hi1 < lo2 - 1 or hi2 < lo1 - 1:
            gaps.append((a, c))
    assert not gaps, f"the wall steps clear of itself between {gaps[:3]}"
    assert vert in ys, f"nothing stands on the vertex {vert}"
    assert all(vert in {tuple(c) for c in seg["cells"]} for seg in part["segments"]), \
        "the vertex is not in both segments, so neither one owns the join"
    return (f"{len(line)} columns in 2 segments, floors "
            f"{[s['floor_y'] for s in part['segments']]}, relief "
            f"{part['sited']['relief']}, no gap and none at the vertex {list(vert)}")


@case
def t_a3_a_point_is_sited_at_its_cell_and_built_facing_its_facing():
    """A3's acceptance for a point: what stands there is turned the way the plan says.

        A gate that faces the wrong way is a wall with a hole beside the road. The part
        carries `at` and `facing`, `site()` prepares a pad round the cell rather than a
        rectangle somebody drew, and the type builds its posts *across* the facing and
        leaves the way through open *along* it -- which is what this reads back off the
        blocks.
        
    """
    rnd, b, part = _built("point", "types/_point.py", {"height": 3})
    assert part["ground"] in ("plinth", "platform", "deck"), part
    x, z = part["at"]
    assert [x, z] == list(_part_fixture("point")["at"]), part
    ahead = {"north": (0, -1), "south": (0, 1),
             "east": (1, 0), "west": (-1, 0)}[part["facing"]]
    across = (ahead[1], ahead[0])
    y = part["floor_y"]
    posts = [(x + across[0] * s, z + across[1] * s) for s in (-1, 1)]
    through = [(x + ahead[0] * s, z + ahead[1] * s) for s in (-1, 0, 1)]
    for (px, pz) in posts:
        assert all(b._pending.get((px, y + dy, pz)) == _voice_block("frame", "post")
                   for dy in (1, 2, 3)), f"no post across the facing at {(px, pz)}"
    for (tx, tz) in through:
        blocked = [dy for dy in (1, 2, 3)
                   if b._pending.get((tx, y + dy, tz)) in (_voice_block("frame", "post"),
                                                           _voice_block("trim"))]
        assert not blocked, \
            f"the way through at {(tx, tz)} is walled at {blocked} -- this gate faces " \
            f"across {part['facing']} rather than along it"
    return (f"point at {[x, z]} facing {part['facing']}, {part['ground']} at "
            f"y={y}: posts across at {posts}, the way through open along it")


@case
def t_a3_an_area_is_one_level_and_walk_reachable_from_the_lane():
    """A3's acceptance for an area: a square is ground, so the test is on foot."""
    rnd, be = _cache(pipeline.PART_FIXTURE_ROUND)
    vol = rnd.volume()
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    net = rnd.network()
    b.frontage = Frontage(vol, net) if net else None
    b.registry = stages._Registry(rnd.state)
    part = b.site(_part_fixture("area"), mat=MAT)
    b.resolve_steps()
    assert part["ground"] in ("plinth", "platform", "deck"), part
    walk, cols = walkable_from_the_lane(rnd, vol, b, part)
    assert walk == cols, \
        (f"{walk} of {cols} columns of the square are walk-reachable from the lane -- "
         f"{part['sited']['reason']}")

    # ...and the type then paves it and puts a well in it, on the level the library left
    _r, tb, tpart = _built("area", "types/_area.py")
    paved = {(x, z) for (x, y, z), blk in tb._pending.items()
             if y == tpart["floor_y"] and blk == _voice_block("floor")}
    # ...every column but the shaft of its own well, which is water by construction
    well = {(x, z) for (x, y, z) in tb.fitting_cells}
    bare = [(x, z) for x in range(tpart["x0"], tpart["x1"] + 1)
            for z in range(tpart["z0"], tpart["z1"] + 1)
            if (x, z) not in paved and (x, z) not in well]
    assert not bare, f"{len(bare)} columns of the square are neither paved nor a well: {bare[:4]}"
    assert tb.fitting_cells, "the well was not placed through fitting()"
    return (f"{part['ground']} {part['x1'] - part['x0'] + 1}x"
            f"{part['z1'] - part['z0'] + 1} at y={part['floor_y']}, {walk}/{cols} "
            f"columns on foot from the lane, then paved with a well of "
            f"{len(tb.fitting_cells)} registered cells in it")


@case
def t_a3_site_refuses_an_edge_it_cannot_grade_rather_than_guessing():
    """Refusal by name, as everywhere else in this library.

        A segment at an angle the lattice cannot draw is refused and named. Since the
        ground round (phase 4b) a **45-degree** run is sited -- a staircase of the wall's
        width, one block of offset per column, which the library decides and a wall type
        draws in its own frame -- and any other angle is still no.
        
    """
    rnd, be = _cache(pipeline.PART_FIXTURE_ROUND)
    vol = rnd.volume()
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    net = rnd.network()
    b.frontage = Frontage(vol, net) if net else None
    b.registry = stages._Registry(rnd.state)
    at = _part_fixture("edge")["path"][0]
    bad = b.site({"kind": "edge", "label": "skew",
                  "path": [at, [at[0] + 8, at[1] + 5]], "width": 1}, mat=MAT)
    assert not bad["sited"]["ok"] and "45 degrees" in bad["sited"]["reason"], bad
    assert not b._pending, \
        f"a refused edge laid {len(b._pending)} blocks before refusing"
    diag = b.site({"kind": "edge", "label": "diagonal",
                   "path": [at, [at[0] + 8, at[1] + 8]], "width": 1}, mat=MAT)
    assert diag["sited"]["ok"] and diag["segments"][0]["axis"] == "d", diag["sited"]
    assert len(diag["segments"][0]["cells"]) == 9
    short = b.site({"kind": "edge", "label": "one_point", "path": [at]}, mat=MAT)
    assert not short["sited"]["ok"] and "two vertices" in short["sited"]["reason"], short
    return f"{bad['sited']['reason'][:80]}...; and a one-vertex edge is refused too"


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:64s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:64s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:64s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} siting cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
