"""The library owns entry, and the API doc is the only API doc.

    $PY scripts/test_approach.py

Four things are asserted here, in the order the spec registers them:

  1. **approach() makes a jump-only door walkable.** Synthetic cases first -- a door on
     a plinth, a door already at grade, a door with nothing to approach -- and then the
     registered acceptance: the sixteen product-claim first drafts, each re-executed
     with one `approach()` call appended, where every candidate whose door could only be
     jumped to must come back walk-reachable from outdoors.
  2. **E002 is walk-only.** A door you can only jump to is an error; the same door with
     a step laid to it is not.
  3. **Preflight rejects `try`**, and lets a grandfathered round through.
  4. **One API document.** Every name the build environment exposes is written out in
     `buildlib.API_DOC`, and nothing else in the tree writes a build-API signature.

the cached world, the plots, the network and the programs. It is the case that decides
the spec's acceptance, so it is not silently skipped: it says loudly that it could not
run.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import lint, observe, offline  # noqa: E402
from ethoslm.buildlib import API_DOC, Builder  # noqa: E402
from ethoslm.circulate import Network, Threshold  # noqa: E402
from ethoslm.frontage import Frontage  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R7 = os.path.join(ROOT, "out", "site_b")

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


# ------------------------------------------------------------------- fixtures
SX = SZ = 40
Y0, GROUND = 50, 60
G = 2 * (GROUND + 1)          # the half-height a walker stands at on flat ground


def world(extra: dict | None = None) -> Volume:
    b = {}
    for x in range(SX):
        for z in range(SZ):
            for y in range(Y0, GROUND + 1):
                b[(x, y, z)] = "stone"
    b.update(extra or {})
    return Volume.from_blocks(b, 0, Y0, 0, SX, 64, SZ)


def lane(cells, label, door, facing="north") -> Network:
    """A stub network: a run of lane cells at ground level and one threshold."""
    return Network({(x, z): {"y": GROUND, "rank": 0, "face": None} for (x, z) in cells},
                   [Threshold(label, door[0], door[2] + 1, GROUND, facing, tuple(door))])


def builder(vol: Volume, net: Network | None = None) -> Builder:
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    b.frontage = Frontage(vol, net) if net else None
    return b


def walk_from_outside(vol: Volume, pending: dict, door) -> bool:
    """Is the doorway walk-reachable from the open ground round the edge?"""
    from ethoslm import stages
    applied = stages.apply_pending(vol, pending)
    nav = observe.Nav(applied)
    s = nav.stance_near(door[0], door[2], door[1], tol=2)
    if s is None:
        return False
    return (door[0], door[2], s) in set(
        nav.flood(nav.perimeter_seeds(inset=2, step=3), max_jumps=0))


def hut_on_a_plinth(vol_extra: dict, floor: int, wall: int = 3) -> tuple:
    """A one-room hut whose floor sits `floor - GROUND` blocks above the ground, with a
    door in its south wall. The defect the whole round is about, in eight lines."""
    b = dict(vol_extra)
    x0, z0, x1, z1 = 14, 14, 20, 20
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            for y in range(GROUND + 1, floor + 1):
                b[(x, y, z)] = "stone_bricks"          # the plinth, solid to grade
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            for y in range(floor + 1, floor + 1 + wall):
                b[(x, y, z)] = "stone_bricks" if edge else "air"
    door = (17, floor + 1, z1)
    b[(door[0], door[1], door[2])] = "oak_door[facing=south,half=lower]"
    b[(door[0], door[1] + 1, door[2])] = "oak_door[facing=south,half=upper]"
    return b, door


# ----------------------------------------------------------- 1. approach() works
@case
def t_a_raised_door_becomes_walkable():
    extra, door = hut_on_a_plinth({}, floor=GROUND + 3)
    vol = world(extra)
    net = lane([(17, z) for z in range(24, 30)], "hut", door)
    b = builder(vol, net)
    assert not walk_from_outside(vol, {}, door), "fixture is already walkable"
    res = b.approach("hut", *door)
    b.resolve_steps()
    assert res["ok"], f"approach refused: {res['reason']}"
    assert res["cells"] > 0, "nothing was laid"
    assert walk_from_outside(vol, b._pending, door), \
        "approach reported ok and the door is still not walkable on foot"
    return f"{res['cells']} columns laid up a 3-block plinth"


@case
def t_a_door_at_grade_lays_nothing():
    b0 = {}
    x0, z0, x1, z1 = 14, 14, 20, 20
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            for y in range(GROUND + 1, GROUND + 4):
                b0[(x, y, z)] = "stone_bricks" if edge else "air"
    door = (17, GROUND + 1, z1)
    b0[door] = "oak_door[facing=south,half=lower]"
    b0[(door[0], door[1] + 1, door[2])] = "oak_door[facing=south,half=upper]"
    vol = world(b0)
    net = lane([(17, z) for z in range(24, 30)], "hut", door)
    b = builder(vol, net)
    res = b.approach("hut", *door)
    assert res["ok"], res["reason"]
    assert res["cells"] == 0, f"laid {res['cells']} columns at a door already at grade"
    assert not b._pending, "a door already walkable had blocks placed at it anyway"
    return "0 columns, 0 blocks"


@case
def t_no_doorway_is_reported_not_guessed():
    b = builder(world())
    res = b.approach("nothing_here")
    assert not res["ok"], "approach claimed success with no door and no threshold"
    assert "no door" in res["reason"], res["reason"]
    assert not b._pending, "blocks placed with nothing to approach"
    return res["reason"][:46]


@case
def t_it_will_not_cut_through_the_building():
    """The path may not take a wall out to reach a door. Ring the hut in its own
    masonry to head height and there is no line to the door that does not go through
    it -- the honest answer is a refusal, not a hole."""
    extra, door = hut_on_a_plinth({}, floor=GROUND + 3)
    vol = world()
    net = lane([(17, z) for z in range(24, 30)], "hut", door)
    b = builder(vol, net)
    for (p, blk) in extra.items():
        b.place_block(p[0], p[1], p[2], blk)
    for x in range(12, 23):                     # a solid apron wall round the plinth
        for z in (12, 22):
            for y in range(GROUND + 1, GROUND + 6):
                b.place_block(x, y, z, "stone_bricks")
    for z in range(12, 23):
        for x in (12, 22):
            for y in range(GROUND + 1, GROUND + 6):
                b.place_block(x, y, z, "stone_bricks")
    before = dict(b._pending)
    res = b.approach("hut", *door)
    assert not res["ok"], "approach claimed to reach a door walled in on every side"
    cut = [p for p in before if b._pending.get(p) != before[p]]
    assert not cut, f"approach overwrote {len(cut)} blocks of the builder's own wall"
    return res["reason"][:56]


@case
def t_it_does_not_eat_the_door_it_was_called_for():
    """A door has **no collision** in the walk model -- `observe._classify` gives an
        `oak_door` (0, 0, 0) -- so an approach that only protected blocks that get in the
        way treated the leaf, its jambs and its lintel as empty air, and cleared headroom
        straight through them. The building came back walkable and with no door in it, and
        W002 was the only thing that noticed. Anything the program placed that is not its
        own air is now protected.
        
    """
    extra, door = hut_on_a_plinth({}, floor=GROUND + 3)
    vol = world()
    net = lane([(17, z) for z in range(26, 32)], "hut", door)
    b = builder(vol, net)
    for (p, blk) in extra.items():
        b.place_block(p[0], p[1], p[2], blk)
    # ...and the things a doorway() puts round a leaf, in the columns beside it.
    for dx in (-1, 1):
        for dy in (0, 1):
            b.place_block(door[0] + dx, door[1] + dy, door[2], "spruce_log[axis=y]")
    b.place_block(door[0], door[1] + 2, door[2], "stripped_spruce_log[axis=x]")
    for x in range(door[0] - 2, door[0] + 3):          # the builder's own cleared air
        for y in range(door[1], door[1] + 3):
            b.place_block(x, y, door[2] + 1, "air")
    doorway = {p: v for p, v in b._pending.items()
               if p[2] in (door[2],) and abs(p[0] - door[0]) <= 1}
    res = b.approach("hut")
    b.resolve_steps()
    eaten = {p: (v, b._pending.get(p)) for p, v in doorway.items()
             if b._pending.get(p) != v}
    assert not eaten, f"approach() overwrote the doorway itself: {eaten}"
    assert res["ok"], res["reason"]
    assert walk_from_outside(vol, b._pending, door), res["reason"]
    return f"{res['cells']} columns laid, leaf, jambs and lintel untouched"


@case
def t_the_sixteen_product_claim_first_drafts():
    """The registered acceptance. Every candidate whose door was jump-only must come
    back walk-reachable with one `approach()` call appended, and no candidate whose
    door was already walkable may be broken by it."""
    from ethoslm import pipeline
    diag_path = os.path.join(R7, "product-claim", "entry_diagnosis.json")
    cfg = os.path.join(ROOT, "rounds", "site_b_selection.json")
    cache = offline.world_cache("site_b")
    if not os.path.exists(diag_path):
        # The drafts are the subject. A checkout that does not carry that recorded round
        # has nothing here to be wrong about; one that carries the drafts and not the
        # ground it was measured on does.
        return "SKIPPED -- no recorded drafts on this checkout"
    if not os.path.exists(cache):
        raise AssertionError(
            "COULD NOT RUN: the recorded drafts are here and the pre-build cache they "
            "were measured on is not -- this is the case the spec's acceptance is "
            "written against, so a missing fixture is a failure and not a skip")
    plots = {"chapel": "byre", "forge": "granary",
             "granary": "quarry_shed", "windmill": "watch_spur"}
    diag = json.load(open(diag_path))["candidates"]
    rnd = pipeline.Round.load(cfg)
    be = pipeline.OfflineBackend(rnd)
    vol = be.volume
    scratch = os.path.join(ROOT, "out", "site_d", "approach_check")
    os.makedirs(scratch, exist_ok=True)
    fixed = kept = 0
    for prompt, label in sorted(plots.items()):
        for i in range(4):
            cid = f"{prompt}/c{i}"
            src = os.path.join(R7, "arms", "product-claim", prompt, f"c{i}",
                               f"{prompt}.py")
            if not os.path.exists(src):
                raise AssertionError(f"COULD NOT RUN: {src} is missing")
            was = {tuple(d["at"]): d for d in diag[cid]["doors"]}
            jump_only = [p for p, d in was.items()
                         if d["jump_from_outside"] and not d["walk_from_outside"]]
            walked = [p for p, d in was.items() if d["walk_from_outside"]]
            p = os.path.join(scratch, f"{prompt}_c{i}.py")
            open(p, "w").write(open(src).read()
                               + f"\n\napproach({label!r})\n")
            b = be.execute(p, vol)
            for door in jump_only:
                assert walk_from_outside(vol, b._pending, door), \
                    f"{cid}: the door at {door} is still not walkable on foot"
                fixed += 1
            for door in walked:
                assert walk_from_outside(vol, b._pending, door), \
                    f"{cid}: approach broke a door at {door} that was already walkable"
                kept += 1
    assert fixed == 7, f"expected the record's 7 jump-only doors, found {fixed}"
    return f"{fixed} jump-only doors now walkable, {kept} walkable doors kept"


# --------------------------------------------------------------- 2. E002 on foot
def _ctx(vol, plots=None, net=None):
    return lint.Context.build(vol, plots or [], network=net,
                              region=(0, 0, SX - 1, SZ - 1))


@case
def t_e002_fires_on_a_door_you_can_only_jump_to():
    # One course of plinth: the sill a block above the ground you arrive on, which is a
    # jump the old E002 allowed and 47% of the unreachable floor in the record.
    extra, door = hut_on_a_plinth({}, floor=GROUND + 1)
    ctx = _ctx(world(extra))
    codes = [f for f in lint.lint(ctx).findings if f.code == "E002"]
    assert codes, ("E002 is silent on a door a block above the ground it is "
                   "approached from -- it is reading from_anywhere again")
    assert "jump" in codes[0].message, codes[0].message
    return codes[0].message[-42:]


@case
def t_e002_is_silent_once_the_way_in_is_laid():
    extra, door = hut_on_a_plinth({}, floor=GROUND + 2)
    vol = world(extra)
    net = lane([(17, z) for z in range(24, 30)], "hut", door)
    b = builder(vol, net)
    res = b.approach("hut", *door)
    b.resolve_steps()
    assert res["ok"], res["reason"]
    from ethoslm import stages
    ctx = _ctx(stages.apply_pending(vol, b._pending), net=net)
    bad = [f for f in lint.lint(ctx).findings if f.code == "E002"]
    assert not bad, f"E002 still fires after approach(): {[f.message for f in bad]}"
    return "E002 clear"


# ---------------------------------------------------------------- 3. no `try`
@case
def t_preflight_rejects_try():
    r = lint.preflight('try:\n    place_block(0, 0, 0, "stone")\nexcept Exception:\n'
                       '    pass\n')
    assert not r.ok, "a program that swallows its own exceptions passed preflight"
    assert [f.code for f in r.findings] == ["E012"], [f.code for f in r.findings]
    clean = lint.preflight('place_block(0, 0, 0, "stone")\n')
    assert clean.ok, [f.message for f in clean.findings]
    return r.findings[0].message[:46]


@case
def t_a_grandfathered_round_still_runs():
    src = 'try:\n    place_block(0, 0, 0, "stone")\nexcept Exception:\n    pass\n'
    assert lint.preflight(src, allow_try=True).ok, \
        "allow_try did not let a program the record already contains through"
    cfgs = [json.load(open(os.path.join(ROOT, "rounds", f)))
            for f in ("site_b.json", "site_c.json")]
    assert all(c["flags"].get("allow_try") for c in cfgs), \
        "rounds 7 and 8 store programs written before the rule and must say so"
    return "rounds 7 and 8 declare it; nothing else does"


# ------------------------------------------------------------ 4. one API document
@case
def t_every_public_method_is_in_the_api_doc():
    missing = [n for n in offline.BUILD_API
               if not re.search(r"^ {4}" + re.escape(n) + r"\(", API_DOC, re.M)]
    assert not missing, f"exposed to programs and undocumented: {missing}"
    return f"{len(offline.BUILD_API)} names, all written out"

@case
def t_nothing_else_composes_an_api_section():
    """One document. Two files may read it; none may restate it.

        `make_briefs.py`'s API section omitted `check_door` and `check_walkable` while the
        settlement brief had them, so which checks a builder knew about depended on which
        script composed its brief -- and 0 of 16 product-claim programs called either.
        
    """
    readers = set()
    offenders = []
    sig = re.compile(r"^\s+(" + "|".join(re.escape(n) for n in offline.BUILD_API)
                     + r")\(.*\)\s*->", re.M)
    # Only the places that *compose a brief*. `src/ethoslm` writes these signatures in its
    # own docstrings, which is documentation of the code for whoever maintains it and
    # never reaches a builder.
    for base in (os.path.join(ROOT, "scripts"), os.path.join(R7, "product-claim")):
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                text = open(p, encoding="utf-8", errors="replace").read()
                if "API_DOC" in text:
                    readers.add(f)
                if f == "test_approach.py":
                    continue
                if sig.search(text):
                    offenders.append(os.path.relpath(p, ROOT))
    assert not offenders, f"these compose their own API section: {offenders}"
    # The brief makers that are *here*: `make_briefs.py` is a file of the recorded round
    # it was written in, so a checkout without that record has one of them.
    want = {"make_settlement_prompts.py"}
    if os.path.isdir(os.path.join(R7, "product-claim")):
        want.add("make_briefs.py")
    assert want <= readers, \
        f"every brief maker must read API_DOC; want {sorted(want)}, found {sorted(readers)}"
    return f"{len(readers)} files read it, none restate it"


def main():
    bad = 0
    for name, fn in CASES:
        try:
            note = fn()
            print(f"ok   {name:46s} {note}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:46s} {e}")
        except Exception as e:                    # noqa: BLE001 - reported, not raised
            bad += 1
            import traceback
            print(f"ERR  {name:46s} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad}/{len(CASES)} approach cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
