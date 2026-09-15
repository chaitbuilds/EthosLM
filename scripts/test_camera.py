"""The camera, against worlds built to make it render rock -- and one built not to.

Same shape and same discipline as `test_lint.py`: synthetic volumes in memory, no
server, no Chunky, sub-second. Every case asserts **both** directions, and the ones
that matter assert the third: that the *old* code -- `render.aim` with no Sightline,
which is exactly what every shot list did before this -- fails the same case. A test
that passes on the broken code is not evidence that anything was fixed.

The defect this exists for.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import render  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

SX, SY, SZ = 48, 40, 48
X0, Y0, Z0 = 0, 40, 0
GROUND = 50


def solid_ground(b=None, top=GROUND):
    b = {} if b is None else b
    for x in range(SX):
        for z in range(SZ):
            for y in range(Y0, top + 1):
                b[(x, y, z)] = "stone"
    return b


def hut(b, x0=20, z0=20, x1=27, z1=27, top=GROUND):
    """A walled, roofed hut standing on `top`. The subject of every shot here."""
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            b[(x, top + 5, z)] = "oak_planks"
            for y in range(top + 1, top + 5):
                if x in (x0, x1) or z in (z0, z1):
                    b[(x, y, z)] = "stone_bricks"
    return b


def cliff(b, x_from, x_to, top):
    """A wall of rock from x_from to x_to, up to y=top. The escarpment, in miniature."""
    for x in range(x_from, x_to + 1):
        for z in range(SZ):
            for y in range(Y0, top + 1):
                b[(x, y, z)] = "stone"
    return b


def sight_for(blocks):
    return render.Sightline(Volume.from_blocks(blocks, X0, Y0, Z0, SX, SY, SZ))


CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


SUBJECT = (23.5, GROUND + 3, 23.5)          # the middle of the hut


# ------------------------------------------------------------------- placement

@case
def t_open_camera_is_left_exactly_where_it_was():
    """The control, and the compatibility guarantee. A camera standing in open air with a
    clear view must not move by so much as a block.
    """
    s = sight_for(hut(solid_ground()))
    pos = (8.5, GROUND + 6, 8.5)
    shot = render.aim("ne", pos, SUBJECT, 70, s)
    assert shot.ok, shot.reason
    assert shot.repair == {}, f"an unobstructed camera was moved: {shot.repair}"
    assert shot.view.position == pos, f"{shot.view.position} != {pos}"
    plain = render.aim("ne", pos, SUBJECT, 70, None)
    assert plain.view.position == shot.view.position
    assert abs(plain.view.yaw - shot.view.yaw) < 1e-12
    return "unmoved, and identical to the unchecked camera"


@case
def t_camera_specified_inside_stone_is_moved_into_air():
    """The defect, exactly: a camera whose position is inside a solid block. It must
    come back somewhere it can see from -- and the old code must not."""
    b = hut(solid_ground())
    cliff(b, 0, 14, GROUND + 14)
    s = sight_for(b)
    pos = (7.5, GROUND + 6, 20.5)               # buried in the cliff
    assert not s.free(pos), "test is not exercising the defect: the camera is in air"

    broken = render.aim("ne", pos, SUBJECT, 70, None)
    assert broken.ok and not s.free(broken.view.position), \
        "the unchecked camera did not stay inside the rock -- test discriminates nothing"

    shot = render.aim("ne", pos, SUBJECT, 70, s)
    assert shot.ok, f"no camera found: {shot.reason}"
    assert s.free(shot.view.position), "the repaired camera is still inside a block"
    assert s.clearance(shot.view.position, SUBJECT) >= render.MIN_CLEARANCE
    assert shot.repair, "moved but recorded no repair"
    return f"moved out of rock by {shot.repair['rise']}/{shot.repair['dolly']}/" \
           f"{shot.repair['orbit']} (rise/dolly/orbit)"


@case
def t_camera_in_an_air_pocket_has_no_line_of_sight():
    """In air and still blind. This is the case the in-air test alone would pass and
    the clearance test exists for: a one-block pocket inside the rock."""
    b = hut(solid_ground())
    cliff(b, 0, 14, GROUND + 14)
    for dy in (0, 1):
        b[(7, GROUND + 6 + dy, 20)] = "air"
    s = sight_for(b)
    pos = (7.5, GROUND + 6, 20.5)
    assert s.free(pos), "the pocket is not open air; test proves nothing"
    assert s.clearance(pos, SUBJECT) < render.MIN_CLEARANCE, \
        "the pocket has a clear view; test proves nothing"
    shot = render.aim("ne", pos, SUBJECT, 70, s)
    assert shot.ok, f"no camera found: {shot.reason}"
    assert s.clearance(shot.view.position, SUBJECT) >= render.MIN_CLEARANCE
    return f"pocket clearance {s.clearance(pos, SUBJECT):.1f} -> " \
           f"{s.clearance(shot.view.position, SUBJECT):.1f}"


@case
def t_a_wall_between_camera_and_subject_is_reported_not_rendered():
    """No repair exists, so the shot is declared failed and says why. Silently
    rendering this is the `render_views` failure -- `failed: []` for 24 shots while two
    of the frames were a door leaf edge to edge."""
    b = solid_ground()
    hut(b)
    # bury the camera's whole column: rock from the floor to above the search ceiling,
    # all the way round, so no rise, dolly or orbit within the bounds finds air
    for x in range(0, 16):
        for z in range(SZ):
            for y in range(Y0, GROUND + 34):
                b[(x, y, z)] = "stone"
    s = sight_for(b)
    pos = (7.5, GROUND + 6, 20.5)
    shot = render.aim("ne", pos, SUBJECT, 70, s, max_rise=6, max_dolly=6, max_orbit=30)
    assert not shot.ok, "a camera entombed in rock was accepted"
    assert shot.view is None, "a failed shot still handed back a camera"
    assert "inside a solid block" in shot.reason and "positions tried" in shot.reason, \
        f"the failure does not say why: {shot.reason!r}"

    broken = render.aim("ne", pos, SUBJECT, 70, None)
    assert broken.ok, "the unchecked camera also refused -- test discriminates nothing"
    return f"declared failed: {shot.reason[:58]}..."


@case
def t_search_bounds_are_honoured():
    """A bounded search, and the bound is the caller's. With orbiting forbidden -- what
    a shot looking *down a lane* has to ask for -- the same camera that a free search
    repairs is refused."""
    b = hut(solid_ground())
    cliff(b, 0, 14, GROUND + 14)
    s = sight_for(b)
    pos = (7.5, GROUND + 6, 20.5)
    free = render.aim("ne", pos, SUBJECT, 70, s)
    tight = render.aim("ne", pos, SUBJECT, 70, s, max_rise=0, max_dolly=0, max_orbit=0)
    assert free.ok, "the free search failed; nothing is being compared"
    assert not tight.ok, "a search bounded to no moves at all still moved the camera"
    return "free search repairs, zero-move search refuses"


# ---------------------------------------------------------------- frame checks

def _png(path, value):
    import cv2
    cv2.imwrite(path, np.full((40, 60, 3), value, np.uint8))
    return path


@case
def t_a_black_frame_is_detected_and_a_normal_one_is_not(tmp="/tmp/ethoslm_cam_test"):
    """The backstop. Placement can be right and the frame still come back as rock, so
    the frame itself is measured -- which is the check that turns 'the process exited
    0' into 'there is a building in the picture'."""
    os.makedirs(tmp, exist_ok=True)
    black = _png(os.path.join(tmp, "black.png"), 0)
    dim = _png(os.path.join(tmp, "dim.png"), 8)
    good = _png(os.path.join(tmp, "good.png"), 90)
    assert render.frame_black(black), "a solid black frame was not detected"
    assert render.frame_black(dim), "a frame at mean 8 was not detected"
    assert not render.frame_black(good), "a normal frame was called black"
    assert render.frame_black(os.path.join(tmp, "absent.png")), \
        "a missing frame was not counted as black"
    rep = render.check_frames({"eye": black, "ne": good, "sw": good, "aerial": dim})
    assert not rep["ok"] and rep["black"] == ["aerial", "eye"], rep
    return f"means {rep['means']}"


@case
def t_a_card_refuses_to_compose_a_black_panel(tmp="/tmp/ethoslm_cam_card"):
    """And the refusal reaches the caller. a composer that quietly makes the card anyway is
    how that became a number in a report.
    """
    # Asked of `card.from_dir` directly. It used to go through `e1d_shotlist.card`,
    # which was a four-line wrapper round exactly this call in a script from a finished
    # experiment -- and importing that script to test the composer is how the shot list
    # ended up living in `scripts/`.
    from ethoslm import card as card_mod
    os.makedirs(tmp, exist_ok=True)
    shots = ("eye", "ne", "sw", "aerial")
    for k in shots:
        _png(os.path.join(tmp, f"good_{k}.png"), 90)
        _png(os.path.join(tmp, f"bad_{k}.png"), 90)
    _png(os.path.join(tmp, "bad_ne.png"), 0)

    def card(tag, strict=True):
        return card_mod.from_dir(tmp, tag, os.path.join(tmp, f"{tag}_card.png"),
                                 layout="quad", size=(60, 40), strict=strict)

    assert card("good") is not None, "a clean card was refused"
    assert card("bad") is None, "a card with a black panel was composed"
    why = card_mod.DAMAGE.get("bad", "")
    assert "ne" in why, card_mod.DAMAGE
    assert card("bad", strict=False) is not None, \
        "strict=False did not restore the old behaviour"
    return f"refused with: {why}"


# --------------------------------------------------------------------- geometry

@case
def t_first_hit_is_exact_on_a_known_wall():
    """The ray march itself, against a hand-checkable answer, because everything above
    rests on it."""
    b = {}
    for y in range(Y0, Y0 + 8):
        for z in range(SZ):
            b[(20, y, z)] = "stone"
    s = sight_for(b)
    pos = (10.5, Y0 + 3, 24.5)
    assert abs(s.first_hit(pos, (30.5, Y0 + 3, 24.5)) - 9.5) < 1e-6, \
        s.first_hit(pos, (30.5, Y0 + 3, 24.5))
    assert s.first_hit(pos, (12.5, Y0 + 3, 24.5)) == float("inf"), "phantom hit"
    assert s.first_hit((20.5, Y0 + 3, 24.5), (30.5, Y0 + 3, 24.5)) == 0.0, \
        "a camera inside the wall reported clear air"
    # outside the cached box reads as air: nothing is known there and a camera that has
    # to leave the box to photograph an edge building must not be vetoed on no evidence
    assert not s.solid(-5, Y0 + 3, 24), "outside the volume was treated as solid"
    return "9.5 to the wall, inf past nothing, 0.0 from inside it"


@case
def t_glass_is_seen_through_and_a_shut_door_is_not():
    """Opacity is the light rule, not the movement rule. A camera behind a window can
    photograph a room; a camera behind a shut door cannot."""
    b = {}
    for y in range(Y0, Y0 + 8):
        for z in range(SZ):
            b[(20, y, z)] = "glass"
    s = sight_for(b)
    assert s.first_hit((10.5, Y0 + 3, 24.5), (30.5, Y0 + 3, 24.5)) == float("inf")
    b[(20, Y0 + 3, 24)] = "oak_door[facing=north,half=lower,open=false]"
    s2 = sight_for(b)
    assert s2.first_hit((10.5, Y0 + 3, 24.5), (30.5, Y0 + 3, 24.5)) < 10.0
    return "glass transparent, shut door opaque"


# ------------------------------------------------------- the consolidation cases Three
# properties the pipeline consolidation has to hold, each cheap and each the thing that
# stops the duplication growing back.

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "ethoslm")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


@case
def c_no_module_outside_render_builds_a_camera():
    """The one that stops it happening again.

        Eight files constructed a `View`, and the camera-inside-rock defect had to be fixed
        in four of them. Nothing outside `render.py` may build one now -- ask for a subject
        and get a checked `Shot`, or use `render.raw_shot` and say why.
        
    """
    import ast
    offenders = []
    for d in (SRC, SCRIPTS):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or (d == SRC and f == "render.py"):
                continue
            tree = ast.parse(open(os.path.join(d, f)).read())
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = n.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else "")
                if name == "View":
                    offenders.append(f"{f}:{n.lineno}")
    assert not offenders, f"cameras built outside render.py: {offenders}"
    return "no View() outside src/ethoslm/render.py"


@case
def c_the_no_camera_test_would_catch_one():
    """...and it discriminates: a file that does build one is found."""
    import ast
    tree = ast.parse("from ethoslm import render\nv = render.View('x', (0,0,0), 0, 0, 'PINHOLE', 70)\n")
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", "")) == "View"]
    assert len(found) == 1
    return "the scan finds a planted View() call"


#: Every panel of the two standing shot lists, with the camera position it was rendered
#: from.
STANDING = [("site_c", "site_c", "site_c"), ("site_c", "site_b", "site_b")]


def _standing_shots():
    """(stored record, recomputed Shot) for all 120 standing panels, or None if the
    caches are absent -- this case needs out/, which a fresh worktree does not have."""
    from ethoslm import offline
    from ethoslm.circulate import Network
    sys.path.insert(0, SCRIPTS)
    root = os.path.abspath(os.path.join(SCRIPTS, ".."))
    out = []
    for name, prefix, src in STANDING:
        shots_json = os.path.join(root, "out", name, "step4_shots",
                                  f"shots_{prefix}.json")
        state = os.path.join(root, "out", src)
        if not (os.path.exists(shots_json)
                and os.path.exists(os.path.join(state, "world_built.npz"))):
            return None
        stored = json.load(open(shots_json))["shots"]
        built = offline.load_volume(os.path.join(state, "world_built.npz"))
        net = Network.load(os.path.join(state, "network.json"))
        sight = render.Sightline(built)
        for p in render.merged_plots(state):
            tag = f"{prefix}_{p['label']}"
            if tag not in stored:
                continue
            centre, span = render.plot_subject(p, render.mid_y(built, p))
            got = render.building_card_shots(
                centre, span, threshold=net.threshold(p["label"]) if net else None,
                sight=sight, geom=render.E1D)
            for k, rec in stored[tag].items():
                out.append((f"{tag}_{k}", rec, got.get(k)))
    return out


@case
def c_a7_the_two_whole_place_frames_are_placeable_on_round_11():
    """A7, on the last town this project built.

        "Not black" is `render.check_frames` over the rendered PNGs and it needs Chunky, so
        what is asserted here is the half that can be: both cameras stand in open air, both
        have a clear line to the place, and neither is looking at rock.
        
    """
    from ethoslm import offline
    from ethoslm.circulate import Network
    root = os.path.abspath(os.path.join(SCRIPTS, ".."))
    state = os.path.join(root, "out", "site_f")
    if not os.path.exists(os.path.join(state, "world_built.npz")):
        return "SKIPPED."
    s = json.load(open(os.path.join(state, "site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    centre = (X + S / 2, (s["stats"]["min"] + s["stats"]["max"]) / 2, Z + S / 2)
    built = offline.load_volume(os.path.join(state, "world_built.npz"))
    net = Network.load(os.path.join(state, "network.json"))
    sight = render.Sightline(built)
    bearing = render.approach_bearing(net, centre)
    gate = net.thresholds[0]
    shots = render.place_card_shots(centre, S, gate=gate, bearing=bearing, sight=sight)

    assert set(shots) == set(render.PLACE_SHOTS), sorted(shots)
    for key, shot in sorted(shots.items()):
        # the skyline is a ladder since phase 4; the first rung is the standing camera
        shot = shot[0] if isinstance(shot, list) else shot
        assert shot.ok, f"{key}: {shot.reason}"
        assert sight.free(shot.view.position, 1), f"{key} stands in rock"
        hit = sight.first_hit(shot.view.position, centre)
        assert hit > 1.0, f"{key} has rock against the lens at {hit:.1f}"
    # ...and the skyline camera is where it says it is: sixty blocks outside the gate,
    # at eye level, looking in
    sky = shots["place_skyline"][0].view.position
    assert abs(abs(sky[0] - gate.door[0]) + abs(sky[2] - gate.door[2])
               - render.PLACE_SKYLINE_BACK) <= 1, (sky, gate.door)
    assert sky[1] == gate.y + render.PLACE_SKYLINE_EYE, sky
    # ...and it stands **outside**, which is what "looking in" means. A gate is a hole
    # in a wall with a town on one side of it, and stepping back along a threshold's
    # approach direction.
    d_gate = (gate.door[0] - centre[0]) ** 2 + (gate.door[2] - centre[2]) ** 2
    d_cam = (sky[0] - centre[0]) ** 2 + (sky[2] - centre[2]) ** 2
    assert d_cam > d_gate, ("the skyline camera stands inside the place", sky, gate.door)
    # with no gate there is no arrival, and the frame is absent rather than invented
    assert set(render.place_card_shots(centre, S, sight=sight)) == {"place_aerial"}
    return (f"aerial from bearing {bearing:.0f} at {shots['place_aerial'].view.position[1]:.0f}, "
            f"skyline {render.PLACE_SKYLINE_BACK} blocks off the {gate.facing} side of "
            f"{gate.id} at y={sky[1]:.0f}; both placeable, both with a clear line in")


@case
def c_not_one_camera_moved_on_either_standing_town():
    """Byte-identity, at the only place it can be checked without re-rendering."""
    got = _standing_shots()
    if got is None:
        return "SKIPPED -- no out/ caches in this worktree"
    moved = []
    for name, rec, shot in got:
        want = rec["position"]
        have = list(shot.view.position) if (shot and shot.ok) else None
        if want != have:
            moved.append(f"{name}: {want} -> {have}")
    assert len(got) == 120, f"expected 120 standing panels, found {len(got)}"
    assert not moved, f"cameras moved: {moved[:4]}"
    return f"{len(got)} panels, every camera on the same float"


@case
def c_and_that_check_fails_if_a_camera_moves():
    """The control. Nudge one geometry constant by a hundredth of a span and the
    check above has to notice -- otherwise it is asserting nothing."""
    got = _standing_shots()
    if got is None:
        return "SKIPPED -- no out/ caches in this worktree"
    import dataclasses
    nudged = dataclasses.replace(render.E1D,
                                 close_dist=render.E1D.close_dist + 0.01)
    centre, span = (0.0, 70.0, 0.0), 40.0
    a = render.building_card_shots(centre, span, geom=render.E1D)
    b = render.building_card_shots(centre, span, geom=nudged)
    assert a["ne"].view.position != b["ne"].view.position
    assert a["aerial"].view.position == b["aerial"].view.position, \
        "close_dist must not move the aerial"
    return "a 0.01-span nudge is visible; it does not leak into other panels"


@case
def c_every_card_is_the_same_bytes():
    """The other half of byte-identity: the composer.

        One card function replaced five. The judge keys on `sha256(image_a)`, so a gutter
        one pixel wide in the wrong place would invalidate every cached verdict in the
        project. Digests frozen in `rounds/card-digests.json` from the old composers.
        
    """
    import hashlib
    root = os.path.abspath(os.path.join(SCRIPTS, ".."))
    man = os.path.join(root, "rounds", "card-digests.json")
    if not os.path.exists(man):
        return "SKIPPED -- no digest manifest"
    digests = {k: v for k, v in json.load(open(man)).items() if not k.startswith("_")}
    present = {k: v for k, v in digests.items()
               if os.path.exists(os.path.join(root, k))}
    if not present:
        return "SKIPPED -- no cards in this worktree"
    bad = [k for k, v in present.items()
           if hashlib.sha256(open(os.path.join(root, k), "rb").read()).hexdigest() != v]
    assert not bad, f"{len(bad)} cards changed: {bad[:3]}"
    return f"{len(present)} judged cards byte-identical"


@case
def c_recipes_agree_with_the_arithmetic_they_replaced():
    """`orbit_shot` reproduces each original caller's expression, operation for
    operation. Written out longhand here so the two can be read against each other."""
    import math
    cx, cy, cz, span = -1103.5, 88.0, 1247.5, 41.0

    a = math.radians(45)
    d = span * 0.8
    want_ne = (cx - d * math.sin(a), cy + span * 0.4, cz - d * math.cos(a))
    d2 = span * 1.05
    want_air = (cx - d2 * math.sin(a) * 0.72, cy + d2 * 0.85,
                cz - d2 * math.cos(a) * 0.72)

    got = render.building_card_shots((cx, cy, cz), span, geom=render.E1D)
    assert got["ne"].view.position == want_ne, (got["ne"].view.position, want_ne)
    assert got["aerial"].view.position == want_air
    assert got["aerial"].target == (cx, cy + span * 0.12, cz)
    return "close quarter and aerial land on the original floats exactly"


@case
def c_a_shot_list_refuses_rather_than_returns_rock():
    """End to end through the recipe, not the primitive: a building at the foot of a
    cliff, where the requested three-quarter is inside the rock."""
    b = solid_ground({})
    hut(b, 20, 20, 27, 27)
    cliff(b, 0, 14, GROUND + 26)
    s = sight_for(b)
    plot = {"x0": 20, "z0": 20, "x1": 27, "z1": 27, "label": "hut"}
    centre, span = render.plot_subject(plot, GROUND + 3)
    unchecked = render.building_card_shots(centre, span, geom=render.E1D)
    checked = render.building_card_shots(centre, span, sight=s, geom=render.E1D)
    assert all(sh.ok for sh in unchecked.values()), "unchecked never refuses"
    assert set(checked) == set(unchecked) == {"ne", "sw", "aerial"}
    moved_or_failed = [k for k, sh in checked.items()
                       if (not sh.ok) or sh.repair]
    assert moved_or_failed, "the checked list should have noticed the cliff"
    return f"unchecked: all ok; checked: {sorted(moved_or_failed)} repaired or refused"


@case
def t_bss3_mid_y_answers_for_a_patch_with_nothing_standing_on_it():
    """Halfway up nothing is the ground, and a camera move needs that answer.

        Every card this project has ever framed is on a plot with a building on it, so
        `mid_y` could take the mass for granted until a flythrough asked for the height of
        the world under each step of a camera move. The path from a city's outer gate to
        its palace crosses open ground, and the first step over it killed the whole
        flythrough with `zero-size array to reduction operation minimum` after 123 frames
        had already been shot.
        
    """
    empty = Volume.from_blocks(solid_ground(), X0, Y0, Z0, SX, SY, SZ)
    bare = {"x0": 4, "z0": 4, "x1": 12, "z1": 12}
    got = render.mid_y(empty, bare)
    assert got == GROUND, f"open ground read {got}, not {GROUND}"
    with_hut = Volume.from_blocks(hut(solid_ground()), X0, Y0, Z0, SX, SY, SZ)
    built = render.mid_y(with_hut, {"x0": 20, "z0": 20, "x1": 27, "z1": 27})
    assert built > GROUND, f"the hut reads {built}, which is not above the ground"
    # ...and a flythrough over both is one camera per step, with none of them dropped.
    path = [{"i": i, "at": [8, 8 + i], "look": [24, 24]} for i in range(4)]
    shots = render.flythrough_shots(path, with_hut)
    assert len(shots) == 4, sorted(shots)
    return (f"open ground {got}, the hut {built}, and a four-step move over both "
            f"places {len(shots)} cameras")


# ------------------------------------------------ demo-polish, phase 1d: six camera
# rules Each gates on a measured condition the standing towns never meet, so
# `c_not_one_camera _moved_on_either_standing_town` above is the byte-identity half of
# every case here.

def _wall_world(height: int, z0: int = 30, z1: int = 32, gate_x=(29, 33)):
    """Flat ground with a wall `height` high running along x, and a gap in it for the
    gate. The gate's own pad is 5x5 at x 29..33, z 29..33."""
    b = solid_ground({})
    for x in range(SX):
        if gate_x[0] <= x <= gate_x[1]:
            continue
        for z in range(z0, z1 + 1):
            for y in range(GROUND + 1, GROUND + 1 + height):
                b[(x, y, z)] = "sandstone"
    return b


def _gate_fixture(height: int):
    from ethoslm.circulate import Threshold
    vol = Volume.from_blocks(_wall_world(height), X0, Y0, Z0, SX, 80, SZ)
    gate = {"label": "gate", "kind": "point", "passage": True,
            "x0": 29, "z0": 29, "x1": 33, "z1": 33, "y0": GROUND}
    wall = {"label": "wall", "kind": "edge", "x0": 0, "z0": 30, "x1": SX - 1, "z1": 32,
            "rects": [[0, 30, SX - 1, 32]]}
    # the lane reaches the doorstep from the south (z 34), facing north into the pad
    t = Threshold("gate", 31, 34, GROUND, "north", (31, GROUND + 1, 31))
    return vol, gate, wall, t


@case
def t_dp1d_a_gate_in_a_great_wall_is_framed_on_the_wall_and_shot_from_outside():
    """Finding 2. A 5x5 point in a wall of forty: the subject is the wall at the gate,
    the eye-level rung stands on the road outside the wall, and nothing stands in the
    passage."""
    vol, gate, wall, t = _gate_fixture(40)
    edge = render.edge_of_point(gate, [gate, wall])
    assert edge is wall, "the gate's wall was not found from its rectangle"
    sub = render.gate_subject(gate, edge, vol, GROUND, threshold=t)
    assert sub is not None, "a 40-high wall over a 15-span point did not trigger the rule"
    assert sub["rise"] == 40, sub
    assert sub["span"] == render.GATE_SPAN_MULT * 40, sub
    assert sub["centre"] == (31.0, GROUND + 20.0, 31.0), sub["centre"]
    # the wall's middle (z 30..32) is the gate's own row, so outward is the lane's side
    assert sub["outward"] == (0, 1), sub["outward"]
    shots = render.gate_card_ladder(sub, t, sight=render.Sightline(vol))
    assert set(shots) == set(render.CARD_SHOTS)
    for rung in shots["eye"]:
        assert rung.ok, rung.reason
        x, y, z = rung.view.position
        assert z > 33.5, f"the eye rung stands at z={z}, not outside the wall"
        assert not (29.5 <= z <= 33.5), "an eye rung stands in the passage"
    x, y, z = shots["eye"][0].view.position
    assert z == 31 + 0.5 + render.GATE_EYE_BACK * 40, (z, "first rung stand-back")
    # ...and the orbit rungs frame the wall's span, not the pad's
    own_centre, own_span = render.plot_subject(gate, GROUND + 3)
    plain = render.card_shot_ladder(own_centre, own_span)
    d_wall = ((shots["ne"][0].view.position[0] - 31) ** 2
              + (shots["ne"][0].view.position[2] - 31) ** 2) ** 0.5
    d_pad = ((plain["ne"][0].view.position[0] - 31) ** 2
             + (plain["ne"][0].view.position[2] - 31) ** 2) ** 0.5
    assert d_wall > 2 * d_pad, (d_wall, d_pad)
    return (f"rise {sub['rise']}, span {sub['span']:.0f}, eye rung {render.GATE_EYE_BACK * 40:.0f} "
            f"back outside the wall; three-quarter stands {d_wall:.0f} off against {d_pad:.0f}")


@case
def t_dp1d_a_gate_in_a_low_wall_keeps_the_frame_it_always_had():
    """The gate rule's control, and the standing towns' guarantee. Below `GATE_WALL_RATIO`
    times the span the rule is silent and the pad is the subject.
    """
    vol, gate, wall, t = _gate_fixture(5)
    sub = render.gate_subject(gate, render.edge_of_point(gate, [gate, wall]), vol,
                              GROUND, threshold=t)
    assert sub is None, f"a wall of five over a span of fifteen triggered the rule: {sub}"
    # ...and just over the bar it fires, so the bar is where it says
    vol2, *_ = _gate_fixture(int(render.GATE_WALL_RATIO * 15))
    assert render.gate_subject(gate, wall, vol2, GROUND, threshold=t) is not None
    # a point that is not a passage is never a gate, whatever it stands in
    vol3, gate3, wall3, t3 = _gate_fixture(40)
    assert render.gate_subject(dict(gate3, passage=False), wall3, vol3, GROUND) is None
    return (f"silent at 5 over 15, fires at {int(render.GATE_WALL_RATIO * 15)}, and never "
            f"on a point with no passage")


@case
def t_dp1d_dim_is_a_rung_of_the_ladder(tmp="/tmp/ethoslm_cam_dim"):
    """Finding 3. `shoot_all` climbs the ladder on a dim frame the way it does on a
    black one; a later rung that comes back lit is kept and the record says what it
    repaired; when every rung is dim the brightest is kept and named. A first rung that
    is lit is untouched: tried 1, the same bytes."""
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    real = render.shoot
    calls = []

    def stub(shot, chunks, png, **kw):
        m = float(shot.name.split(":")[1].split(",")[len(calls_for(shot))])
        calls.append(shot.name)
        _png(png, int(m))
        return {"png": png, "status": "black" if m < render.BLACK_MEAN else "ok",
                "mean": m, "seconds": 0.0, "repair": {}}

    def calls_for(shot):
        return [c for c in calls if c == shot.name]

    def ladder(name, means):
        # a raw camera: this case is about the ladder, not where a camera stands
        return [render.raw_shot(f"{name}:{','.join(str(m) for m in means)}",
                                (0.0, 0.0, 0.0), 0.0, 0.0) for _ in means]
    render.shoot = stub
    try:
        got = render.shoot_all({"a": ladder("a", [50]),
                                "b": ladder("b", [16, 40]),
                                "c": ladder("c", [16, 14, 17])}, [], tmp, tag="t")
    finally:
        render.shoot = real
    a, b, c = got["a"], got["b"], got["c"]
    assert a["tried"] == 1 and "dim_repaired" not in a and "dim_kept" not in a, a
    assert b["tried"] == 2 and b["mean"] == 40 and b["dim_repaired"] == [16], b
    assert c["dim_kept"] and c["mean"] == 17 and c["dim_ladder"] == [16, 14, 17], c
    assert abs(render.frame_mean(os.path.join(tmp, "t_c.png")) - 17) < 0.5, \
        "the brightest dim rung was not the frame put back on disk"
    assert not [f for f in os.listdir(tmp) if ".rung" in f], "rung files left behind"
    summ = render.shot_summary({"t": got})
    assert summ["dim_repaired"] == [("t_b", [16], 40)], summ["dim_repaired"]
    assert summ["dim_kept"] == [("t_c", 17)], summ["dim_kept"]
    assert "dim_reported_not_enforced" not in summ
    return "lit first rung untouched; 16 -> 40 repaired at rung 2; 16,14,17 kept the 17 and named it"


@case
def t_dp1d_the_arrival_frame_gives_the_wall_its_third():
    """Finding 7. When `skyline_back` has scaled the stand-back past its floor, the
    gate's foot sits at the lower third of the frame; at the floor the aim is the
    float it always was."""
    import math
    from ethoslm.circulate import Threshold
    centre = (256.0, 80.0, 256.0)
    gate = Threshold("g", 256, 20, 62, "north", (256, 63, 18))

    def elevation(shot):
        p, t = shot.view.position, shot.target
        d = (t[0] - p[0], t[1] - p[1], t[2] - p[2])
        return math.asin(d[1] / math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))

    floor = render.place_card_shots(centre, 512, gate=gate)["place_skyline"][0]
    assert floor.target == (256.0, 80.0 + 512 * 0.05, 256.0), floor.target
    back = render.skyline_back(50)
    assert back > render.PLACE_SKYLINE_BACK
    scaled = render.place_card_shots(centre, 512, gate=gate, rise=50)["place_skyline"][0]
    foot = math.atan2(-render.PLACE_SKYLINE_EYE, back)
    want = math.radians(70) / 2 * (1 - 2 * render.PLACE_SKYLINE_FOOT)
    got = elevation(scaled) - foot
    assert abs(got - want) < 1e-6, (math.degrees(got), math.degrees(want))
    assert elevation(scaled) > elevation(floor), "the scaled aim did not lift the axis"
    # the same rule, stated as where the foot lands in the frame
    frac = 0.5 - got / math.radians(70)
    assert abs(frac - render.PLACE_SKYLINE_FOOT) < 1e-6, frac
    return (f"floor aim unchanged; at {back:.0f} back the axis is {math.degrees(got):.1f} "
            f"deg over the foot, which is {frac:.3f} of the way up a 70-degree frame")


def _two_ring_plan(size: int = 96):
    """Two nested ring walls, a gate on each **with the inner gate sorting first**, and a
    palace at the centre. The case for one arrival rule."""
    from ethoslm.pipeline import stages_media
    o, i = 4, 30
    ring = lambda a, b: [[a, a], [b, a], [b, b], [a, b], [a, a]]        # noqa: E731
    parts = [
        {"kind": "edge", "name": "ring_wall_outer", "defines": "ring_wall",
         "type": "wall", "seed": 1, "params": {}, "width": 1, "path": ring(o, size - o)},
        {"kind": "edge", "name": "ring_wall_inner", "defines": "ring_wall",
         "type": "wall", "seed": 2, "params": {}, "width": 1, "path": ring(i, size - i)},
        {"kind": "point", "name": "a_gate", "defines": "ring_gate", "type": "ring_gate",
         "seed": 3, "params": {}, "at": [size // 2, i], "facing": "north"},
        {"kind": "point", "name": "b_gate", "defines": "ring_gate", "type": "ring_gate",
         "seed": 4, "params": {}, "at": [size // 2, o], "facing": "north"},
    ]
    plan = {"parts": parts,
            "compounds": [{"name": "palace", "x0": 40, "z0": 40, "x1": 55, "z1": 55}]}
    spec = {"defining_parts": [
        {"name": "ring_wall", "relation": "concentric", "kind": "edge", "count": 2},
        {"name": "palace", "relation": "centre", "kind": "area", "count": 1}]}
    site = {"origin": [0, 0], "size": size}
    return plan, spec, site, stages_media


@case
def t_dp1d_one_rule_for_the_arrival_gate():
    """The arrival is the outermost ring's gate for both the skyline and the
    flythrough, even when an inner gate sorts first and comes first in plan order."""
    from ethoslm.circulate import Network, Threshold
    from ethoslm import pipeline
    plan, spec, site, sm = _two_ring_plan()
    g = sm.arrival_gate(plan, spec)
    assert g is not None and g["name"] == "b_gate", g
    # plan order and name order both put a_gate first: the old rule's answer
    assert pipeline.plan_parts(plan)[2]["name"] == "a_gate"
    net = Network({}, [Threshold("a_gate", 48, 31, 60, "north", (48, 61, 30)),
                       Threshold("b_gate", 48, 5, 60, "north", (48, 61, 4))])

    class R:
        def plan(self_): return plan
        def place_spec(self_): return spec
        def parts(self_): return pipeline.plan_parts(plan)
    t = pipeline._gate_threshold(R(), net)
    assert t is not None and t.id == "b_gate", t
    path = sm.flythrough_path(plan, spec, site)
    assert path[0]["at"][1] < 4, path[0]      # begins outside the outer gate
    # with no rings in the spec the rule falls back to plan order, as it always did
    assert sm.arrival_gate(plan, {"defining_parts": []})["name"] == "a_gate"
    return "outer gate b_gate for the skyline, the flythrough and the threshold; plan order without rings"


@case
def t_dp1d_the_flythrough_stops_short_of_its_subject():
    """Finding 8. The last camera stands the subject's half-span plus `FLY_STANDOFF`
    short of the centre, looks at the centre, and its pitch is under the limit. The
    old path ended over the centre looking straight down."""
    import math
    plan, spec, site, sm = _two_ring_plan()
    path = sm.flythrough_path(plan, spec, site)
    last, look = path[-1]["at"], path[-1]["look"]
    assert look == [47.5, 47.5], look
    assert all(p["look"] == look for p in path), "the look moved during the move"
    half = (55 - 40) / 2.0
    short = math.hypot(look[0] - last[0], look[1] - last[1])
    assert abs(short - (half + render.FLY_STANDOFF)) < 0.05, (short, half)
    # the cameras over flat ground: the final pitch is under the registered limit
    b = {}
    for x in range(-64, 160):
        for z in range(-64, 160):
            b[(x, GROUND, z)] = "stone"
    vol = Volume.from_blocks(b, -64, Y0, -64, 224, 40, 224)
    shots = render.flythrough_shots(path, vol)
    final = shots[f"fly_{len(path) - 1:03d}"]
    p, t = final.view.position, final.target
    d = (t[0] - p[0], t[1] - p[1], t[2] - p[2])
    down = -math.degrees(math.asin(d[1] / math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)))
    assert 0 < down < render.FLY_FINAL_PITCH_MAX, down
    return (f"ends {short:.1f} short of the centre ({half:.1f} + {render.FLY_STANDOFF}), "
            f"final pitch {down:.1f} deg down against {render.FLY_FINAL_PITCH_MAX:.0f}")


@case
def t_dp1d_the_whole_place_frames_load_the_chunks_the_camera_stands_over():
    """Finding 9. The aerial stands `0.95 * S` off the centre, which is outside a
    site loaded to sixteen blocks; the chunk list now reaches every camera."""
    from ethoslm.circulate import Threshold
    X, Z, S = 0, 0, 64
    centre = (X + S / 2, 70.0, Z + S / 2)
    gate = Threshold("g", 32, 2, 62, "north", (32, 63, 1))
    shots = render.place_card_shots(centre, S, gate=gate, rise=48)
    old = {tuple(c) for c in render.chunk_list(X - 16, Z - 16, X + S + 16, Z + S + 16)}
    new = {tuple(c) for c in render.place_chunks(X, Z, S, shots)}
    missing_old, missing_new = [], []
    for k, sh in shots.items():
        for rung in (sh if isinstance(sh, list) else [sh]):
            px, _py, pz = rung.view.position
            c = (int(px) // 16, int(pz) // 16)
            (missing_old if c not in old else []).append(k)
            (missing_new if c not in new else []).append(k)
    assert not missing_new, f"cameras outside the loaded chunks: {missing_new}"
    assert missing_old, "the old chunk list already reached every camera; nothing measured"
    assert old < new
    return (f"{len(old)} chunks reached neither {missing_old}; {len(new)} reach both cameras")


@case
def t_dp4_the_arrival_camera_climbs_out_of_the_ground():
    """Phase 4, and the rule the render found. A scaled stand-back puts the arrival camera
    hundreds of blocks outside the site, on ground nothing levelled and no volume
    covers, at the *gate's* level. The first rung does not move, and each one after it
    lifts the eye and re-aims from where it now stands.
    """
    import math
    from ethoslm.circulate import Threshold
    centre = (256.0, 80.0, 256.0)
    gate = Threshold("g", 256, 20, 62, "north", (256, 63, 18))

    def elevation(shot):
        p, t = shot.view.position, shot.target
        d = (t[0] - p[0], t[1] - p[1], t[2] - p[2])
        return math.asin(d[1] / math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))

    rungs = render.place_card_shots(centre, 512, gate=gate, rise=50)["place_skyline"]
    assert isinstance(rungs, list) and len(rungs) == render.PLACE_SKYLINE_RUNGS, rungs
    # the first rung is the validated camera, unmoved: this is what keeps every recorded
    # arrival frame the bytes it always was
    one = render.place_card_shots(centre, 512, gate=gate, rise=50)["place_skyline"][0]
    assert one.view.position == rungs[0].view.position == (
        256.5, gate.y + render.PLACE_SKYLINE_EYE, 18 - render.skyline_back(50) + 0.5), \
        rungs[0].view.position
    # each rung after it stands `PLACE_SKYLINE_LIFT` higher, over the same column
    for n, r in enumerate(rungs):
        p = r.view.position
        assert (p[0], p[2]) == (rungs[0].view.position[0], rungs[0].view.position[2])
        assert p[1] == rungs[0].view.position[1] + n * render.PLACE_SKYLINE_LIFT, p
    # ...and keeps the composition: the gate's foot still lands at PLACE_SKYLINE_FOOT
    back = render.skyline_back(50)
    for n, r in enumerate(rungs):
        eye = render.PLACE_SKYLINE_EYE + n * render.PLACE_SKYLINE_LIFT
        got = elevation(r) - math.atan2(-eye, back)
        frac = 0.5 - got / math.radians(70)
        assert abs(frac - render.PLACE_SKYLINE_FOOT) < 1e-6, (n, frac)
    # the lift clears the bank that buried the demo's own camera: surface y 69 at the
    # camera's column, gate level y 62, so rung 1 stands at 64 and rung 2 at 80
    assert rungs[0].view.position[1] < 69 < rungs[1].view.position[1], \
        [r.view.position[1] for r in rungs]
    # at the floor there is a ladder too, and its first rung is the old float exactly
    flat = render.place_card_shots(centre, 512, gate=gate)["place_skyline"]
    assert flat[0].target == (256.0, 80.0 + 512 * 0.05, 256.0), flat[0].target
    assert flat[0].view.position[1] == gate.y + render.PLACE_SKYLINE_EYE
    return (f"{len(rungs)} rungs at y "
            f"{[int(r.view.position[1]) for r in rungs]}, foot held at "
            f"{render.PLACE_SKYLINE_FOOT:.3f} on every one; first rung unmoved")


@case
def t_dp4_the_plot_registry_carries_what_the_camera_rules_read():
    """1d's *a gate is framed on its wall* read `passage`, `kind`, `y0` and `rects`
    off `merged_plots`, and `merged_plots` returned a label and four numbers. The rule
    was dead in the pipeline while its own case passed on hand-built plot dicts, and
    all three of the demo's gates were photographed as close masonry a second time.

    The case is the wiring, not the rule: what the registry writes down comes out the
    other side, and `edge_of_point` finds the wall a gate stands in from it."""
    import tempfile
    rows = [
        {"label": "wall", "x0": 0, "z0": 0, "x1": 40, "z1": 40, "kind": "edge",
         "y0": 63, "rects": [[0, 0, 40, 2], [0, 38, 40, 40]]},
        {"label": "gate", "x0": 18, "z0": 0, "x1": 22, "z1": 2, "kind": "point",
         "passage": True, "y0": 63},
        {"label": "house", "x0": 8, "z0": 10, "x1": 16, "z1": 18},
        {"label": "house", "x0": 8, "z0": 18, "x1": 16, "z1": 24},
    ]
    with tempfile.TemporaryDirectory() as d:
        json.dump(rows, open(os.path.join(d, "plots.json"), "w"))
        got = {p["label"]: p for p in render.merged_plots(d)}
    assert set(got) == {"wall", "gate", "house"}, sorted(got)
    assert got["gate"]["passage"] is True and got["gate"]["kind"] == "point"
    assert got["gate"]["y0"] == 63
    assert got["wall"]["rects"] == [[0, 0, 40, 2], [0, 38, 40, 40]]
    # the two rows of one label still merge to one rectangle, and a row with no `rects`
    # gets its own as one
    assert (got["house"]["x0"], got["house"]["z0"], got["house"]["x1"],
            got["house"]["z1"]) == (8, 10, 16, 24), got["house"]
    assert got["house"]["rects"] == [[8, 10, 16, 18], [8, 18, 16, 24]]
    # ...and the rule the wiring exists for now finds its wall
    e = render.edge_of_point(got["gate"], list(got.values()))
    assert e is not None and e["label"] == "wall", e
    return ("passage, kind, y0 and rects come through; two rows of a label merge to "
            "one rectangle; edge_of_point finds the gate's wall")


@case
def t_dp4_a_reused_frame_is_one_whose_camera_did_not_move(tmp="/tmp/ethoslm_cam_reuse"):
    """`reuse`'s docstring said "when its camera did not move" and nothing compared
    the camera: any frame on disk that was not black and not dim was kept. Wiring up
    *a gate is framed on its wall* gave all three of the demo's gates a camera on a
    span four times their own and the render printed `kept` twelve times.

    The scene this call asks for is written beside the frame, and a reuse is allowed
    only when the scene it would ask for now is that one."""
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    png = _png(os.path.join(tmp, "f.png"), 120)
    a = render.raw_shot("f", (0.0, 80.0, 0.0), 0.0, 0.0)
    b = render.raw_shot("f", (0.0, 96.0, 0.0), 0.0, 0.0)      # the camera moved
    chunks, wide = [[0, 0]], [[0, 0], [1, 0]]
    jvm = []
    real = render._java

    def stub(args, timeout=3600):
        jvm.append(args[1] if len(args) > 1 else args[0])
        return subprocess.CompletedProcess(args, 0, "", "")
    render._java = stub
    try:
        def shoot(shot, ch=chunks):
            return render.shoot(shot, ch, png, scene="f", reuse=True,
                                record_kind=None)
        # 1. no sidecar: kept, as every cached frame in this project is, and said so
        n = len(jvm)
        r = shoot(a)
        assert r["reused"] and r["camera_checked"] is False, r
        assert len(jvm) == n, "a frame with no sidecar was re-rendered"
        # 2. rendered once: the sidecar is written beside the frame
        r = render.shoot(a, chunks, png, scene="f", reuse=False, record_kind=None)
        assert "reused" not in r and r["status"] == "ok", r
        assert len(jvm) == n + 2, jvm               # -render and -snapshot
        assert os.path.exists(png + ".scene.json")
        # 3. the same camera: kept, and now it is checked
        n = len(jvm)
        r = shoot(a)
        assert r["reused"] and r["camera_checked"] is True, r
        assert len(jvm) == n, "a frame whose camera did not move was re-rendered"
        # 4. the camera moved: re-shot
        r = shoot(b)
        assert "reused" not in r, r
        assert len(jvm) == n + 2, "a frame whose camera moved was kept"
        # 5. ...and so does a changed chunk list, which is finding 9's whole fix
        n = len(jvm)
        r = shoot(b, wide)
        assert "reused" not in r, r
        assert len(jvm) == n + 2, "a frame whose chunk list changed was kept"
    finally:
        render._java = real
    return ("no sidecar keeps and says camera_checked false; a matching scene keeps; "
            "a moved camera and a changed chunk list re-shoot")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"ok   {name:52s} {fn()}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name:52s} {e}")
        except Exception as e:
            bad += 1
            print(f"ERR  {name:52s} {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} camera cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
