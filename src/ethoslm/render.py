"""Chunky headless rendering, and **the one place a camera is built.**

Chunky stable 2.4.6 stops at MC 1.20.4, so this uses the 2.5.0 snapshot. Scenes are
minted as JSON here rather than through the GUI.

So a shot is three things rather than one: a position that is checked against the world
before it renders (`Sightline.free`), a line of sight that is required and searched for
(`aim`), and a frame that is measured after it comes back (`frame_mean`). A shot that
cannot be placed is **declared failed and named**; a frame that comes back below
`BLACK_MEAN` is **declared black and named**. Neither is composed into a card.
`render_views` recording `failed: []` for 24 shots while two frames were a door leaf
edge-to-edge is the same failure this exists to stop.

**Why every camera now lives here.** That defect was fixable in one place and was
instead fixed in four, because the shot list had been reimplemented independently in
`e1d_shotlist.py`, `step3_render.py`, `step4_render.py` and `render_views.py`. Eight
files constructed a `View`. So the module now owns not just the primitives but the
*recipes*: `orbit_shot`, `overhead_shot`, `door_shot`, `lane_door_shot`, `lane_shot`,
`eye_shot`, and the named shot lists built from them (`building_card_shots`,
`bearing_shots`, `town_quarter_shots`, `fixed_three`). Their constants sit in
`CardGeometry` records a few lines apart, where a drift between two of them is visible
rather than a diff away, and `scripts/test_camera.py` asserts that nothing outside this
file constructs a `View`."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHUNKY = os.path.join(ROOT, "run", "chunky")
CHUNKY_HOME = os.path.join(CHUNKY, "home")
SCENES = os.path.join(CHUNKY_HOME, "scenes")
JAVA = os.environ.get("ETHOSLM_JAVA") or os.path.join(ROOT, "run", "jdk", "bin", "java")
MC_JAR = os.path.join(CHUNKY_HOME, "minecraft", "versions", "1.21.11", "1.21.11.jar")
LOADER = os.environ.get("ETHOSLM_JAVA_LOADER", "")
WORLD = os.path.join(ROOT, "run", "server", "world")


@dataclass
class View:
    name: str
    position: tuple[float, float, float]
    yaw: float          # radians
    pitch: float        # radians, -pi/2 looks straight down
    projection: str     # PINHOLE | PARALLEL
    fov: float


# Chunky's orientation is spherical and measured off the DOWN axis, which is not obvious
# and cost a calibration sweep to pin down (see scripts/calibrate_camera.py): pitch 0 ->
# straight down pitch -pi/2 -> level (Chunky's default) pitch -pi -> straight up yaw was
# calibrated the same way against coloured axis markers: yaw 0 -> -X, yaw pi/2 -> +Z,
# yaw pi -> +X, yaw -pi/2 -> -Z (north, the default). So with phi = -pitch the view
# direction is (-sin(phi)cos(yaw), -cos(phi), sin(phi)sin(yaw)). In the overhead view
# (yaw -pi/2) image right is +X and image up is -Z.
DOWN = 0.0
LEVEL = -math.pi / 2


def _look(pos, target) -> tuple[float, float]:
    dx, dy, dz = (target[0] - pos[0], target[1] - pos[1], target[2] - pos[2])
    yaw = math.atan2(dz, -dx)
    pitch = -math.acos(-dy / math.sqrt(dx * dx + dy * dy + dz * dz))
    return yaw, pitch


def views_for(cx: float, cy: float, cz: float, span: float) -> list[View]:
    """Three fixed views framing a build of size `span` centred on (cx, cy, cz).

        Kept as the unchecked form because `render()` frames a build on ground it has just
        made and has no cached volume to check against. `fixed_three` is the aimed version.
        
    """
    d = span * 1.1
    # 45 degrees below horizontal: shallower angles run out of loaded chunks and fill
    # most of the frame with sky.
    aerial_pos = (cx - d * 0.62, cy + d * 0.88, cz + d * 0.62)
    ay, ap = _look(aerial_pos, (cx, cy + span * 0.15, cz))
    # ground camera stands back and looks slightly up at the build, eye height 2
    ground_pos = (cx - d * 1.15, cy + 2, cz + d * 1.15)
    gy, gp = _look(ground_pos, (cx, cy + span * 0.3, cz))
    return [
        View("overhead", (cx, cy + span * 2.5, cz), -math.pi / 2, DOWN, "PARALLEL", span * 1.4),
        View("aerial45", aerial_pos, ay, ap, "PINHOLE", 60),
        View("ground", ground_pos, gy, gp, "PINHOLE", 75),
    ]


# ------------------------------------------------------------------- sightlines

#: A panel whose mean pixel is below this is rock, not a building. Taken from the
#: measurement that found the defect. Frozen here before it was used.
BLACK_MEAN = 12.0

#: A frame at or above `BLACK_MEAN` and below this is **dim**: not rock, but deep shade
#: -- an eye-level camera in an unlit gate passage. Demo-polish, phase 1d (finding 3):
#: `shot_summary` reported these and enforced nothing, and four of the demo's 210 panels
#: are on disk at means of 16.3 to 22.4 because of it. Dim climbs the same ladder black
#: does; only when every rung is dim is the brightest kept, and named. Twice the black
#: bar, which is where the report's own band already ended.
DIM_MEAN = 2 * BLACK_MEAN

#: How far the view ray must travel before it hits anything. This is what "line of
#: sight" has to mean here, and it took three formulations to get to it. It cannot mean
#: "the ray reaches the point it is aimed at": the subject *is* solid, and every good
#: aerial ever rendered in this project hits a roof long before it arrives. It cannot
#: mean "a fraction of the subject's bounding box is unoccluded" either -- `built_cells`
#: on a scarp calls the whole forty-block rock column under a dwelling "the building",
#: which is the same 2-D-plot abstraction failure that put the cameras in the cliff to
#: begin with, so that measure scores bright, correct panels at zero. What separates the
#: damaged panels from the good ones, measured over all 120 panels of both towns' shot
#: lists, is simply **how much open air is in front of the lens**: every one of the 13
#: black panels is 0.00, and the lowest good panel is 2.52. 2.0 sits in that gap. Worth
#: saying plainly: on this evidence the in-air test alone already catches 13 of 13. This
#: bound is the guard for the case that has not happened yet -- a camera in a one-block
#: air pocket inside the rock, which is "in air" and still photographs nothing.
MIN_CLEARANCE = 2.0


#: The outer gate's eye-level rung stood under an acacia and the panel was leaves, which
#: the sightline lets through and the ladder had no rung for. A rung whose eye stands
#: within `FOLIAGE_REACH` blocks of leaves is passed over for the next rung.
FOLIAGE_REACH = 3


class Sightline:
    """Occlusion over a cached `observe.Volume`, so a camera can be checked before it
        renders rather than after a judge has already been shown the result.

        Opacity is `Volume.tables()["opaque"]`, which is the light rule and not the movement
        rule -- glass and bars are see-through, a shut door is not, leaves are. That is the
        right vocabulary here: the question is what a camera can see, not what a player can
        walk through.

        Cells outside the cached volume read as **air**. Nothing is known about them, and a
        camera that has to leave the box to see a building at the edge of one is a camera
        this class must not veto on no evidence.
        
    """

    def __init__(self, vol):
        t = vol.tables()
        self.vol = vol
        self.opaque = t["opaque"][vol.codes]
        self.x0, self.y0, self.z0 = vol.x0, vol.y0, vol.z0
        self.sx, self.sy, self.sz = vol.codes.shape

    @classmethod
    def load(cls, path: str) -> "Sightline":
        from . import offline
        return cls(offline.load_volume(path))

    def solid(self, x: int, y: int, z: int) -> bool:
        i, j, k = x - self.x0, y - self.y0, z - self.z0
        if not (0 <= i < self.sx and 0 <= j < self.sy and 0 <= k < self.sz):
            return False
        return bool(self.opaque[i, j, k])

    def foliage(self, pos, reach: int = FOLIAGE_REACH) -> int:
        """How many leaf blocks stand within `reach` of `pos`. Leaves are see-through
        to `visible` and opaque to a lens: a camera in a canopy photographs leaves."""
        import numpy as np
        leafy = getattr(self, "_leafy", None)
        if leafy is None:
            names = [str(n).split("[")[0] for n in self.vol.palette]
            leafy = np.array([n.endswith("_leaves") for n in names], bool)
            self._leafy = leafy
        x, y, z = (int(math.floor(pos[0])) - self.x0, int(math.floor(pos[1])) - self.y0,
                   int(math.floor(pos[2])) - self.z0)
        x0, x1 = max(0, x - reach), min(self.sx, x + reach + 1)
        y0, y1 = max(0, y - reach), min(self.sy, y + reach + 1)
        z0, z1 = max(0, z - reach), min(self.sz, z + reach + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return 0
        return int(leafy[self.vol.codes[x0:x1, y0:y1, z0:z1]].sum())

    def free(self, pos, clearance: int = 1) -> bool:
        """Is a camera at this point in open air, with `clearance` cells above it?

                The cell above matters because a camera wedged under a slab renders the
                underside of the slab across the top of the frame, which is not black and is
                still not a photograph of a building.
                
        """
        x, y, z = (int(math.floor(c)) for c in pos)
        return not any(self.solid(x, y + d, z) for d in range(clearance + 1))

    def first_hit(self, pos, target) -> float:
        """Distance along pos->target at which the ray first enters an opaque cell,
        or `inf`. Amanatides & Woo voxel traversal, in world coordinates."""
        d = [target[i] - pos[i] for i in range(3)]
        dist = math.sqrt(sum(c * c for c in d))
        if dist < 1e-9:
            return 0.0 if self.solid(*(int(math.floor(c)) for c in pos)) else math.inf
        d = [c / dist for c in d]
        cell = [int(math.floor(c)) for c in pos]
        step, tmax, tdelta = [0, 0, 0], [math.inf] * 3, [math.inf] * 3
        for i in range(3):
            if d[i] > 1e-12:
                step[i] = 1
                tmax[i] = (cell[i] + 1 - pos[i]) / d[i]
                tdelta[i] = 1.0 / d[i]
            elif d[i] < -1e-12:
                step[i] = -1
                tmax[i] = (cell[i] - pos[i]) / d[i]
                tdelta[i] = -1.0 / d[i]
        if self.solid(*cell):
            return 0.0
        t = 0.0
        # +2 so a ray that only clears the subject by a cell is still traversed past it
        while t <= dist + 2.0:
            a = tmax.index(min(tmax))
            t = tmax[a]
            cell[a] += step[a]
            tmax[a] += tdelta[a]
            if t > dist + 2.0:
                break
            if self.solid(*cell):
                return max(t, 0.0)
        return math.inf

    def clearance(self, pos, target) -> float:
        """Open air in front of the lens: how far the view ray gets before it hits
        something, capped at the distance to the subject."""
        return min(self.first_hit(pos, target), math.dist(pos, target))

    def visible(self, pos, target, min_clearance: float = MIN_CLEARANCE) -> bool:
        return self.clearance(pos, target) >= min_clearance


# ---------------------------------------------------------------------- aiming

@dataclass
class Shot:
    """One camera, and the whole truth about how it got there."""
    name: str
    view: View | None
    target: tuple
    ok: bool
    reason: str = ""
    repair: dict = field(default_factory=dict)
    requested: tuple | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "reason": self.reason,
                "repair": self.repair,
                "position": list(self.view.position) if self.view else None,
                "requested": list(self.requested) if self.requested else None}


#: The bounded search. Ordered by cost so the smallest repair that works is the one
#: taken, and so a shot that needed no repair is byte-identical to what the old code
#: would have rendered.
MAX_RISE = 16
MAX_DOLLY_OUT = 18
MAX_ORBIT = 90


def _candidates(pos, target, max_rise=MAX_RISE, max_dolly=MAX_DOLLY_OUT,
                max_orbit=MAX_ORBIT):
    """(cost, position, repair) for every camera this search will consider.

        Three moves, in the order the spec puts them and one it does not. Out along the
        view axis and up are the two the spec names, and on a scarp they are often not
        enough: a close three-quarter view of a dwelling cut into a cliff has forty blocks
        of rock behind it, so backing out goes *deeper* into the hill. Orbiting the same
        radius is the move that finds air, and it is bounded at a quarter turn and charged
        a cost so it is never preferred to simply standing back.

        A caller whose shot *means* a direction -- looking down a lane -- passes
        `max_orbit=0` and gets only the moves that keep the shot the shot it asked for.
        
    """
    tx, ty, tz = target
    dx, dz = pos[0] - tx, pos[2] - tz
    radius = math.hypot(dx, dz)
    bearing = math.atan2(dz, dx) if radius > 1e-9 else 0.0
    dy = pos[1] - ty
    orbits = [0]
    for k in range(1, max_orbit // 15 + 1):
        orbits += [15 * k, -15 * k]
    dollies = list(range(0, max_dolly + 1)) + \
        [-d for d in range(1, min(max_dolly, int(max(radius - 3, 0))) + 1)]
    out = []
    for orbit in orbits:
        a = bearing + math.radians(orbit)
        for dolly in dollies:
            r = radius + dolly
            if r < 2.0:
                continue
            for rise in range(0, max_rise + 1):
                cost = rise + abs(dolly) + abs(orbit) * 0.12
                p = (tx + r * math.cos(a), ty + dy + rise, tz + r * math.sin(a))
                out.append((cost, p, {"rise": rise, "dolly": dolly, "orbit": orbit}))
    out.sort(key=lambda c: c[0])
    return out


def aim(name: str, pos, target, fov: float, sight: "Sightline | None" = None,
        projection: str = "PINHOLE", headroom: int = 1,
        min_clearance: float = MIN_CLEARANCE, budget: int = 4000,
        max_rise: int = MAX_RISE, max_dolly: int = MAX_DOLLY_OUT,
        max_orbit: int = MAX_ORBIT) -> Shot:
    """A View that has been checked against the world, or a Shot that says why not.

        With no `sight` this is the old behaviour exactly -- position, `_look`, done -- so
        every caller that has no cached volume to hand keeps working and keeps its results
        comparable. With one, the requested position is tested and, if it is inside a block
        or has rock against the lens, the bounded search above runs and the cheapest working
        camera wins. If none does, `ok` is False and `reason` names which test failed.
        
    """
    def view_at(p):
        yaw, pitch = _look(p, target)
        return View(name, p, yaw, pitch, projection, fov)

    def good(p):
        if not sight.free(p, headroom):
            return False, "inside a solid block"
        if not sight.visible(p, target, min_clearance):
            return False, (f"no line of sight -- only "
                           f"{sight.clearance(p, target):.1f} blocks of air ahead")
        return True, ""

    target = tuple(float(c) for c in target)
    pos = tuple(float(c) for c in pos)
    if sight is None:
        return Shot(name, view_at(pos), target, True, "", {}, pos)
    ok, why = good(pos)
    if ok:
        return Shot(name, view_at(pos), target, True, "", {}, pos)
    tried = 0
    for cost, p, repair in _candidates(pos, target, max_rise, max_dolly, max_orbit):
        if not any(repair.values()):
            continue                      # the requested camera; already rejected
        tried += 1
        if tried > budget:
            break
        if good(p)[0]:
            repair = dict(repair, was=why, cost=round(cost, 2))
            return Shot(name, view_at(p), target, True, "", repair, pos)
    return Shot(name, None, target, False,
                f"{why}; no camera within rise<={max_rise}, "
                f"dolly<=+{max_dolly}, orbit<={max_orbit} deg can see it "
                f"({tried} positions tried)", {}, pos)


# --------------------------------------------------------------- shot recipes One
# function per *kind* of camera, and every shot list in the project is built out of
# them. Each reproduces its original caller's arithmetic in the original order, because
# a float that rounds differently is a camera that moved, and a camera that moved is a
# judge cache that misses and an experiment that stops reproducing.

def orbit_shot(name: str, centre, *, bearing: float, dist: float, rise: float,
               fov: float, pull: float = 1.0, target_lift: float = 0.0,
               sight: "Sightline | None" = None, **aim_kw) -> Shot:
    """A camera standing off `centre` on a compass bearing, looking back at it.

        Every close three-quarter, every aerial, every quarter view of a town and every
        ring of bearings in this project is this function with different numbers. Bearing
        is degrees clockwise from north, the convention all four shot lists already used:
        45 is from the northeast.

        `pull` scales the horizontal stand-off *after* the bearing is applied, which is the
        odd-looking `d * sin(a) * 0.72` in the aerial recipes -- it pulls the camera in
        without lowering it, so the shot looks down more steeply. It defaults to 1.0, and
        multiplying by 1.0 is exact, so the close-quarter callers are unaffected.
        
    """
    cx, cy, cz = centre
    a = math.radians(bearing)
    pos = (cx - dist * math.sin(a) * pull, cy + rise, cz - dist * math.cos(a) * pull)
    return aim(name, pos, (cx, cy + target_lift, cz), fov, sight, **aim_kw)


def overhead_shot(name: str, centre, span: float) -> Shot:
    """Straight down, orthographic, from two and a half spans up.

        Never aimed, and the reason is not laziness: there is nothing above a town, and a
        parallel projection has no single view ray to test -- every pixel is its own.
        
    """
    cx, cy, cz = centre
    v = View(name, (cx, cy + span * 2.5, cz), -math.pi / 2, DOWN, "PARALLEL", span * 1.4)
    return Shot(name, v, centre, True)


def section_shot(name: str, centre, span: float, height: float) -> Shot:
    """A doll's-house plan: straight down, orthographic, from `height`. The caller
    clips the roof off with `scene_json(yclip=...)`; this is only the camera."""
    cx, _, cz = centre
    return Shot(name, View(name, (cx, height, cz), -math.pi / 2, DOWN, "PARALLEL", span),
                centre, True)


def eye_shot(name: str, pos, target, fov: float,
             sight: "Sightline | None" = None, **aim_kw) -> Shot:
    """A camera at an explicit position looking at an explicit point. The escape hatch
    for shots whose geometry is neither an orbit nor a door -- E3's street eyes."""
    return aim(name, pos, target, fov, sight, **aim_kw)


def raw_shot(name: str, pos, yaw: float, pitch: float,
             projection: str = "PINHOLE", fov: float = 70) -> Shot:
    """A camera pointed by raw yaw and pitch rather than at a subject.

        The one legitimate reason to want this is calibration: `scripts/calibrate_camera.py`
        sweeps angles against coloured axis markers to find out what Chunky's orientation
        convention actually is, and it cannot aim at anything because what "aim" means is
        the thing being measured. Everything else should be asking for a subject.
        
    """
    return Shot(name, View(name, tuple(pos), yaw, pitch, projection, fov),
                tuple(pos), True)


#: Which way a threshold's `facing` points, as a unit step in (x, z). Was defined
#: separately in e1d_shotlist, step3_render, render_views and e3_resite.
FACE = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def door_shot(name: str, threshold, *, back: int = 9, eye: int = 3,
              target_lift: float = 1.0, fov: float = 70,
              sight: "Sightline | None" = None, **aim_kw) -> Shot:
    """Stand `back` blocks off a reserved threshold, on its approach side, at the
        threshold's own height plus `eye`, and look at the doorway.

        The straight-line form: it steps back along the approach direction without asking
        the lane how high it is there. That is what E1d measured and what step 3 and step 4
        were judged on, so it is kept exactly. `lane_door_shot` is the form that follows
        the ground.
        
    """
    dx, dz = FACE[threshold.facing]
    door = threshold.door
    pos = (door[0] - dx * back + 0.5, threshold.y + eye, door[2] - dz * back + 0.5)
    target = (door[0] + 0.5, door[1] + target_lift, door[2] + 0.5)
    return aim(name, pos, target, fov, sight, **aim_kw)


def lane_door_shot(name: str, threshold, cells: dict, *, back: int = 9, eye: int = 2,
                   look_down: float = 0.0, fov: float = 80,
                   sight: "Sightline | None" = None, **aim_kw):
    """Back off along the *lane*, following its heights, and look at the door.

        Returns (Shot, subject_centre). Different from `door_shot` in that the camera
        stands on the ground a person would actually walk in on rather than floating at a
        fixed height above the threshold, which matters where the lane is in a cutting.
        
    """
    dx, dz = FACE[threshold.facing]
    px, pz, py = threshold.x, threshold.z, threshold.y
    for k in range(1, back + 1):
        c = (threshold.x - dx * k, threshold.z - dz * k)
        if c in cells:
            px, pz, py = c[0], c[1], cells[c]["y"]
        else:
            break
    pos = (px + 0.5, py + 1 + eye, pz + 0.5)
    target = (threshold.door[0] + 0.5, threshold.door[1] + 1.5 - look_down,
              threshold.door[2] + 0.5)
    if (px, pz) == (threshold.x, threshold.z):   # nowhere to back off to
        pos = (px + 0.5 - dx * 3, py + 1 + eye, pz + 0.5 - dz * 3)
    return aim(name, pos, target, fov, sight, **aim_kw), tuple(threshold.door)


def lane_shot(name: str, a, b, cells: dict, *, eye: int = 2, fov: float = 85,
              default_y: int = 70, sight: "Sightline | None" = None) -> Shot:
    """Eye level on lane cell `a`, looking down the lane at cell `b`.

        Aimed with `max_orbit=0, max_dolly=0`: a shot that *means* a direction is no longer
        that shot if the search orbits it round to find air, so only standing up out of a
        cutting is allowed to repair it.
        
    """
    ya = cells[a]["y"] if a in cells else default_y
    yb = cells[b]["y"] if b in cells else ya
    pos = (a[0] + 0.5, ya + 1 + eye, a[1] + 0.5)
    target = (b[0] + 0.5, yb + 1 + eye, b[1] + 0.5)
    return aim(name, pos, target, fov, sight, max_orbit=0, max_dolly=0)


# ------------------------------------------------------------------ shot lists

@dataclass(frozen=True)
class CardGeometry:
    """The numbers behind one four-panel building card.

        Two of these exist and they differ, which is the point of writing them down beside
        each other: E1d's list is the one that was validated and judged, step 3 stood its
        cameras slightly further back. Nothing reconciles them, because reconciling them
        would move a camera.
        
    """
    close_dist: float          # x span
    close_rise: float          # x span
    close_fov: float
    aerial_dist: float         # x span
    aerial_height: float       # x aerial_dist
    aerial_lift: float         # x span
    aerial_fov: float
    aerial_pull: float
    door_back: int
    door_eye: int
    door_fov: float
    pad: int                   # chunk padding round the plot


#: The validated list: eye-level at the door, close three-quarters from the northeast
#: and southwest, one northeast aerial. Never framed on a defect. Used by E1d, by step
#: 4's checks, and by every settlement photographed since.
E1D = CardGeometry(0.8, 0.4, 70, 1.05, 0.85, 0.12, 60, 0.72, 9, 3, 70, 24)

#: Step 3's readout. Same four panels, cameras a little further out and higher.
STEP3 = CardGeometry(0.9, 0.45, 70, 1.15, 0.85, 0.12, 60, 0.72, 9, 3, 70, 12)

#: The four keys of a building card, in the order a card composer lays them out.
CARD_SHOTS = ("eye", "ne", "sw", "aerial")


def building_card_shots(centre, span: float, *, threshold=None,
                        sight: "Sightline | None" = None,
                        geom: CardGeometry = E1D) -> dict:
    """The four neutral views of one building, each checked against the world.

        `centre` is (cx, cy, cz) and `span` the footprint's long side plus its margin --
        both computed by `plot_subject` for a plot rectangle. Returns {key: Shot}; the
        `eye` key is absent when the structure has no reserved threshold, which is what
        the composer's blank panel is for.
        
    """
    views = {}
    if threshold is not None:
        views["eye"] = door_shot("eye", threshold, back=geom.door_back,
                                 eye=geom.door_eye, fov=geom.door_fov, sight=sight)
    for key, bearing in (("ne", 45), ("sw", 225)):
        views[key] = orbit_shot(key, centre, bearing=bearing,
                                dist=span * geom.close_dist,
                                rise=span * geom.close_rise,
                                fov=geom.close_fov, sight=sight)
    d = span * geom.aerial_dist
    views["aerial"] = orbit_shot("aerial", centre, bearing=45, dist=d,
                                 rise=d * geom.aerial_height,
                                 target_lift=span * geom.aerial_lift,
                                 fov=geom.aerial_fov, pull=geom.aerial_pull,
                                 sight=sight)
    return views


#: How far back and up each fallback camera stands, as a multiple of the first one's
#: numbers. Ordered, and the first entry is the validated camera itself, so a shot that
#: needs no fallback renders exactly the bytes it always did.
CARD_FALLBACKS = ((1.0, 1.0), (1.45, 1.7), (1.9, 2.3))


def card_shot_ladder(centre, span: float, *, threshold=None,
                     sight: "Sightline | None" = None,
                     geom: CardGeometry = E1D) -> dict:
    """{key: [Shot, ...]} -- the validated camera first, then further-back fallbacks.

    So each key gets a short ladder. Nothing about the first rung changes, which is what
    keeps every standing card byte-identical; the rungs below it are only ever reached
    by a shot that would otherwise have been thrown away."""
    out: dict = {k: [] for k in CARD_SHOTS if k != "eye" or threshold is not None}
    for back_mul, rise_mul in CARD_FALLBACKS:
        if threshold is not None:
            out["eye"].append(door_shot(
                "eye", threshold, back=int(round(geom.door_back * back_mul)),
                eye=int(round(geom.door_eye * rise_mul)), fov=geom.door_fov,
                sight=sight))
        for key, bearing in (("ne", 45), ("sw", 225)):
            out[key].append(orbit_shot(
                key, centre, bearing=bearing, dist=span * geom.close_dist * back_mul,
                rise=span * geom.close_rise * rise_mul, fov=geom.close_fov,
                sight=sight))
        d = span * geom.aerial_dist * back_mul
        out["aerial"].append(orbit_shot(
            "aerial", centre, bearing=45, dist=d, rise=d * geom.aerial_height,
            target_lift=span * geom.aerial_lift, fov=geom.aerial_fov,
            pull=geom.aerial_pull, sight=sight))
    return out


def plot_subject(plot: dict, cy: float, margin: int = 10):
    """(centre, span) for a plot rectangle -- the subject a card is framed on."""
    cx = (plot["x0"] + plot["x1"]) / 2
    cz = (plot["z0"] + plot["z1"]) / 2
    span = max(plot["x1"] - plot["x0"], plot["z1"] - plot["z0"]) + margin
    return (cx, cy, cz), span


#: **A point part on an edge is framed on the edge.** Demo-polish, 1d (finding 2). A
#: gate is a `point` with a 5x5 pad, so `plot_subject` frames a span of fifteen -- and
#: the demo's outer gate stands in a wall of forty-eight, so all four of its panels were
#: close masonry and the eye-level one was inside the passage. When the edge a passage
#: point stands on rises at least `GATE_WALL_RATIO` times the point's own span above the
#: point's floor, the subject is the wall at the gate: a span of `GATE_SPAN_MULT` times
#: that rise, centred at the wall's mid-height, and the eye-level rung stands
#: `GATE_EYE_BACK` times the rise back on the road **outside** the wall, looking
#: `GATE_EYE_LIFT` times the rise up it. the demo's is forty-eight.
GATE_WALL_RATIO = 2.0
GATE_SPAN_MULT = 1.5
GATE_EYE_BACK = 0.5
GATE_EYE_LIFT = 0.25


def edge_rise(built, x: int, z: int, y: int, span: int | None = None) -> int:
    """The tallest standing column within `span` of (x, z), above `y`. 0 with nothing
    to read. `approach_rise` is this at a gate's reserved doorstep; the default span
    is its `PLACE_SKYLINE_SPAN`."""
    from . import observe
    if built is None:
        return 0
    span = PLACE_SKYLINE_SPAN if span is None else int(span)
    sub = built.sub(int(x) - span, int(z) - span, 2 * span + 1, 2 * span + 1)
    h, _wet = observe.ground_heights(sub)
    return max(0, int(h.max()) - int(y))


def edge_of_point(plot: dict, plots: list):
    """The edge row of a plot registry whose swept rectangles hold this point's
    centre, or None. A gate stands *in* its wall, so the wall's segment holds it."""
    cx = (plot["x0"] + plot["x1"]) // 2
    cz = (plot["z0"] + plot["z1"]) // 2
    for e in plots:
        if e.get("kind") != "edge" or e.get("label") == plot.get("label"):
            continue
        for r in (e.get("rects") or [(e["x0"], e["z0"], e["x1"], e["z1"])]):
            if min(r[0], r[2]) <= cx <= max(r[0], r[2]) and \
                    min(r[1], r[3]) <= cz <= max(r[1], r[3]):
                return e
    return None


def gate_subject(plot: dict, edge: dict, built, base_y: int, margin: int = 10,
                 threshold=None):
    """{centre, span, rise, outward} for a passage point on an edge, or None where the
        rule does not apply -- the edge is not `GATE_WALL_RATIO` times the point's own span
        above the point's floor. See `GATE_WALL_RATIO`.

        `outward` is the unit step (dx, dz) from the gate to the side of the wall away
        from the wall's own middle, which is where the road in stands.
        
    """
    if not plot.get("passage") or edge is None:
        return None
    cx = (plot["x0"] + plot["x1"]) / 2
    cz = (plot["z0"] + plot["z1"]) / 2
    own = max(plot["x1"] - plot["x0"], plot["z1"] - plot["z0"]) + margin
    rise = edge_rise(built, int(cx), int(cz), int(base_y))
    if rise < GATE_WALL_RATIO * own:
        return None
    # The wall runs along the long axis of the segment the gate stands in; outward is
    # across it, away from the wall's own middle -- and where the gate *is* the wall's
    # middle across that axis (a straight run, not a ring), the road's side of the
    # reserved doorstep, which is the side a person arrives from.
    seg = None
    for r in (edge.get("rects") or [(edge["x0"], edge["z0"], edge["x1"], edge["z1"])]):
        if min(r[0], r[2]) <= int(cx) <= max(r[0], r[2]) and \
                min(r[1], r[3]) <= int(cz) <= max(r[1], r[3]):
            seg = r
            break
    seg = seg or (edge["x0"], edge["z0"], edge["x1"], edge["z1"])
    along_x = abs(seg[2] - seg[0]) >= abs(seg[3] - seg[1])
    ex = (edge["x0"] + edge["x1"]) / 2
    ez = (edge["z0"] + edge["z1"]) / 2
    off = (cz - ez) if along_x else (cx - ex)
    if off == 0 and threshold is not None:
        dx, dz = FACE[threshold.facing]
        sign = -(dz if along_x else dx) or 1
    else:
        sign = 1 if off >= 0 else -1
    outward = (0, sign) if along_x else (sign, 0)
    return {"centre": (cx, float(base_y) + rise / 2.0, cz),
            "span": float(GATE_SPAN_MULT * rise), "rise": int(rise),
            "outward": outward}


def gate_card_ladder(subject: dict, threshold, *, sight: "Sightline | None" = None,
                     geom: "CardGeometry" = None) -> dict:
    """`card_shot_ladder` for a gate framed on its wall: the three orbit rungs on the
    wall's span, and the eye-level rungs on the road outside the wall looking in."""
    import dataclasses
    geom = geom or E1D
    out = card_shot_ladder(subject["centre"], subject["span"], threshold=None,
                           sight=sight, geom=geom)
    if threshold is None:
        return out
    ox, oz = subject["outward"]
    facing = {(0, -1): "south", (0, 1): "north", (1, 0): "west", (-1, 0): "east"}[
        (ox, oz)]                       # door_shot steps back *against* the facing
    t = dataclasses.replace(threshold, facing=facing)
    rise = subject["rise"]
    out["eye"] = [door_shot("eye", t, back=int(round(GATE_EYE_BACK * rise * back_mul)),
                            eye=int(round(geom.door_eye * rise_mul)),
                            target_lift=GATE_EYE_LIFT * rise, fov=geom.door_fov,
                            sight=sight)
                  for back_mul, rise_mul in CARD_FALLBACKS]
    out["eye"] = pass_foliage(out["eye"], sight)
    return {k: out[k] for k in CARD_SHOTS}


def pass_foliage(rungs: list, sight: "Sightline | None") -> list:
    """The ladder with every rung whose eye stands in a canopy moved behind the rungs
    that do not, each marked; a rung that stands clear is the bytes it was. Where every
    rung is in leaves the order is kept and each says so. See `FOLIAGE_REACH`."""
    if sight is None or not rungs:
        return rungs
    clear, leafy = [], []
    for r in rungs:
        n = sight.foliage(r.view.position) if (r.ok and r.view is not None) else 0
        if n:
            r.repair = dict(r.repair, foliage=int(n))
            leafy.append(r)
        else:
            clear.append(r)
    return clear + leafy


# `pipeline.stage_render` and `pipeline.stage_finish` reached these four functions
# through `scripts/step4_render.py`, which reached one of them through
# `scripts/e1d_shotlist.py`, which reached one of *those* through
# `scripts/audit_mutants.py` -- so running a round imported three finished experiments,
# and the shot list lived in the scripts directory with the scripts that had used it
# once. They are the geometry the cameras above are pointed at, so they live here with
# the cameras, and the three experiment scripts are gone.

def merged_plots(state: str) -> list:
    """One rectangle per structure, from a settlement's plot registry.

        **What the registry knows about a part travels with its rectangle.** Demo-polish,
        phase 4. This returned four numbers and a label, and threw away the `kind`,
        `passage`, `y0` and `rects` the registry writes down -- so `stage_render`'s
        `p.get("passage")` was never true and 1d's *a gate is framed on its wall* was dead
        code in the only pipeline that runs it, while its own case passed on plot dicts
        built by hand. All three of the demo's gates were photographed as close masonry
        again. A rule proved on a fixture is not wired until something reads it off what
        the pipeline actually holds; the case for that is `test_camera`'s
        `dp4_the_plot_registry_carries_what_the_camera_rules_read`.
        
    """
    out: dict = {}
    for p in json.load(open(os.path.join(state, "plots.json"))):
        r = (min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
             max(p["x0"], p["x1"]), max(p["z0"], p["z1"]))
        prev = out.get(p["label"])
        if prev is None:
            row = dict(p)
            row["rects"] = list(p.get("rects") or [list(r)])
            out[p["label"]] = row | {"x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]}
        else:
            prev["rects"].extend(p.get("rects") or [list(r)])
            prev.update(x0=min(prev["x0"], r[0]), z0=min(prev["z0"], r[1]),
                        x1=max(prev["x1"], r[2]), z1=max(prev["z1"], r[3]))
    return [out[k] for k in sorted(out)]


def surrounding_floor(vol, plot: dict) -> int:
    """The ground level around a plot, in local y: the median surface of a ring just
    outside it. What `built_cells` measures the placed mass *above*, and what `mid_y`
    falls back to where there is no placed mass at all."""
    import numpy as np
    x0, x1 = plot["x0"] - vol.x0, plot["x1"] - vol.x0
    z0, z1 = plot["z0"] - vol.z0, plot["z1"] - vol.z0
    t = vol.tables()
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))[vol.codes]
    ring = []
    for (a, b) in ((x0 - 3, z0 - 3), (x1 + 3, z1 + 3)):
        a = min(max(a, 0), vol.shape[0] - 1)
        b = min(max(b, 0), vol.shape[2] - 1)
        col = solid[a, :, b]
        ring.append(int(np.max(np.nonzero(col)[0])) if col.any() else 0)
    return int(np.median(ring))


def built_cells(vol, plot: dict):
    """The placed mass on a plot: solid cells above the ground surrounding it."""
    import numpy as np
    x0, x1 = plot["x0"] - vol.x0, plot["x1"] - vol.x0
    z0, z1 = plot["z0"] - vol.z0, plot["z1"] - vol.z0
    t = vol.tables()
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))[vol.codes]
    floor = surrounding_floor(vol, plot)
    cells = []
    for lx in range(max(x0, 0), min(x1 + 1, vol.shape[0])):
        for lz in range(max(z0, 0), min(z1 + 1, vol.shape[2])):
            for ly in range(floor, vol.shape[1]):
                if solid[lx, ly, lz]:
                    cells.append((lx, ly, lz))
    return np.array(cells, int) if cells else np.zeros((0, 3), int)


def mid_y(vol, plot: dict) -> float:
    """The height a card is framed at: halfway up the mass standing on this plot.

        Not the ground and not the ridge. A camera aimed at the ground photographs a roof
        plane from below and a camera aimed at the ridge photographs the sky.

        **Where nothing is standing here, halfway up nothing is the ground.** A card is
        always framed on a plot with a building on it, so this could not come up until a
        flythrough asked for the height of the world under each step of a camera move: the
        path from a city's outer gate to its palace crosses open ground, a patch with no
        placed mass on it gave `built_cells` an empty array, and the move died on the first
        such step with `zero-size array to reduction operation minimum`. Found by shooting
        one.
        
    """
    c = built_cells(vol, plot)
    if len(c) == 0:
        return vol.y0 + surrounding_floor(vol, plot)
    return vol.y0 + (int(c[:, 1].min()) + int(c[:, 1].max())) / 2


def cache_built(state: str) -> str:
    """The settlement as it now stands, to `out/<name>/world_built.npz`.

        Read-only, and deliberately never touches `world.npz` or `world_prebuild.npz` --
        those are the fixtures every dry run and every built-mass diff is taken against.
        Needs a live server: it is a read of the world through GDMC-HTTP.
        
    """
    from gdpc.vector_tools import Rect

    from . import observe, offline, world
    s = json.load(open(os.path.join(state, "site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    pad = 16
    ed = world.editor()
    ed.loadWorldSlice(Rect((X - pad, Z - pad), (S + 2 * pad, S + 2 * pad)), cache=True)
    h = ed.worldSlice.heightmaps[world.HEIGHTMAP].astype(int) - 1
    y0, y1 = max(-64, int(h.min()) - 8), int(h.max()) + 24
    vol = observe.Volume.from_world_slice(ed.worldSlice, X - pad, Z - pad,
                                          S + 2 * pad, S + 2 * pad, y0, y1)
    out = os.path.join(state, "world_built.npz")
    offline.save_volume(vol, out)
    print(f"cached {vol.shape} y {y0}..{y1} -> {out}", flush=True)
    return out


def town_aerials(state: str, frames_dir: str, sight: "Sightline | None" = None,
                 spp: int = 40) -> dict:
    """Four settlement aerials, one per quarter, framed on the whole site."""
    s = json.load(open(os.path.join(state, "site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    centre = (X + S / 2, (s["stats"]["min"] + s["stats"]["max"]) / 2, Z + S / 2)
    chunks = chunk_list(X - 16, Z - 16, X + S + 16, Z + S + 16)
    return shoot_all(town_quarter_shots(centre, S, sight=sight), chunks, frames_dir,
                     tag="town", scene_prefix="step4_", spp=spp, size=(1200, 700),
                     reuse=True)


def bearing_shots(centre, span: float, bearings=(45, 225), *,
                  sight: "Sightline | None" = None, dist: float = 1.05,
                  height: float = 0.85, lift: float = 0.12, fov: float = 60,
                  pull: float = 0.72, prefix: str = "") -> dict:
    """A ring of aerials round one subject. E1c's two opposite quarters (a west-face
    defect is invisible from the northeast) and E3's four are the same recipe."""
    out = {}
    d = span * dist
    for b in bearings:
        key = f"{prefix}{b}"
        out[key] = orbit_shot(key, centre, bearing=b, dist=d, rise=d * height,
                              target_lift=span * lift, fov=fov, pull=pull,
                              sight=sight)
    return out


def town_quarter_shots(centre, size: float, *, sight: "Sightline | None" = None,
                       dist: float = 0.95, height: float = 0.62, lift: float = 0.05,
                       fov: float = 70) -> dict:
    """The whole settlement from each quarter. `size` is the site's side."""
    out = {}
    for name, bearing in (("ne", 45), ("se", 135), ("sw", 225), ("nw", 315)):
        out[name] = orbit_shot(name, centre, bearing=bearing, dist=size * dist,
                               rise=size * height, target_lift=size * lift,
                               fov=fov, sight=sight)
    return out


#: The two frames a **place** is photographed with, as against the four a building is. a
#: walled district, from outside it, at the height a person is.
PLACE_SHOTS = ("place_aerial", "place_skyline")

#: How far outside the gate the skyline camera stands, and how high its eye is. Sixty
#: because that is far enough that a 96-block district fills the frame at 70 degrees and
#: near enough that the wall still reads as a wall rather than as a line. See
#: `skyline_back`.
PLACE_SKYLINE_BACK = 60
PLACE_SKYLINE_EYE = 2

#: The height of face `PLACE_SKYLINE_BACK` was chosen against. When the sixty was
#: written the tallest thing any committed type could build was `wall` at twenty, and an
#: eye at `PLACE_SKYLINE_EYE` sees eighteen blocks of it: the top of the wall stands
#: `atan(18/60)`, about seventeen degrees, above the horizon, which at seventy degrees
#: of field leaves the top quarter of the frame for the sky and whatever rises behind
#: it. A camera that keeps that angle keeps that frame whatever it is standing in front
#: of.
PLACE_SKYLINE_RISE = 18

#: How far either side of the gate the height in front of the camera is measured. The
#: gate's own tower and the wall running away from it on both sides -- not the mountain
#: a hundred blocks off, which is the setting and not the thing being stood back from.
PLACE_SKYLINE_SPAN = 16

#: Where the gate's foot sits in the arrival frame **when the stand-back has been
#: scaled** past its floor: at the boundary of the lower third. Demo-polish, 1d (finding
#: 7). `place_card_shots` aims at the site's centre from `gate.y + 2`, and at the floor
#: of sixty that puts a wall of twenty where the validated frame had it; scaled to 173
#: for a wall of forty-eight the same aim puts the wall's foot at mid-frame and the
#: ground the city never touched fills the lower half. A person arriving looks at the
#: wall, not the road under their feet. Gated on `back > PLACE_SKYLINE_BACK`, which no
#: standing town triggers, so every recorded skyline camera is the float it always was.
PLACE_SKYLINE_FOOT = 1.0 / 3.0

#: **The arrival camera climbs too.** Demo-polish, phase 4, and the one rule the render
#: itself found. `skyline_back` scales the stand-back to what stands in front of the
#: camera, and with the gate built to its wall's crown (2b) that is **216 blocks outside
#: the site** -- ground no volume this project holds covers, that no plan touched and
#: that nothing levelled. with `ok: True`, because `Sightline` is built on the site
#: volume and the camera is nowhere in it. It gets the same ladder: the first rung is
#: the validated camera and does not move, so every recorded arrival frame is the bytes
#: it always was, and each rung after it lifts the eye `PLACE_SKYLINE_LIFT` blocks and
#: re-aims from where it now stands. A rung is only ever reached by a frame that would
#: otherwise have been thrown away. Sixteen because it is `MAX_RISE`, the lift the
#: bounded camera search is already allowed inside a volume it can see; three rungs
#: because 2 x 16 clears any bank a `site()` pass would itself have cut. Registered
#: before the frame was re-shot.
PLACE_SKYLINE_LIFT = 16
PLACE_SKYLINE_RUNGS = 3


def skyline_pitch(back: float, fov: float, foot: float = PLACE_SKYLINE_FOOT,
                  eye: float = PLACE_SKYLINE_EYE) -> float:
    """The elevation of the view axis, in radians, that puts a point `eye` blocks
        below the camera and `back` blocks ahead of it `foot` of the way up the frame.

        Chunky's `fov` is the **vertical** field: its pinhole projector maps the frame's
        half-height to tan(fov/2). A point at elevation e over the axis lands at
        0.5 + e / fov of the way up (small angles), so the foot at `foot` is `fov * (foot -
        0.5)` below the axis, and the axis is that far above the line to the foot.
        
    """
    half = math.radians(fov) / 2.0
    below_axis = half * (1.0 - 2.0 * foot)          # the foot, this far under the axis
    to_foot = math.atan2(-float(eye), float(back))   # the line to the foot: under level
    return to_foot + below_axis


def skyline_back(rise: float | None) -> float:
    """How far outside the gate the arrival camera stands, given the height in front.

        `rise` is the height of the tallest thing on the approach above the gate's own
        level. `None` -- nothing measured it -- is the floor, which is what every recorded
        frame in this project was shot at.
        
    """
    if not rise:
        return float(PLACE_SKYLINE_BACK)
    face = float(rise) - PLACE_SKYLINE_EYE
    return max(float(PLACE_SKYLINE_BACK),
               PLACE_SKYLINE_BACK * face / float(PLACE_SKYLINE_RISE))


def approach_rise(built, gate, span: int = PLACE_SKYLINE_SPAN) -> int:
    """The height of the tallest standing column near a gate, above the gate's own level.

        Read off the built volume rather than the plan, because what stands in front of the
        camera is a fact about the world: a wall, its gate tower, and a bank the ground pass
        left. 0 where nothing near the gate rises above it, which is a place with no wall.
        
    """
    if built is None or gate is None:
        return 0
    return edge_rise(built, int(gate.door[0]), int(gate.door[2]), int(gate.y), span)


def place_card_shots(centre, size: float, *, gate=None, bearing: float = 45,
                     sight: "Sightline | None" = None, fov: float = 70,
                     rise: float | None = None) -> dict:
    """The two whole-place frames. A7.

        `centre` is (cx, cy, cz) of the site and `size` its side.

          place_aerial   the whole place from the quarter it is approached from, at 45
                         degrees -- the one view that shows a district *is* a district.
          place_skyline  eye level, `skyline_back(rise)` blocks outside the gate, looking
                         in. This is the arrival: what a person walking up to the place
                         sees before they are in it, which is the only view a town has that
                         a building does not.

        `gate` is a `circulate.Threshold` -- the gate's reserved doorstep -- and the
        skyline camera stands back along its facing. With no gate there is no arrival to
        photograph and the key is absent, which is what the composer's blank panel is for:
        a place with no way in is a fact about the plan, not a missing picture.

        `rise` is the height of what stands in front of the camera, over the gate's own
        level (`approach_rise`); without it the stand-back is the constant it has always
        been.
        
    """
    cx, cy, cz = centre
    out = {}
    d = size * 0.95
    out["place_aerial"] = orbit_shot("place_aerial", centre, bearing=bearing,
                                     dist=d, rise=d * 0.62,
                                     target_lift=size * 0.05, fov=fov, sight=sight)
    if gate is not None:
        dx, dz = FACE[gate.facing]
        door = gate.door
        back = skyline_back(rise)
        # **Outward.** `door_shot`'s convention steps back along a threshold's approach
        # direction, which for a building's front door is away from the building. A gate
        # is a hole in a wall and has a town on one side of it, and on this district
        # that convention put the camera inside the walls among the turrets. So the sign
        # is chosen by the place: whichever of the two stands further from the middle of
        # it is the one looking in.
        opts = [(door[0] - dx * back + 0.5,
                 gate.y + PLACE_SKYLINE_EYE,
                 door[2] - dz * back + 0.5),
                (door[0] + dx * back + 0.5,
                 gate.y + PLACE_SKYLINE_EYE,
                 door[2] + dz * back + 0.5)]
        pos = max(opts, key=lambda p: (p[0] - cx) ** 2 + (p[2] - cz) ** 2)
        # **The arrival camera climbs too**, and the first rung is the validated camera
        # unmoved. See `PLACE_SKYLINE_LIFT`.
        rungs = []
        for n in range(PLACE_SKYLINE_RUNGS):
            p = (pos[0], pos[1] + n * PLACE_SKYLINE_LIFT, pos[2])
            target = (cx, cy + size * 0.05, cz)
            if back > PLACE_SKYLINE_BACK:
                # **The wall gets its third.** The yaw still points at the middle of the
                # place; the pitch is set so the gate's foot sits at
                # `PLACE_SKYLINE_FOOT` of the frame. See the constant. Measured from the
                # rung's own eye, so a lifted camera keeps the same composition.
                dist = math.hypot(cx - p[0], cz - p[2])
                target = (cx, p[1] + dist * math.tan(
                    skyline_pitch(back, fov,
                                  eye=PLACE_SKYLINE_EYE + n * PLACE_SKYLINE_LIFT)), cz)
            rungs.append(aim("place_skyline", p, target, fov, sight))
        out["place_skyline"] = rungs
    return out


def place_chunks(X: int, Z: int, S: int, shots: dict, margin: int = 16) -> list:
    """The chunks the whole-place frames need: the site, plus every camera's own
        stand-off outside it, plus a margin. Demo-polish, 1d (finding 9).

        `stage_render` loaded the site plus sixteen blocks, and the aerial -- which stands
        `0.95 * S` off the centre -- showed the site as a floating slab with sheer sides
        where the world beyond the loaded chunks was nothing. A camera sees what it stands
        over; load to where it stands.
        
    """
    pad = 0.0
    for shot in shots.values():
        for sh in (shot if isinstance(shot, (list, tuple)) else [shot]):
            if not sh.ok or sh.view is None:
                continue
            px, _py, pz = sh.view.position
            pad = max(pad, X - px, px - (X + S), Z - pz, pz - (Z + S))
    pad = int(math.ceil(pad)) + int(margin)
    return chunk_list(X - pad, Z - pad, X + S + pad, Z + S + pad)


def approach_bearing(net, centre) -> float:
    """Which quarter a place is approached from, in degrees clockwise from north.

        The lane cell furthest from the centre is where the town is arrived at -- the road
        in has to end somewhere and that somewhere is the edge of the site -- and the
        aerial stands over that quarter. Rounded to the nearest 45 so the answer is one of
        the eight compass points and two runs of the same town give the same camera.
        
    """
    if net is None or not net.cells:
        return 45.0
    cx, _cy, cz = centre
    fx, fz = max(net.cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cz) ** 2)
    b = math.degrees(math.atan2(fx - cx, -(fz - cz))) % 360
    return float(round(b / 45.0) * 45 % 360)


def fixed_three(centre, span: float, *, sight: "Sightline | None" = None) -> dict:
    """`views_for`, aimed. Overhead orthographic, 45-degree aerial, ground level.

        The aerial and ground cameras stand off in +z rather than -z, which is why they are
        written out here instead of going through `orbit_shot`: no bearing reproduces
        `(-0.62d, +0.62d)` in floating point, and a camera that lands on a different float
        is a camera that moved.
        
    """
    cx, cy, cz = centre
    d = span * 1.1
    aerial_pos = (cx - d * 0.62, cy + d * 0.88, cz + d * 0.62)
    ground_pos = (cx - d * 1.15, cy + 2, cz + d * 1.15)
    return {
        "overhead": overhead_shot("overhead", centre, span),
        "aerial45": aim("aerial45", aerial_pos, (cx, cy + span * 0.15, cz), 60, sight),
        "ground": aim("ground", ground_pos, (cx, cy + span * 0.3, cz), 75, sight),
    }


# ------------------------------------------------------------------ frame checks

def frame_mean(png: str) -> float:
    """The mean pixel of a rendered frame. -1.0 if it cannot be read at all."""
    import cv2
    im = cv2.imread(png)
    return -1.0 if im is None else float(im.mean())


def frame_dim(mean: float | None, threshold: float = DIM_MEAN) -> bool:
    """Above the black bar and still in deep shade. See `DIM_MEAN`."""
    return mean is not None and BLACK_MEAN <= mean < threshold


def frame_black(png: str, threshold: float = BLACK_MEAN) -> bool:
    """Did this frame come back as rock? A missing frame counts as black."""
    return frame_mean(png) < threshold


def check_frames(paths: dict, threshold: float = BLACK_MEAN) -> dict:
    """{key: png} -> {"ok": bool, "black": [key], "means": {key: mean}}.

        The thing a card composer is required to call before it composes.
        
    """
    means = {k: frame_mean(p) for k, p in paths.items()}
    black = sorted(k for k, m in means.items() if m < threshold)
    return {"ok": not black, "black": black,
            "means": {k: round(v, 2) for k, v in means.items()}}


def scene_json(name: str, view: View, chunks: list[list[int]], width: int, height: int,
               spp: int, world_path: str = WORLD, yclip: tuple[int, int] = (-64, 320)) -> dict:
    return {
        "sdfVersion": 9,
        "name": name,
        "width": width,
        "height": height,
        "exposure": 1.0,
        "postprocess": "GAMMA",
        "outputMode": "PNG",
        "renderTime": 0,
        "spp": 0,
        "sppTarget": spp,
        "rayDepth": 5,
        "pathTrace": True,
        "dumpFrequency": 0,
        "saveSnapshots": False,
        "emittersEnabled": True,
        "emitterIntensity": 13.0,
        "sunEnabled": True,
        "stillWater": True,
        "waterOpacity": 0.42,
        "waterVisibility": 9.0,
        "useCustomWaterColor": False,
        "fogDensity": 0.0,
        "fastFog": True,
        "biomeColorsEnabled": True,
        "transparentSky": False,
        "renderActors": False,
        # Chunky's headless defaults are the pre-1.18 world bounds (0..256). On a 1.21
        # world that clips everything above y=0, so the render is the cut face of the
        # stone layer. These must be set explicitly.
        "yMin": -64, "yMax": 320, "yClipMin": yclip[0], "yClipMax": yclip[1],
        "world": {"path": world_path, "dimension": 0},
        "camera": {
            "name": "camera 1",
            "position": {"x": view.position[0], "y": view.position[1], "z": view.position[2]},
            "orientation": {"roll": 0.0, "pitch": view.pitch, "yaw": view.yaw},
            "projectionMode": view.projection,
            "fov": view.fov,
            "dof": "Infinity",
            "focalOffset": 2.0,
        },
        "sun": {"altitude": 0.9, "azimuth": 2.2, "intensity": 1.25, "color": {"red": 1.0, "green": 1.0, "blue": 1.0},
                "drawTexture": True, "radius": 0.03},
        "sky": {"skyYaw": 0.0, "skyMirrored": True, "skyLight": 1.0, "mode": "SIMULATED",
                "horizonOffset": 0.1, "cloudsEnabled": False},
        "cameraPresets": {},
        "chunkList": chunks,
        "entities": [],
        "actors": [],
        "materials": {},
    }


def finished_chunks(chunks: list, regions=None) -> tuple:
    """`(kept, dropped)`: the chunks of `chunks` the world save holds as finished ground,
    and the rest. A frame over a chunk the world never made is a grey slab, and the
    arrival camera of the concentric run stood 216 blocks out over exactly that.
    `regions` is an `ethoslm.regions.Regions`; None reads the save.
    """
    from . import regions as _regions
    regs = regions if regions is not None else _regions.Regions()
    kept, dropped = [], []
    cache: dict = {}
    for cx, cz in chunks:
        key = (int(cx), int(cz))
        if key not in cache:
            try:
                regs.chunk(*key)
                cache[key] = True
            except _regions.NotGenerated:
                cache[key] = False
            except Exception:                    # noqa: BLE001 -- unreadable is unmade
                cache[key] = False
        (kept if cache[key] else dropped).append([key[0], key[1]])
    if regions is None:
        regs.close()
    return kept, dropped


def onto_finished(shot: "Shot", finished: set, step: int = 8) -> "Shot":
    """`shot` dollied in along its line of sight until it stands over a chunk in
    `finished` (a set of `(cx, cz)`), marked; or the shot itself where it already
    does or nothing on the line does."""
    if not shot.ok or shot.view is None:
        return shot
    px, py, pz = shot.view.position
    tx, ty, tz = shot.target
    if (int(math.floor(px)) >> 4, int(math.floor(pz)) >> 4) in finished:
        return shot
    dx, dz = tx - px, tz - pz
    n = math.hypot(dx, dz)
    if n < 1e-9:
        return shot
    ux, uz = dx / n, dz / n
    d = float(step)
    while d < n:
        qx, qz = px + ux * d, pz + uz * d
        if (int(math.floor(qx)) >> 4, int(math.floor(qz)) >> 4) in finished:
            moved = aim(shot.name, (qx, py, qz), shot.target, shot.view.fov, None)
            moved.repair = dict(shot.repair, dollied_onto_finished_ground=round(d, 1))
            return moved
        d += step
    return shot


def chunk_list(x0: int, z0: int, x1: int, z1: int, pad: int = 1) -> list[list[int]]:
    cx0, cz0 = x0 // 16 - pad, z0 // 16 - pad
    cx1, cz1 = x1 // 16 + pad, z1 // 16 + pad
    return [[cx, cz] for cx in range(cx0, cx1 + 1) for cz in range(cz0, cz1 + 1)]


def _java(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    libs = os.path.join(CHUNKY, "lib", "*")
    # javax.imageio needs a writable temp dir to decode textures; /tmp is read-only in
    # this sandbox and the failure is silent apart from a font-texture warning.
    tmp = os.environ.get("TMPDIR", "/tmp")
    cmd = ([LOADER] if LOADER else []) + [JAVA, "-Dchunky.home=" + CHUNKY_HOME, "-Djava.awt.headless=true",
           "-Djava.io.tmpdir=" + tmp, "-Xmx8G",
           "-cp", libs, "se.llbit.chunky.main.Chunky"] + args
    env = dict(os.environ)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def frame_margin(span: float) -> int:
    """Half-side of the chunk box round a subject of size `span`.

        The margin has to stop growing with span or a settlement-scale render asks for
        thousands of chunks and the JVM dies.
        
    """
    return int(min(span * 2.2, span * 1.15 + 60))


def shoot(shot: Shot, chunks: list, png: str, *, scene: str | None = None,
          spp: int = 40, size: tuple[int, int] = (900, 560), threads: int = 14,
          world_path: str = WORLD, yclip: tuple[int, int] = (-64, 320),
          reuse: bool = False, tag: str = "", record_kind: str | None = "frame") -> dict:
    """Render one checked shot to `png`, and say what came back.

        **The only place in this project that writes a Chunky scene and calls the JVM.** It
        was six places, each with its own idea of what to do when the frame came back wrong,
        which is how `render_views` came to record `failed: []` for twenty-four shots while
        two of its frames were a door leaf filling the image edge to edge.

        The returned record always carries a `status`, and there are only four:

            placement       no camera could be placed; nothing was rendered
            render_failed   Chunky was run and produced no file
            black           a frame came back below `BLACK_MEAN` -- rock, not a building
            ok              a frame, with its mean pixel measured

        `reuse` keeps a frame already on disk when its camera did not move and it is not
        black. That is what makes a re-run cheap and, more importantly, what keeps a
        standing fixture comparable with itself: the panels that were always fine are the
        same bytes.

        **And "its camera did not move" is now read and not asserted.** Demo-polish, phase
        4. Nothing compared the camera: a frame on disk that was not black and not dim was
        kept whatever was asked for. So when 1d's *a gate is framed on its wall* was
        finally wired up and every gate got a new camera on a span four times its own, the
        render printed `kept` twelve times and would have shipped the old close-masonry
        panels again. Chunky is a pure function of the scene, so the scene this call asked
        for is written beside the frame as `<png>.scene.json` and a reuse is allowed only
        when the scene it would ask for now is that one, byte for byte -- which also
        catches a changed chunk list, a changed sample count and a changed frame size.

        A frame with **no** sidecar is reused as before and the record says
        `camera_checked: false`. Every cached frame in this project predates the sidecar,
        and throwing them all away would cost more than it is worth and break the
        byte-identity the standing fixtures are for; what it may not do is claim they were
        checked.
        
    """
    rec = shot.as_dict()
    rec["png"] = png
    name = scene or (f"{tag}_{shot.name}" if tag else shot.name)
    want = (scene_json(name, shot.view, chunks, size[0], size[1], spp,
                       world_path, yclip) if shot.ok else None)
    side = png + ".scene.json"
    if reuse and shot.ok and not shot.repair and os.path.exists(png) \
            and not frame_black(png) and not frame_dim(frame_mean(png)):
        had = os.path.exists(side)
        same = True
        if had:
            try:
                same = json.load(open(side)) == want
            except (ValueError, OSError):
                same = False
        if same:
            rec.update(status="ok", mean=round(frame_mean(png), 2), reused=True,
                       camera_checked=had, seconds=0.0)
            return rec
    if not shot.ok:
        rec.update(status="placement", mean=None, seconds=0.0)
        if record_kind:
            from .measure import record
            record(record_kind, tag=tag, shot=shot.name, status="placement")
        return rec

    sdir = os.path.join(SCENES, name)
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, name + ".json"), "w") as f:
        json.dump(want, f, indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    t0 = time.perf_counter()
    # -f: the scene has no octree dump on first load; -reload-chunks builds it
    p = _java(["-texture", MC_JAR, "-render", name, "-f",
               "-target", str(spp), "-threads", str(threads), "-reload-chunks"])
    snap = _java(["-snapshot", name, png])
    secs = round(time.perf_counter() - t0, 1)
    if not os.path.exists(png):
        rec.update(status="render_failed", mean=None, seconds=secs,
                   stderr=(p.stderr or "")[-600:],
                   snapshot_err=(snap.stderr or "")[-400:])
        if record_kind:
            from .measure import record
            record(record_kind, tag=tag, shot=shot.name, status="render_failed")
        return rec
    # the camera this frame *is*, beside the frame, so the next run's `reuse` can read
    # it instead of assuming it
    with open(side, "w") as f:
        json.dump(want, f, indent=1)
    mean = frame_mean(png)
    rec.update(status="black" if mean < BLACK_MEAN else "ok", mean=round(mean, 2),
               seconds=secs, chunks=len(chunks), spp=spp)
    if record_kind:
        from .measure import record
        record(record_kind, tag=tag, shot=shot.name, seconds=secs,
               mean=round(mean, 2), status=rec["status"])
    return rec


def shoot_all(shots: dict, chunks: list, out_dir: str, *, tag: str = "",
              png_prefix: str | None = None, scene_prefix: str = "", **kw) -> dict:
    """`shoot` over a shot list. Frames land at `<out_dir>/<png_prefix><key>.png`,
        scenes are named `<scene_prefix><png_prefix><key>`. `png_prefix` defaults to the
        tag, which is what a per-building shot list wants; `render()` passes "" because its
        frames have always been `overhead.png` and not `<tag>_overhead.png`.

        A value may be **a list of shots** instead of one: the next camera is tried only
        when the one before it comes back unplaceable, black or not rendered at all, and the
        first frame that is actually a frame is the one kept. `tried` on the record says how
        far down the list it had to go, and it is 1 for every shot that has ever worked --
        so a card made of first-choice cameras is the same bytes it always was.
        
    """
    out = {}
    pre = f"{tag}_" if png_prefix is None and tag else (png_prefix or "")
    for key, shot in shots.items():
        stem = f"{pre}{key}"
        png = os.path.join(out_dir, f"{stem}.png")
        ladder = list(shot) if isinstance(shot, (list, tuple)) else [shot]
        rec = None
        # **Dim is a rung.** Demo-polish, 1d: a frame in deep shade climbs the ladder
        # the way a black one does. The rungs share one file, so a dim frame is set
        # aside before the next rung overwrites it, and if every rung is dim the
        # brightest is put back and the record says so by name.
        dim_rungs: list = []
        for n, s in enumerate(ladder, 1):
            rec = shoot(s, chunks, png, scene=f"{scene_prefix}{stem}", tag=tag, **kw)
            rec["tried"] = n
            note = f"  [{s.repair}]" if s.repair else ""
            dim = rec["status"] == "ok" and frame_dim(rec.get("mean"))
            if rec["status"] == "placement":
                print(f"  {stem}: SHOT FAILED -- {s.reason}", flush=True)
            elif rec["status"] == "ok" and rec.get("reused"):
                print(f"  {stem}: kept  mean {rec['mean']:.1f}", flush=True)
            else:
                flag = ("  BLACK" if rec["status"] == "black"
                        else "  DIM" if dim else "")
                m = "n/a" if rec["mean"] is None else f"{rec['mean']:.1f}"
                print(f"  {stem}: {rec['seconds']:.0f}s  mean {m}{flag}{note}"
                      + (f"  (camera {n} of {len(ladder)})" if n > 1 else ""),
                      flush=True)
            if rec["status"] == "ok" and not dim:
                if dim_rungs:
                    rec["dim_repaired"] = [r["mean"] for r, _f in dim_rungs]
                break
            if dim and n < len(ladder) and os.path.exists(png):
                aside = f"{png}.rung{n}"
                shutil.copyfile(png, aside)
                dim_rungs.append((rec, aside))
            elif dim:
                dim_rungs.append((rec, None))
            if n < len(ladder):
                print(f"  {stem}: {'dim' if dim else rec['status']} -- standing "
                      f"further back and trying camera {n + 1} of {len(ladder)}",
                      flush=True)
        if rec is not None and dim_rungs and \
                (rec["status"] != "ok" or frame_dim(rec.get("mean"))):
            # every rung was dim (or worse): keep the brightest dim frame, and say so
            best, aside = max(dim_rungs, key=lambda t: t[0]["mean"])
            if aside is not None and os.path.exists(aside):
                shutil.copyfile(aside, png)
            rec = dict(best, dim_kept=True,
                       dim_ladder=[r["mean"] for r, _f in dim_rungs])
        for _r, aside in dim_rungs:
            if aside is not None and os.path.exists(aside):
                os.remove(aside)
        out[key] = rec
    return out


#: A4: how high over the ground a flythrough camera flies, and its field of view. Low
#: enough that the walls are the horizon and not a pattern seen from a plane, which is
#: the whole reason a city gets a camera move and a town got two stills.
FLY_RISE = 26
FLY_FOV = 80

#: **The flythrough stops short of its subject.** Demo-polish, 1d (finding 8). The path
#: ended over the subject's centre, so the last camera looked straight down into the
#: palace court and the throne hall's roof was the last thing a viewer saw. A camera
#: move to a thing stops short of it: the path ends `FLY_STANDOFF` blocks before the
#: subject's near edge -- its half-span along the approach, plus this -- with the look
#: held on the centre, and the final camera's pitch below `FLY_FINAL_PITCH_MAX` degrees
#: under level. Registered before the path was recomputed.
FLY_STANDOFF = 24
FLY_FINAL_PITCH_MAX = 40.0

#: The path stops `FLY_STANDOFF` short of the subject's near edge as the rule says, and
#: when the subject is a *walled* compound its wall stands on that edge: the demo's last
#: eight frames were the outside of the palace ring wall from nineteen blocks and the
#: throne hall, which is what the move is for, was never seen. So the tallest column
#: between the final camera and the look point -- stopping `FLY_STANDOFF` short of the
#: look point, because the last stretch is the subject itself -- is read off the world,
#: the final camera stands `FLY_RISE` above it as it stands `FLY_RISE` above the ground
#: everywhere else, and the move eases up to that over its last `FLY_LIFT_FRAMES`
#: frames. A move over open ground is the move it was. Registered before the path was
#: reshot.
FLY_LIFT_FRAMES = 8
FLY_CORRIDOR = 2

#: The lift above cleared the last thing before the subject and nothing else: the
#: concentric run's frames 14-20 flew at y 91-94 into a ring wall whose crown stood at
#: 99. Now the crown height along the path is read between every pair of steps, each
#: step stands `FLY_RISE` above the tallest thing within the next `FLY_LIFT_FRAMES`
#: steps (lifting ahead of each wall), and comes down no faster than `FLY_SETTLE` a
#: frame after it (settling). Open ground reads no crown and the move is the move it
#: was.
FLY_SETTLE = 3

#: How far inside a **compound's** rectangle the corridor reaches: a compound's own wall
#: stands at least `placeplan.COMPOUND_WALL_INSET` (3) inside its edge and is up to five
#: wide, so the wall is between the camera and the subject and the halls behind it are
#: the subject. A plot or an area at the centre has no wall of its own and the corridor
#: stops at its edge.
FLY_INTO_COMPOUND = 8


def _enters(a, b, rect) -> float | None:
    """The distance along `a`->`b` (both `(x, z)`) at which the line first enters the
    axis-aligned `rect` (x0, z0, x1, z1), or None where it never does."""
    dx, dz = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    n = math.hypot(dx, dz)
    if n < 1e-9:
        return None
    t0, t1 = 0.0, n
    for p0, d, lo, hi in ((float(a[0]), dx / n, rect[0], rect[2] + 1),
                          (float(a[1]), dz / n, rect[1], rect[3] + 1)):
        if abs(d) < 1e-12:
            if not (lo <= p0 < hi):
                return None
            continue
        ta, tb = (lo - p0) / d, (hi - p0) / d
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return t0 if t0 <= t1 else None


def corridor_end(step: dict) -> float | None:
    """How far along a step's line of sight the obstacle corridor reaches: to where the
    line enters the subject's rectangle, plus `FLY_INTO_COMPOUND` for a compound; None
    where the step names no subject (the caller falls back to `FLY_STANDOFF` short of
    the look point)."""
    rect = step.get("subject")
    if not rect:
        return None
    t = _enters(step["at"], step["look"], rect)
    if t is None:
        return None
    return t + (FLY_INTO_COMPOUND if step.get("compound") else 0.0)


def column_tops(vol):
    """The highest non-air y of every column of `vol`, `vol.y0 - 1` where none."""
    import numpy as np
    names = [str(n).split("[")[0] for n in vol.palette]
    air = np.array([n in ("air", "cave_air", "void_air") for n in names], bool)
    solid = ~air[vol.codes]
    idx = solid.shape[1] - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    return np.where(solid.any(axis=1), idx + vol.y0, vol.y0 - 1)


def tallest_between(vol, a, b, *, stop: float | None = None,
                    margin: float = FLY_STANDOFF, corridor: int = FLY_CORRIDOR,
                    tops=None):
    """The highest column top on the line from `a` to `b` (both `(x, z)`), sampled out
    to `stop` blocks along it -- `margin` short of `b` where no stop is given -- over
    a corridor `corridor` columns either side. None where there is nothing to sample
    or nothing on the line is inside the volume."""
    tops = column_tops(vol) if tops is None else tops
    dx, dz = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    n = math.hypot(dx, dz)
    reach = min(n, float(stop)) if stop is not None else n - margin
    if reach <= 0:
        return None
    ux, uz = dx / n, dz / n
    best = None
    for i in range(int(reach) + 1):
        ix = int(math.floor(a[0] + ux * i)) - vol.x0
        iz = int(math.floor(a[1] + uz * i)) - vol.z0
        for ox in range(-corridor, corridor + 1):
            for oz in range(-corridor, corridor + 1):
                px, pz = ix + ox, iz + oz
                if 0 <= px < tops.shape[0] and 0 <= pz < tops.shape[1]:
                    v = int(tops[px, pz])
                    if v >= vol.y0 and (best is None or v > best):
                        best = v
    return best


def flythrough_shots(path: list, vol, *, sight: "Sightline | None" = None,
                     rise: float = FLY_RISE, fov: float = FLY_FOV) -> dict:
    """One camera per step of `path`, each looking at where the path ends. A4.

        `path` is `stages_media.flythrough_path`'s answer -- `{"i", "at": [x, z],
        "look": [x, z]}` -- and the y of every camera is taken off the world under it, so
        the move follows the ground up to the palace rather than through it. The final
        camera stands above the tallest thing between it and the look point, and the last
        `FLY_LIFT_FRAMES` frames ease up to it; see `FLY_LIFT_FRAMES`.
        
    """
    steps = list(path)
    ys = []
    for step in steps:
        x, z = float(step["at"][0]), float(step["at"][1])
        ys.append(mid_y(vol, {"x0": int(x) - 4, "z0": int(z) - 4,
                              "x1": int(x) + 4, "z1": int(z) + 4}))
    crossed = crown_lift(ys, steps, vol)
    ys = crossed["ys"]
    lift = {}
    if steps:
        last = steps[-1]
        top = tallest_between(vol, (float(last["at"][0]), float(last["at"][1])),
                              (float(last["look"][0]), float(last["look"][1])),
                              stop=corridor_end(last))
        if top is not None and top + rise > ys[-1] + rise:
            dy = float(top - ys[-1])
            n = min(FLY_LIFT_FRAMES, len(steps))
            for j in range(n):
                k = len(steps) - n + j
                ys[k] = ys[k] + dy * (j + 1) / float(n)
            lift = {"tallest": int(top), "lift": round(dy, 1), "frames": n}
            # **...and stands back as far as the pitch cap requires.** A camera lifted
            # above a wall close to a small subject would look down at it past
            # `FLY_FINAL_PITCH_MAX`; the stand-off is then measured from the height of
            # the thing it had to clear: the final camera moves out along the approach
            # until its pitch is under the cap, and the eased frames move with it.
            lx, lz = float(last["look"][0]), float(last["look"][1])
            ly = mid_y(vol, {"x0": int(lx) - 8, "z0": int(lz) - 8,
                             "x1": int(lx) + 8, "z1": int(lz) + 8})
            drop = (ys[-1] + rise) - (ly + 6)
            ax, az = float(last["at"][0]), float(last["at"][1])
            have = math.hypot(lx - ax, lz - az)
            need = drop / math.tan(math.radians(FLY_FINAL_PITCH_MAX)) + 0.5
            if drop > 0 and need > have and have > 1e-9:
                ux, uz = (lx - ax) / have, (lz - az) / have
                new_last = (lx - ux * need, lz - uz * need)
                base = steps[len(steps) - n - 1] if len(steps) > n else None
                bx, bz = ((float(base["at"][0]), float(base["at"][1])) if base
                          else (ax, az))
                steps = [dict(st) for st in steps]
                for j in range(n):
                    k = len(steps) - n + j
                    f = (j + 1) / float(n)
                    steps[k]["at"] = [round(bx + (new_last[0] - bx) * f, 2),
                                      round(bz + (new_last[1] - bz) * f, 2)]
                lift["dolly"] = round(need - have, 1)
    out = {}
    for step, y in zip(steps, ys):
        x, z = float(step["at"][0]), float(step["at"][1])
        lx, lz = float(step["look"][0]), float(step["look"][1])
        ly = mid_y(vol, {"x0": int(lx) - 8, "z0": int(lz) - 8,
                         "x1": int(lx) + 8, "z1": int(lz) + 8})
        shot = aim(
            f"fly_{int(step['i']):03d}", (x + 0.5, y + rise, z + 0.5),
            (lx + 0.5, ly + 6, lz + 0.5), fov, sight,
            # A camera move is a *move*: orbiting one frame of it to find air would put
            # a jump in the middle of the shot. It rises and stands back or it is
            # reported as unplaceable and the frame is dropped.
            max_orbit=0)
        if lift and step is steps[-1]:
            shot.repair = dict(shot.repair, **lift)
        k = int(step["i"])
        if crossed["lifted"].get(k):
            shot.repair = dict(shot.repair, crown=crossed["lifted"][k])
        out[f"fly_{int(step['i']):03d}"] = shot
    return out


def crown_lift(ys: list, steps: list, vol, *, ahead: int = FLY_LIFT_FRAMES,
               settle: float = FLY_SETTLE) -> dict:
    """The ground heights `ys` of a camera path, lifted to clear every crown it
        crosses. See `FLY_SETTLE`.

        Between each pair of steps the tallest column in the corridor is read; a step's
        floor is the tallest crown within the next `ahead` steps, so the move is already
        up when it reaches a wall; and after a wall the floor falls no faster than
        `settle` a frame. A step over open ground whose crowns are all under its own
        ground height is untouched, so a move that crosses nothing is the move it was.
        Returns `{"ys": [...], "lifted": {i: {...}}}`.
        
    """
    n = len(steps)
    if n < 2:
        return {"ys": list(ys), "lifted": {}}
    tops = column_tops(vol)
    crown = []
    for i in range(n - 1):
        a = (float(steps[i]["at"][0]), float(steps[i]["at"][1]))
        b = (float(steps[i + 1]["at"][0]), float(steps[i + 1]["at"][1]))
        t = tallest_between(vol, a, b, stop=math.hypot(b[0] - a[0], b[1] - a[1]),
                            margin=0.0, tops=tops)
        crown.append(t)
    out = list(ys)
    lifted = {}
    for i in range(n):
        ahead_crowns = [c for c in crown[i:i + ahead + 1] if c is not None]
        if ahead_crowns:
            top = max(ahead_crowns)
            if top > out[i]:
                lifted[i] = {"crown": int(top), "lift": round(float(top - ys[i]), 1)}
                out[i] = float(top)
    for i in range(1, n):
        if out[i] < out[i - 1] - settle:
            out[i] = out[i - 1] - settle
            lifted.setdefault(i, {})["settling"] = True
    return {"ys": out, "lifted": lifted}


def shoot_flythrough(path: list, vol, out_dir: str, *, tag: str = "fly",
                     scene_prefix: str = "", sight: "Sightline | None" = None,
                     skip: bool = False, **kw) -> dict:
    """The flythrough, as frames. `skip` writes the cameras and renders nothing.

        The path and the cameras are arithmetic over the plan and cost nothing; the frames
        are thirteen seconds apiece. So a round that wants the record without the hour asks
        for `skip` and still has, on disk, exactly which cameras it did not shoot.
        
    """
    shots = flythrough_shots(path, vol, sight=sight)
    if skip:
        return {k: {"status": "not_rendered", "mean": None, "seconds": 0.0,
                    "shot": s.as_dict(),
                    "why": "the flythrough cameras are written; the frames are not "
                           "rendered in this run"}
                for k, s in shots.items()}
    xs = [p["at"][0] for p in path] + [p["look"][0] for p in path]
    zs = [p["at"][1] for p in path] + [p["look"][1] for p in path]
    chunks = chunk_list(int(min(xs)) - 32, int(min(zs)) - 32,
                        int(max(xs)) + 32, int(max(zs)) + 32)
    chunks, _dropped = finished_chunks(chunks)
    return shoot_all(shots, chunks, out_dir, tag=tag, scene_prefix=scene_prefix, **kw)


def shot_summary(report: dict) -> dict:
    """Roll a {tag: {key: rec}} shot report up into the counts a caller reports.

        `dim` is above the black bar and still in deep shade. It was reported and never
        enforced; since demo-polish (1d, finding 3) a dim frame climbs the ladder the way a
        black one does, so what is reported here is which panels were **repaired** by a
        further rung and which were **kept** dim because every rung was.
        
    """
    shots = [(t, k, r) for t, ks in report.items() for k, r in ks.items()]
    return {
        "panels": len(shots),
        "black": [f"{t}_{k}" for t, k, r in shots if r["status"] == "black"],
        "unplaceable": [f"{t}_{k}" for t, k, r in shots if r["status"] == "placement"],
        "render_failed": [f"{t}_{k}" for t, k, r in shots
                          if r["status"] == "render_failed"],
        "repaired": [f"{t}_{k}" for t, k, r in shots if r.get("repair")],
        "dim_repaired": sorted((f"{t}_{k}", r["dim_repaired"], r["mean"])
                               for t, k, r in shots if r.get("dim_repaired")),
        "dim_kept": sorted((f"{t}_{k}", r["mean"]) for t, k, r in shots
                           if r.get("dim_kept")),
    }


def render(tag: str, centre: tuple[float, float, float], span: float,
           bounds: tuple[int, int, int, int] | None = None, spp: int = 64,
           size: tuple[int, int] = (1000, 640), threads: int = 14,
           out_dir: str | None = None, world_path: str = WORLD,
           sight: "Sightline | None" = None) -> dict:
    """Render the three fixed views. Returns {view: {png, seconds, ...}}."""
    out_dir = out_dir or os.path.join(ROOT, "out", tag)
    os.makedirs(out_dir, exist_ok=True)
    if bounds is None:
        m = frame_margin(span)
        bounds = (int(centre[0]) - m, int(centre[2]) - m,
                  int(centre[0]) + m, int(centre[2]) + m)
    chunks = chunk_list(*bounds)
    return shoot_all(fixed_three(centre, span, sight=sight), chunks, out_dir,
                     tag=tag, png_prefix="", scene_prefix=f"{tag}_",
                     spp=spp, size=size, threads=threads,
                     world_path=world_path, record_kind="frame")
