"""The library the model writes against.

Deliberately thin. Four primitives and nothing else, so that what the model reaches
for and cannot find is readable as a design signal. Do not grow this before looking
at the transcripts.
"""
from __future__ import annotations

import zlib

from gdpc import Block
from gdpc.interface import placeBlocks

from . import world
from .prims import Primitives

MAX_BLOCKS = 2_000_000  # guard against a runaway loop filling the world, per 512 of site


def max_blocks_for(footprint: int | None) -> int:
    """`MAX_BLOCKS` scaled by the area of the site a builder works on: the guard is against
    a runaway loop, and a runaway on a 768 site is a runaway at 2.25 times the blocks.
    None is 512.
    """
    side = float(footprint or 512)
    return int(MAX_BLOCKS * max(1.0, (side / 512.0) ** 2))

#: The three ids that mean "nothing here". Placing one is a decision to clear, which is
#: why `approach()` may cross a program's own air and may cross nothing else it wrote.
AIR = ("air", "cave_air", "void_air")

#: Which way each facing goes, in (dx, dz). The same mapping `observe.CODE_DIR` uses,
#: written here so a flight's ascent direction and the direction the walk model expects
#: a tread to face are one fact rather than two.
_FACE_DIR = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


class BuildError(RuntimeError):
    pass


# A builder counting "materials" would not count stone_brick_stairs and mossy stone
# bricks as two materials. Collapse variants and shapes to the underlying material so
# the palette size can be compared against the 1-4 craft rule honestly.
_PREFIX = ("mossy_", "cracked_", "chiseled_", "polished_", "smooth_", "stripped_",
           "cut_", "dark_", "light_")
_SUFFIX = ("_stairs", "_slab", "_wall", "_fence", "_fence_gate", "_door", "_trapdoor",
           "_button", "_pressure_plate", "_pane", "_bricks", "_brick", "_block",
           "_log", "_planks", "_sign", "_hanging_sign", "_carpet", "_bed", "_tiles",
           "_tile")


def _families(palette: dict) -> set:
    """The material families a palette is made of, one name per material."""
    out = set()
    for k in palette:
        if k in ("air", "cave_air", "water", "lava"):
            continue
        changed = True
        while changed:
            changed = False
            for s in sorted(_SUFFIX, key=len, reverse=True):
                if k.endswith(s) and len(k) > len(s):
                    k = k[: -len(s)]
                    changed = True
                    break
        # Prefixes stack too: `chiseled_polished_blackstone` is one strip from
        # `polished_blackstone`, which is a family name no palette ever declares.
        changed = True
        while changed:
            changed = False
            for p in _PREFIX:
                if k.startswith(p) and len(k) > len(p):
                    k = k[len(p):]
                    changed = True
                    break
        out.add(k or "stone")
    return out


def _top_face(state: str) -> int:
    """Where this block's top face sits inside its own cell, in half-blocks.

        0 is nothing you can stand on, 1 a slab, 2 a full cube. The absolute half-height a
        walker stands at on a block in cell y is `2 * y + _top_face(state)` -- the same
        arithmetic `observe.Nav` builds its standing array from.
        
    """
    if not state:
        return 0
    from . import observe
    name = state.split("[")[0]
    _lower, _upper, surface, _rule = observe._classify(name, observe.parse_props(state))
    return int(surface)


def _occupies(state: str) -> bool:
    """Does this block get in the way of standing in, or walking through, its cell?

        Deferred to `observe._classify` for the same reason `prims._is_surface` is: the
        rule an `approach()` lays a path by and the rule the walk model judges that path by
        have to be one rule, or the library is measuring the world differently from the
        thing that reads the world back.
        
    """
    if not state:
        return False
    from . import observe
    name = state.split("[")[0]
    lower, upper, _surface, _rule = observe._classify(name, observe.parse_props(state))
    return bool(lower or upper)


def _material(name: str):
    from .prims import material
    return material(name)


def _solid(name: str) -> str:
    """The one cube a role needs, family or block id. See `prims.solid`."""
    from .prims import solid
    return solid(name)


#: Where joinery comes from, in the order it is looked for. A door, a fence, a trapdoor
#: or a gate is a *timber* thing and only the wood families have one, so the roles are
#: tried frame first -- which is the timber of the voice by definition -- and the
#: shell's own default is the last word.
_JOINERY_ROLES = ("frame", "trim", "floor", "wall", "footing", "roof")

#: What each joinery shape falls back to when no role in the voice has one. Oak, because
#: that is what `doorway()` has silently placed on every building this project has ever
#: made, and naming it here is the first time it has been a decision.
_JOINERY_DEFAULT = {"door": "oak_door", "fence": "oak_fence",
                    "trapdoor": "oak_trapdoor", "gate": "oak_fence_gate",
                    "button": "oak_button"}


def joinery(mat, kind: str = "door") -> str:
    """The door, fence, trapdoor or gate of a voice.

        `mat` is a palette -- the roles dict a part carries -- or a single family. A stone
        voice has no door of its own and the game has none to give it, so this is the one
        place a *fallback* is right rather than a silent substitution: what is being chosen
        is which timber the joinery is, and there is always a timber somewhere in a palette
        or, failing that, oak.

        Deterministic in the palette, so two runs of the same voice hang the same door.
        
    """
    from .prims import shape
    roles = _mat_roles(mat) if not isinstance(mat, str) else {"frame": mat}
    for role in _JOINERY_ROLES:
        fam = roles.get(role)
        if not fam:
            continue
        try:
            return shape(fam, kind)
        except ValueError:
            continue
    if kind not in _JOINERY_DEFAULT:
        raise ValueError(f"no role of this voice has a {kind} and there is no default "
                         f"for one: {sorted(_JOINERY_DEFAULT)}")
    return _JOINERY_DEFAULT[kind]


def _wall_block(fam: str) -> str:
    """The wall (fence-height masonry) block of a material family.

        Read off `circulate.WALLS`, which is the same table the lane pass rails a drop
        with, so a yard wall and the kerb outside it are the same block.
        
    """
    from .circulate import WALLS
    return WALLS.get(fam, "cobblestone_wall")


#: What a building is made of, by role. A voice declares these; `building()` fills in
#: whatever a caller leaves out from the wall, because a building all of one material is
#: a plain building and a building missing a role is a crash.
_MAT_ROLES = ("wall", "roof", "footing", "frame", "floor", "trim")

#: **What a voice may say its ground is**, and the one role that is not a building's. A
#: voice names the six materials a building is made of; the ground between the buildings
#: belonged to nobody, so a terrace took its cover from whatever it could sample and a
#: plateau laid its footing and left it. `ground` is optional and outside `_MAT_ROLES`
#: on purpose: a role that defaults to the wall is a decision nobody made, and a voice
#: that does not name a ground gets **the setting's** -- the plain it stands on -- which
#: is the right answer and not a default.
_GROUND_ROLE = "ground"


def _ground_role(mat) -> str | None:
    """The voice's `ground`, where it names one, off whatever shape `mat` arrives in."""
    if isinstance(mat, dict):
        got = mat.get(_GROUND_ROLE)
        return str(got) if got else None
    return None


#: Optional, like `ground`, and outside `_MAT_ROLES` for the same reason: a voice that
#: does not name one is not defaulting to anything, it is a voice of one stone. Where it
#: does, `TypeBuilder` faces `WALL_ALT_SHARE` of the buildings in it, chosen by the
#: part's own seed, and the type is not told: it builds the same shape out of whichever
#: `voice["wall"]` it is handed, which is exactly the claim the two-voice test makes
#: about a type and a voice.
_WALL_ALT = "wall_alt"

#: How many of them. A quarter: enough that a street is not one extrusion, few enough
#: that the ring still reads as one voice rather than as two.
WALL_ALT_SHARE = 0.25


def _mat_roles(mat) -> dict:
    """Normalise `mat` -- None, a family name, or a dict of roles -- to every role."""
    if mat is None:
        base = {"wall": "cobblestone", "roof": "dark_oak"}
    elif isinstance(mat, str):
        base = {"wall": mat}
    else:
        base = {k: v for k, v in dict(mat).items() if k in _MAT_ROLES and v}
    wall = base.get("wall", "cobblestone")
    return {r: base.get(r, wall) for r in _MAT_ROLES}


#: The roof parameters `building()` passes straight through to `roof()`. They live in
#: the roof spec rather than in `building()`'s own signature because they *are* the
#: roof: a voice that says "irimoya, two tiers, upturned" is describing one thing, and a
#: planner that hands that down should be able to hand down one dict.
_ROOF_EXTRAS = ("profile", "ends", "eave", "tiers")


def _roof_spec(roof, x0: int, z0: int, x1: int, z1: int):
    """(style, axis, pitch, extras) from a style name or a dict.

        The default axis runs the ridge along the *long* side, which is what a roof does
        when nobody thinks about it and the only sane default: a ridge across a longhouse
        is two hips pretending to be a gable.
        
    """
    long_x = (x1 - x0) >= (z1 - z0)
    if isinstance(roof, str):
        d = {"style": roof}
    else:
        d = dict(roof or {})
    style = d.get("style", "gable")
    pitch = tuple(d.get("pitch") or (1, 1))
    extras = {k: d[k] for k in _ROOF_EXTRAS if d.get(k) is not None}
    axis = d.get("axis")
    if style == "shed":
        # A shed's `axis` is the compass letter it falls *toward*, not an axis, and a
        # ridge letter handed to it is a KeyError two frames down. Default: down the
        # short span, which is the way a lean-to actually falls.
        return style, (axis if axis in ("n", "s", "e", "w")
                       else ("s" if long_x else "e")), pitch, extras
    return (style, (axis if axis in ("x", "z") else ("z" if long_x else "x")), pitch,
            extras)


def _wing_rect(wing, main):
    """(rectangle, refusal) for a wing: a rectangle sharing one full edge line."""
    if not wing:
        return None, None
    r = wing.get("rect") if isinstance(wing, dict) else wing
    if not r or len(r) != 4:
        return None, ("a wing is a rectangle (x0, z0, x1, z1) sharing one wall with "
                      "the footprint, or a dict with a 'rect' key")
    a, c = int(min(r[0], r[2])), int(max(r[0], r[2]))
    b, d = int(min(r[1], r[3])), int(max(r[1], r[3]))
    mx0, mz0, mx1, mz1 = main
    shares = (a == mx1 or c == mx0 or b == mz1 or d == mz0)
    if not shares:
        return None, (f"a wing has to share a wall: x {a}..{c}, z {b}..{d} touches no "
                      f"edge of the footprint x {mx0}..{mx1}, z {mz0}..{mz1} -- "
                      f"nothing has been laid")
    if c - a < 2 or d - b < 2:
        return None, f"a wing of {c - a + 1}x{d - b + 1} has no inside"
    return (a, b, c, d), None


def _outshot_rect(outshot, main, avoid: str | None = None):
    """(rectangle, side, refusal) for a lean-to along one side of the footprint.

        `avoid` is the wall the door is in. A lean-to across the front door is a lean-to
        the caller did not mean and a refusal it did not need to see, so the default side
        is the longest one that is not it.
        
    """
    if not outshot:
        return None, None, None
    if isinstance(outshot, dict):
        side = outshot.get("side")
        depth = int(outshot.get("depth", 2))
    else:
        side, depth = None, int(outshot)
    x0, z0, x1, z1 = main
    if side is None:                      # the longest side that is not the door's
        order = (["south", "north", "east", "west"] if (x1 - x0) >= (z1 - z0)
                 else ["east", "west", "south", "north"])
        side = next(s for s in order if s != avoid)
    depth = max(2, depth)
    r = {"north": (x0, z0 - depth, x1, z0), "south": (x0, z1, x1, z1 + depth),
         "west": (x0 - depth, z0, x0, z1), "east": (x1, z0, x1 + depth, z1)}.get(side)
    if r is None:
        return None, None, (f"an outshot's side is north, south, east or west, "
                            f"not {side!r}")
    return r, side, None


#: The least a range of a courtyard building may be, and the least the yard may be. A
#: range of two is two walls with nothing between them; a yard of two is a light well.
COURT_RANGE_MIN = 3
COURT_YARD_MIN = 3


def _courtyard_rects(courtyard, main):
    """({yard, ranges, depths}, refusal) for a building laid round an open yard.

        `courtyard` is the size of the **yard** -- the open ground in the middle -- as
        `(w, d)`, or a single number for a square one. Everything else is arithmetic, for
        `Builder.pad_extent`'s reason: the conversion between what a caller asks for and
        what the geometry allows belongs to one of the two layers, and this is it.

        The ring is four shells, not one hollow rectangle. North and south run the full
        width; west and east run between them, so every corner is masonry both of its
        neighbours own and no cell is walled twice by accident:

            +----------------- N -----------------+
            | W |                             | E |
            |   |            yard             |   |
            | W |                             | E |
            +----------------- S -----------------+

        A range's shell is `COURT_RANGE_MIN` deep at the least, which is a wall, a room and
        a wall; the yard is `COURT_YARD_MIN` square at the least. Anything smaller is
        refused with both numbers named, because a courtyard house on a pad that cannot
        hold one is the refusal a caller can act on.
        
    """
    if not courtyard:
        return None, None
    if isinstance(courtyard, (int, float)):
        w = d = int(courtyard)
    elif isinstance(courtyard, dict):
        r = courtyard.get("yard") or courtyard.get("rect") or ()
        if len(tuple(r)) != 2:
            return None, ("a courtyard is the size of its yard, (w, d), or one number "
                          "for a square one")
        w, d = int(r[0]), int(r[1])
    elif len(tuple(courtyard)) == 2:
        w, d = int(courtyard[0]), int(courtyard[1])
    else:
        return None, ("a courtyard is the size of its yard, (w, d), or one number for "
                      "a square one")
    x0, z0, x1, z1 = main
    W, D = x1 - x0 + 1, z1 - z0 + 1
    if w < COURT_YARD_MIN or d < COURT_YARD_MIN:
        return None, (f"a yard of {w}x{d} is not a yard: a courtyard building needs at "
                      f"least {COURT_YARD_MIN}x{COURT_YARD_MIN} of open ground in the "
                      f"middle of it")
    # The ranges, split as evenly as the remainder allows, deeper on the far side of an
    # odd one so the yard does not drift.
    west, east = (W - w) // 2, W - w - (W - w) // 2
    north, south = (D - d) // 2, D - d - (D - d) // 2
    thin = [n for n, v in (("west", west), ("east", east),
                           ("north", north), ("south", south))
            if v < COURT_RANGE_MIN]
    if thin:
        need = w + 2 * COURT_RANGE_MIN, d + 2 * COURT_RANGE_MIN
        return None, (f"a footprint of {W}x{D} cannot hold a yard of {w}x{d}: its "
                      f"{thin[0]} range would be {min(west, east, north, south)} deep "
                      f"and a range is at least {COURT_RANGE_MIN} -- a wall, a room "
                      f"and a wall. This yard needs {need[0]}x{need[1]}")
    yx0, yx1 = x0 + west, x1 - east
    yz0, yz1 = z0 + north, z1 - south
    ranges = {
        "north": (x0, z0, x1, yz0 - 1),
        "south": (x0, yz1 + 1, x1, z1),
        "west": (x0, yz0 - 1, yx0 - 1, yz1 + 1),
        "east": (yx1 + 1, yz0 - 1, x1, yz1 + 1),
    }
    return ({"yard": (yx0, yz0, yx1, yz1), "ranges": ranges,
             "depths": {"north": north, "south": south,
                        "west": west, "east": east}}, None)


def _shared_side(rect, main) -> str | None:
    """Which side of `rect` is the main mass's own wall, if any."""
    a, b, c, d = rect
    if a == main[2]:
        return "west"
    if c == main[0]:
        return "east"
    if b == main[3]:
        return "north"
    if d == main[1]:
        return "south"
    return None


def _stair_room(main, storey: int) -> str | None:
    """Why this footprint cannot carry a flight between its storeys, or None."""
    a, b, c, d = main
    inside = (c - a - 1, d - b - 1)
    need = storey + 2                  # the cell you set off from, the treads, a landing
    along, across = (inside if inside[0] >= inside[1] else inside[::-1])
    if along >= need and across >= 3:
        return None
    return (f"a footprint of {c - a + 1}x{d - b + 1} has an inside of "
            f"{inside[0]}x{inside[1]}, and a flight up a storey of {storey} blocks "
            f"needs {need} columns along it and three across, so a second flight can "
            f"go against the opposite wall rather than over the first -- nothing has "
            f"been laid. Widen it, drop to one storey, or pass stair=\"none\" and lay "
            f"the way up yourself")


def _inside_of(door, main):
    """The cell one step in from a door on the rim of `main`, or None."""
    if not door:
        return None
    a, b, c, d = main
    dx, dz = int(door[0]), int(door[-1])
    if dx == a:
        return (a + 1, dz)
    if dx == c:
        return (c - 1, dz)
    if dz == b:
        return (dx, b + 1)
    if dz == d:
        return (dx, d - 1)
    return (dx, dz)


def _l_path(start, end, avoid: set, y: int) -> list:
    """The cells of an L from `start` to `end`, the arm order that keeps clear of
    `avoid` at level `y` preferred. Both ends included."""
    (sx, sz), (ex, ez) = start, end

    def leg(along_x_first: bool) -> list:
        out = []
        if along_x_first:
            out += [(x, sz) for x in _run(sx, ex)]
            out += [(ex, z) for z in _run(sz, ez)]
        else:
            out += [(sx, z) for z in _run(sz, ez)]
            out += [(x, ez) for x in _run(sx, ex)]
        return list(dict.fromkeys(out))

    first, second = leg(True), leg(False)
    hits = [sum(1 for (x, z) in p if (x, y, z) in avoid) for p in (first, second)]
    return first if hits[0] <= hits[1] else second


def _run(a: int, b: int) -> range:
    return range(a, b + 1) if b >= a else range(a, b - 1, -1)


def _inset_shared(rect, share: str | None, by: int):
    """`rect`, pulled `by` blocks in on the side it shares with the main mass."""
    a, b, c, d = rect
    if share == "west":
        a += by
    elif share == "east":
        c -= by
    elif share == "north":
        b += by
    elif share == "south":
        d -= by
    return (min(a, c), min(b, d), max(a, c), max(b, d))


def _refusing(fn, *a, **k) -> dict:
    """Run one part of a building; turn a material refusal into that part's own refusal."""
    try:
        return fn(*a, **k)
    except ValueError as e:                # noqa: BLE001 -- this *is* the refusal
        return {"ok": False, "cells": 0, "reason": str(e)}


def _mid_two(lo: int, hi: int) -> list:
    """The one or two cells in the middle of a run -- where a link through a wall goes."""
    if hi < lo:
        return []
    mid = (lo + hi) // 2
    return [mid] if hi == lo else [mid, min(mid + 1, hi)]


def _line_between(a, b) -> list:
    """Every column from `a` to `b` inclusive, where the two share a row or a column.

        The way through one range of a courtyard: the door is in the outer wall and the
        passage comes out on the yard, and between them is a straight line across the depth
        of that range. Diagonals are not a passage and are refused by giving nothing back.
        
    """
    (ax, az), (bx, bz) = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    if ax == bx:
        lo, hi = sorted((az, bz))
        return [(ax, z) for z in range(lo, hi + 1)]
    if az == bz:
        lo, hi = sorted((ax, bx))
        return [(x, az) for x in range(lo, hi + 1)]
    return []


def _door_cell(facing: str, reserved, main, blocked):
    """((x, z), refusal) for the door leaf: on the perimeter, in the wall you walk at.

        The reserved threshold decides which column where it can; where the plan's rectangle
        and the routed doorstep disagree the door slides along its own wall rather than
        moving to another one, because which wall the door is in is the circulation pass's
        decision and this call is not entitled to overrule it.
        
    """
    x0, z0, x1, z1 = main
    wall = Builder._DOOR_WALL.get(facing, "z1")
    if wall in ("z0", "z1"):
        dz = z0 if wall == "z0" else z1
        dx = int(reserved[0]) if reserved else (x0 + x1) // 2
        dx = min(max(dx, x0 + 1), x1 - 1)
        cell = (dx, dz)
    else:
        dx = x0 if wall == "x0" else x1
        dz = int(reserved[2]) if reserved else (z0 + z1) // 2
        dz = min(max(dz, z0 + 1), z1 - 1)
        cell = (dx, dz)
    for r in blocked:
        if r[0] <= cell[0] <= r[2] and r[1] <= cell[1] <= r[3]:
            return None, (f"the door has to go at ({cell[0]},{cell[1]}) in the "
                          f"{facing}-facing wall and a wing or outshot stands on it "
                          f"(x {r[0]}..{r[2]}, z {r[1]}..{r[3]}) -- nothing has been "
                          f"laid; put the wing on another side")
    return cell, None


class Builder(Primitives):
    """Collects block writes and flushes them to the world in bulk.

        A single position written more than once keeps the last write: GDMC-HTTP rejects
        duplicate positions inside one request with "Duplicate instruction", so they are
        de-duplicated here rather than surfaced to the model.
        
    """

    def __init__(self, site):
        #: The patch of world this builder reads through. the one that prepares the
        #: ground a part stands on. See `Builder.site`.
        self.world_site = site
        self._pending: dict[tuple[int, int, int], str] = {}
        self.calls = {"place_block": 0, "place_cuboid": 0, "fill_region": 0,
                      "get_height": 0, "get_block": 0}
        #: Positions written, counting a position written twice twice. `_pending` is
        #: positions *standing* -- the last write at each. The gap between the two is
        #: what a program placed and then covered over, and it is the cheapest half of
        #: "placed versus attempted": nothing in this stack has ever compared what a
        #: program meant to put somewhere with what is actually there.
        self.writes = 0
        self.results = None
        #: What `approach()` and `flight()` laid, as [{"label", "kind", "cells"}]. The
        #: finishing pass reads this and keeps off those columns -- see `_record_path`.
        self.paths: list[dict] = []
        #: The plot registry, when this program was given one. `check_walkable()` reads
        #: the rectangles a program has reserved out of it, so its default scope is the
        #: plot rather than the bounding box of everything placed.
        self.registry = None
        #: (x, z) -> the y of the pad `site()` laid there. The ground, as the library
        #: has left it: `grade()` answers from this where it has an entry, so every
        #: primitive that carries something down to real ground carries it down to the
        #: pad and not through it. Written by `site()` and by nothing else.
        self._sited: dict[tuple[int, int], int] = {}
        #: The ground contract's resolution for the build this part is in
        #: (`ground.Resolved`), or None: `site()` lays what it settled, and `grade()`
        #: answers with a settled level for any column the contract owns. v2, B1.
        self.ground = None
        #: The cells `fitting()` has laid, and the cells `dais()` has. see
        #: `observe.floor_stances`. A dais is recorded too and is **floor**.
        self.fitting_cells: set = set()
        #: **A flight's cells are the flight's.** Every tread, its landing, the cell at
        #: the foot of it and the headroom over all of those, recorded by `flight()` so
        #: that `fitting()` refuses to stand a barrel on the foot of a stair or over the
        #: well it climbs into. The temple type furnished its upper storeys without
        #: knowing where its own flight was and sealed its top room with a barrel; a
        #: type cannot know, so the library holds the cells.
        self.flight_cells: set = set()
        #: **And the way to the foot of a flight is the flight's too.** `flight_cells`
        #: holds the treads, the landing and the one cell you set off from; nothing held
        #: the cells a person crosses to *reach* that one. A type furnishing its own
        #: ground floor -- a partition, a hearth, a fence, a counter -- can seal the
        #: corner the foot stands in without touching a cell the flight owns, and every
        #: storey above is then floor nobody can walk to: E003 and E011 as a pair, 27
        #: pairs of them in a city and 44 in the one before it. `building()` records a
        #: walkable line from the storey's own way in to each foot while the shell is
        #: still empty -- when a line always exists -- and `TypeBuilder` puts back
        #: anything written into it, the way it puts back a doorstep.
        self.flight_way: set = set()
        self.dais_cells: set = set()
        #: The columns `dress_ground` has put the skin back on. The ground pass reaches
        #: wherever `clear_trees` reached, which is past the plot by however far a tree
        #: rooted outside it leaned in, and a grass block laid where that tree stood is
        #: not somebody building off their plot. Recorded rather than inferred, for the
        #: reason `fitting_cells` is.
        self.dressed: set = set()
        #: (x, z) -> (y, block) where `clear_ground_cover` took a block that was the
        #: **ground** and not something growing on it -- a grass block, a snow block, a
        #: moss block. Taking one leaves the column a block lower than it was found, and
        #: a one-block rise is a jump. Recorded rather than inferred, for the reason
        #: `fitting_cells` and `dressed` are, and read by `_dress_worked` alone -- so no
        #: stored program's block count moves.
        self.stripped: dict = {}
        #: True only while `site()` is preparing ground. See `stripped`.
        self._library_ground = False
        #: The most blocks this builder may queue: `MAX_BLOCKS` on a 512 site, scaled by
        #: area where the driver says the site is bigger (`max_blocks_for`).
        self.max_blocks = MAX_BLOCKS
        #: Every part `site()` has prepared, in order, as it handed it to `build()`.
        #: What the ground under each instance turned out to be -- deck, platform or
        #: plinth, the floor level, the pad, what was laid -- is a fact about the run
        #: that the readout should be able to report without re-deriving it.
        self.parts: list = []
        #: Set in exactly one place. New work never sets this.
        self.allow_collide = False

    # --- placing nothing, after having placed something -------------------- A refusal
    # that has already laid half a building is not a refusal. Whether the cell the door
    # opens onto is standable is a question about the *finished* shell, because the
    # shell's own doorstep is what makes it standable. So `building()` marks the
    # builder, goes on, and winds back to the mark if its own `doorway()` refuses -- and
    # the refusal is the real one, from the real primitive, rather than a second copy of
    # the rule that would drift away from it.

    def _mark(self) -> dict:
        """Everything a call can change and a refusal has to be able to give back."""
        return {"pending": dict(self._pending), "writes": self.writes,
                "steps": list(self._step_queue()),
                "flight_seq": getattr(self, "_flight_seq", 0),
                "paths": list(self.paths),
                "fittings": set(self.fitting_cells), "dais": set(self.dais_cells),
                "flights": set(self.flight_cells), "flight_way": set(self.flight_way)}

    def _rollback(self, mark: dict) -> None:
        self._pending = dict(mark["pending"])
        self.writes = mark["writes"]
        self._steps_pending = list(mark["steps"])
        self._flight_seq = mark["flight_seq"]
        self.paths = list(mark["paths"])
        # A refusal that has already laid half a building is not a refusal, and neither
        # is one that leaves the furniture register saying a barrel is there.
        self.fitting_cells = set(mark["fittings"])
        self.dais_cells = set(mark["dais"])
        self.flight_cells = set(mark.get("flights", ()))
        self.flight_way = set(mark.get("flight_way", ()))

    # --- the API the model sees -------------------------------------------
    def place_block(self, x: int, y: int, z: int, block: str) -> None:
        """Place one block. `block` is a Minecraft block id, e.g. "oak_planks",
        optionally with state, e.g. "oak_stairs[facing=north,half=bottom]"."""
        self.calls["place_block"] += 1
        self.writes += 1
        self._pending[(int(x), int(y), int(z))] = block

    def place_cuboid(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int,
                     block: str) -> None:
        """Fill a solid box between the two corners, inclusive."""
        self.calls["place_cuboid"] += 1
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    self.writes += 1
                    self._pending[(x, y, z)] = block
        if len(self._pending) > self.max_blocks:
            raise BuildError(f"more than {self.max_blocks} blocks queued")

    def fill_region(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int,
                    block: str, replace: str | None = None) -> None:
        """Fill a box, optionally only where the existing block is `replace`.

                **`replace` reads pending writes first**, with the same precedence as
                `get_block`, so a program sees its own work. This used to test `replace`
                against the pre-build world alone, so a `replace="air"` fill queued after a
                roof read every roof cell as the air that stood there before the pass began
                and overwrote it: every temple in every voice was a solid box from its eave
                up, and lint called them clean. The case is `test_building.py`'s `dp1a`.
                
        """
        self.calls["fill_region"] += 1
        want = replace.split("[")[0].split(":")[-1] if replace is not None else None
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    if want is not None:
                        cur = self._pending.get((x, y, z))
                        if cur is None:
                            cur = self.world_site.editor.getBlock((x, y, z)).id
                        if cur.split("[")[0].split(":")[-1] != want:
                            continue
                    self.writes += 1
                    self._pending[(x, y, z)] = block
        if len(self._pending) > self.max_blocks:
            raise BuildError(f"more than {self.max_blocks} blocks queued")

    def get_height(self, x: int, z: int) -> int:
        """The y of the topmost solid block at (x, z) — the surface to build on."""
        self.calls["get_height"] += 1
        return self.world_site.height(int(x), int(z))

    def get_block(self, x: int, y: int, z: int) -> str:
        """The block id at a position, without the namespace. Reads pending writes
        first, so a program sees its own work."""
        self.calls["get_block"] = self.calls.get("get_block", 0) + 1
        p = (int(x), int(y), int(z))
        if p in self._pending:
            return self._pending[p].split("[")[0].split(":")[-1]
        try:
            return self.world_site.editor.worldSlice.getBlockGlobal(p).id.split(":")[-1]
        except Exception:
            return "air"

    # --- the circulation network, as a question this program can ask ------ Attached by
    # the harness when a circulation pass has already run. These are the whole point of
    # building circulation first: a pass that can check its own frontage while it is
    # running does not need a critique pass to find out it built a door onto a bank.
    frontage = None

    def check_door(self, x: int, y: int, z: int) -> dict:
        """Can a person walk to this doorway from the lane, without jumping, given
        everything this program has placed so far? {"ok", "reason", "jumps", ...}."""
        if self.frontage is None:
            return {"ok": True, "reason": "no circulation network in this run",
                    "jumps": None}
        # _pending_view, not _pending: a queued tread is a decision this program has
        # already made, and a flight of stairs the check cannot see reads as a bare rise
        # -- which is a jump. It reported the watch tower's own six-tread flight as
        # "reachable only by jumping 6 times" until this line said `view`.
        return self.frontage.check_door(x, y, z, self._pending_view())

    def nearest_lane(self, x: int, z: int):
        """The closest cell of the built circulation network to a column."""
        return None if self.frontage is None else self.frontage.nearest_lane(x, z)

    # --- is what I have built actually attached to itself? ----------------- The second
    # precondition, and the one that generalises past settlements: a door only means
    # something in a town, but "every part of this is held up by another part" is true
    # of a castle, a ship or a statue. an air band all the way round, lanterns hanging
    # off nothing -- and no check we had could see it. Same shape as check_door: the
    # answer accounts for the blocks this program has decided on and not yet flushed.
    _vol = None

    def _pending_view(self) -> dict:
        """Everything this program has decided on, queued treads included.

                A queued tread is a decision the program has made; a check that could not see it
                would judge a flight of stairs that is not there yet. Deciding is pure -- the
                queue is not consumed, so the facings are settled again, against a more finished
                world, when the pass actually resolves them.
                
        """
        decided = self._decide_steps()
        if not decided:
            return self._pending
        view = dict(self._pending)
        view.update(decided)
        return view

    def _world_volume(self):
        if self._vol is None:
            from . import observe
            ws = self.world_site.editor.worldSlice
            h = ws.heightmaps[world.HEIGHTMAP].astype(int) - 1
            self._vol = observe.Volume.from_world_slice(
                ws, self.world_site.x, self.world_site.z, self.world_site.sx, self.world_site.sz,
                max(0, int(h.min()) - 8), int(h.max()) + 40)
        return self._vol

    def check_attached(self, x0=None, z0=None, x1=None, z1=None, margin: int = 3,
                       above: int | None = None) -> dict:
        """Is everything placed so far one mass, standing on the ground?

                Returns {"ok", "floating": [{"cells", "bbox", "example"}, ...]}. A piece of a
                build that touches nothing else is either a mistake or something you meant to
                hang; the check reports it and the program decides. Eaves, jetties and balconies
                are *not* reported -- they are part of the grounded mass, which is what makes
                this readable where `columns_floating` never was.

                Defaults to the bounding box of everything this program has placed.

                `above` drops every piece that starts below that level, **before** the ten the
                answer carries are chosen. It is None here and it is `part['floor_y']` when a
                *type* asks -- see `TypeBuilder.check_attached` and thread 44. Nothing else
                moves: the support question is asked over the same volume either way, so a
                caller that does not pass it reads exactly what it read before.
                
        """
        from . import observe
        pending = self._pending_view()
        if not pending:
            return {"ok": True, "floating": [], "reason": "nothing placed yet"}
        xs = [p[0] for p in pending]
        zs = [p[2] for p in pending]
        x0 = (min(xs) if x0 is None else x0) - margin
        x1 = (max(xs) if x1 is None else x1) + margin
        z0 = (min(zs) if z0 is None else z0) - margin
        z1 = (max(zs) if z1 is None else z1) + margin
        sub = self._world_volume().sub(x0, z0, x1 - x0 + 1, z1 - z0 + 1)
        sub.overlay({p: b for p, b in pending.items()
                     if x0 <= p[0] <= x1 and z0 <= p[2] <= z1})
        floating = observe.unsupported(sub)
        if above is not None:
            floating = [p for p in floating if int(p["bbox"][1]) >= int(above)]
        return {"ok": not floating, "floating": floating[:10],
                "reason": ("everything is one mass standing on the ground" if not floating
                           else f"{len(floating)} pieces are attached to nothing")}

    def threshold(self, label: str):
        """The threshold the circulation pass reserved for a structure: where the lane
        meets its doorway, and which way that doorway faces."""
        return None if self.frontage is None else self.frontage.threshold(label)

    # --- the doorstep decides the floor ------------------------------------
    def floor_from_threshold(self, label: str, x: int | None = None,
                             z: int | None = None) -> dict | None:
        """Where the ground floor goes, read off the doorstep instead of guessed.

        Returns {"floor_y", "stand_y", "door", "facing", "lane", "source"}: `floor_y` is
        the level the floor *block* goes at and `stand_y` the cell a person occupies
        standing on it, which is also where the door's lower leaf goes. Falls back to
        the nearest lane cell when a structure has no reserved threshold, and to the
        ground at (x, z) when there is no circulation network at all -- each says which
        in "source", because a floor derived from a guess should not look like a floor
        derived from a doorstep."""
        t = self.threshold(label) if label else None
        if t:
            return {"floor_y": t["floor_block_y"], "stand_y": t["stand_y"],
                    "door": tuple(t["door"]), "facing": t["facing"],
                    "lane": tuple(t["lane"]), "source": "threshold"}
        if x is not None and z is not None:
            lane = self.nearest_lane(x, z)
            if lane:
                dx, dz = x - lane["x"], z - lane["z"]
                facing = ("east" if dx > 0 else "west") if abs(dx) >= abs(dz) else \
                         ("south" if dz > 0 else "north")
                return {"floor_y": lane["y"], "stand_y": lane["y"] + 1,
                        "door": None, "facing": facing,
                        "lane": (lane["x"], lane["z"], lane["y"]),
                        "source": f"nearest lane, {lane['distance']} blocks away"}
            g = self.get_height(int(x), int(z))
            return {"floor_y": g, "stand_y": g + 1, "door": None, "facing": None,
                    "lane": None, "source": "no circulation network: ground level"}
        return None

    # --- enclosed space with no way into it --------------------------------
    def seal_voids(self, x0: int, z0: int, x1: int, z1: int,
                   block: str = "stone_bricks", *, max_cells: int = 64,
                   margin: int = 2, apply: bool = True) -> dict:
        """Find pockets nobody can get into, and close the ones that can be proved.

          enclosed     the room is enclosed standing space, not a porch or an overhang,
                       and its air does not reach the outside world at all;
          unreachable  no walk path into it from anywhere outdoors;
          yours        every cell of it is inside (x0,z0)-(x1,z1).

        Anything else is reported and left alone -- including a pocket bigger than
        `max_cells`, because a room of sixty cells you cannot enter is a door you forgot,
        not a hole to fill, and the caller is the one who knows which.

        Returns {"filled", "blocks", "left": [...]}, where every entry in `left` says
        which of the three it could not prove."""
        from . import observe
        x0, x1 = min(x0, x1), max(x0, x1)
        z0, z1 = min(z0, z1), max(z0, z1)
        sub = self._world_volume().sub(x0 - margin, z0 - margin,
                                       x1 - x0 + 1 + 2 * margin,
                                       z1 - z0 + 1 + 2 * margin)
        sub.overlay(self._pending_view())
        nav = observe.Nav(sub)
        sky_open, sealed = observe.shelter(sub)
        rooms = observe.rooms(nav, sky_open, region=(x0, z0, x1 + 1, z1 + 1))
        outside = nav.flood(nav.perimeter_seeds(inset=1, step=2))

        filled = blocks = 0
        left = []
        for r in rooms:
            b = r["bbox"]
            why = []
            if r.get("enclosure", 1.0) < 0.85:
                why.append("open to the sky at its edge -- a porch, not a room")
            if set(map(tuple, r["stances"])) & set(outside):
                continue                       # you can walk into it: nothing wrong here
            # The air this space is made of, and whether any of it escapes. `sealed` is
            # air that cannot be reached *by air* from the volume's shell, so a pocket
            # any part of which is unsealed has a way out to the sky and must not be
            # packed -- that is exactly the mistake that filled 625 cells over an eave.
            pocket, escapes = self._air_pocket(sub, sealed, r["stances"])
            if escapes:
                why.append("its air reaches the open sky, so it is not a sealed pocket")
            if any(not (x0 <= p[0] <= x1 and z0 <= p[2] <= z1) for p in pocket):
                why.append("part of it lies outside the footprint you gave")
            if len(pocket) > max_cells:
                why.append(f"{len(pocket)} cells, past max_cells={max_cells}")
            if why:
                left.append({"cells": r["cells"], "air": len(pocket), "bbox": b,
                             "reason": "; ".join(why)})
                continue
            if apply:
                for (px, py, pz) in pocket:
                    self.place_block(px, py, pz, block)
                    blocks += 1
            filled += 1
        return {"filled": filled, "blocks": blocks, "left": left,
                "ok": not left,
                "reason": ("nothing enclosed and unreachable in this footprint"
                           if not filled and not left else
                           f"{filled} pockets closed, {len(left)} reported and untouched")}

    # --- can a person walk about in what I have built? --------------------- The third
    # precondition, and the interior counterpart of check_door. A door you can reach is
    # not a building you can use. A person on foot found it. This is the thing they
    # found, made askable while the program is still running -- which is the only
    # mechanism that has ever changed a builder's behaviour here.
    def check_walkable(self, label: str | None = None, x0=None, z0=None, x1=None,
                       z1=None, margin: int = 4) -> dict:
        """Walking only -- no jumping -- how much of each room's floor can you reach
        from this structure's own doorway, given everything placed so far?

        Returns {"ok", "rooms": [{"bbox", "cells", "walkable", "fraction"}, ...],
        "reason"}. `ok` is False only when a room cannot be walked into **at all**;
        a room with a step up to a platform comes back as a fraction and is not a
        failure, because a raised platform is architecture and a check that forbade it
        would deform the design.

        Seeds are the doorways on this footprint and in the ring just outside it, plus
        the one the circulation pass reserved for `label` if there is one; with no
        doorway placed yet there is nothing to answer and `ok` is True with a reason
        saying so."""
        pending = self._pending_view()
        if x0 is not None:
            return self._walkable_in(
                (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1)), label, margin,
                "the box you gave")
        rects = self._plot_rects(label)
        if len(rects) == 1:
            lab, rect = rects[0]
            return self._walkable_in(rect, lab, margin, f"the plot you reserved as {lab}")
        if rects:
            per = []
            for lab, rect in rects:
                r = self._walkable_in(rect, lab, margin,
                                      f"the plot you reserved as {lab}")
                per.append({"plot": lab, **r})
            rooms = [{**rm, "plot": p["plot"]} for p in per for rm in p["rooms"]]
            shut = sum(1 for rm in rooms if rm["walkable"] == 0)
            return {"ok": not shut, "rooms": rooms, "plots": per,
                    "doors": sum(p.get("doors", 0) for p in per),
                    "reason": (f"{shut} rooms across {len(per)} plots cannot be walked "
                               f"into at all from their own doorways"
                               if shut else
                               f"every room on all {len(per)} of your plots can be "
                               f"walked into; ask by label for one of them")}
        if self.registry is not None:
            # There *is* a registry and this label is not in it.
            return {"ok": True, "rooms": [], "scope": None, "reason":
                    (f"no plot called {label!r} has been reserved"
                     if label else
                     "you have reserved no plot on this pass")
                    + " -- call reserve() first, or give a box to judge"}
        if not pending:
            return {"ok": True, "rooms": [], "reason": "nothing placed yet"}
        # No registry at all: this is a single structure rather than a settlement pass,
        # and the box round what it built is the only footprint there is.
        xs = [p[0] for p in pending]
        zs = [p[2] for p in pending]
        return self._walkable_in((min(xs), min(zs), max(xs), max(zs)), label, margin,
                                 "the bounding box of everything you have placed, "
                                 "because this run has no plot registry")

    def _plot_rects(self, label: str | None) -> list:
        """(label, rectangle) for the plots a `check_walkable` with no box should judge.

                The plot registry is where a program's own footprints are: it reserved them
                itself, before it built, so this is a lookup rather than a guess. With a label,
                that one plot; without, the plots claimed on this pass -- a wave of three
                structures is three answers, not one box round all of them.
                
        """
        reg = self.registry
        if reg is None:
            return []
        want = [p for p in reg.plots_list() if p["label"] == label] if label else \
            list(getattr(reg, "claimed_this_pass", ()) or ())
        return [(p["label"], (min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
                              max(p["x0"], p["x1"]), max(p["z0"], p["z1"])))
                for p in want]

    def _walkable_in(self, rect: tuple, label: str | None, margin: int,
                     scope: str) -> dict:
        from . import observe
        pending = self._pending_view()
        if not pending:
            return {"ok": True, "rooms": [], "reason": "nothing placed yet"}
        a, b, c, d = rect
        sub = self._world_volume().sub(a - margin, b - margin,
                                       c - a + 1 + 2 * margin, d - b + 1 + 2 * margin)
        sub.overlay(pending)
        nav = observe.Nav(sub)
        sky_open, _ = observe.shelter(sub)
        rooms = observe.rooms(nav, sky_open, region=(a, b, c + 1, d + 1),
                              fittings=self.fitting_cells)

        seeds = []
        for (x, y, z) in sub.find(lambda s: s.split("[")[0].endswith("_door")
                                  or s.split("[")[0].endswith("_fence_gate")):
            if observe.parse_props(sub.state(x, y, z)).get("half") == "upper":
                continue
            if not (a - 1 <= x <= c + 1 and b - 1 <= z <= d + 1):
                continue
            s = nav.stance_near(x, z, y, tol=2)
            if s is not None:
                seeds.append((x, z, s))
        t = self.threshold(label) if label else None
        if t:
            dx, dy, dz = t["door"]
            s = nav.stance_near(dx, dz, dy, tol=2)
            if s is not None and (dx, dz, s) not in seeds:
                seeds.append((dx, dz, s))
        if not seeds:
            return {"ok": True, "rooms": [], "scope": scope, "reason":
                    f"no doorway on {scope} yet -- nothing to walk in from"}
        reach = set(nav.flood(seeds, max_jumps=0))

        from .lint import Context as _Ctx
        out, shut, natural = [], 0, 0
        for r in rooms:
            # A porch or an overhang, not an interior. One threshold, read from the
            # linter's own constant since A4 -- this check, E011 and `interior_walk` are
            # three readers of the same question and they now agree by construction.
            if r.get("enclosure", 1.0) < _Ctx.ENCLOSED:
                continue
            # ...and a space this much of whose walls are natural is a cave under the
            # site, whoever owns the surface above it. `lint.Context.MADE`, the same
            # constant and the same rule the linter attributes rooms to plots by --
            # applied here because a builder handed its own hillside as fifteen
            # unreachable rooms stops believing the instrument.
            if r.get("made", 1.0) < 0.25:
                natural += 1
                continue
            # The floor, not every stance: the cell on top of the barrel this program
            # just placed through `fitting()` is not somewhere it failed to make
            # walkable, and the cell on top of the dais it laid **is**. The builder
            # holding this check reads the same definition, off the same register, that
            # the round scoring it will read -- `observe.floor_stances`.
            st = set(map(tuple, r["floor"]))
            got = len(st & reach)
            shut += got == 0
            out.append({"bbox": r["bbox"], "cells": len(st), "walkable": got,
                        "fraction": round(got / len(st), 3) if st else 0.0})
        out.sort(key=lambda r: r["fraction"])
        low = [r for r in out if r["fraction"] < 0.5]
        aside = (f" ({natural} natural spaces under {scope} are not counted)"
                 if natural else "")
        return {"ok": not shut, "rooms": out, "doors": len(seeds), "scope": scope,
                "natural": natural,
                "reason": (f"{shut} of {len(out)} rooms in {scope} cannot be walked "
                           f"into at all from your own doorway -- the floor is above "
                           f"the level your feet are at when you stand in it" if shut
                           else
                           f"every room in {scope} can be walked into; {len(low)} have "
                           f"under half their floor walk-reachable (a platform is fine)"
                           if low else
                           f"every one of {len(out)} rooms in {scope} is fully walkable "
                           f"from your own doorway") + aside}

    # --- getting in, by construction ---------------------------------------
    # `check_door` and `check_walkable` tell a builder that nobody can get in. Nine
    # rounds say that is not enough: told exactly where the defect was, in the linter's
    # own words, 14 of 16 builders still shipped a building below half walkable, and 11
    # of 16 had written their own reachability sweep by the second draft. The builder's
    # simulated ground and the measured ground disagree, and the builder's is the one
    # inside the loop. So this is the rule made a primitive instead of a check: the
    # entry counterpart of `steps()` and `doorway()`, laying the one thing that keeps
    # being got wrong rather than reporting that it was got wrong again.

    #: How far from the doorway an approach may look for the lane. Wide enough for a
    #: nine-block climb switching back on itself -- the granary's, which is the worst
    #: case on the record -- and small enough that the Nav over it costs milliseconds.
    APPROACH_RADIUS = 24

    def approach(self, label: str | None = None, x: int | None = None,
                 y: int | None = None, z: int | None = None, *,
                 mat: str | None = None, width: int = 2,
                 radius: int | None = None) -> dict:
        """From the doorway to the lane, a path a person can walk without jumping.

                Returns {"ok", "cells", "reason"}. `cells` is how many columns of path were
                laid; zero with `ok` true means the door was already walkable and nothing was
                built, which is the common case and the cheap one.

                The route is searched over the ground as it now stands -- everything this
                program has decided included -- and laid as a flight: a full course where the
                way is level, a slab where it rises half a block, a tread from `steps()` where
                it rises a whole one, headroom cut above all of it, `width` columns across. It
                never cuts through anything this program has placed and never touches a lane
                cell, so it cannot fix a door by demolishing the wall it is in or by burying
                the network the town walks on.

                With no coordinates it takes the lowest door leaf this program has placed
                nearest the threshold reserved for `label`, and falls back to the reserved
                doorstep itself when the program has placed no leaf yet.
                
        """
        from . import observe
        from .prims import material as _material
        radius = int(self.APPROACH_RADIUS if radius is None else radius)
        t = self.threshold(label) if label else None
        pending = self._pending_view()

        door = self._approach_door(pending, t, x, y, z)
        if door is None:
            return {"ok": False, "cells": 0,
                    "reason": "nothing to approach: this program has placed no door "
                              "and no threshold was reserved for this structure"}
        dx, dy, dz = door

        sub = self._world_volume().sub(dx - radius, dz - radius,
                                       2 * radius + 1, 2 * radius + 1)
        sub.overlay({p: b for p, b in pending.items()
                     if abs(p[0] - dx) <= radius and abs(p[2] - dz) <= radius})
        nav = observe.Nav(sub)
        ds = nav.stance_near(dx, dz, dy, tol=2)
        if ds is None:
            return {"ok": False, "cells": 0,
                    "reason": f"nothing can stand in the doorway at ({dx},{dy},{dz}) -- "
                              f"it is blocked, or there is no floor at that height"}

        net = self.frontage.net if self.frontage else None
        lanes = {(lx, lz) for (lx, lz, _) in (net.surface() if net else [])}
        seeds = []
        for (lx, lz, ly) in (net.surface() if net else []):
            if abs(lx - dx) > radius or abs(lz - dz) > radius:
                continue
            s = nav.stance_near(lx, lz, ly + 1, tol=1)
            if s is not None:
                seeds.append((lx, lz, s))
        if not seeds:
            # No network, or none of it within reach: the open ground round the edge of
            # the working area is what a person arrives on, and is the same seed set
            # every from-outdoors measurement in this project uses.
            seeds = nav.perimeter_seeds(inset=1, step=2)
        reach = set(nav.flood(seeds, max_jumps=0))
        if (dx, dz, ds) in reach:
            # **The way is written down even when nothing had to be laid.** A door
            # reachable over ground as it stands is reachable over columns nobody has
            # claimed, and a type building on its plot then laid its footing course
            # across them: two of a city's doors were walkable when their parts were
            # sited and not when they were built, on ground the approach had never made
            # -- it had only walked it. The columns walked are recorded as the way in,
            # so `_way_in` holds them open against the type. Lane cells are the lane's
            # already.
            walked = nav.route(seeds, (dx, dz, ds), max_jumps=0) or []
            cols = [(x, z) for (x, z, _s) in walked if (x, z) not in lanes]
            if cols:
                self._record_path(label, "approach", cols)
            return {"ok": True, "cells": 0, "walked": len(cols),
                    "reason": ("the door is already walk-reachable from outside"
                               + (f"; the {len(cols)} columns it is walked over are "
                                  f"held open" if cols else ""))}

        # What this program has *decided*, minus the air it cleared. A builder that
        # cleared headroom over its own doorstep wrote air there, and a path refusing to
        # cross air the builder asked for is a path that cannot leave the door -- every
        # candidate whose door was jump-only failed exactly there. But the test cannot
        # be "does it get in the way", because **a door does not**: an oak_door is no-
        # collision in the walk model, so an approach that only protected solid blocks
        # cut its headroom straight through the leaf, the jambs and the lintel and left
        # the building it had just made walkable with no door in it. Anything the
        # program put somewhere on purpose is now protected; only its own air is
        # crossed.
        built = {p for p, b in pending.items()
                 if b.split("[")[0].split(":")[-1] not in AIR}
        before_lay = set(self._pending)
        route = self._approach_route(sub, nav, reach, lanes, built, (dx, dz), ds)
        if route is None:
            return {"ok": False, "cells": 0,
                    "reason": f"no walkable line could be laid from the door at "
                              f"({dx},{dy},{dz}) to open ground within {radius} blocks "
                              f"without cutting through what this program has built"}

        fam = mat or (net.notes.get("material") if net else None) or "cobblestone"
        full, _stairs, slab = _material(fam)
        laid, cols = self._lay_approach(sub, route, full, slab, fam, width, lanes, built)
        self._record_path(label, "approach", cols)
        # The way in is a worked piece of ground like any other. The path itself is
        # paved in `fam` and is not what this catches -- it is the columns beside it
        # whose headroom was cut, which come out as the subsoil the cut exposed with the
        # litter of whatever was growing there still lying on them.
        if cols:
            xs = [c[0] for c in cols]
            zs = [c[1] for c in cols]
            self._dress_worked(before_lay,
                               (min(xs), min(zs), max(xs), max(zs)))

        # Answered against the world this call has just changed, not against the world
        # it was asked in. A primitive that reports success without re-asking is a check
        # that trusts its own arithmetic, which is the failure this exists for.
        sub2 = self._world_volume().sub(dx - radius, dz - radius,
                                        2 * radius + 1, 2 * radius + 1)
        pend2 = self._pending_view()
        sub2.overlay({p: b for p, b in pend2.items()
                      if abs(p[0] - dx) <= radius and abs(p[2] - dz) <= radius})
        nav2 = observe.Nav(sub2)
        seeds2 = []
        for (lx, lz, ly) in (net.surface() if net else []):
            if abs(lx - dx) > radius or abs(lz - dz) > radius:
                continue
            s = nav2.stance_near(lx, lz, ly + 1, tol=1)
            if s is not None:
                seeds2.append((lx, lz, s))
        if not seeds2:
            seeds2 = nav2.perimeter_seeds(inset=1, step=2)
        ds2 = nav2.stance_near(dx, dz, dy, tol=2)
        ok = ds2 is not None and (dx, dz, ds2) in set(nav2.flood(seeds2, max_jumps=0))
        return {"ok": bool(ok), "cells": laid,
                "reason": (f"{laid} columns of path laid; the door at ({dx},{dy},{dz}) "
                           f"is now walk-reachable from outside" if ok else
                           f"{laid} columns of path laid and the door at "
                           f"({dx},{dy},{dz}) is still not walk-reachable -- something "
                           f"else is in the way")}

    @staticmethod
    def _approach_door(pending: dict, t, x, y, z):
        """Which doorway an `approach()` with no coordinates is for."""
        from . import observe
        if x is not None and z is not None:
            return (int(x), int(y), int(z))
        leaves = []
        for (px, py, pz), b in pending.items():
            nm = b.split("[")[0].split(":")[-1]
            if not (nm.endswith("_door") or nm.endswith("_fence_gate")):
                continue
            if observe.parse_props(b).get("half") == "upper":
                continue
            leaves.append((px, py, pz))
        ref = tuple(t["door"]) if t else None
        if leaves:
            if ref:
                return min(leaves, key=lambda p: (max(abs(p[0] - ref[0]),
                                                      abs(p[2] - ref[2])), p[1]))
            return min(leaves, key=lambda p: p[1])
        return ref

    #: What one column of approach costs to lay, per block of fill under it and per
    #: solid block cut out above it. Cutting is eight times fill because a path that
    #: tunnels through a bank is a path that took a wall out of somebody's building; the
    #: search would rather go the long way round, and on this ground it always can.
    _FILL_COST, _CUT_COST, _STEP_COST = 1, 8, 4

    def _approach_route(self, sub, nav, reach: set, lanes: set, placed: set,
                        door: tuple, ds: int):
        """The cheapest flight from the doorstep to ground somebody can already walk on.

                A Dijkstra over (column, surface half-height), where a state is *the path this
                call would build there* rather than the ground that is there now. Three
                transitions and no others, which is what makes the result walkable by
                construction rather than by inspection:

                    level      the next column carries its surface at the same height
                    half       one half-block down: a slab, and a free step back up
                    whole      one whole block down: the higher column is a **tread**, and a
                               tread is the only thing in the walk model you can step a whole
                               block up onto without jumping

                The flight descends and never climbs, so every tread has one up-slope side and
                `steps()` cannot be asked to face a tread two ways at once. The first column out
                of the doorway may only be level or half a block down, because a whole-block
                rise there would need a tread *in the doorway*, and the doorway is the builder's
                floor rather than ours to re-lay.

                **A column is settled once and carries one height.** A column has one surface,
                so a route that came back to a column it had already used at a different height
                is not a route -- the second visit overwrites the first and the flight is left
                with a two-block rise in the middle of it. The first version of this searched
                (column, height) freely and produced exactly that: a switchback oscillating
                between two adjacent columns, gaining a block a turn, walkable on paper and
                impossible in the world. Dijkstra settles in cost order, so taking the first
                height a column is settled at and dropping the rest makes every route physical
                by construction; a real switchback turns through fresh columns and is unaffected.
                
        """
        import heapq
        y0, sy, _ = sub.y0, sub.shape[1], None
        dx, dz = door

        def dirs(cx, cz):
            return ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1))

        def cell_of(s):
            return (s // 2 - 1) if s % 2 == 0 else (s - 1) // 2

        top_cache: dict = {}

        def ground_top(cx, cz):
            """Topmost solid cell in the column, as the walk model classes solidity."""
            if (cx, cz) in top_cache:
                return top_cache[(cx, cz)]
            g = y0 - 1
            for yy in range(y0 + sy - 1, y0 - 1, -1):
                if sub.inside(cx, yy, cz) and _occupies(sub.state(cx, yy, cz)):
                    g = yy
                    break
            top_cache[(cx, cz)] = g
            return g

        wet_cache: dict = {}

        def water_top(cx, cz):
            if (cx, cz) not in wet_cache:
                wet_cache[(cx, cz)] = self._approach_water(sub, cx, cz)
            return wet_cache[(cx, cz)]

        def cost_at(cx, cz, s):
            """None where a column may not carry a path; otherwise what laying it costs."""
            if (cx, cz) in lanes or not nav.in_col(cx, cz):
                return None
            cell = cell_of(s)
            if not (y0 <= cell and cell + 2 < y0 + sy):
                return None
            # The two cells a person occupies standing here are the ones that matter:
            # cutting those out of this program's own build is how a path fixes a door
            # by taking the wall out of it. The course *under* their feet may well be
            # this program's own -- its doorstep, its terrace, its plinth -- and laying
            # a path along the top of that is the normal case, not a violation.
            for c in (cell + 1, cell + 2):
                if (cx, c, cz) in placed:
                    return None
            # ...and the third cell is priced, not forbidden. A flight rises, and a
            # person mid-step on a rising path has their head in the third cell above
            # the column they left (observe.STAIR_CLEAR), so a route that has to cut one
            # out is a worse route -- but a route that ducks under this program's own
            # eave and is level all the way is still a route, and refusing it outright
            # would lose doors this used to make walkable.
            cut = sum(1 for c in (cell + 1, cell + 2, cell + 3)
                      if sub.inside(cx, c, cz) and _occupies(sub.state(cx, c, cz)))
            # the building was sited correctly and had no way in. A jetty stands *above*
            # the waterline on piles, so the course below it is water and is not the
            # path's to fill.
            wet = water_top(cx, cz)
            if wet is not None:
                # At the waterline or above it, never under: a ford across a puddle one
                # block deep is a course laid at the surface, and a way in across a tarn
                # is a jetty a course or two over it. Below the waterline is a trench in
                # a lake, which is what the bed-measured price used to buy.
                if cell < wet:
                    return None
                fill = cell - wet
            else:
                fill = max(0, cell - ground_top(cx, cz))
            return self._STEP_COST + self._FILL_COST * fill + self._CUT_COST * cut

        def goal_from(cx, cz, s, came):
            """Ground already walkable that this column hands you onto, or None."""
            for k, (nx, nz) in enumerate(dirs(cx, cz)):
                for ns in nav.stances_in_column(nx, nz):
                    if (nx, nz, ns) not in reach:
                        continue
                    d = s - ns
                    if d == 0 or d == 1 or (d == 2 and s % 2 == 0 and k == came):
                        return (nx, nz, ns)
            return None

        start: list = []
        for k, (nx, nz) in enumerate(dirs(dx, dz)):
            for s in (ds, ds - 1):
                c = cost_at(nx, nz, s)
                if c is not None:
                    start.append((c, (nx, nz, s, k)))
        best: dict = {}
        prev: dict = {}
        heap: list = []
        for c, st in start:
            if st not in best or c < best[st]:
                best[st] = c
                prev[st] = None
                heapq.heappush(heap, (c, -st[2], st))
        seen: set = set()
        settled_col: set = {(dx, dz)}
        while heap:
            c, _neg, st = heapq.heappop(heap)
            if st in seen or c > best.get(st, c):
                continue
            seen.add(st)
            cx, cz, s, came = st
            if (cx, cz) in settled_col:
                continue
            settled_col.add((cx, cz))
            end = goal_from(cx, cz, s, came)
            if end is not None:
                path = []
                cur = st
                while cur is not None:
                    path.append(cur[:3])
                    cur = prev[cur]
                return list(reversed(path)) + [("goal",) + end]
            for k, (nx, nz) in enumerate(dirs(cx, cz)):
                if (nx, nz) in settled_col:
                    continue
                for ns in (s, s - 1, s - 2):
                    if ns == s - 2 and (s % 2 or k != came):
                        # A whole-block drop is a tread, and a tread's top face is a
                        # block top rather than a slab. It also cannot be the cell a
                        # flight turns on: a tread has one raised quarter and it must
                        # point up-slope, so a tread entered from the south and left to
                        # the west would have to face two ways. Turn on the landings.
                        continue
                    nc = cost_at(nx, nz, ns)
                    if nc is None:
                        continue
                    nst = (nx, nz, ns, k)
                    if nst in seen:
                        continue
                    nd = c + nc
                    if nd < best.get(nst, 1 << 30):
                        best[nst] = nd
                        prev[nst] = st
                        # **Highest surface first at equal cost.** A column is settled
                        # once and the flight only ever descends, so a column settled
                        # half a block low can never be raised again -- and the ordering
                        # was the state tuple, which sorts by x, then z, then
                        # *ascending* height. The slab-height branch tied with the full-
                        # course one, won the tie-break by being numerically smaller,
                        # settled every column half a block under the lane and arrived
                        # unable to step up onto it. `approach()` came back "no walkable
                        # line could be laid" for a door six blocks from a level lane.
                        heapq.heappush(heap, (nd, -ns, nst))
        return None

    def _lay_approach(self, sub, route: list, full: str, slab: str, fam: str,
                      width: int, lanes: set, built: set) -> tuple[int, list]:
        """Place the flight the search found: surface, fill under it, headroom over it.

                Returns (columns laid, the columns themselves). The second is what the finishing
                pass has to be told about -- see `_record_path`.

                Treads go through `steps()` rather than down as stair states, for the reason
                every tread in this library does: the facing is decided at the end from the
                finished ground on both sides, so a flight laid against a wall or a bank comes
                out facing up-slope rather than facing whatever happened to be beside its
                bottom step. They are queued run by run, because a turn in a flight is two runs
                and one run cannot face two ways.

                Width is laid as parallel rows at the identical heights, so the second column
                of a two-wide flight is the same flight and not a landing bolted to its side.
                The widening is held to the same two rules the centreline is -- off the lane,
                and never through anything this program has built -- and simply narrows back to
                one column wherever it is not, because a path a block narrower is a path and a
                path through somebody's wall is not.
                
        """
        cells = [c for c in route if c[0] != "goal"]
        end = next((c[1:] for c in route if c[0] == "goal"), None)
        levels = [c[2] for c in cells] + ([end[2]] if end else [cells[-1][2]])
        # Which way the flight is going at each cell. A tread is stepped up onto from
        # its downhill neighbour, so this is also the tread's axis -- and it has to be
        # handed to `steps()` rather than left to be inferred, because a lone tread
        # between two landings has no neighbour in its own run to infer it from and
        # falls back to z. That put one flight's bottom step across its own direction of
        # travel and left the door it served a jump away.
        nexts = cells + [(end or cells[-1])]
        axes = [("x" if nexts[i + 1][0] != c[0] else "z")
                for i, c in enumerate(cells)]
        # ...and which way each tread's raised quarter points, for the same reason
        # `flight()` passes `prefer`: a tread the flight does not continue past has no
        # run to read its direction off, and without this it is demoted to a slab. The
        # route descends away from the door, so a tread's up-slope side is the side the
        # door is on -- the opposite of the cell the route goes to next. Measured: on a
        # six-block bank a three-block descent came out as three slabs and left the door
        # it had just been laid for a two-block climb away.
        prefers = []
        for i, c in enumerate(cells):
            n = nexts[i + 1]
            prefers.append(("east" if c[0] > n[0] else "west") if axes[i] == "x"
                           else ("south" if c[2] > n[2] else "north"))
        rows = self._approach_rows(cells, max(1, int(width)))

        # Everything this call is about to lay a surface on. The queue is cleared of any
        # tread of the *builder's* still standing at one of them first: a tread is
        # placed at flush, after this call has returned, so a program that queued a step
        # where the path now runs would have the last word and put a block back where
        # the flight needs a slab. The check inside `approach()` saw the flight; the
        # world got the tread.
        own = {(col[0], (s // 2 - 1) if s % 2 == 0 else (s - 1) // 2, col[1])
               for row in rows for i, col in enumerate(row) if col
               for s in [cells[i][2]]}
        q = self._step_queue()
        q[:] = [t for t in q if (t["x"], t["y"], t["z"]) not in own]

        # ...and what stands there now, the builder's own decided treads included, so a
        # course already at the right height is recognised rather than re-laid.
        view = self._pending_view()
        laid = 0
        cols: list = []
        for n, row in enumerate(rows):
            run: list = []
            run_axis, run_prefer = "z", None
            for i, col in enumerate(row):
                s = cells[i][2]
                cell = (s // 2 - 1) if s % 2 == 0 else (s - 1) // 2
                if col is not None and n and not self._approach_free(
                        col, cell, lanes, built):
                    col = None
                if col is None:
                    if run:
                        self.steps(run, fam, axis=run_axis, prefer=run_prefer)
                        run = []
                    continue
                px, pz = col
                tread = (s - levels[i + 1]) == 2
                # A jetty over water and fill over ground. The lake below is left alone
                # -- see `cost_at` and `_site_lay` -- so what carries the deck is a pile
                # every other column of the run, driven to the bed. Filling instead
                # would pour the whole route solid and drain what it crosses.
                wet = self._approach_water(sub, px, pz)
                g = wet if wet is not None else self._approach_ground(sub, px, pz)
                if wet is not None and not i % 2:
                    bed = self._approach_ground(sub, px, pz)
                    for c in range(bed + 1, wet + 1):
                        have = view.get((px, c, pz))
                        if have is None or have.split("[")[0].split(":")[-1] in AIR:
                            self.place_block(px, c, pz, full)
                for c in range(g + 1, cell):
                    have = view.get((px, c, pz))
                    if have is None or have.split("[")[0].split(":")[-1] in AIR:
                        self.place_block(px, c, pz, full)
                if tread:
                    if run and axes[i] != run_axis:
                        self.steps(run, fam, axis=run_axis, prefer=run_prefer)
                        run = []
                    run_axis, run_prefer = axes[i], prefers[i]
                    run.append((px, cell, pz))
                else:
                    # A course already standing at exactly this height is this
                    # structure's own doorstep or terrace: walk along it rather than re-
                    # lay it in the lane's material. A course at a *different* height is
                    # not the path, and leaving it is how the first version of this laid
                    # a flight whose bottom step was a block too high -- a full cube
                    # where the search had planned a slab, and a two-block rise with
                    # nothing to step onto.
                    have = view.get((px, cell, pz))
                    if not (have and 2 * cell + _top_face(have) == s):
                        self.place_block(px, cell, pz,
                                         f"{slab}[type=bottom]" if s % 2 else full)
                    if run:
                        self.steps(run, fam, axis=run_axis, prefer=run_prefer)
                        run = []
                # Three cells of headroom over every column of the flight, not two: the
                # path rises, and a body mid-step is over the lower column at the upper
                # column's height. Nothing this program placed is cut here -- the test
                # is still "the world's, or our own air".
                for c in (cell + 1, cell + 2, cell + 3):
                    have = view.get((px, c, pz))
                    if have is None or have.split("[")[0].split(":")[-1] in AIR:
                        self.place_block(px, c, pz, "air")
                laid += 1
                cols.append((px, pz))
            if run:
                self.steps(run, fam, axis=run_axis, prefer=run_prefer)
        return laid, cols

    @staticmethod
    def _approach_free(col: tuple, cell: int, lanes: set, built: set) -> bool:
        """May the widening put a course at `cell` in this column?"""
        return (col not in lanes
                and (col[0], cell + 1, col[1]) not in built
                and (col[0], cell + 2, col[1]) not in built)

    @staticmethod
    def _approach_rows(cells: list, width: int) -> list:
        """The flight as `width` parallel rows of columns, centreline first.

                Offsets are perpendicular to the way the flight is going at that cell, and a
                row entry is None where the offset lands back on the centreline -- which is
                what happens at a turn, and where a widened flight would otherwise write over
                its own treads at two different heights.
                
        """
        centre = [(x, z) for (x, z, _s) in cells]
        rows = [centre]
        if width <= 1:
            return rows
        perp = []
        for i in range(len(cells)):
            a = cells[max(0, i - 1)]
            b = cells[min(len(cells) - 1, i + 1)]
            vx, vz = b[0] - a[0], b[2] - a[2]
            perp.append((-vz, vx) if (vx or vz) else (0, 1))
        taken = set(centre)
        for k in range(1, width):
            row = []
            for i, (cx, cz) in enumerate(centre):
                px, pz = perp[i]
                col = (cx + k * (1 if px > 0 else -1 if px < 0 else 0),
                       cz + k * (1 if pz > 0 else -1 if pz < 0 else 0))
                row.append(None if col in taken else col)
            taken |= {c for c in row if c}
            rows.append(row)
        return rows

    @staticmethod
    def _approach_ground(sub, cx: int, cz: int) -> int:
        for yy in range(sub.y0 + sub.shape[1] - 1, sub.y0 - 1, -1):
            if sub.inside(cx, yy, cz) and _occupies(sub.state(cx, yy, cz)):
                return yy
        return sub.y0 - 1

    @staticmethod
    def _approach_water(sub, cx: int, cz: int) -> int | None:
        """The waterline in this column, or None where the column is dry.

                The first thing met coming down: water means this column is lake and the way
                across it is a jetty; anything you can stand on means it is ground and the way
                across it is fill. Read off the same view the route is searched over, so a
                column the pass has already decked reads as decked and not as water.
                
        """
        from . import observe
        for yy in range(sub.y0 + sub.shape[1] - 1, sub.y0 - 1, -1):
            if not sub.inside(cx, yy, cz):
                continue
            s = sub.state(cx, yy, cz)
            if s.split("[")[0].split(":")[-1] in observe._LIQUID:
                return yy
            if _occupies(s):
                return None
        return None

    # --- getting upstairs, by construction. Entry became reliable -- 23 of 24 doors --
    # the moment the rule moved out of a check and into a primitive. The defect then
    # moved upstairs. Nothing owned the internal stair. `steps()` lays treads and
    # decides facings but nobody asks it to connect storey to storey, and
    # `storey_steps()` is about massing. This is the counterpart primitive, and it is
    # deliberately the narrowest thing that answers the measured defect: from this floor
    # to that floor, walkable, or say why not.

    def flight(self, label: str | None, x: int, z: int, y0: int, y1: int,
               facing: str, *, mat: str | None = None) -> dict:
        """A straight internal stair from floor `y0` up to floor `y1`, or a refusal.

                Returns {"ok", "cells", "removed", "reason"}. `cells` counts the treads and the
                landing laid; `removed` counts cells cleared to open the stairwell and the
                headroom over it.

                `x`, `z` is where the **first tread** goes, at `(x, y0 + 1, z)` -- one block in
                from where a person stands on the lower floor, in the direction `facing`. Each
                tread after it steps one column along and one block up, so a flight from `y0` to
                `y1` is `y1 - y0` treads, and the cell past the last one is a landing on the
                upper floor. `facing` is the way you climb: "north", "south", "east" or "west".
                One column wide, because a stair a person can walk up is the thing being built
                here and a grand double flight is a design decision, not a primitive.

                Three things it does that a program written by hand keeps getting wrong:

                  - **The treads go through `steps()`**, so their facing is decided at the end
                    from the finished ground, up-slope, the same way every other tread in this
                    library is decided. A stair facing down-slope is not walkable in the walk
                    model and it is what E004 exists for.
                  - **It cuts the stairwell.** Two cells are cleared over every tread and over
                    the landing -- exactly the two cells a person standing there occupies -- and
                    that is what opens the hole through an upper floor the flight passes into.
                    A flight laid under an unbroken floor is a flight into a ceiling.
                  - **It refuses rather than damaging the building.** If a tread or the landing
                    would replace something the program has placed that is not floor or air --
                    a wall, a chimney, a fitting -- nothing is laid at all and the reason names
                    the cell. A floor may be cut, because cutting a floor is what a stairwell
                    is; a wall may not, because a stair through a wall is a hole.

                Then it asks the walk model, over the world this call has just changed, whether
                the landing can be reached on foot from the cell at the foot of the flight, and
                reports what it finds. A primitive that reported success from its own arithmetic
                would be a check that trusts itself, which is the failure `approach()` exists to
                avoid.
                
        """
        from . import observe
        from .prims import material as _material
        x, z, y0, y1 = int(x), int(z), int(y0), int(y1)
        d = _FACE_DIR.get(facing)
        if d is None:
            return {"ok": False, "cells": 0, "removed": 0,
                    "reason": f"facing must be north, south, east or west, "
                              f"not {facing!r}"}
        dx, dz = d
        rise = y1 - y0
        if rise <= 0:
            return {"ok": False, "cells": 0, "removed": 0,
                    "reason": f"a flight climbs: the upper floor y1={y1} must be above "
                              f"the lower floor y0={y0}"}

        treads = [(x + i * dx, y0 + 1 + i, z + i * dz) for i in range(rise)]
        land = (x + rise * dx, y1, z + rise * dz)
        foot = (x - dx, z - dz)          # the cell you set off from, on the lower floor

        # Everything below this line may place a block, and the last thing this call
        # does is *walk* what it laid. Mark, go on, and wind back to the mark.
        mark = self._mark()
        view = self._pending_view()
        for cell in treads + [land]:
            what = self._flight_blocked(view, cell)
            if what is not None:
                where = "the landing" if cell == land else "a tread"
                return {"ok": False, "cells": 0, "removed": 0,
                        "reason": f"{where} would replace the {what} you placed at "
                                  f"({cell[0]},{cell[1]},{cell[2]}), which is not floor "
                                  f"or air -- nothing has been laid"}

        fam = mat or "cobblestone"
        full, _stairs, _slab = _material(fam)
        xs = [foot[0], land[0]] + [t[0] for t in treads]
        zs = [foot[1], land[2]] + [t[2] for t in treads]
        m = 4
        a, b = min(xs) - m, min(zs) - m
        sub = self._world_volume().sub(a, b, max(xs) + m - a + 1, max(zs) + m - b + 1)
        sub.overlay({p: bl for p, bl in view.items()
                     if a <= p[0] <= max(xs) + m and b <= p[2] <= max(zs) + m})

        # The landing first, so the surface probe that decides the top tread's facing
        # sees something at the top of the flight rather than the void it is cut into.
        # Left alone where a full-height surface already stands there, which is the
        # normal case: the landing is the upper storey's own floor.
        have = view.get(land)
        if not (have and _top_face(have) == 2):
            standing = sub.state(*land) if sub.inside(*land) else ""
            if not (standing and _top_face(standing) == 2):
                self.place_block(land[0], land[1], land[2], full)

        removed = 0
        for (cx, cy, cz) in treads + [land]:
            # Three cells, not two. Two is what a person occupies standing still; a
            # person *mid-step* is lifted to the upper tread's height while still over
            # the lower one, and their head is in the third. See observe.STAIR_CLEAR.
            for c in (cy + 1, cy + 2, cy + 3):
                here = view.get((cx, c, cz))
                if here is None and sub.inside(cx, c, cz):
                    here = sub.state(cx, c, cz)
                if here and here.split("[")[0].split(":")[-1] not in AIR:
                    self.place_block(cx, c, cz, "air")
                    removed += 1

        # `prefer` matters for a one-block rise: a lone tread has no run to read its own
        # direction off, and without this it would be demoted to a slab and the flight
        # would be a half-block step to nowhere.
        self.steps(treads, fam, axis=("x" if dx else "z"), prefer=facing)
        self._record_path(label, "flight",
                          [(t[0], t[2]) for t in treads] + [(land[0], land[2])])

        # ...and resolved *here*, before the walk, rather than left on the queue for
        # `flush()`. A queued tread is not in `_pending`, so a builder that inspected
        # its own work after calling this saw a flight that was not there and built a
        # second one by hand. `_pending_view()` settles the queue for the checks that
        # ask; nothing settled it for the program itself.
        self.resolve_steps()

        # Until here a flight laid treads and nothing else, so a flight that runs down
        # the middle of a room -- rather than against a wall, which is the only case
        # anybody had looked at -- is four stair blocks hanging in mid air with air
        # under every one of them. That is a real defect and the linter says so: E010,
        # "held up by nothing". It is also a defect the library was handing to its
        # callers to clean up, and `shop_house` did exactly that -- it read
        # `check_attached()`, found a one-cell floating piece it had not meant to place,
        # and deleted it, taking out the third tread of its own stair. **Nine of the
        # city's thirteen sealed rooms are that**: a flight whose own type erased a
        # tread, an E004 on the tread below it because the run it was part of no longer
        # continues, and two storeys nobody can walk into. One cell under each tread,
        # where it is air, is all it takes: the cell under the first tread is the lower
        # floor, the cell under the second rests on that floor, and each one after it is
        # face-adjacent to the tread before it, so the whole flight is one mass standing
        # on the floor it starts from. Laid *after* `resolve_steps()`, so it cannot move
        # a facing the treads were resolved with, and inside the mark, so a refusal
        # still gives it back.
        for (cx, cy, cz) in treads:
            under = (cx, cy - 1, cz)
            here = view.get(under)
            if here is None and sub.inside(*under):
                here = sub.state(*under)
            if here is None or here.split("[")[0].split(":")[-1] in AIR:
                self.place_block(under[0], under[1], under[2], full)

        ok, why = self._flight_reachable(a, b, max(xs) + m, max(zs) + m,
                                         foot, y0, land, y1)
        if not ok:
            self._rollback(mark)
            return {"ok": False, "cells": 0, "removed": 0,
                    "reason": (f"a flight from y={y0} to y={y1} was laid and {why} -- "
                               f"so nothing has been laid: the volume is as it was "
                               f"before this call")}
        n = len(treads) + 1
        # ...and the flight's cells are held against furniture from here on: the treads,
        # the landing, the foot cell and the headroom a person climbing occupies over
        # each. See `flight_cells`.
        held = set()
        for (cx, cy, cz) in treads + [land]:
            held.update((cx, cy + k, cz) for k in range(0, 4))
        held.update((foot[0], y0 + k, foot[1]) for k in range(1, 4))
        self.flight_cells.update(held)
        return {"ok": True, "cells": n, "removed": removed,
                # Where the flight starts and where it arrives, so a caller can hold the
                # way to the one and from the other open. See `flight_way`.
                "foot": (foot[0], y0, foot[1]), "landing": (land[0], land[1], land[2]),
                "reason": (f"{len(treads)} treads and a landing laid from y={y0} to "
                           f"y={y1}; the landing is walk-reachable from the foot of the "
                           f"flight")}

    @staticmethod
    def _flight_blocked(view: dict, cell: tuple) -> str | None:
        """What this program has placed at a cell that a flight may not replace.

                None where the cell is free to build in. Three cases count as free: the program
                placed nothing there (the cell belongs to the world, and cutting a stair into
                the hillside a building sits in is ordinary), the program placed air, or the
                program placed something with open space directly above it -- which is what a
                **floor** is, and cutting a floor is precisely what a stairwell does. Something
                with more of itself stacked on top is a wall, a chimney, a pier or a mass, and a
                stair driven through one of those is a hole in a building.
                
        """
        b = view.get(cell)
        if b is None or b.split("[")[0].split(":")[-1] in AIR:
            return None
        above = view.get((cell[0], cell[1] + 1, cell[2]))
        if above is None or above.split("[")[0].split(":")[-1] in AIR:
            return None
        return b.split("[")[0].split(":")[-1]

    def _flight_reachable(self, a, b, c, d, foot, y0, land, y1):
        """Can you walk from the foot of the flight up onto its landing, now?

                Asked against a volume rebuilt from the world with everything this program has
                decided overlaid -- the queued treads included, which is what `_pending_view`
                settles -- because a flight whose treads are still in the queue is a flight the
                walk model cannot see.
                
        """
        from . import observe
        sub = self._world_volume().sub(a, b, c - a + 1, d - b + 1)
        sub.overlay({p: bl for p, bl in self._pending_view().items()
                     if a <= p[0] <= c and b <= p[2] <= d})
        nav = observe.Nav(sub)
        fs = nav.stance_near(foot[0], foot[1], y0 + 1, tol=2)
        if fs is None:
            return False, (f"nobody can stand at the foot of it, at "
                           f"({foot[0]},{y0 + 1},{foot[1]}) -- there is no floor there, "
                           f"or something is standing in it")
        ls = nav.stance_near(land[0], land[2], y1 + 1, tol=2)
        if ls is None:
            return False, (f"nobody can stand on the landing at "
                           f"({land[0]},{land[1] + 1},{land[2]}) -- something is in the "
                           f"way above it")
        reach = nav.flood([(foot[0], foot[1], fs)], max_jumps=0,
                          bounds=(a, b, c, d))
        if (land[0], land[2], ls) in reach:
            return True, ""
        return False, ("the landing still cannot be walked to from it -- something "
                       "beside the flight is blocking the way, or a tread could not be "
                       "justified and was demoted to a slab")

    # --- siting, by construction. The seventh time this law has been applied and the
    # first time at the type layer. Five to ten `get_height` reads, a
    # `foundation_to_grade` or a hand-cut terrace, its own `steps()` down to the lane.
    # three of them wet, one 96.4% water, one with 24 blocks of relief -- that hand-
    # written ground work gave 5 of 12 against a bar of 10, with one instance sealed at
    # 0.0% walkable. The reading is the one this project has taken six times: the types
    # were doing the library's job. So the ground is not the type's to touch. `site()`
    # prepares it -- sounds the bed, decides between a deck, a platform and a plinth,
    # lays the way up from the lane -- and hands back the part with the floor level on
    # it. `TypeBuilder` is the other half: it refuses a type that reaches for the ground
    # anyway. The three cases are the three the measurement found, in the order they
    # dominate: water is not ground, relief has to be cut or filled to one level, and
    # everything else is a plinth.

    #: How far inside its plot a sited footprint sits. Two, because a platform is laid
    #: one block wider than the footprint all round (that ledge is where a person stands
    #: to open the door) and `building()` refuses a footprint that leaves the plot.
    SITE_INSET = 2

    #: Relief across a footprint that a plinth can absorb. Above it the ground is cut
    #: and filled to one level instead. Three is one storey's worth of step: at four a
    #: person is climbing the outside of the building to get in at the door.
    SITE_RELIEF = 3

    #: The fraction of a footprint that has to be standing water before the answer is a
    #: deck on piles rather than a plinth poured into a lake.
    SITE_WET = 0.25

    #: The largest pad `site()` will cut out of a plot on broken ground. On a plot with
    #: 24 blocks of relief, levelling the whole of it is a quarry; the flattest
    #: rectangle this size in it is a terrace. On level ground the whole inset plot is
    #: used.
    SITE_PAD = 13

    #: The relief a *pad* may carry before it is worth looking for a smaller and calmer
    #: one. Well above `SITE_RELIEF`, because a platform is a good answer to a slope and
    #: the point of a big plot is a big building: shrinking a 12x23 plot to 7x7 to avoid
    #: five blocks of fill throws the plot away to save the podium.
    SITE_PAD_RELIEF = 8

    #: The four kinds of part, and the four shapes of ground under them. A plot is a
    #: building's ground, an **edge** is a wall's, a **point** is a gate's and an
    #: **area** is a square's. A place is not a list of buildings -- a wall is an edge,
    #: a gate is a point, a market square is an area -- and until this the type layer
    #: could only say "a building on a rectangle".
    PART_KINDS = ("plot", "edge", "point", "area")

    #: The pad a point stands on, when the part does not say. Five, so a gatehouse has a
    #: course to stand on outside its own footprint on every side.
    SITE_POINT = 5

    #: **A gate is sized by its wall.** Demo-polish, 2b. A point's pad was the type's
    #: minimum whatever it stood in, so the demo's outer gate was a 5x5 pad in a wall of
    #: forty-eight and the wall was split full-height round a twelve-high gatehouse. The
    #: pad along the wall is the wall's height over this: forty-eight gives sixteen,
    #: thirty-six twelve, twenty six -- and it is made odd, so the anchor is a centre
    #: cell, and held to the type's band. Registered before any gate was stood.
    POINT_PAD_PER_HEIGHT = 3

    @classmethod
    def point_pad(cls, height: int | None, band=None) -> int:
        """The side of the pad a point on an edge `height` high stands on. See
        `POINT_PAD_PER_HEIGHT`. `band` is the type's declared footprint
        (lo_w, lo_d, hi_w, hi_d); the pad never leaves it."""
        n = cls.SITE_POINT
        if height:
            n = max(n, int(height) // cls.POINT_PAD_PER_HEIGHT)
        if band:
            hi = int(min(band[2], band[3]))
            n = max(int(min(band[0], band[1])), min(n, hi if hi % 2 else hi - 1))
        if n % 2 == 0:
            n -= 1
        return max(3, n)

    # `clear_trees` takes the canopy and `_site_lay`'s cut takes the bank, and both
    # leave the subsoil they exposed lying on the surface. `Primitives.BARE_GROUND` is
    # what it puts back over.

    def _dress_worked(self, before: set, box: tuple, cover=None) -> int:
        """`before` is the pending set as it stood when the call started, so the columns
        are the ones the call is answerable for and nobody else's. `cover` defaults, as
        `dress_ground`'s always has, to whatever the undisturbed ground around the box
        is made of -- so this works on grass, sand, podzol or terracotta without being
        told which site it is on."""
        # Never a lane cell: the columns the circulation pass owns are the town's, which
        # is the rule `approach()` states in as many words and `_site_lay` skips on.
        # `clear_trees` is not lane-aware -- it walks a tree outward from its trunk --
        # so a lane column can be in the changed set, and putting grass back on a
        # cobbled lane would be the pad going over the street by a different route.
        lanes = self._site_lanes()
        cols = sorted({(p[0], p[2]) for p in set(self._pending) - before}
                      - lanes)
        if not cols:
            return 0
        n = self.dress_ground(box[0], box[1], box[2], box[3], cover=cover,
                              columns=cols, worked=True)
        return n + self._restore_stripped(cols, lanes)

    def _restore_stripped(self, cols: list, lanes: set) -> int:
        """Put back the ground `clear_ground_cover` took, at the level it took it from.

                **The ground pass does not leave a column lower than it found it.** See
                `prims.clear_ground_cover` for what it was taking and why re-skinning the
                subsoil was not enough. Three columns are left alone, and each is a column
                whose level is somebody's decision rather than an accident: a column `site()`
                laid a pad on (the pad *is* the ground there now, cut or filled), a lane
                column, and any column this pass has built something in at or above the block
                it took -- a wall standing where the turf was is not turf missing.
                
        """
        if not self.stripped:
            return 0
        put = 0
        sited = getattr(self, "_sited", None) or {}
        for (x, z) in cols:
            got = self.stripped.get((x, z))
            if got is None or (x, z) in sited or (x, z) in lanes:
                continue
            y, block = got
            if any((self._pending.get((x, h, z)) or "air").split("[")[0] not in AIR
                   for h in range(y, y + 3)):
                continue
            if (self.get_block(x, y, z) or "air").split("[")[0] not in AIR:
                continue
            self.place_block(x, y, z, block)
            put += 1
            if getattr(self, "dressed", None) is not None:
                self.dressed.add((x, z))
        return put

    # --- the palette, as shapes rather than block names. ---- A type says a role and a
    # shape; these say which block. Both are on the Builder and so both reach a type
    # through `TypeBuilder`, which is the point: the 244 block literals that welded the
    # fourteen committed types to one palette are all one of these two calls.

    @staticmethod
    def block(mat: str, kind: str = "full") -> str:
        """The block of one shape of a material family. Refuses by name. See `shape()`."""
        from .prims import shape
        return shape(mat, kind)

    @staticmethod
    def joinery(mat, kind: str = "door") -> str:
        """The door, fence, trapdoor or gate of a voice. Never refuses. See `joinery()`."""
        return joinery(mat, kind)

    def _part_ground(self, part: dict, mat) -> str:
        """A ring round the part's own rectangle rather than the rectangle itself: the
                columns inside it are about to become a plinth or a platform, and what a garden
                wants to be planted in is what the ground **beside** it is.
                
        """
        r = None
        if all(k in part for k in ("x0", "z0", "x1", "z1")):
            r = (int(min(part["x0"], part["x1"])), int(min(part["z0"], part["z1"])),
                 int(max(part["x0"], part["x1"])), int(max(part["z0"], part["z1"])))
        elif part.get("at"):
            at = part["at"]
            r = (int(at[0]) - 2, int(at[-1]) - 2, int(at[0]) + 2, int(at[-1]) + 2)
        elif part.get("path"):
            xs = [int(c[0]) for c in part["path"]]
            zs = [int(c[1]) for c in part["path"]]
            r = (min(xs), min(zs), max(xs), max(zs))
        if r is None:
            return _solid(_ground_role(mat) or "grass_block")
        x0, z0, x1, z1 = r
        ring = [(x, z) for x in range(x0 - 2, x1 + 3) for z in (z0 - 2, z1 + 2)]
        ring += [(x, z) for z in range(z0 - 2, z1 + 3) for x in (x0 - 2, x1 + 2)]
        return self.setting_cover(ring, mat)["cover"]

    def site(self, part: dict, *, mat=None, roof=None) -> dict:
        """Prepare the ground a part stands on, hand back the floor, and dress it.

                **And hand the type the voice.** `part['voice']` is the six material roles of
                the settlement -- wall, footing, frame, roof, trim, floor -- and every material a
                type places comes through it. Written here because this is
                already the one call that stands between the plan and `build()`, and because the
                ground work is faced in the same palette: a stone platform under a building whose
                footing is stone is a plinth, and under one whose footing is something else it is
                somebody else's building. `part['roof']` is the voice's silhouette, as `roof()`'s
                own four parameters, for a type that wants to take that from the voice too.

                **The ground is settled before it is laid.** v2, B1. What this part needs of the
                ground -- its pad and its level, or a wall's footing along its run -- is a
                declaration of the ground contract (`ethoslm.ground`), and what is laid here is
                the contract's resolution of it: the level held within reach of the part's own
                ground as found and never below the waterline, whoever declared. A build hands
                every part the one resolution it settled before the first block (`self.ground`);
                a bare call with none declares this one part and resolves it here, on the ground
                as found, through the same resolver. Either way the record on `sited` says which.
                
        """
        from . import ground as _ground
        before = set(self._pending)
        # **The seventh role. Read here and nowhere else, because this is the last
        # moment the ground under the part is the ground as found -- `_site_part` lays
        # the pad in the footing family the moment after. The voice's `ground` where it
        # names one, the setting's surface where it does not.
        ground = self._part_ground(part, mat)
        label = part.get("label")
        res = self.ground
        decl = res.declaration(label) if (res is not None and label) else None
        settled = "the build's contract"
        if decl is None:
            one = _ground.Contract()
            dec = self.declare(part, mat=mat, contract=one)
            if dec.get("ok") and len(one):
                res = one.resolve(self.bed, self.wet, relief=self.SITE_RELIEF)
                decl = res.declaration(label)
                settled = "a contract of this one part, on the ground as found"
        decision = None
        if decl is not None and res is not None:
            decision = dict(decl.decision)
            lvl = res.level_of(label)
            if lvl is not None:
                decision["level"] = int(lvl)
            decision["contract"] = {"settled_by": settled, "class": decl.cls,
                                    "kind": decl.kind, "columns": len(decl.columns),
                                    "won": len(res.columns_of(label)),
                                    "clamped": res.clamped(label),
                                    "seams": {f"{a}|{b}": dict(v) for (a, b), v
                                              in res.seams_of(label).items()}}
        out = self._site_part(part, mat=mat, decision=decision)
        out["voice"] = {**_mat_roles(mat), _GROUND_ROLE: ground}
        alt = mat.get(_WALL_ALT) if isinstance(mat, dict) else None
        if alt:
            out["voice"][_WALL_ALT] = str(alt)
        out["roof"] = dict(roof) if roof else None
        x0, z0, x1, z1 = (out.get("x0"), out.get("z0"), out.get("x1"), out.get("z1"))
        if x0 is not None:
            out.setdefault("sited", {})["swept"] = self._sweep_hanging(before)
            out["sited"]["dressed"] = self._dress_worked(
                before, (int(x0), int(z0), int(x1), int(z1)))
        out["way"] = self._way_in(out)
        return out

    def _way_in(self, part: dict) -> list:
        """The cells that have to stay open for a person to get in at this part's door.

                B2. `site()` has just levelled the doorstep, laid whatever
                path the ground needed and checked -- against the world as it now stands --
                that a person can walk from the lane to the door. Every one of those columns is
                a decision the library has already made and measured, and the type that is
                about to be handed the part has no business in any of them. Four halls in a
                city walled their own way in and the readout called it four unreachable doors
                and four rooms nobody can walk into; `TypeBuilder` holds these open now, so the
                class cannot come back through a different type.

                Two cells per column -- the one a person stands in and the one their head is in
                -- read off the world as the library leaves it, so a path that climbs is held
                open along its own slope rather than at one level.
                
        """
        door = ((int(part["door"][0]), int(tuple(part["door"])[-1]))
                if part.get("door") else None)
        cols = [door] if door else []
        # The way in that was walked rather than laid stops at the pad's edge: the pad
        # is the type's floor to lay, and a square paving its own ground over a route
        # held open would have the paving put back out from under its lamps.
        inside = None
        if all(part.get(k) is not None for k in ("x0", "z0", "x1", "z1")):
            inside = (int(part["x0"]), int(part["z0"]), int(part["x1"]), int(part["z1"]))
        for p in self.paths:
            if p.get("label") == part.get("label") and p.get("kind") == "approach":
                for c in p["cells"]:
                    cx, cz = int(c[0]), int(c[1])
                    if inside and inside[0] <= cx <= inside[2] and \
                            inside[1] <= cz <= inside[3] and (cx, cz) != door:
                        continue
                    cols.append((cx, cz))
        out = []
        for (x, z) in sorted(set(cols)):
            # Through what this call has placed, not off the world's stale heightmap: a
            # jetty over a lake is entirely `_pending` at this point, and reading
            # `get_height` gave the lake bed -- seven blocks under the path it is
            # supposed to be holding open. Same reason `dress_ground(worked=True)` reads
            # `_worked_top`.
            g = self._worked_top(x, z)
            out += [[x, g + 1, z], [x, g + 2, z]]
        return out

    #: How far past the columns a pass changed the hanging sweep looks, and how far past
    #: *that* it reads for support. A vine hangs from a leaf and the leaf is on a branch
    #: that reaches in from a tree standing outside the plot, so support is computed
    #: over a wider volume than the sweep touches -- otherwise the sweep would take the
    #: tree's own canopy for want of being able to see its trunk.
    SWEEP_MARGIN = 2
    SWEEP_SUPPORT = 10

    def _sweep_hanging(self, before: set) -> int:
        """Not inside `clear_trees`. This is the ground pass, and the ground pass is the
        library's.

                **Differential**, for the reason `_site_sweep` only takes small pieces and E010
                only counts what is on a plot: a jungle has plants the world generated holding
                nothing, and taking those would be demolition rather than tidying. What this
                removes is what *this pass* orphaned -- unsupported after, minus unsupported
                before.
        """
        from . import observe
        cols = {(p[0], p[2]) for p in set(self._pending) - before}
        if not cols:
            return 0
        m, s = self.SWEEP_MARGIN, self.SWEEP_SUPPORT
        x0, x1 = min(c[0] for c in cols), max(c[0] for c in cols)
        z0, z1 = min(c[1] for c in cols), max(c[1] for c in cols)
        box = (x0 - m - s, z0 - m - s, x1 - x0 + 1 + 2 * (m + s),
               z1 - z0 + 1 + 2 * (m + s))
        region = (x0 - m, z0 - m, x1 + m, z1 + m)
        was = self._world_volume().sub(*box)
        already = set(observe.unsupported_vegetation(was, region=region))
        now = self._world_volume().sub(*box)
        now.overlay({p: b for p, b in self._pending_view().items()
                     if box[0] <= p[0] < box[0] + box[2]
                     and box[1] <= p[2] < box[1] + box[3]})
        gone = 0
        for pos in observe.unsupported_vegetation(now, region=region):
            if pos in already:
                continue
            self.place_block(pos[0], pos[1], pos[2], "air")
            gone += 1
        return gone

    # --- the ground contract: decide, then lay ------------------------------ v2, B1.
    # `site()` used to decide and lay in one breath, and the deciding half read the
    # volume as the parts before it had left it. The deciding half is `declare()` now:
    # what this part needs of the ground, read off the ground as found and handed to
    # `ground.Contract`, which settles every part's columns at once, before a block is
    # placed. The laying half reads the settled decision and lays it.

    def declare(self, part: dict, *, mat=None, contract=None) -> dict:
        """What this part needs of the ground, read off the ground as found.

        The deciding half of `site()`: a plot's pad -- the flattest rectangle inside its
        plot -- and its level (the doorstep's, held within reach of the pad's own
        ground; the waterline over water); a point's pad round its anchor; an area's
        rectangle; an edge's footing, a level per segment along its run. Returns the
        decision -- `kind`, `rect` or `segments`, `level`, `ground`, `grade`, `relief`,
        the threshold it followed -- and, where `contract` is given, declares it there:
        a **platform** for a pad, its ledge with it, in the `footprint` class (a square
        or a court is `designed` ground, a field a `field`); a **profile** for an edge.
        The columns the circulation pass owns are never declared: a pad is allowed to
        be near somebody else's front door and is not allowed to be on top of it.
        Refuses, laying nothing, exactly where `site()` refused."""
        kind = part.get("kind", "plot")
        if kind == "edge":
            dec = self._decide_edge(part)
        elif kind == "point":
            at = part.get("at") or [part.get("x0"), part.get("z0")]
            ax, az = int(at[0]), int(at[-1])
            size = max(3, int(part.get("size", self.SITE_POINT)))
            h = (size - 1) // 2
            dec = self._decide_rect(part, (ax - h, az - h, ax + h, az + h), kind="point")
            dec["at"] = [ax, az]
        elif kind == "area":
            x0, x1 = int(min(part["x0"], part["x1"])), int(max(part["x0"], part["x1"]))
            z0, z1 = int(min(part["z0"], part["z1"])), int(max(part["z0"], part["z1"]))
            dec = self._decide_rect(part, (x0, z0, x1, z1), kind="area")
        elif kind in ("plot", None):
            px0, px1 = int(min(part["x0"], part["x1"])), int(max(part["x0"], part["x1"]))
            pz0, pz1 = int(min(part["z0"], part["z1"])), int(max(part["z0"], part["z1"]))
            dec = self._decide_rect(part, self._site_pad(px0, pz0, px1, pz1,
                                                         part.get("attached")),
                                    kind="plot")
        else:
            dec = {"ok": False, "kind": kind,
                   "reason": f"site() prepares the ground for a "
                             f"{', '.join(self.PART_KINDS)}; it does not know what a "
                             f"{kind!r} is"}
        if contract is not None and dec.get("ok"):
            self._declare_into(contract, part, dec)
        return dec

    def _decide_rect(self, part: dict, rect: tuple, *, kind: str) -> dict:
        """The level of a rectangle of ground, decided as `site()` always has."""
        x0, z0, x1, z1 = rect
        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        water = {c: self.wet(c[0], c[1]) for c in cols}
        wet_cols = [c for c in cols if water[c] is not None]
        relief = max(bed.values()) - min(bed.values())
        fr = self.floor_from_threshold(part.get("label"),
                                       (x0 + x1) // 2, (z0 + z1) // 2) or {}
        want = fr.get("floor_y") if fr.get("source") == "threshold" else None
        waterline = max(water[c] for c in wet_cols) if wet_cols else None
        if len(wet_cols) > self.SITE_WET * len(cols):
            ground = "deck"
            floor_y = waterline + 1
            floor_y = max(floor_y, want) if want is not None else floor_y
        elif relief > self.SITE_RELIEF:
            ground = "platform"
            floor_y = max(bed.values()) if want is None else self._within_reach(want, bed)
        else:
            ground = "plinth"
            floor_y = max(bed.values()) if want is None else self._within_reach(want, bed)
        return {"ok": True, "kind": kind, "rect": [int(v) for v in rect],
                "level": int(floor_y), "ground": ground, "relief": int(relief),
                "wet_columns": len(wet_cols), "columns": len(cols),
                "grade": [int(min(bed.values())), int(max(bed.values()))],
                "waterline": (int(waterline) if waterline is not None else None),
                "want": (int(want) if want is not None else None),
                "threshold": {"source": fr.get("source"), "facing": fr.get("facing"),
                              "door": (list(fr["door"]) if fr.get("door") else None)}}

    #: The area types that are **designed** ground -- levelled for what happens on them
    #: -- against the ones that are a field: what is left between the plots, planted. A
    #: declaration's class, and the contract's precedence.
    DESIGNED_AREAS = ("square", "plaza", "court", "market")

    def _declare_into(self, contract, part: dict, dec: dict) -> None:
        """This part's decision, as declarations of the contract."""
        from . import ground as _ground
        label = part.get("label") or part.get("name") or f"part_{len(self.parts)}"
        lanes = self._site_lanes()
        if dec["kind"] == "edge":
            floor_of = {tuple(c): y for c, y in dec["floor_of"]}
            contract.profile(label, {c: y for c, y in floor_of.items() if c not in lanes},
                             cls="footprint",
                             rect=(dec["x0"], dec["z0"], dec["x1"], dec["z1"]),
                             reason=dec["reason"], decision=dec)
            return
        x0, z0, x1, z1 = dec["rect"]
        tname = str(part.get("type") or "")
        if dec["kind"] == "area":
            cls = ("designed" if any(tname.startswith(w) for w in self.DESIGNED_AREAS)
                   else "field")
        else:
            cls = "footprint"
        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        cols += _ground.ledge((x0, z0, x1, z1))
        contract.platform(label, [c for c in cols if c not in lanes], dec["level"],
                          cls=cls, rect=(x0, z0, x1, z1),
                          reason=(f"{dec['ground']} at y={dec['level']} over ground "
                                  f"y={dec['grade'][0]}..{dec['grade'][1]}, "
                                  f"{dec['wet_columns']} of {dec['columns']} columns wet"
                                  + (f"; the doorstep at y={dec['want']}"
                                     if dec.get("want") is not None else "")
                                  + ", and the ledge one column out"),
                          decision=dec)

    def _site_part(self, part: dict, *, mat=None, decision: dict | None = None) -> dict:
        """Prepare the ground a part stands on, and say where its floor is.

        Called by the driver **before** the type's `build()`, once per instance. The
        type is then handed the part back with `floor_y`, the sited `footprint` as its
        own rectangle, and the cell its door goes in -- and builds from `floor_y` up,
        touching no ground at all. See `TypeBuilder` for the half that enforces that.

        `decision` is the contract's settled decision for this part (`declare()`'s
        record with the resolved `level`); without one the part is decided here, on the
        ground as found, as it always was. For a plot, in the order the measurement
        found them: the pad, then `approach()`, from the doorway the circulation pass
        reserved to the lane, over the ground this call has just made -- so the pad is
        walk-reachable from the lane *before* the type has placed a block, and every
        door the type then puts on it is reachable too.

        Returns the part: the same dict with `x0, z0, x1, z1` narrowed to the sited
        footprint, plus `floor_y`, `door`, `facing`, `ground` (which of the three) and
        `sited` (what was laid, and what the ground was)."""
        kind = part.get("kind", "plot")
        m = _mat_roles(mat)
        if kind == "edge":
            return self._site_edge(part, m, decision)
        if kind == "point":
            return self._site_point(part, m, decision)
        if kind == "area":
            return self._site_area(part, m, decision)
        if kind not in ("plot", None):
            return {**part, "ground": "unsited",
                    "sited": {"ok": False,
                              "reason": f"site() prepares the ground for a "
                                        f"{', '.join(self.PART_KINDS)}; it does not "
                                        f"know what a {kind!r} is"}}
        px0, px1 = int(min(part["x0"], part["x1"])), int(max(part["x0"], part["x1"]))
        pz0, pz1 = int(min(part["z0"], part["z1"])), int(max(part["z0"], part["z1"]))
        dec = decision or self._decide_rect(part, self._site_pad(px0, pz0, px1, pz1,
                                                                 part.get("attached")),
                                            kind="plot")
        rect = tuple(int(v) for v in dec["rect"])
        x0, z0, x1, z1 = rect
        self.clear_trees(x0 - 2, z0 - 2, x1 + 2, z1 + 2)
        # B2: this clearing is the library's own, so what it takes off the ground is
        # written down and `_dress_worked` puts the level back. See
        # `prims.clear_ground_cover` and `_restore_stripped`.
        self._library_ground = True
        try:
            self.clear_ground_cover(x0 - 2, z0 - 2, x1 + 2, z1 + 2)
        finally:
            self._library_ground = False
        self._site_sweep(px0, pz0, px1, pz1)

        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        water = {c: self.wet(c[0], c[1]) for c in cols}
        wet_cols = [c for c in cols if water[c] is not None]
        relief = int(dec["relief"])
        floor_y = int(dec["level"])
        ground = dec["ground"]
        laid = self._site_lay(rect, floor_y, bed, water, m)

        fr = self.floor_from_threshold(part.get("label"),
                                       (x0 + x1) // 2, (z0 + z1) // 2) or {}
        facing = fr.get("facing") or "north"
        door, why = _door_cell(facing, fr.get("door"), rect, [])
        # The pad is the ground now, and the doorstep is on it. Written down before the
        # approach is laid, so that `approach()`, `building()` and every instrument that
        # asks where this structure's floor goes get the same answer -- and so that
        # `plinth(to_grade)` inside the shell that follows fills nothing.
        lanes = self._site_lanes()
        for x in range(x0 - 1, x1 + 2):
            for z in range(z0 - 1, z1 + 2):
                if (x, z) not in lanes:           # nothing was laid there; see _site_lay
                    self._sited[(x, z)] = int(floor_y)
        if self.frontage is not None and part.get("label") and door is not None:
            # (x, y, z), as a threshold's door has always been -- `_door_cell` reads
            # index 0 and index 2 off it and a two-element door is an IndexError in the
            # middle of `building()`.
            self.frontage.sited[part["label"]] = {
                "floor_block_y": int(floor_y), "stand_y": int(floor_y) + 1,
                "door": [door[0], int(floor_y) + 1, door[1]]}
        ap = ({"ok": False, "cells": 0, "reason": why} if door is None else
              self.approach(part.get("label"), door[0], floor_y + 1, door[1]))
        out = {**part, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
                "floor_y": int(floor_y), "footprint": [x0, z0, x1, z1],
                "door": list(door) if door else None, "facing": facing,
                "ground": ground,
                "sited": {"ok": bool(ap.get("ok")), "relief": int(relief),
                          "wet_columns": len(wet_cols), "columns": len(cols),
                          "grade": list(dec["grade"]),
                          "laid": laid, "approach": ap,
                          **({"contract": dec["contract"]} if dec.get("contract") else {}),
                          "reason": (f"{ground} at y={floor_y} over ground "
                                     f"y={dec['grade'][0]}..{dec['grade'][1]}, "
                                     f"{dec['wet_columns']} of {dec['columns']} columns wet; "
                                     + str(ap.get("reason")))}}
        self.parts.append(out)
        return out

    def _within_reach(self, want: int, bed: dict) -> int:
        """The wanted floor where it is within `SITE_RELIEF` of the pad's own ground, else
        the pad's ground.

                A doorstep the circulation pass levelled decides the floor, and measured
                across the fixtures it lies one below to two above the pad's ground:
                the lane is cut into the slope. That is followed, as it always was. A doorstep
                recorded eighteen blocks under the pad -- a courtyard house in the concentric
                run stood at y=38 over ground 56..62, and its sited record then became the
                ground its neighbour was sited to -- is not this pad's doorstep, and the floor
                stops at the pad's ground: its lowest column below, its highest above. Over
                water the waterline already does this (`floor_y = max(floor_y, want)`).
                
        """
        lo, hi = min(bed.values()), max(bed.values())
        if want < lo - self.SITE_RELIEF:
            return int(lo)
        if want > hi + self.SITE_RELIEF:
            return int(hi)
        return int(want)

    #: The pad sizes tried, biggest first: the whole inset plot, then the flattest
    #: `SITE_PAD` of it, then nine, then seven. The first whose ground is level enough
    #: to plinth wins; if none is, the smallest and flattest is what gets platformed.
    #: Biggest first because a bigger pad is more room for a type to vary its massing
    #: in, and smallest last because a 7x7 terrace on a hillside is a building and a
    #: 15x15 one is a quarry.
    _SITE_SIZES = (None, "SITE_PAD", 9, 7)

    @classmethod
    def pad_extent(cls, part: dict) -> tuple:
        """(w, d) of the **largest** pad `site()` can hand `build()` for this part.

        Answered off the geometry alone, with no world read, which is what lets a plan
        be validated before a block is placed and a part be refused before it is sited.
        It is an **upper bound**: `_site_pad` may take a smaller and flatter rectangle
        on broken ground, so a pad that fails this could never have fitted, and one that
        passes it may still come back smaller. For an edge the pair is (swept width,
        the run of its **longest segment**), because an edge's ground is a line and
        `_site_edge` hands the type one level per segment: what a wall has to build in
        one go is a segment, not the polyline."""
        kind = part.get("kind", "plot")
        if kind == "edge":
            path = [(int(p[0]), int(p[1])) for p in (part.get("path") or [])]
            run = max((abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1
                       for a, b in zip(path, path[1:])), default=0)
            return (max(1, int(part.get("width", 1))), run)
        if kind == "point":
            n = max(3, int(part.get("size", cls.SITE_POINT)))
            return (n, n)
        w = abs(int(part["x1"]) - int(part["x0"])) + 1
        d = abs(int(part["z1"]) - int(part["z0"])) + 1
        if kind == "area":
            return (w, d)
        # A plot, and `_site_pad`'s own two cases: inset by `SITE_INSET` on every side,
        # and by one where two would leave less than five columns across -- and, v2 C2,
        # by nothing on a side the plan says is **attached**.
        ix0, iz0, ix1, iz1 = cls._insets(w, d, part.get("attached"))
        return (max(0, w - ix0 - ix1), max(0, d - iz0 - iz1))

    @classmethod
    def _insets(cls, w: int, d: int, attached=None) -> tuple:
        """(west, north, east, south): how far a plot's pad is inset on each side.
        `SITE_INSET` on a free side, one where two would leave less than five columns
        across, and **nothing on an attached side** (v2, C2): the next house stands
        against it and the party wall is on the plot's edge."""
        a = set(attached or ())
        i = cls.SITE_INSET
        ins = [0 if s in a else i for s in ("west", "north", "east", "south")]
        if (w - 1 - ins[0] - ins[2]) < 4 or (d - 1 - ins[1] - ins[3]) < 4:
            ins = [min(v, 1) for v in ins]
        return tuple(ins)

    def _site_pad(self, px0: int, pz0: int, px1: int, pz1: int, attached=None) -> tuple:
        """The rectangle inside a plot that gets prepared: inset, and flattest if broken."""
        w, d = px1 - px0 + 1, pz1 - pz0 + 1
        ix0, iz0, ix1, iz1 = self._insets(w, d, attached)
        # Too small to inset by two, but **never inset by none** on a free side.
        x0, z0 = px0 + ix0, pz0 + iz0
        x1, z1 = px1 - ix1, pz1 - iz1
        full = (x1 - x0 + 1, z1 - z0 + 1)
        sizes: list = []
        for cap in self._SITE_SIZES:
            n = getattr(self, cap) if isinstance(cap, str) else cap
            wd = full if n is None else (min(full[0], n), min(full[1], n))
            if wd not in sizes and wd[0] >= 3 and wd[1] >= 3:
                sizes.append(wd)
        best = (x0, z0, x1, z1)
        for (w, d) in sizes:
            if (w, d) == full:
                r = (x0, z0, x1, z1)
            else:
                got = self.flattest_rect(x0, z0, x1, z1, w, d, step=1)
                if got is None:
                    continue
                r = (got[0], got[1], got[0] + w - 1, got[1] + d - 1)
            hs = [self.bed(x, z) for x in range(r[0], r[2] + 1)
                  for z in range(r[1], r[3] + 1)]
            best = r
            if max(hs) - min(hs) <= self.SITE_PAD_RELIEF:
                break
        return best

    #: The biggest floating piece the ground sweep will take. A seed pod is one cell and
    #: an orphaned clump is a few; anything bigger is somebody's build or a rock arch,
    #: and taking it would be demolition rather than sweeping.
    SITE_SWEEP = 8

    def _site_sweep(self, px0: int, pz0: int, px1: int, pz1: int) -> int:
        """Take the vegetation the world left standing on nothing, over the whole plot."""
        from . import observe
        sub = self._world_volume().sub(px0, pz0, px1 - px0 + 1, pz1 - pz0 + 1)
        sub.overlay({p: b for p, b in self._pending_view().items()
                     if px0 <= p[0] <= px1 and pz0 <= p[2] <= pz1})
        gone = 0
        for piece in observe.unsupported(sub):
            if piece["cells"] > self.SITE_SWEEP:
                continue
            if not observe._is_vegetation(piece["example"]):
                continue
            a, b, c, d, e, f = piece["bbox"]
            for x in range(a, d + 1):
                for y in range(b, e + 1):
                    for z in range(c, f + 1):
                        if observe._is_vegetation(sub.state(x, y, z)):
                            self.place_block(x, y, z, "air")
                            gone += 1
        return gone + self._site_canopy(px0, pz0, px1, pz1)

    def _site_canopy(self, px0: int, pz0: int, px1: int, pz1: int) -> int:
        """A canopy whose trunk is already gone, standing over the plot we are preparing.

                The one canopy that *is* ours is the one over the pad we are about to build on,
                and the rule is the rule `clear_trees` already states: a tree comes out whole.
                So support is read `SWEEP_SUPPORT` past the plot -- a canopy still held by a
                trunk standing just outside it is not floating and is not touched -- and what
                is taken is the whole connected cluster of anything orphaned that reaches over
                the plot, out to the edge of that same window, because half a canopy taken is
                the floating-leaves giveaway again with the other half left hanging.
                
        """
        from . import observe
        s = self.SWEEP_SUPPORT
        w, d = px1 - px0 + 1 + 2 * s, pz1 - pz0 + 1 + 2 * s
        sub = self._world_volume().sub(px0 - s, pz0 - s, w, d)
        sub.overlay({p: b for p, b in self._pending_view().items()
                     if px0 - s <= p[0] <= px1 + s and pz0 - s <= p[2] <= pz1 + s})
        loose = set(observe.unsupported_vegetation(sub))
        if not loose:
            return 0
        over = {p for p in loose if px0 <= p[0] <= px1 and pz0 <= p[2] <= pz1}
        take: set = set()
        stack = sorted(over)
        while stack:
            p = stack.pop()
            if p in take:
                continue
            take.add(p)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        q = (p[0] + dx, p[1] + dy, p[2] + dz)
                        if q in loose and q not in take:
                            stack.append(q)
        for (x, y, z) in sorted(take):
            self.place_block(x, y, z, "air")
        return len(take)

    def bed(self, x: int, z: int) -> int:
        """The y of the **ground** under a column: not the waterline, not the canopy."""
        from . import observe
        y = self.grade(int(x), int(z))
        for _ in range(self.MAX_SOUNDING):
            if not observe._is_vegetation(self._ground_block(int(x), y, int(z))):
                return y
            y -= 1
        return y

    def _site_lanes(self) -> set:
        """The columns the circulation pass owns, which are not the pad's.

        Every threshold, not only this part's. A pad is allowed to be near somebody
        else's front door and is not allowed to be on top of it."""
        net = self.frontage.net if self.frontage is not None else None
        if net is None:
            return set()
        keep = {(lx, lz) for (lx, lz) in net.cells}
        for t in net.thresholds:
            keep.add((t.x, t.z))
            keep.add((int(t.door[0]), int(t.door[2])))
        return keep

    # --- the ground a *place* stands on. `site()` prepares the ground one part stands
    # on, and that is the right size of thing for a house: a pad, inset in its plot,
    # decided per part. It is the wrong size of thing for a place. A walled town with a
    # market square and a keep wants one piece of level ground its centre sits on, and
    # until now the only way to get it was for the site search to find one -- so a
    # sentence that asked for a town was answerable only where the world happened to
    # have already built the shelf for it. `plateau()` is the library making that ground
    # instead of looking for it. It is `terrace()`'s shape.

    #: The **least** a plateau's retaining face is carried below its own level. It is
    #: always carried to grade as well -- a face that stopped short left a cavity
    #: between its own bottom course and the hillside, which `scripts/test_place.py`'s
    #: A4 case read as "a room of 78 cells cannot be walked into" the first time it ran.
    #: This is the minimum for the case the fall is shallow: a face shorter than four
    #: reads as a kerb rather than as retaining.
    PLATEAU_FACE = 4

    #: The **least** wide the feathered ring outside a plateau is, as `terrace()` takes
    #: it. The ring is widened to the fall it has to cover, so that no column of it
    #: rises more than one block over the one outside it -- which is the difference
    #: between a slope a person walks up and a slope a person cannot. A plateau is cut
    #: before the circulation pass exists, so at the moment it is laid the feather is
    #: the only way onto it and `approach()` has no lane to reach for.
    PLATEAU_FEATHER = 6

    #: ...and the widest, because a feather is a blend and not a landscape. Beyond this
    #: the ring is bigger than the plateau it eases and the way on is the lane's job.
    PLATEAU_FEATHER_MAX = 24

    #: The largest plateau this call will cut, on a side. A plateau bigger than this is
    #: not a terrace, it is the site search having failed and the world being flattened
    #: to hide it -- and the spec bounds a plateau to the innermost defining part.
    PLATEAU_MAX = 128

    #: ...and what that number **is**, which is a quarter of the side of the site it was
    #: registered against. 128 and 512 were registered together, and a city's ground
    #: ceiling is now 768, so a bound that does not move with it is a bound that says a
    #: bigger place must have a smaller middle. On a 512 site this is 128 exactly and
    #: nothing about any place already on the record moves.
    PLATEAU_MAX_SHARE = 0.25

    @classmethod
    def plateau_max(cls, site_side: int | None = None) -> int:
        """The largest plateau a place of this footprint may have cut in it."""
        if not site_side:
            return cls.PLATEAU_MAX
        return max(cls.PLATEAU_MAX, int(round(int(site_side) * cls.PLATEAU_MAX_SHARE)))

    def _plateau_landing(self, x0: int, z0: int, x1: int, z1: int, net) -> tuple:
        """The column of a plateau's edge the lane comes closest to. Deterministic.

                The edge and not the middle: the way on is laid from the perimeter, so a
                `approach()` that started in the centre would carve a ramp across the level
                ground the plateau exists to make.
                
        """
        edge = ([(x, z) for x in range(x0, x1 + 1) for z in (z0, z1)]
                + [(x, z) for z in range(z0 + 1, z1) for x in (x0, x1)])
        lanes = [(lx, lz) for (lx, lz, _ly) in net.surface()] or list(net.cells)
        best, key = edge[0], None
        for (ex, ez) in edge:
            d = min((ex - lx) ** 2 + (ez - lz) ** 2 for (lx, lz) in lanes)
            k = (d, ex, ez)
            if key is None or k < key:
                best, key = (ex, ez), k
        return best

    def plateau(self, area, y: int | None = None, *, mat=None,
                feather: int | None = None, label: str | None = None,
                bound: int | None = None) -> dict:
        """Level an area to `y`, retain it, feather it, sweep it and re-skin it. A4.

        Refuses by name and lays nothing where the area is bigger than `PLATEAU_MAX` on
        a side, or where it is smaller than three. Never writes on a lane cell or a
        reserved doorstep, for the reason `_site_lay` does not: those columns are the
        circulation pass's and a plateau over a threshold buries the way in.

        Returns what it did: the level, the relief it took out, the columns filled, cut
        and retained, what the feather touched, and the sweep and dress counts."""
        before = set(self._pending)
        part_label = label
        if isinstance(area, dict):
            x0 = int(min(area["x0"], area["x1"]))
            z0 = int(min(area["z0"], area["z1"]))
            x1 = int(max(area["x0"], area["x1"]))
            z1 = int(max(area["z0"], area["z1"]))
            part_label = label or area.get("label") or area.get("name")
        else:
            a, b, c, d = area
            x0, z0, x1, z1 = int(min(a, c)), int(min(b, d)), int(max(a, c)), int(max(b, d))
        w, dpt = x1 - x0 + 1, z1 - z0 + 1
        if w < 3 or dpt < 3:
            return {"ok": False, "reason": f"a plateau is at least 3x3 and this is "
                                           f"{w}x{dpt}", "columns": 0}
        # The bound is the place's where the caller knows it (`plateau_max`), and the
        # registered 128 where it does not: a quarter of the side of the site.
        cap = int(bound or self.PLATEAU_MAX)
        if w > cap or dpt > cap:
            return {"ok": False, "columns": 0,
                    "reason": f"a plateau is at most {cap} on a side and "
                              f"this is {w}x{dpt}; a place that needs more level ground "
                              f"than that needs a different site, not a bigger cut"}
        m = _mat_roles(mat)
        foot_full = _solid(m["footing"])
        lanes = self._site_lanes()
        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        water = {c: self.wet(c[0], c[1]) for c in cols}
        was = max(bed.values()) - min(bed.values())
        if y is None:
            y = int(sorted(bed.values())[len(bed) // 2])
        y = int(y)

        # The feather is as wide as the fall it has to cover, so the ramp off the edge
        # rises at most a block a column and a person can walk down it. Measured from
        # the ground just outside the ring rather than assumed.
        if feather is None:
            ring0 = ([(x, z) for x in range(x0 - 2, x1 + 3) for z in (z0 - 2, z1 + 2)]
                     + [(x, z) for z in range(z0 - 1, z1 + 2) for x in (x0 - 2, x1 + 2)])
            fall = max((abs(y - self.bed(x, z)) for (x, z) in ring0), default=0)
            f = int(min(self.PLATEAU_FEATHER_MAX,
                        max(self.PLATEAU_FEATHER, fall)))
        else:
            f = int(feather)

        # `clear_ground_cover` takes the turf off this same ring, and a cover sampled
        # after it has been taken is a cover sampled off a cut.
        rim = [(x, z) for i in range(1, f + 1)
               for (x, z) in ([(xx, zz) for xx in range(x0 - 1 - i, x1 + 2 + i)
                               for zz in (z0 - 1 - i, z1 + 1 + i)]
                              + [(xx, zz) for zz in range(z0 - i, z1 + 1 + i)
                                 for xx in (x0 - 1 - i, x1 + 1 + i)])]
        feather_cover = self.setting_cover(rim, mat)

        # Clear first, and over the feathered ring too.
        self.clear_trees(x0 - f, z0 - f, x1 + f, z1 + f)
        self.clear_ground_cover(x0 - f, z0 - f, x1 + f, z1 + f)

        filled = cut = decked = skipped = 0
        for (x, z) in cols:
            if (x, z) in lanes:
                skipped += 1
                continue
            b, wl = bed[(x, z)], water[(x, z)]
            if wl is not None and b < y:
                # Water inside the plateau is filled to the level, from the bed. This is
                # the one place a plateau differs from a deck: a deck is a floor over a
                # lake a building stands on, and a plateau is ground -- a market square
                # half in a pond is not a market square.
                for yy in range(b + 1, y + 1):
                    self.place_block(x, yy, z, foot_full)
                decked += 1
            elif b < y:
                for yy in range(min(b, y), y + 1):
                    self.place_block(x, yy, z, foot_full)
                filled += 1
            else:
                self.place_block(x, y, z, foot_full)
                filled += 1
            for yy in range(y + 1, max(y, b) + 5):
                if self.get_block(x, yy, z).split("[")[0] not in AIR:
                    self.place_block(x, yy, z, "air")
                    cut += 1
            self._sited[(x, z)] = y

        # The retaining face: the ring one column outside, carried down in the footing
        # **to grade**, and never less than `PLATEAU_FACE` courses. Without it a cut
        # plateau on a fall is a rectangle of ground with air under its low side; with a
        # face that stops short it is a rectangle of ground with a sealed room under its
        # low side, which is what a fixed depth gave and what A4's case caught.
        retained = 0
        ring = ([(x, z) for x in range(x0 - 1, x1 + 2) for z in (z0 - 1, z1 + 1)]
                + [(x, z) for z in range(z0, z1 + 1) for x in (x0 - 1, x1 + 1)])
        for (x, z) in ring:
            if (x, z) in lanes:
                skipped += 1
                continue
            b = self.bed(x, z)
            if b >= y:
                continue
            for yy in range(min(b, y - self.PLATEAU_FACE), y + 1):
                self.place_block(x, yy, z, foot_full)
            retained += 1
            self._sited[(x, z)] = y

        # ...and then `terrace()`'s own feather, which is the shape that stopped a hard
        # rectangular cut reading as a cut. Only outside the retaining ring, and never
        # on a lane: `terrace()` is not lane-aware, so the ring is walked here. the
        # ground look's fourth finding, at two scales. A feather is a blend between a
        # piece of designed ground and the land round it, so its top course is the
        # land's: `setting_cover` at the column, read before anything here is laid.
        feathered = 0
        cover_at = feather_cover["at"]
        for i in range(1, f + 1):
            t = i / (f + 1.0)
            edge = ([(x, z) for x in range(x0 - 1 - i, x1 + 2 + i)
                     for z in (z0 - 1 - i, z1 + 1 + i)]
                    + [(x, z) for z in range(z0 - i, z1 + 1 + i)
                       for x in (x0 - 1 - i, x1 + 1 + i)])
            for (x, z) in edge:
                if (x, z) in lanes:
                    continue
                g = self.bed(x, z)
                target = int(round(y * (1 - t) + g * t))
                if g > target:
                    for yy in range(target + 1, g + 4):
                        if self.get_block(x, yy, z).split("[")[0] not in AIR:
                            self.place_block(x, yy, z, "air")
                else:
                    for yy in range(g, target):
                        self.place_block(x, yy, z, foot_full)
                self.place_block(x, target, z, cover_at(x, z))
                feathered += 1

        # ...and the way onto it. A feathered edge is a ramp of about two blocks a
        # column and a person who cannot jump cannot climb one, so a plateau without
        # this is a level platform nobody can reach. `approach()` is what `site()` calls
        # for exactly this and it lays a flight: a slab where the way rises half a block
        # and a tread where it rises a whole one. From the edge column nearest the lane,
        # because that is the shortest climb and it is where a person actually arrives.
        ap = {"ok": True, "cells": 0, "reason": "no network: nothing to approach from"}
        net = self.frontage.net if self.frontage is not None else None
        if net is not None and net.cells:
            ex, ez = self._plateau_landing(x0, z0, x1, z1, net)
            ap = self.approach(part_label, ex, y + 1, ez)

        swept = self._sweep_hanging(before)
        dressed = self._dress_worked(before, (x0 - f - 1, z0 - f - 1,
                                              x1 + f + 1, z1 + f + 1))
        now = [self.bed(x, z) for (x, z) in cols if (x, z) not in lanes]
        return {"ok": True, "y": y, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
                "approach": ap,
                "columns": len(cols), "relief_before": int(was),
                "relief_after": int(max(now) - min(now)) if now else 0,
                "filled": filled, "flooded": decked, "cut": cut, "retained": retained,
                "feathered": feathered, "swept": swept, "dressed": dressed,
                "lane_columns_left_alone": skipped,
                "feather_cover": {k: v for k, v in feather_cover.items() if k != "at"},
                "reason": (f"a plateau of {w}x{dpt} at y={y}, {was} of relief taken to "
                           f"{max(now) - min(now) if now else 0}; {filled} filled, "
                           f"{cut} cut, {retained} retained, {feathered} feathered, "
                           f"{swept} swept, {dressed} dressed")}

    #: How many columns of a piece of designed ground are sampled to find what the
    #: setting's surface is. A few thousand is a reading of a 344,000-column annulus
    #: that cannot be wrong about which block is commonest and costs nothing.
    SETTING_SAMPLE = 4096

    def setting_cover(self, cols, mat=None) -> dict:
        """What a piece of designed ground is dressed in, column by column.

                The cover is the **setting's**, at the column:

                  * where the voice names a `ground` role it is that, everywhere -- a voice that
                    means its city to stand on dry paved stone can say so, and the prose `ground`
                    note stays prose;
                  * otherwise the column's own natural surface where that column has one, so a
                    terrace that crosses a shore is sand at the shore and grass inland;
                  * otherwise -- the column is water, or subsoil a cut exposed -- the commonest
                    natural surface over this piece of ground, which is what the setting is;
                  * and `grass_block` last, where the ground offers nothing at all.

                Read off the world **as found**, never off worked ground: that is the whole of
                the difference. Returns the per-column function, the fallback, and why.
                
        """
        voiced = _ground_role(mat)
        if voiced:
            block = _solid(voiced)
            return {"at": (lambda x, z: block), "cover": block, "from": "voice",
                    "why": f"the voice names its ground: {voiced}"}
        cols = list(cols)
        step = max(1, len(cols) // self.SETTING_SAMPLE)
        seen: dict = {}
        for (x, z) in cols[::step]:
            b = self.get_block(x, self.get_height(x, z), z).split("[")[0]
            if b in self.NATURAL_COVER or b.endswith("_terracotta"):
                seen[b] = seen.get(b, 0) + 1
        back = max(seen, key=lambda k: seen[k]) if seen else "grass_block"

        def at(x, z):
            b = self.get_block(x, self.get_height(x, z), z).split("[")[0]
            if b in self.NATURAL_COVER or b.endswith("_terracotta"):
                return b
            return back
        return {"at": at, "cover": back, "from": "setting",
                "sampled": len(cols[::step]), "found": dict(sorted(seen.items())),
                "why": (f"the natural surface at each column, {back} where it is water "
                        f"or a cut")}

    #: The six `plateau()` uses, widened to the fall it has to cover so no column rises
    #: more than a block over the one outside it.
    TERRACE_FEATHER = PLATEAU_FEATHER

    #: ...and the most blocks one terrace call may lay. `MAX_BLOCKS` is the builder's
    #: own ceiling and this is the terrace's, under it: a ring's annulus is laid as one
    #: call and committed, so a city of four rings is four bounded fills rather than one
    #: that the queue refuses at two million.
    TERRACE_MAX_BLOCKS = 1_500_000

    #: The four sides of a rectangle a terrace may face and feather.
    TERRACE_SIDES = ("north", "south", "west", "east")

    #: A merlon two columns wide and a crenel one between them. A parapet is a fact
    #: about walls and not about one wall type, so it is here and the types read it --
    #: `wall` drew its merlon two or three wide by seed and `great_wall` one, so two
    #: walls of one city crowned themselves differently and neither read as a
    #: battlement. Two to one is what a merlon is: wide enough to stand behind, narrow
    #: enough that the gap is a gap.
    MERLON_WIDTH = 2
    CRENEL_WIDTH = 1

    def merlon(self, i: int) -> bool:
        """Is the `i`th column of a parapet a merlon rather than a crenel?"""
        return (int(i) % (self.MERLON_WIDTH + self.CRENEL_WIDTH)) < self.MERLON_WIDTH

    def terrace_annulus(self, outer, y: int, *, inner=None, mat=None,
                        cover: str | None = None, label: str | None = None,
                        feather: int | None = None, sides=None) -> dict:
        """Level the ground between two rectangles to `y`, and dress the outer edge.

                `outer` and `inner` are `(x0, z0, x1, z1)`, corners inclusive; the annulus is
                every column of `outer` not in `inner` (all of `outer` with no `inner`). Each
                column is brought to `y`: filled from the bed in the footing family with
                `cover` on top -- the voice's ground, or the undisturbed ground round the
                annulus where none is given -- water inside filled from the bed, and what
                stands over `y` cut to headroom. The **outer** edge is retained one column out
                in the footing to grade, never less than `PLATEAU_FACE`, and feathered outside
                that as `plateau()` does: a slope of a block a column, dressed and never a bare
                cut. The inner edge is nothing: the ring inside stands higher and its own
                outer edge is the step. `sides` names which of `TERRACE_SIDES` of `outer` are
                faced and feathered -- all four by default; a strip of an annulus laid as one
                of several pieces names only the sides that are the annulus's own edge.

                Refuses, laying nothing, where the fill would exceed `TERRACE_MAX_BLOCKS`
                (the estimate is the sum over columns of the fill and cut each needs), and
                never writes on a lane cell or a reserved doorstep, as `plateau()` does not.
                Returns the level, the columns, and what was filled, flooded, cut, retained,
                feathered and dressed.
        """
        before = set(self._pending)
        x0, z0, x1, z1 = (int(min(outer[0], outer[2])), int(min(outer[1], outer[3])),
                          int(max(outer[0], outer[2])), int(max(outer[1], outer[3])))
        hole = None
        if inner is not None:
            hole = (int(min(inner[0], inner[2])), int(min(inner[1], inner[3])),
                    int(max(inner[0], inner[2])), int(max(inner[1], inner[3])))
        y = int(y)
        m = _mat_roles(mat)
        foot_full = _solid(m["footing"])
        lanes = self._site_lanes()
        sides = set(self.TERRACE_SIDES if sides is None else sides)

        def rim(i):
            """The columns `i` outside the outer rectangle on the sides named, the
            corners with either side that meets them."""
            out = set()
            if "north" in sides:
                out.update((x, z0 - i) for x in range(x0 - i, x1 + i + 1))
            if "south" in sides:
                out.update((x, z1 + i) for x in range(x0 - i, x1 + i + 1))
            if "west" in sides:
                out.update((x0 - i, z) for z in range(z0 - i, z1 + i + 1))
            if "east" in sides:
                out.update((x1 + i, z) for z in range(z0 - i, z1 + i + 1))
            return sorted(out)

        def in_hole(x, z):
            return hole is not None and hole[0] <= x <= hole[2] and hole[1] <= z <= hole[3]
        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)
                if not in_hole(x, z)]
        if not cols:
            return {"ok": False, "columns": 0, "reason": "the inner rectangle covers "
                                                          "the outer: no annulus"}
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        water = {c: self.wet(c[0], c[1]) for c in cols}
        # The cost, before a block is laid: fill from the bed to the level, the cut
        # above it, and the cover course. Refused by name over the bound.
        est = sum((y - b if b < y else 0) + (self.grade(c[0], c[1]) - y if b > y else 0) + 1
                  for c, b in bed.items())
        if est > self.TERRACE_MAX_BLOCKS:
            return {"ok": False, "columns": len(cols), "estimated_blocks": int(est),
                    "reason": f"a terrace of {len(cols)} columns at y={y} would lay "
                              f"about {est} blocks against {self.TERRACE_MAX_BLOCKS}; "
                              f"lay it in pieces or lower the step"}
        # Read before a block is laid, so what it reads is the ground as found.
        if cover is None:
            dress = self.setting_cover(cols, mat)
        else:
            block = _solid(cover)
            dress = {"at": (lambda x, z: block), "cover": block, "from": "given",
                     "why": f"the caller asked for {cover}"}
        cover_at = dress["at"]
        cover = dress["cover"]
        # the feather is as wide as the fall outside the outer edge asks, read on the
        # sides that are faced and on no other: a strip's unfaced side is the next
        # strip's ground, and a fall read there would feather the faced sides for it
        if feather is None:
            fall = max((abs(y - self.bed(x, z)) for (x, z) in rim(2)), default=0)
            f = int(min(self.PLATEAU_FEATHER_MAX, max(self.TERRACE_FEATHER, fall)))
        else:
            f = int(feather)
        # Trees over the annulus and its feather, whole. **Not the ground cover**: the
        # fill lays every column's top and the cut takes what stands over the level, so
        # there is nothing for `clear_ground_cover` to do here -- and what it would do
        # is lift the turf off the terrace laid beside this one (its box reaches `f`
        # past the rectangle), leaving that terrace a block lower than it was laid.
        self.clear_trees(x0 - f, z0 - f, x1 + f, z1 + f)
        was = max(bed.values()) - min(bed.values())
        filled = flooded = cut = skipped = 0
        for (x, z) in cols:
            if (x, z) in lanes:
                skipped += 1
                continue
            b, wl = bed[(x, z)], water[(x, z)]
            if b < y:
                for yy in range(b + 1, y):
                    self.place_block(x, yy, z, foot_full)
                if wl is not None:
                    flooded += 1
                else:
                    filled += 1
            else:
                filled += 1
            self.place_block(x, y, z, cover_at(x, z))
            for yy in range(y + 1, max(y, b) + 5):
                if self.get_block(x, yy, z).split("[")[0] not in AIR:
                    self.place_block(x, yy, z, "air")
                    cut += 1
            self._sited[(x, z)] = y
        # the retaining face, one column outside the outer edge, to grade
        retained = 0
        for (x, z) in rim(1):
            if (x, z) in lanes:
                skipped += 1
                continue
            b = self.bed(x, z)
            if b >= y:
                continue
            for yy in range(min(b, y - self.PLATEAU_FACE), y + 1):
                self.place_block(x, yy, z, foot_full)
            retained += 1
            self._sited[(x, z)] = y
        # the feather, outside the face: a slope from the level to the ground, a block a
        # column, its top dressed in the cover so it is ground and not a cut
        feathered = 0
        for i in range(1, f + 1):
            t = i / (f + 1.0)
            for (x, z) in rim(i + 1):
                if (x, z) in lanes:
                    continue
                g = self.bed(x, z)
                target = int(round(y * (1 - t) + g * t))
                if g > target:
                    for yy in range(target + 1, g + 4):
                        if self.get_block(x, yy, z).split("[")[0] not in AIR:
                            self.place_block(x, yy, z, "air")
                    self.place_block(x, target, z, cover_at(x, z))
                elif g < target:
                    for yy in range(g, target):
                        self.place_block(x, yy, z, foot_full)
                    self.place_block(x, target, z, cover_at(x, z))
                else:
                    continue
                self._sited[(x, z)] = target
                feathered += 1
        swept = self._sweep_hanging(before)
        dressed = self._dress_worked(before, (x0 - f - 1, z0 - f - 1,
                                              x1 + f + 1, z1 + f + 1), cover=cover)
        return {"ok": True, "y": y, "outer": [x0, z0, x1, z1],
                "inner": list(hole) if hole else None, "label": label,
                "sides": sorted(sides),
                "columns": len(cols), "relief_before": int(was),
                "filled": filled, "flooded": flooded, "cut": cut, "retained": retained,
                "feathered": feathered, "feather": f, "swept": swept, "dressed": dressed,
                "estimated_blocks": int(est), "blocks": len(self._pending) - len(before),
                "lane_columns_left_alone": skipped, "cover": cover,
                "dress": {k: v for k, v in dress.items() if k != "at"},
                "reason": (f"a terrace of {len(cols)} columns at y={y}, {was} of relief "
                           f"taken to 0; {filled} filled, {flooded} filled from a bed "
                           f"under water, {cut} cut, {retained} retained, {feathered} "
                           f"feathered over {f}, {swept} swept, {dressed} dressed")}

    def _site_lay(self, rect, floor_y: int, bed: dict, water: dict, m,
                  ledge_cut: int | None = None) -> dict:
        """Lay the pad: piles under water, fill under ground, and cut what stands over.

                One pass and not three, because the three cases are one thing done to different
                columns: bring every column of the footprint to `floor_y` and clear the space a
                person needs above it. A course wider than the footprint all round, which is
                what makes it a platform rather than a thicker plinth -- that ledge is where a
                person stands to open the door, where `building()` lays its doorstep, and where
                `approach()` finishes.
                
        """
        x0, z0, x1, z1 = rect
        foot_full, deck_full = _solid(m["footing"]), _solid(m["floor"])
        lanes = self._site_lanes()
        filled = piles = decked = cut = skipped = 0
        for x in range(x0 - 1, x1 + 2):
            for z in range(z0 - 1, z1 + 2):
                if (x, z) in lanes:
                    skipped += 1
                    continue
                ledge = (x, z) not in bed
                if ledge:                          # the ledge, one column out
                    b, w = self.bed(x, z), self.wet(x, z)
                else:
                    b, w = bed[(x, z)], water[(x, z)]
                if w is not None:
                    # A pile every other column, so the deck reads as standing on piles
                    # rather than as a mole of masonry filling the water. Writing `air`
                    # between the bed and the deck drains the lake and leaves a sealed
                    # cavity under the building. A deck on piles over a lake has a lake
                    # under it, and water is not standing space.
                    solid = not ((x - x0) % 2 or (z - z0) % 2)
                    if solid:
                        for y in range(b + 1, floor_y):
                            self.place_block(x, y, z, foot_full)
                        piles += 1
                    self.place_block(x, floor_y, z, deck_full)
                    decked += 1
                else:
                    for y in range(min(b, floor_y), floor_y + 1):
                        self.place_block(x, y, z, foot_full)
                    filled += 1
                # Up to the higher of a person's headroom and whatever bank stands over
                # this column: a pad levelled to the doorstep can be *below* the ground
                # on the high side, and a fixed seven courses would leave the hillside
                # standing inside the building. ...except on the ledge of a part that
                # asked for less: a point in a wall stands in the wall, and its ledge
                # columns *are* the wall. Cut to the bank's height there and the wall is
                # split full-height a column either side of the gate tower -- two slits
                # where there was one. `ledge_cut` is the headroom a person needs and no
                # more.
                top = max(floor_y, b) + 5
                if ledge and ledge_cut is not None:
                    top = min(top, floor_y + 1 + int(ledge_cut))
                for y in range(floor_y + 1, top):
                    if self.get_block(x, y, z).split("[")[0] not in AIR:
                        self.place_block(x, y, z, "air")
                        cut += 1
        return {"filled": filled, "decked": decked, "piles": piles, "cut": cut,
                "lane_columns_left_alone": skipped}

    # --- the other three kinds of ground. A place is not a list of buildings. A wall is
    # an **edge**, a gate is a **point**, a market square is an **area**, and until now
    # `site()` knew one shape of ground -- a rectangle with a door on it -- so a type
    # could only ever be a building on a plot. These are the same three cases (`deck`,
    # `platform`, `plinth`) applied to the other three shapes, through the same
    # `_site_lay` where the shape is a rectangle and through `_site_lay_columns` where
    # it is a line.

    def _site_lay_columns(self, floor_of: dict, m, parity=(0, 0)) -> dict:
        """Lay a footing under a set of columns, each to its own level.

                `_site_lay`'s rules over an arbitrary column set rather than a rectangle: fill
                under ground, a pile every other column under water, and cut what stands over.
                A wall does not get a rectangle -- it gets a line, and a line up a hillside is
                not one level -- so the floor is per column and the caller decides how it steps.
                
        """
        foot_full, deck_full = _solid(m["footing"]), _solid(m["floor"])
        lanes = self._site_lanes()
        filled = piles = decked = cut = skipped = 0
        for (x, z), floor_y in sorted(floor_of.items()):
            if (x, z) in lanes:
                skipped += 1
                continue
            b, w = self.bed(x, z), self.wet(x, z)
            if w is not None:
                if not ((x - parity[0]) % 2 or (z - parity[1]) % 2):
                    for y in range(b + 1, floor_y):
                        self.place_block(x, y, z, foot_full)
                    piles += 1
                self.place_block(x, floor_y, z, deck_full)
                decked += 1
            else:
                for y in range(min(b, floor_y), floor_y + 1):
                    self.place_block(x, y, z, foot_full)
                filled += 1
            for y in range(floor_y + 1, max(floor_y, b) + 5):
                if self.get_block(x, y, z).split("[")[0] not in AIR:
                    self.place_block(x, y, z, "air")
                    cut += 1
            self._sited[(x, z)] = int(floor_y)
        return {"filled": filled, "decked": decked, "piles": piles, "cut": cut,
                "lane_columns_left_alone": skipped}

    @staticmethod
    def _edge_run(a, b) -> list:
        """The columns of one segment, inclusive of both ends: along x, along z, or."""
        (ax, az), (bx, bz) = a, b
        if ax == bx:
            step = 1 if bz >= az else -1
            return [(ax, z) for z in range(az, bz + step, step)]
        if az == bz:
            step = 1 if bx >= ax else -1
            return [(x, az) for x in range(ax, bx + step, step)]
        sx, sz = (1 if bx > ax else -1), (1 if bz > az else -1)
        return [(ax + sx * i, az + sz * i) for i in range(abs(bx - ax) + 1)]

    @staticmethod
    def diagonal(a, b) -> bool:
        """Is the segment a -> b a 45-degree run? A run that is neither axial nor
        45 degrees is refused: a wall on an integer lattice is one or the other."""
        return a[0] != b[0] and a[1] != b[1] and abs(b[0] - a[0]) == abs(b[1] - a[1])

    def _decide_edge(self, part: dict) -> dict:
        """An edge's footing, decided: a level per segment along its run.

                A wall is not a building and it is not a rectangle. It runs, it turns, and it
                crosses whatever the ground does; what the library owes a wall type is a footing
                it can stand a course on and **one level per segment**, so that the type builds
                along a segment at a time and the joins at the vertices are a join and not a
                gap. The segments are handed back in order with their own `floor_y`, the
                vertices are named, and the columns of each segment are the swept width.
        """
        path = [(int(p[0]), int(p[1])) for p in (part.get("path") or [])]
        width = max(1, int(part.get("width", 1)))
        if len(path) < 2:
            return {"ok": False, "kind": "edge",
                    "reason": "an edge is a polyline: it needs at least two "
                              f"vertices and this one has {len(path)}"}
        for a, b in zip(path, path[1:]):
            if a[0] != b[0] and a[1] != b[1] and not self.diagonal(a, b):
                return {"ok": False, "kind": "edge",
                        "reason": f"the segment {list(a)} -> {list(b)} runs at an angle "
                                  f"that is neither along x, along z nor 45 degrees; a "
                                  f"corner is a vertex"}
        half = (width - 1) // 2
        segs = []
        for a, b in zip(path, path[1:]):
            line = self._edge_run(a, b)
            if self.diagonal(a, b):
                sx, sz = (1 if b[0] > a[0] else -1), (1 if b[1] > a[1] else -1)
                across = (-sz, sx)                       # the perpendicular diagonal
                axis = "d"
            else:
                across = (0, 1) if a[0] != b[0] else (1, 0)
                axis = "x" if a[0] != b[0] else "z"
            cells = sorted({(x + across[0] * d, z + across[1] * d)
                            for (x, z) in line for d in range(-half, half + 1)})
            segs.append({"a": list(a), "b": list(b), "axis": axis, "cells": cells})
        cols = sorted({c for s in segs for c in s["cells"]})
        xs = [c[0] for c in cols]
        zs = [c[1] for c in cols]
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        relief = max(bed.values()) - min(bed.values())
        # One level for the whole wall where the ground allows it, and a level per
        # segment where it does not -- so a wall over a rise steps at its corners, which
        # is where a wall is entitled to step, rather than in the middle of a run. the
        # footing below is laid to it as to any other.
        one = max(bed.values()) if relief <= self.SITE_RELIEF else None
        if part.get("level") is not None:
            one = int(part["level"])
        floor_of: dict = {}
        for s in segs:
            s["floor_y"] = int(one if one is not None
                               else max(bed[c] for c in s["cells"]))
            for c in s["cells"]:
                floor_of[c] = max(floor_of.get(c, -1 << 30), s["floor_y"])
        floor_y = min(s["floor_y"] for s in segs)
        return {"ok": True, "kind": "edge", "path": [list(p) for p in path],
                "width": width, "level": int(floor_y),
                "segments": [{"a": s["a"], "b": s["b"], "axis": s["axis"],
                              "floor_y": s["floor_y"],
                              "cells": [list(c) for c in s["cells"]]} for s in segs],
                "floor_of": [[list(c), int(y)] for c, y in sorted(floor_of.items())],
                "x0": min(xs), "z0": min(zs), "x1": max(xs), "z1": max(zs),
                "columns": len(cols), "relief": int(relief), "ground": "footing",
                "grade": [int(min(bed.values())), int(max(bed.values()))],
                "one": (int(one) if one is not None else None),
                "terrace": (int(part["level"]) if part.get("level") is not None else None),
                "reason": (f"a footing under {len(cols)} columns in {len(segs)} "
                           f"segment(s) over ground y={min(bed.values())}.."
                           f"{max(bed.values())}; "
                           + (f"one level at y={floor_y}"
                              + (" (the ring's terrace)"
                                 if part.get("level") is not None else "")
                              if one is not None
                              else "a level per segment at y="
                                   + ",".join(str(s["floor_y"]) for s in segs)))}

    def _site_edge(self, part: dict, m, decision: dict | None = None) -> dict:
        """An edge: a polyline with a width, graded segment by segment. See
        `_decide_edge` for the decision; this lays it."""
        dec = decision or self._decide_edge(part)
        if not dec.get("ok"):
            return {**part, "ground": "unsited",
                    "sited": {"ok": False, "cells": 0, "reason": dec.get("reason")}}
        path = [(int(p[0]), int(p[1])) for p in dec["path"]]
        width = int(dec["width"])
        segs = [{"a": s["a"], "b": s["b"], "axis": s["axis"], "floor_y": int(s["floor_y"]),
                 "cells": [tuple(c) for c in s["cells"]]} for s in dec["segments"]]
        cols = sorted({c for s in segs for c in s["cells"]})
        xs = [c[0] for c in cols]
        zs = [c[1] for c in cols]
        # These two calls took the *bounding box* of the path, which for a ring wall is
        # the city inside it -- 243x243 for an inner ring, 760x760 for a great wall --
        # and `clear_ground_cover` takes a `grass_block`, because `VEGETATION` contains
        # the string `grass`. So every ring wall lifted the turf off its own city: of
        # the 49,403 columns of cover the upper ring's terrace laid, 39,267 were air
        # afterwards and the terrace's footing fill was what a person saw. The register
        # `_dress_worked` reads puts the level back only for the columns that call
        # dressed, which is the wall's own line. A part prepares the ground it stands
        # on; a wall stands on a line.
        near = sorted({(x + dx, z + dz) for (x, z) in cols
                       for dx in (-1, 0, 1) for dz in (-1, 0, 1)})
        self.clear_trees(0, 0, 0, 0, columns=near)
        self.clear_ground_cover(0, 0, 0, 0, columns=near)
        # ...and the sweep segment by segment, each over its own run's rectangle, which
        # is a line and not the square the whole path bounds.
        for s in segs:
            sxs = [c[0] for c in s["cells"]]
            szs = [c[1] for c in s["cells"]]
            self._site_sweep(min(sxs) - 1, min(szs) - 1, max(sxs) + 1, max(szs) + 1)
        floor_of = {tuple(c): int(y) for c, y in dec["floor_of"]}
        laid = self._site_lay_columns(floor_of, m, parity=(min(xs), min(zs)))
        floor_y = int(dec["level"])
        out = {**part, "kind": "edge", "path": [list(p) for p in path], "width": width,
               "floor_y": floor_y,
               "segments": [{"a": s["a"], "b": s["b"], "axis": s["axis"],
                             "floor_y": s["floor_y"], "cells": [list(c) for c in s["cells"]]}
                            for s in segs],
               "vertices": [list(p) for p in path[1:-1]],
               "x0": min(xs), "z0": min(zs), "x1": max(xs), "z1": max(zs),
               "ground": "footing",
               "sited": {"ok": True, "relief": int(dec["relief"]), "columns": len(cols),
                         "segments": len(segs), "laid": laid,
                         "grade": list(dec["grade"]),
                         "level": dec.get("terrace"),
                         **({"contract": dec["contract"]} if dec.get("contract") else {}),
                         "reason": dec["reason"]}}
        self.parts.append(out)
        return out

    def _site_point(self, part: dict, m, decision: dict | None = None) -> dict:
        """A point: a pad round an anchor, facing the way the plan says.

                A gatehouse is not sited on a rectangle somebody drew round it -- it is sited on
                the cell the plan put it at, facing the way the plan turned it. Otherwise this
                is a plot: the same three cases, the doorstep the circulation pass levelled if
                there is one, and the way in laid before the type is called.
                
        """
        at = part.get("at") or [part.get("x0"), part.get("z0")]
        ax, az = int(at[0]), int(at[-1])
        size = max(3, int(part.get("size", self.SITE_POINT)))
        h = (size - 1) // 2
        rect = (ax - h, az - h, ax + h, az + h)
        return self._site_rect(part, m, rect, kind="point",
                               facing=part.get("facing"), at=[ax, az], door_at=(ax, az),
                               decision=decision)

    def _site_area(self, part: dict, m, decision: dict | None = None) -> dict:
        """An area: a rectangle brought to one level and joined to the lane.

                A square is the one part that is all ground: what the library owes it is a level
                surface at a level a person can walk onto from the lane, and what the type does
                with it is paving, a well and four stalls.
                
        """
        x0, x1 = int(min(part["x0"], part["x1"])), int(max(part["x0"], part["x1"]))
        z0, z1 = int(min(part["z0"], part["z1"])), int(max(part["z0"], part["z1"]))
        return self._site_rect(part, m, (x0, z0, x1, z1), kind="area",
                               facing=part.get("facing"),
                               door_at=((x0 + x1) // 2, (z0 + z1) // 2),
                               decision=decision)

    def _site_rect(self, part, m, rect, *, kind, facing=None, at=None,
                   door_at=None, decision: dict | None = None) -> dict:
        """The plot case over a rectangle somebody else chose: sound, decide, lay, join.

                `site()`'s own body decides which rectangle inside a plot to prepare, because a
                plot is a piece of land and a building is smaller than it. A point and an area
                are not: a gate is at a cell and a square is a rectangle. So the rectangle is
                given and the rest -- deck over water, platform over relief, plinth otherwise,
                then `approach()` from it to the lane -- is the same.
                
        """
        dec = decision or self._decide_rect(part, rect, kind=kind)
        x0, z0, x1, z1 = (int(v) for v in dec["rect"])
        rect = (x0, z0, x1, z1)
        self.clear_trees(x0 - 2, z0 - 2, x1 + 2, z1 + 2)
        self.clear_ground_cover(x0 - 2, z0 - 2, x1 + 2, z1 + 2)
        self._site_sweep(x0, z0, x1, z1)
        cols = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        bed = {c: self.bed(c[0], c[1]) for c in cols}
        water = {c: self.wet(c[0], c[1]) for c in cols}
        wet_cols = [c for c in cols if water[c] is not None]
        relief = int(dec["relief"])
        fr = self.floor_from_threshold(part.get("label"),
                                       (x0 + x1) // 2, (z0 + z1) // 2) or {}
        ground = dec["ground"]
        floor_y = int(dec["level"])
        # a point standing in a wall keeps the wall either side of it: its ledge is cut
        # to a person's headroom, not to the wall's height
        laid = self._site_lay(rect, floor_y, bed, water, m,
                              ledge_cut=(3 if (kind == "point" and part.get("edge"))
                                         else None))
        lanes = self._site_lanes()
        for x in range(x0 - 1, x1 + 2):
            for z in range(z0 - 1, z1 + 2):
                if (x, z) not in lanes:
                    self._sited[(x, z)] = int(floor_y)
        face = facing or fr.get("facing") or "north"
        dx, dz = door_at or ((x0 + x1) // 2, (z0 + z1) // 2)
        # **An area's door is where the circulation reserved it.** The way into a square
        # or a field is the doorway the pass levelled one step in from the lane, on
        # whichever side the lane came to; the rectangle's centre was written here
        # instead, and every area type that reads `part["door"]` -- the field opens its
        # border at it -- opened on the side nearest its centre while the reserved
        # doorway stayed a post. 168 of 168 areas refused (E008), then 98 with the
        # doorway mid-face.
        if kind == "area" and fr.get("source") == "threshold" and fr.get("door"):
            dx, dz = int(fr["door"][0]), int(tuple(fr["door"])[-1])
        if self.frontage is not None and part.get("label"):
            self.frontage.sited[part["label"]] = {
                "floor_block_y": int(floor_y), "stand_y": int(floor_y) + 1,
                "door": [int(dx), int(floor_y) + 1, int(dz)]}
        ap = self.approach(part.get("label"), int(dx), floor_y + 1, int(dz))
        out = {**part, "kind": kind, "x0": x0, "z0": z0, "x1": x1, "z1": z1,
               "floor_y": int(floor_y), "footprint": [x0, z0, x1, z1],
               "facing": face, "ground": ground,
               "sited": {"ok": bool(ap.get("ok")), "relief": int(relief),
                         "wet_columns": len(wet_cols), "columns": len(cols),
                         "grade": list(dec["grade"]),
                         "laid": laid, "approach": ap,
                         **({"contract": dec["contract"]} if dec.get("contract") else {}),
                         "reason": (f"{ground} at y={floor_y} over ground "
                                    f"y={dec['grade'][0]}..{dec['grade'][1]}, "
                                    f"{dec['wet_columns']} of {dec['columns']} columns wet; "
                                    + str(ap.get("reason")))}}
        if at is not None:
            out["at"] = list(at)
        self.parts.append(out)
        return out

    # --- the shell, by construction ---------------------------------------- Five for
    # six now: stair facing, doors, first-run errors, entry, the internal stair -- a
    # physical rule dies when the library owns it and survives when it is only checked.
    # What is left is not one rule. It is the shell itself. A person called two towns
    # "the same in a different font": the forms repeat because form was decided in three
    # fixed texts and never varied. So the shell moves here as a small family of
    # parametric forms, and the model is left with what it is actually good at --
    # composing the settlement and furnishing the rooms. This is deliberately one call
    # and not six. A builder that has to remember the order (plinth, walls, floors,
    # openings, door, approach, roof, chimney, flights) is a builder that will get the
    # order wrong, and every one of those orderings has cost a round: a floor above its
    # own sill, a roof over an unbroken ceiling, an approach laid before the door.

    #: The height of one storey, in blocks: the floor block plus three clear cells.
    #: Three, not two, because `observe.STAIR_CLEAR` is three -- a storey a person can
    #: walk *up into* has to have room for their head over the top tread.
    STOREY = 4

    #: It stands in the wall line, where the roof plane is at the eave, so four courses
    #: clear the roof and carry the cap. It ran to `ridge + 2`, and under a civic roof
    #: twenty courses high the great hall's stacks were twenty-five blocks of red
    #: masonry. Applied through `TypeBuilder` (every committed type) and by name; a
    #: stored program is the program it was.
    CHIMNEY_ABOVE_EAVE = 4

    #: **Registered.** A roof rises at most this multiple of the walls under it -- half
    #: again their height. The civic silhouette's steep segment repeats to the ridge, so
    #: on a 32-plot two-storey hall (walls of 8) it rose fourteen courses to a flat cap:
    #: a ziggurat taller than the hall. A profile over the cap is not refused, it is
    #: eased: every segment's run is doubled until the ridge fits, so the silhouette
    #: keeps its shape at a shallower pitch. Through `TypeBuilder`.
    ROOF_RISE_MAX = 1.5

    #: Which side of a rectangle you are walking toward when you walk `facing`. A
    #: threshold's `facing` is the way you go through the door, so the door is in the
    #: wall on the far side from the lane.
    _DOOR_WALL = {"north": "z1", "south": "z0", "east": "x0", "west": "x1"}

    def building(self, label: str | None, x0: int, z0: int, x1: int, z1: int,
                 storeys: int, roof, *, wing=None, outshot=None, porch=None,
                 chimney=None, openings: str = "rhythm", stair: str = "auto",
                 mat=None, dormers: int = 0, jetty=None, oriel=None,
                 brackets: bool = False, flashing: bool = False, yard=None,
                 deck: bool = False, platform: int = 0, courtyard=None,
                 chimney_cap: int | None = None, rise_max: float | None = None) -> dict:
        """The whole shell of one building, in one call, or a refusal.

        `(x0, z0)-(x1, z1)` is the footprint, corners inclusive. `storeys` counts floors
        including the ground one; they sit at `floor_y`, `floor_y + 4`, `floor_y + 8`...
        `roof` is a style name from `roof()` -- "gable", "hip", "gambrel", "mansard",
        "shed", "flat" -- or a dict `{"style", "axis", "pitch"}` when you want to say
        which way the ridge runs and how steep it is.

        `mat` is the palette: a dict of any of `wall`, `roof`, `footing`, `frame`,
        `floor`, `trim` (a bare material family name sets all of them). Everything is a
        material *family*, as `roof()` takes -- the stairs, slabs and walls of it are
        looked up for you.

        The order is not yours to choose, because every wrong order has cost a round:

          plinth to grade -> the shell hollowed out of the ground -> floors ->
          walls with a base course and posts -> openings on a rhythm -> the door at the
          reserved threshold, jambed -> approach() -> the roof with a one-block
          overhang -> the chimney *through* the roof plane -> a flight between every
          pair of storeys.

        Extras, each optional and each a form rather than a decoration:

          wing     a second rectangle sharing one full edge with the footprint, given
                   the same floors and eaves and roofed across the other axis so the
                   two roofs meet. An opening is cut in the shared wall at every floor,
                   because a wing you cannot walk into is dead floor.
          outshot  a single-storey lean-to along one side under a shallower `shed`,
                   `{"side": "north|south|east|west", "depth": n}` or just a depth.
          porch    a roofed cell in front of the door, `True` or a depth. Its roof sits
                   at the eaves so it can never take the headroom the way in needs.
          courtyard the shell laid as a **ring round an open yard**, `(w, d)` or one
                   number for a square yard. Four ranges, the door from the lane through
                   the range that faces it and into the yard, a way from the yard into
                   every other range, the yard floored and open to the sky. It refuses
                   when the footprint cannot hold a yard of 3x3 with ranges of 3.
                   A courtyard takes no wing, outshot or porch: the ranges are the
                   massing, and hanging a lean-to off one of them is a different
                   building.
          chimney  `True`, a side name, or an `(x, z)` column: a masonry stack from the
                   real ground, up the outside of a wall, out **through** the roof
                   plane and capped. Never a column standing in the roof.
          chimney_cap  courses the stack rises above the eave, in place of the ridge;
                   `rise_max` the multiple of the wall height the roof may rise. Both
                   are `TypeBuilder`'s to pass (`CHIMNEY_ABOVE_EAVE`, `ROOF_RISE_MAX`);
                   absent, the roof and the stack are what they always were.

        `openings="none"` leaves the walls blank; `stair="none"` leaves the storeys
        unconnected and is for a building whose upper floor is reached some other way.

        Returns `{"ok", "floors", "rooms", "door", "ridge_y", "cells", ...}` -- the
        floor levels, the interior rectangle of each room as `(x0, y, z0, x1, z1)`, the
        door leaf, the ridge height and how many positions the call wrote. Everything
        else is still yours: furnish the rooms, break the symmetry, put things by hand.

        **It refuses, placing nothing**, if the footprint (or a wing, outshot or porch)
        leaves the plot reserved for `label`, or if a wing would swallow the wall the
        door has to go in. A refusal names what it could not do; it is the library
        saying its vocabulary does not reach, and that is worth more than a bad
        building."""
        x0, x1 = int(min(x0, x1)), int(max(x0, x1))
        z0, z1 = int(min(z0, z1)), int(max(z0, z1))
        storeys = max(1, int(storeys))
        m = _mat_roles(mat)
        style, axis, pitch, roof_extras = _roof_spec(roof, x0, z0, x1, z1)

        if x1 - x0 < 2 or z1 - z0 < 2:
            return {"ok": False, "cells": 0, "reason":
                    f"a footprint of {x1 - x0 + 1}x{z1 - z0 + 1} has no inside: a "
                    f"building needs at least 3 by 3 so that one block of wall all "
                    f"round leaves a room"}

        # --- where the floor goes, and which wall the door is in ------------ Read
        # first, because both are answers, not decisions: the circulation pass levelled
        # this doorstep before the program started and it decided which side of the
        # building the town arrives on. An outshot's default side then avoids that wall
        # rather than colliding with it.
        fr = self.floor_from_threshold(label, (x0 + x1) // 2, (z0 + z1) // 2) or {}
        grade_y = int(fr.get("floor_y", self.get_height((x0 + x1) // 2,
                                                        (z0 + z1) // 2)))
        # A podium raises the floor and nothing else: every height below is measured
        # from `floor_y`, so the shell, the roof and the stairs need to know nothing
        # about it, and `grade_y` is kept because the steps back down have to land on
        # the level the town actually arrives on.
        platform = max(0, int(platform))
        floor_y = grade_y + platform
        stand_y = floor_y + 1
        facing = fr.get("facing") or "north"
        door_side = {"z0": "north", "z1": "south",
                     "x0": "west", "x1": "east"}[self._DOOR_WALL[facing]]

        # --- the extras, as rectangles, before anything is placed -----------
        court, why = _courtyard_rects(courtyard, (x0, z0, x1, z1))
        if why:
            return {"ok": False, "cells": 0, "reason": why}
        if court and (wing or outshot or porch):
            return {"ok": False, "cells": 0, "reason":
                    "a courtyard building is its four ranges: a wing, an outshot or a "
                    "porch on one of them is a different building, and asking for both "
                    "is asking for two"}
        wing_r, why = _wing_rect(wing, (x0, z0, x1, z1))
        if why:
            return {"ok": False, "cells": 0, "reason": why}
        out_r, out_side, why = _outshot_rect(outshot, (x0, z0, x1, z1),
                                             avoid=door_side)
        if why:
            return {"ok": False, "cells": 0, "reason": why}

        # --- does all of it lie on the ground this program reserved? --------
        plot = self._plot_rect(label)
        if plot is not None:
            for name, r in (("footprint", (x0, z0, x1, z1)), ("wing", wing_r),
                            ("outshot", out_r)):
                if r and not (plot[0] <= r[0] and r[2] <= plot[2]
                              and plot[1] <= r[1] and r[3] <= plot[3]):
                    return {"ok": False, "cells": 0, "reason":
                            f"the {name} x {r[0]}..{r[2]}, z {r[1]}..{r[3]} leaves the "
                            f"plot you reserved as {label!r}, which is x {plot[0]}.."
                            f"{plot[2]}, z {plot[1]}..{plot[3]} -- nothing has been "
                            f"laid; move it or reserve the ground first"}

        # ...and does the inside hold the stair those storeys need? Asked here, with the
        # rectangle and nothing else, because the answer is arithmetic and because a
        # building whose upper floor nobody can reach is the defect this whole round is
        # about. Refusing is the library saying "not at this size", which a builder can
        # act on; building it anyway is dead floor nobody sees until a person walks in.
        # `stair="none"` is how you say the way up is yours to lay.
        if stair != "none" and storeys > 1:
            # For a courtyard the question is asked of each range, because a ring of
            # four narrow ranges has no room the size of its own footprint in it.
            for r in ([(x0, z0, x1, z1)] if not court
                      else list(court["ranges"].values())):
                why = _stair_room(r, self.STOREY)
                if why:
                    return {"ok": False, "cells": 0, "reason": why}

        # --- ...and where in that wall the door goes ------------------------
        door, why = _door_cell(facing, fr.get("door"), (x0, z0, x1, z1),
                               [r for r in (wing_r, out_r) if r])
        if why:
            return {"ok": False, "cells": 0, "reason": why}

        # ------------------------------------------------------------------ Nothing
        # above this line has placed a block. Everything below it does -- and `mark` is
        # how the one refusal that cannot be decided from the rectangle still places
        # nothing. See `_mark`.
        # ------------------------------------------------------------------
        wrote0 = self.writes
        mark = self._mark()
        eave_y = floor_y + self.STOREY * storeys
        # nothing where not asked
        cap_kw = ({"rise_max": max(1, int(float(rise_max) * (eave_y - floor_y)))}
                  if rise_max is not None else {})
        wall_full = _solid(m["wall"])
        foot_full = _solid(m["footing"])
        frame_full = _solid(m["frame"])
        floor_full = _solid(m["floor"])
        trim_full = _solid(m["trim"])

        pad = 2
        self.clear_trees(x0 - pad, z0 - pad, x1 + pad, z1 + pad)
        self.clear_ground_cover(x0 - pad, z0 - pad, x1 + pad, z1 + pad)

        # (rectangle, storeys, the side of it that is the main mass's wall). The share
        # matters twice over: that wall is built once, by the main mass, and it is not
        # glazed, because glazing an inside wall is a window into the next room.
        if court:
            # North and south run the full width and are built whole; west and east run
            # between them and share their end walls, so a corner is laid once.
            shells = [(court["ranges"]["north"], storeys, None),
                      (court["ranges"]["south"], storeys, None),
                      (court["ranges"]["west"], storeys, None),
                      (court["ranges"]["east"], storeys, None)]
        else:
            shells = [((x0, z0, x1, z1), storeys, None)]
        if wing_r:
            shells.append((wing_r, storeys, _shared_side(wing_r, (x0, z0, x1, z1))))
        if out_r:
            shells.append((out_r, 1, _shared_side(out_r, (x0, z0, x1, z1))))

        rooms, floors = [], [floor_y + self.STOREY * k for k in range(storeys)]
        extras: dict = {}
        # The deck first, and before the plinth: a building over water stands on piles
        # to the bed, and the plinth is what would otherwise be poured into a lake.
        if deck:
            extras["deck"] = _refusing(self._building_deck, (x0, z0, x1, z1),
                                       floor_y, m)
        # ...and the podium before the shell, for the same reason the plinth comes
        # before the walls: it is the ground this building stands on. One block wider
        # than the footprint all round, so the door's own front cell is on it and the
        # steps down have somewhere to start.
        if platform:
            extras["platform"] = self._building_platform(
                (x0, z0, x1, z1), floor_y, grade_y, platform, m)
        for rect, n, _share in shells:
            self.plinth(rect[0], rect[1], rect[2], rect[3], floor_y, foot_full,
                        courses=1, to_grade=True)
            top = floor_y + self.STOREY * n
            # Hollow the shell out of whatever is standing here -- a bank, a tree the
            # clear missed, the plinth's own fill -- and lay the upper floors in the
            # same sweep. Only the inside, and never a cell that is about to carry a
            # floor: the wall band is written once, with the wall, and a program that
            # cleared first and built after pays for every block twice.
            upper = {floor_y + self.STOREY * k for k in range(1, n)}
            for x in range(rect[0] + 1, rect[2]):
                for z in range(rect[1] + 1, rect[3]):
                    for y in range(floor_y + 1, top):
                        self.place_block(x, y, z,
                                         floor_full if y in upper else "air")
            for k in range(n):
                rooms.append((rect[0] + 1, floor_y + self.STOREY * k, rect[1] + 1,
                              rect[2] - 1, rect[3] - 1))

        # --- walls: one block thick, a base course, posts on a rhythm -------
        for rect, n, share in shells:
            a, b, c, d = rect
            top = floor_y + self.STOREY * n - 1
            for side, run in (("north", (a, b, c, b)), ("south", (a, d, c, d)),
                              ("west", (a, b, a, d)), ("east", (c, b, c, d))):
                if side == share:
                    continue        # the main mass's own wall; built once
                self.wall(run[0], floor_y + 1, run[1], run[2], top, run[3],
                          wall_full, post=frame_full, spacing=5, base=foot_full,
                          base_height=1, band=trim_full if top > floor_y + 2 else None)

        # --- the jetty, before the openings that go in the wall it moves.
        # `_building_openings` is told to skip the storey the jetty owns.
        jetty_k = None
        if jetty is not None:
            extras["jetty"] = _refusing(self._building_jetty, jetty,
                                        (x0, z0, x1, z1), floor_y, storeys, m,
                                        openings != "none")
            jetty_k = extras["jetty"].get("storey") if extras["jetty"]["ok"] else None

        # --- openings, on each face of each storey --------------------------
        lights = []
        if openings != "none":
            lights = self._building_openings(shells, floor_y, door, skip=jetty_k)

        # --- the yard, and the ways between it and the ranges --------------- Before
        # the door, deliberately: `doorway()` refuses a leaf whose front cell is not air
        # over standable ground, and for the entrance range the cell behind the door is
        # inside the range and the cell in front is outside it -- but the passage from
        # that range **into** the yard has to exist before the walk model is asked
        # anything, or every check reads a ring of four sealed ranges.
        if court:
            extras["courtyard"] = self._building_courtyard(
                court, (x0, z0, x1, z1), floor_y, floors, storeys, door, facing, m)

        # --- the wing and the outshot are rooms, not sheds ------------------
        for rect, n, share in shells[1:] if not court else ():
            self._building_link((x0, z0, x1, z1), rect, floor_y, n, share)

        if oriel is not None:
            extras["oriel"] = _refusing(self._building_oriel, oriel,
                                        (x0, z0, x1, z1), floor_y, storeys, m, door,
                                        door_side, [r for r in (wing_r, out_r) if r])
        if brackets:
            extras["brackets"] = _refusing(self._building_brackets,
                                           (x0, z0, x1, z1), floor_y, eave_y, m)

        # --- the doorstep, then the door ------------------------------------
        self._building_doorstep(door, facing, floor_y, foot_full)
        # **The leaf comes from the voice.** `doorway()` defaults to `oak_door`, so
        # every building this project has ever made -- in the fell voice, in blackstone,
        # in two Japanese voices -- has hung an oak door and nothing said so. A1's rule
        # is that the palette is the voice's; a door is part of the palette.
        dres = self.doorway(door[0], stand_y, door[1], facing, m["wall"],
                            leaf=joinery(m, "door"), jamb="build", lintel=trim_full)
        if not dres["ok"]:
            # so the shell went up with no way in and every check called it a building.
            # The door is not a decoration; a building without one is not a building, so
            # this is the call's own refusal and everything it laid is wound back.
            self._rollback(mark)
            return {"ok": False, "cells": 0,
                    "door": (door[0], stand_y, door[1]), "front": dres.get("front"),
                    "reason": (f"this building has no way in and nothing has been laid: "
                               f"the door at ({door[0]},{stand_y},{door[1]}) facing "
                               f"{facing} was refused -- {dres['reason']}")}
        if porch:
            self._building_porch(door, facing, floor_y, porch, m)

        # --- the roof, and the chimney through it ---------------------------
        if court:
            # One roof per range, its ridge along the range's own long axis, and the
            # overhang cut to nothing on the yard side: a one-block overhang all round
            # would meet its opposite number over a three-wide yard and roof the
            # courtyard over, which is a building with a dark room in the middle rather
            # than a house round a yard.
            ridge_y = eave_y
            yx0, yz0, yx1, yz1 = court["yard"]
            for side, rect in court["ranges"].items():
                # **Four sheds, all falling into the yard, and the corners belong to the
                # north and south ranges.** A ring of four roofs asks two questions and
                # both were answered by measurement. Roofing a corner *twice* leaves a
                # tread of one slope with the rise of the other behind it: twelve E004
                # on the first courtyard this ever built. Roofing every range so that it
                # falls *outward* puts the joint at a ridge edge with a lower roof
                # beyond it, which is eight more. Falling **inward** puts every joint at
                # an eave, where the neighbour is higher and a tread has nothing to face
                # wrongly at: zero. It is also what a courtyard roof does -- the water
                # goes to the yard.
                r = (rect if side in ("north", "south")
                     else (rect[0], yz0, rect[2], yz1))
                ridge_y = max(ridge_y, self.roof(
                    r[0], r[1], r[2], r[3], eave_y, m["roof"],
                    style="shed", axis={"north": "s", "south": "n",
                                        "west": "e", "east": "w"}[side],
                    pitch=(1, 2), overhang=0))
        else:
            ridge_y = self.roof(x0, z0, x1, z1, eave_y, m["roof"], style=style,
                                axis=axis, pitch=pitch, overhang=1, **roof_extras,
                                **cap_kw)
        if wing_r:
            ridge_y = max(ridge_y, self.roof(
                wing_r[0], wing_r[1], wing_r[2], wing_r[3], eave_y, m["roof"],
                style=style, axis=("x" if axis == "z" else "z"), pitch=pitch,
                overhang=1, **roof_extras, **cap_kw))
        if out_r:
            # Pulled two blocks in on the side it leans against, so that the shed's own
            # one-block overhang stops on the main wall instead of past it. `roof()`
            # overhangs on all four sides and an outshot's roof sits at first-floor
            # level, so an unclipped one drives solid blocks straight through the room
            # above -- which is how the way in stopped being walkable the first time
            # this was built. `axis` for a shed is the compass letter it falls toward.
            rr = _inset_shared(out_r, shells[-1][2], 2)
            self.roof(rr[0], rr[1], rr[2], rr[3], floor_y + self.STOREY, m["roof"],
                      style="shed", axis=out_side[0], pitch=(1, 3), overhang=1)
        flue = None
        if chimney:
            flue = _refusing(self._building_chimney, chimney, (x0, z0, x1, z1), door,
                             floor_y, ridge_y, m,
                             None if chimney_cap is None else eave_y + int(chimney_cap))
            if flue.get("ok") is False:          # a footing with no slab to cap it
                extras["chimney"] = flue
                flue = None
        if dormers:
            extras["dormers"] = _refusing(
                self._building_dormers, int(dormers), (x0, z0, x1, z1), eave_y,
                ridge_y, style, axis, m)
        if flashing:
            extras["flashing"] = _refusing(self._building_flashing, flue, m)

        # --- and only now, the way in --------------------------------------- After the
        # roof, not before it, and this is the one place the order departs from the one
        # written down. `approach()` answers against the world as it stands, and a
        # building without its roof on is an open box: on a bank whose uphill ground is
        # level with the wall head, the walk model correctly finds a way in over the
        # wall top and down inside, `approach()` correctly reports the door already
        # reachable and lays nothing, and then the roof closes the lid and the door is a
        # door nobody can use. Measured, on a six-block slope, on the first build this
        # call ever made. A way in has to be laid against the finished shell or it is
        # not a way in. ...and the way down off the podium first, so `approach()` is
        # asked about a door it can actually reach and lays its path from the foot of
        # the steps rather than reporting a hall two blocks in the air.
        if platform and extras["platform"].get("ok"):
            extras["platform"]["steps"] = self._platform_steps(
                door, facing, floor_y, grade_y, platform, m)
        ap = self.approach(label, door[0], stand_y, door[1])

        # --- a flight between every pair of storeys -------------------------
        stairs = []
        if stair != "none" and storeys > 1:
            got = _refusing(self._building_stairs, label, (x0, z0, x1, z1), floors,
                            door, m["wall"])
            stairs = got if isinstance(got, list) else [got]
            self._hold_flight_ways((x0, z0, x1, z1), floors, door, stairs)

        # --- the yard, last, because it must not stand on the way in --------
        if yard is not None:
            extras["yard"] = _refusing(self._building_yard, yard, (x0, z0, x1, z1),
                                       label, door, facing, m,
                                       [r for r in (wing_r, out_r) if r])

        if court:
            extras["courtyard"]["yard"] = list(court["yard"])
        return {"ok": True, "floors": floors, "rooms": rooms,
                "courtyard": (dict(extras["courtyard"]) if court else None),
                "door": (door[0], stand_y, door[1]), "ridge_y": ridge_y,
                "cells": self.writes - wrote0, "eave_y": eave_y,
                "wing": wing_r, "outshot": out_r, "chimney": flue,
                "lights": lights, "stairs": stairs, "extras": extras,
                "jambs": dres.get("jambs"), "approach": ap,
                "reason": (f"{storeys} storey(s) at y={floors}, eaves {eave_y}, ridge "
                           f"{ridge_y}, door at ({door[0]},{stand_y},{door[1]}) facing "
                           f"{facing}; " + str(ap.get("reason")))}

    # --- the parts of a building, each small enough to read ----------------
    def _plot_rect(self, label: str | None):
        """The rectangle this program reserved for `label`, or None."""
        rects = self._plot_rects(label) if label else []
        return rects[0][1] if len(rects) == 1 else None

    # --- the vocabulary between the shell and the furniture. Three or more of five
    # builders hand-rasterised each of them. None is a physical rule and none is a form
    # -- they are words, and the library did not have them. These are the words. Each is
    # a small method, each is given the shell rather than coordinates, and each returns
    # {"ok", "reason", ...} rather than raising: a refusal here is a decoration that
    # could not be laid, which is not a reason to lose the building.

    @staticmethod
    def _perimeter(rect) -> list:
        """The columns of a rectangle's own edge, clockwise from the north-west."""
        a, b, c, d = rect
        out = [(x, b) for x in range(a, c + 1)]
        out += [(c, z) for z in range(b + 1, d + 1)]
        out += [(x, d) for x in range(c - 1, a - 1, -1)]
        out += [(a, z) for z in range(d - 1, b, -1)]
        return out

    def _building_platform(self, main, floor_y: int, grade_y: int, n: int, m) -> dict:
        """The podium a hall stands on: N courses of footing, carried down to grade.

        One block wider than the footprint all round, which is what makes it a platform
        rather than a thicker plinth: the ledge is where a person stands to open the
        door, where `_building_doorstep` lays the doorstep, and where the steps down
        start. `plinth(to_grade=True)` carries every column to real ground, so on a
        slope the podium is deep on the low side and shallow on the high one, as a
        stone platform is."""
        a, b, c, d = main
        res = self.plinth(a, b, c, d, floor_y, _solid(m["footing"]),
                          courses=n + 1, overhang=1, to_grade=True)
        return {"ok": True, "courses": n, "floor_y": floor_y, "grade_y": grade_y,
                "cells": res["blocks"], "lowest": res["lowest"],
                "reason": (f"a podium {n} block(s) high under x {a - 1}..{c + 1}, "
                           f"z {b - 1}..{d + 1}, floor at y={floor_y} over ground at "
                           f"y={grade_y}")}

    def _platform_steps(self, door, facing, floor_y: int, grade_y: int, n: int,
                        m) -> dict:
        """A flight straight out from the door, down the podium to the ground.

                One tread a course, laid out from the ledge the door opens onto, each carried to
                real ground under it so the flight is masonry and not a staircase in the air.
                `steps()` decides the facings from the finished ground, as it does everywhere --
                a podium on a slope has a different number of real courses on each side and a
                flight that guessed its facings would be backwards on one of them.
                
        """
        dx, dz = _FACE_DIR[facing]
        try:
            _f, _s, _sl = _material(m["footing"])
        except ValueError as e:               # noqa: BLE001 -- refuse, do not crash
            return {"ok": False, "cells": 0, "reason": str(e)}
        full = _solid(m["footing"])
        lanes = ({(x, z) for (x, z) in self.frontage.net.cells}
                 if self.frontage is not None and self.frontage.net else set())
        cells, laid = [], 0
        for k in range(1, n + 1):
            # k blocks out from the door's own front cell, k courses down
            px, pz = door[0] - dx * (k + 1), door[1] - dz * (k + 1)
            if (px, pz) in lanes:
                break                          # the lane is already the ground here
            y = floor_y - k
            self.plinth(px, pz, px, pz, y - 1, full, courses=1, to_grade=True)
            for up in range(1, 4):             # three clear cells over every tread
                self.place_block(px, y + up, pz, "air")
            cells.append((px, y, pz))
            laid += 1
        if not cells:
            return {"ok": False, "cells": 0,
                    "reason": "the lane already runs up to the podium, so there is "
                              "nothing to step down onto"}
        self.steps(cells, m["footing"])
        return {"ok": True, "cells": laid, "treads": [list(c) for c in cells],
                "reason": (f"{laid} tread(s) down the {facing} side of the podium, "
                           f"from y={floor_y} to y={floor_y - laid} against ground at "
                           f"y={grade_y}")}

    def _building_deck(self, main, floor_y: int, m) -> dict:
        """A platform on piles to the bed, wherever this building stands over water."""
        a, b, c, d = main
        skirt = 2
        deck_full = _solid(m["floor"])
        pile = _solid(m["footing"])
        cols, piles = 0, 0
        for x in range(a - skirt, c + skirt + 1):
            for z in range(b - skirt, d + skirt + 1):
                top = self.wet(x, z)
                if top is None:
                    continue
                g = self.grade(x, z)
                cols += 1
                # a pile every other column, so the deck reads as standing on piles
                # rather than as a mole of masonry filling the water -- and the columns
                # between the piles are left as they are. See `_site_lay`.
                if not ((x - a) % 2 or (z - b) % 2):
                    for yy in range(g + 1, floor_y):
                        self.place_block(x, yy, z, pile)
                    piles += 1
                self.place_block(x, floor_y, z, deck_full)
        return {"ok": bool(cols), "cells": cols, "piles": piles,
                "reason": (f"{cols} wet columns decked at y={floor_y} on {piles} piles "
                           f"carried to the bed" if cols else
                           "nothing under or beside this footprint is water -- a deck "
                           "is what a building standing in a lake needs and this one "
                           "is on dry ground")}

    def _building_jetty(self, storey, main, floor_y: int, storeys: int, m,
                        glazed: bool) -> dict:
        """The named storey, carried out one block on every outside face.

                The whole of a jetty is that the storey above is *bigger*, so this moves the
                wall rather than hanging a band of blocks off it: a bracket course under it, the
                floor of the storey carried out to the new line, the wall standing on that, the
                old wall of that storey taken away so the room reaches the new one, and the
                openings punched in the wall that now stands outside.
                
        """
        try:
            k = int(storey)
        except (TypeError, ValueError):
            return {"ok": False, "reason": f"a jetty is the number of a storey, "
                                           f"not {storey!r}"}
        if k < 1 or k >= storeys:
            return {"ok": False, "reason":
                    f"storey {k} cannot jetty: a jetty is an upper storey carried out "
                    f"over the one below it, and this building has storeys 0 to "
                    f"{storeys - 1} with 0 standing on the ground"}
        a, b, c, d = main
        fy = floor_y + self.STOREY * k
        top = floor_y + self.STOREY * (k + 1) - 1
        wall_full = _solid(m["wall"])
        floor_full = _solid(m["floor"])
        frame_fam = m["frame"]
        _f, frame_stairs, _s = _material(frame_fam)
        ring = self._perimeter((a - 1, b - 1, c + 1, d + 1))
        for (x, z) in ring:
            # the bracket, pointing back at the wall it is carried on
            if x == a - 1:
                face = "east"
            elif x == c + 1:
                face = "west"
            elif z == b - 1:
                face = "south"
            else:
                face = "north"
            self.place_block(x, fy - 1, z, f"{frame_stairs}[facing={face},half=top]")
            self.place_block(x, fy, z, floor_full)
            for y in range(fy + 1, top + 1):
                self.place_block(x, y, z, wall_full)
            if k < storeys - 1:
                self.place_block(x, top + 1, z, floor_full)
        # the wall this storey used to have, taken out so the room reaches the new one
        opened = 0
        for (x, z) in self._perimeter(main):
            for y in range(fy + 1, top + 1):
                self.place_block(x, y, z, "air")
                opened += 1
        at = []
        if glazed:
            for run in ((a - 1, b - 1, c + 1, b - 1), (a - 1, d + 1, c + 1, d + 1),
                        (a - 1, b, a - 1, d), (c + 1, b, c + 1, d)):
                horiz = run[0] != run[2]
                lo, hi = (run[0], run[2]) if horiz else (run[1], run[3])
                cells = list(range(lo + 2, hi - 1, 3))
                if cells:
                    self.openings(run[0], fy, run[1], run[2], run[3], at=cells,
                                  width=1, sill=1, head=2)
                    at += cells
        return {"ok": True, "storey": k, "cells": len(ring), "openings": len(at),
                "reason": (f"storey {k} carried out one block on a bracket course at "
                           f"y={fy - 1}, {len(ring)} columns, {opened} cells of the old "
                           f"wall opened into the room, {len(at)} openings in the new "
                           f"one")}

    def _building_oriel(self, oriel, main, floor_y: int, storeys: int, m, door,
                        door_side: str, blocked) -> dict:
        """A projecting bay on brackets, glazed on three sides, roofed with a slab."""
        if not (isinstance(oriel, (tuple, list)) and len(oriel) == 2):
            return {"ok": False, "reason": "an oriel is (face, storey) -- which wall it "
                                           "stands on and which floor it is the window of"}
        face, k = str(oriel[0]).lower(), int(oriel[1])
        if face not in ("north", "south", "east", "west"):
            return {"ok": False, "reason": f"an oriel's face is north, south, east or "
                                           f"west, not {face!r}"}
        if k < 0 or k >= storeys:
            return {"ok": False, "reason": f"this building has storeys 0 to "
                                           f"{storeys - 1}; there is no storey {k}"}
        if k == 0 and face == door_side:
            return {"ok": False, "reason":
                    f"the {face} wall at ground level is the wall the door is in -- an "
                    f"oriel there stands in the way of the way in; put it on another "
                    f"face or a storey up"}
        a, b, c, d = main
        # the wall line, and the middle three columns of it
        if face in ("north", "south"):
            wz = b if face == "north" else d
            mid = (a + c) // 2
            wall = [(mid - 1, wz), (mid, wz), (mid + 1, wz)]
        else:
            wx = a if face == "west" else c
            mid = (b + d) // 2
            wall = [(wx, mid - 1), (wx, mid), (wx, mid + 1)]
        # An oriel projects *out* of the wall it stands in, which for a wall named by
        # the compass side it is on is that compass direction.
        step = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[face]
        out = [(x + step[0], z + step[1]) for (x, z) in wall]
        for r in blocked:
            for (x, z) in out:
                if r[0] <= x <= r[2] and r[1] <= z <= r[3]:
                    return {"ok": False, "reason":
                            f"a wing or outshot stands where the oriel would project, "
                            f"at ({x},{z}) -- nothing has been laid"}
        fy = floor_y + self.STOREY * k
        floor_full = _solid(m["floor"])
        trim_full = _solid(m["trim"])
        _f, frame_stairs, frame_slab = _material(m["frame"])
        back = {"north": "south", "south": "north", "east": "west",
                "west": "east"}[face]
        for (x, z) in out:
            self.place_block(x, fy - 1, z, f"{frame_stairs}[facing={back},half=top]")
            self.place_block(x, fy, z, floor_full)
            for y in (fy + 1, fy + 2):
                self.place_block(x, y, z, "glass")
            self.place_block(x, fy + 3, z, f"{frame_slab}[type=bottom]")
        # the corners of the projection, so the bay is a box and not three panes
        for (x, z) in (out[0], out[-1]):
            for y in (fy + 1, fy + 2):
                self.place_block(x, y, z, trim_full)
        # ...and the wall behind it opened, or the oriel is a window into masonry
        for (x, z) in wall:
            for y in (fy + 1, fy + 2):
                self.place_block(x, y, z, "air")
        return {"ok": True, "face": face, "storey": k, "cells": len(out),
                "reason": (f"an oriel three wide on the {face} wall at y={fy}, carried "
                           f"on a bracket course, glazed on three sides and roofed with "
                           f"a slab")}

    def _building_brackets(self, main, floor_y: int, eave_y: int, m) -> dict:
        """A corbel course under the eaves, on every outside face."""
        if eave_y - 1 <= floor_y + 1:
            return {"ok": False, "reason":
                    f"there is no wall to corbel: the eaves are at y={eave_y} and the "
                    f"floor at y={floor_y}"}
        _f, trim_stairs, _s = _material(m["trim"])
        n = 0
        for (x, z) in self._perimeter((main[0] - 1, main[1] - 1,
                                       main[2] + 1, main[3] + 1)):
            if x == main[0] - 1:
                face = "east"
            elif x == main[2] + 1:
                face = "west"
            elif z == main[1] - 1:
                face = "south"
            else:
                face = "north"
            self.place_block(x, eave_y - 1, z, f"{trim_stairs}[facing={face},half=top]")
            n += 1
        return {"ok": True, "cells": n,
                "reason": f"{n} corbels under the eaves at y={eave_y - 1}"}

    def _building_dormers(self, n: int, main, eave_y: int, ridge_y: int, style: str,
                          axis: str, m) -> dict:
        """Window boxes through the roof slope, spaced along the faces it slopes to."""
        if style == "flat" or ridge_y - eave_y < 2:
            return {"ok": False, "reason":
                    f"a {style} roof rising {ridge_y - eave_y} blocks has no slope to "
                    f"put a dormer in -- nothing has been laid"}
        a, b, c, d = main
        y = eave_y + 1
        placed = []
        if axis == "z" or style in ("hip", "mansard"):
            faces = (("south", b + 1), ("north", d - 1))
            span = (a + 2, c - 2)
        else:
            faces = (("east", a + 1), ("west", c - 1))
            span = (b + 2, d - 2)
        if span[1] - span[0] < 2:
            return {"ok": False, "reason": "the roof is too short to space a dormer "
                                           "along it without hitting a hip"}
        step = max(3, (span[1] - span[0]) // max(1, n))
        for i in range(int(n)):
            face, line = faces[i % 2]
            p = span[0] + 1 + (i // 2) * step
            if p > span[1]:
                break
            if face in ("south", "north"):
                self.dormer(p, y, line, m["roof"], facing=face, width=3)
                placed.append((p, y, line))
            else:
                self.dormer(line, y, p, m["roof"], facing=face, width=3)
                placed.append((line, y, p))
        return {"ok": bool(placed), "cells": len(placed), "at": placed,
                "reason": (f"{len(placed)} of {n} dormers through the slope at y={y}"
                           if placed else "no dormer would fit on this roof")}

    def _building_flashing(self, flue, m) -> dict:
        """The skirt where the stack comes out through the roof plane."""
        if not flue:
            return {"ok": False, "reason": "there is no chimney on this building to "
                                           "flash -- pass chimney= as well"}
        _f, _st, slab = _material(m["footing"])
        cx, cz = flue["x"], flue["z"]
        n = 0
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if not (dx or dz):
                    continue
                # The roof plane is the highest **block** laid in the column, not the
                # highest write: a roof that clears its own attic or sets a gable back
                # writes air above its slope, and a slab laid on top of an air write is
                # a slab held up by nothing. Voice contract: the cottage in the ochre
                # voice flashed its stack at y=77 over a roof whose surface was 74.
                col = [y for (x, y, z), blk in self._pending.items()
                       if (x, z) == (cx + dx, cz + dz) and y <= flue["top"]
                       and not str(blk).startswith("air")]
                if not col:
                    continue
                self.place_block(cx + dx, max(col) + 1, cz + dz, f"{slab}[type=bottom]")
                n += 1
        return {"ok": bool(n), "cells": n,
                "reason": (f"{n} courses of flashing round the stack at ({cx},{cz})"
                           if n else "the stack meets no roof plane to flash against")}

    def _building_yard(self, yard, main, label, door, facing: str, m, blocked) -> dict:
        """A wall course round a piece of ground, with a gap where the lane comes in."""
        if not (isinstance(yard, (tuple, list)) and len(yard) == 4):
            return {"ok": False, "reason": "a yard is a rectangle (x0, z0, x1, z1) of "
                                           "ground to put a wall round"}
        a, b = int(min(yard[0], yard[2])), int(min(yard[1], yard[3]))
        c, d = int(max(yard[0], yard[2])), int(max(yard[1], yard[3]))
        plot = self._plot_rect(label)
        if plot is not None and not (plot[0] <= a and c <= plot[2]
                                     and plot[1] <= b and d <= plot[3]):
            return {"ok": False, "reason":
                    f"the yard x {a}..{c}, z {b}..{d} leaves the plot you reserved as "
                    f"{label!r} -- nothing has been laid"}
        if c - a < 2 or d - b < 2:
            return {"ok": False, "reason": f"a yard of {c - a + 1}x{d - b + 1} is not a "
                                           f"piece of ground"}
        fam = m["footing"]
        wall_block = _wall_block(fam)
        dx, dz = _FACE_DIR[facing]
        step = (door[0] - dx, door[1] - dz)      # the doorstep, which stays clear
        lanes = ({(x, z) for (x, z) in self.frontage.net.cells}
                 if self.frontage is not None and self.frontage.net else set())
        paths = {(cx, cz) for p in self.paths for (cx, cz) in p["cells"]}
        keep = lanes | paths | {step, (door[0], door[1])}
        # ...and the ring round the door, because a wall one block off a doorstep is the
        # same defect as a wall on it: you cannot walk out of the building.
        keep |= {(step[0] + i, step[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
        # the gap: the perimeter column nearest the lane, and its neighbours -- **among
        # the columns the wall would actually stand in.** Where the building's own wall
        # is the stretch of perimeter nearest the doorstep (a hall pinned to the plot's
        # edge with its door in that edge, the yard behind it), a gap chosen on the
        # building's cells is no gap, and the yard behind the hall is a court nobody can
        # walk into: the temple's cloister, on real ground.
        ring = self._perimeter((a, b, c, d))

        def taken(p):
            return any(r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]
                       for r in [main] + list(blocked))
        n_ring = len(ring)

        def dist(p):
            return abs(p[0] - step[0]) + abs(p[1] - step[1])
        # three consecutive free columns, the middle one nearest the doorstep; failing
        # that (a yard the building fills to within two columns) the nearest free one
        triples = [i for i in range(n_ring)
                   if not any(taken(ring[(i + k) % n_ring]) for k in (-1, 0, 1))]
        if triples:
            i = min(triples, key=lambda j: dist(ring[j]))
        else:
            free = [j for j in range(n_ring) if not taken(ring[j])] or list(range(n_ring))
            i = min(free, key=lambda j: dist(ring[j]))
        gap = {ring[(i + k) % n_ring] for k in (-1, 0, 1)}
        near = ring[i]
        laid = 0
        for (x, z) in ring:
            if (x, z) in gap or (x, z) in keep:
                continue
            if any(r[0] <= x <= r[2] and r[1] <= z <= r[3]
                   for r in [main] + list(blocked)):
                continue                       # the building itself stands here
            g = self.grade(x, z)
            here = self.get_block(x, g + 1, z)
            if here and here.split("[")[0].split(":")[-1] not in AIR:
                continue                       # something already stands here
            self.place_block(x, g + 1, z, wall_block)
            laid += 1
        return {"ok": bool(laid), "cells": laid, "gap": sorted(gap),
                "reason": (f"{laid} columns of yard wall round x {a}..{c}, z {b}..{d}, "
                           f"with a {len(gap)}-cell gap at {near} where the lane comes "
                           f"in" if laid else
                           "every column of this yard's perimeter is the building, a "
                           "lane, or the way to the door -- nothing has been laid")}

    def _building_openings(self, shells, floor_y, door, skip=None) -> list:
        """Openings on a rhythm along every outside face, at every floor.

                An outside face is one no other part of this building stands against: a wing's
                shared wall and the stretch of the main wall an outshot leans on are inside
                walls, and a window there is a window into the next room. The door's own bay is
                left to `doorway()`.

                `skip` is a storey whose wall is not where this call thinks it is -- a jetty
                has already moved it out a block and glazed the wall that now stands there.
                
        """
        out = []
        others = [r for r, _n, _s in shells]
        for rect, n, share in shells:
            a, b, c, d = rect
            for k in range(n):
                if skip is not None and k == skip:
                    continue
                y = floor_y + self.STOREY * k
                for side, run in (("north", (a + 1, b, c - 1, b)),
                                  ("south", (a + 1, d, c - 1, d)),
                                  ("west", (a, b + 1, a, d - 1)),
                                  ("east", (c, b + 1, c, d - 1))):
                    if side == share or run[0] > run[2] or run[1] > run[3]:
                        continue
                    horiz = run[0] != run[2]
                    lo, hi = (run[0], run[2]) if horiz else (run[1], run[3])
                    at = []
                    for p in range(lo + 1, hi, 3):
                        cx, cz = (p, run[1]) if horiz else (run[0], p)
                        if abs(cx - door[0]) <= 1 and abs(cz - door[1]) <= 1:
                            continue
                        if any(r is not rect and r[0] <= cx <= r[2]
                               and r[1] <= cz <= r[3] for r in others):
                            continue
                        out.append((cx, y, cz))
                        at.append(p)
                    if at:
                        self.openings(run[0], y, run[1], run[2], run[3], at=at,
                                      width=1, sill=1, head=2)
        return out

    #: Which wall of a range faces the yard. The ranges are named for where they stand,
    #: so the north range's yard wall is its south one.
    _COURT_INNER = {"north": "south", "south": "north",
                    "west": "east", "east": "west"}

    def _building_courtyard(self, court, main, floor_y, floors, storeys, door,
                            facing, m) -> dict:
        """The yard, and the ways between it and the four ranges.

                Three things, and each is the reason a courtyard could not be asked for before:

                **The yard is ground, not a room.** Its floor is laid at `floor_y` in the
                footing, and everything above it, up past the eaves, is cleared -- so it is
                open to the sky and the walk model reads it as outdoors. A yard with a lid on
                it is a hall, and a hall of that span is a hall with no roof anybody can build.

                **The way in goes through a range, not round it.** The lane arrives at one outer
                wall and the door is in that wall, so the cell behind it is inside the entrance
                range and the passage from there to the yard is a passage through a building
                rather than a way round one. It is an **L** and not a straight line, because the
                circulation pass puts the door anywhere on the perimeter including a corner,
                where no straight line reaches the yard at all: the first version of this cut
                into a corner and left a whole range sealed.

                **Every range opens onto the yard.** Two cells wide and two high, in the middle
                of each inner wall, at every floor -- the same opening `_building_link` cuts for
                a wing, and for the same reason: a range you have to go outside and round to get
                into is dead floor, and the walk model says so.
                
        """
        yx0, yz0, yx1, yz1 = court["yard"]
        foot_full = _solid(m["footing"])
        floor_full = _solid(m["floor"])
        # The floor of the yard, and the sky over it. Cleared past the eaves so the four
        # roofs falling into it have nothing of the yard left in their way.
        top = floor_y + self.STOREY * storeys + 3
        paved = 0
        for x in range(yx0, yx1 + 1):
            for z in range(yz0, yz1 + 1):
                self.place_block(x, floor_y, z, foot_full)
                paved += 1
                for y in range(floor_y + 1, top + 1):
                    self.place_block(x, y, z, "air")
        # **Every range opens onto the yard**, two cells wide in the middle of its own
        # inner wall, at every floor -- the same opening `_building_link` cuts for a
        # wing, and for the same reason.
        ways = {}
        for side, rect in court["ranges"].items():
            inner = self._COURT_INNER[side]
            a, b, c, d = rect
            if inner in ("north", "south"):
                wz = b if inner == "north" else d
                cells = [(x, wz) for x in _mid_two(max(a, yx0), min(c, yx1))]
            else:
                wx = a if inner == "west" else c
                cells = [(wx, z) for z in _mid_two(max(b, yz0), min(d, yz1))]
            laid = []
            for k in range(storeys):
                y = floor_y + self.STOREY * k
                for (cx, cz) in cells:
                    for dy in (1, 2):
                        self.place_block(cx, y + dy, cz, "air")
                    laid.append([cx, y + 1, cz])
            ways[side] = laid
        # **And the way in from the lane is a passage, not a hole.** The door is where
        # the circulation pass put it, which is anywhere along the outer perimeter --
        # including a corner, where no straight line reaches the yard at all. So the
        # passage is an L: one step in off the wall, along it until it lines up with the
        # yard, then straight in. Two cells high, floored, and cut through whatever
        # range wall stands in it.
        entrance, corridor = self._courtyard_passage(court, door, main)
        for (cx, cz) in corridor:
            self.place_block(cx, floor_y, cz, floor_full)
            for dy in (1, 2):
                self.place_block(cx, floor_y + dy, cz, "air")
        return {"ok": True, "paved": paved, "entrance": entrance, "ways": ways,
                "corridor": [list(c) for c in corridor], "cleared_to": top,
                "reason": (f"a yard of {yx1 - yx0 + 1}x{yz1 - yz0 + 1} paved at "
                           f"y={floor_y} and open to the sky, entered through the "
                           f"{entrance} range by a passage of {len(corridor)} cells, "
                           f"with a way from the yard into all {len(ways)} ranges")}

    @staticmethod
    def _courtyard_passage(court, door, main):
        """(which range the town arrives through, the cells of the way in).

                The door sits on the outer perimeter and the yard is in the middle, so the
                passage steps in off the wall, runs along inside it until it is opposite the
                yard, and then goes straight in. Where the door is already opposite the yard the
                first leg is one cell and the passage is straight, which is the ordinary case;
                where it is in a corner the L is the only way there is.
                
        """
        x0, z0, x1, z1 = main
        yx0, yz0, yx1, yz1 = court["yard"]
        dx, dz = int(door[0]), int(door[1])
        if dx <= x0:
            side, ax, into = "west", dx + 1, yx0 - 1
        elif dx >= x1:
            side, ax, into = "east", dx - 1, yx1 + 1
        elif dz <= z0:
            side, ax, into = "north", dz + 1, yz0 - 1
        else:
            side, ax, into = "south", dz - 1, yz1 + 1
        if side in ("west", "east"):
            az = min(max(dz, yz0), yz1)
            along = [(ax, z) for z in range(min(dz, az), max(dz, az) + 1)]
            inward = [(x, az) for x in range(min(ax, into), max(ax, into) + 1)]
        else:
            axx = min(max(dx, yx0), yx1)
            along = [(x, ax) for x in range(min(dx, axx), max(dx, axx) + 1)]
            inward = [(axx, z) for z in range(min(ax, into), max(ax, into) + 1)]
        cells = [(dx, dz)] + along + inward
        seen, out = set(), []
        for c in cells:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return side, out

    def _building_link(self, main, rect, floor_y, n, share) -> None:
        """Cut a way through the wall a wing or an outshot shares with the main mass.

                Two cells high and up to two wide at every floor they have in common. A wing you
                have to go outside and round to get into is not a wing, and under the corrected
                walk model its floor is simply floor nobody reaches.
                
        """
        a, b, c, d = rect
        if share in ("west", "east"):
            wx = a if share == "west" else c
            cells = [(wx, z) for z in _mid_two(max(b, main[1]) + 1,
                                               min(d, main[3]) - 1)]
        elif share in ("north", "south"):
            wz = b if share == "north" else d
            cells = [(x, wz) for x in _mid_two(max(a, main[0]) + 1,
                                               min(c, main[2]) - 1)]
        else:
            return
        for k in range(n):
            y = floor_y + self.STOREY * k
            for (cx, cz) in cells:
                for dy in (1, 2):
                    self.place_block(cx, y + dy, cz, "air")

    def _building_doorstep(self, door, facing, floor_y, block) -> int:
        """One course of the footing at the door's own level, one block out, three wide.

                Where the circulation pass has already levelled the doorstep this is nothing --
                and it never touches a lane cell, because re-paving the lane in this building's
                footing is not this call's business. Where there is no lane it is the difference
                between a walkable door and an unwalkable one, and the reason is exact: without
                it the way in starts on a *slab* half a block below the sill, `steps()` cannot
                justify the tread that steps up onto a slab (`_is_surface` says a bottom slab is
                not ground, which is the rule the linter judges treads by), the tread is demoted
                and the flight ends one block short of the door it was laid for. Measured on a
                six-block bank; a porch happens to fix it, and a door should not need a porch.
                
        """
        dx, dz = _FACE_DIR[facing]
        perp = ((1, 0), (-1, 0)) if dx == 0 else ((0, 1), (0, -1))
        lanes = ({(x, z) for (x, z) in self.frontage.net.cells}
                 if self.frontage is not None and self.frontage.net else set())
        n = 0
        for k in (0,) + perp:
            px = door[0] - dx + (k[0] if k else 0)
            pz = door[1] - dz + (k[1] if k else 0)
            if (px, pz) in lanes:
                continue
            self.plinth(px, pz, px, pz, floor_y, block, courses=1, to_grade=True)
            n += 1
        return n

    def _building_porch(self, door, facing, floor_y, porch, m) -> None:
        """A roofed cell in front of the door: two posts and a roof at the eaves.

                At the eaves and never lower, because the way in is measured walking: the three
                cells above the doorstep are `observe.STAIR_CLEAR` and a porch roof that took
                one of them would make the door this call has just hung unreachable. The roofed
                rectangle stops one block out from the wall and the roof's own overhang carries
                it back over the door, so nothing is driven into the wall itself.
                
        """
        depth = max(2, 2 if porch is True else int(porch))
        dx, dz = _FACE_DIR[facing]
        frame = _solid(m["frame"])
        ox, oz = door[0] - dx * depth, door[1] - dz * depth
        perp = ((1, 0), (-1, 0)) if dx == 0 else ((0, 1), (0, -1))
        a, c = sorted((door[0] - dx, ox))
        b, d = sorted((door[1] - dz, oz))
        # The deck first, carried to real ground. A porch whose posts start at the
        # doorstep is a porch standing on nothing wherever the ground falls away from
        # the door -- which is most doors on this kind of site -- and it is also where
        # `approach()` has to set off from. Give it floor at the door's own level and
        # the way in starts on the porch instead of three blocks under it.
        self.plinth(a + perp[1][0], b + perp[1][1], c + perp[0][0], d + perp[0][1],
                    floor_y, _solid(m["footing"]), courses=1, to_grade=True)
        for sx, sz in perp:
            for y in range(floor_y + 1, floor_y + self.STOREY):
                self.place_block(ox + sx, y, oz + sz, frame)
        self.roof(a, b, c, d, floor_y + self.STOREY, m["roof"], style="shed",
                  axis={"north": "s", "south": "n", "east": "w", "west": "e"}[facing],
                  pitch=(1, 3), overhang=1)

    def _building_chimney(self, chimney, main, door, floor_y, ridge_y, m,
                          top_cap: int | None = None) -> dict:
        """A masonry stack in the wall line, up **through** the roof plane, and capped.
        `top_cap` is the highest the stack may go (the eave plus `CHIMNEY_ABOVE_EAVE`,
        through `TypeBuilder`); None is the ridge and two, as it always was."""
        a, b, c, d = main
        if isinstance(chimney, (tuple, list)) and len(chimney) == 2:
            cx, cz = int(chimney[0]), int(chimney[1])
        else:
            side = chimney if isinstance(chimney, str) else None
            if side is None:        # the wall the door is not in, opposite it by choice
                on_z = door[1] in (b, d)
                side = (("north" if door[1] == d else "south") if on_z else
                        ("west" if door[0] == c else "east"))
            cx, cz = {"north": ((a + c) // 2, b), "south": ((a + c) // 2, d),
                      "west": (a, (b + d) // 2), "east": (c, (b + d) // 2)}[side]
            if (cx, cz) == tuple(door):             # never in the doorway
                cx, cz = (cx + 2, cz) if cz in (b, d) else (cx, cz + 2)
        stack = _solid(m["footing"])
        cap = _material(m["footing"])[2]
        base = min(floor_y, self.get_height(cx, cz))
        top = ridge_y + 2 if top_cap is None else min(ridge_y + 2, int(top_cap))
        for y in range(base, top + 1):
            self.place_block(cx, y, cz, stack)
        self.place_block(cx, top + 1, cz, f"{cap}[type=bottom]")
        return {"x": cx, "z": cz, "top": top + 1, "from_y": base}

    def _building_stairs(self, label, main, floors, door, mat) -> list:
        """A flight between every pair of storeys, alternating between opposite walls.

                Two rules decide the placement and both are physical.

                **Against a wall.** A tread's neighbours in a flight are diagonal, so a run in
                open air is `y1 - y0` separate pieces of stair held up by nothing, and E010
                says so. Laid tight against a wall every tread abuts the mass.

                **Never over the flight below it.** A return flight one lane in from the first
                puts its bottom tread directly above the top tread of the one beneath -- and
                three clear cells over every tread is the whole of `observe.STAIR_CLEAR`, so
                that is a stair you walk up into a stair. Alternating between opposite walls
                gives each flight its own columns and its own headroom; you cross the floor
                between them, which is what a person does in a real building anyway.

                The first flight goes against the wall furthest from the door, so the stair is
                not standing in the way in.
                
        """
        a, b, c, d = main
        need = self.STOREY + 2                     # foot, four treads, landing
        along_x = (c - a) >= (d - b)
        inside = (c - a - 1, d - b - 1)
        across = inside[1] if along_x else inside[0]
        if (inside[0] if along_x else inside[1]) < need or across < 3:
            return [{"ok": False, "reason":
                     f"no room for a flight: the inside is {inside[0]}x{inside[1]} and "
                     f"a storey of {self.STOREY} needs {need} columns along the flight "
                     f"and three across, so a second flight can go against the "
                     f"opposite wall instead of over the first"}]
        out = []
        for k in range(len(floors) - 1):
            if along_x:
                far = (d - 1) if door[1] - b < d - door[1] else (b + 1)
                lane = far if k % 2 == 0 else (b + 1 if far == d - 1 else d - 1)
                res = self.flight(label, a + 2, lane, floors[k], floors[k + 1],
                                  "east", mat=mat)
            else:
                far = (c - 1) if door[0] - a < c - door[0] else (a + 1)
                lane = far if k % 2 == 0 else (a + 1 if far == c - 1 else c - 1)
                res = self.flight(label, lane, b + 2, floors[k], floors[k + 1],
                                  "south", mat=mat)
            out.append({"y0": floors[k], "y1": floors[k + 1], **res})
        return out

    def _hold_flight_ways(self, main, floors, door, stairs) -> None:
        """Hold a walkable line from each storey's own way in to the foot of its flight.

                `flight()` holds its treads, its landing and the one cell you set off from.
                Nothing held the floor between that cell and the door, and a type furnishing
                its own storey does not know where its flight is: a partition, a hearth and a
                fence laid across the corner the foot stands in seal every storey above
                without touching one cell the flight owns. That is E003 and E011 as a pair on
                the same room, and it is the largest single class of error a city reports.

                The line is drawn **here**, inside `building()`, because here the storey is an
                empty box and an L between any two of its cells always exists. Two L's join
                two cells; the one that keeps clear of the flight's own columns is taken. The
                cells are held two high -- a person and their head -- and `TypeBuilder` puts
                back whatever is written into them, exactly as it does a doorstep.

                **Writes only, and `fitting()` is deliberately not refused on it.** A flight's
                own cells are refused to furniture (`flight_cells`) and that stands. Refusing
                the *way* as well was measured and made things worse: `cottage` reads a
                refusal and builds something else, and eighteen refused fittings took it from
                twelve sealed instances in a sweep of 360 to twenty-two. A rule that changes
                what a committed type decides is not the rule that takes a wall back out of a
                corridor, and only the second one is this.
                
        """
        a, b, c, d = main
        if c - a < 2 or d - b < 2:
            return
        inside = (a + 1, b + 1, c - 1, d - 1)
        start = _inside_of(door, main)
        for k, st in enumerate(stairs):
            if not isinstance(st, dict) or not st.get("ok") or not st.get("foot"):
                break
            fx, fy, fz = (int(v) for v in st["foot"])
            if not (inside[0] <= fx <= inside[2] and inside[1] <= fz <= inside[3]):
                start = None
            if start is None:
                break
            line = _l_path(start, (fx, fz), self.flight_cells, fy + 1)
            # **Only where the floor is already under it.** A cell with nothing to stand
            # on is not part of anybody's way, and `fitting()` fills its own floor as it
            # places -- a hearth brings its own stone. Holding such a cell refuses the
            # fitting *and* the block it was going to lay, which is a hole in the floor
            # rather than a way kept clear, and holes are what the class this rule
            # exists for is made of.
            view = self._pending_view()
            for cx, cz in line:
                here = view.get((cx, fy, cz))
                if here is None and self._world_volume().inside(cx, fy, cz):
                    here = self._world_volume().state(cx, fy, cz)
                if not here or here.split("[")[0].split(":")[-1] in AIR:
                    continue
                self.flight_way.add((cx, fy + 1, cz))
                self.flight_way.add((cx, fy + 2, cz))
            self.flight_way -= self.flight_cells
            land = st.get("landing")
            start = (int(land[0]), int(land[2])) if land else None

    # --- the step inside a room, by construction. 169 of its 202 cells were in rooms a
    # person could walk into and 85 of them were **exactly one whole block up** -- a
    # dais, a sleeping platform, a stall floor, a loft edge with no tread onto it.
    # `steps()` can lay that tread and nothing ever asked it to. This is the counterpart
    # primitive: a raised floor that brings its own way up.

    def dais(self, x0: int, z0: int, x1: int, z1: int, y: int, mat: str) -> dict:
        """A raised floor one block up, with its own slab step on its longest open side.

                The floor blocks go at `y`; the room's own floor is the course at `y - 1`, so a
                dais is exactly the one-block rise this project measured 85 of. The step is a
                bottom slab laid in the room's floor cells along whichever side has the most
                open floor beside it -- half a block up onto the slab, half a block up onto the
                dais, and neither of them is a jump.

                Returns {"ok", "side", "cells", "step_cells", "reason"}.
                
        """
        x0, x1 = int(min(x0, x1)), int(max(x0, x1))
        z0, z1 = int(min(z0, z1)), int(max(z0, z1))
        y = int(y)
        full, _stairs, slab = _material(mat)
        view = self._pending_view()

        def open_floor(cx, cz) -> bool:
            """Is this a cell of room floor you could stand on, beside the dais?"""
            here = view.get((cx, y, cz), self.get_block(cx, y, cz))
            below = view.get((cx, y - 1, cz), self.get_block(cx, y - 1, cz))
            return (here.split("[")[0].split(":")[-1] in AIR
                    and _occupies(below) and _top_face(below) == 2)

        sides = {
            "north": [(cx, z0 - 1) for cx in range(x0, x1 + 1)],
            "south": [(cx, z1 + 1) for cx in range(x0, x1 + 1)],
            "west": [(x0 - 1, cz) for cz in range(z0, z1 + 1)],
            "east": [(x1 + 1, cz) for cz in range(z0, z1 + 1)],
        }
        best, open_cells = None, []
        for name, cells in sides.items():
            got = [c for c in cells if open_floor(*c)]
            if len(got) > len(open_cells):
                best, open_cells = name, got

        for cx in range(x0, x1 + 1):
            for cz in range(z0, z1 + 1):
                self.place_block(cx, y, cz, full)
                # Written down as a dais, which is the opposite of writing it down as
                # furniture: a dais is floor (`observe.floor_stances`), and a dais with
                # nothing to step onto it from is floor nobody can reach.
                self.dais_cells.add((cx, y, cz))
        if best is None:
            return {"ok": False, "side": None, "cells": (x1 - x0 + 1) * (z1 - z0 + 1),
                    "step_cells": 0,
                    "reason": "the dais is laid, and no side of it has open room floor "
                              "beside it to step from -- nothing could be stepped onto "
                              "it, so put it against floor a person can stand on"}
        for (cx, cz) in open_cells:
            self.place_block(cx, y, cz, f"{slab}[type=bottom]")
        return {"ok": True, "side": best,
                "cells": (x1 - x0 + 1) * (z1 - z0 + 1), "step_cells": len(open_cells),
                "reason": f"{len(open_cells)} slab step(s) laid along its {best} side, "
                          f"so the dais is a walk up from the floor beside it"}

    # --- what the finishing pass must keep off -----------------------------
    def _record_path(self, label: str | None, kind: str, columns: list) -> None:
        """Remember the columns a way-in laid, beside the plot it was laid for."""
        if not columns:
            return
        self.paths.append({"label": label, "kind": kind,
                           "cells": [[int(cx), int(cz)]
                                     for (cx, cz) in sorted(set(columns))]})

    @staticmethod
    def _air_pocket(vol, sealed, stances):
        """The air a set of stances is standing in, and whether any of it is unsealed."""
        from collections import deque
        seeds = [(x, s // 2, z) for (x, z, s) in map(tuple, stances)]
        seen, escapes = set(), False
        q = deque(seeds)
        while q:
            p = q.popleft()
            if p in seen or not vol.inside(*p):
                continue
            if vol.name(*p) not in AIR:
                continue
            seen.add(p)
            if not sealed[p[0] - vol.x0, p[1] - vol.y0, p[2] - vol.z0]:
                escapes = True
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                q.append((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
        return seen, escapes

    # Blocks whose appearance depends on their neighbours: a fence's north/south/east/
    # west, a wall's up/height, a stair's shape, a pane's connections. See flush().
    _CONNECTIVE = ("_fence", "_fence_gate", "_wall", "_pane", "_stairs", "iron_bars",
                   "_bars", "chest", "redstone_wire", "_door")

    def _is_connective(self, block: str) -> bool:
        name = block.split("[")[0].split(":")[-1]
        return (any(name.endswith(s) for s in self._CONNECTIVE)
                and not name.endswith("_wall_sign") and not name.endswith("_wall_torch"))

    # --- harness side ------------------------------------------------------
    def flush(self, chunk: int = 50_000, conn_passes: int = 1) -> dict:
        # Queued treads first: their whole point is to be decided against the finished
        # world, and this is the moment the world stops changing. A program that forgot
        # to call resolve_steps() would otherwise ship a staircase with no stairs in it.
        self.resolve_steps()
        items = [((x, y, z), Block(b if ":" in b else "minecraft:" + b))
                 for (x, y, z), b in self._pending.items()]
        ok = failed = 0
        errors: dict[str, int] = {}
        import time
        t0 = time.perf_counter()
        for i in range(0, len(items), chunk):
            res = placeBlocks(items[i:i + chunk], doBlockUpdates=False, host=world.HOST)
            for good, r in res:
                if good:
                    ok += 1
                else:
                    failed += 1
                    errors[str(r)] = errors.get(str(r), 0) + 1

        # Second pass: connective blocks, re-placed with block updates ON. The bulk pass
        # runs with updates off so it cannot set off gravity or water across a whole
        # settlement -- but that same flag is what computes connection states, so every
        # fence we have ever built stood as an isolated post and every stair was
        # shape=straight. A human found it by walking the town; no render and no metric
        # we had could show it. scripts/test_block_updates.py measures both settings
        # side by side: updates off gives 0/3 fences joined, on gives 3/3 and mitres the
        # stair corner. Confining the second pass to connective blocks keeps the physics
        # risk to their immediate neighbours. One sweep is enough, and that is measured
        # rather than assumed. wave 1 had a stubborn residue of missed wall joins with
        # exactly that shape. scripts/test_wall_joins.py builds the case both ways and
        # reads it back: one sweep and two give byte-identical states, all joins made.
        # The residue was a defect in the *check*, which was counting walls beside leaf
        # litter. The parameter stays so the question can be re-asked cheaply.
        conn = [it for it in items if self._is_connective(str(it[1].id))]

        # ...and the connective blocks we did not place but stood next to. A fence's
        # state is computed from its neighbours, so *changing a neighbour* leaves it
        # stale. Any pass that writes next to somebody else's fence owes it an update.
        CONN_PROPS = ("north", "south", "east", "west", "up", "shape")
        written = set(self._pending)
        seen: set = set()
        for (x, y, z) in written:
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                               (0, 0, 1), (0, 0, -1)):
                p = (x + dx, y + dy, z + dz)
                if p in written or p in seen:
                    continue
                seen.add(p)
                try:
                    blk = self.world_site.editor.worldSlice.getBlockGlobal(p)
                except Exception:
                    continue
                name = str(blk.id).split(":")[-1]
                if not name or not self._is_connective(name):
                    continue
                # drop the connection properties and let the game recompute them
                keep = {k: v for k, v in (blk.states or {}).items()
                        if k not in CONN_PROPS}
                state = name + ("[" + ",".join(f"{k}={v}" for k, v in sorted(keep.items()))
                                + "]" if keep else "")
                conn.append((p, Block("minecraft:" + state)))
        for _ in range(conn_passes):
            for i in range(0, len(conn), chunk):
                placeBlocks(conn[i:i + chunk], doBlockUpdates=True, host=world.HOST)

        secs = time.perf_counter() - t0
        self.results = {"placed": ok, "failed": failed, "errors": errors,
                        "seconds": round(secs, 2),
                        "blocks_per_sec": round(len(items) / secs) if secs else 0,
                        "unique_positions": len(items),
                        "connective_rejoined": len(conn), "calls": self.calls,
                        **self.audit()}
        return self.results

    # --- objective checks for the tells GDMC judges name ------------------
    LIGHT = {"torch", "wall_torch", "lantern", "soul_lantern", "campfire", "glowstone",
             "sea_lantern", "shroomlight", "jack_o_lantern", "candle", "end_rod",
             "redstone_lamp", "froglight", "beacon", "fire", "soul_torch", "lava"}

    def audit(self) -> dict:
        palette: dict[str, int] = {}
        for b in self._pending.values():
            key = b.split("[")[0].split(":")[-1]
            palette[key] = palette.get(key, 0) + 1
        lights = sum(n for k, n in palette.items()
                     if any(k.endswith(w) or k == w for w in self.LIGHT))

        # terrain contact: per column, the lowest block placed vs the original ground
        col_bottom: dict[tuple[int, int], int] = {}
        for (x, y, z) in self._pending:
            k = (x, z)
            if k not in col_bottom or y < col_bottom[k]:
                col_bottom[k] = y
        gaps = 0        # columns whose lowest block floats clear of the ground
        buried = 0      # columns cut into the ground
        ground_h, floor_h = [], []
        for (x, z), y in col_bottom.items():
            g = self.world_site.height(x, z)
            ground_h.append(g)
            floor_h.append(y)
            if y > g + 1:
                gaps += 1
            elif y < g:
                buried += 1
        import statistics
        return {
            "palette_size": len(palette),
            "palette": dict(sorted(palette.items(), key=lambda kv: -kv[1])),
            "material_families": sorted(_families(palette)),
            "light_sources": lights,
            "footprint_columns": len(col_bottom),
            "ground_relief_under_build": (max(ground_h) - min(ground_h)) if ground_h else 0,
            "floor_level_stdev": round(statistics.pstdev(floor_h), 2) if len(floor_h) > 1 else 0,
            "columns_floating": gaps,
            "columns_cut_into_ground": buried,
        }

    def type_builder(self, part: dict, role: str | None = None) -> "TypeBuilder":
        """This builder as a *type* is allowed to see it. See `TypeBuilder`.

                The refusal list is put on the builder as well, because the builder is what the
                parts stage gets back and `_type_refusals` is the row it reports. A refusal
                nobody can read is a refusal nobody acts on.

                `role` is the type's own `ROLE`; a `civic` type is handed the voice's civic
                silhouette where the voice names one (demo-polish, 2a).
                
        """
        tb = TypeBuilder(self, part, role=role)
        self._type_refusals = tb.refused
        return tb


class TypeBuilder:
    """The library a type is written against: all of it, minus the ground.

    So the ground is not the type's to touch, and this says so twice over:

      by name    `foundation_to_grade`, `terrace`, `plinth`, `grade`, `clear_trees`,
                 `clear_ground_cover` and `approach` refuse, naming the call and what
                 does it instead. `get_height` stays readable -- a type has to be able
                 to *look* at the ground; it may not work it.
      by cell    a `place_block`, `place_cuboid` or `fill_region` below `floor_y`, or
                 with a stair in it, raises with the cell named. Stairs come from
                 `steps()`, `flight()`, `roof()` and the library's own siting, which
                 decide their facing from the finished ground; a tread placed directly
                 is E004 waiting for a slope, and `minka/byre/24` carried four of them.

    Everything else is the same object: `building()`, `roof()`, `fitting()`,
    `flight()`, `check_walkable()` and the rest are the Builder's own methods, so what
    they place is placed on the real builder and is not second-guessed here. The rule
    is about what the *type program* writes, which is exactly what preflight's E013
    checks statically and this checks as it runs."""

    #: The calls a type may not make, and what does the job instead.
    FORBIDDEN = {
        "foundation_to_grade": "site() carries the footprint to real ground for you",
        "terrace": "site() cuts and fills a footprint to one level for you",
        "plinth": "site() lays the base under the footprint for you",
        "grade": "site() has already sounded the bed; build from part['floor_y'] up",
        "clear_trees": "site() clears the footprint and its margin for you",
        "clear_ground_cover": "site() clears the footprint and its margin for you",
        "approach": "site() lays the way in from the lane before you are called, and "
                    "building() lays the way to its own door",
    }

    #: Placing one of these directly is how E004 comes back. `steps()` and everything
    #: built on it decide a tread's facing from the finished ground; a literal stair
    #: state in a type program is a facing decided by a model reading a heightmap.
    _TREAD = "_stairs"

    def __init__(self, b: Builder, part: dict, role: str | None = None):
        self._b = b
        self._part = dict(part)
        #: The sited dict itself, so what this builder records about the part (the
        #: floor's stand-ins) reaches `Builder.parts[-1]` and the driver's record.
        self._orig_part = part
        self._floor_y = part.get("floor_y")
        #: The settlement's palette, by role.
        self.voice = dict(part.get("voice") or _mat_roles(None))
        # **`ground` is always on the voice a type reads.** `site()` writes the
        # setting's own surface there, and a builder standing a type outside the
        # pipeline gets the plain's. A type that asks for it never has to know which it
        # got.
        self.voice.setdefault(_GROUND_ROLE, "grass_block")
        #: Where the voice names a second wall material, a share of the buildings are
        #: faced in it -- chosen here, by the part's own seed and name, so the type
        #: never knows and its shape is unchanged. A street of thirty houses all in one
        #: sandstone reads as one extruded building; the same street with a quarter of
        #: it in the voice's second stone reads as a street. No type names either
        #: material.
        self.wall_alt = False
        alt = self.voice.get(_WALL_ALT)
        if alt:
            key = f"{part.get('name') or part.get('label') or ''}:{part.get('seed') or 0}"
            self.wall_alt = (zlib.crc32(key.encode()) % 1000) < int(WALL_ALT_SHARE * 1000)
            if self.wall_alt:
                self.voice["wall"] = str(alt)
        self.voice.pop(_WALL_ALT, None)
        #: The voice's silhouette, as `roof()`'s own parameters, or None. **A civic type
        #: gets the voice's civic silhouette** where the voice names one
        #: (`part["roof"]["civic"]`, demo-polish 2a) and the voice's one roof where it
        #: does not; every other role gets the one roof. Chosen here, by the type's
        #: `ROLE`, so that no type names any of it and E015 still refuses one that does.
        spec = dict(part.get("roof") or {})
        civic = spec.pop("civic", None)
        #: True or None leaves the type's choice as it is. Resolved from the voice and
        #: the place's form by the driver before the part is composed; a stored program
        #: that composes a part itself carries None and builds what it built.
        self.chimney = spec.pop("chimney", None)
        if civic:
            civic = {k: v for k, v in dict(civic).items() if k != "chimney"}
        if role == "civic" and civic:
            spec = dict(civic)
        self.roof_spec = spec or None
        self.role = role
        #: Every refusal this type ran into, in order, for the readout.
        self.refused: list = []
        #: `site()` writes the cell a person stands in to open this part's door into
        #: `part["door"]`, having levelled it, laid the way in from the lane to it, and
        #: checked that a person can walk to it; `TypeBuilder.doorway()` already refuses
        #: a leaf hung anywhere else. What nothing stopped was the type then *building
        #: on* it, and four of the city's thirteen sealed rooms are exactly that: `hall`
        #: floods the ground inside its plot from the lane under a rule stricter than
        #: the walk model -- it will not take a one-block step up without a slab under
        #: it -- and carries its footing course over everything the flood did not reach,
        #: which on a deck is the whole deck including its own doorstep. A granite wall
        #: in the one cell you have to stand in, and the building behind it is sealed.
        #: So the two cells above the doorstep are held open here, the way a tread and a
        #: cell below `floor_y` are held. The check runs after the write and puts the
        #: cell back rather than raising, because the write is usually a cuboid that is
        #: right about everything else it covers.
        d = part.get("door")
        held = []
        if d is not None and self._floor_y is not None and len(tuple(d)) >= 2:
            dx, dz = int(tuple(d)[0]), int(tuple(d)[-1])
            held += [(dx, int(self._floor_y) + 1, dz), (dx, int(self._floor_y) + 2, dz)]
        # ...and the rest of the way in, which `site()` measured and wrote down. See
        # `Builder._way_in`: the doorstep alone was not enough on a deck in a lake,
        # where the way in is four columns of jetty and a type walled the far end.
        held += [(int(c[0]), int(c[1]), int(c[2])) for c in (part.get("way") or [])]
        self._doorstep = tuple(dict.fromkeys(held))

    #: The silhouette keys a voice owns. Voice contract, B2: `roof()` and `building()`
    #: are handed these from `part["roof"]` **whatever the type passes**, so a type can
    #: neither ignore the voice's roof nor quietly override it. fourteen types never did
    #: and six read it and assumed their own where it was silent, and a whole city was
    #: composed in one silhouette and checked in another. This is the same law applied a
    #: tenth time: the thing is done by the library, by construction, and a type that
    #: names one of these is refused at preflight (E015).
    SILHOUETTE = ("profile", "ends", "eave", "tiers")

    def _silhouette(self, k: dict) -> dict:
        """`k` with the voice's silhouette written over whatever it says."""
        if not self.roof_spec:
            return k
        out = dict(k)
        for key in self.SILHOUETTE:
            if self.roof_spec.get(key) is not None:
                out[key] = self.roof_spec[key]
        if out.get("profile"):
            # `roof()` ignores `pitch` when it has a profile; dropping it here says so
            # rather than leaving two answers on the call.
            out.pop("pitch", None)
        return out

    def roof(self, *a, **k):
        """`roof()`, handed the voice's silhouette. See `SILHOUETTE`."""
        return self._b.roof(*a, **self._silhouette(k))

    def building(self, *a, **k):
        """`building()`, its roof spec handed the voice's silhouette. See `SILHOUETTE`.

        The roof is `building()`'s seventh positional or its `roof=` keyword, a style
        name or a dict; either way it leaves here as a dict carrying the voice's
        profile, ends, eave and tiers where the voice has them."""
        if self.roof_spec:
            a = list(a)
            spec = k["roof"] if "roof" in k else (a[6] if len(a) > 6 else None)
            d = {"style": spec} if isinstance(spec, str) else dict(spec or {})
            d = self._silhouette(d)
            if "roof" in k or len(a) <= 6:
                k["roof"] = d
            else:
                a[6] = d
        # where it says yes or nothing the type's choice stands, its stack capped at the
        # eave plus `CHIMNEY_ABOVE_EAVE`; and the roof rises at most `ROOF_RISE_MAX` of
        # the walls under it. Every committed type builds through here.
        if self.chimney is False:
            k.pop("chimney", None)
        k.setdefault("chimney_cap", Builder.CHIMNEY_ABOVE_EAVE)
        k.setdefault("rise_max", Builder.ROOF_RISE_MAX)
        res = self._b.building(*a, **k)
        # **The door the shell hung is held open too.** The doorstep held above is the
        # cell `site()` reserved; the shell hangs its leaf on its own wall, which on a
        # pad the building does not fill is a few columns off it, and a type that then
        # dresses its walls -- screens, panels, a footing course -- dresses over its own
        # door. A minka in the city did exactly that and its hall had no way in. The
        # leaf's two cells are put back like the doorstep's.
        d = (res or {}).get("door") if isinstance(res, dict) else None
        if isinstance(d, (list, tuple)) and len(d) == 3 and d[0] is not None:
            held = [(int(d[0]), int(d[1]), int(d[2])), (int(d[0]), int(d[1]) + 1, int(d[2]))]
            self._doorstep = tuple(dict.fromkeys(list(self._doorstep) + held))
        return res

    #: `voices.SOLID` lets `roles.floor` be any block "because it is only ever laid as a
    #: cube", and the committed types shape the floor role anyway -- a floor slab, a
    #: step, a bed or a bench in it -- because every voice on disk had a family there
    #: and nothing had ever passed a bare block in. The first voice a spec call authored
    #: with one (`packed_mud`) crashed thirty of a ring's parts at once. The library
    #: keeps both promises here, by construction: the cube of the floor is the floor,
    #: and where a type asks for a shape the game does not have of it, the voice's
    #: **footing** -- a family by the validator -- stands in. Which calls stood in is
    #: written on the part (`voice_stood_in`) so the record says so. The Builder calls
    #: that take a material to shape, and where it sits: a keyword `mat=` everywhere,
    #: and these positions where it is positional.
    _SHAPED = {"block": 0, "steps": 1, "dais": 6, "window": 5, "roof": 5,
               "roof_cone": 4, "dormer": 3, "flight": None, "fitting": None}

    def _shapeable(self, mat, kind: str = "stairs", call: str = ""):
        """`mat` where it has the shape asked of it, else the footing family."""
        from .prims import family, shape as _shape
        if mat is None or not isinstance(mat, str) or family(mat) is not None:
            return mat
        if kind == "full":
            return mat
        try:
            _shape(mat, kind)
            return mat
        except ValueError:
            pass
        fallback = self.voice.get("footing") or "cobblestone"
        if family(fallback) is None:
            fallback = "cobblestone"
        rec = self._part.setdefault("voice_stood_in", {})
        key = f"{call or 'shape'}:{mat}"
        rec[key] = rec.get(key, 0) + 1
        orig = getattr(self, "_orig_part", None)
        if orig is not None:
            orig["voice_stood_in"] = rec
        return fallback

    def block(self, mat, kind: str = "full") -> str:
        """`Builder.block()`, the floor's shapes standing in as `_shapeable` says."""
        return self._b.block(self._shapeable(mat, kind, "block"), kind)

    def _shaped(self, name: str):
        """A Builder method whose material is passed through `_shapeable` first."""
        fn = getattr(self._b, name)
        pos = self._SHAPED[name]

        def call(*a, **k):
            a = list(a)
            if "mat" in k:
                k["mat"] = self._shapeable(k["mat"], "stairs", name)
            elif pos is not None and len(a) > pos:
                a[pos] = self._shapeable(a[pos], "stairs", name)
            if name != "fitting" or not self._b.flight_way:
                return fn(*a, **k)
            # **Furniture on the way to a flight is taken back out, not refused.** A
            # refusal is something a type reads and acts on, and one committed type acts
            # on it by building something else: eighteen refused fittings took `cottage`
            # from twelve sealed instances in a sweep of 360 to twenty-two. The cells
            # are the ones that have to stay clear, and clearing them after the fact
            # leaves the type's own decisions where they were.
            was = {c: self._b._pending.get(c) for c in self._b.flight_way}
            out = fn(*a, **k)
            for c, before in was.items():
                here = self._b._pending.get(c)
                if here is None or here == before:
                    continue
                if before is None:
                    del self._b._pending[c]
                else:
                    self._b.place_block(c[0], c[1], c[2], before)
                self._b.fitting_cells.discard(c)
                self.refused.append({
                    "call": "fitting",
                    "reason": (f"a type does not furnish the way to its own flight: "
                               f"fitting() put {here.split('[')[0]!r} at "
                               f"({c[0]},{c[1]},{c[2]}), which building() holds open "
                               f"between this storey's way in and the foot of its "
                               f"flight. It has been taken back out.")})
            return out
        call.__name__ = name
        return call

    def __getattr__(self, name: str):
        if name in self.FORBIDDEN:
            return self._refuse(name)
        if name in ("place_block", "place_cuboid", "fill_region"):
            return self._guarded(name)
        if name == "doorway":
            return self._doorway
        if name in self._SHAPED:
            return self._shaped(name)
        return getattr(self._b, name)

    def door_cell(self, x0: int, z0: int, x1: int, z1: int):
        """**The cell of this rectangle's rim where the way in is**, or None.

                So it is the library's: the reserved doorway, or the threshold's own door or
                lane cell, **snapped to the nearest cell of the rectangle** -- because the
                doorway the circulation levelled is one step in from the lane and a sited pad
                may be inset past it. An area with no threshold anywhere near gets None and
                draws its border closed, which is right for a field in the middle of a belt.
                
        """
        want = None
        d = self._part.get("door")
        if d is not None:
            want = (int(d[0]), int(d[-1]))
        else:
            th = self._b.threshold(self._part.get("label")
                                   or self._part.get("name"))
            for key in ("door", "lane"):
                got = (th or {}).get(key)
                if got is not None:
                    want = (int(got[0]), int(got[-1]))
                    break
        if want is None:
            return None
        x = min(max(want[0], int(x0)), int(x1))
        z = min(max(want[1], int(z0)), int(z1))
        reach = {"west": x - x0, "east": x1 - x, "north": z - z0, "south": z1 - z}
        side = min(reach, key=lambda k: reach[k])
        if side == "west":
            return (int(x0), int(z))
        if side == "east":
            return (int(x1), int(z))
        if side == "north":
            return (int(x), int(z0))
        return (int(x), int(z1))

    def check_attached(self, *a, **k):
        """`check_attached()`, asked only about this part's own half of the world."""
        if self._floor_y is not None and "above" not in k:
            k["above"] = int(self._floor_y)
        return self._b.check_attached(*a, **k)

    def _doorway(self, *a, **k):
        """The door goes where `site()` reserved it, and nowhere else.

        `building()` is unaffected: it calls the Builder's own `doorway()` and puts the
        leaf at the reserved cell itself. This is the refusal for a type that goes round
        it -- and, like every other refusal here, it hands back a value rather than
        raising, because that is how the rest of the library says no."""
        x = k.get("x", a[0] if a else None)
        z = k.get("z", a[2] if len(a) > 2 else None)
        want = self._part.get("door")
        if want is None or x is None or z is None:
            return self._b.doorway(*a, **k)
        wx, wz = int(want[0]), int(want[-1])
        if (int(x), int(z)) != (wx, wz):
            r = {"ok": False, "jambs": 0, "open_sides": [], "front": None,
                 "reason": (f"a type does not decide where the town arrives: "
                            f"doorway() was asked for ({int(x)},{int(z)}) and site() "
                            f"reserved ({wx},{wz}). The circulation pass levelled that "
                            f"doorstep and laid the lane to it -- put the door at "
                            f"part['door'], or let building() put it there for you.")}
            self.refused.append({"call": "doorway", "reason": r["reason"]})
            return r
        return self._b.doorway(*a, **k)

    def _refuse(self, name: str):
        why = self.FORBIDDEN[name]
        def refused(*a, **k):                                    # noqa: ANN202
            r = {"ok": False, "cells": 0,
                 "reason": (f"a type does not touch the ground: {name}() is the "
                            f"library's and not yours -- {why}. Build from "
                            f"part['floor_y'] up.")}
            self.refused.append({"call": name, "reason": r["reason"]})
            return r
        return refused

    @staticmethod
    def _write_args(name: str, a: tuple, k: dict) -> tuple:
        """(x, ys, z, block) out of a write call, positional or by keyword.

                Read out by hand rather than through `inspect.signature`, because this runs on
                every block a type places -- tens of thousands an instance -- and a refusal
                that cost a second a building would be a refusal nobody could afford.
                
        """
        if name == "place_block":
            x, z = k.get("x", a[0] if a else None), k.get("z", a[2] if len(a) > 2 else None)
            y = k.get("y", a[1] if len(a) > 1 else None)
            return x, [y], z, k.get("block", a[3] if len(a) > 3 else "")
        x, z = k.get("x0", a[0] if a else None), k.get("z0", a[2] if len(a) > 2 else None)
        y0 = k.get("y0", a[1] if len(a) > 1 else None)
        y1 = k.get("y1", a[4] if len(a) > 4 else None)
        return x, [y for y in (y0, y1) if y is not None], z, \
            k.get("block", a[6] if len(a) > 6 else "")

    def _guarded(self, name: str):
        fn = getattr(self._b, name)
        floor_y = self._floor_y

        def guarded(*a, **k):                                    # noqa: ANN202
            x, ys, z, block = self._write_args(name, a, k)
            if self._TREAD in str(block):
                raise BuildError(
                    f"a type does not place stairs: {name}() was asked for "
                    f"{block!r} at ({x},{ys[0] if ys else '?'},{z}) -- a tread's "
                    f"facing is decided from the finished ground by steps(), "
                    f"flight(), roof() and site(), never written down. Use one of "
                    f"those.")
            if floor_y is not None and ys and int(min(ys)) < int(floor_y):
                raise BuildError(
                    f"a type does not build below its own floor: {name}() at "
                    f"({x},{int(min(ys))},{z}) reaches under part['floor_y'], which "
                    f"is {int(floor_y)}. The ground under this part has been prepared "
                    f"for you; build from y={int(floor_y)} up.")
            held = self._held()
            if not held:
                return fn(*a, **k)
            was = [self._b._pending.get(c) for c in held]
            out = fn(*a, **k)
            self._clear_doorstep(name, was, held)
            return out
        return guarded

    def _held(self) -> tuple:
        """Every cell a type's write is put back out of: its own doorstep and the way
        `site()` laid to it, and the floor `building()` holds between each storey's way
        in and the foot of its flight. See `Builder.flight_way`."""
        way = self._b.flight_way
        if not way:
            return self._doorstep
        return tuple(dict.fromkeys(list(self._doorstep) + sorted(way)))

    def _clear_doorstep(self, name: str, was: list, held: tuple) -> None:
        """Put the doorstep back if the write that just ran stood in it. `_doorstep`.

                Compared against what stood there **before this call**, so it only ever undoes
                the call it is looking at: a rule that read the cell alone would take the *door
                leaf* out the next time the type placed a block anywhere, because a leaf
                `doorway()` hung at `part["door"]` is something standing in that cell too. The
                cell is read off `_pending` -- the only thing that can have filled it since
                `site()` levelled it is this program -- and the refusal is recorded by name, so
                an author reads "you built over your own way in" where they read every other
                refusal rather than reading E002 in a city.
        """
        doorstep = set(self._doorstep)
        for cell, before in zip(held, was):
            here = self._b._pending.get(cell)
            if here is None or here == before:
                continue
            if here.split("[")[0].split(":")[-1] in AIR:
                continue
            if before is None:
                del self._b._pending[cell]
            else:
                self._b.place_block(cell[0], cell[1], cell[2], before)
            what = ("a type does not build on its own doorstep"
                    if cell in doorstep else
                    "a type does not build across the way to its own flight")
            why = ("which is a cell part['door'] and part['way'] reserve for the "
                   "person opening the door -- site() levelled it and laid the way in "
                   "to it" if cell in doorstep else
                   "which is a cell building() holds open between this storey's way "
                   "in and the foot of its flight -- build over it and the storeys "
                   "above are floor nobody can walk to")
            self.refused.append({
                "call": name,
                "reason": (f"{what}: {name}() put "
                           f"{here.split('[')[0]!r} at ({cell[0]},{cell[1]},{cell[2]}), "
                           f"{why}. It has been put back to "
                           + (f"{before.split('[')[0]!r}" if before else
                              "the ground site() left") + ".")})


API_DOC = '''
You are writing a Python program that builds a structure in a Minecraft world.

You do not place blocks by emitting them. You write code, and the code places the blocks.
Write the whole program in one fenced ```python block; it is executed as-is.

## Reading the world

    get_height(x, z) -> int      y of the topmost solid block. Build on get_height(x,z)+1.
    get_block(x, y, z) -> str    block id at a position, e.g. "grass_block", "air".

## Placing blocks

    place_block(x, y, z, block)  one block. Ids have no namespace: "stone_bricks".
                                 Block states are allowed: "oak_stairs[facing=north,half=bottom]",
                                 "oak_log[axis=x]", "lantern[hanging=true]". "air" removes.

Ids are validated against the server's registry before your program runs, and one bad id
rejects the whole file. This world is **Minecraft 1.21.11**, where `chain` was renamed
**`iron_chain`** — every arm of the last experiment was bounced on that one id.
    place_cuboid(x0,y0,z0, x1,y1,z1, block)          solid box, corners inclusive
    fill_region(x0,y0,z0, x1,y1,z1, block, replace=None)   box, optionally only where
                                 the existing block is `replace`

## Shapes — use these instead of rasterising by hand

    disc(cx, y, cz, r, block)                    filled circle
    ring(cx, y, cz, r, block, thickness=1)       circle outline, even thickness, no gaps
    cylinder(cx, y0, cz, r, height, block, hollow=True, thickness=1)
    sphere(cx, cy, cz, r, block, hollow=False)
    dome(cx, y0, cz, r, block, hollow=True)      upper hemisphere sitting on y0
    line(x0,y0,z0, x1,y1,z1, block)              3D line, one block per step
    path(points, width, block, follow_ground=True)
        points is [(x,z), ...]. Lays an even-width path along the polyline and clears
        headroom above it. Do not stamp your own discs along a curve — it blobs.

## Stairs — never place a tread by hand

    steps(cells, mat, axis=None, prefer=None, half="bottom", demote="slab")
        cells is [(x,y,z), ...] along a flight. Queues the treads; each one's facing
        is decided at the end from the finished ground on both sides, so the raised
        quarter always points up-slope. A tread that cannot be justified becomes a
        slab, and a rising tread the flight does not continue past is refused. You do
        not need to call anything to finish: they are resolved when the pass flushes.
    step(x, y, z, mat, axis="z", prefer=None)      one tread, same rules
        `mat` is a material family ("cobblestone") or a stair id.
        `prefer` is which way a tread on level ground should face.
        This is for treads. A stair used as a corbel, a chair or a sill bracket is
        not a tread -- place that one directly with place_block.
    resolve_steps()
        Place every queued tread now, facings decided against the world as it stands.
        You do not need to call this: it happens for you when the pass finishes. Call
        it only if you want to read a flight back with get_block before you go on.

## Roofs

    roof(x0, z0, x1, z1, y, mat, style="gable", axis="z", pitch=(1,1),
         overhang=1, profile=None, ends=None, eave="straight", tiers=1) -> ridge_y

        Roofs the rectangle with eaves at height y. Returns the ridge height.
        style : "gable" | "hip" | "gambrel" | "mansard" | "shed" | "flat"
        axis  : "z" ridge runs east-west (slopes face north/south)
                "x" ridge runs north-south (slopes face east/west)
                for "shed": the direction it slopes down toward, "n"|"s"|"e"|"w"
        pitch : (rise, run). (1,1) is 45 degrees. (1,2) and (1,3) are shallow.
                (2,1) and (3,1) are steep. This is a real choice — vary it.
        mat   : a material family name, e.g. "spruce", "deepslate_tile", "brick",
                "oxidized_copper", "sandstone", "blackstone", "cherry", "prismarine",
                "mud_brick", "nether_brick", "quartz", "tuff_brick", "end_stone_brick",
                "hay_block". The roof uses the matching stairs and slabs automatically.

        The six styles are presets over four further parameters. Reach for these when
        the roof you want is not one of the six:

        profile : [(rise, run), ...] from the eave up to the ridge, overriding `pitch`.
                Each segment holds its pitch for its own `run` and the last one
                repeats. "Shallow, then steep at the ridge" is [(1,2), (2,1)].
        ends  : (near, far), each "gable", "hip", "half-hip" or "irimoya" — what each
                end of the ridge does. A half-hip is gabled below and hipped at the
                top. An **irimoya** is the other way round: hipped below with a gable
                above it, set back one course.
        eave  : "straight", "flared" (the eave course held one step longer, so the roof
                lands shallower where it overhangs) or "upturned" (the overhang lifted
                a block above the course it leaves — the turned-up eave).
        tiers : stack the roof this many times, each tier pulled in from the one below
                with a one-block wall course between them. A two-tiered hall is
                tiers=2, not two roof() calls.

    roof_cone(cx, y0, cz, r, mat, pitch=(1,1)) -> top_y     conical roof for a round tower
    dormer(x, y, z, mat, facing="south", width=3)           window box breaking a roof plane

## Terrain

    clear_trees(x0, z0, x1, z1, margin=2) -> int
        Removes whole trees, trunk and canopy, including trees rooted outside the box
        whose branches overhang it. Always call this before building on wooded ground.
    clear_ground_cover(x0, z0, x1, z1)     strips grass, flowers, ferns, snow
    foundation_to_grade(x0, z0, x1, z1, y, block, skirt=0)
        Fills every column from the real ground up to y-1, following the terrain, so
        nothing floats over a slope.
    terrace(x0, z0, x1, z1, y, block, feather=4)
        Levels an area and blends the edge outward, instead of cutting a hard rectangle.
    flattest_rect(x0, z0, x1, z1, w, d) -> (x, z, floor_y, spread)
        Finds the flattest w by d footprint in a region. Use it to site the build.

## Detail

    wall(x0,y0,z0, x1,y1,z1, block, post=None, spacing=4, base=None,
         base_height=1, band=None)
        A wall with posts at intervals, a base course and a top band.
    window(x, y, z, facing, mat, width=2, height=2, sill=True, shutters=False)
    doorway(x, y, z, facing, mat, leaf="oak_door", jamb="build", lintel=True,
            threshold=None) -> {"ok", "jambs", "open_sides", "reason"}
        A door **and the wall it needs around it**: both halves of the leaf, a jamb on
        each side, a lintel over it. (x, y, z) is the lower leaf — `stand_y` from
        floor_from_threshold(), not the floor block. A door hung at the end of a wall
        run has open air beside it and you can see round the side of it; `jamb="build"`
        fills the missing side, `jamb="refuse"` places nothing and tells you which side
        is open so you can move the door instead. Do not place door leaves by hand.

## Interiors — what a room of each purpose contains

    fitting(kind, x, y, z, facing="north", *, mat="cobblestone", extent=1,
            room=None, block=None, flue_to=None) -> {"ok", "cells", "reason", "cell"}
        Everything after `facing` is keyword-only: fitting(..., mat="sandstone"), not
        fitting(..., "sandstone"). One piece of equipment, built the way it actually
        is. (x, y, z) is the cell it
        stands in; `facing` is the side you use it from, which is what hand-placed
        furniture gets wrong — a furnace facing a wall, a bed with its foot in the
        masonry, an anvil with nowhere to stand.

        hearth (flue_to= carries the flue to a ridge height), forge, anvil, workbench,
        store (extent runs barrels along a wall), bed, table, trough, fodder, light,
        bookshelf, bench, shelf, rug, oven (a furnace with a masonry surround), well.

        **It refuses rather than colliding.** Every cell it needs has to be air with
        something to stand on under it; if one is not, nothing is placed at all and
        `reason` names the cell. `ok` False is not an error — it is the room telling
        you there is already something there. Move it and call again.

        `room` is what the room is *for*, in the planner's words. The only thing that
        reads it is `light`: a hall gets a lantern hung high, a house a candle, a store
        a torch on the wall, a shrine glowstone under a slab. Say nothing and every
        room in the town gets a lantern, which is what happened last time.

        It is a vocabulary, not a decorator: it knows what a forge is made of, and
        nothing about which room anything belongs in. Where things go is yours.
    plinth(x0, z0, x1, z1, y, block, courses=1, course=1, batter=0, overhang=0,
           cap=None, to_grade=True)
        The base a building stands on. The floor block is at y. Courses step out by
        `batter` as they go down and the whole thing is carried to real ground, so
        nothing hangs over a slope. `overhang`/`cap` give the top course a ledge.
    openings(x0, y, z0, x1, z1, at=None, spacing=4, width=1, sill=2, head=3,
             block=None, sill_block=None, lintel=None, reveal=0,
             inward=None) -> [(x,y,z), ...]
        Openings on a rhythm along one straight run of wall. y is the floor level and
        every height is measured from it: the opening runs from y+sill to y+head.
        `reveal` sets the glazing that many blocks back from the face, toward
        `inward`, and clears in front of it. `at` overrides the rhythm.
        `block` defaults to what fills an opening of that shape -- see glazing below.

## Glazing — a pane is not a block

A glass *pane* is thin and joins to whatever is beside it. In a hole one block wide it
is a glazed slit and looks right; in a hole two or more wide, or set back in a reveal,
it reads as an empty hole with sticks in it. `openings()` and `window()` pick for you:
a pane at one block wide and flush, `glass` for anything wider or recessed. Pass
`block=`/`glass=` only when you want something else on purpose — a lattice, bars,
shutters.
    storey_steps(x0, z0, x1, z1, axis="x", bays=3, storey=3, anchor=None)
        -> [{"x0","z0","x1","z1","floor_y","drop"}, ...]
        How a footprint steps down a slope. Each bay's floor is its own median ground
        snapped to a whole number of storeys below the anchor, and no bay is more than
        one storey off its neighbour. storey=1 is a terraced row that follows the
        ground; storey=4 drops a floor at a time down a hillside. Decides nothing
        about what goes on the bays.

Ordinary Python is available: loops, math, random, functions, comprehensions.
Writing the same position twice is fine; the last write wins.

## Coordinates

x and z are horizontal, y is vertical and up. Sea level is y=62.
North is -z, south is +z, east is +x, west is -x.

## The lane outside, and where the floor goes

Where a circulation pass has run, the lanes are already in the ground and your
structure has a doorstep reserved on one of them. These four answer over everything
your program has decided so far, not just what was in the world when you started.

    threshold(label) -> {"id", "lane", "facing", "door", "floor_block_y", "stand_y"}
        The doorstep the circulation pass reserved for this structure, and which way
        you face walking in off the lane. None where nothing was reserved.
    nearest_lane(x, z) -> {"x", "z", "y", "distance"}
        The closest lane cell to a column. None where there is no network.
    floor_from_threshold(label, x=None, z=None)
        -> {"floor_y", "stand_y", "door", "facing", "lane", "source"}
        The circulation pass levelled and cleared your doorstep before you started;
        this is where it is. Put the floor *block* at floor_y and the door's lower
        leaf at stand_y. Do not work a floor level out from the terrain under the
        building: a floor a block above its own doorway sill is a room nobody can
        step into, and that is where six of them came from last round.
    approach(label, x=None, y=None, z=None, mat=None, width=2)
        -> {"ok", "cells", "reason"}
        **Lays the way in.** From your doorway to the lane, following the ground: a
        slab where it rises half a block, a tread where it rises a whole one, headroom
        cut over all of it, two columns wide. Walking only -- no jumping -- which is
        the standard the lanes themselves are held to and the standard your door is
        checked against. It never cuts through anything you have placed and never
        touches a lane cell, and where the door is already walkable it lays nothing
        and says so. Call it once, after the door and the ground around it are in.

## The shell — one call, not four hundred lines

    building(label, x0, z0, x1, z1, storeys, roof, wing=None, outshot=None,
             porch=None, chimney=None, openings="rhythm", stair="auto", mat=None)
        -> {"ok", "floors", "rooms", "door", "ridge_y", "eave_y", "cells", "stairs", ...}

        **Lays a whole building.** The footprint is (x0,z0)-(x1,z1), corners inclusive.
        `storeys` counts floors including the ground one; they sit at floor_y,
        floor_y+4, floor_y+8, and the ground floor is read off your reserved doorstep,
        so you do not work it out. `roof` is a style name — "gable", "hip", "gambrel",
        "mansard", "shed", "flat" — or {"style":..., "axis":..., "pitch":(r,run)}.
        `mat` is the palette by role: {"wall":..., "roof":..., "footing":...,
        "frame":..., "floor":..., "trim":...}, all material *families*.

        It lays, in this order: the plinth carried down to real ground; the shell
        hollowed out of whatever is standing there; the upper floors; walls one block
        thick with a base course and posts; openings on a rhythm on every outside face;
        the doorstep; the door, jambed and lintelled, at the threshold reserved for you;
        the roof with a one-block overhang; the chimney *through* the roof plane; the
        way in with approach(); and a flight between every pair of storeys.

        Extras, each a form and not a decoration:
            wing     = (x0, z0, x1, z1)   a second rectangle sharing one full edge,
                       same floors and eaves, roofed across the other axis so the two
                       roofs meet, with a way through the shared wall at every floor
            outshot  = 3  or  {"side": "north|south|east|west", "depth": 3}
                       a single-storey lean-to under a shallower shed roof
            porch    = True or a depth: a roofed cell in front of the door, on its own
                       deck, roofed at the eaves so it never takes the headroom the way
                       in needs
            chimney  = True, a side name, or (x, z)

        `roof` also takes the parametric keys: {"style":..., "pitch":...,
        "profile":[(1,2),(2,1)], "ends":("irimoya","irimoya"), "eave":"upturned",
        "tiers":2} is handed straight to roof().

        And the vocabulary between the shell and the furniture. **Each of these is a
        parameter, not a set of coordinates.** Do not rasterise them by hand:

            dormers   = 2                 window boxes through the roof slope
            jetty     = 1                 that storey carried out a block on brackets,
                                          wall, floor, glazing and all
            oriel     = ("east", 1)       a projecting bay on that face at that storey
            brackets  = True              a corbel course under the eaves
            flashing  = True              the skirt where the stack meets the slope
            yard      = (x0, z0, x1, z1)  a wall course round a piece of ground, with a
                                          gap where the lane comes in
            deck      = True              a platform on piles to the bed, wherever this
                                          building stands over water

        They come back in `extras` as {"ok", "reason", ...} each. `ok` False is one
        decoration that could not be laid and never costs you the building: read the
        reason, move it or drop it.

        `openings="none"` leaves the walls blank. `stair="none"` leaves the storeys
        unconnected, for a building whose way up you are laying yourself.

        **It refuses, placing nothing, and says why** if the footprint or an extra
        leaves your plot, if a wing would swallow the wall the door has to go in, or if
        the inside is too small to carry the flight those storeys need. A refusal is
        the library telling you its vocabulary does not reach that shape: move it,
        widen it, or build that one by hand.

        `rooms` comes back as [(x0, y, z0, x1, z1), ...] — the interior rectangle and
        floor level of every room it made. Furnish them. Everything else about the
        building is still yours: place anything you like by hand afterwards.

    dais(x0, z0, x1, z1, y, mat) -> {"ok", "side", "cells", "step_cells", "reason"}
        A raised floor one block up that lays its own slab step on its longest open
        side. Use it for a sleeping platform, a stall floor, a raised end of a hall.
        A one-block rise with no tread onto it is floor nobody can reach.

## Getting upstairs

    flight(label, x, z, y0, y1, facing, mat=None)
        -> {"ok", "cells", "removed", "reason"}
        **Lays the stair between two floors.** A straight internal flight from the
        floor at y0 to the floor at y1, one column wide. (x, z) is where the *first
        tread* goes, at (x, y0+1, z) -- one block in from where you stand on the lower
        floor, in the direction you are climbing. `facing` is that direction:
        "north", "south", "east" or "west". Each tread after it steps one column along
        and one block up, so the flight is y1-y0 treads and the cell past the last one
        is a landing on the upper floor.

        It cuts the stairwell through the floor it comes up into and clears the two
        cells over every tread, so you do not need to open a hole first. It refuses,
        placing nothing, if a tread or the landing would replace something you placed
        that is not floor or air -- a wall, a chimney, a fitting -- and names the cell,
        so a refusal means move the flight, not force it. Then it walks the flight
        itself and tells you whether the landing can be reached from the bottom.

        A loft, an upper storey, a tower chamber or a cellar needs one of these. Do not
        lay treads up through a floor by hand.

## Checking your own work

    check_door(x, y, z) -> {"ok", "reason", "jumps", "nearest_lane"}
        Can a person walk to this doorway from the lane, without jumping? Call it on
        every door you place, after the walls around it. If `ok` is False, either move
        the door somewhere else on the network or call approach() to lay the way to it.
    check_walkable(label=None, x0=None, z0=None, x1=None, z1=None)
        -> {"ok", "rooms": [{"bbox", "cells", "walkable", "fraction"}], "reason"}
        Walking only, from your own doorway: how much of each room's floor can be
        reached. `ok` is False when a room cannot be walked into at all -- that is an
        error, fix it. A `fraction` below 1 is a report, not a verdict: a raised
        sleeping platform or a dais is architecture. Call it after your floors, stairs
        and fittings are in. Scoped to the plot you reserved for `label`; with no
        label it reports each plot you reserved on this pass, one by one.
    check_attached() -> {"ok", "floating": [...]}
        Every piece of what you have placed that is held up by nothing. Eaves,
        jetties and balconies are not listed -- they reach the ground through the
        build. Call it before you finish.
    seal_voids(x0, z0, x1, z1, block, max_cells=64) -> {"filled", "left": [...]}
        Enclosed standing space with no way into it, inside the footprint you give.
        Fills only pockets it can prove are enclosed, unreachable and yours; anything
        else it reports and leaves alone, with the reason. Do not write your own
        version that floods and packs -- one did, and packed 625 cells of stone above
        its own eaves.
    dress_ground(x0, z0, x1, z1, cover=None)
        Finish ground you have worked: sweep leaf litter and loose growth off it
        and put the surface skin back. `cover` defaults to whatever the
        undisturbed ground around the area is made of.

## Rules

- Stay inside the build area you are given.
- No file access, no input, no imports beyond math and random.
- **No `try`.** Do not catch exceptions. If a call fails you will be shown the line
  and the error and given the chance to fix it; a program that swallows its own
  failures builds part of a building and reports a whole one, and the checks above
  cannot tell the two apart. A program containing `try` is rejected before it runs.
'''


CRAFT_VILLAGE = """
## How builders make this look built, not generated



These are guidelines from the Minecraft building literature, not rules. Break any of
them on purpose. But a build that ignores all of them reads as generated.

- **Depth beats decoration.** A flat run of one block is the giveaway. Post every 3-5
  blocks, recess windows a block, run a slab ledge at the base, put an upside-down stair
  under a sill. Varying the surface by one block is what makes light and shadow.
- **Overhang the roof** past the walls by 1-2. The shadow line is what separates the
  building from the sky.
- **Vary the pitch.** A 45-degree gable on everything is the single clearest sign that
  a machine built it. Shallow (1,2) reads calm; steep (2,1) reads defensive.
- **Contrast the roof against the walls.** Do not roof a wooden house in the same wood.
- **Odd widths** give a true centre block for a door, a window or a ridge.
- **Keep the palette small** — about 3-5 materials that share a tone but differ in
  texture. More than that reads as confetti. Variants of one material (mossy, cracked,
  polished) count as one.
- **Big shapes first, details last.** Silhouette carries further than texture.
- **Break symmetry somewhere.** Push a wing, a porch or a chimney off-centre.
- **Scale the details to the build.** Big dormers and chimneys overwhelm a small house.
- **Interiors: give each room a focal point** — a hearth, a window seat, a stair, a
  workbench — one, not all, and a different one in the next room. Lay the furniture out
  from it. Do not fill every corner; empty floor is fine, and identical repeated rooms
  are worse than fewer varied ones. Light in layers.
- **Meet the ground honestly.** Follow the slope, step the foundation, feather the edge.
  A flat pad on a hillside is the most-cited tell there is.
"""
