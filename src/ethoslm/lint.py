"""A build linter: deterministic checks for *broken*, never for *ugly*.

Correctness is not taste and this project conflated them for four rounds. Backwards
stairs and unreachable doors are bugs; they want checks that run on every pass, cost no
model calls, and give the same answer twice. Whether a town is beautiful is not in here
and must not get in here -- nine years of GDMC say that scorer does not exist.

**Shape borrowed from compilers**, per the spec's "worth stealing from":

    ERROR    the build is broken. Something cannot be entered, reached, or placed.
             A pass with errors has failed, whatever it looks like.
    WARNING  legitimate but probably not meant. A room nobody lit; a door you can only
             reach by scrambling up a bank.
    STYLE    craft-rule departures. Reported, never enforced.

**Where it runs.** Two places, because the cheap one has to come first:

**Blocking or annotating.** Errors do not roll the world back on their own: that
decision needs a human or a caller with a snapshot, and a half-built pass is often more
informative than none. `Report.ok` is False and the caller decides. Warnings and style
annotate only. This is deliberate -- a linter that silently reverted work would make
the failure invisible, which is the thing we are trying to stop doing.

Every check names the taxonomy row it exists to catch. A check that cannot point at one
does not belong here."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import observe, registry

ERROR, WARNING, STYLE = "error", "warning", "style"
_ORDER = {ERROR: 0, WARNING: 1, STYLE: 2}


@dataclass
class Finding:
    code: str                     # E002, W001 -- stable, greppable
    severity: str
    message: str                  # one line, specific, no hedging
    pos: tuple | None = None
    detail: dict = field(default_factory=dict)


#: Which question a check asks. "build" judges a structure and holds for a castle, a
#: ship or a statue as much as a house; "place" judges a settlement -- circulation,
#: thresholds, plots -- and means nothing for a single object. The split exists so the
#: build family can be run on something that is not a town, which is the whole point of
#: the project, without inventing a second linter for it.
BUILD, PLACE = "build", "place"


#: Which kinds of part have an inside. A part is a plot, an edge, a point or an area,
#: and two of those four are not things anybody goes into: a wall is an **edge** and a
#: gate is a **point**. Sheltered standing space on a wall's swept line is the space
#: between its merlons and under its arches; charging it to the wall as interior floor
#: made a city's headline walkability number partly a reading of its own masonry. An
#: `area` stays in, because the one area type this project has that is a building -- a
#: palace compound -- is an area, and an area that builds no rooms contributes nothing
#: either way. The spec's third clause, "an area without rooms", is therefore a no-op
#: and is left as one rather than written as a second opinion about which areas those
#: are. Its second clause, "a type whose FORM is fortification", is subsumed: every
#: fortification in this place is an edge or a point, and the one that is a plot -- a
#: keep -- is a building a person walks into and is not excluded.
INTERIOR_KINDS = ("plot", "area")


def plot_rects(p: dict) -> list:
    """What a registry row actually covers: its `rects` if it has them, else its box."""
    rs = p.get("rects")
    if not rs:
        return [(min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
                 max(p["x0"], p["x1"]), max(p["z0"], p["z1"]))]
    return [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))
            for r in rs]


def plot_covers(p: dict, x, z, margin: int = 0) -> bool:
    """Is this point on the ground the row is answerable for? See `plot_rects`."""
    return any(a - margin <= x <= c + margin and b - margin <= z <= d + margin
               for (a, b, c, d) in plot_rects(p))


def rects_overlap(a: list, b: list) -> bool:
    """Do two rectangle sets share a column? See `plot_rects`."""
    return any(p[0] <= q[2] and q[0] <= p[2] and p[1] <= q[3] and q[1] <= p[3]
               for p in a for q in b)


def lane_stances(nav: observe.Nav, network) -> list:
    """Where a walker stands on each cell of a declared circulation network.

        One implementation, because three things seed a walk-only flood from the lane --
        `Context.from_outdoors`, `snapshot()` and `_doors_from_the_lane` -- and three copies
        of "where the lane is standable" would be three answers. `tol=1`, not 2: a stance a
        whole block above the lane surface is a stance on top of something that has been put
        there, and seeding from it would call somebody's plinth the street.

        Empty where there is no network, or where nothing on it can be stood on.
        
    """
    if network is None:
        return []
    out = []
    for (x, z, y) in network.surface():
        s = nav.stance_near(x, z, y + 1, tol=1)
        if s is not None:
            out.append((x, z, s))
    return out


@dataclass
class Check:
    code: str
    severity: str
    title: str
    catches: str                  # the taxonomy row this exists for
    fn: object
    family: str = BUILD


CHECKS: list[Check] = []


def check(code: str, severity: str, title: str, catches: str, family: str = BUILD):
    def deco(fn):
        CHECKS.append(Check(code, severity, title, catches, fn, family))
        return fn
    return deco


@dataclass
class Context:
    """Everything the checks share, computed once.

        Building this is the expensive part (a second or two on a 192x192 settlement) and
        every check after it is essentially free, which is what lets the whole suite run on
        every pass.
        
    """

    vol: observe.Volume
    nav: observe.Nav
    sky_open: object
    sealed: object
    rooms: list
    plots: list = field(default_factory=list)
    circulation: dict = field(default_factory=dict)
    from_anywhere: dict = field(default_factory=dict)
    #: The walk-only flood -- `max_jumps=0` -- **seeded from the lane** where a
    #: circulation network exists, and from the volume's perimeter only where there is
    #: none. E002, W001 and `diagnose_entry` read this one. Every other reachability
    #: guarantee in the suite still reads `from_anywhere` and still means "a person
    #: could get there, possibly by pressing space repeatedly" -- see `interior_walk`.
    #: `perimeter_seeds` are the edge of whatever rectangle the world happened to be
    #: cached into -- 48 blocks out for a pre-build volume, 16 for a built one -- and a
    #: walk-only flood inward from there over broken ground rises half a block at a time
    #: and gets nowhere. five of six builders spent effort on twenty findings that were
    #: all artefacts. From the lane the same volume gives 16 of 16. The lane is the
    #: outdoors a settlement's doors actually front onto, it is walk-only end to end by
    #: construction, and it does not move when the cache does.
    from_outdoors: dict = field(default_factory=dict)
    from_network: dict = field(default_factory=dict)
    doors: list = field(default_factory=list)
    placed: dict | None = None       # a pass's own palette, when linting one pass
    #: What the circulation pass says it built, and what it reserved. The network is
    #: still *derived* for every check that asks whether something is walkable -- see
    #: `circulation` above. This is only ever used to ask the different question of
    #: whether the thing that was built is still there and still joined up.
    network: object = None           # circulate.Network
    sites: list = field(default_factory=list)     # planned footprints, pre-build
    before: dict | None = None       # a reachability snapshot taken before this pass
    #: (x0, z0, x1, z1) of what this settlement is answerable for. The volume is wider
    #: than the site on purpose -- you cannot measure whether you can walk *out* of a
    #: place from inside it -- but a vanilla village 40 blocks west of the site is not
    #: our build, and reporting its stairs as our defects is noise. Checks that judge
    #: placed blocks are confined to this; checks that judge reachability are not.
    region: tuple | None = None
    #: The settlement's declared voice: role -> material family. Declaring it is what
    #: makes palette drift checkable at all.
    voice: dict | None = None
    #: Labels of the plots this pass claimed. Ground inside them is ground it owns and
    #: is entitled to build over and enclose; ground outside them is not.
    own_plots: tuple = ()
    #: The massing stage's occupancy, when this build came through the staged pipeline:
    #: {(x, z): (y_lo, y_hi)} per column, from `massing_occupancy`. None -- the usual
    #: case, and every pre-staging caller -- disables W010 entirely.
    massing: dict | None = None
    #: Which seeds `from_outdoors` was flooded from: "lane" or "perimeter". Reported so
    #: a reading of E002 says which question it answered.
    outdoor_seeds: str = "perimeter"
    #: The library's record of which cells it laid as furniture --
    #: `Builder.fitting_cells` -- or None where there is no record, which is every
    #: cached town whose builder is long gone. `observe.floor_stances` reads it. A
    #: number read with it and a number read without it are not the same number, and
    #: `stage_readout` reports which of the two any figure is.
    fittings: set | None = None

    #: A room this open to the sky is a porch, an arcade or a mesa overhang, not an
    #: interior. Only rooms at or above this are held to the "must be enterable" rule.
    ENCLOSED = 0.85
    #: ...and a space whose walls are this natural is a cave, whoever owns the surface
    #: above it. Below this a room is not attributed to a plot at all.
    MADE = 0.25
    #: Below this share of its floor walk-reachable from its own door, a room is
    #: **reported with the number** (W011). Pre-registered at 0.5 before it was run
    #: against any settlement, and deliberately generous: a raised sleeping platform, a
    #: mezzanine or a dais is architecture, and this project has caught a check
    #: deforming a design three times already -- the seal-check that packed 625 cells,
    #: the E004 rule that invented flues at the ridge, `_families` pushing a roof
    #: material. Half the floor is where "a platform in the corner" stops being a
    #: plausible description of what you built.
    WALKABLE_WARN = 0.5

    _interior_walk: list | None = field(default=None, repr=False, compare=False)

    @classmethod
    def build(cls, vol: observe.Volume, plots: list | None = None,
              placed: dict | None = None, network=None, sites: list | None = None,
              before: dict | None = None, region: tuple | None = None,
              own_plots=(), voice: dict | None = None,
              massing: dict | None = None, fittings: set | None = None,
              base: observe.Volume | None = None) -> "Context":
        nav = observe.Nav(vol)
        sky_open, sealed = observe.shelter(vol)
        rooms = observe.rooms(nav, sky_open,
                              region=(region[0], region[1], region[2] + 1, region[3] + 1)
                              if region else None, fittings=fittings)
        # On foot, from the lane. See `from_outdoors` above for the measurement that
        # moved these seeds.
        lane_seeds = lane_stances(nav, network)
        # ...and the lane is where the circulation flood looks first, which is an order
        # and not an answer: see `Nav.circulation`.
        circ = nav.circulation(sky_open, first=lane_seeds)
        seeds = nav.perimeter_seeds(inset=2, step=3)
        # ...and the jumping flood starts from **both**. It used to start from the
        # perimeter alone, on the reasoning that a jumping flood does reach inward from
        # the edge of a cache -- which is true of a whole settlement and false of a cut
        # one. The lake is between them. E003 then reported the building's one room,
        # walked into off its own doorstep, as "enclosed standing space with no way in".
        # The lane is outdoors by construction and is where a person arrives, so seeding
        # from it as well can only add reachability, and adds none where the perimeter
        # already reaches it.
        anywhere = nav.flood(seeds + lane_seeds)
        outdoors = nav.flood(lane_seeds or seeds, max_jumps=0)
        doors = []
        for (x, y, z) in vol.find(lambda s: s.split("[")[0].endswith("_door")
                                  or s.split("[")[0].endswith("_fence_gate")):
            if observe.parse_props(vol.state(x, y, z)).get("half") == "upper":
                continue
            if region and not (region[0] <= x <= region[2]
                               and region[1] <= z <= region[3]):
                continue
            # `region` says "a vanilla village 40 blocks west of the site is not our
            # build" and confines the checks that judge placed blocks to it -- but a
            # city's footprint is 512 a side, and **the site the system chose has a
            # generated village inside it**. They are a fact about the ground, not a
            # defect of the build, and no bar this project registered was ever about
            # them: the counts it was calibrated on -- 26 of 26, 1 of 14, 1 of 62 -- are
            # the build's own doors. `base` is the world as it stood before this build.
            # Where it is not given nothing changes, which is every caller that has no
            # such snapshot.
            if base is not None and base.inside(x, y, z):
                stood = base.state(x, y, z) or ""
                if stood.split("[")[0].endswith(("_door", "_fence_gate")):
                    continue
            doors.append((x, y, z))
        # Jump cost measured *from the walkable network*, which is the thing a person is
        # actually standing on. Measuring it from the perimeter instead produces "cannot
        # be walked to -- 0 jumps", which is both wrong and self-contradictory.
        from_network = nav.flood(list(circ)) if circ else {}
        ctx = cls(vol, nav, sky_open, sealed, rooms, plots or [], circ, anywhere,
                  outdoors, from_network, doors, placed, network, sites or [], before,
                  region, voice, tuple(own_plots), massing,
                  "lane" if lane_seeds else "perimeter", fittings)
        for r in ctx.rooms:
            r["plot"] = ctx.room_owner(r)
        return ctx

    # --- the built network, as it now stands ------------------------------
    def lane_stances(self) -> list:
        """Where a walker stands on each cell the circulation pass laid, or None if
        something is now standing there."""
        if not self.network:
            return []
        # tol=1, not 2: a stance a whole block above the lane surface is a stance on top
        # of something that was put there, which is the case this is looking for.
        return [((x, z), self.nav.stance_near(x, z, y + 1, tol=1))
                for (x, z, y) in self.network.surface()]

    def lane_pieces(self, cap: int = 8) -> list:
        """The lane, split into however many pieces you can actually walk between.

                Floods from a lane stance without jumping, then from the first lane stance the
                flood did not reach, and so on. One piece is the answer; anything else is a
                settlement with a cliff through the middle of it, which is precisely the failure
                the spec says no check we had would have reported.
                
        """
        todo = [(x, z, s) for (x, z), s in self.lane_stances() if s is not None]
        pieces = []
        left = set(todo)
        while left and len(pieces) < cap:
            seed = next(p for p in todo if p in left)
            got = set(self.nav.flood([seed], max_jumps=0))
            piece = left & got
            if not piece:
                piece = {seed}
            pieces.append(piece)
            left -= piece
        if left:
            pieces.append(left)
        return pieces

    def plot_at(self, x, z) -> str | None:
        """Which plot covers this point. None means it belongs to no structure -- on a
                badlands site that is usually a natural overhang or a cave, not an interior.

                This is the **ground** question -- which part is answerable for this column --
                and it is what E009, E010, S002 and W008 ask. Which part a *room* belongs to is
                `room_owner`, and it is a different question with a third dimension in it.
                
        """
        for p in self.plots:
            if plot_covers(p, x, z):
                return p["label"]
        return None

    def room_owner(self, room: dict) -> str | None:
        """Whose interior this room is, or None -- in which case it is landscape.

                1. **Was it made or found?** `made >= MADE`, unchanged since the hierarchy round.
                2. **Is the part it stands on a thing with an inside?** See `INTERIOR_KINDS`.
                3. **Is the room in the building, or under it?** A part records `y0`, the level
                   the library sited it at, and a type may not write below `part["floor_y"]` --
                   preflight's E013 refuses it by name and has since the hierarchy round. So a room lying
                   **entirely below** `y0` cannot be anything the part built: it is a cave under
                   the plot, and it is the mechanism of open thread 6 at a city's scale. Ba Sing
                   Se charged one minka a 5,802-cell cavern at y=1 under its plot, `made` 0.27
                   against a threshold of 0.25 and not one cell of it walkable, plus a
                   952-cell one at y=17 under another. Those two rooms alone are 6,754 of the
                   21,875 floor cells the city was read over, and the minka -- the type the
                   round blamed -- reads 27.0% with them and **100.0%** without.

                A row with no `y0` is a part built before this was recorded, and gets test 3
                waived rather than guessed at, which is why no earlier round's number moves.
                The rooms it lets through are still checked by tests 1 and 2.
                
        """
        if room.get("made", 1.0) < self.MADE:
            return None
        b = room["bbox"]
        cx, cz = (b[0] + b[3]) / 2, (b[2] + b[5]) / 2
        for p in self.plots:
            if not plot_covers(p, cx, cz):
                continue
            if p.get("kind", "plot") not in INTERIOR_KINDS:
                continue
            y0 = p.get("y0")
            if y0 is not None and b[4] < y0:
                continue
            return p["label"]
        return None

    def door_stance(self, x, y, z):
        """The stance in the doorway itself, within a block of the sill."""
        return self.nav.stance_near(x, z, y, tol=2)

    # --- can you actually walk about inside? -------------------------------
    def interior_walk(self, margin: int = 4) -> list[dict]:
        """Per room on a plot: what fraction of its floor you can **walk** to from that
        building's own doorways. Cached, because three things ask for it.

        Seeded from the building's **own** doorways, not from the perimeter, because
        the question a builder needs answered is "can somebody who has come through my
        door get about inside", and that is answerable while the program is running.
        Bounded to the plot plus a margin for the same reason it is seeded that way."""
        if self._interior_walk is not None:
            return self._interior_walk
        by_plot: dict = {}
        for p in self.plots:
            by_plot[p["label"]] = (min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
                                   max(p["x0"], p["x1"]), max(p["z0"], p["z1"]))
        # a door "belongs" to a plot if it stands on it or in the ring just outside it
        # -- on what the row **covers** since A3, so a wall whose bounding box contains
        # the district does not adopt every door in it. `leaves` counts door *blocks*
        # and `seeds` counts doorways you can stand in: a doorway obstructed at head
        # height has the first and not the second, and that is a building you cannot
        # walk into, not a building with no door.
        seeds: dict = {}
        leaves: dict = {}
        for (x, y, z) in self.doors:
            for p in self.plots:
                if plot_covers(p, x, z, margin=1):
                    lab = p["label"]
                    leaves[lab] = leaves.get(lab, 0) + 1
                    s = self.door_stance(x, y, z)
                    if s is not None:
                        seeds.setdefault(lab, []).append((x, z, s))
        reach: dict = {}
        for lab, (a, b, c, d) in by_plot.items():
            if lab not in seeds:
                continue
            reach[lab] = set(self.nav.flood(
                seeds[lab], max_jumps=0,
                bounds=(a - margin, b - margin, c + margin, d + margin)))
        out = []
        for r in self.rooms:
            lab = r.get("plot")
            if not lab:
                continue
            # A porch, an arcade, a mesa overhang or the covered band at a two-tier
            # roof's junction is not an interior, and this is the **third** reader of
            # "is this a room". this one counted them, so a builder's own checker and
            # the number its round is read against were answering different questions on
            # the same building. A covered band reads at enclosure 0.0 and two of the
            # three readers drop it. `Context.ENCLOSED` is the one threshold.
            if r.get("enclosure", 1.0) < self.ENCLOSED:
                continue
            # The threshold is not floor. A stance in the doorway itself is part of the
            # room component, and counting it made "you cannot walk in at all" come back
            # as one cell out of thirty-seven rather than as zero. ...and neither is the
            # top of a barrel the library placed. But the top of a dais is.
            # `observe.floor_stances` is the one definition of what a room's floor is
            # and every measure of walkability in this project reads it -- this call,
            # `pipeline.diagnose_entry`, `pipeline.measure_program`,
            # `Builder.check_walkable` and `scripts/walk_fraction.py`. See its
            # docstring.
            st = set(map(tuple, r["floor"])) - set(seeds.get(lab, ()))
            if not st:
                continue
            got = st & reach.get(lab, set())
            out.append({"plot": lab, "bbox": r["bbox"], "cells": len(st),
                        "walkable": len(got),
                        "fraction": round(len(got) / len(st), 3),
                        "enclosure": r.get("enclosure", 1.0),
                        "doors": leaves.get(lab, 0),
                        "thresholds": len(seeds.get(lab, ()))})
        self._interior_walk = out
        return out


# ------------------------------------------------------------------ correctness

@check("E001", ERROR, "invalid block state",
       "Four blocks rejected as unknown ids")
def e001_block_states(ctx: Context):
    for state, errs in registry.check_all(ctx.vol.palette).items():
        yield Finding("E001", ERROR, f"{state}: {errs[0]}", detail={"state": state})


@check("E002", ERROR, "door not reachable on foot",
       "Buildings cannot be entered; doorways lead nowhere")
def e002_door_unreachable(ctx: Context):
    """On foot means on foot: `max_jumps=0`, the standard the lanes have always been
    held to."""
    for (x, y, z) in ctx.doors:
        s = ctx.door_stance(x, y, z)
        if s is None:
            yield Finding("E002", ERROR,
                          f"no standable threshold in the doorway at ({x},{y},{z})",
                          (x, y, z))
        elif (x, z, s) not in ctx.from_outdoors:
            jumps = ctx.from_anywhere.get((x, z, s))
            yield Finding("E002", ERROR,
                          f"door at ({x},{y},{z}) cannot be reached on foot from "
                          f"anywhere outdoors"
                          + (f" -- getting to it needs {jumps} jump"
                             f"{'s' if jumps != 1 else ''}" if jumps else ""),
                          (x, y, z))


@check("E003", ERROR, "room not reachable from any entrance",
       "Buildings cannot be entered; doorways lead nowhere")
def e003_room_unreachable(ctx: Context):
    for r in ctx.rooms:
        if not r.get("plot"):
            continue        # not on any plot: a cave or a cliff, not somebody's interior
        if r.get("enclosure", 1.0) < Context.ENCLOSED:
            continue        # a porch or an overhang; nothing to be shut out of
        if not (set(map(tuple, r["stances"])) & set(ctx.from_anywhere)):
            b = r["bbox"]
            yield Finding("E003", ERROR,
                          f"{r['plot']}: a room of {r['cells']} cells at "
                          f"({b[0]},{b[1]},{b[2]}) cannot be walked into",
                          (b[0], b[1], b[2]), {"plot": r["plot"], "cells": r["cells"]})


@check("E011", ERROR, "you cannot walk into this room from its own door",
       "Buildings cannot be entered; doorways lead nowhere")
def e011_interior_not_walkable(ctx: Context):
    """The interior half of E002/E003, and the one check this suite was missing.

        E003 asks whether a room can be *reached*, and reads `from_anywhere`, which allows
        unlimited jumping. So it passes a room whose floor sits a block above its own
        doorway sill -- you get in by hopping -- and its own advice text says to "look for
        floors laid a block above their own doorway sill", which is the defect it cannot
        detect, written down in the check that cannot detect it.

        **Severity is earned, and this is the part that matters.** A room you cannot walk
        into at all is broken: that is an ERROR, and it is the case the person on foot hit.
        A room with a step up to a platform is *architecture*, and a linter that forbids it
        is a linter that deforms designs -- so anything short of nothing is a WARNING that
        reports the fraction and nothing more. There is deliberately no rule here about
        where furniture may go, no minimum fraction that fails a build, and no second check.
        
    """
    for r in ctx.interior_walk():
        if r["enclosure"] < Context.ENCLOSED:
            continue          # a porch or an overhang; nothing to be shut out of
        b, pct = r["bbox"], f"{r['fraction']:.0%}"
        where = (b[0], b[1], b[2])
        if not r["doors"]:
            continue          # no door anywhere is W002's finding, not this one
        if r["walkable"] == 0:
            yield Finding("E011", ERROR,
                          f"{r['plot']}: none of the {r['cells']} floor cells of the "
                          f"room at ({b[0]},{b[1]},{b[2]}) can be walked to from its "
                          f"own doorway -- you can only get in by jumping",
                          where, {"plot": r["plot"], "fraction": r["fraction"],
                                  "cells": r["cells"], "walkable": 0})
        elif r["fraction"] < Context.WALKABLE_WARN:
            yield Finding("W011", WARNING,
                          f"{r['plot']}: {pct} of the {r['cells']} floor cells of the "
                          f"room at ({b[0]},{b[1]},{b[2]}) can be walked to from its "
                          f"own doorway; the rest needs a jump",
                          where, {"plot": r["plot"], "fraction": r["fraction"],
                                  "cells": r["cells"], "walkable": r["walkable"]})


@check("W011", WARNING, "part of a room's floor needs a jump to reach",
       "Buildings cannot be entered; doorways lead nowhere")
def w011_partly_walkable(ctx: Context):
    """Registered so the code has a title and a fix in the catalogue. E011 emits it --
    one check, two severities, exactly as E009 emits W006."""
    return iter(())


@check("E004", ERROR, "stair faces down-slope",
       "Stairs placed backwards throughout")
def e004_backwards_stairs(ctx: Context):
    for s in observe.backwards_stairs(ctx.vol, region=ctx.region):
        yield Finding("E004", ERROR,
                      f"{s['state'].split('[')[0]} at {s['pos']} faces down-slope "
                      f"(ahead y={s['ahead_y']}, behind y={s['behind_y']})",
                      tuple(s["pos"]))


@check("E005", ERROR, "connective block never joined up",
       "Fences and decoration placed with no reason / build reads as unfinished")
def e005_unconnected(ctx: Context):
    j = observe.unconnected_joins(ctx.vol, region=ctx.region)
    for kind in ("fence", "wall", "pane"):
        if j[kind]["missed_joins"]:
            yield Finding("E005", ERROR,
                          f"{j[kind]['missed_joins']} {kind} joins that should exist "
                          f"do not ({j[kind]['blocks']} {kind} blocks) -- placed "
                          f"without block updates",
                          detail={"kind": kind, **j[kind]})


@check("E006", ERROR, "element built inside another plot",
       "Buildings do not connect to each other properly", PLACE)
def e006_plot_intersection(ctx: Context):
    """Overlap of what the parts **cover**, which for an edge is its swept line. A3."""
    for i, a in enumerate(ctx.plots):
        for b in ctx.plots[i + 1:]:
            # ...except a gate in its wall. A part whose type declares `PASSAGE` is the
            # one crossing of an edge, and a crossing that did not touch the thing it
            # crosses would be a gap beside a gate. `pipeline.part_registry_row` is what
            # writes the flag and it comes off the type, not off the plan.
            if ((a.get("passage") and b.get("kind") == "edge")
                    or (b.get("passage") and a.get("kind") == "edge")):
                continue
            if rects_overlap(plot_rects(a), plot_rects(b)):
                yield Finding("E006", ERROR,
                              f"plots {a['label']} and {b['label']} overlap",
                              detail={"a": a["label"], "b": b["label"]})


@check("E007", ERROR, "circulation is not one network",
       "Buildings do not connect to each other properly", PLACE)
def e007_network_split(ctx: Context):
    """The check the spec says nothing we had would make.

        A circulation pass that produces two networks with a cliff between them has failed,
        and every other check would call it fine: each half is walkable, every door on it is
        reachable, every plot on it fronts something.
        
    """
    if not ctx.network:
        return
    buried = [c for c, s in ctx.lane_stances() if s is None]
    if buried:
        yield Finding("E007", ERROR,
                      f"{len(buried)} lane cells cannot be stood on -- the first at "
                      f"{list(buried[0])}", tuple(buried[0]) if buried else None,
                      {"buried": len(buried)})
    pieces = ctx.lane_pieces()
    if len(pieces) > 1:
        sizes = sorted((len(p) for p in pieces), reverse=True)
        for p in pieces[1:]:
            x, z, s = min(p)
            yield Finding("E007", ERROR,
                          f"a piece of the lane network of {len(p)} stances near "
                          f"({x},{s // 2},{z}) cannot be walked to from the rest",
                          (x, s // 2, z), {"sizes": sizes})


@check("E008", ERROR, "reserved threshold obstructed",
       "Entrances placed where no one would arrive from -- the headline defect", PLACE)
def e008_threshold_obstructed(ctx: Context):
    """The one thing shared state carries, and the only reason it carries it.

        The circulation pass owns the few blocks where a lane meets a doorway, and records
        each as a position and an approach direction. Nothing the world can be asked will
        recover that intent later -- reading a wall does not tell you which side was meant
        to be the front -- so a later pass building over one has to be caught here.
        
    """
    if not ctx.network:
        return
    for t in ctx.network.thresholds:
        s = ctx.nav.stance_near(t.x, t.z, t.y + 1, tol=1)
        if s is None:
            yield Finding("E008", ERROR,
                          f"the threshold reserved for {t.id} at ({t.x},{t.y},{t.z}) "
                          f"has been built over", (t.x, t.y, t.z), {"id": t.id})
            continue
        if ctx.circulation and (t.x, t.z, s) not in ctx.circulation:
            jumps = ctx.from_network.get((t.x, t.z, s))
            how = f"{jumps} jumps off it" if jumps else "not connected to it on foot"
            yield Finding("E008", ERROR,
                          f"the threshold reserved for {t.id} at ({t.x},{t.y},{t.z}) "
                          f"is no longer on the walkable network -- {how}",
                          (t.x, t.y, t.z), {"id": t.id, "jumps": jumps})
            continue
        # The doorstep, not just the lane it comes off. This pass levelled the cell the
        # door leaf stands in; if a builder walled it up, or floored it a block higher,
        # the lane still passes and the door is still unenterable -- which is the
        # headline defect wearing a different hat.
        dx, dy, dz = t.door
        ds = ctx.nav.stance_near(dx, dz, dy, tol=1)
        if ds is None or (ctx.circulation and (dx, dz, ds) not in ctx.circulation):
            yield Finding("E008", ERROR,
                          f"the doorway reserved for {t.id} at ({dx},{dy},{dz}) cannot "
                          f"be walked into off its own threshold", (dx, dy, dz),
                          {"id": t.id, "door": True})


@check("E009", ERROR, "this pass severed a connection that existed before it",
       "Buildings do not connect to each other properly", PLACE)
def e009_network_regression(ctx: Context):
    """Buildings grade the terrain they sit on, so a pass can cut a network that was
        correct when it was built. Nothing detected that, and it is the same class of
        failure as the ordering defect: invisible, and only exhibited by a defect nobody was
        looking for.

        Compares like with like. A stance the pass *built over* is not a regression -- a
        wall is allowed to occupy ground. A stance that is still standable and is no longer
        reachable is one, and it is the only thing counted here.
        
    """
    if not ctx.before:
        return
    seeds = [p for p in ctx.before["seeds"] if ctx.nav.can_stand(*p)]
    now = set(ctx.nav.flood(seeds, max_jumps=0)) if seeds else set()
    # Standing on a leaf is not standing on the town. A server keeps ticking while a
    # pass runs.
    lost = [p for p in ctx.before["reachable"]
            if p not in now and ctx.nav.can_stand(*p)
            and not observe._is_vegetation(ctx.vol.name(p[0], p[2] // 2 - 1, p[1]))]
    if not lost:
        return
    gone = len(ctx.before["seeds"]) - len(seeds)
    ex = sorted(lost)[:3]
    # Severity by what was cut off, not by how much. Wave 1 lost 22 stances of open
    # hillside two blocks outside its own plot, while the lane stayed one walkable piece
    # of 633 cells: that is worth saying and is not the same failure as cutting the town
    # in half. What makes it an error is losing ground the town *needs* -- lane, a
    # reserved threshold, or ground inside somebody's plot. Written down because the
    # distinction was drawn after wave 1 tripped the check, which is exactly when a
    # check is most likely to be bent to fit; the benign case is still reported.
    lanes = {(x, z) for (x, z, _) in ctx.network.surface()} if ctx.network else set()
    thresholds = {(t.x, t.z) for t in ctx.network.thresholds} if ctx.network else set()
    # A pass's own plots are excluded: enclosing the ground inside a footprint you just
    # claimed is building, not severing. Wave 2 was charged with 167 stances of its own
    # house-row yards before this was drawn.
    def needs(p):
        if (p[0], p[1]) in lanes or (p[0], p[1]) in thresholds:
            return True
        owner = ctx.plot_at(p[0], p[1])
        return bool(owner) and owner not in ctx.own_plots

    needed = [p for p in lost if needs(p)]
    where = f" -- e.g. {[[x, s // 2, z] for x, z, s in ex]}"
    if needed or gone:
        yield Finding("E009", ERROR,
                      f"{len(lost)} stances that were walkable before this pass no "
                      f"longer are, though nothing was built on them; {len(needed)} of "
                      f"them are lane, threshold or plot ground"
                      + (f" and {gone} lane cells were built over" if gone else "")
                      + where, (ex[0][0], ex[0][2] // 2, ex[0][1]),
                      {"lost": len(lost), "needed": len(needed), "seeds_lost": gone})
    else:
        yield Finding("W006", WARNING,
                      f"{len(lost)} stances of open ground outside every plot can no "
                      f"longer be walked to, though nothing was built on them; the "
                      f"network itself is intact" + where,
                      (ex[0][0], ex[0][2] // 2, ex[0][1]), {"lost": len(lost)})


@check("E010", ERROR, "a piece of the build is attached to nothing",
       "Fences and decoration placed with no reason / build reads as unfinished")
def e010_unsupported(ctx: Context):
    """Found by a human on foot in hall whose roof and top course of walls"""
    rects = [(p["x0"], p["z0"], p["x1"], p["z1"]) for p in ctx.plots]
    for f in observe.unsupported(ctx.vol, region=ctx.region,
                                 regions=rects if rects else None):
        if f.get("natural"):
            continue        # the hillside under the plot, not the build on it
        b = f["bbox"]
        cx, cz = (b[0] + b[3]) // 2, (b[2] + b[5]) // 2
        plot = ctx.plot_at(cx, cz)
        if not plot:
            continue
        yield Finding("E010", ERROR,
                      f"{plot}: {f['cells']} blocks at ({b[0]},{b[1]},{b[2]}) are held "
                      f"up by nothing (e.g. {f['example'].split('[')[0]})",
                      (b[0], b[1], b[2]), {"plot": plot, "cells": f["cells"]})


# --------------------------------------------------------------------- warnings

@check("W007", WARNING, "a one-block void runs through the build",
       "Buildings do not connect to each other properly")
def w007_void_band(ctx: Context):
    """A warning and not an error, deliberately: an attic crawlspace is the same shape, and
    the linter does not get to decide which one a builder meant. The number is what
    matters -- every other building in that town peaked at 3 to 14 cells on a level."""
    import numpy as np
    t = ctx.vol.tables()
    c = ctx.vol.codes
    # Canopy, not vegetation: a stripped log laid along the head of a wall is a top
    # plate, and counting it as a tree put a 45-cell "void" through a hall whose walls
    # and roof were touching all the way round. See observe._is_canopy.
    veg = np.array([observe._is_canopy(s) for s in ctx.vol.palette], bool)
    solid = ((t["lower"].astype(bool) | t["upper"].astype(bool)) & ~veg)[c]
    gap = solid[:, :-2, :] & ~solid[:, 1:-1, :] & solid[:, 2:, :]
    for p in ctx.plots:
        sl = (slice(p["x0"] - ctx.vol.x0, p["x1"] - ctx.vol.x0 + 1), slice(None),
              slice(p["z0"] - ctx.vol.z0, p["z1"] - ctx.vol.z0 + 1))
        lv = gap[sl].sum(axis=(0, 2))
        if not len(lv):
            continue
        k = int(lv.argmax())
        if int(lv[k]) >= 24:
            yield Finding("W007", WARNING,
                          f"{p['label']}: {int(lv[k])} cells of one-block void at "
                          f"y={k + ctx.vol.y0 + 1} -- a roof standing off its walls "
                          f"looks like this", (p["x0"], k + ctx.vol.y0 + 1, p["z0"]),
                          {"plot": p["label"], "cells": int(lv[k])})

@check("W006", WARNING, "ground cut off that nothing needs",
       "Buildings do not connect to each other properly", PLACE)
def w006_benign_regression(ctx: Context):
    """Reported by E009's check, not by this one. Registered so the catalogue lists it
    and so `lint(only={"W006"})` selects the pass that can produce it."""
    if False:
        yield None


@check("W005", WARNING, "planned site not served by circulation",
       "Buildings do not connect to each other properly", PLACE)
def w005_site_unserved(ctx: Context):
    """Asked of the *plan*, before anything is built on it. Once a structure exists
    W003 asks the same question of its plot; this is the version that can still be
    answered while the route can still be moved."""
    if not ctx.sites or not ctx.circulation:
        return
    for s in ctx.sites:
        # Where you would stand on this site's doorstep: the ground in the ring just
        # outside its footprint. Asking only whether a lane passes within a few blocks
        # horizontally is the mistake this project keeps making -- a lane five blocks
        # below a shelf is next to it on a map and nowhere near it on foot.
        ring = []
        for m in (1, 2):
            ring += [(x, s["z0"] - m) for x in range(s["x0"] - m, s["x1"] + m + 1)]
            ring += [(x, s["z1"] + m) for x in range(s["x0"] - m, s["x1"] + m + 1)]
            ring += [(s["x0"] - m, z) for z in range(s["z0"] - m, s["z1"] + m + 1)]
            ring += [(s["x1"] + m, z) for z in range(s["z0"] - m, s["z1"] + m + 1)]
        near = False
        for (x, z) in ring:
            g = ctx.nav.ground_stance(x, z)
            if g is not None and (x, z, g) in ctx.circulation:
                near = True
                break
        if not near:
            yield Finding("W005", WARNING,
                          f"nothing on the walkable network stands on the doorstep of "
                          f"planned site {s['id']} (x {s['x0']}..{s['x1']}, "
                          f"z {s['z0']}..{s['z1']})",
                          detail={"id": s["id"]})

@check("W001", WARNING, "door not on the circulation network",
       "Entrances placed where no one would arrive from -- the headline defect", PLACE)
def w001_door_off_circulation(ctx: Context):
    if not ctx.circulation:
        return
    for (x, y, z) in ctx.doors:
        s = ctx.door_stance(x, y, z)
        if s is None or (x, z, s) not in ctx.from_outdoors:
            continue        # already an E002; do not say it twice
        if (x, z, s) not in ctx.circulation:
            jumps = ctx.from_network.get((x, z, s))
            how = (f"{jumps} jumps off it" if jumps
                   else "not connected to it on foot at all")
            yield Finding("W001", WARNING,
                          f"door at ({x},{y},{z}) does not front the walkable "
                          f"network -- {how}", (x, y, z), {"jumps": jumps})


@check("W002", WARNING, "plot with an interior but no door",
       "Buildings cannot be entered; doorways lead nowhere")
def w002_no_door(ctx: Context):
    for p in ctx.plots:
        if not any(r.get("plot") == p["label"] for r in ctx.rooms):
            continue
        if any(p["x0"] - 1 <= x <= p["x1"] + 1 and p["z0"] - 1 <= z <= p["z1"] + 1
               for (x, _, z) in ctx.doors):
            continue
        yield Finding("W002", WARNING,
                      f"{p['label']} has an interior but no door or gate anywhere on it",
                      detail={"plot": p["label"]})


@check("W003", WARNING, "plot not on the circulation network",
       "Town not fully legible from the ground", PLACE)
def w003_plot_off_circulation(ctx: Context):
    if not ctx.circulation:
        return
    for p in ctx.plots:
        on = any(p["x0"] <= x <= p["x1"] and p["z0"] <= z <= p["z1"]
                 for (x, z, _) in ctx.circulation)
        if on:
            continue
        off = [j for (x, z, _), j in ctx.from_network.items()
               if p["x0"] <= x <= p["x1"] and p["z0"] <= z <= p["z1"]]
        jumps = min(off) if off else None
        how = (f"{jumps} jumps off the network" if jumps
               else "not connected to it on foot at all")
        yield Finding("W003", WARNING,
                      f"{p['label']} does not touch the walkable network -- {how}",
                      detail={"plot": p["label"], "jumps": jumps})


@check("W004", WARNING, "room below light 8",
       "Interiors that exist are thin and repetitive")
def w004_dark_room(ctx: Context):
    for r in ctx.rooms:
        if not r.get("plot"):
            continue
        cells = [(x, s // 2, z) for (x, z, s) in r["stances"]][:400]
        lv = sorted(observe.light(ctx.vol, cells).values())
        if not lv:
            continue
        med = lv[len(lv) // 2]
        if med < 8:
            b = r["bbox"]
            yield Finding("W004", WARNING,
                          f"{r['plot']}: room at ({b[0]},{b[1]},{b[2]}) has median "
                          f"light {med} -- mobs spawn below 8",
                          (b[0], b[1], b[2]), {"plot": r["plot"], "light": med})


def massing_occupancy(pending: dict) -> dict:
    """A massing program's occupancy, per column: {(x, z): (y_lo, y_hi)} over its
    non-air writes. This is the artefact one stage hands the next, and the thing W010
    measures the elaboration against."""
    cols: dict = {}
    for (x, y, z), b in pending.items():
        if b.split("[")[0].split(":")[-1] in ("air", "cave_air", "void_air"):
            continue
        lo, hi = cols.get((x, z), (y, y))
        cols[(x, z)] = (min(lo, y), max(hi, y))
    return cols


@check("W010", WARNING, "elaboration departed from the agreed massing",
       "The staged pipeline's contract: the mass that was judged is the mass that ships")
def w010_massing_conformance(ctx: Context):
    """Compares the built volume's occupancy against the massing stage's, and **reports
        the delta without ruling on it**. A builder that thickened a wall or added a
        buttress is not broken -- this project has caught a check deforming a design three
        times, and this one is not allowed to be the fourth. WARNING, never ERROR, and the
        tolerance is wide: >=85% of the massing's occupied columns still occupied, ridge
        within one course.

        A column "still occupied" means: solid within one course of the massing's **top**
        in that column. The band, not the whole column, because both false comforts live
        below it -- the ground the massing stood on still fills the bottom of a column the
        building has walked away from, and a wide roof plane slid three blocks sideways
        still crosses most of its old columns' midriffs. Against the top band a
        translation moves every top by the translation, while a thickened wall, a carved
        opening or a chimney leaves the band full. Canopy is excluded -- a tree over the
        plot is not a ridge.
        
    """
    if not ctx.massing:
        return
    import numpy as np
    vol = ctx.vol
    t = vol.tables()
    veg = np.array([observe._is_canopy(s) for s in vol.palette], bool)
    solid = ((t["lower"].astype(bool) | t["upper"].astype(bool)) & ~veg)[vol.codes]
    sy = vol.shape[1]

    held, missing = 0, []
    ridge = max(hi for _, hi in ctx.massing.values())
    built_ridge = None
    for (x, z), (lo, hi) in ctx.massing.items():
        lx, lz = x - vol.x0, z - vol.z0
        if not (0 <= lx < vol.shape[0] and 0 <= lz < vol.shape[2]):
            missing.append((x, z))
            continue
        a = min(max(hi - 1 - vol.y0, 0), sy - 1)
        b = min(max(hi + 1 - vol.y0, 0), sy - 1)
        col = solid[lx, a:b + 1, lz]
        if col.any():
            held += 1
            top = int(np.nonzero(solid[lx, :, lz])[0].max()) + vol.y0
            built_ridge = top if built_ridge is None else max(built_ridge, top)
        else:
            missing.append((x, z))
    cov = held / len(ctx.massing)
    delta = (built_ridge - ridge) if built_ridge is not None else None
    if cov < 0.85 or delta is None or abs(delta) > 1:
        ex = ", ".join(f"({x},{z})" for x, z in missing[:3])
        yield Finding(
            "W010", WARNING,
            f"the built mass holds {cov:.0%} of the massing's {len(ctx.massing)} "
            f"columns ({len(missing)} empty" + (f", e.g. {ex}" if ex else "") + ") "
            + (f"and its ridge is {delta:+d} of the massing's y={ridge}"
               if delta is not None else "and nothing reaches the massing's height")
            + " -- a report of the departure, not a verdict on it",
            (missing[0][0], ridge, missing[0][1]) if missing else None,
            {"coverage": round(cov, 3), "columns": len(ctx.massing),
             "missing": len(missing), "ridge_massing": ridge,
             "ridge_delta": delta})


# ------------------------------------------------------------------------ style

@check("S001", STYLE, "material placed that the settlement is not made of",
       "Palette families drift across passes")
def s001_palette(ctx: Context):
    """Measured against what this settlement **declared it is made of**, when it declared
    anything, and against the craft rule only when it did not.

    Still STYLE, never enforced. A builder that needs a sixth material for a reason is
    not broken, and a linter that ruled on that would be the aesthetic scorer this
    project has refused to build for six rounds."""
    if not ctx.placed:
        return
    from .buildlib import _families
    fams = _families(ctx.placed)
    if ctx.voice:
        # A planner writes prose into a palette value -- "jungle_log (sparingly, on the
        # shrine and the hall only)" -- so take the first word and keep it only if it
        # looks like a block. Otherwise the check's own message becomes the essay.
        import re
        declared = _families({m: 1 for m in
                              (re.split(r"[^a-z_]", str(v).strip().lower())[0]
                               for v in ctx.voice.values())
                              if m and len(m) > 2})
        extra = sorted(fams - declared)
        if extra:
            counts = {}
            for block, n in ctx.placed.items():
                fam = _families({block: 1})       # empty for air and water
                if fam and next(iter(fam)) in extra:
                    f = next(iter(fam))
                    counts[f] = counts.get(f, 0) + n
            worst = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
            yield Finding("S001", STYLE,
                          f"{len(extra)} materials placed that are not in the "
                          f"settlement's voice ({', '.join(sorted(declared))}): "
                          + ", ".join(f"{f} x{n}" for f, n in worst),
                          detail={"extra": extra, "declared": sorted(declared)})
        return
    if len(fams) > 6:
        yield Finding("S001", STYLE,
                      f"{len(fams)} material families placed "
                      f"(craft rule is 1-4, and 6 is generous)",
                      detail={"families": sorted(fams)})


@check("S002", STYLE, "built of the same material as the ground it stands on",
       "Palette camouflaged the town into the terrain at distance")
def s002_camouflage(ctx: Context):
    """Its planner chose a palette "colour-matched to the local strata" -- terracotta on a
    terracotta mesa -- which is good reasoning and a bad outcome: at settlement distance
    the town camouflaged into the ground and stopped reading as architecture. Nothing
    could see it but a person looking from far enough away.

    This is not a beauty judgement and cannot become one. It is a fact about material
    identity: the dominant family a pass placed, against the dominant family of the
    landscape it placed it in, sampled from the ground outside every plot."""
    if not ctx.placed:
        return
    from .buildlib import _families
    import numpy as np
    counts = {}
    for block, n in ctx.placed.items():
        fam = _families({block: 1})
        if fam:
            counts[next(iter(fam))] = counts.get(next(iter(fam)), 0) + n
    if not counts:
        return
    built = max(counts, key=lambda k: counts[k])

    vol = ctx.vol
    sx, _, sz = vol.shape
    wild = {}
    for x in range(vol.x0, vol.x0 + sx, 7):
        for z in range(vol.z0, vol.z0 + sz, 7):
            if ctx.plot_at(x, z):
                continue
            s2 = ctx.nav.ground_stance(x, z)
            if s2 is None:
                continue
            fam = _families({vol.name(x, s2 // 2 - 1, z): 1})
            if fam:
                wild[next(iter(fam))] = wild.get(next(iter(fam)), 0) + 1
    if not wild:
        return
    land = max(wild, key=lambda k: wild[k])
    if built == land:
        yield Finding("S002", STYLE,
                      f"the most-placed material is {built}, which is also what the "
                      f"ground here is made of -- at settlement distance this reads as "
                      f"landscape rather than as building",
                      detail={"built": built, "landscape": land})


#: Blocks that mean somebody uses this room. Not decoration -- equipment, a fire, a bed,
#: somewhere to put something down. Kept next to the check so the vocabulary a builder
#: is given (`prims.Primitives.FITTING_BLOCKS`) and the vocabulary the linter looks for
#: are read side by side and cannot drift apart silently.
FITTING_NAMES = {
    "campfire", "soul_campfire", "furnace", "blast_furnace", "smoker", "anvil",
    "chipped_anvil", "damaged_anvil", "smithing_table", "crafting_table", "barrel",
    "chest", "trapped_chest", "loom", "cartography_table", "fletching_table",
    "grindstone", "stonecutter", "brewing_stand", "cauldron", "water_cauldron",
    "lava_cauldron", "composter", "lectern", "bookshelf", "chiseled_bookshelf",
    "bell", "hay_block", "hopper", "beehive", "flower_pot", "candle", "lantern",
    "soul_lantern", "torch", "wall_torch", "shelf",
}


@check("W009", WARNING, "a room with nothing in it",
       "Interiors that exist are thin and repetitive")
def w009_empty_room(ctx: Context):
    """An interior a person can walk into and find bare."""
    for r in ctx.rooms:
        plot = r.get("plot")
        if not plot or len(r.get("stances", ())) < 12:
            continue
        b = r["bbox"]
        found = False
        for x in range(b[0], b[3] + 1):
            for z in range(b[2], b[5] + 1):
                for y in range(b[1], b[4] + 1):
                    if ctx.vol.name(x, y, z) in FITTING_NAMES:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            yield Finding("W009", WARNING,
                          f"{plot}: the room at ({b[0]},{b[1]},{b[2]}) is "
                          f"{len(r['stances'])} cells of empty floor -- nothing in it "
                          f"says what happens here",
                          (b[0], b[1], b[2]),
                          {"plot": plot, "cells": len(r["stances"])})


@check("S003", STYLE, "an opening glazed with something that does not fill it",
       "Buildings read as unfinished up close")
def s003_glazing(ctx: Context):
    """STYLE, and it stays STYLE: a lattice window is a real thing and a checker does not
    get to forbid one. It reports the count and the widest run so a person can look."""
    vol = ctx.vol
    panes = vol.find(lambda s: s.split("[")[0].endswith("_pane"))
    if not panes:
        return
    byrow: dict = {}
    for (x, y, z) in panes:
        byrow.setdefault(("x", y, z), set()).add(x)
        byrow.setdefault(("z", y, x), set()).add(z)
    wide = {}
    for (axis, a, b), cs in byrow.items():
        cs = sorted(cs)
        run = [cs[0]]
        for c in cs[1:] + [None]:
            if c is not None and c == run[-1] + 1:
                run.append(c)
                continue
            if len(run) > 1:
                pos = ((run[0], a, b) if axis == "x" else (b, a, run[0]))
                wide[pos] = max(wide.get(pos, 0), len(run))
            run = [c] if c is not None else run
    if not wide:
        return
    worst = sorted(wide.items(), key=lambda kv: -kv[1])[:4]
    cells = sum(1 for p in panes
                if any(abs(p[0] - q[0]) + abs(p[2] - q[2]) < 8 and p[1] == q[1]
                       for q, _ in worst))
    yield Finding("S003", STYLE,
                  f"{len(wide)} openings are glazed with panes across more than one "
                  f"block, so the glazing does not fill them -- widest "
                  f"{max(w for _, w in worst)} at "
                  + ", ".join(f"({q[0]},{q[1]},{q[2]})" for q, _ in worst[:2]),
                  worst[0][0], {"openings": len(wide), "panes": len(panes),
                                "near_worst": cells})


@check("W008", WARNING, "a door leaf with nothing beside it",
       "Entrances placed where no one would arrive from")
def w008_door_jamb(ctx: Context):
    """A warning, not an error. A gate leaf hung on a free-standing pier is a real building,
    and the difference between that and a mistake is a judgement. What is reported is the
    fact: this leaf has open air on one side at its own height and at head height."""
    vol = ctx.vol
    air = {"air", "cave_air", "water", "void_air"}

    def base(x, y, z):
        return (vol.name(x, y, z) or "").split("[")[0]

    for (x, y, z) in vol.find(lambda s: s.split("[")[0].endswith("_door")):
        st = vol.state(x, y, z) or ""
        if "half=lower" not in st or "facing=" not in st:
            continue
        face = st.split("facing=")[1].split(",")[0].split("]")[0]
        sides = ((1, 0), (-1, 0)) if face in ("north", "south") else ((0, 1), (0, -1))
        for dx, dz in sides:
            if (base(x + dx, y, z + dz) in air
                    and base(x + dx, y + 1, z + dz) in air):
                plot = ctx.plot_at(x, z)
                yield Finding("W008", WARNING,
                              f"{plot or 'unclaimed'}: the door at ({x},{y},{z}) has "
                              f"open air beside it -- it sits at the end of a wall run "
                              f"with no jamb on one side",
                              (x, y, z), {"plot": plot, "facing": face})
                break


# ----------------------------------------------------------------------- report

@dataclass
class Report:
    findings: list
    seconds: float = 0.0

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self):
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def within(self, plots: list) -> "Report":
        """Only the findings that belong to these plots."""
        labels = {p["label"] for p in plots}
        rects = [r for p in plots for r in plot_rects(p)]

        def mine(f):
            if f.detail.get("plot") in labels or f.detail.get("id") in labels:
                return True
            if f.pos is None:
                return False
            x, _, z = f.pos
            return any(a <= x <= c and b <= z <= d for a, b, c, d in rects)

        return Report([f for f in self.findings if mine(f)], self.seconds)

    def by_code(self) -> dict:
        out: dict[str, list] = {}
        for f in self.findings:
            out.setdefault(f.code, []).append(f)
        return dict(sorted(out.items(),
                           key=lambda kv: (_ORDER[kv[1][0].severity], kv[0])))

    def summary(self, examples: int = 3) -> str:
        """Grouped, capped, one line per finding. A linter nobody reads is not a linter,
        and 447 backwards stairs printed in full is a linter nobody reads."""
        titles = {c.code: c.title for c in CHECKS}
        lines = []
        for code, fs in self.by_code().items():
            sev = fs[0].severity.upper()
            lines.append(f"{sev:7s} {code}  {titles.get(code, '')}  "
                         f"({len(fs)} finding{'s' if len(fs) != 1 else ''})")
            for f in fs[:examples]:
                lines.append(f"          {f.message}")
            if len(fs) > examples:
                lines.append(f"          ... and {len(fs) - examples} more")
        if not lines:
            lines.append("no findings")
        lines.append("")
        lines.append(f"{len(self.errors)} errors, {len(self.warnings)} warnings, "
                     f"{len(self.findings) - len(self.errors) - len(self.warnings)} style"
                     f"  ({self.seconds:.1f}s)")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"ok": self.ok, "seconds": round(self.seconds, 2),
                "counts": {c: len(f) for c, f in self.by_code().items()},
                "findings": [{"code": f.code, "severity": f.severity,
                              "message": f.message, "pos": list(f.pos) if f.pos else None,
                              **({"detail": f.detail} if f.detail else {})}
                             for f in self.findings]}


def lint(ctx: Context, only: set | None = None, family: str | None = None) -> Report:
    import time
    t0 = time.perf_counter()
    findings = []
    for c in CHECKS:
        if only and c.code not in only:
            continue
        if family and c.family != family:
            continue
        findings.extend(c.fn(ctx))
    findings.sort(key=lambda f: (_ORDER[f.severity], f.code))
    return Report(findings, time.perf_counter() - t0)


#: What to do about each code, in the words that worked. Wave 1 went from 73 findings to
#: 1. Keeping them here makes the loop a mechanism instead of me remembering what I said
#: last time.
FIXES = {
    "E001": "Block ids are validated against the server's registry before your program "
            "runs, and one bad id rejects the whole file. This is Minecraft 1.21.11 -- "
            "`chain` is now `iron_chain`.",
    "E002": "A doorway nothing can walk to on foot -- walking only, no jumping, the "
            "standard the lanes outside are held to. A door whose sill sits a block "
            "above the ground you arrive on is a door nobody can enter. "
            "approach(label) lays the path: from the doorway to the lane, following "
            "the ground, a slab where it rises half a block and a tread where it "
            "rises a whole one. Call it after the door is in.",
    "E014": "A type is a **form**; the palette is the settlement's. Every material "
            "comes from `part['voice']` -- `wall`, `footing`, `frame`, `roof`, `trim`, "
            "`floor` -- and every shape of one from `b.block(fam, kind)`, where kind is "
            "`full`, `stairs`, `slab`, `wall`, `post`, `bare`, `accent`, `fine`, or "
            "`b.joinery(part['voice'], 'door')` for a door, fence or trapdoor. A block "
            "name in a type file welds that type to one voice: the same house cannot "
            "then be built in ochre stone under green tile without writing it again.",
    "E015": "A type is a **form**; the roof's silhouette is the voice's. `roof()` and "
            "`building()` are handed the voice's `profile`, `ends`, `eave` and `tiers` "
            "from `part['roof']` by the library, whatever the type passes, and where "
            "the voice names none the library's own default stands. A pitch, a pair of "
            "ends, an eave or a tier count written into a type file is a building that "
            "has decided what every voice's roof looks like -- pass `style` and `axis` "
            "if you must, and nothing else about the roof's shape.",
    "E013": "A type builds from `part['floor_y']` up and never below it. The ground "
            "under the part -- the bed sounded, a deck on piles over water, a platform "
            "cut and filled where there is relief, a plinth where there is not, and "
            "the way in from the lane -- is laid by `site()` before your `build()` is "
            "called, and `part` comes to you with the floor level, the footprint and "
            "the door cell already on it. Hand-written ground work is what stopped "
            "three checked types standing up on ground their checkers had not seen.",
    "E012": "Take the `try` out and let it raise. If a call fails, the harness hands "
            "you the traceback and you fix the line; if you catch it, the building "
            "quietly loses whatever that call was going to place and every check "
            "downstream reads the result as the building you meant to make.",
    "E003": "Enclosed standing space with no walk path in from any entrance. Either open "
            "it to the rest of the building or fill it; a room nobody can enter is not a "
            "room. Look for floors laid a block above their own doorway sill -- "
            "floor_from_threshold(label) gives you the level the doorstep is at. "
            "seal_voids(x0, z0, x1, z1, block) closes the pockets it can prove are "
            "enclosed, unreachable and inside your footprint, and reports the rest.",
    "E011": "A room you can only get into by jumping. The floor is laid above the "
            "level your feet are at when you stand in your own doorway, or the only "
            "route in is over a block. floor_from_threshold(label) gives you the level "
            "the doorstep is at -- put the ground floor there. check_walkable(label) "
            "answers this over your own unflushed blocks, before you finish, and "
            "reports the fraction per room; call it and fix what it names. A step up "
            "to a platform is fine and is only ever a warning; a floor nobody can step "
            "onto is not.",
    "W011": "Part of a room's floor needs a jump. This is a report, not a verdict -- a "
            "raised sleeping platform, a dais or a mezzanine is architecture, and "
            "nothing here says otherwise. If the fraction is low and you did not mean "
            "a platform, the usual cause is a floor laid a block above its own doorway "
            "sill, or a fitting() placed across the only route; check_walkable(label) "
            "will tell you which room.",
    "E004": "A stair's `facing` names the side its raised quarter is on, and on any "
            "slope, flight of steps or run of treads it must point *up*-slope. 'ahead' "
            "is the surface one block the way it faces, 'behind' one block the other "
            "way; ahead lower than behind is the failure. Do not place treads by hand: "
            "steps(cells, mat) queues a flight and decides every facing at the end from "
            "the finished ground on both sides, demotes a tread it cannot justify to a "
            "slab, and refuses a rising tread the flight does not continue past. A tall "
            "stack at the *eave* of a roof also trips this -- the stairs beside it have "
            "a wall behind them and a slope ahead.",
    # The sentence that used to end E004. Roofs pierced by chimney and flue columns.
    # Advice text that pushes a builder toward a worse build is the failure mode this
    # project has now named four times, and this is the fourth.
    "E005": "Connective blocks -- fences, walls, panes -- that never joined to their "
            "neighbours. The library re-places them with block updates on, so if these "
            "are reported the states were forced by hand.",
    "E006": "Two plots overlap. Reserve before you build and move if reserve() is False.",
    "E007": "Lane cells of the circulation network can no longer be stood on, or the "
            "network is in pieces. Your grading, plinth, skirt or eaves went outside "
            "your plot onto public ground. Query nearest_lane(x, z) before you fill or "
            "cut anything beyond your own footprint.",
    "E008": "The threshold reserved for a structure -- or the doorway off it -- has been "
            "built over or raised. That ground belongs to the circulation pass: put the "
            "floor at the level it gives you and leave the lane cell alone.",
    "E009": "This pass cut ground off that was walkable before it ran, and it is ground "
            "the town needs: lane, a reserved threshold, or somebody else's plot.",
    "E010": "A piece of your build is held up by nothing. Call check_attached() before "
            "you finish -- it lists every piece attached to nothing, and eaves and "
            "jetties are not listed because they reach the ground through the build. "
            "A roof one block too high looks exactly like this.",
    "W001": "A door reachable only by jumping. Move it to somewhere on the network and "
            "call check_door(x, y, z) again; it answers with your own unflushed blocks.",
    "W002": "A structure with an interior and no door anywhere on its plot.",
    "W003": "A plot the walkable network does not reach.",
    "W004": "A room below light level 8, measured by propagation and not by counting "
            "torches. Mobs spawn there.",
    "W005": "A planned site no route reaches.",
    "W006": "Ground outside every plot that can no longer be walked to. The network is "
            "intact, so this is a judgement call rather than a break.",
    "W007": "One level, one block of air, solid above and below, right across a "
            "footprint -- what a roof standing off its walls looks like. An attic "
            "crawlspace has the same shape, so check which one you built.",
    "W010": "Your elaboration departs from the massing that was judged and selected. "
            "If the departure is deliberate -- a buttress, a thicker wall, a porch -- "
            "leave it; this is a report, not a verdict. If it is not deliberate, put "
            "the mass back where the massing put it.",
    "S001": "Material families beyond the craft rule. Variants (mossy, cracked, "
            "stripped, polished) and shapes (stairs, slabs, walls) of one material count "
            "as one, so the number is genuinely different materials.",
}


def findings_brief(report: "Report", title: str, examples: int = 6,
                   header: str = "") -> str:
    """A lint report as something you can hand straight back to the pass that caused it."""
    titles = {c.code: c.title for c in CHECKS}
    out = [f"# {title}", ""]
    if header:
        out += [header, ""]
    if not report.findings:
        out += ["The linter reports nothing. Nothing to fix."]
        return "\n".join(out)
    out += [f"A deterministic linter read the result out of the world -- pathfinding and "
            f"block state, not opinion. It reports {len(report.errors)} errors and "
            f"{len(report.warnings)} warnings.", ""]
    for code, fs in report.by_code().items():
        out.append(f"## {code} -- {titles.get(code, '')} ({len(fs)} "
                   f"finding{'s' if len(fs) != 1 else ''})")
        out.append("")
        if code in FIXES:
            out += [FIXES[code], ""]
        for f in fs[:examples]:
            out.append(f"    {f.message}")
        if len(fs) > examples:
            out.append(f"    ... and {len(fs) - examples} more")
        out.append("")
    return "\n".join(out)


def snapshot(nav: observe.Nav, network=None, seeds: list | None = None) -> dict:
    """What was walkable before a pass ran, to be handed to the next Context as
        `before=`. The flood costs about 0.2 s on a 192x192 settlement, which is why this
        can run after **every** pass rather than once at the end.

        Seeded from the lane cells the circulation pass declared, not from the derived
        network: the seeds have to be the same before and after or the comparison measures
        the seeds moving rather than the town changing. What is *reachable* from them is
        still derived, which is the part that matters.

        Takes a Nav rather than a Context on purpose -- a pass wants this before it runs,
        and building a whole Context (rooms, light, shelter) to answer it would cost several
        seconds a pass for nothing.
        
    """
    if seeds is None:
        seeds = lane_stances(nav, network)
    return {"seeds": seeds, "reachable": set(nav.flood(seeds, max_jumps=0))}


#: Where a **material** goes, as opposed to where a block goes. `registry.BLOCK_ARGS` is
#: the block table and is validated against the registry; this is the family table, and
#: between them they are every argument of the build API that decides what something is
#: made of. validating by position is what keeps it precise. A rule that flagged every
#: material-shaped literal anywhere in a file also flags `"bamboo"` in a list of plants
#: a pass may clear and `"oak_log["` in a test for whether a floating piece is a tree,
#: and neither of those is a palette.
MATERIAL_ARGS = {
    "roof": ({5}, {"mat"}),
    "roof_cone": ({4}, {"mat"}),
    "dormer": ({3}, {"mat"}),
    "steps": ({1}, {"mat"}),
    "flight": (set(), {"mat"}),
    "fitting": (set(), {"mat", "block"}),
    "building": (set(), {"mat"}),
    "dais": ({5}, {"mat"}),
    "doorway": ({4}, {"mat", "leaf", "lintel", "threshold"}),
    "seal_voids": ({4}, {"block"}),
    "plinth": ({5}, {"block", "mat"}),
    "site": (set(), {"mat"}),
    "plateau": (set(), {"mat"}),
    "approach": (set(), {"mat"}),
    "emit": (set(), {"mat", "kerb", "rail"}),
}


#: Where a **silhouette** goes. Voice contract, B2, on E014's principle: judged by
#: position, so it is precise and has no false alarms. `roof()` takes `style` and
#: `pitch` by position or by name and `profile`, `ends`, `eave` and `tiers` by name;
#: `building()` takes its roof as one spec, a style name or a dict of the same keys.
#: `style` is deliberately **not** here: gable, hip, shed and flat are what a roof *is*
#: over a given mass -- a lean-to is a shed by construction -- and the voice's `ends`
#: and `profile`, handed to `roof()` by `TypeBuilder`, override a style's own ends and
#: slope wherever the voice has them. What a type may not write down is the slope, the
#: ends, the eave and the tiers, which are the four things a voice's `roof` names.
SILHOUETTE_KEYS = ("pitch", "profile", "ends", "eave", "tiers")
SILHOUETTE_ARGS = {"roof": ({8: "pitch"}, set(SILHOUETTE_KEYS)),
                   "building": ({6: "roof"}, {"roof"})}


def _silhouette_literal(node) -> bool:
    """Is this AST node a literal silhouette value -- a string, a number, or a tuple or
    list of those (a pitch, a pair of ends, a profile)? `None` is not: `ends=None` is
    `roof()`'s own "the style's ends", which is the one thing a type may say."""
    import ast
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(_silhouette_literal(e) for e in node.elts)
    return False


def _material_positions() -> dict:
    """Every argument of the build API that names a block or a material family."""
    out = {k: (set(p), set(k2)) for k, (p, k2) in MATERIAL_ARGS.items()}
    for name, (pos, kws) in registry.BLOCK_ARGS.items():
        p, k = out.setdefault(name, (set(), set()))
        p |= set(pos)
        k |= set(kws)
    return out


def palette_literal(text: str) -> str | None:
    """The material family a string literal names, or None.

        The one question E014 asks, and it is answered by the library's own resolver rather
        than by a list kept here: `prims.family()` decides what material a block belongs to,
        and a literal is a palette literal when it *is* a family, is a block of one, or is a
        block named after one -- `spruce`, `stripped_spruce_log[axis=y]`, `dark_oak_door`,
        `bamboo_fence`, `quartz_pillar`.

        It deliberately does not catch a block that is not made of a material family: `air`,
        `water`, `lantern`, `torch`, `barrel`, `bookshelf`, `terracotta`. Those are not
        palette -- they are what a room is furnished and cleared with, they read the same in
        every voice, and a check that flagged them would be a check about vocabulary rather
        than about the palette. What A1 is undoing is a type welded to a *voice*, and a
        voice is six material families.
        
    """
    from .prims import MATERIALS, family
    s = str(text).split("[")[0].split(":")[-1]
    if not s:
        return None
    fam = family(s)
    if fam is not None:
        return fam
    for f in MATERIALS:
        if s == f or s.startswith(f + "_") or s.endswith("_" + f):
            return f
    return None


def preflight(source: str, *, allow_try: bool = False, forbid=None,
              palette: bool = False) -> Report:
    """Validate a pass's program before it writes anything.

    **E012 -- no `try`.** A program that catches its own exceptions cannot be checked
    and cannot be trusted to have built what it says it built. Every one of the sixteen
    repair-round programs wrapped its calls in blanket `except Exception` -- 46 clauses
    at draft 0 and 75 at draft 2 -- and chapel/c0 went from 2,995 blocks to 772 with
    seventeen swallowed failures and one lint error against it. Nothing anywhere
    reported that most of the chapel had silently failed to be placed. The harness
    raises on real errors and hands the builder a scrubbed traceback to fix, which is
    the feedback it needs; there is no legitimate use for `except` in a build program.

    This is preflight, not the check suite: E012 is not in `CHECKS`, does not appear in
    `catalogue()`, and judges the source rather than the world. The frozen suite is
    still frozen."""
    import ast
    import time
    t0 = time.perf_counter()
    findings = [Finding("E001", ERROR, f"{lit}: {errs[0]}", detail={"state": lit})
                for lit, errs in registry.check_source(source).items()]
    # A program that does not parse is already an E001 from `check_source`, which
    # compiles it to find the block literals; saying so twice helps nobody. Parsed
    # whichever checks are on: `allow_try` used to skip the parse itself, which silently
    # took E013 and E014 with it. It turns off *E012* and nothing else.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    for node in (ast.walk(tree) if (tree is not None and not allow_try) else ()):
        if isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            findings.append(Finding(
                "E012", ERROR,
                f"line {node.lineno}: this program catches its own exceptions. A "
                f"`try` block hides a call that did not place what you meant it to, "
                f"and nothing downstream can tell the difference between a building "
                f"you built and a building that failed quietly",
                detail={"line": node.lineno}))
    for node in (ast.walk(tree) if (tree is not None and forbid) else ()):
        name = (node.attr if isinstance(node, ast.Attribute) else
                node.id if isinstance(node, ast.Name) else None)
        if name not in (forbid or ()):
            continue
        why = forbid[name] if isinstance(forbid, dict) else \
            "the library sites the part before your build() is called"
        findings.append(Finding(
            "E013", ERROR,
            f"line {node.lineno}: this is a type, and it calls {name}(). A type does "
            f"not touch the ground -- {why}. Build from part['floor_y'] up",
            detail={"line": node.lineno, "call": name}))
    # **E014. A type is a *form* and the palette is the settlement's, handed to it as
    # `part['voice']`.
    seen_lines: set = set()
    where = _material_positions() if palette else {}
    for node in (ast.walk(tree) if (tree is not None and palette) else ()):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        spec = where.get(name)
        if not spec:
            continue
        positions, keywords = spec
        args = [a for i, a in enumerate(node.args) if i in positions]
        args += [kw.value for kw in node.keywords if kw.arg in keywords]
        # A palette handed over as a dict of roles is the same decision written out:
        # `mat={"wall": "cobblestone", ...}` is a type declaring a voice.
        flat = []
        for a in args:
            if isinstance(a, ast.Dict):
                flat += list(a.values)
            elif isinstance(a, (ast.List, ast.Tuple, ast.Set)):
                flat += list(a.elts)
            else:
                flat.append(a)
        for a in flat:
            if not isinstance(a, ast.Constant) or not isinstance(a.value, str):
                continue
            fam = palette_literal(a.value)
            if fam is None or (a.lineno, a.value) in seen_lines:
                continue
            seen_lines.add((a.lineno, a.value))
            findings.append(Finding(
                "E014", ERROR,
                f"line {a.lineno}: this is a type, and it names the material "
                f"{a.value!r} ({fam}) where {name}() takes what a thing is made of. A "
                f"type is a form and the palette is the voice's: take every material "
                f"from part['voice'] -- wall, footing, frame, roof, trim, floor -- and "
                f"every shape of one from b.block(part['voice'][role], 'stairs'|'slab'|"
                f"'wall'|'post'|'bare'|'accent'|'fine') or b.joinery(part['voice'], "
                f"'door'|'fence'|'trapdoor')",
                detail={"line": a.lineno, "literal": a.value, "family": fam,
                        "call": name}))
    # **E015 -- a type does not name a silhouette.** Voice contract, B2, and E014's
    # mechanism a second time: a fact about the source, judged before it runs. The
    # roof's slope, ends, eave and tiers are the voice's, handed to `roof()` and
    # `building()` by `TypeBuilder` from `part['roof']`.
    for node in (ast.walk(tree) if (tree is not None and palette) else ()):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        spec = SILHOUETTE_ARGS.get(name)
        if not spec:
            continue
        positions, keywords = spec
        pairs = [(positions[i], a) for i, a in enumerate(node.args) if i in positions]
        pairs += [(kw.arg, kw.value) for kw in node.keywords if kw.arg in keywords]
        flat = []
        for key, a in pairs:
            if key == "roof":
                # `building(roof=...)`: a style name is a literal silhouette; a dict is
                # opened and its silhouette keys read as `roof()`'s own.
                if isinstance(a, ast.Dict):
                    for k_, v_ in zip(a.keys, a.values):
                        if isinstance(k_, ast.Constant) and k_.value in SILHOUETTE_KEYS:
                            flat.append((k_.value, v_))
                continue
            flat.append((key, a))
        for key, a in flat:
            if not _silhouette_literal(a) or (a.lineno, key) in seen_lines:
                continue
            seen_lines.add((a.lineno, key))
            lit = ast.unparse(a)
            findings.append(Finding(
                "E015", ERROR,
                f"line {a.lineno}: this is a type, and it names the roof's {key} "
                f"({lit}) where {name}() takes the silhouette. A type composes the "
                f"massing; the voice sets the silhouette -- roof() and building() are "
                f"handed the voice's profile, ends, eave and tiers from part['roof'] "
                f"for you, and where the voice is silent the library's own default "
                f"stands. Leave {key} out",
                detail={"line": a.lineno, "key": key, "literal": lit, "call": name}))
    return Report(findings, time.perf_counter() - t0)


def catalogue() -> str:
    """The check list, and the taxonomy row each one exists for."""
    rows = ["code  severity  family  check                                  catches"]
    for c in sorted(CHECKS, key=lambda c: (_ORDER[c.severity], c.code)):
        rows.append(f"{c.code}  {c.severity:8s}  {c.family:6s}  {c.title:38s} "
                    f"{c.catches}")
    return "\n".join(rows)
