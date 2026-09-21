"""**Six questions asked of the world that stands, not of the record that describes it.**

The design round's third contract. What established a feature before this was one of
three things, and none of them is use:

  * a **label** -- `FUNCTION = "market"` on the type file, which `intent._function_measure`
    reads and reports as a measurement of function;
  * an **occupied bounding box** -- `construction._verify_rect` asks whether half the
    columns of the rectangle the type claimed have something standing on them, which is
    the same answer for a stall, a bench, a bell and a pile of rubble;
    **the composition round closes that where the library lays an identifiable block**:
    see `FEATURE_BLOCKS`. A hearth is a fire, a forge is a furnace, and a stall carries a
    counter or its goods. Where this build lays no block that identifies a feature the
    record says so (`identity: "unsupported"`) instead of inventing a proxy;
  * a **generator assertion** -- `emitted.features = {"stalls": True}`, the type's own
    account of what it did, believed where nothing could check it.

And all three were read off the part's **own emission**, before the next part was built.
A courtyard a later wall filled in, a forge a neighbour's terrace buried and a door the
finishing pass paved over were all recorded as delivered, because nobody asked again
afterwards.

So: six predicates, on the **assembled** volume, after all construction.

    check(world, part, want, *, plan=None, registry=None)
        -> {"holds": bool | None,
            "method": "observed" | "inferred" | "declared" | "unsupported",
            "evidence": {...}, "subjects": [...], "why": str}

**Method and coverage are two different facts and this module keeps them apart.**
`method` says how the answer was got: `observed` off the assembled blocks, `inferred`
from geometry that implies it, `declared` where the only thing behind it is the type's
own word, `unsupported` where nothing can decide. `holds: None` is the honest answer
wherever the question cannot be decided, **and it never counts as satisfied anywhere**;
a caller that reads `holds` as truthiness reads None as False, which is the safe way
round, and a caller that wants to report coverage reads `method`.

**No new pathfinder.** Every reachability answer here comes from `observe.Nav` through
`lint.Context`, which is the one walk model this project has (`observe.WALK_MODEL`).
`World.of` assembles that context once per built world and the six predicates are then
essentially free, which is what lets them run on every part rather than on a sample.
"""
from __future__ import annotations

import json
import os

#: The six wants, in the order the contract names them.
WANTS = ("entrance_connected", "passage_connected", "equipment_reachable",
         "circulation_clear", "court_accessible", "range_relation")

#: The observation methods, weakest last. Nothing here ever returns `holds: True` with
#: `method: "declared"`: a declaration is what this module exists to stop counting.
METHODS = ("observed", "inferred", "declared", "unsupported")

#: Features that are **equipment**: a thing the function needs in order to be that
#: function, which a person has to be able to walk up to. Read off the names
#: `construction.CLAIMED_FEATURES` already carries, less the ones that are openings in
#: the fabric (a chimney, a jetty) rather than something you use.
EQUIPMENT = ("stalls", "forge", "hearth", "counter", "altar", "dais", "benches",
             "bell", "shopfront", "main_hall", "screen")

#: Features that are **open ground** -- a court, a yard, an aisle. The same list
#: `construction.OPEN_FEATURES` verifies by openness rather than by mass.
COURTS = ("courtyard", "court", "yard")

#: What share of a room's floor has to be walkable from that part's own doorways for the
#: part to be connected inside. `lint.w011_partly_walkable` warns below this and it is
#: the same bar, on purpose: two instruments disagreeing about what "you can get around
#: in there" means is how a place passes one and fails the other.
PASSAGE_FRACTION = 0.5

#: How far, in columns, a stance may be from a piece of equipment and still count as
#: standing at it. One column is the cell you stand in to use it; two allows a counter
#: with a stall behind it.
REACH = 2

#: How many of a court's four sides must carry the part's own mass for the court to be
#: *enclosed by ranges* rather than merely surrounded by whatever was there. Three of
#: four is the same bar `construction.measure` holds a storey's wall ring to.
COURT_SIDES = 3

#: How far out from a part's footprint the ground has to be walkable and on the main
#: outdoor component for its circulation to be clear. The 1- and 2-ring, which is the
#: question `lint.w005_site_unserved` asks of a planned site, asked here of a built one.
APRON = 2

#: **What makes a feature that feature and not an occupied rectangle.** The composition
#: round's second evidence connection, and the defect is the one this module's own
#: header admitted: `_stands_in` counts any non-air column within `FEATURE_COURSES` of
#: the floor, so a pile of rubble in a hearth rectangle answered exactly as a fire did.
#: A word here is a **substring of a block id**, matched against the block the assembled
#: world carries; the ids are the ones the library actually lays, read off
#: `prims.Builder.FITTING_BLOCKS` and the two market types, not guessed: * `hearth` --
#: `FITTING_BLOCKS["hearth"]` lays `campfire[lit=true]` on a stone base
#: (`prims.fitting`), and `types/court_large.py` lays the same through `furnish`. A lit
#: furnace or smoker is the other fire this library knows. `iron_bars` is in the
#: fitting's vocabulary as a grate and is **not** here: a grate is not a fire. * `forge`
#: -- `FITTING_BLOCKS["forge"]` is a furnace, `["anvil"]` an anvil or a smithing table;
#: `types/workshop.py` reports whichever of them it laid. * `stalls` --
#: `types/market.py._stall` and `types/square.py._stall` both lay a counter (the voice's
#: footing under a **slab** top) and the booth's goods (a `barrel` crate, a
#: `crafting_table`, a fence-and-pressure-plate table, or a `lantern` hung under the
#: hood). Measured on the two production types: a 30x18 market carries 20 slabs, 10
#: barrels and 10 lanterns in its stalls rectangle, and a 40x40 square's four booths
#: carry a slab or a table each. Plain voice masonry -- rubble -- carries none of them.
#: A feature **absent** from this table has no identifiable block in this build. Its
#: mass is still measured and its identity is reported `unsupported`, which is the
#: honest answer and not a proxy: nothing in `types/worship.py` distinguishes the
#: cobblestone of a dais from the cobblestone of a plinth, and pretending otherwise
#: would be the same mistake one level down.
FEATURE_BLOCKS = {
    "hearth": ("campfire", "furnace", "smoker", "fire", "lava"),
    "forge": ("furnace", "blast_furnace", "anvil", "smithing_table"),
    "stalls": ("slab", "barrel", "chest", "crafting_table", "lantern",
               "pressure_plate", "bookshelf", "cauldron"),
}

#: How far above a court's floor the sky is looked for. Six storeys: a court with that
#: much clear air over it is open to anything this build puts above a court, and
#: scanning to the top of the volume costs more and finds nothing. See
#: `court_accessible`.
SKY_COURSES = 24


# ------------------------------------------------------------------ the world

class World:
    """The assembled world, read once, with what was standing when it was read.

        Not a `lint.Context` subclass: a context is the reachability state and this is that
        plus the **provenance** the contract asks for -- which parts stood, what the built
        volume's digest was, and which state directory the answer is about. A predicate that
        cannot say which world it looked at is a predicate that can be quoted about another.
        
    """

    def __init__(self, ctx, rows: list, digest: str | None = None,
                 state: str | None = None, network=None, registry=None):
        self.ctx = ctx
        self.rows = list(rows or [])
        self.by_name = {str(r.get("part")): r for r in self.rows if r.get("part")}
        #: `{part: True}` for every leaf that stood. The predicates record this, so an
        #: answer about a court says which neighbours were up when it was read.
        self.standing = {str(r.get("part")): bool(r.get("stood",
                                                        r.get("status") == "built"))
                         for r in self.rows if r.get("part")}
        self.digest = digest
        self.state = state
        self.network = network
        self.registry = list(registry or (ctx.plots if ctx is not None else []))
        self._walk = None

    # --- construction ---------------------------------------------------------
    @classmethod
    def of(cls, state: str, *, volume=None, base=None, registry=None, network=None,
           region=None, parts=None) -> "World":
        """The built world in a round's state directory, assembled.

                Reads `world_built.npz` (the world **after every wave**), the plot registry with
                its floors, the lane network and `parts.json`. Everything is overridable so a
                probe or a fixture can be checked with the same predicates as a town.
                
        """
        from . import circulate, lint, offline, settlement
        vol = volume if volume is not None else _load(offline, state, "world_built.npz")
        if vol is None:
            raise FileNotFoundError(f"{state}: no world_built.npz to read; the "
                                    f"predicates are asked of the assembled world and "
                                    f"there is not one here yet")
        if base is None:
            base = _load(offline, state, "world.npz")
        if registry is None:
            registry = settlement.registry_with_floors(state)
        if network is None:
            p = os.path.join(state, "network.json")
            network = circulate.Network.load(p) if os.path.exists(p) else None
        if region is None:
            region = _region(state)
        if parts is None:
            parts = _rows(os.path.join(state, "parts.json"))
        ctx = lint.Context.build(vol, plots=registry, network=network, region=region,
                                 base=base)
        from . import deps
        return cls(ctx, parts, digest=deps.content_print(
            os.path.join(state, "world_built.npz")), state=state, network=network,
            registry=registry)

    @classmethod
    def assembled(cls, builder) -> "object":
        """The builder's ground with its pending blocks laid in: the world it made.

                Separate from `of_builder` so a caller can take this, change it -- remove a
                stall, wall off a court -- and hand the changed world back. Asking a predicate
                about a world somebody has damaged on purpose is how the predicate is shown to
                be answering the world and not the record.
                
        """
        vol = getattr(builder, "_vol", None)
        if vol is None:
            raise ValueError("no volume behind this builder")
        built = vol.sub(vol.x0, vol.z0, vol.shape[0], vol.shape[2])
        built.overlay(dict(getattr(builder, "_pending", None) or {}))
        return built

    @classmethod
    def of_builder(cls, builder, part: dict, *, volume=None) -> "World":
        """One probe-built type, as a world of one part.

                The adoption gate's case (`growth.gate`): a type authored this run has no round
                and no `world_built.npz`, and the question "can a person walk in and use it" is
                exactly as answerable on the flat probe volume as on a town. `volume` is the
                **assembled** world where a caller has one of its own (see `assembled`); with
                none, the builder's ground plus its pending blocks is it.
                
        """
        from . import lint
        base = getattr(builder, "_vol", None)
        if base is None:
            raise ValueError("no volume behind this builder")
        built = volume if volume is not None else cls.assembled(builder)
        vol = base
        x0, z0, x1, z1 = _rect(part)
        ctx = lint.Context.build(built, plots=[{"label": part.get("name") or "probe",
                                                "x0": x0, "z0": z0, "x1": x1, "z1": z1,
                                                "kind": part.get("kind", "plot"),
                                                **({"y0": int(part["floor_y"])}
                                                   if part.get("floor_y") is not None
                                                   else {})}],
                                 region=(vol.x0, vol.z0, vol.x0 + vol.shape[0] - 1,
                                         vol.z0 + vol.shape[2] - 1),
                                 base=base)
        row = {"part": part.get("name") or "probe", "status": "built", "stood": True,
               "kind": part.get("kind", "plot"), "type": part.get("type"),
               "x0": x0, "z0": z0, "x1": x1, "z1": z1,
               "floor_y": part.get("floor_y"),
               "emitted": part.get("emitted") or {}}
        return cls(ctx, [row], digest=None, state=None)

    # --- shared reads ---------------------------------------------------------
    def walk(self, margin: int = 4) -> dict:
        """`{plot: [{bbox, cells, walkable, fraction, enclosure, doors}, ...]}`.

                `lint.Context.interior_walk`'s measurement, **without its enclosure filter**.
                That method drops any room under `Context.ENCLOSED` (0.85) because its three
                callers are asking "is this an airtight interior somebody has failed to walk
                into"; the question here is "can somebody who came through the door get around
                in there", which a cottage with wide openings has as much as a sealed hall does.
                On the retained farm every cottage room measures 0.38-0.74 enclosure, so with
                the filter this predicate answered `unsupported` for all sixteen of them. Same
                seeds, same flood, same bounds -- the `enclosure` each room measured is on every
                row, so a reader can still tell a hall from a lean-to.
                
        """
        if self._walk is not None:
            return self._walk
        ctx = self.ctx
        boxes, seeds, leaves = {}, {}, {}
        for p in ctx.plots:
            boxes[p["label"]] = (min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
                                 max(p["x0"], p["x1"]), max(p["z0"], p["z1"]))
        from . import lint
        for (x, y, z) in ctx.doors:
            for p in ctx.plots:
                if lint.plot_covers(p, x, z, margin=1):
                    leaves[p["label"]] = leaves.get(p["label"], 0) + 1
                    s = ctx.door_stance(x, y, z)
                    if s is not None:
                        seeds.setdefault(p["label"], []).append((x, z, s))
        reach = {}
        for lab, (a, b, c, d) in boxes.items():
            if lab in seeds:
                reach[lab] = set(ctx.nav.flood(
                    seeds[lab], max_jumps=0,
                    bounds=(a - margin, b - margin, c + margin, d + margin)))
        got: dict = {}
        for r in ctx.rooms:
            lab = r.get("plot")
            if not lab:
                continue
            floor = [c for c in (r.get("floor") or ())
                     if c not in set(seeds.get(lab) or ())]
            walkable = sum(1 for c in floor if c in reach.get(lab, ()))
            got.setdefault(lab, []).append(
                {"bbox": r.get("bbox"), "cells": len(floor),
                 "walkable": walkable,
                 "fraction": round(walkable / float(len(floor)), 3) if floor else None,
                 "enclosure": r.get("enclosure"), "made": r.get("made"),
                 "doors": leaves.get(lab, 0)})
        self._walk = got
        return got

    def provenance(self) -> dict:
        """What was standing when this world was read. On every answer."""
        stood = sorted(n for n, ok in self.standing.items() if ok)
        return {"built_digest": self.digest, "state": self.state,
                "standing": len(stood), "parts": len(self.standing),
                "standing_parts": stood[:40],
                "read": "the assembled volume after every wave"}


def _load(offline, state: str, name: str):
    p = os.path.join(state, name)
    return offline.load_volume(p) if os.path.exists(p) else None


def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    doc = json.load(open(path))
    return [r for w in (doc.get("waves") or []) for r in (w.get("parts") or [])]


def _region(state: str):
    p = os.path.join(state, "site.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    o, n = s.get("origin"), int(s.get("size") or 0)
    if not o or not n:
        return None
    return (int(o[0]), int(o[1]), int(o[0]) + n - 1, int(o[1]) + n - 1)


def _rect(part: dict) -> tuple:
    fp = part.get("footprint")
    if fp and len(fp) == 4:
        x0, z0, x1, z1 = [int(v) for v in fp]
    else:
        x0, z0 = int(part.get("x0", 0)), int(part.get("z0", 0))
        x1, z1 = int(part.get("x1", 0)), int(part.get("z1", 0))
    return (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))


# ------------------------------------------------------------------ the answer

def answer(holds, method: str, why: str, subjects=(), **evidence) -> dict:
    """One predicate's answer, in the one shape. See the module."""
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}, not {method!r}")
    if holds and method in ("declared", "unsupported"):
        raise ValueError(f"{method} evidence cannot establish a predicate: a "
                         f"declaration and an absence are the two things this module "
                         f"exists to stop counting as use")
    return {"holds": None if holds is None else bool(holds), "method": method,
            "evidence": dict(evidence), "subjects": [str(s) for s in subjects],
            "why": why}


def check(world: "World", part, want: str, *, plan=None, registry=None) -> dict:
    """Does `part` deliver `want` in the world that stands? See the module."""
    if want not in WANTS:
        raise ValueError(f"no predicate called {want!r}; this module answers "
                         f"{', '.join(WANTS)}")
    row = part if isinstance(part, dict) else (world.by_name.get(str(part)) or {})
    name = str(row.get("part") or row.get("name") or part)
    prov = world.provenance()
    if not world.standing.get(name, row.get("status") == "built"):
        return answer(None, "unsupported",
                      f"{name} is not standing in the assembled world, so nothing about "
                      f"its use can be observed there", [name], **prov)
    return _CHECKS[want](world, name, row, prov)


# ------------------------------------------------------------------ predicates

def _plot_rows(world: "World", name: str) -> list:
    return [p for p in world.registry if str(p.get("label")) == name]


def _doors_of(world: "World", name: str) -> list:
    """Every door leaf in the assembled world on this part's ground or its one-ring."""
    from . import lint
    rows = _plot_rows(world, name)
    if not rows:
        return []
    out = []
    for (x, y, z) in world.ctx.doors:
        if any(lint.plot_covers(p, x, z, margin=1) for p in rows):
            out.append((int(x), int(y), int(z)))
    return out


def entrance_connected(world: "World", name: str, row: dict, prov: dict) -> dict:
    """A way in that a person can **walk** to from outdoors.

        Not "the type placed a door": the door leaf has to be there in the assembled
        volume, a person has to be able to stand at it, and that stance has to be in the
        walk-only flood from the lane. `lint.e002_door_unreachable` asks this of every door
        in a town; this asks it of one part and answers in this module's shape.
        
    """
    doors = _doors_of(world, name)
    if not doors:
        from . import lint
        rooms = world.walk().get(name) or []
        # **A canopy is not an interior.** A market's stalls shelter standing space and
        # `observe.rooms` finds it, which would make every market a building nobody can
        # get into. The bar is `lint.Context.ENCLOSED`, the same one `interior_walk`
        # uses for the same reason; below it there is no inside for a door to be a door
        # to and this predicate has nothing to decide.
        inside = [r for r in rooms
                  if float(r.get("enclosure") or 0.0) >= lint.Context.ENCLOSED]
        if not inside:
            return answer(None, "unsupported",
                          f"{name} carries no door leaf and encloses nothing at "
                          f"{lint.Context.ENCLOSED:.0%} in the assembled world "
                          f"({len(rooms)} sheltered space(s), none of them an "
                          f"interior): there is nothing here for an entrance to be an "
                          f"entrance to", [name], doors=0, rooms=len(rooms),
                          enclosed=0, **prov)
        return answer(False, "observed",
                      f"{name} encloses {len(inside)} interior(s) and carries no door "
                      f"leaf in the assembled world", [name], doors=0,
                      rooms=len(rooms), enclosed=len(inside), **prov)
    reach = []
    for (x, y, z) in doors:
        s = world.ctx.door_stance(x, y, z)
        reach.append({"at": [x, y, z], "stance": s,
                      "from_outdoors": bool(s is not None
                                            and (x, z, s) in world.ctx.from_outdoors)})
    ok = [d for d in reach if d["from_outdoors"]]
    return answer(bool(ok), "observed",
                  (f"{len(ok)} of {len(doors)} door(s) of {name} can be walked to from "
                   f"outdoors" if ok else
                   f"none of {name}'s {len(doors)} door(s) can be walked to from "
                   f"outdoors: it stands and nobody can get in"),
                  [name], doors=reach, walkable=len(ok),
                  seeds=world.ctx.outdoor_seeds, **prov)


def passage_connected(world: "World", name: str, row: dict, prov: dict) -> dict:
    """Every room of the part reachable from that part's **own** doorways.

        `lint.Context.interior_walk` floods each room walk-only from the doors on its own
        plot; a room under `PASSAGE_FRACTION` is one a person gets into and cannot get
        round. A part with no enclosed room is `unsupported` -- a square has no passages and
        is not thereby badly connected.
        
    """
    rooms = world.walk().get(name) or []
    if not rooms:
        return answer(None, "unsupported",
                      f"{name} encloses no room in the assembled world, so it has no "
                      f"passage to be connected", [name], rooms=0, **prov)
    rooms = [r for r in rooms if r.get("fraction") is not None]
    if not rooms:
        return answer(None, "unsupported",
                      f"{name}'s room(s) have no floor a person could stand on, so "
                      f"there is nothing to walk", [name], **prov)
    short = [r for r in rooms if float(r.get("fraction") or 0.0) < PASSAGE_FRACTION]
    ev = list(rooms)
    return answer(not short, "observed",
                  (f"every one of {name}'s {len(rooms)} room(s) is at least "
                   f"{PASSAGE_FRACTION:.0%} walkable from its own doorways" if not short
                   else f"{len(short)} of {name}'s {len(rooms)} room(s) fall under "
                        f"{PASSAGE_FRACTION:.0%} walkable from its own doorways "
                        f"(least {min(float(r.get('fraction') or 0.0) for r in short):.0%})"),
                  [name], rooms=ev, unreachable=len(short), bar=PASSAGE_FRACTION, **prov)


def _claimed_rects(row: dict, among) -> dict:
    """`{feature: [rect, ...]}` the type claimed, for the features in `among`.

        A **list** per feature, because a feature can stand in several places: a market
        square's stalls are four corner booths, and the box round all four is most of the
        square. `construction.rects_of` is the one normaliser; see the defect it records.
        
    """
    from . import construction
    em = row.get("emitted") if isinstance(row.get("emitted"), dict) else {}
    rects = em.get("rects")
    if not isinstance(rects, dict):
        build = row.get("build") if isinstance(row.get("build"), dict) else {}
        rects = ((build.get("emitted") or {}).get("rects")
                 if isinstance(build.get("emitted"), dict) else None)
    out = {}
    for f, r in (rects or {}).items():
        if f in among:
            got = construction.rects_of(r)
            if got:
                out[str(f)] = got
    return out


def _floor_of(world: "World", name: str, row: dict):
    fy = row.get("floor_y")
    if isinstance(fy, int):
        return fy
    for p in _plot_rows(world, name):
        if isinstance(p.get("y0"), int):
            return int(p["y0"])
    return None


#: What share of a claimed rectangle has to carry something for the thing to be standing
#: in it. `construction._verify_rect`'s bar, on purpose: two instruments that disagree
#: about whether a booth is there is how a place passes one check and fails the other.
STANDS = 0.5


def _stands_in(world: "World", rect, fy: int) -> tuple:
    """`(columns carrying something above the floor, columns in the rectangle)`.

        **A count is not a verdict.** One block left in a razed booth is not a booth, and
        the caller compares against `STANDS` rather than against zero.
        
    """
    from . import construction
    vol = world.ctx.vol
    x0, z0, x1, z1 = rect
    n = tot = 0
    hi = fy + construction.FEATURE_COURSES + 1
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for z in range(min(z0, z1), max(z0, z1) + 1):
            tot += 1
            if any(vol.name(x, y, z) != "air" for y in range(fy + 1, hi)):
                n += 1
    return (n, tot)


def identifies(feature: str) -> tuple:
    """The block words that make `feature` that feature, or `()` where none do.

        `()` is not "no blocks": it is **this build lays nothing that identifies it**, and
        every caller reports that rather than falling back to the mass. See `FEATURE_BLOCKS`.
        
    """
    return tuple(FEATURE_BLOCKS.get(str(feature)) or ())


def _identity_in(world: "World", rect, fy: int, words) -> list:
    """The identifying blocks standing in `rect`, in the feature's own storey.

        The same window `_stands_in` counts mass in, so a feature whose mass stands and whose
        identity does not is one fact about one rectangle and not two instruments disagreeing.
        
    """
    from . import construction
    vol = world.ctx.vol
    x0, z0, x1, z1 = rect
    hi = fy + construction.FEATURE_COURSES + 1
    out = []
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for z in range(min(z0, z1), max(z0, z1) + 1):
            for y in range(fy + 1, hi):
                n = vol.name(x, y, z)
                if n != "air" and any(w in n for w in words):
                    out.append(n)
    return out


def _sky_over(world: "World", cells, fy: int) -> int:
    """How many of these columns carry nothing above a court's headroom."""
    vol = world.ctx.vol
    top = min(fy + SKY_COURSES, vol.y0 + vol.shape[1] - 1)
    n = 0
    for (x, z) in cells:
        if all(vol.name(x, y, z) == "air" for y in range(fy + 4, top + 1)):
            n += 1
    return n


def _reachable_at(world: "World", rect, fy: int, flood: dict) -> list:
    """The stances in `flood` within `REACH` columns of `rect`, at about floor level."""
    x0, z0, x1, z1 = rect
    nav, out = world.ctx.nav, []
    for x in range(min(x0, x1) - REACH, max(x0, x1) + REACH + 1):
        for z in range(min(z0, z1) - REACH, max(z0, z1) + REACH + 1):
            s = nav.stance_near(x, z, fy + 1, tol=3)
            if s is not None and (x, z, s) in flood:
                out.append((x, z, s))
    return out


def _inside_reach(world: "World", name: str) -> dict:
    """Walk-only reachability from this part's own doorways, plus from outdoors.

        The union, because a market's stalls stand in the open and a forge stands in a
        room: "can a person get to it" is one question and the two floods are two ways in
        to it, not two different standards.
        
    """
    ctx = world.ctx
    seeds = []
    for (x, y, z) in _doors_of(world, name):
        s = ctx.door_stance(x, y, z)
        if s is not None:
            seeds.append((x, z, s))
    got = dict(ctx.from_outdoors)
    if seeds:
        got.update(ctx.nav.flood(seeds, max_jumps=0))
    return got


def equipment_reachable(world: "World", name: str, row: dict, prov: dict) -> dict:
    """The thing the function needs still stands, and a person can walk up to it.

        Two failures, told apart, because they have different owners: equipment that is
        **gone** from the assembled world (a later part overwrote it, or the type never laid
        it and said it had) is a build finding; equipment that stands and has no reachable
        stance beside it is a layout or a circulation finding. A part that claims a feature
        and gives no rectangle is `declared` and `holds: None` -- the growth brief already
        says a declared feature with no rectangle is not evidence, and this is that rule
        asked of the world instead of of the return value.
        
    """
    rects = _claimed_rects(row, EQUIPMENT)
    em = row.get("emitted") if isinstance(row.get("emitted"), dict) else {}
    claimed = {f: v for f, v in (em.get("features") or {}).items()
               if f in EQUIPMENT and v}
    fy = _floor_of(world, name, row)
    if not rects:
        if claimed:
            return answer(None, "declared",
                          f"{name} says it emitted {sorted(claimed)} and gave no "
                          f"rectangle for any of them; a declared feature with no "
                          f"rectangle cannot be looked for", [name],
                          declared=sorted(claimed), rects=0, **prov)
        return answer(None, "unsupported",
                      f"{name} claims no equipment, so there is none to find", [name],
                      **prov)
    if fy is None:
        return answer(None, "unsupported",
                      f"{name} records no floor level, so its rectangles cannot be "
                      f"read off the assembled volume", [name],
                      rects=sorted(rects), **prov)
    flood = _inside_reach(world, name)
    rows, gone, unreachable, wrong = [], [], [], []
    for f, places in sorted(rects.items()):
        # **each place the feature stands, on its own.** A market with two of its four
        # booths razed by a neighbour's siting is not a market with stalls, and this is
        # where that becomes two numbers instead of one word.
        words = identifies(f)
        here, missing, shut, unnamed = [], 0, 0, 0
        for rect in places:
            stood, cells = _stands_in(world, rect, fy)
            up = stood >= STANDS * cells
            # **...and mass is not identity.** A rectangle half full of the voice's own
            # masonry is a rectangle half full of the voice's own masonry. Where the
            # library lays a block that makes this feature that feature, it has to be
            # there; where it lays none, the mass is what there is and the record says
            # so.
            marks = _identity_in(world, rect, fy, words) if words else None
            named = marks is None or bool(marks)
            there = bool(up and named)
            at = _reachable_at(world, rect, fy, flood) if there else []
            here.append({"rect": rect, "cells": cells, "standing": stood,
                         "share": round(stood / float(cells), 3) if cells else 0.0,
                         "up": bool(up), "there": there, "stances": len(at),
                         "identity": ("unsupported" if marks is None
                                      else sorted(set(marks))),
                         "identified": None if marks is None else len(marks),
                         "looked_for": list(words)})
            missing += not up
            unnamed += bool(up and not named)
            shut += bool(there and not at)
        rows.append({"feature": f, "places": len(places),
                     "standing": len(places) - missing,
                     "unidentified": unnamed, "unreachable": shut,
                     "identity": "unsupported" if not words else list(words), "at": here})
        if missing:
            gone.append(f)
        elif unnamed:
            wrong.append(f)
        elif shut:
            unreachable.append(f)
    ok = not gone and not wrong and not unreachable
    return answer(ok, "observed",
                  (f"{name}: every one of {sorted(rects)} stands in the assembled world "
                   f"and can be walked up to"
                   + (f" ({', '.join(sorted(f for f in rects if not identifies(f)))}: "
                      f"mass only, this build lays no block that identifies them)"
                      if any(not identifies(f) for f in rects) else "") if ok else
                   f"{name}: " + "; ".join(
                       ([f"{gone} no longer stands in the assembled world"] if gone else [])
                       + ([f"{wrong} occupies its rectangle and carries none of the "
                           f"block(s) that make it that feature "
                           f"({', '.join(sorted(w for f in wrong for w in identifies(f)))})"]
                          if wrong else [])
                       + ([f"{unreachable} stands and no stance beside it is reachable "
                           f"on foot"] if unreachable else []))),
                  [name], features=rows, gone=gone, unidentified=wrong,
                  unreachable=unreachable, floor_y=fy, **prov)


def circulation_clear(world: "World", name: str, row: dict, prov: dict) -> dict:
    """A part whose apron a later terrace buried, or which ended up on an island of its
        own, is one nobody walks past. `lint.Context.circulation` is the largest walk-only
        outdoor component and is the thing to be on; being merely walkable somewhere is not.
        
    """
    if not world.ctx.circulation:
        return answer(None, "unsupported",
                      f"this world has no outdoor circulation component large enough to "
                      f"measure ({world.ctx.nav.vol.shape}); nothing can be said about "
                      f"{name}'s", [name], **prov)
    rect = _rect(row) if row.get("x0") is not None else None
    if rect is None:
        rows = _plot_rows(world, name)
        if not rows:
            return answer(None, "unsupported",
                          f"{name} has no footprint on the registry, so there is no "
                          f"apron to look at", [name], **prov)
        rect = (min(int(p["x0"]) for p in rows), min(int(p["z0"]) for p in rows),
                max(int(p["x1"]) for p in rows), max(int(p["z1"]) for p in rows))
    x0, z0, x1, z1 = rect
    nav, circ = world.ctx.nav, world.ctx.circulation
    ring, on, buried = [], 0, 0
    for k in range(1, APRON + 1):
        for x in range(x0 - k, x1 + k + 1):
            for z in (z0 - k, z1 + k):
                ring.append((x, z))
        for z in range(z0 - k + 1, z1 + k):
            for x in (x0 - k, x1 + k):
                ring.append((x, z))
    for (x, z) in ring:
        s = nav.ground_stance(x, z)
        if s is None:
            buried += 1
        elif (x, z, s) in circ:
            on += 1
    share = on / float(len(ring)) if ring else 0.0
    return answer(bool(on), "observed",
                  (f"{on} of {len(ring)} apron column(s) round {name} are on the "
                   f"settlement's walking circulation ({share:.0%})" if on else
                   f"not one of the {len(ring)} apron column(s) round {name} is on the "
                   f"settlement's walking circulation: it stands off the network"),
                  [name], apron=len(ring), on_circulation=on, unstandable=buried,
                  share=round(share, 3), rect=list(rect), **prov)


def court_accessible(world: "World", name: str, row: dict, prov: dict) -> dict:
    """The court is open and paved **now**, and can be reached from inside the part.

        `construction._verify_open` asks the first half at emission. This asks it again on
        the assembled volume -- where a neighbour's wall, a later terrace or a finishing
        pass may have filled it -- and adds the two halves nobody asked: that a person
        standing in the part can get out into it, and that it is **open to the sky**.

        **A court is a floor and the sky over it.** The composition round: paved at the floor
        and clear for three courses is also true of a room, a cellar and the ground floor of
        an arcade, so a court a later storey or a neighbour's roof was carried over remained
        a court by this measurement. `SKY_COURSES` above the headroom have to be clear over
        `construction.OPEN_STANDS` of its cells -- the same share the paving is held to, so a
        portico along one side is not a filled court and a roof over it is.
        
    """
    rects = _claimed_rects(row, COURTS)
    fy = _floor_of(world, name, row)
    if not rects:
        return answer(None, "unsupported",
                      f"{name} claims no court, so there is none to reach", [name],
                      **prov)
    if fy is None:
        return answer(None, "unsupported",
                      f"{name} records no floor level, so its court cannot be read off "
                      f"the assembled volume", [name], rects=sorted(rects), **prov)
    from . import construction
    vol, flood = world.ctx.vol, _inside_reach(world, name)
    rows, filled, roofed, cut_off = [], [], [], []
    for f, places in sorted(rects.items()):
        for rect in places:
            x0, z0, x1, z1 = rect
            cells = [(x, z) for x in range(min(x0, x1), max(x0, x1) + 1)
                     for z in range(min(z0, z1), max(z0, z1) + 1)]
            open_cells = [c for c in cells
                          if vol.name(c[0], fy, c[1]) != "air"
                          and all(vol.name(c[0], fy + k, c[1]) == "air"
                                  for k in (1, 2, 3))]
            sky = _sky_over(world, open_cells, fy)
            at = _reachable_at(world, rect, fy, flood) if open_cells else []
            share = len(open_cells) / float(len(cells)) if cells else 0.0
            sky_share = sky / float(len(cells)) if cells else 0.0
            rows.append({"feature": f, "rect": rect, "cells": len(cells),
                         "open": len(open_cells), "share": round(share, 3),
                         "sky": sky, "sky_share": round(sky_share, 3),
                         "sky_courses": SKY_COURSES, "stances": len(at)})
            if share < construction.OPEN_STANDS:
                if f not in filled:
                    filled.append(f)
            elif sky_share < construction.OPEN_STANDS:
                if f not in roofed:
                    roofed.append(f)
            elif not at and f not in cut_off:
                cut_off.append(f)
    ok = not filled and not roofed and not cut_off
    return answer(ok, "observed",
                  (f"{name}: {sorted(rects)} is paved, open to the sky and reachable "
                   f"from inside" if ok else f"{name}: " + "; ".join(
                       ([f"{filled} is no longer open paved ground in the assembled "
                         f"world"] if filled else [])
                       + ([f"{roofed} is paved and something stands over it within "
                           f"{SKY_COURSES} course(s): a covered court is a room"]
                          if roofed else [])
                       + ([f"{cut_off} is open and no stance in it is reachable from "
                           f"this part's own doors"] if cut_off else []))),
                  [name], courts=rows, filled=filled, roofed=roofed, cut_off=cut_off,
                  floor_y=fy, **prov)


def range_relation(world: "World", name: str, row: dict, prov: dict) -> dict:
    """The court is enclosed by the part's own ranges, and they open onto it.

        The one predicate about architectural **organisation** rather than about access: a
        courtyard house is a ring of ranges round a yard, and a yard with a range on one
        side and open ground on three is a building with a garden. Measured as the sides of
        the court that carry this part's own mass at wall height (`COURT_SIDES` of four),
        and as the rooms of this part whose floor touches the court's edge.
        
    """
    rects = _claimed_rects(row, COURTS)
    fy = _floor_of(world, name, row)
    if not rects:
        return answer(None, "unsupported",
                      f"{name} claims no court, so it has no court-and-range relation "
                      f"to have", [name], **prov)
    if fy is None:
        return answer(None, "unsupported",
                      f"{name} records no floor level", [name], **prov)
    vol = world.ctx.vol
    rows, bad = [], []
    rooms = world.walk().get(name) or []
    for f, places in sorted(rects.items()):
        # a court is one place; a type claiming several is asked about the first, which
        # is the only one a "court and its ranges" relation is defined over
        rect = places[0]
        x0, z0, x1, z1 = rect
        sides = {"north": [(x, z0 - 1) for x in range(x0, x1 + 1)],
                 "south": [(x, z1 + 1) for x in range(x0, x1 + 1)],
                 "west": [(x0 - 1, z) for z in range(z0, z1 + 1)],
                 "east": [(x1 + 1, z) for z in range(z0, z1 + 1)]}
        stood = {}
        for side, cells in sides.items():
            n = sum(1 for (x, z) in cells
                    if any(vol.name(x, y, z) != "air" for y in range(fy + 1, fy + 4))
                    and world.ctx.plot_at(x, z) == name)
            stood[side] = {"cells": len(cells), "standing": n,
                           "ranged": bool(cells and n >= 0.5 * len(cells))}
        ranged = sum(1 for v in stood.values() if v["ranged"])
        # a range that opens onto the court: a room of this part whose bounding box
        # touches the court's edge on a side that carries mass
        touching = sum(1 for r in rooms
                       if _touches(r.get("bbox"), rect))
        rows.append({"feature": f, "rect": rect, "sides": stood, "ranged": ranged,
                     "rooms_on_court": touching})
        if ranged < COURT_SIDES or not touching:
            bad.append(f)
    ok = not bad
    return answer(ok, "inferred",
                  (f"{name}: the court is ranged on {rows[0]['ranged']} of four sides by "
                   f"this part's own mass and {rows[0]['rooms_on_court']} of its room(s) "
                   f"adjoin it" if ok else
                   f"{name}: {bad} " + ("is not a court enclosed by this part's ranges: "
                                        + "; ".join(f"{r['feature']} ranged on "
                                                    f"{r['ranged']} of four side(s), "
                                                    f"{r['rooms_on_court']} room(s) on it"
                                                    for r in rows if r["feature"] in bad))),
                  [name], courts=rows, bar=COURT_SIDES, **prov)


def _touches(bbox, rect) -> bool:
    """Does a room's bounding box share an edge with the court's rectangle?"""
    if not bbox or len(bbox) != 6:
        return False
    rx0, rz0, rx1, rz1 = int(bbox[0]), int(bbox[2]), int(bbox[3]), int(bbox[5])
    cx0, cz0, cx1, cz1 = rect
    return not (rx0 > cx1 + 1 or rx1 < cx0 - 1 or rz0 > cz1 + 1 or rz1 < cz0 - 1)


_CHECKS = {"entrance_connected": entrance_connected,
           "passage_connected": passage_connected,
           "equipment_reachable": equipment_reachable,
           "circulation_clear": circulation_clear,
           "court_accessible": court_accessible,
           "range_relation": range_relation}


# ------------------------------------------------------------------ the sweep

def features_for(world: "World", part, *, wants=WANTS, plan=None) -> dict:
    """Every predicate for one part, in one call: `{want: answer}`.

        What `construction.confirm` and `growth.gate` consume, and what a caller reporting
        coverage reads: the answers carry their own method, so a part with three observed
        predicates and three unsupported ones is not a part that scored a half.
        
    """
    return {w: check(world, part, w, plan=plan) for w in wants}


def says(got: dict) -> str:
    """One line over a `features_for` map, for a log or a finding."""
    by = {"observed": [], "inferred": [], "declared": [], "unsupported": []}
    for w, a in sorted(got.items()):
        by[a["method"]].append(f"{w}={'holds' if a['holds'] else
                                      ('fails' if a['holds'] is False else 'open')}")
    return "; ".join(f"{m}: {', '.join(v)}" for m, v in by.items() if v) or "nothing asked"
