"""What a built section actually demonstrates, measured off its own artifacts.

`scripts/check_design.py:516` compared `len(section.json["demonstrated"])` with the
length of the round file's list. Nothing in the repository wrote that file: it was hand-
authored by the agent running the round, and the same report conceded in prose that the
contrasting ring fabrics "cannot be seen in it at all". A gate that counts asserted
relationships is not a gate; the review's own conclusion was that a numerical gate
cannot overrule a material negative reading, and the deeper point is that this number
was never a measurement in the first place.

So the record is derived here, from the artifacts production wrote:

  `parts.json`        what stood, each part's emitted features, rects and footprint
  `usable.json`       the final-world predicates re-read on the assembled world
  `circulation.json`  the walk check over the lanes, and the thresholds served
  `network.json`      the lane cells and each part's threshold
  `plan.json`         the districts, their characters and their leaves
  `resolution.json`   each region's scope, developable, allocated and built columns

Every relationship answers `demonstrated`, `failed` or **`unmeasured`**, and the third is
not a synonym for the first: a measurement this build did not take holds nothing. The
numbers are reported whatever the verdict, because the numbers are the evidence and the
verdict is a reading of them.

The contrast test is the one worth stating plainly, since it is the relationship the last
round claimed and did not have. It is decided on **built** geometry -- what stood, how
big it is, how close it stands to its neighbour, and how much of each side's ground it
covers -- and never on a type name, a character word or a lot allocation. Two fabrics
whose difference is real are different in at least two of those three ways.
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics

#: The least a side of the boundary may contribute before it is context rather than
#: fabric. Eight complete structures is the smallest number that can show a street's
#: rhythm on both sides of a gate; below it the section has a built side and a backdrop.
#: Registered before the section was selected.
SIDE_STRUCTURES = 8

#: How different two fabrics have to be on a measure before the difference is the
#: composition's and not the seed's. 1.4x is the ratio the registered characters
#: themselves differ by at the smallest -- the lower ring's 40-column block against the
#: middle ring's 48 is 1.2, its 6x8 lot against 13x13 is 3.5 -- so a fabric pair that
#: cannot reach 1.4 on any of built cover, footprint or spacing is not laying the
#: contrast its own character asked for.
CONTRAST_RATIO = 1.4

#: How many of the three quantitative contrasts (built cover, median footprint, median
#: neighbour gap) must hold. Two, so one coincidence is not a contrast and one failure
#: is not a refutation.
CONTRAST_MEASURES = 2

#: Types that are an anchor: ground the programme reserved for a use of its own.
ANCHOR_TYPES = ("market", "square", "plaza")

#: Types that are a court: enclosed open ground belonging to a building or a group.
#: **The types a court can be.** `plaza` and `square` joined this list in the
#: neighbourhood delivery round, and the reason is the round's own composition: a court
#: a block's four ranges enclose is paved open ground, and the compiler now draws it
#: from `district_compile.ENCLOSED_COURT_TYPES` -- paved first, because a `yard` fences
#: its own perimeter and a court the buildings already enclose does not want a second
#: fence. Leaving them out of this list is how the two crowded-ring courts that stood on
#: the built section vanished from its court record between one build and the next.
COURT_TYPES = ("court_large", "court_small", "courtyard_house", "yard", "garden",
               "plaza", "square")

#: The `emitted.rects` keys that are a **court** rather than a building. `usable.COURTS`
#: is the same list one level down; kept here so `_side_measures` can measure a court
#: without importing the predicate module. See `_side_measures` for why the share that
#: used to be called `court_share` was the footprint of a court-type *building*.
COURT_RECTS = ("courtyard", "court", "yard", "plaza", "square")

#: Types whose ground is open by intent rather than by omission.
OPEN_TYPES = ("plaza", "garden", "grove", "yard", "field")

RELATIONSHIPS = ("contrast", "anchor", "courts", "route", "features", "functions")


# ------------------------------------------------------------------ reading the state

def _load(state: str, name: str):
    p = os.path.join(state, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def _rows(parts_record: dict | None) -> list:
    return [r for w in (parts_record or {}).get("waves", []) or []
            for r in (w.get("parts") or [])]


def _rect_area(rect) -> int:
    if not rect or len(rect) != 4:
        return 0
    return (abs(int(rect[2]) - int(rect[0])) + 1) * (abs(int(rect[3]) - int(rect[1])) + 1)


def _footprint(row: dict) -> list | None:
    em = row.get("emitted") or {}
    for key in ("footprint",):
        r = em.get(key)
        if isinstance(r, (list, tuple)) and len(r) == 4:
            return [int(v) for v in r]
    r = (em.get("rects") or {}).get("main")
    if isinstance(r, (list, tuple)) and len(r) == 4:
        return [int(v) for v in r]
    return None


def _building_rect(row: dict) -> list | None:
    """**The building, as distinct from everything the part wrote.**

        The neighbourhood delivery round. `emitted.footprint` is the bounding box of every
        block a part laid, which includes the platform `site()` lays one course wider than
        the pad all round, the eaves and the doorstep. That is the right rectangle for "how
        much ground did this part disturb" and the wrong one for "how far apart do these two
        houses stand": two neighbours whose *platforms* touch measure a gap of zero while
        their walls are a column apart, and a terrace with a column of daylight between every
        pair of houses reads as continuous.

        `emitted.rects["main"]` is the type's own declaration of the building it built, and
        it is what a neighbour gap is between. Falls back to the emitted extent where a type
        publishes no rectangle, and `_side_measures` records which was read and keeps the old
        figure beside the corrected one.
        
    """
    r = ((row.get("emitted") or {}).get("rects") or {}).get("main")
    if isinstance(r, (list, tuple)) and len(r) == 4:
        return [int(v) for v in r]
    return _footprint(row)


def _centre(rect) -> tuple:
    return ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)


def _gap(a: list, b: list) -> int:
    """L1 gap between two rectangles, 0 where they touch or overlap."""
    return (max(0, max(b[0] - a[2], a[0] - b[2]))
            + max(0, max(b[1] - a[3], a[1] - b[3])))


def _clip_area(rect, section) -> int:
    x0 = max(min(rect[0], rect[2]), min(section[0], section[2]))
    x1 = min(max(rect[0], rect[2]), max(section[0], section[2]))
    z0 = max(min(rect[1], rect[3]), min(section[1], section[3]))
    z1 = min(max(rect[1], rect[3]), max(section[1], section[3]))
    if x1 < x0 or z1 < z0:
        return 0
    return (x1 - x0 + 1) * (z1 - z0 + 1)


def _districts(state: str, plan: dict | None) -> list:
    """Every district of this design, with its rectangle and its resolved demand.

        Three shapes, in order of preference, because the same design is written down three
        ways and only one of them is the one a stage happens to have loaded:

          `plan.place.json`  the solver's own record: districts with their rects, their
                             column record and the demand binding copied onto them;
          `resolution.json`  the region record, which carries the four column counts;
          `plan.json`        the built tree, whose districts are nodes with `kind:
                             "district"` under a root district rather than a top-level list.

        `section.record` used to read `plan["districts"]` alone, which exists in the first
        and not in the third -- so on a built state every side measured 0 ground columns and
        `built_cover` came back `None`. A measurement that quietly answers None is the thing
        this module exists to refuse, so it reads all three.
        
    """
    got: dict = {}
    place = _load(state, "plan.place.json") or {}
    for d in place.get("districts") or []:
        if d.get("name") and d.get("x0") is not None:
            got[str(d["name"])] = dict(d)
    for r in (_load(state, "resolution.json") or {}).get("regions") or []:
        name = str(r.get("name") or "")
        if not name:
            continue
        row = got.setdefault(name, {"name": name})
        rect = r.get("rect")
        if rect and row.get("x0") is None:
            row.update({"x0": rect[0], "z0": rect[1], "x1": rect[2], "z1": rect[3]})
        # **The density and the form, beside the columns.** The spatial-design round: a
        # district's *obligations* are read off its character, its arrangement and its
        # compile record (`demand.court_obligation`), and none of those three was
        # carried here -- so on the delivered candidate every district answered "no
        # character" and the section could not name one court-owing subject.
        # `resolution.json` is the one shape that always has the density word.
        for k in ("scope_columns", "developable_columns", "allocated_columns",
                  "built_columns", "built_from", "density", "role", "character",
                  "arrangement", "courts", "lots"):
            if r.get(k) is not None:
                row.setdefault(k, r[k])

    def walk(node):
        if isinstance(node, list):
            for n in node:
                walk(n)
            return
        if not isinstance(node, dict):
            return
        if node.get("kind") == "district" and node.get("name") \
                and node.get("x0") is not None:
            row = got.setdefault(str(node["name"]), {"name": str(node["name"])})
            for k in ("x0", "z0", "x1", "z1", "demand", "character", "defines",
                      "density", "arrangement", "courts"):
                if node.get(k) is not None:
                    row.setdefault(k, node[k])
        for key in ("districts", "children", "parts", "quarters"):
            if key in node:
                walk(node[key])
    walk((plan or {}).get("parts") or [])
    walk((plan or {}).get("districts") or [])
    # **...and the character the district was actually compiled with** (the fabric reset
    # round): where none of the three shapes carries one, the district's own plan file
    # does. Without it a district that declared no shared court (a street composition of
    # courtyard houses, `courtyard_share: 0`) read as owing one at its density word's
    # registered default, and a revision that removed its houses was charged a court it
    # never adopted.
    for name, row in got.items():
        if row.get("character") is None:
            pf = _load(state, f"plan.district.{name}.json") or {}
            if isinstance(pf.get("character"), dict):
                row["character"] = dict(pf["character"])
    return list(got.values())


def _usable_of(usable: dict | None) -> dict:
    """`{(part, want): answer}` off `usable.json`."""
    out = {}
    for c in (usable or {}).get("checks") or []:
        out[(c.get("part"), c.get("want"))] = c
    return out


def _holds(answer: dict | None) -> bool:
    """True only for an affirmative answer measured on the assembled world.

        `holds: None` is not a pass and neither is `method: declared`: an unknown answer is
        owed, not satisfied. `intent._usable_verdict` counted a declared answer as a
        predicate that had run; this is the same rule written once, positively.
        
    """
    return bool(answer) and answer.get("holds") is True \
        and str(answer.get("method")) in ("observed", "inferred")


# ------------------------------------------------------------------ the relationships

def _of(r: dict, prefix: str) -> bool:
    """Does this built row belong under `prefix` (a district or a side's ring)? By its
    part name, or -- for a leaf whose name the plan did not prefix -- by the district
    the plan put it in (`_district`, set by `record` from the leaf's `in`)."""
    return (str(r.get("part", "")).startswith(prefix)
            or str(r.get("_district") or "").startswith(prefix))


def _side_measures(side: str, prefix: str, rows: list, districts: list,
                   section: list, enclosure=None) -> dict:
    """What one side of the boundary actually built, in columns.

        **Mass, and the rectangle it stands in, apart.** The neighbourhood round. Two
        measurements here were the enclosing rectangle under a name that said otherwise, and
        both of them decide the section's registered contrast test:

          * `built_columns` was `sum(_rect_area(footprint))`. The crowded side is row houses,
            whose mass fills 0.99 of their rectangle; the calm side is courtyard houses, whose
            rectangles **contain their courts**. So the measure that decides whether two
            fabrics differ systematically flattered the side with the courts -- and flatters
            it more now that the courts are a real share of the pad rather than a 2x2 light
            well. `construction.confirm` writes `emitted.occupied_columns`; that is what
            `built_cover` is over where a row carries it, the rectangle figure stays beside
            it under its own name, and `from` says which;
          * `court_share` did not measure courts at all. It was the footprint area of every
            part whose *type* is in `COURT_TYPES` -- courts and the ranges round them
            together, which is a building footprint and not a court. Both types now publish
            `emitted.rects.courtyard`, so the court is measured from its own rectangle and
            the old figure keeps an accurate name (`court_type_columns`).

        Where a row carries no `occupied_columns` it is counted in `parts_without_mass` and
        its rectangle is used, rather than being silently taken as nothing: an unmeasured
        part is not a part that built nothing.
        
    """
    mine = [r for r in rows
            if r.get("kind", "plot") == "plot" and r.get("stood")
            and _of(r, prefix)]
    foots = [f for f in (_footprint(r) for r in mine) if f]
    areas = [_rect_area(f) for f in foots]
    mass, no_mass = 0, []
    for r in mine:
        got = (r.get("emitted") or {}).get("occupied_columns")
        if isinstance(got, int):
            mass += int(got)
        else:
            no_mass.append(str(r.get("part")))
            mass += _rect_area(_footprint(r) or [])
    # **the gap between the buildings, and the gap between everything they laid.** The
    # corrected ruler and the old one, both on the record: see `_building_rect`.
    walls = [f for f in (_building_rect(r) for r in mine) if f]
    published = sum(1 for r in mine
                    if ((r.get("emitted") or {}).get("rects") or {}).get("main"))
    gaps = []
    for i, a in enumerate(walls):
        others = [b for j, b in enumerate(walls) if j != i]
        if others:
            gaps.append(min(_gap(a, b) for b in others))
    gaps_extent = []
    for i, a in enumerate(foots):
        others = [b for j, b in enumerate(foots) if j != i]
        if others:
            gaps_extent.append(min(_gap(a, b) for b in others))
    mine_d = [d for d in (districts or [])
              if str(d.get("name", "")).startswith(prefix)]
    ground = sum(_clip_area([d.get("x0"), d.get("z0"), d.get("x1"), d.get("z1")], section)
                 for d in mine_d
                 if all(d.get(k) is not None for k in ("x0", "z0", "x1", "z1")))
    here = [r for r in rows
            if _of(r, prefix) and r.get("stood")]
    court_type = sum(_rect_area(_footprint(r) or []) for r in here
                     if str(r.get("type")) in COURT_TYPES)
    courts, unpublished = 0, []
    for r in here:
        em = r.get("emitted") if isinstance(r.get("emitted"), dict) else {}
        rects = em.get("rects") or {}
        got = [rects[k] for k in COURT_RECTS if isinstance(rects.get(k), (list, tuple))
               and len(rects[k]) == 4]
        if got:
            courts += sum(_rect_area(c) for c in got)
        elif (enclosure or {}).get(str(r.get("part"))):
            # **A block's court is the open part itself, not a rect a building claims.**
            # The block design round. `COURT_RECTS` is what a *courtyard house*
            # publishes about the yard inside it; a court that a block's four ranges
            # enclose is an area leaf whose whole footprint **is** the court, and it
            # publishes no such rect -- so the crowded ring's composed courts were
            # counted as zero columns and `court_share` read 0.0 beside two courts
            # standing in the world. The claim the compiler wrote on the leaf
            # (`enclosure`) is what says this part is one; `usable.court_enclosed` is
            # what says the ranges really stand round it.
            courts += _rect_area(_footprint(r) or [])
        elif str(r.get("type")) in COURT_TYPES:
            unpublished.append(str(r.get("part")))
    storeys = [int((r.get("emitted") or {}).get("storeys") or 0) for r in mine
               if (r.get("emitted") or {}).get("storeys")]
    cz = [_centre(f)[1] for f in foots]
    over = "emitted.occupied_columns" if not no_mass else (
        "emitted.occupied_columns where the row carries it and the enclosing rectangle "
        f"for the {len(no_mass)} that do not")
    return {"side": side, "prefix": prefix, "structures": len(mine),
            "built_columns": int(mass),
            "built_rectangle_columns": sum(areas),
            "parts_without_mass": sorted(no_mass)[:40],
            "ground_columns": ground,
            "built_cover": round(mass / ground, 4) if ground else None,
            "built_cover_by_rectangle": round(sum(areas) / ground, 4) if ground else None,
            # **The median footprint is the building's, beside the extent's.** The block
            # design round, after a second independent reader: `median_footprint` was
            # the area of `emitted.footprint` -- the bounding box of every block a part
            # laid, platform, ledge and eaves included -- while `median_neighbour_gap`
            # beside it in the same record was corrected off `emitted.rects.main` in the
            # delivery round. One contrast test, two rulers, and on the building
            # rectangle the direction of the size contrast between the two fabrics
            # **reverses**. Both are published; the one the contrast test reads is the
            # building's, the same rectangle the gap is between.
            "median_footprint": (statistics.median([_rect_area(f) for f in walls])
                                 if walls else None),
            "median_footprint_by_extent": (statistics.median(areas) if areas else None),
            "median_neighbour_gap": statistics.median(gaps) if gaps else None,
            # the old ruler, retained beside the corrected one: the gap between the
            # whole of what two parts laid, platforms and eaves included
            "median_neighbour_gap_by_extent": (statistics.median(gaps_extent)
                                               if gaps_extent else None),
            "buildings_publishing_a_rect": int(published),
            "court_columns": int(courts),
            "court_share": round(courts / ground, 4) if ground else None,
            "courts_unpublished": sorted(unpublished)[:40],
            "court_type_columns": int(court_type),
            "court_type_share": round(court_type / ground, 4) if ground else None,
            "median_storeys": statistics.median(storeys) if storeys else None,
            "centre_z": round(statistics.median(cz), 1) if cz else None,
            "from": (f"`median_neighbour_gap` is between the buildings "
                     f"(emitted.rects['main'], published by {published} of {len(mine)}) "
                     f"and `median_neighbour_gap_by_extent` is the old figure, between "
                     f"the whole of what each part laid -- two neighbours whose "
                     f"platforms touch measure zero on the second and their real "
                     f"distance on the first; built cover over {over}; "
                     f"`built_rectangle_columns` and "
                     f"`built_cover_by_rectangle` are the enclosing rectangles the same "
                     f"figure used to be; `court_columns` is emitted.rects"
                     f"{sorted(COURT_RECTS)} and `court_type_columns` is the footprint of "
                     f"every {list(COURT_TYPES)} part; plan district rects clipped to "
                     f"the section")}


def _contrast(registered: dict, rows: list, districts: list, section: list,
              boundary_runs: list, enclosure=None) -> dict:
    sides = dict(registered.get("sides") or {})
    if len(sides) < 2:
        return {"status": "unmeasured",
                "how": "the section registers fewer than two sides of a boundary"}
    got = [_side_measures(k, v, rows, districts, section, enclosure=enclosure)
           for k, v in sorted(sides.items())]
    a, b = got[0], got[1]
    enough = [s["side"] for s in got if s["structures"] >= SIDE_STRUCTURES]
    held, tried = [], []

    def ratio(key):
        va, vb = a.get(key), b.get(key)
        if not va or not vb:
            return None
        return round(max(va, vb) / min(va, vb), 3)

    for key in ("built_cover", "median_footprint"):
        r = ratio(key)
        tried.append({"measure": key, "ratio": r, "a": a.get(key), "b": b.get(key),
                      "bar": CONTRAST_RATIO})
        if r is not None and r >= CONTRAST_RATIO:
            held.append(key)
    ga, gb = a.get("median_neighbour_gap"), b.get("median_neighbour_gap")
    gap_holds = (ga is not None and gb is not None
                 and (min(ga, gb) <= 0 < max(ga, gb)
                      or (min(ga, gb) > 0 and max(ga, gb) / min(ga, gb) >= CONTRAST_RATIO)))
    tried.append({"measure": "median_neighbour_gap", "a": ga, "b": gb,
                  "holds": bool(gap_holds),
                  "bar": "attached on one side and detached on the other, or a "
                         f"{CONTRAST_RATIO}x difference in spacing"})
    if gap_holds:
        held.append("median_neighbour_gap")
    # **On opposite sides of the boundary, not merely both inside the section.** A
    # section can hold two fabrics and a wall and still not be a transition.
    across = None
    if boundary_runs and a.get("centre_z") is not None and b.get("centre_z") is not None:
        zs = [p[1] for run in boundary_runs for p in run.get("path") or []]
        if zs:
            line = statistics.median(zs)
            across = (a["centre_z"] - line) * (b["centre_z"] - line) < 0
    ok = (len(enough) == 2 and len(held) >= CONTRAST_MEASURES and across is True)
    return {"status": "demonstrated" if ok else
            ("unmeasured" if across is None else "failed"),
            "how": (f"{a['side']} built {a['structures']} structure(s) covering "
                    f"{a['built_cover']} of its ground at a median footprint of "
                    f"{a['median_footprint']} columns and a median neighbour gap of "
                    f"{a['median_neighbour_gap']}; {b['side']} built {b['structures']} "
                    f"covering {b['built_cover']} at {b['median_footprint']} columns and "
                    f"a gap of {b['median_neighbour_gap']}. "
                    f"{len(held)} of 3 measures differ by {CONTRAST_RATIO}x or more "
                    f"({', '.join(held) or 'none'}); "
                    + ("the two fabrics stand on opposite sides of the boundary"
                       if across else "they do not straddle the boundary"
                       if across is False else
                       "no boundary run was found to measure across")),
            "measured": {"sides": got, "tests": tried, "held": held,
                         "across_boundary": across,
                         "sides_with_enough_fabric": enough,
                         "bars": {"side_structures": SIDE_STRUCTURES,
                                  "ratio": CONTRAST_RATIO,
                                  "measures_needed": CONTRAST_MEASURES}},
            "from": ["parts.json", "plan.place.json", "resolution.json"]}


def _street_cells(state: str, network: dict | None) -> set:
    """Every column a street runs over: the routed arterial and the lanes the
    circulation laid. What an anchor edge that is *open* is open onto."""
    out = set()
    place = _load(state, "plan.place.json") or {}
    for c in ((place.get("arterials") or {}).get("cells") or []):
        with contextlib.suppress(Exception):
            out.add((int(c[0]), int(c[1])))
    for key in ("cells", "lanes"):
        for c in ((network or {}).get(key) or []):
            with contextlib.suppress(Exception):
                out.add((int(c[0]), int(c[-1])))
    return out


def _edges(rect: list, buildings: list, streets: set) -> dict:
    """**Which edges of an anchor are fronts, and which are streets.** The fabric reset
    round's ruler, replacing a universal proximity count: for each side of the anchor's
    own rectangle, the share of its edge columns that meet a standing building's wall
    within `ANCHOR_EDGE_REACH` straight out (the walk round a market and a lane), how
    many of those buildings have their door on the wall facing the anchor, and whether
    the side is open onto a street instead. A side is **fronted** where at least
    `ANCHOR_EDGE_SHARE` of it meets walls and at least one door faces it."""
    x0, z0, x1, z1 = rect
    out = {}
    for side in ("north", "south", "west", "east"):
        if side in ("north", "south"):
            cols = [(x, z0 if side == "north" else z1) for x in range(x0, x1 + 1)]
            step = (0, -1 if side == "north" else 1)
        else:
            cols = [(x0 if side == "west" else x1, z) for z in range(z0, z1 + 1)]
            step = (-1 if side == "west" else 1, 0)
        met, doors, street = 0, set(), 0
        for (cx, cz) in cols:
            hit = None
            for k in range(1, ANCHOR_EDGE_REACH + 1):
                px, pz = cx + step[0] * k, cz + step[1] * k
                hit = next((b for b in buildings
                            if b[1][0] <= px <= b[1][2] and b[1][1] <= pz <= b[1][3]), None)
                if hit is not None:
                    break
                if (px, pz) in streets:
                    street += 1
                    break
            if hit is None:
                continue
            met += 1
            name, br, door = hit
            if door and len(door) >= 2:
                dx, dz = int(door[0]), int(door[-1])
                face = {"north": br[3], "south": br[1], "west": br[2], "east": br[0]}[side]
                on_face = (abs(dz - face) <= 1 if side in ("north", "south")
                           else abs(dx - face) <= 1)
                if on_face:
                    doors.add(name)
        n = float(len(cols))
        out[side] = {"walls": round(met / n, 3), "doors_facing": sorted(doors),
                     "street": round(street / n, 3),
                     "fronted": bool(met / n >= ANCHOR_EDGE_SHARE and doors),
                     "open_to_street": bool(street / n >= ANCHOR_EDGE_SHARE)}
    return out


def fabric_measures(state: str, sec=None) -> dict:
    """**The street as a person meets it**, measured off the build's own records: the
        fabric reset round's numbers, for a finding to cite and a ledger row to close on.

          * `anchor.fronted_sides` -- of the section's market, how many edges off the street
            are fronted (`_edges`), and `anchor.street_sides` how many open onto a street;
          * `courts.min_side_median` -- the median, over the courtyard houses that stood, of
            the shorter side of the court each emits; `courts.share_7` the share of them whose
            court is at least 7 across both ways; `courts.with_court` how many emit one;
          * `joins.touching_share` -- of the lots the plan joins by a party wall, the share
            whose built walls touch.
        
    """
    rows = [r for r in _rows(_load(state, "parts.json")) if r.get("stood")]
    if sec and len(sec) == 4:
        x0, z0, x1, z1 = sec
        def _in(r):
            b = _building_rect(r)
            return bool(b) and x0 <= (b[0] + b[2]) / 2 <= x1 and z0 <= (b[1] + b[3]) / 2 <= z1
        rows = [r for r in rows if _in(r)]
    out: dict = {}
    network = _load(state, "network.json")
    streets = _street_cells(state, network)
    buildings = [(str(r.get("part")), _building_rect(r), r.get("door")) for r in rows
                 if r.get("kind", "plot") == "plot" and _building_rect(r)]
    markets = [r for r in rows if str(r.get("type")) in ANCHOR_TYPES]
    if markets:
        fp = _footprint(markets[0])
        if fp:
            e = _edges(fp, buildings, streets)
            out["anchor.fronted_sides"] = sum(1 for v in e.values()
                                              if v["fronted"] and not v["open_to_street"])
            out["anchor.street_sides"] = sum(1 for v in e.values() if v["open_to_street"])
    sides = []
    with_court = 0
    for r in rows:
        if str(r.get("type")) != "courtyard_house":
            continue
        em = r.get("emitted") or {}
        rects = em.get("rects") or {}
        c = rects.get("court") or rects.get("courtyard") or rects.get("yard")
        if (em.get("features") or {}).get("courtyard"):
            with_court += 1
        if isinstance(c, (list, tuple)) and len(c) == 4:
            sides.append(min(abs(c[2] - c[0]) + 1, abs(c[3] - c[1]) + 1))
    if sides:
        sides.sort()
        out["courts.min_side_median"] = sides[len(sides) // 2]
        out["courts.share_7"] = round(sum(1 for v in sides if v >= 7) / len(sides), 3)
        out["courts.with_court"] = with_court
    with contextlib.suppress(Exception):
        out.update(ground_measures(state, sec, rows))
    # **doors a street is above** (the design resolution round, the independent reader's
    # r3): a built part in the section whose reserved doorstep stands more than a step
    # off the lane it is entered from -- a shop door at 65 in a trench under a street at
    # 69
    with contextlib.suppress(Exception):
        names = {str(r.get("part")) for r in rows}
        sunk = [t for t in ((network or {}).get("thresholds") or [])
                if str(t.get("id")) in names and t.get("step") is not None
                and int(t["step"]) > 1]
        out["doors.off_street"] = len(sunk)
    return out


#: A ridge is a column standing this far over both its neighbours along one axis.
RIDGE_HIGH = 3


def ground_measures(state: str, sec, rows: list) -> dict:
    """**The ground between the buildings, as a person crosses it** (the design
    resolution round): `ground.step_max`, the largest difference of level between two
    adjacent district pieces the section holds (the parent's decision, off
    `plan.place.json`); and `ground.ridges`, the columns of open ground in the section
    standing `RIDGE_HIGH` or more over both neighbours along x or z on the built world --
    the fins the fabric reset's reader counted -- with every stood part's footprint and a
    column round it left out, since a wall is not a ridge."""
    import numpy as np
    from . import offline, observe
    out: dict = {}
    place = _load(state, "plan.place.json") or {}
    ds = [d for d in place.get("districts") or [] if d.get("x1") is not None
          and d.get("level") is not None and int(d.get("structures") or 0) > 0]
    if sec and len(sec) == 4:
        x0, z0, x1, z1 = sec
        ds = [d for d in ds if d["x0"] <= x1 and d["x1"] >= x0 and d["z0"] <= z1
              and d["z1"] >= z0]
    p = os.path.join(state, "world_built.npz")
    vol = h = _w = None
    if os.path.exists(p):
        vol = offline.load_volume(p)
        h, _w = observe.ground_heights(vol)
    foot = set()
    for r in rows:
        b = _building_rect(r) or _footprint(r)
        if b:
            foot |= {(x, z) for x in range(b[0] - 1, b[2] + 2) for z in range(b[1] - 1, b[3] + 2)}

    def built_level(d):
        """The median ground of a district as built: its open columns (not a building
        or the column round one) on the assembled world -- the level a person walks at
        between the buildings, whatever the plan said the terrace was."""
        if h is None:
            return None
        vals = [int(h[x - vol.x0, z - vol.z0])
                for x in range(int(d["x0"]), int(d["x1"]) + 1, 2)
                for z in range(int(d["z0"]), int(d["z1"]) + 1, 2)
                if (x, z) not in foot and 0 <= x - vol.x0 < h.shape[0]
                and 0 <= z - vol.z0 < h.shape[1]]
        return int(np.median(vals)) if vals else None
    steps, planned = [], []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            gx = max(a["x0"], b["x0"]) - min(a["x1"], b["x1"]) - 1
            gz = max(a["z0"], b["z0"]) - min(a["z1"], b["z1"]) - 1
            if (gx <= 8 and gz < 0) or (gz <= 8 and gx < 0):
                planned.append(abs(int(a["level"]) - int(b["level"])))
                la, lb = built_level(a), built_level(b)
                if la is not None and lb is not None:
                    steps.append(abs(la - lb))
    # **the step as built** (the median open ground of each piece), with the plan's
    # levels beside it: a relevel that the terraces never cut is not an improvement
    if steps:
        out["ground.step_max"] = max(steps)
    if planned:
        out["ground.step_max_planned"] = max(planned)
    if sec and len(sec) == 4 and vol is not None:
        x0, z0, x1, z1 = sec
        built = np.zeros((x1 - x0 + 1, z1 - z0 + 1), bool)
        for r in rows:
            b = _building_rect(r) or _footprint(r)
            if not b:
                continue
            a0, a1 = max(x0, b[0] - 1), min(x1, b[2] + 1)
            c0, c1 = max(z0, b[1] - 1), min(z1, b[3] + 1)
            if a1 >= a0 and c1 >= c0:
                built[a0 - x0:a1 - x0 + 1, c0 - z0:c1 - z0 + 1] = True
        hh = h[x0 - vol.x0:x1 - vol.x0 + 1, z0 - vol.z0:z1 - vol.z0 + 1].astype(int)
        c = hh[1:-1, 1:-1]
        ax = np.minimum(c - hh[:-2, 1:-1], c - hh[2:, 1:-1])
        az = np.minimum(c - hh[1:-1, :-2], c - hh[1:-1, 2:])
        ridge = (np.maximum(ax, az) >= RIDGE_HIGH) & ~built[1:-1, 1:-1] \
            & ~built[:-2, 1:-1] & ~built[2:, 1:-1] & ~built[1:-1, :-2] & ~built[1:-1, 2:]
        out["ground.ridges"] = int(ridge.sum())
        # ...**and the holes**, the mirror of a ridge (the independent reader's n1, a
        # slot 9-16 deep beside the ring street): open ground standing `RIDGE_HIGH` or
        # more under both neighbours along one axis, and water standing inside a piece
        # that carries buildings (n2, a pond six below the lane piece it was laid in)
        bx = np.minimum(hh[:-2, 1:-1] - c, hh[2:, 1:-1] - c)
        bz = np.minimum(hh[1:-1, :-2] - c, hh[1:-1, 2:] - c)
        pit = (np.maximum(bx, bz) >= RIDGE_HIGH) & ~built[1:-1, 1:-1]
        out["ground.pits"] = int(pit.sum())
        wet = _w[x0 - vol.x0:x1 - vol.x0 + 1, z0 - vol.z0:z1 - vol.z0 + 1]
        in_piece = np.zeros_like(wet, dtype=bool)
        for d in ds:
            a0, a1 = max(x0, int(d["x0"])), min(x1, int(d["x1"]))
            c0, c1 = max(z0, int(d["z0"])), min(z1, int(d["z1"]))
            if a1 >= a0 and c1 >= c0:
                in_piece[a0 - x0:a1 - x0 + 1, c0 - z0:c1 - z0 + 1] = True
        out["ground.water_in_pieces"] = int((wet & in_piece).sum())
    return out


def _walls_note(edges: dict) -> str:
    return ", ".join(f"{k} {v['walls']:.0%}" for k, v in edges.items())


def _anchor(rows: list, usable: dict, plan: dict | None, streets=None) -> dict:
    # **A block's court is not the quarter's anchor.** The quarter design round: courts
    # are laid as `plaza` tiles, so every composed court counted as an anchor subject
    # and the relationship passed or failed on paving that belongs to a block of houses.
    # A court is the part a block's ranges claim to enclose (`enclosure`/`court_site`).
    courts = set()
    with contextlib.suppress(Exception):
        from . import pipeline
        courts = {str(p.get("name")) for p in pipeline.plan_parts(plan or {})
                  if p.get("enclosure") or p.get("court_site")}
    anchors = [r for r in rows if str(r.get("type")) in ANCHOR_TYPES
               and str(r.get("part")) not in courts]
    stood = [r for r in anchors if r.get("stood")]
    if not anchors:
        return {"status": "failed",
                "how": "the section built no market, square or plaza: the programme's "
                       "anchor is not in it",
                "measured": {"anchors": 0}, "from": ["parts.json"]}
    # **...and a market is integrated by its neighbours, not by its stalls.** The
    # quarter design round, and the audit's "a market's reachable equipment does not
    # establish frontage, approaches or its place among homes": this verdict was the
    # equipment predicate alone, and the delivered market passed with its nearest
    # standing building eighteen columns away. Asked now of the built world as well:
    # which standing buildings of the section stand within `ANCHOR_NEIGHBOUR_REACH` of
    # the anchor's own rectangle (a street and a lot's clearance), and on how many of
    # its four sides. Belonging is at least `ANCHOR_NEIGHBOURS` of them on at least
    # `ANCHOR_SIDES` sides -- an anchor with a building across its street and one along
    # its frontage -- reported with the gaps, and required beside the equipment.
    buildings = [(str(r.get("part")), _building_rect(r)) for r in rows
                 if r.get("stood") and r.get("kind", "plot") == "plot"
                 and _building_rect(r)]
    got = []
    for r in stood:
        em = r.get("emitted") or {}
        eq = usable.get((r.get("part"), "equipment_reachable"))
        rect = _footprint(r) or []
        near, sides = [], set()
        if len(rect) == 4:
            x0, z0, x1, z1 = rect
            for name, b in buildings:
                g = _gap(rect, b)
                if g > ANCHOR_NEIGHBOUR_REACH:
                    continue
                near.append({"part": name, "gap": int(g)})
                if b[3] < z0:
                    sides.add("north")
                if b[1] > z1:
                    sides.add("south")
                if b[2] < x0:
                    sides.add("west")
                if b[0] > x1:
                    sides.add("east")
        nearest = min((_gap(rect, b) for _n, b in buildings), default=None) \
            if len(rect) == 4 else None
        proximity = len(near) >= ANCHOR_NEIGHBOURS and len(sides) >= ANCHOR_SIDES
        # **the chosen space and its active edges**, not a proximity count (the fabric
        # reset round): the edges off the street must be fronts facing the anchor
        doors_of = {str(q.get("part")): q.get("door") for q in rows}
        edges = (_edges(rect, [(n_, b_, doors_of.get(n_)) for n_, b_ in buildings],
                        set(streets or ())) if len(rect) == 4 else {})
        fronted = [k for k, v in edges.items() if v["fronted"]]
        open_ = [k for k, v in edges.items() if v["open_to_street"] and not v["fronted"]]
        others = [k for k in edges if k not in open_]
        want = min(ANCHOR_SIDES, len(others))
        belongs = bool(open_) and len([k for k in fronted if k in others]) >= max(1, want)
        got.append({"part": r.get("part"), "type": r.get("type"),
                    "edges": edges, "fronted_sides": fronted, "street_sides": open_,
                    "proximity_rule": {"neighbours": len(near), "sides": sorted(sides),
                                       "holds": proximity,
                                       "note": "the quarter design round's ruler, kept "
                                               "as a diagnostic beside the edge reading"},
                    "floor_columns": _rect_area(rect),
                    "features": em.get("features"),
                    "feature_method": em.get("features_method"),
                    "equipment_reachable": {"holds": (eq or {}).get("holds"),
                                            "method": (eq or {}).get("method"),
                                            "why": str((eq or {}).get("why"))[:200]},
                    "neighbours": sorted(near, key=lambda q: q["gap"]),
                    "neighbour_sides": sorted(sides),
                    "nearest_building_gap": nearest,
                    "belongs": belongs,
                    "affirmative": bool(_holds(eq) and belongs)})
    ok = [g for g in got if g["affirmative"]]
    return {"status": "demonstrated" if ok else "failed" if stood else "failed",
            "how": (f"{len(stood)} of {len(anchors)} anchor(s) stood; "
                    f"{len(ok)} carry an affirmative final-world predicate on their "
                    f"equipment being reachable, open onto a street on at least one side "
                    f"and fronted on {ANCHOR_SIDES} of their other sides (or all of them "
                    f"where fewer remain) by walls within {ANCHOR_EDGE_REACH} column(s) "
                    f"over {ANCHOR_EDGE_SHARE:.0%} of the edge with a door facing it: "
                    + "; ".join(f"{g['part']}: street on {g['street_sides'] or 'no side'}, "
                                f"fronted on {g['fronted_sides'] or 'no side'} "
                                f"(walls {_walls_note(g['edges'])}), "
                                f"equipment "
                                f"{'held' if g['equipment_reachable']['holds'] else 'not held'}"
                                for g in got)),
            "measured": {"anchors": len(anchors), "stood": len(stood), "anchors_": got,
                         "bars": {"edge_reach": ANCHOR_EDGE_REACH,
                                  "edge_share": ANCHOR_EDGE_SHARE,
                                  "sides": ANCHOR_SIDES,
                                  "proximity_reach": ANCHOR_NEIGHBOUR_REACH,
                                  "proximity_neighbours": ANCHOR_NEIGHBOURS}},
            "from": ["parts.json", "usable.json"]}


#: How far a standing building may be from a market and still be its neighbour: a street
#: (`PLOT_LANE` 5) and the clearance a lot keeps (3), in columns.
ANCHOR_NEIGHBOUR_REACH = 8
#: **The fabric reset round's anchor ruler.** How far out from an anchor's edge a wall
#: may stand and still front it: the walk round a market and a lane
#: (`streetplan.MARKET_WALK` 2 + `placeplan.PLOT_LANE` 5 - 1).
ANCHOR_EDGE_REACH = 6
#: ...and how much of the edge those walls must cover for it to be a front.
ANCHOR_EDGE_SHARE = 0.5
#: A market among houses has at least this many of them, on at least this many sides.
ANCHOR_NEIGHBOURS = 3
ANCHOR_SIDES = 2


def _court_area(row: dict, answer: dict | None) -> int:
    """**How big the court this part actually built is**, in columns, or 0.

        The predicate's own measurement first (`usable.court_accessible` reports each court
        it walked, with its cell count), then the rectangle the type emitted under one of
        `COURT_RECTS`. Reported per part because "every predicate holds" and "every court is
        3x3" are the same record read two ways, and only one of them is a court a person can
        use. The predicate is not weakened to say so: a 2x2 light well that is entered, open
        and enclosed still holds, and now it holds at four columns where a reader can see it.
        
    """
    cells = 0
    for c in ((answer or {}).get("evidence") or {}).get("courts") or []:
        try:
            cells = max(cells, int(c.get("cells") or 0))
        except (TypeError, ValueError):
            continue
    if cells:
        return int(cells)
    rects = (row.get("emitted") or {}).get("rects") or {}
    return max([_rect_area(rects.get(k)) for k in COURT_RECTS] + [0])


def _in_section(d: dict, sec) -> bool:
    """Does this district's rectangle meet the section being measured?

        **A section is a section, and a court owed three rings away is not failed here.**
        The neighbourhood delivery round, and the audit's own words: "the new court
        denominator includes all 18 court-owing districts of the city in a section record:
        `_districts` collects the whole plan and `_courts` gets no section boundary. This
        should not force a whole-city build. Required missing courts inside the evaluated
        neighbourhood must remain owed."

        So the denominator is the court-owing districts the section's rectangle actually
        reaches, and a district outside it is listed as out of scope rather than counted and
        failed. Everything inside it stays owed exactly as before.
        
    """
    if not sec or len(sec) != 4 or d.get("x1") is None:
        return True
    sx0, sz0, sx1, sz1 = (min(sec[0], sec[2]), min(sec[1], sec[3]),
                          max(sec[0], sec[2]), max(sec[1], sec[3]))
    dx0, dz0 = min(int(d["x0"]), int(d["x1"])), min(int(d["z0"]), int(d["z1"]))
    dx1, dz1 = max(int(d["x0"]), int(d["x1"])), max(int(d["z0"]), int(d["z1"]))
    return not (dx1 < sx0 or dx0 > sx1 or dz1 < sz0 or dz0 > sz1)


def _open_ground(state: str | None, plan: dict | None) -> set:
    """**Open ground is not a court** (the fabric reset round): the parts a
    street-composed district lays as the ground its streets and lots leave -- gardens and
    yards the plan names as open ground (`open_ground` on the leaf, or the district's
    `open` quarter) -- claim no enclosure and belong to nobody's ranges, so they are not
    subjects of the courts relationship. A court a block's ranges enclose, a yard behind
    a house and a courtyard house's own court still are."""
    out = set()
    try:
        from . import pipeline
        for leaf in pipeline.plan_parts(plan or {}):
            if leaf.get("kind", "plot") != "area":
                continue
            name = str(leaf.get("name") or "")
            if leaf.get("open_ground") or any(str(q).endswith("_open")
                                               for q in (leaf.get("in") or [])):
                out.add(name)
    except Exception:                                  # noqa: BLE001 -- nothing named
        return set()
    return out


def _courts(rows: list, usable: dict, districts: list, sec=None, open_ground=None) -> dict:
    """Every court-owing subject in the section, and what the world shows for each.

        **The denominator is the subjects, not the survivors.** The neighbourhood review's
        fourth finding, measured on the delivered candidate: this collected the parts that
        **stood** and whose type is a court type, so a failed courtyard house, a part the
        plan never placed and an answer the library could not give all left the set silently
        -- and four standing courts out of nine attempted subjects read as `demonstrated`.
        Three kinds of subject are in scope here and every one of them is counted whether or
        not anything stood on it:

          * a **part whose type is a court** (`COURT_TYPES`) or that claimed a `courtyard`
            feature, or that `demand.required_by_part` bound the token to -- including the
            ones that did not stand;
          * a **district that adopted a courtyard-block form** (`demand.court_obligation`):
            a character or arrangement with a `courtyard_share`, the registered character of
            its density, or a compile record that laid courts. This is the subject that could
            not previously exist: the obligation is the *form's*, no leaf of the district need
            be a courtyard-house type, and the crowded district asked for twenty-four
            courtyard blocks while publishing nothing;
          * anything already carrying the token on `emitted.required`, which is the
            production binding and may reach a part this function would not have guessed.

        `asked` is the subjects an affirmative-or-negative predicate actually answered for;
        `unsupported` is the answers the library could not give; `omitted` is every subject
        with no evidence of any kind, by name. `subjects > asked` is a **failure** and not an
        `unmeasured`: a court nobody could ask about is a court that was not demonstrated,
        which is the whole of what this relationship claims.
        
    """
    from . import demand as demand_mod

    def token_bound(r: dict) -> bool:
        req = (r.get("emitted") or {}).get("required") or {}
        if isinstance(req, dict):
            return any(str(t) == "courtyard" for t in req)
        return any(str(t) == "courtyard" for t in (req or ()))

    _og = set(open_ground or ())
    courts = [r for r in rows
              if str(r.get("part")) not in _og]
    courts = [r for r in courts
              if str(r.get("type")) in COURT_TYPES
              or ((r.get("emitted") or {}).get("features") or {}).get("courtyard")
              is not None
              or token_bound(r)]
    stood = [r for r in courts if r.get("stood")]
    owing, out_of_section = {}, []
    for d in districts or []:
        got = demand_mod.court_obligation(d)
        if not (got and d.get("name")):
            continue
        if _in_section(d, sec):
            owing[str(d["name"])] = got
        else:
            out_of_section.append(str(d["name"]))
    per_part = []
    for r in courts:
        acc = usable.get((r.get("part"), "court_accessible"))
        # **...and whether anything stands round it.** The block design round, and the
        # audit's fourth cause read back on this reader: the gate this feeds says "the
        # courts are entered, open and **enclosed** on the assembled world" and the only
        # predicate it had was `court_accessible`, which asks whether a court is paved,
        # open to the sky and reachable. Two paved tiles standing in open cobble
        # answered it and the section record called them demonstrated.
        # `usable.court_enclosed` asks the other half -- are there buildings on all four
        # sides of this court, on the blocks -- and where a court claims an enclosure,
        # holding it is part of being a court. A part that claims none (a market floor,
        # a verge) is not asked.
        enc = usable.get((r.get("part"), "court_enclosed"))
        claims_enclosure = bool(enc) and str(enc.get("method") or "") not in (
            "", "inapplicable")
        em = r.get("emitted") or {}
        per_part.append({
            "part": r.get("part"), "type": r.get("type"),
            "district": r.get("_district"),
            "subject": "part", "stood": bool(r.get("stood")),
            "court_columns": _court_area(r, acc),
            "court_accessible": {"holds": (acc or {}).get("holds"),
                                 "method": (acc or {}).get("method"),
                                 "why": str((acc or {}).get("why"))[:200]},
            "court_enclosed": ({"holds": enc.get("holds"),
                                "method": enc.get("method"),
                                "gaps": (enc.get("evidence") or {}).get("gaps"),
                                "why": str(enc.get("why"))[:200]} if enc else None),
            "claims_enclosure": claims_enclosure,
            "claimed": (em.get("features") or {}).get("courtyard"),
            "omitted_by_type": "courtyard" in list(em.get("omitted") or ()),
            "affirmative": bool(_holds(acc)
                                and (not claims_enclosure or _holds(enc)))})
    for name, got in sorted(owing.items()):
        acc = usable.get((name, "court_accessible"))
        # a district's court is demonstrated by the courts standing **in** it: a leaf
        # whose name is prefixed by the district's, which is the same join `_features`
        # and `_side_measures` make.
        mine = [g for g in per_part if str(g["part"] or "").startswith(name)
                or g.get("district") == name]
        held = [g for g in mine if g["affirmative"]]
        if acc:
            answer = {"holds": acc.get("holds"), "method": acc.get("method"),
                      "why": str(acc.get("why"))[:200]}
        elif held:
            # the district's own court is demonstrated by the courts standing in it, and
            # the evidence is those parts' own observed answers rather than a new one
            answer = {"holds": True, "method": "inferred",
                      "why": (f"{len(held)} of {len(mine)} court(s) in this district "
                              f"hold on the assembled world: "
                              f"{', '.join(str(g['part']) for g in held[:4])}")}
        elif mine:
            answer = {"holds": False, "method": "inferred",
                      "why": (f"this district adopted a courtyard block "
                              f"({got.get('from')}) and none of the {len(mine)} "
                              f"court subject(s) in it holds on the assembled world")}
        else:
            # **Nothing to ask.** The district owes a court and no part of it is one:
            # not `unsupported` (the library can answer this question), not a pass, and
            # not silence. It is an omitted subject and the relationship fails on it.
            answer = {"holds": None, "method": None,
                      "why": (f"this district adopted a courtyard block "
                              f"({got.get('from')}) and no part of it is a court at "
                              f"all, so nothing was asked")}
        per_part.append({
            "part": name, "type": None, "subject": "district",
            "stood": bool(mine),
            "court_columns": max([g["court_columns"] for g in held] + [0]),
            "courts_in_it": len(mine), "courts_held_in_it": len(held),
            "courtyard_share": got.get("share"), "courts_laid": got.get("courts"),
            "obligation_from": got.get("from"),
            "court_accessible": answer,
            "affirmative": answer["holds"] is True})
    subjects = len(per_part)
    # **A measurement gap and a question that does not arise are two different facts.**
    # The neighbourhood delivery round: `usable` answered both `unsupported`, so a part
    # that has no court (nothing owed, nothing missing) and a part whose court could not
    # be read off the volume (a gap in the instrument) were one number. Neither
    # establishes anything -- a measurement exception never becomes an affirmative
    # outcome -- and only one of them is owed.
    unsupported = [g for g in per_part
                   if str(g["court_accessible"].get("method")) == "unsupported"]
    inapplicable = [g for g in per_part
                    if str(g["court_accessible"].get("method")) == "inapplicable"]
    # **Asked** is a predicate that ran and said something. A subject that did not
    # stand, or whose predicate the library cannot support, was not asked -- and is
    # still a subject, which is the correction.
    asked_rows = [g for g in per_part
                  if g["court_accessible"].get("holds") is not None
                  and str(g["court_accessible"].get("method")) not in
                  ("unsupported", "inapplicable", "declared", "None", "")]
    asked_names = {g["part"] for g in asked_rows}
    omitted = [g["part"] for g in per_part if g["part"] not in asked_names]
    ok = [g for g in asked_rows if g["affirmative"]]
    owed = [g for g in asked_rows if not g["affirmative"]]
    asked_share = None
    for got in owing.values():
        if got.get("share"):
            asked_share = max(asked_share or 0.0, float(got["share"]))
    # **Asked, and answered.** A court whose predicate answered `unsupported` claims no
    # court to look at and is not evidence either way; a court that was asked and did
    # not answer affirmatively is owed. So the bar is: at least one court was asked, and
    # every court that was asked holds. One affirmative answer out of eight standing
    # courts is not "the courts are courts", which is what a `len(ok) > 0` test would
    # have called it -- **and every subject has to have been asked**, which is the
    # clause the review found missing and the reason a `failed` can now come out of a
    # set in which nothing said no.
    status = ("failed" if subjects > len(asked_rows) or owed
              else "demonstrated" if asked_rows else "unmeasured")
    areas = sorted({g["court_columns"] for g in ok if g["court_columns"]})
    return {"status": status,
            "how": (f"{subjects} court-owing subject(s) in scope "
                    f"({len(courts)} part(s), {len(owing)} district(s) whose adopted "
                    f"form owes a court); {len(stood)} of {len(courts)} part(s) stood; "
                    f"{len(asked_rows)} subject(s) were asked whether their court is "
                    f"reachable and open on the assembled world and {len(ok)} hold"
                    + (f", {len(owed)} owed" if owed else "")
                    + (f"; {len(omitted)} never asked ({', '.join(str(x) for x in omitted[:4])})"
                       if omitted else "")
                    + (f"; the courts that hold measure {areas} column(s)" if areas else "")
                    + (f"; the largest courtyard share any district in the section asked "
                       f"for is {asked_share}" if asked_share else "")
                    + (f"; {len(out_of_section)} court-owing district(s) of this place "
                       f"lie outside the section and stay owed elsewhere rather than "
                       f"being failed here" if out_of_section else "")),
            "measured": {"subjects": subjects, "courts": len(courts), "stood": len(stood),
                         "districts_owing": sorted(owing),
                         "courts_": per_part, "per_part": per_part,
                         "asked": len(asked_rows), "held": len(ok),
                         "unsupported": len(unsupported),
                         "inapplicable": len(inapplicable),
                         "omitted": omitted,
                         "owed": [g["part"] for g in owed],
                         "court_columns_held": areas,
                         "largest_courtyard_share_asked": asked_share,
                         # named, not counted: the obligation is theirs and is not this
                         # section's to demonstrate or to fail
                         "out_of_section": sorted(out_of_section),
                         "section_rect": [int(v) for v in sec] if sec else None},
            "from": ["parts.json", "usable.json", "plan.json", "plan.place.json"]}


#: How far outside the section's own rectangle a traversal may wander. A route from a
#: gate to an anchor keeps to the section; the margin is the lane and its verges, so a
#: lane that runs a few columns outside the registered line is not a broken route.
WALK_MARGIN = 12


def _walk_section(state: str, sec, thresholds: list) -> dict | None:
    """**The route, walked on the final assembled blocks.**

        The neighbourhood delivery round, and the audit's first evidence limit: "`section._route`
        reads a planned network check and finds gate/anchor threshold names. It does not
        trace the registered route through final blocks. The problem is what the route
        verdict measures."

        `circulate.walk_check` re-derives reachability over the **planned lane graph** under
        the movement rules, which is the right check at planning time and answers a different
        question afterwards: it cannot see a pad laid over a lane, a wall closed across a
        threshold or a terrace cut under one. This floods the built volume itself, from the
        first threshold in scope, under the same walk model every other physical predicate in
        this project uses (`observe.Nav`, no jumping), bounded to the section and its margin,
        and asks which of the other thresholds a person standing at the first one can
        actually reach.

        Returns None where there is no built volume to read -- which is honestly unmeasured
        and is not a pass.
        
    """
    import os as _os
    from . import observe, offline
    art = None
    for name in ("world_built.npz",):
        p = _os.path.join(state, name)
        if _os.path.exists(p):
            art = p
            break
    if art is None or not thresholds:
        return None
    vol = offline.load_volume(art)
    nav = observe.Nav(vol)
    at = {}
    for t in thresholds:
        s0 = nav.stance_near(int(t["x"]), int(t["z"]), int(t["y"]) + 1, tol=1)
        at[str(t["id"])] = None if s0 is None else (int(t["x"]), int(t["z"]), int(s0))
    stood = {k: v for k, v in at.items() if v is not None}
    if not stood:
        return {"artifact": _os.path.basename(art), "artifact_path": art,
                "reached": [], "unreached": sorted(at),
                "no_stance": sorted(at), "from": None,
                "why": "no threshold of this section has a stance on the assembled "
                       "world: nothing could be walked from"}
    bounds = None
    if sec and len(sec) == 4:
        x0, z0, x1, z1 = (min(sec[0], sec[2]), min(sec[1], sec[3]),
                          max(sec[0], sec[2]), max(sec[1], sec[3]))
        # **The box a walk is bounded to has to contain the doorsteps it is asked
        # about.** The block design round, found by an independent reader of the
        # delivered block. A section whose `unit` is the block grows past its registered
        # rectangle -- a block is taken whole or not at all -- so two of this block's
        # doorsteps stood at x=-5724 while the flood was bounded to x <= -5728, four
        # columns outside a box they could never enter. `stance_near`, which builds the
        # threshold list, is not bounded, so they were counted among the 53 and could
        # not be among the reached: **"2 thresholds not walkable" was a measurement of
        # the bounding box** and unbounded the same walk reaches 47 of 47.
        for v in stood.values():
            x0, z0 = min(x0, int(v[0])), min(z0, int(v[1]))
            x1, z1 = max(x1, int(v[0])), max(z1, int(v[1]))
        bounds = (x0 - WALK_MARGIN, z0 - WALK_MARGIN, x1 + WALK_MARGIN, z1 + WALK_MARGIN)
    start = sorted(stood)[0]
    got = nav.flood([stood[start]], max_jumps=0, bounds=bounds)
    reached = sorted(k for k, v in stood.items() if v in got)
    unreached = sorted(set(stood) - set(reached))
    return {"artifact": _os.path.basename(art), "artifact_path": art,
            "from": start, "stances": len(got),
            "thresholds": len(at), "with_stance": len(stood),
            "no_stance": sorted(k for k, v in at.items() if v is None),
            "reached": reached, "unreached": unreached,
            "bounds": list(bounds) if bounds else None,
            "why": (f"walked on the assembled world from `{start}`'s own doorstep under "
                    f"the walk model with no jumping, inside the section, every "
                    f"doorstep it is asked about and "
                    f"{WALK_MARGIN} column(s) round them: {len(got)} stance(s) reachable, "
                    f"{len(reached)} of {len(stood)} threshold(s) among them")}


def _route(rows: list, circulation: dict | None, network: dict | None,
           registered: dict, state: str = "", sec=None) -> dict:
    wc = (circulation or {}).get("walk_check") or {}
    if not wc:
        return {"status": "unmeasured",
                "how": "no walk check was recorded for this build",
                "measured": {}, "from": ["circulation.json"]}
    built = {r.get("part") for r in rows if r.get("stood")}
    thresholds = [t for t in (network or {}).get("thresholds") or []
                  if t.get("id") in built]
    gates = [t for t in thresholds if "gate" in str(t.get("id"))]
    anchors = [t for t in thresholds
               if any(str(t.get("id", "")).endswith(s) or s in str(t.get("id", ""))
                      for s in ANCHOR_TYPES)]
    # a sample, for a reader to look at -- and the true count beside it as
    # `unreachable_n`. This clause formatted `len(unreachable)`, so **every walk failure
    # of twenty stances or more was reported as exactly twenty**. A truncated list is
    # not a measurement; the measurement is beside it, and the prose says which it is
    # quoting.
    unreachable = list(wc.get("unreachable") or [])
    n_unreachable = wc.get("unreachable_n")
    if n_unreachable is None:
        n_unreachable = len(unreachable)
    reached, cells = wc.get("reached"), wc.get("cells")
    one = bool(cells) and reached == cells and not int(n_unreachable or 0)
    # **...over the section's own lanes.** The quarter design round, after its
    # independent reader: the planned graph was asked to be connected across the whole
    # city, so the section failed on 36 stances at z 770..778 -- outside the registered
    # rectangle, and every listed one walkable on the built world. The question this
    # relationship asks is whether the section's route is one; the planned check is re-
    # run on the stored network and counted inside the section and the walk's margin,
    # with the whole-city figure kept beside it.
    within = None
    _rect0 = (sec.get("rect") if isinstance(sec, dict) else sec) if sec else None
    if state and _rect0 and len(_rect0) == 4:
        with contextlib.suppress(Exception):
            r0 = [int(v) for v in _rect0]
            box = (min(r0[0], r0[2]) - WALK_MARGIN, min(r0[1], r0[3]) - WALK_MARGIN,
                   max(r0[0], r0[2]) + WALK_MARGIN, max(r0[1], r0[3]) + WALK_MARGIN)
            from . import circulate
            within = circulate.walk_check(
                circulate.Network.load(os.path.join(state, "network.json")), within=box)
    if within is not None:
        one = not int(within.get("unreachable_within_n") or 0)
    # **...and the verdict is the walk on the final blocks, not the walk on the plan.**
    # The delivery round. The planned check stays on the record beside it -- it is the
    # check that catches a lane the planner could not have built, and it caught several
    # -- but a relationship that says "walkable end to end on the built world" is
    # answered by walking the built world. Where there is no built volume the
    # relationship is `unmeasured` and says so; an absent artifact has never been a pass
    # in this module and is not one here.
    walk = None
    if state:
        try:
            walk = _walk_section(state, sec, thresholds)
        except Exception as e:                   # noqa: BLE001 -- reported, not raised
            walk = {"error": f"{type(e).__name__}: {e}",
                    "why": "the traversal on the assembled world could not be run"}
    if walk is None:
        return {"status": "unmeasured",
                "how": (f"the planned lane graph is {'connected' if one else 'not '
                        'connected'} ({reached} of {cells} stance(s), "
                        f"{int(n_unreachable)} unreachable), and there is no assembled "
                        f"world to walk the section's own route on. A connected planned "
                        f"network is not this relationship"),
                "measured": {"walk_check": wc, "thresholds": len(thresholds),
                             "gates": [t.get("id") for t in gates],
                             "anchors": [t.get("id") for t in anchors],
                             "walked": None},
                "from": ["circulation.json", "network.json", "parts.json"]}
    walked_ok = bool(walk.get("reached")) and not walk.get("unreached") \
        and not walk.get("no_stance") and not walk.get("error")
    walked_gates = [g for g in (walk.get("reached") or [])
                    if any(str(g) == str(t.get("id")) for t in gates)]
    walked_anchors = [a for a in (walk.get("reached") or [])
                      if any(str(a) == str(t.get("id")) for t in anchors)]
    ok = (one and walked_ok and len(gates) >= 1 and len(anchors) >= 1
          and bool(walked_gates) and bool(walked_anchors))
    if within is not None:
        wc = dict(wc, section=within)
    return {"status": "demonstrated" if ok else "failed",
            "how": (f"walked on {walk.get('artifact')}: "
                    f"{len(walk.get('reached') or [])} of {walk.get('with_stance')} "
                    f"built threshold(s) reachable on foot from `{walk.get('from')}` "
                    f"without jumping, {len(walk.get('unreached') or [])} not reachable"
                    + (f", {len(walk.get('no_stance') or [])} with no stance at all"
                       if walk.get("no_stance") else "")
                    + (f" ({', '.join(str(x) for x in (walk.get('unreached') or [])[:4])})"
                       if walk.get("unreached") else "")
                    + f"; {len(walked_gates)} gate(s) and {len(walked_anchors)} anchor(s) "
                      f"among them of the {len(gates)} and {len(anchors)} that stood"
                    + (f"; {walk['error']}" if walk.get("error") else "")
                    + f". The planned lane graph, separately: {reached} of {cells} "
                      f"stance(s), {int(n_unreachable)} unreachable"
                    + (f" ({len(unreachable)} of them listed)"
                       if len(unreachable) < int(n_unreachable) else "")
                    + (f"; inside the section and {WALK_MARGIN} column(s) round it, "
                       f"{within.get('cells_within')} stance(s), "
                       f"{within.get('unreachable_within_n')} unreachable"
                       if within is not None else "")),
            "measured": {"walk_check": wc, "thresholds": len(thresholds),
                         "gates": [t.get("id") for t in gates],
                         "anchors": [t.get("id") for t in anchors],
                         "walked": walk,
                         "walked_gates": walked_gates,
                         "walked_anchors": walked_anchors,
                         "planned_network_connected": bool(one)},
            "from": ["circulation.json", "network.json", "parts.json",
                     str(walk.get("artifact"))]}


def _features(rows: list, usable: dict, districts: list) -> dict:
    """Every obligation of every built part in scope, and what the world shows for it.

        **The production binding, read where production writes it.** The neighbourhood
        round's finding about this function: it collected the *district's* `demand.required`
        tokens, matched them to parts by name prefix, and then decided each one by reading
        `emitted.features` and `emitted.features_method` -- a second, parallel account of
        what a part owes and whether it delivered, running beside the one the build itself
        produces. Two consequences, both of them the kind this project keeps closing:

          * a part whose obligations came from anywhere other than its district's token list
            had none here, however many the build had recorded on it;
          * `emitted.features[tok] is True` with an `inferred` method was a pass, which is
            the type's own claim measured at emission -- while the answer the **assembled
            world** gave the same feature sat one key away in `emitted.usable` and was never
            consulted.

        So this reads `emitted.required` -- the per-part binding `demand.required_by_part`
        answers and `construction.confirm` stamps onto every row it re-reads -- and decides
        each token through `construction.evidence_for`, which is the same helper
        `confirm` computes `emitted.owed` with. One binding, one decider, one answer.

        **Two dimensions, and they are not the same question.** What the obligation is --
        `held` or `owed` -- and what the evidence is: measured, or nothing decided.

          * `held` is an affirmative answer on the assembled world (`observed`) or off
            measured geometry (`inferred`). Nothing else is held;
          * `owed` is every other subject, which is `construction.evidence_for`'s own `owed`
            flag and `OWED_REASONS`' own meaning of `unmeasured`: *"the one that matters
            most: nothing asked"*. `held + owed == subjects`, always;
          * `failed` and `unknown` are `owed` split by what the evidence was -- a predicate
            that ran and said no, against `unsupported`, `declared` or `unmeasured`, where
            nothing decided. `failed + unknown == owed`, and `unknown` is reported so that an
            undecided answer is never read as a measured failure **or** as a pass. Every owed
            row carries its own `reason`, so the distinction survives per subject and not
            only in a total.

        **A required feature with no evidence is owed, not absent.** The composition round's
        rule, and the reason the split above is this way round rather than the other: the rule
        exists to stop an unmeasured requirement reading as a satisfied one, and a part that
        stood and was never asked is the case it was written for. Making "nothing asked" its
        own third status outside `owed` would have let a relationship with every subject
        unasked report something other than a failure, which is the escape one level up.

        A row that carries no `emitted.required` at all is a row `confirm` never re-read;
        the district binding answers for it as it used to, the row says so in `from`, and the
        parts it happened to are named in the measurement.
        
    """
    from . import construction

    by_district = {}
    for d in districts or []:
        dem = d.get("demand") or {}
        for tok in dem.get("required") or ():
            by_district.setdefault(str(d.get("name")), set()).add(str(tok))
    record = {"waves": [{"parts": list(rows)}]}
    held, owed, no_binding, by_part = [], [], [], {}
    for r in rows:
        em = r.get("emitted") if isinstance(r.get("emitted"), dict) else None
        if em is None or not r.get("part"):
            continue
        name = str(r["part"])
        need, source = em.get("required"), "parts.json emitted.required"
        if need is None:
            need = sorted({t for dn, toks in by_district.items()
                           if name.startswith(dn) for t in toks})
            source = "plan.json demand binding (this row carries no emitted.required)"
            if need:
                no_binding.append(name)
        if not need:
            continue
        by_part[name] = sorted(str(t) for t in need)
        for tok in sorted(str(t) for t in need):
            a = construction.evidence_for(record, name, tok)
            row = {"part": name, "feature": tok, "holds": a.get("holds"),
                   "method": a.get("method"), "reason": a.get("reason"),
                   "want": a.get("want"), "read_at": a.get("read_at"),
                   "stood": bool(r.get("stood", r.get("status") == "built")),
                   "evidence": ("measured" if a.get("holds") is not None else "unknown"),
                   "from": source, "why": str(a.get("why"))[:200]}
            (held if not a.get("owed") else owed).append(row)
    # a part that stood but whose predicates could not be decided is owed, not passed
    undecided = [{"part": p, "want": w, "method": a.get("method")}
                 for (p, w), a in usable.items()
                 if a.get("holds") is None
                 and str(a.get("method")) not in ("unsupported", "inapplicable")]
    # nothing required of anything in scope is not a demonstration that requirements
    # survived: there was nothing to survive, and saying so is the honest answer
    subjects = len(held) + len(owed)
    unknown = [o for o in owed if o["evidence"] == "unknown"]
    failed = [o for o in owed if o["evidence"] == "measured"]
    by_reason: dict = {}
    for o in owed:
        by_reason[str(o.get("reason"))] = by_reason.get(str(o.get("reason")), 0) + 1
    status = ("unmeasured" if not subjects
              else "demonstrated" if not owed else "failed")
    return {"status": status,
            "how": (f"{len(by_part)} part(s) in scope carry {subjects} required "
                    f"feature(s) between them: {len(held)} affirmatively held on the "
                    f"assembled world and {len(owed)} owed"
                    + (f" -- {len(failed)} measured and refused, {len(unknown)} with "
                       f"nothing decided ({', '.join(f'{k}: {v}' for k, v in sorted(by_reason.items()))})"
                       if owed else "")
                    + (f"; {len(undecided)} predicate answer(s) undecided and therefore "
                       f"unresolved" if undecided else "")
                    + (f"; {len(no_binding)} part(s) answered from the district binding "
                       f"because construction recorded no per-part obligation on them"
                       if no_binding else "")),
            "measured": {"subjects": subjects, "held": len(held), "owed": len(owed),
                         "failed": len(failed), "unknown": len(unknown),
                         "owed_by_reason": by_reason,
                         "held_": held[:40], "owed_": owed[:40],
                         "unknown_": unknown[:40], "failed_": failed[:40],
                         "undecided": undecided[:40],
                         "required_by_part": by_part,
                         "parts_without_emitted_binding": sorted(no_binding)[:40],
                         "required_by_district": {k: sorted(v)
                                                  for k, v in by_district.items()}},
            "from": ["parts.json emitted.required (demand.required_by_part, stamped by "
                     "construction.confirm)",
                     "construction.evidence_for over parts.json emitted.usable",
                     "plan.json demand binding, for rows construction never re-read"]}


def _functions(state: str, rows: list, plan: dict | None, reg: dict, sec) -> dict:
    """**The homes and shops that work, subject by subject, on the assembled world.**
        The design resolution round.

        A home is a part of a dwelling type in the section; it **works** when it stood, its
        door is reached on foot on the assembled world (`usable.entrance_connected`,
        observed), and -- for a type that publishes a form plan -- the form its use owes is
        measured on the blocks (`ethoslm.formplan.measure_row`): a courtyard house's ranges clear
        to their room depths round a court of the asked size, a shop house standing the storeys
        its use owes. A doorway record is not a standing building, and a court widened by
        thinning its rooms to corridors is not a working home. A shop is a part carrying a
        trade. Every subject is listed, so a revision that removes a working home can be told
        from one that replaces it (`pipeline.promote`).
        
    """
    from . import formplan, offline
    from .pipeline.stages_plan import load_type
    sides = dict((reg or {}).get("sides") or {})
    leaves = {}
    with contextlib.suppress(Exception):
        from . import pipeline as _pl
        for leaf in _pl.plan_parts(plan or {}):
            leaves[str(leaf.get("name"))] = leaf
    vol = None
    for f in ("world_built.npz", "world_finished.npz"):
        pth = os.path.join(state, f)
        if os.path.exists(pth):
            with contextlib.suppress(Exception):
                vol = offline.load_volume(pth)
                break
    decl_fn: dict = {}

    def func_of(t):
        if t not in decl_fn:
            pth = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "..", "types", f"{t}.py")
            fn = None
            with contextlib.suppress(Exception):
                fn = load_type(os.path.abspath(pth)).get("function")
            decl_fn[t] = fn
        return decl_fn[t]
    homes, shops = [], []
    for r in rows:
        t = str(r.get("type") or "")
        rect = _building_rect(r) or _footprint(r)
        if not rect or not t:
            continue
        if sec and any(sec) and not _clip_area(rect, sec):
            continue
        cx, cz = _centre(rect)
        if sec and any(sec) and not (sec[0] <= cx <= sec[2] and sec[1] <= cz <= sec[3]):
            continue
        params = dict(r.get("params") or {})
        is_home = func_of(t) == "dwelling"
        is_shop = "trade" in params
        if not (is_home or is_shop):
            continue
        side = next((k for k, pref in sides.items() if _of(r, str(pref))), None)
        em = r.get("emitted") or {}
        stood = bool(r.get("stood")) and int(em.get("storeys") or 0) >= 1
        ent = ((em.get("usable") or {}).get("entrance_connected") or {})
        entered = ent.get("holds") is True
        form_ok, form_why, got = None, None, None
        leaf = leaves.get(str(r.get("part"))) or {}
        if stood and vol is not None and formplan.plan_fn(t) is not None:
            with contextlib.suppress(Exception):
                got = formplan.measure_row(vol, r)
            owed = dict(params)
            if leaf.get("form"):
                owed.update((leaf.get("form") or {}).get("params") or {})
            form_ok, form_why = formplan.row_meets(t, got, owed)
        working = bool(stood and entered and form_ok is not False)
        row = {"part": str(r.get("part")), "type": t, "side": side, "stood": stood,
               "entered": entered, "form": form_ok, "form_why": form_why,
               "owes_form": bool(leaf.get("form")), "working": working}
        if is_home:
            homes.append(row)
        if is_shop:
            shops.append(row)
    by_side = {}
    for h in homes:
        k = str(h["side"])
        by_side.setdefault(k, {"homes": 0, "working": 0})
        by_side[k]["homes"] += 1
        by_side[k]["working"] += int(h["working"])
    work_h = sum(1 for h in homes if h["working"])
    work_s = sum(1 for h in shops if h["working"])
    bad = [h for h in homes + shops if not h["working"]]
    status = ("unmeasured" if not homes else "demonstrated" if not bad else "failed")
    return {"status": status,
            "how": (f"{len(homes)} home(s) in the section, {work_h} working (stood, "
                    f"entered on foot, and the form it owes measured on the blocks); "
                    f"{len(shops)} shop(s), {work_s} working"
                    + (f"; not working: {', '.join(h['part'] for h in bad[:8])}"
                       if bad else "")
                    + ("" if vol is not None else "; no built world to measure forms on")),
            "measured": {"homes": len(homes), "homes_working": work_h,
                         "homes_entered": sum(1 for h in homes
                                              if h["stood"] and h["entered"]),
                         "shops": len(shops), "shops_working": work_s,
                         "by_side": by_side,
                         "homes_": homes[:200], "shops_": shops[:200],
                         "forms_measured": sum(1 for h in homes + shops
                                               if h["form"] is not None)},
            "from": ["parts.json rows (stood, emitted.usable.entrance_connected)",
                     "ethoslm.formplan.measure_row over world_built.npz",
                     "plan.json leaves' `form` (what each use owes)"]}


# ------------------------------------------------------------------ the record

def record(state: str, *, registered: dict | None = None, of: str = "") -> dict:
    """The section's own record, measured. Written by `pipeline.stage_section`."""
    parts_record = _load(state, "parts.json")
    plan = _load(state, "plan.json")
    usable = _usable_of(_load(state, "usable.json"))
    circulation = _load(state, "circulation.json")
    network = _load(state, "network.json")
    rows = _rows(parts_record)
    reg = dict(registered or (parts_record or {}).get("sample", {}).get("registered")
               or {})
    sample = (parts_record or {}).get("sample") or {}
    sec = [int(v) for v in (reg.get("rect") or sample.get("rect") or [0, 0, 0, 0])]
    asked = list(reg.get("demonstrate") or sample.get("question") or [])
    runs = list(sample.get("boundary_runs") or [])

    districts = _districts(state, plan)
    # **every built row knows its district** (the fabric reset round): a leaf whose name
    # is unique in the plan is not prefixed with its district, so a join by name missed
    # eight of the street-composed quarter's buildings and the calm side under-counted
    # itself (the independent reader's r1). The plan's own `in` says where each leaf is.
    with contextlib.suppress(Exception):
        from . import pipeline as _pl
        _dn = sorted((str(d.get("name")) for d in districts if d.get("name")),
                     key=len, reverse=True)
        _where = {}
        for leaf in _pl.plan_parts(plan or {}):
            for q in reversed(leaf.get("in") or []):
                hit = next((n for n in _dn if str(q) == n or str(q).startswith(n + "_")),
                           None)
                if hit:
                    _where[str(leaf.get("name"))] = hit
                    break
        for r in rows:
            if str(r.get("part")) in _where:
                r["_district"] = _where[str(r.get("part"))]
    # **which parts claim a block enclosure**, off the plot registry the compiler wrote
    # it into: a court a block's ranges enclose is an area leaf and publishes no court
    # rect, so without this the one measure that would show it reads zero
    enclosure = {str(q.get("label")): q.get("enclosure")
                 for q in (_load(state, "plots.json") or [])
                 if isinstance(q, dict) and q.get("enclosure")}
    got = {"contrast": _contrast(reg, rows, districts, sec, runs,
                                 enclosure=enclosure),
           "anchor": _anchor(rows, usable, plan,
                             streets=_street_cells(state, network)),
           "courts": _courts(rows, usable, districts, sec,
                             open_ground=_open_ground(state, plan)),
           "route": _route(rows, circulation, network, reg, state=state, sec=sec),
           "features": _features(rows, usable, districts),
           "functions": _functions(state, rows, plan, reg, sec)}
    rel = []
    for i, key in enumerate(RELATIONSHIPS):
        r = dict(got[key])
        r["id"] = key
        r["relationship"] = asked[i] if i < len(asked) else key
        rel.append(r)
    types: dict = {}
    for r in rows:
        if r.get("stood"):
            types[str(r.get("type"))] = types.get(str(r.get("type")), 0) + 1
    # **The construction check of the world that stands, not of the one the lanes were
    # laid on.** The block design round, after a second independent reader:
    # `circulation.json`'s `lint` is the check the *circulation* stage ran, over the
    # dry-run volume as the lanes left it -- before a single building. The delivered
    # section published `{E002: 15, E003: 17, E005: 3, E010: 1}` under a heading a
    # reader takes as its own, copied from a check written two and a half hours earlier
    # against a different world, while the build's own check reported three `E005`
    # findings and nothing else. `lint.json` is the build's, `round.json`'s `lint` stage
    # is where the driver records it; the circulation figure is kept beside it under its
    # own name so the two are never read as one.
    lint = _load(state, "lint.json") or {}
    if not (lint.get("counts") or lint.get("errors")):
        rj = _load(state, "round.json") or {}
        got = ((rj.get("results") or {}).get("lint")
               or (rj.get("stages") or {}).get("lint") or rj.get("lint") or {})
        if isinstance(got, dict) and got.get("counts"):
            lint = got
    lint_of_lanes = (circulation or {}).get("lint") or {}
    return {
        # **3**: the neighbourhood delivery round's two corrected rulers -- the court
        # denominator is the section's own (`_in_section`) and the route verdict is a
        # traversal of the assembled world (`_walk_section`) rather than a reading of
        # the planned lane graph. A record written under an earlier version answered
        # different questions; a reader comparing two candidates has to re-derive the
        # older one rather than set the two side by side.
        "record": "section", "version": 3, "of": of or os.path.basename(state),
        "by": "ethoslm.section.record -- measured off this build's own artifacts, not "
              "asserted; see the module docstring for why that distinction is the point",
        "registered": reg,
        "relationships": rel,
        "demonstrated": [r["id"] for r in rel if r["status"] == "demonstrated"],
        "failed": [r["id"] for r in rel if r["status"] == "failed"],
        "unmeasured": [r["id"] for r in rel if r["status"] == "unmeasured"],
        "reported_not_targeted": {
            "parts": sum(1 for r in rows if r.get("stood")),
            "attempted": len(rows),
            "types": types,
            "rect": sec,
            "columns": _rect_area(sec),
            "of_plan": sample.get("of_plan"),
            "cut_out": len(sample.get("cut_out") or []),
            "boundary_columns": sum(int(r.get("columns") or 0) for r in runs),
            "construction_errors": {k: v for k, v in (lint.get("counts") or {}).items()
                                    if str(k).startswith("E")},
            "construction_errors_from": (
                "the construction check over the assembled world this section was built "
                "into" if lint.get("counts") else
                "no construction check of the built world was found in this state "
                "directory; the figure beside it is the circulation stage's, of the "
                "world as the lanes left it and before any building"),
            "lane_stage_errors": {k: v
                                  for k, v in (lint_of_lanes.get("counts") or {}).items()
                                  if str(k).startswith("E")},
            "seconds": sum(float(r.get("seconds") or 0) for r in rows),
        },
        "not_claimed": [
            "whole-city identity and recognition",
            "the tradition the sources name (the library builds `east_asian`)",
            "the palace and its processional sequence",
            "the agrarian belt and the upper ring",
            "whole-city extent, cost and the final live world",
        ],
        "candidate": (parts_record or {}).get("candidate"),
        "built_digest": (_load(state, "usable.json") or {}).get("built_digest"),
        # the street and ground measures a finding cites and the promotion guard reads
        "fabric": _fabric_safe(state, sec),
    }


def _fabric_safe(state, sec) -> dict:
    try:
        return fabric_measures(state, sec)
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        return {"unmeasured": f"{type(e).__name__}: {e}"}
