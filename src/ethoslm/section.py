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
COURT_TYPES = ("court_large", "court_small", "courtyard_house", "yard", "garden")

#: Types whose ground is open by intent rather than by omission.
OPEN_TYPES = ("plaza", "garden", "grove", "yard", "field")

RELATIONSHIPS = ("contrast", "anchor", "courts", "route", "features")


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
        for k in ("scope_columns", "developable_columns", "allocated_columns",
                  "built_columns", "built_from"):
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
            for k in ("x0", "z0", "x1", "z1", "demand", "character", "defines"):
                if node.get(k) is not None:
                    row.setdefault(k, node[k])
        for key in ("districts", "children", "parts", "quarters"):
            if key in node:
                walk(node[key])
    walk((plan or {}).get("parts") or [])
    walk((plan or {}).get("districts") or [])
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

def _side_measures(side: str, prefix: str, rows: list, districts: list,
                   section: list) -> dict:
    """What one side of the boundary actually built, in columns."""
    mine = [r for r in rows
            if r.get("kind", "plot") == "plot" and r.get("stood")
            and str(r.get("part", "")).startswith(prefix)]
    foots = [f for f in (_footprint(r) for r in mine) if f]
    areas = [_rect_area(f) for f in foots]
    gaps = []
    for i, a in enumerate(foots):
        others = [b for j, b in enumerate(foots) if j != i]
        if others:
            gaps.append(min(_gap(a, b) for b in others))
    mine_d = [d for d in (districts or [])
              if str(d.get("name", "")).startswith(prefix)]
    ground = sum(_clip_area([d.get("x0"), d.get("z0"), d.get("x1"), d.get("z1")], section)
                 for d in mine_d
                 if all(d.get(k) is not None for k in ("x0", "z0", "x1", "z1")))
    court = sum(_rect_area(_footprint(r) or [])
                for r in rows
                if str(r.get("part", "")).startswith(prefix) and r.get("stood")
                and str(r.get("type")) in COURT_TYPES)
    storeys = [int((r.get("emitted") or {}).get("storeys") or 0) for r in mine
               if (r.get("emitted") or {}).get("storeys")]
    cz = [_centre(f)[1] for f in foots]
    return {"side": side, "prefix": prefix, "structures": len(mine),
            "built_columns": sum(areas),
            "ground_columns": ground,
            "built_cover": round(sum(areas) / ground, 4) if ground else None,
            "median_footprint": statistics.median(areas) if areas else None,
            "median_neighbour_gap": statistics.median(gaps) if gaps else None,
            "court_columns": court,
            "court_share": round(court / ground, 4) if ground else None,
            "median_storeys": statistics.median(storeys) if storeys else None,
            "centre_z": round(statistics.median(cz), 1) if cz else None,
            "from": "parts.json emitted footprints, plan district rects clipped to the "
                    "section"}


def _contrast(registered: dict, rows: list, districts: list, section: list,
              boundary_runs: list) -> dict:
    sides = dict(registered.get("sides") or {})
    if len(sides) < 2:
        return {"status": "unmeasured",
                "how": "the section registers fewer than two sides of a boundary"}
    got = [_side_measures(k, v, rows, districts, section)
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


def _anchor(rows: list, usable: dict, plan: dict | None) -> dict:
    anchors = [r for r in rows if str(r.get("type")) in ANCHOR_TYPES]
    stood = [r for r in anchors if r.get("stood")]
    if not anchors:
        return {"status": "failed",
                "how": "the section built no market, square or plaza: the programme's "
                       "anchor is not in it",
                "measured": {"anchors": 0}, "from": ["parts.json"]}
    got = []
    for r in stood:
        em = r.get("emitted") or {}
        eq = usable.get((r.get("part"), "equipment_reachable"))
        got.append({"part": r.get("part"), "type": r.get("type"),
                    "floor_columns": _rect_area(_footprint(r) or []),
                    "features": em.get("features"),
                    "feature_method": em.get("features_method"),
                    "equipment_reachable": {"holds": (eq or {}).get("holds"),
                                            "method": (eq or {}).get("method"),
                                            "why": str((eq or {}).get("why"))[:200]},
                    "affirmative": _holds(eq)})
    ok = [g for g in got if g["affirmative"]]
    return {"status": "demonstrated" if ok else "failed" if stood else "failed",
            "how": (f"{len(stood)} of {len(anchors)} anchor(s) stood; "
                    f"{len(ok)} carry an affirmative final-world predicate on their "
                    f"equipment being reachable"
                    + (f" ({', '.join(g['part'] for g in ok)})" if ok else
                       "; no anchor's use is demonstrated on the assembled world")),
            "measured": {"anchors": len(anchors), "stood": len(stood), "anchors_": got},
            "from": ["parts.json", "usable.json"]}


def _courts(rows: list, usable: dict, districts: list) -> dict:
    courts = [r for r in rows
              if str(r.get("type")) in COURT_TYPES or
              ((r.get("emitted") or {}).get("features") or {}).get("courtyard")]
    stood = [r for r in courts if r.get("stood")]
    got = []
    for r in stood:
        acc = usable.get((r.get("part"), "court_accessible"))
        got.append({"part": r.get("part"), "type": r.get("type"),
                    "court_accessible": {"holds": (acc or {}).get("holds"),
                                         "method": (acc or {}).get("method"),
                                         "why": str((acc or {}).get("why"))[:200]},
                    "affirmative": _holds(acc)})
    ok = [g for g in got if g["affirmative"]]
    # **Asked, and answered.** A court whose predicate answered `unsupported` claims no
    # court to look at and is not evidence either way; a court that was asked and did
    # not answer affirmatively is owed. So the bar is: at least one court was asked, and
    # every court that was asked holds. One affirmative answer out of eight standing
    # courts is not "the courts are courts", which is what a `len(ok) > 0` test would
    # have called it.
    asked_rows = [g for g in got
                  if str(g["court_accessible"].get("method")) != "unsupported"]
    owed = [g for g in asked_rows if not g["affirmative"]]
    asked = None
    for d in districts or []:
        ch = d.get("character") or {}
        if ch.get("courtyard_share"):
            asked = max(asked or 0.0, float(ch["courtyard_share"]))
    status = ("demonstrated" if asked_rows and not owed
              else "failed" if owed else "unmeasured")
    return {"status": status,
            "how": (f"{len(stood)} of {len(courts)} court(s) stood; "
                    f"{len(asked_rows)} were asked whether their court is reachable and "
                    f"open on the assembled world and {len(ok)} hold"
                    + (f", {len(owed)} owed" if owed else "")
                    + (f"; the largest courtyard share any district in the section asked "
                       f"for is {asked}" if asked else "")),
            "measured": {"courts": len(courts), "stood": len(stood), "courts_": got,
                         "asked": len(asked_rows), "held": len(ok),
                         "owed": [g["part"] for g in owed],
                         "largest_courtyard_share_asked": asked},
            "from": ["parts.json", "usable.json", "plan.json"]}


def _route(rows: list, circulation: dict | None, network: dict | None,
           registered: dict) -> dict:
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
    unreachable = list(wc.get("unreachable") or [])
    reached, cells = wc.get("reached"), wc.get("cells")
    one = bool(cells) and reached == cells and not unreachable
    ok = one and len(gates) >= 1 and len(anchors) >= 1
    return {"status": "demonstrated" if ok else "failed",
            "how": (f"{reached} of {cells} lane stance(s) reachable from one another, "
                    f"{len(unreachable)} unreachable; the section's built thresholds "
                    f"include {len(gates)} gate(s) and {len(anchors)} anchor(s) on the "
                    f"same network"),
            "measured": {"walk_check": wc, "thresholds": len(thresholds),
                         "gates": [t.get("id") for t in gates],
                         "anchors": [t.get("id") for t in anchors]},
            "from": ["circulation.json", "network.json", "parts.json"]}


def _features(rows: list, usable: dict, districts: list) -> dict:
    """Every required feature of every part in scope, and whether the world shows it.

        The demand binding says what a part is required to deliver; `emitted.features` says
        what construction laid; `usable.json` says what the assembled world shows. A feature
        that is required and not affirmatively shown is owed, and an owed feature is not a
        pass -- which is the whole of the review's finding about `_function_measure`.
        
    """
    required = {}
    for d in districts or []:
        dem = d.get("demand") or {}
        for tok in dem.get("required") or ():
            required.setdefault(str(d.get("name")), set()).add(str(tok))
    owed, shown = [], []
    for r in rows:
        name = str(r.get("part", ""))
        em = r.get("emitted") or {}
        want = set()
        for dname, toks in required.items():
            if name.startswith(dname):
                want |= toks
        for tok in sorted(want):
            got = (em.get("features") or {}).get(tok)
            method = (em.get("features_method") or {}).get(tok)
            row = {"part": name, "feature": tok, "emitted": got, "method": method}
            if got is True and method in ("inferred", "observed"):
                shown.append(row)
            else:
                owed.append(row)
    # a part that stood but whose predicates could not be decided is owed, not passed
    undecided = [{"part": p, "want": w, "method": a.get("method")}
                 for (p, w), a in usable.items()
                 if a.get("holds") is None and str(a.get("method")) != "unsupported"]
    # nothing required of anything in scope is not a demonstration that requirements
    # survived: there was nothing to survive, and saying so is the honest answer
    status = ("unmeasured" if not (shown or owed)
              else "demonstrated" if not owed else "failed")
    return {"status": status,
            "how": (f"{len(shown)} required feature(s) affirmatively shown, "
                    f"{len(owed)} owed"
                    + (f" ({', '.join(sorted({o['feature'] for o in owed}))})"
                       if owed else "")
                    + (f"; {len(undecided)} predicate answer(s) undecided and therefore "
                       f"unresolved" if undecided else "")),
            "measured": {"shown": len(shown), "owed": owed[:40],
                         "undecided": undecided[:40],
                         "required_by_district": {k: sorted(v)
                                                  for k, v in required.items()}},
            "from": ["plan.json demand binding", "parts.json emitted", "usable.json"]}


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
    got = {"contrast": _contrast(reg, rows, districts, sec, runs),
           "anchor": _anchor(rows, usable, plan),
           "courts": _courts(rows, usable, districts),
           "route": _route(rows, circulation, network, reg),
           "features": _features(rows, usable, districts)}
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
    lint = (circulation or {}).get("lint") or {}
    return {
        "record": "section", "version": 2, "of": of or os.path.basename(state),
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
    }
