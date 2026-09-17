"""Part geometry, type declarations and plan validation."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import time
from dataclasses import dataclass, field

from .. import pipeline as _pipeline
from .. import card as card_mod
from .. import measure as measure_mod
from .. import offline, settlement, verdicts
from .. import spec as spec_mod
from ..measure import record
from .round import Round


PART_GEOMETRY = ("label", "kind", "x0", "z0", "x1", "z1",
                 "path", "width", "at", "facing", "size", "passage",
                 # demo-polish 2b: a point standing on an edge carries the edge's type,
                 # height and floor (`edge`), and an edge carries the points standing on
                 # it (`gates`). Written by `stages_build.annotate_gates`; read by the
                 # gate types and the wall types.
                 "edge", "gates",
                 # An edge's terrace level, from the layout, so the wall stands on the
                 # terrace edge at one level, and the face the spec's words chose for it
                 "level", "face")


PART_LEAF_KINDS = ("plot", "edge", "point", "area")


PART_GROUP_KINDS = ("district", "quarter")


def plan_parts(plan: dict) -> list:
    """Every leaf of the plan tree, in order, each carrying the groups it is inside.

        An old plan -- `{"structures": [...]}`, every round up to 11 -- is read as one
        flat list of plot leaves, so nothing that ever ran stops running. A new plan is
        `{"parts": [...]}` where a node is either a group with `children` or a leaf with a
        kind and its geometry, and `in` on each leaf is the names of its ancestors, outermost
        first: that is the only new fact a tree carries and it is what a wave, a lint scope
        and a card set are going to want.
        
    """
    if not plan.get("parts"):
        return [{**s, "kind": "plot", "name": s.get("name", s.get("id")), "in": []}
                for s in (plan.get("structures") or [])]
    out: list = []

    def walk(nodes, ancestry):
        for n in nodes:
            kind = n.get("kind", "plot")
            if kind in PART_GROUP_KINDS or n.get("children"):
                walk(n.get("children") or [], ancestry + [n.get("name", kind)])
            else:
                out.append({**n, "kind": kind, "name": n.get("name", n.get("id")),
                            "in": list(ancestry)})
    walk(plan["parts"], [])
    return out


def plan_plots(plan: dict) -> list:
    """The plot leaves of a plan, in plot-registry shape: label and rectangle.

        The same list an old plan's `structures` gives, which is the compatibility A4 is
        held to: every stage downstream reads plots and none of them has to know whether
        the plan that produced them was a tree.
        
    """
    out = []
    for p in plan_parts(plan):
        if p.get("kind", "plot") != "plot":
            continue
        out.append({"label": p["name"],
                    "x0": min(p["x0"], p["x1"]), "z0": min(p["z0"], p["z1"]),
                    "x1": max(p["x0"], p["x1"]), "z1": max(p["z0"], p["z1"])})
    return out


def part_rect(part: dict) -> tuple:
    """The rectangle a part is answerable for, before it has been sited.

        A plot and an area are rectangles already; an **edge** is its swept polyline and a
        **point** is its pad. Everything downstream that scopes a report to "this part's
        ground" -- `standard_report`, `plot_at`, the per-part lint -- needs a rectangle, and
        this is the one definition of it, so a wall is not judged on the town beside it.
        
    """
    kind = part.get("kind", "plot")
    if kind == "edge":
        half = max(1, int(part.get("width", 1))) // 2
        xs = [int(p[0]) for p in part["path"]]
        zs = [int(p[1]) for p in part["path"]]
        return (min(xs) - half, min(zs) - half, max(xs) + half, max(zs) + half)
    if kind == "point":
        at = part.get("at") or [part.get("x0"), part.get("z0")]
        h = (max(3, int(part.get("size", 5))) - 1) // 2
        return (int(at[0]) - h, int(at[-1]) - h, int(at[0]) + h, int(at[-1]) + h)
    return (min(part["x0"], part["x1"]), min(part["z0"], part["z1"]),
            max(part["x0"], part["x1"]), max(part["z0"], part["z1"]))


def part_rects(part: dict) -> list:
    """Everything else is one rectangle and is unchanged."""
    kind = part.get("kind", "plot")
    if kind != "edge" or not part.get("path"):
        x0, z0, x1, z1 = part_rect(part)
        return [(x0, z0, x1, z1)]
    half = max(1, int(part.get("width", 1))) // 2
    path = [(int(p[0]), int(p[1])) for p in part["path"]]
    out = []
    for a, b in zip(path, path[1:]):
        if a[0] != b[0] and a[1] != b[1]:
            # the box is the ground the wall encloses, and a wall's plot is its own
            # swept line
            sx, sz = (1 if b[0] > a[0] else -1), (1 if b[1] > a[1] else -1)
            for i in range(abs(b[0] - a[0]) + 1):
                cx, cz = a[0] + sx * i, a[1] + sz * i
                xs = [cx - sz * d for d in range(-half, half + 1)]
                zs = [cz + sx * d for d in range(-half, half + 1)]
                r = (min(xs), min(zs), max(xs), max(zs))
                if r not in out:
                    out.append(r)
            continue
        across = (0, 1) if a[0] != b[0] else (1, 0)
        x0, x1 = min(a[0], b[0]) - across[0] * half, max(a[0], b[0]) + across[0] * half
        z0, z1 = min(a[1], b[1]) - across[1] * half, max(a[1], b[1]) + across[1] * half
        r = (x0, z0, x1, z1)
        if r not in out:
            out.append(r)
    return out or [part_rect(part)]


def plan_ground(parts: list, vol=None) -> dict:
    """Per leaf: the relief and the water over the ground it is drawn on. A2.

        `plot_ground`'s measurement over an arbitrary rectangle rather than over a round's
        own registry, so a plan can be classified before anything has been reserved. None
        where there is no volume to read, and every ground check then reports itself
        unreadable rather than passing.
        
    """
    if vol is None:
        return {}
    from .. import observe
    h, wet = observe.ground_heights(vol)
    out = {}
    for p in parts:
        hs, w = [], 0
        for (x0, z0, x1, z1) in part_rects(p):
            for x in range(x0, x1 + 1):
                for z in range(z0, z1 + 1):
                    ix, iz = x - vol.x0, z - vol.z0
                    if not (0 <= ix < vol.codes.shape[0]
                            and 0 <= iz < vol.codes.shape[2]):
                        continue
                    hs.append(int(h[ix, iz]))
                    w += int(bool(wet[ix, iz]))
        if not hs:
            continue
        out[p["name"]] = {"columns": len(hs), "relief": max(hs) - min(hs),
                          "water_pct": round(100 * w / len(hs), 1),
                          "class": ground_class(max(hs) - min(hs),
                                                100 * w / len(hs))}
    return out


PLAN_CHECKS = ("type", "form", "role", "footprint", "ground", "frontage", "overlap")


def plan_failures(parts: list, decls: dict, ground: dict | None = None,
                  network=None, *, form=None) -> list:
    """Every leaf of a plan that cannot be built, with the need it fails, named.

        `decls` is {type name: declaration or None}; a None is a type the plan names and the
        disk does not have. `ground` is `plan_ground`'s measures, or empty where there is no
        volume yet, in which case the ground class is not checked and says so.

        `form` is the place's **form family** and not its palette. The palette used to be
        checked here -- a leaf whose type declared a foreign voice was named -- and finding
        that every leaf of a whole town was foreign to the voice it was built in was the
        right answer to the wrong question: nothing was wrong with those buildings, and the
        palette was never the type's to declare. A type no longer declares one, so what is
        left to check is the one thing it does declare about itself.
        
    """
    out = []

    def fail(p, check, why, **more):
        out.append({"part": p["name"], "type": p.get("type"),
                    "kind": p.get("kind", "plot"), "check": check, "why": why, **more})

    for p in parts:
        t = p.get("type")
        decl = decls.get(t)
        if decl is None:
            fail(p, "type", f"the plan names a type called {t!r} and "
                            f"types/{t}.py is not on disk")
            continue
        needs = decl["needs"]
        if form and not form_ok(decl.get("form"), form):
            fail(p, "form", f"{p['name']}: type {t} is a "
                 f"{decl.get('form') or 'type of no declared form'} and this place is "
                 f"{form}; a place is built in one family of form, plus the "
                 f"{' and '.join(UNIVERSAL_FORMS)} every place has",
                 allowed=[form, *UNIVERSAL_FORMS])
        # A2: ...and what it is **for**, which the form does not say. The leaf carries
        # its district's role -- `placeplan.district_plots` and `placeplan.assemble`
        # stamp it from the defining part the district was drawn for -- so this is the
        # one place the two declarations are put side by side.
        admits = (tuple(p["admits"]) if p.get("admits") is not None
                  else (COMPOUND_ROLES if p.get("compound") else ()))
        if not role_ok(decl.get("role"), p.get("role"), compound=bool(p.get("compound")),
                       admits=admits):
            fail(p, "role", f"{p['name']}: type {t} is a {decl.get('role')} building "
                 f"and this is a {p.get('role')} "
                 f"{'compound' if p.get('compound') else 'district'}; it is built out "
                 f"of what it is for, plus the {' and '.join(UNIVERSAL_ROLES)} "
                 f"buildings any part of a place may hold"
                 + (f" and the {' and '.join(admits)} types a walled compound has"
                    if admits else ""),
                 allowed=[p.get("role"), *UNIVERSAL_ROLES, *admits],
                 role=decl.get("role"))
        if decl.get("kind", "plot") != p.get("kind", "plot"):
            fail(p, "type", f"this leaf is a {p.get('kind', 'plot')!r} and "
                            f"{t} builds a {decl.get('kind', 'plot')!r}")
            continue
        why = needs_footprint_failure(p, needs)
        if why:
            from ..buildlib import Builder
            fail(p, "footprint", why, needs=list(needs["footprint"]),
                 pad=list(Builder.pad_extent(p)))
        g = (ground or {}).get(p["name"])
        if needs["ground"] != "any" and g and g["class"] not in needs["ground"]:
            fail(p, "ground", f"this ground is {g['class']} ({g['relief']} of relief, "
                              f"{g['water_pct']}% water) and this type stands on "
                              f"{', '.join(needs['ground'])}",
                 needs=list(needs["ground"]), got=g["class"])
        if needs["frontage"] == "lane":
            if p.get("kind", "plot") == "edge":
                fail(p, "frontage", "an edge is an obstacle to the network, not a "
                                    "place it routes to, and this type asks to be "
                                    "arrived at from the lane")
            elif network is not None and network.threshold(p["name"]) is None:
                fail(p, "frontage", "the circulation pass reserved no doorstep for "
                                    "this part, so nobody arrives at it")

    # A **name is an identity** everywhere downstream: `plots.json` is keyed by label,
    # `Network.threshold` looks a part up by it, `plot_at` attributes a room by it, and
    # `parts.json` records what stood by it. Two leaves with one name is therefore a
    # defect in its own right, and it is one a plan made in several calls can produce
    # without anybody being wrong. Reported here, before it can turn into a registry
    # with a part missing from it.
    seen: dict = {}
    for p in parts:
        seen.setdefault(p["name"], []).append(p)
    for name, group in seen.items():
        if len(group) > 1:
            out.append({"part": name, "type": group[0].get("type"),
                        "kind": group[0].get("kind", "plot"), "check": "name",
                        "why": f"{len(group)} leaves of this plan are called "
                               f"{name!r}, and a part's name is what the plot registry, "
                               f"the circulation pass and the readout identify it by: "
                               + "; ".join(
                                   f"one in {'/'.join(q.get('in') or ['the plan'])}"
                                   for q in group)})

    # ...and against each other. Overlap is a failure whatever the types say; coming
    # closer than either part's declared clearance is a failure of the wider claim.
    # **Indexed by position and not by name.** Keyed by name, two leaves sharing one
    # name share one rectangle, and the pair then reports as a part overlapping itself
    # wherever it stands. A duplicate name is caught above; this must not be able to
    # invent an overlap out of one.
    cells = [part_rects(p) for p in parts]
    clear = [(decls.get(p.get("type")) or {}).get("needs",
                                                  NEEDS_DEFAULT)["clearance"]
             for p in parts]
    for i, a in enumerate(parts):
        for j in range(i + 1, len(parts)):
            b = parts[j]
            # A gate stands **in** the wall, and that is the one overlap a place is made
            # of: `PASSAGE` is the type saying a network may cross it, and a crossing
            # that did not touch the thing it crosses would be a gap beside a gate.
            # Every other pair is held to its clearance.
            kinds = {a.get("kind", "plot"), b.get("kind", "plot")}
            crosses = any((decls.get(q.get("type")) or {}).get("passage")
                          for q in (a, b))
            if crosses and "edge" in kinds:
                continue
            # v2, C2: ...and two **attached** leaves on a shared frontage, party wall to
            # party wall: the one other touch a place is made of.
            if party_wall(a, b, decls):
                continue
            m = max(clear[i], clear[j])
            grown = [(r[0] - m, r[1] - m, r[2] + m, r[3] + m) for r in cells[j]]
            if not _rects_overlap(cells[i], grown):
                continue
            touching = _rects_overlap(cells[i], cells[j])
            out.append({"part": a["name"], "type": a.get("type"),
                        "kind": a.get("kind", "plot"), "check": "overlap",
                        "other": b["name"],
                        "why": (f"{a['name']} and {b['name']} overlap"
                                if touching else
                                f"{a['name']} and {b['name']} are closer than the "
                                f"{m} block(s) of clearance one of them needs")})
    return out


def party_wall(a: dict, b: dict, decls: dict) -> bool:
    """Do these two leaves stand party wall to party wall? v2, C2.

        Both plots of types that declare `ATTACHED`, both fronting the same side, and
        flank against flank -- edge-adjacent across the axis their front runs along, with
        their runs overlapping, and not overlapping each other. A touch, and only that.
        
    """
    if a.get("kind", "plot") != "plot" or b.get("kind", "plot") != "plot":
        return False
    if not all((decls.get(q.get("type")) or {}).get("attached") for q in (a, b)):
        return False
    front = a.get("front")
    if not front or front != b.get("front"):
        return False
    ra, rb = part_rect(a), part_rect(b)
    if _rects_overlap([ra], [rb]):
        return False
    if front in ("north", "south"):
        beside = ra[2] + 1 == rb[0] or rb[2] + 1 == ra[0]
        along = ra[1] <= rb[3] and rb[1] <= ra[3]
    else:
        beside = ra[3] + 1 == rb[1] or rb[3] + 1 == ra[1]
        along = ra[0] <= rb[2] and rb[0] <= ra[2]
    return beside and along


def _rects_overlap(a: list, b: list) -> bool:
    return any(p[0] <= q[2] and q[0] <= p[2] and p[1] <= q[3] and q[1] <= p[3]
               for p in a for q in b)


def needs_table(decls: dict) -> str:
    """Every type's `NEEDS`, as the table the planner's brief carries. A2.

        The planner cannot draw a plot big enough for a townhouse without being told what a
        townhouse needs, and the plot is not the pad: what it has to draw is
        `min + 2 * SITE_INSET` on each axis, so the table gives the plot and not only the
        pad. That arithmetic is `Builder.pad_extent`'s and is done here rather than left to
        a model.
        
    """
    from ..buildlib import Builder
    i = 2 * Builder.SITE_INSET
    rows = ["| type | builds | smallest plot | largest plot | pad it stands on | "
            "never on a pad of | keeps clear | frontage | ground |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(decls):
        d = decls[name]
        if d is None:
            continue
        n = d["needs"]
        lo_w, lo_d, hi_w, hi_d = n["footprint"]
        kind = d.get("kind", "plot")
        if kind == "edge":
            lo = f"width {lo_w}, run {lo_d}"
            hi = f"width {hi_w}, run {hi_d}"
            pad = "as drawn"
        elif kind == "point":
            lo, hi = f"size {lo_w}", f"size {hi_w}"
            pad = "as drawn"
        elif kind == "area":
            lo, hi = f"{lo_w}x{lo_d}", f"{hi_w}x{hi_d}"
            pad = "as drawn"
        else:
            lo, hi = f"{lo_w + i}x{lo_d + i}", f"{hi_w + i}x{hi_d + i}"
            pad = f"{lo_w}x{lo_d} to {hi_w}x{hi_d}"
        # The sizes between the two the type is measured broken at, **as pads**: the one
        # unit the inset rule below cannot move. A plan that draws one of these gets a
        # broken building and is handed back, so the planner is told before it draws.
        never = ", ".join(str(v) for v in (n.get("except") or ())) or "-"
        ground = ("any ground" if n["ground"] == "any"
                  else ", ".join(n["ground"]) + " only")
        rows.append(f"| `{name}` | {kind} | {lo} | {hi} | {pad} | {never} | "
                    f"{n['clearance']} | {n['frontage']} | {ground} |")
    rows += ["",
             f"**The plot is not the pad, and the pad is what is checked.** The library "
             f"insets a plot by {Builder.SITE_INSET} on every side to make the pad a "
             f"building stands on -- so the plots above are the pads plus {i} -- "
             f"**except that a plot with any side of {4 + 2 * Builder.SITE_INSET} or "
             f"less is inset by one on every side, not {Builder.SITE_INSET}**: an 8x8 "
             f"plot is a 6x6 pad, an 8x10 plot a 6x8 pad, and a 9x9 plot a 5x5 pad. "
             f"Work out the pad your plot gives and "
             f"hold it to the pad band and the never-on list; a pad on that list is a "
             f"broken building and is refused by name. An edge, a point and an area are "
             f"given as drawn. **Keeps clear** is the ground a type needs round it: two "
             f"parts stand at least the larger of their two figures apart, plus one, "
             f"and a part stands that far from a wall."]
    return "\n".join(rows)


def part_registry_row(part: dict, passage: bool = False) -> dict:
    """One row of `plots.json` for a part: its label, its box and its rectangles.

        The box is kept because every reader that only wants a bound -- a lint region, a
        card crop, a flood's limit -- has always read it and is still right to. `rects` is
        what the part actually covers, and is what overlap and attribution are decided on.

        `passage` is the type's own `PASSAGE`, carried into the registry because a gate
        **stands in** the wall it crosses and that is the one overlap a place is made of.
        Without it E006 reports the district's own gate as a defect.
        
    """
    x0, z0, x1, z1 = part_rect(part)
    rects = part_rects(part)
    row = {"label": part.get("name") or part.get("label"),
           "x0": x0, "z0": z0, "x1": x1, "z1": z1}
    if len(rects) > 1 or tuple(rects[0]) != (x0, z0, x1, z1):
        row["rects"] = [list(r) for r in rects]
    if passage:
        row["passage"] = True
    # It used to be written only where the row had rectangles or a passage, because
    # those were the only two readers that asked. `lint.Context.room_owner` asks it of
    # every row now -- a wall is an edge and a gate is a point and neither has an inside
    # -- and a field that is present on some rows and absent on others is a field a
    # reader has to guess at.
    row["kind"] = part.get("kind", "plot")
    return row


def fixture_part(f: dict, plots: dict) -> dict | None:
    """The part one fixture names: a plot of its round, or a part given by geometry."""
    if f.get("kind") in ("edge", "point", "area"):
        p = {k: f[k] for k in PART_GEOMETRY if k in f}
        p["label"] = f.get("part") or f.get("plot")
        p["kind"] = f["kind"]
        return p
    return plots.get(f.get("plot"))


def _type_forbidden() -> dict:
    from ..buildlib import TypeBuilder
    return dict(TypeBuilder.FORBIDDEN)


TYPE_FORBIDDEN = _type_forbidden()


#: What a type file declares at the top level. A type is a **form** and the palette is
#: the settlement's, so a type declares which family of form it belongs to and says
#: nothing at all about materials. Every one of the seven types written before A1
#: answered `STYLE` with a voice's *blurb* rather than its name -- seven for seven, open
#: thread 11 -- which is a contract that needed an example rather than a sentence.
#: `FORM` is four words and it is one of them.
TYPE_DECLARATIONS = ("FORM", "PARAMS")

#: The families of form a type may belong to. Two are regional -- what the building
#: tradition is -- and two are functional, because a wall and a market square are the
#: same thing in every tradition and a place that filtered them out by region would be a
#: walled town with no wall.
FORMS = ("european_vernacular", "east_asian", "fortification", "civic")

#: The forms admissible in a place of any family. See `form_ok`.
UNIVERSAL_FORMS = ("fortification", "civic")


def read_form(ns: dict, where: str = "a type") -> str:
    """`FORM` off a type's namespace, checked. Refuses by name."""
    got = ns.get("FORM")
    if got not in FORMS:
        raise ValueError(
            f"{where}: FORM is one of {', '.join(FORMS)}, not {got!r}. A type is a form "
            f"and the palette is the settlement's: this is one of four words, and every "
            f"material the file places comes from part['voice']")
    return str(got)


#: What a type is **for**. `FORM` says which building tradition a type belongs to; this
#: says what part of a settlement it belongs in, and the two are independent -- a
#: farmhouse and a shop-house are both `east_asian` and one of them does not go in a
#: city street. See `spec.ROLES`, which is the same four words asked of a defining part.
ROLES = ("urban", "rural", "civic", "defensive")

#: The roles admissible in a district of any role. A temple, a square and a palace stand
#: in the fields as readily as in a street, which is the same exemption
#: `UNIVERSAL_FORMS` makes and for the same reason. `defensive` is deliberately **not**
#: here: a wall and a gate are place-level parts, no district is asked to hold one, and
#: a district that put a gatehouse among its houses is a district that has misread its
#: brief.
UNIVERSAL_ROLES = ("civic",)


def read_role(ns: dict, where: str = "a type") -> str | None:
    """`ROLE` off a type's namespace, checked. None where the file does not declare one.

        Optional here for `NEEDS`' reason: a type written before A2 says nothing and is
        therefore admissible anywhere, which is exactly what the plan did before A2 existed.
        `test_types.py` is what holds every committed file to declaring it.
        
    """
    got = ns.get("ROLE")
    if got is None:
        return None
    if got not in ROLES:
        raise ValueError(
            f"{where}: ROLE is one of {', '.join(ROLES)}, not {got!r}. It is what this "
            f"building is for -- which part of a settlement it belongs in -- and it is "
            f"a different question from FORM, which is what tradition it is built in")
    return str(got)


#: The roles a **compound** of the default composition admits over its own: a walled
#: great thing has a wall and a gate whatever it is for. `placeplan.COMPOUND_ROLES` is
#: the same tuple; it lives there for the brief and here for the refusal, and both are
#: `spec.COMPOSITION_DEFAULT`'s. v2, C0: a compound's leaves carry what their family
#: admits (`admits`, stamped by `placeplan.compound_parts`), and a leaf that carries
#: none -- a plan from before -- admits this.
COMPOUND_ROLES = tuple(spec_mod.COMPOSITION_DEFAULT["admits"])


def role_ok(role: str | None, district_role: str | None, *,
            compound: bool = False, admits=None) -> bool:
    """May a type of this role stand in a district of that role?

        Yes when the district names no role, when the type declares none, when the two are
        the same, or when the type is one of `UNIVERSAL_ROLES` -- and, inside a compound,
        when it is one of the roles the compound's family admits (`admits`; `COMPOUND_ROLES`
        where the leaf says nothing).
        
    """
    if not district_role or not role:
        return True
    extra = tuple(admits) if admits is not None else (COMPOUND_ROLES if compound else ())
    return role == district_role or role in UNIVERSAL_ROLES or role in extra


def form_ok(form: str | None, place_form: str | None) -> bool:
    """May a type of this form stand in a place of that family?

        Yes when the place names no family, when the two are the same, or when the type is
        one of the functional forms -- a wall, a gate, a keep, a square. A town in the east
        Asian family still has a wall round it, and the wall is a fortification in both.
        
    """
    if not place_form:
        return True
    return form == place_form or form in UNIVERSAL_FORMS


NEEDS_DEFAULT = {"footprint": (3, 3, 64, 64), "except": (), "frontage": "any",
                 "ground": "any", "clearance": 0}


GROUND_CLASSES = ("dry", "steep", "wet")


GROUND_OF_SITING = {"plinth": "dry", "platform": "steep", "deck": "wet",
                    "footing": "dry"}


def read_needs(ns: dict, where: str = "a type") -> dict:
    """`NEEDS` off a type's namespace, checked and filled out. Refuses by name.

        A declaration nobody validates is a comment. Every field is optional and defaults
        from `NEEDS_DEFAULT`; a field that is present and malformed is an error here, once,
        rather than a plan validated against nonsense.
        
    """
    got = ns.get("NEEDS")
    if got is None:
        return dict(NEEDS_DEFAULT)
    if not isinstance(got, dict):
        raise ValueError(f"{where}: NEEDS is {type(got).__name__}, and it is a dict of "
                         f"{sorted(NEEDS_DEFAULT)}")
    unknown = sorted(set(got) - set(NEEDS_DEFAULT))
    if unknown:
        raise ValueError(f"{where}: NEEDS has no field called {unknown[0]!r}; it "
                         f"declares {sorted(NEEDS_DEFAULT)}")
    out = dict(NEEDS_DEFAULT)
    fp = got.get("footprint", out["footprint"])
    if len(tuple(fp)) != 4 or not all(isinstance(v, int) and v > 0 for v in fp):
        raise ValueError(f"{where}: NEEDS['footprint'] is "
                         f"(min_w, min_d, max_w, max_d) of whole blocks, not {fp!r}")
    fp = tuple(int(v) for v in fp)
    if fp[0] > fp[2] or fp[1] > fp[3]:
        raise ValueError(f"{where}: NEEDS['footprint'] {fp} has a minimum bigger than "
                         f"its maximum")
    out["footprint"] = fp
    # `palace` was clean at 8-12 and again at 19-28 and declared 12 as its ceiling
    # because the fixture was 12. A reliability measurement was deciding architectural
    # scale. The declaration is the whole of what was measured now: the envelope from
    # the smallest clean size to the largest, and by name every size between them the
    # sweep stood the type at and found dirty, which the plan refuses on either axis.
    ex = got.get("except", ())
    if isinstance(ex, (int, bool)) or not hasattr(ex, "__iter__") \
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in ex):
        raise ValueError(f"{where}: NEEDS['except'] is a tuple of whole sizes the sweep "
                         f"found broken inside the footprint, not {ex!r}")
    lo, hi = min(fp[0], fp[1]), max(fp[2], fp[3])
    bad = [v for v in ex if not lo <= v <= hi]
    if bad:
        raise ValueError(f"{where}: NEEDS['except'] names {bad[0]}, which is outside "
                         f"the footprint {fp}; a size outside the envelope is already "
                         f"refused by it")
    out["except"] = tuple(sorted(set(int(v) for v in ex)))
    fr = got.get("frontage", out["frontage"])
    if fr not in ("lane", "any"):
        raise ValueError(f"{where}: NEEDS['frontage'] is 'lane' or 'any', not {fr!r}")
    out["frontage"] = fr
    gr = got.get("ground", out["ground"])
    if gr != "any":
        gr = list(gr)
        bad = [g for g in gr if g not in GROUND_CLASSES]
        if bad or not gr:
            raise ValueError(f"{where}: NEEDS['ground'] is 'any' or a non-empty list "
                             f"of {list(GROUND_CLASSES)}, not {got.get('ground')!r}")
    out["ground"] = gr
    cl = got.get("clearance", out["clearance"])
    if not isinstance(cl, int) or isinstance(cl, bool) or cl < 0:
        raise ValueError(f"{where}: NEEDS['clearance'] is a whole number of blocks, "
                         f"not {cl!r}")
    out["clearance"] = int(cl)
    return out


def needs_footprint_failure(part: dict, needs: dict) -> str | None:
    """Why this part's pad is outside the type's declared footprint, or None. A1."""
    from ..buildlib import Builder
    kind = part.get("kind", "plot")
    w, d = Builder.pad_extent(part)
    lo_w, lo_d, hi_w, hi_d = needs["footprint"]
    #: A plot is the pad plus the library's inset on every side; an edge, a point and an
    #: area are given as drawn. Same conversion the brief's table makes -- and the plot
    #: quoted is the one the plan **drew**, not the pad plus four, because a plot with a
    #: side under nine is inset by one and quoting it as the pad plus four told a
    #: planner its 8x8 was a 10x10.
    i = 0 if kind in ("edge", "point", "area") else 2 * Builder.SITE_INSET
    unit = kind if i == 0 else "plot"
    if i:
        W = abs(int(part["x1"]) - int(part["x0"])) + 1
        D = abs(int(part["z1"]) - int(part["z0"])) + 1
        drawn = (f"the plot this part offers is {W}x{D}, which the library insets to a "
                 f"pad of {w}x{d}" + (" (a side under nine is inset by one)"
                                      if (W - w) < i or (D - d) < i else "")
                 + ", and this type")
    else:
        W, D = w + i, d + i
        drawn = f"the {unit} this part offers is {W}x{D} and this type"
    if kind == "edge":
        got, lo, hi = (w, d), (lo_w, lo_d), (hi_w, hi_d)
        names = ("width", "run")
    else:
        got = (min(w, d), max(w, d))
        lo = (min(lo_w, lo_d), max(lo_w, lo_d))
        hi = (min(hi_w, hi_d), max(hi_w, hi_d))
        names = ("across", "along")
    pu = "a pad of " if i else ""
    for j in (0, 1):
        if got[j] < lo[j]:
            return (f"{drawn} needs at least {pu}{lo_w}x{lo_d}: {got[j]} {names[j]} is "
                    f"under {lo[j]}" + (f" (a plot of {lo_w + i}x{lo_d + i})" if i
                                        else ""))
        if got[j] > hi[j]:
            return (f"{drawn} is written for at most {pu}{hi_w}x{hi_d}: {got[j]} "
                    f"{names[j]} is over {hi[j]}"
                    + (f" (a plot of {hi_w + i}x{hi_d + i})" if i else ""))
    # ...and a size the sweep measured broken is refused on either axis, by name. The
    # envelope says what the type reaches; this says where inside it the type does not
    # stand, and a plan that draws that size gets a broken building and is told so here
    # instead. Said in **pads**, the unit the inset rule cannot move.
    ex = tuple(needs.get("except") or ())
    if kind != "edge" and ex:
        hit = [got[j] for j in (0, 1) if got[j] in ex]
        if hit:
            return (f"{drawn} is measured broken on {pu}{hit[0]} "
                    f"{'across' if hit[0] == got[0] else 'along'}: its sweep found it "
                    f"dirty on pads of {', '.join(str(v) for v in ex)} inside "
                    f"{lo_w}x{lo_d} to {hi_w}x{hi_d}")
    return None


def ground_class(relief: int, water_pct: float) -> str:
    """Which of `GROUND_CLASSES` a piece of ground is, by `site()`'s own two thresholds.

        Not a fourth opinion about terrain: `Builder.SITE_WET` and `Builder.SITE_RELIEF` are
        what `site()` decides between a deck, a platform and a plinth on, and this is the
        same decision made from the measures `plot_ground` already computes.
        
    """
    from ..buildlib import Builder
    if water_pct > 100 * Builder.SITE_WET:
        return "wet"
    if relief > Builder.SITE_RELIEF:
        return "steep"
    return "dry"


def load_type(path: str) -> dict:
    """Read a type file's declarations without building anything.

        The module is executed in a bare namespace, which is safe precisely because the
        contract says so: a type file defines a function and some constants, and anything
        that placed a block at import time would place it once per instance. A file that
        breaks that is caught here, once, rather than twelve buildings later.
        
    """
    src = open(path).read()
    ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
    exec(compile(src, path, "exec"), ns)                       # noqa: S102
    missing = [d for d in TYPE_DECLARATIONS if d not in ns]
    if missing or not callable(ns.get("build")):
        raise ValueError(
            f"{path}: a type declares {', '.join(TYPE_DECLARATIONS)} and defines "
            f"build(b, part, seed, **params); this one is missing "
            f"{', '.join(missing + ([] if callable(ns.get('build')) else ['build']))}")
    return {"path": path, "src": src, "form": read_form(ns, where=path),
            # A2: what it is for, beside what it is built in. See `read_role`.
            "role": read_role(ns, where=path),
            "params": dict(ns["PARAMS"]), "lines": len(src.splitlines()),
            # A3: which kind of part this type builds, and whether a network may cross
            # it. Optional, and a plot where it is not said, because every type written
            # before A3 is a building on a plot and none of them says so.
            "kind": ns.get("KIND", "plot"), "passage": bool(ns.get("PASSAGE", False)),
            # v2, C2: a plot type whose flanks are party walls says so, and may then
            # stand touching the next such leaf on a shared frontage.
            "attached": bool(ns.get("ATTACHED", False)),
            # a layout draws an octagon only for a type that does
            "diagonal": bool(ns.get("DIAGONAL_RUNS", False)),
            # What it needs from the ground. Optional here for the same reason. Every
            # file under `types/` declares it and `test_types.py` is what says so.
            "needs": read_needs(ns, where=path),
            "declares_needs": ns.get("NEEDS") is not None}


def check_params(spec: dict, params: dict | None, where: str = "a type") -> dict:
    """The instance's parameters, checked against the type's own declaration.

        Refuses **by name**, which is the whole point: a config that asks a cottage for
        `storeys=9` or for a parameter it does not have should be told which one and what
        the type actually said, not handed a building that quietly ignored it.

        A declared parameter the config leaves out is filled from the declaration -- the
        low end of a range, the first of a choice -- and travels in the record, so an
        instance is always fully described by what is written down.
        
    """
    got = dict(params or {})
    unknown = sorted(set(got) - set(spec))
    if unknown:
        raise ValueError(f"{where}: no parameter called {unknown[0]!r}; this type "
                         f"declares {sorted(spec) or 'none'}")
    out = {}
    for name, rule in spec.items():
        kind = rule[0]
        if name not in got:
            out[name] = int(rule[1]) if kind == "int" else list(rule[1])[0]
            continue
        v = got[name]
        if kind == "int":
            lo, hi = int(rule[1]), int(rule[2])
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                raise ValueError(f"{where}: {name}={v!r} is out of range; this type "
                                 f"declares {name} as a whole number from {lo} to {hi}")
            out[name] = int(v)
        elif kind == "choice":
            if v not in list(rule[1]):
                raise ValueError(f"{where}: {name}={v!r} is not one of this type's "
                                 f"choices, which are {list(rule[1])}")
            out[name] = v
        else:
            raise ValueError(f"{where}: {name} is declared as {kind!r}; a parameter is "
                             f"(\"int\", lo, hi) or (\"choice\", [...])")
    return out


#: How many values of an `("int", lo, hi)` parameter a sweep stands a type at before it
#: starts sampling. Four covers every storey count and dormer count in the project; a
#: wall's `height` of 3 to 20 is eighteen and is sampled at the ends and the middle,
#: because the ends are where a type breaks and the middle is where it is used.
SWEEP_MAX = 4


def param_combinations(spec: dict, most: int = SWEEP_MAX) -> list:
    """Every combination of a type's own `PARAMS` a checker stands it at. B1.

        Every value of a choice, and every value of an int range no wider than `most` --
        which is every `storeys`, every `dormers` and every `width` this project has. A
        wider range is sampled at its low end, its high end and the middle, because a
        declaration of 3 to 20 crossed with two other parameters is a sweep nobody would
        run and therefore an instrument nobody would have.

        This is `scripts/type_needs.py`'s own enumeration, moved here so the needs sweep
        and the type checker cross the same space: the arithmetic between two instruments
        belongs to one of them.
        
    """
    import itertools
    keys = sorted(spec)
    vals = []
    for k in keys:
        rule = spec[k]
        if rule[0] != "int":
            vals.append(list(rule[1]))
            continue
        lo, hi = int(rule[1]), int(rule[2])
        whole = list(range(lo, hi + 1))
        vals.append(whole if len(whole) <= most
                    else sorted({lo, (lo + hi) // 2, hi}))
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def stage_place_spec(rnd, be, results: dict) -> dict:
    """The first stage of a round that is given nothing but a sentence, and the only one in
    the whole pipeline that reads the request. What comes back is checked by
    `ethoslm.spec.read_spec` and every number in it -- the band, the structure count, the
    footprint, the ceiling -- is derived here rather than asked for, so the same sentence
    is the same place every time it is run."""
    from .. import spec as spec_mod
    p = rnd.rel("place.json")
    brief = rnd.rel("place_spec_prompt.md")
    if os.path.exists(p):
        doc = json.load(open(p))
        # the answer as written is kept beside it.
        first_p = rnd.rel("place.rejected.1.json")
        val_p = rnd.rel("spec_validation.json")
        if os.path.exists(first_p) and os.path.exists(val_p):
            val = json.load(open(val_p))
            fields = sorted({f.get("check") for a in val.get("attempts", [])
                             for f in a.get("failures", []) if f.get("check")})
            n = len(val.get("attempts", [])) + 1
            kept = rnd.rel(f"place.attempt.{n}.json")
            if not os.path.exists(kept):
                json.dump(doc, open(kept, "w"), indent=1)
            doc = spec_mod.merge_hand_back(json.load(open(first_p)), doc, fields)
            doc["hand_back"] = {"fields_taken": fields, "from_attempt": n,
                                "rest_from": os.path.relpath(first_p, _pipeline.ROOT)}
        try:
            s = spec_mod.read_spec({k: v for k, v in doc.items() if k != "hand_back"},
                                   rnd.sentence or None)
        except spec_mod.SpecError as e:
            # **Handed back once, by name.** The refusal goes on the end of the brief,
            # the answer is set aside, the record says which field, and the same
            # isolated call is asked again; refused twice, the round stops.
            val = (json.load(open(rnd.rel("spec_validation.json")))
                   if os.path.exists(rnd.rel("spec_validation.json")) else {"attempts": []})
            n = len(val["attempts"]) + 1
            failure = {"part": getattr(e, "part", None), "check": getattr(e, "field", None),
                       "why": str(e)}
            val["attempts"].append({"level": "place_spec", "attempt": n,
                                    "failures": [failure],
                                    "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            json.dump(val, open(rnd.rel("spec_validation.json"), "w"), indent=1)
            os.replace(p, rnd.rel(f"place.rejected.{n}.json"))
            if n >= 2:
                return {"status": "error", "stop": True,
                        "error": f"the place spec was refused twice; the last: {e}"}
            field = failure["check"] or "the field named"
            with open(brief, "a") as fh:
                fh.write("\n\n---\n\n## Your place spec was checked and it does not read\n\n"
                         "1 thing in what you wrote cannot be read as a spec. Nothing has "
                         "been planned and nothing is in the world. Write the file again, in "
                         "the same place, and change **only** the field named here: every "
                         "other field is taken from your first answer whatever you write.\n\n"
                         f"- **{failure['part'] or 'the spec'}** ({field}) -- {e}\n")
            return {"status": "needs_model", "role": "spec", "request": brief, "write": p,
                    "attempt": n + 1, "handed_back": failure,
                    "note": "handed back once: the refused field named on the brief, and "
                            "only that field is taken from the next answer"}
        # this is where it lands on disk, so the planner briefs, the type instances and
        # the place read all see the same palette from the same place a hand-written
        # voice comes from.
        wrote = None
        # ...and every voice a **ring** authored, on the same terms: a district's
        # palette is a file under voices/ before anything is planned, so the district
        # brief, the type instance and the place read all see it from the same place.
        to_author = ([(s["voice"], s["authored_voice"])] if s.get("authored_voice")
                     else []) + sorted((s.get("authored_voices") or {}).items())
        if to_author:
            from .. import voices as _voices
            wrote = []
            for name, doc in to_author:
                try:
                    _voices.author(name, doc)
                except _voices.VoiceError as e:
                    return {"status": "error", "stop": True,
                            "error": f"{p} authors a voice that will not load: {e}"}
                wrote.append(os.path.relpath(_voices.path_for(name), _pipeline.ROOT))
                print(f"   authored voice {name} -> {wrote[-1]}", flush=True)
            wrote = wrote[0] if len(wrote) == 1 else wrote
        json.dump(s, open(rnd.rel("place.checked.json"), "w"), indent=1)
        print(spec_mod.summary(s), flush=True)
        return {"status": "read", "spec": s, "path": p,
                "authored_voice": wrote,
                "summary": spec_mod.summary(s)}
    if not rnd.sentence:
        return {"status": "error", "stop": True,
                "error": "a round that plans a place carries the sentence it is "
                         "planning: `sentence` in the config, and nothing else"}
    if not os.path.exists(brief):
        os.makedirs(rnd.state, exist_ok=True)
        open(brief, "w").write(spec_brief(rnd.sentence, p))
    return {"status": "needs_model", "role": "spec", "request": brief, "write": p,
            "note": "one call, a fixed schema: the sentence becomes a place spec and "
                    "nothing about the site, the scale or the geometry is decided here"}


SPEC_BRIEF = """# Read a sentence into a place spec

Somebody has asked for a place to be built. Here is the whole of what they said:

> {sentence}

Your one job is to read that sentence into a **place spec**: a small, fixed JSON object
saying what kind of place it is and what makes it that place. You are not siting it, not
sizing it, not planning it and not building it. Every one of those happens later and none
of them is yours.

## What you decide

**`kind`** — one of `{kinds}`.

**`invariants`** — **for a named place, first.** Where the sentence names a place somebody
would recognise -- a real or a fictional capital, a castle, a shrine, a fortress -- write
one short paragraph, before anything else, stating its defining spatial facts the way a
person who knows it would list them: what is at the centre; the rings or quarters from the
centre outward, roughly what share of the whole each is, which of them are walled, and how
each is built and coloured; and the land it stands in. Then write the schema below **from
that paragraph and nothing else**: every ring you named becomes a district part carrying
its `ring`, `share`, `walled` and `voice`, every wall you named is counted in the
concentric wall part, the land goes in `setting`. A place nobody has heard of gets `null`.

**`defining_parts`** — the things without which this is not the place that was asked for.
"Walled" means a wall; "with a market square" means a square; "and a keep" means a keep;
a ringed capital means its rings, its walls and the compound at its centre. A defining part
is **not** every building in the place: the houses are not defining parts, the district
that holds them is. Each one is:

    {{"name":      "short_lower_case_slug",
      "kind":      one of {part_kinds},
      "family":    one of {families},
      "relation":  one of {relations},
      "count":     how many of this part there are (three rings is count 3),
      "structures": how many of the place's ordinary buildings this part accounts for
                    (0 for a wall or a gate; the houses go in the districts),
      "needs":     what ground *this part* wants, or omit it entirely,
      "forms":     which families of form this part's buildings may be, or omit it,
      "density":   one of {densities}, or omit it,
      "role":      one of {roles}, or omit it,
      "ring":      for a district of a concentric place: 0 nearest the centre, then
                   1, 2, ... outward; omit it where the place has no rings,
      "share":     with `ring`: this ring's fraction of the place's whole area,
      "walled":    with `ring`: true where a wall bounds this ring on its outside,
      "voice":     with `ring`: the palette this ring is built in -- a voice name, null
                   for the place's own, or a voice you write (the same object as the
                   place's `voice` below),
      "character": for a district: what it is like, as words and a few numbers --
                   {{"frontage": {frontages}, "block": columns along a street,
                   "lot_depth": columns back from it, "attached": true | false,
                   "courtyard_share": 0..1, "open_share": 0..1,
                   "variety": 0..1 of the lot's own side -- how far one lot's
                   size may stray from the next's, so a street is a rhythm and
                   not a comb, "storeys": [lo, hi] -- the band the buildings on
                   it take their height from, clamped into what each type
                   declares, so a run of roofs steps rather than lying flat,
                   "landmarks": [{{"type": a type name, "notes": ...}}]}} -- every
                   field optional (the density word fills the rest); omit the whole
                   object and the district is planned plot by plot instead,
      "notes":     one sentence on what it is and why the sentence implies it}}

  - **Rings.** A concentric place -- rings of districts around one thing at the middle --
    is laid out by arithmetic from the four ring fields, and no planner draws its rings
    freehand: the ring rectangles come from the cumulative shares, centred on the site's
    centre; a wall stands at every `walled` boundary, the outermost of them being the
    place's great wall; one gate per walled ring on one axis; and districts tile every
    ring edge to edge. The shares are what a person sees from the air: a belt of farmland
    that is half the place is half the place, and a `share` of 0.1 makes it a hedge. The
    rings' shares sum to at most 1 and **what is left is the centre's**. The `concentric`
    wall part's `count` is the number of walled rings and is refused if it says otherwise.
    **Where the rings are classes, each ring carries its own voice**, and no two of them
    share one: the poor ring and the court's ring are not the same colour, and a place
    whose rings differ in wealth and not in palette reads from the air as one quarter
    repeated. The axis runs **from the darkest and plainest at the outside to the
    brightest and richest at the centre** -- mud and rough stone under dark tile on the
    edge, dressed stone under a strong roof in the middle, white stone and a gilded roof
    at the court. Each voice card below carries the colour of every material in it, so
    choose with your eyes: pick a voice from the list for each ring, or write one, and
    say in its `blurb` which end of the axis it is. For example, a ringed capital: a
    palace compound at the `centre`; ring 0 the court's quarter, share 0.06, low, not
    walled; ring 1 the merchants', share 0.1, medium; ring 2 the artisans' and the poor,
    share 0.2, dense, walled; ring 3 the farm belt, share 0.6, sparse, rural, walled --
    two walled rings, so `count` 2 on the concentric wall. A castle with two baileys is the
    same shape at a smaller scale: the keep at the centre, ring 0 the inner bailey,
    walled, ring 1 the outer bailey, walled.

  - `kind` is what the library builds it as: an `edge` is a wall or a quay, a `point` is
    a gate or a well head -- **a gate is sized by its wall and is not drawn**: put it at
    a cell on the wall's line and the library sizes its pad from the wall's height and
    builds it to the wall's crown -- an `area` is a square or a market, a `plot` is a building, and
    a `group` is a division of the place that later calls fill in — a district or a
    quarter of houses, **or a great thing that is itself a composition of parts**: a
    castle (`group` of family `keep`), a monastery, a cathedral close. A great thing is
    planned as a place inside the place — its own wall, gates, halls and courts — and
    a `palace` or a `monument` is always one, whatever kind you give it.
  - `relation` is how it sits: `concentric` for rings inside rings, `centre` for the
    thing at the middle, `perimeter` for what goes round the outside, `gateway` for a way
    through, `edge` for what sits at the boundary, `throughout` for what is spread over
    the place, `quarter` for a division of it, `beside_the_centre` for what flanks it,
    and three that name another part in `of`: `near` for what stands on a side of that
    part, `along` for what runs beside an edge's line, `on` for a point at a cell of
    an edge's line (a tower on the wall). The library places every part from these
    words; nothing here is a coordinate.
  - **`needs`** is the one place a *part* may speak about ground, and it exists because
    one number over a whole place says nothing a plan can act on. A compound at the
    centre wants a level square and says so; a ring of farmland on a hillside does not;
    a wall climbs whatever is there. Write `{{"max_relief": <blocks over this part's own
    ground>, "plateau": <the side of the level square this part stands on>}}`, either
    field or both, and omit `needs` altogether where the part does not care. **The part
    at the `centre` is the one whose `needs` decide where the whole place goes**: its
    `max_relief` is what the middle of every candidate site is scored against and its
    `plateau` is how much ground is levelled for it, so a number here that is too small
    is a search that finds nothing and one that is too large is a quarry.
  - **`forms`** is a list drawn from `{forms}` — which building traditions this part's
    buildings may be from, where that differs from the place's own. Omit it and the
    place's `form` applies, which is the right answer almost always.
  - **`density`** says how thickly a part that holds structures is built up. It is a
    word and not a number: the ground each structure takes is computed from it here.
  - **`character`** is how a district is *made*, and it is what you write for a
    district in place of any plan of it: a district with a character is compiled --
    streets at the block size, blocks of lots at the frontage the types declare,
    buildings fronting the streets, courtyards and open ground by the shares, the
    landmark on the block nearest the middle -- and no call draws its plots. Say
    `attached` where the sentence means terraces or a street of party walls,
    `frontage: "open"` where the buildings stand apart in their own ground, a large
    `open_share` for orchards, fields and commons, a `courtyard_share` where the
    houses share yards behind them, and a `landmark` where one building is the
    district's own -- a hall, a temple, a market. The numbers are approximate and the
    compiler holds the result to the density's count and cover.
  - **`role`** is what the part is *for*, and it decides which types a later call may
    build in it: a `rural` district is farmhouses and fields, an `urban` one is street
    houses and shops, a `civic` one is what a place holds at its middle, a `defensive`
    one is the wall and its gates. Omit it and it is taken from the family — a wall is
    defensive, a house is urban, a district is rural where it is sparse and urban
    otherwise — which is right for almost every part. Say it where that would be wrong.

**`setting`** — the land the place stands *in*, which is the first thing anyone sees of
it and the one thing the search cannot infer from the parts. Four words, or `null`:

    {{"surface": one of {surfaces}, or null,
      "water":   "some" | "none" | null,
      "relief":  "flat" | "rolling" | "steep" | "any" | null,
      "biome":   a list drawn from {biomes}, or null,
      "notes":   one sentence on why the sentence implies it}}

  - `surface` is what the ground should predominantly be made of: `green` is grass and
    plains, meadow, forest floor; `sand` is desert; `badlands` is red sand and
    terracotta; `snow` is snowfields and ice; `stone` is bare rock, gravel and
    mountainside; `earth` is dirt, mud and clay. A named place with a known setting
    gets that setting -- a capital on a plain is `green`, a caravan fort is `sand`, a
    mountain hold is `snow` or `stone`. Write `null` only where the sentence says or
    implies nothing about the land at all.
  - `water` is whether the footprint should hold water: `some` for a lake, a river
    or a harbour, `none` for a place that is dry by definition, `null` for no opinion.
  - `relief` is the lie of the whole footprint: `flat` for a plain, a delta, a steppe;
    `rolling` for hill country a town climbs; `steep` for a place that *wants* a
    mountain -- a cliff keep, a mountain village, a hold on a crag -- which is a floor
    on the fall across the footprint and a preference for high ground; `any` where
    the sentence does not care. A place whose centre wants level ground says so in
    that part's `needs`; this word is about everything else.
  - `biome` is the world's own word for the land, and it is what the trees, the grass
    and the water are: `plains` is open grassland and meadow; `savanna` is dry
    acacia grass; `forest` is woodland of any kind; `jungle`, `desert`, `badlands`,
    `swamp`, `snowy` and `mountain` are what they say. Write the list a person who
    knows the place would accept -- a capital on a green plain is `["plains"]`, a
    hill fort might take `["plains", "forest"]` -- and `null` where the sentence
    implies nothing. The search holds a candidate to having most of its ground in
    the list and prefers more of it; a river or a coast across a square counts
    against it. Every word here is a need, not a hint: a square that fails one is
    never chosen, and if nothing meets them the search says so and stops.
  - **No percentages, no block names.** How much of the land the surface has to be,
    how much of the ground the biome list has to be, and how much fall a `flat` or
    a `steep` footprint may carry, are registered numbers the search holds every
    candidate to; you say which words.

**`voice`** — one of three things.

  - the **name** of one of the voices below, if the sentence names or strongly implies a
    style one of them already is;
  - `null`, if it does not: a voice is chosen against the ground later, and guessing one
    here is a decision made without the information it needs;
  - **a voice you write**, if the sentence asks for a palette none of these is. This is
    the case for a named place with a known look — ochre stone under green tile is not
    on the list below and never will be. Write it as an object:

        {{"name":  "short_lower_case_slug",
          "roles": {{"wall": ..., "footing": ..., "frame": ...,
                    "roof": ..., "trim": ..., "floor": ...,
                    "wall_alt": ... or omit it, "ground": ... or omit it}},
          "roof":  {{"profile": [[rise, run], ...] or null,
                    "ends": "gable"|"hip"|"half-hip"|"irimoya" or null,
                    "eave": "straight"|"flared"|"upturned",
                    "tiers": 1}},
          "roof_civic": the same four keys, or omit it,
          "notes": {{"blurb": ..., "construction": ..., "roofs": ...,
                    "ground": ..., "signature": ...}}}}

    `roof` is the silhouette every building in the place is roofed with. `roof_civic`
    is optional and is the roof of the place's **civic** buildings -- its temple, its
    hall, the halls of its palace -- where the tradition roofs those more grandly than
    its houses: more tiers, a steeper ridge. Write it where the sentence's place has
    that hierarchy and omit it where every roof is the same; a house never gets it.

    **`wall_alt`** is optional and is a **second** wall material. A quarter of the
    buildings are faced in it, chosen by the library, so a street of thirty houses is
    not one extruded building thirty times over. Write it where the tradition builds in
    more than one stone or more than one timber -- a second course colour, a brick among
    the render -- and omit it where the place really is one material. It is a family
    like `wall` is.

    **`ground`** is optional and is the **ground between the buildings**: the terraces,
    the ramps, the open ground the place designs. Write it **only** where the place's
    ground is genuinely made rather than grown -- a quarried city, a paved precinct, a
    desert town of beaten earth. Omit it otherwise, and the ground is the land's own
    surface at every column: grass on a plain, sand on a shore. Omitting it is the
    normal answer and a place that is not a quarry should omit it. Any block, like
    `floor`, because it is only ever laid as a cube.

    Every role but `floor` must name a **material family** — a material this game has
    stairs and a slab of, because the shell lays every one of them in all three shapes.
    The families are, each with **the colour of its cube** so you can choose one with
    your eyes rather than from memory (a family with no colour beside it is one this
    project has no reading for):

{families_line}

    `floor` alone may be any block, because it is only ever laid as a cube. A voice that
    names a family the game has no stairs of is refused with the role and the family
    named, and nothing is planned.

{voices}

**`form`** — which family of form this place is built in: one of `{forms}`. The two
regional ones say what the building tradition is; a place is planned out of types of its
own family plus the `fortification` and `civic` ones, which are what a wall, a gate, a
keep and a market square are in every tradition. `null` accepts every committed type,
which is right when the sentence implies no tradition at all.

**`notes`** — a sentence or two on your reading, for the record.

## What you must not do

- **Do not give any number of structures, any size, any footprint or any coordinate.**
  The band is fixed by the kind and by any explicit count in the sentence, and it is
  computed from those and not from you. A spec with a size in it will be rejected.
- **Do not turn a named place into its real extent.** A named place contributes its
  defining parts and their **proportions** — the rings, their shares, their walls, the
  compound at the centre — and never its size in blocks; a place that does not fit is
  scaled by the ceiling, and a spec that asked for the real thing would be scaled from a
  number nobody measured.
- Do not add defining parts the sentence does not imply. A hamlet is a hamlet; if nothing
  says it is walled, it has no wall.

## Output

Write a single JSON file to {out}:

{{
  "kind": "...",
  "invariants": "the named place's defining spatial facts, or null",
  "defining_parts": [ ... ],
  "setting": {{"surface": ..., "water": ..., "relief": ..., "biome": [...], "notes": "..."}} or null,
  "voice": null,
  "form": null,
  "notes": "..."
}}

Use the Write tool. Output only the file; reply with the kind and the number of defining
parts.
"""


def spec_brief(sentence: str, out_path: str) -> str:
    from .. import groundread, spec as spec_mod, styles
    from ..prims import MATERIALS
    # **Every voice with the colour of the three materials a person sees of it**, the
    # craft round (E2): a call choosing a palette per ring is choosing by colour, and a
    # blurb alone asks it to know Minecraft's block list by heart. `ceremonial` is said
    # out loud, because a voice that is a quarter's and not a place's is a candidate for
    # a ring and never an answer to `voice` at the place level.
    def _line(k, v):
        seen = ", ".join(f"{r} {v['palette'][r]}{styles._colour_note(v['palette'][r])}"
                         for r in ("wall", "roof", "trim") if v["palette"].get(r))
        return (f"  - `{k}`{' — **ceremonial**, for a ring and not for a whole place'
                            if v.get('ceremonial') else ''} — "
                f"{v['blurb'].splitlines()[0][:110]}\n      {seen}; "
                f"value {v['value']['darkest'] * 100:.0f}-"
                f"{v['value']['lightest'] * 100:.0f}% of white")

    voices = "\n".join(_line(k, v) for k, v in styles.VOICES.items())
    lines = _family_colours()
    return SPEC_BRIEF.format(
        sentence=sentence.strip(), out=out_path,
        kinds=" | ".join(spec_mod.KINDS),
        part_kinds=list(spec_mod.PART_KINDS), families=list(spec_mod.FAMILIES),
        relations=list(spec_mod.RELATIONS), forms=" | ".join(spec_mod.FORMS),
        densities=" | ".join(sorted(spec_mod.DENSITIES,
                                    key=lambda k: -spec_mod.DENSITIES[k])),
        roles=" | ".join(spec_mod.ROLES),
        frontages=" | ".join(f'"{f}"' for f in spec_mod.FRONTAGES),
        surfaces=" | ".join(f'"{s}"' for s in groundread.SURFACES),
        biomes=" | ".join(f'"{b}"' for b in groundread.BIOMES if b != "any"),
        families_line="\n".join(f"        {ln}" for ln in lines),
        voices="The voices this project already has:\n\n" + voices)


#: How the colour of a material is named to a voice's author. A voice card showed a
#: block's **name** and never its colour, so an author choosing "a green roof" had to
#: know from memory that `oxidized_copper` is the only green the shell can lay in stairs
#: and slabs, and could not weigh it against prismarine or a dark tile.
#: `preview.block_colour` is this project's own block-to-colour table -- the one that
#: draws every preview -- so the answer already existed and nobody had ever put it in
#: front of the model. No material is chosen for the author; the colour is stated beside
#: the name and the choice stays theirs.
COLOUR_NAMES = (
    ((0, 20), "black"), ((20, 45), "very dark"), ((45, 70), "dark"),
    ((70, 105), "mid"), ((105, 145), "light"), ((145, 255), "pale"))


def colour_of(block: str) -> str:
    """"light warm grey", "mid green": the luma band and the hue, off `block_colour`."""
    from ..preview import block_colour
    r, g, bl = block_colour(block)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bl
    band = next(n for (lo, hi), n in COLOUR_NAMES if lo <= luma < hi)
    mx, mn = max(r, g, bl), min(r, g, bl)
    if mx - mn < 18:
        return f"{band} grey"
    if g > r + 20 and bl > r + 20:
        hue = "teal"
    elif r >= g and r >= bl:
        hue = "red" if r - g > 60 else ("orange" if g > bl else "warm")
    elif g >= r and g >= bl:
        hue = "green" if g - max(r, bl) > 12 else "olive"
    else:
        hue = "blue" if bl - max(r, g) > 20 else "cool"
    return f"{band} {hue}"


def _family_colours() -> list:
    """Every material family with the colour of its cube, three to a line.

        A family the colour table has no entry for is named **without** a colour rather
        than with a guessed one: `block_colour` answers magenta for a block it does not
        know, and a wrong colour is worse than none.
        
    """
    from ..preview import UNKNOWN, block_colour
    from ..prims import MATERIALS, solid
    rows = []
    for fam in sorted(MATERIALS):
        try:
            block = solid(fam)
            rows.append(f"{fam} ({colour_of(block)})"
                        if block_colour(block) != UNKNOWN else fam)
        except Exception:                        # noqa: BLE001 -- name it without one
            rows.append(fam)
    return ["; ".join(rows[i:i + 3]) for i in range(0, len(rows), 3)]


def stage_site_search(rnd, be, results: dict) -> dict:
    """Deterministic and no model call: the sentence has already said what the place needs
    and this is arithmetic over the world. What goes on the record is the top three with
    their scores and every radius the scan reached, so the choice can be argued with."""
    import subprocess
    p = rnd.rel("site_search.json")
    if os.path.exists(p):
        got = json.load(open(p))
        return {"skipped": "already searched -- the site is a fixture once chosen",
                "chosen": got.get("chosen"), "attempt": got.get("attempt"),
                "top": got.get("top", [])[:3], "path": p}
    spec = rnd.place_spec()
    if not spec:
        return {"status": "error", "stop": True,
                "error": "no place.json: the site search is scored on what the place "
                         "needs and nothing has said what that is"}
    if not be.live and getattr(be, "dry_run", False):
        # The dry run's search: every grid square `out/sites/` already holds, scored the
        # same way and ranked in the same order, with **no server at all**. A square
        # nothing has read is recorded as unread rather than guessed at, which is the
        # difference between "no site" and "no ground", and the readout reports which.
        got = _dry_site_search(rnd)
        json.dump(got, open(p, "w"), indent=1)
        if not got.get("chosen"):
            why = got.get("best_failed") or {}
            return {"status": "error", "stop": True, "record": p,
                    "error": (f"nothing met the needs over {got.get('candidates')} "
                              f"square(s); the best, {why.get('origin')}, fails "
                              f"{why.get('failures')}" if got.get("candidates") else
                              f"the dry-run site search ranked nothing: "
                              f"{got['candidates']} square(s) scored, "
                              f"{got['squares']['unread']} unread at "
                              f"{got['footprint']}x{got['footprint']}. A dry run scores "
                              f"the ground already in out/sites/; ground nothing has "
                              f"ever read has to be read once, by a session that writes "
                              f"nothing: scripts/find_site.py --fresh"),
                    "squares": got["squares"]}
        return {"chosen": got["chosen"], "attempt": got.get("attempt", "dry run"),
                "squares": got["squares"], "top": got.get("top", []), "path": p,
                "dry_run": True}
    # the same chunks a session would serve, byte for byte, and no server anywhere.
    # Ground the files do not hold is unread and never made; a round that needs ground
    # made says so with `max_new_squares` and runs live.
    offline = not be.live
    if offline and rnd.flags.get("max_new_squares"):
        return {"status": "error", "stop": True,
                "error": "max_new_squares asks the server to make ground, and a file "
                         "cannot: run this stage --live"}
    r = subprocess.run(
        [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
         os.path.join(_pipeline.ROOT, "scripts", "find_site.py"),
         *(["--offline"] if offline else []),
         # The same read is sixteen times the area and none of it is kept. The grid walk
         # reads one square at a time and **caches every one under `out/sites/`**, so a
         # scan is paid for once ever and the dry run that follows scores exactly the
         # ground this session read, with no server at all.
         "--fresh",
         # The **checked** spec, not the model's raw answer: `read_spec` fills in the
         # band, the count and the footprint, and the search is scored on those.
         "--spec", rnd.rel("place.checked.json"), "--out", p,
         "--sentence", rnd.sentence,
         # Ground the region files do not cover is ground the server would have to
         # *make*, and making it cannot be undone short of the snapshot. None, unless
         # the round says how many and takes the entry in the ledger.
         "--max-new", str(int(rnd.flags.get("max_new_squares") or 0)),
         *(["--radius", str(int(rnd.flags["search_radius"]))]
           if rnd.flags.get("search_radius") else []),
         # and the preflight's bound, by name.
         *(["--locate"] if be.live and rnd.flags.get("locate_biome", True) else []),
         *(["--bound", str(float(rnd.flags["search_bound_seconds"]))]
           if rnd.flags.get("search_bound_seconds") else [])],
        env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name), text=True)
    if not os.path.exists(p):
        return {"status": "error", "stop": True, "returncode": r.returncode,
                "error": "find_site.py chose nothing and wrote no record"}
    got = json.load(open(p))
    if not got.get("chosen"):
        # the round stops here rather than building on ground the spec refused.
        why = got.get("best_failed") or {}
        return {"status": "error", "stop": True, "record": p,
                "squares": got.get("squares"), "refused": got.get("refused"),
                "error": ("the site search's preflight refused to start: "
                          + json.dumps(got.get("refused"))[:400]
                          if got.get("refused") else
                          f"nothing met the needs over {got.get('candidates')} "
                          f"square(s); the best, {why.get('origin')}, fails "
                          f"{why.get('failures')}")}
    if be.live:
        be.rebind()
    return {"chosen": got["chosen"], "attempt": got["attempt"],
            "read_by": got.get("read_by"), "squares": got.get("squares"),
            "terraform": got.get("terraform"),
            "size_band_dropped": got.get("size_band_dropped"),
            "top": got.get("top", []), "path": p}


def compound_cut(rnd, spec, site, terra, n: int, x0: int, z0: int, m: dict) -> tuple:
    """How big the cut is, and where -- when the part it is being cut for is a compound.

        The window moves with the size, because the flattest 40x40 in a square is not the
        flattest 83x83 in it: `_flattest_window` finds the new one off the ground as the
        round has it cached -- dry first, then least relief, then nearest the middle of the
        site. With no cached volume to read, the recorded window is grown about its own
        centre and clamped inside the site, which is the same square when nothing moved.

        Returns `(n, x0, z0, m)` unchanged where the part is not a compound: a district, a
        market square, a keep, a hamlet with no plateau at all cuts what it always cut.
        
    """
    from .. import placeplan, spec as spec_mod
    from ..buildlib import Builder
    part = next((d for d in (spec.get("defining_parts") or [])
                 if d.get("name") == terra.get("part")), None)
    X, Z, S = int(site["origin"][0]), int(site["origin"][1]), int(site["size"])
    if not part or not spec_mod.compound(part):
        # **A concentric place's plateau is at the site's centre**, whatever the part
        # is: the rings are drawn round that point. See `_centred_window`.
        if spec_mod.rings(spec):
            return (n, *_centred_window(rnd, X, Z, S, n, m,
                                        terrace=terrace_for(rnd, spec, site)))
        return n, x0, z0, m
    want = int(min(placeplan.compound_ground(spec=spec, site_side=int(site["size"]))["side"],
                   Builder.plateau_max(int(site["size"])), int(site["size"])))
    if spec_mod.rings(spec):
        # Dry-before-flat found the driest flattest window anywhere in the site, and on
        # the demo's site that was ninety blocks east of the centre, round a lake: every
        # ring was then drawn about the wrong point. A concentric place is centred by
        # definition, the search now refuses a wet core (`find_site.search_needs`), and
        # the plateau fills what water is left from the bed -- so the window is the
        # site's centre, and nothing else.
        want = max(want, n)
        return (want, *_centred_window(rnd, X, Z, S, want, m,
                                       terrace=terrace_for(rnd, spec, site)))
    if want <= n:
        return n, x0, z0, m
    got = _flattest_window(rnd, X, Z, S, want)
    if got is None:
        cx = int(x0) + n // 2
        cz = int(z0) + n // 2
        nx = max(X, min(cx - want // 2, X + S - want))
        nz = max(Z, min(cz - want // 2, Z + S - want))
        return want, nx, nz, m
    nx, nz, y, relief = got
    return want, nx, nz, {**m, "size": want, "x": nx, "z": nz, "y": y,
                          "relief": relief}


def terrace_for(rnd, spec: dict, site: dict) -> dict | None:
    from .. import placeplan, spec as spec_mod
    rings = spec_mod.rings(spec)
    if not rings:
        return None
    try:
        vol = rnd.volume()
    except Exception:                       # noqa: BLE001 -- reported as "no volume"
        vol = None
    med = placeplan.site_median(vol, site)
    if med is None:
        return None
    return placeplan.terrace_levels(med, len(rings))


def _centred_window(rnd, X: int, Z: int, S: int, n: int, m: dict,
                    terrace: dict | None = None) -> tuple:
    """A centred place's plateau is cut at the site's centre, not at the driest flattest
    ground anywhere in it.
    """
    import numpy as np
    from .. import observe
    x0 = X + (S - n) // 2
    z0 = Z + (S - n) // 2
    try:
        vol = rnd.volume()
    except Exception:                       # noqa: BLE001 -- reported as "no volume"
        vol = None
    if vol is None:
        out = {**m, "size": n, "x": x0, "z": z0, "centred": True}
        if terrace:
            out.update(y=int(terrace["podium"]), terrace=terrace)
        return x0, z0, out
    h, wet = observe.ground_heights(vol)
    i, j = x0 - vol.x0, z0 - vol.z0
    win = h[i:i + n, j:j + n].astype(np.int32)
    damp = np.asarray(wet)[i:i + n, j:j + n]
    if win.size == 0:
        out = {**m, "size": n, "x": x0, "z": z0, "centred": True}
        if terrace:
            out.update(y=int(terrace["podium"]), terrace=terrace)
        return x0, z0, out
    # the level is the median of the **land**: water stands at its own surface, and a
    # plateau laid at a pond's level is a plateau laid one block too low
    land = win[~damp] if (~damp).any() else win
    out = {**m, "size": n, "x": x0, "z": z0, "y": int(np.median(land)),
           "relief": int(win.max() - win.min()),
           "water_pct": round(100.0 * float(damp.mean()), 1),
           "centred": True}
    if terrace:
        out.update(window_median=int(np.median(land)), y=int(terrace["podium"]),
                   terrace=terrace)
    return x0, z0, out


def _flattest_window(rnd, X: int, Z: int, S: int, n: int):
    """(x, z, y, relief) of the driest, flattest `n`x`n` window in the site, nearest its
        middle.

        **Dry before flat**, because a lake is the flattest thing on any site: the surface of
        standing water is one level over its whole extent, so "least relief" alone picks the
        middle of it every time -- the first 83x83 this found on the green site was 5,727
        columns of water out of 6,889. Water is not ground and this is where that is said.
        
    """
    import numpy as np
    from .. import observe
    try:
        vol = rnd.volume()
    except Exception:                       # noqa: BLE001 -- reported as "no volume"
        vol = None
    if vol is None:
        return None
    h, wet = observe.ground_heights(vol)
    i, j = X - vol.x0, Z - vol.z0
    inner = h[i:i + S, j:j + S].astype(np.int32)
    damp = np.asarray(wet)[i:i + S, j:j + S].astype(np.int32)
    if min(inner.shape) < n:
        return None

    def slide(a, fn):
        """`fn` over every window of `n` along axis 0, then along axis 1."""
        for _ in range(2):
            m = a.shape[0] - n + 1
            r = a[:m].copy()
            for k in range(1, n):
                fn(r, a[k:k + m], out=r)
            a = r.T
        return a

    rel = slide(inner, np.maximum) - slide(inner, np.minimum)
    box = np.pad(np.cumsum(np.cumsum(damp, 0), 1), ((1, 0), (1, 0)))
    water = (box[n:, n:] - box[:-n, n:] - box[n:, :-n] + box[:-n, :-n])
    ii, jj = np.indices(rel.shape)
    mid = (S - n) / 2.0
    near = (ii - mid) ** 2 + (jj - mid) ** 2
    p = int(np.lexsort((jj.ravel(), ii.ravel(), near.ravel(), rel.ravel(),
                        water.ravel()))[0])
    a, b = int(p // rel.shape[1]), int(p % rel.shape[1])
    win = inner[a:a + n, b:b + n]
    return (X + a, Z + b, int(np.median(win)), int(rel[a, b]))


def settle_designed(vol, pieces: list, *, relief=None) -> tuple:
    """Designed ground -- a podium, a ring's terraces, the level run and the ramp
        outside a gate -- declared and settled through the one ground contract. v2, B1.

        `pieces` is `[(label, rect, level, what), ...]`, corners inclusive; each is a
        **platform** of the `designed` class, declared in that order, and the resolution
        settles every column once and derives the seams between the pieces and the
        ground as found -- a step of `TERRACE_STEP` between two rings is a retaining face
        by the drop, a ramp's tread a kerb. Returns `(resolved, record)`; the caller lays
        each piece at `resolved.level_of(label)`.
        
    """
    from .. import ground as _ground
    from ..buildlib import Builder
    found = _ground.Found(vol)
    contract = _ground.Contract()
    for label, rect, level, what in pieces:
        contract.platform(label, tuple(int(v) for v in rect), int(level), cls="designed",
                          reason=f"{what} at y={int(level)}",
                          decision={"ground": what, "rect": [int(v) for v in rect]})
    resolved = contract.resolve(found.bed, found.wet,
                                relief=(Builder.SITE_RELIEF if relief is None else relief),
                                surface=found.surface)
    rec = resolved.record()
    rec["pieces"] = [{"label": label, "rect": [int(v) for v in rect],
                      "level": int(resolved.level_of(label)), "what": what}
                     for label, rect, level, what in pieces]
    return resolved, rec


def _dry_plateau(rnd, spec, site, voice, terra, n, x0, z0, m, grew=None) -> dict:
    """The plateau, offline: cut into the cached volume and the volume written back.

        The one stage of a dry run that changes the ground it is scored on, and it has to,
        because the alternative is planning a palace compound on a hillside the live run
        will have levelled. What it must not do is touch the world: this writes to
        `<state>/<base_volume>` and keeps the volume as it was beside it, so the cut is
        reversible in a way ground work in the world never is (open thread 18).
        
    """
    from ..buildlib import Builder
    import shutil
    p = rnd.rel("plateau.json")
    vp = rnd.rel(rnd.base_volume)
    if not os.path.exists(vp):
        return {"status": "error", "stop": True,
                "error": f"a dry run cuts its plateau into "
                         f"{os.path.relpath(vp, _pipeline.ROOT)} and there is none"}
    before = rnd.rel(rnd.base_volume.replace(".npz", ".before-plateau.npz"))
    if not os.path.exists(before):
        shutil.copyfile(vp, before)
    vol = offline.load_volume(before)
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    # **The podium is a declaration of the ground contract** (v2, B1): designed ground
    # at the level the search or the terrace arithmetic chose, settled and its seams
    # derived before the cut is made. One piece; the same resolver as a city's pads.
    rect = (x0, z0, x0 + n - 1, z0 + n - 1)
    settled, srec = settle_designed(vol, [(terra["part"], rect, int(m["y"]), "podium")])
    # The bound is the **place's**, not the registered 128: a quarter of the side of the
    # site (`Builder.plateau_max`). Found by running it.
    rec = b.plateau(rect, int(settled.level_of(terra["part"])),
                    mat=pipeline_voice(voice), label=terra["part"],
                    bound=Builder.plateau_max(int(site["size"])))
    cut = dict(b._pending)
    if rec.get("ok") and cut:
        offline.save_volume(vol.overlay(cut), vp)
    out = {"part": terra["part"], "voice": voice, "plateau": rec,
           "placed": len(cut) if rec.get("ok") else 0,
           "rect": [x0, z0, x0 + n - 1, z0 + n - 1], "compound_ground": grew,
           "contract": {k: srec[k] for k in ("columns", "seam_totals", "pieces",
                                             "registered")},
           "terrace": m.get("terrace"),
           "dry_run": True, "volume": os.path.relpath(vp, _pipeline.ROOT),
           "volume_before": os.path.relpath(before, _pipeline.ROOT),
           "scope": terra.get("scope", "core"),
           "note": "offline: the cut is applied to the volume this run is scored "
                   "against and to nothing else; no block is written to the world"}
    json.dump(out, open(p, "w"), indent=1)
    print(f"   plateau (dry): {rec.get('reason')}", flush=True)
    # The briefing describes ground this stage has just changed, so it is written again
    # from the volume as it now is.
    sp = rnd.rel("site.json")
    if os.path.exists(sp):
        os.replace(sp, rnd.rel("site.before-plateau.json"))
        X, Z = site["origin"]
        S = int(site["size"])
        s = site_brief_from_volume(offline.load_volume(vp), X, Z, S)
        was = json.load(open(rnd.rel("site.before-plateau.json")))
        if was.get("search"):
            s["search"] = was["search"]
        json.dump(s, open(sp, "w"), indent=1)
        out["site_rewritten"] = {"path": sp, "relief": s["stats"]["relief"],
                                 "was": was["stats"]["relief"]}
        out["site_before"] = rnd.rel("site.before-plateau.json")
    json.dump(out, open(p, "w"), indent=1)
    return out


def _find_site_module():
    """`scripts/find_site.py` as a module. It is a script and this is a stage."""
    import importlib.util
    p = os.path.join(_pipeline.ROOT, "scripts", "find_site.py")
    s = importlib.util.spec_from_file_location("find_site", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _dry_site_search(rnd) -> dict:
    """The site search with no server: every square `out/sites/` holds, ranked."""
    fs = _find_site_module()
    spec = rnd.place_spec()
    got = fs.search_fresh(spec, editor=None, log=lambda *a: None)
    got["generated_by"] = "pipeline.stage_site_search --dry-run"
    got["attempt"] = "dry run, off the cached grid"
    return got


def site_brief_from_volume(vol, X: int, Z: int, S: int) -> dict:
    """The terrain briefing, measured off a cached volume rather than a server.

        The same numbers `_commands/prepare_settlement.py` reads and in the same shape,
        including the transpose its own comment is about: the grid is stored the way its
        caption reads, because the earliest settlement plans were each made against a map
        of the site rotated a quarter turn.
        
    """
    import numpy as np
    from .. import observe
    h, _wet = observe.ground_heights(vol)
    i, j = X - vol.x0, Z - vol.z0
    inner = h[i:i + S, j:j + S].astype(int)
    cell = S // 12
    n = cell * 12
    box = inner[:n, :n]
    mean_g = box.reshape(12, cell, 12, cell).mean(axis=(1, 3)).round().astype(int).T
    max_g = box.reshape(12, cell, 12, cell).max(axis=(1, 3)).astype(int).T
    min_g = box.reshape(12, cell, 12, cell).min(axis=(1, 3)).astype(int).T
    names = [s.split("[")[0] for s in vol.palette]
    surface: dict = {}
    for gx in range(0, S, 12):
        for gz in range(0, S, 12):
            y = int(inner[gx, gz]) - vol.y0
            if not (0 <= y < vol.codes.shape[1]):
                continue
            b = names[int(vol.codes[i + gx, y, j + gz])]
            surface[b] = surface.get(b, 0) + 1
    checks = [{"x": X + px, "z": Z + pz, "world_y": int(inner[px, pz]),
               "grid_cell": [pz // cell, px // cell],
               "grid_y": int(mean_g[pz // cell][px // cell])}
              for (px, pz) in ((S // 6, S // 6), (S // 2, S // 6),
                               (S // 6, S // 2), (5 * S // 6, S // 6))]
    return {"origin": [int(X), int(Z)], "size": int(S),
            "stats": {"min": int(inner.min()), "max": int(inner.max()),
                      "relief": int(inner.max() - inner.min()),
                      "std": round(float(inner.std()), 1)},
            "mean_grid": mean_g.tolist(),
            "roughness_grid": (max_g - min_g).tolist(),
            "surface_blocks": surface,
            "orientation_check": checks,
            "grid": {"cell": int(cell), "summarised": int(n),
                     "note": f"the 12x12 grid summarises the {n}x{n} inside this "
                             f"{S}x{S} site; {S - n} column(s) on the east and south "
                             f"edges are described by the cell beside them"},
            "read_from": "a cached volume, offline"}


def stage_plateau(rnd, be, results: dict) -> dict:
    """**After the site briefing and before the plan**, and both halves of that matter.
    After, because the voice this ground is faced in is chosen against the site's own
    surface census and there is no census until `stage_site` has read one -- a plateau
    in somebody else's stone is the ground work the type layer spent three rounds
    learning not to do. Before the plan, because the grids a planner reads have to
    describe the ground **as it stands**: a planner told the middle of its site is a
    hillside, which has since been levelled, plans round a hill that is not there.
    So the briefing this stage invalidates, it writes again."""
    from ..buildlib import Builder
    p = rnd.rel("plateau.json")
    if os.path.exists(p):
        got = json.load(open(p))
        return {"skipped": "already cut -- ground work is not idempotent", **got}
    got = rnd.site_search() or {}
    terra = got.get("terraform")
    spec = rnd.place_spec()
    if not terra and spec and spec_mod_rings(spec):
        # The rings rise to it whether or not the search had to level anything, so the
        # part at the centre is marked here, by the same rule the search marks it.
        from .. import spec as spec_mod
        core = spec_mod.core(spec)
        if core is not None:
            terra = {"part": core["name"], "family": core.get("family"),
                     "kind": core.get("kind"), "plateau": int(_find_site_module()
                                                              ._plateau_size(spec)),
                     "relief_allowed": None, "scope": "core",
                     "why": "a concentric place's podium is designed ground: the "
                            "rings rise to it, so it is cut at the site's centre "
                            "whether or not the search had to level anything"}
    if not terra:
        return {"skipped": "the site search marked no part for terraforming: the "
                           "ground it chose is flat enough at the footprint"}
    dry = not be.live and getattr(be, "dry_run", False)
    if not be.live and not dry:
        return {"status": "error", "stop": True,
                "error": "a plateau moves ground; run this stage --live"}
    site = pipeline_site(rnd)
    voice = place_voice(rnd, spec, site)
    m = (got.get("chosen") or {}).get("measures", {}).get("plateau") or \
        {"size": int(terra["plateau"]), "x": int(site["origin"][0]),
         "z": int(site["origin"][1]), "y": None, "relief": None}
    S = int(site["size"])
    n = int(min(terra["plateau"], Builder.plateau_max(S)))
    x0, z0 = int(m["x"]), int(m["z"])
    # `terraform.plateau` is `needs.plateau`, which a spec call writes about a *part*
    # before any type exists, and the compound answering that part is then held inside
    # it. The search applies the same rule when it ranks (`find_site._plateau_size`),
    # and it is applied again here because this is where the cut is actually made and a
    # round may be handed a search record another round wrote. A place with no compound
    # at its centre is untouched: `compound_cut` returns the number it was given.
    from .. import placeplan as _placeplan
    grew = None
    n, x0, z0, m = compound_cut(rnd, spec, site, terra, n, x0, z0, m)
    if n != int(min(terra["plateau"], Builder.plateau_max(S))):
        grew = {"was": int(terra["plateau"]), "now": n,
                "why": _placeplan.compound_ground(spec=spec, site_side=S)["why"],
                "window": [x0, z0, x0 + n - 1, z0 + n - 1],
                "relief": m.get("relief"), "y": m.get("y")}
        print(f"   plateau: a compound's ground is the library's -- {grew['was']} -> "
              f"{n}: {grew['why']}", flush=True)
    if dry:
        # **The cut goes into the volume the run is scored against, and nowhere else.**
        # A dry run that skipped the plateau would plan a palace on a hillside and then
        # measure it on ground the live run will not have; a dry run that wrote to the
        # world would be the live write the spec forbids until every bar holds. So the
        # ground is moved in the cache and the cache is written back, with the volume as
        # it was kept beside it.
        out = _dry_plateau(rnd, spec, site, voice, terra, n, x0, z0, m, grew)
        be.refresh()
        return out
    b = Builder(be.site)
    rect = (x0, z0, x0 + n - 1, z0 + n - 1)
    settled, srec = settle_designed(be.volume, [(terra["part"], rect, int(m["y"]), "podium")])
    rec = b.plateau(rect, int(settled.level_of(terra["part"])),
                    mat=pipeline_voice(voice), label=terra["part"],
                    bound=Builder.plateau_max(int(site["size"])))
    placed = be.commit(b) if rec.get("ok") else {"placed": 0}
    # **Ground is published where it is cut.** v2, A1: a commit no longer writes
    # through, and this cut is read back out of the *server* by the re-briefing below --
    # `prepare_settlement.py` in its own process -- so a plateau left in the volume
    # would be a plan made against the hillside it was levelled out of. A1 moves the
    # per-part round trips, not the ground passes: there is one plateau.
    if be.live:
        be.publish()
    out = {"part": terra["part"], "voice": voice, "plateau": rec,
           "placed": placed.get("placed"), "compound_ground": grew,
           "terrace": m.get("terrace"),
           "rect": [x0, z0, x0 + n - 1, z0 + n - 1],
           "contract": {k: srec[k] for k in ("columns", "seam_totals", "pieces",
                                             "registered")}}
    json.dump(out, open(p, "w"), indent=1)
    be.rebind()
    print(f"   plateau: {rec.get('reason')}", flush=True)
    # The briefing describes ground this stage has just changed. Written again, from the
    # world as it now is, and the old one kept beside it.
    sp = rnd.rel("site.json")
    if os.path.exists(sp):
        os.replace(sp, rnd.rel("site.before-plateau.json"))
        out["site_rewritten"] = stage_site(rnd, be, results)
        out["site_before"] = rnd.rel("site.before-plateau.json")
    json.dump(out, open(p, "w"), indent=1)
    return out


def spec_mod_rings(spec: dict) -> list:
    from .. import spec as spec_mod
    return spec_mod.rings(spec)


#: 5.4 microseconds), and the most blocks the terraces of one place may lay: twelve
#: million, which the write stage puts into the world in about four minutes at the
#: concentric run's rate. The preflight estimates every ring's fill and cut off the
#: volume before a block is laid, prints both, and refuses over the bound.
COST_TERRACE_BLOCK_S = 6e-6
TERRACES_BOUND_BLOCKS = 12_000_000


def preflight_terraces(vol, layout: dict, reach: int = 2,
                       bound: int = TERRACES_BOUND_BLOCKS, log=print) -> dict:
    """The fill and the cut every ring's terrace will take, off the volume, before
    any is laid: for each annulus the sum over its columns of the level less the
    ground where the ground is lower and the ground less the level where higher."""
    import numpy as np
    from .. import observe
    h, wet = observe.ground_heights(vol)
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    rings = layout.get("rings") or []
    comp = layout.get("compound_rect")
    per = []
    total_fill = total_cut = total_cols = 0
    for k, r in enumerate(rings):
        if r.get("level") is None:
            continue
        oh = int(r["outer"]) + reach
        ih = (int(rings[k - 1]["outer"]) + reach if k else
              (max(comp[2] - comp[0], comp[3] - comp[1]) // 2 if comp else 0))
        i0, i1 = cx - oh - vol.x0, cx + oh - vol.x0 + 1
        j0, j1 = cz - oh - vol.z0, cz + oh - vol.z0 + 1
        i0c, j0c = max(0, i0), max(0, j0)
        hh = h[i0c:i1, j0c:j1]
        xs = np.arange(i0c, i1) + vol.x0 - cx
        zs = np.arange(j0c, j1) + vol.z0 - cz
        cheb = np.maximum(np.abs(xs)[:, None], np.abs(zs)[None, :])
        mask = (cheb > ih) & (cheb <= oh)
        lvl = int(r["level"])
        fill = int(np.clip(lvl - hh[mask], 0, None).sum()) + int(mask.sum())
        cut = int(np.clip(hh[mask] - lvl, 0, None).sum())
        per.append({"ring": r["name"], "level": lvl, "columns": int(mask.sum()),
                    "fill": fill, "cut": cut,
                    "water_columns": int(np.asarray(wet)[i0c:i1, j0c:j1][mask].sum())})
        total_fill += fill
        total_cut += cut
        total_cols += int(mask.sum())
    est = total_fill + total_cut
    out = {"rings": per, "columns": total_cols, "fill": total_fill, "cut": total_cut,
           "estimated_blocks": est,
           "estimated_seconds": round(est * COST_TERRACE_BLOCK_S, 1),
           "bound_blocks": int(bound), "refused": est > bound,
           "registered": {"COST_TERRACE_BLOCK_S": COST_TERRACE_BLOCK_S,
                          "TERRACES_BOUND_BLOCKS": TERRACES_BOUND_BLOCKS}}
    log(f"   preflight: {len(per)} terrace(s) over {total_cols:,} columns, about "
        f"{total_fill:,} blocks of fill and {total_cut:,} of cut, "
        f"{out['estimated_seconds']}s, against a bound of {bound:,} blocks"
        + (" -- REFUSED" if out["refused"] else ""))
    return out


def stage_terraces(rnd, be, results: dict) -> dict:
    """**After the plan and before the circulation**: the rings of a concentric place are
    levelled to the terrace levels the layout recorded, outermost first, each ring one
    level and every ring a step above the one outside it.

        After the plan because the annuli are the layout's; before the circulation because
        the lanes are routed on the ground as it stands and a lane routed across a hillside
        that is then terraced is a lane in the air. Offline the fill goes into the volume
        the run is scored against (`world.before-plateau.npz` stays the world as it was);
        live it is committed ring by ring, each its own bounded call. A place with no
        rings, or a layout with no levels, is untouched and says so.
        
    """
    from .. import placeplan
    from ..buildlib import Builder
    p = rnd.rel("terraces.json")
    if os.path.exists(p):
        got = json.load(open(p))
        return {"skipped": "already terraced -- ground work is not idempotent", **got}
    plan = rnd.plan()
    layout = (plan or {}).get("layout") or {}
    terrace = layout.get("terrace")
    rings = layout.get("rings") or []
    if not rings or not terrace or any(r.get("level") is None for r in rings):
        return {"skipped": "no terrace levels on the plan's layout: the place is not "
                           "concentric, or was laid out with no ground to read"}
    dry = not be.live and getattr(be, "dry_run", False)
    if not be.live and not dry:
        return {"status": "error", "stop": True,
                "error": "a terrace moves ground; run this stage --live"}
    spec = rnd.place_spec()
    site = pipeline_site(rnd)
    voice = place_voice(rnd, spec, site)
    mat = pipeline_voice(voice)
    cx, cz = int(layout["centre"][0]), int(layout["centre"][1])
    comp = layout.get("compound_rect")
    #: the terrace runs under the wall on the ring's outer boundary: the wall's swept
    #: half-width and one column more, so the wall stands on the terrace and its
    #: retaining face is the column outside that
    reach = 2
    pre = preflight_terraces(rnd.volume() if dry else be.volume, layout, reach,
                             bound=int(rnd.flags.get("terraces_bound_blocks")
                                       or TERRACES_BOUND_BLOCKS))
    if pre["refused"]:
        out = {"status": "error", "stop": True, "preflight": pre,
               "error": f"the terraces preflight refused: about {pre['estimated_blocks']:,} "
                        f"blocks against {pre['bound_blocks']:,}"}
        json.dump(out, open(p, "w"), indent=1)
        return out
    from ..buildlib import max_blocks_for
    guard = max_blocks_for(int(site["size"]))
    records = []
    t0 = time.perf_counter()
    order = sorted(range(len(rings)), key=lambda k: -k)      # outermost first
    if dry:
        vp = rnd.rel(rnd.base_volume)
        before = rnd.rel(rnd.base_volume.replace(".npz", ".before-plateau.npz"))
        if not os.path.exists(before):
            import shutil
            shutil.copyfile(vp, before)
    def lay(rect, sides, level, name):
        """One bounded piece: laid and committed, or split in two and laid again."""
        if dry:
            vol = offline.load_volume(vp)
            b = Builder(offline.OfflineSite(vol))
            b._vol = vol
        else:
            b = Builder(be.site)
        b.max_blocks = guard
        rec = b.terrace_annulus(rect, level, mat=mat, label=name, sides=sides)
        if not rec.get("ok") and rec.get("estimated_blocks"):
            # over the bound: halve it along its longer axis, the cut between the halves
            # faced by neither, and lay each half
            x0, z0, x1, z1 = rect
            if x1 - x0 < 8 and z1 - z0 < 8:
                return [rec]
            if x1 - x0 >= z1 - z0:
                mid = (x0 + x1) // 2
                halves = [((x0, z0, mid, z1), sides - {"east"}),
                          ((mid + 1, z0, x1, z1), sides - {"west"})]
            else:
                mid = (z0 + z1) // 2
                halves = [((x0, z0, x1, mid), sides - {"south"}),
                          ((x0, mid + 1, x1, z1), sides - {"north"})]
            out = []
            for hr, hs in halves:
                out += lay(hr, hs, level, name)
            return out
        if dry:
            cut = dict(b._pending)
            if rec.get("ok") and cut:
                offline.save_volume(vol.overlay(cut), vp)
            rec["placed"] = len(cut) if rec.get("ok") else 0
        else:
            rec["placed"] = (be.commit(b) if rec.get("ok") else {"placed": 0}).get("placed")
            if be.live:
                be.publish()             # A1: ground is published where it is cut
            be.rebind()
        return [rec]

    all_sides = set(Builder.TERRACE_SIDES)
    # **Every ring's terrace is a declaration of the ground contract** (v2, B1): the
    # annulus as four strips, each a platform of designed ground at the ring's level,
    # declared outermost first and settled once -- every column owned by one ring, the
    # step between two rings a retaining face by the drop -- before a block is laid.
    # Each strip is then laid at its settled level, as a bounded call of its own facing
    # only the sides that are the annulus's outer edge: north and south the full width,
    # west and east between them.
    ring_pieces: dict = {}
    for k in order:
        r = rings[k]
        oh = int(r["outer"]) + reach
        outer = (cx - oh, cz - oh, cx + oh, cz + oh)
        if k == 0:
            inner = tuple(int(v) for v in comp) if comp else None
        else:
            ih = int(rings[k - 1]["outer"]) + reach
            inner = (cx - ih, cz - ih, cx + ih, cz + ih)
        if inner is None:
            pieces = [(outer, all_sides)]
        else:
            ix0, iz0, ix1, iz1 = inner
            x0, z0, x1, z1 = outer
            pieces = [((x0, z0, x1, iz0 - 1), {"north", "west", "east"}),
                      ((x0, iz1 + 1, x1, z1), {"south", "west", "east"}),
                      ((x0, iz0, ix0 - 1, iz1), {"west"}),
                      ((ix1 + 1, iz0, x1, iz1), {"east"})]
        ring_pieces[k] = {"outer": outer, "inner": inner, "level": int(r["level"]),
                          "pieces": [(f"{r['name']}/{i}", rect, sides)
                                     for i, (rect, sides) in enumerate(pieces)]}
    declared = [(label, rect, ring_pieces[k]["level"], "terrace")
                for k in order for (label, rect, _s) in ring_pieces[k]["pieces"]]
    settled, srec = settle_designed(offline.load_volume(vp) if dry else be.volume, declared)
    print(f"   ground contract: {srec['columns']:,} columns of terrace settled over "
          f"{len(declared)} pieces; seams {srec['seam_totals']}", flush=True)
    for k in order:
        r = rings[k]
        outer, inner = ring_pieces[k]["outer"], ring_pieces[k]["inner"]
        level = ring_pieces[k]["level"]
        t1 = time.perf_counter()
        recs = []
        for label, rect, sides in ring_pieces[k]["pieces"]:
            recs += lay(rect, sides, int(settled.level_of(label)), r["name"])
        placed = sum(int(q.get("placed") or 0) for q in recs)
        ok = all(q.get("ok") for q in recs)
        rec = {"ok": ok, "pieces": len(recs),
               "columns": sum(int(q.get("columns") or 0) for q in recs),
               "filled": sum(int(q.get("filled") or 0) for q in recs),
               "flooded": sum(int(q.get("flooded") or 0) for q in recs),
               "cut": sum(int(q.get("cut") or 0) for q in recs),
               "retained": sum(int(q.get("retained") or 0) for q in recs),
               "feathered": sum(int(q.get("feathered") or 0) for q in recs),
               "estimated_blocks": max(int(q.get("estimated_blocks") or 0) for q in recs),
               "blocks": sum(int(q.get("blocks") or 0) for q in recs),
               "reason": (f"a terrace of {sum(int(q.get('columns') or 0) for q in recs)} "
                          f"columns at y={level} in {len(recs)} piece(s): "
                          f"{sum(int(q.get('filled') or 0) for q in recs)} filled, "
                          f"{sum(int(q.get('flooded') or 0) for q in recs)} from a bed "
                          f"under water, {sum(int(q.get('cut') or 0) for q in recs)} cut, "
                          f"{sum(int(q.get('retained') or 0) for q in recs)} retained, "
                          f"{sum(int(q.get('feathered') or 0) for q in recs)} feathered"
                          if ok else "; ".join(str(q.get("reason")) for q in recs
                                               if not q.get("ok"))),
               "piece_records": recs}
        records.append({"ring": r["name"], "index": k, "level": level,
                        "outer": list(outer), "inner": list(inner) if inner else None,
                        "placed": placed, "terrace": rec,
                        "seconds": round(time.perf_counter() - t1, 1)})
        print(f"   terrace {r['name']} at y={level}: {rec.get('reason')}", flush=True)
        if not ok:
            out = {"status": "error", "stop": True, "rings": records,
                   "error": f"the terrace of {r['name']} was refused: {rec.get('reason')}"}
            json.dump(out, open(p, "w"), indent=1)
            return out
    # **Every gate's approach**, after the rings: the level run outside the gate and the
    # ramp down to the ring beyond, the same pieces the road was planned on
    # (`placeplan.gate_approach_pieces`, read here off the volume as the rings left it).
    # Each piece is a small terrace of its own, faced on its far and lateral sides and
    # open toward the gate; the ring's level and the step it stands on stay exactly what
    # they were. The second dry run had the road climb the terrace step at the wall
    # line, under the gate.
    from .. import observe
    approaches = []
    all_parts = _pipeline.plan_parts(plan)
    decls_now = type_declarations(all_parts)
    gates = placeplan.approach_points(layout, [
        q for q in all_parts if q.get("kind") == "point"
        and (decls_now.get(q.get("type")) or {}).get("passage")])
    if gates:
        vol_now = offline.load_volume(rnd.rel(rnd.base_volume)) if dry else be.volume
        h_now, _wet = observe.ground_heights(vol_now)
        # ...declared with the rings and settled by the same contract. The approaches
        # are read off the volume as the rings left it -- a ramp's foot is where it
        # meets the feathered ground outside the wall, which is on the ground only once
        # the rings are -- so they are declared after the rings are laid and before any
        # approach is: the one resolution of both is the record.
        found_aps = []
        for g in gates:
            ap = placeplan.gate_approach_pieces(h_now, vol_now.x0, vol_now.z0, layout, g)
            if ap:
                found_aps.append(ap)
        ap_declared = [(f"{ap['gate']}_approach/{i}", tuple(piece["rect"]),
                        int(piece["level"]), "gate approach")
                       for ap in found_aps for i, piece in enumerate(ap["pieces"])]
        if ap_declared:
            settled, srec = settle_designed(vol_now, declared + ap_declared)
            print(f"   ground contract: {len(ap_declared)} approach piece(s) settled with "
                  f"the rings; seams {srec['seam_totals']}", flush=True)
        for ap in found_aps:
            ox, oz = ap["outward"]
            lateral = {"north", "south"} if ox else {"west", "east"}
            far_side = ({1: "east", -1: "west"}[ox] if ox else {1: "south", -1: "north"}[oz])
            recs = []
            for n_piece, piece in enumerate(ap["pieces"]):
                last = n_piece == len(ap["pieces"]) - 1
                sides = set(lateral) | ({far_side} if last else set())
                recs += lay(tuple(piece["rect"]), sides,
                            int(settled.level_of(f"{ap['gate']}_approach/{n_piece}")),
                            f"{ap['gate']}_approach")
            ap["laid"] = [{"level": q.get("y"), "columns": q.get("columns"),
                           "ok": q.get("ok"), "placed": q.get("placed")} for q in recs]
            ap["placed"] = sum(int(q.get("placed") or 0) for q in recs)
            ap["ok"] = all(q.get("ok") for q in recs)
            approaches.append(ap)
            print(f"   approach {ap['gate']}: {len(ap['pieces'])} piece(s) from y={ap['inside']} "
                  f"to {ap['foot']}, {ap['placed']} blocks", flush=True)
            if not ap["ok"]:
                out = {"status": "error", "stop": True, "rings": records,
                       "approaches": approaches,
                       "error": f"the approach of {ap['gate']} was refused"}
                json.dump(out, open(p, "w"), indent=1)
                return out
    out = {"rings": records, "approaches": approaches, "terrace": terrace, "voice": voice,
           "preflight": pre,
           "contract": {**{k: srec[k] for k in ("columns", "owned_by_class", "seam_totals",
                                                 "pieces", "registered")},
                        "settled": "the rings before a block was laid; the gates' "
                                   "approaches with them, once the feathered ground "
                                   "their ramps meet was on the ground"},
           "registered": {"TERRACE_STEP": placeplan.TERRACE_STEP,
                          "TERRACE_MAX_BLOCKS": Builder.TERRACE_MAX_BLOCKS,
                          "TERRACES_BOUND_BLOCKS": TERRACES_BOUND_BLOCKS},
           "placed": sum(int(r["placed"] or 0) for r in records)
                     + sum(int(a.get("placed") or 0) for a in approaches),
           "seconds": round(time.perf_counter() - t0, 1), "dry_run": bool(dry)}
    # The briefing describes ground this stage has just changed: written again.
    sp = rnd.rel("site.json")
    if os.path.exists(sp):
        was = json.load(open(sp))
        X, Z = site["origin"]
        S = int(site["size"])
        vol = offline.load_volume(rnd.rel(rnd.base_volume)) if dry else be.volume
        s2 = site_brief_from_volume(vol, X, Z, S)
        if was.get("search"):
            s2["search"] = was["search"]
        json.dump(s2, open(sp, "w"), indent=1)
        out["site_rewritten"] = {"path": sp, "relief": s2["stats"]["relief"],
                                 "was": was["stats"]["relief"]}
    json.dump(out, open(p, "w"), indent=1)
    if dry:
        # The pieces were laid into the file, not into the backend's cached volume; the
        # circulation after this reads `be.volume` and must read the terraces.
        be.refresh()
    return out


def pipeline_site(rnd) -> dict | None:
    from .. import pipeline
    return pipeline.settlement_site(rnd)


def pipeline_voice(voice):
    """The palette the ground work is faced in, for a named voice."""
    from .. import pipeline
    return pipeline.voice_palette(voice or None)


def stage_site(rnd, be, results: dict) -> dict:
    """Live only; refuses to overwrite.

        `scripts/prepare_settlement.py`, as a stage. The site file is a fixture in exactly
        the sense the base volume is -- every grid the planner reads and every relief number
        quoted afterwards comes out of it -- so a round that re-made it between stages would
        be a round whose plan was made against a different map from the one on disk.
    """
    p = rnd.rel("site.json")
    if os.path.exists(p):
        s = json.load(open(p))
        return {"skipped": "already prepared -- the site briefing is a fixture",
                "path": p, "origin": s["origin"], "size": s["size"],
                "relief": s["stats"]["relief"]}
    site = rnd.site or rnd.chosen_site()
    if not be.live and getattr(be, "dry_run", False):
        if not site:
            return {"status": "error", "stop": True,
                    "error": "this round's config names no site and no site search has "
                             "chosen one"}
        X, Z = site["origin"]
        S = int(site["size"])
        vp = rnd.rel(rnd.base_volume)
        if not os.path.exists(vp):
            return {"status": "error", "stop": True,
                    "error": f"a dry run measures the site off {os.path.relpath(vp, _pipeline.ROOT)} "
                             f"and there is none: the ground under ({X},{Z}) "
                             f"{S}x{S} has to be read once, by a session that writes "
                             f"nothing"}
        s = site_brief_from_volume(offline.load_volume(vp), X, Z, S)
        search = rnd.site_search()
        if search:
            s["search"] = {"attempt": search["attempt"],
                           "top": search.get("top", [])[:3],
                           "terraform": search.get("terraform"),
                           "size_band_dropped": search.get("size_band_dropped"),
                           "record": rnd.rel("site_search.json")}
        json.dump(s, open(p, "w"), indent=1)
        return {"path": p, "origin": s["origin"], "size": s["size"],
                "relief": s["stats"]["relief"], "search": bool(search),
                "dry_run": True}
    if not be.live:
        return {"error": "preparing a site reads the server; run this stage --live"}
    if not site:
        return {"error": "this round's config names no site and no site search has "
                         "chosen one"}
    import subprocess
    X, Z = site["origin"]
    S = site["size"]
    r = subprocess.run(
        [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
         os.path.join(_pipeline.ROOT, "src", "ethoslm", "pipeline", "_commands", "prepare_settlement.py"),
         str(X), str(Z), str(S)],
        env=dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name), capture_output=True, text=True)
    out = {"returncode": r.returncode, "log_tail": (r.stdout or r.stderr)[-800:]}
    if os.path.exists(p):
        s = json.load(open(p))
        # Why this ground, beside the ground. The briefing is the one file every later
        # pass reads about the site, and a site chosen by a search whose reasoning is in
        # a different file is a site whose reasoning nothing reads.
        search = rnd.site_search()
        if search:
            s["search"] = {"attempt": search["attempt"],
                           "top": search.get("top", [])[:3],
                           "terraform": search.get("terraform"),
                           "size_band_dropped": search.get("size_band_dropped"),
                           "record": rnd.rel("site_search.json")}
            json.dump(s, open(p, "w"), indent=1)
        out.update(path=p, origin=s["origin"], size=s["size"],
                   relief=s["stats"]["relief"], search=bool(search))
    else:
        out["error"] = "prepare_settlement.py wrote no site.json"
    return out


def stage_plan(rnd, be, results: dict) -> dict:
    """The plan. One call per level where the round has a place spec; A5,"""
    spec = rnd.place_spec()
    if spec is None:
        return stage_plan_flat(rnd, be, results)
    return stage_plan_levels(rnd, be, results, spec)


def _record_level(rnd, level: str, fails: list, checked: list) -> int:
    """Write this level's attempt to the one validation log, and say which attempt.

        A **pass** is recorded once and only once. The stage is re-entered every time the
        wait loop asks again -- that is what drives a plan of four calls through one stage --
        so a level that has already passed would otherwise write a row on every poll and the
        log would say a place was validated forty times.
        
    """
    p = rnd.rel("plan_validation.json")
    was = json.load(open(p)) if os.path.exists(p) else {"attempts": []}
    mine = [a for a in was["attempts"] if a.get("level") == level]
    if not fails and mine and not mine[-1]["failures"]:
        return len(mine)
    n = len(mine) + 1
    was["attempts"].append({"level": level, "attempt": n, "failures": fails,
                            "checked": checked, "leaves": None,
                            "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
    os.makedirs(rnd.state, exist_ok=True)
    json.dump(was, open(p, "w"), indent=1)
    return n


def _hand_back_level(rnd, brief: str, answer: str, level: str, attempt: int,
                     fails: list, decls: dict) -> None:
    """Put a level's failures at the end of its own brief and move its answer aside."""
    from .. import pipeline
    if os.path.exists(answer):
        os.replace(answer, answer.replace(".json", f".rejected.{attempt}.json"))
    lines = ["", "", "---", "",
             f"## Your {level} plan was checked and it does not build", "",
             f"{len(fails)} thing(s) in what you wrote cannot be built as drawn. "
             "Nothing has been built and nothing is in the world. Write the file "
             "again, in the same place, fixing these -- and only these -- and keeping "
             "everything that is not named here as it was.", ""]
    for f in fails:
        lines.append(f"- **{f.get('part')}** ({f.get('check')}) — {f['why']}")
    if decls:
        lines += ["", "The sizes every type needs, again, as **plots** with the "
                      "library's inset already added:", "",
                  pipeline.needs_table(decls), ""]
    with open(brief, "a") as fh:
        fh.write("\n".join(lines) + "\n")


def district_asks(rnd, spec: dict, site: dict, place: dict, types, voice) -> dict:
    """Every district not yet planned, each with its brief written, as one
    `needs_model` entry per district -- the batch the driver hands out at once."""
    from .. import placeplan
    asks = {}
    n = len(place.get("districts") or [])
    for d in (place.get("districts") or []):
        db = rnd.rel(f"district_{d['name']}_prompt.md")
        dp = rnd.rel(f"plan.district.{d['name']}.json")
        # v2, C1: **a district with a character is compiled, and no model is asked.**
        # The file the compiler writes is the one a model used to write, at the same
        # seam, held to the same validator below; the record says what it laid.
        part = placeplan._district_part(spec, d)
        # ...and v2, C5: where the compiler's answer was refused and the character's
        # author has written a new one, it is taken here, before anything is compiled.
        if apply_character(rnd, spec, part):
            for f in (dp, rnd.rel(f"district_{d['name']}_compiled.json")):
                if os.path.exists(f):
                    os.remove(f)
        if not os.path.exists(dp) and spec_mod.character(part) is not None:
            from .. import district_compile
            os.makedirs(rnd.state, exist_ok=True)
            role = spec_mod.district_role(spec, d)
            _t, mine = placeplan.types_card(types, spec.get("form"), role)
            got, rec = district_compile.compile_district(
                d, part, place, mine, spec=spec,
                seed=int(rnd.flags.get("seed") or 1))
            json.dump(got, open(dp, "w"), indent=1)
            json.dump(rec, open(rnd.rel(f"district_{d['name']}_compiled.json"), "w"),
                      indent=1)
            print(f"   district {d['name']}: compiled from its character -- "
                  f"{rec['lots']} lots on {rec['blocks']} blocks, plot cover "
                  f"{rec['plot_cover']:.0%}, {rec['undeveloped_share']:.0%} "
                  f"undeveloped", flush=True)
            continue
        if not os.path.exists(db):
            os.makedirs(rnd.state, exist_ok=True)
            open(db, "w").write(
                placeplan.district_brief(spec, site, d, place, dp, types, voice))
        if not os.path.exists(dp):
            asks[f"plan/district/{d['name']}"] = {
                "status": "needs_model", "role": "plan", "request": db, "write": dp,
                "level": f"district/{d['name']}",
                "note": f"one district of {n}: the plots inside {d['name']} and "
                        f"nothing else; asked as a batch"}
    return asks


CHARACTER_BRIEF = """# A district was compiled from your character and refused

> {sentence}

The district **{district}** of `{part}` was laid out from the character you wrote --
no model drew its plots -- and what came out does not pass the validator every
district is held to. Nothing has been built and nothing is in the world.

## The district

x {x0}..{x1}, z {z0}..{z1} -- {w} by {d} blocks, {columns} columns. It is a
**{density}** district and it is **{role}**, and it was asked for **{structures}**
structures.

## The character you wrote

{character}

## What the compiler laid from it

{laid}

## What it is short of

{failures}

## What a character can change

{fields}

The defaults per density word, which fill any field you leave out:

{defaults}

**What is actually going on, in the compiler's own arithmetic.** A district's ground
goes to three things: the lots, the open ground between them, and the lanes. Lots a
**lane** apart (`frontage: "open"`) leave five blocks between neighbours that nothing
can be planted in, because a piece of open ground has to keep three clear of a
building -- so an open fabric spends a quarter of its ground on lanes and cannot be
covered past about three fifths. Lots a **clearance** apart (`frontage: "street"`)
pack tighter, and whole blocks given to open ground (`open_share`) or to a court
behind each row (`courtyard_share`) are what the cover is actually made of, because a
big piece of open ground tiles far better than the scraps between houses. A longer
block (`block`) is fewer streets. The compiler already lengthens the block and raises
the shares as far as it can on its own; what it cannot do is change what kind of
fabric you asked for.

## Output

Write a single JSON file to {out}:

{{"character": {{...the whole character for this part, every field you want...}},
  "why": "one or two sentences on what you changed and why"}}

Only this part's character changes; every other part of the spec, the site, the size
and the voice stand. This is asked **once**: a second refusal stops the run.
"""


def apply_character(rnd, spec: dict, part: dict) -> bool:
    """Take a character the author wrote after a refusal onto the spec on disk. v2, C5.

        True where one was applied, so the caller drops what was compiled from the old
        one. Refused by name into the record where it does not read.
        
    """
    if not part or not spec_mod.district(part):
        return False
    p = rnd.rel(f"character.{part['name']}.json")
    if not os.path.exists(p):
        return False
    try:
        doc = json.load(open(p))
    except ValueError:
        return False
    got = doc.get("character")
    probe = dict(part)
    try:
        spec_mod.read_character(got, probe, f"the character for {part['name']}")
    except spec_mod.SpecError as e:
        doc["refused"] = str(e)
        json.dump(doc, open(p, "w"), indent=1)
        return False
    if probe.get("character") == part.get("character"):
        return False
    # **Taken once.** The answer stays on disk as the record of what was asked and
    # answered, and `applied` is what was taken from it; without this the file is
    # applied again on every re-entry and undoes anything written over it later -- a
    # hand-back from the plan stage silently put back the character the preview's
    # revision had just replaced.
    if doc.get("applied") == probe["character"]:
        return False
    raw_p = (rnd.rel("place.checked.json")
             if os.path.exists(rnd.rel("place.checked.json")) else rnd.rel("place.json"))
    raw = json.load(open(raw_p))
    for rp in raw.get("defining_parts") or []:
        if rp.get("name") == part["name"]:
            rp["character"] = probe["character"]
    json.dump(raw, open(raw_p, "w"), indent=1)
    part["character"] = probe["character"]
    for q in spec.get("defining_parts") or []:
        if q.get("name") == part["name"]:
            q["character"] = probe["character"]
    doc["applied"] = probe["character"]
    json.dump(doc, open(p, "w"), indent=1)
    print(f"   character: {part['name']} taken from its author's second answer -- "
          f"{json.dumps(probe['character'])}", flush=True)
    return True


def character_hand_back(rnd, spec: dict, d: dict, part: dict, fails: list,
                        n: int) -> dict:
    """Hand one compiled district's refusal back to the character's author. v2, C5."""
    from .. import placeplan
    x0, x1 = min(d["x0"], d["x1"]), max(d["x0"], d["x1"])
    z0, z1 = min(d["z0"], d["z1"]), max(d["z0"], d["z1"])
    rec_p = rnd.rel(f"district_{d['name']}_compiled.json")
    rec = json.load(open(rec_p)) if os.path.exists(rec_p) else {}
    laid = "  (the compiler wrote no record)"
    if rec:
        laid = "\n".join([
            f"  - **{rec['lots']} lots** of {rec['lot'][0]}x{rec['lot'][1]} "
            f"({rec['house']}) on {rec['blocks']} block(s) of {rec['block']}, "
            f"{rec['block_kinds']}",
            f"  - the plots cover **{rec['plot_cover']:.0%}** of the district and the "
            f"plots and the open ground together **{rec['ground_cover']:.0%}**; "
            f"**{rec['undeveloped_share']:.0%}** is assigned to nothing",
            f"  - every column: {rec['assigned']}",
            f"  - it already tried: " + ", ".join(
                f"open {t['open_share']:g}/court {t['courtyard_share']:g}"
                f"/block {t['block']} -> {t['lots']} lots, ground "
                f"{t['ground_cover']:.0%}" for t in (rec.get("tries") or [])[:8]),
        ])
    fields = "\n".join(f"  - `{k}`" for k in spec_mod.CHARACTER_FIELDS)
    fields += ("\n\n`frontage` is one of " + ", ".join(f"`{f}`" for f in spec_mod.FRONTAGES)
               + "; `block` and `lot_depth` are whole numbers of columns; `attached` is "
                 "true or false; the two shares are 0 to 1; `landmarks` is a list of "
                 "{\"type\": a type name}.")
    out_p = rnd.rel(f"character.{part['name']}.json")
    brief_p = rnd.rel(f"character_{d['name']}_prompt.md")
    open(brief_p, "w").write(CHARACTER_BRIEF.format(
        sentence=(spec.get("sentence") or rnd.sentence or ""), district=d["name"],
        part=part["name"], x0=x0, x1=x1, z0=z0, z1=z1, w=x1 - x0 + 1, d=z1 - z0 + 1,
        columns=(x1 - x0 + 1) * (z1 - z0 + 1),
        density=part.get("density") or "medium", role=part.get("role") or "urban",
        structures=d.get("structures"),
        character=json.dumps(part.get("character"), indent=1),
        laid=laid,
        failures="\n".join(f"  - **{f.get('check')}** -- {f['why']}" for f in fails),
        fields=fields,
        defaults=json.dumps(spec_mod.CHARACTER_DEFAULTS, indent=1), out=out_p))
    return {"plan": {"status": "needs_model", "role": "spec", "request": brief_p,
                     "write": out_p, "level": f"district/{d['name']}", "attempt": n,
                     "failures": fails, "compiled": True,
                     "note": f"the district compiled from {part['name']}'s character "
                             f"fails its validator; the character's author is asked "
                             f"once for one that does not"}}


def stage_plan_levels(rnd, be, results: dict, spec: dict) -> dict:
    """Plan the place, then plan each district. A5."""
    from .. import pipeline, placeplan, spec as spec_mod
    site = pipeline.settlement_site(rnd)
    if not site:
        return {"plan": {"status": "error", "stop": True,
                         "error": "no site.json: a place is planned against ground and "
                                  "no site has been prepared"}}
    types = rnd.flags.get("types")
    # v2, C3: **the library grows when the spec asks for a form it lacks.** A defining
    # part no committed type builds is a type authored blind, checked and adopted here,
    # before the place is planned -- a run's worth of them and no more.
    from .. import growth
    gaps = growth.type_gaps(spec, types)
    if gaps:
        grown = growth.stage(rnd, be, spec, gaps, types=types, site=site)
        if grown is not None:
            return grown
    voice = place_voice(rnd, spec, site)
    _table, decls = placeplan.types_card(types, spec.get("form"))
    if not decls:
        return {"plan": {"status": "error", "stop": True,
                         "error": f"no committed type is of the form family "
                                  f"{spec.get('form')!r} this place asks for"}}

    # The ground levelled for a defining part, where the plateau stage cut one: its
    # rectangle and level go into the place brief and the compound answering that part
    # is held to it.
    plateau = plateau_record(rnd)

    # --- level 1: the place ------------------------------------------------
    pb = rnd.rel("place_plan_prompt.md")
    pp = rnd.rel("plan.place.json")
    # **The place level is solved, not asked for.** v2, B2: every defining part is
    # placed by its relation -- `placesolve.solve_place`, the ring arithmetic where the
    # spec declares rings and the relation solver everywhere else -- written to the same
    # file the planner used to write, and then held to the same validator as the proof.
    # There is no brief and no hand-back: nobody to hand it back to, and the model never
    # writes a coordinate at any level. A place the solver cannot lay out stops here, by
    # name.
    arithmetic = True
    if not os.path.exists(pp):
        from .. import placesolve
        os.makedirs(rnd.state, exist_ok=True)
        place, lfails = placesolve.solve_place(spec, site, plateau, decls, voice,
                                               vol=_plan_volume(rnd, be),
                                               seed=int(rnd.flags.get("seed") or 1))
        if lfails:
            _record_level(rnd, "place", lfails, ["rings", "centred", "shares",
                                                 "coverage", "type", "relation",
                                                 "vetoes", "district"])
            return {"plan": {"status": "error", "stop": True, "level": "place",
                             "arithmetic": True, "failures": lfails,
                             "error": "the place cannot be laid out on this site: "
                                      + "; ".join(f"{f.get('part')}: {f['why']}"
                                                  for f in lfails[:6])}}
        json.dump(place, open(pp, "w"), indent=1)
        lay = place.get("layout") or {}
        print(f"   place: {len(place['districts'])} districts, "
              f"{len(place['parts'])} parts and {len(place['compounds'])} compound(s) "
              + (f"laid out by arithmetic; gates on the {lay['axis_side']}"
                 if lay.get("axis_side") else
                 f"placed by relation" + (f"; {len(lay.get('demoted') or [])} veto(es) "
                                          f"demoted" if lay.get("demoted") else "")),
              flush=True)
    if not os.path.exists(pp):
        return {"plan": {"status": "error", "stop": True, "level": "place",
                         "error": "the place level was not written and nothing here "
                                  "asks a model for it"}}
    place = json.load(open(pp))
    place.setdefault("voice", voice)
    vol = _plan_volume(rnd, be)
    ground = pipeline.plan_ground(
        [{**p, "name": p.get("name")} for p in (place.get("parts") or [])], vol)
    # A4: **the roads between the districts, before the districts.** No model call --
    # the nodes are the gates and the district centres the place level has just drawn,
    # and the routing is the one the lanes get. It is written into the place plan, so
    # every district brief below can be told where its own road already runs.
    place["arterials"] = _stage_arterials(rnd, place, decls, vol)
    json.dump(place, open(pp, "w"), indent=1)
    fails = placeplan.place_failures(place, spec, site, decls, ground=ground, voice=voice,
                                     plateau=plateau)
    if fails:
        n = _record_level(rnd, "place", fails, ["spec", "wall", "gate", "site",
                                                "district", "compound", "scale",
                                                "plateau", "overlap", "leaf"])
        if n >= 2 or arithmetic:
            return {"plan": {"status": "error", "stop": True, "level": "place",
                             "attempt": n, "failures": fails, "arithmetic": arithmetic,
                             "error": ("the arithmetic's place level fails its own "
                                       "validator: " if arithmetic else
                                       "the place level fails validation twice: ")
                                      + "; ".join(f"{f.get('part')}: {f['why']}"
                                                  for f in fails[:6])}}
        _hand_back_level(rnd, pb, pp, "place", n, fails, decls)
        return {"plan": {"status": "needs_model", "role": "plan", "request": pb, "write": pp,
                         "level": "place", "attempt": n, "failures": fails,
                         "note": "the place level was checked and handed back once"}}
    _record_level(rnd, "place", [], ["spec", "wall", "gate", "site", "district",
                                     "compound", "scale", "plateau", "overlap", "leaf"]
                  + (["arithmetic"] if arithmetic else []))

    # --- level 2: each compound ------------------------------------------- A great
    # thing is a place inside the place, planned by a call of its own before the
    # districts are, out of the same kinds of part: its wall, its gates, its halls and
    # its courts. Validated and handed back once, exactly as a district is.
    compounds = {}
    for c in (place.get("compounds") or []):
        cb = rnd.rel(f"compound_{c['name']}_prompt.md")
        cp = rnd.rel(f"plan.compound.{c['name']}.json")
        # **An axial compound is laid by the library and no model is asked**, the craft
        # round (E5), at the same seam a compiled district is read from: a family whose
        # composition declares an axis is a *sequence* -- the gate, a forecourt, a hall,
        # an inner court and the greatest hall at the far end -- and an order is not
        # something a validator can ask for after the fact.
        if not os.path.exists(cp):
            cpart = next((d for d in spec_mod.compounds(spec)
                          if placeplan._answers(c, d)), None)
            _ct, cdecl = placeplan.compound_types(types, spec, cpart or {})
            laid, why = placeplan.compound_axial(c, cpart, place, cdecl or decls,
                                                 spec=spec, seed=1)
            if laid is not None:
                json.dump(laid, open(cp, "w"), indent=1)
                json.dump(why, open(rnd.rel(f"compound_{c['name']}_axial.json"), "w"),
                          indent=1)
        if not os.path.exists(cb) and not os.path.exists(cp):
            open(cb, "w").write(placeplan.compound_brief(
                spec, site, c, place, cp, types, voice, plateau=plateau))
        if not os.path.exists(cp):
            return {"plan": {"status": "needs_model", "role": "plan", "request": cb, "write": cp,
                             "level": f"compound/{c['name']}",
                             "note": f"one compound of {len(place['compounds'])}: the "
                                     f"wall, gates, halls and courts inside "
                                     f"{c['name']} and nothing else"}}
        got = json.load(open(cp))
        part = next((d for d in spec_mod.compounds(spec)
                     if placeplan._answers(c, d)), None)
        cparts = placeplan.compound_parts(got, part, c["name"], spec)
        _, cdecls = placeplan.compound_types(types, spec, part or {})
        cg = pipeline.plan_ground(cparts, _plan_volume(rnd, be))
        cfails = placeplan.compound_failures(c, got, place, cdecls or decls, ground=cg,
                                             spec=spec)
        if cfails:
            n = _record_level(rnd, f"compound/{c['name']}", cfails,
                              ["compound", "type", "form", "role", "footprint",
                               "ground", "overlap", "wall", "gate", "inside",
                               "composed"])
            if n >= 2:
                return {"plan": {"status": "error", "stop": True,
                                 "level": f"compound/{c['name']}", "attempt": n,
                                 "failures": cfails,
                                 "error": f"compound {c['name']} fails validation "
                                          f"twice: " + "; ".join(
                                              f"{f.get('part')}: {f['why']}"
                                              for f in cfails[:6])}}
            _hand_back_level(rnd, cb, cp, f"compound {c['name']}", n, cfails,
                             cdecls or decls)
            return {"plan": {"status": "needs_model", "role": "plan", "request": cb, "write": cp,
                             "level": f"compound/{c['name']}", "attempt": n,
                             "failures": cfails,
                             "note": "this compound was checked and handed back once"}}
        _record_level(rnd, f"compound/{c['name']}", [],
                      ["compound", "type", "form", "role", "footprint", "ground",
                       "overlap", "wall", "gate", "inside", "composed"])
        compounds[c["name"]] = got

    # one `needs_model` entry per district -- so the calls are issued as a batch of
    # isolated subagents rather than one per re-entry of this stage. The concentric
    # run's 24 districts were 24 re-entries in series; nothing in one district's plan
    # depends on another's.
    districts = {}
    asks = district_asks(rnd, spec, site, place, types, voice)
    if asks:
        first = next(iter(asks.values()))
        return {"plan": {**first, "batch": len(asks),
                         "note": f"{len(asks)} district(s) asked for as a batch; each "
                                 f"is its own entry below"}, **asks}
    for d in (place.get("districts") or []):
        db = rnd.rel(f"district_{d['name']}_prompt.md")
        dp = rnd.rel(f"plan.district.{d['name']}.json")
        got = json.load(open(dp))
        role = spec_mod.district_role(spec, d)
        plots = placeplan.district_plots(got, role, spec)
        dg = pipeline.plan_ground(plots, _plan_volume(rnd, be))
        dfails = placeplan.district_failures(d, got, place, decls, ground=dg,
                                             form=spec.get("form"), role=role,
                                             part=placeplan._district_part(spec, d),
                                             spec=spec)
        if dfails:
            n = _record_level(rnd, f"district/{d['name']}", dfails,
                              ["district", "count", "cover", "ground_cover",
                               "type", "role", "footprint", "ground", "overlap"])
            # v2, C1 and C5: a **compiled** district is not the district planner's to
            # hand back -- no model drew it -- but the **character** it was compiled
            # from is a model's, and that is who is answerable for a fabric the
            # validator refuses. So the refusal goes back to the character's author,
            # once, with what the compiler laid and what it was short of; refused twice,
            # the run stops by name.
            part = placeplan._district_part(spec, d)
            if spec_mod.character(part) is not None:
                if n >= 2:
                    return {"plan": {"status": "error", "stop": True,
                                     "level": f"district/{d['name']}", "attempt": n,
                                     "failures": dfails, "compiled": True,
                                     "error": f"the compiled district {d['name']} fails "
                                              f"its validator twice: " + "; ".join(
                                                  f"{f.get('part')}: {f['why']}"
                                                  for f in dfails[:6])}}
                return character_hand_back(rnd, spec, d, part, dfails, n)
            if n >= 2:
                return {"plan": {"status": "error", "stop": True,
                                 "level": f"district/{d['name']}", "attempt": n,
                                 "failures": dfails,
                                 "error": f"district {d['name']} fails validation "
                                          f"twice: " + "; ".join(
                                              f"{f.get('part')}: {f['why']}"
                                              for f in dfails[:6])}}
            # A2: **the role-filtered table, not the whole one.** The hand-back re-
            # states every type's needs so the planner can act on the failure, and
            # composing it from the place-level `decls` handed a rural district the
            # sizes of a keep, a wall and a minka -- types the role check would refuse
            # the moment it named one. A rule the planner cannot read is a hand-back
            # spent for nothing, and this is the second time in two rounds that the
            # brief and the refusal have had to be made to agree.
            _, mine = placeplan.types_card(types, spec.get("form"), role)
            _hand_back_level(rnd, db, dp, f"district {d['name']}", n, dfails,
                             mine or decls)
            return {"plan": {"status": "needs_model", "role": "plan", "request": db, "write": dp,
                             "level": f"district/{d['name']}", "attempt": n,
                             "failures": dfails,
                             "note": "this district was checked and handed back once"}}
        _record_level(rnd, f"district/{d['name']}", [],
                      ["district", "type", "footprint", "ground", "overlap"])
        districts[d["name"]] = got

    # --- assemble, validate the whole tree, and write the registry ---------
    plan = placeplan.assemble(place, districts, spec, compounds)
    json.dump(plan, open(rnd.rel("plan.json"), "w"), indent=1)
    parts = pipeline.plan_parts(plan)
    whole = pipeline.plan_failures(
        parts, type_declarations(parts),
        ground=pipeline.plan_ground(parts, _plan_volume(rnd, be)),
        network=rnd.network(), form=spec.get("form"))
    if whole:
        # The levels each passed and the assembly does not, which is the seam A5 exists
        # to expose: two districts drawn far enough apart at the place level whose plots
        # meet at their shared edge. Reported and stopped rather than built.
        return {"plan": {"status": "error", "stop": True, "level": "assembled",
                         "failures": whole,
                         "error": "every level passed and the assembled plan does not: "
                                  + "; ".join(f"{f['part']}: {f['why']}"
                                              for f in whole[:6])}}
    return _write_registry(rnd, plan, parts, levels=plan["levels"], voice=voice)


def plateau_record(rnd) -> dict | None:
    """What `stage_plateau` cut, or None: the part it was cut for, its rectangle and
    its level. Read off `plateau.json` so a plan can be told where the ground is."""
    p = rnd.rel("plateau.json")
    if not os.path.exists(p):
        return None
    got = json.load(open(p))
    if not got.get("rect") or not (got.get("plateau") or {}).get("ok"):
        return None
    # the layout takes its levels from here rather than reading a median again off the
    # volume the cut has since changed. The run that read the record without this
    # planned rings one block above the plateau it stood on.
    return {"part": got.get("part"), "rect": [int(v) for v in got["rect"]],
            "y": (got.get("plateau") or {}).get("y"), "terrace": got.get("terrace")}


def place_voice(rnd, spec: dict | None, site: dict | None) -> str:
    """The voice this place is in, decided once and written down."""
    if rnd.voice:
        return rnd.voice
    p = rnd.rel("voice.json")
    if os.path.exists(p):
        return json.load(open(p))["voice"]
    if not (spec and site):
        return ""
    got = _choose_voice(spec, site)
    from .. import placeread, styles
    ok, says = placeread._palette(got, site)
    os.makedirs(rnd.state, exist_ok=True)
    json.dump({"voice": got, "chosen_by": "ethoslm.place._choose_voice",
               "why": says, "contrast_holds": bool(ok),
               "named_in_the_sentence": bool(spec.get("voice")),
               "considered": sorted(styles.VOICES),
               "note": "The sentence named none and "
                       "the ground chose one, by the rule a settlement on a terrace taught: a "
                       "town the colour of its own hillside"},
              open(p, "w"), indent=1)
    return got


def _choose_voice(spec: dict, site: dict, check_types: bool = True) -> str:
    """The voice, chosen against the ground. Deterministic, and no model call.

        `check_types` false skips the refusal below: the library's growth (v2, C3) needs
        the voice for a brief before the missing type exists.

        **Every voice covers every defining part now**, which is A1 in one function. This
        used to ask, of each of seven voices, whether the committed types declaring *that
        voice* could stand up the place's walls, gates and squares -- and the answer was
        usually no, because a wall existed in one voice only and writing a second wall meant
        writing the same wall again. A type is a form, the palette is handed to it, so what
        is left to ask is whether the committed types of this place's **form family** cover
        its defining parts, and then which palette reads against this ground.
        
    """
    from .. import placeread, styles
    if spec.get("voice") in styles.VOICES:
        return spec["voice"]
    from ..placeplan import types_card
    _, decls = types_card(form=spec.get("form"))
    typed = [{"type": n, "kind": d["kind"]} for n, d in decls.items()]
    missing = [p["name"] for p in spec["defining_parts"]
               if p["kind"] != "group" and not any(
                   t["kind"] == p["kind"] and (t["type"] == p["family"]
                     or t["type"].startswith(p["family"] + "_")) for t in typed)]
    if missing and check_types:
        raise ValueError(f"no committed type of the form family "
                         f"{spec.get('form') or 'any'} builds "
                         f"{', '.join(missing)}: the place cannot be planned until one "
                         f"is written")
    every = sorted(styles.VOICES)
    if not every:
        raise ValueError("there are no voices under voices/")
    # **A ceremonial palette is a candidate and never a default.** The craft round, E2:
    # a voice may declare itself the palette of a place's richest quarter, and this is
    # the choice made when nothing said -- a whole place in white stone under a gilded
    # roof has no centre. A spec call naming one for a ring is answered above, where a
    # named voice is returned whatever it says about itself.
    available = [v for v in every if not styles.VOICES[v]["ceremonial"]] or every
    ok = [v for v in available if placeread._palette(v, site)[0]]
    return (ok or available)[0]


def _write_registry(rnd, plan, parts, *, levels=None, voice=None) -> dict:
    from .. import pipeline
    decls = type_declarations(parts)
    rects = [pipeline.part_registry_row(
        q, passage=bool((decls.get(q.get("type")) or {}).get("passage")))
        for q in parts]
    json.dump(rects, open(rnd.rel("plots.json"), "w"), indent=1)
    kinds: dict = {}
    for q in parts:
        kinds[q.get("kind", "plot")] = kinds.get(q.get("kind", "plot"), 0) + 1
    return {"plan": {"status": "planned", "leaves": len(parts), "kinds": kinds,
                     "depth": max((len(q["in"]) for q in parts), default=0),
                     "quarters": sorted({q["in"][-1] for q in parts if q.get("in")}),
                     "types": sorted({q.get("type") for q in parts if q.get("type")}),
                     "levels": levels, "voice": voice,
                     "plots_written": rnd.rel("plots.json"),
                     "centre": plan.get("centre")}}


def stage_plan_flat(rnd, be, results: dict) -> dict:
    """The planner, asked for a tree of parts. A4.

        That last step is new and it is what A4 makes possible. `plots.json` has always been
        written by `reserve()` when a pass commits, because a plot was ground a builder
        claimed. Here every part of the place is in the plan before anything is built, so
        the plan *is* the registry -- and the lint, the walk model and the cards can be
        scoped to a wall or a square exactly as they are to a house.
        
    """
    from .. import pipeline
    import subprocess
    prompt = rnd.rel("planner_prompt.md")
    if rnd.state_dir and not os.path.exists(prompt):
        # `make_settlement_prompts.py` takes its output directory from $ETHOSLM_SETTLEMENT
        # at import, so it writes to `out/<name>` and cannot be pointed at a shadow. A
        # stage that cannot honour a shadowed state directory says so.
        return {"plan": {"status": "error", "error":
                         "stage_plan writes planner_prompt.md through "
                         "make_settlement_prompts.py, which takes its directory from "
                         "$ETHOSLM_SETTLEMENT and cannot be shadowed -- run it against "
                         "the round's own state, or write the prompt yourself first"}}
    if not os.path.exists(prompt):
        env = dict(os.environ, ETHOSLM_SETTLEMENT=rnd.name, ETHOSLM_TREE="1")
        if rnd.voice:
            env["ETHOSLM_VOICE"] = rnd.voice
        # Which types this place may be made of. Named by the round, because the
        # committed types are not all in one voice and a plan that reached for a
        # Japanese temple in a white-render district would be a confound rather than a
        # choice. Unset means every type on disk.
        if rnd.flags.get("types"):
            env["ETHOSLM_TYPES"] = ",".join(rnd.flags["types"])
        intent = (rnd.types or {}).get("intent_file") or rnd.flags.get("intent_file")
        if intent:
            env["ETHOSLM_INTENT"] = os.path.join(_pipeline.ROOT, intent)
        r = subprocess.run(
            [os.path.join(_pipeline.ROOT, ".venv", "bin", "python"),
             os.path.join(_pipeline.ROOT, "scripts", "make_settlement_prompts.py")],
            env=env, capture_output=True, text=True)
        if not os.path.exists(prompt):
            return {"plan": {"status": "error",
                             "error": (r.stderr or r.stdout)[-600:]}}
    p = rnd.rel("plan.json")
    if not os.path.exists(p):
        return {"plan": {"status": "needs_model", "role": "plan", "request": prompt,
                         "write": p,
                         "note": "the planner is asked for A4's tree of parts; every "
                                 "leaf names a type from types/ and a seed"}}
    plan = rnd.plan()
    parts = rnd.parts()
    bad = [q["name"] for q in parts
           if not os.path.exists(os.path.join(_pipeline.ROOT, "types", f"{q.get('type')}.py"))]

    # The plan is checked against what the types need **before** anything is built. A
    # failing plan goes back to the planner once, in the same brief, with the failing
    # leaves and their needs named; a second failure stops the round.
    val = validate(rnd, be, parts)
    if val["failures"]:
        if val["attempt"] >= 2:
            return {"plan": {"status": "error", "attempt": val["attempt"],
                             "failures": val["failures"], "stop": True,
                             "error": f"the plan fails validation twice: "
                                      f"{len(val['failures'])} leaf/leaves cannot be "
                                      f"built. " + "; ".join(
                                          f"{f['part']}: {f['why']}"
                                          for f in val["failures"][:6])}}
        return {"plan": {"status": "needs_model", "role": "plan", "request": prompt,
                         "write": p, "attempt": val["attempt"],
                         "failures": val["failures"],
                         "note": "the plan was checked against every type's NEEDS and "
                                 "returned to the planner once; the failing leaves are "
                                 "named at the end of the brief"}}

    decls = type_declarations(parts)
    rects = [pipeline.part_registry_row(
        q, passage=bool((decls.get(q.get("type")) or {}).get("passage")))
        for q in parts]
    json.dump(rects, open(rnd.rel("plots.json"), "w"), indent=1)
    kinds: dict = {}
    for q in parts:
        kinds[q.get("kind", "plot")] = kinds.get(q.get("kind", "plot"), 0) + 1
    return {"plan": {"status": "planned", "leaves": len(parts), "kinds": kinds,
                     "depth": max((len(q["in"]) for q in parts), default=0),
                     "quarters": sorted({tuple(q["in"]) and q["in"][-1]
                                         for q in parts if q["in"]}),
                     "types": sorted({q.get("type") for q in parts if q.get("type")}),
                     "unknown_types": bad,
                     "validated": val["checked"], "attempt": val["attempt"],
                     "plots_written": rnd.rel("plots.json"),
                     "centre": plan.get("centre")}}


def type_declarations(parts: list) -> dict:
    """{type name: declaration or None} for every type a plan names."""
    from .. import pipeline
    out: dict = {}
    for q in parts:
        t = q.get("type")
        if t in out:
            continue
        f = os.path.join(_pipeline.ROOT, "types", f"{t}.py") if t else None
        try:
            out[t] = pipeline.load_type(f) if f and os.path.exists(f) else None
        except Exception:                    # noqa: BLE001 -- a bad type is no type
            out[t] = None
    return out


def _stage_arterials(rnd, place: dict, decls: dict, vol) -> dict | None:
    """Route the arterials once and keep them. A4.

        Kept, for `stage_plateau`'s reason: a road re-routed after a district has been
        planned against it is a different road, and the districts joined the first one.
        
    """
    # **Kept for the plan they were routed from, and no longer.** The rule is that a
    # road re-routed after a district has been planned against it is a different road --
    # and before the place level passes there are no districts, so keeping the first
    # attempt's road through a hand-back keeps a road drawn round walls that have since
    # moved. One failure, wholly ours, on the round's second and last attempt. So the
    # cache is keyed by the geometry it was routed from. An unchanged place plan gets
    # the identical road; a changed one gets a new one; and once the districts are
    # planned nothing changes the place plan again.
    import hashlib
    from .. import placeplan
    designed = placeplan.designed_terrace(place)
    key = hashlib.sha256(json.dumps(
        {"parts": place.get("parts"), "districts": place.get("districts"),
         **({"terrace": designed,
             "approach": [placeplan.TERRACE_REACH, placeplan.GATE_APPROACH,
                          placeplan.GATE_APPROACH_HALF, placeplan.GATE_RAMP_RUN],
             "band": [placeplan.EDGE_BAND, placeplan.EDGE_BAND_COST]}
            if designed else {})},
        sort_keys=True).encode()).hexdigest()[:16]
    p = rnd.rel("arterials.json")
    if os.path.exists(p):
        got = json.load(open(p))
        if got.get("of_plan") == key:
            return got
        os.replace(p, rnd.rel(f"arterials.{got.get('of_plan', 'unkeyed')}.json"))
        print(f"   arterials: the place plan changed, so the road is routed again "
              f"(the one before it is kept beside it)", flush=True)
    if len(place.get("districts") or []) < 2 or vol is None:
        return None
    # `placeplan` imports `pipeline`, so it is imported here rather than at module
    # scope. It was not imported at all.
    from .. import observe
    heights, _wet = observe.ground_heights(vol)
    # and the circulation then laid the road at those levels through rings since
    # levelled twenty blocks below it: every gate threshold on an earthwork, 384 of 439
    # off the walkable network. Where the layout carries terrace levels the heights the
    # road is graded on are the terraces' and the podium's, the same rectangles the
    # terraces stage lays; the road then meets each ring at its level and climbs a step
    # at each gate.
    ground = "as found"
    band = None
    if designed:
        gates = [p for p in (place.get("parts") or []) if p.get("kind") == "point"
                 and (decls.get(p.get("type")) or {}).get("passage")]
        base = placeplan.designed_heights(heights, vol.x0, vol.z0, place["layout"],
                                          gates=gates, approaches=False)
        heights = placeplan.designed_heights(heights, vol.x0, vol.z0, place["layout"],
                                             gates=gates)
        band = placeplan.terrace_edge_band(heights.shape, vol.x0, vol.z0, place["layout"],
                                           gates=gates, designed=base)
        ground = "as designed"
    net, nodes = placeplan.plan_arterials(place, decls, heights, vol.x0, vol.z0,
                                          extra_cost=band)
    got = placeplan.arterial_record(net, nodes, place)
    got["of_plan"] = key
    got["ground"] = ground
    os.makedirs(rnd.state, exist_ok=True)
    json.dump(got, open(p, "w"), indent=1)
    print(f"   arterials: {len(got['cells'])} columns joining "
          f"{len(got['nodes'])} nodes", flush=True)
    return got


def _plan_volume(rnd, be):
    """The ground to classify a plan against, or None where there is none yet.

        A plan is drawn before the round caches its base volume, so this is the live
        backend's world where there is one and the cache where the round has already made
        it. With neither, the ground half of the validation says so rather than passing.
        
    """
    try:
        return be.volume
    except Exception:                        # noqa: BLE001 -- reported as unreadable
        return None


def validate(rnd, be, parts: list) -> dict:
    """Check a plan against its types' `NEEDS`, and hand a failing one back once. A2.

        Every failure is written to `plan_failures.json` and appended to the planner's own
        brief, and the plan it failed on is kept beside it as `plan.rejected.<n>.json`, so
        what the planner was told and what it had said are both on disk. The plan file
        itself is removed, which is what makes `run()`'s wait loop ask again: the stage is
        re-entered, sees no plan, and waits for the next one.
        
    """
    from .. import pipeline
    decls = type_declarations(parts)
    vol = _plan_volume(rnd, be)
    ground = pipeline.plan_ground(parts, vol)
    spec = rnd.place_spec() or {}
    fails = pipeline.plan_failures(parts, decls, ground=ground, network=rnd.network(),
                                   form=spec.get("form"))
    log = rnd.rel("plan_validation.json")
    was = json.load(open(log)) if os.path.exists(log) else {"attempts": []}
    attempt = len(was["attempts"]) + 1
    checked = [c for c in pipeline.PLAN_CHECKS if c != "ground" or ground]
    row = {"attempt": attempt, "leaves": len(parts), "failures": fails,
           "checked": checked,
           "ground": "measured" if ground else "no volume to read yet",
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    was["attempts"].append(row)
    json.dump(was, open(log, "w"), indent=1)
    if not fails:
        return {"failures": [], "attempt": attempt, "checked": checked}
    if attempt < 2:
        _hand_back(rnd, fails, attempt, decls)
    return {"failures": fails, "attempt": attempt, "checked": checked}


def _hand_back(rnd, fails: list, attempt: int, decls: dict) -> None:
    """Put the failures in the planner's brief and take the failing plan out of the way.

        In the same brief, not a new one: the planner is answering the question it was
        already asked, and a second brief would be a second experiment.
        
    """
    from .. import pipeline
    p = rnd.rel("plan.json")
    if os.path.exists(p):
        os.replace(p, rnd.rel(f"plan.rejected.{attempt}.json"))
    lines = ["", "", "---", "",
             "## Your plan was checked and it does not build",
             "",
             f"{len(fails)} leaf/leaves of the plan you wrote cannot be built as "
             "drawn. Nothing has been built and nothing is in the world. Write the "
             "plan again, in the same file, fixing these -- and only these -- and "
             "keeping everything that is not named here as it was.", ""]
    for f in fails:
        lines.append(f"- **{f['part']}** ({f['kind']}, `{f['type']}`) — {f['why']}")
    lines += ["",
              "The sizes every type needs, again, as **plots** with the library's "
              "inset already added:", "",
              pipeline.needs_table(decls), ""]
    with open(rnd.rel("planner_prompt.md"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
