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

    def walk(nodes, ancestry, answers):
        for n in nodes:
            kind = n.get("kind", "plot")
            # **The defining part a leaf answers, inherited.** The review's third
            # finding, and it is why capability reconciliation found nothing in
            # production: `placeplan.assemble` writes `defines` on the *quarter*, this
            # flattener carried only the ancestor names, and `capability._types_used`
            # reads leaf `defines` -- so the forty houses of a district were compared
            # with nothing at all and only the handful of leaves that carry `defines`
            # themselves were ever checked. A leaf answers the nearest defining part
            # above it; where it names its own, its own wins.
            mine = n.get("defines") or n.get("compound") or answers
            if kind in PART_GROUP_KINDS or n.get("children"):
                walk(n.get("children") or [], ancestry + [n.get("name", kind)], mine)
            else:
                row = {**n, "kind": kind, "name": n.get("name", n.get("id")),
                       "in": list(ancestry)}
                if mine and not row.get("answers"):
                    row["answers"] = mine
                out.append(row)
    walk(plan["parts"], [], None)
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
        # **A character-declared landmark is the district's own deliberate choice.** The
        # closure round's held-out hamlet: a smithy declared as the homes district's
        # landmark was refused because a workshop is an urban type in a rural district.
        # The role rule keeps a compiler from filling a farm quarter with shops; it does
        # not overrule a landmark the character's author named by type.
        landmark = str(p.get("name") or "").startswith("landmark_")
        if not landmark and not role_ok(decl.get("role"), p.get("role"), compound=bool(p.get("compound")),
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


#: `{(path, size, mtime_ns): declaration}` -- a type file read once per process. **The
#: realization round, found by profiling the compiler.** `_compile_once` asks
#: `placeplan.fabric` and `district_target` what a lot of this density costs, and each
#: of those walks every committed type; one compile of an 84x35 district executed 768
#: type files and spent 2.7 of its 2.9 seconds inside `compile()`. The district
#: compiler's search runs that up to 28 times, so one small district cost 96 seconds --
#: and the reason planning could not afford to ask the *actual* construction logic how
#: many houses a rectangle holds was almost entirely this. Keyed on the file's identity
#: and not its name, so a type edited mid-run is read again: the same rule
#: `deps.content_print` uses, for the same reason.
_TYPE_CACHE: dict = {}


def load_type(path: str) -> dict:
    """Read a type file's declarations without building anything.

        The module is executed in a bare namespace, which is safe precisely because the
        contract says so: a type file defines a function and some constants, and anything
        that placed a block at import time would place it once per instance. A file that
        breaks that is caught here, once, rather than twelve buildings later.

        Cached per file identity -- see `_TYPE_CACHE`. The returned declaration is treated
        as read-only by every caller in this build; it is the same object each time.
        
    """
    try:
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _TYPE_CACHE:
        return _TYPE_CACHE[key]
    src = open(path).read()
    ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
    exec(compile(src, path, "exec"), ns)                       # noqa: S102
    missing = [d for d in TYPE_DECLARATIONS if d not in ns]
    if missing or not callable(ns.get("build")):
        raise ValueError(
            f"{path}: a type declares {', '.join(TYPE_DECLARATIONS)} and defines "
            f"build(b, part, seed, **params); this one is missing "
            f"{', '.join(missing + ([] if callable(ns.get('build')) else ['build']))}")
    got = {"path": path, "src": src, "form": read_form(ns, where=path),
            # A2: what it is for, beside what it is built in. See `read_role`.
            "role": read_role(ns, where=path),
            "params": dict(ns["PARAMS"]), "lines": len(src.splitlines()),
            # A3: which kind of part this type builds, and whether a network may cross
            # it. Optional, and a plot where it is not said, because every type written
            # before A3 is a building on a plot and none of them says so.
            "kind": ns.get("KIND", "plot"), "passage": bool(ns.get("PASSAGE", False)),
            # the architecture round: a type may say which **family** of part it builds.
            # Optional, because the family has always been read off the committed name
            # and every file on disk predates this; where a file says it, its own word
            # wins (`capability.family_of`).
            "family": ns.get("FAMILY"),
            # the integration round: a type may say which **named tradition** it is
            # built in -- `japanese`, `german` -- which `FORM`'s four coarse families
            # cannot express. Optional and unset on every committed file, which is
            # exactly why a request for a Japanese village leaves its tradition
            # requirement `unresolved` instead of passing on a form family.
            "tradition": ns.get("TRADITION"),
            # the realization round: **what this type is for**, which is not what it is
            # built in (`FORM`), what work it is for (`ROLE`) or what kind of part it is
            # (`FAMILY`). The review: "An allowed plot type and a coarse role do not
            # establish that a dwelling's function has been fulfilled." A role of
            # `rural` is satisfied by a hall, a temple and a barn; a sentence asking for
            # houses people live in is not. Optional, and a type that declares none
            # fulfils no function -- which is the honest answer and the one that keeps
            # the obligation visible rather than letting the nearest label close it.
            "function": ns.get("FUNCTION"),
            # v2, C2: a plot type whose flanks are party walls says so, and may then
            # stand touching the next such leaf on a shared frontage.
            "attached": bool(ns.get("ATTACHED", False)),
            # a layout draws an octagon only for a type that does
            "diagonal": bool(ns.get("DIAGONAL_RUNS", False)),
            # What it needs from the ground. Optional here for the same reason. Every
            # file under `types/` declares it and `test_types.py` is what says so.
            "needs": read_needs(ns, where=path),
            "declares_needs": ns.get("NEEDS") is not None,
            # **What this type actually fills, which is not always its rectangle.** The
            # design round: the great wall lays piers, a parapet corbel and a switchback
            # stair outside its declared band, so planning, ground, routing and the
            # checks each had their own guess at how much room it takes. A type that
            # knows publishes `occupied(part, **params)` and everything reads that one
            # answer; a type that does not is its footprint plus its declared clearance.
            # See `types/great_wall.occupied` and `ground.occupied_envelope`.
            "occupied": ns.get("occupied") if callable(ns.get("occupied")) else None}
    if key is not None:
        _TYPE_CACHE[key] = got
    return got


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


CLAIMS_BRIEF = """# Read the sources, and say what they establish

> {sentence}

This request was read as a **{kind}**{about}. {why}

{evidence}

## What is asked

Write the **claims** these sources establish about the place this request asks for, and
nothing else. A claim is one fact that a planner could act on: how the place is
organised, what it is bounded by, what its buildings are made of and how they are
roofed, how big it is, what stands at its middle.

Every claim carries the **id of the source it came from**. A claim you cannot attach to
one of the sources above is either left out or marked `"inferred": true` with the
reasoning -- and a reading whose claims are all inferred is an honest weak reading,
which is far better than a confident wrong one. Where two sources disagree, say so in
`conflicts` rather than choosing silently.

Do not write a place spec here. Do not invent a source, a url or a quotation. Nothing
about the site, the size, the coordinates or the geometry is yours.

## Output

Reply with one JSON document:

{{"claims": [{{"id": "claim/1", "says": "...", "about": "layout|construction|scale|materials|hierarchy",
             "source": "src/1", "confidence": "high|medium|low", "inferred": false,
             "conflicts": "..." }}],
  "uncertainty": [{{"about": "...", "why": "..."}}],
  "inferred": ["..."]}}
"""


#: What a terminal agent is asked for when nothing else can retrieve. Deliberately a
#: contract and not an instruction to be clever: the fields are the ones
#: `contracts.SOURCE_FIELDS` requires, and a source that cannot carry them is not one.
RESEARCH_BRIEF = """# Research this request

    {sentence}

This build has **no retrieval provider configured**, and this request needs evidence:
{why}

You are the retrieval. Find out what is actually known about {about}, from sources you
have really read, and write `{write}`.

## What to write

```json
{{"sources": [{{"id": "short-slug",
               "title": "the page's own title",
               "url": "https://...",
               "accessed": "YYYY-MM-DD",
               "text": "the passage you read, quoted, not summarised",
               "supports": "what this source is here to establish"}}],
  "uncertainty": ["what the sources disagree about, or do not say"],
  "note": "how you searched and what you could not find"}}
```

`text` is what makes this a source: it is the passage, quoted. Its identity is
fingerprinted from those bytes and every claim made later has to name the source it came
from, so a paraphrase from memory is worse than nothing here. A request you can find no
source for gets `"sources": []` and a `note` saying so -- that is an honest answer and
the run will carry the obligation as unresolved rather than pretend.

## What this is for

The claims a later step draws from these sources inform the design, and the identity
requirement (`{about}`) can only be resolved by evidence that exists plus a judgment of
what was built. Neither an empty reading nor a confident recollection resolves it.
"""


def _retrieval_provider():
    """The configured retrieval provider, or None where there is none.

        One seam, so "can this run retrieve" is asked once and can be answered by a test.
        
    """
    from .. import evidence as evid
    prov = evid.provider()
    return None if getattr(prov, "name", "none") == "none" else prov


def _research_handoff(rnd, sentence: str, out_dir: str) -> dict | None:
    """**The review's missing job.** `stage_reading` completed an empty reading whenever no
        provider was configured, and the claims job -- the only agent job in the reading --
        ran *after* sources existed. So a request that names a place got `provider: none`,
        no sources, no claims, and an identity obligation with nothing behind it, and there
        was no state in which a terminal agent could supply what was missing. A terminal
        agent is a supported runtime; this is the state it answers in.
        
    """
    from .. import contracts, evidence as evid
    if not evid.needs_evidence(sentence):
        return None
    if _retrieval_provider() is not None:
        return None
    answer = os.path.join(out_dir, "sources.json")
    if os.path.exists(answer):
        return None
    what = evid.classify(sentence)
    # **the question this answer will belong to.** A research answer is keyed to its
    # sentence for the same reason a gathering is: the review reproduced a run that
    # changed its sentence and kept the previous one's research, and a terminal answer
    # is exactly as transferable as a retrieved one, which is to say not at all.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "asked.json"), "w") as fh:
        json.dump({"sentence": sentence,
                   "sentence_digest": contracts.digest(sentence)}, fh, indent=1)
    about = (f"{what['name']!r}" if what.get("name") else
             f"{what['tradition']} building" if what.get("tradition") else
             "this kind of place")
    brief = rnd.rel("research_prompt.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(brief, "w") as fh:
        fh.write(RESEARCH_BRIEF.format(sentence=sentence, why=what["why"], about=about,
                                       write=answer))
    return {"status": "needs_model", "role": "research", "request": brief,
            "write": answer,
            "note": (f"no retrieval provider is configured and this request needs "
                     f"evidence about {about}; the agent driving this round is the "
                     f"retrieval, and writes the sources it actually read")}


def _adopt_research(sentence: str, out_dir: str) -> dict | None:
    """A terminal agent's `sources.json`, as a gathering. None where there is none.

        Every source is written to disk under its own id and fingerprinted **from the bytes
        that were written**, so the identity in the record is the identity of the text a
        later claim will be drawn from and not a number the answer supplied about itself.
        
    """
    from .. import contracts, evidence as evid
    p = os.path.join(out_dir, "sources.json")
    if not os.path.exists(p):
        return None
    asked = os.path.join(out_dir, "asked.json")
    was = json.load(open(asked)) if os.path.exists(asked) else {}
    if was.get("sentence_digest") != contracts.digest(sentence):
        # the answer to a different question: set aside by name, never adopted
        for f in ("sources.json", "asked.json"):
            if os.path.exists(os.path.join(out_dir, f)):
                os.replace(os.path.join(out_dir, f),
                           os.path.join(out_dir, f"stale.{f}"))
        print(f"   reading: the research on disk answers "
              f"{was.get('sentence') or 'another sentence'}; it is set aside",
              flush=True)
        return None
    doc = json.load(open(p))
    what = evid.classify(sentence)
    sources, refused = [], []
    for i, s in enumerate(doc.get("sources") or []):
        sid = str(s.get("id") or f"source_{i}").strip()
        text = str(s.get("text") or "")
        missing = [k for k in ("title", "url", "accessed") if not str(s.get(k) or "").strip()]
        if not text.strip():
            missing.append("text")
        if missing:
            refused.append({"id": sid, "missing": missing})
            continue
        rel = f"{evid._slug(sid) if hasattr(evid, '_slug') else sid}.txt"
        with open(os.path.join(out_dir, rel), "w") as fh:
            fh.write(text)
        sources.append({"id": sid, "title": str(s["title"]), "url": str(s["url"]),
                        "accessed": str(s["accessed"]),
                        "fingerprint": evid._fingerprint(text.encode("utf-8")),
                        "media": "text/plain", "supports": s.get("supports") or "",
                        "provider": "terminal", "bytes": len(text.encode("utf-8")),
                        "path": rel})
    return {"classification": what, "queries": list(doc.get("queries") or []),
            "sources": sources, "provider": "terminal" if sources else "none",
            "refused": refused,
            "uncertainty": list(doc.get("uncertainty") or []),
            "note": (str(doc.get("note") or "")
                     + (f"; {len(refused)} source(s) refused for missing "
                        f"title, url, access date or text" if refused else "")),
            "hits": []}


def stage_reading(rnd, be, results: dict) -> dict:
    """What this request is about, before anything is planned. The architecture round.

        Two records come out of it, and they are different things on purpose:

          `intent.json`   the **sentence's own** requirements, read by rule with no model
                          call at all (`ethoslm.intent`). Immutable. This is what every later
                          check is scored against, because a check generated from a model's
                          interpretation of the sentence cannot find what that
                          interpretation dropped.
          `reading.json`  what was found out about it: the classification, the queries, the
                          sources actually retrieved with their fingerprints, and the claims
                          a model made **from those sources**.

        A request that needs no evidence -- one that describes the place it wants -- gets an
        empty, honest reading and no model call. A request that does, and has no retrieval
        provider configured, gets a reading that says so: `provider: none`, every claim
        inferred. Neither is a failure and both are visible.
        
    """
    from .. import contracts, deps, evidence as evid, intent as intent_mod
    if not rnd.sentence:
        return {"skipped": "this round carries no sentence, so there is nothing to read"}
    fresh, why = deps.check(rnd, "reading")
    if fresh and contracts.load(rnd, "reading") is not None:
        rec = contracts.load(rnd, "reading")
        return {"skipped": why, "classification": rec["classification"],
                "sources": len(rec["sources"]), "claims": len(rec["claims"]),
                "provider": rec["provider"]}
    os.makedirs(rnd.state, exist_ok=True)
    it = intent_mod.read(rnd.sentence)
    # **One producer for `intent.json`.** The review's fifth finding, reproduced by the
    # closure runner: this stage stamped `intent.json` as its output, `stage_interpret`
    # rewrote the same file, and the two took turns calling each other stale on every
    # unchanged replay. The rules' own reading is kept beside the record as
    # `intent.rules.json` -- it is what the cross-check quotes -- and `intent.json` is
    # written here only where no interpretation of this sentence exists yet, so a round
    # with no interpreter keeps the rules' reading exactly as it always did.
    json.dump(it, open(rnd.rel("intent.rules.json"), "w"), indent=1)
    interp = contracts.load(rnd, "interpretation")
    if interp is None or interp.get("sentence") != rnd.sentence \
            or contracts.load(rnd, "intent") is None:
        contracts.save(rnd, "intent", it)
    print(f"   intent: {len(it['requirements'])} requirement(s) read from the sentence"
          + (": " + ", ".join(r["id"] for r in it["requirements"])
             if it["requirements"] else " (it states none outright)"), flush=True)
    out_dir = rnd.rel("evidence")
    gath_p = rnd.rel("gathering.json")
    # carrying the old sentence's classification, its queries and its named city into a
    # request that never mentioned one. A research answer is keyed to its question.
    want = contracts.digest(rnd.sentence)
    gathering = None
    if os.path.exists(gath_p):
        was = json.load(open(gath_p))
        if was.get("sentence_digest") == want:
            gathering = was
        elif was.get("sentence_digest") is None and not was.get("sources"):
            # a gathering from before this key existed, and an empty one: its
            # classification is re-derived below rather than trusted
            print("   reading: the gathering on disk predates the sentence key and "
                  "retrieved nothing; it is made again", flush=True)
        else:
            os.replace(gath_p, rnd.rel("gathering.stale.json"))
            print(f"   reading: the gathering on disk was made for a different "
                  f"sentence ({was.get('classification', {}).get('name') or 'unnamed'});"
                  f" it is set aside and this request is researched again", flush=True)
    if gathering is None:
        # **The research handoff, before the empty answer.** With a provider this is
        # `evid.gather`; with none and a request that needs evidence, the agent driving
        # the round is asked to be the retrieval, and its answer is adopted as the
        # gathering. An empty reading is still an available outcome -- a request that
        # describes the place it wants needs no evidence, and an agent that finds no
        # source says so -- but it is no longer what happens by default when nobody
        # asked.
        ask = _research_handoff(rnd, rnd.sentence, out_dir)
        if ask is not None:
            print(f"   reading: {ask['note']}", flush=True)
            return {"reading": ask, "requirements": len(it["requirements"])}
        gathering = _adopt_research(rnd.sentence, out_dir)
        if gathering is None:
            gathering = evid.gather(rnd.sentence, out_dir)
        gathering["sentence_digest"] = want
        gathering["sentence"] = rnd.sentence
        json.dump(gathering, open(gath_p, "w"), indent=1)
        for stale in ("reading.claims.json",):
            if os.path.exists(rnd.rel(stale)):
                os.replace(rnd.rel(stale), rnd.rel(stale + ".stale"))
    claims_p = rnd.rel("reading.claims.json")
    if gathering.get("sources") and not os.path.exists(claims_p):
        brief_p = rnd.rel("reading_prompt.md")
        what = gathering["classification"]
        excerpts = []
        for s in gathering["sources"]:
            text = open(os.path.join(out_dir, s["path"])).read()[:6000]
            excerpts.append(f"### `{s['id']}` {s['title']}\n<{s['url']}> "
                            f"(accessed {s['accessed']}, {s['fingerprint']})\n\n"
                            f"```\n{text}\n```")
        open(brief_p, "w").write(CLAIMS_BRIEF.format(
            sentence=rnd.sentence, kind=what["kind"],
            about=(f" named {what['name']!r}" if what.get("name") else
                   f" in the {what['tradition']} tradition" if what.get("tradition")
                   else ""),
            why=what["why"],
            evidence="## The sources\n\n" + "\n\n".join(excerpts)))
        # **Under `reading`, so the driver can see it.** `round._needs_model` reads the
        # *entries* of a stage result, and this was returned flat -- so the one agent
        # job the reading stage already had was invisible to the loop that waits for
        # agent jobs, and a round with an unanswered claims job went straight on to plan
        # a place from a reading nobody had written.
        return {"reading": {
            "status": "needs_model", "role": "spec", "request": brief_p,
            "write": claims_p,
            "note": f"{len(gathering['sources'])} source(s) retrieved by the "
                    f"`{gathering['provider']}` provider; the claims they establish, "
                    f"each carrying the id of the source it came from"}}
    claims, uncertainty, inferred = [], [], []
    if os.path.exists(claims_p):
        doc = json.load(open(claims_p))
        claims = list(doc.get("claims") or [])
        uncertainty = list(doc.get("uncertainty") or [])
        inferred = list(doc.get("inferred") or [])
    rec = evid.reading_of(rnd.sentence, gathering, claims)
    if uncertainty:
        rec["uncertainty"] = uncertainty + list(rec["uncertainty"])
    if inferred:
        rec["inferred"] = inferred + list(rec["inferred"])
    try:
        contracts.save(rnd, "reading", rec)
    except contracts.ContractError as e:
        # A claim with no source and no `inferred` flag is refused **here**, before it
        # can become a fact about the place. The answer is set aside and named.
        os.replace(claims_p, rnd.rel("reading.claims.rejected.json"))
        return {"status": "error", "stop": True,
                "error": f"the reading's claims do not meet the reading contract: {e}"}
    deps.stamp(rnd, "reading", outputs=["reading.json"],
               note=f"provider {rec['provider']}")
    print(f"   reading: {rec['classification']}, {len(rec['sources'])} source(s) via "
          f"`{rec['provider']}`, {len(rec['claims'])} claim(s)", flush=True)
    return {"classification": rec["classification"], "provider": rec["provider"],
            "sources": len(rec["sources"]), "claims": len(rec["claims"]),
            "queries": rec["queries"], "requirements": len(it["requirements"]),
            "needs_evidence": evid.needs_evidence(rnd.sentence),
            "note": rec.get("note") or ""}


def stage_interpret(rnd, be, results: dict) -> dict:
    """**What the sentence means**, read by an agent and cross-checked by the rules.

        Between the reading and the spec, and both halves of that matter. After the reading,
        because what has been sourced about a named place is what lets its sentence be
        interpreted at all. Before the spec, because the programme is designed *from* the
        meaning: an interpretation produced after the design would be a description of it.

        The agent's answer becomes `interpretation.json`; `intent.json` is written again from
        it, with the phrase rules' own reading of the same sentence as a cross-check and
        every disagreement between them on the record. A round whose agent declines to
        interpret keeps the rules' reading exactly as it was, so nothing that ever ran stops
        running -- it simply keeps the weaker reading, and the record says which it has.
        
    """
    from .. import contracts, deps, interpret as interpret_mod
    if not rnd.sentence:
        return {"skipped": "this round carries no sentence, so there is nothing to "
                           "interpret"}
    fresh, _why = deps.check(rnd, "interpretation")
    got = contracts.load(rnd, "interpretation")
    p = rnd.rel("interpretation.json")
    answer_p = rnd.rel("interpretation.answer.json")
    if fresh and got is not None:
        return {"skipped": _why, "reads": len(got["reads"]),
                "checks": {k: len(v) for k, v in (got.get("checks") or {}).items()
                           if isinstance(v, list)}}
    if not os.path.exists(answer_p):
        brief_p = rnd.rel("interpretation_prompt.md")
        os.makedirs(rnd.state, exist_ok=True)
        open(brief_p, "w").write(interpret_mod.brief(
            rnd.sentence, answer_p, contracts.load(rnd, "reading")))
        return {"interpretation": {
            "status": "needs_model", "role": "spec", "request": brief_p,
            "write": answer_p,
            "note": ("the sentence read into scoped, related, reasoned requirements; "
                     "the phrase rules cross-check this and do not define it")}}
    try:
        rec = interpret_mod.load(answer_p, rnd.sentence)
    except (contracts.ContractError, ValueError) as e:
        # refused by name and left on disk, so the answer and its refusal are both
        # readable; the round stops rather than designing from a reading it could not
        return {"status": "error", "stop": True,
                "error": f"the interpretation was refused: {e}"}
    rec["checks"] = interpret_mod.cross_check(rnd.sentence, rec)
    contracts.save(rnd, "interpretation", rec)
    it = interpret_mod.requirements(rnd.sentence, rec,
                                    reading=contracts.load(rnd, "reading"))
    contracts.save(rnd, "intent", it)
    ck = rec["checks"]
    print(f"   interpreted: {len(rec['reads'])} reading(s) -> "
          f"{len(it['requirements'])} requirement(s); cross-check "
          f"{len(ck['agreed'])} agreed, {len(ck['read_only'])} read only, "
          f"{len(ck['rules_only'])} by rule only, {len(ck['contradicts'])} "
          f"contradicted, {len(ck['unsupported_phrase'])} refused", flush=True)
    for c in ck["contradicts"]:
        print(f"     ! {c['id']}: {c['why']}", flush=True)
    for c in ck["unsupported_phrase"]:
        print(f"     x {c['id']}: {c['why']}", flush=True)
    with contextlib.suppress(ValueError):
        # **The interpretation's output is the interpretation.** `intent.json` is
        # written from it here and its `status`/`why` are rewritten by every check
        # afterwards; binding this stamp to that file made the interpretation stale the
        # moment the place was resolved, and an unchanged replay re-interpreted the
        # sentence every time. The semantic content of `intent.json` is what the
        # `intent` fingerprint kind hashes, and that is what every downstream artifact
        # depends on.
        deps.stamp(rnd, "interpretation", outputs=["interpretation.json"],
                   note=f"{len(rec['reads'])} reading(s) from {rec['source']}")
    return {"reads": len(rec["reads"]), "requirements": len(it["requirements"]),
            "checks": {k: len(v) for k, v in ck.items() if isinstance(v, list)},
            "contradicts": ck["contradicts"], "refused": ck["unsupported_phrase"]}


def explicit_count_of(rnd) -> dict | None:
    """The explicit count the intent record carries, in `spec.count_in`'s shape, or None.

        The interpretation's `count` requirement -- `{"n", "about", "what"}` -- is the
        sentence's own number read by the agent and cross-checked by the rules. Where the
        record has one it is the count; where it has none the spec falls back to the regex,
        which is every round that predates the interpretation stage.
        
    """
    from .. import contracts
    rec = contracts.load(rnd, "intent")
    for r in (rec or {}).get("requirements") or []:
        w = r.get("wants") or {}
        if r.get("kind") == "count" and w.get("n") and r.get("status") != "unsupported":
            return {"n": int(w["n"]), "about": bool(w.get("about")),
                    "phrase": str(r.get("phrase") or ""), "what": str(w.get("what") or ""),
                    "requirement": r["id"]}
    return None


#: How much of a place's square the fabric, its land and its lanes actually occupy. The
#: rest is the arterial's band, the margins and the ground between districts that no
#: district owns. Registered from the expression round's farm (a 192 square holding
#: 26,600 columns of district against 36,864) and used only to turn a demand in columns
#: into a side; the layout refuses or accepts on its own arithmetic afterwards.
FOOTPRINT_PACKING = 0.72


def _footprint_from_demand(rnd, spec: dict) -> dict | None:
    """**A place is as big as what it has to hold.** Raise `needs.footprint` where the
        resolved demand needs more ground than the count alone implies.

        The design round's first contract, at the one place it had not reached: the
        footprint. `spec.footprint_for` derives it from the count and the kind, so sixteen
        cottages are a 192-column village whether each stands on a 12x10 lot or on the 24x24
        one a **required second storey** forces -- four times the ground, on a square that
        never moved. The expression round could not notice, because nothing asked the type
        what the request needed of it before the place was sized.

        So the demand is resolved here, before the site is searched, and every part's
        `land_need` is summed over `FOOTPRINT_PACKING`. The footprint only ever **grows**,
        never past the kind's ceiling, and the record says by how much and from what. A
        place whose demand fits the count's own square is untouched.
        
    """
    import math
    from .. import contracts as contracts_mod, placesolve
    try:
        decls = _decls_for(rnd, spec)
        resolved, _rec = placesolve.resolved_spec(
            spec, rnd_capabilities(rnd), contracts_mod.load(rnd, "intent"), decls=decls)
        need = 0
        rows = []
        for part in resolved.get("defining_parts") or []:
            got = placesolve.land_need(part, int(part.get("structures") or 0), decls,
                                       resolved)
            need += int(got.get("columns") or 0)
            if got.get("columns"):
                rows.append(f"{part.get('name')}: {got['columns']:,} columns"
                            + (f" at a lot of {got['lot'][0]}x{got['lot'][1]}"
                               if got.get("lot") else ""))
    except Exception as e:                        # noqa: BLE001 -- reported, not raised
        return {"was": spec["needs"]["footprint"], "now": spec["needs"]["footprint"],
                "why": (f"the demand could not be resolved before the site was sized "
                        f"({type(e).__name__}: {e}); the footprint is the count's")}
    if not need:
        return None
    side = int(math.ceil(math.sqrt(need / FOOTPRINT_PACKING)))
    cap = int(spec_mod.footprint_ceiling(spec.get("kind")))
    was = int(spec["needs"]["footprint"])
    now = min(max(was, side), cap)
    if now <= was:
        return None
    spec["needs"]["footprint"] = now
    spec["footprint_from"] = {
        # `spec.read_spec` reads `footprint_from.footprint` back as the least footprint
        # this place may be re-read at, so a rewrite of the checked spec keeps it
        "footprint": now,
        "was": was, "wanted": side, "now": now, "ceiling": cap,
        "columns": int(need), "packing": FOOTPRINT_PACKING, "parts": rows,
        "why": (f"the resolved demand needs {need:,} columns of district and land -- "
                + "; ".join(rows)
                + f" -- which is {side} a side at {FOOTPRINT_PACKING:.0%} packing, "
                  f"against the {was} the count alone asks for"
                + (f"; held at the {cap} ceiling for a {spec.get('kind')}"
                   if side > cap else ""))}
    return {"was": was, "now": now, "why": spec["footprint_from"]["why"]}


def rnd_capabilities(rnd):
    from .. import contracts as _contracts
    return _contracts.load(rnd, "capabilities")


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
        # **The count the interpretation read, handed to the spec.** The closure round's
        # retained first failure: `read_spec` derived the band from `count_in` alone and
        # sized "sixteen low cottages" at fifty-two, because `sixteen` is not in the
        # regex's table and nothing consulted `intent.json`, where the reader had
        # written n=16 exact. An explicit quantity is the sentence's and it survives
        # from here on.
        count = explicit_count_of(rnd)
        try:
            s = spec_mod.read_spec({k: v for k, v in doc.items() if k != "hand_back"},
                                   rnd.sentence or None, count=count)
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
        # **A negotiation survives the stage that would re-derive it away.** Found by
        # replaying the held-out village: this stage rewrites the checked spec from
        # `place.json` every run, so the band a repair had moved -- and the record of
        # the bound it moved inside -- were gone by the time the plan stage read the
        # spec back, the spec fingerprint went back to what it had been, the plan was
        # called stale, and the whole place was laid out again. An unchanged replay was
        # never warm and the same repair was made every time. `spec.read_spec` already
        # restores a negotiated band from `negotiated`; what was missing is that nothing
        # carried `negotiated` forward. It is carried only where the *model's* answer is
        # unchanged, so a genuinely new spec starts clean.
        checked = rnd.rel("place.checked.json")
        if os.path.exists(checked):
            was = json.load(open(checked))
            if was.get("negotiated") and was.get("sentence") == s.get("sentence") \
                    and not s.get("negotiated"):
                s = spec_mod.read_spec({**s, "negotiated": was["negotiated"]},
                                       rnd.sentence or None, count=count)
                print(f"   spec: {len(was['negotiated'])} negotiation(s) carried "
                      f"forward; the band reads {s['size_band']} and not the "
                      f"kind's own", flush=True)
            # **And a character its author revised survives the same way.** The same
            # defect as the negotiation above, one field along: `stage_preview`'s
            # revision and `apply_character` both write the new character into the
            # *checked* spec, and this stage rewrites the checked spec from the model's
            # raw answer every run -- so a revision that had been applied, compiled and
            # inspected was erased by the next invocation, the districts were laid out
            # again from the character the model first wrote, and the run reported the
            # place the revision had replaced. Found by running the shore village
            # through its second inspection. Carried only where the *model's* answer for
            # that part is unchanged, so a genuinely new spec starts clean.
            raw_parts = {q.get("name"): q for q in (doc.get("defining_parts") or [])}
            kept = []
            for q in was.get("defining_parts") or []:
                name = q.get("name")
                mine = next((r for r in s["defining_parts"]
                             if r.get("name") == name), None)
                if mine is None or q.get("character") is None:
                    continue
                if (raw_parts.get(name) or {}).get("character") == q.get("character"):
                    continue          # the model's own character, not a revision
                if mine.get("character") != q["character"]:
                    mine["character"] = q["character"]
                    kept.append(name)
            if kept:
                s = spec_mod.read_spec(s, rnd.sentence or None, count=count)
                print(f"   spec: the revised character(s) of {', '.join(kept)} carried "
                      f"forward; the districts are compiled from what their author "
                      f"wrote and not from the first answer", flush=True)
        grew = _footprint_from_demand(rnd, s)
        json.dump(s, open(checked, "w"), indent=1)
        print(spec_mod.summary(s), flush=True)
        if grew:
            print(f"   footprint: {grew['was']} -> {grew['now']}; {grew['why']}",
                  flush=True)
        # An `unsupported` requirement is not a shortage of ground or of budget. It is
        # this library saying it has no family, no policy and no form for the thing that
        # was asked for, and that is known the moment the sentence has been read and the
        # programme written. Searching for ground to put it on cannot change it, and
        # whatever goes wrong during that search becomes the reason the run failed. So
        # the refusal is made where it is decided, with its own reasons, and nothing
        # downstream gets the chance to fail first for a reason of its own.
        gate = _unsupported_gate(rnd, s)
        if gate is not None:
            return gate
        return {"status": "read", "spec": s, "path": p,
                "authored_voice": wrote,
                "summary": spec_mod.summary(s)}
    if not rnd.sentence:
        return {"status": "error", "stop": True,
                "error": "a round that plans a place carries the sentence it is "
                         "planning: `sentence` in the config, and nothing else"}
    if not os.path.exists(brief):
        os.makedirs(rnd.state, exist_ok=True)
        open(brief, "w").write(spec_brief(rnd.sentence, p) + _reading_note(rnd))
    return {"status": "needs_model", "role": "spec", "request": brief, "write": p,
            "note": "one call, a fixed schema: the sentence becomes a place spec and "
                    "nothing about the site, the scale or the geometry is decided here"}


def _unsupported_gate(rnd, spec: dict):
    """Stop the round where the request names things this build cannot express.

        Returns a stage result, or None where every hard requirement is something this build
        can at least attempt. See the call site: the point is that the refusal is reached
        **at the stage that knows it**, so that it is the reason the run ends rather than
        whatever the ground search happens to fail on afterwards.

        `unsupported` is written by `intent.read` and by the interpretation's cross-check,
        and it means one thing: no family, no policy, no form. It is never written by a
        measurement of a plan, so nothing later in the run can resolve it -- which is
        exactly why continuing costs a site search and buys nothing.
        
    """
    from .. import contracts
    it = contracts.load(rnd, "intent")
    rows = [r for r in (it or {}).get("requirements") or []
            if r.get("status") == "unsupported" and r.get("hard")]
    if not rows:
        return None
    # written into the findings record too, so the readout and any reader of
    # `findings.json` see the same refusal the driver stopped on
    findings = contracts.make(
        "findings", stage="place_spec.unsupported",
        note="the request names things this build has no family, policy or form for",
        findings=[{"id": f"find/{r['id']}", "says": f"{r['says']}: {r['why']}",
                   "requirement": r["id"], "part": None,
                   "evidence": {"phrase": r.get("phrase"), "wants": r.get("wants")},
                   "owner": "capability", "blocks": "fidelity", "severity": "error",
                   "seen_by": "place_spec.unsupported", "fixed": False} for r in rows])
    contracts.save(rnd, "findings", findings)
    print(f"   spec: {len(rows)} requirement(s) this build cannot express; the round "
          f"ends incomplete and says which", flush=True)
    for r in rows:
        print(f"     - {r['id']}: {r['why']}", flush=True)
    return {
        "status": "blocked", "stop": True, "blocks": "fidelity",
        "spec": spec_mod.summary(spec),
        "unsupported": [{"id": r["id"], "says": r["says"], "why": r["why"],
                         "phrase": r.get("phrase")} for r in rows],
        "error": (f"this build cannot express {len(rows)} hard requirement(s) of this "
                  f"request, and no later stage can resolve one: "
                  + "; ".join(f"{r['id']} -- {r['why']}" for r in rows)
                  + ". Nothing is sited and nothing is built, because the place that "
                    "was asked for is not one this library can make and a place it "
                    "could make would be a different request")}


def _reading_note(rnd) -> str:
    """What the reading stage found out, and what the sentence requires outright.

        Appended to the spec brief rather than written into it, so a round with no reading
        stage -- every round before this one -- composes exactly the brief it always did.

        The requirement table is here for one reason: a spec that omits something the
        sentence says outright is refused downstream by `intent.coverage` **whatever the
        brief said**, and telling the call what it will be held to is cheaper than handing
        it back.
        
    """
    from .. import contracts, evidence as evid
    it = contracts.load(rnd, "intent")
    reading = contracts.load(rnd, "reading")
    interp = contracts.load(rnd, "interpretation")
    if it is None and reading is None and interp is None:
        return ""
    out = []
    # **The interpretation first, because the programme is designed from the meaning.**
    # Without this the spec call saw a table of rule-derived requirements and never saw
    # the scope of a negation, the relation between two kinds of building or the
    # hierarchy between them -- so a reading that carried all three reached the design
    # as a list of features, which is the same loss the phrase table was making one
    # stage earlier.
    if interp is not None and interp.get("reads"):
        out.append("\n\n---\n\n## What the sentence means\n")
        out.append("An interpreter read the request before you saw it. **Design from "
                   "this.** Each line is a requirement the finished place is checked "
                   "against; `scope` names the part of the place it is about, and a "
                   "requirement with a scope needs a defining part that answers that "
                   "name.\n")
        for r in interp["reads"]:
            out.append(f"- **{r['kind']}** `{r['id']}`"
                       + (f" *(in {r['scope']})*" if r.get("scope") else "")
                       + f" — {r['says']}"
                       + (f"  `{json.dumps(r['wants'])}`" if r.get("wants") else "")
                       + (f"\n    - why: {r['why']}" if r.get("why") else ""))
        for r in interp.get("uncertain") or []:
            out.append(f"- *uncertain*: {r}")
        if interp.get("unread"):
            out.append("\nThe interpreter could not read: "
                       + ", ".join(f"`{u}`" for u in interp["unread"])
                       + ". These stay open obligations; do not invent parts for them.")
        ck = interp.get("checks") or {}
        if ck.get("contradicts"):
            out.append("\n**The phrase rules read some of this the other way round** "
                       "and the disagreement is on the record: "
                       + "; ".join(str(c.get("why")) for c in ck["contradicts"][:4])
                       + ". Design from the interpretation above.")
    out.append("\n\n---\n\n## What the sentence requires outright\n")
    if it and it["requirements"]:
        out.append("These were read from the sentence by rule, before you saw it, and "
                   "the finished place is checked against them whatever this spec "
                   "says. A defining part answering each of them is not optional:\n")
        out.append(contracts.table(it))
        out.append("\nA requirement marked `unsupported` is one this build has no way "
                   "to express. Do **not** substitute something else for it: leave it "
                   "out, and the run will report it as unmet.")
    else:
        out.append("The sentence states no requirement this build reads by rule. What "
                   "the place is, is yours to read.")
    if reading is not None:
        out.append("\n\n## What was found out about this request\n")
        out.append(evid.brief(reading))
    return "\n".join(out)


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
    from .. import deps
    p = rnd.rel("site_search.json")
    if os.path.exists(p):
        got = json.load(open(p))
        # **Reused only where what it was scored against has not moved.** The
        # architecture audit reproduced this: change a village's spec into a city's and
        # the stage returned the village's site, because its whole test for "already
        # done" was that the file exists. The site is still a fixture once chosen -- a
        # round whose spec is unchanged never re-searches -- but a site chosen for a
        # different place is stale, and the run says so rather than planning a city on a
        # village's square. A round with no stamp at all (every round before this one,
        # and every shipped fixture) is reused exactly as it was.
        fresh, why = deps.check(rnd, "site_search")
        # **A fixture is an import, and a search that chose nothing is not a site.** The
        # closure round's second retained failure: the first run's search failed on a
        # wrongly-sized footprint and wrote its record; the spec was corrected; and this
        # branch read "no stamp" as "a shipped fixture" and handed the failed record
        # back as the chosen site. `deps.legacy` is the question.
        legacy = deps.legacy(rnd, "site_search")
        if got.get("chosen") and (fresh or legacy):
            return {"skipped": ("already searched -- the site is an imported fixture"
                                if legacy and not fresh else why),
                    "chosen": got.get("chosen"), "attempt": got.get("attempt"),
                    "dependencies": "imported" if legacy and not fresh else "warm",
                    "top": got.get("top", [])[:3], "path": p}
        os.replace(p, rnd.rel("site_search.stale.json"))
        deps.invalidate(rnd, "site_search")
        print(f"   site search: "
              + (why if got.get("chosen") else "the record on disk chose no site")
              + "; the stale record is kept beside it and the search is run again",
              flush=True)
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
        deps.stamp(rnd, "site_search", outputs=["site_search.json"],
                   note="the dry run's search over the squares out/sites/ holds")
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
    deps.stamp(rnd, "site_search", outputs=["site_search.json"],
               note="scored against this spec's needs")
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
    """**...in the order the sentence's own words ask for**, the design round. A place
        whose rings carry elevation words -- a hill town's `lower` and `upper` -- means them
        literally, and the expression round put the dense *lower* ring on the terrace four
        blocks above the sparse *upper* one because the words were read as ring ranks.
        `placeplan.terrace_ranks` reads them and `concentric_layout` re-orders the levels on
        the plan; the **podium** is cut from the record this function writes, before the
        plan exists, so the ranks are passed here too or the cut is made in the old order.
        
    """
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
    ranks, _why = placeplan.terrace_ranks(rings)
    return placeplan.terrace_levels(med, len(rings), ranks=ranks)


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
    from .stages_build import UNREADABLE_SITES
    fs = _find_site_module()
    spec = rnd.place_spec()
    got = fs.search_fresh(spec, editor=None, log=lambda *a: None)
    ref_p = rnd.rel(UNREADABLE_SITES)
    refused = (json.load(open(ref_p)).get("refused") or []
               if os.path.exists(ref_p) else [])
    if refused:
        # a ranked square says where it is as `x`/`z`; the chosen one says `origin`
        def _at(row):
            o = row.get("origin")
            if o:
                return (int(o[0]), int(o[1]))
            if row.get("x") is not None:
                return (int(row["x"]), int(row["z"]))
            return None

        bad = {(int(r["origin"][0]), int(r["origin"][1])) for r in refused}
        ranked = [t for t in (got.get("top") or []) if _at(t) not in bad]
        got["top"] = ranked
        got["refused_unreadable"] = refused
        chosen = got.get("chosen")
        if chosen and _at(chosen) in bad:
            nxt = ranked[0] if ranked else None
            got["chosen"] = ({"origin": list(_at(nxt)), "size": int(nxt["size"]),
                              "score": nxt.get("score"), "meets": nxt.get("meets", True),
                              "measures": nxt.get("measures", nxt)}
                             if nxt and _at(nxt) else None)
            if not got["chosen"]:
                got["best_failed"] = {
                    "origin": list(_at(chosen) or []),
                    "failures": (f"every square this search ranked has been chosen and "
                                 f"refused: {len(refused)} of them could not be read off "
                                 f"this save's region files")}
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
        # **Already cut is a fact about the world, and stale is a fact about the
        # design.** The review's fourth finding reached this stage: the test was the
        # existence of the JSON file, so a candidate whose spec, site or terrain had
        # moved under it went on standing on ground that had been levelled for the
        # design before it -- and said nothing. Ground work genuinely is not idempotent,
        # so this still does not cut again; what it does now is *say* that the cut it is
        # reusing was made for another candidate, and route that to the owner who can
        # decide, rather than reporting a clean skip.
        from .. import deps as deps_mod
        fresh, why = deps_mod.check(rnd, "plateau")
        # **Never stamped is not stale.** A plateau cut before this stage stamped
        # anything -- every round in the record, and this stage until the line below --
        # has no stamp at all, and reading that as "made for a different candidate"
        # would stop every one of them. `recorded` is the question that distinguishes
        # "this cut says nothing about what it was made from" from "it says, and what it
        # says has moved".
        if fresh or not deps_mod.recorded(rnd, "plateau") \
                or deps_mod.legacy(rnd, "plateau"):
            return {"skipped": "already cut -- ground work is not idempotent", **got}
        return {**got, "skipped": "already cut -- ground work is not idempotent",
                "stale": why, "status": "error", "stop": True,
                "error": (f"the ground was levelled for a different candidate and a cut "
                          f"cannot be taken back: {why}. A fresh round directory is the "
                          f"way to plan this design on unprepared ground")}
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
    # **Only a part at the centre, or one that asks for a level square, is levelled
    # for.** The closure round's held-out hamlet: a shoreline place with no centre part
    # had its first district marked as the "core" and a 48x48 square cut for it at the
    # site's edge -- levelled ground nobody asked for, overlapping the shore band. A
    # district stands on the ground as found.
    if spec:
        part = next((q for q in spec.get("defining_parts") or []
                     if q.get("name") == terra.get("part")), None)
        if part is not None and part.get("relation") != "centre" \
                and not (part.get("needs") or {}).get("plateau"):
            return {"skipped": (f"the search marked {terra.get('part')} for "
                                f"terraforming and it is a {part.get('relation')} "
                                f"{part.get('kind')} that asks for no level square; "
                                f"the ground stands as found"),
                    "part": terra.get("part")}
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
    # **What this cut was made for, so the next run can tell.** Ground work is not
    # idempotent and this stage has always refused to do it twice; what it could not do
    # was say whether the cut it was reusing belonged to the candidate in hand. It can
    # now, and the branch at the top of this stage reads it.
    from .. import deps as deps_mod
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "plateau", outputs=["plateau.json"],
                       note=str(rec.get("reason") or "the ground was levelled"))
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


def stage_ground(rnd, be, results: dict) -> dict:
    """**The ground this design asks for -- proposed from the design, applied from the
        baseline.** The design round's second contract, and the stage `stage_plateau` was
        doing the job of before there was a design to do it from.

        `stage_plateau` cuts **before** the plan, sized by the site search's demand (a
        default of 48) and faced in whatever voice the spec carried at the time, and nothing
        afterwards re-sizes or re-paves it. The expression round's held-out village failed
        its read on exactly that: a stone chapel standing on twelve cottage footprints of
        black paving, with `shrink_anchor` tried and rolled back because the re-solve lost
        nine houses. The cut was not the design's; it was the search's.

        So this runs **after** the plan, when the anchor the layout actually drew exists:

          1. `ground.propose` derives the anchor and its apron from that anchor, the paving
             from the voice the resolved design gives that ground, the levels from the
             layout's terrace record, the protected routes from the circulation, and the
             occupied envelope of every part from the types themselves;
          2. `ground.evaluate` asks, **before anything is cut**, whether the proposal is of
             this baseline, inside the site, no larger than the anchor plus its apron, and
             clear of every protected route and occupied envelope;
          3. `ground.apply` makes the prepared volume **from the immutable baseline**, so a
             revision's cut replaces the previous one instead of adding to it.

        A revision therefore re-proposes and re-applies from the baseline, and the terraces
        below are re-laid because the ground moved under them.

        **Offline only.** The live driver still prepares its ground through `stage_plateau`
        and `stage_terraces`, which write blocks ring by ring through the backend; this
        round did not port that path and does not claim it. A live run is untouched and this
        stage says so rather than pretending.
        
    """
    import shutil
    from .. import deps as deps_mod, ground as ground_mod
    plan = rnd.plan()
    if not plan:
        return {"skipped": "no plan yet: there is no design to ask the ground for"}
    dry = not be.live and getattr(be, "dry_run", False)
    if not dry:
        return {"skipped": ("the live driver prepares its ground through the plateau "
                            "and terrace stages; the proposed-ground path is the dry "
                            "driver's and this round did not port it")}
    spec = rnd.place_spec() or {}
    site = pipeline_site(rnd)
    place = _load_place(rnd) or plan
    decls = _decls_for(rnd, spec)
    # **the baseline is the ground as it was found, and it has to exist before a cut.**
    # `deps.baseline_path` falls back to the working volume where nothing has been cut
    # yet, and applying a proposal onto that would make the prepared ground the next
    # run's "baseline" -- which is how cuts accumulate.
    vp = rnd.rel(rnd.base_volume)
    before = rnd.rel(rnd.base_volume.replace(".npz", deps_mod.BASELINE))
    if not os.path.exists(before) and os.path.exists(vp):
        shutil.copyfile(vp, before)
    base_p = deps_mod.baseline_path(rnd)
    # A concentric place carries a podium in its terrace record; a village gathered
    # about a square carries none, and its centre is still designed ground that has to
    # be level -- the plateau stage settled one, and what this stage changes is the
    # *extent* and the *facing*, not the fact that the ground was levelled.
    rec = plateau_record(rnd) or {}
    y = ((rec.get("plateau") or {}).get("y") if isinstance(rec.get("plateau"), dict)
         else rec.get("y"))
    try:
        proposal = ground_mod.propose(spec, site, place, base_p,
                                      allocation=(place.get("layout") or {}).get("allocation"),
                                      decls=decls,
                                      anchor_level=int(y) if y is not None else None)
    except Exception as e:                        # noqa: BLE001 -- the owner reports
        return {"status": "error", "stop": True,
                "error": f"the ground proposal could not be made: {type(e).__name__}: {e}"}
    p = rnd.rel("ground_proposal.json")
    prev = json.load(open(p)) if os.path.exists(p) else None
    fresh, why = deps_mod.check(rnd, "ground", plan=plan)
    if prev and prev.get("print") == proposal.get("print") and fresh:
        return {"skipped": f"the prepared ground is this design's own: {why}",
                "print": proposal.get("print"),
                "pieces": len(proposal.get("pieces") or [])}
    verdict = ground_mod.evaluate(proposal, place, base_p)
    if not verdict.get("ok"):
        json.dump({**proposal, "evaluated": verdict}, open(p, "w"), indent=1)
        return {"status": "error", "stop": True,
                "error": ("the ground this design asks for was refused before anything "
                          "was cut: " + "; ".join(str(w) for w in verdict.get("why") or [])),
                "conflicts": verdict.get("conflicts"), "written": p}
    prepared = ground_mod.apply(proposal, offline.load_volume(base_p))
    offline.save_volume(prepared, vp)
    if hasattr(be, "refresh"):
        be.refresh()
    json.dump({**proposal, "evaluated": verdict}, open(p, "w"), indent=1)
    # **What was laid on this ground goes with it.** The terraces are cut into the
    # prepared ground and the lanes are routed over it: a lane stance is a y as well as
    # an x and a z, and ground remade under a network of them leaves every stance in the
    # air or under the turf. Found by running the farm: a voice revision re-proposed the
    # ground, the apply went back to the baseline as it must, and the construction check
    # came back with 2,663 of 5,243 lane cells whose recorded stance no longer matched
    # the ground -- 1,060 of them with the surface block simply gone. Both records are
    # set aside by name so the stages that own them make them again on the ground that
    # now exists. ...**and the snapshot the lanes are routed from.** `stage_circulation`
    # keeps `world.before-lanes.npz` -- the ground as it stood before the first lane --
    # and routes from it every time, so a replan's lanes replace the previous lanes
    # instead of joining them. That snapshot is of the ground this stage has just
    # replaced, and saving it back over the base discards the new cut: an independent
    # reader found the farm's square still paved in the voice its revision had
    # abandoned, 1,175 apron columns of it, with `ground_proposal.json` truthfully
    # reporting 11,693 drystone blocks laid from the baseline that never reached the
    # world. Ground that moved invalidates the snapshot of it.
    moved = bool(prev and prev.get("print") != proposal.get("print"))
    dropped = []
    if moved:
        for f in ("terraces.json", "network.json", "circulation.json",
                  rnd.base_volume.replace(".npz", ".before-lanes.npz")):
            if os.path.exists(rnd.rel(f)):
                os.replace(rnd.rel(f), rnd.rel(f"stale.ground.{os.path.basename(f)}"))
                dropped.append(os.path.basename(f))
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "ground", plan=plan,
                       outputs=["ground_proposal.json", rnd.base_volume],
                       note=f"{len(proposal.get('pieces') or [])} piece(s) applied from "
                            f"the baseline")
    laid = (proposal.get("applied") or {}).get("laid") or []
    for row in laid:
        print(f"   ground: {row.get('piece')} {row.get('rect')} at y={row.get('level')} "
              f"in `{row.get('voice')}`", flush=True)
    print(f"   ground: {len(laid)} piece(s) laid from the baseline, "
          f"{(proposal.get('applied') or {}).get('blocks', 0):,} blocks"
          + (f"; {', '.join(dropped)} set aside because the ground moved under them"
             if dropped else ""), flush=True)
    return {"pieces": len(proposal.get("pieces") or []), "laid": len(laid),
            "refused": (proposal.get("applied") or {}).get("refused"),
            "print": proposal.get("print"), "from_baseline": base_p,
            "ground_moved": moved, "dropped": dropped, "written": p,
            "from": proposal.get("from")}


def _load_place(rnd):
    p = rnd.rel("plan.place.json")
    return json.load(open(p)) if os.path.exists(p) else None


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
        # **The same terraces for a re-laid plan are the same ground.** The closure
        # round's transfer case: the preview's repair laid the plan out again, `plan`
        # was invalidated and `ground` with it, and the parts stage refused to build on
        # ground "cut for a different candidate" whose rings, levels and annuli had not
        # moved by a column. Ground work is not idempotent, so nothing is cut again;
        # what is asked is whether the cut on disk is the cut this plan designs, and
        # where it is the ground is stamped for this plan.
        from .. import deps as deps_mod
        fresh, why = deps_mod.check(rnd, "ground", plan=rnd.plan())
        if not fresh and deps_mod.recorded(rnd, "ground"):
            lay = (rnd.plan() or {}).get("layout") or {}
            def _levels(rows):
                return sorted((str(r.get("name") or r.get("ring")), r.get("level"))
                              for r in rows or [])
            same = (got.get("terrace") == lay.get("terrace")
                    and _levels(got.get("rings")) == _levels(lay.get("rings")))
            if same:
                note = _stamp_ground(rnd, rnd.plan(),
                                     "terraces unchanged across a re-laid plan; the cut "
                                     "on disk is this plan's and is stamped for it")
                print(f"   terraces: {note}", flush=True)
                return {"skipped": "already terraced -- and the terrace design is "
                                   "unchanged, so the ground is stamped for this plan",
                        "restamped": note, **got}
            return {"status": "error", "stop": True, **got,
                    "error": (f"the ground was terraced for a different design and a cut "
                              f"cannot be taken back: {why}. A fresh round directory is "
                              f"the way to plan this design on unprepared ground")}
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
    # **The prepared ground is this design's, and is stamped as such.** The review's
    # sixth finding, second half: preparation and planning were separate decisions and
    # the terraces made the plan's own terrain dependency stale -- the plan asked for
    # the ground work and the ground work invalidated the plan. `terrain` is now the
    # baseline, which no stage writes, and the worked volume is an *output* stamped
    # against the plan it was cut for, so `stage_parts` can ask whether the ground it is
    # about to build on was prepared for this candidate.
    from .. import deps as deps_mod
    out["ground_stamp"] = _stamp_ground(rnd, plan, f"terraces: {out['placed']} block(s)")
    if dry:
        # The pieces were laid into the file, not into the backend's cached volume; the
        # circulation after this reads `be.volume` and must read the terraces.
        be.refresh()
    return out


def _stamp_ground(rnd, plan: dict | None, note: str) -> str:
    """Record that the volume on disk is the ground prepared for `plan`."""
    from .. import deps as deps_mod
    base = rnd.rel(rnd.base_volume)
    was = rnd.rel(rnd.base_volume.replace(".npz", deps_mod.BASELINE))
    if not os.path.exists(was) and os.path.exists(base):
        # nothing has cut into it yet, so what is here *is* the baseline
        import shutil
        shutil.copyfile(base, was)
    try:
        deps_mod.stamp(rnd, "ground", outputs=[rnd.base_volume],
                       plan=plan if plan is not None else rnd.plan(), note=note)
    except ValueError as e:
        return f"not stamped: {e}"
    return note


def ground_for_this_design(rnd) -> tuple:
    """`(ok, why)` -- was the ground under this plan prepared for this plan?

        A round that never prepared any ground answers `(True, ...)`: standing on the site
        as it was found is a legitimate design and is not a stale artifact. What is refused
        is ground cut for a *different* candidate, which is what a repaired plan built on
        the previous plan's terraces would be.
        
    """
    from .. import deps as deps_mod
    if not deps_mod.recorded(rnd, "ground"):
        return (True, "the place stands on "
                      "the site as it was found")
    return deps_mod.check(rnd, "ground", plan=rnd.plan())


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
    site = rnd.site or rnd.chosen_site()
    if os.path.exists(p):
        s = json.load(open(p))
        # the briefing is of the chosen site or it is set aside (the closure round)
        if site and (list(s.get("origin") or []) != list(site["origin"])
                     or int(s.get("size") or 0) != int(site["size"])):
            os.replace(p, rnd.rel("site.stale.json"))
            print(f"   site: the briefing on disk is of ({s.get('origin')}) "
                  f"{s.get('size')} and the search chose {site['origin']} "
                  f"{site['size']}; it is set aside and the site is read again", flush=True)
        else:
            return {"skipped": "already prepared -- the site briefing is a fixture",
                    "path": p, "origin": s["origin"], "size": s["size"],
                    "relief": s["stats"]["relief"]}
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


#: Where the adopted arrangements are recorded: one row per district, with the ladder.
ARRANGEMENT_RECORD = "arrangements.json"

#: Which districts have already had their allocation re-asked after a cover shortfall.
#: One per district per run: a second identical ask is a loop, and the driver's re-entry
#: bound should not be what stops it.
COVER_REASK_RECORD = "cover_reasks.json"


def _reask_record(rnd) -> dict:
    p = rnd.rel(COVER_REASK_RECORD)
    return json.load(open(p)) if os.path.exists(p) else {}


def _arrange_districts(rnd, spec: dict, place: dict, site: dict, decls: dict,
                       types, vol=None) -> dict:
    """Give every compiled district the arrangement its ground actually holds.

        One pass, before validation and before the resolution is written. For each district
        the model gave a character, `arrange` compiles the rectangle, climbs the ground and
        fabric rungs where the result is short of the allocator's proposal, and hands back
        what it adopted. This writes that arrangement as the district's plan file -- the one
        the assembler and the builder read -- and moves the district's promise onto it.

        **The promise moves; the request does not.** A place left short of the structures its
        sentence or its kind asked for is short, `short_total` says by how much, and the
        capacity finding that reports it belongs to the scale owner. Nothing here is allowed
        to make a request smaller.

        Idempotent: a district whose plan file is already on disk and current is left alone,
        so re-entering this stage does not re-lay a place that has not moved.
        
    """
    from .. import (arrange as arrange_mod, contracts as contracts_mod,
                    deps as deps_mod, placeplan)
    rows, adopted, short_total = [], False, 0
    seed = int(rnd.flags.get("seed") or 1)
    # what was arranged last time, per district, so a changed character is visible
    by_name = {r.get("district"): r
               for r in (arrangements(rnd).get("districts") or [])}
    for d in (place.get("districts") or []):
        part = placeplan._district_part(spec, d)
        if not part or spec_mod.character(part) is None:
            continue
        dp = rnd.rel(f"plan.district.{d['name']}.json")
        # **A character its author has revised is taken before the ground is arranged.**
        # `district_asks` has always done this, and it runs *after* this pass -- so a
        # revision reached the compiler through that seam with the allocation the old
        # fabric had been given, and the arrangement, the capacity re-ask and the
        # recovery ladder were all skipped. The shore village's revision was rolled back
        # for a cover it missed by forty-two columns while its own rectangle held two
        # more houses than it had been asked for.
        if apply_character(rnd, spec, part):
            for f in (dp, rnd.rel(f"district_{d['name']}_compiled.json")):
                if os.path.exists(f):
                    os.remove(f)
        if os.path.exists(dp):
            continue
        role = spec_mod.district_role(spec, d)
        _t, mine = placeplan.types_card(types, spec.get("form"), role)
        # **What the allocator asked for, once.** Arranging against the count the last
        # arrangement *adopted* is not idempotent and it descends: the compiler sizes
        # its lots from the ask, so asking for the four houses a rectangle gave makes
        # four bigger houses and the next pass measures three. Found by running the loop
        # fixture -- the band walked 81, 58, 53, 52 over four repair passes, each one a
        # true measurement of a district that had been asked for less than the time
        # before. The proposal is the layout's and is remembered; what moves is the
        # promise.
        if d.get("proposed_structures") is None:
            d["proposed_structures"] = int(d.get("structures") or 0)
        # **A change of fabric re-asks the allocation.** The allocator's ask is a number
        # about a rectangle *built a particular way*: halve the lot and the same ground
        # holds more houses, and holding the old ask fixed means the district covers
        # less of itself and its own validator refuses it. Found by running the shore
        # village's revision -- the inspection asked for "more, smaller houses", the
        # character delivered smaller lots, the count stayed at three, the cover fell to
        # 14% against a floor of 38% and the revision was rolled back for doing exactly
        # what it was asked to do. So where the character has moved since the
        # arrangement on record, the proposal is asked of the construction logic again
        # -- `arrange.capacity`, the same compiler -- and the district may hold what its
        # new fabric actually fits. The allocator's own ask is the floor, so a revision
        # can add houses and never take the programme's distribution away.
        ch_print = contracts_mod.digest(spec_mod.character(part))
        was_print = (by_name.get(d["name"]) or {}).get("character_print")
        asked = int((d.get("reasked") or {}).get("to") or d["proposed_structures"])
        if was_print is not None and was_print != ch_print and not d.get("exact"):
            held = arrange_mod.capacity(d, part, place, mine or decls, spec=spec,
                                        seed=seed)
            if held > asked:
                print(f"   arranged {d['name']}: the character changed and this "
                      f"rectangle now fits {held}; the allocation is re-asked from "
                      f"{asked}", flush=True)
                # **The proposal is never rewritten.** The closure round: a re-ask is a
                # recorded decision beside the allocator's original ask, so regenerating
                # from the record reproduces the same arrangement and the original is
                # still readable.
                d["reasked"] = {"from": asked, "to": int(held),
                                "why": "the character changed and the rectangle fits more"}
                asked = int(held)
        # the district is refused on its count **and** its cover, so the ladder answers
        # both: `district_target` is the one place either number is derived **The
        # validator's own number, from the validator's own call.** Computed with the
        # role-filtered table instead, this came out at 948 where `district_failures`
        # refused at 1099 -- `district_target`'s `usable_columns` reads the declarations
        # it is given -- and the ladder therefore stopped on a district it believed
        # covered its ground and the validator did not. Two derived numbers about the
        # same thing, disagreeing, which is the defect this whole round is about; the
        # place-level table is what the refusal uses, so it is what the ladder is
        # measured against.
        floor = 0
        with contextlib.suppress(Exception):
            floor = int(placeplan.district_target(d, part, place,
                                                  decls)["min_plot_columns"])
        extra = ({"intent": contracts_mod.load(rnd, "intent")}
                 if "intent" in arrange_mod.arrange.__code__.co_varnames else {})
        got = arrange_mod.arrange(d, part, place, mine or decls, spec=spec, site=site,
                                  seed=seed, proposed=int(asked),
                                  cover_floor=floor, **extra)
        if not got["ok"]:
            return {"status": "error", "stop": True, "level": f"district/{d['name']}",
                    "arrangement": {k: got.get(k) for k in
                                    ("proposed", "realized", "attempts", "limit")},
                    "error": (f"no arrangement can be laid in {d['name']}: "
                              f"{got['why']}")}
        # the rectangle and the pool the ladder settled on become this district's
        if got.get("adopted_rect"):
            if got.get("grew"):
                d.setdefault("extent_from", [int(d["x0"]), int(d["z0"]),
                                             int(d["x1"]), int(d["z1"])])
            d["x0"], d["z0"], d["x1"], d["z1"] = [int(v) for v in got["adopted_rect"]]
        if got.get("adopted_fabric"):
            d["fabric_types"] = list(got["adopted_fabric"])
        was = int(d.get("structures") or 0)
        # **A region the ground fits fewer than a district's worth of houses in is not a
        # district.** It becomes open ground with an owner: still part of the place,
        # still counted by the coverage clause, asked for nothing and held to nothing it
        # cannot meet. The same rule `placesolve` and `placeshore` apply to their own
        # thin sectors, applied where the measurement actually is.
        d["structures"] = 0 if got["thin"] else int(got["realized"])
        if got["thin"]:
            d["purpose"] = (f"Open ground with an owner: the arrangement holds "
                            f"{got['realized']} house(s), under the {got['least']} a "
                            f"district is held to. {d.get('purpose') or ''}").strip()
        else:
            json.dump(got["plan"], open(dp, "w"), indent=1)
            json.dump(got["record"],
                      open(rnd.rel(f"district_{d['name']}_compiled.json"), "w"), indent=1)
        _drop_assembled(rnd)
        adopted = True
        short_total += int(got["short"])
        rows.append({**{k: got.get(k) for k in
                        ("district", "part", "proposed", "realized", "short", "thin",
                         "grew", "adopted_rect", "adopted_fabric", "types", "lot",
                         "house", "plot_cover", "ground_cover", "undeveloped_share",
                         "open_land", "streets", "attempts", "limit",
                         # **why a region is short, in the compiler's own terms**, with
                         # a field of its own rather than only at the head of `limit`
                         "why_short",
                         "covers", "cover_floor")},
                     # the fabric this arrangement was made of, so the next pass can
                     # tell a changed character from an unchanged one
                     "character_print": ch_print})
        moved = [a for a in got["attempts"][1:] if a.get("changed")]
        print(f"   arranged {d['name']}: proposed {was}, realized {got['realized']}"
              + (f" after {', '.join(a['action'] for a in moved)}" if moved else "")
              + (f"; open ground with an owner" if got["thin"] else "")
              + (f"; short {got['short']}" if got["short"] else ""), flush=True)
    if rows:
        rec = {"candidate": deps_mod.candidate_id(rnd), "districts": rows,
               "short_total": int(short_total),
               "promised": sum(int(d.get("structures") or 0)
                               for d in place.get("districts") or []),
               "note": ("each district's promise is the arrangement its own ground "
                        "holds, laid by the construction logic that builds it; the "
                        "ladder tried ground and fabric before any number moved")}
        json.dump(rec, open(rnd.rel(ARRANGEMENT_RECORD), "w"), indent=1)
    return {"adopted": adopted, "rows": rows, "short_total": int(short_total)}


def arrangements(rnd) -> dict:
    p = rnd.rel(ARRANGEMENT_RECORD)
    return json.load(open(p)) if os.path.exists(p) else {}


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


#: How many districts one run may re-allocate before it stops and reports what is left.
#: A bound on a **layout** repair, registered here beside the code that spends it: the
#: action reconciles a promise with a measured capacity, and a run that has to do it
#: sixteen times has a layout problem and not sixteen allocation problems.
LAYOUT_REALLOCATIONS = 8

#: Where the layout owner's re-allocations are recorded, so the bound survives the stage
#: being re-entered and a reader can see what was moved and why.
REALLOCATION_RECORD = "layout_repairs.json"


def _owner_attempts(rnd, district: str) -> dict:
    """Every layout action tried on this district, applied and refused."""
    rec = _reallocations(rnd)
    return {"applied": [{k: a.get(k) for k in ("action", "from", "to", "candidate")}
                        for a in rec.get("applied") or []
                        if a.get("district") == district],
            "refused": [a.get("why") for a in rec.get("refused") or []
                        if a.get("district") == district]}


def _owner_says(rnd, district: str) -> str:
    """One clause naming what the owner tried, for the stop's own message."""
    got = _owner_attempts(rnd, district)
    if not got["applied"] and not got["refused"]:
        return ""
    did = ", ".join(str(a.get("action") or "allocation") for a in got["applied"])
    return (f". The layout owner "
            + (f"applied {did}" if did else "had no action it could apply")
            + (f" and refused {len(got['refused'])} further change(s), the first "
               f"because {got['refused'][0][:140]}" if got["refused"] else "")
            + "; the limiting constraint is this district's own ground")


def _reallocations(rnd) -> dict:
    p = rnd.rel(REALLOCATION_RECORD)
    return json.load(open(p)) if os.path.exists(p) else {"applied": [], "refused": []}


#: How far a district may grow into the free ground beside it, as a share of its own
#: area, and the least free margin worth taking. A bound on an **extent** repair.
DISTRICT_GROWTH_MAX = 0.5
DISTRICT_GROWTH_MIN = 4


def _occupied(place: dict, mine: str) -> list:
    """Every rectangle of the place level except `mine`'s: what growth may not take."""
    out = []
    for key in ("districts", "compounds"):
        for r in place.get(key) or []:
            if r.get("name") != mine and r.get("x1") is not None:
                out.append([int(r["x0"]), int(r["z0"]), int(r["x1"]), int(r["z1"])])
    for p in place.get("parts") or []:
        with contextlib.suppress(Exception):
            out.append([int(v) for v in _pipeline.part_rect(p)])
    return out


def _grow_into_free_ground(place: dict, d: dict, site: dict) -> list | None:
    """A larger rectangle for `d` inside the site, touching nothing else. None if none.

        **The spatial alternative the recovery did not have.** The review's fourth finding:
        the allocation action "does not search alternative region geometry, lot types, site
        extent or ground works", so a district that could not hold its promise had exactly
        one answer -- promise less -- and a district promised one house could become zero
        and take the residential demand with it. A promise that a rectangle cannot keep is
        as much a question about the rectangle as about the promise.

        Grown one side at a time, each as far as the nearest other region or the site edge,
        within `DISTRICT_GROWTH_MAX` of its own area. A ring sector is **not** grown: its
        rectangle is a chord of an annulus and moving it is a decision the ring arithmetic
        owns, which is a different action from this one and is named as unavailable rather
        than attempted badly.
        
    """
    if d.get("ring") is not None or (d.get("level") is not None
                                     and place.get("layout", {}).get("rings")):
        return None
    ox, oz = int(site["origin"][0]), int(site["origin"][1])
    n = int(site["size"])
    bound = [ox, oz, ox + n - 1, oz + n - 1]
    x0, z0, x1, z1 = int(d["x0"]), int(d["z0"]), int(d["x1"]), int(d["z1"])
    area0 = (x1 - x0 + 1) * (z1 - z0 + 1)
    # **Bounded against the rectangle this district was first laid out as**, not against
    # the one the last growth produced. The same arithmetic the review found the scale
    # negotiation getting wrong -- a limit re-applied to its own output is a decay rate
    # rather than a limit -- and it is the same arithmetic in the opposite direction:
    # run three times, a half-again bound grows a district to two and a quarter times
    # its size, which is a different district.
    was = d.get("extent_from") or [x0, z0, x1, z1]
    origin = (int(was[2]) - int(was[0]) + 1) * (int(was[3]) - int(was[1]) + 1)
    most = int(origin * (1.0 + DISTRICT_GROWTH_MAX))
    if area0 >= most:
        return None
    others = _occupied(place, d.get("name"))

    def clear(rect):
        if not (bound[0] <= rect[0] and rect[2] <= bound[2]
                and bound[1] <= rect[1] and rect[3] <= bound[3]):
            return False
        return not any(rect[0] <= o[2] and o[0] <= rect[2]
                       and rect[1] <= o[3] and o[1] <= rect[3] for o in others)

    got = [x0, z0, x1, z1]
    for i, step in ((0, -1), (1, -1), (2, 1), (3, 1)):
        while True:
            nxt = list(got)
            nxt[i] += step
            if (nxt[2] - nxt[0] + 1) * (nxt[3] - nxt[1] + 1) > most or not clear(nxt):
                break
            got = nxt
    grown = (got[2] - got[0] + 1) * (got[3] - got[1] + 1)
    return got if grown - area0 >= DISTRICT_GROWTH_MIN * max(
        got[2] - got[0] + 1, got[3] - got[1] + 1) else None


def _smaller_fabric(rnd, spec: dict, d: dict, decls: dict) -> list | None:
    """A smaller approved plot type for this district's fabric, or None.

        The **capability** alternative. The compiler reaches for the widest-envelope type of
        the district's role first, which is the right default and the wrong answer in a
        thirty-column strip: a type whose least footprint is smaller is a different spatial
        answer to the same promise, and it is one the capability record already approved.
        
    """
    from ..district_compile import _plot_range
    pool = d.get("fabric_types")
    if not pool or len(pool) < 2:
        return None
    sized = []
    for name in pool:
        decl = decls.get(name)
        if not decl or decl.get("kind", "plot") != "plot":
            continue
        with contextlib.suppress(Exception):
            lo, _hi, _ex = _plot_range(decl)
            sized.append((lo, name))
    sized.sort()
    if len(sized) < 2 or sized[0][1] == pool[0]:
        return None
    return [n for _lo, n in sized]


def _district_capacity_repair(rnd, spec: dict, place: dict, d: dict, laid: int,
                              dfails: list, site: dict | None = None,
                              decls: dict | None = None) -> dict | None:
    """Reconcile one district's promise with the capacity its own ground gave.

        **The allocation repair, and the general rule behind the city's stop.** A ring
        sector 30 columns across was promised two lots; compilation laid one and covered
        693 of 5,670 columns against the validator's floor of 1,149, and the run asked the
        district's character's author for better adjectives -- twice -- and then stopped. No
        adjective makes a 30-column strip hold two lots. The promise was wrong, it was made
        by the allocator from an area estimate, and the allocator is the layer that can move
        it.

        So: the district's `structures` becomes what the ground actually realized, and a
        region that realizes fewer than `DISTRICT_MIN_STRUCTURES` becomes **open ground with
        an owner** -- asked for nothing, still part of its ring, still counted by the
        coverage clause. That is the rule `placesolve` and `placeshore` have had for their
        own thin sectors all along; this is the ring layout learning it, at compile time,
        where the measurement actually is.

        This never touches a sentence's count. What the place promises in total falls, and
        that surfaces as a `capacity` finding routed to `scale`, which refuses to negotiate
        an explicit count. Reducing an inferred allocation is not the same act as reducing
        the request, and the two stay separate records on purpose.

        Returns the change, or None where this is not this owner's finding.
        
    """
    from .. import placeplan
    kinds = {f.get("check") for f in dfails}
    # **An exact district over the density's ceiling is the layout's, not the
    # character's.** The closure round: the count is the sentence's and stands; the
    # rectangle is what can move. Grow it into free ground; where none is free, refuse
    # by name rather than asking a model for different adjectives.
    if "cover_over" in kinds and d.get("exact"):
        if site:
            bigger = _grow_into_free_ground(place, d, site)
            if bigger is not None:
                return {"refused": False, "district": d["name"], "action": "extent",
                        "from": int(d.get("structures") or 0),
                        "to": int(d.get("structures") or 0), "open_ground": False,
                        "thin": False, "rect": bigger,
                        "why": (f"{d['name']} holds its exact {d.get('structures')} "
                                f"house(s) over the density's ceiling; the count is the "
                                f"sentence's, so the ground grows into the free ground "
                                f"beside it"),
                        "failures": [f.get("check") for f in dfails]}
        return {"refused": True, "district": d["name"], "action": "extent",
                "why": (f"{d.get('structures')} house(s) cannot be laid as "
                        f"`{(placeplan._district_part(spec, d) or {}).get('density')}` on "
                        f"this rectangle and no free ground lies beside it: the count is "
                        f"the sentence's and the density word cannot be met on this "
                        f"ground")}
    if not (kinds & {"cover", "ground_cover", "farmland_cover", "count"}):
        return None
    promised = int(d.get("structures") or 0)
    least = placeplan.DISTRICT_MIN_STRUCTURES
    #: **A region the ground fits fewer than a district's worth of houses in is not a
    #: district.** `placesolve` and `placeshore` have had this rule for their own thin
    #: sectors since v2; the ring layout did not, and the city's 30x189 strip is what
    #: that costs: promised one house, laid one house, and then held to a *district's*
    #: plot cover -- 693 of 5,670 columns against a floor of 1,149 -- which one house in
    #: a strip thirty columns across cannot reach at any density and no adjective can
    #: change. The strip is not short of houses; it is not a district. It becomes open
    #: ground with an owner: still part of its ring, still counted by the ring's
    #: coverage clause, asked for nothing and held to nothing it cannot meet.
    thin = (laid == promised and 0 < promised < least
            and bool(kinds & {"cover", "ground_cover", "farmland_cover"}))
    if laid >= promised and not thin:
        return None
    rec = _reallocations(rnd)
    if len(rec["applied"]) >= LAYOUT_REALLOCATIONS:
        return {"refused": True,
                "why": (f"{len(rec['applied'])} district(s) have already been "
                        f"re-allocated this run, which is the registered bound of "
                        f"{LAYOUT_REALLOCATIONS}; the layout is wrong at a level this "
                        f"action cannot reach")}
    # **What has been tried on *this* district of *this* candidate.** The record
    # survives the whole run, and a candidate the run has since replaced took its
    # actions with it: scoping by district alone meant a district whose extent had been
    # grown on a design that no longer exists was refused the action on the design that
    # does. The same rule as the plan-level budget, and the reason `candidate` is
    # stamped on every applied reallocation.
    from .. import deps as _deps_t
    here = _deps_t.candidate_id(rnd)
    tried = {a.get("action", "allocation") for a in rec["applied"]
             if a.get("district") == d["name"] and a.get("candidate") == here}
    # **A spatial answer before a smaller promise.** The order is the point: the ground
    # and the types are what a layout owner can actually change, and moving the promise
    # is what is left when neither will move. Each action is tried once per district --
    # a second attempt at the same lever on the same rectangle is a loop.
    if site and "extent" not in tried:
        bigger = _grow_into_free_ground(place, d, site)
        if bigger is not None:
            w0, d0 = d["x1"] - d["x0"] + 1, d["z1"] - d["z0"] + 1
            return {"refused": False, "district": d["name"], "action": "extent",
                    "from": promised, "to": promised, "open_ground": False,
                    "thin": False, "rect": bigger,
                    "why": (f"the allocator promised {promised} structure(s) in "
                            f"{w0}x{d0} columns and the compiler realized {laid}; there "
                            f"is free ground beside this district and it is taken -- "
                            f"{bigger[2] - bigger[0] + 1}x{bigger[3] - bigger[1] + 1} "
                            f"columns, touching no other region and inside the site. "
                            f"The promise is unchanged: what moves is the ground it was "
                            f"made about"),
                    "failures": [f.get("check") for f in dfails]}
    if decls and "fabric" not in tried:
        pool = _smaller_fabric(rnd, spec, d, decls)
        if pool is not None:
            return {"refused": False, "district": d["name"], "action": "fabric",
                    "from": promised, "to": promised, "open_ground": False,
                    "thin": False, "fabric_types": pool,
                    "why": (f"the allocator promised {promised} structure(s) and the "
                            f"compiler realized {laid} out of `{d['fabric_types'][0]}`, "
                            f"the widest type the capability record approved here; the "
                            f"pool is re-ordered smallest-footprint first (`{pool[0]}`) "
                            f"and the district is compiled again. The promise is "
                            f"unchanged and so is what the record approved"),
                    "failures": [f.get("check") for f in dfails]}
    to = 0 if (thin or laid < least) else laid
    # **And a conversion that takes the place below what was accepted is refused.** The
    # review: "A district that lays its promised one house but fails cover can become
    # zero-target open ground, removing the residential cover demand. This can be a
    # legitimate design revision only if the broader programme still holds."
    if to < promised:
        from .. import repair as repair_mod
        rest = sum(int(o.get("structures") or 0) for o in place.get("districts") or []
                   if o.get("name") != d["name"]) + to
        band = repair_mod.accepted_band(spec)
        floor = int(round((band[0] if band else 0) * repair_mod.SCALE_NEGOTIATION_MIN))
        if floor and rest < floor:
            return {"refused": True, "district": d["name"], "action": "allocation",
                    "why": (f"moving {d['name']} from {promised} to {to} would leave "
                            f"the place promising {rest} structures against the "
                            f"{floor} that is {repair_mod.SCALE_NEGOTIATION_MIN:.0%} of "
                            f"the {band[0]} floor this build first accepted for a "
                            f"{spec.get('kind')}; the district cannot hold its promise "
                            f"and the place cannot afford to give it up, which is a "
                            f"layout failure and not an allocation to revise")}
    w, dep = d["x1"] - d["x0"] + 1, d["z1"] - d["z0"] + 1
    why = (f"the allocator promised {promised} structure(s) in {w}x{dep} columns, and "
           + (f"compiling that rectangle with the actual streets, lots and setbacks "
              f"realized {laid}" if laid < promised else
              f"the {laid} it realized cannot cover a district's ground in a rectangle "
              f"this shape"))
    return {
        "refused": False, "district": d["name"], "from": promised, "to": to,
        "open_ground": to == 0, "thin": bool(thin),
        "why": why + (f"; under {least} structures it is open ground with an owner "
                      f"rather than a district held to a district's count and cover, "
                      f"which it can meet neither of"
                      if to == 0 else "; the promise is moved to the measured capacity"),
        "failures": [f.get("check") for f in dfails]}


def _apply_reallocation(rnd, place: dict, got: dict) -> None:
    """Write a re-allocation onto the place level and drop what it invalidates."""
    from .. import deps as deps_mod
    for d in place.get("districts") or []:
        if d["name"] == got["district"]:
            d["structures"] = int(got["to"])
            d["notes"] = (d.get("notes") or "") + (
                f" Re-allocated by the layout owner: {got['why']}.")
            # the spatial and capability alternatives write geometry and a pool; the
            # allocation action writes a count. All three are this owner's and all three
            # are recorded in the same place
            if got.get("rect"):
                d.setdefault("extent_from",
                             [int(d["x0"]), int(d["z0"]), int(d["x1"]), int(d["z1"])])
                d["x0"], d["z0"], d["x1"], d["z1"] = [int(v) for v in got["rect"]]
            if got.get("fabric_types"):
                d["fabric_types"] = list(got["fabric_types"])
            if got["open_ground"]:
                d["purpose"] = (f"Open ground inside this ring, with an owner. "
                                f"{d.get('purpose') or ''}")
    json.dump(place, open(rnd.rel("plan.place.json"), "w"), indent=1)
    for f in (f"plan.district.{got['district']}.json",
              f"district_{got['district']}_compiled.json",
              f"district_{got['district']}_prompt.md",
              "plan.json", "plots.json", "network.json", "circulation.json"):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    rec = _reallocations(rnd)
    from .. import deps as _deps_c
    got = {**got, "candidate": _deps_c.candidate_id(rnd)}
    rec["applied"].append(got)
    json.dump(rec, open(rnd.rel(REALLOCATION_RECORD), "w"), indent=1)
    # **Stamped again, not invalidated.** Found by running it: invalidating `plan` made
    # the next entry of this stage call the place level stale, move it aside and solve
    # it again from the spec -- which threw the re-allocation away and produced the same
    # promise, which failed the same way, three times in a row until the bound stopped
    # it. Nothing this artifact depends on has moved; the artifact was *deliberately
    # edited* by its owner, and the stamp has to say so, or the freshness rule that
    # exists to protect a repair is what undoes it.
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "plan", outputs=["plan.place.json"],
                       note=f"{got['district']} re-allocated {got['from']} -> "
                            f"{got['to']} by the layout owner")


def _resolve_record(rnd, spec: dict, place: dict, site: dict,
                    caps: dict | None = None, plan: dict | None = None,
                    realized: dict | None = None) -> tuple:
    """Write `resolution.json` and `findings.json` for a laid-out place level.

        Returns the findings record. Never raises: a round whose sentence states no
        requirement gets an empty findings record, which is a true statement about it.
        
    """
    from .. import contracts, deps, intent as intent_mod, resolve as resolve_mod
    it = contracts.load(rnd, "intent")
    if it is None:
        it = intent_mod.read(spec.get("sentence") or rnd.sentence or "")
        contracts.save(rnd, "intent", it)
    from .. import capability as cap_mod
    reading = contracts.load(rnd, "reading")
    # **The capability record is reconciled against the place before it is read.**
    # Matching wrote down which type answers each part; the layout then built the place.
    # `agreements` makes the record follow the place and turns a type the capability
    # rules would refuse into a finding instead of a silence.
    if caps is not None:
        # **With the assembled tree where there is one.** The review's third finding was
        # exactly this argument: `agreements` grew a `plan` parameter, the regression
        # exercised it, and its only production caller went on passing three arguments
        # -- so the fabric a district was compiled out of and the halls inside a
        # compound were never reconciled with anything. A helper the production path
        # does not call with the thing it needs is a helper that has not been connected.
        caps, cap_rows = cap_mod.agreements(caps, place, _decls_for(rnd, spec),
                                            plan=plan)
        contracts.save(rnd, "capabilities", caps)
    else:
        cap_rows = []
    checked, resolution, findings = resolve_mod.findings_for(
        spec, place, site, it, capabilities=caps, reading=reading, plan=plan,
        decls=_decls_for(rnd, spec))
    if realized is not None:
        resolution, realized_rows = resolve_mod.with_realized(
            resolution, realized, plan=plan,
            subject=((spec.get("explicit_count") or {}) or {}).get("what"))
        # the capacity check runs **again**, against what the ground gave rather than
        # against what the allocator budgeted; the first answer was about a number that
        # no longer describes this plan
        rows = [f for f in findings["findings"]
                if not f["id"].startswith(("find/capacity/", "find/extent/"))]
        short = resolve_mod.capacity_findings(spec, place, resolution, checked)
        rows += short
        rows += resolve_mod.unclaimed_ground(spec, place, site, short)
        findings = contracts.make("findings", findings=[*rows, *realized_rows],
                                  stage="resolve.findings_for+realized",
                                  note=findings["note"])
    # **What the districts could not cover, said once, where a reader looks.** A
    # district at the capacity of its own rectangle is not refused for its cover (see
    # `_arrange_districts`), and the obligation does not thereby disappear: it becomes a
    # finding owned by the layout, which is the layer that chose the rectangle.
    cover_rows = []
    for d in (place.get("districts") or []):
        lim = d.get("cover_limited")
        if not lim:
            continue
        cover_rows.append({
            "id": f"find/cover/{d['name']}", "requirement": None, "part": d["name"],
            "says": f"{d['name']}: {lim['why']}",
            # **Fidelity and not feasibility.** The plan can be built -- every house in
            # it stands on ground that holds it -- and what is short is how much of its
            # own rectangle the district covers, which is a statement about the place
            # being less closely built than its density word implies. Blocking
            # feasibility on it would stop a buildable design for a quality it is
            # already reporting, and the layout owner's two spatial actions have been
            # tried and measured by the time this is written.
            "evidence": dict(lim), "owner": "layout", "blocks": "fidelity",
            "severity": "warning", "seen_by": "arrange.cover", "fixed": False})
    if cap_rows or cover_rows:
        findings = contracts.make("findings",
                                  findings=[*findings["findings"], *cap_rows,
                                            *cover_rows],
                                  stage=findings["stage"], note=findings["note"])
    # the programme's entities, bound once and carried on the intent record
    from .. import spec as _spec_e
    checked["entities"] = _spec_e.entities(spec)
    contracts.save(rnd, "intent", checked)
    contracts.save(rnd, "resolution", resolution)
    contracts.save(rnd, "findings", findings)
    with contextlib.suppress(ValueError):
        deps.stamp(rnd, "resolution",
                   outputs=["resolution.json", "findings.json"],
                   note=resolution["policy"])
    print(f"   resolved: {resolve_mod.says(resolution, findings)}", flush=True)
    for f in findings["findings"][:6]:
        print(f"     - {f['owner']}/{f['blocks']}: {f['says'][:120]}", flush=True)
    return caps, findings


def _decls_for(rnd, spec: dict) -> dict:
    from .. import placeplan
    _t, decls = placeplan.types_card(rnd.flags.get("types"), spec.get("form"))
    return decls or {}


#: Where the plan stage's own repair passes are recorded, and the bound on them. Two:
#: the point is to close a diagnosis this stage produced, not to search for a place.
PLAN_REPAIR_RECORD = "plan_repairs.json"
#: Per **candidate**: how many passes one design gets before its remaining findings are
#: reported rather than chased.
PLAN_REPAIR_BUDGET = 2
#: Per **run**: the cap that bounds the loop, because every applied repair produces a
#: new candidate and so a fresh per-candidate budget.
PLAN_REPAIR_TOTAL = 6


def _plan_repair(rnd, be, spec: dict, place: dict, found: dict) -> dict | None:
    """One bounded repair pass over the plan level's own findings, or None.

        Returns a stage result asking the driver to re-enter where something changed, and
        None where nothing did -- which is the ordinary case and costs one pass over a list.
        
    """
    from .. import repair as repair_mod
    rows = [f for f in (found or {}).get("findings") or []
            if f["blocks"] == "feasibility" and f["owner"] in repair_mod.ACTS_ON
            and not f.get("fixed")]
    if not rows:
        return None
    p = rnd.rel(PLAN_REPAIR_RECORD)
    was = json.load(open(p)) if os.path.exists(p) else {"passes": []}
    # **The budget belongs to the candidate, under a cap that belongs to the run.**
    # Found by running the held-out village: a pass spent on a candidate a crash had
    # left behind counted against the candidate that replaced it, so the design that was
    # actually in hand got one repair instead of two and the round stopped with a
    # finding its own owner had an action for. A budget that cannot say which design it
    # was spent on is a number a rollback restores to the wrong place -- which is the
    # round's "repair budgets belong to a candidate identity". The run-wide cap is what
    # still bounds the loop, because every repair makes a new candidate.
    from .. import deps as deps_mod
    now = deps_mod.candidate_id(rnd)
    #: The lineage in hand. A deliberate revision by the principal opens a new one
    #: (`stages_media._new_repair_lineage`); everything else shares the run's first.
    line = int(was.get("lineage") or 0)
    here = [q for q in was["passes"] if int(q.get("lineage") or 0) == line]
    mine = [q for q in here if q.get("candidate") == now]
    if len(mine) >= PLAN_REPAIR_BUDGET:
        return None
    if len(here) >= PLAN_REPAIR_TOTAL:
        print(f"   plan: {len(here)} repair pass(es) on this design is the "
              f"registered cap of {PLAN_REPAIR_TOTAL}"
              + (f" ({len(was['passes'])} this run, across {line + 1} lineage(s))"
                 if len(here) != len(was["passes"]) else ""), flush=True)
        return None
    place_p = rnd.rel("plan.place.json")
    got = repair_mod.apply(rnd, spec, {**found, "findings": rows},
                           place=place, place_path=place_p)
    from .. import deps as _deps_c
    was["passes"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "lineage": line,
                          # **whose candidate this pass was spent on.** A budget that
                          # cannot say which design it was spent repairing is a number a
                          # rollback restores to the wrong place.
                          "candidate": _deps_c.candidate_id(rnd),
                          "findings": [f["id"] for f in rows],
                          "applied": got["applied"], "refused": got["refused"]})
    json.dump(was, open(p, "w"), indent=1)
    if not got.get("changed_plan"):
        return None
    from .. import deps as deps_mod
    # **The place that measured the capacity is the place that is kept.** Found by
    # running it: a scale repair moved the band's floor down to the capacity the shore
    # band gave, the place was then solved *again from the smaller target*, the band's
    # depth negotiation had less to carry, the new place held fewer houses still -- and
    # the same finding came back one size smaller, three passes running. A repair that
    # re-derives the thing it was repairing from the repaired value is a descent and not
    # a fix. What the repair changes is the programme's inferred target; the geometry
    # that measured it is evidence, and evidence is not re-derived. So the place level
    # stands and is stamped again -- its `spec` input moved on purpose, by its owner --
    # and only what is downstream of the target is dropped: the districts, whose lot
    # sizes come from it, and the plan assembled out of them.
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "plan", outputs=["plan.place.json"],
                       note=f"kept across a plan-level repair: {repair_mod.says(got)}")
    for f in ("plan.json", "plots.json", "network.json", "circulation.json"):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    for f in sorted(os.listdir(rnd.state)):
        if f.startswith(("plan.district.", "district_")) and os.path.isfile(rnd.rel(f)):
            os.remove(rnd.rel(f))
    print(f"   plan: {repair_mod.says(got)}; the place is resolved again", flush=True)
    return {"plan": {"status": "reenter", "level": "place", "repair": got,
                     "why": (f"a repair changed a planning decision at the place level "
                             f"({repair_mod.says(got)}); the plan is laid out and "
                             f"resolved again before anything is compiled")}}


def site_capability_facts(rnd, spec: dict, site: dict | None) -> dict:
    """The site's and this place's facts that matching has to ask `fits` about.

        `ground` is the class the parts stand on, `relief` is how much the ground falls
        across the place, and `round_boundaries` says whether this place's walls are drawn
        round -- which needs types that draw a diagonal run. All three are already read
        somewhere in this module and none of them reached the matcher.
        
    """
    from .. import intent as intent_mod, placeplan
    facts = intent_mod.site_facts(site)
    relief = facts["relief"]
    water = facts["water"]
    ground = None
    if relief is not None and water is not None:
        ground = ground_class(int(relief), float(water) * 100.0)
    # round where any wall this place declares is described as round: `wall_round_for`
    # reads the wall part's own words beside the spec's invariants, which is where a
    # spec that means a ring rather than a rectangle says so
    walls = [p for p in (spec or {}).get("defining_parts") or []
             if p.get("family") == "wall"]
    return {"ground": ground, "relief": relief,
            "round_boundaries": any(placeplan.wall_round_for(w, spec) for w in walls)}


#: What blocks a plan being reported planned. A finding that says it blocks feasibility
#: blocks feasibility whatever its severity: the review found the city reporting
#: `planned` with two `blocks: feasibility` district shortfalls open, on the grounds
#: that both were warnings. Severity is how loudly a finding is said and `blocks` is
#: what it stops; reading the first as the second is how a warning became a pass.
def blocking(found: dict | None, state: str = "feasibility") -> list:
    """The open findings that block `state`. Empty is what a clean plan looks like."""
    return [f for f in (found or {}).get("findings") or []
            if f.get("blocks") == state and not f.get("fixed")]


def stage_plan_levels(rnd, be, results: dict, spec: dict) -> dict:
    """Plan the place, then plan each district. A5."""
    from .. import contracts as contracts_mod, pipeline, placeplan, spec as spec_mod
    site = pipeline.settlement_site(rnd)
    if not site:
        return {"plan": {"status": "error", "stop": True,
                         "error": "no site.json: a place is planned against ground and "
                                  "no site has been prepared"}}
    types = rnd.flags.get("types")
    # v2, C3: **the library grows when the spec asks for a form it lacks.** A defining
    # part no committed type builds is a type authored blind, checked and adopted here,
    # before the place is planned -- a run's worth of them and no more.
    from .. import arrange as arrange_mod, capability, contracts, growth
    # **What the library can build of this programme, written down before it is used.**
    # The record is the one `growth` opens its gaps from and the one the findings read
    # their uncovered capabilities from, so "no type builds this" is one answer with one
    # reason rather than two rules that can disagree. **With the site's own facts and
    # this place's own boundaries.** The review's first finding: matching was called
    # with `names` alone, so `fits` answered "will this type stand here" with the
    # ground, the relief and the roundness of the boundary all missing -- three of the
    # constraints the matcher exists to apply. They are facts about the *pair*, not
    # about the type, and a caller that does not pass them is asking a question with
    # half its terms absent.
    facts = site_capability_facts(rnd, spec, site)
    caps = capability.match(spec, names=types, ground=facts["ground"],
                            relief=facts["relief"],
                            round_boundaries=facts["round_boundaries"],
                            intent=contracts.load(rnd, "intent"))
    contracts.save(rnd, "capabilities", caps)
    gaps = growth.type_gaps(spec, types, intent=contracts.load(rnd, "intent"))
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
    # **A place level is reused only where what it was laid out from has not moved.**
    # The same rule the site search and the preview now obey: a plan whose spec, site,
    # terrain, types or record schema has changed under it is stale, and re-entering
    # this stage on a plan drawn for a different programme is how a repair silently
    # fails to land. A round with no stamp -- every round before this one, and every
    # shipped fixture -- is reused exactly as it was.
    from .. import deps as deps_mod
    if os.path.exists(pp):
        if not deps_mod.legacy(rnd, "plan"):
            fresh, why = deps_mod.check(rnd, "plan")
            if not fresh:
                print(f"   plan: {why}; the place is laid out again", flush=True)
                os.replace(pp, rnd.rel("plan.place.stale.json"))
                for f in ("plan.json", "plots.json", "network.json",
                          "circulation.json"):
                    if os.path.exists(rnd.rel(f)):
                        os.remove(rnd.rel(f))
                for f in sorted(os.listdir(rnd.state)):
                    if f.startswith(("plan.district.", "district_")) \
                            and os.path.isfile(rnd.rel(f)):
                        os.remove(rnd.rel(f))
                deps_mod.invalidate(rnd, "plan")
    if not os.path.exists(pp):
        from .. import placesolve
        os.makedirs(rnd.state, exist_ok=True)
        # **The capability record is the authority on which type each part is built as,
        # and this is where it reaches the thing that decides.** The review's first
        # finding, in one argument: `caps` was written on the line above, `solve_place`
        # took a `caps` parameter, and the call did not pass it -- so matching chose a
        # type, the solver chose another, and `agreements` reconciled the record to
        # whatever the layout had done afterwards. A reconciliation is not a decision.
        # **The requirements reach composition.** The closure round: the relation solver
        # lays an `around` relation's districts on three sides of its object and an
        # explicit count exactly, because it is handed the intent record and not a lossy
        # spec of it. **A round may pin a layout choice its own comparison depends on.**
        # The composition round registers a section around a named gate; the side the
        # gates stand on is measured from the ground under them and can flip on a few
        # columns of ring width, which would move the registered section's subject out
        # from under it. `flags.axis_side` is that pin, and the layout records both the
        # answer it was given and the one it measured.
        _alloc = {**(rnd.flags.get("allocation") or {}),
                  **({"axis_side": rnd.flags["axis_side"]}
                     if rnd.flags.get("axis_side") else {})} or None
        place, lfails = placesolve.solve_place(spec, site, plateau, decls, voice,
                                               vol=_plan_volume(rnd, be),
                                               seed=int(rnd.flags.get("seed") or 1),
                                               caps=caps, allocation=_alloc,
                                               intent=contracts.load(rnd, "intent"), envelope_cache=os.path.join(rnd.state, "envelopes.json"))
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
    # **What each district actually holds, from the logic that will build it.** The
    # realization round's second boundary. Until this, the place level promised each
    # district a count from `spec.structures_for` -- ground divided by what a house of
    # that density costs, capped by a grid estimate -- and the compiler then laid
    # whatever the rectangle really held. The two disagreed, the validator refused the
    # district for the difference, and the run asked a model for better adjectives. Now
    # the promise *is* the arrangement: `arrange` runs the district compiler, tries the
    # ground and the fabric before it moves any number, and the file it adopted is the
    # file that gets built. The count the place carries afterwards is what the ground
    # gave, and whatever the place is short of its request surfaces as a capacity
    # finding for the scale owner rather than as a district that cannot pass.
    arranged = _arrange_districts(rnd, spec, place, site, decls, types, vol)
    if arranged.get("stop"):
        return {"plan": arranged}
    if arranged.get("adopted"):
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

    # **The resolved design, and what is still wrong with it.** The architecture round:
    # the place level says where everything is, and until now nothing said which
    # *requirement* each region answers or which stage could move it. `resolution.json`
    # is that link and `findings.json` is what it turns up -- capacity short of the
    # band, a requirement the spec dropped, a capability nothing on disk covers -- each
    # routed to the layer that can repair it. Written here, before the districts are
    # compiled, because a wall the sentence asked for and the spec omitted should be
    # found before four hundred buildings and not after them.
    caps, found = _resolve_record(rnd, spec, place, site, caps)
    # **Feasibility is repaired here, before four hundred buildings are compiled.** The
    # review's fourth finding: the only repair pass in the system ran inside
    # `stage_preview`, which needs a complete plan and a reading, so a place-level
    # feasibility failure could not reach it at all -- the city stopped, and the pass
    # that might have acted on it was three stages downstream behind a prerequisite it
    # had just failed to produce. A finding about the plan is answered at the plan.
    got = _plan_repair(rnd, be, spec, place, found)
    if got is not None:
        return got
    with contextlib.suppress(ValueError):
        deps_mod.stamp(rnd, "plan", outputs=["plan.place.json"],
                       note=(place.get("layout") or {}).get("policy") or "relations")

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
            # **The capability record's approved types reach the composer.** The closure
            # round: the palace was laid out of temple and plaza where the record had
            # approved hall and square, and the reconciliation refused the place for a
            # disagreement between two choosers.
            laid, why = placeplan.compound_axial(c, cpart, place, cdecl or decls,
                                                 spec=spec, seed=1, caps=caps)
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
        # **At capacity and short only of cover is not "spread the plots out".** This
        # clause's own message asks the district to spread its plots over the whole
        # rectangle rather than into one corner, and a district holding every house its
        # ground fits at the fabric it was given, evenly, is not in one corner: its
        # fabric does not fill it, which is a capacity fact and a finding for the layout
        # owner rather than a refusal of the district. Decided **here** and not at the
        # arrangement, because the two places compute `district_target` against
        # different arterials -- a repair re-routes the road between them and
        # `developable_columns` reads it -- so a floor measured there was 948 where the
        # refusal here was 1099. One number, taken where the refusal is made. ...and a
        # district over its ceiling that holds the least a district may hold
        # (`DISTRICT_MIN_STRUCTURES`) at the smallest lot its types admit cannot be
        # asked for fewer: the overshoot is the layout's finding (the closure round's
        # city).
        least_n = placeplan.DISTRICT_MIN_STRUCTURES
        laid_over = sum(1 for q in plots if q.get("kind", "plot") == "plot")
        if dfails and all(f.get("check") == "cover_over" for f in dfails) \
                and laid_over <= least_n and not d.get("exact"):
            d["cover_limited"] = {
                "covered": int((dfails[0] or {}).get("covered") or 0),
                "ceiling": int((dfails[0] or {}).get("ceiling") or 0),
                "realized": int(laid_over), "over": True,
                "why": (f"this district holds {laid_over} house(s), the least a district "
                        f"may hold, at the smallest lot its types admit, and they cover "
                        f"more of its ground than its density's ceiling allows; the "
                        f"count cannot be asked for fewer, so the overshoot is the "
                        f"layout's finding")}
            json.dump(place, open(rnd.rel("plan.place.json"), "w"), indent=1)
            with contextlib.suppress(ValueError):
                deps_mod.stamp(rnd, "plan", outputs=["plan.place.json"],
                               note="annotated with a district's cover ceiling")
            print(f"   district {d['name']}: at the least a district may hold and over "
                  f"its density's ceiling; carried as a layout finding rather than "
                  f"refused", flush=True)
            dfails = []
        if dfails and all(f.get("check") in ("cover", "farmland_cover") for f in dfails):
            part_here = placeplan._district_part(spec, d)
            _t2, mine2 = placeplan.types_card(types, spec.get("form"), role)
            held = 0
            with contextlib.suppress(Exception):
                held = arrange_mod.capacity(d, part_here, place, mine2 or decls,
                                            spec=spec,
                                            seed=int(rnd.flags.get("seed") or 1))
            laid_now = sum(1 for q in plots if q.get("kind", "plot") == "plot")
            # a district whose allocation has already been re-asked has had this owner's
            # action, and what came back is what the ground gives: the shortfall is a
            # finding from here on and not a second refusal
            spent = d["name"] in _reask_record(rnd)
            # **An exact count is never asked to be denser than it is.** The closure
            # round's transfer case: five houses on a ring sector of five thousand
            # columns cannot cover a dense district's ground, and asking the character's
            # author for different adjectives is asking the wrong layer. The count is
            # the sentence's; the rectangle the ring gave it is the layout's; the
            # shortfall is carried as the layout owner's finding and the density clause
            # will say what it measures.
            if d.get("exact") and laid_now >= int(d.get("structures") or 0):
                held = laid_now
                d["cover_limited"] = {
                    "covered": int((dfails[0] or {}).get("covered") or 0),
                    "floor": int((dfails[0] or {}).get("floor") or 0),
                    "realized": int(laid_now), "capacity": int(held),
                    "why": (f"this district lays {laid_now} house(s), which is every "
                            f"one its rectangle holds at the fabric it was given, and "
                            f"they cover {(dfails[0] or {}).get('covered')} of the "
                            f"{(dfails[0] or {}).get('floor')} columns its density asks "
                            f"for. The plots are not clustered; this fabric does not "
                            f"fill this ground")}
                json.dump(place, open(rnd.rel("plan.place.json"), "w"), indent=1)
                # **An annotation is not a replacement.** The closure round: writing
                # `cover_limited` onto the place file after its stamp made the plan
                # "replaced", invalidated the ground with it, and the parts stage then
                # refused to build on terraces cut for this very design.
                with contextlib.suppress(ValueError):
                    deps_mod.stamp(rnd, "plan", outputs=["plan.place.json"],
                                   note="annotated with a district's cover limit")
                print(f"   district {d['name']}: at the capacity of its own rectangle "
                      f"and short of its cover; carried as a layout finding rather than "
                      f"refused", flush=True)
                dfails = []
            elif held and laid_now < held:
                # **Short of cover and below the capacity of its own ground: ask for the
                # houses that fit.** The arrangement pass sets a district's promise
                # once, from the allocator's ask; a district refused here for cover
                # while its rectangle demonstrably holds more houses is a promise that
                # was made too small, and the layer that can change it is the one that
                # made it. The file is dropped so the arrangement lays it again against
                # the new ask -- the same construction logic, measured, not an estimate.
                d.pop("cover_limited", None)
                was_ask = int((d.get("reasked") or {}).get("to")
                              or d.get("proposed_structures") or d.get("structures") or 0)
                # **Once per district.** The same rule every recovery action in this
                # build has: the re-ask raises the promise, the arrangement lays what
                # the ground gives, and if that is still short the answer is that this
                # rectangle does not carry this fabric -- not another identical ask. A
                # place-level re-solve can drop `proposed_structures`, so the fact that
                # this district has already been re-asked is recorded on the district
                # itself, where it survives.
                asked_before = _reask_record(rnd)
                if held > was_ask and d["name"] not in asked_before \
                        and not d.get("exact"):
                    asked_before[d["name"]] = {"from": was_ask, "to": int(held),
                                               "laid": int(laid_now)}
                    json.dump(asked_before, open(rnd.rel(COVER_REASK_RECORD), "w"),
                              indent=1)
                    # the proposal stands; the re-ask is recorded beside it
                    d["reasked"] = {"from": was_ask, "to": int(held),
                                    "why": "short of its cover and the rectangle holds more"}
                    json.dump(place, open(rnd.rel("plan.place.json"), "w"), indent=1)
                    for f in (rnd.rel(f"plan.district.{d['name']}.json"),
                              rnd.rel(f"district_{d['name']}_compiled.json")):
                        if os.path.exists(f):
                            os.remove(f)
                    print(f"   district {d['name']}: short of its cover and its "
                          f"rectangle holds {held} where {was_ask} was asked for; the "
                          f"allocation is re-asked and the ground arranged again",
                          flush=True)
                    return {"plan": {
                        "status": "reenter", "level": f"district/{d['name']}",
                        "why": (f"{d['name']} was asked for {was_ask} house(s), laid "
                                f"{laid_now} and covers less of its ground than its "
                                f"density asks; the same construction logic says the "
                                f"rectangle holds {held}, so the allocation is re-asked")}}
            else:
                d.pop("cover_limited", None)
        if dfails:
            n = _record_level(rnd, f"district/{d['name']}", dfails,
                              ["district", "count", "cover", "ground_cover",
                               "farmland_cover", "type", "role", "footprint",
                               "ground", "overlap"])
            # v2, C1 and C5: a **compiled** district is not the district planner's to
            # hand back -- no model drew it -- but the **character** it was compiled
            # from is a model's, and that is who is answerable for a fabric the
            # validator refuses. So the refusal goes back to the character's author,
            # once, with what the compiler laid and what it was short of; refused twice,
            # the run stops by name. **The layout owner gets the finding before the
            # character's author does.** A compiled district short of its own cover is
            # first of all a question about the promise the allocator made, and only
            # after that a question about the fabric it was to be filled with. Asking a
            # model for a denser character in a 30-column strip is asking the wrong
            # layer, and it is what this run did twice before stopping.
            laid = sum(1 for q in plots if q.get("kind", "plot") == "plot")
            realloc = _district_capacity_repair(rnd, spec, place, d, laid, dfails,
                                                site=site, decls=decls)
            if realloc is not None and not realloc["refused"]:
                _apply_reallocation(rnd, place, realloc)
                print(f"   layout: {d['name']} "
                      f"{realloc.get('action', 'allocation')} "
                      f"{realloc['from']} -> {realloc['to']}; {realloc['why']}",
                      flush=True)
                return {"plan": {"status": "reenter",
                                 "level": f"district/{d['name']}",
                                 "repair": realloc, "failures": dfails,
                                 "why": (f"the layout owner changed "
                                         f"{realloc.get('action', 'allocation')} on "
                                         f"{d['name']}: {realloc['why'][:160]}")}}
            if realloc is not None and realloc["refused"]:
                rec_r = _reallocations(rnd)
                rec_r["refused"].append({"district": d["name"],
                                         "why": realloc["why"]})
                json.dump(rec_r, open(rnd.rel(REALLOCATION_RECORD), "w"), indent=1)
            part = placeplan._district_part(spec, d)
            if spec_mod.character(part) is not None:
                if n >= 2:
                    # **What the owner tried, beside what is still wrong.** A stop that
                    # quotes only the validator reads as "the compiler failed", and what
                    # actually happened is that every action the layout owner has was
                    # tried on this district and each was bounded -- which is the
                    # limiting constraint and is the thing worth reporting.
                    return {"plan": {"status": "error", "stop": True,
                                     "level": f"district/{d['name']}", "attempt": n,
                                     "failures": dfails, "compiled": True,
                                     "owner_tried": _owner_attempts(rnd, d["name"]),
                                     "error": f"the compiled district {d['name']} fails "
                                              f"its validator twice: " + "; ".join(
                                                  f"{f.get('part')}: {f['why']}"
                                                  for f in dfails[:6])
                                              + _owner_says(rnd, d["name"])}}
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
    # **The resolution is written again, over the plan that exists.** The review's third
    # finding: `resolution.json` summarised a place level whose districts had not been
    # compiled -- lot counts were promises, boundaries were null, and the checks that
    # read it were reading intentions. Now the record is made once before the districts
    # compile, where it is what catches an omitted wall cheaply, and **again here** over
    # the assembled tree, where `lots` is what the ground gave and a frontage check has
    # actual doors to look at.
    caps2 = contracts_mod.load(rnd, "capabilities")
    caps2, found2 = _resolve_record(rnd, spec, place, site, caps2, plan=plan,
                                    realized=_realized_lots(
                                        plan, [d["name"] for d in
                                               (place.get("districts") or [])]))
    again = _plan_repair(rnd, be, spec, place, found2)
    if again is not None:
        return again
    # **A plan with an open feasibility finding is not a planned plan.** The review's
    # fifth finding: `_plan_repair` returns None both when it has fixed everything and
    # when its budget is spent or no supported action changes anything, and the caller
    # wrote the registry and reported `planned` either way. The city's two district
    # shortfalls sat in `findings.json` as `blocks: feasibility` while the round called
    # itself planned, because both were `severity: warning` -- but severity is how
    # loudly a finding is said and `blocks` is what it stops. Exhaustion is an outcome
    # and it is this one: the plan stands, the record says what is unsolved and who owns
    # it, and nothing downstream builds it.
    stuck = blocking(found2)
    if stuck:
        spent = _repair_passes(rnd)
        return {"plan": {
            "status": "blocked", "stop": True, "level": "assembled",
            "blocks": "feasibility", "repair_passes": spent,
            "findings": [{k: f.get(k) for k in ("id", "owner", "says", "severity")}
                         for f in stuck],
            "error": (f"the assembled plan carries {len(stuck)} open feasibility "
                      f"finding(s) after {spent} repair pass(es), the registered cap "
                      f"being {PLAN_REPAIR_BUDGET} per candidate and "
                      f"{PLAN_REPAIR_TOTAL} per run: "
                      + "; ".join(f"{f['owner']}: {f['says'][:100]}"
                                  for f in stuck[:4]))}}
    return _write_registry(rnd, plan, parts, levels=plan["levels"], voice=voice)


def _repair_passes(rnd) -> int:
    p = rnd.rel(PLAN_REPAIR_RECORD)
    return len((json.load(open(p)) if os.path.exists(p) else {}).get("passes") or [])


def _realized_lots(plan: dict, districts) -> dict:
    """`{district name: plots the compiler actually laid in it}`.

        Off the assembled tree and off nothing else. This is the number a promise is
        reconciled against, and the whole reason it exists is that the promise and the
        realization were being computed by two different rules that never met.

        A leaf's `in` names the **quarter** it is in -- `homes_shore_4_row_0` -- and a
        district is a level above that. Found by running it: taking the innermost name
        attributed every lot to a quarter, every district read as zero, and the reconciler
        raised a shortfall against every region in the place. A count that is wrong in the
        direction of "nothing was built" is worse than no count, because it looks exactly
        like the defect it exists to find.
        
    """
    from .. import pipeline as _pipeline
    names = sorted({str(d) for d in districts}, key=len, reverse=True)
    out = {n: 0 for n in names}
    for leaf in _pipeline.plan_parts(plan):
        if leaf.get("kind", "plot") != "plot":
            continue
        where = [w for w in (leaf.get("in") or []) if w]
        hit = None
        for w in reversed(where):
            hit = next((n for n in names if w == n or w.startswith(n + "_")), None)
            if hit:
                break
        if hit:
            out[hit] += 1
    return out


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
    from .. import deps as _deps_a
    with contextlib.suppress(ValueError):
        # `plan.json` only: the plot registry is the file construction annotates with
        # the floor each part was sited at, and a registry that grows a fact is not a
        # replaced output
        _deps_a.stamp(rnd, "assembled", outputs=["plan.json"],
                      note=f"{len(parts)} leaves; plots.json beside it")
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
    from .. import deps as _deps_a
    with contextlib.suppress(ValueError):
        # `plan.json` only: the plot registry is the file construction annotates with
        # the floor each part was sited at, and a registry that grows a fact is not a
        # replaced output
        _deps_a.stamp(rnd, "assembled", outputs=["plan.json"],
                      note=f"{len(parts)} leaves; plots.json beside it")
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
    # **Keyed on the geometry the road is routed from, and nothing else.** The closure
    # round, found by replaying the proof: the key hashed every field of every district,
    # the arrangement then wrote the adopted count, the pool and the re-ask onto the
    # districts, and the next invocation called the place "changed", routed a new road
    # through a district compiled against the old one, and refused that district for a
    # plot standing on the arterial. A road is routed from where things stand.
    geometry = {
        "parts": [{k: v for k, v in (q or {}).items()
                   if k in ("name", "kind", "type", "x0", "z0", "x1", "z1", "path", "at",
                            "level", "defines", "half", "rect")}
                  for q in (place.get("parts") or [])],
        "districts": [{k: v for k, v in (d or {}).items()
                       if k in ("name", "x0", "z0", "x1", "z1", "level", "ring", "defines")}
                      for d in (place.get("districts") or [])],
        "compounds": [{k: v for k, v in (c or {}).items()
                       if k in ("name", "x0", "z0", "x1", "z1", "level")}
                      for c in (place.get("compounds") or [])],
        "layout": {k: (place.get("layout") or {}).get(k)
                   for k in ("rings", "wall", "centre", "anchor_path")}}
    key = hashlib.sha256(json.dumps(
        {**geometry,
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
        # ...and the districts compiled against the old road go with it: a district laid
        # beside a road that has moved is laid beside nothing
        dropped = 0
        for f in sorted(os.listdir(rnd.state)):
            if f.startswith(("plan.district.", "district_")) and os.path.isfile(rnd.rel(f)):
                os.remove(rnd.rel(f))
                dropped += 1
        _drop_assembled(rnd)
        print(f"   arterials: the place plan's geometry changed, so the road is routed "
              f"again (the one before it is kept beside it"
              + (f"; {dropped} district file(s) compiled against it are dropped" if dropped
                 else "") + ")", flush=True)
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


def _drop_assembled(rnd) -> None:
    """Retire the assembled plan and the lanes routed for it.

        The closure round, found by replaying the proof: a district was laid again and the
        assembled tree was then validated against `network.json` -- the lanes routed to the
        plots of the arrangement before it -- and refused for doorsteps the circulation pass
        had reserved for houses that no longer stood there. A plan and its lanes belong to
        the districts they were assembled from.
        
    """
    for f in ("plan.json", "plots.json", "network.json", "circulation.json"):
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))


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
