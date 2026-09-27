"""**How a place is planned**: repeated fabric, individual composition, or both.

Chosen before the design job, from the interpreted place spec and what the library can
express, and recorded with its reasons in `planning.json`:

  - **fabric** -- rings and wards composed in a grain (`citydesign.GRAINS`): lots packed
    along streets by rule, one dwelling form per grain, landmarks and a monument at named
    places. Right where the spec gives a district its `character` (the rhythm of lots it
    is made of) and every other part is one the fabric's landmark vocabulary composes.
  - **composed** -- the design places each building, outdoor room and path itself, each
    with its use, form and relationships (`citydesign.COMPOSITION_DOC`). Right where the
    spec plans its parts individually: a group with no `character` is planned building
    by building, and a building of its own use (a smithy, a hall) is a decision about
    that building.
  - **mixed** -- fabric, with the individually planned parts composed in a ward or ring
    of their own (grain `composed`): a city with a composed quarter.

The choice reads the spec's own statement of how each part is made, not words in the
sentence and not a building count: a small planned settlement whose spec gives its
cottages a character is fabric; a large place with no character anywhere is composed.
What the library cannot express for a part is listed as unsupported, never swapped.

Nothing here writes a design. `stage_planning` (`pipeline/stages_design.py`) records the
choice, keeps it while its inputs are unchanged, and when they change and the choice
changes with them, retires the design made under the old one.
"""
from __future__ import annotations

import hashlib
import json
import os

STRATEGIES = ("fabric", "composed", "mixed")

#: spec families whose parts the fabric composes as a whole (a landmark form, the
#: monument or a compound), by the landmark or composer that holds them
FABRIC_HOLDS = {"square": "a landmark (square, market, plaza, garden, grove, pool, park)",
                "palace": "the monument or a compound", "monument": "the monument",
                "temple": "a temple landmark", "wall": "a ring wall",
                "gate": "a gate on a ring wall", "district": "a grain",
                "quarter": "a grain"}

#: what each spec family asks of a composed building's form, by declared `FUNCTION`
#: (`types/*.py`); a family with no entry is not a building of its own
FAMILY_USES = {"house": ("dwelling",), "workshop": ("work", "trade"),
               "hall": ("civic",), "temple": ("worship",), "keep": ("defensive",),
               "tower": ("defensive",), "square": ("market",)}


def _forms_by_function(root: str) -> dict:
    """{function: [plot forms declaring it]} off the type files, and the areas by name."""
    from .pipeline import load_type
    tdir = os.path.join(root, "types")
    by: dict = {}
    areas = []
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        try:
            d = load_type(os.path.join(tdir, f))
        except Exception:                          # noqa: BLE001 -- not a usable form
            continue
        if d.get("kind") == "area":
            areas.append(f[:-3])
        if d.get("kind") == "plot" and d.get("function"):
            by.setdefault(d["function"], []).append(f[:-3])
    return {"functions": by, "areas": areas}


def capability(root: str) -> dict:
    """What the library can express, as the chooser reads it: the fabric's grains and
    landmark forms, and the composed vocabulary (forms by function, open-ground kinds)."""
    from . import citydesign as CD
    got = _forms_by_function(root)
    return {"grains": sorted(k for k in CD.GRAINS if k != "composed"),
            "landmarks": list(CD.LANDMARK_FORMS),
            "functions": got["functions"], "areas": got["areas"],
            "spaces": sorted(CD.SPACE_KINDS)}


def _part_view(p: dict) -> dict:
    """The fields of a spec part the choice depends on, and nothing else: a changed note
    or voice does not re-plan a place."""
    return {"name": p.get("name"), "kind": p.get("kind"), "family": p.get("family"),
            "relation": p.get("relation"), "of": p.get("of"),
            "structures": int(p.get("structures") or 0), "count": int(p.get("count") or 1),
            "character": bool(p.get("character")), "role": p.get("role")}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def classify(part: dict, cap: dict) -> dict:
    """How one spec part asks to be planned: `fabric`, `individual` or `either`, why, and
    what the library lacks for it (`unsupported`)."""
    v = _part_view(part)
    fam, kind = v["family"], v["kind"]
    row = {"part": v["name"], "kind": kind, "family": fam}
    if kind in ("edge", "point"):
        return {**row, "planning": "either",
                "why": f"a {fam} is laid on its line by either strategy"}
    if kind == "group":
        if fam in ("palace", "monument"):
            return {**row, "planning": "either",
                    "why": f"a {fam} is a composition of its own (the monument or a "
                           f"compound), whatever holds the rest"}
        if v["character"]:
            return {**row, "planning": "fabric",
                    "why": "the spec gives its character: a rhythm of lots, composed "
                           "by a grain"}
        if not v["structures"]:
            return {**row, "planning": "either",
                    "why": "a group with no buildings of its own"}
        return {**row, "planning": "individual",
                "why": "a group of buildings with no character: planned building by "
                       "building (the spec's own rule for a district without one)"}
    if kind == "area":
        return {**row, "planning": "either",
                "why": "open ground: a landmark of the fabric, or a space of a "
                       "composition"}
    # a building of its own
    uses = FAMILY_USES.get(fam)
    forms = sorted({f for u in (uses or ()) for f in cap["functions"].get(u, [])})
    if fam in ("temple", "palace", "monument") and not forms:
        return {**row, "planning": "either",
                "why": f"a {fam} the fabric composes as {FABRIC_HOLDS.get(fam)}"}
    out = {**row, "planning": "individual",
           "why": f"a building of its own use ({', '.join(uses or [fam])}): where it "
                  f"stands and what it faces are decisions about it",
           "forms": forms}
    if uses and not forms:
        out["unsupported"] = (f"no form in the library declares the function "
                              f"{' or '.join(uses)}")
    return out


def choose(spec: dict, root: str) -> dict:
    """The planning record for `spec` (read by `spec.read_spec`)."""
    cap = capability(root)
    parts = [classify(p, cap) for p in spec.get("defining_parts") or []]
    fab = [r["part"] for r in parts if r["planning"] == "fabric"]
    ind = [r["part"] for r in parts if r["planning"] == "individual"]
    if fab and ind:
        strategy = "mixed"
        why = (f"{', '.join(fab)} {'is' if len(fab) == 1 else 'are'} repeated fabric and "
               f"{', '.join(ind)} {'is' if len(ind) == 1 else 'are'} planned "
               f"individually: fabric rings with the individual parts composed in a "
               f"ward or ring of their own")
    elif ind:
        strategy = "composed"
        why = (f"{', '.join(ind)} {'is' if len(ind) == 1 else 'are'} planned building "
               f"by building and no part is repeated fabric: the design places each "
               f"building, outdoor room and path")
    else:
        strategy = "fabric"
        why = ((f"{', '.join(fab)} {'is' if len(fab) == 1 else 'are'} repeated fabric "
                if fab else "no part asks for individual planning ")
               + "and every other part is one the fabric composes: rings in grains")
    inputs = {"spec": _digest({"kind": spec.get("kind"),
                               "parts": [_part_view(p) for p in
                                         spec.get("defining_parts") or []]}),
              "capability": _digest(cap)}
    return {"record": "planning", "strategy": strategy, "why": why, "parts": parts,
            "unsupported": [{"part": r["part"], "why": r["unsupported"]}
                            for r in parts if r.get("unsupported")],
            "inputs": inputs}


def digest(rec: dict) -> str:
    """The identity of a choice: the strategy and which parts are fabric and which
    individual -- not its prose, and not the parts either strategy lays alike."""
    return _digest({"strategy": rec.get("strategy"),
                    "parts": sorted((r["part"], r["planning"]) for r in rec.get("parts") or []
                                    if r["planning"] != "either")})


def expects(strategy: str, design: dict) -> str | None:
    """Why `design` does not follow `strategy`, or None."""
    rings = design.get("rings") or []
    grains = [r.get("grain") for r in rings] + [w.get("grain") for r in rings
                                                for w in r.get("wards") or []]
    comp = design.get("composition")
    if strategy == "composed":
        if "composed" not in grains or not comp:
            return ("the planning strategy is `composed`: the place's land is a ring (or "
                    "rings) of grain `composed` and the design carries a `composition`")
        other = [g for g in grains if g not in ("composed", "open", None)]
        if other:
            return (f"the planning strategy is `composed`, and grains {sorted(set(other))} "
                    f"are repeated fabric: compose those parts, or the spec must say "
                    f"they are fabric (a district with a character)")
    elif strategy == "fabric":
        if "composed" in grains or comp:
            return ("the planning strategy is `fabric`: no part of the spec is planned "
                    "individually, so the design has no `composed` grain or composition")
    elif strategy == "mixed":
        if "composed" not in grains or not comp:
            return ("the planning strategy is `mixed`: the individually planned parts are "
                    "composed in a ring or ward of grain `composed` with a `composition`")
        if not [g for g in grains if g not in ("composed", "open", None)]:
            return ("the planning strategy is `mixed`: the repeated parts are rings or "
                    "wards in a fabric grain")
    return None
