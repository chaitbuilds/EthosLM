"""**One deterministic material pass over owned surfaces.** The expression round's
bounded experiment, and nothing more than that until the comparison says otherwise.

    recipe_for(voice)                   the voice's `variants` block, validated
    apply(built, surfaces, recipe,      -> (finished volume, record)
          settings, seed, mode)
    key_for(...)                        the identity of a finished world

Three modes on identical geometry, cameras and lighting:

    none         the structural world, untouched -- the control
    random       weighted variants per role, scattered uniformly, at **the same rate
                 per family the contextual pass produced in the same architectural
                 context** -- one voice's courtyard house against that same voice's
                 courtyard house, never a town's two voices pooled -- so the two arms
                 differ in where and not in how much or of what
    contextual   coherent patches from a low-frequency field over world coordinates
                 (one field for the whole place, so a patch crosses a party wall),
                 plus condition masks the surface record carries: damp base courses
                 where a wall meets the ground or stands near water, sheltered faces
                 under an overhang kept cleaner, runoff under an opening, exposure
                 by face toward the weather. `settings.condition` scales the masks:
                 `maintained` keeps the damp and weathered shares small, `weathered`
                 lets them run.

Rules that do not move: only cells the surface record owns under an editable role are
touched, **after that record has been reconciled against the assembled world** --
`surfaces.reconcile`, which drops what a later part overwrote, what any part
protected, what no longer stands and what nothing can see, and recomputes exposure
from the finished volume rather than from the neighbourhood one part saw;
protected cells (treads, doors, glass, fittings, lights, the ground) never;
**a cell the type declared part of a figure** never (`surfaces.FLAGS["figure"]`, out of
`Primitives.figure`: the market's chequer, a court's laid paving, a wall's string
course -- the three patterns the design round's pass replaced with camouflage because
the record said who owned a cell and not what the cell was for);
the substitute is **the same shape of a compatible family** -- a stair stays a stair
with its state, a slab a slab, a wall a wall -- or the block is left; where the family
has no such shape nothing is invented. The decision for a cell is a hash of `(seed,
x, y, z, role)` and the field is a hash of lattice points, so the answer is the same
in any order, on any worker count, and a replay from the same structural base is the
same bytes. The pass reads the base's roles and the recipe, never a previously
finished volume, so finishing an already-finished world from the same record is the
same world: nothing accumulates.

Grounded in Dorsey, Pedersen and Hanrahan's account of weathering as flow and
material interaction (sources/INDEX.md); the masks here are the cheap design
hypothesis the plan names, not that simulation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from . import prims, surfaces as _surfaces
from .observe import Volume
from .prims import EDITABLE_ROLES, MATERIALS

MODES = ("none", "random", "contextual")
CONDITIONS = ("maintained", "weathered")
#: What a variant may be conditioned on. Registered before any recipe used them.
WHEN = ("any", "patch", "damp", "exposed", "sheltered", "runoff")
#: How the two settings scale each condition's share.
SETTING_SCALE = {"maintained": {"any": 1.0, "patch": 1.0, "damp": 0.35, "exposed": 0.25,
                                "sheltered": 1.0, "runoff": 0.3},
                 "weathered": {"any": 1.0, "patch": 1.0, "damp": 1.0, "exposed": 1.0,
                               "sheltered": 1.0, "runoff": 1.0}}
#: The lattice pitch of the patch field, in blocks: a patch is a few blocks across.
PATCH_SCALE = 5
#: Which side the weather comes from, by default; a setting may say otherwise.
WEATHER_FROM = "west"
_FACE_BIT = {"north": _surfaces.FLAGS["open_north"], "south": _surfaces.FLAGS["open_south"],
             "east": _surfaces.FLAGS["open_east"], "west": _surfaces.FLAGS["open_west"]}


class RecipeError(ValueError):
    """A recipe that will not load, with the role and the family named."""


# ------------------------------------------------------------------ recipes

def recipe_for(voice) -> dict:
    """`{role: [{"family", "weight", "when"}, ...]}` off a voice name or dict.

        A family must be one `prims.MATERIALS` knows or a bare block that exists; a weight
        is a share in (0, 1]; `when` is one of `WHEN`. A voice with no `variants` has an
        empty recipe, and the pass then changes nothing -- which is a legitimate voice.
        
    """
    from . import voices as _voices
    doc = _voices.load(voice) if isinstance(voice, str) else dict(voice)
    raw = doc.get("variants") or {}
    if not isinstance(raw, dict):
        raise RecipeError("`variants` is a mapping of role to a list of variants")
    out: dict = {}
    for role, rows in raw.items():
        if role not in EDITABLE_ROLES:
            raise RecipeError(f"variants for {role!r}: a material pass edits only "
                              f"{', '.join(EDITABLE_ROLES)}")
        got = []
        for i, r in enumerate(rows or []):
            fam = str((r or {}).get("family") or "").strip()
            if not fam or (prims.family(fam) is None and not prims._has_block("minecraft:" + fam)):
                raise RecipeError(f"variants for {role!r}[{i}]: {fam!r} is neither a "
                                  f"material family nor a block")
            w = float((r or {}).get("weight") or 0)
            if not (0 < w <= 1):
                raise RecipeError(f"variants for {role!r}[{i}]: weight {w} is not in (0, 1]")
            when = str((r or {}).get("when") or "any")
            if when not in WHEN:
                raise RecipeError(f"variants for {role!r}[{i}]: `when` is one of "
                                  f"{', '.join(WHEN)}, not {when!r}")
            got.append({"family": fam, "weight": w, "when": when})
        if got:
            out[role] = got
    return out


def recipe_digest(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def key_for(built_digest: str | None, recipe: dict, settings: dict, seed: int,
            mode: str) -> str:
    """The identity of one finished world: structural base, recipe, settings, seed,
    mode. Two finished worlds with the same key are the same bytes."""
    doc = {"built": built_digest, "recipe": recipe_digest(recipe),
           "settings": {k: settings[k] for k in sorted(settings)}, "seed": int(seed),
           "mode": mode}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the fields

def _u(seed: int, *parts) -> float:
    """A uniform in [0, 1) from a hash of the parts. The only randomness here."""
    h = hashlib.sha256(("|".join(str(p) for p in (seed,) + parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _lattice(seed: int, i: int, j: int, k: int, tag: str) -> float:
    return _u(seed, "lattice", tag, i, j, k)


def field(seed: int, x: int, y: int, z: int, scale: int = PATCH_SCALE,
          tag: str = "patch") -> float:
    """A smooth value in [0, 1] over world coordinates: value noise on a lattice of
    `scale` blocks, interpolated. One field for the whole place."""
    fx, fy, fz = x / float(scale), y / float(scale * 2), z / float(scale)
    i, j, k = math.floor(fx), math.floor(fy), math.floor(fz)
    tx, ty, tz = fx - i, fy - j, fz - k
    sx, sy, sz = (t * t * (3 - 2 * t) for t in (tx, ty, tz))

    def c(di, dj, dk):
        return _lattice(seed, i + di, j + dj, k + dk, tag)
    x00 = c(0, 0, 0) + (c(1, 0, 0) - c(0, 0, 0)) * sx
    x10 = c(0, 1, 0) + (c(1, 1, 0) - c(0, 1, 0)) * sx
    x01 = c(0, 0, 1) + (c(1, 0, 1) - c(0, 0, 1)) * sx
    x11 = c(0, 1, 1) + (c(1, 1, 1) - c(0, 1, 1)) * sx
    y0 = x00 + (x10 - x00) * sy
    y1 = x01 + (x11 - x01) * sy
    return y0 + (y1 - y0) * sz


_THRESHOLDS: dict = {}


def patch_threshold(seed: int, tag: str, share: float) -> float:
    """The field value above which a patch covers about `share` of a surface.

    The field is value noise and is not uniform -- interpolation pulls it toward the
    middle -- so a share is a **quantile of the field itself**, read off a fixed
    sample of it (a 48x48x3 grid of world coordinates), never a guess. Deterministic
    in the seed and the tag, cached."""
    key = (int(seed), str(tag), round(float(share), 4))
    got = _THRESHOLDS.get(key)
    if got is None:
        vals = sorted(field(seed, i * 3, 60 + j * 4, k * 3, tag=tag)
                      for i in range(48) for j in range(3) for k in range(48))
        q = min(len(vals) - 1, max(0, int((1.0 - float(share)) * len(vals))))
        got = _THRESHOLDS[key] = vals[q]
    return got


# ------------------------------------------------------------------ shapes

_SHAPE_SUFFIXES = (("_stairs", "stairs"), ("_slab", "slab"), ("_wall", "wall"),
                   ("_fence_gate", "gate"), ("_fence", "fence"), ("_trapdoor", "trapdoor"),
                   ("_door", "door"), ("_button", "button"), ("_pillar", "post"),
                   ("_log", "post"), ("_wood", "bare"))


def shape_of(state: str) -> tuple:
    """(kind, state suffix) of a block: `cobblestone_stairs[facing=north]` is
    `("stairs", "[facing=north]")`; a family's cube is `("full", "")`."""
    name, _, rest = str(state).partition("[")
    name = name.split(":")[-1]
    suffix = f"[{rest}" if rest else ""
    for fam, (full, stairs, slab) in MATERIALS.items():
        if name == full:
            return "full", suffix
        if name == stairs:
            return "stairs", suffix
        if name == slab:
            return "slab", suffix
    for suf, kind in _SHAPE_SUFFIXES:
        if name.endswith(suf):
            return kind, suffix
    return "full", suffix


def substitute(state: str, family: str) -> str | None:
    """The same shape of `family`, with the state kept, or None where the family has
    no such shape. Never a different shape: the physics of the cell are its shape's."""
    kind, suffix = shape_of(state)
    try:
        block = prims.shape(family, kind)
    except ValueError:
        return None
    if block.split("[")[0] == str(state).split("[")[0]:
        return None
    return f"{block}{suffix}"


# ------------------------------------------------------------------ the pass

def _conditions(flags: int, floor_y, y: int, weather_bit: int) -> dict:
    F = _surfaces.FLAGS
    sheltered = bool(flags & F["sheltered"])
    exposed_face = bool(flags & (F["open_north"] | F["open_south"] | F["open_east"]
                                 | F["open_west"]))
    # **Damp needs a face for the water to come out of.** The design round's cause 2:
    # `damp` was `ground_contact | water_near | base_course` alone, and a floor is in
    # ground contact by definition -- every cell of it, everywhere -- so a `footing` or
    # `floor` variant with `when: damp` repainted a whole paved court at its full
    # weight. That is what turned the city's market floor and the palace courts grey
    # (`out/des-material/comparison.json`, criterion 5). Damp is what wicks out of the
    # ground into a **vertical** surface somebody looks at, so it wants one of the four
    # side faces open as well. Measured on `out/des-city/surfaces.json`: the share of
    # cells this calls damp falls from 35.2% to 4.1% on `footing` and 9.3% to 0.7% on
    # `floor` -- the inside of the paving -- and from 11.26% to 10.76% on `wall` and not
    # at all on `frame`, which is the base course the condition was written for.
    damp = exposed_face and bool(flags & (F["ground_contact"] | F["water_near"]
                                          | F["base_course"]))
    weather = bool(flags & weather_bit)
    return {"any": 1.0,
            "patch": 1.0,
            "damp": 1.0 if damp else 0.0,
            "exposed": (1.0 if weather else 0.4) if (exposed_face and not sheltered) else 0.0,
            "sheltered": 1.0 if sheltered else 0.0,
            "runoff": 1.0 if flags & F["under_opening"] else 0.0}


def _decide_contextual(role: str, variants: list, cell: list, seed: int, scale: dict,
                       floor_y, weather_bit: int) -> str | None:
    x, y, z, _si, flags = cell
    cond = _conditions(int(flags), floor_y, y, weather_bit)
    u = _u(seed, role, x, y, z)
    acc = 0.0
    # conditions first, in the order a mason would see them; the patch field last
    for when in ("damp", "runoff", "exposed", "sheltered", "any"):
        for v in variants:
            if v["when"] != when:
                continue
            p = v["weight"] * cond[when] * scale.get(when, 1.0)
            if p <= 0:
                continue
            acc += p
            if u < acc:
                return v["family"]
    for v in variants:
        if v["when"] != "patch":
            continue
        # a patch is where the field is high; the weight is the share of ground it
        # covers, so the threshold is 1 - weight on a roughly uniform field
        tag = f"patch:{role}:{v['family']}"
        f = field(seed, x, y, z, tag=tag)
        if f > patch_threshold(seed, tag, v["weight"] * scale.get("patch", 1.0)):
            return v["family"]
    return None


def _decide_random(role: str, rates: dict, cell: list, seed: int) -> str | None:
    x, y, z, _si, _flags = cell
    u = _u(seed, role, x, y, z)
    acc = 0.0
    for fam, p in sorted(rates.items()):
        acc += p
        if u < acc:
            return fam
    return None


def context_of(prec: dict) -> str:
    """The architectural context a part's surfaces belong to: its voice and its type.

        **The unit the random arm has to be matched on.** The expression round pooled
        variant rates by role across the whole place, so a town of two voices scattered one
        voice's stone over the other's walls and a wall type was matched against a house
        -- which is a different palette at a different rate, not a control for *where* the
        variants went. Rates are collected per context and the random arm draws from the
        context's own; a context whose voice has no recipe for a role gets nothing in
        either arm.
        
    """
    return f"{prec.get('voice_name') or '-'}|{prec.get('type') or prec.get('kind') or '-'}"


def _recipe_of(recipe: dict, prec: dict) -> dict:
    """The recipe for one part: a flat recipe, or the part's own voice's out of
    `{"by_voice": {voice name: recipe}}` -- a town of three voices has three."""
    if isinstance(recipe, dict) and "by_voice" in recipe:
        return (recipe.get("by_voice") or {}).get(prec.get("voice_name")) or {}
    return recipe or {}


def _recipe_empty(recipe: dict) -> bool:
    if isinstance(recipe, dict) and "by_voice" in recipe:
        return not any((recipe.get("by_voice") or {}).values())
    return not recipe


def plan(surfaces_doc: dict, recipe: dict, settings: dict | None, seed: int,
         mode: str, *, honour_figures: bool = True) -> dict:
    """The substitutions, as data: `{(x, y, z): (state, family)}` and the rates.

        `apply` writes these into a volume; the plan is what the comparison compares and
        what the tests read. Cells are visited in sorted order and each decision depends
        on nothing but the cell, the record and the recipe.

        `honour_figures=False` is a **diagnostic** and nothing else: it plans as if the
        types had declared no figure, so a comparison can show what the guard is holding
        back on the same cameras. It is never how a world is finished; every caller that
        finishes a world leaves it alone.
        
    """
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}, not {mode!r}")
    settings = dict(settings or {})
    condition = str(settings.get("condition") or "maintained")
    if condition not in CONDITIONS:
        raise ValueError(f"condition is one of {', '.join(CONDITIONS)}, not {condition!r}")
    scale = SETTING_SCALE[condition]
    weather_bit = _FACE_BIT.get(str(settings.get("weather_from") or WEATHER_FROM),
                                _FACE_BIT[WEATHER_FROM])
    out: dict = {}
    counts: dict = {}
    totals: dict = {}
    parts = (surfaces_doc or {}).get("parts") or []
    if mode == "none" or _recipe_empty(recipe):
        return {"mode": mode, "cells": out, "counts": counts, "totals": totals,
                "rates": {}, "condition": condition, "figure_refused": 0}
    fig_bit = _surfaces.FLAGS["figure"] if honour_figures else 0
    figure_refused = 0
    # contextual decisions, or the rates the random arm matches
    ctx: dict = {}
    ctotals: dict = {}          # (context, role) -> cells the recipe could reach
    ccounts: dict = {}          # (context, role) -> {family: cells it took}
    for prec in parts:
        states = prec.get("states") or []
        floor_y = prec.get("floor_y")
        con = context_of(prec)
        for role, cells in sorted((prec.get("cells") or {}).items()):
            variants = _recipe_of(recipe, prec).get(role)
            if not variants:
                continue
            # **A declared figure is not noise to be overwritten.** The type that laid
            # the market's chequer, the court's laid paving or the wall's string course
            # said so at the write (`Primitives.figure`, `surfaces.FLAGS["figure"]`),
            # and a pass that had no way to know replaced all three with camouflage in
            # the design round. Refused here rather than in the decision, and taken out
            # of the denominator too, so the random arm is still matched cell for cell
            # on the cells a recipe may actually reach.
            figs = [c for c in cells if int(c[4]) & fig_bit]
            if figs:
                figure_refused += len(figs)
                cells = [c for c in cells if not (int(c[4]) & fig_bit)]
                if not cells:
                    continue
            totals[role] = totals.get(role, 0) + len(cells)
            ctotals[(con, role)] = ctotals.get((con, role), 0) + len(cells)
            for cell in sorted(cells):
                fam = _decide_contextual(role, variants, cell, seed, scale, floor_y,
                                         weather_bit)
                if fam is None:
                    continue
                st = states[cell[3]]
                new = substitute(st, fam)
                if new is None:
                    continue
                ctx[(cell[0], cell[1], cell[2])] = (new, fam, role, st)
                counts.setdefault(role, {})
                counts[role][fam] = counts[role].get(fam, 0) + 1
                ccounts.setdefault((con, role), {})
                ccounts[(con, role)][fam] = ccounts[(con, role)].get(fam, 0) + 1
    rates = {role: {fam: n / float(totals[role]) for fam, n in fams.items()}
             for role, fams in counts.items() if totals.get(role)}
    crates = {k: {fam: n / float(ctotals[k]) for fam, n in fams.items()}
              for k, fams in ccounts.items() if ctotals.get(k)}
    bucket_rates = {f"{con}|{role}": v for (con, role), v in sorted(crates.items())}
    if mode == "contextual":
        return {"mode": mode, "cells": ctx, "counts": counts, "totals": totals,
                "rates": rates, "bucket_rates": bucket_rates, "condition": condition,
                "figure_refused": figure_refused}
    # random: the same families at the same rate **in the same context**, scattered
    rnd: dict = {}
    rcounts: dict = {}
    rbucket: dict = {}
    for prec in parts:
        states = prec.get("states") or []
        con = context_of(prec)
        for role, cells in sorted((prec.get("cells") or {}).items()):
            rate = crates.get((con, role))
            if not rate:
                continue
            # the control arm refuses a figure for the same reason, or it would be
            # matched on a different pool of cells than the arm it is a control for
            for cell in sorted(cells):
                if int(cell[4]) & fig_bit:
                    continue
                fam = _decide_random(role, rate, cell, seed + 7919)
                if fam is None:
                    continue
                new = substitute(states[cell[3]], fam)
                if new is None:
                    continue
                rnd[(cell[0], cell[1], cell[2])] = (new, fam, role, states[cell[3]])
                rcounts.setdefault(role, {})
                rcounts[role][fam] = rcounts[role].get(fam, 0) + 1
                rbucket.setdefault((con, role), {})
                rbucket[(con, role)][fam] = rbucket[(con, role)].get(fam, 0) + 1
    return {"mode": mode, "cells": rnd, "counts": rcounts, "totals": totals,
            "rates": rates, "bucket_rates": bucket_rates, "matched_rates": {
                role: {fam: n / float(totals[role]) for fam, n in fams.items()}
                for role, fams in rcounts.items() if totals.get(role)},
            "matched_bucket_rates": {
                f"{con}|{role}": {fam: n / float(ctotals[(con, role)])
                                  for fam, n in fams.items()}
                for (con, role), fams in sorted(rbucket.items()) if ctotals.get((con, role))},
            "condition": condition, "figure_refused": figure_refused}


def apply(built: Volume, surfaces_doc: dict, recipe: dict, settings: dict | None = None,
          seed: int = 1, mode: str = "contextual", *,
          built_digest: str | None = None, reconcile: bool = True,
          honour_figures: bool = True) -> tuple:
    """(finished volume, record). The input volume is not modified.

        The finished volume is a copy of `built` with the plan's substitutions written in.
        Every substituted cell is one the record owns under an editable role; the block
        keeps its shape and state. `record.key` is `key_for(...)` of the inputs.

        The record is **reconciled against `built` first** (`surfaces.reconcile`) unless it
        already has been: a per-part record is made before its neighbours exist, so a cell
        a later part overwrote belongs to the later part, a cell any part protected is
        protected, a cell whose block no longer stands is gone, and a cell with no air on
        any face is not worth editing. Exposure is recomputed there from the assembled
        world, so the conditions the contextual arm reads are the assembled world's.
        
    """
    if reconcile and not (surfaces_doc or {}).get("reconciled"):
        surfaces_doc, _rep = _surfaces.reconcile(surfaces_doc or {}, built)
    p = plan(surfaces_doc, recipe, settings, seed, mode, honour_figures=honour_figures)
    codes = built.codes.copy()
    palette = list(built.palette)
    index = {s: i for i, s in enumerate(palette)}
    written = skipped = stale = restated = 0
    for (x, y, z) in sorted(p["cells"]):
        new, fam, _role, old = p["cells"][(x, y, z)]
        if not built.inside(x, y, z):
            skipped += 1
            continue
        # **The structural world is authoritative.** A cell the record owned can have
        # been cleared since -- a later part's sweep, a doorstep held open -- and a
        # block written into air there is a new block with new physics (the rings
        # comparison: two lint errors from twelve such cells). Only a cell that still
        # holds the block the record saw is finished -- **the same block in the same
        # state**. This compared bare names until the composition round, so a stair re-
        # faced or a slab moved to the top half after the record was made counted as
        # unchanged, and the substitution below then wrote the record's own stale suffix
        # back over it and turned the block round. `surfaces.same_state` says why it
        # compares the properties both sides name.
        stands = built.state(x, y, z)
        if not _surfaces.same_state(old, stands):
            stale += 1
            continue
        # ...and the suffix comes off **the block that stands**, not off the record. The
        # two agree cell for cell on every retained candidate, and they will not on a
        # volume cached off a live world, where the game has normalised the state the
        # builder abbreviated: re-emitting `[facing=north]` over a stair the world holds
        # as `[facing=north,half=top,shape=inner_left]` would keep the family and throw
        # the geometry away. The shape rule ("a stair stays a stair with its state")
        # means the state that is actually there.
        fresh_new = substitute(stands, fam)
        if fresh_new is None:
            skipped += 1
            continue
        if fresh_new != new:
            restated += 1
            new = fresh_new
        i = index.get(new)
        if i is None:
            i = index[new] = len(palette)
            palette.append(new)
        codes[x - built.x0, y - built.y0, z - built.z0] = i
        written += 1
    out = Volume(built.x0, built.y0, built.z0, codes, palette)
    rec = {"record": "material", "version": 1, "mode": mode, "seed": int(seed),
           "settings": dict(settings or {}), "condition": p["condition"],
           "recipe": recipe_digest(recipe),
           "built_digest": built_digest or (surfaces_doc or {}).get("built_digest"),
           "key": key_for(built_digest or (surfaces_doc or {}).get("built_digest"),
                          recipe, dict(settings or {}), seed, mode),
           "substituted": written, "outside_volume": skipped, "stale_skipped": stale,
           "restated_from_the_standing_block": restated,
           "figure_refused": p.get("figure_refused", 0),
           "counts": p["counts"], "totals": p["totals"], "rates": p["rates"],
           "bucket_rates": p.get("bucket_rates") or {},
           **({"matched_rates": p["matched_rates"]} if "matched_rates" in p else {}),
           **({"matched_bucket_rates": p["matched_bucket_rates"]}
              if "matched_bucket_rates" in p else {}),
           "owned_cells": _surfaces.owned_cells(surfaces_doc),
           "editable_cells": _surfaces.editable_cells(surfaces_doc),
           "reconciled": (surfaces_doc or {}).get("reconciled")}
    return out, rec


def volume_digest(vol: Volume) -> str:
    """A content digest of a volume: codes and palette, independent of memory."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vol.codes).tobytes())
    h.update(json.dumps(list(vol.palette)).encode())
    h.update(json.dumps([vol.x0, vol.y0, vol.z0]).encode())
    return h.hexdigest()[:16]


def finished_path(state_dir: str, key: str) -> str:
    return os.path.join(state_dir, "finished", f"world_finished.{key}.npz")
