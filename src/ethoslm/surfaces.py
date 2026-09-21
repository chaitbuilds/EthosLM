"""**Owned surfaces**: what construction laid, by part, role and exposure, editable.

The expression round's material interface. `construction.surfaces` counted block
families; a family cannot say whether a cobblestone is a wall or a floor, which part
laid it, or whether it is a tread a person walks on. The builder now records the role
at the write (`buildlib.Builder._tag`, packed beside every pending block), and this
module turns that into the compact record a material pass reads:

    record(builder, part)      one part's owned cells by role, with exposure context
    write(path, records)       `surfaces.json` for a candidate; `read(path)` back
    histogram(rec)             the old `construction.surfaces` shape, from the record
    flags_of(flags)            the exposure bits of one cell, as names

A cell is `[x, y, z, state_index, flags]` under its role, with the part's own state
table beside it. `flags` says what a pass may condition on: which faces are open to
air, whether the sky is open above, ground contact, water within two, whether an open
face is sheltered under an overhang, and whether the cell sits under an opening -- and
one bit that is not about the weather at all, `figure`, which says the type that laid
the cell declared it part of a pattern it drew (`Primitives.figure`) and no pass may
repaint it. The
protected cells -- treads, doors, glass, fittings, lights, the ground the library laid
-- are listed apart and are never edited; a cell nobody owns is not in the record at
all, so a pass cannot reach the landscape or another part's work.

Nothing here decides a material. It is the ownership a pass is not allowed to guess.
"""
from __future__ import annotations

import json
import os

from .prims import SURFACE_ROLES, EDITABLE_ROLES

AIR = frozenset(("air", "cave_air", "void_air"))

#: The exposure bits on a cell, registered here before any pass conditioned on them.
FLAGS = {"open_north": 1, "open_south": 2, "open_east": 4, "open_west": 8,
         "open_up": 16, "open_down": 32, "sky": 64, "ground_contact": 128,
         "water_near": 256, "sheltered": 512, "under_opening": 1024,
         "base_course": 2048,
         #: **This cell is part of a figure somebody drew.** Not an exposure bit: the
         #: others say what the weather does to a cell, this says what the type meant by
         #: it. Declared with `Primitives.figure`, carried through `reconcile` the way
         #: ground contact is, and refused by `material.plan`. Composition round; the
         #: design round's cause 4 (`out/des-material/comparison.json`).
         "figure": 4096}
_FACES = (("open_north", 0, -1), ("open_south", 0, 1), ("open_east", 1, 0),
          ("open_west", -1, 0))
#: How far a wall may stand above the ground it meets and still be a base course.
BASE_COURSES = 2
#: How far above a cell an overhang may be and still shelter its open face. Two: the
#: course under an eave and the one below it; a wall's own upper courses are not an
#: overhang, and with four every wall under a roof read as sheltered.
SHELTER_REACH = 2
#: The roles that are the ground a wall stands on, for the base-course rule: the world's
#: own surface, the pad and footing the library laid, a path, a tread.
GROUND_ROLES = ("ground", "footing", "path", "step")
WATER_REACH = 2


def _name(state: str) -> str:
    return str(state).split("[")[0].split(":")[-1]


def _parse(state: str) -> tuple:
    """(bare name, {property: value}) of a block state."""
    name, _, rest = str(state).partition("[")
    props: dict = {}
    if rest:
        for kv in rest.rstrip("]").split(","):
            k, _, v = kv.partition("=")
            props[k.strip()] = v.strip()
    return name.split(":")[-1], props


def same_state(saw: str, stands: str) -> bool:
    """Is the block that stands the block the record saw -- **including its state**?

        Composition round, Q1. Freshness used to be decided on the bare name
        (`_name`), so a cell whose stair was re-faced the other way, whose slab moved from
        bottom to top or whose log changed axis after the record was made still read
        `fresh`, and `material.substitute` then re-emitted the *record's* suffix over it and
        silently turned the block back round. Orientation is physics here -- a tread faces
        the way somebody climbs -- so a cell whose state has moved is stale and is left
        alone.
    """
    a, pa = _parse(saw)
    b, pb = _parse(stands)
    if a != b:
        return False
    return all(pb[k] == v for k, v in pa.items() if k in pb)


def flags_of(flags: int) -> list:
    return [k for k, v in FLAGS.items() if flags & v]


def _world_name(builder, x: int, y: int, z: int) -> str:
    """The block the world holds at a cell the builder did not write."""
    vol = getattr(builder, "_vol", None)
    if vol is not None:
        return _name(vol.state(x, y, z))
    try:
        return _name(builder.world_site.editor.getBlock((x, y, z)).id)
    except Exception:                                   # noqa: BLE001 -- a reading
        return "air"


def _solid_name(n: str) -> bool:
    return n not in AIR and n != "water" and n != "lava"


def record(builder, part: dict, *, part_index: int | None = None,
           voice_name: str | None = None) -> dict:
    """One part's owned surfaces off the builder's emission record.

        `part_index` selects the builder's part where it built several; by default the
        last part sited (the way `instantiate_part` builds: one part per builder).
        
    """
    owner = getattr(builder, "_owner", None) or {}
    pending = getattr(builder, "_pending", None) or {}
    figures = getattr(builder, "figure_cells", None) or {}
    idx = part_index if part_index is not None else max(0, len(builder.parts) - 1) \
        if getattr(builder, "parts", None) else 0
    states: list = []
    sidx: dict = {}
    by_role: dict = {}
    protected: list = []
    how = {"context": 0, "tag": 0, "family": 0, "hint": 0}
    hows = ("context", "tag", "family", "hint")
    ground_y: dict = {}
    fig_named: dict = {}

    def is_solid(x, y, z) -> bool:
        s = pending.get((x, y, z))
        if s is not None:
            return _solid_name(_name(s))
        return _solid_name(_world_name(builder, x, y, z))

    def is_open(x, y, z) -> bool:
        s = pending.get((x, y, z))
        n = _name(s) if s is not None else _world_name(builder, x, y, z)
        return n in AIR

    def water_near(x, y, z) -> bool:
        for dx in range(-WATER_REACH, WATER_REACH + 1):
            for dz in range(-WATER_REACH, WATER_REACH + 1):
                for dy in (-1, 0):
                    if (x + dx, y + dy, z + dz) in pending:
                        continue
                    if _world_name(builder, x + dx, y + dy, z + dz) == "water":
                        return True
        return False

    def ground_at(x, z) -> int | None:
        """The level a wall stands on in this column: the highest solid that is the
        world's own or the library's ground work (the pad, the footing, a path)."""
        if (x, z) in ground_y:
            return ground_y[(x, z)]
        vol = getattr(builder, "_vol", None)
        g = None
        if vol is not None and vol.inside(x, vol.y0, z):
            for y in range(vol.y0 + vol.shape[1] - 1, vol.y0 - 1, -1):
                s = pending.get((x, y, z))
                if s is not None:
                    o = owner.get((x, y, z))
                    role = SURFACE_ROLES[o & 15] if o is not None else "unknown"
                    if role in GROUND_ROLES and _solid_name(_name(s)):
                        g = y
                        break
                    continue
                if _solid_name(_name(vol.state(x, y, z))):
                    g = y
                    break
        ground_y[(x, z)] = g
        return g

    cells = sorted(c for c, v in owner.items() if (v >> 7) == idx and c in pending)
    for (x, y, z) in cells:
        v = owner[(x, y, z)]
        role = SURFACE_ROLES[v & 15]
        prot = bool(v & 16)
        how[hows[(v >> 5) & 3]] += 1
        st = pending[(x, y, z)]
        st = str(st).split(":")[-1]
        si = sidx.get(st)
        if si is None:
            si = sidx[st] = len(states)
            states.append(st)
        if prot or role not in EDITABLE_ROLES:
            protected.append([x, y, z, si, role])
            continue
        f = 0
        fig = figures.get((x, y, z))
        if fig is not None:
            # a figure the type declared: what the pattern is, not what the weather does
            # to it, and the one bit a material pass is refused on
            f |= FLAGS["figure"]
            fig_named[fig] = fig_named.get(fig, 0) + 1
        open_faces = []
        for key, dx, dz in _FACES:
            if is_open(x + dx, y, z + dz):
                f |= FLAGS[key]
                open_faces.append((dx, dz))
        if is_open(x, y + 1, z):
            f |= FLAGS["open_up"]
        if is_open(x, y - 1, z):
            f |= FLAGS["open_down"]
        # sky: nothing solid above in this column, pending or world
        sky = True
        for yy in range(y + 1, y + 40):
            if is_solid(x, yy, z):
                sky = False
                break
        if sky:
            f |= FLAGS["sky"]
        # ground contact: the block below is the world's or the library's ground work
        below = (x, y - 1, z)
        if below not in pending:
            if _solid_name(_world_name(builder, x, y - 1, z)):
                f |= FLAGS["ground_contact"]
        else:
            o = owner.get(below)
            if o is not None and SURFACE_ROLES[o & 15] in GROUND_ROLES \
                    and role not in GROUND_ROLES:
                f |= FLAGS["ground_contact"]
        g = ground_at(x, z)
        if g is not None and 1 <= y - g <= BASE_COURSES and role not in GROUND_ROLES:
            f |= FLAGS["base_course"]
        if open_faces and water_near(x, y, z):
            f |= FLAGS["water_near"]
        # sheltered: an open face under an overhang -- something solid within reach in
        # the column one step OUT from the face, never this column's own courses
        if open_faces:
            shel = False
            for dx, dz in open_faces:
                for k in range(1, SHELTER_REACH + 1):
                    if is_solid(x + dx, y + k, z + dz):
                        shel = True
                        break
                if shel:
                    break
            if shel:
                f |= FLAGS["sheltered"]
        # under an opening: a door or glass one or two above
        for k in (1, 2):
            o = owner.get((x, y + k, z))
            if o is not None and SURFACE_ROLES[o & 15] in ("door", "glass"):
                f |= FLAGS["under_opening"]
                break
        by_role.setdefault(role, []).append([x, y, z, si, f])
    voice = part.get("voice") if isinstance(part.get("voice"), dict) else {}
    return {"part": part.get("name") or part.get("label"), "type": part.get("type"),
            "kind": part.get("kind", "plot"), "floor_y": part.get("floor_y"),
            "voice": {k: str(v) for k, v in voice.items() if isinstance(v, str)},
            "voice_name": (voice_name or (part.get("voice")
                                          if isinstance(part.get("voice"), str) else None)),
            "states": states, "cells": by_role, "protected": protected,
            "counts": {r: len(v) for r, v in by_role.items()},
            "figures": dict(sorted(fig_named.items())),
            "how": how, "owned": len(cells)}


# ------------------------------------------------ the record against the world v2,
# design round, C2. `record` is made inside one part's builder, before the part next to
# it exists. Four things can be true of a recorded cell by the time the place is
# assembled, and the pass may edit none of them: stale the block the record saw is not
# the block that stands (a later sweep cleared it, a doorstep was held open).
# `material.apply` already refuses these; reconciling them here makes the count visible
# instead of a silent skip. overwritten two parts wrote the same cell and the later one
# won. The cell belongs to the later part -- under *its* role and *its* voice -- and the
# earlier part's claim on it is not permission. protected any part's protected list
# beats every part's editable claim. A tread, a door, a glass, a light, the ground.
# unexposed no air on any of its six faces in the assembled world. Nobody can see it, so
# it is not worth an edit and not worth a comparison. and one thing must be recomputed
# rather than trusted: **exposure**. `open_south` was true of a wall that had nothing
# south of it when it was built; the house built next to it later is what a material
# pass is conditioned on now.

def reconcile(doc: dict, built, *, note: str = "") -> tuple:
    """(reconciled doc, report) of an emission record against the assembled world.

        `built` is the finished structural volume -- `world_built.npz`, after every part.
        The returned doc has the same shape as `read()` gives, with the cells that survive
        and their exposure flags recomputed from that world; the report says how many of
        each kind were dropped and why. Nothing is added: a cell the record does not own
        stays unowned, because unknown ownership is not permission to recolour the
        landscape or another part's work.
        
    """
    parts = list((doc or {}).get("parts") or [])
    pal = list(built.palette)
    pal_name = [_name(s) for s in pal]
    #: the palette **with its state**, for the freshness test: `same_state`, not `_name`
    pal_state = [str(s).split(":")[-1] for s in pal]
    air_code = {i for i, n in enumerate(pal_name) if n in AIR}
    # numpy is imported here and not at the top: everything else in this module is a
    # record a builder writes, and a reader of `surfaces.json` should not need an array
    # library to read one. A whole-volume mask of the city would be 132 MB, so the
    # neighbour lookups below gather at the cells instead.
    import numpy as np

    def gather(cx, cy, cz):
        """The palette code at world cells, -1 outside the volume."""
        ix, iy, iz = cx - built.x0, cy - built.y0, cz - built.z0
        ok = ((ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < built.shape[0])
              & (iy < built.shape[1]) & (iz < built.shape[2]))
        out = np.full(len(cx), -1, np.int64)
        if ok.any():
            out[ok] = built.codes[ix[ok], iy[ok], iz[ok]]
        return out

    # every editable claim, flat, in build order; the last writer of a cell owns it
    xs, ys, zs, pidx, roles, sidx, flags = [], [], [], [], [], [], []
    for i, prec in enumerate(parts):
        for role, cells in sorted((prec.get("cells") or {}).items()):
            for c in cells:
                xs.append(c[0]); ys.append(c[1]); zs.append(c[2])
                pidx.append(i); roles.append(role); sidx.append(c[3]); flags.append(c[4])
    n = len(xs)
    rep = {"record": "surfaces_reconciled", "version": 1, "parts": len(parts),
           "recorded_editable": n, "protected_recorded": sum(
               len(p.get("protected") or []) for p in parts)}
    if not n:
        return dict(doc or {}, parts=parts, reconciled=rep), rep
    X = np.array(xs, np.int64); Y = np.array(ys, np.int64); Z = np.array(zs, np.int64)
    P = np.array(pidx, np.int64)
    #: the later part wins: walk in build order and keep the last writer per cell
    last: dict = {}
    for k in range(n):
        last[(xs[k], ys[k], zs[k])] = k
    mine = np.zeros(n, bool)
    mine[np.fromiter(last.values(), np.int64, len(last))] = True
    # any part's protected cell beats every part's editable claim
    prot = {(r[0], r[1], r[2]) for p in parts for r in (p.get("protected") or [])}
    prot_role = {(r[0], r[1], r[2]): (r[4] if len(r) > 4 else "unknown")
                 for p in parts for r in (p.get("protected") or [])}
    clash = np.fromiter(((xs[k], ys[k], zs[k]) in prot for k in range(n)), bool, n)
    # The block that stands, against the block the record saw, **with its state**. This
    # compared bare names until the composition round, so a cell whose stair had been
    # re-faced or whose slab had moved to the top half since the record was made still
    # read fresh and the pass wrote the record's own stale suffix back over it. The
    # freshness test is `same_state`, which says why it compares the properties both
    # sides name. A cache of the two-state answer, because a city record holds 167,270
    # cells over 145 distinct states and 763 palette entries.
    here = gather(X, Y, Z)
    _fresh_cache: dict = {}

    def _is_fresh(k: int) -> bool:
        c = int(here[k])
        key = (c, pidx[k], sidx[k])
        got = _fresh_cache.get(key)
        if got is None:
            got = _fresh_cache[key] = (
                c >= 0 and same_state(parts[pidx[k]]["states"][sidx[k]], pal_state[c]))
        return got
    fresh = np.fromiter((_is_fresh(k) for k in range(n)), bool, n)
    # exposure, recomputed against the assembled world
    open_bits = np.zeros(n, np.int64)
    for key, dx, dy, dz in (("open_north", 0, 0, -1), ("open_south", 0, 0, 1),
                            ("open_east", 1, 0, 0), ("open_west", -1, 0, 0),
                            ("open_up", 0, 1, 0), ("open_down", 0, -1, 0)):
        nb = gather(X + dx, Y + dy, Z + dz)
        isair = np.isin(nb, list(air_code)) if air_code else np.zeros(n, bool)
        open_bits |= np.where(isair, FLAGS[key], 0)
    seen = open_bits != 0
    keep = mine & ~clash & fresh & seen
    rep.update({
        "overwritten_by_a_later_part": int((~mine).sum()),
        "protected_by_another_part": int((mine & clash).sum()),
        "stale_block_no_longer_stands": int((mine & ~clash & ~fresh).sum()),
        "unexposed_in_the_assembled_world": int((mine & ~clash & fresh & ~seen).sum()),
        "kept": int(keep.sum()),
        "how": ("every editable claim flattened in build order; the last writer owns "
                "the cell; any part's protected list wins; the block that stands is "
                "compared with the block the record saw; the six faces are read off "
                "the assembled volume, not off the part's own neighbourhood")})
    # sheltered and under_opening follow from the new faces; ground contact, water and
    # the base course are of the ground the part stands on, which no later part owns, so
    # those bits are carried over -- and so is `figure`, which is not about the
    # assembled world at all. It is what the type that laid the cell said the cell is,
    # and no later neighbour can make a chequer stop being a chequer. A cell a later
    # part *overwrote* is dropped whole by `mine`, so carrying the bit cannot protect
    # somebody else's block.
    carry = (FLAGS["ground_contact"] | FLAGS["water_near"] | FLAGS["base_course"]
             | FLAGS["figure"])
    F_open = (FLAGS["open_north"] | FLAGS["open_south"] | FLAGS["open_east"]
              | FLAGS["open_west"])
    shel = np.zeros(n, bool)
    for key, dx, dz in _FACES:
        face = (open_bits & FLAGS[key]) != 0
        for k in range(1, SHELTER_REACH + 1):
            nb = gather(X + dx, Y + k, Z + dz)
            solid = (nb >= 0) & ~np.isin(nb, list(air_code) or [-1])
            shel |= face & solid
    under = np.fromiter(
        (any(prot_role.get((xs[j], ys[j] + k, zs[j])) in ("door", "glass")
             for k in (1, 2)) for j in range(n)), bool, n)
    sky = np.array([f & FLAGS["sky"] != 0 for f in flags], bool)
    newflags = (open_bits | np.where(shel, FLAGS["sheltered"], 0)
                | np.where(under, FLAGS["under_opening"], 0)
                | np.where(sky, FLAGS["sky"], 0)
                | (np.array(flags, np.int64) & carry))
    moved = int(((newflags & F_open) != (np.array(flags, np.int64) & F_open))[keep].sum())
    rep["exposure_recomputed_on_kept"] = moved
    figbit = (np.array(flags, np.int64) & FLAGS["figure"]) != 0
    rep["figure_cells_recorded"] = int(figbit.sum())
    rep["figure_cells_kept"] = int((figbit & keep).sum())
    figs: dict = {}
    for prec in parts:
        for k, v in (prec.get("figures") or {}).items():
            figs[k] = figs.get(k, 0) + int(v)
    rep["figures"] = dict(sorted(figs.items()))
    out_parts = []
    at = 0
    for i, prec in enumerate(parts):
        by_role: dict = {}
        for role, cells in sorted((prec.get("cells") or {}).items()):
            for c in cells:
                if keep[at]:
                    by_role.setdefault(role, []).append(
                        [c[0], c[1], c[2], c[3], int(newflags[at])])
                at += 1
        out_parts.append({**prec, "cells": by_role,
                          "counts": {r: len(v) for r, v in by_role.items()},
                          "owned": sum(len(v) for v in by_role.values())
                          + len(prec.get("protected") or [])})
    out = dict(doc or {}, parts=out_parts, reconciled=rep,
               note=(note or (doc or {}).get("note") or ""))
    return out, rep


def editable_cells(doc: dict) -> int:
    """How many cells a material pass may edit: the record's own arithmetic, so a
    reconciled doc and a raw one are compared on the same number."""
    return sum(len(c) for p in (doc or {}).get("parts") or []
               for c in (p.get("cells") or {}).values())


def histogram(rec: dict) -> dict:
    """The shape `construction.surfaces` always returned, from the emission record."""
    roles: dict = {}
    for r, cells in (rec.get("cells") or {}).items():
        roles[r] = roles.get(r, 0) + len(cells)
    for row in rec.get("protected") or []:
        r = row[4] if len(row) > 4 else "unknown"
        roles[r] = roles.get(r, 0) + 1
    by_face = {"north": 0, "south": 0, "east": 0, "west": 0, "up": 0}
    exposed = 0
    for row in (rec.get("cells") or {}).get("wall") or []:
        f = row[4]
        for key, face in (("open_north", "north"), ("open_south", "south"),
                          ("open_east", "east"), ("open_west", "west")):
            if f & FLAGS[key]:
                by_face[face] += 1
                exposed += 1
        if f & FLAGS["open_up"]:
            by_face["up"] += 1
    total = sum(roles.values())
    return {"blocks": total, "roles": roles, "exposed_wall_faces": exposed,
            "by_face": by_face, "known": bool(rec.get("voice")),
            "recorded": "at emission", "how": rec.get("how"),
            "protected": len(rec.get("protected") or []),
            "why": ("owned at the write: the role each block was laid as, the part "
                    "that laid it, and what is protected")}


def write(path: str, records: list, *, candidate: str | None = None,
          built_digest: str | None = None, note: str = "") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc = {"record": "surfaces", "version": 1, "candidate": candidate,
           "built_digest": built_digest, "flags": FLAGS, "roles": list(SURFACE_ROLES),
           "editable": list(EDITABLE_ROLES), "parts": list(records), "note": note}
    with open(path, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    return path


def read(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def owned_cells(doc: dict) -> int:
    return sum(p.get("owned") or 0 for p in (doc or {}).get("parts") or [])
