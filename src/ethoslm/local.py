"""**The local unit of production: one block, with the city round it held still.**

The block design round, and the audit's "why the previous prompt did not close this":

    The "small block" also reduced the wrong part of the cycle. `rounds/nd-block.json`
    seeds the city plan, prepared ground and roads, and runs cropped construction and
    inspection. It is useful for type debugging. It does not make spatial planning and
    revision local. `improve.py:_rebuild` still runs ground, terraces and circulation
    before construction.

Measured on this project's own block, cold, before this module existed: ground 9 s,
**terraces 120 s**, **circulation 358 s**, parts 160 s, finish 2 s, lint 57 s, section 2
s.

So a round may declare a **scope**: a rectangle, and a margin round it.

  - inside the scope, everything is re-decided cold -- the districts are re-compiled,
    their ground is re-proposed and re-terraced, their lanes are re-routed, their parts
    are re-built and re-read;
  - outside it, the seed's terrain, structures, terraces and roads are **boundary
    conditions**: read, crossed, connected to, and never re-decided.

That is a smaller *complete* rebuild and not an incremental scheduler: every stage runs
its own ordinary code over a smaller world, and the dependency that genuinely crosses the
boundary -- the arterial road, the ring wall, the gate, the ground the block is entered
across -- is inside the margin by construction. Where a change needs more than its
declared scope the stage says so and the scope has to be widened explicitly, which is the
audit's "expose dependencies that genuinely require a larger scope; reject undeclared
changes before building".

Nothing here decides anything about a place. It answers three questions -- what is my
scope, is this inside it, and put that piece of the world back as it was -- and the
stages keep their own decisions.
"""
from __future__ import annotations

import numpy as np

#: How far outside the scope rectangle the boundary conditions reach. The lanes, the
#: terraces and the gate approaches all read ground outside the footprint they serve,
#: and `stages_build.CACHE_PAD` has taken this much for the same reason since the cache
#: stage existed. A block's own street is inside its rectangle; the road it joins is
#: not.
SCOPE_MARGIN = 48


def scope_of(rnd) -> dict | None:
    """`flags.local` is `{"rect": [x0,z0,x1,z1], "margin": int}`; `rect` defaults to the
        registered section's, because a round that registers a section and asks for a local
        path means that section. A round with no `local` flag is every round before this one
        and runs the whole city, which is what the section build still does.
        
    """
    loc = (getattr(rnd, "flags", None) or {}).get("local")
    if not loc:
        return None
    if loc is True:
        loc = {}
    rect = loc.get("rect") or ((rnd.flags.get("section") or {}).get("rect"))
    if not rect or len(rect) != 4:
        return None
    x0, z0, x1, z1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    x0, x1 = min(x0, x1), max(x0, x1)
    z0, z1 = min(z0, z1), max(z0, z1)
    m = int(loc.get("margin", SCOPE_MARGIN))
    return {"rect": (x0, z0, x1, z1), "margin": m,
            "outer": (x0 - m, z0 - m, x1 + m, z1 + m),
            "why": loc.get("why") or
                   "the local unit this round plans, prepares, routes, builds and "
                   "revises; everything outside it is a boundary condition"}


def meets(scope: dict | None, rect) -> bool:
    """Does `rect` (x0, z0, x1, z1) reach into the scope's outer bound?

        True for every rectangle when there is no scope, so a caller can ask unconditionally.
        
    """
    if not scope or not rect:
        return True
    ax0, az0, ax1, az1 = (int(v) for v in rect)
    ax0, ax1 = min(ax0, ax1), max(ax0, ax1)
    az0, az1 = min(az0, az1), max(az0, az1)
    bx0, bz0, bx1, bz1 = scope["outer"]
    return ax0 <= bx1 and bx0 <= ax1 and az0 <= bz1 and bz0 <= az1


def inside(scope: dict | None, rect) -> bool:
    """Is `rect` wholly inside the scope's own rectangle -- not its margin?"""
    if not scope:
        return True
    if not rect:
        return False
    ax0, az0, ax1, az1 = (int(v) for v in rect)
    bx0, bz0, bx1, bz1 = scope["rect"]
    return ax0 >= bx0 and az0 >= bz0 and ax1 <= bx1 and az1 <= bz1


def part_meets(scope: dict | None, part: dict) -> bool:
    """Does any rectangle of this plan leaf reach into the scope? Edges and points
    included -- a wall crossing the margin is a boundary condition and is kept."""
    if not scope:
        return True
    from . import pipeline
    try:
        rects = pipeline.part_rects(part)
    except Exception:                            # noqa: BLE001 -- an unplaceable leaf
        return False
    return any(meets(scope, r) for r in rects)


def restore_rect(dst, src, rect) -> int:
    """Copy the columns of `rect` from volume `src` into volume `dst`, whole.

        Used to take back the local lanes before they are routed again: the ground outside
        the scope keeps the roads the seed laid, and the ground inside it goes back to what
        it was before any lane so a re-routing replaces its lanes instead of joining them.
        This is the one operation that crosses the boundary, it is a copy of blocks and it
        decides nothing.

        The two volumes are mapped by world coordinate and their palettes are reconciled, so
        a source block absent from the destination's palette is added rather than lost.
        Returns the number of columns copied.
        
    """
    x0, z0, x1, z1 = (int(v) for v in rect)
    x0, x1 = min(x0, x1), max(x0, x1)
    z0, z1 = min(z0, z1), max(z0, z1)
    ax0 = max(x0, dst.x0, src.x0)
    az0 = max(z0, dst.z0, src.z0)
    ax1 = min(x1, dst.x0 + dst.codes.shape[0] - 1, src.x0 + src.codes.shape[0] - 1)
    az1 = min(z1, dst.z0 + dst.codes.shape[2] - 1, src.z0 + src.codes.shape[2] - 1)
    if ax1 < ax0 or az1 < az0:
        return 0
    ay0 = max(dst.y0, src.y0)
    ay1 = min(dst.y0 + dst.codes.shape[1], src.y0 + src.codes.shape[1])
    if ay1 <= ay0:
        return 0
    # the source's palette, expressed in the destination's
    index = {s: i for i, s in enumerate(dst.palette)}
    remap = np.zeros(len(src.palette), np.uint16)
    for i, s in enumerate(src.palette):
        j = index.get(s)
        if j is None:
            j = len(dst.palette)
            dst.palette.append(s)
            index[s] = j
        remap[i] = j
    got = src.codes[ax0 - src.x0:ax1 - src.x0 + 1,
                    ay0 - src.y0:ay1 - src.y0,
                    az0 - src.z0:az1 - src.z0 + 1]
    dst.codes[ax0 - dst.x0:ax1 - dst.x0 + 1,
              ay0 - dst.y0:ay1 - dst.y0,
              az0 - dst.z0:az1 - dst.z0 + 1] = remap[got]
    dst._tables = None
    return int((ax1 - ax0 + 1) * (az1 - az0 + 1))


def record(scope: dict | None, *, of: str, kept: int, redone: int,
           note: str = "") -> dict:
    """The line a stage writes about what it did locally and what it left alone.

        A local stage has to say which of its pieces it re-decided and which it took as a
        boundary condition, because a reader looking at a block's ground has to be able to
        tell "this was designed for this candidate" from "this is the last candidate's and
        was kept on purpose". Absence would read as the first.
        
    """
    if not scope:
        return {}
    return {"local": {"of": of, "rect": list(scope["rect"]), "margin": scope["margin"],
                      "redone": int(redone), "kept_as_boundary": int(kept),
                      "why": note or scope["why"]}}


def outside_names(rnd, place: dict | None = None) -> set:
    """Their plans are boundary conditions. Empty without a scope."""
    import json
    import os
    scope = scope_of(rnd)
    if scope is None:
        return set()
    if place is None:
        for f in ("plan.place.json", "plan.place.stale.json"):
            p = rnd.rel(f)
            if os.path.exists(p):
                try:
                    place = json.load(open(p))
                except ValueError:
                    place = None
                if place:
                    break
    out = set()
    for d in (place or {}).get("districts") or []:
        if d.get("x1") is not None and not meets(scope, (d["x0"], d["z0"], d["x1"], d["z1"])):
            out.add(str(d["name"]))
    try:
        from . import placeplan
        for n, r in placeplan.compound_rects(place or {}).items():
            if not meets(scope, tuple(r)):
                out.add(str(n))
    except Exception:                                  # noqa: BLE001 -- no compounds
        pass
    return out


def retire_plans(rnd, *, compounds: bool = False, lanes: bool = False,
                 place: dict | None = None) -> dict:
    """**Retire the compiled district (and compound) plans a replan re-derives -- except,
    under a local scope, those outside it**, which are the scope's boundary condition
    (the fabric reset round: every re-plan path deleted every plan, so a scale repair or
    a character revision re-compiled the crowded ring this revision must keep). `lanes`
    retires the network too, and only where there is no scope."""
    import os
    keep = outside_names(rnd, place)
    pre = ("plan.district.", "district_") + (("plan.compound.", "compound_")
                                             if compounds else ())
    removed, kept = 0, 0
    for f in sorted(os.listdir(rnd.state)):
        if not (f.startswith(pre) and os.path.isfile(rnd.rel(f))):
            continue
        if any(f in (f"plan.district.{n}.json", f"district_{n}_compiled.json",
                     f"district_{n}_prompt.md", f"plan.compound.{n}.json",
                     f"compound_{n}_compiled.json", f"compound_{n}_prompt.md")
               for n in keep):
            kept += 1
            continue
        os.remove(rnd.rel(f))
        removed += 1
    if lanes and scope_of(rnd) is None:
        for f in ("network.json", "circulation.json"):
            if os.path.exists(rnd.rel(f)):
                os.remove(rnd.rel(f))
    return {"removed": removed, "kept_outside_scope": kept}
