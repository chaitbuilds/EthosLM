"""The ground contract: what a part needs of the ground, settled once. v2, B1.

Until this module, each part sited its own pad as it built and the result was the
ground for the next part -- `site()` read the volume as the parts before it had left
it, and a courtyard house in the concentric run stood at y=38 over ground 56..62
because its neighbour's sited record had become the ground it was sited to. Terraces
existed only for rings, the plateau was its own stage, and a gate's approach was a
third shape of the same idea.

Now every part and every network **declares** what it needs of the ground:

  * a **platform** -- one level over a set of columns: a pad, a square, a terrace, a
    podium, the level run outside a gate;
  * a **profile** -- a level per column: a wall's footing along its run, a lane;
  * **preserve** -- the columns as found: water, a verge;
  * a **clearance** -- columns nothing may claim.

One resolver settles every column before a block is placed, by a **total precedence**
over the classes a declaration is made in -- `PRECEDENCE`, water and footprints first,
then declared ground such as plazas and courts, then streets, then doorsteps, fields
and verges -- and, within a class, by the order the declarations were made. The seams
between neighbours are **derived from the drop**, never declared: a kerb, a bank, a
retaining face. After resolution the ground is read-only (`Resolved`), and the surface
heightmap the build reads is the one fixed here.

Two things the resolver holds by construction, whoever declared:

  * a platform's level is within `relief` of its own ground as found -- the lowest of
    its columns below, the highest above -- so no pad can be sited to a neighbour's
    record; and
  * over water a platform is a deck: never below the waterline.

Every column is owned once; every declaration knows which of its columns it won; and
the record (`Resolved.record()`) says every number and why.
"""
from __future__ import annotations

import json
import os
from types import MappingProxyType

import numpy as np

#: The classes a declaration is made in, first wins. Registered.
PRECEDENCE = ("water", "footprint", "designed", "street", "doorstep", "field", "verge")

#: What a declaration asks of its columns.
KINDS = ("platform", "profile", "preserve", "clearance")

#: The seams, by the drop between two neighbouring columns that different declarations
#: own, or one owns and the other is the ground as found. Registered: a drop of one is a
#: kerb, stepped without a stair; two or three is a bank -- a ramp or a flight; more is
#: a retaining face. Nothing declares a seam.
SEAM_KERB = 1
SEAM_BANK = 3

#: How far a platform's level may stand from its own ground as found. `Builder`'s
#: `SITE_RELIEF`, restated here so this module has no builder in it.
RELIEF = 3


def seam_kind(drop: int) -> str | None:
    """The seam a drop of `drop` blocks between two neighbours is: None where level."""
    d = abs(int(drop))
    if d == 0:
        return None
    if d <= SEAM_KERB:
        return "kerb"
    if d <= SEAM_BANK:
        return "bank"
    return "face"


class Declaration:
    """One thing one part asks of the ground. Frozen once made."""

    __slots__ = ("label", "kind", "cls", "columns", "level", "rect", "reason",
                 "decision", "index", "_frozen")

    def __init__(self, label: str, kind: str, cls: str, columns: dict, *,
                 level: int | None = None, rect=None, reason: str = "",
                 decision: dict | None = None, index: int = -1):
        if kind not in KINDS:
            raise ValueError(f"a declaration is one of {KINDS}, not {kind!r}")
        if cls not in PRECEDENCE:
            raise ValueError(f"a declaration's class is one of {PRECEDENCE}, "
                             f"not {cls!r}")
        if kind == "platform" and level is None:
            raise ValueError(f"{label}: a platform declares a level")
        cols = {}
        for c, v in dict(columns).items():
            key = (int(c[0]), int(c[1]))
            cols[key] = (None if v is None else int(v))
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "columns", MappingProxyType(cols))
        object.__setattr__(self, "level", None if level is None else int(level))
        object.__setattr__(self, "rect", tuple(int(v) for v in rect) if rect else None)
        object.__setattr__(self, "reason", str(reason))
        object.__setattr__(self, "decision", dict(decision or {}))
        object.__setattr__(self, "index", int(index))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"a declaration is frozen once made ({self.label}.{name})")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (f"Declaration({self.label!r}, {self.kind}, {self.cls}, "
                f"{len(self.columns)} columns"
                + (f", level {self.level}" if self.level is not None else "") + ")")


class Contract:
    """The declarations of one build, before any of them is settled."""

    def __init__(self):
        self._decls: list[Declaration] = []

    # --- the four things a part may ask --------------------------------------

    def _add(self, label, kind, cls, columns, **more) -> Declaration:
        d = Declaration(label, kind, cls, columns, index=len(self._decls), **more)
        self._decls.append(d)
        return d

    def platform(self, label: str, columns, level: int, cls: str = "footprint", *,
                 rect=None, reason: str = "", decision: dict | None = None) -> Declaration:
        """One level over `columns` -- an iterable of (x, z), or a rectangle
        (x0, z0, x1, z1) corners inclusive."""
        cols = _columns(columns)
        return self._add(label, "platform", cls, {c: int(level) for c in cols},
                         level=int(level), rect=rect or _rect_of(columns),
                         reason=reason, decision=decision)

    def profile(self, label: str, levels: dict, cls: str = "footprint", *,
                rect=None, reason: str = "", decision: dict | None = None) -> Declaration:
        """A level per column: `{(x, z): y}`."""
        return self._add(label, "profile", cls, dict(levels), rect=rect,
                         reason=reason, decision=decision)

    def preserve(self, label: str, columns, cls: str = "verge", *,
                 reason: str = "") -> Declaration:
        """These columns as found."""
        return self._add(label, "preserve", cls, {c: None for c in _columns(columns)},
                         rect=_rect_of(columns), reason=reason)

    def clearance(self, label: str, columns, cls: str = "verge", *,
                  reason: str = "") -> Declaration:
        """Nothing may claim these columns; they stay as found."""
        return self._add(label, "clearance", cls, {c: None for c in _columns(columns)},
                         rect=_rect_of(columns), reason=reason)

    def network(self, net, *, label: str = "lanes") -> list:
        """What a circulation network asks: every lane cell at its level (a street)
        and every reserved threshold -- the lane cell and the doorstep -- at the
        threshold's level (a doorstep)."""
        out = []
        if net is None:
            return out
        lanes = {(int(x), int(z)): int(rec["y"]) for (x, z), rec in net.cells.items()}
        if lanes:
            out.append(self.profile(label, lanes, cls="street",
                                    reason=f"the circulation network: {len(lanes)} "
                                           f"lane cells at the level each was laid"))
        for t in net.thresholds:
            cols = {(int(t.x), int(t.z)): int(t.y),
                    (int(t.door[0]), int(t.door[2])): int(t.y)}
            out.append(self.profile(f"{t.id}/doorstep", cols, cls="doorstep",
                                    reason=f"the doorstep the circulation pass reserved "
                                           f"for {t.id}, levelled to its lane at y={t.y}"))
        return out

    @property
    def declarations(self) -> tuple:
        return tuple(self._decls)

    def __len__(self) -> int:
        return len(self._decls)

    # --- the resolution ------------------------------------------------------

    def resolve(self, bed=None, wet=None, *, relief: int = RELIEF,
                surface=None) -> "Resolved":
        """Settle every declared column. Returns the read-only `Resolved`.

                `bed(x, z)` is the ground as found under a column -- the bed, never the
                waterline -- and `wet(x, z)` the waterline or None; both may be None where
                the declarations carry their own readings (`decision["grade"]`,
                `decision["waterline"]`), and where `bed` is given it is the authority. `surface`
                is the heightmap the build then reads, fixed here: an object with `.x0`,
                `.z0` and a 2-D array `.h`, or None.
                
        """
        order = sorted(range(len(self._decls)),
                       key=lambda i: (PRECEDENCE.index(self._decls[i].cls), i))
        owner: dict = {}
        level: dict = {}
        clamped: dict = {}
        decks: dict = {}
        for i in order:
            d = self._decls[i]
            lvl_of = dict(d.columns)
            if d.kind == "platform":
                lvl, note = _held_level(d, bed, wet, relief)
                if note:
                    clamped[i] = note
                if lvl != d.level:
                    lvl_of = {c: lvl for c in lvl_of}
                    decks[i] = lvl
            for c, v in lvl_of.items():
                if c in owner:
                    continue
                if v is None:
                    if bed is None:
                        continue
                    g = bed(c[0], c[1])
                    if g is None:
                        continue
                    v = int(g)
                owner[c] = i
                level[c] = int(v)
        won: dict = {i: {} for i in range(len(self._decls))}
        for c, i in owner.items():
            won[i][c] = level[c]
        seams = _seams(self._decls, owner, level, bed)
        return Resolved(self._decls, owner, level, won, seams, clamped, decks,
                        relief=relief, surface=surface)


def _held_level(d: Declaration, bed, wet, relief: int) -> tuple:
    """A platform's level, held within `relief` of its own ground as found and never
    below the waterline over water. `(level, note)`; the note says what moved."""
    lvl = int(d.level)
    dec = d.decision or {}
    lo = hi = None
    # The ground a platform is held to is its own: the columns of its rectangle where it
    # has one (a pad's ledge lies outside it and stands at the pad's level whatever the
    # bank beside it does), else every column it declared.
    own = list(d.columns)
    if d.rect is not None:
        x0, z0, x1, z1 = d.rect
        own = [c for c in own if x0 <= c[0] <= x1 and z0 <= c[1] <= z1] or own
    if bed is not None:
        gs = [bed(x, z) for (x, z) in own]
        gs = [int(g) for g in gs if g is not None]
        if gs:
            lo, hi = min(gs), max(gs)
    if lo is None and dec.get("grade"):
        lo, hi = int(dec["grade"][0]), int(dec["grade"][1])
    water = None
    if wet is not None:
        ws = [wet(x, z) for (x, z) in d.columns]
        ws = [int(w) for w in ws if w is not None]
        water = max(ws) if ws else None
    if water is None and dec.get("waterline") is not None:
        water = int(dec["waterline"])
    deck = bool(dec.get("ground") == "deck")
    note = None
    # Designed ground -- a terrace, a podium, a plaza the layout levelled -- stands at
    # the level it was designed to, cut or filled; what is held within reach of its own
    # ground is a footprint, and a field.
    held = d.cls in ("footprint", "field")
    if lo is not None and not deck and held:
        if lvl < lo - relief:
            note = {"asked": lvl, "got": int(lo), "why": f"below its own ground "
                    f"y={lo}..{hi} by more than {relief}: held at the lowest column"}
            lvl = int(lo)
        elif lvl > hi + relief:
            note = {"asked": lvl, "got": int(hi), "why": f"above its own ground "
                    f"y={lo}..{hi} by more than {relief}: held at the highest column"}
            lvl = int(hi)
    if deck and water is not None and lvl <= water:
        note = {"asked": lvl, "got": int(water) + 1,
                "why": f"a deck over water at y={water} stands above the waterline"}
        lvl = int(water) + 1
    return lvl, note


def _seams(decls: list, owner: dict, level: dict, bed) -> dict:
    """Every seam between two owners, or an owner and the ground, by kind. Derived
    from the drop and nothing else. `{(a, b): {"kerb": n, "bank": n, "face": n}}`
    with `b` a label or `"ground"`."""
    out: dict = {}
    for (x, z), i in owner.items():
        a = decls[i].label
        y = level[(x, z)]
        for (nx, nz) in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            j = owner.get((nx, nz))
            if j is not None:
                # each pair of owners counted once, from the earlier declaration's side
                if j <= i:
                    continue
                b = decls[j].label
                other = level[(nx, nz)]
            else:
                if bed is None:
                    continue
                g = bed(nx, nz)
                if g is None:
                    continue
                b, other = "ground", int(g)
            k = seam_kind(y - other)
            if k is None:
                continue
            rec = out.setdefault((a, b), {"kerb": 0, "bank": 0, "face": 0})
            rec[k] += 1
    return out


class Resolved:
    """The ground, settled. Read-only: every accessor answers off the resolution and
    nothing here can be assigned to after it is made."""

    __slots__ = ("_decls", "_owner", "_level", "_won", "_seams", "_clamped", "_decks",
                 "_by_label", "relief", "surface", "_frozen")

    def __init__(self, decls, owner, level, won, seams, clamped, decks, *,
                 relief: int, surface=None):
        object.__setattr__(self, "_decls", tuple(decls))
        object.__setattr__(self, "_owner", MappingProxyType(dict(owner)))
        object.__setattr__(self, "_level", MappingProxyType(dict(level)))
        object.__setattr__(self, "_won", MappingProxyType(
            {i: MappingProxyType(dict(v)) for i, v in won.items()}))
        object.__setattr__(self, "_seams", MappingProxyType(
            {k: MappingProxyType(dict(v)) for k, v in seams.items()}))
        object.__setattr__(self, "_clamped", MappingProxyType(dict(clamped)))
        object.__setattr__(self, "_decks", MappingProxyType(dict(decks)))
        by: dict = {}
        for d in self._decls:
            by.setdefault(d.label, []).append(d)
        object.__setattr__(self, "_by_label", MappingProxyType(
            {k: tuple(v) for k, v in by.items()}))
        object.__setattr__(self, "relief", int(relief))
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        raise AttributeError("the resolved ground is read-only")

    def __delattr__(self, name):
        raise AttributeError("the resolved ground is read-only")

    # --- per column ------------------------------------------------------------

    def level(self, x: int, z: int) -> int | None:
        """The settled level of a column, or None where nothing declared it."""
        return self._level.get((int(x), int(z)))

    def owner(self, x: int, z: int) -> str | None:
        i = self._owner.get((int(x), int(z)))
        return None if i is None else self._decls[i].label

    def cls(self, x: int, z: int) -> str | None:
        i = self._owner.get((int(x), int(z)))
        return None if i is None else self._decls[i].cls

    def height(self, x: int, z: int) -> int | None:
        """The surface the build reads at a column: the settled level where one is
        declared, else the heightmap fixed at resolution, else None."""
        y = self.level(x, z)
        if y is not None:
            return y
        s = self.surface
        if s is None:
            return None
        i, j = int(x) - s.x0, int(z) - s.z0
        if 0 <= i < s.h.shape[0] and 0 <= j < s.h.shape[1]:
            return int(s.h[i, j])
        return None

    # --- per declaration ---------------------------------------------------------

    @property
    def declarations(self) -> tuple:
        return self._decls

    def has(self, label: str) -> bool:
        return label in self._by_label

    def declaration(self, label: str, kind: str | None = None,
                    cls: str | None = None) -> Declaration | None:
        """The first declaration under `label` (of that kind or class, where asked)."""
        for d in self._by_label.get(label, ()):
            if (kind is None or d.kind == kind) and (cls is None or d.cls == cls):
                return d
        return None

    def declarations_of(self, label: str) -> tuple:
        return self._by_label.get(label, ())

    def columns_of(self, label: str) -> dict:
        """The columns this label **won**, at their settled levels (a new dict)."""
        out: dict = {}
        for d in self._by_label.get(label, ()):
            out.update(self._won.get(d.index, {}))
        return out

    def level_of(self, label: str) -> int | None:
        """The settled level of a platform declared under `label`, or None."""
        d = self.declaration(label, kind="platform")
        if d is None:
            return None
        return self._decks.get(d.index, d.level)

    def clamped(self, label: str) -> dict | None:
        for d in self._by_label.get(label, ()):
            if d.index in self._clamped:
                return dict(self._clamped[d.index])
        return None

    @property
    def seams(self) -> MappingProxyType:
        return self._seams

    def seams_of(self, label: str) -> dict:
        out: dict = {}
        for (a, b), rec in self._seams.items():
            if a == label or b == label:
                out[(a, b)] = dict(rec)
        return out

    def __len__(self) -> int:
        return len(self._level)

    # --- the record ---------------------------------------------------------------

    def record(self) -> dict:
        """Every number and why, as JSON."""
        decls = []
        for d in self._decls:
            won = self._won.get(d.index, {})
            decls.append({
                "label": d.label, "kind": d.kind, "class": d.cls,
                "level": self._decks.get(d.index, d.level),
                "asked": d.level, "rect": list(d.rect) if d.rect else None,
                "columns": len(d.columns), "won": len(won),
                "clamped": dict(self._clamped[d.index]) if d.index in self._clamped else None,
                "ground": (d.decision or {}).get("ground"),
                "grade": (d.decision or {}).get("grade"),
                "reason": d.reason})
        by_class: dict = {}
        for i in self._owner.values():
            c = self._decls[i].cls
            by_class[c] = by_class.get(c, 0) + 1
        seams = [{"between": [a, b], **dict(rec)}
                 for (a, b), rec in sorted(self._seams.items())]
        totals = {"kerb": 0, "bank": 0, "face": 0}
        for rec in self._seams.values():
            for k in totals:
                totals[k] += rec[k]
        return {"by": "ground.Contract.resolve", "columns": len(self._level),
                "declarations": decls, "owned_by_class": by_class,
                "seams": seams, "seam_totals": totals,
                "registered": {"PRECEDENCE": list(PRECEDENCE), "SEAM_KERB": SEAM_KERB,
                               "SEAM_BANK": SEAM_BANK, "RELIEF": self.relief}}

    # --- on disk, and across processes -----------------------------------------

    def __getstate__(self) -> dict:
        return _pack(self)

    def __setstate__(self, state: dict):
        got = _unpack(state)
        for k in ("_decls", "_owner", "_level", "_won", "_seams", "_clamped", "_decks",
                  "_by_label", "relief", "surface"):
            object.__setattr__(self, k, getattr(got, k))
        object.__setattr__(self, "_frozen", True)

    def save(self, path: str) -> str:
        """The resolution as one `.npz` a worker process reads back."""
        state = _pack(self)
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path


class Surface:
    """A heightmap fixed at resolution: `x0`, `z0` and the 2-D array `h`."""

    __slots__ = ("x0", "z0", "h")

    def __init__(self, x0: int, z0: int, h):
        self.x0, self.z0, self.h = int(x0), int(z0), np.asarray(h)


def load(path: str) -> Resolved:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as f:
        state = {k: f[k] for k in f.files if k != "meta"}
        state.update(json.loads(str(f["meta"])))
    return _unpack(state)


def _pack(r: Resolved) -> dict:
    cols = sorted(r._level)
    xs = np.array([c[0] for c in cols], np.int32)
    zs = np.array([c[1] for c in cols], np.int32)
    ys = np.array([r._level[c] for c in cols], np.int32)
    ow = np.array([r._owner[c] for c in cols], np.int32)
    decls = []
    for d in r._decls:
        decls.append({"label": d.label, "kind": d.kind, "cls": d.cls,
                      "level": d.level, "rect": list(d.rect) if d.rect else None,
                      "reason": d.reason, "decision": _jsonable(d.decision),
                      "columns": [[c[0], c[1], v] for c, v in d.columns.items()]})
    out = {"x": xs, "z": zs, "y": ys, "owner": ow, "decls": decls,
           "seams": [[a, b, dict(v)] for (a, b), v in r._seams.items()],
           "clamped": {str(i): dict(v) for i, v in r._clamped.items()},
           "decks": {str(i): v for i, v in r._decks.items()},
           "relief": r.relief, "surface": None}
    if r.surface is not None:
        out["surface"] = {"x0": r.surface.x0, "z0": r.surface.z0}
        out["surface_h"] = np.asarray(r.surface.h, np.int32)
    return out


def _unpack(state: dict) -> Resolved:
    decls = []
    for i, d in enumerate(state["decls"]):
        decls.append(Declaration(d["label"], d["kind"], d["cls"],
                                 {(c[0], c[1]): c[2] for c in d["columns"]},
                                 level=d["level"], rect=d["rect"], reason=d["reason"],
                                 decision=d.get("decision") or {}, index=i))
    xs, zs, ys, ow = (np.asarray(state[k]) for k in ("x", "z", "y", "owner"))
    owner = {(int(x), int(z)): int(o) for x, z, o in zip(xs, zs, ow)}
    level = {(int(x), int(z)): int(y) for x, z, y in zip(xs, zs, ys)}
    won: dict = {i: {} for i in range(len(decls))}
    for c, i in owner.items():
        won[i][c] = level[c]
    seams = {(a, b): dict(v) for a, b, v in state.get("seams", [])}
    clamped = {int(k): dict(v) for k, v in (state.get("clamped") or {}).items()}
    decks = {int(k): int(v) for k, v in (state.get("decks") or {}).items()}
    surface = None
    if state.get("surface") is not None and "surface_h" in state:
        surface = Surface(state["surface"]["x0"], state["surface"]["z0"],
                          np.asarray(state["surface_h"]))
    return Resolved(decls, owner, level, won, seams, clamped, decks,
                    relief=int(state.get("relief", RELIEF)), surface=surface)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ------------------------------------------------------------------ the ground as found

def bed_heights(vol) -> np.ndarray:
    """The topmost block a body stands on per column that is neither vegetation nor
    water: the bed, in world y -- what `Builder.bed()` sounds to, as an array."""
    from . import observe
    t = vol.tables()
    c = vol.codes
    occupies = (t["lower"][c].astype(bool) | t["upper"][c].astype(bool))
    veg = np.array([observe._is_vegetation(s) for s in vol.palette], bool)[c]
    firm = occupies & ~veg & ~t["liquid"][c].astype(bool)
    idx = firm.shape[1] - 1 - np.argmax(firm[:, ::-1, :], axis=1)
    return np.where(firm.any(axis=1), idx + vol.y0, vol.y0 - 1).astype(int)


def waterlines(vol) -> np.ndarray:
    """The topmost water block per column, in world y, or `y0 - 1` where dry."""
    t = vol.tables()
    c = vol.codes
    liquid = t["liquid"][c].astype(bool)
    idx = liquid.shape[1] - 1 - np.argmax(liquid[:, ::-1, :], axis=1)
    return np.where(liquid.any(axis=1), idx + vol.y0, vol.y0 - 1).astype(int)


class Found:
    """The ground as found over one volume: `bed(x, z)` and `wet(x, z)` off arrays
    read once, and the `surface` the build then holds to."""

    def __init__(self, vol):
        from . import offline
        self.x0, self.z0 = int(vol.x0), int(vol.z0)
        self._bed = bed_heights(vol)
        self._water = waterlines(vol)
        self.surface = Surface(vol.x0, vol.z0, offline.surface_heights(vol))
        self.y0 = int(vol.y0)

    def _ij(self, x, z):
        i, j = int(x) - self.x0, int(z) - self.z0
        if 0 <= i < self._bed.shape[0] and 0 <= j < self._bed.shape[1]:
            return i, j
        return None

    def bed(self, x: int, z: int) -> int | None:
        ij = self._ij(x, z)
        return None if ij is None else int(self._bed[ij])

    def wet(self, x: int, z: int) -> int | None:
        ij = self._ij(x, z)
        if ij is None:
            return None
        w = int(self._water[ij])
        return w if w > int(self._bed[ij]) else None


# ------------------------------------------------------------------ helpers

def _columns(columns) -> list:
    """(x, z) pairs from an iterable of pairs or a rectangle (x0, z0, x1, z1)."""
    if isinstance(columns, tuple) and len(columns) == 4 \
            and all(isinstance(v, (int, np.integer)) for v in columns):
        x0, z0, x1, z1 = (int(v) for v in columns)
        x0, x1 = min(x0, x1), max(x0, x1)
        z0, z1 = min(z0, z1), max(z0, z1)
        return [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
    return [(int(c[0]), int(c[1])) for c in columns]


def _rect_of(columns):
    if isinstance(columns, tuple) and len(columns) == 4 \
            and all(isinstance(v, (int, np.integer)) for v in columns):
        x0, z0, x1, z1 = (int(v) for v in columns)
        return (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
    cols = [(int(c[0]), int(c[1])) for c in columns]
    if not cols:
        return None
    return (min(c[0] for c in cols), min(c[1] for c in cols),
            max(c[0] for c in cols), max(c[1] for c in cols))


def ledge(rect, columns=None) -> list:
    """The ring one column outside a rectangle: where a pad's platform is laid a
    course wider than the pad, and where a person stands to open the door."""
    x0, z0, x1, z1 = rect
    out = [(x, z) for x in range(x0 - 1, x1 + 2) for z in (z0 - 1, z1 + 1)]
    out += [(x, z) for z in range(z0, z1 + 1) for x in (x0 - 1, x1 + 1)]
    return out
