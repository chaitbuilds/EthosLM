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


# ----------------------------------------------- proposed ground, owned by the design
# The design round's third contract. The expression round's held-out village failed its
# read on this and nothing could answer it: `stage_plateau` cut and paved **before** the
# plan, sized by `find_site._plateau_size` -- a default of 48 -- and faced in whatever
# voice the plan had at the time. A stone chapel ended up standing on twelve cottage
# footprints of black paving, `shrink_anchor` was tried and rolled back because the re-
# solve lost nine houses, and no owner re-sized or re-paved the ground. So the prepared
# ground is an artifact of the resolved design and has three states that never mix:
# baseline the terrain as it was observed. Immutable, `deps.baseline_path`. proposal
# what this design asks of it -- the anchor and its apron, the paving, the levels, the
# protected routes, and the occupied envelope of every part. Derived from the design, so
# a voice revision re-paves it and an anchor resize re-cuts it. applied the prepared
# volume, **always** made from the baseline. A revision regenerates the whole proposal
# and applies it again from the baseline, so cuts never accumulate. `evaluate` runs
# between the second and the third, against the arrangement, so a cut that no longer
# belongs to this design is caught rather than imposed.

#: **The apron a thing at the centre stands on**, as a share of its own footprint on
#: every side. Registered: a quarter of the anchor's own side, at least `APRON_MIN` and
#: at most `APRON_MAX` columns, which for a 24-column chapel is 6 columns of paving and
#: for a 72-column temple compound is 18. The default this replaces was 48 columns of
#: plateau whatever stood on it -- for the held-out chapel, twelve cottage footprints of
#: paving round one building.
APRON_SHARE = 0.25
APRON_MIN = 3
APRON_MAX = 18

#: How far outside a piece's own rectangle its feather reaches, for the conflict check:
#: `Builder.PLATEAU_FEATHER` is the least and the cut's own fall decides the rest, so
#: this is the bound a proposal is evaluated against rather than the exact ramp.
FEATHER_REACH = 4

_OCCUPIED_CACHE: dict = {}


def _type_occupied(decl: dict | None):
    """The `occupied()` a type file publishes, or None.

        A type that knows its own solid extent beyond the band the layout drew for it says
        so (`types/great_wall.occupied`), and planning, terrain, routing, emission and the
        checks all read that one answer rather than four guesses. Loaded from the file the
        declaration names, cached on its identity, so an edited type is re-read.
        
    """
    path = (decl or {}).get("path")
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns)
    except OSError:
        return None
    if key in _OCCUPIED_CACHE:
        return _OCCUPIED_CACHE[key]
    fn = None
    try:
        ns: dict = {"__name__": "__ethoslm_type__", "__file__": path}
        exec(compile(open(path).read(), path, "exec"), ns)      # noqa: S102
        got = ns.get("occupied")
        fn = got if callable(got) else None
    except Exception:                            # noqa: BLE001 -- a type that cannot say
        fn = None
    _OCCUPIED_CACHE[key] = fn
    return fn


def occupied_envelope(part: dict, decls: dict | None = None) -> dict:
    """**The ground one part actually fills, and the ground it needs kept free.**

        One envelope, four consumers: planning draws around it, the ground proposal cuts to
        it, routing keeps off it and the checks measure against it. The review's second
        cause: "The great wall's piers and batter extend beyond its nominal width.
        Increasing road clearance avoids one collision but does not establish a common
        occupied envelope for planning, terrain, circulation, ownership and checking."

        Returns `{"part", "type", "rects", "solid", "clearance", "envelope", "extends",
        "from"}` -- `rects` the band the layout drew, `solid` the same rectangles grown by
        whatever the type says projects past them, `envelope` the bounding rectangle of
        `solid` grown by the clearance, and `from` the sentence that says where each number
        came from.
        
    """
    from . import pipeline
    name = part.get("name")
    tname = str(part.get("type") or "")
    decl = (decls or {}).get(tname) or {}
    rects = [tuple(int(v) for v in r)
             for r in pipeline.part_rects({**part, "name": name})]
    clear = int(((decl.get("needs") or {}) or pipeline.NEEDS_DEFAULT).get(
        "clearance", pipeline.NEEDS_DEFAULT["clearance"]))
    src = [f"the band the layout drew: {len(rects)} rectangle(s)",
           f"`{tname or 'unknown type'}` keeps {clear} clear"]
    grow = 0
    extends = None
    fn = _type_occupied(decl)
    if fn is not None:
        try:
            extends = fn(part, **(part.get("params") or {}))
        except Exception:                        # noqa: BLE001 -- the band stands
            extends = None
    if isinstance(extends, dict):
        # the certain projections either side, and the one that hangs past a face the
        # siting has not chosen yet -- reserved on both sides, because a planner that
        # cannot wait for the siting has to reserve the ground it might take
        grow = max(int(extends.get("outer") or 0), int(extends.get("inner") or 0)) \
            + int(extends.get("either") or 0)
        clear = max(clear, int(extends.get("clearance") or 0))
        src.append(f"`{tname}` publishes its own occupation: {extends.get('why')}")
    solid = [(r[0] - grow, r[1] - grow, r[2] + grow, r[3] + grow) for r in rects]
    # **The ground reserved is the solid plus the clearance, rectangle by rectangle.** A
    # closed ring of wall is a polyline and its bounding box is the town inside it;
    # reserving the box would refuse every piece of ground the wall encloses. The
    # bounding box is still published as `envelope`, because a caller that wants one
    # number for "where is this part" should not have to compute it, and `reserved` is
    # what a conflict is decided against.
    reserved = [(r[0] - clear, r[1] - clear, r[2] + clear, r[3] + clear) for r in solid]
    env = (min(r[0] for r in reserved), min(r[1] for r in reserved),
           max(r[2] for r in reserved), max(r[3] for r in reserved))
    return {"part": name, "type": tname or None,
            "rects": [list(r) for r in rects], "solid": [list(r) for r in solid],
            "reserved": [list(r) for r in reserved],
            "projects": int(grow), "clearance": int(clear),
            "envelope": [int(v) for v in env], "extends": extends, "from": src}


def _anchor_of(spec: dict, place: dict) -> dict | None:
    """The part the ground is prepared for: the compound or leaf at the centre."""
    core = None
    for p in (spec or {}).get("defining_parts") or []:
        if p.get("relation") == "centre":
            core = p
            break
    if core is None:
        return None
    for c in (place or {}).get("compounds") or []:
        if c.get("defines") == core["name"] or c.get("name") == core["name"]:
            return {**c, "kind": "area", "_part": core}
    for p in (place or {}).get("parts") or []:
        if p.get("defines") == core["name"] or p.get("name") == core["name"]:
            return {**p, "_part": core}
    return None


def apron_for(rect, *, share: float = APRON_SHARE) -> tuple:
    """`(rect grown by its apron, the apron's width, why)` -- the paved ground round a
    thing at the centre, from the thing's own size and not from a default."""
    x0, z0, x1, z1 = (int(v) for v in rect)
    side = max(x1 - x0 + 1, z1 - z0 + 1)
    a = int(max(APRON_MIN, min(APRON_MAX, round(side * float(share)))))
    return ((x0 - a, z0 - a, x1 + a, z1 + a), a,
            f"an apron of {a} columns on every side: {share:g} of the anchor's own "
            f"{side}-column side, held between {APRON_MIN} and {APRON_MAX}")


def propose(spec: dict, site: dict, place: dict, baseline, *,
            allocation: dict | None = None, decls: dict | None = None,
            anchor_level: int | None = None) -> dict:
    """**What this design asks of the ground**, derived from the design and nothing else.

        `baseline` is the observed terrain -- a path (`deps.baseline_path`) or a loaded
        volume -- and it is recorded, never modified. The proposal owns:

          * the **anchor and its apron**, sized from the anchor the layout actually drew
            (`apron_for`), so an anchor that shrank re-cuts and one that grew re-cuts;
          * the **paving**, in the voice the resolved design gives that ground, so a voice
            revision re-paves it;
          * the **levels**: the layout's terrace record, ring by ring;
          * the **protected routes**: the arterial's cells and every lane the circulation
            reserved, which no piece may bury;
          * the **occupied envelope** of every part (`occupied_envelope`), published once
            for planning, ground, routing, emission and the checks.

        Returns the proposal. Nothing here touches the world; `apply` does, from the
        baseline.
        
    """
    from . import contracts
    lay = (place or {}).get("layout") or {}
    if decls is None:
        try:
            from .placeplan import types_card
            _t, decls = types_card()
        except Exception:                        # noqa: BLE001 -- no types on disk
            decls = {}
    base = _baseline_id(baseline)
    X, Z = int(site["origin"][0]), int(site["origin"][1])
    S = int(site["size"])
    pieces, src = [], []
    terrace = lay.get("terrace") or {}
    place_voice = place.get("voice")
    anchor = _anchor_of(spec, place)
    if anchor is not None and anchor.get("x1") is not None:
        rect = (int(min(anchor["x0"], anchor["x1"])), int(min(anchor["z0"], anchor["z1"])),
                int(max(anchor["x0"], anchor["x1"])), int(max(anchor["z0"], anchor["z1"])))
        grown, apron, why = apron_for(rect)
        grown = (max(X, grown[0]), max(Z, grown[1]),
                 min(X + S - 1, grown[2]), min(Z + S - 1, grown[3]))
        # **The level the anchor's ground stands at.** A concentric place has a podium
        # in its terrace record; a village gathered about a square has no terrace at all
        # and its centre is still designed ground that has to be level. `anchor_level`
        # is what the caller settled it at -- the stage reads it off the plateau record
        # -- and a proposal with no level cuts nothing, which is the honest answer for a
        # place whose centre the design asks nothing of.
        level = terrace.get("podium")
        if level is None and anchor_level is not None:
            level = int(anchor_level)
        if level is None and anchor.get("level") is not None:
            level = int(anchor["level"])
        voice = anchor.get("voice") or place_voice
        pieces.append({
            "label": anchor.get("defines") or anchor.get("name"),
            "part": anchor.get("name"), "what": "podium",
            "rect": [int(v) for v in grown], "apron": int(apron),
            "own_rect": [int(v) for v in rect],
            "level": None if level is None else int(level), "voice": voice,
            "from": [f"the anchor `{anchor.get('name')}` as the layout drew it: "
                     f"{rect[2] - rect[0] + 1}x{rect[3] - rect[1] + 1}", why,
                     f"paved in `{voice}`, the voice this design gives this ground"
                     if voice else "no voice on this ground",
                     (f"levelled to y={level}"
                      + (" (the layout's podium)" if terrace.get("podium") is not None
                         else " (the level this round settled the anchor's ground at)")
                      ) if level is not None else "no level: the ground as found"]})
        src.append(f"the anchor's ground is {grown[2] - grown[0] + 1} columns a side, "
                   f"from the anchor and its apron and not from a default")
    for k, r in enumerate(lay.get("rings") or []):
        if r.get("level") is None:
            continue
        pieces.append({
            "label": f"terrace/{r['name']}", "part": r.get("name"), "what": "terrace",
            "ring": int(r.get("ring") or k), "inner": r.get("inner"),
            "outer": r.get("outer"), "rect": None,
            "level": int(r["level"]), "voice": r.get("voice") or place_voice,
            "from": [f"ring {r.get('ring')} of the resolved design stands at "
                     f"y={r['level']}",
                     f"faced in `{r.get('voice') or place_voice}`"]})
    protected = _protected(place)
    occupied = {}
    for p in (place or {}).get("parts") or []:
        if not p.get("name"):
            continue
        occupied[p["name"]] = occupied_envelope(p, decls)
    out = {"record": "ground_proposal", "version": 1, "by": "ground.propose",
           "baseline": base, "site": {"origin": [X, Z], "size": S},
           "voice": place_voice, "levels": dict(terrace) or None,
           "pieces": pieces, "protected": protected, "occupied": occupied,
           "allocation": dict(allocation or {}),
           "print": None, "from": src or ["this design asks nothing of the ground"]}
    out["print"] = contracts.digest(
        base.get("print"), [[p.get("label"), p.get("rect"), p.get("level"),
                             p.get("voice")] for p in pieces],
        sorted((k, tuple(v["envelope"])) for k, v in occupied.items()))
    return out


def _baseline_id(baseline) -> dict:
    """What the ground was before this design touched it, as an identity."""
    from . import contracts
    if isinstance(baseline, str):
        try:
            from .deps import content_print
            return {"path": baseline, "print": content_print(baseline)}
        except Exception:                        # noqa: BLE001 -- the path is the id
            return {"path": baseline, "print": None}
    if baseline is None:
        return {"path": None, "print": None}
    return {"path": getattr(baseline, "path", None),
            "print": contracts.digest(int(getattr(baseline, "x0", 0)),
                                      int(getattr(baseline, "z0", 0)),
                                      int(getattr(baseline, "y0", 0)))}


def _protected(place: dict) -> list:
    """The ground no piece of this proposal may bury: the arterial's cells and the
    lanes and doorsteps the circulation reserved."""
    out = []
    cells = ((place or {}).get("arterials") or {}).get("cells") or []
    if cells:
        out.append({"what": "arterial", "cells": [[int(c[0]), int(c[1])] for c in cells],
                    "why": "the road the place is reached by: a cut across it is a "
                           "place nobody can walk into"})
    for r in ((place or {}).get("arterials") or {}).get("routes") or []:
        if r.get("cells"):
            out.append({"what": "route", "name": r.get("name"),
                        "cells": [[int(c[0]), int(c[1])] for c in r["cells"]],
                        "why": "a route the design reserved"})
    return out


def evaluate(proposal: dict, place: dict, baseline=None) -> dict:
    """**Is this proposal still the ground this arrangement needs?** Run before `apply`.

        Four questions, and every one of them was a way the expression round's ground went
        wrong:

          * is the proposal of **this** baseline? A cut made from another terrain is an
            obsolete cut and cannot be taken back once it is made;
          * does every piece lie inside the site?
          * does the anchor's piece actually hold the anchor the arrangement drew -- and is
            it no larger than the anchor plus its apron? A plateau of twelve cottage
            footprints round one chapel is this question answered `no`;
          * does any piece bury a protected route, or cut across another part's occupied
            envelope at a level that part was not designed for?

        `{"ok", "why", "conflicts"}`. `why` is the list of sentences; `conflicts` the
        pieces and subjects that disagree.
        
    """
    why, conflicts, around = [], [], []
    site = proposal.get("site") or {}
    X, Z = int(site.get("origin", [0, 0])[0]), int(site.get("origin", [0, 0])[1])
    S = int(site.get("size") or 0)
    if baseline is not None:
        now = _baseline_id(baseline)
        was = proposal.get("baseline") or {}
        if was.get("print") and now.get("print") and was["print"] != now["print"]:
            conflicts.append({"what": "baseline", "was": was.get("path"),
                              "now": now.get("path")})
            why.append("this proposal was made from a different terrain than the one "
                       "in hand: ground work is not idempotent, so it is regenerated "
                       "from the baseline rather than applied")
    occupied = proposal.get("occupied") or {}
    for piece in proposal.get("pieces") or []:
        rect = piece.get("rect")
        if not rect:
            continue
        x0, z0, x1, z1 = (int(v) for v in rect)
        if S and not (X <= x0 and x1 < X + S and Z <= z0 and z1 < Z + S):
            conflicts.append({"what": "site", "piece": piece.get("label"),
                              "rect": [x0, z0, x1, z1]})
            why.append(f"`{piece.get('label')}` reaches x {x0}..{x1}, z {z0}..{z1} and "
                       f"the site is x {X}..{X + S - 1}, z {Z}..{Z + S - 1}")
        if piece.get("what") == "podium":
            own = piece.get("own_rect")
            here = _anchor_rect(place, piece)
            if here is not None and own is not None and list(here) != list(own):
                conflicts.append({"what": "anchor", "piece": piece.get("label"),
                                  "proposed_for": list(own), "arrangement": list(here)})
                why.append(f"`{piece.get('label')}` was cut for an anchor of "
                           f"{own[2] - own[0] + 1}x{own[3] - own[1] + 1} and the "
                           f"arrangement draws it "
                           f"{here[2] - here[0] + 1}x{here[3] - here[1] + 1}: the "
                           f"ground belongs to a design that has been revised")
            check = here if here is not None else own
            if check is not None:
                grown, apron, _w = apron_for(check)
                room = (x1 - x0 + 1) * (z1 - z0 + 1)
                want = (min(grown[2], X + S - 1) - max(grown[0], X) + 1) \
                    * (min(grown[3], Z + S - 1) - max(grown[1], Z) + 1)
                if room > want * 1.2:
                    conflicts.append({"what": "apron", "piece": piece.get("label"),
                                      "columns": int(room), "wants": int(want)})
                    why.append(f"`{piece.get('label')}` covers {room} columns and the "
                               f"anchor and its {apron}-column apron want {want}: the "
                               f"rest is paving nothing asked for")
        # **A route the piece meets is laid around; a route it must bury is refused.**
        # The point of this check is that a cut cannot be taken back -- so what it has
        # to stop is ground that *destroys* a route, and what it must not stop is a
        # podium whose apron reaches a road it can simply not pave. `apply` leaves the
        # protected columns as it found them (`_ProtectedRoutes`), so the question here
        # is the narrower one: does the thing the piece exists for stand on the route?
        # For a podium that is the anchor's **own** rectangle and not its apron; for a
        # piece with no own rectangle it is the piece itself, because there is nothing
        # of it left once the route is taken out.
        own = piece.get("own_rect") or piece.get("rect")
        ox0, oz0, ox1, oz1 = (int(v) for v in own)
        for pro in proposal.get("protected") or []:
            cells = pro.get("cells") or []
            hit = [c for c in cells
                   if x0 <= int(c[0]) <= x1 and z0 <= int(c[1]) <= z1]
            if not hit:
                continue
            must = [c for c in hit if ox0 <= int(c[0]) <= ox1 and oz0 <= int(c[1]) <= oz1]
            if must:
                conflicts.append({"what": "protected", "piece": piece.get("label"),
                                  "route": pro.get("what"), "cells": len(must),
                                  "around": len(hit) - len(must)})
                why.append(f"`{piece.get('label')}` stands on {len(must)} cell(s) of "
                           f"the {pro.get('what')} it may not bury: the part itself is "
                           f"drawn over the route, so the ground it needs cannot be "
                           f"laid around it")
            else:
                around.append({"piece": piece.get("label"), "route": pro.get("what"),
                               "cells": len(hit),
                               "why": (f"`{piece.get('label')}`'s apron reaches "
                                       f"{len(hit)} cell(s) of the {pro.get('what')}; "
                                       f"the ground is laid around them and they are "
                                       f"left as found")})
        # **The solid a part fills, not the room it keeps clear.** A clearance band is
        # ground by definition -- it is the room round a building that nothing may be
        # built in -- and paving it at the level the building stands on is what a piece
        # of designed ground is *for*. The farm's hall stands `on the square` because
        # the sentence says so; testing the square's apron against the hall's reserve
        # refused the one arrangement the request asks for, over a two-column sliver
        # that the same record reported as `projects: 0`. What is a conflict is a piece
        # crossing a part's **solid** extent, or levelling the ground a part stands on
        # to a level it was not sited at.
        for name, env in occupied.items():
            if name == piece.get("part"):
                continue
            solid = [e for e in (env.get("solid") or [env.get("envelope")]) if e
                     and x0 <= e[2] and e[0] <= x1 and z0 <= e[3] and e[1] <= z1]
            if not solid:
                continue
            at = env.get("level")
            if at is not None and piece.get("level") is not None \
                    and int(at) == int(piece["level"]):
                continue                      # the same ground, at the level it stands at
            conflicts.append({"what": "occupation", "piece": piece.get("label"),
                              "part": name, "solid": list(solid[0]),
                              "rectangles": len(solid),
                              "part_level": at, "piece_level": piece.get("level")})
            why.append(f"`{piece.get('label')}` cuts across the solid `{name}` fills "
                       f"({env.get('projects')} column(s) of it beyond the band the "
                       f"layout drew, over {len(solid)} rectangle(s) of its run)"
                       + (f", and levels it to y={piece.get('level')} where `{name}` "
                          f"is sited at y={at}" if at is not None else ""))
    return {"ok": not conflicts,
            "why": why or ["the proposal is of this baseline, inside the site, cut for "
                           "the anchor the arrangement draws, and clear of every "
                           "protected route and occupied envelope"]
            + [a["why"] for a in around],
            "conflicts": conflicts,
            # the routes a piece reaches and lays around: not a conflict, and not
            # silence either -- `apply` reports how many columns it actually left
            "around": around}


def _anchor_rect(place: dict, piece: dict):
    """The rectangle the arrangement currently draws for this piece's part."""
    for key in ("compounds", "parts", "districts"):
        for r in (place or {}).get(key) or []:
            if r.get("name") == piece.get("part") and r.get("x1") is not None:
                return (int(min(r["x0"], r["x1"])), int(min(r["z0"], r["z1"])),
                        int(max(r["x0"], r["x1"])), int(max(r["z0"], r["z1"])))
    return None


class _ProtectedRoutes:
    """The proposal's protected routes in the shape `Builder._site_lanes` reads.

        `Builder.plateau` already leaves the circulation pass's columns alone and counts
        them (`lane_columns_left_alone`) -- it asks `self.frontage.net` for them. A ground
        proposal's protected routes are the same kind of thing and want the same treatment,
        so they are handed over through the same seam rather than through a second one.
        `surface()` is empty because these are cells to keep off and not a network to
        arrive from; `_plateau_landing` falls back to `cells` for the way on.
        
    """

    __slots__ = ("cells", "thresholds")

    def __init__(self, cells):
        self.cells = {(int(c[0]), int(c[1])) for c in cells}
        self.thresholds = ()

    def surface(self):
        """Empty: these are columns to keep off, not a network to arrive from. The
        plateau's own way-on then falls back to the open ground round it, and
        `approach` records that it had nothing to lay rather than building a stair to
        a road this proposal is not the owner of."""
        return []

    def threshold(self, label):
        return None


def _protected_cells(proposal: dict) -> set:
    out = set()
    for pro in proposal.get("protected") or []:
        for c in pro.get("cells") or []:
            out.add((int(c[0]), int(c[1])))
    return out


def apply(proposal: dict, baseline_volume):
    """**The prepared ground, always made from the baseline.**

        A revision regenerates the whole proposal and applies it again from the terrain as
        it was observed, so cuts never accumulate: the second cut of a design that changed
        its mind is the first cut of the design it changed to. Returns the prepared volume
        and records what it laid on `proposal["applied"]`.

        The caller keeps the baseline. Nothing here writes to the world.
        
    """
    from .buildlib import Builder
    from . import offline
    vol = baseline_volume
    b = Builder(offline.OfflineSite(vol))
    b._vol = vol
    # **Designed ground is laid around a protected route, not over it.** The route is
    # given to the builder through the seam it already reads (`Builder._site_lanes` ->
    # `frontage.net`), so every piece of this proposal leaves those columns as it found
    # them and says how many it left.
    protected = _protected_cells(proposal)
    if protected:
        from .frontage import Frontage
        b.frontage = Frontage(vol, _ProtectedRoutes(protected))
    laid, refused = [], []
    for piece in proposal.get("pieces") or []:
        rect = piece.get("rect")
        if not rect:
            refused.append({"piece": piece.get("label"),
                            "why": "no rectangle: a terrace is laid by its ring and not "
                                   "by this call"})
            continue
        mat = None
        try:
            from . import pipeline as _pipeline
            mat = _pipeline.voice_palette(piece.get("voice") or proposal.get("voice")
                                          or None)
        except Exception:                        # noqa: BLE001 -- the default palette
            mat = None
        x0, z0, x1, z1 = (int(v) for v in rect)
        # **A level the design did not fix is taken from the baseline**, not from a
        # default: `Builder.plateau(y=None)` cuts to the median of the ground it is
        # standing on, which for a place with no terraces is the level its own site
        # gives it.
        level = piece.get("level")
        rec = b.plateau((x0, z0, x1, z1), None if level is None else int(level), mat=mat,
                        label=str(piece.get("label")),
                        bound=Builder.plateau_max(int((proposal.get("site") or {})
                                                      .get("size") or 0) or None))
        (laid if rec.get("ok") else refused).append(
            {"piece": piece.get("label"), "rect": [x0, z0, x1, z1],
             "level": rec.get("y") if level is None else int(level),
             "level_from": ("the design's terrace" if level is not None
                            else "the baseline's own median under this ground"),
             "voice": piece.get("voice"),
             **({k: rec[k] for k in ("columns", "filled", "cut", "reason",
                                     "lane_columns_left_alone") if k in rec}
                if rec.get("ok") else {"why": rec.get("reason")})})
    cut = dict(b._pending)
    proposal["applied"] = {
        "from_baseline": (proposal.get("baseline") or {}).get("path"),
        "laid": laid, "refused": refused, "blocks": len(cut),
        "protected_columns": len(protected),
        "protected_left_alone": sum(int(x.get("lane_columns_left_alone") or 0)
                                    for x in laid),
        "why": "the prepared ground is made from the baseline every time: a revision "
               "regenerates the proposal and applies it again, so two cuts of two "
               "designs never add up"}
    return vol.overlay(cut) if cut else vol


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
