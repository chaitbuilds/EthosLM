"""**What a part is required to deliver, resolved before its lot is drawn.**

The design round's first contract, and the defect it closes, measured on the imported
code through the real entry point:

    envelope.lot_for("cottage", {"storeys": 2})                        -> lot_min [5, 5]
    envelope.lot_for("cottage", {"storeys": 2}, features=("storeys",)) -> lot_min [15, 17]

`placesolve.fabric_lot` and `district_compile._storeys_fit` both asked the first form.
With an empty feature set `envelope._stands` checks no requested feature at all, so a
standing one-storey shell answers a two-storey query and a required second storey costs
the layout nothing: a 5x5 lot is a correct answer to a question nobody asked.

The question was never only "how big is this type". It is **what does this part owe, and
to which requirement**, and that has five parts that were resolved in five different
places at three different times:

    the approved pool     which types the capability record admitted for this fabric,
                          in its order -- not the general library, which is what
                          `fabric_lot` fell back to while the record said otherwise
    the parameters        the storeys band the character and the type agree on, and the
                          floor of it, which is what every building of the part must
                          reach -- with a parameter a requirement fixes outright
                          (`of two storeys`) over the top of it, because a band is an
                          inference about variation and an exact clause is not
    required features     the tokens a requirement makes mandatory **of this part**,
                          with the requirement ids behind them and the spatial scope
    optional features     the tokens the type declares it *may* deliver. An optional
                          band may be revised down; a required floor may not, and the
                          difference between those two sentences is the whole of what
                          separates the farm case from its control
    the physical context  the voice, the ground, the clearance, whether the house is
                          attached, which way it fronts and at which seed -- what the
                          answer is an answer *about*

    resolve(spec, intent, part, decls, ...)  -> the demand
    lot(d, cache=...)                        -> the lot it needs, or an honest refusal
    storeys_admitted(d, w, depth, ...)       -> what a lot of this size admits
    validate_lot(d, lot, ...)                -> may an explicit lot width be trusted
    relax(d)                                 -> the same demand one storey lower, where
                                                nothing requires the floor

**A refusal is never a smaller default.** Where nothing in the pool delivers a required
feature, `lot` answers `lot_min: None` with `refused` set and `binding: True`, and the
caller propagates it. Where only an *inferred* choice cannot be delivered the answer is
still a refusal, with `binding: False`: the caller may `relax` the demand and ask again,
which is a decision on the record rather than a default substituted underneath one.

**Which requirement reaches which part is `capability.requirements_for`'s answer** and
not `envelope.required_by`'s. The distinction has a measurement behind it: this module
first asked `envelope.required_by`, which filters on `scope` alone, and the farm
sentence's `feature/market` carries no scope while its feature word maps to `stalls` --
so every cottage, every field and the hall were asked to deliver market stalls, every
one of them refused honestly, and the refusal was for a capability gap that does not
exist. `envelope.required_by` is right for the question it answers (which tokens does
this set of requirements name); it is the wrong question to ask **of a part**. See
`_asks`.
"""
from __future__ import annotations

from . import envelope

#: The marker token, re-exported so a caller needs one import.
STOREYS = envelope.STOREYS

#: The context a demand carries and an envelope answer is keyed on. `ground` and
#: `clearance` are what the probe was run against; today a probe builds on flat ground
#: with free flanks and says so, and the day one can be run on a slope the certificates
#: measured on a plane will not be mistaken for it.
CONTEXT_DEFAULT = {"voice": None, "ground": "flat", "clearance": None,
                   "attached": None, "frontage": None, "seed": 1}

#: Parameters a demand names on behalf of the whole fabric. A demand is about a *part*
#: -- sixteen cottages -- and not about one building, so the parameters it fixes are the
#: ones a requirement or the district's character fixes for all of them. Everything else
#: the district compiler still varies per lot under its own seed.
FABRIC_PARAMS = ("storeys",)


# ------------------------------------------------------------------ the demand

def resolve(spec: dict | None, intent: dict | None, part: dict | None,
            decls: dict | None, *, allocation: dict | None = None,
            voice: str | None = None, context: dict | None = None) -> dict:
    """The demand a defining part makes, before anything is sized.

        `part` is the defining part or the district row that defines it; `decls` the type
        declarations the run is working from (`placeplan.types_card`). `context` carries the
        physical context and, under `capabilities`, the capability record -- which is the
        authority on the approved pool where the part does not already carry one.
        
    """
    part = dict(part or {})
    spec = dict(spec or {})
    ctx_in = dict(context or {})
    reqs = list((intent or {}).get("requirements") or [])
    asks = _asks(spec, intent, part)
    by_token = {t: list(asks["why"].get(t) or ()) for t in asks["required"]}
    types, pool_from = _pool(part, spec, decls, ctx_in.get("capabilities"), allocation)
    band, band_from = _storeys_band(part, decls, types, allocation,
                                    fixed=asks["params"])
    params = _params(part, band, allocation, fixed=asks["params"])
    # **Required is what a requirement says; optional is what the type offers besides.**
    # Every token a requirement named is required, whether or not a type declares it --
    # a token nothing can deliver is a capability gap and a refusal, never a quiet
    # subtraction from the ask. `unsupported` is the cheap half of that answer: a type
    # that declares `FEATURES` has stated its whole vocabulary, so a token outside every
    # approved type's declared vocabulary is a gap that needs no probe to establish. A
    # type declaring none has stated nothing and is probed like any other.
    offered, stated = set(), []
    for t in types:
        got = set(envelope.declared_features(t))
        offered |= got
        stated.append(bool(got))
    required = tuple(by_token)
    unsupported = tuple(t for t in required
                        if t != STOREYS and t not in offered
                        and stated and all(stated))
    optional = tuple(sorted(offered - set(required)))
    if band and STOREYS not in required and band[0] > 1:
        # the density word's own band, or the type's: an inference, and revisable
        optional = tuple(sorted(set(optional) | {STOREYS}))
    scope = next((r.get("scope") for r in reqs
                  if r.get("id") in set(asks["ids"]) and r.get("scope")), None)
    ctx = _context(part, spec, voice, ctx_in)
    return {"part": part.get("name") or part.get("defines"),
            "defines": part.get("defines") or part.get("name"),
            "types": list(types), "pool_from": pool_from,
            "params": dict(params), "storeys_band": list(band) if band else None,
            "band_from": band_from,
            "required": tuple(required), "optional": tuple(optional),
            "unsupported": tuple(unsupported),
            "requirements": list(asks["ids"]),
            "required_by": {k: list(v) for k, v in by_token.items()},
            "fixed": dict(asks["params"]), "fixed_by": {k: list(v) for k, v in
                                                        asks["why"].items()
                                                        if k in asks["params"]},
            "scope": scope, "context": ctx,
            "why": (f"{part.get('name') or 'the place'} is built from "
                    f"{types or 'no approved type'} ({pool_from}); "
                    f"{_norm_params(params)}; required {list(required) or 'nothing'}"
                    + (f" by {[i for ids in by_token.values() for i in ids]}"
                       if by_token else "")
                    + (f"; optional {list(optional)}" if optional else "")
                    + (f"; {list(unsupported)} is asked for and no approved type "
                       f"declares it" if unsupported else ""))}


# ------------------------------------------------------ the binding, for the evidence

#: The token normaliser, re-exported so an evidence layer needs one import. `market` is
#: `stalls`; see `envelope.feature_token` for the defect that made it necessary.
feature_token = envelope.feature_token


def required_by_part(spec: dict | None, intent: dict | None, place: dict | None = None, *,
                     parts=None, declares=None) -> dict:
    """**The required feature tokens of every part of a resolved place**, by part name.

            {part_name: {token: [requirement_id, ...]}}

        The binding the evidence layers were re-deriving from the raw intent record. It exists
        because nothing exposed it: `capability.requirements_for` answers this for a *defining
        part* and `demand.resolve` carries it as `required_by`, and both are consumed inside
        the sizing path and thrown away, so `construction.confirm`, `intent._function_measure`
        and `pipeline/improve` each rebuilt a worse version out of the sentence's own words.

        Two kinds of key in the answer, and they mean different things:

        `place` is the resolved place (`plan.place.json` or the checked spec); `parts` are the
        leaves, defaulting to every leaf of `place`. `declares` overrides the type-vocabulary
        lookup (`{type: (token, ...)}`) so a probe or a fixture is not made to execute the
        library. A part with nothing required is absent from the answer rather than present
        and empty: "this part owes nothing" is the ordinary case and does not need a row.
        
    """
    from . import capability
    doc = dict(place or spec or {})
    reqs = [r for r in (intent or {}).get("requirements") or []
            if isinstance(r, dict) and r.get("status") != "unsupported"]
    out: dict = {}

    def add(name, token, rid):
        if not name or not token:
            return
        got = out.setdefault(str(name), {}).setdefault(str(token), [])
        if rid and rid not in got:
            got.append(rid)

    defining = list(doc.get("defining_parts") or doc.get("parts") or [])
    for p in defining:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        try:
            got = capability.requirements_for(spec or doc, intent, p)
        except Exception:                        # noqa: BLE001 -- reported as no binding
            continue
        for t in got.get("required") or ():
            for rid in (got.get("why") or {}).get(t) or [None]:
                add(p["name"], t, rid)

    leaves = list(parts) if parts is not None else _leaves(doc)
    vocab = dict(declares or {})
    for leaf in leaves:
        if not isinstance(leaf, dict) or not (leaf.get("name") or leaf.get("part")):
            continue
        name = leaf.get("name") or leaf.get("part")
        tname = str(leaf.get("type") or "")
        can = _declared(tname, vocab)
        claimed = set((((leaf.get("emitted") or {}) if isinstance(leaf.get("emitted"), dict)
                        else {}).get("features") or {}))
        for r in reqs:
            for t, ids in envelope.required_by([r], _scope_row(leaf)).items():
                # the leaf owes the token only if it is the thing that answers it
                if t == STOREYS:
                    if leaf.get("kind", "plot") == "plot":
                        for rid in ids or [None]:
                            add(name, t, rid)
                    continue
                if t in can or t in claimed:
                    for rid in ids or [None]:
                        add(name, t, rid)
    return out


def _scope_row(leaf: dict) -> dict:
    """A leaf in the shape `intent._in_scope` and `envelope.required_by` read."""
    return {"name": leaf.get("name") or leaf.get("part"),
            "defines": leaf.get("defines") or leaf.get("name") or leaf.get("part"),
            "answers": leaf.get("answers"),
            "in": list(leaf.get("in") or ())}


def _declared(tname: str, vocab: dict) -> set:
    if not tname:
        return set()
    if tname in vocab:
        return {str(t) for t in vocab[tname] or ()}
    return {str(t) for t in envelope.declared_features(tname)}


def _leaves(doc: dict) -> list:
    from . import pipeline as _pipeline
    try:
        return list(_pipeline.plan_parts(doc))
    except Exception:                            # noqa: BLE001 -- a spec has no leaves
        return list(doc.get("parts") or [])


def required_for(binding: dict, part) -> tuple:
    """`(token, ...)` -- what one part owes, out of `required_by_part`'s answer.

        The shape `construction.confirm(..., required=...)` takes per part, in token order.
        
    """
    name = str(part if not isinstance(part, dict)
               else (part.get("part") or part.get("name")))
    return tuple((binding or {}).get(name) or ())


def _asks(spec: dict, intent: dict | None, part: dict) -> dict:
    """Which clauses of the sentence **this part** has to answer.

        `capability.requirements_for` (the coordinator's), and not
        `envelope.required_by`, which filters only on `scope`. The difference is a measured
        defect, found by the coordinator on the `des-farm` records: `feature/market` carries
        no scope and its feature word maps to `stalls`, so every cottage, every field and
        the hall of a farming village were being asked to emit market stalls -- and
        `demand.lot` then refused all four honestly, for a capability gap that does not
        exist. `envelope.required_by` is right for what it does (which tokens does this set
        of requirements name) and is the wrong question to ask **of a part**.

        Returns `{"ids", "required", "params", "why"}`. `optional` is not taken from there:
        what a type may choose is the type's business and is read off its `FEATURES` below.
        
    """
    from . import capability
    fn = getattr(capability, "requirements_for", None)
    if fn is None:
        raise RuntimeError(
            "ethoslm.capability.requirements_for is the binding from a requirement to a "
            "part and it is not in this build. `demand.resolve` will not fall back to "
            "`envelope.required_by`: that rule filters on scope alone and asks every "
            "part of a place for every unscoped feature, which is the defect this "
            "function exists to close")
    got = fn(spec, intent, part)
    return {"ids": list(got.get("ids") or ()),
            "required": tuple(got.get("required") or ()),
            "params": dict(got.get("params") or {}),
            "why": dict(got.get("why") or {})}


def _pool(part: dict, spec: dict, decls: dict | None, caps: dict | None,
          allocation: dict | None) -> tuple:
    """`([type, ...], where it came from)` -- the approved pool, in its order.

        The part's own `fabric_types` first, because that is where `placesolve._write_fabric`
        puts the capability record's approval; then the record itself, for a caller asking
        before the place has been written; the general library **last and labelled**, because
        the review's first finding is that `fabric_lot` estimated a parent from a type the
        record had not approved, and a pool nothing approved should be visible rather than
        indistinguishable.
        
    """
    got = [str(t) for t in (part.get("fabric_types") or []) if t]
    if got:
        return (got, "the part's approved fabric pool")
    name, defines = part.get("name"), part.get("defines") or part.get("name")
    for e in (caps or {}).get("entries") or []:
        w = e.get("wants") or {}
        if not e.get("matched") or w.get("of") != "fabric":
            continue
        if w.get("part") not in (name, defines):
            continue
        pool = [t for t in [e.get("type"), *(e.get("alternatives") or [])] if t]
        if pool:
            return (pool, "the capability record's approved fabric")
    over = (allocation or {}).get("types") or {}
    named = over.get(name) or over.get(defines)
    if named:
        return ([str(t) for t in named], "the allocation's named types")
    from . import district_compile as dc
    from . import placeplan as _placeplan
    role = part.get("role") or _placeplan.DENSITY_ROLE.get(part.get("density") or "medium")
    names = [n for n, _d in dc.house_types(decls or {}, role, spec.get("form"))]
    return (names, "the general library: no capability record approved a pool for this "
                   "part, and the pool is therefore not an approval")


def _declared_band(decls: dict | None, tname: str) -> tuple | None:
    spec_p = (((decls or {}).get(tname) or {}).get("params") or {}).get("storeys")
    if isinstance(spec_p, (list, tuple)) and len(spec_p) >= 3 and spec_p[0] == "int":
        return (int(spec_p[1]), int(spec_p[2]))
    return None


def _storeys_band(part: dict, decls: dict | None, types, allocation,
                  fixed: dict | None = None) -> tuple:
    """`([lo, hi], where it came from)` -- the storeys band this part is built to.

        The character's band clamped into the band the leading approved type declares, which
        is `district_compile._storeys_band`'s rule asked one level up, where the lot has not
        been drawn yet. None where no approved type has a storeys parameter at all.

        **A requirement that fixes the storeys outright is not a band.** "of two storeys" is
        exact and hard, and the character's `[1, 2]` under it is an inference about how the
        district varies, not a licence for half of it to stand one storey. Where a
        requirement fixes the number the band is that number, and `band_from` says which
        requirement made it so. A number the approved type cannot build is left where it is
        and refused by the envelope, which is the honest answer and not a smaller one.
        
    """
    ch = part.get("character") or {}
    over = ((allocation or {}).get("characters") or {}).get(part.get("name")) or {}
    want = over.get("storeys") or ch.get("storeys")
    declared = next((b for b in (_declared_band(decls, t) for t in types) if b), None)
    pin = (fixed or {}).get(STOREYS)
    if isinstance(pin, int):
        return ([int(pin), int(pin)],
                f"a requirement fixes the storeys at {pin}"
                + (f"; the character asked {list(want)} and the type declares "
                   f"{list(declared)}" if want and declared else
                   (f"; the type declares {list(declared)}" if declared else "")))
    if declared is None:
        return (None, "no approved type declares a storeys parameter")
    lo, hi = declared
    if want and len(want) >= 2:
        w_lo, w_hi = int(want[0]), int(want[1])
        if max(lo, w_lo) <= min(hi, w_hi):
            return ([max(lo, w_lo), min(hi, w_hi)],
                    f"the character's {[w_lo, w_hi]} inside the type's {[lo, hi]}")
        # the two bands do not meet: the type's own is what it can build, and the
        # character's ask is not quietly widened to cover it
        return ([lo, hi], f"the character asks {[w_lo, w_hi]} and the type declares "
                          f"{[lo, hi]}; they do not meet and the type's band is what "
                          f"can be built")
    return ([lo, hi], "the type's own declared band; the character names none")


def _params(part: dict, band, allocation: dict | None,
            fixed: dict | None = None) -> dict:
    """The parameters the demand fixes for the whole fabric: the band's **floor**, and
        over it whatever a requirement fixes outright.

        The floor and not the middle: every building of this part has to reach it, and a lot
        sized for the average is a lot half the part cannot use. **A requirement-fixed
        parameter wins over a band**: the farm's fields district carries a `[1, 2]` character
        band and holds counted cottages of the sixteen the sentence asked for, and those
        cottages still stand two storeys.
        
    """
    out: dict = {}
    if band:
        out[STOREYS] = int(band[0])
    for k, v in ((part.get("character") or {}).items()):
        if k in FABRIC_PARAMS and k not in out and isinstance(v, int):
            out[k] = int(v)
    for k, v in (fixed or {}).items():
        if isinstance(v, (int, bool)) or isinstance(v, str):
            out[k] = v
    return out


def _context(part: dict, spec: dict, voice: str | None, ctx: dict) -> dict:
    ch = part.get("character") or {}
    out = dict(CONTEXT_DEFAULT)
    out["voice"] = voice or ctx.get("voice") or spec.get("voice")
    out["attached"] = ch.get("attached") if ch.get("attached") is not None \
        else ctx.get("attached")
    out["frontage"] = ch.get("frontage") or ctx.get("frontage")
    for k in ("ground", "clearance", "seed"):
        if ctx.get(k) is not None:
            out[k] = ctx[k]
    return out


def _norm_params(params: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(params.items())) or "no fixed parameter"


# ------------------------------------------------------------------ the lot

def features_asked(d: dict) -> tuple:
    """Every token the envelope is asked to certify for this demand.

        The required tokens, **and `storeys` whenever the demand fixes a storey count**: the
        parameter means "build this many" and a query that does not ask for them to stand is
        answered by a shell that did not. This is why no production query here can have an
        empty feature set while a storeys parameter is on the demand.
        
    """
    got = list(d.get("required") or ())
    if isinstance((d.get("params") or {}).get(STOREYS), int) and STOREYS not in got:
        got.append(STOREYS)
    return tuple(got)


def _answer(d: dict, tname: str, got: dict, asked, params: dict, key: str) -> dict:
    return {"lot_min": list(got["lot_min"]), "lot_pref": list(got.get("lot_pref")
                                                             or got["lot_min"]),
            "source": got.get("source") or "probed", "type": tname,
            "features": dict(got.get("features") or {}),
            "asked": list(asked), "params": dict(params),
            "refused": None, "binding": bool(_binding(d, asked)),
            "context": dict(d.get("context") or {}), "key": key,
            "why": f"{tname}: {got.get('why') or ''}"}


def _binding(d: dict, tokens) -> bool:
    """Is any of these tokens one a requirement made mandatory?"""
    return bool(set(tokens or ()) & set(d.get("required") or ()))


def lot(d: dict, *, cache: str | None = None) -> dict:
    """The lot this demand needs, from the first approved type that delivers it.

        `{"lot_min", "lot_pref", "source", "features", "refused", "why", "key"}` and the
        type, the tokens asked and whether a refusal is `binding`. **A refusal carries
        `lot_min: None`**; there is no smaller default in this function and a caller that
        substitutes one has undone the contract.
        
    """
    asked = features_asked(d)
    params = dict(d.get("params") or {})
    ctx = dict(d.get("context") or {})
    voice, seed = ctx.get("voice"), int(ctx.get("seed") or 1)
    types = list(d.get("types") or [])
    if not types:
        return _refusal(d, asked, params, "no type is approved for this part, so there "
                                          "is no envelope to ask", [])
    gap = [t for t in (d.get("unsupported") or ()) if t in asked]
    if gap:
        # every approved type has declared its whole feature vocabulary and this token
        # is outside all of them: a capability gap, and no lot closes one
        return _refusal(d, asked, params,
                        f"no approved type of {types} declares {gap}, which "
                        f"{d.get('requirements')} require; this is a capability gap and "
                        f"not a lot", [(t, gap, "") for t in types])
    misses = []
    for tname in types:
        key = envelope._key(tname, params, asked, voice, seed, ctx)
        got = envelope.lot_for(tname, params, features=asked, voice=voice, seed=seed,
                               cache=cache, context=ctx)
        if got.get("lot_min"):
            return _answer(d, tname, got, asked, params, key)
        missing = [f for f, v in (got.get("features") or {}).items() if not v]
        misses.append((tname, missing, got.get("why") or ""))
    return _refusal(d, asked, params,
                    "; ".join(f"{t}: {w}" for t, _m, w in misses), misses)


def _refusal(d: dict, asked, params: dict, why: str, misses: list) -> dict:
    missing = sorted({f for _t, ms, _w in misses for f in ms}) or list(asked)
    binding = _binding(d, missing) or _binding(d, asked)
    return {"lot_min": None, "lot_pref": None, "source": "refused", "type": None,
            "features": {f: False for f in missing},
            "asked": list(asked), "params": dict(params),
            "refused": (f"no approved type delivers {_norm_params(params)} with "
                        f"{list(asked)}"),
            "binding": bool(binding), "context": dict(d.get("context") or {}),
            "key": None,
            "why": (f"{d.get('part')}: " + why
                    + (f". {sorted(set(missing) & set(d.get('required') or ()))} "
                       f"is required by {d.get('requirements')}, so this refusal is the "
                       f"answer and not a smaller lot" if binding else
                       ". Nothing requires it: the demand may be relaxed and asked "
                       "again, which is a decision and not a default"))}


def relax(d: dict) -> dict | None:
    """The same demand one storey lower, or None where the floor is required.

        The optional-choice control's one legitimate move. An **inferred** band -- the
        density word's, or the type's own -- may be revised down, and this is the only way
        down: it returns a new demand whose floor is lower and whose record says so, so the
        revision is on the demand a caller acts from rather than inside a sizing rule.
        
    """
    band = d.get("storeys_band")
    floor = (d.get("params") or {}).get(STOREYS)
    if not band or not isinstance(floor, int) or floor <= 1:
        return None
    if STOREYS in (d.get("required") or ()) or STOREYS in (d.get("fixed") or {}):
        return None
    if floor <= int(band[0]) and int(band[0]) <= 1:
        return None
    out = dict(d)
    out["params"] = {**(d.get("params") or {}), STOREYS: floor - 1}
    out["storeys_band"] = [min(int(band[0]), floor - 1), int(band[1])]
    out["band_from"] = (f"{d.get('band_from')}; relaxed from a floor of {floor} to "
                        f"{floor - 1}, which nothing requires")
    out["why"] = f"{d.get('why')}; relaxed to {floor - 1} storey(s)"
    return out


# ------------------------------------------------------------------ a lot in hand

_ADMITS: dict = {}


def storeys_admitted(d: dict, w: int, depth: int, *, cache: str | None = None) -> dict:
    """The most storeys of this demand's band a `w` x `depth` lot actually admits.

        `{"storeys": int | None, "asked": int, "band": [lo, hi], "holds": bool,
          "required": bool, "type": str | None, "why": str}`

        What `district_compile._storeys_fit` was doing, asked of the resolved demand instead
        of a type name and a band: the query carries the feature tokens (so a shell that
        stood without the storeys is not an answer), the voice and the physical context, and
        its cache is the envelope's -- which is keyed on the generator's content digest, so
        an edited type cannot answer from the certificate of the file it replaced. The old
        `_FIT_CACHE` was keyed on `(type, storeys)` alone and did all three of those wrong.

        **`storeys: None` is not "one".** Where not even the band's floor fits, `holds` is
        False and `storeys` is None, and the caller keeps asking for the floor: the ask is
        what a requirement made it, and the shortfall belongs in the emitted constraint where
        the obligation ledger can see it, not in a silently lowered parameter.
        
    """
    band = d.get("storeys_band")
    floor = (d.get("params") or {}).get(STOREYS)
    required = STOREYS in (d.get("required") or ())
    if not band or not isinstance(floor, int):
        return {"storeys": None, "asked": None, "band": None, "holds": True,
                "required": required, "type": None,
                "why": "this demand fixes no storeys, so no lot is short of them"}
    lo, hi = int(band[0]), int(band[1])
    ctx = dict(d.get("context") or {})
    voice, seed = ctx.get("voice"), int(ctx.get("seed") or 1)
    tname = next(iter(d.get("types") or []), None)
    if not tname:
        return {"storeys": None, "asked": floor, "band": [lo, hi], "holds": False,
                "required": required, "type": None,
                "why": "no type is approved for this part"}
    w, depth = int(w), int(depth)
    memo = (tname, lo, hi, w, depth, voice, seed, tuple(sorted(ctx.items(), key=str)),
            tuple(features_asked(d)))
    hit = _ADMITS.get(memo)
    if hit is not None:
        return dict(hit)
    best = None
    for st in range(hi, lo - 1, -1):
        params = {**(d.get("params") or {}), STOREYS: st}
        asked = features_asked({**d, "params": params})
        got = envelope.lot_for(tname, params, features=asked, voice=voice, seed=seed,
                               cache=cache, context=ctx)
        if got.get("lot_min") and envelope.fits((w, depth), got):
            best = st
            break
    out = {"storeys": best, "asked": floor, "band": [lo, hi],
           "holds": bool(best is not None and best >= floor),
           "required": required, "type": tname,
           "why": (f"a {w}x{depth} lot admits {best} storey(s) of {tname}"
                   if best is not None else
                   f"a {w}x{depth} lot admits no storey count in {[lo, hi]} of {tname}"
                   f"{'; the floor is required and the shortfall is the constraint'
                      if required else ''}")}
    _ADMITS[memo] = dict(out)
    return out


def validate_lot(d: dict, lot_wd, *, cache: str | None = None) -> dict:
    """`{"ok", "why"}` -- may this explicit lot be trusted to hold the demand?

        A `character.lot_width` is a decision a model wrote into the spec, and until now the
        compiler took it as given: the fabric arithmetic honoured it, the envelope was never
        asked about it, and a twelve-column frontage declared for a part that owes two
        storeys was simply built one storey high. This is the same query `lot` makes, asked
        of a lot somebody already chose.
        
    """
    need = lot(d, cache=cache)
    if need.get("refused"):
        return {"ok": False, "need": None, "lot": list(lot_wd) if lot_wd else None,
                "binding": bool(need.get("binding")), "why": need["why"]}
    ok = envelope.fits(lot_wd, need)
    mw, md = need["lot_min"]
    return {"ok": bool(ok), "need": list(need["lot_min"]),
            "lot": [int(lot_wd[0]), int(lot_wd[1])],
            "type": need.get("type"), "binding": bool(need.get("binding")),
            "why": (f"a {int(lot_wd[0])}x{int(lot_wd[1])} lot "
                    + ("holds" if ok else "does not hold")
                    + f" {need.get('type')} at {_norm_params(need['params'])} with "
                      f"{need['asked']}, which needs {mw}x{md}"
                    + ("" if ok else
                       (f"; {sorted(set(need['asked']) & set(d.get('required') or ()))} "
                        f"is required by {d.get('requirements')}"
                        if need.get("binding") else
                        "; nothing requires it and the demand may be relaxed")))}
