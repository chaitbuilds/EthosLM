"""**Feature-aware envelopes: the lot a type needs to deliver what was asked of it.**

The expression round's second contract. The closure round measured, after construction,
that a cottage asked for two storeys on a 12x10 lot emitted one and that the type needs
16x14 for two (`construction.constraint`, `needs.lot_min`). Nothing consumed that before
the lot was drawn: `district_compile` admitted generic footprint bounds and chose storeys
separately, so the layout learned the envelope only from the wreckage. This module is the
question asked **before** the lot exists, by the layout owner and by the improve stage,
of the same type rules, and answered the same way whichever caller asks.

    lot_for(type_name, params, features=(...), voice=None, seed=1, cache=None)
        -> {"lot_min": [w, d], "lot_pref": [w, d], "source": "declared"|"probed",
            "features": {name: bool}, "why": str}
    required_features(requirements, part) -> tuple
        the feature tokens a part's type must deliver, read off the intent record
    table(type_name, seeds=(1, 2, 3)) -> list of ENVELOPE rows
        the measured table a type file may carry, produced by
        `scripts/type_needs.py --envelope <type>`

**Declared before probed, probed before guessed.** A type may carry an `ENVELOPE`
table -- rows of `{"params": {...}, "features": [...], "lot_min": [w, d],
"lot_pref": [w, d]}` measured by executing the type on flat ground -- and a row whose
parameters and features cover the question answers it at no cost. Otherwise the type is
probed through `construction.probe_build`/`outcome`, the lot grown from the type's own
declared minimum until the request survives, at one seed for `lot_min` and at three for
`lot_pref`; the answer is cached in memory and, where the caller names one, on disk.

**A flat-ground probe is evidence, not a promise.** The lot returned is what the type
needs on a plane; a slope, a lane on two sides or a neighbour's clearance can take more,
and construction's emitted outcome remains authoritative. What this closes is the other
half: a lot the type cannot use on *any* ground is never drawn on purpose.

Feature tokens are the names `construction.outcome` reports under `emitted.features`
(`outshot`, `courtyard`, `chimney`, `stalls`, `forge`, `hearth`, ...) plus one marker,
`storeys`, meaning "the storeys in `params` must actually stand". A token the type never
reports is answered `False` and named in `why`, which is a capability gap and not a
small lot.

**A certificate names what it was measured of.** The design round's first cause: `_key`
hashed the type's *name*, its parameters and its features, so an edited or newly authored
generator answered from the answer measured of the file it replaced -- and the authoring
path this project has exists precisely to edit and author generators mid-run. A key now
carries `_deps_print`: the content digest of the type file, of the builder modules that
execute it, and of the voice palette it is built in, together with the physical context
the probe was run under. The on-disk cache carries `CACHE_VERSION`; a document written
under another one still loads and is **ignored rather than trusted**, because an old
certificate that cannot say what it was measured of is not evidence.
"""
from __future__ import annotations

import json
import os

#: How far a probe grows a lot on each axis from where it starts. The same bound
#: `construction.PROBE_GROWTH` uses after construction, so the two instruments agree.
GROWTH = 12
#: The seeds `lot_pref` is held to: a lot that delivers the request at every one of
#: these is a lot the layout can draw without the seed deciding the storeys.
PREF_SEEDS = (1, 2, 3)
#: The marker token for "the requested storeys must stand".
STOREYS = "storeys"

#: **What a stored certificate is a certificate of.** Bumped when the key's meaning
#: changes; entries written under another version are read and dropped. The design
#: round's version is the first that binds an answer to the generator that produced it.
CACHE_VERSION = "envelopes-v2-demand"

#: The modules that execute a type file, and whose content therefore decides what the
#: type emits. A type imports nothing of this project (they take `b` and the stdlib), so
#: this list is the whole of what stands behind a probe besides the type itself.
BUILDER_MODULES = ("buildlib.py", "prims.py", "voices.py", "styles.py")

_MEM: dict = {}


# ------------------------------------------------------------------ requirements

def required_features(requirements, part: dict | None = None) -> tuple:
    """The feature tokens a part's type must deliver, off the intent record.

        A `quality/height` requirement makes the planned storeys a requirement (`storeys`);
        a `feature` or `function` requirement whose wants name a courtyard, a workshop, a
        market or a hearth maps to the feature the type has to emit. A requirement with a
        scope applies only to a part inside that scope; `part` is the leaf or defining part
        (`name`, `defines`, `in`) the question is asked for, and None asks for the place.
        
    """
    return tuple(required_by(requirements, part))


def required_by(requirements, part: dict | None = None) -> dict:
    """`{token: [requirement_id, ...]}` -- the tokens and **what made each required**.

        The design round's first contract asks a demand to carry the requirement ids behind
        every required feature, so that a refusal can be reported against the thing that was
        asked for rather than against a token. `required_features` is this in the shape its
        existing callers take. Insertion order is requirement order, which is the order a
        reader of the intent record meets them in.
        
    """
    from . import intent as intent_mod
    out: dict = {}

    def add(token, rid):
        out.setdefault(token, [])
        if rid and rid not in out[token]:
            out[token].append(rid)

    for r in requirements or []:
        if isinstance(r, dict) and r.get("status") == "unsupported":
            continue
        kind = r.get("kind")
        w = r.get("wants") or {}
        scope = r.get("scope")
        if part is not None and scope and not intent_mod._in_scope(
                {"name": part.get("name"), "defines": part.get("defines") or part.get("name"),
                 "in": part.get("in") or []}, scope):
            continue
        rid = r.get("id")
        if kind == "quality" and w.get("axis") == "height":
            add(STOREYS, rid)
        said = {str(w.get(k) or "").lower() for k in ("family", "feature", "function", "what")}
        said.discard("")
        for token, words in FEATURE_WORDS.items():
            if said & set(words):
                add(token, rid)
    return out


#: Which requirement words ask for which emitted feature. A word absent here asks for no
#: feature -- a `dwelling` is answered by a type declaring the function, and the storeys
#: it stands are the `storeys` token's business.
FEATURE_WORDS = {
    "courtyard": ("courtyard", "court", "siheyuan", "courtyard_house"),
    "stalls": ("market", "stalls", "bazaar"),
    "forge": ("smithy", "smith", "forge", "workshop", "smithing"),
    "hearth": ("hearth", "kitchen"),
    "shopfront": ("shop", "shops", "shopfront"),
    "gate": ("gate",),
}


#: `word -> token`, the inverse of `FEATURE_WORDS`, built once.
_TOKEN_OF = {w: t for t, ws in FEATURE_WORDS.items() for w in ws}


def feature_token(word) -> str | None:
    """The **emitted-feature token** a requirement's word asks for, or None.

        `market -> stalls`, `smithy -> forge`, `court -> courtyard`; a token is its own word.
        None where the word asks for no emitted feature at all.

        The composition round's fifth evidence connection, and the defect it closes is one
        string comparison: `pipeline/improve._required_tokens` collected the sentence's own
        words, so `feature/market` gave `{"market"}` while the construction constraint it was
        about emits `what = "stalls"`, and `obligation.from_constraint` computed
        `material = "stalls" in {"market"}` -- False. The row was never selected, on a
        requirement the sentence makes hard. `FEATURE_WORDS` is the table that already knew
        the two words are the same ask; this is it read the other way round.
        
    """
    w = str(word or "").strip().lower()
    if not w:
        return None
    if w in FEATURE_WORDS or w == STOREYS:
        return w
    got = _TOKEN_OF.get(w)
    if got is None and w.endswith("s"):
        got = _TOKEN_OF.get(w[:-1])
    if got is None:
        got = _TOKEN_OF.get(w + "s")
    return got


# ------------------------------------------------------------------ the answer

def _norm(params: dict | None) -> dict:
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, (int, float)):
            out[str(k)] = int(v)
        elif v is not None:
            out[str(k)] = str(v)
    return dict(sorted(out.items()))


#: `type_name -> digest`, for the life of the process. A solve asks the same type for a
#: dozen lots and the digest of `buildlib.py` does not change between two of them.
_DEPS: dict = {}


def _deps_print(type_name: str, voice: str | None) -> str:
    """The identity of everything that decides what a probe of this type emits.

        The type file's bytes, the builder modules that execute it, and the voice palette it
        is built in. `deps.content_print` is the same hash the dependency contract uses, so a
        type file that changes without changing length is a different type here too.
        
    """
    key = (type_name, voice or None)
    got = _DEPS.get(key)
    if got is None:
        from . import contracts, deps
        from . import pipeline as _pipeline
        rows = [("type", content := deps.content_print(
            os.path.join(_pipeline.ROOT, "types", f"{type_name}.py")))]
        for mod in BUILDER_MODULES:
            rows.append((mod, deps.content_print(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), mod))))
        if voice:
            rows.append(("voice", deps.content_print(
                os.path.join(_pipeline.ROOT, "voices", f"{voice}.json"))))
        # a type with no file on disk fingerprints as absent rather than as anything
        got = _DEPS[key] = contracts.digest(rows, None if content else "no such type")
    return got


#: The physical context a probe was run under, in the order it is keyed. A probe builds
#: on flat ground with no neighbour, and naming that here is what stops a flat-ground
#: certificate from being read as an answer about a terrace on a slope.
CONTEXT_KEYS = ("ground", "clearance", "attached", "frontage")


def _context_print(context: dict | None) -> list:
    return [(k, _norm({k: (context or {}).get(k)}).get(k)) for k in CONTEXT_KEYS]


def _key(type_name: str, params: dict, features, voice, seed,
         context: dict | None = None) -> str:
    return json.dumps([CACHE_VERSION, type_name, _deps_print(type_name, voice),
                       _norm(params), sorted(set(features or ())), voice or None,
                       int(seed), _context_print(context)], sort_keys=True)


def _load_cache(path: str | None) -> dict:
    """The entries of a cache document written under **this** version of the key.

        A document from another version loads and is dropped: its keys were made of a
        different question, and an answer whose question cannot be recovered is not one.
        
    """
    if path and os.path.exists(path):
        try:
            doc = json.load(open(path))
        except (OSError, ValueError):
            return {}
        if isinstance(doc, dict) and doc.get("version") == CACHE_VERSION:
            return dict(doc.get("entries") or {})
    return {}


def _save_cache(path: str | None, doc: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    json.dump({"version": CACHE_VERSION, "entries": doc,
               "note": ("each key carries the digest of the type file, the builder "
                        "modules and the voice it was measured with; a document of "
                        "another version is ignored, never trusted")},
              open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def declared_table(type_name: str) -> list:
    """The `ENVELOPE` rows a type file carries, or an empty list."""
    from . import construction
    try:
        ns = construction._type_ns(type_name)
    except Exception:                          # noqa: BLE001 -- no type, no table
        return []
    rows = ns.get("ENVELOPE")
    return [dict(r) for r in rows] if isinstance(rows, (list, tuple)) else []


_FEATURES: dict = {}


def declared_features(type_name: str) -> tuple:
    """The feature tokens a type file says it can be asked for (`FEATURES`), or ().

        What a type *may* choose to deliver, as against what a requirement *asks* it to:
        `ethoslm.demand` reports the two apart so that a token nothing required can be revised
        away and a token a requirement named cannot. `pipeline.load_type` does not carry
        `FEATURES`, so the file is executed here and remembered per its content digest.
        
    """
    key = (type_name, _deps_print(type_name, None))
    got = _FEATURES.get(key)
    if got is None:
        from . import construction
        try:
            ns = construction._type_ns(type_name)
        except Exception:                      # noqa: BLE001 -- no type, no features
            ns = {}
        got = _FEATURES[key] = tuple(str(f) for f in (ns.get("FEATURES") or ()))
    return got


def declared_needs(type_name: str) -> dict:
    from . import construction, pipeline
    try:
        ns = construction._type_ns(type_name)
        return pipeline.read_needs(ns, type_name)
    except Exception:                          # noqa: BLE001
        return dict(pipeline.NEEDS_DEFAULT)


def _row_answers(row: dict, params: dict, features) -> bool:
    """Does a declared row cover the question? Its parameters must equal the asked ones
    it names (a row naming fewer covers more), and its features must include every one
    asked for."""
    rp = _norm(row.get("params") or {})
    want = _norm(params)
    for k, v in rp.items():
        if want.get(k) != v:
            return False
    have = set(row.get("features") or ())
    return set(features or ()) <= have


def _delivered(got: dict, params: dict, features, said: dict | None = None,
               type_name: str | None = None) -> dict:
    """Which of the asked features the emitted outcome delivers. `storeys` is delivered
        when the measured storeys reach the parameter; a named feature when it is measured
        or verified; a feature the outcome looked for and did not find
        (`claimed_not_found`) is never delivered.

        **A type's own word is not evidence about a type that undertook to prove it.** The
        design round. Two branches believed the generator: `source == "declared"` -- a claim
        with no rectangle behind it -- counted as delivered, and a token
        `construction.outcome` has no verifier for at all fell through to
        `emitted.features`. That is the right answer for a file written before any of this
        existed, and the wrong one for a type declaring `FEATURES`, because the growth brief
        its author was handed says in so many words that **a declared feature with no
        rectangle is not evidence**. So a token inside the type's own declared vocabulary is
        delivered only where the geometry says so, and a type that declares no vocabulary is
        believed exactly as it was. Checked on all four committed types that declare
        `FEATURES` -- market, square, courtyard_house, worship -- every one of whose tokens
        is emitted with a rectangle and verified, so nothing that passed stops passing.
        
    """
    feats = got.get("features") or {}
    src = got.get("features_source") or {}
    told = (said or {}).get("features") or {}
    vocab = set(declared_features(type_name)) if type_name else set()
    out = {}
    for f in features or ():
        if f == STOREYS:
            want = params.get(STOREYS)
            st = got.get("storeys")
            out[f] = bool(isinstance(st, int) and (not isinstance(want, int) or st >= want))
        elif f in vocab:
            out[f] = bool(feats.get(f)) and src.get(f) in ("measured", "verified")
        elif f in feats or f in src:
            out[f] = bool(feats.get(f)) and src.get(f) != "claimed_not_found"
        else:
            out[f] = bool(told.get(f))
    return out


def _stands(type_name: str, w: int, d: int, params: dict, features, seed: int,
            voice) -> tuple:
    """(True where the request survives on a w x d lot, the delivered map)."""
    from . import construction
    try:
        b, sited, res = construction.probe_build(type_name, w, d, params, seed=seed,
                                                 voice=voice)
    except Exception:                          # noqa: BLE001 -- a probe reports
        return False, {}
    if not res.get("ok"):
        return False, {}
    got = construction.outcome(b, sited, None, params)
    if not got.get("stood", True):
        return False, {}
    dl = _delivered(got, params, features,
                    res.get("emitted") if isinstance(res.get("emitted"), dict) else None,
                    type_name=type_name)
    return all(dl.values()), dl


def _cands(w0: int, d0: int, k: int) -> list:
    return sorted({(w0 + k, d0 + j) for j in range(0, k + 1)}
                  | {(w0 + j, d0 + k) for j in range(0, k + 1)},
                  key=lambda c: (c[0] * c[1], c))


def probe(type_name: str, params: dict, features=(), *, seed: int = 1,
          voice: str | None = None, start: tuple | None = None) -> dict:
    """Find the least lot on which the request survives, inside the type's declared band.

        Squares first from the declared minimum up to the declared maximum plus `GROWTH`,
        then each axis shrunk back while the request still stands, so a narrow-and-deep
        answer is found as well as a square one. `lot_min` is the least lot (by area) at
        `seed`; `lot_pref` the least lot at which it survives at every seed in `PREF_SEEDS`.
        A request nothing in the band delivers answers `None` for both, with the features
        that never appeared named.
        
    """
    needs = declared_needs(type_name)
    fp = needs.get("footprint") or (3, 3, 64, 64)
    w0, d0 = (int(start[0]), int(start[1])) if start else (int(fp[0]), int(fp[1]))
    hi = max(int(fp[2]), int(fp[3])) + GROWTH
    params = dict(params or {})
    features = tuple(features or ())
    tried = [0]
    delivered_last: dict = {}

    def stands(w, d, s):
        tried[0] += 1
        ok, dl = _stands(type_name, w, d, params, features, s, voice)
        if dl:
            delivered_last.update(dl)
        return ok

    def least(seeds) -> tuple | None:
        sq = None
        for side in range(max(w0, d0), hi + 1):
            if all(stands(side, side, s) for s in seeds):
                sq = side
                break
        if sq is None:
            return None
        best = (sq, sq)
        # shrink one axis at a time while it still stands
        for axis in (0, 1):
            w, d = best
            while True:
                nw, nd = (w - 1, d) if axis == 0 else (w, d - 1)
                if nw < w0 or nd < d0:
                    break
                if not all(stands(nw, nd, s) for s in seeds):
                    break
                w, d = nw, nd
            if w * d < best[0] * best[1]:
                best = (w, d)
        return best

    lot_min = least((int(seed),))
    lot_pref = None
    if lot_min:
        lot_pref = least(tuple(PREF_SEEDS)) or lot_min
        if lot_pref[0] * lot_pref[1] < lot_min[0] * lot_min[1]:
            lot_pref = lot_min
    never = [f for f, v in delivered_last.items() if not v] if not lot_min else []
    return {"lot_min": list(lot_min) if lot_min else None,
            "lot_pref": list(lot_pref) if lot_pref else None,
            "source": "probed",
            "features": ({f: True for f in features} if lot_min else delivered_last),
            "probes": tried[0],
            "why": (f"{type_name} delivers {_norm(params)} with {list(features)} from "
                    f"{lot_min[0]}x{lot_min[1]} at seed {seed}, and at seeds "
                    f"{list(PREF_SEEDS)} from {lot_pref[0]}x{lot_pref[1]}" if lot_min else
                    f"{type_name} delivers no lot up to {hi}x{hi} with {_norm(params)} and "
                    f"{list(features)}: " + (f"{never} never appeared" if never
                                              else "no shell stood"))}


def lot_for(type_name: str, params: dict | None, *, features=(), voice: str | None = None,
            seed: int = 1, cache: str | None = None, context: dict | None = None) -> dict:
    """The lot a type needs to deliver `params` with `features`. See the module.

        `context` is the physical context the answer is about -- the ground, the clearance,
        whether the house is attached and which way it fronts. It is keyed and recorded and
        it does not yet change the probe, which builds on flat ground with free flanks: what
        it buys today is that a flat-ground certificate cannot be read as an answer about a
        terrace on a slope once one of them can be measured. `ethoslm.demand` fills it.
        
    """
    params = dict(params or {})
    features = tuple(f for f in (features or ()) if f)
    key = _key(type_name, params, features, voice, seed, context)
    if key in _MEM:
        return dict(_MEM[key])
    disk = _load_cache(cache)
    if key in disk:
        _MEM[key] = disk[key]
        return dict(disk[key])
    for row in declared_table(type_name):
        if _row_answers(row, params, features) and row.get("lot_min"):
            got = {"lot_min": [int(v) for v in row["lot_min"]],
                   "lot_pref": [int(v) for v in (row.get("lot_pref") or row["lot_min"])],
                   "source": "declared", "features": {f: True for f in features},
                   "context": _context_print(context), "tested_on": "flat ground",
                   "why": (f"{type_name}'s ENVELOPE table: {row.get('params')} with "
                           f"{row.get('features')} from {row['lot_min'][0]}x"
                           f"{row['lot_min'][1]}" + (f" ({row['why']})" if row.get("why")
                                                     else ""))}
            _MEM[key] = got
            disk[key] = got
            _save_cache(cache, disk)
            return dict(got)
    got = probe(type_name, params, features, seed=seed, voice=voice)
    got["context"] = _context_print(context)
    got["tested_on"] = "flat ground"
    _MEM[key] = got
    disk[key] = got
    _save_cache(cache, disk)
    return dict(got)


def fits(lot, need: dict) -> bool:
    """Does a `(w, d)` lot hold what `lot_for` said is needed?"""
    if not need or not need.get("lot_min") or not lot:
        return False
    w, d = int(lot[0]), int(lot[1])
    mw, md = need["lot_min"]
    return (w >= mw and d >= md) or (w >= md and d >= mw)


# ------------------------------------------------------------------ the table

def _feature_sets(type_name: str, params: dict) -> list:
    """The feature sets worth a row: none, storeys, and each feature the type's
    parameters or declared `FEATURES` name, with storeys."""
    sets = [(), (STOREYS,)]
    for f in declared_features(type_name):
        sets.append((STOREYS, f))
    return sets


def table(type_name: str, seeds=PREF_SEEDS, *, params_list: list | None = None,
          voice: str | None = None) -> list:
    """Measure the ENVELOPE rows for a type: every parameter combination its `PARAMS`
    declares (or `params_list`) crossed with the feature sets, each probed from the
    type's declared minimum. This is what `scripts/type_needs.py --envelope` runs."""
    from . import construction, pipeline
    ns = construction._type_ns(type_name)
    combos = params_list or pipeline.param_combinations(ns.get("PARAMS") or {}, most=64)
    rows = []
    for params in combos:
        for feats in _feature_sets(type_name, params):
            got = probe(type_name, params, feats, seed=int(seeds[0]), voice=voice)
            if not got.get("lot_min"):
                rows.append({"params": _norm(params), "features": list(feats),
                             "lot_min": None, "lot_pref": None, "why": got["why"]})
                continue
            rows.append({"params": _norm(params), "features": list(feats),
                         "lot_min": got["lot_min"], "lot_pref": got["lot_pref"],
                         "why": f"probed at seeds {list(seeds)}"})
    return rows


def transcribe(type_path: str, rows: list, command: str) -> str:
    """Write (or replace) the `ENVELOPE` block in a type file. Returns the block."""
    import re
    body = (json.dumps(rows, indent=1).replace(": null", ": None")
            .replace(": true", ": True").replace(": false", ": False"))
    block = (f"#: **The lots this type needs, measured** by `{command}`: each row is the\n"
             f"#: least lot on which the parameters stand with the features named, at one\n"
             f"#: seed (`lot_min`) and at seeds {list(PREF_SEEDS)} (`lot_pref`). Read by\n"
             f"#: `ethoslm.envelope.lot_for` before a lot is drawn; the outcome construction\n"
             f"#: measures afterwards remains authoritative.\n"
             f"ENVELOPE = {body}\n")
    src = open(type_path).read()
    pat = re.compile(r"#: \*\*The lots this type needs, measured\*\*.*?\nENVELOPE = \[.*?\n\]\n",
                     re.S)
    if pat.search(src):
        src = pat.sub(lambda _m: block, src, count=1)
    else:
        m = re.search(r"\nNEEDS = \{.*?\n\}\n", src, re.S)
        if m:
            src = src[:m.end()] + "\n" + block + src[m.end():]
        else:
            src = src + "\n" + block
    open(type_path, "w").write(src)
    return block
