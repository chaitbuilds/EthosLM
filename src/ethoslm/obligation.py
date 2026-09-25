"""**One ledger of what the place still owes**, across readings and across rebuilds.

The design round's fourth contract. `pipeline/improve.py` dispositions two lists --
the judge's findings and the emitted construction constraints -- into one document and
then selects actions out of the first list only. Four escapes follow from that, and the
review found all four by tracing the control flow:

  1. **a reading that omits an old finding closes it.** The cycle is closed with
     `c["closed"] = c["finding"] in closed_ids`, and the next reading's `closed` list is
     the reader's. A reader that simply does not mention a finding has closed it.
  2. **an applied-but-ineffective action removes the finding from action selection.**
     `open_material` excludes every finding in `acted`, whether or not the action worked,
     so one refused move ends the place's chances with that finding for ever.
  3. **a missing measurement permits closure.** `_measure_moved` answers
     `toward_target: None` where a measure is absent on either reading, and closure is
     rejected only on `toward_target is False`.
  4. **emitted constraints never reach action selection.** They are dispositioned into
     `disp["constraints"]` and no action is ever chosen from that list.

So: one row shape for a judged finding and an emitted constraint, one file
(`obligations.json`) in the round's state, and four rules that are the negations above.

    upsert(ledger, rows, source, ...)   findings and constraints, one shape, one ledger
    open_rows(ledger, ...)              what is still owed
    close(ledger, id, evidence, ...)    only with fresh relevant evidence on this candidate
    act / effect                        what was tried, and whether it moved the measure
    admissible(ledger, id, candidate, among)   the bounded alternatives still open

    from_finding(f) / from_constraint(c, required)   the two producers, into one shape

**The ledger never closes a row by itself and never closes one on absence.** `close`
refuses without a measurement of the row's own measure, on the row's own candidate;
`upsert` moves `last_seen` and leaves `disposition` exactly where it was. Everything a
row knows about itself -- who owns it, what would settle it, what has been tried on it
and what that did -- is on the row, because the thing that went wrong before is that
those four facts lived in four places and only one of them was consulted.
"""
from __future__ import annotations

import json
import os
import time

#: Where the ledger lives inside a round's state.
RECORD = "obligations.json"

#: The dispositions a row may carry. `open` is the only one `open_rows` returns and the
#: only one a row is born with: an **optional** row is not thereby disposed of, which is
#: the acceptance table's "an optional change with no disposition" falsifier. `accepted`
#: and `deferred` are decisions somebody made and they say so.
DISPOSITIONS = ("open", "closed", "accepted", "deferred", "refused")

#: Where a row came from. Two producers, one shape: that is the whole contract.
ORIGINS = ("finding", "constraint")

#: The record's own version, so a ledger written by another build is read as what it is.
VERSION = 1


# ------------------------------------------------------------------ the shape

def row(rid: str, origin: str, *, about=None, says: str = "", subjects=(),
        owner=None, measure=None, target=None, acceptance=None, material=True,
        requirement=None, part=None, evidence=None, source: str = "",
        at: str | None = None) -> dict:
    """One obligation, in the one shape. See the module."""
    if origin not in ORIGINS:
        raise ValueError(f"origin is one of {ORIGINS}, not {origin!r}")
    now = at or time.strftime("%Y-%m-%dT%H:%M:%S")
    return {"id": str(rid), "origin": origin, "about": about, "says": says,
            "subjects": [str(s) for s in subjects or ()], "owner": owner,
            "measure": measure, "target": target, "acceptance": acceptance,
            "material": bool(material), "requirement": requirement, "part": part,
            "actions": [], "disposition": "open", "why": "",
            "evidence": list(evidence or []),
            "first_seen": now, "last_seen": now, "last_reported": source or None,
            # the candidates this row has been reported on, the candidate its closure
            # was made against, and every closure that was later invalidated. See
            # `upsert`.
            "seen_on": [], "closed_against": None, "reopened": [],
            "omitted_by": [], "sources": [source] if source else []}


def from_finding(f: dict, *, source: str = "", at: str | None = None) -> dict:
    """A judged finding of the built reading, as an obligation.

        `material` is `improve.is_material`'s rule, kept here rather than imported so the
        ledger can be read without the pipeline: the reading says so outright, or it names
        a contextual target, and a finding with neither is a suggestion.
        
    """
    mat = bool(f.get("material")) if f.get("material") is not None else bool(f.get("target"))
    out = row(f.get("id") or f"find/{f.get('about')}", "finding",
               about=f.get("about"), says=f.get("says") or "",
               subjects=f.get("subjects") or (), owner=f.get("owner"),
               measure=(f.get("measure")
                        or (f.get("target") or {}).get("measure")
                        if isinstance(f.get("target"), dict) else f.get("measure")),
               target=f.get("target"), acceptance=f.get("acceptance"),
               material=mat, requirement=f.get("requirement"), part=f.get("part"),
               source=source, at=at)
    # the reader's own action document (a revision, a relevel), carried on the row so a
    # later pass on another candidate's reading still knows what was asked
    if isinstance(f.get("action"), dict):
        out["action"] = dict(f["action"])
    return out


def from_constraint(c: dict, required=(), *, part: str | None = None,
                    source: str = "", at: str | None = None) -> dict:
    """An emitted construction constraint, as an obligation **of the same shape**.

        The review's fourth escape: a constraint was dispositioned into its own list and no
        action was ever selected from it, so "the layout gave a 12x10 lot where 15x17
        delivers the storeys" was recorded by the run that could have fixed it and acted on
        by nobody. Here it is a row like any other, with the lot it needs as its
        `acceptance` and `enlarge_lots` as the action its owner admits.

        `required` is what the request **requires**, in any of three shapes: the feature
        tokens themselves, the requirement *words* the sentence used, or the per-part binding
        `demand.required_by_part` answers (`{part: {token: [ids]}}`). Whichever it is, both
        sides are normalised through `envelope.feature_token` before they are compared.

        **A word is not a token, and comparing them refused every constraint the round cares
        about.** The composition round's fifth evidence connection, measured on the imported
        code: `pipeline/improve._required_tokens` collects the sentence's own words, so
        `feature/market` yields `{"market"}`; a construction constraint about the same
        requirement emits `what = "stalls"`; and this line computed
        `material = "stalls" in {"market"}` -- False. The row was created, was never material,
        was never in `open_rows(led, material=True)` and was never selected, on a hard
        requirement. `feature_token` maps the word to the token the emission uses, and where
        the binding is given the answer is about **this part** rather than about the place.
        
    """
    what = str(c.get("what"))
    where = part or c.get("part")
    lot = (c.get("needs") or {}).get("lot")
    lot_min = (c.get("needs") or {}).get("lot_min")
    mat, why_mat = _material_token(what, required, where)
    return row(f"constraint/{what}/{where}", "constraint",
               about=what, says=c.get("why") or "",
               subjects=[str(where)] if where else (), owner=c.get("owner"),
               measure=f"emitted.{what}", material=mat,
               target=({"measure": f"emitted.{what}", "value": c.get("requested"),
                        "direction": "up"} if c.get("requested") is not None else None),
               acceptance=({"lot": lot, "lot_min": lot_min,
                            "action": "enlarge_lots"} if lot_min else None),
               part=where, evidence=[{"type": c.get("type"), "emitted": c.get("emitted"),
                                      "requested": c.get("requested"),
                                      "needs": c.get("needs"), "material": why_mat}],
               requirement=_requirement_ids(what, required, where) or None,
               source=source, at=at)


def _tokens_of(required, part=None) -> tuple:
    """`(tokens, ids)` -- what is required, normalised, for this part or for the place."""
    from . import envelope
    got = required
    if isinstance(required, dict):
        # the per-part binding: this row is about this part and no other
        got = required.get(str(part)) or {}
    tokens, ids = set(), []
    if isinstance(got, dict):
        for w, rs in got.items():
            t = envelope.feature_token(w) or str(w)
            tokens.add(t)
            for r in rs or ():
                if r not in ids:
                    ids.append(str(r))
    else:
        for w in got or ():
            tokens.add(envelope.feature_token(w) or str(w))
    return (tuple(sorted(tokens)), ids)


def _material_token(what: str, required, part=None) -> tuple:
    """`(material, why)` -- is a constraint about `what` one the place owes?"""
    from . import envelope
    token = envelope.feature_token(what) or str(what)
    tokens, _ids = _tokens_of(required, part)
    if isinstance(required, dict) and str(part) not in required:
        return (False, f"the binding names no required feature for `{part}`, so `{token}` "
                       f"is this type's own choice on this lot and not an obligation")
    ok = token in tokens
    return (ok, (f"`{what}` is the token `{token}`, which {sorted(tokens)} requires"
                 if ok else
                 f"`{what}` normalises to `{token}` and the required tokens are "
                 f"{sorted(tokens) or 'none'}: the type's own choice on this lot"))


def _requirement_ids(what: str, required, part=None) -> list:
    from . import envelope
    token = envelope.feature_token(what) or str(what)
    got = required.get(str(part)) if isinstance(required, dict) else None
    if isinstance(got, dict):
        return [str(r) for r in (got.get(token) or got.get(what) or ())]
    return []


# ------------------------------------------------------------------ the ledger

def empty() -> dict:
    return {"record": "obligations", "version": VERSION, "rows": {}, "passes": [],
            "note": "one row per obligation, findings and emitted constraints alike; "
                    "a reading that omits a row does not close it"}


def load(state: str) -> dict:
    """The ledger in a round's state, or a new one. **Survives a rebuild**: the file is
    not one of the artifacts the improve stage sets aside, and a row opened against one
    candidate is still owed by the next."""
    p = os.path.join(state, RECORD)
    if not os.path.exists(p):
        return empty()
    try:
        doc = json.load(open(p))
    except (OSError, ValueError):
        return empty()
    if not isinstance(doc, dict) or doc.get("version") != VERSION:
        return empty()
    doc.setdefault("rows", {})
    doc.setdefault("passes", [])
    return doc


def save(state: str, ledger: dict) -> str:
    os.makedirs(state, exist_ok=True)
    p = os.path.join(state, RECORD)
    tmp = p + ".tmp"
    json.dump(ledger, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return p


def upsert(ledger: dict, rows, source: str, *, candidate: str | None = None,
           at: str | None = None) -> dict:
    """Fold a pass's rows into the ledger. Returns the ledger.

        `rows` are already in this module's shape (`from_finding` / `from_constraint`).
        `source` names the pass -- the reading, the parts record -- and is what an omission
        is recorded against.

        **A row this pass did not mention stays exactly as it was.** Its `last_seen` moves,
        because the ledger looked; its `disposition` does not, because looking is not
        finding. That one line is the difference between "the judge no longer mentions the
        fragmented groups" and "the fragmented groups were fixed".

        **...and a row this pass *did* mention, on a candidate other than the one it was
        closed against, reopens.** The composition round's third evidence connection, and the
        defect is the mirror of the one above: this function refreshed a closed row's wording
        and left its `disposition` at `closed`, and wrote `candidate` only on first insert. So
        a finding closed against candidate A that the reading of candidate B finds again kept
        `disposition: "closed"` and candidate A's name, was invisible to
        `open_rows(led, material=True)`, was never selected for an action, and got no
        `omitted_by` trace either -- the omission loop skips rows that are not open. A
        repeated failure cannot remain closed. What was closed, against which candidate and
        why that closure no longer holds is kept on `reopened`, which is the history the
        round asks for; nothing is deleted.

        The other invariant is untouched: **a reading that merely omits an open row does not
        close it, and a pass that does not mention a closed row does not reopen it.** Only
        being reported again, on a different candidate, reopens anything.
        
    """
    now = at or time.strftime("%Y-%m-%dT%H:%M:%S")
    seen = set()
    for r in rows or []:
        rid = r["id"]
        seen.add(rid)
        was = ledger["rows"].get(rid)
        if was is None:
            ledger["rows"][rid] = {**r, "first_seen": now, "last_seen": now,
                                   "last_reported": source,
                                   "sources": [source] if source else [],
                                   **({"candidate": candidate,
                                       "seen_on": [candidate]} if candidate else {})}
            continue
        # **What a row says is refreshed; what has happened to it is not.** A second
        # reading may word a finding differently or measure it again, and the actions
        # tried on it and the disposition somebody gave it are this ledger's, not the
        # reader's.
        for k in ("about", "says", "subjects", "owner", "measure", "target",
                  "acceptance", "material", "requirement", "part", "action"):
            if r.get(k) not in (None, "", [], {}):
                was[k] = r[k]
        was["last_seen"] = now
        was["last_reported"] = source
        if source and source not in was.get("sources", []):
            was.setdefault("sources", []).append(source)
        for e in r.get("evidence") or []:
            if e not in was.get("evidence", []):
                was.setdefault("evidence", []).append(e)
        if candidate:
            if candidate not in (was.get("seen_on") or []):
                was.setdefault("seen_on", []).append(candidate)
            _recur(was, candidate, source, now)
    for rid, was in ledger["rows"].items():
        if rid in seen:
            continue
        # `last_seen` moves for every row, whatever its disposition: the ledger looked.
        # `omitted_by` records which pass did not mention it -- also for every row, so a
        # closed row that nothing has re-reported is a fact a reader can see rather than
        # a silence. Neither changes a disposition.
        was["last_seen"] = now
        if source and source not in was.get("omitted_by", []):
            was.setdefault("omitted_by", []).append(source)
    ledger["passes"].append({"source": source, "candidate": candidate, "at": now,
                             "reported": sorted(seen),
                             "reopened": sorted(rid for rid in seen
                                                if (ledger["rows"].get(rid) or {})
                                                .get("reopened_at") == now),
                             "open_after": len(open_rows(ledger))})
    return ledger


def closed_against(r: dict):
    """The candidate a row's closure was made against, or None.

        `close` stamps it and `upsert` clears it when the closure is invalidated, so a
        reopened row answers None even though the closing evidence is still on it. For a row
        closed before this field existed it is read off the closing evidence entry, and
        failing that off the candidate the row was opened on.
        
    """
    if "closed_against" in r:
        return r["closed_against"]
    for e in reversed(r.get("evidence") or []):
        if isinstance(e, dict) and e.get("candidate"):
            return e["candidate"]
    return r.get("candidate")


def _recur(was: dict, candidate: str, source: str, now: str) -> None:
    """Reopen a **closed** row that has recurred on a different candidate. See `upsert`."""
    if was.get("disposition") != "closed":
        return
    against = closed_against(was)
    if against is None or str(against) == str(candidate):
        return
    why = (f"this obligation was closed against candidate {against} "
           f"({str(was.get('why') or 'no reason recorded')}) and {source or 'a later pass'} "
           f"reports it again on candidate {candidate}: a repeated failure cannot remain "
           f"closed, and the evidence that closed it is of a design this is not")
    was.setdefault("reopened", []).append(
        {"was": "closed", "closed_against": against, "closed_why": was.get("why"),
         "recurred_on": candidate, "by": source, "at": now, "why": why})
    was["disposition"] = "open"
    was["reopened_at"] = now
    was["closed_against"] = None
    was["why"] = why


def open_rows(ledger: dict, *, material: bool | None = None,
              origin: str | None = None, candidate: str | None = None) -> list:
    """What is still owed. Every row whose disposition is `open`, in id order.

        `material=True` is what blocks completion; `material=False` is the set of optional
        rows nobody has disposed of, which is a finding about the run and not about the
        place. `candidate` filters to rows opened against one candidate; by default a row
        is owed whichever candidate found it, because a hard requirement does not stop being
        owed because the design moved.
        
    """
    out = []
    for rid in sorted(ledger.get("rows") or {}):
        r = ledger["rows"][rid]
        if r.get("disposition") != "open":
            continue
        if material is not None and bool(r.get("material")) != material:
            continue
        if origin is not None and r.get("origin") != origin:
            continue
        # a row reported on this candidate is owed here whatever candidate opened it: a
        # row opened against A and found again on B is B's obligation too (see `upsert`)
        if candidate is not None and r.get("candidate") not in (None, candidate) \
                and candidate not in (r.get("seen_on") or []):
            continue
        out.append(r)
    return out


def undisposed(ledger: dict) -> list:
    """Optional rows nobody has decided about. See `DISPOSITIONS`."""
    return open_rows(ledger, material=False)


# ------------------------------------------------------------------ what was tried

def act(ledger: dict, rid: str, action: str | None, candidate: str, *,
        applied: bool, why: str = "", record=None, at: str | None = None,
        key: str | None = None) -> dict:
    """Record that `action` was tried on this row, on this candidate. Returns the row.

        Whether it **worked** is `effect`'s answer and is `None` until something measures
        it: an action is applied, then the world is rebuilt and read, then the measure is
        compared. Recording "applied" as "effective" is the shortcut that closed findings
        nothing had fixed.

        `key` is the caller's fingerprint of **what this action carries**, where an action
        name is not the whole of it. See `tried_here`.
        
    """
    r = _row(ledger, rid)
    r.setdefault("actions", []).append(
        {"action": action, "candidate": candidate, "applied": bool(applied),
         "effective": None, "why": why, "record": record,
         **({"key": str(key)} if key is not None else {}),
         "at": at or time.strftime("%Y-%m-%dT%H:%M:%S")})
    return r


def effect(ledger: dict, rid: str, candidate: str, before: dict | None,
           after: dict | None) -> dict:
    """Did the last applied action on this candidate move the row's cited measure?

        `{"measure", "before", "after", "target", "moved": bool | None, "why"}`, written
        onto the action and returned. `moved: None` means **the measure was not recorded on
        both readings**, which is not evidence of anything and in particular is not evidence
        of improvement -- see `close`, which refuses on it.
        
    """
    r = _row(ledger, rid)
    tried = [a for a in r.get("actions") or []
             if a.get("candidate") == candidate and a.get("applied")]
    got = _moved(r, before, after)
    if tried:
        tried[-1]["effective"] = got["moved"]
        tried[-1]["measured"] = got
    return got


def _moved(r: dict, before: dict | None, after: dict | None) -> dict:
    m = r.get("measure") or ((r.get("target") or {}).get("measure")
                             if isinstance(r.get("target"), dict) else None)
    if not m:
        return {"measure": None, "before": None, "after": None, "moved": None,
                "why": "this row cites no measure, so nothing can be said to have moved"}
    a, b = (before or {}).get(m), (after or {}).get(m)
    if a is None or b is None:
        return {"measure": m, "before": a, "after": b, "moved": None,
                "why": (f"{m} was not recorded on both readings "
                        f"({'before' if a is None else 'after'} is missing); an absent "
                        f"measurement is not evidence of improvement")}
    t = r.get("target") if isinstance(r.get("target"), dict) else {}
    value, direction = t.get("value"), str(t.get("direction") or "")
    moved = None
    # a target with a direction is met by going past it that way (the design resolution
    # round: a step brought from 7 to 2 against a target of "down to 6" read as "moved
    # away from its target"); only a target with no direction is a point to approach
    if value is not None and direction not in ("up", "down"):
        try:
            moved = abs(float(b) - float(value)) < abs(float(a) - float(value))
        except (TypeError, ValueError):
            moved = None
    if moved is None and direction in ("up", "down"):
        try:
            moved = (float(b) < float(a)) if direction == "down" else (float(b) > float(a))
        except (TypeError, ValueError):
            moved = None
    if moved is None:
        try:
            moved = float(b) != float(a)
        except (TypeError, ValueError):
            moved = (b != a)
    return {"measure": m, "before": a, "after": b, "target": value,
            "direction": direction or None, "moved": bool(moved),
            "why": f"{m}: {a} -> {b}" + (f" against {value}" if value is not None else "")}


def admissible(ledger: dict, rid: str, candidate: str, among) -> list:
    """The bounded actions still open on this row, on this candidate.

        **An applied action that did not move the measure leaves the row open and leaves
        the other actions admissible.** The review's second escape, negated: `improve`
        excluded every finding in `acted` from further selection, so a `resize_ring` that
        was applied and did nothing ended the ring's chances. What is refused is the
        *unchanged retry* -- the same action on the same candidate -- and nothing else.
        
    """
    r = ledger.get("rows", {}).get(str(rid)) or {}
    tried = {a.get("action") for a in r.get("actions") or []
             if a.get("candidate") == candidate}
    return [a for a in among or () if a not in tried]


def tried_here(ledger: dict, rid: str, candidate: str, action: str | None,
               key: str | None = None) -> bool:
    """Has this exact action been tried on this candidate? The unchanged retry.

        **Exact means the action and what it carries.** The block design round. Some actions
        are a name and nothing else -- `terrace`, `compact_bay` -- and for those the name is
        the action. A `character` revision is a name and a document: the reading says which
        parts and what their characters become. This module refused the second of two
        different character revisions as a repeat of the first, because both are spelled
        `character`.

        `key` is the caller's fingerprint of the document. Two attempts with different keys
        are two different actions under one name; two with the same key, or two with none,
        are the unchanged retry this exists to refuse.
        
    """
    for a in (ledger.get("rows", {}).get(str(rid)) or {}).get("actions") or []:
        if a.get("candidate") != candidate or a.get("action") != action:
            continue
        if a.get("key") == key:
            return True
    return False


# ------------------------------------------------------------------ closing one

#: How near a numeric target counts as reached where the target names no direction. Five
#: per cent: a measure quoted to two figures that lands on the number.
TARGET_TOLERANCE = 0.05


#: The prefix a row's measure carries when the thing it cites is a **feature the
#: construction emitted**, not a view measure of the place. `from_constraint` writes
#: `emitted.<token>`; see `close`, and `FEATURE_EVIDENCE` for what closes one.
EMITTED = "emitted."

#: The key in `close`'s evidence carrying **post-build feature evidence**:
#: evidence["features"] = {"<part>/<feature>": answer, ...} A key may also be the tuple
#: `(part, feature)`, or just `"<feature>"` where the caller has already narrowed to one
#: part. An `answer` is `construction.evidence_for`'s dict -- `{"holds", "method",
#: "reason", "why", ...}` -- or a bare bool for a caller that only has the verdict.
#: `confirm` writes exactly these answers onto `emitted.usable` and `emitted.owed`, so
#: the ledger and the parts record agree by construction.
FEATURE_EVIDENCE = "features"


def close(ledger: dict, rid: str, evidence: dict, *, candidate: str,
          reading: dict | None = None) -> dict:
    """Close a row -- **only** with fresh relevant evidence on the current candidate.

        `{"closed": bool, "why": str, "row": row}`. `candidate` is required and is the
        candidate being judged **now**: a caller that cannot say which design it is looking
        at cannot close anything, and that is deliberate rather than defensive. Six refusals,
        and each of them is one of the escapes this module exists to close:

          * evidence of another candidate: a reading of a withdrawn design cannot close an
            obligation about this one;
          * no measurement of the row's own `measure`: an absent measure is not evidence,
            and "the new reading no longer mentions it" is an absent measure;
          * a measurement that did not reach the row's `acceptance` or its `target`;
          * a cited measure that moved **away** from its target;
          * **a measurement that is not about this row's subject** (the composition round);
          * **an `emitted.<feature>` row offered a view measure instead of feature
            evidence** (the composition round).

        **The row's own measure, on the row's own subject.** This decided relevance by a
        single string lookup of `measure` in a flat global measures dict, so a finding about
        `middle_ring_north_west` closed on `undeveloped_share` measured over the whole place:
        the measure matched by name and nothing asked what it was a measurement *of*. Three
        shapes are now understood, in this order --

            evidence["value"]                   the caller has already measured this row
            evidence["by_subject"][s][measure]  measured per district or part
            evidence["measures"][measure]       the view's own measures, of the place

        -- and the third one is refused for a row that names subjects whenever the evidence
        says what it is about (`evidence["subject"]`) and that is not one of them, or carries
        `by_subject` for other subjects and not for this row's. Where the evidence names no
        subject at all the whole-place measure is taken as it always was, and the row records
        that nothing said what it was a measurement of.

        **An `emitted.<feature>` row closes from post-build feature evidence.** `improve.py`
        only ever offered these rows `inspect.MEASURES` -- `clusters`, `cluster_gap`,
        `square_scale`, `open_to_built`, `undeveloped_share`, `plots` -- which contains no
        `emitted.*` key, so `constraint/stalls/<part>` could not be closed by any evidence
        that existed and stayed open for ever whatever the world did. What settles it is
        whether the assembled world carries the feature: `evidence["features"]`, whose answers
        are `construction.evidence_for`'s. An unknown or unsupported answer does **not** close
        it, which is the same rule one level up.

        `reading` is the whole reading where the caller has one, so the evidence recorded on
        the row names what was read and not merely a number.
        
    """
    r = _row(ledger, rid)
    here = evidence.get("candidate") or candidate
    if here != candidate:
        return _refuse(r, f"the evidence is of candidate {here} and this is {candidate}; "
                          f"a reading of another design does not close this obligation")
    m = r.get("measure") or ((r.get("target") or {}).get("measure")
                             if isinstance(r.get("target"), dict) else None)
    feature = _feature_answer(r, m, evidence)
    if feature is not None and not feature.get("affirms"):
        return _refuse(r, feature["why"])
    value, about, why_about = _value_for(r, m, evidence, feature)
    if about == "mismatch":
        return _refuse(r, str(why_about))
    if m and value is None:
        return _refuse(r, f"the evidence carries no {m}: this row cites that measure and "
                          f"an absent measurement is not evidence of improvement")
    if not m and not evidence.get("observed"):
        return _refuse(r, "this row cites no measure and the evidence records no "
                          "observation; there is nothing here to close it with")
    t = r.get("target") if isinstance(r.get("target"), dict) else {}
    if feature is None and t.get("value") is not None \
            and evidence.get("before") is not None:
        got = _moved(r, {m: evidence["before"]}, {m: value})
        if got["moved"] is False:
            return _refuse(r, f"the cited measure moved away from its target -- "
                              f"{got['why']}")
    if feature is not None:
        # the feature evidence **is** the acceptance: the assembled world carries what
        # this row said was missing, and that is the whole question the row asks
        ok, why = (True, feature["why"])
    else:
        ok, why = _accepted(r, value, evidence)
    if not ok:
        return _refuse(r, why)
    r["disposition"] = "closed"
    r["closed_against"] = here
    r["why"] = why + (f"; {why_about}" if why_about and about == "the place" else "")
    r.setdefault("evidence", []).append(
        {"candidate": here, "measure": m, "value": value,
         "observed": evidence.get("observed"), "about": about,
         "feature": (feature or {}).get("answer"),
         "reading": (reading or {}).get("id") or evidence.get("reading"),
         "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "why": r["why"]})
    return {"closed": True, "why": r["why"], "row": r}


def subjects_of(r: dict) -> list:
    """Every name this row is about: its `subjects` and its `part`. See `close`."""
    got = [str(s) for s in (r.get("subjects") or ()) if s]
    if r.get("part") and str(r["part"]) not in got:
        got.append(str(r["part"]))
    return got


def _value_for(r: dict, m, evidence: dict, feature) -> tuple:
    """`(value, about, why)` -- this row's measurement, of this row's subject.

        `about` is what the value was measured of: a subject name, `"the place"`, `"feature"`,
        `"the caller"`, or `"mismatch"` with `why` naming the mismatch that refuses closure.
        
    """
    if feature is not None:
        return (True, "feature", feature["why"])
    if "value" in evidence:
        return (evidence["value"], "the caller", None)
    if not m:
        return (None, None, None)
    mine = subjects_of(r)
    by = evidence.get("by_subject")
    if isinstance(by, dict) and by:
        for s in mine:
            if isinstance(by.get(s), dict) and by[s].get(m) is not None:
                return (by[s][m], s, f"{m} measured on {s}, this row's own subject")
        if mine:
            return (None, "mismatch",
                    f"this row is about {mine} and the evidence measures {m} on "
                    f"{sorted(by)}: a measurement of another subject does not close it")
    said = evidence.get("subject") or evidence.get("subjects") or evidence.get("about")
    if said is not None and mine:
        names = [str(said)] if isinstance(said, str) else [str(x) for x in said or ()]
        if names and not (set(names) & set(mine)):
            return (None, "mismatch",
                    f"this row is about {mine} and the evidence is a measurement of "
                    f"{names}: a measurement of another subject does not close it")
    got = (evidence.get("measures") or {}).get(m)
    return (got, "the place",
            (f"the evidence does not say which subject `{m}` is a measurement of; this "
             f"row is about {mine} and the value is taken as the place's" if mine
             else None))


def _feature_answer(r: dict, m, evidence: dict):
    """The post-build feature evidence for an `emitted.<feature>` row, or None.

        `None` means this row is not about an emitted feature, or no feature evidence was
        offered for it -- in which case the ordinary measure path runs and refuses on absence.
        Otherwise `{"affirms": bool, "why": str, "answer": the answer}`.
        
    """
    if not isinstance(m, str) or not m.startswith(EMITTED):
        return None
    token = m[len(EMITTED):]
    got = evidence.get(FEATURE_EVIDENCE)
    if not isinstance(got, dict) or not got:
        return None
    mine = subjects_of(r)
    answer = key = None
    for k, v in got.items():
        name, _, feat = (k if isinstance(k, str) else "/".join(str(x) for x in k)) \
            .rpartition("/")
        if feat != token:
            continue
        if name and mine and name not in mine:
            continue
        answer, key = v, (f"{name}/{feat}" if name else feat)
        if name:
            break
    if answer is None:
        return None
    if isinstance(answer, bool):
        answer = {"holds": answer, "method": "observed" if answer else "observed",
                  "why": f"the caller reports {token} {'stands' if answer else 'absent'}"}
    holds, method = answer.get("holds"), str(answer.get("method") or "")
    ok = holds is True and method in ("observed", "inferred")
    return {"affirms": ok, "answer": dict(answer), "key": key,
            "why": (f"the assembled world carries `{token}` on {key}: {answer.get('why')}"
                    if ok else
                    f"the post-build evidence for `{token}` on {key} answers "
                    f"holds={holds!r} by `{method or 'nothing'}`"
                    + (f" ({answer.get('reason')})" if answer.get("reason") else "")
                    + f": {answer.get('why')}. An unknown or declared answer is not "
                      f"evidence that the feature is there")}


def _accepted(r: dict, value, evidence: dict) -> tuple:
    """Does this measurement meet what the row said would settle it?"""
    acc = r.get("acceptance")
    if isinstance(acc, dict) and acc.get("lot_min") and evidence.get("lot"):
        need, lot = acc["lot_min"], evidence["lot"]
        ok = ((int(lot[0]) >= int(need[0]) and int(lot[1]) >= int(need[1]))
              or (int(lot[0]) >= int(need[1]) and int(lot[1]) >= int(need[0])))
        return (ok, (f"the lot is now {lot} against the {need} this constraint asked "
                     f"for" if ok else
                     f"the lot is {lot} and this constraint asks for {need}"))
    if isinstance(acc, dict) and acc.get("at_least") is not None:
        try:
            ok = float(value) >= float(acc["at_least"])
        except (TypeError, ValueError):
            return (False, f"{value!r} is not a number and this row is accepted at "
                           f"{acc['at_least']}")
        return (ok, f"{r.get('measure')} is {value} against the {acc['at_least']} this "
                    f"row is accepted at")
    if isinstance(acc, dict) and acc.get("at_most") is not None:
        try:
            ok = float(value) <= float(acc["at_most"])
        except (TypeError, ValueError):
            return (False, f"{value!r} is not a number and this row is accepted at "
                           f"{acc['at_most']}")
        return (ok, f"{r.get('measure')} is {value} against the {acc['at_most']} this "
                    f"row is accepted at")
    if isinstance(acc, str) and acc:
        if not evidence.get("observed"):
            return (False, f"this row is accepted when {acc!r}, and the evidence records "
                           f"no observation of it")
        return (True, f"observed: {acc}")
    # **A target is what settles the row, not a direction of travel.** Closing on "it
    # moved toward the target" is how a square thirteen times its cottages becomes a
    # square twelve times its cottages and a closed finding. Where the row cites a
    # number, the measurement has to reach it.
    t = r.get("target") if isinstance(r.get("target"), dict) else {}
    if t.get("value") is not None:
        try:
            got, want = float(value), float(t["value"])
        except (TypeError, ValueError):
            return (False, f"{value!r} is not a number and this row's target is "
                           f"{t['value']}")
        d = str(t.get("direction") or "")
        ok = (got <= want if d == "down" else got >= want if d == "up"
              else abs(got - want) <= TARGET_TOLERANCE * max(abs(want), 1.0))
        return (ok, (f"{r.get('measure')} is {got} against the target {want}"
                     + (f" ({d})" if d else "")
                     + ("" if ok else ", which it has not reached")))
    return (True, f"{r.get('measure') or 'the row'} measured "
                  f"{value if value is not None else 'as observed'} on this candidate")


def _refuse(r: dict, why: str) -> dict:
    """A closure the ledger would not make. **The row stays open**, and the reason it
    stays open is on it, which is how a run says "this was claimed closed and is not"."""
    r["why"] = why
    r.setdefault("refused_closures", []).append(
        {"why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"closed": False, "why": why, "row": r}


def dispose(ledger: dict, rid: str, disposition: str, why: str) -> dict:
    """Give a row an explicit disposition other than closure.

        The only way an **optional** row leaves `open_rows`: somebody decided, and the
        decision and its reason are on the record. `closed` is not available here -- that is
        `close`'s job and it needs evidence.
        
    """
    if disposition not in DISPOSITIONS or disposition == "closed":
        raise ValueError(f"disposition is one of "
                         f"{[d for d in DISPOSITIONS if d != 'closed']}, not "
                         f"{disposition!r}; closure needs evidence and goes through "
                         f"`close`")
    r = _row(ledger, rid)
    if r.get("material") and disposition in ("accepted", "deferred"):
        raise ValueError(f"{rid} is material: the place is not finished without it, and "
                         f"it cannot be accepted or deferred. Close it with evidence or "
                         f"leave it open")
    r["disposition"] = disposition
    r["why"] = why
    return r


def _row(ledger: dict, rid: str) -> dict:
    got = ledger.setdefault("rows", {}).get(str(rid))
    if got is None:
        raise KeyError(f"no obligation {rid!r} in this ledger; a row is opened by "
                       f"`upsert` and closed by id")
    return got


# ------------------------------------------------------------------ a readout

def says(ledger: dict) -> str:
    """One line, for a log."""
    rows = list((ledger.get("rows") or {}).values())
    by: dict = {}
    for r in rows:
        by[r.get("disposition")] = by.get(r.get("disposition"), 0) + 1
    mat = len(open_rows(ledger, material=True))
    return (f"{len(rows)} obligation(s): "
            + (", ".join(f"{v} {k}" for k, v in sorted(by.items())) or "none")
            + f"; {mat} open and material"
            + (f", {len(undisposed(ledger))} optional and undisposed"
               if undisposed(ledger) else ""))


def table(ledger: dict) -> list:
    """Every row, flattened for a readout or a report."""
    out = []
    for rid in sorted(ledger.get("rows") or {}):
        r = ledger["rows"][rid]
        out.append({"id": rid, "origin": r.get("origin"), "owner": r.get("owner"),
                    "material": bool(r.get("material")),
                    "disposition": r.get("disposition"),
                    "actions": [f"{a.get('action')}@{str(a.get('candidate'))[:8]}"
                                f"{'' if a.get('applied') else ' (refused)'}"
                                f"{'' if a.get('effective') is not False else ' (no effect)'}"
                                for a in r.get("actions") or []],
                    "omitted_by": len(r.get("omitted_by") or []),
                    "why": r.get("why"), "says": str(r.get("says"))[:160]})
    return out
