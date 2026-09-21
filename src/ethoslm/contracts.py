"""The versioned records the planning loop is made of, and the one place they are checked.

Before this module a stage handed the next one a dict and hoped. The audit's finding was
not that any single dict was wrong -- it was that **a requirement could disappear between
two of them and nothing could say where**: the sentence became a spec, the spec became a
plan, and an omitted wall was indistinguishable from a wall nobody asked for.

So the loop is written down as six records, each with a version, a producer, a consumer
and an invariant, and each read through `read()`, which **refuses a field it does not
know** rather than dropping it. A stage that invents a field is a stage that has changed
the contract, and it says so on the first run instead of on the run where the field
mattered.

    intent        the sentence, immutable, and every explicit requirement with an ID
                  producer: `ethoslm.intent.read`      consumer: everything after it
                  invariant: `sentence` and every requirement's `says` never change; a
                  requirement's `status` is the only mutable field on it

    reading       what was found out about the request: sourced claims, references,
                  uncertainty and the choices that were inferred rather than asked for
                  producer: `ethoslm.evidence.stage_reading`   consumer: programme, preview
                  invariant: a claim carries the source that supports it or is marked
                  `inferred`; a model's recollection is not a source

    programme     what the place has to be: functions, relations, organisation,
                  hierarchy, construction features and the scale constraints
                  producer: the place-spec stage    consumer: capabilities, layout
                  invariant: every entry is `explicit` (from a requirement) or
                  `inferred` (revisable, with a reason)

    capabilities  what the library can and cannot build of that programme
                  producer: `ethoslm.capability.match`   consumer: layout, growth
                  invariant: a match names the type **and the envelope it was checked
                  at**; an uncovered requirement is listed, never silently substituted

    resolution    the resolved spatial design: regions, boundaries, routes, levels,
                  surfaces and lots, each linked back to the requirement it serves
                  producer: the layout policies   consumer: the compiler and the builder
                  invariant: a promised lot is a lot the compiler can realise

    findings      what is wrong, whose job it is, and what it blocks
                  producer: every check     consumer: the repair router
                  invariant: `blocks` is one of feasibility, construction or fidelity,
                  and those three are never merged

`plan feasible`, `sample built` and `finished place` are three different states and this
module keeps them apart: `findings.blocks` is the field that says which one an open
finding is holding up.
"""
from __future__ import annotations

import hashlib
import json
import os
import time


class ContractError(ValueError):
    """A record that does not meet its contract. Never caught to paper over."""

    def __init__(self, message: str, *, record: str | None = None,
                 field: str | None = None):
        super().__init__(message)
        self.record = record
        self.field = field


#: The status a requirement may be in. `open` is the only one a producer may write; the
#: rest are a checker's answer. **`satisfied` is never inferred from a plan being
#: valid** -- it is written by the check that actually looked.
STATUSES = ("open", "satisfied", "unresolved", "unsupported", "failed")

#: What an open finding is holding up. Three states, never merged: a plan can be
#: feasible and unbuilt, built and unfaithful, or faithful and impossible.
BLOCKS = ("feasibility", "construction", "fidelity")

#: The stage that owns a finding -- the layer that can actually repair it. A finding
#: routed to a layer that cannot change the thing it names is a finding that will be
#: answered with prose.
OWNERS = ("reading", "programme", "scale", "site", "layout", "capability", "fabric",
          "surface", "material", "build")


def _str(v, where: str, record: str, *, allow_none=False) -> str | None:
    if v is None and allow_none:
        return None
    if not isinstance(v, str):
        raise ContractError(f"{record}.{where} is a string, not {type(v).__name__}",
                            record=record, field=where)
    return v


def _bool(v, where: str, record: str) -> bool:
    if not isinstance(v, bool):
        raise ContractError(f"{record}.{where} is true or false, not {v!r}",
                            record=record, field=where)
    return v


def _list(v, where: str, record: str) -> list:
    if not isinstance(v, list):
        raise ContractError(f"{record}.{where} is a list, not {type(v).__name__}",
                            record=record, field=where)
    return v


def _dict(v, where: str, record: str) -> dict:
    if not isinstance(v, dict):
        raise ContractError(f"{record}.{where} is an object, not {type(v).__name__}",
                            record=record, field=where)
    return v


def _known(doc: dict, fields: tuple, record: str, where: str = "") -> None:
    """**The whole point of the module.** An unknown field is an error and not a
    shrug: a stage that writes `organisation` into a record whose reader has never
    heard of it has changed the contract, and the run says so here."""
    unknown = sorted(set(doc) - set(fields))
    if unknown:
        raise ContractError(
            f"{record}{where}: unknown field(s) {unknown}; this record's fields are "
            f"{sorted(fields)}. A field this version does not know is refused rather "
            f"than dropped -- add it to the contract, or say it in a field that exists",
            record=record, field=unknown[0])


# ------------------------------------------------------------------ the records

#: One explicit requirement, with the ID everything downstream links to. `scope` is the
#: realization round's, and it is what makes a negation a negation. "a dense lower
#: district and a sparse upper district" states two requirements about two different
#: parts of one place, and a requirement that cannot say *which* part it is about
#: collapses them into one global claim -- which is exactly what the review found: "One
#: global dense requirement; the scoped contrast is unread." None is the whole place,
#: which is what every requirement written before this meant and still means.
REQUIREMENT_FIELDS = ("id", "says", "kind", "hard", "source", "phrase", "wants",
                      "status", "why", "evidence", "owner", "scope",
                      # the design round: **how** this status was established, which is
                      # a different question from how much of the place was built. See
                      # `intent.METHODS`.
                      "method")

#: One acquired claim: what it says, and the source identity that supports it.
CLAIM_FIELDS = ("id", "says", "about", "source", "confidence", "inferred", "conflicts")

#: One retrieved source. **The fingerprint is not optional**: a source whose bytes have
#: changed is a different source and the replay has to be able to say so.
SOURCE_FIELDS = ("id", "title", "url", "accessed", "fingerprint", "media", "supports",
                 "provider", "bytes", "path")

#: One programme entry: something the place has to be or do.
PROGRAMME_FIELDS = ("id", "says", "kind", "of", "from", "inferred", "why",
                    "requirement", "value")

#: One capability match or gap. `used` is what the place was actually built out of for
#: this want -- every type, not the first one -- written back by
#: `capability.agreements`. A record that says only what was selected cannot be compared
#: with what was built.
CAPABILITY_FIELDS = ("id", "requirement", "wants", "kind", "family", "form", "role",
                     "matched", "type", "alternatives", "envelope", "checked", "why",
                     "status", "used")

#: One resolved region of the design.
REGION_FIELDS = ("name", "policy", "role", "defines", "rect", "boundary", "inner",
                 "holes", "level", "routes", "access", "surface", "lots", "realized",
                 "faces", "requirement", "density", "voice", "anchor", "notes",
                 # the closure round: the denominator the density clause and the
                 # compiler's target share, so both measure cover over the same ground
                 "developable_columns",
                 # the expression round: the dimension this region was sized to, with
                 # the programme inputs it was derived from (`{"what", "value",
                 # "from"}`)
                 "target", "land_use", "lot_min", "hierarchy",
                 # **the design round: four columns, kept apart.** A density clause is
                 # about the ground the *requirement* names, and the expression round
                 # measured it over whatever the allocation happened to leave built --
                 # the ring layout called its remainder `open` and the checker dropped
                 # it. So: `scope_columns` is what the requirement is about and is fixed
                 # at resolution; `developable_columns` is what the compiler may build
                 # on; `allocated_columns` is what it drew; `built_columns` is what
                 # stands. `open_requested` carries the id of an explicit requirement
                 # asking for open ground, and is the **only** way ground leaves the
                 # denominator; `scope_of` names the region an inferred remainder was
                 # cut from, so it can be put back.
                 "rect_columns", "scope_columns", "allocated_columns", "built_columns",
                 "open_requested", "scope_of", "columns_from", "built_from",
                 # the arrangement the parent and the child negotiated (rows, bay width,
                 # frontage), with the capacity the actual compiler reported for each
                 # alternative it considered
                 "arrangement")

#: One finding.
FINDING_FIELDS = ("id", "says", "requirement", "part", "evidence", "owner", "blocks",
                  "severity", "seen_by", "fixed")


def _requirement(r, i: int, record: str) -> dict:
    _dict(r, f"requirements[{i}]", record)
    _known(r, REQUIREMENT_FIELDS, record, f".requirements[{i}]")
    out = {"id": _str(r.get("id"), f"requirements[{i}].id", record),
           "says": _str(r.get("says"), f"requirements[{i}].says", record),
           "kind": _str(r.get("kind"), f"requirements[{i}].kind", record),
           "hard": _bool(r.get("hard", True), f"requirements[{i}].hard", record),
           "source": _str(r.get("source") or "sentence",
                          f"requirements[{i}].source", record),
           "phrase": _str(r.get("phrase"), f"requirements[{i}].phrase", record,
                          allow_none=True),
           "wants": _dict(r.get("wants") or {}, f"requirements[{i}].wants", record),
           "status": _str(r.get("status") or "open",
                          f"requirements[{i}].status", record),
           "why": _str(r.get("why") or "", f"requirements[{i}].why", record),
           "owner": _str(r.get("owner"), f"requirements[{i}].owner", record,
                         allow_none=True),
           "scope": _str(r.get("scope"), f"requirements[{i}].scope", record,
                         allow_none=True),
           "evidence": _list(r.get("evidence") or [],
                             f"requirements[{i}].evidence", record),
           "method": _str(r.get("method"), f"requirements[{i}].method", record,
                          allow_none=True)}
    if not out["id"]:
        raise ContractError(f"{record}.requirements[{i}] has no id", record=record)
    if out["status"] not in STATUSES:
        raise ContractError(f"{record}.requirements[{i}].status is one of "
                            f"{list(STATUSES)}, not {out['status']!r}", record=record,
                            field="status")
    if out["owner"] is not None and out["owner"] not in OWNERS:
        raise ContractError(f"{record}.requirements[{i}].owner is one of {list(OWNERS)},"
                            f" not {out['owner']!r}", record=record, field="owner")
    return out


def _finding(f, i: int, record: str) -> dict:
    _dict(f, f"findings[{i}]", record)
    _known(f, FINDING_FIELDS, record, f".findings[{i}]")
    out = {"id": _str(f.get("id"), f"findings[{i}].id", record),
           "says": _str(f.get("says"), f"findings[{i}].says", record),
           "requirement": _str(f.get("requirement"), f"findings[{i}].requirement",
                               record, allow_none=True),
           "part": _str(f.get("part"), f"findings[{i}].part", record, allow_none=True),
           "evidence": _dict(f.get("evidence") or {}, f"findings[{i}].evidence", record),
           "owner": _str(f.get("owner"), f"findings[{i}].owner", record),
           "blocks": _str(f.get("blocks"), f"findings[{i}].blocks", record),
           "severity": _str(f.get("severity") or "error",
                            f"findings[{i}].severity", record),
           "seen_by": _str(f.get("seen_by") or "", f"findings[{i}].seen_by", record),
           "fixed": _bool(f.get("fixed", False), f"findings[{i}].fixed", record)}
    if out["owner"] not in OWNERS:
        raise ContractError(f"{record}.findings[{i}].owner is one of {list(OWNERS)}, "
                            f"not {out['owner']!r} -- a finding is routed to the layer "
                            f"that can repair it", record=record, field="owner")
    if out["blocks"] not in BLOCKS:
        raise ContractError(f"{record}.findings[{i}].blocks is one of {list(BLOCKS)}, "
                            f"not {out['blocks']!r}: a feasible plan, a built sample "
                            f"and a finished place are three different states",
                            record=record, field="blocks")
    return out


def _rows(doc: dict, key: str, fields: tuple, record: str) -> list:
    out = []
    for i, r in enumerate(_list(doc.get(key) or [], key, record)):
        _dict(r, f"{key}[{i}]", record)
        _known(r, fields, record, f".{key}[{i}]")
        out.append(dict(r))
    return out


#: `{record: (version, top-level fields)}`. The version is bumped when a reader would
#: mis-read an older document, and `read()` refuses a version it does not know.
VERSIONS = {
    "intent": (1, ("record", "version", "sentence", "digest", "requirements", "t",
                   "note",
                   # the expression round: the programme's entities -- what is a counted
                   # building, what is land, what is an amenity, a compound, a boundary
                   # -- bound once here and read by the solvers and the selectors
                   "entities")),
    "reading": (1, ("record", "version", "sentence", "classification", "sources",
                    "claims", "uncertainty", "inferred", "queries", "provider", "t",
                    "note")),
    "programme": (1, ("record", "version", "entries", "scale", "t", "note")),
    "capabilities": (1, ("record", "version", "entries", "cap", "t", "note")),
    "resolution": (1, ("record", "version", "policy", "regions", "site", "centre",
                       "bounds", "negotiated", "t", "note")),
    "findings": (1, ("record", "version", "findings", "stage", "t", "note")),
    # **What the sentence means, read by an agent and checked by the rules.** The
    # realization round's first boundary. `intent` is a phrase table, and a phrase table
    # is a finite instrument pointed at an open domain: it read "without a wall or a
    # temple" as requiring a temple, "not dense" as requiring dense, and "small houses
    # around a big temple" as requiring a temple and nothing else. Those are not gaps to
    # be filled with more rows. So meaning is read *here*, by a reader that can carry
    # scope, relation and hierarchy, and the rules become what cross-checks it. The
    # sentence is immutable and lives on this record unchanged; everything else is an
    # interpretation, each with its own reason, and each revisable against evidence.
    "interpretation": (1, ("record", "version", "sentence", "reads", "unread",
                           "uncertain", "checks", "source", "revisions", "t", "note")),
    # The direct test closed an identity with one cited claim and a Boolean verdict,
    # supplying no plan and no built result, while the final place reader answered
    # "nothing has been read about this name yet" for the same run. So a judgment is
    # bound to three things or it is not a judgment: the **candidate** it is of, the
    # **output** that was looked at, and the **claims** it was judged against. A verdict
    # with no evidence behind it is somebody's impression; evidence with no verdict is a
    # design nobody checked.
    "judgment": (1, ("record", "version", "sentence", "candidate", "verdicts",
                     "looked_at", "t", "note")),
}

#: One verdict: what obligation, what was decided, what was looked at, and why.
VERDICT_FIELDS = ("about", "name", "recognisable", "holds", "why", "cites", "from",
                  "confidence")

#: One thing an interpretation says the sentence says. kind one of
#: `INTERPRETATION_KINDS` says the claim, in the reader's own words wants the structured
#: content: the feature, the axis and value, the two sides of a relation, the count --
#: whatever that kind needs to be measurable scope which part of the place it is about;
#: None is the whole place phrase the span of the sentence it was read from, so the
#: cross-check can find it hard whether the sentence states it or the reader inferred it
#: why the reason, which is what makes an interpretation revisable rather than a second
#: set of rules nobody may question
READ_FIELDS = ("id", "kind", "says", "wants", "scope", "phrase", "hard", "why",
               "evidence", "confidence")

#: The kinds of thing a reader may say a sentence says. Deliberately the vocabulary of
#: *requirements* and not of buildings: what comes out of here has to be checkable
#: against a plan, and a kind nothing can measure is a kind that certifies itself.
INTERPRETATION_KINDS = ("feature", "absent", "count", "quality", "layout", "orientation",
                        "setting", "identity", "tradition", "function", "relation",
                        "hierarchy")

#: The relations a reader may state between two parts of a place, and what each means as
#: a measurement. `around` is the review's own counterexample -- "small houses around a
#: big temple" -- and it is a fact about bearings, not about a list of parts.
RELATIONS = ("around", "along", "facing", "inside", "beside", "between", "above")


def read(kind: str, doc: dict) -> dict:
    """One record, checked. Raises `ContractError`; never repairs silently.

        A document with no `record`/`version` is a **legacy** document and is adapted by
        `adapt()` first, explicitly, so "this came from before the contract" is a thing on
        the record rather than a guess.
        
    """
    if kind not in VERSIONS:
        raise ContractError(f"{kind!r} is not a record; the records are "
                            f"{sorted(VERSIONS)}")
    version, fields = VERSIONS[kind]
    if not isinstance(doc, dict):
        raise ContractError(f"a {kind} record is an object, not {type(doc).__name__}",
                            record=kind)
    if doc.get("record") is None and doc.get("version") is None:
        doc = adapt(kind, doc)
    if doc.get("record") != kind:
        raise ContractError(f"this is a {doc.get('record')!r} record, not a {kind!r} one",
                            record=kind, field="record")
    got = doc.get("version")
    if got != version:
        raise ContractError(f"{kind} record version {got!r}; this build reads version "
                            f"{version}. An older record is adapted on purpose, not "
                            f"read by accident", record=kind, field="version")
    _known(doc, fields, kind)
    out = dict(doc)
    out.setdefault("t", time.strftime("%Y-%m-%dT%H:%M:%S"))
    out.setdefault("note", "")
    if kind == "intent":
        out["sentence"] = _str(doc.get("sentence"), "sentence", kind)
        if not (out["sentence"] or "").strip():
            raise ContractError("an intent record carries the sentence it was made "
                                "from, and the sentence never changes", record=kind,
                                field="sentence")
        out["requirements"] = [_requirement(r, i, kind) for i, r in
                               enumerate(_list(doc.get("requirements") or [],
                                               "requirements", kind))]
        ids = [r["id"] for r in out["requirements"]]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ContractError(f"intent: two requirements share the id(s) {dupes}; an "
                                f"id is what a finding links back to", record=kind,
                                field="requirements")
        out["digest"] = doc.get("digest") or digest(out["sentence"])
    elif kind == "reading":
        out["sentence"] = _str(doc.get("sentence") or "", "sentence", kind)
        out["classification"] = _str(doc.get("classification") or "unclassified",
                                     "classification", kind)
        out["sources"] = _rows(doc, "sources", SOURCE_FIELDS, kind)
        out["claims"] = _rows(doc, "claims", CLAIM_FIELDS, kind)
        # **A source is a thing somebody can go and look at.** The integration review's
        # counterexample, one line after the one below it: a reading was accepted whose
        # only source was `{"id": "invented"}`, so the referenced-id rule was satisfied
        # by a bare identifier and "this claim is evidenced" meant "this claim cites a
        # string that appears twice in the same file". A citation resolving to nothing
        # is the defect; a citation resolving to an empty record is the same defect with
        # one more step.
        for i, s in enumerate(out["sources"]):
            missing = [f for f in ("id", "title", "accessed", "fingerprint")
                       if not str(s.get(f) or "").strip()]
            if not str(s.get("url") or s.get("path") or "").strip():
                missing.append("url or path")
            if missing:
                raise ContractError(
                    f"reading.sources[{i}] ({s.get('id')}) is missing "
                    f"{', '.join(missing)}. A source is a title, where it was read, when "
                    f"it was read and the identity of what was read; an id alone is a "
                    f"name for a source nobody retrieved",
                    record=kind, field="sources")
        have = {s.get("id") for s in out["sources"]}
        for i, c in enumerate(out["claims"]):
            if not c.get("source") and not c.get("inferred"):
                raise ContractError(
                    f"reading.claims[{i}] ({c.get('id')}) cites no source and is not "
                    f"marked `inferred`. A model's recollection is not evidence: either "
                    f"name the source it came from or say it was inferred",
                    record=kind, field="claims")
            # **A source id that names nothing is worse than no source id.** The
            # integration review's eighth finding, reproduced in one call: a claim
            # citing `nonexistent-source` was accepted as sourced, so "this is
            # evidenced" and "this is invented" were the same record with different
            # spelling. A citation is a reference and a reference has to resolve.
            if c.get("source") and c["source"] not in have:
                raise ContractError(
                    f"reading.claims[{i}] ({c.get('id')}) cites source "
                    f"{c['source']!r}, and this reading retrieved "
                    f"{sorted(have) or 'none'}. A claim that names a source nobody "
                    f"fetched is an inferred claim wearing a citation: mark it "
                    f"`inferred`, or retrieve the source it came from",
                    record=kind, field="claims")
        out["uncertainty"] = _list(doc.get("uncertainty") or [], "uncertainty", kind)
        out["inferred"] = _list(doc.get("inferred") or [], "inferred", kind)
        out["queries"] = _list(doc.get("queries") or [], "queries", kind)
        out["provider"] = _str(doc.get("provider") or "none", "provider", kind)
    elif kind == "interpretation":
        out["sentence"] = _str(doc.get("sentence") or "", "sentence", kind)
        if not out["sentence"].strip():
            raise ContractError("an interpretation carries the sentence it is of, "
                                "unchanged; the sentence is the one thing here that is "
                                "not an interpretation", record=kind, field="sentence")
        out["reads"] = _rows(doc, "reads", READ_FIELDS, kind)
        seen = set()
        for i, r in enumerate(out["reads"]):
            if r.get("kind") not in INTERPRETATION_KINDS:
                raise ContractError(
                    f"interpretation.reads[{i}] ({r.get('id')}) is a "
                    f"{r.get('kind')!r}, and a reading is one of "
                    f"{list(INTERPRETATION_KINDS)}. A kind nothing can measure is a "
                    f"claim that certifies itself", record=kind, field="reads")
            if not str(r.get("id") or "").strip():
                raise ContractError(f"interpretation.reads[{i}] has no id",
                                    record=kind, field="reads")
            if r["id"] in seen:
                raise ContractError(f"interpretation.reads[{i}] repeats the id "
                                    f"{r['id']!r}", record=kind, field="reads")
            seen.add(r["id"])
            # **A reading says why, or it is a second set of rules.** The whole reason
            # meaning moved out of the phrase table is that the table could not be
            # argued with; a reading with no reason cannot be either.
            if not str(r.get("why") or "").strip():
                raise ContractError(
                    f"interpretation.reads[{i}] ({r['id']}) gives no reason. An "
                    f"interpretation is revisable *because* it says why it reads the "
                    f"sentence that way; one that does not is a rule with no table",
                    record=kind, field="reads")
            if r.get("kind") == "relation":
                w = r.get("wants") or {}
                if w.get("relation") not in RELATIONS:
                    raise ContractError(
                        f"interpretation.reads[{i}] ({r['id']}) states the relation "
                        f"{w.get('relation')!r}, and this build can measure "
                        f"{list(RELATIONS)}. A relation it cannot measure is carried as "
                        f"an `unread` clause instead of as a requirement nothing checks",
                        record=kind, field="reads")
        out["unread"] = _list(doc.get("unread") or [], "unread", kind)
        out["uncertain"] = _list(doc.get("uncertain") or [], "uncertain", kind)
        out["checks"] = _dict(doc.get("checks") or {}, "checks", kind)
        out["source"] = _str(doc.get("source") or "agent", "source", kind)
        out["revisions"] = _list(doc.get("revisions") or [], "revisions", kind)
    elif kind == "judgment":
        out["sentence"] = _str(doc.get("sentence") or "", "sentence", kind)
        out["candidate"] = _str(doc.get("candidate"), "candidate", kind,
                                allow_none=True)
        out["looked_at"] = _list(doc.get("looked_at") or [], "looked_at", kind)
        out["verdicts"] = _rows(doc, "verdicts", VERDICT_FIELDS, kind)
        if not out["candidate"]:
            raise ContractError(
                "a judgment names the candidate it is of. An inspection that cannot say "
                "which design it looked at is evidence about nothing in particular, and "
                "carrying one across a revision is how a place gets qualified on a "
                "reading of the place before it", record=kind, field="candidate")
        if not out["looked_at"]:
            raise ContractError(
                "a judgment names what it looked at -- the renders, the built volume, "
                "the plan. A verdict with no output behind it is an opinion about a "
                "sentence", record=kind, field="looked_at")
        for i, v in enumerate(out["verdicts"]):
            if not str(v.get("why") or "").strip():
                raise ContractError(f"judgment.verdicts[{i}] gives no reason",
                                    record=kind, field="verdicts")
            if v.get("recognisable") is None and v.get("holds") is None:
                raise ContractError(
                    f"judgment.verdicts[{i}] about {v.get('about')!r} decides nothing. "
                    f"An inspection that returns prose is a reading; a verdict says "
                    f"whether the obligation is met", record=kind, field="verdicts")
    elif kind == "programme":
        out["entries"] = _rows(doc, "entries", PROGRAMME_FIELDS, kind)
        for i, e in enumerate(out["entries"]):
            if "inferred" not in e:
                raise ContractError(
                    f"programme.entries[{i}] ({e.get('id')}) does not say whether it is "
                    f"inferred. An inferred choice is revisable and an explicit "
                    f"requirement is not, and nothing downstream can tell them apart "
                    f"without this field", record=kind, field="entries")
        out["scale"] = _dict(doc.get("scale") or {}, "scale", kind)
    elif kind == "capabilities":
        out["entries"] = _rows(doc, "entries", CAPABILITY_FIELDS, kind)
        out["cap"] = doc.get("cap")
    elif kind == "resolution":
        out["policy"] = _str(doc.get("policy"), "policy", kind)
        out["regions"] = _rows(doc, "regions", REGION_FIELDS, kind)
        out["site"] = _dict(doc.get("site") or {}, "site", kind)
        out["centre"] = doc.get("centre")
        out["bounds"] = _dict(doc.get("bounds") or {}, "bounds", kind)
        out["negotiated"] = _list(doc.get("negotiated") or [], "negotiated", kind)
    elif kind == "findings":
        out["findings"] = [_finding(f, i, kind) for i, f in
                           enumerate(_list(doc.get("findings") or [], "findings", kind))]
        out["stage"] = _str(doc.get("stage") or "", "stage", kind)
    return out


def make(kind: str, **fields) -> dict:
    """A fresh record of `kind`, checked on the way out."""
    version, _f = VERSIONS[kind]
    return read(kind, {"record": kind, "version": version, **fields})


def adapt(kind: str, doc: dict) -> dict:
    """A document written before the contract, labelled as what it is.

        Explicit rather than implicit: a legacy record keeps every field it can be read
        into and is **marked** `legacy`, so nothing downstream can mistake an old plan's
        prose for a reading somebody verified. The one thing this never does is relabel:
        an adapted reading has no sources, and a claim with no source is `inferred`.
        
    """
    version, fields = VERSIONS[kind]
    out = {"record": kind, "version": version,
           "note": f"adapted from a document with no record header ({len(doc)} field(s))"}
    for k, v in doc.items():
        if k in fields:
            out[k] = v
    if kind == "intent" and not out.get("sentence"):
        out["sentence"] = str(doc.get("sentence") or doc.get("says") or "")
    if kind == "reading":
        out["sources"] = out.get("sources") or []
        out["claims"] = [dict(c, inferred=True) if not c.get("source") else dict(c)
                         for c in (out.get("claims") or [])]
        out["provider"] = out.get("provider") or "legacy"
    if kind == "programme":
        out["entries"] = [dict(e, inferred=bool(e.get("inferred", True)))
                          for e in (out.get("entries") or [])]
    return out


def digest(*parts) -> str:
    """The short content fingerprint the records and the dependency graph share."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (dict, list)):
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
        elif isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ------------------------------------------------------------------- on disk

#: Where each record lives inside a round's state.
PATHS = {"intent": "intent.json", "reading": "reading.json",
         "interpretation": "interpretation.json", "judgment": "judgment.json",
         "programme": "programme.json", "capabilities": "capabilities.json",
         "resolution": "resolution.json", "findings": "findings.json"}


def path_for(rnd, kind: str) -> str:
    return rnd.rel(PATHS[kind])


def save(rnd, kind: str, doc: dict) -> str:
    """Check, then write. A record that does not read is never on disk."""
    checked = read(kind, doc)
    p = path_for(rnd, kind)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(checked, fh, indent=1)
    return p


def load(rnd, kind: str) -> dict | None:
    """The record, checked, or None where the round has none."""
    p = path_for(rnd, kind)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return read(kind, json.load(fh))


def status_of(intent: dict) -> dict:
    """`{status: [requirement ids]}` -- the one summary every readout wants."""
    out: dict = {}
    for r in (intent or {}).get("requirements") or []:
        out.setdefault(r["status"], []).append(r["id"])
    return out


def unmet(intent: dict, *, hard_only: bool = True) -> list:
    """The requirements that are not satisfied. **An open requirement is not met**:
    nothing may read "we never checked" as "it is there"."""
    return [r for r in (intent or {}).get("requirements") or []
            if r["status"] != "satisfied" and (r["hard"] or not hard_only)]


def table(intent: dict) -> str:
    """The requirement table, for a brief and for a report."""
    rows = ["| id | says | hard | status | why |", "|---|---|---|---|---|"]
    for r in (intent or {}).get("requirements") or []:
        rows.append(f"| `{r['id']}` | {r['says']} | {'hard' if r['hard'] else 'soft'} "
                    f"| {r['status']} | {r['why'] or ''} |")
    return "\n".join(rows)
