"""**What the sentence means, read by an agent and cross-checked by the rules.**

The realization round's first boundary. `intent.read` is a phrase table, and the review
showed what a phrase table does to an open domain -- not that it misses things, but that
missing everything reads exactly like asking for nothing, and that a near miss reads as
the opposite of what was asked:

    "without a wall or a temple"                 wall absent; temple **required**
    "not dense"                                  dense required; `not` an unread word
    "dense lower district and a sparse upper"    one global dense requirement
    "small houses around a big temple"           a temple; the hierarchy gone

Adding rows does not fix any of those. A negation that scopes over one conjunct, a
contrast between two districts of one place, and a size relation between two kinds of
building are **structure**, and a table of phrases has nowhere to put structure.

So the order is inverted. An agent reads the sentence into an `interpretation` record --
scoped, related, reasoned, and with the sentence carried unchanged beside it -- and the
rules become the *cross-check*: they run on the same sentence, and every disagreement
between them is reported rather than resolved silently. What survives is:

  - what the reader says, where the rules agree or say nothing;
  - what the rules say and the reader missed, as a requirement marked `source: "rules"`,
    because a table that finds a wall the reader forgot is still right about the wall;
  - what the reader says and the sentence does not support, **refused** -- a reading has
    to quote the span it read;
  - every content word neither of them claimed, as a `clause/` obligation, exactly as
    before.

The sentence is immutable. Interpretations are revisable with reasons, which is why
`why` is required on every one of them: a reading that cannot say why it reads the
sentence that way is a second set of rules that nobody is allowed to argue with.
"""
from __future__ import annotations

import json
import re

from . import contracts, intent as intent_mod


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "x"


def read_answer(sentence: str, doc: dict, *, source: str = "agent") -> dict:
    """An agent's answer, as a checked `interpretation` record. Raises `ContractError`.

        The answer is a list of readings; everything else on the record -- the sentence, the
        unread words, the checks -- is filled in here, so the reader is answerable for the
        meaning and not for the bookkeeping.
        
    """
    reads = []
    for i, r in enumerate(doc.get("reads") or []):
        got = dict(r)
        got.setdefault("id", f"{got.get('kind', 'read')}/{_slug(got.get('says') or i)}")
        got.setdefault("hard", True)
        got.setdefault("evidence", [])
        got.setdefault("wants", {})
        reads.append(got)
    return contracts.make("interpretation", sentence=str(sentence), reads=reads,
                          unread=list(doc.get("unread") or []),
                          uncertain=list(doc.get("uncertain") or []),
                          source=source,
                          note=str(doc.get("note") or "")
                               or "read from the sentence by an agent; the rules in "
                                  "`ethoslm.intent` cross-check it and every disagreement "
                                  "is on the record")


#: What a rule-derived requirement and a reading have to share to be about the same
#: thing. Deliberately coarse: the point of the cross-check is to notice that one of
#: them has a wall and the other does not, not to arbitrate their wording.
def _subject(kind: str, wants: dict) -> tuple:
    w = wants or {}
    # `capability` is the kind `intent.read` gives a feature the schema has **no family
    # for** -- a harbour, a windmill. It is the same subject as the reader's `feature`
    # for that thing, and treating the two as different subjects meant the rules'
    # `unsupported` verdict on it was recorded as a separate finding and then dropped
    # for sharing an id with the reading. A refusal that goes missing is the one failure
    # this whole case exists to prevent.
    if kind in ("feature", "absent", "capability"):
        return (kind if kind == "absent" else "feature", str(w.get("feature") or ""))
    if kind == "quality":
        return ("quality", f"{w.get('axis')}/{w.get('value')}")
    if kind == "layout":
        return ("layout", str(w.get("policy") or ""))
    if kind == "orientation":
        return ("orientation", str(w.get("faces") or ""))
    if kind == "setting":
        return ("setting", str(w.get("setting") or ""))
    if kind == "identity":
        return ("identity", str(w.get("name") or "").lower())
    if kind == "tradition":
        return ("tradition", str(w.get("tradition") or "").lower())
    if kind == "count":
        return ("count", "structures")
    return (kind, str(w.get("what") or w.get("function") or ""))


#: Which axes hold two values that cannot both be true of one scope. Used to tell a
#: disagreement from an addition: "dense" and "sparse" about the same district are the
#: same subject read two ways, and `density/dense` and `height/tall` are two subjects.
_OPPOSED = {"density": ("dense", "sparse"), "height": ("tall", "low")}


def _conflicts(key: tuple) -> list:
    """The subjects that would contradict `key` if the rules stated one of them."""
    kind, what = key
    if kind == "feature":
        return [("absent", what)]
    if kind == "absent":
        return [("feature", what)]
    if kind == "quality" and "/" in what:
        axis, value = what.split("/", 1)
        return [("quality", f"{axis}/{v}") for v in _OPPOSED.get(axis, ())
                if v != value]
    return []


def _count_class(what) -> str:
    """The class a count's unit word belongs to: every house word is `houses`; any other
    word is its own slug with a plural `s` dropped; empty where none was given."""
    w = _slug(str(what or ""))
    if not w:
        return ""
    if w in intent_mod._HOUSE_WORDS:
        return "houses"
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _value_disagreement(read: dict, rules: list) -> dict | None:
    """Where a reading and a rule about the same subject state different values.

        A count is `n` and whether it is `about`; a quality is its value on its axis (an
        opposed value is handled as a contradiction by `_conflicts`, so this is the same
        value read to a different bound). None where they agree or where there is no rule.
        
    """
    if not rules:
        return None
    w = read.get("wants") or {}
    q = rules[0]
    qw = q.get("wants") or {}
    if read.get("kind") == "count" and q.get("kind") == "count":
        # **The subject too.** "exactly 50 houses" read as fifty halls is the same
        # number about a different thing; the house words are one class, and any other
        # unit word is compared as itself (singular and plural alike).
        mine, theirs = _count_class(w.get("what")), _count_class(qw.get("what"))
        if mine and theirs and mine != theirs:
            return {"values": {"reader": {"n": w.get("n"), "about": bool(w.get("about")),
                                          "what": w.get("what")},
                               "rules": {"n": qw.get("n"), "about": bool(qw.get("about")),
                                         "what": qw.get("what")}},
                    "why": (f"the reader counts `{w.get('what')}` and the sentence "
                            f"counts `{qw.get('what')}`; a count is a count of the thing "
                            f"the sentence named, and the sentence's subject stands")}
        if int(w.get("n") or 0) != int(qw.get("n") or 0) \
                or bool(w.get("about")) != bool(qw.get("about")):
            return {"values": {"reader": {"n": w.get("n"), "about": bool(w.get("about"))},
                               "rules": {"n": qw.get("n"), "about": bool(qw.get("about"))}},
                    "why": (f"the reader says {w.get('n')} "
                            f"({'about' if w.get('about') else 'exactly'}) and the "
                            f"sentence's own number reads {qw.get('n')} "
                            f"({'about' if qw.get('about') else 'exactly'}); a number is "
                            f"read literally and the sentence's stands")}
    return None


def cross_check(sentence: str, interp: dict) -> dict:
    """Run the rules on the same sentence and compare. Returns the `checks` record.

        Four outcomes, and every one of them is kept:

          `agreed`      the rules and the reader say the same thing about a subject
          `rules_only`  the rules found something the reader did not say
          `read_only`   the reader said something the rules have no row for -- the ordinary
                        case for a relation, a hierarchy or a scoped contrast, and the whole
                        reason this stage exists
          `contradicts` they disagree about the same subject: a feature one requires and the
                        other says is absent. **Neither wins here.** It is reported, and the
                        requirement it produces is `unresolved` until somebody decides.

        `unsupported_phrase` is the other direction: a reading whose `phrase` is not in the
        sentence is refused outright. A reader that may quote anything can say anything.
        
    """
    rules = intent_mod.read(sentence)
    text = intent_mod._words(sentence)
    by_rule = {}
    for r in rules["requirements"]:
        if r["kind"] == "clause":
            continue
        by_rule.setdefault(_subject(r["kind"], r["wants"]), []).append(r)
    agreed, rules_only, read_only, contradicts, refused = [], [], [], [], []
    seen = set()
    for r in (interp or {}).get("reads") or []:
        phrase = str(r.get("phrase") or "").lower().strip()
        if phrase and intent_mod._span(text, phrase) is None:
            refused.append({"id": r["id"], "phrase": r.get("phrase"),
                            "why": ("this span is not in the sentence; a reading quotes "
                                    "what it read")})
            continue
        key = _subject(r["kind"], r.get("wants") or {})
        seen.add(key)
        # **What the rules say about the same subject in the other direction.** Both
        # cases are the same defect in the table -- it cannot represent a negation's
        # scope, so it reads "without a wall or a temple" as requiring a temple and "not
        # dense" as requiring dense. A rule contradicted this way is recorded and is not
        # carried on as a second, opposite requirement.
        others = [k for k in _conflicts(key) if k in by_rule and k not in seen]
        if others and key not in by_rule:
            contradicts.append({"id": r["id"], "subject": list(key),
                                "rules": [q["id"] for k in others for q in by_rule[k]],
                                "why": (f"the reader reads `{key[0]}/{key[1]}` and the "
                                        f"rules read "
                                        + ", ".join(f"`{k[0]}/{k[1]}`" for k in others))})
            seen.update(others)
            continue
        # **The same subject read to a different value is a disagreement, not an
        # agreement.** The review reproduced `Build exactly 50 houses.` interpreted as
        # one house with "one agreement, no contradiction": the count subject matched
        # and the numbers were never compared. A quoted span establishes where words
        # occurred, not that the reading preserves their meaning.
        differs = _value_disagreement(r, by_rule.get(key, []))
        if differs:
            contradicts.append({"id": r["id"], "subject": list(key),
                                "rules": [q["id"] for q in by_rule[key]],
                                "values": differs["values"], "why": differs["why"],
                                "rule_wins": True})
            continue
        (agreed if key in by_rule else read_only).append(
            {"id": r["id"], "subject": list(key),
             "rules": [q["id"] for q in by_rule.get(key, [])]})
    for key, rows in sorted(by_rule.items()):
        if key in seen:
            continue
        rules_only.append({"subject": list(key), "rules": [q["id"] for q in rows],
                           "says": rows[0]["says"]})
    return {"agreed": agreed, "rules_only": rules_only, "read_only": read_only,
            "contradicts": contradicts, "unsupported_phrase": refused,
            "rule_requirements": len(rules["requirements"]),
            "note": ("the phrase tables in `ethoslm.intent` were run on the same sentence; "
                     "they cross-check the reading and do not define it")}


def _completed(kind: str, wants: dict) -> dict:
    """Fill the parts of a `wants` that are facts about **this library**, not the request."""
    if kind == "tradition" and "nearest_form" not in wants:
        t = str(wants.get("tradition") or "").lower()
        if t in intent_mod.TRADITION_FORMS:
            wants["nearest_form"] = intent_mod.TRADITION_FORMS[t]
    if kind == "quality" and not wants.get("bound"):
        v = str(wants.get("value") or "")
        if v in intent_mod.QUALITY_BOUNDS:
            wants["bound"] = intent_mod.QUALITY_BOUNDS[v]
    if kind == "layout" and "supported" not in wants:
        p = str(wants.get("policy") or "")
        got = next((ok for name, ok, _ph in intent_mod.LAYOUTS if name == p), None)
        if got is not None:
            wants["supported"] = bool(got)
    if kind == "feature" and "family" not in wants:
        f = str(wants.get("feature") or "")
        got = next(((fam, True) for rid, fam, _ph in intent_mod.FEATURES if rid == f),
                   None)
        if got is not None:
            wants["family"] = got[0]
    return wants


def requirements(sentence: str, interp: dict | None, reading: dict | None = None) -> dict:
    """The intent record for a sentence **with** an interpretation of it.

        Without one this is `intent.read` exactly as it was, so every round that predates
        this stage reads the same. With one, the readings are the requirements -- scoped,
        related, reasoned -- the rules contribute whatever they found and the reader did not,
        contradictions are carried `unresolved` rather than decided, and the independent
        unclaimed-word check runs over both sets of spans together.
        
    """
    if not interp:
        return intent_mod.read(sentence)
    checks = cross_check(sentence, interp)
    refused = {r["id"] for r in checks["unsupported_phrase"]}
    contradicted = {c["id"] for c in checks["contradicts"] if not c.get("rule_wins")}
    # **A value the reader changed is refused as a disagreement and the sentence's own
    # number is carried.** See `_value_disagreement`: the reading is recorded in the
    # checks and not carried as a requirement; the rules' count is, marked as theirs,
    # with the reader's `what` (the thing counted) kept where the reader gave one.
    overruled = {c["id"]: c for c in checks["contradicts"] if c.get("rule_wins")}
    # **A claim id a reading cites has to be a claim the run actually made.** A reading
    # that cites research nobody did is inferred meaning wearing a citation, and it
    # cannot qualify anything. The bad ids are stripped, recorded, and the reading
    # stands on its own span.
    have = {}
    for c in (reading or {}).get("claims") or []:
        srcs = {x.get("id") for x in (reading or {}).get("sources") or []}
        if c.get("id"):
            # a claim that exists resolves, sourced or inferred; whether it can qualify
            # anything is the identity check's question (`intent.judgment_usable`)
            have[c["id"]] = True
    unsupported_evidence = []
    text = intent_mod._words(sentence)
    reqs, claimed = [], []
    # **What this build cannot express, by subject.** `intent.read` marks a feature with
    # no family, a layout with no policy and a tradition with no form family
    # `unsupported`, and that is a statement about *this library* -- not about the
    # sentence, and so not something a reader of the sentence has any standing to
    # overrule. Without this the interpreter's own reading of "a windmill" replaced the
    # rules' "the schema has no family for one" with a plain open requirement, and the
    # request that the whole refusal case is about quietly became buildable.
    rules_all = intent_mod.read(sentence)
    cannot = {_subject(q["kind"], q["wants"]): q for q in rules_all["requirements"]
              if q["status"] == "unsupported"}

    def span_of(phrase):
        at = intent_mod._span(text, str(phrase).lower()) if phrase else None
        if at is not None:
            claimed.append(at)
        return at

    for r in (interp or {}).get("reads") or []:
        if r["id"] in refused:
            continue
        if r["id"] in overruled:
            span_of(r.get("phrase"))
            continue
        span_of(r.get("phrase"))
        status, why = "open", str(r.get("why") or "")
        evidence = []
        for cid in list(r.get("evidence") or []):
            if cid in have and have[cid]:
                evidence.append(cid)
            else:
                unsupported_evidence.append({"id": r["id"], "claim": cid,
                                             "why": "no claim of this id was read for "
                                                    "this request"})
        if reading is None:
            evidence = list(r.get("evidence") or [])
        if r["id"] in contradicted:
            # **The reader wins this one, and the reason is about the instruments.** A
            # `feature`/`absent` disagreement about the same feature is a disagreement
            # about *scope* -- "without a wall or a temple" -- and the phrase table has
            # no way to express scope at all: it matches "without a wall", then matches
            # "temple" further along and states a requirement for one. An instrument
            # that cannot represent the distinction being argued about does not get a
            # vote on it. So the reading stands and the disagreement stays on the
            # record, where `checks.contradicts` carries it and the resolution reports
            # it; what is refused is a reader overriding a rule *silently*.
            why = (f"{why} -- the phrase rules read this span the other way round, "
                   f"having no way to represent the scope of a negation; the "
                   f"disagreement is recorded in the interpretation's own checks")
        wants = _completed(r["kind"], dict(r.get("wants") or {}))
        no_family = cannot.get(_subject(r["kind"], wants))
        if no_family is not None:
            status = "unsupported"
            why = (f"{why} -- and this build cannot express it: "
                   f"{no_family.get('why') or 'no family, policy or form family for it'}"
                   f". The reading stands and the capability does not")
            wants = {**no_family["wants"], **wants}
        reqs.append({"id": r["id"], "says": str(r.get("says") or r["id"]),
                     "kind": r["kind"], "wants": wants,
                     "hard": bool(r.get("hard", True)), "source": "interpretation",
                     "phrase": r.get("phrase"), "scope": r.get("scope"),
                     "status": status, "why": why, "evidence": evidence,
                     "owner": None})
    # what the rules found and the reader did not say: kept, and marked as the rules'
    rules = intent_mod.read(sentence)
    want = {tuple(x["subject"]) for x in checks["rules_only"]}
    want |= {tuple(c["subject"]) for c in overruled.values()}
    have_ids = {q["id"] for q in reqs}
    reads_by_id = {r["id"]: r for r in (interp or {}).get("reads") or []}
    for q in rules["requirements"]:
        if q["kind"] == "clause":
            continue
        if _subject(q["kind"], q["wants"]) not in want or q["id"] in have_ids:
            continue
        span_of(q.get("phrase"))
        over = next((c for c in overruled.values() if q["id"] in c["rules"]), None)
        if over is not None:
            rd = reads_by_id.get(over["id"]) or {}
            wants = dict(q["wants"])
            if (rd.get("wants") or {}).get("what") and not wants.get("what"):
                wants["what"] = rd["wants"]["what"]        # the sentence's own stands
            reqs.append({**q, "wants": wants, "source": "rules", "scope": rd.get("scope"),
                         "why": (q.get("why") or "") + (
                             f" -- the sentence's own number, kept over the reading "
                             f"`{over['id']}` which read it as {over['values']['reader']}"
                             f": {over['why']}")})
            continue
        reqs.append({**q, "source": "rules", "scope": None,
                     "why": (q.get("why") or "") + (
                         " -- read from the sentence by rule; the interpretation did "
                         "not state it, and a rule that finds a requirement the reader "
                         "missed is still right about the requirement")})
    if unsupported_evidence:
        checks["unsupported_evidence"] = unsupported_evidence
    # ...and every content word neither of them claimed. The independent check, over
    # both
    for w, at in intent_mod.unclaimed_words(sentence, claimed):
        reqs.append({"id": f"clause/{_slug(w)}", "says": f"the sentence says {w!r}",
                     "kind": "clause", "wants": {"word": w}, "hard": True,
                     "source": "sentence", "phrase": w, "scope": None,
                     "status": "unresolved", "evidence": [], "owner": "reading",
                     "why": ("neither the interpretation nor any rule in this reader "
                             "claims this word, so it is carried as an unread clause "
                             "rather than dropped; whatever it asks for is neither "
                             "planned nor refused")})
    # ids are what findings link back to, and two readers can name the same thing
    out, taken = [], set()
    for q in reqs:
        q = dict(q)
        if q["id"] in taken:
            q["id"] = f"{q['id']}~{q['source']}"
        taken.add(q["id"])
        out.append(q)
    return contracts.make(
        "intent", sentence=str(sentence), requirements=out,
        note=(f"read by an agent interpretation ({(interp or {}).get('source')}) and "
              f"cross-checked against the phrase rules: {len(checks['agreed'])} agreed, "
              f"{len(checks['read_only'])} read only, {len(checks['rules_only'])} by "
              f"rule only, {len(checks['contradicts'])} contradicted, "
              f"{len(checks['unsupported_phrase'])} refused for quoting what the "
              f"sentence does not say"
              + (f", {len(unsupported_evidence)} citation(s) of claims the run never "
                 f"made stripped" if unsupported_evidence else "")))


BRIEF = """# Read this sentence into a place programme

> {sentence}

You are the **interpreter**. Nothing else in this system reads the sentence for meaning:
what you write here is what the design, the construction and the final check are held
to. A separate set of phrase rules reads the same sentence and cross-checks you; where
you and it disagree, both are recorded and neither wins.

Read what the sentence **says**, not what would make a nice place. Where it implies
something, say so and mark it `"hard": false`. Where you cannot tell, put it in
`uncertain` rather than guessing.

## What must survive

- **Scope.** "a dense lower district and a sparse upper district" is two readings with
  two different `scope`s, not one global "dense". `scope` is the part of the place a
  reading is about -- use the words the sentence uses for it -- or `null` for the whole.
- **Negation, scoped.** "without a wall or a temple" makes the wall absent **and** the
  temple absent. A negation that reaches one conjunct and not the other is the single
  most common way this system has mis-read a request.
- **Quantity.** How many, and whether the sentence is exact ("eight houses") or
  approximate ("about sixty").
- **Relation and hierarchy.** "small houses around a big temple" says the temple is
  greater than the houses (`hierarchy`) and that they stand around it (`relation`).
  These are measurable and they are checked.
- **Function.** What a part is *for* -- a working waterfront, a dwelling, a market. A
  type whose role or tradition label happens to match does not fulfil a function.
- **Identity, tradition, setting** where the sentence names them.

## The readings you may write

{kinds}

`relation` takes `{{"subject": ..., "relation": ..., "object": ...}}` where the relation
is one of {relations}. `hierarchy` takes `{{"greater": ..., "lesser": ..., "axis": ...}}`
with the axis `size` or `height`. `quality` takes `{{"axis": ..., "value": ...}}` with
the axis `density` or `height`. `absent` and `feature` take `{{"feature": ...}}`;
`count` takes `{{"n": ..., "about": true|false}}`.

{evidence}

## Output

Write a single JSON file to {out}:

```json
{{"reads": [{{"id": "...", "kind": "...", "says": "...", "wants": {{...}},
             "scope": null, "phrase": "the span of the sentence you read it from",
             "hard": true, "why": "why the sentence says this"}}],
  "unread": ["any word or phrase you could not interpret"],
  "uncertain": ["what you could not decide, and why"]}}
```

`phrase` must be a span that actually appears in the sentence -- a reading that quotes
what the sentence does not say is refused. `why` is required on every reading: it is
what makes an interpretation revisable rather than a rule nobody may question.
"""


def brief(sentence: str, out_path: str, reading: dict | None = None) -> str:
    """The interpreter's brief: the sentence, the vocabulary and whatever was sourced."""
    kinds = "\n".join(f"- `{k}`" for k in contracts.INTERPRETATION_KINDS)
    ev = ""
    claims = [c for c in (reading or {}).get("claims") or [] if c.get("says")]
    if claims:
        ev = ("## What has been read about this request\n\n"
              + "\n".join(f"- {c['says']}"
                          + (f"  [{c.get('source')}]" if c.get("source") else
                             "  (inferred, no source)")
                          for c in claims[:24])
              + "\n\nUse this to interpret the sentence. It does not replace the "
                "sentence and it cannot add a requirement the sentence does not make.\n")
    return BRIEF.format(sentence=sentence, out=out_path, kinds=kinds, evidence=ev,
                        relations=", ".join(f"`{r}`" for r in contracts.RELATIONS))


def load(path: str, sentence: str, *, source: str = "agent") -> dict:
    """An interpretation off disk, checked. Raises for an answer that will not read."""
    return read_answer(sentence, json.load(open(path)), source=source)
