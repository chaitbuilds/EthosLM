"""What each artifact was made from, so a stale one can be told from a warm one.

The audit's seventh finding, reproduced in a temporary round directory: change a village
spec into a much larger city and `stage_site_search` returns the site it chose for the
village, because the stage's test for "already done" is `os.path.exists`. Change the
plan and `stage_preview` returns the drawing it made of the old one, because its test is
a `done` flag in its own record. Neither stage is *wrong* to reuse work.

This module is the missing question. An artifact records the **fingerprint of its
inputs**; a stage that finds one asks whether those inputs still fingerprint the same
way; a mismatch invalidates that artifact and everything downstream of it.

`fingerprint(rnd, *kinds)` is the whole interface. `check(rnd, artifact)` answers
`(fresh, why)`, and `stamp(rnd, artifact)` writes the current one. Nothing here deletes
anything: a stale artifact is *reported* stale and the stage that owns it decides, which
is what keeps an intentionally pinned fixture pinned.

An **interrupted** run is the other half. A `done` marker written before the file it
describes is a lie the next run believes, so `stamp` writes the marker last and `check`
refuses a stamp whose named outputs are missing.
"""
from __future__ import annotations

import hashlib
import json
import os

from . import contracts

#: The dependency kinds, and the order they are reported in. **`terrain` and `prepared`
#: are two different grounds and the review's last finding is that they were one.**
#: `terrain` is the ground as it was found -- the baseline, an *input* to every design
#: decision. `prepared` is the ground this candidate's own plateau and terraces cut into
#: it -- an *output* of the accepted design. While both read `world.npz` a run that
#: prepared its ground invalidated the plan that asked for the preparation, which the
#: integration report records happening to the shoreline case, and the only ways out of
#: a circle like that are to stop checking or to stop preparing. `sentence` is the
#: realization round's, and it exists because `intent` stopped being a stable key for
#: the stages that run *before* the meaning is read. The interpret stage rewrites
#: `intent.json` from the agent's reading, so an artifact keyed on the requirement ids
#: -- the reading of the sources, the interpretation itself -- would go stale the moment
#: the thing it produced was written, and the two stages would take turns invalidating
#: each other for ever. What those artifacts actually depend on is the request, which
#: never changes. `site_demand` is the design round's, and it exists because `spec` is
#: the fingerprint of everything a spec says while the site search consumes a tenth of
#: it. The expression report records a 25-minute search re-run twice on the city and
#: choosing the same square, because a voice revision moved `_spec_print`. What a search
#: is actually scored against is `SITE_DEMAND_SPEC` and `SITE_DEMAND_PART` below; a
#: change to any of those still invalidates it, and a change to a voice, a character or
#: a district's prose does not. `ground_proposal` is the other half of contract 6: the
#: prepared ground is an artifact of the **proposal**, so a design change that leaves
#: the proposal identical leaves the cut warm.
KINDS = ("sentence", "intent", "spec", "site_demand", "site", "terrain", "prepared",
         "schema", "types", "model", "reading", "ground_proposal")

#: Where the ground as it was found is kept. The plateau stage has always written this
#: before cutting; it is now also what `terrain` is the identity of.
BASELINE = ".before-plateau.npz"


def baseline_path(rnd) -> str:
    was = rnd.rel(rnd.base_volume.replace(".npz", BASELINE))
    return was if os.path.exists(was) else rnd.rel(rnd.base_volume)

#: Which artifact depends on what. The one table; a stage reads its own row. An artifact
#: not named here has no dependency contract yet and is reused as it always was --
#: recorded, so "this one is not keyed" is a fact rather than a silence.
DEPENDS = {
    # **What the search consumes, and not what the spec says.** The design round's fifth
    # contract. `scripts/find_site.py` scores a square against `search_needs` -- the
    # place's needs, the core part's and the outer parts', the setting, the plateau the
    # centre asks for and the ring shares a designed ground is terraced to -- and
    # against nothing else. `intent` stays because a requirement can change what the
    # place is for; `spec` goes, because it carried the voice, the characters and every
    # district's notes into a stage that reads none of them.
    "site_search": ("intent", "site_demand", "schema"),
    "plateau": ("intent", "spec", "site", "terrain", "schema"),
    "plan": ("intent", "spec", "site", "terrain", "schema", "types"),
    # **The resolution is made out of the plan.** The review invalidated `plan` and
    # found `resolution` still warm, because this row named every input the plan has and
    # not the plan itself -- so a design that had been withdrawn kept a record
    # describing it, and `findings.json` beside that record went on saying what was
    # wrong with a plan that no longer existed.
    "resolution": ("intent", "spec", "site", "terrain", "schema", "types", "plan"),
    # **The prepared ground is an artifact of the design, and is checked before anything
    # is built on it.** "Prepare ground from this design and check the result before
    # construction": the preparation reads the baseline and the **proposal**, so those
    # are its inputs, and the volume it writes is its output. The design round keys it
    # on `ground_proposal` rather than on the whole plan: the cut is made from the
    # proposal, and a plan change that leaves the proposal identical -- a voice, a
    # leaf's parameters -- did not change the ground. Until `ground.propose` writes one,
    # `ground_proposal` falls back to the plan's identity, so this row means exactly
    # what it meant before.
    "ground": ("intent", "site", "terrain", "schema", "ground_proposal"),
    "preview": ("intent", "spec", "site", "plan", "schema", "types", "model"),
    "reading": ("sentence", "schema", "model"),
    "capabilities": ("intent", "spec", "schema", "types"),
    # **The interpretation is of the sentence and of what was read about it.** Not of
    # the spec: a programme designed from a reading cannot be one of its inputs, and
    # keying it on the spec would make every scale negotiation re-ask what the sentence
    # means. `schema` is here because the record's own vocabulary is part of what an
    # answer was written against. **...and of what was read about it.** The review's
    # fifth finding: changing the saved research input left the interpretation warm.
    # `reading` is the content of `reading.json` -- its sources' fingerprints and its
    # claims -- not the file.
    "interpretation": ("sentence", "reading", "schema", "model"),
    # the closure round's reading of the built world: of the plan, the types it was
    # built with and the model that read it
    "inspection": ("intent", "spec", "site", "plan", "schema", "types", "model"),
    # the closure round: the assembled tree and the plot registry are outputs of their
    # own artifact, so an interrupted write of `plan.json` is stale and not warm
    "assembled": ("intent", "spec", "site", "terrain", "schema", "types"),
    # the built world: of the assembled plan, the types it was built with and the
    # prepared ground it was built on (the closure round) **...and the prepared ground
    # is now actually named.** The composition round: this row said "and the prepared
    # ground it was built on" and named `terrain`, which is the baseline -- the ground
    # as it was *found*. Nothing here moved when a candidate's own plateau, terraces and
    # paving changed, so `stage_parts` could return a warm build across a ground change:
    # its warm return precedes `ground_for_this_design`, which is the only thing that
    # asked, and even that asks about the `ground` artifact and not about this one.
    # `prepared` is the baseline and the proposal cut into it; see `fingerprint`.
    "built": ("intent", "spec", "site", "terrain", "prepared", "schema", "types", "plan",
              "model"),
}

#: **Which artifacts a narrow fingerprint kind is made of**, so `invalidate` cascades
#: through the kind as well as through the artifact name. The design round narrowed two
#: rows: `site_search` keys on `site_demand` where it used to key on the whole `spec`,
#: and `ground` keys on `ground_proposal` where it used to key on the whole `plan`.
#: `invalidate` drops the artifacts that name the withdrawn one in `DEPENDS`, so without
#: this table withdrawing the plan would stop withdrawing the ground that was cut for it
#: -- the narrower key would have bought proportional iteration at the price of a
#: cascade the last round put in on purpose.
DERIVED_FROM = {"ground_proposal": ("plan", "ground"),
                # `prepared` is the proposal and the baseline, so withdrawing either the
                # plan or the ground withdraws every artifact built on the prepared
                # ground -- which is `built`, and is the composition round's point
                "prepared": ("plan", "ground"),
                "site_demand": ("spec",)}

#: Where the stamps live inside a round's state.
RECORD = "deps.json"

#: The key inside that record holding the artifacts this run **withdrew**, and why. Not
#: a stamp and never read as one; see `invalidate`.
INVALID = "invalidated"


#: The fields of a defining part that are comment: changing one changes nothing a
#: downstream artifact was made from. Written as exclusions, like `PLAN_PROSE`.
SPEC_PART_PROSE = frozenset(("purpose", "why", "comment", "description",
                             # a marker of how `structures` was filled, not a decision;
                             # `structures` itself is hashed (the closure round)
                             "structures_inferred"))


def _spec_print(rnd) -> str:
    for f in ("place.checked.json", "place.json"):
        p = rnd.rel(f)
        if os.path.exists(p):
            doc = json.load(open(p))
            # the fields a downstream artifact actually depends on, not the prose: re-
            # wording a district's notes does not invalidate a site search.
            keep = {k: doc.get(k) for k in
                    ("kind", "structures", "size_band", "needs", "setting", "form",
                     "voice", "explicit_count", "unmet",
                     # **the prose the geometry is read out of.** `invariants` is where
                     # a spec says its rings are round (`placeplan.wall_round_for`) and
                     # what mass its wall is; excluding it as "prose" meant a change
                     # that moves every boundary in the place left every artifact warm.
                     "invariants", "sentence")}
            # **Every field of a defining part, less the prose.** The review changed a
            # district's `land_use` from settled to farmland -- which moves the ground
            # cover the compiler works to and the target its validator refuses on -- and
            # this fingerprint did not move, because the field was not on a list
            # somebody had remembered to extend. A hash of the fields somebody thought
            # of is a hash of the fields somebody thought of, so the rule here is the
            # one `PLAN_PROSE` already uses one level down: everything is a dependency
            # until it is argued out. `notes` stays hashed on purpose -- a part's notes
            # are read for its wall's mass and its roundness, so they are geometry and
            # not comment.
            keep["parts"] = [_stable({k: v for k, v in p.items()
                                      if k not in SPEC_PART_PROSE})
                             for p in (doc.get("defining_parts") or [])]
            return contracts.digest(keep)
    return "none"


#: **What the site search is scored against, at the place level.** Read off
#: `scripts/find_site.py`: `search_needs` reads `spec["needs"]` and `spec["kind"]` (the
#: relief band) and `spec["setting"]`; `core_size` reads the plateau the centre asks
#: for; the size band is what a failed search drops one of. Written as an **inclusion**
#: list and not an exclusion list, unlike `_spec_print`, because the whole point of this
#: kind is that it is narrower than the spec: a field added to the spec must be argued
#: *into* a site demand rather than silently made one.
SITE_DEMAND_SPEC = ("kind", "needs", "setting", "size_band", "structures",
                    "explicit_count")

#: ...and per defining part. `needs` and `relation` are what `spec.core_needs`,
#: `spec.outer_parts` and `find_site.core_size` fold in; `ring`/`share`/`walled` are
#: what `find_site.designed_layout` terraces to; `kind`/`family`/`count`/`structures`
#: are what `placeplan.compound_ground` sizes a compound's plateau from. `character`,
#: `voice`, `density`, `role`, `notes` and `forms` are absent on purpose.
SITE_DEMAND_PART = ("name", "kind", "family", "relation", "of", "count", "structures",
                    "needs", "ring", "share", "walled")


def _site_demand_print(rnd) -> str:
    """The fingerprint of what the **search** consumes, and of nothing else."""
    for f in ("place.checked.json", "place.json"):
        p = rnd.rel(f)
        if os.path.exists(p):
            doc = json.load(open(p))
            keep = {k: _stable(doc.get(k)) for k in SITE_DEMAND_SPEC}
            keep["parts"] = [_stable({k: v for k, v in p.items()
                                      if k in SITE_DEMAND_PART})
                             for p in (doc.get("defining_parts") or [])]
            return contracts.digest(keep)
    return "none"


#: Where a ground proposal lives, and what its identity is inside it. **The file
#: `stage_ground` actually writes, and the print the proposal actually computes.** The
#: composition round's sixth evidence connection, and this constant was wrong in both
#: halves: it named `ground.json` while `stage_ground` writes `ground_proposal.json`
#: (`stages_plan.stage_ground`), and `ground.json` is a *live, different* artifact --
#: `stages_build.settle_ground`'s declaration record, which `ARTIFACT_FILES["ground"]`
#: also names. Of the seven keys this used to hash, only `levels` exists in what
#: `ground.propose` writes, so `got` was almost always a one-key dict or empty and the
#: fingerprint fell through to the plan's -- which is the very thing keying the cut on
#: the proposal was for. `ground.propose` already computes its own identity:
#: `proposal["print"]` is the digest of the baseline's print, every piece's label,
#: rectangle, level and voice, and every part's occupied envelope (`ground.py`). That is
#: what the cut is made from and it is what this kind is now keyed on.
#: `GROUND_PROPOSAL_KEYS` stays as the fallback for a proposal document written before
#: `print` existed, with the keys the record really has.
GROUND_PROPOSAL_FILE = "ground_proposal.json"
GROUND_PROPOSAL_PRINT = "print"
GROUND_PROPOSAL_KEYS = ("baseline", "levels", "pieces", "protected", "occupied",
                        "allocation", "site", "voice")

#: ...and where `settle_ground` writes its own record, which is not a proposal and is
#: not read here. Named so that the two documents cannot be confused again.
GROUND_DECLARED_FILE = "ground.json"


def _ground_proposal_print(rnd, plan=None) -> str:
    """The identity of the ground **proposal** this candidate's cut was made from.

        The proposal and not the plan: a revision that changes a leaf's parameters or a
        voice did not change the ground, and re-cutting it from the baseline for that is
        work proportional to the record rather than to the change. A round with no proposal
        document falls back to the plan's own fingerprint, which is what `DEPENDS["ground"]`
        named before this kind existed.
        
    """
    p = rnd.rel(GROUND_PROPOSAL_FILE)
    if os.path.exists(p):
        try:
            doc = json.load(open(p))
        except (OSError, ValueError):
            doc = None
        if isinstance(doc, dict):
            if doc.get(GROUND_PROPOSAL_PRINT):
                return contracts.digest("ground_proposal",
                                        doc[GROUND_PROPOSAL_PRINT])
            got = {k: _stable(doc[k]) for k in GROUND_PROPOSAL_KEYS if k in doc}
            if got:
                return contracts.digest(got)
    return contracts.digest("no proposal; the plan is the proposal",
                            fingerprint(rnd, ("plan",), plan=plan)["plan"])


#: `{(path, size, mtime_ns): sha256}` -- the content digest of a file, remembered for as
#: long as the process lives. A fingerprint is asked for several times a stage and a
#: base volume is tens of megabytes; hashing it once per run and not once per question
#: is the difference between a cheap check and a reason not to check.
_CONTENT: dict = {}


def content_print(path: str) -> str | None:
    """The sha256 of a file's bytes, or None where it is not there.

        **A size is not an identity.** The integration review's fifth finding, twice: a
        type file whose semantics changed without its byte length changing fingerprinted
        the same (`HEIGHT = 10` -> `HEIGHT = 90`), and a terrain volume was keyed by its
        file size. Both are "the cheap thing that is usually right", and the whole purpose
        of this module is to be right when it matters rather than usually.
        
    """
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    got = _CONTENT.get(key)
    if got is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = _CONTENT[key] = h.hexdigest()[:32]
    return got


def _types_print(rnd) -> str:
    from . import pipeline as _pipeline
    rows = []
    for sub, ext in (("types", ".py"), ("voices", ".json")):
        root = os.path.join(_pipeline.ROOT, sub)
        for f in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if f.endswith(ext):
                rows.append((f, content_print(os.path.join(root, f))))
    return contracts.digest(rows)


def _schema_print() -> str:
    return contracts.digest({k: v[0] for k, v in contracts.VERSIONS.items()},
                            SCHEMA_REVISION)


#: Bumped by hand when a change to this build would make an artifact on disk mean
#: something different. Cheap, explicit, and the one thing a content hash of the source
#: tree cannot be: a *decision* that the old artifacts are still good. A name and not a
#: date, so the public tree carries no round number. Bumped by the unification round.
#: Several artifacts on disk mean something different under it: a plan carries a
#: `fabric_types` pool and an `extent_from` rectangle, a resolution keeps promised,
#: allocated and realized apart, a capability entry records what was `used`, and a
#: compiled district was chosen against a ground-cover target that a settled district is
#: no longer held to. A round that kept its old documents would be reading all four as
#: though they still said what they used to. Bumped by the realization round. Artifacts
#: on disk mean something different under it: an intent requirement carries a `scope`
#: and may be a `relation`, a `hierarchy` or a `function`; a district's promise is the
#: arrangement its ground actually holds rather than an area estimate; a stamp binds the
#: identity of its outputs and not only of its inputs; and an import authorises the
#: files it names rather than a directory. Bumped by the design round. Four artifacts on
#: disk mean something different under it: `envelopes.json` is keyed on the generator
#: that produced each answer and not only on its name, so an entry from before it is an
#: answer to a question that cannot be recovered; `site_search.json` is keyed on what
#: the search consumes rather than on the whole spec, so a stamp made under the old row
#: would hold a site warm against a change this one invalidates and stale against
#: changes it does not; a construction outcome's `features_source` distinguishes a
#: feature observed in the assembled world from one the type declared; and
#: `obligations.json` is the ledger findings and emitted constraints now share, so a
#: dispositions file alone no longer says what is owed. **Not** bumped by the
#: composition round, and that is a decision rather than an omission. Its record changes
#: are additive: a part's row gains `emitted.required` and `emitted.owed`,
#: `features_source` gains `not_identified`, an obligation gains `seen_on`,
#: `closed_against` and `reopened`, and a `built` stamp gains the `prepared` kind. Every
#: reader of an older document gets `None` from the new fields and behaves as it did,
#: and `check` compares only the kinds a stamp actually recorded -- so a stamp made
#: before `prepared` existed stays warm on the kinds it named.
SCHEMA_REVISION = "records-v4-design"


#: The fields of a plan leaf that are prose: changing one changes nothing that is built.
#: Everything else is a build input and is hashed. Written as a list of exclusions on
#: purpose -- a field added to the planner is hashed by default and has to be argued out
#: of the fingerprint rather than argued into it.
PLAN_PROSE = frozenset(("notes", "note", "why", "comment", "children", "parts"))


def _stable(v):
    """A JSON value in a shape `digest` orders the same way every time."""
    if isinstance(v, dict):
        return sorted((str(k), _stable(x)) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return [_stable(x) for x in v]
    return v


def fingerprint(rnd, kinds=KINDS, *, plan=None) -> dict:
    """The current fingerprint of each named dependency kind."""
    out = {}
    for k in kinds:
        if k == "sentence":
            out[k] = contracts.digest(rnd.sentence or "")
        elif k == "intent":
            # **The content of every requirement, not its id.** The review changed a
            # requirement from global dense to upper-quarter sparse under the same id
            # and the candidate identity did not move. What a design was made from is
            # what the requirement *says* -- kind, wants, scope, whether it is hard --
            # and not the label; and not its `status`, `why` or `evidence`, which the
            # checks rewrite on every pass and which describe the design rather than the
            # ask.
            rec = contracts.load(rnd, "intent")
            out[k] = contracts.digest(rnd.sentence or "", sorted(
                (r["id"], r.get("kind"), _stable(r.get("wants") or {}),
                 r.get("scope"), bool(r.get("hard", True)), r.get("says"),
                 r.get("phrase"), r.get("source"))
                for r in (rec or {}).get("requirements") or []))
        elif k == "reading":
            rec = contracts.load(rnd, "reading")
            out[k] = contracts.digest(
                (rec or {}).get("classification"),
                sorted((s.get("id"), s.get("fingerprint"))
                       for s in (rec or {}).get("sources") or []),
                sorted((c.get("id"), c.get("says"), c.get("source"))
                       for c in (rec or {}).get("claims") or []),
                list((rec or {}).get("uncertainty") or []))
        elif k == "spec":
            out[k] = _spec_print(rnd)
        elif k == "site_demand":
            out[k] = _site_demand_print(rnd)
        elif k == "ground_proposal":
            out[k] = _ground_proposal_print(rnd, plan=plan)
        elif k == "site":
            s = rnd.site or rnd.chosen_site()
            if not s and os.path.exists(rnd.rel("site.json")):
                s = json.load(open(rnd.rel("site.json")))
            out[k] = contracts.digest(s and [s["origin"], s["size"]])
        elif k == "terrain":
            # the ground as it was found, not as this candidate cut it
            p = baseline_path(rnd)
            out[k] = contracts.digest(os.path.basename(rnd.base_volume),
                                      content_print(p))
        elif k == "prepared":
            # **The ground as this candidate's own cut left it.** The composition round:
            # this kind was computed here, documented at the top of the module, and
            # named by `KINDS` and by no `DEPENDS` row -- a dependency nothing could
            # request, with a docstring arguing for it. It was also the wrong
            # measurement: the digest of the *working* volume, which construction then
            # writes into, so an artifact keyed on it would have gone stale the moment
            # anything was built on the ground it describes. What a prepared ground is,
            # is the baseline plus the proposal that was cut into it. Both of those are
            # stable across construction, and either moving is exactly the change that
            # must not be reused: `DEPENDS["built"]` names this, so a warm build cannot
            # be returned across a ground change.
            out[k] = contracts.digest(
                "prepared", _ground_proposal_print(rnd, plan=plan),
                contracts.digest(os.path.basename(rnd.base_volume),
                                 content_print(baseline_path(rnd))))
        elif k == "schema":
            out[k] = _schema_print()
        elif k == "types":
            out[k] = _types_print(rnd)
        elif k == "model":
            from . import pipeline as _pipeline
            # the routing table's **content**, not merely that one is configured: a
            # different model behind the same environment variable is a different run.
            out[k] = contracts.digest(
                {n: os.environ.get(n) for n in sorted(os.environ)
                 if n.startswith("ETHOSLM_MODEL") or n.startswith("ETHOSLM_RETRIEVAL")},
                content_print(os.path.join(_pipeline.ROOT, "models.json")),
                rnd.flags.get("seed"))
        elif k == "plan":
            doc = plan if plan is not None else rnd.plan()
            # **Every leaf of the tree, not the top-level list.** A plan is a tree and
            # `doc["parts"]` is its first level; the review changed a nested plot's
            # extent from 10 to 50 columns and the fingerprint did not move, because the
            # plot was two levels down. `plan_parts` is the flattening every other
            # reader of a plan already uses.
            from . import pipeline as _pipeline2
            leaves = _pipeline2.plan_parts(doc or {})
            # **Every field the builder reads, not a list somebody remembered.** The
            # review changed a leaf's `attached` -- which `buildlib` consumes to decide
            # whether a house has party walls -- and the fingerprint did not move. A
            # hash of an incomplete field list is a hash of the fields somebody thought
            # of, so the rule here is the authoritative one: everything except the
            # fields that are demonstrably prose.
            out[k] = contracts.digest(
                [sorted((str(f), _stable(v)) for f, v in p.items()
                        if f not in PLAN_PROSE)
                 for p in leaves],
                (doc or {}).get("voice"))
    return out


def _load(rnd) -> dict:
    p = rnd.rel(RECORD)
    return json.load(open(p)) if os.path.exists(p) else {}


def stamp(rnd, artifact: str, *, outputs=(), plan=None, note: str = "") -> dict:
    """Record what `artifact` was made from. **Written last**, after its outputs exist.

        `outputs` are the files the artifact claims to have produced; `check` refuses a
        stamp whose outputs are gone, which is what an interrupted run leaves behind.
        
    """
    rec = _load(rnd)
    kinds = DEPENDS.get(artifact, KINDS)
    missing = _unusable(rnd, outputs)
    if missing:
        raise ValueError(f"{artifact}: refusing to stamp before its outputs exist "
                         f"({[o for o, _w in missing]}). A done marker written first "
                         f"is a done marker the next run believes")
    rec[artifact] = {"inputs": fingerprint(rnd, kinds, plan=plan),
                     "outputs": list(outputs),
                     # **What was accepted, not merely what was depended on.** The
                     # review replaced a stamped `plan.json` with `{"parts": []}` and a
                     # stamped prepared volume with unrelated bytes, and both artifacts
                     # stayed warm: every *input* identity was unchanged and nothing
                     # ever asked whether the output was still the document that was
                     # accepted. An artifact is its inputs and its outputs, and this is
                     # the half that was missing.
                     "output_digests": {o: content_print(rnd.rel(o))
                                        for o in (outputs or [])},
                     "note": note, "schema": SCHEMA_REVISION}
    # made again: the tombstone goes, because this is a new artifact and not the one
    # that was withdrawn
    if artifact in (rec.get(INVALID) or {}):
        rec[INVALID] = {k: v for k, v in rec[INVALID].items() if k != artifact}
    os.makedirs(rnd.state, exist_ok=True)
    with open(rnd.rel(RECORD), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec[artifact]


#: What each named output has to carry to be that output at all. One key from the list
#: is enough: the point is to tell a document from a husk, not to re-validate a record
#: `contracts` already reads.
OUTPUT_KEYS = {
    "plan.json": ("parts", "levels"),
    "plan.place.json": ("parts", "districts", "compounds"),
    "resolution.json": ("regions", "policy"),
    "findings.json": ("findings",),
    "capabilities.json": ("entries",),
    "preview.json": ("drawn", "done", "revisions"),
}


def _unusable(rnd, outputs) -> list:
    """The named outputs that are missing, empty, or not the document they claim to be.

        **A file that is there is not an output.** The review interrupted a run by
        truncating `plan.json` to nothing and `check` called the artifact warm, because the
        test was `os.path.exists`. A zero-byte plan and a half-written one are exactly what
        an interrupted run leaves, which is the state this whole module exists to notice.
        
    """
    bad = []
    for o in outputs or []:
        p = rnd.rel(o)
        if not os.path.exists(p):
            bad.append((o, "is not there"))
            continue
        if os.path.getsize(p) == 0:
            bad.append((o, "is zero bytes"))
            continue
        if o.endswith(".json"):
            try:
                with open(p) as fh:
                    doc = json.load(fh)
            except Exception as e:             # noqa: BLE001 -- reported, not raised
                bad.append((o, f"does not read as JSON ({type(e).__name__})"))
                continue
            # **Parsing is not content.** The review replaced a stamped `plan.json` with
            # `{}` and the artifact stayed warm: the identities of its *inputs* were all
            # unchanged, and nothing ever asked whether the output still held the
            # document it claimed to be. An empty object parses, and an empty plan is
            # exactly what a crashed writer leaves behind.
            if not doc:
                bad.append((o, "holds an empty document"))
                continue
            want = OUTPUT_KEYS.get(os.path.basename(o))
            if want and not any(k in doc for k in want):
                bad.append((o, f"carries none of {', '.join(want)}, so it is not the "
                               f"document the stamp claims"))
    return bad


def check(rnd, artifact: str, *, plan=None) -> tuple:
    """`(fresh, why)` -- may this artifact be reused?

        `fresh` is False when there is no stamp, when the artifact is known to have been
        invalidated, when an output named by the stamp is missing or unreadable, or when any
        dependency's fingerprint has moved. `why` names the kind that moved, which is the
        difference between "rerun it" and "rerun everything".
        
    """
    doc = _load(rnd)
    rec = doc.get(artifact)
    if not rec:
        if artifact in (doc.get(INVALID) or {}):
            return (False, f"{artifact} was invalidated ("
                           f"{doc[INVALID][artifact]}) and has not been made again; "
                           f"a withdrawn claim is not a missing one")
        return (False, f"{artifact} carries no dependency stamp: what it was made from "
                       f"is unknown, so it is not reused")
    bad = _unusable(rnd, rec.get("outputs"))
    if bad:
        return (False, f"{artifact} was stamped with output(s) "
                       + "; ".join(f"{o} which {w}" for o, w in bad)
                       + ": an interrupted run, not a finished one")
    # **The output has to still be the output.** See `stamp`. A stamp made before this
    # rule existed records no digests and is checked exactly as it was, so nothing that
    # ever ran stops running; anything stamped since is bound to what it produced.
    changed = [o for o, d in (rec.get("output_digests") or {}).items()
               if d is not None and content_print(rnd.rel(o)) != d]
    if changed:
        return (False, f"{artifact} was stamped over {', '.join(sorted(changed))}, and "
                       f"the file(s) on disk are not the ones it accepted; a replaced "
                       f"output is not a warm one")
    now = fingerprint(rnd, tuple(rec["inputs"]), plan=plan)
    moved = [k for k, v in rec["inputs"].items() if now.get(k) != v]
    if moved:
        return (False, f"{artifact} was made when {', '.join(moved)} fingerprinted "
                       f"differently; it and everything downstream of it are stale")
    return (True, f"{artifact} is warm: {', '.join(sorted(rec['inputs']))} are "
                  f"unchanged")


def invalidate(rnd, artifact: str, why: str = "") -> list:
    """Withdraw the stamps of `artifact` and everything that depends on it.

        Returns the names withdrawn. The files are left alone on purpose: a stage decides
        what to do with its own outputs, and this module only ever withdraws the claim that
        they are current.

        **What is withdrawn is remembered.** The review's fifth finding, and it is the one
        that turns a safety mechanism into a hazard: invalidating `preview` deleted its
        stamp, and `stage_preview`'s rule for an artifact with no stamp is "this predates
        the contract, reuse it as a legacy fixture" -- so invalidating an artifact *made it
        reusable*. Removing a stamp cannot be how a known-bad artifact becomes a
        grandfathered one, so an invalidation writes a tombstone and `legacy` asks for it.
        
    """
    rec = _load(rnd)
    # the kinds this artifact is an input of, so a row keyed on a narrow kind is
    # withdrawn with the artifact the kind is made of (see `DERIVED_FROM`)
    kinds = {k for k, srcs in DERIVED_FROM.items() if artifact in srcs}
    dropped = [a for a in list(rec) if a != INVALID
               and (a == artifact or artifact in DEPENDS.get(a, ())
                    or (kinds & set(DEPENDS.get(a, ()))))]
    # **An unstamped artifact is withdrawn too.** An imported fixture has no stamp, so
    # nothing was dropped and no tombstone was written -- and `legacy` then went on
    # saying the withdrawn document was a fixture in good standing. Withdrawing names
    # the artifact whether or not this run had made it.
    if artifact not in dropped:
        dropped.append(artifact)
    tomb = dict(rec.get(INVALID) or {})
    for a in dropped:
        rec.pop(a, None)
        tomb[a] = why or f"invalidated with {artifact}"
    rec[INVALID] = tomb
    os.makedirs(rnd.state, exist_ok=True)
    with open(rnd.rel(RECORD), "w") as fh:
        json.dump(rec, fh, indent=1)
    return dropped


#: Where a round records that its unstamped artifacts were **deliberately imported**: a
#: shipped fixture, adopted on purpose, with a reason. See `legacy`.
IMPORTED = "imported.json"


def import_legacy(state: str, why: str, files=()) -> dict:
    """Adopt the unstamped artifacts in `state` as a fixture, on the record.

        Takes the directory rather than the round because `Round.state` is what calls it,
        while it is deciding what that directory holds.
        
    """
    import time as _time
    rec = {"why": why, "files": sorted(files),
           "at": _time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, IMPORTED), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def imported(rnd) -> dict | None:
    p = rnd.rel(IMPORTED)
    return json.load(open(p)) if os.path.exists(p) else None


def legacy(rnd, artifact: str) -> bool:
    """May `artifact` be reused as a document from before the dependency contract?

        True only where it has never been stamped, has never been invalidated, **and this
        round has an import record saying its unstamped documents were adopted on purpose**.

        The last clause is the unification round's. "Explicitly import legacy fixtures
        rather than silently trusting missing stamps": a missing stamp used to mean "this
        predates the contract, reuse it", which is indistinguishable from "a stage crashed
        before it could stamp anything" and from "somebody dropped a file in the directory".
        A fixture is a deliberate import and now says so; anything else is made again, which
        costs a run and cannot be wrong.
        
    """
    doc = _load(rnd)
    if artifact in doc or artifact in (doc.get(INVALID) or {}):
        return False
    rec = imported(rnd)
    if rec is None:
        return False
    # **An import authorises the files it names and no others.** The review's
    # counterexample: a fixture that shipped only `place.json` made an unstamped
    # `preview` -- which nothing had imported and no stage had made -- reusable, because
    # the permission was granted to the directory. A run that writes one document into
    # an imported directory has not thereby adopted everything else in it.
    files = set(rec.get("files") or [])
    if not files:
        # an import record written before this rule named no files; it kept its
        # directory-wide meaning and is left with it rather than silently narrowed
        return True
    return any(f in files for f in ARTIFACT_FILES.get(artifact, (f"{artifact}.json",)))


#: Which file on disk each artifact *is*, for the import permission above. An artifact
#: absent from this table is looked up as `<name>.json`, which is every one of them that
#: has a single obvious document.
ARTIFACT_FILES = {
    "plan": ("plan.place.json", "plan.json"),
    "assembled": ("plan.json",),
    "built": ("parts.json", "world_built.npz"),
    "site_search": ("site_search.json",),
    "plateau": ("plateau.json",),
    "resolution": ("resolution.json", "findings.json"),
    # the proposal `stage_ground` writes, and `settle_ground`'s declaration record
    # beside it. Both are named so an import that adopted either one keeps its
    # permission
    "ground": ("ground_proposal.json", "ground.json"),
    "preview": (os.path.join("preview", "preview.json"), "preview.json"),
    "reading": ("reading.json",),
    "capabilities": ("capabilities.json",),
    "interpretation": ("interpretation.json",),
    "judgment": ("judgment.json",),
}


def recorded(rnd, artifact: str) -> bool:
    """or withdrawn it?

        The question "was any ground work done here" is not the question "may this
        unstamped document be reused", and conflating them made a round that never touched
        its terrain look like one whose terrain was prepared for another candidate.
        
    """
    doc = _load(rnd)
    return artifact in doc or artifact in (doc.get(INVALID) or {})


def candidate_id(rnd, plan=None) -> str:
    """The sentence's requirements, the programme they were read into and the plan those
        produced: change any of them and this is a different candidate. Stamped onto the
        records that belong to one -- the inspections, the repair passes -- so that a
        budget spent on a withdrawn candidate and a reading of a plan that no longer exists
        are both things a reader can see rather than infer.
        
    """
    got = fingerprint(rnd, ("intent", "spec", "plan"), plan=plan)
    return contracts.digest(got)[:16]


def table(rnd) -> list:
    """Every stamp and whether it is warm, for a readout."""
    out = []
    for a in sorted(k for k in _load(rnd) if k != INVALID):
        fresh, why = check(rnd, a)
        out.append({"artifact": a, "fresh": bool(fresh), "why": why})
    return out
