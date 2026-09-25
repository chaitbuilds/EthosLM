"""**An improvement survives its complete trial, or the best result is still there.**

The neighbourhood round's rollback ended at replanning. `improve._apply_layout` snapshots
the design records, applies the action, lays the plan out again, and restores where the
*plan* fails -- and once planning succeeded, `stage_improve` rebuilt, and when the rebuild
stopped at lint it returned `blocked` and left the failed trial on disk as the candidate.
Both of that round's cycles stopped at lint. Its second failed revision, `829fb25adb7b6b84`,
is what it delivered: a candidate with a failed route relationship, four courts standing of
nine subjects, and an inspection describing a different world.

So the transaction boundary moves out to where the evidence is. A **trial** is a candidate
under test; the **accepted** candidate is the best result this lineage has reached and is
kept whole on disk beside it. A trial is promoted only when

  1. its rebuild completed -- ground, terraces, circulation, parts, finish, lint, material,
     section, inspection -- and
  2. every requirement the accepted candidate met, it still meets (`protected`), and
  3. **the improvement it was applied for is observed on the built world**, not estimated
     before it.

and it is rejected, with its reason retained, on any of the three. A rejected trial's
evidence is kept (`trials.json`, and the refusals that rejected it); its world is not, and
the accepted candidate is put back byte for byte.

**What a candidate is** is the question this module has to answer honestly, and the answer
is the one `stages_media.REVISION_FILES` already argues for, plus what construction and
inspection wrote: a candidate is its design records, its dependency stamps, the world
volumes, the parts record and the ground under it, the obligations ledger and the reading
that judged it. Restoring some of those is not restoring a candidate -- which is how the
neighbourhood round ended with frames identifying `829...` and an `inspection/views.json`
identifying `90b...`.

This is deliberately not a general transaction framework. It is a directory of files, a
copy out and a copy back, and a record of what was tried; the round's own instruction is
"reuse valid outputs or rebuild a small candidate completely when simpler; do not build a
general transaction framework for its own sake".
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time

from .stages_media import REVISION_DIRS, REVISION_FILES

#: Where the best retained result of this lineage is kept, whole.
ACCEPTED = "accepted"

#: The record of every trial: what was tried, on what, and what happened to it.
RECORD = "trials.json"

#: **The rest of the artifact boundary**, beyond the design records `REVISION_FILES`
#: names: what construction emitted, what the checks measured and what the reading said.
#: A trial that is rolled back but keeps the trial's `parts.json` is a candidate whose
#: plan and whose built evidence describe two different worlds, which is exactly the
#: defect the neighbourhood round's independent reader found in its frames. **The ledger
#: is deliberately not in it.** `obligations.json` records what is owed *and what has
#: been tried on which candidate*; restoring it with the accepted candidate would erase
#: the record that the rejected trial's action was ever attempted, and the loop would
#: spend its whole budget applying the same refused action to the same candidate. What a
#: trial leaves behind is exactly the knowledge that it was tried and failed.
BUILT_FILES = ("parts.json", "surfaces.json", "usable.json", "section.json",
               "lint.json", "material.json",
               "ground_proposal.json", "paths.json",
               "world_finished.npz", "world.before-lanes.npz")

#: Directories of the candidate beyond `REVISION_DIRS`: the per-part construction
#: records and the inspection that judged the built world.
BUILT_DIRS = ("parts", "inspection")

#: **Files the boundary gained after some accepted copies were kept** (the fabric reset
#: round: `sectors.json`, its `sectors.<district>.laid` markers and
#: `terrace_levels.json`). An accepted copy whose boundary record does not say it
#: `covers` them lacks them because it never copied them, not because its candidate did
#: not have them, so putting it back leaves them where they are rather than deleting a
#: decision the accepted candidate was in fact made with. A copy kept since is exact
#: about them either way.
LATE_FILES = ("sectors.", "terrace_levels.json")


def _files(state: str) -> list:
    """Every file of the candidate that lives at the top of the state directory."""
    out = []
    for f in sorted(os.listdir(state)):
        p = os.path.join(state, f)
        if not os.path.isfile(p):
            continue
        if f.startswith(REVISION_FILES) or f in BUILT_FILES:
            out.append(f)
    return out


def _dirs(state: str) -> list:
    return [d for d in (*REVISION_DIRS, *BUILT_DIRS)
            if os.path.isdir(os.path.join(state, d))]


def boundary(rnd) -> dict:
    """What this module considers the candidate, as a record a reader can check.

        Published rather than implied: a rollback that silently covers less than it claims is
        the defect, so the set it covers is a value.
        
    """
    fs, ds = _files(rnd.state), _dirs(rnd.state)
    return {"files": fs, "dirs": ds,
            "bytes": int(sum(os.path.getsize(os.path.join(rnd.state, f)) for f in fs)),
            "covers": sorted(set(REVISION_FILES) | set(BUILT_FILES)),
            "from": ["stages_media.REVISION_FILES", "stages_media.REVISION_DIRS",
                     "promote.BUILT_FILES", "promote.BUILT_DIRS"]}


def _copy(state: str, dst: str) -> dict:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in _files(state):
        shutil.copy2(os.path.join(state, f), os.path.join(dst, f))
        n += 1
    for d in _dirs(state):
        shutil.copytree(os.path.join(state, d), os.path.join(dst, d))
        n += sum(len(fs) for _r, _d, fs in os.walk(os.path.join(dst, d)))
    return {"files": n, "at": dst}


def _put_back(state: str, src: str, covers=None) -> dict:
    """The accepted candidate, back over whatever the trial left.

        Anything of the boundary the trial wrote and the accepted candidate does not have is
        **removed**, not left: a district file for a district the accepted plan does not
        contain is a stale dependency, and carrying one past a rollback is the defect the
        review's fifth finding named. `covers` is what the accepted copy's boundary covered
        when it was kept; a copy that does not record it predates `LATE_FILES`, which are
        then left as they are (see there).
        
    """
    have = set(os.listdir(src)) if os.path.isdir(src) else set()
    gone, left = [], []
    for f in _files(state):
        if f not in have and covers is None and f.startswith(LATE_FILES):
            left.append(f)
            continue
        if f not in have:
            os.remove(os.path.join(state, f))
            gone.append(f)
    for d in _dirs(state):
        if d not in have:
            shutil.rmtree(os.path.join(state, d))
            gone.append(d + "/")
    n = 0
    for f in sorted(have):
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(state, f))
            n += 1
        elif os.path.isdir(s):
            dst = os.path.join(state, f)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(s, dst)
            n += sum(len(fs) for _r, _d, fs in os.walk(dst))
    out = {"restored": n, "removed": gone}
    if left:
        out["left"] = {"files": left,
                       "why": "the accepted copy predates these being in the boundary"}
    return out


# ------------------------------------------------------------------ the record

#: **Where a correction to evidence is recorded.** The neighbourhood delivery round, and
#: the independent reader's own finding about the round before it: after three trials
#: had established that the crowded ring's missing court was its approved pool and not
#: its arrangement, the agent reading's attribution of that finding was corrected **by
#: hand**, in both the live copy and the copy kept inside the accepted snapshot -- a
#: directory this module's own code calls "kept whole, byte for byte". The reader is
#: right that a correction to a reading wants a mechanism rather than an editor, and
#: this is it. Append-only, and deliberately **outside** the rollback boundary for the
#: same reason the obligations ledger is: restoring it would erase the record that the
#: correction was made, and a correction that can be rolled back is a hand edit with
#: extra steps. A reader of a corrected document applies the rows; the document itself
#: is never rewritten.
CORRECTIONS = "corrections.json"


def corrections(rnd, of: str | None = None) -> list:
    """Every recorded correction, or those of one document."""
    p = os.path.join(rnd.state, CORRECTIONS)
    if not os.path.exists(p):
        return []
    with contextlib.suppress(Exception):
        rows = json.load(open(p)).get("rows") or []
        return [r for r in rows if of is None or str(r.get("of")) == str(of)]
    return []


def correct(rnd, *, of: str, path: str, was, now, why: str,
            candidate: str | None = None) -> dict:
    """Record a correction to a piece of evidence, without editing the evidence.

        `of` is the document (`"inspection/views.json"`), `path` a dotted route into it
        (`"reading.findings[b1].owner"`), `was`/`now` the values and `why` the reason. The
        row carries the candidate it is about, so a correction cannot silently travel to a
        reading of another world.
        
    """
    p = os.path.join(rnd.state, CORRECTIONS)
    doc = {"record": "corrections", "rows": []}
    if os.path.exists(p):
        with contextlib.suppress(Exception):
            doc = json.load(open(p))
    row = {"of": str(of), "path": str(path), "was": was, "now": now, "why": str(why),
           "candidate": candidate, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    doc.setdefault("rows", []).append(row)
    doc["note"] = ("append-only, outside the rollback boundary: a correction to evidence "
                   "is recorded here and applied by whoever reads the document; the "
                   "document itself is never rewritten")
    json.dump(doc, open(p, "w"), indent=1)
    return row


def apply_corrections(rows: list, findings: list) -> list:
    """Apply the correction rows that name a finding's field, in order.

        `path` is `reading.findings[<id>].<field>`; anything else is recorded and not
        applied here, because this function only knows about findings.
        
    """
    out = [dict(f) for f in findings]
    by_id = {str(f.get("id")): f for f in out}
    for r in rows:
        path = str(r.get("path") or "")
        if not path.startswith("reading.findings["):
            continue
        rid, _, field = path[len("reading.findings["):].partition("].")
        f = by_id.get(rid)
        if f is None or not field:
            continue
        f["was"] = {**(f.get("was") or {}), field: f.get(field)}
        f[field] = r.get("now")
        f.setdefault("corrected", []).append(
            {"field": field, "was": r.get("was"), "why": r.get("why"), "t": r.get("t")})
    return out


def load(rnd) -> dict:
    p = os.path.join(rnd.state, RECORD)
    if os.path.exists(p):
        with contextlib.suppress(Exception):
            return json.load(open(p))
    return {"record": "trials", "accepted": None, "open": None, "trials": []}


def save(rnd, rec: dict) -> None:
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(os.path.join(rnd.state, RECORD), "w"), indent=1)


def accept(rnd, candidate: str, why: str, *, evidence: dict | None = None) -> dict:
    """Keep this candidate as the best retained result, whole, on disk."""
    rec = load(rnd)
    got = _copy(rnd.state, os.path.join(rnd.state, ACCEPTED))
    rec["accepted"] = {"candidate": str(candidate), "why": why,
                       "evidence": dict(evidence or {}),
                       "boundary": boundary(rnd), "kept": got,
                       "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save(rnd, rec)
    return rec["accepted"]


def accepted(rnd) -> dict | None:
    got = (load(rnd).get("accepted") or None)
    if got and os.path.isdir(os.path.join(rnd.state, ACCEPTED)):
        return got
    return None


def begin(rnd, *, candidate: str, finding: str, action: str, measures: dict | None = None,
          why: str = "") -> dict:
    """Open a trial on the accepted candidate. Accepts it first where none is kept yet.

        Returns the trial row. The accepted candidate is on disk before the action runs, so
        the rollback does not depend on this process surviving: the re-entry that closes the
        cycle is a **different driver invocation**, which is why an in-memory snapshot could
        never have covered the rebuild.
        
    """
    rec = load(rnd)
    if not accepted(rnd):
        accept(rnd, candidate, why or "the candidate this lineage reached before any "
                                      "trial was opened", evidence={"measures": measures})
        rec = load(rnd)
    trial = {"n": len(rec.get("trials") or []) + 1, "from": str(candidate),
             "finding": finding, "action": action, "state": "open",
             "measures_before": dict(measures or {}),
             "accepted_at_open": (rec.get("accepted") or {}).get("candidate"),
             "opened": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec.setdefault("trials", []).append(trial)
    rec["open"] = trial["n"]
    save(rnd, rec)
    return trial


def _open_trial(rec: dict) -> dict | None:
    n = rec.get("open")
    for t in rec.get("trials") or []:
        if t.get("n") == n and t.get("state") == "open":
            return t
    return None


def built(rnd, *, candidate: str, stages: dict | None = None) -> dict | None:
    """The trial's rebuild finished and this is what it produced. Not a promotion."""
    rec = load(rnd)
    t = _open_trial(rec)
    if t is None:
        return None
    t["candidate"] = str(candidate)
    t["rebuilt"] = sorted((stages or {}).keys())
    save(rnd, rec)
    return t


def abandon(rnd, *, why: str) -> dict | None:
    """Close an open trial that never changed anything. Nothing is restored, because
    nothing moved: `_apply_layout` puts the plan back itself where it refuses."""
    rec = load(rnd)
    t = _open_trial(rec)
    if t is None:
        return None
    t.update(state="abandoned", why=str(why),
             closed=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["open"] = None
    save(rnd, rec)
    return t


def reject(rnd, *, why: str, evidence: dict | None = None,
           restore: bool = True) -> dict | None:
    """Put the accepted candidate back and keep the trial's reason.

        The world the trial built is not kept -- it is tens of megabytes of a world nobody
        will look at again -- and everything that says *why* it was rejected is.
        
    """
    rec = load(rnd)
    t = _open_trial(rec)
    if t is None:
        return None
    t.update(state="rejected", why=str(why), evidence=dict(evidence or {}),
             closed=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if restore and os.path.isdir(os.path.join(rnd.state, ACCEPTED)):
        t["restored"] = _put_back(
            rnd.state, os.path.join(rnd.state, ACCEPTED),
            covers=((rec.get("accepted") or {}).get("boundary") or {}).get("covers"))
        t["restored_to"] = (rec.get("accepted") or {}).get("candidate")
    rec["open"] = None
    save(rnd, rec)
    return t


def promote(rnd, *, candidate: str, why: str, evidence: dict | None = None) -> dict | None:
    """The trial becomes the accepted candidate: the new best retained result."""
    rec = load(rnd)
    t = _open_trial(rec)
    if t is None:
        return None
    t.update(state="promoted", candidate=str(candidate), why=str(why),
             evidence=dict(evidence or {}),
             closed=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["open"] = None
    save(rnd, rec)
    accept(rnd, candidate, f"promoted from trial {t['n']}: {why}", evidence=evidence)
    return t


# ---------------------------------------------------- what a trial must preserve

#: The section relationships a trial may not lose. Not a score and not a new bar: the
#: **status the accepted candidate already reached**, per relationship, is what a trial
#: is held to. A baseline that failed `route` is not protected on `route`; one that
#: demonstrated it is. The round's words: "an initially flawed baseline remains labelled
#: as such; its known failures do not authorize new regressions."
PROTECTED_STATUS = ("demonstrated",)

#: **The qualities a trial is answerable for, beyond the labels.** The neighbourhood
#: delivery round, and the audit's fifth cause in its second half: "promotion protects
#: demonstrated relationship labels and continued presence of each type. It does not
#: protect every explicit requirement, individual obligation, or quantitative quality of
#: already-failing relationships. A trial can worsen an already-failing quality without
#: changing its status." Each row is `(label, path into the section record,
#: "up"|"down")`, the direction being the way the number has to go to be **better**. The
#: path walks `relationships[id]`, then `measured`, then the dotted key. They fall in
#: two classes and the difference is the whole of how they are used: * **hard** -- an
#: explicit requirement or an adopted obligation that was met and is not any more: a
#: required feature that was affirmatively held and is now owed, a threshold that could
#: be walked to and cannot, a court that held and does not. These reject a trial exactly
#: as a lost relationship does. * **accounted** -- a composition quality: mass, count,
#: enclosure, the contrast between the two sides. A trial may cost one of these and
#: still be right; what it may not do is cost one silently. Every one that moved, in
#: either direction, is written onto the promotion as a tradeoff, including the ones
#: already failing.
PROTECTED_HARD = (
    ("features held", "features.held", "up"),
    ("features owed", "features.owed", "down"),
    ("courts held", "courts.held", "up"),
    ("courts owed", "courts.owed_n", "down"),
    ("thresholds not walkable", "route.walked.unreached_n", "down"),
    ("thresholds with no stance", "route.walked.no_stance_n", "down"),
    # **the functions the neighbourhood delivers** (the design resolution round): a home
    # that stood and was entered, a home that works as the form its use owes, a shop
    # that works. A revision may replace any of them -- new parts, new ids -- and may
    # not lose one: a removed working home does not leave the denominator **the ground
    # between the buildings** (the design resolution round's registered outcome 3): fins
    # and ridges of open ground. Compared with a tolerance (`RIDGE_TOLERANCE`), since a
    # rebuild re-cuts a whole scope and moves a few columns either way; a trial that
    # doubles them has made the ground worse
    ("ground ridges", "fabric.ground.ridges", "down"),
    # ...and its holes and the water left standing in a piece (the independent reader's
    # n1 and n2, which the ridge count could not see)
    ("ground pits", "fabric.ground.pits", "down"),
    ("ground water in pieces", "fabric.ground.water_in_pieces", "down"),
    ("homes entered", "functions.homes_entered", "up"),
    ("homes working", "functions.homes_working", "up"),
    ("shops working", "functions.shops_working", "up"),
)

PROTECTED_ACCOUNTED = (
    ("crowded built cover", "contrast.sides[crowded].built_cover", "up"),
    ("crowded structures", "contrast.sides[crowded].structures", "up"),
    ("crowded neighbour gap", "contrast.sides[crowded].median_neighbour_gap", "down"),
    ("crowded court share", "contrast.sides[crowded].court_share", "up"),
    ("crowded median storeys", "contrast.sides[crowded].median_storeys", "up"),
    ("calm built cover", "contrast.sides[calm].built_cover", "up"),
    ("calm structures", "contrast.sides[calm].structures", "up"),
    ("courts asked", "courts.asked", "up"),
    ("court subjects never asked", "courts.omitted_n", "down"),
    ("anchors standing", "anchor.stood", "up"),
    ("homes", "functions.homes", "up"),
    ("shops", "functions.shops", "up"),
    # the step between the pieces as built.
    ("ground step", "fabric.ground.step_max", "down"),
)


def _dig(sec: dict, path: str):
    """One protected quantity off a section record, or None.

        `<relationship>.<dotted key under `measured`>`, with two conveniences the paths above
        rely on: a key ending `_n` is the **length** of the list at the key without it, and a
        missing step answers None rather than raising -- an absent measurement is not a zero.
        
    """
    parts = str(path).split(".")
    rid, rest = parts[0], parts[1:]
    if rid == "fabric":
        # the section's own fabric measures (`section.fabric_measures`), flat by key
        v = (sec.get("fabric") or {}).get(".".join(rest))
        return v if isinstance(v, (int, float)) else None
    node = next((r for r in (sec.get("relationships") or [])
                 if str(r.get("id")) == rid), None)
    if node is None:
        return None
    cur = node.get("measured")
    for k in rest:
        # `sides[crowded]` -- a row of a list, found by the value of its own name key.
        # `contrast.sides` is a list of two side records and the side is what names
        # them.
        if "[" in k and k.endswith("]"):
            key, want = k[:-1].split("[", 1)
            if not isinstance(cur, dict) or key not in cur:
                return None
            rowsl = cur[key]
            if not isinstance(rowsl, list):
                return None
            cur = next((r for r in rowsl
                        if isinstance(r, dict)
                        and want in (str(r.get("side")), str(r.get("id")),
                                     str(r.get("name")))), None)
            if cur is None:
                return None
            continue
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
            continue
        if k.endswith("_n") and isinstance(cur, dict) and k[:-2] in cur:
            got = cur[k[:-2]]
            cur = len(got) if isinstance(got, (list, tuple, set)) else got
            continue
        return None
    if isinstance(cur, (list, tuple, set)):
        return len(cur)
    return cur if isinstance(cur, (int, float)) else None


def _quantities(sec: dict) -> dict:
    """Every protected quantity of a section record, by label."""
    out = {}
    for label, path, way in PROTECTED_HARD + PROTECTED_ACCOUNTED:
        out[label] = {"path": path, "better": way, "value": _dig(sec, path)}
    return out


def _moved(was, now, way: str) -> str | None:
    """`"better"`, `"worse"` or None where nothing can be compared."""
    if was is None or now is None:
        return None
    try:
        a, b = float(was), float(now)
    except (TypeError, ValueError):
        return None
    if a == b:
        return None
    up = b > a
    return "better" if (up if way == "up" else not up) else "worse"


def tradeoffs(rnd, want: dict | None = None) -> dict:
    """**What this trial cost and what it bought**, over the protected quantities.

        Written onto a promotion so that "the best retained result" is a claim a reader can
        check rather than one scalar improving while three others fall. A quality already
        failing on the accepted candidate is in here on the same terms as one that was
        passing: the round's own words are "including qualities already failing".
        
    """
    want = want or protected(rnd)
    if not want.get("measured"):
        return {"measured": False, "why": want.get("why")}
    was = want.get("quantities") or {}
    now = _quantities(_section(rnd.state))
    rows = []
    for label, path, way in PROTECTED_HARD + PROTECTED_ACCOUNTED:
        a = (was.get(label) or {}).get("value")
        b = (now.get(label) or {}).get("value")
        how = _moved(a, b, way)
        if how is None and a == b:
            continue
        row_ = {"what": label, "path": path, "better": way,
                "was": a, "now": b, "moved": how,
                "hard": any(label == h[0] for h in PROTECTED_HARD)}
        if label in GROUND_TOLERATED and how == "worse" and a is not None \
                and b is not None \
                and float(b) <= float(a) * RIDGE_TOLERANCE[0] + RIDGE_TOLERANCE[1]:
            # said on the row, not left to the reader to infer from `regressions: []`
            row_["tolerated"] = (f"within the guard's tolerance for a re-cut scope: at "
                                 f"most {RIDGE_TOLERANCE[0]}x the accepted count plus "
                                 f"{RIDGE_TOLERANCE[1]} columns")
        rows.append(row_)
    worse = [r for r in rows if r["moved"] == "worse"]
    better = [r for r in rows if r["moved"] == "better"]
    subj = _subject_regressions(want.get("subjects"), _subjects(_section(rnd.state)))
    return {"measured": True, "rows": rows,
            "subjects_removed": (subj or {}).get("removed"),
            "worse": [r["what"] for r in worse], "better": [r["what"] for r in better],
            "unmeasured": [r["what"] for r in rows if r["moved"] is None],
            "why": (f"{len(better)} protected quality(ies) improved and {len(worse)} "
                    f"fell: better {[r['what'] for r in better]}, worse "
                    f"{[r['what'] for r in worse]}")}


def _section(state: str) -> dict:
    p = os.path.join(state, "section.json")
    if os.path.exists(p):
        with contextlib.suppress(Exception):
            return json.load(open(p))
    return {}


def _rels(sec: dict) -> dict:
    return {str(r.get("id")): str(r.get("status"))
            for r in (sec.get("relationships") or []) if r.get("id")}


def _stood(state: str) -> dict:
    """`{type: count}` of what stood, and the totals, off a parts record."""
    p = os.path.join(state, "parts.json")
    if not os.path.exists(p):
        return {}
    with contextlib.suppress(Exception):
        rec = json.load(open(p))
        rows = [r for w in (rec.get("waves") or []) for r in (w.get("parts") or [])]
        out: dict = {}
        for r in rows:
            if r.get("stood"):
                out[str(r.get("type"))] = out.get(str(r.get("type")), 0) + 1
        return {"by_type": out, "stood": sum(out.values()), "parts": len(rows),
                "candidate": rec.get("candidate")}
    return {}


def protected(rnd) -> dict:
    """What the **accepted** candidate holds, as the set a trial must still hold.

        Measured off the kept copy, so it is a fact about the candidate on disk and not about
        whatever the state directory happens to contain when this is asked.
        
    """
    src = os.path.join(rnd.state, ACCEPTED)
    if not os.path.isdir(src):
        return {"measured": False,
                "why": "no accepted candidate is kept: nothing is protected yet"}
    sec = _section(src)
    rels = {k: v for k, v in _rels(sec).items() if v in PROTECTED_STATUS}
    return {"measured": True, "relationships": rels,
            "built": _stood(src),
            # the quantities beside the labels, so a trial that worsens an already
            # failing quality without changing its status can be seen to have done it
            "quantities": _quantities(sec),
            # ...and the subjects under the court and feature totals
            "subjects": _subjects(sec),
            "why": (f"the accepted candidate demonstrates "
                    f"{', '.join(sorted(rels)) or 'no relationship'}; a trial that loses "
                    f"one of them is a regression whatever else it gains, and the "
                    f"quantities beside them are what it is answerable for even where "
                    f"the label was already failing")}


def regressions(rnd, want: dict | None = None) -> list:
    """Every protected requirement the **current** state no longer meets.

        Empty is the answer that lets a promotion happen, and it is never returned from an
        absence: a trial whose section record is missing regresses every protected
        relationship, because nothing measured them.
        
    """
    want = want or protected(rnd)
    if not want.get("measured"):
        return []
    return between(want, _section(rnd.state), _stood(rnd.state))


def protected_of(state: str) -> dict:
    """`protected` for a candidate kept at `state` (a directory), for comparing two
    candidates under one ruler outside a round (`scripts/test_promotion_functions.py`)."""
    sec = _section(state)
    return {"measured": bool(sec),
            "relationships": {k: v for k, v in _rels(sec).items()
                              if v in PROTECTED_STATUS},
            "built": _stood(state), "quantities": _quantities(sec),
            "subjects": _subjects(sec)}


def between(want: dict, now_sec: dict, now_stood: dict | None) -> list:
    """The regressions of a trial whose section record is `now_sec` against `want`."""
    now = _rels(now_sec)
    out = []
    for rid, was in sorted((want.get("relationships") or {}).items()):
        got = now.get(rid)
        if got != was:
            out.append({"what": "relationship", "id": rid, "was": was,
                        "now": got or "unmeasured",
                        "why": (f"the accepted candidate's `{rid}` is `{was}` and this "
                                f"trial's is `{got or 'unmeasured'}`")})
    was_built = (want.get("built") or {}).get("by_type") or {}
    now_built = (now_stood or {}).get("by_type") or {}
    for t, n in sorted(was_built.items()):
        if int(now_built.get(t, 0)) <= 0 < int(n):
            out.append({"what": "type", "id": t, "was": int(n), "now": 0,
                        "why": (f"the accepted candidate stands {n} `{t}` and this trial "
                                f"stands none: a form the place had is gone")})
    # **...and the explicit requirements and adopted obligations, which are not
    # labels.** The delivery round. A required feature that was affirmatively held and
    # is now owed, a court that held and does not, a threshold that could be walked to
    # and cannot: each is a thing the place had and lost, whatever the relationship's
    # label says, and the label may not have moved at all because the relationship was
    # already failing.
    now_q = _quantities(now_sec)
    was_q = want.get("quantities") or {}
    # **Held and lost, subject by subject** (the fabric reset round). The court and
    # feature totals compared a count, and a count cannot tell "a court that held and
    # does not" -- which is what these rows protect -- from "a house the revised design
    # no longer lays": a revision that took two houses off a hillside lost two held
    # courts by the count and nothing by the rule. Where both records carry their
    # subjects, a regression is a subject present in both that held and now does not, or
    # a new subject that does not hold; a subject the design removed is a tradeoff
    # (`tradeoffs`), named there. The threshold rows stay counts.
    by_subject = _subject_regressions(want.get("subjects"), _subjects(now_sec))
    for label, path, way in PROTECTED_HARD:
        if by_subject is not None and label in SUBJECT_LABELS:
            continue
        a = (was_q.get(label) or {}).get("value")
        b = (now_q.get(label) or {}).get("value")
        if label in GROUND_TOLERATED and a is not None and b is not None \
                and float(b) <= float(a) * RIDGE_TOLERANCE[0] + RIDGE_TOLERANCE[1]:
            continue
        if _moved(a, b, way) == "worse":
            out.append({"what": "quantity", "id": label, "was": a, "now": b,
                        "path": path,
                        "why": (f"the accepted candidate's `{label}` is {a} and this "
                                f"trial's is {b}: an explicit requirement or adopted "
                                f"obligation the place had and lost, which the "
                                f"relationship's own label does not show")})
    out += list((by_subject or {}).get("regressions") or [])
    return out


#: A trial's ridge count may exceed the accepted candidate's by this much (factor, plus
#: columns) before it is a regression.
RIDGE_TOLERANCE = (1.25, 10)
#: The ground quantities that tolerance applies to.
GROUND_TOLERATED = ("ground ridges", "ground pits", "ground water in pieces")

#: The protected totals the subject rule answers instead, where it can.
SUBJECT_LABELS = ("features held", "features owed", "courts held", "courts owed")


def _subjects(sec: dict) -> dict | None:
    """`{"courts": {part: held}, "features": {(part, feature): held}}` off a section
    record, or None where it carries no per-subject lists."""
    rels = {r.get("id"): r for r in (sec or {}).get("relationships") or []}
    courts = ((rels.get("courts") or {}).get("measured") or {}).get("courts_")
    feats = (rels.get("features") or {}).get("measured") or {}
    if not isinstance(courts, list) or not any(isinstance(feats.get(k), list)
                                                for k in ("held_", "failed_", "owed_")):
        return None
    out = {"courts": {str(c.get("part")): bool(c.get("affirmative")) for c in courts},
           "features": {}}
    fn = (rels.get("functions") or {}).get("measured") or {}
    if isinstance(fn.get("homes_"), list):
        out["homes"] = {str(h.get("part")): bool(h.get("working"))
                        for h in fn["homes_"]}
    for k, held in (("held_", True), ("failed_", False), ("owed_", False),
                    ("unknown_", False)):
        for f in feats.get(k) or []:
            if isinstance(f, dict):
                out["features"][f"{f.get('part')}/{f.get('feature')}"] = held
    return out


def _subject_regressions(was: dict | None, now: dict | None) -> dict | None:
    if not was or not now:
        return None
    regs, removed = [], []
    for kind in ("courts", "features", "homes"):
        if kind not in was and kind not in now:
            continue
        a, b = was.get(kind) or {}, now.get(kind) or {}
        for subj, held in sorted(b.items()):
            if subj in a and a[subj] and not held:
                regs.append({"what": "subject", "id": f"{kind}:{subj}", "was": True,
                             "now": False,
                             "why": f"`{subj}` held its {kind[:-1]} on the accepted "
                                    f"candidate and does not on this trial"})
            elif subj not in a and not held:
                regs.append({"what": "subject", "id": f"{kind}:{subj}", "was": None,
                             "now": False,
                             "why": f"`{subj}` is a new {kind[:-1]} subject of this trial "
                                    f"and does not hold"})
        gone = sorted(set(a) - set(b))
        removed += [f"{kind}:{s}" for s in gone]
        # **a removed subject that held is paid for by a replacement that holds, or it
        # is a loss** (the design resolution round). The fabric reset round's rule made
        # every removed subject a tradeoff, and its promoted court revision was granted
        # by deleting the hillside houses -- one of them a court that held. Old ids are
        # not immutable: a redesign may take a subject away and lay another; what it may
        # not do is take away a held one and lay nothing that holds in its place.
        lost = [s for s in gone if a.get(s)]
        gained = [s for s in sorted(set(b) - set(a)) if b.get(s)]
        if len(lost) > len(gained):
            regs.append({"what": "subject", "id": f"{kind}:removed", "was": len(lost),
                         "now": len(gained),
                         "why": (f"{len(lost)} {kind[:-1]} subject(s) that held were "
                                 f"removed ({', '.join(lost[:6])}) and {len(gained)} new "
                                 f"one(s) hold in their place: a subject removed is not a "
                                 f"subject repaired")})
    return {"regressions": regs, "removed": removed}
