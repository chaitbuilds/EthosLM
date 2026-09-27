"""**Which volume a state delivers, said once and read everywhere.**

    adopt(state_dir, path, kind=, why=, candidate=)   -> the record, `delivered.json`
    resolve(state_dir)                                  -> {path, kind, source, note, ...}
    adopted(state_dir)                                  -> the absolute path, note logged

The finishing pass wrote `world_finished.npz` beside `world_built.npz`, while the live
write (`stages_media.stage_write`) and the view scripts (`scripts/ca_views.py`,
`scripts/ca_eye.py`) read `world_built.npz`, so the user could be shown a different
volume from the one that was judged. This module makes the delivered artifact an explicit
record rather than a file-name convention.

**The rule** (`resolve`), in order:

1. `delivered.json` names a volume, the file is there, and its content digest is the one
   recorded at adoption -> that volume (`source: "adopted"`). A *finished* adoption also
   records the digest of the structural world it was finished from; if
   `world_built.npz` has since changed, the finished volume is stale and step 1 fails.
2. Step 1 failed because the adopted file changed or went stale -> **the mismatch is
   reported** (`mismatch` is set and the note says why) and the fallback of step 3 is
   used. Nothing silently reads a changed file as if it were the judged one.
3. Nothing adopted (or step 2) -> `world_finished.npz` if a finishing pass wrote one
   that is newer than `world_built.npz`, else `world_built.npz` (`source: "fallback"`).
   A finished volume older than the structural one was finished from an earlier build
   and is not used.

Structural checks (lint, predicates, occupancy) keep reading `world_built.npz`; this is
only about what is *shown and written*.
"""
from __future__ import annotations

import json
import os
import time

RECORD = "delivered.json"
BUILT = "world_built.npz"
FINISHED = "world_finished.npz"
KINDS = ("structural", "finished")


class ArtifactError(ValueError):
    """An adoption that cannot be recorded, named."""


def _digest(path: str) -> str | None:
    from .deps import content_print
    return content_print(path)


def adopt(state_dir: str, path: str, *, kind: str, why: str,
          candidate: str | None = None) -> dict:
    """Record `path` as the volume `state_dir` delivers. Returns the record written.

        `path` may be absolute or relative to `state_dir`; it must exist and lie inside the
        state directory (a delivered artifact travels with its state). `kind` is
        'structural' (the built world) or 'finished' (a finishing pass over it).

    """
    if kind not in KINDS:
        raise ArtifactError(f"kind is one of {', '.join(KINDS)}, not {kind!r}")
    if not why:
        raise ArtifactError("an adoption says why")
    state = os.path.abspath(state_dir)
    p = path if os.path.isabs(path) else os.path.join(state, path)
    p = os.path.abspath(p)
    if not os.path.exists(p):
        raise ArtifactError(f"{p} is not on disk; nothing to adopt")
    if candidate is None:
        candidate = candidate_of(state)
    rel = os.path.relpath(p, state)
    if rel.startswith(".."):
        raise ArtifactError(f"{p} is outside the state directory {state}")
    built = os.path.join(state, BUILT)
    rec = {"record": "delivered", "version": 1, "path": rel, "kind": kind,
           "digest": _digest(p), "candidate": candidate, "why": str(why),
           "adopted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "built_digest": _digest(built) if os.path.exists(built) else None}
    tmp = os.path.join(state, RECORD + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, os.path.join(state, RECORD))
    return rec


def candidate_of(state_dir: str) -> str | None:
    """The candidate id `parts.json` records for this state, if any."""
    p = os.path.join(os.path.abspath(state_dir), "parts.json")
    try:
        return (json.load(open(p)) or {}).get("candidate") if os.path.exists(p) else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def record(state_dir: str) -> dict | None:
    p = os.path.join(os.path.abspath(state_dir), RECORD)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def _fallback(state: str) -> tuple:
    built = os.path.join(state, BUILT)
    fin = os.path.join(state, FINISHED)
    if os.path.exists(fin) and (not os.path.exists(built)
                                or os.path.getmtime(fin) >= os.path.getmtime(built)):
        return fin, "finished", (f"nothing adopted: {FINISHED} is newer than {BUILT}, so "
                                 f"the finishing pass's volume is used")
    if os.path.exists(fin):
        return built, "structural", (f"nothing adopted: {FINISHED} is older than {BUILT} "
                                     f"(finished from an earlier build), so {BUILT} is used")
    return built, "structural", f"nothing adopted and no finishing pass: {BUILT} is used"


def resolve(state_dir: str) -> dict:
    """The delivered volume of a state and how it was decided. See the module rule."""
    state = os.path.abspath(state_dir)
    rec = record(state)
    mismatch = None
    if rec and rec.get("path"):
        p = os.path.abspath(os.path.join(state, rec["path"]))
        if not os.path.exists(p):
            mismatch = f"the adopted {rec['path']} is no longer on disk"
        elif rec.get("digest") and _digest(p) != rec["digest"]:
            mismatch = (f"the adopted {rec['path']} changed after adoption (digest "
                        f"{rec['digest'][:12]} recorded, {(_digest(p) or '')[:12]} now)")
        elif rec.get("kind") == "finished" and rec.get("built_digest") \
                and os.path.exists(os.path.join(state, BUILT)) \
                and _digest(os.path.join(state, BUILT)) != rec["built_digest"]:
            mismatch = (f"the adopted finished {rec['path']} was finished from a "
                        f"{BUILT} that has since changed; it is stale")
        else:
            return {"path": p, "kind": rec.get("kind"), "source": "adopted",
                    "candidate": rec.get("candidate"), "digest": rec.get("digest"),
                    "why": rec.get("why"), "mismatch": None,
                    "note": f"adopted {rec.get('kind')} artifact {rec['path']}: "
                            f"{rec.get('why')}"}
    p, kind, note = _fallback(state)
    if mismatch:
        note = f"ADOPTION MISMATCH -- {mismatch}; falling back: {note}"
    return {"path": p, "kind": kind, "source": "fallback",
            "candidate": (rec or {}).get("candidate"), "digest": _digest(p),
            "why": None, "mismatch": mismatch, "note": note}


def adopted(state_dir: str, log=print) -> str:
    """The absolute path of the volume this state delivers; logs which and why."""
    got = resolve(state_dir)
    if log:
        log(f"   artifact: {os.path.basename(got['path'])} ({got['kind']}, "
            f"{got['source']}) -- {got['note']}")
    return got["path"]
