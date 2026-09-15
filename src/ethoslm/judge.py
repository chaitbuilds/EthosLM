"""Pairwise, bracketed, blinded, cached judgement over preview images.

    judge(candidates, question) -> (winner, log)

Never an absolute score. VLM absolute scoring agrees with human ratings 32-34% of the
time, pairwise ~91.5%, and the same scene has scored 75% and 55% on rendering angle
alone. A judge that emits a number is a bug; this one only ever answers "A or B".

Four properties, each load-bearing:

**Single elimination.** k candidates cost k-1 comparisons, not k(k-1)/2.

**Position-swapped.** Every comparison runs twice with the images swapped. A pair whose
two orientations disagree is recorded as **tied**, not resolved -- the tie rate is the
judge's noise floor and it is reported, not hidden. In the bracket a tie advances the
earlier-listed candidate, deterministically, and says so in the log.

**Blinded.** The model sees two images called a.png and b.png in a scratch directory
named by the judgement's own hash, and nothing else. No candidate filenames, no labels,
no "this is the mutant". This is the easiest thing in the spec to get wrong and it
invalidates every result if it is got wrong, so the staging is done here and not by
callers.

**Cached and replayable.** Every judgement is keyed on sha256(image_a) + sha256(image_b)
+ question and appended to out/judge_cache.jsonl. A warm-cache re-run makes zero model
calls and returns the identical answer -- which is what makes the one non-deterministic
component in the pipeline reproducible. A consequence worth knowing: two byte-identical
images produce the *same* key in both orientations, so identical content can never be
resolved -- it ties, structurally, at the cost of at most one model call."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time

from .measure import model_call
from .settlement import ROOT

CACHE = os.path.join(ROOT, "out", "judge_cache.jsonl")

PROMPT = ("Look at the two images.\n"
          "Image A: {a}\n"
          "Image B: {b}\n\n"
          "{question}\n\n"
          "Reply with exactly one letter on the first line -- A or B -- "
          "then one sentence of reason.")


class JudgementNeeded(RuntimeError):
    """No model in-process and the cache cannot answer. `requests` is the list of
    staged judgements: each has a `prompt` and blinded `scratch_a`/`scratch_b` images
    already on disk. Answer each, `fulfil` it, and run the bracket again."""

    def __init__(self, requests: list):
        self.requests = requests
        super().__init__(f"{len(requests)} judgements need a model call")


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _scratch_dir(cache_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(cache_path)), "judge_scratch")


def request(a: str, b: str, question: str, cache_path: str = CACHE) -> dict:
    """The judgement of a-vs-b as a record: content-addressed key, blinded paths,
    the exact prompt. Pure -- nothing is written until `stage`."""
    sa, sb = _sha(a), _sha(b)
    key = hashlib.sha256((sa + sb + question).encode()).hexdigest()
    d = os.path.join(_scratch_dir(cache_path), key[:16])
    ca, cb = os.path.join(d, "a.png"), os.path.join(d, "b.png")
    return {"key": key, "sha_a": sa, "sha_b": sb, "question": question,
            "a": str(a), "b": str(b), "scratch_a": ca, "scratch_b": cb,
            "prompt": PROMPT.format(a=ca, b=cb, question=question)}


def stage(req: dict) -> dict:
    """Write the blinded copies the model will be shown. The scratch directory is named
    by the judgement's hash, so nothing about either candidate leaks through a path."""
    os.makedirs(os.path.dirname(req["scratch_a"]), exist_ok=True)
    shutil.copyfile(req["a"], req["scratch_a"])
    shutil.copyfile(req["b"], req["scratch_b"])
    return req


def _parse(answer: str) -> str | None:
    """The verdict letter. A line that is nothing but the letter wins, and the *last*
    such line wins over earlier ones -- two E1c agents broke the one-letter format,
    narrated, then corrected themselves, and the first-standalone-letter rule read
    both of them wrong. Cached verdicts are stored at fulfil time, so this change
    touches no replay; it was registered before E1d ran."""
    bare = [ln.strip().upper() for ln in (answer or "").splitlines()
            if ln.strip().upper() in ("A", "B")]
    if bare:
        return bare[-1]
    m = re.search(r"\b([ABab])\b", answer or "")
    return m.group(1).upper() if m else None


def load_cache(cache_path: str = CACHE) -> dict:
    """key -> record, first written wins: a replay returns the answer it gave."""
    out: dict = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            line = line.strip()
            if line:
                r = json.loads(line)
                out.setdefault(r["key"], r)
    return out


def fulfil(req: dict, answer: str, seconds: float = 0.0, stage_name: str = "judge",
           cache_path: str = CACHE) -> dict:
    """Record a model's answer to a staged request: append to the cache, cost the call.
    This is the only place a judgement enters the cache, and the only place a model
    call is recorded -- a warm-cache run therefore records nothing."""
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": req["key"],
           "sha_a": req["sha_a"], "sha_b": req["sha_b"], "question": req["question"],
           "answer": answer, "verdict": _parse(answer)}
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    model_call(stage_name, req["prompt"], answer or "", seconds, key=req["key"][:16])
    return row


def compare(a: str, b: str, question: str, ask=None, stage_name: str = "judge",
            cache: dict | None = None, cache_path: str = CACHE) -> str:
    """One position-swapped comparison: "a", "b", or "tie".

        Two model calls (or cache hits), one per orientation. The orientations must agree
        on a *candidate* -- not a letter -- or the pair is tied.
        
    """
    cache = load_cache(cache_path) if cache is None else cache
    verdicts, missing = [], []
    for (p, q) in ((a, b), (b, a)):
        req = request(p, q, question, cache_path)
        if req["key"] in cache:
            verdicts.append(cache[req["key"]]["verdict"])
        elif ask is None:
            missing.append(stage(req))
        else:
            stage(req)
            t0 = time.perf_counter()
            ans = ask(req["scratch_a"], req["scratch_b"], req["prompt"])
            row = fulfil(req, ans, time.perf_counter() - t0, stage_name, cache_path)
            cache[req["key"]] = row
            verdicts.append(row["verdict"])
    if missing:
        raise JudgementNeeded(missing)
    v1, v2 = verdicts
    first = {"A": "a", "B": "b"}.get(v1)         # orientation 1: A means candidate a
    second = {"A": "b", "B": "a"}.get(v2)        # orientation 2: A means candidate b
    return first if first is not None and first == second else "tie"


def judge(candidates: list, question: str, ask=None, stage_name: str = "judge",
          cache_path: str = CACHE) -> tuple:
    """Single-elimination bracket over `candidates` (paths to images).

        Returns (winner_path, log). The log carries every comparison, the tie count and
        the tie rate; a tie advances the earlier-listed candidate and is marked. With no
        `ask` and a cold cache, raises JudgementNeeded carrying *every* staged judgement
        the current round needs, so the caller can answer them in one fan-out.
        
    """
    if not candidates:
        raise ValueError("no candidates")
    cache = load_cache(cache_path)
    field = list(candidates)
    log = {"question": question, "candidates": [str(c) for c in candidates],
           "rounds": [], "comparisons": 0, "ties": 0}
    rnd = 0
    while len(field) > 1:
        rnd += 1
        nxt, round_log, missing = [], [], []
        for i in range(0, len(field) - 1, 2):
            a, b = field[i], field[i + 1]
            try:
                res = compare(a, b, question, ask, stage_name, cache, cache_path)
            except JudgementNeeded as e:
                missing += e.requests
                continue
            winner = a if res in ("a", "tie") else b
            nxt.append(winner)
            round_log.append({"round": rnd, "a": str(a), "b": str(b),
                              "winner": str(winner), "tied": res == "tie"})
        if missing:
            raise JudgementNeeded(missing)
        if len(field) % 2:
            nxt.append(field[-1])
            round_log.append({"round": rnd, "a": str(field[-1]), "b": None,
                              "winner": str(field[-1]), "tied": False, "bye": True})
        log["rounds"] += round_log
        log["comparisons"] += sum(1 for r in round_log if not r.get("bye"))
        log["ties"] += sum(1 for r in round_log if r["tied"])
        field = nxt
    log["winner"] = str(field[0])
    log["tie_rate"] = (log["ties"] / log["comparisons"]) if log["comparisons"] else 0.0
    return field[0], log
