"""One judging path: candidates plus a question in, verdicts and a summary out.

`judge.py` owns the thing that must not change -- blinding, position-swapping, the
content-addressed cache, the bracket. This module owns the part that had been
copy-pasted into six files instead: collect the comparisons a round needs, stage the
ones the cache cannot answer, and count the ones it can.

The staging loop is the reason this is worth having. Every judged experiment here runs
the same three-step dance -- try every comparison against the warm cache, and if any of
them misses, write *all* the misses to one requests file and exit 2 so the orchestrating
agent can answer them in a single fan-out rather than one round trip at a time. Written
six times, it drifted six ways: two of the six forgot to keep going after the first
miss, so a cold run staged one judgement per invocation.
"""
from __future__ import annotations

import json
import math
import os

from . import judge


def wilson(k: int, n: int, z: float = 1.96):
    """The Wilson 95% interval on k of n. Registered in advance in E1d and reported
    ever since, so that a near-miss is interpretable without anyone touching a bar."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(mid - hw, 3), round(mid + hw, 3)]


class Session:
    """One round of judging against one cache.

        `compare(a, b)` returns "a", "b", "tie" -- or None when the cache could not answer
        and no model was supplied, in which case the request is held in `needed` and the
        caller carries on. Nothing raises: a run that cannot finish should still tell you
        *everything* it is missing.
        
    """

    def __init__(self, question: str, *, stage_name: str = "judge", ask=None,
                 cache: dict | None = None, cache_path: str = judge.CACHE):
        self.question = question
        self.stage_name = stage_name
        self.ask = ask
        self.cache_path = cache_path
        self.cache = judge.load_cache(cache_path) if cache is None else cache
        self.needed: list = []
        self.calls = 0

    def compare(self, a, b, question: str | None = None):
        q = self.question if question is None else question
        before = len(self.cache)
        try:
            res = judge.compare(a, b, q, ask=self.ask, stage_name=self.stage_name,
                                cache=self.cache, cache_path=self.cache_path)
        except judge.JudgementNeeded as e:
            self.needed += e.requests
            return None
        self.calls += max(0, len(self.cache) - before)
        return res

    def stage(self, path: str) -> int:
        """Write the outstanding requests where an agent can answer them."""
        if not self.needed:
            return 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(self.needed, open(path, "w"), indent=1)
        print(f"{len(self.needed)} judgements need a model -- staged at {path}")
        return len(self.needed)


def tally(results, *, correct="a") -> dict:
    """Count a list of verdicts the way every mutant experiment here counts them.

        `correct` names which side should win -- "a" in the mutant arms, where a is always
        the intact building. It may also be a list, one entry per result, for an experiment
        whose correct side is not the same on every pair: E-craft's blind picks were
        recorded per structure before any judgement, and three of its four are the
        no-library arm. A tie counts against, as pre-registered in E1a and never moved.
        
    """
    sides = ([correct] * len(results) if isinstance(correct, str)
             else list(correct))
    if len(sides) != len(results):
        raise ValueError("one correct side per result, or one for all")
    pairs = [(r, c) for r, c in zip(results, sides) if r in ("a", "b", "tie")]
    judged = [r for r, _ in pairs]
    n_c = sum(1 for r, c in pairs if r == c)
    n_w = sum(1 for r, c in pairs if r != c and r != "tie")
    n_t = sum(1 for r in judged if r == "tie")
    return {"judged": len(judged), "correct": n_c, "wrong": n_w, "ties": n_t,
            "correct_rate": round(n_c / len(judged), 3) if judged else None,
            "tie_rate": round(n_t / len(judged), 3) if judged else None,
            "wilson95": wilson(n_c, len(judged))}


def meets(summary: dict, prereg: dict) -> bool:
    """The pre-registered floor, applied. Never computes a threshold -- it reads one."""
    return bool(summary["judged"]
                and summary["correct_rate"] >= prereg["min_correct_rate"]
                and summary["tie_rate"] < prereg["max_tie_rate"])


def preregistration(path: str, default: dict) -> dict:
    """The thresholds for this experiment: whatever was written the first time.

        Moving a threshold after seeing a number is the failure this project exists to
        avoid, so the first write wins forever and every experiment reads its floor back
        off disk rather than out of its own source.
        
    """
    if os.path.exists(path):
        try:
            got = json.load(open(path)).get("preregistered")
            if got:
                return got
        except (ValueError, OSError):
            pass
    return default
