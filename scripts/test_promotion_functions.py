#!/usr/bin/env python3
"""**The promotion guard judges the whole inhabited result, not its surviving fraction.**

    $PY scripts/test_promotion_functions.py

The design resolution round. The fabric reset round promoted its court revision (trial 6,
`f0052092f211acd5`) over the accepted `b7e4b74f2a6dadf6` because a removed subject was a
tradeoff: two hillside courtyard houses were deleted, one of them a court that held. Both
candidates are re-read here under **one ruler** -- `section.record` as it now stands, with
its `functions` relationship (homes and shops that stood, were entered on foot and, where
the type publishes a form plan, measure the form their use owes on the blocks) -- and
compared with `promote.between`, the same comparison `promote.regressions` makes in a
round. Nothing in either frozen directory is written.

    p1  the deletion is rejected: the trial against the accepted candidate regresses
        (a held court removed with nothing in its place; homes entered fell)
    p2  a legitimate replacement is admitted: the same trial with the removed hillside
        houses laid again under new ids (a redesign may replace parts), working this time
        -- rooms, court and door measured -- regresses nothing on the subjects or the
        functions
    p3  a replacement that does not hold is not a replacement: the same new houses laid
        again but short of their rooms regress
    p4  the ruler shows the court revision's cost on a quality already failing: the
        calm side's courtyard houses' least room depth fell from the accepted candidate
        to the trial (both below the form's minimum, so both already fail)
"""
from __future__ import annotations

import copy
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import section  # noqa: E402
from ethoslm.pipeline import promote  # noqa: E402

ACCEPTED = os.path.join(ROOT, "out", "fr-city.accepted-snapshot")
TRIAL = os.path.join(ROOT, "out", "fr-frozen-city")
REG = json.load(open(os.path.join(ROOT, "rounds", "fr-city.json")))["flags"]["section"]

_SEC: dict = {}


def sec_of(state: str) -> dict:
    if state not in _SEC:
        _SEC[state] = section.record(state, registered=REG)
    return copy.deepcopy(_SEC[state])


def want_of(state: str, sec: dict | None = None) -> dict:
    sec = sec if sec is not None else sec_of(state)
    return {"measured": True,
            "relationships": {k: v for k, v in promote._rels(sec).items()
                              if v in promote.PROTECTED_STATUS},
            "built": promote._stood(state), "quantities": promote._quantities(sec),
            "subjects": promote._subjects(sec)}


def _rel(sec, rid):
    return next(r for r in sec["relationships"] if r["id"] == rid)


def p1_deletion_is_rejected():
    regs = promote.between(want_of(ACCEPTED), sec_of(TRIAL), promote._stood(TRIAL))
    ids = [r["id"] for r in regs]
    assert "courts:removed" in ids, ids
    assert "homes entered" in ids, ids
    return f"rejected on {ids}"


def _replaced(ok: bool) -> dict:
    """The trial's record with the accepted candidate's two hillside houses laid again
    under new ids: their court subjects and home subjects come back renamed."""
    acc, now = sec_of(ACCEPTED), sec_of(TRIAL)
    gone = ["middle_ring_north_west_2_s2_00", "middle_ring_north_west_2_s1_00"]
    ca = {c["part"]: c for c in _rel(acc, "courts")["measured"]["courts_"]}
    cn = _rel(now, "courts")["measured"]
    for g in gone:
        if g in ca:
            c = dict(ca[g], part=g.replace("north_west_2", "north_west_2r"))
            c["affirmative"] = bool(ok)
            cn["courts_"].append(c)
    fa = _rel(acc, "features")["measured"]
    fn = _rel(now, "features")["measured"]
    for k in ("held_", "failed_", "owed_", "unknown_"):
        for f in fa.get(k) or []:
            if f.get("part") in gone:
                f2 = dict(f, part=f["part"].replace("north_west_2", "north_west_2r"))
                (fn.setdefault("held_" if ok else "failed_", [])).append(f2)
    ha = {h["part"]: h for h in _rel(acc, "functions")["measured"]["homes_"]}
    fm = _rel(now, "functions")["measured"]
    for g in gone:
        # the redesign lays working houses in their place (rooms and court measured), or
        # -- the negative -- houses that neither stand entered nor work
        h = dict(ha[g], part=g.replace("north_west_2", "north_west_2r"),
                 working=bool(ok), entered=bool(ok), form=bool(ok))
        fm["homes_"].append(h)
        fm["homes"] += 1
        fm["homes_entered"] += int(bool(h["stood"] and h["entered"]))
        fm["homes_working"] += int(bool(h["working"]))
    return now


def p2_replacement_is_admitted():
    now = _replaced(ok=True)
    regs = promote.between(want_of(ACCEPTED), now, promote._stood(TRIAL))
    bad = [r["id"] for r in regs
           if r["id"].startswith(("courts", "features", "homes", "shops"))
           or r["what"] == "subject"]
    assert not bad, bad
    rest = [r["id"] for r in regs]
    return f"no subject or function regression; remaining rows {rest or 'none'}"


def p3_replacement_that_does_not_hold():
    now = _replaced(ok=False)
    regs = promote.between(want_of(ACCEPTED), now, promote._stood(TRIAL))
    ids = [r["id"] for r in regs]
    assert any(i.startswith(("courts:", "homes")) for i in ids), ids
    return f"rejected on {ids}"


def p4_room_depth_fell_on_a_failing_quality():
    def least(sec):
        out = []
        for h in _rel(sec, "functions")["measured"]["homes_"]:
            if h["type"] == "courtyard_house" and h["side"] == "calm" and h["form_why"]:
                depths = [int(x.split()[-1]) for x in
                          h["form_why"].split(";")[0].replace("rooms ", "").split(", ")]
                out.append(min(depths))
        return sorted(out)
    a, b = least(sec_of(ACCEPTED)), least(sec_of(TRIAL))
    assert a and b and max(b) < max(a) and sum(b) / len(b) < sum(a) / len(a), (a, b)
    return f"least clear room depth per house: accepted {a}, trial {b}"


def main() -> int:
    ok = fail = 0
    for fn in (p1_deletion_is_rejected, p2_replacement_is_admitted,
               p3_replacement_that_does_not_hold, p4_room_depth_fell_on_a_failing_quality):
        try:
            print(f"ok   {fn.__name__}: {fn()}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            fail += 1
    print(f"\n{ok}/{ok + fail} promotion cases pass")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
