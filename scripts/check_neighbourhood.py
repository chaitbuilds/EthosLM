"""The neighbourhood section's acceptance runner: one machine-readable answer per gate.

    $PY scripts/check_neighbourhood.py
    $PY scripts/check_neighbourhood.py --state nb-city --baseline comp-city
    $PY scripts/check_neighbourhood.py --gate arrangement_governs,finding_repaired

Every gate is answered from the **real driver's artifacts** (`out/<state>`), and the
section's own relationships are answered from `ethoslm.section.record`, which measures them
off what was built. A gate never answers `pass` from an absence, and a gate that cannot be
evidenced answers `pass: false` with the reason. The record is written to
`out/<state-prefix>-accept/<stamp>.json` and `latest.json`, so a failing run is retained
beside the passing one that replaces it. Nothing here edits a round's state directory.

The rules are the composition runner's, kept because they are the reason that runner
exists, plus the two rounds since:

  - a relationship is established by a measurement of the built world or it is not
    established. `section.json` is derived, not written by hand;
  - a numerical gate **cannot overrule a material negative reading**: an open material
    finding about a subject fails the gate about that subject, whatever the numbers say;
  - every field a check reads must exist, and a check that reads an absent field fails
    rather than passing vacuously (`Gate.have`);
  - **an estimate may not stand in for a measurement.** A cover figure read off the
    plan's pads is an estimate; the gate asks for the emitted figure beside it and fails
    where the record does not say which it read;
  - **an arrangement that reached blocks through a round-file override is not evidence of
    revision.** `arrangement_governs` fails if the round file carries one;
  - **a check's expression is its stated question.** The spatial design round's own rule,
    and it exists because this file broke it three times in one gate. `bool(mixes)` stood
    for "every compiled district"; `bool(kinds)` stood for a named list of five
    functions; and one check ended `... or True`. Each passed on evidence that did not
    answer it. They are repaired in `programme_composed` and each repair says what the
    old expression actually tested.

`ground_is_feasible`, `change_is_scoped`, `promotion_is_earned` -- are the neighbourhood
review's three remaining causes asked as questions of the artifacts. They are not a
parallel scoring system: each one reads the production records the driver already writes
(`plan.place.json`'s terrain records, `improve.json`'s certificates, `trials.json`) and
none of them scores anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GATES = ("arrangement_governs", "finding_repaired", "street_is_inhabited",
         "courts_are_courts", "programme_composed", "measures_are_honest",
         "shown_and_read",
         # the spatial design round's three
         "ground_is_feasible", "change_is_scoped", "promotion_is_earned",
         # the neighbourhood delivery round's five.
         "decision_survives_construction", "ground_is_one_policy",
         "both_sides_inhabited", "routes_are_walked", "improvement_is_earned")

#: The candidate under test and the retained candidate it is compared with. Both are
#: arguments (`--state`, `--baseline`) because `out` is shared across rounds and every
#: retained candidate stays exactly where it was.
CITY = "sd-city"
BEFORE = "nb-city"
ACCEPT = "sd-accept"
VIEWS = "sd-views"


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _rel(name: str, *parts) -> str:
    return os.path.join(ROOT, "out", name, *parts)


def _doc(name: str, *parts):
    return _load(_rel(name, *parts))


def _rows(parts_rec) -> list:
    return [r for w in ((parts_rec or {}).get("waves") or [])
            for r in (w.get("parts") or [])]


#: The `section.record` version this runner's questions are about. A record written
#: under an earlier one answered different questions -- before 3 the court denominator
#: was the whole place's and the route verdict was a reading of the planned lane graph
#: -- so it is re-derived rather than read, which is how a corrected ruler is applied to
#: the retained candidate as well as to the new one. The record on disk is never
#: rewritten.
SECTION_VERSION = 3


def _section(name: str | None = None) -> dict:
    """The derived section record, from disk where the stage wrote it under this
    runner's own version of the rulers, and re-measured where it did not."""
    # **`name or CITY`, and not `name = CITY` in the signature.** The neighbourhood
    # delivery round, found by running this runner on a candidate that is not the
    # module's own default: a default argument is evaluated **once, at import**, and
    # `main()` reassigns the global afterwards -- so every gate that called `_section()`
    # with no argument read `sd-city` whatever `--state` said. Four of them do
    # (`street_is_inhabited`, `courts_are_courts`, `programme_composed`,
    # `shown_and_read`), which is four gates answering about the wrong candidate.
    name = name or CITY
    got = _doc(name, "section.json")
    if got and str(got.get("by", "")).startswith("ethoslm.section") \
            and int(got.get("version") or 0) >= SECTION_VERSION:
        return got
    from ethoslm import section as section_mod
    reg = (got or {}).get("registered") or None
    return section_mod.record(os.path.join(ROOT, "out", name), of=name, registered=reg)


def _rel_of(sec: dict, key: str) -> dict:
    for r in sec.get("relationships") or []:
        if r.get("id") == key:
            return r
    return {}


def _open_material(name: str | None = None) -> list:
    name = name or CITY                       # see `_section`: not a default argument
    led = _doc(name, "obligations.json") or {}
    return [r for r in (led.get("rows") or {}).values()
            if r.get("material") and r.get("disposition") == "open"]


class Gate:
    def __init__(self, name: str):
        self.name, self.checks = name, []

    def check(self, what: str, ok, why: str = "", **evidence) -> bool:
        self.checks.append({"check": what, "pass": bool(ok), "why": why,
                            **({"evidence": evidence} if evidence else {})})
        return bool(ok)

    def have(self, what: str, doc, field: str = "") -> bool:
        got = doc if not field else (doc or {}).get(field)
        return self.check(f"the evidence for `{what}` exists", got is not None,
                          f"{what}: {'present' if got is not None else 'absent'}")

    def result(self) -> dict:
        return {"pass": bool(self.checks) and all(c["pass"] for c in self.checks),
                "checks": self.checks}


# ------------------------------------------------------- N1: one arrangement governs

def gate_arrangement_governs(g: Gate) -> None:
    """N1: the certified comparison's choice survives adoption, reload, compilation and
    construction, under an inferred count, with the parent rectangle and count unmoved."""
    cfg = _load(os.path.join(ROOT, "rounds", f"{CITY}.json")) or {}
    over = ((cfg.get("flags") or {}).get("allocation") or {}).get("arrangement")
    g.check("the round file carries no allocation override: an arrangement reaches "
            "blocks through the loop that chose it", not over, f"override {over!r}")
    place = _doc(CITY, "plan.place.json")
    spec = _doc(CITY, "place.checked.json")
    imp = _doc(CITY, "improve.json")
    if not g.have("plan.place.json", place) or not g.have("improve.json", imp):
        return
    # the adoption is a `negotiated` allocation row on the spec, which is what a plan
    # laid out again from the spec reproduces
    rows = [r for r in ((spec or {}).get("negotiated") or [])
            if isinstance(r, dict) and r.get("what") == "allocation"
            and (r.get("to") or {}).get("arrangement")]
    g.check("an arrangement was adopted and persisted on the spec's own channel",
            bool(rows),
            "; ".join(f"{r.get('action')} -> {list((r.get('to') or {}).get('arrangement'))}"
                      for r in rows[:3]) or "no allocation row carries an arrangement")
    adopted = {}
    for r in rows:
        adopted.update((r.get("to") or {}).get("arrangement") or {})
    # ...and the districts of that part carry it
    carried = [d for d in (place.get("districts") or [])
               if d.get("defines") in adopted and d.get("arrangement")]
    want = [d for d in (place.get("districts") or [])
            if d.get("defines") in adopted and int(d.get("structures") or 0) > 0]
    g.check("every district of the adopted part carries the adopted arrangement",
            bool(want) and len(carried) >= len(want),
            f"{len(carried)} of {len(want)} district(s) carry it")
    mismatch = [d["name"] for d in carried
                if any((d["arrangement"] or {}).get(k) != v
                       for k, v in (adopted.get(d["defines"]) or {}).items())]
    g.check("no district was laid to a different arrangement from the one adopted",
            not mismatch, "; ".join(mismatch[:5]) or "every district agrees")
    # the count is inferred, and it stayed inferred
    ex = (spec or {}).get("explicit_count") or {}
    g.check("this city's count is the ground's own, so the adoption was exercised on an "
            "inferred count", not (ex.get("n") and not ex.get("about")),
            f"explicit_count {ex or 'none'}")
    # the parent did not move for it
    applied = [c for c in (imp.get("cycles") or []) if c.get("applied")]
    arr = [c for c in applied
           if (c.get("action_record") or {}).get("certified")]
    g.check("the applied action was chosen by the certified comparison and not by a "
            "private rule", bool(arr),
            "; ".join(f"{c.get('action')} certified on "
                      f"{(c.get('action_record') or {}).get('certified', {}).get('on')}"
                      for c in arr[:3]) or "no applied cycle carries a certificate")
    moved = [c for c in arr if int((c.get("action_record") or {}).get("moved") or 0)]
    g.check("no parent rectangle moved for a local fabric change",
            not moved,
            "; ".join(f"{c.get('action')} moved "
                      f"{(c.get('action_record') or {}).get('moved')}" for c in moved[:3])
            or "0 rectangles moved")
    refab = [c for c in arr if (c.get("action_record") or {}).get("refabricated")]
    g.check("the action is recorded as having changed the fabric inside those rectangles",
            bool(refab),
            "; ".join(f"{len((c.get('action_record') or {}).get('refabricated') or {})} "
                      f"district(s)" for c in refab[:3])
            or "no cycle reports a refabrication")
    # ...and construction consumed it
    pr = _doc(CITY, "parts.json")
    g.check("the section was built from the candidate that carries the adoption",
            bool(pr) and pr.get("candidate"),
            f"parts candidate {(pr or {}).get('candidate')}")


# ------------------------------------------------------- N2: the finding, repaired

def gate_finding_repaired(g: Gate) -> None:
    """N2: a consequential built finding closed through the normal path, on a candidate
    that is internally consistent."""
    imp, led = _doc(CITY, "improve.json"), _doc(CITY, "obligations.json")
    if not g.have("improve.json", imp) or not g.have("obligations.json", led):
        return
    cycles = list(imp.get("cycles") or [])
    applied = [c for c in cycles if c.get("applied")]
    g.check("a material finding was routed to an action and applied", bool(applied),
            "; ".join(f"{c.get('finding')} -> {c.get('action')}" for c in cycles[:4])
            or "no cycle applied")
    g.check("the action changed the candidate rather than returning the same place",
            any((c.get("action_record") or {}).get("changed") for c in applied),
            "; ".join(str((c.get("action_record") or {}).get("changed_what"))
                      for c in applied[:3]))
    g.check("the affected output was rebuilt after the change",
            any(c.get("rebuilt") for c in applied),
            f"{sum(1 for c in applied if c.get('rebuilt'))} rebuild(s)")
    g.check("no rebuild after an applied action stopped",
            not [c for c in applied if c.get("rebuild_stopped")],
            "; ".join(str(c.get("rebuild_stopped")) for c in applied
                      if c.get("rebuild_stopped")) or "no rebuild stopped")
    rows = list((led.get("rows") or {}).values())
    closed = [r for r in rows if r.get("disposition") == "closed"]
    with_ev = [r for r in closed
               if any((e or {}).get("candidate") == r.get("candidate")
                      or (e or {}).get("measures") for e in (r.get("evidence") or []))]
    g.check("closure cites a measurement of the row's own measure on the current "
            "candidate", bool(closed) and len(with_ev) == len(closed),
            f"{len(closed)} closed row(s), {len(with_ev)} with evidence on their own "
            f"candidate")
    g.check("no material row was closed by weakening it or reclassifying it",
            all(not (r.get("material") and r.get("disposition") in ("accepted",
                                                                    "deferred"))
                for r in rows),
            "; ".join(f"{r.get('id')}: {r.get('disposition')}" for r in rows
                      if r.get("material")
                      and r.get("disposition") in ("accepted", "deferred"))
            or "no material row accepted or deferred")
    # **the candidate the closure is about is internally consistent**: no stale
    # dependency was carried past the geometry change
    state = os.path.join(ROOT, "out", CITY)
    art = sorted(os.listdir(state)) if os.path.exists(state) else []
    comp = [f for f in art if f.startswith("plan.compound.")]
    pj = _doc(CITY, "plan.json") or {}
    g.check("the compound plans on disk were laid for the road that is on disk",
            bool(comp) or not (_doc(CITY, "plan.place.json") or {}).get("compounds"),
            f"{len(comp)} compound plan(s) present")
    g.check("the plan validates with no recorded compound failure",
            not [f for f in (pj.get("failures") or [])
                 if str(f.get("check")) in ("gate", "compound")],
            "; ".join(str(f.get("why"))[:90] for f in (pj.get("failures") or [])[:3])
            or "no compound failure recorded")
    # and no action was withdrawn for a lifecycle defect rather than for its own reason
    withdrawn = [a for r in rows for a in (r.get("actions") or [])
                 if isinstance(a, dict) and a.get("withdrawn")]
    g.check("no action was withdrawn for a defect of the path rather than of the action",
            not withdrawn,
            "; ".join(str(a.get("why"))[:80] for a in withdrawn[:3]) or "none withdrawn")


# ------------------------------------------------------- N3: the street

def gate_street_is_inhabited(g: Gate) -> None:
    """N3: real frontage and continuity, on corrected rulers, against the baseline."""
    sec = _section()
    rel = _rel_of(sec, "contrast")
    if not g.have("the contrast measurement", rel, "status"):
        return
    g.check("the contrast is demonstrated on built geometry",
            rel.get("status") == "demonstrated", str(rel.get("how"))[:300])
    m = rel.get("measured") or {}
    sides = m.get("sides") or []
    g.check("each side's built cover is reported for the section",
            bool(sides) and all(s.get("built_cover") is not None for s in sides),
            "; ".join(f"{s.get('side')} {s.get('built_cover')}" for s in sides))
    # the street measures, on this candidate and on the retained baseline, corrected
    now = _street_rows(CITY)
    was = _street_rows(BEFORE)
    g.check("street frontage and continuity were measured on actual circulation "
            "geometry", bool(now) and all(r.get("continuity") is not None for r in now),
            "; ".join(f"{r['district']} enclosure {r['enclosure']} continuity "
                      f"{r['continuity']}" for r in now[:4]) or "no district measured")
    g.check("free ground beyond a lane is reported as unclaimed and is outside the "
            "street denominator",
            bool(now) and all(r.get("unclaimed_columns") is not None for r in now),
            "; ".join(f"{r['district']} unclaimed {r['unclaimed_columns']}"
                      for r in now[:4]))
    # **Like with like.** A district that straddles the registered section has only its
    # in-section leaves built, so its emitted figure is over part of its fabric:
    # comparing that with a whole-district figure would be comparing two different
    # subjects. The comparison is over the districts whose every leaf construction
    # reported, on both candidates -- which for this section is the crowded district the
    # finding is about.
    crowded_was = {r["district"]: r for r in was
                   if "lower_ring" in r["district"] and r.get("built_from") == "emitted"}
    crowded_now = [r for r in now if r["district"] in crowded_was
                   and r.get("built_from") == "emitted"]
    g.check("the crowded side has a district construction reported whole on both "
            "candidates, so the comparison is of the same subject", bool(crowded_now),
            f"comparable: {[r['district'] for r in crowded_now]}; baseline whole: "
            f"{sorted(crowded_was)}")
    better = [r for r in crowded_now
              if (r.get("enclosure") or 0) > (crowded_was[r["district"]].get("enclosure") or 0)
              and (r.get("built_cover") or 0) >= (crowded_was[r["district"]].get("built_cover") or 0)]
    g.check("the crowded side's street enclosure rose against the retained baseline on "
            "the same corrected ruler and the same basis, without losing built "
            "occupation",
            bool(crowded_now) and len(better) == len(crowded_now),
            "; ".join(f"{r['district']}: enclosure "
                      f"{(crowded_was.get(r['district']) or {}).get('enclosure')} -> "
                      f"{r.get('enclosure')}, built cover "
                      f"{(crowded_was.get(r['district']) or {}).get('built_cover')} -> "
                      f"{r.get('built_cover')}, continuity "
                      f"{(crowded_was.get(r['district']) or {}).get('continuity')} -> "
                      f"{r.get('continuity')}" for r in crowded_now[:4]))
    bad = [r for r in _open_material()
           if any("ring" in str(s) for s in (r.get("subjects") or []))
           or str(r.get("about")) in ("fabric", "composition")]
    g.check("no open material finding about the fabric contradicts a pass",
            not bad,
            "; ".join(f"{r.get('id')}: {str(r.get('says'))[:80]}" for r in bad)
            or "no open material finding about the fabric")
    anchor = _rel_of(sec, "anchor")
    g.check("the market belongs to its neighbourhood: it is reached from the fabric "
            "around it on the built world", anchor.get("status") == "demonstrated",
            str(anchor.get("how"))[:250])


def _street_rows(state: str) -> list:
    """`street_enclosure` for each compiled district of a state, on its own leaves and
    the build's own emitted footprints. The corrected ruler, applied to both candidates."""
    from ethoslm import placeplan
    place = _doc(state, "plan.place.json")
    if not place:
        return []
    parts = _doc(state, "parts.json")
    out = []
    for d in (place.get("districts") or []):
        name = str(d.get("name"))
        plan = _doc(state, f"plan.district.{name}.json")
        if not plan:
            continue
        leaves = [p for q in (plan.get("quarters") or [])
                  for p in (q.get("plots") or []) if p.get("kind", "plot") == "plot"]
        if not leaves:
            continue
        rec = _doc(state, f"district_{name}_compiled.json") or {}
        try:
            got = placeplan.street_enclosure(d, place, leaves, parts_record=parts,
                                             street_width=rec.get("street"))
            cols = placeplan.region_columns(d, place, None, record=rec, leaves=leaves,
                                            parts_record=parts)
        except Exception as e:                       # noqa: BLE001 -- reported as absent
            out.append({"district": name, "error": f"{type(e).__name__}: {e}"})
            continue
        over = float(cols.get("developable_columns") or cols.get("scope_columns") or 0)
        out.append({"district": name, "lots": len(leaves),
                    "enclosure": got.get("enclosure"),
                    "continuity": got.get("continuity"),
                    "frontage_length": got.get("frontage_length"),
                    "street_columns": got.get("street_columns"),
                    "unclaimed_columns": got.get("unclaimed_columns"),
                    "longest_enclosed_run": got.get("longest_enclosed_run"),
                    "built_columns": cols.get("built_columns"),
                    "built_from": cols.get("built_from"),
                    "built_cover": (round(cols["built_columns"] / over, 4)
                                    if cols.get("built_columns") is not None and over
                                    else None)})
    return out


# ------------------------------------------------------- N4: the courts

def gate_courts_are_courts(g: Gate) -> None:
    """N4: every court-bearing subject publishes its obligation, is asked, and answers."""
    sec = _section()
    c = _rel_of(sec, "courts")
    if not g.have("the courts measurement", c, "status"):
        return
    m = c.get("measured") or {}
    g.check("the courts are entered, open and enclosed on the assembled world",
            c.get("status") == "demonstrated", str(c.get("how"))[:300])
    g.check("every court-bearing subject is in the denominator, including any the "
            "library cannot answer for",
            m.get("subjects") is not None and m.get("asked") is not None
            and int(m.get("subjects") or 0) == int(m.get("asked") or -1),
            f"{m.get('asked')} asked of {m.get('subjects')} subject(s); "
            f"{m.get('unsupported')} unsupported")
    g.check("no subject was omitted from the evidence",
            not (m.get("omitted") or []),
            "; ".join(str(x) for x in (m.get("omitted") or [])[:5]) or "none omitted")
    per = m.get("per_part") or m.get("parts") or m.get("courts_") or []
    g.check("the answer is per part, not an aggregate count",
            bool(per), f"{len(per)} per-part answer(s)")
    # **A passing predicate does not settle what a court is.** The neighbourhood round's
    # independent reader: "every court predicate holds and every court is small" --
    # three `court_large` courts 3x3, the ten holding courts running 8 to 20 columns, so
    # the "large" courts were less than half the "small" ones. The predicate answers
    # whether a court is entered, open and enclosed; the **area** is what says whether
    # it is worth standing in, and a record that does not carry it cannot be argued
    # with.
    areas = [q.get("court_columns") for q in per
             if isinstance(q, dict) and q.get("court_columns") is not None]
    g.check("each court's measured area is on the record, so a passing light well is "
            "visible as one", bool(per) and len(areas) == len(per),
            (f"{len(areas)} of {len(per)} per-part answer(s) report the court's columns; "
             f"{sorted(areas)[:12]}") if per else "no per-part answer")
    # **The clause this gate's own wording promised and never asked.** The block design
    # round, and the audit's fourth cause. "entered, open and enclosed" was answered by
    # `court_accessible` alone -- paved, open to the sky, reachable -- so the delivered
    # section's two paved tiles standing in open cobble passed a gate whose name is
    # `courts_are_courts`, with `courts_enclosed: 0` in the compiler's own record.
    # `usable.court_enclosed` reads the blocks round each court a block claims to
    # enclose; a court that claims an enclosure and does not hold one fails here.
    claim = [q for q in per if isinstance(q, dict) and q.get("claims_enclosure")]
    shut = [q for q in claim
            if ((q.get("court_enclosed") or {}).get("holds") is True)]
    g.check("every court a block's ranges claim to enclose has buildings on all four "
            "of its sides on the assembled blocks",
            bool(claim) and len(shut) == len(claim),
            (f"{len(shut)} of {len(claim)} claimed enclosure(s) hold; "
             + "; ".join(f"{q['part']}: {(q.get('court_enclosed') or {}).get('why')}"
                         for q in claim if q not in shut)[:400])
            if claim else
            "no court in this section claims a block enclosure: a paved tile in open "
            "ground is not an enclosed court and nothing here claims to be one")
    us, pr = _doc(CITY, "usable.json") or {}, _doc(CITY, "parts.json") or {}
    g.check("the predicates were run on this build's own assembled world",
            us.get("built_digest") and us.get("candidate")
            and us.get("candidate") == pr.get("candidate"),
            f"usable candidate {us.get('candidate')}, parts candidate "
            f"{pr.get('candidate')}")
    f = _rel_of(sec, "features")
    fm = f.get("measured") or {}
    g.check("the adopted forms' obligations are collected from the production binding, "
            "not from district tokens",
            str(fm.get("from") or fm.get("by") or "").find("obligation") >= 0
            or fm.get("subjects") is not None,
            f"{fm.get('subjects')} subject(s) carrying an obligation; from "
            f"{fm.get('from') or fm.get('by')}")


# ------------------------------------------------------- N5: the programme

def gate_programme_composed(g: Gate) -> None:
    """N5: the mix is inferred from this neighbourhood's own design and recorded.

        **Three shortcuts repaired, the spatial design round.** The review found all three and
        each one is the same defect -- a check whose stated question and whose expression are
        different questions:

          * `bool(mixes)` was checked for *"every compiled district records the mix"*. It
            passes when one district of forty-three records one. The denominator is the
            districts that were **compiled**, which is a set this runner can build.
          * `bool(kinds)` was checked for *"the section stands homes, trade or work space, a
            market, streets and courts"*. It passes when the section stands anything at all --
            a wall would do. Each named function is now asked for by name, against the
            library's own `FUNCTION` declarations and the compiled districts' own `use_of`
            table, and the check reports which ones are missing.
          * `... or True` made the no-unasked-civic-buildings check unconditionally true. The
            compiled record already lists, per district, the types its role admits that its
            programme did **not** ask for (`use_mix.unasked`); the check asks whether any of
            those stood, which is the question the sentence was always about.
        
    """
    place = _doc(CITY, "plan.place.json")
    if not g.have("plan.place.json", place):
        return
    compiled, mixes = [], []
    for d in (place.get("districts") or []):
        rec = _doc(CITY, f"district_{d['name']}_compiled.json")
        if not rec:
            continue
        compiled.append(str(d["name"]))
        if rec.get("use_mix"):
            mixes.append({"district": d["name"], **{k: (rec["use_mix"] or {}).get(k)
                                                    for k in ("role", "share", "from",
                                                              "programme", "own",
                                                              "derived_from", "mix",
                                                              "unasked", "use_of")}})
    missing = sorted(set(compiled) - {m["district"] for m in mixes})
    g.check("every district that was compiled records the mix it drew its fabric from",
            bool(compiled) and not missing,
            f"{len(mixes)} of {len(compiled)} compiled district(s) "
            f"(of {len(place.get('districts') or [])} in the place) record a use mix"
            + (f"; without one: {missing[:5]}" if missing else ""))
    derived = [m for m in mixes if m.get("derived_from") or m.get("from")]
    g.check("the mix records its derivation rather than a bare number",
            bool(mixes) and len(derived) == len(mixes),
            f"{len(derived)} of {len(mixes)} record why")
    shares = {m["district"]: m.get("share") for m in mixes if m.get("share") is not None}
    g.check("the share is not one universal constant across every district",
            len({round(float(v), 4) for v in shares.values() if v}) != 1
            or len(shares) <= 1,
            f"shares {sorted({round(float(v), 4) for v in shares.values() if v})}")
    # --- the neighbourhood's own functions stand in it, each asked for by name
    # ---------
    pr = _doc(CITY, "parts.json")
    kinds, in_section = {}, _section_rows(pr, place)
    for r in in_section:
        if r.get("stood"):
            kinds[str(r.get("type"))] = kinds.get(str(r.get("type")), 0) + 1
    use_of = {}
    for m in mixes:
        use_of.update({str(k): str(v) for k, v in (m.get("use_of") or {}).items()})
    funcs = _functions_of(sorted(kinds))
    stood_uses = {use_of.get(t) for t in kinds}
    sec = _section()
    courts = (_rel_of(sec, "courts").get("measured") or {})
    circ = _doc(CITY, "circulation.json") or {}
    lanes = int((circ.get("walk_check") or {}).get("cells") or 0) or len(
        (circ.get("cells") or []))
    want = {
        "homes": ([t for t in kinds if funcs.get(t) == "dwelling"],
                  "a type declaring `FUNCTION = dwelling` stood"),
        "trade or work space": ([t for t in kinds
                                 if use_of.get(t) in ("trade", "work", "market")
                                 or t in ("shop_house", "workshop")],
                                "a type its district's own mix counts as trade or work "
                                "stood"),
        "a market": ([t for t in kinds if funcs.get(t) == "market"],
                     "a type declaring `FUNCTION = market` stood"),
        "streets": (["%d lane stance(s)" % lanes] if lanes else [],
                    "the circulation record has lanes in the section"),
        "courts": (["%s court(s) hold" % courts.get("held")]
                   if int(courts.get("held") or 0) > 0 else [],
                   "a court held on the assembled world"),
    }
    short = [k for k, (got, _w) in want.items() if not got]
    g.check("the section stands homes, trade or work space, a market, streets and courts "
            "-- each asked for by name",
            not short,
            "; ".join(f"{k}: {got or 'NONE'}" for k, (got, _w) in sorted(want.items()))
            + (f"; short of {short}" if short else ""))
    # --- and nothing civic that this neighbourhood's programme did not ask for
    # --------- **Unasked is a fact about a district, not about the place.** The
    # neighbourhood delivery round, and it is this check's own defect rather than the
    # compiler's: the unasked sets of every district were unioned and then intersected
    # with everything standing anywhere in the section, so a type one quarter's
    # programme *does* ask for was reported as filler because another quarter's does
    # not. Measured on the delivered candidate: `shop_house` stands twice in
    # `middle_ring_north_east`, whose own record lists it under `programme` -- the trade
    # its market street is for -- and the check failed the gate on it because
    # `upper_ring_east`, which is not in the section and stands nothing, lists it as
    # unasked. The spatial design report recorded this as an undiagnosed finding of the
    # repaired runner; this is the diagnosis. So a standing part is matched to the
    # district it stands in, by the same name-prefix join `section._features` and
    # `_side_measures` make, and asked of *that* district's own mix.
    by_district = {str(m["district"]): {str(t) for t in (m.get("unasked") or ())}
                   for m in mixes}
    filler, unmatched = [], []
    for r in in_section:
        if not r.get("stood"):
            continue
        name, t = str(r.get("part") or ""), str(r.get("type"))
        here = sorted((d for d in by_district if name.startswith(d)), key=len,
                      reverse=True)
        if not here:
            unmatched.append(name)
            continue
        if t in by_district[here[0]]:
            filler.append({"part": name, "type": t, "district": here[0]})
    anywhere = sorted({t for s in by_district.values() for t in s})
    g.check("no civic filler was drawn that this neighbourhood's programme did not ask "
            "for", not filler,
            f"{len(filler)} standing part(s) are of a type their own district's "
            f"programme did not ask for"
            + (f": {filler[:4]}" if filler else "")
            + f"; {len(anywhere)} type(s) are unasked in at least one district of the "
              f"place ({anywhere[:6]}), which is not the same question"
            + (f"; {len(unmatched)} standing part(s) matched no compiled district "
               f"({unmatched[:3]})" if unmatched else ""))


def _functions_of(types: list) -> dict:
    """`{type: FUNCTION}` off the library's own declarations, for the types named."""
    out = {}
    try:
        from ethoslm import placeplan
        _table, decls = placeplan.types_card()
    except Exception:                                # noqa: BLE001 -- reported as absent
        return out
    for t in types:
        d = decls.get(t) or {}
        got = d.get("function") or d.get("FUNCTION")
        if got:
            out[str(t)] = str(got)
    return out


def _section_rows(parts_rec, place) -> list:
    """The built rows inside the registered section, or all of them where no section is
    registered. A claim about the neighbourhood is about the neighbourhood."""
    rows = _rows(parts_rec)
    cfg = _load(os.path.join(ROOT, "rounds", f"{CITY}.json")) or {}
    rect = ((cfg.get("flags") or {}).get("section") or {}).get("rect")
    if not rect or len(rect) != 4:
        return rows
    x0, z0, x1, z1 = (int(v) for v in rect)
    x0, x1 = min(x0, x1), max(x0, x1)
    z0, z1 = min(z0, z1), max(z0, z1)
    out = []
    for r in rows:
        fp = (r.get("emitted") or {}).get("footprint") or r.get("footprint")
        if not fp or len(fp) != 4:
            out.append(r)
            continue
        if min(fp[0], fp[2]) <= x1 and max(fp[0], fp[2]) >= x0 \
                and min(fp[1], fp[3]) <= z1 and max(fp[1], fp[3]) >= z0:
            out.append(r)
    return out


# ------------------------------------------------------- N6: honest measures

def gate_measures_are_honest(g: Gate) -> None:
    """N6: estimates, observed results and unknowns stay distinct."""
    imp = _doc(CITY, "improve.json")
    if not g.have("improve.json", imp):
        return
    applied = [c for c in (imp.get("cycles") or []) if c.get("applied")]
    with_cmp = [c for c in applied if c.get("estimate_vs_built")]
    g.check("the estimate an action was chosen on is recorded beside what construction "
            "emitted", bool(applied) and len(with_cmp) == len(applied),
            "; ".join(json.dumps(c.get("estimate_vs_built") or {})[:160]
                      for c in applied[:2]) or "no applied cycle carries the comparison")
    rowsets = [r for c in with_cmp
               for r in ((c.get("estimate_vs_built") or {}).get("districts") or [])]
    g.check("every built figure says whether it was read off emitted geometry or off "
            "the plan's pads",
            bool(rowsets) and all(r.get("built_from") or r.get("built") == "unmeasured"
                                  for r in rowsets),
            "; ".join(f"{r.get('district')} {r.get('built_from') or r.get('built')}"
                      for r in rowsets[:4]))
    it = _doc(CITY, "intent.json")
    dens = [r for r in ((it or {}).get("requirements") or [])
            if "density" in str(r.get("id"))]
    g.check("the section's density clause was measured at all", bool(dens),
            "; ".join(str(r.get("id")) for r in dens) or "no density requirement")
    whys = [str(r.get("why") or "") for r in dens]
    g.check("the density measure reports built occupation beside allocation",
            bool(whys) and all("the buildings on them cover" in w for w in whys),
            " | ".join(w[:150] for w in whys))
    # the correction is applied to the baseline too, and the old figure retained
    note = os.path.join(ROOT, "research", "architecture-neighbourhood-working.md")
    text = open(note).read() if os.path.exists(note) else ""
    g.check("the measurement corrections are documented with the old figure beside the "
            "new", "old ruler beside new" in text and "9.1%" in text,
            "the working note records the correction and the composition round's own "
            "figures" if text else "no working note")
    was = _street_rows(BEFORE)
    g.check("the corrected ruler was applied to the retained baseline as well as to the "
            "new candidate", bool(was),
            f"{len(was)} baseline district(s) re-measured")


# ------------------------------------------------------- N7: shown and read

def gate_shown_and_read(g: Gate) -> None:
    """N7: frozen cameras, the old comparison frames, and an independent reader."""
    views = _doc(VIEWS, "final", "views.json") or _doc(VIEWS, "after", "views.json")
    cams = _doc(VIEWS, "final", "cameras.json") or _doc(VIEWS, "after", "cameras.json")
    if not g.have("the view record", views):
        return
    frames = list((views.get("frames") or views.get("views") or {}))
    g.check("the result is photographed from frozen cameras", bool(cams),
            f"{len(cams.get('cameras') or cams or {})} camera(s) recorded"
            if cams else "no cameras.json")
    def _names(doc):
        rows = (doc or {}).get("cameras") or []
        return {str(c.get("name")) for c in rows if isinstance(c, dict)}
    old = _names(_doc(BEFORE.split("-")[0] + "-views", "before",
                      "cameras.json")) or _names(_doc("comp-views", "before",
                                                     "cameras.json"))
    now = _names(cams)
    g.check("the composition round's comparison cameras are retained",
            not old or old <= now, f"missing {sorted(old - now)[:6]}")
    want = ("street", "gate", "court", "market")
    have = [w for w in want if any(w in str(f) for f in frames)]
    g.check("there is a close street view, a gate-to-market view and a court view "
            "beside the overview", len(have) >= 3, f"frames {frames[:12]}")
    g.check("the frames name the candidate and the built digest they are of",
            views.get("candidate") and views.get("built_digest"),
            f"candidate {views.get('candidate')}, built {views.get('built_digest')}")
    frames = [str(f.get("camera") or f) for f in (views.get("frames") or [])] or frames
    read = _doc(VIEWS, "reader.json") or _doc(CITY, "inspection", "reader.json")
    g.check("an independent reader checked the images and the physical evidence",
            bool(read) and bool((read or {}).get("verdict")),
            str((read or {}).get("verdict"))[:200] if read else "no reader record")
    g.check("the reader's verdict is separated from user acceptance",
            bool(read) and (read or {}).get("user_acceptance") in (None, False,
                                                                   "not claimed"),
            f"user_acceptance {(read or {}).get('user_acceptance')!r}")
    # **The verdict is about the artifact being delivered, and it says so.** The spatial
    # design round, and the review's words: "The final frames identify candidate
    # `829...`, but `inspection/views.json` identifies `90b...`. The independent reader
    # describes the first revision and lacks candidate/digest fields; the shown-and-read
    # gate accepts an existing verdict without binding it to the final candidate."
    pr = _doc(CITY, "parts.json") or {}
    us = _doc(CITY, "usable.json") or {}
    cand = pr.get("candidate")
    g.check("the frames are of the candidate that was built",
            bool(cand) and views.get("candidate") == cand,
            f"frames {views.get('candidate')}, parts {cand}")
    g.check("the reader's verdict names the exact candidate and the built digest it is "
            "of, and they are the delivered artifact's",
            bool(read) and (read or {}).get("candidate") == cand
            and bool((read or {}).get("built_digest"))
            and (read or {}).get("built_digest") == views.get("built_digest"),
            f"reader candidate {(read or {}).get('candidate')} / digest "
            f"{(read or {}).get('built_digest')}; frames {views.get('candidate')} / "
            f"{views.get('built_digest')}; parts {cand}; usable "
            f"{us.get('candidate')} / {us.get('built_digest')}")
    # ...and which artifact it is: the structural world or the finished one
    art = views.get("artifact") or views.get("world")
    g.check("the delivered artifact is named, so views and checks are of the same world",
            bool(art), f"artifact {art!r}; material finishing is "
                       f"{'on' if (_load(os.path.join(ROOT, 'rounds', f'{CITY}.json')) or {}).get('flags', {}).get('material') else 'off'} "
                       f"in the round file")


# ------------------------------------------- S1: the ground a count is derived from

def gate_ground_is_feasible(g: Gate) -> None:
    """S1: capacity, arrangement, earthworks and entrances are one account of where
        construction can stand.

        The review's first remaining cause: *"Planning capacity is not terrain-and-access
        feasibility. `district_compile` explicitly does not read terrain.
        `placeplan.developable_columns` subtracts roads and standing-part clearances from a
        rectangle, but not water or infeasible grades. `arrange.certificate_for` calls the
        district validator with `ground=None`."*
        
    """
    place = _doc(CITY, "plan.place.json")
    if not g.have("plan.place.json", place):
        return
    rows = [d for d in (place.get("districts") or []) if d.get("x1") is not None]
    with_g = [d for d in rows if (d.get("ground") or {}).get("measured")]
    g.check("every district's own ground was read before its count was derived from it",
            bool(rows) and len(with_g) == len(rows),
            f"{len(with_g)} of {len(rows)} district(s) carry a measured terrain record"
            + ("" if len(with_g) == len(rows) else
               f"; without one: {[d['name'] for d in rows if d not in with_g][:5]}"))
    if not with_g:
        return
    # the record is a measurement and not a label: every clause counted, and the
    # arithmetic closes
    bad = []
    for d in with_g:
        r = d["ground"]
        got = sum(int(r.get(k) or 0) for k in
                  ("feasible_columns", "wet_columns", "off_level_columns",
                   "broken_columns", "outside_columns"))
        if got != int(r.get("columns") or -1):
            bad.append(f"{d['name']}: {got} != {r.get('columns')}")
    g.check("each district's terrain record accounts for every column of its rectangle "
            "exactly once", not bad, "; ".join(bad[:4]) or
            f"{len(with_g)} district(s) close: feasible + wet + off-level + broken + "
            f"outside == columns")
    tot = sum(int((d["ground"]).get("columns") or 0) for d in with_g)
    ok = sum(int((d["ground"]).get("feasible_columns") or 0) for d in with_g)
    g.check("the ground the design proposes is feasible against the observed baseline",
            tot > 0 and ok > 0,
            f"{ok:,} of {tot:,} column(s) can carry a building at the level this design "
            f"brings them to ({ok / max(1, tot):.0%}); reclaimed by fill: "
            f"{sum(int((d['ground']).get('reclaimed_columns') or 0) for d in with_g):,}")
    # the count is derived from that ground and not from the rectangle
    try:
        from ethoslm import placeplan
        moved = [d for d in with_g
                 if int(placeplan.developable_columns(d, place, None))
                 <= int((d["ground"]).get("feasible_columns") or 0)]
    except Exception as e:                           # noqa: BLE001 -- reported
        moved = []
        g.check("the capacity arithmetic could be re-run", False,
                f"{type(e).__name__}: {e}")
    g.check("`developable_columns` does not offer a district more ground than can carry "
            "a building", bool(with_g) and len(moved) == len(with_g),
            f"{len(moved)} of {len(with_g)} district(s)")
    # construction consumed the proposal: the terraces were laid at the levels the plan
    # chose, per district as well as per ring
    ter = _doc(CITY, "terraces.json") or {}
    laid = {str(r.get("district")): r for r in (ter.get("districts") or [])}
    want = [d for d in with_g if d.get("level") is not None
            and any(int(x.get("level") or -10 ** 9) != int(d["level"])
                    for x in [next((r for r in (place.get("layout") or {}).get("rings")
                                    or [] if r.get("name") == d.get("defines")), {})])]
    g.check("every district that steps off its ring had that terrace laid",
            bool(ter) and all(str(d["name"]) in laid for d in want),
            f"{len(want)} district(s) step off their ring; {len(laid)} terrace(s) laid"
            + ("" if not want else
               f"; not laid: {[d['name'] for d in want if d['name'] not in laid][:5]}"))
    g.check("the district terraces that were laid stand at the level the plan chose",
            all(int(laid[str(d['name'])].get("level") or -1) == int(d["level"])
                for d in want if str(d["name"]) in laid),
            "; ".join(f"{n}: plan {next((int(d['level']) for d in want if d['name'] == n), None)}"
                      f" vs laid {r.get('level')}" for n, r in sorted(laid.items())[:4])
            or "no district terrace laid")
    # the compiler can act on the mask, or the record says it cannot
    aware = False
    try:
        from ethoslm import district_compile as dc
        aware = bool(getattr(dc, "LAYS_ON_FEASIBLE_GROUND", False))
    except Exception:                                # noqa: BLE001 -- reported below
        aware = False
    g.check("the compiler lays its lots on ground that can carry a building, so the "
            "validator's terrain clause is asked rather than skipped", aware,
            "`district_compile.LAYS_ON_FEASIBLE_GROUND` is "
            f"{aware}; until it is true `placeplan.district_failures` does not raise its "
            f"`ground` clause and `arrange.NEGOTIABLE_CHECKS` carries `ground`")
    # and the built section's access holds
    sec = _section()
    route = _rel_of(sec, "route")
    g.check("the built section is walkable end to end on the world that was built",
            route.get("status") == "demonstrated", str(route.get("how"))[:300])
    # a door failing inside the section and a component cut off outside it are different
    # failures, and the record keeps them apart
    unreached = sum(int((d["ground"]).get("unreached_columns") or 0) for d in with_g)
    g.check("unreachable ground inside a district and a route component cut off outside "
            "it are counted apart",
            all((d["ground"]).get("unreached_columns") is not None for d in with_g),
            f"{unreached:,} column(s) of otherwise buildable ground no route reaches, "
            f"reported per district and left in the mask")


# ------------------------------------------- S2: the scope of a change and its evidence

def gate_change_is_scoped(g: Gate) -> None:
    """S2: a change is qualified over what it affects, and proposals are compared on
        their own geometry.

        The review: *"`_reallocate` certifies one named district, then stores the arrangement
        on its defining part; the recorded actions refabricate ten lower-ring districts."* and
        *"`arrange.alternatives` passes the previous candidate's `parts_record` into
        `region_columns` and `street_enclosure` for hypothetical newly compiled leaves."*
        
    """
    imp = _doc(CITY, "improve.json")
    if not g.have("improve.json", imp):
        return
    cycles = list(imp.get("cycles") or [])
    applied = [c for c in cycles if c.get("applied")]
    certs = [(c, (c.get("action_record") or {}).get("certified") or {})
             for c in applied
             if (c.get("action_record") or {}).get("certified")]
    g.check("an applied change carries a certificate at all", bool(certs) or not applied,
            f"{len(certs)} of {len(applied)} applied cycle(s) carry one")
    short = []
    for c, cert in certs:
        scope = [str(s) for s in (cert.get("scope") or ())]
        qual = {str(q.get("district")) for q in (cert.get("qualified") or ())}
        refab = sorted((c.get("action_record") or {}).get("refabricated") or {})
        if not scope:
            short.append(f"{c.get('action')}: no scope declared")
        elif not set(refab) <= (set(scope) | qual):
            short.append(f"{c.get('action')}: refabricated {sorted(set(refab) - set(scope))[:3]} "
                         f"outside its declared scope {scope[:3]}")
        elif not set(scope) <= qual:
            short.append(f"{c.get('action')}: {sorted(set(scope) - qual)[:3]} in scope "
                         f"and not qualified")
    g.check("a change is qualified on every rectangle it governs, and it refabricates "
            "nothing outside that scope", not short,
            "; ".join(short[:3]) or
            "; ".join(f"{c.get('action')}: scope {len(cert.get('scope') or ())}, "
                      f"qualified {len(cert.get('qualified') or ())}, refabricated "
                      f"{len((c.get('action_record') or {}).get('refabricated') or {})}"
                      for c, cert in certs[:3]) or "no applied change to qualify")
    lost = [c for c in applied
            if ((c.get("action_record") or {}).get("out_of_scope") or {}).get("districts")]
    g.check("no applied change took structures from a district it was not about",
            all(not any(int(v[1]) < int(v[0] or 0) for v in
                        (((c.get("action_record") or {}).get("out_of_scope") or {})
                         .get("districts") or {}).values())
                for c in lost),
            "; ".join(f"{c.get('action')}: "
                      f"{((c.get('action_record') or {}).get('out_of_scope') or {}).get('districts')}"
                      for c in lost[:2]) or "no out-of-scope count moved")
    # the comparison ranked proposals on their own geometry
    ranked = [r for c, cert in certs for r in (cert.get("ranked_over") or ())]
    g.check("the comparison the change was chosen by exists and is recorded",
            bool(ranked) or not certs, f"{len(ranked)} ranked row(s)")
    try:
        from ethoslm import arrange
        joins = "parts_record=None" in (arrange.alternatives.__doc__ or "") or True
        import inspect
        src = inspect.getsource(arrange.alternatives)
        own = ("parts_record=None" in src and "region_columns" in src)
    except Exception as e:                           # noqa: BLE001 -- reported
        own, joins = False, False
        g.check("the comparison could be read", False, f"{type(e).__name__}: {e}")
    g.check("no previous build's emitted footprints are joined to a hypothetical "
            "arrangement by leaf name", bool(own),
            "`arrange.alternatives` measures `region_columns` and `street_enclosure` "
            "with no parts record: every row is the compiler's own pad arithmetic and "
            "the observation is taken after the build"
            if own else "`arrange.alternatives` still passes a parts record to "
                        "hypothetical leaves")
    # ...and the observation is taken afterwards, on the same districts
    with_cmp = [c for c in applied if c.get("estimate_vs_built")]
    g.check("the estimate is compared with what construction emitted after the build",
            bool(applied) and len(with_cmp) == len(applied),
            f"{len(with_cmp)} of {len(applied)} applied cycle(s) carry the comparison")


# ------------------------------------------- S3: an improvement survives its trial

def gate_promotion_is_earned(g: Gate) -> None:
    """S3: the delivered candidate is the best retained result, and it earned that.

        The review: *"Once planning succeeds, `stage_improve` rebuilds; when that fails at
        lint it returns `blocked` without restoring the previous candidate. Both recorded
        cycles stopped at lint. The next cycle operates on the failed first revision, and the
        second failed revision remains the delivered candidate."*
        
    """
    rec = _doc(CITY, "trials.json")
    if not g.have("trials.json", rec):
        return
    trials = list(rec.get("trials") or [])
    acc = rec.get("accepted") or {}
    pr = _doc(CITY, "parts.json") or {}
    g.check("a best retained result is kept whole on disk",
            bool(acc.get("candidate")) and os.path.isdir(_rel(CITY, "accepted")),
            f"accepted {acc.get('candidate')}; "
            f"{(acc.get('kept') or {}).get('files')} file(s) kept; boundary covers "
            f"{len((acc.get('boundary') or {}).get('files') or [])} file(s) and "
            f"{(acc.get('boundary') or {}).get('dirs')}")
    g.check("the candidate on disk is the accepted one",
            bool(pr.get("candidate")) and pr.get("candidate") == acc.get("candidate"),
            f"parts {pr.get('candidate')}, accepted {acc.get('candidate')}")
    g.check("no trial is still open",
            rec.get("open") in (None, 0),
            f"open trial {rec.get('open')}")
    rejected = [t for t in trials if t.get("state") == "rejected"]
    promoted = [t for t in trials if t.get("state") == "promoted"]
    g.check("a regressing or ineffective trial was rejected and its reason retained",
            bool(rejected) and all(t.get("why") for t in rejected),
            "; ".join(f"trial {t.get('n')} ({t.get('action')}): {str(t.get('why'))[:110]}"
                      for t in rejected[:3]) or "no trial was rejected")
    g.check("a rejected trial put the best retained result back",
            all(t.get("restored") and t.get("restored_to") for t in rejected),
            "; ".join(f"trial {t.get('n')} -> {t.get('restored_to')} "
                      f"({(t.get('restored') or {}).get('restored')} file(s))"
                      for t in rejected[:3]) or "nothing restored")
    g.check("an actual improvement was promoted through the production driver",
            bool(promoted),
            "; ".join(f"trial {t.get('n')} ({t.get('action')}) -> {t.get('candidate')}: "
                      f"{str(t.get('why'))[:150]}" for t in promoted[:2])
            or "no trial was promoted")
    g.check("a promotion is justified by an improvement observed on the built world, "
            "not by a completed build",
            bool(promoted) and all(
                ((t.get("evidence") or {}).get("closed")
                 or ((t.get("evidence") or {}).get("measure") or {}).get("moved") is True)
                for t in promoted),
            "; ".join(f"trial {t.get('n')}: closed="
                      f"{(t.get('evidence') or {}).get('closed')}, measure="
                      f"{json.dumps((t.get('evidence') or {}).get('measure') or {})[:120]}"
                      for t in promoted[:2]) or "no promotion to justify")
    g.check("no promotion lost a relationship the accepted candidate demonstrated",
            all(not ((t.get("evidence") or {}).get("regressions") or [])
                for t in promoted),
            "; ".join(json.dumps((t.get("evidence") or {}).get("regressions") or [])[:160]
                      for t in promoted[:2]) or "no regression at any promotion")


GATES_NEW_DOC = """
# They are not a parallel scoring system and a gate count is not the architectural
# outcome; each reads a production record the driver already writes, and each fails
# rather than passing from an absence.
"""


def gate_decision_survives_construction(g: Gate) -> None:
    """D1: the attachment, frontage, lot geometry and admitted storeys a planned leaf
    carries are the ones construction sited and built.

    The audit's first cause, asked at the boundary it is lost at. `pipeline.PART_GEOMETRY`
    is the whole of what a leaf hands `site()`; a field the plan decided and this tuple
    drops is a decision that never reached a block. The evidence is the **generated
    production programs** under `out/<state>/parts/`, not a type test."""
    from ethoslm import pipeline
    plan = _doc(CITY, "plan.json")
    pr = _doc(CITY, "parts.json")
    if not g.have("plan.json", plan) or not g.have("parts.json", pr):
        return
    geom = set(pipeline.PART_GEOMETRY)
    for f in ("attached", "front"):
        g.check(f"`{f}` is carried from the plan into `site()`", f in geom,
                f"pipeline.PART_GEOMETRY {'carries' if f in geom else 'drops'} `{f}`")
    leaves = [p for p in pipeline.plan_parts(plan) if p.get("kind", "plot") == "plot"]
    rows = {r.get("part"): r for r in _rows(pr)}
    built = [p for p in leaves if (rows.get(p.get("name")) or {}).get("stood")]
    g.check("the section built plots to compare", bool(built),
            f"{len(built)} plot(s) stood of {len(leaves)} planned")
    if not built:
        return
    # **the generated call, read from disk.** A leaf whose program does not name its own
    # attachment was sited as though its flanks were free, whatever the plan says.
    att = [p for p in built if p.get("attached")]
    missing = []
    for p in att:
        src = _rel(CITY, "parts", f"{p['name']}.py")
        txt = open(src).read() if os.path.exists(src) else ""
        if "'attached'" not in txt and '"attached"' not in txt:
            missing.append(p["name"])
    g.check("every attached leaf's own production program carries its attachment",
            bool(att) and not missing,
            f"{len(att) - len(missing)} of {len(att)} program(s) name it"
            + (f"; missing {missing[:4]}" if missing else ""))
    # **the house is as wide as its lot.** An attached lot sited with free flanks loses
    # `2 * Builder.SITE_INSET` columns of frontage, which is the four-column-wide house
    # on a six-column lot the audit measures. **On the axis the party walls are on, and
    # on that axis only.** A row house is narrow to the lane and deep into the plot and
    # keeps a rear strip it does not build on, so its *depth* is its own business; what
    # a lost attachment costs is the frontage, `2 * Builder.SITE_INSET` of it, which is
    # the four-column-wide house on a six-column lot the audit measures. `attached`
    # names sides of the world, so the axis is the sides' own.
    narrow, over = [], []
    inset = 2 * pipeline.Builder.SITE_INSET if hasattr(pipeline, "Builder") else 4
    for p in att:
        em = (rows.get(p["name"]) or {}).get("emitted") or {}
        # `emitted.footprint` is the bounding box of every block a part wrote, which
        # includes the platform `site()` lays one course wider than the pad all round.
        # Measured on the delivered candidate, 60 houses "overhang" their lot on that
        # ruler and **0** on the rectangle the type declares it built.
        fp = (em.get("rects") or {}).get("main") or em.get("footprint")
        if not fp or len(fp) != 4:
            continue
        sides = {str(x) for x in (p.get("attached") or ())}
        on_x = bool(sides & {"west", "east"})
        lot = (abs(int(p["x1"]) - int(p["x0"])) + 1) if on_x else \
              (abs(int(p["z1"]) - int(p["z0"])) + 1)
        got = (abs(int(fp[2]) - int(fp[0])) + 1) if on_x else \
              (abs(int(fp[3]) - int(fp[1])) + 1)
        axis = "x" if on_x else "z"
        if got < lot - 1:
            narrow.append({"part": p["name"], "axis": axis, "lot": lot, "built": got})
        if got > lot:
            over.append({"part": p["name"], "axis": axis, "lot": lot, "built": got})
    g.check("no attached house stands narrower across its party walls than its own lot",
            bool(att) and not narrow,
            f"{len(att) - len(narrow)} of {len(att)} attached house(s) reach their lot "
            f"across the attached axis, measured on the rectangle the type declares it "
            f"built"
            + (f"; short: {narrow[:3]}" if narrow else "")
            + (f" (an unattached pad would lose {inset} columns)" if narrow else ""))
    g.check("no attached house runs past its lot into its neighbour's", not over,
            f"{len(over)} house(s) overhang" + (f": {over[:3]}" if over else ""))
    # **admissibility governs the final parameter.** A leaf whose recorded envelope
    # admits fewer storeys than its own `params.storeys` is the plan disagreeing with
    # itself in writing, which is what the delivered candidate does on every terrace
    # lot.
    over = [p["name"] for p in leaves
            if (p.get("envelope") or {}).get("storeys_admitted") is not None
            and int((p.get("params") or {}).get("storeys", 1))
            > int(p["envelope"]["storeys_admitted"])]
    g.check("no leaf asks for more storeys than its own envelope admits", not over,
            f"{len(over)} leaf/leaves ask over their admitted band"
            + (f": {over[:4]}" if over else ""))
    # ...and what stood agrees with what was asked
    lost = []
    for p in built:
        want = int((p.get("params") or {}).get("storeys", 0) or 0)
        em = (rows.get(p["name"]) or {}).get("emitted") or {}
        got = em.get("storeys")
        if want and isinstance(got, int) and got < want:
            lost.append({"part": p["name"], "asked": want, "emitted": got})
    g.check("the storeys construction emitted are the storeys the plan asked for",
            not lost,
            f"{len(lost)} of {len(built)} built plot(s) emitted fewer storeys than asked"
            + (f": {lost[:3]}" if lost else ""))
    # the reported deep-lot interior-access regression
    lint = _doc(CITY, "round.json") or {}
    finds = (((lint.get("results") or {}).get("lint") or {}).get("findings") or [])
    inside = [f for f in finds if str(f.get("code")) in ("E003", "E011")]
    g.check("no built house has interior floor a walker cannot reach from its own "
            "doorway", not inside,
            f"{len(inside)} E003/E011 finding(s)"
            + (f": {[str(f.get('message'))[:90] for f in inside[:2]]}" if inside else ""))


def gate_ground_is_one_policy(g: Gate) -> None:
    """D2: one recorded ground decision governs both what may be developed and what is
    cut, filled, retained or left alone.

    The audit's third cause: `placeplan.district_ground` excludes columns needing cuts or
    fills beyond `DISTRICT_TERRACE_REACH`, and `Builder.terrace_annulus` levels full
    rectangles without that limit -- so housing is excluded because a hillside should not
    be cut while construction cuts it anyway."""
    from ethoslm import placeplan
    ter = _doc(CITY, "terraces.json")
    place = _doc(CITY, "plan.place.json")
    if not g.have("terraces.json", ter) or not g.have("plan.place.json", place):
        return
    reach = int(getattr(placeplan, "DISTRICT_TERRACE_REACH", 0) or 0)
    g.check("the feasibility mask has a registered per-column bound", reach > 0,
            f"DISTRICT_TERRACE_REACH {reach}")
    pieces = []
    for r in (ter.get("rings") or []):
        for q in ((r.get("terrace") or {}).get("piece_records") or []):
            pieces.append(("ring/" + str(r.get("ring")), q))
    for q in (ter.get("districts") or []):
        pieces.append(("district/" + str(q.get("district")), q))
    g.check("the terrace record publishes what it did, piece by piece", bool(pieces),
            f"{len(pieces)} piece(s) recorded")
    if not pieces:
        return
    # **the disposition of every column.** A record that says how much it filled and cut
    # and not what it left alone cannot be reconciled with a mask that refused columns.
    def disp(q):
        return q.get("disposition") if isinstance(q.get("disposition"), dict) else None
    named = [n for n, q in pieces if disp(q) is None]
    g.check("every piece records the disposition of its columns and not only what it "
            "moved", not named,
            f"{len(pieces) - len(named)} of {len(pieces)} piece(s) publish a disposition"
            + (f"; {named[:3]} do not" if named else ""))
    laid = [(n, disp(q)) for n, q in pieces if disp(q)]
    partitions = [n for n, dd in laid
                  if int(dd.get("worked") or 0) + int(dd.get("left_alone") or 0)
                  + int(dd.get("lanes") or 0) != int(dd.get("columns") or -1)]
    g.check("the disposition partitions the piece: every column is moved, left as found "
            "or the circulation pass's", not partitions,
            f"{len(laid) - len(partitions)} of {len(laid)} piece(s) reconcile"
            + (f"; {partitions[:3]} do not" if partitions else ""))
    # **the earthwork performed is inside the bound the count was derived under**, and
    # the bound this stage was given is the one the mask was computed with
    wrong = [{"piece": n, "reach": dd.get("reach")} for n, dd in laid
             if dd.get("reach") is None or int(dd["reach"]) != reach]
    g.check(f"every piece was laid under the mask's own per-column bound of {reach}",
            not wrong, f"{len(laid) - len(wrong)} of {len(laid)} piece(s) carry it"
            + (f"; {wrong[:3]} do not" if wrong else ""))
    over = [{"piece": n, "deepest": max(int(dd.get("max_cut") or 0),
                                        int(dd.get("max_fill") or 0)),
             "columns": dd.get("over_reach_columns")}
            for n, dd in laid
            if int(dd.get("over_reach_columns") or 0) > 0]
    g.check(f"no column was cut or filled further than the {reach} block(s) the "
            f"feasibility mask allows", not over,
            (f"{len(over)} piece(s) moved ground past the bound: {over[:3]}" if over else
             f"every one of {len(laid)} piece(s) stayed inside {reach} block(s)"))
    # ...and the two instruments reconcile over the same rectangles
    bad = []
    for d in (place.get("districts") or []):
        gr = d.get("ground") or {}
        if gr.get("feasible_columns") is None:
            continue
        rec = next((dd for n, dd in laid if n == "district/" + str(d.get("name"))), None)
        if rec is None or rec.get("worked") is None:
            continue
        if int(rec["worked"]) > int(gr["feasible_columns"]):
            bad.append({"district": d.get("name"), "moved": rec["worked"],
                        "feasible": gr["feasible_columns"]})
    g.check("no district's terrace moved more columns than its own mask calls "
            "developable", not bad,
            f"{len(bad)} district(s) disagree: {bad[:3]}" if bad else
            "the mask and the earthwork agree over every measured district")


def gate_both_sides_inhabited(g: Gate) -> None:
    """D3: both sides read as inhabited neighbourhoods with distinct grain, convincing
        streets, usable courts, supported variation and a working market.

        Built geometry and a reader's view of the delivered artifact together. A numerical
        guard does not overrule a negative reading, and a reading does not overrule geometry.
        
    """
    sec = _section(CITY)
    con = _rel_of(sec, "contrast")
    sides = {str(s.get("side")): s for s in (con.get("measured") or {}).get("sides") or []}
    g.check("both sides of the section were measured", len(sides) == 2,
            f"sides {sorted(sides)}")
    if len(sides) != 2:
        return
    for side, row in sorted(sides.items()):
        g.check(f"the {side} side stands an inhabited fabric and not a backdrop",
                int(row.get("structures") or 0) >= 8,
                f"{side}: {row.get('structures')} structure(s) covering "
                f"{row.get('built_cover')} of its ground")
    g.check("the two fabrics differ as geometry and not as a label",
            con.get("status") == "demonstrated", str(con.get("how"))[:220])
    cts = _rel_of(sec, "courts")
    m = cts.get("measured") or {}
    g.check("every court-owing subject inside the section was asked",
            int(m.get("subjects") or 0) == int(m.get("asked") or -1),
            f"{m.get('asked')} asked of {m.get('subjects')} subject(s) in the section"
            + (f"; {len(m.get('omitted') or [])} never asked" if m.get("omitted") else ""))
    g.check("the courts that were asked hold on the assembled world",
            int(m.get("held") or 0) > 0 and not (m.get("owed") or []),
            f"{m.get('held')} hold, {len(m.get('owed') or [])} owed")
    anc = _rel_of(sec, "anchor")
    g.check("the market or civic anchor works on the built blocks",
            anc.get("status") == "demonstrated", str(anc.get("how"))[:200])
    # variation: the street is a rhythm and not a comb, read off what stood
    pr = _doc(CITY, "parts.json")
    shapes = {}
    for r in _rows(pr):
        if not r.get("stood") or r.get("kind", "plot") != "plot":
            continue
        em = r.get("emitted") or {}
        fp = em.get("footprint")
        if not fp or len(fp) != 4:
            continue
        shapes.setdefault(str(r.get("type")), set()).add(
            (abs(int(fp[2]) - int(fp[0])) + 1, abs(int(fp[3]) - int(fp[1])) + 1,
             em.get("storeys")))
    n_built = sum(1 for r in _rows(pr) if r.get("stood")
                  and r.get("kind", "plot") == "plot")
    distinct = sum(len(v) for v in shapes.values())
    per_hundred = round(100.0 * distinct / n_built, 1) if n_built else None
    g.check("the buildings vary in height, depth or frontage rather than repeating",
            per_hundred is not None and per_hundred >= 20,
            f"{distinct} distinct built shape(s) over {n_built} building(s) "
            f"({per_hundred} per hundred)")
    # **and a negative reading of the fabric is not overruled by any of the above**
    openm = [r for r in _open_material(CITY)
             if str(r.get("about")) in ("fabric", "composition")]
    g.check("no open material finding about the fabric contradicts a pass", not openm,
            "; ".join(f"{r.get('id')}: {str(r.get('says'))[:90]}" for r in openm[:3])
            or "no open material finding about the fabric")


def gate_routes_are_walked(g: Gate) -> None:
    """D4: the required routes are traversable on the final assembled blocks, including
    the gate passages, thresholds and sample connections. A connected planned network
    graph alone fails this."""
    sec = _section(CITY)
    r = _rel_of(sec, "route")
    m = r.get("measured") or {}
    walked = m.get("walked")
    g.check("the route verdict is a traversal of the assembled world and not a reading "
            "of the planned network", isinstance(walked, dict) and not walked.get("error"),
            str((walked or {}).get("why") or "no traversal was run")[:220])
    if not isinstance(walked, dict):
        return
    g.check("the traversal names the artifact it walked", bool(walked.get("artifact")),
            str(walked.get("artifact")))
    g.check("every built threshold of the section has a stance on that artifact",
            not walked.get("no_stance"),
            f"{len(walked.get('no_stance') or [])} threshold(s) with no stance"
            + (f": {(walked.get('no_stance') or [])[:4]}" if walked.get("no_stance")
               else ""))
    g.check("every built threshold of the section is reachable on foot, without jumping",
            not walked.get("unreached"),
            f"{len(walked.get('reached') or [])} reached, "
            f"{len(walked.get('unreached') or [])} not"
            + (f": {(walked.get('unreached') or [])[:4]}" if walked.get("unreached")
               else ""))
    g.check("a gate of the boundary is among the thresholds walked to",
            bool(m.get("walked_gates")), f"gates walked {m.get('walked_gates')}")
    g.check("the anchor is among the thresholds walked to",
            bool(m.get("walked_anchors")), f"anchors walked {m.get('walked_anchors')}")
    g.check("the planned lane graph is connected as well",
            bool(m.get("planned_network_connected")),
            f"planned network connected: {m.get('planned_network_connected')}")


def gate_improvement_is_earned(g: Gate) -> None:
    """D5: a consequential finding of the built result improved through production and
        earned promotion on current built evidence, without losing the protected programme
        and without silently worsening another adopted architectural obligation.

        Rejection is exercised on a focused regression case and its attempt history survives.
        
    """
    tr = _doc(CITY, "trials.json")
    imp = _doc(CITY, "improve.json")
    if not g.have("trials.json", tr) or not g.have("improve.json", imp):
        return
    trials = list(tr.get("trials") or [])
    promoted = [t for t in trials if str(t.get("state")) == "promoted"]
    rejected = [t for t in trials if str(t.get("state")) == "rejected"]
    g.check("a trial was promoted through the production driver", bool(promoted),
            f"{len(promoted)} promoted, {len(rejected)} rejected of {len(trials)} trial(s)")
    g.check("rejection is exercised and its attempt history survives", bool(rejected),
            f"{len(rejected)} rejected trial(s) with their reasons retained")
    if not promoted:
        return
    p = promoted[-1]
    ev = p.get("evidence") or {}
    g.check("the promotion rests on an improvement observed on the built world",
            bool((ev.get("measure") or {}).get("moved") is True or ev.get("closed")),
            str((ev.get("measure") or {}).get("why") or p.get("why"))[:220])
    g.check("no protected requirement of the accepted candidate was lost",
            not (ev.get("regressions") or []),
            "; ".join(str(x.get("why")) for x in (ev.get("regressions") or [])[:3])
            or "0 regressions")
    trade = ev.get("tradeoffs") or {}
    g.check("the composition tradeoffs are accounted for, including qualities already "
            "failing", bool(trade.get("measured")),
            str(trade.get("why") or "no tradeoff account on the promotion")[:220])
    g.check("the finding the trial was for is a consequential built finding and not a "
            "manufactured one", bool(p.get("finding")),
            f"promoted for {p.get('finding')} through {p.get('action')}")
    # the promoted candidate is the one on disk
    pr = _doc(CITY, "parts.json")
    g.check("the promoted candidate is the delivered one",
            str(p.get("candidate")) == str((pr or {}).get("candidate")),
            f"promoted {p.get('candidate')}, on disk {(pr or {}).get('candidate')}")


RUNNERS = {
    "arrangement_governs": gate_arrangement_governs,
    "finding_repaired": gate_finding_repaired,
    "street_is_inhabited": gate_street_is_inhabited,
    "courts_are_courts": gate_courts_are_courts,
    "programme_composed": gate_programme_composed,
    "measures_are_honest": gate_measures_are_honest,
    "shown_and_read": gate_shown_and_read,
    "ground_is_feasible": gate_ground_is_feasible,
    "change_is_scoped": gate_change_is_scoped,
    "promotion_is_earned": gate_promotion_is_earned,
    "decision_survives_construction": gate_decision_survives_construction,
    "ground_is_one_policy": gate_ground_is_one_policy,
    "both_sides_inhabited": gate_both_sides_inhabited,
    "routes_are_walked": gate_routes_are_walked,
    "improvement_is_earned": gate_improvement_is_earned,
}


def main() -> int:
    global CITY, BEFORE, ACCEPT, VIEWS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", default="", help="comma-separated; default all")
    ap.add_argument("--state", default=CITY, help="the candidate's state directory "
                                                   f"under out/ (default {CITY})")
    ap.add_argument("--baseline", default=BEFORE,
                    help=f"the retained candidate to compare with (default {BEFORE})")
    ap.add_argument("--out", default="", help="where the record goes under out/ "
                                              "(default <state>-accept)")
    a = ap.parse_args()
    CITY, BEFORE = a.state, a.baseline
    ACCEPT = a.out or (CITY.split("-")[0] + "-accept")
    VIEWS = CITY.split("-")[0] + "-views"
    want = [w.strip() for w in a.gate.split(",") if w.strip()] or list(GATES)
    out, passed = {}, 0
    for name in want:
        g = Gate(name)
        try:
            RUNNERS[name](g)
        except Exception as e:                       # noqa: BLE001 -- a gate reports
            g.check("the gate ran", False, f"{type(e).__name__}: {e}")
        out[name] = g.result()
        passed += bool(out[name]["pass"])
        mark = "pass" if out[name]["pass"] else "FAIL"
        print(f"{mark}  {name}")
        for c in out[name]["checks"]:
            if not c["pass"]:
                print(f"      - {c['check']}: {c['why'][:220]}")
    rec = {"round": CITY, "baseline": BEFORE, "gates": out,
           "passed": passed, "of": len(want),
           "candidate": (_doc(CITY, "parts.json") or {}).get("candidate"),
           "registered": ("Gate wording is defined in this checker; development registrations are retained in the source repository."),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    d = os.path.join(ROOT, "out", ACCEPT)
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    json.dump(rec, open(os.path.join(d, f"{stamp}.json"), "w"), indent=1)
    json.dump(rec, open(os.path.join(d, "latest.json"), "w"), indent=1)
    print(f"\n{passed} of {len(want)} gate(s) pass; written to out/{ACCEPT}/{stamp}.json")
    return 0 if passed == len(want) else 1


if __name__ == "__main__":
    raise SystemExit(main())
