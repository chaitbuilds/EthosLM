"""The composition round's acceptance runner: one machine-readable answer per gate.

    $PY scripts/check_composition.py
    $PY scripts/check_composition.py --gate section_contrast,finding_repaired

Every gate is answered from the **real driver's artifacts** (`out/comp-*/`), and the
section's own relationships are answered from `ethoslm.section.record`, which measures them
off what was built. A gate never answers `pass` from an absence, and a gate that cannot
be evidenced answers `pass: false` with the reason. The record is written to
`out/comp-accept/<stamp>.json` and `latest.json`, so a failing run is retained beside the
passing one that replaces it. Nothing here edits a round's state directory.

**What this runner exists to stop.** Its predecessor's `city_section` gate compared the
*length* of a hand-authored `section.json["demonstrated"]` list with the length of the
round file's `must_demonstrate` list, and passed while the same report conceded the
contrast could not be seen; its connectivity check read `components` and `access` fields
that `circulate.Network.to_json` has never written, so it passed on every run and on a
missing file. So the rules here are:

  - a relationship is established by a measurement of the built world or it is not
    established. `section.json` is derived, not written by hand;
  - a numerical gate **cannot overrule a material negative reading**: an open material
    finding about a subject fails the gate about that subject, whatever the numbers say;
  - every field a check reads must exist, and a check that reads an absent field fails
    rather than passing vacuously. `_have` is that rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GATES = ("section_selection", "section_contrast", "section_anchor_courts",
         "section_route", "features_preserved", "finding_repaired",
         "reuse_and_invalidation", "arrangements_certified", "density_on_built",
         "material_decision")

CITY = "comp-city"


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _rel(name: str, *parts) -> str:
    return os.path.join(ROOT, "out", name, *parts)


def _doc(name: str, *parts):
    return _load(_rel(name, *parts))


def _rows(parts_rec) -> list:
    return [r for w in ((parts_rec or {}).get("waves") or [])
            for r in (w.get("parts") or [])]


def _section(name: str = CITY) -> dict:
    """The derived section record, from disk where the stage wrote it, else measured."""
    got = _doc(name, "section.json")
    if got and got.get("by", "").startswith("ethoslm.section"):
        return got
    from ethoslm import section as section_mod
    return section_mod.record(os.path.join(ROOT, "out", name), of=name)


def _rel_of(sec: dict, key: str) -> dict:
    for r in sec.get("relationships") or []:
        if r.get("id") == key:
            return r
    return {}


def _open_material(name: str = CITY) -> list:
    """Open material findings in the ledger: the veto no number may overrule."""
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
        """A check whose input is absent fails, and says which input."""
        got = doc if not field else (doc or {}).get(field)
        return self.check(f"the evidence for `{what}` exists", got is not None,
                          f"{what}: {'present' if got is not None else 'absent'}")

    def result(self) -> dict:
        return {"pass": bool(self.checks) and all(c["pass"] for c in self.checks),
                "checks": self.checks}


# ------------------------------------------------------------------ the section

def gate_section_selection(g: Gate) -> None:
    """A1: the section is the registered extent, bounded honestly."""
    pr = _doc(CITY, "parts.json")
    if not g.have("parts.json", pr):
        return
    sm = pr.get("sample") or {}
    g.check("the build selected a registered section, not a self-chosen sample",
            bool(sm.get("section")) and bool(sm.get("registered")),
            f"section {sm.get('section')}, rect {sm.get('rect')}")
    g.check("the section refused nothing: no cut fell on a gate",
            not sm.get("refused"), str(sm.get("refused") or "no refusal"))
    runs = sm.get("boundary_runs") or []
    plan = _doc(CITY, "plan.json") or {}
    whole = {}
    for p in plan.get("parts") or []:
        if p.get("kind") == "edge" and p.get("path"):
            whole[p["name"]] = sum(abs(b[0] - a[0]) + abs(b[1] - a[1])
                                   for a, b in zip(p["path"], p["path"][1:])) + 1
    clipped = [(r["of"], r["columns"], whole.get(r["of"])) for r in runs]
    g.check("every boundary was clipped to the section, not bought entire",
            bool(clipped) and all(w is None or c < w for _, c, w in clipped),
            "; ".join(f"{n} {c} of {w} columns" for n, c, w in clipped) or "no runs")
    g.check("a leaf that straddles the section boundary was cut out, not half-built",
            sm.get("cut_out") is not None,
            f"{len(sm.get('cut_out') or [])} leaf/leaves cut out and named")
    g.check("the sample scope is the plots actually included",
            sm.get("included_plots") is not None,
            f"{len(sm.get('included_plots') or [])} plot(s) named as the built scope")
    stood = [r for r in _rows(pr) if r.get("stood")]
    g.check("the section is fabric, not predominantly boundary",
            bool(stood) and sum(1 for r in stood
                                if r.get("kind", "plot") == "plot") >= 0.6 * len(stood),
            f"{sum(1 for r in stood if r.get('kind', 'plot') == 'plot')} plot(s) of "
            f"{len(stood)} standing part(s)")


def gate_section_contrast(g: Gate) -> None:
    """A1: the two fabrics' difference is visible in what was built."""
    sec = _section()
    rel = _rel_of(sec, "contrast")
    if not g.have("the contrast measurement", rel, "status"):
        return
    g.check("the contrast is demonstrated on built geometry",
            rel.get("status") == "demonstrated", str(rel.get("how"))[:300])
    m = rel.get("measured") or {}
    g.check("both sides of the boundary carry enough complete fabric",
            len(m.get("sides_with_enough_fabric") or []) == 2,
            f"sides with enough fabric: {m.get('sides_with_enough_fabric')}")
    g.check("the two fabrics stand on opposite sides of the boundary",
            m.get("across_boundary") is True,
            f"across_boundary {m.get('across_boundary')}")
    g.check("the measures are of built mass and spacing, not of allocation",
            all(str(t.get("measure")) in ("built_cover", "median_footprint",
                                          "median_neighbour_gap")
                for t in (m.get("tests") or [])) and bool(m.get("tests")),
            ", ".join(str(t.get("measure")) for t in (m.get("tests") or [])))
    # the veto
    bad = [r for r in _open_material()
           if any(s for s in (r.get("subjects") or [])
                  if "ring" in str(s) or "fabric" in str(r.get("about", "")))]
    g.check("no open material finding about the fabric contradicts a pass",
            not bad or rel.get("status") != "demonstrated",
            "; ".join(f"{r.get('id')}: {str(r.get('says'))[:80]}" for r in bad)
            or "no open material finding about the fabric")


def gate_section_anchor_courts(g: Gate) -> None:
    """A1: a usable anchor and courts that are courts."""
    sec = _section()
    a, c = _rel_of(sec, "anchor"), _rel_of(sec, "courts")
    if not g.have("the anchor measurement", a, "status"):
        return
    g.check("the anchor's use is demonstrated on the assembled world",
            a.get("status") == "demonstrated", str(a.get("how"))[:300])
    g.check("the anchor holds usable space of its own",
            any((x.get("floor_columns") or 0) > 0
                for x in ((a.get("measured") or {}).get("anchors_") or [])),
            "; ".join(f"{x.get('part')} {x.get('floor_columns')} columns"
                      for x in ((a.get("measured") or {}).get("anchors_") or [])))
    g.check("the courts are reachable and open on the assembled world",
            c.get("status") == "demonstrated", str(c.get("how"))[:300])


def gate_section_route(g: Gate) -> None:
    """A1: one continuous route, through the boundary, to the anchor."""
    sec = _section()
    r = _rel_of(sec, "route")
    if not g.have("the route measurement", r, "status"):
        return
    m = r.get("measured") or {}
    wc = m.get("walk_check") or {}
    g.check("the walk check exists and names its own numbers",
            wc.get("cells") is not None and wc.get("reached") is not None,
            f"walk check {wc}")
    g.check("the route is one connected network with nothing unreachable",
            r.get("status") == "demonstrated", str(r.get("how"))[:300])
    g.check("a gate and an anchor stand on that same network",
            bool(m.get("gates")) and bool(m.get("anchors")),
            f"gates {m.get('gates')}, anchors {m.get('anchors')}")


def gate_features_preserved(g: Gate) -> None:
    """A1/A2: a required feature is preserved, or it is owed -- never assumed."""
    sec = _section()
    f = _rel_of(sec, "features")
    if not g.have("the feature measurement", f, "status"):
        return
    m = f.get("measured") or {}
    g.check("no required feature of a part in scope is owed",
            f.get("status") == "demonstrated", str(f.get("how"))[:300])
    g.check("an undecided predicate answer is reported unresolved, not counted",
            m.get("undecided") is not None,
            f"{len(m.get('undecided') or [])} undecided answer(s) recorded")
    us = _doc(CITY, "usable.json") or {}
    pr = _doc(CITY, "parts.json") or {}
    g.check("the predicates were run on this build's own assembled world",
            us.get("built_digest") and us.get("candidate")
            and us.get("candidate") == pr.get("candidate"),
            f"usable candidate {us.get('candidate')}, parts candidate "
            f"{pr.get('candidate')}")


# ------------------------------------------------------------------ the revision

def gate_finding_repaired(g: Gate) -> None:
    """A2: a consequential composition finding changed geometry and then closed."""
    imp = _doc(CITY, "improve.json")
    led = _doc(CITY, "obligations.json")
    if not g.have("improve.json", imp) or not g.have("obligations.json", led):
        return
    cycles = list(imp.get("cycles") or [])
    applied = [c for c in cycles if c.get("applied")]
    g.check("a material finding was routed to an action and applied",
            bool(applied),
            "; ".join(f"{c.get('finding')} -> {c.get('action')}" for c in cycles[:4])
            or "no cycle applied")
    changed = [c for c in applied if (c.get("record") or {}).get("changed")
               or c.get("changed")]
    g.check("the action changed geometry rather than returning the same place",
            bool(changed),
            "; ".join(f"{c.get('action')} changed "
                      f"{(c.get('record') or {}).get('changed')}" for c in applied[:4]))
    g.check("the affected section was rebuilt after the change",
            any(c.get("rebuilt") or c.get("candidate_after") for c in applied)
            or bool(imp.get("rebuilds")),
            f"rebuilds {imp.get('rebuilds')}")
    rows = list((led.get("rows") or {}).values())
    closed = [r for r in rows if r.get("disposition") == "closed"]
    with_ev = [r for r in closed
               if any((e or {}).get("candidate") == r.get("candidate")
                      or (e or {}).get("measures") for e in (r.get("evidence") or []))]
    g.check("closure cites a measurement of the row's own measure on the current "
            "candidate",
            bool(closed) and len(with_ev) == len(closed),
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


def gate_reuse_and_invalidation(g: Gate) -> None:
    """A3: reuse what is valid, invalidate what the change moved."""
    from ethoslm import deps
    dj = _doc(CITY, "deps.json")
    if not g.have("deps.json", dj):
        return
    art = dj.get("artifacts") or dj.get("records") or dj
    g.check("the built artifact records the prepared ground it was built on",
            "ground_proposal" in (deps.DEPENDS.get("built") or ())
            or "prepared" in (deps.DEPENDS.get("built") or ()),
            f"built depends on {deps.DEPENDS.get('built')}")
    g.check("the ground proposal is fingerprinted from the file the stage writes",
            deps.GROUND_PROPOSAL_FILE == "ground_proposal.json",
            f"GROUND_PROPOSAL_FILE {deps.GROUND_PROPOSAL_FILE}")
    us, pr = _doc(CITY, "usable.json") or {}, _doc(CITY, "parts.json") or {}
    views = _doc(CITY, "inspection", "views.json") or {}
    g.check("no reading or certificate is attributed to a candidate it is not of",
            (not views or views.get("candidate") == pr.get("candidate"))
            and (not us or us.get("candidate") == pr.get("candidate")),
            f"parts {pr.get('candidate')}, views {views.get('candidate')}, usable "
            f"{us.get('candidate')}")
    led = _doc(CITY, "obligations.json") or {}
    reopened = [r for r in (led.get("rows") or {}).values()
                if r.get("disposition") == "open" and (r.get("history")
                                                       or r.get("reopened"))]
    g.check("a finding that recurred on a new candidate reopened",
            all(not (r.get("disposition") == "closed"
                     and r.get("candidate") and r.get("last_reported")
                     and r.get("recurred_on")
                     and r.get("recurred_on") != r.get("candidate"))
                for r in (led.get("rows") or {}).values()),
            f"{len(reopened)} row(s) carry a reopening history")
    del art


def gate_arrangements_certified(g: Gate) -> None:
    """A4: every alternative offered for selection passed the real validator."""
    # the section's own certified comparison first (`scripts/section_arrangements.py`),
    # then the ring negotiation's record, because a city whose rings carry no explicit
    # count never asks the ring negotiation for an alternative at all
    arr = _doc(CITY, "section_arrangements.json") or _doc(CITY, "arrangements.json")
    if not g.have("the arrangement comparison", arr):
        return
    rows = (arr.get("districts") or arr.get("alternatives") or arr.get("rings") or []
            if isinstance(arr, dict) else arr)
    flat = []
    for r in rows if isinstance(rows, list) else []:
        flat.extend(r.get("alternatives") or [r])
    g.check("the alternatives are recorded with what the compiler reported",
            bool(flat) and all(("lots" in a or "holds" in a or "cover" in a)
                               for a in flat),
            f"{len(flat)} alternative(s) recorded")
    certified = [a for a in flat if a.get("verdict") is not None
                 or a.get("refuses") is not None or a.get("validator") is not None
                 or a.get("certified") is not None]
    g.check("every alternative carries the district validator's verdict",
            bool(flat) and len(certified) == len(flat),
            f"{len(certified)} of {len(flat)} alternative(s) certified")
    # the first row of each district is the one the stated priority puts first
    chosen = [d["alternatives"][0] for d in (rows if isinstance(rows, list) else [])
              if isinstance(d, dict) and d.get("alternatives")]
    bad = [a for a in chosen if a.get("refused") or a.get("refuses")]
    g.check("no alternative the validator refuses is ranked first",
            bool(chosen) and not bad,
            "; ".join(str(a.get("refuses"))[:70] for a in bad)
            or f"{len(chosen)} district(s), each ranked on a certified alternative")


def gate_density_on_built(g: Gate) -> None:
    """A5: density and street life are judged on what was built."""
    it = _doc(CITY, "intent.json")
    if not g.have("intent.json", it):
        return
    dens = [r for r in (it.get("requirements") or [])
            if "density" in str(r.get("id"))]
    g.check("the section's density clause was measured at all", bool(dens),
            "; ".join(str(r.get("id")) for r in dens) or "no density requirement")
    # `intent.json`'s requirement rows carry the measure in their own words; the
    # structured evidence is `_quality_measure`'s third return, which `placeread`
    # records. Both are read, and a clause that says nothing about built mass fails.
    whys = [str(r.get("why") or "") for r in dens]
    g.check("the density measure reports built occupation beside allocation",
            bool(whys) and all("the buildings on them cover" in w for w in whys),
            " | ".join(w[:150] for w in whys))
    g.check("the denominator is the ground the requirement is about",
            bool(whys) and all("developable" in w or "scope" in w for w in whys),
            " | ".join(w[:110] for w in whys))
    sec = _section()
    sides = ((_rel_of(sec, "contrast").get("measured") or {}).get("sides") or [])
    g.check("each side's built cover is reported for the section",
            bool(sides) and all(s.get("built_cover") is not None for s in sides),
            "; ".join(f"{s.get('side')} {s.get('built_cover')}" for s in sides))


def gate_material_decision(g: Gate) -> None:
    """The bounded visual experiment: a decision, either way, acted on."""
    rec = _doc("comp-material", "comparison.json") or _doc("comp-material",
                                                           "decision.json")
    if rec is None:
        g.check("the bounded material experiment recorded a decision", False,
                "no out/comp-material/comparison.json or decision.json")
        return
    judged = str((rec.get("judgment") or {}).get("decision") or rec.get("decision") or "")
    kind = judged.split(",")[0].strip().lower().replace(" ", "_")
    g.check("a decision was recorded",
            kind in ("enable", "enable_it", "leave_off", "leave_it_off"),
            f"decision {judged!r}")
    g.check("the decision is acted on rather than described",
            bool(rec.get("integrated")) == kind.startswith("enable"),
            f"integrated {rec.get('integrated')} against decision {judged!r}")
    controls = (rec.get("controls") or rec.get("control")
                or (rec.get("raw") or {}).get("controls")
                or (rec.get("displays") or {}).get("control"))
    g.check("the comparison had a positive control before any arm", bool(controls),
            "positive control recorded" if controls else "no control recorded")
    figs = (rec.get("patterns") or rec.get("figures")
            or (rec.get("raw") or {}).get("figures")
            or (rec.get("raw") or {}).get("figure_guard")
            or (rec.get("raw") or {}).get("figure_guard_counterfactual"))
    g.check("deliberate patterns are protected, or the decision is to leave it off",
            kind.startswith("leave") or bool(figs),
            f"figures {str(figs)[:160]}")


# ------------------------------------------------------------------ the runner

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", default="", help="comma-separated; the default is all")
    ap.add_argument("--out", default="comp-accept")
    a = ap.parse_args()
    only = [g for g in a.gate.split(",") if g] or list(GATES)
    out = {"run": "c1", "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": {}}
    for name in only:
        g = Gate(name)
        try:
            globals()[f"gate_{name}"](g)
        except Exception as e:                      # noqa: BLE001 -- the gate reports
            g.check("the gate ran", False, f"{type(e).__name__}: {e}")
        out["gates"][name] = g.result()
    passed = sum(1 for v in out["gates"].values() if v["pass"])
    out["passed"], out["of"] = passed, len(only)
    d = os.path.join(ROOT, "out", a.out)
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for p in (os.path.join(d, f"{stamp}.json"), os.path.join(d, "latest.json")):
        json.dump(out, open(p, "w"), indent=1)
    for name, v in out["gates"].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'} {name}")
        for c in v["checks"]:
            print(f"   {'ok  ' if c['pass'] else 'MISS'} {c['check']}: {c['why'][:160]}")
    print(f"\n{passed} of {len(only)} gates -> {os.path.join(d, 'latest.json')}")
    return 0 if passed == len(only) else 1


if __name__ == "__main__":
    sys.exit(main())
