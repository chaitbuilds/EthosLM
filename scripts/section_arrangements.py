"""Compare the arrangements available to the registered section's own districts.

    $PY scripts/section_arrangements.py --state out/comp-city --round rounds/comp-city.json

The round asks for "a few substantial arrangements compared cheaply", on the actual
streets, anchors, ground constraints and resolved demands, **through the compiler and
the validator** -- and it asks for that comparison before a candidate is selected, not
for a score invented afterwards. `placeplan.negotiate_ring` does this for a ring whose
*width* is in question; a city whose rings carry no explicit count never asks that
question (`concentric_layout` gates the negotiation on one), so its districts were never
offered an alternative at all.

This asks the same question one layer down, on each district's own rectangle and its own
terrain: `arrange.alternatives` compiles every arrangement the library has for that
fabric, certifies each through `placeplan.district_failures`, and reports what the
compiler actually laid -- lots, built and allocated occupation, street enclosure,
reservations kept, and the validator's verdict. The ordering is a stated priority over
measured quantities and not a score; `arrange.alternatives`' own docstring is where that
priority is written down.

Nothing here changes a plan. It writes `section_arrangements.json` into the state
directory, which is the record a selection can be argued with.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import arrange, placeplan                                   # noqa: E402


def _districts(place: dict, prefixes) -> list:
    out = []
    for d in place.get("districts") or []:
        name = str(d.get("name") or "")
        if any(name.startswith(p) for p in prefixes):
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--round", dest="round_file", required=True)
    ap.add_argument("--out", default="section_arrangements.json")
    a = ap.parse_args()

    cfg = json.load(open(a.round_file))
    sec = (cfg.get("flags") or {}).get("section") or {}
    prefixes = sorted(set((sec.get("sides") or {}).values()))
    place = json.load(open(os.path.join(a.state, "plan.place.json")))
    spec_p = os.path.join(a.state, "place.checked.json")
    spec = json.load(open(spec_p)) if os.path.exists(spec_p) else None
    pr_p = os.path.join(a.state, "parts.json")
    parts_record = json.load(open(pr_p)) if os.path.exists(pr_p) else None
    _card, decls = placeplan.types_card(None, (spec or {}).get("form"))
    by_name = {str(p.get("name")): p for p in (spec or {}).get("defining_parts") or []}

    rows, t0 = [], time.time()
    for d in _districts(place, prefixes):
        part = by_name.get(str(d.get("defines") or "")) or {}
        got = arrange.alternatives(d, part, place, decls, spec=spec,
                                   seed=int((cfg.get("flags") or {}).get("seed") or 1),
                                   ceiling=d.get("structures"),
                                   parts_record=parts_record)
        rows.append({"district": d.get("name"), "of": d.get("defines"),
                     "rect": [d.get("x0"), d.get("z0"), d.get("x1"), d.get("z1")],
                     "asked": d.get("structures"), "alternatives": got})
        print(f"{d.get('name')}: {len(got)} alternative(s)")
        for r in got:
            print(f"   {'*' if r is got[0] else ' '} {str(r.get('action')):12s} "
                  f"lots {str(r.get('lots')):>4} built "
                  f"{r.get('built_cover')} allocated {r.get('allocated_cover')} "
                  f"enclosure {r.get('enclosure')} reservations "
                  f"{r.get('reservations_kept')}/{r.get('reservations_required')} "
                  f"{'REFUSED ' + str(r.get('refuses'))[:60] if r.get('refused') else 'ok'}")
    doc = {"record": "section_arrangements", "version": 1,
           "of": os.path.basename(os.path.abspath(a.state)),
           "section": sec.get("id"), "rect": sec.get("rect"),
           "by": "arrange.alternatives -- one compile per alternative on the district's "
                 "own rectangle, each certified through placeplan.district_failures",
           "ordering": "a stated priority over measured quantities, not a score: laid "
                       "and not refused, then reservations kept, then the frontage its "
                       "own character asked for, then built occupation, then street "
                       "enclosure, then the count",
           "districts": rows, "seconds": round(time.time() - t0, 1),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = os.path.join(a.state, a.out)
    json.dump(doc, open(p, "w"), indent=1)
    print(f"\n-> {p} ({doc['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
