"""Every committed type under its own checker, in two voices, at zero model calls.

    $PY scripts/check_types.py [name ...] [--out DIR] [--jobs N]

Voice contract, B1. `pipeline.check_type` stands a type on its registered fixtures at
both seeds in the voice its author was given **and** in the round's place's own voice
(or, where that is the author's, the voice on disk whose silhouette is least like it;
`pipeline.check_voices`), and says whether the two read the same. This runs that over
`types/*.py`.

A type's authoring voice is read from the round configs that authored it; a reference
type is checked in the pair itself.

Exit status is the number of types whose two voices differ.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import pipeline  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(ROOT, "rounds", "types_g.json")
#: The configs that authored a type, in the order they are consulted.
AUTHORING = ("types_a.json", "types_b.json", "types_c.json", "types_d.json",
             "types_e.json", "types_f.json", "types_g.json")


def authoring_voices() -> dict:
    """{type name: the voice its author was given}, off the round configs."""
    out = {}
    for f in AUTHORING:
        p = os.path.join(ROOT, "rounds", f)
        if not os.path.exists(p):
            continue
        t = json.load(open(p)).get("types") or {}
        for spec in t.get("list") or []:
            if spec.get("voice"):
                out[spec["name"]] = spec["voice"]
    return out


def jsonable(res: dict) -> dict:
    rows = []
    for r in res["rows"]:
        q = {k: v for k, v in r.items() if k != "report"}
        q["findings"] = sorted(f"{f.code}@{f.pos}" for f in r["report"].findings
                               if f.code.startswith("E"))
        rows.append(q)
    return {**{k: v for k, v in res.items() if k not in ("rows", "text")}, "rows": rows}


def check_one(name: str, out_dir: str, voices_by: dict, sweep: bool = True) -> dict:
    rnd = pipeline.Round.load(CONFIG)
    be = pipeline.OfflineBackend(rnd)
    path = os.path.join(ROOT, "types", f"{name}.py")
    decl = pipeline.load_type(path)
    spec = {"name": name, "part": decl["kind"], "file": path}
    voice = voices_by.get(name)
    voices = pipeline.check_voices(voice, rnd.voice_name())
    fixtures = pipeline._fixtures_for(rnd, spec)
    seeds = pipeline._seeds_for(rnd, spec)
    t0 = time.perf_counter()
    res = pipeline.check_type(rnd, be, path, [], seeds, fixtures=fixtures,
                              voice=voice, voices=voices, sweep=sweep)
    doc = jsonable(res)
    doc.update(type=name, kind=decl["kind"], authoring_voice=voice, voices_run=voices,
               swept=sweep,
               fixtures=[f"{f['round']}/{f.get('plot') or f.get('part')}"
                         for f in fixtures],
               seeds=seeds, seconds=round(time.perf_counter() - t0, 1),
               walk_model=__import__("ethoslm.observe", fromlist=["WALK_MODEL"]).WALK_MODEL)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(doc, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
    open(os.path.join(out_dir, f"{name}.findings.md"), "w").write(res["text"])
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "voice-contract", "b1"))
    ap.add_argument("--jobs", type=int, default=1,
                    help="types checked at once; each check uses ETHOSLM_WORKERS processes")
    ap.add_argument("--no-sweep", action="store_true",
                    help="stand each type at one set of parameters, as before B1")
    a = ap.parse_args()
    sweep = not a.no_sweep
    names = a.names or sorted(os.path.basename(f)[:-3]
                              for f in glob.glob(os.path.join(ROOT, "types", "*.py")))
    voices_by = authoring_voices()
    from concurrent.futures import ProcessPoolExecutor
    docs = {}
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = {ex.submit(check_one, n, a.out, voices_by, sweep): n
                    for n in names}
            for f, n in futs.items():
                docs[n] = f.result()
                _say(docs[n])
    else:
        for n in names:
            docs[n] = check_one(n, a.out, voices_by, sweep)
            _say(docs[n])
    summary = {n: {"kind": d["kind"], "voices": d["voices"], "coupled": d["coupled"],
                   "instances": d["instances"], "authoring_voice": d["authoring_voice"],
                   "params": d.get("params"), "dirty_params": _dirty(d),
                   "seconds": d["seconds"]} for n, d in docs.items()}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    bad = [n for n, d in docs.items() if d["coupled"]]
    print(f"\n{len(names) - len(bad)} of {len(names)} types read the same in both voices"
          + (f"; coupled to a silhouette: {', '.join(sorted(bad))}" if bad else ""))
    # B1's own answer, and the one this sweep exists for: which types are dirty, and at
    # which of their own declared parameters. A type clean at the low end of every range
    # and dirty three storeys up is a type the city finds out about.
    dirty = {n: _dirty(d) for n, d in docs.items() if _dirty(d)}
    print(f"{len(names) - len(dirty)} of {len(names)} types are clean at every set of "
          f"parameters they declare")
    for n in sorted(dirty):
        for k, d in sorted(dirty[n].items()):
            print(f"   {n:12s} {k:34s} {d['errors']} err, {d['entry_lines']} on foot"
                  + (f", {d['roofless']} no roof" if d.get("roofless") else "")
                  + f"  {', '.join(d['codes'])}")
    return len(bad)


def _dirty(doc: dict) -> dict:
    """The parameter sets a type is not clean at. B1."""
    return {k: d for k, d in (doc.get("params") or {}).items()
            if d["errors"] or d["crashed"] or d.get("roofless")}


def _say(d: dict) -> None:
    line = f"{d['type']:12s} {d['seconds']:7.1f}s "
    for v, r in d["voices"].items():
        line += (f" | {v}: {r['clean']}/{r['instances']} clean, {r['errors']} err"
                 + (f", {r['roofless']} no roof" if r.get("roofless") else ""))
    print(line + ("   <-- COUPLED" if d["coupled"] else ""), flush=True)


if __name__ == "__main__":
    sys.exit(main())
