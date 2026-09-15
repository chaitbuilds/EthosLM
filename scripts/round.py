"""Run a round. The whole command.

A round is `rounds/<name>.json` -- site, voice, intent, pipeline flags, wave assignment,
shot list, card sets, judgements and their pre-registered thresholds. Adding a round
means adding a file, not a script. Results land in `out/<name>/round.json` beside the
pre-registration they are judged against.

Default stages are the deterministic ones: execute the stored wave programs, compose the
cards, replay the judgements. `--stage` names others; `render` needs Chunky and `--live`
gives the stages that write blocks a server.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="path to rounds/<name>.json")
    ap.add_argument("--stage", default="",
                    help="comma-separated; known: "
                         + ",".join(pipeline.STAGES)
                         + ". The default is the deterministic three, or -- for a round "
                           "that carries a `sentence` -- the whole of `pipeline.PLACE`, "
                           "because a place is one config and one command.")
    ap.add_argument("--live", action="store_true",
                    help="use a running server as the backend")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="the whole place, from the sentence to the readout, with the "
                         "world in a volume and no server anywhere: the search reads "
                         "the squares out/sites/ already holds, the plateau is cut into "
                         "the cached volume rather than into the world, and the renders "
                         "are skipped. This is the run that is iterated until every bar "
                         "holds; the live one is not the test.")
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="a stage waiting on a builder sleeps and asks again, up to "
                         "this long. What makes a live settlement round one command: "
                         "the server dies with the shell that started it, so the "
                         "builders have to be answered while it is still up.")
    ap.add_argument("--poll", type=int, default=300, metavar="SECONDS",
                    help="how often --wait looks again")
    ap.add_argument("--measure", default="", metavar="NAME[,NAME]",
                    help="with --stage readout: re-read only these bars, under the "
                         "instruments as they now stand, into readout.reread.json "
                         "beside the recorded readout rather than over it. What a "
                         "corrected ruler is re-read with: the recorded row stays, the "
                         "re-read sits beside it, and SUPERSEDED names the cause.")
    a = ap.parse_args()

    # A runaway fill dies by name rather than taking the host. A place of 768 with seven
    # hundred leaves solved its circulation in fourteen gigabytes and the twelve-
    # gigabyte default killed it with no message at all -- an address-space bound this
    # process cannot raise is a SIGKILL, not a `MemoryError`, so nothing is logged and
    # nothing names the cause. A bound that decides whether a round finishes belongs in
    # the round file beside the other bounds, not in an environment variable somebody
    # has to remember.
    try:
        import resource
        wants_render = "render" in (a.stage or "") or (
            not a.stage and getattr(a, "dry_run", False) is False)
        try:
            from_round = int((json.load(open(a.config)).get("flags") or {})
                             .get("memory_kb") or 0)
        except Exception:                        # noqa: BLE001 -- the loader reports it
            from_round = 0
        kb = int(os.environ.get("ETHOSLM_MEMORY_KB") or from_round
                 or (20_000_000 if wants_render else 12_000_000))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        want = kb * 1024
        if soft == resource.RLIM_INFINITY or soft > want:
            resource.setrlimit(resource.RLIMIT_AS, (want, hard))
    except (ImportError, ValueError, OSError):
        pass

    rnd = pipeline.Round.load(a.config)
    if a.dry_run:
        if a.live:
            ap.error("--dry-run and --live are the two answers to the same question")
        rnd.flags = dict(rnd.flags, dry_run=True)
    if a.measure:
        rnd.flags = dict(rnd.flags, measures=[m.strip() for m in a.measure.split(",")
                                              if m.strip()])
    stages = [s.strip() for s in a.stage.split(",") if s.strip()] \
        or list(pipeline.default_stages(rnd))
    backend = pipeline.LiveBackend(rnd) if a.live \
        else pipeline.OfflineBackend(rnd, dry_run=a.dry_run)
    res = pipeline.run(rnd, stages, backend, wait=a.wait, poll=a.poll)

    bad = 0
    for name in stages:
        r = res.get(name, {})
        if name == "programs":
            for w, rec in sorted(r.items()):
                if not isinstance(rec, dict):
                    continue
                mark = ("OK " if rec.get("reproduces") else
                        "   " if rec.get("recorded") is None else "!! ")
                bad += rec.get("reproduces") is False
                print(f"  {mark}{w}: {rec.get('pending', rec.get('error'))}"
                      + (f"  (recorded {rec['recorded']})"
                         if rec.get("recorded") is not None else ""))
        elif name == "judge":
            for jn, jr in r.items():
                if jn == "seconds":
                    continue
                if jr.get("status") != "done":
                    print(f"  !! {jn}: {jr}")
                    bad += 1
                    continue
                if "winners" in jr:
                    print(f"  {jn}: " + ", ".join(
                        f"{k} -> {v}" for k, v in jr["winners"].items()))
                    continue
                print(f"  {jn}: {jr['correct']}/{jr['judged']} correct, "
                      f"{jr['ties']} ties, rate {jr['correct_rate']}"
                      + (f", pass={jr['pass']}" if "pass" in jr else ""))
        elif name == "cards":
            for cn, cr in r.items():
                if cn == "seconds":
                    continue
                print(f"  {cn}: {cr['composed']} composed, "
                      f"{len(cr['refused'])} refused")
        elif name == "waves":
            for w, rec in sorted(r.items()):
                if not isinstance(rec, dict) or "status" not in rec:
                    continue          # "adopted", "seconds": not a wave
                st = rec["status"]
                if st == "needs_model":
                    bl = rec.get("blinded") or {}
                    print(f"  .. {w}: needs a builder")
                    print(f"       BRIEF {bl.get('brief', rec['request'])}")
                    print(f"       WRITE {bl.get('write')}")
                    if bl.get("check"):
                        print(f"       CHECK {bl['check']}")
                    print(f"       THEN TOUCH {bl.get('done')}")
                    bad += 1
                elif st in ("error", "no_brief", "crashed"):
                    print(f"  !! {w}: {rec.get('error', st)}")
                    bl = (rec.get("blinded") or {})
                    if rec.get("bounced"):
                        print(f"       bounce {rec['bounce']}, retired "
                              f"{rec['retired']}")
                        print(f"       BRIEF {bl.get('brief')}")
                        print(f"       ERROR {bl.get('error')}")
                        print(f"       WRITE {bl.get('write')}")
                    bad += 1
                else:
                    runs = rec.get("checks") or []
                    print(f"  OK {w}: {rec.get('pending')} blocks, "
                          f"{len(runs)} check run(s), lint {rec.get('lint')}"
                          + (f", commit {rec['commit'].get('placed')} placed"
                             if rec.get("commit") else ""))
        elif name in ("candidates", "measures"):
            for cid, rec in sorted(r.items()):
                if not isinstance(rec, dict) or "status" not in rec:
                    continue      # "adopted", "written", "seconds": not a candidate
                st = rec["status"]
                if st == "needs_model":
                    bl = rec.get("blinded") or {}
                    print(f"  .. {cid}: needs a builder")
                    print(f"       BRIEF {bl.get('brief', rec['request'])}")
                    print(f"       WRITE {bl.get('write')}")
                    if bl.get("check"):
                        print(f"       CHECK {bl['check']}")
                    for im in bl.get("images") or []:
                        print(f"       IMAGE {im}")
                    print(f"       THEN TOUCH {bl.get('done')}")
                    bad += 1
                elif st in ("error", "no_program", "crashed"):
                    print(f"  !! {cid}: {rec.get('error', st)}")
                    bl = rec.get("blinded") or {}
                    if rec.get("bounced"):
                        print(f"       bounce {rec['bounce']}, retired "
                              f"{rec['retired']}")
                        print(f"       BRIEF {bl.get('brief')}")
                        print(f"       ERROR {bl.get('error')}")
                        print(f"       WRITE {bl.get('write')}")
                        for im in bl.get("images") or []:
                            print(f"       IMAGE {im}")
                    bad += 1
                elif name == "candidates":
                    print(f"  OK {cid}: {rec['pending']} blocks, conformance "
                          f"{rec.get('conformance')}, lint {rec.get('lint')}")
                else:
                    print(f"  OK {cid}: {rec['blocks']} blocks, "
                          f"{rec['lint_errors']} lint errors, walk "
                          f"{rec['walk_pct']}%, ridge cv "
                          f"{rec['variety_ridge_cv']}")
        elif name == "revise":
            for cid, rec in sorted(r.get("candidates", {}).items()):
                st = rec.get("status")
                walks = [d["measures"].get("walk_pct") for d in rec.get("drafts", [])]
                trail = " -> ".join(f"{w}%" for w in walks) if walks else "-"
                if st == "needs_model":
                    bl = rec.get("blinded") or {}
                    print(f"  .. {cid}: draft {rec['awaiting_draft']} needs a builder "
                          f"({trail})")
                    print(f"       BRIEF    {bl.get('brief')}")
                    print(f"       FINDINGS {bl.get('findings')}")
                    print(f"       WRITE    {bl.get('write')}")
                    print(f"       THEN TOUCH {bl.get('done')}")
                    bad += 1
                elif st in ("crashed", "no_program"):
                    print(f"  !! {cid}: {rec.get('error', st)}")
                    bl = rec.get("blinded") or {}
                    if rec.get("bounced"):
                        print(f"       bounce {rec['bounce']}, retired "
                              f"{rec['retired']}")
                        print(f"       BRIEF    {bl.get('brief')}")
                        print(f"       ERROR    {bl.get('error')}")
                        print(f"       WRITE    {bl.get('write')}")
                    bad += 1
                else:
                    print(f"  OK {cid}: {st} at draft "
                          f"{rec.get('clean_at', len(walks) - 1)}  {trail}")
            rr = r.get("reproduces_recorded") or {}
            if rr.get("checked"):
                print(f"  draft 0 reproduces the recorded rows: "
                      f"{rr['reproduce']}/{rr['checked']}"
                      + (f"  MISMATCHED {list(rr['mismatched'])}"
                         if rr.get("mismatched") else ""))
                bad += rr["reproduce"] != rr["checked"]
            ro = r.get("readout") or {}
            if ro:
                p, s = ro["primary"], ro["secondary"]
                print(f"  primary: {p['below_threshold']}/{p['of']} below "
                      f"{ro['threshold']}% (first drafts: 9 of 16) -> "
                      f"{p['branch'] or 'incomplete'}"
                      f"   repaired-by-deletion {len(p['repaired_by_deletion'])}")
                print(f"  secondary: pooled within-prompt SD "
                      f"{s['pooled_within_prompt_sd']} (bar 14.4)  "
                      f"per prompt {s['per_prompt_sd']}")
                print(f"  own lint errors at 0: {ro['own_lint_errors_zero']['n']}"
                      f"/{ro['own_lint_errors_zero']['of']} (bar >=14)")
        elif name == "selection":
            if r.get("tie_rate") is None:
                print("  " + json.dumps(r)[:400])
            else:
                print(f"  ties {r['ties']}/{r['judged']} = {r['tie_rate']} "
                      f"(bar {r['tie_rate_bar']}) -> "
                      f"{'discriminates' if r['discriminates'] else 'STOPS AT STAGE 1'}")
                print(f"  copeland {r['copeland']}  winner {r['winner']} "
                      f"({r['winner_by']})")
                cv = r.get("convergent") or {}
                if cv.get("verdict"):
                    print(f"  unseen measures: {cv['verdict']} "
                          f"({cv['above_median']} above / {cv['below_median']} "
                          f"below median of 3)")
                hl = r.get("human_look") or {}
                if hl.get("status"):
                    print(f"  human look: {hl['status']} -- {hl.get('dir')}")
        else:
            print("  " + json.dumps(r)[:400])
    print(f"\n-> {res.get('round_json') or rnd.rel('readout.reread.json')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
