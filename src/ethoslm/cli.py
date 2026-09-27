"""**Start from a sentence**: the newcomer's entry point over the round controller.

    scripts/ethoslm start "Build a small fishing village on a lake." --name lakeside \
        --world /path/to/an/unaltered/save
    scripts/ethoslm status lakeside
    scripts/ethoslm resume lakeside
    scripts/ethoslm views lakeside [--move]
    scripts/ethoslm deliver lakeside --save /path/to/a/copy/of/the/save --host http://localhost:9000
    scripts/ethoslm doctor [--world SAVE]

A place is built in stages, and some stages are **jobs for an agent**: reading the
sentence into a programme, the place spec, the design and its adoption. Nothing here
needs a model API. The command stops at each job and says exactly which prompt to read
and which file to write; a terminal agent (or a person) writes the answer and runs
`resume`. Waiting on an answer is not success, and `status` says so.

Everything a run makes is under `out/<name>/` (the state) and `out/<name>-atlas/` (the
terrain atlas of the save). Construction is **dry**: the regions are built in that
state, off the save's unaltered ground, and nothing touches a world until `deliver`,
which writes the built regions into the save you name through a running GDMC-HTTP server
(and snapshots that save first).

This module wraps `pipeline.run` and the design stages; it is not a second runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time

from .pipeline.round import ROOT

#: The stages a sentence-to-place run drives, in order (`pipeline.round.DESIGN_DRY`).
STAGES = ("reading", "interpret", "place_spec", "planning", "design_references", "design",
          "design_compare", "design_resolve", "regions", "region_views")

#: Python modules a run needs, and what for. `cv2` only textures the views.
NEEDS = (("numpy", "numpy", True), ("scipy", "scipy", True), ("PIL", "pillow", True),
         ("nbt", "NBT", True), ("gdpc", "gdpc==8.1.0", True),
         ("cv2", "opencv-python-headless", False), ("requests", "requests", True))

NAME = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


def _state(name: str) -> str:
    return os.path.join(ROOT, "out", name)


def _config(name: str) -> str:
    # named for the place, so the controller reports into `round.json` beside it
    return os.path.join(_state(name), f"{name}.json")


def _lock(name: str) -> str:
    return os.path.join(_state(name), ".running")


def say(*lines) -> None:
    for ln in lines:
        print(ln, flush=True)


# ---------------------------------------------------------------- doctor

def doctor(world: str | None = None, *, quiet: bool = False) -> list:
    """What is missing before any expensive work: modules, and the save if one is
    named. Returns the problems (empty when ready)."""
    bad = []
    for mod, pkg, required in NEEDS:
        try:
            importlib.import_module(mod)
        except Exception:                          # noqa: BLE001 -- reported
            if required:
                bad.append(f"python module `{mod}` is missing: pip install {pkg}")
            elif not quiet:
                say(f"note: `{mod}` is not installed ({pkg}); views are drawn in flat "
                    f"colours")
    if world is not None:
        if not os.path.isdir(world):
            bad.append(f"the save {world} is not a directory")
        elif not os.path.exists(os.path.join(world, "level.dat")):
            bad.append(f"{world} has no level.dat: name the save's own directory")
        elif not os.path.isdir(os.path.join(world, "region")) or not any(
                f.endswith(".mca") for f in os.listdir(os.path.join(world, "region"))):
            bad.append(f"{world}/region holds no region files: generate terrain first "
                       f"(open the save in Minecraft or a server and let it generate, "
                       f"or pre-generate with a chunk generator)")
    jar = os.environ.get("ETHOSLM_MC_JAR")
    if not quiet and not jar:
        say("note: ETHOSLM_MC_JAR is not set; views are drawn in flat colours unless the "
            "client jar is found under run/chunky")
    return bad


# ---------------------------------------------------------------- the run

def _round(name: str):
    from . import pipeline
    return pipeline.Round.load(_config(name))


def _bound_memory(rnd) -> None:
    try:
        import resource
        kb = int(os.environ.get("ETHOSLM_MEMORY_KB") or rnd.flags.get("memory_kb")
                 or 12_000_000)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft == resource.RLIM_INFINITY or soft > kb * 1024:
            resource.setrlimit(resource.RLIMIT_AS, (kb * 1024, hard))
    except (ImportError, ValueError, OSError):
        pass


def drive(name: str, stages=STAGES) -> dict:
    """Run the controller over `stages` for this place, holding a lock so `status` can
    say it is working, and return its results."""
    from . import pipeline
    rnd = _round(name)
    _bound_memory(rnd)
    lock = _lock(name)
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().split()[0])
            os.kill(pid, 0)
            raise SystemExit(f"{name} is already running (pid {pid}); see `status`")
        except (ProcessLookupError, ValueError):
            pass
    open(lock, "w").write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    try:
        be = pipeline.OfflineBackend(rnd, dry_run=True)
        return pipeline.run(rnd, list(stages), be)
    finally:
        os.remove(lock)


def start(a) -> int:
    if not NAME.match(a.name or ""):
        raise SystemExit("--name is a short lower-case slug: letters, digits and hyphens")
    world = os.path.abspath(a.world)
    bad = doctor(world)
    if bad:
        say("Not ready:", *(f"  - {b}" for b in bad))
        return 2
    if os.path.exists(_config(a.name)) and not a.again:
        raise SystemExit(f"{a.name} already exists at {os.path.relpath(_state(a.name))}: "
                         f"`resume` it, or give another --name")
    os.makedirs(_state(a.name), exist_ok=True)
    atlas_dir = os.path.abspath(a.atlas) if a.atlas else os.path.join(
        ROOT, "out", f"{a.name}-atlas")
    from . import atlas
    say(f"reading the save's terrain into {os.path.relpath(atlas_dir)} "
        f"(heights, water and biomes; once per save)")
    rec = atlas.build(world, atlas_dir, workers=a.workers)
    if not rec.get("read") and not rec.get("cached"):
        say("the save has no readable terrain", json.dumps(rec)[:400])
        return 2
    cfg = {
        "name": a.name,
        "sentence": a.sentence.strip(),
        "intent": "started from a sentence by `ethoslm start`",
        "flags": {"dry_run": True, "seed": int(a.seed), "workers": int(a.workers),
                  "memory_kb": int(a.memory_gb * 1_000_000),
                  "design": {"world": world, "terrain": world, "atlas": atlas_dir,
                             "massing": False, "build": "all", "views": "auto",
                             "workers": int(a.workers)}},
    }
    json.dump(cfg, open(_config(a.name), "w"), indent=1)
    say(f"started {a.name}: state {os.path.relpath(_state(a.name))}")
    drive(a.name)
    return report(a.name)


def resume(a) -> int:
    if not os.path.exists(_config(a.name)):
        raise SystemExit(f"no place called {a.name}: `start` one")
    stages = STAGES if not a.no_build else STAGES[:STAGES.index("regions")]
    drive(a.name, stages)
    return report(a.name)


# ---------------------------------------------------------------- status

def _delivery_current(name: str, regions: dict) -> bool:
    """A delivery receipt covers the current built diffs in the selected save."""
    try:
        st = _state(name)
        with open(os.path.join(st, "written.json")) as f:
            receipt = json.load(f)
        with open(_config(name)) as f:
            world = (json.load(f).get("flags", {}).get("design") or {}).get("world")
        if not world or not receipt.get("world") or receipt.get("complete") is False:
            return False
        if os.path.realpath(os.path.join(ROOT, receipt["world"])) != os.path.realpath(
                os.path.join(ROOT, world)):
            return False
        wanted = set(regions.get("built") or []) | set(regions.get("kept") or [])
        if not wanted:
            return False
        for rid in wanted:
            row = receipt.get("regions", {}).get(rid) or {}
            if row.get("status", "complete") != "complete" or row.get("failed"):
                return False
            with open(os.path.join(st, "regions", rid, "diff.npz"), "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            if row.get("diff") != digest:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def state_of(name: str) -> dict:
    """Where this place stands: `working`, `waiting` (on an agent), `blocked`,
    `designed` (built dry, views drawn), `delivered`, or `new`, with the next action."""
    st = _state(name)
    if os.path.exists(_lock(name)):
        pid = open(_lock(name)).read().split()[0]
        try:
            os.kill(int(pid), 0)
            return {"state": "working", "next": f"wait: the run (pid {pid}) is working; "
                                                 f"`status {name}` again later"}
        except (ProcessLookupError, ValueError):
            pass
    rj = os.path.join(st, "round.json")
    if not os.path.exists(rj):
        return {"state": "new", "next": f"scripts/ethoslm resume {name}"}
    res = (json.load(open(rj)).get("results") or {})
    if res.get("pending"):
        p = res["pending"]
        asks = p.get("waiting") or []
        return {"state": "waiting", "stage": p.get("stage"), "jobs": asks,
                "next": "an agent answers each job -- READ the prompt, WRITE the answer "
                        f"file -- then: scripts/ethoslm resume {name}"}
    if res.get("stopped"):
        s = res["stopped"]
        return {"state": "blocked", "stage": s.get("stage"), "why": s.get("why"),
                "next": "read the reason; fix the input it names (or answer the job it "
                        f"hands back), then: scripts/ethoslm resume {name}"}
    regions = res.get("regions") or {}
    views = res.get("region_views") or {}
    written = _delivery_current(name, regions)
    if regions.get("built") is not None or regions.get("kept"):
        failed = regions.get("failed") or []
        out = {"state": "delivered" if written else "designed",
               "regions": {"built": regions.get("built"), "kept": regions.get("kept"),
                           "failed": failed},
               "views": views.get("views") or [],
               "next": (f"look at the views; deliver with: scripts/ethoslm deliver {name} "
                        f"--save <a copy of the save> --host <its GDMC-HTTP address>")}
        part_failures = {rid: row["failed"] for rid, row in
                         (regions.get("regions") or {}).items() if row.get("failed")}
        if failed or part_failures:
            out["state"] = "blocked"
            out["next"] = (f"construction failed: regions {failed}, parts {part_failures}; "
                           f"see out/{name}/regions/")
        return out
    return {"state": "incomplete", "next": f"scripts/ethoslm resume {name}"}


def _planning(name: str) -> dict:
    try:
        with open(os.path.join(_state(name), "planning.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def report(name: str) -> int:
    s = state_of(name)
    say("", f"{name}: {s['state'].upper()}")
    pl = _planning(name)
    if pl:
        say(f"  planning: {pl.get('strategy')} -- {pl.get('why')}")
        for u in pl.get("unsupported") or []:
            say(f"  unsupported: {u['part']}: {u['why']}")
    if s.get("stage"):
        say(f"  at stage: {s['stage']}")
    for j in s.get("jobs") or []:
        say(f"  job ({j.get('role') or j.get('entry')}):",
            f"    READ  {os.path.relpath(j['request']) if j.get('request') else '-'}",
            f"    WRITE {os.path.relpath(j['write']) if j.get('write') else '-'}")
        if j.get("check"):
            say(f"    CHECK {j['check']}")
        if j.get("note"):
            say(f"    NOTE  {j['note']}")
    if s.get("why"):
        say(f"  why: {s['why']}")
    if s.get("regions"):
        say(f"  regions: {json.dumps(s['regions'])}")
    try:
        with open(os.path.join(_state(name), "composition.built.json")) as f:
            cb = json.load(f)
        say(f"  composition: {cb['built']}/{len(cb['buildings'])} buildings built "
            f"({json.dumps(cb['uses_built'])}); spaces "
            + ", ".join(f"{sp['id']} {sp['built']}/{sp['tiles']}" for sp in cb["spaces"]))
        for bid, what in (cb.get("not_delivered") or {}).items():
            say(f"  not delivered: {bid} was asked for {', '.join(what)} and the blocks do "
                f"not show it (revise its lot or its choice)")
    except (OSError, ValueError, KeyError):
        pass
    for v in s.get("views") or []:
        say(f"  view: {v}")
    say(f"  next: {s['next']}")
    return 0 if s["state"] in ("designed", "delivered") else 3 \
        if s["state"] in ("waiting", "working") else 1


def status(a) -> int:
    if not os.path.exists(_config(a.name)):
        raise SystemExit(f"no place called {a.name}")
    return report(a.name)


# ---------------------------------------------------------------- views and delivery

def views(a) -> int:
    rnd = _round(a.name)
    from .pipeline import stages_design as SD
    res = SD.stage_region_views(rnd, None, {})
    for v in res.get("views") or []:
        say(f"view: {v}")
    if a.move:
        got = SD.place_move(rnd, frames=a.frames, size=(a.width, a.width * 9 // 16))
        say(f"move: {got['gif']} ({got['frames']} frames, {got['seconds']} s)",
            "in game, the same move:", *(f"  {c}" for c in got["commands"]))
    return 0


def deliver(a) -> int:
    rnd = _round(a.name)
    s = state_of(a.name)
    if s["state"] not in ("designed", "delivered"):
        raise SystemExit(f"{a.name} is {s['state']}, not built: {s['next']}")
    save = os.path.abspath(a.save)
    cfg = rnd.flags.get("design") or {}
    if os.path.realpath(save) == os.path.realpath(cfg.get("terrain") or ""):
        raise SystemExit("the save to deliver into is the save the terrain was read from; "
                         "deliver into a copy of it, so the ground as found stays readable")
    if not os.path.exists(os.path.join(save, "level.dat")):
        raise SystemExit(f"{save} is not a save (no level.dat)")
    try:
        import requests
        requests.get(a.host.rstrip("/") + "/version", timeout=5).raise_for_status()
    except Exception as e:                         # noqa: BLE001 -- explained
        raise SystemExit(
            f"no GDMC-HTTP server answers at {a.host} ({type(e).__name__}). Delivery "
            f"writes through a running Minecraft server serving {save} with the "
            f"GDMC-HTTP mod; start one on that save (never on a save another server has "
            f"open), then run this again. See README, 'Quick start', step 4.")
    if not a.yes:
        n = len(os.listdir(rnd.rel("regions")))
        say(f"This writes {n} built region(s) of {a.name} into {save} through {a.host}.",
            "The save is snapshotted first under run/snapshots/. Pass --yes to write.")
        return 1
    cfg = dict(cfg, world=save, host=a.host.rstrip("/"))
    rnd.flags = dict(rnd.flags, design=cfg)
    from .pipeline import stages_design as SD
    res = SD.stage_region_write(rnd, None, {})
    if res.get("stop"):
        say(f"not delivered: {res.get('error')}")
        return 1
    doc = json.load(open(_config(a.name)))
    doc["flags"]["design"].update(world=save, host=cfg["host"])
    json.dump(doc, open(_config(a.name), "w"), indent=1)
    say(f"delivered: wrote {res.get('written')}, kept {res.get('kept')}; record "
        f"{res.get('record')}")
    pose = SD.place_pose(rnd)
    if pose:
        say(f"stand here in game: {pose}")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ethoslm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="start a place from a sentence")
    p.add_argument("sentence")
    p.add_argument("--name", required=True, help="a short slug; the state is out/<name>")
    p.add_argument("--world", required=True,
                   help="the save whose unaltered terrain the place is designed on "
                        "(read only)")
    p.add_argument("--atlas", help="where the terrain atlas goes (default out/<name>-atlas)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--memory-gb", type=float, default=12.0, dest="memory_gb")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--again", action="store_true",
                   help="rewrite the config of an existing place (keeps its state)")
    p.set_defaults(fn=start)
    p = sub.add_parser("status", help="where a place stands and what to do next")
    p.add_argument("name")
    p.set_defaults(fn=status)
    p = sub.add_parser("resume", help="continue after a job is answered")
    p.add_argument("name")
    p.add_argument("--no-build", action="store_true", dest="no_build",
                   help="stop after the design is resolved; build nothing")
    p.set_defaults(fn=resume)
    p = sub.add_parser("views", help="draw the built place again, and a camera move")
    p.add_argument("name")
    p.add_argument("--move", action="store_true", help="also a five-second move (GIF)")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--width", type=int, default=640)
    p.set_defaults(fn=views)
    p = sub.add_parser("deliver", help="write the built place into a save")
    p.add_argument("name")
    p.add_argument("--save", required=True, help="the save to write into (a copy)")
    p.add_argument("--host", default="http://localhost:9000",
                   help="the GDMC-HTTP address of the server serving that save")
    p.add_argument("--yes", action="store_true", help="write (otherwise only explain)")
    p.set_defaults(fn=deliver)
    p = sub.add_parser("doctor", help="check the environment")
    p.add_argument("--world")
    p.set_defaults(fn=lambda a: (lambda bad: (say(*(bad or ["ready"])), 1 if bad else 0)[1])(
        doctor(os.path.abspath(a.world) if a.world else None)))
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
