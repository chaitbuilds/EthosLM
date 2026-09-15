"""Blinded build requests, checker jobs and response collection."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import time
from dataclasses import dataclass, field

from .. import pipeline as _pipeline
from .. import card as card_mod
from .. import measure as measure_mod
from .. import offline, parallel, settlement, verdicts
from ..measure import record
from .round import Round


BUILD_SCRATCH = os.path.join(_pipeline.ROOT, "out", "build_scratch")


def _blind_dir(rnd: Round, sub: str, i: int) -> str:
    """`sub` is the candidate's arm path (`selection/wave1/c2`). Taken as a string
    rather than looked up from a candidate id so the arms of a sight round, which are
    not selection candidates, blind through exactly this code and hash to the same
    scheme."""
    import hashlib
    key = f"{rnd.name}/{sub}/{i}"
    return os.path.join(BUILD_SCRATCH,
                        hashlib.sha256(key.encode()).hexdigest()[:16])


CHECK_LINES = """

## Before you finish

`check.py` is in this directory. Run it -- `python check.py` -- and it executes your
program against the real ground it will be built on, runs the checks the finished
build is held to, and rewrites `findings.md` beside it with what they say. It takes
about forty seconds and changes nothing you have written.

Run it as often as you like: after the massing, after the openings, after the
fittings, whenever you want to know what is actually there rather than what you think
is there. Read `findings.md` each time and fix what it names.

Run it once more on your finished program before you write `done`. A `done` written
without it is not accepted.
"""


def _blind_build(rnd: Round, sub: str, req_path: str,
                 check: tuple | None = None, role: str | None = None) -> dict:
    """Stage a build request where nothing about the candidate reaches the builder.

        `stages.build_from_files` writes its request under `arms/<arm>/<wave>/<cid>/builds/`,
        and for a selection round **that path is itself the leak**: a builder handed
        `.../selection/wave1/c2/builds/build_0.py` has been told precisely the thing the
        spec says it must not be told -- that it is one of several and that something will
        be compared. So the brief and any images are copied into a hash-named scratch
        directory, exactly the blinding `judge.stage` applies to a card, and the program is
        copied back by `_collect_blinded` on the next run.

        A revision request carries a second document. `findings` is what the deterministic
        checks read out of the world the previous draft built, and it is written beside the
        brief as `findings.md` rather than folded into it, because the two are different
        questions: the brief is what to build, the findings are what is wrong with the build
        already in front of you. The directory holds those two and the file the builder
        writes back, and nothing else -- a directory listing is itself a message.

        `check` is `(plots, wave)` and adds the fourth file, `check.py`. It carries a key
        and nothing else; the job it names lives under `out/check_jobs/`, outside anything
        the builder can see, so the directory listing still says nothing.

        `role` is the model role the job is submitted under -- `type` for a type author,
        `revise` for a findings revision, `build` otherwise -- and `model.router()` decides
        whether that role is answered by the supervising agent (the default) or by an API
        model. The record carries the role so the driver can say which.
        
    """
    import shutil
    i = int(os.path.basename(req_path)[len("build_request_"):-len(".json")])
    req = json.load(open(req_path))
    d = _blind_dir(rnd, sub, i)
    os.makedirs(d, exist_ok=True)
    brief = req["brief"]
    out = {"dir": d, "brief": os.path.join(d, "brief.md"),
           "write": os.path.join(d, "program.py"),
           "done": os.path.join(d, "done"), "i": i}
    if check is not None and check[0] == "type":
        # The type checker: same `check.py`, same key, a different job behind it. The
        # builder's directory listing is identical either way, which is the point.
        out["check"] = stage_type_check(rnd, d, os.path.basename(d), check[1])
        brief = brief.rstrip("\n") + TYPE_CHECK_LINES
    elif check is not None:
        out["check"] = stage_check(rnd, d, os.path.basename(d), check[0], check[1])
        brief = brief.rstrip("\n") + CHECK_LINES
    open(os.path.join(d, "brief.md"), "w").write(brief)
    if req.get("findings"):
        open(os.path.join(d, "findings.md"), "w").write(req["findings"])
        out["findings"] = os.path.join(d, "findings.md")
    imgs = []
    for src in req.get("images") or []:
        dst = os.path.join(d, os.path.basename(src).split("_", 1)[-1])
        shutil.copyfile(src, dst)
        imgs.append(dst)
    out["images"] = imgs
    if role is None:
        role = "type" if (check is not None and check[0] == "type") else "build"
    out["role"] = role
    from ..model import router
    return router().builder(role, out).submit(out)


def _collect_blinded(rnd: Round, subs=None) -> dict:
    """Adopt any blinded program a builder has answered. Copy, never move: the scratch
    directory stays as the record of exactly what that builder was shown."""
    import hashlib
    import shutil
    got = {}
    if subs is None:
        subs = [(cid, _pipeline._cand_arm(rnd, cid)) for cid in _pipeline._cand_ids(rnd)]
    for cid, sub in subs:
        builds = rnd.rel("arms", sub, "builds")
        if not os.path.isdir(builds):
            continue
        listing = sorted(os.listdir(builds))
        for f in listing:
            if not f.startswith("build_request_"):
                continue
            i = int(f[len("build_request_"):-len(".json")])
            dest = os.path.join(builds, f"build_{i}.py")
            src = os.path.join(_blind_dir(rnd, sub, i), "program.py")
            if os.path.exists(dest) or not os.path.exists(src):
                continue
            # A builder writes its program in chunks -- 150 lines at a time, because a
            # single long tool call is truncated by the output cap -- so a program.py on
            # disk is not necessarily a finished one, and adopting a half-written
            # program would put a truncated candidate into the experiment and read as a
            # bad builder. The builder therefore says when it is done, and the driver
            # believes the marker rather than the mtime.
            if (rnd.candidates or rnd.arms or rnd.revise).get("require_done", True) \
                    and not os.path.exists(
                    os.path.join(os.path.dirname(src), "done")):
                got[f"{cid}/{i}"] = "no `done` marker -- builder still writing"
                continue
            # A bounce retires the crashed program and leaves the scratch directory
            # holding it until the builder overwrites it. Re-adopting those bytes would
            # replay the same crash for ever and read as a builder that could not be
            # taught, so the retired programs are checked against by content.
            sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
            retired = {hashlib.sha256(open(os.path.join(builds, c), "rb").read())
                       .hexdigest() for c in listing
                       if c.startswith(f"build_{i}.crashed")}
            if sha in retired:
                got[f"{cid}/{i}"] = "unchanged since the bounce -- not adopted"
                continue
            # `done` is refused if the checker was never run. The record of a run is the
            # `check_run` row on the measurement log rather than a file in the
            # directory, so the refusal costs the builder no extra thing to see and the
            # count of runs per builder is on the log whether or not anybody asks.
            d = os.path.dirname(src)
            if os.path.exists(os.path.join(d, "check.py")):
                runs = checked_runs(os.path.basename(d))
                if not runs:
                    got[f"{cid}/{i}"] = ("`done` without ever running check.py -- "
                                         "not adopted")
                    continue
                if runs[-1].get("sha256") != sha:
                    # Reported, not refused: the spec's rule is that the check must have
                    # been run, and a builder that fixed a typo after its last check has
                    # still been through the loop. The fact travels so the round can say
                    # how many finished on a checked program.
                    got[f"{cid}/{i}.note"] = "last check.py run was on earlier bytes"
            shutil.copyfile(src, dest)
            got[f"{cid}/{i}"] = dest
    return got


def scrub_traceback(tb: str) -> str:
    """A traceback with everything but the builder's own program taken out of it.

        Found by a test after three bounces had already gone out. A raw `format_exc` names
        the driver frame that raised (`stage_candidates`) and the absolute path of the
        program it ran -- `.../arms/selection/wave1/c3/wave1.py` -- and a builder shown that
        has been told it is candidate 3 of something called selection, which is the one
        thing the spec says must not happen. The frames from this package and from
        `scripts/` are dropped and what is left is the line that raised and the exception.
        
    """
    keep, drop_next = [], False
    ours = (os.sep + "ethoslm" + os.sep, os.sep + "scripts" + os.sep)
    for line in tb.splitlines():
        s = line.strip()
        if s.startswith('File "'):
            drop_next = any(o in line for o in ours)
            if drop_next:
                continue
            # the path is the leak; the line number in the builder's own file is not
            path = s.split('"')[1]
            keep.append(line.replace(path, "your program"))
            continue
        if drop_next and (line.startswith("    ") or line.startswith("\t")):
            continue
        drop_next = False
        keep.append(line)
    return "\n".join(keep)


def _bounce(rnd: Round, sub: str, builds: str, tb: str) -> dict:
    """Retire a crashed program and stage the identical job again, once.

        A program that preflights and then raises is not a candidate; it is a typo. Step
        3's five arms all tripped the same block rename and were fixed with one message
        each, and this is that, as a mechanism: the crashed program is kept beside its
        replacement, the traceback goes into the *builder's own* blinded directory as
        `error.md`, and the cap comes off the config so nobody can quietly bounce one
        candidate more than another.
        
    """
    # A waves round carries none of the three experiment blocks, so its bounce cap lives
    # in `flags` beside the rest of its pipeline settings.
    block = rnd.candidates or rnd.arms or rnd.revise or rnd.flags
    cap = int(block.get("max_bounces", 0))
    if not cap:
        return {"bounced": False, "why": "max_bounces is 0"}
    i = max((int(f[len("build_request_"):-len(".json")])
             for f in os.listdir(builds) if f.startswith("build_request_")),
            default=0)
    prog = os.path.join(builds, f"build_{i}.py")
    n = 1 + sum(1 for f in os.listdir(builds)
                if f.startswith(f"build_{i}.crashed"))
    if n > cap:
        return {"bounced": False, "why": f"already bounced {n - 1} of {cap}"}
    os.replace(prog, os.path.join(builds, f"build_{i}.crashed{n}.py"))
    d = _blind_dir(rnd, sub, i)
    open(os.path.join(d, "error.md"), "w").write(
        "# Your program did not run\n\nIt passed the block-id check and then raised "
        "when it was executed:\n\n```\n"
        + scrub_traceback(tb).strip()[-2000:] + "\n```\n\n"
        "Every signature you need is written out in the brief. Fix this and write "
        "the whole program again to `program.py` in this directory. Nothing else "
        "about the brief has changed.\n")
    for stale in ("done",):
        if os.path.exists(os.path.join(d, stale)):
            os.remove(os.path.join(d, stale))
    return {"bounced": True, "bounce": n, "retired": f"build_{i}.crashed{n}.py",
            "blinded": {"dir": d, "brief": os.path.join(d, "brief.md"),
                        "error": os.path.join(d, "error.md"),
                        "write": os.path.join(d, "program.py"),
                        "done": os.path.join(d, "done"),
                        "images": sorted(os.path.join(d, f)
                                         for f in os.listdir(d)
                                         if f.endswith(".png"))}}


CHECK_PY = '''"""Check what your program builds. Run me as often as you like:

    python check.py

I execute `program.py` in this directory against the real ground it will be built on,
run the same checks the finished build is held to, and rewrite `findings.md` beside me
with what they say. I write nothing else and I change nothing you have written. About
forty seconds.
"""
import os
import subprocess
import sys

ROOT = {root!r}
KEY = {key!r}
VENV = os.path.join(ROOT, ".venv", "bin", "python")
ENV_SH = os.path.join(ROOT, "scripts", "env.sh")

# The venv's numpy is a native wheel and needs libstdc++ and zlib out of the nix store;
# this host's /lib64 loader is a stub, so without LD_LIBRARY_PATH the import fails and
# so does the first run of this checker. `env.sh` is where the three store paths are
# written down, so it is asked rather than copied -- one path in the project, not two
# that can drift. **The test is whether the store paths are in the variable, not whether
# the variable is set.** This host presets a partial LD_LIBRARY_PATH of its own, so the
# old guard -- `if "LD_LIBRARY_PATH" not in os.environ` -- saw it set, skipped its own
# sourcing, and the import failed anyway.
if os.path.exists(ENV_SH):
    p = subprocess.run(["bash", "-c", 'source "$1"; printf %s "$LD_LIBRARY_PATH"',
                        "_", ENV_SH], capture_output=True, text=True)
    want = p.stdout.strip() if p.returncode == 0 else ""
    if want:
        have = os.environ.get("LD_LIBRARY_PATH", "")
        missing = [d for d in want.split(":") if d and d not in have.split(":")]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                [d for d in [have] if d] + missing) if have else want

if os.path.abspath(sys.executable) != os.path.abspath(VENV) and os.path.exists(VENV):
    os.execv(VENV, [VENV, os.path.abspath(__file__)] + sys.argv[1:])

sys.path.insert(0, os.path.join(ROOT, "src"))
from ethoslm import pipeline                                          # noqa: E402

sys.exit(pipeline.run_check(KEY, os.path.dirname(os.path.abspath(__file__))))
'''


CHECK_JOBS = (os.environ.get("ETHOSLM_CHECK_JOBS")
              or os.path.join(_pipeline.ROOT, "out", "check_jobs"))


def _planned_rects(rnd: Round, labels: list) -> list:
    """The wave's footprints as the planner drew them, in plot-registry shape."""
    want = set(labels)
    return [p for p in rnd.plots() if p["label"] in want]


def stage_check(rnd: Round, d: str, key: str, plots: list, wave: str) -> str:
    """Put `check.py` in a builder's directory and its job where only we can see it."""
    os.makedirs(CHECK_JOBS, exist_ok=True)
    rects = (_planned_rects(rnd, plots)
             if os.path.exists(rnd.rel("plan.json")) else [])
    json.dump({"kind": "program", "config": rnd.path, "plots": list(plots),
               "wave": wave, "key": key, "rects": rects},
              open(os.path.join(CHECK_JOBS, f"{key}.json"), "w"), indent=1)
    p = os.path.join(d, "check.py")
    open(p, "w").write(CHECK_PY.format(root=_pipeline.ROOT, key=key))
    return p


TYPE_CHECK_LINES = """

## Before you finish

`check.py` is in this directory. Run it -- `python check.py` -- and it instantiates the
`build()` in `program.py` on **every piece of ground above, with two different seeds
each, in two different voices** -- the one described above and one with another palette
and another roof silhouette -- runs the checks a finished build is held to on each one,
and rewrites `findings.md` beside it. It takes a few minutes and changes nothing you have
written. Your type has to be clean in **both** voices: it is a form, and the palette and
the roof's silhouette are the voice's.

`findings.md` opens with one line per instance and is then written **worst first**: the
instance with the most wrong with it is the first thing you read. A type is only as good
as its worst instance, which is why it is ordered that way -- an instance that is clean
on one plot and sealed on another is a type that is not finished.

Run it as often as you like: after the massing, after the openings, after the fittings.
Run it once more on your finished program before you write `done`. A `done` written
without it is not accepted.
"""


FIXTURE_ROUNDS = ("site_b", "site_d", "site_e")


FIXTURE_HELD_BACK = "site_f"


FIXTURE_RULES = (
    ("driest_flat", "dry, and the flattest and biggest of the dry"),
    ("most_relief", "the greatest relief anywhere in the three rounds"),
    ("wettest", "the most water"),
    ("smallest", "the fewest columns"),
    ("largest", "the most columns"),
    ("waterside", "some water, but not a lake: the edge of one"),
)


def plot_ground(round_name: str) -> list:
    """Every plot of a round's pre-build cache, measured: columns, relief, water."""
    from .. import observe
    state = os.path.join(_pipeline.ROOT, "out", round_name)
    vol = offline.load_volume(os.path.join(state, "world.npz"))
    h, wet = observe.ground_heights(vol)
    plots = json.load(open(os.path.join(state, "plots.json")))
    seen: dict = {}
    for p in plots:
        seen[p["label"]] = seen.get(p["label"], 0) + 1
    out = []
    for p in plots:
        if seen[p["label"]] > 1:
            continue
        x0, x1 = min(p["x0"], p["x1"]), max(p["x0"], p["x1"])
        z0, z1 = min(p["z0"], p["z1"]), max(p["z0"], p["z1"])
        hs, w = [], 0
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                ix, iz = x - vol.x0, z - vol.z0
                if not (0 <= ix < vol.codes.shape[0] and 0 <= iz < vol.codes.shape[2]):
                    continue
                hs.append(int(h[ix, iz]))
                w += int(bool(wet[ix, iz]))
        if not hs:
            continue
        out.append({"round": round_name, "plot": p["label"], "columns": len(hs),
                    "relief": max(hs) - min(hs),
                    "water_pct": round(100 * w / len(hs), 1)})
    return out


def check_fixtures(rounds=FIXTURE_ROUNDS) -> list:
    """The six plots a type is checked on, chosen by rule.

        One command, from `plots.json` and the caches, reproducing itself: the rules are
        total orders over measured numbers with the round and the label as the last
        tie-break, so there is nothing here for a later reader to have to take on trust.
        Each fixture is taken from the plots the earlier rules have not already taken, so
        six rules give six different plots.
        
    """
    pool = [row for r in rounds for row in plot_ground(r)]
    rules = {
        "driest_flat": lambda p: (p["water_pct"], p["relief"], -p["columns"]),
        "most_relief": lambda p: (-p["relief"], p["water_pct"], -p["columns"]),
        "wettest": lambda p: (-p["water_pct"], -p["columns"]),
        "smallest": lambda p: (p["columns"], p["water_pct"]),
        "largest": lambda p: (-p["columns"], p["water_pct"]),
        # the edge of the water rather than the middle of it: the wettest plot that is
        # still mostly ground, which is where a building actually meets a shoreline
        "waterside": lambda p: (-p["water_pct"] if p["water_pct"] < 50 else 1e9,
                                -p["columns"]),
    }
    # Taken by *label*, not by (round, label): a fixture is named to its author by the
    # label alone -- in the brief's ground section and in `findings.md`'s table -- so
    # two plots called `smithy` from two rounds would be one piece of ground with two
    # heightmaps as far as the builder reading it could tell.
    out, taken = [], set()
    for name, why in FIXTURE_RULES:
        left = [p for p in pool if p["plot"] not in taken]
        pick = min(left, key=lambda p: (*rules[name](p), p["round"], p["plot"]))
        taken.add(pick["plot"])
        out.append({**pick, "rule": name, "why": why})
    return out


PART_FIXTURE_ROUND = "site_b"


PART_EDGE_CELLS = 40


PART_AREA_SIZE = 12


PART_EDGE_LOOP = 41


PART_EDGE_RELIEF = 12


PART_EDGE_CLIFF = 4


#: 151 a side is a closed loop of 600 columns, which is fifteen times the 40-column L
#: and nearly four times the 160-column loop -- and it is the size a ring wall of a
#: 512-block city actually is. Three fixtures have cost this project two rounds between
#: them by being too small to fail on: `wall` was 4 of 4 on the L and 0 of 4 on the
#: loop, and the class it failed on was one no declaration could express. A wall asked
#: to climb sixty blocks of relief over six hundred columns, twenty high and five thick,
#: is a different thing again.
PART_EDGE_BIG = 151

#: ...and the ground it may be cut from, largest first. Every one is a **pre-build**
#: cache -- the volume its round was written *against* -- so nothing is standing on any
#: of it and the only thing the line has to keep clear of is the reserved rectangles.
#: The list is in order of how much ground each holds, and the search takes the first
#: that can hold the ring, because a 600-column loop is bigger than most of what this
#: project has on disk.
PART_EDGE_BIG_SOURCES = (("city_b", "world.npz"),
                         ("town", "world.npz"),
                         ("district", "world.npz"),
                         ("site_c", "world_prebuild.npz"))


def big_loop_fixture(sources=PART_EDGE_BIG_SOURCES, n: int = PART_EDGE_BIG,
                     floor: int = 101) -> dict | None:
    """The city-scale ring: a closed loop of `4n - 4` columns on unbuilt ground. A3.

        The same discipline as the other three -- a total order over measured numbers with
        the position as the last tie-break, computed off a cache already on disk. The relief
        rule is the closed loop's own: nearest `PART_EDGE_RELIEF`, because what a ring has to
        do that a straight run does not is step at its vertices as the ground rises under it,
        and a ring chosen for being flat is a ring that never has to.

        The ring shrinks rather than vanishing. `n` is what the spec asks for and `floor` is
        the smallest ring worth cutting; where no ground on disk holds the full size, the
        largest one that fits is taken and **the answer says so**, because a fixture quietly
        a third smaller than the one that was registered is worse than a miss reported.
        
    """
    for round_name, cache in sources:
        got = None
        for side in range(int(n), int(floor) - 1, -2):
            got = _big_loop(round_name, cache, side)
            if got is not None:
                break
        if got is not None:
            got["asked_for"] = 4 * int(n) - 4
            got["short_by"] = got["asked_for"] - got["columns"]
            got["sources_tried"] = [f"{r}/{c}" for r, c in sources
                                    [:1 + [s[0] for s in sources].index(round_name)]]
            return got
    return None


#: A market square is 12x12 and a palace with a main hall, two side halls and a
#: courtyard wall round them is not: 48 is the plateau this project cuts for the centre
#: of a place, which is the ground a compound is meant to stand on.
PART_AREA_BIG = 48


def big_area_fixture(sources=PART_EDGE_BIG_SOURCES, n: int = PART_AREA_BIG,
                     floor: int = 24) -> dict | None:
    """The compound-scale area: the flattest clear `n x n` on unbuilt ground. A3.

        `check_parts`' own area rule -- flattest first -- on the same ground the city ring is
        cut from, and for the same reason: there is no 48x48 anywhere on a 192-block site
        that its town has not spoken for, and an area type asked to build a palace on a
        12x12 square is being asked the wrong question.
        
    """
    for round_name, cache in sources:
        for side in range(int(n), int(floor) - 1, -4):
            got = _big_area(round_name, cache, side)
            if got is not None:
                got["asked_for"] = int(n)
                got["short_by"] = int(n) - side
                return got
    return None


def _big_area(round_name: str, cache: str, n: int) -> dict | None:
    """One square of one size on one cache, or None. See `big_area_fixture`."""
    from .. import observe
    import numpy as np
    if not os.path.exists(os.path.join(_pipeline.ROOT, "rounds", f"{round_name}.json")):
        return None
    rnd, _be = _fixture_round(round_name)
    p = rnd.rel(cache)
    if not os.path.exists(p):
        return None
    vol = offline.load_volume(p)
    h, wet = observe.ground_heights(vol)
    w, _y, d = vol.codes.shape
    if w < n + 2 or d < n + 2:
        return None
    taken = set()
    for q in _plots_of(rnd):
        for x in range(min(q["x0"], q["x1"]) - 2, max(q["x0"], q["x1"]) + 3):
            for z in range(min(q["z0"], q["z1"]) - 2, max(q["z0"], q["z1"]) + 3):
                taken.add((x, z))
    net = rnd.network()
    for (lx, lz, _ly) in (net.surface() if net else []):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                taken.add((lx + dx, lz + dz))
    # **Nearest the lane, after the flatness** -- `check_parts`' own area rule, which
    # this fixture was cut without. The first `compound` was chosen for flatness alone
    # and landed **146 blocks from the nearest lane cell**: `part["door"]` and
    # `threshold()` come back `None` there, `check_door` answers "no lane within 36
    # blocks of this doorway -- that is a distance, not a defect", and every "cannot be
    # walked to from outdoors" line on it is downstream of the plot being off-network.
    # Two of the eight types were read on it and both said so unprompted. The flattest
    # ground on a site is a mesa top forty blocks above the nearest lane, and a compound
    # nobody can walk to is the wrong fixture for the same reason a square nobody can
    # walk to is.
    lane = [(lx, lz) for (lx, lz, _ly) in (net.surface() if net else [])]

    def to_lane(cx, cz) -> int:
        return min((max(abs(lx - cx), abs(lz - cz)) for (lx, lz) in lane), default=0)

    hh = h.astype(np.int64)
    best = None
    for i in range(1, w - n - 1):
        for j in range(1, d - n - 1):
            if wet[i:i + n, j:j + n].any():
                continue
            if taken and any((vol.x0 + i + a, vol.z0 + j + b) in taken
                             for a in range(0, n, 4) for b in range(0, n, 4)):
                continue
            box = hh[i:i + n, j:j + n]
            key = (int(box.max() - box.min()),
                   to_lane(vol.x0 + i + n // 2, vol.z0 + j + n // 2), i, j)
            if best is None or key < best[0]:
                best = (key, i, j)
    if best is None:
        return None
    _k, i, j = best
    x, z = int(vol.x0 + i), int(vol.z0 + j)
    return {"round": round_name, "cache": cache, "kind": "area", "part": "compound",
            "x0": x, "z0": z, "x1": x + n - 1, "z1": z + n - 1,
            "side": n, "relief": best[0][0],
            "lane_distance": to_lane(x + n // 2, z + n // 2), "rule": "area_compound",
            "why": f"the flattest {n}x{n} of dry, unspoken-for ground on "
                   f"{round_name}'s pre-build cache, nearest the lane -- a "
                   f"compound's worth, against the {PART_AREA_SIZE}x{PART_AREA_SIZE} a "
                   f"market square is"}


def _big_loop(round_name: str, cache: str, n: int) -> dict | None:
    """One ring of one size on one cache, or None. See `big_loop_fixture`."""
    from .. import observe
    if not os.path.exists(os.path.join(_pipeline.ROOT, "rounds", f"{round_name}.json")):
        return None
    rnd, _be = _fixture_round(round_name)
    p = rnd.rel(cache)
    if not os.path.exists(p):
        return None
    vol = offline.load_volume(p)
    h, wet = observe.ground_heights(vol)
    w, _y, d = vol.codes.shape
    if w < n + 2 or d < n + 2:
        return None
    # **The line, and not the ground it encloses.** A ring of 151 a side does not fit
    # anywhere on this cache that is clear of the round's own plot rectangles. What has
    # to be clear is the **swept line**, so a ring that goes round the reserved
    # rectangles is a legitimate fixture and a ring that runs through one is not.
    taken = set()
    for q in _plots_of(rnd):
        for x in range(min(q["x0"], q["x1"]) - 2, max(q["x0"], q["x1"]) + 3):
            for z in range(min(q["z0"], q["z1"]) - 2, max(q["z0"], q["z1"]) + 3):
                taken.add((x, z))
    net = rnd.network()
    for (lx, lz, _ly) in (net.surface() if net else []):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                taken.add((lx + dx, lz + dz))

    def clear(i, j):
        x, z = vol.x0 + i, vol.z0 + j
        return not any((x + a, z + b) in taken
                       for a in (0, n - 1) for b in range(n)) \
            and not any((x + a, z + b) in taken
                        for b in (0, n - 1) for a in range(n))
    # Cumulative sums over the dry mask, so "is every column of this ring dry" is four
    # array reads rather than 600. The loop is 4n-4 columns and the search is over every
    # origin in a 288-square: written the slow way it is fifty million lookups.
    import numpy as np
    dry = (~wet).astype(np.int64)
    # **Nearest the lane, after the relief** -- `check_parts`' own second key, and the
    # thing this fixture was missing. The line must stay *off* the lanes, which `clear`
    # already enforces, and it must still be somewhere a person could arrive at. A ring
    # nobody can walk to is the wrong fixture for the same reason a fixture set with no
    # water in it was the wrong fixture set.
    lane = [(lx, lz) for (lx, lz, _ly) in (net.surface() if net else [])]

    def to_lane(cx, cz) -> int:
        return min((max(abs(lx - cx), abs(lz - cz)) for (lx, lz) in lane), default=0)

    hh = h.astype(np.int64)
    best = None
    for i in range(1, w - n - 1):
        for j in range(1, d - n - 1):
            top, bot = slice(i, i + n), slice(j, j + n)
            edges = (dry[i, bot].sum() + dry[i + n - 1, bot].sum()
                     + dry[top, j].sum() + dry[top, j + n - 1].sum())
            if edges != 4 * n or not clear(i, j):
                continue
            hs = np.concatenate([hh[i, bot], hh[i + n - 1, bot],
                                 hh[top, j], hh[top, j + n - 1]])
            relief = int(hs.max() - hs.min())
            key = (abs(relief - PART_EDGE_RELIEF),
                   to_lane(vol.x0 + i + n // 2, vol.z0 + j + n // 2), i, j)
            if best is None or key < best[0]:
                best = (key, i, j, relief)
    if best is None:
        return None
    _k, i, j, relief = best
    x, z = int(vol.x0 + i), int(vol.z0 + j)
    # The gate goes on the side the lane is nearest, at the middle of it --
    # `wall_ring`'s own rule, and decided from the ground rather than chosen.
    sides = {"north": ((x + n // 2, z), "north"),
             "south": ((x + n // 2, z + n - 1), "south"),
             "west": ((x, z + n // 2), "west"),
             "east": ((x + n - 1, z + n // 2), "east")}
    where, facing = min(sides.values(), key=lambda s: (to_lane(*s[0]), s[0]))
    return {"round": round_name, "cache": cache, "kind": "edge", "part": "wall_city",
            "side": n,
            "path": [[x, z], [x + n - 1, z], [x + n - 1, z + n - 1],
                     [x, z + n - 1], [x, z]],
            "width": 1, "relief": relief, "columns": 4 * n - 4,
            "gate": {"at": list(where), "facing": facing},
            "lane_distance": to_lane(x + n // 2, z + n // 2),
            "rule": "edge_city_loop",
            "why": f"a closed square loop of {4 * n - 4} columns, {n} a side, on the "
                   f"unbuilt ground of {round_name}'s pre-build cache, every column of "
                   f"it dry, chosen for relief along the line nearest "
                   f"{PART_EDGE_RELIEF} and then by position"}


def _plots_of(rnd: Round) -> list:
    """A round's plot registry, or nothing.

        Ground that has never been built on has no registry, and that is not a missing file
        -- it is the whole reason a city's own unbuilt footprint is the only piece of ground
        on disk big enough to cut a 600-column ring out of.
        
    """
    p = rnd.rel("plots.json")
    return json.load(open(p)) if os.path.exists(p) else []


def _free_ground(round_name: str):
    """(volume, network, free) for a round: which columns belong to nobody.

        Free is inside the site, two clear of every plot rectangle, two clear of every lane
        cell and every reserved doorstep, and dry. A wall, a gate and a square have to be
        cut out of ground the settlement has not already spoken for, or the fixture is
        measuring a collision with a standing town rather than a part on ground.
        
    """
    from .. import observe
    rnd, _be = _fixture_round(round_name)
    vol = rnd.volume()
    net = rnd.network()
    s = rnd.site or json.load(open(rnd.rel("site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    plots = json.load(open(rnd.rel("plots.json")))
    h, wet = observe.ground_heights(vol)
    taken = set()
    for p in plots:
        for x in range(min(p["x0"], p["x1"]) - 2, max(p["x0"], p["x1"]) + 3):
            for z in range(min(p["z0"], p["z1"]) - 2, max(p["z0"], p["z1"]) + 3):
                taken.add((x, z))
    for (x, z, _y) in (net.surface() if net else []):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                taken.add((x + dx, z + dz))
    for t in (net.thresholds if net else []):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                taken.add((t.x + dx, t.z + dz))
    free = {}
    for x in range(X + 4, X + S - 4):
        for z in range(Z + 4, Z + S - 4):
            ix, iz = x - vol.x0, z - vol.z0
            if not (0 <= ix < vol.codes.shape[0] and 0 <= iz < vol.codes.shape[2]):
                continue
            if (x, z) in taken:
                continue
            if wet[ix, iz]:
                continue
            free[(x, z)] = int(h[ix, iz])
    return rnd, vol, net, free


def check_parts(round_name: str = PART_FIXTURE_ROUND) -> list:
    """One fixture of each new part kind, chosen by rule. A3.

        The same discipline as `check_fixtures`: a total order over measured numbers with
        the position as the last tie-break, computed by one command off a cache that is
        already on disk, reproducing itself. Nothing here is a judgement about a nice spot.

          edge   the L of two equal runs, 40 columns of centre line, on free ground, with
                 the least relief along it. A wall is the part that most has to cross what
                 the ground does, so the fixture is chosen for the ground it crosses being
                 *measured* rather than for being flat -- least relief first is what makes
                 it reproducible, and `sited["relief"]` is what says what it got.
          point  a **lane end**: a cell of the circulation network with exactly one lane
                 neighbour, facing away from it. That is where a gate goes -- the town
                 stops there -- and it is the one anchor a settlement defines for itself.
          area   the flattest free square of 12, which is a market square's worth.
        
    """
    rnd, vol, net, free = _free_ground(round_name)
    out = []

    lane = [(lx, lz) for (lx, lz, _y) in (net.surface() if net else [])]

    def to_lane(x, z) -> int:
        # Chebyshev to the nearest lane cell. A wall, a gate and a square are parts of a
        # *town*: the flattest ground on the site is a mesa top forty blocks above the
        # nearest lane, and a square nobody can walk to is the wrong fixture for the
        # same reason a fixture set with no water in it was the wrong fixture set.
        return min((max(abs(lx - x), abs(lz - z)) for (lx, lz) in lane), default=0)

    half = PART_EDGE_CELLS // 2
    best = None
    for (x, z) in sorted(free):
        arms = [[(x + i, z) for i in range(half)],
                [(x + half - 1, z + i) for i in range(1, half + 1)]]
        cells = [c for arm in arms for c in arm]
        if any(c not in free for c in cells):
            continue
        hs = [free[c] for c in cells]
        key = (max(hs) - min(hs), to_lane(x, z), x, z)
        if best is None or key < best[0]:
            best = (key, (x, z))
    if best is not None:
        (x, z) = best[1]
        out.append({"round": round_name, "kind": "edge", "part": "wall_L",
                    "path": [[x, z], [x + half - 1, z], [x + half - 1, z + half]],
                    "width": 1, "relief": best[0][0],
                    "rule": "edge_least_relief",
                    "why": f"{PART_EDGE_CELLS} columns in two runs meeting at one "
                           f"right angle, on free ground, least relief first and "
                           f"then nearest the lane"})

    # And a wall, which is a closed loop on relief with a gate in it. The rule is the
    # same shape as the L's -- a total order over measured numbers with the position as
    # the last tie-break -- and it is chosen for relief **near** a number rather than
    # least, because what a wall has to do that an L does not is step at its vertices as
    # the ground rises under it.
    n = PART_EDGE_LOOP
    best = None
    for (x, z) in sorted(free):
        ring = ([(x + i, z) for i in range(n)] + [(x + i, z + n - 1) for i in range(n)]
                + [(x, z + j) for j in range(1, n - 1)]
                + [(x + n - 1, z + j) for j in range(1, n - 1)])
        if any(c not in free for c in ring):
            continue
        hs = [free[c] for c in ring]
        key = (abs((max(hs) - min(hs)) - PART_EDGE_RELIEF), to_lane(x, z), x, z)
        if best is None or key < best[0]:
            best = (key, (x, z), max(hs) - min(hs), len(ring))
    if best is not None:
        (x, z) = best[1]
        # The gate goes on the side the town arrives from, which is the side nearest the
        # lane, at the middle of it and facing out of the loop. That is where a gate is,
        # and it is decided from the ground rather than chosen.
        sides = {
            "north": ((x + n // 2, z), "north"),
            "south": ((x + n // 2, z + n - 1), "south"),
            "west": ((x, z + n // 2), "west"),
            "east": ((x + n - 1, z + n // 2), "east"),
        }
        where, facing = min(sides.values(), key=lambda s: (to_lane(*s[0]), s[0]))
        out.append({"round": round_name, "kind": "edge", "part": "wall_ring",
                    "path": [[x, z], [x + n - 1, z], [x + n - 1, z + n - 1],
                             [x, z + n - 1], [x, z]],
                    "width": 1, "relief": best[2], "columns": best[3],
                    "gate": {"at": list(where), "facing": facing},
                    "rule": "edge_closed_loop",
                    "why": f"a closed square loop of {best[3]} columns on free ground, "
                           f"chosen for relief along the line nearest "
                           f"{PART_EDGE_RELIEF} and then nearest the lane; the gate is "
                           f"the middle of the side the lane is closest to"})

    # A5 cost a round by proving that one fixture per part kind is not a fixture set:
    # the standing `wall` was 4 of 4 on the 40-column L and 0 of 4 on the closed loop,
    # and the class it failed on -- a turret door nobody can reach, two blocks held up
    # by nothing at a vertex -- is one `NEEDS` cannot express. Both existing fixtures
    # are chosen for ground the wall runs *along*. Neither says what a wall does where
    # its ground **stops**, which is the case a walled place on real terrain meets at
    # every escarpment: a run whose last column has a drop off the end of it.
    from .. import observe as _observe
    _h, _wet = _observe.ground_heights(vol)

    def _ground_at(x, z):
        ix, iz = x - vol.x0, z - vol.z0
        if not (0 <= ix < vol.codes.shape[0] and 0 <= iz < vol.codes.shape[2]):
            return None
        return int(_h[ix, iz])

    n = PART_EDGE_CELLS
    best = None
    for (x, z) in sorted(free):
        for (dx, dz) in ((1, 0), (0, 1)):
            run = [(x + dx * i, z + dz * i) for i in range(n)]
            if any(c not in free for c in run):
                continue
            beyond = [(x + dx * (n + k), z + dz * (n + k)) for k in range(0, 3)]
            gs = [_ground_at(*c) for c in beyond]
            if any(g is None for g in gs):
                continue
            # The drop off the end: how far the ground falls in the three columns past
            # the last one the wall stands on. Greatest first, which is the opposite of
            # the L's rule and is the point -- the L is chosen for the least relief it
            # can find and this is chosen for the most sudden end it can find.
            drop = free[run[-1]] - min(gs)
            hs = [free[c] for c in run]
            key = (-drop, max(hs) - min(hs), to_lane(x, z), x, z)
            if best is None or key < best[0]:
                best = (key, run[0], run[-1], drop, max(hs) - min(hs))
    if best is not None and best[3] >= PART_EDGE_CLIFF:
        (ax, az), (bx, bz) = best[1], best[2]
        out.append({"round": round_name, "kind": "edge", "part": "wall_cliff",
                    "path": [[ax, az], [bx, bz]], "width": 1,
                    "relief": best[4], "drop": best[3],
                    "rule": "edge_ends_at_a_cliff",
                    "why": f"a straight run of {n} columns on free ground whose last "
                           f"column has a {best[3]}-block drop within three columns "
                           f"past the end of it; greatest drop first, then least relief "
                           f"along the line, then nearest the lane"})

    ends = []
    if net:
        cells = set(net.cells)
        for (cx, cz) in sorted(cells):
            near = [(cx + dx, cz + dz) for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (cx + dx, cz + dz) in cells]
            if len(near) == 1:
                ends.append(((cx, cz), near[0]))
    if ends:
        (cx, cz), (nx, nz) = ends[0]
        facing = ("east" if nx < cx else "west") if nz == cz else \
                 ("south" if nz < cz else "north")
        out.append({"round": round_name, "kind": "point", "part": "lane_end",
                    "at": [cx, cz], "facing": facing, "size": 5,
                    "rule": "point_lane_end",
                    "why": "a cell of the lane network with one lane neighbour, "
                           "facing away from it -- where the town stops"})

    n = PART_AREA_SIZE
    best = None
    for (x, z) in sorted(free):
        cells = [(x + i, z + j) for i in range(n) for j in range(n)]
        if any(c not in free for c in cells):
            continue
        hs = [free[c] for c in cells]
        key = (max(hs) - min(hs), to_lane(x, z), x, z)
        if best is None or key < best[0]:
            best = (key, (x, z))
    if best is not None:
        (x, z) = best[1]
        out.append({"round": round_name, "kind": "area", "part": "square",
                    "x0": x, "z0": z, "x1": x + n - 1, "z1": z + n - 1,
                    "relief": best[0][0], "rule": "area_flattest",
                    "why": f"the flattest free {n}x{n} of ground the settlement has "
                           f"not spoken for, nearest the lane"})
    return out


def _fixtures(rnd: Round) -> list:
    """From the config, so what a builder was checked against is on disk and registered
        before the builder was called; `check_fixtures()` is what wrote it and
        `test_siting` is what asserts the two still agree.
        
    """
    t = rnd.types or {}
    if t.get("check_fixtures"):
        return [dict(f) for f in t["check_fixtures"]]
    # An older config names plots and nothing else: they are its own round's.
    return [{"round": rnd.name, "plot": p}
            for p in (t.get("check_plots") or t.get("plots") or [])]


def _fixtures_for(rnd: Round, spec: dict) -> list:
    """A plot type is checked on the plots `check_fixtures` names; an edge, a point or an
    area is checked on the part of its own kind in `check_parts`, because there is no
    wall, gate or square in a settlement's plot registry. One list per kind, from the
    config, registered before the builder was called."""
    kind = spec.get("part", "plot")
    if kind == "plot":
        from ..slopefixture import descriptors
        path = spec.get("file") or os.path.join(_pipeline.ROOT, "types", spec["name"] + ".py")
        decl = _pipeline.load_type(path) if os.path.exists(path) else {"needs": spec.get("needs", {})}
        # A small pad on a bank, and the largest pad the type declares on one. Nothing
        # in this project had ever stood a large plot on a slope -- the terrain bank's
        # six sites have none and the needs sweep was a plane -- and a city found out
        # for it.
        return _fixtures(rnd) + descriptors(decl)
    t = rnd.types or {}
    return [dict(f) for f in (t.get("check_parts") or []) if f.get("kind") == kind]


def _seeds_for(rnd: Round, spec: dict) -> list:
    """The seeds this type's checker instantiates it at. See `_fixtures_for`.

        A plot type has six pieces of ground and two seeds; an edge, a point and an area
        have one piece each, so they get four seeds instead -- the same eight-ish instances,
        varied on the axis there is one of.
        
    """
    t = rnd.types or {}
    if spec.get("part", "plot") == "plot":
        return list(t.get("check_seeds") or (1, 2))
    return list(t.get("check_part_seeds") or (1, 2, 3, 4))


def stage_type_check(rnd: Round, d: str, key: str, spec: dict) -> str:
    """`check.py` for a *type*: the same file, over that type's own instances."""
    ROOT = _pipeline.ROOT
    os.makedirs(CHECK_JOBS, exist_ok=True)
    t = rnd.types
    json.dump({"kind": "type", "config": rnd.path, "key": key, "type": spec["name"],
               "fixtures": _fixtures_for(rnd, spec), "voice": spec.get("voice"),
               # B1: the two voices the checker stands the type in, decided here so the
               # record of what an author was checked against is on disk.
               "voices": _pipeline.check_voices(spec.get("voice")),
               # B1: an author's checker crosses every set of parameters the type
               # declares, unless the config pins one. A type is checked where the plan
               # may ask for it, not at the low end of its own ranges.
               "sweep": bool(spec.get("check_sweep", True)),
               "plots": list(t.get("check_plots") or t.get("plots") or []),
               "seeds": _seeds_for(rnd, spec),
               "params": dict(spec.get("check_params") or {})},
              open(os.path.join(CHECK_JOBS, f"{key}.json"), "w"), indent=1)
    p = os.path.join(d, "check.py")
    open(p, "w").write(CHECK_PY.format(root=ROOT, key=key))
    return p


#: **The checker reads a silhouette.** Demo-polish, phase 1b. Nothing did. This is the
#: one read that would have caught it: of the columns of a plot instance that rise a
#: storey above its floor, the share whose topmost block is the voice's roof family. A
#: building's top is its roof. the lowest clean reading is 0.85, and what is not roof on
#: a house is its chimney and its dormer cheeks. The temple read 0.089-0.345. The bar is
#: 0.6: under every clean type by a quarter and over the box by the same.
ROOF_SHARE_MIN = 0.6

#: A column counts as the building, and not the court, the yard or a lantern on the
#: platform, when its topmost block stands this far above the part's floor: one storey
#: and a course.
ROOF_SHARE_RISE = 5


def roof_share(pending: dict, rect, floor_y: int, roof_family: str) -> dict:
    """The share of a plot instance's built columns whose topmost block is the voice's
        roof family. See `ROOF_SHARE_MIN`.

        `rect` is the plot (x0, z0, x1, z1); `roof_family` is `part["voice"]["roof"]`,
        resolved through `prims.shape` to every shape the family is laid in, so a ridge cap
        slab, a hip stair and a full course all count. Returns `{"columns", "roofed",
        "share"}`; `share` is None where nothing rises a storey, which is a plot with no
        building on it and not a plot with no roof.
        
    """
    from ..prims import shape
    shapes = set()
    for kind in ("full", "stairs", "slab", "wall"):
        try:
            shapes.add(shape(roof_family, kind))
        except ValueError:
            continue
    x0, z0, x1, z1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    fy = int(floor_y)
    top: dict = {}
    for (x, y, z), blk in pending.items():
        if not (x0 <= x <= x1 and z0 <= z <= z1) or y <= fy:
            continue
        name = blk.split("[")[0].split(":")[-1]
        if name == "air":
            continue
        got = top.get((x, z))
        if got is None or y > got[0]:
            top[(x, z)] = (y, name)
    built = [t for t in top.values() if t[0] >= fy + ROOF_SHARE_RISE]
    roofed = sum(1 for _y, name in built if name in shapes)
    return {"columns": len(built), "roofed": roofed,
            "share": (round(roofed / len(built), 3) if built else None)}


def check_type(rnd: Round, be, prog: str, labels: list, seeds: list,
               params: dict | None = None, fixtures: list | None = None,
               voice: str | None = None, voices: list | None = None,
               sweep: bool = False) -> dict:
    """Instantiate one type over every fixture, seed, **two voices and every set of
        parameters it declares**, and say what the checks say.

        One execution per seed and round rather than one per instance: the plots of a round
        are disjoint, so all of a seed's instances there are composed into one program and
        linted against one Context. That is what keeps a check run to minutes; the
        per-instance answers are identical because `standard_report` scopes the build
        family to a plot and the place family is a fact about the town either way.

        **And every set of parameters, since B1.** `sweep` crosses
        `param_combinations(PARAMS)` into the work as a fourth axis. Without it a checker
        stands a type at *one* point of its own declared space -- `check_params` fills a
        missing parameter from the low end of an int range and the first of a choice -- so
        a `shop_house` declaring `storeys` 2 to 3 was only ever checked at two, a `hall`
        declaring three uses was only ever checked as a moot, and a `wall` declaring
        `height` 3 to 20 and `width` 1 to 5 was only ever checked three high and one
        thick. The city then asked for three storeys, a market, and twenty by three. This
        is the project's standing law one dimension further: a type must carry to ground
        its author never saw, to a voice its author never saw, and to the **parameters the
        planner actually gives it**.

        Returns the findings text and a row per instance, worst first.
    """
    import traceback
    t0 = time.perf_counter()
    try:
        decl = _pipeline.load_type(prog)
    except Exception as e:                       # noqa: BLE001 -- this *is* the report
        return {"text": ("# Your file is not a type yet\n\n"
                         + f"{type(e).__name__}: {e}\n\n"
                         "A type declares `FORM` and `PARAMS` at the top level and "
                         "defines `build(b, part, seed, **params)`, and nothing at the "
                         "top level may place a block.\n"),
                "seconds": round(time.perf_counter() - t0, 1), "rows": [],
                "instances": 0, "errors": 1, "entry_lines": 0, "crashed": True,
                "blocks": 0, "writes": 0}
    if not fixtures:
        fixtures = [{"round": rnd.name, "plot": l} for l in labels]
    where = os.path.basename(prog)
    if sweep and not params:
        sets = [_pipeline.check_params(decl["params"], c, where=where)
                for c in _pipeline.param_combinations(decl["params"])]
    else:
        sets = [_pipeline.check_params(decl["params"], params, where=where)]
    kw = sets[0]
    voices = list(voices or _pipeline.check_voices(voice))
    # A6: one work item per (fixture round, seed, voice) -- which is what a composed
    # program already is -- run across processes and read back in the order a serial
    # loop produces them. The rows are the answer and the rows do not move; what moves
    # is where the twenty-four of them are computed. `ETHOSLM_WORKERS=1` is the serial
    # arm. B1 adds the parameter set as a fourth axis; with one set this is the same
    # work.
    work = [(name, seed, v, i) for i in range(len(sets)) for v in voices
            for name in sorted({f["round"] for f in fixtures}) for seed in seeds]
    got = parallel.par_map(
        functools.partial(_check_type_job, rnd, be, prog, decl, fixtures, sets), work)
    rows = [r for batch in got for r in batch["rows"]]
    crashed = any(batch["crashed"] for batch in got)
    # Worst first: what is wrong with it, then how little of it you can walk in. A type
    # is its worst instance, so its worst instance is the first thing its author reads.
    rows.sort(key=lambda r: (not r["crashed"],
                             -(r["errors"] + r["entry_lines"] + int(bool(r.get("roofless")))),
                             r["walk_pct"] if r["walk_pct"] is not None else -1,
                             voices.index(r["voice"])))
    by_voice = voice_summary(rows, voices)
    by_params = param_summary(rows, sets)
    return {"text": _type_findings(rows, decl, kw, by_voice, by_params),
            "seconds": round(time.perf_counter() - t0, 1), "rows": rows,
            "instances": len(rows),
            "errors": sum(r["errors"] for r in rows),
            "entry_lines": sum(r["entry_lines"] for r in rows),
            "roofless": sum(1 for r in rows if r.get("roofless")),
            "blocks": max((r["pending"] for r in rows), default=0),
            "writes": max((r["writes"] for r in rows), default=0),
            "crashed": crashed, "voices": by_voice,
            "coupled": coupled_to_silhouette(by_voice),
            "swept": sets if sweep else None,
            "params": by_params}


def param_summary(rows: list, sets: list) -> dict:
    """What each set of parameters read, over the same fixtures, seeds and voices. B1.

        Keyed by the parameters as they are written down, because that is what an author
        has to change and what a plan has to avoid asking for.
        
    """
    out = {}
    for kw in sets:
        key = ", ".join(f"{k}={kw[k]}" for k in sorted(kw)) or "(no parameters)"
        mine = [r for r in rows if r.get("params") == kw]
        out[key] = {"instances": len(mine),
                    "clean": sum(1 for r in mine if not r["errors"]
                                 and not r["entry_lines"] and not r["crashed"]
                                 and not r.get("roofless")),
                    "errors": sum(r["errors"] for r in mine),
                    "entry_lines": sum(r["entry_lines"] for r in mine),
                    "crashed": sum(1 for r in mine if r["crashed"]),
                    "roofless": sum(1 for r in mine if r.get("roofless")),
                    "codes": sorted({c for r in mine for c in r.get("codes", ())})}
    return out


def voice_summary(rows: list, voices: list) -> dict:
    """What each voice read, over the same fixtures and seeds. B1."""
    out = {}
    for v in voices:
        mine = [r for r in rows if r["voice"] == v]
        out[v] = {"instances": len(mine),
                  "clean": sum(1 for r in mine
                               if not r["errors"] and not r["entry_lines"]
                               and not r["crashed"] and not r.get("roofless")),
                  "walkable": sum(1 for r in mine if r["walk_pct"] is not None
                                  and r["walk_pct"] >= 90),
                  "errors": sum(r["errors"] for r in mine),
                  "entry_lines": sum(r["entry_lines"] for r in mine),
                  "crashed": sum(1 for r in mine if r["crashed"]),
                  "roofless": sum(1 for r in mine if r.get("roofless"))}
    return out


def coupled_to_silhouette(by_voice: dict) -> dict | None:
    """The B1 verdict: None when every voice read the same, else which differ and how.

        Read on the counts and not on the cells, because a roof that follows the voice is
        *supposed* to put its stairs in different places; what it is not supposed to do is
        put them where the linter objects in one voice and not the other.
        
    """
    if len(by_voice) < 2:
        return None
    keys = ("clean", "errors", "entry_lines", "crashed", "roofless")
    vals = {k: {v: d.get(k, 0) for v, d in by_voice.items()} for k in keys}
    moved = {k: vals[k] for k in keys if len(set(vals[k].values())) > 1}
    return moved or None


def _check_type_job(rnd, be, prog, decl, fixtures, sets, work) -> dict:
    """One fixture round at one seed in one voice at one set of parameters: compose,
        run, lint, and read every instance.

        Everything a worker needs is an argument, and everything it hands back is data.
        `_fixture_round` is cached per process, so a forked worker inherits whatever the
        parent had already loaded and reads the rest once.
        
    """
    import traceback
    name, seed, voice, pi = work
    kw = sets[pi]
    mat = _pipeline.voice_palette(voice)
    vroof = _pipeline.voice_roof(voice)
    rows, crashed = [], False
    frnd, fbe = (rnd, be) if name == rnd.name else _fixture_round(name)
    plots = {p["label"]: p for p in _plots_of(frnd)}
    want = [q for q in (_pipeline.fixture_part(f, plots) for f in fixtures
                        if f["round"] == name) if q is not None]
    # A3: an edge, a point and an area are not in the round's plot registry, so the
    # rectangle each is answerable for is added to the Context's plots -- otherwise
    # `standard_report` falls back to the whole town and a wall is judged on it.
    # `part_registry_row` writes one rectangle per segment and `lint.plot_rects` reads
    # them; this composed the row by hand off `part_rect` and got the box. It cost
    # nothing while an edge fixture was a 40-column L, and on A3's 131-a-side city ring
    # the box is **seventeen thousand columns of hillside**: the wall's author was
    # handed forty sealed cave pockets and twenty-one floating jungle logs at identical
    # coordinates in all six of its runs, most of them tens of blocks off the line, one
    # of them a 952-cell room outside the ring entirely. A wall's bounding box is the
    # town it encloses, and this is the second time that sentence has cost a round.
    extra = [_pipeline.part_registry_row({**p, "name": p["label"]})
             for p in want if p.get("kind") in ("edge", "point", "area")]
    instances = [(p, seed, kw) for p in want]
    src = _pipeline.instantiated_source(decl["src"], instances, mat=mat,
                                        roof=vroof)
    b, err = None, None
    try:
        b = _run_src(frnd, fbe, prog, src,
                     allow_collide=bool(rnd.flags.get("allow_collide")))
    except Exception:                    # noqa: BLE001 -- this *is* the report
        err = scrub_traceback(traceback.format_exc())[-2000:]
        crashed = True
    pending = b._pending if b is not None else {}
    # **and the level the library sited each part at**, so the checker attributes a room
    # the way the round does. Without it the type checker is the one reader of
    # `room["plot"]` that A1 does not reach, and it charged `palace` a 266-cell cavern
    # at y=1 under a 48x48 compound. `site()` records `floor_y` on every part it
    # prepares; this is that record, put where `room_owner` reads it.
    sited = {p.get("label"): p.get("floor_y") for p in (getattr(b, "parts", None) or [])
             if p.get("floor_y") is not None}
    rows_in = list(plots.values()) + extra
    if sited:
        rows_in = [dict(q, y0=int(sited[q["label"]])) if q.get("label") in sited else q
                   for q in rows_in]
    ctx = _pipeline.build_context(frnd, fbe, None, pending=pending,
                        plots=rows_in if (extra or sited) else None,
                        fittings=b.fitting_cells if b is not None else None)
    for p in want:
        mine = [q for q in ctx.plots if q["label"] == p["label"]]
        rep = _pipeline.standard_report(ctx, mine or ctx.plots)
        lines = _pipeline.entry_lines(_pipeline.diagnose_entry(ctx, p["label"]))
        errs = len([f for f in rep.findings if f.code.startswith("E")])
        # The same question the round asks, off the same measure. Until A4 this read
        # `interior_walk`, which counted every room on the plot; A4 gives that call the
        # `ENCLOSED` filter its two siblings already applied, and under it no room in
        # this project is an interior -- every one of them joins the sheltered band
        # outside its own door and reads 0.14 to 0.7 against a threshold of 0.85. A
        # checker that came back "0 rooms, no number" on a finished building is worse
        # than no checker, so the walkability a type's author reads is now the one
        # `measure_program` reads and the one its round is scored on: the room's floor,
        # walked from outdoors. `shut` is what `interior_walk` still reports, which is
        # rooms cut off from that band -- the E011 candidates and nothing else.
        walkable = _type_walkable(ctx)
        cells = walk = n_rooms = 0
        for r in ctx.rooms:
            if r.get("plot") != p["label"]:
                continue
            st = set(map(tuple, r["floor"]))
            n_rooms += 1
            cells += len(st)
            walk += len(st & walkable)
        shut = [r for r in ctx.interior_walk() if r.get("plot") == p["label"]]
        # A plot the lane network does not reach has no "walkable from outdoors" to
        # report: the flood is seeded from the lane and there is no lane. `None` and a
        # flag, rather than 0.0 -- a number that can only be zero is not a measurement,
        # and 0.0% beside a palace that is entirely walkable is worse than no number.
        off = bool(_pipeline.diagnose_entry(ctx, p["label"]).get("off_network"))
        # 1b: a plot instance carries its roof on top, or it is not clean. Read off the
        # pending set against the plot's own rectangle and the floor `site()` gave it.
        roof = {"columns": 0, "roofed": 0, "share": None}
        if p.get("kind", "plot") == "plot" and b is not None and mat and mat.get("roof"):
            fy = sited.get(p["label"], p.get("y0"))
            if fy is not None:
                roof = roof_share(pending, (p["x0"], p["z0"], p["x1"], p["z1"]),
                                  int(fy), mat["roof"])
        roofless = roof["share"] is not None and roof["share"] < ROOF_SHARE_MIN
        rows.append({
            "round": name, "plot": p["label"],
            "kind": p.get("kind", "plot"), "seed": seed, "voice": voice,
            # B1: which point of the type's own declared space this instance is.
            "params": dict(kw),
            "codes": sorted(f.code for f in rep.findings if f.code.startswith("E")),
            "errors": errs,
            "entry_lines": len(lines), "lines": lines, "off_network": off,
            "walk_pct": (None if off else
                         round(100 * walk / cells, 1) if cells else None),
            "rooms": n_rooms, "shut": len(shut),
            "roof_share": roof["share"], "roof_columns": roof["columns"],
            "roofless": roofless,
            "crashed": bool(err), "error": err,
            "report": rep, "pending": len(pending), "writes":
                (b.writes if b is not None else 0),
        })
    return {"rows": rows, "crashed": crashed}


def _type_walkable(ctx) -> set:
    """Walk-only reachability from the checker context's real outside source.

        Context chooses the circulation lane where one exists and the volume perimeter
        otherwise. Re-running a perimeter-only flood here made a valid building on a cut
        fixture report zero walkability even though its lane reached every floor cell.
        
    """
    return set(ctx.from_outdoors)


@functools.lru_cache(maxsize=8)
def _fixture_round(name: str):
    """(Round, OfflineBackend) for a fixture's own round, cached."""
    if name.startswith(("slope_", "slopebig_", "slopepair_")):
        from ..slopefixture import make
        return make(name, _pipeline.ROOT)
    rnd = _pipeline.Round.load(os.path.join(_pipeline.ROOT, "rounds", f"{name}.json"))
    return rnd, _pipeline.OfflineBackend(rnd)


def _run_src(rnd: Round, be, prog: str, src: str, allow_collide: bool | None = None):
    """Execute composed source under the program's own name, on the round's ground.

        `allow_collide` is the **checking** round's, never the fixture's. It used to be
        read off `rnd.flags`, which for a fixture borrowed from another round is that
        round's config -- and one early round ran with collisions allowed, so every type
        checked on its two plots stood with `fitting()` refusing nothing: a barrel on the
        foot of a flight, a bookshelf in a dais's step, and a sealed storey the city would
        never have built. A type is checked under the city's rules. Demo-polish, 1c.
        
    """
    from .. import offline, stages
    if allow_collide is None:
        allow_collide = bool(rnd.flags.get("allow_collide"))
    return offline.run_program(prog, be.volume, network=rnd.network(),
                               plots=stages._Registry(rnd.state),
                               allow_collide=bool(allow_collide), src=src)


def _type_findings(rows: list, decl: dict, kw: dict, by_voice: dict | None = None,
                   by_params: dict | None = None) -> str:
    """`findings.md` for a type: a table, the two voices against each other, then a
    section per instance, worst first."""
    head = ["# What your type builds, on every piece of ground, both seeds, in two "
            "voices", "",
            f"`FORM` is {decl['form']!r} and this run used "
            + (", ".join(f"{k}={v!r}" for k, v in sorted(kw.items()))
               if kw else "no parameters")
            + ".", "",
            "| ground | seed | voice | parameters | own errors | on foot | walkable "
            "| roof |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pr = ", ".join(f"{k}={v}" for k, v in sorted((r.get("params") or {}).items()))
        rs = r.get("roof_share")
        head.append(f"| {r['plot']} | {r['seed']} | {r.get('voice') or '-'} | "
                    f"{pr or '-'} | "
                    f"{r['errors']} | {r['entry_lines']} | "
                    + (f"{r['walk_pct']}%" if r["walk_pct"] is not None else "-")
                    + " | " + ("-" if rs is None else f"{rs:.2f}"
                               + (" **no roof**" if r.get("roofless") else ""))
                    + " |")
    if any(r.get("roofless") for r in rows):
        head += ["", f"**Some instances have no roof on top.** `roof` is the share of "
                 f"the building's columns whose topmost block is the voice's roof family, "
                 f"and it is under {ROOF_SHARE_MIN} on the rows marked. A building's top "
                 f"is its roof: something the type placed after `building()` -- a fill, a "
                 f"packed loft, a parapet in the wall material -- is standing over the "
                 f"roof the shell laid. Take the ridge from what `building()` returns and "
                 f"build nothing above it."]
    if by_params and len(by_params) > 1:
        head += ["", "## The same type at every set of parameters it declares", "",
                 "B1: a checker that stands a type at one point of its own declared "
                 "space is a checker that has not been run. Every instance above was "
                 "stood at each of these, and **a type is its worst row here too**: "
                 "the plan will ask for any of them.", "",
                 "| parameters | instances | clean | own errors | on foot | codes |",
                 "|---|---|---|---|---|---|"]
        for k, d in by_params.items():
            head.append(f"| {k} | {d['instances']} | {d['clean']} | {d['errors']} "
                        f"| {d['entry_lines']} | {', '.join(d['codes']) or '-'} |")
    if by_voice:
        head += ["", "## The same type in two voices", "",
                 "Every instance above was stood twice: once in the voice you were "
                 "given and once in a voice with a different palette **and a "
                 "different roof silhouette** -- the profile, the ends, the eave and "
                 "the tiers that `roof()` and `building()` are handed from "
                 "`part['roof']`. A type is a form, so it has to be clean in both.", "",
                 "| voice | instances | clean | walkable | own errors | on foot |",
                 "|---|---|---|---|---|---|"]
        for v, d in by_voice.items():
            head.append(f"| {v} | {d['instances']} | {d['clean']} | {d['walkable']} "
                        f"| {d['errors']} | {d['entry_lines']} |")
        moved = coupled_to_silhouette(by_voice)
        if moved:
            head += ["", "**The two voices do not read the same, so your type's "
                     "geometry is following the silhouette and breaking under one of "
                     "them.** "
                     + "; ".join(f"{k}: " + ", ".join(f"{v} {n}" for v, n in d.items())
                                 for k, d in moved.items())
                     + ". The roof is not yours to shape: `roof()` is handed the "
                     "voice's profile, ends, eave and tiers whatever you pass, so build "
                     "the massing so that it stands under *any* of them -- take the "
                     "ridge height from what `roof()` returns rather than from a pitch "
                     "you assumed, keep walls, floors and flights clear of where a "
                     "steeper or shallower slope lands, and do not lean on an overhang, "
                     "a tier or an eave being any particular shape. Then run me again: "
                     "the two rows have to match."]
        else:
            head += ["", "The two voices read the same. Your type is a form."]
    head += ["", "Worst first below. A type is its worst instance.", ""]
    body = []
    for r in rows:
        body.append(f"\n---\n\n# {r['plot']}, seed {r['seed']}, "
                    f"voice {r.get('voice') or '-'}\n")
        if r["error"]:
            body.append("**Your program did not run to the end on this instance.** "
                        "It stopped here:\n\n```\n" + r["error"].strip() + "\n```\n")
        body.append(_pipeline.revision_findings(r["report"], r["lines"], r["pending"],
                                      f"{r['plot']} seed {r['seed']} "
                                      f"voice {r.get('voice') or '-'}"))
    return "\n".join(head) + "\n" + "\n".join(body)


def placed_versus_attempted(b, error: str | None) -> str:
    """The one line the findings brief has never carried."""
    if error:
        return ("## Placed versus attempted\n\n"
                "**Your program did not run to the end.** It stopped here:\n\n"
                "```\n" + error.strip() + "\n```\n\n"
                f"{len(b._pending) if b is not None else 0} blocks had been placed when "
                f"it stopped; nothing after that line ran, so everything the checks "
                f"above say is about a part of the building. Fix the line and run this "
                f"check again.\n")
    over = b.writes - len(b._pending)
    return ("## Placed versus attempted\n\n"
            f"Your program ran to the end. It wrote {b.writes} positions and "
            f"{len(b._pending)} blocks stand"
            + (f"; {over} were written over by a later call, which is normal where you "
               f"carve an opening back out of a wall and a mistake where you did not "
               f"mean to build the same thing twice" if over else "")
            + ".\n")


def check_program(rnd: Round, be, prog: str, labels: list, wave: str,
                  rects: list | None = None) -> dict:
    """The same three things the findings brief has always been made of, and nothing the
        builder is not already told: `standard_report` (the build family on its own plots,
        the place family over the town), the on-foot lines the linter is silent on, and now
        placed versus attempted.
        
    """
    import traceback
    t0 = time.perf_counter()
    b, err = None, None
    try:
        b = be.execute(prog, be.volume)
    except Exception:                     # noqa: BLE001 -- this *is* the report
        err = scrub_traceback(traceback.format_exc())[-2000:]
    pending = b._pending if b is not None else {}
    ctx = _pipeline.build_context(rnd, be, prog, pending=pending, plots=rects or None,
                        fittings=b.fitting_cells if b is not None else None)
    mine = [p for p in ctx.plots if p["label"] in set(labels)] or ctx.plots
    rep = _pipeline.standard_report(ctx, mine)
    lines: list = []
    for lab in sorted({p["label"] for p in mine}):
        lines += _pipeline.entry_lines(_pipeline.diagnose_entry(ctx, lab))
    text = _pipeline.revision_findings(rep, lines, len(pending), wave,
                             extra=placed_versus_attempted(b, err))
    return {"text": text, "seconds": round(time.perf_counter() - t0, 1),
            "blocks": len(pending), "writes": (b.writes if b is not None else 0),
            "errors": len([f for f in rep.findings if f.code.startswith("E")]),
            "entry_lines": len(lines), "crashed": bool(err)}


def run_check(key: str, d: str) -> int:
    """`check.py`'s whole body. Rewrites `findings.md` and records that it ran.

        The run goes on `out/measurements.jsonl` as a `check_run` row keyed by the scratch
        directory and the program's own bytes, which is the only record of how many times a
        builder looked -- and, per the spec, the thing `done` is refused without.
        
    """
    import hashlib
    job = json.load(open(os.path.join(CHECK_JOBS, f"{key}.json")))
    prog = os.path.join(d, "program.py")
    if not os.path.exists(prog):
        print("no program.py in this directory yet -- write it first, then run me")
        return 2
    src = open(prog).read()
    from .. import lint
    is_type = job.get("kind") == "type"
    pre = lint.preflight(src, forbid=(_pipeline.TYPE_FORBIDDEN if is_type else None),
                         palette=is_type)
    if not pre.ok:
        open(os.path.join(d, "findings.md"), "w").write(
            "# Your program was rejected before it ran\n\n"
            + "\n".join(f"  - {f.code}: {f.message}" for f in pre.findings)
            + "\n\n" + "\n".join(lint.FIXES[f.code] for f in pre.findings
                                 if f.code in lint.FIXES) + "\n")
        print(f"rejected before it ran: {[f.code for f in pre.findings]}"
              f"  -- see findings.md")
        return 1
    rnd = _pipeline.Round.load(job["config"])
    if job.get("kind") == "type":
        res = check_type(rnd, _pipeline.OfflineBackend(rnd), prog, job["plots"], job["seeds"],
                         params=job.get("params"), fixtures=job.get("fixtures"),
                         voice=job.get("voice"), voices=job.get("voices"),
                         sweep=bool(job.get("sweep", True)))
        wave = job.get("type")
    else:
        res = check_program(rnd, _pipeline.OfflineBackend(rnd), prog, job["plots"], job["wave"],
                            rects=job.get("rects"))
        wave = job["wave"]
    open(os.path.join(d, "findings.md"), "w").write(res["text"])
    record("check_run", key=key, settlement=rnd.name, wave=wave,
           sha256=hashlib.sha256(src.encode()).hexdigest(), blocks=res["blocks"],
           writes=res["writes"], errors=res["errors"],
           entry_lines=res["entry_lines"], crashed=res["crashed"],
           seconds=res["seconds"], instances=res.get("instances"))
    if res.get("instances"):
        clean = sum(1 for r in res["rows"]
                    if not r["errors"] and not r["entry_lines"])
        print(f"checked {res['instances']} instances in {res['seconds']}s: "
              f"{clean} of {res['instances']} clean, {res['errors']} own errors and "
              f"{res['entry_lines']} on-foot findings over all of them"
              + (" -- YOUR PROGRAM CRASHED" if res["crashed"] else "")
              + "".join(f"\n  in {v}: {d['clean']} of {d['instances']} clean, "
                        f"{d['errors']} own errors, {d['entry_lines']} on-foot"
                        for v, d in (res.get("voices") or {}).items())
              + ("\n  THE TWO VOICES DIFFER: your type is coupled to a silhouette -- "
                 "see findings.md" if res.get("coupled") else "")
              + "\nfindings.md has been rewritten, worst instance first; read it.")
        return 0
    print(f"checked in {res['seconds']}s: {res['blocks']} blocks, {res['errors']} "
          f"errors, {res['entry_lines']} on-foot findings"
          + (" -- YOUR PROGRAM CRASHED" if res["crashed"] else "")
          + "\nfindings.md has been rewritten; read it.")
    return 0


def checked_runs(key: str) -> list:
    """Every `check_run` this builder made, oldest first."""
    out = []
    if not os.path.exists(measure_mod.LOG):
        return out
    for line in open(measure_mod.LOG):
        if key not in line:
            continue
        row = json.loads(line)
        if row.get("kind") == "check_run" and row.get("key") == key:
            out.append(row)
    return out
