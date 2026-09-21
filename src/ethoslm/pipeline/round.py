"""Round configuration, execution backends and stage orchestration."""
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
from .. import offline, settlement, verdicts
from ..measure import record


ROOT = settlement.ROOT


def settlement_site(rnd) -> dict | None:
    """Never 's."""
    p = rnd.rel("site.json")
    return json.load(open(p)) if os.path.exists(p) else None


@dataclass
class Round:
    """One round, as a file. Everything a fresh agent would otherwise have to be told."""
    name: str
    intent: str = ""
    #: **one sentence**. A round that carries one chooses its own kind, its own scale,
    #: its own site and its own voice, and `site` and `voice` below are empty on
    #: purpose.
    sentence: str = ""
    voice: str = ""
    site: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    base_volume: str = "world.npz"
    waves: list = field(default_factory=list)
    shots: dict = field(default_factory=dict)
    cards: list = field(default_factory=list)
    judgements: list = field(default_factory=list)
    preregistered: dict = field(default_factory=dict)
    #: Selection, as config: which wave's brief, how many independent candidates, the
    #: plots they build on, and how they are photographed, measured and read. A round
    #: that selects is a round with this field, not a round with a script.
    candidates: dict = field(default_factory=dict)
    #: Arms, as config: the place-look-adjust round. Which wave's brief, the fixed cycle
    #: sequence every arm walks, how many independent candidates per arm, and which arms
    #: see a preview of what they have built so far. A round that tests a *loop* is a
    #: round with this field, not a round with a script -- and the cycle text living
    #: here rather than in code is what makes "the prompts were fixed before the run" a
    #: thing on disk instead of a claim.
    arms: dict = field(default_factory=dict)
    #: Revision, as config: which finished first drafts go back through the standard
    #: findings loop, how many times, and what the composed brief is made of. A round
    #: that measures what the *loop* does to work that already exists is a round with
    #: this field. Its subjects are named by the round configs they were built under
    #: rather than by a restatement of their plots, briefs and base volumes -- a second
    #: copy of those is a second chance to get them wrong.
    revise: dict = field(default_factory=dict)
    #: Types, as config: a list of `{name, file, voice, kind, instances}`. A type is a
    #: *file* -- `types/<name>.py`, committed, with `FORM`, `PARAMS` and `build(b, plot,
    #: seed, **params)` -- and an instance is `{plot, seed, params}`.
    types: dict = field(default_factory=dict)
    #: Set by fixtures, and by a check job -- `check.py` runs in a separate process and
    #: has to reach the same directory the stage that staged it was writing to.
    state_dir: str = ""
    path: str = ""

    @classmethod
    def load(cls, path: str) -> "Round":
        d = json.load(open(path))
        known = {f for f in cls.__dataclass_fields__ if f != "path"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"{path}: unknown round fields {sorted(unknown)} -- "
                             f"known fields are {sorted(known)}")
        return cls(path=os.path.abspath(path), **d)

    @property
    def state(self) -> str:
        """A round is a config and one command, and that has to hold for a round whose
                ground, site briefing and plan the repository *ships* -- the offline build a
                reader with no server, no key and none of this project's caches can run. The
                fixture under `fixtures/<name>/` is read-only input; the run writes to
                `out/<name>/` like every other round, so running it twice is running it twice
                and not editing a fixture.
                
        """
        if self.state_dir:
            return self.state_dir
        out = os.path.join(ROOT, "out", self.name)
        seed = os.path.join(offline.FIXTURES, self.name)
        if os.path.isdir(seed):
            # File by file, and never over one that is already there: another command
            # may have made the directory before this one looked, and a rule that seeded
            # only an absent directory then left the round with none of its own state.
            # What a run writes always wins; nothing is written back.
            import shutil
            took = []
            for f in sorted(os.listdir(seed)):
                src, dst = os.path.join(seed, f), os.path.join(out, f)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.makedirs(out, exist_ok=True)
                    shutil.copyfile(src, dst)
                    took.append(f)
            # **An import is a decision and it is recorded here.** `deps.legacy` used to
            # read a missing dependency stamp as "this predates the contract, reuse it",
            # which is the same answer it gives a directory a stage crashed half-way
            # through. This is the one place documents arrive from outside the run, so
            # this is where "they were adopted on purpose" is written down.
            if took and not os.path.exists(os.path.join(out, "imported.json")):
                from .. import deps as _deps
                _deps.import_legacy(
                    out, f"seeded from the shipped fixture `fixtures/{self.name}/`; "
                         f"these documents were not made by this run and are adopted "
                         f"as given", took)
        return out

    def rel(self, *parts) -> str:
        return os.path.join(self.state, *parts)

    def volume(self):
        return offline.load_volume(self.rel(self.base_volume))

    def network(self):
        from ..circulate import Network
        p = self.rel("network.json")
        return Network.load(p) if os.path.exists(p) else None

    # --- the plan, which is a tree ----------------------------------------
    def plan(self) -> dict:
        p = self.rel("plan.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def parts(self) -> list:
        """Every leaf of the plan, in order. See `plan_parts`."""
        return _pipeline.plan_parts(self.plan())

    def plots(self) -> list:
        """The plot leaves, in plot-registry shape. See `plan_plots`.

                A4: the plan is a tree, and this is the one call that flattens it, so every
                stage that used to read `plan["structures"]` keeps working whichever shape the
                plan on disk is.
                
        """
        return _pipeline.plan_plots(self.plan())

    # ------------------
    def place_spec(self) -> dict | None:
        """`place.json` as the schema validator reads it, or None. A1."""
        from .. import spec as spec_mod
        c = self.rel("place.checked.json")
        p = self.rel("place.json")
        if os.path.exists(c):
            return spec_mod.read_spec(json.load(open(c)), self.sentence or None)
        if not os.path.exists(p):
            return None
        return spec_mod.read_spec(json.load(open(p)), self.sentence or None)

    def site_search(self) -> dict | None:
        """What `scripts/find_site.py` decided and why. A3."""
        p = self.rel("site_search.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def chosen_site(self) -> dict | None:
        """`{"origin": [x, z], "size": n}` from the search, or None. A3.

                The one place a round with no site in its config gets one, and it is
                deliberately the *search's* own answer: `site.json` is written from this by
                `prepare_settlement.py`, so reading it back would be reading our arithmetic
                twice and would hide a disagreement between them.
                
        """
        got = (self.site_search() or {}).get("chosen")
        if not got:
            return None
        return {"origin": [int(got["origin"][0]), int(got["origin"][1])],
                "size": int(got["size"])}

    def voice_name(self) -> str:
        """The config's, or the one the place chose."""
        return (self.voice or (self.plan() or {}).get("voice")
                or ((self.place_spec() or {}) or {}).get("voice") or "")


class OfflineBackend:
    """The world as a cached volume. Nothing is written anywhere.

        This is the backend every replay and every dry run uses, and the reason a round can
        be reproduced on a machine with no Minecraft server running.
    """
    live = False

    def __init__(self, rnd: Round, dry_run: bool = False):
        self.round = rnd
        self.dry_run = bool(dry_run or rnd.flags.get("dry_run"))
        self._vol = None

    @property
    def volume(self):
        if self._vol is None:
            self._vol = self.round.volume()
        return self._vol

    def execute(self, program: str, vol=None):
        from .. import stages
        return offline.run_program(program, self.volume if vol is None else vol,
                                   network=self.round.network(),
                                   plots=stages._Registry(self.round.state),
                                   allow_collide=bool(
                                       self.round.flags.get("allow_collide")))

    def commit(self, builder):
        if not self.dry_run:
            return {"placed": 0, "failed": 0, "note": "offline: nothing written"}
        # **A dry run's commit is real, and it is real into the volume.** A place is
        # built in waves and each wave is linted on the world the ones before it left:
        # the walls, then the squares, then a quarter at a time. Offline, `commit` threw
        # every wave away, so wave two would have been written against bare ground and
        # the whole-place lint at the end would have had nothing in it. Here the pending
        # set goes into the cached volume and the next wave sees it, which is what the
        # live backend's `save-all flush` does and the only thing about it that the
        # measurements depend on.
        pending = dict(getattr(builder, "_pending", {}) or {})
        if pending:
            self._vol = self.volume.overlay(pending)
        return {"placed": len(pending), "failed": 0,
                "note": "dry run: written into the cached volume and nowhere else"}

    def refresh(self):
        """Forget the volume: the next read is off the file. A dry stage that writes
        the base volume back -- the plateau, the terraces -- says so here, else the
        stage after it reads the ground as it stood before, from this cache."""
        self._vol = None

    def save(self, path: str) -> str:
        """Write the volume as it now stands. What a dry run's `world_built.npz` is."""
        from .. import offline as _offline
        return _offline.save_volume(self.volume, path)


class LiveBackend:
    """The world as a running server. Writes blocks.

        Deliberately thin: it holds the editor and the site every existing pass already
        builds for itself, so that the orchestration above it is the same code in both
        modes. The passes themselves are unchanged.
        
    """
    live = True

    def __init__(self, rnd: Round, pad: int = 48):
        from .. import world
        self.round = rnd
        self.pad = pad
        self.editor = world.editor()
        self.site = None
        self.X = self.Z = self.S = None
        self._vol = None
        # A1: what has been committed into the volume and not yet written into the
        # world. One publish owes the server this and nothing else.
        self.blocks: dict = {}
        # `settlement.site_info()` falls back to it -- and every stage after the search
        # would be reading the wrong patch of world. A backend that cannot say where it
        # is yet says so instead of guessing.
        try:
            self._bind()
        except Exception:                    # noqa: BLE001 -- bound when it is known
            pass

    def _bind(self) -> bool:
        """True when it now is."""
        from .. import world
        if self.site is not None:
            return True
        s = self.round.site or self.round.chosen_site() or settlement_site(self.round)
        if not s:
            return False
        self.X, self.Z = s["origin"]
        self.S = s["size"]
        self.site = world.load_site(self.editor, self.X - self.pad, self.Z - self.pad,
                                    self.S + 2 * self.pad, self.S + 2 * self.pad)
        return True

    def rebind(self) -> bool:
        """Forget the site and read it again. What a stage calls after it moves one."""
        self.site = None
        self._vol = None
        return self._bind()

    @property
    def volume(self):
        if not self._bind():
            raise RuntimeError(
                "this backend has no site yet: the round names none, no site.json has "
                "been written and no site search has chosen one")
        if self._vol is None:
            from gdpc.vector_tools import Rect
            from .. import observe, world
            self.editor.loadWorldSlice(
                Rect((self.X - self.pad, self.Z - self.pad),
                     (self.S + 2 * self.pad, self.S + 2 * self.pad)), cache=True)
            h = self.editor.worldSlice.heightmaps[world.HEIGHTMAP].astype(int) - 1
            inner = h[self.pad:self.pad + self.S, self.pad:self.pad + self.S]
            y0, y1 = max(0, int(inner.min()) - 8), int(h.max()) + 48
            self._vol = observe.Volume.from_world_slice(
                self.editor.worldSlice, self.X - self.pad, self.Z - self.pad,
                self.S + 2 * self.pad, self.S + 2 * self.pad, y0, y1)
            # A1. `refresh()` means "read the ground again" -- a pass that ran in its
            # own process has moved it -- and it never meant "forget what we built".
            # While every commit went over the wire the two were the same thing; now
            # that they are not, this line is what keeps them the same.
            if self.blocks:
                self._vol = self._vol.overlay(self.blocks)
        return self._vol

    def refresh(self):
        self._vol = None

    def execute(self, program: str, vol=None):
        return OfflineBackend(self.round).execute(program, vol)

    def commit(self, builder):
        """Take a part's writes into this backend's own volume. **Nothing is sent to
                the server here.** v2, A1.

                So a live round now builds its parts against the volume, like a dry run, and
                the world is written once. `blocks` is what is owed the world, in the order it
                was laid, and a part's own record still says how many blocks it placed.
                
        """
        pending = dict(getattr(builder, "_pending", {}) or {})
        if pending:
            self._vol = self.volume.overlay(pending)
            self.blocks.update(pending)
        return {"placed": len(pending), "failed": 0,
                "note": "the world is written once, "
                        "by publish()"}

    def publish(self) -> dict:
        """Everything committed since the last publish, into the world, in one pass.

                One write and one read-back. `Builder.flush` chunks the bulk pass with block
                updates off and then re-places the connective blocks with them on, which is
                the one thing only a running server computes -- a fence's joins, a wall's
                posts, a stair's mitre -- and it is the reason this is a server's job at all.
                The read-back is the single `loadWorldSlice` the per-part path did per part.
                
        """
        if not self.blocks:
            return {"published": 0, "note": "nothing was committed to publish"}
        from ..buildlib import Builder
        if not self._bind():
            return {"error": "this backend has no site yet: nothing can be published"}
        b = Builder(self.site)
        for (x, y, z), state in self.blocks.items():
            b.place_block(x, y, z, state)
        t0 = time.perf_counter()
        res = b.flush()
        self.editor.runCommand("save-all flush")
        time.sleep(8)
        published, self.blocks = len(self.blocks), {}
        self.refresh()                     # one read-back, not one per part
        return {"published": published, "placed": res.get("placed"),
                "failed": res.get("failed"),
                "seconds": round(time.perf_counter() - t0, 1)}

    def run_pass(self, name: str) -> dict:
        """One committed pass, through the proven live path -- scripts/settlement_run.py:"""
        import subprocess
        p = subprocess.run(
            [os.path.join(ROOT, ".venv", "bin", "python"),
             os.path.join(ROOT, "src", "ethoslm", "pipeline", "_commands", "settlement_run.py"), name],
            env=dict(os.environ, ETHOSLM_SETTLEMENT=self.round.name),
            capture_output=True, text=True)
        self.refresh()
        passes = self.round.rel("passes.json")
        rec = (json.load(open(passes)).get(name, {})
               if os.path.exists(passes) else {})
        return {"returncode": p.returncode, "log_tail": p.stdout[-2000:],
                **{k: rec.get(k) for k in ("placed", "failed", "error", "lint")
                   if k in rec}}


def _report_path(rnd: Round) -> str:
    stem = os.path.splitext(os.path.basename(rnd.path))[0]
    return rnd.rel("round.json" if stem == rnd.name else f"round.{stem}.json")


def _candidate_or_none(rnd) -> str | None:
    from .. import deps
    try:
        return deps.candidate_id(rnd)
    except Exception:                          # noqa: BLE001 -- a report never fails
        return None


def stage_report(rnd: Round, be, results: dict) -> dict:
    """`<state>/round.json`.

        One file per round, which is the thing the 61 measurement kinds were standing in
        for. The pre-registration is written once and never rewritten -- the first write
        wins, as it has since E1a.
    """
    p = _report_path(rnd)
    prev = json.load(open(p)) if os.path.exists(p) else {}
    merged = dict(prev.get("results") or {})
    # **A merge that keeps every answer keeps the withdrawn ones too.** The review's
    # fourth finding, at the report: this merged the new results over the old and
    # nothing retired what the new run had superseded, so the held-out village's report
    # carried an old *pending preview* and a new *stopped plan* at the same time -- two
    # statements about two different candidates, in one document, with no way for a
    # reader to tell which was current. Two rules, and both are about what this
    # invocation has actually re-decided. A stage that ran again replaces its own entry
    # (the plain merge below). A stage *downstream* of the earliest one that ran has not
    # been re-decided and its answer is about the candidate before this pass, so it is
    # withdrawn by name rather than carried. `pending` and `stopped` are this
    # invocation's or they are gone: they say what the round is doing now.
    order = list(default_stages(rnd))
    ran = [s for s in order if s in results]
    withdrawn = {}
    if ran:
        after = order[order.index(ran[0]):]
        for name in after:
            if name in merged and name not in results:
                withdrawn[name] = (f"withdrawn: {ran[0]} ran again in this invocation "
                                   f"and {name} is downstream of it, so its answer "
                                   f"describes the candidate before this pass")
                merged.pop(name, None)
    for state in ("pending", "stopped"):
        if state in merged and state not in results:
            withdrawn[state] = (f"withdrawn: the round is no longer {state}; this "
                                f"invocation did not report it")
            merged.pop(state, None)
    merged.update(results)
    from .. import deps as _deps_r
    doc = {"round": rnd.name, "config": rnd.path, "intent": rnd.intent,
           "voice": rnd.voice, "flags": rnd.flags, "base_volume": rnd.base_volume,
           "preregistered": prev.get("preregistered") or rnd.preregistered,
           "results": merged,
           "stages_last_run": sorted(results),
           "withdrawn": withdrawn,
           # which design these results are about, so two reports can be told apart
           "candidate": _candidate_or_none(rnd),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(rnd.state, exist_ok=True)
    json.dump(doc, open(p, "w"), indent=1)
    return {"round_json": p}


def _place_stage(name: str):
    """A separate module because they are a different job -- a *place* built from typed
    parts rather than a settlement built by builder calls -- and because this file is
    already the largest in the project. Bound late so importing `pipeline` does not
    import it."""
    def stage(rnd, be, results):                                 # noqa: ANN202
        from .. import place
        return getattr(place, f"stage_{name}")(rnd, be, results)
    stage.__name__ = f"stage_{name}"
    return stage


DETERMINISTIC = ("programs", "cards", "judge")


PLACE = ("reading", "interpret", "place_spec", "site_search", "site", "plateau", "plan",
         "ground", "terraces",
         "circulation", "cache", "preview", "parts", "finish", "lint", "material",
         "section", "inspect", "improve",
         "render", "cards", "qualify", "place_check", "judge", "readout")

#: The same place, offline, with the world in a volume. Two differences from `PLACE` and
#: both are forced: - `cache` runs **before** `plateau`, because offline a plateau is a
#: cut applied to the volume the run is scored against and there is no volume until the
#: ground has been read; - `render`, `cards` and `judge` are absent. Chunky needs the
#: world on disk and a dry run has not written one; the frames are the live run's and A4
#: is what bounds them when it gets there. ...and `cache` runs **before** `site` as well
#: (v2, C5): offline the site briefing is measured off the cached volume, so the ground
#: has to be in hand before it is read. ...and `preview` runs **after** `circulation`
#: (v2, C4, found by looking at one): a building drawn before the lanes are routed has
#: no way in, so `site()` refuses it and the preview's five buildings are five empty
#: pads. The stage re-routes the lanes itself where its revision changes the plan.
PLACE_DRY = ("reading", "interpret", "place_spec", "site_search", "cache", "site", "plateau", "plan",
             "ground", "terraces", "circulation", "preview", "parts", "finish", "lint", "material",
             "section", "inspect", "improve", "qualify", "place_check", "readout")


def default_stages(rnd: Round) -> tuple:
    if rnd.sentence and rnd.flags.get("plan_only"):
        # **A dry plan-only run is the dry list, cut at the plan.** It took the live
        # list, which has no `cache` stage, so a dry run that only wanted a plan chose a
        # site and then stopped because the ground it had just chosen had never been
        # read. Two flags, one order: `dry_run` decides *which* stages exist and
        # `plan_only` decides *how many* of them run.
        got = PLACE_DRY if rnd.flags.get("dry_run") else PLACE
        return got[:got.index("plan") + 1]
    if rnd.sentence and rnd.flags.get("dry_run"):
        return PLACE_DRY
    return PLACE if rnd.sentence else DETERMINISTIC


def _stopped(res: dict) -> str | None:
    """The reason a stage gives for the round being over, or None. See `run`."""
    if res.get("stop"):
        return str(res.get("error") or res.get("stop"))
    for v in res.values():
        if isinstance(v, dict) and v.get("stop"):
            return str(v.get("error") or v.get("stop"))
    return None


#: How many times one stage is re-entered on routed answers before the driver gives up.
#: A plan of a place and thirteen districts is fourteen answers and a hand-back each;
#: sixty-four is far past any stage in the record.
ROUTED_REENTRIES = 64


def _needs_model(res: dict) -> list:
    """Every agent job this stage result is waiting on, however it was returned.

        **One rule, and it is `model.staged`'s.** This read the result's *entries* only, so
        a stage that answers flat -- `stage_place_spec`, `stage_reading`'s claims job, the
        site briefing -- was invisible to the loop that waits for agent work and to the gate
        that stops the round for it. The Persian case found it in production: the spec job
        was staged, nothing saw it, and the site search ran next and stopped on the
        `place.json` the unanswered job was going to write. The router has always looked in
        all three places; the controller now looks in the same three.
        
    """
    from ..model import staged
    return list(staged(res))


def _reenter(res: dict) -> str | None:
    """Why this stage has asked to be run again, or None.

        **A state the driver executes, and the reason it exists.** The integration review
        drove the production return shape of a repaired preview through this very function's
        caller: the stage returned a dict whose `note` said "the stage re-enters to inspect
        the plan it produced", the driver read the note as it reads any other string, and
        the round went on to build a plan nobody had inspected. Prose is not control flow.
        A stage that has changed the candidate under itself says `status: "reenter"` -- on
        the result or on one of its entries -- and this is what the loop below asks.
        
    """
    if res.get("status") == "reenter":
        return str(res.get("why") or "the stage asked to be run again")
    for k, v in res.items():
        if isinstance(v, dict) and v.get("status") == "reenter":
            return f"{k}: {v.get('why') or 'the stage asked to be run again'}"
    return None


def _drive_reentries(rnd: Round, be, results: dict, name: str, res: dict) -> dict:
    """Run `name` again for as long as it asks to be, recording every pass.

        Bounded by `ROUTED_REENTRIES`, and a stage still asking at the bound is a stage in a
        loop: the round is stopped by name rather than left to spin. A stage that asks to
        re-enter and then asks for a model call is answered by the caller's wait loop on the
        next turn, which is why this returns rather than swallowing that state.
        
    """
    from ..model import router
    trail = list(res.get("reentries") or [])
    for _ in range(ROUTED_REENTRIES):
        why = _reenter(res)
        if not why:
            break
        trail.append(why)
        print(f"   re-enter {name}: {why}", flush=True)
        res = _pipeline.STAGES[name](rnd, be, results)
        for _ in range(ROUTED_REENTRIES):
            if not router().fulfil(res):
                break
            res = _pipeline.STAGES[name](rnd, be, results)
    else:
        return {**res, "status": "error", "stop": True, "reentries": trail,
                "error": (f"{name} asked to be re-entered {ROUTED_REENTRIES} times and "
                          f"is still asking: {trail[-1] if trail else 'no reason'}")}
    if trail:
        res = {**res, "reentries": trail}
    return res


def run(rnd: Round, stages=DETERMINISTIC, backend=None, wait: int = 0,
        poll: int = 300) -> dict:
    """Execute `stages` of `rnd` and write `<state>/round.json`.

        `wait` is what makes a settlement round one command. The builders are
        out-of-process agents; a stage that needs one reports `needs_model` and stops, and
        every round so far has been driven by a person re-running the command when the
        programs appeared. That is fine offline and impossible live: this sandbox gives
        every shell its own network namespace, so the server a round is committing to dies
        with the call that started it, and a round whose builders take an hour cannot be
        two calls. So a stage that is waiting on a model sleeps and asks again, up to
        `wait` seconds, and the paths it is waiting on are printed every time it does.

        It re-enters the *same* stage and nothing else. Re-running the whole list would
        re-cache the base volume between waves, which is the one thing `stage_cache`
        exists to refuse.
        
    """
    be = backend or OfflineBackend(rnd)
    results = {}
    for name in stages:
        if name not in _pipeline.STAGES:
            raise ValueError(f"no stage {name!r}; known stages are {list(_pipeline.STAGES)}")
        t0 = time.perf_counter()
        print(f"== {rnd.name}: {name}", flush=True)
        res = _pipeline.STAGES[name](rnd, be, results)
        deadline = time.time() + wait
        # A role routed to an API model is answered here, in-process, and the stage is
        # re-entered at once with the answer on disk; a role left to the supervising
        # agent is what the wait below is for. `fulfil` returns 0 when nothing is
        # routed, so a round with no API configured runs exactly as before.
        from ..model import router
        for _ in range(ROUTED_REENTRIES):
            if not router().fulfil(res):
                break
            res = _pipeline.STAGES[name](rnd, be, results)
        res = _drive_reentries(rnd, be, results, name, res)
        while wait and _needs_model(res) and time.time() < deadline:
            for rec in _needs_model(res):
                bl = rec.get("blinded") or {}
                print(f"   waiting on {rec.get('role') or 'agent'}: "
                      f"WRITE {bl.get('write', rec.get('write') or rec.get('request'))}"
                      + (f"  CHECK {bl['check']}" if bl.get("check") else ""),
                      flush=True)
            print(f"   sleeping {poll}s ({int(deadline - time.time())}s of patience "
                  f"left)", flush=True)
            time.sleep(poll)
            res = _pipeline.STAGES[name](rnd, be, results)
            for _ in range(ROUTED_REENTRIES):
                if not router().fulfil(res):
                    break
                res = _pipeline.STAGES[name](rnd, be, results)
            res = _drive_reentries(rnd, be, results, name, res)
        results[name] = res
        secs = round(time.perf_counter() - t0, 2)
        results[name].setdefault("seconds", secs)
        record("round", name=rnd.name, stage=name, seconds=secs)
        # **A stage still waiting on an agent is a stage that has not finished.** The
        # integration review drove this loop: a preview returning an unanswered judge
        # job was followed by `parts`, because `wait=0` skips the loop above entirely
        # and nothing after it asked whether the stage was still pending. Every stage
        # downstream then ran against a candidate nobody had inspected, and the only
        # thing preventing construction was a person restricting the stage list by hand.
        # Pending is an **outcome**, like stopping: the round says what it is waiting
        # for and where to write the answer, and the next invocation resumes at the same
        # stage with the answer on disk. That is what makes a terminal agent a
        # sufficient runtime rather than a thing the driver hopes somebody supervises.
        waiting = _needs_model(res)
        if waiting:
            asks = []
            for rec in waiting:
                bl = rec.get("blinded") or {}
                asks.append({"entry": rec.get("level") or rec.get("role") or "agent",
                             "role": rec.get("role"),
                             "request": bl.get("request") or rec.get("request"),
                             "write": bl.get("write") or rec.get("write"),
                             "check": bl.get("check")})
            results["pending"] = {"stage": name, "waiting": asks,
                                  "why": (f"{name} is waiting on {len(asks)} agent "
                                          f"answer(s); the round resumes at this stage "
                                          f"when they are written")}
            for a in asks:
                print(f"   PENDING {name}/{a['entry']}: WRITE {a['write']}"
                      + (f"  READ {a['request']}" if a.get("request") else ""),
                      flush=True)
            break
        # A stage may declare the round over. A plan that fails validation twice stops
        # the round with the list, and the stages after it would otherwise route lanes
        # to footprints nobody accepted and lint a place nobody built. The only thing a
        # driver can do with a round whose plan is refused is say so.
        stop = _stopped(res)
        if stop:
            results["stopped"] = {"stage": name, "why": stop}
            print(f"   STOP: {stop}", flush=True)
            break
    # A re-read of named bars (`--measure`) is a reading beside the record, not a run of
    # the round: it writes `readout.reread.json` and leaves `round.json` alone.
    if not rnd.flags.get("measures"):
        results.update(stage_report(rnd, be, results))
    return results
