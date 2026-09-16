"""The stage driver's contract: flags off is today's pipeline, exactly.

The control arm is the baseline every experiment in the ladder is judged against, so the
first thing to prove is that `stages.run` with every flag off produces the very pending
block set `offline.run_program` produces. (What this deliberately does not test: re-
generating a program and comparing, because a model call is not reproducible. The
control arm is tested on the deterministic half of the path; the model half is what the
ladder's arms measure.)

Then the staged half, with stub models:

  - W010 is silent on a program judged against a massing extracted from itself, and
    fires when the building moves three blocks -- the spec's acceptance pair.
  - split runs massing -> preview -> elaboration, reports conformance, and a
    non-conforming elaboration is reported (never erred).
  - sight renders exactly one draft preview and offers exactly one revision.
  - control renders nothing and calls the model exactly once.

And then the cycle loop, whose whole claim is that two arms differ in exactly one
thing. That is tested as a diff rather than as a description: the same cycles run
twice, once seeing and once blind, every prompt captured, and the texts required to be
identical apart from the one sentence that names the attachment. If that case ever
fails, arm B has stopped being a control and the round measures nothing.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import lint, offline, stages  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R7 = os.path.join(ROOT, "out", "site_b")


def temp_site():
    """The volume linked, the small state files copied, so no arm of any test can touch the
    fixture.
    """
    d = tempfile.mkdtemp(prefix="stages_test_")
    os.symlink(offline.world_cache("site_b"), os.path.join(d, "world.npz"))
    for f in ("plan.json", "plots.json", "network.json", "site.json"):
        p = offline.fixture_path("site_b", f)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(d, f))
    return d


class _WriteRound:
    """The smallest thing `stage_write` reads: a state directory and a shot spec."""

    def __init__(self, d):
        from ethoslm import pipeline
        self.state = d
        self.name = "write_test"
        self.shots = {"snapshot": None, "verify_samples": 12}
        self.site = json.load(open(os.path.join(d, "site.json")))
        self.base_volume = "world.npz"
        self.flags = {}
        self._pipeline = pipeline

    def rel(self, *parts):
        return os.path.join(self.state, *parts)

    def volume(self):
        return offline.load_volume(self.rel(self.base_volume))


class _FakeLive:
    """A live backend with no server: the editor answers off the pre-build cache and
    the site collects what a Builder flushes, which is what the case is about."""

    live = True

    def __init__(self, rnd, pre, wrote):
        self.round = rnd
        self.pre = pre
        self.wrote = wrote
        self.editor = self
        self.site = offline.OfflineSite(pre)

    # -- the bits of gdpc.Editor stage_write uses
    def getBlock(self, pos):
        class _Id:
            def __init__(self, v):
                self.id = v
        return _Id(self.pre.state(*pos).split("[")[0])

    def runCommand(self, cmd):
        return None

    def refresh(self):
        self._vol = None


def main():
    if not (os.path.exists(offline.world_cache("site_b"))
            and os.path.exists(offline.fixture_path("site_b", "wave1.py"))):
        print("skip: no pre-build cache and recorded wave for that site")
        return 0
    cases = []
    site = temp_site()
    vol = offline.load_volume(os.path.join(site, "world.npz"))
    wave1 = offline.fixture_path("site_b", "wave1.py")
    wave1_src = open(wave1).read()

    # --- the control arm is today's path, exactly --------------------------
    base = offline.run_program(wave1, vol, network=None,
                               plots=stages._Registry(site))
    calls = {"n": 0}

    def canned(brief, images):
        calls["n"] += 1
        assert images is None, "control arm must not show the model images"
        return wave1_src

    res = stages.run("the wave 1 brief", site, build=canned, vol=vol,
                     arm="test_control", do_lint=False)
    same = res["builder"]._pending == base._pending
    cases.append(("control: identical pending block set to offline.run_program",
                  same, f"{len(res['builder']._pending)} vs {len(base._pending)}"))
    cases.append(("control: one model call, no images, no massing, no bracket",
                  calls["n"] == 1 and "massing" not in res["report"]
                  and "bracket" not in res["report"]
                  and not [f for f in os.listdir(res["arm_dir"])
                           if f.endswith(".png")], f"{calls['n']} calls"))

    # --- W010: silent against a massing extracted from itself --------------
    occ = lint.massing_occupancy(base._pending)
    applied = stages.apply_pending(vol, base._pending)
    ctx = lint.Context(applied, None, None, None, [], massing=occ)
    silent = list(lint.w010_massing_conformance(ctx))
    cases.append(("W010 silent on wave 1 vs a massing extracted from itself",
                  not silent, silent[0].message if silent else ""))

    moved = {(x + 3, z): span for (x, z), span in occ.items()}
    ctx2 = lint.Context(applied, None, None, None, [], massing=moved)
    fired = list(lint.w010_massing_conformance(ctx2))
    cases.append(("W010 fires when the building moves three blocks",
                  len(fired) == 1 and fired[0].severity == lint.WARNING,
                  fired[0].detail.get("coverage") if fired else "no finding"))
    cases.append(("W010 is a warning, never an error",
                  all(f.severity == lint.WARNING for f in fired), ""))

    # --- split: massing -> preview -> elaboration, conformance reported ----
    MASSING = ("place_cuboid(1600, 71, -700, 1612, 77, -688, 'stone_bricks')\n"
               "roof(1600, -700, 1612, -688, 78, 'spruce')\n")
    ELAB_GOOD = MASSING + \
        "place_cuboid(1603, 72, -701, 1605, 74, -701, 'spruce_planks')\n"
    ELAB_AWAY = ("place_cuboid(1620, 71, -680, 1632, 77, -668, 'stone_bricks')\n"
                 "roof(1620, -680, 1632, -668, 78, 'spruce')\n")

    def staged(sources):
        it = iter(sources)
        return lambda brief, images: next(it)

    brief = {"massing": "mass it", "elaboration": "elaborate:\n{massing_program}"}
    res2 = stages.run(brief, site, split=True, build=staged([MASSING, ELAB_GOOD]),
                      vol=vol, arm="test_split", do_lint=False)
    r2 = res2["report"]
    cases.append(("split: massing stage ran, previewed, and conformance is silent",
                  r2.get("conformance") == "silent" and r2.get("massing", {})
                  .get("columns", 0) > 100,
                  f"{r2.get('massing', {}).get('columns')} columns"))
    mass_imgs = [f for f in os.listdir(res2["arm_dir"]) if "massing" in f
                 and f.endswith(".png")]
    cases.append(("split: the massing was rendered for the elaborator",
                  len(mass_imgs) == 2, str(mass_imgs)))

    # The massing card must be a picture of the massing. the card is 13x13 of building
    # plus 6 of pad.
    cropped = stages._preview_mod().crop(
        stages.apply_pending(vol, res2["builder"]._pending),
        *stages._bounds(res2["builder"]._pending), pad=6)
    cases.append(("split: the massing card is cropped to the built mass, not the site",
                  cropped.shape[0] <= 40 and cropped.shape[2] <= 40,
                  f"{cropped.shape[0]}x{cropped.shape[2]} columns "
                  f"(volume is {vol.shape[0]}x{vol.shape[2]})"))

    res3 = stages.run(brief, site, split=True, build=staged([MASSING, ELAB_AWAY]),
                      vol=vol, arm="test_split_away", do_lint=False)
    conf = res3["report"].get("conformance")
    cases.append(("split: an elaboration that walked away is reported",
                  isinstance(conf, list) and conf[0]["coverage"] < 0.85,
                  str(conf if not isinstance(conf, list) else conf[0]["coverage"])))

    # --- select: a bracket over massing candidates -------------------------
    def tie_judge(a, b, prompt):
        return "A"                       # pure position: every pair ties

    res4 = stages.run(brief, site, split=True, select=4,
                      build=staged([MASSING, ELAB_AWAY.replace("1620", "1621"),
                                    ELAB_GOOD]),
                      ask=tie_judge, vol=vol, arm="test_select", do_lint=False)
    sel = res4["report"].get("selection")
    cases.append(("select is adaptive: a tie ends the search after one challenge",
                  sel is not None and sel["comparisons"] == 1
                  and sel["history"][0]["result"] == "tie"
                  and sel["winner"].endswith("massing_0.py"),
                  str(sel)))

    # --- sight: one look, one revision -------------------------------------
    seen = {"images": None, "calls": 0}

    def sighted(brief_text, images):
        seen["calls"] += 1
        if images:
            seen["images"] = images
        return MASSING

    res5 = stages.run("brief", site, sight=True, build=sighted, vol=vol,
                      arm="test_sight", do_lint=False)
    cases.append(("sight: the model is called twice and shown its own build",
                  seen["calls"] == 2 and seen["images"]
                  and all(os.path.exists(p) for p in seen["images"]),
                  f"{seen['calls']} calls, {len(seen['images'] or [])} images"))

    # --- place, look, adjust ---------------------------------------------- The claim
    # the sight round rests on is that arms B and C differ in exactly one thing. So the
    # test is a diff: run the same cycles twice, once seeing and once blind, capture
    # every prompt, and require the texts to be identical apart from the one sentence
    # that names the attachment.
    CYCLES = [{"name": "massing", "brief": "Decide the masses."},
              {"name": "shell", "brief": "Build the walls and the roof."},
              {"name": "detail", "brief": "Add trim and depth."}]

    def recorder(store):
        def build(brief_text, images):
            store.append({"brief": brief_text, "images": list(images or [])})
            return MASSING
        return build

    seen_c, seen_b = [], []
    resC = stages.run_cycles("the brief", site, cycles=CYCLES, see=True,
                             build=recorder(seen_c), vol=vol, arm="test_cycles_C",
                             name="w", do_lint=False)
    resB = stages.run_cycles("the brief", site, cycles=CYCLES, see=False,
                             build=recorder(seen_b), vol=vol, arm="test_cycles_B",
                             name="w", do_lint=False)

    cases.append(("cycles: one model call per cycle, in both arms",
                  len(seen_c) == len(seen_b) == len(CYCLES),
                  f"C {len(seen_c)}, B {len(seen_b)} for {len(CYCLES)} cycles"))
    cases.append(("cycles: the seeing arm is shown its own build after cycle 1",
                  not seen_c[0]["images"]
                  and all(seen_c[i]["images"] for i in (1, 2))
                  and all(os.path.exists(p) for p in seen_c[2]["images"]),
                  f"images per cycle: {[len(s['images']) for s in seen_c]}"))
    cases.append(("cycles: the blind arm is shown nothing, ever",
                  not any(s["images"] for s in seen_b),
                  f"{sum(len(s['images']) for s in seen_b)} images"))
    diffs = [c["brief"].replace("\n" + stages.SEEN + "\n", "") != b["brief"]
             for c, b in zip(seen_c, seen_b)]
    cases.append(("cycles: B and C read byte-identical text but for SEEN",
                  not any(diffs), f"{sum(diffs)} of {len(diffs)} cycles differ"))
    cases.append(("cycles: ...and SEEN really is in C's text and not in B's",
                  stages.SEEN in seen_c[1]["brief"]
                  and stages.SEEN not in seen_b[1]["brief"], ""))
    cases.append(("cycles: each cycle carries the whole accumulated program forward",
                  "## Your program so far" not in seen_c[0]["brief"]
                  and all("## Your program so far" in seen_c[i]["brief"]
                          for i in (1, 2)), ""))
    cases.append(("cycles: both arms land the same pending set from the same stub",
                  resC["builder"]._pending == resB["builder"]._pending,
                  f"{len(resC['builder']._pending)} blocks"))
    cases.append(("cycles: every cycle's program is kept, not overwritten",
                  len([f for f in os.listdir(resC["arm_dir"])
                       if f.endswith(".py")]) == len(CYCLES),
                  f"{sorted(f for f in os.listdir(resC['arm_dir']) if f.endswith('.py'))}"))

    # A cycle that does not preflight must not end the candidate: the accumulated
    # program stays where it was and the loop goes on. Losing a candidate to one typo
    # measures the model's typing rather than the loop.
    def flaky(brief_text, images):
        this_pass = brief_text.split("# This pass")[-1][:80]
        return ("place_cuboid(0, 0, 0, 1, 1, 1, 'not_a_real_block')"
                if CYCLES[1]["brief"] in this_pass else MASSING)

    resF = stages.run_cycles("the brief", site, cycles=CYCLES, see=False,
                             build=flaky, vol=vol, arm="test_cycles_F", name="w",
                             do_lint=False)
    cases.append(("cycles: a cycle that fails preflight is survived, not fatal",
                  resF["builder"] is not None
                  and resF["report"]["cycles_landed"] == len(CYCLES) - 1,
                  f"{resF['report'].get('cycles_landed')} of {len(CYCLES)} landed"))

    # --- Part R: the built volume written into the world it is photographed in ---
    # `stage_write` is the answer to a premise this project had wrong: Chunky renders
    # the save at run/server/world and a dry run writes no block anywhere, so a city in
    # a volume could not be looked at. It refuses off a live backend by name, and on one
    # it writes the difference between the round's own pre-build cache and its
    # world_built.npz -- every block the build decided and not one more.
    from ethoslm import pipeline
    from ethoslm.buildlib import Builder as _B
    from ethoslm import world as world_mod

    cases.append(("stage_write refuses on an offline backend, by name",
                  "skipped" in pipeline.stage_write(
                      _WriteRound(site), pipeline.OfflineBackend(_WriteRound(site)), {}),
                  ""))
    pre = offline.load_volume(os.path.join(site, "world.npz"))
    built = stages.apply_pending(pre, base._pending)
    offline.save_volume(built, os.path.join(site, "world_built.npz"))
    s_ = json.load(open(os.path.join(site, "site.json")))
    X, Z, S = s_["origin"][0], s_["origin"][1], s_["size"]
    wrote = {}
    fake = _FakeLive(_WriteRound(site), pre, wrote)
    # No server here, so the one call that needs one is stubbed: what is under test is
    # which cells `stage_write` decides to place, not that gdpc can place them.
    real_flush, real_snap = _B.flush, world_mod.snapshot
    _B.flush = lambda self, **k: (wrote.update(self._pending),
                                  {"placed": len(self._pending), "failed": 0})[1]
    world_mod.snapshot = lambda tag: f"stub:{tag}"
    try:
        got = pipeline.stage_write(_WriteRound(site), fake, {})
    finally:
        _B.flush, world_mod.snapshot = real_flush, real_snap
    want = {k: v for k, v in base._pending.items()
            if X - 24 <= k[0] <= X + S - 1 + 24 and Z - 24 <= k[2] <= Z + S - 1 + 24
            and pre.state(*k).split("[")[0] != v.split("[")[0].split(":")[-1]}
    cases.append(("stage_write writes the diff between the cache and what was built",
                  got.get("written") == len(wrote) and len(wrote) >= len(want) > 0,
                  f"{got.get('written')} cells, {len(wrote)} placed, "
                  f"{len(want)} the wave changed"))
    cases.append(("stage_write verifies the ground and snapshots before it writes",
                  got.get("verified", {}).get("mismatches") == 0
                  and bool(got.get("snapshot")),
                  f"{got.get('verified')}"))

    # --- A1: a live round publishes once, not once per part ---------------- The live
    # path used to flush every part over the wire, `save-all flush`, drop the volume and
    # pull the padded site back out of the server -- once per part. What is under test
    # is the contract that replaced it: a commit goes into the volume and nowhere else,
    # the world is owed the difference, and one `publish()` pays it in one
    # `Builder.flush` (whose second pass is the connective states only a server
    # computes) and one read-back. The server here is the pre-build cache.
    import time as _time
    import types as _types
    from ethoslm import observe as _observe

    class _CountingEditor:
        """The three things `LiveBackend.volume` and `publish` ask a server for."""

        def __init__(self, vol):
            self.loads = 0
            self.commands = []
            hm = offline.surface_heights(vol) + 1
            self.worldSlice = _types.SimpleNamespace(
                heightmaps={world_mod.HEIGHTMAP: hm}, _vol=vol)

        def loadWorldSlice(self, rect, cache=True):
            self.loads += 1

        def runCommand(self, cmd):
            self.commands.append(cmd)

    class _Pending:
        def __init__(self, cells):
            self._pending = dict(cells)

    live_pre = offline.load_volume(os.path.join(site, "world.npz"))
    ed = _CountingEditor(live_pre)
    be_live = pipeline.LiveBackend.__new__(pipeline.LiveBackend)
    be_live.round, be_live.pad, be_live.editor = _WriteRound(site), 0, ed
    be_live.site = offline.OfflineSite(live_pre)
    be_live.X, be_live.Z = X, Z
    be_live.S, be_live._vol, be_live.blocks = S, None, {}
    parts = [_Pending({(X + 4 + i, 80, Z + 4 + j): "minecraft:stone_bricks"
                       for j in range(3)}) for i in range(4)]
    wrote2, slept = {}, []
    real_from_ws = _observe.Volume.__dict__["from_world_slice"]
    real_sleep = _time.sleep
    _observe.Volume.from_world_slice = staticmethod(
        lambda ws, *a, **k: offline.load_volume(os.path.join(site, "world.npz")))
    _B.flush = lambda self, **k: (wrote2.update(self._pending),
                                  {"placed": len(self._pending), "failed": 0})[1]
    _time.sleep = lambda s: slept.append(s)
    try:
        for p in parts:
            be_live.commit(p)
        committed = dict(be_live.blocks)
        in_volume = all(be_live.volume.name(*c) == "stone_bricks" for c in committed)
        loads_building, wrote_building = ed.loads, dict(wrote2)
        cmds_building = list(ed.commands)
        pub = be_live.publish()
        after = be_live.volume                   # the one read-back
        loads_after = ed.loads
    finally:
        _observe.Volume.from_world_slice = real_from_ws
        _B.flush, _time.sleep = real_flush, real_sleep
    cases.append(("A1: a part's commit reaches the volume and not the world",
                  in_volume and not wrote_building and not cmds_building
                  and len(committed) == 12,
                  f"{len(committed)} cells committed, {len(wrote_building)} written, "
                  f"{len(cmds_building)} server command(s)"))
    cases.append(("A1: four parts are one read of the world, not four",
                  loads_building == 1,
                  f"{loads_building} loadWorldSlice call(s) over four parts"))
    cases.append(("A1: one publish writes every committed block, once",
                  wrote2 == committed and pub.get("published") == len(committed)
                  and ed.commands == ["save-all flush"] and not be_live.blocks,
                  f"{pub.get('published')} published, {len(wrote2)} placed, "
                  f"commands {ed.commands}"))
    cases.append(("A1: publishing reads the world back once, and off the world",
                  loads_after == loads_building + 1 and after is not None,
                  f"{loads_after} read(s) in all for four parts and one publish"))

    # --- report always lands on disk ---------------------------------------
    on_disk = json.load(open(os.path.join(res2["arm_dir"], "report.json")))
    cases.append(("every arm writes its report where the ladder reads it",
                  on_disk["arm"] == "test_split", ""))

    # --- B0: the parts stage writes its record of ways in; it does not append ---- A
    # stage run twice on the same state left every `approach()` row twice in
    # `paths.json`, and the finish pass reads that file. One plot on the slope fixture
    # with its own lane, built twice off the same cache: one set of rows, both times.
    import test_place_spec
    from ethoslm import pipeline
    from ethoslm.pipeline import stages_build
    tmp = tempfile.mkdtemp(prefix="ethoslm-b0-paths-")
    try:
        rnd2, _plan = test_place_spec._voiced_fixture("white_render_dark_frame", tmp)
        first = stages_build.stage_parts(rnd2, pipeline.OfflineBackend(rnd2, dry_run=True), {})
        rows1 = json.load(open(os.path.join(tmp, "paths.json")))
        second = stages_build.stage_parts(rnd2, pipeline.OfflineBackend(rnd2, dry_run=True), {})
        rows2 = json.load(open(os.path.join(tmp, "paths.json")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    approaches = [r for r in rows1 if r.get("kind") == "approach"]
    cases.append(("B0: a parts stage run twice records each way in once",
                  first.get("built") == 1 and second.get("built") == 1
                  and rows1 and rows1 == rows2 and approaches
                  and all(r.get("label") == "short_axis_slope" for r in rows1),
                  f"{len(rows1)} row(s) after one run and {len(rows2)} after two, "
                  f"{len(approaches)} of them approaches, all for the one part"))

    fails = 0
    for label, ok, detail in cases:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        fails += not ok
    print(f"\n{len(cases) - fails}/{len(cases)} stage cases pass")
    shutil.rmtree(site, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
