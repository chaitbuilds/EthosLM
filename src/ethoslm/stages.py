"""The stage driver: generate, look, and only then commit -- every stage behind a flag.

    run(brief, site, *, sight=False, split=False, select=0, spaces=False)

With every flag off this takes **exactly today's path** -- one brief, one program,
preflight, execute, lint -- with no preview rendered, no massing stage and no judge
called. That default is the control arm, and it is how we prove the rebuild has not
silently changed the thing it is measured against.

The one design decision that matters, from the spec: **the artefact between stages is
the program itself, at successive levels of finish**. The massing stage emits a short
program of place_cuboid + roof; the elaboration stage receives that program plus a
picture of it and elaborates it. No new schema, no scratch volume -- which is why
preflight, dry runs, lint, flush and rollback all keep working with zero changes.

Flags, cheapest first:

Like the judge, this module has **no model in it**. `build(brief, images) -> source`
is the seam; the orchestrating agent (or a test stub) is the model. Everything else --
execution, preview, conformance, the report -- is deterministic and tested as such.

Each arm writes to its own directory under <site>/arms/<arm>/ and never overwrites
another. Arms on disk are why this project knows library >> no library, and that is
the only reason it knows it."""
from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np

from . import lint, observe, offline, settlement
from .measure import record

ROOT = settlement.ROOT

#: The judge's questions, verbatim from the spec. Deliberately not leading: a question
#: that names the defect you are looking for turns the judge into a checker with extra
#: steps.
QUESTION_MASSING = ("Which of these two reads more like a place a person built? "
                    "Answer A or B and give one sentence.")
QUESTION_ELEVATION = ("Which of these two buildings is better made? "
                      "A or B, one sentence.")


def _preview_mod():
    """The shared deterministic preview renderer."""
    from . import preview
    return preview


def arm_name(sight=False, split=False, select=0, spaces=False) -> str:
    parts = [n for n, on in (("sight", sight), ("split", split),
                             ("select%d" % select if select else "", select),
                             ("spaces", spaces)) if on]
    return "+".join(parts) if parts else "control"


class _Registry:
    """A plot registry read from one settlement directory, never committed. Arms are
    offline: they must see the fixture's plots and must not write them."""

    def __init__(self, state_dir: str):
        p = os.path.join(state_dir, "plots.json")
        self.plots = json.load(open(p)) if os.path.exists(p) else []
        self.claimed_this_pass: list = []

    reserve = settlement.PlotRegistry.reserve
    plots_list = settlement.PlotRegistry.plots_list


def apply_pending(vol: observe.Volume, pending: dict) -> observe.Volume:
    """A new volume with a program's writes applied. The input volume is untouched --
    the pre-build cache is a fixture and stays one."""
    codes = vol.codes.copy()
    palette = list(vol.palette)
    index = {s: i for i, s in enumerate(palette)}
    sx, sy, sz = vol.shape
    for (x, y, z), b in pending.items():
        lx, ly, lz = x - vol.x0, y - vol.y0, z - vol.z0
        if not (0 <= lx < sx and 0 <= ly < sy and 0 <= lz < sz):
            continue
        s = b.split(":")[-1]
        if s not in index:
            index[s] = len(palette)
            palette.append(s)
        codes[lx, ly, lz] = index[s]
    return observe.Volume(vol.x0, vol.y0, vol.z0, codes, palette)


def _bounds(pending: dict) -> tuple | None:
    """(x0, z0, x1, z1) of what a program wrote -- the argument order `preview.crop`
    takes. It used to return (xmin, zmin, xmax, zmax) and be unpacked as if it were
    (xmin, xmax, zmin, zmax), which crossed the axes: the crop ran from min(xmin, zmin)
    to max(xmin, zmin) in x and the same scramble in z, so on any site whose x and z
    have different signs it clamped to the whole volume. That is why every step-3
    massing card is a picture of a hillside with a building somewhere in it, why both
    adaptive comparisons tied, and why the judges' reasons described terrain."""
    if not pending:
        return None
    xs = [p[0] for p in pending]
    zs = [p[2] for p in pending]
    return min(xs), min(zs), max(xs), max(zs)


#: A card's long side, in pixels, before it is handed to a model. Cropping to the built
#: mass makes the picture right and small -- one massing came out 150x120, its elevation
#: 68x64, which is a thumbnail, not a card. Upscaled by an integer nearest-neighbour
#: repeat, so the image stays a pure function of the volume and the judge's content
#: cache still hits.
CARD_PX = 480


def _upscale(img, target: int = CARD_PX):
    k = max(1, target // max(img.shape[0], img.shape[1]))
    return np.repeat(np.repeat(img, k, 0), k, 1) if k > 1 else img


def _render(vol, pending, out_dir, tag, grey=False):
    """Preview a program's result, cropped to what it built: isometric + elevation."""
    pv = _preview_mod()
    import cv2
    applied = apply_pending(vol, pending)
    b = _bounds(pending)
    v = pv.crop(applied, b[0], b[1], b[2], b[3], pad=6) if b else applied
    paths = []
    for name, img in ((f"{tag}_iso.png", pv.preview(v, grey=grey)),
                      (f"{tag}_elevation.png", pv.elevation(v, grey=grey))):
        p = os.path.join(out_dir, name)
        cv2.imwrite(p, _upscale(img)[:, :, ::-1])
        paths.append(p)
    return paths


def _execute(source: str, path: str, vol, network, state_dir, allow_try: bool = False,
             allow_collide: bool = False):
    """Preflight, then run. Today's path, exactly: same preflight, same run_program,
    a fresh registry per execution the way every pass gets one.

    `allow_try` is passed straight to preflight and is only ever true when a round
    config says its stored programs predate the no-`try` rule -- see lint.preflight."""
    open(path, "w").write(source)
    pre = lint.preflight(source, allow_try=allow_try)
    if not pre.ok:
        return None, pre
    b = offline.run_program(path, vol, network=network, plots=_Registry(state_dir),
                            allow_collide=allow_collide)
    return b, pre


def _brief_text(brief, stage: str) -> str:
    """`brief` is one text, or {"massing": ..., "elaboration": ..., ["spaces": ...]}
    when the pipeline is split. Asking for a section the caller did not write is an
    error -- silently substituting the whole brief would un-split the experiment."""
    if isinstance(brief, str):
        return brief
    return brief[stage]


def run(brief, site, *, sight: bool = False, split: bool = False, select: int = 0,
        spaces: bool = False, build=None, ask=None, arm: str | None = None,
        name: str = "pass", vol=None, do_lint: bool = True,
        allow_try: bool = False, allow_collide: bool = False,
        lint_scope: tuple | None = None):
    """One pass through the staged pipeline. `site` is a settlement state directory
    (out/<name>) holding world.npz, plan.json, plots.json, network.json.

    `build(brief_text, image_paths_or_None) -> program source` is the model seam.
    `ask` is handed to the judge when select > 0. Returns a dict with the builder,
    the report, and the arm directory; also written to <site>/arms/<arm>/report.json."""
    if build is None:
        raise ValueError("run() needs a build callable -- there is no model in here")
    arm = arm or arm_name(sight, split, select, spaces)
    out_dir = os.path.join(site, "arms", arm)
    os.makedirs(out_dir, exist_ok=True)

    if vol is None:
        vol = offline.load_volume(os.path.join(site, "world.npz"))
    from .circulate import Network
    net_path = os.path.join(site, "network.json")
    network = Network.load(net_path) if os.path.exists(net_path) else None

    t_start = time.perf_counter()
    rep: dict = {"arm": arm, "flags": {"sight": sight, "split": split,
                                       "select": select, "spaces": spaces},
                 "name": name, "stages": []}
    massing_occ = None
    images = None

    # ---- massing stage (split only) --------------------------------------
    if split:
        mbrief = _brief_text(brief, "massing")
        if spaces and isinstance(brief, dict) and "spaces" in brief:
            mbrief = brief["spaces"] + "\n\n" + mbrief
        # Adaptive-k, per the ratified decision: fixed k=8 costs ~1M tokens per
        # settlement in selection alone. Generate one massing; each further candidate
        # challenges the running best in one position-swapped comparison; a challenger
        # that loses or ties ENDS the search -- the incumbent has survived a challenge.
        # `select` is the cap, not the count.
        from . import judge as judge_mod
        best = None
        k = max(select, 1)
        history = []
        for i in range(k):
            src = build(mbrief, None)
            mp = os.path.join(out_dir, f"{name}_massing_{i}.py"
                              if k > 1 else f"{name}_massing.py")
            mb, pre = _execute(src, mp, vol, network, site, allow_try, allow_collide)
            entry = {"stage": "massing", "candidate": i, "program": mp,
                     "preflight_errors": len(pre.errors)}
            if mb is None:
                rep["stages"].append(entry)
                continue
            tag = os.path.basename(mp)[:-3]
            imgs = _render(vol, mb._pending, out_dir, tag, grey=True)
            entry.update(pending=len(mb._pending), images=imgs)
            rep["stages"].append(entry)
            cand = (mp, mb, imgs)
            if best is None:
                best = cand
                continue
            res = judge_mod.compare(best[2][0], cand[2][0], QUESTION_MASSING,
                                    ask=ask, stage_name="massing_select")
            history.append({"challenger": mp, "result": res})
            if res == "b":
                best = cand                    # challenger wins; search continues
            else:
                break                          # incumbent survived; stop spending
        if best is None:
            rep["error"] = "no massing candidate survived preflight"
            _finish(rep, out_dir, t_start)
            return {"arm_dir": out_dir, "report": rep, "builder": None}
        if history:
            rep["selection"] = {"winner": best[0], "comparisons": len(history),
                                "history": history}
        chosen = best
        mpath, mbuilder, images = chosen
        massing_occ = lint.massing_occupancy(mbuilder._pending)
        rep["massing"] = {"program": mpath, "columns": len(massing_occ),
                          "ridge": max(h for _, h in massing_occ.values())
                          if massing_occ else None}
        ebrief = (_brief_text(brief, "elaboration")
                  .replace("{massing_program}", open(mpath).read()))
        stage_brief = ebrief
    else:
        stage_brief = _brief_text(brief, "elaboration") if isinstance(brief, dict) \
            else brief

    # ---- the build (control path when nothing above ran) ------------------
    src = build(stage_brief, images)
    ppath = os.path.join(out_dir, f"{name}.py")
    b, pre = _execute(src, ppath, vol, network, site, allow_try, allow_collide)
    rep["stages"].append({"stage": "build", "program": ppath,
                          "preflight_errors": len(pre.errors)})
    if b is None:
        rep["error"] = "preflight rejected the program"
        rep["preflight"] = pre.to_json()
        _finish(rep, out_dir, t_start)
        return {"arm_dir": out_dir, "report": rep, "builder": None}

    # ---- sight: one look, one revision ------------------------------------
    if sight:
        imgs = _render(vol, b._pending, out_dir, f"{name}_draft")
        revise = (stage_brief
                  + "\n\n## Your build, as it stands\n\n"
                  "The images attached are an isometric and a front elevation of "
                  "exactly what your program builds, rendered from its dry run. Look "
                  "at them the way a builder steps back and looks. Then return the "
                  "full program again, revised where the pictures say it is wrong -- "
                  "or unchanged where they say it is right.\n\n"
                  "```python\n" + src + "\n```\n")
        src2 = build(revise, imgs)
        p2 = os.path.join(out_dir, f"{name}_revised.py")
        b2, pre2 = _execute(src2, p2, vol, network, site, allow_try, allow_collide)
        rep["stages"].append({"stage": "sight_revision", "program": p2,
                              "preflight_errors": len(pre2.errors),
                              "draft_images": imgs})
        if b2 is not None:
            b, ppath = b2, p2

    rep["pending"] = len(b._pending)
    rep["program"] = ppath

    # ---- conformance (split only) -----------------------------------------
    if massing_occ:
        applied = apply_pending(vol, b._pending)
        ctx = lint.Context(applied, None, None, None, [], massing=massing_occ)
        findings = list(lint.w010_massing_conformance(ctx))
        rep["conformance"] = ([f.detail for f in findings] if findings
                              else "silent")
        json.dump([[x, z, lo, hi] for (x, z), (lo, hi) in sorted(massing_occ.items())],
                  open(os.path.join(out_dir, f"{name}_massing.json"), "w"))

    # ---- lint: the floor, unchanged ---------------------------------------
    if do_lint:
        applied = apply_pending(vol, b._pending)
        s = json.load(open(os.path.join(site, "site.json"))) \
            if os.path.exists(os.path.join(site, "site.json")) else None
        region = lint_scope
        if region is None and s:
            X, Zc = s["origin"]
            region = (X, Zc, X + s["size"] - 1, Zc + s["size"] - 1)
        ctx = lint.Context.build(applied, _Registry(site).plots, network=network,
                                 region=region, massing=massing_occ,
                                 fittings=b.fitting_cells)
        report = lint.lint(ctx)
        rep["lint"] = report.to_json()["counts"]
        rep["lint_scope"] = list(region) if region else None

    _finish(rep, out_dir, t_start)
    return {"arm_dir": out_dir, "report": rep, "builder": b, "program": ppath}


def _finish(rep, out_dir, t_start):
    rep["seconds"] = round(time.perf_counter() - t_start, 2)
    json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    record("stage_run", arm=rep["arm"], name=rep.get("name"),
           pending=rep.get("pending"), seconds=rep["seconds"],
           error=rep.get("error"))


# ------------------------------------------------------- place, look, adjust The
# `sight` flag above is one look at a finished program: write the whole thing, render it
# once, revise once. It lost to control in step 3 and the record has said "sight does
# not help" ever since. That is a code review with a photograph attached. The audit's
# prescription was "a human builder places, steps back, looks, adjusts", and *sight
# during construction* has never been implemented here. The rule this obeys is the
# determinism rule, applied to a mechanism that adds model turns: **the loop is code and
# the prompts are data, frozen before the run.** The model never decides when to look,
# what to look at, or how many passes to take, because every one of those decisions is a
# place improvisation would creep back into a harness the pipeline round just finished
# making deterministic.

#: The one difference between the arm that sees and the arm that does not. Both get
#: byte-identical cycle text; this sentence is appended for the arm holding the images,
#: because an unexplained attachment is its own confound. It names what the pictures are
#: and nothing else -- no defect, no measure, no instruction about what to change.
SEEN = ("The images attached are an isometric and a front elevation of exactly what "
        "your program currently builds, rendered from its dry run.")

#: How the accumulated program is carried between cycles. The artefact between stages is
#: the program itself, at successive levels of finish -- the same decision the split
#: pipeline made, and the reason preflight, dry runs, lint and rollback keep working.
CARRY = ("## Your program so far\n\nThis is the program as it stands. Continue it: "
         "return the **whole** program again, including everything below, with this "
         "pass's work added.\n\n```python\n{program}\n```\n")


def run_cycles(brief, site, *, cycles, see: bool = False, build=None,
               arm: str | None = None, name: str = "pass", vol=None,
               do_lint: bool = True, allow_try: bool = False,
               allow_collide: bool = False, lint_scope: tuple | None = None):
    """One candidate, built over a fixed sequence of cycles.

        `cycles` is a list of {"name", "brief"} read off the round config -- the fixed
        cycle structure, written down before the run and identical for every arm and every
        candidate. `see` decides only whether the builder is handed a preview of what its
        own accumulated program currently builds; the text it reads is the same either way
        apart from `SEEN`.

        A cycle whose program does not preflight or does not run leaves the accumulated
        program where it was and is recorded as such. That is deliberate: the alternative
        is to end the candidate on one typo, and a candidate lost at cycle 2 measures the
        model's typing rather than the loop.
        
    """
    if build is None:
        raise ValueError("run_cycles() needs a build callable -- no model in here")
    if not cycles:
        raise ValueError("run_cycles() needs at least one cycle")
    arm = arm or ("sight_cycles" if see else "blind_cycles")
    out_dir = os.path.join(site, "arms", arm)
    os.makedirs(out_dir, exist_ok=True)

    if vol is None:
        vol = offline.load_volume(os.path.join(site, "world.npz"))
    from .circulate import Network
    net_path = os.path.join(site, "network.json")
    network = Network.load(net_path) if os.path.exists(net_path) else None

    base = _brief_text(brief, "elaboration") if isinstance(brief, dict) else brief
    t_start = time.perf_counter()
    rep: dict = {"arm": arm, "flags": {"sight": see, "cycles": len(cycles)},
                 "name": name, "stages": []}

    source = None            # the accumulated program
    builder = None
    ppath = None
    images = None
    for i, cyc in enumerate(cycles):
        text = base + "\n\n# This pass\n\n" + cyc["brief"]
        if source is not None:
            text += "\n\n" + CARRY.replace("{program}", source)
        if see and images:
            text += "\n" + SEEN + "\n"
        src = build(text, images if (see and source is not None) else None)
        p = os.path.join(out_dir, f"{name}_{i}_{cyc['name']}.py")
        b, pre = _execute(src, p, vol, network, site, allow_try, allow_collide)
        entry = {"stage": "cycle", "cycle": i, "name": cyc["name"], "program": p,
                 "preflight_errors": len(pre.errors), "saw": bool(images) if see
                 else False}
        if b is None:
            entry["rejected"] = "preflight"
            rep["stages"].append(entry)
            continue
        source, builder, ppath = src, b, p
        entry["pending"] = len(b._pending)
        # Look *after* the cycle that placed, so the next cycle opens on a picture of
        # what is actually standing. Rendered offline through the dry-run path, cropped
        # to the built mass -- 6 ms, byte-deterministic, and the same call the previewer
        # is tested on.
        if see:
            images = _render(vol, b._pending, out_dir, f"{name}_{i}_{cyc['name']}")
            entry["images"] = images
        rep["stages"].append(entry)

    if builder is None:
        rep["error"] = "no cycle produced a program that runs"
        _finish(rep, out_dir, t_start)
        return {"arm_dir": out_dir, "report": rep, "builder": None}

    rep["pending"] = len(builder._pending)
    rep["program"] = ppath
    rep["cycles_landed"] = sum(1 for s in rep["stages"] if "pending" in s)

    if do_lint:
        applied = apply_pending(vol, builder._pending)
        s = json.load(open(os.path.join(site, "site.json"))) \
            if os.path.exists(os.path.join(site, "site.json")) else None
        region = lint_scope
        if region is None and s:
            X, Zc = s["origin"]
            region = (X, Zc, X + s["size"] - 1, Zc + s["size"] - 1)
        ctx = lint.Context.build(applied, _Registry(site).plots, network=network,
                                 region=region, fittings=builder.fitting_cells)
        rep["lint"] = lint.lint(ctx).to_json()["counts"]
        rep["lint_scope"] = list(region) if region else None

    _finish(rep, out_dir, t_start)
    return {"arm_dir": out_dir, "report": rep, "builder": builder, "program": ppath}


class BuildNeeded(RuntimeError):
    """No model in-process and the arm needs a program. Mirrors judge.JudgementNeeded:
    the request (brief text and any images) is on disk; the orchestrating agent
    answers it by writing build_<i>.py beside it and re-running the arm, which
    replays deterministically -- execution is pure and judgements are cached -- up
    to the next missing program."""

    def __init__(self, request_path: str, brief: str, images):
        self.request_path = request_path
        self.brief = brief
        self.images = images
        super().__init__(f"a build is needed: answer {request_path}")


def build_from_files(dir_path: str):
    """The file-backed model seam for stages.run. Call i returns dir/build_<i>.py if
    it exists; otherwise it writes dir/build_request_<i>.json and raises BuildNeeded."""
    os.makedirs(dir_path, exist_ok=True)
    counter = {"i": 0}

    def build(brief, images):
        i = counter["i"]
        counter["i"] += 1
        p = os.path.join(dir_path, f"build_{i}.py")
        if os.path.exists(p):
            return open(p).read()
        req = os.path.join(dir_path, f"build_request_{i}.json")
        json.dump({"i": i, "images": images or [], "brief": brief},
                  open(req, "w"), indent=1)
        raise BuildNeeded(req, brief, images)
    return build
