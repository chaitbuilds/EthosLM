"""Cards, rendering and blinded comparative judgements."""
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
from .round import Round


# At a city's four hundred it is six hours of Chunky inside a server session that has to
# hold the whole run, which is not a rendering problem: it is the round failing to
# finish. So what a place is photographed as is a **budget**, and it is a config field
# rather than a rule, because which frames are worth an hour is a question about what
# the round is for. The two whole-place frames, every defining part, a seeded sample of
# the ordinary buildings, and a flythrough from the outer gate to the palace written as
# frames.

#: The default budget. `sample` is the spec's number.
RENDER_BUDGET = {
    "place": True,
    "compound": True,
    "defining": True,       # every leaf that answers a defining part: palace, gates, walls
    "sample": 24,           # ...and this many ordinary buildings, drawn by seed
    "sample_seed": 1,
    "flythrough": True,
}

#: How many frames the flythrough is. Fifty at 13 s is eleven minutes, which buys about
#: two seconds of video at 24 fps and is the whole of what a still renderer can give a
#: camera move without becoming the round.
FLYTHROUGH_FRAMES = 48

#: The hard ceiling on frames a place renders, whatever the budget says. A budget is a
#: config field and a config field is a thing somebody can write 400 into; this is the
#: number that says the render stage is not allowed to become the round.
MAX_FRAMES = 128

#: Four hours. The preflight prints the rows by kind, the frames and the estimate before
#: the first frame.
COST_FRAME_S = 14.0
RENDER_BOUND_S = 4 * 3600.0


def preflight_render(rows: list, budget: dict, bound: float = RENDER_BOUND_S,
                     log=print) -> dict:
    """What the render is about to cost, printed before it is spent, refused over
    `bound`. `rows` are `budget_frames`' answer; every subject but the flythrough is
    four panels, the flythrough is its frames."""
    by = {}
    for r in rows:
        by[r["why"]] = by.get(r["why"], 0) + 1
    frames = sum(4 if r["why"] != "flythrough" else 1 for r in rows)
    if budget.get("flythrough"):
        frames += FLYTHROUGH_FRAMES
    est = frames * COST_FRAME_S
    out = {"rows": len(rows), "by_kind": by, "frames": int(frames),
           "estimated_seconds": round(est), "bound_seconds": float(bound),
           "refused": est > bound,
           "registered": {"COST_FRAME_S": COST_FRAME_S, "RENDER_BOUND_S": RENDER_BOUND_S,
                          "MAX_FRAMES": MAX_FRAMES}}
    log(f"   preflight: {len(rows)} subject rows {by}, about {frames} frames and "
        f"{est / 60:.0f} min against a bound of {bound / 60:.0f}"
        + (" -- REFUSED" if out["refused"] else ""))
    return out


def render_budget(plan: dict, spec: dict | None, shots: dict | None) -> dict:
    """The budget in force: the round's `shots["budget"]` over `RENDER_BUDGET`."""
    got = dict(RENDER_BUDGET)
    for k, v in ((shots or {}).get("budget") or {}).items():
        if k not in RENDER_BUDGET:
            raise ValueError(f"a render budget has no field called {k!r}; it declares "
                             f"{sorted(RENDER_BUDGET)}")
        got[k] = v
    got["sample"] = max(0, min(int(got["sample"]), MAX_FRAMES))
    return got


def _defines(spec: dict | None) -> set:
    return {d["name"] for d in (spec or {}).get("defining_parts", [])}


def budget_frames(plan: dict, spec: dict | None, budget: dict) -> list:
    """Which subjects this place is photographed as, in order. Deterministic. A4.

        One row per frame, each saying *why* it is being taken, so the readout can report
        what was rendered and what was not rather than leaving "we did not photograph 376
        buildings" to be inferred from a directory listing.
        
    """
    parts = _pipeline.plan_parts(plan)
    named = _defines(spec)
    compounds = {c["name"] for c in (plan.get("compounds") or [])}

    def answers(p):
        """Does this leaf answer a defining part? By `defines`; else **by name, and only
        for a leaf that is not a district's**.
        """
        n = p.get("name", "")
        if p.get("defines") in named:
            return True
        anc = p.get("in") or []
        if anc and anc[-1] == "defining":
            # every leaf the place level drew is a defining part -- a gate laid by the
            # arithmetic for a spec that named no gate part included
            return True
        if anc and anc[-1] not in compounds:
            return False
        return n in named or any(n.startswith(d + "_") for d in named)

    out = []
    if budget.get("place"):
        out += [{"name": "place_near", "why": "place", "kind": "place"},
                {"name": "place_far", "why": "place", "kind": "place"}]
    # A compound is photographed as **one thing** -- the palace, not eight cards of its
    # halls -- the way the place is: its whole rectangle framed as one subject.
    if budget.get("compound"):
        out += [{"name": c["name"], "why": "compound", "kind": "compound",
                 "rect": [min(c["x0"], c["x1"]), min(c["z0"], c["z1"]),
                          max(c["x0"], c["x1"]), max(c["z0"], c["z1"])]}
                for c in sorted(plan.get("compounds") or [], key=lambda c: c["name"])]
    defining = [p for p in parts if p.get("kind", "plot") != "quarter" and answers(p)]
    if budget.get("defining"):
        out += [{"name": p["name"], "why": "defining", "kind": p.get("kind", "plot"),
                 "type": p.get("type")} for p in sorted(defining,
                                                        key=lambda p: p["name"])]
    plots = sorted((p for p in parts if p.get("kind", "plot") == "plot"
                    and not answers(p)), key=lambda p: p["name"])
    n = min(int(budget.get("sample") or 0), len(plots))
    if n:
        # Seeded and reproducible, and spread over the whole place rather than drawn at
        # random.
        import random
        idx = sorted(random.Random(int(budget.get("sample_seed", 1)))
                     .sample(range(len(plots)), n))
        out += [{"name": plots[i]["name"], "why": "sample", "kind": "plot",
                 "type": plots[i].get("type")} for i in idx]
    if budget.get("flythrough"):
        out += [{"name": f"fly_{i:03d}", "why": "flythrough", "kind": "flythrough",
                 "step": i} for i in range(FLYTHROUGH_FRAMES)]
    return out[:MAX_FRAMES]


def _passage(part: dict) -> bool:
    """Does this part's type declare `PASSAGE`? A bad or absent type is not a gate."""
    t = part.get("type")
    f = os.path.join(_pipeline.ROOT, "types", f"{t}.py") if t else None
    try:
        return bool(f and os.path.exists(f) and _pipeline.load_type(f)["passage"])
    except Exception:                        # noqa: BLE001 -- a bad type is not a gate
        return False


def arrival_gate(plan: dict, spec: dict | None, parts: list | None = None):
    """The part a place is entered by. **One rule, both callers.** Demo-polish, 1d.

        A passage point standing on the outermost ring where the spec names rings
        (`placeread.rings`), else the first part in plan order whose type declares a
        passage -- which is every open town's and walled town's answer, unchanged. Until
        this `_gate_threshold` took the first passage part in plan order on the stated
        invariant "at most one way in", which is a walled town's and not a city's, while
        `flythrough_path` preferred the outermost ring's gate: the demo's arrival frame
        stood outside whichever gate sorted first. Returns the plan leaf, or None.
        
    """
    from .. import placeread
    parts = _pipeline.plan_parts(plan) if parts is None else parts
    gates = [p for p in parts if p.get("kind") == "point" and _passage(p)]
    ring = (placeread.rings(spec or {"defining_parts": []}, parts) or [None])[0]
    if ring is not None:
        from ..placeplan import _edge_cells
        cells = set(_edge_cells(ring["part"]))
        on = [g for g in gates if g.get("at")
              and (int(g["at"][0]), int(g["at"][-1])) in cells]
        if on:
            return sorted(on, key=lambda p: p["name"])[0]
    return gates[0] if gates else None


def flythrough_path(plan: dict, spec: dict | None, site: dict,
                    steps: int = FLYTHROUGH_FRAMES) -> list:
    """The camera path: from outside the outermost gate, in, to the thing at the centre.

        Deterministic, from the plan's own geometry and no camera list. A place read that can
        say which ring is outermost (`placeread.rings`) can say which gate a person arrives
        at, and the centre is the defining part the spec centres the place on.
        
    """
    from .. import placeread, render
    parts = _pipeline.plan_parts(plan)
    X, Z, S = site["origin"][0], site["origin"][1], int(site["size"])
    end = (X + S / 2.0, Z + S / 2.0)
    rect = None
    compound = False
    from ..placeplan import compound_rects
    comps = compound_rects(plan)
    for d in (spec or {}).get("defining_parts", []):
        if d["relation"] != "centre":
            continue
        # A compound at the centre ends the move over the whole of it, not over
        # whichever of its halls sorts first.
        mine = [r for n, r in comps.items()
                if n == d["name"] or n.startswith(d["name"] + "_")]
        if mine:
            rect = mine[0]
            compound = True
            end = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
            break
        for p in placeread._matching(parts, d):
            rect = _pipeline.part_rect({**p, "name": p.get("name")})
            end = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
            break
    g = arrival_gate(plan, spec, parts)
    if g is None:
        # no passage part: any point with a position, by name, as before
        pts = sorted((p for p in parts if p.get("kind") == "point" and p.get("at")),
                     key=lambda p: p["name"])
        g = pts[0] if pts else None
    start = ((float(g["at"][0]), float(g["at"][-1])) if g is not None and g.get("at")
             else (X + S / 2.0, float(Z)))
    # Begin outside the gate, on the line the road comes in on, and end **short of** the
    # centre: the subject's half-span along the approach plus `FLY_STANDOFF`, the look
    # held on the centre. See `render.FLY_STANDOFF`.
    dx, dz = end[0] - start[0], end[1] - start[1]
    n = max(1.0, (dx * dx + dz * dz) ** 0.5)
    ux, uz = dx / n, dz / n
    half = 0.0
    if rect is not None:
        half = ((rect[2] - rect[0]) / 2.0 if abs(ux) >= abs(uz)
                else (rect[3] - rect[1]) / 2.0)
    stop = min(n - 1.0, half + render.FLY_STANDOFF)
    last = (end[0] - stop * ux, end[1] - stop * uz)
    out0 = (start[0] - 48 * ux, start[1] - 48 * uz)
    # and, for a compound, the depth its own wall stands at inside that edge -- and
    # lifts the final camera above it.
    subject = ({"subject": [int(v) for v in rect], "compound": bool(compound)}
               if rect is not None else {})
    return [{"i": i,
             "at": [round(out0[0] + (last[0] - out0[0]) * i / (steps - 1), 2),
                    round(out0[1] + (last[1] - out0[1]) * i / (steps - 1), 2)],
             "look": [round(end[0], 2), round(end[1], 2)], **subject}
            for i in range(steps)] if steps > 1 else \
        [{"i": 0, "at": [round(out0[0], 2), round(out0[1], 2)],
          "look": [round(end[0], 2), round(end[1], 2)], **subject}]


def stage_cards(rnd: Round, be, results: dict) -> dict:
    out = {}
    for spec in rnd.cards:
        frames = _resolve(spec["frames"])
        out_dir = _resolve(spec.get("out", spec["frames"]))
        made, refused = [], {}
        for tag in spec["tags"]:
            p = card_mod.from_dir(
                frames, tag, os.path.join(out_dir, f"{tag}_card.png"),
                layout=spec.get("layout", "quad"),
                size=tuple(spec["size"]) if spec.get("size") else None,
                strict=spec.get("strict", True),
                blank_keys=tuple(spec.get("blank_keys", ("eye",))))
            (made.append(p) if p else
             refused.__setitem__(tag, card_mod.DAMAGE.get(tag)))
        out[spec["name"]] = {
            "frames": frames, "composed": len(made), "refused": refused,
            # The refusal has to travel. A card refused this run may still be on disk
            # from a run that composed it leniently, and a judge stage that resolved
            # paths off the filesystem would quietly judge the stale one -- which is
            # exactly the "nobody looked" failure the refusal exists to stop.
            "refused_paths": [os.path.join(out_dir, f"{t}_card.png") for t in refused],
        }
    return out


def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_pipeline.ROOT, p)


def stage_judge(rnd: Round, be, results: dict) -> dict:
    """A judgement is a question, a list of (label, card A, card B), and the rule that
        was registered before anything was rendered. Warm, this makes zero model calls and
        returns the identical verdicts; cold, it stages every missing judgement to one file
        so an agent answers them in a single fan-out.
        
    """
    import hashlib
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    refused = {}
    for cset in (results.get("cards") or {}).values():
        if isinstance(cset, dict):
            for t, why in (cset.get("refused") or {}).items():
                for rp in cset.get("refused_paths", []):
                    if os.path.basename(rp) == f"{t}_card.png":
                        refused[os.path.abspath(rp)] = why
    out = {}
    for spec in rnd.judgements:
        js = verdicts.Session(spec["question"], stage_name=spec.get(
            "stage_name", spec["name"]))
        rows = []
        for pair in spec["pairs"]:
            a, b = _resolve(pair["a"]), _resolve(pair["b"])
            row = {"label": pair.get("label", ""), "a": pair["a"], "b": pair["b"]}
            if not (os.path.exists(a) and os.path.exists(b)):
                row["result"] = "no_cards"
            elif os.path.abspath(a) in refused or os.path.abspath(b) in refused:
                row.update(result="no_cards",
                           damage=refused.get(os.path.abspath(a))
                           or refused.get(os.path.abspath(b)))
            elif spec.get("skip_identical") and sha(a) == sha(b):
                # Two byte-identical images key the same judgement in both orientations,
                # so the pair can only ever tie -- structurally, at the cost of a model
                # call. That is the *previewer's* blindness rather than the judge's, and
                # E1a registered it as reported-and-excluded before anything was judged.
                row.update(result="identical_render", identical_render=True)
            else:
                res = js.compare(a, b)
                if res is not None:
                    row["result"] = res
            rows.append(row)
        if js.needed:
            staged = rnd.rel(f"{spec['name']}_requests.json")
            js.stage(staged)
            out[spec["name"]] = {"status": "needs_model", "role": "judge",
                                 "requests": len(js.needed), "staged": staged,
                                 "stage_name": js.stage_name}
            continue
        res = {"status": "done", "question": spec["question"], "rows": rows,
               "identical_render": sum(1 for r in rows
                                       if r.get("identical_render"))}
        if spec.get("score", "floor") == "winners":
            # E3 and step 3 do not have a correct side: each pair names its own winner
            # and the pre-registered reading is about which arm won, not a rate.
            res["winners"] = {r["label"]: r.get("result") for r in rows}
            # The tie rate still travels. A round-robin over siblings from one brief is
            # scored this way and its *tie* rate is the headline, not a side note.
            seen = [r.get("result") for r in rows
                    if r.get("result") in ("a", "b", "tie")]
            res["judged"] = len(seen)
            res["ties"] = sum(1 for r in seen if r == "tie")
            res["tie_rate"] = (round(res["ties"] / len(seen), 3) if seen else None)
        else:
            # A pair may name its own correct side: E-craft's blind picks were recorded
            # per structure before any judgement, and three of the four are the no-
            # library arm.
            default = spec.get("correct", "a")
            summary = verdicts.tally(
                [r.get("result") for r in rows],
                correct=[p.get("correct", default) for p in spec["pairs"]])
            floor = spec.get("floor")
            if floor:
                summary["pass"] = verdicts.meets(summary, floor)
                summary["floor"] = floor
            res.update(summary)
            record("judgement", name=spec["name"], settlement=rnd.name,
                   judged=summary["judged"], correct=summary["correct"],
                   wrong=summary["wrong"], ties=summary["ties"])
        out[spec["name"]] = res

    # Cross-judgement agreement -- check 3's actual statistic. Two instruments are
    # compared on the same pairs, and the number that matters is how often they say the
    # same thing, not how often either says "a". Reported both over every pair and over
    # the pairs where both were decisive, because the step-4 finding was that the cheap
    # instrument abstains rather than disagrees.
    for spec in rnd.judgements:
        other = spec.get("agreement_with")
        if not other or out.get(spec["name"], {}).get("status") != "done":
            continue
        mine = {r["label"]: r.get("result") for r in out[spec["name"]]["rows"]}
        theirs = {r["label"]: r.get("result") for r in out[other]["rows"]}
        both = [k for k in mine
                if mine[k] in ("a", "b", "tie") and theirs.get(k) in ("a", "b", "tie")]
        dec = [k for k in both if mine[k] != "tie" and theirs[k] != "tie"]
        agree = sum(1 for k in both if mine[k] == theirs[k])
        out[spec["name"]]["agreement"] = {
            "with": other, "n": len(both), "agree": agree,
            "rate": round(agree / len(both), 3) if both else None,
            "decisive_both": len(dec),
            "rate_decisive": (round(sum(1 for k in dec if mine[k] == theirs[k])
                                    / len(dec), 3) if dec else None),
        }
    return out


def stage_write(rnd: Round, be, results: dict) -> dict:
    """**Part R, and it exists because of a premise this project got wrong.** `stage_render` loads `world_built.npz` only to place its cameras; Chunky
        renders the Minecraft save at `run/server/world`. A dry run leaves the city in a
        volume and writes no block anywhere, so "the city" was for a long time a thing no
        instrument could photograph and no person could look at -- and every appearance
        claim about it was a claim about a number. The editor has always been able to write
        a volume's blocks to a server; that is how a standing settlement is reset and
        restored around a candidate render. So appearance costs **one server run** and not
        an offline exporter: the built volume written in, and the shot list and the
        flythrough shot off the saved world.

        What is written is the difference between the round's own pre-build cache and its
        `world_built.npz`, which is every block the build decided and nothing else, and the
        world is snapshotted first -- ground work is not idempotent and this is the undo. The live world is verified against the pre-build cache on a
        sample before anything is written: writing a city onto ground that is not the
        ground it was planned on would be a worse failure than not writing it.

        This is the live write when the bars hold and a rehearsal when they do not. It
        decides neither; it writes what it is given and says how much.
        
    """
    from .. import offline, stages, world as world_mod
    from ..buildlib import Builder
    if not be.live:
        return {"skipped": "stage_write writes blocks; run it on a live backend, "
                           "inside scripts/mcrun.sh"}
    built_p = rnd.rel("world_built.npz")
    if not os.path.exists(built_p):
        return {"error": f"no {built_p} -- there is nothing built to write"}
    spec = dict(rnd.shots or {})
    # **The world as the server has it, which is not the round's base volume.** A dry
    # run cuts its plateau into the cached volume and emits its lanes into it, and saves
    # both over `world.npz` -- so the base volume is 43,255 cells ahead of the ground
    # the server is standing on, and a write of `world.npz -> world_built.npz` would
    # leave the city on unlevelled ground with no streets in it. What the world was when
    # it was read is `world.before-plateau.npz`, and the difference between that and the
    # built volume is every block this place is made of, its ground work and its
    # circulation included.
    base_p = rnd.rel("world.before-plateau.npz")
    pre = offline.load_volume(base_p) if os.path.exists(base_p) else rnd.volume()
    built = offline.load_volume(built_p)
    s = rnd.site or json.load(open(rnd.rel("site.json")))
    X, Z, S = s["origin"][0], s["origin"][1], s["size"]
    m = int(spec.get("write_margin", 24))
    x0, z0 = max(X - m, pre.x0, built.x0), max(Z - m, pre.z0, built.z0)
    x1 = min(X + S - 1 + m, pre.x0 + pre.shape[0] - 1, built.x0 + built.shape[0] - 1)
    z1 = min(Z + S - 1 + m, pre.z0 + pre.shape[2] - 1, built.z0 + built.shape[2] - 1)
    t0 = time.perf_counter()
    diff = _pipeline.region_diff(pre, built, x0, z0, x1, z1)
    print(f"  region ({x0},{z0})..({x1},{z1}), {len(diff)} cells to write "
          f"({time.perf_counter() - t0:.0f}s to read)", flush=True)
    if not diff:
        return {"written": 0, "note": "the built volume and the pre-build cache agree; "
                                      "nothing to write"}

    import numpy as np
    ed = be.editor
    rng = np.random.default_rng(7)
    n = min(int(spec.get("verify_samples", 40)), len(diff))
    bad, checked = 0, []
    for i in rng.choice(len(diff), size=n, replace=False):
        x, y, z, want, _ = diff[i]
        got = ed.getBlock((x, y, z)).id.split(":")[-1]
        if got != want.split("[")[0]:
            bad += 1
            checked.append({"at": [x, y, z], "cache": want, "world": got})
    if bad > int(spec.get("verify_max_bad", 2)):
        return {"error": f"the live world does not match this round's pre-build cache "
                         f"({bad}/{n} mismatches) -- nothing was written",
                "mismatches": checked[:8]}
    print(f"  ground verified against the pre-build cache ({bad}/{n} mismatches)",
          flush=True)
    snap = world_mod.snapshot(spec.get("snapshot", f"{rnd.name}.before-write"))
    print(f"  snapshot {snap}", flush=True)

    t1 = time.perf_counter()
    b = Builder(be.site)
    for (x, y, z, _, state) in diff:
        b.place_block(x, y, z, state)
    placed = b.flush()
    ed.runCommand("save-all flush")
    time.sleep(8)
    be.refresh()
    secs = round(time.perf_counter() - t1, 1)
    print(f"  {placed.get('placed')} placed, {placed.get('failed')} failed in {secs}s",
          flush=True)
    return {"written": len(diff), "placed": placed.get("placed"),
            "failed": placed.get("failed"), "seconds": secs,
            "region": [x0, z0, x1, z1], "snapshot": snap,
            "from": os.path.basename(base_p if os.path.exists(base_p)
                                     else rnd.base_volume),
            "verified": {"samples": n, "mismatches": bad},
            "note": "the built volume, written into run/server/world so Chunky can see "
                    "it. What this is a write *of* -- a city that holds every bar or "
                    "one that does not -- is the readout's answer and not this stage's"}


def stage_map(rnd: Round, be, results: dict) -> dict:
    """The plan as a picture: `<state>/plan_map.png`. v2, A6.

        Deterministic, offline, under a second, and it needs nothing but the plan -- so it
        can be asked for the moment `plan` has run and long before a block is placed, which
        is the point of it. `preview.plan_map` is the drawing; this is the stage.
        
    """
    import cv2
    from .. import preview as preview_mod
    plan = rnd.plan()
    if not plan:
        return {"error": "no plan.json: there is no plan to draw"}
    site = rnd.site or (json.load(open(rnd.rel("site.json")))
                        if os.path.exists(rnd.rel("site.json")) else None)
    scale = int((rnd.shots or {}).get("map_scale", 1))
    t0 = time.perf_counter()
    img = preview_mod.plan_map(plan, rnd.network(),
                               {"origin": site["origin"], "size": site["size"]}
                               if site else None, scale=scale)
    out = rnd.rel("plan_map.png")
    os.makedirs(rnd.state, exist_ok=True)
    cv2.imwrite(out, img[:, :, ::-1])
    secs = round(time.perf_counter() - t0, 2)
    record("preview", settlement=rnd.name, seconds=secs, shape=list(img.shape),
           out=os.path.basename(out))
    return {"map": out, "shape": list(img.shape), "scale": scale, "seconds": secs,
            "parts": len(_pipeline.plan_parts(plan)),
            "lane_cells": len((rnd.network().cells if rnd.network() else {}))}


def stage_sheet(rnd: Round, be, results: dict) -> dict:
    """A sheet of instances per type: `<state>/sheets/<type>.png`. v2, A6.

        The types are the round's own `types.list`, the fixture and the seeds are the ones
        its **checker** would use for a type of that kind (`blind._fixtures_for`,
        `_seeds_for`) -- a wall is not stood on a house's plot. A type is authored blind and
        read as a table of numbers; this is the first way to *look* at one that does not
        cost a town.
        
    """
    import cv2
    from .. import preview as preview_mod
    from . import blind
    t = rnd.types or {}
    specs = [s for s in (t.get("list") or []) if s.get("name")]
    # which is the ground it is actually going to stand on, and a better subject than a
    # fixture borrowed from another round.
    own: dict = {}
    if not specs and rnd.flags.get("types"):
        specs = [{"name": n} for n in rnd.flags["types"]]
        plots = [p for p in blind._plots_of(rnd) if p.get("kind", "plot") == "plot"]
        if plots:
            big = max(plots, key=lambda q: (q["x1"] - q["x0"]) * (q["z1"] - q["z0"]))
            own["plot"] = {"round": rnd.name, "plot": big["label"]}
        # ...and a wall, a gate and a square are not in a plot registry, so they are
        # stood on this place's own first part of their kind, by its geometry.
        for leaf in rnd.parts() or []:
            kind = leaf.get("kind", "plot")
            if kind in ("edge", "point", "area") and kind not in own:
                own[kind] = {"round": rnd.name, "kind": kind, "part": leaf["name"],
                             **{k: leaf[k] for k in _pipeline.PART_GEOMETRY
                                if k in leaf}}
    if not specs:
        return {"error": "`types.list` or `flags.types` is "
                         "where they are"}
    out_dir = rnd.rel("sheets")
    os.makedirs(out_dir, exist_ok=True)
    sheets, failed = {}, {}
    for spec in specs:
        name = spec["name"]
        t0 = time.perf_counter()
        try:
            decl = _pipeline.load_type(spec.get("file")
                                       or os.path.join(_pipeline.ROOT, "types",
                                                       f"{name}.py"))
            want = dict(spec, part=spec.get("part") or decl["kind"])
            fixtures = blind._fixtures_for(rnd, want) or (
                [own[want["part"]]] if own.get(want["part"]) else [])
            if not fixtures:
                failed[name] = (f"no fixture of kind {want['part']!r} in this round, "
                                f"and no plot of its own to stand it on")
                continue
            img = preview_mod.instances(
                name, fixtures[0], seeds=blind._seeds_for(rnd, want) or (21, 22, 23),
                voice=spec.get("voice") or rnd.voice_name() or None)
        except Exception as e:                   # noqa: BLE001 -- reported, not raised
            failed[name] = f"{type(e).__name__}: {e}"
            continue
        p = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(p, img[:, :, ::-1])
        sheets[name] = {"path": p, "shape": list(img.shape),
                        "fixture": f"{fixtures[0]['round']}/"
                                   f"{fixtures[0].get('plot') or fixtures[0].get('part')}",
                        "seconds": round(time.perf_counter() - t0, 2)}
    return {"sheets": sheets, "failed": failed}


def stage_render(rnd: Round, be, results: dict) -> dict:
    """The round's shot list. Needs Chunky; needs a server only if the world on disk
    is stale, which is what `--no-server` in `step4_render` has always meant."""
    from .. import render
    from ..circulate import Network
    spec = rnd.shots or {}
    built = offline.load_volume(rnd.rel("world_built.npz"))
    net = Network.load(rnd.rel("network.json"))
    sight = render.Sightline(built)
    frames_dir = rnd.rel(*spec.get("frames", ("step4_shots", "frames")))
    os.makedirs(frames_dir, exist_ok=True)
    geom = getattr(render, spec.get("geometry", "E1D"))
    prefix = spec.get("prefix", "r")
    report = {}
    # A4: which buildings get a card is a budget now, and at a city's four hundred it is
    # a *sample* of them. `None` -- a round with no budget and no place spec -- is every
    # plot, which is what every round before this one did.
    plan = rnd.plan()
    budget = render_budget(plan, rnd.place_spec(), spec) if plan else None
    wanted = None
    budget_rec = None
    if budget is not None:
        rows = budget_frames(plan, rnd.place_spec(), budget)
        pre = preflight_render(rows, budget,
                               bound=float(spec.get("render_bound_seconds")
                                           or RENDER_BOUND_S))
        if pre["refused"]:
            return {"status": "error", "stop": True, "preflight": pre,
                    "error": f"the render preflight refused: about "
                             f"{pre['estimated_seconds']}s for {pre['frames']} frames "
                             f"against {pre['bound_seconds']}s"}
        wanted = {f["name"] for f in rows if f["why"] in ("defining", "sample")}
        budget_rec = {
            "budget": budget, "frames": len(rows), "preflight": pre,
            "of_plots": len([p for p in _pipeline.plan_parts(plan)
                             if p.get("kind", "plot") == "plot"]),
            "by_why": {w: len([f for f in rows if f["why"] == w])
                       for w in ("place", "compound", "defining", "sample",
                                 "flythrough")},
            "note": "A4: a card per building at 400 structures is six hours of Chunky, "
                    "so the ordinary buildings are sampled by seed and every defining "
                    "part is photographed"}
    merged = render.merged_plots(rnd.state)
    gates_framed = {}
    for p in merged:
        if wanted is not None and p["label"] not in wanted:
            continue
        tag = f"{prefix}_{p['label']}"
        centre, span = render.plot_subject(p, render.mid_y(built, p))
        threshold = net.threshold(p["label"]) if net else None
        # A ladder, not one camera: a panel that comes back black is re-shot from
        # further out before the strict card guard is allowed to refuse the card. The
        # first rung is the validated camera, so nothing that already works moves. **A
        # gate in a great wall is framed on the wall**, 1d: see `gate_subject`.
        gate = None
        if p.get("passage") and p.get("kind") == "point":
            base = threshold.y if threshold is not None else p.get("y0")
            if base is None:
                base = render.surrounding_floor(built, p)
            gate = render.gate_subject(p, render.edge_of_point(p, merged), built,
                                       int(base), threshold=threshold)
        if gate is not None:
            shots = render.gate_card_ladder(gate, threshold, sight=sight, geom=geom)
            gates_framed[p["label"]] = {k: v for k, v in gate.items()
                                        if k != "centre"} | {
                "centre": list(gate["centre"]), "own_span": span}
            reach = int(gate["span"] * 1.2) + geom.pad
            chunks = render.chunk_list(int(gate["centre"][0]) - reach,
                                       int(gate["centre"][2]) - reach,
                                       int(gate["centre"][0]) + reach,
                                       int(gate["centre"][2]) + reach)
        else:
            shots = render.card_shot_ladder(centre, span, threshold=threshold,
                                            sight=sight, geom=geom)
            chunks = render.chunk_list(p["x0"] - geom.pad, p["z0"] - geom.pad,
                                       p["x1"] + geom.pad, p["z1"] + geom.pad)
        report[tag] = render.shoot_all(
            shots, chunks, frames_dir, tag=tag, scene_prefix=f"{rnd.name}_",
            spp=spec.get("spp", 40),
            size=tuple(spec.get("size", (900, 560))), reuse=True)
    if budget and budget.get("compound"):
        # Each compound as one subject, framed on its whole rectangle with the same four
        # cameras a building gets, and its eye-level frame from the doorstep of its gate
        # where the network reserved one.
        for f in (rows if budget is not None else []):
            if f["why"] != "compound":
                continue
            r = f["rect"]
            box = {"label": f["name"], "x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3]}
            tag = f"{prefix}_{f['name']}"
            centre, span = render.plot_subject(box, render.mid_y(built, box), margin=16)
            gate = None
            if net is not None:
                gates = [p for p in _pipeline.plan_parts(plan)
                         if p.get("compound") == f["name"] and p.get("kind") == "point"]
                for g in sorted(gates, key=lambda p: p["name"]):
                    gate = net.threshold(g["name"])
                    if gate is not None:
                        break
            shots = render.card_shot_ladder(centre, span, threshold=gate, sight=sight,
                                            geom=geom)
            chunks = render.chunk_list(r[0] - geom.pad, r[1] - geom.pad,
                                       r[2] + geom.pad, r[3] + geom.pad)
            report[tag] = render.shoot_all(
                shots, chunks, frames_dir, tag=tag, scene_prefix=f"{rnd.name}_",
                spp=spec.get("spp", 40), size=tuple(spec.get("size", (900, 560))),
                reuse=True)
    if spec.get("place"):
        # A7: the two frames a *place* is photographed with, beside the four each
        # building gets. Opt-in per round, because every round before this one was a set
        # of buildings and photographing it as one thing would be a claim.
        s = rnd.site or json.load(open(rnd.rel("site.json")))
        X, Z, S = s["origin"][0], s["origin"][1], s["size"]
        cy = render.mid_y(built, {"x0": X, "z0": Z, "x1": X + S - 1, "z1": Z + S - 1})
        centre = (X + S / 2, cy, Z + S / 2)
        gate = _pipeline._gate_threshold(rnd, net)
        # The distance was a constant chosen when the tallest thing a type could build
        # was a wall of twenty, and a great wall of forty-eight filled the frame from
        # the ground to the sky. Measured off the built world, because what stands in
        # front of the camera is a fact about the world and not about the plan.
        place_shots = render.place_card_shots(
            centre, S, gate=gate, sight=sight,
            bearing=render.approach_bearing(net, centre),
            rise=render.approach_rise(built, gate))
        # **The frame's chunks, not the site's** (1d, finding 9): each whole-place
        # camera loads the site plus its own stand-off, so the aerial does not show the
        # site as a slab floating over nothing. A chunk the save never made renders as a
        # grey slab, so the list is clipped to finished chunks and a camera standing
        # over an unmade one is dollied in along its line of sight until it stands over
        # ground that exists.
        report["place"] = {}
        for key, shot in place_shots.items():
            chunks, dropped = render.finished_chunks(render.place_chunks(X, Z, S, {key: shot}))
            have = {(c[0], c[1]) for c in chunks}
            rungs = list(shot) if isinstance(shot, (list, tuple)) else [shot]
            rungs = [render.onto_finished(r, have) for r in rungs]
            if dropped:
                print(f"  {key}: {len(dropped)} chunk(s) the world never made left "
                      f"out of the render", flush=True)
            report["place"].update(render.shoot_all(
                {key: rungs if len(rungs) > 1 else rungs[0]}, chunks,
                frames_dir, tag=f"{prefix}_place", scene_prefix=f"{rnd.name}_",
                spp=spec.get("spp", 40), size=(1200, 700), reuse=True))
    if budget and budget.get("flythrough"):
        # A4: the camera move from outside the outer gate to the palace, as frames. The
        # path is written whether or not the frames are shot, because the path is a fact
        # about the plan and the frames are an hour of Chunky.
        s = rnd.site or json.load(open(rnd.rel("site.json")))
        path = flythrough_path(plan, rnd.place_spec(), s)
        json.dump(path, open(rnd.rel("flythrough.json"), "w"), indent=1)
        report["flythrough"] = render.shoot_flythrough(
            path, built, frames_dir, tag=f"{prefix}_fly",
            scene_prefix=f"{rnd.name}_", spp=spec.get("spp", 40),
            size=tuple(spec.get("fly_size", (960, 540))), sight=sight,
            skip=not spec.get("fly", True))
    return {**render.shot_summary(report), "shots": report,
            **({"budget": budget_rec} if budget_rec else {}),
            **({"gates_framed_on_wall": gates_framed} if gates_framed else {})}


def stage_candidate_render(rnd: Round, be, results: dict) -> dict:
    """Every candidate photographed at ground truth, one at a time, on the real site.

    Live only, and the whole of it is the protocol for borrowing a standing settlement
    without damaging it -- step 3's readout, generalised out of the script it was
    written in:

        verify the live world against the built cache on a sample of the diff
        snapshot
        reset the wave's region to its pre-build state from that diff
        per candidate: execute offline, write the pending set, flush, shoot, revert
        write the built state back"""
    from .. import offline, render, settlement as settlement_mod, stages
    from ..buildlib import Builder
    c = rnd.candidates
    if not c:
        return {"note": "this round has no candidates to render"}
    if not be.live:
        return {"skipped": "candidate_render writes blocks; run it on a live backend "
                           "inside an mcrun.sh session"}
    spec = dict(c.get("shots") or {})
    frames_dir = os.path.join(_pipeline._cand_dir(rnd), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pre = rnd.volume()
    built = offline.load_volume(rnd.rel("world_built.npz"))
    net = rnd.network()
    ps = _pipeline._cand_plots(rnd)
    m = int(spec.get("margin", 8))
    x0 = min(p["x0"] for p in ps) - m
    x1 = max(p["x1"] for p in ps) + m
    z0 = min(p["z0"] for p in ps) - m
    z1 = max(p["z1"] for p in ps) + m
    diff = _pipeline.region_diff(pre, built, x0, z0, x1, z1)
    print(f"  region ({x0},{z0})..({x1},{z1}), {len(diff)} diff cells", flush=True)

    ed, site = be.editor, be.site
    import numpy as np
    rng = np.random.default_rng(7)
    n = min(int(spec.get("verify_samples", 40)), len(diff))
    bad = 0
    for i in rng.choice(len(diff), size=n, replace=False):
        x, y, z, _, want = diff[i]
        got = ed.getBlock((x, y, z)).id.split(":")[-1]
        bad += got != want.split("[")[0]
    if bad > int(spec.get("verify_max_bad", 2)):
        return {"error": f"the live world does not match the built cache "
                         f"({bad}/{n} mismatches) -- nothing was written"}
    print(f"  baseline verified ({bad}/{n} mismatches)", flush=True)
    snap = settlement_mod.snapshot(spec.get("snapshot", f"{rnd.name}.candidates"))

    def write(cells):
        b = Builder(site)
        for (x, y, z, st) in cells:
            b.place_block(x, y, z, st)
        b.flush()
        ed.runCommand("save-all flush")
        time.sleep(4)

    def pre_state(x, y, z):
        lx, ly, lz = x - pre.x0, y - pre.y0, z - pre.z0
        if (0 <= lx < pre.shape[0] and 0 <= ly < pre.shape[1]
                and 0 <= lz < pre.shape[2]):
            return pre.palette[pre.codes[lx, ly, lz]]
        return "air"

    geom = getattr(render, spec.get("geometry", "STEP3"))
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    cy = float(spec.get("cy", 76))
    span = max(x1 - x0, z1 - z0) / 2 + m
    threshold = net.threshold(spec["door"]) if (net and spec.get("door")) else None
    chunks = render.chunk_list(int(cx - span - geom.pad), int(cz - span - geom.pad),
                               int(cx + span + geom.pad), int(cz + span + geom.pad))

    report, written = {}, {}
    write([(x, y, z, p) for (x, y, z, p, _) in diff])
    print("  region reset to pre-build", flush=True)
    try:
        for cid in _pipeline._cand_ids(rnd):
            prog = _pipeline._cand_program(rnd, cid)
            if not os.path.exists(prog):
                report[cid] = {"status": "no_program"}
                continue
            pend = be.execute(prog, pre)._pending
            write([(x, y, z, st) for (x, y, z), st in pend.items()])
            written[cid] = len(pend)
            print(f"  {cid}: {len(pend)} blocks written", flush=True)
            sight = render.Sightline(stages.apply_pending(pre, pend))
            shots = render.building_card_shots((cx, cy, cz), span,
                                               threshold=threshold, sight=sight,
                                               geom=geom)
            tag = f"{spec.get('prefix', 'sel')}_{cid}"
            report[tag] = render.shoot_all(
                shots, chunks, frames_dir, tag=tag,
                scene_prefix=spec.get("scene_prefix", f"{rnd.name}_sel_"),
                spp=int(spec.get("spp", 40)),
                size=tuple(spec.get("size", (900, 560))), reuse=True)
            write([(x, y, z, pre_state(x, y, z)) for (x, y, z) in pend])
            record("pass_run", name=f"candidate_render/{cid}", settlement=rnd.name,
                   placed=len(pend), failed=0, seconds=None)
    finally:
        write([(x, y, z, b2) for (x, y, z, _, b2) in diff])
        print("  standing settlement restored", flush=True)
    return {**render.shot_summary({k: v for k, v in report.items()
                                   if isinstance(v, dict) and "status" not in v}),
            "snapshot": snap, "blocks_written": written,
            "frames": frames_dir, "shots": report}


def stage_type_render(rnd: Round, be, results: dict) -> dict:
    """Every instance photographed at ground truth, on the real site, plot by plot.

    `stage_candidate_render`'s protocol, run once per plot instead of once per round,
    because a type round's instances are spread across a standing settlement and the
    bounding box of all four plots is the whole town. Per plot:

        verify the live world against the built cache on a sample of the diff
        snapshot
        reset that plot's region to its pre-build state from the diff
        per instance on it: execute offline, write, flush, shoot, revert
        write the built state back"""
    from .. import offline, render, settlement as settlement_mod, stages
    from ..buildlib import Builder
    if not rnd.types:
        return {"note": "this round has no types to render"}
    if not be.live:
        return {"skipped": "type_render writes blocks; run it on a live backend "
                           "inside an mcrun.sh session"}
    spec = dict(rnd.types.get("shots") or {})
    frames_dir = os.path.join(_pipeline._type_out(rnd), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pre = rnd.volume()
    built = offline.load_volume(rnd.rel("world_built.npz"))
    net = rnd.network()
    plots = {p["label"]: p for p in json.load(open(rnd.rel("plots.json")))}
    by_plot: dict = {}
    for t in _pipeline._types(rnd):
        for (plot, seed, params, iid) in _pipeline._type_rows(rnd, t):
            by_plot.setdefault(plot["label"], []).append((t, plot, seed, iid))

    ed, site = be.editor, be.site
    geom = getattr(render, spec.get("geometry", "STEP3"))
    m = int(spec.get("margin", 8))
    report, written, verified = {}, {}, {}
    snap = settlement_mod.snapshot(spec.get("snapshot", f"{rnd.name}.types"))

    def write(cells):
        b = Builder(site)
        for (x, y, z, st) in cells:
            b.place_block(x, y, z, st)
        b.flush()
        ed.runCommand("save-all flush")
        time.sleep(4)

    def pre_state(x, y, z):
        lx, ly, lz = x - pre.x0, y - pre.y0, z - pre.z0
        if (0 <= lx < pre.shape[0] and 0 <= ly < pre.shape[1]
                and 0 <= lz < pre.shape[2]):
            return pre.palette[pre.codes[lx, ly, lz]]
        return "air"

    for label in sorted(by_plot):
        p = plots[label]
        x0, z0 = p["x0"] - m, p["z0"] - m
        x1, z1 = p["x1"] + m, p["z1"] + m
        diff = _pipeline.region_diff(pre, built, x0, z0, x1, z1)
        print(f"  {label}: region ({x0},{z0})..({x1},{z1}), {len(diff)} diff cells",
              flush=True)
        import numpy as np
        rng = np.random.default_rng(7)
        n = min(int(spec.get("verify_samples", 40)), len(diff))
        bad = 0
        for i in rng.choice(len(diff), size=n, replace=False):
            x, y, z, _, want = diff[i]
            got = ed.getBlock((x, y, z)).id.split(":")[-1]
            bad += got != want.split("[")[0]
        verified[label] = {"samples": n, "mismatches": int(bad)}
        if bad > int(spec.get("verify_max_bad", 2)):
            report[label] = {"error": f"the live world does not match the built cache "
                                      f"({bad}/{n}) -- this plot was not touched"}
            continue
        print(f"  {label}: baseline verified ({bad}/{n})", flush=True)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        span = max(x1 - x0, z1 - z0) / 2 + m
        threshold = net.threshold(label) if net else None
        chunks = render.chunk_list(int(cx - span - geom.pad),
                                   int(cz - span - geom.pad),
                                   int(cx + span + geom.pad),
                                   int(cz + span + geom.pad))
        write([(x, y, z, ps) for (x, y, z, ps, _) in diff])
        print(f"  {label}: reset to pre-build", flush=True)
        try:
            for (t, plot, seed, iid) in by_plot[label]:
                prog = _pipeline._instance_program(rnd, t, plot, seed)
                if not os.path.exists(prog):
                    report[iid] = {"status": "no_program"}
                    continue
                pend = be.execute(prog, pre)._pending
                write([(x, y, z, st) for (x, y, z), st in pend.items()])
                written[iid] = len(pend)
                cy = float(spec.get("cy") or render.mid_y(
                    stages.apply_pending(pre, pend), plot))
                sight = render.Sightline(stages.apply_pending(pre, pend))
                shots = render.building_card_shots((cx, cy, cz), span,
                                                   threshold=threshold, sight=sight,
                                                   geom=geom)
                tag = f"{spec.get('prefix', 'types_a')}_{t['name']}_{label}"
                report[tag] = render.shoot_all(
                    shots, chunks, frames_dir, tag=tag,
                    scene_prefix=spec.get("scene_prefix", f"{rnd.name}_r13_"),
                    spp=int(spec.get("spp", 40)),
                    size=tuple(spec.get("size", (900, 560))), reuse=True)
                write([(x, y, z, pre_state(x, y, z)) for (x, y, z) in pend])
                record("pass_run", name=f"type_render/{iid}", settlement=rnd.name,
                       placed=len(pend), failed=0, seconds=None)
        finally:
            write([(x, y, z, b2) for (x, y, z, _, b2) in diff])
            print(f"  {label}: standing settlement restored", flush=True)
    return {**render.shot_summary({k: v for k, v in report.items()
                                   if isinstance(v, dict) and "status" not in v
                                   and "error" not in v}),
            "snapshot": snap, "blocks_written": written, "verified": verified,
            "frames": frames_dir, "shots": report}


def _avg_ranks(vals: dict, higher_is_better: bool) -> dict:
    """Average ranks, 1 = best, ties sharing the mean of the places they span."""
    srt = sorted(((-v if higher_is_better else v), c) for c, v in vals.items())
    out, i = {}, 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1][0] == srt[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[srt[k][1]] = r
        i = j + 1
    return out


def _pair(rows: list, a: str, b: str):
    """The verdict for a specific ordered pair of candidates, as the judge returned it
    ("a" means the label's left-hand candidate won)."""
    for r in rows:
        if r.get("label") == f"{a}_vs_{b}":
            return r.get("result")
        if r.get("label") == f"{b}_vs_{a}":
            return {"a": "b", "b": "a"}.get(r.get("result"), r.get("result"))
    return None


def stage_selection(rnd: Round, be, results: dict) -> dict:
    """Read the round-robin against the rule that was registered before it ran.

        Three questions in the order the spec puts them, and the order matters:

          1. **the tie rate over same-brief siblings** -- the cheapest thing to find out and
             the one that can end the experiment. No discrimination, no selection.
          2. **the winner**, by the registered Copeland rule, tie-broken the way the
             bracket has always broken ties: deterministically, and marked as a tie-break
             rather than as a preference.
          3. **convergent validity** -- does that winner also rank above the median sibling
             on the measures the judge never saw? If it does, selection is picking better
             buildings. If it does not, selection is optimising photogeny over substance,
             and that is the most valuable thing this experiment could find.

        Nothing here computes a threshold. Every bar is read off the round's own
        pre-registration, which is written to `round.<config>.json` on the first run and
        never rewritten.
        
    """
    c = rnd.candidates
    if not c:
        return {"note": "this round has no candidates to select between"}
    prereg = (json.load(open(_pipeline._report_path(rnd))).get("preregistered")
              if os.path.exists(_pipeline._report_path(rnd)) else None) or rnd.preregistered
    name = c.get("judgement", "selection_rr")
    jr = (results.get("judge") or {}).get(name)
    if not jr or jr.get("status") != "done":
        return {"status": "no_judgement",
                "note": f"judgement {name!r} is not done: "
                        f"{(jr or {}).get('status', 'not run')}"}
    ids = _pipeline._cand_ids(rnd)
    rows = jr["rows"]

    score = {k: 0.0 for k in ids}
    beat = {k: [] for k in ids}
    ties = judged = 0
    for r in rows:
        res = r.get("result")
        if res not in ("a", "b", "tie"):
            continue
        lo, hi = r["label"].split("_vs_")
        judged += 1
        if res == "tie":
            ties += 1
            score[lo] += 0.5
            score[hi] += 0.5
        else:
            win, lose = (lo, hi) if res == "a" else (hi, lo)
            score[win] += 1.0
            beat[win].append(lose)
    tie_rate = round(ties / judged, 3) if judged else None
    bar = float((prereg.get("stage1") or {}).get("tie_rate_bar", 0.50))
    discriminates = tie_rate is not None and tie_rate < bar

    best = max(score.values())
    top = [k for k in ids if score[k] == best]
    if len(top) == 1:
        winner, how = top[0], "outright"
    else:
        sub = {k: sum(1 for o in top if o in beat[k]) for k in top}
        allowed = all(_pair(rows, a, b) in ("a", "b")
                      for i, a in enumerate(top) for b in top[i + 1:])
        ordered = sorted(top, key=lambda k: (-sub[k], ids.index(k)))
        if allowed and len(set(sub.values())) == len(top):
            winner, how = ordered[0], "head_to_head"
        else:
            winner, how = sorted(top, key=ids.index)[0], "index_tie_break"

    out = {"tie_rate": tie_rate, "ties": ties, "judged": judged,
           "tie_rate_bar": bar,
           "wilson95_tie_rate": verdicts.wilson(ties, judged) if judged else None,
           "copeland": score, "beat": beat, "winner": winner, "winner_by": how,
           "discriminates": discriminates,
           "stage1_reading": ("the judge separates same-brief siblings"
                              if discriminates else
                              "the judge cannot separate work from one brief -- "
                              "selection is not available as a lever")}
    record("selection", name=name, settlement=rnd.name, judged=judged, ties=ties,
           tie_rate=tie_rate, winner=winner, discriminates=discriminates)
    if not discriminates:
        out["stopped_at"] = "stage 1, per the pre-registration"
        return _write_selection(rnd, out)

    meas = (results.get("measures") or {})
    have = {k: meas.get(k) for k in ids
            if isinstance(meas.get(k), dict) and meas[k].get("status") == "measured"}
    if len(have) != len(ids):
        out["convergent"] = {"status": "no_measures",
                             "note": "run --stage measures first; the unseen "
                                     "measures must exist before a verdict is read "
                                     "against them"}
        return _write_selection(rnd, out)

    directed = (("lint_errors", False), ("walk_pct", True),
                ("variety_ridge_cv", True))
    ranks, above, below = {}, 0, 0
    for key, higher in directed:
        vals = {k: (have[k].get(key) if have[k].get(key) is not None else
                    (0.0 if higher else float("inf"))) for k in ids}
        r = _avg_ranks(vals, higher)
        ranks[key] = {"values": {k: have[k].get(key) for k in ids},
                      "ranks": r, "winner_rank": r[winner],
                      "higher_is_better": higher,
                      "winner_above_median": r[winner] < 2.5}
        above += r[winner] < 2.5
        below += r[winner] > 2.5
    verdict = ("convergent" if above >= 2 else
               "goodhart" if below >= 2 else "inconclusive")
    out["convergent"] = {
        "measures": ranks, "above_median": above, "below_median": below,
        "verdict": verdict,
        "reading": {
            "convergent": "the judge's winner is also better by instruments that "
                          "never saw an image -- selection picks better buildings",
            "goodhart": "the judge's winner is worse by the unseen instruments -- "
                        "selection is optimising photogeny over substance",
            "inconclusive": "the unseen instruments do not agree either way",
        }[verdict],
        "blocks_reported_not_ranked": {k: have[k].get("blocks") for k in ids},
    }
    out["human_look"] = _stage_human_look(rnd, winner, ids)
    return _write_selection(rnd, out)


def _stage_human_look(rnd: Round, winner: str, ids: list) -> dict:
    """The winner beside a seeded-random sibling, blind, for a person.

        The last question the judge does not get to answer. The sibling is drawn by a
        registered seed rather than picked, the sides are assigned by the same generator,
        and the key is written where a person can check it *after* writing a verdict --
        the same discipline `judge.py` applies to a model, applied to a human.
        
    """
    import shutil

    import numpy as np
    cfg = (rnd.candidates.get("human_look") or {})
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    others = sorted(k for k in ids if k != winner)
    sibling = others[int(rng.integers(0, len(others)))]
    winner_is_a = bool(rng.integers(0, 2))
    d = os.path.join(_pipeline._cand_dir(rnd), "human_look")
    os.makedirs(d, exist_ok=True)
    pre = (rnd.candidates.get("shots") or {}).get("prefix", "sel")
    src = {k: os.path.join(_pipeline._cand_dir(rnd), f"{pre}_{k}_card.png")
           for k in (winner, sibling)}
    missing = [k for k, p in src.items() if not os.path.exists(p)]
    if missing:
        return {"status": "no_cards", "missing": missing}
    a, b = (winner, sibling) if winner_is_a else (sibling, winner)
    shutil.copyfile(src[a], os.path.join(d, "a.png"))
    shutil.copyfile(src[b], os.path.join(d, "b.png"))
    key = {"seed": seed, "winner": winner, "sibling": sibling,
           "a": a, "b": b,
           "do_not_open_first": "write verdict.json before reading this file"}
    json.dump(key, open(os.path.join(d, "key.json"), "w"), indent=1)
    open(os.path.join(d, "ASK.md"), "w").write(
        "# For a person, and only a person\n\n"
        "Two buildings, same site, same brief, rendered identically. One was chosen "
        "by the judge; the other is a sibling drawn at random by a registered seed. "
        "Which of the two is better made?\n\n"
        f"    A  {os.path.join(d, 'a.png')}\n"
        f"    B  {os.path.join(d, 'b.png')}\n\n"
        "Write your answer to `verdict.json` in this directory as\n\n"
        '    {"verdict": "A" | "B" | "tie", "reason": "one sentence"}\n\n'
        "and only then open `key.json`. No model may answer this: it is the ground "
        "truth of the experiment and the one part of it the judge does not touch.\n")
    vp = os.path.join(d, "verdict.json")
    got = json.load(open(vp)) if os.path.exists(vp) else None
    return {"status": "answered" if got else "outstanding", "dir": d,
            "seed": seed, "sibling": sibling, "verdict": got,
            "note": "a person answers this; an unanswered look is reported as "
                    "outstanding and never as a result"}


def _write_selection(rnd: Round, out: dict) -> dict:
    p = os.path.join(_pipeline._cand_dir(rnd), "selection.json")
    os.makedirs(_pipeline._cand_dir(rnd), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    return {**out, "written": p}


# ------------------------------------------------------------ the loop before the city
# v2, C4. A city was judged after four hundred copies of a design nobody had looked at:
# the map, the landmark and five buildings are drawn the moment the plan exists --
# offline, deterministic, seconds -- written to disk, handed to the judge for a reading
# and to the principal for one bounded revision of the characters or the voice, and then
# the build.

#: How many representative buildings the preview draws: the plot types the plan names
#: most, one instance each on the round's own largest plot of that type.
PREVIEW_BUILDINGS = 5

#: How many revisions the loop admits before the build. One: the point is to look once
#: before four hundred copies, not to iterate a design by model call.
PREVIEW_REVISIONS = 1

READING_BRIEF = """# Read a plan before it is built

> {sentence}

{intent}

Three pictures are attached, drawn from the plan of this place before a block is laid:

1. **The map** -- the whole plan at one pixel a column: districts, rings, lanes, plots
   (brown), areas (green), walls (dark), gates and doors.
2. **The landmark** -- {landmark}, the greatest single thing the plan holds, built
   once on its own plot.
3. **Five buildings** -- one each of the plot types the plan names most, left to
   right: {buildings}.

{voice}

## How the districts are made

{characters}

## What is asked

Write a page, as a person who will walk this place: **what would they see wrong that
no check reports?** The fabric first -- do the streets, blocks and frontages read as
a place built on purpose, or as a grid; is the ground between the houses somebody's
or nobody's; is the density what the sentence means -- then the landmark, then the
buildings and the voice. Name each finding with the picture it is in and the
district or type it is about. Say what is right too, in a line. No score, no
numbers: what a person sees.

## And the same findings as data

Then write a second file, `{findings}`:

```json
{{"findings": [{{"id": "r1", "about": "composition|fabric|access|voice|scale",
                "says": "one sentence", "measure": "one of {measure_names} or null",
                "owner": "layout|fabric|voice|scale"}}],
  "closed": ["ids of the earlier reading's findings, listed below, that this plan no longer shows"],
  "right": ["what is right, briefly"]}}
```

`owner` is the decision that would have to change: `fabric` is a district's character
(lot size, frontage, storeys, open ground), `voice` the palette, `layout` where the
districts and the parts stand, `scale` how big the place is. `measure` names the number
that would move if the finding were fixed, or null.

{earlier}
"""

REVISION_BRIEF = """# One revision before the build

> {sentence}

{intent}

The plan of this place has been drawn and read once. The three pictures are attached
again -- the map, the landmark ({landmark}) and five buildings ({buildings}) -- and
this is the reading:

---

{reading}

---

## What you may change, once

**The characters of the districts** -- how each district is made, in words and a few
numbers -- and **the voice** the place is built in. Nothing else: the place's parts,
its site and its size stand. The compiler lays every district out again from what you
write, deterministically, and the build follows; there is no second look.

The characters as they stand:

{characters}

A character is an object of these fields, every one optional; a field you leave out
keeps the density word's default:

{fields}

The voice as it stands is `{voice_name}`. The voices on disk:

{voices}

The reading's findings, as data:

{findings}

## Output

Reply with one JSON document:

{{"characters": {{"<district part name>": {{...a character, whole...}}}},
  "voice": "<a voice name>" or null,
  "caused_by": ["the ids of the findings above this change answers"],
  "why": "one or two sentences"}}

`characters` holds only the districts you change, each with its whole character;
`{{}}` changes none. `voice` null keeps the voice. `caused_by` names the findings the
change is for -- a revision answers a finding or it is not a revision. If the reading
finds nothing worth a change, say so in `why` and change nothing.
"""


def _preview_landmark(parts: list) -> dict | None:
    """The plan's greatest single thing: the largest plot leaf inside a compound,
    else the largest civic plot, else the largest plot."""
    from . import load_type
    plots = [p for p in parts if p.get("kind", "plot") == "plot" and p.get("type")]
    if not plots:
        return None

    def area(p):
        r = _pipeline.part_rect(p)
        return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)

    def role(p):
        try:
            return load_type(os.path.join(_pipeline.ROOT, "types",
                                          f"{p['type']}.py")).get("role")
        except Exception:                        # noqa: BLE001 -- not a landmark then
            return None

    pool = [p for p in plots if p.get("compound")] or plots
    civic = [p for p in pool if role(p) == "civic"]
    pool = civic or pool
    return max(pool, key=lambda p: (area(p), p["name"]))


def _preview_buildings(parts: list, n: int = PREVIEW_BUILDINGS) -> list:
    """The `n` plot types the plan names most, each with the leaf it is drawn from:
    the largest plot of that type."""
    plots = [p for p in parts if p.get("kind", "plot") == "plot" and p.get("type")]
    by: dict = {}
    for p in plots:
        by.setdefault(p["type"], []).append(p)

    def area(p):
        r = _pipeline.part_rect(p)
        return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)

    order = sorted(by, key=lambda t: (-len(by[t]), t))[:n]
    return [(t, len(by[t]), max(by[t], key=lambda p: (area(p), p["name"])))
            for t in order]


def _preview_characters(rnd: Round, spec: dict | None) -> list:
    """Every district part's character as the spec wrote it, with what the compiler
    laid from it where it did."""
    from .. import spec as spec_mod
    out = []
    for p in (spec or {}).get("defining_parts") or []:
        if not spec_mod.district(p):
            continue
        row = {"part": p["name"], "density": p.get("density"), "role": p.get("role"),
               "character": p.get("character")}
        laid = []
        for f in sorted(os.listdir(rnd.state)) if os.path.isdir(rnd.state) else []:
            if f.startswith("district_") and f.endswith("_compiled.json"):
                rec = json.load(open(rnd.rel(f)))
                if rec.get("part") == p["name"]:
                    laid.append({k: rec.get(k) for k in
                                 ("district", "house", "lot", "block", "lots",
                                  "party_walls", "courts", "open", "verges",
                                  "plot_cover", "ground_cover", "undeveloped_share",
                                  "block_kinds", "raised")})
        row["compiled"] = laid
        out.append(row)
    return out


def _draw_preview(rnd: Round, plan: dict, parts: list, site, voice: str | None,
                  tag: str) -> dict:
    """The map, the landmark and the buildings, to `<state>/preview/`."""
    import cv2
    from .. import preview as preview_mod
    d = rnd.rel("preview")
    os.makedirs(d, exist_ok=True)
    t0 = time.perf_counter()
    out: dict = {"images": {}, "seconds": {}}
    img = preview_mod.plan_map(plan, rnd.network(), site)
    p = os.path.join(d, f"map{tag}.png")
    cv2.imwrite(p, img[:, :, ::-1])
    out["images"]["map"] = p
    out["seconds"]["map"] = round(time.perf_counter() - t0, 2)
    lm = _preview_landmark(parts)
    if lm is not None:
        t1 = time.perf_counter()
        img = preview_mod.instances(lm["type"], {"round": rnd.name, "plot": lm["name"]},
                                    seeds=(int(lm.get("seed") or 1),),
                                    params=dict(lm.get("params") or {}),
                                    voice=lm.get("voice") or voice, rnd=rnd)
        p = os.path.join(d, f"landmark{tag}.png")
        cv2.imwrite(p, img[:, :, ::-1])
        out["images"]["landmark"] = p
        out["landmark"] = {"name": lm["name"], "type": lm["type"],
                           "compound": lm.get("compound"),
                           "rect": list(_pipeline.part_rect(lm))}
        out["seconds"]["landmark"] = round(time.perf_counter() - t1, 2)
    reps = _preview_buildings(parts)
    if reps:
        import numpy as np
        t2 = time.perf_counter()
        strips = []
        for t, n, leaf in reps:
            strips.append(preview_mod.instances(
                t, {"round": rnd.name, "plot": leaf["name"]},
                seeds=(int(leaf.get("seed") or 1),),
                params=dict(leaf.get("params") or {}),
                voice=leaf.get("voice") or voice, scale=1, rnd=rnd))
        gutter = 6
        H = max(s.shape[0] for s in strips)
        W = sum(s.shape[1] for s in strips) + gutter * (len(strips) - 1)
        sheet = np.full((H, W, 3), preview_mod.BG, np.uint8)
        at = 0
        for s in strips:
            sheet[H - s.shape[0]:H, at:at + s.shape[1]] = s
            at += s.shape[1] + gutter
        sheet = np.repeat(np.repeat(sheet, 2, 0), 2, 1)
        p = os.path.join(d, f"buildings{tag}.png")
        cv2.imwrite(p, sheet[:, :, ::-1])
        out["images"]["buildings"] = p
        out["buildings"] = [{"type": t, "leaves": n, "drawn_from": leaf["name"]}
                            for t, n, leaf in reps]
        out["seconds"]["buildings"] = round(time.perf_counter() - t2, 2)
    out["seconds"]["all"] = round(time.perf_counter() - t0, 2)
    return out


#: What a revision or a repair may touch, and therefore what is put back where one is
#: refused. **The whole candidate and not the plan files.** The integration review's
#: fifth finding, reproduced: a rejected repair was rolled back and the round kept the
#: rejected candidate's `resolution.json` and `findings.json`, so the plan said one
#: thing and the record of what was wrong with it said another. A candidate is the plan,
#: the programme it was laid out from, the records that describe it, the ground it was
#: prepared on and the evidence it was judged by; restoring some of those is not
#: restoring a candidate.
REVISION_FILES = ("plan.", "district_", "place.json", "place.checked.json",
                  "plots.json", "network.json", "circulation.json", "voice.json",
                  "character.",
                  # the records of the design and what is wrong with it
                  "resolution.json", "findings.json", "capabilities.json",
                  "intent.json", "reading.json",
                  # what each artifact was made from, so a rolled-back candidate does
                  # not keep the rejected one's freshness claims
                  "deps.json",
                  # the ground the candidate was prepared on, and the arterials it was
                  # routed with
                  "ground.json", "plateau.json", "terraces.json", "arterials.",
                  # **the decisions a repair made, and the ground it made them on.** The
                  # review rolled a candidate back and found the rejected candidate's
                  # `layout_repairs.json` and `plan_repairs.json` still on disk beside
                  # the restored plan -- so the accepted design carried the record of
                  # revisions that had been withdrawn, and the repair budget those files
                  # count was spent on a candidate that no longer existed. The terrain
                  # was outside the snapshot for the same reason and is the one file the
                  # terraces pass actually rewrites.
                  "layout_repairs.json", "plan_repairs.json", "reallocations.json",
                  "world.npz", "world_built.npz", "world.quarters-from.npz",
                  # **the parent's regional decisions and the markers of what they
                  # laid** (the fabric reset round): a rolled-back plan beside the
                  # rejected candidate's strip cut, or without the `.laid` marker that
                  # tells the next pass its district was already re-laid for its form,
                  # is two designs at once. The ring levels are the ground those cuts
                  # were screened at.
                  "sectors.", "terrace_levels.json")

#: The preview directory is part of the candidate too: a reading and a drawing are
#: judgments **of a particular plan**, and keeping them across a rollback is how a
#: candidate ends up carrying another candidate's inspection.
REVISION_DIRS = ("preview",)


def _candidate_files(rnd: Round) -> list:
    out = [f for f in sorted(os.listdir(rnd.state))
           if f.startswith(REVISION_FILES) and os.path.isfile(rnd.rel(f))]
    for d in REVISION_DIRS:
        root = rnd.rel(d)
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, f)):
                out.append(os.path.join(d, f))
    return out


def _snapshot(rnd: Round) -> dict:
    return {f: open(rnd.rel(f), "rb").read() for f in _candidate_files(rnd)}


def _restore(rnd: Round, snap: dict) -> None:
    for f in _candidate_files(rnd):
        if f not in snap:
            os.remove(rnd.rel(f))
    for f, blob in snap.items():
        os.makedirs(os.path.dirname(os.path.abspath(rnd.rel(f))), exist_ok=True)
        with open(rnd.rel(f), "wb") as fh:
            fh.write(blob)


def _replan(rnd, be):
    """Lay the plan out again, **driven to an outcome** rather than called once.

        The review's fifth finding, last part: "The preview's revision and repair helpers
        call `stage_plan` once directly. A legitimate `reenter` or pending agent response is
        not driven to completion there." A stage that changes the candidate under itself
        returns `reenter` and the driver runs it again; a helper that calls it once reads
        that state as a result and decides the revision failed. Plan, preview and revision
        have to use the same control path or they are three different systems that look
        alike.
        
    """
    from . import round as driver, stages_plan
    res = stages_plan.stage_plan(rnd, be, {})
    return driver._drive_reentries(rnd, be, {}, "plan", res)


def _finish_preview(rnd: Round, rec: dict, rec_p: str, plan: dict | None) -> dict:
    """Mark this inspection done **and stamp what it was made from**. One exit.

        There were two, and only one of them stamped. The other -- "the revision budget is
        spent, so the loop is over" -- wrote `done` and returned, so the preview that had
        just been through a revision carried no dependency stamp at all; the next entry read
        "carries no dependency stamp: what it was made from is unknown", withdrew the
        drawings and the reading, and asked the judge to read the same candidate again. A
        completion that does not record what it completed is not a completion, and an
        inspection loop that re-opens itself on the next invocation is not bounded.
        
    """
    from .. import deps
    rec["done"] = True
    os.makedirs(os.path.dirname(rec_p), exist_ok=True)
    json.dump(rec, open(rec_p, "w"), indent=1)
    with contextlib.suppress(ValueError):
        deps.stamp(rnd, "preview", outputs=["preview/preview.json"], plan=plan,
                   note=f"{rec.get('revisions', 0)} revision(s), "
                        f"{len(rec.get('repairs') or [])} repair pass(es)")
    return rec


def _withdraw_preview(rnd: Round, d: str, rec: dict, why: str) -> dict:
    """Retire a preview's drawings and readings: they are about a candidate that is gone.

        One implementation, because there are two ways to find that out -- the stamped plan
        fingerprinted differently, or the candidate moved under a *pending* inspection -- and
        two copies of "withdraw the evidence" would be two answers. Invalidating the stamp of
        an artifact whose contents are still reused is not invalidating anything, so the
        drawings, the reading and the revision go with it.
        
    """
    from .. import deps
    rec = dict(rec)
    rec["done"] = False
    rec["stale"] = why
    rec["withdrawn"] = {"drawn": sorted(rec.get("drawn") or {}), "why": why,
                        "candidate": rec.get("candidate"),
                        # **the evidence is kept, its authority is withdrawn.** The
                        # closure round: a revision that was applied and then overtaken
                        # by a moved candidate is still a thing that happened
                        "applied": rec.get("applied"), "readings": rec.get("readings"),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec["drawn"] = {}
    rec.pop("reading", None)
    rec.pop("revision", None)
    rec.pop("applied", None)
    rec.pop("readings", None)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith(("reading", "revision")) and f.endswith((".md", ".json")):
            os.replace(os.path.join(d, f), os.path.join(d, f"stale.{f}"))
    deps.invalidate(rnd, "preview", why)
    return rec


def _new_repair_lineage(rnd: Round, why: str) -> int:
    """Open a fresh plan-repair lineage. Returns its number. See `_apply_revision`.

        The passes already spent stay in the record and are marked with the lineage they
        were spent in, so "this run has made eleven repair passes" remains true and
        readable; what the run-wide cap counts is the passes of the lineage in hand.
        
    """
    from . import stages_plan
    p = rnd.rel(stages_plan.PLAN_REPAIR_RECORD)
    was = json.load(open(p)) if os.path.exists(p) else {"passes": []}
    now = int(was.get("lineage") or 0) + 1
    for q in was["passes"]:
        q.setdefault("lineage", now - 1)
    was["lineage"] = now
    was.setdefault("lineages", []).append({"lineage": now, "why": why,
                                           "spent_before": len(was["passes"])})
    os.makedirs(rnd.state, exist_ok=True)
    json.dump(was, open(p, "w"), indent=1)
    return now


def _apply_revision(rnd: Round, be, spec: dict, doc: dict) -> dict:
    """The principal's one revision: the characters it names replace the spec's, the
    voice it names replaces the place's, and the plan is laid out again from them --
    the districts recompiled, the place level and the compounds standing."""
    from .. import spec as spec_mod, styles
    from . import stages_plan
    applied: dict = {"characters": {}, "voice": None, "refused": []}
    # **A revision that cannot be planned is not applied.** The plan on disk is what the
    # build stands on, and the first cut of this stage deleted it before laying the
    # place out again from the new characters -- so a revision the validator refused
    # left the round with no plan at all and nothing to fall back to. What the revision
    # may touch is snapshotted and put back.
    snap = _snapshot(rnd)
    chars = doc.get("characters") or {}
    if not isinstance(chars, dict):
        applied["refused"].append("`characters` is not an object")
        chars = {}
    raw_p = rnd.rel("place.checked.json") if os.path.exists(
        rnd.rel("place.checked.json")) else rnd.rel("place.json")
    raw = json.load(open(raw_p))
    parts = {p["name"]: p for p in spec["defining_parts"]}
    for name, ch in chars.items():
        p = parts.get(name)
        if p is None or not spec_mod.district(p):
            applied["refused"].append(f"{name}: not a district part of this place")
            continue
        try:
            probe = dict(p)
            spec_mod.read_character(ch, probe, f"revision ({name})")
        except spec_mod.SpecError as e:
            applied["refused"].append(f"{name}: {e}")
            continue
        for rp in raw.get("defining_parts") or []:
            if rp.get("name") == name:
                rp["character"] = probe["character"]
        applied["characters"][name] = probe["character"]
        for f in (rnd.rel(f"plan.district.{name}.json"),
                  rnd.rel(f"district_{name}_compiled.json")):
            if os.path.exists(f):
                os.remove(f)
        # a district drawn for this part under another name
        pp = rnd.rel("plan.place.json")
        if os.path.exists(pp):
            for dd in json.load(open(pp)).get("districts") or []:
                if dd.get("defines") == name:
                    for f in (rnd.rel(f"plan.district.{dd['name']}.json"),
                              rnd.rel(f"district_{dd['name']}_compiled.json")):
                        if os.path.exists(f):
                            os.remove(f)
    if applied["characters"]:
        json.dump(raw, open(raw_p, "w"), indent=1)
    v = doc.get("voice")
    if v:
        if v not in styles.VOICES:
            applied["refused"].append(f"voice {v!r} is not on disk")
        else:
            os.makedirs(rnd.state, exist_ok=True)
            json.dump({"voice": v, "chosen_by": "the principal's revision at the preview",
                       "why": str(doc.get("why") or "")},
                      open(rnd.rel("voice.json"), "w"), indent=1)
            pp = rnd.rel("plan.place.json")
            if os.path.exists(pp):
                place = json.load(open(pp))
                place["voice"] = v
                json.dump(place, open(pp, "w"), indent=1)
            applied["voice"] = v
    if applied["characters"] or applied["voice"]:
        # **The lanes go first.** The assembled plan is checked against the network, and
        # a network routed to the plots as they were refuses every plot the revision
        # moved -- "the circulation pass reserved no doorstep for this part" for every
        # house in the district. The snapshot puts them back if the revision is refused
        # for a reason of its own.
        from .. import local as _local_r
        for f in ("plan.json", "plots.json") + (
                ("network.json", "circulation.json") if _local_r.scope_of(rnd) is None
                else ()):
            if os.path.exists(rnd.rel(f)):
                os.remove(rnd.rel(f))
        # **And the districts the revision is about.** A character revision changes what
        # a district is made of, and the compiled district on disk is made of the old
        # one: left there, the plan stage's arrangement pass finds a file and skips, so
        # the revised fabric reached the compiler through `district_asks` with the
        # allocation the *old* fabric had been given and none of the capacity re-ask or
        # the recovery ladder ran. The shore village's revision was rolled back for a
        # cover it missed by forty-two columns while its own rectangle held two more
        # houses than it had been asked for. (under a local scope, not those outside it:
        # `local.retire_plans`)
        _local_r.retire_plans(rnd)
        # **A revision the principal asked for is a new design, and it gets the plan
        # stage's budget rather than the remains of the one before it.** The run-wide
        # cap on plan repairs exists to bound a *loop* -- each applied repair makes a
        # new candidate, and a new candidate would otherwise carry a fresh per-candidate
        # budget for ever. A deliberate revision is not that loop: it is the one bounded
        # change the inspection is allowed, and arriving at it with the run's budget
        # already spent on the design being replaced meant the revised place could not
        # be negotiated onto its own ground and was rolled back for it. Found by running
        # the loop fixture. The ledger is carried over, not discarded -- what was spent
        # stays readable -- and the lineage is what the cap counts.
        applied["lineage"] = _new_repair_lineage(
            rnd, f"the principal's revision at the preview: "
                 f"{str(doc.get('why') or '')[:120]}")
        # **The place level stands across a revision of its fabric.** A character says
        # what a district is made of; it does not say where the districts are. But a
        # character is part of the spec, the spec is what the place level is keyed on,
        # and the plan stage therefore called the place stale and solved it again from
        # the negotiated target -- producing *smaller districts*, on which the finer
        # fabric the inspection had asked for could not cover its own ground, and the
        # revision was rolled back for it. Re-deriving the geometry a decision was
        # measured against is the descent `stages_plan._plan_repair` already refuses for
        # a repair; a deliberate revision gets the same rule. The districts and
        # everything under them are dropped above; the layout is kept, and stamped so by
        # its owner.
        if applied["characters"] and os.path.exists(rnd.rel("plan.place.json")):
            from .. import deps as _deps_v
            with contextlib.suppress(ValueError):
                _deps_v.stamp(rnd, "plan", outputs=["plan.place.json"],
                              note=(f"kept across the principal's revision of "
                                    f"{', '.join(sorted(applied['characters']))}: a "
                                    f"character changes what a district is made of and "
                                    f"not where it is"))
        applied["plan"] = _replan(rnd, be)
        # **The lanes are the plan's, so a revision re-routes them.** The preview stands
        # after the circulation pass, because a building with no way in is an empty pad;
        # a revision that moves the plots leaves a network routed to plots that are not
        # there, and the stage that caused that is the stage that fixes it, before the
        # parts are built against either.
        p = (applied["plan"].get("plan") or {}) if isinstance(applied["plan"], dict) else {}
        ok = (not p.get("stop") and p.get("status") not in ("error", "needs_model")
              and os.path.exists(rnd.rel("plan.json")))
        if not ok:
            _restore(rnd, snap)      # the plan, the plots and the lanes as they were
            applied["refused"].append(
                "the revision was not applied and the plan stands as it was: the place "
                "laid out again from it " + (
                    f"fails at {p.get('level')} -- {p.get('error')}" if p.get("error")
                    else f"did not come back planned ({p.get('status')})"))
            applied["characters"], applied["voice"] = {}, None
            applied["rolled_back"] = True
            return applied
        applied["circulation"] = _pipeline.stage_circulation(rnd, be, {})
    return applied


def _repair_pass(rnd: Round, be, spec: dict, rec: dict, site) -> dict:
    """One bounded repair pass over `findings.json`, rechecked.

        Returns the record of it: what was routed where, what was applied, what was refused
        and -- where a planning decision changed -- the findings that stand *after* the plan
        was laid out again. A pass that changes nothing is recorded too, because "the
        findings were read and none of them was this layer's to fix" is a result.
        
    """
    from .. import contracts, repair as repair_mod
    from . import stages_plan
    findings = contracts.load(rnd, "findings")
    if findings is None or not findings["findings"]:
        return {"changed": False, "says": "no finding was open on the plan",
                "applied": [], "refused": [], "routed": {}}
    before = [f["id"] for f in findings["findings"]]
    snap = _snapshot(rnd)
    place_p = rnd.rel("plan.place.json")
    place = json.load(open(place_p)) if os.path.exists(place_p) else None
    got = repair_mod.apply(rnd, spec, findings, place=place, place_path=place_p)
    out = {"changed": False, "says": repair_mod.says(got), "before": before,
           "applied": got["applied"], "refused": got["refused"],
           "routed": got["routed"], "bounds": got["bounds"]}
    if not got.get("changed_plan"):
        return out
    # The plots and the lanes are laid out again from the repaired candidate. **The
    # place level survives a layout repair and not a scale one**: a frontage action
    # writes its decision *onto* `plan.place.json` (each district's `faces`), and
    # deleting the file would throw the repair away and lay the same place again; a
    # scale action changes the programme, and the place level has to be solved from it.
    # **The place level stands across a repair, whichever owner made it.** A layout
    # action writes its decision onto `plan.place.json` and deleting the file throws the
    # repair away; a scale action moves an *inferred target* to the capacity this very
    # geometry measured, and solving the place again from the smaller target measures a
    # smaller capacity and re-opens the same finding one size down. Found by running
    # both. What is dropped is what is downstream of the target: the districts, whose
    # lot sizes come from it, and the plan assembled out of them.
    from .. import local as _local_r
    doomed = ["plan.json", "plots.json"] + (["network.json", "circulation.json"]
                                            if _local_r.scope_of(rnd) is None else [])
    for f in doomed:
        if os.path.exists(rnd.rel(f)):
            os.remove(rnd.rel(f))
    _local_r.retire_plans(rnd)
    from .. import deps as _deps
    with contextlib.suppress(ValueError):
        _deps.stamp(rnd, "plan", outputs=["plan.place.json"],
                    note=f"kept across a preview repair: {repair_mod.says(got)}")
    again = _replan(rnd, be)
    p = (again.get("plan") or {}) if isinstance(again, dict) else {}
    ok = (not p.get("stop") and p.get("status") not in ("error", "needs_model")
          and os.path.exists(rnd.rel("plan.json")))
    if not ok:
        _restore(rnd, snap)
        out["rolled_back"] = True
        out["says"] = (f"the repair was not applied and the plan stands as it was: laid "
                       f"out again it " + (f"fails at {p.get('level')} -- {p.get('error')}"
                                           if p.get("error") else
                                           f"did not come back planned ({p.get('status')})"))
        return out
    out["circulation"] = _pipeline.stage_circulation(rnd, be, {})
    after = contracts.load(rnd, "findings")
    out["after"] = [f["id"] for f in (after or {}).get("findings") or []]
    out["closed"] = sorted(set(before) - set(out["after"]))
    out["opened"] = sorted(set(out["after"]) - set(before))
    out["changed"] = True
    out["says"] = (f"{repair_mod.says(got)}; rechecked: "
                   f"{len(out['closed'])} finding(s) closed, "
                   f"{len(out['opened'])} opened, {len(out['after'])} open")
    # the plan changed, so the drawing of the one before it is not this plan's
    rec["drawn"] = {}
    rec["revisions"] = 0
    for f in sorted(os.listdir(rnd.rel("preview"))):
        if f.startswith("reading") and f.endswith(".md"):
            os.replace(rnd.rel("preview", f),
                       rnd.rel("preview", f.replace(".md", ".before_repair.md")))
    return out


def _earlier_findings(rec: dict, tag: str) -> list:
    """The findings of the reading before this one, so the judge can say what closed."""
    if not tag:
        return []
    n = int(tag.strip("_") or 0)
    prev = "" if n <= 1 else f"_{n - 1}"
    got = (rec.get("readings") or {}).get(prev) or {}
    return list(got.get("findings") or [])


def _verify_improvement(rnd: Round, rec: dict, tag: str) -> dict:
    """Did the second reading close the findings the revision was for, and did the
    measures it named move the right way? Written onto `rec["applied"]["improvement"]`."""
    from . import inspect as inspect_mod
    imp = rec["applied"].setdefault("improvement", {})
    cited = list(rec["applied"].get("caused_by") or [])
    closed = set((rec.get("readings") or {}).get(tag, {}).get("closed") or [])
    cmp = imp.get("compare") or {}
    imp["closed"] = sorted(c for c in cited if c in closed)
    imp["still_open"] = sorted(c for c in cited if c not in closed)
    measured = bool(imp.get("cited_measures"))
    imp["verified"] = (bool(cited) and not imp["still_open"] and not cmp.get("worse")
                       and (not measured or bool(cmp.get("better"))))
    imp["verified_by"] = f"reading{tag}"
    imp["why"] = (f"the second reading closes {len(imp['closed'])} of {len(cited)} cited "
                  f"finding(s)" + (f"; still open: {imp['still_open']}"
                                   if imp["still_open"] else "")
                  + (f"; measures worse: {cmp.get('worse')}" if cmp.get("worse") else "")
                  + (f"; measures better: {cmp.get('better')}" if cmp.get("better") else ""))
    if not cited:
        imp["why"] = "the revision cited no finding, so there is nothing to verify"
    print(f"   preview: revision {imp['why']}", flush=True)
    return imp


def stage_preview(rnd: Round, be, results: dict) -> dict:
    """The loop before the city. v2, C4.

        After the plan and before the parts: the map, the landmark and five representative
        buildings are drawn to `<state>/preview/`, handed to the **judge** for a reading
        (what a person would see wrong that no check reports) and, with the reading, to
        the **principal** for one bounded revision of the districts' characters or the
        voice -- applied, the plan laid out again and redrawn -- and then the build. Each
        ask is a `needs_model` the driver answers as it answers the spec's; a round with
        no place spec (a recorded plan) is drawn and read, and has nothing to revise.
        
    """
    from .. import spec as spec_mod, styles
    plan = rnd.plan()
    if not plan:
        return {"error": "no plan.json: there is nothing to preview"}
    d = rnd.rel("preview")
    os.makedirs(d, exist_ok=True)
    rec_p = os.path.join(d, "preview.json")
    rec = json.load(open(rec_p)) if os.path.exists(rec_p) else {"revisions": 0,
                                                                "drawn": {},
                                                                "repairs": []}
    rec.setdefault("repairs", [])
    # **Which design this inspection is of.** A reading and a drawing are judgments of
    # one plan; the record now says which, so an inspection carried across a candidate
    # is visible rather than inferred from a `revisions` counter.
    from .. import deps as _deps_c
    #: **Which design this inspection is of -- compared before it is overwritten.** The
    #: review's fourth finding, and the bug was the order of these two lines: the new
    #: candidate id was assigned first and the freshness branch below only ran when the
    #: previous preview was already `done`, so a *pending* preview of a plan that had
    #: since changed kept its drawing and its refusal, performed zero redraws, took the
    #: new id and returned `done: true`. Binding an inspection to a candidate means
    #: asking whether it is still that candidate, in every state and not only the
    #: finished one.
    now = _deps_c.candidate_id(rnd, plan=plan)
    was = rec.get("candidate")
    # **...and to the types and the model it was drawn with.** The fresh checker: a
    # pending preview whose type fingerprint had moved kept its drawing, because the
    # pending branch compared the candidate id alone.
    inputs_now = _deps_c.fingerprint(rnd, ("types", "model"))
    inputs_was = rec.get("inputs")
    moved_inputs = bool(inputs_was) and inputs_was != inputs_now and not rec.get("done")
    if moved_inputs and was == now:
        why = ("the types or the model this inspection was drawn with changed while it "
               "was pending: what was drawn and whatever was read of it are about a "
               "design that is gone")
        print(f"   preview: {why}; it is drawn and read again", flush=True)
        rec = _withdraw_preview(rnd, d, rec, why)
    rec["inputs"] = inputs_now
    if was and was != now and not rec.get("done"):
        why = (f"the candidate moved from {was} to {now} while this inspection was "
               f"pending: what was drawn and whatever was read of it are about a design "
               f"that is gone")
        print(f"   preview: {why}; it is drawn and read again", flush=True)
        rec = _withdraw_preview(rnd, d, rec, why)
    rec["candidate"] = now
    if rec.get("done"):
        # **Done, of this plan.** The architecture audit's seventh finding: the stage's
        # whole test was this flag, so a plan that had changed under it came back with
        # the drawing of the one before. The loop is still bounded -- looked at once,
        # revised at most once, then built -- but "already done" now means "done, and
        # nothing it was made from has moved".
        from .. import deps
        # **"Never stamped" and "stamped, then withdrawn" are different.** They were the
        # same test -- `"preview" not in deps.json` -- and invalidating the preview
        # therefore *qualified* it as a legacy fixture and returned the old drawing.
        # `deps.legacy` is the question that was meant.
        legacy = deps.legacy(rnd, "preview")
        fresh, why = deps.check(rnd, "preview", plan=plan)
        if fresh or legacy:
            return {"preview": rec,
                    "dependencies": "warm" if not legacy else "unstamped"}
        print(f"   preview: {why}; it is drawn and read again", flush=True)
        # **And the drawing and the reading go with it.** The review reproduced the
        # bypass in this exact branch: the check noticed the changed plan, the stamp was
        # invalidated, `done` was cleared -- and then the code below found the current
        # tag already in `drawn` and `reading_1.md` already on disk, returned both, and
        # set `done` again. Invalidating the *stamp* of an artifact whose contents are
        # still reused is not invalidating anything. A drawing and a reading are of one
        # plan; when that plan moves they are evidence about a candidate that is gone.
        rec = _withdraw_preview(rnd, d, rec, why)
        json.dump(rec, open(rec_p, "w"), indent=1)
    spec = rnd.place_spec()
    site = rnd.site or (json.load(open(rnd.rel("site.json")))
                        if os.path.exists(rnd.rel("site.json")) else None)
    site = {"origin": site["origin"], "size": site["size"]} if site else None
    voice = rnd.voice_name() or None
    parts = _pipeline.plan_parts(plan)
    tag = "" if rec["revisions"] == 0 else f"_{rec['revisions']}"
    if tag not in rec["drawn"]:
        rec["drawn"][tag] = _draw_preview(rnd, plan, parts, site, voice, tag)
        json.dump(rec, open(rec_p, "w"), indent=1)
    drawn = rec["drawn"][tag]
    images = [drawn["images"][k] for k in ("map", "landmark", "buildings")
              if k in drawn["images"]]
    lm = drawn.get("landmark") or {}
    landmark_says = (f"`{lm['name']}`, a `{lm['type']}`" + (f" in the compound "
                     f"`{lm['compound']}`" if lm.get("compound") else "")
                     if lm else "none: the plan holds no building")
    buildings_says = ", ".join(f"`{b['type']}` ({b['leaves']} in the plan)"
                               for b in drawn.get("buildings") or []) or "none"
    chars = _preview_characters(rnd, spec)
    chars_says = "\n".join(
        f"- **{c['part']}** ({c['density'] or 'medium'}, {c['role'] or 'urban'}): "
        + (f"character `{json.dumps(c['character'])}`" if c["character"] is not None
           else "planned plot by plot, no character")
        + ("".join(f"\n    - compiled as `{l['district']}`: {l['lots']} lots of "
                   f"{l['lot']} on blocks of {l['block']} ({l['house']}), "
                   f"{l['party_walls']} party walls, {l['courts']} courts, {l['open']} "
                   f"open tiles, {l['verges']} verges; plots {l['plot_cover']:.0%}, "
                   f"ground {l['ground_cover']:.0%}, undeveloped "
                   f"{l['undeveloped_share']:.0%}"
                   + (f"; raised {l['raised']}" if l.get("raised") else "")
                   for l in c["compiled"]))
        for c in chars) or "(this plan was recorded, not planned from a spec)"
    sentence = (spec or {}).get("sentence") or rnd.sentence or plan.get("intent", "")
    intent = plan.get("intent", "")
    # 1. the reading
    reading_p = os.path.join(d, f"reading{tag}.md")
    findings_p = os.path.join(d, f"reading{tag}.findings.json")
    from . import inspect as inspect_mod
    if not os.path.exists(reading_p):
        brief_p = os.path.join(d, f"reading_prompt{tag}.md")
        earlier_rows = _earlier_findings(rec, tag)
        earlier = (("## The earlier reading's findings\n\n"
                    + "\n".join(f"- `{f.get('id')}` {f.get('says')}" for f in earlier_rows))
                   if earlier_rows else "")
        open(brief_p, "w").write(READING_BRIEF.format(
            sentence=sentence, intent=intent, landmark=landmark_says,
            buildings=buildings_says,
            voice=styles.voice_card(voice) if voice in styles.VOICES else "",
            characters=chars_says, findings=findings_p,
            measure_names=", ".join(f"`{m}`" for m in inspect_mod.MEASURES),
            earlier=earlier))
        rec["reading"] = {"request": brief_p, "write": reading_p, "findings": findings_p,
                          "candidate": rec.get("candidate")}
        json.dump(rec, open(rec_p, "w"), indent=1)
        return {"preview": rec,
                "reading": {"status": "needs_model", "role": "judge", "request": brief_p,
                            "write": reading_p, "images": images,
                            "candidate": rec.get("candidate"),
                            "note": "the preview's reading: what a person would see "
                                    "wrong, from the map, the landmark and five "
                                    "buildings; no score -- and the same as data"}}
    # **The reading as data, beside the prose.** The closure round: a revision has to
    # answer a finding by id, and a second reading has to say which of those it no
    # longer sees, or "revised and inspected again" is two pages of prose with nothing
    # between them a runner can check.
    rec.setdefault("readings", {})
    if tag not in rec["readings"]:
        got = json.load(open(findings_p)) if os.path.exists(findings_p) else None
        rec["readings"][tag] = {"path": reading_p, "structured": got is not None,
                                "findings": list((got or {}).get("findings") or []),
                                "closed": list((got or {}).get("closed") or []),
                                "right": list((got or {}).get("right") or [])}
        json.dump(rec, open(rec_p, "w"), indent=1)
    if tag and rec.get("applied") and (rec["applied"].get("improvement") or {}).get(
            "verified") is None:
        _verify_improvement(rnd, rec, tag)
        json.dump(rec, open(rec_p, "w"), indent=1)
    # 1b. **the repairs the findings route to a layer that can make them.** The
    # architecture round. Before this the only thing the loop could change was words: a
    # district's character and the place's voice. A finding about *scale* -- the
    # resolved design holds eleven structures against a band of twenty-four -- had
    # nobody to go to, and a repair that did land was never looked at again. So:
    # `repair.apply` makes the repairs this build can make, each inside a bound that is
    # written down and never against an explicit requirement; the plan is laid out again
    # from the repaired programme; the resolution and findings are recomputed; and the
    # stage **goes back to the reading**, so the repaired plan is inspected and not
    # merely redrawn. A repair whose plan will not lay out is rolled back by the
    # snapshot this stage already keeps.
    if spec is not None and not rec.get("repaired"):
        got = _repair_pass(rnd, be, spec, rec, site)
        rec["repairs"].append(got)
        rec["repaired"] = True
        json.dump(rec, open(rec_p, "w"), indent=1)
        if got.get("changed"):
            print(f"   preview: {got['says']}; the plan was laid out again and is read "
                  f"a second time", flush=True)
            # **An executable state, not a note.** The review drove the production
            # return shape through the real driver and it went straight on to `parts`:
            # the stage said in prose that it would re-enter and nothing re-entered it.
            # `status: "reenter"` is what `pipeline.round.run` acts on.
            return {"preview": rec, "repair": got, "status": "reenter",
                    "why": ("a repair changed a planning decision; this stage is "
                            "re-entered to inspect the plan it produced, and the plan "
                            "before it has not been inspected")}
    # 2. the revision, once, and only where there is a spec to revise
    if rec["revisions"] >= PREVIEW_REVISIONS or spec is None:
        return {"preview": _finish_preview(rnd, rec, rec_p, plan)}
    revision_p = os.path.join(d, "revision.json")
    if not os.path.exists(revision_p):
        prompt_p = os.path.join(d, "revision_prompt.md")
        fields = "\n".join(f"- `{k}`" for k in spec_mod.CHARACTER_FIELDS)
        fields += ("\n\nfrontage is one of " + ", ".join(f"`{f}`" for f in spec_mod.FRONTAGES)
                   + "; block and lot_depth are whole numbers of columns; attached is "
                     "true or false; the shares are 0 to 1; landmarks is a list of "
                     "{\"type\": a type name}.\n\nThe defaults per density word: "
                   + json.dumps(spec_mod.CHARACTER_DEFAULTS))
        voices = "\n".join(f"- `{k}` -- {v['blurb'].splitlines()[0][:110]}"
                            for k, v in styles.VOICES.items())
        rows = (rec.get("readings") or {}).get(tag, {}).get("findings") or []
        open(prompt_p, "w").write(REVISION_BRIEF.format(
            sentence=sentence, intent=intent, landmark=landmark_says,
            buildings=buildings_says, reading=open(reading_p).read().strip(),
            characters=chars_says, fields=fields, voice_name=voice or "(none)",
            voices=voices,
            findings=("\n".join(f"- `{f.get('id')}` ({f.get('owner')}, "
                                f"measure {f.get('measure')}): {f.get('says')}"
                                for f in rows)
                      or "(the reading was not written as data; cite nothing)")))
        rec["revision"] = {"request": prompt_p, "write": revision_p}
        json.dump(rec, open(rec_p, "w"), indent=1)
        return {"preview": rec,
                "revision": {"status": "needs_model", "role": "spec", "request": prompt_p,
                             "write": revision_p, "images": images,
                             "note": "one bounded revision of the districts' "
                                     "characters or the voice, before the build"}}
    # 3. apply it, lay the plan out again, draw again
    doc = json.load(open(revision_p))
    before_plan = rnd.plan()
    arr_p = rnd.rel("arrangements.json")
    before = inspect_mod.measure(before_plan, site,
                                 json.load(open(arr_p)) if os.path.exists(arr_p) else None)
    applied = _apply_revision(rnd, be, spec, doc)
    rec["revisions"] += 1
    rec["applied"] = {k: v for k, v in applied.items() if k != "plan"}
    rec["applied"]["why"] = str(doc.get("why") or "")
    # **What the revision was for, by id, and what it measurably did.** The closure
    # round's gate: a real finding causes a bounded change, and a current inspection
    # verifies the improvement. `caused_by` is the judge's finding ids; `improvement` is
    # the composition before and after, against the measures those findings named, and
    # `verified` is written when the revised candidate has been read again.
    rows = (rec.get("readings") or {}).get(tag, {}).get("findings") or []
    by_id = {f.get("id"): f for f in rows}
    cited = [c for c in (doc.get("caused_by") or []) if c in by_id]
    if not cited and rows and (doc.get("characters") or doc.get("voice")):
        cited = [f["id"] for f in rows if f.get("owner") in ("fabric", "voice")]
    rec["applied"]["caused_by"] = cited
    rec["applied"]["caused_by_findings"] = [by_id[c] for c in cited]
    measures = sorted({str(by_id[c].get("measure")) for c in cited
                       if by_id[c].get("measure")})
    after_plan = rnd.plan()
    after = inspect_mod.measure(after_plan, site,
                                json.load(open(arr_p)) if os.path.exists(arr_p) else None)
    rec["applied"]["improvement"] = {
        "before": before, "after": after, "cited_measures": measures,
        "compare": inspect_mod.compare(before, after, measures),
        "verified": False if applied.get("rolled_back") else None,
        "why": ("the revision was rolled back" if applied.get("rolled_back") else
                "written when the revised candidate has been read again")}
    if applied.get("plan") is not None:
        rec["applied"]["plan"] = {k: v for k, v in (applied["plan"].get("plan") or {}).items()
                                  if k in ("status", "error", "level")} \
            if isinstance(applied["plan"], dict) else None
    plan = rnd.plan()
    parts = _pipeline.plan_parts(plan)
    voice = rnd.voice_name() or None
    tag = f"_{rec['revisions']}"
    # **The record follows the candidate its own revision produced.** A revision changes
    # the design on purpose, so the identity this inspection is bound to moves with it;
    # without this the next entry finds the id it wrote before the revision, decides an
    # inspection was carried across a candidate, and withdraws the revision it had just
    # made. Deliberate change and stale evidence are different things and this is where
    # they are told apart.
    rec["candidate"] = _deps_c.candidate_id(rnd, plan=plan)
    if plan:
        rec["drawn"][tag] = _draw_preview(rnd, plan, parts, site, voice, tag)
    json.dump(rec, open(rec_p, "w"), indent=1)
    # **The revision is a changed candidate, so it is inspected.** The review: after the
    # character/voice revision this stage drew the result and set `done`, so the saved
    # village finished with `revisions: 1`, new images and no `reading_1.md` -- the
    # thing that was actually built was the one thing nobody looked at. Re-entering here
    # reaches the reading branch above with the new tag, which asks for a reading of
    # *this* candidate; the next entry finds it on disk and finishes.
    reading_now = os.path.join(d, f"reading{tag}.md")
    if applied.get("plan") is not None and not applied.get("rolled_back") \
            and not os.path.exists(reading_now):
        return {"preview": rec, "status": "reenter",
                "why": (f"revision {rec['revisions']} changed the candidate; it is read "
                        f"before it is built, like the one before it")}
    return {"preview": _finish_preview(rnd, rec, rec_p, plan)}
