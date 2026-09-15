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
