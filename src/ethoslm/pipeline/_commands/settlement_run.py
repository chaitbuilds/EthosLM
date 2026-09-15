'Execute one or more settlement build passes against the live world.\n\nEach pass is a separate program written by a separate model call. They share the world\n(later passes read what earlier ones built), a plot registry (so they do not build on\ntop of each other), and -- since the circulation pass now runs first -- a network they\ncan *ask questions of while they are running*:\n\n    check_door(x, y, z)   can a person walk to this doorway from the lane, without\n                          jumping, given everything I have placed so far?\n    threshold(label)      the threshold the circulation pass reserved for this structure\n    nearest_lane(x, z)    the closest lane cell to a column\n\nAround every pass, three things happen that did not before:'
import sys, os, json, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from gdpc.vector_tools import Rect

from ethoslm import lint, observe, offline, settlement, world
from ethoslm.buildlib import Builder
from ethoslm.frontage import Frontage
from ethoslm.measure import record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
site_info = settlement.site_info()
X, Z = site_info["origin"]
S = site_info["size"]
PAD = 48

API = offline.BUILD_API

names = sys.argv[1:]
net = settlement.load_network()
_plan_path = os.path.join(settlement.STATE, "plan.json")
VOICE_PALETTE = (json.load(open(_plan_path)).get("palette")
                 if os.path.exists(_plan_path) else None)
reg = settlement.PlotRegistry()
summary = {}
print(f"settlement {settlement.NAME}: site ({X},{Z}) {S}x{S}, "
      + (f"{len(net.cells)} lane cells, {len(net.thresholds)} reserved thresholds"
         if net else "no circulation network -- passes cannot check their own frontage"))


def observe_now(ed):
    """The world as it stands, decoded once: the volume every check below shares.

        The floor of the y window is set from the **site's** ground, not the padded
        region's. The pad reaches into a lake bed thirty blocks lower, and a window that
        starts there includes the cave systems under the town: `rooms()` finds them, they
        are sheltered and enclosed, `plot_at` is two-dimensional and attributes anything
        under a plot to it, and wave 1 was accused of eighteen unreachable rooms at y=1.
        
    """
    ed.loadWorldSlice(Rect((X - PAD, Z - PAD), (S + 2 * PAD, S + 2 * PAD)), cache=True)
    h = ed.worldSlice.heightmaps[world.HEIGHTMAP].astype(int) - 1
    inner = h[PAD:PAD + S, PAD:PAD + S]
    y0, y1 = max(0, int(inner.min()) - 8), int(h.max()) + 48
    vol = observe.Volume.from_world_slice(ed.worldSlice, X - PAD, Z - PAD,
                                          S + 2 * PAD, S + 2 * PAD, y0, y1)
    return vol


for name in names:
    src_path = os.path.join(settlement.STATE, f"{name}.py")
    if not os.path.exists(src_path):
        print(f"!! {name}: no program at {src_path}")
        continue
    # Preflight before a single block is written. this costs milliseconds and needs no
    # server.
    source = open(src_path).read()
    pre = lint.preflight(source)
    if not pre.ok:
        print(f"!! {name}: preflight failed, not running")
        print(pre.summary())
        summary[name] = {"error": "preflight", "preflight": pre.to_json()}
        record("preflight_reject", name=name, errors=len(pre.errors))
        continue

    t0 = time.perf_counter()
    ed = world.editor()
    site = world.load_site(ed, X - PAD, Z - PAD, S + 2 * PAD, S + 2 * PAD)
    vol0 = observe_now(ed)
    nav0 = observe.Nav(vol0)
    snap = lint.snapshot(nav0, net) if net else None
    front = Frontage(vol0, net) if net else None
    setup_s = time.perf_counter() - t0
    print(f"\n=== {name}: world read in {setup_s:.1f}s"
          + (f", {len(snap['reachable'])} stances walkable from the lane" if snap else ""))

    b = Builder(site)
    b.frontage = front
    b.registry = reg
    env = {n: getattr(b, n) for n in API}
    env["reserve"] = reg.reserve
    env["plots"] = reg.plots_list
    env["__name__"] = "__build__"
    env["__builtins__"] = __builtins__

    t0 = time.perf_counter()
    err = None
    try:
        exec(compile(source, src_path, "exec"), env)
    except Exception:
        err = traceback.format_exc()
    exec_s = time.perf_counter() - t0

    res = b.flush() if b._pending else {"placed": 0, "failed": 0, "errors": {}}
    outside = 0
    bbox = None
    if b._pending:
        xs = [p[0] for p in b._pending]; ys = [p[1] for p in b._pending]
        zs = [p[2] for p in b._pending]
        bbox = [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]
        outside = sum(1 for p in b._pending
                      if not (X <= p[0] < X + S and Z <= p[2] < Z + S))

    # --- what it did to the town ---------------------------------------------------
    t0 = time.perf_counter()
    ed2 = world.editor()
    vol1 = observe_now(ed2)
    ctx = lint.Context.build(vol1, reg.plots_list(), placed=res.get("palette"),
                             network=net, before=snap,
                             region=(X, Z, X + S - 1, Z + S - 1),
                             own_plots=[p["label"] for p in reg.claimed_this_pass],
                             voice=VOICE_PALETTE)
    # Two questions, kept apart. What did *this pass* build wrong (the build family, on
    # its own plots), and what did it do to *the town* (the place family, everywhere).
    # Reporting one number for both is how a pass gets blamed for its neighbour's
    # stairs.
    report = lint.lint(ctx)
    mine = lint.lint(ctx, family=lint.BUILD).within(reg.claimed_this_pass)
    town = lint.lint(ctx, family=lint.PLACE)
    lint_s = time.perf_counter() - t0
    print(f"--- {name}'s own work ---")
    print(mine.summary())
    print(f"--- the town, after it ---")
    print(town.summary())
    print(f"(lint in {lint_s:.1f}s)")
    brief = lint.findings_brief(
        lint.Report(mine.findings + town.findings, lint_s),
        f"{name}: what the linter found",
        header=(f"Your program ran against the world and placed {res['placed']} blocks. "
                f"Revise it to fix what follows. The world is rolled back to the state "
                f"before your pass and your revised program re-run whole, so write a "
                f"complete build, not a patch."))
    open(os.path.join(settlement.STATE, f"{name}_findings.md"), "w").write(brief)

    fr = front.report() if front else {}
    if fr.get("checked"):
        print(f"frontage: the pass checked {fr['checked']} doorways while running, "
              f"{fr['failed_when_asked']} came back not on the network")

    # What this pass laid as a way in, written down before the finishing pass runs. It
    # is the one thing finish_run.py cannot read back out of the world: a dressed
    # approach path and an undressed one look identical from the blocks.
    laid = settlement.add_paths([{**p, "pass": name} for p in b.paths])

    summary[name] = {"error": err, "exec_seconds": round(exec_s, 2), "bbox": bbox,
                     "paths": [{"label": p["label"], "kind": p["kind"],
                                "columns": len(p["cells"])} for p in b.paths],
                     "outside_site": outside, "placed": res["placed"],
                     "failed": res["failed"], "errors": res["errors"],
                     "palette_size": res.get("palette_size"),
                     "families": len(res.get("material_families", [])),
                     "plots_claimed": len(reg.claimed_this_pass),
                     "frontage": fr, "lint": report.to_json()}
    record("settlement_pass", name=name, settlement=settlement.NAME, ok=err is None,
           placed=res["placed"], failed=res["failed"], outside_site=outside,
           plots_claimed=len(reg.claimed_this_pass),
           families=len(res.get("material_families", [])),
           frontage_checks=fr.get("checked", 0),
           frontage_failed=fr.get("failed_when_asked", 0),
           lint_errors=len(report.errors), lint_warnings=len(report.warnings),
           lint_counts=report.to_json()["counts"], seconds=round(exec_s, 2))
    print(f"=== {name}: placed {res['placed']}, failed {res['failed']}, "
          f"plots +{len(reg.claimed_this_pass)}, outside {outside}, "
          f"{len(report.errors)} errors, "
          f"{sum(len(p['cells']) for p in b.paths)} columns of way-in recorded "
          f"({len(laid)} rows in paths.json)")
    if err:
        print(err[-1500:])
    reg.claimed_this_pass = []

reg.commit()
p = os.path.join(settlement.STATE, "passes.json")
old = json.load(open(p)) if os.path.exists(p) else {}
old.update(summary)
json.dump(old, open(p, "w"), indent=1)
print("\nplots now:", len(reg.plots))
