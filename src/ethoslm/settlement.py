"""Shared state across build passes.

A settlement cannot be written as one program — 8.4M blocks and a dozen structures do
not fit in one context. So it is built in passes, and the passes need somewhere to
agree. This is the smallest thing that works: a plot registry on disk.

Deliberately not a semantic layer. It records what ground is claimed and by whom, and
nothing about what a building means. Anything a later pass needs to know about an
earlier one, beyond its footprint, it reads out of the world itself.
"""
from __future__ import annotations

import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#: Which settlement this process is working on. it is still standing, still full of
#: known defects, and its plot registry must not be overwritten by a later round.
#: $ETHOSLM_SETTLEMENT names the directory under out/ so a new round is a new name rather
#: than a lost fixture.
NAME = os.environ.get("ETHOSLM_SETTLEMENT", "settlement")
STATE = os.path.join(ROOT, "out", NAME)


#: The shared state a snapshot carries with the world. Everything else under STATE --
#: prompts, programs, findings, renders -- is a record of the run, not state a pass
#: reads, so it is not rolled back.
STATE_FILES = frozenset({"plots.json", "network.json", "plan.json", "site.json",
                         "passes.json", "paths.json"})


def _path(name: str) -> str:
    os.makedirs(STATE, exist_ok=True)
    return os.path.join(STATE, name)


def site_info() -> dict:
    """The terrain briefing for this settlement's site."""
    p = _path("site.json")
    if os.path.exists(p):
        return json.load(open(p))
    old = os.path.join(ROOT, "out", "settlement_site.json")
    if NAME in ("settlement", "") and os.path.exists(old):
        return json.load(open(old))
    raise RuntimeError(
        f"{NAME} has no site briefing at {p}: a pass that reads the ground has to be "
        f"told which ground, and falling back to out/settlement_site.json would cache "
        f"another round's site under this round's name")


def network_path() -> str:
    return _path("network.json")


def load_network():
    """The circulation network this settlement was built around, or None if none has
    been laid yet. Only its *declaration*: whether it is still walkable is a question
    for the world, and lint derives that separately."""
    from .circulate import Network
    return Network.load(network_path())


def load_plots() -> list[dict]:
    p = _path("plots.json")
    return json.load(open(p)) if os.path.exists(p) else []


def save_plots(plots: list[dict]) -> None:
    json.dump(plots, open(_path("plots.json"), "w"), indent=1)


def registry_with_floors(state_dir: str) -> list[dict]:
    """`plots.json` from a state directory, with `y0` filled in from `parts.json`."""
    p = os.path.join(state_dir, "plots.json")
    plots = json.load(open(p)) if os.path.exists(p) else []
    q = os.path.join(state_dir, "parts.json")
    if not os.path.exists(q):
        return plots
    doc = json.load(open(q))
    floors = {r["part"]: r.get("floor_y")
              for w in (doc.get("waves") or []) for r in (w.get("parts") or [])
              if r.get("status") == "built" and r.get("floor_y") is not None}
    for row in plots:
        if "y0" not in row and row.get("label") in floors:
            row["y0"] = int(floors[row["label"]])
    return plots


def load_paths() -> list[dict]:
    """Every way in that a pass laid: `approach()` and `flight()` paths, per plot."""
    p = _path("paths.json")
    return json.load(open(p)) if os.path.exists(p) else []


def add_paths(rows: list[dict]) -> list[dict]:
    """Append what a pass laid to the record, and return the whole of it."""
    if not rows:
        return load_paths()
    all_rows = load_paths() + [dict(r) for r in rows]
    json.dump(all_rows, open(_path("paths.json"), "w"), indent=1)
    return all_rows


def path_columns(paths: list[dict] | None = None) -> set:
    """The columns every recorded way in occupies, as {(x, z)}."""
    return {(int(x), int(z))
            for row in (load_paths() if paths is None else paths)
            for (x, z) in row.get("cells", ())}


def snapshot(tag: str) -> str:
    """Freeze the world **and this settlement's shared state** under one name."""
    from . import world
    out = world.snapshot(tag)
    dest = os.path.join(world.SNAP_DIR, f"{tag}.state")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if os.path.isdir(STATE):
        os.makedirs(dest, exist_ok=True)
        for f in STATE_FILES:
            p = os.path.join(STATE, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dest, f))
    return out


def restore(tag: str) -> None:
    """Roll the world and this settlement's shared state back to a snapshot."""
    from . import world
    world.restore(tag)
    src = os.path.join(world.SNAP_DIR, f"{tag}.state")
    if not os.path.isdir(src):
        return
    held = set(os.listdir(src))
    for f in held:
        shutil.copy2(os.path.join(src, f), os.path.join(STATE, f))
    for f in STATE_FILES - held:
        p = os.path.join(STATE, f)
        if os.path.exists(p):
            os.remove(p)


def _note_conflict(want: tuple, label: str, blocker: dict) -> None:
    """Record a refused claim, append-only, to <state>/plot_conflicts.json.

        A `reserve()` that returns False is the plot abstraction saying no, and until now it
        said so only to the program that asked -- which then moved, silently, and nothing
        downstream could tell a building that chose its ground from a building that was
        pushed off it. Step 4 pre-registers "did any planned structure have to move because
        plots cannot overlap in 2-D?" as a question to be answered with evidence, and this
        is the evidence. Deliberately not a lint check and not a failure: overlapping claims
        are legal and expected on a cliff, where two dwellings share one column of ground.
        
    """
    try:
        p = os.path.join(STATE, "plot_conflicts.json")
        rows = json.load(open(p)) if os.path.exists(p) else []
        rows.append({"label": label,
                     "wanted": {"x0": want[0], "z0": want[1],
                                "x1": want[2], "z1": want[3]},
                     "blocked_by": blocker.get("label"),
                     "blocker": {k: blocker[k] for k in ("x0", "z0", "x1", "z1")}})
        os.makedirs(STATE, exist_ok=True)
        json.dump(rows, open(p, "w"), indent=1)
    except Exception:
        pass          # instrumentation must never be able to fail a build pass


class PlotRegistry:
    """Reserve ground so two passes cannot build in the same place."""

    def __init__(self):
        self.plots = load_plots()
        self.claimed_this_pass: list[dict] = []

    def reserve(self, x0: int, z0: int, x1: int, z1: int, label: str) -> bool:
        """Claim a rectangle of ground. Returns False if it overlaps an existing plot,
        in which case nothing is claimed and you should move."""
        a = (min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
        for p in self.plots:
            b = (p["x0"], p["z0"], p["x1"], p["z1"])
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                _note_conflict(a, label, p)
                return False
        rec = {"x0": a[0], "z0": a[1], "x1": a[2], "z1": a[3], "label": label}
        self.plots.append(rec)
        self.claimed_this_pass.append(rec)
        return True

    def plots_list(self) -> list[dict]:
        """Every plot claimed so far, by any pass, as {x0,z0,x1,z1,label}."""
        return [dict(p) for p in self.plots]

    def commit(self) -> None:
        save_plots(self.plots)
