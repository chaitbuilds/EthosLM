"""Block id and block-state validation, against the game's own registry.

The data is `src/ethoslm/data/blocks_1.21.11.json`, distilled by
scripts/dump_block_registry.py from the vanilla server's own data generator. No jar, no
server, no network needed at check time.

**Do not substitute the client jar's blockstates files for this.** That was the first
attempt and it is wrong: those files enumerate only the properties that affect the
*model*, so every stair and slab appears to have no `waterlogged` and campfire no
`signal_fire`. Checked against the town it called 87 of the 152 block states standing
in the world invalid, every one of them real. Render data is not registry data."""
from __future__ import annotations

import difflib
import json
import os
from functools import lru_cache

VERSION = "1.21.11"
DATA = os.path.join(os.path.dirname(__file__), "data", f"blocks_{VERSION}.json")


@lru_cache(maxsize=1)
def load(path: str = DATA) -> dict:
    """{block_id: {property: [legal values]}} for every block in the game."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run scripts/dump_block_registry.py to regenerate it")
    return json.load(open(path))


def validate(state: str, reg: dict | None = None) -> list[str]:
    """Errors in one block state string, e.g. "oak_stairs[facing=east]". [] if valid."""
    reg = reg if reg is not None else load()
    s = state.split(":")[-1].strip()
    name = s.split("[")[0]
    if name not in reg:
        near = suggest(name, reg)
        return [f"unknown block id {name!r}" + (f" (nearest: {near})" if near else "")]
    if "[" not in s:
        return []
    if not s.endswith("]"):
        return [f"{name}: malformed block state {state!r}"]
    props = reg[name]
    errs = []
    for part in s.split("[", 1)[1][:-1].split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            errs.append(f"{name}: malformed property {part!r}")
            continue
        k, v = (t.strip() for t in part.split("=", 1))
        if k not in props:
            known = ", ".join(sorted(props)) or "none"
            errs.append(f"{name}: no property {k!r} (has: {known})")
        elif v not in props[k]:
            errs.append(f"{name}: {k}={v!r} invalid (accepts: {', '.join(props[k])})")
    return errs


def suggest(name: str, reg: dict | None = None, n: int = 3) -> str:
    reg = reg if reg is not None else load()
    return ", ".join(difflib.get_close_matches(name, list(reg), n=n, cutoff=0.6))


def check_all(states) -> dict:
    """Validate many states at once. Returns {state: [errors]} for the bad ones only."""
    reg = load()
    return {s: e for s in states if (e := validate(s, reg))}


# Which arguments of the build API are block ids. Validating by *position* is what keeps
# this precise: guessing which string literals look like blocks either misses the
# unknown ids (the whole point) or flags style="gable" and facing="north" as typos.
# `mat` on roof()/roof_cone()/dormer() is a material *family* name, not a block id, so
# those functions are deliberately absent.
BLOCK_ARGS = {
    "place_block": ({3}, {"block"}),
    "place_cuboid": ({6}, {"block"}),
    "fill_region": ({6}, {"block", "replace"}),
    "disc": ({4}, {"block"}),
    "ring": ({4}, {"block"}),
    "cylinder": ({5}, {"block"}),
    "sphere": ({4}, {"block"}),
    "dome": ({4}, {"block"}),
    "line": ({6}, {"block"}),
    "path": ({2}, {"block"}),
    "foundation_to_grade": ({5}, {"block"}),
    "terrace": ({5}, {"block"}),
    "wall": ({6}, {"block", "post", "base", "band"}),
    "window": (set(), {"glass"}),
}


def check_source(text: str) -> dict:
    """Validate the block ids a program passes to the build API, before it runs.

    Only *literal* strings can be checked here. Model programs build ids dynamically all
    the time (f"{mat}_stairs[facing={face}]"), and those are caught after the fact by
    E001 against the palette actually placed. Preflight is the cheap first line, not the
    only one."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return {"<source>": [f"does not parse: {e}"]}
    reg = load()
    out: dict[str, list] = {}

    def consider(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            errs = validate(node.value, reg)
            if errs:
                out.setdefault(node.value, errs)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        spec = BLOCK_ARGS.get(name)
        if not spec:
            continue
        positions, keywords = spec
        for i, arg in enumerate(node.args):
            if i in positions:
                consider(arg)
        for kw in node.keywords:
            if kw.arg in keywords:
                consider(kw.value)
    return out


if __name__ == "__main__":
    import sys
    reg = load()
    print(f"{len(reg)} block ids in {VERSION}")
    probes = sys.argv[1:] or [
        "cut_red_sandstone_stairs", "chain", "cauldron[level=3]",
        # states that must pass -- all present in the standing town
        "iron_chain", "water_cauldron[level=3]", "stone_bricks",
        "oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]",
        "campfire[facing=north,lit=true,signal_fire=false,waterlogged=false]",
        "red_sandstone_slab[type=bottom,waterlogged=false]",
        # and states that must fail
        "oak_stairs[facing=up]", "oak_stairs[flavour=vanilla]",
    ]
    bad = 0
    for p in probes:
        errs = validate(p, reg)
        bad += bool(errs)
        print(f"  {'FAIL' if errs else 'ok  '}  {p:66s} {'; '.join(errs)}")
