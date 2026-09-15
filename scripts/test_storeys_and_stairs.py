"""The court fix: the one weak type, and the two trivia under it.

    $PY scripts/test_storeys_and_stairs.py

  F1. **A storey is only built when the stair that reaches it stands** (thread 49).
      `court_large` chose where its flight went on cells that were free *enough* --
      `would_shut` filters out a cell something already stands in, because for a chest
      that cell is simply taken -- so a run of six columns was allowed to cross a
      partition wall. `flight()` will not drive a stair through a wall: it says so and
      lays nothing, and the type laid the floor, the walls and the windows over the
      refusal anyway. E003 and E011, in pairs, at six of the type's nine parameter sets.
  F2. **And a storey its own two stairs would cut in two is not built at all.** The
      well of the flight arriving is a hole across one row of the range and the first
      three columns of the flight leaving are solid across another, so a north range
      two rows deep came back in two halves with nobody able to reach the far one. The
      floor is walked before a block of it is laid.
  F3. **A slab ahead of a roof stair, at its own level, is the course carrying on**
      (thread 50). `backwards_stairs` reads what *fills* a cell and a bottom slab does
      not fill one, so a roof laid as alternating stair and slab courses -- any shallow
      profile -- read as a fall of one block at every stair in it.
  F4. **A stair that is the highest thing on its own axis is not on a slope** (thread
      50). Both sides at or below it means there is no fall through the cell to have
      got the wrong way round: a ridge cap, a finial, the top step of a stepped gable.
      A reversed flight keeps the surface it came down from behind it, one course up,
      and is still reported -- which is the control here.
  F5. **A checker fixture is written through a rename.** Every worker that reads one
      builds it again, to identical bytes; a worker that opened `plots.json` while
      another had it truncated read an empty file and killed a whole swept run at eight
      jobs while surviving at one.

F1 and F2 build their own ground through `slopefixture`; nothing here needs `out/`
beyond the scratch directory that writes, and nothing needs a server.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import buildlib, observe, pipeline, slopefixture  # noqa: E402
from ethoslm.pipeline import blind  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(ROOT, "rounds", "types_g.json")
CASES = []


def case(fn):
    CASES.append((fn.__name__[3:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# --------------------------------------------- F1/F2. the type on real ground

def _stand(fixture: str, plot: str, seed: int, voice: str, params: dict,
           name: str = "court_large") -> dict:
    """One instance of one type on one fixture, read the way its checker reads it."""
    rnd = pipeline.Round.load(CONFIG)
    be = pipeline.OfflineBackend(rnd)
    path = os.path.join(ROOT, "types", f"{name}.py")
    decl = pipeline.load_type(path)
    kw = pipeline.check_params(decl["params"], params, where=f"{name}.py")
    got = blind._check_type_job(rnd, be, path, decl,
                                [{"round": fixture, "plot": plot}], [kw],
                                (fixture, seed, voice, 0))
    return got["rows"][0]


def _flights_of(fn):
    """Every answer `flight()` gave while `fn` ran."""
    said = []
    orig = buildlib.Builder.flight

    def traced(self, label, x, z, y0, y1, facing, **kw):
        r = orig(self, label, x, z, y0, y1, facing, **kw)
        said.append(r)
        return r

    buildlib.Builder.flight = traced
    try:
        out = fn()
    finally:
        buildlib.Builder.flight = orig
    return out, said


#: The ground both of these stand on: two pads at the top of `court_large`'s declared
#: band, on one bank, the clearance the plan validator admits apart. `pair_downhill` is
#: where the type failed, and the uphill neighbour is what made it fail.
PAIR = "slopepair_32_32"


@case
def t_f1_a_storey_is_only_built_when_the_stair_that_reaches_it_stands():
    """Thread 49's class, on the fixture that found it.

        Before the fix this instance read `E003` -- "a room of 119 cells cannot be walked
        into" -- and `E011` beside it, because the flight had been refused with "a tread
        would replace the spruce_planks you placed", and the floor went on over it.
        
    """
    rows = []
    for voice in ("japanese_minka", "ochre_stone_green_tile"):
        (row, said) = _flights_of(
            lambda v=voice: _stand(PAIR, "pair_downhill", 1, v,
                                   {"storeys": 3, "yard": "well"}))
        refused = [r for r in said if not r.get("ok")]
        assert not refused, (voice, [r["reason"] for r in refused])
        assert row["errors"] == 0, (voice, row["codes"], row["errors"])
        assert row["entry_lines"] == 0, (voice, row["lines"])
        rows.append((voice, len(said), row["walk_pct"]))
    return "; ".join(f"{v}: {n} flight(s), all standing, {w}% walkable, 0 errors"
                     for (v, n, w) in rows)


@case
def t_f2_a_storey_two_stairs_would_cut_in_two_is_not_built():
    """The other half of thread 49, and the general rule under it.

        The type is stood at every storey count it declares on both plots of the pair, in
        one voice; whatever it decides to build, no room it leaves may be unreachable and
        no floor cell may be off the walk from outdoors. What it is allowed to decide is to
        stop lower: a range it cannot cross upstairs stands single and low.
        
    """
    seen = []
    for plot in ("pair_uphill", "pair_downhill"):
        for storeys in (1, 2, 3):
            row = _stand(PAIR, plot, 2, "japanese_minka",
                         {"storeys": storeys, "yard": "garden"})
            assert row["errors"] == 0, (plot, storeys, row["codes"])
            assert row["entry_lines"] == 0, (plot, storeys, row["lines"])
            seen.append(f"{plot}/{storeys}: {row['walk_pct']}%")
    return ", ".join(seen)


# ------------------------------------------------- F3/F4. the roof-stair check

def _roofy(kinds: dict, base: int = 68, size: int = 24) -> observe.Volume:
    """A plane of stone with air over it, and whatever `kinds` puts in it.

        `kinds` is {(x, y, z): block state}; everything at or below `base` is stone.
        
    """
    palette = ["air", "stone",
               "stone_brick_stairs[facing=east,half=bottom]",
               "stone_brick_stairs[facing=west,half=bottom]",
               "stone_brick_slab[type=bottom]"]
    y0, y1 = base - 6, base + 12
    codes = np.zeros((size, y1 - y0 + 1, size), dtype=np.int32)
    codes[:, :base - y0 + 1, :] = 1
    vol = observe.Volume(0, y0, 0, codes, palette)
    vol.overlay(dict(kinds))
    return vol


def _reported(vol, at) -> bool:
    return any(tuple(f["pos"]) == tuple(at) for f in observe.backwards_stairs(vol))


@case
def t_f3_a_slab_ahead_at_the_stairs_own_level_is_the_course_carrying_on():
    """Thread 50, two of the three: a ridge course and a lean-to.

        The stair stands two courses over the ground with a wall a course above it behind:
        that is what makes the old comparison call it backwards. What is ahead decides it,
        and a bottom slab ahead at the stair's own level is the roof going on at that
        level -- the same rule as "a stair whose facing side stands at or above its own
        level faces up", read on the half block `local_top` was blind to.
        
    """
    st = "stone_brick_stairs[facing=east,half=bottom]"
    wall = {(9, 70, 10): "stone", (9, 71, 10): "stone"}
    with_slab = _roofy({**wall, (10, 70, 10): st,
                        (11, 70, 10): "stone_brick_slab[type=bottom]"})
    without = _roofy({**wall, (10, 70, 10): st})
    assert not _reported(with_slab, (10, 70, 10)), "a slab ahead is the course going on"
    assert _reported(without, (10, 70, 10)), "with nothing ahead it is still reported"
    return ("a roof stair with a slab ahead at its own level is not reported; "
            "the same stair with air ahead of it is")


@case
def t_f4_a_stair_higher_than_both_its_neighbours_is_not_on_a_slope():
    """Thread 50, the third: the top step of a stepped gable.

        And the control, which is what stops this being a hole in the check: a tread whose
        flight keeps climbing **behind** it -- a reversed flight, a serrated roof -- has a
        surface above its own level behind it and is reported exactly as before.
        
    """
    st = "stone_brick_stairs[facing=east,half=bottom]"
    top = _roofy({(9, 69, 10): "stone", (10, 70, 10): st})
    reversed_flight = _roofy({(9, 71, 10): "stone", (9, 70, 10): "stone",
                              (10, 70, 10): st})
    assert not _reported(top, (10, 70, 10)), "the top of a rise faces nothing"
    assert _reported(reversed_flight, (10, 70, 10)), "a reversed tread is still wrong"
    return ("the top step of a rise is not reported; a tread with the flight still "
            "climbing behind it is")


# ------------------------------------------------------- F5. the fixture write

@case
def t_f5_a_checker_fixture_is_written_through_a_rename():
    """What a swept run at eight jobs died of, and what a rename is for."""
    d = tempfile.mkdtemp(prefix="ethoslm-fixture-")
    try:
        p = os.path.join(d, "plots.json")
        before = json.dumps([{"label": "a"}])
        open(p, "w").write(before)

        def boom(tmp):
            tmp.write_text("half a fi")
            raise RuntimeError("the worker died mid-write")

        try:
            slopefixture._atomic(__import__("pathlib").Path(p), boom)
        except RuntimeError:
            pass
        assert open(p).read() == before, "a failed write left the file damaged"
        left = [f for f in os.listdir(d) if f != "plots.json"]
        assert not left, f"a temporary was left behind: {left}"

        after = json.dumps([{"label": "b"}])
        slopefixture._atomic(__import__("pathlib").Path(p),
                             lambda tmp: tmp.write_text(after))
        assert open(p).read() == after
        assert os.listdir(d) == ["plots.json"], os.listdir(d)
        return ("a failed fixture write leaves the file as it was and no temporary "
                "behind; a good one renames onto it")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------- the runner

def main():
    ok = bad = skipped = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Skip as e:
            skipped += 1
            print(f"skip {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok} of {ok + bad} cases pass, {skipped} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
