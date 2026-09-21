"""The expression round's capability cases: forms and functions that are **measured**.

    $PY scripts/test_expression_capabilities.py

Every case runs the production entry points -- `construction.probe_build`/`outcome`,
`envelope.lot_for`, `growth.type_gaps`/`stage`/`gate` -- and asserts what stood, not what
a type declared:

  a. the courtyard house stands its court open to the sky and its gate hung, in two
     voices, and says so in `emitted` with rectangles;
  b. the market stands rows of stalls (solid) either side of an aisle (open), and a pad
     too small for a row reports `stalls` omitted instead of calling paving a market;
  c. `envelope.lot_for` answers from a type's ENVELOPE table where one covers the
     question and the declared lot agrees with a fresh probe;
  d. the cottage's lot grows with the storeys asked of it, and a required outshot costs
     more lot than none;
  e. a function the request names and no type of the form declares opens a growth gap
     whose brief carries the feature contract, and a run past its cap refuses by name;
  f. the adoption gate refuses a clean type that does not declare the gap's function.

they skip without them.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import construction as C, envelope, growth, pipeline, spec as spec_mod  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


def _open_share(b, rect, fy) -> float:
    x0, z0, x1, z1 = rect
    cells = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
    ok = 0
    for (x, z) in cells:
        paved = C._name(b._pending.get((x, fy, z), "air")) not in C.AIR
        clear = all(C._name(b._pending.get((x, fy + k, z), "air")) in C.AIR for k in (1, 2, 3))
        ok += int(paved and clear)
    return ok / float(len(cells))


def _solid_share(b, rect, fy) -> float:
    x0, z0, x1, z1 = rect
    cells = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
    n = sum(1 for (x, z) in cells
            if any(y >= fy + 1 and C._name(blk) not in C.AIR
                   for (px, y, pz), blk in b._pending.items() if (px, pz) == (x, z)))
    return n / float(len(cells))


# ------------------------------------------------------------------ a. the courtyard
# house

@case
def t_a_the_courtyard_house_stands_its_court_open_and_its_gate_hung():
    got = []
    for voice, lot, params in (("packed_earth_and_dark_tile", (18, 18), {"storeys": 1, "screen": "wall"}),
                               ("ochre_stone_green_tile", (26, 22), {"storeys": 2, "screen": "planted"}),
                               ("drystone_and_thatch", (13, 13), {"storeys": 1, "screen": "none"})):
        b, sited, res = C.probe_build("courtyard_house", *lot, params, seed=3, voice=voice)
        assert res.get("ok"), (voice, lot, res.get("reason"))
        em = res["emitted"]
        fy = sited["floor_y"]
        assert em["features"]["courtyard"] and em["features"]["gate"] and em["features"]["main_hall"], em["features"]
        court = em["rects"]["courtyard"]
        share = _open_share(b, court, fy)
        assert share >= 0.8, (voice, lot, "the court is not open ground", share)
        gx, gz = em["rects"]["gate"][0], em["rects"]["gate"][1]
        hung = any(C._name(b._pending.get((gx, fy + k, gz), "air")).endswith("door") for k in (1, 2))
        assert hung, (voice, lot, "no door leaf at the gate cell")
        outcome = C.outcome(b, sited, None, params)
        assert outcome["features_source"].get("gate") in ("verified", None) or outcome["features"].get("gate") is not False, outcome
        refusals = getattr(b, "_type_refusals", None) or []
        assert not refusals, (voice, lot, refusals)
        got.append((voice, lot, em["storeys"], round(share, 2), em["features"]["wings"],
                    em["features"]["screen"]))
    return f"court open and gate hung in three voices: {got}"


# ------------------------------------------------------------------ b. the market

@case
def t_b_the_market_stands_stall_rows_either_side_of_an_open_aisle():
    b, sited, res = C.probe_build("market", 34, 20, {"goods": "produce", "canopy": "gable"},
                                  seed=1, voice="packed_earth_and_dark_tile")
    assert res.get("ok"), res
    em = res["emitted"]
    fy = sited["floor_y"]
    assert em["features"]["rows"] == 2 and em["features"]["stalls"] >= 6, em["features"]
    assert _solid_share(b, em["rects"]["stalls"], fy) >= 0.5, "the stall rows are not solid"
    assert _open_share(b, em["rects"]["aisle"], fy) >= 0.8, "the aisle is not open"
    b2, s2, r2 = C.probe_build("market", 8, 8, {}, seed=1, voice="packed_earth_and_dark_tile")
    assert r2.get("ok") and "stalls" in r2["emitted"]["omitted"], r2.get("emitted")
    b3, s3, r3 = C.probe_build("square", 20, 20, {}, seed=1, voice="drystone_and_thatch")
    assert r3["emitted"]["features"]["stalls"] == 4 and r3["emitted"]["rects"].get("stalls"), r3["emitted"]
    return (f"{em['features']['stalls']} stalls in 2 rows, aisle open; an 8x8 pad reports "
            f"stalls omitted; the square reports its 4 booths with a rectangle")


# ------------------------------------------------------------------ c./d. envelopes

@case
def t_c_a_declared_envelope_row_answers_and_agrees_with_a_probe():
    rows = envelope.declared_table("cottage")
    if not rows:
        raise Skip("cottage carries no ENVELOPE table yet (scripts/type_needs.py --envelope)")
    params = {"storeys": 2, "outshot": "scullery"}
    envelope._MEM.clear()
    declared = envelope.lot_for("cottage", params, features=("storeys",))
    assert declared["source"] == "declared", declared
    probed = envelope.probe("cottage", params, ("storeys",), seed=1)
    assert probed["lot_min"] == declared["lot_min"], (declared, probed)
    return (f"cottage storeys=2 declared {declared['lot_min']} (pref {declared['lot_pref']}), "
            f"probed {probed['lot_min']} in {probed['probes']} probes")


@case
def t_d_the_cottages_lot_grows_with_the_storeys_asked_of_it():
    areas = []
    for st in (1, 2, 3):
        got = envelope.lot_for("cottage", {"storeys": st, "outshot": "store"}, features=("storeys",))
        assert got["lot_min"], got
        areas.append(got["lot_min"][0] * got["lot_min"][1])
    assert areas[0] <= areas[1] <= areas[2] and areas[2] > areas[0], areas
    bare = envelope.lot_for("cottage", {"storeys": 1, "outshot": "byre"}, features=("storeys",))
    lean = envelope.lot_for("cottage", {"storeys": 1, "outshot": "byre"}, features=("storeys", "outshot"))
    assert lean["lot_min"][0] * lean["lot_min"][1] >= bare["lot_min"][0] * bare["lot_min"][1], (bare, lean)
    feats = envelope.required_features([
        {"kind": "quality", "wants": {"axis": "height", "value": "low"}},
        {"kind": "feature", "wants": {"feature": "market", "family": "square"}},
        {"kind": "function", "wants": {"function": "smithing", "what": "smithy"}}])
    assert feats == ("storeys", "stalls", "forge"), feats
    return (f"storeys 1/2/3 need {areas} columns; a byre lean-to needs {lean['lot_min']} "
            f"against {bare['lot_min']} without; required_features reads {feats}")


# ------------------------------------------------------------------ e./f. growth by
# function

SENTENCE = "Build a small town of twelve houses with a brewery and a tannery."
SIZE, GROUND = 192, 64


def _spec(form="east_asian"):
    doc = {"kind": "town", "form": form, "voice": None,
           "defining_parts": [{"name": "homes", "kind": "group", "family": "district",
                               "relation": "throughout", "count": 1, "structures": 12,
                               "character": {}}]}
    return spec_mod.read_spec(json.loads(json.dumps(doc)), SENTENCE)


def _intent(*fns):
    return {"requirements": [
        {"id": f"function/{fn}", "kind": "function", "hard": True, "status": "open",
         "wants": {"function": fn, "what": fn}, "phrase": fn} for fn in fns]}


def _round(tmp, spec):
    import numpy as np
    from ethoslm import offline
    from ethoslm.observe import Volume
    for r in ("site_b", "site_d", "site_e"):
        if not os.path.exists(os.path.join(ROOT, "out", r, "plots.json")):
            raise Skip(f"needs out/{r}/plots.json for the default fixtures")
    state = os.path.join(ROOT, "out", "growth_fixture_b")
    shutil.rmtree(state, ignore_errors=True)
    os.makedirs(state, exist_ok=True)
    y0 = GROUND - 14
    codes = np.zeros((SIZE, 70, SIZE), np.uint16)
    codes[:, :GROUND - y0, :] = 3
    codes[:, GROUND - y0, :] = 1
    offline.save_volume(Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"]),
                        os.path.join(state, "world.npz"))
    site = {"origin": [0, 0], "size": SIZE}
    json.dump({**site, "stats": {"min": GROUND, "max": GROUND, "relief": 0, "std": 0.0},
               "mean_grid": [[GROUND] * 12 for _ in range(12)],
               "roughness_grid": [[0] * 12 for _ in range(12)],
               "surface_blocks": {"grass_block": 100}},
              open(os.path.join(state, "site.json"), "w"))
    json.dump(spec, open(os.path.join(state, "place.json"), "w"))
    cfg = os.path.join(tmp, "growth_fixture_b.json")
    json.dump({"name": "growth_fixture_b", "state_dir": state, "sentence": SENTENCE,
               "site": site, "flags": {"dry_run": True, "growth_sweep": False}},
              open(cfg, "w"), indent=1)
    rnd = pipeline.Round.load(cfg)
    from ethoslm.pipeline import blind
    for n in ("brewing", "tanning", "milling"):
        shutil.rmtree(blind._blind_dir(rnd, os.path.join("growth", n), 0), ignore_errors=True)
    return rnd, pipeline.OfflineBackend(rnd, dry_run=True)


@case
def t_e_a_function_no_type_declares_opens_a_growth_gap_with_a_feature_contract():
    s = _spec()
    it = _intent("brewing", "dwelling", "market")
    gaps = growth.type_gaps(s, None, intent=it)
    names = [(g["name"], g["kind"], g.get("function"), g.get("answers")) for g in gaps]
    assert names == [("brewing", "plot", "brewing", ["function/brewing"])], names
    # `market` is answered by types/market.py (FUNCTION market, civic form) and
    # `dwelling` by the form's houses; only the brewery opens a gap
    unsupported = _intent("brewing")
    unsupported["requirements"][0]["status"] = "unsupported"
    assert growth.type_gaps(s, None, intent=unsupported) == [], "an unsupported requirement opens nothing"
    tmp = tempfile.mkdtemp()
    rnd, be = _round(tmp, s)
    got = growth.stage(rnd, be, s, gaps, site={"origin": [0, 0], "size": SIZE},
                       voice="packed_earth_and_dark_tile")
    assert got and got["plan"]["status"] == "needs_model" and got["plan"]["role"] == "type", got
    brief = open(growth.record_of(rnd)["asked"]["brewing"]["brief"]).read()
    assert "The feature contract" in brief and "FUNCTION = 'brewing'" in brief and "rects" in brief, brief[-600:]
    rec = growth.record_of(rnd)
    assert rec["asked"]["brewing"]["tspec"]["function"] == "brewing"
    assert rec["asked"]["brewing"]["tspec"]["answers"] == ["function/brewing"]
    # past the cap: a third function in a run that asked two is refused by name
    rec["asked"]["tanning"] = dict(rec["asked"]["brewing"], part="tanning")
    growth._save(rnd, rec)
    third = growth.function_gaps(s, _intent("milling"))
    refused = growth.stage(rnd, be, s, third, site={"origin": [0, 0], "size": SIZE},
                           voice="packed_earth_and_dark_tile")
    assert refused["plan"]["status"] == "error" and "cap" in refused["plan"]["error"], refused
    assert "milling" in (growth.record_of(rnd)["capped"] or []), growth.record_of(rnd)
    return ("brewing opens a plot gap answering function/brewing; the brief names the "
            "feature contract; market and dwelling open nothing; a third ask is capped by name")


@case
def t_f_the_gate_refuses_a_clean_type_that_does_not_declare_the_function():
    s = _spec()
    tmp = tempfile.mkdtemp()
    rnd, be = _round(tmp, s)
    part = growth.function_gaps(s, _intent("brewing"))[0]
    fixtures = growth.default_fixtures("plot")
    tspec = growth._tspec(rnd, s, part, "packed_earth_and_dark_tile", fixtures)
    tspec["check_sweep"] = False
    got = growth.gate(rnd, be, tspec, os.path.join(ROOT, "types", "temple.py"),
                      "packed_earth_and_dark_tile")
    assert got["passes"] is False and got.get("function_refused"), {k: v for k, v in got.items() if k != "text"}
    # ...and one that does declare it, in the form, passes the same gate on the same
    # ground
    tspec2 = dict(tspec, function="dwelling", form="east_asian")
    got2 = growth.gate(rnd, be, tspec2, os.path.join(ROOT, "types", "courtyard_house.py"),
                       "packed_earth_and_dark_tile")
    assert got2["passes"] is True and not got2.get("function_refused"), \
        {k: v for k, v in got2.items() if k != "text"}
    return (f"temple (no FUNCTION) refused for a brewing gap: {got['function_refused'][:80]}; "
            f"courtyard_house passes a dwelling gap on {got2['instances']} instances")


# ------------------------------------------------------------------ the runner

def main() -> int:
    ok = fail = skipped = 0
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"ok   {name:70} {msg}")
        except Skip as e:
            skipped += 1
            print(f"skip {name:70} {e}")
        except Exception as e:                   # noqa: BLE001 -- a suite reports
            fail += 1
            import traceback
            print(f"FAIL {name:70} {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{ok + fail} expression capability cases pass, {skipped} skipped")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
