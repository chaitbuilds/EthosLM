#!/usr/bin/env python3
"""The spatial-design round's building-form and court-obligation work, measured.

Two things the round's brief says the library cannot do, and the cases that hold them.

    F1  **a building form determines the space it needs.** `row_house` has been a 6x6
        box since it was written and the retained section is a hundred and thirty-one
        one-storey sheds because of it. Five measured claims, and the conjunction of the
        first four is what makes the fifth an architectural gain rather than a trade:

        F1a the type's declared band carries real depth, and the band is the one the
            registered sweep measured;
        F1b the ground a dense house costs does not become the 10x10 square plot the
            area-only ranking gave it, and the lot stays a terrace;
        F1c extending one type's valid depth does not redefine "dense" for the library:
            `dense_plot`, `density_lot` and all four default fabrics stand still;
        F1d a narrow deep terrace and a large square plot are distinguishable decisions,
            because `density_lot` carries a shape and `_attached_lot` ranks on it;
        F1e the lot the crowded ring is now handed lets the same type, in the same
            architectural language, stand at more than one height.

    F2  **an adopted district form owes its court.** A district that adopted a
        courtyard-block arrangement publishes a court subject even where no leaf of it is
        a courtyard-house type, the section's court record publishes every subject beside
        what was asked, and a set with subjects nobody asked about reads `failed` rather
        than `demonstrated`.

Every number in a readout is measured here, through the production entry points
(`placeplan.density_lot`, `placeplan.fabric`, `district_compile._attached_lot`,
`construction.probe_build`, `demand.required_by_part`, `section._courts`). The "before"
figures are the retained ones and are stated beside the corrected ones rather than
replaced. No world is built and nothing under `out/` is written.

Just over a minute: F1e stands eighteen small houses on flat probe ground and F3 compiles
one real district of the retained section twice, once with its terrain and once without.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ethoslm import (construction, demand, district_compile as dc,  # noqa: E402
                   feasible, placeplan, section as section_mod, spec as spec_mod)
from ethoslm.buildlib import Builder                                # noqa: E402

CASES = []

#: Every one of these was read through the same entry point the case below reads.
RETAINED = {
    # `placeplan.fabric("dense", "urban", {"attached": True})` on the retained library
    "dense_attached_lot": [6, 8],
    "dense_attached_per_house": 111.5,
    "dense_attached_lots_per_block": 3,
    "dense_attached_plot_share": 0.4304,
    # the regression the old area-only ranking produced the moment the band widened
    "regression_lot": [10, 10],
    "regression_per_house": 198.0,
    # `dense_plot` and `density_lot`, which must not move at all
    "dense_plot": {"pad": 6, "plot": 10, "columns": 100, "type": "row_house"},
    "density_sides": {"sparse": 20, "low": 15, "medium": 12, "dense": 10},
    "density_per_house": {"sparse": 1250.0, "low": 571.4, "medium": 323.4,
                          "dense": 231.9},
    # the pads the retained section actually gave its ninety-six row houses
    "section_pads": [(5, 6), (5, 7)],
    "courts_status": "demonstrated",
    "courts_asked": 4,
    "courts_subjects": None,
}


class Skip(Exception):
    """This case has no fixture here."""


def case(fn):
    CASES.append(fn)
    return fn


# ------------------------------------------------------------------ the instruments

def row_house() -> object:
    """The committed type file's own namespace, read and never executed as a build."""
    spec = importlib.util.spec_from_file_location(
        "_row_house_form", os.path.join(ROOT, "types", "row_house.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lot_for_pad(pw: int, pd: int) -> tuple:
    """A plot whose sited pad is `pw` x `pd`, asked through `Builder.pad_extent` so this
    and the library cannot drift. `test_neighbourhood_evidence._lot_for_pad`'s rule."""
    for w in range(pw, pw + 8):
        for d in range(pd, pd + 8):
            if tuple(Builder.pad_extent({"kind": "plot", "x0": 0, "z0": 0,
                                         "x1": w - 1, "z1": d - 1})) == (pw, pd):
                return (w, d)
    raise AssertionError(f"no plot gives a pad of {pw}x{pd}")


def stand(w: int, d: int, storeys: int, front: str, seed: int) -> dict:
    """One row house on one lot. `construction.probe_build`, the same instrument
    `envelope.probe` and the evidence suite use."""
    params = {"storeys": storeys, "front": front}
    b, sited, res = construction.probe_build("row_house", w, d, params, seed=seed)
    if isinstance(res, dict) and res.get("ok") is False:
        return {"ok": False, "why": res.get("reason")}
    out = construction.outcome(b, sited, None, params)
    rect = (out.get("rects") or {}).get("main")
    return {"ok": True, "storeys": out.get("storeys"), "height": out.get("height"),
            "pad": (sited["x1"] - sited["x0"] + 1, sited["z1"] - sited["z0"] + 1),
            "rect": rect,
            "inside": bool(rect) and sited["x0"] <= rect[0] and rect[2] <= sited["x1"]
            and sited["z0"] <= rect[1] and rect[3] <= sited["z1"],
            "depth": (max(rect[2] - rect[0], rect[3] - rect[1]) + 1) if rect else None,
            "frontage": (min(rect[2] - rect[0], rect[3] - rect[1]) + 1) if rect else None}


def dense_terrace() -> dict:
    """The fabric the crowded ring is actually built of: dense, urban, party walls."""
    return placeplan.fabric("dense", "urban", {"attached": True})


# --------------------------------------------- F1a. the band is the measured band

@case
def t_f1a_the_declared_band_carries_depth_and_is_inside_the_measured_one():
    """`row_house.NEEDS["footprint"]`, against `rounds/type-needs.json`.

        The band read `(4, 4, 6, 6)` and the file's own note said why the honest wider band
        could not be taken: widening it moved the library's dense lot. The brief's first
        requirement is eight columns of depth, which is what `top` needs for a longitudinal
        flight -- and the second is that taking it must not move the dense lot. This case
        holds the first; `t_f1b` and `t_f1c` hold the second.
        
    """
    mod = row_house()
    lo_w, lo_d, hi_w, hi_d = mod.NEEDS["footprint"]
    bank = json.load(open(os.path.join(ROOT, "rounds", "type-needs.json")))
    want = bank["types"]["row_house"]["band"]["footprint"]
    # the declaration is inside the measured envelope -- `test_types`' own rule,
    # asserted here too because this case is the reason the envelope moved
    assert [lo_w, lo_d] >= [want[0], want[1]] and [hi_w, hi_d] <= [want[2], want[3]], \
        (mod.NEEDS["footprint"], want)
    # ...and it carries the depth a stair needs. Eight columns of house, which `_sizes`
    # takes one row of the pad's depth off once the pad is eight deep.
    assert hi_d >= 9, mod.NEEDS["footprint"]
    assert hi_d > 6, "the band did not widen at all"
    # ...and the type says what it can do with the ground, which is the other half
    table = getattr(mod, "STOREY_PAD", None)
    assert table and table.get(2), "row_house declares no STOREY_PAD"
    pw, pd = table[2]
    assert pd >= 9 and pw >= 5, table
    # the declared pad for two storeys is one the band admits
    assert pw >= min(lo_w, lo_d) and pd <= max(hi_w, hi_d), (table, mod.NEEDS)
    return (f"declared band {lo_w}x{lo_d}..{hi_w}x{hi_d} (was 4x4..6x6), inside the "
            f"registered sweep's measured envelope {want[0]}x{want[1]}..{want[2]}x"
            f"{want[3]}; STOREY_PAD {dict(table)}: two storeys need a {pw}x{pd} pad and "
            f"the old ceiling of 6 gave the type six")


# --------------------------------------------- F1b. the dense house's ground

@case
def t_f1b_the_dense_terrace_keeps_its_shape_and_does_not_become_a_square_plot():
    """`placeplan.fabric("dense", "urban", {"attached": True})`, before and after.

        What this case holds, and does not pretend otherwise about:

          * the lot is a **terrace**: its frontage is at most the retained 6 and its depth
            is greater, so it is not the square plot and not any rounding of one;
          * it costs **less ground than the regression**, by a wide margin;
          * its lot is inside the hundred columns `dense` has always allowed one house;
          * and the ground per house **does** rise against the retained 111.5, because a
            block of deeper lots is a deeper block. The rise is measured and named here
            rather than hidden: what is bought with it is in `t_f1e`.
        
    """
    f = dense_terrace()
    w, ld = f["lot"]
    per = float(f["columns_per_structure"])
    assert f["attached"] and f["type"] == "row_house", f
    # a terrace, not a square plot
    assert ld > w, ("the dense attached lot is not deeper than it is wide", f["lot"])
    assert w <= RETAINED["dense_attached_lot"][0], ("the frontage widened", f["lot"])
    assert list(f["lot"]) != RETAINED["regression_lot"], f["lot"]
    # inside the ground the word has always allowed one house
    allowance = int(placeplan.density_lot("dense", "urban")["columns"])
    assert w * ld <= allowance, (f["lot"], allowance)
    # and well under the regression's cost
    assert per < RETAINED["regression_per_house"] * 0.8, (per, RETAINED)
    # ...and denser in the thing a lot is for: more of the block is house
    assert f["plot_share"] > RETAINED["dense_attached_plot_share"], f
    # the block holds its street frontage rather than its lot count, which is where a
    # quarter of the rise is paid back: the same 30 columns of frontage the density's
    # own square lot bought at three lots
    assert int(f["lots_per_block"]) > RETAINED["dense_attached_lots_per_block"], f
    # what this lot would cost on the old rule, in the fabric's own arithmetic: three
    # lots a block whatever their width, which is the number that made a narrow frontage
    # charge itself for being narrow
    n0 = RETAINED["dense_attached_lots_per_block"]
    old_per = ((n0 * w + f["street"]) * (f["block_depth"] + f["street"])
               / (n0 * (2.0 - 2.0 * f["open_share"] - f["courtyard_share"])))
    return (f"the dense terrace is {w}x{ld} = {w * ld} columns of lot (was "
            f"{RETAINED['dense_attached_lot'][0]}x{RETAINED['dense_attached_lot'][1]} = "
            f"{RETAINED['dense_attached_lot'][0] * RETAINED['dense_attached_lot'][1]}), "
            f"inside the {allowance} a dense house is allowed; {f['lots_per_block']} "
            f"lots a block against {RETAINED['dense_attached_lots_per_block']}, "
            f"{per:.1f} columns a house against the retained "
            f"{RETAINED['dense_attached_per_house']} and the area-ranked regression's "
            f"{RETAINED['regression_per_house']} on a "
            f"{RETAINED['regression_lot'][0]}x{RETAINED['regression_lot'][1]} plot; "
            f"{old_per:.1f} if the block still held {n0} "
            f"lots whatever their width; plots are {f['plot_share']:.1%} of the block "
            f"against {RETAINED['dense_attached_plot_share']:.1%}")


# --------------------------------------------- F1c. the library does not move

@case
def t_f1c_widening_one_types_depth_does_not_redefine_dense_for_the_library():
    """`dense_plot`, `density_lot` and all four registered fabrics, unmoved.

        The coupling the round exists to cut runs through **squares**: `dense_plot` reads
        only the sizes a type admits on both axes at once, takes the urban type whose square
        ceiling is lowest, and makes that the lot every dense district is built of. Widening
        `row_house`'s *depth* touches none of it, and this is the case that says so with the
        retained numbers beside the measured ones -- the farm, the hamlet and the hill-town
        fixtures are sized off exactly these.
        
    """
    d = placeplan.dense_plot()
    for k, v in RETAINED["dense_plot"].items():
        assert d[k] == v, (k, d[k], v)
    moved = []
    for word in ("sparse", "low", "medium", "dense"):
        role = placeplan.DENSITY_ROLE.get(word)
        lot = placeplan.density_lot(word, role)
        if int(lot["side"]) != RETAINED["density_sides"][word]:
            moved.append(f"{word} lot side {lot['side']} != "
                         f"{RETAINED['density_sides'][word]}")
        assert lot["columns"] == spec_mod.columns_per_plot({"density": word}), (word, lot)
        f = placeplan.fabric(word, role)
        if abs(float(f["columns_per_structure"])
               - RETAINED["density_per_house"][word]) > 0.05:
            moved.append(f"{word} {f['columns_per_structure']} a house != "
                         f"{RETAINED['density_per_house'][word]}")
        # the default fabric of every word is still the square lot, unattached
        if list(f["lot"]) != [RETAINED["density_sides"][word]] * 2:
            moved.append(f"{word} lot {f['lot']}")
    assert not moved, moved
    # ...and the shares the whole ladder is derived from are the same numbers
    sh = placeplan.occupancy_shares()
    assert [sh[w]["plot_side"] for w in ("sparse", "low", "medium", "dense")] \
        == [20, 15, 12, 10], {w: sh[w]["plot_side"] for w in sh}
    return (f"dense_plot {d['pad']}x{d['pad']} pad, {d['plot']}-square plot, "
            f"{d['columns']} columns off `{d['type']}` -- unchanged; density_lot sides "
            + ", ".join(f"{w} {placeplan.density_lot(w, placeplan.DENSITY_ROLE[w])['side']}"
                        for w in ("sparse", "low", "medium", "dense"))
            + "; the four default fabrics cost "
            + ", ".join(f"{w} {placeplan.fabric(w, placeplan.DENSITY_ROLE[w])['columns_per_structure']:.1f}"
                        for w in ("sparse", "low", "medium", "dense"))
            + " columns a house, every one of them the retained figure")


# --------------------------------------------- F1d. narrow deep vs large square

@case
def t_f1d_a_narrow_terrace_and_a_square_plot_are_different_decisions():
    """`density_lot`'s shape, and `district_compile._attached_lot` ranking on it.

        A density word that can only say "a hundred columns" cannot tell a narrow deep
        terrace from a large square plot: both are a hundred columns. The proof is the
        ranking itself -- asked for the same ground with no shape, the library returns the
        square; asked for the same ground with the terrace's shape, it returns the terrace.
        
    """
    mod = row_house()
    decl = {"needs": dict(mod.NEEDS), "path": os.path.join(ROOT, "types", "row_house.py")}
    flanks = ("west", "east")
    area = int(placeplan.density_lot("dense", "urban")["columns"])
    square = dc._attached_lot(decl, 10, flanks, True, area=area)
    lot = placeplan.density_lot("dense", "urban", {"attached": True})
    shape = (int(lot["width"]), int(lot["depth"]))
    narrow = dc._attached_lot(decl, 10, flanks, True, area=area, shape=shape)
    assert square and narrow, (square, narrow)
    # the two questions have two answers, and the shapeless one is the square
    assert square != narrow, (square, narrow)
    assert square[0] == square[1], ("the shapeless ask is not square", square)
    assert narrow[1] > narrow[0], ("the shaped ask is not a terrace", narrow)
    # the shape is carried on the record, and the old keys are untouched
    assert lot["side"] == 10 and lot["columns"] == 100, lot
    assert lot["shape"] == [lot["width"], lot["depth"]], lot
    # ...and the depth the shape asks for is the depth the answer honours: the frontage
    # may widen (a lot with no flank attached has to stand too), the depth may not
    # shrink
    assert narrow[1] >= shape[1], (narrow, shape)
    # ...and an exact lot an adopted arrangement asked for still comes back unrounded
    exact = dc._attached_lot(decl, 10, flanks, True, area=area, shape=shape,
                             want=(6, 16))
    assert exact == (6, 16), exact
    return (f"asked for {area} columns with no shape, `_attached_lot` returns "
            f"{square[0]}x{square[1]} -- a square plot of {square[0] * square[1]}; asked "
            f"for the same {area} columns in the shape `dense` asks for "
            f"({shape[0]}x{shape[1]}, because {lot['shape_why']}), it returns "
            f"{narrow[0]}x{narrow[1]} of {narrow[0] * narrow[1]}; an adopted "
            f"6x16 comes back unrounded")


# --------------------------------------------- F1e. variation has physical room

@case
def t_f1e_the_lot_the_crowded_ring_now_gets_stands_more_than_one_height():
    """The same type, the same language, on the lot the fabric actually hands it.

        The retained section's ninety-six row houses stood on 5x6 and 5x7 pads, **every one
        of them one storey and six blocks high**, half of them having asked for two. This
        asks the type for the same three storey counts and the same three fronts on the pad
        the new dense terrace gives it, and on the two pads the section gave it as the
        control. Nothing here is decoration or substitution: one type, one voice, one seed
        set, and the only thing that changed is the lot.
        
    """
    f = dense_terrace()
    w, ld = int(f["lot"][0]), int(f["lot"][1])
    # **The pad, not the lot.** A house sees the ground `site()` hands it, and a party-
    # walled lot keeps no inset on its flanks -- so the fabric's `w`x`ld` lot is a
    # different pad from the same rectangle standing free. `probe_build` builds on a
    # free lot, so the pad is computed here and a free plot that gives the *same pad* is
    # what the type is stood on: `type_needs._plot_for`'s own translation, and the
    # reason this case reads 6x9 where the fabric reads 6x13.
    pad = Builder.pad_extent({"kind": "plot", "x0": 0, "z0": 0,
                              "x1": w - 1, "z1": ld - 1,
                              "attached": ["west", "east"]})
    plot = lot_for_pad(*pad)
    was = Builder.pad_extent({"kind": "plot", "x0": 0, "z0": 0,
                              "x1": RETAINED["dense_attached_lot"][0] - 1,
                              "z1": RETAINED["dense_attached_lot"][1] - 1,
                              "attached": ["west", "east"]})
    got, bad = [], []
    for storeys in (1, 2, 3):
        for front in ("lattice", "screen", "open"):
            for seed in (1, 2):
                r = stand(plot[0], plot[1], storeys, front, seed=seed)
                assert r["ok"], (plot, storeys, front, r.get("why"))
                assert tuple(r["pad"]) == tuple(pad), (r["pad"], pad)
                got.append(r)
                if not r["inside"]:
                    bad.append((storeys, front, r["rect"], r["pad"]))
    heights = sorted({g["height"] for g in got if g["height"]})
    storeys = sorted({g["storeys"] for g in got if g["storeys"]})
    # the claim: more than one height, and more than one storey, in one language
    assert len(storeys) >= 2, got
    assert len(heights) >= 3, heights
    # ...and no house leaves its own pad, which a party wall cannot absorb
    assert not bad, bad
    # the control: the pads the section actually gave it carry one storey whatever is
    # asked, so the flat roofline was a fact about the lots and not about the type
    flat = []
    for pw, pd in RETAINED["section_pads"]:
        lot = lot_for_pad(pw, pd)
        for s in (1, 2, 3):
            r = stand(lot[0], lot[1], s, "screen", seed=1)
            assert r["ok"], (lot, s, r.get("why"))
            flat.append((r["storeys"], r["height"]))
    assert {s for s, _h in flat} == {1}, flat
    return (f"the dense terrace's {w}x{ld} lot gives a {pad[0]}x{pad[1]} pad with its "
            f"flanks attached, where the retained {RETAINED['dense_attached_lot'][0]}x"
            f"{RETAINED['dense_attached_lot'][1]} lot gave {was[0]}x{was[1]}; over "
            f"{len(got)} instances on that pad the same type stands {storeys} storey(s) "
            f"and {heights} blocks high -- {len(heights)} heights in one architectural "
            f"language, and every house inside its own pad. On the "
            f"{RETAINED['section_pads']} pads the retained section gave it: "
            f"{sorted({h for _s, h in flat})} blocks and one storey whatever is asked")


# --------------------------------------------- F2. an adopted form owes its court

@case
def t_f2_an_adopted_courtyard_block_publishes_a_court_with_no_courtyard_leaf():
    """`demand.required_by_part` and `section._courts`, on a district of row houses.

        The positive control is in the same case: give the district one court that holds and
        one that failed to stand, and the record still fails, because the failed one is a
        subject and is in the denominator by name.
        
    """
    place = {"kind": "city", "form": None, "voice": None,
             "defining_parts": [{"name": "lower_ring", "kind": "group",
                                 "family": "district", "relation": "throughout",
                                 "count": 1}],
             "districts": [{"name": "lower_ring_north", "kind": "district",
                            "defines": "lower_ring", "density": "dense",
                            "x0": 0, "z0": 0, "x1": 99, "z1": 99},
                           {"name": "upper_ring_north", "kind": "district",
                            "defines": "upper_ring", "density": "sparse",
                            "x0": 0, "z0": 200, "x1": 99, "z1": 299}],
             "leaves": [{"name": "lower_ring_north_a", "in": ["lower_ring"],
                         "type": "row_house", "kind": "plot"},
                        {"name": "lower_ring_north_b", "in": ["lower_ring"],
                         "type": "row_house", "kind": "plot"}]}
    # the obligation exists, and it is the **district's**, not any leaf's
    owed = demand.court_districts(place)
    assert "lower_ring_north" in owed, owed
    assert float(owed["lower_ring_north"]["share"]) > 0, owed
    # ...and the sparse ring, whose registered character asks for no courtyard block,
    # owes nothing: the rule is the adopted form and not the word "district"
    assert "upper_ring_north" not in owed, owed
    bound = demand.required_by_part(None, {"requirements": []}, place,
                                    parts=place["leaves"],
                                    declares={"row_house": ()})
    assert demand.COURT_TOKEN in (bound.get("lower_ring_north") or {}), bound
    assert bound["lower_ring_north"][demand.COURT_TOKEN] == [demand.COURT_FORM_ID], bound
    assert not (bound.get("lower_ring_north_a") or {}), bound
    # ...and with no court in the section at all the record names the subject and fails
    empty = section_mod._courts([], {}, place["districts"])
    m0 = empty["measured"]
    assert empty["status"] == "failed", empty
    assert m0["subjects"] == 1 and m0["asked"] == 0, m0
    assert m0["omitted"] == ["lower_ring_north"], m0
    # the gate's four keys are all present and mean what it reads them as
    for k in ("subjects", "asked", "omitted", "per_part"):
        assert k in m0, (k, sorted(m0))
    # ...and the positive control: one court holds, one did not stand. Still failed,
    # because a subject that did not stand is a subject.
    rows = [{"part": "lower_ring_north_c0", "type": "court_small", "stood": True,
             "emitted": {"features": {"courtyard": True},
                         "rects": {"courtyard": [0, 0, 2, 2]}}},
            {"part": "lower_ring_north_c1", "type": "court_small", "stood": False,
             "emitted": {}}]
    usable = {("lower_ring_north_c0", "court_accessible"):
              {"holds": True, "method": "observed", "why": "entered, open, enclosed",
               "evidence": {"courts": [{"cells": 9}]}}}
    got = section_mod._courts(rows, usable, place["districts"])
    m = got["measured"]
    assert got["status"] == "failed", got
    assert m["subjects"] == 3, m
    assert m["asked"] == 2 and m["held"] == 2, m
    assert m["omitted"] == ["lower_ring_north_c1"], m
    # the district itself is affirmative off the court that holds in it...
    dist = next(g for g in m["per_part"] if g["subject"] == "district")
    assert dist["affirmative"] and dist["courts_held_in_it"] == 1, dist
    # ...and a held court reports the ground it measures, so "every predicate holds" and
    # "every court is 3x3" are the same sentence
    held = next(g for g in m["per_part"] if g["part"] == "lower_ring_north_c0")
    assert held["court_columns"] == 9, held
    assert m["court_columns_held"] == [9], m
    return (f"a dense district of two row houses and no courtyard-house leaf publishes "
            f"`{demand.COURT_TOKEN}` under `{demand.COURT_FORM_ID}` from "
            f"{owed['lower_ring_north']['from']}; with no court in the section the record "
            f"is {empty['status']} at {m0['asked']} asked of {m0['subjects']} subject(s) "
            f"and names {m0['omitted']} omitted -- the retained record said "
            f"`{RETAINED['courts_status']}` at {RETAINED['courts_asked']} asked of "
            f"{RETAINED['courts_subjects']} subject(s). With one court holding and one "
            f"that did not stand: {got['status']}, {m['asked']} of {m['subjects']} "
            f"subject(s) asked, {m['held']} hold, omitted {m['omitted']}, and the court "
            f"that holds measures {m['court_columns_held']} column(s)")


# --------------------------------------------- F3. the lots stand on ground

#: The retained section's hardest district and the ring level its terrace stands on --
#: 35.4% of its rectangle can carry a building, which is the case that decides whether
#: "do not lay on refused ground" is a rule or a way of refusing a district.
HARD = ("middle_ring_north_west", 79)


def hard_district():
    """(district, part, decls, place, spec, ground) for `HARD`, off the retained
    section's own artifacts and its own baseline volume. Read, never written."""
    from ethoslm import offline
    d = os.path.join(ROOT, "out", "nb-city")
    vol_p = os.path.join(d, "world.before-plateau.npz")
    plan_p = os.path.join(d, "plan.place.json")
    if not (os.path.exists(vol_p) and os.path.exists(plan_p)):
        raise Skip("no out/nb-city baseline volume and plan to read ground from")
    place = json.load(open(plan_p))
    spec = json.load(open(os.path.join(d, "place.json")))
    x = next((q for q in place["districts"] if q["name"] == HARD[0]), None)
    if x is None:
        raise Skip(f"no district {HARD[0]} in the retained section")
    part = {p["name"]: p for p in spec["defining_parts"]}[x["defines"]]
    role = spec_mod.district_role(spec, x)
    _t, mine = placeplan.types_card(None, spec.get("form"), role)
    ground = placeplan.district_ground(x, offline.load_volume(vol_p),
                                       ring_level=HARD[1])
    return x, part, mine, place, spec, ground


@case
def t_f3_the_compiler_lays_its_lots_on_ground_that_can_carry_a_building():
    """`district_compile` against `ethoslm.feasible`'s mask, on the section's own terrain.

        The review's first finding at the level it bites. `placeplan.developable_columns`
        subtracts the ground that cannot be prepared, so a district's **count** is honest;
        the block grid was still laid over the whole rectangle, so the count was right and
        the lots were on the lake. The arterial's band and a standing part's clearance are
        ground this compiler has always refused to build on, and water and a grade the
        terrace cannot reach are the same kind of fact.

        Two things have to hold together and only one of them is obvious:

          * **no plot most of whose columns the mask refuses**, which is the rule;
          * **the district still compiles**, which is the constraint. On this district 35.4%
            of the rectangle can carry a building; a bar of "every column" lays nothing at
            all and turns a district that should come out smaller into one that comes out
            refused.
        
    """
    x, part, decls, place, spec, g = hard_district()
    share = float(g["feasible_columns"]) / float(g["columns"])
    assert g["measured"], g
    assert dc.LAYS_ON_FEASIBLE_GROUND, "the compiler does not declare that it reads the mask"
    plain, r0 = dc.compile_district(dict(x), part, place, decls, spec=spec, seed=1)
    onto, r1 = dc.compile_district(dict(x, ground=g, level=g.get("level")),
                                   part, place, decls, spec=spec, seed=1)

    def leaves(f):
        return [p for q in f.get("quarters") or [] for p in q.get("plots") or []]

    def feasible_share(p) -> float | None:
        c = feasible.cover(g, (p["x0"], p["z0"], p["x1"], p["z1"]))
        if c["feasible_columns"] is None or not c["columns"]:
            return None
        return c["feasible_columns"] / float(c["columns"])

    def off_ground(f):
        out = []
        for p in leaves(f):
            frac = feasible_share(p)
            if frac is not None and frac < dc.GROUND_FOUNDED:
                out.append((p["name"], p.get("kind", "plot"), round(frac, 3)))
        return out

    bad0, bad1 = off_ground(plain), off_ground(onto)
    # **the district still compiles, and on this mesa it now lays no house at all.** The
    # block design round. The two lots this case used to keep were kept by a *share*:
    # half a lot and half its skirt over the mask, which the audit's first cause is
    # exactly about -- "the critical cells can lie in the rejected half of both
    # rectangles". With the hard conditions beside it (`ENTRY_RUN`: a run of named
    # columns on the lot's own street face that the design prepares, outside for the
    # doorstep and inside for the floor it opens onto; `pad_founded`: a rectangle the
    # size this type needs, all of it prepared) both of them go, and the record says
    # which rule took each one. A mesa whose shelf cannot carry a house with a way into
    # it is open ground with an owner, which is what this district's own
    # `undeveloped_share` of 0.64 says; laying two houses nobody can walk to on it was
    # the flattering answer.
    assert leaves(onto), r1
    assert r1["dropped"]["entrance"] or r1["dropped"]["pad"], r1["dropped"]
    # ...and nothing it laid stands mostly on ground the design cannot prepare
    assert not bad1, bad1
    # ...where the same district laid without the mask does
    assert bad0, ("the control laid nothing on refused ground, so this district is not "
                  "the counterexample it was chosen to be", bad0)
    # ...and it is fewer lots, by the number the record names
    assert r1["lots"] < r0["lots"], (r1["lots"], r0["lots"])
    assert r1["lots_refused_for_ground"] > 0, r1
    assert r1["ground_measured"] and r0["ground_measured"] is False, (r0, r1)
    assert r1["dropped"]["ground"] == r1["lots_refused_for_ground"], r1["dropped"]
    # every refusal is named by the rule that made it, and the four are kept apart
    assert set(r1["dropped"]) == {"arterial", "standing", "ground", "entrance", "pad"}, \
        r1["dropped"]
    shares = sorted(s for s in (feasible_share(p) for p in leaves(onto))
                    if s is not None)
    return (f"{HARD[0]} terraces to y={g['level']} and {g['feasible_columns']} of its "
            f"{g['columns']} columns ({share:.1%}) can carry a building; compiled "
            f"without the mask it lays {r0['lots']} lot(s) and {len(leaves(plain))} "
            f"leaves, {len(bad0)} of them mostly on ground the design cannot prepare "
            f"(worst {min(b[2] for b in bad0):.0%} feasible). With the mask it still "
            f"compiles: {r1['lots']} lot(s), {len(leaves(onto))} leaves, "
            f"{r1['lots_refused_for_ground']} refused for ground and "
            f"{r1['ground_refused_columns']} infeasible column(s) in the rectangle, "
            f"and no leaf below the {dc.GROUND_FOUNDED:.0%} bar -- the leaves stand on "
            f"{shares[0]:.0%}..{shares[-1]:.0%} feasible ground; dropped "
            f"{r1['dropped']}")


# ------------------------------------------------------------------ the runner

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok = fail = skipped = 0
    t0 = time.time()
    for fn in CASES:
        name = fn.__name__[2:]
        if only and only not in name:
            continue
        try:
            said = fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            skipped += 1
            continue
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            fail += 1
            continue
        ok += 1
        print(f"ok   {name}: {said}")
    print(f"\n{ok}/{ok + fail} spatial form cases pass"
          + (f", {skipped} skipped" if skipped else "")
          + f" ({time.time() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
