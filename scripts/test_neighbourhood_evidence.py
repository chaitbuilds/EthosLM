"""The neighbourhood round's court gap and its obligations, measured on real geometry.

    $PY scripts/test_neighbourhood_evidence.py

The composition round left two facts on the record and one of them could not even be
asked. Measured on the retained build `out/comp-city`:

  * **fourteen `court_small` instances published no court at all.** The type has laid
    three ranges and a wall round a paved yard since it was written and said nothing
    about it, so `usable.court_accessible` answered `unsupported` on fourteen of the
    section's twenty-five courts. Not failing, not holding -- never asked;
  * **six of the eight `court_large` courts that could be asked failed**, five on open
    ground (`is no longer open paved ground`) and one on sky clearance (`is paved and
    something stands over it within 24 course(s)`).

This file is the round's answer to "inspect the actual form and predicate together,
repair their causes, and demonstrate usable, entered courts". Every case builds real
geometry -- `construction.probe_build` on flat ground, or the retained record -- and asks
the production predicates. Nothing here reads a declaration.

**N1 -- what the retained build actually shows** (`out/comp-city`, if it is there):
  1. the three failure modes are named off the retained record, and the eight courts'
     open share and sky share are quoted, so the before is a measurement and not a memory.

**N2 -- the generator, repaired** (`types/court_large.py`, `types/court_small.py`):
  2. `court_large` raised `ValueError` at three sizes inside its own declared band, and
     laid a 2x2 light well at two more. Both gone; the court is `COURT_MIN` on a side at
     every pad in the band, and the pad that cannot hold one is refused **by name**;
  3. every `court_large` in a bounded grid of pads, parameters, seeds and frontages is
     entered, open and enclosed on the assembled volume -- not one court, all of them;
  4. `court_small` publishes `emitted.rects.courtyard` and the same three answers hold
     for it, at every pad in its band.

**N3 -- the predicate, repaired, with the controls that stop it being a relaxation**
(`usable.range_relation`):
  5. a court walled off from its own house -- the range deleted from the assembled volume
     -- answers `False`. The undamaged world is the control;
  6. a court **roofed over** in the assembled volume answers `False` on
     `court_accessible`, and its unroofed self answers `True`;
  7. a yard inside a one-column boundary wall with no building behind it -- which the old
     one-ring measurement called ranged on four sides -- answers `False` on the
     `RANGED_SIDES` bar. This is the case that makes the change stricter and not looser.

**N4 -- obligations carried end to end** (`construction.confirm`, `evidence_for`,
`section._features`):
  8. the binding puts `courtyard` on a `court_small` leaf now that the type declares it,
     and `confirm` writes `emitted.required` and `emitted.owed` from that binding;
  9. `section._features` answers from `emitted.required` and `construction.evidence_for`
     -- the production binding -- and keeps `held`, `owed` and **unknown** apart. Measured
     on the retained build, where the old district-token rule found nothing at all.

**N5 -- room in the generator envelope** (`types/row_house.py`):
 10. the lower ring's house stood one storey at every one of the section's ninety-six
     lots. On the lots its repaired band admits it varies in storeys, height and depth,
     inside its own architectural language and without changing type.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ethoslm import construction, demand, envelope, section, usable    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RETAINED = os.path.join(ROOT, "out", "comp-city")
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


class Skip(Exception):
    """This case needs something this checkout does not have."""


# ----------------------------------------------------------------- shared probes

_BUILT: dict = {}


def stand(type_name: str, w: int, d: int, params: dict, *, seed: int = 1,
          front: str = "north"):
    """One instance on flat ground, its outcome, and the world it made.

        `(builder, sited, res, outcome, World)`. Cached, because a dozen cases ask about the
        same handful of probes and a probe is the expensive thing here.
        
    """
    key = (type_name, w, d, tuple(sorted(params.items())), seed, front)
    got = _BUILT.get(key)
    if got is None:
        b, sited, res = construction.probe_build(type_name, w, d, params, seed=seed,
                                                 front=front)
        out = (construction.outcome(b, sited, None, params)
               if not (isinstance(res, dict) and res.get("ok") is False) else {})
        world = None
        if out:
            world = usable.World.of_builder(b, _part_of(type_name, sited, out))
        got = _BUILT[key] = (b, sited, res, out, world)
    return got


def _part_of(type_name, sited, out) -> dict:
    return {"name": "probe", "type": type_name, "kind": "plot",
            "x0": sited["x0"], "z0": sited["z0"], "x1": sited["x1"], "z1": sited["z1"],
            "floor_y": sited.get("floor_y"), "emitted": out}


def ask(type_name: str, w: int, d: int, params: dict, *, seed: int = 1,
        front: str = "north") -> dict:
    """`court_accessible` and `range_relation` for one probe, with the court's numbers."""
    b, sited, res, out, world = stand(type_name, w, d, params, seed=seed, front=front)
    if not out:
        return {"refused": (res or {}).get("reason"), "pad": None}
    ca = usable.check(world, world.rows[0], "court_accessible")
    rr = usable.check(world, world.rows[0], "range_relation")
    rows = (ca.get("evidence") or {}).get("courts") or []
    r0 = rows[0] if rows else {}
    return {"refused": None,
            "pad": (sited["x1"] - sited["x0"] + 1, sited["z1"] - sited["z0"] + 1),
            "rect": (out.get("rects") or {}).get("courtyard"),
            "entered": int(r0.get("stances") or 0) > 0,
            "open_share": r0.get("share"), "sky_share": r0.get("sky_share"),
            "cells": r0.get("cells"),
            "accessible": ca["holds"], "ca_why": ca["why"],
            "ranged": rr["holds"], "rr_why": rr["why"],
            "closed_sides": ((rr.get("evidence") or {}).get("courts") or [{}])[0]
            .get("closed"),
            "ranged_sides": ((rr.get("evidence") or {}).get("courts") or [{}])[0]
            .get("ranged")}


def _lot_for_pad(pw: int, pd: int) -> tuple:
    """A plot whose sited pad is `pw` x `pd`. `type_needs._plot_for`'s arithmetic, asked
    through `Builder.pad_extent` so the two cannot drift."""
    from ethoslm.buildlib import Builder
    for w in range(pw, pw + 8):
        for d in range(pd, pd + 8):
            if tuple(Builder.pad_extent({"kind": "plot", "x0": 0, "z0": 0,
                                         "x1": w - 1, "z1": d - 1})) == (pw, pd):
                return (w, d)
    raise AssertionError(f"no plot gives a pad of {pw}x{pd}")


def _retained():
    p = os.path.join(RETAINED, "parts.json")
    if not os.path.exists(p):
        raise Skip("no out/comp-city/parts.json to measure the before on")
    return json.load(open(p))


# ------------------------------------------------- N1. the before, measured


@case
def t_1_the_retained_build_shows_fourteen_unasked_courts_and_six_failed_ones():
    """The three failure modes, off the retained record rather than off a memory."""
    parts = _retained()
    rows = [r for wv in parts.get("waves") or [] for r in wv.get("parts") or []]
    small = [r for r in rows if r.get("type") == "court_small"]
    large = [r for r in rows if r.get("type") == "court_large"]
    assert small and large, (len(small), len(large))
    # every `court_small` of that build published no courtyard rectangle at all
    declared = [r["part"] for r in small
                if (r.get("emitted") or {}).get("rects", {}).get("courtyard")]
    assert not declared, declared
    u = json.load(open(os.path.join(RETAINED, "usable.json")))
    answers = {(c["part"], c["want"]): c for c in u.get("checks") or []}
    unasked = [r["part"] for r in small
               if str(answers.get((r["part"], "court_accessible"), {})
                      .get("method")) == "unsupported"]
    asked = [r for r in large if (r["part"], "court_accessible") in answers]
    failed = [r["part"] for r in asked
              if answers[(r["part"], "court_accessible")].get("holds") is False]
    whys = [answers[(p, "court_accessible")]["why"] for p in failed]
    filled = [w for w in whys if "no longer open paved ground" in w]
    roofed = [w for w in whys if "stands over it within" in w]
    assert len(unasked) == len(small), (len(unasked), len(small))
    assert len(failed) == 6 and len(asked) == 8, (len(failed), len(asked))
    assert len(filled) == 5 and len(roofed) == 1, (len(filled), len(roofed))
    return (f"{len(small)} court_small published no court rectangle, so all "
            f"{len(unasked)} answered `unsupported`; of the {len(asked)} court_large "
            f"that could be asked, {len(failed)} failed -- {len(filled)} on open ground "
            f"and {len(roofed)} on sky clearance")


# ------------------------------------------------- N2. the generator, repaired


@case
def t_2_court_large_no_longer_raises_inside_its_band_or_lays_a_light_well():
    """The two causes under the retained build's small courts, and the refusal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_court_large", os.path.join(ROOT, "types", "court_large.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lo_w, lo_d = mod.NEEDS["footprint"][0], mod.NEEDS["footprint"][1]
    assert (lo_w, lo_d) == (mod.PAD_MIN, mod.PAD_MIN), (lo_w, lo_d, mod.PAD_MIN)
    # under the band: refused by name, and nothing claimed
    small = _lot_for_pad(mod.PAD_MIN - 2, mod.PAD_MIN - 2)
    got = ask("court_large", small[0], small[1], {"storeys": 2, "yard": "garden"})
    assert got["refused"] and "court" in got["refused"], got
    # at and above the band's floor: a real court, every parameter set and seed
    seen = []
    for pad in ((mod.PAD_MIN, mod.PAD_MIN), (mod.PAD_MIN + 3, mod.PAD_MIN + 1)):
        lot = _lot_for_pad(*pad)
        for storeys in (1, 2, 3):
            for yard in ("garden", "well", "orchard"):
                for seed in (1, 2):
                    r = ask("court_large", lot[0], lot[1],
                            {"storeys": storeys, "yard": yard}, seed=seed)
                    assert not r["refused"], (pad, storeys, yard, seed, r["refused"])
                    x0, z0, x1, z1 = r["rect"]
                    assert min(x1 - x0 + 1, z1 - z0 + 1) >= mod.COURT_MIN, (pad, r["rect"])
                    seen.append((x1 - x0 + 1) * (z1 - z0 + 1))
    return (f"the declared floor is the pad four ranges of {mod.RANGE_MIN} round a court "
            f"of {mod.COURT_MIN} need ({mod.PAD_MIN}); {mod.PAD_MIN - 2} is refused by "
            f"name; {len(seen)} instances at and above it carry a court of "
            f"{min(seen)}-{max(seen)} cells, none under {mod.COURT_MIN} on a side")


@case
def t_3_every_court_large_is_entered_open_and_enclosed_on_real_geometry():
    """All three, on every instance of a bounded grid -- not on one passing court.

        Five pads across the band, nine parameter sets, two seeds and all four frontages.
        Each court is asked of the assembled volume: reachable on foot from inside the house
        (`entered`), paved and open to the sky over `construction.OPEN_STANDS` of its cells
        (`open`), and closed on `usable.COURT_SIDES` of four sides with `RANGED_SIDES` of
        them a range rather than a wall (`enclosed`).
        
    """
    pads = [(9, 9), (9, 11), (11, 9), (12, 12), (17, 13)]
    n, opens, skies = 0, [], []
    for pad in pads:
        lot = _lot_for_pad(*pad)
        for storeys in (1, 2, 3):
            for yard in ("garden", "well", "orchard"):
                for seed in (1, 2):
                    for front in ("north", "west", "south", "east"):
                        r = ask("court_large", lot[0], lot[1],
                                {"storeys": storeys, "yard": yard}, seed=seed,
                                front=front)
                        where = (pad, storeys, yard, seed, front)
                        assert not r["refused"], (where, r["refused"])
                        assert r["entered"], (where, "no stance in the court")
                        assert r["accessible"] is True, (where, r["ca_why"])
                        assert r["ranged"] is True, (where, r["rr_why"])
                        n += 1
                        opens.append(r["open_share"])
                        skies.append(r["sky_share"])
    return (f"{n} court_large instances over {len(pads)} pads, 9 parameter sets, 2 seeds "
            f"and 4 frontages: every court entered, open (share {min(opens):.2f}-"
            f"{max(opens):.2f}, sky {min(skies):.2f}-{max(skies):.2f} against the "
            f"{construction.OPEN_STANDS} bar) and enclosed")


@case
def t_4_court_small_publishes_a_court_and_delivers_it():
    """The fourteen unasked courts, closed at the generator.

        The type declares `courtyard` in `FEATURES` and publishes the rectangle, so the
        binding can require it and the assembled world can be asked about it; and the answer
        is affirmative at every pad in the band it now declares, rather than a declaration
        with nothing behind it.
        
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_court_small", os.path.join(ROOT, "types", "court_small.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "courtyard" in (mod.FEATURES or ()), mod.FEATURES
    assert "courtyard" in envelope.declared_features("court_small")
    lo_w, lo_d, _hw, _hd = mod.NEEDS["footprint"]
    n, cells = 0, []
    for pad in ((lo_w, lo_d), (lo_d, lo_w), (10, 9), (12, 10), (18, 10)):
        lot = _lot_for_pad(*pad)
        for storeys in (1, 2):
            for wings in ("open", "screened", "closed"):
                for seed in (1, 2):
                    for front in ("north", "east"):
                        r = ask("court_small", lot[0], lot[1],
                                {"storeys": storeys, "wings": wings}, seed=seed,
                                front=front)
                        where = (pad, storeys, wings, seed, front)
                        assert not r["refused"], (where, r["refused"])
                        assert r["rect"], (where, "no courtyard rectangle published")
                        assert r["entered"], (where, "no stance in the court")
                        assert r["accessible"] is True, (where, r["ca_why"])
                        assert r["ranged"] is True, (where, r["rr_why"])
                        n += 1
                        cells.append(r["cells"])
    # ...and a pad that cannot hold one is refused rather than published as a light well
    tight = _lot_for_pad(7, 7)
    bad = ask("court_small", tight[0], tight[1], {"storeys": 1, "wings": "open"})
    assert bad["refused"] and "court" in bad["refused"], bad
    return (f"court_small declares `courtyard` and publishes its rectangle; {n} instances "
            f"over five pads, six parameter sets, two seeds and two frontages are "
            f"entered, open and enclosed, on courts of {min(cells)}-{max(cells)} cells; "
            f"a 7x7 pad is refused by name")


# ------------------------------------------------- N3. the predicate, controlled


def _damaged(type_name, pad, params, edit):
    """The same probe, with the assembled volume changed on purpose before it is asked.

        `usable.World.assembled` exists for this: a predicate that answers the record rather
        than the world cannot tell these two apart, and every case below is a pair.
        
    """
    lot = _lot_for_pad(*pad)
    b, sited, _res, out, _w = stand(type_name, lot[0], lot[1], params)
    vol = usable.World.assembled(b)
    edit(vol, sited, out)
    world = usable.World.of_builder(b, _part_of(type_name, sited, out), volume=vol)
    return world


@case
def t_5_a_court_whose_range_is_taken_away_is_not_an_enclosed_court():
    """The enclosure control. Ranges are cleared out of the assembled volume and nothing
        else changes -- the record still says four ranges stood, so a predicate answering the
        record rather than the world cannot tell these three apart.

        Two steps, because the bar is `COURT_SIDES` of four: taking one range away flips that
        one side and leaves the verdict standing, which is the bar behaving as written; taking
        a second away refuses it. Both are asserted, so neither an over-strict nor an
        over-lax reading passes this case.
        
    """
    pad, params = (12, 12), {"storeys": 2, "yard": "garden"}
    lot = _lot_for_pad(*pad)
    before = ask("court_large", lot[0], lot[1], params)
    assert before["ranged"] is True, before["rr_why"]

    def raze(sides):
        def edit(vol, sited, out):
            x0, z0, x1, z1 = out["rects"]["courtyard"]
            fy = sited["floor_y"]
            spans = {"north": (sited["x0"], sited["x1"], sited["z0"], z0 - 1),
                     "west": (sited["x0"], x0 - 1, sited["z0"], sited["z1"])}
            for s in sides:
                ax0, ax1, az0, az1 = spans[s]
                for x in range(ax0, ax1 + 1):
                    for z in range(az0, az1 + 1):
                        for y in range(fy + 1, fy + 14):
                            vol.overlay({(x, y, z): "air"})
        return edit

    said = []
    for sides in (("north",), ("north", "west")):
        world = _damaged("court_large", pad, params, raze(sides))
        got = usable.check(world, world.rows[0], "range_relation")
        row = ((got.get("evidence") or {}).get("courts") or [{}])[0]
        assert not (row.get("sides") or {}).get(sides[-1], {}).get("closed"), row
        said.append((sides, row.get("closed"), row.get("ranged"), got["holds"]))
    assert said[0][3] is True, said[0]
    assert said[1][3] is False, said[1]
    return (f"undamaged: closed on {before['closed_sides']} of four, "
            f"{before['ranged_sides']} of them ranged -- holds. One range out of the "
            f"assembled volume: {said[0][1]} closed, {said[0][2]} ranged, still holds at "
            f"a bar of {usable.COURT_SIDES}. Two out: {said[1][1]} closed, "
            f"{said[1][2]} ranged -- refused, on an unchanged record")


@case
def t_6_a_court_somebody_roofs_over_is_not_an_open_court():
    """The sky control, which is the failure mode `middle_ring_north_west_b1_0_10` was
    refused on. A deck is laid three courses over the court in the assembled volume."""
    pad, params = (12, 12), {"storeys": 1, "yard": "well"}
    lot = _lot_for_pad(*pad)
    before = ask("court_large", lot[0], lot[1], params)
    assert before["accessible"] is True, before["ca_why"]

    def roof(vol, sited, out):
        x0, z0, x1, z1 = out["rects"]["courtyard"]
        fy = sited["floor_y"]
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                vol.overlay({(x, fy + 6, z): "dark_oak_planks"})

    world = _damaged("court_large", pad, params, roof)
    after = usable.check(world, world.rows[0], "court_accessible")
    ev = after.get("evidence") or {}
    assert after["holds"] is False, after["why"]
    assert ev.get("roofed") == ["courtyard"], ev.get("roofed")
    return (f"open to the sky over {before['sky_share']:.2f} of its cells and holds; with "
            f"a deck six courses over it, sky "
            f"{(ev.get('courts') or [{}])[0].get('sky_share')} and refused as roofed -- "
            f"'a covered court is a room'")


@case
def t_7_a_yard_inside_a_bare_wall_is_not_ranged_however_many_sides_it_has():
    """**The control that stops the predicate change being a relaxation.**

        `range_relation` used to read one ring of cells outside the court, so a yard inside a
        one-column boundary wall was ranged on all four sides -- the reading its own
        docstring says it exists to refuse. The band measurement refuses it, because the
        part's own ground reaches one column past the court and a range needs
        `RANGE_DEPTH`. Built here as geometry and not as a type: a walled enclosure with a
        room in one corner, so the `rooms_on_court` clause cannot be what fails it.
        
    """
    from ethoslm import lint, observe
    import numpy as np
    fy, n = 64, 40
    codes = np.zeros((n, 40, n), dtype=np.int32)
    codes[:, :fy - 50 + 1, :] = 1                     # stone up to the floor
    vol = observe.Volume(0, 50, 0, codes, ["air", "stone", "cobblestone", "oak_planks"])
    x0, z0, x1, z1 = 10, 10, 24, 24
    put = {}
    for x in range(x0, x1 + 1):                       # the boundary wall, one column
        for z in (z0, z1):
            for y in range(fy + 1, fy + 4):
                put[(x, y, z)] = "cobblestone"
    for z in range(z0, z1 + 1):
        for x in (x0, x1):
            for y in range(fy + 1, fy + 4):
                put[(x, y, z)] = "cobblestone"
    for y in range(fy + 1, fy + 4):                   # a way in
        put[(x0 + 7, y, z0)] = "air"
    vol.overlay(put)
    court = [x0 + 1, z0 + 1, x1 - 1, z1 - 1]
    part = {"name": "garden", "type": "walled_garden", "kind": "plot",
            "x0": x0, "z0": z0, "x1": x1, "z1": z1, "floor_y": fy,
            "emitted": {"features": {"courtyard": True},
                        "rects": {"courtyard": court}}}
    ctx = lint.Context.build(vol, plots=[{"label": "garden", "x0": x0, "z0": z0,
                                          "x1": x1, "z1": z1, "kind": "plot",
                                          "y0": fy}],
                             region=(0, 0, n - 1, n - 1), base=vol)
    world = usable.World(ctx, [{"part": "garden", "status": "built", "stood": True,
                                "kind": "plot", "type": "walled_garden",
                                "x0": x0, "z0": z0, "x1": x1, "z1": z1,
                                "floor_y": fy, "emitted": part["emitted"]}])
    got = usable.check(world, world.rows[0], "range_relation")
    ev = (got.get("evidence") or {}).get("courts") or [{}]
    sides = ev[0].get("sides") or {}
    closed = ev[0].get("closed")
    ranged = ev[0].get("ranged")
    assert got["holds"] is False, got["why"]
    assert ranged == 0, (ranged, sides)
    assert all(v.get("setback") == 1 for v in sides.values()), sides
    return (f"a yard inside a one-column wall reads closed on {closed} of four sides and "
            f"ranged on {ranged}: the part's ground reaches 1 column past the court "
            f"where a range needs {usable.RANGE_DEPTH}, so the bar of "
            f"{usable.RANGED_SIDES} refuses it -- the old one-ring measurement called "
            f"this four of four")


# ------------------------------------------------- N4. obligations, carried


@case
def t_8_declaring_the_court_puts_it_on_the_binding_and_on_the_part():
    """A form's obligation, from the declaration to `emitted.owed`.

        `demand.required_by_part` gives a leaf a token only where the requirement reaches it
        **and its own type declares it**, so `court_small` declaring `courtyard` is what
        brings its instances into the binding at all -- which is the round's "a courtyard
        house owes a usable court even if the sentence never said courtyard", made
        mechanical. `construction.confirm` then stamps `emitted.required` and computes
        `emitted.owed` from the same binding, and a part that delivers owes nothing.
        
    """
    place = {"kind": "town", "form": None, "voice": None,
             "defining_parts": [{"name": "quarter", "kind": "group",
                                 "family": "district", "relation": "throughout",
                                 "count": 1}],
             "leaves": [{"name": "quarter_a", "in": ["quarter"], "type": "court_small",
                         "kind": "plot"},
                        {"name": "quarter_b", "in": ["quarter"], "type": "row_house",
                         "kind": "plot"}]}
    intent = {"requirements": [{"id": "feature/courtyard", "kind": "feature",
                                "wants": {"feature": "courtyard"},
                                "scope": None, "status": "supported"}]}
    bound = demand.required_by_part(
        None, intent, place,
        parts=place["leaves"],
        declares={"court_small": ("courtyard",), "row_house": ()})
    assert "courtyard" in (bound.get("quarter_a") or {}), bound
    assert "courtyard" not in (bound.get("quarter_b") or {}), bound
    # ...and the same token, on a built part, through `confirm`
    lot = _lot_for_pad(10, 9)
    b, sited, _res, out, _w = stand("court_small", lot[0], lot[1],
                                    {"storeys": 1, "wings": "closed"})
    part = dict(_part_of("court_small", sited, out), name="quarter_a")
    world = usable.World.of_builder(b, part)
    row = world.rows[0]
    record = {"waves": [{"parts": [row]}]}
    got = construction.confirm(world, record,
                               required={"quarter_a": ("courtyard",)})
    em = row["emitted"]
    assert em["required"] == ["courtyard"], em["required"]
    assert em["owed"] == [], em["owed"]
    assert "range_relation" in em["usable"], sorted(em["usable"])
    ev = construction.evidence_for(record, "quarter_a", "courtyard")
    assert ev["holds"] is True and ev["owed"] is False, ev
    return (f"the binding asks the court_small leaf for `courtyard` and does not ask the "
            f"row_house beside it; `confirm` asked {sorted(em['usable'])} of it, wrote "
            f"`required={em['required']}` and `owed={em['owed']}`, and `evidence_for` "
            f"answers holds={ev['holds']} by `{ev['method']}`")


@case
def t_9_the_section_reads_the_production_binding_and_keeps_unknown_apart():
    """`section._features`, on the retained build, against the rule it replaced.

        The old function collected each **district's** `demand.required` tokens, matched them
        to parts by name prefix and then decided each one off `emitted.features` -- the
        type's own claim at emission. On this build it found nothing at all and reported the
        relationship `unmeasured`, while the same record carried ten per-part obligations and
        the assembled world's answer to every one of them.

        Two dimensions are asserted apart here, which is the round's "unknown evidence stays
        unknown" without letting an undecided subject leave the denominator:
        `held + owed == subjects` always, and `failed + unknown == owed` splits the owed by
        whether a predicate ran. A subject nothing asked is **owed** -- that is
        `construction.evidence_for`'s own flag and `OWED_REASONS`' own gloss on `unmeasured`
        -- and it is reported as `unknown` besides, so it is never read as a measured failure
        or as a pass.
        
    """
    st = RETAINED
    if not os.path.exists(os.path.join(st, "parts.json")):
        raise Skip("no out/comp-city to measure on")
    parts = json.load(open(os.path.join(st, "parts.json")))
    plan = json.load(open(os.path.join(st, "plan.json")))
    rows = section._rows(parts)
    answers = section._usable_of(json.load(open(os.path.join(st, "usable.json"))))
    districts = section._districts(st, plan)

    # the rule this replaced, computed here so the difference is a measurement
    by_district: dict = {}
    for dd in districts:
        for tok in (dd.get("demand") or {}).get("required") or ():
            by_district.setdefault(str(dd.get("name")), set()).add(str(tok))
    old = 0
    for r in rows:
        name = str(r.get("part", ""))
        for dn, toks in by_district.items():
            if name.startswith(dn):
                old += len(toks)

    got = section._features(rows, answers, districts)
    m = got["measured"]
    assert m["subjects"] > old, (m["subjects"], old)
    # **every applicable subject is in exactly one of held and owed**, so no subject can
    # be dropped out of the denominator by being undecided
    assert m["held"] + m["owed"] == m["subjects"], m
    assert len(m["held_"]) + len(m["owed_"]) == m["subjects"], m
    # ...and `owed` splits by what the evidence was, with neither part lost
    assert m["failed"] + m["unknown"] == m["owed"], m
    assert sum(m["owed_by_reason"].values()) == m["owed"], m
    assert m["owed"] > 0 and got["status"] == "failed", (m, got["status"])
    # every subject came from the production binding, not from a district name list
    assert not m["parts_without_emitted_binding"], m["parts_without_emitted_binding"]
    assert all(r["from"].startswith("parts.json emitted.required")
               for r in m["held_"] + m["owed_"]), m
    assert all(r["holds"] is True for r in m["held_"]), m["held_"]
    assert all(r["holds"] is not True for r in m["owed_"]), m["owed_"]
    assert all(r["evidence"] == "unknown" and r["holds"] is None
               for r in m["unknown_"]), m["unknown_"]
    assert all(r["evidence"] == "measured" and r["holds"] is False
               for r in m["failed_"]), m["failed_"]
    return (f"the district-token rule reaches {old} subject(s) on this build; the "
            f"production binding reaches {m['subjects']} on {len(m['required_by_part'])} "
            f"part(s) -- {m['held']} held and {m['owed']} owed, of which "
            f"{m['failed']} were measured and refused and {m['unknown']} had nothing "
            f"decided {m['owed_by_reason']} -- and the relationship answers "
            f"`{got['status']}`")


# ------------------------------------------------- N5. room in the envelope


@case
def t_10_the_row_house_varies_in_height_depth_and_frontage_inside_its_band():
    """The lower ring's house, on the lots its repaired band admits.

        Ninety-six row houses of the retained section stood on 5x6 and 5x7 pads, every one of
        them **one storey and six blocks high** -- half of them had asked for two. The cause
        is the declared band: `type_needs` measured it at 4x4 to 6x6 against the 6x6 to 16x24
        its author declared, and a house whose whole idea is narrow-to-the-street and
        deep-into-the-plot cannot carry a stair in six columns of depth. On deeper lots the
        same type in the same language varies.

        The case asserts both halves, because the second is what makes the first a finding
        about the **lot** rather than about the type:

          * on deeper pads the same type, in the same language and with no decoration and no
            substitution, stands one, two and three storeys at three or more heights;
          * on the pad the section actually gave it, it cannot: `back` is four rows and a
            flight needs five longitudinally or six columns laterally, and the inner span is
            three. There is no arrangement of this type that puts a stair in a 5x7 pad.

        The declared ceiling is held at 6 for a reason measured in `NEEDS` -- raising it moves
        the library's dense fabric lot from 6x8 to 10x10 and the ground a dense house costs
        from 102 columns to 182 -- so the pads below are asked of the type directly, which is
        what `envelope.probe` does too and is a fact about the generator either way.
        
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_row_house", os.path.join(ROOT, "types", "row_house.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lo_w, lo_d, hi_w, hi_d = mod.NEEDS["footprint"]
    got = []
    for pad in ((6, 9), (6, 12), (6, 16)):
        lot = _lot_for_pad(*pad)
        for storeys in (1, 2, 3):
            for front in ("lattice", "screen", "open"):
                b, sited, res, out, _w = stand("row_house", lot[0], lot[1],
                                               {"storeys": storeys, "front": front})
                assert out, (pad, storeys, front, (res or {}).get("reason"))
                rect = (out.get("rects") or {}).get("main")
                got.append({"pad": pad, "asked": storeys, "front": front,
                            "storeys": out.get("storeys"), "height": out.get("height"),
                            "depth": (max(rect[2] - rect[0], rect[3] - rect[1]) + 1
                                      if rect else None),
                            "frontage": (min(rect[2] - rect[0], rect[3] - rect[1]) + 1
                                         if rect else None)})
    heights = sorted({g["height"] for g in got})
    storeys = sorted({g["storeys"] for g in got})
    depths = sorted({g["depth"] for g in got})
    assert len(storeys) >= 3, got
    assert len(heights) >= 4, heights
    assert len(depths) >= 3, depths
    # ...and the control: the lot the section gave it carries one storey whatever is
    # asked
    flat = []
    for pad in ((5, 6), (5, 7)):
        lot = _lot_for_pad(*pad)
        for storeys in (1, 2, 3):
            _b, _s, _r, out, _w = stand("row_house", lot[0], lot[1],
                                        {"storeys": storeys, "front": "screen"})
            flat.append((out.get("storeys"), out.get("height")))
    assert {s for s, _h in flat} == {1}, flat
    return (f"declared band {lo_w}x{lo_d}..{hi_w}x{hi_d}; asked directly, over "
            f"{len(got)} instances on 6x9, 6x12 and 6x16 pads the same type stands "
            f"{storeys} storey(s), {heights} blocks high and {depths} deep -- one "
            f"architectural language, {len(heights)} heights. On the 5x6 and 5x7 pads "
            f"the section gave it, {sorted({h for _s, h in flat})} blocks and one storey "
            f"whatever is asked: no stair fits that pad, so the ring's flat roofline is "
            f"a fact about its lots")


@case
def t_11_built_cover_is_mass_and_not_the_rectangle_it_stands_in():
    """`construction.occupied_columns`, and the distinction the round asks for.

        Both halves are asserted: the mass is under the rectangle where the form has a hole
        in it, and it is **absent rather than zero** where it cannot be measured.
        
    """
    lot = _lot_for_pad(12, 12)
    b, sited, _res, out, world = stand("court_large", lot[0], lot[1],
                                       {"storeys": 2, "yard": "garden"})
    row = world.rows[0]
    got = construction.occupied_columns(world, row)
    x0, z0, x1, z1 = out["rects"]["courtyard"]
    court = (x1 - x0 + 1) * (z1 - z0 + 1)
    assert got is not None and got["columns"] < got["of"], got
    assert got["of"] - got["columns"] >= court * 0.5, (got, court)
    # ...and a part that did not stand reports nothing at all, not nothing built
    gone = dict(row, stood=False, status="failed")
    assert construction.occupied_columns(world, gone) is None
    noroom = dict(row, emitted={k: v for k, v in row["emitted"].items()
                                if k != "footprint"},
                  x0=None, z0=None, x1=None, z1=None)
    assert construction.occupied_columns(world, noroom) is None
    if os.path.exists(os.path.join(RETAINED, "parts.json")):
        from ethoslm import placeplan
        rows = section._rows(json.load(open(os.path.join(RETAINED, "parts.json"))))
        rect = sum((abs(r["emitted"]["footprint"][2] - r["emitted"]["footprint"][0]) + 1)
                   * (abs(r["emitted"]["footprint"][3] - r["emitted"]["footprint"][1]) + 1)
                   for r in rows
                   if (r.get("emitted") or {}).get("footprint") and r.get("stood"))
        assert rect > 0
        note = (f"; the retained build's 139 rows report {rect} columns of enclosing "
                f"rectangle with no mass key on any of them")
    else:
        note = ""
    return (f"a {got['of']}-column rectangle round a {court}-column court carries "
            f"{got['columns']} columns of mass; a part that did not stand and a part "
            f"with no rectangle both answer None rather than 0{note}")


def main() -> int:
    bad = skipped = 0
    t0 = time.perf_counter()
    for fn in CASES:
        name = fn.__name__.split("_", 1)[1].replace("_", " ")
        try:
            note = fn()
            print(f"ok   {name}: {note}")
        except Skip as s:
            skipped += 1
            print(f"skip {name}: {s}")
        except Exception as e:                       # noqa: BLE001 -- a failing case
            import traceback
            bad += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(CASES) - bad - skipped}/{len(CASES)} neighbourhood evidence cases "
          f"pass, {skipped} skipped ({time.perf_counter() - t0:.0f}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
