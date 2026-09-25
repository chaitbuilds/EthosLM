"""**The plan's attachment, through the probe and into the house that stands.**

    $PY scripts/test_delivery_form.py

The neighbourhood delivery round, worker `form`. The losses these cases are registered
against were all measured first, on the retained evidence and on the block build:

 1. **One lot, three pads, three answers.** `Builder._insets` drops a plot's inset on a
    side the plan says is attached, so a 6x13 lot is a 4x11 pad standing free and a 6x9
    pad between two party walls. `construction.probe_build` built every probe free on
    all four sides, so `envelope.lot_for` and `demand.storeys_admitted` answered the
    wrong question: every terrace lot in `bb3a83d6ace538aa` recorded
    `storeys_admitted: 1` and every house in the crowded ring stood one storey on a lot
    deep enough for two. The counterexample is the free-flank answer and it is kept:
    that lot really does admit one storey when nothing stands beside it.
 2. **The rotated district.** A street running along z fronts its lots east or west, so
    the flanks are the north and south sides. The flank is the pair perpendicular to the
    front, never a fixed pair of compass names.
 3. **The generated production call.** Composed with `pipeline.instantiated_source` and
    executed with `offline.run_program`, which is exactly what `stages_build`
    does -- not a direct call to the type. The pad reaches the plot's edge on an
    attached flank and is inset on a free one, and the house that stands is as wide as
    its pad.
 4. **The roof cavity.** With the attachment carried, `out/nd-block` stopped the round on
    `E003 lower_ring_north_2_b3_1_00: a room of 15 cells at (-5706,72,656) cannot be
    walked into`, with `E011` beside it, five times, plus eleven `E004`. A one-storey
    house between taller neighbours had a pocket under their oversailing eaves. Built
    here as a row through the generated production calls and linted, with a hand-built
    sealed box in the same volume as the control that the instrument still fires.
 5. **The rear veranda and the declared front.** The engawa's way in was a `doorway()`
    call the library refuses by design, seventy-six times in the delivered section and
    read by an independent reader as seventy-six front doors in the wrong place. And
    `part["front"]` is the plan's word, now carried into `site()`; the doorstep
    inference is the fallback and says when it was used.

Nothing here needs a cached world or a model call.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ethoslm import (  # noqa: E402
    construction, demand, envelope, lint, offline, pipeline, stages,
)
from ethoslm.buildlib import Builder  # noqa: E402
from ethoslm.observe import Volume  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASES = []

#: The ground everything here stands on: one level plane, deep enough to plinth into.
#: Synthetic on purpose and for the same reason `type_needs.py` says -- what is being
#: measured is the pad a *lot* gives and the house that stands on it, and a slope would
#: confound that with `_site_pad`'s search for the flattest rectangle.
GROUND = 64

#: The voice the block was built in, so that what stands here is what stood there.
MAT = {"wall": "mud_brick", "footing": "cobblestone", "frame": "spruce",
       "roof": "deepslate_tile", "trim": "spruce", "floor": "packed_mud"}
ROOF = {"profile": [[1, 1]], "ends": "hip", "eave": "straight", "tiers": 1,
        "overhang": 1, "chimney": False}


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


def _flat(size: int, y: int = GROUND) -> Volume:
    """A plane of stone with its grass surface **at** `y`, air above. `test_compile`'s
    own fixture ground, so that the two read the same numbers."""
    y0 = y - 14
    codes = np.zeros((size, 70, size), np.uint16)
    codes[:, :y - y0, :] = 3
    codes[:, y - y0, :] = 1
    return Volume(0, y0, 0, codes, ["air", "grass_block", "dirt", "stone"])


def _network(plots: list):
    """A lane along each leaf's **declared front**, with the doorstep the circulation
        pass would reserve on it.

        Without this `site()` has no threshold and falls back to its own default corner, so
        the declared front and the reserved doorstep disagree on every part -- which is a
        property of the harness, not of production, and would make every case here measure
        the fallback instead of the decision. `stage_circulation` does this for real;
        `test_compile` case 7 asserts that what it reserves faces the leaf's own `front`.
        
    """
    from ethoslm.circulate import Network, Threshold
    cells, ths = {}, []
    for leaf, _seed, _params in plots:
        x0, z0, x1, z1 = leaf["x0"], leaf["z0"], leaf["x1"], leaf["z1"]
        mx, mz = (x0 + x1) // 2, (z0 + z1) // 2
        # `door_at` moves the reserved doorstep along the front, which is the one thing
        # case 6 has to vary. It is not in `pipeline.PART_GEOMETRY`, so it reaches the
        # network and never the type -- the type only ever sees what `site()` hands it.
        if leaf.get("door_at") is not None:
            mx = mz = int(leaf["door_at"])
        side = leaf.get("front") or "north"
        if side in ("north", "south"):
            lz = z0 - 1 if side == "north" else z1 + 1
            step = -1 if side == "north" else 1
            for x in range(x0 - 2, x1 + 3):
                for k in (0, 1):
                    cells[(x, lz + step * k)] = {"y": GROUND, "rank": 1, "face": None}
            ths.append(Threshold(id=leaf["label"], x=mx, z=lz, y=GROUND,
                                 facing=("south" if side == "north" else "north"),
                                 door=(mx, GROUND, z0 if side == "north" else z1)))
        else:
            lx = x0 - 1 if side == "west" else x1 + 1
            step = -1 if side == "west" else 1
            for z in range(z0 - 2, z1 + 3):
                for k in (0, 1):
                    cells[(lx + step * k, z)] = {"y": GROUND, "rank": 1, "face": None}
            ths.append(Threshold(id=leaf["label"], x=lx, z=mz, y=GROUND,
                                 facing=("east" if side == "west" else "west"),
                                 door=(x0 if side == "west" else x1, GROUND, mz)))
    return Network(cells=cells, thresholds=ths)


def _run(plots: list, *, size: int = 96) -> tuple:
    """Build these leaves the way `stages_build.instantiate_part` builds one.

        `plots` is `[(leaf, seed, params), ...]` where `leaf` carries the keys
        `pipeline.PART_GEOMETRY` names. Returns `(builder, volume, program source)`.

        The point of going the long way round -- compose the type's own source with the
        driver's `site()`/`build()` calls, write the program, execute it with
        `offline.run_program` -- is that the round asks for the decision verified *through a
        generated production call*. A direct `build(...)` would not exercise
        `PART_GEOMETRY`'s filter, which is where the attachment was being dropped.
        
    """
    decl = pipeline.load_type(os.path.join(ROOT, "types", "row_house.py"))
    instances = [(dict(leaf, kind="plot"), int(seed), dict(params or {}))
                 for leaf, seed, params in plots]
    src = pipeline.instantiated_source(decl["src"], instances, mat=MAT, roof=ROOF)
    vol = _flat(size)
    with tempfile.TemporaryDirectory() as tmp:
        json.dump([{"label": leaf["label"], "x0": leaf["x0"], "z0": leaf["z0"],
                    "x1": leaf["x1"], "z1": leaf["z1"]} for leaf, _s, _p in instances],
                  open(os.path.join(tmp, "plots.json"), "w"))
        prog = os.path.join(tmp, "program.py")
        open(prog, "w").write(src)
        b = offline.run_program(prog, vol, network=_network(instances),
                                plots=stages._Registry(tmp), src=src)
    return b, stages.apply_pending(vol, b._pending), src


def _lint(b, built, plots: list) -> list:
    """The build family of the linter, over these plots and nothing else."""
    reg = [{"label": leaf["label"], "x0": leaf["x0"], "z0": leaf["z0"],
            "x1": leaf["x1"], "z1": leaf["z1"]} for leaf, _s, _p in plots]
    region = (min(p["x0"] for p in reg) - 8, min(p["z0"] for p in reg) - 8,
              max(p["x1"] for p in reg) + 8, max(p["z1"] for p in reg) + 8)
    ctx = lint.Context.build(built, plots=reg, region=region, fittings=b.fitting_cells)
    return pipeline.standard_report(ctx, reg).findings


def _leaf(label, x0, z0, x1, z1, attached=(), front="north", wall_alt=False):
    return {"label": label, "kind": "plot", "x0": x0, "z0": z0, "x1": x1, "z1": z1,
            "attached": list(attached), "front": front, "wall_alt": bool(wall_alt)}


def _demand(band=(1, 2)) -> dict:
    """A resolved demand for a `row_house` fabric, as `district_compile` holds one."""
    return {"part": "terrace", "types": ["row_house"], "params": {"storeys": band[0]},
            "storeys_band": [int(band[0]), int(band[1])], "required": [],
            "context": dict(demand.CONTEXT_DEFAULT)}


# ------------------------------------------------- 1. one lot, three flank counts

@case
def t_1_the_same_lot_admits_different_storeys_by_how_many_flanks_are_attached():
    """**The free-end condition, made measurable.** A 6x13 lot, three flank counts.

        The pads are `Builder._insets`'s and are arithmetic; the storeys are
        `demand.storeys_admitted`'s and are a probe of the real generator. The 0-flank
        answer is the counterexample and it is true: that lot standing free admits one
        storey, which is what the delivered section recorded for every one of its terrace
        lots because the probe was built free whatever the plan said.
        
    """
    lot = _leaf("one", 0, 0, 5, 12)
    pads, admits, keys = {}, {}, {}
    d = _demand()
    for n in (0, 1, 2):
        sides = construction.probe_flanks("north", n)
        assert len(sides) == n, (n, sides)
        pads[n] = tuple(Builder.pad_extent(dict(lot, attached=sides)))
        got = demand.storeys_admitted(d, 6, 13, flanks=n)
        admits[n] = got["storeys"]
        assert got["flanks"] == n, got
        keys[n] = envelope._key("row_house", {"storeys": 2}, ("storeys",), None, 1,
                                {"attached": n})
    # the pad is the plan's word made physical: no inset on a side a neighbour is on
    assert pads[0] == (4, 11) and pads[1] == (5, 11) and pads[2] == (6, 9), pads
    # ...and the storeys follow the pad and not the plot
    assert admits[0] == 1, admits          # the counterexample, and it is the truth
    assert admits[1] == 2 and admits[2] == 2, admits
    # **A certificate names what it was measured of.** Three questions, three keys.
    assert len(set(keys.values())) == 3, keys
    # a bare `attached: true` off a character is a fact about a fabric and not about a
    # lot, so it resolves to the free-standing floor and never to a pad a lot may not
    # have
    assert envelope.flanks({"attached": True}) == 0
    assert envelope.flanks({"attached": ["west", "east"]}) == 2
    assert envelope.flanks({"attached": 1}) == 1
    # an attached answer is read the way round it was measured
    need = {"lot_min": [5, 13]}
    assert envelope.fits((13, 5), need) and not envelope.fits((13, 5), need, oriented=True)
    return (f"a 6x13 lot is a {pads[0][0]}x{pads[0][1]} pad free, "
            f"{pads[1][0]}x{pads[1][1]} with one flank and {pads[2][0]}x{pads[2][1]} "
            f"with two; row_house admits {admits[0]}, {admits[1]} and {admits[2]} "
            f"storeys on them, under three distinct envelope keys")


# ------------------------------------------------- 2. the rotated district

@case
def t_2_attachment_is_applied_to_the_axis_the_lot_is_long_on():
    """**A street running along z fronts east or west.** The flanks are the pair
    perpendicular to the front, so the same lot turned ninety degrees attaches on north
    and south -- and the pad it gives is the same pad, turned.
    """
    turned = {}
    for front, want in (("north", ("west", "east")), ("south", ("west", "east")),
                        ("east", ("north", "south")), ("west", ("north", "south"))):
        assert tuple(construction.probe_flanks(front, 2)) == want, front
    # the lot along x with its flanks north and south is the lot along z with its flanks
    # west and east, turned: same pad, both axes
    along_z = Builder.pad_extent(_leaf("a", 0, 0, 5, 12, ["west", "east"], "north"))
    along_x = Builder.pad_extent(_leaf("b", 0, 0, 12, 5, ["north", "south"], "east"))
    assert tuple(along_z) == (6, 9) and tuple(along_x) == (9, 6), (along_z, along_x)
    turned["pads"] = [list(along_z), list(along_x)]
    # ...and the house built on the turned lot is as wide as its lot across its own
    # frontage, which for a west/east front is the z axis
    leaf = _leaf("rot", 20, 20, 43, 25, ["north", "south"], "east")
    b, built, _src = _run([(leaf, 7, {"storeys": 2})], size=96)
    sited = b.parts[-1]
    res = sited["build"]
    assert res.get("ok"), res
    assert sited["z0"] == leaf["z0"] and sited["z1"] == leaf["z1"], sited
    assert sited["x0"] > leaf["x0"] and sited["x1"] < leaf["x1"], sited
    got_w = int(res["width"])
    assert got_w == leaf["z1"] - leaf["z0"] + 1, (got_w, leaf)
    rects = res["emitted"]["rects"]["main"]
    assert rects[1] == leaf["z0"] and rects[3] == leaf["z1"], rects
    turned["house"] = {"width": got_w, "depth": int(res["depth"]),
                       "storeys": int(res["storeys"])}
    errs = [f for f in _lint(b, built, [(leaf, 7, {})]) if f.code.startswith("E")]
    assert not errs, [f"{f.code} {f.message}" for f in errs]
    return (f"the flanks are the sides perpendicular to the front on all four fronts; "
            f"pads {turned['pads']}; a lot long on x with its north and south flanks "
            f"attached stands a {turned['house']['width']} wide by "
            f"{turned['house']['depth']} deep house of {turned['house']['storeys']} "
            f"storey(s), wall on the plot's edge, and lints clean")


# ------------------------------------------------- 3. the generated production call

@case
def t_3_the_generated_production_call_fills_the_lot_on_its_attached_flanks():
    """**Through `instantiated_source`, as `stages_build.instantiate_part` does.**

        The attached flank has no inset and the house's wall is on the plot's edge; the free
        flank keeps its inset and the house stays inside it. The free-flank half is the
        control: it is what every leaf in the delivered section got, whatever the plan said.
        
    """
    mid = _leaf("mid", 26, 20, 31, 43, ["west", "east"], "north")
    end = _leaf("end", 32, 20, 37, 43, ["west"], "north")
    free = _leaf("free", 44, 20, 49, 43, [], "north")
    rows = [(mid, 11, {"storeys": 2}), (end, 12, {"storeys": 2}),
            (free, 13, {"storeys": 2})]
    b, _built, src = _run(rows, size=96)
    # the program really is the composed one, and it really carries the attachment
    assert "attached" in src and "'front': 'north'" in src, src[-400:]
    said = {}
    for leaf, _seed, _p in rows:
        sited = [p for p in b.parts if p.get("label") == leaf["label"]][-1]
        res = sited["build"]
        assert res.get("ok"), (leaf["label"], res)
        att = set(leaf["attached"])
        # the pad reaches the plot's edge on the attached flanks and not on the free
        # ones
        assert (sited["x0"] == leaf["x0"]) == ("west" in att), (leaf["label"], sited)
        assert (sited["x1"] == leaf["x1"]) == ("east" in att), (leaf["label"], sited)
        pad_w = sited["x1"] - sited["x0"] + 1
        em = res["emitted"]
        rect = em["rects"]["main"]
        said[leaf["label"]] = {"pad_w": pad_w, "house_w": int(res["width"]),
                              "fills": bool(em["fills_pad"]), "verge": int(em["verge"]),
                              "wall_x": [rect[0], rect[2]]}
        if len(att) == 2:
            # **as wide as its lot.** A column short here is a hole in the terrace
            assert res["width"] == pad_w, (leaf["label"], res["width"], pad_w)
            assert rect[0] == leaf["x0"] and rect[2] == leaf["x1"], (leaf["label"], rect)
            assert em["verge"] > 0, em
        if not att:
            # the control: nothing attached, the inset stands on all four sides
            assert rect[0] > leaf["x0"] and rect[2] < leaf["x1"], (leaf["label"], rect)
            assert em["verge"] == 0, em
    assert said["end"]["wall_x"][0] == end["x0"], said["end"]
    return "; ".join(f"{k}: pad {v['pad_w']} house {v['house_w']} walls {v['wall_x']} "
                     f"verge {v['verge']}" for k, v in said.items())


# ------------------------------------------------- 4. the roof cavity

@case
def t_4_a_low_house_between_taller_neighbours_leaves_no_room_with_no_way_in():
    """**The regression `out/nd-block` stopped the round on.**

        Three lots of the block's own geometry, the middle one a storey lower than its
        neighbours, built through the generated production calls and linted. Before the
        verge this was `E003`/`E011` -- a one-column gallery the depth of the house, floored
        by the low roof, walled by the party walls and roofed by a neighbour's oversailing
        eave -- and `E004` where a roof tread read as down-slope into it.

        The control is in the same volume: a sealed box of exactly that shape, built by hand
        beside the row, which the linter does report. An instrument that cannot fail is not
        evidence that anything passed.
        
    """
    lots = [_leaf("r0", 20, 20, 25, 43, ["east"], "north"),
            _leaf("r1", 26, 20, 31, 43, ["west", "east"], "north"),
            _leaf("r2", 32, 20, 37, 43, ["west"], "north")]
    rows = [(lots[0], 21, {"storeys": 2}), (lots[1], 22, {"storeys": 1}),
            (lots[2], 23, {"storeys": 2})]
    b, built, _src = _run(rows, size=96)
    heights = []
    for leaf, _s, _p in rows:
        sited = [p for p in b.parts if p.get("label") == leaf["label"]][-1]
        assert sited["build"].get("ok"), (leaf["label"], sited["build"])
        heights.append(int(sited["build"]["storeys"]))
    assert heights == [2, 1, 2], heights      # the very shape that produced the cavity
    found = _lint(b, built, rows)
    errs = [f for f in found if f.code.startswith("E")]
    assert not errs, [f"{f.code} {f.message}" for f in errs]
    shut = [f for f in found if f.code in ("E003", "E011", "W009")]
    assert not [f for f in shut if f.code in ("E003", "E011")], \
        [f.message for f in shut]

    # ...and the control, in the same volume and read by the same checker: a 1x8 pocket
    # two blocks high with a lid on it, of the shape the verge closed.
    ctrl = built
    px, pz, py = 60, 24, GROUND + 1
    blocks = {}
    for z in range(pz - 1, pz + 9):
        for x in (px - 1, px + 1):
            for y in range(py, py + 3):
                blocks[(x, y, z)] = "stone_bricks"
    for z in (pz - 1, pz + 8):
        for y in range(py, py + 3):
            blocks[(px, y, z)] = "stone_bricks"
    for z in range(pz, pz + 8):
        blocks[(px, py + 2, z)] = "stone_bricks"
    ctrl = stages.apply_pending(ctrl, blocks)
    reg = [{"label": "control", "x0": px - 2, "z0": pz - 2, "x1": px + 2, "z1": pz + 9}]
    cctx = lint.Context.build(ctrl, plots=reg,
                              region=(px - 6, pz - 6, px + 6, pz + 14))
    control = [f.code for f in pipeline.standard_report(cctx, reg).findings
               if f.code in ("E003", "E011")]
    assert control, "the sealed control is not reported: the instrument is not live"
    return (f"a 2/1/2-storey row of three on 6x24 lots stands with "
            f"{len(errs)} own lint error(s) and no room with no way in; the sealed "
            f"control beside it reports {sorted(set(control))}")


# ------------------------------------------------- 5. the rear way in and the front

@case
def t_5_the_rear_veranda_is_entered_and_the_front_is_the_plans():
    """**Two type-level defects the audit named, both at a library boundary.**

        The engawa's way in was a `doorway()` call, which `TypeBuilder._doorway` refuses
        anywhere but the reserved arrival cell -- correctly, and seventy-six times in the
        delivered section, where an independent reader read the refusals as seventy-six
        front doors put in the wrong place. It is an `openings()` threshold now and the
        refusal is gone.

        And the front is the plan's where the doorstep is in it. The control is the part
        that carries no `front` at all: the doorstep inference still answers, and the record
        says which of the two was used.
        
    """
    leaf = _leaf("rear", 26, 20, 31, 43, ["west", "east"], "north")
    b, built, _src = _run([(leaf, 31, {"storeys": 1})], size=96)
    sited = b.parts[-1]
    res = sited["build"]
    assert res.get("ok") and res["rear"] >= 1, res
    refusals = [r for r in (getattr(b, "_type_refusals", None) or [])
                if r.get("call") == "doorway"]
    assert not refusals, refusals
    # the veranda is covered ground and is walkable from the house's own inside
    errs = [f for f in _lint(b, built, [(leaf, 31, {})]) if f.code.startswith("E")]
    assert not errs, [f"{f.code} {f.message}" for f in errs]
    assert res["emitted"]["front_from"] == "plan", res["emitted"]

    # the control: no `front` on the part, and the doorstep still answers
    ns = {"__name__": "__ethoslm_type__",
          "__file__": os.path.join(ROOT, "types", "row_house.py")}
    exec(compile(pipeline.load_type(os.path.join(ROOT, "types", "row_house.py"))["src"],
                 "row_house", "exec"), ns)                              # noqa: S102
    bare = {"x0": 0, "z0": 0, "x1": 5, "z1": 23, "door": [3, 23]}
    ax, s, how = ns["_front_edge"](bare)
    assert (ax, s, how) == ("z", -1, "door"), (ax, s, how)
    said = ns["_front_edge"](dict(bare, front="south"))
    assert said == ("z", -1, "plan"), said
    # ...and a declared front the doorstep is not in falls back, and says so
    over = ns["_front_edge"](dict(bare, front="north"))
    assert over == ("z", -1, "door_over_plan"), over
    return ("the rear threshold is built and no doorway() refusal is recorded; the "
            "front is the plan's where the doorstep is in it, the doorstep's where "
            "there is no plan, and reported as `door_over_plan` where they disagree")


# ------------------------------------------------- 6. the row's end

@case
def t_6_a_rows_end_is_entered_wherever_its_doorstep_was_reserved():
    """**The free end condition, at every column of its own frontage.**

        The section build's one remaining construction error, and it is an end lot:

            E008 the doorway reserved for lower_ring_north_2_b0_0_40 at (-5828,67,621)
                 cannot be walked into off its own threshold

        A 6x9 lot attached on one flank insets to a 5x7 pad; `_sizes` gives the side veranda
        the column a free flank allows; and the four that remain, pinned to the party wall,
        put the house's **corner** on the reserved doorstep -- three courses of corner post
        where the door should be, and `Nav.stance_near` at the door cell is None.

        So the doorstep is swept across every column of the frontage, on both end
        conditions, on the middle condition and on a detached lot, and a stance has to exist
        at the cell `site()` reserved. The counterexample is in the record: before `held()`
        this failed at exactly one column of each end -- the one the pin's corner landed
        on -- and nowhere else, which is why eighty-three of eighty-four doors on the
        section were fine.
        
    """
    lot = (20, 20, 25, 28)                     # 6 x 9, the section leaf's own shape
    widths, bad = {}, []
    for att in (["east"], ["west"], ["west", "east"], []):
        for dx in range(lot[0], lot[2] + 1):
            leaf = _leaf("end", *lot, att, "north")
            leaf["door_at"] = dx
            b, built, _src = _run([(leaf, 5, {"storeys": 1})], size=96)
            sited = b.parts[-1]
            res = sited["build"]
            assert res.get("ok"), (att, dx, res)
            dcx, dcz = sited["door"]
            from ethoslm import observe
            st = observe.Nav(built).stance_near(dcx, dcz, sited["floor_y"] + 1, tol=1)
            if st is None:
                bad.append((tuple(att), dx, sited["door"]))
            # the party wall is never given up to get there
            rect = res["emitted"]["rects"]["main"]
            if "west" in att:
                assert rect[0] == sited["x0"], (att, dx, rect, sited["x0"])
            if "east" in att:
                assert rect[2] == sited["x1"], (att, dx, rect, sited["x1"])
            widths.setdefault(tuple(att), set()).add(int(res["width"]))
    assert not bad, bad
    # ...and the slack is spent only where it is needed: an end still varies its width
    assert len(widths[("east",)]) > 1 and len(widths[("west",)]) > 1, widths
    assert widths[("west", "east")] == {6}, widths          # no slack, none spent
    return ("24 doorsteps swept across the frontage of a 6x9 lot on four attachments -- "
            "both ends, the middle and detached -- every one of them walkable at the "
            "cell site() reserved, every party wall still on its plot's edge; end "
            f"frontages {sorted(widths[('east',)])} and {sorted(widths[('west',)])} "
            f"columns, a middle always {sorted(widths[('west', 'east')])}")


def main():
    ok = bad = 0
    for name, fn in CASES:
        try:
            says = fn()
        except Exception as e:                       # noqa: BLE001 -- reported
            bad += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        ok += 1
        print(f"ok   {name}: {says}")
    print(f"\n{ok}/{ok + bad} delivery form cases pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
