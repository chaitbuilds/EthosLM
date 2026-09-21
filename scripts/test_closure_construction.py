"""The closure round's construction outcomes, as measurements with positive controls.

    $PY scripts/test_closure_construction.py

Every case builds a real type on a flat offline volume through the same `site()` and
`build()` the parts stage runs, and reads the result off the blocks the builder emitted
(`ethoslm.construction`). A type's own account of what it delivered is checked against
that measurement and never stands in for it.

  1. The three-storey cottage counterexample: on a 9x9 and a 12x10 lot one storey
     stands; on a 28x12 control three do. Read off floor levels, not `params.storeys`.
  2. A requested feature survives after a real adjustment to its cause: the constraint
     names the lot the cottage needs, the cottage is rebuilt on it, three storeys stand.
  3. Every plot type the proofs use returns what survived, and the type's storeys agree
     with the measured storeys on a lot that carries them.
  4. A fallback that dropped a storey is still a house with a way in: the emitted small
     cottage is lint-clean for sealed rooms.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import construction as C  # noqa: E402

CASES = []


def case(fn):
    CASES.append((fn.__name__[2:], fn))
    return fn


@case
def t_a_the_cottage_counterexample_is_measured():
    got = C.probe_storeys("cottage", 3, (9, 9), (28, 12))
    assert got["small"]["planned"] == 3 and got["small"]["emitted"] == 1, got["small"]
    assert got["control"]["planned"] == 3 and got["control"]["emitted"] == 3, got["control"]
    mid = C.probe_storeys("cottage", 3, (12, 10), (28, 12))
    assert mid["small"]["emitted"] == 1, mid["small"]
    # the type's own account agrees and names the reduction
    b, sited, res = C.probe_build("cottage", 9, 9, {"storeys": 3, "outshot": "store"})
    em = res["emitted"]
    assert em["storeys"] == 1 and "storeys" in em["omitted"], em
    assert "storeys 3 -> 1" in (em["fallback"] or ""), em["fallback"]
    o = C.outcome(b, sited, None, {"storeys": 3, "outshot": "store"})
    assert o["storeys"] == 1 and o["measured"] and o["levels"] == [sited["floor_y"]], o
    assert "storeys" in o["omitted"], o
    return ("cottage storeys=3: 9x9 -> 1, 12x10 -> 1, 28x12 -> 3, read off floor "
            "levels; the type's record names the plan reduction")


@case
def t_b_a_lost_storey_names_the_lot_and_a_rebuild_delivers_it():
    leaf = {"name": "h", "type": "cottage", "kind": "plot", "x0": 0, "z0": 0,
            "x1": 11, "z1": 9, "seed": 1, "params": {"storeys": 3, "outshot": "store"}}
    b, sited, res = C.probe_build("cottage", 12, 10, leaf["params"], seed=1)
    o = C.outcome(b, sited, None, leaf["params"])
    assert o["storeys"] == 1, o
    c = C.constraint(leaf, None, o)
    assert c and c["what"] == "storeys" and c["owner"] == "layout", c
    w, d = c["needs"]["lot_min"]
    assert (w, d) != (12, 10) and w >= 12 and d >= 10, c
    b2, s2, r2 = C.probe_build("cottage", w, d, leaf["params"], seed=1)
    o2 = C.outcome(b2, s2, None, leaf["params"])
    assert o2["storeys"] == 3 and not o2["omitted"], o2
    # control: a lot that already delivers gets no constraint **about its storeys**. The
    # design round narrowed `_verify_rect`'s window to the courses a feature's own
    # storey holds (`FEATURE_COURSES`), and this lot turns out to lay no hearth. That is
    # a constraint about the fire and not about the height, and this case is about the
    # height.
    b3, s3, r3 = C.probe_build("cottage", 28, 12, leaf["params"], seed=1)
    o3 = C.outcome(b3, s3, None, leaf["params"])
    got3 = C.constraint(dict(leaf, x1=27, z1=11), None, o3)
    assert got3 is None or got3["what"] != "storeys", got3
    return (f"a 12x10 lot lost two storeys; the constraint names {w}x{d} and the "
            f"cottage rebuilt on it stands three storeys; 28x12 needs no more ground "
            f"for its storeys"
            + (f" (it does lose `{got3['what']}`, which is a different question)"
               if got3 else ""))


@case
def t_c_every_plot_type_says_what_survived_and_the_measurement_agrees():
    cases = [("cottage", 20, 14, {"storeys": 3, "outshot": "byre"}, None, 3),
             ("hall", 16, 14, {"storeys": 2, "dormers": 2, "use": "moot"}, None, 2),
             ("minka", 18, 16, {"storeys": 2, "plan": "wing"}, "japanese_minka", 2),
             ("row_house", 11, 18, {"storeys": 3}, "white_render_dark_frame", 3),
             ("townhouse", 10, 12, {"storeys": 3}, "white_render_dark_frame", 3),
             ("shop_house", 14, 16, {"storeys": 2}, "packed_earth_and_dark_tile", 2),
             ("farmstead", 14, 14, {"storeys": 2}, "japanese_minka", 2),
             ("workshop", 16, 16, {}, "drystone_and_thatch", 1),
             ("temple", 18, 18, {"storeys": 2, "plan": "hall"}, "japanese_temple", 2)]
    bad = []
    for (t, w, d, params, voice, want) in cases:
        b, sited, res = C.probe_build(t, w, d, params, voice=voice)
        em = res.get("emitted")
        o = C.outcome(b, sited, None, params)
        if not res.get("ok") or not isinstance(em, dict):
            bad.append((t, "no emitted record", res.get("reason")))
            continue
        for k in ("requested", "storeys", "attempt", "fallback", "omitted", "features"):
            if k not in em:
                bad.append((t, f"emitted lacks {k}"))
        if em["storeys"] != o["storeys"] or o["storeys"] != want:
            bad.append((t, f"type says {em['storeys']}, measured {o['storeys']}, "
                           f"wanted {want}"))
        if want == params.get("storeys", want) and em["omitted"] and \
                "storeys" in em["omitted"]:
            bad.append((t, "storeys reported omitted on a lot that carries them"))
    assert not bad, bad
    # ...and a reduction the type made is on its record, on a lot that forces it
    b, sited, res = C.probe_build("row_house", 9, 14, {"storeys": 3},
                                  voice="white_render_dark_frame")
    em = res["emitted"]
    assert em["storeys"] == 2 and "storeys" in em["omitted"] and "lot:" in em["fallback"], em
    b, sited, res = C.probe_build("shop_house", 14, 16, {"storeys": 3},
                                  voice="packed_earth_and_dark_tile")
    em = res["emitted"]
    assert em["storeys"] == 2 and "flights" in em["fallback"], em
    return ("9 plot types return `emitted`; the type's storeys and the measured storeys "
            "agree on lots that carry them, and a lot that cannot names the cause")


@case
def t_d_a_fallback_is_still_a_house_with_a_way_in():
    from ethoslm import lint
    bad = []
    for lot in ((9, 9), (12, 10)):
        b, sited, res = C.probe_build("cottage", lot[0], lot[1],
                                      {"storeys": 3, "outshot": "store"}, seed=4)
        vol = b._vol.overlay(b._pending)
        plot = {"label": "probe", "x0": sited["x0"] - 2, "z0": sited["z0"] - 2,
                "x1": sited["x1"] + 2, "z1": sited["z1"] + 2}
        ctx = lint.Context.build(vol, plots=[plot], region=(0, 0, 95, 95))
        for f in lint.lint(ctx).errors:
            if f.code in ("E003", "E011"):
                bad.append((lot, f.code, str(f)[:80]))
    assert not bad, bad
    return "the one-storey fallback on both small lots has no sealed room"


@case
def t_e_surfaces_are_charged_through_the_voice_and_unknown_stays_unknown():
    b, sited, res = C.probe_build("cottage", 20, 14, {"storeys": 2, "outshot": "byre"},
                                  voice="drystone_and_thatch")
    s = C.surfaces(b, sited)
    assert s["known"] and s["roles"].get("wall") and s["roles"].get("roof"), s
    assert s["exposed_wall_faces"] > 0 and sum(s["by_face"].values()) > 0, s
    b2, sited2, _ = C.probe_build("cottage", 20, 14, {"storeys": 2, "outshot": "byre"})
    s2 = C.surfaces(b2, sited2)
    assert s2["known"] and s2["roles"]
    return (f"{s['blocks']} blocks charged: {dict(sorted(s['roles'].items()))}; "
            f"{s['exposed_wall_faces']} exposed wall faces")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = fail = 0
    for name, fn in CASES:
        if only and only not in name:
            continue
        try:
            says = fn()
        except AssertionError as e:
            fail += 1
            print(f"FAIL {name}: {e}")
            continue
        except Exception as e:                      # noqa: BLE001 -- reported by name
            fail += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            continue
        ok += 1
        print(f"ok   {name:60} {says}")
    print(f"\n{ok}/{ok + fail} closure construction cases pass")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
