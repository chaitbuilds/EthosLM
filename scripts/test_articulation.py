"""The articulation measure's contract, on fixtures where the answer is known.

A new measurement is worth exactly as much as the cases that pin down what it does not
measure, and this one has two ways to be wrong that would both look like a result:

  1. **Roof pitch read as depth.** A pitched roof's outward surface steps back a block
     a course. A measure that sampled it would rank buildings by roof form and call it
     articulation, and every reading taken from it would be about roofs.
  2. **Anything off the wall plane read as relief.** Look through an unglazed window and
     the first solid cell is the far interior wall ten blocks back. Look at a projecting
     wing and the first solid cell is five blocks proud. Neither is the wall's surface,
     which is why the band around the modal plane is symmetric, and both halves of that
     are a case here. The asymmetric first version of this measure ranked the flattest
     of the four selection candidates highest, which is how the defect was found.

So the fixtures are a flat box and that same box altered one way at a time -- pilasters,
a recessed panel, a jettied upper storey, a gable, windows, a wing -- and each case names
which of those the measure must and must not respond to. The box is the control and it
must read 0.000: a measure that cannot say "this wall is flat" cannot say anything.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethoslm import articulation, observe  # noqa: E402

W, D, H = 13, 11, 7          # box footprint and wall height, on ground at y=0
X0, Z0 = 4, 4                # its corner inside a 21x21 pad of grass
PLOT = [{"label": "box", "x0": 0, "z0": 0, "x1": 20, "z1": 20}]


def box(*, pilasters=False, recess=False, jetty=False, gable=False, windows=False,
        wing=False, wall="stone_bricks"):
    """A hollow box on flat ground, altered one way at a time.

        Every variant keeps the same footprint and the same wall height, so any difference
        in the number is the alteration and not the size of the thing.
        
    """
    b = {}
    for x in range(21):
        for z in range(21):
            b[(x, 0, z)] = "grass_block"
    x1, z1 = X0 + W - 1, Z0 + D - 1

    def is_wall(x, z):
        return x in (X0, x1) or z in (Z0, z1)

    for y in range(1, H + 1):
        for x in range(X0, x1 + 1):
            for z in range(Z0, z1 + 1):
                if is_wall(x, z):
                    b[(x, y, z)] = wall
    if jetty:
        # the upper half oversails the lower by one block on all four sides
        for y in range(H // 2 + 1, H + 1):
            for x in range(X0 - 1, x1 + 2):
                for z in range(Z0 - 1, z1 + 2):
                    if x in (X0 - 1, x1 + 1) or z in (Z0 - 1, z1 + 1):
                        b[(x, y, z)] = wall
    if pilasters:
        # a post standing one block proud of the wall, every third bay, full height
        for y in range(1, H + 1):
            for x in range(X0 + 2, x1 - 1, 3):
                b[(x, y, Z0 - 1)] = wall
                b[(x, y, z1 + 1)] = wall
            for z in range(Z0 + 2, z1 - 1, 3):
                b[(X0 - 1, y, z)] = wall
                b[(x1 + 1, y, z)] = wall
    if recess:
        # a panel set back one block in the middle of each long wall, full height
        for y in range(1, H + 1):
            for x in range(X0 + 4, x1 - 3):
                del b[(x, y, Z0)]
                del b[(x, y, z1)]
                b[(x, y, Z0 + 1)] = wall
                b[(x, y, z1 - 1)] = wall
    if windows:
        # 1x2 holes right through the wall, on the same rhythm as the pilasters. The
        # openings in facing walls are deliberately offset by one, so that looking
        # through one lands on the far wall rather than out the other side: a column of
        # pure air is not a sample at all, and the case is about what the measure does
        # with a sample whose first solid cell is ten blocks back.
        for y in (3, 4):
            for x in range(X0 + 2, x1 - 1, 3):
                b.pop((x, y, Z0), None)
                b.pop((x + 1, y, z1), None)
            for z in range(Z0 + 2, z1 - 1, 3):
                b.pop((X0, y, z), None)
                b.pop((x1, y, z + 1), None)
    if wing:
        # a wing standing five blocks proud of the north wall, full height: a separate
        # mass rather than anything happening on the wall's surface. Centred, so its own
        # side walls are five blocks in from the east and west planes too and no face of
        # it lands inside the band by accident.
        for y in range(1, H + 1):
            for x in range(X0 + 5, X0 + 8):
                for z in range(Z0 - 5, Z0):
                    if x in (X0 + 5, X0 + 7) or z == Z0 - 5:
                        b[(x, y, z)] = wall
    if gable:
        # a ridge along x, stepping back a block a course on the two long sides
        for i in range(1, D // 2 + 1):
            for x in range(X0, x1 + 1):
                for z in range(Z0 + i, z1 - i + 1):
                    if z in (Z0 + i, z1 - i):
                        b[(x, H + i, z)] = wall
    return observe.Volume.from_blocks(b, 0, 0, 0, 21, H + D, 21)


def a(vol) -> float:
    return articulation.measure(vol, PLOT)["articulation"]


def full(vol) -> dict:
    return articulation.measure(vol, PLOT)


def main():
    cases = []

    flat = full(box())
    cases.append(("a flat box reads 0.000 -- the control",
                  flat["articulation"] == 0.0,
                  f"{flat['articulation']}, {flat['samples']} samples, "
                  f"relief {flat['relief']}"))
    cases.append(("...and it found walls to measure at all",
                  flat["samples"] > 100 and flat["structures"] == 1,
                  f"{flat['samples']} samples on {flat['structures']} structure"))

    pil = full(box(pilasters=True))
    cases.append(("pilasters standing proud of the wall score above flat",
                  pil["articulation"] > 0.1, f"{pil['articulation']}"))
    # The fixture stands every third bay one block proud, so a third of the wall should
    # deviate. That the number lands on the fraction the fixture built is the case that
    # says it is measuring bays and not something correlated with them.
    cases.append(("...and the number is the fraction of bays that are proud, 1 in 3",
                  abs(pil["articulation"] - 1 / 3) < 0.05, f"{pil['articulation']}"))
    # Relief comes out a little over one block rather than exactly one because of the
    # returns -- at a corner, the side wall's own pilasters are seen edge-on through the
    # front face's plane and are genuinely set back further. Named here so it is a known
    # property rather than a surprise in a later reading; it inflates `relief` slightly
    # and leaves `articulation`, the headline, untouched.
    cases.append(("...measured in blocks: relief is near one block per deviation",
                  0.8 <= pil["relief"] / pil["articulation"] <= 1.5,
                  f"relief {pil['relief']} over articulation {pil['articulation']} "
                  f"= {pil['relief'] / pil['articulation']:.2f} blocks, "
                  f"max {pil['max_relief']}"))

    rec = full(box(recess=True))
    cases.append(("a recessed panel scores above flat",
                  rec["articulation"] > 0.1, f"{rec['articulation']}"))

    jet = full(box(jetty=True))
    cases.append(("a jettied upper storey scores above flat",
                  jet["articulation"] > 0.2, f"{jet['articulation']}"))
    cases.append(("...which a per-column measure would have missed entirely",
                  jet["relief"] > 0.2, f"relief {jet['relief']} blocks"))

    # --- the two ways this measure could be a lie -------------------------
    gab = full(box(gable=True))
    cases.append(("a pitched roof is NOT articulation: the gabled box still reads 0",
                  gab["articulation"] == 0.0,
                  f"{gab['articulation']}, {gab['sloped']} sloped samples set aside"))
    cases.append(("...and the roof really was there to be miscounted",
                  gab["sloped"] > flat["sloped"] + 20,
                  f"sloped {flat['sloped']} flat -> {gab['sloped']} gabled"))

    win = full(box(windows=True))
    cases.append(("a hole is NOT a recess: windows do not move the number",
                  win["articulation"] == 0.0,
                  f"{win['articulation']}, {win['beyond']} samples off the plane"))
    cases.append(("...and the windows really were there to be miscounted",
                  win["beyond"] > 20, f"{win['beyond']} samples more than "
                  f"{articulation.MAX_RECESS} blocks off the modal plane"))

    # The symmetric half of the same case. A wing standing well proud of the wall is
    # massing, not surface relief; the first version of this measure capped only the far
    # side and ranked the flattest of the four selection candidates highest.
    wing = full(box(wing=True))
    cases.append(("a projecting wing is massing, not relief: the number does not move",
                  wing["articulation"] == 0.0,
                  f"{wing['articulation']}, {wing['beyond']} samples off the plane"))
    cases.append(("...and the wing really was there to be miscounted",
                  wing["beyond"] > 20, f"{wing['beyond']} samples more than "
                  f"{articulation.MAX_RECESS} blocks off the modal plane"))

    both = full(box(pilasters=True, windows=True, gable=True))
    cases.append(("pilasters still register through a gable and windows",
                  abs(both["articulation"] - pil["articulation"]) < 0.06,
                  f"{both['articulation']} against {pil['articulation']} alone"))

    # --- ordering, which is the only use it is put to ---------------------
    order = [("flat", flat), ("windows", win), ("gable", gab), ("wing", wing),
             ("pilasters", pil), ("recess", rec), ("jetty", jet)]
    vals = [v["articulation"] for _, v in order]
    cases.append(("the flat variants rank below every articulated one",
                  max(vals[:4]) < min(vals[4:]),
                  ", ".join(f"{k} {v['articulation']}" for k, v in order)))

    # --- a pure function of the volume ------------------------------------
    v = box(pilasters=True)
    cases.append(("a pure function: two calls, the same numbers",
                  full(v) == full(v), ""))

    # --- it declines rather than guesses ----------------------------------
    empty = articulation.measure(observe.Volume.from_blocks(
        {(x, 0, z): "grass_block" for x in range(21) for z in range(21)},
        0, 0, 0, 21, 8, 21), PLOT)
    cases.append(("bare ground has no wall face and is reported as none",
                  empty["structures"] == 0 and empty["articulation"] is None,
                  f"{empty['structures']} structures, {empty['articulation']}"))

    # --- and it is not a lint check ---------------------------------------
    from ethoslm import lint
    codes = {c for c in dir(lint) if c.startswith(("e0", "w0"))}
    cases.append(("the check suite is untouched: no articulation check exists",
                  not any("articul" in c for c in dir(lint)),
                  f"{len(codes)} check functions in lint, none of them this"))

    fails = 0
    for label, ok, detail in cases:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
        fails += not ok
    print(f"\n{len(cases) - fails}/{len(cases)} articulation cases pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
