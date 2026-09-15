"""Roof invariants, checked without a server, a world or a render.

No world needed. Run this before anything that costs seconds.

1. Stair facing must point up-slope. A stair's `facing` names the side its raised
quarter is on (assets/minecraft/models/block/stairs.json: the unrotated model, which
blockstates maps to facing=east, has its top element at x 8..16 = +X). Facing down-slope
leaves the tall face exposed on every course: a serrated roof. 2. Pitch must stay
clamped to [1:4, 2:1]. Above 2 rise per 1 run each course leaves a flat tread under a
tall face and the roof reads as a ziggurat. Callers do ask for (3,1) -- one did, and
produced a copper ziggurat -- so the clamp is load-bearing. every style, both axes, five
pitches, three rectangles, three overhangs -- digested from the code as it stood before
`profile`, `ends`, `eave` and `tiers` existed. A preset that moves is a preset that has
stopped replaying, and the digest names which one.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm import observe  # noqa: E402
from ethoslm.prims import Primitives  # noqa: E402

DIR = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


class FakeBuilder(Primitives):
    """Collects placements instead of writing them. Ground is flat at y=64."""

    def __init__(self):
        self.blocks: dict[tuple[int, int, int], str] = {}

    def place_block(self, x, y, z, block):
        self.blocks[(int(x), int(y), int(z))] = block

    def get_height(self, x, z):
        return 64


def surface(blocks):
    """Highest placed block per column -- the roof's outer surface."""
    top: dict[tuple[int, int], int] = {}
    for (x, y, z) in blocks:
        if blocks[(x, y, z)] == "air":
            continue
        k = (x, z)
        if k not in top or y > top[k]:
            top[k] = y
    return top


def stairs_of(blocks):
    out = []
    for (x, y, z), b in blocks.items():
        if "_stairs[" in b and "facing=" in b:
            f = b.split("facing=")[1].split(",")[0].split("]")[0]
            out.append(((x, y, z), f))
    return out


def check_facing(label, blocks):
    """Every stair's raised side must be the up-slope side.

        Tested as "not strictly down-slope" rather than "strictly up-slope": at the ridge
        column both neighbours are lower, so no facing is up-slope there and the apex is
        not a defect. Inverted facing on a real slope makes top[down] > top[up], which this
        catches on every course.
        
    """
    top = surface(blocks)
    bad = []
    for (x, y, z), f in stairs_of(blocks):
        dx, dz = DIR[f]
        up, down = (x + dx, z + dz), (x - dx, z - dz)
        if up not in top or down not in top:
            continue  # eave column: no neighbour to compare against
        if top[down] > top[up]:
            bad.append(((x, y, z), f, top[down], top[(x, z)], top[up]))
    return bad


def check_pitch(label, blocks, limit=2):
    """No course may rise more than MAX_RATIO above its neighbour.

        The ridge column is excluded: gable caps it with a slab one above the ridge, which
        is a ridge tile, not a course.
        
    """
    top = surface(blocks)
    ridge = max(top.values())
    bad = []
    for (x, z), h in top.items():
        if h == ridge:
            continue
        for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            n = (x + dx, z + dz)
            if n in top and top[n] != ridge and abs(top[n] - h) > limit:
                bad.append(((x, z), h, n, top[n]))
    return bad


CASES = []
for style in ("gable", "hip", "gambrel", "mansard"):
    for axis in ("x", "z"):
        for pitch in ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (5, 1), (1, 9)):
            CASES.append((style, axis, pitch))
for axis in ("n", "s", "e", "w"):
    for pitch in ((1, 1), (1, 2), (2, 1), (3, 1)):
        CASES.append(("shed", axis, pitch))


def main():
    fails = 0
    checked = 0
    for style, axis, pitch in CASES:
        b = FakeBuilder()
        b.roof(0, 0, 12, 16, 70, "stone_brick", style=style, axis=axis, pitch=pitch)
        label = f"{style:8s} axis={axis} pitch={pitch}"
        f_bad = check_facing(label, b.blocks)
        p_bad = check_pitch(label, b.blocks)
        checked += 1
        if f_bad or p_bad:
            fails += 1
            print(f"FAIL {label}: {len(f_bad)} backwards stairs, "
                  f"{len(p_bad)} over-pitch steps")
            for row in (f_bad + p_bad)[:3]:
                print("     ", row)
        else:
            n = len(stairs_of(b.blocks))
            print(f"ok   {label}  ({n} stairs, ridge span "
                  f"{max(surface(b.blocks).values()) - 70})")

    # roof_cone shares _clamp_pitch and must not ziggurat either
    for pitch in ((1, 1), (2, 1), (4, 1), (1, 3)):
        b = FakeBuilder()
        b.roof_cone(0, 70, 0, 8, "stone_brick", pitch=pitch)
        p_bad = check_pitch("cone", b.blocks)
        checked += 1
        if p_bad:
            fails += 1
            print(f"FAIL roof_cone pitch={pitch}: {len(p_bad)} over-pitch steps")
        else:
            print(f"ok   roof_cone pitch={pitch}")

    # A1. The presets are the parameters at their defaults, block for block.
    moved = []
    want = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "roof_presets.json")))
    for key, digest in sorted(want.items()):
        rect, oh, style, ax, pitch = key.split("|")
        x0, z0, x1, z1 = (int(v) for v in rect.split(","))
        rise, run = (int(v) for v in pitch.split(","))
        b = FakeBuilder()
        ridge = b.roof(x0, z0, x1, z1, 70, "stone_brick", style=style, axis=ax,
                       pitch=(rise, run), overhang=int(oh))
        payload = json.dumps(sorted((list(k), v) for k, v in b.blocks.items()))
        got = f"{ridge}:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
        if got != digest:
            moved.append(key)
    checked += 1
    if moved:
        fails += 1
        print(f"FAIL {len(moved)} of {len(want)} preset roofs moved: {moved[:6]}")
    else:
        print(f"ok   presets byte-identical  ({len(want)} roofs, "
              f"{len({k.split('|')[2] for k in want})} styles)")

    # A1. ...and the roof the presets could not draw: an irimoya on an upturned eave
    # over two tiers, which is the silhouette a Japanese hall has. Three things about
    # it, and each was broken once while it was being written: it stands up, every block
    # of it is held by another block, and the ridge is not a line of slabs hanging over
    # the hip at either end of it.
    b = FakeBuilder()
    ridge = b.roof(0, 0, 14, 18, 70, "thatch", style="hip", pitch=(1, 2),
                   profile=[(1, 2), (2, 1)], ends=("irimoya", "irimoya"),
                   eave="upturned", tiers=2)
    vol = observe.Volume.from_blocks(b.blocks, -6, 64, -6, 32, 48, 40)
    floating = observe.unsupported(vol)
    top = surface(b.blocks)
    holes = [(x, z) for (x, z), h in top.items() if h == ridge
             for (nx, nz) in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1))
             if (nx, nz) not in top and -1 <= nx <= 15 and -1 <= nz <= 19]
    checked += 1
    if floating or holes or ridge <= 70:
        fails += 1
        print(f"FAIL irimoya/upturned/2 tiers: {len(floating)} floating, "
              f"{len(holes)} holes beside the ridge, ridge {ridge}")
    else:
        print(f"ok   irimoya, upturned eave, 2 tiers  ({len(b.blocks)} blocks, ridge "
              f"{ridge}, 0 floating, no hole beside the ridge)")

    # A1. The other two ends and the other eave, on the same fixture: each has to be a
    # different roof from a plain hip and each has to stand up.
    for kind, eave in (("half-hip", "flared"), ("gable", "flared"),
                       ("hip", "upturned")):
        b = FakeBuilder()
        r = b.roof(0, 0, 14, 18, 70, "spruce", style="hip", pitch=(1, 1),
                   ends=(kind, kind), eave=eave)
        v = observe.Volume.from_blocks(b.blocks, -6, 64, -6, 32, 48, 40)
        f_bad = check_facing(kind, b.blocks)
        fl = observe.unsupported(v)
        checked += 1
        if f_bad or fl:
            fails += 1
            print(f"FAIL ends={kind} eave={eave}: {len(f_bad)} backwards stairs, "
                  f"{len(fl)} floating")
        else:
            print(f"ok   ends={kind:9s} eave={eave:9s} ({len(b.blocks)} blocks, "
                  f"ridge {r})")

    # ---- A2. an upturned eave leaves no enclosed void, at any tier. It went into the
    # record unconfirmed. This is the fixture that answers it: a walled hall with the
    # roof that reported it -- irimoya ends, an upturned eave, two tiers, a (1,2)/(2,1)
    # profile -- built into a world and read by the same `observe.shelter` every
    # interior check is built on. **What is there and what is not.** The lift places a
    # full cube under the raised course, so where the tier above overhangs the tier
    # below its eave roofs over the gap between the two: 168 covered air cells on this
    # fixture that a straight eave does not produce, in a band one column wide round the
    # junction. None of it is *enclosed* -- it is open sideways, under the eave, which
    # is what the shadow gap between two tiers of a Japanese roof is -- and
    # `observe.rooms` reads it at enclosure 0.0, an edge rather than an interior. So the
    # void the report named is not there, and the band that is there is architecture.
    # What it exposes instead is an inconsistency between three readers of one question,
    # and that is on the record as an open thread rather than fitted to here:
    # `check_walkable` and E011 both drop a component below `lint.Context.ENCLOSED` and
    # `Context.interior_walk` does not.
    from ethoslm import lint as lint_mod                                # noqa: PLC0415
    hall = FakeBuilder()
    for x in range(0, 15):
        for z in range(0, 19):
            for y in range(64, 70):
                if x in (0, 14) or z in (0, 18):
                    hall.place_block(x, y, z, "stone_bricks")
            hall.place_block(x, 63, z, "stone_bricks")
    hall.place_block(7, 64, 0, "oak_door[facing=north,half=lower]")
    hall.place_block(7, 65, 0, "oak_door[facing=north,half=upper]")
    tier_ridge = hall.roof(0, 0, 14, 18, 70, "thatch", style="hip", pitch=(1, 2),
                           profile=[(1, 2), (2, 1)], ends=("irimoya", "irimoya"),
                           eave="upturned", tiers=2)
    blocks = dict(hall.blocks)
    for x in range(-8, 24):
        for z in range(-8, 28):
            for y in range(56, 63):
                blocks.setdefault((x, y, z), "stone")
    hv = observe.Volume.from_blocks(blocks, -8, 56, -8, 34, 44, 40)
    sky, sealed = observe.shelter(hv)
    roof_y0 = 70 - hv.y0
    shut = int(sealed[:, roof_y0:, :].sum())
    up_rooms = [r for r in observe.rooms(observe.Nav(hv), sky, region=(0, 0, 15, 19))
                if r["bbox"][1] >= 70]
    inside = [r for r in up_rooms if r["enclosure"] >= lint_mod.Context.ENCLOSED]
    checked += 1
    if shut or inside or tier_ridge <= 70:
        fails += 1
        print(f"FAIL upturned eave, 2 tiers: {shut} cells of enclosed void above the "
              f"eave, {len(inside)} of them read as an interior, ridge {tier_ridge}")
    else:
        band = sum(r["cells"] for r in up_rooms)
        print(f"ok   upturned eave leaves no enclosed void  (2 tiers, ridge "
              f"{tier_ridge}, 0 sealed cells above the eave; the {band}-cell band "
              f"under the upper eave reads at enclosure "
              f"{max([r['enclosure'] for r in up_rooms], default=0.0)})")

    # The same invariant, checked the way it will be checked in the world: build the
    # roof into a Volume and run the world-side detector over it. Catching it in both
    # places matters -- the library check only sees roofs the library drew, and models
    # place plenty of stairs by hand.
    b = FakeBuilder()
    b.roof(0, 0, 12, 16, 70, "stone_brick", style="gable", axis="z", pitch=(1, 1))
    vol = observe.Volume.from_blocks(b.blocks, -4, 64, -4, 24, 32, 32)
    good = observe.backwards_stairs(vol)
    flipped = {p: (s.replace("facing=north", "facing=X").replace("facing=south", "facing=north")
                   .replace("facing=X", "facing=south"))
               for p, s in b.blocks.items()}
    vol2 = observe.Volume.from_blocks(flipped, -4, 64, -4, 24, 32, 32)
    bad_world = observe.backwards_stairs(vol2)
    print(f"\nworld-side detector: {len(good)} backwards on the fixed roof, "
          f"{len(bad_world)} on the same roof inverted")
    if good or not bad_world:
        fails += 1
        print("FAIL observe.backwards_stairs does not discriminate")
    checked += 1

    print(f"\n{checked - fails}/{checked} roof cases pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
