'A picture of a build in a tenth of a second, off the cached volume.\n\nChunky path-traces at ~39 s a frame, which makes looking a batch job you do at the end.\nA builder cannot iterate against that, and in seven rounds none has: no settlement\nbuilder has ever been shown a picture of anything it built.\n\nNothing about massing, silhouette, roof form, opening rhythm or how a place sits on its\nground needs path tracing. It needs an isometric projection, flat shading by face\nnormal, and block colours. That is a surface extraction and a sort, and the `Volume` a\ndry run already holds in memory is the only input.\n\nThis is a *design* instrument, not a judging one. Chunky is still what you render when\nyou want to know what it looks like; this is what you render when you want to know\nwhether the roof is the right height, twenty times in a row.\n\nTwo projections:\n\n    preview(vol)             isometric from the (+x, +y, +z) octant -- massing in the round\n    elevation(vol, facing)   orthographic front elevation -- roof pitch and storey rhythm\n                             read off this, and off a single isometric they do not\n\nEverything here is a pure function of the volume: stable sort, no clock, no randomness.\nByte-identical output on the same input is a tested property, because the judge caches\non image content and a renderer with a wobble would quietly re-buy every judgement.'
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ethoslm import observe, offline, settlement  # noqa: E402
from ethoslm.measure import record                # noqa: E402

#: Block colour by keyword, first match wins. Deliberately coarse: the question this
#: answers is "is that the right shape", not "is that the right shade of andesite".
#: Anything unmatched comes out magenta so a missing entry is visible rather than silent
#: -- and the audit is a test now: every state in all three standing towns' palettes
#: must resolve to a non-magenta colour (scripts/test_preview.py).
COLOURS = [
    ("bubble_column", (58, 92, 168)), ("water", (58, 92, 168)),
    ("lava", (207, 92, 26)), ("magma", (120, 60, 30)), ("ice", (160, 200, 236)),
    ("_leaves", (58, 104, 44)), ("leaf_litter", (110, 84, 50)),
    ("grass_block", (106, 150, 68)),
    ("seagrass", (70, 120, 60)), ("dry_grass", (188, 160, 92)),
    ("grass", (110, 160, 70)), ("fern", (98, 140, 64)),
    ("dead_bush", (150, 110, 60)), ("bush", (78, 118, 52)),
    ("vine", (70, 110, 50)), ("lichen", (96, 130, 110)), ("cobweb", (230, 230, 230)),
    ("moss", (90, 130, 60)), ("podzol", (88, 62, 36)), ("mycelium", (110, 96, 100)),
    ("dandelion", (222, 206, 62)), ("poppy", (190, 60, 50)),
    ("lilac", (186, 140, 190)), ("peony", (216, 160, 190)),
    ("rose_bush", (190, 60, 50)), ("lily_of_the_valley", (210, 220, 200)),
    ("wheat", (206, 186, 96)), ("carrots", (100, 150, 60)),
    ("potatoes", (100, 150, 60)), ("beetroots", (110, 140, 60)),
    ("melon", (110, 150, 50)), ("pumpkin", (198, 118, 32)),
    ("cocoa", (140, 90, 50)), ("cactus", (78, 118, 42)),
    ("farmland", (110, 76, 50)), ("dirt_path", (148, 120, 68)),
    ("coarse_dirt", (126, 92, 62)),
    ("dirt", (134, 96, 67)), ("mud", (60, 50, 46)), ("clay", (160, 166, 179)),
    ("gravel", (131, 127, 126)), ("sand", (219, 207, 163)),
    ("terracotta", (152, 94, 67)), ("blackstone", (42, 36, 42)),
    ("deepslate", (72, 72, 76)), ("basalt", (80, 78, 84)), ("obsidian", (20, 16, 30)),
    ("tuff", (108, 109, 102)), ("calcite", (223, 226, 220)),
    ("amethyst", (150, 110, 200)), ("lapis", (40, 70, 150)),
    ("cobblestone", (127, 127, 127)), ("andesite", (136, 136, 136)),
    ("diorite", (188, 188, 188)), ("granite", (149, 103, 86)),
    ("stone_brick", (122, 122, 122)), ("sandstone", (216, 203, 155)),
    ("spawner", (40, 50, 60)), ("stone", (125, 125, 125)),
    ("brick", (150, 97, 83)), ("nether", (60, 30, 34)),
    ("brown_mushroom", (140, 105, 80)), ("red_mushroom", (190, 50, 45)),
    ("mushroom", (160, 140, 120)),
    ("_bed", (170, 60, 55)), ("rail", (130, 120, 110)), ("ladder", (150, 118, 70)),
    ("furnace", (110, 110, 110)), ("smoker", (110, 110, 110)),
    ("cauldron", (60, 60, 64)), ("anvil", (70, 70, 74)),
    ("_table", (150, 118, 70)), ("loom", (150, 118, 70)),
    ("barrel", (140, 108, 62)), ("chest", (156, 120, 66)),
    ("composter", (130, 100, 58)), ("bee_nest", (196, 160, 80)),
    ("flower_pot", (152, 94, 67)), ("potted", (152, 94, 67)),
    ("birch", (215, 199, 153)), ("spruce", (114, 84, 48)), ("dark_oak", (66, 43, 20)),
    ("jungle", (154, 110, 77)), ("acacia", (168, 90, 50)), ("mangrove", (117, 54, 48)),
    ("cherry", (226, 176, 179)), ("bamboo", (194, 176, 92)),
    ("oak", (162, 130, 78)), ("log", (109, 85, 51)), ("plank", (162, 130, 78)),
    ("hay", (196, 160, 30)), ("wool", (222, 222, 222)),
    # `"copper" in n` matched `copper_block` and `cut_copper` too, so this table said a
    # fresh copper roof is the green of an oxidized one -- which is the colour a voice's
    # author is now told a material is, and telling them wrong is worse than telling
    # them nothing. The scan takes the first key that matches, so the unweathered states
    # go first.
    ("exposed_copper", (161, 125, 99)), ("weathered_copper", (124, 154, 119)),
    ("oxidized_copper", (110, 160, 130)), ("copper", (192, 107, 79)),
    # ...and prismarine, which this table had no entry for at all and drew magenta.
    ("dark_prismarine", (52, 88, 72)), ("prismarine", (99, 156, 143)),
    ("iron", (200, 200, 200)), ("gold", (240, 205, 80)), ("coal", (40, 40, 40)),
    ("glass", (180, 220, 232)), ("lantern", (255, 214, 130)),
    ("torch", (255, 214, 130)), ("campfire", (255, 170, 90)),
    ("snow", (240, 244, 248)), ("slate", (72, 72, 76)),
]
UNKNOWN = (255, 0, 255)
BG = 235                       # background grey, also the "colour" of air
AIR = {"air", "cave_air", "void_air"}
#: Face brightness. Top faces catch the light, the two side faces differ so a corner
#: reads as a corner -- which is the whole reason to shade at all.
FACE = {"up": 1.00, "x": 0.78, "z": 0.60}


def block_colour(state: str) -> tuple:
    n = state.split("[")[0]
    if n in AIR:
        return (BG, BG, BG)    # never drawn (air is not solid); resolved so the
    for key, rgb in COLOURS:   # palette audit can demand zero magenta
        if key in n:
            return rgb
    return UNKNOWN


def _solid(vol: observe.Volume, clip: bool = True) -> np.ndarray:
    """The occupancy the projections draw, clipped to ground - 8.

        A cropped building otherwise sits on the whole column of subsurface stone down to
        the volume floor, and the substrate is taller than the architecture. The floor is
        taken from the lowest *surface* column in the volume -- on any crop that includes
        ground that is ground, not roof -- minus 8 so a cellar or an undercroft still shows.
        
    """
    t = vol.tables()
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))[vol.codes]
    if clip:
        floor = int(offline.surface_heights(vol).min()) - 8
        keep = np.arange(vol.shape[1]) + vol.y0 >= floor
        solid = solid & keep[None, :, None]
    return solid


def crop(vol: observe.Volume, x0: int, z0: int, x1: int, z1: int,
         pad: int = 4) -> observe.Volume:
    """A sub-volume around a footprint, in world coordinates, palette shared."""
    ax0 = max(min(x0, x1) - pad - vol.x0, 0)
    ax1 = min(max(x0, x1) + pad - vol.x0 + 1, vol.shape[0])
    az0 = max(min(z0, z1) - pad - vol.z0, 0)
    az1 = min(max(z0, z1) + pad - vol.z0 + 1, vol.shape[2])
    return observe.Volume(vol.x0 + ax0, vol.y0, vol.z0 + az0,
                          vol.codes[ax0:ax1, :, az0:az1], vol.palette)


def _trim(img: np.ndarray, scale: int) -> np.ndarray:
    used = np.nonzero((img != BG).any(axis=2).any(axis=1))[0]
    usedx = np.nonzero((img != BG).any(axis=2).any(axis=0))[0]
    if len(used) and len(usedx):
        img = img[used[0]:used[-1] + 1, usedx[0]:usedx[-1] + 1]
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    return img


def preview(vol: observe.Volume, scale: int = 2, grey: bool = False,
            clip: bool = True, _sort_kind: str = "stable") -> np.ndarray:
    """Isometric view from the (+x, +y, +z) octant. Painter's algorithm over the shell.

        `grey=True` throws the palette away and renders mass only, which is what you look at
        when the question is the silhouette and the materials would only distract.

        Deterministic by construction: the painter's sort is stable, so two voxels at the
        same depth resolve by array order every time, on every platform. `_sort_kind` exists
        only so the test can exhibit what an unstable sort does to a tie.
        
    """
    solid = _solid(vol, clip)
    sx, sy, sz = solid.shape

    # The shell facing the camera: a voxel with air on any of its +x, +y or +z sides.
    pad = np.zeros((sx + 1, sy + 1, sz + 1), bool)
    pad[:sx, :sy, :sz] = solid
    up = solid & ~pad[:sx, 1:sy + 1, :sz]
    fx = solid & ~pad[1:sx + 1, :sy, :sz]
    fz = solid & ~pad[:sx, :sy, 1:sz + 1]
    vis = up | fx | fz
    x, y, z = np.nonzero(vis)
    if not len(x):
        return np.full((2, 2, 3), BG, np.uint8)

    # Standard voxel isometric; depth increases away from the camera.
    px = (x - z) + (sx + sz)
    py = ((x + z) // 2 - y) + (sy + (sx + sz) // 2)
    depth = x + y + z
    order = np.argsort(depth, kind=_sort_kind)     # far first, near overwrites

    W, H = int(px.max()) + 2, int(py.max()) + 2
    img = np.full((H, W, 3), BG, np.uint8)

    if grey:
        base = np.full((len(x), 3), 190, np.float32)
    else:
        lut = np.array([block_colour(s) for s in vol.palette], np.float32)
        base = lut[vol.codes[x, y, z]]

    shade = np.where(up[x, y, z], FACE["up"],
                     np.where(fx[x, y, z], FACE["x"], FACE["z"]))[:, None]
    # A one-tap ambient occlusion: how enclosed the cell above is. Cheap, and it is what
    # makes an eaves line and a recessed opening legible at all.
    occ = pad[:sx, 1:sy + 1, :sz].astype(np.float32)
    ao = 1.0 - 0.25 * occ[x, y, z][:, None]
    col = np.clip(base * shade * ao, 0, 255).astype(np.uint8)

    img[py[order], px[order]] = col[order]
    return _trim(img, scale)


def elevation(vol: observe.Volume, facing: str = "south", scale: int = 2,
              grey: bool = False, clip: bool = True) -> np.ndarray:
    """Orthographic front elevation: the wall as a wall, one horizontal axis flattened.

        `facing` names the elevation you are looking at -- "south" is the south front, seen
        by a viewer standing to the south. Roof pitch, storey rhythm and opening spacing
        read off this; from a single isometric they do not, which is why a massing
        judgement gets both.

        Depth is not discarded: a cell set back from the frontmost plane is drawn darker,
        one step per two blocks of recess, so a recessed opening, a jetty and a stepped
        row read as such rather than flattening into one plane.
        
    """
    solid = _solid(vol, clip)
    sx, sy, sz = solid.shape

    if facing in ("south", "north"):
        s = solid if facing == "north" else solid[:, :, ::-1]
        # first solid along the view axis, per (x, y)
        hit = s.any(axis=2)
        d = np.argmax(s, axis=2)                     # distance from the viewer
        zi = d if facing == "north" else (sz - 1 - d)
        cols, rows = np.nonzero(hit)
        codes = vol.codes[cols, rows, zi[cols, rows]]
        dist = d[cols, rows]
        width = sx
    elif facing in ("east", "west"):
        s = solid if facing == "west" else solid[::-1, :, :]
        hit = s.any(axis=0)
        d = np.argmax(s, axis=0)
        xi = d if facing == "west" else (sx - 1 - d)
        rows, cols = np.nonzero(hit)                 # hit is (y, z) here
        codes = vol.codes[xi[rows, cols], rows, cols]
        dist = d[rows, cols]
        width = sz
    else:
        raise ValueError(f"facing must be a compass point, not {facing!r}")

    img = np.full((sy, width, 3), BG, np.uint8)
    if grey:
        base = np.full((len(rows), 3), 190, np.float32)
    else:
        lut = np.array([block_colour(s2) for s2 in vol.palette], np.float32)
        base = lut[codes]
    fade = np.clip(1.0 - 0.06 * (dist // 2), 0.55, 1.0)[:, None]
    col = np.clip(base * fade, 0, 255).astype(np.uint8)
    img[sy - 1 - rows, cols] = col
    return _trim(img, scale)


if __name__ == "__main__":
    import cv2
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(settlement.STATE,
                                                             "preview.png")
    src = os.path.join(settlement.STATE, "world_built.npz")
    vol = offline.load_volume(src)
    jobs = [(out, lambda: preview(vol)),
            (out.replace(".png", "_massing.png"), lambda: preview(vol, grey=True)),
            (out.replace(".png", "_elevation.png"), lambda: elevation(vol))]
    for path, fn in jobs:
        t0 = time.perf_counter()
        img = fn()
        secs = time.perf_counter() - t0
        cv2.imwrite(path, img[:, :, ::-1])
        print(f"{path}: {img.shape[1]}x{img.shape[0]} in {secs:.2f}s")
        record("preview", settlement=settlement.NAME, seconds=round(secs, 2),
               shape=list(vol.shape), out=os.path.basename(path))
