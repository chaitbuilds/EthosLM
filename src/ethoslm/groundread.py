"""Measured occupied-block shares, explicit reservations, and **what the ground is made
of** for site selection.

Three questions a square of ground is asked before a place is put on it, all answered
here so the live scan, the cached scan and the grid walk read the same answer:

  - has somebody built on it? -- `occupancy`, a material census, and `excluded`, the
    ledger of every site this project has worked;
  - what is its surface? -- `surface_class`, the setting: green, sand, badlands, snow,
    stone or earth, per column, off the block a column's ground is made of;
  - and, from those, `virgin_failures` and the surface share the search scores.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from . import observe

ROOT = Path(__file__).resolve().parents[2]
MAX_MAN_MADE_SHARE = 0.005
RELIEF_BANDS = {"village": (8, 25), "town": (6, 20), "city": (4, 15)}

#: What the world grows and pours that is nobody's build. `observe._is_natural` is
#: deliberately blind to vegetation -- it answers "was this room's wall *made*", where a
#: placed log is a corner post -- and this census borrowed it as it stood. Over a square
#: kilometre of ground that made it a **forest census**: on the cached worlds 92 to 98
#: per cent of what it counted as artificial was leaves and logs, every forested square
#: of a city's search read over the 0.5% cap, and the one square that met a city's needs
#: was the one with 5.7% canopy. A tree is not a build. Neither is lava, a coral reef, a
#: kelp forest or a cactus, all of which the generator lays.
_WORLDGEN_EXACT = {"lava", "flowing_lava", "fire", "soul_fire", "kelp", "kelp_plant",
                   "seagrass", "tall_seagrass", "sponge", "wet_sponge", "dead_bush",
                   "bone_block", "sea_pickle", "lily_pad", "moss_carpet",
                   "pale_moss_carpet", "hanging_roots", "spore_blossom", "glow_lichen",
                   "sculk_vein", "sculk_catalyst", "sculk_sensor", "sculk_shrieker",
                   "cave_vines", "cave_vines_plant", "vine", "twisting_vines",
                   "twisting_vines_plant", "weeping_vines", "weeping_vines_plant",
                   "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "bamboo",
                   "bamboo_sapling", "cactus", "sugar_cane", "cobweb", "snow",
                   "powder_snow", "packed_mud", "mangrove_propagule",
                   "pale_hanging_moss", "pale_moss_block", "short_grass", "tall_grass",
                   "fern", "large_fern", "creaking_heart", "pointed_dripstone",
                   "amethyst_cluster", "large_amethyst_bud", "medium_amethyst_bud",
                   "small_amethyst_bud", "chorus_plant", "chorus_flower", "end_stone",
                   "nether_wart_block", "warped_wart_block", "shroomlight",
                   "mushroom_stem", "brown_mushroom_block", "red_mushroom_block",
                   "melon", "pumpkin", "bubble_column", "water", "flowing_water"}
_WORLDGEN_SUFFIX = ("_leaves", "_log", "_wood", "_stem", "_hyphae", "_sapling",
                    "_propagule", "_roots", "_fungus", "_sprouts", "_bush", "_flower",
                    "_petals", "_coral", "_coral_block", "_coral_fan",
                    "_coral_wall_fan", "_mushroom", "_wart", "_tulip", "_orchid",
                    "_bud", "_egg", "_froglight", "_ice", "_lichen")


def made(name: str) -> bool:
    """Is this block somebody's build, as far as a material can say?

        Conservative in one direction only: a natural stone laid as a wall is invisible,
        which the explicit reservations exist to cover. What it must never do is call the
        world's own trees, reefs and lava a build, because a filter that refuses every
        forest is a filter for deserts.
        
    """
    if observe._is_natural(name) or observe.is_growing(name):
        return False
    if name in _WORLDGEN_EXACT or name.endswith(_WORLDGEN_SUFFIX):
        return False
    return True


def occupancy(vol):
    """Per-column artificial and occupied counts, using the room census's tables.

        This is a conservative material census, not authorship detection. World-generated
        *structures* -- a village's cobblestone, a trial chamber's deepslate bricks -- can
        contribute; unmodified stone constructions can be invisible. Explicit site
        reservations cover known builds independently of material. Vegetation, liquids and
        the rest of what the generator grows are **not** counted: see `made`.
        
    """
    table = vol.tables()
    occupied = table["lower"].astype(bool) | table["upper"].astype(bool)
    artificial = occupied & np.array([made(s.split("[")[0]) for s in vol.palette],
                                     dtype=bool)
    return artificial[vol.codes].sum(axis=1), occupied[vol.codes].sum(axis=1)


@lru_cache(maxsize=1)
def exclusions():
    return json.loads((ROOT / "src/ethoslm/data/site-exclusions.json").read_text())


def excluded(x, z, width, depth):
    doc = exclusions()
    margin = doc["margin"]
    result = []
    for row in doc["sites"]:
        a, b, c, d = row["rect"]
        if x <= c + margin and x + width - 1 >= a - margin \
                and z <= d + margin and z + depth - 1 >= b - margin:
            result.append(row["name"])
    return result


def virgin_failures(measures):
    """Named refusal reasons; unread block material is never certified as virgin."""
    result = []
    share = measures.get("man_made_share")
    if share is None:
        result.append("occupied-block census is unavailable")
    elif share > MAX_MAN_MADE_SHARE:
        result.append(f"man-made share {share:.8%} exceeds 0.5%")
    if measures.get("excluded_by"):
        result.append("reserved site: " + ", ".join(measures["excluded_by"]))
    return result


# ------------------------------------------------------------------ the surface The
# setting. A city's search scored relief, water, canopy, gravity and a plateau, and not
# one of those says what colour the ground is. The surface is read off the block each
# column's ground is made of, classed into the few words a sentence uses for a setting,
# and a place spec may ask for one of them (`spec.read_setting`).

#: The classes a surface can be, in the words a sentence would use. Anything a column's
#: ground is made of that is none of these -- a built block, a coral head -- is
#: unclassed and counts towards no setting.
SURFACES = ("green", "earth", "sand", "badlands", "snow", "stone")

_SURFACE_EXACT = {
    "grass_block": "green", "podzol": "green", "moss_block": "green",
    "pale_moss_block": "green",
    "dirt": "earth", "coarse_dirt": "earth", "rooted_dirt": "earth", "mud": "earth",
    "packed_mud": "earth", "clay": "earth", "mycelium": "earth", "dirt_path": "earth",
    "farmland": "earth",
    "sand": "sand", "suspicious_sand": "sand", "sandstone": "sand",
    "red_sand": "badlands", "red_sandstone": "badlands", "terracotta": "badlands",
    "snow": "snow", "snow_block": "snow", "powder_snow": "snow", "ice": "snow",
    "packed_ice": "snow", "blue_ice": "snow", "frosted_ice": "snow",
    "stone": "stone", "deepslate": "stone", "andesite": "stone", "diorite": "stone",
    "granite": "stone", "tuff": "stone", "calcite": "stone", "basalt": "stone",
    "smooth_basalt": "stone", "blackstone": "stone", "gravel": "stone",
    "suspicious_gravel": "stone", "cobblestone": "stone", "mossy_cobblestone": "stone",
    "dripstone_block": "stone", "obsidian": "stone", "magma_block": "stone",
    "netherrack": "stone", "end_stone": "stone", "bedrock": "stone",
    "amethyst_block": "stone", "budding_amethyst": "stone", "sculk": "stone",
}

#: What lies *on* the ground and says more about the setting than the ground does: a
#: snow layer over grass is a snowfield, and `observe.ground_heights` -- rightly, for a
#: lane -- sounds through it to the grass.
SURFACE_COVER = ("snow", "powder_snow")


def surface_class(name: str) -> str | None:
    """Which setting one block of ground belongs to, or None where it is none of them."""
    if name in _SURFACE_EXACT:
        return _SURFACE_EXACT[name]
    if name.endswith("_terracotta"):
        return "badlands"
    if name.endswith("_ore") or name.startswith("infested_"):
        return "stone"
    return None


def surface_names(vol) -> np.ndarray:
    """The block each column's ground is made of, by name, as an array of `str`.

        The firm ground `ground_heights` finds -- under the trees, not on top of them --
        with the one exception `SURFACE_COVER` names: a layer of snow on that ground is
        what the ground reads as. A column with no ground reads `""`; a column under water
        reads the bed, which the census then leaves out of the land it is a share of.
        
    """
    names = np.array([s.split("[")[0] for s in vol.palette], dtype=object)
    t = vol.tables()
    c = vol.codes
    solid = t["lower"][c].astype(bool) & t["upper"][c].astype(bool)
    veg = np.array([observe._is_vegetation(s) for s in vol.palette], bool)[c]
    firm = solid & ~veg
    idx = firm.shape[1] - 1 - np.argmax(firm[:, ::-1, :], axis=1)
    has = firm.any(axis=1)
    ground = np.take_along_axis(c, idx[:, None, :], axis=1)[:, 0, :]
    above = np.take_along_axis(c, np.minimum(idx + 1, c.shape[1] - 1)[:, None, :],
                               axis=1)[:, 0, :]
    out = names[ground]
    cover = np.isin(names[above], SURFACE_COVER)
    out = np.where(cover, names[above], out)
    return np.where(has, out, "")


def surface_census(codes: np.ndarray, palette, wet: np.ndarray, top: int = 8) -> dict:
    """The setting of a footprint: each class's share of the **land**, and the blocks.

        `codes` index `palette` per column; `wet` marks the columns that are water, which
        are not land and count towards no class -- a lake in a plain is a plain with a lake
        in it, not a plain that is a fifth less green. Shares are of the land columns and
        sum to at most one; `land_pct` says how much of the footprint that was.
        
    """
    codes = np.asarray(codes)
    wet = np.asarray(wet, bool)
    land = ~wet
    n_land = int(land.sum())
    counts = np.bincount(codes[land].ravel(), minlength=len(palette)) if n_land else \
        np.zeros(len(palette), np.int64)
    classes = {k: 0 for k in SURFACES}
    for i, name in enumerate(palette):
        k = surface_class(str(name))
        if k and counts[i]:
            classes[k] += int(counts[i])
    order = sorted(((str(palette[i]), int(counts[i])) for i in range(len(palette))
                    if counts[i] and str(palette[i])), key=lambda kv: (-kv[1], kv[0]))
    return {"read": True,
            "land_pct": round(100.0 * n_land / max(1, codes.size), 1),
            "classes": {k: round(100.0 * v / n_land, 1) if n_land else 0.0
                        for k, v in classes.items()},
            "top": [[k, round(100.0 * v / n_land, 1)] for k, v in order[:top]]}


# What a rendered place stands on is its surface. Then a city was put on a square whose
# own record said 77% savanna and 16% river, under an invariants paragraph that said "a
# flat open green plain": the surface was green enough and the biome was a comment. A
# biome is the world's own word for a setting -- it is what the trees, the grass colour
# and the water are -- so a place spec may now prefer a list of them, in the few words a
# sentence uses, and the search scores the share of a square's cells that are any of
# them.

#: The classes a biome can be, in the words a sentence would use. `any` is no
#: preference. Anything the game calls a biome that is none of these -- a cave, an
#: ocean, a beach, a river -- is unclassed and counts towards no preference, which is
#: how a river through a plain costs the plain its share: a place asked for a plain is
#: asked for a plain and not for a river.
BIOMES = ("plains", "savanna", "forest", "jungle", "desert", "badlands", "swamp",
          "snowy", "mountain", "any")

_BIOME_EXACT = {
    "plains": "plains", "sunflower_plains": "plains", "meadow": "plains",
    "cherry_grove": "plains",
    "savanna": "savanna", "savanna_plateau": "savanna", "windswept_savanna": "savanna",
    "forest": "forest", "flower_forest": "forest", "birch_forest": "forest",
    "old_growth_birch_forest": "forest", "dark_forest": "forest", "taiga": "forest",
    "old_growth_pine_taiga": "forest", "old_growth_spruce_taiga": "forest",
    "windswept_forest": "forest",
    "jungle": "jungle", "sparse_jungle": "jungle", "bamboo_jungle": "jungle",
    "desert": "desert",
    "badlands": "badlands", "wooded_badlands": "badlands", "eroded_badlands": "badlands",
    "swamp": "swamp", "mangrove_swamp": "swamp",
    "snowy_plains": "snowy", "snowy_taiga": "snowy", "ice_spikes": "snowy",
    "snowy_slopes": "snowy", "grove": "snowy", "frozen_peaks": "snowy",
    "snowy_beach": "snowy",
    "windswept_hills": "mountain", "windswept_gravelly_hills": "mountain",
    "stony_peaks": "mountain", "jagged_peaks": "mountain",
}


def biome_class(name: str) -> str | None:
    """Which setting one biome belongs to, or None where it is none of them."""
    n = str(name).split(":")[-1]
    return _BIOME_EXACT.get(n)


def biome_census(codes: np.ndarray, palette, top: int = 8) -> dict:
    """The biomes of a footprint: each class's share of its cells, and the biomes.

        Over every cell, water included: a river is a biome of its own and a plain with a
        river across it is less of a plain, which is the one thing the surface census -- a
        share of the land -- cannot say and the reason this census exists beside it.
        
    """
    codes = np.asarray(codes)
    counts = np.bincount(codes.ravel(), minlength=len(palette))
    n = max(1, int(codes.size))
    classes = {k: 0 for k in BIOMES if k != "any"}
    for i, name in enumerate(palette):
        k = biome_class(str(name))
        if k and counts[i]:
            classes[k] += int(counts[i])
    order = sorted(((str(palette[i]).split(":")[-1], int(counts[i]))
                    for i in range(len(palette)) if counts[i]),
                   key=lambda kv: (-kv[1], kv[0]))
    return {"read": True, "cells": int(codes.size),
            "classes": {k: round(100.0 * v / n, 1) for k, v in classes.items()},
            "top": [[name, round(100.0 * c / n, 1)] for name, c in order[:top]]}


def biome_share(census: dict, want) -> float:
    """The share (0..1) of a footprint's cells whose biome is any class in `want`."""
    if not census or not census.get("read"):
        return 0.0
    words = [w for w in (want or []) if w != "any"]
    return round(sum(float(census["classes"].get(w, 0.0)) for w in words) / 100.0, 6)
