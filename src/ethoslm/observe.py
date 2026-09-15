"""Observation that is not rendering.

**judge at the cheapest observation that can exhibit the defect.** A render cannot
exhibit reachability, so four rounds of render-only judgement missed every defect a
human found in ten minutes on foot.

This module is the layer the linter is built on. Cheapest first:

  Volume    a dense voxel array of a region, decoded in bulk from the chunk sections
            GDPC already holds in memory. Seconds for 192x192x112, versus most of an
            hour if you call getBlockGlobal per voxel.
  Nav       a walk model: which cells a player can stand in, and which stances connect
            to which. This is the piece that answers "can you get in", which nothing in
            the project could do before.
  shelter   which air is under cover, and which is sealed off from the sky entirely.
  rooms     connected components of sheltered standing space -- the interiors.
  light     propagated block light and sky light, instead of a count of light sources.

It is the real client's movement code, so it cannot disagree with the game. It is the
wrong answer *here*. Node is not installed and nix cannot install it (no writable daemon
socket), so it would mean downloading a Node tarball and driving it through the store
loader the way setup_runtime.sh does for the JDK, then babysitting a bot over a network
namespace that is destroyed at the end of every shell call -- all before learning
anything. The BFS is one file, needs no new dependency, runs in-process during a pass,
and answers exactly the question the linter asks. The cost is that its movement rules
are a *model* of the game rather than the game, so they are stated explicitly in RULES
and checked against the real thing by scripts/nav_course.py, which builds every rule as
an obstacle in-world for a human to walk.

Heights are in **half-blocks** throughout, because every surface a player stands on in
Minecraft is at a whole or half block: a slab top is 0.5, a full block is 1.0. Integer
half-units keep the arithmetic exact and the rules readable.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- rules

RULES = """Movement model (half-blocks; 2 half-blocks = one block).

stand       feet rest on a surface at height s, and the CLEAR=4 half-blocks above s (2
blocks; the player is 1.8 tall) are free of collision. walk        to a neighbouring
stance at most STEP=1 higher: free. Minecraft's step height is 0.6, so a slab, or a
stair's low side, is walked onto. stair-walk  a rise of exactly 2 is also free when the
target is a stair entered from its low side (its `facing` points the way you are moving)
**and the source column has STAIR_CLEAR=6 half-blocks free above it** -- three cells,
not two. Mid-step a body straddles the lower tread and the upper one, so the head enters
the third cell above the lower tread. This is the difference between a staircase you
walk up and one you must jump up. jump        a rise of exactly JUMP=2 otherwise:
allowed, but needs JUMP_CLEAR=6 free at the source, and is counted, so a route that only
works by jumping can be told apart from one you can walk. fall        any drop up to
FALL=6 (3 blocks, the damage-free limit). doors       wooden doors and fence gates are
passable -- a player opens them. Iron doors are not, without redstone. water       not
traversable by default: "reachable on foot" should not mean "swim". lava, fire, cactus,
powder snow, magma, sweet berries: never traversable. diagonals   not used. The player
is 0.6 wide and cannot cut an inside corner between two blocks; refusing diagonals is
the conservative direction.
"""

CLEAR = 4         # headroom to stand, half-blocks (player is 1.8 tall)
STEP = 1          # free step up (Minecraft step height 0.6)
JUMP = 2          # jumpable step up (jump height 1.25)
JUMP_CLEAR = 6    # headroom needed at the source of a jump
#: Headroom needed at the *source* of a stair-walk, in half-blocks -- three cells. The
#: same six as JUMP_CLEAR and named separately because it is a different fact.
#: `flight()` cleared two cells over every tread and `Nav` asked for two, and neither
#: models a body **straddling two tread columns mid-step**. Halfway up a step your feet
#: are already at the upper tread's height while you are still over the lower one, so
#: your head is in the third cell above the lower tread. Two cells there is a ceiling
#: you walk into. Nothing but a person could see it, which is what open thread 2 was
#: for.
STAIR_CLEAR = 6
FALL = 6          # damage-free fall

#: Which walk metric a number was read under. A figure quoted without its version is not
#: comparable to one read under another, and this project has now moved the ruler four
#: times under bars that never moved. Every round config and every readout carries it,
#: and `stage_readout` refuses to compare a recorded row across versions unless the row
#: is in `test_types.SUPERSEDED` with its cause. 1 jump-permissive: reachability meant
#: `flood(max_jumps=None)` everywhere. 2 walk-only interiors: `interior_walk`, seeded
#: from the building's own doors. 3 three clear cells over a stair tread. 4 floor is
#: floor, by geometry. It took the top of a barrel out -- and a dais with it. 5 floor is
#: floor, by record. See `floor_stances`. 6 a room is a room for all three readers. It
#: moves `from_own_door_walking_pct`; it does not move `from_outdoors_walking_pct`,
#: which reads `Context.rooms` and never read this.
WALK_MODEL = 6

# --------------------------------------------------------- block collision classes
# (occupies lower half, occupies upper half, height of top face) in half-blocks. A
# surface of 0 means you cannot stand on it. Anything not matched is treated as a full
# cube, which is what most blocks are; Volume.tables() records what fell through as
# `unknown_solids` so the table is extended against evidence, not guesswork.

NO_COLLISION = (0, 0, 0)
SLAB = (1, 0, 1)
TOP_SLAB = (0, 1, 2)
FULL = (1, 1, 2)
FENCE = (1, 1, 2)      # 1.5 tall; also fills the lower half of the cell above (TALL)

TALL = ("_fence", "_wall", "_fence_gate")

_PASSABLE_EXACT = {
    "air", "cave_air", "void_air", "light", "structure_void",
    "torch", "wall_torch", "soul_torch", "soul_wall_torch", "redstone_torch",
    "redstone_wall_torch", "lever", "tripwire", "tripwire_hook", "string", "cobweb",
    "ladder", "vine", "glow_lichen", "sculk_vein", "rail", "powered_rail",
    "detector_rail", "activator_rail", "redstone_wire", "repeater", "comparator",
    "flower_pot", "end_rod", "lightning_rod", "chain", "iron_chain", "copper_chain",
    "water", "flowing_water", "bubble_column", "nether_portal", "end_portal",
    "end_gateway", "snow", "moss_carpet", "pale_moss_carpet", "hanging_roots",
    "small_amethyst_bud", "medium_amethyst_bud", "large_amethyst_bud",
    "amethyst_cluster", "sculk_shrieker", "lily_pad", "scaffolding", "big_dripleaf_stem",
}
_PASSABLE_SUFFIX = (
    "_sapling", "_button", "_pressure_plate", "_sign", "_banner", "_carpet", "_torch",
    "_rail", "_candle", "_wall_sign", "_hanging_sign", "_coral_fan", "_coral_wall_fan",
    "_bud", "_vines", "_sprouts", "_roots", "_litter", "_wall_torch", "_wall_fan",
)
_PASSABLE_CONTAINS = (
    "grass", "fern", "flower", "tulip", "orchid", "allium", "daisy", "cornflower",
    "poppy", "dandelion", "lily", "bush", "mushroom", "sapling", "seagrass", "kelp",
    "sugar_cane", "bamboo", "dead_bush", "petals", "wheat", "carrots", "potatoes",
    "beetroots", "melon_stem", "pumpkin_stem", "nether_wart", "cocoa", "sweet_berry",
    "cave_vines", "twisting_vines", "weeping_vines", "azalea", "spore_blossom",
    "dripleaf", "torchflower", "pitcher", "firefly", "sculk_vein",
)

#: Built shapes whose names contain a word from _PASSABLE_CONTAINS and which are not
#: plants. They skip the substring rule and fall through to the shape rules below, so a
#: bamboo slab is a slab and a bamboo stair is a stair. `thatch` is hay over **bamboo**
#: stairs and slabs, because hay has neither and bamboo is the closest tone that does.
#: The contains-rule read every one of those as a bamboo shoot, so the walk model and
#: the linter looked straight through a thatched roof -- 1,882 blocks of it, of which
#: only the cubes were visible. A material family the checker cannot see is not a
#: family, so this is part of adding one.
_SHAPED_NOT_PLANT = frozenset((
    "bamboo_stairs", "bamboo_slab", "bamboo_fence", "bamboo_fence_gate",
    "bamboo_door", "bamboo_trapdoor", "bamboo_pressure_plate", "bamboo_button",
))

#: Full cubes whose *names* contain a word from _PASSABLE_CONTAINS. Checked first,
#: because the substring rule is a heuristic and these are the cases where it is simply
#: wrong. `grass_block` is the one that matters: the contains-rule is there for
#: short_grass and tall_grass, and it caught the block those grow *on*.
_SOLID_EXACT = {
    "grass_block", "dried_kelp_block", "mangrove_roots", "muddy_mangrove_roots",
    "bamboo_block", "stripped_bamboo_block", "bamboo_planks", "bamboo_mosaic",
    "red_mushroom_block", "brown_mushroom_block", "mushroom_stem",
}
_SOLID_SUFFIX = ("_mushroom_block", "bamboo_planks", "bamboo_mosaic", "_kelp_block")

_LIQUID = {"water", "flowing_water", "bubble_column"}
_DEADLY = {"lava", "flowing_lava", "fire", "soul_fire", "magma_block", "campfire",
           "soul_campfire", "sweet_berry_bush", "cactus", "wither_rose", "powder_snow"}

# Emitted block light -- the blocks a settlement actually uses, not the full table.
EMISSION = {
    "torch": 14, "wall_torch": 14, "soul_torch": 10, "soul_wall_torch": 10,
    "lantern": 15, "soul_lantern": 10, "glowstone": 15, "sea_lantern": 15,
    "shroomlight": 15, "jack_o_lantern": 15, "end_rod": 14, "campfire": 15,
    "soul_campfire": 10, "beacon": 15, "conduit": 15, "lava": 15, "flowing_lava": 15,
    "fire": 15, "soul_fire": 10, "redstone_lamp": 15, "crying_obsidian": 10,
    "magma_block": 3, "glow_lichen": 7, "sculk_catalyst": 6, "amethyst_cluster": 5,
    "ochre_froglight": 15, "verdant_froglight": 15, "pearlescent_froglight": 15,
    "copper_bulb": 15, "brewing_stand": 1, "furnace": 13, "smoker": 13,
    "blast_furnace": 13, "candle": 3, "enchanting_table": 7, "respawn_anchor": 15,
}

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
DIR_CODE = {"north": 1, "south": 2, "east": 3, "west": 4}
CODE_DIR = {1: (0, -1), 2: (0, 1), 3: (1, 0), 4: (-1, 0)}


def parse_props(state: str) -> dict:
    if "[" not in state:
        return {}
    out = {}
    for part in state.split("[", 1)[1].rstrip("]").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _classify(name: str, props: dict) -> tuple[int, int, int, str]:
    """(lower, upper, surface, rule). `rule` is "" when nothing matched and the block
    was assumed to be a full cube, so the guesses can be reported rather than hidden."""
    if name in _SOLID_EXACT or any(name.endswith(s) for s in _SOLID_SUFFIX):
        return (*FULL, "full cube")
    if name in _PASSABLE_EXACT:
        return (*NO_COLLISION, "passable")
    if any(name.endswith(s) for s in _PASSABLE_SUFFIX):
        return (*NO_COLLISION, "passable")
    if name not in _SHAPED_NOT_PLANT and any(c in name for c in _PASSABLE_CONTAINS):
        return (*NO_COLLISION, "passable")
    if name in _DEADLY:
        return (*FULL, "deadly")
    if name.endswith("_door"):
        # A player opens a wooden door on the way through; iron needs redstone.
        return (*FULL, "iron door") if name.startswith("iron") else (*NO_COLLISION, "door")
    if name.endswith("_trapdoor"):
        if props.get("open") == "true":
            return (*NO_COLLISION, "open trapdoor")
        # A shut bottom trapdoor is 3/16 thick: a floor you walk over, not an obstacle.
        return ((*SLAB, "trapdoor") if props.get("half", "bottom") == "bottom"
                else (*TOP_SLAB, "trapdoor"))
    if name.endswith("_fence_gate"):
        # Passable open or shut, for the same reason as a wooden door: a player opens it
        # on the way through. Treating a shut gate as a wall makes every penned or gated
        # yard read as unreachable, which is a statement about sheep, not people.
        return (*NO_COLLISION, "gate")
    if name.endswith("_fence") or (name.endswith("_wall")
                                   and not name.endswith("_wall_sign")):
        return (*FENCE, "fence")
    if name.endswith("_slab"):
        t = props.get("type", "bottom")
        return ((*FULL, "double slab") if t == "double" else
                ((*SLAB, "slab") if t == "bottom" else (*TOP_SLAB, "slab")))
    if name.endswith("_bed"):
        return (*SLAB, "bed")
    if name.endswith("_stairs"):
        return (*FULL, "stairs")
    if _known(name):
        return (*FULL, "full cube")
    return (*FULL, "")


# -------------------------------------------------------------------------- volume

def _decode_section(sec) -> np.ndarray:
    """4096 palette indices for one 16x16x16 chunk section, vectorised."""
    ba = sec.blockStatesBitArray
    if ba is None:
        return np.zeros(4096, np.uint16)
    bits = ba._bitsPerEntry
    per_long = 64 // bits
    longs = np.array(list(ba.longArray), dtype=np.int64).view(np.uint64)
    idx = np.arange(4096, dtype=np.int64)
    lo = idx // per_long
    sh = ((idx % per_long) * bits).astype(np.uint64)
    return ((longs[lo] >> sh) & np.uint64((1 << bits) - 1)).astype(np.uint16)


def _state_str(tag) -> str:
    name = str(tag["Name"].value).split(":")[-1]
    props = {}
    if "Properties" in tag:
        for k in tag["Properties"]:
            props[str(k)] = str(tag["Properties"][k].value)
    if not props:
        return name
    return name + "[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]"


@dataclass
class Volume:
    """A dense voxel view of a region, in world coordinates."""

    x0: int
    y0: int
    z0: int
    codes: np.ndarray            # uint16 (sx, sy, sz), indexes into `palette`
    palette: list[str]
    seconds: float = 0.0
    unknown_solids: dict = field(default_factory=dict)
    _tables: dict | None = field(default=None, repr=False)

    @classmethod
    def from_world_slice(cls, ws, x0: int, z0: int, sx: int, sz: int,
                         y0: int, y1: int) -> "Volume":
        t0 = time.perf_counter()
        sy = y1 - y0
        codes = np.zeros((sx, sy, sz), np.uint16)
        palette: list[str] = ["air"]
        index: dict[str, int] = {"air": 0}
        cr = ws._chunkRect

        for key, sec in ws._sections.items():
            cx, sec_y, cz = int(key.x), int(key.y), int(key.z)
            bx = (int(cr.offset.x) + cx) * 16
            bz = (int(cr.offset.y) + cz) * 16
            by = sec_y * 16
            if (by + 16 <= y0 or by >= y1 or bx + 16 <= x0 or bx >= x0 + sx
                    or bz + 16 <= z0 or bz >= z0 + sz):
                continue
            local = np.empty(len(sec.blockPalette), np.uint16)
            for i, tag in enumerate(sec.blockPalette):
                s = _state_str(tag)
                if s not in index:
                    index[s] = len(palette)
                    palette.append(s)
                local[i] = index[s]
            # section index is y*256 + z*16 + x -> (y, z, x), transpose to (x, y, z)
            block = np.transpose(local[_decode_section(sec)].reshape(16, 16, 16),
                                 (2, 0, 1))
            ax0, ax1 = max(bx, x0), min(bx + 16, x0 + sx)
            ay0, ay1 = max(by, y0), min(by + 16, y1)
            az0, az1 = max(bz, z0), min(bz + 16, z0 + sz)
            codes[ax0 - x0:ax1 - x0, ay0 - y0:ay1 - y0, az0 - z0:az1 - z0] = \
                block[ax0 - bx:ax1 - bx, ay0 - by:ay1 - by, az0 - bz:az1 - bz]

        return cls(x0, y0, z0, codes, palette,
                   seconds=round(time.perf_counter() - t0, 2))

    @classmethod
    def from_blocks(cls, blocks: dict, x0: int, y0: int, z0: int,
                    sx: int, sy: int, sz: int) -> "Volume":
        """Build a volume from a {(x, y, z): state} dict, for tests and for checking a
        pass's pending writes before they are flushed."""
        palette = ["air"]
        index = {"air": 0}
        codes = np.zeros((sx, sy, sz), np.uint16)
        for (x, y, z), s in blocks.items():
            if not (x0 <= x < x0 + sx and y0 <= y < y0 + sy and z0 <= z < z0 + sz):
                continue
            s = s.split(":")[-1]
            if s not in index:
                index[s] = len(palette)
                palette.append(s)
            codes[x - x0, y - y0, z - z0] = index[s]
        return cls(x0, y0, z0, codes, palette)

    # --- queries ----------------------------------------------------------
    @property
    def shape(self):
        return self.codes.shape

    def inside(self, x: int, y: int, z: int) -> bool:
        sx, sy, sz = self.codes.shape
        return (self.x0 <= x < self.x0 + sx and self.y0 <= y < self.y0 + sy
                and self.z0 <= z < self.z0 + sz)

    def state(self, x: int, y: int, z: int) -> str:
        if not self.inside(x, y, z):
            return "air"
        return self.palette[int(self.codes[x - self.x0, y - self.y0, z - self.z0])]

    def name(self, x: int, y: int, z: int) -> str:
        return self.state(x, y, z).split("[")[0]

    def sub(self, x0: int, z0: int, sx: int, sz: int) -> "Volume":
        """A box out of this volume, sharing nothing mutable.

                Cheap because the decode already happened: this is a numpy slice and a palette
                copy. It is what lets a build pass ask about one doorway without paying for the
                whole settlement -- the difference between a check a model can call mid-run and
                a check it cannot.
                
        """
        ax0 = max(x0, self.x0)
        az0 = max(z0, self.z0)
        ax1 = min(x0 + sx, self.x0 + self.codes.shape[0])
        az1 = min(z0 + sz, self.z0 + self.codes.shape[2])
        codes = self.codes[ax0 - self.x0:ax1 - self.x0, :,
                           az0 - self.z0:az1 - self.z0].copy()
        return Volume(ax0, self.y0, az0, codes, list(self.palette))

    def overlay(self, blocks: dict) -> "Volume":
        """Write a {(x, y, z): state} dict into this volume, in place.

                A build pass's writes are pending until it flushes, so a check that reads only
                the world cannot see the wall the pass just decided on. This is how a pass is
                judged on what it is about to do rather than on what it did last time.
                
        """
        index = {s: i for i, s in enumerate(self.palette)}
        for (x, y, z), s in blocks.items():
            if not self.inside(x, y, z):
                continue
            s = s.split(":")[-1]
            i = index.get(s)
            if i is None:
                i = index[s] = len(self.palette)
                self.palette.append(s)
            self.codes[x - self.x0, y - self.y0, z - self.z0] = i
        self._tables = None
        return self

    def counts(self) -> dict:
        vals, n = np.unique(self.codes, return_counts=True)
        return {self.palette[int(v)]: int(c) for v, c in zip(vals, n)}

    def find(self, predicate) -> list[tuple[int, int, int]]:
        """Every position whose block state satisfies `predicate(state_string)`."""
        hits = [i for i, s in enumerate(self.palette) if predicate(s)]
        if not hits:
            return []
        mask = np.isin(self.codes, np.array(hits, np.uint16))
        pts = np.argwhere(mask)
        return [(int(a) + self.x0, int(b) + self.y0, int(c) + self.z0)
                for a, b, c in pts]

    # --- per-palette-entry lookup tables ---------------------------------
    def tables(self) -> dict:
        """Vectorised block properties, one row per palette entry.

                Computing these once and indexing them with `codes` is what makes the layer
                cheap: everything downstream is a numpy gather, never a per-voxel branch.
                
        """
        if self._tables is not None:
            return self._tables
        n = len(self.palette)
        cols = {k: np.zeros(n, t) for k, t in
                (("lower", np.uint8), ("upper", np.uint8), ("surface", np.uint8),
                 ("tall", bool), ("opaque", bool), ("emit", np.uint8),
                 ("liquid", bool), ("deadly", bool), ("door", bool), ("ladder", bool),
                 ("stair_face", np.uint8))}
        unknown: dict[str, int] = {}
        for i, s in enumerate(self.palette):
            name = s.split("[")[0]
            props = parse_props(s)
            lo, up, su, rule = _classify(name, props)
            cols["lower"][i], cols["upper"][i], cols["surface"][i] = lo, up, su
            # fences and walls are 1.5 tall and overflow into the cell above; gates are
            # passable (see _classify) so they must not
            cols["tall"][i] = rule == "fence"
            cols["liquid"][i] = name in _LIQUID
            cols["deadly"][i] = name in _DEADLY
            cols["door"][i] = name.endswith("_door") or name.endswith("_fence_gate")
            cols["ladder"][i] = name in ("ladder", "vine", "scaffolding",
                                         "twisting_vines", "weeping_vines")
            # Light opacity is not movement collision, and conflating them makes every
            # shut room read as daylit. A closed door is walked through (a player opens
            # it) but stops light dead; glass and bars are the opposite.
            shut_door = (name.endswith("_door") and props.get("open") != "true")
            glassy = "glass" in name or name == "iron_bars" or name.endswith("_pane")
            cols["opaque"][i] = ((bool(lo and up) and not cols["liquid"][i] and not glassy)
                                 or shut_door)
            for k, v in EMISSION.items():
                if name == k or name.endswith("_" + k):
                    cols["emit"][i] = max(int(cols["emit"][i]), v)
            if props.get("lit") == "false":
                cols["emit"][i] = 0
            if name.endswith("_stairs"):
                cols["stair_face"][i] = DIR_CODE.get(props.get("facing", ""), 0)
            if not rule:
                unknown[name] = unknown.get(name, 0) + 1
        self.unknown_solids = unknown
        self._tables = cols
        return cols


_KNOWN_SUFFIX = ("_planks", "_log", "_wood", "_stairs", "_slab", "_bricks", "_brick",
                 "_block", "_stone", "_terracotta", "_concrete", "_concrete_powder",
                 "_glass", "_pane", "_wool", "_sandstone", "_ore", "_tiles", "_tile",
                 "_deepslate", "_copper", "_shulker_box", "_leaves", "_mud", "_basalt",
                 "_nylium", "_hyphae", "_stem", "_glazed_terracotta")
_KNOWN_EXACT = {
    "stone", "dirt", "coarse_dirt", "rooted_dirt", "grass_block", "podzol", "mycelium",
    "sand", "red_sand", "gravel", "cobblestone", "bricks", "glass", "obsidian",
    "bedrock", "clay", "ice", "packed_ice", "blue_ice", "netherrack", "end_stone",
    "bookshelf", "chiseled_bookshelf", "crafting_table", "furnace", "barrel", "chest",
    "trapped_chest", "ender_chest", "loom", "smoker", "blast_furnace", "composter",
    "cartography_table", "smithing_table", "fletching_table", "lectern", "jukebox",
    "note_block", "hay_block", "melon", "pumpkin", "carved_pumpkin", "jack_o_lantern",
    "bone_block", "sponge", "target", "dried_kelp_block", "honey_block", "slime_block",
    "mud", "packed_mud", "calcite", "tuff", "basalt", "blackstone", "terracotta",
    "glowstone", "sea_lantern", "shroomlight", "magma_block", "soul_sand", "soul_soil",
    "snow_block", "powder_snow", "moss_block", "dirt_path", "farmland", "cauldron",
    "water_cauldron", "lava_cauldron", "powder_snow_cauldron", "beacon", "lantern",
    "soul_lantern", "campfire", "soul_campfire", "hopper", "dispenser", "dropper",
    "observer", "piston", "sticky_piston", "redstone_lamp", "redstone_block", "tnt",
    "bell", "anvil", "chipped_anvil", "damaged_anvil", "grindstone", "stonecutter",
    "enchanting_table", "brewing_stand", "conduit", "respawn_anchor", "lodestone",
    "spawner", "trial_spawner", "vault", "decorated_pot", "flower_pot", "bee_nest",
    "beehive", "sculk", "sculk_catalyst", "amethyst_block", "budding_amethyst",
    "dripstone_block", "pointed_dripstone", "mangrove_roots", "muddy_mangrove_roots",
    "quartz_block", "purpur_block", "prismarine", "dark_prismarine", "andesite",
    "granite", "diorite", "copper_bulb", "ladder", "suspicious_sand",
    "suspicious_gravel", "mud_bricks", "resin_block", "creaking_heart", "pale_moss_block",
}


def _known(name: str) -> bool:
    """Are we confident this is a full cube? Only used to report the guesses."""
    return name in _KNOWN_EXACT or any(name.endswith(s) for s in _KNOWN_SUFFIX)


# ----------------------------------------------------------------------------- nav

class Nav:
    """The walk model. See RULES.

        Occupancy and standability are computed once as vectorised arrays over *half-block
        levels*, then flattened to `bytes`, because the BFS makes millions of single-element
        lookups and bytes indexing is far faster than numpy scalar indexing.

        Every half-height in this class's API is **absolute**: s = 2*y for a surface at
        block level y, not an offset into the volume. The arrays are indexed relative, and
        self.h0 is the only place the two meet. An earlier version leaked the relative form
        out through stances_in_column() while stance_near() compared it against an absolute
        2*y, so asking for the stance at a door returned the stance on the roof above it --
        and the town's door reachability was silently measured on its roofs.
        
    """

    def __init__(self, vol: Volume, allow_swim: bool = False, max_fall: int = FALL):
        t0 = time.perf_counter()
        self.vol = vol
        self.allow_swim = allow_swim
        self.max_fall = max_fall
        t = vol.tables()
        c = vol.codes
        sx, sy, sz = c.shape
        self.sx, self.sy, self.sz = sx, sy, sz
        self.H = 2 * sy + CLEAR          # half-levels, padded with free space on top

        low = t["lower"][c].astype(bool)
        up = t["upper"][c].astype(bool)
        surf = t["surface"][c]
        blocked = t["deadly"][c].copy()
        if not allow_swim:
            blocked |= t["liquid"][c]
        low |= blocked
        up |= blocked
        # nothing you would stand on top of: lava and fire have no top face worth
        # having, and a route over a campfire or magma block is not a route
        surf = np.where(blocked, 0, surf)
        # a 1.5-high block also fills the lower half of the cell above it
        low[:, 1:, :] |= t["tall"][c][:, :-1, :]

        occ = np.zeros((sx, self.H, sz), bool)
        occ[:, 0:2 * sy:2, :] = low
        occ[:, 1:2 * sy:2, :] = up

        # a block's top face sits at 2*y + surface
        top = np.zeros((sx, self.H, sz), bool)
        top[:, 1:2 * sy + 1:2, :] |= (surf == 1)
        top[:, 2:2 * sy + 2:2, :] |= (surf == 2)

        free = ~occ
        head = free.copy()
        for k in range(1, CLEAR):
            head[:, :self.H - k, :] &= free[:, k:, :]
            head[:, self.H - k:, :] = False
        stand = top & head

        self.h0 = 2 * vol.y0                # absolute half-height of array level 0
        self.stand_arr = stand              # kept for vectorised selection
        self.occ = occ.reshape(-1).astype(np.uint8).tobytes()
        self.stand = stand.reshape(-1).astype(np.uint8).tobytes()
        # One array for both whole-block rises: a jump needs JUMP_CLEAR at the source
        # and a stair-walk needs STAIR_CLEAR, and they are the same six half-blocks for
        # the same reason -- you are lifted while still over the column you left.
        h6 = free.copy()
        for k in range(1, max(JUMP_CLEAR, STAIR_CLEAR)):
            h6[:, :self.H - k, :] &= free[:, k:, :]
            h6[:, self.H - k:, :] = False
        self.head6 = h6.reshape(-1).astype(np.uint8).tobytes()
        self.stair = t["stair_face"][c].reshape(-1).astype(np.uint8).tobytes()
        self.ladder = t["ladder"][c].reshape(-1).astype(np.uint8).tobytes()
        self.n_stand = int(stand.sum())
        self.seconds = round(time.perf_counter() - t0, 2)

    # --- indexing --------------------------------------------------------- `h` is an
    # absolute half-height everywhere in the public API; h0 converts.
    def _h(self, x, h, z):
        return ((x - self.vol.x0) * self.H + (h - self.h0)) * self.sz \
            + (z - self.vol.z0)

    def _c(self, x, y, z):
        return ((x - self.vol.x0) * self.sy + (y - self.vol.y0)) * self.sz \
            + (z - self.vol.z0)

    def in_col(self, x, z) -> bool:
        return (self.vol.x0 <= x < self.vol.x0 + self.sx
                and self.vol.z0 <= z < self.vol.z0 + self.sz)

    def in_h(self, h) -> bool:
        return self.h0 <= h < self.h0 + self.H

    def can_stand(self, x, z, s) -> bool:
        return (self.in_col(x, z) and self.in_h(s)
                and self.stand[self._h(x, s, z)] == 1)

    def occupied(self, x, y, z) -> bool:
        """Is anything standing in this cell -- a wall, a barrel, a candle, a stair?

                The lower half only, which is what "something is here" means: a top slab hangs
                in the upper half and you walk under it, and a fern occupies neither and is not
                there at all as far as a body is concerned. Read by `floor_stances`, which needs
                to know whether a block is *piled on* the floor or hangs over it.
                
        """
        return (self.in_col(x, z) and self.in_h(2 * y)
                and self.occ[self._h(x, 2 * y, z)] == 1)

    def stances_in_column(self, x, z) -> list[int]:
        """Absolute half-heights a player can stand at in this column, lowest first."""
        if not self.in_col(x, z):
            return []
        base = (x - self.vol.x0) * self.H * self.sz + (z - self.vol.z0)
        st = self.stand
        return [self.h0 + k for k in range(self.H) if st[base + k * self.sz]]

    def ground_stance(self, x, z) -> int | None:
        """Topmost stance in a column: where you land arriving from outside."""
        st = self.stances_in_column(x, z)
        return st[-1] if st else None

    def stance_near(self, x, z, y, tol: int | None = None) -> int | None:
        """The stance closest to block level y -- how a door or a plot centre is seeded."""
        st = self.stances_in_column(x, z)
        if not st:
            return None
        best = min(st, key=lambda s: abs(s - 2 * y))
        if tol is not None and abs(best - 2 * y) > tol:
            return None
        return best

    # --- transitions ------------------------------------------------------
    def neighbours(self, x, z, s):
        """(x, z, s, needs_jump) reachable from this stance in one move."""
        out = []
        occ, stand = self.occ, self.stand
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = x + dx, z + dz
            if not self.in_col(nx, nz):
                continue
            base = (nx - self.vol.x0) * self.H * self.sz + (nz - self.vol.z0)
            hi = min(s + JUMP, self.h0 + self.H - 1)
            lo = max(s - self.max_fall, self.h0)
            for ns in range(hi, lo - 1, -1):
                if not stand[base + (ns - self.h0) * self.sz]:
                    continue
                # the target column must be clear from the landing height up past the
                # height you stepped off at, or you would walk into a wall
                if any(occ[base + (h - self.h0) * self.sz]
                       for h in range(ns + CLEAR,
                                      min(s + CLEAR, self.h0 + self.H))):
                    break
                rise = ns - s
                if rise <= STEP:
                    out.append((nx, nz, ns, False))
                # A whole-block rise -- a tread or a jump -- lifts you while you are
                # still over the column you set off from, so both need three cells of
                # headroom *here*, not two. `head6` is that test and it used to guard
                # the jump alone; the stair walked out from under it, and a person found
                # the consequence with their head. See STAIR_CLEAR.
                elif not self.head6[self._h(x, s, z)]:
                    break
                # the block whose top face is at ns sits in cell (ns-1)//2: that holds
                # for a full block (top 2y+2) and for a slab or stair (top 2y+1)
                elif self._stair_step(nx, (ns - 1) // 2, nz, dx, dz):
                    out.append((nx, nz, ns, False))
                else:
                    out.append((nx, nz, ns, True))
                break
        # ladders: climb within the column
        y = s // 2
        if self.vol.inside(x, y, z) and self.ladder[self._c(x, y, z)]:
            for ns in (s + 2, s - 2):
                ny = ns // 2
                if (self.in_h(ns) and self.vol.inside(x, ny, z)
                        and self.ladder[self._c(x, ny, z)]
                        and self.head6[self._h(x, ns, z)]):
                    out.append((x, z, ns, False))
        return out

    def _stair_step(self, x, cell, z, dx, dz) -> bool:
        """Is the block at (x, cell, z) a stair entered from its low side?

                A stair's `facing` names the side its raised quarter is on (verified against
                assets/minecraft/models/block/stairs.json), so you walk onto it freely when you
                approach from the opposite side -- when `facing` points the way you are moving.
                
        """
        if not (0 <= cell - self.vol.y0 < self.sy) or not self.in_col(x, z):
            return False
        f = self.stair[self._c(x, cell, z)]
        return bool(f) and CODE_DIR[f] == (dx, dz)

    # --- search -----------------------------------------------------------
    def flood(self, seeds, limit: int = 5_000_000, max_jumps: int | None = None,
              bounds: tuple | None = None) -> dict:
        """Every stance reachable from `seeds` -> the *fewest* jumps needed to get there.

                0-1 BFS: a walk costs 0 and goes on the front of the queue, a jump costs 1 and
                goes on the back, so the first time a stance is settled the count is minimal.
                A plain FIFO BFS records the jumps along whichever path happened to arrive
                first, which overstated the cost of reaching the town's doors by an order of
                magnitude on the first run of this.

                `bounds` is (x0, z0, x1, z1) inclusive, and confines the search to it. Asking
                "can I walk from this building's own doorstep to this room's floor" does not
                need the other fifteen buildings in the town, and flooding them anyway costs a
                settlement-sized BFS per building.
                
        """
        dist: dict[tuple[int, int, int], int] = {}
        q = deque()
        if bounds:
            bx0, bz0, bx1, bz1 = bounds

            def inside(x, z):
                return bx0 <= x <= bx1 and bz0 <= z <= bz1
        else:
            def inside(x, z):
                return True
        for (x, z, s) in seeds:
            if (self.can_stand(x, z, s) and inside(x, z)
                    and (x, z, s) not in dist):
                dist[(x, z, s)] = 0
                q.append((x, z, s))
        settled = set()
        while q and len(settled) < limit:
            x, z, s = q.popleft()
            if (x, z, s) in settled:
                continue
            settled.add((x, z, s))
            j = dist[(x, z, s)]
            for nx, nz, ns, jump in self.neighbours(x, z, s):
                k = (nx, nz, ns)
                nj = j + (1 if jump else 0)
                if max_jumps is not None and nj > max_jumps:
                    continue
                if not inside(nx, nz):
                    continue
                if nj < dist.get(k, 1 << 30):
                    dist[k] = nj
                    (q.append if jump else q.appendleft)(k)
        return dist

    def route(self, seeds, target, max_jumps: int = 0, bounds: tuple | None = None):
        """The stances walked from `seeds` to `target`, or None: the **fewest steps**
        within `max_jumps`, with the way it came recorded, so a caller that has just
        found a door reachable can say which columns the walk crosses and hold them
        open. A plain breadth-first order rather than `flood`'s 0-1 order, because the
        thing wanted here is the shortest walk and not the fewest jumps: the 0-1
        queue's front-loading of walks is a depth-first wander that came back with
        ninety columns for a door twenty blocks off the lane.
        `target` is (x, z, s); the list runs from the seed to the target."""
        parent: dict = {}
        dist: dict = {}
        q = deque()
        if bounds:
            bx0, bz0, bx1, bz1 = bounds

            def inside(x, z):
                return bx0 <= x <= bx1 and bz0 <= z <= bz1
        else:
            def inside(x, z):
                return True
        for (x, z, s) in seeds:
            if self.can_stand(x, z, s) and inside(x, z) and (x, z, s) not in dist:
                dist[(x, z, s)] = 0
                parent[(x, z, s)] = None
                q.append((x, z, s))
        target = tuple(target)
        settled = set()
        while q:
            cur = q.popleft()
            if cur in settled:
                continue
            settled.add(cur)
            if cur == target:
                break
            x, z, s = cur
            j = dist[cur]
            for nx, nz, ns, jump in self.neighbours(x, z, s):
                k = (nx, nz, ns)
                nj = j + (1 if jump else 0)
                if max_jumps is not None and nj > max_jumps:
                    continue
                if not inside(nx, nz):
                    continue
                if k not in dist:
                    dist[k] = nj
                    parent[k] = cur
                    q.append(k)
        if target not in settled:
            return None
        out, cur = [], target
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return out[::-1]

    def circulation(self, sky_open, min_size: int = 200, first=()) -> dict:
        """The settlement's walkable outdoor network: where you can actually get about.

                Derived, not seeded from a declared centre. It is the largest set of *outdoor*
                stances reachable from a single stance **without jumping once** -- the ground a
                person can walk around on. This is the distinction the spec draws and that four
                rounds missed: a door reachable from open air, given enough scrambling, is not
                the same as a door that fronts the street.

                Reachability is not symmetric here (you can fall somewhere you cannot climb
                back out of), so these are coverings rather than true components. That is the
                right bias: the network is where you can get *to*.
        """
        outdoor = []
        for x in range(self.vol.x0, self.vol.x0 + self.sx):
            for z in range(self.vol.z0, self.vol.z0 + self.sz):
                s = self.ground_stance(x, z)
                if s is None:
                    continue
                y = min(max(s // 2 - self.vol.y0, 0), self.sy - 1)
                if sky_open[x - self.vol.x0, y, z - self.vol.z0]:
                    outdoor.append((x, z, s))
        outdoor_set = set(outdoor)
        seen: set = set()
        best: dict = {}
        lead = [seed for seed in first if seed in outdoor_set]
        for seed in lead + outdoor:
            if seed in seen:
                continue
            # Nothing unvisited can be bigger than what is unvisited. On a wooded site
            # every treetop is its own little component and the first version of this
            # flooded thousands of them. Both exits are exact; this one fires far
            # earlier.
            if len(outdoor) - len(seen) <= len(best):
                break
            if not any(not jump for _nx, _nz, _ns, jump in self.neighbours(*seed)):
                seen.add(seed)
                continue
            comp = self.flood([seed], max_jumps=0)
            seen |= comp.keys()
            if len(comp) > len(best):
                best = comp
            if len(best) >= len(outdoor) // 2:
                break        # nothing left can beat it
        return best if len(best) >= min_size else {}

    def perimeter_seeds(self, inset: int = 1, step: int = 1) -> list:
        """Ground stances around the edge of the volume: 'anywhere outdoors'."""
        vol = self.vol
        xs = range(vol.x0 + inset, vol.x0 + self.sx - inset, step)
        zs = range(vol.z0 + inset, vol.z0 + self.sz - inset, step)
        out = []
        for x in xs:
            for z in (vol.z0 + inset, vol.z0 + self.sz - 1 - inset):
                s = self.ground_stance(x, z)
                if s is not None:
                    out.append((x, z, s))
        for z in zs:
            for x in (vol.x0 + inset, vol.x0 + self.sx - 1 - inset):
                s = self.ground_stance(x, z)
                if s is not None:
                    out.append((x, z, s))
        return out


# ---------------------------------------------------------------------- surfaces

#: Blocks that stand on the ground rather than being it. Minecraft's own heightmaps are
#: no help here: MOTION_BLOCKING_NO_LEAVES drops leaves but keeps *trunks*, so on a
#: wooded site `get_height` reports the top of a tree and anything that routes on it
#: climbs over the forest. Deliberately narrow -- grass_block, podzol and moss are
#: ground, whatever their names contain. `_propagule` is a sapling that hangs. Nothing
#: called it vegetation, so nothing swept it and `grade()` was entitled to call it the
#: ground under a column.
_VEG_SUFFIX = ("_leaves", "_log", "_wood", "_stem", "_hyphae", "_mushroom_block",
               "_wart_block", "_sapling", "_propagule")
_VEG_EXACT = {"bamboo", "cactus", "sugar_cane", "mushroom_stem", "melon", "pumpkin",
              "shroomlight", "nether_wart_block", "snow", "cobweb", "scaffolding"}


def _is_vegetation(state: str) -> bool:
    name = state.split("[")[0]
    return name in _VEG_EXACT or any(name.endswith(s) for s in _VEG_SUFFIX)


def ground_heights(vol: Volume) -> tuple[np.ndarray, np.ndarray]:
    """(height, wet) per column, in world y.

        `height` is the top of the ground a lane could be laid on: the highest solid block
        that is not vegetation, or the water surface where that is higher -- a route across
        a pond is a causeway at water level, not a trench along the bed. `wet` marks those
        columns so a router can prefer to go round.

        Columns with no solid ground at all come back as y0 - 1.
        
    """
    t = vol.tables()
    c = vol.codes
    solid = t["lower"][c].astype(bool) & t["upper"][c].astype(bool)
    veg = np.array([_is_vegetation(s) for s in vol.palette], bool)[c]
    liquid = t["liquid"][c]
    firm = solid & ~veg

    def topmost(mask):
        idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1, :], axis=1)
        return np.where(mask.any(axis=1), idx + vol.y0, vol.y0 - 1)

    ground = topmost(firm)
    water = topmost(liquid)
    wet = water > ground
    return np.where(wet, water, ground), wet


# ------------------------------------------------------------------ shelter / rooms

def shelter(vol: Volume) -> tuple[np.ndarray, np.ndarray]:
    """(sky_open, sealed) boolean grids over the volume's cells.

        sky_open  nothing solid anywhere above this cell in its column.
        sealed    air that cannot be reached *by air* from the volume's outer shell: a
                  fully walled interior. Air that is neither is under cover but open to the
                  world -- a porch, an arcade, the inside of a doorway.
        
    """
    t = vol.tables()
    c = vol.codes
    solid = (t["lower"][c].astype(bool) & t["upper"][c].astype(bool))
    free = ~solid
    above = np.cumsum(solid[:, ::-1, :], axis=1)[:, ::-1, :] - solid
    sky_open = free & (above == 0)

    from scipy import ndimage
    struct = ndimage.generate_binary_structure(3, 1)     # 6-connectivity
    lab, n = ndimage.label(free, structure=struct)
    shell = set()
    for sl in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        shell.update(np.unique(sl).tolist())
    shell.discard(0)
    outside = np.isin(lab, np.array(sorted(shell)))
    return sky_open, free & ~outside


#: What the ground is already made of. A space bounded by these was *found*, not built:
#: it is a cave, and it belongs to the hill rather than to whoever owns the surface
#: above it. Deliberately conservative -- cobblestone, bricks and planks are absent, so
#: a stone-lined cellar still reads as somebody's cellar.
_NATURAL = {
    "stone", "deepslate", "andesite", "diorite", "granite", "tuff", "calcite", "basalt",
    "dirt", "coarse_dirt", "rooted_dirt", "grass_block", "podzol", "mycelium", "mud",
    "clay", "gravel", "sand", "red_sand", "sandstone", "red_sandstone", "netherrack",
    "blackstone", "obsidian", "magma_block", "soul_sand", "soul_soil", "snow_block",
    "ice", "packed_ice", "blue_ice", "moss_block", "dripstone_block", "sculk",
    "amethyst_block", "budding_amethyst", "bedrock", "terracotta", "smooth_basalt",
    # A dripstone cave is made of `dripstone_block` **and** the spikes hanging off it,
    # and only the first was on this list.
    "pointed_dripstone",
    # A mangrove swamp's floor is made of roots. They are terrain generation, not a tree
    # and not a build.
    "mangrove_roots", "muddy_mangrove_roots",
}


def _is_natural(name: str) -> bool:
    # Vegetation is deliberately *not* natural here. A placed oak log is a corner post
    # and a leaf ceiling is a bower: both are somebody's build. Trees only count as
    # vegetation where the question is "where is the ground", which is ground_heights.
    return (name in _NATURAL or name.endswith("_ore") or name.endswith("_terracotta")
            or name.startswith("infested_"))


def floor_stances(nav: Nav, comp, fittings=None) -> list:
    """The **floor** of a room: the surface you walk on, without the furniture on it.

    The library already knows what it placed. `fitting()` records every cell it lays in
    `Builder.fitting_cells`, and `dais()` records its own in `Builder.dais_cells`. So:

      a stance is not floor  when the cell holding it up is a **registered fitting
                             cell** -- the barrel, the hay bale, the candle, the bench,
                             the hearth's masonry surround, the shelf.
      everything else is     the storey above, the mezzanine on joists, the loft you can
                             only jump to, and **the dais** -- which is floor, and an
                             unstepped one is six cells of floor nobody can reach and is
                             reported as such through W011.

    `fittings` is that record: a set of `(x, y, z)` cells. **None means no record**, and
    with no record nothing is furniture -- a cached town whose builder is long gone is
    read the way it was built, every solid inside it counted, and the readout says which
    it was. That is a weaker instrument and it is named rather than approximated: there
    is no vocabulary of furniture blocks here and nothing to keep in step with
    `prims.FITTING_BLOCKS`, because a block id cannot tell a hearth's surround from the
    same cobblestone laid as a counter and only one of those is furniture."""
    fit = fittings or ()
    # the block whose top face is at s sits in cell (s-1)//2 -- see `neighbours`
    return sorted((x, z, s) for (x, z, s) in comp if (x, (s - 1) // 2, z) not in fit)


def rooms(nav: Nav, sky_open: np.ndarray, region=None, min_cells: int = 4,
          fittings=None) -> list[dict]:
    """Connected components of *sheltered* standing space -- the interiors.

        Sheltered, not sealed: a building with an open doorway is not airtight, but its
        inside is still an inside. A cell is sheltered when it is not open to the sky --
        the same test a person uses walking in out of the rain.

        `fittings` is the library's record of what it placed as furniture, and is passed
        straight to `floor_stances`; None means there is no record. See its docstring.
        
    """
    vol = nav.vol
    x0, z0, x1, z1 = region or (vol.x0, vol.z0, vol.x0 + nav.sx, vol.z0 + nav.sz)

    # sheltered stances, selected vectorised: a stance at half-height s sits in cell
    # s//2, so the per-cell sky_open grid is repeated across both half-levels
    shelt = np.zeros((nav.sx, nav.H, nav.sz), bool)
    covered = ~sky_open
    shelt[:, 0:2 * nav.sy:2, :] = covered
    shelt[:, 1:2 * nav.sy:2, :] = covered
    sel = nav.stand_arr & shelt
    pts = np.argwhere(sel)
    cand = {(int(a) + vol.x0, int(d) + vol.z0, int(b) + nav.h0)
            for a, b, d in pts
            if x0 <= int(a) + vol.x0 < x1 and z0 <= int(d) + vol.z0 < z1}
    out = []
    todo = set(cand)
    while todo:
        start = todo.pop()
        comp = {start}
        q = deque([start])
        while q:
            x, z, s = q.popleft()
            for nx, nz, ns, _ in nav.neighbours(x, z, s):
                k = (nx, nz, ns)
                if k in todo:
                    todo.discard(k)
                    comp.add(k)
                    q.append(k)
        if len(comp) >= min_cells:
            xs = [p[0] for p in comp]
            zs = [p[1] for p in comp]
            ss = [p[2] for p in comp]
            # How enclosed is it? A stance you can step from straight out into open sky
            # is on an edge. A room is mostly not edge; a mesa overhang or a porch
            # mostly is. Without this, a badlands site contributes dozens of "rooms"
            # that are cliffs, and every count downstream is meaningless.
            edge = 0
            for (x, z, s) in comp:
                for nx, nz, ns, _ in nav.neighbours(x, z, s):
                    ny = min(max(ns // 2 - vol.y0, 0), nav.sy - 1)
                    if sky_open[nx - vol.x0, ny, nz - vol.z0]:
                        edge += 1
                        break
            # Was this space **made or found**? A cave under a house is sheltered,
            # enclosed standing space with a plot above it, and `plot_at` is two
            # dimensional, so it was being reported as somebody's unreachable room. What
            # tells them apart is what the walls are made of.
            made = built = 0
            for (x, z, sh) in list(comp)[:200]:
                y = sh // 2
                # walls and ceiling, not the floor: a house on a hillside stands on the
                # hillside, and counting its dirt floor would call it a cave
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                                   (0, 2, 0)):
                    # A fern is not a wall. The census used to count every neighbour
                    # that was not air, so a tuft of grass and a lantern on a post were
                    # evidence that somebody had walled this space. Only what a body
                    # actually runs into can shut a space in.
                    if not nav.occupied(x + dx, y + dy, z + dz):
                        continue
                    made += 1
                    if not _is_natural(vol.name(x + dx, y + dy, z + dz)):
                        built += 1
            out.append({"cells": len(comp),
                        "bbox": [min(xs), min(ss) // 2, min(zs),
                                 max(xs), max(ss) // 2, max(zs)],
                        "enclosure": round(1 - edge / len(comp), 2),
                        "made": round(built / made, 2) if made else 0.0,
                        "stances": sorted(comp),
                        "floor": floor_stances(nav, comp, fittings)})
    out.sort(key=lambda r: -r["cells"])
    return out


# --------------------------------------------------------------------------- light

def light(vol: Volume, cells, margin: int = 16) -> dict:
    """Propagated light level at each requested cell.

        Block light spreads from emitters, losing 1 per block, through anything that is not
        a full opaque cube. Sky light is approximated by seeding every sky-open cell at 15
        and letting it decay the same way; that slightly over-lights deep overhangs, and is
        still incomparably better than counting torches, which is all this project has
        measured until now. Computed on a box around the requested cells, so asking about
        one room costs a fraction of a second.
        
    """
    cells = list(cells)
    if not cells:
        return {}
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    zs = [c[2] for c in cells]
    bx0 = max(vol.x0, min(xs) - margin)
    bx1 = min(vol.x0 + vol.shape[0], max(xs) + margin + 1)
    by0 = max(vol.y0, min(ys) - margin)
    by1 = min(vol.y0 + vol.shape[1], max(ys) + margin + 1)
    bz0 = max(vol.z0, min(zs) - margin)
    bz1 = min(vol.z0 + vol.shape[2], max(zs) + margin + 1)

    t = vol.tables()
    sub = vol.codes[bx0 - vol.x0:bx1 - vol.x0, by0 - vol.y0:by1 - vol.y0,
                    bz0 - vol.z0:bz1 - vol.z0]
    opaque = t["opaque"][sub]
    emit = t["emit"][sub]
    solid_full = np.cumsum(opaque[:, ::-1, :], axis=1)[:, ::-1, :] - opaque
    sky = (~opaque) & (solid_full == 0)

    lev = np.zeros(sub.shape, np.int16)
    lev[sky] = 15
    lev = np.maximum(lev, emit.astype(np.int16))
    q = deque(map(tuple, np.argwhere(lev > 0)))
    sx, sy, sz = sub.shape
    while q:
        a, b, c = q.popleft()
        v = int(lev[a, b, c]) - 1
        if v <= 0:
            continue
        for da, db, dc in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)):
            na, nb, nc = a + da, b + db, c + dc
            if not (0 <= na < sx and 0 <= nb < sy and 0 <= nc < sz):
                continue
            if opaque[na, nb, nc] or lev[na, nb, nc] >= v:
                continue
            lev[na, nb, nc] = v
            q.append((na, nb, nc))
    return {(x, y, z): int(lev[x - bx0, y - by0, z - bz0])
            for (x, y, z) in cells
            if bx0 <= x < bx1 and by0 <= y < by1 and bz0 <= z < bz1}


# ---------------------------------------------------------------- stair orientation

#: Blocks with no full solid side face, which a fence, wall or pane will not join to.
#: Composed with the collision table rather than standing alone: anything the movement
#: model already calls passable -- short_grass, leaf_litter, a carpet -- is excluded
#: before this list is consulted.
_NO_SIDE_FACE = ("_leaves", "_stairs", "_slab", "_fence", "_fence_gate", "_wall",
                 "_pane", "_door", "_trapdoor", "_bars", "_sign", "_bed", "_chest",
                 "_head", "_pot", "_carpet", "_rail", "_lantern", "_torch", "_cauldron",
                 "_candle", "_button", "_pressure_plate", "_glass", "_shulker_box",
                 "_stem", "_anvil", "_amethyst_bud", "_cluster")
_NO_SIDE_FACE_EXACT = {
    "air", "cave_air", "void_air", "water", "flowing_water", "lava", "flowing_lava",
    "glass", "ice", "frosted_ice", "honey_block", "slime_block", "barrier",
    "scaffolding", "cactus", "sea_lantern", "beacon", "conduit", "hopper", "cauldron",
    "composter", "grindstone", "bell", "lectern", "brewing_stand", "enchanting_table",
    "flower_pot", "decorated_pot", "campfire", "soul_campfire", "farmland",
    "dirt_path", "snow", "cobweb", "ladder", "bamboo", "sugar_cane", "chain",
    "iron_chain", "copper_chain", "lantern", "soul_lantern", "spawner", "trial_spawner",
}


def _full_face(name: str) -> bool:
    return (name not in _NO_SIDE_FACE_EXACT
            and not any(name.endswith(s) for s in _NO_SIDE_FACE))


def unconnected_joins(vol: Volume, region=None) -> dict:
    """Connective blocks that never joined up to their neighbours.

        Found by a human walking the course, not by any check we had: fences in the town
        stand as isolated posts. The cause is ours. We write blocks with
        `doBlockUpdates=False` (world.editor, buildlib.flush) because bulk placement must
        not trigger gravity and water flow -- but that same flag is what computes the
        *connection* states. A fence placed by a player recomputes north/south/east/west
        against its neighbours; a fence we place keeps the default, which is all false.

        The same applies to walls, panes, iron bars, and to a stair's `shape`, which is how
        inner and outer corners are formed -- so no stair we have ever placed has mitred a
        corner. This counts the joins that should exist and do not.
        
    """
    c = vol.codes
    # What a fence or a wall will actually reach out to: a **full solid square face**.
    # Not everything the movement model calls solid. Leaves are solid to stand on and a
    # wall ignores them; a stair is a full cube in the collision table and a wall joins
    # it only from the side its back is on. The check stays conservative on purpose: it
    # exists to catch the block-update bug, where *every* join is missing, not to
    # adjudicate corner cases.
    cols = vol.tables()
    cube = cols["lower"].astype(bool) & cols["upper"].astype(bool)
    face = cube & np.array([_full_face(s.split("[")[0]) for s in vol.palette], bool)
    solid = face[c]
    sx, sy, sz = c.shape
    SIDES = {"north": (0, 0, -1), "south": (0, 0, 1),
             "east": (1, 0, 0), "west": (-1, 0, 0)}

    def kind(name):
        if name.endswith("_fence"):
            return "fence"
        if name.endswith("_wall") and not name.endswith("_wall_sign"):
            return "wall"
        if name.endswith("_pane") or name == "iron_bars":
            return "pane"
        return None

    codes_of = {}
    for i, s in enumerate(vol.palette):
        k = kind(s.split("[")[0])
        if k:
            codes_of[i] = (k, s.split("[")[0], parse_props(s))

    out = {"fence": {"blocks": 0, "missed_joins": 0},
           "wall": {"blocks": 0, "missed_joins": 0},
           "pane": {"blocks": 0, "missed_joins": 0},
           "examples": []}
    if codes_of:
        mask = np.isin(c, np.array(sorted(codes_of), np.uint16))
        for (a, bb, d) in np.argwhere(mask):
            k, name, props = codes_of[int(c[a, bb, d])]
            x, y, z = int(a) + vol.x0, int(bb) + vol.y0, int(d) + vol.z0
            if region and not (region[0] <= x <= region[2]
                               and region[1] <= z <= region[3]):
                continue
            out[k]["blocks"] += 1
            for side, (dx, dy, dz) in SIDES.items():
                na, nd = a + dx, d + dz
                if not (0 <= na < sx and 0 <= nd < sz):
                    continue
                nb = vol.palette[int(c[na, bb, nd])].split("[")[0]
                joins = solid[na, bb, nd] or nb == name
                if not joins:
                    continue
                v = props.get(side, "false")
                if v in ("false", "none"):
                    out[k]["missed_joins"] += 1
                    if len(out["examples"]) < 8:
                        out["examples"].append(
                            {"pos": [x, y, z], "state": vol.state(x, y, z),
                             "side": side, "neighbour": nb})
    # stair shapes: every corner we have ever built is square, not mitred
    shapes: dict[str, int] = {}
    for i, s in enumerate(vol.palette):
        if s.split("[")[0].endswith("_stairs"):
            sh = parse_props(s).get("shape", "straight")
            n = int((c == i).sum())
            shapes[sh] = shapes.get(sh, 0) + n
    out["stair_shapes"] = shapes
    return out


#: Blocks that hold something up without being solid themselves. A lantern hung on a
#: chain is supported; without these in the mask it would read as floating.
_ATTACHMENT = {"chain", "iron_chain", "copper_chain", "ladder", "scaffolding"}
#: `vine` belongs here by function and not in practice: hanging from leaves the check
#: discounts, every strand in a jungle reads as floating.

#: What this check discounts, and it is **not** the same as vegetation. The reason for
#: discounting anything is the orphaned canopy a lane leaves when it takes a trunk out
#: from under it -- leaves and the vines hanging off them. A *log* is the other thing
#: entirely: `_is_natural` already says so ("a placed oak log is a corner post"), and
#: every timber-framed build in this project is posted, plated and trussed in logs.
#: Discounting them made a tie beam invisible, so the lanterns hanging from it were
#: reported as held up by nothing.
_CANOPY_SUFFIX = ("_leaves", "_sapling")
_CANOPY_EXACT = {"vine", "glow_lichen", "hanging_roots", "moss_carpet",
                 "pale_moss_carpet", "cave_vines", "cave_vines_plant",
                 "twisting_vines", "weeping_vines", "cobweb"}


def _is_canopy(state: str) -> bool:
    name = state.split("[")[0]
    return name in _CANOPY_EXACT or any(name.endswith(s) for s in _CANOPY_SUFFIX)


#: The rest of the things that grow on something else. `_is_vegetation` is what a body
#: can stand on and `_is_canopy` is what a floating-mass check discounts, and neither of
#: them is the question "does this hang from the block beside it".
_ATTACHED_EXACT = {"cocoa", "pink_petals", "leaf_litter", "spore_blossom", "sculk_vein",
                   "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "lily_pad",
                   "sea_pickle", "glow_berries", "chorus_flower", "chorus_plant"}


def is_growing(state: str) -> bool:
    """Is this block a plant -- something that stands on, or hangs from, another block?"""
    name = state.split("[")[0]
    return (name in _ATTACHED_EXACT or _is_vegetation(state) or _is_canopy(state)
            or name in ("short_grass", "tall_grass", "fern", "large_fern", "dead_bush")
            or name.endswith(("_flower", "_bush", "_roots", "_fungus", "_sprouts")))


def unsupported_vegetation(vol: Volume, region=None) -> list:
    """Every plant in `region` that no longer has the block it hangs from or stands on.

    Support is a **chain**, which is what makes this one rule rather than a table of
    which face each plant attaches to: a plant is supported if one of its six neighbours
    is something that is not a plant, or is a plant that is itself supported. A hanging
    vine is held by the vine above it and ultimately by the block that one hangs from;
    take the block and the whole strand goes. Nothing that is standing on the ground is
    ever touched, because the ground is a neighbour that is not a plant.

    `region` is (x0, z0, x1, z1) and bounds what is **reported**; support is computed
    over the whole volume it is given, so a leaf whose trunk stands just outside the
    patch is held up by it."""
    from scipy import ndimage

    t = vol.tables()
    c = vol.codes
    grow = np.array([is_growing(s) for s in vol.palette], bool)
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool))
    veg = grow[c]
    holds = solid[c] & ~veg
    if not veg.any():
        return []
    cross = ndimage.generate_binary_structure(3, 1)
    seed = veg & ndimage.binary_dilation(holds, structure=cross)
    supported = ndimage.binary_propagation(seed, mask=veg, structure=cross)
    out = []
    for (a, b, d) in np.argwhere(veg & ~supported):
        x, y, z = int(a) + vol.x0, int(b) + vol.y0, int(d) + vol.z0
        if region and not (region[0] <= x <= region[2] and region[1] <= z <= region[3]):
            continue
        out.append((x, y, z))
    return out


def unsupported(vol: Volume, region=None, cap: int = 60, regions=None) -> list[dict]:
    """Masses of placed blocks that nothing holds up.

    Found by a human on foot, not by any check we had: a house whose roof and top course
    of walls sat one block too high, leaving an air band all the way round, and lanterns
    hung on fence posts with nothing above them. Both are the same defect seen twice --
    a piece of a build that is not attached to the build.

    Everything that is not air, liquid or vegetation is labelled by 6-connectivity; the
    largest component is the ground and everything standing on it, and every other
    component is floating. An eave, a jetty or a balcony is part of the grounded mass and
    is not reported, which is what makes this different from `columns_floating` -- the
    metric this project has never been able to read, because it counts overhangs as
    defects and one honest build scored 1209.

    Canopy is excluded outright: after a lane clears the trunks out of a jungle the
    orphaned leaves are genuinely floating, in thousands of pieces, and none of it is
    ours. Logs are *not* canopy here -- see _is_canopy."""
    from scipy import ndimage

    t = vol.tables()
    c = vol.codes
    veg = np.array([_is_canopy(s) for s in vol.palette], bool)
    att = np.array([s.split("[")[0] in _ATTACHMENT for s in vol.palette], bool)
    solid = (t["lower"].astype(bool) | t["upper"].astype(bool) | att) & ~veg
    mask = solid[c]

    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    ground = int(sizes.argmax())

    # Every component's box in one pass. The obvious version -- `argwhere(lab == i)`
    # inside the loop -- scans the whole volume once per component, which is fine while
    # the cap stops it after sixty and quadratic the moment the cap stops binding.
    # Eleven minutes against four seconds.
    boxes = ndimage.find_objects(lab)
    out = []
    for i in np.argsort(sizes)[::-1]:
        i = int(i)
        if i == 0 or i == ground or sizes[i] == 0:
            continue
        sl = boxes[i - 1]
        if sl is None:
            continue
        x0, y0, z0 = (sl[0].start + vol.x0, sl[1].start + vol.y0, sl[2].start + vol.z0)
        x1, y1, z1 = (sl[0].stop - 1 + vol.x0, sl[1].stop - 1 + vol.y0,
                      sl[2].stop - 1 + vol.z0)
        if region and not (region[0] <= x0 <= region[2] and region[1] <= z0 <= region[3]):
            continue
        if regions is not None:
            cx, cz = (int(x0) + int(x1)) // 2, (int(z0) + int(z1)) // 2
            if not any(r[0] <= cx <= r[2] and r[1] <= cz <= r[3] for r in regions):
                continue
        here = lab[sl] == i
        a, b, d = (np.argwhere(here)[0] + (sl[0].start, sl[1].start, sl[2].start))
        # Is this mass **the landscape**? A jungle has plenty of genuinely unsupported
        # landscape in it and a limestone hillside has floating stones under it; a
        # component made entirely of what the ground is made of is one of those, and it
        # is not somebody's build however far inside their plot rectangle it lies. One
        # block of andesite at y=35, thirty-four blocks under the surface of `east_row`,
        # reported identically as an error against three independently written types
        # that had never been anywhere near it.
        names = {vol.palette[k] for k in np.unique(c[sl][here])}
        out.append({"cells": int(sizes[i]),
                    "bbox": [int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)],
                    "natural": all(_is_natural(n.split("[")[0]) for n in names),
                    "example": vol.state(int(a) + vol.x0, int(b) + vol.y0,
                                         int(d) + vol.z0)})
        if len(out) >= cap:
            break
    return out


def backwards_stairs(vol: Volume, region=None, window: int = 2) -> list[dict]:
    """Stairs whose raised side faces down-slope, as built in the world.

        scripts/test_roof_rules.py catches this in the library's *output*; this catches it
        in the *world*, including stairs a model program placed by hand, which the library
        never sees. A stair is backwards when the surface one block along its `facing` is
        lower than the surface one block behind it: the raised quarter is on the low side,
        which is what makes a roof serrated and a staircase something you jump up.

        The comparison is **local**, over a +/-`window` band around the stair's own level.
        Using the highest solid block in the whole column instead -- the obvious first
        implementation, and wrong -- compares a staircase inside a house against the roof
        above it, and reported 44% of the town's stairs as backwards, most of them
        spuriously. A stair with no surface on either side within the band (a corbel, a
        chair, a sill bracket) is not judged at all.
        
    """
    t = vol.tables()
    c = vol.codes
    solid = (t["lower"][c].astype(bool) & t["upper"][c].astype(bool))
    #: What fills the bottom half of a cell and so has a top face to stand a course on:
    #: a full block, a stair, and **a bottom slab**, which `solid` does not count.
    standing = t["lower"][c].astype(bool)
    sx, sy, sz = c.shape
    faces = t["stair_face"][c]
    halves = np.array([1 if "half=top" in s else 0 for s in vol.palette], np.uint8)
    upside = halves[c]

    out = []
    for (a, b, d) in np.argwhere(faces > 0):
        x, y, z = int(a) + vol.x0, int(b) + vol.y0, int(d) + vol.z0
        if region and not (region[0] <= x <= region[2] and region[1] <= z <= region[3]):
            continue
        dx, dz = CODE_DIR[int(faces[a, b, d])]
        fa, fd, ba, bd = a + dx, d + dz, a - dx, d - dz
        if not (0 <= fa < sx and 0 <= fd < sz and 0 <= ba < sx and 0 <= bd < sz):
            continue
        lo, hi = max(b - window, 0), min(b + window + 1, sy)

        def local_top(ca, cd):
            col = solid[ca, lo:hi, cd]
            hits = np.flatnonzero(col)
            return int(hits[-1]) + lo if len(hits) else None

        ahead, behind = local_top(fa, fd), local_top(ba, bd)
        # **A stair whose facing side stands at or above its own level faces up.** The
        # raised quarter is on the side of the surface it faces, and if that surface is
        # no lower than the tread the tread cannot be pointing down anything -- whatever
        # is behind it. Without this a hipped eave beside an acacia canopy two blocks
        # up, or beside the uphill ground of a pad cut into a slope, is reported on
        # every stair of the eave: it faces the roof body at its own height and the
        # check compared the canopy behind to the roof ahead and called the roof lower.
        # The gable voice never showed it because a gable end has no stairs. Voice
        # contract, found by standing the reference type in two voices.
        if ahead is not None and ahead >= b:
            continue
        # **...and a slab ahead at the stair's own level is that same rule, read on the
        # half block it was blind to.** `local_top` reads what *fills* a cell, and a
        # bottom slab does not fill one -- but its top face is a surface, and a roof
        # laid as alternating stair and slab courses, which is what any shallow profile
        # is, then reads as a fall of one block at every stair in it. Two of the three
        # E004 a whole city carried were a ridge course and a lean-to whose next course
        # along was a slab.
        if standing[fa, b, fd]:
            continue
        # **A same-facing tread directly ahead at the same level is a run.** The flight
        # rule below exempts a tread whose flight climbs on the way it faces; a profile
        # that holds level for a course -- an upturned eave beside a shallow segment --
        # is the same run before it climbs, and its first tread reads "ahead lower" only
        # because a stair is not a surface to `local_top`. Same round, same fixture.
        same = int(faces[a, b, d])
        if 0 <= fa < sx and 0 <= fd < sz and int(faces[fa, b, fd]) == same:
            continue
        # A stack is not a slope. If the column behind the stair keeps going up past the
        # window, what is behind it is a wall or a chimney, and comparing the stair to
        # it says nothing about which way the ground falls. Without this, a flue brought
        # out at the eave reports every roof stair beside it as backwards -- and the
        # "fix" is to move the chimney to the ridge, which is a craft rule invented by a
        # checker. Gable-end stacks are a real building, and a check does not get to
        # forbid one.
        if behind is not None and hi < sy and solid[ba, hi, bd]:
            continue
        if ahead is None or behind is None or behind <= ahead:
            continue
        # **A stair that is the highest thing along its own axis is not on a slope.**
        # With both sides at or below its own level there is no fall running through
        # this cell for it to have got the wrong way round: it is the top step of a
        # stepped gable, a ridge cap or a finial, and the drop in front of it is the
        # step it exists to make. A reversed flight and a serrated roof both keep the
        # surface they came down from **behind** them, a course up, so both are still
        # reported -- which the inversion control in scripts/test_circulation.py is what
        # says.
        if behind <= b:
            continue
        # A tread whose flight carries on climbing the way it faces is a tread in a
        # flight, and what stands behind it says nothing about the slope it is on. The
        # bottom step of a stair that starts against a wall is the case: the wall is
        # higher than the step above, so column-against-column this looks backwards, and
        # it is the *only* way that step can face. they were right about the town, and
        # the fix belongs in prims.steps, which now decides a flight as one run. This is
        # the other half: the check must not then call the corrected tread wrong.
        # Verified against the deliberate inversion control in
        # scripts/test_circulation.py, which still reports every tread of a reversed
        # flight.
        nb, nd_ = b + 1, d + dz
        if (0 <= fa < sx and nb < sy and 0 <= nd_ < sz
                and int(faces[fa, nb, nd_]) == same):
            continue
        out.append({"pos": [x, y, z], "state": vol.state(x, y, z),
                    "half": "top" if upside[a, b, d] else "bottom",
                    "ahead_y": ahead + vol.y0, "behind_y": behind + vol.y0})
    return out
