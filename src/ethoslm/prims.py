"""Voxel primitives the model should not have to write itself.

Everything here exists to remove a specific observed failure. Nothing here makes a
design decision for the model — pitch, style, material and massing are all its choice."""
from __future__ import annotations

import math

# family -> (full block, stairs, slab). Roofs need all three shapes of one material.
MATERIALS = {
    "oak": ("oak_planks", "oak_stairs", "oak_slab"),
    "spruce": ("spruce_planks", "spruce_stairs", "spruce_slab"),
    "birch": ("birch_planks", "birch_stairs", "birch_slab"),
    "dark_oak": ("dark_oak_planks", "dark_oak_stairs", "dark_oak_slab"),
    "jungle": ("jungle_planks", "jungle_stairs", "jungle_slab"),
    "acacia": ("acacia_planks", "acacia_stairs", "acacia_slab"),
    "cherry": ("cherry_planks", "cherry_stairs", "cherry_slab"),
    "mangrove": ("mangrove_planks", "mangrove_stairs", "mangrove_slab"),
    "stone_brick": ("stone_bricks", "stone_brick_stairs", "stone_brick_slab"),
    "mossy_stone_brick": ("mossy_stone_bricks", "mossy_stone_brick_stairs",
                          "mossy_stone_brick_slab"),
    "cobblestone": ("cobblestone", "cobblestone_stairs", "cobblestone_slab"),
    "mossy_cobblestone": ("mossy_cobblestone", "mossy_cobblestone_stairs",
                          "mossy_cobblestone_slab"),
    "deepslate_tile": ("deepslate_tiles", "deepslate_tile_stairs", "deepslate_tile_slab"),
    "deepslate_brick": ("deepslate_bricks", "deepslate_brick_stairs", "deepslate_brick_slab"),
    "brick": ("bricks", "brick_stairs", "brick_slab"),
    "sandstone": ("sandstone", "sandstone_stairs", "sandstone_slab"),
    "red_sandstone": ("red_sandstone", "red_sandstone_stairs", "red_sandstone_slab"),
    "nether_brick": ("nether_bricks", "nether_brick_stairs", "nether_brick_slab"),
    "blackstone": ("blackstone", "blackstone_stairs", "blackstone_slab"),
    "polished_blackstone_brick": ("polished_blackstone_bricks",
                                  "polished_blackstone_brick_stairs",
                                  "polished_blackstone_brick_slab"),
    "quartz": ("quartz_block", "quartz_stairs", "quartz_slab"),
    "prismarine": ("prismarine", "prismarine_stairs", "prismarine_slab"),
    "dark_prismarine": ("dark_prismarine", "dark_prismarine_stairs", "dark_prismarine_slab"),
    "purpur": ("purpur_block", "purpur_stairs", "purpur_slab"),
    "andesite": ("andesite", "polished_andesite_stairs", "polished_andesite_slab"),
    "granite": ("granite", "polished_granite_stairs", "polished_granite_slab"),
    "diorite": ("diorite", "polished_diorite_stairs", "polished_diorite_slab"),
    "tuff_brick": ("tuff_bricks", "tuff_brick_stairs", "tuff_brick_slab"),
    "copper": ("copper_block", "cut_copper_stairs", "cut_copper_slab"),
    "weathered_copper": ("weathered_copper", "weathered_cut_copper_stairs",
                         "weathered_cut_copper_slab"),
    "oxidized_copper": ("oxidized_copper", "oxidized_cut_copper_stairs",
                        "oxidized_cut_copper_slab"),
    "mud_brick": ("mud_bricks", "mud_brick_stairs", "mud_brick_slab"),
    "end_stone_brick": ("end_stone_bricks", "end_stone_brick_stairs", "end_stone_brick_slab"),
    # Two families the voices already asked for in words and the table could not answer.
    # `thatch` is hay over bamboo shapes because hay_block has no stairs or slab of its
    # own and bamboo is the closest tone that does.
    "bamboo": ("bamboo_planks", "bamboo_stairs", "bamboo_slab"),
    "thatch": ("hay_block", "bamboo_stairs", "bamboo_slab"),
    # **Stone**: this game has `stone_stairs` and `stone_slab`, so stone is a family by
    # this table's own definition, and the table never listed it. The cost was invisible
    # until a voice asked for it: `ochre_stone_green_tile` -- the only voice a model has
    # ever written -- names `smooth_stone` for its floor, and `smooth_stone` is the
    # smooth face of stone, so `family()` stripped `smooth_` and found nothing. Five
    # committed types pass `part["voice"]["floor"]` to `material()`, through
    # `fitting()`, `steps()` and `b.block()`, and every one of them **refused by name**
    # in that voice. A city could not be built in the palette it wrote for itself, and
    # nothing had ever tried, because `stage_parts` was reading the config's empty voice
    # field rather than the one the place chose.
    "stone": ("stone", "stone_stairs", "stone_slab"),
}

#: Shapes a block id wears that its family does not. Stripped one at a time, checking
#: `MATERIALS` after each, so `dark_oak_planks` stops at `dark_oak` and never strips on
#: to `oak` -- which is a different material and a different colour.
_SHAPE_SUFFIX = ("_stairs", "_slab", "_planks", "_bricks", "_brick", "_tiles", "_tile",
                 "_block", "_log", "_wood", "_wall")
_SHAPE_PREFIX = ("stripped_", "cut_", "smooth_", "polished_", "chiseled_")

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

VEGETATION = ("_leaves", "_log", "_wood", "_sapling", "_mushroom", "vine", "moss_carpet",
              "grass", "fern", "flower", "bush", "bamboo", "cactus", "sugar_cane",
              "snow", "dead_bush", "tulip", "orchid", "allium", "daisy", "cornflower",
              "poppy", "dandelion", "lily", "azalea", "pumpkin", "melon", "_stem",
              "pink_petals", "sculk", "seagrass", "kelp", "coral", "firefly_bush",
              "_propagule")


def family(name: str) -> str | None:
    """The material family a name belongs to, or None if there is no such family.

        A family is a material in all three shapes. A caller often names a *block* instead
        -- `bamboo_planks` for bamboo, `stripped_spruce_log` for spruce -- and that is the
        same material said differently, so the shape is stripped off and the answer checked
        against the table at every step. `hay_block` has no family: there are no hay stairs
        and no hay slab, and nothing here will invent them.
        
    """
    n = str(name).split("[")[0].split(":")[-1]
    for _ in range(6):
        if n in MATERIALS:
            return n
        # A family is named in the singular and a block in the plural -- the family is
        # `stone_brick` and the block is `stone_bricks`, and `bricks` is `brick`.
        if n.endswith("s") and n[:-1] in MATERIALS:
            return n[:-1]
        for s in sorted(_SHAPE_SUFFIX, key=len, reverse=True):
            if n.endswith(s) and len(n) > len(s):
                n = n[: -len(s)]
                break
        else:
            # A shape word in the *middle* of a name: copper's stairs and slabs are
            # `oxidized_cut_copper_stairs`, so stripping the suffix leaves
            # `oxidized_cut_copper`, which the prefix rule cannot see. Voice contract,
            # A2: the palette clause read a green copper roof as belonging to no family
            # and E014 and S001 had been blind to it the same way.
            if "_cut_" in n:
                n = n.replace("_cut_", "_", 1)
                continue
            for p in _SHAPE_PREFIX:
                if n.startswith(p) and len(n) > len(p):
                    n = n[len(p):]
                    break
            else:
                return None
    return n if n in MATERIALS else None


def material(name: str) -> tuple[str, str, str]:
    """(full, stairs, slab) for a material family. **Refuses an unknown family.**

    `family()` resolves a block id to its material first, so `bamboo_planks` and
    `stripped_spruce_log` are the bamboo and spruce families rather than errors."""
    fam = family(name)
    if fam is None:
        raise ValueError(
            f"{name!r} is not a material family: there are no stairs and no slab of it, "
            f"so a roof, a tread or a cap cannot be made from it. Name one of "
            f"{', '.join(sorted(MATERIALS))} -- or, if you want this exact block "
            f"somewhere a shape is not needed, place it yourself.")
    return MATERIALS[fam]


def solid(name: str) -> str:
    """The full block of a material family, or a block id given literally.

        The other half of `material()`'s refusal. A wall, a floor or a post needs one cube
        and no shapes, and a palette is allowed to name that cube directly -- so this
        accepts `white_concrete_powder` where `material()` will not, and still refuses a
        name that is neither a family nor a real block.
        
    """
    fam = family(name)
    if fam is not None:
        return MATERIALS[fam][0]
    n = str(name).split("[")[0].split(":")[-1]
    if _has_block("minecraft:" + n):
        return str(name)
    raise ValueError(f"{name!r} is neither a material family nor a block in this "
                     f"version of Minecraft")


# Every one of the fourteen committed types therefore wrote its own --
# `"dark_oak_planks"`, `"quartz_" "bricks"`, `"cobblestone_wall"`, `"dark_oak_door"`,
# `"stripped_dark_oak_log[axis=y]"` -- 244 block literals across the fourteen files, and
# *that* is what welded a type to one palette. So the library owns the shape the way it
# owns everything else physical: a type names a **role and a shape** and this says which
# block. It refuses by name where the game has no such block.

#: How each shape is spelled, in the order the candidates are tried. The first that is a
#: real block in this version wins; a family with none of them has no such shape. The
#: `{f}` is the family name and `{full}` the family's own cube.
_SHAPE_FORMS = {
    "full":     ("{full}",),
    "stairs":   (),          # from MATERIALS, which is where the pair is registered
    "slab":     (),
    "wall":     ("{f}_wall", "{full_stem}_wall"),
    "fence":    ("{f}_fence",),
    "gate":     ("{f}_fence_gate",),
    "door":     ("{f}_door",),
    "trapdoor": ("{f}_trapdoor",),
    "button":   ("{f}_button",),
    # An upright member: a log, a pillar, or the cube where the family has neither.
    "post":     ("{f}_pillar", "stripped_{f}_log", "{f}_log", "{full_stem}_pillar",
                 "{full}"),
    # The stripped, pale face of a timber; the cube for anything that is not a timber.
    "bare":     ("stripped_{f}_log", "stripped_{f}_wood", "{full}"),
    # An ornamented face -- the chiselled, carved or patterned variant.
    "accent":   ("chiseled_{f}", "chiseled_{full_stem}", "chiseled_{full}",
                 "{f}_bricks", "cracked_{f}_bricks", "{full}"),
    # A fine, close-grained face -- the smooth or polished variant.
    "fine":     ("smooth_{f}", "polished_{f}", "{full_stem}_bricks", "smooth_{full}",
                 "polished_{full}", "{full}"),
}

#: The shapes a type may ask for. `stairs` and `slab` come from `MATERIALS`, because a
#: family *is* a material in those three shapes and that table is where it is
#: registered.
SHAPES = tuple(sorted(set(_SHAPE_FORMS) | {"stairs", "slab"}))


def shape(name: str, kind: str = "full") -> str:
    """The block of one shape of one material family.

        `name` is a family or any block of one -- `spruce`, `stripped_spruce_log`, whatever
        a voice's role happens to say -- and `kind` is one of `SHAPES`. Refuses by name
        rather than substituting: a type that asks for a fence of a stone family is asking
        for something the game does not have, and the answer is the refusal and not a wall.
        
    """
    if kind not in SHAPES:
        raise ValueError(f"a shape is one of {', '.join(SHAPES)}, not {kind!r}")
    fam = family(name)
    if fam is None:
        # **A bare block that belongs to no family**, which is a thing a voice may
        # legitimately hold: `voices.SOLID` says `roles.floor` need not be a family
        # "because it is only ever laid as a cube". `ochre_stone_green_tile`, whose
        # floor is `smooth_stone` -- is the only one of eight whose floor is not a
        # family. So every shop-house in a city crashed the moment the voice actually
        # reached the types, and nothing had ever noticed because nothing had ever
        # passed that voice in. `smooth_stone` has a slab and no stairs, and what comes
        # back is `smooth_stone_slab` and a refusal by name. The cube of a thing that is
        # already a cube is itself.
        cand = name if kind == "full" else f"{name}_{kind}"
        if _has_block("minecraft:" + cand):
            return cand
        raise ValueError(
            f"{name!r} is not a material family and Minecraft "
            f"{registry_version()} has no {cand}, so it has no {kind}. Name a "
            f"family -- one of {', '.join(sorted(MATERIALS))} -- or ask for a shape "
            f"this block has")
    if kind not in SHAPES:
        raise ValueError(f"a shape is one of {', '.join(SHAPES)}, not {kind!r}")
    full, stairs, slab = MATERIALS[fam]
    if kind == "stairs":
        return stairs
    if kind == "slab":
        return slab
    stem = full[:-1] if full.endswith("s") else full
    for form in _SHAPE_FORMS[kind]:
        cand = form.format(f=fam, full=full, full_stem=stem)
        if _has_block("minecraft:" + cand):
            return cand
    raise ValueError(
        f"there is no {fam} {kind} in Minecraft {registry_version()}: a {kind} of this "
        f"material does not exist, so nothing here will substitute one. Ask for a "
        f"shape this family has, or choose the role whose family does")


def registry_version() -> str:
    from . import registry
    return registry.VERSION


def _has_block(block_id: str) -> bool:
    """Is this a real block in this Minecraft version? Offline, against the server's own
    registry — the same data `preflight` rejects a program with."""
    from . import registry
    try:
        return not registry.check_all([block_id])
    except Exception:
        return False


def _occupies_cell(state: str) -> bool:
    """Does a body collide with this block at all -- is it something the ground can be?

        The same `lower or upper` the offline heightmap is computed from, so `grade()`
        stops exactly where `surface_heights` would have put the surface. `_is_surface`
        below is the stricter `lower and upper`, which a slab fails: sounding with that
        would walk a probe straight through the top of a slabbed roof.
        
    """
    if not state or state in ("air", "cave_air", "void_air"):
        return False
    from . import observe
    name = state.split("[")[0].split(":")[-1]
    lower, upper, _surface, _rule = observe._classify(name, observe.parse_props(state))
    return bool(lower or upper)


def _is_surface(state: str) -> bool:
    """Does this block fill its cell -- is it ground you stand *on top of*?

        Deferred to `observe`, deliberately: the rule that decides which way a tread faces
        and the rule the linter judges it by have to be the same rule, or the library is
        measuring the world differently from the thing that reads the world back. This is
        observe's `lower and upper`, which is what backwards_stairs compares columns with.
        
    """
    if not state or state in ("air", "cave_air", "void_air"):
        return False
    from . import observe
    name = state.split("[")[0]
    lower, upper, _surface, _rule = observe._classify(name, observe.parse_props(state))
    return bool(lower and upper)


class Primitives:
    """Mixed into Builder. Assumes self.place_block / self.get_height exist."""

    # ---------------------------------------------------------------- curves
    def disc(self, cx: int, y: int, cz: int, r: float, block: str) -> None:
        """Filled circle in the horizontal plane. Rasterised by true distance so the
        outline is smooth rather than lumpy."""
        ri = int(math.ceil(r))
        for dx in range(-ri, ri + 1):
            for dz in range(-ri, ri + 1):
                if math.hypot(dx, dz) <= r + 0.35:
                    self.place_block(cx + dx, y, cz + dz, block)

    def ring(self, cx: int, y: int, cz: int, r: float, block: str, thickness: int = 1) -> None:
        """Circle outline of constant visual thickness, guaranteed 8-connected.

                Walking the angle and rounding is what keeps the thickness even; a distance-band
                test gives a ring that is fat on the axes and thin on the diagonals.
                
        """
        seen = set()
        steps = max(16, int(8 * r))
        for i in range(steps):
            a = 2 * math.pi * i / steps
            for t in range(thickness):
                rr = r - t
                if rr < 0:
                    continue
                seen.add((int(round(cx + rr * math.cos(a))),
                          int(round(cz + rr * math.sin(a)))))
        for (x, z) in seen:
            self.place_block(x, y, z, block)

    def cylinder(self, cx: int, y0: int, cz: int, r: float, height: int, block: str,
                 hollow: bool = True, thickness: int = 1) -> None:
        for i in range(height):
            if hollow:
                self.ring(cx, y0 + i, cz, r, block, thickness)
            else:
                self.disc(cx, y0 + i, cz, r, block)

    def sphere(self, cx: int, cy: int, cz: int, r: float, block: str,
               hollow: bool = False) -> None:
        ri = int(math.ceil(r))
        for dx in range(-ri, ri + 1):
            for dy in range(-ri, ri + 1):
                for dz in range(-ri, ri + 1):
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d <= r + 0.35 and (not hollow or d >= r - 0.9):
                        self.place_block(cx + dx, cy + dy, cz + dz, block)

    def dome(self, cx: int, y0: int, cz: int, r: float, block: str,
             hollow: bool = True) -> None:
        """Upper hemisphere sitting on y0."""
        ri = int(math.ceil(r))
        for dy in range(0, ri + 1):
            for dx in range(-ri, ri + 1):
                for dz in range(-ri, ri + 1):
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d <= r + 0.35 and (not hollow or d >= r - 0.9):
                        self.place_block(cx + dx, y0 + dy, cz + dz, block)

    def line(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int,
             block: str) -> None:
        """3D Bresenham. One block per step, no gaps, no doubling."""
        dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        sz = 1 if z1 > z0 else -1
        n = max(dx, dy, dz)
        if n == 0:
            self.place_block(x0, y0, z0, block)
            return
        for i in range(n + 1):
            t = i / n
            self.place_block(int(round(x0 + (x1 - x0) * t)),
                             int(round(y0 + (y1 - y0) * t)),
                             int(round(z0 + (z1 - z0) * t)), block)

    def path(self, points: list[tuple[int, int]], width: float, block: str,
             follow_ground: bool = True, y: int | None = None) -> None:
        """A path of even width along a polyline, laid onto the ground."""
        if len(points) < 2:
            return
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        pad = int(width) + 2
        half = width / 2.0

        def dist_to_polyline(px, pz):
            best = 1e9
            for i in range(len(points) - 1):
                ax, az = points[i]
                bx, bz = points[i + 1]
                vx, vz = bx - ax, bz - az
                L2 = vx * vx + vz * vz
                if L2 == 0:
                    d = math.hypot(px - ax, pz - az)
                else:
                    t = max(0.0, min(1.0, ((px - ax) * vx + (pz - az) * vz) / L2))
                    d = math.hypot(px - (ax + t * vx), pz - (az + t * vz))
                best = min(best, d)
            return best

        for x in range(min(xs) - pad, max(xs) + pad + 1):
            for z in range(min(zs) - pad, max(zs) + pad + 1):
                if dist_to_polyline(x, z) <= half:
                    yy = self.get_height(x, z) if follow_ground else y
                    self.place_block(x, yy, z, block)
                    for k in range(1, 4):
                        self.place_block(x, yy + k, z, "air")

    # ----------------------------------------------------------------- roofs Above
    # about 2 rise per 1 run each course leaves a one-block flat tread under a tall
    # vertical face, and the roof reads as a stepped ziggurat rather than a steep slope.
    # Clamp rather than let a caller ask for something that cannot look right.
    MAX_RATIO = 2.0
    MIN_RATIO = 0.25

    def _clamp_pitch(self, rise: int, run: int) -> tuple[int, int]:
        rise, run = max(1, int(rise)), max(1, int(run))
        if rise / run > self.MAX_RATIO:
            return 2, 1
        if rise / run < self.MIN_RATIO:
            return 1, 4
        return rise, run

    def _slope_profile(self, span: int, rise: int, run: int) -> list[tuple[int, str]]:
        """For each step out from the eave, the height and whether it steps this course.

                Returns [(height_offset, "stair"|"slab"|"full"), ...]. This is what makes pitch
                real: 1-in-1 is all stairs, 1-in-2 alternates slab and stair (a shallow roof),
                2-in-1 stacks a full block under each stair (a steep roof).

                One slope of one pitch is the one-segment case of `_segment_profile`, and it is
                written that way rather than duplicated so that the six presets and a caller's
                own `profile=` cannot drift apart: every preset roof in the record is this call.

                It does **not** clamp, and that is not an oversight: every caller inside this
                module clamps first, and one of them -- the gambrel's shallow upper slope --
                asks for (1, run*2), which on a (1,3) gambrel is 1-in-6 and below MIN_RATIO.
                Clamping here would redraw every gambrel in the record.
                
        """
        return self._segment_profile(span, [(rise, run)], clamp=False)

    def _segment_profile(self, span: int, segments,
                         clamp: bool = True) -> list[tuple[int, str]]:
        """The same, for a slope that changes pitch on the way up.

                `segments` is [(rise, run), ...] from the eave to the ridge. Each segment holds
                its pitch for `run` steps out from where the one before it ended, gaining `rise`
                over that run; the last segment repeats for as long as the span lasts, so a
                profile shorter than the roof is a statement about the eave rather than a hole.

                This is the whole of what `profile=` means, and it is why an irimoya can be
                asked for at all: "shallow (1,2) then steep (2,1) at the ridge" is two segments
                and nothing else in the library has to know what a Japanese roof is.
                
        """
        segs = [(self._clamp_pitch(*s) if clamp else (max(1, int(s[0])),
                                                      max(1, int(s[1]))))
                for s in (segments or [(1, 1)])]
        out: list[tuple[int, str]] = []
        base, idx, k = 0, 0, 0
        for _d in range(max(0, span)):
            rise, run = segs[min(idx, len(segs) - 1)]
            h = base + (k * rise) // run
            h_next = base + ((k + 1) * rise) // run
            out.append((h, "stair" if h_next > h else "slab"))
            k += 1
            if k >= run and idx < len(segs) - 1:
                base += rise
                idx += 1
                k = 0
        return out

    #: What each end of the ridge does, for the four presets that have ends. `ends=None`
    #: means "whatever this style has always done", which is what keeps the presets
    #: byte-identical -- see `roof`.
    _STYLE_ENDS = {"gable": ("gable", "gable"), "gambrel": ("gable", "gable"),
                   "hip": ("hip", "hip"), "mansard": ("hip", "hip")}

    #: The four things the end of a ridge can be.
    END_KINDS = ("gable", "hip", "half-hip", "irimoya")

    #: What an eave can do beyond stopping. "flared" holds the outermost course level
    #: for one step longer -- the eave segment's run plus one; "upturned" lifts the
    #: courses over the overhang a block and leaves the course they lift off standing,
    #: so the tip is up and the eave is still one mass.
    EAVES = ("straight", "flared", "upturned")

    def roof(self, x0: int, z0: int, x1: int, z1: int, y: int, mat: str,
             style: str = "gable", axis: str = "z", pitch: tuple[int, int] = (1, 1),
             overhang: int = 1, solid_fill: bool = True, *,
             profile=None, ends=None, eave: str = "straight", tiers: int = 1,
             rise_max: int | None = None, _stop_out: int | None = None) -> int:
        """Build a roof over the rectangle (x0,z0)-(x1,z1) with its eaves at height y.

        style: "gable" | "hip" | "gambrel" | "mansard" | "shed" | "flat"
        axis:  "z" ridge runs east-west (slopes face north and south)
               "x" ridge runs north-south (slopes face east and west)
               for "shed", the direction the roof slopes down toward: "n"|"s"|"e"|"w"
        pitch: (rise, run). (1,1)=45 deg, (1,2)=shallow, (2,1)=steep, (1,3)=very shallow.
        Returns the y of the ridge, so you can put a chimney through it.

            profile   [(rise, run), ...] from the eave to the ridge, overriding `pitch`.
                      Each segment holds for its own `run` and the last one repeats.
                      "shallow then steep at the ridge" is [(1,2), (2,1)].
            ends      (near, far) from "gable", "hip", "half-hip", "irimoya" -- what
                      each end of the ridge does. A half-hip is gabled below and hipped
                      at the top; an **irimoya** is the other way round, hipped below
                      with a gable above it set back one course, which is the roof a
                      Japanese hall has and the one this library could not draw.
            eave      "straight", "flared" (the eave course held one step longer, so
                      the roof lands shallower where it overhangs) or "upturned" (the
                      overhang lifted a block above the course it leaves).
            tiers     stack the roof this many times, each tier pulled in from the one
                      below with a one-block wall course between them.

        `ends=None` is the preset's own ends and is what every stored program replays
        with; naming `ends` explicitly is what turns the ridge cap on for a roof that
        has a ridge line, which no preset but "gable" has ever had.

        `_stop_out` is internal: it cuts the slope off that many courses out from the
        eave and is how `tiers` draws every tier but the top one."""
        if int(tiers) > 1:
            return self._roof_tiers(x0, z0, x1, z1, y, mat, style, axis, pitch,
                                    overhang, solid_fill, profile, ends, eave,
                                    int(tiers), rise_max=rise_max)
        full, stairs, slab = material(mat)
        rx0, rx1 = x0 - overhang, x1 + overhang
        rz0, rz1 = z0 - overhang, z1 + overhang

        if style == "flat":
            for x in range(rx0, rx1 + 1):
                for z in range(rz0, rz1 + 1):
                    self.place_block(x, y, z, full)
            for x in range(rx0, rx1 + 1):
                self.place_block(x, y + 1, rz0, slab)
                self.place_block(x, y + 1, rz1, slab)
            for z in range(rz0, rz1 + 1):
                self.place_block(rx0, y + 1, z, slab)
                self.place_block(rx1, y + 1, z, slab)
            return y + 1

        rise, run = self._clamp_pitch(*pitch)
        ridge_y = y

        if style == "shed":
            # `axis` is the direction the roof slopes *down* toward. A stair's `facing`
            # points at its raised quarter (see _roof_block), which is up-slope — the
            # opposite direction.
            facing = {"n": "south", "s": "north", "e": "west", "w": "east"}[axis]
            span = (rz1 - rz0 + 1) if axis in ("n", "s") else (rx1 - rx0 + 1)
            prof = self._roof_profile(span, style, rise, run, profile, eave,
                                      rise_max=rise_max)
            for i, (h, kind) in enumerate(prof):
                if _stop_out is not None and i > _stop_out:
                    continue
                lift, kind = (self._upturn(prof, h, kind, overhang)
                              if eave == "upturned" and i < overhang else (0, kind))
                if axis in ("n", "s"):
                    # slope descends toward `axis`, so the eave (h=0) is on that side
                    z = rz1 - i if axis == "s" else rz0 + i
                    cells = [(x, z) for x in range(rx0, rx1 + 1)]
                else:
                    x = rx1 - i if axis == "e" else rx0 + i
                    cells = [(x, z) for z in range(rz0, rz1 + 1)]
                for (cx, cz) in cells:
                    self._roof_block(cx, y + h + bool(lift), cz, kind, stairs, slab,
                                     full, facing)
                    if lift:
                        self.place_block(cx, y + h, cz, full)
                    if solid_fill:
                        for yy in range(y, y + h):
                            self.place_block(cx, yy, cz, full)
                ridge_y = max(ridge_y, y + h + bool(lift))
            return ridge_y

        if style in ("gable", "gambrel", "mansard", "hip"):
            span_z = rz1 - rz0 + 1
            span_x = rx1 - rx0 + 1
            half_z = (span_z + 1) // 2
            half_x = (span_x + 1) // 2

            pz = self._roof_profile(half_z, style, rise, run, profile, eave,
                                    rise_max=rise_max)
            px = self._roof_profile(half_x, style, rise, run, profile, eave,
                                    rise_max=rise_max)
            end_lo, end_hi = self._roof_ends(style, ends)
            top_h = max(p[0] for p in (pz + px)) if (pz or px) else 0
            hs: dict[tuple[int, int], int] = {}

            for x in range(rx0, rx1 + 1):
                for z in range(rz0, rz1 + 1):
                    dz = min(z - rz0, rz1 - z)
                    dx = min(x - rx0, rx1 - x)
                    if _stop_out is not None and min(dz, dx) > _stop_out:
                        continue
                    hz, kz = pz[min(dz, len(pz) - 1)]
                    fz = "south" if (z - rz0) <= (rz1 - z) else "north"
                    hx, kx = px[min(dx, len(px) - 1)]
                    fx = "east" if (x - rx0) <= (rx1 - x) else "west"
                    if axis == "z":
                        # the ridge runs along x, so the z faces are the slopes and the
                        # x faces are the two ends
                        side, end = (dz, hz, kz, fz), (dx, hx, kx, fx)
                        end_kind = end_lo if (x - rx0) <= (rx1 - x) else end_hi
                    else:
                        side, end = (dx, hx, kx, fx), (dz, hz, kz, fz)
                        end_kind = end_lo if (z - rz0) <= (rz1 - z) else end_hi
                    h, kind, face = self._roof_cell(side, end, end_kind, axis, top_h)
                    # The ridge column of an odd span has no up-slope side: the tie rule
                    # hands it a stair facing one way, and under a profile whose last
                    # segment climbs steeply the neighbour on that side stands lower and
                    # the one behind higher -- a tread facing down its own roof, which
                    # is what E004 reads. A full course there, under the cap. Only where
                    # a profile was given, so every preset roof in the record is the
                    # roof it always was.
                    if profile and kind == "stair":
                        on_ridge = ((axis == "z" and (z - rz0) == (rz1 - z))
                                    or (axis == "x" and (x - rx0) == (rx1 - x)))
                        if on_ridge:
                            kind = "full"
                    lift, kind = (self._upturn(pz if face in ("north", "south") else px,
                                               h, kind, overhang)
                                  if eave == "upturned" and min(dz, dx) < overhang
                                  else (0, kind))
                    self._roof_block(x, y + h + bool(lift), z, kind, stairs, slab,
                                     full, face)
                    if lift:
                        self.place_block(x, y + h, z, full)
                    if solid_fill:
                        for yy in range(y, y + h):
                            self.place_block(x, yy, z, full)
                    hs[(x, z)] = y + h + bool(lift)
                    ridge_y = max(ridge_y, y + h + bool(lift))

            # Cap the ridge line with slabs so it does not come to a jagged point. A
            # preset gets exactly the cap it always got -- only "gable" has one --
            # because `ends=None` is the preset. Ask for ends explicitly and any end
            # that is not a full hip means the roof has a ridge *line* rather than a
            # point, and a line wants capping: an irimoya without this is a row of
            # single blocks along the top with air beside them.
            capped = (style == "gable") if ends is None else \
                any(e != "hip" for e in (end_lo, end_hi))
            if capped and _stop_out is None:
                # ...on the columns that actually reached the ridge, and no others. On a
                # gable that is the whole ridge band and the cap is what it always was;
                # on a hipped or irimoya end the ridge stops short, and a cap run to the
                # eave is a line of slabs hanging in the air over the hip.
                if axis == "z":
                    zc0, zc1 = rz0 + half_z - 1, rz1 - half_z + 1
                    band = [(x, z) for x in range(rx0, rx1 + 1)
                            for z in range(min(zc0, zc1), max(zc0, zc1) + 1)]
                else:
                    xc0, xc1 = rx0 + half_x - 1, rx1 - half_x + 1
                    band = [(x, z) for z in range(rz0, rz1 + 1)
                            for x in range(min(xc0, xc1), max(xc0, xc1) + 1)]
                for (x, z) in band:
                    if hs.get((x, z)) == ridge_y:
                        self.place_block(x, ridge_y + 1, z, slab)
            return ridge_y
        raise ValueError(f"unknown roof style {style!r}")

    def _roof_ends(self, style: str, ends) -> tuple[str, str]:
        """What each end of the ridge does. `None` is the style's own, which is what
        keeps every stored program's roof byte-identical."""
        if ends is None:
            return self._STYLE_ENDS.get(style, ("gable", "gable"))
        if isinstance(ends, str):
            ends = (ends, ends)
        a, b = (list(ends) + list(ends))[:2]
        for e in (a, b):
            if e not in self.END_KINDS:
                raise ValueError(f"a roof end is one of {self.END_KINDS}, not {e!r}")
        return a, b

    def _roof_profile(self, n: int, style: str, rise: int, run: int, segments,
                      eave: str, rise_max: int | None = None) -> list[tuple[int, str]]:
        """One slope of this roof, out from the eave."""
        if rise_max is not None:
            for k in range(1, 17):
                segs = ([(a, b * k) for a, b in segments] if segments else None)
                prof = self._roof_profile(n, style, rise, run * k, segs, eave)
                if not prof or max(h for h, _kind in prof) <= int(rise_max):
                    return prof
            return prof
        if segments:
            prof = self._segment_profile(n, segments)
        elif style == "gambrel":
            # steep lower third, shallow above: more usable attic volume
            steep = self._slope_profile(n, *self._clamp_pitch(max(rise, 2), run))
            shal = self._slope_profile(n, 1, max(run * 2, 2))
            knee = max(1, n // 3)
            prof = []
            for d in range(n):
                if d < knee:
                    prof.append(steep[d])
                else:
                    base = steep[knee - 1][0]
                    prof.append((base + shal[d - knee][0], shal[d - knee][1]))
        elif style == "mansard":
            steep = self._slope_profile(n, *self._clamp_pitch(max(rise, 2), run))
            knee = max(1, (2 * n) // 3)
            prof = [steep[d] if d < knee else (steep[knee - 1][0], "flat")
                    for d in range(n)]
        else:
            prof = self._slope_profile(n, rise, run)
        if eave == "flared" and n > 0:
            # the eave segment's run, plus one: the outermost course is held level for
            # one step longer and the whole slope lands one step further in
            prof = [(prof[0][0], "slab")] + prof[:n - 1]
        return prof

    def _roof_cell(self, side, end, end_kind: str, axis: str, top_h: int):
        """(height, kind, facing) for one column, given both slopes and what the end is.

                `side` and `end` are each (distance, height, kind, facing). The end of a ridge
                is the only place the four kinds differ, and each is one rule:

                    gable     the end does not slope: the side profile decides everything, and
                              the end wall carries straight up to the ridge.
                    hip       whichever edge is nearer decides. **The tie goes to the z
                              profile**, which is what the old hip branch did by comparing
                              `dz <= dx` whatever the axis was, and is why this reproduces every
                              hip and mansard in the record block for block.
                    half-hip  gabled below, hipped at the top: the end slope starts a knee up,
                              so it only bites near the ridge.
                    irimoya   hipped below, gabled above, and the gable set back one course --
                              the outermost end column stays hipped, so the upper gable stands
                              on the lower hip's own slope rather than hanging off its edge.
                
        """
        sd, sh, sk, sf = side
        ed, eh, ek, ef = end
        if end_kind == "gable":
            return sh, sk, sf
        nearer = (ed < sd) if axis == "z" else (ed <= sd)
        if end_kind == "hip":
            return (eh, ek, ef) if nearer else (sh, sk, sf)
        if end_kind == "half-hip":
            knee = max(1, (top_h + 2) // 3)
            return (eh + knee, ek, ef) if eh + knee < sh else (sh, sk, sf)
        # irimoya
        brk = max(1, (top_h + 1) // 2)
        if eh >= brk and ed >= 1:
            return sh, sk, sf
        return (eh, ek, ef) if nearer else (sh, sk, sf)

    def _roof_tiers(self, x0, z0, x1, z1, y, mat, style, axis, pitch, overhang,
                    solid_fill, profile, ends, eave, tiers: int,
                    rise_max: int | None = None) -> int:
        """Stack the roof `tiers` times, a one-block wall course between the tiers.

                A two-tiered roof is not a taller roof. It is a low roof round the outside of
                the building, a short wall standing on the rectangle that roof leaves, and
                another roof on top of that -- which is the thing `roof()` could not say, and
                the reason a temple hall had to be rasterised by hand.

                Each tier below the top is cut off `band` courses in from its own eave and the
                next tier starts on the rectangle that is left. A tier with no rectangle left
                to stand on is the last tier, whatever `tiers` said: refusing to draw the top
                of a roof would leave a hole in it.

                **The rectangle between the tiers is filled, not left open.** A tier drawn with
                `_stop_out` places nothing inside its own cut, so a two-tier roof used to hand
                back a lantern-shaped cavity between the lower roof and the upper one -- open to
                the room below, enclosed by the wall course, and reported by W007 as a void and
                by the room finder as a room nobody can walk into. It is a roof, and the inside
                of a roof is solid, so this fills it with the same block the wall course between
                the tiers is made of. `solid_fill=False` is a caller asking for a hollow roof
                and gets one, here as everywhere else.
                
        """
        ridge = y
        a0, b0, a1, b1 = int(x0), int(z0), int(x1), int(z1)
        yy = int(y)
        full, _stairs, _slab = material(mat)
        rise, run = self._clamp_pitch(*pitch)
        for t in range(max(1, int(tiers))):
            band = max(1, min(a1 - a0, b1 - b0) // 6)
            last = (t == tiers - 1) or (a1 - a0) - 2 * band < 4 \
                or (b1 - b0) - 2 * band < 4
            # the cap is on the whole stack: what this tier may rise is what is left of
            # it above the tiers already drawn
            left = None if rise_max is None else max(1, int(rise_max) - (yy - int(y)))
            if last:
                return max(ridge, self.roof(a0, b0, a1, b1, yy, mat, style=style,
                                            axis=axis, pitch=pitch, overhang=overhang,
                                            solid_fill=solid_fill, profile=profile,
                                            ends=ends, eave=eave, rise_max=left))
            # the lower tier: the same roof, stopped `band` courses in from its eave
            prof = self._roof_profile(band + overhang + 1, style, rise, run, profile,
                                      eave, rise_max=left)
            cut = prof[min(band + overhang, len(prof) - 1)][0]
            self.roof(a0, b0, a1, b1, yy, mat, style=style, axis=axis, pitch=pitch,
                      overhang=overhang, solid_fill=solid_fill, profile=profile,
                      ends=ends, eave=eave, rise_max=left, _stop_out=band + overhang)
            ridge = max(ridge, yy + cut)
            a0, b0, a1, b1 = a0 + band, b0 + band, a1 - band, b1 - band
            # the body of the roof between this tier and the next, so there is no cavity
            # between them -- everything from the eave course of the tier just drawn up
            # to the level the wall course stands at
            if solid_fill:
                for x in range(a0 + 1, a1):
                    for z in range(b0 + 1, b1):
                        for yv in range(yy, yy + cut + 2):
                            self.place_block(x, yv, z, full)
            # the wall course the tier above stands on
            for x in range(a0, a1 + 1):
                for z in (b0, b1):
                    self.place_block(x, yy + cut + 1, z, full)
            for z in range(b0, b1 + 1):
                for x in (a0, a1):
                    self.place_block(x, yy + cut + 1, z, full)
            yy = yy + cut + 2
        return ridge

    def _upturn(self, prof, h: int, kind: str, overhang: int) -> tuple[int, str]:
        """(lift, kind) for a course of an upturned eave: how far the overhang course
                rises and what it is made of, decided from the course it leaves.

                Voice contract, found by standing the reference type in two voices. The upturn
                used to lift the overhang course a whole block whatever the profile did next,
                which on a shallow eave -- a (1,2) segment, or the second course of ochre's
                (1,1)-then-(1,2) -- put a stair a block above a course that had risen half of
                one: a lip facing down into its own roof, E004 on every eave cell the moment
                ground or a canopy stood within two blocks of it, and invisible on flat ground
                because nothing stood behind the eave to compare it to. An upturn is the eave
                tip *not dropping*, never the eave standing proud of the roof: it rises as far
                as the course it leaves has risen, as a stair where that course is a full step
                up and as a slab where it is half of one, and not at all where it is level.
                
        """
        inner_h, inner_kind = prof[min(overhang, len(prof) - 1)]
        if inner_h >= h + 1:
            return 1, ("stair" if (inner_kind == "stair" or inner_h >= h + 2) else "slab")
        return 0, kind

    def _roof_block(self, x, y, z, kind, stairs, slab, full, facing):
        # Stair convention, read out of the 1.21.11 client jar rather than guessed:
        # blockstates/oak_stairs.json maps facing=east to the unrotated model, and
        # models/block/stairs.json puts that model's raised element at x 8..16 (+X). So
        # `facing` names the side the raised quarter is on, and on a roof slope it must
        # point *up*-slope toward the ridge. Pointing it down-slope leaves the tall face
        # exposed on every course and the roof reads as serrated (out/stairtest2.png,
        # left; the fixed version is out/stairtest3.png).
        if kind == "stair":
            self.place_block(x, y, z, f"{stairs}[facing={facing},half=bottom]")
        elif kind == "slab":
            self.place_block(x, y, z, f"{slab}[type=bottom]")
        else:
            self.place_block(x, y, z, full)

    def roof_cone(self, cx: int, y0: int, cz: int, r: float, mat: str,
                  pitch: tuple[int, int] = (1, 1)) -> int:
        """Conical roof for a round tower. Each course is a proper ring, so the cone
        is smooth instead of lumpy."""
        full, stairs, slab = material(mat)
        rise, run = self._clamp_pitch(*pitch)
        y = y0
        rr = r
        while rr > 0:
            self.ring(cx, y, cz, rr, f"{slab}[type=bottom]" if rr > r - 0.5 else full, 1)
            self.ring(cx, y, cz, rr, full, 1)
            y += max(1, rise // max(run, 1))
            rr -= max(1.0, run / max(rise, 1))
        self.place_block(cx, y, cz, full)
        return y

    def dormer(self, x: int, y: int, z: int, mat: str, facing: str = "south",
               width: int = 3, glass: str = "glass_pane") -> None:
        """A small gabled window box breaking a large roof plane."""
        full, stairs, slab = material(mat)
        h = 3
        w = width // 2
        if facing in ("north", "south"):
            for dx in range(-w, w + 1):
                for dy in range(h):
                    self.place_block(x + dx, y + dy, z, full if abs(dx) == w else "air")
            for dx in range(-w, w + 1):
                self.place_block(x + dx, y + 1, z, glass if abs(dx) < w else full)
            for dx in range(-w - 1, w + 2):
                self.place_block(x + dx, y + h, z, f"{stairs}[facing={facing},half=bottom]")
        else:
            for dz in range(-w, w + 1):
                for dy in range(h):
                    self.place_block(x, y + dy, z + dz, full if abs(dz) == w else "air")
            for dz in range(-w, w + 1):
                self.place_block(x, y + 1, z + dz, glass if abs(dz) < w else full)
            for dz in range(-w - 1, w + 2):
                self.place_block(x, y + h, z + dz, f"{stairs}[facing={facing},half=bottom]")

    # ----------------------------------------------------------------- steps A stair is
    # the one block in the game whose orientation is a *correctness* rule and not a
    # choice: its raised quarter belongs on the up-slope side, or the flight is
    # something you jump up and the surface reads as serrated. and three of five
    # builders independently wrote the same answer for themselves, which is the evidence
    # that it belongs here instead of in five programs. What they converged on is
    # **deferral**: queue the treads, and decide each one's facing at the end from the
    # ground on both sides *as it will finally stand*. A tread's neighbours are usually
    # built after it, so a facing decided at the moment of placement is decided against
    # a world that does not exist yet. Owned here, because two good builders would agree
    # on all of it: - the raised quarter points up-slope; - where neither orientation
    # can be justified, a stair is the wrong block, and a slab -- which cannot face
    # anywhere -- is the right one; - a rising tread the flight does not continue past
    # is a step to nowhere: it walks fine and its raised quarter overhangs open ground.
    # `circulate` already refuses to build one on a lane (see _landings); this refuses
    # it in a build. Left to the caller, because they would disagree: the material,
    # whether treads are top-half, which way a tread on level ground should face
    # (`prefer`), and whether a tread that cannot be justified becomes a slab or a full
    # block (`demote`).

    #: How far above and below a tread to look for the surface either side. The same
    #: window observe.backwards_stairs uses -- the primitive and the check that judges
    #: it must read the ground the same way or one of them is lying.
    STEP_WINDOW = 2

    def _step_queue(self) -> list:
        q = getattr(self, "_steps_pending", None)
        if q is None:
            q = self._steps_pending = []
        return q

    def _step_blocks(self, mat: str) -> tuple[str, str, str]:
        """(stairs, slab, full) for a material family or an explicit stair id."""
        name = mat.split("[")[0]
        if name in MATERIALS:
            full, stairs, slab = material(name)
            return stairs, slab, full
        for full, stairs, slab in MATERIALS.values():
            if stairs == name:
                return stairs, slab, full
        if name.endswith("_stairs"):
            base = name[: -len("_stairs")]
            return name, base + "_slab", base
        full, stairs, slab = material(name)
        return stairs, slab, full

    def step(self, x: int, y: int, z: int, mat: str, axis: str = "z", *,
             prefer: str | None = None, half: str = "bottom",
             demote: str = "slab", flight: int | None = None) -> None:
        """Queue one tread at (x, y, z). See `steps`.

                `flight` is set by `steps()` and names the run this tread belongs to. A lone
                tread has none and is decided against the ground beside it, which is the only
                thing there is to go on.
                
        """
        self._step_queue().append({"x": int(x), "y": int(y), "z": int(z), "mat": mat,
                                   "axis": "x" if axis == "x" else "z",
                                   "prefer": prefer, "half": half, "demote": demote,
                                   "flight": flight})

    def steps(self, cells, mat: str, *, axis: str | None = None,
              prefer: str | None = None, half: str = "bottom",
              demote: str = "slab") -> None:
        """Queue a flight of stairs that cannot face the wrong way.

                `cells` is [(x, y, z), ...] in order along the flight -- the order is only used
                to work out which axis each tread runs on; the facing itself is read off the
                finished ground, so a flight queued backwards comes out the same way round.
                Nothing is placed until `resolve_steps()` runs, which `Builder.flush()` does for
                you, so a tread's facing is always decided against the world as it will stand.

                axis    "x" or "z" for a tread whose neighbours in the flight do not say
                        (a single tread, or a turn). Ignored where the flight does say.
                prefer  which way a tread on *level* ground should face. Without it a tread
                        that cannot be justified is demoted rather than guessed at.
                half    "bottom" or "top", as for any stair.
                demote  "slab" or "full": what an unjustifiable tread becomes instead.

                This is for treads -- steps, flights, terrace edges. A stair used as a corbel,
                a chair or a sill bracket is not a tread and should be placed directly; it has
                no up-slope side and this would flatten it to a slab.
                
        """
        cells = [(int(x), int(y), int(z)) for (x, y, z) in cells]
        self._flight_seq = getattr(self, "_flight_seq", 0) + 1
        fid = self._flight_seq
        for i, (x, y, z) in enumerate(cells):
            ax = axis
            for j in (i + 1, i - 1):
                if 0 <= j < len(cells):
                    dx, dz = cells[j][0] - x, cells[j][2] - z
                    if bool(dx) != bool(dz):
                        ax = "x" if dx else "z"
                        break
            self.step(x, y, z, mat, ax or "z", prefer=prefer, half=half, demote=demote,
                      flight=fid)

    def _surface_probe(self, queued: dict):
        """Topmost surface in a column within STEP_WINDOW of y, queued treads included.

                Queued treads count: they are the flight itself, and a tread asked about its
                neighbour before the neighbour exists reads the bare hillside and turns round.

                Only the block *id* is available through the build API, so a slab's `type` and
                a stair's `half` cannot be seen here. That matches observe.backwards_stairs,
                which counts a full cube (or a stair, which fills its cell for collision) as
                surface and a bottom slab as not -- both read the ground the same way, which is
                what makes the check able to judge this.
                
        """
        w = self.STEP_WINDOW

        def top(x: int, z: int, y: int):
            best = None
            for ty in queued.get((x, z), ()):
                if abs(ty - y) <= w and (best is None or ty > best):
                    best = ty
            for yy in range(y + w, y - w - 1, -1):
                if best is not None and yy <= best:
                    break
                if _is_surface(self.get_block(x, yy, z)):
                    return yy
            return best
        return top

    @staticmethod
    def _flight_facings(queue: list) -> dict:
        """(flight, axis) -> the one facing every tread of that run takes.

        A run whose treads are all at one level says nothing and is left to the ground
        probe and `prefer`; so is a lone `step()`, which has no run to belong to. A turn
        is two runs, because the treads either side of it have different axes."""
        runs: dict = {}
        for t in queue:
            if t.get("flight") is None:
                continue
            runs.setdefault((t["flight"], t["axis"]), []).append(t)
        out = {}
        for key, treads in runs.items():
            if len(treads) < 2:
                continue
            axis = key[1]
            along = sorted(treads, key=lambda t: t["x"] if axis == "x" else t["z"])
            rise = along[-1]["y"] - along[0]["y"]
            if rise == 0:
                continue
            lo, hi = ("west", "east") if axis == "x" else ("north", "south")
            out[key] = hi if rise > 0 else lo
        return out

    def _decide_steps(self) -> dict:
        """(x, y, z) -> block state for every queued tread. Decides, places nothing."""
        q = self._step_queue()
        if not q:
            return {}
        queued: dict = {}
        for t in q:
            queued.setdefault((t["x"], t["z"]), []).append(t["y"])
        top = self._surface_probe(queued)
        run_facing = self._flight_facings(q)

        out = {}
        for t in q:
            x, y, z = t["x"], t["y"], t["z"]
            stairs, slab, full = self._step_blocks(t["mat"])
            if t["axis"] == "x":
                (lo, dlo), (hi, dhi) = ("west", (-1, 0)), ("east", (1, 0))
            else:
                (lo, dlo), (hi, dhi) = ("north", (0, -1)), ("south", (0, 1))
            a = top(x + dlo[0], z + dlo[1], y)
            b = top(x + dhi[0], z + dhi[1], y)
            va = -1 << 20 if a is None else a
            vb = -1 << 20 if b is None else b
            # A flight decides once, for all of its treads. The run's own cells say
            # which way is up without asking the world at all, and that is strictly
            # better information than the two columns beside one tread.
            facing = run_facing.get((t["flight"], t["axis"]))
            if facing not in (lo, hi):
                if va != vb:
                    facing = lo if va > vb else hi
                elif t["prefer"] in (lo, hi):
                    facing = t["prefer"]
                else:
                    facing = None
            ahead = a if facing == lo else (b if facing == hi else None)
            # `ahead < y` is the step to nowhere: the raised quarter has nothing at its
            # own level to abut, so it overhangs whatever is below.
            if facing is None or ahead is None or ahead < y:
                out[(x, y, z)] = (full if t["demote"] == "full"
                                  else f"{slab}[type={'top' if t['half'] == 'top' else 'bottom'}]")
            else:
                out[(x, y, z)] = f"{stairs}[facing={facing},half={t['half']}]"
        return out

    def resolve_steps(self) -> dict:
        """Place every queued tread, facing decided from the finished ground.

                Idempotent, and called for you by `Builder.flush()` -- a queue a program forgot
                to resolve would be a flight of stairs that never got built, which is a worse
                failure than the one this exists to prevent.
                
        """
        decided = self._decide_steps()
        placed = demoted = 0
        for (x, y, z), b in decided.items():
            self.place_block(x, y, z, b)
            if "_stairs[" in b:
                placed += 1
            else:
                demoted += 1
        self._step_queue().clear()
        return {"treads": placed + demoted, "stairs": placed, "demoted": demoted}

    # --------------------------------------------------------------- terrain
    def clear_trees(self, x0: int, z0: int, x1: int, z1: int, margin: int = 2,
                    columns=None) -> int:
        """Remove whole trees over an area — trunk *and* canopy.

                Clearing a box leaves the canopy of any tree rooted just outside it hanging in
                mid-air. "Incomplete removal of trees, resulting in floating leaves" is the
                single most-cited giveaway in the GDMC judging literature, so this walks
                outward from each trunk instead of clipping to the box.

                `columns` is the list of columns to look for trunks in, in place of the
                rectangle, for a caller whose ground is a line rather than a box: the swept run
                of a wall is a few thousand columns and its bounding box is the city inside it.
                The margin is applied round each column.
                
        """
        removed = 0
        trunks = []
        if columns is not None:
            look = {(x + dx, z + dz) for (x, z) in columns
                    for dx in range(-margin, margin + 1)
                    for dz in range(-margin, margin + 1)}
        else:
            look = [(x, z) for x in range(x0 - margin, x1 + margin + 1)
                    for z in range(z0 - margin, z1 + margin + 1)]
        for (x, z) in sorted(look):
            g = self.get_height(x, z)
            for y in range(g - 1, g + 3):
                b = self.get_block(x, y, z)
                if b.endswith("_log") or b.endswith("_stem") or b.endswith("_wood"):
                    trunks.append((x, y, z))
                    break
        seen = set()
        stack = list(trunks)
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            b = self.get_block(*p)
            # A mangrove propagule hangs from the leaves rather than being one, so the
            # flood stopped at it and left a single block at (1634,70,414) with air
            # above and below it. E010, "1 blocks are held up by nothing".
            if not any(v in b for v in ("_log", "_wood", "_leaves", "_stem", "vine",
                                        "shroomlight", "_hyphae", "_propagule")):
                continue
            self.place_block(p[0], p[1], p[2], "air")
            removed += 1
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                      (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                      (0, 1, 1), (0, 1, -1), (1, 1, 0), (-1, 1, 0)):
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if q not in seen and len(seen) < 60000:
                    stack.append(q)
        return removed

    def clear_ground_cover(self, x0: int, z0: int, x1: int, z1: int,
                           columns=None) -> None:
        """Strip grass, flowers, ferns and snow off the surface of an area.

                **And write down where it took the ground rather than the cover.** `VEGETATION`
                contains the string `grass` and so does `grass_block`, which is A6's finding:
                this call has been lifting the turf off every pad it has ever prepared. A6
                answered it by putting a skin back on afterwards, and that fixed what the
                ground *looks* like and not what it is: a column whose grass block is taken and
                whose subsoil is then re-skinned is a column **one block lower than it was
                found**, and a one-block rise is a jump in the walk model. On the edge of a
                deck that is the difference between walking in at the door and not: four halls
                on decks were sealed by exactly that step and by the footing course a type laid
                over ground it could not walk to. `_dress_worked` reads this register and puts
                the level back; nothing else does, so what a stored program places here is byte
                for byte what it placed before.
                
        """
        from .observe import is_growing
        # Only where the **library** is preparing ground for a part. A stored program
        # that clears its own cover and then calls `approach()` must place exactly what
        # it placed in September -- `dress_ground(worked=True)` is off by default for
        # the same reason, and the first version of this register was on for everybody
        # and moved a product-claim draft's doorstep.
        seen = getattr(self, "stripped", None) \
            if getattr(self, "_library_ground", False) else None
        cells = columns if columns is not None else [
            (x, z) for x in range(min(x0, x1), max(x0, x1) + 1)
            for z in range(min(z0, z1), max(z0, z1) + 1)]
        for (x, z) in cells:
            g = self.get_height(x, z)
            for y in range(g, g + 3):
                b = self.get_block(x, y, z)
                if b != "air" and any(v in b for v in VEGETATION) and "_log" not in b:
                    if seen is not None and not is_growing(b):
                        seen.setdefault((int(x), int(z)), (int(y), b))
                    self.place_block(x, y, z, "air")

    #: How far below the surface a `grade()` probe will look for the bed. Deeper than
    #: any lake this project has built on and shallow enough that a column of air over a
    #: cave does not walk the probe to bedrock.
    MAX_SOUNDING = 24

    def grade(self, x: int, z: int) -> int:
        """The y of the **ground** under a column: the bed, where the column is water.

        A dry run does not have the defect (the offline heightmap is computed from what
        a body collides with, and a body does not collide with water), which is exactly
        why nothing caught it until a person swam under a building. So this is a
        sounding rather than a lookup: start at the surface and go down while what is
        there is water, or air, or anything else you cannot stand on. Where the surface
        is already something you can stand on -- every dry column, and every column in
        every dry run this project has ever done -- it is `get_height` and costs one
        block read.

        A column the library has **sited** answers with the pad it laid, because that is
        what the ground under it now is: `plinth(to_grade)` inside a `building()` on a
        prepared pad must fill nothing, and over water it would otherwise pour the lake
        solid and take the piles out from under the deck. Only `site()` writes that
        record and no stored program calls `site()`, so nothing in the record moves."""
        s = getattr(self, "_sited", None)
        if s:
            g = s.get((int(x), int(z)))
            if g is not None:
                return g
        top = int(self.get_height(x, z))
        y = top
        for _ in range(self.MAX_SOUNDING):
            if _occupies_cell(self._ground_block(x, y, z)):
                return y
            y -= 1
        return top

    def _ground_block(self, x: int, y: int, z: int) -> str:
        """What the **world** has at a position, before this program decided anything.

                `grade()` has to read the world and not the pending set. Every stored program
                in the record clears its ground before it lays a plinth, and clearing writes
                `air` at the old surface -- so a sounding that could see its own work would walk
                straight down through the ground it was about to stand on, and thirteen wave
                programs would stop replaying to the counts their dry runs recorded.
                
        """
        try:
            b = self.world_site.editor.worldSlice.getBlockGlobal((int(x), int(y), int(z)))
            return str(b.id).split(":")[-1]
        except Exception:
            return self.get_block(x, y, z)

    def wet(self, x: int, z: int) -> int | None:
        """The top of the water standing in a column, or None where it is dry."""
        g = self.grade(x, z)
        y = g + 1
        top = None
        for _ in range(self.MAX_SOUNDING):
            b = self._ground_block(x, y, z).split("[")[0]
            if b not in ("water", "flowing_water", "bubble_column"):
                break
            top = y
            y += 1
        return top

    def foundation_to_grade(self, x0: int, z0: int, x1: int, z1: int, y: int,
                            block: str, skirt: int = 0) -> None:
        """Fill each column from the real ground up to y-1."""
        for x in range(min(x0, x1) - skirt, max(x0, x1) + skirt + 1):
            for z in range(min(z0, z1) - skirt, max(z0, z1) + skirt + 1):
                g = self.grade(x, z)
                for yy in range(min(g, y - 1), y):
                    self.place_block(x, yy, z, block)

    def terrace(self, x0: int, z0: int, x1: int, z1: int, y: int, block: str,
                feather: int = 4) -> None:
        """Level an area to y, then blend the edge outward over `feather` blocks.

                A hard rectangular cut where a build meets a hillside is a named tell. The
                feathered ring eases the platform back into the natural slope.
                
        """
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for z in range(min(z0, z1), max(z0, z1) + 1):
                g = self.get_height(x, z)
                if g >= y:
                    for yy in range(y, g + 6):
                        self.place_block(x, yy, z, "air")
                for yy in range(min(g, y), y):
                    self.place_block(x, yy, z, block)
        for ring_i in range(1, feather + 1):
            t = ring_i / (feather + 1.0)
            for x in range(min(x0, x1) - ring_i, max(x0, x1) + ring_i + 1):
                for z in range(min(z0, z1) - ring_i, max(z0, z1) + ring_i + 1):
                    on_edge = (x in (min(x0, x1) - ring_i, max(x0, x1) + ring_i) or
                               z in (min(z0, z1) - ring_i, max(z0, z1) + ring_i))
                    if not on_edge:
                        continue
                    g = self.get_height(x, z)
                    target = int(round(y * (1 - t) + g * t))
                    if g > target:
                        for yy in range(target + 1, g + 4):
                            self.place_block(x, yy, z, "air")
                    else:
                        for yy in range(g, target + 1):
                            self.place_block(x, yy, z, block)

    #: What a worked column may not be left as: the subsoil a cut or a clearance
    #: exposed. `Builder.BARE` is this list and `site()` and `approach()` dress every
    #: column they change against it.
    BARE_GROUND = ("dirt", "coarse_dirt", "rooted_dirt", "farmland")

    #: Ground cover that is debris once the thing it grew under is gone.
    _LITTER = ("leaf_litter", "short_grass", "tall_grass", "fern", "large_fern",
               "dead_bush", "vine", "glow_lichen", "hanging_roots", "moss_carpet",
               "azalea", "flowering_azalea", "sugar_cane", "bamboo", "cocoa",
               "twisting_vines", "weeping_vines", "cave_vines")

    def _worked_top(self, x: int, z: int) -> int:
        """The y of the top of this column **as this program has left it**."""
        y = int(self.get_height(x, z)) + 2
        for _ in range(self.MAX_SOUNDING):
            b = self.get_block(x, y, z).split("[")[0]
            # The ground, not what is lying on it: leaf litter, a fern and a flower are
            # all things a body walks through, and the skin goes back under them.
            if b not in ("air", "cave_air", "void_air") and _occupies_cell(b):
                return y
            y -= 1
        return int(self.get_height(x, z))

    #: What the undisturbed surface of a site can be made of. somebody's cobblestone, a
    #: lane, a lake -- is not what the skin should be put back as.
    NATURAL_COVER = ("grass_block", "sand", "red_sand", "podzol", "mycelium",
                     "coarse_dirt", "gravel", "snow_block", "moss_block", "terracotta",
                     "red_sandstone", "sandstone", "stone")

    def _cover_around(self, x0: int, z0: int, x1: int, z1: int,
                      worked: bool = False) -> str | None:
        """The commonest natural surface on a ring round this patch, or None."""
        radii = (3, 7, 14) if worked else (3,)
        for r in radii:
            ring: dict = {}
            edge = [(x, z) for x in range(x0 - r, x1 + r + 1) for z in (z0 - r, z1 + r)]
            edge += [(x, z) for z in range(z0 - r, z1 + r + 1) for x in (x0 - r, x1 + r)]
            for (x, z) in edge:
                y = self._worked_top(x, z) if worked else self.get_height(x, z)
                b = self.get_block(x, y, z)
                ring[b] = ring.get(b, 0) + 1
            natural = {k: v for k, v in ring.items()
                       if k in self.NATURAL_COVER or k.endswith("_terracotta")}
            if natural:
                return max(natural, key=lambda k: natural[k])
        return None

    def dress_ground(self, x0: int, z0: int, x1: int, z1: int, cover: str | None = None,
                     sweep: bool = True, columns=None, worked: bool = False) -> int:
        """Finish a piece of ground you have worked: sweep the debris, put the skin back.

                `cover` defaults to whatever the *undisturbed* ground immediately around the area
                is made of, so this works on grass, sand, podzol or terracotta without being told.
                Pass it explicitly for a yard you mean to be beaten earth.
        """
        x0, x1 = min(x0, x1), max(x0, x1)
        z0, z1 = min(z0, z1), max(z0, z1)
        if cover is None:
            cover = self._cover_around(x0, z0, x1, z1, worked)
        touched = 0
        cells = columns if columns is not None else [
            (x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        # By record, like `fitting()`: a column this call put the skin back on is ground
        # work and not somebody building outside their plot, and the only thing that
        # knows which is which is the call that did it. `scripts/test_ground.py`'s fifth
        # assertion reads it.
        seen = getattr(self, "dressed", None)
        for (x, z) in cells:
            g = self._worked_top(x, z) if worked else self.get_height(x, z)
            if sweep:
                for y in range(g + 1, g + 3):
                    b = self.get_block(x, y, z)
                    if b in self._LITTER or b.endswith("_litter"):
                        self.place_block(x, y, z, "air")
                        touched += 1
                        if seen is not None:
                            seen.add((x, z))
            if cover and self.get_block(x, g, z) in self.BARE_GROUND:
                self.place_block(x, g, z, cover)
                touched += 1
                if seen is not None:
                    seen.add((x, z))
        return touched

    def flattest_rect(self, x0: int, z0: int, x1: int, z1: int, w: int, d: int,
                      step: int = 2) -> tuple[int, int, int, int]:
        """Find the flattest w x d footprint in a region.

        Returns (x, z, floor_y, spread). Use it to site a build instead of guessing —
        and check the spread before committing to a footprint that will need a plinth."""
        best = None
        for x in range(min(x0, x1), max(x0, x1) - w + 2, step):
            for z in range(min(z0, z1), max(z0, z1) - d + 2, step):
                hs = [self.get_height(x + dx, z + dz)
                      for dx in range(0, w, max(1, w // 4))
                      for dz in range(0, d, max(1, d // 4))]
                spread = max(hs) - min(hs)
                med = sorted(hs)[len(hs) // 2]
                if best is None or spread < best[3]:
                    best = (x, z, med + 1, spread)
        return best

    # ---------------------------------------------------------------- detail
    def wall(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int,
             block: str, post: str | None = None, spacing: int = 4,
             base: str | None = None, base_height: int = 1,
             band: str | None = None) -> None:
        """A wall with depth: posts at intervals, a base course, a top band.

                A flat run of one block type is the most-cited beginner tell in the craft
                literature. This gives the wall articulation for free.
                
        """
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    b = block
                    if base and y - min(y0, y1) < base_height:
                        b = base
                    elif band and y == max(y0, y1):
                        b = band
                    self.place_block(x, y, z, b)
        if post:
            horiz = "x" if abs(x1 - x0) >= abs(z1 - z0) else "z"
            lo, hi = (min(x0, x1), max(x0, x1)) if horiz == "x" else (min(z0, z1), max(z0, z1))
            for c in list(range(lo, hi + 1, spacing)) + [hi]:
                for y in range(min(y0, y1), max(y0, y1) + 1):
                    if horiz == "x":
                        for z in range(min(z0, z1), max(z0, z1) + 1):
                            self.place_block(c, y, z, post)
                    else:
                        for x in range(min(x0, x1), max(x0, x1) + 1):
                            self.place_block(x, y, c, post)

    # Form, where two good builders would disagree and the disagreement is the point.
    # Their *shapes* are here; every number in them is a parameter with a default, and
    # none of them decides how much to clear, what to build from or how big to make it.

    def plinth(self, x0: int, z0: int, x1: int, z1: int, y: int, block: str, *,
               courses: int = 1, course: int = 1, batter: int = 0, overhang: int = 0,
               cap: str | None = None, to_grade: bool = True,
               max_depth: int = 32) -> dict:
        """The base a building stands on: battered courses carried down to real ground.

        Returns {"columns", "blocks", "lowest"}."""
        x0, x1 = min(x0, x1), max(x0, x1)
        z0, z1 = min(z0, z1), max(z0, z1)
        # Read the ground *before* laying a single course. Measuring it afterwards asks
        # a column how high it is once you have already built on it, and the answer is
        # the plinth: it then fills nothing and the whole base hangs over the slope.
        grow_max = overhang + max(0, courses - 1) * batter
        ground = {(x, z): self.grade(x, z)
                  for x in range(x0 - grow_max, x1 + grow_max + 1)
                  for z in range(z0 - grow_max, z1 + grow_max + 1)}
        n = blocks = 0
        lowest = y
        for k in range(max(1, courses)):
            grow = overhang + k * batter
            top = y - k * course
            bot = top - course + 1
            for x in range(x0 - grow, x1 + grow + 1):
                for z in range(z0 - grow, z1 + grow + 1):
                    for yy in range(bot, top + 1):
                        self.place_block(x, yy, z, cap if (cap and k == 0) else block)
                        blocks += 1
            lowest = min(lowest, bot)
        if to_grade:
            for x in range(x0 - grow_max, x1 + grow_max + 1):
                for z in range(z0 - grow_max, z1 + grow_max + 1):
                    n += 1
                    for yy in range(max(ground[(x, z)], lowest - max_depth), lowest):
                        self.place_block(x, yy, z, block)
                        blocks += 1
        return {"columns": n, "blocks": blocks, "lowest": lowest}

    @staticmethod
    def _glazing(block, width: int, reveal: int) -> str:
        """What actually fills an opening of this shape.

        A caller who wants a lattice says so by passing the pane itself, and a caller who
        wants shutters, bars or ice passes those. Only the default is decided here."""
        if block not in (None, "auto"):
            return block
        return "glass_pane" if (width <= 1 and reveal == 0) else "glass"

    def openings(self, x0: int, y: int, z0: int, x1: int, z1: int, *,
                 at: list | None = None, spacing: int = 4, width: int = 1,
                 sill: int = 2, head: int = 3, block: str | None = None,
                 sill_block: str | None = None, lintel: str | None = None,
                 reveal: int = 0, inward: str | None = None,
                 margin: int = 1) -> list:
        """Openings on a rhythm along one straight run of wall.

                `y` is the floor block level and every height is measured from it: an opening
                occupies y+`sill` up to y+`head` inclusive, so sill and head are the two numbers
                a builder actually argues about. `reveal` sets the glazing that many blocks back
                from the wall face, toward `inward`, and clears what is in front of it -- the
                "recess a window a block" rule from the craft guide, which is depth and not
                decoration.

                `at` places openings at exact positions along the run; without it they fall on
                `spacing`, inset by `margin` from each end so an opening never lands on a
                corner. Returns the (x, y, z) of each opening's lowest, first cell.

                `block` defaults to whatever fills an opening of this shape -- a pane for a
                one-wide slit, glass for anything wider or recessed. Pass one to override.
                
        """
        if (x0 != x1) == (z0 != z1):
            raise ValueError("openings() takes one straight run of wall")
        block = self._glazing(block, width, reveal)
        horiz = x0 != x1
        lo, hi = (min(x0, x1), max(x0, x1)) if horiz else (min(z0, z1), max(z0, z1))
        if at is None:
            at = list(range(lo + margin, hi - margin - width + 2, max(1, spacing)))
        dx, dz = DIRS[inward] if inward else (0, 0)
        out = []
        for c in at:
            for i in range(width):
                cx = (c + i) if horiz else x0
                cz = z0 if horiz else (c + i)
                if not (lo <= (cx if horiz else cz) <= hi):
                    continue
                for j in range(sill, head + 1):
                    for d in range(reveal):
                        self.place_block(cx + dx * d, y + j, cz + dz * d, "air")
                    self.place_block(cx + dx * reveal, y + j, cz + dz * reveal, block)
                if sill_block:
                    self.place_block(cx, y + sill - 1, cz, sill_block)
                if lintel:
                    self.place_block(cx, y + head + 1, cz, lintel)
            out.append((x0 if not horiz else c, y + sill, z0 if horiz else c))
        return out

    def storey_steps(self, x0: int, z0: int, x1: int, z1: int, *, axis: str = "x",
                     bays: int = 3, storey: int = 3, anchor: int | None = None,
                     max_drop: int | None = None) -> list:
        """How a footprint steps down a slope, in whole storeys.

                Splits the footprint into `bays` along `axis` and gives each one a floor level:
                the bay's own median ground, snapped to a whole number of `storey`s below the
                anchor, and clamped so neighbouring bays are never more than one storey apart --
                which is what stops a row of houses reading as a staircase. With storey=1 it is
                a terraced row that follows the ground; with storey=4 it is a building that
                drops a floor at a time down a hillside.

                Returns [{"x0", "z0", "x1", "z1", "floor_y", "drop"}], highest bay first in
                floor level. Where the bays go and what happens on them is the caller's.
                
        """
        x0, x1 = min(x0, x1), max(x0, x1)
        z0, z1 = min(z0, z1), max(z0, z1)
        bays = max(1, int(bays))
        storey = max(1, int(storey))
        along = (x0, x1) if axis == "x" else (z0, z1)
        span = along[1] - along[0] + 1
        edges = [along[0] + (span * i) // bays for i in range(bays + 1)]

        rects, meds = [], []
        for i in range(bays):
            a, b = edges[i], edges[i + 1] - 1
            r = (a, z0, b, z1) if axis == "x" else (x0, a, x1, b)
            hs = sorted(self.get_height(x, z)
                        for x in range(r[0], r[2] + 1)
                        for z in range(r[1], r[3] + 1))
            rects.append(r)
            meds.append(hs[len(hs) // 2] if hs else 0)

        base = max(meds) if anchor is None else anchor
        drops = [max(0, int(round((base - m) / storey))) for m in meds]
        if max_drop is not None:
            drops = [min(d, max_drop) for d in drops]
        for _ in range(bays):                    # settle to at most one storey a bay
            for i in range(1, bays):
                drops[i] = min(drops[i], drops[i - 1] + 1)
            for i in range(bays - 2, -1, -1):
                drops[i] = min(drops[i], drops[i + 1] + 1)
        out = []
        for r, d in zip(rects, drops):
            out.append({"x0": r[0], "z0": r[1], "x1": r[2], "z1": r[3],
                        "floor_y": base - d * storey, "drop": d})
        return out

    #: What each fitting is made of. Ids only, so `registry.check_all` can validate the
    #: whole vocabulary offline and a builder can never reach for a block that does not
    #: exist in 1.21.11.
    FITTING_BLOCKS = {
        "hearth":    ("campfire", "iron_bars"),
        "forge":     ("furnace", "blast_furnace"),
        "anvil":     ("anvil", "smithing_table"),
        "workbench": ("crafting_table", "barrel"),
        "store":     ("barrel", "chest"),
        "bed":       ("red_bed",),
        "table":     ("oak_fence", "oak_pressure_plate"),
        "trough":    ("water_cauldron", "cauldron"),
        "fodder":    ("hay_block",),
        "light":     ("lantern", "candle", "wall_torch", "glowstone"),
        # Every room came out store + table + bed + light because that was the whole
        # vocabulary. These are the words that were missing, and no more than six -- a
        # vocabulary you can hold in your head is one a builder will actually use.
        "bookshelf": ("bookshelf",),
        "bench":     ("oak_stairs",),
        "shelf":     ("oak_slab",),
        "rug":       ("white_carpet",),
        "oven":      ("furnace", "cobblestone_slab"),
        "well":      ("water", "cobblestone_wall"),
    }

    #: Which light a room of each purpose gets, keyed by substrings of the planner's own
    #: `kind` -- the same matching `styles.register_for` uses. Lantern is the default
    #: and only the default.
    LIGHTS_BY_ROOM = (
        (("temple", "shrine"), "glowstone"),
        (("hall", "inn", "market", "gate"), "hanging_lantern"),
        (("house", "dwelling", "croft", "row", "guest"), "candle"),
        (("granary", "barn", "byre", "store", "stable", "shed"), "torch"),
        (("smith", "forge", "kiln", "mill", "watch", "tower"), "lantern"),
    )

    #: Cells a fitting lays that must be air with floor under them, versus cells that
    #: need only be air, versus cells it is entitled to replace outright. See `fitting`.
    _NEED_FLOOR, _NEED_CLEAR, _NEED_NOTHING = "floor", "clear", None

    def fitting(self, kind: str, x: int, y: int, z: int, facing: str = "north", *,
                mat: str = "cobblestone", extent: int = 1, room: str | None = None,
                block: str | None = None, flue_to: int | None = None,
                dry: bool = False) -> dict:
        """One piece of interior equipment, placed the way it is actually built -- or a
        refusal.

        `dry=True` answers the same question and places nothing: `cells` is every cell
        this call *would* fill, so a caller offering the piece to its own flood fill
        first can offer what the library actually lays rather than a guess at it. That
        guess is a defect class of its own -- a well is a 3x3 curb round an open shaft,
        built up rather than dug down, and a type that tested three cells for it was
        free to wall a courtyard in two with the one fitting every room opened onto.

        `(x, y, z)` is the cell the fitting stands **in** -- the floor block is at y-1.
        `facing` is the way it is used from, which is the thing a program gets wrong by
        hand: a furnace facing a wall, a bed whose foot is inside the masonry, an anvil
        with no room to stand at it.

        This is a **vocabulary, not a decorator**. It knows what a forge is made of and
        which way a barrel opens; it decides nothing about which room anything goes in,
        and it will happily put a bed in a byre. Layout is the builder's.

        kinds: hearth (with `flue_to` to carry the flue to a ridge height), forge, anvil,
        workbench, store (`extent` runs it along the wall), bed, table, trough, fodder,
        light, bookshelf, bench, shelf, rug, oven, well."""
        full, stairs, slab = material(mat)
        wood = mat if _has_block(f"{mat}_fence") else "oak"
        back = {"north": (0, 1), "south": (0, -1), "east": (-1, 0), "west": (1, 0)}[facing]
        plan: list = []

        def put(px, py, pz, b, need=self._NEED_FLOOR):
            plan.append((int(px), int(py), int(pz), b, need))

        if kind == "hearth":
            put(x, y - 1, z, full, self._NEED_NOTHING)   # a fire stands on stone
            put(x, y, z, block or "campfire[lit=true,signal_fire=false]")
            # The surround is masonry this fitting *lays*, not a cell it needs empty, so
            # a hearth built against a wall is a hearth built the way the craft guide
            # asks for and not a collision.
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (dx, dz) != (-back[0], -back[1]):     # open on the side you use it
                    put(x + dx, y, z + dz, full, self._NEED_NOTHING)
            put(x, y + 1, z, "air", self._NEED_NOTHING)
            if flue_to is not None:                      # a flue that reaches the sky
                # No requirement on a flue's cells: a flue's whole job is to go through
                # the floor above and out of the roof, and a check that refused it there
                # would be refusing the thing being built.
                for fy in range(y + 1, int(flue_to) + 2):
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        put(x + dx, fy, z + dz, full, self._NEED_NOTHING)
                    put(x, fy, z, "air", self._NEED_NOTHING)
        elif kind == "forge":
            put(x, y, z, block or f"furnace[facing={facing},lit=true]")
        elif kind == "anvil":
            put(x, y - 1, z, full, self._NEED_NOTHING)
            put(x, y, z, block or f"anvil[facing={facing}]")
        elif kind == "workbench":
            put(x, y, z, block or "crafting_table")
        elif kind == "store":
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i,
                    block or f"barrel[facing={facing},open=false]")
        elif kind == "bed":
            # A bed's `facing` is the way its head points, so the foot block lies
            # *behind* the head. found by a person walking the town.
            fx, fz = x + back[0], z + back[1]
            put(x, y, z, f"{block or 'red_bed'}[facing={facing},part=head]")
            put(fx, y, fz, f"{block or 'red_bed'}[facing={facing},part=foot]")
        elif kind == "table":
            # Built out of `mat`, not out of oak. The first version of this hardcoded
            # oak fence and oak pressure plate, which would have put oak furniture in a
            # blackstone-and-birch town -- a fitting is equipment, and equipment is made
            # of what the place is made of.
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i, block or f"{wood}_fence")
                put(x + step[0] * i, y + 1, z + step[1] * i, f"{wood}_pressure_plate",
                    self._NEED_CLEAR)
        elif kind == "trough":
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i,
                    block or "water_cauldron[level=3]")
        elif kind == "fodder":
            for i in range(max(1, extent)):
                put(x, y + i, z, block or "hay_block[axis=y]",
                    self._NEED_FLOOR if i == 0 else self._NEED_CLEAR)
        elif kind == "light":
            self._fitting_light(put, x, y, z, facing, back, room, block, mat)
        elif kind == "bookshelf":
            # Against a wall and shoulder height, which is what a bookshelf is: one
            # course on the floor and one above it, run along `extent`.
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i, block or "bookshelf")
                put(x + step[0] * i, y + 1, z + step[1] * i, block or "bookshelf",
                    self._NEED_CLEAR)
        elif kind == "bench":
            # A stair used as a seat is not a tread: it is placed outright rather than
            # queued, exactly as `steps()`' own documentation says it must be.
            _full, seat_stairs, _slb = material(wood)
            face_back = {"north": "south", "south": "north",
                         "east": "west", "west": "east"}[facing]
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i,
                    block or f"{seat_stairs}[facing={face_back},half=bottom]")
        elif kind == "shelf":
            # A board on a wall. Air is all it needs under it -- the point of a shelf is
            # that it is not standing on the floor.
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            for i in range(max(1, extent)):
                put(x + step[0] * i, y, z + step[1] * i,
                    block or f"{material(wood)[2]}[type=top]", self._NEED_CLEAR)
        elif kind == "rug":
            step = (1, 0) if facing in ("north", "south") else (0, 1)
            perp = (0, 1) if step == (1, 0) else (1, 0)
            for i in range(max(1, extent)):
                for j in range(max(1, extent)):
                    put(x + step[0] * i + perp[0] * j, y, z + step[1] * i + perp[1] * j,
                        block or "white_carpet")
        elif kind == "oven":
            # A furnace with a masonry surround, which is the difference between a
            # kitchen and a furnace left on a floor.
            put(x, y, z, block or f"furnace[facing={facing},lit=true]")
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (dx, dz) != (-back[0], -back[1]):
                    put(x + dx, y, z + dz, full, self._NEED_NOTHING)
            put(x, y + 1, z, f"{slab}[type=bottom]", self._NEED_NOTHING)
        elif kind == "well":
            # Built up rather than dug down, which is what the register says a well is.
            from .circulate import WALLS
            curb = block or WALLS.get(mat) or full
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dz) == (0, 0):
                        continue
                    put(x + dx, y, z + dz, curb, self._NEED_NOTHING)
            put(x, y, z, "air")                    # the shaft: floor, and open above it
            put(x, y - 1, z, "water", self._NEED_NOTHING)
        else:
            raise ValueError(f"unknown fitting {kind!r}; "
                             f"have {sorted(self.FITTING_BLOCKS)}")

        bad = None if getattr(self, "allow_collide", False) \
            else self._fitting_conflict(plan)
        if dry:
            return {"ok": bad is None, "cells": [(p[0], p[1], p[2]) for p in plan],
                    "cell": None if bad is None else list(bad[0]),
                    "reason": (f"a {kind} would fill {len(plan)} cell(s)"
                               if bad is None else
                               f"a {kind} needs the cell at {bad[0]} and {bad[1]}")}
        if bad is not None:
            cell, what = bad
            return {"ok": False, "cells": [], "cell": list(cell),
                    "reason": f"a {kind} needs the cell at "
                              f"({cell[0]},{cell[1]},{cell[2]}) and {what} -- nothing "
                              f"has been placed; move it or clear the cell first"}
        for (px, py, pz, b, _need) in plan:
            self.place_block(px, py, pz, b)
        cells = [(p[0], p[1], p[2]) for p in plan]
        # ...and write down that these cells are furniture. The one thing that can tell
        # a barrel from a dais is the call that placed it, and every walkability number
        # in the project reads this record through `observe.floor_stances`. A hearth's
        # masonry surround is in here and the identical cobblestone a type lays as a
        # counter is not, which is the whole difference.
        self._record_fitting(cells)
        return {"ok": True, "cells": cells,
                "reason": f"{kind} placed in {len(plan)} cell(s)"}

    def _record_fitting(self, cells) -> None:
        """Remember cells `fitting()` laid, as furniture. See `observe.floor_stances`."""
        try:
            reg = self.fitting_cells
        except AttributeError:
            reg = self.fitting_cells = set()
        reg.update((int(x), int(y), int(z)) for (x, y, z) in cells)

    def _fitting_light(self, put, x, y, z, facing, back, room, block, mat) -> None:
        """The light a room of this purpose gets. See `LIGHTS_BY_ROOM`."""
        if block:
            put(x, y, z, block)
            return
        k = (room or "").lower()
        choice = "lantern"
        for keys, name in self.LIGHTS_BY_ROOM:
            if any(w in k for w in keys):
                choice = name
                break
        # A wall torch with no wall behind it and a hanging lantern with no ceiling over
        # it are both blocks the game will drop on the next update. Where the room does
        # not have what the light needs, it falls back to the one light that only needs
        # a floor -- rather than refusing, because a room with no light in it is worse
        # than a room lit the ordinary way.
        if choice == "torch" and not _is_surface(
                self.get_block(x + back[0], y, z + back[1])):
            choice = "lantern"
        if choice == "hanging_lantern" and not _is_surface(
                self.get_block(x, y + 1, z)):
            choice = "lantern"
        if choice == "candle":
            put(x, y, z, "candle[candles=3,lit=true,waterlogged=false]")
        elif choice == "torch":
            put(x, y, z, f"wall_torch[facing={facing}]", self._NEED_CLEAR)
        elif choice == "hanging_lantern":
            put(x, y, z, "lantern[hanging=true,waterlogged=false]", self._NEED_CLEAR)
        elif choice == "glowstone":
            put(x, y, z, "glowstone")
            put(x, y + 1, z, f"{material(mat)[2]}[type=bottom]", self._NEED_CLEAR)
        else:
            put(x, y, z, "lantern[hanging=false,waterlogged=false]")

    def _fitting_conflict(self, plan: list):
        """The first cell this fitting cannot have, and why, or None.

                A cell the plan itself fills counts as floor for the cell above it -- an anvil
                brings its own block and a hearth its own stone, and refusing those would refuse
                the fitting for the foundation it is carrying.
                
        """
        laid = {(p[0], p[1], p[2]): p[3] for p in plan}

        def air(b) -> bool:
            return b.split("[")[0].split(":")[-1] in ("air", "cave_air", "void_air")

        # A flight's cells are the flight's, whatever the fitting needs of them: a
        # barrel on the foot of a stair, or over the well it climbs into, seals the
        # storey above. See `Builder.flight_cells`.
        held = getattr(self, "flight_cells", None) or ()
        for (px, py, pz, _b, _need) in plan:
            if (px, py, pz) in held:
                return ((px, py, pz), "it is a flight's -- a tread, its landing, the "
                                      "foot of it or the headroom over them")
        for (px, py, pz, _b, need) in plan:
            if need is self._NEED_NOTHING:
                continue
            here = self.get_block(px, py, pz)
            if not air(here):
                return (px, py, pz), f"{here} is standing in it"
            if need == self._NEED_CLEAR:
                continue
            below = laid.get((px, py - 1, pz))
            if below is not None:
                if air(below):
                    return (px, py - 1, pz), "there is nothing under it to stand on"
                continue
            if not _is_surface(self.get_block(px, py - 1, pz)):
                return ((px, py - 1, pz),
                        "there is no floor under it -- it would hang in the air")
        return None

    def doorway(self, x: int, y: int, z: int, facing: str, mat: str, *,
                leaf: str = "oak_door", jamb: str = "build",
                lintel: str | bool = True, threshold: str | None = None,
                hinge: str = "left", front: str = "refuse") -> dict:
        """A door, with the wall it needs around it.

        `(x, y, z)` is the **lower leaf** — the cell a person stands in to walk through,
        which is `stand_y` from `floor_from_threshold()`, not the floor block.

        What this owns, because two builders would agree on all of it: both halves of the
        leaf in the right order, a **jamb on each side**, and a lintel over it. What it
        does not own: where the door goes, which way it faces, what it is made of.

            jamb="build"   fill a missing side with the wall material  (default)
            jamb="refuse"  place nothing and say which side is open, so the caller moves
                           the door rather than growing a wall it did not design

        Returns {"ok", "jambs", "open_sides", "front", "reason"}."""
        # `solid`, not `material`: a jamb and a lintel are cubes, so a palette may name
        # the block it wants here directly and does not owe this call a whole family.
        full = solid(mat)
        fdx, fdz = DIRS[facing]
        fx, fz = x - fdx, z - fdz                    # the cell you walk in *from*
        here = self.get_block(fx, y, fz)
        under = self.get_block(fx, y - 1, fz)
        blocked = _occupies_cell(here)
        no_floor = not _is_surface(under)
        if front == "refuse" and (blocked or no_floor):
            return {"ok": False, "jambs": 0, "open_sides": [], "front": [fx, y, fz],
                    "reason": (f"the cell this door opens onto, ({fx},{y},{fz}), is "
                               + (f"{here} -- something is standing in the way of "
                                  f"walking through it" if blocked else
                                  f"over {under or 'nothing'}, which is not ground a "
                                  f"person can stand on")
                               + " -- nothing has been laid; clear it, lay the "
                                 "doorstep first, or put the door somewhere else")}
        sides = ((1, 0), (-1, 0)) if facing in ("north", "south") else ((0, 1), (0, -1))
        open_sides = []
        for dx, dz in sides:
            here = self.get_block(x + dx, y, z + dz)
            over = self.get_block(x + dx, y + 1, z + dz)
            if not _is_surface(here) or not _is_surface(over):
                open_sides.append([x + dx, y, z + dz])
        if open_sides and jamb == "refuse":
            return {"ok": False, "jambs": 0, "open_sides": open_sides,
                    "reason": f"{len(open_sides)} side(s) of this doorway are open air; "
                              f"the door would stand at the end of a wall run"}
        built = 0
        for (jx, jy, jz) in open_sides:
            for dy in (0, 1):
                self.place_block(jx, jy + dy, jz, full)
                built += 1
        self.place_block(x, y, z, f"{leaf}[facing={facing},half=lower,hinge={hinge}]")
        self.place_block(x, y + 1, z, f"{leaf}[facing={facing},half=upper,hinge={hinge}]")
        if lintel:
            self.place_block(x, y + 2, z, full if lintel is True else lintel)
        if threshold:
            self.place_block(x, y - 1, z, threshold)
        return {"ok": True, "jambs": built, "open_sides": open_sides,
                "front": [fx, y, fz],
                "reason": (f"built {built} block(s) of jamb where the wall stopped short"
                           if built else "the wall already framed it on both sides")}

    def window(self, x: int, y: int, z: int, facing: str, mat: str,
               width: int = 2, height: int = 2, glass: str | None = None,
               sill: bool = True, shutters: bool = False) -> None:
        """Glazed opening with a sill and lintel, recessed so it reads with depth.

                `glass` defaults to what fills an opening this wide: a pane at one block, glass
                beyond. See `_glazing`.
                
        """
        glass = self._glazing(glass, width, 0)
        full, stairs, slab = material(mat)
        horiz = facing in ("north", "south")
        for i in range(width):
            for j in range(height):
                if horiz:
                    self.place_block(x + i, y + j, z, glass)
                else:
                    self.place_block(x, y + j, z + i, glass)
        if sill:
            for i in range(-1, width + 1):
                if horiz:
                    self.place_block(x + i, y - 1, z, f"{slab}[type=top]")
                    self.place_block(x + i, y + height, z, full)
                else:
                    self.place_block(x, y - 1, z + i, f"{slab}[type=top]")
                    self.place_block(x, y + height, z + i, full)
        if shutters:
            a, b = ("east", "west") if horiz else ("south", "north")
            if horiz:
                self.place_block(x - 1, y, z, f"{mat}_trapdoor[facing={a},open=true]")
                self.place_block(x + width, y, z, f"{mat}_trapdoor[facing={b},open=true]")
            else:
                self.place_block(x, y, z - 1, f"{mat}_trapdoor[facing={a},open=true]")
                self.place_block(x, y, z + width, f"{mat}_trapdoor[facing={b},open=true]")
