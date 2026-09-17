"""Style seeds — the mechanism against boilerplate.

So each run gets a seed: a small palette, a roof direction, a massing rule and one
signature move. These are starting points the model may depart from, not a spec. The
point is to start it somewhere other than the attractor."""
from __future__ import annotations

import os
from collections.abc import Mapping

STYLES = {
    "lowland_stone": {
        "blurb": "Heavy lowland stone cottage. Squat, thick-walled, built to sit out weather.",
        "palette": {"wall": "cobblestone", "trim": "stone_brick", "roof": "dark_oak",
                    "floor": "spruce", "accent": "mossy_cobblestone"},
        "roof": "steep gable or half-hip, pitch (2,1)",
        "massing": "low and wide, one and a half storeys, roof taller than the walls",
        "signature": "an outshot lean-to on one long side, roofed at a shallower pitch",
    },
    "chalk_and_timber": {
        "blurb": "Pale chalk render over a dark timber frame. Tall, narrow, upper floor jettied.",
        "palette": {"wall": "quartz", "trim": "dark_oak", "roof": "deepslate_tile",
                    "floor": "oak", "accent": "stripped_dark_oak_log"},
        "roof": "gable, pitch (1,1), deep overhang",
        "massing": "narrow frontage, three storeys, each overhanging the one below",
        "signature": "a first-floor oriel window carried on brackets",
    },
    "sandstone_courtyard": {
        "blurb": "Dry-country sandstone. Flat roofs, parapets, deep window reveals, shade.",
        "palette": {"wall": "sandstone", "trim": "smooth_sandstone", "roof": "sandstone",
                    "floor": "cut_sandstone", "accent": "terracotta"},
        "roof": "flat with a parapet, plus one shed-roofed loggia",
        "massing": "blocks of differing height around an open courtyard",
        "signature": "a colonnade of square piers along one side of the courtyard",
    },
    "copper_and_brick": {
        "blurb": "Brick with oxidised copper roofing. Industrial, tall windows, iron.",
        "palette": {"wall": "brick", "trim": "polished_blackstone_brick",
                    "roof": "oxidized_copper", "floor": "stone_brick", "accent": "iron_block"},
        "roof": "mansard, so the top storey lives inside the roof",
        "massing": "a tall rectangular block with a lower wing at right angles",
        "signature": "dormers punched through the mansard, evenly spaced",
    },
    "deep_forest_dark": {
        "blurb": "Dark wood and blackstone, sunk into the trees. Little glass, much shadow.",
        "palette": {"wall": "dark_oak", "trim": "blackstone", "roof": "deepslate_brick",
                    "floor": "dark_oak", "accent": "mossy_cobblestone"},
        "roof": "gambrel, steep below and shallow above",
        "massing": "built into the slope so one side is a storey lower than the other",
        "signature": "a covered walkway on posts leading to the door",
    },
    "cliff_fortress": {
        "blurb": "Defensive grey stone. Battered walls, small openings, hard geometry.",
        "palette": {"wall": "stone_brick", "trim": "deepslate_brick", "roof": "deepslate_tile",
                    "floor": "andesite", "accent": "chiseled_stone_bricks"},
        "roof": "hip, pitch (1,2), or flat with crenellation",
        "massing": "a main mass with a taller round tower fused into one corner",
        "signature": "the wall base battered — thicker at the bottom, stepping in as it rises",
    },
    "cherry_pavilion": {
        "blurb": "Light timber pavilion. Wide shallow eaves, open sides, raised floor.",
        "palette": {"wall": "cherry", "trim": "stripped_cherry_log", "roof": "cherry",
                    "floor": "birch", "accent": "white_concrete"},
        "roof": "hip, pitch (1,3), very wide overhang",
        "massing": "raised on a plinth above the ground, open on at least one side",
        "signature": "the roof extending far past the walls to make a shaded veranda",
    },
    "harbour_prismarine": {
        "blurb": "Sea-stone and weathered wood. Damp, salt-worn, half over the water.",
        "palette": {"wall": "prismarine", "trim": "dark_prismarine", "roof": "spruce",
                    "floor": "spruce", "accent": "copper"},
        "roof": "shed roofs at differing heights, none matching",
        "massing": "irregular — parts on land, parts on piles over water",
        "signature": "timber piles and a deck standing out over the water",
    },
}


def card(name: str) -> str:
    """The seed text handed to the model. Deliberately short and non-binding."""
    s = STYLES[name]
    pal = "\n".join(f"    {k:7} {v}" for k, v in s["palette"].items())
    return f"""## Starting direction (a seed, not a spec)

{s['blurb']}

Suggested palette — keep to about this many materials, vary the texture not the count:
{pal}

Roof direction:  {s['roof']}
Massing:         {s['massing']}
Signature move:  {s['signature']}

Depart from any of this where the site or the brief argues for something else. What
matters is that the result is coherent and specific, not that it matches the seed.
Do not produce a generic timber-framed cottage with a 45-degree gable roof.
"""


# --------------------------------------------------------------------------- voices A
# *voice* is the settlement-scale version of a seed, and it exists because the two
# findings this project already has pull against each other. Variation has to be forced.
# Palette drifted 16 -> 21 -> 18 -> 15 -> 25 families across five passes and the shared
# plan's palette slowed it without holding it. Uniformity has to be enforced. Handing
# every structure its own seed satisfies the first and destroys the second. The
# resolution is the one real places use: **uniform in material, varied in form.** A
# village reads as one place because the same three or four materials recur everywhere;
# it reads as worth walking through because the massing, the roofs and the signature
# moves differ -- and differ *by what a building is for*, which is the purpose-
# conditioned detail the rebuild spec calls its hardest open item. So a voice fixes the
# materials and the construction logic, and a *register* varies the form by purpose. The
# palette being declared is what finally makes it checkable: S001 now measures what a
# pass placed against what the settlement said it was made of, instead of counting
# families against a number out of a craft book.

# A voice used to be a literal in this file, which is why the place spec could choose a
# palette and never make one. It is `voices/<name>.json` now, validated on load by
# `ethoslm.voices`, and this is the reading view every pass written before A2 still asks
# its questions through: `VOICES[name]["palette"]` is the six roles and `["blurb"]`,
# `["construction"]`, `["roofs"]`, `["ground"]` and `["signature"]` are the notes.
# `["roof"]` is new and is the silhouette as `roof()`'s own parameters. It is a live
# view of the directory rather than a snapshot, because a voice the spec step *authors*
# has to be visible to the planner in the same run that wrote it.

class _VoiceTable(Mapping):
    """Every voice under `voices/`, re-read when the directory changes."""

    def __init__(self, directory=None):
        self._dir = directory
        self._stamp = None
        self._cache: dict = {}

    def _load(self) -> dict:
        from . import voices as _voices
        d = self._dir or _voices.DIR
        names = _voices.names(d)
        stamp = tuple((n, os.path.getmtime(os.path.join(d, f"{n}.json")))
                      for n in names)
        if stamp != self._stamp:
            self._cache = {n: _view(_voices.load(n, d)) for n in names}
            self._stamp = stamp
        return self._cache

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __repr__(self):
        return f"<voices: {', '.join(self._load())}>"


def _view(voice: dict) -> dict:
    """One validated voice, in the shape every pass before A2 reads."""
    out = {"name": voice["name"], "palette": dict(voice["roles"]),
           "roles": dict(voice["roles"]), "roof": dict(voice["roof"]),
           "roof_civic": (dict(voice["roof_civic"]) if voice.get("roof_civic") else None),
           "chimney": voice.get("chimney"),
           "ceremonial": bool(voice.get("ceremonial")),
           "value": voice["value"]}
    out.update({k: v for k, v in voice["notes"].items()})
    for k in ("blurb", "construction", "roofs", "ground", "signature"):
        out.setdefault(k, "")
    return out


VOICES = _VoiceTable()


#: The fields of a voice's silhouette, as `roof()` takes them. What `partner_voice`
#: counts differences over.
SILHOUETTE_FIELDS = ("ends", "eave", "tiers", "profile")


def explicit_silhouette(name: str) -> bool:
    """Does this voice declare a silhouette of its own -- a roof style (`ends`) or a
    profile -- rather than leaving `roof()` to its default?"""
    r = VOICES[name]["roof"] or {}
    return bool(r.get("ends") or r.get("profile"))


def silhouette_distance(a: str, b: str) -> int:
    """How unlike two voices' silhouettes are: the fields of `SILHOUETTE_FIELDS` on
    which they differ, plus one where one declares a silhouette and the other none.
    A silent voice against an explicit one differs on everything."""
    ra, rb = dict(VOICES[a]["roof"] or {}), dict(VOICES[b]["roof"] or {})
    if explicit_silhouette(a) != explicit_silhouette(b):
        return len(SILHOUETTE_FIELDS) + 1
    return sum(1 for f in SILHOUETTE_FIELDS if ra.get(f) != rb.get(f))


def partner_voice(name: str) -> str:
    """The voice on disk whose silhouette is least like this one's -- **derived from
        the directory, never named**. v2, C0.

        Among the voices that declare a silhouette, the one at the greatest
        `silhouette_distance` from `name`, ties by name; a partner with an explicit
        silhouette and not a silent one, because a type whose own default silhouette is
        its author's stands in a silent voice exactly as it stands in that one. Where no
        other explicit voice is on disk, the voice at the greatest distance of any, and
        where `name` is the only voice, `name` itself.
        
    """
    others = [v for v in sorted(VOICES) if v != name]
    if not others:
        return name
    explicit = [v for v in others if explicit_silhouette(v)]
    pool = explicit or others
    return max(pool, key=lambda v: (silhouette_distance(name, v), [-ord(c) for c in v]))


def silent_voice() -> str:
    """The first voice on disk, by name, that declares no silhouette -- what a checker
    stands a type in when nobody said a voice and there is no place: the type's own
    `roof()` default, under the plainest palette the directory has. The first by name
    of all where every voice is explicit."""
    names = sorted(VOICES)
    if not names:
        raise ValueError("there are no voices under voices/")
    return next((v for v in names if not explicit_silhouette(v)), names[0])



#: How a building's *purpose* changes its form inside a fixed voice. This is the
#: variation that does not cost coherence: same materials, different register. Keyed by
#: substrings of the planner's own `kind` field, so it needs no new vocabulary.
REGISTERS = {
    "hall": "civic: the widest span and the tallest ridge in the settlement, near-"
            "symmetrical, the one building whose door is centred and whose approach is "
            "formal. It should be legible as the centre from anywhere it is visible.",
    "green": "civic open ground: almost no building -- paving, a wall or kerb, trees, "
             "and one small built thing at its centre worth walking to.",
    "square": "civic open ground: as a green, but paved throughout and edged by the "
              "buildings that face it.",
    "market": "civic: mostly roof and posts, open on at least two sides, floor paved "
              "rather than boarded.",
    "temple": "civic: taller than it is wide, the only building permitted a vertical "
              "emphasis, openings narrow and high.",
    "shrine": "civic: small, solid, disproportionately well built for its size.",
    "tower": "defensive: no ornament, openings minimal, mass tapering or battered, "
             "visible from the whole settlement and meant to be steered by.",
    "watch": "defensive: as the tower, but squat and part of whatever it guards.",
    "gate": "defensive: two masses with a passage between them, the roadway continuing "
            "through rather than stopping.",
    "granary": "working: raised clear of the ground, few openings, the structure of the "
               "raising visible.",
    "barn": "working: one large volume, big doors on the long side, no upper storey.",
    "byre": "working: low, long, open along one side, the yard part of the building.",
    "smith": "working: open-fronted, a chimney or flue as the tallest element, the yard "
             "as important as the shed.",
    "kiln": "working: a solid mass with a stack, surrounded by clear ground.",
    "mill": "working: tall, narrow, one moving-looking element dominating.",
    "well": "working: small, central, built up rather than dug down.",
    "cistern": "working: mostly below the ground line, its roof a place people stand on.",
    "house": "dwelling: modest, repeated with real differences -- width, storey count, "
             "roof pitch and door side should vary between neighbours in a row.",
    "dwelling": "dwelling: as house.",
    "croft": "dwelling: a house with its yard and outbuilding treated as one composition.",
    "row": "dwelling: a terrace under one roofline whose units differ in width and "
           "frontage, not a repeated stamp.",
    "guest": "dwelling: larger than a house, more openings, a public face on the lane.",
    "inn": "dwelling: the largest dwelling, a yard, and the most glazing in the town.",
    "farm": "outlying: loose, fenced, buildings placed for use rather than for frontage.",
    "garden": "outlying: walls and ground work, almost no roof.",
    "terrace": "outlying: retaining walls and steps only -- landform, not building.",
    "barrow": "outlying: earth and stone, no timber, no openings.",
}


#: What a room of each purpose **contains**, in plain words. The registers above say
#: what a building looks like from outside and nothing about what is in it, and
#: `prims.py` carried no furniture vocabulary at all -- no hearth, no bench, no store.
#: *some are good and some are just random blocks, it might not be reaching for the
#: right blocks*. That is exactly what it was doing. This is the vocabulary, not a
#: layout: it names the things, the builder still decides where they go.
FITTINGS = {
    "hall": "a hearth on the long wall, trestles or benches down the sides, a raised "
            "place at one end, and light hung high rather than stood on the floor",
    "temple": "one focus at the far end from the door and nothing else competing with it",
    "shrine": "a single offering place, lit",
    "market": "stalls or trestles under the roof, storage crates against the back",
    "tower": "a way up that a person can actually climb, a floor to stand on at the top, "
             "somewhere to keep fuel or arms on the way",
    "watch": "as the tower, plus a fire or lamp at the top",
    "gate": "a guard room with a seat, a light, and somewhere to put a lamp at night",
    "granary": "grain bins or barrels stood in rows clear of the wall, a boarded loft, "
               "and a hoist or ladder to reach it",
    "barn": "fodder stacked at one end, tools on the wall, cart room in the middle",
    "byre": "stall divisions along one side, a trough, fodder above or behind, a lamp",
    "smith": "a furnace or forge, an anvil on a block clear of it, a water barrel to "
             "quench in, fuel stored dry, tools hung within reach of the anvil",
    "kiln": "the firing chamber, fuel stacked outside it, ware set out to dry",
    "mill": "the stones and the gearing they drive, sacks stacked, a hoist under the roof",
    "well": "the shaft, something to draw with, somewhere to stand a vessel",
    "cistern": "the water, a place to draw from it, steps down to that place",
    "house": "a hearth, somewhere to sleep, somewhere to eat, storage, and light -- and "
             "no two houses in a row laid out identically",
    "dwelling": "as house",
    "croft": "as house, plus the working end: tools, fodder, storage",
    "row": "as house, and each unit different from its neighbour",
    "inn": "as house at greater scale: a common room with the hearth and tables, "
           "separate sleeping rooms above, a store",
    "guest": "as house, with more beds and less working equipment",
}


def register_for(kind: str) -> str:
    """The register for a structure, from the planner's own words for what it is."""
    k = (kind or "").lower()
    for key, text in REGISTERS.items():
        if key in k:
            return text
    return ("general: let the purpose decide the form -- span, height, how open it is, "
            "and where its door faces are all consequences of what happens inside.")


def fittings_for(kind: str) -> str:
    """What a building of this purpose has inside it, from the planner's own words.

        Empty string where nothing is known: a check that fires on a barrow because nobody
        wrote down what is inside a barrow would be the checker inventing a craft rule again.
        
    """
    k = (kind or "").lower()
    for key, text in FITTINGS.items():
        if key in k:
            return text
    return ""


def _colour_note(material: str) -> str:
    """` (mid warm grey)` for a material the colour table knows, and nothing for one it
        does not. v2, A7.

        The card has always named the materials and never said what colour they are, so
        every pass that reads it -- a builder, a type's author, the judge's brief -- has had
        to know Minecraft's block list by heart. `block_colour` has had the answer all
        along. A family with no reading is named **without** a colour rather than with a
        guessed one, which is the rule the family list already follows: a wrong colour is
        worse than none, and magenta is what "nobody has met this block" looks like.
        
    """
    from .preview import UNKNOWN, block_colour
    try:
        from .pipeline.stages_plan import colour_of
        from .prims import solid
        block = solid(material)
    except Exception:                            # noqa: BLE001 -- name it without one
        return ""
    if block_colour(block) == UNKNOWN:
        return ""
    return f" ({colour_of(block)})"


def voice_card(name: str, site_surface: dict | None = None) -> str:
    """The settlement's voice, as the text every pass reads."""
    v = VOICES[name]
    pal = "\n".join(f"    {k:9} {mat}{_colour_note(mat)}"
                    for k, mat in v["palette"].items())
    ground = ""
    if site_surface:
        # The surface census samples the top block of a column, which in woodland is a
        # trunk. Telling a builder the ground is made of oak, and so not to build in it,
        # is advice about the trees.
        from .observe import _is_vegetation
        ground_only = {k: v for k, v in site_surface.items() if not _is_vegetation(k)}
        top = sorted((ground_only or site_surface).items(), key=lambda kv: -kv[1])[:3]
        ground = ("\nThe ground here is mostly "
                  + ", ".join(k for k, _ in top)
                  + ". The walls and roofs above must not be that material or a variant "
                    "of it. A settlement the colour of its own hillside disappears at "
                    "distance -- it happened once, every measurement said the build was "
                    "fine, and only a person looking at it from far enough away could "
                    "tell.\n")
    return f"""## The voice of this settlement

{v['blurb']}

**Materials.** These, and as few others as you can manage. Variants of one material
(mossy, cracked, polished, stripped) and its shapes (stairs, slabs, walls) count as the
same material, so use them freely -- it is *new families* that make a place look like a
sample book.

{pal}
{ground}
**Construction.** {v['construction']}

**Roofs.** {v['roofs']}

**Meeting the ground.** {v['ground']}

**The signature move**, which should appear somewhere in most buildings and never
identically: {v['signature']}

This is the voice of the whole settlement, not of one building. Every structure is made
of these materials and built this way. What differs between them is **form, driven by
purpose** -- span, height, how open they are, how they meet the ground -- and that
difference is where all the variety comes from. Two buildings in the same voice that
differ only in footprint are the failure this is here to prevent.
"""
