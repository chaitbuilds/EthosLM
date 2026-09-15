"""A voice is data, and the model may write one.

A voice was a hand-written entry in a dict in `styles.py`, which meant the place spec
could only ever *choose* a palette and never *make* one. not for want of a type, or a
plan, or ground, but because the palette was source code.

So a voice is a file:

    voices/<name>.json
      {"roles":  {"wall", "footing", "frame", "roof", "trim", "floor"}
                 ...and, optional, "ground" -- what the ground between the buildings
                 is -- and "wall_alt", a second face a share of them are built in,
       "roof":   {"profile", "ends", "eave", "tiers"},
       "roof_civic": the same four keys, optional -- the roof of a civic building,
       "notes":  {"blurb", "construction", "roofs", "ground", "signature"}}

`roles` is the palette -- one material family per role, and the six roles are exactly
`buildlib._MAT_ROLES`, so a voice names every material a building is made of and there
is no role left to default. `roof` is the silhouette as `roof()`'s own four parameters,
because a voice that says "irimoya, two tiers, upturned eaves" is describing something
the library has been able to draw for several generations of the shell and had no way
to be *told*. `notes` is the prose every brief already carried.

**A voice is validated on load**, which is the half that makes it safe to let a model
write one. Five of the six roles are laid in stairs and slabs somewhere in the shell --
a roof is stairs, a flight is stairs, a plinth is slabs, a band is slabs -- so those
five must be material *families* in `prims.MATERIALS`, and a family is by definition a
material in all three shapes. `floor` is only ever laid as a cube, so it may be a bare
block id. That distinction is not invented here: it is which of `_material()` and
`_solid()` the library already calls on each role, and the material refusal is what
stands between a role and something it cannot be shaped from -- a thatched roof once
came out in stone brick and every instrument called it correct.

The value range is recorded rather than judged. A palette of six near-blacks is a
legitimate voice (`blackstone_and_ash` is close to one) and this file will not invent a
contrast rule; what it will do is measure the spread, so that the one instrument that
*does* care -- the place read's ground contrast -- has a number to read.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIR = os.path.join(ROOT, "voices")

#: The six roles a voice names, in the order a building is made in. The same tuple as
#: `buildlib._MAT_ROLES`; asserted equal by `test_types.py` rather than imported, so a
#: voice file does not depend on the build library to be read.
ROLES = ("wall", "footing", "frame", "roof", "trim", "floor")

#: The roles the library lays in stairs or slabs as well as cubes, and so the roles that
#: must name a family `prims.material()` knows. Each one is a line in `buildlib`:
#: `roof()` takes a family; `flight(mat=)` gets `wall`; `plinth`, `steps` and the
#: chimney cap get `footing`; the jetty and the bracket course get `frame`; the band and
#: the lintel get `trim`.
SHAPED = ("wall", "footing", "frame", "roof", "trim")

#: And the role that is only ever a cube, so a voice may name a block that has no stairs
#: -- rammed earth, concrete powder, a carpet of terracotta.
SOLID = ("floor",)

#: The six above are what a *building* is made of; the ground a place designs -- the
#: terraces, the feathers, the ramps -- belonged to no role at all, so it was dressed in
#: whatever the library could sample and a city on a plain came out on one block of
#: quarry stone from wall to wall. `ground` is **optional**, and its absence is not a
#: default: a voice that does not name one gets the setting's own surface at each
#: column, which is the plain it stands on (`buildlib.Builder.setting_cover`). Laid as a
#: cube and never shaped, so it is validated as `floor` is -- a plain's grass has no
#: stairs. The prose `ground` note stays prose and is read by the briefs alone. ...and
#: **a second wall material**, also optional. A ring built from one `wall` family is one
#: extruded building thirty times over, and a type cannot vary it without naming one.
#: `TypeBuilder` faces `buildlib.WALL_ALT_SHARE` of them in this one, by the part's own
#: seed, and the type is never told which it got. Shaped like `wall` is -- stairs, slabs
#: and cubes -- so it is validated as a family.
OPTIONAL = ("ground", "wall_alt")

#: Which of the optional roles must be a material **family** rather than a block id.
OPTIONAL_SHAPED = ("wall_alt",)

#: What a voice may say about its roof, and the default of each. These are `roof()`'s
#: own keyword arguments; `None` for `profile` and `ends` is the roof style's own, which
#: is what keeps every stored program byte-identical.
ROOF_KEYS = {"profile": None, "ends": None, "eave": "straight", "tiers": 1}

#: **A second silhouette, for what a place holds at its middle.** Demo-polish, 2a. A
#: voice carried one silhouette, so a capital's throne hall stood under the same roof as
#: its smallest house and nothing in the roofscape said which was which (the render
#: look's finding 6). `roof_civic` is optional, takes the same four keys as `roof` and
#: is validated by the same rules; `TypeBuilder` hands it to `roof()` and `building()`
#: in place of `roof` for a type whose `ROLE` is `civic`, and falls back to `roof` where
#: the voice is silent. No type names any of it.
ROOF_CIVIC = "roof_civic"

#: The prose a brief reads. Optional, every one of them: a voice with no notes is a
#: palette, which is a legitimate thing for a model to author in one line.
NOTE_KEYS = ("blurb", "construction", "roofs", "ground", "signature")

#: `hall` is `civic` and civic is admitted into every tradition, so the great hall of an
#: east Asian capital carried two red masonry stacks to its ridge; nothing anywhere tied
#: a chimney to a form or a voice. A voice may now say `chimney: true|false`; where it
#: says nothing the place's form decides (`chimney_default`). The voice *permits*: the
#: type still decides per building by purpose and seed, as `cottage` and `hall` do and
#: `minka` never does, so a permitted chimney is never every building.
CHIMNEY_BY_FORM = {"east_asian": False, "european_vernacular": True}


def chimney_default(form: str | None) -> bool:
    """Whether buildings in a place of this form carry chimneys when the voice does
    not say: no for the east Asian tradition, yes for the European one and for a
    place with no tradition named -- which is what every building did before."""
    return bool(CHIMNEY_BY_FORM.get(form or "", True))

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class VoiceError(ValueError):
    """A voice that will not load, with the role and the family named."""


def _material_families() -> dict:
    from .prims import MATERIALS
    return MATERIALS


def validate(voice: dict, where: str = "a voice") -> dict:
    """Check one voice and return it filled out. Refuses by name.

        Every refusal names the role and what was in it, because the caller is as likely to
        be a model that just wrote the file as a person who just edited it, and "invalid
        voice" is not something either of them can act on.
        
    """
    from .prims import family
    if not isinstance(voice, dict):
        raise VoiceError(f"{where}: a voice is a JSON object with 'roles', 'roof' and "
                         f"'notes' in it, not a {type(voice).__name__}")
    # `value` is this function's **own output** -- the luma spread it measures off the
    # roles -- and it is accepted on the way in and recomputed rather than refused. A
    # validator that will not read what it just wrote is not a validator: the first
    # voice a place spec ever authored went `read_spec` -> validate ->
    # `stage_place_spec` -> `author` -> validate and was refused on the second pass,
    # naming a field the model had never written. the case is kept.
    unknown = sorted(set(voice) - {"name", "roles", "roof", ROOF_CIVIC, "notes", "value",
                                   "chimney"})
    if unknown:
        raise VoiceError(f"{where}: no such field as {unknown[0]!r}; a voice has "
                         f"'roles', 'roof', 'notes' and, if it wants them, "
                         f"'{ROOF_CIVIC}' and 'chimney'")
    chimney = voice.get("chimney")
    if chimney is not None and not isinstance(chimney, bool):
        raise VoiceError(f"{where}: 'chimney' is true, false, or absent for the place's "
                         f"form to decide, not {chimney!r}")
    roles = voice.get("roles")
    if not isinstance(roles, dict):
        raise VoiceError(f"{where}: 'roles' is an object naming a material for each of "
                         f"{', '.join(ROLES)}")
    missing = [r for r in ROLES if not roles.get(r)]
    if missing:
        raise VoiceError(
            f"{where}: 'roles' does not name a material for {missing[0]!r}. A voice "
            f"names all six -- {', '.join(ROLES)} -- because a role that defaults to "
            f"the wall is a decision nobody made: every building in the place then has "
            f"its floor the colour of its walls and nothing says why")
    spare = sorted(set(roles) - set(ROLES) - set(OPTIONAL))
    if spare:
        raise VoiceError(
            f"{where}: 'roles' has a role called {spare[0]!r} and the library has no "
            f"such role. The six the shell is made of are {', '.join(ROLES)}, and "
            f"{', '.join(OPTIONAL)} is the one a voice may add; anything else a voice "
            f"wants to say belongs in 'notes'")
    fams = _material_families()
    for role in SHAPED + tuple(r for r in OPTIONAL_SHAPED if roles.get(r)):
        mat = str(roles[role])
        fam = family(mat)
        if fam is None:
            raise VoiceError(
                f"{where}: 'roles.{role}' is {mat!r}, which is not a material family: "
                f"there are no {mat} stairs and no {mat} slab, and the {role} is laid "
                f"in stairs and slabs as well as cubes. Name one of "
                f"{', '.join(sorted(fams))} -- or, if this exact block is wanted "
                f"somewhere no shape is needed, that is what 'roles.floor' is for")
    for role in SOLID + tuple(r for r in OPTIONAL
                              if roles.get(r) and r not in OPTIONAL_SHAPED):
        from .prims import solid
        try:
            solid(str(roles[role]))
        except ValueError as e:
            raise VoiceError(f"{where}: 'roles.{role}' is {roles[role]!r} -- {e}") from e
    roof = voice.get("roof") or {}
    if not isinstance(roof, dict):
        raise VoiceError(f"{where}: 'roof' is an object of "
                         f"{', '.join(sorted(ROOF_KEYS))}, not a "
                         f"{type(roof).__name__}")
    spare = sorted(set(roof) - set(ROOF_KEYS))
    if spare:
        raise VoiceError(f"{where}: 'roof' has no parameter called {spare[0]!r}; it "
                         f"takes {', '.join(sorted(ROOF_KEYS))}, which are roof()'s own")
    out_roof = dict(ROOF_KEYS)
    out_roof.update({k: v for k, v in roof.items()})
    _check_roof(out_roof, where)
    civic = voice.get(ROOF_CIVIC)
    out_civic = None
    if civic is not None:
        if not isinstance(civic, dict):
            raise VoiceError(f"{where}: '{ROOF_CIVIC}' is an object of "
                             f"{', '.join(sorted(ROOF_KEYS))}, the roof of a civic "
                             f"building, not a {type(civic).__name__}")
        spare = sorted(set(civic) - set(ROOF_KEYS))
        if spare:
            raise VoiceError(f"{where}: '{ROOF_CIVIC}' has no parameter called "
                             f"{spare[0]!r}; it takes {', '.join(sorted(ROOF_KEYS))}, "
                             f"which are roof()'s own")
        out_civic = dict(ROOF_KEYS)
        out_civic.update({k: v for k, v in civic.items()})
        _check_roof(out_civic, f"{where} ({ROOF_CIVIC})")
    notes = voice.get("notes") or {}
    if not isinstance(notes, dict):
        raise VoiceError(f"{where}: 'notes' is an object of prose, not a "
                         f"{type(notes).__name__}")
    spare = sorted(set(notes) - set(NOTE_KEYS))
    if spare:
        raise VoiceError(f"{where}: 'notes' has no field called {spare[0]!r}; it takes "
                         f"{', '.join(NOTE_KEYS)}")
    out = {"name": voice.get("name"),
           "roles": {**{r: str(roles[r]) for r in ROLES},
                     **{r: str(roles[r]) for r in OPTIONAL if roles.get(r)}},
           "roof": out_roof,
           ROOF_CIVIC: out_civic,
           "chimney": chimney,
           "notes": {k: str(v) for k, v in notes.items() if v}}
    out["value"] = value_range(out["roles"])
    return out


def _check_roof(roof: dict, where: str) -> None:
    """The four roof parameters, against `roof()`'s own vocabulary."""
    from .prims import Primitives
    ends = roof.get("ends")
    if ends is not None:
        got = (ends, ends) if isinstance(ends, str) else tuple(ends)[:2]
        for e in got:
            if e not in Primitives.END_KINDS:
                raise VoiceError(f"{where}: 'roof.ends' is one of "
                                 f"{', '.join(Primitives.END_KINDS)} (or a pair of "
                                 f"them), not {e!r}")
        roof["ends"] = list(got) if not isinstance(ends, str) else ends
    eave = roof.get("eave", "straight")
    if eave not in Primitives.EAVES:
        raise VoiceError(f"{where}: 'roof.eave' is one of "
                         f"{', '.join(Primitives.EAVES)}, not {eave!r}")
    tiers = roof.get("tiers", 1)
    if not isinstance(tiers, int) or isinstance(tiers, bool) or not 1 <= tiers <= 4:
        raise VoiceError(f"{where}: 'roof.tiers' is a whole number of stacked roofs "
                         f"from 1 to 4, not {tiers!r}")
    prof = roof.get("profile")
    if prof is None:
        return
    try:
        segs = [(int(a), int(b)) for a, b in prof]
    except Exception as e:                       # noqa: BLE001 -- reported by name
        raise VoiceError(f"{where}: 'roof.profile' is a list of (rise, run) pairs from "
                         f"the eave to the ridge, not {prof!r}") from e
    if not segs or any(b <= 0 or a < 0 for a, b in segs):
        raise VoiceError(f"{where}: 'roof.profile' segments are (rise, run) with a run "
                         f"of at least one block: {prof!r}")
    roof["profile"] = [list(s) for s in segs]


def value_range(roles: dict) -> dict:
    """How light and how dark this palette is, measured, not judged.

        The one number a palette has that nobody had written down. `preview.block_colour`
        is the project's own block-to-colour table and this is its luma, so a voice's spread
        is measured by the same thing that draws the previews. Recorded rather than
        enforced: `blackstone_and_ash` is a legitimate voice and it is nearly flat.
        
    """
    from .preview import block_colour
    from .prims import solid
    out = {}
    for role, mat in roles.items():
        try:
            block = solid(str(mat))
        except ValueError:
            block = str(mat)
        r, g, b = block_colour(block)
        out[role] = round((0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0, 3)
    vals = sorted(out.values())
    return {"by_role": out, "lightest": vals[-1], "darkest": vals[0],
            "spread": round(vals[-1] - vals[0], 3)}


def path_for(name: str) -> str:
    return os.path.join(DIR, f"{name}.json")


def load(name: str, directory: str | None = None) -> dict:
    """One voice off disk, validated. Raises `VoiceError` naming the file."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise VoiceError(f"a voice is named in lower case with underscores, like "
                         f"'ochre_and_green_tile', not {name!r}")
    p = os.path.join(directory or DIR, f"{name}.json")
    if not os.path.exists(p):
        raise VoiceError(f"there is no voice called {name!r}: {os.path.relpath(p, ROOT)}"
                         f" is not on disk. The voices this project has are "
                         f"{', '.join(names(directory)) or 'none'}")
    try:
        raw = json.load(open(p))
    except json.JSONDecodeError as e:
        raise VoiceError(f"{os.path.relpath(p, ROOT)}: not JSON -- {e}") from e
    raw.setdefault("name", name)
    if raw.get("name") != name:
        raise VoiceError(f"{os.path.relpath(p, ROOT)}: the file calls itself "
                         f"{raw['name']!r} and it is named {name!r}; the file name is "
                         f"the voice's name")
    return validate(raw, where=os.path.relpath(p, ROOT))


def names(directory: str | None = None) -> list:
    d = directory or DIR
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def load_all(directory: str | None = None) -> dict:
    """Every voice on disk, validated. A bad file stops the load and names itself."""
    return {n: load(n, directory) for n in names(directory)}


def author(name: str, voice: dict, directory: str | None = None) -> dict:
    """Write a voice the model authored, after validating it.

        The point of the whole change: the place spec may hand back a voice that does not
        exist yet, and this is where it becomes one. Validated **before** it is written, so
        a refused voice leaves nothing on disk to be picked up by the next run.
        
    """
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise VoiceError(f"a voice is named in lower case with underscores, like "
                         f"'ochre_and_green_tile', not {name!r}")
    got = validate({**voice, "name": name}, where=f"the authored voice {name!r}")
    d = directory or DIR
    os.makedirs(d, exist_ok=True)
    out = {"name": name, "roles": got["roles"], "roof": got["roof"],
           "notes": got["notes"]}
    if got.get(ROOF_CIVIC):
        out[ROOF_CIVIC] = got[ROOF_CIVIC]
    with open(os.path.join(d, f"{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return got
