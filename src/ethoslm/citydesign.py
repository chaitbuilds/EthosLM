"""**The spatial design of a place**: the record the model writes and the compiler consumes.

When the model only chooses types and counts, fixed arithmetic lays everything else --
`concentric_layout` squares, `district_asks` strips of cardinal lanes,
`_lay_court_city`'s fixed court sequence -- and no prompt can express a design outside
that vocabulary. This record is the intermediate the model *designs in*: hierarchy,
boundaries, streets, block grain, courts, landmarks, entrances, levels, and the visual
choice of materials -- at the level of organisation, never as voxels or a city-sized
coordinate list. `ethoslm.cityresolve` resolves it deterministically (dimensions, lots,
feasibility, ground) and returns what it could not honour as findings addressed to the
design, which is the parent that can change them.

Ownership, stated once (and printed into the design prompt):

  - **the model decides** the silhouette and boundary outline, the extent within the
    measured bounds, the rings and what each is for, the street hierarchy and its
    orientation, each ring's grain and its parameters, the landmarks and where they
    stand, the monument's composition, the ground policy per ring, and the palette
    after looking at the comparison sheets;
  - **the compiler resolves** coordinates, wall paths and gate runs, street rasters,
    blocks, lots and their sites, the nested compositions' dimensions, the designed
    ground and its costs, region tiling, and every shortfall -- a lot that does not fit,
    a ring too narrow for its grain, a retaining wall over its bound -- as a finding.

Schema `ethoslm.design/1` (see `SCHEMA_DOC` for the prose the design job reads).
"""
from __future__ import annotations

import copy
import json

from . import boundary

SCHEMA = "ethoslm.design/1"

#: The grains a ring (or a ward of one) can be composed in. Each is a reusable composer
#: in `cityresolve`; the parameters are the design's, the lot sizes the forms'.
GRAINS = {
    "courts": {
        "what": "lanes lined on both sides by courtyard houses standing back to back "
                "between them, crossed by streets whose frontage is shop houses; a "
                "hutong / siheyuan grain",
        "params": {"lane_dir": ("choice", ["east_west", "north_south"]),
                   "lane_width": ("int", 3, 5), "street_width": ("int", 5, 9),
                   "street_every": ("int", 3, 12),  # dwelling lots between streets
                   "dwelling": ("form", "courtyard_house"),
                   "court": ("int", 5, 11), "lot_width": ("band", 15, 22),
                   "lot_depth": ("band", 16, 24),
                   "front": ("form", "shop_house"), "front_storeys": ("band", 1, 3),
                   "front_depth": ("int", 10, 14),
                   "square_every": ("int", 0, 6)},
    },
    "rows": {
        "what": "narrow attached row houses fronting close lanes, back to back, with "
                "shop rows on the streets, wells/small squares where streets cross; a "
                "crowded commoners' grain",
        "params": {"lane_dir": ("choice", ["east_west", "north_south"]),
                   "lane_width": ("int", 2, 4), "street_width": ("int", 4, 8),
                   "street_every": ("int", 4, 20),
                   "dwelling": ("form", "row_house"), "storeys": ("band", 1, 3),
                   "lot_width": ("band", 5, 9), "lot_depth": ("band", 8, 13),
                   "front": ("form", "shop_house"), "front_storeys": ("band", 1, 3),
                   "front_depth": ("int", 9, 13),
                   "square_every": ("int", 0, 6)},
    },
    "estates": {
        "what": "large walled multi-court compounds (a gate hall in the compound wall, a "
                "front court, a main hall, side ranges, a rear residential court and a "
                "garden), on streets between them; an elite grain",
        "params": {"street_width": ("int", 5, 9), "lane_dir": ("choice",
                                                               ["east_west", "north_south"]),
                   "estate_width": ("band", 32, 60), "estate_depth": ("band", 44, 90),
                   "courts": ("int", 1, 3), "hall_storeys": ("band", 1, 2),
                   "garden": ("choice", ["rear", "side", "none"]),
                   "gate": ("choice", ["centre", "corner"])},
    },
    "fields": {
        "what": "farmland in fields along roads with farmsteads and small hamlets; an "
                "agrarian belt",
        "params": {"street_width": ("int", 3, 7), "field": ("band", 24, 64),
                   "farmstead_every": ("int", 2, 8), "lane_dir": ("choice",
                                                                  ["east_west",
                                                                   "north_south"])},
    },
    "clusters": {
        "what": "a small settlement's grain: houses gathered in short runs on their own "
                "lots, fronting lanes that follow the place's own outline (a lane every "
                "two lots' depth, concentric with the boundary) and spokes out from its "
                "centre; what no lot takes is left as gardens and yards. The dwelling "
                "is any dwelling form in the library; its lot is that form's own size",
        "params": {"dwelling": ("form", "a dwelling form"), "storeys": ("band", 1, 3),
                   "lot_width": ("band", 5, 14), "lot_depth": ("band", 6, 16),
                   "lane_width": ("int", 2, 5), "spokes": ("int", 0, 12),
                   "run": ("int", 1, 8)},
    },
    "open": {
        "what": "ground kept as landscape: parkland, lake shore, grove, gardens -- owned "
                "and finished, not built. `cover` may be a list, mixed piece by piece; "
                "`trees` is the crown of any grove",
        "params": {"cover": ("choice", ["grove", "garden", "field", "as_found"]),
                   "trees": ("choice", ["broad", "palm"])},
    },
    "composed": {
        "what": "land composed individually by the design's `composition`: each "
                "building, outdoor room and path placed by the design with its use, "
                "form and what it faces; land none of them takes is left as found",
        "params": {},
    },
}

#: The outdoor rooms a composition lays, and the area form each is built as. A kind is a
#: use of open ground; its form is the library's (`types/<form>.py`).
SPACE_KINDS = {"green": "green", "square": "square", "market": "market", "yard": "yard",
               "garden": "garden", "field": "field", "orchard": "grove",
               "grove": "grove", "pool": "pool", "plaza": "plaza"}
#: spaces people walk across to a door (paved or common ground)
SPACE_WALKED = ("green", "square", "market", "plaza", "yard")
#: the uses a composed building can be for; each is a form's declared `FUNCTION`
USES = ("dwelling", "work", "trade", "civic", "worship", "store", "defensive")
#: path ranks, and the street raster each is laid as
PATH_RANKS = {"road": 2, "lane": 1, "path": 1}

#: What a ring's `shade` may ask of its lanes. `lanes`: the houses fronting a lane of
#: the ring carry slatted beams out over it from their front walls, so a narrow lane is
#: roofed between the houses either side (a form that offers `shade: lane`).
SHADE = ("none", "lanes")

#: The ground policies a ring or the monument can ask for. `designground` realizes them.
GROUND = {
    "terrace": "one level for the ring (or per ward where `wards` say), its edges retained "
               "at the ring's walls; costs are cut/fill and retained face",
    "podium": "a raised platform `rise` blocks over the ring's level, retained on every "
              "side, with ramps/stairs on the axis",
    "graded": "the ground as found smoothed within `relief` of its own average, buildings "
              "on local pads; soft slopes, no retaining walls inside the ring",
    "preserve": "the ground as found kept; only building pads, roads and wall bands are "
                "made; fields follow the land",
}

MONUMENT_KINDS = ("forecourt", "gate_hall", "court", "hall", "garden", "bridge_court")

#: the landmark forms `cityresolve._landmarks` composes
LANDMARK_FORMS = ("market", "plaza", "square", "garden", "grove", "temple", "compound",
                  "park", "forecourt", "pool")

SCHEMA_DOC = """## The design record (`ethoslm.design/1`)

Write JSON. Only `schema`, `id`, `site`, `extent` and `rings` are required; a place with
no monument, landmarks, radials, gates or walls omits them. The example below is a
walled concentric capital only to show every field; it is not a template. Coordinates are never absolute: sizes are blocks, positions are bearings
(compass degrees clockwise from north, or words `north`, `south_east`, ...) and fractions
of the extent's radius measured from the centre. The compiler turns this into walls,
streets, blocks, lots, compounds and ground, and returns what it could not honour.

```
{
 "schema": "ethoslm.design/1",
 "id": "short-id",
 "lineage": "the accepted design's id, when this revises one (its unchanged choices kept)",
 "brief": {"silhouette": "...", "grain": "...", "monument": "...", "ground": "...",
           "scale": "...", "references": ["ref ids you looked at"]},
 "site": {"candidate": "site id from the list", "why": "..."},
 "extent": {"radius": 720, "why": "space demand and cost, see the table"},
 "boundary": {"shape": "circle" | "ellipse" | "superellipse" | "polygon",
              "aspect": 0.85, "rotation": 0, "n": 3, "points": [[x,z],...], "round": 0.1,
              "why": "..."},
 "axis": {"bearing": 180, "why": "the principal approach"},
 "gates": [{"bearing": 180, "rings": "all"}, {"bearing": 90, "rings": ["outer ring names"]}],
 "rings": [                                  // from the centre outward
   {"name": "palace", "outer": 0.15, "role": "monument",
    "wall": {"height": 14, "width": 3, "type": "wall" | "great_wall", "crown": "crenellated"},
    "ground": {"policy": "podium", "rise": 6}, "voice": "voice name"},
   {"name": "...", "outer": 0.34, "grain": "estates" | "courts" | "rows" | "fields" | "open",
    "params": {...grain parameters...},
    "wards": [{"from": 150, "to": 210, "grain": "...", "params": {...}, "why": "..."}],
    "wall": {...} | null, "ground": {"policy": "terrace" | "graded" | "preserve"},
    "roads": {"inner": 0 | width, "outer": 0 | width},   // ring roads along its edges
    "shade": "none" | "lanes",   // lanes roofed by the houses either side of them
    "voice": "...", "purpose": "..."}
 ],
 "radials": [{"bearing": 180, "width": 11, "role": "processional avenue"}],
 "landmarks": [{"name": "grand_market", "ring": "...", "bearing": 180, "at": 0.5,
                "form": "market" | "temple" | "plaza" | "garden" | "grove" | "square"
                        | "compound" | "park" | "forecourt" | "pool",
                "size": [w, d], "side": "east" | "west" | "on",
                "span": "ring",        // a forecourt the ring's whole depth, on a
                                       // cardinal bearing (its size is [width])
                "front": "north",      // a compound's gate side (default: toward the
                                       // centre); a lane is carried to the street
                "params": {...},       // compound: courts, hall_storeys, garden, gate,
                                       //   hall_use, hall_eaves, hall_platform, hall_roof
                                       // park: layout, pavilion (side, 0 for none)
                                       // forecourt: edge "grove" | "none", paving
                "why": "the purpose that explains the break in the fabric"}],
 "hierarchy": [{"greater": "royal_palace", "lesser": "upper_ring estates",
                "by": ["precinct", "silhouette", "approach", "height"],
                "why": "the interpretation read it asks for"}],
 "monument": {"name": "royal_palace", "ring": "palace", "enter": 180,
   // The sequence is laid at the sizes asked. If the ring cannot hold it the compiler
   // says so as a blocking finding naming the ring `outer` it needs: give it the land
   // (reallocating the rings outside it) or change the composition; it is never
   // shortened to fit. A hall may ask `tiers` (1-5 stacked eaves stepping in), `base`
   // (0-14, a tall faced podium under its stone tiers) and `entrance: "porch"`
   // (projecting gabled roofs over the entrance). `beside` sets a pair of courts,
   // each round its own form, either side of an element at the same depth:
   //   "beside": {"form": "round_altar" | "ceremonial_hall", "court": 57, "size": 45,
   //              "gap": 10, "params": {"tiers": 3, "top": "pavilion"}}
   // "band": "halls" keeps the paved axis as wide as the halls it joins, leaving the
   // land beside narrower courts to the wings (default: as wide as the widest element)
   "sequence": [{"kind": "forecourt", "depth": 24},
                {"kind": "gate_hall", "width": 29, "depth": 13, "storeys": 1, "eaves": 1,
                 "platform": 1, "roof": "hip_gable"},
                {"kind": "court", "depth": 50, "width": 80,
                 "flanks": {"form": "side_hall", "count": 2, "width": 19, "depth": 11}},
                {"kind": "hall", "role": "principal", "width": 51, "depth": 33,
                 "storeys": 3, "eaves": 2, "platform": 3, "roof": "hip"},
                {"kind": "court", "depth": 30},
                {"kind": "hall", "role": "rear", "width": 37, "depth": 19, ...},
                {"kind": "garden", "depth": 30}],
   "wings": {"grain": "estates", "params": {...}, "why": "residences flanking the axis"}},
 "palette": {"rings": {"ring name": "voice"}, "walls": "voice", "monument": "voice",
             "ground": "voice whose floor, footing and trim pave the streets and face
                        the retaining walls (omit: the library's stone and packed earth)",
             "compared": ["sheet paths you looked at"], "why": "..."},
 "decisions": {"model": ["..."], "compiler": ["..."]}
}
```
"""


COMPOSITION_DOC = """## The composition (`composition`, for land of grain `composed`)

Composed land is designed building by building. Positions are **local offsets in blocks
from the place's centre** (`[dx, dz]`: +x east, +z south), read off the site sheet's
grid; sizes are blocks. Nothing is packed for you: what you do not place stays as the
land is.

```
"composition": {
 "programme": [                    // what the place needs, and why, before any position
   {"use": "dwelling" | "work" | "trade" | "civic" | "worship" | "store",
    "what": "households farming the lower fields", "count": 8,
    "source": "request" | "inferred",   // request: a read of the sentence names it
    "reads": ["interpretation read ids it answers"], "why": "..."}],
 "spaces": [                       // outdoor rooms: the ground between the buildings
   {"id": "green", "kind": "green" | "square" | "market" | "yard" | "garden" | "field"
                            | "orchard" | "grove" | "pool" | "plaza",
    "at": [dx, dz], "size": [w, d], "params": {...the form's own...}, "why": "..."}],
 "groups": [                       // buildings that belong together, and why
   {"id": "green_row", "purpose": "homes facing the green", "around": "green"}],
 "buildings": [
   {"id": "smithy", "use": "work", "form": "workshop", "params": {"trade": "smithy"},
    "at": [dx, dz], "size": [w, d],   // the lot: the form's pad plus one column each side
    "faces": "south" | "<space id>" | "<path id>",   // the side its door is on
    "voice": "a palette for this building alone (omit: the ring's)",
    "group": "green_row", "why": "by the road in, where carts arrive"}],
 "paths": [                        // how people move: every door reaches one
   {"id": "street", "rank": "road" | "lane" | "path", "width": 3,
    "points": [[dx, dz], [dx, dz], ...],   // straight runs between the points
    "why": "..."}]
}
```

- A building's `use` is its form's declared function (the forms table says each); its
  `params` are the form's own, each within the form's range -- a choice you do not make
  is drawn by the form. The lot must hold the form's pad; `faces` names the side its
  door is on, or the space or path it addresses. A parameter you choose is probed on
  the lot you give: where the form would leave it out (an outshot, a storey) the design
  is handed back with a lot that delivers it.
- Every door reaches a path or a walked space (green, square, market, plaza, yard)
  within a few blocks of its front, and some path reaches the boundary. A lot never
  stands on a path, a space, water or another lot; a space may be crossed by a path.
- Spaces are laid at the level the ground allows their use (a square levelled, a field
  on a gentle slope); ground too steep or wet for a use is returned, not substituted.
- Groups and programme are what the inspection reads the place against: that the green
  is addressed by the buildings you said face it, that the working places stand where
  their work is, that each programme entry is carried by buildings of that use.
"""


def library_form(name) -> dict | None:
    """The declaration of a type in the library, or None. A grain's `form` parameters
    name any form on disk whose declaration fits the slot, not a fixed list."""
    import os
    from .pipeline.stages_plan import load_type
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        return None
    path = os.path.join(os.path.dirname(__file__), "..", "..", "types", f"{name}.py")
    if not os.path.exists(path):
        return None
    try:
        return load_type(path)
    except Exception:                               # noqa: BLE001 -- not a usable form
        return None


class DesignError(ValueError):
    """The design record is not one the compiler can read. The message names the field."""


def _num(v, where, lo=None, hi=None):
    try:
        f = float(v)
    except Exception as e:                          # noqa: BLE001
        raise DesignError(f"{where}: a number, not {v!r}") from e
    if lo is not None and f < lo or hi is not None and f > hi:
        raise DesignError(f"{where}: {f} is outside {lo}..{hi}")
    return f


def read(doc: dict) -> dict:
    """The design, checked and normalised. Raises `DesignError` naming the field."""
    if not isinstance(doc, dict):
        raise DesignError("the design is a JSON object")
    d = copy.deepcopy(doc)
    if d.get("schema") not in (None, SCHEMA):
        raise DesignError(f"schema is {SCHEMA!r}")
    d["schema"] = SCHEMA
    ext = d.get("extent") or {}
    d["extent"] = {**ext, "radius": int(_num(ext.get("radius"), "extent.radius", 48, 2048))}
    b = dict(d.get("boundary") or {"shape": "circle"})
    if b.get("shape", "circle") not in boundary.SHAPES:
        raise DesignError(f"boundary.shape is one of {boundary.SHAPES}")
    d["boundary"] = b
    ax = d.get("axis") or {"bearing": 180}
    d["axis"] = {**ax, "bearing": boundary.bearing(ax.get("bearing", 180))}
    rings = d.get("rings") or []
    if not rings:
        raise DesignError("rings: at least one ring (the place itself)")
    last = 0.0
    names = set()
    for i, r in enumerate(rings):
        w = f"rings[{i}]"
        if not r.get("name"):
            raise DesignError(f"{w}.name is required")
        if r["name"] in names:
            raise DesignError(f"{w}.name {r['name']!r} is used twice")
        names.add(r["name"])
        r["outer"] = _num(r.get("outer"), f"{w}.outer", 0.02, 1.0)
        if r["outer"] <= last:
            raise DesignError(f"{w}.outer {r['outer']} must exceed the ring inside it "
                              f"({last})")
        last = r["outer"]
        g = r.get("grain")
        if r.get("role") != "monument":
            if g not in GRAINS:
                raise DesignError(f"{w}.grain is one of {sorted(GRAINS)}, not {g!r}")
            _check_forms(g, r.get("params") or {}, w)
        if r.get("shade", "none") not in SHADE:
            raise DesignError(f"{w}.shade is one of {SHADE}")
        for j, wd in enumerate(r.get("wards") or []):
            if wd.get("grain") not in GRAINS:
                raise DesignError(f"{w}.wards[{j}].grain is one of {sorted(GRAINS)}")
            wd["from"] = boundary.bearing(wd.get("from", 0))
            wd["to"] = boundary.bearing(wd.get("to", 0))
        gp = (r.get("ground") or {}).get("policy", "terrace")
        if gp not in GROUND:
            raise DesignError(f"{w}.ground.policy is one of {sorted(GROUND)}")
        r["ground"] = {**(r.get("ground") or {}), "policy": gp}
        wall = r.get("wall")
        if wall:
            wall["height"] = int(_num(wall.get("height", 12), f"{w}.wall.height", 3, 48))
            wall["width"] = int(_num(wall.get("width", 3), f"{w}.wall.width", 1, 9))
    if abs(last - 1.0) > 1e-6:
        raise DesignError(f"the outermost ring's outer must be 1.0 (the boundary), not {last}")
    d["rings"] = rings
    gates = []
    for i, g in enumerate(d.get("gates") or [{"bearing": d["axis"]["bearing"],
                                              "rings": "all"}]):
        gates.append({**g, "bearing": boundary.bearing(g.get("bearing"))})
    d["gates"] = gates
    for i, rd in enumerate(d.get("radials") or []):
        rd["bearing"] = boundary.bearing(rd.get("bearing"))
        rd["width"] = int(_num(rd.get("width", 7), f"radials[{i}].width", 3, 21))
    d["radials"] = d.get("radials") or []
    mon = d.get("monument")
    if mon:
        if mon.get("ring") not in names:
            raise DesignError(f"monument.ring {mon.get('ring')!r} is not a ring")
        for j, s in enumerate(mon.get("sequence") or []):
            if s.get("kind") not in MONUMENT_KINDS:
                raise DesignError(f"monument.sequence[{j}].kind is one of {MONUMENT_KINDS}")
        mon["enter"] = boundary.bearing(mon.get("enter", d["axis"]["bearing"]))
    for i, lm in enumerate(d.get("landmarks") or []):
        if lm.get("ring") not in names:
            raise DesignError(f"landmarks[{i}].ring {lm.get('ring')!r} is not a ring")
        if lm.get("form", "plaza") not in LANDMARK_FORMS:
            raise DesignError(f"landmarks[{i}].form is one of {LANDMARK_FORMS}")
        lm["bearing"] = boundary.bearing(lm.get("bearing", 0))
    d["landmarks"] = d.get("landmarks") or []
    composed = any(r.get("grain") == "composed" for r in rings) or any(
        w.get("grain") == "composed" for r in rings for w in r.get("wards") or [])
    if d.get("composition") is not None or composed:
        if not composed:
            raise DesignError("composition: a composition needs land of grain `composed` "
                              "(a ring or a ward) to stand on")
        d["composition"] = read_composition(d.get("composition"))
    return d


_CARDINAL = ("north", "south", "east", "west")


def _pair(v, where, lo=None, hi=None, integer=False) -> list:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise DesignError(f"{where}: two numbers [x, z], not {v!r}")
    got = [_num(v[0], f"{where}[0]", lo, hi), _num(v[1], f"{where}[1]", lo, hi)]
    return [int(round(g)) for g in got] if integer else got


def check_params(form: str, decl: dict, params: dict, where: str) -> dict:
    """`params` held to the form's own declared `PARAMS`: every key the form's, every
    value in its range. Raises `DesignError` naming the key and the range."""
    decl_p = decl.get("params") or {}
    out = {}
    for k, v in (params or {}).items():
        spec = decl_p.get(k)
        if spec is None:
            raise DesignError(f"{where}.params.{k}: `{form}` has no parameter {k!r} "
                              f"(it has {sorted(decl_p) or 'none'})")
        if spec[0] == "int":
            n = int(_num(v, f"{where}.params.{k}"))
            if not int(spec[1]) <= n <= int(spec[2]):
                raise DesignError(f"{where}.params.{k}: `{form}` takes {k} "
                                  f"{spec[1]}..{spec[2]}, not {v!r}")
            out[k] = n
        elif spec[0] == "choice":
            if v not in spec[1]:
                raise DesignError(f"{where}.params.{k}: `{form}` takes {k} one of "
                                  f"{list(spec[1])}, not {v!r}")
            out[k] = v
        else:
            out[k] = v
    return out


def read_composition(comp) -> dict:
    """A composition, checked and normalised: every element named once, every form a
    library form of the use it is asked for, every parameter the form's own. Geometry
    (fit, overlap, access, ground) is the compiler's and comes back as findings."""
    if not isinstance(comp, dict):
        raise DesignError("composition is an object: programme, spaces, groups, "
                          "buildings, paths")
    c = copy.deepcopy(comp)
    ids: dict = {}

    def claim(i, where):
        if not isinstance(i, str) or not i:
            raise DesignError(f"{where}.id is a short name")
        if i in ids:
            raise DesignError(f"{where}.id {i!r} is also {ids[i]}")
        ids[i] = where
    for i, p in enumerate(c.get("programme") or []):
        w = f"composition.programme[{i}]"
        if p.get("use") not in USES:
            raise DesignError(f"{w}.use is one of {USES}, not {p.get('use')!r}")
        if p.get("source", "inferred") not in ("request", "inferred"):
            raise DesignError(f"{w}.source is `request` or `inferred`")
        p["source"] = p.get("source", "inferred")
        p["count"] = int(_num(p.get("count", 1), f"{w}.count", 0, 400))
    for i, s in enumerate(c.get("spaces") or []):
        w = f"composition.spaces[{i}]"
        claim(s.get("id"), w)
        if s.get("kind") not in SPACE_KINDS:
            raise DesignError(f"{w}.kind is one of {sorted(SPACE_KINDS)}, not "
                              f"{s.get('kind')!r}")
        s["at"] = _pair(s.get("at"), f"{w}.at", -2048, 2048)
        s["size"] = _pair(s.get("size"), f"{w}.size", 3, 256, integer=True)
        form = SPACE_KINDS[s["kind"]]
        decl = library_form(form)
        if decl is None:
            raise DesignError(f"{w}.kind {s['kind']!r}: its form `{form}` is not in the "
                              f"library")
        s["params"] = check_params(form, decl, s.get("params") or {}, w)
    for i, g in enumerate(c.get("groups") or []):
        claim(g.get("id"), f"composition.groups[{i}]")
    for i, p in enumerate(c.get("paths") or []):
        w = f"composition.paths[{i}]"
        claim(p.get("id"), w)
        if p.get("rank", "lane") not in PATH_RANKS:
            raise DesignError(f"{w}.rank is one of {sorted(PATH_RANKS)}")
        p["rank"] = p.get("rank", "lane")
        p["width"] = int(_num(p.get("width", 3 if p["rank"] != "path" else 2),
                              f"{w}.width", 1, 9))
        pts = p.get("points") or []
        if len(pts) < 2:
            raise DesignError(f"{w}.points: at least two [dx, dz]")
        p["points"] = [_pair(q, f"{w}.points[{j}]", -2048, 2048) for j, q in
                       enumerate(pts)]
    groups = {g["id"] for g in c.get("groups") or []}
    for i, b in enumerate(c.get("buildings") or []):
        w = f"composition.buildings[{i}]"
        claim(b.get("id"), w)
        decl = library_form(b.get("form"))
        if decl is None or decl.get("kind") != "plot":
            raise DesignError(f"{w}.form: {b.get('form')!r} is not a building form in the "
                              f"library (types/<name>.py)")
        fn = decl.get("function")
        if b.get("use") not in USES:
            raise DesignError(f"{w}.use is one of {USES}, not {b.get('use')!r}")
        if fn != b["use"]:
            raise DesignError(f"{w}: `{b['form']}` is a "
                              f"{fn + ' form' if fn else 'form with no declared use'}, "
                              f"not a {b['use']} building; choose a form whose function is "
                              f"{b['use']} (the forms table), or author one")
        b["params"] = check_params(b["form"], decl, b.get("params") or {}, w)
        b["at"] = _pair(b.get("at"), f"{w}.at", -2048, 2048)
        if b.get("size") is not None:
            b["size"] = _pair(b["size"], f"{w}.size", 3, 128, integer=True)
        if b.get("group") is not None and b["group"] not in groups:
            raise DesignError(f"{w}.group {b['group']!r} is not a group")
        if b.get("voice") is not None:
            from . import voices
            if b["voice"] not in voices.names():
                raise DesignError(f"{w}.voice {b['voice']!r} is not a voice in the library")
    for i, b in enumerate(c.get("buildings") or []):
        f = b.get("faces")
        if f is not None and f not in _CARDINAL and f not in ids:
            raise DesignError(f"composition.buildings[{i}].faces: a side "
                              f"({', '.join(_CARDINAL)}) or a space or path id, not {f!r}")
    for i, g in enumerate(c.get("groups") or []):
        a = g.get("around")
        if a is not None and a not in ids:
            raise DesignError(f"composition.groups[{i}].around {a!r} names nothing")
    if not (c.get("buildings") or c.get("spaces")):
        raise DesignError("composition: nothing is placed (no buildings, no spaces)")
    for k in ("programme", "spaces", "groups", "buildings", "paths"):
        c[k] = c.get(k) or []
    return c


def _check_forms(grain: str, params: dict, where: str) -> None:
    """Every `form` parameter names a plot form in the library (a dwelling where the slot
    is one): refused by name otherwise, never swapped for another."""
    for k, spec in GRAINS[grain]["params"].items():
        if spec[0] != "form" or k not in params:
            continue
        d = library_form(params[k])
        if d is None or d.get("kind") != "plot":
            raise DesignError(f"{where}.params.{k}: {params[k]!r} is not a plot form in the "
                              f"library (types/<name>.py)")
        if k == "dwelling" and d.get("function") not in (None, "dwelling"):
            raise DesignError(f"{where}.params.{k}: {params[k]!r} is a "
                              f"{d.get('function')}, not a dwelling")


def ring_of(design: dict, name: str) -> dict | None:
    for r in design.get("rings") or []:
        if r.get("name") == name:
            return r
    return None


def digest(design: dict) -> str:
    """A stable identity of the design's content (not its prose)."""
    import hashlib
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:16]
