"""Write the planner brief and the per-pass builder briefs.

API doc + terrain briefing + style seed + request. **the circulation network**, because
it is now in the world before any builder runs, and a builder that cannot see it cannot
front onto it.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ethoslm.buildlib import API_DOC, CRAFT_VILLAGE
from ethoslm import settlement, styles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
s = settlement.site_info()
X, Z = s["origin"]
S = s["size"]


def grids() -> str:
    mean = "\n".join("    " + " ".join(f"{v:3d}" for v in row) for row in s["mean_grid"])
    rough = "\n".join("    " + " ".join(f"{v:3d}" for v in row) for row in s["roughness_grid"])
    surf = ", ".join(f"{k} ({v})" for k, v in sorted(s["surface_blocks"].items(),
                                                     key=lambda kv: -kv[1]))
    return f"""The site is x {X} to {X + S - 1}, z {Z} to {Z + S - 1} — {S} by {S} blocks.
Surface height runs from y={s['stats']['min']} to y={s['stats']['max']}. That is
{s['stats']['relief']} blocks of relief. This is broken country, not a meadow.

Mean surface height per 16-block cell. Rows run north to south (increasing z),
columns west to east (increasing x):

{mean}

Roughness within each of those cells (highest minus lowest block in the cell). A cell
above about 25 contains a cliff or a steep face; you cannot put a building flat on it:

{rough}

Surface materials: {surf}.

get_height(x, z) gives the exact surface at any column.
"""


VOICE = os.environ.get("ETHOSLM_VOICE", "drystone_and_thatch")


def api_sections(*headers: str) -> str:
    """Sections of the one API doc, selected by their `## ` headers -- shared by
    reference, so the massing brief cannot drift from the doc builders read."""
    parts = API_DOC.split("\n## ")
    keep = [parts[0]]
    for p in parts[1:]:
        if p.split("\n", 1)[0].strip().startswith(headers):
            keep.append(p)
    return "\n## ".join(keep)


#: When the planner is asked for the outdoor rooms first (`spaces`, stage 1). Additive:
#: nothing that reads plan.json breaks if the key is absent, and the schema is kept
#: deliberately this small -- the value is in being asked for the space first, not in
#: the record being rich.
SPACES = """
**Before you place a single footprint**, decide the outdoor rooms: a village is defined
by the ground between the buildings. Add to the plan a top-level key:

  "spaces": [
    {"id": "green", "kind": "green|street|yard|court",
     "x0": 0, "z0": 0, "x1": 0, "z1": 0,
     "enclosed_by": ["moot_hall", "east_row", "smithy"]}
  ]

A space is outdoor ground that stays unbuilt, and the structures in `enclosed_by` are
the ones whose walls form its edge. Derive the footprints *from* the spaces -- a house
fronts the street because the street needed a second side, not because a rectangle was
free there.
"""


#: **the plan is a tree**, and this is what the planner is asked for when `$ETHOSLM_TREE`
#: is set. It replaces the flat `structures` list in the output schema and nothing else
#: in the brief, because everything else in it -- the ground, the voice, the circulation
#: pass, "no two neighbours the same" -- is the same job. Two things are new and both
#: are the point. A part is one of **four kinds**: a plot is a building's ground, an
#: edge is a wall's, a point is a gate's, an area is a square's. And a part is built by
#: **naming a type**, not by describing a building: every part of this place is an
#: instance of a committed type, so what the planner decides is what goes where and
#: which type it is, and never how it is made.
TREE_SCHEMA = '''  "parts": [
    {
      "kind": "district",
      "name": "short_slug",
      "notes": "what this district is and why it is here",
      "children": [
        {
          "kind": "quarter",
          "name": "short_slug",
          "notes": "what this quarter is for",
          "children": [
            {"kind": "plot", "name": "short_slug", "type": "<a type below>",
             "seed": 1, "params": {"storeys": 2},
             "x0": 0, "z0": 0, "x1": 0, "z1": 0,
             "notes": "what this one is for and how it sits on its ground"},

            {"kind": "edge", "name": "short_slug", "type": "<a type below>",
             "seed": 1, "params": {},
             "path": [[0, 0], [0, 0], [0, 0]], "width": 1,
             "notes": "a wall, a terrace edge, a quay: a polyline of corners. Every
                       segment runs along x or along z; a corner is a vertex"},

            {"kind": "point", "name": "short_slug", "type": "<a type below>",
             "seed": 1, "params": {},
             "at": [0, 0], "facing": "north|south|east|west",
             "notes": "a gate, a well head, a marker: one cell and the way it faces"},

            {"kind": "area", "name": "short_slug", "type": "<a type below>",
             "seed": 1, "params": {},
             "x0": 0, "z0": 0, "x1": 0, "z1": 0,
             "notes": "a square, a market, a yard: ground that stays open"}
          ]
        }
      ]
    }
  ]
}

A **group** (`district`, `quarter`) has `children` and no geometry. A **leaf** (`plot`,
`edge`, `point`, `area`) has geometry, names a `type` and a `seed`, and is built by
instantiating that type: there is no builder in this run and no part is written by hand.
Nest as deeply as the place needs -- that is what a tree is for -- and put every leaf
inside a quarter so the waves have something to be.

`seed` is what makes two instances of one type different buildings. **No two adjacent
parts may share both `type` and `seed`.**

Geometry rules: every leaf lies inside the site; no two plots overlap; an edge's
segments run along one axis at a time; leave 5 blocks between footprints for the lanes.

## The types you may use

Every part is an instance of one of these. Each was written and checked on its own, and
`params` are the only things about it you may vary.

'''


#: The plot a planner draws is **not** the ground a building gets: the library insets it
#: on every side before the type is called. So the table gives the plot, the arithmetic
#: is done here, and the plan is checked against it before a block is placed.
NEEDS_CARD = '''

### What each type needs from the ground

**The plot you draw is not the ground the building gets.** The library prepares every
part before its type is called: it insets the plot, sounds the bed, and lays a deck, a
platform or a plinth. The sizes below are the **plot**, with that inset already added,
so draw at least the smallest and at most the largest.

{table}

A plan is checked against this before anything is built, and a part outside it is
refused rather than built into nothing. If a type you want needs more ground than you
have, use a different type or make the plot bigger -- do not shrink the plot and hope.
'''


def types_card(names=None) -> str:
    """The types a plan may name, with their kind and their `PARAMS`. A4.

        Read off the committed files rather than restated here, because a second copy of a
        type's parameters is a second chance for a plan to ask for one that does not exist.
        A type's kind is what its `KIND` says, and a plot where it does not say -- which is
        every type written before A3.
        
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "src"))
    from ethoslm import pipeline as _pipeline
    d = os.path.join(root, "types")
    out = ["| type | kind | form | params |", "|---|---|---|---|"]
    decls = {}
    for f in sorted(os.listdir(d)):
        name = f[:-3]
        if not f.endswith(".py") or (names is not None and name not in names) \
                or (names is None and name.startswith("_")):
            continue
        decl = _pipeline.load_type(os.path.join(d, f))
        decls[name] = decl
        kind = decl["kind"] + (", passage" if decl["passage"] else "")
        params = ", ".join(
            f"{k}: " + (f"{v[1]}..{v[2]}" if v[0] == "int" else "|".join(map(str, v[1])))
            for k, v in sorted(decl["params"].items())) or "none"
        out.append(f"| `{name}` | {kind} | {decl['form']} | {params} |")
    # A2: and what each one needs from the ground, read off the same files.
    return "\n".join(out) + NEEDS_CARD.format(table=_pipeline.needs_table(decls))


PLANNER = f"""# Plan a settlement

## The place

{grids()}

{styles.voice_card(VOICE, s.get("surface_blocks"))}

## What you are doing

People are going to live here. Decide what gets built, and where.

You are not building it. You are producing the plan that later programs will build from,
one structure at a time. The reason for the split is arithmetic, not taste: a region this
size is millions of blocks and a dozen structures, and that does not fit in one program.

**What happens to your plan first.** Before any structure is built, a circulation pass
routes a walkable network between your footprints and builds it into the ground — cut,
filled, stepped, retained. Every later structure then fronts onto that network, and each
one is given a reserved threshold where its lane meets its door. So:

- Your footprints decide **where people need to go**. The routes decide **how**.
- Leave room between footprints for the lanes: 5 blocks is a minimum, more where the
  ground is steep, because a route across a rise needs length to climb it.
- A footprint on a shelf a router cannot reach is a structure nobody will be able to
  walk to. Look at the roughness grid before you place one.
- Say which structure is the **centre**. The routes converge there and a court is built
  around it.

## What matters

The hard part here is not what the buildings look like. It is that this is severe ground
and the result has to read as **one place that grew here**, rather than a set of objects
each dropped somewhere flat.

Think about where people would actually settle on this terrain and why — water, shelter,
level ground, defensibility, the route through. Think about what is at the centre and
what is at the edge.

Decide the size of the settlement yourself. Decide what kinds of structure it needs. Say
in `kind` what each one is *for* -- hall, byre, smithy, croft, gate, row of houses -- in
plain words. Later passes turn that into how the building is shaped, so "workshop for the
potter" tells them more than "building 4".

## Output

Write a single JSON file to {settlement.STATE}/plan.json with this shape:

{{
  "intent":  "a few sentences: what this settlement is, why it is here, what should
              make it feel like one place. Later passes read this and nothing else
              about your reasoning, so put what they need in it.",
  "centre":  "the id of the structure the town is centred on",
  "voice": "{VOICE}",
  "palette": {{ "role": "material" }},   // the voice's palette, with any change you make
                                         // to it stated here. Every pass builds from
                                         // this and a linter measures what they place
                                         // against it, so it is a contract, not a hint.
  "circulation_material": "one material family for the lanes, e.g. cobblestone,
              red_sandstone, mud_brick, deepslate_brick, stone_brick, sandstone",
  "structures": [
    {{
      "id": "short_slug",
      "kind": "what it is",
      "x0": 0, "z0": 0, "x1": 0, "z1": 0,
      "form": "block | longhouse | l-plan | tower | hall-and-wing | outshot",
      "roof": "one of the roof forms named in the voice above",
      "notes": "what this one is for, how it should sit on its ground, and how it
                relates to its neighbours"
    }}
  ]
}}

Footprints must lie inside the site and must not overlap each other.

**`form` and `roof` are yours to decide and they are the shape of the settlement.** A
person walked the last two towns this system built and said they "look the same in a
different font": the palette changed and the buildings did not, because form was never
decided anywhere. It is decided here now. Choose each one from what the building is for
and how its ground falls — a longhouse down a contour, a tower where the site wants a
marker, hall-and-wing for the largest thing in the place, an outshot on a working
building that needs a lean-to. **No two structures whose footprints are neighbours may
share both `form` and `roof`.** Two the same at opposite ends of the town is a village;
two the same side by side is a copy.

Use the Write tool. Output only the file; reply with the number of structures.
"""

BUILDER = """# Build part of a settlement

{api}

## The place

{grids}

## The plan

This settlement was planned in an earlier pass. Here it is in full:

{plan}

{voice}

## The circulation network — already built, already in the ground

{network}

## Already standing

{built}

Everything listed above is already in the world. You can read it with get_height and
get_block — the ground under you may already be someone's wall, road or terrace.

## Plot registry

    reserve(x0, z0, x1, z1, label) -> bool
        Claim ground. Returns False if it overlaps a plot someone already claimed, in
        which case nothing was claimed and you must adjust.
    plots() -> list of dicts with x0, z0, x1, z1, label

Claim before you build.

## Your assignment this pass

{assignment}

## What matters

Build these so they belong to the same settlement as what is already there, and to this
ground. Match the plan's palette. Meet the terrain honestly — a flat pad cut into a
hillside is the most obvious possible failure. Where a structure needs level ground,
terrace it and feather the edge; where it can step down the slope instead, step it.

Every signature named below is written out in the API section at the top of this brief,
which is the only place they are written out.

**Build the shell with `building()`.** One call lays the plinth, the walls, the floors,
the openings, the door, the way in, the roof, the chimney and the stairs of one
structure, in the order that works, and it is given the form and the roof this
structure is. Give it the footprint, the storey count, the roof form and the palette by
role, and read what it returns: `rooms` is the interior rectangle and floor level of
every room it made, and those rooms are what you furnish. It refuses rather than
building something wrong, and a refusal names what it could not do — read it and move
the thing, do not work around it by hand.

What is left for you is the part a library cannot do: **which forms go where, how they
sit together, and what is inside them.** Compose the wave, then furnish it with
`fitting()` and `dais()` — a room with a focal point and a story, not the same four
objects in every building. Anything the library will not give you, place by hand
afterwards; `building()` never stops you writing blocks.

**Front onto the lane.** Your building's door goes on the threshold reserved for it, at
the floor height given, facing the way stated. That threshold is ground the circulation
pass owns: do not build on it, do not bury it, do not fence it off. If you must move a
door, move it to somewhere else on the network — and check it with `check_door`.

Call it on **every door you place**, after you have placed the walls around it. It
answers with everything your program has decided so far, not just what was in the world
when you started. If `ok` is False the door is one nobody can walk to; move it and check
again. Do not finish a pass with a door that fails.

**Lay the way in with `approach(label)`.** It is the counterpart of `steps()`: from your
doorway to the lane, following the ground, a slab where it rises half a block and a tread
where it rises a whole one, two columns wide, headroom cut. Call it once, after the door
and the ground around it are in. Where the door is already walkable it lays nothing and
tells you so, which costs a second and is the answer you want. Where it is not, this is
the difference between a building and a building nobody can get into: the last nine
rounds shipped raised floors, porches at deck height and doors a block above their own
approach, and every one of them was one call away from being enterable.

**Lay the ground floor at the doorstep, not at a height you chose** —
`floor_from_threshold(label)`. `floor_y` is the course your floor blocks go in and
`stand_y` is the cell a person occupies standing on it. Use them. A floor laid one block
above its own doorway sill is a room you cannot step into — your feet are on the sill and
the floor is at your knees — and it is invisible from outside, so nobody notices until
they walk in.

**Check that a person can walk about inside**, with `check_walkable(label)`, before you
finish. Walking only, no jumping, from your own doorway. `ok` is False when a room cannot
be walked into at all — fix those, they are errors. A `fraction` below 1 is a *report*: a
raised sleeping platform or a dais is architecture and nothing here says otherwise, but
if a fraction is low and you did not mean a platform, something is in the way — usually
a floor above its own sill, or a `fitting()` across the only route.

Call it after your floors, stairs and fittings are in, the way you call `check_door`
after the walls around a door. It answers with everything your program has decided so
far. Lanes have always been held to a walk-only standard — that is why the router
builds a stair for every one-block rise. Hold your interiors to the same one: a person
who cannot jump should be able to walk in at the door and reach the rooms.

**Check that your build holds together**, with `check_attached()`, before you finish. It
answers over everything you have placed so far. Any piece it lists is attached to
nothing — a roof course one block too high, a lantern hung under a beam that is not
there, a chimney that starts above the ridge. Eaves, jetties and balconies are *not*
listed, because they are part of the mass that reaches the ground. If it lists anything,
fix it before you finish.

Do not sever the lane either: if you fill or cut ground the network runs across, you
have cut the town in half, and that is checked after your pass runs.

You may depart from the plan's footprint by a few blocks if the ground demands it. You
may not build somewhere else entirely.

Return one fenced ```python block and nothing else.
"""


def _local_map(net, rect, ids, radius=14):
    """A small plan of the lanes around one footprint. Text, because it is the only
    representation a model reads without a renderer."""
    x0, z0, x1, z1 = rect
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    ax0, ax1 = cx - radius, cx + radius
    az0, az1 = cz - radius, cz + radius
    th = {(t.x, t.z): t for t in net.thresholds}
    rows = []
    for z in range(az0, az1 + 1):
        row = []
        for x in range(ax0, ax1 + 1):
            if (x, z) in th:
                row.append("D")
            elif (x, z) in net.cells:
                r = net.cells[(x, z)]["rank"]
                row.append("+" if r == 0 else ("=" if r <= 1 else "-"))
            elif x0 <= x <= x1 and z0 <= z <= z1:
                row.append("#")
            else:
                row.append(".")
        rows.append(f"  z={z:>6}  " + "".join(row))
    head = f"            x={ax0} .. {ax1}"
    return head + "\n" + "\n".join(rows)


def network_brief(ids: list) -> str:
    net = settlement.load_network()
    if not net:
        return "None was built for this settlement. Site your doors by judgement."
    plan = json.load(open(os.path.join(settlement.STATE, "plan.json")))
    rects = {p["id"]: (min(p["x0"], p["x1"]), min(p["z0"], p["z1"]),
                       max(p["x0"], p["x1"]), max(p["z0"], p["z1"]))
             for p in plan["structures"]}
    out = [f"{len(net.cells)} cells of lane are already laid across this site, in "
           f"{net.notes.get('material', 'stone')}. It is walkable end to end without "
           f"jumping: every rise of one block is a stair, and the routes were chosen to "
           f"avoid grades no one can climb. Buildings front onto it.\n",
           "You can query it while your program runs: threshold(), nearest_lane(), "
           "check_door(), approach() and check_walkable() are all written out in the "
           "API section above.\n",
           "The lane is walk-only and so are your interiors: check_walkable answers "
           "the same question inside the building that check_door answers outside it, "
           "and approach() is what lays the way to a door that fails either.\n"]
    for sid in ids:
        t = net.threshold(sid)
        if not t:
            out.append(f"### {sid}\n\nNo threshold was reserved — no route reached this "
                       f"footprint. Put its door where nearest_lane() says the network "
                       f"is, and check it.\n")
            continue
        out.append(
            f"### {sid}\n\n"
            f"Reserved threshold: the lane cell ({t.x}, {t.y}, {t.z}). You arrive on it "
            f"walking **{t.facing}**.\n"
            f"Put the door leaf at ({t.door[0]}, {t.door[1]}, {t.door[2]}), "
            f"`facing={t.facing}`. The doorstep is already laid: the floor block there "
            f"is at y={t.y} and a person stands in the cell at y={t.y + 1}, so your "
            f"ground floor is the course at y={t.y} and its walls start at y={t.y + 1}."
            f"\n\n"
            f"Lanes around this plot — `#` your footprint, `=` main way, `-` lane, "
            f"`+` court, `D` your threshold:\n\n```\n"
            + _local_map(net, rects[sid], ids) + "\n```\n")
    return "\n".join(out)


def plan_brief(ids: list, near: int = 48) -> str:
    """The plan, as much of it as this pass can act on.

        Dumping plan.json whole made it 2,032 words of a 4,625-word brief -- 44% of what a
        builder reads, most of it notes on structures a hundred blocks away that it is not
        building. The intent is what makes the town cohere and stays in full; a neighbour's
        notes matter because you can see it from your plot; the far side of the site is a
        line each.
        
    """
    plan = json.load(open(os.path.join(settlement.STATE, "plan.json")))
    mine = [p for p in plan["structures"] if p["id"] in ids]

    def centre_of(p):
        return ((p["x0"] + p["x1"]) / 2, (p["z0"] + p["z1"]) / 2)

    out = ["**What this settlement is** (the planner's words, and the only account of "
           "its reasoning any pass gets):", "", plan.get("intent", ""), ""]
    out += [f"**Centre:** {plan.get('centre', 'not stated')}. "
            f"**Lanes:** {plan.get('circulation_material', 'stone')}.", "",
            "**Palette held across every structure:**", ""]
    for role, mat in (plan.get("palette") or {}).items():
        out.append(f"  - {role}: {mat}")
    near_ids, far = [], []
    for p in plan["structures"]:
        if p["id"] in ids:
            continue
        d = min(max(abs(centre_of(p)[0] - centre_of(m)[0]),
                    abs(centre_of(p)[1] - centre_of(m)[1])) for m in mine)
        (near_ids if d <= near else far).append(p)
    out += ["", "**Your neighbours** — near enough to be seen from your plots, so they "
            "have to belong to the same place:", ""]
    for p in near_ids:
        out.append(f"  - **{p['id']}** ({p['kind']}): x {p['x0']}..{p['x1']}, "
                   f"z {p['z0']}..{p['z1']}\n{form_line(p)}    {p['notes']}")
    if far:
        out += ["", "**Elsewhere in the settlement**, for context only:", "",
                "  " + "; ".join(f"{p['id']} ({p['kind']})" for p in far)]
    return "\n".join(out)


def spaces_brief(ids: list) -> str:
    """The outdoor rooms this pass's walls form the edge of.

        `SPACES` above is addressed to the planner: it asks for a top-level key. A builder
        handed that text is being told to edit a JSON file it is not writing, which is what
        step 3's `+spaces` arm actually did. `plan_brief` meanwhile does not mention the
        planner's spaces at all, so until now nothing downstream of the planner has ever
        seen them -- the outdoor rooms were designed and then thrown away.

        E3 killed spaces-as-plat and said that if the idea returned it would return as
        terrain-following outdoor rooms rather than as a street grid. This is that, at brief
        level: the space comes first, and the footprint is derived from the edge it has to
        make.
        
    """
    plan = json.load(open(os.path.join(settlement.STATE, "plan.json")))
    spaces = plan.get("spaces") or []
    mine = [s for s in spaces if set(s.get("enclosed_by") or []) & set(ids)]
    if not mine:
        return ""
    out = ["## The outdoor rooms come first", "",
           "A settlement is defined by the ground *between* its buildings, and on this "
           "ground the outdoor room is the only level thing there is. The planner "
           "decided these before it placed a footprint, and your walls are what make "
           "their edges:", ""]
    for s in mine:
        others = [w for w in (s.get("enclosed_by") or []) if w not in ids]
        out.append(f"  - **{s['id']}** ({s.get('kind', 'space')}): "
                   f"x {s['x0']}..{s['x1']}, z {s['z0']}..{s['z1']}"
                   + (f" -- its other edges are {', '.join(others)}" if others else ""))
    out += ["", "Decide the shape of these before you decide a single footprint, and "
            "then derive the footprints from them. This ground stays unbuilt: a "
            "landing, a terrace or a court is a room, and a wall that wanders into it "
            "has taken the room away. Where a space is a stair or a street, it climbs "
            "-- follow it, do not flatten it.", ""]
    return "\n".join(out)


def voice_brief(ids: list, fittings: bool = True) -> str:
    """The settlement's voice, palette and per-structure registers -- the tuned
    section both the massing and the elaboration brief read. Fittings are the one
    part massing has no use for."""
    plan_json = json.load(open(os.path.join(settlement.STATE, "plan.json")))
    voice = styles.voice_card(plan_json.get("voice", VOICE),
                              s.get("surface_blocks"))
    if plan_json.get("palette"):
        voice += ("\nThe planner set this settlement's palette to:\n\n"
                  + "\n".join(f"    {k:9} {v}" for k, v in plan_json["palette"].items())
                  + "\n\nWhere that differs from the voice above, the planner wins.\n")
    regs, fits = [], []
    for st in plan_json["structures"]:
        if st["id"] in ids:
            regs.append(f"  - **{st['id']}** ({st['kind']}) — "
                        + (f"{st['form']} under a {st['roof']} roof. "
                           if st.get("form") and st.get("roof") else "")
                        + styles.register_for(st["kind"]))
            f = styles.fittings_for(st["kind"])
            if f:
                fits.append(f"  - **{st['id']}**: {f}")
    if regs:
        voice += ("\n### What your structures are, and what that makes them\n\n"
                  "Same materials as everything else in the settlement. The variety comes "
                  "from form following purpose, not from new materials:\n\n"
                  + "\n".join(regs) + "\n")
    if fits and fittings:
        # no actual interior". The cause was that nothing told a builder what is *in* a
        # building and the library had no furniture in it. Both are fixed; this is the
        # half that reaches the builder.
        voice += ("\n### What is inside them\n\n"
                  "Build the inside as well as the outside. `fitting(kind, x, y, z, "
                  "facing)` places equipment the way it is really built — it owns "
                  "orientation and clearance, you decide where things go. Rooms big "
                  "enough to furnish and left bare are reported:\n\n"
                  + "\n".join(fits) + "\n")
    return voice


def builder_brief(assignment: str, ids: list) -> str:
    plots = settlement.load_plots()
    built = ("\n".join(f"  - {p['label']}: x {p['x0']}..{p['x1']}, z {p['z0']}..{p['z1']}"
                       for p in plots) if plots else "  (nothing yet — you are first)")
    return BUILDER.format(api=API_DOC + CRAFT_VILLAGE, grids=grids(),
                          plan=plan_brief(ids), built=built, voice=voice_brief(ids),
                          network=network_brief(ids), assignment=assignment)


MASSING = """# Mass part of a settlement

{api}

## The place

{grids}

## The plan

This settlement was planned in an earlier pass. Here it is in full:

{plan}

{voice}

## Your assignment this pass

{assignment}

## What a massing is

You are not building these structures. You are deciding their **masses** — the
decisions that read from thirty blocks away — as a short program that a later pass
will elaborate into the finished building. About 60 lines of `place_cuboid` and
`roof`, with `line` or `cylinder` only where a form genuinely needs one. Decide:

- the footprint each mass actually takes on its ground, within its planned rectangle
- storey heights and where the eaves line sits
- roof form, axis and pitch — this is the silhouette, so it is most of the judgement
- how each mass meets the sloped ground: cut into the bank, stood on a plinth,
  stepped down the fall. A flat pad on a hillside is the most obvious possible failure.

Use **one material for everything: `stone_bricks`.** The massing is judged as a grey
mass — its silhouette against the ground — so materials, trim, openings, glazing,
interiors, furniture and paths do not exist at this stage. They belong to the pass
that elaborates this program, and anything you spend on them here is spent twice.

Return one fenced ```python block and nothing else.
"""

ELABORATE = """## The massing — already decided, already judged

The program below is this pass's **massing**: its footprints, heights and roof forms
were rendered and selected before you were called, and the images attached are an
isometric and a front elevation of exactly what it builds. **Elaborate this program;
do not redesign it.** Keep its occupancy — a conformance check reports where the
built mass departs from the massing — and depart only where building it well demands
it: a thicker wall, a buttress, a porch. Replace its placeholder `stone_bricks` with
the settlement's palette. Add what a massing deliberately has none of: openings,
trim, glazing, interiors, the way the wall reads at twenty blocks and at five.

```python
{massing_program}
```

"""


def massing_brief(assignment: str, ids: list) -> str:
    """The massing brief: site, plan, voice, registers, and the shapes part of the
    API doc. Nothing about materials in depth, fittings, glazing or detail."""
    return MASSING.format(
        api=api_sections("Reading the world", "Placing blocks", "Shapes", "Roofs",
                         "Coordinates", "Rules"),
        grids=grids(), plan=plan_brief(ids), voice=voice_brief(ids, fittings=False),
        assignment=assignment)


def elaboration_brief(assignment: str, ids: list,
                      massing_program: str = "{massing_program}") -> str:
    """The elaboration brief: the massing program and its previews, in front of the
    full builder brief. `massing_program` defaults to a placeholder so the prompt can
    be written before the massing has been selected; stages.run fills it."""
    return (ELABORATE.format(massing_program=massing_program)
            + builder_brief(assignment, ids))


def form_line(s: dict) -> str:
    """A structure's form and roof, stated as a fact about the building.

        Not as an instruction about variety: a builder told "make these different from each
        other" spends its reply on the instruction, and a builder told "this one is a
        longhouse under a shallow gable" builds a longhouse. The variety is the planner's
        decision and it has already been made by the time this is read.
        
    """
    parts = [f"**{v}**" for v in (s.get("form"), s.get("roof")) if v]
    return f"    Form: {', '.join(parts)}.\n" if parts else ""


def assignment_for(ids: list[str]) -> str:
    plan = json.load(open(os.path.join(settlement.STATE, "plan.json")))
    chosen = [s for s in plan["structures"] if s["id"] in ids]
    missing = set(ids) - {s["id"] for s in chosen}
    if missing:
        raise SystemExit(f"unknown ids: {missing}")
    lines = [f"  - **{s['id']}** ({s['kind']}): x {s['x0']}..{s['x1']}, "
             f"z {s['z0']}..{s['z1']}\n{form_line(s)}    {s['notes']}" for s in chosen]
    return ("Build exactly these, and nothing else:\n\n" + "\n".join(lines) +
            "\n\nOne program builds all of them.")


def emit(pass_name: str, ids: list[str], kind: str = "builder") -> str:
    briefs = {"builder": builder_brief, "massing": massing_brief,
              "elaborate": elaboration_brief}
    text = briefs[kind](assignment_for(ids), ids)
    suffix = "_prompt.md" if kind == "builder" else f"_{kind}_prompt.md"
    p = os.path.join(settlement.STATE, f"{pass_name}{suffix}")
    open(p, "w").write(text)
    return p


if __name__ == "__main__":
    os.makedirs(settlement.STATE, exist_ok=True)
    argv = sys.argv[1:]
    kind = "builder"
    for flag, k in (("--massing", "massing"), ("--elaborate", "elaborate")):
        if flag in argv:
            argv.remove(flag)
            kind = k
    if len(argv) > 1:
        print(emit(argv[0], argv[1:], kind))
    else:
        planner = PLANNER
        # Inserted, never substituted: the tuned sections are the ones that earned their
        # wording over seven rounds, and a request that is *about this settlement* is a
        # different thing from the standing brief. $ETHOSLM_INTENT names a file whose text
        # goes in ahead of "What matters", so the record of what was asked for is a file
        # in the repo rather than a paraphrase.
        intent_path = os.environ.get("ETHOSLM_INTENT")
        if intent_path:
            planner = planner.replace(
                "## What matters",
                "## What this settlement is\n\n" + open(intent_path).read().strip()
                + "\n\n## What matters")
        if os.environ.get("ETHOSLM_SPACES"):
            planner = planner.replace("## What matters", SPACES + "\n## What matters")
        if os.environ.get("ETHOSLM_TREE"):
            # A4: the plan is a tree of parts and every part is a type instance. The
            # flat `structures` schema and the paragraph asking the planner to choose a
            # form and a roof are both replaced -- form is the type's now, and which
            # type a part is *is* the choice this paragraph was asking for.
            i = planner.index('  "structures": [')
            want = [n for n in os.environ.get("ETHOSLM_TYPES", "").split(",") if n]
            planner = (planner[:i] + TREE_SCHEMA + types_card(want or None)
                       + "\n\nUse the Write tool. Output only the file; reply with the "
                         "number of leaf parts and the depth of the tree.\n")
        open(os.path.join(settlement.STATE, "planner_prompt.md"), "w").write(planner)
        print(f"wrote {settlement.STATE}/planner_prompt.md"
              + (" (with spaces)" if os.environ.get("ETHOSLM_SPACES") else "")
              + (f" (intent from {intent_path})" if intent_path else ""))
