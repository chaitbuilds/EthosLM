# EthosLM

**One sentence in, a finished place out.** Type "Build a walled town with a market and a
keep," or "Build a fishing village," and a complete settlement gets planned, placed, built
and checked inside a Minecraft world, with no person involved at any step. You give no
coordinates, no size and no style, and you answer no follow-up questions. Nothing is left
for you to fix afterwards. The system picks where to build, how big to make it, what palette
of blocks to use, and lays every block itself.

What this project delivers is the **architecture** behind that. It is general. The towns it
builds are tests of the architecture, not the product.

## Why Minecraft

Minecraft is predictable and built from whole units: a fixed set of blocks, a grid of round
numbers, known physics, and lighting you can work out in advance. Everything that normally
makes 3D modelling hard, like curved surfaces, texture mapping, materials and shading, does
not exist here. What is left is deciding which block goes where, and that is the real
problem.

## How it works

**The model writes the plan, the library does the building.** A model reads your sentence
and the shape of the ground, then writes short programs that call a building library. The
library turns those programs into blocks. The model decides *what*: the location, the
layout, the kinds of buildings, the palette. The library decides *how*: every physical
rule, and the buildings themselves.

**A bug stays fixed once the library owns it.** If the library builds something correctly
every time, that kind of bug can never come back. If you only check for it afterwards, it
keeps returning. Stair direction, doors, entrances, indoor staircases, building on uneven
ground, water, outer walls, the monument, and the route up to an upper floor were all repeat
offenders until the library took charge of them, and none has come back since. The library
grows by admitting what it cannot do yet, and that becomes the next thing to add.

**A place is a tree of parts, and a type is a shape.** A plan is a hierarchy: a city holds
rings, rings hold districts, districts hold plots. Walls are the edges, gates are the
openings, and squares and fields are the open areas. Every leaf of that tree is an instance
of a *type*, which is a generator for one kind of part. You write a type once, test it
across different random seeds, sizes, ground and palettes, and after that you can place
it anywhere for free. A type states what ground it needs and what it is for, but never names
a material. It takes its palette from the settlement's *voice*. A monument is not simply a
large type. It is a place in its own right, planned recursively into halls, courtyards, an
inner wall and gates, which is also how a castle works.

**The builder can check its own work, and the library is verified offline.** Every call the
model makes carries the real checker with it: the linter and the walking model, running on
the actual cached ground. The model can run that checker as often as it likes before it
finishes. The library itself is verified with no model involved: a hand-written reference
type is built across a bank of saved ground, producing hundreds of buildings in a few
minutes and making zero model calls. Model calls are for questions about the model, never
for finding a bug in the library.

**Two ways of reviewing, and neither gives a score.** Automatic checks catch what is
*broken*, never what is *ugly*, and they are a fixed minimum standard. Separately, a judge
compares two builds side by side, without being told which is which or which came first, and
says which one is better made. Only a person decides whether something is *good*, and that
is a diagnosis of what the checks cannot see. It is never a stage in the automated loop.

**The harness is predictable, and only the content varies.** A run is one config file and
one command. Every stage, camera angle, decision and record is code. A `--dry-run` builds
the whole place in memory with no server, so "prove it before you write it" is a rule a run
can actually follow.

## Where it stands

A city now stands, built from a single sentence: concentric ring walls, gates, buildings
that get denser as you move from the farmland to the wealthy centre, a grand palace
district, all on green ground the system chose, in a palette it wrote itself. It is built,
checked, walkable and rendered. What is left is polish.

## Layout

    src/ethoslm/  the library, linter, walk model, judge, deterministic preview and the
                      model adapter; pipeline/ holds the staged driver (plan, build,
                      measure, media, and the seam where authoring is blinded)
    types/            one file per kind of part: FORM, ROLE, KIND, PARAMS, NEEDS and
                      build(b, part, seed, **params). A part arrives already placed,
                      carrying the settlement's palette; a type names neither ground
                      nor material
    voices/           one JSON per style: material roles, roof shape, prose for the
                      brief. Checked on load; the model can write one from a sentence
    rounds/           one JSON per run: the input, the flags, and the thresholds that were
                      registered before it ran
    scripts/          round.py (run a config), check_types.py, type_needs.py,
                      terrain_bank.py, test_*.py (the offline suites), server scripts
    fixtures/         cached worlds and saved plans that the offline tools rely on
    out/, run/        renders, caches, worlds, server. Never committed

## Running it

Linux or macOS, Python 3.12 or newer. On Windows, use WSL. On NixOS only, see **Native
libraries** at the end.

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install --no-deps gdpc==8.1.0   # see requirements.txt
    source scripts/env.sh

**The offline suites.** No server, no model, no key. Every suite that writes blocks says so
and skips.

    for t in scripts/test_*.py; do "$PY" "$t" || echo "FAILED $t"; done

**A place, built offline from a world we ship.** A saved plan on a cached world that comes
with the repository: placed, built out of the committed types, linted, with no server and no
model call anywhere in it. About three minutes.

    "$PY" scripts/round.py rounds/example.json --dry-run --stage parts,finish,lint

**A place from a sentence.** `rounds/town.json` is one sentence and nothing else, with no
coordinate, no size and no palette. This one calls a model, so read **Models** first.

    "$PY" scripts/round.py rounds/town.json --dry-run                  # no server
    bash scripts/mcrun.sh scripts/round.py rounds/town.json --live --wait

**Models.** By default, every call to the model is handed to a supervising agent, which is
how all of the development work was done: the run stops, writes the request to disk, and
carries on once an answer is there. To run it without supervision, `models.json` maps each
role to a model. The roles are the place spec, the planner, type authoring, revision and the
vision judge. `model.py` speaks both the Anthropic and the OpenAI-compatible wire formats,
so any hosted provider or local server works. Set `ETHOSLM_MODEL_API` and a key, or override
a single role with `ETHOSLM_MODEL_<ROLE>`.

**The pipeline runs Python that a model wrote.** That is the design, since the model writes
programs against the library, but it does mean a run executes model-written code in your
process and on your file system. Run it somewhere that is acceptable: a container, a
throwaway user account, or a virtual machine.

**A server.** The live path needs Minecraft with the GDMC-HTTP mod on `localhost:9000`
(Fabric or NeoForge, GDMC-HTTP 1.8.4, WorldEdit 7.4.2, Minecraft 1.21.11). `scripts/mcrun.sh`
runs one session: server up, build, flush to disk, server down. Nothing offline needs it.

**Native libraries (NixOS only).** Put the store paths in an untracked `.env` (`NIX_GLIBC`,
`NIX_GCCLIB`, `NIX_ZLIB`) so the wheels can find them. Everywhere else the system supplies
them. `ETHOSLM_LIBRARY_PATH`, `ETHOSLM_PYTHON` and `ETHOSLM_JAVA` override the rest.

## Licence

MIT. See `LICENSE`.

<!-- PRINCIPAL: decisions to make before this is published.
     1. The clone line and any badges: no repository URL is written anywhere in this file.
     2. Asset and corpus licensing is unread and is a hard gate: Mojang's EULA, and the
        block-registry data in src/ethoslm/data/blocks_*.json, which was dumped from a
        server jar. Decide whether that file may ship as it is, be regenerated by the
        reader, or be replaced.
     3. Whether to publish the cached worlds and renders as a download, and under what
        terms; fixtures/ ships the small ones the suites need and nothing else.
     4. Whether to name the demo the system was built against anywhere in the public repo.
-->
