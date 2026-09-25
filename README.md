# EthosLM

**One sentence in, a finished place out** is the goal. EthosLM is an open research project
building a general architecture for creating convincing Minecraft places from natural
language. A request such as “Build a walled city” or “Build a fishing village” should lead
to a complete place whose organisation, buildings and landscape emerge from the request,
references and reusable capabilities.

The immediate target is an expansive, beautiful city, followed by smaller contrasting
examples. The system is under development; reliable autonomous composition and finished
city quality remain open work.

## Architecture

**The model composes; the library builds.** Interpretation and references produce a
programme of requirements and relationships. Capability selection, spatial design and
ground preparation turn that programme into buildable parts. Generators construct the
buildings, boundaries and open spaces; inspection feeds findings back to the decisions
that can address them. Terminal agents can answer staged jobs without a model API.

A place is hierarchical: regions contain streets, buildings and open land, and a large
compound can contain smaller parts. Types supply reusable forms; voices supply material
and roof profiles. Compiled building sites carry pad, floor, orientation, entrance and
landing decisions into routing and construction.

Physical checks measure emitted blocks, access and required features. Visual inspection
evaluates composition and character. A feasible plan or a passing proxy does not establish
a convincing place. Revisions need current evidence, explicit dependencies and recoverable
candidates. Local rebuilds reuse unaffected work.

Shared finishing can add contextual materials and detail over owned surfaces. Its value
must be demonstrated in built views alongside the larger architectural decisions.

## Where it stands

The production path reaches whole small builds and city sections. Courtyard and shop
houses now share constructive geometry across lot admission and construction: usable
rooms, courts, storeys and row-end conditions determine what fits. Built inspection and
production revisions improve access and ground while protecting working buildings.

The latest evaluated section has usable courtyard homes, two-storey shops and a market
with built fronts. It still reads as separate clusters rather than a complete neighbourhood.
Parent allocation retains older land budgets, and composition checks can exempt missing
frontage. Current work connects complete architectural and spatial demand to parent layout,
so streets, district dimensions and site choices can change before construction.

City hierarchy, connected transitions, expansion, shared finishing and demo capture remain.

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
