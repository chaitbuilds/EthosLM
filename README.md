# EthosLM

EthosLM builds Minecraft places from a sentence. A terminal agent reads the request,
writes the place's programme and proposes whole-place designs; the library compiles the
chosen design into streets, lots, ground and buildings and constructs it region by
region. No model API is needed: every judgement is a staged job with a prompt to read
and a file to write, which a terminal agent (or a person) answers.

It has produced a large walled city and a small desert village through agent-guided
design and iteration. The city is the accepted demo; the village demonstrated transfer
but its repetitive composition was not visually accepted. Small-place layouts currently
favour houses on lanes round a centre. Richer functional variety and other organisations
remain limited. Unsupported schema values are refused; semantic coverage still needs
agent review.

## Install

Python 3.12 or newer, and a Minecraft Java save whose terrain is generated.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps gdpc==8.1.0
scripts/ethoslm doctor --world /path/to/save
```

On NixOS, put native library paths in an untracked `.env` (`NIX_GLIBC`, `NIX_GCCLIB`,
`NIX_ZLIB`) or set `ETHOSLM_LIBRARY_PATH`. `ETHOSLM_PYTHON` names another interpreter.
Views are textured when `ETHOSLM_MC_JAR` names a Minecraft client jar, flat otherwise.

## A place from a sentence

```sh
scripts/ethoslm start "Build a small sandstone oasis village with flat-roofed homes, shaded lanes and gardens around a pool of water." \
    --name oasis --world /path/to/save
```

`start` reads the save's terrain once into `out/oasis-atlas/` (heights, water, biomes;
the save is only read) and runs until the first job. Then, until the place is built:

```sh
scripts/ethoslm status oasis     # WORKING, WAITING (on a job), BLOCKED, or DESIGNED
# the agent reads the prompt named after READ and writes the file named after WRITE
scripts/ethoslm resume oasis
```

The jobs, in order: the sentence read into requirements; the place spec (kind, setting,
palette); two or three whole-place designs, written against the candidate sites, the
forms the library has and the grains it composes; and the adoption of one after they
are compiled and compared. An answer that does not read is handed back with the reason
(`NOTE`), and `status` keeps saying WAITING until it is answered. Waiting is not success.

When the design is adopted, `resume` builds every region **dry** -- in `out/oasis/`, off
the save's unaltered ground -- and draws views (`out/oasis/design/views/`). Nothing is
written into any world. `scripts/ethoslm views oasis --move` also draws a five-second
camera move over it (`views/move/move.gif`) and prints the same move as two `/tp` poses.

## Delivering a place

Delivery writes the built regions into a save you choose through a running server with
the GDMC-HTTP mod. Use a **copy** of the save the terrain was read from, and never open
one save in two servers.

```sh
cp -r /path/to/save /path/to/oasis-save
# serve the copy with a Fabric server and the GDMC-HTTP mod; `scripts/server.sh` starts
# one by hand (ETHOSLM_SERVER_DIR names the server directory, ETHOSLM_JAVA the java)
scripts/ethoslm deliver oasis --save /path/to/oasis-save --host http://localhost:9000
scripts/ethoslm deliver oasis --save /path/to/oasis-save --host http://localhost:9000 --yes
```

The first command only says what it would do. The save is snapshotted under
`run/snapshots/` before the first write; the command prints a `/tp` pose to stand at.
To walk the save with an unmodded client, restart its server with the GDMC-HTTP mod
moved out of `mods/`.

## Terminal agents and model APIs

A terminal agent answers the jobs by reading each prompt, looking at the images it
names (candidate site maps, plan previews) and writing JSON. `models.json` can route
roles to an API for supported stages; the design workflow above relies on a supervising
agent. A fully unattended text-to-world workflow has not been demonstrated.

## Repository

- `src/ethoslm/`: interpretation, design compiler, ground, construction, measurement.
- `types/`, `voices/`: building forms with their measured lot sizes, and palettes.
- `scripts/`: the entry point, the round controller, a server helper and the tests.
- `fixtures/`: small cached terrain and recorded plans the offline tests read.
- `out/`, `run/`: generated state, atlases, snapshots and worlds (not version controlled).

`scripts/round.py rounds/example.json --dry-run --stage parts,finish,lint` builds a
recorded plan on shipped terrain with no server and no save. Run tests with the environment's
Python; some cases require a server or recorded inputs and skip without them. See
[test coverage](TESTING.md) for exclusions and known failing development suites.
Generated type code runs in the build process. The public site-reservation ledger starts
empty; reservations from one world are not restrictions on another.

## Licence

See [LICENSE](LICENSE).
