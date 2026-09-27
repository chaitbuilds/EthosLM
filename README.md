# EthosLM

Turn one sentence into a place in Minecraft.

![A walled city built with EthosLM](docs/basingse.jpg)

You describe a place in plain English. An AI agent designs it on real terrain from your world. EthosLM builds every block and writes the result into your Minecraft save.

It supports small settlements and large walled cities through an experimental, agent-guided workflow. Visual quality still needs review.

## How it works

EthosLM splits the work between an AI agent and code.

The agent decides:

- what the sentence means
- what kind of place it is
- the layout of the whole place
- which design to keep
- the materials

The code handles:

- reading the terrain
- sizes and building counts
- the exact position of every wall, street and lot
- shaping the ground
- placing every block
- checking and measuring the result
- drawing pictures of it

Whenever the agent is needed, the run pauses and writes a prompt file in `out/<name>/`. The agent reads it, writes its answer as a JSON file and runs `resume`. The run picks up where it left off. If an answer breaks the rules, EthosLM sends it back with the reason and the agent tries again.

Here is the full path from sentence to finished place.

### 1. Read the land

When you run `start`, EthosLM opens your save's files directly. Minecraft does not need to be running. It records the ground height, water depth and biomes across the world. This terrain map is saved in `out/<name>-atlas/` and only has to be made once per save.

### 2. Read the sentence

The sentence gets two separate readings.

- Rules. A fixed set of rules picks out anything the sentence states outright, such as a wall, a number, a market, a shoreline or "unwalled". No AI is involved. Each one gets an ID so the result can be checked against it.
- The agent. The agent reads the full meaning, including things rules can't catch, like "without a wall or a temple" or "small houses around a big temple". It has to quote the words it is reading from and say why.

When the two readings disagree, both are kept and the disagreement is written down. Any word that neither reading covers is tracked too, so nothing in the request gets quietly dropped.

### 3. Research, if needed

If the sentence names a real place or tradition, like the Forbidden City or a Japanese village, the agent gathers sources and writes down facts from them. Every fact is linked to the source it came from. The agent also collects reference images and writes a short visual brief for the design step.

A sentence that fully describes its own place skips this step.

### 4. Decide what kind of place it is

The agent picks:

- the kind of place: hamlet, village, town, city, keep or monument
- the parts that define it, for example three rings of walls with a palace in the middle
- a materials palette from `voices/`, or a new one if nothing there fits

EthosLM then works out the numbers itself. A village gets 12 to 40 buildings. A city gets 120 to 400. A number in the sentence, like "sixteen cottages", overrides this. Because the code does the math, the same sentence always gets the same size.

The spec also selects a planning strategy: repeated fabric for districts with a shared character, individual composition for parts planned building by building, or a mixture of both. This choice follows the spec's organisation, not a city-versus-village size threshold. Both paths use the same spatial compiler and construction pipeline.

### 5. Find sites

EthosLM scans the terrain map for good spots at a few different sizes. It ranks them by how flat they are, how much water they have and how much land is usable. If the request calls for a setting, like a desert, it only offers sites in matching biomes. Each site comes with a map.

### 6. Design

The agent gets one prompt with everything it needs: the request, the research, the candidate sites and their maps, the building types and their lot sizes, the layout patterns, the ground options and sample sheets of each palette.

It writes 2 or 3 different designs for the whole place. Each design covers:

- the outline and overall size
- rings from the centre out, and what each ring is for
- main and side streets, and which way they run
- which layout pattern fills each ring
- landmarks and where they stand
- walls and gates, if the place has them
- a central monument, like a palace, if there is one
- how the ground is treated in each ring
- the palette

For repeated fabric, the agent sets the organisation and the code lays out the individual lots. In an individual composition, the agent places each building, outdoor space and path with local coordinates, a use and a reason. The compiler checks their fit, access and ground. It probes chosen building parameters on flat pads and reports features that construction cannot deliver on the actual terrain.

The layout patterns are:

| Pattern | What it makes |
| --- | --- |
| `courts` | Lanes of courtyard houses, with shop houses on the cross streets |
| `rows` | Narrow row houses on close lanes, shops on the streets, small squares where streets cross |
| `estates` | Large walled compounds with a gate hall, main hall, several courtyards and a garden |
| `fields` | Farmland along roads, with farmsteads and small hamlets |
| `clusters` | Small settlements: houses in short runs along lanes that follow the outline, with spokes from the centre |
| `open` | Parkland, lake shore, groves and gardens |

The ground options are:

| Option | What it does |
| --- | --- |
| `terrace` | One flat level for the ring, held up by retaining walls |
| `podium` | A raised platform with ramps or stairs |
| `graded` | The natural ground smoothed into gentle slopes |
| `preserve` | The land stays as it is, with flat pads for buildings, roads and walls |

### 7. Compare the designs

EthosLM turns each design into a real plan on its site and draws a plan image. For each one it reports:

- how many homes and compounds it has
- an estimate of how many people would live there
- how much earth has to be cut and filled
- how long and tall the retaining walls get
- any problems it found

The agent looks at the plans and picks one. It can adjust any setting of the winner and confirm its palette.

### 8. Lock in the plan

EthosLM works out every exact detail of the chosen design: wall paths, gates, streets, blocks, lots, compounds and the final height of every column of ground. It splits the place into regions of 192 by 192 blocks.

If something can't work, like a lot that doesn't fit or a retaining wall that is too tall, the agent gets a revision job that points at the exact setting to change.

### 9. Build

Each region is built on its own, all offline:

1. Start from the land as it was found, with any finished neighbouring regions around it.
2. Shape the ground: terraces with faced retaining walls, gentle slopes, paved streets that stay walkable on hills, and banks where land meets water.
3. Build each building from its type in `types/`, using the chosen palette. A type is a Python file that builds one kind of thing, like a courtyard house, a gate tower or a field, at any lot size.
4. Measure what was actually built. If a house was asked for three storeys and only one fit, the record says so.
5. Save only the blocks that changed.

On a later run, regions whose inputs haven't changed are kept as they are. A change only rebuilds the regions it touches.

### 10. Draw pictures

EthosLM draws aerial and street-level pictures of the result with its own renderer. This needs no Minecraft and no server. At this point `status` says `designed`.

### 11. Review

You or your agent look at the pictures. If something should change, ask your agent to revise the design and resume.

### 12. Deliver

EthosLM checks that your server is running and backs up the save. Then it writes each region's changed blocks into the world through GDMC-HTTP. It keeps a record of what it wrote, so if delivery stops partway you can run it again. When it finishes, it prints where to stand in the game to see the place.

## What you need

- Python 3.12 or newer.
- A Minecraft Java Edition save. EthosLM builds on land that already exists in the save. If the area isn't generated yet, your agent can generate it by running a server on the save.
- A terminal AI agent, such as Claude Code. The agent does the design work, so you do not need a model API key.
- To put the build into Minecraft: a Fabric Minecraft server with the GDMC-HTTP mod. You only need this for the last step.

## Install

```sh
git clone https://github.com/chaitbuilds/EthosLM.git
cd EthosLM
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps gdpc==8.1.0
```

The last line installs gdpc on its own so it does not replace the OpenCV package from the line before it.

Check your setup:

```sh
scripts/ethoslm doctor --world /path/to/your/save
```

It prints `ready` when everything is in place.

## Quick start

**1. Start a place from a sentence.**

```sh
scripts/ethoslm start "Build a small sandstone oasis village with flat-roofed homes, shaded lanes and gardens around a pool of water." \
    --name oasis --world /path/to/your/save
```

EthosLM only reads this save. It never changes it.

**2. Let your agent answer the design jobs.**

The run pauses whenever it needs a design decision. Ask your terminal agent to run `scripts/ethoslm status oasis` and follow what it says. Each job names a prompt to read and a JSON file to write. After the agent writes the answer, it runs:

```sh
scripts/ethoslm resume oasis
```

Repeat until `status` says `designed`. That means the place is built offline and ready to review.

**3. Look at the result.**

```sh
scripts/ethoslm views oasis --move
```

Pictures are saved in `out/oasis/design/views/`. The `--move` flag also makes a short flyover GIF.

**4. Put it into Minecraft.**

Make a copy of your save. Start your GDMC-HTTP server on the copy. Then run:

```sh
scripts/ethoslm deliver oasis --save /path/to/save-copy --host http://localhost:9000
```

This shows what will be written. Add `--yes` to write it. EthosLM backs up the save first, then prints the coordinates to visit in game.

Only run one server on a save at a time.

## Commands

| Command | What it does |
| --- | --- |
| `start` | Start a new place from a sentence |
| `status` | Show where a place stands and what to do next |
| `resume` | Continue after a job is answered |
| `views` | Render the built place again. Add `--move` for a flyover GIF |
| `deliver` | Write the built place into a save. Add `--yes` to write |
| `doctor` | Check your setup |

Run `scripts/ethoslm <command> --help` for all options.

## Settings

You can set these in your shell or in a `.env` file in the repo folder.

| Variable | What it does |
| --- | --- |
| `ETHOSLM_MC_JAR` | Path to a Minecraft client jar. Views use real textures instead of flat colours |
| `ETHOSLM_PYTHON` | Use a different Python |
| `ETHOSLM_SERVER_DIR` | Server folder for the optional helper `scripts/server.sh` |
| `ETHOSLM_LIBRARY_PATH` | Extra native library paths, for NixOS |
| `ETHOSLM_MODEL_API` | Send some jobs to a model API such as Anthropic or OpenAI. See `models.json`. The design steps still need your agent |

## Try it without Minecraft

This builds a saved example plan using the terrain samples in `fixtures/`. You need no save and no server.

```sh
.venv/bin/python scripts/round.py rounds/example.json --dry-run --stage parts,finish,lint
```

## Tests

Each test is a Python script in `scripts/`. Run one like this:

```sh
.venv/bin/python scripts/test_pipeline.py
```

Tests that need a server skip on their own. [TESTING.md](TESTING.md) lists the suites that are left out and why.

## Project layout

| Folder | What is inside |
| --- | --- |
| `src/ethoslm/` | The main library and the command line tool |
| `types/` | Building types: houses, halls, temples, walls, gates, fields and more |
| `voices/` | Material palettes |
| `rounds/` | Saved run settings, including the offline example |
| `fixtures/` | Small terrain samples for tests and the example |
| `scripts/` | The `ethoslm` command, the server helper, tests and tools |
| `out/`, `run/` | Generated files. Git ignores these |

## Limits

- It is strongest at housing rows, courtyard neighbourhoods, estates, fields and monumental compounds.
- Places can use repeated fabric, individual composition or both. Individual composition supports different uses and forms, but does not guarantee visual variety.
- Composed building lots and outdoor spaces are rectangular, and buildings face cardinal directions. Paths can bend and run diagonally. Dimensions, roofs and extensions can vary within a form; repeating one house form can still look repetitive.
- A build can pass every check and still look off. Always review the views before you deliver.

## Licence

MIT. See [LICENSE](LICENSE).
