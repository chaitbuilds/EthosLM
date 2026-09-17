# Proposal

A proposed architecture, not a record of one. Nothing here is built. Every finding was read
off the tree at `11e76c0` and is cited so it can be checked or refuted.

The goal is a system that builds any place a sentence can name, as well as that place
deserves, without being a ring-city generator with a general-sounding vocabulary.

## Findings

**1. There is one whole-place layout, and it is concentric.**
`placeplan.concentric_layout` is the only arithmetic that lays out a place. Everything else
goes through `placesolve`, which seeds candidates from a relation, places the defining parts
against a centre and a perimeter, then tiles the ground left over "as the ring layout's
strips do". A place with no centre and no perimeter has no layout. Asked for a gridded city
on islands, the system would invent a middle and band the place around it.

**2. `RELATIONS` describes parts, not orders.**
Eleven words: `concentric`, `centre`, `edge`, `perimeter`, `gateway`, `throughout`,
`quarter`, `beside_the_centre`, `near`, `along`, `on`. One of them, `concentric`, names how
a whole place is organised. The other ten place one part against another. There is no way
to say a place is a grid, a line along a river, a cluster of settlements, or terraces up a
slope.

**3. A named place contributes prose, and nothing structural is derived from it.**
`spec_brief` asks a spec call that names a recognisable place to write an `invariants`
paragraph first: what is at the centre, the rings outward with their shares, which are
walled, how each is built and coloured, the land it stands in. `spec.py` validates it with
one `isinstance(str)` check. Three mechanical decisions are taken from it by keyword regex
in `placeplan.py`: roundness, wall mass, wall face. Beyond those it is appended to the
plan's `intent` string as prose for later calls to read. No ring, share, wall, voice or
landmark in the built plan comes from that paragraph.

**4. The questions the brief asks are the ring city's questions.**
The invariants instruction asks for the centre, the rings, their shares and their walls.
For a place organised any other way the form has no fields, so the reading either distorts
the place into rings or returns nothing usable.

**5. The centre's size is a residual with no floor.**
`spec.centre_share` is `1.0 - sum(ring shares)`. `_check_rings` refuses shares summing above
1.0 and accepts shares summing to 0.95, which leaves the palace five per cent of the place
and is legal. `CENTRE_SHARE_MIN = 0.5` exists only in its definition and in
`stages_measure`, which reports whether the compound cleared it after the city is standing.
The note beside the constant records the last reading at 34%.

**6. Round rings are blocked below the wall.**
`great_wall` draws a diagonal run at any depth. The districts tile a ring as rectangles in a
square annulus and know nothing about a diagonal, so at a regular octagon's chamfer the
outermost ring's districts overlap their own wall and coverage falls to 57% against
`RING_COVERAGE` 0.6. `RING_CHAMFER` is held at 0.25 and the code records the regular octagon
as "measured and not delivered".

**7. Footprint has never moved.**
`CEILING_FOOTPRINT_BY_KIND = {"city": 768}` is byte-identical at the first published commit
and at HEAD. Structure counts rose because `fabric_ratio` rose. The city got denser, not
larger.

**8. The type vocabulary has holes and acquisition is capped at two.**
Computed against `growth.type_gaps`: with no form set, `tower` and `bridge` have no
committed type. With `form: east_asian`, `workshop` joins them; with
`form: european_vernacular`, `temple` does. `GROWTH_CAP = 2`, so a sentence implying three
absent families stops the run by name. Authoring happens inside the run, where a build is
waiting on it.

**9. Only the judge can see.**
`model.py` sets `IMAGE_ROLES = ("judge",)`. A vision tier is configured and used once, at
the end, to compare two finished builds. Nothing at the front of the pipeline can look at a
picture of what it is being asked to make.

**10. Nothing accumulates except types.**
Types and voices persist and compound. The reading of a place, the vision assessment and
the judge's verdicts are produced and discarded. Every run re-derives what a good city of
its kind looks like from nothing.

## Proposed system

### Read the sentence

One cheap call decides what kind of request this is: a place someone could recognise, a
tradition with no specific referent, or a description that stands on its own. One field,
recorded.

### Determine the order

For a named place the first question is not how many rings it has. It is what the place is
organised by. The answer is one of a small set of **layout families**, and it decides every
question asked afterwards:

- **rings** around a centre: shares, which are walled, what is at the middle;
- **grid**: block size, avenue spacing, where the grid breaks and why, how height is zoned,
  what punches holes in it;
- **linear**: what it follows, how far it runs, what stands at each end, how deep it goes
  back from the line;
- **cluster**: how many settlements, how far apart, what joins them;
- **terraced**: how many levels, the drop between them, what sits on top.

Each family needs its own arithmetic, the way `concentric_layout` is the arithmetic for
rings. This is the largest item in this document and the one the rest depends on.

### Gather in the terms that order implies

Not a search for the place. One targeted retrieval per field the layout family needs, so
the schema is the query plan and every answer carries its source.

Two classes of source, for different things:

- **the place itself**, in images, maps and text, for what it is and how it is organised;
- **existing builds of it in the medium**, for how that becomes blocks. A build of a named
  place has already decided what block its wall is, how tall, what the roofs are made of and
  how tight the streets run, which is the conversion this pipeline finds hardest. Several
  are retrieved and compared by the pairwise vision judge that already exists, and the best
  is read for palette, proportion and massing. Read as reference, never imported as
  geometry: the asset licensing gate in the README gets worse, not better, if builds are
  copied.

### Look

The vision role opens to the front of the pipeline, not only to the judge. Proportion of
wall to city, roof colour, street tightness and silhouette are visual facts and no amount of
text retrieval substitutes for them.

### Convert

Evidence becomes the schema: shares as decimals that sum to one, walled or not, a palette
per ring or per zone, a density and a character per district. This is the hardest step in
the pipeline and it gets its own stage so it can be inspected and argued with, rather than
being buried inside a larger call.

### Gate

Nothing enters ungated. Structural facts hold only where more than one source agrees.
Numbers pass the consistency checks that already exist, plus a floor under the centre.
A person looks once, the first time a place is read.

### Acquire vocabulary before planning

The reading names the types the place needs that the library does not have. All of them are
authored in one pass, in parallel, offline against saved ground with no server and no build
waiting, and kept only where every instance stands clean. Planning begins with the
vocabulary already complete, so there is no cap to hit and no run to stop.

### Plan and build

The layout family lays out the place. The solver places what the layout does not. The
district compiler turns each district's character into streets, blocks, lots and buildings.
Ground is settled for every column before a block is placed. Types build. Lint, walk,
measure, render, judge.

### Also needed

- **A floor under the centre.** `_check_rings` already refuses shares that sum above one. It
  should refuse, or scale back, shares that leave the centre under `CENTRE_SHARE_MIN`.
- **Districts that follow the shape of their wall**, so a ring can be round.
- **Size as a declared input** with its own bounds and its own measure, so a larger place
  fails early rather than at hour nine.
- **Words for movement**: a relation for a route between parts, and a family of types that
  builds one.

## Order

Layout families first. Without them this is a ring-city generator and everything else is
decoration on that. Vocabulary as its own stage second, because it unblocks everything the
reading will ask for. Then retrieval with vision and in-medium references. The centre floor
and the district tiling are small and can go at any point. Size and movement follow.

## Open problems

**Reproducibility.** The reading is regenerated per run and the sources it reads can change
between runs, so two runs of one sentence can differ in ways that have nothing to do with
the library. Against `SEARCH_BOUND_S 7200`, `PARTS_BOUND_S 21600` and `RENDER_BOUND_S 14400`
that means a long run cannot be attributed to a change. This design does not solve it.

**Gating a reading.** A type is adopted only if every instance stands. A voice is validated
on load. There is no equivalent automatic check that a reading of a real place is correct,
and cross-source agreement plus one human look is weaker than every other gate in the
system.

**Judgment does not compound.** Types accumulate, readings and verdicts do not. A system
that keeps its verdicts has a record of what has worked; one that discards them starts over
each time.
