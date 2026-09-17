# Proposal

A proposed architecture, not a record of one. Nothing here is built. Every finding below
was read off the tree at `88bd82a` and is cited so it can be checked or refuted.

Three goals, in this order: a system that can build a named place as well as the place
deserves; a system that wastes fewer of its own runs; a system that is more general than
the one demo it was built against.

## Findings

**1. The referent pathway exists, and carries prose only.**
`stages_plan.spec_brief` asks a spec call that names a recognisable place to write an
`invariants` paragraph first, stating what is at the centre, the rings outward with their
shares, which are walled, how each is built and coloured, and the land it stands in, then
to write the schema from that paragraph and nothing else. The paragraph is validated in
`spec.py` by one `isinstance(str)` check. Three mechanical decisions are taken from it, by
keyword regex in `placeplan.py`: roundness, wall mass and wall face. Beyond those it is
appended verbatim to the plan's `intent` string, so downstream planner calls read it as
prose. It is therefore carried, but nothing structural is derived from it: no ring, share,
wall, voice or landmark in the built plan comes from the paragraph.

**2. The paragraph and the structure are never reconciled.**
The model writes the invariants, then restates them as defining parts, and the two can
disagree. This is already known: `rounds/concentric.json` preregisters a report on
"whether the fresh spec call wrote fewer walled rings than the invariants it listed",
with the policy "reported, not patched". A contradiction between a spec's own two halves
is currently an observation, not an error.

**3. The centre's size is a residual, and its floor is a measure rather than a clamp.**
`spec.centre_share` is `1.0 - sum(ring shares)`: the centre gets whatever the spec's rings
did not take. `concentric_layout` sizes the centre square from it, so the mechanism is real
and spec-driven. What is missing is a floor. `_check_rings` refuses shares that sum above
1.0 and accepts shares that sum to 0.95, which leaves the palace five per cent of the place
and is legal. `CENTRE_SHARE_MIN = 0.5` exists only in its definition and in
`stages_measure`, which reports whether the compound cleared it once the city is standing.
The note beside the constant records the last reading at 34% and "from the air it reads as
a compound among compounds rather than as the thing four terraces rise to".

**4. Round rings are blocked below the wall, not at it.**
`great_wall` draws a diagonal run at any depth, so an octagon is buildable. The blocker is
the districts, which tile a ring as rectangles in a square annulus and know nothing about a
diagonal: at a regular octagon's chamfer the outermost ring's districts overlap their own
wall and coverage falls to 57% against `RING_COVERAGE` 0.6. `RING_CHAMFER` is held at 0.25
and the code records the regular octagon as "measured and not delivered". Separately,
roundness fires only on words in the spec, and `wall_round_for` refuses to read a place's
name, so a sentence that names a round city gets square rings unless the spec call writes
roundness into its invariants.

**5. Footprint has never moved.**
`CEILING_FOOTPRINT_BY_KIND = {"city": 768}` is byte-identical at the first published commit
and at HEAD. Structure counts rose because `fabric_ratio` rose, not because the city grew.
The band is now 547 to 1824 and a recorded ringed-city spec reads 1186 where it read 585.
The city is denser, not larger.

**6. The type vocabulary has holes, and growth is capped at two.**
A spec may declare fourteen families. Computed against `growth.type_gaps` at HEAD: with no
form set, `house`, `tower` and `bridge` have no committed type. With `form: east_asian`,
`workshop` joins them. With `form: european_vernacular`, `temple` does. `house` matters
least, since the brief tells the spec call that houses are not defining parts. `GROWTH_CAP
= 2`, so a sentence implying three absent families stops the run by name.

**7. There is no transit.**
`spec.RELATIONS` is eleven words, none of which describes a route between places. There is
no type of any family for a road, a causeway, a viaduct or a rail. A city whose defining
feature is how you cross it cannot say so.

**8. The spec is regenerated per run, and runs are long.**
`rounds/ground.json` registers `SEARCH_BOUND_S 7200`, `PARTS_BOUND_S 21600` and
`RENDER_BOUND_S 14400`: twelve hours of bounds before planning, linting and measuring. The
one call that decides what is being built is unpinned, so two runs of the same sentence can
differ in ring count, shares and voices. A round can reuse an earlier round's plan, as
`rounds/demo.json` does, but that pins the plan and everything under it rather than pinning
the spec and letting the plan regenerate. There is no way to hold the sentence's reading
fixed while a library change is tested beneath it.

## Proposed system

**A. A referent is a committed artifact.**
`referents/<name>.json`: the centre, the rings with index, share, walled, voice and
character, the setting, the landmarks, and the spatial invariants including roundness.
Validated on load like a voice. Authored once, reviewed by hand, versioned, diffable,
reused across runs at no cost. The prose paragraph stays, as the record of the reading, not
as the carrier of the facts.

**B. Defining parts are derived from the referent, not restated.**
Where a referent file is loaded, the district parts, wall count, shares and voices come out
of it. The model does not retype what it just wrote. Finding 2 stops being a class of error
because the two halves are one object. Where no referent is loaded, nothing changes.

**C. A missing referent is authored by refusal.**
The pattern already exists for types in `growth.py`: detect the gap before planning, stage a
blinded authoring through the same brief and checker, gate the answer, cap it, record it,
stop by name on failure. A referent is the same shape of problem and should use the same
machinery, so that an unknown place produces a reviewable file rather than a silent guess.

**D. A tradition is a character table.**
`traditions/<name>.json`: the form, its candidate voices, and the character defaults a place
of that tradition is built to. Today a Japanese village gets its form and palette right and
takes its frontage, attachment and courtyard share from the generic density defaults, which
know nothing about the tradition. This is the smallest change on the list and it is what
makes the general path as good as the named one.

**E. The centre's floor is a clamp, not a report.**
`_check_rings` already refuses shares that sum above one. It should also refuse, or scale
back, shares that leave the centre under `CENTRE_SHARE_MIN`, so a palace cannot be planned
into insignificance. This is a validation change at spec time rather than new layout
machinery, and it is the cheapest item on this list.

**F. District tiling follows the ring's shape.**
The districts tile the annulus the wall actually draws, chamfer included, so the chamfer is
free to rise to a regular octagon without pushing coverage under its floor. This is the one
item that unblocks round rings, and it is a layout change, as the code already says.

**G. Footprint becomes a declared scale with staged bounds.**
Raising 768 is one constant; whether the solver, the compiler, the parts stage and the
render budget hold at two or three times that is unknown and is the real work. Scale should
be a declared input with its own bounds and its own measure, so a larger city fails early
and legibly instead of at hour nine.

**H. Transit is a relation and a family.**
One relation for a route between two parts, and a family of types that build it. Without
this the system cannot describe a city organised around movement, which is a general gap
that the named case happens to make obvious.

## What each buys

| | Best named build | Fewer wasted runs | More general |
|---|---|---|---|
| A referent as artifact | yes | yes | neutral |
| B parts derived | yes | yes | neutral |
| C referent by refusal | yes | no | yes |
| D tradition table | no | no | yes |
| E centre enforced | yes | no | yes |
| F tiling follows shape | yes | no | yes |
| G scale staged | yes | yes | yes |
| H transit | yes | no | yes |

## Order

A and B first, because they are cheap and because every long run after them is an
experiment that can be attributed. E and F next, because they are the two known blockers
between the current build and a good one, and both are scoped. D is an afternoon and makes
the unnamed path as strong as the named one. G is a constant plus an unknown amount of
work behind it. C and H are new machinery and should follow evidence from the rest.

## Where this could be wrong

A committed referent is a cache of a model's knowledge, and a wrong file is inherited
silently by every run that loads it, where a regenerated paragraph at least fails visibly
and differently. Deriving parts from a structured object removes the model's freedom to
notice what the schema did not anticipate. The invariants contradiction in finding 2 was
observed before the current brief existed and may already be rarer than the record
suggests. None of A to D changes what a frame looks like; only E, F and G do.

The cheapest test of the whole proposal is one run on the current brief, then a
hand-written referent file built from what it wrote, then the same run again.
