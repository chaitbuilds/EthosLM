"""A courtyard house of the Chinese tradition: four ranges round a walled court.

The siheyuan organisation, which is what the retrieved claim `chinese-architecture`
("modelled on Chinese building, specifically the Forbidden City and other places around
Beijing", closure-city reading) points at for a dwelling: the Beijing courtyard house is
the ordinary house of that city and the palace is the same organisation at monumental
scale. The form decisions and the claim each rests on:

  * **an enclosed court with the ranges facing into it, blank to the lane** -- the
    hierarchy of the Forbidden City is courts within walls, entered on axis through a
    gate; `chinese-architecture`. The library's `building(courtyard=)` lays exactly
    that ring: four ranges, the yard paved and open to the sky, the way in a passage
    through the range the lane arrives at, and an opening from the yard into every
    range (`buildlib._building_courtyard`).
  * **a main hall opposite the gate, raised a course and roofed higher** -- the
    principal hall stands at the back of the court on a platform, the side ranges
    lower; `chinese-architecture`, and a design decision beyond the claim's text: the
    source names the Forbidden City and not the room order of a house.
  * **a screen inside the gate** where the court is wide enough for one, so the way
    in turns; the same decision, same limit.
  * **the lower ring is crowded** (`lower-ring-is-crowded-upper-is-not`): the type
    stands on 9x9 and goes to 32x32, so a crowded quarter of small courts and a
    spacious one of large courts are the same type at two lot sizes.

What it **emits** is measured by `construction.outcome` and not taken on trust:
`emitted.features` names the court, the gate, the main hall, the wings and the screen,
each with a rectangle in `emitted.rects`; a court that did not stay open or a gate that
did not hang reads as `claimed_not_found` downstream. This is a form: every block is
`b.voice[role]` and the roof is the voice's profile.
"""

import random

KIND = "plot"
FORM = "east_asian"
ROLE = "urban"
FUNCTION = "dwelling"
ATTACHED = True
#: The features this type can be asked to deliver, for `envelope.table`.
FEATURES = ("courtyard", "gate", "main_hall", "screen")

PARAMS = {
    # **One storey, measured.** The Beijing courtyard house is single-storey, and so is
    # this one for a second reason the record keeps honest: `building(courtyard=)` opens
    # every range onto the court at every floor, so a two-storey ring's upper rooms in
    # the three ranges without the flight open onto air and read E003/E011 (26 of 40
    # clean on the checker's fixtures at storeys=2, 40 of 40 at one). A two-storey ring
    # needs the shell to link its upper floors, which is the library's to add; until
    # then the type declares what stands.
    "storeys": ("int", 1, 1),
    "screen": ("choice", ["wall", "planted", "none"]),
    # **The least court the design asks for** (the fabric reset round): 0 is the type's
    # own proportion (`_range_depth`); a number is the court's least side, and the
    # ranges are made shallower -- never under 3 -- until the court is at least that.
    # Set by the district's character (`court_least`), never drawn at random.
    "court": ("int", 0, 15),
}

NEEDS = {
    # measured by scripts/type_needs.py; a yard of 3 inside ranges of 3 is 9 across. The
    # fabric reset round re-swept it (square pads on the plane and the bank, both seeds,
    # every parameter): clean 9 to 22, broken from 24 (an E002 door at 24, 28 and 32),
    # so the ceiling moved from 17 to 22. The square sweep builds no neighbour;
    # **attached**, `scripts/test_attached_forms.py` and a sweep of 40 runs of four
    # (lots 13-19 wide x 15-21 deep, all four fronts, plane and bank, mixed screens and
    # seeds) stood every house lint-clean with no party-line gap and nothing over the
    # plot line once `site()` is kept off the attached side (see that script). The court
    # a lot gives, from `_range_depth` unchanged (ranges of 4 on pads 13-17): a 15x17
    # lot is a 13x13 pad at a row's end and a 5x5 court, 15x13 between two neighbours
    # and a 7x5 court; 15x19 between neighbours is 7x7, 17x19 is 9x7.
    "footprint": (9, 9, 22, 22),
    "frontage": "lane",
    "ground": "any",
    "clearance": 2,
}

#: **The lots this type needs, measured** by `envelope.table('courtyard_house',
#: n_flanks=k) for k in 0, 1, 2 (the instrument `scripts/type_needs.py --envelope` runs,
#: which measures k=0 only; driver kept at out/fr-work-types/envelope_flanks.py)`: each
#: row is the least lot on which the parameters stand with the features named, at one
#: seed (`lot_min`) and at seeds [1, 2, 3] (`lot_pref`). Read by
#: `ethoslm.envelope.lot_for` before a lot is drawn; the outcome construction measures
#: afterwards remains authoritative.
ENVELOPE = [
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "lot_min": [
   19,
   15
  ],
  "lot_pref": [
   19,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "lot_min": [
   19,
   15
  ],
  "lot_pref": [
   19,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "lot_min": [
   13,
   13
  ],
  "lot_pref": [
   13,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "lot_min": None,
  "lot_pref": None,
  "why": "courtyard_house delivers no lot up to 34x34 on a lot with free flanks with {'screen': 'none', 'storeys': 1} and ['storeys', 'screen']: ['screen'] never appeared"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 1,
  "lot_min": [
   17,
   15
  ],
  "lot_pref": [
   17,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 1,
  "lot_min": [
   17,
   15
  ],
  "lot_pref": [
   17,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 1,
  "lot_min": [
   11,
   13
  ],
  "lot_pref": [
   11,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 1,
  "lot_min": None,
  "lot_pref": None,
  "why": "courtyard_house delivers no lot up to 34x34 on a lot with 1 flank(s) attached with {'screen': 'none', 'storeys': 1} and ['storeys', 'screen']: ['screen'] never appeared"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "wall",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 2,
  "lot_min": [
   15,
   15
  ],
  "lot_pref": [
   15,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "planted",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 2,
  "lot_min": [
   15,
   15
  ],
  "lot_pref": [
   15,
   15
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "courtyard"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "gate"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "main_hall"
  ],
  "flanks": 2,
  "lot_min": [
   9,
   13
  ],
  "lot_pref": [
   9,
   13
  ],
  "why": "probed at seeds [1, 2, 3]"
 },
 {
  "params": {
   "screen": "none",
   "storeys": 1
  },
  "features": [
   "storeys",
   "screen"
  ],
  "flanks": 2,
  "lot_min": None,
  "lot_pref": None,
  "why": "courtyard_house delivers no lot up to 34x34 on a lot with 2 flank(s) attached with {'screen': 'none', 'storeys': 1} and ['storeys', 'screen']: ['screen'] never appeared"
 }
]

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

#: **The rooms come first** (the design resolution round). The clear depth, wall to
#: wall, each range needs for what it is used for: the main hall opposite the gate is
#: the principal room and is deeper (a bed or a table and a way past it); the wings and
#: the gate range are rooms of one bay (a bed along the wall and a way beside it). A
#: range is this plus its outer and its court wall. The fabric reset round's court
#: revision thinned every range to a clear depth of one -- a corridor -- because the
#: court was sized first and the rooms took what was left; here the court is what the
#: rooms leave, and a lot that cannot give both is refused with the lot it would need.
ROOM_CLEAR = {"hall": 4, "wing": 3, "gate": 3}
#: How deep the hall may grow where the pad has depth to spare, and how deep a wing.
ROOM_CLEAR_MAX = {"hall": 5, "wing": 4, "gate": 3}
#: The court this type makes where the design asks for none: a court of five is a yard a
#: person crosses in four steps with a well in one corner; smaller is a light well.
COURT_LEAST = 5
#: A court longer than it is wide by more than this gives the extra to the hall, and one
#: wider than it is long by more than this gives it to the wings: the proportion the
#: rooms keep, so a large lot is a larger house and not only a larger yard.
COURT_SLACK = 2


def form_plan(pad_w: int, pad_d: int, *, params: dict | None = None,
              front: str = "north", attached=()) -> dict:
    """**The house this pad holds, decided before a block is laid.**

        `pad_w` is the pad along the lane, `pad_d` from the lane back. Returns
        `{"ok", "court": [w, d], "clear": {range: depth}, "depths": {world side: depth},
          "need": [pad_w, pad_d], "why"}` -- `need` the least pad that holds these rooms
        round the asked court. The one calculation admission (`ethoslm.formplan`) and
        construction (`build`) both read, so a lot the compiler admits is the house the
        builder lays.
        
    """
    params = dict(params or {})
    court = max(COURT_LEAST, int(params.get("court") or 0))
    wall2 = 2
    gate = ROOM_CLEAR["gate"] + wall2
    hall = ROOM_CLEAR["hall"] + wall2
    wing = ROOM_CLEAR["wing"] + wall2
    need = [court + 2 * wing, court + gate + hall]
    cw, cd = pad_w - 2 * wing, pad_d - gate - hall
    base = {"need": need, "asked_court": court,
            "rooms": {k: v for k, v in ROOM_CLEAR.items()}}
    if cw < court or cd < court:
        return {**base, "ok": False, "court": [cw, cd],
                "why": (f"a {pad_w}x{pad_d} pad holds a court of {cw}x{cd} round rooms "
                        f"of {ROOM_CLEAR['hall']} (hall), {ROOM_CLEAR['wing']} (wings) and "
                        f"{ROOM_CLEAR['gate']} (gate range) clear; a court of {court} "
                        f"needs a pad of {need[0]}x{need[1]}")}
    # the spare, shared between the court and the rooms in proportion
    while cd > cw + COURT_SLACK and hall < ROOM_CLEAR_MAX["hall"] + wall2:
        hall += 1
        cd -= 1
    while cw > cd + COURT_SLACK + 1 and wing < ROOM_CLEAR_MAX["wing"] + wall2:
        wing += 1
        cw -= 2
    back = OPP.get(front, "south")
    flank = ("west", "east") if front in ("north", "south") else ("north", "south")
    depths = {front: gate, back: hall, flank[0]: wing, flank[1]: wing}
    return {**base, "ok": True, "court": [cw, cd], "depths": depths,
            "clear": {"hall": hall - wall2, "wing": wing - wall2, "gate": gate - wall2},
            "why": (f"a {pad_w}x{pad_d} pad: gate range {gate - wall2}, wings "
                    f"{wing - wall2}, hall {hall - wall2} clear round a {cw}x{cd} court")}


def _ranges_clear(res: dict, entrance: str, far: str) -> dict:
    """The clear depth of each range the shell laid, by use (gate, hall, wings)."""
    court = ((res.get("extras") or {}).get("courtyard") or {})
    depths = court.get("depths") or {}
    if not depths:
        return {}
    out = {}
    for side, d in depths.items():
        use = "gate" if side == entrance else "hall" if side == far else f"wing_{side}"
        out[use] = int(d) - 2
    return out


def _range_depth(side_len: int, storeys: int) -> int:
    """How deep a range is, from the size of the pad along that axis."""
    if side_len >= 24:
        return 6
    if side_len >= 18:
        return 5
    if side_len >= 13:
        return 4
    return 3


def _cells(rect):
    x0, z0, x1, z1 = rect
    return [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]


def _flood(free, start):
    seen, stack = set(), [start]
    while stack:
        c = stack.pop()
        if c in seen or c not in free:
            continue
        seen.add(c)
        stack += [(c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)]
    return seen


def _fit(b, kind, cells, y, facing, free, entry, blocked, **kw):
    """Place one fitting on the first cell that keeps every free cell reachable."""
    for c in cells:
        if c in blocked or c not in free:
            continue
        trial = free - {c}
        if entry in trial and _flood(trial, entry) != trial:
            continue
        r = b.fitting(kind, c[0], y, c[1], facing, **kw)
        if isinstance(r, dict) and r.get("ok"):
            used = {c}
            for cc in r.get("cells") or []:
                if isinstance(cc, (list, tuple)) and len(cc) >= 3:
                    used.add((int(cc[0]), int(cc[2])))
            return c, used
    return None, set()


def build(b, part, seed, **params):
    rng = random.Random(int(seed) * 7919 + 0x51)
    voice = b.voice
    label = part["label"]
    fy = part["floor_y"]
    px0, pz0, px1, pz1 = part["x0"], part["z0"], part["x1"], part["z1"]
    W, D = px1 - px0 + 1, pz1 - pz0 + 1
    storeys = params.get("storeys", 1)
    asked_storeys = storeys if isinstance(storeys, int) else 1
    storeys = 1
    screen = params.get("screen", "wall")
    if screen not in ("wall", "planted", "none"):
        screen = "wall"
    asked = {"storeys": asked_storeys, "screen": screen}

    want_court = int(params.get("court") or 0)
    if want_court:
        asked["court"] = want_court
    # **the rooms first, the court what they leave** (`form_plan`): the same answer the
    # compiler admitted this lot on. The front is the side the door is on.
    ddx, ddz = (part.get("door") or [None, None])[0], (part.get("door") or [None, None])[-1]
    front = (part.get("front") or ("west" if ddx == px0 else "east" if ddx == px1
                                   else "north" if ddz == pz0 else "south"))
    along_x = front in ("north", "south")
    fp = form_plan(W if along_x else D, D if along_x else W, params=params, front=front,
                   attached=part.get("attached") or ())
    if fp["ok"]:
        attempts = [(1, {"depths": fp["depths"]})]
    else:
        # **a lot the plan did not admit** -- a retained plan drawn before rooms were
        # sized, or a court asked beyond the lot. The house still stands on the type's
        # old proportion, and the record says the rooms are short rather than hiding it.
        rw, rd = _range_depth(W, storeys), _range_depth(D, storeys)
        if want_court:
            rw = max(3, min(rw, (W - want_court) // 2))
            rd = max(3, min(rd, (D - want_court) // 2))
        attempts = [(1, (W - 2 * rw, D - 2 * rd)), (1, (W - 6, D - 6))]
    res, used_st = None, None
    rung = None
    main = (px0, pz0, px1, pz1)
    for i, (st, yd) in enumerate(attempts):
        if not isinstance(yd, dict) and (yd[0] < 3 or yd[1] < 3):
            continue
        r = b.building(label, *main, st, "hip", mat=dict(voice), courtyard=yd,
                       openings="rhythm", stair="auto")
        if isinstance(r, dict) and r.get("ok"):
            res, used_st, rung = r, st, i
            break
    if res is None:
        return {"ok": False, "label": label, "reason": "no courtyard ring stood",
                "emitted": {"requested": asked, "storeys": 0, "attempt": None,
                            "fallback": "no shell stood",
                            "omitted": ["storeys", "courtyard"], "features": {}}}
    court = res["courtyard"]
    yx0, yz0, yx1, yz1 = court["yard"]
    entrance = court["entrance"]
    corridor = [tuple(c) for c in court["corridor"]]
    far = OPP[entrance]
    door = res.get("door")
    dx, dz = (int(door[0]), int(door[2])) if door and len(door) == 3 else (part["door"][0], part["door"][-1])

    # ---- blank to the lane and to the neighbours -------------------------------------
    # The fabric reset round. A siheyuan looks inward: from the hutong one sees a grey
    # wall and a gate, and the rooms take their light from the court. `building()` lays
    # windows on a rhythm along every outside face, so the lane face read as a row of
    # windows, and on a side the plan says is **attached** they were windows in a party
    # wall looking into the next house's wall a block away. So the openings the shell
    # cut in the gate's face and in every attached face are closed again with the wall;
    # the court-side openings, and those on a free flank or the rear, stand.
    attached = sorted({str(v).strip().lower() for v in (part.get("attached") or ())}
                      & {"north", "south", "east", "west"})
    blank = set(attached) | {entrance}
    on_face = {"north": lambda x, z: z == pz0, "south": lambda x, z: z == pz1,
               "west": lambda x, z: x == px0, "east": lambda x, z: x == px1}
    wall_blk = b.block(voice["wall"])
    blanked = 0
    for lt in res.get("lights") or ():
        if not (isinstance(lt, (list, tuple)) and len(lt) == 3):
            continue
        lx, ly, lz = (int(v) for v in lt)
        if (lx, lz) == (dx, dz) or not any(on_face[s](lx, lz) for s in blank):
            continue
        for yy in (ly + 1, ly + 2):
            b.place_block(lx, yy, lz, wall_blk)
        blanked += 1

    # ---- the ranges, from the record the shell wrote --------------------------------
    rooms = [tuple(r) for r in (res.get("rooms") or []) if len(r) == 5]
    ground = [r for r in rooms if r[1] == fy]
    upper = [r for r in rooms if r[1] > fy]

    def range_of(room):
        x0, _y, z0, x1, z1 = room
        cx, cz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        if cz < yz0:
            return "north"
        if cz > yz1:
            return "south"
        return "west" if cx < yx0 else "east"

    by_range = {}
    for r in ground:
        by_range.setdefault(range_of(r), []).append(r)

    # ---- the main hall opposite the gate: a course up, and a lantern of its own ------
    hall_rect = None
    for r in by_range.get(far, []):
        x0, _y, z0, x1, z1 = r
        hall_rect = [x0 - 1, z0 - 1, x1 + 1, z1 + 1]
        step = b.block(voice["floor"], "slab")
        # a raised threshold band along the court side of the hall
        if far == "north":
            cells = [(x, z1 + 1) for x in range(x0, x1 + 1)]
        elif far == "south":
            cells = [(x, z0 - 1) for x in range(x0, x1 + 1)]
        elif far == "west":
            cells = [(x1 + 1, z) for z in range(z0, z1 + 1)]
        else:
            cells = [(x0 - 1, z) for z in range(z0, z1 + 1)]
        ways = {(int(c[0]), int(c[2])) for c in (court["ways"].get(far) or [])}
        for (x, z) in cells:
            if (x, z) in ways or (x, z) in corridor:
                continue
            if b.get_block(x, fy + 1, z) == "air" and b.get_block(x, fy, z) != "air":
                # the platform edge reads as a course of the footing under the sill
                b.place_block(x, fy, z, b.block(voice["footing"], "accent"))
        break

    # ---- the screen inside the gate, where the court is wide enough ------------------
    screen_rect = None
    if screen != "none" and corridor:
        # the passage's last cell before the court, and the way it faces into it
        inward = {"south": (0, -1), "north": (0, 1), "west": (1, 0), "east": (-1, 0)}[entrance]
        last = next((c for c in corridor
                     if yx0 <= c[0] + inward[0] <= yx1 and yz0 <= c[1] + inward[1] <= yz1
                     and not (yx0 <= c[0] <= yx1 and yz0 <= c[1] <= yz1)), corridor[-1])
        # two cells into the court: one is the court's edge row, which the ways along it
        # need clear
        sx, sz = last[0] + 2 * inward[0], last[1] + 2 * inward[1]
        across = (inward[1], inward[0])
        cells = [(sx + across[0] * k, sz + across[1] * k) for k in (-1, 0, 1)]
        room_each_side = all(yx0 + 1 <= c[0] <= yx1 - 1 and yz0 + 1 <= c[1] <= yz1 - 1
                             for c in cells)
        yard_w = (yx1 - yx0 + 1) if across[0] else (yz1 - yz0 + 1)
        yard_d = (yz1 - yz0 + 1) if across[0] else (yx1 - yx0 + 1)
        if room_each_side and yard_w >= 7 and yard_d >= 5:
            if screen == "wall":
                for (x, z) in cells:
                    b.place_block(x, fy + 1, z, b.block(voice["wall"]))
                    b.place_block(x, fy + 2, z, b.block(voice["wall"]))
                    b.place_block(x, fy + 3, z, b.block(voice["trim"], "slab"))
            else:
                for (x, z) in cells:
                    b.place_block(x, fy + 1, z, b.joinery(voice, "fence"))
                b.place_block(cells[1][0], fy + 1, cells[1][1], "flowering_azalea")
            screen_rect = [min(c[0] for c in cells), min(c[1] for c in cells),
                           max(c[0] for c in cells), max(c[1] for c in cells)]

    # ---- the court: a well or a lantern, off the way through -------------------------
    keep = set(corridor)
    for side, laid in (court.get("ways") or {}).items():
        for c in laid:
            keep.add((int(c[0]), int(c[2])))
            keep.add((int(c[0]) + (1 if side == "west" else -1 if side == "east" else 0),
                      int(c[2]) + (1 if side == "north" else -1 if side == "south" else 0)))
    if screen_rect:
        for c in _cells(screen_rect):
            keep.add(c)
    yard_cells = set(_cells((yx0, yz0, yx1, yz1)))
    court_free = yard_cells - keep
    court_entry = corridor[-1] if corridor else next(iter(yard_cells))
    if (yx1 - yx0 + 1) >= 5 and (yz1 - yz0 + 1) >= 5:
        corner = [(yx0 + 1, yz0 + 1), (yx1 - 1, yz0 + 1), (yx0 + 1, yz1 - 1), (yx1 - 1, yz1 - 1)]
        rng.shuffle(corner)
        _fit(b, "well", corner, fy + 1, OPP[far], yard_cells - keep | {court_entry},
             court_entry, keep, mat=voice["footing"], room="courtyard")
    else:
        corner = [(yx0, yz0), (yx1, yz0), (yx0, yz1), (yx1, yz1)]
        rng.shuffle(corner)
        _fit(b, "light", corner, fy + 1, OPP[far], yard_cells - keep | {court_entry},
             court_entry, keep, mat=voice["trim"], room="yard")

    # ---- what the rooms are for
    # --------------------------------------------------------
    def furnish(room, plan, side, blocked):
        x0, y, z0, x1, z1 = room
        cells = set(_cells((x0, z0, x1, z1)))
        # the way from the yard into this range, and the way in off the lane
        ways = {(int(c[0]), int(c[2])) for c in (court["ways"].get(side) or [])}
        entry = None
        for w in ways:
            for dx_, dz_ in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                c = (w[0] + dx_, w[1] + dz_)
                if c in cells:
                    entry = c
                    break
            if entry:
                break
        if entry is None:
            entry = next(iter(cells))
        blocked = set(blocked) | ways | {entry}
        for w in ways:
            for dx_ in (-1, 0, 1):
                for dz_ in (-1, 0, 1):
                    blocked.add((w[0] + dx_, w[1] + dz_))
        free = cells - {c for c in blocked if c != entry}
        edge = [c for c in cells if c[0] in (x0, x1) or c[1] in (z0, z1)]
        rng.shuffle(edge)
        mid = [c for c in cells if c not in edge]
        rng.shuffle(mid)
        stand = y + 1
        for kind, kw, where in plan:
            got, used = _fit(b, kind, where(edge, mid), stand, OPP[side], free, entry,
                             blocked, **kw)
            if got:
                free -= used
                blocked |= used

    stairs = set()

    def _collect(obj, depth=0):
        if depth > 6:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _collect(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            ints = [v for v in obj if isinstance(v, int) and not isinstance(v, bool)]
            if len(obj) == 3 and len(ints) == 3:
                stairs.add((int(obj[0]), int(obj[2])))
                return
            for v in obj:
                _collect(v, depth + 1)
    _collect(res.get("stairs"))
    # **and the way the shell holds open** from each storey's way in to the foot of its
    # flight (`Builder.flight_way`): a fitting there is taken back out with a refusal
    for c in (getattr(b, "flight_way", None) or ()):
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            stairs.add((int(c[0]), int(c[2])))
    # the foot of a flight and the way to it are the shell's: a fitting there is taken
    # back out with a refusal, so nothing is offered within a cell of a tread
    for (sx, sz) in list(stairs):
        for dx_ in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                stairs.add((sx + dx_, sz + dz_))
    lane_way = set(corridor) | {(dx, dz)}
    for side, rs in by_range.items():
        for r in rs:
            area = (r[3] - r[0] + 1) * (r[4] - r[2] + 1)
            if any(r[0] <= sx <= r[3] and r[2] <= sz <= r[4] for (sx, sz) in stairs):
                # the range that carries the flight: its floor is the way to the stair,
                # and building() holds that open; a light only, on the far wall
                furnish(r, [("light", {"mat": voice["trim"], "room": "house"}, lambda e, m: e)],
                        side, stairs | lane_way)
                continue
            if side == far:
                plan = [("bed", {"mat": voice["floor"]}, lambda e, m: e),
                        ("table", {"mat": voice["floor"]}, lambda e, m: m if area >= 12 else []),
                        ("light", {"mat": voice["trim"], "room": "hall"}, lambda e, m: e),
                        ("store", {"mat": voice["floor"], "extent": 1}, lambda e, m: e if area >= 15 else [])]
            elif side == entrance:
                plan = [("bench", {"mat": voice["floor"]}, lambda e, m: e),
                        ("light", {"mat": voice["trim"], "room": "store"}, lambda e, m: e),
                        ("store", {"mat": voice["floor"], "extent": 1}, lambda e, m: e if area >= 12 else [])]
            else:
                plan = [("bed", {"mat": voice["floor"]}, lambda e, m: e if area >= 9 else []),
                        ("shelf", {"mat": voice["floor"]}, lambda e, m: e),
                        ("light", {"mat": voice["trim"], "room": "house"}, lambda e, m: e)]
            furnish(r, plan, side, stairs | lane_way)
    for r in upper:
        side = range_of(r)
        plan = [("bed", {"mat": voice["floor"]}, lambda e, m: e),
                ("light", {"mat": voice["trim"], "room": "house"}, lambda e, m: e)]
        furnish(r, plan, side, stairs)

    # ---- the record: what stands, said by the type, measured by construction ---------
    wings = [s for s in ("north", "south", "west", "east") if s not in (entrance, far)]
    rects = {"main": list(main), "courtyard": [yx0, yz0, yx1, yz1],
             "gate": [dx, dz, dx, dz]}
    if hall_rect:
        rects["main_hall"] = hall_rect
    for s in wings:
        rects[f"wing_{s}"] = [int(v) for v in (res["extras"]["courtyard"].get("ranges") or {}).get(s, ())] \
            if isinstance(res.get("extras"), dict) and (res["extras"].get("courtyard") or {}).get("ranges") else None
    rects = {k: v for k, v in rects.items() if v}
    if screen_rect:
        rects["screen"] = screen_rect
    gave = []
    if rung:
        gave.append("ladder: a smaller court")
    if screen != "none" and not screen_rect:
        gave.append("plan: the court is too narrow for a screen inside the gate")
    emitted = {
        "requested": asked,
        "storeys": int(used_st), "attempt": int(rung or 0),
        "fallback": "; ".join(gave) or None,
        "omitted": ([] if used_st >= asked_storeys else ["storeys"])
                   + ([] if (screen == "none" or screen_rect) else ["screen"]),
        "features": {"courtyard": True, "gate": True, "main_hall": bool(hall_rect),
                     "wings": len(wings), "screen": bool(screen_rect), "chimney": False},
        "rects": rects,
        "floors": list(res.get("floors") or []),
        "court": {"yard": [yx0, yz0, yx1, yz1], "entrance": entrance, "hall": far,
                  "asked": want_court or None,
                  "held": (None if not want_court else
                           min(yx1 - yx0 + 1, yz1 - yz0 + 1) >= want_court)},
        # the rooms the plan decided and the ranges the shell laid: the clear depth of
        # each range, wall to wall, by its use -- measured again on the blocks by
        # `ethoslm.formplan.measure_courtyard`
        "form_plan": {k: fp.get(k) for k in ("ok", "court", "clear", "need", "why")},
        "ranges_clear": _ranges_clear(res, entrance, far),
        "rooms_short": (not fp["ok"]),
        # what the plan said about the lot's flanks, and the windows closed for it
        "attached": attached,
        "blanked_openings": blanked,
    }
    if door and len(door) == 3:
        b.check_door(door[0], door[1], door[2])
    b.seal_voids(px0, pz0, px1, pz1, b.block(voice["wall"]), max_cells=400)
    b.check_walkable(label)
    b.check_attached()
    return {"ok": True, "label": label, "ridge_y": res.get("ridge_y"), "emitted": emitted}
