"""The original request, read into requirements with IDs, and checked against the place.

**This module reads the sentence and nothing else.** Not the spec, not the plan, not a
model's interpretation of any of them. That is the whole of why it exists: the audit's
counterexample was a sentence asking for a walled village whose spec omitted the wall,
and every downstream check passed because every downstream check is generated *from the
spec*. A check generated from the thing being checked cannot find an omission.

So there are two readings of every request and they are kept apart:

  - the **model's**, which is `place.json` -- rich, structural, and answerable for how
    the place is organised;
  - **this one**, which is narrow, literal, deterministic and has no opinions. It finds
    the handful of things a sentence says outright -- a wall, a count, a market, a
    shoreline, "unwalled" -- gives each an ID, and then asks whether the place that was
    planned and built actually has them.

A requirement this module cannot express is recorded `unsupported`, which is a failure
and not a shrug: "the system does not do harbours" is a true and useful answer, and
quietly building a village without one is not.

Nothing here weakens: `coverage` may write `satisfied`, `failed`, `unresolved` or
`unsupported` on a requirement and may never change its `says`, its `wants` or its
`hard` flag. A repair that wants an easier requirement has to change the sentence, and
the sentence is immutable.

no requirement at all -- and `holds()` therefore returned True before a single block was
laid. The same for "Build a Japanese village.", and "German", "bakery" and "aqueduct"
all fell out of a combined request. A keyword table is a finite instrument pointed at an
open domain, and the failure mode of a finite instrument is not "it misses things": it
is that **missing everything reads exactly like asking for nothing**.

So three more kinds of requirement come out of a sentence, and none of them is a keyword
that names a building:

`unclaimed_words` is the independent check that produces the last of those: it walks the
sentence and asks which columns no requirement's own phrase covers. It is the one thing
here that is *not* a table of things we thought of -- it is a table of everything else --
and it is what makes an empty supported subset impossible to mistake for a satisfied
request.
"""
from __future__ import annotations

import contextlib as _contextlib

import os
import re

from . import contracts, spec as spec_mod

#: The sentence's own words for a thing the place must contain, and the family the
#: library would build it as. `None` is a thing the sentence can name and the schema
#: **cannot express** -- recorded `unsupported` rather than approximated by a square.
#: Ordered longest-phrase-first inside each row so "market square" is a market and not a
#: square, and matched on word boundaries so "walled" never fires on "stonewalled".
FEATURES = (
    ("wall",     "wall",     ("city wall", "town wall", "ring wall", "ring walls",
                              "curtain wall", "ramparts", "rampart", "walled",
                              "walls", "wall")),
    ("gate",     "gate",     ("gatehouse", "gateway", "gates", "gate")),
    ("market",   "square",   ("market square", "marketplace", "market place",
                              "market")),
    ("square",   "square",   ("public square", "plaza", "piazza", "square")),
    ("keep",     "keep",     ("citadel", "fortress", "castle", "keep")),
    ("palace",   "palace",   ("palace",)),
    ("temple",   "temple",   ("cathedral", "monastery", "temple", "shrine",
                              "church")),
    ("tower",    "tower",    ("watchtower", "bell tower", "tower")),
    ("bridge",   "bridge",   ("bridge",)),
    ("monument", "monument", ("monument", "obelisk", "statue")),
    ("workshop", "workshop", ("workshops", "workshop", "forge", "smithy")),
    ("farmland", "district", ("farmland", "farmsteads", "farms", "fields")),
    # ...and the things a sentence says that this schema has no family for. A place
    # asked for a harbour and given a square has been given a different place.
    ("harbour",  None,       ("harbour", "harbor", "docks", "dock", "quay", "wharf",
                              "port")),
    ("mine",     None,       ("mine", "quarry")),
    ("mill",     None,       ("windmill", "watermill", "mill")),
)

#: The words that make a feature word an **absence** where the feature's own phrases are
#: not enough. A negation is generated for every feature from its own phrases (see
#: `_absent_phrases`); this table holds the ones a phrase cannot be derived from --
#: `unwalled` is not "without a wall" spelled differently, it is a different word. **The
#: review's counterexample, and why the table alone was the defect.** `ABSENT` had one
#: row, for walls, so "Build a village without a temple." read straight through its own
#: negation and stated a requirement for a temple. A negation that scopes over one
#: feature is not a negation; it is a special case, and the sentence it reverses is
#: exactly the sentence a reader would write to test one.
ABSENT = {"wall": ("unwalled", "open village", "unfortified")}

#: How a negation is written in front of a feature's own phrase. Applied to every phrase
#: of every feature, longest first, so "without a market square" negates the market and
#: not the square inside its name.
NEGATIONS = ("without a ", "without an ", "without any ", "without ",
             "with no ", "no ", "not a ", "lacking a ", "lacking ")

#: The whole-place organisations the sentence can ask for, and whether the system has a
#: layout policy for each. A policy the system lacks is `unsupported` and the run says
#: so before it builds something else.
LAYOUTS = (
    ("concentric", True,  ("concentric", "ring walls", "rings of", "in rings",
                           "ringed")),
    ("shoreline",  True,  ("shoreline", "shore line", "the shore", "coastline",
                           "the coast", "waterfront", "lakeside", "riverside",
                           "along the water", "beside the water", "by the sea",
                           "by the lake", "seafront")),
    ("grid",       False, ("grid", "gridiron", "grid of streets", "rectangular blocks")),
    ("linear",     False, ("strung along", "linear", "one long street",
                           "along the road", "ribbon")),
    ("terraced",   False, ("terraced", "on terraces", "stepped up the", "up the hillside")),
)

#: What the sentence says the buildings face. A frontage rule is a spatial requirement
#: and the shoreline case turns on it.
FACING = (("water", ("facing the water", "face the water", "facing the sea",
                     "face the sea", "facing the lake", "overlooking the water",
                     "overlooking the sea", "looking out over the water")),
          ("street", ("facing the street", "face the street", "fronting the street")))

#: **What the sentence says the buildings are like**, and the axis each word moves. The
#: review's first counterexample: `Build a dense city with tall buildings.` produced no
#: requirement at all, because `dense` and `tall` were listed among the words a sentence
#: spends on plain description. They are not description -- they are the two properties
#: of the fabric a reader would name first, they are decided by the compiler and by each
#: type's own parameters, and both are measurable on the plan that results.
QUALITIES = (
    ("density", (("dense", ("densely", "dense", "crowded", "closely packed",
                            "tightly packed", "packed")),
                 ("sparse", ("sparsely", "sparse", "scattered", "spread out",
                             "widely spaced")))),
    ("height",  (("tall", ("tall", "high", "towering", "many-storeyed",
                           "multi-storey")),
                 ("low", ("low-slung", "low", "squat", "single-storey")))),
)

#: Where a quality is measured, as a share of the range the library itself declares.
#: **Relative to the library and not to a block count on purpose.** "Tall" is not twelve
#: blocks; it is a place whose buildings are built near the top of what their own types
#: can do, and that is a statement the compiler can act on and a check can measure
#: without a constant that means nothing outside one voice. `storeys` is the parameter
#: every residential type declares a band for.
QUALITY_BOUNDS = {"dense": 0.30, "sparse": 0.12, "tall": 0.62, "low": 0.38}

#: Ground the sentence asks for, checked against the site the search chose.
SETTINGS = (("cliff", ("on a cliff", "clifftop", "on the cliffs", "escarpment")),
            ("island", ("on an island", "island")),
            ("mountain", ("in the mountains", "mountainside", "on the mountain")),
            ("valley", ("in a valley", "in the valley")),
            ("forest", ("in the forest", "in a forest", "in the woods")),
            ("river", ("on a river", "beside a river", "by the river", "riverbank")))

#: The form family the library builds each named tradition **nearest** to, and `None`
#: where it has nothing at all. Coarse on purpose and labelled coarse everywhere it is
#: used: `european_vernacular` is the nearest thing on disk to a German village and it
#: is not German, so a tradition whose family is present is `unresolved` and never
#: `satisfied`. A tradition with no family is `unsupported`, which is the same answer
#: the schema already gives a harbour.
TRADITION_FORMS = {
    "japanese": "east_asian", "chinese": "east_asian", "korean": "east_asian",
    "german": "european_vernacular", "bavarian": "european_vernacular",
    "french": "european_vernacular", "italian": "european_vernacular",
    "tuscan": "european_vernacular", "english": "european_vernacular",
    "tudor": "european_vernacular", "norse": "european_vernacular",
    "viking": "european_vernacular", "scandinavian": "european_vernacular",
    "dutch": "european_vernacular", "swiss": "european_vernacular",
    "alpine": "european_vernacular", "medieval": "european_vernacular",
    "renaissance": "european_vernacular", "georgian": "european_vernacular",
    "victorian": "european_vernacular",
    # ...and the traditions this library has no form family for at all.
    "spanish": None, "moorish": None, "ottoman": None, "persian": None,
    "mughal": None, "aztec": None, "mayan": None, "inca": None, "greek": None,
    "roman": None, "russian": None,
}

#: The words a sentence spends on grammar, on the kind of place, on the unit it counts
#: in, and on plain description -- the ones that carry no requirement of their own.
#: Everything **else** a sentence says and no requirement claimed is an unread clause.
#: This list is the one place in the module where being wrong is cheap in one direction
#: and expensive in the other: a word wrongly listed here is a requirement silently
#: lost, which is the whole defect, and a word wrongly left out is a `clause/`
#: obligation a reader has to look at. So it holds function words and nothing that could
#: name a thing. **And nothing that could describe one.** The review found `dense`,
#: `tall`, `sparse`, `roof`, `high` and `low` in this list, and a sentence made of them
#: alone therefore asked for nothing. They are in `QUALITIES` above, or -- `roof` --
#: they fall through to a `clause/` obligation, which is the honest answer for a word no
#: rule here reads.
_ACCOUNTED = frozenset("""
a an the this that these those there here it its their his her our your my
and or but nor plus also then than as so if when while
of in on at by for to from into onto over under near beside along beneath above
around about across through between among amid within without with inside outside
beyond behind before after up down out off per upon toward towards against
is are be been being was were has have had do does did will would can could
should may might must shall
build builds built building make makes made making create creates created creating
lay lays laid set sets put puts place places placed raise raises raised give gives
show shows want wants need needs
village villages town towns city cities hamlet hamlets settlement settlements
capital port city-state metropolis borough
place places district districts quarter quarters neighbourhood neighborhood
ward wards sector sectors area areas region regions zone zones part parts
house houses home homes dwelling dwellings cottage cottages residence residences
building buildings structure structures wall-of
people population inhabitants family families household households
big small large little great greater short wide narrow long deep
old new modest plain simple fine grand humble ordinary typical proper real whole
entire full complete busy quiet open closed
one two three four five six seven eight nine ten eleven twelve dozen
few several many some most all both each every any no none
its it's there's
following follow follows leading led running run runs strung set-out laid-out
exactly precisely just roughly approximately about around-about
looking facing fronting overlooking standing sitting lying
""".split())

#: **Every number word to a hundred, and the compounds.** The closure round's retained
#: first failure: `spec._NUMBER_WORDS` skipped thirteen to nineteen and every compound,
#: so "sixteen low cottages" read as no count at all and the village was sized from the
#: kind's band while the interpretation beside it said sixteen, exactly. A number the
#: reader cannot read is a requirement silently lost, which is the one direction this
#: module may not be wrong in. `spec.count_in` is asked first and this table second.
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_WORDS = dict(spec_mod._NUMBER_WORDS)
NUMBER_WORDS.update(_ONES)
NUMBER_WORDS.update(_TENS)
for _t, _tv in _TENS.items():
    for _o, _ov in list(_ONES.items())[:9]:
        NUMBER_WORDS[f"{_t}-{_o}"] = _tv + _ov
        NUMBER_WORDS[f"{_t} {_o}"] = _tv + _ov
NUMBER_WORDS.update({"a dozen": 12, "dozen": 12, "two dozen": 24, "three dozen": 36,
                     "a hundred": 100, "one hundred": 100})
_NUM = NUMBER_WORDS


def number_in(text: str) -> tuple | None:
    """The first number in `text` as `(n, phrase)`, digits or words, or None."""
    words = "|".join(re.escape(w) for w in sorted(_NUM, key=len, reverse=True))
    m = re.search(rf"(?<![a-z0-9])(\d{{1,3}}(?:,\d{{3}})+|\d{{1,6}}|{words})(?![a-z0-9])",
                  (text or "").lower())
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    return (int(raw) if raw.isdigit() else int(_NUM[raw])), m.group(1)


#: The words a sentence counts its ordinary buildings in.
_COUNT_UNITS = r"(?:houses?|homes?|dwellings?|buildings?|structures?|cottages?|halls?|" \
               r"townhouses?|farmsteads?|huts?|minkas?|residences?)"


def count_of(sentence: str) -> dict | None:
    """The explicit count of structures a sentence names, and what it counts.

        `spec.count_in`'s answer where it has one, and otherwise the same question asked
        with the full number table above -- so `sixteen` reads. Returns
        `{"n", "about", "phrase", "what"}` or None. `what` is the unit word the number
        attaches to (`cottages`), which is what a count is a count *of*: the hall on the
        square is not the seventeenth cottage.
        
    """
    text = (sentence or "").lower()
    got = spec_mod.count_in(sentence)
    words = "|".join(re.escape(w) for w in sorted(_NUM, key=len, reverse=True))
    m = re.search(rf"(\b(?:{'|'.join(spec_mod._ABOUT_WORDS[:-1])})\s+)?"
                  rf"(?<![a-z0-9])(\d{{1,3}}(?:,\d{{3}})+|\d{{1,6}}|{words})(?![a-z0-9])"
                  rf"((?:\s+[a-z\-]+){{0,3}}?)\s+({_COUNT_UNITS})\b", text)
    if got is None and m is None:
        return None
    if m is None:
        return {**got, "what": None}
    raw = m.group(2).replace(",", "")
    n = int(raw) if raw.isdigit() else int(_NUM[raw])
    about = bool(m.group(1)) or "or so" in text or "~" in text
    if got is not None and int(got["n"]) != n:
        # two readers disagree about the number: the regex's is kept and the
        # disagreement is visible in the phrase
        return {**got, "what": m.group(4)}
    return {"n": n, "about": about, "phrase": m.group(0).strip(), "what": m.group(4)}


#: The words a sentence counts a building's floors in.
_STOREY_UNITS = r"(?:storeys?|storeyed|storied|stories|story|floors?)"


def storeys_of(sentence: str) -> dict | None:
    """The **exact** number of storeys a sentence states outright, or None.

        `low` and `tall` are bands -- a share of what each type's own parameters admit --
        and "cottages of two storeys" is not a band at all. It is the same kind of statement
        as "sixteen cottages": an exact number the design has to deliver on every building
        it is about, or refuse. The distinction is the one the expression round could not
        make. Its farm planned sixteen cottages at two storeys, built them at one because
        the lots were 5x5, and the improve stage legitimately adopted a one-storey band --
        legitimately, because the two storeys there were an *inferred* choice under a
        request for `low` cottages. Under this requirement that revision is not available:
        the lot has to grow or the design has to say it cannot.
        
    """
    text = _words(sentence)
    words = "|".join(re.escape(w) for w in sorted(_NUM, key=len, reverse=True))
    m = re.search(rf"(?<![a-z0-9])(\d{{1,2}}|{words})[\s\-]+({_STOREY_UNITS})"
                  rf"(?![a-z0-9])", text)
    if not m:
        return None
    raw = m.group(1)
    n = int(raw) if raw.isdigit() else int(_NUM.get(raw) or 0)
    if not 1 <= n <= 12:
        return None
    return {"n": n, "phrase": m.group(0).strip()}


def _words(sentence: str) -> str:
    return " " + re.sub(r"[^a-z0-9,~\- ]+", " ", (sentence or "").lower()) + " "


def _span(text: str, phrase: str) -> tuple | None:
    m = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
    return (m.start(), m.end()) if m else None


def _has(text: str, phrase: str) -> bool:
    return _span(text, phrase) is not None


def _free(text: str, phrase: str, taken: list) -> tuple | None:
    """The span of `phrase`, unless a longer phrase already claimed those columns.

        "a market square" is one thing and not two: `market square` matches first, and the
        bare `square` that would otherwise match inside it is refused here. Without this the
        sentence would state a requirement it does not state, which is the same class of
        error as losing one.
        
    """
    at = _span(text, phrase)
    if at is None:
        return None
    if any(at[0] < b and a < at[1] for a, b in taken):
        return None
    return at


#: Words that break the link between a number and the thing it counts. "about sixty
#: houses with a market square" is sixty houses and one market, and a window that
#: counted words alone read it as sixty markets -- found by running it.
_BREAKS = frozenset(("with", "and", "or", "plus", "around", "about", "inside", "within",
                     "outside", "beyond", "of", "in", "on", "at", "by", "under", "over",
                     "beside", "near", "the", "a", "an", "its", "their", "houses",
                     "house", "homes", "home", "buildings", "building", "structures",
                     "structure", "dwellings", "dwelling", "cottages", "cottage",
                     "halls", "hall"))


def _count_before(text: str, phrase: str) -> int | None:
    """The number standing immediately before `phrase`, as a multiplicity.

        "three concentric ring walls" is three walls; "a walled town" is one. At most three
        words may stand between, and none of them may be one of `_BREAKS`: a number
        separated from the phrase by "with", "and" or another unit noun is counting
        something else.
        
    """
    words = "|".join(sorted(_NUM, key=len, reverse=True))
    m = re.search(rf"\b(\d{{1,3}}|{words})\b((?:\s+[a-z\-]+){{0,3}})\s+"
                  rf"{re.escape(phrase)}(?![a-z0-9])", text)
    if not m:
        return None
    if any(w in _BREAKS for w in m.group(2).split()):
        return None
    raw = m.group(1)
    return int(raw) if raw.isdigit() else _NUM[raw]


def _absent_phrases(rid: str, phrases) -> tuple:
    """Every way this sentence could say the place has **no** feature `rid`.

        A negation in front of each of the feature's own phrases, longest first, plus the
        words in `ABSENT` that no negation prefix would produce. Longest first matters for
        the same reason it matters in the feature table itself: `without a market square`
        has to be found before `without a square`, or the sentence states a requirement it
        does not state.
        
    """
    out = [f"{n}{p}" for p in phrases for n in NEGATIONS]
    out += list(ABSENT.get(rid) or ())
    return tuple(sorted(set(out), key=len, reverse=True))


def unclaimed_words(sentence: str, claimed: list) -> list:
    """The content words of `sentence` that no requirement's own phrase covers.

        **The independent check.** Every other function in this module walks a table of
        phrases and asks the sentence whether it contains them; this one walks the *sentence*
        and asks the requirements whether they account for it. The two fail differently on
        purpose: a table with a hole in it returns a confident empty answer, and this returns
        the word that fell through it.

        `claimed` is the list of `(start, end)` spans the requirements took out of
        `_words(sentence)`. What comes back is `[(word, span)]` in the order the sentence
        says them, once each.
        
    """
    text = _words(sentence)
    out, seen = [], set()
    for m in re.finditer(r"[a-z][a-z\-]{1,}", text):
        at = (m.start(), m.end())
        if any(at[0] < b and a < at[1] for a, b in claimed):
            continue
        w = m.group(0)
        if w in _ACCOUNTED or w in _NUM or w in seen:
            continue
        seen.add(w)
        out.append((w, at))
    return out


def read(sentence: str) -> dict:
    """The intent record for a sentence: every requirement it states outright.

        Deliberately literal about what it recognises, and **never silent about what it does
        not**. What it finds it finds exactly; what it cannot express it says it cannot
        express; what it cannot even parse becomes a `clause/` obligation rather than
        vanishing. What a sentence merely implies is left to the model's reading, which is a
        different record.
        
    """
    from . import evidence as evidence_mod
    text = _words(sentence)
    reqs: list = []
    #: Every column of the sentence some requirement has accounted for. `taken` is the
    #: longest-phrase-wins window the feature table needs; this is the wider record the
    #: independent check reads, and a phrase enters it whatever kind of requirement it
    #: belongs to.
    claimed: list = []

    def add(rid, says, kind, wants, *, hard=True, phrase=None, status="open", why=""):
        reqs.append({"id": rid, "says": says, "kind": kind, "wants": wants,
                     "hard": bool(hard), "source": "sentence", "phrase": phrase,
                     "status": status, "why": why, "evidence": [], "owner": None})
        if phrase:
            at = _span(text, str(phrase).lower())
            if at is not None:
                claimed.append(at)

    # 1. how many structures. `spec.count_in` is the one reader of this and stays so:
    # "about sixty houses" has to be the same band wherever it is asked.
    count = count_of(sentence)
    if count:
        add("count/structures",
            f"{'about ' if count['about'] else 'exactly '}{count['n']} "
            f"{count.get('what') or 'structures'}",
            "count", {"n": int(count["n"]), "about": bool(count["about"]),
                      **({"what": count["what"]} if count.get("what") else {})},
            phrase=count["phrase"],
            hard=True)

    # 2. what the place must, and must not, contain
    taken: list = []
    for rid, family, phrases in FEATURES:
        absent_words = _absent_phrases(rid, phrases)
        if any(_has(text, w) for w in absent_words):
            w0 = next(w for w in absent_words if _has(text, w))
            taken.append(_span(text, w0))
            add(f"absent/{rid}", f"no {rid}: the sentence says so outright", "absent",
                {"family": family, "feature": rid}, phrase=w0)
            continue
        hit, at = None, None
        for p in phrases:
            at = _free(text, p, taken)
            if at is not None:
                hit = p
                break
        if hit is None:
            continue
        taken.append(at)
        if family is None:
            add(f"feature/{rid}", f"a {rid}", "capability",
                {"feature": rid, "family": None}, phrase=hit,
                status="unsupported",
                why=f"the sentence asks for a {rid} and this schema has no family for "
                    f"one; it is neither planned nor substituted")
            continue
        n = _count_before(text, hit) or 1
        add(f"feature/{rid}", f"{n} x {rid}" if n > 1 else f"a {rid}", "feature",
            {"family": family, "feature": rid, "count": int(n)}, phrase=hit)

    # 3. how the whole place is organised
    for name, supported, phrases in LAYOUTS:
        hit = next((p for p in phrases if _has(text, p)), None)
        if hit is None:
            continue
        add(f"layout/{name}", f"laid out {name}", "layout",
            {"policy": name, "supported": bool(supported)}, phrase=hit,
            status="open" if supported else "unsupported",
            why="" if supported else f"no layout policy in this build lays a place out "
                                     f"{name}")

    # 4. what the buildings face
    for what, phrases in FACING:
        hit = next((p for p in phrases if _has(text, p)), None)
        if hit is not None:
            add(f"facing/{what}", f"the buildings face the {what}", "orientation",
                {"faces": what}, phrase=hit)

    # 5. the ground it asks for. **Hard, and measured against the site that was
    # chosen.** It was soft and permanently `unresolved`, which is a requirement that
    # can never be missed: "Build a village on a cliff." held while the village stood on
    # flat ground. A site record carries its own relief, water and biome, so this is a
    # measurement and not a shrug.
    for what, phrases in SETTINGS:
        hit = next((p for p in phrases if _has(text, p)), None)
        if hit is not None:
            add(f"setting/{what}", f"the place stands {hit}", "setting",
                {"setting": what}, phrase=hit, hard=True)

    # 5b. **what the buildings are like.** The fabric's density and height are decided
    # by the compiler and by each type's own parameters, and both are measurable on the
    # plan. See `QUALITIES`.
    for axis, values in QUALITIES:
        hit = None
        for value, phrases in values:
            hit = next(((value, p) for p in phrases if _has(text, p)), None)
            if hit:
                break
        if hit is None:
            continue
        value, phrase = hit
        add(f"quality/{axis}/{value}", f"the fabric is {value}", "quality",
            {"axis": axis, "value": value, "bound": QUALITY_BOUNDS[value]},
            phrase=phrase, hard=True)

    # 5c. **an exact storey count is a number, not a band.** See `storeys_of`. It takes
    # the same `height` axis, so `envelope.required_features` already makes it the
    # `storeys` token every envelope query must carry, and it is measured as an exact
    # count on what construction emitted rather than as a share of a band.
    st = storeys_of(sentence)
    if st:
        add(f"quality/height/storeys/{st['n']}",
            f"the buildings stand {st['n']} storey(s), exactly", "quality",
            {"axis": "height", "value": "storeys", "storeys": int(st["n"]),
             "exact": True},
            phrase=st["phrase"], hard=True,
            why="an exact storey count is delivered or refused; it is not a band a "
                "revision may adopt its way out of")

    # 6. **who the place is, and how it is built.** Neither is a part the schema can
    # place, and both were being dropped whole. `evidence.classify` already tells a
    # named place from a named tradition from a description, by rule, and the
    # classification is the same one the reading stage goes looking for sources with.
    what = evidence_mod.classify(sentence)
    if what.get("name"):
        add(f"identity/{_slug(what['name'])}",
            f"the place is {what['name']}, and is recognisable as it", "identity",
            {"name": what["name"]}, phrase=what["name"].lower(), hard=True,
            why="a named place is a fidelity obligation from the moment it is named")
    if what.get("tradition"):
        t = what["tradition"]
        form = TRADITION_FORMS.get(t, None)
        add(f"tradition/{t}", f"the buildings are built in the {t} tradition",
            "tradition", {"tradition": t, "nearest_form": form}, phrase=t,
            hard=True,
            status="open" if form else "unsupported",
            why="" if form else (f"this library has no form family for {t} construction "
                                 f"and will not substitute one it has"))

    # 7. **everything the sentence says that nothing above claimed.** The independent
    # check, and the only requirement here that is not generated from a table of things
    # somebody thought of. `bakery` and `aqueduct` come out here.
    for w, at in unclaimed_words(sentence, claimed):
        reqs.append({"id": f"clause/{_slug(w)}", "says": f"the sentence says {w!r}",
                     "kind": "clause", "wants": {"word": w},
                     "hard": True, "source": "sentence", "phrase": w,
                     "status": "unresolved", "evidence": [], "owner": "reading",
                     "why": ("no rule in this reader claims this word, so it is carried "
                             "as an unread clause rather than dropped; whatever it asks "
                             "for is neither planned nor refused")})

    return contracts.make("intent", sentence=str(sentence), requirements=reqs,
                          note="read from the sentence alone, by rule, with no model "
                               "call and no reference to any spec; every content word "
                               "the rules did not claim is carried as a `clause/`")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "x"


# ------------------------------------------------------------------ the checking

def _landmark_family(lm: dict) -> str | None:
    """The family a character's landmark is, by its type's declaration or its name."""
    from . import capability as cap_mod
    t = str((lm or {}).get("type") or "")
    if not t:
        return None
    decl = _plan_decls([{"type": t}]).get(t) or {}
    if decl.get("family"):
        return str(decl["family"])
    for fam in spec_mod.FAMILIES:
        if cap_mod._named_for(t, fam):
            return fam
    return None


def _families_in(spec: dict) -> dict:
    """`{family: total count declared}` over the spec's defining parts **and the
        landmarks their characters declare**.

        A smithy the spec puts in a district's character (`landmarks: [{"type":
        "workshop"}]`) is declared as surely as a defining part is, and the compiler lays
        it; the held-out hamlet failed `feature/workshop` at the spec stage for a workshop
        the spec had declared one level down.
        
    """
    out: dict = {}
    for p in (spec or {}).get("defining_parts") or []:
        out[p["family"]] = out.get(p["family"], 0) + int(p.get("count") or 1)
        for lm in ((p.get("character") or {}).get("landmarks") or []):
            fam = _landmark_family(lm)
            if fam:
                out[fam] = out.get(fam, 0) + 1
    return out


def _standing(parts: list, parts_record: dict | None) -> dict:
    """`{part name: did it stand}` -- empty where nothing has been built yet."""
    if not parts_record:
        return {}
    out = {}
    for w in (parts_record or {}).get("waves", []):
        for r in w.get("parts", []):
            out[r["part"]] = bool(r.get("stood", r.get("status") == "built"))
    return out


def _leaves_of_family(spec: dict, parts: list, family: str) -> list:
    """The plan leaves that answer a defining part of `family`.

        By the defining part the leaf says it answers, which is the only link that survives
        a planner renaming things: the place level writes `defines` on every leaf it lays.
        
    """
    names = {p["name"] for p in (spec or {}).get("defining_parts") or []
             if p["family"] == family}
    if not names:
        return []
    out = []
    for p in parts:
        d = p.get("defines")
        if d in names or (p.get("compound") in names):
            out.append(p)
            continue
        if any(p.get("name", "") == n or p.get("name", "").startswith(n + "_")
               for n in names):
            out.append(p)
    return out


def leaves_built_as(parts: list, family: str, decls: dict | None = None) -> list:
    """The plan leaves that **are** a `family`, whatever any spec declares.

        The integration review's second finding, and it is a one-line defect with a wide
        blast radius: `_leaves_of_family` starts from the families the *spec* declares, so
        when the question is "is there a wall the spec never mentioned?" the answer is
        computed from a list that cannot contain one. An absence check generated from a
        document cannot see what the document omits -- which is the exact sentence this
        module's own docstring opens with, applied one level further down.

        So this reads the leaf: its own `family` where it carries one, then the `FAMILY` its
        type declares, then the name the type is committed under (`keep`, `keep_*`), then
        the name the planner gave the part. Nothing here consults the spec at all.
        
    """
    from . import capability as cap_mod
    out = []
    for p in parts:
        said = p.get("family")
        if said:
            if str(said) == family:
                out.append(p)
            continue
        tname = str(p.get("type") or "")
        decl = (decls or {}).get(tname) or {}
        if decl.get("family"):
            if str(decl["family"]) == family:
                out.append(p)
            continue
        if tname and cap_mod._named_for(tname, family):
            out.append(p)
            continue
        if cap_mod._named_for(str(p.get("name") or ""), family):
            out.append(p)
    return out


def _plan_decls(parts: list) -> dict:
    """`{type name: declaration}` for the types this plan actually uses, or `{}`.

        Off disk and cached by the type loader; a plan naming a type this checkout does not
        have simply contributes nothing, because a missing declaration must never turn into
        a confident answer about what a part is.
        
    """
    from . import pipeline as _pipeline
    import os
    out: dict = {}
    for name in sorted({str(p.get("type") or "") for p in parts if p.get("type")}):
        path = os.path.join(_pipeline.ROOT, "types", f"{name}.py")
        try:
            out[name] = _pipeline.load_type(path)
        except Exception:                 # noqa: BLE001 -- absent is not an answer
            continue
    return out


def _mentions(spec: dict | None, word: str) -> list:
    """Where, if anywhere, the spec's own prose says `word`. Prose, not proof."""
    out = []
    for p in (spec or {}).get("defining_parts") or []:
        blob = " ".join(str(p.get(k) or "") for k in ("name", "family", "notes",
                                                      "purpose"))
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", blob.lower()):
            out.append(p.get("name") or "?")
    for k in ("invariants", "notes"):
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])",
                     str((spec or {}).get(k) or "").lower()):
            out.append(f"spec.{k}")
    return out


def _anchor_of(resolution: dict | None, plan: dict | None) -> dict:
    """The shore anchor, with its path, from wherever the run recorded it."""
    b = ((resolution or {}).get("bounds") or {}).get("anchor")
    if isinstance(b, dict) and b.get("path"):
        return b
    lay = (plan or {}).get("layout") or {}
    if lay.get("anchor"):
        a = dict(lay["anchor"])
        if lay.get("anchor_path"):
            a["path"] = lay["anchor_path"]
        return a
    for r in (resolution or {}).get("regions") or []:
        a = r.get("anchor")
        if isinstance(a, dict) and a.get("kind") == "shoreline":
            return a
    return {}


def _rect_to_path(rect, path) -> float:
    """The least distance from rectangle `rect` to any column of `path`, in columns."""
    x0, z0, x1, z1 = [float(v) for v in rect]
    best = float("inf")
    for p in path:
        px, pz = float(p[0]), float(p[1])
        dx = max(x0 - px, 0.0, px - x1)
        dz = max(z0 - pz, 0.0, pz - z1)
        best = min(best, (dx * dx + dz * dz) ** 0.5)
    return best


def _policy_geometry(policy: str, resolution: dict | None, plan: dict | None) -> tuple:
    """Does the resolved design carry the geometry its policy's name promises?

        `(ok, why)`. One clause per policy and each one is a **measurement on the columns
        the design occupies**, which is the thing the review found missing in both:

          `shoreline`   there is a shore anchor derived from the ground, with a path, and
                        each district actually stands within the band behind it. The old
                        clause asked only that a path and a region existed, and certified a
                        district a million columns offshore.
          `concentric`  the rings carry real radii and those radii nest. The old clause read
                        `half`, which the city's own rings do not record -- they record
                        `inner` and `outer` -- and a missing value was silently skipped, so
                        two empty dicts named `one` and `two` read as a concentric city.

        A policy nothing here knows how to measure says so **and does not certify itself**:
        inventing a pass is what this function exists to stop.
        
    """
    if policy == "shoreline":
        anchor = _anchor_of(resolution, plan)
        path = [p for p in (anchor.get("path") or []) if p is not None]
        if not path:
            return (False, "the design records no shoreline path; a shoreline policy "
                           "that cannot say where the shore is has not laid out "
                           "against one")
        regions = [r for r in (resolution or {}).get("regions") or []
                   if r.get("lots") is not None and r.get("rect")]
        if not regions:
            return (False, "no region of the design is a district of the shore band")
        depth = ((resolution or {}).get("bounds") or {}).get("band_depth")
        far = []
        for r in regions:
            x0, z0, x1, z1 = [float(v) for v in r["rect"]]
            # **How deep a shore district may be is how long it is.** A band district
            # runs along the water and is shallower than it is long; where the design
            # recorded its own band depth that number is used instead, because it is the
            # promise the layout made. Either way the bound is the design's own and not
            # a constant invented here.
            reach = float(depth) if depth else max(x1 - x0, z1 - z0)
            gap = _rect_to_path([x0, z0, x1, z1], path)
            if gap > max(reach, 1.0):
                far.append((r["name"], round(gap), round(reach)))
        if far:
            return (False, "these district(s) are not in the band behind the shore: "
                           + "; ".join(f"{n} stands {g} columns from the shore path "
                                       f"against a band of {b}" for n, g, b in far[:4]))
        return (True, f"a shore path of {len(path)} points read off the ground, and "
                      f"{len(regions)} district(s) measured into the band behind it")
    if policy == "concentric":
        rings = (((resolution or {}).get("bounds") or {}).get("rings") or [])
        if not rings:
            rings = ((plan or {}).get("layout") or {}).get("rings") or []
        if len(rings) < 2:
            return (False, f"a concentric place is rings inside rings and this design "
                           f"has {len(rings)}")
        radii = []
        for r in rings:
            if not isinstance(r, dict):
                radii.append(None)
                continue
            lo, hi = r.get("inner"), r.get("outer", r.get("half"))
            radii.append((float(lo), float(hi))
                         if isinstance(lo, (int, float))
                         and isinstance(hi, (int, float)) else None)
        blank = [r.get("name", "?") if isinstance(r, dict) else "?"
                 for r, got in zip(rings, radii) if got is None]
        if blank:
            # **A missing radius is not a nesting ring.** It used to be skipped, so a
            # design whose rings recorded nothing at all passed on the strength of there
            # being two of them.
            return (False, f"{len(blank)} of {len(rings)} rings record no radii "
                           f"({', '.join(str(b) for b in blank[:4])}), so whether they "
                           f"nest is not something this design says")
        bad = [(rings[i].get("name", i), radii[i], radii[i + 1])
               for i in range(len(radii) - 1)
               if not (radii[i][0] < radii[i][1] <= radii[i + 1][0] < radii[i + 1][1])]
        if bad:
            return (False, "the rings do not nest: "
                           + "; ".join(f"{n} spans {a[0]:g}-{a[1]:g} and the ring "
                                       f"outside it spans {b[0]:g}-{b[1]:g}"
                                       for n, a, b in bad[:3]))
        return (True, f"{len(rings)} rings, measured from {radii[0][0]:g} to "
                      f"{radii[-1][1]:g} columns and each inside the next")
    return (False, f"this build has no geometric test for the `{policy}` policy, so "
                   f"whether the design is laid out that way is not established; the "
                   f"policy's own record is a word it wrote about itself")


#: What "the buildings face the water" is, as a measurement, registered here rather than
#: chosen after looking at a run. **Both numbers are the review's, and the second is the
#: one that was missing.** The implemented rule was "no home fronts away", which
#: certified a village whose every home fronted *along* the shore and none at the water
#: -- the absence of the opposite condition is not the condition. So a majority of the
#: homes must actually front the water, and a plan that cannot say which way most of its
#: homes front has not answered the question at all and says `unresolved` rather than
#: counting the silence as a pass.
WATER_FACING_SHARE = 0.5
MEASURED_SHARE_MIN = 0.75

#: **How much of a place has to be built in a tradition before it is built in it.**
#: Registered before any run measured it. Half: a place where most of what a person
#: walks past is in the named tradition is built in it, and one labelled building among
#: a hundred is a building in that tradition standing in a place that is not.
TRADITION_SHARE = 0.5

#: How far out from a lot's front side a street may be and still be its street, and how
#: far off a lane's centre line the lot may stand. A doorstep opens onto a lane it can
#: reach; `clearance` on the types that declare one is 2 columns, and a lane is drawn at
#: its own width.
STREET_REACH = 8
STREET_SLACK = 2


def _seg_distance(px: float, pz: float, a, b) -> float:
    """Distance from a column to the segment `a`--`b`."""
    ax, az, bx, bz = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    dx, dz = bx - ax, bz - az
    span = dx * dx + dz * dz
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / span))
    return ((px - ax - t * dx) ** 2 + (pz - az - t * dz) ** 2) ** 0.5


def _streets_in(plan: dict | None, parts: list) -> list:
    """The lanes and roads of a plan: `(path, width)` per run of street.

        A street is an `edge` leaf with a path -- which is what `circulate` routes and what
        `district_compile` lays a block's lane as -- plus whatever the plan records as its
        own `streets`. Nothing about a *plot* is a street: the review's counterexample read
        a `front` field on a lot and called it street frontage with no street in the plan.
        
    """
    from . import pipeline as _pipeline
    rows = []
    seen = set()
    for p in [*(parts or []), *(_pipeline.plan_parts(plan or {}) if plan else [])]:
        if p.get("kind") != "edge" or not p.get("path"):
            continue
        key = (p.get("name"), len(p["path"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(([[float(c[0]), float(c[1])] for c in p["path"]],
                     float(p.get("width") or 3)))
    for run in ((plan or {}).get("streets") or []):
        path = run.get("path") if isinstance(run, dict) else run
        if path:
            rows.append(([[float(c[0]), float(c[1])] for c in path],
                         float((run or {}).get("width") or 3)
                         if isinstance(run, dict) else 3.0))
    return rows


#: The column just outside a lot's front side, per side of the world.
_OUTWARD = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


def _fronts_a_street(p: dict, streets: list) -> bool:
    """Does lot `p` have a street outside the side its door is in?"""
    side = p.get("front")
    step = _OUTWARD.get(str(side))
    if step is None:
        return False
    try:
        x0, z0, x1, z1 = (float(p["x0"]), float(p["z0"]), float(p["x1"]),
                          float(p["z1"]))
    except (KeyError, TypeError, ValueError):
        return False
    mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
    ex = x0 if side == "west" else x1 if side == "east" else mx
    ez = z0 if side == "north" else z1 if side == "south" else mz
    for out in range(0, STREET_REACH + 1):
        cx, cz = ex + step[0] * out, ez + step[1] * out
        for path, width in streets:
            reach = width / 2 + STREET_SLACK
            if len(path) == 1:
                if _seg_distance(cx, cz, path[0], path[0]) <= reach:
                    return True
                continue
            if any(_seg_distance(cx, cz, a, b) <= reach
                   for a, b in zip(path, path[1:])):
                return True
    return False


def _fronts(faces: str, resolution: dict | None, parts: list,
            plan: dict | None = None) -> dict:
    """Which way the plan's homes actually front, against what the sentence asked.

        **The review's sharpest counterexamples, and there were three.** The saved shoreline
        village recorded `resolution.bounds.faces == "water"`, the check read that field, the
        requirement passed -- and nine of fifteen homes had their doors in the wall away from
        the water. That one was fixed. The two that replaced it were subtler and are the
        reason this function now counts rather than looks for an absence: a village whose
        every home fronted *along* the shore passed, because none fronted away from it; and
        one home with a `front` field and no street anywhere in the plan passed "facing the
        street", because a lot's own field was the whole of the evidence.

        So: a majority of the homes must front the water beside them, most of them must be
        measurable at all, and a street must be a street that is in the plan.
        
    """
    from . import placeshore
    plots = [p for p in parts if p.get("kind", "plot") == "plot"]
    if not plots:
        return {"status": "open", "why": "nothing is planned yet", "evidence": {}}
    if faces != "water":
        streets = _streets_in(plan, parts)
        if not streets:
            return {"status": "unresolved",
                    "why": (f"this plan holds no street, lane or road, so whether "
                            f"{len(plots)} lot(s) front one is not a question it can "
                            f"answer; a lot's own `front` field is not a street"),
                    "evidence": {"plots": len(plots), "streets": 0}}
        on = [p for p in plots if _fronts_a_street(p, streets)]
        unknown = [p for p in plots if not p.get("front")]
        share = len(on) / float(len(plots))
        if len(plots) - len(unknown) < MEASURED_SHARE_MIN * len(plots):
            return {"status": "unresolved",
                    "why": (f"{len(unknown)} of {len(plots)} lots record no front, so "
                            f"street frontage is measured on too few of them to answer"),
                    "evidence": {"plots": len(plots), "unknown": len(unknown),
                                 "on_a_street": len(on)}}
        if share < WATER_FACING_SHARE:
            return {"status": "failed",
                    "why": (f"{len(on)} of {len(plots)} lots have a street outside the "
                            f"side their door is in, measured against "
                            f"{len(streets)} run(s) of street in the plan"),
                    "evidence": {"plots": len(plots), "on_a_street": len(on),
                                 "streets": len(streets),
                                 "examples": [p.get("name") for p in plots
                                              if p not in on][:6]}}
        return {"status": "satisfied",
                "why": (f"{len(on)} of {len(plots)} lots front one of "
                        f"{len(streets)} run(s) of street, measured from the side each "
                        f"door is in"),
                "evidence": {"sample": [p["name"] for p in on[:8]],
                             "on_a_street": len(on), "streets": len(streets)}}
    anchor = _anchor_of(resolution, plan)
    if not anchor.get("path"):
        return {"status": "unresolved",
                "why": ("the design records no shore path, so which way the water lies "
                        "from each home is not a question this plan can answer"),
                "evidence": {"plots": len(plots)}}
    got = placeshore.fronts_facing_water(anchor, plots)
    away = [r for r in got["rows"] if r["class"] == "away"]
    unknown, toward, along = got["unknown"], got["toward"], got["along"]
    measured = toward + along + len(away)
    if away:
        return {"status": "failed",
                "why": (f"{len(away)} of {len(plots)} homes front away from the water "
                        f"beside them -- {', '.join(str(r['part']) for r in away[:4])}"
                        f" -- against {toward} that face it"),
                "evidence": {"away": len(away), "toward": toward, "along": along,
                             "unknown": unknown,
                             "examples": [{k: r[k] for k in ("part", "front", "water")}
                                          for r in away[:6]]}}
    if measured < MEASURED_SHARE_MIN * len(plots):
        return {"status": "unresolved",
                "why": (f"{unknown} of {len(plots)} homes record no front this build "
                        f"can measure, so whether the place faces the water is neither "
                        f"met nor missed on {measured} home(s)"),
                "evidence": {"unknown": unknown, "toward": toward, "along": along,
                             "measured": measured, "plots": len(plots)}}
    if toward < WATER_FACING_SHARE * len(plots):
        return {"status": "failed",
                "why": (f"{toward} of {len(plots)} homes front the water beside them "
                        f"and {along} front along the shore; the sentence asks for a "
                        f"place whose homes face the water, and no home fronting away "
                        f"from it is a weaker statement than the one it makes"),
                "evidence": {"toward": toward, "along": along, "away": 0,
                             "unknown": unknown, "share": round(
                                 toward / float(len(plots)), 3),
                             "bound": WATER_FACING_SHARE}}
    return {"status": "satisfied",
            "why": (f"{toward} of {len(plots)} home(s) front the water beside them and "
                    f"{along} front along the shore; none fronts away"
                    + (f"; {unknown} record no front" if unknown else "")),
            "evidence": {"sample": [str(r["part"]) for r in got["rows"][:8]
                                    if r["class"] == "toward"],
                         "toward": toward, "along": along, "unknown": unknown}}


def _word_matches(p: dict, what: str) -> bool:
    """Is leaf `p` one of the things the sentence calls `what`?

        By the leaf's own family, the family its type declares, the type's committed name and
        the planner's name for it -- the same ladder `leaves_built_as` walks, plus the plain
        English words a sentence uses for a dwelling, because "small houses around a big
        temple" says `houses` and the library calls them `cottage` and `townhouse`.
        
    """
    from . import capability as cap_mod
    want = _slug(what)
    if not want:
        return False
    if want in _HOUSE_WORDS:
        fam = str(p.get("family") or "")
        if fam and fam not in ("district", "square"):
            return fam in ("house", "dwelling") or cap_mod._named_for(
                str(p.get("type") or ""), "house")
        if str(p.get("kind", "plot")) != "plot":
            return False
        # **A house is a type that declares the dwelling function**, where the type is
        # on disk to ask. A hall answers the same rural role and is not a house; the
        # closure round found `select(parts, "cottages")` returning the hall on the
        # square, which would have made it the seventeenth cottage. A type this checkout
        # cannot load keeps the old rule -- a plot that is not a civic or defensive
        # family -- so a recorded plan still reads.
        decl = _plan_decls([p]).get(str(p.get("type") or ""))
        if decl is not None:
            return str(decl.get("function") or "") == "dwelling" or cap_mod._named_for(
                str(p.get("type") or ""), "house")
        return not any(
            cap_mod._named_for(str(p.get("type") or ""), f)
            for f in ("temple", "palace", "keep", "wall", "gate", "tower", "monument",
                      "hall", "workshop", "shop"))
    # **The word the sentence uses, and the family the library builds it as.** A
    # sentence says `market`; `FEATURES` says a market is built as a `square`; the
    # plan's leaf carries the family and never the word. Without this the houses-around-
    # the-market relation found nothing to be around and reported itself unmeasurable --
    # which is honest and useless, and hides the difference between "the square is not
    # there" and "the checker does not know what a market is called here". **A plural is
    # the same word.** A sentence counts `townhouses` and the library's type is
    # `townhouse`; `cottages` only worked because both forms happen to be in
    # `_HOUSE_WORDS` above. Worker A found it from the other side -- `_parts_for_word`
    # had the same hole -- and it cost the held-out village its whole count clause:
    # `select('townhouses')` found none of thirteen standing townhouses, while
    # `select('townhouse')` and `select('houses')` each found them all. English plurals
    # are not a vocabulary to be maintained; the singular of the word that was asked is
    # tried where the word itself finds nothing.
    fams = {want} | {f for rid, f, _ph in FEATURES if rid == want and f}
    if want.endswith("es") and len(want) > 4:
        fams.add(want[:-2])
    if want.endswith("s") and len(want) > 3:
        fams.add(want[:-1])
    # **A whole token of the slug, not a prefix.** `royal_palace` answers `palace`. A
    # token is a word of the name, so `palace_row` still answers `palace` here and the
    # compound rule in `select` is what refuses a row of houses named for its position.
    for got in (p.get("family"), p.get("type"), p.get("name"), p.get("defines"),
                p.get("answers"), p.get("compound")):
        if got and _token_hit(got, fams):
            return True
    return any(cap_mod._named_for(str(p.get("type") or ""), f) or
               cap_mod._named_for(str(p.get("name") or ""), f) for f in fams)


def _token_hit(value, fams) -> bool:
    """Is any of `fams` a whole token of `value`'s slug, or the slug itself?"""
    slug = _slug(str(value))
    toks = set(slug.split("_"))
    return slug in fams or any(f in toks or slug.startswith(f + "_") for f in fams)


#: The words a sentence uses for a **great thing** built as a compound of parts. A
#: compound's leaves -- its halls, courts, wall and gates -- answer the word together,
#: as one rectangle, because "the city stands around the palace" is about the palace and
#: not about one of its halls.
_COMPOUND_WORDS = frozenset(("palace", "monument", "keep", "castle", "citadel",
                             "fortress", "compound", "monastery", "cathedral"))


#: The words a sentence uses for the whole place as the subject of a relation.
_PLACE_WORDS = frozenset(("city", "town", "village", "hamlet", "settlement", "place",
                          "fabric", "quarters", "districts", "rings", "buildings"))


#: The words a sentence uses for "the ordinary buildings people live in". The library
#: names them `cottage`, `townhouse`, `minka`; a sentence never does.
_HOUSE_WORDS = frozenset(("house", "houses", "home", "homes", "dwelling", "dwellings",
                          "cottage", "cottages", "residence", "residences"))


def _rect_in(r, rect) -> bool:
    """Does the leaf rectangle `r` stand inside the region rectangle `rect`?

        Whole or not at all, the same rule the section sampler uses: a leaf that straddles a
        district boundary belongs to neither, and counting half of it into a street
        measurement would be counting a building that is not on that street.
        
    """
    if not r or not rect or len(r) < 4 or len(rect) < 4:
        return False
    x0, x1 = min(rect[0], rect[2]), max(rect[0], rect[2])
    z0, z1 = min(rect[1], rect[3]), max(rect[1], rect[3])
    return bool(x0 <= r[0] and r[2] <= x1 and z0 <= r[1] and r[3] <= z1)


def _in_scope(p: dict, scope: str | None) -> bool:
    """Is this leaf inside the part of the place a scoped requirement is about?

        `None` is the whole place. Otherwise the scope is matched against the leaf's
        ancestry -- the group names `plan_parts` carries on every leaf -- and against the
        defining part it answers, so "the lower district" selects the leaves of the district
        whose name or purpose says `lower`.
        
    """
    if not scope:
        return True
    want = [w for w in _slug(scope).split("_") if w and w not in
            ("the", "a", "an", "district", "quarter", "region", "part", "of")]
    if not want:
        return True
    blob = "_".join(_slug(str(x)) for x in
                    [*(p.get("in") or []), p.get("defines") or "", p.get("answers") or "",
                     p.get("name") or ""])
    return all(w in blob for w in want)


# ------------------------------------------------------- I1: the shared selectors The
# closure round's first interface. Generation and checking used to resolve the words of
# a sentence to parts of a place by two different rules -- the checker by
# `_word_matches` and `_in_scope`, the layout by whatever it had to hand -- and a
# relation the checker measured on one set of leaves was laid out over another. One
# selector, called by both, so a disagreement between them is impossible rather than
# merely unlikely.

#: The scope words that name a division of the place rather than a thing in it.
_SCOPE_WORDS = frozenset(("district", "quarter", "ring", "region", "zone", "belt",
                          "sector", "ward", "neighbourhood", "neighborhood"))
#: The words a sentence uses for working land, and the leaf families that are it.
_FIELD_WORDS = frozenset(("field", "fields", "farmland", "farms", "farm", "cropland",
                          "arable", "paddies", "paddy", "orchard", "orchards",
                          "pasture", "pastures"))
_FIELD_TYPES = ("field", "orchard", "pasture", "paddy")


def _is_field_leaf(p: dict, use: str | None = None) -> bool:
    """An area leaf that is working land: built as a field family, or -- the expression
    round's entity binding -- an area whose defining part is land of the use asked for
    (an orchard is laid as a `grove` and is an orchard by what it answers)."""
    from . import capability as cap_mod
    t = str(p.get("type") or "")
    if str(p.get("kind") or "") != "area":
        return False
    if any(cap_mod._named_for(t, f) for f in _FIELD_TYPES):
        return True
    # the leaf's own land use (stamped by the compiler from its district), then the
    # defining part it answers, then its ancestry -- `in: [district, defines]`
    got = None
    for name in [p.get("land_use"), p.get("defines"), p.get("answers"),
                 *reversed(list(p.get("in") or [])), p.get("name")]:
        if not name:
            continue
        got = spec_mod.land_use_of({"name": str(name), "family": p.get("family")})
        if got:
            break
    return got is not None and (use is None or got == use or
                                (use == "farmland" and got in ("orchard", "pasture")))


def _land_word(word: str) -> str | None:
    """The land use a sentence word names, through the spec's one vocabulary."""
    return spec_mod.land_use_of({"name": word})


def select(parts: list, word: str) -> list:
    """**The one selector.** The leaves of a plan a sentence word is about.

        `cottages`, `houses`, `homes` are the dwellings; `market` is what `FEATURES` builds
        a market as; `hall`, `temple`, `wall` are read by family, declared family, committed
        type name and the planner's name; `fields`/`farmland` are the field areas (and, for
        a district-level question, the leaves inside a farmland district); a phrase naming
        a division of the place -- `upper quarter`, `the lower district`, `farm belt` --
        selects by ancestry. Case, articles and plurals do not matter. Empty where the word
        names nothing in this plan, which is `unresolved` and never a pass.
        
    """
    parts = list(parts or [])
    w = _slug(word)
    if not w:
        return []
    words = [x for x in w.split("_") if x and x not in ("the", "a", "an", "of", "its")]
    if not words:
        return []
    if any(x in _SCOPE_WORDS for x in words) and len(words) > 1:
        return [p for p in parts if _in_scope(p, word)]
    if len(words) == 1 and words[0] in _FIELD_WORDS:
        use = _land_word(words[0])
        got = [p for p in parts if _is_field_leaf(p, use)]
        if got:
            return got
        # a farmland district drawn before its fields are: its own leaves
        return [p for p in parts if any(_slug(str(a)) in ("fields", "farmland", "farms")
                                        or _slug(str(a)).startswith(("field_", "farm"))
                                        for a in (p.get("in") or []))
                or _slug(str(p.get("defines") or "")) in ("fields", "farmland", "farms")]
    if len(words) == 1 and words[0] in _COMPOUND_WORDS:
        return _compound_groups(parts, words[0])
    if len(words) == 1 and words[0] in _PLACE_WORDS:
        # "the city stands around the palace": the city is its fabric -- every plot
        # outside any compound -- and not the palace it stands around
        return [p for p in parts if p.get("kind", "plot") == "plot"
                and not p.get("compound")]
    if len(words) == 1:
        return [p for p in parts if _word_matches(p, words[0])]
    # a multi-word phrase that is not a scope: the last word is the thing, the rest
    # qualify it by ancestry where they can ("farm cottages", "temple halls")
    head = words[-1]
    got = [p for p in parts if _word_matches(p, head)]
    narrowed = [p for p in got if _in_scope(p, " ".join(words[:-1]))]
    return narrowed or got


def _compound_groups(parts: list, word: str) -> list:
    """The leaves answering a compound word, **grouped by compound** into one leaf each.

        A leaf answers where its `compound` carries the word as a token, or its family or
        type is the word, or -- for a leaf outside any compound -- its `answers`/`defines`
        carries the word and it is not a dwelling: a `palace_row` district of cottages is
        named for where it stands and is not a palace. Each compound becomes one record
        with the union of its leaves' rectangles, so a relation's object is the compound
        and a hierarchy compares the compound's footprint, not one hall's.
        
    """
    fams = {word} | {f for rid, f, _ph in FEATURES if rid == word and f}
    by_comp: dict = {}
    loose = []
    for p in parts:
        comp = p.get("compound")
        if comp and _token_hit(comp, fams):
            by_comp.setdefault(str(comp), []).append(p)
            continue
        if any(_token_hit(g, fams) for g in (p.get("family"), p.get("type")) if g):
            (by_comp.setdefault(str(comp), []) if comp else loose).append(p)
            continue
        if comp:
            continue
        decl = _plan_decls([p]).get(str(p.get("type") or ""))
        dwelling = decl is not None and str(decl.get("function") or "") == "dwelling"
        if not dwelling and any(_token_hit(g, fams)
                                for g in (p.get("answers"), p.get("defines"))
                                if g):
            loose.append(p)
    out = []
    for comp, leaves in by_comp.items():
        rects = [r for r in (_rect_of_leaf(q) for q in leaves) if r is not None]
        rec = {"name": comp, "kind": "compound", "compound": comp, "answers": comp,
               "family": word, "leaves": [q.get("name") for q in leaves],
               "in": list((leaves[0].get("in") or [])) if leaves else []}
        if rects:
            rec.update(x0=min(r[0] for r in rects), z0=min(r[1] for r in rects),
                       x1=max(r[2] for r in rects), z1=max(r[3] for r in rects))
        out.append(rec)
    return out + loose


def _region_as_leaf(r: dict) -> dict:
    """A place-level record in the shape `_word_matches` and `_in_scope` read."""
    out = dict(r)
    out.setdefault("name", r.get("name"))
    if "kind" not in out:
        out["kind"] = "district" if r.get("structures") is not None else "plot"
    if not out.get("in"):
        out["in"] = [r.get("defines") or r.get("name") or ""]
    return out


def select_regions(place: dict, word: str) -> list:
    """`select`, at the place level, before a plan exists.

        Over the place plan's `parts` (the solid defining parts the solver placed), its
        `districts` (each carrying `defines`, the defining part it answers) and its
        `compounds`. A district answers the word its defining part is named or purposed
        for: a `fields` district answers `fields`, a `homes` district answers `houses`.
        
    """
    place = place or {}
    w = _slug(word)
    if not w:
        return []
    words = [x for x in w.split("_") if x and x not in ("the", "a", "an", "of", "its")]
    rows = []
    for key in ("parts", "districts", "compounds"):
        for r in place.get(key) or []:
            if isinstance(r, dict) and r.get("x1") is not None:
                rows.append((key, r))
    out = []
    for key, r in rows:
        leaf = _region_as_leaf(r)
        blob = " ".join(str(r.get(k) or "") for k in ("name", "defines", "purpose",
                                                        "notes")).lower()
        if any(x in _SCOPE_WORDS for x in words) and len(words) > 1:
            if _in_scope(leaf, word):
                out.append(r)
            continue
        if len(words) == 1 and words[0] in _FIELD_WORDS:
            # **through the entity bindings.** The falsification pass: `orchard` fell
            # into this branch and was held to `farmland`, so the orchard region the
            # solver had laid answered nothing. A land word selects the regions whose
            # land use is that word's -- an orchard the orchard, `fields` any farmland.
            use = _land_word(words[0])
            got = spec_mod.land_use_of({"name": r.get("defines") or r.get("name"),
                                        "family": r.get("family"), "notes": blob})
            if key == "districts" and got is not None and (
                    got == use or (use == "farmland" and got in ("farmland", "orchard",
                                                                  "pasture"))
                    or (use is None and got == "farmland")):
                out.append(r)
            elif key == "parts" and _is_field_leaf(
                    {**leaf, "family": r.get("family"), "notes": blob}, use):
                out.append(r)
            continue
        head = words[-1]
        if head in _HOUSE_WORDS:
            if key == "districts" and spec_mod.land_use(
                    {"name": r.get("defines") or r.get("name"), "notes": blob}) \
                    != "farmland" and int(r.get("structures") or 0) > 0:
                out.append(r)
            elif key == "parts" and _word_matches(leaf, head):
                out.append(r)
            continue
        if head in _COMPOUND_WORDS and key == "districts":
            continue          # a district named for the palace is not the palace
        if _word_matches(leaf, head) or re.search(
                rf"(?<![a-z]){re.escape(head)}(?![a-z])", blob):
            if len(words) == 1 or _in_scope(leaf, " ".join(words[:-1])):
                out.append(r)
    return out


#: **What a density word means, registered once.** `metric` is the share of the
#: districts' **developable** ground the lots stand on -- the ground the compiler can
#: put anything on, less the arterial's band and the standing parts' clearances, which
#: is `placeplan.developable_columns` and is the denominator the compiler's own target
#: is computed over. Both bounds, because a word bounds from both sides: a sparse
#: quarter with no houses at all is not sparse, it is empty, and a dense one has no
#: ceiling. The review found the compiler's `sparse` objective at ~19% of the rectangle
#: against a check that capped it at 12% over a different denominator, and the two could
#: not both be met by any arrangement. These are the checker's numbers and the compiler
#: is held to the same ones (interface I1); nothing here is moved to make a saved case
#: pass. `dense` and `sparse` keep the bounds `QUALITY_BOUNDS` registered before the
#: realization round.
DENSITY_TARGETS = {
    "sparse": (0.03, 0.12),
    "low":    (0.08, 0.22),
    "medium": (0.15, 0.34),
    "dense":  (0.30, None),
}

#: The least of an allocated lot that emitted mass may fill before the allocation is
#: emptier than anything this library builds -- so a density figure reached with lots
#: like that is a statement about allocation and not about fabric. **Measured, not
#: chosen.** Over the 34 buildings of the design round's retained city section
#: (`out/des-city/parts.json`, emitted footprint against the plot rect it was laid on):
#: median 0.51, minimum **0.45**, and by type `court_large` 0.50, `hall` 0.51,
#: `shop_house` 0.56, `temple` 0.57, `courtyard_house` 0.59. Every type in this library
#: fills at least 0.45 of its lot, so 0.40 cannot be reached by building; it can only be
#: reached by allocating ground that carries nothing. That is exactly the escape route
#: the design review named -- "enlarging empty lots ... cannot establish density".
MIN_LOT_FILL = 0.40


def density_target(value: str, role: str | None = None) -> dict:
    """`{"metric", "lo", "hi", "denominator", "why"}` -- the one statement of a
    density word. The role does not move the bounds; it is accepted so a caller that
    knows it can say so and a future rule that needs it has the seam."""
    v = str(value or "medium").lower()
    if v not in DENSITY_TARGETS:
        return {"metric": "lot_cover", "lo": None, "hi": None,
                "denominator": "developable_columns",
                "why": f"this build has no density meaning for `{v}`"}
    lo, hi = DENSITY_TARGETS[v]
    return {"metric": "lot_cover", "lo": lo, "hi": hi,
            "denominator": "developable_columns",
            "why": (f"`{v}`: the lots cover "
                    + (f"at least {lo:.0%}" if lo is not None else "any share")
                    + (f" and at most {hi:.0%}" if hi is not None else "")
                    + " of the developable ground of the districts it is about; "
                      "measured by the checker and targeted by the compiler over the "
                      "same columns")}


def lot_cover(plots: list, regions: list) -> dict:
    """The density metric, measured over **the ground the requirement is about**.

            {"cover", "built", "ground", "denominator", "excluded", "built_columns",
             "built_cover"}

        `regions` are resolution regions (or districts) carrying `rect` and, where the
        record has them, `scope_columns`, `developable_columns` and `built_columns`. The
        denominator is the requested scope where every region records it, the developable
        columns where every region records those, and the rectangles' columns otherwise --
        and the record says which, so a number over the wrong denominator cannot pass as
        the metric.

        **An inferred remainder does not leave the question.** The expression round let the
        ring layout keep a dense ring's leftover as `surface: "open"` sectors and then
        dropped every open region from the denominator, so the ring's "27.5% dense" was
        measured over the part of the ring that happened to get lots. Naming a remainder
        open is an allocation decision, and an allocation decision cannot narrow the scope
        of the requirement it is being judged against. Ground **an explicit requirement asks
        to be open** -- a park, a green, land the sentence reserves -- does leave, carries
        `open_requested` with the requirement id, and is named in `excluded`.

        **And a bigger empty lot is not denser building.** `built` is the area of the lots
        the compiler drew; `built_columns`, where the regions record it, is the footprint
        construction actually emitted. `cover` is the first and `built_cover` the second,
        and a cover figure that improved only because the lots grew shows it here.
        
    """
    ground, scope, dev, built_cols = 0.0, 0.0, 0.0, 0.0
    all_scope, all_dev, all_built = True, True, True
    excluded = []
    for r in regions or []:
        rect = r.get("rect")
        if not rect:
            continue
        if r.get("open_requested"):
            excluded.append({"region": r.get("name"),
                             "requirement": r.get("open_requested"),
                             "columns": r.get("scope_columns")
                             or (abs(float(rect[2]) - float(rect[0])) + 1)
                             * (abs(float(rect[3]) - float(rect[1])) + 1)})
            continue
        x0, z0, x1, z1 = [float(v) for v in rect]
        ground += (abs(x1 - x0) + 1) * (abs(z1 - z0) + 1)
        for key, acc in (("scope_columns", "scope"), ("developable_columns", "dev"),
                         ("built_columns", "built_cols")):
            if r.get(key) is None:
                if acc == "scope":
                    all_scope = False
                elif acc == "dev":
                    all_dev = False
                else:
                    all_built = False
        if r.get("scope_columns") is not None:
            scope += float(r["scope_columns"])
        if r.get("developable_columns") is not None:
            dev += float(r["developable_columns"])
        if r.get("built_columns") is not None:
            built_cols += float(r["built_columns"])
    built = 0.0
    for p in plots or []:
        r = _rect_of_leaf(p)
        if r is not None:
            built += (abs(r[2] - r[0]) + 1) * (abs(r[3] - r[1]) + 1)
    if all_scope and scope:
        denom, named = scope, "scope_columns"
    elif all_dev and dev:
        denom, named = dev, "developable_columns"
    else:
        denom, named = ground, "rect"
    return {"cover": (built / denom) if denom else None, "built": int(built),
            "ground": int(denom), "denominator": named,
            "excluded": excluded,
            "built_columns": int(built_cols) if (all_built and built_cols) else None,
            "built_cover": (built_cols / denom) if (all_built and built_cols and denom)
                           else None}


def relation_measure(req, parts: list) -> tuple:
    """`(status, why, evidence)` -- the relation the sentence states, on these parts.

        The public face of `_relation_measure`, so a layout scores a candidate arrangement
        with the checker's own rule instead of a rule of its own. `req` is a requirement
        record or its `wants` (`subject`, `relation`, `object`).
        
    """
    w = req.get("wants") if isinstance(req, dict) and "wants" in req else req
    return _relation_measure(dict(w or {}), list(parts or []))


def _path_of(p: dict) -> list:
    """An edge leaf's path as `[(x, z)]`, or `[]`."""
    out = []
    for a in (p.get("path") or []):
        try:
            out.append((float(a[0]), float(a[-1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _rect_of_leaf(p: dict) -> tuple | None:
    """The leaf's rectangle, or -- for an edge that records only a path -- the bounding
    rectangle of that path.
    """
    try:
        return (float(p["x0"]), float(p["z0"]), float(p["x1"]), float(p["z1"]))
    except (KeyError, TypeError, ValueError):
        pass
    path = _path_of(p)
    if path:
        return (min(a for a, _b in path), min(b for _a, b in path),
                max(a for a, _b in path), max(b for _a, b in path))
    try:
        from . import pipeline as _pipeline
        rects = _pipeline.part_rects({**p, "name": p.get("name")})
        if rects:
            return (float(min(r[0] for r in rects)), float(min(r[1] for r in rects)),
                    float(max(r[2] for r in rects)), float(max(r[3] for r in rects)))
    except Exception:                          # noqa: BLE001 -- no geometry, no rect
        pass
    return None


def _in_polygon(path: list, x: float, z: float) -> bool:
    """Ray cast, half-open; a point on the line is not inside."""
    pts = list(path)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return False
    hit = False
    for (ax, az), (bx, bz) in zip(pts, pts[1:] + pts[:1]):
        if (az > z) != (bz > z):
            cut = ax + (z - az) * (bx - ax) / (bz - az)
            if cut > x:
                hit = not hit
    return hit


def _closed_ring(p: dict) -> list:
    """The object's closed ring, where it is an edge whose path closes; else `[]`."""
    path = _path_of(p)
    return path if len(path) >= 4 and path[0] == path[-1] else []


def _gap_to_object(r: tuple, obj: dict, orect: tuple) -> float:
    """The subject rectangle's distance to the object: to its line where the object is
    an edge with a path, to its rectangle's edge otherwise."""
    path = _path_of(obj)
    if str(obj.get("kind") or "") == "edge" and path:
        return _rect_to_line(r, path)
    return _edge_gap(r, orect)


def _rect_to_line(r: tuple, path: list) -> float:
    """The least distance from rectangle `r` to the polyline `path`, segment-wise.

    `_rect_to_path` measures to the path's vertices, which for a wall drawn as a few
    long runs puts a field five columns off the line twenty columns from its nearest
    corner. A segment crossing the rectangle is at distance zero."""
    x0, z0, x1, z1 = [float(v) for v in r]
    if len(path) == 1:
        path = [path[0], path[0]]
    best = float("inf")
    corners = ((x0, z0), (x1, z0), (x0, z1), (x1, z1))
    for a, b in zip(path, path[1:]):
        # an endpoint inside the rectangle, or a corner on the segment: touching
        for px, pz in (a, b):
            dx = max(x0 - px, 0.0, px - x1)
            dz = max(z0 - pz, 0.0, pz - z1)
            best = min(best, (dx * dx + dz * dz) ** 0.5)
        for c in corners:
            best = min(best, _seg_distance(c[0], c[1], a, b))
        # a segment crossing the rectangle without an endpoint in it: it crosses one of
        # the rectangle's own sides
        sides = ((corners[0], corners[1]), (corners[1], corners[3]),
                 (corners[3], corners[2]), (corners[2], corners[0]))
        if best > 0 and any(_segments_cross(a, b, s0, s1) for s0, s1 in sides):
            return 0.0
        if best == 0:
            return 0.0
    return best


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and o1 != 0 and o2 != 0 \
        and o3 != 0 and o4 != 0


def _footprint(p: dict) -> float:
    r = _rect_of_leaf(p)
    return 0.0 if r is None else abs(r[2] - r[0] + 1) * abs(r[3] - r[1] + 1)


def _centre(p: dict) -> tuple | None:
    r = _rect_of_leaf(p)
    return None if r is None else ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)


def _storeys(p: dict, decls: dict) -> float | None:
    got = (p.get("params") or {}).get("storeys")
    return float(got) if isinstance(got, (int, float)) else None


def sample_scope(parts_record: dict | None, parts: list) -> set | None:
    """The names of the leaves a **construction sample** attempted, or None.

        A parts record carrying `sample` (`stages_build.sample_parts`) built the plots of
        its named quarters and the parts joining them and nothing else, and the round's
        registered rule is that a sample qualifies only its constructed scope. The scope is
        the record's own rows, the plots of the sampled quarters and the joining parts;
        everything else in the plan was not built, and a clause that counts standing
        leaves is measured inside this set and says so.
        
    """
    sm = (parts_record or {}).get("sample")
    if not sm:
        return None
    names = {r["part"] for w in (parts_record or {}).get("waves", [])
             for r in w.get("parts", []) if r.get("part")}
    # **A registered section takes a quarter partly** (the composition round): its cut
    # falls between buildings, so a quarter on the section boundary contributes the
    # plots that fit and no others, and `included_plots` names them. Adding every plot
    # of a partly-taken quarter would put leaves nobody attempted into the scope a
    # clause about standing leaves is measured over -- a sample scope that flatters
    # itself. Where the record names its plots, they are the scope; where it does not
    # (`sample_parts`, which takes whole quarters), the quarter's plots are.
    if sm.get("included_plots") is not None:
        names |= {str(n) for n in sm["included_plots"]}
    else:
        quarters = set(sm.get("quarters") or [])
        for p in parts or []:
            if p.get("kind", "plot") == "plot" \
                    and (p.get("in") or [None])[-1] in quarters:
                names.add(p["name"])
    names |= {str(n) for n in (sm.get("joining_parts") or [])}
    return names


#: The evidence a requirement carries when a sample left it unbuilt: the place read
#: treats it as neither met nor missed, and `limits.sample` names it.
OUTSIDE_SAMPLE = "outside the sample: nothing of it was built"


def emitted_of(parts_record: dict | None) -> dict:
    """`{part name: emitted outcome}` off `parts.json`, interface I3.

        What construction actually delivered for each part -- storeys, features, omissions,
        fallback -- measured by the construction worker off the emitted geometry. Empty
        where nothing was built or the record predates the interface, which reads as "not
        measured" and never as "as planned".
        
    """
    out = {}
    for w in (parts_record or {}).get("waves", []):
        for r in w.get("parts", []):
            if isinstance(r.get("emitted"), dict):
                out[r["part"]] = r["emitted"]
    return out


def emitted_findings(parts: list, parts_record: dict | None) -> list:
    """The parts whose construction dropped what the plan asked of them.

        A type that silently reduces three storeys to one on a nine-column lot returns
        success and the planned parameters, and every check that read `params.storeys`
        called the place tall. With the emitted outcome on the record the difference is a
        finding: `layout` where the outcome names the envelope as the cause (the lot is
        the layout's decision), `build` otherwise. `blocks: construction` -- the plan is
        feasible and the buildings stand; what stands is not what was asked.
        
    """
    got = emitted_of(parts_record)
    if not got:
        return []
    out = []
    for p in parts:
        e = got.get(p.get("name"))
        if not e:
            continue
        planned = (p.get("params") or {}).get("storeys")
        emitted = e.get("storeys")
        omitted = list(e.get("omitted") or [])
        if (isinstance(planned, (int, float)) and isinstance(emitted, (int, float))
                and int(emitted) != int(planned)) or omitted or e.get("fallback"):
            why = str(e.get("why") or e.get("fallback") or "")
            owner = "layout" if any(k in why.lower() for k in
                                    ("lot", "envelope", "pad", "footprint", "too small",
                                     "narrow")) else "build"
            says = (f"{p.get('name')} ({p.get('type')}): planned {planned} storey(s), "
                    f"emitted {emitted}" if emitted != planned else
                    f"{p.get('name')} ({p.get('type')}): built with a fallback")
            if omitted:
                says += f"; omitted {', '.join(str(x) for x in omitted[:4])}"
            if why:
                says += f" -- {why[:140]}"
            out.append({"id": f"find/emitted/{p.get('name')}", "says": says,
                        "requirement": None, "part": p.get("name"),
                        "evidence": {"planned_storeys": planned, "emitted": e},
                        "owner": owner, "blocks": "construction",
                        "severity": "error", "seen_by": "intent.emitted",
                        "fixed": False})
    return out


def _hierarchy_measure(w: dict, parts: list, decls: dict) -> tuple:
    """`(status, why, evidence)` -- is the greater thing actually greater?

        The review's counterexample is "small houses around a big temple", whose size
        hierarchy disappeared entirely. It is a measurement: the greater part's footprint (or
        its storeys, on the `height` axis) against the lesser's, on the plan that exists.
        Nothing here reads an adjective.
        
    """
    axis = str(w.get("axis") or "size")
    great = select(parts, str(w.get("greater") or ""))
    great_names = {q.get("name") for q in great} | {
        n for q in great for n in (q.get("leaves") or [])}
    less = [p for p in select(parts, str(w.get("lesser") or ""))
            if p.get("name") not in great_names]
    if not great or not less:
        return ("unresolved",
                f"the plan holds {len(great)} part(s) answering "
                f"`{w.get('greater')}` and {len(less)} answering `{w.get('lesser')}`, so "
                f"which is the greater is not a question it can answer",
                {"greater": len(great), "lesser": len(less), "axis": axis})
    if axis == "height":
        g = [x for x in (_storeys(p, decls) for p in great) if x is not None]
        l = [x for x in (_storeys(p, decls) for p in less) if x is not None]
        unit = "storey(s)"
    else:
        g = [_footprint(p) for p in great if _footprint(p)]
        l = [_footprint(p) for p in less if _footprint(p)]
        unit = "columns"
    if not g or not l:
        return ("unresolved",
                f"the plan records no {axis} for one side of this comparison",
                {"greater_measured": len(g), "lesser_measured": len(l)})
    gm, lm = max(g), sorted(l)[len(l) // 2]
    ev = {"greater": round(gm, 1), "lesser_median": round(lm, 1), "axis": axis,
          "unit": unit, "greater_parts": len(great), "lesser_parts": len(less)}
    if gm > lm:
        return ("satisfied",
                f"the greatest `{w.get('greater')}` is {gm:g} {unit} against a median "
                f"`{w.get('lesser')}` of {lm:g}", ev)
    return ("failed",
            f"the sentence makes `{w.get('greater')}` the greater and the greatest one "
            f"is {gm:g} {unit} against a median `{w.get('lesser')}` of {lm:g}", ev)


#: How much of the way round an object the things said to stand **around** it must
#: reach, as a share of the four quadrants, and how far out they may stand as a multiple
#: of the object's own half-diagonal. Registered here, before any run measured them.
AROUND_QUADRANTS = 3
AROUND_REACH = 8.0
#: How far a thing said to stand **beside** (or on, or by) another may be from that
#: other's own edge, in columns: a lane and a clearance apart is beside; a district away
#: is not. Registered here. The review reproduced `inside` and `beside` sharing one
#: proximity rule measured centre to centre, so a house forty columns clear of a temple
#: passed `inside`; each relation is now the measurement its word means.
BESIDE_REACH = 12.0


def _edge_gap(a: tuple, b: tuple) -> float:
    """The least distance between two rectangles' edges; 0 where they touch or overlap."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dz = max(b[1] - a[3], a[1] - b[3], 0.0)
    return (dx * dx + dz * dz) ** 0.5


def _contained(inner: tuple, outer: tuple) -> bool:
    return (outer[0] <= inner[0] and inner[2] <= outer[2]
            and outer[1] <= inner[1] and inner[3] <= outer[3])


def _relation_measure(w: dict, parts: list) -> tuple:
    """`(status, why, evidence)` -- do the subjects stand in this relation to the object?

        `around` is bearings: the subjects have to reach round the object rather than sit in
        a clump on one side of it, which is the difference between a temple in a village and
        a temple beside one. `inside` is containment of the subject's rectangle in the
        object's. `beside` (and `on`, read as beside) is the subject's edge within
        `BESIDE_REACH` of the object's edge. `along` is the subjects strung within reach of
        the object's edge and spread along its length. A relation this build cannot measure
        says so and certifies nothing.
        
    """
    rel = str(w.get("relation") or "")
    subs = select(parts, str(w.get("subject") or ""))
    objs = [p for p in select(parts, str(w.get("object") or "")) if p not in subs]
    if not subs or not objs:
        return ("unresolved",
                f"the plan holds {len(subs)} `{w.get('subject')}` and {len(objs)} "
                f"`{w.get('object')}`, so this relation is not measurable on it",
                {"subjects": len(subs), "objects": len(objs), "relation": rel})
    obj = max(objs, key=_footprint)
    oc = _centre(obj)
    orect = _rect_of_leaf(obj)
    if oc is None or orect is None:
        return ("unresolved", "the object of this relation records no rectangle", {})
    half = max(1.0, ((orect[2] - orect[0]) ** 2 + (orect[3] - orect[1]) ** 2) ** 0.5 / 2)
    rows = [(p, _rect_of_leaf(p)) for p in subs]
    rows = [(p, r) for p, r in rows if r is not None]
    if not rows:
        return ("unresolved", "no subject of this relation records a rectangle", {})
    base = {"subjects": len(rows), "relation": rel, "subject": str(w.get("subject") or ""),
            "object": obj.get("answers") or obj.get("defines") or obj.get("name"),
            "object_leaf": obj.get("name"), "object_rect": [int(v) for v in orect]}
    if rel == "around":
        pts = [(((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0), p) for p, r in rows]
        dists = [((c[0] - oc[0]) ** 2 + (c[1] - oc[1]) ** 2) ** 0.5 for c, _p in pts]
        near = [d for d in dists if d <= AROUND_REACH * half]
        quads = {(int(c[0] >= oc[0]), int(c[1] >= oc[1]))
                 for (c, _p), d in zip(pts, dists) if d <= AROUND_REACH * half}
        ev = {**base, "within_reach": len(near), "quadrants": len(quads),
              "need_quadrants": AROUND_QUADRANTS, "reach": round(AROUND_REACH * half, 1)}
        if len(quads) >= AROUND_QUADRANTS and len(near) >= max(2, len(pts) // 2):
            return ("satisfied",
                    f"{len(near)} of {len(pts)} `{w.get('subject')}` stand within "
                    f"{ev['reach']:g} columns of `{obj.get('name')}`, reaching "
                    f"{len(quads)} of 4 sides of it", ev)
        return ("failed",
                f"{len(near)} of {len(pts)} `{w.get('subject')}` are near "
                f"`{obj.get('name')}` and they reach {len(quads)} of 4 sides of it, "
                f"against the {AROUND_QUADRANTS} this build calls standing around "
                f"something; they are beside it, not around it", ev)
    if rel == "inside":
        # **Containment, not proximity.** The review's counterexample: a house at
        # [40,44]^2 wholly outside a temple at [0,19]^2 passed `inside`. For a wall --
        # an edge whose path is a closed ring -- inside is inside the ring itself, every
        # corner of the subject, and not inside the ring's bounding box.
        ring = _closed_ring(obj)
        if ring:
            def _contained_in(r, _orect):
                return all(_in_polygon(ring, x, z) for x, z in
                           ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])))
        else:
            _contained_in = _contained
        inside = [p for p, r in rows if _contained_in(r, orect)]
        ev = {**base, "inside": len(inside), "object_ring": bool(ring),
              "examples_outside": [p.get("name") for p, r in rows
                                   if not _contained_in(r, orect)][:4]}
        ok = len(inside) >= max(1, (len(rows) + 1) // 2) and len(inside) == len(rows)
        return (("satisfied" if ok else "failed"),
                f"{len(inside)} of {len(rows)} `{w.get('subject')}` stand wholly inside "
                f"`{obj.get('name')}`"
                + (f"'s ring of {len(ring) - 1} points" if ring else
                   f" ({int(orect[0])},{int(orect[1])})-({int(orect[2])},{int(orect[3])})")
                + ("" if ok else "; containment is the measurement and being near is "
                                 "not it"), ev)
    orects = [q for q in (_rect_of_leaf(o) for o in objs) if q is not None]
    obj_geoms = [(o, q) for o, q in ((o, _rect_of_leaf(o)) for o in objs)
                 if q is not None]

    def gap_to_nearest(r):
        return min(_gap_to_object(r, o, q) for o, q in obj_geoms)
    if rel in ("beside", "on", "near", "by"):
        # **The unit is the thing the sentence names.** "a hall on the square" is one
        # building, measured itself; "fields beside the houses" is the fields as working
        # ground -- a farmland district's tiles, a court's, a grove's -- and asking half
        # of the tiles individually to touch a house asks a question the sentence does
        # not. Found by running the closure proof: 1 of 8 field tiles within reach and
        # the fields, as a block, four columns from the homes. So a plural subject of
        # area leaves, or of several leaves of one district, is measured as its union
        # (the bounding rectangle of its tiles) to the nearest object's edge, and at
        # least one tile has to reach itself, so a block that merely spans past a house
        # from far away does not pass on its extent. Single buildings keep the per-
        # subject rule. `evidence.unit` says which was applied.
        ancestry = {tuple(p.get("in") or []) for p, _r in rows}
        grouped = (all(str(p.get("kind") or "") == "area" for p, _r in rows)
                   or (len(rows) > 1 and len(ancestry) == 1 and ancestry != {()}))
        each = [gap_to_nearest(r) for _p, r in rows]
        if grouped:
            union = (min(r[0] for _p, r in rows), min(r[1] for _p, r in rows),
                     max(r[2] for _p, r in rows), max(r[3] for _p, r in rows))
            gap = gap_to_nearest(union)
            touching = sum(1 for g in each if g <= BESIDE_REACH)
            ok = gap <= BESIDE_REACH and touching >= 1
            ev = {**base, "unit": "group", "union_rect": [int(v) for v in union],
                  "gap": round(gap, 1), "tiles_within_reach": touching,
                  "reach": BESIDE_REACH}
            return (("satisfied" if ok else "failed"),
                    f"the {len(rows)} `{w.get('subject')}` as one block "
                    f"({int(union[0])},{int(union[1])})-({int(union[2])},{int(union[3])})"
                    f" stand {gap:.0f} column(s) from the edge of "
                    + (f"`{obj.get('name')}`" if len(objs) == 1 else
                       f"the nearest of {len(objs)} `{w.get('object')}`")
                    + f", against {BESIDE_REACH:g}, with {touching} tile(s) within reach "
                      f"themselves", ev)
        gaps = each
        close = [g for g in gaps if g <= BESIDE_REACH]
        ev = {**base, "unit": "each", "within_reach": len(close), "reach": BESIDE_REACH,
              "gaps": [round(g, 1) for g in sorted(gaps)[:8]]}
        ok = len(close) >= max(1, (len(rows) + 1) // 2)
        return (("satisfied" if ok else "failed"),
                f"{len(close)} of {len(rows)} `{w.get('subject')}` stand within "
                f"{BESIDE_REACH:g} columns of the edge of "
                + (f"`{obj.get('name')}`" if len(objs) == 1 else
                   f"the nearest of {len(objs)} `{w.get('object')}`")
                + (f"; the nearest is {min(gaps):.0f} away" if not ok else ""), ev)
    if rel == "along":
        # strung along the object's edge: most within reach of it, and spread over at
        # least half its longer side rather than bunched at one end
        gaps = [(gap_to_nearest(r), r) for _p, r in rows]
        close = [r for g, r in gaps if g <= BESIDE_REACH * 2]
        span = 0.0
        if close:
            along_x = (orect[2] - orect[0]) >= (orect[3] - orect[1])
            lo = min((r[0] + r[2]) / 2 if along_x else (r[1] + r[3]) / 2 for r in close)
            hi = max((r[0] + r[2]) / 2 if along_x else (r[1] + r[3]) / 2 for r in close)
            length = max(1.0, (orect[2] - orect[0]) if along_x else (orect[3] - orect[1]))
            span = (hi - lo) / length
        ev = {**base, "within_reach": len(close), "spread": round(span, 2)}
        ok = len(close) >= max(2, (len(rows) + 1) // 2) and span >= 0.5
        return (("satisfied" if ok else "failed"),
                f"{len(close)} of {len(rows)} `{w.get('subject')}` stand within "
                f"{BESIDE_REACH * 2:g} columns of `{obj.get('name')}` and span "
                f"{span:.0%} of its length", ev)
    return ("unresolved",
            f"this build has no measurement for the relation `{rel}`, so whether the "
            f"plan carries it is not established", {"relation": rel})


def _usable_verdict(parts: list, types: set, parts_record: dict | None) -> dict:
    """What the assembled world said about the parts that answer a function.

        `{"held": n, "failed": [(part, want, why)], "owed": [(part, want, method)],
          "unsupported": n, "subjects": n, "measured_subjects": n, "method": ...}`.
        `construction.confirm` writes `emitted.usable` on every built row after all
        construction; where it ran, a function's status is a measurement, and where it did
        not, the only thing behind the status is a type declaring itself.

        **An answer that decided nothing is not a predicate that ran.** The design review's
        finding, in the code that produced it: this function counted `ran += 1` for every
        answer whose method was not `unsupported`, so a `declared` answer with `holds: None`
        -- `equipment_reachable`'s honest "this part claims equipment and published no
        rectangle to look in" -- incremented the count, `holds is False` was the only failure
        test, and `_function_measure` then reported `satisfied` with `method: "observed"` and
        the words "1 final-world predicate(s) on them hold". Nothing held. The three
        categories are now kept apart and all three are returned:

          `held`         an affirmative answer measured on the assembled world;
          `failed`       the world refuses it;
          `owed`         asked and undecided -- `declared`, or `holds: None` for any other
                         reason. Unresolved, and it stays that way until something measures
                         it. This is the category that used to be counted as a pass.

        `unsupported` is counted separately and is neither owed nor held: a wall carries no
        door leaf, and asking it about entrances is a question that does not apply.
        
    """
    rows = {r.get("part"): r for w2 in (parts_record or {}).get("waves") or []
            for r in (w2.get("parts") or [])}
    held, failed, owed, unsup = 0, [], [], 0
    subjects, measured = 0, 0
    for p in parts:
        if str(p.get("type") or "") not in types:
            continue
        subjects += 1
        em = (rows.get(p.get("name")) or {}).get("emitted") or {}
        got = em.get("usable") or {}
        any_here = False
        # **A required feature with no affirmative answer is owed, whatever the
        # predicates said.** `construction.confirm` writes `emitted.owed` for exactly
        # this: the tokens this part was required to deliver and has no `observed` or
        # `inferred` evidence for, with the reason (`not_in_world`, `not_identified`,
        # `declared`, `unsupported`, `unmeasured`). A function cannot be satisfied over
        # a part that owes one.
        for o in em.get("owed") or []:
            if isinstance(o, dict) and o.get("feature"):
                owed.append((p.get("name"), f"feature/{o['feature']}",
                             str(o.get("reason") or "owed")))
        for want, a in got.items():
            if not isinstance(a, dict):
                continue
            method = str(a.get("method") or "")
            if method == "unsupported":
                unsup += 1
                continue
            if a.get("holds") is True and method in ("observed", "inferred"):
                held += 1
                any_here = True
            elif a.get("holds") is False:
                failed.append((p.get("name"), want, str(a.get("why"))[:120]))
                any_here = True
            else:
                owed.append((p.get("name"), want, method or "unknown"))
        measured += 1 if any_here else 0
    return {"held": held, "failed": failed, "owed": owed, "unsupported": unsup,
            "subjects": subjects, "measured_subjects": measured,
            # the old key, kept for readers that count it -- and it now counts only
            # answers that decided something
            "ran": held + len(failed),
            "method": "observed" if held or failed else "declared"}


def _function_measure(w: dict, parts: list, capabilities: dict | None,
                      parts_record: dict | None = None) -> tuple:
    """`(status, why, evidence)` -- is this function actually fulfilled?

        **A role label is not a function.** The review's third finding in its semantic form:
        "An allowed plot type and a coarse role do not establish that a dwelling's function
        has been fulfilled", and the village's three halls and two temples in an eight-house
        fishing village are what that costs. A function is fulfilled when a committed type
        that *declares* it is what the plan actually built, and it is `unsupported` when
        nothing on disk declares it -- which keeps the obligation visible instead of letting
        the nearest-looking building close it.
        
    """
    # the function named, not the word the sentence used for the thing that has it:
    # `{"what": "house", "function": "dwelling"}` is a request for a dwelling
    what = str(w.get("function") or w.get("what") or "")
    entries = [e for e in (capabilities or {}).get("entries") or []
               if str((e.get("wants") or {}).get("function") or "") == what
               or str((e.get("envelope") or {}).get("function") or "") == what]
    declared = sorted({str(e.get("type")) for e in entries
                       if e.get("matched") and e.get("type")})
    if not entries:
        return ("unsupported",
                f"no committed type in this library declares the function `{what}`, so "
                f"the place cannot be said to fulfil it; a type whose role or tradition "
                f"label happens to suit is not the function",
                {"function": what, "declared": []})
    if not declared:
        return ("failed",
                f"the capability record has a want for the function `{what}` and matched "
                f"no type to it", {"function": what, "wants": len(entries)})
    used = sorted({str(p.get("type")) for p in parts
                   if str(p.get("type") or "") in set(declared)})
    if not parts:
        return ("open", f"types declaring `{what}` were selected ({', '.join(declared)});"
                        f" nothing is planned yet", {"declared": declared})
    if not used:
        return ("failed",
                f"{len(declared)} type(s) declaring the function `{what}` were selected "
                f"({', '.join(declared[:4])}) and the plan's {len(parts)} leaf(s) use "
                f"none of them", {"declared": declared, "used": []})
    # **...and a type that declares it is not a function that is fulfilled.** The
    # review's fourth finding, and the design round's third boundary: until here the
    # answer above was the whole measurement, so a market whose stalls a later terrace
    # buried and a chapel nobody can walk into both read `satisfied`. Where
    # `construction.confirm` has re-read these parts on the assembled world, its
    # predicates decide; where it has not, this stays a declaration and says so.
    seen = _usable_verdict(parts, set(declared), parts_record)
    if seen["failed"]:
        return ("failed",
                (f"the plan builds {len(used)} type(s) declaring `{what}` and the "
                 f"assembled world refuses {len(seen['failed'])} of the predicates that "
                 f"decide use: "
                 + "; ".join(f"{n} {wt}: {why}" for n, wt, why in seen["failed"][:3])),
                {"declared": declared, "used": used, "method": "observed",
                 "failed": [list(f) for f in seen["failed"][:8]], "ran": seen["ran"],
                 "held": seen["held"], "owed": [list(o) for o in seen["owed"][:8]]})
    # **An owed predicate leaves the function unresolved.** The composition round: an
    # undecided answer used to be counted as a predicate that held, so a function could
    # report observed success because a *different* predicate ran. A function whose
    # applicable evidence is owed is not failed either -- nothing refused it -- so the
    # honest status is `unresolved`, and the words say which subject owes what.
    if seen["owed"]:
        return ("unresolved",
                (f"the plan builds {len(used)} type(s) declaring `{what}` and "
                 f"{len(seen['owed'])} applicable predicate answer(s) are undecided, so "
                 f"the assembled world has not established use: "
                 + "; ".join(f"{n} {wt} ({m})" for n, wt, m in seen["owed"][:3])),
                {"declared": declared, "used": used, "method": "unresolved",
                 "owed": [list(o) for o in seen["owed"][:8]], "held": seen["held"],
                 "ran": seen["ran"]})
    return ("satisfied",
            f"the plan builds {len(used)} type(s) that declare the function `{what}`: "
            f"{', '.join(used[:4])}"
            + (f", and {seen['held']} final-world predicate(s) on them hold"
               if seen["held"] else
               "; no final-world predicate has been asked of them, so this is what the "
               "types declare of themselves and not a measurement of use"),
            {"declared": declared, "used": used, "method": seen["method"],
             "ran": seen["ran"], "held": seen["held"],
             "subjects": seen["subjects"],
             "measured_subjects": seen["measured_subjects"],
             "unsupported": seen["unsupported"]})


#: What a judge may have looked at for a verdict to be about **built** output.
BUILT_EVIDENCE = ("world_built.npz", ".png", "parts.json")


def judgment_usable(judgment: dict | None, verdict: dict | None,
                    reading: dict | None) -> tuple:
    """`(ok, why)` -- may this verdict close an obligation?

        Three things, and the review found each missing: the judgment looked at **built**
        output (a plan alone is intentions); the verdict **cites** claims; and every cited
        claim is one this run actually made from a retrieved source. A negative verdict is
        consumed whatever this says -- refusing needs no evidence -- and a positive one
        that fails here leaves the obligation `unresolved`, naming what is missing.
        
    """
    if judgment is None or verdict is None:
        return (False, "no judgment")
    looked = [str(x) for x in (judgment.get("looked_at") or [])]
    if not any(x.endswith(BUILT_EVIDENCE) and "plan.json" not in os.path.basename(x)
               for x in looked):
        return (False, "the judgment looked at no built output -- no built world, no "
                       "render, no parts record -- so it is a judgment of intentions")
    cites = [str(c) for c in (verdict.get("cites") or [])]
    if not cites:
        return (False, "the verdict cites no claim; a judgement with no evidence behind "
                       "it does not close an obligation")
    srcs = {s.get("id") for s in (reading or {}).get("sources") or []}
    claims = {c.get("id"): c for c in (reading or {}).get("claims") or []}
    bad = [c for c in cites if c not in claims]
    unsourced = [c for c in cites if c in claims and not (claims[c].get("source") in srcs)]
    if bad:
        return (False, f"the verdict cites {bad[:3]}, which is no claim this run made; "
                       f"a citation of research nobody did is refused")
    if unsourced:
        return (False, f"the verdict cites {unsourced[:3]}, inferred claims with no "
                       f"retrieved source behind them")
    return (True, f"{len(cites)} sourced claim(s) cited, judged on "
                  f"{os.path.basename(next(x for x in looked if x.endswith(BUILT_EVIDENCE)))}")


def _verdict_for(judgment: dict | None, about: str, name: str | None = None) -> dict | None:
    """The inspection's verdict on one obligation, or None where it made none."""
    for v in (judgment or {}).get("verdicts") or []:
        if str(v.get("about")) != about:
            continue
        if name and str(v.get("name") or "").lower() != str(name).lower():
            continue
        return v
    return None


def _share(v) -> float | None:
    """A share, from a record that may have written it as a percentage."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f / 100.0 if f > 1.0 else f


def site_facts(site: dict | None) -> dict:
    """The ground's own measurements, from wherever the site record wrote them.

        A site is written by the search and again by `prepare_settlement`, and the two put
        the same facts in different places -- `stats.relief` against `measures.relief`,
        `surface_blocks` against `water_pct`. One reader, so a check cannot be right about a
        site chosen one way and wrong about the same site chosen the other.
        
    """
    site = site or {}
    m = site.get("measures") or (site.get("chosen") or {}).get("measures") or {}
    stats = site.get("stats") or {}
    out = {"relief": None, "water": None, "forest": None, "biome": {}, "grid": None}
    for got in (site.get("relief"), stats.get("relief"), m.get("relief"),
                (m.get("core") or {}).get("relief")):
        if isinstance(got, (int, float)):
            out["relief"] = float(got)
            break
    for got in (site.get("water_pct"), m.get("water_pct")):
        got = _share(got)
        if got is not None:
            out["water"] = got
            break
    if out["water"] is None and site.get("surface_blocks"):
        blocks = site["surface_blocks"]
        total = sum(int(v) for v in blocks.values()) or 1
        out["water"] = sum(int(v) for k, v in blocks.items() if "water" in k) / total
    for got in (site.get("forest_pct"), m.get("forest_pct")):
        got = _share(got)
        if got is not None:
            out["forest"] = got
            break
    classes = (m.get("biome") or {}).get("classes") or site.get("biomes") or {}
    out["biome"] = {k: _share(v) or 0.0 for k, v in classes.items()}
    out["grid"] = site.get("mean_grid")
    return out


def _steepest_step(grid) -> float | None:
    """The largest fall between two neighbouring cells of the site's height grid.

        What tells a cliff from a slope: the same total relief spread evenly over a site is a
        hillside, and concentrated into one step it is a face. None where there is no grid,
        which is an honest "this record does not say" and not a pass.
        
    """
    if not grid or len(grid) < 2 or len(grid[0]) < 2:
        return None
    best = 0.0
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            for di, dj in ((0, 1), (1, 0)):
                a, b = i + di, j + dj
                if a < len(grid) and b < len(grid[a]):
                    best = max(best, abs(float(grid[a][b]) - float(v)))
    return best


def _rim_water(site: dict | None) -> float | None:
    """How much of the site's own edge is water, from its surface census. None if unsaid.

        An island's rim is water on every side; a lakeside's is water on one. The site
        record's `edge_blocks` carries the census of the border columns where a search wrote
        one; without it this returns None and the clause says so rather than certifying.
        
    """
    site = site or {}
    edge = (site.get("edge_blocks")
            or ((site.get("measures") or {}).get("edge") or {}).get("blocks")
            or ((site.get("chosen") or {}).get("edge_blocks")))
    if isinstance(edge, dict) and edge:
        total = sum(int(v) for v in edge.values()) or 1
        return sum(int(v) for k, v in edge.items() if "water" in str(k)) / total
    grid = site.get("water_grid") or ((site.get("measures") or {}).get("water_grid"))
    if grid and len(grid) >= 3 and len(grid[0]) >= 3:
        rim = [float(v) for i, row in enumerate(grid) for j, v in enumerate(row)
               if i in (0, len(grid) - 1) or j in (0, len(row) - 1)]
        return (sum(1 for v in rim if v) / len(rim)) if rim else None
    return None


def _dished(grid) -> float | None:
    """How much lower a site's middle is than its rim, in blocks. A valley is dished."""
    if not grid or len(grid) < 3 or len(grid[0]) < 3:
        return None
    rim, core = [], []
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            edge = i in (0, len(grid) - 1) or j in (0, len(row) - 1)
            (rim if edge else core).append(float(v))
    if not rim or not core:
        return None
    return sum(rim) / len(rim) - sum(core) / len(core)


#: What each setting the sentence can name **is**, on the ground the search chose.
#: Registered here, so a run cannot pass a setting by choosing the threshold afterwards.
#: `cliff_step` and `island_land` are the realization round's, and both are the review's
#: finding that these clauses certified properties they never measured: relief alone
#: made a hillside a cliff, and a water share alone made a lake shore an island. A cliff
#: is a *steep* fall and an island is land with water on every side of it, and both of
#: those are questions the site's own height grid and surface census can answer.
SETTING_BOUNDS = {"cliff": 24.0, "cliff_step": 8.0, "mountain": 40.0, "valley": 4.0,
                  "forest": 0.25, "river": 0.02, "island": 0.30, "island_land": 0.55}


def _setting_measure(what: str, site: dict | None) -> tuple:
    """`(status, why)` -- does the chosen site stand where the sentence says?

        Every clause is a number off the site record. A setting this build has no
        measurement for is `unresolved` and names what it would need, which is the honest
        answer and the one that keeps the obligation visible.
        
    """
    f = site_facts(site)
    relief, water, forest = f["relief"], f["water"], f["forest"]
    biome = f["biome"]
    if what == "cliff":
        if relief is None:
            return ("unresolved", "the site record carries no relief, so a cliff is "
                                  "neither measured nor missed")
        bound = SETTING_BOUNDS["cliff"]
        if relief < bound:
            return ("failed", f"the site falls {relief:g} blocks across itself and a "
                              f"cliff in this build is {bound:g}; the place does not "
                              f"stand on one")
        # **Relief alone is not a cliff, and this used to certify one.** The review:
        # "Site relief alone certifies a cliff; no local slope, enclosure or channel
        # geometry is needed." A site that falls 30 blocks evenly across 200 columns is
        # a hillside. A cliff is that fall happening *somewhere*, steeply, and the site
        # record carries the grid that would say so -- so where it does, the steepest
        # step is measured, and where it does not, this says what it would need instead
        # of passing.
        step = _steepest_step(f["grid"])
        if step is None:
            return ("unresolved",
                    f"the site falls {relief:g} blocks across itself, at or past the "
                    f"{bound:g} this build calls a cliff, and its record carries no "
                    f"height grid -- so whether that fall is a cliff or an even slope is "
                    f"not something this site record says")
        need = SETTING_BOUNDS["cliff_step"]
        return (("satisfied", f"the site falls {relief:g} blocks across itself and "
                              f"{step:.0f} of them across one step of its own grid, at "
                              f"or past the {need:g} this build calls a cliff face")
                if step >= need else
                ("failed", f"the site falls {relief:g} blocks across itself but never "
                           f"more than {step:.0f} across one step of its grid, against "
                           f"the {need:g} a cliff face would put there; this is a slope "
                           f"and not a cliff"))
    if what == "mountain":
        share = max(biome.get("mountain", 0.0), biome.get("snowy", 0.0))
        if relief is None:
            return ("unresolved", "the site record carries no relief")
        bound = SETTING_BOUNDS["mountain"]
        return (("satisfied", f"{relief:g} blocks of relief and {share:.0%} mountain "
                              f"biome") if relief >= bound or share >= 0.2 else
                ("failed", f"{relief:g} blocks of relief and {share:.0%} mountain "
                           f"biome, against {bound:g} blocks or a fifth of the site"))
    if what == "valley":
        dish = _dished(f["grid"])
        if dish is None or relief is None:
            return ("unresolved", "the site record carries no height grid, so whether "
                                  "the ground is dished is not something it says")
        bound = SETTING_BOUNDS["valley"]
        return (("satisfied", f"the middle of the site lies {dish:.1f} blocks below its "
                              f"rim across {relief:g} blocks of relief")
                if dish >= bound else
                ("failed", f"the middle of the site lies {dish:.1f} blocks below its "
                           f"rim, against the {bound:g} this build calls a valley"))
    if what == "forest":
        share = forest if forest is not None else (biome.get("forest", 0.0)
                                                   + biome.get("jungle", 0.0))
        bound = SETTING_BOUNDS["forest"]
        return (("satisfied", f"{share:.0%} of the site is wooded")
                if share >= bound else
                ("failed", f"{share:.0%} of the site is wooded, against the {bound:.0%} "
                           f"this build calls a forest"))
    if what == "river":
        if water is None:
            return ("unresolved", "the site record carries no water share")
        bound = SETTING_BOUNDS["river"]
        return (("satisfied", f"{water:.1%} of the site is water and the biomes include "
                              f"{'river' if biome.get('river') else 'standing water'}")
                if water >= bound else
                ("failed", f"{water:.1%} of the site is water, against the {bound:.0%} "
                           f"a river crossing it would put there"))
    if what == "island":
        if water is None:
            return ("unresolved", "the site record carries no water share")
        bound = SETTING_BOUNDS["island"]
        if water < bound:
            return ("failed", f"{water:.0%} of the site is water, against the "
                              f"{bound:.0%} an island would stand in")
        # **Water on every side, not water somewhere.** A site that is a third water
        # because a lake fills one corner is a lakeside, and this certified it an
        # island. An island's own rim is water; the site's edge is what says so.
        rim = _rim_water(site)
        if rim is None:
            return ("unresolved",
                    f"{water:.0%} of the site is water and its record does not say "
                    f"where; whether the land has water on every side of it, which is "
                    f"what an island is, is not something this site record answers")
        need = SETTING_BOUNDS["island_land"]
        return (("satisfied", f"{water:.0%} of the site is water and {rim:.0%} of its "
                              f"own rim is, so the land it holds has water around it")
                if rim >= need else
                ("failed", f"{water:.0%} of the site is water and only {rim:.0%} of its "
                           f"rim is, against the {need:.0%} an island's own edge would "
                           f"be; the water is beside this place and not around it"))
    return ("unresolved", f"this build has no measurement for `{what}` on a site record")


def _storey_band(decl: dict | None) -> tuple | None:
    """The `storeys` band a type declares, as `(low, high)`, or None."""
    spec = ((decl or {}).get("params") or {}).get("storeys")
    if not spec or len(list(spec)) < 3 or spec[0] != "int":
        return None
    return (float(spec[1]), float(spec[2]))


def _exact_storeys_measure(want: int, plots: list, decls: dict,
                           parts_record: dict | None) -> tuple:
    """`(status, why, evidence)` -- do the buildings stand the storeys the sentence asked?

        **Exactly, on every one of them, off what construction emitted.** A share of a band
        is the wrong instrument for a number: sixteen cottages of which ten reach two
        storeys is 87% of the way up a 1..3 band and is still not what was asked. Before
        anything is built this reads the planned parameter and says so; once a parts record
        exists the planned number is not evidence at all, because a lot too short emits
        fewer storeys and returns success.
        
    """
    emitted = emitted_of(parts_record)
    rows, short, unmeasured = [], [], []
    for p in plots:
        band = _storey_band(decls.get(str(p.get("type") or "")))
        planned = (p.get("params") or {}).get("storeys")
        if band is None and planned is None:
            continue                      # not a building with storeys to count
        e = emitted.get(p.get("name")) if emitted else None
        got = e.get("storeys") if isinstance(e, dict) else None
        if got is None and emitted and parts_record:
            unmeasured.append(p.get("name"))
            continue
        if got is None:
            got = planned
        if not isinstance(got, (int, float)):
            unmeasured.append(p.get("name"))
            continue
        rows.append((p.get("name"), int(got)))
        if int(got) != int(want):
            short.append((p.get("name"), int(got)))
    read_off = "emitted" if (emitted and parts_record) else "planned params"
    if not rows:
        return ("unresolved",
                f"no building of this design records the storeys it stands, so the "
                f"{want} the sentence asks for is not measurable on it",
                {"want": int(want), "measured": 0, "plots": len(plots)})
    ev = {"want": int(want), "measured": len(rows), "plots": len(plots),
          "short": [{"part": n, "storeys": s} for n, s in short[:12]],
          "unmeasured": unmeasured[:12], "read_off": read_off}
    if short or unmeasured:
        return ("failed",
                (f"{len(rows) - len(short)} of {len(plots)} building(s) stand {want} "
                 f"storey(s)"
                 + (f"; {len(short)} stand "
                    f"{sorted({s for _n, s in short})} instead" if short else "")
                 + (f"; {len(unmeasured)} were not measured" if unmeasured else "")
                 + f" (read off {'what construction emitted' if read_off == 'emitted' else 'the planned params'})"),
                ev)
    return ("satisfied",
            (f"all {len(rows)} building(s) stand {want} storey(s), read off "
             f"{'what construction emitted' if read_off == 'emitted' else 'the planned params'}"),
            ev)


def _quality_measure(axis: str, value: str, bound: float, parts: list,
                     resolution: dict | None, decls: dict,
                     scope: str | None = None, parts_record: dict | None = None,
                     wants: dict | None = None) -> tuple:
    """`(status, why, evidence)` -- is the fabric as the sentence says it is?

        `density` is how much of the districts' ground the lots actually occupy, and
        `height` is how far up its own declared band each building was built. Both are
        measured on the plan and both are decisions a compiler can change, which is what
        makes them repairable rather than adjectives.
        
    """
    plots = [p for p in parts if p.get("kind", "plot") == "plot"]
    if not plots:
        return ("open", "nothing is planned yet", {})
    if axis == "height" and (wants or {}).get("exact") and (wants or {}).get("storeys"):
        return _exact_storeys_measure(int(wants["storeys"]), plots, decls, parts_record)
    if axis == "density":
        # **The one metric, over the one denominator.** See `density_target`: the lots'
        # share of the developable ground of the districts this requirement is about --
        # its own quarter's for a scoped word, every district's otherwise.
        regions = []
        for r in (resolution or {}).get("regions") or []:
            if scope and not _in_scope({"name": r.get("name"),
                                        "defines": r.get("part") or r.get("defines"),
                                        "in": [r.get("name")]}, scope):
                continue
            if r.get("lots") is not None and r.get("rect"):
                regions.append(r)
        if not regions:
            for p in parts:
                if p.get("kind") == "district" and p.get("x1") is not None:
                    regions.append({"rect": [p["x0"], p["z0"], p["x1"], p["z1"]],
                                    "developable_columns": p.get("developable_columns")})
        if not regions:
            return ("unresolved", "no district of this design records the ground it "
                                  "covers, so how densely it is built is not measurable",
                    {"plots": len(plots)})
        m = lot_cover(plots, regions)
        t = density_target(value)
        got = m["cover"]
        if got is None:
            return ("unresolved", "the districts this is about record no ground", m)
        # **A lot cover bought by emptying the lots is not density.** The design review:
        # "Enlarging empty lots or excluding unused ground cannot establish density",
        # and this measure decided on allocated lot cover alone while computing the
        # built figure beside it and never consulting it. Both are reported now and the
        # built one can refuse: `fill` is how much of the allocated ground carries
        # emitted mass, and below `MIN_LOT_FILL` the allocation is emptier than anything
        # this library's own types build (see the constant).
        fill = (round(m["built_columns"] / m["built"], 4)
                if m.get("built") and m.get("built_columns") is not None else None)
        # **...and what the street sees.** The spatial stream's two measures, on the
        # districts this requirement is about. Reported beside the cover figure, never
        # instead of it, and `unavailable` where the record cannot answer.
        occupation, enclosure = [], []
        with _contextlib.suppress(Exception):
            from . import placeplan as _pp
            for r in regions:
                d = dict(r, name=r.get("name"),
                         **({} if r.get("rect") is None else
                            {"x0": r["rect"][0], "z0": r["rect"][1],
                             "x1": r["rect"][2], "z1": r["rect"][3]}))
                occupation.append({"district": r.get("name"),
                                   **_pp.built_occupation(d, parts_record)})
                # **...and the enclosure is measured on this district's own leaves.**
                # The neighbourhood round, found by reading the call: `leaves` was
                # `None`, `street_enclosure` answers `unavailable` on no leaves by
                # design, and so every district of every reading in the record reported
                # `unavailable` for the one measure the street is judged by. The leaves
                # are the plots standing inside this district's own rectangle.
                mine = ([p for p in plots
                         if _rect_in(_pp.pipeline.part_rect(p), r.get("rect"))]
                        if r.get("rect") else [])
                enclosure.append({"district": r.get("name"), "leaves": len(mine),
                                  **_pp.street_enclosure(d, None, mine or None,
                                                         parts_record=parts_record)})
        ev = {"coverage": round(got, 4), "lo": t["lo"], "hi": t["hi"], "bound": bound,
              "plots": len(plots), "district_columns": m["ground"],
              "denominator": m["denominator"], "metric": t["metric"],
              "built_columns": m.get("built_columns"),
              "built_cover": (round(m["built_cover"], 4)
                              if m.get("built_cover") is not None else None),
              "allocated_columns": m.get("built"),
              "lot_fill": fill, "min_lot_fill": MIN_LOT_FILL,
              "built_occupation": occupation[:8], "street_enclosure": enclosure[:8]}
        in_band = ((t["lo"] is None or got >= t["lo"])
                   and (t["hi"] is None or got <= t["hi"]))
        empty = fill is not None and fill < MIN_LOT_FILL
        ok = in_band and not empty
        band = ((f"at least {t['lo']:.0%}" if t["lo"] is not None else "")
                + (" and " if t["lo"] is not None and t["hi"] is not None else "")
                + (f"at most {t['hi']:.0%}" if t["hi"] is not None else ""))
        return (("satisfied" if ok else "failed"),
                (f"the lots cover {got:.1%} of the districts' "
                 f"{'developable' if m['denominator'] != 'rect' else 'rectangle'} "
                 f"ground against the {band} this build calls {value}"
                 + (f", and the buildings on them cover {m['built_cover']:.1%} of the "
                    f"same ground" if m.get("built_cover") is not None else "")
                 + (f" -- but the emitted mass fills only {fill:.0%} of the ground the "
                    f"lots were allocated, under the {MIN_LOT_FILL:.0%} that is the "
                    f"least any type in this library builds, so the cover figure is "
                    f"allocation and not fabric" if empty else "")), ev)
    if axis == "height":
        # **What was built, where construction has said.** The review's counterexample:
        # `cottage.build(storeys=3)` on a 9x9 lot emits one storey and the production
        # height helper read the planned three. `emitted.storeys` (interface I3) is the
        # measurement; the planned parameter is what is read only before anything is
        # built, and the evidence says which was read.
        emitted = emitted_of(parts_record)
        rows, read_off = [], ("emitted" if emitted else "planned params")
        for p in plots:
            band = _storey_band(decls.get(str(p.get("type") or "")))
            if band is None or band[1] <= band[0]:
                continue
            got = (p.get("params") or {}).get("storeys")
            e = emitted.get(p.get("name"))
            if e is not None and isinstance(e.get("storeys"), (int, float)):
                got = e["storeys"]
            elif emitted and parts_record:
                continue          # built, and this part's outcome was not measured
            if not isinstance(got, (int, float)):
                continue
            rows.append((float(got) - band[0]) / (band[1] - band[0]))
        if len(rows) < MEASURED_SHARE_MIN * len(plots):
            return ("unresolved",
                    f"{len(rows)} of {len(plots)} lots carry a storey count against a "
                    f"type that declares a band, so how tall this place was built is "
                    f"not measurable on it",
                    {"measured": len(rows), "plots": len(plots)})
        got = sum(rows) / len(rows)
        ev = {"storey_share": round(got, 3), "bound": bound, "measured": len(rows),
              "read_off": read_off}
        ok = got >= bound if value == "tall" else got <= bound
        return (("satisfied" if ok else "failed"),
                (f"the buildings stand {got:.0%} of the way up the storey band their "
                 f"own types declare, against the {bound:.0%} this build calls {value}"
                 f" (read off {'what construction emitted' if read_off == 'emitted' else 'the planned params'}, {len(rows)} measured)"),
                ev)
    return ("unresolved", f"this build has no measurement for `{axis}`", {})


#: **How a property was established.** The design round's third boundary: the expression
#: round labelled every clause `built_place` the moment every leaf stood, including the
#: clauses it had measured on the plan. Coverage -- which subjects were built -- and
#: method -- what the answer was read off -- are two independent questions, and only
#: their conjunction is built evidence. `observed` read off the built world or off what
#: construction emitted `plan` measured on the planned geometry; true of the drawing,
#: not the place `declared` taken from a type's or a record's own statement of itself
#: `judged` an inspection's verdict, which no measurement can replace `site` measured on
#: the chosen site's own terrain record `unsupported` nothing in this build can decide
#: it
METHODS = ("observed", "plan", "declared", "judged", "site", "unsupported")

#: The method each requirement kind is answered by **when nothing has been built**. A
#: parts record can promote some of them; see `_method_of`.
METHOD_BY_KIND = {
    "count": "plan", "feature": "plan", "absent": "plan", "layout": "plan",
    "relation": "plan", "hierarchy": "plan", "orientation": "plan", "quality": "plan",
    "setting": "site", "identity": "judged", "tradition": "judged",
    "function": "declared", "clause": "unsupported",
}


def _method_of(req: dict, parts_record: dict | None, promoted: dict | None = None) -> str:
    """How this requirement's status was established. See `METHODS`.

        `promoted` is what the measurement itself said it read -- the height measure reports
        `read_off`, the function measure reports whether the features were verified on built
        blocks -- because only the instrument knows. Everything else falls back to the kind's
        own answer, and **nothing is promoted to `observed` without a parts record**: a plan
        is a drawing however complete it is.
        
    """
    said = (promoted or {}).get(req.get("id"))
    if said in METHODS:
        return said
    kind = str(req.get("kind") or "")
    base = METHOD_BY_KIND.get(kind, "plan")
    if not parts_record:
        return "plan" if base in ("observed", "plan") else base
    # presence and absence are counted over the leaves that actually stood
    if kind in ("count", "feature", "absent"):
        return "observed"
    return base


def coverage(intent: dict, spec: dict | None = None, plan: dict | None = None,
             parts_record: dict | None = None, resolution: dict | None = None,
             site: dict | None = None, reading: dict | None = None,
             capabilities: dict | None = None, judgment: dict | None = None) -> tuple:
    """Check every requirement against what was actually planned and built.

        Returns `(intent, findings)` -- the intent record with each requirement's `status`
        and `why` filled in, and a findings record routed to the layer that could repair
        each miss. Both are checked records.

        The states are kept honest by what is in hand:

          - with a **spec** only, a requirement can reach `failed` (the spec omits it) but
            never `satisfied`: a spec that names a wall has not built one;
          - with a **plan**, a feature reaches `planned`, which this records as `open` with
            a reason, because a drawn wall is not a standing wall;
          - with a **parts record**, it reaches `satisfied` or `failed` for good.

        **Nothing here is allowed to read a declaration as a measurement.** The integration
        review's second finding, in three places and all three fixed below: an absence was
        checked against the families the spec declared (so an undeclared wall was invisible),
        a frontage against `resolution.bounds.faces` (a word), and a layout against its own
        policy name. A check whose evidence is the thing being checked has not checked
        anything.
        
    """
    from . import pipeline as _pipeline
    intent = contracts.read("intent", dict(intent))
    parts = _pipeline.plan_parts(plan) if plan else []
    stood = _standing(parts, parts_record)
    scope = sample_scope(parts_record, parts)
    tag = f" (sample of {len(scope)} of {len(parts)} leaves)" if scope else ""
    fams = _families_in(spec) if spec else {}
    decls = _plan_decls(parts) if parts else {}
    findings: list = []
    #: `{requirement id: method}` -- what an instrument says it actually read, where
    #: only the instrument can know. Consumed by `_method_of` below.
    promoted: dict = {}

    def finding(req, says, owner, blocks, **evidence):
        findings.append({"id": f"find/{req['id']}", "says": says,
                         "requirement": req["id"], "part": None,
                         "evidence": dict(evidence), "owner": owner, "blocks": blocks,
                         "severity": "error" if req["hard"] else "warning",
                         "seen_by": "intent.coverage", "fixed": False})

    for req in intent["requirements"]:
        w = req["wants"]
        # **A scoped requirement is measured on its own part of the place.** The whole
        # point of `scope`: "a dense lower district and a sparse upper district" is two
        # measurements over two sets of leaves, and measuring either over the whole
        # place answers a question nobody asked. `mine` is the leaves this requirement
        # is about, and for an unscoped requirement it is every leaf, as it always was.
        mine = ([p for p in parts if _in_scope(p, req.get("scope"))]
                if req.get("scope") else parts)
        if req.get("scope") and parts and not mine:
            req["status"] = "unresolved"
            req["owner"] = "layout"
            req["why"] = (f"this requirement is about `{req['scope']}` and no part of "
                          f"the plan answers that name, so it is neither met nor missed")
            finding(req, req["why"], "layout", "feasibility", scope=req["scope"])
            continue
        if req["status"] == "unsupported":
            # Written at read time and never revised by a later stage: the system does
            # not acquire a harbour by planning a village.
            finding(req, f"{req['says']}: {req['why']}", "capability", "fidelity",
                    phrase=req["phrase"])
            continue
        if req["kind"] == "feature":
            fam, want = w["family"], int(w.get("count") or 1)
            declared = fams.get(fam, 0)
            if spec is None:
                req["status"], req["why"] = "open", "nothing has been read yet"
                continue
            if declared < want:
                req["status"] = "failed"
                req["why"] = (f"the sentence asks for {want} x {fam} and the spec's "
                              f"defining parts declare {declared}")
                req["owner"] = "reading"
                finding(req, req["why"], "reading", "fidelity",
                        family=fam, wanted=want, declared=declared)
                continue
            if not plan:
                req["status"] = "open"
                req["why"] = f"the spec declares {declared} x {fam}; nothing is planned yet"
                continue
            # a landmark the compiler laid (`landmark_workshop`) answers no defining
            # part and is the thing itself: read by what it is, as an absence is
            mine = _leaves_of_family(spec, parts, fam)
            names = {p.get("name") for p in mine}
            mine += [p for p in leaves_built_as(parts, fam, decls)
                     if p.get("name") not in names]
            if len(mine) < want:
                req["status"] = "failed"
                req["why"] = (f"the spec declares {declared} x {fam} and the plan holds "
                              f"{len(mine)} part(s) answering them")
                req["owner"] = "layout"
                finding(req, req["why"], "layout", "feasibility",
                        family=fam, wanted=want, planned=len(mine))
                continue
            if not parts_record:
                req["status"] = "open"
                req["why"] = (f"{len(mine)} part(s) are planned for it; a drawn "
                              f"{fam} is not a standing one")
                continue
            # **A sample qualifies its constructed scope.** Standing is counted over the
            # sampled leaves; a feature none of whose leaves were sampled is neither met
            # nor missed, and says so.
            mine_s = [p for p in mine if scope is None or p.get("name") in scope]
            if scope is not None and not mine_s:
                req["status"] = "open"
                req["why"] = (f"{len(mine)} part(s) are planned for it and none is in "
                              f"the construction sample; {OUTSIDE_SAMPLE}{tag}")
                req["evidence"] = [OUTSIDE_SAMPLE]
                continue
            want_s = min(want, len(mine_s)) if scope is not None else want
            up = [p["name"] for p in mine_s if stood.get(p["name"], False)]
            if len(up) < want_s:
                req["status"] = "failed"
                req["why"] = (f"{len(mine_s)} planned, {len(up)} standing, against "
                              f"{want_s} asked for{tag}")
                req["owner"] = "build"
                finding(req, req["why"], "build", "construction",
                        family=fam, wanted=want_s, standing=len(up))
            else:
                req["status"], req["why"] = "satisfied", f"{len(up)} standing{tag}"
                req["evidence"] = up[:8]
        elif req["kind"] == "absent":
            fam = w["family"]
            if spec is None:
                continue
            if fams.get(fam, 0):
                req["status"] = "failed"
                req["why"] = (f"the sentence says the place has no {fam} and the spec "
                              f"declares {fams[fam]}")
                req["owner"] = "reading"
                finding(req, req["why"], "reading", "fidelity", family=fam)
                continue
            # **Every leaf in the plan, by what it is.** Not by what the spec declared:
            # the whole point of an absence is that it is about something nobody
            # declared, and `_leaves_of_family` starts from the declarations.
            drawn = leaves_built_as(parts, fam, decls) if plan else []
            if drawn:
                req["status"] = "failed"
                req["why"] = (f"the plan draws {len(drawn)} {fam}(s) the sentence says "
                              f"are not there -- {', '.join(d.get('name', '?') for d in drawn[:3])}"
                              f" -- and the spec declares none, so nothing generated "
                              f"from the spec could have seen them")
                req["owner"] = "layout"
                finding(req, req["why"], "layout", "fidelity", family=fam,
                        parts=[d.get("name") for d in drawn[:8]])
            elif not plan:
                req["status"] = "open"
                req["why"] = f"no {fam} is declared; nothing is planned yet"
            else:
                req["status"] = "satisfied"
                req["why"] = (f"no {fam} is declared, and no leaf of the plan is one: "
                              f"{len(parts)} leaf(s) read by their own type")
                req["evidence"] = [f"{len(parts)} leaves checked"]
        elif req["kind"] == "count":
            if spec is None:
                continue
            n, about = int(w["n"]), bool(w["about"])
            # **A count is a count of the thing the sentence counted.** "sixteen
            # cottages ... with a hall" is sixteen cottages and one hall, and a check
            # that counted every plot read seventeen. `select` is the one rule for which
            # leaves the word names; with no `what` every plot is counted, as before.
            what = str(w.get("what") or "")
            if _slug(what) in ("building", "buildings", "structure", "structures"):
                what = ""          # the generic unit: every plot, as before
            mine_c = ([p for p in select(parts, what) if p.get("kind", "plot") == "plot"]
                      if what else [p for p in parts if p.get("kind", "plot") == "plot"])
            if parts_record and scope is not None:
                # the sample's own count: every sampled one stands, and the whole
                # place's planned count is the sentence's, reported beside it
                in_s = [p for p in mine_c if p.get("name") in scope]
                got_s = sum(1 for p in in_s if stood.get(p["name"]))
                planned = len(mine_c)
                lo, hi = ((max(1, int(round(n * (1 - spec_mod.ABOUT)))),
                           int(round(n * (1 + spec_mod.ABOUT)))) if about else (n, n))
                ok = got_s == len(in_s) and lo <= planned <= hi
                req["status"] = "satisfied" if ok else "failed"
                req["why"] = (f"{got_s} of {len(in_s)} sampled {what or 'structures'} "
                              f"stand; {planned} planned in all, against {lo}-{hi}"
                              f"{tag}")
                if not ok:
                    req["owner"] = "build" if planned >= lo else "scale"
                    finding(req, req["why"], req["owner"],
                            "construction" if planned >= lo else "feasibility",
                            asked=[lo, hi], planned=planned, sampled=len(in_s),
                            standing=got_s)
                continue
            if parts_record:
                got = sum(1 for p in mine_c if stood.get(p["name"]))
                where = f"{what or 'structures'} stand"
            elif plan:
                got = len(mine_c)
                where = f"{what or 'structures'} are planned"
            else:
                got, where = int(spec.get("structures") or 0), "are asked for"
            lo, hi = ((max(1, int(round(n * (1 - spec_mod.ABOUT)))),
                       int(round(n * (1 + spec_mod.ABOUT)))) if about else (n, n))
            if lo <= got <= hi:
                req["status"] = "satisfied"
                req["why"] = f"{got} {where}, against {lo}-{hi}"
            else:
                # **The band the spec carries is not consulted here on purpose.** The
                # audit's case: a ceiling reduced "exactly 2000 houses" to 1,824 and
                # moved the band down with it, and `in_band` then passed. The sentence
                # said two thousand. It still says two thousand.
                req["status"] = "failed"
                req["why"] = (f"{got} {where}, against the {lo}-{hi} the sentence asks "
                              f"for")
                req["owner"] = "scale"
                finding(req, req["why"], "scale",
                        "fidelity" if parts_record else "feasibility",
                        asked=[lo, hi], got=got,
                        ceiling=(spec or {}).get("ceiling"),
                        scaled_from=(spec or {}).get("scaled_from"))
        elif req["kind"] == "layout":
            got = (resolution or {}).get("policy")
            if got is None and plan is not None:
                got = ((plan.get("layout") or {}).get("policy")
                       or ("concentric" if (plan.get("layout") or {}).get("rings")
                           else None))
            if got is None:
                req["status"] = "open"
                req["why"] = "no layout policy has been resolved yet"
                continue
            if got == w["policy"]:
                # **And the geometry that policy's name promises is actually there.** A
                # `policy` field is a word a layout wrote about itself; the review's
                # second finding is that coverage was reading exactly such words. What
                # each policy owes the record is checked instead.
                ok, why = _policy_geometry(w["policy"], resolution, plan)
                if ok:
                    req["status"] = "satisfied"
                    req["why"] = f"laid out by the {got} policy: {why}"
                else:
                    req["status"] = "failed"
                    req["why"] = (f"the resolved design says `{got}` and its geometry "
                                  f"does not carry it: {why}")
                    req["owner"] = "layout"
                    finding(req, req["why"], "layout", "feasibility",
                            wanted=w["policy"], got=got)
            else:
                req["status"] = "failed"
                req["why"] = (f"the sentence asks for a {w['policy']} place and the "
                              f"{got} policy laid it out")
                req["owner"] = "layout"
                finding(req, req["why"], "layout", "fidelity",
                        wanted=w["policy"], got=got)
        elif req["kind"] == "orientation":
            got = _fronts(w["faces"], resolution, parts, plan)
            req["status"], req["why"] = got["status"], got["why"]
            if got["status"] in ("failed", "unresolved"):
                req["owner"] = "layout"
                finding(req, req["why"], "layout",
                        "fidelity" if parts_record else "feasibility",
                        wanted=w["faces"], **got["evidence"])
            else:
                req["evidence"] = got["evidence"].get("sample", [])
        elif req["kind"] == "setting":
            if site is None:
                req["status"] = "open"
                req["why"] = "no site has been chosen yet"
                continue
            got, why = _setting_measure(w["setting"], site)
            req["status"], req["why"] = got, why
            if got == "satisfied":
                req["evidence"] = [why]
            else:
                req["owner"] = "site"
                finding(req, f"{req['says']}: {why}", "site",
                        "fidelity" if parts_record else "feasibility",
                        setting=w["setting"], **site_facts(site))
        elif req["kind"] == "quality":
            if not plan:
                req["status"] = "open"
                req["why"] = (f"the sentence says the fabric is {w['value']}; nothing "
                              f"is planned yet")
                continue
            got, why, ev = _quality_measure(w["axis"], w["value"],
                                            float(w.get("bound")
                                                  if w.get("bound") is not None
                                                  else QUALITY_BOUNDS.get(w["value"], 0.0)),
                                            mine, resolution, decls,
                                            scope=req.get("scope"),
                                            parts_record=parts_record, wants=w)
            req["status"], req["why"] = got, why
            # **the height measure says what it read; the density measure reads lots.**
            # A cover figure is the share of the ground the *drawn lots* take, which is
            # a fact about the plan until a built-mass denominator exists.
            promoted[req["id"]] = ("observed" if ev.get("read_off") == "emitted"
                                   else ("observed" if ev.get("built_columns") is not None
                                         else "plan"))
            if got == "satisfied":
                req["evidence"] = [why]
            elif got != "open":
                # **the compiler owns both.** How closely lots are packed is the
                # district compiler's decision and how many storeys a building has is
                # the parameter the same pass chooses; a finding about either goes where
                # it can actually be changed. **A quality the plan misses is a fidelity
                # fact about a buildable plan.** The transfer case measured its lower
                # quarter at 5% against the 30% `dense` asks, the finding blocked
                # feasibility, and the plan stage stopped a place that could have been
                # built, inspected and then refused at the place read for exactly this.
                # Feasibility is for a plan that cannot be laid out or cannot hold what
                # was asked; a density or a height word is answered by the place read,
                # which fails on it.
                req["owner"] = "layout"
                finding(req, req["why"], "layout", "fidelity",
                        axis=w["axis"], value=w["value"], **ev)
        elif req["kind"] == "identity":
            # evidence informs the design, and whether the result is the place is a
            # judgment. What this can do, and what the round requires it to do, is
            # refuse to let the obligation disappear: it is carried, it names what is
            # missing, and it never reads `satisfied`.
            srcs = len((reading or {}).get("sources") or [])
            claims = [c for c in (reading or {}).get("claims") or []
                      if c.get("source")]
            if reading is None:
                req["status"] = "open"
                req["why"] = "nothing has been read about this name yet"
                continue
            # **And the judgment that can close it.** The integration round left this
            # requirement able to read `unresolved` and nothing else, which the review
            # named as the other half of the same defect: a tradition had a false-pass
            # route and an identity had no completion route at all. An obligation that
            # cannot be met is not an obligation, it is a permanent refusal, and a run
            # that did the research and made the judgment had nowhere to record it. Two
            # things close it and neither alone does: sourced claims about the name, and
            # a verdict from an inspection that looked at what was built. A verdict with
            # no evidence behind it is somebody's impression; evidence with no verdict
            # is a design nobody checked.
            verdict = _verdict_for(judgment, "identity", w["name"])
            req["owner"] = "reading"
            usable, u_why = judgment_usable(judgment, verdict, reading)
            said = bool(verdict.get("recognisable")) if verdict is not None else None
            if verdict is not None and claims and (usable or not said):
                req["status"] = "satisfied" if said else "failed"
                req["why"] = (
                    f"{len(claims)} sourced claim(s) from {srcs} source(s) informed the "
                    f"design and the inspection of what was built "
                    f"{'recognises' if said else 'does not recognise'} it as "
                    f"{w['name']}: {verdict.get('why') or 'no reason given'}")
                req["evidence"] = [c.get("id") for c in claims[:6]] + [
                    str(verdict.get("from") or "judgment")]
                if not said:
                    finding(req, req["why"], "reading", "fidelity", name=w["name"],
                            sources=srcs, sourced_claims=len(claims), judged=False)
                continue
            req["status"] = "unresolved"
            if verdict is not None and claims and not usable:
                req["why"] = (f"an inspection judged this place {w['name']} and the "
                              f"judgment cannot close it: {u_why}")
            elif not claims:
                req["why"] = (f"{srcs} source(s) and no sourced claim about "
                              f"{w['name']}: the design of this place is inferred, and "
                              f"whether it is {w['name']} is not established")
            elif verdict is None:
                req["why"] = (f"{len(claims)} sourced claim(s) from {srcs} source(s) "
                              f"informed the design and no inspection has judged "
                              f"whether the result is recognisably {w['name']}; the "
                              f"evidence is in hand and the judgment is not")
                req["evidence"] = [c.get("id") for c in claims[:8]]
            else:
                req["why"] = (f"an inspection judged this place {w['name']} and no "
                              f"sourced claim about {w['name']} informed its design, so "
                              f"the judgment has nothing behind it")
            finding(req, req["why"], "reading", "fidelity", name=w["name"],
                    sources=srcs, sourced_claims=len(claims),
                    judged=verdict is not None)
        elif req["kind"] == "tradition":
            t, nearest = w["tradition"], w.get("nearest_form")
            form = (spec or {}).get("form")
            declared = sorted({str(e.get("type")) for e in
                               (capabilities or {}).get("entries") or []
                               if e.get("matched")
                               and str((e.get("envelope") or {}).get("tradition")
                                       or "") == t})
            # **A type the plan uses, not a type the record named.** The review's
            # counterexample was one capability entry carrying `tradition: japanese` and
            # **no plan at all** satisfying a request for a Japanese village: the record
            # is a statement of what was selected, and what was selected is not what was
            # built until the plan says so.
            used = sorted({str(p.get("type")) for p in parts
                           if str(p.get("type") or "") in declared})
            # **The inspection's verdict on the built place comes first, both ways.** A
            # negative judgment fails the tradition whatever the types declare -- a
            # label cannot overrule what a judge saw standing -- and a positive one
            # closes it only where it is usable: judged on built output, citing claims
            # this run actually sourced. A positive verdict that is not usable leaves
            # the obligation unresolved and names why.
            verdict = _verdict_for(judgment, "tradition", t)
            v_said = ((verdict.get("holds") if verdict.get("holds") is not None
                       else verdict.get("recognisable")) if verdict is not None else None)
            v_ok, v_why = judgment_usable(judgment, verdict, reading)
            if verdict is not None and v_said is False:
                req["status"] = "failed"
                req["owner"] = "capability"
                req["why"] = (f"the inspection of what was built does not find it built "
                              f"in the {t} tradition: {verdict.get('why') or 'no reason'}"
                              + (f" ({len(used)} declaring type(s) in the plan, which a "
                                 f"judgment of the built place overrules)" if used else ""))
                finding(req, req["why"], "capability", "fidelity", tradition=t,
                        judged=False, declared=declared[:8], used=used[:8])
            elif verdict is not None and v_said and v_ok:
                req["status"] = "satisfied"
                req["why"] = (f"the inspection of what was built finds it in the {t} "
                              f"tradition ({v_why}): {verdict.get('why') or ''}")
                req["evidence"] = [str(c) for c in verdict.get("cites") or []] + used[:4]
            elif verdict is not None and v_said and not v_ok:
                req["status"] = "unresolved"
                req["owner"] = "reading"
                req["why"] = (f"an inspection judged this place built in the {t} "
                              f"tradition and the judgment cannot close it: {v_why}")
                finding(req, req["why"], "reading", "fidelity", tradition=t,
                        judged=True, usable=False)
            elif spec is None:
                req["status"] = "open"
                req["why"] = "nothing has been read yet"
            elif declared and not plan:
                req["status"] = "open"
                req["why"] = (f"{len(declared)} type(s) declaring the {t} tradition were "
                              f"selected ({', '.join(declared[:4])}); nothing is planned "
                              f"yet, and a selected type is not a built one")
            elif declared and not used:
                req["status"] = "failed"
                req["why"] = (f"{len(declared)} type(s) declaring the {t} tradition were "
                              f"selected ({', '.join(declared[:4])}) and the plan's "
                              f"{len(parts)} leaf(s) use none of them")
                req["owner"] = "capability"
                finding(req, req["why"], "capability", "fidelity", tradition=t,
                        declared=declared[:8], used=[])
            elif used:
                # **How much of the place is built that way, not whether one part is.**
                # The review's counterexample: one Japanese-labelled type among 99 other
                # plots satisfied the entire tradition requirement, before anything was
                # built. A tradition is a property of the fabric, so it is measured as a
                # share of the leaves, against a bar registered here rather than chosen
                # after a run. Below the bar it is `unresolved` and names the share --
                # the tradition is present and the place is not built in it.
                mine_t = [p for p in parts if str(p.get("type") or "") in set(used)
                          and (not parts_record or stood.get(p["name"], False))]
                plots_t = [p for p in parts if p.get("kind", "plot") == "plot"]
                share = len(mine_t) / float(len(plots_t)) if plots_t else 0.0
                if share >= TRADITION_SHARE:
                    req["status"] = "satisfied"
                    req["why"] = (f"{len(mine_t)} of {len(plots_t)} leaf/leaves "
                                  f"({share:.0%}) are built of type(s) declaring the {t} "
                                  f"tradition: {', '.join(used[:4])}")
                    req["evidence"] = used[:8]
                else:
                    req["status"] = "unresolved"
                    req["why"] = (f"{len(mine_t)} of {len(plots_t)} leaf/leaves "
                                  f"({share:.0%}) are built of type(s) declaring the {t} "
                                  f"tradition ({', '.join(used[:4])}), against the "
                                  f"{TRADITION_SHARE:.0%} this build asks before it "
                                  f"calls a place built in one; the tradition is present "
                                  f"in the place and the place is not built in it")
                    req["owner"] = "capability"
                    finding(req, req["why"], "capability", "fidelity", tradition=t,
                            declared=declared[:8], used=used[:8],
                            share=round(share, 3), bound=TRADITION_SHARE)
            elif form != nearest:
                req["status"] = "failed"
                req["why"] = (f"the sentence asks for {t} construction, whose nearest "
                              f"form family in this library is `{nearest}`, and the "
                              f"spec chose `{form}`")
                req["owner"] = "reading"
                finding(req, req["why"], "reading", "fidelity", tradition=t,
                        nearest=nearest, got=form)
            else:
                req["status"] = "unresolved"
                req["why"] = (f"the place is built in `{form}`, which is the nearest "
                              f"form family this library has to {t}; no committed type "
                              f"declares the {t} tradition itself, so the tradition is "
                              f"not qualified -- only approximated")
                req["owner"] = "capability"
                finding(req, req["why"], "capability", "fidelity", tradition=t,
                        nearest=nearest)
        elif req["kind"] == "hierarchy":
            # **"a big temple" among "small houses" is a measurement.** See
            # `_hierarchy_measure`; the relative hierarchy the review found disappearing
            # between the sentence and the design is checked on the plan that exists.
            if not plan:
                req["status"] = "open"
                req["why"] = f"{req['says']}; nothing is planned yet"
                continue
            got, why, ev = _hierarchy_measure(w, mine, decls)
            req["status"], req["why"] = got, why
            if got == "satisfied":
                req["evidence"] = [why]
            else:
                req["owner"] = "layout"
                finding(req, req["why"], "layout",
                        "fidelity" if parts_record else "feasibility", **ev)
        elif req["kind"] == "relation":
            if not plan:
                req["status"] = "open"
                req["why"] = f"{req['says']}; nothing is planned yet"
                continue
            got, why, ev = _relation_measure(w, mine)
            req["status"], req["why"] = got, why
            if got == "satisfied":
                req["evidence"] = [why]
            else:
                req["owner"] = "layout"
                finding(req, req["why"], "layout",
                        "fidelity" if parts_record else "feasibility", **ev)
        elif req["kind"] == "function":
            got, why, ev = _function_measure(w, parts if plan else [], capabilities,
                                             parts_record)
            req["status"], req["why"] = got, why
            # a function is `observed` only where the predicates that decide use ran on
            # the assembled world; a type declaring the function is a declaration
            promoted[req["id"]] = str(ev.get("method") or "declared")
            if got == "satisfied":
                req["evidence"] = [why]
            elif got != "open":
                req["owner"] = "capability"
                finding(req, req["why"], "capability", "fidelity", **ev)
        elif req["kind"] == "clause":
            word = w["word"]
            where = _mentions(spec, word)
            req["status"] = "unresolved"
            req["owner"] = "reading"
            req["why"] = (f"nothing in this build reads the word {word!r}; "
                          + (f"the spec mentions it in {', '.join(where[:3])}, which is "
                             f"prose and not a checked part"
                             if where else
                             "no part of the programme answers it, and it is neither "
                             "planned nor refused"))
            finding(req, req["why"], "reading", "fidelity", word=word,
                    mentioned_in=where[:4])
    findings += emitted_findings(parts, parts_record)
    for req in intent["requirements"]:
        req["method"] = _method_of(req, parts_record, promoted)
    return (contracts.read("intent", intent),
            contracts.make("findings", findings=findings, stage="intent.coverage",
                           note="every miss routed to the layer that can repair it"))


def holds(intent: dict) -> bool:
    """Does every hard requirement hold? **An open requirement does not.**"""
    return not contracts.unmet(intent, hard_only=True)


def says(intent: dict) -> str:
    """One line, for a readout."""
    by = contracts.status_of(intent)
    return ", ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())) or "no requirements"
