"""Finding out about a request at runtime, from sources, with the sources written down.

The audit's first finding: `stage_place_spec` asks a model for a place schema and there
is no stage anywhere that **acquires** anything. its recollection is not evidence,
nobody can check it, and a run cannot tell the difference between a fact and a confident
sentence. Worse, the one thing that *was* outside the loop -- a supervising agent
browsing and hand-writing a reference sheet -- is the capability the whole system is
supposed to have.

So: a bounded retrieval interface, a configurable provider behind it, and a reading
record whose every claim either cites a source that was actually fetched or says it was
inferred (`contracts.read("reading")` refuses anything else).

    classify   named reference, architectural tradition, or self-contained description.
    queries    what to go and look for, derived from the sentence -- never from a
               hardcoded table of places.
    retrieve   through a provider: `none`, `recorded` (a cassette, for deterministic
               tests and warm replay) or `http` (a real configured search endpoint).
    persist    id, title, url, access date, **content fingerprint**, media type, the
               claim it supports, and the bytes, under `<state>/evidence/`.

What this module deliberately does **not** do is invent. A provider that is not
configured returns nothing and says so; the reading is then honestly an inferred one and
every claim in it is marked `inferred`, which is a different and visibly weaker thing
than a sourced reading. `needs_evidence` is what a caller asks to find out whether a run
is in that state.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

from . import contracts

#: How many sources one reading may acquire. A bound, not a target: a reading is a
#: handful of facts about a place, and an unbounded crawl is a different program.
SOURCE_CAP = 6

#: How many bytes of one source are kept and shown to the reader.
SOURCE_BYTES = 200_000

#: How long one fetch may take.
TIMEOUT = 30

#: What a retrieval provider may be. `none` is the default, and the default is honest.
PROVIDERS = ("none", "recorded", "http")

#: Words that make a request an **architectural tradition** rather than a named place. A
#: tradition is a *how*, a named place is a *which*, and they are searched for
#: differently.
TRADITIONS = ("japanese", "chinese", "korean", "german", "bavarian", "french",
              "italian", "tuscan", "english", "tudor", "norse", "viking", "scandinavian",
              "dutch", "spanish", "moorish", "ottoman", "persian", "mughal", "aztec",
              "mayan", "inca", "swiss", "alpine", "russian", "greek", "roman",
              "medieval", "renaissance", "georgian", "victorian")

#: Words that say the sentence describes the place itself rather than naming one.
DESCRIBED = ("build a", "build an", "a village", "a town", "a city", "a hamlet",
             "with a", "following the", "facing the")


def classify(sentence: str) -> dict:
    """What kind of request this is, and the name or tradition it turns on.

        Rules, not a model call: a run has to be able to say *why* it went looking for
        something, and "the model thought it was a place" is not a reason anybody can check.
        
    """
    s = (sentence or "").strip()
    low = s.lower()
    tradition = next((t for t in TRADITIONS if re.search(rf"\b{t}\b", low)), None)
    # A proper noun the sentence names: two or more capitalised words in a row, or one
    # capitalised word that is not the first of the sentence and not a common noun.
    body = re.sub(r"^\s*(build|make|create)\s+", "", s, flags=re.I)
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", body)
    single = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", body)
    name = (caps[0] if caps else (single[0] if single else None))
    if name and tradition and name.lower() == tradition:
        name = None
    if name:
        return {"kind": "named_reference", "name": name, "tradition": tradition,
                "why": f"the sentence names {name!r}, which this build has no facts "
                       f"about until it goes and finds some"}
    if tradition:
        return {"kind": "tradition", "name": None, "tradition": tradition,
                "why": f"the sentence asks for {tradition} construction, which is a "
                       f"question about how buildings are made"}
    return {"kind": "self_contained", "name": None, "tradition": None,
            "why": "the sentence describes the place it wants; nothing has to be looked "
                   "up for it"}


def queries(sentence: str, what: dict | None = None) -> list:
    """What to go and look for. Derived from the sentence, never from a table of places.

        Three questions at most, and each one is about something this system can act on:
        what the place is organised like, what its buildings are made of and how big it is.
        
    """
    what = what or classify(sentence)
    if what["kind"] == "named_reference":
        n = what["name"]
        return [f"{n} city layout plan walls districts",
                f"{n} architecture buildings roofs materials",
                f"{n} size scale description"]
    if what["kind"] == "tradition":
        t = what["tradition"]
        return [f"{t} village layout streets plan",
                f"{t} vernacular house construction roof materials"]
    return []


# ------------------------------------------------------------------ providers

class NoProvider:
    """No retrieval is configured. Returns nothing and says why, every time."""

    name = "none"

    def __init__(self, why: str = ""):
        self.why = why or ("no retrieval provider is configured: set ETHOSLM_RETRIEVAL "
                           "to `recorded` (a cassette) or `http` (a search endpoint), "
                           "and see models.json's `retrieval` block")

    def search(self, query: str, most: int = 4) -> list:
        return []

    def fetch(self, url: str) -> dict | None:
        return None


class RecordedProvider:
    """A cassette on disk: what a previous real retrieval returned.

        This is what makes the whole path testable and replayable. `<dir>/index.json` maps
        a query to its results and a url to the file holding its bytes; nothing reaches the
        network. A recorded run is labelled `recorded` on every source it produces, so no
        report can mistake it for a fresh one.
        
    """

    name = "recorded"

    def __init__(self, root: str):
        self.root = root
        p = os.path.join(root, "index.json")
        self.index = json.load(open(p)) if os.path.exists(p) else {"queries": {},
                                                                   "pages": {}}

    def search(self, query: str, most: int = 4) -> list:
        return list((self.index.get("queries") or {}).get(query) or [])[:most]

    def fetch(self, url: str) -> dict | None:
        row = (self.index.get("pages") or {}).get(url)
        if not row:
            return None
        p = os.path.join(self.root, row["file"])
        if not os.path.exists(p):
            return None
        return {"content": open(p, "rb").read()[:SOURCE_BYTES],
                "media": row.get("media", "text/plain"),
                "title": row.get("title") or url}


class HttpProvider:
    """A real, configured search endpoint and plain HTTP fetch.

        Deliberately generic: `models.json`'s `retrieval` block names the URL, the method,
        the query parameter, the key's environment variable and where the results live in
        the answer, so any JSON search API -- Brave, SearXNG, Tavily, a proxy of your own --
        is configuration rather than code.
        
    """

    name = "http"

    def __init__(self, conf: dict, env=None):
        env = dict(os.environ if env is None else env)
        self.conf = dict(conf or {})
        self.url = env.get(self.conf.get("search_url_env") or "",
                           self.conf.get("search_url") or "")
        self.key = env.get(self.conf.get("key") or "") if self.conf.get("key") else None
        self.method = (self.conf.get("method") or "GET").upper()
        self.param = self.conf.get("query_param") or "q"
        self.path = self.conf.get("results_path") or "results"
        self.keys = {"title": self.conf.get("title_key") or "title",
                     "url": self.conf.get("url_key") or "url",
                     "snippet": self.conf.get("snippet_key") or "snippet"}
        self.headers = dict(self.conf.get("headers") or {})
        if self.key and self.conf.get("key_header"):
            self.headers[self.conf["key_header"]] = self.key

    @property
    def ready(self) -> str | None:
        if not self.url:
            return "no search_url is configured for the http retrieval provider"
        if self.conf.get("key") and not self.key:
            return f"{self.conf['key']} is not set"
        return None

    def _dig(self, doc, path: str):
        for step in path.split("."):
            if isinstance(doc, dict):
                doc = doc.get(step)
            elif isinstance(doc, list) and step.isdigit():
                doc = doc[int(step)]
            else:
                return None
        return doc

    def search(self, query: str, most: int = 4) -> list:
        if self.ready:
            return []
        if self.method == "GET":
            url = self.url + ("&" if "?" in self.url else "?") + \
                urllib.parse.urlencode({self.param: query})
            req = urllib.request.Request(url, headers=self.headers)
        else:
            body = json.dumps({self.param: query, **(self.conf.get("body") or {})})
            req = urllib.request.Request(
                self.url, data=body.encode(),
                headers={**self.headers, "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:   # noqa: S310
            doc = json.loads(fh.read().decode("utf-8", "replace"))
        rows = self._dig(doc, self.path) or []
        out = []
        for r in rows[:most]:
            if not isinstance(r, dict):
                continue
            out.append({"title": str(r.get(self.keys["title"]) or ""),
                        "url": str(r.get(self.keys["url"]) or ""),
                        "snippet": str(r.get(self.keys["snippet"]) or "")[:800]})
        return [r for r in out if r["url"]]

    def fetch(self, url: str) -> dict | None:
        req = urllib.request.Request(url, headers={"user-agent": "ethoslm/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:   # noqa: S310
                media = fh.headers.get_content_type()
                blob = fh.read(SOURCE_BYTES)
        except Exception:                      # noqa: BLE001 -- a source that will not
            return None                        # fetch is a source this run does not have
        return {"content": blob, "media": media, "title": url}


def provider(env=None, models: dict | None = None):
    """The configured retrieval provider, or `NoProvider` with the reason."""
    env = dict(os.environ if env is None else env)
    want = (env.get("ETHOSLM_RETRIEVAL") or "none").strip().lower()
    if want not in PROVIDERS:
        return NoProvider(f"ETHOSLM_RETRIEVAL is {want!r}; it is one of {list(PROVIDERS)}")
    if want == "none":
        return NoProvider()
    if want == "recorded":
        root = env.get("ETHOSLM_RETRIEVAL_CASSETTE") or ""
        if not root or not os.path.isdir(root):
            return NoProvider(f"ETHOSLM_RETRIEVAL=recorded needs "
                              f"ETHOSLM_RETRIEVAL_CASSETTE to name a directory; "
                              f"{root!r} is not one")
        return RecordedProvider(root)
    if models is None:
        from . import model as model_mod
        try:
            models = model_mod.load_models(env=env)
        except Exception:                      # noqa: BLE001 -- reported below
            models = {}
    conf = ((models.get("retrieval") or {}).get("providers") or {}).get(
        env.get("ETHOSLM_RETRIEVAL_PROVIDER") or
        (models.get("retrieval") or {}).get("default") or "search")
    p = HttpProvider(conf or {}, env=env)
    why = p.ready
    return p if not why else NoProvider(f"the http retrieval provider is not usable: "
                                        f"{why}")


# ------------------------------------------------------------------ the gathering

def _fingerprint(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:32]


def _text_of(blob: bytes, media: str) -> str:
    """The readable text of a fetched page. Tags stripped, entities left alone.

        Nine lines rather than a dependency, for the reason the rest of this project gives:
        what the reader needs is prose to quote, and a parser that is wrong about a `<div>`
        is not wrong about the sentence inside it.
        
    """
    s = blob.decode("utf-8", "replace")
    if "html" in (media or ""):
        s = re.sub(r"(?is)<(script|style|nav|footer)[^>]*>.*?</\1>", " ", s)
        s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def gather(sentence: str, out_dir: str, *, prov=None, cap: int = SOURCE_CAP) -> dict:
    """Go and find out about this request. Returns the raw gathering, not a reading.

        A gathering is sources and the text of each; turning it into **claims** is a
        reading, and that is a model's job with the sources in front of it. Keeping the two
        apart is what lets a claim carry the id of the source it came from.
        
    """
    what = classify(sentence)
    qs = queries(sentence, what)
    prov = prov if prov is not None else provider()
    os.makedirs(out_dir, exist_ok=True)
    sources, seen, hits = [], set(), []
    note = getattr(prov, "why", "")
    for q in qs:
        try:
            rows = prov.search(q, most=max(1, cap // max(1, len(qs)) + 1))
        except Exception as e:                 # noqa: BLE001 -- recorded, not raised
            hits.append({"query": q, "error": str(e), "results": 0})
            continue
        hits.append({"query": q, "results": len(rows)})
        for r in rows:
            if len(sources) >= cap or r["url"] in seen:
                continue
            seen.add(r["url"])
            got = None
            try:
                got = prov.fetch(r["url"])
            except Exception:                  # noqa: BLE001 -- a source we do not have
                got = None
            if not got:
                continue
            blob = got["content"]
            sid = f"src/{len(sources) + 1}"
            name = f"{len(sources) + 1}-" + re.sub(r"[^a-z0-9]+", "-",
                                                   (r["title"] or "source").lower())[:48]
            path = os.path.join(out_dir, name + ".txt")
            text = _text_of(blob, got.get("media", ""))
            with open(path, "w") as fh:
                fh.write(text[:SOURCE_BYTES])
            sources.append({
                "id": sid, "title": r["title"] or got.get("title") or r["url"],
                "url": r["url"], "accessed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fingerprint": _fingerprint(blob),
                "media": got.get("media", "text/plain"),
                "supports": r.get("snippet", "")[:400],
                "provider": prov.name, "bytes": len(blob),
                "path": os.path.relpath(path, out_dir)})
    return {"classification": what, "queries": qs, "searched": hits,
            "sources": sources, "provider": prov.name, "note": note,
            "dir": out_dir}


def reading_of(sentence: str, gathering: dict, claims=None, *,
               inferred_note: str = "") -> dict:
    """The reading record for a gathering, with whatever claims have been made of it.

        With no claims and no sources this is the honest empty reading: classification,
        the queries it would have asked, and the reason it asked nobody.
        
    """
    return contracts.make(
        "reading", sentence=str(sentence),
        classification=gathering["classification"]["kind"],
        sources=list(gathering.get("sources") or []),
        claims=list(claims or []),
        uncertainty=([] if gathering.get("sources") else
                     [{"about": "everything", "why": gathering.get("note")
                       or "no source was retrieved for this request"}]),
        inferred=([] if gathering.get("sources") else
                  [gathering["classification"]["why"]]),
        queries=list(gathering.get("queries") or []),
        provider=gathering.get("provider") or "none",
        note=inferred_note or gathering.get("note") or "")


def needs_evidence(sentence: str) -> bool:
    """Would this request be better for going and looking something up?

        True for a named place and for a named tradition; false for a description, which
        carries its own facts. A run that answers True and retrieves nothing is not wrong --
        it is a run whose reading is inferred, and it says so.
        
    """
    return classify(sentence)["kind"] in ("named_reference", "tradition")


def brief(reading: dict, most: int = 4) -> str:
    """The evidence, as it goes into a brief: what was found, and where it came from."""
    if not reading:
        return "No reading was made for this request."
    rows = [f"**What this request was read as:** {reading['classification']}."]
    if reading.get("sources"):
        rows.append(f"\n**Sources retrieved** (provider `{reading['provider']}`):\n")
        for s in reading["sources"][:most]:
            rows.append(f"- `{s['id']}` [{s['title']}]({s['url']}) -- accessed "
                        f"{s['accessed']}, {s['bytes']} bytes, {s['fingerprint']}")
    else:
        rows.append(f"\n**No source was retrieved.** {reading.get('note') or ''} "
                    f"Everything below is inferred, and a reading with no source is a "
                    f"weaker thing than one with sources: say what you are unsure of.")
    if reading.get("claims"):
        rows.append("\n**Claims:**\n")
        for c in reading["claims"]:
            rows.append(f"- {c.get('says')} "
                        + (f"(`{c.get('source')}`)" if c.get("source")
                           else "(inferred)"))
    if reading.get("uncertainty"):
        # **An uncertainty is a sentence or a record, and this assumed a record.** The
        # contract allows either -- `reading.uncertainty` is a plain list -- and the
        # brief the claims job is given asks for prose, so an agent that answered the
        # way it was asked crashed the spec brief that was about to quote it back.
        rows.append("\n**Uncertain:** "
                    + "; ".join(str(u.get("why") or u) if isinstance(u, dict) else str(u)
                                for u in reading["uncertainty"]))
    return "\n".join(rows)
