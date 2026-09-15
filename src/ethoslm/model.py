"""The model seam: roles, providers, transports, and the one place that decides which
model does which job.

Every model call this project makes is a *staged request*: a stage writes a brief, names
the file it wants back, reports `needs_model`, and the driver waits for the file. The
supervising agent answers by writing files -- that is `SubagentBuilder`, and it stays the
default whenever no API is configured. This module adds the other way of answering the
same request: a routed API call, per role, on a provider of the user's choosing.

Six roles. `spec` reads the sentence into a place spec (and authors the voice inside the
same call); `plan` plans a level (place, compound, district, or the flat planner); `type`
authors a type through the tool loop with its checker; `build` authors a part or a wave
the same way; `revise` is the findings revision, the same loop with a second document;
`judge` is the pairwise vision judge. The map from role to model is `models.json` (or
`ETHOSLM_MODELS`), and every role can be overridden from the environment.

Two wire formats and no dependency. `anthropic` is the Messages API; `openai` is Chat
Completions, which is what every other hosted provider and every local server exposes.
The loop and the roles speak one neutral shape -- Anthropic's content blocks, because
that is what the loop was written and stub-tested against -- and a transport converts.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Protocol
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Every model role in the pipeline, and the tier each one defaults to. A tier is a name
#: a provider entry in `models.json` resolves to a model id, so switching provider is
#: one variable and the roles keep their relative weight.
ROLES = ("spec", "plan", "type", "build", "revise", "judge")

TIER_OF = {"spec": "small", "plan": "frontier", "type": "frontier", "build": "frontier",
           "revise": "small", "judge": "vision"}

#: Roles whose request always carries images. `build` and `revise` sometimes do, and are
#: checked per job.
IMAGE_ROLES = ("judge",)

#: Output ceilings per role. A program is written in chunks and a plan level is one JSON
#: document, so nothing here needs the long tail.
MAX_TOKENS = {"spec": 8192, "plan": 16384, "type": 8192, "build": 8192,
              "revise": 8192, "judge": 512}

APIS = ("anthropic", "openai")

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif"}


class Builder(Protocol):
    """Consume a staged job, preserving its response paths."""

    def submit(self, job: dict) -> dict: ...


class SubagentBuilder:
    """Leave the file-based job for the supervising agent to fulfil."""

    def submit(self, job: dict) -> dict:
        return job


TOOLS = [
    {"name": "write_program", "description": "Write program.py in chunks. Use append=false for the first chunk or a replacement, append=true for subsequent chunks.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"}, "append": {"type": "boolean"}},
         "required": ["text", "append"], "additionalProperties": False}},
    {"name": "read_file", "description": "Read the current program.py or findings.md.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "enum": ["program.py", "findings.md"]}},
         "required": ["name"], "additionalProperties": False}},
    {"name": "check", "description": "Run check.py on the current program and return findings. Run again after each final edit, before finish.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "finish", "description": "Write done after checking the final program.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


# --- the wire ---------------------------------------------------------------------

def http_json(url: str, body: dict, headers: dict, timeout: int = 600) -> dict:
    """One JSON POST. An HTTP error carries the status and the body, which is where a
    provider says what it did not like about the request."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"{url} answered {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} unreachable: {e.reason}") from None


class AnthropicTransport:
    """The Messages API. The neutral shape *is* this shape, so the payload passes
    through: `model`, `max_tokens`, `messages`, `tools`, `system`."""

    api = "anthropic"

    def __init__(self, base_url: str, key: str | None, http=None, extra: dict | None = None,
                 timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.http = http or http_json
        self.extra = dict(extra or {})
        self.timeout = timeout

    def __call__(self, payload: dict) -> dict:
        if not self.key:
            raise RuntimeError("no API key for the anthropic provider")
        body = {k: v for k, v in payload.items()
                if k in ("model", "max_tokens", "messages", "tools", "system")}
        body.update(self.extra)
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        answer = self.http(self.base_url + "/messages", body, headers, self.timeout)
        usage = answer.get("usage") or {}
        return {"content": answer.get("content") or [],
                "stop_reason": answer.get("stop_reason"),
                "stop_details": answer.get("stop_details"),
                "usage": {"input_tokens": int(usage.get("input_tokens") or 0),
                          "output_tokens": int(usage.get("output_tokens") or 0)}}


def _data_url(block: dict) -> str:
    src = block["source"]
    return f"data:{src['media_type']};base64,{src['data']}"


def to_openai(payload: dict) -> dict:
    """The neutral request as a Chat Completions body."""
    messages = []
    if payload.get("system"):
        messages.append({"role": "system", "content": payload["system"]})
    for m in payload["messages"]:
        content = m["content"]
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"],
                                   "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            row = {"role": "assistant", "content": text}
            if calls:
                row["tool_calls"] = calls
            messages.append(row)
            continue
        # A user turn: tool results go back as `tool` messages, in order, straight after
        # the assistant turn that asked; everything else is one user message.
        parts = []
        for b in content:
            t = b.get("type")
            if t == "tool_result":
                c = b.get("content")
                messages.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                 "content": c if isinstance(c, str) else json.dumps(c)})
            elif t == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif t == "image":
                parts.append({"type": "image_url", "image_url": {"url": _data_url(b)}})
        if parts:
            messages.append({"role": "user", "content": parts})
    body = {"model": payload["model"], "messages": messages,
            "max_tokens": payload["max_tokens"]}
    if payload.get("tools"):
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"], "description": t["description"],
                                       "parameters": t["input_schema"]}}
                         for t in payload["tools"]]
    return body


FINISH = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
          "content_filter": "refusal", "function_call": "tool_use"}


def from_openai(answer: dict) -> dict:
    """A Chat Completions answer as the neutral shape."""
    choice = (answer.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks = []
    text = msg.get("content")
    if isinstance(text, list):                   # some servers answer in parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        blocks.append({"type": "text", "text": text})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except ValueError:
            args = {"_arguments": raw}         # the loop reports what is missing
        blocks.append({"type": "tool_use", "id": call.get("id") or f"call_{len(blocks)}",
                       "name": fn.get("name"), "input": args if isinstance(args, dict) else {}})
    usage = answer.get("usage") or {}
    reason = choice.get("finish_reason")
    return {"content": blocks,
            "stop_reason": FINISH.get(reason, "tool_use" if any(
                b["type"] == "tool_use" for b in blocks) else "end_turn"),
            "stop_details": None,
            "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                      "output_tokens": int(usage.get("completion_tokens") or 0)}}


class OpenAITransport:
    """Chat Completions at any base URL: OpenAI, OpenRouter, Groq, Together, DeepSeek,
    Mistral, xAI, Gemini's compatibility endpoint, Ollama, llama.cpp, vLLM, LM Studio,
    a LiteLLM proxy. The key is sent as a bearer token; a local server that ignores it
    still gets one, because some require the header to be present."""

    api = "openai"

    def __init__(self, base_url: str, key: str | None, http=None, extra: dict | None = None,
                 timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.http = http or http_json
        self.extra = dict(extra or {})
        self.timeout = timeout
        self._tokens_field = "max_tokens"

    def __call__(self, payload: dict) -> dict:
        body = to_openai(payload)
        body.update(self.extra)
        if self._tokens_field != "max_tokens":
            body[self._tokens_field] = body.pop("max_tokens")
        headers = {"authorization": f"Bearer {self.key or 'none'}"}
        url = self.base_url + "/chat/completions"
        try:
            answer = self.http(url, body, headers, self.timeout)
        except RuntimeError as e:
            # OpenAI's own endpoint retired `max_tokens` for its newer models and says
            # so in the 400; most compatible servers know only `max_tokens`. Asked once,
            # remembered for the rest of the session.
            if "max_completion_tokens" in str(e) and "max_tokens" in body:
                self._tokens_field = "max_completion_tokens"
                body["max_completion_tokens"] = body.pop("max_tokens")
                answer = self.http(url, body, headers, self.timeout)
            else:
                raise
        return from_openai(answer)


# --- the route table --------------------------------------------------------------

def models_path(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("ETHOSLM_MODELS") or os.path.join(ROOT, "models.json")


def load_models(path: str | None = None, env=None) -> dict:
    """`models.json`, checked. A provider names its api kind, base URL and key variable;
    a tier is a model id or `{"model": id, "vision": false}`."""
    path = path or models_path(env)
    with open(path) as fh:
        doc = json.load(fh)
    providers = doc.get("providers") or {}
    if not providers:
        raise ValueError(f"{path} names no providers")
    for name, p in providers.items():
        if p.get("api") not in APIS:
            raise ValueError(f"{path}: provider {name!r} has api {p.get('api')!r}; "
                             f"one of {APIS}")
        if not p.get("base_url"):
            raise ValueError(f"{path}: provider {name!r} has no base_url")
    doc.setdefault("tiers", {})
    doc.setdefault("roles", {})
    for role, entry in doc["roles"].items():
        if role not in ROLES:
            raise ValueError(f"{path}: {role!r} is not a role; roles are {ROLES}")
        if not isinstance(entry, str):
            raise ValueError(f"{path}: roles.{role} must be a tier, provider/model or "
                             f"subagent")
    return doc


class Route:
    """One role's answer: which provider, which model, over which transport."""

    def __init__(self, role: str, provider: str, entry: dict, model, env, http=None):
        self.role = role
        self.provider = provider
        self.api = entry["api"]
        base = entry["base_url"]
        if entry.get("base_url_env") and env.get(entry["base_url_env"]):
            base = env[entry["base_url_env"]]
        self.base_url = base
        self.key_env = entry.get("key")
        self.key = env.get(self.key_env) if self.key_env else None
        if isinstance(model, dict):
            self.model = model["model"]
            self.vision = bool(model.get("vision", True))
        else:
            self.model = str(model)
            self.vision = bool(entry.get("vision", True))
        self.extra = dict(entry.get("extra") or {})
        limits = entry.get("max_tokens")
        self.max_tokens = int(limits.get(role, MAX_TOKENS[role])
                              if isinstance(limits, dict) else MAX_TOKENS[role])
        self.timeout = int(entry.get("timeout", 600))
        self._http = http

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def ready(self) -> str | None:
        """None when the route can be called; otherwise why not."""
        if self.key_env and not self.key:
            return f"{self.key_env} is not set"
        return None

    def transport(self) -> Callable[[dict], dict]:
        why = self.ready
        if why:
            raise RuntimeError(f"role {self.role!r} is routed to {self.name} but {why}")
        cls = AnthropicTransport if self.api == "anthropic" else OpenAITransport
        return cls(self.base_url, self.key, http=self._http, extra=self.extra,
                   timeout=self.timeout)


def _split(spec: str) -> tuple:
    """`provider/model`, split on the first slash: OpenRouter model ids carry one."""
    if "/" not in spec:
        raise ValueError(f"{spec!r} is not provider/model")
    p, m = spec.split("/", 1)
    return p, m


class Router:
    """The route table, resolved from `models.json` and the environment once.

        `ETHOSLM_MODEL_API` unset means every role is the subagent's -- today's path. Set to a
        provider name it routes every role to that provider through the file's tiers.
        `ETHOSLM_MODEL_<ROLE>` (`ETHOSLM_MODEL_BUILD=openai/gpt-5`, `ETHOSLM_MODEL_JUDGE=subagent`)
        overrides one role whatever the switch says, which is how a per-role provider is
        done. `ETHOSLM_MODEL_NAME` with `ETHOSLM_MODEL_API` is the older knob and still means
        "that model for every role".
        
    """

    def __init__(self, doc: dict | None = None, env=None, http=None):
        self.env = dict(os.environ if env is None else env)
        self.doc = doc if doc is not None else load_models(env=self.env)
        self.http = http
        self.routes: dict = {}
        self.errors: dict = {}
        api = (self.env.get("ETHOSLM_MODEL_API") or "").strip()
        legacy = (self.env.get("ETHOSLM_MODEL_NAME") or "").strip()
        providers = self.doc["providers"]
        if api and api not in providers:
            raise ValueError(f"unknown ETHOSLM_MODEL_API {api!r}; models.json names "
                             f"{sorted(providers)}")
        for role in ROLES:
            spec = self.env.get(f"ETHOSLM_MODEL_{role.upper()}")
            try:
                if spec is not None:
                    spec = spec.strip()
                    self.routes[role] = None if spec in ("", "subagent") \
                        else self._route(role, *_split(spec))
                    continue
                if not api:
                    self.routes[role] = None
                    continue
                entry = self.doc["roles"].get(role, TIER_OF[role])
                if entry == "subagent":
                    self.routes[role] = None
                elif "/" in entry:
                    self.routes[role] = self._route(role, *_split(entry))
                elif legacy:
                    self.routes[role] = self._route(role, api, legacy)
                else:
                    tiers = self.doc["tiers"].get(api) or {}
                    if entry not in tiers:
                        raise ValueError(f"models.json: provider {api!r} has no tier "
                                         f"{entry!r} for role {role!r}")
                    self.routes[role] = self._route(role, api, tiers[entry])
            except ValueError as e:
                self.errors[role] = str(e)
                self.routes[role] = None

    def _route(self, role: str, provider: str, model) -> Route:
        entry = self.doc["providers"].get(provider)
        if entry is None:
            raise ValueError(f"role {role!r}: no provider {provider!r} in models.json "
                             f"({sorted(self.doc['providers'])})")
        r = Route(role, provider, entry, model, self.env, http=self.http)
        if role in IMAGE_ROLES and not r.vision:
            raise ValueError(f"role {role!r} needs vision and {r.name} is declared "
                             f"without it")
        return r

    def route(self, role: str) -> Route | None:
        if role not in ROLES:
            raise ValueError(f"{role!r} is not a role; roles are {ROLES}")
        if role in self.errors:
            raise ValueError(self.errors[role])
        return self.routes[role]

    def table(self) -> list:
        rows = []
        for role in ROLES:
            if role in self.errors:
                rows.append({"role": role, "route": "error", "note": self.errors[role]})
                continue
            r = self.routes[role]
            if r is None:
                rows.append({"role": role, "route": "subagent",
                             "note": "staged for the supervising agent"})
            else:
                rows.append({"role": role, "route": r.name, "api": r.api,
                             "base_url": r.base_url, "vision": r.vision,
                             "note": r.ready or "ready"})
        return rows

    # -- the builder roles --------------------------------------------------------

    def builder(self, role: str = "build", job: dict | None = None) -> Builder:
        r = self.route(role)
        if r is None:
            return SubagentBuilder()
        if job and job.get("images") and not r.vision:
            raise RuntimeError(f"role {role!r} carries {len(job['images'])} image(s) and "
                               f"{r.name} is declared without vision")
        return ApiBuilder(r.transport(), model=r.model, max_tokens=r.max_tokens,
                          role=role, provider=r.provider)

    # -- the text roles ------------------------------------------------------------

    REPLY = ("\n\n---\n\nYou are answering through an API and cannot write files. Reply "
             "with the complete contents of the file described above -- `{name}` -- and "
             "nothing else: no fence, no preamble, no commentary after it.")

    def answer(self, role: str, request: str, write: str) -> dict:
        """One text role: the brief in, the file out. A JSON answer is parsed before it
        is written, and a reply that is not JSON is sent back once with the error."""
        r = self.route(role)
        if r is None:
            raise RuntimeError(f"role {role!r} is not routed")
        brief = Path(request).read_text()
        want_json = write.endswith(".json")
        content = [{"type": "text", "text": brief.rstrip("\n")
                    + self.REPLY.format(name=os.path.basename(write))}]
        messages = [{"role": "user", "content": content}]
        send = r.transport()
        usage = {"input_tokens": 0, "output_tokens": 0}
        t0 = time.perf_counter()
        text = ""
        for attempt in range(2):
            ans = send({"model": r.model, "max_tokens": r.max_tokens, "messages": messages})
            for k in usage:
                usage[k] += ans["usage"][k]
            if ans["stop_reason"] == "refusal":
                raise RuntimeError(f"{r.name} refused the {role} request: "
                                   f"{ans.get('stop_details')}")
            if ans["stop_reason"] == "max_tokens":
                raise RuntimeError(f"{r.name} ran out of output tokens ({r.max_tokens}) "
                                   f"on the {role} request")
            text = "".join(b.get("text", "") for b in ans["content"]
                           if b.get("type") == "text")
            if not want_json:
                break
            try:
                doc = parse_json_answer(text)
                text = json.dumps(doc, indent=1)
                break
            except ValueError as e:
                if attempt:
                    raise RuntimeError(f"{r.name} did not answer the {role} request "
                                       f"with JSON: {e}") from None
                messages.append({"role": "assistant", "content": ans["content"]})
                messages.append({"role": "user", "content": [{"type": "text", "text":
                    f"That reply was not a JSON document ({e}). Reply again with only "
                    f"the JSON document."}]})
        seconds = time.perf_counter() - t0
        os.makedirs(os.path.dirname(os.path.abspath(write)), exist_ok=True)
        Path(write).write_text(text if text.endswith("\n") else text + "\n")
        from .measure import model_call
        model_call(role, brief, text, seconds, key=os.path.basename(write),
                   model=r.model, provider=r.provider, tokens_in=usage["input_tokens"],
                   tokens_out=usage["output_tokens"], note="usage from the provider")
        return {"write": write, "model": r.name, "usage": usage,
                "seconds": round(seconds, 2)}

    # -- the judge -----------------------------------------------------------------

    def ask(self, role: str = "judge"):
        """The `ask(scratch_a, scratch_b, prompt) -> text` callable `judge.compare`
        takes, or None when the role is staged."""
        r = self.route(role)
        if r is None:
            return None
        send = r.transport()
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        def ask(a: str, b: str, prompt: str) -> str:
            content = [{"type": "text", "text": "Image A:"}, image_block(a),
                       {"type": "text", "text": "Image B:"}, image_block(b),
                       {"type": "text", "text": prompt}]
            ans = send({"model": r.model, "max_tokens": r.max_tokens,
                        "messages": [{"role": "user", "content": content}]})
            for k in self.last_usage:
                self.last_usage[k] += ans["usage"][k]
            if ans["stop_reason"] == "refusal":
                raise RuntimeError(f"{r.name} refused a judgement: {ans.get('stop_details')}")
            return "".join(b.get("text", "") for b in ans["content"]
                           if b.get("type") == "text")
        return ask

    # -- the driver's hook ---------------------------------------------------------

    def fulfil(self, res: dict) -> int:
        """Answer every routed request a stage result is waiting on. Returns how many
                were answered this call, so the driver re-enters the stage at once instead of
                sleeping on a file that is already there.

                A builder job is answered *inside* the stage (`_blind_build` submits it), so
                what is counted here is the record saying the API ran -- `api_usage` on the
                blinded job -- and never the mere presence of `done`, which would re-count an
                answer the stage has not yet adopted and loop for ever.
                
        """
        n = 0
        for rec in staged(res):
            role = rec.get("role")
            bl = rec.get("blinded") or {}
            if bl.get("api_usage"):
                n += 1
                continue
            if role in ("spec", "plan") and self.route(role) is not None:
                got = self.answer(role, rec["request"], rec["write"])
                print(f"   answered {role} via {got['model']}: {got['write']} "
                      f"({got['usage']['input_tokens']} in, "
                      f"{got['usage']['output_tokens']} out, {got['seconds']} s)",
                      flush=True)
                n += 1
            elif role == "judge" and rec.get("staged") and self.route(role) is not None:
                from . import judge as judge_mod
                ask = self.ask(role)
                cache = judge_mod.load_cache(rec.get("cache") or judge_mod.CACHE)
                with open(rec["staged"]) as fh:
                    requests = json.load(fh)
                k = 0
                for req in requests:
                    if req["key"] in cache:
                        continue
                    t0 = time.perf_counter()
                    text = ask(req["scratch_a"], req["scratch_b"], req["prompt"])
                    judge_mod.fulfil(req, text, time.perf_counter() - t0,
                                     rec.get("stage_name", "judge"),
                                     rec.get("cache") or judge_mod.CACHE)
                    k += 1
                if k:
                    print(f"   answered {k} judgement(s) via {self.route(role).name} "
                          f"({self.last_usage['input_tokens']} in, "
                          f"{self.last_usage['output_tokens']} out)", flush=True)
                n += k
        return n


def staged(res: dict):
    """Every `needs_model` record in a stage result: the result itself (the spec stage
    answers flat), its values (`plan`, a wave, a type), and their values (a revise
    candidate under `candidates`)."""
    if isinstance(res, dict) and res.get("status") == "needs_model":
        yield res
        return
    for v in (res or {}).values():
        if not isinstance(v, dict):
            continue
        if v.get("status") == "needs_model":
            yield v
            continue
        for w in v.values():
            if isinstance(w, dict) and w.get("status") == "needs_model":
                yield w


def image_block(path: str) -> dict:
    image = Path(path)
    media = MEDIA.get(image.suffix.lower())
    if not media:
        raise ValueError(f"unsupported image: {image.name}")
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.b64encode(image.read_bytes()).decode()}}


def parse_json_answer(text: str):
    """The JSON document in a reply: a bare document, one inside a code fence, or one
    with prose round it. The outermost braces win."""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    try:
        return json.loads(s[a:b + 1])
    except ValueError as e:
        raise ValueError(str(e)) from None


_router = None


def router(reload: bool = False) -> Router:
    """The one router. Re-read when asked; tests set the environment and reload."""
    global _router
    if _router is None or reload:
        _router = Router()
    return _router


# --- the builder loop -------------------------------------------------------------

class ApiBuilder:
    """Run the same brief through the client-tool protocol over any transport.

        A transport takes the neutral request and answers in the neutral shape; the two in
        this module speak the Anthropic and the OpenAI wire. Checker results are
        observations, including reported defects; finishing requires a completed check of
        the current bytes, not a claim by the model that it checked.
        
    """

    def __init__(self, transport=None, *, model=None, max_turns=128,
                 max_tokens=8192, check_timeout=1800, role="build", provider=None):
        self.transport = transport or self._request
        self.model = model or os.environ.get("ETHOSLM_MODEL_NAME")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.check_timeout = check_timeout
        self.role = role
        self.provider = provider

    def _request(self, payload):
        """The standing default: Anthropic with the key from the environment."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for ApiBuilder")
        return AnthropicTransport("https://api.anthropic.com/v1", key)(payload)

    def submit(self, job: dict) -> dict:
        if not self.model:
            raise RuntimeError("ETHOSLM_MODEL_NAME is required for ApiBuilder")
        directory = Path(job["dir"])
        program, done = directory / "program.py", directory / "done"
        if done.exists():
            return job
        brief = Path(job["brief"]).read_text()
        content = [{"type": "text", "text": brief}]
        if job.get("findings"):
            content.append({"type": "text", "text": Path(job["findings"]).read_text()})
        for path in job.get("images", []):
            content.append(image_block(path))
        messages = [{"role": "user", "content": content}]
        checked = None
        usage = {"input_tokens": 0, "output_tokens": 0}
        turns = checks = 0
        t0 = time.perf_counter()

        def digest():
            return hashlib.sha256(program.read_bytes()).hexdigest()

        for _ in range(self.max_turns):
            turns += 1
            answer = self.transport({"model": self.model, "max_tokens": self.max_tokens,
                                     "messages": messages, "tools": TOOLS})
            for key in usage:
                usage[key] += int(answer.get("usage", {}).get(key, 0))
            blocks = answer.get("content", [])
            # Never execute a partially generated tool call.
            if answer.get("stop_reason") == "max_tokens":
                raise RuntimeError("model output was truncated; done was not written")
            if answer.get("stop_reason") == "refusal":
                raise RuntimeError(f"the model refused the brief: {answer.get('stop_details')}")
            messages.append({"role": "assistant", "content": blocks})
            results = []
            finished = False
            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                name, args = block.get("name"), block.get("input", {})
                error = False
                try:
                    if finished:
                        raise ValueError("finish must be the final tool call")
                    if name == "write_program":
                        if not isinstance(args.get("text"), str) or not isinstance(args.get("append"), bool):
                            raise ValueError("write_program requires text and a boolean append")
                        with program.open("a" if args["append"] else "w") as stream:
                            stream.write(args["text"])
                        checked = None
                        result = f"program.py contains {program.stat().st_size} bytes"
                    elif name == "read_file":
                        if args.get("name") not in ("program.py", "findings.md"):
                            raise ValueError("only program.py and findings.md may be read")
                        result = (directory / args["name"]).read_text()
                    elif name == "check":
                        before = digest()
                        findings = directory / "findings.md"
                        old_stamp = findings.stat().st_mtime_ns if findings.exists() else None
                        proc = subprocess.run([sys.executable, "check.py"], cwd=directory,
                                              capture_output=True, text=True,
                                              timeout=self.check_timeout)
                        if (proc.returncode not in (0, 1) or not findings.exists()
                                or findings.stat().st_mtime_ns == old_stamp):
                            raise RuntimeError("check.py did not produce findings: " + proc.stderr[-2000:])
                        if before != digest():
                            raise RuntimeError("program.py changed during its check")
                        checked = before
                        checks += 1
                        result = findings.read_text()
                    elif name == "finish":
                        if checked is None or checked != digest():
                            raise ValueError("run check on the current program before finish")
                        finished = True
                        result = "finished"
                    else:
                        raise ValueError(f"unknown tool {name!r}")
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    result, error = str(exc), True
                    if finished:
                        finished = False
                results.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": result, "is_error": error})
            if finished:
                done.touch()
                self._record(job, brief, program, usage, turns, checks,
                             time.perf_counter() - t0)
                return {**job, "api_usage": usage}
            if not results:
                raise RuntimeError("model ended without checking and finishing program.py")
            messages.append({"role": "user", "content": results})
        raise RuntimeError("model tool loop exhausted; done was not written")

    def _record(self, job, brief, program, usage, turns, checks, seconds):
        """The call on the measurement log with the provider's own token counts, keyed
        the way `_record_call` keys its estimate so the estimate is not written over
        it, and `usage.json` beside the program for the round's own reading."""
        if not self.provider:
            return
        sha = hashlib.sha256(program.read_bytes()).hexdigest()
        row = {"model": self.model, "provider": self.provider, "turns": turns,
               "check_runs": checks, "sha256": sha, **usage,
               "seconds": round(seconds, 2)}
        Path(job["dir"], "usage.json").write_text(json.dumps(row, indent=1) + "\n")
        from .measure import model_call
        model_call(self.role, brief, program.read_text(), seconds,
                   key=os.path.basename(job["dir"]), sha256=sha, model=self.model,
                   provider=self.provider, tokens_in=usage["input_tokens"],
                   tokens_out=usage["output_tokens"], turns=turns, check_runs=checks,
                   note="usage from the provider")


def configured_builder(role: str = "build", job: dict | None = None) -> Builder:
    """The builder for one role, off the route table; the default remains the
    file-based seam."""
    return router().builder(role, job)
