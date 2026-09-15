"""The file protocol, the API tool loop, the route table and both wire formats -- with
no real model call and no key anywhere."""
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ethoslm import model
from ethoslm.model import (ApiBuilder, SubagentBuilder, AnthropicTransport, OpenAITransport,
                         Router, load_models, to_openai, from_openai, parse_json_answer)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")


def use(tool_name, **args):
    return {"type": "tool_use", "id": "call_" + tool_name, "name": tool_name, "input": args}


class AdapterTests(unittest.TestCase):
    def job(self, directory):
        d = Path(directory)
        (d / "brief.md").write_text("The exact original brief.\n")
        (d / "check.py").write_text(
            'from pathlib import Path\n'
            'compile(Path("program.py").read_text(), "program.py", "exec")\n'
            'Path("findings.md").write_text("checked the program")\n')
        return {"dir": str(d), "brief": str(d / "brief.md"),
                "write": str(d / "program.py"), "done": str(d / "done"), "images": []}

    def transport(self, *turns):
        self.calls = []
        turns = iter(turns)
        def send(payload):
            self.calls.append(copy.deepcopy(payload))
            blocks = next(turns)
            return {"content": blocks, "stop_reason": "tool_use",
                    "usage": {"input_tokens": 3, "output_tokens": 2}}
        return send

    def test_subagent_is_the_unchanged_seam(self):
        job = {"dir": "unused", "write": "unused/program.py"}
        self.assertIs(SubagentBuilder().submit(job), job)

    def test_chunks_check_and_done_with_exact_brief(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            send = self.transport([use("write_program", text="a = 1\n", append=False)],
                                  [use("write_program", text="b = 2\n", append=True)],
                                  [use("check")], [use("finish")])
            result = ApiBuilder(send, model="stub").submit(job)
            self.assertEqual(Path(job["write"]).read_text(), "a = 1\nb = 2\n")
            self.assertTrue(Path(job["done"]).exists())
            self.assertEqual(self.calls[0]["messages"][0]["content"][0]["text"],
                             Path(job["brief"]).read_text())
            self.assertEqual(result["api_usage"], {"input_tokens": 12, "output_tokens": 8})
            self.assertEqual(self.calls[-1]["messages"][-1]["content"][0]["tool_use_id"],
                             "call_check")
            self.assertEqual(self.calls[-1]["messages"][-1]["content"][0]["content"],
                             "checked the program")

    def test_edit_invalidates_check(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            send = self.transport([use("write_program", text="a=1\n", append=False)],
                [use("check")], [use("write_program", text="b=2\n", append=True)],
                [use("finish")], [])
            with self.assertRaisesRegex(RuntimeError, "without checking"):
                ApiBuilder(send, model="stub").submit(job)
            self.assertFalse(Path(job["done"]).exists())
            self.assertTrue(self.calls[-1]["messages"][-1]["content"][0]["is_error"])

    def test_outside_read_and_unchecked_finish_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            send = self.transport([use("read_file", name="../secret")], [use("finish")], [])
            with self.assertRaises(RuntimeError):
                ApiBuilder(send, model="stub").submit(job)
            self.assertFalse(Path(job["done"]).exists())
            self.assertTrue(self.calls[1]["messages"][-1]["content"][0]["is_error"])

    def test_truncated_tool_is_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            send = lambda p: {"content": [use("write_program", text="partial", append=False)],
                              "stop_reason": "max_tokens"}
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                ApiBuilder(send, model="stub").submit(job)
            self.assertFalse(Path(job["write"]).exists())
            self.assertFalse(Path(job["done"]).exists())

    def test_done_job_does_not_call_transport(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            Path(job["done"]).touch()
            def forbidden(payload):
                self.fail("answered job called the transport")
            self.assertEqual(ApiBuilder(forbidden, model="stub").submit(job), job)

    def test_crashed_checker_cannot_reuse_stale_findings(self):
        with tempfile.TemporaryDirectory() as d:
            job = self.job(d)
            Path(d, "findings.md").write_text("old findings")
            Path(d, "check.py").write_text("raise RuntimeError('checker crashed')\n")
            send = self.transport([use("write_program", text="a=1\n", append=False)],
                                  [use("check")], [use("finish")], [])
            with self.assertRaisesRegex(RuntimeError, "without checking"):
                ApiBuilder(send, model="stub").submit(job)
            self.assertFalse(Path(job["done"]).exists())


# --- the route table ------------------------------------------------------------------

DOC = load_models(str(Path(__file__).resolve().parents[1] / "models.json"))


def routes(env, doc=DOC):
    r = Router(doc=doc, env=env)
    return {k: (v.name if v else None) for k, v in r.routes.items()}


class RouteTests(unittest.TestCase):
    def test_nothing_configured_is_the_subagent_everywhere(self):
        self.assertEqual(routes({}), {r: None for r in model.ROLES})
        self.assertIsInstance(Router(doc=DOC, env={}).builder("type"), SubagentBuilder)

    def test_one_provider_routes_every_role_through_its_tiers(self):
        got = routes({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"})
        t = DOC["tiers"]["anthropic"]
        # One call, and it is asked for a named place's invariants before the schema
        self.assertEqual(got, {"spec": "anthropic/" + t["frontier"],
                               "plan": "anthropic/" + t["frontier"],
                               "type": "anthropic/" + t["frontier"],
                               "build": "anthropic/" + t["frontier"],
                               "revise": "anthropic/" + t["small"],
                               "judge": "anthropic/" + t["vision"]})

    def test_a_role_may_go_to_another_provider_or_stay_staged(self):
        env = {"ETHOSLM_MODEL_API": "openai", "OPENAI_API_KEY": "k",
               "ETHOSLM_MODEL_JUDGE": "anthropic/claude-sonnet-5",
               "ETHOSLM_MODEL_SPEC": "openrouter/google/gemini-2.5-flash",
               "ETHOSLM_MODEL_REVISE": "subagent"}
        got = routes(env)
        self.assertEqual(got["judge"], "anthropic/claude-sonnet-5")
        # an OpenRouter id carries a slash of its own: split on the first only
        self.assertEqual(got["spec"], "openrouter/google/gemini-2.5-flash")
        self.assertIsNone(got["revise"])
        self.assertEqual(got["build"], "openai/" + DOC["tiers"]["openai"]["frontier"])

    def test_a_role_routes_without_the_global_switch(self):
        got = routes({"ETHOSLM_MODEL_BUILD": "local/qwen2.5-coder:32b"})
        self.assertEqual(got["build"], "local/qwen2.5-coder:32b")
        self.assertIsNone(got["spec"])

    def test_the_round_18_knob_still_means_one_model_for_every_role(self):
        got = routes({"ETHOSLM_MODEL_API": "anthropic", "ETHOSLM_MODEL_NAME": "claude-x",
                      "ANTHROPIC_API_KEY": "k"})
        self.assertEqual(set(got.values()), {"anthropic/claude-x"})

    def test_a_missing_key_is_named_and_refused_at_the_call(self):
        r = Router(doc=DOC, env={"ETHOSLM_MODEL_API": "anthropic"})
        self.assertEqual(r.route("build").ready, "ANTHROPIC_API_KEY is not set")
        with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
            r.builder("build")
        self.assertIn("ANTHROPIC_API_KEY", [row["note"] for row in r.table()][0])

    def test_a_local_endpoint_needs_no_key_and_reads_its_url_from_the_environment(self):
        r = Router(doc=DOC, env={"ETHOSLM_MODEL_API": "local",
                                 "ETHOSLM_LOCAL_BASE_URL": "http://box:8080/v1"})
        rt = r.route("build")
        self.assertIsNone(rt.ready)
        self.assertEqual(rt.base_url, "http://box:8080/v1")
        self.assertIsInstance(rt.transport(), OpenAITransport)

    def test_unknown_provider_and_unknown_tier_are_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "unknown ETHOSLM_MODEL_API 'nope'"):
            Router(doc=DOC, env={"ETHOSLM_MODEL_API": "nope"})
        doc = copy.deepcopy(DOC)
        del doc["tiers"]["openai"]["vision"]
        r = Router(doc=doc, env={"ETHOSLM_MODEL_API": "openai", "OPENAI_API_KEY": "k"})
        with self.assertRaisesRegex(ValueError, "no tier 'vision' for role 'judge'"):
            r.route("judge")
        self.assertEqual(r.route("build").name, "openai/" + doc["tiers"]["openai"]["frontier"])

    def test_a_model_without_vision_may_not_judge_or_see_a_brief_image(self):
        doc = copy.deepcopy(DOC)
        doc["tiers"]["local"]["vision"] = {"model": "textonly", "vision": False}
        doc["tiers"]["local"]["frontier"] = {"model": "textonly", "vision": False}
        r = Router(doc=doc, env={"ETHOSLM_MODEL_API": "local"})
        with self.assertRaisesRegex(ValueError, "needs vision"):
            r.route("judge")
        self.assertIsInstance(r.builder("build", {"images": []}), ApiBuilder)
        with self.assertRaisesRegex(RuntimeError, "without vision"):
            r.builder("build", {"images": ["x.png"]})

    def test_the_file_is_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "models.json")
            p.write_text(json.dumps({"providers": {"x": {"api": "grpc", "base_url": "u"}}}))
            with self.assertRaisesRegex(ValueError, "api 'grpc'"):
                load_models(str(p))
            p.write_text(json.dumps({"providers": {"x": {"api": "openai", "base_url": "u"}},
                                     "roles": {"painter": "x/y"}}))
            with self.assertRaisesRegex(ValueError, "'painter' is not a role"):
                load_models(str(p))


# --- the two wire formats -------------------------------------------------------------

def fake_http(*answers):
    """An `http(url, body, headers, timeout)` that records every request and answers in
    order; the last answer repeats."""
    log = []
    answers = list(answers)

    def http(url, body, headers, timeout):
        log.append({"url": url, "body": copy.deepcopy(body), "headers": dict(headers)})
        return copy.deepcopy(answers.pop(0) if len(answers) > 1 else answers[0])
    http.log = log
    return http


NEUTRAL = {"model": "m", "max_tokens": 99, "tools": model.TOOLS, "system": "sys",
           "messages": [
               {"role": "user", "content": [
                   {"type": "text", "text": "brief"},
                   {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                "data": "AAAA"}}]},
               {"role": "assistant", "content": [
                   {"type": "text", "text": "writing"},
                   {"type": "tool_use", "id": "c1", "name": "write_program",
                    "input": {"text": "a = 1\n", "append": False}}]},
               {"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "c1",
                    "content": "program.py contains 6 bytes", "is_error": False}]}]}


class WireTests(unittest.TestCase):
    def test_anthropic_passes_the_neutral_shape_through_with_its_headers(self):
        http = fake_http({"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                          "usage": {"input_tokens": 5, "output_tokens": 1}})
        t = AnthropicTransport("https://api.anthropic.com/v1/", "sk", http=http,
                               extra={"output_config": {"effort": "high"}})
        ans = t(NEUTRAL)
        req = http.log[0]
        self.assertEqual(req["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "sk")
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(req["body"]["messages"], NEUTRAL["messages"])
        self.assertEqual(req["body"]["tools"], model.TOOLS)
        self.assertEqual(req["body"]["system"], "sys")
        self.assertEqual(req["body"]["output_config"], {"effort": "high"})
        self.assertNotIn("thinking", req["body"])
        self.assertEqual(ans, {"content": [{"type": "text", "text": "hi"}],
                               "stop_reason": "end_turn", "stop_details": None,
                               "usage": {"input_tokens": 5, "output_tokens": 1}})

    def test_openai_request_is_chat_completions_byte_for_byte(self):
        body = to_openai(NEUTRAL)
        self.assertEqual(body, {
            "model": "m", "max_tokens": 99,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [
                    {"type": "text", "text": "brief"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
                {"role": "assistant", "content": "writing", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "write_program",
                                  "arguments": json.dumps({"text": "a = 1\n", "append": False})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "program.py contains 6 bytes"}],
            "tools": [{"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["input_schema"]}} for t in model.TOOLS]})

    def test_openai_answer_becomes_the_neutral_shape(self):
        ans = from_openai({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "call_9", "type": "function", "function": {
                    "name": "check", "arguments": "{}"}},
                {"id": "call_10", "type": "function", "function": {
                    "name": "write_program", "arguments": "not json"}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        self.assertEqual(ans["stop_reason"], "tool_use")
        self.assertEqual(ans["usage"], {"input_tokens": 7, "output_tokens": 3})
        self.assertEqual(ans["content"][0], {"type": "tool_use", "id": "call_9",
                                             "name": "check", "input": {}})
        self.assertEqual(ans["content"][1]["input"], {"_arguments": "not json"})
        self.assertEqual(from_openai({"choices": [{"finish_reason": "length", "message": {
            "content": "x"}}]})["stop_reason"], "max_tokens")
        self.assertEqual(from_openai({"choices": [{"finish_reason": "content_filter",
                                                   "message": {}}]})["stop_reason"], "refusal")

    def test_openai_transport_sends_a_bearer_and_retries_the_retired_token_field(self):
        ok = {"choices": [{"finish_reason": "stop", "message": {"content": "ready"}}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        calls = []

        def http(url, body, headers, timeout):
            calls.append(copy.deepcopy(body))
            if "max_tokens" in body:
                raise RuntimeError(url + " answered 400: Unsupported parameter: "
                                   "'max_tokens' is not supported with this model. "
                                   "Use 'max_completion_tokens' instead.")
            return ok
        t = OpenAITransport("https://api.openai.com/v1", "sk", http=http)
        self.assertEqual(t(NEUTRAL)["content"], [{"type": "text", "text": "ready"}])
        self.assertEqual([("max_tokens" in c, c.get("max_completion_tokens")) for c in calls],
                         [(True, None), (False, 99)])
        t(NEUTRAL)                                  # remembered for the session
        self.assertEqual(len(calls), 3)
        self.assertNotIn("max_tokens", calls[2])
        # a local server that ignores the key still gets the header
        http2 = fake_http(ok)
        OpenAITransport("http://localhost:11434/v1", None, http=http2)(NEUTRAL)
        self.assertEqual(http2.log[0]["headers"]["authorization"], "Bearer none")
        self.assertEqual(http2.log[0]["url"], "http://localhost:11434/v1/chat/completions")

    def test_the_loop_runs_unchanged_over_the_openai_wire(self):
        """The same brief, chunks, check and finish as the stub test above, with the
        transport speaking Chat Completions to a fake server: byte-identical program."""
        def turn(*calls):
            return {"choices": [{"finish_reason": "tool_calls", "message": {
                "content": "", "tool_calls": [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": n, "arguments": json.dumps(a)}}
                    for i, (n, a) in enumerate(calls)]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        http = fake_http(turn(("write_program", {"text": "a = 1\n", "append": False})),
                         turn(("write_program", {"text": "b = 2\n", "append": True})),
                         turn(("check", {})), turn(("finish", {})))
        with tempfile.TemporaryDirectory() as d:
            job = AdapterTests.job(self, d)
            t = OpenAITransport("http://localhost:11434/v1", None, http=http)
            result = ApiBuilder(t, model="stub").submit(job)
            self.assertEqual(Path(job["write"]).read_text(), "a = 1\nb = 2\n")
            self.assertTrue(Path(job["done"]).exists())
            self.assertEqual(result["api_usage"], {"input_tokens": 12, "output_tokens": 8})
            last = http.log[-1]["body"]["messages"]
            self.assertEqual(last[0]["content"][0]["text"], Path(job["brief"]).read_text())
            self.assertEqual(last[-1], {"role": "tool", "tool_call_id": "call_0",
                                        "content": "checked the program"})
            self.assertEqual(last[-2]["tool_calls"][0]["function"]["name"], "check")

    def test_a_refusal_stops_the_loop_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            job = AdapterTests.job(self, d)
            send = lambda p: {"content": [], "stop_reason": "refusal",
                              "stop_details": {"type": "refusal", "category": "x"}}
            with self.assertRaisesRegex(RuntimeError, "refused"):
                ApiBuilder(send, model="stub").submit(job)
            self.assertFalse(Path(job["done"]).exists())


# --- the text roles and the judge ---------------------------------------------------

def stub_router(env, *answers, doc=DOC):
    """A router whose every transport answers the given texts in order."""
    r = Router(doc=doc, env=env, http=fake_http(*[
        {"content": [{"type": "text", "text": a}], "stop_reason": "end_turn",
         "usage": {"input_tokens": 10, "output_tokens": 4}} for a in answers]))
    return r


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ETHOSLM_MEASURE_LOG"] = os.path.join(self.tmp.name, "m.jsonl")
        import importlib
        from ethoslm import measure
        importlib.reload(measure)

    def tearDown(self):
        del os.environ["ETHOSLM_MEASURE_LOG"]
        import importlib
        from ethoslm import measure
        importlib.reload(measure)
        self.tmp.cleanup()

    def test_a_spec_request_is_answered_as_json_and_written(self):
        r = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"},
                        'Here it is:\n```json\n{"kind": "town", "n": 1}\n```\nDone.')
        req, out = Path(self.tmp.name, "place_spec_prompt.md"), Path(self.tmp.name, "place.json")
        req.write_text("Read a sentence.\n\nWrite a single JSON file to place.json.\n")
        got = r.answer("spec", str(req), str(out))
        self.assertEqual(json.loads(out.read_text()), {"kind": "town", "n": 1})
        self.assertEqual(got["model"],
                         "anthropic/" + DOC["tiers"]["anthropic"][DOC["roles"]["spec"]])
        sent = r.http.log[0]["body"]["messages"][0]["content"][0]["text"]
        self.assertTrue(sent.startswith("Read a sentence."))
        self.assertIn("cannot write files", sent)
        self.assertIn("`place.json`", sent)
        rows = [json.loads(l) for l in open(os.environ["ETHOSLM_MEASURE_LOG"])]
        self.assertEqual(rows[-1]["kind"], "model_call")
        self.assertEqual((rows[-1]["stage"], rows[-1]["tokens_in"], rows[-1]["tokens_out"]),
                         ("spec", 10, 4))

    def test_a_reply_that_is_not_json_is_sent_back_once(self):
        r = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"},
                        "I would rather explain.", '{"ok": true}')
        req, out = Path(self.tmp.name, "b.md"), Path(self.tmp.name, "plan.json")
        req.write_text("plan it")
        r.answer("plan", str(req), str(out))
        self.assertEqual(json.loads(out.read_text()), {"ok": True})
        second = r.http.log[1]["body"]["messages"]
        self.assertEqual(len(second), 3)
        self.assertIn("not a JSON document", second[-1]["content"][0]["text"])
        r2 = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"},
                         "no", "still no")
        with self.assertRaisesRegex(RuntimeError, "did not answer the plan request with JSON"):
            r2.answer("plan", str(req), str(Path(self.tmp.name, "p2.json")))

    def test_parse_json_answer_takes_bare_fenced_or_surrounded(self):
        self.assertEqual(parse_json_answer('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_answer('```\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_answer('Sure. {"a": {"b": 2}} That is all.'), {"a": {"b": 2}})
        with self.assertRaises(ValueError):
            parse_json_answer("nothing here")

    def test_the_judge_sees_both_images_and_the_prompt(self):
        a, b = Path(self.tmp.name, "a.png"), Path(self.tmp.name, "b.png")
        a.write_bytes(PNG); b.write_bytes(PNG + b"\x00")
        r = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"},
                        "B\nthe second has a roof")
        ask = r.ask("judge")
        self.assertEqual(ask(str(a), str(b), "Which is better made?"), "B\nthe second has a roof")
        content = r.http.log[0]["body"]["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["text", "image", "text", "image", "text"])
        self.assertEqual(content[1]["source"]["data"], base64.b64encode(PNG).decode())
        self.assertEqual(content[-1]["text"], "Which is better made?")
        self.assertEqual(r.http.log[0]["body"]["model"], DOC["tiers"]["anthropic"]["vision"])
        self.assertIsNone(Router(doc=DOC, env={}).ask("judge"))

    def test_fulfil_answers_routed_records_and_counts_an_api_builder_once(self):
        r = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"},
                        '{"kind": "hamlet"}')
        req, out = Path(self.tmp.name, "s.md"), Path(self.tmp.name, "place.json")
        req.write_text("x")
        flat = {"status": "needs_model", "role": "spec", "request": str(req), "write": str(out)}
        self.assertEqual(r.fulfil(flat), 1)
        self.assertTrue(out.exists())
        # a builder job the API just ran is counted; one merely adopted is not
        ran = {"w1": {"status": "needs_model", "blinded": {"done": "d", "api_usage": {"input_tokens": 1}}}}
        self.assertEqual(r.fulfil(ran), 1)
        self.assertEqual(r.fulfil({"w1": {"status": "needs_model", "blinded": {"done": "d"}}}), 0)
        # nested under `candidates`, as the revise stage reports
        self.assertEqual(r.fulfil({"candidates": {"c0": ran["w1"]}}), 1)
        # nothing routed: nothing answered, nothing sent
        self.assertEqual(Router(doc=DOC, env={}).fulfil(flat), 0)
        self.assertEqual(len(r.http.log), 1)

    def test_fulfil_answers_staged_judgements_into_the_cache(self):
        from ethoslm import judge
        a, b = Path(self.tmp.name, "a.png"), Path(self.tmp.name, "b.png")
        a.write_bytes(PNG); b.write_bytes(PNG + b"\x00")
        cache = os.path.join(self.tmp.name, "judge_cache.jsonl")
        req = judge.stage(judge.request(str(a), str(b), "which?", cache))
        staged = Path(self.tmp.name, "q_requests.json")
        staged.write_text(json.dumps([req]))
        r = stub_router({"ETHOSLM_MODEL_API": "anthropic", "ANTHROPIC_API_KEY": "k"}, "A")
        rec = {"q": {"status": "needs_model", "role": "judge", "staged": str(staged),
                     "stage_name": "q", "cache": cache}}
        self.assertEqual(r.fulfil(rec), 1)
        self.assertEqual(judge.load_cache(cache)[req["key"]]["verdict"], "A")
        self.assertEqual(r.fulfil(rec), 0)          # warm now: nothing to ask


if __name__ == "__main__":
    unittest.main()
