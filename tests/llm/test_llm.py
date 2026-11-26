import json
from types import SimpleNamespace

from httpx2 import Client, MockTransport, Request, Response
from openai import APIError, OpenAI
import pytest

import benchmark
import llm


TEMPLATE = (
    (("system", "Instructions {kind}"), ("user", "Query\n{query}\n\nContext\n{context}")),
    {"kind": "Ångström"},
    ("query", "context"),
    "str",
)
STATE = {"query": "Extract {all} values.", "context": "MiXeD 3E-4\n#{markdown}"}
PILOT_CLI = [
    "--method", "vcl", "--dataset", "docred", "--instance", "docred-dev-0244", "--provider", "openai",
]
MOCK_KEY = "synthetic-key-for-offline-tests"


@pytest.fixture(autouse=True)
def forbid_unmocked_openai(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An offline test unexpectedly attempted to construct an OpenAI client")

    monkeypatch.setattr(llm, "OpenAI", unexpected)


@pytest.fixture
def fake_provider(monkeypatch):
    observed = {"requests": []}

    class Provider:
        def __init__(self, **options):
            observed["options"] = options
            self.responses = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def create(self, **request):
            observed["requests"].append(request)
            result = observed["result"]
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(llm, "OpenAI", Provider)
    return observed


def response(text='{"relations":[]}', status="completed", **fields):
    defaults = {
        "output_text": text,
        "output": [],
        "status": status,
        "model": "reported-snapshot",
        "id": "simulated-response-id",
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }
    return SimpleNamespace(**(defaults | fields))


@pytest.mark.parametrize("key_name", ["OPENAI_API_KEY", "OPEN_AI_KEY"])
def test_loads_both_key_aliases_without_changing_dotenv(tmp_path, monkeypatch, key_name):
    for name in ("OPENAI_API_KEY", "OPEN_AI_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    original = f"{key_name}={MOCK_KEY}\nOPENAI_MODEL=from-file\n"
    dotenv.write_bytes(original.encode())
    assert llm.live_settings(dotenv) == ("from-file", MOCK_KEY)
    assert dotenv.read_bytes() == original.encode()
    monkeypatch.setenv("OPENAI_API_KEY", "standard-environment-key")
    monkeypatch.setenv("OPEN_AI_KEY", "alias-environment-key")
    monkeypatch.setenv("OPENAI_MODEL", "from-process")
    assert llm.live_settings(dotenv, "from-cli") == ("from-cli", "standard-environment-key")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert llm.live_settings(dotenv) == ("from-process", "alias-environment-key")


def test_missing_credentials_and_model_errors(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "OPEN_AI_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        llm.live_settings(tmp_path / ".env")
    with pytest.raises(ValueError, match="OPENAI_API_KEY or OPEN_AI_KEY"):
        llm.live_settings(tmp_path / ".env", "chosen-model")


@pytest.mark.parametrize(
    ("override", "finish_reason", "status"),
    [(None, "stop", "success"), ("gemini-3.5-flash-lite", "length", "incomplete")],
)
def test_gemini_cli_sends_one_request_to_google(tmp_path, monkeypatch, override, finish_reason, status):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path / "results")
    monkeypatch.delenv("GEMINI_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    (tmp_path / ".env").write_text(f"GEMINI_KEY={MOCK_KEY}\nOPEN_AI_KEY=other-test-key\n", encoding="utf-8")
    expected_model = override or "gemini-3.1-flash-lite"
    raw = '{"relations":[]}' if status == "success" else '{"relations":['
    requests = []

    def handle(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/openai/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {MOCK_KEY}"
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == expected_model and body["max_tokens"] == 4096
        return Response(200, json={
            "id": "mock-gemini",
            "object": "chat.completion",
            "created": 0,
            "model": expected_model,
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": raw},
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 10, "total_tokens": 227},
        })

    def mock_sdk(**options):
        assert options["max_retries"] == 0
        return OpenAI(**options, http_client=Client(transport=MockTransport(handle)))

    monkeypatch.setattr(llm, "OpenAI", mock_sdk)
    args = [*PILOT_CLI[:-2], "--live"]
    if override:
        args.extend(["--model", override])
    assert benchmark.main(args) == (0 if status == "success" else 1)
    run = json.loads(next(benchmark.RESULTS.glob("*.jsonl")).read_bytes())
    assert run["provider"] == "gemini" and run["reported_model"] == expected_model
    assert run["status"] == status and run["raw_response"] == raw
    assert run["api_attempts"] == len(requests) == 1
    assert run["calls"][0]["messages"] == requests[0]["messages"]
    assert run["usage"] == {
        "input_tokens": 200, "output_tokens": 10, "total_tokens": 227,
        "cached_tokens": None, "reasoning_tokens": None,
    }
    assert MOCK_KEY not in json.dumps(run)


def test_cli_dry_run_uses_no_credentials_and_can_inspect_oversized_prompt(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("A dry run should not load live settings")

    monkeypatch.setattr(benchmark, "live_settings", forbidden)
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*PILOT_CLI, "--max-input-tokens", "1"]) == 0
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == "dry_run"
    assert run["usage"] is None and run["api_attempts"] == 0 and run["llm_calls"] == 1
    assert run["would_exceed_input_limit"] is True
    assert run["max_output_tokens"] == 4_096
    assert "Charles Quef" in run["calls"][0]["messages"][1]["content"]


def test_input_limit_counts_all_rendered_messages_and_blocks_before_sdk():
    messages = llm.render_messages(TEMPLATE, STATE)
    total_bytes = sum(len(item["role"].encode()) + len(item["content"].encode()) for item in messages)
    estimate = (total_bytes + 3) // 4
    assert estimate > (len(STATE["context"].encode()) + 3) // 4
    model = llm.LLM(model="invented-for-test", api_key=MOCK_KEY, max_input_tokens=estimate - 1)
    assert model.invoke(TEMPLATE, STATE) is None
    call = model.calls[0]
    assert call["status"] == "input_too_large"
    assert call["input_utf8_bytes"] == total_bytes
    assert call["estimated_input_tokens"] == estimate
    assert call["api_attempts"] == 0
    assert call["messages"] == messages
    assert "{all}" in messages[1]["content"] and "#{markdown}" in messages[1]["content"]


def test_cli_live_rejection_preserves_the_oversized_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: ("chosen-model", MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*PILOT_CLI, "--live", "--max-input-tokens", "1"]) == 1
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == "input_too_large"
    assert run["api_attempts"] == 0 and run["llm_calls"] == 1
    assert run["prediction"] is None and run["raw_response"] is None
    assert run["usage"] is None
    assert "[5]" in run["calls"][0]["messages"][1]["content"]


def test_equal_to_limit_sends_exact_prompt_once_and_records_usage(fake_provider):
    actual_usage = SimpleNamespace(
        input_tokens=50, output_tokens=10, total_tokens=60,
        input_tokens_details=SimpleNamespace(cached_tokens=12),
        output_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )
    fake_provider["result"] = response(usage=actual_usage)
    messages = llm.render_messages(TEMPLATE, STATE)
    _, estimate = llm.measure_input(messages)
    model = llm.LLM(model="gpt-4.1-mini", api_key=MOCK_KEY, max_input_tokens=estimate, max_output_tokens=60)
    assert model.invoke(TEMPLATE, STATE) == '{"relations":[]}'
    assert fake_provider["options"]["api_key"] == MOCK_KEY
    assert fake_provider["options"]["max_retries"] == 0
    assert fake_provider["requests"] == [{
        "model": "gpt-4.1-mini", "input": messages, "max_output_tokens": 60,
        "truncation": "disabled", "store": False, "temperature": 0,
    }]
    call = model.calls[0]
    assert call["usage"] == {
        "input_tokens": 50, "output_tokens": 10, "total_tokens": 60,
        "cached_tokens": 12, "reasoning_tokens": 4,
    }
    assert call["estimated_input_tokens"] == estimate
    assert call["reported_model"] == "reported-snapshot"
    assert call["api_attempts"] == 1 and call["latency_seconds"] >= 0


def test_incomplete_response_persists_partial_text_and_never_retries(tmp_path, monkeypatch, fake_provider):
    partial = '{"relations":[{"h":0,'
    fake_provider["result"] = response(partial, "incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"))
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: ("gpt-6-luna", MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*PILOT_CLI, "--live"]) == 1
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == "incomplete"
    assert run["prediction"] is None
    assert run["raw_response"] == partial
    assert run["error"] == {"reason": "max_output_tokens"}
    assert run["api_attempts"] == len(fake_provider["requests"]) == 1
    assert "temperature" not in fake_provider["requests"][0]


@pytest.mark.parametrize(("raw", "status"), [("<broken>", "invalid_json"), ('{"rows":[]}', "invalid_schema")])
def test_failed_prediction_is_saved_verbatim(tmp_path, monkeypatch, fake_provider, raw, status):
    fake_provider["result"] = response(raw)
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: ("chosen-model", MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*PILOT_CLI, "--live"]) == 1
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["prediction"] is None and run["raw_response"] == raw
    assert run["status"] == status and run["api_attempts"] == 1


def test_provider_error_does_not_leak_key_or_retry_with_different_model(fake_provider):
    request = Request("POST", "https://api.openai.com/v1/responses")
    fake_provider["result"] = APIError(f"Rejected {MOCK_KEY} and sk-truncated…", request=request, body={"code": "auth"})
    model = llm.LLM(model="chosen-model", api_key=MOCK_KEY)
    assert model.invoke(TEMPLATE, STATE) is None
    assert len(fake_provider["requests"]) == 1
    assert fake_provider["requests"][0]["model"] == "chosen-model"
    call = model.calls[0]
    assert call["status"] == "provider_error"
    assert call["api_attempts"] == 1
    assert call["error"]["code"] == "auth"
    assert MOCK_KEY not in json.dumps(call) and "sk-truncated" not in json.dumps(call)


def test_refusal_preserves_the_provider_text(fake_provider):
    refusal = SimpleNamespace(type="refusal", refusal="Unable to provide the requested output.")
    fake_provider["result"] = response("", output=[SimpleNamespace(type="message", content=[refusal])])
    model = llm.LLM(model="chosen-model", api_key=MOCK_KEY)
    assert model.invoke(TEMPLATE, STATE) == ""
    assert model.calls[0]["status"] == "refused"
    assert model.calls[0]["error"]["refusals"] == [refusal.refusal]


def test_cli_rejects_live_test_split_and_whole_pilot(tmp_path, monkeypatch, capsys):
    directory = benchmark.DATA / benchmark.DATASETS["docred"][0]
    test_id = benchmark.read_records(directory, "test")[0]["id"]
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)

    def forbidden(*args):
        pytest.fail("A disallowed selection should not load credentials")

    monkeypatch.setattr(benchmark, "live_settings", forbidden)
    for selection in (["--instance", test_id], ["--split", "pilot"]):
        with pytest.raises(SystemExit) as error:
            benchmark.main([*PILOT_CLI[:4], *selection, "--live"])
        assert error.value.code == 2
        assert "pilot" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


DEEPREAD_CLI = ["--method", "deepread", "--dataset", "docred", "--instance", "docred-dev-0244"]
TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    for name in ("bm25_search", "read_section")
]


def completion(message, finish="stop", usage=None):
    return {
        "id": "mock-deepread",
        "object": "chat.completion",
        "created": 0,
        "model": llm.DEFAULT_GEMINI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        "usage": usage,
    }


def structured_call(identifier, name, arguments, **extra):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
        **extra,
    }


@pytest.fixture
def gemini_turns(monkeypatch):
    requests, replies = [], []

    def transport(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/openai/chat/completions"
        requests.append(json.loads(request.content))
        assert replies, "Unexpected request after the scripted provider replies"
        code, payload = replies.pop(0)
        return Response(code, json=payload)

    def sdk(**options):
        assert options["max_retries"] == 0 and options["api_key"] == MOCK_KEY
        return OpenAI(**options, http_client=Client(transport=MockTransport(transport)))

    monkeypatch.setattr(llm, "OpenAI", sdk)
    return requests, replies


def test_chat_preserves_extensions_parallel_ids_and_immutable_request_snapshots(gemini_turns):
    requests, replies = gemini_turns
    signature = {"google": {"thought_signature": "opaque-tool-signature"}}
    message_extra = {"google": {"thought_signature": "opaque-message-signature"}}
    calls = [
        structured_call("bm25-1", "bm25_search", {"query": "Alpha"}, extra_content=signature),
        structured_call("read-2", "read_section", {"doc_id": "2", "node_id": "1"}),
    ]
    usage = {
        "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 1},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    replies.extend([
        (200, completion({"content": None, "tool_calls": calls, "extra_content": message_extra}, "tool_calls", usage)),
        (200, completion({"content": '{"rows":[]}'}, usage=usage)),
    ])
    client = llm.LLM(provider="gemini", model=llm.DEFAULT_GEMINI_MODEL, api_key=MOCK_KEY)
    history = [{"role": "system", "content": "Local documents only."}, {"role": "user", "content": "Find evidence."}]
    first = client.chat(history, TOOLS)
    assert first["choices"][0]["message"]["tool_calls"] == calls
    # Emulate a caller that rebuilds the assistant message and loses provider extensions.
    history.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{key: value for key, value in call.items() if key != "extra_content"} for call in calls],
    })
    history.extend({"role": "tool", "tool_call_id": call["id"], "content": '{"ok":true}'} for call in calls)
    assert client.chat(history, TOOLS)["choices"][0]["message"]["content"] == '{"rows":[]}'
    assert len(requests) == 2
    forwarded = requests[1]["messages"][2]
    assert forwarded["tool_calls"] == calls and forwarded["extra_content"] == message_extra
    assert [item["tool_call_id"] for item in requests[1]["messages"][3:]] == ["bm25-1", "read-2"]
    assert requests[1]["tools"] == TOOLS and requests[1]["tool_choice"] == "auto"
    assert not {"include_reasoning", "temperature", "parallel_tool_calls"} & requests[1].keys()
    history[0]["content"] = "Changed later"
    history[2]["tool_calls"][0]["function"]["arguments"] = "Changed later"
    assert client.calls[0]["messages"] == requests[0]["messages"]
    assert client.calls[1]["request"] == requests[1]
    assert client.calls[0]["response_message"]["tool_calls"][0]["extra_content"] == signature
    assert llm.aggregate_usage(client.calls) == {
        "input_tokens": 24, "output_tokens": 6, "total_tokens": 36, "cached_tokens": 2, "reasoning_tokens": 6,
    }
    assert MOCK_KEY not in json.dumps(client.calls)


def test_cumulative_budget_covers_full_requests_and_prevents_second_attempt(gemini_turns):
    requests, replies = gemini_turns
    messages = [{"role": "system", "content": "Requirements."}, {"role": "user", "content": "DocRED task."}]
    payload = {"model": llm.DEFAULT_GEMINI_MODEL, "messages": messages, "max_tokens": 4096,
               "tools": TOOLS, "tool_choice": "auto"}
    exact = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    limit = (exact + 3) // 4
    replies.append((200, completion({"content": "First answer."})))
    client = llm.LLM(
        provider="gemini", model=llm.DEFAULT_GEMINI_MODEL, api_key=MOCK_KEY, max_total_input_tokens=limit,
    )
    assert client.chat(messages, TOOLS)
    assert client.calls[0]["input_utf8_bytes"] == exact
    assert client.chat(messages, TOOLS) is None
    assert client.calls[1]["status"] == "total_input_budget_exceeded"
    assert client.calls[1]["cumulative_estimated_input_tokens"] == limit
    assert client.calls[1]["api_attempts"] == 0 and len(requests) == 1
    assert llm.aggregate_usage(client.calls) is None


def test_agent_request_limit_rejects_schema_and_message_bytes_before_sending():
    messages = [{"role": "user", "content": "X"}]
    client = llm.LLM(provider="gemini", model=llm.DEFAULT_GEMINI_MODEL, api_key=MOCK_KEY, max_input_tokens=10)
    assert llm.measure_input(messages)[1] < 10
    assert client.chat(messages, TOOLS) is None
    call = client.calls[0]
    assert call["status"] == "input_too_large" and call["estimated_input_tokens"] > 10
    assert call["api_attempts"] == client.accepted_input_tokens == 0


@pytest.mark.parametrize(
    ("http_status", "reply", "status", "raw"),
    [
        (429, {"error": {"message": "Project quota exhausted", "status": "RESOURCE_EXHAUSTED"}},
         "provider_error", None),
        (200, completion({"content": '{"relations":['}, "length"), "incomplete", '{"relations":['),
        (200, completion({"content": None, "refusal": "Fixture refusal"}, "content_filter"), "refused", None),
        (200, completion({"content": None, "tool_calls": [
            structured_call("allowed", "bm25_search", {"query": "should not be dispatched"}),
            structured_call("blocked", "semantic_retrieval", {"query": "must not run"}),
        ]}, "tool_calls"), "unexpected_tool", None),
    ],
)
def test_terminal_agent_failures_stop_before_another_request(
    tmp_path, monkeypatch, gemini_turns, http_status, reply, status, raw,
):
    requests, replies = gemini_turns
    replies.append((http_status, reply))
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: (llm.DEFAULT_GEMINI_MODEL, MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*DEEPREAD_CLI, "--live"]) == 1
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == status and run["raw_response"] == raw and run["prediction"] is None
    assert len(requests) == run["api_attempts"] == run["llm_calls"] == 1
    assert run["method_trace"]["termination_reason"] == status
    assert all(event["event"] != "tool_call" for event in run["method_trace"]["events"])
    if http_status == 429:
        assert run["error"]["http_status"] == 429


def test_total_usage_keeps_missing_contributions_unknown():
    present = dict(input_tokens=17, output_tokens=3, total_tokens=20, cached_tokens=None, reasoning_tokens=0)
    incomplete = dict(input_tokens=23, output_tokens=None, total_tokens=31, cached_tokens=5, reasoning_tokens=4)
    calls = [{"api_attempts": 1, "usage": value} for value in [present, incomplete]]
    calls.append({"api_attempts": 0, "usage": None})
    assert llm.aggregate_usage(calls) == {
        "input_tokens": 40, "output_tokens": None, "total_tokens": 51, "cached_tokens": None, "reasoning_tokens": 4,
    }
    calls.append({"api_attempts": 1, "usage": None})
    assert all(value is None for value in llm.aggregate_usage(calls).values())


def test_cli_rejects_deepread_openai_before_loading_credentials(monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("An unsupported combination must be rejected before credentials are loaded.")

    monkeypatch.setattr(benchmark, "live_settings", forbidden)
    with pytest.raises(SystemExit) as error:
        benchmark.main([*DEEPREAD_CLI, "--provider", "openai", "--live"])
    assert error.value.code == 2
    assert "Select --provider gemini" in capsys.readouterr().err


def test_original_agent_round_trip_through_mocked_gemini(tmp_path, monkeypatch, gemini_turns):
    requests, replies = gemini_turns
    metadata = {"google": {"thought_signature": "signed-bm25-turn"}}
    read_metadata = {"google": {"thought_signature": "signed-read-turn"}}
    first_tool = structured_call(
        "lookup", "bm25_search", {"query": "Charles", "scope": "doc", "doc_id": "1"},
        extra_content=metadata,
    )
    second_tool = structured_call(
        "read", "read_section", {"doc_id": "1", "node_id": "1", "start_paragraph": 0, "end_paragraph": 2},
        extra_content=read_metadata,
    )
    usage = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    replies.extend([
        (200, completion({"content": None, "tool_calls": [first_tool], "extra_content": metadata}, "tool_calls", usage)),
        (200, completion({"content": None, "tool_calls": [second_tool]}, "tool_calls", usage)),
        (200, completion({"content": ' {"relations":[]} '}, usage=usage)),
    ])
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: (llm.DEFAULT_GEMINI_MODEL, MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    assert benchmark.main([*DEEPREAD_CLI, "--live"]) == 0
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == "success" and run["prediction"] == {"relations": []}
    assert run["raw_response"] == ' {"relations":[]} '
    assert len(requests) == run["llm_calls"] == run["api_attempts"] == 3
    assert run["usage"]["input_tokens"] == 60 and run["usage"]["total_tokens"] == 72
    assert requests[1]["messages"][2]["extra_content"] == metadata
    assert requests[1]["messages"][2]["tool_calls"] == [first_tool]
    assert requests[2]["messages"][4]["tool_calls"] == [second_tool]
    assert [msg["tool_call_id"] for msg in requests[2]["messages"] if msg["role"] == "tool"] == ["lookup", "read"]
    for request in requests[1:]:
        assert json.loads(request["messages"][-1]["content"])["ok"]


def test_original_agent_recovers_from_an_invalid_aoe_node_id(tmp_path, monkeypatch, gemini_turns):
    requests, replies = gemini_turns
    for node_id, call_id in [("6,7,8", "invalid-read"), ("6", "corrected-read")]:
        call = structured_call(call_id, "read_section", {
            "doc_id": "2", "node_id": node_id, "start_paragraph": 0, "end_paragraph": 1,
        })
        replies.append((200, completion({"content": None, "tool_calls": [call]}, "tool_calls")))
    raw = '```json\n{"rows":[]}\n```'
    replies.append((200, completion({"content": raw})))
    monkeypatch.setattr(benchmark, "live_settings", lambda *args: (llm.DEFAULT_GEMINI_MODEL, MOCK_KEY))
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)

    assert benchmark.main([
        "--method", "deepread", "--dataset", "aoe", "--instance", "academic_PMD_1_en", "--live",
    ]) == 0
    run = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert run["status"] == "success"
    assert run["prediction"] == {"rows": []} and run["raw_response"] == raw
    assert len(requests) == run["llm_calls"] == 3
    assert run["prompt_version"].startswith("deepread-bm25-v2/")
    assert "Pass exactly one doc_id and one node_id" in requests[0]["messages"][0]["content"]

    invalid = requests[1]["messages"][-1]
    assert invalid["tool_call_id"] == "invalid-read"
    error = json.loads(invalid["content"])
    assert error["ok"] is False
    assert error["ref"]["node_id"] == "6,7,8"
    assert error["error"] == "node_id '6,7,8' not found in doc '2'"
    assert {"6", "7", "8"} <= set(error["valid_node_ids"])
    assert "6,7,8" not in error["valid_node_ids"]
    assert "separate structured tool call for each node" in error["hint"]

    corrected = requests[2]["messages"][-1]
    assert corrected["tool_call_id"] == "corrected-read"
    section = json.loads(corrected["content"])
    assert section["ok"] is True
    assert section["ref"] == {"doc_id": "2", "node_id": "6", "paragraph_indexes": [0]}
    assert section["paragraphs"]
    assert "hint" not in section
