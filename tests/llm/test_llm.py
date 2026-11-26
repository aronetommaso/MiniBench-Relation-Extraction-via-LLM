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
