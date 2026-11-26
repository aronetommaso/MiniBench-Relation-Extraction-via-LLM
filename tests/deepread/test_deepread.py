import json
import sys

import pytest

import benchmark
import llm
from methods import deepread_adapter as deepread
from prompts import OUTPUT_KEYS


def pilot(dataset, identifier):
    directory, id_key = benchmark.DATASETS[dataset]
    directory = benchmark.DATA / directory
    raw = next(row for row in benchmark.read_records(directory, "pilot") if row[id_key] == identifier)
    return benchmark.prepare_input(dataset, raw, directory)


@pytest.fixture(autouse=True)
def forbid_unmocked_client(monkeypatch):
    def forbid(**kwargs):
        pytest.fail("DeepRead offline tests must not open a live SDK client.")

    monkeypatch.setattr(llm, "OpenAI", forbid)


@pytest.mark.parametrize(
    ("dataset", "identifier", "documents_expected"),
    [("docred", "docred-dev-0244", 1), ("aoe", "academic_PMD_1_en", 2)],
)
def test_original_agent_executes_both_tools_without_initial_body(dataset, identifier, documents_expected, monkeypatch):
    task, documents = pilot(dataset, identifier)
    original = deepread.load_upstream()["agent.runner"].run_agent
    frames = []

    class ObservedLLM(llm.LLM):
        def chat(self, *args, **kwargs):
            frames.append(sys._getframe(2).f_code)
            return super().chat(*args, **kwargs)

    client = ObservedLLM(
        provider="gemini", model=llm.DEFAULT_GEMINI_MODEL, simulated_response=json.dumps({OUTPUT_KEYS[dataset]: []}),
    )
    raw = deepread.extract(task, documents, client)
    assert frames == [original.__code__] * 3
    assert deepread.DEEPREAD_PROMPT_VERSION.endswith("structured-extraction-en-v2")
    status, prediction, _ = benchmark.parse_prediction(task, raw)
    assert (status, prediction) == ("success", {OUTPUT_KEYS[dataset]: []})
    assert len(client.calls) == 3 and all(call["simulated"] and call["api_attempts"] == 0 for call in client.calls)
    first = client.calls[0]["messages"]
    assert [message["role"] for message in first] == ["system", "user"]
    assert task["query"] in first[1]["content"]
    assert "Directory Structure" in first[0]["content"]
    assert "final answer must consist of valid JSON" in first[0]["content"]
    schemas = client.calls[0]["request"]["tools"]
    assert {tool["function"]["name"] for tool in schemas} == {"bm25_search", "read_section"}
    assert all(doc["name"] in first[0]["content"] for doc in documents)
    combined = "\n".join(message["content"] for message in first)
    assert all(doc["text"] not in combined for doc in documents)
    if dataset == "docred":
        assert all(line not in combined for line in documents[0]["text"].splitlines())
        assert '"entity_id": 17' in combined
        assert json.dumps(task["relations"], ensure_ascii=False) in combined

    trace = client.method_trace
    assert trace["termination_reason"] == "final_answer" and trace["executed_rounds"] == 3
    assert list(trace["source_documents"].values()) == [doc["name"] for doc in documents]
    assert trace["configuration"]["bm25_topk"] == 2
    results = [event for event in trace["events"] if event["event"] == "tool_result"]
    assert [entry["tool"] for entry in results] == ["bm25_search"] * documents_expected + [
        "read_section"
    ] * documents_expected
    for round_index, expected in [(1, "results"), (2, "paragraphs")]:
        infos = [json.loads(msg["content"]) for msg in client.calls[round_index]["messages"] if msg["role"] == "tool"]
        assert len(infos) == round_index * documents_expected
        assert all(info["ok"] and info[expected] for info in infos[-documents_expected:])
    read_docs = {event["result"]["ref"]["doc_id"] for event in results if event["tool"] == "read_section"}
    assert read_docs == set(trace["source_documents"])
    assert len(client.calls[0]["messages"]) == 2


def test_round_limit_returns_no_diagnostic_as_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "RESULTS", tmp_path)
    exit_code = benchmark.main([
        "--method", "deepread", "--dataset", "docred", "--instance", "docred-dev-0244", "--max-rounds", "1",
    ])
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_bytes())
    assert exit_code == 1 and record["status"] == "max_rounds_reached"
    assert record["prediction"] is None and record["raw_response"] is None
    assert record["llm_calls"] == 1 and record["api_attempts"] == 0
    trace = record["method_trace"]
    assert trace["executed_rounds"] == 1 and trace["termination_reason"] == "max_rounds_reached"
    assert any(event["event"] == "max_rounds_reached" for event in trace["events"])
