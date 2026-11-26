import json
from pathlib import PurePosixPath

import pytest

import benchmark
from llm import LLM
from methods import vcl_adapter
from prompts import DOCRED_QUERY, OUTPUT_KEYS


def record_for(dataset, identifier):
    directory, id_key = benchmark.DATASETS[dataset]
    directory = benchmark.DATA / directory
    record = next(record for record in benchmark.read_records(directory, "pilot") if record[id_key] == identifier)
    return directory, record


def render(dataset, directory, record):
    task, documents = benchmark.prepare_input(dataset, record, directory)
    client = LLM(simulated_response=json.dumps({OUTPUT_KEYS[dataset]: []}))
    vcl_adapter.extract(task, documents, client)
    return task, documents, "\n".join(message["content"] for message in client.calls[0]["messages"])


def test_docred_uses_all_sentences_entities_and_relations_without_gold():
    directory, record = record_for("docred", "docred-dev-0244")
    record.update(id="BOOKKEEPING_SENTINEL", labels=[{"evidence": ["DOCRED_GOLD_SENTINEL"]}],
                  group="STRATUM_SENTINEL", source_index="SOURCE_INDEX_SENTINEL")
    record["vertexSet"][0][0]["extra"] = "NESTED_METADATA_SENTINEL"
    task, documents, prompt = render("docred", directory, record)

    assert DOCRED_QUERY in prompt
    assert f'source_name="{record["title"]}"' in prompt
    assert len(task["entities"]) == len(record["vertexSet"]) == 18
    assert len(task["relations"]) == 96
    assert json.dumps(task["relations"], ensure_ascii=False) in prompt
    clusters = [{"entity_id": index, "mentions": mentions} for index, mentions in enumerate(task["entities"])]
    assert json.dumps(clusters, ensure_ascii=False) in prompt
    assert len(record["sents"]) == 6
    for index, sentence in enumerate(record["sents"]):
        rendered = f"[{index}] {' '.join(sentence)}"
        assert rendered in documents[0]["text"] and rendered in prompt
    assert "SENTINEL" not in prompt


def test_aoe_uses_dynamic_schema_and_ordered_untouched_markdown_without_gold():
    directory, record = record_for("aoe", "academic_PMD_1_en")
    record.update(
        record_id="BOOKKEEPING_SENTINEL", answers=[{"paper_name": "AOE_GOLD_SENTINEL"}],
        key_columns=["ALIGNMENT_SENTINEL"], evidence=["EVIDENCE_SENTINEL"],
        benchmark_name="BENCHMARK_SENTINEL", domain="DOMAIN_SENTINEL", task="TASK_SENTINEL",
        ref_doc=["REFERENCE_SENTINEL"], estimated_tokens="TOKENS_SENTINEL",
    )
    for column in record["table_schema"]["columns"]:
        column["valid_type"] = "VALID_TYPE_SENTINEL"
    task, documents, prompt = render("aoe", directory, record)

    assert record["query"] in prompt
    assert [column["name"] for column in task["columns"]] == [
        "paper_name", "method", "result", "models_and_settings",
    ]
    assert json.dumps(task["columns"], ensure_ascii=False) in prompt
    assert all(set(column) == {"name", "about"} for column in task["columns"])
    assert prompt.count("<DOC ") == len(documents) == 2
    position = 0
    for doc, reference in zip(documents, record["documents"]):
        assert doc["name"] == PurePosixPath(reference["source_path"]).name
        assert doc["text"].encode("utf-8") == (directory / reference["path"]).read_bytes()
        position = prompt.index(doc["text"], position) + len(doc["text"])
    assert "SENTINEL" not in prompt


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ('{"relations":[]}', "success"),
        ('{"relations":[{"h":0,"r":"P19","t":1,"evidence":[0,5]}]}', "success"),
        ('```json\n{"relations":[]}\n```', "invalid_json"),
        ('{"relations":[],"relations":[]}', "invalid_json"),
        ('{"relations":NaN}', "invalid_json"),
        ('{"relations":[{"h":0,"r":"P19","t":1,"evidence":[6]}]}', "invalid_schema"),
        ('{"relations":[{"h":true,"r":"P19","t":1,"evidence":[]}]}', "invalid_schema"),
        ('{"relations":[{"h":0,"r":"UNKNOWN","t":1,"evidence":[]}]}', "invalid_schema"),
    ],
)
def test_docred_structural_output_validation(raw, status):
    directory, record = record_for("docred", "docred-dev-0244")
    task, _ = benchmark.prepare_input("docred", record, directory)
    outcome, prediction, error = benchmark.parse_prediction(task, raw)
    assert outcome == status
    assert (prediction is not None) == (status == "success")
    assert (error is None) == (status == "success")


def test_aoe_schema_requires_exact_columns_and_preserves_values():
    directory, record = record_for("aoe", "academic_PMD_1_en")
    task, _ = benchmark.prepare_input("aoe", record, directory)

    row = {"paper_name": "Example", "method": "ABC", "result": "3E-4 kg", "models_and_settings": ""}
    good = benchmark.parse_prediction(task, json.dumps({"rows": [row]}))
    assert good == ("success", {"rows": [row]}, None)
    invalid = benchmark.parse_prediction(task, json.dumps({"rows": [dict(row, extra="field")]}))
    assert invalid[0] == "invalid_schema"
    assert invalid[1] is None
