from pathlib import Path
from types import ModuleType
import sys

import benchmark
from methods import deepread_adapter as deepread


def test_upstream_loader_isolated_and_pinned(monkeypatch):
    names = ("agent", "index", "prompt", "tool", "tool.retrieval")
    foreign = {name: ModuleType(name) for name in names}
    for name, module in foreign.items():
        monkeypatch.setitem(sys.modules, name, module)
    search_path = sys.path[:]
    deepread.load_upstream.cache_clear()
    try:
        modules = deepread.load_upstream()
        for name, module in modules.items():
            relative = Path(*name.split(".")).with_suffix(".py")
            assert Path(module.__file__) == deepread.DEEPREAD_ROOT / relative
        assert deepread.DEEPREAD_REVISION == "25360c9f109cdcd29785bbef7e6a36963b50b374"
        assert sys.path == search_path
        assert all(sys.modules[name] is foreign[name] for name in foreign)
        assert modules["tool.utils"].tiktoken is None
    finally:
        deepread.load_upstream.cache_clear()


def test_docred_indexes_one_paragraph_per_numbered_sentence():
    directory = benchmark.DATA / "relation_extraction"
    record = next(row for row in benchmark.read_records(directory, "pilot") if row["id"] == "docred-dev-0244")
    task, documents = benchmark.prepare_input("docred", record, directory)
    index, sources = deepread.build_index(task, documents)
    assert sources == {"1": record["title"]}
    assert set(index.nodes_by_doc["1"]) == {"1"}
    assert index.nodes_by_doc["1"]["1"]["paragraphs"] == [
        f"[{i}] {' '.join(tokens)}" for i, tokens in enumerate(record["sents"])
    ]
    assert index.neighbor_window is None
    assert len(task["relations"]) == 96
    result = index.read_section("1", "1", 1, 3, include_images=False)
    assert result["ref"]["paragraph_indexes"] == [1, 2]
    assert result["paragraphs"][1]["text"].startswith("[2] ")


def test_aoe_hierarchy_tables_and_inline_text_remain_searchable(tmp_path, monkeypatch):
    inline = "Keep this caption ![Plot 7E-4](https://example.invalid/chart.png) and these measurements."
    markdown_table = "| Name | Score |\n| --- | --- |\n| Foo | 6E-4 |"
    html_table = "<table>\n<tr><td>7E-5</td></tr>\n</table>"
    first = f"# Intro\n\nInitial paragraph.\n\n## Results\n\n{inline}\n\n{markdown_table}\n\n{html_table}\n"
    second = "# Intro\n\nUnique second paper text.\n\n## Results\n\nScientific measurement 9E-6.\n"
    documents = [
        {"name": "First paper.md", "type": "markdown", "text": first},
        {"name": "Second paper.md", "type": "markdown", "text": second},
    ]
    raw_path = tmp_path / "sample.md"
    raw_path.write_text(first, encoding="utf-8")
    mods = deepread.load_upstream()
    parser = mods["index.markdown_parser"]
    upstream = parser.parse_markdown_to_corpus(str(raw_path))
    assert upstream["nodes"][1]["paragraphs"][0]["content"] == "Plot 7E-4"

    def forbid(*args, **kwargs):
        raise AssertionError("The local index must not call embeddings or reranking.")

    monkeypatch.setattr(mods["agent.llm"].requests, "post", forbid)
    index, sources = deepread.build_index({"dataset": "aoe"}, documents)
    assert sources == {"1": "First paper.md", "2": "Second paper.md"}
    for nodes in index.nodes_by_doc.values():
        assert nodes["1"]["title"] == "Intro" and nodes["1"]["children"] == ["2"]
    assert index.nodes_by_doc["1"]["2"]["paragraphs"] == [inline, markdown_table, html_table]
    assert parser.parse_markdown_to_corpus(str(raw_path)) == upstream
    for doc_id, query, text in [("1", "caption", inline), ("2", "measurement", "Scientific measurement 9E-6.")]:
        result = index.bm25_search(query, scope="doc", doc_id=doc_id, top_k=2, include_images=False)
        hit = result["results"][0]
        assert hit["ref"] == {"doc_id": doc_id, "node_id": "2", "paragraph_indexes": [0]}
        assert hit["text"] == text and hit["neighbors"] == []
    result = index.read_section("1", "2", 0, 2, include_images=False)
    assert [paragraph["text"] for paragraph in result["paragraphs"]] == [inline, markdown_table]
