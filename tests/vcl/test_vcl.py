import inspect
from pathlib import Path
import sys
from types import ModuleType

import pytest
import spacy

from llm import LLM
from methods import vcl_adapter


EXPECTED_REVISION = "6b8539edaf55e621f6e551c58e32a065a28da4cf"
TASK = {
    "dataset": "aoe",
    "query": "Find the reported measurements.",
    "columns": [{"name": "Value {SI}", "about": "Preserve scientific notation."}],
}
DOCUMENTS = [
    {"name": 'Paper "One".md', "type": "markdown", "text": "# First Heading\r\n\r\nMixedCase 3E-4 {literal}\r\n"},
    {"name": "Paper Two.md", "type": "markdown", "text": "# Second Heading\n\n**ABC** 7E+8"},
]


@pytest.mark.parametrize("documents", [DOCUMENTS[:1], DOCUMENTS])
def test_calls_pinned_original_operator_once(documents, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The atomic extraction path must not load a spaCy model")

    monkeypatch.setattr(spacy, "load", unexpected)
    operator = vcl_adapter.load_operator()
    assert vcl_adapter.VCL_REVISION == EXPECTED_REVISION
    assert Path(inspect.getfile(operator)).resolve() == (
        vcl_adapter.VCL_ROOT / vcl_adapter.TOOLS / "estrai_semantico.py"
    )
    original_calls = []

    class ObservedLLM(LLM):
        def invoke(self, prompt, state):
            original_calls.append(sys._getframe(1).f_code)
            return super().invoke(prompt, state)

    client = ObservedLLM(simulated_response='{"rows":[]}')
    assert vcl_adapter.extract(TASK, documents, client) == '{"rows":[]}'
    assert original_calls == [operator.__code__]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["api_attempts"] == 0
    prompt = call["messages"][1]["content"]
    assert prompt.count("<DOC ") == len(documents)
    assert prompt.count("</DOC>") == len(documents)
    assert 'source_ref="doc-1" source_name="Paper &quot;One&quot;.md"' in prompt
    assert "Non sono presenti condizioni aggiuntive" not in prompt
    last_end = 0
    for index, document in enumerate(documents, start=1):
        start = prompt.index(f'<DOC source_ref="doc-{index}"')
        assert start >= last_end
        end = prompt.index("</DOC>", start)
        assert prompt[prompt.index("\n", start) + 1:end] == document["text"] + "\n"
        last_end = end


def test_original_formatter_lowercases_lists_and_preserves_strings():
    formatter = vcl_adapter.load_operator().__globals__["format_context"]
    assert "mixedcase 3e-4" in formatter(DOCUMENTS)
    original = "<DOC>\nMixedCase 3E-4\n</DOC>"
    assert formatter(original) == original


def test_loader_leaves_other_methods_imports_alone(monkeypatch):
    names = ("api.dqlEngine.executor.tools.converters", "utils.file_manager")
    existing = {name: ModuleType(name) for name in names}
    for name, module in existing.items():
        monkeypatch.setitem(sys.modules, name, module)
    import_path = list(sys.path)
    vcl_adapter.load_operator.cache_clear()
    try:
        operator = vcl_adapter.load_operator()
        assert operator.__globals__["format_context"]("MiXeD") == "MiXeD"
        assert sys.path == import_path
        assert all(sys.modules[name] is module for name, module in existing.items())
        excluded = ("executor", "tools.integra", "tools.retrieval")
        assert all(f"api.dqlEngine.executor.{name}" not in sys.modules for name in excluded)
    finally:
        vcl_adapter.load_operator.cache_clear()
