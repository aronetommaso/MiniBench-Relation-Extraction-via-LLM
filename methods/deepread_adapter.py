"""Run the pinned DeepRead agent with local BM25 and contiguous section reading."""

import builtins
from copy import deepcopy
from functools import cache
from importlib.util import module_from_spec, resolve_name, spec_from_file_location
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import FunctionType

from llm import API_URLS
from prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS, task_instructions


DEEPREAD_ROOT = Path(__file__).resolve().parents[1] / "external" / "DeepRead"
DEEPREAD_REVISION = "25360c9f109cdcd29785bbef7e6a36963b50b374"
DEEPREAD_PROMPT_VERSION = f"deepread-bm25-v2/{PROMPT_VERSION}"
TOOLS = frozenset({"bm25_search", "read_section"})
READ_SECTION_GUIDANCE = (
    "Pass exactly one doc_id and one node_id from the directory or search results to read_section. "
    "Use a separate structured tool call for each node. After a lookup error, use an ID from the returned list."
)
AGENT_SETTINGS = {
    "bm25_topk": 2,
    "neighbor_window": (0, 0),
    "enable_multimodal": False,
    "enable_vector": False,
    "enable_hybrid": False,
    "enable_semantic": False,
    "disable_bm25": False,
    "disable_regex": True,
    "disable_read": False,
    "tool_fallback": False,
    "enable_reasoning": False,
}
UPSTREAM_MODULES = (
    "agent.runner", "agent.llm", "index.markdown_parser", "prompt.system",
    "tool.bm25_search", "tool.fallback", "tool.hybrid_search", "tool.read_section",
    "tool.regex_search", "tool.retrieval", "tool.schema", "tool.semantic_retrieval",
    "tool.utils", "tool.vector_search",
)
UPSTREAM_PACKAGES = {"agent", "index", "prompt", "tool"}


@cache
def load_upstream():
    paths = {name: Path(*name.split(".")).with_suffix(".py") for name in UPSTREAM_MODULES}
    git = ["git", "-C", str(DEEPREAD_ROOT)]
    try:
        revision = subprocess.run(
            [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot read the DeepRead revision. Initialize external/DeepRead and check Git.") from error
    if revision != DEEPREAD_REVISION:
        raise RuntimeError(f"Expected DeepRead revision {DEEPREAD_REVISION}, found {revision}")
    diff = subprocess.run(
        [*git, "diff", "--quiet", DEEPREAD_REVISION, "--", *(p.as_posix() for p in paths.values())],
        check=False, capture_output=True,
    )
    if diff.returncode:
        raise RuntimeError("The required DeepRead files differ from the pinned revision")

    modules = {}

    def local_import(target, globals=None, locals=None, fromlist=(), level=0):
        absolute = resolve_name("." * level + target, globals["__package__"]) if level else target
        if absolute.split(".")[0] in UPSTREAM_PACKAGES:
            if not fromlist or absolute not in paths:
                raise ImportError(f"Unexpected internal DeepRead import: {absolute}")
            return load(absolute)
        return builtins.__import__(target, globals, locals, fromlist, level)

    def load(name):
        if name in modules:
            return modules[name]
        path = (DEEPREAD_ROOT / paths[name]).resolve()
        if not path.is_relative_to(DEEPREAD_ROOT):
            raise RuntimeError(f"DeepRead source is outside the pinned submodule: {name}")
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}
        # Read the pinned source without registering packages or writing submodule bytecode.
        exec(compile(spec.loader.get_source(name), str(path), "exec", dont_inherit=True), module.__dict__)
        modules[name] = module
        return module

    for name in UPSTREAM_MODULES:
        load(name)
    # Use DeepRead's local whitespace fallback for directory token estimates.
    modules["tool.utils"].tiktoken = None
    return modules


def _with_globals(function, **overrides):
    # Rebind just the adapter hooks in a private namespace for this execution.
    bound = FunctionType(function.__code__, {**function.__globals__, **overrides},
                         function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def build_index(task, documents):
    modules = load_upstream()
    sources = {str(i): document["name"] for i, document in enumerate(documents, 1)}
    if task["dataset"] == "docred":
        document, = documents
        nodes = [{
            "doc_id": "1", "id": "1", "title": document["name"],
            "paragraphs": document["text"].splitlines(), "children": [],
        }]
    else:
        parser = modules["index.markdown_parser"]
        detect_image = parser._extract_image

        def keep_image_line_as_text(line, md_dir):
            return line if detect_image(line, md_dir) is not None else None

        parse = _with_globals(parser.parse_markdown_to_corpus, _extract_image=keep_image_line_as_text)
        nodes = []
        with TemporaryDirectory(prefix="minibench-deepread-") as directory:
            for doc_id, document in zip(sources, documents):
                path = Path(directory) / f"{doc_id}.md"
                path.write_bytes(document["text"].encode("utf-8"))
                corpus = parse(str(path))
                if not corpus["nodes"]:
                    raise ValueError(f"Empty DeepRead corpus for {document['name']}")
                nodes.extend(dict(node, doc_id=doc_id) for node in corpus["nodes"])

    index = modules["tool.retrieval"].DocIndex(nodes, neighbor_window=AGENT_SETTINGS["neighbor_window"])
    return index, sources


class _AgentStop(BaseException):
    """Leave the upstream loop, which otherwise catches transport exceptions."""

    def __init__(self, status, error):
        self.status = status
        self.error = error


def _validate_tool_calls(calls):
    seen = set()
    if not isinstance(calls, list):
        raise _AgentStop("invalid_tool_call", {"message": "Expected a list of structured tool calls."})
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or call.get("type") != "function":
            raise _AgentStop("invalid_tool_call", {"message": "Expected a structured function call."})
        name = function.get("name")
        if not isinstance(name, str) or name not in TOOLS:
            raise _AgentStop("unexpected_tool", {"message": f"DeepRead tool is not enabled: {name!r}"})
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise _AgentStop("invalid_tool_call", {"message": "Expected distinct, nonempty tool call IDs."})
        seen.add(identifier)


def _scripted_turn(index, answer, messages):
    round_index = sum(message["role"] == "assistant" for message in messages)
    calls = []
    if round_index == 0:
        first_paragraphs = {}
        for paragraph in index.par_docs:
            if paragraph["tokens"]:
                first_paragraphs.setdefault(paragraph["doc_id"], paragraph)
        if set(first_paragraphs) != set(index.nodes_by_doc):
            raise _AgentStop("input_error", {"message": "No searchable text in a source document."})
        calls = [
            ("bm25_search", {"query": " ".join(paragraph["tokens"][:4]), "scope": "doc", "doc_id": doc_id})
            for doc_id, paragraph in first_paragraphs.items()
        ]
    elif round_index == 1:
        for message in messages:
            if message["role"] != "tool":
                continue
            result = json.loads(message["content"])
            hits = result.get("results")
            if not hits:
                raise _AgentStop("extraction_error", {"message": "Scripted BM25 returned no hits."})
            reference = hits[0]["ref"]
            start = reference["paragraph_indexes"][0]
            calls.append(("read_section", {
                "doc_id": reference["doc_id"], "node_id": reference["node_id"],
                "start_paragraph": start, "end_paragraph": start + 2,
            }))

    if not calls:
        return {"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}
    tool_calls = [
        {
            "id": f"dry-{round_index + 1}-{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
        for index, (name, arguments) in enumerate(calls, 1)
    ]
    return {"choices": [{
        "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
        "finish_reason": "tool_calls",
    }]}


def extract(task, documents, llm):
    """Execute the original agent loop and return the provider's final or partial text."""
    index, sources = build_index(task, documents)
    events = []
    trace = {
        "configuration": {**AGENT_SETTINGS, "max_rounds": llm.max_rounds},
        "source_documents": sources,
        "directory": index.overview(),
        "events": events,
        "executed_rounds": 0,
        "termination_reason": None,
    }
    llm.method_trace = trace
    original_read_section = index.read_section

    def read_section(doc_id, node_id, start_paragraph, end_paragraph, include_images=True):
        result = original_read_section(doc_id, node_id, start_paragraph, end_paragraph, include_images)
        if not result["ok"]:
            valid_nodes = index.nodes_by_doc.get(str(doc_id))
            result["hint"] = READ_SECTION_GUIDANCE
            if valid_nodes:
                result["valid_node_ids"] = list(valid_nodes)
            else:
                result["valid_doc_ids"] = list(index.nodes_by_doc)
        return result

    index.read_section = read_section

    class Logger:
        round = 0

        def log(self, event, **payload):
            self.round = payload.get("round", self.round)
            events.append({"event": event, "round": self.round, **deepcopy(payload)})

    logger = Logger()
    original_prompt = load_upstream()["prompt.system"].build_system_prompt

    def build_system_prompt(doc_index, tool_names, enable_reasoning):
        structure = original_prompt(doc_index, tool_names, enable_reasoning=enable_reasoning)
        names = "\n".join(f"- {doc_id}: {name}" for doc_id, name in sources.items())
        return (
            f"{structure}\n\n## Source documents\n{names}\n\n{READ_SECTION_GUIDANCE}\n\n"
            f"{SYSTEM_INSTRUCTIONS}\nReturn the final JSON without Markdown formatting."
        )

    def transport(*, payload, **_):
        tools = payload["tools"]
        if {tool["function"]["name"] for tool in tools} != TOOLS:
            raise _AgentStop("unexpected_tool", {"message": "DeepRead advertised an unexpected tool schema."})
        simulate = None
        if llm.simulated_response is not None:
            def simulate(messages):
                return _scripted_turn(index, llm.simulated_response, messages)

        try:
            response = llm.chat(payload["messages"], tools, simulate=simulate)
        except Exception as error:
            raise _AgentStop("provider_error", {
                "type": type(error).__name__, "message": "DeepRead chat transport failed.",
            }) from error
        call = llm.calls[-1]
        if response is None:
            raise _AgentStop(call["status"], call["error"])
        _validate_tool_calls(response["choices"][0]["message"].get("tool_calls") or [])
        return response

    run = _with_globals(
        load_upstream()["agent.runner"].run_agent, http_chat_completions=transport,
        build_system_prompt=build_system_prompt,
    )
    try:
        run(
            model=llm.model, base_url=API_URLS["gemini"], doc_index=index, user_question=task_instructions(task),
            logger=logger, max_rounds=llm.max_rounds, **AGENT_SETTINGS,
        )
    except _AgentStop as error:
        trace.update(termination_reason=error.status, error=error.error)
        if llm.calls and "status" not in llm.calls[-1]:
            llm.calls[-1].update(status=error.status, error=error.error)
        logger.log("agent_stopped", reason=error.status, error=error.error)
    else:
        trace["termination_reason"] = events[-1]["event"]
        if trace["termination_reason"] == "max_rounds_reached":
            trace["error"] = {"message": "DeepRead exhausted its rounds without a final answer."}

    trace["executed_rounds"] = sum(event["event"] == "llm_request" for event in events)
    return llm.calls[-1].get("raw_response") if llm.calls else None
