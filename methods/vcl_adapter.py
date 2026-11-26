"""Call the pinned VCL semantic extractor with a single full-document context."""

import builtins
from functools import cache
from html import escape
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
from types import SimpleNamespace

from prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS, task_instructions


VCL_ROOT = Path(__file__).resolve().parents[1] / "external" / "VCL"
VCL_REVISION = "6b8539edaf55e621f6e551c58e32a065a28da4cf"
VCL_PROMPT_VERSION = f"vcl-full-document-v1/{PROMPT_VERSION}"
TOOLS = Path("api/dqlEngine/executor/tools")
TARGETS = {"docred": "Document-level relation extraction", "aoe": "Structured table extraction"}


def _load_source(path, dependencies=None):
    spec = spec_from_file_location(f"_benchmark_vcl_{path.stem}", path)
    module = module_from_spec(spec)
    if dependencies:
        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not level and fromlist and name in dependencies:
                return dependencies[name]
            return builtins.__import__(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}

    # Read upstream source directly so no bytecode is written into the submodule.
    source = spec.loader.get_source(spec.name)
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


@cache
def load_operator():
    """Return the original function with just its required VCL modules loaded."""
    files = (Path("utils/file_manager.py"), TOOLS / "converters.py", TOOLS / "estrai_semantico.py")
    git = ["git", "-C", str(VCL_ROOT)]
    try:
        revision = subprocess.run(
            [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot read the VCL revision. Initialize external/VCL and check Git is available.") from error
    if revision != VCL_REVISION:
        raise RuntimeError(f"Expected VCL revision {VCL_REVISION}, found {revision}")
    diff = subprocess.run(
        [*git, "diff", "--quiet", VCL_REVISION, "--", *(path.as_posix() for path in files)],
        check=False, capture_output=True,
    )
    if diff.returncode:
        raise RuntimeError("The VCL operator or its required files differ from the pinned revision")

    file_manager = _load_source(VCL_ROOT / files[0])
    converters = _load_source(VCL_ROOT / files[1], {"utils.file_manager": file_manager})
    operator = _load_source(
        VCL_ROOT / files[2], {"api.dqlEngine.executor.tools.converters": converters}
    ).estrai_semantico
    if Path(operator.__code__.co_filename).resolve() != (VCL_ROOT / files[2]).resolve():
        raise RuntimeError("The VCL operator was loaded from an unexpected path")
    return operator


def _context(documents):
    blocks = []
    for index, document in enumerate(documents, start=1):
        name = escape(document["name"], quote=True)
        kind = escape(document["type"], quote=True)
        blocks.append(
            f'<DOC source_ref="doc-{index}" source_name="{name}" source_type="{kind}">\n'
            f'{document["text"]}\n</DOC>'
        )
    return "Context:\n" + "\n\n".join(blocks)


def extract(task, documents, llm):
    """Run the original VCL function once and return its unmodified output."""
    if not documents:
        raise ValueError("VCL extraction requires at least one source document")
    target = TARGETS[task["dataset"]]
    description = task_instructions(task)
    prompt = (
        (
            ("system", "{instructions}"),
            ("user", "Extraction target\n{what}\n\n{description_what}\n\n{context}"),
        ),
        {"instructions": SYSTEM_INSTRUCTIONS},
        ("what", "description_what", "context", "how"),
        "str",
    )
    # VCL supplies Italian text for empty conditions. The templates render the task in English.
    language = SimpleNamespace(
        prompts={"it": {"EstraiSemantico.json": prompt}},
        get_description_from_what={target: description}.__getitem__,
    )
    query = {"command": "estrai semantico", "what": [{"name": target}], "how": {}}
    return load_operator()(_context(documents), query, llm, language)
