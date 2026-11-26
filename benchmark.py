"""Run one extraction method against the local DocRED and AOE samples."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

from llm import (
    CHAT_INPUT_ESTIMATOR, DEFAULT_GEMINI_MODEL, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_ROUNDS, DEFAULT_MAX_TOTAL_INPUT_TOKENS, INPUT_ESTIMATOR, LLM, aggregate_usage, live_settings,
)
from methods import deepread_adapter, vcl_adapter
from prompts import DOCRED_QUERY, OUTPUT_KEYS, validate_prediction


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATASETS = {"docred": ("relation_extraction", "id"), "aoe": ("table_extraction", "record_id")}
MENTION_FIELDS = ("name", "pos", "sent_id", "type")
METHODS = {
    "vcl": (vcl_adapter.extract, vcl_adapter.VCL_REVISION, vcl_adapter.VCL_PROMPT_VERSION),
    "deepread": (deepread_adapter.extract, deepread_adapter.DEEPREAD_REVISION, deepread_adapter.DEEPREAD_PROMPT_VERSION),
}


def read_records(directory, split):
    with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_records(directory, id_key, instance, split, live):
    if split == "pilot":
        return [(split, record) for record in read_records(directory, split)]

    candidates = ("pilot",) if live else ("pilot", "test")
    for current_split in candidates:
        for record in read_records(directory, current_split):
            if record[id_key] == instance:
                return [(current_split, record)]
    if live:
        raise ValueError(f"Live runs require an explicit pilot ID. '{instance}' is not in the pilot split.")
    raise ValueError(f"Instance '{instance}' was not found in the pilot or test split.")


def prepare_input(dataset, record, directory):
    """Project only the allowed inference fields into the method interface."""
    if dataset == "docred":
        sentences = record["sents"]
        task = {
            "dataset": dataset,
            "query": DOCRED_QUERY,
            "sentence_count": len(sentences),
            "entities": [
                [{key: mention[key] for key in MENTION_FIELDS} for mention in cluster]
                for cluster in record["vertexSet"]
            ],
            "relations": json.loads((directory / "relations.json").read_bytes()),
        }
        text = "\n".join(f"[{index}] {' '.join(tokens)}" for index, tokens in enumerate(sentences))
        return task, [{"name": record["title"], "type": "text", "text": text}]

    task = {
        "dataset": dataset,
        "query": record["query"],
        "columns": [{key: column[key] for key in ("name", "about")} for column in record["table_schema"]["columns"]],
    }
    documents = []
    document_root = (directory / "documents").resolve()
    for reference in record["documents"]:
        path = (directory / reference["path"]).resolve()
        if not path.is_relative_to(document_root):
            raise ValueError("AOE document reference must resolve inside the dataset's documents directory.")
        documents.append({
            "name": PurePosixPath(reference["source_path"]).name,
            "type": "markdown",
            "text": path.read_bytes().decode("utf-8"),
        })
    return task, documents


def _unique_object(pairs):
    value = {}
    for key, element in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = element
    return value


def parse_prediction(task, raw):
    payload = raw
    if isinstance(raw, str):
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
            raw.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            payload = fenced[1]
    try:
        prediction = json.loads(payload, object_pairs_hook=_unique_object)
        # Python accepts non-finite numbers that are not valid JSON.
        json.dumps(prediction, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        return "invalid_json", None, {"message": str(error)}
    try:
        validate_prediction(task, prediction)
    except ValueError as error:
        return "invalid_schema", None, {"message": str(error)}
    return "success", prediction, None


def run_instance(args, directory, id_key, split, record, model, api_key):
    started = perf_counter()
    extract, revision, prompt_version = METHODS[args.method]
    deepread = args.method == "deepread"
    simulated = None if args.live else json.dumps({OUTPUT_KEYS[args.dataset]: []})
    llm = LLM(
        model=model,
        api_key=api_key,
        provider=args.provider,
        simulated_response=simulated,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_rounds=args.max_rounds,
        max_total_input_tokens=args.max_total_input_tokens,
    )
    raw = prediction = None
    error = None
    try:
        task, documents = prepare_input(args.dataset, record, directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status, error = "input_error", {"message": str(exc)}
    else:
        try:
            raw = extract(task, documents, llm)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
            status, error = "extraction_error", {"message": str(exc)}
        else:
            if deepread:
                status = llm.method_trace["termination_reason"]
                error = llm.method_trace.get("error")
                if status == "final_answer":
                    status = "completed" if args.live else "dry_run"
            elif len(llm.calls) != 1:
                status, error = "extraction_error", {"message": "Expected one VCL LLM invocation."}
            else:
                call = llm.calls[0]
                status, error = call["status"], call["error"]
            if status in {"completed", "dry_run"}:
                parsed_status, prediction, error = parse_prediction(task, raw)
                if parsed_status != "success" or status == "completed":
                    status = parsed_status

    call = llm.calls[-1] if llm.calls else {}
    result = {
        "instance_id": record[id_key],
        "dataset": args.dataset,
        "split": split,
        "method": args.method,
        "provider": args.provider,
        "model": model,
        "reported_model": call.get("reported_model"),
        "status": status,
        "prediction": prediction,
        "raw_response": raw,
        "usage": call.get("usage"),
        "llm_calls": len(llm.calls),
        "api_attempts": sum(entry["api_attempts"] for entry in llm.calls),
        "latency_seconds": perf_counter() - started,
        "error": error,
        "input_utf8_bytes": call.get("input_utf8_bytes"),
        "estimated_input_tokens": call.get("estimated_input_tokens"),
        "input_token_estimator": INPUT_ESTIMATOR,
        "would_exceed_input_limit": call.get("would_exceed_input_limit"),
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "method_revision": revision,
        "benchmark_revision": args.benchmark_revision,
        "benchmark_dirty": args.benchmark_dirty,
        "prompt_version": prompt_version,
        "calls": llm.calls,
    }
    if deepread:
        result.update(
            reported_model=next(
                (turn["reported_model"] for turn in reversed(llm.calls) if turn.get("reported_model")), None
            ),
            usage=aggregate_usage(llm.calls),
            input_utf8_bytes=sum(turn["input_utf8_bytes"] for turn in llm.calls if turn.get("input_accepted")),
            estimated_input_tokens=llm.accepted_input_tokens,
            input_token_estimator=CHAT_INPUT_ESTIMATOR,
            would_exceed_input_limit=any(turn["would_exceed_input_limit"] for turn in llm.calls),
            max_total_input_tokens=args.max_total_input_tokens,
            max_rounds=args.max_rounds,
            method_trace=llm.method_trace,
        )
    else:
        result["vcl_revision"] = revision
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def benchmark_git_state():
    git = ["git", "-C", str(ROOT)]
    try:
        revision = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        status = subprocess.run([*git, "status", "--porcelain"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision.stdout.strip(), bool(status.stdout.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(METHODS))
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--instance")
    selection.add_argument("--split", choices=["pilot"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Simulate the LLM call (default).")
    mode.add_argument("--live", action="store_true", help="Send one explicitly selected pilot to the provider.")
    parser.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    parser.add_argument("--model", help="Override the selected provider's model in live mode.")
    parser.add_argument("--max-input-tokens", type=positive_int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=positive_int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-rounds", type=positive_int, default=DEFAULT_MAX_ROUNDS, help="DeepRead round limit.")
    parser.add_argument(
        "--max-total-input-tokens", type=positive_int, default=DEFAULT_MAX_TOTAL_INPUT_TOKENS,
        help="Cumulative DeepRead input-token estimate limit.",
    )
    args = parser.parse_args(argv)

    if args.live and not args.instance:
        parser.error("--live requires one explicit pilot --instance.")
    if args.method == "deepread" and args.provider != "gemini":
        parser.error("DeepRead uses Gemini Chat Completions. Select --provider gemini.")
    directory, id_key = DATASETS[args.dataset]
    directory = DATA / directory
    try:
        records = select_records(directory, id_key, args.instance, args.split, args.live)
        if args.live:
            model, api_key = live_settings(ROOT / ".env", args.model, args.provider)
        else:
            model = args.model or (DEFAULT_GEMINI_MODEL if args.method == "deepread" else None)
            api_key = None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    args.benchmark_revision, args.benchmark_dirty = benchmark_git_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_mode = "live" if args.live else "dry-run"
    ok = True
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f"{args.dataset}-{args.method}-{run_mode}-",
        suffix=".jsonl", dir=RESULTS, delete=False,
    ) as stream:
        for split, record in records:
            result = run_instance(args, directory, id_key, split, record, model, api_key)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            estimate = result["estimated_input_tokens"]
            size = f" | {estimate:,} estimated input tokens" if estimate is not None else ""
            if result["would_exceed_input_limit"]:
                size += " | above input limit"
            print(f'{result["instance_id"]} | {result["status"]}{size} | {result["api_attempts"]} API attempts')
            if result["error"]:
                print(json.dumps(result["error"], ensure_ascii=True), file=sys.stderr)
            ok &= result["status"] in {"dry_run", "success"}
    print(f"Results saved to {stream.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
