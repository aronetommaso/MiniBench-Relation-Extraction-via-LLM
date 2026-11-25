"""Build or validate the DocRED and AOE datasets for the thesis mini-benchmark."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from io import BytesIO
from itertools import combinations, cycle, islice
import json
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


DATA = Path(__file__).resolve().parent / "data"
DOCRED_REVISION = "7985b4e0371e6c61a756feb41b7b27becf71c666"
AOE_REVISION = "9ff4acb5478180b5edd5397c9209af8c81389333"
AOE_EVALUATOR_REVISION = "28a4d6b0937dbfc55a425392b6bb5c7f5789a8b4"
DOCRED_URL = f"https://huggingface.co/datasets/thunlp/docred/resolve/{DOCRED_REVISION}/data/"
AOE_URL = f"https://huggingface.co/datasets/tianyumyum/AOE/resolve/{AOE_REVISION}/"
DOCRED_FILES = {
    "dev.json.gz": "6ae4d7f5b0b9d2cbe74b9634ed43b35b7cb5b7c0dc3a16226dbe343139a4ae05",
    "rel_info.json.gz": "7ef27efff537ba89ae66f6f4e60e4908d4df3860a4c2819ea94fa5ed696bdc70",
}
AOE_FILES = {
    "all_AOE_tables.jsonl": (
        AOE_URL + "all_AOE_tables.jsonl",
        "5ec68ccc9abae3731cb7f65865d4be2817c78fa068e40f4d6234329eae792999",
    ),
    "documents.tar": (
        AOE_URL + "documents.tar",
        "b8627391eb13fdf88cea5629c0824f36c7eae114b29fb8cf3fefd8b35a9e98e5",
    ),
    "medical_tree.json": (
        f"https://huggingface.co/api/datasets/tianyumyum/AOE/tree/{AOE_REVISION}/medical?recursive=true",
        "244b41ecfef8be43737da68c1efa43ca4e1c2a8e5c9361ad56d9e1b390ed43cc",
    ),
}
DOCRED_GROUPS = ("complex", "single", "multi")
ACADEMIC_TASKS = ("academic_0", "academic_1")
AOE_KEY_COLUMNS = {
    "academic_0": ["Cited paper title", "Referencing paper title"],
    "academic_1": ["paper_name", "method"],
    "medical_0": ["source_doc"],
    **{f"financial_{number}": ["Filename"] for number in range(7)},
}
DEFAULTS = {
    "docred": {"seed": 42, "pilot_size": 5, "complex_size": 10, "single_size": 10, "multi_size": 10},
    "aoe": {"seed": 42, "pilot_size": 2, "test_size": 12, "max_tokens": 100_000},
}
OUTPUTS = {"docred": "relation_extraction", "aoe": "table_extraction"}
ESTIMATOR = "Sum of ceil(UTF-8 bytes / 4) per source document. Model-independent approximation."


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_oid(data):
    identity = hashlib.sha1(f"blob {len(data)}\0".encode("ascii"), usedforsecurity=False)
    identity.update(data)
    return identity.hexdigest()


def rank(seed, group, identifier):
    return sha256(f"{seed}:{group}:{identifier}".encode("utf-8")), identifier


def json_bytes(value, *, pretty=False):
    options = {"indent": 2} if pretty else {"separators": (",", ":")}
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, **options) + "\n").encode("utf-8")


def jsonl_bytes(records):
    return b"".join(json_bytes(record) for record in records)


def read_source(path, url, expected, download, digest=sha256):
    if path.is_file():
        data = path.read_bytes()
        if digest(data) == expected:
            return data
        problem = "checksum mismatch for"
    else:
        problem = "missing"
    if not download:
        raise ValueError(f"{problem} cached source {path.relative_to(DATA).as_posix()}")
    try:
        request = Request(url, headers={"User-Agent": "thesis-mini-benchmark/1.0"})
        with urlopen(request, timeout=45) as response:
            data = response.read()
    except OSError as error:
        raise ValueError(f"{problem} cached source {path.name} and download failed from {url}: {error}") from error
    if digest(data) != expected:
        raise ValueError(f"checksum mismatch for upstream source {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=path.parent) as temp:
        staged = Path(temp) / path.name
        staged.write_bytes(data)
        staged.replace(path)
    return data


def pinned_sources(dataset, sources, download):
    contents = {}
    metadata = {}
    for name, (url, checksum) in sources.items():
        contents[name] = read_source(DATA / "raw" / dataset / name, url, checksum, download)
        metadata[name] = {"url": url, "sha256": checksum, "bytes": len(contents[name])}
    return contents, metadata


def estimate(byte_count):
    return (byte_count + 3) // 4


def lengths(values):
    values = list(values)
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None, "total": 0}
    return {
        "min": min(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "max": max(values),
        "total": sum(values),
    }


def split_statistics(items, summarize):
    return {
        "all": summarize(items["pilot"] + items["test"]),
        **{split: summarize(group) for split, group in items.items()},
    }


def exclusions(by_reason):
    return {reason: {"count": len(ids), "ids": sorted(ids)} for reason, ids in sorted(by_reason.items())}


def docred_id(index):
    return f"docred-dev-{index:04d}"


def valid_index(index, size):
    return type(index) is int and 0 <= index < size


def docred_features(original, index, vocabulary):
    identifier = docred_id(index)
    sentences = original["sents"]
    entities = original["vertexSet"]
    labels = original["labels"]
    for cluster in entities:
        for mention in cluster:
            sentence = mention["sent_id"]
            start, end = mention["pos"]
            if not valid_index(sentence, len(sentences)) or not (
                type(start) is int and type(end) is int and 0 <= start < end <= len(sentences[sentence])
            ):
                raise ValueError(f"invalid mention span in {identifier}: {mention}")
    evidence_counts = Counter()
    for label in labels:
        if not all(valid_index(label[key], len(entities)) for key in ("h", "t")):
            raise ValueError(f"invalid relation entity index in {identifier}: {label}")
        if label["r"] not in vocabulary:
            raise ValueError(f"unknown relation identifier in {identifier}: {label['r']}")
        evidence = label["evidence"]
        if not isinstance(evidence, list) or any(not valid_index(e, len(sentences)) for e in evidence):
            raise ValueError(f"invalid evidence sentence index in {identifier}: {evidence}")
        evidence_counts[len(evidence)] += 1
    text = "\n".join(" ".join(sentence) for sentence in sentences).encode("utf-8")
    return {
        "id": identifier,
        "source_index": index,
        "document_sha256": sha256(text),
        "utf8_bytes": len(text),
        "estimated_tokens": estimate(len(text)),
        "source_token_count": sum(map(len, sentences)),
        "entities": len(entities),
        "mentions": sum(map(len, entities)),
        "relation_count": len(labels),
        "relation_types": sorted({label["r"] for label in labels}),
        "evidence_counts": dict(sorted(evidence_counts.items())),
    }


def docred_stats(items):
    evidence = Counter()
    for item in items:
        evidence.update(item["evidence_counts"])
    return {
        "instances": len(items),
        "source_documents": len(items),
        "input_utf8_bytes": lengths(item["utf8_bytes"] for item in items),
        "input_estimated_tokens": lengths(item["estimated_tokens"] for item in items),
        "input_source_tokens": lengths(item["source_token_count"] for item in items),
        "entities": sum(item["entities"] for item in items),
        "mentions": sum(item["mentions"] for item in items),
        "relations": sum(item["relation_count"] for item in items),
        "distinct_relation_types": len({relation for item in items for relation in item["relation_types"]}),
        "evidence_sentences_per_relation": dict(sorted(evidence.items())),
    }


def percentile_75(values):
    ordered = sorted(values)
    return ordered[(3 * len(ordered) + 3) // 4 - 1]


def build_docred(settings, download):
    files, sources = pinned_sources(
        "docred", {name: (DOCRED_URL + name, checksum) for name, checksum in DOCRED_FILES.items()}, download
    )
    original = json.loads(gzip.decompress(files["dev.json.gz"]))
    vocabulary = json.loads(gzip.decompress(files["rel_info.json.gz"]))
    if len(original) != 998 or len(vocabulary) != 96:
        raise ValueError("unexpected DocRED development split or relation vocabulary size")
    features = [docred_features(record, index, vocabulary) for index, record in enumerate(original)]
    entity_cutoff = percentile_75(item["entities"] for item in features)
    mention_cutoff = percentile_75(item["mentions"] for item in features)
    pools = defaultdict(list)
    for item in features:
        if item["entities"] >= entity_cutoff or item["mentions"] >= mention_cutoff:
            pools["complex"].append(item)
        counts = item["evidence_counts"]
        single = counts.get(1, 0)
        multi = sum(count for size, count in counts.items() if size >= 2)
        if single != multi:
            pools["single" if single > multi else "multi"].append(item)
    pools["pilot"] = features

    used = set()
    groups = {}
    available = {}
    for group in (*DOCRED_GROUPS, "pilot"):
        candidates = [item for item in pools[group] if item["id"] not in used]
        candidates.sort(key=lambda item: rank(settings["seed"], f"docred-{group}", item["id"]))
        count = settings[f"{group}_size"]
        available[group] = len(candidates)
        if len(candidates) < count:
            raise ValueError(f"DocRED {group} needs {count} documents, with {len(candidates)} available")
        groups[group] = [dict(item, group=group) for item in candidates[:count]]
        used.update(item["id"] for item in groups[group])
    items = {
        "pilot": groups["pilot"],
        "test": [item for group in DOCRED_GROUPS for item in groups[group]],
    }
    artifacts = {
        f"{split}.jsonl": jsonl_bytes(
            {"id": item["id"], "source_index": item["source_index"], **original[item["source_index"]]}
            for item in group
        )
        for split, group in items.items()
    }
    artifacts["relations.json"] = json_bytes(vocabulary, pretty=True)
    manifest = {
        "dataset": "docred",
        "revision": DOCRED_REVISION,
        "source_split": "dev",
        "sources": sources,
        "sampling": settings,
        "token_estimator": ESTIMATOR,
        "complexity_thresholds": {"entities": entity_cutoff, "mentions": mention_cutoff},
        "eligible_at_group_selection": available,
        "instances": items,
        "statistics": {
            **split_statistics(items, docred_stats),
            "by_group": {group: docred_stats(records) for group, records in groups.items()},
        },
        "excluded": exclusions({"not_selected": [item["id"] for item in features if item["id"] not in used]}),
    }
    return artifacts, manifest


def aoe_task(identifier, domain):
    if domain == "medical":
        return "medical_0"
    return f"{domain}_{identifier.rsplit('_', 2)[1]}"


def aoe_candidate(record, archive, medical, archive_oids, max_tokens):
    identifier = record["record_id"]
    domain = identifier.split("_", 1)[0]
    if domain not in ("academic", "financial", "medical"):
        return None, ["domain_out_of_scope"]
    if record["language"] != "en" or not identifier.endswith("_en"):
        return None, ["language_out_of_scope"]

    task = aoe_task(identifier, domain)
    columns = [column["name"] for column in record["table_schema"]["columns"]]
    keys = set(columns)
    answers = record["answers"]
    reasons = []
    if not columns or len(keys) != len(columns) or not set(AOE_KEY_COLUMNS.get(task, [])).issubset(keys):
        reasons.append("invalid_table_schema")
    if not answers or any(not isinstance(row, dict) or row.keys() != keys for row in answers):
        reasons.append("gold_columns_mismatch")

    names = [item.get("filename") for item in record["input_doc"]]
    if len(names) != record["doc_num"] or not names or len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name.endswith(".md") or "/" in name or "\\" in name for name in names
    ):
        return None, [*reasons, "invalid_document_references"]
    paths = [f"{domain}/{name}" for name in names]
    index = medical if domain == "medical" else archive.NameToInfo
    if any(path not in index for path in paths):
        return None, [*reasons, "missing_document"]
    sizes = [index[path]["size"] if domain == "medical" else index[path].file_size for path in paths]
    token_count = sum(map(estimate, sizes))
    if token_count > max_tokens:
        reasons.append("above_token_threshold")
    if domain == "financial" and token_count > 200_000:
        reasons.append("above_financial_short_limit")
    if reasons:
        return None, reasons

    documents = []
    for path, size in zip(paths, sizes):
        if domain == "medical":
            oid = medical[path]["oid"]
        else:
            if path not in archive_oids:
                content = archive.read(path)
                content.decode("utf-8")
                archive_oids[path] = git_blob_oid(content)
            oid = archive_oids[path]
        documents.append({"source_path": path, "oid": oid, "utf8_bytes": size, "estimated_tokens": estimate(size)})
    return {
        "id": identifier,
        "domain": domain,
        "task": task,
        "record": record,
        "documents": documents,
        "document_oids": {document["oid"] for document in documents},
        "utf8_bytes": sum(sizes),
        "estimated_tokens": token_count,
    }, []


def select_aoe(candidates, settings, has_medical):
    secondary_domain = "medical" if has_medical else "financial"
    secondary_task = "medical_0" if has_medical else "financial"
    test_tasks = (*ACADEMIC_TASKS, secondary_task)
    test_counts = Counter(islice(cycle(test_tasks), settings["test_size"]))
    test_pools = {
        task: sorted(
            (item for item in candidates if item["task"] == task or (task == "financial" and item["domain"] == task)),
            key=lambda item: rank(settings["seed"], "aoe-test", item["id"]),
        )
        for task in test_tasks
    }
    domains = ("academic", secondary_domain)
    pilot_counts = dict(zip(domains, ((settings["pilot_size"] + 1) // 2, settings["pilot_size"] // 2)))
    pilot_pools = {
        domain: sorted(
            (item for item in candidates if item["domain"] == domain),
            key=lambda item: (item["estimated_tokens"], rank(settings["seed"], "aoe-pilot", item["id"])),
        )
        for domain in domains
    }
    for task, pool in test_pools.items():
        if len(pool) < test_counts[task]:
            raise ValueError(f"AOE test task {task} needs {test_counts[task]} instances, with {len(pool)} eligible")
    for domain, pool in pilot_pools.items():
        reserved = sum(test_counts[task] for task in test_tasks if task.split("_")[0] == domain)
        if len(pool) < pilot_counts[domain] + reserved:
            raise ValueError(
                f"AOE {domain} needs {pilot_counts[domain]} pilot and {reserved} test instances, "
                f"with {len(pool)} eligible"
            )

    best = None
    for first in combinations(pilot_pools["academic"], pilot_counts["academic"]):
        for second in combinations(pilot_pools[secondary_domain], pilot_counts[secondary_domain]):
            pilot = [item for pair in zip(first, second) for item in pair] + list(first[len(second):])
            pilot_ids = {item["id"] for item in pilot}
            pilot_oids = {oid for item in pilot for oid in item["document_oids"]}
            test = []
            cost = 0
            for task, pool in test_pools.items():
                remaining = [item for item in pool if item["id"] not in pilot_ids]
                if len(remaining) < test_counts[task]:
                    break
                by_overlap = sorted(remaining, key=lambda item: len(pilot_oids & item["document_oids"]))
                chosen = by_overlap[:test_counts[task]]
                test.extend(chosen)
                cost += sum(len(pilot_oids & item["document_oids"]) for item in chosen)
            else:
                if cost == 0:
                    return {"pilot": pilot, "test": test}
                if best is None or cost < best[0]:
                    best = cost, {"pilot": pilot, "test": test}
    if best is not None:
        return best[1]
    raise ValueError("AOE has no distinct pilot/test assignment that satisfies the requested task quotas")


def aoe_stats(items):
    documents = [document for item in items for document in item["documents"]]
    rows = [row for item in items for row in item["answers"]]
    values = [value for row in rows for value in row.values()]
    blank = sum(isinstance(value, str) and not value.strip() for value in values)
    null = sum(value is None for value in values)
    return {
        "instances": len(items),
        "document_references": len(documents),
        "unique_source_paths": len({document["source_path"] for document in documents}),
        "unique_documents": len({document["path"] for document in documents}),
        "input_utf8_bytes": lengths(item["utf8_bytes"] for item in items),
        "input_estimated_tokens": lengths(item["estimated_tokens"] for item in items),
        "document_utf8_bytes": lengths(document["utf8_bytes"] for document in documents),
        "document_estimated_tokens": lengths(document["estimated_tokens"] for document in documents),
        "gold_rows": len(rows),
        "gold_cells": len(values),
        "missing_cells": blank + null,
        "blank_cells": blank,
        "null_cells": null,
        "literal_n_a_cells": sum(isinstance(value, str) and value.strip().casefold() == "n/a" for value in values),
    }


def read_medical_document(document, download):
    source = document["source_path"]
    path = DATA / "raw" / "aoe" / "medical" / (sha256(source.encode("utf-8")) + ".md")
    try:
        return read_source(path, AOE_URL + quote(source, safe="/"), document["oid"], download, git_blob_oid)
    except ValueError as error:
        raise ValueError(f"AOE source document {source}: {error}") from error


def build_aoe(settings, download):
    table_sources = {name: AOE_FILES[name] for name in ("all_AOE_tables.jsonl", "documents.tar")}
    files, sources = pinned_sources("aoe", table_sources, download)
    originals = [json.loads(line) for line in files["all_AOE_tables.jsonl"].splitlines()]
    ids = {item["record_id"] for item in originals}
    if len(ids) != len(originals):
        raise ValueError("duplicate record IDs in upstream AOE JSONL")
    has_medical = any(identifier.startswith("medical_") for identifier in ids)
    medical = {}
    if has_medical:
        tree, tree_sources = pinned_sources("aoe", {"medical_tree.json": AOE_FILES["medical_tree.json"]}, download)
        sources.update(tree_sources)
        medical = {item["path"]: item for item in json.loads(tree["medical_tree.json"]) if item["type"] == "file"}
    rejected = defaultdict(list)
    candidates = []
    with ZipFile(BytesIO(files["documents.tar"])) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate member paths in the AOE document archive")
        archive_oids = {}
        for original in originals:
            item, reasons = aoe_candidate(original, archive, medical, archive_oids, settings["max_tokens"])
            if item is not None:
                candidates.append(item)
            for reason in reasons:
                rejected[reason].append(original["record_id"])
        try:
            selected = select_aoe(candidates, settings, has_medical)
        except ValueError as error:
            eligible = dict(sorted(Counter(item["task"] for item in candidates).items()))
            excluded = {reason: len(group) for reason, group in sorted(rejected.items())}
            raise ValueError(f"{error}. Eligible by task {eligible}. Exclusion counts {excluded}") from error

        picked = selected["pilot"] + selected["test"]
        documents = {doc["source_path"]: doc for item in picked for doc in item["documents"]}
        medical_documents = [doc for path, doc in sorted(documents.items()) if path in medical]
        with ThreadPoolExecutor(max_workers=6) as pool:
            downloaded = {
                doc["source_path"]: content
                for doc, content in zip(medical_documents, pool.map(
                    lambda doc: read_medical_document(doc, download), medical_documents
                ))
            }

        artifacts = {}
        document_info = {}
        for source, document in sorted(documents.items()):
            content = downloaded[source] if source in downloaded else archive.read(source)
            content.decode("utf-8")
            if len(content) != document["utf8_bytes"] or git_blob_oid(content) != document["oid"]:
                raise ValueError(f"AOE document size or content mismatch for {source}")
            checksum = sha256(content)
            path = f"documents/{checksum}.md"
            artifacts[path] = content
            document_info[source] = {
                "source_path": source,
                "path": path,
                "sha256": checksum,
                "source_url": AOE_URL + quote(source, safe="/") if source in medical else AOE_URL + "documents.tar",
                "utf8_bytes": document["utf8_bytes"],
                "estimated_tokens": document["estimated_tokens"],
            }

    records = {}
    instances = {}
    for split, items in selected.items():
        records[split] = []
        instances[split] = []
        for item in items:
            source = item["record"]
            references = [document_info[doc["source_path"]] for doc in item["documents"]]
            metadata = {
                "record_id": item["id"],
                "domain": item["domain"],
                "task": item["task"],
                "utf8_bytes": item["utf8_bytes"],
                "estimated_tokens": item["estimated_tokens"],
            }
            row = {
                **metadata,
                "language": source["language"],
                "benchmark_name": source["benchmark_name"],
                "query": source["query"],
                "table_schema": source["table_schema"],
                "key_columns": AOE_KEY_COLUMNS[item["task"]],
                "answers": source["answers"],
                "documents": [
                    {key: doc[key] for key in ("source_path", "path", "utf8_bytes", "estimated_tokens")}
                    for doc in references
                ],
            }
            primary_keys = source.get("primary_keys", source["table_schema"].get("primary_keys"))
            if primary_keys is not None:
                row["primary_keys"] = primary_keys
            for name in ("ref_doc", "evidence", "evidence_annotations", "supporting_evidence"):
                if name in source:
                    row[name] = source[name]
            records[split].append(row)
            instances[split].append({**metadata, "source_documents": [doc["source_path"] for doc in references]})
        artifacts[f"{split}.jsonl"] = jsonl_bytes(records[split])

    tasks = sorted({item["task"] for item in picked})
    task_stats = {
        task: split_statistics(
            {split: [row for row in rows if row["task"] == task] for split, rows in records.items()}, aoe_stats
        )
        for task in tasks
    }

    split_paths = {
        split: {doc["source_path"] for row in rows for doc in row["documents"]}
        for split, rows in records.items()
    }
    split_hashes = {
        split: {document_info[path]["sha256"] for path in paths} for split, paths in split_paths.items()
    }
    overlap = sorted(split_hashes["pilot"] & split_hashes["test"])
    used = {item["id"] for item in picked}
    rejected["eligible_not_selected"] = [item["id"] for item in candidates if item["id"] not in used]
    manifest = {
        "dataset": "aoe",
        "revision": AOE_REVISION,
        "evaluator_revision": AOE_EVALUATOR_REVISION,
        "sources": sources,
        "sampling": settings,
        "token_estimator": ESTIMATOR,
        "instances": instances,
        "documents": list(document_info.values()),
        "overlap": {
            "source_paths": sorted(split_paths["pilot"] & split_paths["test"]),
            "content": [
                {
                    "sha256": checksum,
                    **{
                        split: sorted(path for path in paths if document_info[path]["sha256"] == checksum)
                        for split, paths in split_paths.items()
                    },
                }
                for checksum in overlap
            ],
        },
        "source_availability": {
            "records_by_domain": dict(sorted(Counter(identifier.split("_", 1)[0] for identifier in ids).items())),
            "eligible_by_task": dict(sorted(Counter(item["task"] for item in candidates).items())),
        },
        "statistics": {
            **split_statistics(records, aoe_stats),
            "by_task": task_stats,
        },
        "excluded": exclusions(rejected),
    }
    return artifacts, manifest


def validate_selection(manifest):
    pilot = manifest["instances"]["pilot"]
    test = manifest["instances"]["test"]
    id_key = "id" if manifest["dataset"] == "docred" else "record_id"
    for split, items in (("pilot", pilot), ("test", test)):
        if len({item[id_key] for item in items}) != len(items):
            raise ValueError(f"duplicate instance identifiers in the {split} split")
    if {item[id_key] for item in pilot} & {item[id_key] for item in test}:
        raise ValueError("instance identifiers overlap between pilot and test")
    requested = manifest["sampling"]
    if manifest["dataset"] == "aoe":
        test_size = requested["test_size"]
        if any(item["estimated_tokens"] > requested["max_tokens"] for item in pilot + test):
            raise ValueError("selected AOE instance exceeds the token threshold")
    else:
        test_size = sum(requested[f"{group}_size"] for group in DOCRED_GROUPS)
    if len(pilot) != requested["pilot_size"] or len(test) != test_size:
        raise ValueError("generated split sizes do not match the requested counts")


def validate_artifacts(root, expected):
    for path, content in expected.items():
        file = root / path
        if not file.is_file():
            raise ValueError(f"missing generated artifact: {path}")
        if file.read_bytes() != content:
            raise ValueError(f"generated artifact differs from the pinned sources and sampling settings: {path}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if extra := actual - expected.keys():
        raise ValueError(f"unexpected generated artifacts: {', '.join(sorted(extra))}")


def publish(output, artifacts):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-", dir=output.parent) as temp:
        parent = Path(temp)
        stage = parent / "new"
        previous = parent / "previous"
        for path, content in artifacts.items():
            file = stage / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
        validate_artifacts(stage, artifacts)
        if output.exists():
            output.rename(previous)
        try:
            stage.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=(*DEFAULTS, "all"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true", help="Validate cached sources and artifacts without writing.")
    for dataset, defaults in DEFAULTS.items():
        for name, default in defaults.items():
            if name != "seed":
                parser.add_argument(f"--{dataset}-{name.replace('_', '-')}", type=int, help=f"Build default is {default}.")
    return parser.parse_args()


def sampling_settings(dataset, args, output):
    defaults = DEFAULTS[dataset]
    settings = defaults.copy()
    if args.validate_only:
        stored = json.loads((output / "manifest.json").read_bytes())["sampling"]
        if stored.keys() != defaults.keys():
            raise ValueError("invalid sampling settings in manifest")
        settings.update(stored)
    for name in defaults:
        supplied = args.seed if name == "seed" else getattr(args, f"{dataset}_{name}")
        if supplied is not None:
            if args.validate_only and supplied != settings[name]:
                raise ValueError(f"validation option {name} differs from the saved sampling settings")
            settings[name] = supplied
        value = settings[name]
        if type(value) is not int or ((name.endswith("_size") and value < 0) or (name == "max_tokens" and value < 1)):
            raise ValueError(f"invalid sampling parameter {name}={value}")
    if not any(value > 0 for name, value in settings.items() if name.endswith("_size")):
        raise ValueError("at least one pilot or test instance must be requested")
    return settings


def main():
    args = options()
    builders = {"docred": build_docred, "aoe": build_aoe}
    datasets = tuple(builders) if args.dataset == "all" else (args.dataset,)
    failed = False
    for dataset in datasets:
        output = DATA / OUTPUTS[dataset]
        try:
            settings = sampling_settings(dataset, args, output)
            artifacts, manifest = builders[dataset](settings, not args.validate_only)
            validate_selection(manifest)
            artifacts["manifest.json"] = json_bytes(manifest, pretty=True)
            if args.validate_only:
                validate_artifacts(output, artifacts)
            else:
                publish(output, artifacts)
            stats = manifest["statistics"]
            print(
                f"{dataset}: {'validated' if args.validate_only else 'built and validated'} "
                f"{stats['pilot']['instances']} pilot and {stats['test']['instances']} test instances, "
                f"{stats['all']['input_estimated_tokens']['total']:,} estimated source tokens"
            )
            if dataset == "aoe":
                print(f"aoe: {len(manifest['overlap']['content'])} shared pilot/test document contents")
        except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
            failed = True
            print(f"{dataset}: {error}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
