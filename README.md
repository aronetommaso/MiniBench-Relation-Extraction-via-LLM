# Thesis mini-benchmark for information extraction

Milestone 1 builds two datasets for inspecting extraction errors on individual examples. DocRED supports document-level relation extraction. AOE supports structured table extraction from multiple Markdown documents. The builder uses Python 3.11 or later and the standard library. Dataset construction and validation require zero paid API calls.

## Build and validate

Run these commands from the repository root.

~~~sh
python build_dataset.py --dataset all --seed 42
python build_dataset.py --dataset all --validate-only
~~~

Use `--dataset docred` or `--dataset aoe` to build one dataset. Run `python build_dataset.py --help` for the sample-size flags and `--aoe-max-tokens` (default 100,000). The initial build downloads missing pinned sources. Subsequent builds reuse verified files in `data/raw/`. Validation reads the sampling settings saved in each manifest, rebuilds the expected outputs in memory, and compares every artifact byte for byte. It uses only the local cache and exits with status 1 on error. A failed build for one dataset preserves its previously valid output.

## Source releases

- **DocRED** uses the human-annotated development split, with 998 documents and 96 relation types. The compressed files at Hugging Face revision `7985b4e0371e6c61a756feb41b7b27becf71c666` decompress to the same bytes as the files linked from the authors' repository. [Official DocRED repository](https://github.com/thunlp/DocRED/tree/master/data) · [Pinned development data](https://huggingface.co/datasets/thunlp/docred/tree/7985b4e0371e6c61a756feb41b7b27becf71c666/data)
- **AOE** uses the root `all_AOE_tables.jsonl` at revision `9ff4acb5478180b5edd5397c9209af8c81389333`. The root file contains 447 records across four domains. The dataset card points to a separate 373-record file that omits Medical. The release's `documents.tar` is a ZIP with Academic and Financial Markdown. Medical Markdown is provided under `medical/`. The pinned release contains document references and gold tables, with no dedicated evidence annotations. The evaluator's alignment columns are taken from revision `28a4d6b0937dbfc55a425392b6bb5c7f5789a8b4`. [Pinned AOE data](https://huggingface.co/datasets/tianyumyum/AOE/tree/9ff4acb5478180b5edd5397c9209af8c81389333) · [Pinned evaluator configuration](https://github.com/tianyumyum/AOE/blob/28a4d6b0937dbfc55a425392b6bb5c7f5789a8b4/config/task_config.yaml)

## Sampling and token estimates

Both datasets rank seeded candidates using the SHA-256 digest of `seed:group:id`, followed by the original ID to break ties. The default seed is 42.

- **DocRED.** First select 10 test documents with at least 24 entity clusters or 32 mentions. These cutoffs are the nearest-rank 75th percentiles of the full split. From the remaining documents select 10 with more single-sentence than multi-sentence evidence annotations, followed by 10 with the reverse distribution. The dominance calculation uses relations with evidence. All original evidence lists, including empty lists, stay in the gold. Sample 5 pilot documents from the remainder. Evidence sentence count serves as a sampling proxy for the distribution of supporting evidence.
- **AOE.** Match the original `input_doc` filenames to Markdown in upstream order. Eligible records have English language metadata, a complete set of source Markdown, consistent schemas and gold rows, and a total estimate within the cap. The test quotas are 4 Citation Context, 4 Methodology Performance, and 4 Medical. The pilot has one Academic and one Medical instance. Candidate pilots are ordered by estimated length within each domain and then by seeded rank. The builder tries Academic candidates first and Medical candidates second until it finds pilot choices that let it fill the test quotas with zero shared content. If a custom size makes overlap unavoidable, it minimizes the summed number of shared document contents across test instances and records the matching source paths and hashes. Custom test sizes allocate slots cyclically to the three tasks and pilot sizes alternate domains starting with Academic. The fallback for a release without Medical uses Academic and eligible English Financial within the configured cap and a 200,000-token estimated Fin-S ceiling. An unattainable quota produces a diagnostic with eligibility and exclusion counts.

The token estimate is `ceil(UTF-8 bytes / 4)` per source document, summed across all referenced documents for each instance. This is a model-independent approximation and can differ from the count produced by an LLM tokenizer. AOE uses the original Markdown bytes. For DocRED, the byte count uses the source token sequence, with a space between tokens and a newline between sentences. The original DocRED token count is recorded separately. The estimates cover source text and exclude prompt overhead.

## Dataset files and prompt allowlist

~~~text
data/
  raw/
    docred/                     # Pinned development split and vocabulary
    aoe/                        # Pinned tables, document ZIP, Medical index and selected Medical files
  relation_extraction/
    pilot.jsonl
    test.jsonl
    relations.json
    manifest.json
  table_extraction/
    pilot.jsonl
    test.jsonl
    manifest.json
    documents/                  # Original Markdown bytes keyed by SHA-256
~~~

DocRED JSONL records retain the original `title`, `sents`, `vertexSet`, and `labels`, plus an ID derived from the zero-based source index. AOE records retain their upstream IDs, task, query, table schema, original gold rows, and ordered Markdown references. Any primary keys explicitly declared by the source remain in the record. The evaluator's alignment column names are stored separately as `key_columns`. AOE reference paths are relative to `data/table_extraction/`. Distinct source paths can refer to the same stored Markdown when their contents are identical.

Future extraction methods must construct prompts from the following allowlist. Dataset records contain evaluation fields, so prompts must use the allowlist instead of serializing an entire record.

| Dataset | Allowed in extraction prompts | Evaluation and audit only |
| --- | --- | --- |
| DocRED | `title`, `sents`, entity clusters and mentions in `vertexSet` including entity types, and the complete vocabulary in `relations.json` | `labels` including relation triples and gold evidence, and source IDs, split, strata, lengths, hashes, and other sampling metadata |
| AOE | `query`, ordered source Markdown and original source names, and the `name` and `about` of the original schema columns | `answers`, any upstream gold evidence, evaluator alignment metadata such as `key_columns` and `valid_type`, and record IDs, domain and task labels, split, lengths, hashes, and other sampling metadata |

Manifests record pinned revisions, verified source hashes, selected IDs, group or task membership, exclusions, overlap, and length statistics by split and task or sampling group. AOE counts JSON nulls and blank or whitespace-only strings as missing cells. It counts literal `N/A` separately and leaves every gold value intact.

## Default build results

The seed-42 build covers 49 instances and **524,912 estimated source tokens**. Length statistics below use estimated tokens per instance.

| Dataset | Split | Instances | Min | Mean | Median | Max | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DocRED | Pilot | 5 | 188 | 249.600 | 259 | 286 | 1,248 |
| DocRED | Test | 30 | 173 | 297.533 | 288 | 491 | 8,926 |
| AOE | Pilot | 2 | 10,441 | 14,978.500 | 14,978.5 | 19,516 | 29,957 |
| AOE | Test | 12 | 18,649 | 40,398.417 | 36,045 | 72,453 | 484,781 |

| DocRED group | Instances | Entities | Relations | Distinct types | Estimated tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 5 | 105 | 48 | 17 | 1,248 |
| High complexity | 10 | 256 | 86 | 29 | 3,612 |
| Single-sentence majority | 10 | 196 | 87 | 29 | 2,593 |
| Multi-sentence majority | 10 | 196 | 152 | 44 | 2,721 |

The DocRED sample contains 753 entity clusters, 1,056 mentions, 373 relations, and 67 distinct relation types. The vocabulary retains all 96 types. Across the sample, 6 relations have zero evidence sentences, 197 have one, 130 have two, 26 have three, 11 have four, and 3 have five.

| AOE split and task | Instances | Source paths | Rows | Cells | Missing | Estimated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pilot · Methodology Performance | 1 | 2 | 2 | 8 | 2 | 19,516 |
| Pilot · Medical | 1 | 3 | 3 | 27 | 0 | 10,441 |
| Test · Citation Context | 4 | 12 | 9 | 36 | 0 | 185,094 |
| Test · Methodology Performance | 4 | 11 | 11 | 44 | 8 | 183,470 |
| Test · Medical | 4 | 20 | 20 | 205 | 1 | 116,217 |

AOE contains 48 source paths and 47 distinct Markdown contents, with zero shared document contents between pilot and test. There are 45 rows, 320 cells, 11 blank cells, zero null cells, and 2 literal `N/A` values. Of the 19 English Financial records, all exceed the default cap and 12 also have gold column names that disagree with the released schema. The exclusions are recorded by reason in the manifest, with individual records allowed to have multiple reasons.

The default build was validated from the local cache. Two builds at seed 42 produced identical SHA-256 hashes for all 54 processed files. Checks with custom sample sizes passed. An insufficient AOE threshold, a missing generated Markdown document, and a missing cached source each produced the expected diagnostic.

## Milestone 2 — VCL FullDocument extraction

The FullDocument baseline calls VCL's original `estrai_semantico(context, query, llm, language)` from the `parallelization` submodule at commit `6b8539edaf55e621f6e551c58e32a065a28da4cf`. The adapter loads the operator, its converters, and the file handler. The benchmark supplies the extraction target and the complete source context directly to the operator. Planning, retrieval, Executor orchestration, per-document calls, and Integra are outside the experimental boundary. A multi-document AOE instance uses one call for all its documents in their original order. [Pinned VCL source](https://github.com/unimib-datAI/VCL/tree/6b8539edaf55e621f6e551c58e32a065a28da4cf)

The shared interface is `extract(task, documents, llm)`. The runner projects dataset records into the inputs listed above. `prompts.py` defines the English task instructions and JSON output contracts for use by later methods. The VCL adapter wraps the source text in VCL-style document blocks and passes the result through the operator's preformatted-string interface. This preserves original casing, scientific notation, and Markdown, which the original list formatter lowercases. The English templates preserve the original prompt's emphasis on complete, objective, source-grounded extraction and request JSON suited to the two datasets. They use the original operator's target and context state.

### Install and run offline

Use Python 3.11 and Git with the pinned submodules initialized. The requirements pin the OpenAI SDK for live requests, python-dotenv for local configuration, spaCy for the original file handler's import, and pytest for the offline suite. The extraction path leaves the Italian spaCy model unloaded.

~~~powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest tests/ -q
& .\.venv\Scripts\python.exe .\benchmark.py --method vcl --dataset docred --instance docred-dev-0244 --dry-run
& .\.venv\Scripts\python.exe .\benchmark.py --method vcl --dataset aoe --instance academic_PMD_1_en --dry-run
~~~

Use `--split pilot` in place of `--instance` to simulate a whole pilot split. Dry runs invoke the actual VCL operator with a simulated LLM, use zero API requests, and read no credentials. Each run writes a JSONL file under the ignored `results/` directory. The `calls[].messages` field contains the complete rendered system and user messages. The result also stores the instance and split, prompt version and VCL commit, requested and reported model, raw output, structural validation status, errors, VCL invocation and API attempt counts, timing, and provider usage when available. The empty predictions in dry runs are simulator fixtures. Incomplete responses retain their available raw text, and failed predictions are recorded explicitly.

### Manual Gemini smoke test

Set `GEMINI_KEY` in the local `.env` or process environment. The runner uses `gemini-3.1-flash-lite` by default and sends the request to Google's Gemini Developer API. Google lists both Flash-Lite models below on the Free Tier. Use an API key from a Free Tier project. Each `--live` command allows one explicit pilot instance and one provider attempt. [Gemini pricing](https://aistudio.google.com/docs/pricing) · [Gemini compatibility API](https://aistudio.google.com/docs/openai)

~~~powershell
Set-Location -LiteralPath 'C:\Users\tomma\Desktop\magistrale\Secondo Anno\MiniBench_Relation_Extraction'
& .\.venv\Scripts\python.exe .\benchmark.py --method vcl --dataset docred --instance docred-dev-0244 --live --max-input-tokens 120000 --max-output-tokens 4096
~~~

If 3.1 is unavailable for your project, rerun the command with `--model gemini-3.5-flash-lite`. You can also set `GEMINI_MODEL` in `.env`. To opt back into OpenAI, pass `--provider openai` and set `OPENAI_MODEL` plus either `OPENAI_API_KEY` or `OPEN_AI_KEY`.

The default limits are 120,000 estimated input tokens and 4,096 output tokens. Before sending a live request, the runner prints the model, rendered UTF-8 size, input estimate, and output limit. It estimates input tokens using `ceil(UTF-8 bytes / 4)` over the roles and content of every rendered message. The estimate includes instructions, schema, and documents, and can differ from the provider's tokenizer count. Oversized prompts are preserved in the local trace and rejected before an API attempt. The client sends the complete prompt with SDK retries disabled. Actual usage and the local estimate appear in separate fields. Gemini uses the provider's sampling defaults. The OpenAI Responses path sends temperature zero for the supported configurations listed in the client. [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)

## Milestone 3 — DeepRead-BM25

DeepRead uses the original `run_agent()`, `build_system_prompt()`, `DocIndex`, `parse_markdown_to_corpus()`, `bm25_search()`, and `read_section()` at commit `25360c9f109cdcd29785bbef7e6a36963b50b374`. It starts from a directory of nodes and reads evidence through local tools. VCL sends the full source context in a single call. Both methods use the same inference allowlist, task contracts, and output validation. The shared prompt is now `structured-extraction-en-v2`. It describes DocRED relation fields without a concrete example triple. The earlier VCL v1 runs remain useful as technical smoke tests. Use v2 for comparisons. [Pinned DeepRead source](https://github.com/Zhanli-Li/DeepRead/tree/25360c9f109cdcd29785bbef7e6a36963b50b374) · [DeepRead paper](https://arxiv.org/abs/2602.05014)

The two tools exposed to Gemini are `bm25_search` with at most two hits and `read_section` with an inclusive start and exclusive end. DeepRead uses `(doc_id, node_id)` coordinates and keeps paragraph order. DocRED has one node with one numbered sentence per paragraph. AOE retains original Markdown titles, section hierarchy, tables, and source filenames under ordered document IDs. The original parser normalizes heading levels and paragraph boundaries. Its image branch ordinarily keeps the alternative text and loses surrounding inline text. The adapter instead preserves recognized image lines as literal text, including captions and links, without fetching images. The BM25 configuration disables neighbor expansion and uses lexical retrieval in place of the embedding and reranking setup from the paper.

The shared client sends DeepRead messages and function schemas through Gemini Chat Completions with the model's sampling defaults. It preserves tool IDs, parallel call order, and Gemini thought signatures from the provider's response. DeepRead supports Gemini in this milestone. VCL also supports OpenAI Responses. Input estimates use UTF-8 bytes divided by four and rounded up. For DeepRead this covers the complete serialized payload in every round, including prior messages, tool arguments and results, and tool schemas. Defaults are 120,000 per-request input tokens, 250,000 cumulative input tokens, 4,096 output tokens per round, and 10 rounds. Budget violations and terminal provider errors stop the loop. Requests use zero automatic retries. Provider usage stays separate from estimates and missing usage stays unknown. [Gemini function calling compatibility](https://aistudio.google.com/docs/openai#function-calling) · [Thought signatures](https://aistudio.google.com/docs/thought-signatures?generate-content=true)

The JSONL contains immutable request snapshots, per-call usage, the aggregate instance fields, and `method_trace` with the directory, source map, upstream events, tool results, and termination reason. It records both the method revision and the benchmark Git revision, plus whether the working tree is dirty. In the offline DocRED example, directory node `(1, 1)` is **Charles Quef** with six paragraphs. Round 1 finds paragraphs 0 and 1 via BM25. Round 2 reads both with ReadSection. Round 3 returns the simulated JSON. In AOE, the two documents have 17 and 12 nodes. The simulator searches each document in round 1 and reads within both in round 2. All seven pilot instances complete in three simulated rounds with zero API attempts. These traces verify the execution path. Their placeholder predictions are for testing.

DeepRead's `deepread-bm25-v2` prompt asks for one document ID and one node ID per ReadSection call. After a lookup error, the adapter returns the original error with the exact valid node IDs for that document, or the valid document IDs if the document is unknown. The shared output parser accepts plain JSON or a single Markdown code block containing JSON. It keeps the original response in `raw_response` and applies the same JSON and schema checks to both formats.

### Offline verification

From the project root, run the tests and the complete pilot splits. NumPy and Requests are direct dependencies of the DeepRead modules and are pinned in `requirements.txt`. DeepRead uses its local whitespace token-count fallback for the directory.

~~~powershell
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest tests/ -q
& .\.venv\Scripts\python.exe .\benchmark.py --method deepread --dataset docred --split pilot --dry-run
& .\.venv\Scripts\python.exe .\benchmark.py --method deepread --dataset aoe --split pilot --dry-run
~~~

For a representative dry run, replace `--split pilot` with `--instance docred-dev-0244` or `--instance academic_PMD_1_en`, respectively.

### Manual Gemini smoke tests

Use `GEMINI_KEY` from the existing environment or `.env`. Each live run requires a single explicit pilot ID. To check the shorter input first, run DocRED before AOE.

~~~powershell
& .\.venv\Scripts\python.exe .\benchmark.py --method deepread --dataset docred --instance docred-dev-0244 --provider gemini --model gemini-3.1-flash-lite --max-rounds 10 --live
& .\.venv\Scripts\python.exe .\benchmark.py --method deepread --dataset aoe --instance academic_PMD_1_en --provider gemini --model gemini-3.1-flash-lite --max-rounds 10 --live
~~~

## Method repositories inspected

DeepRead publishes an official Markdown entry point and BM25 retrieval. MemAgent publishes a quickstart and its original RL checkpoints. Using a shared LLM later will reproduce the MemAgent workflow with that backbone. Reproducing the original RL experiment requires the original checkpoints. [DeepRead](https://github.com/Zhanli-Li/DeepRead) · [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)
