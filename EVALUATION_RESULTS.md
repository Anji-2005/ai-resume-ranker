# Retrieval evaluation results

## Executed baseline — 2026-10-14

Command:

```bash
../ai-resume-ranker-venv/bin/python -m evals.run_whole_document_baseline \
  --dataset data/synthetic_retrieval.json \
  --output evals/latest_whole_document_baseline.json
```

| Method | Queries | Recall@5 | MRR | nDCG@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Original whole-document cosine matcher (`all-MiniLM-L6-v2`) | 5 | 1.0000 | 0.9000 | 0.9262 | 62.207 ms | 219.961 ms |

The machine-readable run output is in
[`evals/latest_whole_document_baseline.json`](evals/latest_whole_document_baseline.json).

## Executed PostgreSQL hybrid retrieval — 2026-10-14

The same five labelled cases were ingested through the parser and evaluated
against real PostgreSQL 16, pgvector HNSW, native `tsvector` ranking, and RRF.

| Method | Queries | Recall@5 | MRR | nDCG@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| PostgreSQL FTS + pgvector + RRF | 5 | 1.0000 | 0.9000 | 0.9262 | 26.270 ms | 45.873 ms |

These timings are single-process workstation measurements after model/package
availability; they are not load-test or statistically significant results.

## Executed deterministic offline quality evaluation — 2026-10-14

Dataset: `data/offline_quality_v1.json` (`offline-quality-v1`).

| Parser fidelity | Redaction recall | Score parity | Structured-output accuracy | Sensitive-output acceptance | Total latency |
|---:|---:|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 19.594 ms |

This is a small synthetic deterministic regression set: two parser cases, four
identifier checks, two score snapshots, and two mocked output-validation cases.
It made zero live-model calls and does not estimate population-level redaction
recall or fairness.

## Executed offline citation-verification evaluation — 2026-10-14

Command:

```bash
../ai-resume-ranker-venv/bin/python -m evals.run_generation_eval \
  --dataset data/generation_eval.json \
  --output evals/latest_generation_eval.json
```

| Method | Cases | Accuracy | Mean latency | Provider cost |
|---|---:|---:|---:|---:|
| Local exact-citation verification | 3 | 1.0000 | 0.006 ms | unavailable (`null`) |

This result measures deterministic citation verification only. It makes no
claim about Gemini/LangChain generation quality, usefulness, fairness, or live
provider latency/cost. `python -m evals.quality_gate` passed on the two saved
artifacts using the documented synthetic thresholds (baseline Recall@5 ≥ 1.0,
MRR and nDCG@5 ≥ 0.8; citation accuracy and deterministic quality metrics =
1.0; sensitive-output acceptance = 0.0). Thresholds remain subject to owner
approval before any controlled pilot.

## Test, container, and security validation — 2026-10-14

- Full suite against disposable `resume_intelligence_test`: **54 passed**.
- Alembic schema consistency: no new upgrade operations detected.
- Docker API image: **573,474,101 bytes**; `torch 2.14.1+cpu`, CUDA unavailable,
  `torch.version.cuda=None`.
- PostgreSQL and API Compose services: healthy.
- Synthetic HTTP smoke: authorization denial, tenant denial, DOCX ingestion,
  embedding/indexing, retrieval, missing-LLM-key error, protected deletion, and
  post-deletion retrieval all passed.
- Container access logging is disabled; post-smoke logs contained startup,
  dependency, and model-loading messages but no request query strings or tokens.
- `pip-audit --local`: no known vulnerabilities after constraining
  `urllib3>=2.8,<3`.
- CycloneDX 1.6 SBOM generation: 22 direct requirement components. Range-based
  entries remain unpinned and therefore do not provide a fully locked build.

## Important limitations

- This is a tiny, labelled synthetic dataset designed as a regression harness,
  not evidence of production relevance, fairness, or latency.
- The first result latency includes ordinary local process/model effects; it is
  not a load-test result.
- Cross-encoder reranking was not benchmarked. It is intentionally optional and
  should be measured separately for relevance, cost, and latency on approved
  evaluation data before enablement.
- No live Gemini call, model-priced token count, concurrency/load test, or
  security penetration test was run for this report. The trace cost field is
  intentionally unavailable unless a deployment supplies validated provider
  rates; its character-to-token conversion is only an estimate.
- Live Gemini generation was not invoked, so provider availability, quality,
  latency, and billed cost remain unverified. Generation validation results are
  mock/deterministic checks only.
