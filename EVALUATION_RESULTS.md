# Evaluation results

Sections are dated and kept in the order they were produced. The **Streamlit
integration** section at the end is the current run; earlier sections are
preserved as historical evidence and were not re-labelled as rerun. The
machine-readable `evals/latest_*.json` artifacts now contain the most recent run.


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

## Streamlit integration validation — 2026-10-14

Everything below was executed on this date, after the final code change, on a
macOS workstation (Python 3.14.5 venv) and Docker. No live LLM was called; no real
resume or recruiting data was used.

### Test suites

| Layer | Command / scope | Result |
|---|---|---|
| Lint | `ruff check .` | passed |
| Unit + API contract + client + UI (no DB) | `pytest -q` | **128 passed, 2 skipped** (integration modules skip without `TEST_DATABASE_URL`) |
| Full suite with isolated pgvector | `TEST_DATABASE_URL=… pytest -q` (disposable `pgvector/pgvector:pg16` container) | **136 passed** |
| PostgreSQL integration only | `pytest -m integration -q` | **8 passed** (3 repository + 5 API/workflow) |
| Legacy runner | `python -m unittest discover -s tests` | 31 tests OK |
| Migrations | `alembic upgrade head` (0001→0003), `downgrade 0002` → `upgrade head`, `alembic check` with LangGraph tables present | passed; no drift |
| Dependency audit | `pip_audit --local` | no known vulnerabilities |
| Whitespace | `git diff --check` | clean |

What the new tests cover:

- **AI (offline, fake providers):** canonical citation ID across context/prompt/verifier/API; context budget including headers (4 budgets); quotes from truncated tails and budget- or limit-excluded chunks rejected; `chunk_key` used as ID rejected; whitespace/invented quotes rejected; identifier redaction before supply; workflow source round-trip without re-expansion; prompt-injection delimiting; schema-invalid and provider failures withheld with fixed messages.
- **Backend API (in-memory fakes):** fail-closed 401/403 on 8 candidate routes × 3 credential cases; reader cannot upload/delete; opaque filename persistence; idempotent upload; list/detail/versions tenant isolation and redaction; POST search filters and scores; generation gate (503 unconfigured, 403 policy-denied) before any provider call; 502/422 safe errors; workflow start → interrupt → reviewer-only review → 409 duplicate; cross-tenant workflow 404; review payload validation.
- **Client:** headers, POST search keeps queries out of URLs, retries only on safe reads, no retries on upload/delete/generation/review, timeout and unreachable messages, no credentials in errors, contract-violation detection, and a full client ↔ FastAPI round trip.
- **UI (Streamlit AppTest):** all 7 pages in Standalone mode; quick-link navigation; blind ranking results show no filenames; Connected-mode overview, library inspect/confirmed delete, empty states, search diagnostics and relevance disclaimer, Ask AI citations with expandable exact excerpts, generation-disabled state, brief → review → history, API-unavailable state, unknown-citation handling.
- **Ranking parity:** `tests/ranking_golden_v1.json` was generated by executing the **original** `app.analyze_resume` from commit `2b0f33c` on three synthetic resumes with a deterministic encoder; the refactored `services/ranking.py` reproduces every score, component, label, and order in blind and non-blind modes, and `DEFAULT_WEIGHTS` is unchanged.
- **Integration (real PostgreSQL 16 + pgvector + PostgresSaver, fake embedder/provider):** document lifecycle, opaque filename in the DB, versioning, current-version-only search, physical deletion of versions and chunks, tenant isolation, ingestion → retrieval → grounded answer, workflow interrupt → **all pooled checkpoint connections closed (simulated restart)** → resume → durable review, duplicate-review 409, hashed reviewer subject, and reconciliation of a lost review row from the checkpoint.

### Offline evaluations (rerun)

| Evaluation | Dataset | Result |
|---|---|---|
| Whole-document cosine baseline (`all-MiniLM-L6-v2`) | `synthetic-retrieval-v1`, 5 queries | Recall@5 1.0000 · MRR 0.9000 · nDCG@5 0.9262 · mean 163.335 ms · P95 670.103 ms |
| PostgreSQL FTS + pgvector + RRF (real MiniLM, disposable DB) | `synthetic-retrieval-v1`, 5 queries | Recall@5 1.0000 · MRR 0.9000 · nDCG@5 0.9262 · mean 35.764 ms · P95 63.171 ms |
| Citation verification | `generation-citation-v2`, **7 cases** (3 original + truncated tail, truncated head, `chunk_key` ID, redacted identifier) | accuracy 1.0000 · mean 0.007 ms · cost `null` |
| Deterministic offline quality | `offline-quality-v1` | parser fidelity, redaction recall, score parity, structured-output accuracy 1.0 · sensitive acceptance 0.0 · 43.018 ms · 0 live calls |
| Quality gate | `python -m evals.quality_gate` | passed |

Relevance metrics are identical to the earlier run; latencies differ because they
include process warm-up on a workstation and are not load measurements.

### Docker Compose and smoke tests

Run as an isolated Compose project (`-p rr-validate`, separate volumes and ports,
test-only credentials) so no existing local stack was touched:

- `docker compose config -q`: passed. `docker compose up --build --wait`: postgres, api, and **streamlit** all healthy.
- API image size (`docker image inspect .Size`): 573,705,554 bytes (previous 573,474,101); `torch 2.14.1+cpu`, CUDA unavailable.
- `scripts/smoke_api.py` against the container: health, auth and tenant denial, session (`generation=unconfigured`), ingestion + idempotency, list/detail with redaction, POST and legacy GET search, **generation gate fails closed (503) for Q&A and workflow start**, workflow history, reader-role delete denied, physical deletion — passed.
- `/_stcore/health` and `/v1/health`: ok.
- `scripts/smoke_streamlit.py` **inside the Streamlit container** (API reached at `http://api:8000`): V2 Connected mode, 7 pages rendered, live hybrid search (real MiniLM + pgvector) rendered the seeded synthetic document, document deleted — passed. Also passed from the host against the published API port.
- Container logs (15 lines) contained no bearer tokens, search queries, document text, synthetic email, upload filenames, or request lines.
- The built image contains no `.env` or `.streamlit/secrets.toml`.

### Not verified

- **Live Gemini generation** through LangChain (structured output, latency, cost, quality) — no live call was made; Ask AI and Recruiter Briefs were exercised only with fake providers and with the fail-closed gate in Docker.
- Cross-encoder reranking relevance and latency (the toggle and plumbing are tested with a fake model; the real model was not benchmarked).
- Visual appearance in a browser (AppTest validates element trees, not rendering); screenshots were not captured.
- Concurrent multi-user load, large batches, real PDF variety, OCR, and fairness.

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
