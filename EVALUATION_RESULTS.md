# Evaluation results

Sections are dated and kept in the order they were produced. The **V3 AI
Investigation validation** section at the end is the current run; earlier sections are
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

## V3 AI Investigation validation — 2026-10-15

This section is the current release-readiness run. It used only fictional data,
the disposable `resume_intelligence_test` database, and deterministic fake
providers. No Gemini request or real candidate record was used.

### Executed checks

| Check | Actual result |
|---|---|
| Complete `pytest -q` with `TEST_DATABASE_URL` | **169 passed**, 84 warnings, 7.46 s |
| PostgreSQL-only `pytest -m integration -q` | **9 passed**, 160 deselected, 5 warnings, 2.03 s |
| Focused agent/API/eval regression tests after recovery fix | **48 passed**, 5 warnings, 2.27 s |
| Ruff / Alembic drift / `git diff --check` | passed / no new operations / clean |
| Dependency audit | `pip-audit --local`: no known vulnerabilities |
| SBOM | CycloneDX generated successfully in `/tmp`; 18 range-based requirements were reported without pinned versions |
| Compose build and health | cached CPU dependency layers; PostgreSQL, API, and Streamlit healthy |
| Container runtime | pgvector **0.8.7**; migration **0004_investigations**; `torch 2.14.1+cpu`, CUDA `False`, `torch.version.cuda=None` |
| API image size | **573,879,333 bytes** |
| Synthetic API smoke | health, auth/tenant denial, ingestion/idempotency, redacted detail, hybrid retrieval, Q&A/workflow/investigation fail-closed generation gate, both histories, deletion passed |
| Connected Streamlit smoke | all **8 pages** rendered, live hybrid search rendered a synthetic record, cleanup passed |

The 84 full-suite warnings are not failures: 78 are Starlette TestClient timeout
deprecations and the remainder are PyMuPDF/Google dependency deprecations. They
remain maintenance work for future dependency updates.

### Investigation-agent benchmark

Command: `python -m evals.run_investigation_eval`. Dataset:
`data/investigation_eval_v1.json` (`investigation-agent-v1`). The evaluator runs
the production LangGraph and validation code with an in-memory repository,
labelled synthetic chunks, and a scripted fake tool-calling provider.

| Metric | Measured result |
|---|---:|
| Scenarios | 12 |
| Task completion | 1.0000 (7/7 task scenarios reached human review) |
| Expected scenario outcome | 1.0000 (12/12) |
| Tool-selection correctness | 1.0000 (10/10 scorable expected sequences) |
| Exact citation validity | 1.0000 (11/11 citations) |
| Unsupported-claim rejection | 1.0000 (1/1 hallucinated quote withheld) |
| Authorization violations | 0 (1/1 unauthorized attempt blocked) |
| Tool calls per investigation | mean 1.75; min 0; max 3 |
| Local orchestration latency | mean 1.892 ms; P95 3.562 ms |
| Mock-reported accounting cost | USD 0.022 total (not billed cost) |
| Live provider calls / cost | 0 / unavailable |

The cases include straightforward and implicit evidence, missing and
contradictory evidence, a two-search adaptive follow-up, multi-document scope,
prompt injection text, an unauthorized document, planning-budget exhaustion,
an invented quote, invalid arguments, and an unavailable tool. The quality gate
requires all correctness/blocking rates above to remain 1.0 and authorization
violations to remain zero.

Other evaluations rerun in the same validation sequence:

| Evaluation | Actual result |
|---|---|
| Whole-document MiniLM baseline | Recall@5 1.0000; MRR 0.9000; nDCG@5 0.9262; mean 82.547 ms; P95 294.993 ms |
| PostgreSQL pgvector + native FTS + RRF | Recall@5 1.0000; MRR 0.9000; nDCG@5 0.9262; mean 26.301 ms; P95 46.630 ms |
| Citation verification (7 cases) | accuracy 1.0000; mean 0.002 ms; provider cost unavailable |
| Offline parser/redaction/score/schema suite | all correctness metrics 1.0000; sensitive acceptance 0.0000; 31.185 ms; zero live calls |
| Combined synthetic quality gate | passed |

Post-smoke container logs contained startup/model-loading metadata only: no
bearer token, synthetic email, upload filename, document passage, or query was
present. Access logging is disabled. This observation is a bounded log review,
not a penetration test or proof that every future dependency log is safe.

### Release readiness

- **Portfolio demonstration:** ready with fictional documents and the documented
  fail-closed configuration. The complete local suite, migrations, evaluations,
  container build, HTTP smoke, and eight-page Streamlit smoke passed.
- **Controlled pilot:** blocked pending an approved identity provider/RBAC,
  a retention and purge policy for historical reports/checkpoints, live-provider
  evaluation on approved non-sensitive cases, and recovery/operations procedures
  for runs interrupted during synchronous planning.
- **Production:** not ready. In addition to pilot blockers, it needs locked and
  reviewed dependencies/model artifacts, load/concurrency and disaster-recovery
  tests, broader parser/redaction/fairness evaluation, malware/OCR isolation
  decisions, monitoring/incident response, and a security assessment.

Known measurement limits: the investigation provider is scripted; its labelled
pipeline does not measure retrieval relevance; orchestration timings exclude
PostgreSQL, network, provider, and load latency; mock costs only exercise budget
accounting. Live Gemini tool choice, output quality, injection resistance,
latency, availability, and billed cost remain **unverified**.

## Background ingestion validation — 2026-10-17

This run used six small fictional PDF/DOCX documents from
`data/synthetic_ingestion_v1.json` and the disposable
`resume_intelligence_test` PostgreSQL/pgvector database. No candidate data or
live model API was used.

| Check | Actual result |
|---|---|
| Complete suite with isolated PostgreSQL | **182 passed**, 101 warnings, 11.11 s |
| PostgreSQL integration only | **15 passed**, 167 deselected, 5 warnings, 4.37 s |
| Focused ingestion/API/client/UI tests | **68 passed**, 100 warnings, 8.95 s |
| Ruff / Alembic drift / `git diff --check` | passed / no new operations / clean |
| Compose rebuild | cached dependency layers; API, worker, and Streamlit images built |
| Runtime stack | PostgreSQL/API/Streamlit healthy; worker running; migration `0005_background_ingestion`; pgvector 0.8.7 |
| Container runtime | `torch 2.14.1+cpu`; CUDA false; `torch.version.cuda=None` |
| API image size | **573,950,109 bytes** |
| Synthetic API smoke | background enqueue/replay, durable status, tenant denial, real worker completion, metrics, retrieval, generation fail-closed, and deletion passed |
| Connected Streamlit smoke | all **8 pages** rendered and live hybrid search returned a fictional document |
| Bounded log scan | 24 API/worker/Streamlit lines; 0 matches for the test token, email, identity-bearing filename, or known document/query text |

Integration coverage includes successful ingestion and payload purge,
tenant-scoped job reads, idempotent replay, bounded exponential retry, expired
lease recovery after simulated worker loss, terminal empty-document handling,
atomic reindex metadata updates, and deletion-before-processing without document
resurrection.

### Ingestion benchmark

Both modes execute the production PostgreSQL claim/worker/parser/persistence
path. The deterministic mode substitutes only the embedding calculation and is
the CI regression gate. The real mode loads the repository's local
`all-MiniLM-L6-v2` SentenceTransformer.

| Mode | Completed | Failed | Wall time | Throughput | Mean job latency | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic 384-d fake | 6/6 | 0 | 0.1084 s | 55.3402 docs/s | 9.8825 ms | 8.9752 ms | 17.1248 ms |
| Real MiniLM, cold process/model load included | 6/6 | 0 | 5.0531 s | 1.1874 docs/s | 830.7300 ms | 24.1255 ms | 4728.6330 ms |

The real run's first model load dominates its P95. Six tiny documents processed
serially on one workstation are not a load, concurrency, capacity, or SLA
measurement. Model-cache state and local PostgreSQL performance materially
affect these timings. The tracked embedding version is the
`sentence-transformers` package version; an immutable upstream model artifact
revision is not yet pinned.

### Remaining validation limits

- Worker crash recovery was simulated by expiring a lease in PostgreSQL; a
  process kill during active parsing/embedding was not injected in Docker.
- One worker was measured. Concurrent claim contention, horizontal scaling,
  sustained load, backpressure, and database growth/retention were not tested.
- Upload bytes live temporarily in PostgreSQL until completion or terminal
  failure. There is no object store, malware scanner, OCR sandbox, or automated
  retention purge for old job metadata.
- Streamlit AppTest and the connected eight-page smoke validate the status UI's
  element tree and API integration, not browser-level visual appearance or
  automatic push updates; users refresh persisted status explicitly.
- Legacy synchronous upload endpoints remain for compatibility. Generation and
  investigation execution remain synchronous and were not changed by this work.
