# Resume Review Assistant — Recruitment Document Intelligence

An explainable, human-in-the-loop recruitment document workspace. A Streamlit
dashboard combines the original **deterministic, explainable candidate ranking**
with a V3 **FastAPI + PostgreSQL/pgvector** backend for persistent resume
storage, **hybrid retrieval**, **citation-verified grounded generation**, and a
durable **LangGraph human-review workflow**.
It also includes one bounded, read-only **AI Investigation Agent** with an
immutable evidence scope, persisted tool trace, verified citations, and human review.

> This application is a decision-support prototype. It prioritizes documents for
> human review and helps reviewers find evidence. It never makes, recommends, or
> records hiring decisions, and must never be the sole basis for an employment action.

## Problem

Recruiters reviewing many resumes need to (1) prioritize against explicit
criteria without a black box, (2) find specific evidence quickly, and (3) use AI
assistance without trusting fluent but unsupported text. This project keeps
scoring deterministic and inspectable, uses retrieval for evidence rather than
ranking, and accepts AI output only when every claim cites an exact excerpt of
the evidence the model was actually given — followed by a recorded human review.

## Features

**Candidate Ranking (works in both modes)**
- Weighted, fully visible score: semantic similarity (local MiniLM), required and
  preferred skill coverage, explicitly stated experience
- Skill evidence snippets, review cautions, candidate filtering, side-by-side comparison
- Blind-review mode (default on): direct-identifier redaction and opaque candidate labels
- In-session human review records; CSV, Excel, and blind-safe audit JSON exports
- Optional original Gemini enrichment with rule-based fallbacks (never changes scores)

**V3 Connected mode**
- **Resume Library** — batch PDF/DOCX upload with durable background status,
  bounded retry/crash recovery, immutable versions, chunk provenance,
  transactional reindexing, operational metrics, and physical deletion
- **AI Candidate Search** — pgvector cosine + PostgreSQL full-text search fused
  with reciprocal-rank fusion, optional cross-encoder reranking, document filters,
  retrieval diagnostics, and a no-LLM requirement evidence lookup
- **Ask AI** — grounded Q&A with numbered citations that expand to the exact
  highlighted excerpt, explicit uncertainty, and no-evidence states
- **Recruiter Briefs** — one bounded LangGraph workflow per document: evidence
  summary, requirement-by-requirement evidence, grounded interview questions,
  then a human approves or rejects the AI-generated notes
- **Review History** — durable, resumable review sessions backed by PostgreSQL checkpoints
- **AI Investigations** — one bounded LangGraph agent dynamically selects from
  five authorized read-only tools, records a safe execution trace, reports
  supported/missing/contradictory evidence, and pauses for review of its notes

**Safeguards**
- Fail-closed token-to-tenant authorization with reader/editor/reviewer roles on every candidate route
- Per-tenant generation policy gate: no candidate text reaches a provider without a configured key *and* a documented provider/purpose/retention policy
- Filenames never stored or returned; opaque `CAND-xxxxxxxx` labels everywhere
- Direct identifiers (email, phone, URL, header name) redacted from model context and V2 display text
- One canonical citation ID; quotes verified only against text actually supplied to the model
- Model output escaped before rendering (no generated links/images/HTML); search queries sent in POST bodies, not URLs
- No access logs, no prompt/resume logging, Streamlit usage telemetry disabled

## Architecture

```mermaid
flowchart LR
  subgraph UI[Streamlit app.py]
    R[Candidate Ranking<br/>services/ranking.py]
    V[Library · Search · Ask AI<br/>Briefs · Investigations · History]
    C[frontend/api_client.py<br/>typed httpx client]
  end
  R -->|in-process| U[utils/: parser, MiniLM,<br/>skills, scoring, privacy]
  V --> C -->|HTTP + bearer token<br/>X-Tenant-ID| A
  subgraph API[FastAPI api/]
    A[auth + tenant/role checks] --> D[documents & search]
    A --> G[grounded generation<br/>policy gate]
    A --> W[workflow + investigation lifecycle]
  end
  A --> IQ[(ingestion_jobs<br/>temporary bounded payload)]
  IQ --> IW[PostgreSQL queue worker<br/>parse · embed · atomic persist]
  IW --> P
  D --> P[(PostgreSQL 16 + pgvector<br/>documents · versions · chunks)]
  D --> RP[retrieval/: dense + FTS → RRF → rerank]
  RP --> P
  G --> RP
  G --> CTX[build_context: budget + redaction] --> LLM[LangChain Gemini<br/>structured output]
  LLM --> CV[citation verification]
  W --> LG[LangGraph RecruiterBriefWorkflow] --> CK[(PostgresSaver checkpoints<br/>workflow_runs · workflow_reviews)]
  W --> IA[Bounded InvestigationAgent<br/>fixed read-only tools] --> IK[(investigation runs · events · reviews)]
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow, boundaries, and caveats.

## Technology

Python 3.14 · Streamlit 1.62 · FastAPI · Pydantic v2 · SQLAlchemy 2 + Alembic ·
PostgreSQL 16 + pgvector (HNSW) · sentence-transformers (all-MiniLM-L6-v2,
ms-marco-MiniLM cross-encoder) · LangChain (`langchain-google-genai`) ·
LangGraph + `langgraph-checkpoint-postgres` · httpx · Docker Compose · pytest · Ruff

## Main workflows

1. **Upload & manage** — Resume Library → queue synthetic PDFs/DOCX → inspect
   persisted job status/metrics → inspect provenance, queue a version or
   embedding reindex, or delete.
2. **Rank** — Candidate Ranking → job description, required/preferred skills,
   minimum years → analyze → explainable breakdown, evidence, comparison, human
   review, exports. Optionally also index the same files into the library.
3. **Search** — AI Candidate Search → natural-language query, optional document
   filter and reranking → passages with candidate label, page, offsets, and RRF scores.
4. **Ask** — Ask AI → question → cited claims; expand a citation to see the exact
   highlighted excerpt and its provenance.
5. **Brief & review** — Recruiter Briefs → select documents and requirements →
   workflow pauses for review → approve or reject the notes with an optional note
   → revisit in Review History.
6. **Investigate & review** — AI Investigations → state an objective → select
   authorized documents → optionally add a job description and explicit
   requirements → wait for the synchronous bounded run → inspect its persisted
   tool sequence and exact citations → approve or reject the generated notes.

### Screenshots

Screenshots are not committed (they could capture candidate data). To produce
them with synthetic data: start the stack, upload fictional resumes, then
capture Overview, Resume Library, Candidate Ranking (Results tab), AI Candidate
Search, Ask AI, and Recruiter Briefs at 1440×900.

## Scoring

| Component | Default weight | Meaning |
|---|---:|---|
| Semantic similarity | 40% | Overall relationship between the job description and resume |
| Required skills | 35% | Percentage of configured required skills found |
| Preferred skills | 10% | Percentage of configured preferred skills found |
| Experience | 15% | Explicitly stated years relative to the configured minimum |

Weights are configurable and normalized automatically. If preferred skills or a
minimum-experience requirement are not configured, their weights are removed and
the remaining components are renormalized. Skill mentions demonstrate textual
evidence, not proficiency. **Search relevance and generated text never feed the
score.** `tests/test_ranking_regression.py` pins scores, components, and order to
output captured from the pre-integration `app.py`.

## Running locally

### Option A — full stack with Docker Compose (V3 Connected mode)

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, put one long random token in BOTH
# RETRIEVAL_AUTH_TOKENS_JSON and API_BEARER_TOKEN (same tenant as API_TENANT_ID).
docker compose up --build
```

- Streamlit: http://localhost:8501 · API: http://localhost:8000 (OpenAPI at `/docs`)
- Containers talk over internal service names (`streamlit → http://api:8000 → postgres:5432`).
- The API applies Alembic migrations on start. Models download on first use into
  the `model_cache` volume.
- Status: `docker compose ps`; logs: `docker compose logs --tail=100 api worker streamlit`.
- Reset all stored data: `docker compose down -v` (deletes the database and model cache volumes).
- Source-mounted development: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build`.

To enable grounded generation for a **synthetic** demo, set `GEMINI_API_KEY` and a
tenant policy, e.g.
`GENERATION_TENANT_POLICY_JSON={"demo-tenant":{"provider":"gemini","purpose":"demo","retention":"synthetic data only"}}`.
AI Investigations additionally requires accurate
`GENAI_INPUT_USD_PER_MILLION_TOKENS` and
`GENAI_OUTPUT_USD_PER_MILLION_TOKENS` values for the configured Gemini model;
the agent fails closed when cost accounting is unavailable.
Without both, Ask AI and Recruiter Briefs show why generation is disabled; search
and ranking keep working.

### Option B — Standalone Streamlit (no database)

```bash
python3 -m venv ../ai-resume-ranker-venv
source ../ai-resume-ranker-venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

With `API_BASE_URL` unset, the app runs in **Standalone mode**: Candidate Ranking
is fully functional and V2 pages explain how to connect. Nothing is saved.

### Option C — local API + Streamlit without Compose for the app

Start PostgreSQL (e.g. `docker compose up postgres`), then:

```bash
export DATABASE_URL=... RETRIEVAL_AUTH_TOKENS_JSON='{"<token>":["demo-tenant"]}'
alembic upgrade head && uvicorn api.main:app --port 8000 --no-access-log
# second shell
DATABASE_URL="$DATABASE_URL" python -m ingestion.worker
# third shell
API_BASE_URL=http://localhost:8000 API_TENANT_ID=demo-tenant API_BEARER_TOKEN=<token> streamlit run app.py
```

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Sidebar shows **V2 unavailable** · "rejected the server-side credentials" | `API_BEARER_TOKEN` is not a key in `RETRIEVAL_AUTH_TOKENS_JSON` |
| "not authorized for this operation or workspace" | Token lacks `API_TENANT_ID` or the needed role (editor/reviewer) |
| "database is unavailable" | PostgreSQL not healthy or `DATABASE_URL` wrong; check `docker compose ps` |
| Ask AI says generation is disabled | Missing `GEMINI_API_KEY` on the **API** or missing tenant policy |
| First search/rerank is slow | MiniLM / cross-encoder downloading into the model cache |
| `POSTGRES_PASSWORD` error on `compose up` | `.env` missing; copy `.env.example` |

## API

All candidate routes require `Authorization: Bearer <token>` and `X-Tenant-ID`;
the token's server-side grant — not the header — is the authority.

| Method & path | Role | Purpose |
|---|---|---|
| `GET /v1/health` | none | Liveness, DB reachability, configured capabilities (no tenant data) |
| `GET /v1/session` | reader | Tenant, roles, generation status for the tenant |
| `GET /v1/documents` | reader | Document summaries (`candidate_ref`, versions, chunk counts) |
| `POST /v1/documents` | editor | Upload; idempotent by content when `source_key` is omitted |
| `POST /v1/ingestion-jobs` | editor | Queue an upload; requires a tenant-scoped `Idempotency-Key` |
| `POST /v1/documents/{id}/ingestion-jobs` | editor | Queue a new immutable version |
| `POST /v1/documents/{id}/reindex` | editor | Queue transactional re-embedding of an authorized version |
| `GET /v1/ingestion-jobs[?status=]` · `GET /v1/ingestion-jobs/{id}` | reader | Durable job state, attempts, safe errors, model identity, and result |
| `GET /v1/ingestion-jobs/metrics` | reader | Tenant-scoped counts, queue age, throughput, and processing latency |
| `GET /v1/documents/{id}?version=` | reader | Version history and redacted chunks with provenance |
| `POST /v1/documents/{id}/versions` | editor | Upload a new version |
| `DELETE /v1/documents/{id}` | editor | Physical deletion of versions, chunks, vectors |
| `POST /v1/documents/search` | reader | Hybrid retrieval (`GET` kept for compatibility) |
| `POST /v1/documents/qa` | reader + policy | Citation-verified Q&A with supplied sources |
| `POST /v1/documents/recruiter-brief` | reader + policy | Unreviewed citation-verified brief (API only) |
| `POST /v1/workflows/recruiter-brief` | reader + policy | Start one review workflow per document (≤5) |
| `GET /v1/workflows[?status=]` · `GET /v1/workflows/{id}` | reader | History; status, pending brief, final review |
| `POST /v1/workflows/{id}/review` | reviewer | Approve/reject AI-generated notes; 409 if not pending |
| `POST /v1/investigations` | reader + policy | Synchronously run one bounded investigation; requires `Idempotency-Key` |
| `GET /v1/investigations[?status=]` · `GET /v1/investigations/{id}` | reader | Durable history, immutable scope, safe tool trace, report and review |
| `POST /v1/investigations/{id}/review` | reviewer | Approve/reject generated investigation notes; never an applicant decision |

## Environment variables

| Variable | Used by | Purpose | Default |
|---|---|---|---|
| `DATABASE_URL` | API, Alembic | SQLAlchemy PostgreSQL URL | required for V2 |
| `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`, `DATABASE_ECHO` | API | Pool tuning / SQL echo | `5`, `10`, `false` |
| `RETRIEVAL_AUTH_TOKENS_JSON` | API | Dev token → tenants (list) or `{"tenants":[],"roles":[]}` | `{}` (deny all) |
| `GENERATION_TENANT_POLICY_JSON` | API | Tenant → `{provider:"gemini",purpose,retention}` | `{}` (generation off) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | API, Streamlit | Provider key/model | unset, `gemini-3.1-flash-lite` |
| `API_BASE_URL` | Streamlit | V2 API base URL; unset = Standalone mode | unset |
| `API_TENANT_ID`, `API_BEARER_TOKEN` | Streamlit | Server-side V2 credentials (never shown in UI) | unset |
| `STREAMLIT_GEMINI_API_KEY` | Compose | Maps to the Streamlit container's `GEMINI_API_KEY` for ranking enrichment | unset |
| `POSTGRES_*`, `API_PORT`, `STREAMLIT_PORT` | Compose | Database settings and published ports | `.env.example` |
| `LANGGRAPH_STRICT_MSGPACK` | API | Restrict checkpoint deserialization | `true` |
| `TEST_DATABASE_URL` | tests | Isolated pgvector DB for integration tests | unset (skipped) |
| `INGESTION_MAX_ATTEMPTS` | API | Maximum attempts per job, bounded 1–10 | `3` |
| `INGESTION_RETRY_BASE_SECONDS` | worker | Exponential retry base, bounded to 300 seconds; computed delay caps at one hour | `5` |
| `INGESTION_LEASE_SECONDS` | worker | Crash-recovery lease, bounded to 3600 seconds | `600` |
| `INGESTION_POLL_SECONDS` | worker | Empty-queue polling interval, bounded to 60 seconds | `2` |
| `GENAI_INPUT/OUTPUT_USD_PER_MILLION_TOKENS` | API | Required investigation-agent cost rates; run fails closed if absent | unset |

`API_BEARER_TOKEN`/`API_TENANT_ID` can also be set in `.streamlit/secrets.toml`.
The token-map adapter is **local development authorization, not production
authentication**; production must use an identity provider's principal/tenant claims.

## Testing and evaluation

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q                                   # unit, API contract, client, UI tests
# Real PostgreSQL/pgvector (isolated database only — tables are truncated):
export TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/resume_intelligence_test
DATABASE_URL="$TEST_DATABASE_URL" alembic upgrade head
DATABASE_URL="$TEST_DATABASE_URL" alembic check
pytest -m integration -q
# Offline evaluations and regression gate (no live model calls):
python -m evals.run_whole_document_baseline
python -m evals.run_generation_eval
python -m evals.run_offline_quality_eval
python -m evals.run_investigation_eval
DATABASE_URL="$TEST_DATABASE_URL" python -m evals.run_postgres_retrieval_eval
python -m evals.run_ingestion_benchmark --database-url "$TEST_DATABASE_URL"
# Optional measurement including the real locally cached MiniLM embedder:
python -m evals.run_ingestion_benchmark --database-url "$TEST_DATABASE_URL" --real-embedder --output evals/latest_ingestion_benchmark_real.json
python -m evals.quality_gate
# Against a running stack (synthetic data; refuses live generation unless explicitly enabled):
SMOKE_API_URL=http://localhost:8000 SMOKE_TENANT_ID=demo-tenant SMOKE_BEARER_TOKEN=<token> python scripts/smoke_api.py
API_BASE_URL=http://localhost:8000 API_TENANT_ID=demo-tenant API_BEARER_TOKEN=<token> python scripts/smoke_streamlit.py
```

Test layers: `tests/test_investigation_agent.py` (conditional tool routing,
follow-up retrieval, authorization, budgets, safe failures, review resume),
`tests/test_investigation_eval.py` (versioned 12-scenario benchmark contract),
`tests/test_context_citations.py` (citation/ID/budget regressions),
`tests/test_api_contracts.py` (routes, auth, roles, policy, safe errors,
workflows), `tests/test_api_client.py` (timeouts, retries, error mapping, live
client↔FastAPI contract), `tests/test_frontend.py` (every page, connected flows,
citations, empty/error states, blind review), `tests/test_ranking_regression.py`
(score parity), `tests/test_ingestion_worker.py` (embedding validation and retry
policy), `tests/integration/` (PostgreSQL persistence, ingestion retry/crash
recovery/reindexing, workflow restart and crash reconciliation). No test calls a
live LLM; CI never does either.

## Manual V3 UI test with fictional resumes

1. Configure `.env` with a local token, `demo-tenant`, a Gemini key, an explicit
   synthetic-data tenant policy, and the current Gemini input/output token rates.
2. Run `docker compose up --build`, wait for PostgreSQL, API, worker, and
   Streamlit to start (the worker has no HTTP health endpoint), then open
   `http://localhost:8501`.
3. In **Resume Library**, upload two fictional DOCX/PDF files. Example content:
   `Alex Example — Built Python/FastAPI services backed by PostgreSQL` and
   `Taylor Example — Operated Kubernetes workloads on AWS`. Do not use real resumes.
4. Open **AI Investigations**, enter `Investigate evidence of production Python
   API ownership and identify missing Kubernetes evidence`, select both opaque
   `CAND-…` documents, optionally paste a fictional job description, and enter
   `Python, Kubernetes` as explicit requirements.
5. Select **Start Investigation**. The request is synchronous; keep the page
   open. When it returns, confirm that the persisted trace contains only actual
   tool calls, every supported claim expands to an exact redacted excerpt, and
   gaps are explicit rather than inferred.
6. Approve or reject the generated notes with a fictional reviewer comment.
   Restart the API and Streamlit containers, reopen **AI Investigations**, and
   confirm the scope, trace, report, and note-review outcome remain available.
7. Repeat with a reader-only token and verify the review controls are disabled.
   Remove the tenant policy and verify new investigations are disabled while
   history remains readable.

Measured results — including what was **not** verified — are in
[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

## Responsible-AI safeguards

- Deterministic scores are computed only from the resume text and configured criteria; AI never alters scores, ranks, or review records.
- Generated content is labelled as AI assistance requiring human review; workflow review approves or rejects the *notes*, never the applicant.
- Every generated claim, requirement, and interview question must cite an exact excerpt of the redacted, budgeted evidence supplied to the model; otherwise the output is withheld.
- Prompts mark documents as untrusted data and forbid ranking, hiring, rejection, and protected-characteristic inference.
- Generation is explicit and user-initiated. V3 has exactly one bounded
  investigation agent with a fixed read-only tool registry; there are no
  sub-agents, dynamic tools, ranking actions, or autonomous employment decisions.

## Known limitations

- Ranking results and ranking review records remain in-session (exported for audit); there is no persisted score-snapshot schema. Ranking stored library documents without re-upload is not implemented.
- Identifier redaction is regex-based and is not guaranteed de-identification; names outside the header line, addresses, and demographic proxies may remain.
- The token-map authorization adapter and role model are development infrastructure, not production identity/RBAC.
- Connected-mode library parsing and embedding use the durable PostgreSQL queue;
  legacy compatibility uploads, deterministic ranking, generation, and agent
  execution remain synchronous.
- Investigation `queued`/`running` records support crash diagnosis and checkpoint
  reconciliation, but they do not imply a background worker or live streaming.
  Pending human review resumes after restart; an execution interrupted while
  still planning is not automatically continued by a worker.
- Deleting a source document removes its versions, chunks, and vectors, but does
  not currently apply a documented retention policy to historical workflow and
  investigation reports/checkpoints that may contain redacted excerpts.
- Generated briefs cannot be edited; reviewers approve or reject them with a note.
- Live Gemini generation has not been validated in this repository; only fake-provider and offline verification were executed.
- Scanned PDFs are detected but OCR is not bundled; experience extraction uses explicit phrases only.
- Evaluation datasets are tiny and synthetic; they are regression harnesses, not evidence of relevance, fairness, or load capacity.
- The investigation benchmark uses a scripted fake provider. It validates the
  real graph, dispatcher, budgets, citation verifier, and failure states, but not
  live Gemini tool selection, latency, cost, or prompt-injection behavior.

## Privacy and security behavior

- Standalone ranking processes uploads in memory. Connected mode temporarily
  stores bounded upload bytes in a tenant-scoped job row until completion or
  terminal failure, then purges them. It persists normalized text, chunks,
  embeddings, model identity, and provenance until authorized deletion; only an
  opaque `document.pdf`/`document.docx` filename is stored.
- Supported uploads are PDF and DOCX up to 10 MB (enforced in Streamlit and API),
  with signature, page-count, DOCX-expansion, and 500-chunk ingestion limits.
- Workflow, investigation, and review records store a hashed principal (`token-sha256:…`), never token material.
- Error responses use fixed messages; validator and provider exception text is never returned.
- Blind review cannot guarantee removal of every identity or demographic proxy.

## Rollback procedure

Record the prior image digest and database revision before a controlled pilot.
If an authorization, privacy, migration, or quality threshold regresses: stop new
intake, unset `API_BASE_URL` to route reviewers to Standalone ranking, redeploy the
prior image digest, and restore the previous application revision. Revision
`0004_investigations` adds investigation audit tables and
`0005_background_ingestion` adds job/embedding metadata. Stop intake and the
worker, allow or explicitly fail queued work, and review both downgrades before
running them. Never downgrade or restore PostgreSQL without a tested backup.

## Optional Gemini enrichment in Candidate Ranking

The ranking page's original enrichment uses `GEMINI_API_KEY` in the **Streamlit**
process (or `.streamlit/secrets.toml`). One request per candidate produces a
summary, suggestions, three interview questions, and requirement evidence; it is
cached in memory and never changes scores. Without a key, rule-based summaries are
shown. Only anonymized text is sent when blind review is on. Free-tier data may be
used by Google to improve its products — use fictional resumes for demos.

## Project structure

```text
app.py                       Streamlit entrypoint: mode detection, navigation
frontend/api_client.py       Typed V2 HTTP client (timeouts, safe retries, safe errors)
frontend/context.py          Standalone/Connected mode, session-scoped caches
frontend/components.py       Citation, brief, answer, status rendering
frontend/views/*.py          Overview, Library, Ranking, Search, Ask AI, Briefs, Investigations, History
services/ranking.py          Deterministic analysis/exports (no Streamlit)
api/main.py, api/workflows.py, api/investigations.py, api/ingestion.py  FastAPI routes
api/auth.py, api/policy.py   Dev token authorization + roles; generation policy gate
retrieval/                   Contracts, budgeted context, RRF, reranker, pipeline
generation/                  Prompts, schemas, citation verification, service
workflows/                   LangGraph recruiter-brief workflow, checkpointers
investigations/              One bounded tool-using agent, schemas, validation, safe trace
ingestion/worker.py           Durable PostgreSQL queue worker
repositories/ingestion_jobs.py Tenant-scoped claims, retries, status and metrics
repositories/, db/, migrations/  PostgreSQL persistence and Alembic revisions
utils/                       Parsing, matching, skills, scoring, privacy, enrichment
evals/, data/                Offline evaluation harness and synthetic datasets
scripts/                     API and Streamlit smoke tests
```
