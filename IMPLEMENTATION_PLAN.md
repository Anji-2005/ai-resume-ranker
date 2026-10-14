# Incremental implementation plan

## Preconditions and success boundary

Approval is required before changing production behavior. Every milestone keeps
the current Streamlit UI and deterministic score results compatible for the same
fixture inputs unless its acceptance criteria explicitly approve a change.

## Dependency direction

Keep current dependencies during the parity phase. Add only when the matching
milestone starts:

| Need | Proposed packages/infrastructure | Notes |
|---|---|---|
| API/contracts | `fastapi`, `uvicorn`, `pydantic` | Separate request schemas from domain models. |
| PostgreSQL/vector/migrations | PostgreSQL with `pgvector`, `sqlalchemy`, `alembic`, `psycopg` | One relational source of truth; no separate vector DB initially. |
| Retrieval/reranking | PostgreSQL FTS, an embedding client, `sentence-transformers` cross-encoder | Add only after provenance/chunks exist. |
| Structured LLM/workflow | `langchain`, narrowly chosen provider integration, `langgraph` | Retain current output schema and fallback behavior; pin compatible versions after a tested lock. |
| Quality/operations | `pytest`, `httpx`, `testcontainers`, OpenTelemetry SDK, migration checks | Introduce as deployment scope requires. |

Do not add LangChain/LangGraph, a vector index, or a database merely for a local
demo. Use a lockfile/constraint file, vulnerability scanning, and SBOM in CI;
exact versions should be selected against the supported Python/runtime matrix at
implementation time, not guessed in this planning document.

## Ordered milestones

### 0. Safety and regression baseline (no behavior change)

**Affected:** `tests/`, `README.md`, new `AGENTS.md`, `ARCHITECTURE.md`, this plan.

- Add golden PDF/DOCX fixtures and score snapshots; capture export behavior.
- Add tests proving blind-mode CSV, Excel, JSON, failure messages, and UI labels
  never disclose filenames or direct identifiers.
- Define threat model, data classification, provider/retention policy, supported
  deployment topology, and role matrix.

**Acceptance:** current deterministic and GenAI fallback tests pass; documented
known leakage tests fail until the approved remediation milestone; no feature
code changed.

### 1. Establish typed domain contracts and fix confidentiality boundaries

**Affected:** new `domain/` (models/interfaces), `utils/privacy.py`,
`utils/scoring.py`, `utils/skills_extractor.py`, `utils/genai_analyzer.py`,
`app.py`, tests.

- Introduce versioned Pydantic/domain DTOs for job criteria, candidate identity,
  parsed text, provenance, analysis, exports, and generation.
- Issue opaque candidate IDs at intake; isolate filename/original metadata from
  blind views/exports. Create distinct blind and authorized audit serializers.
- Replace the global cache contract with a scoped cache interface whose key
  includes tenant/policy/schema/model versions and TTL; make purge verifiable.
- Keep existing score formulas and UI values unchanged.

**Acceptance:** identical fixture score components; no filename/source text in
blind UI, errors, logs, or exports; schema compatibility tests pass.

### 2. Harden parsing and add provenance

**Affected:** `utils/pdf_parser.py`, new `parsing/`, `domain/`, tests; small
adapter update in `app.py`.

- Return normalized text plus page/paragraph/span provenance rather than plain
  strings. Preserve `extract_text_from_bytes` as a compatibility adapter.
- Enforce extracted-text, archive-entry, compression-ratio, and processing-time
  limits; scan uploads at the deployment boundary; retain documented OCR
  unsupported behavior.
- Add corrupt, encrypted, oversized, ZIP-bomb-shaped, Unicode, table, and
  provenance fixture tests.

**Acceptance:** existing PDF/DOCX flows work; every displayed deterministic
evidence item resolves to a source span; unsafe files fail without leaking names
in blind mode.

### 3. Introduce PostgreSQL/pgvector persistence behind repositories

**Affected:** new `db/`, `repositories/`, Alembic migrations, `domain/`,
integration tests; no Streamlit redesign.

- Persist tenant-scoped documents, parsed chunks, immutable score snapshots,
  review events, and minimal generation audit metadata. Encrypt and lifecycle
manage any object storage separately.
- Add idempotency/content hashes, optimistic concurrency for review updates,
  transaction boundaries, retention deletion, and authorization-aware queries.
- Dual-write/read behind a feature flag; retain session state as fallback until
  parity is proven.

**Acceptance:** restart does not lose authorized review history; repeat intake
does not duplicate documents; tenant isolation, deletion, migration rollback,
and export reproducibility tests pass.

### 4. Add FastAPI service façade; Streamlit remains the UI

**Affected:** new `api/`, `services/`, `app.py`, API and contract tests.

- Expose bounded upload, job creation, analysis status/result, review-event,
  and authorized export endpoints. Use asynchronous job execution only if UI
  latency measurements require it.
- Move orchestration from `app.py` into a service interface; Streamlit calls the
  API while retaining its screens, controls, filters, downloads, and human
  review behavior.

**Acceptance:** API/Streamlit contract tests show equivalent scores, evidence,
fallbacks, and review behavior; authenticated/unauthorized/tenant cases pass;
the old direct path remains feature-flagged for rollback.

### 5. Add hybrid retrieval and reranking for evidence-grounded RAG

**Affected:** new `retrieval/`, migrations/indexes, `services/`, tests.

- Chunk only normalized provenance-bearing text; create versioned embeddings.
- Retrieve with tenant/document/policy filters, FTS + pgvector fusion, then
  cross-encoder reranking of a small top-K. Return chunk IDs and spans.
- Do not replace deterministic scoring with retrieval. Use RAG only to select
  evidence for summaries/questions/explanations.

**Acceptance:** retrieval relevance benchmark beats lexical-only baseline on a
curated, privacy-safe set; citations resolve exactly; no cross-tenant result is
possible; latency/cost budgets are met.

### 6. Migrate Gemini enrichment to structured generation and one workflow

**Affected:** `utils/genai_analyzer.py` replacement adapter, new `generation/`,
`workflows/`, evaluation fixtures/tests.

- Preserve the present JSON schema and strict validation first. Use LangChain
  structured output behind an interface; include only retrieved/cited chunks.
- Model the bounded pipeline in one LangGraph: validate → parse → redact →
  analyze → retrieve → generate → validate → persist. Deterministic stages stay
  normal functions; human review remains outside model control.
- Add provider outage, timeout, prompt injection, sensitive-topic, fabricated
  citation, and cache-policy tests.

**Acceptance:** deterministic score/rank is unchanged; generation fallback is
safe; every displayed generated claim is traceable or withheld; no multi-agent
behavior exists.

### 7. Evaluation, operations, and controlled rollout

**Affected:** `evals/`, CI/CD configuration, deployment manifests/runbooks,
observability configuration.

- Establish versioned offline evals for parser fidelity, redaction recall,
  deterministic score parity, retrieval recall/nDCG, citation accuracy,
  structured-output validity, sensitive-content rate, latency, and cost.
- Add privacy-safe telemetry, audit access events, alerts, backup/restore and
  deletion drills, dependency/SBOM scans, and role-based deployment controls.
- Shadow-run with consented synthetic or approved data, compare snapshots, then
  roll out by tenant/feature flag with rollback criteria.

**Acceptance:** predefined quality, privacy, latency, and rollback thresholds
are approved and met; a human owner signs off before production expansion.

## Current validation note — 2026-10-14

The repository-local virtual environment completed 54 tests, including three
PostgreSQL/pgvector integration tests against the disposable
`resume_intelligence_test` database. Alembic reported no schema drift, Ruff and
the synthetic quality gate passed, and the CPU-only Docker API completed the
document smoke flow. Exact measured results and remaining release limitations
are recorded in `EVALUATION_RESULTS.md`.

## Streamlit integration note — 2026-10-14

Milestones 4–6 are now exposed through Streamlit: a typed API client, document
library, hybrid search, grounded Q&A, and the LangGraph review lifecycle over
HTTP (`0003_workflow_runs`). Ranking remains the deterministic in-session path
(milestone 3's score snapshots and review events are still not implemented).
Executed results are in `EVALUATION_RESULTS.md` under "Streamlit integration validation".
