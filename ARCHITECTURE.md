# Architecture: Recruitment Document Intelligence and RAG

## Current state

One Streamlit application (`app.py`) runs in two modes:

- **Standalone mode** (`API_BASE_URL` unset): the original in-memory,
  deterministic Candidate Ranking workflow. Nothing is persisted.
- **V3 Connected mode**: the same ranking, plus the Resume Library, AI Candidate
  Search, Ask AI, Recruiter Briefs, Review History, and AI Investigations pages,
  backed by FastAPI, PostgreSQL/pgvector, the recruiter-brief workflow, and one
  separately bounded read-only investigation agent.

There is one PostgreSQL-backed document-ingestion worker. There is still no
production identity provider, persisted score snapshot, object store, or
production deployment control plane.

## End-to-end data flow

```text
Browser ──> Streamlit (app.py, frontend/views/*)
              │  in-process: services/ranking.py → utils/* (deterministic scoring)
              │
              └─ frontend/api_client.py ── HTTP, server-side bearer token + X-Tenant-ID ──┐
                                                                                           ▼
FastAPI api/main.py, api/workflows.py, api/investigations.py, api/ingestion.py
  authorized(role): token grant ⊇ tenant? role? ── else 401/403, before any DB session
  │
  ├─ Ingestion:  bounded read → ingestion_jobs (queued, temporary bytes) → worker lease/claim
  │              → parse_document_from_bytes (signature/size/pages/DOCX caps) → ≤500 chunks
  │              → MiniLM embed → atomic documents / versions / chunks + completed job
  │              (payload purged at terminal state; filename stored only as document.<ext>)
  ├─ Search:     RetrievalPipeline.search → pgvector cosine + tsvector FTS (tenant, current
  │              version, document filters) → RRF → optional cross-encoder → redacted display text
  ├─ Generation: policy gate (key + tenant policy) → RetrievalPipeline.retrieve → build_context
  │              (budget incl. headers, identifier redaction, canonical IDs) → LangChain Gemini
  │              structured output → strict Pydantic → citation verification → response + sources
  └─ Workflows:  workflow_runs row (tenant index) → RecruiterBriefWorkflow (LangGraph)
                 → PostgresSaver checkpoints → interrupt → reviewer resume → workflow_reviews
  └─ Investigate: investigation_runs (immutable version scope) → InvestigationAgent
                  → model-selected fixed read-only tool → sanitized investigation_events
                  → verified report → interrupt → reviewer resume → investigation_reviews
```

| Capability | Files |
|---|---|
| UI shell, mode detection, navigation | `app.py`, `frontend/context.py`, `frontend/components.py` |
| Typed HTTP client | `frontend/api_client.py`, contracts in `api/schemas.py` |
| Pages | `frontend/views/{overview,library,ranking,search,ask,briefs,investigations,history}.py` |
| Deterministic ranking | `services/ranking.py`, `utils/{scoring,matcher,skills_extractor,candidate_analyzer}.py` |
| Auth, roles, policy | `api/auth.py`, `retrieval/authorization.py`, `api/policy.py`, `api/dependencies.py` |
| Document persistence | `repositories/documents.py`, `db/models.py`, `migrations/versions/0001…0005` |
| Background ingestion | `api/ingestion.py`, `ingestion/worker.py`, `repositories/ingestion_jobs.py`, migration `0005` |
| Retrieval | `retrieval/{pipeline,fusion,reranker,context,contracts}.py`, `services/embeddings.py` |
| Grounded generation | `generation/{service,prompts,schemas,citations,providers}.py`, `api/presenters.py` |
| Review workflow | `workflows/recruiter_brief.py`, `workflows/checkpointing.py`, `repositories/workflow_{runs,reviews}.py` |
| Investigation agent | `investigations/`, `api/investigations.py`, `repositories/investigations.py` |

## Streamlit → FastAPI integration

- Credentials (`API_BASE_URL`, `API_TENANT_ID`, `API_BEARER_TOKEN`) are read
  server-side from the environment or `st.secrets`. There is no editable tenant
  field; even if the header were changed, FastAPI authorizes from the token grant.
- `ApiClient` uses a 3 s connect timeout, 30 s reads, and 180 s for uploads,
  reranked search, and generation. Only idempotent reads (including `POST
  /search`) retry, twice, on connection errors or 502/503/504. Uploads, deletes,
  generation, and reviews never retry automatically.
- Errors become fixed user messages (`ApiError`); tokens never appear in
  messages, `repr`, or logs (`httpx` request logging is suppressed).
- Mode and session (roles, generation status) are refreshed at most every 20 s
  per browser session. Tenant data is cached only in `st.session_state` and
  invalidated after uploads/deletes; `st.cache_data` is not used for tenant data.
- Filenames never leave Streamlit: the client uploads `document.<ext>`. Upload
  results and errors are labelled `File 01…`; stored documents use `candidate_ref`.

## Tenant authorization and roles

`RETRIEVAL_AUTH_TOKENS_JSON` maps a token to tenants (legacy list → all roles) or
to `{"tenants": [...], "roles": [...]}`. `reader` reads/searches/generates,
`editor` uploads and deletes, `reviewer` resumes workflows. Comparison is
constant-time; the principal recorded anywhere is `token-sha256:<12 hex>`. The
`authorized(role)` dependency runs before any session or repository dependency.
Every repository query filters by `tenant_id`; cross-tenant document or workflow
IDs return 404. This adapter is development infrastructure only.

## Parsing and embedding

`utils/pdf_parser.parse_document_from_bytes` validates extension, signature,
size (10 MB), PDF page count, encryption, and DOCX expansion, then produces a
versioned `ParsedDocument` with section-aware chunks and exact offsets (PDF
pages; DOCX `page_number=None`). The API rejects documents over 500 chunks
before embedding. Changed bytes create a new immutable version that becomes the
only retrieval-active version; identical bytes return the existing version.
The connected Resume Library queues this work. The legacy synchronous endpoints
remain available as compatibility contracts.

## Durable background ingestion

The API authorizes the editor before accepting bytes, stores only an opaque
extension and bounded payload in `ingestion_jobs`, and returns `202`. A separate
worker claims due rows with `FOR UPDATE SKIP LOCKED`, a processing lease, and a
maximum attempt count. Retryable failures return to `queued` with bounded
exponential backoff; parser and data-quality failures are terminal. An expired
lease is reclaimable after worker loss.

The worker holds the claimed row lock while parsing, embedding, persisting the
document/version/chunks, marking the result, and purging the payload in one
transaction. A crash before commit therefore leaves neither a partial version
nor a falsely completed job. Document content hashes and tenant-scoped request
keys provide a second idempotency boundary. Reindex jobs replace all vectors for
one authorized version and update its embedding model/package version atomically.
Metrics are derived from tenant-filtered job rows and contain only state counts,
queue age, throughput, and latency. See `BACKGROUND_INGESTION.md` for the
PostgreSQL-versus-Celery decision and operational limits.

## Hybrid retrieval and reranking

`RetrievalPipeline.search` authorizes first, then embeds the query, runs pgvector
cosine and PostgreSQL `tsvector` searches with tenant/current-version/metadata
filters, fuses rank positions with reciprocal-rank fusion (k=60; no claim of
BM25), and optionally reranks the fused set with a process-cached cross-encoder.
Search returns up to 50 ranked chunks without any model-context budget.
`retrieve()` turns the same results into bounded model context. Retrieval
relevance is displayed as RRF/reranker scores only — never as a candidate score.

## Grounded generation and citation boundary

- **Canonical ID.** `RetrievalCandidate.source_id` (the chunk UUID string) is the
  only citation identifier: it labels context blocks, is listed in
  `available_source_chunk_ids`, is required by the verifier, and is returned to
  the UI as `source_chunk_id`.
- **Supplied evidence.** `build_context` enforces the character budget including
  headers, truncates the last excerpt if needed, redacts direct identifiers, and
  records each `ContextSource` (exact excerpt, provenance span, `truncated`).
  `verify_*` checks quotes only against these excerpts, so quotes from truncated
  tails, excluded chunks, or redacted identifiers fail.
- **Gate.** `api/policy.py` requires a provider key and a tenant policy with
  provider, purpose, and retention; otherwise 503/403 before retrieval text can
  reach a provider.
- **Failure handling.** Schema-invalid output (502), provider failure (502), and
  citation failure (422, output withheld) return fixed messages. Traces record
  only metadata (`observability/tracing.py`).
- **Rendering.** Generated text is Markdown-escaped in Streamlit; excerpts are
  HTML-escaped and the cited span is highlighted.

## LangGraph checkpoint and review workflow

`RecruiterBriefWorkflow` (one graph, no agents or tool calling):
`authorize → retrieve_evidence → validate_requirements → generate_brief (≤2 attempts)
→ verify_citations → human_review (interrupt) → persist_review`, with `recover` for
denial, no evidence, retrieval, generation, or citation failures (fixed error codes).

- `POST /v1/workflows/recruiter-brief` creates a tenant-scoped `workflow_runs` row
  per document, runs the graph to the interrupt, and returns the brief and the
  exact supplied sources. Only the requested tenant is checkpointed.
- The checkpoint `thread_id` is the workflow UUID; checkpoints are read only after
  the tenant-filtered run row is found. Responses exclude subjects and grants.
- `POST /v1/workflows/{id}/review` requires `reviewer`, locks the run row, accepts
  only `awaiting_review`, and resumes with server-supplied reviewer and tenant.
  Invalid resumes re-interrupt instead of persisting. A second review returns 409;
  `workflow_reviews` is unique per `(tenant, workflow)`.
- Durability: `PostgresSaver` over a bounded `psycopg_pool` pool. A restarted
  process resumes from PostgreSQL (integration-tested by closing all pooled
  connections between start and review). If a crash separates the checkpoint
  write from the review commit, the next read reconciles the review row
  idempotently from checkpoint state.
- Alembic ignores LangGraph-owned checkpoint tables (`migrations/env.py`).

## Bounded investigation agent

`InvestigationAgent` is the only tool-using agent. Gemini chooses its next action
through provider-supported structured tool calls; LangGraph conditionally routes
to one of `search_evidence`, `inspect_document`, `check_requirements`,
`get_review_history`, or `draft_investigation`. The server dispatcher validates
arguments and tenant/document/version scope before every call. Tools are
read-only, retrieved text is untrusted, and no tool can alter scope, scores,
documents, reviews, or applicant outcomes.

Creation is synchronous. `POST /v1/investigations` commits queued/running rows,
runs at most six tool calls and three planner iterations under token/cost/time
budgets, persists sanitized events, then returns at human review or failure.
There is no worker, polling stream, or fabricated progress. PostgresSaver holds
graph state; `investigation_runs`, `investigation_events`, and
`investigation_reviews` hold tenant-filtered presentation/audit records. GET
routes load the tenant row before checkpoint access. Human review requires the
reviewer role and applies only to generated notes.

### Investigation validation boundary

`data/investigation_eval_v1.json` and `evals/run_investigation_eval.py` exercise
the real graph, conditional router, server tool dispatcher, context redaction,
budget enforcement, structured report validation, and exact-citation verifier
with a deterministic scripted provider. The cases cover direct evidence,
ambiguity, no evidence, adaptive follow-up search, contradictions,
multi-document scope, prompt injection, unauthorized references, budget
exhaustion, invalid arguments, unavailable tools, and hallucinated quotes. CI
runs this benchmark and `evals/quality_gate.py` requires all policy metrics to
remain at their documented thresholds. This is an orchestration regression
test, not a live-model or retrieval-relevance evaluation.

## Standalone fallback

With `API_BASE_URL` unset, `resolve_context` returns Standalone mode without any
network call. V2 pages render an explanation instead of controls; nothing claims
to be saved. If the API is configured but unreachable or unauthorized, the
sidebar shows **V2 unavailable** with the safe reason, and ranking still works.
Rollback to Standalone is achieved by unsetting `API_BASE_URL`.

## Primary risks

1. **Identity proxies (high):** regex redaction (email/phone/URL/header name) is
   not de-identification. Names in body text, addresses, and demographic proxies
   can remain in stored chunks, display text, and model context.
2. **Development authorization (high):** static bearer tokens in environment
   variables; no expiry, rotation, user identity, or audit of access events.
3. **Ranking audit (medium):** ranking results and manual ranking decisions are
   session-only and exported by the user; there is no immutable score snapshot.
4. **Mixed execution model (medium):** library parsing/embedding is background,
   but legacy uploads, reranking, generation, and investigations remain in request threads.
5. **Parser and provider risk (medium):** no malware scanning or OCR isolation;
   live provider behavior, cost, and retention are unvalidated in this repository.
6. **Supply chain (medium):** dependencies are range-constrained, not locked;
   models download at runtime into a cache volume.
7. **Historical-data retention (high):** document deletion removes document
   rows, chunks, and vectors, but historical workflow/investigation rows and
   LangGraph checkpoints can retain redacted excerpts. A retention/deletion
   policy and tested purge path are required before non-synthetic use.
8. **Mid-execution recovery (medium):** review interrupts resume durably, but a
   process loss during synchronous planning leaves a diagnostic queued/running
   record; there is no worker that automatically continues it.
9. **Queue payload storage (medium):** bounded original bytes live temporarily
   in PostgreSQL until terminal processing. There is no object store, malware
   scanner, autoscaling, or high-concurrency claim/load validation.

## Deliberately out of scope

No UI rewrite, autonomous ranking agent, multi-agent system, automatic hiring
actions, brief editing, knowledge graph, Elasticsearch, separate vector database,
Redis/Celery/Kafka, OCR rollout without measured need, or broad skill ontology.
PostgreSQL + pgvector is sufficient until measured scale or relevance demands otherwise.
