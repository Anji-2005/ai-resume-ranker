# Architecture: Recruitment Document Intelligence and RAG

## Current state

One Streamlit application (`app.py`) runs in two modes:

- **Standalone mode** (`API_BASE_URL` unset): the original in-memory,
  deterministic Candidate Ranking workflow. Nothing is persisted.
- **V2 Connected mode**: the same ranking, plus the Resume Library, AI Candidate
  Search, Ask AI, Recruiter Briefs, and Review History pages, all backed by the
  FastAPI service, PostgreSQL/pgvector, and one LangGraph workflow.

There is still no background worker, production identity provider, persisted
score snapshot, object store, or production deployment control plane.

## End-to-end data flow

```text
Browser ──> Streamlit (app.py, frontend/views/*)
              │  in-process: services/ranking.py → utils/* (deterministic scoring)
              │
              └─ frontend/api_client.py ── HTTP, server-side bearer token + X-Tenant-ID ──┐
                                                                                           ▼
FastAPI api/main.py, api/workflows.py
  authorized(role): token grant ⊇ tenant? role? ── else 401/403, before any DB session
  │
  ├─ Documents:  bounded read → parse_document_from_bytes (signature/size/pages/DOCX caps)
  │              → ≤500 chunks → MiniLM embed → documents / document_versions / document_chunks
  │              (filename persisted only as document.<ext>; identical bytes idempotent)
  ├─ Search:     RetrievalPipeline.search → pgvector cosine + tsvector FTS (tenant, current
  │              version, document filters) → RRF → optional cross-encoder → redacted display text
  ├─ Generation: policy gate (key + tenant policy) → RetrievalPipeline.retrieve → build_context
  │              (budget incl. headers, identifier redaction, canonical IDs) → LangChain Gemini
  │              structured output → strict Pydantic → citation verification → response + sources
  └─ Workflows:  workflow_runs row (tenant index) → RecruiterBriefWorkflow (LangGraph)
                 → PostgresSaver checkpoints → interrupt → reviewer resume → workflow_reviews
```

| Capability | Files |
|---|---|
| UI shell, mode detection, navigation | `app.py`, `frontend/context.py`, `frontend/components.py` |
| Typed HTTP client | `frontend/api_client.py`, contracts in `api/schemas.py` |
| Pages | `frontend/views/{overview,library,ranking,search,ask,briefs,history}.py` |
| Deterministic ranking | `services/ranking.py`, `utils/{scoring,matcher,skills_extractor,candidate_analyzer}.py` |
| Auth, roles, policy | `api/auth.py`, `retrieval/authorization.py`, `api/policy.py`, `api/dependencies.py` |
| Document persistence | `repositories/documents.py`, `db/models.py`, `migrations/versions/0001…0003` |
| Retrieval | `retrieval/{pipeline,fusion,reranker,context,contracts}.py`, `services/embeddings.py` |
| Grounded generation | `generation/{service,prompts,schemas,citations,providers}.py`, `api/presenters.py` |
| Review workflow | `workflows/recruiter_brief.py`, `workflows/checkpointing.py`, `repositories/workflow_{runs,reviews}.py` |

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
4. **Synchronous processing (medium):** parsing, embedding, reranking, and
   generation run in request threads; large batches block the UI request.
5. **Parser and provider risk (medium):** no malware scanning or OCR isolation;
   live provider behavior, cost, and retention are unvalidated in this repository.
6. **Supply chain (medium):** dependencies are range-constrained, not locked;
   models download at runtime into a cache volume.

## Deliberately out of scope

No UI rewrite, autonomous ranking agent, multi-agent system, automatic hiring
actions, brief editing, knowledge graph, Elasticsearch, separate vector database,
Redis/Celery/Kafka, OCR rollout without measured need, or broad skill ontology.
PostgreSQL + pgvector is sufficient until measured scale or relevance demands otherwise.
