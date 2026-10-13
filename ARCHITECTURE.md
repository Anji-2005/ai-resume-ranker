# Target architecture: Recruitment Document Intelligence and RAG

## Current state and audit

The repository retains its single-process Streamlit application while adding an
optional V2 FastAPI/PostgreSQL service path. `app.py` still controls the legacy
UI, deterministic analysis, Gemini fallback, session review data, and exports.
The V2 path adds versioned provenance, PostgreSQL/pgvector persistence,
tenant-scoped retrieval, grounded generation services, and one bounded workflow.
There is still no background worker, production identity provider, immutable
score-snapshot/review-event implementation, object-store policy, or production
deployment control plane.

### File-by-file audit

| File | Findings | Migration role |
|---|---|---|
| `app.py` | Strong human-review UI and per-file isolation, but mixes all layers. Session state contains results/review decisions. Blind mode now substitutes candidate labels in progress, failures, result metadata, and audit JSON; exports remain unrestricted downloads. | Retain as UI; replace direct utility calls with API client after parity. |
| `utils/pdf_parser.py` | Validates extension/signature/size, PDF pages, DOCX structure and expanded size. No malware scan, MIME deep inspection, extraction-text cap, ZIP entry/count/compression-ratio caps, OCR/provenance, or async isolation. | Extract to parser service/module returning versioned `ParsedDocument`. |
| `utils/privacy.py` | Redacts email/phone/URL and a heuristic first line, and replaces filenames at blind-review display/export boundaries. It is not reliable de-identification and does not handle headers, addresses, IDs, links in nonstandard form, or demographic proxies. | Replace with a policy-driven redaction pipeline and separate identity mapping. |
| `utils/matcher.py` | Lazy local MiniLM loading and bounded cosine score are simple and deterministic. First-run model download, synchronous batch inference, no embedding version/normalization persistence, and no retrieval contract. | Keep score adapter; add versioned embedding provider and batch interface. |
| `utils/scoring.py` | Clear normalized score contract and optional-criterion behavior. Inputs/results have no typed/versioned schema or immutable score snapshot. | Preserve as scoring domain service; persist `score_policy_version` and components. |
| `utils/skills_extractor.py` | Explainable alias matching and source snippets. Taxonomy is static; sentence splitting and matching can give incomplete/ambiguous evidence. | Retain deterministic extraction; emit evidence spans/page references. |
| `utils/candidate_analyzer.py` | Conservative explicit-years logic and rule fallbacks. “Red flags” may be over-read as negative signals and lack provenance. | Keep as advisory only; return labeled heuristics with evidence. |
| `utils/genai_analyzer.py` | Bounded prompts, JSON schema, output validation, injection instruction, exact-string evidence check, and in-memory LRU are good safeguards. Cache is process-global with no tenant/retention/policy dimension; no provider audit or evaluation harness. | Move behind a structured-generation port with policy gate and evaluation set. |
| `tests/test_core.py` | Covers parsing basics, privacy basics, score contract, skill evidence. | Expand with adversarial parsing/redaction and provenance tests. |
| `tests/test_genai.py` | Good validation/fallback/cache coverage with fakes. | Convert fixtures into a versioned offline evaluation dataset. |
| `tests/test_app.py` | Basic rendering smoke test only. | Preserve; add API/UI parity and blind-export tests. |
| `requirements*.txt` | Fully pinned core runtime but lacks service/database/migration/test-quality tooling. | Split runtime extras and lock resolved dependencies. |
| `.streamlit/*` | Upload cap and secret example exist; no deployment auth, origin, or identity configuration. | Keep local configuration; deploy behind authenticated ingress. |

## Primary risks

1. **Blind-review identity limits (high):** filename leakage in progress, failure paths, and audit JSON is now blocked and regression-tested. Candidate labels remain session/order dependent rather than durable opaque IDs, and the redactor is not comprehensive enough to guarantee de-identification.
2. **Incomplete anonymization (high):** regex-based redaction is not a guarantee and raw text may still include identities, address/contact variants, document properties, and proxy attributes. It is passed to Gemini after only this limited transformation.
3. **No access/retention controls (high):** anyone able to use the deployed Streamlit process can upload, inspect, export, and retain session/browser copies. Process-global LLM cache lacks tenant scoping, TTL, purge/audit events, or policy metadata.
4. **Audit/data consistency (high):** review decisions mutate in-memory result dictionaries and disappear on restart. Exports can be stale, filtered or unfiltered differently, and lack immutable event/version identifiers. Rank can change without a reproducible score/input snapshot.
5. **Parser and availability risk (medium):** parsing/inference/LLM calls run synchronously in the UI. File checks are helpful but do not provide malware scanning, extraction output caps, comprehensive archive-bomb controls, OCR queueing, or resource isolation.
6. **Retrieval and evidence limits (medium):** there is no durable document/chunk provenance, hybrid retrieval, reranking, embedding versioning, or quality measurement. Current evidence is text snippets with no page/character span.
7. **Model/supply-chain/operations risk (medium):** initial model download occurs at runtime and dependencies are range-constrained rather than locked. CI now runs vulnerability scanning, SBOM generation, tests, and offline gates, but there are no production deployment controls or representative load measurements.

## Proposed target architecture

```text
Streamlit UI (existing flow) ── HTTPS ──> FastAPI application
                                      │
                         AuthN/AuthZ + tenant/purpose policy
                                      │
              ┌──────────── document workflow (one LangGraph) ────────────┐
              │ validate → parse/OCR → redact policy → chunk/provenance    │
              │ → embed/index → deterministic score → optional generation │
              └───────────────────────────────────────────────────────────┘
                 │                 │                         │
          object store*       PostgreSQL + pgvector       model providers
          originals optional   metadata, reviews, events  embeddings/LLM
```

`*` Store originals only if a reviewed retention policy requires it; otherwise
store derived, minimized text/provenance. Encrypt objects and database storage,
use per-tenant access controls, lifecycle deletion, and access/audit events.

## Implemented persistence slice

`db/models.py`, `repositories/documents.py`, and Alembic revision
`0001_document_intelligence` now implement the document/version/chunk store.
Each document is unique by `(tenant_id, source_key)`; identical content hashes
are idempotent, while changed bytes create a new version and make it the only
retrieval-active version. Chunks retain parser provenance, a MiniLM-compatible
384-dimensional pgvector embedding, and PostgreSQL full-text index. Retrieval
filters tenant and active documents before reciprocal-rank-fusing lexical and
cosine results. Deletion is tenant-filtered and physical, relying on cascading
foreign keys to remove versions and vectors.

### Data contracts

- `Document`: tenant ID, opaque candidate ID, content hash, intake metadata,
  policy/consent state, lifecycle state. Never use a filename as identity.
- `ParsedDocument`: parser version, source digest, normalized text, pages, and
  chunk-level provenance (`page`, character offsets, extractor confidence).
- `ScoreSnapshot`: immutable job criteria/version, deterministic component
  values, matched/missing skills, evidence references, embedding model version,
  and timestamp.
- `ReviewEvent`: append-only actor, action, rationale, source score snapshot,
  and timestamp. Project current review state from events.
- `Generation`: prompt/schema/model versions, policy decision, structured
  result, source chunk IDs, validation status, TTL/retention class.

### Retrieval and generation

Use PostgreSQL native full-text ranking via `tsvector` plus
pgvector cosine search over provenance-bearing chunks. Apply metadata filters
(tenant, document, policy state) before retrieval; fuse results with reciprocal
rank fusion and rerank only the small candidate set with a cross-encoder. The
RAG context must contain cited chunks, not whole resumes. LangChain supplies
typed structured output and provider adapters; LangGraph represents one bounded,
observable workflow with deterministic nodes and a human-review interruption
point. It is orchestration, not an agent system.

### Implemented retrieval boundary

The `retrieval/` package now enforces authorization before embedding or database
access, then executes dense pgvector and PostgreSQL native `tsvector` searches
with tenant/current-version/metadata filters. It combines rank positions with
reciprocal-rank fusion; no part of the code describes PostgreSQL ranking as
BM25. Optional cross-encoder reranking is bounded to the fused candidate set.
Context selection keeps database UUID source chunk IDs plus parser chunk keys,
so downstream generation can cite exact source chunks.

### Grounded generation boundary

`generation/` separates provider invocation, prompts, Pydantic output schemas,
and deterministic citation verification. The LangChain Gemini adapter provides
structured invocation only; it is not trusted to validate its own evidence.
Recruiter briefs, requirement analysis, interview questions, and document Q&A
are accepted only when citations refer to authorized retrieved chunk UUIDs and
their quotes are exact excerpts. Generated content is never a scoring input and
cannot produce a hiring decision.

### Controlled workflow boundary

`workflows/recruiter_brief.py` is a single LangGraph state machine, not a
multi-agent system. It has explicit serialized state, authorization before
retrieval, bounded generation retries, safe recovery states, citation checking,
and a dynamic human-review interrupt. Production checkpoints use PostgreSQL's
`PostgresSaver`; reviewed briefs are separately persisted idempotently in
`workflow_reviews`. The interruption happens before persistence so replay or
duplicate delivery cannot autonomously alter a candidate outcome.

### Evaluation, tracing, and release boundary

`evals/` contains version-controlled synthetic retrieval and citation-validation
datasets. The existing whole-document cosine matcher remains the retrieval
baseline; `run_generation_eval` is strictly an offline verification test and
does not claim to evaluate LLM quality. `quality_gate.py` applies narrow,
visible thresholds to the measurements written by those commands. CI runs lint,
tests, and the offline gate without provider credentials.

`observability/tracing.py` records generation operation, prompt version, model
identifier, latency, character counts, optional configured-rate cost estimate,
and outcome/error class. It never records raw prompt, resume, or response text.
Identifiers and common secret patterns are redacted before trace values are
emitted. Trace data should still be treated as sensitive operational metadata;
deployment must configure retention, access control, and a central sink.
Container HTTP access logs are disabled because request query strings can contain
confidential recruitment criteria; use the structured redacted trace sink instead.

The baseline Docker Compose configuration is immutable at runtime (no source
bind mount/reload) and requires a local PostgreSQL password from ignored `.env`.
The separate `docker-compose.dev.yml` explicitly opts into a development bind
mount. Image tags and Python dependencies are version-constrained, but an
artifact digest and dependency lock/SBOM are still required before any
production deployment claim.

The Linux API image resolves PyTorch from the official CPU-only wheel index and
constrains the exact `+cpu` build before resolving application dependencies.
Compose health-checks PostgreSQL with `pg_isready` and FastAPI through its local
OpenAPI endpoint. All document operations require both a tenant header and the
development bearer-token authorization adapter; that adapter must be replaced
by deployment identity claims before any controlled pilot.

## Deliberately out of scope initially

No UI rewrite, autonomous ranking agent, multi-agent system, automatic hiring
actions, knowledge graph, Elasticsearch, separate vector database, OCR rollout
without measured need, or broad skill ontology platform. PostgreSQL + pgvector
is sufficient until measured scale/relevance demands otherwise.
