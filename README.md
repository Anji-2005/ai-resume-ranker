# Resume Review Assistant

An explainable, human-in-the-loop Streamlit application that helps recruiters
prioritize resumes against explicitly configured job criteria. It combines
semantic similarity with required skills, preferred skills, and experience while
showing the evidence and score breakdown behind every result.

> This application is a decision-support prototype. It does not make hiring
> decisions and should never be used as the sole basis for an employment action.

## Features

- Lazy, cached Sentence Transformer loading with visible progress
- Weighted and fully visible score components
- Required and preferred skill criteria with alias-aware matching
- Resume evidence snippets for matched skills
- PDF and DOCX support with signature, size, and document limits
- Detection of scanned or text-empty PDFs
- Per-file error isolation for batch processing
- Blind-review mode for common direct identifiers
- Candidate filtering and side-by-side comparison
- CSV, Excel, and JSON audit exports
- Optional Gemini-generated candidate summaries and personalized resume suggestions
- Optional Gemini-generated interview questions and requirement evidence explanations
- Explicit methodology, safeguards, and limitations
- Automated unit and Streamlit rendering tests
- Versioned document/chunk provenance, PostgreSQL/pgvector persistence, native
  PostgreSQL lexical search, dense retrieval, RRF fusion, and optional reranking
- Citation-verified structured generation and one bounded LangGraph human-review workflow
- FastAPI document API with fail-closed token-to-tenant authorization
- Offline quality gates, privacy-safe traces, SBOM generation, and dependency auditing
- Container access logs disabled so confidential query strings are not persisted

## Architecture and technology

The unchanged Streamlit flow remains the deterministic review UI. The V2 service
path uses FastAPI, SQLAlchemy/Alembic, PostgreSQL 16 with pgvector, MiniLM
embeddings, PostgreSQL native full-text ranking, reciprocal-rank fusion,
LangChain structured Gemini integration, and one LangGraph workflow. Documents
flow through validation → provenance parsing → section-aware chunking → embedding
and indexing → authorized retrieval → optional citation-verified generation →
human review. See [ARCHITECTURE.md](ARCHITECTURE.md) for boundaries and caveats.

## Scoring

The default score is composed of:

| Component | Default weight | Meaning |
|---|---:|---|
| Semantic similarity | 40% | Overall relationship between the job description and resume |
| Required skills | 35% | Percentage of configured required skills found |
| Preferred skills | 10% | Percentage of configured preferred skills found |
| Experience | 15% | Explicitly stated years relative to the configured minimum |

Weights are configurable and normalized automatically. If preferred skills or a
minimum-experience requirement are not configured, their weights are removed and
the remaining components are renormalized. Skill mentions demonstrate textual
evidence, not proficiency; a human reviewer must inspect the supporting context.

## Local installation

Python 3.10–3.14 is recommended.

```bash
cd /path/to/ai-resume-ranker
python3 -m venv ../ai-resume-ranker-venv
source ../ai-resume-ranker-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. The language model is downloaded when the first
analysis starts and cached for later use; the initial download requires internet
access.

## Local document API and PostgreSQL

The existing Streamlit app remains standalone. The optional document-intelligence
API adds tenant-scoped, versioned document storage and retrieval without changing
the current scoring path.

```bash
cp .env.example .env
# Edit .env: choose a unique local POSTGRES_PASSWORD and retrieval token.
docker compose up --build
```

This starts PostgreSQL, applies the migration, and serves the API at
`http://localhost:8000`; interactive OpenAPI documentation is at `/docs`.
The default Compose file deliberately has no source bind mount or auto-reload.
For local API editing only, use `docker compose -f docker-compose.yml -f
docker-compose.dev.yml up --build`. `.env` is ignored by Git and secrets are
passed at container runtime, never copied into the image.
The API image installs `torch==2.14.1+cpu` from PyTorch's official CPU wheel
index; local macOS installation continues to use `requirements.txt` normally.
Check service health with `docker compose ps` and logs with
`docker compose logs --tail=100 api postgres`.

The focused API requires `X-Tenant-ID` and `Authorization: Bearer <token>` on
every document operation:

- `POST /v1/documents` — multipart `file` plus stable opaque `source_key`.
  Repeating identical bytes for the same tenant/source key is idempotent; changed
  bytes create a new immutable document version.
- `GET /v1/documents/search?query=...&limit=10` — tenant-filtered hybrid
  full-text + pgvector retrieval returning chunks with source offsets and pages.
- `DELETE /v1/documents/{document_id}` — tenant-filtered physical deletion of
  document versions and vectors through cascading foreign keys.
- `POST /v1/documents/recruiter-brief` — authorized, citation-verified review
  claims, requirement evidence, and interview questions.
- `POST /v1/documents/qa` — authorized, citation-verified document Q&A.

The local fail-closed adapter checks `RETRIEVAL_AUTH_TOKENS_JSON`, for example
`{"local-demo-token":["demo-tenant"]}`. This is a development adapter only;
production must replace it with authenticated principal/tenant claims from the
deployment identity provider. The requested `X-Tenant-ID` is never accepted as
retrieval authority by itself.

Grounded generation uses the optional `langchain-google-genai` adapter for
Pydantic structured output, then verifies every model citation locally against
authorized retrieved chunks. The original direct `google-genai` enrichment path
in `utils/genai_analyzer.py` remains the Streamlit compatibility baseline and
its deterministic fallbacks remain unchanged. Generated material is limited to
human review support: it cannot alter deterministic scores/ranks or make hiring
decisions.

## Controlled recruiter-brief workflow

`workflows/recruiter_brief.py` defines one bounded LangGraph workflow:
authorize → retrieve permitted evidence → validate job requirements → generate
brief → verify citations → interrupt for human brief review → persist the
reviewed brief. It has no autonomous hiring or rejection branch and does not
change deterministic scoring. Existing Streamlit review remains its compatible,
in-session path; the workflow is available to service/API integrations that need
durable review sessions.

Production workflow runs must use `workflows.checkpointing.postgres_checkpointer`
with `DATABASE_URL`; it initializes LangGraph's PostgreSQL checkpoint tables and
uses a stable `thread_id` to resume review. `LANGGRAPH_STRICT_MSGPACK=true` is
set by default to restrict checkpoint deserialization. `InMemorySaver` is used
only by tests. Human review resumes with `Command(resume={"approved": true,
"note": "..."})`; the review record is idempotent by `(tenant_id, workflow_id)`.

Environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy PostgreSQL connection URL for API/Alembic | required outside Compose |
| `DATABASE_POOL_SIZE` | Persistent connection pool size | `5` |
| `DATABASE_MAX_OVERFLOW` | Temporary connection overflow | `10` |
| `DATABASE_ECHO` | Emit SQL for local debugging only | `false` |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT` | Docker Compose PostgreSQL settings | values in `.env.example` |
| `TEST_DATABASE_URL` | Isolated pgvector database used only by integration tests | unset; tests skip |
| `RETRIEVAL_AUTH_TOKENS_JSON` | JSON token-to-permitted-tenant mapping for local search authorization | `{}` (fails closed) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Existing optional enrichment configuration | unchanged |
| `GENAI_INPUT_USD_PER_MILLION_TOKENS`, `GENAI_OUTPUT_USD_PER_MILLION_TOKENS` | Optional trace cost-rate inputs; estimates use characters ÷ 4 | unset, so cost is reported as unavailable |

Never point `TEST_DATABASE_URL` at a shared or production database: the suite
truncates its document tables.

## Retrieval evaluation

`data/synthetic_retrieval.json` is a small labelled regression dataset. Run the
preserved whole-document cosine baseline with
`python -m evals.run_whole_document_baseline`. Run the real PostgreSQL native
full-text + pgvector + RRF evaluator only after starting the database with
`python -m evals.run_postgres_retrieval_eval`. The latter intentionally calls
the production pipeline rather than claiming PostgreSQL is BM25. Current
executed results and limitations are in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

## Offline quality gates and observability

The repository has no live-provider benchmark in CI. Instead, it executes two
small deterministic offline checks: the original whole-document cosine matcher
against `data/synthetic_retrieval.json`, and local citation verification against
`data/generation_eval.json`. Run them, then enforce the checked-in regression
thresholds, with:

```bash
python -m evals.run_whole_document_baseline
python -m evals.run_generation_eval
python -m evals.run_offline_quality_eval
python -m evals.quality_gate
```

`observability/tracing.py` emits structured generation records containing the
operation, prompt version, model name, latency, character counts, outcome, and
an optional cost estimate. It intentionally does not emit prompts, resumes, or
provider responses. Direct identifiers and common secret-bearing values are
redacted before a trace value can be logged. Cost stays `null` unless the active
provider's rates are explicitly configured; character-to-token conversion is an
estimate, not billing data. GitHub Actions runs linting, unit tests, these
offline evaluations, and their regression gate. It does not run a live Gemini
call or shared database integration test.

Generate the same supply-chain checks used by CI with:

```bash
python -m pip_audit --cache-dir /tmp/pip-audit --local
cyclonedx-py requirements requirements.txt --output-file sbom.cdx.json
```

For a disposable PostgreSQL evaluation database, set both `TEST_DATABASE_URL`
and `DATABASE_URL` to that isolated database, apply `alembic upgrade head`, run
`pytest -m integration -q`, then run
`python -m evals.run_postgres_retrieval_eval`. Never point these commands at a
shared or production database.

## Rollback procedure

Before a controlled pilot, record the prior image digest and database revision.
If an authorization, privacy, migration, or quality threshold regresses: stop
new intake, route reviewers back to the unchanged standalone Streamlit path,
redeploy the prior image digest, and restore the previous application revision.
Do not downgrade or restore PostgreSQL until a tested backup exists and the
migration's downgrade/data-loss behavior has been reviewed. Production rollout
thresholds and rollback authority still require an identified human owner.

## Optional Gemini enrichment

The app works without Gemini and retains its deterministic scoring and rule-based
summary behavior. To enable the four optional GenAI features, create a Gemini API
key in [Google AI Studio](https://aistudio.google.com/apikey), copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, and replace the
placeholder value. Alternatively, set the `GEMINI_API_KEY` environment variable.

The default model is `gemini-3.1-flash-lite`. Override it with the `GEMINI_MODEL`
environment variable if needed. Model availability, free-tier limits, and pricing
can change; check Google's current Gemini API documentation before a demonstration.

One Gemini request per candidate produces a contextual summary, personalized
improvement suggestions, three interview questions, and an evidence explanation
for every required skill. Results are cached in process using a hash of the
anonymized input and job configuration, so Streamlit reruns do not repeat calls.
Gemini never changes deterministic scores, ranks, experience values, or review
priorities. If the key is missing, the API is unavailable, or its response is
invalid, the existing rule-based summary and suggestions remain available.

Only anonymized text is sent when blind-review mode is enabled. Inputs longer than
30,000 resume characters or 12,000 job-description characters are bounded for the
request by keeping the beginning and end. Free-tier data may be used by Google to
improve its products. Use fictional or appropriately anonymized resumes for demos,
and verify all generated content against the original source.

## Testing

The core suite uses Python's built-in test runner:

```bash
python -m unittest discover -s tests -v
```

For a development environment with pytest:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

To run the real PostgreSQL/pgvector integration suite after creating an isolated
test database and applying migrations to it:

```bash
export TEST_DATABASE_URL='postgresql+psycopg://resume:replace-with-local-test-password@localhost:5432/resume_intelligence_test'
DATABASE_URL="$TEST_DATABASE_URL" alembic upgrade head
pytest -m integration -q
```

## Structured parsing API

`extract_text_from_bytes(filename, data)` remains the compatibility API for
callers that need plain extracted text. New document-intelligence code should
use `parse_document_from_bytes(filename, data)`, which returns a versioned
`ParsedDocument` with section-aware chunks, exact offsets into normalized text,
and provenance. PDF sections retain their actual page number; DOCX sections and
chunks use `page_number=None` because the format parser does not provide stable
pagination. Chunk size and overlap are configurable per call.

## Privacy and security behavior

- The standalone Streamlit path processes uploads in memory. The optional V2
  API persists normalized document JSON, chunks, embeddings, source filename,
  and provenance in PostgreSQL until an authorized deletion; original upload
  bytes are not stored by the repository.
- Valid Gemini enrichments are cached only in application memory. The V2 API's
  database content must be treated as confidential candidate data.
- Supported uploads are PDF and DOCX, limited to 10 MB.
- PDF signatures, page counts, DOCX structure, and expanded DOCX size are checked.
- Blind-review mode redacts common emails, phone numbers, URLs, and likely header names.
- Blind-review progress, failures, result metadata, and audit exports substitute
  opaque per-run candidate labels for source filenames.
- Blind review cannot guarantee removal of every identity or demographic proxy.
- A production deployment should add authentication, encrypted persistence only if
  necessary, retention controls, malware scanning, privacy-safe audit events, and an
  incident process.

## Known limitations

- Scanned PDFs are detected but OCR is not bundled because reliable OCR requires an
  additional system runtime or managed service.
- Experience extraction intentionally uses explicit phrases such as “5 years of
  experience”; it does not infer duration from employment dates.
- The included skill taxonomy is deterministic and extensible, but not exhaustive.
- Semantic embedding models can reproduce biases from their training data.
- The app has not been validated for autonomous or high-stakes employment decisions.
- Gemini output can be incomplete or incorrect and always requires human verification.
- The CI/evaluation datasets are intentionally tiny and synthetic; passing their
  gates is not a production-readiness, fairness, load, or security certification.
- There is no background job queue; parsing, embedding, retrieval, and provider
  calls execute synchronously in the request/UI process.
- The token-map authorization adapter is local-development infrastructure, not
  production identity, RBAC, purpose limitation, or audit access control.

## Readiness conclusion

The repository is ready for a **portfolio demonstration using synthetic data**:
the complete local suite, PostgreSQL integration tests, container smoke flow,
offline quality gates, dependency audit, and SBOM command were executed
successfully on 2026-10-14. See `EVALUATION_RESULTS.md` for the measurements.

It is **not ready for a controlled pilot or production use**. Those stages need
an approved identity/RBAC provider, durable opaque candidate identity, reviewed
retention and deletion controls, malware/OCR isolation, dependency locking,
representative relevance/fairness/load evaluation, live-provider validation,
backup/restore drills, and owner-approved quality/rollback thresholds.

## Project structure

```text
app.py                       Streamlit workflow and presentation
utils/matcher.py             Lazy model loading and semantic similarity
utils/scoring.py             Weighted, explainable scoring
utils/skills_extractor.py    Skill aliases, matching, and string/provenance evidence
utils/pdf_parser.py          Secure PDF/DOCX validation, parsing, chunks, provenance
domain/schemas.py            Versioned Pydantic document/chunk/evidence contracts
utils/privacy.py             Blind-review redaction
utils/candidate_analyzer.py  Experience and review-quality heuristics
utils/genai_analyzer.py      Optional Gemini prompting, validation, and memory cache
tests/                       Core and application smoke tests
```
