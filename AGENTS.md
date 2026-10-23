# Engineering guardrails

## Purpose and scope

This repository is a human-in-the-loop recruitment document review system. It
may prioritize records for review, but must not make hiring decisions. Preserve
the current deterministic scoring contract and Streamlit workflow unless an
approved migration explicitly replaces an interface.

## Non-negotiable rules

- Treat resumes, filenames, extracted text, embeddings, review outcomes, and
  LLM outputs as confidential candidate data.
- Do not send candidate data to an external model, telemetry, or log sink
  without an explicit configured provider, documented purpose, retention terms,
  and a per-tenant policy check.
- Blind review is a boundary, not a display preference: use a stable opaque
  candidate ID and never expose filename, raw text, direct identifiers, or
  identity-bearing metadata in UI, exports, errors, logs, or cache keys.
- Deterministic scores and human decisions are immutable audit facts. LLMs may
  produce review assistance only; they cannot alter scores, ranks, or outcomes.
- Preserve the existing bounded recruiter-brief workflow. V3 may add exactly
  one bounded InvestigationWorkflow using a fixed registry of tenant-scoped,
  read-only evidence tools. It may not create sub-agents, dynamically register
  tools, perform candidate ranking, make employment decisions, or mutate
  candidate documents. All generated notes require verified citations and
  human review.
- Keep parsing, retrieval, scoring, generation, persistence, and HTTP/UI
  adapters separately testable. Business logic must not import Streamlit,
  FastAPI, or database session objects.

## Change requirements

1. Add or update unit tests for every changed rule; add integration tests for
   persistence/API contracts and evaluation cases for LLM changes.
2. Use versioned schemas for persisted documents, score snapshots, and LLM
   structured outputs. Store parser/model/prompt versions and source spans.
3. Bound file size, page count, extracted text, concurrency, timeouts, retries,
   and cache TTL/size. Reject unsupported or encrypted input safely.
4. Require authorization and tenant scoping before any candidate read, export,
   review update, or vector retrieval.
5. Run tests and migration checks before merge. Never commit secrets, resumes,
   generated exports, database dumps, or model caches.

## Compatibility contract

The following behaviors remain authoritative during migration: PDF/DOCX upload,
deterministic skill and score calculations, explainable evidence, optional
Gemini fallback behavior, Streamlit display, and human review/export flows.
New services must match these contracts through contract tests before the UI is
switched to them.
