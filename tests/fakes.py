"""Shared in-memory test doubles. Synthetic data only; no network, database, or LLM."""

from datetime import datetime, timezone
import hashlib
import json
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

from docx import Document

from repositories.documents import DocumentSummaryRow, DocumentTooLargeError, IngestionResult, VersionRow
from investigations.schemas import ToolInvocation, ToolModelResponse
from retrieval.contracts import RetrievalCandidate
from utils.pdf_parser import parse_document_from_bytes


def synthetic_docx(*paragraphs):
    output = BytesIO()
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(output)
    return output.getvalue()


class FakeEmbedder:
    model_name = "fake-embedder-v1"
    model_version = "1.0-test"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.0] * 384 for _ in texts]


class FakeSession:
    def flush(self):
        pass

    def rollback(self):
        pass


class InMemoryDocumentRepository:
    """Implements the DocumentRepository surface used by the API and retrieval pipeline."""

    def __init__(self, embedder=None):
        self.embedder = embedder or FakeEmbedder()
        self.session = FakeSession()
        self.documents = {}
        self.ingested_filenames = []

    def ingest_bytes(self, tenant_id, source_key, filename, data, max_chunks=None):
        parsed = parse_document_from_bytes(filename, data)
        if max_chunks is not None and len(parsed.chunks) > max_chunks:
            raise DocumentTooLargeError(f"document exceeds the {max_chunks}-chunk ingestion limit")
        self.ingested_filenames.append(filename)
        digest = hashlib.sha256(data).hexdigest()
        document = next((d for d in self.documents.values()
                         if d.tenant_id == tenant_id and d.source_key == source_key), None)
        if document is None:
            document = SimpleNamespace(
                id=uuid4(), tenant_id=tenant_id, source_key=source_key,
                created_at=datetime.now(timezone.utc), versions=[],
            )
            self.documents[document.id] = document
        known = next((v for v in document.versions if v.content_hash == digest), None)
        if known is not None:
            return IngestionResult(document.id, known.id, known.version_number, False, self.embedder.model_name)
        for version in document.versions:
            version.is_current = False
        version = SimpleNamespace(
            id=uuid4(), version_number=len(document.versions) + 1, content_hash=digest, is_current=True,
            media_type=parsed.media_type, parser_name=parsed.parser_name, parser_version=parsed.parser_version,
            embedding_model=self.embedder.model_name,
            embedding_model_version=self.embedder.model_version,
            created_at=datetime.now(timezone.utc), chunks=[],
        )
        for chunk in parsed.chunks:
            version.chunks.append(SimpleNamespace(
                id=uuid4(), chunk_key=chunk.chunk_id, section_id=chunk.section_id, page_number=chunk.page_number,
                source_start=chunk.source_start, source_end=chunk.source_end, content=chunk.text,
            ))
        document.versions.append(version)
        return IngestionResult(document.id, version.id, version.version_number, True, self.embedder.model_name)

    def _tenant_documents(self, tenant_id):
        return [d for d in self.documents.values() if d.tenant_id == tenant_id]

    def get_document(self, tenant_id, document_id):
        document = self.documents.get(document_id)
        return document if document is not None and document.tenant_id == tenant_id else None

    def list_documents(self, tenant_id, limit=100, offset=0):
        rows = []
        for document in self._tenant_documents(tenant_id):
            current = next(v for v in document.versions if v.is_current)
            rows.append(DocumentSummaryRow(
                document.id, document.created_at, document.versions[-1].created_at, current.version_number,
                len(document.versions), current.media_type, len(current.chunks),
            ))
        return rows[offset:offset + limit], len(rows)

    def list_versions(self, tenant_id, document_id):
        document = self.get_document(tenant_id, document_id)
        if document is None:
            return []
        return [VersionRow(
            version_id=v.id, version_number=v.version_number, is_current=v.is_current,
            media_type=v.media_type, parser_name=v.parser_name, parser_version=v.parser_version,
            embedding_model=v.embedding_model, embedding_model_version=v.embedding_model_version,
            content_hash=v.content_hash, chunk_count=len(v.chunks), created_at=v.created_at,
        ) for v in reversed(document.versions)]

    def get_version_chunks(self, tenant_id, document_id, version_number, limit=500):
        document = self.get_document(tenant_id, document_id)
        if document is None:
            return []
        version = next((v for v in document.versions if v.version_number == version_number), None)
        return list(version.chunks) if version else []

    def existing_document_ids(self, tenant_id, document_ids):
        return {d.id for d in self._tenant_documents(tenant_id) if d.id in set(document_ids)}

    def delete_document(self, tenant_id, document_id):
        if self.get_document(tenant_id, document_id) is None:
            return False
        del self.documents[document_id]
        return True

    def _candidates(self, tenant_id, filters, query=None):
        results = []
        for document in self._tenant_documents(tenant_id):
            if filters.document_ids and document.id not in filters.document_ids:
                continue
            version = next(v for v in document.versions if v.is_current)
            for chunk in version.chunks:
                if query and not any(word.casefold() in chunk.content.casefold() for word in query.split()):
                    continue
                results.append(RetrievalCandidate(
                    document_id=document.id, document_version_id=version.id, chunk_id=chunk.id,
                    chunk_key=chunk.chunk_key, section_id=chunk.section_id, content=chunk.content,
                    source_start=chunk.source_start, source_end=chunk.source_end, page_number=chunk.page_number,
                    source_key=document.source_key, media_type=version.media_type,
                ))
        return results

    def search_dense(self, tenant_id, vector, filters, limit):
        return self._candidates(tenant_id, filters)[:limit]

    def search_lexical(self, tenant_id, query, filters, limit):
        return self._candidates(tenant_id, filters, query)[:limit]


class InMemoryRunRepository:
    def __init__(self):
        self.runs = {}

    def create(self, *, tenant_id, workflow_id, created_by, document_ids):
        now = datetime.now(timezone.utc)
        run = SimpleNamespace(
            tenant_id=tenant_id, workflow_id=workflow_id, created_by=created_by,
            document_ids=[str(value) for value in document_ids], status="running", error_code=None,
            outcome=None, created_at=now, updated_at=now,
        )
        self.runs[workflow_id] = run
        return run

    def get_for_tenant(self, tenant_id, workflow_id, *, for_update=False):
        run = self.runs.get(str(workflow_id))
        return run if run is not None and run.tenant_id == tenant_id else None

    def list_for_tenant(self, tenant_id, *, status=None, limit=100):
        return [run for run in self.runs.values() if run.tenant_id == tenant_id and (not status or run.status == status)]

    def update_status(self, run, *, status, error_code=None, outcome=None):
        run.status, run.error_code, run.outcome = status, error_code, outcome
        run.updated_at = datetime.now(timezone.utc)
        return run


class InMemoryIngestionJobRepository:
    def __init__(self):
        self.jobs = {}
        self.session = FakeSession()

    def enqueue(self, **values):
        existing = self.get_by_key(values["tenant_id"], values["idempotency_key"])
        if existing:
            from repositories.ingestion_jobs import JobIdempotencyConflict
            if existing.request_hash != values["request_hash"]:
                raise JobIdempotencyConflict("conflict")
            return existing, False
        now = datetime.now(timezone.utc)
        fields = {
            "id": uuid4(), "status": "queued", "attempt_count": 0,
            "available_at": now, "queued_at": now, "started_at": None, "completed_at": None,
            "lease_expires_at": None, "worker_id": None, "error_code": None,
            "processing_latency_ms": None, "result_document_id": None, "result_version_id": None,
            "result_version_number": None, "result_created": None, "target_document_id": None,
            "target_version_number": None,
        }
        fields.update(values)
        job = SimpleNamespace(**fields)
        self.jobs[job.id] = job
        return job, True

    def get_by_key(self, tenant_id, key):
        return next((job for job in self.jobs.values()
                     if job.tenant_id == tenant_id and job.idempotency_key == key), None)

    def get_for_tenant(self, tenant_id, job_id):
        job = self.jobs.get(job_id)
        return job if job and job.tenant_id == tenant_id else None

    def list_for_tenant(self, tenant_id, *, status=None, limit=100):
        rows = [job for job in self.jobs.values()
                if job.tenant_id == tenant_id and (not status or job.status == status)]
        return list(reversed(rows[-limit:]))

    def metrics(self, tenant_id, *, now=None):
        rows = self.list_for_tenant(tenant_id, limit=200)
        return {
            "counts": {state: sum(job.status == state for job in rows)
                       for state in ("queued", "processing", "completed", "failed")},
            "completed_last_hour": sum(job.status == "completed" for job in rows),
            "failed_last_hour": sum(job.status == "failed" for job in rows),
            "throughput_per_minute_last_hour": sum(job.status == "completed" for job in rows) / 60,
            "mean_processing_latency_ms": None,
            "p95_processing_latency_ms": None,
            "oldest_queued_age_seconds": None,
        }


class InMemoryReviewRepository:
    def __init__(self):
        self.records = {}

    def upsert(self, *, tenant_id, workflow_id, subject, brief, approved, note):
        key = (tenant_id, workflow_id)
        if key in self.records:
            return self.records[key], False
        record = SimpleNamespace(
            tenant_id=tenant_id, workflow_id=workflow_id, subject=subject, reviewed_brief=brief,
            reviewer_approved=approved, reviewer_note=note, reviewed_at=datetime.now(timezone.utc),
        )
        self.records[key] = record
        return record, True

    def get(self, *, tenant_id, workflow_id):
        return self.records.get((tenant_id, workflow_id))


class CitingProvider:
    """Fake structured-output provider that cites the first supplied source exactly."""

    model_name = "fake-provider-v1"

    def __init__(self, quote_from_source=True, fail=False):
        self.quote_from_source = quote_from_source
        self.fail = fail
        self.prompts = []

    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("provider exploded with SECRET-RESUME-TEXT")
        payload = json.loads(prompt.split("SOURCE_DATA_JSON:\n", 1)[1])
        source_id = payload["available_source_chunk_ids"][0]
        excerpt = payload["retrieved_context"].split(f"[source_chunk_id={source_id}]\n", 1)[1]
        excerpt = excerpt.split("\n\n[source_chunk_id=", 1)[0]
        quote = excerpt.strip()[:40] if self.quote_from_source else "Invented quote not in source"
        citation = {"source_chunk_id": source_id, "quote": quote}
        if schema.__name__ == "GroundedAnswer":
            return {"claims": [{"text": "The document mentions this evidence.", "citations": [citation]}]}
        return {
            "claims": [{"text": "The document describes relevant work.", "citations": [citation]}],
            "requirement_evidence": [
                {"requirement": "Python", "status": "supported", "explanation": "Directly stated.", "citations": [citation]},
                {"requirement": "Kubernetes", "status": "unsupported",
                 "explanation": "No supporting evidence was retrieved."},
            ],
            "interview_questions": [{"question": "Describe the project.", "reason": "Verify the evidence.",
                                     "citations": [citation]}],
        }


class InvestigationToolProvider:
    """Adaptive fake tool-calling model for API/UI tests; no network or hidden fixed workflow."""

    model_name = "fake-investigation-tool-model-v1"

    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def invoke_tools(self, prompt, tools):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("provider failure containing confidential text")
        payload = json.loads(prompt.split("UNTRUSTED_INVESTIGATION_DATA_JSON:\n", 1)[1])
        results = payload["tool_results"]
        if not results:
            invocation = ToolInvocation(
                id="search-1", name="search_evidence",
                arguments={"query": payload["objective"], "limit": 5},
            )
        elif payload["explicit_job_requirements"] and not any(
                item["tool_name"] == "check_requirements" for item in results):
            invocation = ToolInvocation(
                id="requirements-1", name="check_requirements",
                arguments={"requirements": payload["explicit_job_requirements"]},
            )
        else:
            search = next((item for item in results if item["tool_name"] == "search_evidence"), None)
            sources = search["output"].get("sources", []) if search else []
            evidence_action = search["action_id"] if search else "search-1"
            supported = []
            gaps = []
            questions = []
            citation = None
            if sources:
                excerpt = sources[0]["excerpt"]
                quote = excerpt.strip()[:min(40, len(excerpt.strip()))]
                citation = {"source_chunk_id": sources[0]["source_chunk_id"], "quote": quote}
                supported = [{"text": "The authorized document contains relevant technical evidence.",
                              "citations": [citation]}]
                questions = [{"question": "Can the reviewer verify the depth of this experience?",
                              "rationale": "The cited excerpt is concise.", "citations": [citation]}]
            else:
                gaps = [{"topic": "Requested evidence", "explanation": "No evidence was found.",
                         "checked_action_ids": [evidence_action]}]
            matrix = []
            for requirement in payload["explicit_job_requirements"]:
                if citation and requirement.casefold() in sources[0]["excerpt"].casefold():
                    matrix.append({"requirement": requirement, "status": "supported",
                                   "explanation": "The requirement is stated in the cited excerpt.",
                                   "citations": [citation], "checked_action_ids": []})
                else:
                    matrix.append({"requirement": requirement, "status": "insufficient",
                                   "explanation": "Insufficient evidence was found for this requirement.",
                                   "citations": [], "checked_action_ids": [evidence_action]})
            invocation = ToolInvocation(id="draft-1", name="draft_investigation", arguments={"report": {
                "supported_findings": supported,
                "missing_evidence": gaps,
                "contradictory_evidence": [],
                "requirement_evidence": matrix,
                "verification_questions": questions,
            }})
        return ToolModelResponse(
            calls=[invocation], input_tokens=100, output_tokens=50, cost_usd=0.001,
        )


class InMemoryInvestigationRepository:
    def __init__(self):
        self.runs = {}
        self.events = {}
        self.reviews = {}
        self.session = FakeSession()

    def create(self, **values):
        now = datetime.now(timezone.utc)
        run = SimpleNamespace(**values, created_at=now, updated_at=now)
        self.runs[run.id] = run
        return run

    def commit_started(self, run):
        return None

    def get_for_tenant(self, tenant_id, investigation_id, *, for_update=False):
        run = self.runs.get(investigation_id)
        return run if run is not None and run.tenant_id == tenant_id else None

    def get_by_idempotency_key(self, tenant_id, key):
        return next((run for run in self.runs.values()
                     if run.tenant_id == tenant_id and run.idempotency_key == key), None)

    def list_for_tenant(self, tenant_id, *, status=None, limit=100):
        rows = [run for run in self.runs.values()
                if run.tenant_id == tenant_id and (not status or run.status == status)]
        return sorted(rows, key=lambda row: row.updated_at, reverse=True)[:limit]

    def update_status(self, run, *, status, error_code=None, outcome=None):
        run.status, run.error_code, run.outcome = status, error_code, outcome
        run.updated_at = datetime.now(timezone.utc)
        return run

    def sync_state(self, run, values, *, status, error_code=None, outcome=None):
        self.update_status(run, status=status, error_code=error_code, outcome=outcome)
        run.document_scope = values.get("scope_documents", run.document_scope)
        run.report = values.get("report") or None
        run.sources = values.get("evidence", [])
        run.elapsed_seconds = float(values.get("execution_seconds", run.elapsed_seconds))
        records = self.events.setdefault(run.id, [])
        for sequence, item in enumerate(values.get("execution_events", []), start=1):
            if sequence <= len(records):
                continue
            records.append(SimpleNamespace(sequence=sequence, **item))
        return run

    def list_events(self, tenant_id, investigation_id):
        run = self.get_for_tenant(tenant_id, investigation_id)
        return list(self.events.get(investigation_id, [])) if run else []

    def add_review(self, run, *, subject, approved, comment):
        key = (run.tenant_id, run.id)
        if key in self.reviews:
            return self.reviews[key], False
        record = SimpleNamespace(
            subject=subject, approved=approved, comment=comment,
            reviewed_at=datetime.now(timezone.utc),
        )
        self.reviews[key] = record
        return record, True

    def get_review(self, tenant_id, investigation_id):
        return self.reviews.get((tenant_id, investigation_id))
