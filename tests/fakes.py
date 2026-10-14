"""Shared in-memory test doubles. Synthetic data only; no network, database, or LLM."""

from datetime import datetime, timezone
import hashlib
import json
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

from docx import Document

from repositories.documents import DocumentSummaryRow, DocumentTooLargeError, IngestionResult, VersionRow
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

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.0] * 384 for _ in texts]


class FakeSession:
    def flush(self):
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
        return [VersionRow(v.id, v.version_number, v.is_current, v.media_type, v.parser_name, v.parser_version,
                           v.content_hash, len(v.chunks), v.created_at) for v in reversed(document.versions)]

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
