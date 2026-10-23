"""Tenant-scoped asynchronous ingestion and reindex job API."""

import hashlib
import json
import os
from pathlib import PurePath
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.exc import IntegrityError

from api import schemas
from api.dependencies import (
    authorized,
    get_document_repository,
    get_ingestion_job_repository,
    get_tenant_id,
)
from repositories.documents import embedding_identity
from repositories.ingestion_jobs import JobIdempotencyConflict
from retrieval.authorization import ROLE_EDITOR, ROLE_READER, AuthorizationContext
from services.embeddings import SentenceTransformerEmbedder
from utils.pdf_parser import MAX_FILE_SIZE


router = APIRouter(tags=["ingestion"])
Reader = Annotated[AuthorizationContext, Depends(authorized(ROLE_READER))]
Editor = Annotated[AuthorizationContext, Depends(authorized(ROLE_EDITOR))]
Tenant = Annotated[str, Depends(get_tenant_id)]
JobStatusFilter = Literal["queued", "processing", "completed", "failed"]

ERROR_MESSAGES = {
    "extraction_failed": "The document could not be parsed safely.",
    "empty_document": "The document contains no extractable text.",
    "invalid_metadata": "The document metadata is invalid.",
    "chunk_limit": "The document exceeds the indexing chunk limit.",
    "embedding_invalid": "The embedding provider returned invalid vectors.",
    "document_not_found": "The requested document version is no longer available.",
    "processing_failed": "Processing failed transiently and exhausted its retry limit.",
}


def _max_attempts():
    try:
        value = int(os.getenv("INGESTION_MAX_ATTEMPTS", "3"))
    except ValueError:
        value = 3
    return max(1, min(value, 10))


async def _read_bounded(file: UploadFile):
    data = await file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="The file exceeds the 10 MB size limit.")
    return data


def _opaque_filename(filename):
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=422, detail="Only PDF and DOCX files are supported.")
    return "document" + suffix


def _request_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _view(job):
    document_id = job.result_document_id or job.target_document_id
    return schemas.IngestionJobView(
        job_id=job.id, job_type=job.job_type, status=job.status,
        attempt_count=job.attempt_count, max_attempts=job.max_attempts,
        embedding_model=job.embedding_model,
        embedding_model_version=job.embedding_model_version,
        candidate_ref=schemas.candidate_ref(document_id) if document_id else None,
        document_id=document_id, version_id=job.result_version_id,
        version_number=job.result_version_number or job.target_version_number,
        created=job.result_created, error_code=job.error_code,
        error=ERROR_MESSAGES.get(job.error_code) if job.status == "failed" else None,
        processing_latency_ms=job.processing_latency_ms,
        queued_at=job.queued_at, started_at=job.started_at, completed_at=job.completed_at,
        next_attempt_at=job.available_at if job.status == "queued" and job.attempt_count else None,
    )


def _enqueue(repository, values, response):
    try:
        job, created = repository.enqueue(**values)
    except JobIdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail="The idempotency key was used for a different request.") from exc
    except IntegrityError:
        repository.session.rollback()
        job = repository.get_by_key(values["tenant_id"], values["idempotency_key"])
        if job is None or job.request_hash != values["request_hash"]:
            raise HTTPException(status_code=409, detail="The ingestion job could not be created idempotently.") from None
        created = False
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return _view(job)


@router.post("/v1/ingestion-jobs", response_model=schemas.IngestionJobView,
             status_code=status.HTTP_202_ACCEPTED)
async def enqueue_document(
    response: Response, authorization: Editor, tenant_id: Tenant,
    file: Annotated[UploadFile, File(...)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128,
                                           pattern=r"^[A-Za-z0-9._:-]+$")],
    source_key: Annotated[str | None, Form(max_length=256)] = None,
    repository=Depends(get_ingestion_job_repository),
):
    data = await _read_bounded(file)
    filename = _opaque_filename(file.filename)
    content_hash = hashlib.sha256(data).hexdigest()
    key = (source_key or "").strip() or "doc-" + content_hash[:24]
    model, model_version = embedding_identity(SentenceTransformerEmbedder())
    request_hash = _request_hash({
        "type": "ingest", "source_key": key, "content_hash": content_hash,
        "filename": filename, "embedding_model": model, "embedding_model_version": model_version,
    })
    return _enqueue(repository, {
        "tenant_id": tenant_id, "idempotency_key": idempotency_key,
        "request_hash": request_hash, "created_by": authorization.subject,
        "job_type": "ingest", "source_key": key, "filename": filename,
        "content_hash": content_hash, "payload": data,
        "embedding_model": model, "embedding_model_version": model_version,
        "max_attempts": _max_attempts(),
    }, response)


@router.post("/v1/documents/{document_id}/ingestion-jobs", response_model=schemas.IngestionJobView,
             status_code=status.HTTP_202_ACCEPTED)
async def enqueue_document_version(
    document_id: UUID, response: Response, authorization: Editor, tenant_id: Tenant,
    file: Annotated[UploadFile, File(...)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128,
                                           pattern=r"^[A-Za-z0-9._:-]+$")],
    documents=Depends(get_document_repository), repository=Depends(get_ingestion_job_repository),
):
    document = documents.get_document(tenant_id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    data = await _read_bounded(file)
    filename = _opaque_filename(file.filename)
    content_hash = hashlib.sha256(data).hexdigest()
    model, model_version = embedding_identity(SentenceTransformerEmbedder())
    request_hash = _request_hash({
        "type": "ingest", "document_id": str(document_id), "content_hash": content_hash,
        "filename": filename, "embedding_model": model, "embedding_model_version": model_version,
    })
    return _enqueue(repository, {
        "tenant_id": tenant_id, "idempotency_key": idempotency_key,
        "request_hash": request_hash, "created_by": authorization.subject,
        "job_type": "ingest", "source_key": document.source_key, "filename": filename,
        "content_hash": content_hash, "payload": data,
        "target_document_id": document_id,
        "embedding_model": model, "embedding_model_version": model_version,
        "max_attempts": _max_attempts(),
    }, response)


@router.post("/v1/documents/{document_id}/reindex", response_model=schemas.IngestionJobView,
             status_code=status.HTTP_202_ACCEPTED)
def enqueue_reindex(
    document_id: UUID, request: schemas.ReindexRequest, response: Response,
    authorization: Editor, tenant_id: Tenant,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128,
                                           pattern=r"^[A-Za-z0-9._:-]+$")],
    documents=Depends(get_document_repository), repository=Depends(get_ingestion_job_repository),
):
    versions = documents.list_versions(tenant_id, document_id)
    if not versions:
        raise HTTPException(status_code=404, detail="document not found")
    version_number = request.version_number or next(item.version_number for item in versions if item.is_current)
    if version_number not in {item.version_number for item in versions}:
        raise HTTPException(status_code=404, detail="document version not found")
    model, model_version = embedding_identity(SentenceTransformerEmbedder())
    request_hash = _request_hash({
        "type": "reindex", "document_id": str(document_id), "version_number": version_number,
        "embedding_model": model, "embedding_model_version": model_version,
    })
    return _enqueue(repository, {
        "tenant_id": tenant_id, "idempotency_key": idempotency_key,
        "request_hash": request_hash, "created_by": authorization.subject,
        "job_type": "reindex", "target_document_id": document_id,
        "target_version_number": version_number,
        "embedding_model": model, "embedding_model_version": model_version,
        "max_attempts": _max_attempts(),
    }, response)


@router.get("/v1/ingestion-jobs", response_model=schemas.IngestionJobListResponse)
def list_jobs(
    authorization: Reader, tenant_id: Tenant,
    status_filter: Annotated[JobStatusFilter | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    repository=Depends(get_ingestion_job_repository),
):
    return schemas.IngestionJobListResponse(
        jobs=[_view(job) for job in repository.list_for_tenant(tenant_id, status=status_filter, limit=limit)]
    )


@router.get("/v1/ingestion-jobs/metrics", response_model=schemas.IngestionMetricsResponse)
def ingestion_metrics(
    authorization: Reader, tenant_id: Tenant, repository=Depends(get_ingestion_job_repository),
):
    return schemas.IngestionMetricsResponse(**repository.metrics(tenant_id))


@router.get("/v1/ingestion-jobs/{job_id}", response_model=schemas.IngestionJobView)
def get_job(
    job_id: UUID, authorization: Reader, tenant_id: Tenant,
    repository=Depends(get_ingestion_job_repository),
):
    job = repository.get_for_tenant(tenant_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ingestion job not found")
    return _view(job)
