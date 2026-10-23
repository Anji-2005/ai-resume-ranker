"""Recruitment document intelligence HTTP API.

Every candidate read, write, search, generation, or review requires a bearer
token whose server-side grant covers the requested ``X-Tenant-ID``. Error details
are fixed strings: exception text from parsers, validators, or providers is never
echoed because it can contain candidate content.
"""

import hashlib
from pathlib import PurePath
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.exc import IntegrityError

from api import presenters, schemas
from api.dependencies import (
    authorized,
    get_database_status,
    get_document_repository,
    get_generation_provider,
    get_generation_status,
    get_retrieval_pipeline,
    get_tenant_id,
    require_generation_enabled,
)
from api.policy import GENERATION_ENABLED, gemini_api_key, gemini_model_name
from api.workflows import router as workflow_router
from api.investigations import router as investigation_router
from api.ingestion import router as ingestion_router
from generation.citations import CitationVerificationError
from generation.service import (
    GenerationOutputError,
    GenerationProviderError,
    GroundedGenerationService,
    NoEvidenceError,
)
from repositories.documents import DocumentRepository, DocumentTooLargeError
from retrieval.authorization import (
    ROLE_EDITOR,
    ROLE_READER,
    AuthorizationContext,
    RetrievalAuthorizationError,
)
from retrieval.contracts import MetadataFilters
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from utils.matcher import MODEL_NAME
from utils.pdf_parser import MAX_FILE_SIZE, ResumeParseError


MAX_INGEST_CHUNKS = 500

app = FastAPI(title="Recruitment Document Intelligence API", version=schemas.API_CONTRACT_VERSION)
app.include_router(workflow_router)
app.include_router(investigation_router)
app.include_router(ingestion_router)

Reader = Annotated[AuthorizationContext, Depends(authorized(ROLE_READER))]
Editor = Annotated[AuthorizationContext, Depends(authorized(ROLE_EDITOR))]
Tenant = Annotated[str, Depends(get_tenant_id)]
Repository = Annotated[DocumentRepository, Depends(get_document_repository)]
Pipeline = Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)]


def _forbidden(exc: RetrievalAuthorizationError):
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@app.get("/v1/health", response_model=schemas.HealthResponse)
def health(database: Annotated[str, Depends(get_database_status)]):
    """Unauthenticated liveness and capability summary; contains no tenant data or secrets."""
    return schemas.HealthResponse(
        status="ok" if database == "ok" else "degraded", database=database,
        generation_provider_configured=bool(gemini_api_key()), embedding_model=MODEL_NAME,
        reranker_model=CrossEncoderReranker().model_name,
    )


@app.get("/v1/session", response_model=schemas.SessionResponse)
def session_info(
    authorization: Reader, tenant_id: Tenant,
    generation: Annotated[str, Depends(get_generation_status)],
):
    return schemas.SessionResponse(
        tenant_id=tenant_id, roles=sorted(authorization.roles), generation=generation,
        generation_model=gemini_model_name() if generation == GENERATION_ENABLED else None,
    )


@app.get("/v1/documents", response_model=schemas.DocumentListResponse)
def list_documents(
    authorization: Reader, tenant_id: Tenant, repository: Repository,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    rows, total = repository.list_documents(tenant_id, limit=limit, offset=offset)
    return schemas.DocumentListResponse(documents=[presenters.document_summary(row) for row in rows], total=total)


async def _read_bounded(file: UploadFile) -> bytes:
    data = await file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="The file exceeds the 10 MB size limit.")
    return data


def _opaque_filename(filename: str | None) -> str:
    """Persist only the extension the parser needs, never an identity-bearing filename."""
    return "document" + PurePath(filename or "").suffix.lower()


def _ingest(repository: DocumentRepository, tenant_id: str, source_key: str, filename: str | None, data: bytes):
    try:
        result = repository.ingest_bytes(
            tenant_id, source_key, _opaque_filename(filename), data, max_chunks=MAX_INGEST_CHUNKS,
        )
        repository.session.flush()
    except ResumeParseError as exc:
        # Parser messages are fixed strings that never include document content.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="A concurrent upload of this document is in progress; retry.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The document could not be processed.",
        ) from exc
    return schemas.IngestResponse(**result.__dict__, candidate_ref=schemas.candidate_ref(result.document_id))


@app.post("/v1/documents", response_model=schemas.IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    authorization: Editor, tenant_id: Tenant,
    file: Annotated[UploadFile, File(...)], repository: Repository,
    source_key: Annotated[str | None, Form(max_length=256)] = None,
):
    data = await _read_bounded(file)
    # Without a caller key, identical bytes map to one document: repeat uploads are idempotent.
    key = (source_key or "").strip() or "doc-" + hashlib.sha256(data).hexdigest()[:24]
    return _ingest(repository, tenant_id, key, file.filename, data)


def _search_filters(document_ids=(), source_keys=(), media_types=()):
    return MetadataFilters(
        document_ids=frozenset(document_ids), source_keys=frozenset(source_keys), media_types=frozenset(media_types),
    )


@app.get("/v1/documents/search", response_model=list[schemas.SearchResult])
def search_documents(
    authorization: Reader, tenant_id: Tenant,
    query: Annotated[str, Query(min_length=1, max_length=2_000)],
    pipeline: Pipeline,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    source_key: Annotated[list[str] | None, Query()] = None,
    media_type: Annotated[list[str] | None, Query()] = None,
    rerank: Annotated[bool, Query()] = False,
):
    """Compatibility route. Prefer ``POST /v1/documents/search`` so queries stay out of URLs."""
    try:
        results = pipeline.search(
            authorization, tenant_id, query, filters=_search_filters((), source_key or (), media_type or ()),
            limit=limit, use_reranker=rerank,
        )
    except RetrievalAuthorizationError as exc:
        raise _forbidden(exc) from exc
    return [presenters.search_result(item) for item in results]


@app.post("/v1/documents/search", response_model=list[schemas.SearchResult])
def search_documents_post(request: schemas.SearchRequest, authorization: Reader, tenant_id: Tenant, pipeline: Pipeline):
    try:
        results = pipeline.search(
            authorization, tenant_id, request.query, filters=_search_filters(request.document_ids),
            limit=request.limit, use_reranker=request.rerank,
        )
    except RetrievalAuthorizationError as exc:
        raise _forbidden(exc) from exc
    return [presenters.search_result(item) for item in results]


@app.get("/v1/documents/{document_id}", response_model=schemas.DocumentDetail)
def get_document(
    document_id: UUID, authorization: Reader, tenant_id: Tenant, repository: Repository,
    version: Annotated[int | None, Query(ge=1)] = None,
):
    document = repository.get_document(tenant_id, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    versions = repository.list_versions(tenant_id, document_id)
    current = next((row for row in versions if row.is_current), versions[0])
    selected = version or current.version_number
    if selected not in {row.version_number for row in versions}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document version not found")
    chunks = repository.get_version_chunks(tenant_id, document_id, selected)
    return schemas.DocumentDetail(
        document_id=document.id, candidate_ref=schemas.candidate_ref(document.id), created_at=document.created_at,
        versions=[presenters.version_info(row) for row in versions], selected_version_number=selected,
        chunks=[presenters.chunk_view(chunk) for chunk in chunks],
    )


@app.post(
    "/v1/documents/{document_id}/versions", response_model=schemas.IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document_version(
    document_id: UUID, authorization: Editor, tenant_id: Tenant,
    file: Annotated[UploadFile, File(...)], repository: Repository,
):
    document = repository.get_document(tenant_id, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    data = await _read_bounded(file)
    return _ingest(repository, tenant_id, document.source_key, file.filename, data)


@app.delete("/v1/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID, authorization: Editor, tenant_id: Tenant, repository: Repository):
    if not repository.delete_document(tenant_id, document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _generation_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, RetrievalAuthorizationError):
        return _forbidden(exc)
    if isinstance(exc, NoEvidenceError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No authorized evidence was retrieved for the selected documents.",
        )
    if isinstance(exc, CitationVerificationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Generated output failed citation verification and was withheld.",
        )
    if isinstance(exc, (GenerationProviderError, GenerationOutputError)):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The generation provider did not return a usable response.",
        )
    raise exc


GENERATION_ERRORS = (
    RetrievalAuthorizationError, NoEvidenceError, CitationVerificationError,
    GenerationProviderError, GenerationOutputError,
)


@app.post("/v1/documents/recruiter-brief", response_model=schemas.BriefResponse)
def create_recruiter_brief(
    request: schemas.BriefRequest, authorization: Reader, tenant_id: Tenant,
    generation: Annotated[str, Depends(get_generation_status)],
    pipeline: Pipeline, provider=Depends(get_generation_provider),
):
    """Unreviewed, citation-verified brief. The Streamlit UI uses the reviewed workflow instead."""
    require_generation_enabled(generation)
    try:
        result = GroundedGenerationService(provider, pipeline).grounded_brief(
            authorization, tenant_id, request.model_dump(include={"description", "required_skills", "preferred_skills"}),
            filters=_search_filters(request.document_ids, request.source_key),
        )
    except GENERATION_ERRORS as exc:
        raise _generation_errors(exc) from exc
    return schemas.BriefResponse(
        **result.output.model_dump(), sources=[presenters.supplied_source(item) for item in result.context.sources],
        generation=presenters.generation_metadata(result),
    )


@app.post("/v1/documents/qa", response_model=schemas.GroundedAnswerResponse)
def answer_document_question(
    request: schemas.QuestionRequest, authorization: Reader, tenant_id: Tenant,
    generation: Annotated[str, Depends(get_generation_status)],
    pipeline: Pipeline, provider=Depends(get_generation_provider),
):
    require_generation_enabled(generation)
    try:
        result = GroundedGenerationService(provider, pipeline).grounded_answer(
            authorization, tenant_id, request.question,
            filters=_search_filters(request.document_ids, request.source_key),
        )
    except GENERATION_ERRORS as exc:
        raise _generation_errors(exc) from exc
    return schemas.GroundedAnswerResponse(
        # All supplied sources are returned so reviewers can inspect cited and uncited evidence.
        **result.output.model_dump(), sources=[presenters.supplied_source(item) for item in result.context.sources],
        generation=presenters.generation_metadata(result) if result.context.sources else None,
    )
