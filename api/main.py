"""Focused HTTP API; authentication integration supplies the tenant header today."""

from typing import Annotated
from uuid import UUID
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.session import get_session_factory
from api.auth import get_retrieval_authorization
from repositories.documents import DocumentRepository, IngestionResult
from retrieval.authorization import AuthorizationContext, RetrievalAuthorizationError
from retrieval.contracts import MetadataFilters
from retrieval.pipeline import RetrievalPipeline
from generation.providers import LangChainGeminiProvider
from generation.service import GroundedGenerationService
from utils.pdf_parser import ResumeParseError


app = FastAPI(title="Recruitment Document Intelligence API", version="0.1.0")


class IngestResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    version_number: int
    created: bool
    embedding_model: str


class RetrievalResponse(BaseModel):
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    chunk_key: str
    section_id: str
    content: str
    source_start: int
    source_end: int
    page_number: int | None
    score: float
    source_chunk_id: str


class RecruiterBriefRequest(BaseModel):
    description: str = Field(min_length=1, max_length=12_000)
    required_skills: list[str] = Field(default_factory=list, max_length=50)
    preferred_skills: list[str] = Field(default_factory=list, max_length=50)
    source_key: list[str] = Field(default_factory=list, max_length=50)


class DocumentQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    source_key: list[str] = Field(default_factory=list, max_length=50)


def get_tenant_id(x_tenant_id: Annotated[str, Header()]) -> str:
    """Temporary tenant boundary until authentication supplies a trusted principal."""
    if not x_tenant_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-Tenant-ID is required")
    return x_tenant_id.strip()


def get_session():
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_authorized_session(
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
):
    """Authorize before creating a database session or touching a repository."""
    try:
        authorization.require_tenant(tenant_id)
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    yield from get_session()


@app.post("/v1/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: Annotated[UploadFile, File(...)],
    source_key: Annotated[str, Form(...)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    session: Annotated[Session, Depends(get_authorized_session)],
):
    try:
        authorization.require_tenant(tenant_id)
        data = await file.read()
        result = DocumentRepository(session).ingest_bytes(tenant_id, source_key, file.filename or "upload", data)
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except (ResumeParseError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _ingestion_response(result)


@app.get("/v1/documents/search", response_model=list[RetrievalResponse])
def search_documents(
    query: Annotated[str, Query(min_length=1, max_length=2_000)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    session: Annotated[Session, Depends(get_authorized_session)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    source_key: Annotated[list[str] | None, Query()] = None,
    media_type: Annotated[list[str] | None, Query()] = None,
    rerank: Annotated[bool, Query()] = False,
):
    filters = MetadataFilters(source_keys=frozenset(source_key or []), media_types=frozenset(media_type or []))
    try:
        context = RetrievalPipeline(DocumentRepository(session)).retrieve(
            authorization, tenant_id, query, filters=filters, limit=limit, use_reranker=rerank,
        )
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return [_retrieval_response(item) for item in context.candidates]


@app.delete("/v1/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    session: Annotated[Session, Depends(get_authorized_session)],
):
    try:
        authorization.require_tenant(tenant_id)
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    if not DocumentRepository(session).delete_document(tenant_id, document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")


@app.post("/v1/documents/recruiter-brief")
def create_recruiter_brief(
    request: RecruiterBriefRequest,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    session: Annotated[Session, Depends(get_authorized_session)],
):
    try:
        service = _grounded_service(session)
        result = service.recruiter_brief(
            authorization, tenant_id, request.model_dump(exclude={"source_key"}),
            filters=MetadataFilters(source_keys=frozenset(request.source_key)),
        )
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/v1/documents/qa")
def answer_document_question(
    request: DocumentQuestionRequest,
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    session: Annotated[Session, Depends(get_authorized_session)],
):
    try:
        result = _grounded_service(session).answer_question(
            authorization, tenant_id, request.question,
            filters=MetadataFilters(source_keys=frozenset(request.source_key)),
        )
    except RetrievalAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return result.model_dump()


def _ingestion_response(result: IngestionResult) -> IngestResponse:
    return IngestResponse(**result.__dict__)


def _retrieval_response(result) -> RetrievalResponse:
    return RetrievalResponse(
        document_id=result.document_id, document_version_id=result.document_version_id,
        chunk_id=result.chunk_id, chunk_key=result.chunk_key, source_chunk_id=str(result.chunk_id),
        section_id=result.section_id, content=result.content, source_start=result.source_start,
        source_end=result.source_end, page_number=result.page_number,
        score=result.reranker_score if result.reranker_score is not None else result.fusion_score,
    )


def _grounded_service(session):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for grounded generation")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"
    return GroundedGenerationService(
        LangChainGeminiProvider(api_key, model_name), RetrievalPipeline(DocumentRepository(session))
    )
