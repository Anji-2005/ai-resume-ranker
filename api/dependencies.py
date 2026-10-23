"""Injectable FastAPI dependencies.

Authorization dependencies are declared before any session/repository dependency
in each route so a denied request never opens a database session.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from api.auth import get_retrieval_authorization
from api.policy import (
    GENERATION_POLICY_DENIED,
    GENERATION_UNCONFIGURED,
    gemini_api_key,
    gemini_model_name,
    generation_status,
)
from db.session import get_session_factory
from repositories.documents import DocumentRepository
from repositories.investigations import InvestigationRepository
from repositories.ingestion_jobs import IngestionJobRepository
from repositories.workflow_reviews import WorkflowReviewRepository
from repositories.workflow_runs import WorkflowRunRepository
from retrieval.authorization import AuthorizationContext, RetrievalAuthorizationError
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from services.embeddings import SentenceTransformerEmbedder


def get_tenant_id(x_tenant_id: Annotated[str, Header()]) -> str:
    """Requested tenant. It is never authority by itself; see ``authorized``."""
    if not x_tenant_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-Tenant-ID is required")
    return x_tenant_id.strip()


def authorized(role: str | None = None):
    """Build a dependency that enforces tenant authorization and, optionally, a role."""

    def dependency(
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        authorization: Annotated[AuthorizationContext, Depends(get_retrieval_authorization)],
    ) -> AuthorizationContext:
        try:
            authorization.require_tenant(tenant_id)
            if role:
                authorization.require_role(role)
        except RetrievalAuthorizationError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return authorization

    return dependency


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


def get_database_status() -> str:
    """Probe database reachability without exposing connection details."""
    from sqlalchemy import text

    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "unavailable"


@lru_cache(maxsize=1)
def get_embedder():
    """One embedder per process; the underlying model is loaded lazily and cached."""
    return SentenceTransformerEmbedder()


@lru_cache(maxsize=1)
def get_reranker():
    """One cross-encoder per process; the model loads on first reranked request only."""
    return CrossEncoderReranker()


def get_document_repository(session=Depends(get_session)) -> DocumentRepository:
    return DocumentRepository(session, embedder=get_embedder())


def get_retrieval_pipeline(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> RetrievalPipeline:
    return RetrievalPipeline(repository, embedder=get_embedder(), reranker=get_reranker())


def get_review_repository(session=Depends(get_session)) -> WorkflowReviewRepository:
    return WorkflowReviewRepository(session)


def get_run_repository(session=Depends(get_session)) -> WorkflowRunRepository:
    return WorkflowRunRepository(session)


def get_investigation_repository(session=Depends(get_session)) -> InvestigationRepository:
    return InvestigationRepository(session)


def get_ingestion_job_repository(session=Depends(get_session)) -> IngestionJobRepository:
    return IngestionJobRepository(session)


def get_generation_status(tenant_id: Annotated[str, Depends(get_tenant_id)]) -> str:
    return generation_status(tenant_id)


def require_generation_enabled(state: str) -> None:
    """Fail closed before any candidate text could reach a provider."""
    if state == GENERATION_UNCONFIGURED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounded generation is not configured on the server.",
        )
    if state == GENERATION_POLICY_DENIED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Grounded generation is not enabled by policy for this tenant.",
        )


def get_generation_provider():
    from generation.providers import LangChainGeminiProvider

    return LangChainGeminiProvider(gemini_api_key(), gemini_model_name())


def get_investigation_provider():
    from investigations.provider import LangChainGeminiToolProvider
    from investigations.schemas import AgentLimits

    limits = AgentLimits()
    return LangChainGeminiToolProvider(
        gemini_api_key(), gemini_model_name(), max_output_tokens=limits.max_output_tokens_per_plan,
    )


def get_checkpointer():
    from workflows.checkpointing import pooled_postgres_checkpointer

    return pooled_postgres_checkpointer()
