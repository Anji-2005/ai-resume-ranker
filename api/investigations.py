"""Synchronous HTTP lifecycle for the bounded V3 investigation agent.

Creation persists queued/running state before invoking the graph in the request
thread. It is deliberately not presented as background execution. PostgreSQL
checkpoints permit pending human reviews to survive process restarts.
"""

import hashlib
import json
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from langgraph.types import Command
from sqlalchemy.exc import IntegrityError

from api import presenters, schemas
from api.dependencies import (
    authorized,
    get_checkpointer,
    get_document_repository,
    get_generation_status,
    get_investigation_provider,
    get_investigation_repository,
    get_retrieval_pipeline,
    get_review_repository,
    get_run_repository,
    get_tenant_id,
    require_generation_enabled,
)
from investigations.schemas import (
    INVESTIGATION_PROMPT_VERSION,
    INVESTIGATION_SCHEMA_VERSION,
    INVESTIGATION_TOOLSET_VERSION,
)
from investigations.state import source_from_dict
from investigations.tools import InvestigationTools
from investigations.workflow import REVIEW_INSTRUCTION, InvestigationAgent, initial_state
from retrieval.authorization import ROLE_READER, ROLE_REVIEWER, AuthorizationContext


router = APIRouter(prefix="/v1/investigations", tags=["investigations"])
InvestigationStatusFilter = Literal["queued", "running", "awaiting_review", "completed", "failed"]
Reader = Annotated[AuthorizationContext, Depends(authorized(ROLE_READER))]
Reviewer = Annotated[AuthorizationContext, Depends(authorized(ROLE_REVIEWER))]
Tenant = Annotated[str, Depends(get_tenant_id)]

ERROR_MESSAGES = {
    "authorization_denied": "The investigation scope is not authorized for this workspace.",
    "invalid_scope": "The investigation scope is invalid.",
    "generation_policy_denied": "Grounded generation is disabled by workspace policy.",
    "invalid_model_action": "The model did not return a valid investigation action.",
    "provider_failed": "The generation provider did not return a usable action.",
    "token_budget_exhausted": "The investigation stopped at its token budget.",
    "cost_budget_exhausted": "The investigation stopped at its cost budget.",
    "cost_tracking_unavailable": "The investigation stopped because model cost tracking is not configured.",
    "wall_clock_budget_exhausted": "The investigation stopped at its time budget.",
    "tool_limit_exhausted": "The investigation stopped at its tool-call limit.",
    "planning_limit_exhausted": "The investigation stopped at its planning limit.",
    "tool_failed": "An authorized investigation tool failed safely.",
    "report_validation_failed": "The generated report failed evidence or citation validation and was withheld.",
    "checkpoint_missing": "The investigation checkpoint is unavailable.",
    "execution_failed": "The investigation failed safely before producing a report.",
}


def _config(investigation_id):
    return {"configurable": {"thread_id": str(investigation_id)}}


def _request_hash(request):
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _agent(*, provider, tools, checkpointer):
    return InvestigationAgent(
        provider=provider,
        tools=tools,
        checkpointer=checkpointer,
        # The FastAPI dependency already applied the repository's fail-closed
        # tenant generation policy before any model call.
        generation_policy_allows=lambda _tenant_id: True,
    )


def _state_status(snapshot):
    values = snapshot.values or {}
    if tuple(snapshot.next) == ("human_review",):
        return values, "awaiting_review", None, None
    if values.get("status") == "completed":
        approved = bool(values.get("review", {}).get("approved"))
        return values, "completed", None, "approved" if approved else "rejected"
    if values.get("status") == "failed":
        return values, "failed", values.get("error_code") or "execution_failed", None
    if not values:
        return values, "failed", "checkpoint_missing", None
    return values, "running", None, None


def _sync(run, repository, agent):
    try:
        snapshot = agent.graph.get_state(_config(run.id))
        values, run_status, error_code, outcome = _state_status(snapshot)
    except Exception:
        values, run_status, error_code, outcome = {}, "failed", "checkpoint_missing", None
    repository.sync_state(
        run, values, status=run_status, error_code=error_code, outcome=outcome,
    )
    review = values.get("review", {})
    if run_status == "completed" and review.get("reviewer"):
        # Idempotently repairs a crash after the durable checkpoint completed
        # but before the request transaction committed its review row.
        repository.add_review(
            run,
            subject=str(review["reviewer"])[:128],
            approved=bool(review.get("approved")),
            comment=str(review.get("note", ""))[:2_000],
        )
    return values


def _scope(run):
    return [schemas.InvestigationScopeDocument(
        document_id=UUID(item["document_id"]),
        candidate_ref=schemas.candidate_ref(UUID(item["document_id"])),
        version_id=UUID(item["version_id"]),
        version_number=item["version_number"],
        media_type=item["media_type"],
    ) for item in run.document_scope]


def _summary(run):
    document_ids = [UUID(value) for value in run.document_ids]
    return {
        "investigation_id": run.id,
        "status": run.status,
        "objective": run.objective,
        "candidate_refs": presenters.candidate_refs(run.document_ids),
        "document_ids": document_ids,
        "scope": _scope(run),
        "error_code": run.error_code,
        "outcome": run.outcome,
        "elapsed_seconds": run.elapsed_seconds,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _view(run, repository):
    view = schemas.InvestigationView(
        **_summary(run),
        job_description=run.job_description,
        requirements=run.requirements,
        events=[schemas.InvestigationEventView(
            sequence=item.sequence,
            tool_name=item.tool_name,
            status=item.status,
            duration_ms=item.duration_ms,
            source_references=item.source_references,
            error_category=item.error_category,
        ) for item in repository.list_events(run.tenant_id, run.id)],
        report=schemas.InvestigationReportView.model_validate(run.report) if run.report else None,
        sources=[presenters.supplied_source(source_from_dict(item)) for item in run.sources],
        instruction=REVIEW_INSTRUCTION if run.status == "awaiting_review" else None,
        error=ERROR_MESSAGES.get(run.error_code) if run.status == "failed" else None,
        model_name=run.model_name,
        prompt_version=run.prompt_version,
        toolset_version=run.toolset_version,
        schema_version=run.schema_version,
    )
    review = repository.get_review(run.tenant_id, run.id)
    if review is not None:
        view.review = schemas.InvestigationReviewRecord(
            decision="approved" if review.approved else "rejected",
            comment=review.comment,
            reviewed_at=review.reviewed_at,
        )
    return view


@router.post("", response_model=schemas.InvestigationView, status_code=status.HTTP_201_CREATED)
def create_investigation(
    request: schemas.InvestigationCreateRequest,
    response: Response,
    authorization: Reader,
    tenant_id: Tenant,
    idempotency_key: Annotated[str, Header(
        alias="Idempotency-Key", min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$",
    )],
    generation: Annotated[str, Depends(get_generation_status)],
    documents=Depends(get_document_repository),
    pipeline=Depends(get_retrieval_pipeline),
    workflow_runs=Depends(get_run_repository),
    workflow_reviews=Depends(get_review_repository),
    repository=Depends(get_investigation_repository),
    provider=Depends(get_investigation_provider),
    checkpointer=Depends(get_checkpointer),
):
    """Run synchronously until review or failure; retrying the same key is idempotent."""
    require_generation_enabled(generation)
    fingerprint = _request_hash(request)
    existing = repository.get_by_idempotency_key(tenant_id, idempotency_key)
    tools = InvestigationTools(pipeline, documents, workflow_runs, workflow_reviews)
    agent = _agent(provider=provider, tools=tools, checkpointer=checkpointer)
    if existing is not None:
        if existing.request_hash != fingerprint:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail="The idempotency key was already used for a different request.")
        response.status_code = status.HTTP_200_OK
        if existing.status in {"queued", "running", "awaiting_review"}:
            _sync(existing, repository, agent)
        return _view(existing, repository)

    document_ids = list(dict.fromkeys(request.document_ids))
    if len(document_ids) != len(request.document_ids):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail="Document selection must not contain duplicates.")
    if set(document_ids) - documents.existing_document_ids(tenant_id, document_ids):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    scope = tools.authorize_scope(authorization, tenant_id, document_ids)
    investigation_id = uuid4()
    try:
        run = repository.create(
            id=investigation_id,
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            request_hash=fingerprint,
            created_by=authorization.subject,
            objective=request.objective.strip(),
            job_description=request.job_description.strip(),
            requirements=[value.strip() for value in request.requirements],
            document_ids=[str(value) for value in document_ids],
            document_scope=[item.model_dump(mode="json") for item in scope],
            status="queued",
            error_code=None,
            outcome=None,
            report=None,
            sources=[],
            model_name=str(getattr(provider, "model_name", "unknown"))[:128],
            prompt_version=INVESTIGATION_PROMPT_VERSION,
            toolset_version=INVESTIGATION_TOOLSET_VERSION,
            schema_version=INVESTIGATION_SCHEMA_VERSION,
            elapsed_seconds=0.0,
        )
        repository.commit_started(run)
    except IntegrityError:
        repository.session.rollback()
        existing = repository.get_by_idempotency_key(tenant_id, idempotency_key)
        if existing is None or existing.request_hash != fingerprint:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail="The investigation could not be created idempotently.") from None
        response.status_code = status.HTTP_200_OK
        return _view(existing, repository)

    repository.update_status(run, status="running")
    repository.commit_started(run)
    try:
        agent.graph.invoke(initial_state(
            investigation_id=run.id,
            tenant_id=tenant_id,
            subject=authorization.subject,
            permitted_tenant_ids=[tenant_id],
            objective=run.objective,
            job_description=run.job_description,
            requirements=run.requirements,
            document_ids=document_ids,
            scope_documents=run.document_scope,
        ), _config(run.id))
        _sync(run, repository, agent)
    except Exception:
        repository.update_status(run, status="failed", error_code="execution_failed")
    return _view(run, repository)


@router.get("", response_model=schemas.InvestigationListResponse)
def list_investigations(
    authorization: Reader,
    tenant_id: Tenant,
    repository=Depends(get_investigation_repository),
    status_filter: Annotated[InvestigationStatusFilter | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
):
    rows = repository.list_for_tenant(tenant_id, status=status_filter, limit=limit)
    return schemas.InvestigationListResponse(investigations=[
        schemas.InvestigationSummary(**_summary(run)) for run in rows
    ])


@router.get("/{investigation_id}", response_model=schemas.InvestigationView)
def get_investigation(
    investigation_id: UUID,
    authorization: Reader,
    tenant_id: Tenant,
    repository=Depends(get_investigation_repository),
    checkpointer=Depends(get_checkpointer),
):
    run = repository.get_for_tenant(tenant_id, investigation_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="investigation not found")
    if run.status in {"queued", "running", "awaiting_review"}:
        empty_tools = InvestigationTools(None, None)
        _sync(run, repository, _agent(provider=None, tools=empty_tools, checkpointer=checkpointer))
    return _view(run, repository)


@router.post("/{investigation_id}/review", response_model=schemas.InvestigationView)
def review_investigation(
    investigation_id: UUID,
    submission: schemas.InvestigationReviewSubmission,
    authorization: Reviewer,
    tenant_id: Tenant,
    repository=Depends(get_investigation_repository),
    checkpointer=Depends(get_checkpointer),
):
    """Approve or reject generated notes. This is not an applicant decision."""
    run = repository.get_for_tenant(tenant_id, investigation_id, for_update=True)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="investigation not found")
    if repository.get_review(tenant_id, run.id) is not None or run.status == "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="These investigation notes have already been reviewed.")
    agent = _agent(provider=None, tools=InvestigationTools(None, None), checkpointer=checkpointer)
    _sync(run, repository, agent)
    if run.status != "awaiting_review":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="This investigation is not awaiting review.")
    agent.graph.invoke(Command(resume={
        "approved": submission.approved,
        "note": submission.comment,
        "tenant_id": tenant_id,
        # Server-supplied identity; the public request cannot choose it.
        "reviewer": authorization.subject,
    }), _config(run.id))
    _sync(run, repository, agent)
    return _view(run, repository)
