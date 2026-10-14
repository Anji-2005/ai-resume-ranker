"""HTTP lifecycle for the single LangGraph recruiter-brief review workflow.

The tenant-filtered ``workflow_runs`` row is always loaded before a checkpoint is
read, so a workflow ID from another tenant is indistinguishable from a missing one.
Responses never include checkpoint internals such as subjects or permitted tenants.
"""

from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.types import Command

from api import presenters, schemas
from api.dependencies import (
    authorized,
    get_checkpointer,
    get_document_repository,
    get_generation_provider,
    get_generation_status,
    get_retrieval_pipeline,
    get_review_repository,
    get_run_repository,
    get_tenant_id,
    require_generation_enabled,
)
from retrieval.authorization import ROLE_READER, ROLE_REVIEWER, AuthorizationContext
from workflows.recruiter_brief import ERROR_MESSAGES, REVIEW_INSTRUCTION, RecruiterBriefWorkflow, source_from_dict


router = APIRouter(prefix="/v1/workflows", tags=["workflows"])

Reader = Annotated[AuthorizationContext, Depends(authorized(ROLE_READER))]
Reviewer = Annotated[AuthorizationContext, Depends(authorized(ROLE_REVIEWER))]
Tenant = Annotated[str, Depends(get_tenant_id)]


def _config(workflow_id: str):
    return {"configurable": {"thread_id": workflow_id}}


def _workflow(review_repository, checkpointer, retrieval_pipeline=None, provider=None):
    # Read and resume paths never retrieve or generate, so they need no pipeline or provider.
    return RecruiterBriefWorkflow(
        retrieval_pipeline=retrieval_pipeline, provider=provider,
        review_repository=review_repository, checkpointer=checkpointer,
    )


def _sync(run, workflow, runs) -> dict:
    """Project checkpoint state onto the run row; returns the checkpoint values."""
    snapshot = workflow.graph.get_state(_config(run.workflow_id))
    values = snapshot.values or {}
    error_code, outcome = None, None
    if tuple(snapshot.next) == ("human_review",):
        state = "awaiting_review"
    elif values.get("status") == "completed":
        state = "completed"
        outcome = "approved" if values["review"]["approved"] else "rejected"
        # Idempotent; restores the review row if a crash separated checkpoint and commit.
        workflow.reconcile(_config(run.workflow_id))
    elif values.get("status") == "failed_recoverable":
        state, error_code = "failed", values.get("error_code") or "workflow_failed"
    elif not values:
        state, error_code = "failed", "checkpoint_missing"
    else:
        state = "running"
    if (run.status, run.error_code, run.outcome) != (state, error_code, outcome):
        runs.update_status(run, status=state, error_code=error_code, outcome=outcome)
    return values


def _summary(run) -> dict:
    return {
        "workflow_id": UUID(run.workflow_id), "status": run.status,
        "candidate_refs": presenters.candidate_refs(run.document_ids),
        "document_ids": [UUID(value) for value in run.document_ids],
        "error_code": run.error_code, "outcome": run.outcome,
        "created_at": run.created_at, "updated_at": run.updated_at,
    }


def _view(run, values, reviews) -> schemas.WorkflowView:
    view = schemas.WorkflowView(**_summary(run))
    sources = [presenters.supplied_source(source_from_dict(item)) for item in values.get("evidence", [])]
    if run.status == "awaiting_review" and values.get("brief"):
        view.brief = schemas.BriefContent.model_validate(values["brief"])
        view.sources = sources
        view.instruction = REVIEW_INSTRUCTION
    elif run.status == "completed":
        record = reviews.get(tenant_id=run.tenant_id, workflow_id=run.workflow_id)
        if record is not None:
            view.brief = schemas.BriefContent.model_validate(record.reviewed_brief)
            view.sources = sources
            view.review = schemas.ReviewRecord(
                decision="approved" if record.reviewer_approved else "rejected",
                note=record.reviewer_note, reviewed_at=record.reviewed_at,
            )
    elif run.status == "failed":
        view.error = ERROR_MESSAGES.get(run.error_code or "", "The workflow could not be completed.")
    return view


@router.post("/recruiter-brief", response_model=schemas.WorkflowStartResponse, status_code=status.HTTP_201_CREATED)
def start_recruiter_brief_workflows(
    request: schemas.WorkflowStartRequest, authorization: Reader, tenant_id: Tenant,
    generation: Annotated[str, Depends(get_generation_status)],
    repository=Depends(get_document_repository), pipeline=Depends(get_retrieval_pipeline),
    runs=Depends(get_run_repository), reviews=Depends(get_review_repository),
    provider=Depends(get_generation_provider), checkpointer=Depends(get_checkpointer),
):
    """Start one bounded workflow per selected document; each stops at the human-review interrupt."""
    require_generation_enabled(generation)
    document_ids = list(dict.fromkeys(request.document_ids))
    if set(document_ids) - repository.existing_document_ids(tenant_id, document_ids):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    workflow = _workflow(reviews, checkpointer, pipeline, provider)
    job = request.model_dump(include={"description", "required_skills", "preferred_skills"})
    views = []
    for document_id in document_ids:
        workflow_id = str(uuid4())
        run = runs.create(
            tenant_id=tenant_id, workflow_id=workflow_id, created_by=authorization.subject,
            document_ids=[document_id],
        )
        workflow.graph.invoke({
            "workflow_id": workflow_id, "tenant_id": tenant_id, "subject": authorization.subject,
            # Only the requested tenant is checkpointed; the full grant is never stored.
            "permitted_tenant_ids": [tenant_id], "job": job, "source_keys": [],
            "document_ids": [str(document_id)],
        }, _config(workflow_id))
        views.append(_view(run, _sync(run, workflow, runs), reviews))
    return schemas.WorkflowStartResponse(workflows=views)


@router.get("", response_model=schemas.WorkflowListResponse)
def list_workflows(
    authorization: Reader, tenant_id: Tenant, runs=Depends(get_run_repository),
    status_filter: Annotated[
        Literal["running", "awaiting_review", "completed", "failed"] | None, Query(alias="status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
):
    rows = runs.list_for_tenant(tenant_id, status=status_filter, limit=limit)
    return schemas.WorkflowListResponse(workflows=[schemas.WorkflowSummary(**_summary(run)) for run in rows])


@router.get("/{workflow_id}", response_model=schemas.WorkflowView)
def get_workflow(
    workflow_id: UUID, authorization: Reader, tenant_id: Tenant,
    runs=Depends(get_run_repository), reviews=Depends(get_review_repository),
    checkpointer=Depends(get_checkpointer),
):
    run = runs.get_for_tenant(tenant_id, str(workflow_id))
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workflow not found")
    values = _sync(run, _workflow(reviews, checkpointer), runs)
    return _view(run, values, reviews)


@router.post("/{workflow_id}/review", response_model=schemas.WorkflowView)
def submit_review(
    workflow_id: UUID, submission: schemas.ReviewSubmission, authorization: Reviewer, tenant_id: Tenant,
    runs=Depends(get_run_repository), reviews=Depends(get_review_repository),
    checkpointer=Depends(get_checkpointer),
):
    """Approve or reject AI-generated notes. This is never an applicant decision."""
    # Row lock serializes concurrent submissions for the same workflow.
    run = runs.get_for_tenant(tenant_id, str(workflow_id), for_update=True)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workflow not found")
    workflow = _workflow(reviews, checkpointer)
    _sync(run, workflow, runs)
    if run.status == "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This workflow has already been reviewed.")
    if run.status != "awaiting_review":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This workflow is not awaiting review.")
    workflow.graph.invoke(Command(resume={
        "approved": submission.approved, "note": submission.note,
        # Server-supplied identity and tenant; the client cannot choose them.
        "reviewer": authorization.subject, "tenant_id": tenant_id,
    }), _config(run.workflow_id))
    values = _sync(run, workflow, runs)
    return _view(run, values, reviews)
