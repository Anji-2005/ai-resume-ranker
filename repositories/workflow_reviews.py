"""Idempotent persistence for human-reviewed workflow outputs."""

from sqlalchemy import select

from db.models import WorkflowReview


class WorkflowReviewRepository:
    def __init__(self, session):
        self.session = session

    def upsert(self, *, tenant_id, workflow_id, subject, brief, approved, note):
        existing = self.session.scalar(select(WorkflowReview).where(
            WorkflowReview.tenant_id == tenant_id, WorkflowReview.workflow_id == workflow_id
        ).with_for_update())
        if existing is not None:
            return existing, False
        review = WorkflowReview(
            tenant_id=tenant_id, workflow_id=workflow_id, subject=subject, status="reviewed",
            reviewed_brief=brief, reviewer_approved=bool(approved), reviewer_note=note or "",
        )
        self.session.add(review)
        self.session.flush()
        return review, True

    def get(self, *, tenant_id, workflow_id):
        return self.session.scalar(select(WorkflowReview).where(
            WorkflowReview.tenant_id == tenant_id, WorkflowReview.workflow_id == workflow_id
        ))
