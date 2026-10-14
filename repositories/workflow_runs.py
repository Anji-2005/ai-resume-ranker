"""Tenant-scoped persistence of workflow run status (not checkpoint contents)."""

from sqlalchemy import func, select

from db.models import WorkflowRun


class WorkflowRunRepository:
    def __init__(self, session):
        self.session = session

    def create(self, *, tenant_id, workflow_id, created_by, document_ids):
        run = WorkflowRun(
            tenant_id=tenant_id, workflow_id=workflow_id, created_by=created_by,
            document_ids=[str(value) for value in document_ids], status="running",
        )
        self.session.add(run)
        self.session.flush()
        return run

    def get_for_tenant(self, tenant_id, workflow_id, *, for_update=False):
        statement = select(WorkflowRun).where(
            WorkflowRun.tenant_id == tenant_id, WorkflowRun.workflow_id == str(workflow_id)
        )
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def list_for_tenant(self, tenant_id, *, status=None, limit=100):
        statement = select(WorkflowRun).where(WorkflowRun.tenant_id == tenant_id)
        if status:
            statement = statement.where(WorkflowRun.status == status)
        return list(self.session.scalars(
            statement.order_by(WorkflowRun.updated_at.desc()).limit(max(1, min(limit, 200)))
        ).all())

    def update_status(self, run, *, status, error_code=None, outcome=None):
        run.status = status
        run.error_code = error_code
        run.outcome = outcome
        run.updated_at = func.now()
        self.session.flush()
        self.session.refresh(run)
        return run
