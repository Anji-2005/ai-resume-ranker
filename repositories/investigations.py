"""Tenant-scoped persistence for bounded AI investigation runs and safe traces."""

from sqlalchemy import func, select

from db.models import InvestigationEvent, InvestigationReview, InvestigationRun


class InvestigationRepository:
    def __init__(self, session):
        self.session = session

    def create(self, **values):
        run = InvestigationRun(**values)
        self.session.add(run)
        self.session.flush()
        return run

    def commit_started(self, run):
        """Make queued/running state durable before synchronous provider work starts."""
        self.session.commit()
        self.session.refresh(run)

    def get_for_tenant(self, tenant_id, investigation_id, *, for_update=False):
        statement = select(InvestigationRun).where(
            InvestigationRun.tenant_id == tenant_id,
            InvestigationRun.id == investigation_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def get_by_idempotency_key(self, tenant_id, key):
        return self.session.scalar(select(InvestigationRun).where(
            InvestigationRun.tenant_id == tenant_id,
            InvestigationRun.idempotency_key == key,
        ))

    def list_for_tenant(self, tenant_id, *, status=None, limit=100):
        statement = select(InvestigationRun).where(InvestigationRun.tenant_id == tenant_id)
        if status:
            statement = statement.where(InvestigationRun.status == status)
        return list(self.session.scalars(
            statement.order_by(InvestigationRun.updated_at.desc()).limit(max(1, min(limit, 200)))
        ).all())

    def update_status(self, run, *, status, error_code=None, outcome=None):
        run.status = status
        run.error_code = error_code
        run.outcome = outcome
        run.updated_at = func.now()
        self.session.flush()
        self.session.refresh(run)
        return run

    def sync_state(self, run, values, *, status, error_code=None, outcome=None):
        run.status = status
        run.error_code = error_code
        run.outcome = outcome
        run.document_scope = values.get("scope_documents", run.document_scope or [])
        run.report = values.get("report") or None
        run.sources = values.get("evidence", [])
        run.elapsed_seconds = float(values.get("execution_seconds", run.elapsed_seconds or 0.0))
        run.updated_at = func.now()
        self._sync_events(run, values.get("execution_events", []))
        self.session.flush()
        self.session.refresh(run)
        return run

    def _sync_events(self, run, events):
        known = set(self.session.scalars(select(InvestigationEvent.sequence).where(
            InvestigationEvent.investigation_id == run.id,
            InvestigationEvent.tenant_id == run.tenant_id,
        )).all())
        for sequence, item in enumerate(events, start=1):
            if sequence in known:
                continue
            self.session.add(InvestigationEvent(
                investigation_id=run.id,
                tenant_id=run.tenant_id,
                sequence=sequence,
                action_id=str(item.get("action_id", ""))[:128],
                tool_name=str(item.get("tool_name", ""))[:64],
                status=str(item.get("status", "error"))[:16],
                duration_ms=max(0.0, float(item.get("duration_ms", 0.0))),
                source_references=[str(value) for value in item.get("source_references", [])],
                error_category=(str(item["error_category"])[:64] if item.get("error_category") else None),
            ))

    def list_events(self, tenant_id, investigation_id):
        return list(self.session.scalars(select(InvestigationEvent).where(
            InvestigationEvent.tenant_id == tenant_id,
            InvestigationEvent.investigation_id == investigation_id,
        ).order_by(InvestigationEvent.sequence)).all())

    def add_review(self, run, *, subject, approved, comment):
        existing = self.get_review(run.tenant_id, run.id)
        if existing is not None:
            return existing, False
        review = InvestigationReview(
            investigation_id=run.id,
            tenant_id=run.tenant_id,
            subject=subject,
            approved=bool(approved),
            comment=comment or "",
        )
        self.session.add(review)
        self.session.flush()
        return review, True

    def get_review(self, tenant_id, investigation_id):
        return self.session.scalar(select(InvestigationReview).where(
            InvestigationReview.tenant_id == tenant_id,
            InvestigationReview.investigation_id == investigation_id,
        ))
