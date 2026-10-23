"""Tenant-scoped durable PostgreSQL queue for document ingestion and reindexing."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, or_, select

from db.models import IngestionJob


MAX_RETRY_BACKOFF_SECONDS = 3_600


class JobIdempotencyConflict(ValueError):
    pass


class IngestionJobRepository:
    def __init__(self, session):
        self.session = session

    def enqueue(self, *, tenant_id, idempotency_key, request_hash, created_by, job_type,
                embedding_model, embedding_model_version, source_key=None, filename=None,
                content_hash=None, payload=None, target_document_id=None,
                target_version_number=None, max_attempts=3):
        existing = self.get_by_key(tenant_id, idempotency_key)
        if existing is not None:
            if existing.request_hash != request_hash:
                raise JobIdempotencyConflict("idempotency key already used for a different ingestion request")
            return existing, False
        job = IngestionJob(
            tenant_id=tenant_id, idempotency_key=idempotency_key,
            request_hash=request_hash, created_by=created_by, job_type=job_type,
            source_key=source_key, filename=filename, content_hash=content_hash,
            payload=payload, target_document_id=target_document_id,
            target_version_number=target_version_number,
            embedding_model=embedding_model, embedding_model_version=embedding_model_version,
            max_attempts=max_attempts,
        )
        self.session.add(job)
        self.session.flush()
        return job, True

    def get_by_key(self, tenant_id: str, key: str):
        return self.session.scalar(select(IngestionJob).where(
            IngestionJob.tenant_id == tenant_id,
            IngestionJob.idempotency_key == key,
        ))

    def get_for_tenant(self, tenant_id: str, job_id: UUID):
        return self.session.scalar(select(IngestionJob).where(
            IngestionJob.tenant_id == tenant_id,
            IngestionJob.id == job_id,
        ))

    def list_for_tenant(self, tenant_id: str, *, status=None, limit=100):
        statement = select(IngestionJob).where(IngestionJob.tenant_id == tenant_id)
        if status:
            statement = statement.where(IngestionJob.status == status)
        return list(self.session.scalars(
            statement.order_by(IngestionJob.created_at.desc()).limit(max(1, min(limit, 200)))
        ).all())

    def claim_next(self, worker_id: str, *, lease_seconds: int, now=None):
        now = now or datetime.now(UTC)
        job = self.session.scalar(
            select(IngestionJob).where(
                IngestionJob.attempt_count < IngestionJob.max_attempts,
                or_(
                    (IngestionJob.status == "queued") & (IngestionJob.available_at <= now),
                    (IngestionJob.status == "processing") & (IngestionJob.lease_expires_at < now),
                ),
            ).order_by(IngestionJob.available_at, IngestionJob.created_at)
            .with_for_update(skip_locked=True).limit(1)
        )
        if job is None:
            return None
        job.status = "processing"
        job.attempt_count += 1
        job.started_at = job.started_at or now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.worker_id = worker_id
        job.error_code = None
        job.updated_at = now
        self.session.flush()
        return job

    def lock_claim(self, job_id: UUID, worker_id: str):
        return self.session.scalar(select(IngestionJob).where(
            IngestionJob.id == job_id,
            IngestionJob.status == "processing",
            IngestionJob.worker_id == worker_id,
        ).with_for_update())

    @staticmethod
    def complete(job, result, *, now=None, latency_ms=0.0):
        now = now or datetime.now(UTC)
        job.status = "completed"
        job.result_document_id = result.document_id
        job.result_version_id = result.version_id
        job.result_version_number = result.version_number
        job.result_created = result.created
        job.embedding_model = result.embedding_model
        job.embedding_model_version = result.embedding_model_version
        job.payload = None
        job.lease_expires_at = None
        job.worker_id = None
        job.error_code = None
        job.processing_latency_ms = max(0.0, float(latency_ms))
        job.completed_at = now
        job.updated_at = now

    @staticmethod
    def record_failure(job, *, error_code: str, retryable: bool, base_backoff_seconds: int,
                       now=None, latency_ms=0.0):
        now = now or datetime.now(UTC)
        exhausted = job.attempt_count >= job.max_attempts
        if retryable and not exhausted:
            delay = min(
                MAX_RETRY_BACKOFF_SECONDS,
                base_backoff_seconds * (2 ** max(0, job.attempt_count - 1)),
            )
            job.status = "queued"
            job.available_at = now + timedelta(seconds=delay)
        else:
            job.status = "failed"
            job.completed_at = now
            job.payload = None
            job.processing_latency_ms = max(0.0, float(latency_ms))
        job.error_code = error_code
        job.lease_expires_at = None
        job.worker_id = None
        job.updated_at = now

    def metrics(self, tenant_id: str, *, now=None):
        now = now or datetime.now(UTC)
        counts = dict(self.session.execute(
            select(IngestionJob.status, func.count(IngestionJob.id))
            .where(IngestionJob.tenant_id == tenant_id)
            .group_by(IngestionJob.status)
        ).all())
        recent = list(self.session.scalars(select(IngestionJob).where(
            IngestionJob.tenant_id == tenant_id,
            IngestionJob.completed_at >= now - timedelta(hours=1),
        )).all())
        completed = [job for job in recent if job.status == "completed"]
        failed = [job for job in recent if job.status == "failed"]
        latencies = sorted(
            job.processing_latency_ms for job in completed if job.processing_latency_ms is not None
        )
        p95_index = max(0, int(len(latencies) * 0.95 + 0.999999) - 1) if latencies else 0
        oldest = self.session.scalar(select(func.min(IngestionJob.queued_at)).where(
            IngestionJob.tenant_id == tenant_id, IngestionJob.status == "queued",
        ))
        return {
            "counts": {state: int(counts.get(state, 0)) for state in (
                "queued", "processing", "completed", "failed",
            )},
            "completed_last_hour": len(completed),
            "failed_last_hour": len(failed),
            "throughput_per_minute_last_hour": len(completed) / 60.0,
            "mean_processing_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "p95_processing_latency_ms": latencies[p95_index] if latencies else None,
            "oldest_queued_age_seconds": max(0.0, (now - oldest).total_seconds()) if oldest else None,
        }
