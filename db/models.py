"""PostgreSQL persistence model for tenant-scoped document intelligence."""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2


class Base(DeclarativeBase):
    pass


class StoredDocument(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "source_key", name="uq_documents_tenant_source"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    source_key: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    versions: Mapped[list["StoredDocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class StoredDocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_number"),
        UniqueConstraint("document_id", "content_hash", name="uq_document_versions_hash"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    filename: Mapped[str] = mapped_column(String(512))
    media_type: Mapped[str] = mapped_column(String(128))
    parser_name: Mapped[str] = mapped_column(String(128))
    parser_version: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(256))
    embedding_model_version: Mapped[str] = mapped_column(String(64))
    parsed_document: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    embedding_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    document: Mapped[StoredDocument] = relationship(back_populates="versions")
    chunks: Mapped[list["StoredDocumentChunk"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan", passive_deletes=True
    )


class StoredDocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_key", name="uq_document_chunks_key"),
        Index("ix_document_chunks_search", "search_vector", postgresql_using="gin"),
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    chunk_key: Mapped[str] = mapped_column(String(128))
    section_id: Mapped[str] = mapped_column(String(128))
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_start: Mapped[int] = mapped_column(Integer)
    source_end: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    search_vector: Mapped[object] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document_version: Mapped[StoredDocumentVersion] = relationship(back_populates="chunks")


class WorkflowReview(Base):
    """Idempotent persisted human review of a generated recruiter brief."""

    __tablename__ = "workflow_reviews"
    __table_args__ = (UniqueConstraint("tenant_id", "workflow_id", name="uq_workflow_reviews_tenant_workflow"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    workflow_id: Mapped[str] = mapped_column(String(128))
    subject: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32))
    reviewed_brief: Mapped[dict] = mapped_column(JSONB)
    reviewer_approved: Mapped[bool] = mapped_column(Boolean)
    reviewer_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WorkflowRun(Base):
    """Tenant-scoped index of recruiter-brief workflow threads.

    The LangGraph checkpoint thread_id equals ``workflow_id``; checkpoints are read
    only after this tenant-filtered row is found, so other tenants cannot reach them.
    """

    __tablename__ = "workflow_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "workflow_id", name="uq_workflow_runs_tenant_workflow"),
        UniqueConstraint("workflow_id", name="uq_workflow_runs_workflow_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    workflow_id: Mapped[str] = mapped_column(String(128))
    created_by: Mapped[str] = mapped_column(String(256))
    document_ids: Mapped[list] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InvestigationRun(Base):
    """Durable tenant-scoped index and safe presentation state for one investigation."""

    __tablename__ = "investigation_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_investigation_runs_tenant_key"),
        Index("ix_investigation_runs_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(256))
    objective: Mapped[str] = mapped_column(Text)
    job_description: Mapped[str] = mapped_column(Text, default="")
    requirements: Mapped[list] = mapped_column(JSONB)
    document_ids: Mapped[list] = mapped_column(JSONB)
    document_scope: Mapped[list] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sources: Mapped[list] = mapped_column(JSONB)
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    toolset_version: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(64))
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InvestigationEvent(Base):
    """Sanitized execution event; never stores prompts, excerpts, or reasoning."""

    __tablename__ = "investigation_events"
    __table_args__ = (
        UniqueConstraint("investigation_id", "sequence", name="uq_investigation_events_sequence"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("investigation_runs.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    action_id: Mapped[str] = mapped_column(String(128))
    tool_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    duration_ms: Mapped[float] = mapped_column(Float)
    source_references: Mapped[list] = mapped_column(JSONB)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InvestigationReview(Base):
    """One reviewer decision on generated notes, not an applicant decision."""

    __tablename__ = "investigation_reviews"
    __table_args__ = (
        UniqueConstraint("tenant_id", "investigation_id", name="uq_investigation_reviews_tenant_run"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    investigation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("investigation_runs.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    subject: Mapped[str] = mapped_column(String(256))
    approved: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str] = mapped_column(Text, default="")
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionJob(Base):
    """Durable PostgreSQL work item; upload bytes are purged at terminal state."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_ingestion_jobs_tenant_key"),
        CheckConstraint("status IN ('queued','processing','completed','failed')",
                        name="ck_ingestion_jobs_status"),
        CheckConstraint("job_type IN ('ingest','reindex')", name="ck_ingestion_jobs_type"),
        CheckConstraint("attempt_count >= 0 AND max_attempts BETWEEN 1 AND 10",
                        name="ck_ingestion_jobs_attempts"),
        Index("ix_ingestion_jobs_claim", "status", "available_at"),
        Index("ix_ingestion_jobs_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(256))
    job_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")
    source_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    filename: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    target_document_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    target_version_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_document_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    result_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    result_version_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_created: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(256))
    embedding_model_version: Mapped[str] = mapped_column(String(64))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, server_default="3")
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
