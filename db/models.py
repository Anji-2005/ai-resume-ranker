"""PostgreSQL persistence model for tenant-scoped document intelligence."""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Computed, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
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
    parsed_document: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
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
