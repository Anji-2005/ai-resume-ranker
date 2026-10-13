"""Create tenant-scoped document, version, and pgvector chunk storage."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision = "0001_document_intelligence"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("source_key", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("tenant_id", "source_key", name="uq_documents_tenant_source"),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=False),
        sa.Column("parser_name", sa.String(length=128), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("parsed_document", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("document_id", "version_number", name="uq_document_versions_number"),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_document_versions_hash"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_key", sa.String(length=128), nullable=False),
        sa.Column("section_id", sa.String(length=128), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("source_start", sa.Integer(), nullable=False),
        sa.Column("source_end", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), sa.Computed("to_tsvector('english', coalesce(content, ''))", persisted=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("document_version_id", "chunk_key", name="uq_document_chunks_key"),
    )
    op.create_index("ix_document_chunks_tenant_id", "document_chunks", ["tenant_id"])
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_document_version_id", "document_chunks", ["document_version_id"])
    op.execute("CREATE INDEX ix_document_chunks_search ON document_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)")


def downgrade():
    op.drop_table("document_chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
