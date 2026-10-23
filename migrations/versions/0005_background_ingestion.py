"""Add PostgreSQL-backed ingestion jobs and embedding identity metadata."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0005_background_ingestion"
down_revision = "0004_investigations"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("document_versions", sa.Column(
        "embedding_model", sa.String(length=256), nullable=False,
        server_default="sentence-transformers/all-MiniLM-L6-v2",
    ))
    op.add_column("document_versions", sa.Column(
        "embedding_model_version", sa.String(length=64), nullable=False, server_default="legacy-untracked",
    ))
    op.add_column("document_versions", sa.Column(
        "embedding_updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()"),
    ))
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=256), nullable=False),
        sa.Column("job_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="queued"),
        sa.Column("source_key", sa.String(length=256), nullable=True),
        sa.Column("filename", sa.String(length=32), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.LargeBinary(), nullable=True),
        sa.Column("target_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("target_version_number", sa.Integer(), nullable=True),
        sa.Column("result_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("result_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("result_version_number", sa.Integer(), nullable=True),
        sa.Column("result_created", sa.Boolean(), nullable=True),
        sa.Column("embedding_model", sa.String(length=256), nullable=False),
        sa.Column("embedding_model_version", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("processing_latency_ms", sa.Float(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("status IN ('queued','processing','completed','failed')",
                           name="ck_ingestion_jobs_status"),
        sa.CheckConstraint("job_type IN ('ingest','reindex')", name="ck_ingestion_jobs_type"),
        sa.CheckConstraint("attempt_count >= 0 AND max_attempts BETWEEN 1 AND 10",
                           name="ck_ingestion_jobs_attempts"),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_ingestion_jobs_tenant_key"),
    )
    op.create_index("ix_ingestion_jobs_tenant_id", "ingestion_jobs", ["tenant_id"])
    op.create_index("ix_ingestion_jobs_claim", "ingestion_jobs", ["status", "available_at"])
    op.create_index("ix_ingestion_jobs_tenant_created", "ingestion_jobs", ["tenant_id", "created_at"])


def downgrade():
    op.drop_table("ingestion_jobs")
    op.drop_column("document_versions", "embedding_updated_at")
    op.drop_column("document_versions", "embedding_model_version")
    op.drop_column("document_versions", "embedding_model")
