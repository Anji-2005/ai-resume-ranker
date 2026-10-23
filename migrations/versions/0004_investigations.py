"""Add durable V3 investigation runs, safe execution events, and note reviews."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004_investigations"
down_revision = "0003_workflow_runs"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "investigation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=256), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("job_description", sa.Text(), nullable=False, server_default=""),
        sa.Column("requirements", postgresql.JSONB(), nullable=False),
        sa.Column("document_ids", postgresql.JSONB(), nullable=False),
        sa.Column("document_scope", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column("report", postgresql.JSONB(), nullable=True),
        sa.Column("sources", postgresql.JSONB(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("toolset_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("elapsed_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_investigation_runs_tenant_key"),
    )
    op.create_index("ix_investigation_runs_tenant_id", "investigation_runs", ["tenant_id"])
    op.create_index("ix_investigation_runs_tenant_status", "investigation_runs", ["tenant_id", "status"])
    op.create_table(
        "investigation_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("investigation_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("action_id", sa.String(length=128), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("source_references", postgresql.JSONB(), nullable=False),
        sa.Column("error_category", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("investigation_id", "sequence", name="uq_investigation_events_sequence"),
    )
    op.create_index("ix_investigation_events_investigation_id", "investigation_events", ["investigation_id"])
    op.create_index("ix_investigation_events_tenant_id", "investigation_events", ["tenant_id"])
    op.create_table(
        "investigation_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("investigation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("investigation_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("tenant_id", "investigation_id", name="uq_investigation_reviews_tenant_run"),
    )
    op.create_index("ix_investigation_reviews_investigation_id", "investigation_reviews", ["investigation_id"])
    op.create_index("ix_investigation_reviews_tenant_id", "investigation_reviews", ["tenant_id"])


def downgrade():
    op.drop_table("investigation_reviews")
    op.drop_table("investigation_events")
    op.drop_table("investigation_runs")
