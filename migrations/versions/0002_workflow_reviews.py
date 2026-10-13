"""Add idempotent persisted human review records for LangGraph workflows."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_workflow_reviews"
down_revision = "0001_document_intelligence"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "workflow_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reviewed_brief", postgresql.JSONB(), nullable=False),
        sa.Column("reviewer_approved", sa.Boolean(), nullable=False),
        sa.Column("reviewer_note", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("tenant_id", "workflow_id", name="uq_workflow_reviews_tenant_workflow"),
    )
    op.create_index("ix_workflow_reviews_tenant_id", "workflow_reviews", ["tenant_id"])


def downgrade():
    op.drop_table("workflow_reviews")
