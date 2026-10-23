"""Versioned HTTP contracts shared by the FastAPI service and the Streamlit client.

Pure Pydantic: no FastAPI, Streamlit, or database imports. No response model
carries a source filename, a source key, or a full document text; candidates are
identified only by opaque document IDs and ``candidate_ref`` labels.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


API_CONTRACT_VERSION = "v3.1"
MAX_SELECTED_DOCUMENTS = 50
MAX_WORKFLOW_DOCUMENTS = 5

GenerationStatus = Literal["enabled", "unconfigured", "policy_denied"]
WorkflowStatus = Literal["running", "awaiting_review", "completed", "failed"]
InvestigationStatus = Literal["queued", "running", "awaiting_review", "completed", "failed"]


def candidate_ref(document_id: UUID) -> str:
    """Stable opaque display label derived only from the random document UUID."""
    return "CAND-" + document_id.hex[:8].upper()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str = API_CONTRACT_VERSION
    database: Literal["ok", "unavailable"]
    generation_provider_configured: bool
    embedding_model: str
    reranker_model: str


class SessionResponse(BaseModel):
    tenant_id: str
    roles: list[str]
    generation: GenerationStatus
    generation_model: str | None = None


class IngestResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    version_number: int
    created: bool
    embedding_model: str
    embedding_model_version: str = "unknown"
    candidate_ref: str | None = None


class DocumentSummary(BaseModel):
    document_id: UUID
    candidate_ref: str
    current_version_number: int
    version_count: int
    media_type: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class DocumentVersionInfo(BaseModel):
    version_id: UUID
    version_number: int
    is_current: bool
    media_type: str
    parser_name: str
    parser_version: str
    embedding_model: str
    embedding_model_version: str
    content_sha256_prefix: str
    chunk_count: int
    created_at: datetime


class ChunkView(BaseModel):
    chunk_id: UUID
    chunk_key: str
    section_id: str
    page_number: int | None
    source_start: int
    source_end: int
    content: str = Field(description="Display text with common direct identifiers redacted")


class DocumentDetail(BaseModel):
    document_id: UUID
    candidate_ref: str
    created_at: datetime
    versions: list[DocumentVersionInfo]
    selected_version_number: int
    chunks: list[ChunkView]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_SELECTED_DOCUMENTS)
    rerank: bool = False


class SearchResult(BaseModel):
    document_id: UUID
    document_version_id: UUID
    candidate_ref: str
    chunk_id: UUID
    chunk_key: str
    section_id: str
    content: str
    source_start: int
    source_end: int
    page_number: int | None
    score: float = Field(description="Reranker score when reranked, otherwise the RRF fusion score")
    fusion_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    reranker_score: float | None = None
    source_chunk_id: str


class Citation(BaseModel):
    source_chunk_id: str
    quote: str


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class SuppliedSource(BaseModel):
    """Evidence exactly as supplied to the model; citations are verified only against this text."""

    source_chunk_id: str
    document_id: UUID
    candidate_ref: str
    section_id: str
    page_number: int | None
    source_start: int
    source_end: int
    truncated: bool
    excerpt: str


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    citations_verified: bool = True
    ai_generated: bool = True


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    source_key: list[str] = Field(default_factory=list, max_length=MAX_SELECTED_DOCUMENTS)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_SELECTED_DOCUMENTS)


class GroundedAnswerResponse(BaseModel):
    claims: list[Claim]
    uncertainty: str | None = None
    sources: list[SuppliedSource] = Field(default_factory=list)
    generation: GenerationMetadata | None = None


class RequirementEvidence(BaseModel):
    requirement: str
    status: Literal["supported", "unsupported", "unclear"]
    explanation: str
    citations: list[Citation] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    question: str
    reason: str
    citations: list[Citation]


class BriefContent(BaseModel):
    claims: list[Claim]
    requirement_evidence: list[RequirementEvidence]
    interview_questions: list[InterviewQuestion]


class BriefRequest(BaseModel):
    description: str = Field(min_length=1, max_length=12_000)
    required_skills: list[str] = Field(default_factory=list, max_length=50)
    preferred_skills: list[str] = Field(default_factory=list, max_length=50)
    source_key: list[str] = Field(default_factory=list, max_length=MAX_SELECTED_DOCUMENTS)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_SELECTED_DOCUMENTS)


class BriefResponse(BriefContent):
    sources: list[SuppliedSource] = Field(default_factory=list)
    generation: GenerationMetadata | None = None


class WorkflowStartRequest(BaseModel):
    description: str = Field(min_length=1, max_length=12_000)
    required_skills: list[str] = Field(default_factory=list, max_length=50)
    preferred_skills: list[str] = Field(default_factory=list, max_length=50)
    document_ids: list[UUID] = Field(min_length=1, max_length=MAX_WORKFLOW_DOCUMENTS)


class ReviewRecord(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str
    reviewed_at: datetime


class WorkflowSummary(BaseModel):
    workflow_id: UUID
    status: WorkflowStatus
    candidate_refs: list[str]
    document_ids: list[UUID]
    error_code: str | None = None
    outcome: Literal["approved", "rejected"] | None = None
    created_at: datetime
    updated_at: datetime


class WorkflowView(WorkflowSummary):
    error: str | None = None
    instruction: str | None = None
    brief: BriefContent | None = None
    sources: list[SuppliedSource] = Field(default_factory=list)
    review: ReviewRecord | None = None


class WorkflowStartResponse(BaseModel):
    workflows: list[WorkflowView]


class WorkflowListResponse(BaseModel):
    workflows: list[WorkflowSummary]


class ReviewSubmission(BaseModel):
    """Approval or rejection of AI-generated notes; never an employment decision."""

    approved: bool
    note: str = Field(default="", max_length=2_000)


class InvestigationCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=2_000)
    document_ids: list[UUID] = Field(min_length=1, max_length=10)
    job_description: str = Field(default="", max_length=12_000)
    requirements: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=50,
    )


class InvestigationScopeDocument(BaseModel):
    document_id: UUID
    candidate_ref: str
    version_id: UUID
    version_number: int
    media_type: str


class InvestigationEventView(BaseModel):
    sequence: int
    tool_name: str
    status: Literal["success", "empty", "error"]
    duration_ms: float
    source_references: list[str]
    error_category: str | None = None


class InvestigationFinding(BaseModel):
    text: str
    citations: list[Citation]


class InvestigationGap(BaseModel):
    topic: str
    explanation: str
    checked_action_ids: list[str]


class InvestigationContradiction(BaseModel):
    statement: str
    citations: list[Citation]


class InvestigationRequirementEvidence(BaseModel):
    requirement: str
    status: Literal["supported", "insufficient", "contradictory"]
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    checked_action_ids: list[str] = Field(default_factory=list)


class InvestigationQuestion(BaseModel):
    question: str
    rationale: str
    citations: list[Citation] = Field(default_factory=list)


class InvestigationReportView(BaseModel):
    supported_findings: list[InvestigationFinding] = Field(default_factory=list)
    missing_evidence: list[InvestigationGap] = Field(default_factory=list)
    contradictory_evidence: list[InvestigationContradiction] = Field(default_factory=list)
    requirement_evidence: list[InvestigationRequirementEvidence] = Field(default_factory=list)
    verification_questions: list[InvestigationQuestion] = Field(default_factory=list)


class InvestigationReviewRecord(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: str
    reviewed_at: datetime


class InvestigationSummary(BaseModel):
    investigation_id: UUID
    status: InvestigationStatus
    objective: str
    candidate_refs: list[str]
    document_ids: list[UUID]
    scope: list[InvestigationScopeDocument]
    error_code: str | None = None
    outcome: Literal["approved", "rejected"] | None = None
    elapsed_seconds: float = 0.0
    created_at: datetime
    updated_at: datetime


class InvestigationView(InvestigationSummary):
    job_description: str = ""
    requirements: list[str] = Field(default_factory=list)
    events: list[InvestigationEventView] = Field(default_factory=list)
    report: InvestigationReportView | None = None
    sources: list[SuppliedSource] = Field(default_factory=list)
    review: InvestigationReviewRecord | None = None
    instruction: str | None = None
    error: str | None = None
    execution_mode: Literal["synchronous"] = "synchronous"
    model_name: str
    prompt_version: str
    toolset_version: str
    schema_version: str


class InvestigationListResponse(BaseModel):
    investigations: list[InvestigationSummary]


class InvestigationReviewSubmission(BaseModel):
    """Approval or rejection of generated notes, never an applicant decision."""

    approved: bool
    comment: str = Field(default="", max_length=2_000)


IngestionJobStatus = Literal["queued", "processing", "completed", "failed"]
IngestionJobType = Literal["ingest", "reindex"]


class ReindexRequest(BaseModel):
    version_number: int | None = Field(default=None, ge=1)


class IngestionJobView(BaseModel):
    job_id: UUID
    job_type: IngestionJobType
    status: IngestionJobStatus
    attempt_count: int
    max_attempts: int
    embedding_model: str
    embedding_model_version: str
    candidate_ref: str | None = None
    document_id: UUID | None = None
    version_id: UUID | None = None
    version_number: int | None = None
    created: bool | None = None
    error_code: str | None = None
    error: str | None = None
    processing_latency_ms: float | None = None
    queued_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    next_attempt_at: datetime | None = None
    execution_mode: Literal["background_worker"] = "background_worker"


class IngestionJobListResponse(BaseModel):
    jobs: list[IngestionJobView]


class IngestionMetricsResponse(BaseModel):
    counts: dict[IngestionJobStatus, int]
    completed_last_hour: int
    failed_last_hour: int
    throughput_per_minute_last_hour: float
    mean_processing_latency_ms: float | None = None
    p95_processing_latency_ms: float | None = None
    oldest_queued_age_seconds: float | None = None
