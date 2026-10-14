"""Versioned HTTP contracts shared by the FastAPI service and the Streamlit client.

Pure Pydantic: no FastAPI, Streamlit, or database imports. No response model
carries a source filename, a source key, or a full document text; candidates are
identified only by opaque document IDs and ``candidate_ref`` labels.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


API_CONTRACT_VERSION = "v2.1"
MAX_SELECTED_DOCUMENTS = 50
MAX_WORKFLOW_DOCUMENTS = 5

GenerationStatus = Literal["enabled", "unconfigured", "policy_denied"]
WorkflowStatus = Literal["running", "awaiting_review", "completed", "failed"]


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
