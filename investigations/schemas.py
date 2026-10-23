"""Versioned contracts for the bounded investigation agent and its tools."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from generation.schemas import ChunkCitation


INVESTIGATION_SCHEMA_VERSION = "investigation-report-v1"
INVESTIGATION_TOOLSET_VERSION = "investigation-tools-v1"
INVESTIGATION_PROMPT_VERSION = "investigation-planner-v1"


class AgentLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tool_calls: int = Field(default=6, ge=1, le=12)
    max_planning_iterations: int = Field(default=3, ge=1, le=6)
    max_total_tokens: int = Field(default=16_000, ge=1_000, le=100_000)
    max_cost_usd: float = Field(default=0.25, gt=0, le=10)
    max_wall_clock_seconds: float = Field(default=180, gt=1, le=900)
    max_evidence_characters: int = Field(default=16_000, ge=1_000, le=50_000)
    max_sources: int = Field(default=16, ge=1, le=50)
    max_output_tokens_per_plan: int = Field(default=1_500, ge=100, le=8_000)


class SearchEvidenceInput(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=5, ge=1, le=8)
    rerank: bool = False


class InspectDocumentInput(BaseModel):
    document_id: UUID
    source_chunk_id: UUID | None = None
    section_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=3, ge=1, le=5)


class CheckRequirementsInput(BaseModel):
    requirements: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _bounded_requirements(self):
        if any(not value.strip() or len(value) > 200 for value in self.requirements):
            raise ValueError("requirements must contain bounded non-empty strings")
        return self


class GetReviewHistoryInput(BaseModel):
    document_id: UUID | None = None
    limit: int = Field(default=10, ge=1, le=25)


class SupportedFinding(BaseModel):
    text: str = Field(min_length=1, max_length=700)
    citations: list[ChunkCitation] = Field(min_length=1, max_length=3)


class MissingEvidence(BaseModel):
    topic: str = Field(min_length=1, max_length=250)
    explanation: str = Field(min_length=1, max_length=500)
    checked_action_ids: list[str] = Field(min_length=1, max_length=6)


class ContradictoryEvidence(BaseModel):
    statement: str = Field(min_length=1, max_length=700)
    citations: list[ChunkCitation] = Field(min_length=2, max_length=4)


class VerificationQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=350)
    rationale: str = Field(min_length=1, max_length=350)
    citations: list[ChunkCitation] = Field(default_factory=list, max_length=3)


class RequirementFinding(BaseModel):
    requirement: str = Field(min_length=1, max_length=200)
    status: Literal["supported", "insufficient", "contradictory"]
    explanation: str = Field(min_length=1, max_length=500)
    citations: list[ChunkCitation] = Field(default_factory=list, max_length=4)
    checked_action_ids: list[str] = Field(default_factory=list, max_length=6)


class InvestigationReport(BaseModel):
    """Evidence notes only. There is deliberately no score or applicant decision."""

    model_config = ConfigDict(frozen=True)

    supported_findings: list[SupportedFinding] = Field(default_factory=list, max_length=10)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list, max_length=10)
    contradictory_evidence: list[ContradictoryEvidence] = Field(default_factory=list, max_length=5)
    requirement_evidence: list[RequirementFinding] = Field(default_factory=list, max_length=20)
    verification_questions: list[VerificationQuestion] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _require_a_result(self):
        if not (self.supported_findings or self.missing_evidence or self.contradictory_evidence):
            raise ValueError("an investigation report needs findings or an explicit evidence gap")
        return self


class DraftInvestigationInput(BaseModel):
    report: InvestigationReport


ToolName = Literal[
    "search_evidence", "inspect_document", "check_requirements",
    "get_review_history", "draft_investigation",
]


class ToolInvocation(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    name: ToolName
    arguments: dict


class ToolModelResponse(BaseModel):
    calls: list[ToolInvocation] = Field(min_length=1, max_length=6)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class ExecutionEvent(BaseModel):
    action_id: str
    tool_name: ToolName
    status: Literal["success", "empty", "error"]
    duration_ms: float = Field(ge=0)
    source_references: list[str] = Field(default_factory=list)
    error_category: str | None = None


class ScopeDocument(BaseModel):
    document_id: UUID
    version_id: UUID
    version_number: int
    media_type: str
