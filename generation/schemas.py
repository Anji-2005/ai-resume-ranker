"""Strict LLM output schemas. Scores, ranks, and hiring decisions are absent by design."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=500)


class GroundedClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=700)
    citations: list[ChunkCitation] = Field(min_length=1, max_length=3)


class RequirementAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirement: str = Field(min_length=1)
    status: Literal["supported", "unsupported", "unclear"]
    explanation: str = Field(min_length=1, max_length=350)
    citations: list[ChunkCitation] = Field(default_factory=list, max_length=3)


class GroundedInterviewQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, max_length=350)
    reason: str = Field(min_length=1, max_length=350)
    citations: list[ChunkCitation] = Field(min_length=1, max_length=3)


class RecruiterBrief(BaseModel):
    """Decision-support material; it deliberately cannot encode a hiring outcome."""

    model_config = ConfigDict(frozen=True)

    claims: list[GroundedClaim] = Field(min_length=1, max_length=5)
    requirement_evidence: list[RequirementAnalysis]
    interview_questions: list[GroundedInterviewQuestion] = Field(min_length=1, max_length=3)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: list[GroundedClaim] = Field(default_factory=list, max_length=5)
    uncertainty: str | None = Field(default=None, max_length=350)

    @model_validator(mode="after")
    def _require_claim_or_uncertainty(self):
        if not self.claims and not self.uncertainty:
            raise ValueError("an answer needs a cited claim or explicit uncertainty")
        return self
