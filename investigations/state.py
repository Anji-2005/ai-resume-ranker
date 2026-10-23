"""Explicit serializable LangGraph state for one investigation thread."""

from typing import TypedDict
from uuid import UUID

from retrieval.contracts import ContextSource, RetrievalCandidate


class InvestigationState(TypedDict, total=False):
    investigation_id: str
    tenant_id: str
    subject: str
    permitted_tenant_ids: list[str]
    objective: str
    job_description: str
    requirements: list[str]
    document_ids: list[str]
    scope_documents: list[dict]
    status: str
    planning_iterations: int
    tool_calls: int
    total_tokens: int
    cost_usd: float
    execution_seconds: float
    pending_calls: list[dict]
    evidence: list[dict]
    tool_results: list[dict]
    execution_events: list[dict]
    report: dict
    review: dict
    error_code: str


def candidate_to_dict(candidate: RetrievalCandidate) -> dict:
    values = dict(candidate.__dict__)
    for key in ("document_id", "document_version_id", "chunk_id"):
        values[key] = str(values[key])
    return values


def source_to_dict(source: ContextSource) -> dict:
    return {
        "source_id": source.source_id,
        "candidate": candidate_to_dict(source.candidate),
        "text": source.text,
        "source_start": source.source_start,
        "source_end": source.source_end,
        "truncated": source.truncated,
    }


def source_from_dict(item: dict) -> ContextSource:
    candidate = dict(item["candidate"])
    for key in ("document_id", "document_version_id", "chunk_id"):
        candidate[key] = UUID(candidate[key])
    return ContextSource(
        source_id=item["source_id"], candidate=RetrievalCandidate(**candidate),
        text=item["text"], source_start=item["source_start"],
        source_end=item["source_end"], truncated=item.get("truncated", False),
    )
