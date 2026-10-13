"""One controlled LangGraph workflow for evidence-backed recruiter briefs.

The workflow has no autonomous hiring/rejection branch. Its only terminal output
is a human-reviewed brief or a safe failure/denial state.
"""

from typing import TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from generation.citations import verify_recruiter_brief
from generation.prompts import recruiter_brief_prompt
from generation.schemas import RecruiterBrief
from retrieval.authorization import AuthorizationContext, RetrievalAuthorizationError
from retrieval.contracts import MetadataFilters, RetrievalCandidate, RetrievalContext


MAX_GENERATION_ATTEMPTS = 2


class JobRequirements(BaseModel):
    description: str = Field(min_length=1, max_length=12_000)
    required_skills: list[str] = Field(default_factory=list, max_length=50)
    preferred_skills: list[str] = Field(default_factory=list, max_length=50)


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    tenant_id: str
    subject: str
    permitted_tenant_ids: list[str]
    job: dict
    source_keys: list[str]
    status: str
    evidence: list[dict]
    brief: dict
    review: dict
    generation_attempts: int
    error: str
    persisted: bool


class RecruiterBriefWorkflow:
    def __init__(self, *, retrieval_pipeline, provider, review_repository, checkpointer):
        self.retrieval_pipeline = retrieval_pipeline
        self.provider = provider
        self.review_repository = review_repository
        self.graph = self._build().compile(checkpointer=checkpointer)

    def _build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("authorize", self._authorize)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("validate_requirements", self._validate_requirements)
        graph.add_node("generate_brief", self._generate_brief)
        graph.add_node("verify_citations", self._verify_citations)
        graph.add_node("human_review", self._human_review)
        graph.add_node("persist_review", self._persist_review)
        graph.add_node("recover", self._recover)
        graph.add_edge(START, "authorize")
        graph.add_conditional_edges("authorize", self._after_authorize, {
            "retrieve_evidence": "retrieve_evidence", "recover": "recover",
        })
        graph.add_conditional_edges("retrieve_evidence", self._after_evidence, {
            "validate_requirements": "validate_requirements", "recover": "recover",
        })
        graph.add_conditional_edges("validate_requirements", self._after_requirements, {
            "generate_brief": "generate_brief", "recover": "recover",
        })
        graph.add_conditional_edges("generate_brief", self._after_generation, {
            "verify_citations": "verify_citations", "recover": "recover",
        })
        graph.add_conditional_edges("verify_citations", self._after_verification, {
            "human_review": "human_review", "recover": "recover",
        })
        graph.add_edge("human_review", "persist_review")
        graph.add_edge("persist_review", END)
        graph.add_edge("recover", END)
        return graph

    def _authorize(self, state: WorkflowState):
        try:
            AuthorizationContext(
                state["subject"], frozenset(state.get("permitted_tenant_ids", []))
            ).require_tenant(state["tenant_id"])
            return {"status": "authorized"}
        except (KeyError, RetrievalAuthorizationError) as exc:
            return {"status": "denied", "error": str(exc)}

    def _retrieve_evidence(self, state: WorkflowState):
        authorization = AuthorizationContext(
            state["subject"], frozenset(state["permitted_tenant_ids"])
        )
        job = JobRequirements.model_validate(state["job"])
        context = self.retrieval_pipeline.retrieve(
            authorization, state["tenant_id"], job.description,
            filters=MetadataFilters(source_keys=frozenset(state.get("source_keys", []))), limit=8,
        )
        if not context.candidates:
            return {"status": "missing_evidence", "error": "No authorized supporting evidence was retrieved."}
        return {"status": "evidence_retrieved", "evidence": [_candidate_to_dict(item) for item in context.candidates]}

    def _validate_requirements(self, state: WorkflowState):
        try:
            job = JobRequirements.model_validate(state["job"])
            return {"status": "requirements_validated", "job": job.model_dump()}
        except Exception as exc:
            return {"status": "invalid_requirements", "error": str(exc)}

    def _generate_brief(self, state: WorkflowState):
        context = _context_from_state(state["evidence"])
        attempts = int(state.get("generation_attempts", 0))
        last_error = None
        while attempts < MAX_GENERATION_ATTEMPTS:
            attempts += 1
            try:
                payload = self.provider.generate(recruiter_brief_prompt(state["job"], context), RecruiterBrief)
                brief = RecruiterBrief.model_validate(payload)
                return {"status": "brief_generated", "brief": brief.model_dump(), "generation_attempts": attempts}
            except Exception as exc:
                last_error = str(exc)
        return {"status": "generation_failed", "generation_attempts": attempts, "error": last_error or "generation failed"}

    def _verify_citations(self, state: WorkflowState):
        try:
            brief = RecruiterBrief.model_validate(state["brief"])
            verified = verify_recruiter_brief(brief, _context_from_state(state["evidence"]))
            return {"status": "brief_verified", "brief": verified.model_dump()}
        except Exception as exc:
            return {"status": "citation_failed", "error": str(exc)}

    def _human_review(self, state: WorkflowState):
        # No side effect may precede interrupt: this node restarts on resume.
        review = interrupt({
            "kind": "recruiter_brief_review", "workflow_id": state["workflow_id"],
            "brief": state["brief"],
            "instruction": "Review/edit the brief. This is not a hiring decision.",
        })
        if not isinstance(review, dict) or not isinstance(review.get("approved"), bool):
            return {"status": "review_invalid", "error": "review must contain boolean approved"}
        return {
            "status": "reviewed",
            "review": {"approved": review["approved"], "note": str(review.get("note", ""))[:2_000]},
        }

    def _persist_review(self, state: WorkflowState):
        review, created = self.review_repository.upsert(
            tenant_id=state["tenant_id"], workflow_id=state["workflow_id"], subject=state["subject"],
            brief=state["brief"], approved=state["review"]["approved"], note=state["review"]["note"],
        )
        return {"status": "completed", "persisted": True}

    @staticmethod
    def _recover(state: WorkflowState):
        return {"status": "failed_recoverable"}

    @staticmethod
    def _after_authorize(state):
        return "retrieve_evidence" if state.get("status") == "authorized" else "recover"

    @staticmethod
    def _after_evidence(state):
        return "validate_requirements" if state.get("status") == "evidence_retrieved" else "recover"

    @staticmethod
    def _after_requirements(state):
        return "generate_brief" if state.get("status") == "requirements_validated" else "recover"

    @staticmethod
    def _after_generation(state):
        return "verify_citations" if state.get("status") == "brief_generated" else "recover"

    @staticmethod
    def _after_verification(state):
        return "human_review" if state.get("status") == "brief_verified" else "recover"


def _candidate_to_dict(candidate: RetrievalCandidate):
    data = dict(candidate.__dict__)
    for key in ("document_id", "document_version_id", "chunk_id"):
        data[key] = str(data[key])
    return data


def _context_from_state(items):
    candidates = []
    for item in items:
        values = dict(item)
        for key in ("document_id", "document_version_id", "chunk_id"):
            values[key] = UUID(values[key])
        candidates.append(RetrievalCandidate(**values))
    return RetrievalContext(
        candidates=tuple(candidates), context_text="\n\n".join(
            f"[source_chunk_id={candidate.chunk_id}]\n{candidate.content}" for candidate in candidates
        ), source_chunk_ids=tuple(str(candidate.chunk_id) for candidate in candidates),
    )
