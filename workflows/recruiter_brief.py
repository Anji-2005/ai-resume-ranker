"""One controlled LangGraph workflow for evidence-backed recruiter briefs.

The workflow has no autonomous hiring/rejection branch. Its only terminal output
is a human-reviewed brief or a safe failure/denial state. "Approved" and
"rejected" refer to the AI-generated notes, never to the applicant.
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
from retrieval.context import context_from_sources
from retrieval.contracts import ContextSource, MetadataFilters, RetrievalCandidate


MAX_GENERATION_ATTEMPTS = 2
MAX_REVIEW_NOTE_CHARS = 2_000
REVIEW_INSTRUCTION = "Approve or reject these AI-generated notes. This is not a hiring decision."

# Fixed, content-free error messages: workflow state is checkpointed and shown to
# reviewers, so exception text (which can echo model output) is never stored.
ERROR_MESSAGES = {
    "authorization_denied": "The workflow subject is not authorized for this tenant.",
    "no_evidence": "No authorized supporting evidence was retrieved.",
    "retrieval_failed": "Evidence retrieval failed.",
    "invalid_requirements": "The job requirements are invalid.",
    "generation_failed": "The generation provider did not return a valid brief.",
    "citation_failed": "The generated brief failed citation verification and was withheld.",
}


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
    document_ids: list[str]
    status: str
    evidence: list[dict]
    brief: dict
    review: dict
    generation_attempts: int
    error: str
    error_code: str
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
        graph.add_conditional_edges("human_review", self._after_review, {
            "persist_review": "persist_review", "human_review": "human_review",
        })
        graph.add_edge("persist_review", END)
        graph.add_edge("recover", END)
        return graph

    @staticmethod
    def _authorization(state: WorkflowState):
        return AuthorizationContext(state["subject"], frozenset(state.get("permitted_tenant_ids", [])))

    def _authorize(self, state: WorkflowState):
        try:
            self._authorization(state).require_tenant(state["tenant_id"])
            return {"status": "authorized"}
        except (KeyError, RetrievalAuthorizationError):
            return _failure("denied", "authorization_denied")

    def _retrieve_evidence(self, state: WorkflowState):
        try:
            job = JobRequirements.model_validate(state["job"])
            filters = MetadataFilters(
                source_keys=frozenset(state.get("source_keys", [])),
                document_ids=frozenset(UUID(value) for value in state.get("document_ids", [])),
            )
            context = self.retrieval_pipeline.retrieve(
                self._authorization(state), state["tenant_id"], job.description, filters=filters, limit=8,
            )
        except RetrievalAuthorizationError:
            return _failure("denied", "authorization_denied")
        except Exception:
            return _failure("retrieval_failed", "retrieval_failed")
        if not context.sources:
            return _failure("missing_evidence", "no_evidence")
        return {"status": "evidence_retrieved", "evidence": [source_to_dict(item) for item in context.sources]}

    def _validate_requirements(self, state: WorkflowState):
        try:
            job = JobRequirements.model_validate(state["job"])
            return {"status": "requirements_validated", "job": job.model_dump()}
        except Exception:
            return _failure("invalid_requirements", "invalid_requirements")

    def _generate_brief(self, state: WorkflowState):
        context = context_from_sources(source_from_dict(item) for item in state["evidence"])
        attempts = int(state.get("generation_attempts", 0))
        while attempts < MAX_GENERATION_ATTEMPTS:
            attempts += 1
            try:
                payload = self.provider.generate(recruiter_brief_prompt(state["job"], context), RecruiterBrief)
                brief = RecruiterBrief.model_validate(payload)
                return {"status": "brief_generated", "brief": brief.model_dump(), "generation_attempts": attempts}
            except Exception:
                continue
        return {**_failure("generation_failed", "generation_failed"), "generation_attempts": attempts}

    def _verify_citations(self, state: WorkflowState):
        try:
            brief = RecruiterBrief.model_validate(state["brief"])
            context = context_from_sources(source_from_dict(item) for item in state["evidence"])
            verified = verify_recruiter_brief(brief, context)
            return {"status": "brief_verified", "brief": verified.model_dump()}
        except Exception:
            return {**_failure("citation_failed", "citation_failed"), "brief": {}}

    def _human_review(self, state: WorkflowState):
        # No side effect may precede interrupt: this node restarts on resume.
        review = interrupt({
            "kind": "recruiter_brief_review", "workflow_id": state["workflow_id"],
            "brief": state["brief"], "instruction": REVIEW_INSTRUCTION,
        })
        if not isinstance(review, dict) or not isinstance(review.get("approved"), bool):
            return {"status": "review_invalid"}
        if review.get("tenant_id") is not None and review["tenant_id"] != state["tenant_id"]:
            return {"status": "review_invalid"}
        reviewer = str(review.get("reviewer") or state["subject"])
        return {
            "status": "reviewed",
            "review": {
                "approved": review["approved"],
                "note": str(review.get("note", ""))[:MAX_REVIEW_NOTE_CHARS],
                "reviewer": reviewer,
            },
        }

    def _persist_review(self, state: WorkflowState):
        self._upsert_review(state)
        return {"status": "completed", "persisted": True}

    def _upsert_review(self, state):
        return self.review_repository.upsert(
            tenant_id=state["tenant_id"], workflow_id=state["workflow_id"],
            subject=state["review"].get("reviewer", state["subject"]),
            brief=state["brief"], approved=state["review"]["approved"], note=state["review"]["note"],
        )

    def reconcile(self, config):
        """Re-persist a completed review idempotently (e.g. after a crash between checkpoint and commit)."""
        values = self.graph.get_state(config).values
        if values.get("status") != "completed" or not values.get("review"):
            return False
        _, created = self._upsert_review(values)
        return created

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

    @staticmethod
    def _after_review(state):
        return "persist_review" if state.get("status") == "reviewed" else "human_review"


def _failure(status, code):
    return {"status": status, "error_code": code, "error": ERROR_MESSAGES[code]}


def _candidate_to_dict(candidate: RetrievalCandidate):
    data = dict(candidate.__dict__)
    for key in ("document_id", "document_version_id", "chunk_id"):
        data[key] = str(data[key])
    return data


def source_to_dict(source: ContextSource) -> dict:
    return {
        "source_id": source.source_id, "candidate": _candidate_to_dict(source.candidate),
        "text": source.text, "source_start": source.source_start, "source_end": source.source_end,
        "truncated": source.truncated,
    }


def source_from_dict(item: dict) -> ContextSource:
    values = dict(item["candidate"])
    for key in ("document_id", "document_version_id", "chunk_id"):
        values[key] = UUID(values[key])
    return ContextSource(
        source_id=item["source_id"], candidate=RetrievalCandidate(**values), text=item["text"],
        source_start=item["source_start"], source_end=item["source_end"], truncated=item["truncated"],
    )
