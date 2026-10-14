from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from retrieval.context import build_context
from retrieval.contracts import RetrievalCandidate
from workflows.recruiter_brief import RecruiterBriefWorkflow


def evidence_context():
    candidate = RetrievalCandidate(
        document_id=uuid4(), document_version_id=uuid4(), chunk_id=uuid4(), chunk_key="section-0001-chunk-0001",
        section_id="section-0001", content="Built Python APIs with PostgreSQL.", source_start=0,
        source_end=33, page_number=1, source_key="candidate-1", media_type="application/pdf",
    )
    return build_context([candidate])


class FakePipeline:
    def __init__(self, context):
        self.context = context
        self.calls = 0

    def retrieve(self, authorization, tenant_id, query, **kwargs):
        self.calls += 1
        authorization.require_tenant(tenant_id)
        return self.context


class FakeProvider:
    def generate(self, prompt, schema):
        chunk_id = prompt.split('available_source_chunk_ids": ["')[1].split('"')[0]
        citation = {"source_chunk_id": chunk_id, "quote": "Built Python APIs"}
        return {
            "claims": [{"text": "The document describes Python API work.", "citations": [citation]}],
            "requirement_evidence": [{"requirement": "Python", "status": "supported", "explanation": "Directly stated.", "citations": [citation]}],
            "interview_questions": [{"question": "How were APIs designed?", "reason": "Verify API work.", "citations": [citation]}],
        }


class FakeReviewRepository:
    def __init__(self):
        self.records = {}
        self.calls = 0

    def upsert(self, **kwargs):
        self.calls += 1
        key = (kwargs["tenant_id"], kwargs["workflow_id"])
        if key in self.records:
            return self.records[key], False
        self.records[key] = kwargs
        return kwargs, True


def workflow(context=None, repository=None):
    return RecruiterBriefWorkflow(
        retrieval_pipeline=FakePipeline(context if context is not None else evidence_context()),
        provider=FakeProvider(), review_repository=repository or FakeReviewRepository(), checkpointer=InMemorySaver(),
    )


def initial_state(workflow_id="run-1", permitted=("tenant-a",)):
    return {
        "workflow_id": workflow_id, "tenant_id": "tenant-a", "subject": "reviewer-1",
        "permitted_tenant_ids": list(permitted),
        "job": {"description": "Python backend", "required_skills": ["Python"]}, "source_keys": [],
    }


def test_permission_denial_does_not_retrieve():
    graph = workflow()
    result = graph.graph.invoke(initial_state(permitted=()), {"configurable": {"thread_id": "denied"}})
    assert result["status"] == "failed_recoverable"
    assert graph.retrieval_pipeline.calls == 0


def test_missing_evidence_recovers_without_generation():
    graph = workflow(build_context([]))
    result = graph.graph.invoke(initial_state(), {"configurable": {"thread_id": "missing"}})
    assert result["status"] == "failed_recoverable"
    assert "No authorized supporting evidence" in result["error"]


def test_interrupt_resume_and_idempotent_persistence():
    repository = FakeReviewRepository()
    graph = workflow(repository=repository)
    config = {"configurable": {"thread_id": "review-thread"}}
    interrupted = graph.graph.invoke(initial_state(), config)
    assert "__interrupt__" in interrupted
    assert repository.calls == 0

    completed = graph.graph.invoke(Command(resume={"approved": True, "note": "Evidence checked"}), config)
    assert completed["status"] == "completed"
    assert repository.calls == 1
    assert repository.records[("tenant-a", "run-1")]["approved"] is True

    # A duplicate delivery on another graph thread cannot create a second reviewed result.
    duplicate_config = {"configurable": {"thread_id": "duplicate-thread"}}
    graph.graph.invoke(initial_state(), duplicate_config)
    graph.graph.invoke(Command(resume={"approved": True, "note": "repeat"}), duplicate_config)
    assert len(repository.records) == 1


def test_invalid_review_reinterrupts_instead_of_crashing_or_persisting():
    repository = FakeReviewRepository()
    graph = workflow(repository=repository)
    config = {"configurable": {"thread_id": "invalid-review"}}
    graph.graph.invoke(initial_state(), config)

    result = graph.graph.invoke(Command(resume={"approved": "yes"}), config)
    assert "__interrupt__" in result
    assert repository.calls == 0
    assert graph.graph.get_state(config).next == ("human_review",)

    completed = graph.graph.invoke(Command(resume={"approved": False, "note": "Weak evidence"}), config)
    assert completed["status"] == "completed"
    assert repository.records[("tenant-a", "run-1")]["approved"] is False


def test_review_for_another_tenant_is_rejected():
    repository = FakeReviewRepository()
    graph = workflow(repository=repository)
    config = {"configurable": {"thread_id": "tenant-mismatch"}}
    graph.graph.invoke(initial_state(), config)

    result = graph.graph.invoke(Command(resume={"approved": True, "tenant_id": "tenant-b"}), config)
    assert "__interrupt__" in result
    assert repository.calls == 0


def test_reviewer_subject_is_persisted_and_interrupt_does_not_offer_editing():
    repository = FakeReviewRepository()
    graph = workflow(repository=repository)
    config = {"configurable": {"thread_id": "reviewer"}}
    interrupted = graph.graph.invoke(initial_state(), config)
    instruction = interrupted["__interrupt__"][0].value["instruction"]
    assert "edit" not in instruction.casefold()
    assert "not a hiring decision" in instruction

    graph.graph.invoke(Command(resume={"approved": True, "reviewer": "reviewer-2"}), config)
    assert repository.records[("tenant-a", "run-1")]["subject"] == "reviewer-2"


def test_failures_store_fixed_error_codes_not_exception_text():
    class ExplodingPipeline:
        calls = 0

        def retrieve(self, *args, **kwargs):
            raise RuntimeError("SECRET resume text in exception")

    graph = workflow()
    graph.retrieval_pipeline = ExplodingPipeline()
    result = graph.graph.invoke(initial_state(), {"configurable": {"thread_id": "explode"}})
    assert result["status"] == "failed_recoverable"
    assert result["error_code"] == "retrieval_failed"
    assert "SECRET" not in result["error"]


def test_bad_citations_withhold_brief_from_checkpoint():
    class FabricatingProvider:
        def generate(self, prompt, schema):
            payload = FakeProvider().generate(prompt, schema)
            payload["claims"][0]["citations"][0]["quote"] = "Managed 40 engineers"
            return payload

    graph = workflow()
    graph.provider = FabricatingProvider()
    result = graph.graph.invoke(initial_state(), {"configurable": {"thread_id": "fabricated"}})
    assert result["error_code"] == "citation_failed"
    assert result["brief"] == {}


def test_reconcile_repersists_completed_review_idempotently():
    repository = FakeReviewRepository()
    graph = workflow(repository=repository)
    config = {"configurable": {"thread_id": "reconcile"}}
    graph.graph.invoke(initial_state(), config)
    graph.graph.invoke(Command(resume={"approved": True}), config)

    repository.records.clear()  # simulate a crash after checkpoint write, before review commit
    assert graph.reconcile(config) is True
    assert graph.reconcile(config) is False
    assert len(repository.records) == 1
