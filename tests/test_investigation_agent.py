"""Offline tests for genuine model-directed investigation tool selection."""

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import pytest

from investigations.schemas import AgentLimits, ToolInvocation, ToolModelResponse
from investigations.tools import InvestigationTools, tool_definitions
from investigations.workflow import InvestigationAgent, initial_state
from retrieval.context import build_context
from retrieval.authorization import AuthorizationContext
from retrieval.authorization import RetrievalAuthorizationError
from retrieval.contracts import RetrievalCandidate


TENANT = "tenant-a"


@dataclass
class Version:
    version_id: UUID
    version_number: int
    is_current: bool
    media_type: str = "application/pdf"


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_key: str
    section_id: str
    content: str
    source_start: int
    source_end: int
    page_number: int | None = 1


class FakeDocumentRepository:
    def __init__(self):
        self.document_a, self.document_b = uuid4(), uuid4()
        self.version_a, self.version_b = uuid4(), uuid4()
        self.chunk_a, self.chunk_b = uuid4(), uuid4()
        self.documents = {
            self.document_a: Version(self.version_a, 1, True),
            self.document_b: Version(self.version_b, 1, True),
        }
        self.chunks = {
            self.document_a: [Chunk(
                self.chunk_a, self.document_a, self.version_a, "a-1", "experience",
                "Built Python APIs with PostgreSQL.", 0, 34,
            )],
            self.document_b: [Chunk(
                self.chunk_b, self.document_b, self.version_b, "b-1", "projects",
                "Maintained Kubernetes services on AWS.", 0, 36,
            )],
        }

    def existing_document_ids(self, tenant_id, document_ids):
        return set(document_ids) & set(self.documents) if tenant_id == TENANT else set()

    def list_versions(self, tenant_id, document_id):
        return [self.documents[document_id]] if tenant_id == TENANT and document_id in self.documents else []

    def get_version_chunks(self, tenant_id, document_id, version_number, limit=500):
        if tenant_id != TENANT or document_id not in self.documents:
            return []
        return self.chunks[document_id][:limit]


class FakePipeline:
    def __init__(self, repository):
        self.repository = repository
        self.queries = []

    def retrieve(self, authorization, tenant_id, query, *, filters, limit, max_characters, **_kwargs):
        authorization.require_tenant(tenant_id)
        self.queries.append(query)
        if "missing" in query.casefold():
            return build_context([])
        document_id = self.repository.document_b if "kubernetes" in query.casefold() else self.repository.document_a
        if document_id not in filters.document_ids:
            return build_context([])
        row = self.repository.chunks[document_id][0]
        candidate = RetrievalCandidate(
            document_id=row.document_id, document_version_id=row.document_version_id,
            chunk_id=row.id, chunk_key=row.chunk_key, section_id=row.section_id,
            content=row.content, source_start=row.source_start, source_end=row.source_end,
            page_number=row.page_number, source_key="synthetic", media_type="application/pdf",
        )
        return build_context([candidate], max_chunks=limit, max_characters=max_characters)


class ScriptedToolModel:
    model_name = "fake-tool-model-v1"

    def __init__(self, steps):
        self.steps = list(steps)
        self.prompts = []

    def invoke_tools(self, prompt, tools):
        self.prompts.append(prompt)
        step = self.steps.pop(0)
        name, arguments = step(prompt) if callable(step) else step
        return ToolModelResponse(
            calls=[ToolInvocation(id=f"action-{len(self.prompts)}", name=name, arguments=arguments)],
            input_tokens=100, output_tokens=50, cost_usd=0.001,
        )


class FailingToolModel:
    def invoke_tools(self, prompt, tools):
        raise RuntimeError("provider response containing confidential text")


def draft_for(repository, *, quote="Built Python APIs", source=None, requirements=()):
    source_id = str(source or repository.chunk_a)
    return ("draft_investigation", {"report": {
        "supported_findings": [{
            "text": "The document describes Python API work.",
            "citations": [{"source_chunk_id": source_id, "quote": quote}],
        }],
        "missing_evidence": [],
        "contradictory_evidence": [],
        "requirement_evidence": [{
            "requirement": requirement,
            "status": "supported",
            "explanation": "The requirement is stated in the cited evidence.",
            "citations": [{"source_chunk_id": source_id, "quote": quote}],
            "checked_action_ids": [],
        } for requirement in requirements],
        "verification_questions": [{
            "question": "What API design constraints applied?",
            "rationale": "A reviewer can verify implementation depth.",
            "citations": [{"source_chunk_id": source_id, "quote": quote}],
        }],
    }})


def make_agent(provider, *, limits=None):
    repository = FakeDocumentRepository()
    pipeline = FakePipeline(repository)
    tools = InvestigationTools(pipeline, repository)
    agent = InvestigationAgent(
        provider=provider(repository) if callable(provider) else provider,
        tools=tools,
        checkpointer=InMemorySaver(),
        limits=limits,
        generation_policy_allows=lambda _tenant_id: True,
    )
    return agent, repository, pipeline


def invoke(agent, repository, *, objective="Investigate technical evidence", document_ids=None,
           requirements=()):
    investigation_id = uuid4()
    config = {"configurable": {"thread_id": str(investigation_id)}}
    agent.graph.invoke(initial_state(
        investigation_id=investigation_id,
        tenant_id=TENANT,
        subject="token-sha256:test",
        permitted_tenant_ids=[TENANT],
        objective=objective,
        document_ids=document_ids or [repository.document_a, repository.document_b],
        requirements=requirements,
    ), config)
    return agent.graph.get_state(config)


def test_model_directs_different_first_tools_for_different_tasks():
    search_agent, search_repo, _ = make_agent(lambda repo: ScriptedToolModel([
        ("search_evidence", {"query": "Python API evidence", "limit": 3}),
        draft_for(repo),
    ]))
    inspect_agent, inspect_repo, _ = make_agent(lambda repo: ScriptedToolModel([
        ("inspect_document", {"document_id": str(repo.document_a), "limit": 1}),
        draft_for(repo),
    ]))

    search_state = invoke(search_agent, search_repo)
    inspect_state = invoke(inspect_agent, inspect_repo)

    assert search_state.next == ("human_review",)
    assert inspect_state.next == ("human_review",)
    assert search_state.values["execution_events"][0]["tool_name"] == "search_evidence"
    assert inspect_state.values["execution_events"][0]["tool_name"] == "inspect_document"


def test_model_can_finish_with_explicit_missing_evidence_after_empty_search():
    agent, repository, _ = make_agent(lambda _repo: ScriptedToolModel([
        ("search_evidence", {"query": "missing COBOL evidence"}),
        ("draft_investigation", {"report": {
            "supported_findings": [],
            "missing_evidence": [{
                "topic": "COBOL",
                "explanation": "No evidence was found in the authorized documents.",
                "checked_action_ids": ["action-1"],
            }],
            "contradictory_evidence": [],
            "verification_questions": [{
                "question": "Can the reviewer verify whether COBOL experience exists elsewhere?",
                "rationale": "The authorized evidence was insufficient.",
                "citations": [],
            }],
        }}),
    ]))

    snapshot = invoke(agent, repository)

    assert snapshot.next == ("human_review",)
    assert snapshot.values["report"]["supported_findings"] == []
    assert snapshot.values["report"]["missing_evidence"][0]["topic"] == "COBOL"


def test_agent_adapts_with_follow_up_search_when_first_search_is_empty():
    agent, repository, pipeline = make_agent(lambda repo: ScriptedToolModel([
        ("search_evidence", {"query": "missing Rust evidence", "limit": 3}),
        ("search_evidence", {"query": "Kubernetes platform evidence", "limit": 3}),
        ("draft_investigation", {"report": {
            "supported_findings": [{
                "text": "The document mentions Kubernetes services.",
                "citations": [{"source_chunk_id": str(repo.chunk_b), "quote": "Kubernetes services"}],
            }],
            "missing_evidence": [{
                "topic": "Rust",
                "explanation": "No evidence was found in the executed Rust search.",
                "checked_action_ids": ["action-1"],
            }],
            "contradictory_evidence": [],
            "verification_questions": [],
        }}),
    ]))

    snapshot = invoke(agent, repository)

    assert snapshot.next == ("human_review",)
    assert pipeline.queries == ["missing Rust evidence", "Kubernetes platform evidence"]
    assert [item["tool_name"] for item in snapshot.values["execution_events"]] == [
        "search_evidence", "search_evidence", "draft_investigation",
    ]
    assert snapshot.values["report"]["missing_evidence"][0]["topic"] == "Rust"


def test_requirement_check_is_restricted_to_explicit_initial_criteria():
    agent, repository, _ = make_agent(lambda repo: ScriptedToolModel([
        ("search_evidence", {"query": "Python evidence"}),
        ("check_requirements", {"requirements": ["Python"]}),
        draft_for(repo, requirements=["Python"]),
    ]))
    snapshot = invoke(agent, repository, requirements=["Python"])
    assert snapshot.next == ("human_review",)
    assert [item["tool_name"] for item in snapshot.values["execution_events"]] == [
        "search_evidence", "check_requirements", "draft_investigation",
    ]

    denied_agent, denied_repo, _ = make_agent(ScriptedToolModel([
        ("check_requirements", {"requirements": ["Go"]}),
    ]))
    denied = invoke(denied_agent, denied_repo, requirements=["Python"])
    assert denied.values["status"] == "failed"
    assert denied.values["error_code"] == "authorization_denied"


def test_document_outside_initial_or_tenant_scope_fails_before_model_call():
    model = ScriptedToolModel([("search_evidence", {"query": "Python"})])
    agent, repository, _ = make_agent(model)

    snapshot = invoke(agent, repository, document_ids=[uuid4()])

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "authorization_denied"
    assert snapshot.values["report"] == {}
    assert model.prompts == []


def test_generation_policy_denial_fails_before_model_or_document_access():
    model = ScriptedToolModel([("search_evidence", {"query": "Python"})])
    repository = FakeDocumentRepository()
    pipeline = FakePipeline(repository)
    agent = InvestigationAgent(
        provider=model,
        tools=InvestigationTools(pipeline, repository),
        checkpointer=InMemorySaver(),
        generation_policy_allows=lambda _tenant_id: False,
    )

    snapshot = invoke(agent, repository)

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "generation_policy_denied"
    assert model.prompts == []
    assert pipeline.queries == []


def test_model_cannot_inspect_a_document_outside_the_initial_scope():
    foreign_document = uuid4()
    agent, repository, _ = make_agent(ScriptedToolModel([
        ("inspect_document", {"document_id": str(foreign_document), "limit": 1}),
    ]))

    snapshot = invoke(agent, repository, document_ids=[repository.document_a])

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "authorization_denied"
    assert snapshot.values["execution_events"][0]["error_category"] == "authorization_denied"
    assert snapshot.values["report"] == {}


def test_invalid_citation_is_rejected_and_report_is_withheld():
    agent, repository, _ = make_agent(lambda repo: ScriptedToolModel([
        ("search_evidence", {"query": "Python API evidence"}),
        draft_for(repo, quote="Invented Java evidence"),
    ]))

    snapshot = invoke(agent, repository)

    assert snapshot.next == ()
    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "report_validation_failed"
    assert snapshot.values["report"] == {}


def test_agent_terminates_when_planning_limit_is_reached():
    limits = AgentLimits(max_planning_iterations=2, max_tool_calls=6)
    agent, repository, pipeline = make_agent(
        lambda _repo: ScriptedToolModel([
            ("search_evidence", {"query": "missing one"}),
            ("search_evidence", {"query": "missing two"}),
        ]),
        limits=limits,
    )

    snapshot = invoke(agent, repository)

    assert snapshot.next == ()
    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "planning_limit_exhausted"
    assert snapshot.values["tool_calls"] == 2
    assert pipeline.queries == ["missing one", "missing two"]
    assert snapshot.values["report"] == {}


def test_provider_failure_never_creates_a_report_or_leaks_exception_text():
    agent, repository, _ = make_agent(FailingToolModel())

    snapshot = invoke(agent, repository)

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "provider_failed"
    assert snapshot.values["report"] == {}
    assert "confidential" not in str(snapshot.values)


def test_malformed_provider_output_fails_without_a_report():
    class MalformedModel:
        @staticmethod
        def invoke_tools(prompt, tools):
            return {"calls": [], "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}

    agent, repository, pipeline = make_agent(MalformedModel())
    snapshot = invoke(agent, repository)

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "invalid_model_action"
    assert snapshot.values["report"] == {}
    assert pipeline.queries == []


def test_agent_terminates_at_tool_call_limit():
    agent, repository, pipeline = make_agent(
        ScriptedToolModel([("search_evidence", {"query": "missing evidence"})]),
        limits=AgentLimits(max_tool_calls=1, max_planning_iterations=3),
    )

    snapshot = invoke(agent, repository)

    assert snapshot.values["status"] == "failed"
    assert snapshot.values["error_code"] == "tool_limit_exhausted"
    assert snapshot.values["tool_calls"] == 1
    assert pipeline.queries == ["missing evidence"]
    assert snapshot.values["report"] == {}


def test_registry_exposes_only_the_five_approved_tools():
    assert {tool.name for tool in tool_definitions()} == {
        "search_evidence", "inspect_document", "check_requirements",
        "get_review_history", "draft_investigation",
    }


@pytest.mark.parametrize("tool_name", [
    "search_evidence", "inspect_document", "check_requirements",
    "get_review_history", "draft_investigation",
])
def test_every_tool_call_reauthorizes_the_pinned_tenant(tool_name):
    repository = FakeDocumentRepository()
    tools = InvestigationTools(FakePipeline(repository), repository)
    authorized = AuthorizationContext("token-sha256:test", frozenset({TENANT}))
    scope = tools.authorize_scope(authorized, TENANT, [repository.document_a])
    foreign_authorization = AuthorizationContext("token-sha256:other", frozenset({"tenant-b"}))

    with pytest.raises(RetrievalAuthorizationError):
        tools.execute(
            tool_name, {}, authorization=foreign_authorization, tenant_id=TENANT,
            scope_documents=[item.model_dump(mode="json") for item in scope],
            evidence=[], valid_action_ids=[], requirements_scope=[],
            remaining_characters=1_000, remaining_sources=5,
        )


def test_review_history_is_tenant_scoped_and_never_returns_note_text():
    repository = FakeDocumentRepository()
    expected_workflow_id = uuid4()

    class Runs:
        @staticmethod
        def list_for_tenant(tenant_id, limit):
            assert tenant_id == TENANT
            assert limit == 200
            return [SimpleNamespace(
                workflow_id=expected_workflow_id,
                document_ids=[str(repository.document_a)],
                status="reviewed",
                outcome="approved_notes",
            )]

    class Reviews:
        @staticmethod
        def get(*, tenant_id, workflow_id: UUID):
            assert tenant_id == TENANT
            assert workflow_id == expected_workflow_id
            return SimpleNamespace(
                reviewer_note="confidential reviewer note",
                reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    tools = InvestigationTools(FakePipeline(repository), repository, Runs(), Reviews())
    authorization = AuthorizationContext("token-sha256:test", frozenset({TENANT}))
    scope = tools.authorize_scope(authorization, TENANT, [repository.document_a])
    result = tools.execute(
        "get_review_history", {"document_id": str(repository.document_a)},
        authorization=authorization, tenant_id=TENANT,
        scope_documents=[item.model_dump(mode="json") for item in scope],
        evidence=[], valid_action_ids=[], requirements_scope=[],
        remaining_characters=1_000, remaining_sources=5,
    )

    assert result.status == "success"
    assert result.planner_payload["review_note_statuses"][0]["note_present"] is True
    assert "confidential reviewer note" not in str(result.planner_payload)


def test_cost_budget_is_enforced_before_any_tool_executes():
    class ExpensiveModel:
        def invoke_tools(self, prompt, tools):
            return ToolModelResponse(
                calls=[ToolInvocation(id="expensive", name="search_evidence", arguments={"query": "Python"})],
                input_tokens=100, output_tokens=50, cost_usd=0.5,
            )

    agent, repository, pipeline = make_agent(ExpensiveModel(), limits=AgentLimits(max_cost_usd=0.01))

    snapshot = invoke(agent, repository)

    assert snapshot.values["error_code"] == "cost_budget_exhausted"
    assert pipeline.queries == []
    assert snapshot.values["report"] == {}


def test_token_and_wall_clock_budgets_fail_closed():
    class TokenHeavyModel:
        def invoke_tools(self, prompt, tools):
            return ToolModelResponse(
                calls=[ToolInvocation(id="tokens", name="search_evidence", arguments={"query": "Python"})],
                input_tokens=20_000, output_tokens=1_000, cost_usd=0.001,
            )

    token_agent, token_repo, token_pipeline = make_agent(TokenHeavyModel())
    token_snapshot = invoke(token_agent, token_repo)
    assert token_snapshot.values["error_code"] == "token_budget_exhausted"
    assert token_pipeline.queries == []

    ticks = iter([0.0, 10.0])
    wall_repo = FakeDocumentRepository()
    wall_pipeline = FakePipeline(wall_repo)
    wall_agent = InvestigationAgent(
        provider=ScriptedToolModel([("search_evidence", {"query": "Python"})]),
        tools=InvestigationTools(wall_pipeline, wall_repo),
        checkpointer=InMemorySaver(),
        limits=AgentLimits(max_wall_clock_seconds=5),
        clock=lambda: next(ticks),
        generation_policy_allows=lambda _tenant_id: True,
    )
    wall_snapshot = invoke(wall_agent, wall_repo)
    assert wall_snapshot.values["error_code"] == "wall_clock_budget_exhausted"
    assert wall_pipeline.queries == []


def test_verified_report_pauses_and_can_resume_for_human_note_review():
    agent, repository, _ = make_agent(lambda repo: ScriptedToolModel([
        ("search_evidence", {"query": "Python API evidence"}),
        draft_for(repo),
    ]))
    investigation_id = uuid4()
    config = {"configurable": {"thread_id": str(investigation_id)}}
    agent.graph.invoke(initial_state(
        investigation_id=investigation_id,
        tenant_id=TENANT,
        subject="token-sha256:test",
        permitted_tenant_ids=[TENANT],
        objective="Investigate Python evidence",
        document_ids=[repository.document_a],
    ), config)

    pending = agent.graph.get_state(config)
    assert pending.next == ("human_review",)
    agent.graph.invoke(Command(resume={
        "approved": True,
        "note": "The citations match the supplied excerpts.",
        "reviewer": "token-sha256:reviewer",
        "tenant_id": TENANT,
    }), config)
    completed = agent.graph.get_state(config).values
    assert completed["status"] == "completed"
    assert completed["review"]["approved"] is True
    assert completed["review"]["reviewer"] == "token-sha256:reviewer"
    assert completed["report"]["supported_findings"]
