"""Streamlit UI tests: navigation, Standalone and Connected modes, citations, errors, blind review.

Connected-mode tests drive the real pages through the real ``ApiClient`` wired to
the FastAPI app (``TestClient``) with in-memory fakes, so UI → client → API
contracts are exercised end to end without a database or LLM.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

import frontend.context as frontend_context
from api import dependencies
from api.main import app as api_app
from frontend.api_client import ApiClient, ApiConfig, ApiError
from retrieval.pipeline import RetrievalPipeline

from fakes import (
    CitingProvider,
    FakeEmbedder,
    InMemoryDocumentRepository,
    InMemoryInvestigationRepository,
    InMemoryIngestionJobRepository,
    InMemoryReviewRepository,
    InMemoryRunRepository,
    InvestigationToolProvider,
    synthetic_docx,
)


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Overview", "Resume Library", "Candidate Ranking", "AI Candidate Search", "Ask AI",
         "Recruiter Briefs", "AI Investigations", "Review History"]
CONFIG = ApiConfig("http://api.test", "tenant-a", "ui-token")


def run_page(page, timeout=30):
    at = AppTest.from_file(APP_PATH, default_timeout=timeout).run()
    at.sidebar.radio(key="nav").set_value(page).run()
    return at


def texts(at):
    values = []
    for kind in ("markdown", "info", "warning", "error", "success", "caption", "header", "subheader"):
        values.extend(str(element.value) for element in getattr(at, kind))
    values.extend(str(expander.label) for expander in at.expander)
    return "\n".join(values)


def click(at, label):
    next(button for button in at.button if button.label == label).click().run()
    return at


@pytest.fixture
def standalone(monkeypatch):
    monkeypatch.setattr(frontend_context, "_cached_client", lambda: None)


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_AUTH_TOKENS_JSON", '{"ui-token":["tenant-a"]}')
    state = {"generation": "enabled", "provider": CitingProvider()}
    repository, runs, reviews, saver = (
        InMemoryDocumentRepository(), InMemoryRunRepository(), InMemoryReviewRepository(), InMemorySaver()
    )
    investigations = InMemoryInvestigationRepository()
    ingestion_jobs = InMemoryIngestionJobRepository()
    api_app.dependency_overrides.update({
        dependencies.get_document_repository: lambda: repository,
        dependencies.get_retrieval_pipeline: lambda: RetrievalPipeline(repository, embedder=FakeEmbedder()),
        dependencies.get_run_repository: lambda: runs,
        dependencies.get_review_repository: lambda: reviews,
        dependencies.get_generation_provider: lambda: state["provider"],
        dependencies.get_investigation_provider: lambda: InvestigationToolProvider(),
        dependencies.get_investigation_repository: lambda: investigations,
        dependencies.get_ingestion_job_repository: lambda: ingestion_jobs,
        dependencies.get_generation_status: lambda: state["generation"],
        dependencies.get_database_status: lambda: "ok",
        dependencies.get_checkpointer: lambda: saver,
    })
    client = ApiClient(CONFIG)
    client._client = TestClient(api_app, base_url=CONFIG.base_url, headers=dict(client._client.headers))
    monkeypatch.setattr(frontend_context, "_cached_client", lambda: client)
    state.update(client=client, repository=repository)
    yield state
    api_app.dependency_overrides.clear()


def seed(client, text="Built Python FastAPI services with PostgreSQL for a fictional logistics project."):
    return client.upload_document("Real Person Name.docx", synthetic_docx(text, "Email: real.person@example.com"))


def test_app_title_and_disclaimer_render(standalone):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert at.exception == []
    assert "Resume Review Assistant" in at.title[0].value
    assert "not hiring decisions" in texts(at) or "No part of this application makes hiring decisions" in texts(at)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_standalone_mode_without_pretending(standalone, page):
    at = run_page(page)
    assert at.exception == []
    body = texts(at)
    assert "Standalone mode" in body
    if page not in {"Overview", "Candidate Ranking"}:
        assert "requires V2 Connected mode" in body


def test_overview_quick_link_navigates(standalone):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    click(at, "Candidate Ranking")
    assert at.sidebar.radio(key="nav").value == "Candidate Ranking"
    assert "Deterministic, explainable scoring" in texts(at)


def test_blind_ranking_results_never_show_filenames(standalone):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["ranking_results"] = [{
        "rank": 1, "candidate": "Candidate 01", "source_filename": "Candidate 01", "score": 81.0,
        "recommendation": "Priority review", "components": {"semantic": 70.0, "required_skills": 100.0,
                                                            "preferred_skills": 50.0, "experience": 100.0},
        "weights": {"semantic": 0.4, "required_skills": 0.35, "preferred_skills": 0.1, "experience": 0.15},
        "matched_required": ["Python"], "missing_required": [], "matched_preferred": ["AWS"],
        "missing_preferred": ["Kubernetes"], "years": 5.0, "experience_level": "Mid-level",
        "evidence": {"Python": ["Built Python APIs."]}, "red_flags": ["No major red flags detected."],
        "suggestions": ["Add impact metrics."], "summary": "Rule-based summary.",
        "genai": {"status": "disabled", "message": "Gemini enrichment is disabled because no API key is configured."},
        "human_decision": "Unreviewed", "human_reason": "",
    }]
    at.session_state["ranking_failures"] = [{"file": "Candidate 02", "error": "The PDF is corrupted."}]
    at.session_state["job_config"] = {"required_skills": ["Python"], "blind_review": True}
    at.run()
    at.sidebar.radio(key="nav").set_value("Candidate Ranking").run()
    assert at.exception == []
    body = texts(at)
    assert "Candidate 01" in body and ".docx" not in body and ".pdf" not in body
    assert any(button.label == "Save human review" for button in at.button)


def test_connected_overview_reports_mode_count_and_generation(connected):
    seed(connected["client"])
    at = run_page("Overview")
    assert at.exception == []
    body = texts(at)
    assert "V2 Connected mode" in body
    assert [metric.value for metric in at.metric][:2] == ["V2 Connected", "1"]
    assert "Grounded generation" in body and "Enabled" in body


def test_library_lists_inspects_and_deletes_with_confirmation(connected):
    ingested = seed(connected["client"])
    at = run_page("Resume Library")
    assert at.exception == []
    assert at.dataframe[0].value["Candidate"].tolist() == [ingested.candidate_ref]
    at.selectbox(key="library_selected").set_value(str(ingested.document_id)).run()
    body = texts(at)
    assert "Version history" in body and "Real Person Name" not in body
    assert "real.person@example.com" not in body + str(at.markdown)

    delete = next(button for button in at.button if button.label == "Delete permanently")
    assert delete.disabled is True  # confirmation required
    at.checkbox(key=f"library_confirm_delete_{ingested.document_id}").check().run()
    click(at, "Delete permanently")
    assert at.exception == []
    assert "was deleted" in texts(at)
    assert connected["client"].list_documents().total == 0


def test_library_empty_state(connected):
    at = run_page("Resume Library")
    assert "No documents are stored for this workspace yet" in texts(at)


def test_library_shows_real_persisted_ingestion_status(connected):
    job = connected["client"].enqueue_document(
        "Fictional Person.docx", synthetic_docx("Synthetic Python evidence."),
        idempotency_key="ui-background-job-001",
    )
    at = run_page("Resume Library")
    assert at.exception == []
    assert str(job.job_id)[:8] in str(at.dataframe[0].value)
    assert "Work is running in the background" in texts(at)
    assert [metric.value for metric in at.metric][:2] == ["1", "0"]


def test_search_shows_provenance_scores_and_relevance_disclaimer(connected):
    ingested = seed(connected["client"])
    at = run_page("AI Candidate Search")
    at.text_input(key="search_query").input("Python services")
    click(at, "Search")
    assert at.exception == []
    body = texts(at)
    assert ingested.candidate_ref in body and "RRF fusion" in body
    assert "not a candidate score" in body


def test_ask_renders_verified_citations_with_exact_excerpts(connected):
    seed(connected["client"])
    at = run_page("Ask AI")
    at.text_area(key="ask_question").input("Which resume mentions Python?")
    click(at, "Ask")
    assert at.exception == []
    body = texts(at)
    assert "[1]" in body and "AI-generated assistance requiring human review" in body
    assert any(expander.label.startswith("[1]") for expander in at.expander)
    assert "real.person@example.com" not in body


def test_ask_disabled_without_generation_policy(connected):
    connected["generation"] = "policy_denied"
    seed(connected["client"])
    at = run_page("Ask AI")
    assert at.exception == []
    assert "disabled for this workspace" in texts(at)
    assert not any(button.label == "Ask" for button in at.button)


def test_brief_generation_review_and_history(connected):
    ingested = seed(connected["client"])
    at = run_page("Recruiter Briefs")
    at.multiselect(key="brief_documents").set_value([str(ingested.document_id)])
    at.text_area(key="brief_description").input("Python backend engineer")
    at.text_area(key="brief_required").input("Python, Kubernetes")
    click(at, "Generate briefs for review")
    assert at.exception == []
    body = texts(at)
    assert "Awaiting review" in body and "Requirement-by-requirement evidence" in body
    assert "not the applicant" in body and "Editing generated briefs is not supported" in body
    workflow_id = connected["client"].list_workflows().workflows[0].workflow_id

    at.text_area(key=f"review_note_{workflow_id}").input("Citations checked")
    click(at, "Submit review")
    assert at.exception == []
    assert "were **approved** by a reviewer" in texts(at)

    history = run_page("Review History")
    assert history.exception == []
    assert history.dataframe[0].value["Candidates"].tolist() == [ingested.candidate_ref]
    history.selectbox(key="history_selected").set_value(str(workflow_id)).run()
    assert "Citations checked" in texts(history)


def test_investigation_setup_trace_report_review_and_reopen(connected):
    ingested = seed(connected["client"])
    at = run_page("AI Investigations")
    at.multiselect(key="investigation_documents").set_value([str(ingested.document_id)])
    at.text_area(key="investigation_objective").input("Investigate Python API evidence")
    at.text_area(key="investigation_job_description").input("Fictional backend platform role")
    at.text_area(key="investigation_requirements").input("Python, Kubernetes")
    click(at, "Start Investigation")
    assert at.exception == []
    body = texts(at)
    assert "Execution trace" in body
    assert "Supported findings" in body
    assert "Missing or insufficient evidence" in body
    assert "Requirement-to-evidence matrix" in body
    assert "Suggested human verification questions" in body
    assert "not a decision about any applicant" in body
    assert "real.person@example.com" not in body
    trace = next(frame.value for frame in at.dataframe if "Tool" in frame.value.columns)
    assert {"search_evidence", "check_requirements", "draft_investigation"} <= set(trace["Tool"].tolist())

    investigation_id = connected["client"].list_investigations().investigations[0].investigation_id
    at.text_area(key=f"investigation_comment_{investigation_id}").input("Verified fictional evidence")
    click(at, "Submit investigation review")
    assert at.exception == []
    assert "were **approved** by a reviewer" in texts(at)

    reopened = run_page("AI Investigations")
    reopened.selectbox(key="investigation_history_selected").set_value(str(investigation_id)).run()
    assert reopened.exception == []
    assert "Verified fictional evidence" in texts(reopened)


def test_investigation_generation_disabled_preserves_history_state(connected):
    connected["generation"] = "policy_denied"
    seed(connected["client"])
    at = run_page("AI Investigations")
    assert at.exception == []
    assert "disabled for this workspace" in texts(at)
    assert next(button for button in at.button if button.label == "Start Investigation").disabled is True


def test_api_unavailable_is_shown_without_pretending(monkeypatch):
    class DownClient:
        configured = True

        def health(self):
            raise ApiError("unreachable", "The V2 API is unreachable. Check that the backend is running.")

    monkeypatch.setattr(frontend_context, "_cached_client", lambda: DownClient())
    for page in ("Overview", "Resume Library", "Ask AI"):
        at = run_page(page)
        assert at.exception == []
        body = texts(at)
        assert "V2 unavailable" in body and "unreachable" in body
        assert "Upload and index" not in [button.label for button in at.button]


def test_unknown_citation_reference_is_not_rendered_as_evidence(connected):
    from api import schemas

    answer = schemas.GroundedAnswerResponse(
        claims=[schemas.Claim(text="Unsupported", citations=[schemas.Citation(source_chunk_id=str(uuid4()),
                                                                               quote="ghost")])],
        sources=[],
    )

    def script():
        import streamlit as st
        from frontend.components import render_answer
        render_answer(st.session_state["answer"], key="t")

    at = AppTest.from_function(script, default_timeout=10)
    at.session_state["answer"] = answer
    at.run()
    assert "unknown source" in texts(at)
    assert "[?]" in texts(at)
