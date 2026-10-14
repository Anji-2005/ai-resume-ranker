"""Streamlit API client: mocked HTTP behavior and a live contract round trip against FastAPI."""

from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api import dependencies
from api.main import app
from frontend.api_client import ApiClient, ApiConfig, ApiError, config_from_env
from retrieval.pipeline import RetrievalPipeline

from fakes import (
    CitingProvider,
    FakeEmbedder,
    InMemoryDocumentRepository,
    InMemoryReviewRepository,
    InMemoryRunRepository,
    synthetic_docx,
)


CONFIG = ApiConfig("http://api.test", "tenant-a", "secret-token-value")


def client_with(handler, config=CONFIG):
    calls = []

    def recording(request):
        calls.append(request)
        return handler(request)

    return ApiClient(config, transport=httpx.MockTransport(recording), sleep=lambda _: None), calls


def test_config_requires_base_url_for_connected_mode(monkeypatch):
    for name in ("API_BASE_URL", "API_TENANT_ID", "API_BEARER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert config_from_env() is None
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    config = config_from_env()
    assert config.base_url == "http://api:8000" and config.token == ""
    assert "secret" not in repr(CONFIG)


def test_headers_are_sent_and_search_query_stays_out_of_url():
    client, calls = client_with(lambda request: httpx.Response(200, json=[]))
    assert client.search("confidential python query", limit=5) == []
    request = calls[0]
    assert request.method == "POST"
    assert "confidential" not in str(request.url)
    assert request.headers["Authorization"] == "Bearer secret-token-value"
    assert request.headers["X-Tenant-ID"] == "tenant-a"


def test_safe_reads_retry_transient_failures():
    responses = iter([httpx.Response(503), httpx.Response(200, json={"documents": [], "total": 0})])
    client, calls = client_with(lambda request: next(responses))
    assert client.list_documents().total == 0
    assert len(calls) == 2


@pytest.mark.parametrize("operation", [
    lambda c: c.delete_document(uuid4()),
    lambda c: c.upload_document("cv.pdf", b"%PDF"),
    lambda c: c.ask("question"),
    lambda c: c.submit_review(uuid4(), approved=True),
])
def test_mutating_and_generation_calls_are_never_retried(operation):
    client, calls = client_with(lambda request: httpx.Response(503, json={"detail": "busy"}))
    with pytest.raises(ApiError):
        operation(client)
    assert len(calls) == 1


def test_errors_are_mapped_without_leaking_credentials():
    client, _ = client_with(lambda request: httpx.Response(403, json={"detail": [{"loc": "x", "input": "secret"}]}))
    with pytest.raises(ApiError) as error:
        client.list_documents()
    assert error.value.kind == "forbidden" and error.value.status == 403
    assert "secret" not in error.value.message


def test_timeouts_and_unreachable_api_produce_clear_messages():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, calls = client_with(timeout)
    with pytest.raises(ApiError, match="did not respond") as error:
        client.get_document(uuid4())
    assert error.value.kind == "timeout" and len(calls) == 3

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    client, _ = client_with(refused)
    with pytest.raises(ApiError, match="unreachable"):
        client.delete_document(uuid4())


def test_missing_credentials_fail_closed_without_network():
    client, calls = client_with(lambda request: httpx.Response(200), ApiConfig("http://api.test", "", ""))
    with pytest.raises(ApiError) as error:
        client.list_documents()
    assert error.value.kind == "unauthenticated" and calls == []


def test_upload_never_sends_original_filename():
    seen = {}

    def handler(request):
        seen["body"] = request.content
        return httpx.Response(201, json={"document_id": str(uuid4()), "version_id": str(uuid4()),
                                         "version_number": 1, "created": True, "embedding_model": "m"})

    client, _ = client_with(handler)
    client.upload_document("Jane Doe Resume.DOCX", b"bytes")
    assert b"Jane Doe" not in seen["body"] and b'filename="document.docx"' in seen["body"]


def test_contract_violation_is_reported():
    client, _ = client_with(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ApiError) as error:
        client.session()
    assert error.value.kind == "contract"


@pytest.fixture
def live_client(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_AUTH_TOKENS_JSON", '{"secret-token-value":["tenant-a"]}')
    repository, runs, reviews, saver = (
        InMemoryDocumentRepository(), InMemoryRunRepository(), InMemoryReviewRepository(), InMemorySaver()
    )
    app.dependency_overrides.update({
        dependencies.get_document_repository: lambda: repository,
        dependencies.get_retrieval_pipeline: lambda: RetrievalPipeline(repository, embedder=FakeEmbedder()),
        dependencies.get_run_repository: lambda: runs,
        dependencies.get_review_repository: lambda: reviews,
        dependencies.get_generation_provider: lambda: CitingProvider(),
        dependencies.get_generation_status: lambda: "enabled",
        dependencies.get_database_status: lambda: "ok",
        dependencies.get_checkpointer: lambda: saver,
    })
    client = ApiClient(CONFIG)
    client._client = TestClient(app, base_url=CONFIG.base_url, headers=dict(client._client.headers))
    yield client
    app.dependency_overrides.clear()


def test_client_round_trip_against_fastapi_contract(live_client):
    data = synthetic_docx("Built Python FastAPI services with PostgreSQL for a fictional project.")
    ingested = live_client.upload_document("synthetic.docx", data)
    assert live_client.list_documents().documents[0].candidate_ref == ingested.candidate_ref
    detail = live_client.get_document(ingested.document_id)
    assert detail.chunks and detail.versions[0].is_current
    assert live_client.search("Python", document_ids=[ingested.document_id])[0].document_id == ingested.document_id
    answer = live_client.ask("Python?", document_ids=[ingested.document_id])
    assert answer.claims and answer.sources
    started = live_client.start_brief_workflows(
        description="Python services", required_skills=["Python"], preferred_skills=[],
        document_ids=[ingested.document_id],
    ).workflows[0]
    assert started.status == "awaiting_review"
    assert live_client.list_workflows("awaiting_review").workflows[0].workflow_id == started.workflow_id
    reviewed = live_client.submit_review(started.workflow_id, approved=True, note="Checked")
    assert reviewed.status == "completed" and reviewed.review.decision == "approved"
    with pytest.raises(ApiError) as duplicate:
        live_client.submit_review(started.workflow_id, approved=True)
    assert duplicate.value.status == 409
    live_client.delete_document(ingested.document_id)
    assert live_client.list_documents().total == 0
