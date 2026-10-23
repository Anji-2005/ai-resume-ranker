"""FastAPI contract tests with in-memory fakes: auth, tenancy, roles, policy, errors, workflows."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from api import dependencies
from api.main import app
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


TOKENS = (
    '{"full-token":["tenant-a"],'
    '"reader-token":{"tenants":["tenant-a"],"roles":["reader"]},'
    '"other-token":["tenant-b"]}'
)
FULL = {"X-Tenant-ID": "tenant-a", "Authorization": "Bearer full-token"}
READER = {"X-Tenant-ID": "tenant-a", "Authorization": "Bearer reader-token"}
OTHER = {"X-Tenant-ID": "tenant-b", "Authorization": "Bearer other-token"}
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
RESUME = synthetic_docx(
    "Built Python FastAPI services with PostgreSQL for a fictional logistics project.",
    "Contact: jane.synthetic@example.com",
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_AUTH_TOKENS_JSON", TOKENS)
    repository = InMemoryDocumentRepository()
    state = {
        "repository": repository, "runs": InMemoryRunRepository(), "reviews": InMemoryReviewRepository(),
        "provider": CitingProvider(), "generation": "enabled", "checkpointer": InMemorySaver(),
        "investigations": InMemoryInvestigationRepository(),
        "ingestion_jobs": InMemoryIngestionJobRepository(),
        "investigation_provider": InvestigationToolProvider(),
    }
    overrides = {
        dependencies.get_document_repository: lambda: state["repository"],
        dependencies.get_retrieval_pipeline: lambda: RetrievalPipeline(state["repository"], embedder=FakeEmbedder()),
        dependencies.get_run_repository: lambda: state["runs"],
        dependencies.get_review_repository: lambda: state["reviews"],
        dependencies.get_generation_provider: lambda: state["provider"],
        dependencies.get_investigation_provider: lambda: state["investigation_provider"],
        dependencies.get_investigation_repository: lambda: state["investigations"],
        dependencies.get_ingestion_job_repository: lambda: state["ingestion_jobs"],
        dependencies.get_generation_status: lambda: state["generation"],
        dependencies.get_database_status: lambda: "ok",
        dependencies.get_checkpointer: lambda: state["checkpointer"],
    }
    app.dependency_overrides.update(overrides)
    state["client"] = TestClient(app)
    yield state
    app.dependency_overrides.clear()


def upload(client, headers=FULL, data=RESUME, name="Jane Synthetic CV.docx"):
    return client.post("/v1/documents", files={"file": (name, data, DOCX)}, headers=headers)


def test_health_is_unauthenticated_and_reports_capabilities_without_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    body = TestClient(app).get("/v1/health").json()
    assert body["generation_provider_configured"] is True
    assert "super-secret-key" not in str(body)
    assert body["database"] in {"ok", "unavailable"}


def test_session_reports_roles_and_generation_status(env):
    body = env["client"].get("/v1/session", headers=READER).json()
    assert body == {"tenant_id": "tenant-a", "roles": ["reader"], "generation": "enabled",
                    "generation_model": body["generation_model"]}


@pytest.mark.parametrize("headers,expected", [
    ({"X-Tenant-ID": "tenant-a"}, 401),
    ({"X-Tenant-ID": "tenant-a", "Authorization": "Bearer unknown"}, 403),
    ({"X-Tenant-ID": "tenant-b", "Authorization": "Bearer full-token"}, 403),
])
def test_every_candidate_route_fails_closed(env, headers, expected):
    client = env["client"]
    document_id, workflow_id = uuid4(), uuid4()
    responses = [
        client.get("/v1/documents", headers=headers),
        client.get(f"/v1/documents/{document_id}", headers=headers),
        client.post("/v1/documents/search", json={"query": "python"}, headers=headers),
        client.post("/v1/documents/qa", json={"question": "python?"}, headers=headers),
        client.get("/v1/workflows", headers=headers),
        client.get(f"/v1/workflows/{workflow_id}", headers=headers),
        client.post(f"/v1/workflows/{workflow_id}/review", json={"approved": True}, headers=headers),
        client.get("/v1/investigations", headers=headers),
        client.get(f"/v1/investigations/{workflow_id}", headers=headers),
        client.post("/v1/investigations", headers={**headers, "Idempotency-Key": str(uuid4())},
                    json={"objective": "Python evidence", "document_ids": [str(document_id)]}),
        client.post(f"/v1/investigations/{workflow_id}/review", json={"approved": True}, headers=headers),
        client.get("/v1/ingestion-jobs", headers=headers),
        client.get("/v1/ingestion-jobs/metrics", headers=headers),
        client.get(f"/v1/ingestion-jobs/{workflow_id}", headers=headers),
        client.post(
            "/v1/ingestion-jobs", headers={**headers, "Idempotency-Key": "denied-ingest-001"},
            files={"file": ("document.docx", RESUME, DOCX)},
        ),
        client.post(
            f"/v1/documents/{document_id}/reindex",
            headers={**headers, "Idempotency-Key": "denied-reindex-001"}, json={},
        ),
        client.post(
            f"/v1/documents/{document_id}/ingestion-jobs",
            headers={**headers, "Idempotency-Key": "denied-version-001"},
            files={"file": ("document.docx", RESUME, DOCX)},
        ),
        client.delete(f"/v1/documents/{document_id}", headers=headers),
    ]
    assert {response.status_code for response in responses} == {expected}
    assert all("full-token" not in response.text for response in responses)


def test_upload_persists_opaque_filename_and_is_idempotent(env):
    first, second = upload(env["client"]), upload(env["client"])
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["created"] is True and second.json()["created"] is False
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["candidate_ref"].startswith("CAND-")
    assert set(env["repository"].ingested_filenames) == {"document.docx"}


def test_reader_cannot_upload_or_delete(env):
    document_id = upload(env["client"]).json()["document_id"]
    assert upload(env["client"], headers=READER).status_code == 403
    assert env["client"].delete(f"/v1/documents/{document_id}", headers=READER).status_code == 403


def test_background_ingestion_jobs_are_idempotent_tenant_scoped_and_observable(env):
    key = "synthetic-background-job-001"
    first = env["client"].post(
        "/v1/ingestion-jobs", headers={**FULL, "Idempotency-Key": key},
        files={"file": ("Identity Bearing Name.docx", RESUME, DOCX)},
    )
    assert first.status_code == 202, first.text
    body = first.json()
    assert body["status"] == "queued" and body["execution_mode"] == "background_worker"
    assert "Identity Bearing" not in first.text

    replay = env["client"].post(
        "/v1/ingestion-jobs", headers={**FULL, "Idempotency-Key": key},
        files={"file": ("different-name.docx", RESUME, DOCX)},
    )
    assert replay.status_code == 200 and replay.json()["job_id"] == body["job_id"]
    conflict = env["client"].post(
        "/v1/ingestion-jobs", headers={**FULL, "Idempotency-Key": key},
        files={"file": ("document.docx", synthetic_docx("Different fictional text."), DOCX)},
    )
    assert conflict.status_code == 409
    assert env["client"].get(f"/v1/ingestion-jobs/{body['job_id']}", headers=OTHER).status_code == 404
    assert env["client"].get("/v1/ingestion-jobs", headers=OTHER).json()["jobs"] == []
    metrics = env["client"].get("/v1/ingestion-jobs/metrics", headers=FULL).json()
    assert metrics["counts"]["queued"] == 1
    assert env["client"].post(
        "/v1/ingestion-jobs", headers={**READER, "Idempotency-Key": "reader-denied-001"},
        files={"file": ("document.docx", RESUME, DOCX)},
    ).status_code == 403


def test_reindex_job_requires_tenant_owned_document(env):
    document_id = upload(env["client"]).json()["document_id"]
    queued = env["client"].post(
        f"/v1/documents/{document_id}/reindex",
        headers={**FULL, "Idempotency-Key": "reindex-contract-001"},
        json={"version_number": 1},
    )
    assert queued.status_code == 202 and queued.json()["job_type"] == "reindex"
    assert env["client"].post(
        f"/v1/documents/{document_id}/reindex",
        headers={**OTHER, "Idempotency-Key": "reindex-contract-002"},
        json={"version_number": 1},
    ).status_code == 404


def test_invalid_upload_returns_safe_message(env):
    response = upload(env["client"], data=b"not a docx", name="secret-name.docx")
    assert response.status_code == 422
    assert "secret-name" not in response.text


def test_list_detail_and_versions_are_tenant_scoped_and_hide_identity(env):
    client = env["client"]
    document_id = upload(client).json()["document_id"]
    changed = synthetic_docx("Built Python and Go services for a fictional project.")
    version = client.post(f"/v1/documents/{document_id}/versions", files={"file": ("x.docx", changed, DOCX)},
                          headers=FULL)
    assert version.status_code == 201 and version.json()["version_number"] == 2

    listing = client.get("/v1/documents", headers=FULL).json()
    assert listing["total"] == 1
    assert listing["documents"][0]["version_count"] == 2
    assert "filename" not in str(listing) and "source_key" not in str(listing)

    detail = client.get(f"/v1/documents/{document_id}?version=1", headers=FULL).json()
    assert [v["version_number"] for v in detail["versions"]] == [2, 1]
    assert detail["selected_version_number"] == 1
    assert "jane.synthetic@example.com" not in str(detail)
    assert "[EMAIL REDACTED]" in str(detail)

    assert client.get("/v1/documents", headers=OTHER).json()["total"] == 0
    assert client.get(f"/v1/documents/{document_id}", headers=OTHER).status_code == 404
    assert client.post(f"/v1/documents/{document_id}/versions", files={"file": ("x.docx", changed, DOCX)},
                       headers=OTHER).status_code == 404
    assert client.delete(f"/v1/documents/{document_id}", headers=OTHER).status_code == 404


def test_post_search_returns_provenance_scores_and_respects_document_filter(env):
    client = env["client"]
    first = upload(client).json()["document_id"]
    upload(client, data=synthetic_docx("Python data pipelines for a second fictional candidate."))
    results = client.post("/v1/documents/search", json={"query": "Python", "document_ids": [first]},
                          headers=FULL).json()
    assert results and {item["document_id"] for item in results} == {first}
    item = results[0]
    assert item["source_chunk_id"] == item["chunk_id"]
    assert item["fusion_score"] > 0 and item["reranker_score"] is None
    assert item["candidate_ref"].startswith("CAND-")


def test_delete_removes_document(env):
    client = env["client"]
    document_id = upload(client).json()["document_id"]
    assert client.delete(f"/v1/documents/{document_id}", headers=FULL).status_code == 204
    assert client.get(f"/v1/documents/{document_id}", headers=FULL).status_code == 404


def test_qa_returns_verified_claims_with_supplied_sources(env):
    client = env["client"]
    upload(client)
    body = client.post("/v1/documents/qa", json={"question": "Python services?"}, headers=FULL).json()
    citation = body["claims"][0]["citations"][0]
    source = next(s for s in body["sources"] if s["source_chunk_id"] == citation["source_chunk_id"])
    assert citation["quote"] in source["excerpt"]
    assert body["generation"]["citations_verified"] is True
    assert body["generation"]["prompt_version"] == "document-qa-v2"
    # The provider never saw the direct identifier.
    assert "jane.synthetic@example.com" not in env["provider"].prompts[0]


def test_qa_without_evidence_is_explicitly_uncertain(env):
    body = env["client"].post("/v1/documents/qa", json={"question": "Kubernetes?"}, headers=FULL).json()
    assert body["claims"] == [] and "No authorized supporting evidence" in body["uncertainty"]
    assert env["provider"].prompts == []


@pytest.mark.parametrize("generation,expected", [("unconfigured", 503), ("policy_denied", 403)])
def test_generation_gate_fails_closed_before_provider(env, generation, expected):
    env["generation"] = generation
    upload(env["client"])
    responses = [
        env["client"].post("/v1/documents/qa", json={"question": "Python?"}, headers=FULL),
        env["client"].post("/v1/documents/recruiter-brief", json={"description": "Python"}, headers=FULL),
        env["client"].post("/v1/workflows/recruiter-brief",
                           json={"description": "Python", "document_ids": [str(uuid4())]}, headers=FULL),
        env["client"].post("/v1/investigations", headers={**FULL, "Idempotency-Key": str(uuid4())},
                           json={"objective": "Python", "document_ids": [str(uuid4())]}),
    ]
    assert {response.status_code for response in responses} == {expected}
    assert env["provider"].prompts == []


def test_provider_failure_and_fabricated_citations_return_safe_errors(env):
    upload(env["client"])
    env["provider"] = CitingProvider(fail=True)
    failed = env["client"].post("/v1/documents/qa", json={"question": "Python?"}, headers=FULL)
    assert failed.status_code == 502 and "SECRET" not in failed.text

    env["provider"] = CitingProvider(quote_from_source=False)
    fabricated = env["client"].post("/v1/documents/qa", json={"question": "Python?"}, headers=FULL)
    assert fabricated.status_code == 422
    assert "Invented quote" not in fabricated.text
    assert "withheld" in fabricated.json()["detail"]


def test_direct_brief_endpoint_returns_requirements_and_sources(env):
    upload(env["client"])
    body = env["client"].post("/v1/documents/recruiter-brief",
                              json={"description": "Python services", "required_skills": ["Python"]},
                              headers=FULL).json()
    assert {item["status"] for item in body["requirement_evidence"]} == {"supported", "unsupported"}
    assert body["sources"] and body["generation"]["prompt_version"] == "recruiter-brief-v2"


def start_workflow(env, document_id, headers=FULL):
    return env["client"].post("/v1/workflows/recruiter-brief", json={
        "description": "Python services", "required_skills": ["Python"], "document_ids": [document_id],
    }, headers=headers)


def test_workflow_start_interrupt_review_and_duplicate_protection(env):
    client = env["client"]
    document_id = upload(client).json()["document_id"]
    started = start_workflow(env, document_id)
    assert started.status_code == 201
    view = started.json()["workflows"][0]
    workflow_id = view["workflow_id"]
    assert view["status"] == "awaiting_review"
    assert view["brief"]["claims"] and view["sources"]
    assert "not a hiring decision" in view["instruction"]
    assert "subject" not in str(view) and "permitted_tenant" not in str(view)

    pending = client.get("/v1/workflows?status=awaiting_review", headers=FULL).json()["workflows"]
    assert [item["workflow_id"] for item in pending] == [workflow_id]

    assert client.post(f"/v1/workflows/{workflow_id}/review", json={"approved": True},
                       headers=READER).status_code == 403
    reviewed = client.post(f"/v1/workflows/{workflow_id}/review",
                           json={"approved": False, "note": "Evidence too thin"}, headers=FULL)
    assert reviewed.status_code == 200
    body = reviewed.json()
    assert body["status"] == "completed" and body["outcome"] == "rejected"
    assert body["review"]["note"] == "Evidence too thin"

    duplicate = client.post(f"/v1/workflows/{workflow_id}/review", json={"approved": True}, headers=FULL)
    assert duplicate.status_code == 409
    assert len(env["reviews"].records) == 1
    record = next(iter(env["reviews"].records.values()))
    assert record.subject.startswith("token-sha256:") and "full-token" not in record.subject

    assert client.get(f"/v1/workflows/{workflow_id}", headers=FULL).json()["outcome"] == "rejected"


def start_investigation(env, document_id, *, key=None, headers=FULL, objective="Investigate Python evidence"):
    return env["client"].post(
        "/v1/investigations",
        headers={**headers, "Idempotency-Key": key or str(uuid4())},
        json={
            "objective": objective,
            "document_ids": [str(document_id)],
            "job_description": "Fictional Python platform role",
            "requirements": ["Python"],
        },
    )


def test_investigation_creation_trace_report_history_and_idempotency(env):
    document_id = UUID(upload(env["client"]).json()["document_id"])
    key = str(uuid4())
    started = start_investigation(env, document_id, key=key)
    assert started.status_code == 201, started.text
    body = started.json()
    investigation_id = body["investigation_id"]
    assert body["status"] == "awaiting_review"
    assert body["execution_mode"] == "synchronous"
    assert [event["tool_name"] for event in body["events"]] == [
        "search_evidence", "check_requirements", "draft_investigation",
    ]
    assert body["report"]["supported_findings"]
    assert body["report"]["requirement_evidence"][0]["status"] == "supported"
    citation = body["report"]["supported_findings"][0]["citations"][0]
    source = next(item for item in body["sources"] if item["source_chunk_id"] == citation["source_chunk_id"])
    assert citation["quote"] in source["excerpt"]
    assert body["scope"][0]["version_number"] == 1
    assert "prompt" not in str(body["events"]).casefold()

    repeated = start_investigation(env, document_id, key=key)
    assert repeated.status_code == 200
    assert repeated.json()["investigation_id"] == investigation_id
    conflict = start_investigation(env, document_id, key=key, objective="Different request")
    assert conflict.status_code == 409

    history = env["client"].get("/v1/investigations?status=awaiting_review", headers=FULL).json()
    assert [item["investigation_id"] for item in history["investigations"]] == [investigation_id]
    assert env["client"].get(f"/v1/investigations/{investigation_id}", headers=OTHER).status_code == 404


def test_investigation_review_requires_reviewer_and_is_note_review_only(env):
    document_id = UUID(upload(env["client"]).json()["document_id"])
    view = start_investigation(env, document_id).json()
    investigation_id = view["investigation_id"]
    assert env["client"].post(
        f"/v1/investigations/{investigation_id}/review",
        headers=READER,
        json={"approved": True},
    ).status_code == 403
    reviewed = env["client"].post(
        f"/v1/investigations/{investigation_id}/review",
        headers=FULL,
        json={"approved": False, "comment": "Evidence gap confirmed"},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "completed"
    assert reviewed.json()["outcome"] == "rejected"
    assert reviewed.json()["review"]["comment"] == "Evidence gap confirmed"
    assert env["client"].post(
        f"/v1/investigations/{investigation_id}/review",
        headers=FULL,
        json={"approved": True},
    ).status_code == 409


def test_investigation_provider_failure_is_safe_and_has_no_report(env):
    document_id = UUID(upload(env["client"]).json()["document_id"])
    env["investigation_provider"] = InvestigationToolProvider(fail=True)
    response = start_investigation(env, document_id)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed" and body["report"] is None
    assert body["error_code"] == "provider_failed"
    assert "confidential" not in response.text.casefold()


def test_workflow_is_invisible_to_other_tenants(env):
    document_id = upload(env["client"]).json()["document_id"]
    workflow_id = start_workflow(env, document_id).json()["workflows"][0]["workflow_id"]
    assert env["client"].get(f"/v1/workflows/{workflow_id}", headers=OTHER).status_code == 404
    assert env["client"].post(f"/v1/workflows/{workflow_id}/review", json={"approved": True},
                              headers=OTHER).status_code == 404
    assert env["client"].get("/v1/workflows", headers=OTHER).json()["workflows"] == []


def test_workflow_rejects_foreign_documents_and_reports_failures_safely(env):
    other_document = upload(env["client"], headers=OTHER).json()["document_id"]
    assert start_workflow(env, other_document).status_code == 404

    document_id = upload(env["client"]).json()["document_id"]
    env["provider"] = CitingProvider(quote_from_source=False)
    view = start_workflow(env, document_id).json()["workflows"][0]
    assert view["status"] == "failed" and view["error_code"] == "citation_failed"
    assert view["brief"] is None and "Invented" not in str(view)
    review = env["client"].post(f"/v1/workflows/{view['workflow_id']}/review", json={"approved": True}, headers=FULL)
    assert review.status_code == 409


def test_review_payload_is_validated(env):
    document_id = upload(env["client"]).json()["document_id"]
    workflow_id = start_workflow(env, document_id).json()["workflows"][0]["workflow_id"]
    for payload in ({"approved": "maybe"}, {"approved": True, "note": "x" * 2_001}, {}):
        assert env["client"].post(f"/v1/workflows/{workflow_id}/review", json=payload,
                                  headers=FULL).status_code == 422
