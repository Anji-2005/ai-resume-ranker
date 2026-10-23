"""End-to-end API integration against real PostgreSQL + pgvector + LangGraph PostgresSaver.

Synthetic documents and a fake LLM provider only. Skipped unless TEST_DATABASE_URL
points at an isolated database.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if not TEST_DATABASE_URL:
    pytest.skip("TEST_DATABASE_URL is not configured", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi import Depends  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from api import dependencies  # noqa: E402
from api.main import app  # noqa: E402
from db.session import get_session_factory, reset_session_factory  # noqa: E402
from repositories.documents import DocumentRepository  # noqa: E402
from repositories.ingestion_jobs import IngestionJobRepository  # noqa: E402
from ingestion.worker import IngestionWorker  # noqa: E402
from retrieval.pipeline import RetrievalPipeline  # noqa: E402
from workflows.checkpointing import close_pooled_checkpointers, pooled_postgres_checkpointer  # noqa: E402

from fakes import CitingProvider, InvestigationToolProvider, synthetic_docx  # noqa: E402


HEADERS = {"X-Tenant-ID": "tenant-a", "Authorization": "Bearer it-token-a"}
OTHER = {"X-Tenant-ID": "tenant-b", "Authorization": "Bearer it-token-b"}
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class FeatureEmbedder:
    model_name = "test-feature-embedder-v1"
    model_version = "1.0-test"

    def embed(self, texts):
        vectors = []
        for value in texts:
            lowered = value.casefold()
            vector = [0.01] * 384
            vector[0] = 1.0 if "python" in lowered else 0.0
            vector[1] = 1.0 if "sql" in lowered or "postgres" in lowered else 0.0
            vectors.append(vector)
        return vectors


def repository_override(session=Depends(dependencies.get_session)):
    return DocumentRepository(session, embedder=FeatureEmbedder())


def pipeline_override(repository=Depends(dependencies.get_document_repository)):
    return RetrievalPipeline(repository, embedder=FeatureEmbedder())


@pytest.fixture(scope="module", autouse=True)
def database():
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(config, "head")
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    reset_session_factory()
    pooled_postgres_checkpointer(TEST_DATABASE_URL)  # creates checkpoint tables once
    yield
    close_pooled_checkpointers()
    reset_session_factory()
    if previous is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous


@pytest.fixture
def client(monkeypatch):
    with get_session_factory()() as session:
        session.execute(text(
            "TRUNCATE ingestion_jobs, investigation_events, investigation_reviews, investigation_runs, "
            "document_chunks, document_versions, documents, workflow_runs, workflow_reviews, "
            "checkpoints, checkpoint_blobs, checkpoint_writes CASCADE"
        ))
        session.commit()
    monkeypatch.setenv("RETRIEVAL_AUTH_TOKENS_JSON", '{"it-token-a":["tenant-a"],"it-token-b":["tenant-b"]}')
    provider = CitingProvider()
    app.dependency_overrides.update({
        dependencies.get_document_repository: repository_override,
        dependencies.get_retrieval_pipeline: pipeline_override,
        dependencies.get_generation_provider: lambda: provider,
        dependencies.get_investigation_provider: lambda: InvestigationToolProvider(),
        dependencies.get_generation_status: lambda: "enabled",
        dependencies.get_checkpointer: lambda: pooled_postgres_checkpointer(TEST_DATABASE_URL),
    })
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def scalar(sql, **params):
    with get_session_factory()() as session:
        return session.execute(text(sql), params).scalar()


def upload(client, data, headers=HEADERS, name="Synthetic Person CV.docx"):
    response = client.post("/v1/documents", files={"file": (name, data, DOCX)}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_reports_database_ok(client):
    assert client.get("/v1/health").json()["database"] == "ok"


def test_document_lifecycle_persistence_and_tenant_isolation(client):
    data = synthetic_docx("Built Python services with PostgreSQL.", "Email: a.person@example.com")
    first = upload(client, data)
    again = upload(client, data)
    assert again["created"] is False and again["document_id"] == first["document_id"]
    document_id = first["document_id"]
    assert scalar("SELECT filename FROM document_versions WHERE document_id = :id", id=document_id) == "document.docx"

    version = client.post(f"/v1/documents/{document_id}/versions", headers=HEADERS,
                          files={"file": ("v2.docx", synthetic_docx("Python and SQL reporting."), DOCX)})
    assert version.status_code == 201 and version.json()["version_number"] == 2
    upload(client, synthetic_docx("Python confidential work for tenant b."), headers=OTHER)

    listing = client.get("/v1/documents", headers=HEADERS).json()
    assert listing["total"] == 1 and listing["documents"][0]["version_count"] == 2
    assert listing["documents"][0]["current_version_number"] == 2
    detail = client.get(f"/v1/documents/{document_id}?version=1", headers=HEADERS).json()
    assert [v["is_current"] for v in detail["versions"]] == [True, False]
    assert "a.person@example.com" not in str(detail)

    results = client.post("/v1/documents/search", json={"query": "Python SQL"}, headers=HEADERS).json()
    assert results and {item["document_id"] for item in results} == {document_id}
    assert all(item["content"].startswith("Python and SQL") for item in results)  # current version only
    assert client.get(f"/v1/documents/{document_id}", headers=OTHER).status_code == 404

    assert client.delete(f"/v1/documents/{document_id}", headers=HEADERS).status_code == 204
    assert scalar("SELECT count(*) FROM document_chunks WHERE document_id = :id", id=document_id) == 0
    assert scalar("SELECT count(*) FROM document_versions WHERE document_id = :id", id=document_id) == 0


def enqueue_background(client, data, *, key="integration-ingestion-job-001", headers=HEADERS):
    response = client.post(
        "/v1/ingestion-jobs", headers={**headers, "Idempotency-Key": key},
        files={"file": ("Fictional Person.docx", data, DOCX)},
    )
    assert response.status_code in {200, 202}, response.text
    return response.json()


def test_background_worker_idempotency_completion_metrics_and_tenant_isolation(client):
    data = synthetic_docx("Built Python services with PostgreSQL for a fictional project.")
    queued = enqueue_background(client, data)
    replay = enqueue_background(client, data)
    assert replay["job_id"] == queued["job_id"]
    assert client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=OTHER).status_code == 404

    assert IngestionWorker(
        get_session_factory(), embedder_factory=FeatureEmbedder, lease_seconds=30,
    ).run_once() is True
    completed = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    assert completed["status"] == "completed" and completed["attempt_count"] == 1
    assert completed["candidate_ref"].startswith("CAND-")
    assert scalar("SELECT payload IS NULL FROM ingestion_jobs WHERE id = :id", id=queued["job_id"]) is True
    assert scalar("SELECT embedding_model_version FROM document_versions") == "1.0-test"
    metrics = client.get("/v1/ingestion-jobs/metrics", headers=HEADERS).json()
    assert metrics["counts"]["completed"] == 1 and metrics["completed_last_hour"] == 1


def test_worker_retries_embedding_failure_with_backoff_then_completes(client):
    queued = enqueue_background(
        client, synthetic_docx("Synthetic retry evidence."), key="integration-retry-job-001",
    )

    class FlakyEmbedder(FeatureEmbedder):
        calls = 0

        def embed(self, texts):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("transient provider failure with confidential payload")
            return super().embed(texts)

    worker = IngestionWorker(
        get_session_factory(), embedder_factory=FlakyEmbedder,
        lease_seconds=30, base_backoff_seconds=5,
    )
    assert worker.run_once() is True
    retry = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    assert retry["status"] == "queued" and retry["attempt_count"] == 1
    assert retry["next_attempt_at"] is not None and "confidential" not in str(retry)
    with get_session_factory().begin() as session:
        session.execute(text("UPDATE ingestion_jobs SET available_at = now() WHERE id = :id"),
                        {"id": queued["job_id"]})
    assert worker.run_once() is True
    completed = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    assert completed["status"] == "completed" and completed["attempt_count"] == 2


def test_expired_processing_lease_recovers_after_worker_crash(client):
    queued = enqueue_background(
        client, synthetic_docx("Synthetic crash recovery evidence."), key="integration-crash-job-001",
    )
    with get_session_factory().begin() as session:
        claimed = IngestionJobRepository(session).claim_next("dead-worker", lease_seconds=30)
        assert str(claimed.id) == queued["job_id"]
        session.execute(text("UPDATE ingestion_jobs SET lease_expires_at = now() - interval '1 second' "
                             "WHERE id = :id"), {"id": queued["job_id"]})
    assert IngestionWorker(
        get_session_factory(), embedder_factory=FeatureEmbedder, lease_seconds=30,
    ).run_once() is True
    recovered = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    assert recovered["status"] == "completed" and recovered["attempt_count"] == 2


def test_terminal_data_quality_failure_is_safe_atomic_and_purges_payload(client):
    queued = enqueue_background(
        client, synthetic_docx(""), key="integration-empty-job-001",
    )
    assert IngestionWorker(
        get_session_factory(), embedder_factory=FeatureEmbedder, lease_seconds=30,
    ).run_once() is True
    failed = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    assert failed["status"] == "failed" and failed["error_code"] == "empty_document"
    assert scalar("SELECT count(*) FROM documents") == 0
    assert scalar("SELECT payload IS NULL FROM ingestion_jobs WHERE id = :id", id=queued["job_id"]) is True


def test_explicit_reindex_updates_embedding_identity_transactionally(client):
    queued = enqueue_background(
        client, synthetic_docx("Synthetic reindex evidence with Python."), key="integration-base-job-001",
    )
    IngestionWorker(get_session_factory(), embedder_factory=FeatureEmbedder).run_once()
    completed = client.get(f"/v1/ingestion-jobs/{queued['job_id']}", headers=HEADERS).json()
    document_id = completed["document_id"]

    class UpdatedEmbedder(FeatureEmbedder):
        model_name = "test-feature-embedder-v2"
        model_version = "2.0-test"

    reindex = client.post(
        f"/v1/documents/{document_id}/reindex",
        headers={**HEADERS, "Idempotency-Key": "integration-reindex-job-001"},
        json={"version_number": 1},
    )
    assert reindex.status_code == 202, reindex.text
    IngestionWorker(get_session_factory(), embedder_factory=UpdatedEmbedder).run_once()
    result = client.get(f"/v1/ingestion-jobs/{reindex.json()['job_id']}", headers=HEADERS).json()
    assert result["status"] == "completed" and result["embedding_model"] == "test-feature-embedder-v2"
    assert scalar("SELECT embedding_model_version FROM document_versions") == "2.0-test"


def test_queued_version_does_not_resurrect_a_deleted_document(client):
    original = upload(client, synthetic_docx("Synthetic original version."))
    queued = client.post(
        f"/v1/documents/{original['document_id']}/ingestion-jobs",
        headers={**HEADERS, "Idempotency-Key": "integration-deleted-target-001"},
        files={"file": ("Fictional Replacement.docx", synthetic_docx("Replacement."), DOCX)},
    )
    assert queued.status_code == 202, queued.text
    assert client.delete(f"/v1/documents/{original['document_id']}", headers=HEADERS).status_code == 204

    assert IngestionWorker(
        get_session_factory(), embedder_factory=FeatureEmbedder, lease_seconds=30,
    ).run_once() is True
    failed = client.get(f"/v1/ingestion-jobs/{queued.json()['job_id']}", headers=HEADERS).json()
    assert failed["status"] == "failed" and failed["error_code"] == "document_not_found"
    assert scalar("SELECT count(*) FROM documents") == 0
    assert scalar("SELECT payload IS NULL FROM ingestion_jobs WHERE id = :id", id=queued.json()["job_id"]) is True


def test_ingestion_retrieval_and_grounded_answer_with_fake_provider(client):
    document_id = upload(client, synthetic_docx("Built Python APIs for a fictional retailer."))["document_id"]
    body = client.post("/v1/documents/qa", json={"question": "Python APIs?", "document_ids": [document_id]},
                       headers=HEADERS).json()
    citation = body["claims"][0]["citations"][0]
    assert citation["quote"] in body["sources"][0]["excerpt"]
    assert body["sources"][0]["document_id"] == document_id


def test_workflow_interrupt_restart_resume_and_durable_review(client):
    document_id = upload(client, synthetic_docx("Built Python APIs with PostgreSQL."))["document_id"]
    started = client.post("/v1/workflows/recruiter-brief", headers=HEADERS, json={
        "description": "Python API engineer", "required_skills": ["Python"], "document_ids": [document_id],
    })
    assert started.status_code == 201, started.text
    workflow_id = started.json()["workflows"][0]["workflow_id"]
    assert started.json()["workflows"][0]["status"] == "awaiting_review"
    assert scalar("SELECT count(*) FROM checkpoints WHERE thread_id = :id", id=workflow_id) > 0

    # Simulate a process restart: drop every pooled checkpoint connection and resume from PostgreSQL.
    close_pooled_checkpointers()
    pending = client.get(f"/v1/workflows/{workflow_id}", headers=HEADERS).json()
    assert pending["status"] == "awaiting_review" and pending["brief"]["claims"]
    assert client.get(f"/v1/workflows/{workflow_id}", headers=OTHER).status_code == 404

    reviewed = client.post(f"/v1/workflows/{workflow_id}/review", headers=HEADERS,
                           json={"approved": True, "note": "Evidence matches the source excerpts."})
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["outcome"] == "approved"
    assert client.post(f"/v1/workflows/{workflow_id}/review", headers=HEADERS,
                       json={"approved": False}).status_code == 409
    assert scalar("SELECT count(*) FROM workflow_reviews WHERE workflow_id = :id", id=workflow_id) == 1
    subject = scalar("SELECT subject FROM workflow_reviews WHERE workflow_id = :id", id=workflow_id)
    assert subject.startswith("token-sha256:") and "it-token" not in subject

    history = client.get("/v1/workflows?status=completed", headers=HEADERS).json()["workflows"]
    assert [item["workflow_id"] for item in history] == [workflow_id]


def test_lost_review_row_is_reconciled_from_checkpoint(client):
    document_id = upload(client, synthetic_docx("Built Python APIs with PostgreSQL."))["document_id"]
    workflow_id = client.post("/v1/workflows/recruiter-brief", headers=HEADERS, json={
        "description": "Python API engineer", "document_ids": [document_id],
    }).json()["workflows"][0]["workflow_id"]
    client.post(f"/v1/workflows/{workflow_id}/review", headers=HEADERS, json={"approved": False})

    with get_session_factory()() as session:  # simulate a crash after checkpoint write, before review commit
        session.execute(text("DELETE FROM workflow_reviews WHERE workflow_id = :id"), {"id": workflow_id})
        session.execute(text("UPDATE workflow_runs SET status = 'awaiting_review', outcome = NULL "
                             "WHERE workflow_id = :id"), {"id": workflow_id})
        session.commit()

    view = client.get(f"/v1/workflows/{workflow_id}", headers=HEADERS).json()
    assert view["status"] == "completed" and view["review"]["decision"] == "rejected"
    assert scalar("SELECT count(*) FROM workflow_reviews WHERE workflow_id = :id", id=workflow_id) == 1


def test_investigation_persists_trace_restart_scope_and_human_review(client):
    document_id = upload(client, synthetic_docx(
        "Built Python APIs with PostgreSQL for a fictional logistics platform."
    ))["document_id"]
    idempotency_key = "synthetic-investigation-key-0001"
    started = client.post(
        "/v1/investigations",
        headers={**HEADERS, "Idempotency-Key": idempotency_key},
        json={
            "objective": "Investigate Python API evidence",
            "document_ids": [document_id],
            "requirements": ["Python"],
        },
    )
    assert started.status_code == 201, started.text
    body = started.json()
    investigation_id = body["investigation_id"]
    assert body["status"] == "awaiting_review"
    assert body["scope"][0]["version_number"] == 1
    assert [event["tool_name"] for event in body["events"]] == [
        "search_evidence", "check_requirements", "draft_investigation",
    ]
    assert scalar("SELECT count(*) FROM investigation_runs WHERE id = :id", id=investigation_id) == 1
    assert scalar("SELECT count(*) FROM investigation_events WHERE investigation_id = :id",
                  id=investigation_id) == 3
    assert scalar("SELECT count(*) FROM checkpoints WHERE thread_id = :id", id=investigation_id) > 0

    close_pooled_checkpointers()
    reopened = client.get(f"/v1/investigations/{investigation_id}", headers=HEADERS)
    assert reopened.status_code == 200
    assert reopened.json()["report"]["supported_findings"]
    assert client.get(f"/v1/investigations/{investigation_id}", headers=OTHER).status_code == 404

    duplicate = client.post(
        "/v1/investigations",
        headers={**HEADERS, "Idempotency-Key": idempotency_key},
        json={
            "objective": "Investigate Python API evidence",
            "document_ids": [document_id],
            "requirements": ["Python"],
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["investigation_id"] == investigation_id

    reviewed = client.post(
        f"/v1/investigations/{investigation_id}/review",
        headers=HEADERS,
        json={"approved": True, "comment": "Synthetic citations verified."},
    )
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["status"] == "completed"
    assert reviewed.json()["outcome"] == "approved"
    assert scalar("SELECT count(*) FROM investigation_reviews WHERE investigation_id = :id",
                  id=investigation_id) == 1

    with get_session_factory()() as session:  # simulate checkpoint/transaction crash separation
        session.execute(text("DELETE FROM investigation_reviews WHERE investigation_id = :id"),
                        {"id": investigation_id})
        session.execute(text("UPDATE investigation_runs SET status = 'awaiting_review', outcome = NULL "
                             "WHERE id = :id"), {"id": investigation_id})
        session.commit()

    recovered = client.get(f"/v1/investigations/{investigation_id}", headers=HEADERS)
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "completed"
    assert recovered.json()["review"]["decision"] == "approved"
    assert scalar("SELECT count(*) FROM investigation_reviews WHERE investigation_id = :id",
                  id=investigation_id) == 1
