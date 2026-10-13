"""Integration coverage against a real PostgreSQL + pgvector database.

Run with TEST_DATABASE_URL pointed at an isolated database after `alembic upgrade
head`. The suite is skipped by default to prevent accidental shared DB writes.
"""

from io import BytesIO
import os

import pytest

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if not TEST_DATABASE_URL:
    pytest.skip("TEST_DATABASE_URL is not configured", allow_module_level=True)

from alembic import command
from alembic.config import Config
from docx import Document
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import StoredDocumentChunk
from repositories.documents import DocumentRepository
from retrieval.authorization import AuthorizationContext
from retrieval.pipeline import RetrievalPipeline


class FakeEmbedder:
    model_name = "test-embedder-v1"

    def embed(self, texts):
        vectors = []
        for value in texts:
            vector = [0.0] * 384
            lowered = value.casefold()
            vector[0] = 1.0 if "python" in lowered else 0.0
            vector[1] = 1.0 if "sql" in lowered else 0.0
            vectors.append(vector)
        return vectors


@pytest.fixture(scope="session")
def engine():
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(config, "head")
    engine = create_engine(TEST_DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_database(engine):
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE document_chunks, document_versions, documents CASCADE"))


@pytest.fixture
def session(engine):
    current = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield current
        current.commit()
    finally:
        current.close()


def resume_bytes(text):
    output = BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(output)
    return output.getvalue()


def test_idempotent_ingestion_versions_changed_content(session):
    repository = DocumentRepository(session, embedder=FakeEmbedder())
    first = repository.ingest_bytes("tenant-a", "candidate-1", "one.docx", resume_bytes("Python APIs"))
    second = repository.ingest_bytes("tenant-a", "candidate-1", "one.docx", resume_bytes("Python APIs"))
    third = repository.ingest_bytes("tenant-a", "candidate-1", "one.docx", resume_bytes("Python and SQL APIs"))

    assert first.created is True
    assert second.created is False
    assert second.version_id == first.version_id
    assert third.version_number == 2
    results = RetrievalPipeline(repository, embedder=FakeEmbedder()).retrieve(
        AuthorizationContext("test", frozenset({"tenant-a"})), "tenant-a", "SQL"
    ).candidates
    assert results and all("SQL" in item.content for item in results)


def test_retrieval_is_tenant_scoped(session):
    repository = DocumentRepository(session, embedder=FakeEmbedder())
    own = repository.ingest_bytes("tenant-a", "candidate-1", "one.docx", resume_bytes("Python platform work"))
    repository.ingest_bytes("tenant-b", "candidate-2", "two.docx", resume_bytes("Python confidential work"))

    results = RetrievalPipeline(repository, embedder=FakeEmbedder()).retrieve(
        AuthorizationContext("test", frozenset({"tenant-a"})), "tenant-a", "Python"
    ).candidates

    assert results
    assert {item.document_id for item in results} == {own.document_id}


def test_deletion_removes_versions_chunks_and_retrieval(session):
    repository = DocumentRepository(session, embedder=FakeEmbedder())
    result = repository.ingest_bytes("tenant-a", "candidate-1", "one.docx", resume_bytes("Python platform work"))
    session.flush()

    assert repository.delete_document("tenant-a", result.document_id) is True
    session.flush()

    assert RetrievalPipeline(repository, embedder=FakeEmbedder()).retrieve(
        AuthorizationContext("test", frozenset({"tenant-a"})), "tenant-a", "Python"
    ).candidates == ()
    assert session.query(StoredDocumentChunk).filter_by(document_id=result.document_id).count() == 0
