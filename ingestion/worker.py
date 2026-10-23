"""Bounded single-process worker for durable PostgreSQL ingestion jobs."""

import json
import logging
import os
import signal
import time
from uuid import uuid4

from db.session import get_session_factory
from repositories.documents import (
    DocumentQualityError,
    DocumentRepository,
    DocumentTooLargeError,
)
from repositories.ingestion_jobs import IngestionJobRepository
from services.embeddings import SentenceTransformerEmbedder
from utils.pdf_parser import ResumeParseError


LOGGER = logging.getLogger("ingestion.worker")
MAX_INGEST_CHUNKS = 500


def _positive_int(name, default, *, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(1, min(value, maximum))


class IngestionWorker:
    def __init__(self, session_factory=None, *, embedder_factory=SentenceTransformerEmbedder,
                 lease_seconds=None, base_backoff_seconds=None, clock=time.perf_counter):
        self.session_factory = session_factory or get_session_factory()
        self.embedder_factory = embedder_factory
        self.lease_seconds = lease_seconds or _positive_int("INGESTION_LEASE_SECONDS", 600, maximum=3_600)
        self.base_backoff_seconds = base_backoff_seconds or _positive_int(
            "INGESTION_RETRY_BASE_SECONDS", 5, maximum=300,
        )
        self.clock = clock
        self.worker_id = "worker-" + uuid4().hex

    def run_once(self) -> bool:
        with self.session_factory.begin() as session:
            job = IngestionJobRepository(session).claim_next(
                self.worker_id, lease_seconds=self.lease_seconds,
            )
            job_id = job.id if job is not None else None
        if job_id is None:
            return False
        self._process(job_id)
        return True

    def _process(self, job_id):
        started = self.clock()
        try:
            with self.session_factory.begin() as session:
                jobs = IngestionJobRepository(session)
                job = jobs.lock_claim(job_id, self.worker_id)
                if job is None:
                    return
                documents = DocumentRepository(session, embedder=self.embedder_factory())
                if job.job_type == "ingest":
                    if not job.payload or not job.filename or not job.source_key:
                        raise DocumentQualityError("invalid_metadata", "The ingestion job metadata is invalid.")
                    result = documents.ingest_bytes(
                        job.tenant_id, job.source_key, job.filename, bytes(job.payload),
                        max_chunks=MAX_INGEST_CHUNKS, target_document_id=job.target_document_id,
                    )
                elif job.job_type == "reindex":
                    if job.target_document_id is None or job.target_version_number is None:
                        raise DocumentQualityError("invalid_metadata", "The reindex job metadata is invalid.")
                    result = documents.reindex_version(
                        job.tenant_id, job.target_document_id, job.target_version_number,
                    )
                else:
                    raise DocumentQualityError("invalid_metadata", "The ingestion job type is invalid.")
                jobs.complete(job, result, latency_ms=(self.clock() - started) * 1_000)
            self._event("completed", None, (self.clock() - started) * 1_000)
        except (ResumeParseError, DocumentTooLargeError, DocumentQualityError, LookupError) as exc:
            code = getattr(exc, "code", None) or (
                "chunk_limit" if isinstance(exc, DocumentTooLargeError)
                else "document_not_found" if isinstance(exc, LookupError)
                else "extraction_failed"
            )
            self._fail(job_id, code, retryable=False, started=started)
        except Exception:
            # Provider/database exception text can contain confidential data and is never logged or persisted.
            self._fail(job_id, "processing_failed", retryable=True, started=started)

    def _fail(self, job_id, code, *, retryable, started):
        status = "ignored"
        with self.session_factory.begin() as session:
            jobs = IngestionJobRepository(session)
            job = jobs.lock_claim(job_id, self.worker_id)
            if job is not None:
                jobs.record_failure(
                    job, error_code=code, retryable=retryable,
                    base_backoff_seconds=self.base_backoff_seconds,
                    latency_ms=(self.clock() - started) * 1_000,
                )
                status = job.status
        self._event(status, code, (self.clock() - started) * 1_000)

    @staticmethod
    def _event(status, error_code, latency_ms):
        LOGGER.info(json.dumps({
            "event": "ingestion_job_finished" if status in {"completed", "failed"} else "ingestion_job_retry",
            "status": status,
            "error_category": error_code,
            "duration_ms": round(max(0.0, latency_ms), 3),
        }, separators=(",", ":")))


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
    poll_seconds = _positive_int("INGESTION_POLL_SECONDS", 2, maximum=60)
    worker = IngestionWorker()
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGGER.info('{"event":"ingestion_worker_started"}')
    while not stopping:
        if not worker.run_once():
            time.sleep(poll_seconds)
    LOGGER.info('{"event":"ingestion_worker_stopped"}')


if __name__ == "__main__":
    main()
