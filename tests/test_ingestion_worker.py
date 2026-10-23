"""Fast unit coverage for ingestion validation and retry policy."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from repositories.documents import DocumentQualityError, _validated_vectors
from repositories.ingestion_jobs import IngestionJobRepository


def test_embedding_validation_rejects_count_dimension_and_non_finite_values():
    with pytest.raises(DocumentQualityError, match="unexpected vector count") as count_error:
        _validated_vectors([], 1)
    assert count_error.value.code == "embedding_invalid"

    with pytest.raises(DocumentQualityError, match="incompatible vector dimension") as dimension_error:
        _validated_vectors([[0.0] * 383], 1)
    assert dimension_error.value.code == "embedding_invalid"

    with pytest.raises(DocumentQualityError, match="non-finite vector") as finite_error:
        _validated_vectors([[float("nan")] + [0.0] * 383], 1)
    assert finite_error.value.code == "embedding_invalid"


def test_retry_backoff_is_exponential_and_bounded_by_attempts():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    job = SimpleNamespace(
        attempt_count=2, max_attempts=3, payload=b"confidential", completed_at=None,
        processing_latency_ms=None, lease_expires_at=now, worker_id="worker", error_code=None,
        updated_at=None, status="processing", available_at=now,
    )
    IngestionJobRepository.record_failure(
        job, error_code="processing_failed", retryable=True,
        base_backoff_seconds=5, now=now, latency_ms=12,
    )
    assert job.status == "queued"
    assert (job.available_at - now).total_seconds() == 10
    assert job.payload == b"confidential"

    job.attempt_count = 9
    job.max_attempts = 10
    IngestionJobRepository.record_failure(
        job, error_code="processing_failed", retryable=True,
        base_backoff_seconds=300, now=now, latency_ms=13,
    )
    assert (job.available_at - now).total_seconds() == 3_600

    job.attempt_count = 10
    IngestionJobRepository.record_failure(
        job, error_code="processing_failed", retryable=True,
        base_backoff_seconds=5, now=now, latency_ms=14,
    )
    assert job.status == "failed"
    assert job.payload is None
    assert job.processing_latency_ms == 14
