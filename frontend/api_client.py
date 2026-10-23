"""Typed HTTP client used by Streamlit in V2 Connected mode.

Credentials come only from server-side configuration (environment variables or
Streamlit secrets); the browser never sees or edits them, and the tenant header
is not authority by itself — FastAPI authorizes every request. Error messages are
fixed, user-safe strings that never include headers, tokens, or response bodies
beyond the API's own fixed ``detail`` strings.
"""

from dataclasses import dataclass
import logging
import os
import time
from typing import TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from api import schemas


# httpx logs request URLs at INFO; keep request metadata out of application logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 30.0
LONG_TIMEOUT = 180.0
INVESTIGATION_TIMEOUT = 210.0  # server budget is 180 s; allow transport and persistence overhead
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.5
RETRYABLE_STATUSES = {502, 503, 504}

Model = TypeVar("Model", bound=BaseModel)


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    tenant_id: str
    token: str

    def __repr__(self):  # never render the token in tracebacks or Streamlit exception views
        return f"ApiConfig(base_url={self.base_url!r}, tenant_id={self.tenant_id!r}, token=<redacted>)"


class ApiError(Exception):
    """A user-presentable API failure."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


def _secret(name, secrets=None):
    value = os.getenv(name, "")
    if not value and secrets is not None:
        try:
            value = secrets.get(name, "")
        except Exception:
            value = ""
    return str(value or "").strip()


def config_from_env(secrets=None) -> ApiConfig | None:
    """Return connection settings, or ``None`` for Standalone mode."""
    base_url = _secret("API_BASE_URL", secrets)
    if not base_url:
        return None
    return ApiConfig(base_url.rstrip("/"), _secret("API_TENANT_ID", secrets), _secret("API_BEARER_TOKEN", secrets))


_STATUS_MESSAGES = {
    401: ("unauthenticated", "The V2 API rejected the server-side credentials. Check API_BEARER_TOKEN."),
    403: ("forbidden", "The configured credentials are not authorized for this operation or workspace."),
    404: ("not_found", "The requested item was not found in this workspace."),
    409: ("conflict", "The request conflicts with the item's current state."),
    413: ("too_large", "The file exceeds the 10 MB size limit."),
    422: ("invalid", "The request could not be processed."),
    502: ("upstream", "The generation provider did not return a usable response."),
    503: ("unavailable", "The V2 service is temporarily unavailable."),
}


def _safe_detail(response: httpx.Response) -> str | None:
    """The API returns fixed string details; ignore structured validation payloads."""
    try:
        detail = response.json().get("detail")
    except Exception:
        return None
    return detail if isinstance(detail, str) and len(detail) <= 300 else None


class ApiClient:
    def __init__(self, config: ApiConfig, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.config = config
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            transport=transport,
            headers={"X-Tenant-ID": config.tenant_id, "Authorization": f"Bearer {config.token}"},
        )

    @property
    def configured(self) -> bool:
        return bool(self.config.tenant_id and self.config.token)

    def close(self):
        self._client.close()

    def _request(self, method, path, *, retry=False, timeout=None, **kwargs) -> httpx.Response:
        if not self.configured and path != "/v1/health":
            raise ApiError("unauthenticated", "V2 credentials are not configured on the Streamlit server.")
        attempts = 1 + (RETRY_ATTEMPTS if retry else 0)
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(method, path, timeout=timeout or READ_TIMEOUT, **kwargs)
            except httpx.TimeoutException:
                if attempt < attempts:
                    self._sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise ApiError("timeout", f"The V2 API did not respond within {timeout or READ_TIMEOUT:.0f} seconds.") from None
            except httpx.TransportError:
                if attempt < attempts:
                    self._sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise ApiError("unreachable", "The V2 API is unreachable. Check that the backend is running.") from None
            if response.status_code in RETRYABLE_STATUSES and attempt < attempts:
                self._sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            if response.is_success:
                return response
            kind, message = _STATUS_MESSAGES.get(
                response.status_code, ("error", "The V2 API returned an unexpected error.")
            )
            raise ApiError(kind, _safe_detail(response) or message, response.status_code)
        raise ApiError("error", "The V2 API returned an unexpected error.")  # pragma: no cover

    @staticmethod
    def _parse(response: httpx.Response, model: type[Model]) -> Model:
        try:
            return model.model_validate(response.json())
        except (ValidationError, ValueError):
            raise ApiError("contract", "The V2 API response did not match the expected contract.") from None

    # Read-only, idempotent operations may be retried.
    def health(self) -> schemas.HealthResponse:
        return self._parse(self._request("GET", "/v1/health", retry=True, timeout=5), schemas.HealthResponse)

    def session(self) -> schemas.SessionResponse:
        return self._parse(self._request("GET", "/v1/session", retry=True), schemas.SessionResponse)

    def list_documents(self, limit: int = 200) -> schemas.DocumentListResponse:
        response = self._request("GET", "/v1/documents", params={"limit": limit}, retry=True)
        return self._parse(response, schemas.DocumentListResponse)

    def get_document(self, document_id: UUID, version: int | None = None) -> schemas.DocumentDetail:
        params = {"version": version} if version else None
        return self._parse(
            self._request("GET", f"/v1/documents/{document_id}", params=params, retry=True), schemas.DocumentDetail
        )

    def search(self, query: str, *, document_ids=(), limit: int = 10, rerank: bool = False) -> list[schemas.SearchResult]:
        # POST keeps confidential queries out of URLs and access logs; it is still a safe, idempotent read.
        request = schemas.SearchRequest(query=query, document_ids=list(document_ids), limit=limit, rerank=rerank)
        response = self._request(
            "POST", "/v1/documents/search", json=request.model_dump(mode="json"), retry=True,
            timeout=LONG_TIMEOUT if rerank else READ_TIMEOUT,
        )
        try:
            return TypeAdapter(list[schemas.SearchResult]).validate_python(response.json())
        except (ValidationError, ValueError):
            raise ApiError("contract", "The V2 API response did not match the expected contract.") from None

    def list_workflows(self, status: str | None = None) -> schemas.WorkflowListResponse:
        params = {"status": status} if status else None
        return self._parse(self._request("GET", "/v1/workflows", params=params, retry=True), schemas.WorkflowListResponse)

    def get_workflow(self, workflow_id: UUID) -> schemas.WorkflowView:
        return self._parse(self._request("GET", f"/v1/workflows/{workflow_id}", retry=True), schemas.WorkflowView)

    def list_investigations(self, status: str | None = None) -> schemas.InvestigationListResponse:
        params = {"status": status} if status else None
        return self._parse(
            self._request("GET", "/v1/investigations", params=params, retry=True),
            schemas.InvestigationListResponse,
        )

    def get_investigation(self, investigation_id: UUID) -> schemas.InvestigationView:
        return self._parse(
            self._request("GET", f"/v1/investigations/{investigation_id}", retry=True),
            schemas.InvestigationView,
        )

    # Mutating or provider-billed operations are never retried automatically.
    def upload_document(self, filename: str, data: bytes) -> schemas.IngestResponse:
        response = self._request("POST", "/v1/documents", files={"file": (_opaque_upload_name(filename), data)},
                                 timeout=LONG_TIMEOUT)
        return self._parse(response, schemas.IngestResponse)

    def upload_version(self, document_id: UUID, filename: str, data: bytes) -> schemas.IngestResponse:
        response = self._request("POST", f"/v1/documents/{document_id}/versions",
                                 files={"file": (_opaque_upload_name(filename), data)}, timeout=LONG_TIMEOUT)
        return self._parse(response, schemas.IngestResponse)

    def enqueue_document(self, filename: str, data: bytes, *, idempotency_key: str) -> schemas.IngestionJobView:
        response = self._request(
            "POST", "/v1/ingestion-jobs",
            headers={"Idempotency-Key": idempotency_key},
            files={"file": (_opaque_upload_name(filename), data)},
            timeout=LONG_TIMEOUT,
        )
        return self._parse(response, schemas.IngestionJobView)

    def enqueue_document_version(self, document_id: UUID, filename: str, data: bytes, *,
                                 idempotency_key: str) -> schemas.IngestionJobView:
        response = self._request(
            "POST", f"/v1/documents/{document_id}/ingestion-jobs",
            headers={"Idempotency-Key": idempotency_key},
            files={"file": (_opaque_upload_name(filename), data)},
            timeout=LONG_TIMEOUT,
        )
        return self._parse(response, schemas.IngestionJobView)

    def enqueue_reindex(self, document_id: UUID, *, version_number: int | None,
                        idempotency_key: str) -> schemas.IngestionJobView:
        request = schemas.ReindexRequest(version_number=version_number)
        response = self._request(
            "POST", f"/v1/documents/{document_id}/reindex",
            headers={"Idempotency-Key": idempotency_key}, json=request.model_dump(),
        )
        return self._parse(response, schemas.IngestionJobView)

    def list_ingestion_jobs(self, status: str | None = None, limit: int = 100) -> schemas.IngestionJobListResponse:
        params = {"limit": limit}
        if status:
            params["status"] = status
        return self._parse(
            self._request("GET", "/v1/ingestion-jobs", params=params, retry=True),
            schemas.IngestionJobListResponse,
        )

    def get_ingestion_job(self, job_id: UUID) -> schemas.IngestionJobView:
        return self._parse(
            self._request("GET", f"/v1/ingestion-jobs/{job_id}", retry=True),
            schemas.IngestionJobView,
        )

    def ingestion_metrics(self) -> schemas.IngestionMetricsResponse:
        return self._parse(
            self._request("GET", "/v1/ingestion-jobs/metrics", retry=True),
            schemas.IngestionMetricsResponse,
        )

    def delete_document(self, document_id: UUID) -> None:
        self._request("DELETE", f"/v1/documents/{document_id}")

    def ask(self, question: str, *, document_ids=()) -> schemas.GroundedAnswerResponse:
        request = schemas.QuestionRequest(question=question, document_ids=list(document_ids))
        response = self._request("POST", "/v1/documents/qa", json=request.model_dump(mode="json"), timeout=LONG_TIMEOUT)
        return self._parse(response, schemas.GroundedAnswerResponse)

    def start_brief_workflows(self, *, description: str, required_skills, preferred_skills,
                              document_ids) -> schemas.WorkflowStartResponse:
        request = schemas.WorkflowStartRequest(
            description=description, required_skills=list(required_skills),
            preferred_skills=list(preferred_skills), document_ids=list(document_ids),
        )
        response = self._request("POST", "/v1/workflows/recruiter-brief", json=request.model_dump(mode="json"),
                                 timeout=LONG_TIMEOUT)
        return self._parse(response, schemas.WorkflowStartResponse)

    def submit_review(self, workflow_id: UUID, *, approved: bool, note: str = "") -> schemas.WorkflowView:
        request = schemas.ReviewSubmission(approved=approved, note=note)
        response = self._request("POST", f"/v1/workflows/{workflow_id}/review", json=request.model_dump())
        return self._parse(response, schemas.WorkflowView)

    def start_investigation(self, *, objective: str, document_ids, idempotency_key: str,
                            job_description: str = "", requirements=()) -> schemas.InvestigationView:
        request = schemas.InvestigationCreateRequest(
            objective=objective,
            document_ids=list(document_ids),
            job_description=job_description,
            requirements=list(requirements),
        )
        response = self._request(
            "POST", "/v1/investigations",
            headers={"Idempotency-Key": idempotency_key},
            json=request.model_dump(mode="json"),
            timeout=INVESTIGATION_TIMEOUT,
        )
        return self._parse(response, schemas.InvestigationView)

    def review_investigation(self, investigation_id: UUID, *, approved: bool,
                             comment: str = "") -> schemas.InvestigationView:
        request = schemas.InvestigationReviewSubmission(approved=approved, comment=comment)
        response = self._request(
            "POST", f"/v1/investigations/{investigation_id}/review",
            json=request.model_dump(),
        )
        return self._parse(response, schemas.InvestigationView)


def _opaque_upload_name(filename: str) -> str:
    """Send only the extension the parser needs; identity-bearing filenames never leave Streamlit."""
    suffix = os.path.splitext(filename or "")[1].lower()
    return "document" + suffix
