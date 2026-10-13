"""Synthetic HTTP smoke test for the running Compose API.

This deliberately uses a fictional document and never calls a live LLM.
"""

from io import BytesIO
import os
from uuid import uuid4

from docx import Document
import httpx


def synthetic_docx():
    output = BytesIO()
    document = Document()
    document.add_heading("Experience", level=1)
    document.add_paragraph("Built Python FastAPI services with PostgreSQL and Docker for a fictional project.")
    document.save(output)
    return output.getvalue()


def main():
    base_url = os.getenv("SMOKE_API_URL", "http://localhost:8000")
    tenant = os.getenv("SMOKE_TENANT_ID", "synthetic-tenant")
    token = os.getenv("SMOKE_BEARER_TOKEN", "local-validation-token")
    headers = {"X-Tenant-ID": tenant, "Authorization": f"Bearer {token}"}
    source_key = f"synthetic-smoke-{uuid4()}"

    with httpx.Client(base_url=base_url, timeout=120) as client:
        unauthorized = client.get("/v1/documents/search", params={"query": "Python"}, headers={"X-Tenant-ID": tenant})
        assert unauthorized.status_code == 401, unauthorized.text

        forbidden = client.get(
            "/v1/documents/search",
            params={"query": "Python"},
            headers={"X-Tenant-ID": "other-tenant", "Authorization": f"Bearer {token}"},
        )
        assert forbidden.status_code == 403, forbidden.text

        ingest = client.post(
            "/v1/documents",
            data={"source_key": source_key},
            files={"file": ("fictional-resume.docx", synthetic_docx(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers=headers,
        )
        assert ingest.status_code == 201, ingest.text
        document_id = ingest.json()["document_id"]

        search = client.get("/v1/documents/search", params={"query": "Python FastAPI"}, headers=headers)
        assert search.status_code == 200, search.text
        assert any(item["document_id"] == document_id for item in search.json()), search.text

        no_llm = client.post(
            "/v1/documents/qa",
            json={"question": "What was built?", "source_key": [source_key]},
            headers=headers,
        )
        assert no_llm.status_code == 422, no_llm.text

        unauthorized_delete = client.delete(
            f"/v1/documents/{document_id}", headers={"X-Tenant-ID": tenant}
        )
        assert unauthorized_delete.status_code == 401, unauthorized_delete.text

        deleted = client.delete(f"/v1/documents/{document_id}", headers=headers)
        assert deleted.status_code == 204, deleted.text

        after_delete = client.get("/v1/documents/search", params={"query": "Python FastAPI"}, headers=headers)
        assert after_delete.status_code == 200, after_delete.text
        assert all(item["document_id"] != document_id for item in after_delete.json())

    print("API smoke passed: auth, tenant denial, ingestion, retrieval, no-LLM fallback, deletion")


if __name__ == "__main__":
    main()
