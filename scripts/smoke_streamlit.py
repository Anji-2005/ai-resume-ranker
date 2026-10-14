"""Streamlit render smoke test against a live V2 API (synthetic data only, no LLM calls).

Runs the real ``app.py`` headlessly with Streamlit's AppTest, connected to the
API at ``API_BASE_URL`` using ``API_TENANT_ID``/``API_BEARER_TOKEN``. It seeds one
fictional document through the client, renders every page, performs a search,
and deletes the document again.
"""

from io import BytesIO
from pathlib import Path
import sys

from docx import Document
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.api_client import ApiClient, config_from_env  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Overview", "Resume Library", "Candidate Ranking", "AI Candidate Search", "Ask AI",
         "Recruiter Briefs", "Review History"]


def synthetic_docx():
    output = BytesIO()
    document = Document()
    document.add_paragraph("Built Python data services with PostgreSQL for a fictional streaming company.")
    document.save(output)
    return output.getvalue()


def main():
    config = config_from_env()
    assert config is not None, "API_BASE_URL must be set for the connected Streamlit smoke test"
    client = ApiClient(config)
    seeded = client.upload_document("synthetic.docx", synthetic_docx())
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=60).run()
        assert not at.exception, at.exception
        badge = " ".join(str(element.value) for element in at.sidebar.success)
        assert "V2 Connected mode" in badge, f"expected connected mode, sidebar shows: {badge or at.sidebar.error}"
        for page in PAGES:
            at.sidebar.radio(key="nav").set_value(page).run()
            assert not at.exception, (page, at.exception)
        at.sidebar.radio(key="nav").set_value("AI Candidate Search").run()
        at.text_input(key="search_query").input("Python PostgreSQL").run()
        next(button for button in at.button if button.label == "Search").click().run()
        assert not at.exception, at.exception
        rendered = " ".join(str(element.value) for element in at.markdown)
        assert seeded.candidate_ref in rendered, "search results did not render the seeded document"
    finally:
        client.delete_document(seeded.document_id)
    print(f"Streamlit smoke passed: connected mode, {len(PAGES)} pages rendered, live hybrid search rendered "
          f"{seeded.candidate_ref}, seeded document deleted")


if __name__ == "__main__":
    main()
