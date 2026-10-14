"""Resume Review Assistant — explainable, human-in-the-loop recruitment document intelligence.

Standalone mode runs the original in-memory deterministic ranking. V2 Connected
mode (``API_BASE_URL`` set) adds persistent documents, hybrid search, grounded
Q&A, and durable recruiter-brief review through the FastAPI service.
"""

import streamlit as st

from frontend.components import inject_css, mode_badge
from frontend.context import invalidate_documents, resolve_context
from frontend.views import ask, briefs, history, library, overview, ranking, search
from services.ranking import analyze_resume_bytes, build_export_frame, excel_bytes  # noqa: F401  (compat exports)
from utils.matcher import MODEL_NAME


st.set_page_config(page_title="Resume Review Assistant", page_icon="📄", layout="wide")
inject_css()

PAGES = {
    "Overview": "🏠",
    "Resume Library": "📁",
    "Candidate Ranking": "📊",
    "AI Candidate Search": "🔎",
    "Ask AI": "💬",
    "Recruiter Briefs": "📝",
    "Review History": "🗂️",
}

ctx = resolve_context()

with st.sidebar:
    st.title("📄 Resume Review Assistant")
    st.caption("Explainable candidate screening with human-owned decisions")
    mode_badge(ctx)
    page = st.radio("Navigate", list(PAGES), key="nav", format_func=lambda name: f"{PAGES[name]}  {name}")
    st.divider()
    if ctx.mode != "standalone" and st.button("Reconnect / refresh status", width="stretch"):
        invalidate_documents()
        ctx = resolve_context(force=True)
        st.rerun()
    st.caption(
        "Scores and AI output are decision-support signals for human reviewers, not hiring decisions, and must "
        "never be the sole basis for an employment action."
    )

if page == "Overview":
    overview.render(ctx, list(PAGES))
elif page == "Resume Library":
    library.render(ctx)
elif page == "Candidate Ranking":
    ranking.render(ctx)
elif page == "AI Candidate Search":
    search.render(ctx)
elif page == "Ask AI":
    ask.render(ctx)
elif page == "Recruiter Briefs":
    briefs.render(ctx)
elif page == "Review History":
    history.render(ctx)

st.divider()
st.caption(f"Model: {MODEL_NAME} · Local decision-support prototype · No automated hiring decisions")
