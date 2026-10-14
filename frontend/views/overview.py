"""Overview: mode, pipeline status, and entry points to each capability."""

import streamlit as st

from frontend.api_client import ApiError
from frontend.context import CONNECTED, STANDALONE, load_documents

GENERATION_STATUS = {
    "enabled": ("Enabled", "Gemini via LangChain structured output, gated by tenant policy"),
    "unconfigured": ("Not configured", "No GEMINI_API_KEY on the API server"),
    "policy_denied": ("Disabled by policy", "No generation policy for this tenant"),
}


def _go(page: str):
    st.session_state["nav"] = page


def _status_row(label: str, ok: bool | None, detail: str):
    icon = "🟢" if ok else ("⚪" if ok is None else "🔴")
    st.markdown(f"{icon} **{label}** — {detail}")


def render(ctx, pages):
    st.header("Recruitment Document Intelligence")
    st.markdown(
        "A human-in-the-loop workspace for reviewing resumes against explicit job criteria. Deterministic, "
        "explainable scoring prioritizes candidates for review; retrieval and grounded generation help reviewers "
        "find and verify evidence. **No part of this application makes hiring decisions.**"
    )
    documents = []
    if ctx.mode == CONNECTED:
        try:
            documents = load_documents(ctx)
        except ApiError as exc:
            st.warning(f"Could not load the library: {exc.message}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Operating mode", {"connected": "V2 Connected", "standalone": "Standalone"}.get(ctx.mode, "V2 unavailable"))
    m2.metric("Indexed documents", len(documents) if ctx.mode == CONNECTED else "—")
    m3.metric("Grounded generation", GENERATION_STATUS.get(ctx.generation, ("—",))[0] if ctx.connected else "—")

    st.subheader("Pipeline status")
    with st.container(border=True):
        _status_row("Deterministic ranking", True, "Local MiniLM similarity + skills + experience (always available)")
        if ctx.mode == STANDALONE:
            _status_row("V2 API", None, "Not configured — Standalone mode keeps ranking fully functional")
        else:
            health = ctx.health
            _status_row("V2 API", ctx.health is not None, ctx.error or "Reachable and authorized")
            _status_row("PostgreSQL + pgvector", bool(health and health.database == "ok"),
                        "Documents, chunks, embeddings, workflow checkpoints")
            _status_row("Embeddings", bool(health), health.embedding_model if health else "Unavailable")
            _status_row("Cross-encoder reranker", bool(health),
                        f"{health.reranker_model} (optional, loads on first use)" if health else "Unavailable")
            if ctx.session:
                label, detail = GENERATION_STATUS[ctx.session.generation]
                _status_row("Grounded generation", ctx.session.generation == "enabled", f"{label}: {detail}")
                st.caption(f"Workspace `{ctx.session.tenant_id}` · roles: {', '.join(ctx.session.roles)}")

    st.subheader("How it works")
    st.markdown(
        "1. **Upload & manage resumes** — files are validated, parsed with provenance, chunked, embedded, and "
        "versioned in PostgreSQL. Filenames are never stored.\n"
        "2. **Rank candidates** — the original deterministic score: semantic similarity, required/preferred skill "
        "coverage, and stated experience, with evidence and human review.\n"
        "3. **Search** — hybrid dense + full-text retrieval fused with reciprocal-rank fusion.\n"
        "4. **Ask AI / Recruiter Briefs** — answers and briefs are accepted only when every citation is an exact "
        "excerpt of the evidence supplied to the model.\n"
        "5. **Review** — a person approves or rejects AI-generated notes; the decision is persisted."
    )
    st.subheader("Go to")
    targets = [page for page in pages if page != "Overview"]
    for row_start in range(0, len(targets), 3):
        columns = st.columns(3)
        for column, page in zip(columns, targets[row_start:row_start + 3]):
            column.button(page, key=f"quick_{page}", on_click=_go, args=(page,), width="stretch")
