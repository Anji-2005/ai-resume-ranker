"""Reusable Streamlit rendering helpers for evidence, citations, and status."""

import html

import streamlit as st

from frontend.context import CONNECTED, MODE_LABELS, STANDALONE


AI_NOTICE = (
    "**AI-generated assistance requiring human review.** Every claim below was checked to cite an exact "
    "excerpt of the evidence supplied to the model, but citation checks do not prove the claim is complete "
    "or correct. This is not a hiring recommendation."
)
RELEVANCE_NOTICE = (
    "Search relevance shows how closely a passage matches your query text. It is not a candidate score, "
    "a ranking, or a measure of competence. Deterministic scoring lives in **Candidate Ranking**."
)
GENERATION_REASONS = {
    "unconfigured": "Grounded generation is disabled: the API server has no Gemini API key configured.",
    "policy_denied": (
        "Grounded generation is disabled for this workspace: no generation policy (provider, purpose, "
        "retention) is configured for the tenant in GENERATION_TENANT_POLICY_JSON."
    ),
}
STATUS_BADGES = {
    "supported": ":green-badge[Supported]",
    "unsupported": ":red-badge[No evidence]",
    "unclear": ":orange-badge[Unclear]",
    "awaiting_review": ":orange-badge[Awaiting review]",
    "completed": ":green-badge[Reviewed]",
    "failed": ":red-badge[Failed]",
    "running": ":blue-badge[Running]",
    "approved": ":green-badge[Notes approved]",
    "rejected": ":gray-badge[Notes rejected]",
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.rr-excerpt {background: rgba(37,99,235,0.06); border-left: 3px solid #2563EB; padding: .6rem .8rem;
             border-radius: 4px; white-space: pre-wrap; font-size: .92rem; line-height: 1.45;}
.rr-excerpt mark {background: #FDE68A; color: #172033; padding: 0 .1rem; border-radius: 2px;}
</style>
"""


_MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|<>~$:")


def md(text) -> str:
    """Escape model- or document-derived text so it renders literally.

    Prevents generated Markdown (links, images, badges, HTML) from being rendered,
    which could otherwise exfiltrate data or impersonate UI elements.
    """
    return "".join("\\" + char if char in _MARKDOWN_SPECIAL else char for char in str(text or ""))


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def mode_badge(ctx):
    label = MODE_LABELS[ctx.mode]
    if ctx.mode == CONNECTED:
        st.success(f"● {label}", icon=None)
    elif ctx.mode == STANDALONE:
        st.info(f"○ {label} · in-memory ranking only")
    else:
        st.error(f"● {label}")


def require_connected(ctx, feature: str) -> bool:
    """Render an honest explanation when a V2-only feature cannot be used."""
    if ctx.connected:
        return True
    if ctx.mode == STANDALONE:
        st.info(
            f"**{feature}** requires V2 Connected mode. Set `API_BASE_URL`, `API_TENANT_ID`, and "
            "`API_BEARER_TOKEN` on the Streamlit server (see README) or run `docker compose up --build`. "
            "Candidate Ranking works without it."
        )
    else:
        st.error(f"**{feature}** is unavailable: {ctx.error or 'the V2 API could not be reached.'}")
    return False


def require_generation(ctx) -> bool:
    if ctx.generation == "enabled":
        return True
    st.warning(GENERATION_REASONS.get(ctx.generation, "Grounded generation is unavailable."))
    st.caption("Search and deterministic ranking remain available. There is no deterministic substitute for "
               "generated answers in V2, so nothing is generated.")
    return False


def show_api_error(error, action: str):
    st.error(f"{action} failed: {error.message}")


def highlighted_excerpt(excerpt: str, quote: str | None = None) -> str:
    """HTML-escape the excerpt and highlight the exact quoted span if present."""
    if quote and quote in excerpt:
        start = excerpt.index(quote)
        body = (html.escape(excerpt[:start]) + "<mark>" + html.escape(quote) + "</mark>"
                + html.escape(excerpt[start + len(quote):]))
    else:
        body = html.escape(excerpt)
    return f'<div class="rr-excerpt">{body}</div>'


def provenance(source) -> str:
    page = f"page {source.page_number}" if source.page_number else "no page (DOCX)"
    truncated = " · excerpt truncated to context budget" if getattr(source, "truncated", False) else ""
    return (f"{source.candidate_ref} · {page} · section {source.section_id} · "
            f"chars {source.source_start}–{source.source_end}{truncated}")


def source_index(sources) -> dict:
    return {source.source_chunk_id: (number, source) for number, source in enumerate(sources, start=1)}


def citation_markers(citations, index) -> str:
    return " ".join(f"[{index[c.source_chunk_id][0]}]" if c.source_chunk_id in index else "[?]" for c in citations)


def render_citations(citations, index, key: str):
    for position, citation in enumerate(citations):
        entry = index.get(citation.source_chunk_id)
        if entry is None:
            st.error("A citation referenced an unknown source and was not displayed.")
            continue
        number, source = entry
        with st.expander(f"[{number}] “{md(citation.quote[:90])}” — {source.candidate_ref}", expanded=False):
            st.markdown(highlighted_excerpt(source.excerpt, citation.quote), unsafe_allow_html=True)
            st.caption(provenance(source))


def render_claims(claims, index, key: str):
    for number, claim in enumerate(claims, start=1):
        st.markdown(f"**{number}.** {md(claim.text)} {citation_markers(claim.citations, index)}")
        render_citations(claim.citations, index, key=f"{key}-claim-{number}")


def render_sources(sources, key: str):
    with st.expander(f"Evidence supplied to the model ({len(sources)} excerpts)"):
        for number, source in enumerate(sources, start=1):
            st.markdown(f"**[{number}]** {provenance(source)}")
            st.markdown(highlighted_excerpt(source.excerpt), unsafe_allow_html=True)


def render_answer(answer, key: str):
    index = source_index(answer.sources)
    if answer.claims:
        render_claims(answer.claims, index, key)
    if answer.uncertainty:
        st.warning(f"**Uncertainty:** {md(answer.uncertainty)}")
    if not answer.claims and not answer.sources:
        st.info("No supporting evidence was found in the selected documents, so no answer was generated.")
    if answer.generation:
        st.caption(f"Model {answer.generation.model} · prompt {answer.generation.prompt_version} · "
                   "citations verified against supplied excerpts")
    if answer.sources:
        render_sources(answer.sources, key)


def render_brief(brief, sources, key: str):
    index = source_index(sources)
    st.markdown("##### Evidence-backed summary")
    render_claims(brief.claims, index, key)
    st.markdown("##### Requirement-by-requirement evidence")
    for number, item in enumerate(brief.requirement_evidence, start=1):
        st.markdown(f"{STATUS_BADGES.get(item.status, item.status)} **{md(item.requirement)}** — {md(item.explanation)} "
                    f"{citation_markers(item.citations, index) if item.citations else ''}")
        if item.citations:
            render_citations(item.citations, index, key=f"{key}-req-{number}")
    st.markdown("##### Interview questions grounded in the resume")
    for number, question in enumerate(brief.interview_questions, start=1):
        st.markdown(f"**Q{number}.** {md(question.question)} {citation_markers(question.citations, index)}")
        st.caption(f"Why: {md(question.reason)}")
        render_citations(question.citations, index, key=f"{key}-q-{number}")
    if sources:
        render_sources(sources, key)


def empty_state(message: str, hint: str | None = None):
    st.info(message)
    if hint:
        st.caption(hint)
