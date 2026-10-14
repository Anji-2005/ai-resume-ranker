"""AI Candidate Search: hybrid pgvector + PostgreSQL full-text retrieval with RRF and optional reranking."""

import time
from uuid import UUID

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import RELEVANCE_NOTICE, empty_state, highlighted_excerpt, md, require_connected, show_api_error
from frontend.context import document_labels, load_documents
from utils.skills_extractor import parse_skill_list

MAX_REQUIREMENTS = 8


def _document_filter(documents, key: str):
    labels = document_labels(documents)
    selected = st.multiselect(
        "Limit to documents (leave empty for all accessible resumes)", list(labels),
        format_func=labels.get, key=key,
    )
    return [UUID(value) for value in selected]


def _result_card(rank: int, item, reranked: bool):
    with st.container(border=True):
        page = f"page {item.page_number}" if item.page_number else "no page (DOCX)"
        st.markdown(f"**#{rank} · {item.candidate_ref}** · {page} · section `{item.section_id}` · "
                    f"chars {item.source_start}–{item.source_end}")
        st.markdown(highlighted_excerpt(item.content), unsafe_allow_html=True)
        ranks = [f"RRF fusion {item.fusion_score:.4f}",
                 f"dense rank {item.dense_rank or '—'}", f"lexical rank {item.lexical_rank or '—'}"]
        if reranked and item.reranker_score is not None:
            ranks.append(f"cross-encoder {item.reranker_score:.3f}")
        st.caption(" · ".join(ranks) + f" · chunk `{item.source_chunk_id}`")


def _run_search(ctx, query, document_ids, limit, rerank):
    started = time.perf_counter()
    results = ctx.client.search(query, document_ids=document_ids, limit=limit, rerank=rerank)
    return results, (time.perf_counter() - started) * 1_000


def _search_tab(ctx, documents):
    with st.form("search_form", border=True):
        query = st.text_input("Natural-language query", key="search_query", max_chars=2_000,
                              placeholder="e.g. built REST APIs with Python and PostgreSQL")
        document_ids = _document_filter(documents, "search_documents")
        col1, col2 = st.columns(2)
        limit = col1.slider("Maximum passages", 1, 50, 10, key="search_limit")
        rerank = col2.toggle("Cross-encoder reranking", value=False, key="search_rerank",
                             help="Re-scores the fused candidates with ms-marco-MiniLM. Slower; the model "
                                  "downloads on first use.")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if not query.strip():
            st.error("Enter a search query.")
            return
        try:
            with st.spinner("Embedding the query and running hybrid retrieval…"):
                results, latency = _run_search(ctx, query, document_ids, limit, rerank)
            st.session_state["search_last"] = {"results": results, "latency": latency, "rerank": rerank}
        except ApiError as exc:
            show_api_error(exc, "Search")
            return
    last = st.session_state.get("search_last")
    if not last:
        empty_state("Enter a query to search stored resumes.", RELEVANCE_NOTICE)
        return
    results = last["results"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Passages", len(results))
    m2.metric("Documents", len({item.document_id for item in results}))
    m3.metric("Reranking", "On" if last["rerank"] else "Off")
    m4.metric("Round-trip latency", f"{last['latency']:.0f} ms")
    st.caption(RELEVANCE_NOTICE)
    if not results:
        empty_state("No passages matched this query in the selected documents.")
    for rank, item in enumerate(results, start=1):
        _result_card(rank, item, last["rerank"])


def _requirements_tab(ctx, documents):
    st.caption("Runs one hybrid search per requirement and shows the strongest passages. No language model "
               "is used; absence of a passage is not proof the skill is missing.")
    with st.form("requirement_form", border=True):
        requirements_text = st.text_area("Requirements (comma- or newline-separated)", key="requirements_input",
                                         placeholder="Python, PostgreSQL, REST API design")
        document_ids = _document_filter(documents, "requirement_documents")
        per_requirement = st.slider("Passages per requirement", 1, 5, 3, key="requirement_limit")
        submitted = st.form_submit_button("Find evidence", type="primary")
    if not submitted:
        return
    requirements = parse_skill_list(requirements_text)[:MAX_REQUIREMENTS]
    if not requirements:
        st.error("Enter at least one requirement.")
        return
    progress = st.progress(0.0, text="Searching…")
    for index, requirement in enumerate(requirements, start=1):
        progress.progress(index / len(requirements), text=f"Searching {index} of {len(requirements)}")
        try:
            results, _ = _run_search(ctx, requirement, document_ids, per_requirement, False)
        except ApiError as exc:
            show_api_error(exc, f"Search for {requirement}")
            continue
        with st.expander(f"{md(requirement)} · {len(results)} passage(s)", expanded=index == 1):
            if not results:
                st.info("No matching passages were retrieved.")
            for rank, item in enumerate(results, start=1):
                _result_card(rank, item, False)
    progress.empty()


def render(ctx):
    st.header("AI Candidate Search")
    st.caption("Query → MiniLM embedding → pgvector cosine + PostgreSQL full-text → reciprocal-rank fusion → "
               "optional cross-encoder → source evidence")
    if not require_connected(ctx, "AI Candidate Search"):
        return
    try:
        documents = load_documents(ctx)
    except ApiError as exc:
        show_api_error(exc, "Loading the library")
        return
    if not documents:
        empty_state("No documents are indexed yet.", "Upload resumes in the Resume Library first.")
        return
    search_tab, requirements_tab = st.tabs(["Search", "Requirement evidence lookup"])
    with search_tab:
        _search_tab(ctx, documents)
    with requirements_tab:
        _requirements_tab(ctx, documents)
