"""Ask AI: grounded, citation-verified question answering over stored resumes."""

from uuid import UUID

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import AI_NOTICE, empty_state, md, render_answer, require_connected, require_generation, show_api_error
from frontend.context import document_labels, load_documents

MAX_HISTORY = 10
SUGGESTED_QUESTIONS = [
    "Which resumes mention experience with Python and PostgreSQL?",
    "What evidence describes API development experience?",
    "What technical skills are mentioned?",
    "What project experience involves cloud infrastructure?",
]


def _use_suggestion(question: str):
    st.session_state["ask_question"] = question


def render(ctx):
    st.header("Ask AI — Resume Intelligence")
    st.caption("Retrieves authorized evidence, asks the configured model for structured claims, and accepts "
               "only claims whose citations are exact excerpts of the supplied evidence.")
    if not require_connected(ctx, "Ask AI"):
        return
    if not require_generation(ctx):
        return
    try:
        documents = load_documents(ctx)
    except ApiError as exc:
        show_api_error(exc, "Loading the library")
        return
    if not documents:
        empty_state("No documents are indexed yet.", "Upload resumes in the Resume Library first.")
        return
    st.info(AI_NOTICE)

    st.markdown("**Suggested questions**")
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for position, (column, question) in enumerate(zip(columns, SUGGESTED_QUESTIONS)):
        column.button(question, key=f"suggest_{position}", on_click=_use_suggestion, args=(question,),
                      width="stretch")

    labels = document_labels(documents)
    with st.form("ask_form", border=True):
        question = st.text_area("Your question", key="ask_question", max_chars=2_000, height=90)
        selected = st.multiselect("Limit to documents (optional)", list(labels), format_func=labels.get,
                                  key="ask_documents")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        if not question.strip():
            st.error("Enter a question.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating a cited answer…"):
                    answer = ctx.client.ask(question.strip(), document_ids=[UUID(value) for value in selected])
                history = st.session_state.setdefault("ask_history", [])
                history.insert(0, {"question": question.strip(), "answer": answer})
                del history[MAX_HISTORY:]
            except ApiError as exc:
                show_api_error(exc, "Generating the answer")

    history = st.session_state.get("ask_history", [])
    if not history:
        empty_state("Ask a question to see a grounded answer with citations.")
        return
    if st.button("Clear answers", key="ask_clear"):
        st.session_state.pop("ask_history", None)
        st.rerun()
    for number, entry in enumerate(history):
        with st.container(border=True):
            st.markdown(f"**Q:** {md(entry['question'])}")
            render_answer(entry["answer"], key=f"answer-{number}")
