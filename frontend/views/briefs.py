"""Recruiter Briefs: start the LangGraph workflow and review AI-generated notes."""

from uuid import UUID

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import (
    AI_NOTICE,
    STATUS_BADGES,
    empty_state,
    md,
    render_brief,
    require_connected,
    require_generation,
    show_api_error,
)
from frontend.context import document_labels, load_documents
from retrieval.authorization import ROLE_REVIEWER
from utils.skills_extractor import parse_skill_list

MAX_DOCUMENTS = 5


def render_review_panel(ctx, view, key: str):
    """Show one workflow's brief, evidence, and the approve/reject form for its AI-generated notes."""
    st.markdown(f"{STATUS_BADGES.get(view.status, view.status)} "
                f"{STATUS_BADGES.get(view.outcome, '') if view.outcome else ''} "
                f"**{', '.join(view.candidate_refs)}** · workflow `{view.workflow_id}`")
    if view.status == "failed":
        st.error(view.error or "The workflow could not be completed.")
        st.caption("Nothing was persisted. Adjust the requirements or documents and start a new brief.")
        return
    if view.brief is None:
        st.info("This workflow has no brief to display.")
        return
    st.info(AI_NOTICE)
    render_brief(view.brief, view.sources, key=key)
    if view.status == "completed" and view.review:
        verdict = "approved" if view.review.decision == "approved" else "rejected"
        st.success(f"These AI-generated notes were **{verdict}** by a reviewer on "
                   f"{view.review.reviewed_at:%Y-%m-%d %H:%M} UTC.")
        if view.review.note:
            st.caption(f"Reviewer note: {md(view.review.note)}")
        return
    if view.status != "awaiting_review":
        return
    st.markdown("##### Human review")
    st.caption("You are approving or rejecting the AI-generated notes for accuracy and usefulness — "
               "not the applicant. Editing generated briefs is not supported.")
    can_review = ctx.has_role(ROLE_REVIEWER)
    with st.form(f"review_form_{key}"):
        decision = st.radio("Decision on these notes", ["Approve notes", "Reject notes"], horizontal=True,
                            key=f"review_decision_{key}", disabled=not can_review)
        note = st.text_area("Reviewer note (optional)", max_chars=2_000, key=f"review_note_{key}",
                            disabled=not can_review,
                            placeholder="e.g. citations verified against the source excerpts")
        submitted = st.form_submit_button("Submit review", type="primary", disabled=not can_review)
    if not can_review:
        st.caption("Your credentials do not include the reviewer role.")
    if submitted:
        try:
            updated = ctx.client.submit_review(view.workflow_id, approved=decision == "Approve notes", note=note)
            st.session_state.setdefault("workflow_views", {})[str(updated.workflow_id)] = updated
            st.rerun()
        except ApiError as exc:
            show_api_error(exc, "Submitting the review")


def render(ctx):
    st.header("Recruiter Briefs")
    st.caption("Authorize → retrieve evidence → validate requirements → generate → verify citations → "
               "human review → persist (one bounded LangGraph workflow per document)")
    if not require_connected(ctx, "Recruiter Briefs"):
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
    labels = document_labels(documents)
    with st.form("brief_form", border=True):
        selected = st.multiselect(f"Candidate documents (up to {MAX_DOCUMENTS}; one brief each)", list(labels),
                                  format_func=labels.get, max_selections=MAX_DOCUMENTS, key="brief_documents")
        description = st.text_area("Job description *", height=160, key="brief_description", max_chars=12_000)
        col1, col2 = st.columns(2)
        required = col1.text_area("Required skills", height=80, key="brief_required")
        preferred = col2.text_area("Preferred skills", height=80, key="brief_preferred")
        submitted = st.form_submit_button("Generate briefs for review", type="primary")
    if submitted:
        if not selected:
            st.error("Select at least one document.")
        elif not description.strip():
            st.error("Enter a job description.")
        else:
            try:
                with st.spinner(f"Running {len(selected)} workflow(s): retrieval, generation, citation checks…"):
                    started = ctx.client.start_brief_workflows(
                        description=description.strip(), required_skills=parse_skill_list(required),
                        preferred_skills=parse_skill_list(preferred),
                        document_ids=[UUID(value) for value in selected],
                    )
                views = st.session_state.setdefault("workflow_views", {})
                st.session_state["brief_current"] = []
                for view in started.workflows:
                    views[str(view.workflow_id)] = view
                    st.session_state["brief_current"].append(str(view.workflow_id))
            except ApiError as exc:
                show_api_error(exc, "Generating briefs")
    current = st.session_state.get("brief_current", [])
    views = st.session_state.get("workflow_views", {})
    if not current:
        empty_state("Select documents and describe the role to generate evidence-backed briefs.",
                    "Pending reviews can also be resumed later from Review History.")
        return
    for workflow_id in current:
        view = views.get(workflow_id)
        if view is not None:
            with st.container(border=True):
                render_review_panel(ctx, view, key=workflow_id)
