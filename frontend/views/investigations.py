"""AI Investigations: bounded tool execution, verified reports, and human note review."""

import hashlib
import json
from uuid import UUID, uuid4

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import (
    AI_NOTICE,
    GENERATION_REASONS,
    STATUS_BADGES,
    citation_markers,
    empty_state,
    md,
    render_citations,
    render_sources,
    show_api_error,
    source_index,
)
from frontend.context import document_labels, load_documents
from retrieval.authorization import ROLE_REVIEWER
from utils.skills_extractor import parse_skill_list


MAX_DOCUMENTS = 10


def _render_trace(view):
    st.markdown("#### Execution trace")
    st.caption("Persisted tool events only. Prompts, private reasoning, document excerpts, and credentials are not logged.")
    if not view.events:
        if view.status in {"queued", "running"}:
            st.info("No tool execution event has been persisted yet.")
        else:
            st.info("This investigation has no persisted tool events.")
        return
    st.dataframe([{
        "Step": event.sequence,
        "Tool": event.tool_name,
        "Status": event.status,
        "Duration (ms)": round(event.duration_ms, 1),
        "Sources": ", ".join(event.source_references) or "—",
        "Error category": event.error_category or "",
    } for event in view.events], hide_index=True, width="stretch")


def _render_report(view):
    if view.report is None:
        if view.status == "failed":
            st.error(view.error or "The investigation failed safely without a report.")
        else:
            st.info("No verified investigation report is available yet.")
        return
    st.info(AI_NOTICE)
    report = view.report
    index = source_index(view.sources)

    st.markdown("#### Supported findings")
    if not report.supported_findings:
        st.info("No supported findings were documented.")
    for number, finding in enumerate(report.supported_findings, start=1):
        st.markdown(f"**{number}.** {md(finding.text)} {citation_markers(finding.citations, index)}")
        render_citations(finding.citations, index, key=f"investigation-supported-{number}")

    st.markdown("#### Missing or insufficient evidence")
    if not report.missing_evidence:
        st.caption("No separate evidence gaps were documented.")
    for gap in report.missing_evidence:
        st.warning(f"**{md(gap.topic)}:** {md(gap.explanation)}")

    st.markdown("#### Contradictory evidence")
    if not report.contradictory_evidence:
        st.caption("No contradictory evidence was documented.")
    for number, item in enumerate(report.contradictory_evidence, start=1):
        st.markdown(f"**{number}.** {md(item.statement)} {citation_markers(item.citations, index)}")
        render_citations(item.citations, index, key=f"investigation-contradiction-{number}")

    st.markdown("#### Requirement-to-evidence matrix")
    if not report.requirement_evidence:
        st.caption("No explicit requirements were supplied for this investigation.")
    else:
        st.dataframe([{
            "Requirement": item.requirement,
            "Status": item.status,
            "Evidence": item.explanation,
            "Citations": citation_markers(item.citations, index) if item.citations else "—",
        } for item in report.requirement_evidence], hide_index=True, width="stretch")
        for number, item in enumerate(report.requirement_evidence, start=1):
            if item.citations:
                render_citations(item.citations, index, key=f"investigation-requirement-{number}")

    st.markdown("#### Suggested human verification questions")
    if not report.verification_questions:
        st.caption("No verification questions were generated.")
    for number, item in enumerate(report.verification_questions, start=1):
        st.markdown(f"**Q{number}.** {md(item.question)} {citation_markers(item.citations, index)}")
        st.caption(f"Why verify: {md(item.rationale)}")
        render_citations(item.citations, index, key=f"investigation-question-{number}")
    if view.sources:
        render_sources(view.sources, key=f"investigation-sources-{view.investigation_id}")


def _render_review(ctx, view):
    if view.status == "completed" and view.review:
        verb = "approved" if view.review.decision == "approved" else "rejected"
        st.success(f"These generated investigation notes were **{verb}** by a reviewer on "
                   f"{view.review.reviewed_at:%Y-%m-%d %H:%M} UTC.")
        if view.review.comment:
            st.caption(f"Reviewer comment: {md(view.review.comment)}")
        return
    if view.status != "awaiting_review":
        return
    st.markdown("#### Human review of generated notes")
    st.caption("This approval or rejection applies only to the accuracy and usefulness of the AI-generated notes. "
               "It is not a decision about any applicant.")
    can_review = ctx.has_role(ROLE_REVIEWER)
    key = str(view.investigation_id)
    with st.form(f"investigation_review_{key}"):
        decision = st.radio(
            "Decision on these notes", ["Approve notes", "Reject notes"], horizontal=True,
            key=f"investigation_decision_{key}", disabled=not can_review,
        )
        comment = st.text_area(
            "Reviewer comment (optional)", max_chars=2_000,
            key=f"investigation_comment_{key}", disabled=not can_review,
        )
        submitted = st.form_submit_button("Submit investigation review", type="primary", disabled=not can_review)
    if not can_review:
        st.caption("Your credentials do not include the reviewer role.")
    if submitted:
        try:
            updated = ctx.client.review_investigation(
                view.investigation_id,
                approved=decision == "Approve notes",
                comment=comment,
            )
            st.session_state["investigation_current"] = str(updated.investigation_id)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc, "Submitting the investigation review")


def _render_view(ctx, view):
    st.markdown(f"### Investigation {str(view.investigation_id)[:8]}")
    st.markdown(f"{STATUS_BADGES.get(view.status, view.status)} "
                f"{STATUS_BADGES.get(view.outcome, '') if view.outcome else ''}")
    st.markdown(f"**Objective:** {md(view.objective)}")
    st.caption("Immutable scope: " + ", ".join(
        f"{item.candidate_ref} v{item.version_number}" for item in view.scope
    ))
    a, b, c = st.columns(3)
    a.metric("Status", view.status.replace("_", " ").title())
    b.metric("Elapsed", f"{view.elapsed_seconds:.2f} s")
    c.metric("Tool calls", str(len(view.events)))
    st.caption(f"Synchronous execution · model {view.model_name} · prompt {view.prompt_version} · "
               f"schema {view.schema_version}")
    if view.status in {"queued", "running"}:
        st.info("Execution has not reached a terminal or review state. Refresh to reconcile the persisted checkpoint.")
    if view.status == "failed":
        st.error(view.error or "The investigation failed safely.")
        if view.error_code:
            st.caption(f"Failure category: {md(view.error_code)}")
    _render_trace(view)
    _render_report(view)
    _render_review(ctx, view)


def render(ctx):
    st.header("AI Investigations")
    st.caption("One bounded, read-only LangGraph agent selects authorized evidence tools, verifies citations, "
               "and pauses for human review. It never ranks candidates or makes employment decisions.")
    if not ctx.connected:
        from frontend.components import require_connected
        require_connected(ctx, "AI Investigations")
        return

    generation_enabled = ctx.generation == "enabled"
    if not generation_enabled:
        st.warning(GENERATION_REASONS.get(ctx.generation, "Grounded generation is unavailable."))

    try:
        documents = load_documents(ctx)
    except ApiError as exc:
        show_api_error(exc, "Loading authorized documents")
        documents = []

    st.markdown("### Start an investigation")
    st.caption("Selected document versions form an immutable tenant-scoped evidence boundary. Resume passages are "
               "redacted and bounded before model use; tool outputs are treated as untrusted data.")
    labels = document_labels(documents)
    with st.form("investigation_start", border=True):
        objective = st.text_area(
            "Investigation objective *", max_chars=2_000, key="investigation_objective",
            placeholder="e.g. Investigate evidence of production Python API ownership and identify gaps",
            disabled=not generation_enabled,
        )
        selected = st.multiselect(
            f"Authorized resumes (up to {MAX_DOCUMENTS})", list(labels), format_func=labels.get,
            max_selections=MAX_DOCUMENTS, key="investigation_documents",
            disabled=not generation_enabled,
        )
        job_description = st.text_area(
            "Job description (optional)", max_chars=12_000, height=120,
            key="investigation_job_description", disabled=not generation_enabled,
        )
        requirements_text = st.text_area(
            "Explicit requirements (optional, comma or line separated)", height=80,
            key="investigation_requirements", disabled=not generation_enabled,
        )
        started = st.form_submit_button(
            "Start Investigation", type="primary",
            disabled=not generation_enabled or not documents,
        )
    if not documents:
        st.caption("Upload fictional or authorized resumes in Resume Library before starting an investigation.")
    if started:
        if not objective.strip():
            st.error("Enter an investigation objective.")
        elif not selected:
            st.error("Select at least one authorized resume.")
        else:
            requirements = parse_skill_list(requirements_text)
            fingerprint = hashlib.sha256(json.dumps({
                "objective": objective.strip(),
                "documents": sorted(selected),
                "job_description": job_description.strip(),
                "requirements": requirements,
            }, sort_keys=True).encode("utf-8")).hexdigest()
            pending = st.session_state.get("investigation_pending_submission")
            key = pending["key"] if pending and pending["fingerprint"] == fingerprint else str(uuid4())
            st.session_state["investigation_pending_submission"] = {"key": key, "fingerprint": fingerprint}
            try:
                with st.spinner("Running synchronously: the agent is selecting tools and validating evidence…"):
                    view = ctx.client.start_investigation(
                        objective=objective.strip(),
                        document_ids=[UUID(value) for value in selected],
                        idempotency_key=key,
                        job_description=job_description.strip(),
                        requirements=requirements,
                    )
                st.session_state["investigation_current"] = str(view.investigation_id)
                st.session_state.pop("investigation_pending_submission", None)
                st.rerun()
            except ApiError as exc:
                show_api_error(exc, "Starting the investigation")

    st.markdown("### Investigation history")
    st.button("↻ Refresh investigations", key="investigation_refresh")
    try:
        investigations = ctx.client.list_investigations().investigations
    except ApiError as exc:
        show_api_error(exc, "Loading investigation history")
        return
    if not investigations:
        empty_state("No investigations have been created in this workspace.")
        return
    st.dataframe([{
        "Investigation": str(item.investigation_id)[:8],
        "Objective": item.objective,
        "Candidates": ", ".join(item.candidate_refs),
        "Status": item.status.replace("_", " ").title(),
        "Notes review": (item.outcome or "—").title(),
        "Started": item.created_at,
        "Updated": item.updated_at,
    } for item in investigations], hide_index=True, width="stretch", column_config={
        "Started": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "Updated": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
    })
    options = {str(item.investigation_id): item for item in investigations}
    current = st.session_state.get("investigation_current")
    if current not in options:
        current = None
    selected_history = st.selectbox(
        "Open an investigation", [None, *options],
        index=[None, *options].index(current) if current else 0,
        key="investigation_history_selected",
        format_func=lambda value: "Select an investigation…" if value is None else
        f"{value[:8]} · {options[value].status.replace('_', ' ').title()} · {options[value].objective[:70]}",
    )
    selected_id = selected_history or current
    if not selected_id:
        return
    try:
        view = ctx.client.get_investigation(UUID(selected_id))
    except ApiError as exc:
        show_api_error(exc, "Opening the investigation")
        return
    with st.container(border=True):
        _render_view(ctx, view)
