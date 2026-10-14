"""Review History: durable, tenant-scoped workflow sessions and their review status."""

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import empty_state, require_connected, show_api_error
from frontend.views.briefs import render_review_panel

STATUS_OPTIONS = {
    "All": None, "Awaiting review": "awaiting_review", "Completed": "completed", "Failed": "failed",
}


def render(ctx):
    st.header("Review History")
    st.caption("Recruiter-brief workflows persist in PostgreSQL checkpoints, so pending reviews survive restarts.")
    if not require_connected(ctx, "Review History"):
        return
    col1, col2 = st.columns([3, 1])
    choice = col1.segmented_control("Status", list(STATUS_OPTIONS), default="All", key="history_status")
    col2.button("↻ Refresh", key="history_refresh", width="stretch")
    try:
        workflows = ctx.client.list_workflows(STATUS_OPTIONS.get(choice or "All")).workflows
    except ApiError as exc:
        show_api_error(exc, "Loading review history")
        return
    if not workflows:
        empty_state("No recruiter-brief workflows match this filter.",
                    "Generate a brief in Recruiter Briefs to start one.")
        return
    labels = {"awaiting_review": "Awaiting review", "completed": "Reviewed", "failed": "Failed", "running": "Running"}
    st.dataframe([{
        "Workflow": str(item.workflow_id)[:8], "Candidates": ", ".join(item.candidate_refs),
        "Status": labels.get(item.status, item.status),
        "Notes decision": {"approved": "Approved", "rejected": "Rejected"}.get(item.outcome, "—"),
        "Error": item.error_code or "", "Started": item.created_at, "Updated": item.updated_at,
    } for item in workflows], hide_index=True, width="stretch", column_config={
        "Started": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "Updated": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
    })
    options = {str(item.workflow_id): item for item in workflows}
    selected = st.selectbox(
        "Open a workflow", [None, *options], key="history_selected",
        format_func=lambda value: "Select a workflow…" if value is None else
        f"{value[:8]} · {', '.join(options[value].candidate_refs)} · {labels.get(options[value].status)}",
    )
    if not selected:
        return
    try:
        view = ctx.client.get_workflow(options[selected].workflow_id)
    except ApiError as exc:
        show_api_error(exc, "Loading the workflow")
        return
    with st.container(border=True):
        render_review_panel(ctx, view, key=f"history-{selected}")
