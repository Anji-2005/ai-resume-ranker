"""Resume Library: persistent, tenant-scoped V2 document storage."""

import hashlib
from uuid import UUID, uuid4

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import empty_state, highlighted_excerpt, require_connected, show_api_error
from frontend.context import document_labels, invalidate_documents, load_documents
from retrieval.authorization import ROLE_EDITOR

MAX_BATCH_FILES = 20


def _upload_batch(ctx, files):
    """Queue files one by one; the worker performs parsing and indexing later."""
    rows = []
    with st.status(f"Queueing {len(files)} file(s)…", expanded=True) as status:
        for index, file in enumerate(files, start=1):
            # Filenames are never displayed or sent: they may identify the candidate.
            label = f"File {index:02d}"
            st.write(f"{label}: validating upload bounds and creating a durable job…")
            try:
                data = file.getvalue()
                key = "upload-" + hashlib.sha256(data).hexdigest()[:32]
                job = ctx.client.enqueue_document(file.name, data, idempotency_key=key)
                rows.append({"File": label, "Result": "Queued", "Job": str(job.job_id)[:8],
                             "Status": job.status})
            except ApiError as exc:
                rows.append({"File": label, "Result": "❌ " + exc.message, "Job": "—", "Status": "failed"})
        failed = sum(row["Status"] == "failed" for row in rows)
        status.update(
            label=f"Queued {len(files) - failed} file(s); {failed} upload request(s) failed",
            state="error" if failed == len(files) else "complete", expanded=False,
        )
    st.session_state["library_upload_report"] = rows


def _job_status_panel(ctx):
    header, refresh = st.columns([4, 1])
    header.subheader("Ingestion operations")
    refresh.button("↻ Refresh jobs", key="library_jobs_refresh", width="stretch")
    try:
        jobs = ctx.client.list_ingestion_jobs(limit=100).jobs
        metrics = ctx.client.ingestion_metrics()
    except ApiError as exc:
        show_api_error(exc, "Loading ingestion operations")
        return
    cols = st.columns(4)
    for column, state in zip(cols, ("queued", "processing", "completed", "failed"), strict=True):
        column.metric(state.replace("_", " ").title(), metrics.counts.get(state, 0))
    st.caption(
        f"Last hour: {metrics.completed_last_hour} completed · {metrics.failed_last_hour} failed · "
        f"{metrics.throughput_per_minute_last_hour:.3f} completed/min"
    )
    if not jobs:
        empty_state("No ingestion jobs have been submitted for this workspace.")
        return
    rows = []
    for job in jobs:
        result = job.error or (job.candidate_ref if job.status == "completed" else "—")
        rows.append({
            "Job": str(job.job_id)[:8], "Type": job.job_type, "Status": job.status,
            "Attempts": f"{job.attempt_count}/{job.max_attempts}", "Result": result,
            "Last error": job.error_code or "—",
            "Version": job.version_number, "Latency ms": job.processing_latency_ms,
            "Queued": job.queued_at,
        })
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"Queued": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")})
    if any(job.status in {"queued", "processing"} for job in jobs):
        st.info("Work is running in the background. Refresh this panel for persisted status; no progress is fabricated.")
    if any(job.status == "completed" for job in jobs):
        invalidate_documents()


def _document_table(documents):
    rows = [{
        "Candidate": doc.candidate_ref, "Current version": doc.current_version_number,
        "Versions": doc.version_count, "Format": "PDF" if doc.media_type.endswith("pdf") else "DOCX",
        "Chunks": doc.chunk_count, "First uploaded": doc.created_at, "Last updated": doc.updated_at,
    } for doc in documents]
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={
            "First uploaded": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Last updated": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )


def _document_detail(ctx, document_id: str):
    version_key = f"library_version_{document_id}"
    try:
        detail = ctx.client.get_document(UUID(document_id), st.session_state.get(version_key))
    except ApiError as exc:
        show_api_error(exc, "Loading the document")
        return
    st.markdown(f"#### {detail.candidate_ref}")
    st.caption(f"Opaque document ID `{detail.document_id}` · filenames are never stored or shown")
    versions = [{
        "Version": v.version_number, "Current": "✓" if v.is_current else "", "Chunks": v.chunk_count,
        "Parser": f"{v.parser_name} {v.parser_version}", "Content SHA-256": v.content_sha256_prefix + "…",
        "Embedding": f"{v.embedding_model} · {v.embedding_model_version}",
        "Uploaded": v.created_at,
    } for v in detail.versions]
    st.markdown("**Version history**")
    st.dataframe(versions, hide_index=True, width="stretch",
                 column_config={"Uploaded": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    numbers = [v.version_number for v in detail.versions]
    if len(numbers) > 1:
        st.selectbox("View extracted content for version", numbers, index=numbers.index(detail.selected_version_number),
                     key=version_key)
    st.markdown(f"**Extracted content and provenance · version {detail.selected_version_number}**")
    st.caption("Common direct identifiers (email, phone, URL, header name) are redacted for display. "
               "Offsets refer to the parser's normalized text.")
    if not detail.chunks:
        empty_state("This version has no indexed text.")
    for chunk in detail.chunks:
        page = f"page {chunk.page_number}" if chunk.page_number else "no page (DOCX)"
        with st.expander(f"{chunk.chunk_key} · {page} · chars {chunk.source_start}–{chunk.source_end}"):
            st.markdown(highlighted_excerpt(chunk.content), unsafe_allow_html=True)

    can_edit = ctx.has_role(ROLE_EDITOR)
    left, right = st.columns(2)
    with left:
        st.markdown("**Upload a new version**")
        new_version = st.file_uploader("Replacement PDF or DOCX", type=["pdf", "docx"],
                                       key=f"library_new_version_{document_id}", disabled=not can_edit)
        if st.button("Store new version", key=f"library_store_version_{document_id}",
                     disabled=not (can_edit and new_version)):
            try:
                job = ctx.client.enqueue_document_version(
                    UUID(document_id), new_version.name, new_version.getvalue(),
                    idempotency_key="version-" + uuid4().hex,
                )
                st.success(f"Queued background job {str(job.job_id)[:8]} for a new version.")
                st.session_state.pop(version_key, None)
            except ApiError as exc:
                show_api_error(exc, "Uploading the new version")
        st.markdown("**Reindex embeddings**")
        st.caption("Queues this displayed version for transactional re-embedding with the configured model.")
        if st.button("Queue reindex", key=f"library_reindex_{document_id}", disabled=not can_edit):
            try:
                job = ctx.client.enqueue_reindex(
                    UUID(document_id), version_number=detail.selected_version_number,
                    idempotency_key="reindex-" + uuid4().hex,
                )
                st.success(f"Queued reindex job {str(job.job_id)[:8]}.")
            except ApiError as exc:
                show_api_error(exc, "Queueing reindex")
    with right:
        st.markdown("**Delete document**")
        st.caption("Physically deletes every version, chunk, and embedding for this candidate document.")
        confirmed = st.checkbox(f"I understand {detail.candidate_ref} will be permanently deleted",
                                key=f"library_confirm_delete_{document_id}", disabled=not can_edit)
        if st.button("Delete permanently", type="primary", key=f"library_delete_{document_id}",
                     disabled=not (can_edit and confirmed)):
            try:
                ctx.client.delete_document(UUID(document_id))
                invalidate_documents()
                st.session_state.pop("library_selected", None)
                st.session_state["library_flash"] = f"{detail.candidate_ref} was deleted."
                st.rerun()
            except ApiError as exc:
                show_api_error(exc, "Deleting the document")
    if not can_edit:
        st.caption("Your credentials do not include the editor role, so upload and delete are disabled.")


def render(ctx):
    st.header("Resume Library")
    st.caption("Upload → validate → parse → embed → store versions and chunks in PostgreSQL/pgvector")
    if not require_connected(ctx, "Resume Library"):
        return
    if flash := st.session_state.pop("library_flash", None):
        st.success(flash)

    with st.container(border=True):
        st.markdown("**Add resumes**")
        files = st.file_uploader(
            "PDF or DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True, key="library_uploads",
            help=f"Up to {MAX_BATCH_FILES} files per batch, 10 MB each. Re-uploading identical bytes is idempotent.",
            disabled=not ctx.has_role(ROLE_EDITOR),
        )
        if files and len(files) > MAX_BATCH_FILES:
            st.error(f"Select at most {MAX_BATCH_FILES} files per batch.")
        elif st.button("Upload and index", type="primary", disabled=not files, key="library_upload"):
            _upload_batch(ctx, files)
        if report := st.session_state.get("library_upload_report"):
            st.dataframe(report, hide_index=True, width="stretch")

    _job_status_panel(ctx)

    header, refresh = st.columns([4, 1])
    header.subheader("Stored documents")
    refresh_clicked = refresh.button("↻ Refresh", key="library_refresh", width="stretch")
    try:
        documents = load_documents(ctx, refresh=refresh_clicked)
    except ApiError as exc:
        show_api_error(exc, "Loading the library")
        return
    if not documents:
        empty_state("No documents are stored for this workspace yet.",
                    "Upload synthetic PDF or DOCX resumes above to enable search, Q&A, and briefs.")
        return
    _document_table(documents)
    labels = document_labels(documents)
    selected = st.selectbox("Inspect a document", [None, *labels], key="library_selected",
                            format_func=lambda value: "Select a document…" if value is None else labels[value])
    if selected:
        _document_detail(ctx, selected)
