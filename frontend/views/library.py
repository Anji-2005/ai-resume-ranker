"""Resume Library: persistent, tenant-scoped V2 document storage."""

from uuid import UUID

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import empty_state, highlighted_excerpt, require_connected, show_api_error
from frontend.context import document_labels, invalidate_documents, load_documents
from retrieval.authorization import ROLE_EDITOR

MAX_BATCH_FILES = 20


def _upload_batch(ctx, files):
    """Upload files one by one with per-file status; failures never stop the batch."""
    rows = []
    with st.status(f"Processing {len(files)} file(s)…", expanded=True) as status:
        for index, file in enumerate(files, start=1):
            # Filenames are never displayed or sent: they may identify the candidate.
            label = f"File {index:02d}"
            st.write(f"{label}: validating, parsing, embedding, and indexing…")
            try:
                result = ctx.client.upload_document(file.name, file.getvalue())
                outcome = "Stored new document" if result.created and result.version_number == 1 else (
                    f"Stored as version {result.version_number}" if result.created else "Already stored (unchanged)"
                )
                rows.append({"File": label, "Result": "✅ " + outcome, "Candidate": result.candidate_ref,
                             "Version": result.version_number})
            except ApiError as exc:
                rows.append({"File": label, "Result": "❌ " + exc.message, "Candidate": "—", "Version": None})
        failed = sum(row["Result"].startswith("❌") for row in rows)
        status.update(
            label=f"Processed {len(files)} file(s): {len(files) - failed} stored, {failed} failed",
            state="error" if failed == len(files) else "complete", expanded=False,
        )
    invalidate_documents()
    st.session_state["library_upload_report"] = rows


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
                result = ctx.client.upload_version(UUID(document_id), new_version.name, new_version.getvalue())
                st.success(f"Stored version {result.version_number}." if result.created
                           else "This exact file is already stored; no new version was created.")
                st.session_state.pop(version_key, None)
                invalidate_documents()
            except ApiError as exc:
                show_api_error(exc, "Uploading the new version")
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
