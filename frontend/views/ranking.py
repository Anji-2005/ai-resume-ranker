"""Candidate Ranking: the original deterministic, in-session review workflow.

Scoring, Gemini enrichment, filtering, comparison, human review records, and
exports behave exactly as before; only presentation is improved.
"""

from datetime import datetime, timezone
import json
import os

import streamlit as st

from frontend.api_client import ApiError
from frontend.components import empty_state
from frontend.context import invalidate_documents
from services.ranking import HUMAN_DECISIONS, analyze_resume_bytes, build_export_frame, excel_bytes, rank_results
from utils.genai_analyzer import clear_enrichment_cache
from utils.matcher import MODEL_NAME, get_model
from utils.pdf_parser import ResumeParseError
from utils.privacy import safe_candidate_label, safe_failure_record, sanitize_audit_records
from utils.skills_extractor import extract_skills, parse_skill_list


COMPONENT_LABELS = {
    "semantic": "Semantic similarity",
    "required_skills": "Required skills",
    "preferred_skills": "Preferred skills",
    "experience": "Experience",
}


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return get_model()


def get_gemini_api_key():
    """Read optional Gemini credentials without requiring a secrets file."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("GEMINI_API_KEY", "")).strip()


def analyze_resume(uploaded_file, index, job, model, genai_api_key=""):
    return analyze_resume_bytes(uploaded_file.name, uploaded_file.getvalue(), index, job, model, genai_api_key)


def render_score_breakdown(candidate):
    columns = st.columns(4)
    for column, key in zip(columns, COMPONENT_LABELS):
        with column:
            st.metric(COMPONENT_LABELS[key], f"{candidate['components'][key]:.1f}%")
            st.progress(min(1.0, candidate["components"][key] / 100))
            weight = candidate["weights"][key] * 100
            st.caption(f"Weight: {weight:.0f}% · contributes {candidate['components'][key] * candidate['weights'][key]:.1f} pts")


def _skill_chips(matched, missing):
    chips = [f":green-badge[{skill}]" for skill in matched] + [f":red-badge[{skill} missing]" for skill in missing]
    return " ".join(chips) or "None configured"


def _scoring_settings():
    with st.expander("Scoring settings", expanded=False):
        blind_review = st.toggle(
            "Blind-review mode", value=True, key="blind_review",
            help="Redacts common direct identifiers before scoring and replaces filenames with candidate IDs.",
        )
        st.markdown("**Score weights**")
        col1, col2, col3, col4 = st.columns(4)
        semantic_weight = col1.slider("Semantic match", 0, 100, 40, 5, key="w_semantic")
        required_weight = col2.slider("Required skills", 0, 100, 35, 5, key="w_required")
        preferred_weight = col3.slider("Preferred skills", 0, 100, 10, 5, key="w_preferred")
        experience_weight = col4.slider("Experience", 0, 100, 15, 5, key="w_experience")
        weight_total = semantic_weight + required_weight + preferred_weight + experience_weight
        if weight_total == 0:
            st.error("At least one scoring weight must be greater than zero.")
        elif weight_total != 100:
            st.caption(f"Weights total {weight_total}%; they will be normalized automatically.")
        if st.button("Delete session results", key="delete_session_results"):
            for key in ["ranking_results", "ranking_failures", "job_config", "audit_timestamp"]:
                st.session_state.pop(key, None)
            clear_enrichment_cache()
            st.success("Candidate results were removed from this session.")
    return blind_review, {
        "semantic": semantic_weight, "required_skills": required_weight,
        "preferred_skills": preferred_weight, "experience": experience_weight,
    }, weight_total


def _index_in_library(ctx, uploaded_resumes, blind_review):
    """Optionally persist the same uploads to the V2 library; ranking itself stays in-session."""
    stored, failed = 0, 0
    for index, resume in enumerate(uploaded_resumes, start=1):
        try:
            ctx.client.upload_document(resume.name, resume.getvalue())
            stored += 1
        except ApiError as exc:
            failed += 1
            label = safe_candidate_label(resume.name, index, blind_review)
            st.warning(f"{label} was not indexed in the Resume Library: {exc.message}")
    invalidate_documents()
    if stored:
        st.success(f"Indexed {stored} file(s) in the Resume Library for search, Q&A, and briefs.")


def _configure_tab(ctx, blind_review, weights, weight_total):
    st.subheader("Define the role")
    job_description = st.text_area(
        "Job description *", height=220, key="job_description",
        placeholder="Describe responsibilities, qualifications, and the role context…",
    )
    detected_job_skills = extract_skills(job_description)
    if detected_job_skills:
        st.caption("Detected skills: " + ", ".join(detected_job_skills))
        if st.button("Use detected skills as required criteria"):
            st.session_state["required_skills_input"] = ", ".join(detected_job_skills)
    col1, col2 = st.columns(2)
    with col1:
        required_input = st.text_area(
            "Required skills", height=100, key="required_skills_input",
            help="Comma- or newline-separated. Use the detected-skills button for suggestions.",
        )
    with col2:
        preferred_input = st.text_area(
            "Preferred skills", height=100, key="preferred_skills_input",
            help="Optional skills that improve a score but are not mandatory.",
        )
    minimum_years = st.number_input(
        "Minimum relevant experience (years)", min_value=0.0, max_value=40.0,
        value=0.0, step=0.5, key="minimum_years",
    )
    uploaded_resumes = st.file_uploader(
        "Upload resumes *", type=["pdf", "docx"], accept_multiple_files=True, key="ranking_uploads",
        help="PDF or DOCX, maximum 10 MB each. Scanned PDFs are detected but require OCR.",
    )
    index_in_library = False
    if ctx.connected:
        index_in_library = st.checkbox(
            "Also index these files in the Resume Library (V2)", value=False, key="ranking_index_library",
            help="Stores parsed text and embeddings in PostgreSQL for search, Q&A, and briefs. "
                 "Ranking results themselves remain in this session only.",
        )
    analyze_button = st.button(
        "Analyze candidates", type="primary", disabled=(weight_total == 0), width="stretch",
    )
    if not analyze_button:
        return
    required_skills = parse_skill_list(required_input)
    preferred_skills = [skill for skill in parse_skill_list(preferred_input) if skill not in required_skills]
    if not job_description.strip():
        st.error("Enter a job description before analyzing resumes.")
        return
    if not uploaded_resumes:
        st.error("Upload at least one resume.")
        return
    if not required_skills:
        st.error("Add at least one required skill so the ranking has explicit criteria.")
        return
    job = {
        "description": job_description.strip(),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "minimum_years": float(minimum_years),
        "blind_review": blind_review,
        "weights": weights,
    }
    results, failures = [], []
    genai_api_key = get_gemini_api_key()
    progress = st.progress(0, text="Loading the language model…")
    try:
        model = load_embedding_model()
    except Exception as exc:
        progress.empty()
        st.error(f"The language model could not be loaded: {exc}")
        return
    for index, resume in enumerate(uploaded_resumes, start=1):
        display_name = safe_candidate_label(resume.name, index, job["blind_review"])
        progress.progress(
            (index - 1) / len(uploaded_resumes),
            text=f"Analyzing {index} of {len(uploaded_resumes)}: {display_name}",
        )
        try:
            results.append(analyze_resume(resume, index, job, model, genai_api_key=genai_api_key))
        except ResumeParseError as exc:
            failures.append(safe_failure_record(resume.name, index, exc, job["blind_review"]))
        except Exception as exc:
            failures.append(safe_failure_record(
                resume.name, index, f"Unexpected analysis error: {exc}", job["blind_review"],
            ))
    progress.progress(1.0, text="Analysis complete")
    rank_results(results)
    st.session_state["ranking_results"] = results
    st.session_state["ranking_failures"] = failures
    st.session_state["job_config"] = job
    st.session_state["audit_timestamp"] = datetime.now(timezone.utc).isoformat()
    if results:
        st.success(f"Analyzed {len(results)} candidate(s). Open the Results tab.")
    if failures:
        st.warning(f"Skipped {len(failures)} file(s). See Results for details.")
    if index_in_library:
        _index_in_library(ctx, uploaded_resumes, blind_review)


def _candidate_detail(item):
    render_score_breakdown(item)
    st.caption("Deterministic score components above · narrative text below may be AI-generated")
    genai_result = item.get("genai", {})
    if genai_result.get("status") == "success":
        st.caption("AI-generated candidate summary · Verify against resume evidence")
    else:
        st.caption(genai_result.get("message", "Rule-based summary shown; GenAI enrichment was not used."))
    st.write(item["summary"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Required skill coverage**")
        st.markdown(_skill_chips(item["matched_required"], item["missing_required"]))
        st.markdown("**Preferred skill coverage**")
        st.markdown(_skill_chips(item["matched_preferred"], item.get("missing_preferred", [])))
    with right:
        st.markdown("**Missing required skills**")
        st.write(", ".join(item["missing_required"]) or "None")
        st.markdown("**Experience estimate**")
        st.write(f"{item['years']:g} explicitly stated years · {item['experience_level']}")
    st.markdown("**Resume evidence**")
    if item["evidence"]:
        for skill, snippets in item["evidence"].items():
            st.markdown(f"- **{skill}:** " + (" / ".join(snippets) if snippets else "No snippet found"))
    else:
        st.write("No matched-skill evidence found.")
    if genai_result.get("status") == "success":
        st.markdown("**AI requirement evidence**")
        st.caption("AI-generated analysis; exact excerpts are verified against submitted text.")
        for evidence_item in genai_result["requirement_evidence"]:
            st.markdown(
                f"- **{evidence_item['requirement']} — "
                f"{evidence_item['status'].title()}:** {evidence_item['explanation']}"
            )
            if evidence_item["evidence"]:
                st.caption(f"Resume excerpt: “{evidence_item['evidence']}”")
        st.markdown("**AI-tailored interview questions**")
        st.caption("Suggestions for human reviewers; do not use as autonomous decisions.")
        for question_item in genai_result["interview_questions"]:
            st.markdown(f"- {question_item['question']}")
            st.caption(f"Reason: {question_item['reason']}")
    st.markdown("**Review cautions**")
    for flag in item["red_flags"]:
        st.markdown(f"- {flag}")
    st.markdown("**Resume improvement suggestions**")
    if genai_result.get("status") == "success":
        st.caption("AI-generated suggestions; only add claims that are truthful.")
    for suggestion in item["suggestions"]:
        st.markdown(f"- {suggestion}")
    st.markdown("**Human review record**")
    decision = st.selectbox(
        "Outcome", HUMAN_DECISIONS,
        index=HUMAN_DECISIONS.index(item.get("human_decision", "Unreviewed")),
        key=f"decision_{item['rank']}_{item['candidate']}",
    )
    rationale = st.text_input(
        "Reviewer rationale", value=item.get("human_reason", ""),
        placeholder="Record evidence-based reasoning; avoid protected characteristics.",
        key=f"reason_{item['rank']}_{item['candidate']}",
    )
    if st.button("Save human review", key=f"save_{item['rank']}_{item['candidate']}"):
        item["human_decision"] = decision
        item["human_reason"] = rationale.strip()
        st.success("Human review record saved for this session and future exports.")


def _results_tab():
    results = st.session_state.get("ranking_results", [])
    failures = st.session_state.get("ranking_failures", [])
    if not results:
        empty_state("Configure the role and analyze resumes to see results.",
                    "Results stay in this browser session; use the exports below to keep an audit copy.")
        if failures:
            st.subheader("Files requiring attention")
            st.dataframe(failures, hide_index=True, width="stretch")
        return
    st.subheader("Candidate ranking")
    filter_col, skill_col = st.columns(2)
    with filter_col:
        minimum_score = st.slider("Minimum review score", 0, 100, 0, key="minimum_score_filter")
    with skill_col:
        job_skills = st.session_state["job_config"]["required_skills"]
        skill_filter = st.multiselect("Must contain these required skills", job_skills, key="skill_filter")
    visible = [item for item in results if item["score"] >= minimum_score and
               all(skill in item["matched_required"] for skill in skill_filter)]
    frame = build_export_frame(visible)
    if visible:
        best = visible[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Highest-ranked", best["candidate"])
        m2.metric("Review score", f"{best['score']:.1f}%")
        m3.metric("Review priority", best["recommendation"])
        m4.metric("Candidates shown", f"{len(visible)} / {len(results)}")
        st.bar_chart(frame.set_index("Candidate")[["Review Score"]], horizontal=True)
        progress = st.column_config.ProgressColumn
        st.dataframe(
            frame, hide_index=True, width="stretch",
            column_config={
                "Review Score": progress("Review Score", format="%.1f", min_value=0, max_value=100),
                "Semantic Similarity": progress("Semantic", format="%.1f", min_value=0, max_value=100),
                "Required Skill Coverage": progress("Required", format="%.1f", min_value=0, max_value=100),
                "Preferred Skill Coverage": progress("Preferred", format="%.1f", min_value=0, max_value=100),
                "Experience Match": progress("Experience", format="%.1f", min_value=0, max_value=100),
            },
        )
    else:
        st.warning("No candidates match the active filters.")

    st.subheader("Candidate details")
    for item in visible:
        with st.expander(f"#{item['rank']} · {item['candidate']} · {item['score']:.1f}% · {item['recommendation']}"):
            _candidate_detail(item)

    if len(visible) >= 2:
        st.subheader("Side-by-side comparison")
        selected = st.multiselect(
            "Choose 2–4 candidates", [item["candidate"] for item in visible], max_selections=4, key="compare",
        )
        if len(selected) >= 2:
            comparison = frame[frame["Candidate"].isin(selected)].set_index("Candidate").T
            st.dataframe(comparison, width="stretch")

    st.subheader("Export")
    export_frame = build_export_frame(results)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download CSV", export_frame.to_csv(index=False).encode("utf-8"),
        "candidate_review.csv", "text/csv", width="stretch",
    )
    try:
        download_cols[1].download_button(
            "Download Excel", excel_bytes(export_frame), "candidate_review.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
        )
    except ImportError:
        download_cols[1].caption("Install openpyxl to enable Excel export.")
    audit_results, audit_failures = sanitize_audit_records(
        results, failures, bool(st.session_state.get("job_config", {}).get("blind_review", False)),
    )
    audit = {
        "generated_at": st.session_state.get("audit_timestamp"),
        "model": MODEL_NAME,
        "job_configuration": st.session_state.get("job_config"),
        "candidate_results": audit_results,
        "failed_files": audit_failures,
    }
    download_cols[2].download_button(
        "Download audit JSON", json.dumps(audit, indent=2),
        "candidate_review_audit.json", "application/json", width="stretch",
    )
    if failures:
        st.subheader("Files requiring attention")
        st.dataframe(failures, hide_index=True, width="stretch")


def _methodology_tab():
    st.subheader("How scoring works")
    st.markdown(
        "The final review score is a normalized weighted average of semantic similarity, "
        "required-skill coverage, preferred-skill coverage, and explicitly stated experience. "
        "Every component and its weight are shown in the candidate detail view."
    )
    st.subheader("Safeguards")
    st.markdown(
        "- Blind-review mode removes common direct identifiers before scoring.\n"
        "- Missing criteria reduce a score only when the recruiter explicitly configured them.\n"
        "- Invalid files are isolated so one failure cannot stop the batch.\n"
        "- Ranking processes resume contents in memory; ranking results are not persisted.\n"
        "- Review labels prioritize human attention; they are not employment decisions."
    )
    st.subheader("Known limitations")
    st.markdown(
        "- Scanned PDFs require an external OCR service.\n"
        "- Experience is estimated only from explicit claims such as ‘5 years of experience’.\n"
        "- Skill matching cannot determine proficiency from a mention alone. Evidence must be reviewed.\n"
        "- Blind review reduces direct identifiers but cannot guarantee removal of every demographic proxy.\n"
        "- Semantic similarity can reproduce biases present in model training data. Human review is mandatory."
    )


def render(ctx):
    st.header("Candidate Ranking")
    st.caption("Deterministic, explainable scoring against explicit criteria · results stay in this session")
    blind_review, weights, weight_total = _scoring_settings()
    setup_tab, results_tab, methodology_tab = st.tabs(
        ["1. Configure & analyze", "2. Results", "Methodology & limitations"]
    )
    with setup_tab:
        _configure_tab(ctx, blind_review, weights, weight_total)
    with results_tab:
        _results_tab()
    with methodology_tab:
        _methodology_tab()
