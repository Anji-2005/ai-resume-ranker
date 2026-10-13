"""AI Resume Ranker — explainable, human-in-the-loop candidate review."""

from datetime import datetime, timezone
from io import BytesIO
import json
import os

import pandas as pd
import streamlit as st

from utils.candidate_analyzer import (
    detect_red_flags,
    estimate_experience_level,
    estimate_years_experience,
    generate_candidate_summary,
    generate_improvement_suggestions,
)
from utils.matcher import MODEL_NAME, calculate_match_score, get_model
from utils.genai_analyzer import clear_enrichment_cache, enrich_candidate, gemini_model_name
from utils.pdf_parser import ResumeParseError, extract_text_from_bytes
from utils.privacy import (
    anonymize_resume_text,
    safe_candidate_label,
    safe_failure_record,
    sanitize_audit_records,
)
from utils.scoring import calculate_score_breakdown, review_recommendation
from utils.skills_extractor import extract_skills, find_skill_evidence, parse_skill_list


st.set_page_config(page_title="Resume Review Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return get_model()


def build_export_frame(results):
    rows = []
    for item in results:
        rows.append({
            "Rank": item["rank"],
            "Candidate": item["candidate"],
            "Review Score": item["score"],
            "Review Priority": item["recommendation"],
            "Semantic Similarity": item["components"]["semantic"],
            "Required Skill Coverage": item["components"]["required_skills"],
            "Preferred Skill Coverage": item["components"]["preferred_skills"],
            "Experience Match": item["components"]["experience"],
            "Estimated Experience (years)": item["years"],
            "Matched Required Skills": ", ".join(item["matched_required"]),
            "Missing Required Skills": ", ".join(item["missing_required"]),
            "Matched Preferred Skills": ", ".join(item["matched_preferred"]),
            "Human Review Outcome": item.get("human_decision", "Unreviewed"),
            "Reviewer Rationale": item.get("human_reason", ""),
        })
    return pd.DataFrame(rows)


def excel_bytes(frame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Candidate ranking", index=False)
    return output.getvalue()


def get_gemini_api_key():
    """Read optional Gemini credentials without requiring a secrets file."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("GEMINI_API_KEY", "")).strip()


def analyze_resume(uploaded_file, index, job, model, genai_api_key=""):
    data = uploaded_file.getvalue()
    raw_text = extract_text_from_bytes(uploaded_file.name, data)
    if len(raw_text.split()) < 20:
        raise ResumeParseError(
            "Very little text was extracted. This may be a scanned PDF; OCR is required."
        )

    scoring_text = anonymize_resume_text(raw_text) if job["blind_review"] else raw_text
    semantic = calculate_match_score(job["description"], scoring_text, model=model)
    years = estimate_years_experience(scoring_text)
    breakdown = calculate_score_breakdown(
        semantic, scoring_text, job["required_skills"], job["preferred_skills"],
        years, job["minimum_years"], job["weights"],
    )
    matched_all = breakdown["matched_required"] + breakdown["matched_preferred"]
    evidence = find_skill_evidence(scoring_text, matched_all)
    flags = detect_red_flags(scoring_text, breakdown["missing_required"])
    label = safe_candidate_label(uploaded_file.name, index, job["blind_review"])
    result = {
        "candidate": label,
        "source_filename": label if job["blind_review"] else uploaded_file.name,
        "score": breakdown["final_score"],
        "components": breakdown["components"],
        "weights": breakdown["weights"],
        "matched_required": breakdown["matched_required"],
        "missing_required": breakdown["missing_required"],
        "matched_preferred": breakdown["matched_preferred"],
        "missing_preferred": breakdown["missing_preferred"],
        "years": years,
        "experience_level": estimate_experience_level(scoring_text),
        "evidence": evidence,
        "red_flags": flags,
        "suggestions": generate_improvement_suggestions(breakdown["missing_required"], flags),
        "human_decision": "Unreviewed",
        "human_reason": "",
    }
    result["recommendation"] = review_recommendation(
        result["score"], result["components"]["required_skills"]
    )
    result["summary"] = generate_candidate_summary(
        result["score"], matched_all, result["missing_required"], result["experience_level"]
    )
    result["genai"] = {
        "status": "disabled",
        "message": "Gemini enrichment is disabled because no API key is configured.",
        "model": None,
        "interview_questions": [],
        "requirement_evidence": [],
    }
    if genai_api_key:
        try:
            enrichment = enrich_candidate(scoring_text, job, result, genai_api_key)
            result["summary"] = enrichment["candidate_summary"]
            result["suggestions"] = enrichment["improvement_suggestions"]
            result["genai"] = {
                "status": "success",
                "message": "Gemini enrichment generated; verify all AI-generated content.",
                "model": gemini_model_name(),
                "interview_questions": enrichment["interview_questions"],
                "requirement_evidence": enrichment["requirement_evidence"],
            }
        except Exception:
            result["genai"] = {
                "status": "unavailable",
                "message": "Gemini enrichment was unavailable; rule-based fallbacks are shown.",
                "model": gemini_model_name(),
                "interview_questions": [],
                "requirement_evidence": [],
            }
    return result


def render_score_breakdown(candidate):
    labels = {
        "semantic": "Semantic similarity",
        "required_skills": "Required skills",
        "preferred_skills": "Preferred skills",
        "experience": "Experience",
    }
    columns = st.columns(4)
    for column, key in zip(columns, labels):
        with column:
            st.metric(labels[key], f"{candidate['components'][key]:.1f}%")
            st.caption(f"Weight: {candidate['weights'][key] * 100:.0f}%")


st.title("📄 Resume Review Assistant")
st.caption("Explainable candidate screening with human-owned decisions")
st.info(
    "This tool prioritizes resumes for human review. Scores are decision-support signals, "
    "not hiring decisions, and should never be used as the sole basis for employment actions."
)

with st.sidebar:
    st.header("Review controls")
    blind_review = st.toggle(
        "Blind-review mode", value=True,
        help="Redacts common direct identifiers before scoring and replaces filenames with candidate IDs.",
    )
    st.subheader("Score weights")
    semantic_weight = st.slider("Semantic match", 0, 100, 40, 5)
    required_weight = st.slider("Required skills", 0, 100, 35, 5)
    preferred_weight = st.slider("Preferred skills", 0, 100, 10, 5)
    experience_weight = st.slider("Experience", 0, 100, 15, 5)
    weight_total = semantic_weight + required_weight + preferred_weight + experience_weight
    if weight_total == 0:
        st.error("At least one scoring weight must be greater than zero.")
    elif weight_total != 100:
        st.caption(f"Weights total {weight_total}%; they will be normalized automatically.")
    st.divider()
    st.caption("Uploads are processed in memory. This version does not save resume contents.")
    if st.button("Delete session results", use_container_width=True):
        for key in ["ranking_results", "ranking_failures", "job_config", "audit_timestamp"]:
            st.session_state.pop(key, None)
        clear_enrichment_cache()
        st.success("Candidate results were removed from this session.")

setup_tab, results_tab, methodology_tab = st.tabs(
    ["1. Configure & analyze", "2. Results", "Methodology & limitations"]
)

with setup_tab:
    st.subheader("Define the role")
    job_description = st.text_area(
        "Job description *", height=220,
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
            "Preferred skills", height=100,
            help="Optional skills that improve a score but are not mandatory.",
        )
    minimum_years = st.number_input(
        "Minimum relevant experience (years)", min_value=0.0, max_value=40.0,
        value=0.0, step=0.5,
    )
    uploaded_resumes = st.file_uploader(
        "Upload resumes *", type=["pdf", "docx"], accept_multiple_files=True,
        help="PDF or DOCX, maximum 10 MB each. Scanned PDFs are detected but require OCR.",
    )
    analyze_button = st.button(
        "Analyze candidates", type="primary", disabled=(weight_total == 0),
        use_container_width=True,
    )

    if analyze_button:
        required_skills = parse_skill_list(required_input)
        preferred_skills = [skill for skill in parse_skill_list(preferred_input)
                            if skill not in required_skills]
        if not job_description.strip():
            st.error("Enter a job description before analyzing resumes.")
        elif not uploaded_resumes:
            st.error("Upload at least one resume.")
        elif not required_skills:
            st.error("Add at least one required skill so the ranking has explicit criteria.")
        else:
            job = {
                "description": job_description.strip(),
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "minimum_years": float(minimum_years),
                "blind_review": blind_review,
                "weights": {
                    "semantic": semantic_weight,
                    "required_skills": required_weight,
                    "preferred_skills": preferred_weight,
                    "experience": experience_weight,
                },
            }
            results, failures = [], []
            genai_api_key = get_gemini_api_key()
            progress = st.progress(0, text="Loading the language model…")
            try:
                model = load_embedding_model()
            except Exception as exc:
                progress.empty()
                st.error(f"The language model could not be loaded: {exc}")
                model = None
            if model is not None:
                for index, resume in enumerate(uploaded_resumes, start=1):
                    display_name = safe_candidate_label(resume.name, index, job["blind_review"])
                    progress.progress(
                        (index - 1) / len(uploaded_resumes),
                        text=f"Analyzing {index} of {len(uploaded_resumes)}: {display_name}",
                    )
                    try:
                        results.append(analyze_resume(
                            resume, index, job, model, genai_api_key=genai_api_key
                        ))
                    except ResumeParseError as exc:
                        failures.append(safe_failure_record(
                            resume.name, index, exc, job["blind_review"]
                        ))
                    except Exception as exc:
                        failures.append(safe_failure_record(
                            resume.name,
                            index,
                            f"Unexpected analysis error: {exc}",
                            job["blind_review"],
                        ))
                progress.progress(1.0, text="Analysis complete")
                results.sort(key=lambda item: item["score"], reverse=True)
                for rank, item in enumerate(results, start=1):
                    item["rank"] = rank
                st.session_state["ranking_results"] = results
                st.session_state["ranking_failures"] = failures
                st.session_state["job_config"] = job
                st.session_state["audit_timestamp"] = datetime.now(timezone.utc).isoformat()
                if results:
                    st.success(f"Analyzed {len(results)} candidate(s). Open the Results tab.")
                if failures:
                    st.warning(f"Skipped {len(failures)} file(s). See Results for details.")

with results_tab:
    results = st.session_state.get("ranking_results", [])
    failures = st.session_state.get("ranking_failures", [])
    if not results:
        st.info("Configure the role and analyze resumes to see results.")
    else:
        st.subheader("Candidate ranking")
        filter_col, skill_col = st.columns(2)
        with filter_col:
            minimum_score = st.slider("Minimum review score", 0, 100, 0)
        with skill_col:
            job_skills = st.session_state["job_config"]["required_skills"]
            skill_filter = st.multiselect("Must contain these required skills", job_skills)
        visible = [item for item in results if item["score"] >= minimum_score and
                   all(skill in item["matched_required"] for skill in skill_filter)]
        frame = build_export_frame(visible)
        if visible:
            best = visible[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Highest-ranked", best["candidate"])
            m2.metric("Review score", f"{best['score']:.1f}%")
            m3.metric("Review priority", best["recommendation"])
            chart = frame.set_index("Candidate")[["Review Score"]]
            st.bar_chart(chart)
            st.dataframe(frame, hide_index=True, use_container_width=True)
        else:
            st.warning("No candidates match the active filters.")

        st.subheader("Candidate details")
        for item in visible:
            with st.expander(
                f"#{item['rank']} · {item['candidate']} · {item['score']:.1f}% · {item['recommendation']}"
            ):
                render_score_breakdown(item)
                genai_result = item.get("genai", {})
                if genai_result.get("status") == "success":
                    st.caption("AI-generated candidate summary · Verify against resume evidence")
                else:
                    st.caption(genai_result.get(
                        "message", "Rule-based summary shown; GenAI enrichment was not used."
                    ))
                st.write(item["summary"])
                left, right = st.columns(2)
                with left:
                    st.markdown("**Matched required skills**")
                    st.write(", ".join(item["matched_required"]) or "None")
                    st.markdown("**Matched preferred skills**")
                    st.write(", ".join(item["matched_preferred"]) or "None")
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
                    "Outcome",
                    ["Unreviewed", "Shortlist", "Hold for review", "Not selected"],
                    index=["Unreviewed", "Shortlist", "Hold for review", "Not selected"].index(
                        item.get("human_decision", "Unreviewed")
                    ),
                    key=f"decision_{item['rank']}_{item['candidate']}",
                )
                rationale = st.text_input(
                    "Reviewer rationale",
                    value=item.get("human_reason", ""),
                    placeholder="Record evidence-based reasoning; avoid protected characteristics.",
                    key=f"reason_{item['rank']}_{item['candidate']}",
                )
                if st.button("Save human review", key=f"save_{item['rank']}_{item['candidate']}"):
                    item["human_decision"] = decision
                    item["human_reason"] = rationale.strip()
                    st.success("Human review record saved for this session and future exports.")

        if len(visible) >= 2:
            st.subheader("Side-by-side comparison")
            selected = st.multiselect(
                "Choose 2–4 candidates", [item["candidate"] for item in visible],
                max_selections=4,
            )
            if len(selected) >= 2:
                comparison = frame[frame["Candidate"].isin(selected)].set_index("Candidate").T
                st.dataframe(comparison, use_container_width=True)

        st.subheader("Export")
        export_frame = build_export_frame(results)
        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Download CSV", export_frame.to_csv(index=False).encode("utf-8"),
            "candidate_review.csv", "text/csv", use_container_width=True,
        )
        try:
            workbook = excel_bytes(export_frame)
            download_cols[1].download_button(
                "Download Excel", workbook, "candidate_review.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            download_cols[1].caption("Install openpyxl to enable Excel export.")
        audit_results, audit_failures = sanitize_audit_records(
            results,
            failures,
            bool(st.session_state.get("job_config", {}).get("blind_review", False)),
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
            "candidate_review_audit.json", "application/json", use_container_width=True,
        )
        if failures:
            st.subheader("Files requiring attention")
            st.dataframe(pd.DataFrame(failures), hide_index=True, use_container_width=True)

with methodology_tab:
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
        "- Resume contents are processed in memory and are not persisted by this application.\n"
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

st.divider()
st.caption(f"Model: {MODEL_NAME} · Local decision-support prototype · No automated hiring decisions")
