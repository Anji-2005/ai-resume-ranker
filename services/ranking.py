"""Deterministic candidate analysis and exports, independent of Streamlit.

Moved verbatim from ``app.py`` so the ranking contract can be regression-tested
without a UI. Score formulas live in ``utils/``; this module only orchestrates
them. Optional Gemini enrichment may replace narrative text but never a score.
"""

from io import BytesIO

import pandas as pd

from utils.candidate_analyzer import (
    detect_red_flags,
    estimate_experience_level,
    estimate_years_experience,
    generate_candidate_summary,
    generate_improvement_suggestions,
)
from utils.genai_analyzer import enrich_candidate, gemini_model_name
from utils.matcher import calculate_match_score
from utils.pdf_parser import ResumeParseError, extract_text_from_bytes
from utils.privacy import anonymize_resume_text, safe_candidate_label
from utils.scoring import calculate_score_breakdown, review_recommendation
from utils.skills_extractor import find_skill_evidence


HUMAN_DECISIONS = ["Unreviewed", "Shortlist", "Hold for review", "Not selected"]


def analyze_resume_bytes(filename, data, index, job, model, genai_api_key=""):
    raw_text = extract_text_from_bytes(filename, data)
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
    label = safe_candidate_label(filename, index, job["blind_review"])
    result = {
        "candidate": label,
        "source_filename": label if job["blind_review"] else filename,
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


def rank_results(results):
    """Sort by score (stable for ties, preserving upload order) and assign 1-based ranks."""
    results.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


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
