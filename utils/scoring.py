"""Explainable candidate scoring."""

from utils.skills_extractor import compare_skills


DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "required_skills": 0.35,
    "preferred_skills": 0.10,
    "experience": 0.15,
}


def percentage(part, whole, empty_default=100.0):
    if not whole:
        return float(empty_default)
    return round(100 * len(part) / len(whole), 2)


def experience_score(candidate_years, minimum_years):
    if minimum_years <= 0:
        return 100.0
    return round(min(1.0, max(0.0, candidate_years / minimum_years)) * 100, 2)


def normalize_weights(weights=None):
    selected = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total = sum(max(0.0, float(value)) for value in selected.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: max(0.0, float(value)) / total for key, value in selected.items()}


def calculate_score_breakdown(semantic_score, resume_text, required_skills,
                              preferred_skills, candidate_years, minimum_years,
                              weights=None):
    _, _, matched_required, missing_required = compare_skills("", resume_text, required_skills)
    _, _, matched_preferred, missing_preferred = compare_skills("", resume_text, preferred_skills)
    components = {
        "semantic": round(float(semantic_score), 2),
        "required_skills": percentage(matched_required, required_skills),
        "preferred_skills": percentage(matched_preferred, preferred_skills),
        "experience": experience_score(candidate_years, minimum_years),
    }
    active_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    # An unconfigured criterion must not award free points or affect the score.
    if not preferred_skills:
        active_weights["preferred_skills"] = 0
    if minimum_years <= 0:
        active_weights["experience"] = 0
    normalized = normalize_weights(active_weights)
    final_score = round(sum(components[key] * normalized[key] for key in components), 2)
    return {
        "final_score": final_score, "components": components, "weights": normalized,
        "matched_required": matched_required, "missing_required": missing_required,
        "matched_preferred": matched_preferred, "missing_preferred": missing_preferred,
    }


def review_recommendation(score, required_coverage):
    """Return a review priority, never an automated employment decision."""
    if score >= 80 and required_coverage >= 75:
        return "Priority review"
    if score >= 60 and required_coverage >= 50:
        return "Potential match"
    return "Manual review required"
