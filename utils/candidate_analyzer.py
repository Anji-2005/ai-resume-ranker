import re


def estimate_experience_level(resume_text):
    text = resume_text.lower()

    years = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text)

    if years:
        max_years = max([int(y) for y in years])

        if max_years >= 6:
            return "Senior"
        elif max_years >= 3:
            return "Mid-level"
        elif max_years >= 1:
            return "Junior"

    if "intern" in text or "internship" in text:
        return "Intern"

    return "Entry-level"


def estimate_years_experience(resume_text):
    """Extract the highest explicit experience claim; conservative by design."""
    text = resume_text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)(?:\s+of)?(?:\s+professional)?\s+experience",
        r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    ]
    years = [float(value) for pattern in patterns for value in re.findall(pattern, text)]
    return max(years, default=0.0)


def generate_candidate_summary(score, matched_skills, missing_skills, experience_level):
    matched = ", ".join(matched_skills[:5]) if matched_skills else "limited relevant skills"
    missing = ", ".join(missing_skills[:5]) if missing_skills else "no major required skills"

    return (
        f"This candidate appears to be a {experience_level} profile with a match score of {score}%. "
        f"They show strength in {matched}. "
        f"Key missing areas include {missing}."
    )


def detect_red_flags(resume_text, missing_skills):
    text = resume_text.lower()
    red_flags = []

    if len(text.split()) < 250:
        red_flags.append("Resume appears too short or lacks detail.")

    if not re.search(r"\d+%|\d+x|\d+\+|increased|reduced|improved|optimized|saved", text):
        red_flags.append("Few or no quantified achievements found.")

    if len(missing_skills) >= 5:
        red_flags.append("Several required skills are missing.")

    if "project" not in text and "internship" not in text and "experience" not in text:
        red_flags.append("Limited evidence of projects, internships, or work experience.")

    if not red_flags:
        red_flags.append("No major red flags detected.")

    return red_flags


def generate_improvement_suggestions(missing_skills, red_flags):
    suggestions = []

    if missing_skills:
        suggestions.append(
            "Add or improve evidence for these missing skills: "
            + ", ".join(missing_skills[:5])
        )

    for flag in red_flags:
        if "quantified" in flag.lower():
            suggestions.append("Add measurable achievements such as percentages, numbers, or impact metrics.")
        elif "short" in flag.lower():
            suggestions.append("Expand resume details with more projects, tools, responsibilities, and outcomes.")
        elif "projects" in flag.lower():
            suggestions.append("Add clear project or internship experience with technologies used and business impact.")

    if not suggestions:
        suggestions.append("Resume is reasonably strong. Improve by adding more measurable business impact.")

    return suggestions

def calculate_skill_coverage(job_skills, matched_skills):
    """
    Calculate what percentage of required job skills are present in the resume.
    """

    if not job_skills:
        return 0

    coverage = (len(matched_skills) / len(job_skills)) * 100

    return round(coverage, 2)
