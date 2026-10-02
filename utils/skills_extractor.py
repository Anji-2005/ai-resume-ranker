"""Deterministic skill extraction with aliases and source evidence."""

import re


SKILL_ALIASES = {
    "Python": ["python"], "SQL": ["sql", "structured query language"],
    "Microsoft Excel": ["excel", "microsoft excel", "ms excel"],
    "Power BI": ["power bi", "powerbi"], "Tableau": ["tableau"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Pandas": ["pandas"], "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch", "torch"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data visualisation"],
    "Statistics": ["statistics", "statistical analysis"],
    "FastAPI": ["fastapi"], "Flask": ["flask"], "Django": ["django"],
    "Streamlit": ["streamlit"], "HTML": ["html"], "CSS": ["css"],
    "JavaScript": ["javascript", "java script"], "TypeScript": ["typescript"],
    "React": ["react", "react.js", "reactjs"], "Git": ["git"],
    "GitHub": ["github"], "Docker": ["docker", "containerization", "containerisation"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud platform"],
    "Financial Analysis": ["financial analysis"],
    "Risk Analysis": ["risk analysis", "risk analytics"],
    "Fraud Detection": ["fraud detection"], "Consulting": ["consulting"],
    "Business Analytics": ["business analytics"],
    "Project Management": ["project management"], "Agile": ["agile", "scrum"],
    "REST APIs": ["rest api", "restful api", "rest apis"], "Linux": ["linux"],
}


def _alias_pattern(alias):
    # Alphanumeric boundaries prevent matches such as SQL inside NoSQL.
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", re.I)


def canonicalize_skill(skill):
    candidate = skill.strip().lower()
    if not candidate:
        return ""
    for canonical, aliases in SKILL_ALIASES.items():
        if candidate == canonical.lower() or candidate in {item.lower() for item in aliases}:
            return canonical
    return skill.strip()


def parse_skill_list(value):
    """Parse comma/newline-separated skills and remove duplicates."""
    found = []
    for raw in re.split(r"[,;\n]", value or ""):
        skill = canonicalize_skill(raw)
        if skill and skill.casefold() not in {item.casefold() for item in found}:
            found.append(skill)
    return found


def extract_skills(text):
    text = text or ""
    return [canonical for canonical, aliases in SKILL_ALIASES.items()
            if any(_alias_pattern(alias).search(text) for alias in aliases)]


def _contains_skill(text, skill):
    canonical = canonicalize_skill(skill)
    aliases = SKILL_ALIASES.get(canonical, [canonical])
    return any(_alias_pattern(alias).search(text or "") for alias in aliases)


def find_skill_evidence(text, skills, max_sentences=2):
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text or "") if part.strip()]
    evidence = {}
    for skill in skills:
        evidence[skill] = [chunk for chunk in chunks if _contains_skill(chunk, skill)][:max_sentences]
    return evidence


def compare_skills(job_description, resume_text, required_skills=None):
    job_skills = required_skills or extract_skills(job_description)
    resume_skills = extract_skills(resume_text)
    matched = [skill for skill in job_skills if _contains_skill(resume_text, skill)]
    missing = [skill for skill in job_skills if skill not in matched]
    return job_skills, resume_skills, matched, missing
