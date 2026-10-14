"""Synthetic ranking fixtures and a deterministic encoder for score-parity regression tests."""

import numpy as np

from fakes import synthetic_docx


VOCABULARY = [
    "python", "sql", "postgresql", "api", "fastapi", "docker", "aws", "kubernetes", "react",
    "machine", "learning", "data", "pipelines", "backend", "testing", "cloud", "team", "years",
]


class BagOfWordsEncoder:
    """Deterministic stand-in for SentenceTransformer.encode (no model download)."""

    def encode(self, texts, show_progress_bar=False):
        rows = []
        for text in texts:
            words = text.casefold().replace(",", " ").replace(".", " ").split()
            rows.append([float(words.count(term)) + 0.1 for term in VOCABULARY])
        return np.array(rows)


JOB = {
    "description": "Backend engineer building Python APIs with FastAPI, PostgreSQL and Docker on AWS cloud.",
    "required_skills": ["Python", "SQL", "Docker"],
    "preferred_skills": ["AWS", "Kubernetes"],
    "minimum_years": 3.0,
    "weights": {"semantic": 40, "required_skills": 35, "preferred_skills": 10, "experience": 15},
}

RESUMES = [
    ("Alex Example.docx", [
        "Alex Example",
        "Backend engineer with 5 years of experience building Python APIs with FastAPI and PostgreSQL.",
        "Wrote SQL data pipelines, containerized services with Docker and deployed on AWS cloud.",
        "Improved API latency by 40% and led testing for a team of four on a fictional project.",
    ]),
    ("Blair Sample.docx", [
        "Blair Sample",
        "Frontend developer with 2 years of experience in React and testing.",
        "Some Python scripting and SQL reporting for a fictional data team project.",
        "Contact blair.sample@example.com or +1 555 010 0199 for details.",
    ]),
    ("Casey Placeholder.docx", [
        "Casey Placeholder",
        "Platform engineer with 7 years of experience running Kubernetes and Docker on AWS.",
        "Built Python automation and backend tooling; maintained PostgreSQL and SQL backups.",
        "Reduced cloud cost by 25% across a fictional infrastructure project.",
    ]),
]


def resume_files():
    return [(name, synthetic_docx(*paragraphs)) for name, paragraphs in RESUMES]
