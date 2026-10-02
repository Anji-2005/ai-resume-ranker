"""Privacy helpers for blind review and safe display."""

import re


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)


def anonymize_resume_text(text):
    """Remove common direct identifiers before scoring in blind-review mode."""
    cleaned = EMAIL_RE.sub("[EMAIL REDACTED]", text or "")
    cleaned = PHONE_RE.sub("[PHONE REDACTED]", cleaned)
    cleaned = URL_RE.sub("[URL REDACTED]", cleaned)
    lines = cleaned.splitlines()
    if lines and 1 <= len(lines[0].split()) <= 5 and not any(char.isdigit() for char in lines[0]):
        lines[0] = "[NAME REDACTED]"
    return "\n".join(lines)


def safe_candidate_label(filename, index, blind_review):
    return f"Candidate {index:02d}" if blind_review else filename
