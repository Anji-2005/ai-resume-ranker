"""Privacy helpers for blind review and safe display."""

from copy import deepcopy
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


def safe_failure_record(filename, index, error, blind_review):
    """Build a failure record without echoing an identity-bearing filename."""
    label = safe_candidate_label(filename, index, blind_review)
    message = str(error)
    if blind_review and filename:
        message = message.replace(filename, label)
    return {"file": label, "error": message}


def sanitize_audit_records(results, failures, blind_review):
    """Return export-safe copies of candidate and failure records."""
    safe_results = deepcopy(results)
    safe_failures = deepcopy(failures)
    if not blind_review:
        return safe_results, safe_failures

    for index, result in enumerate(safe_results, start=1):
        result["source_filename"] = result.get("candidate") or f"Candidate {index:02d}"
    for index, failure in enumerate(safe_failures, start=1):
        original = str(failure.get("file", ""))
        label = f"Candidate {index:02d}"
        failure["file"] = label
        if original:
            failure["error"] = str(failure.get("error", "")).replace(original, label)
    return safe_results, safe_failures
