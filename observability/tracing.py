"""Structured traces that intentionally omit raw resumes, prompts, and secrets."""

from dataclasses import asdict, dataclass
import json
import logging
import os
import re
from time import perf_counter

from utils.privacy import EMAIL_RE, PHONE_RE, URL_RE, anonymize_resume_text


_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|token)\s*[:=]\s*\S+|"
    r"authorization\s*[:=]\s*bearer\s+\S+|bearer\s+\S+"
)


def redact_trace_value(value):
    """Redact direct identifiers and common secret-bearing fields before logging."""
    text = str(value or "")
    # The resume helper intentionally treats a first line as a probable name.
    # Preserve short operational identifiers (for example, model names) while
    # retaining that stronger heuristic for multiline candidate-derived text.
    if "\n" in text:
        text = anonymize_resume_text(text)
    else:
        text = EMAIL_RE.sub("[EMAIL REDACTED]", text)
        text = PHONE_RE.sub("[PHONE REDACTED]", text)
        text = URL_RE.sub("[URL REDACTED]", text)
    return _SECRET_RE.sub("[SECRET REDACTED]", text)


def estimated_cost_usd(input_chars, output_chars):
    """Return an explicit estimate only when deployment pricing is configured."""
    try:
        input_rate = float(os.getenv("GENAI_INPUT_USD_PER_MILLION_TOKENS", ""))
        output_rate = float(os.getenv("GENAI_OUTPUT_USD_PER_MILLION_TOKENS", ""))
    except ValueError:
        return None
    # Character-to-token conversion is approximate; trace labels it as an estimate.
    return round(((input_chars / 4 * input_rate) + (output_chars / 4 * output_rate)) / 1_000_000, 8)


@dataclass(frozen=True)
class GenerationTrace:
    operation: str
    prompt_version: str
    model_name: str
    latency_ms: float
    input_characters: int
    output_characters: int
    estimated_cost_usd: float | None
    outcome: str
    error_class: str | None = None


class StructuredTracer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("recruitment_intelligence.trace")

    def emit_generation(self, **kwargs):
        # Keep the sink boundary defensive if a future caller accidentally passes
        # candidate-derived values in a nominally metadata-only string field.
        for field in ("operation", "prompt_version", "model_name", "outcome", "error_class"):
            if kwargs.get(field) is not None:
                kwargs[field] = redact_trace_value(kwargs[field])
        trace = GenerationTrace(**kwargs)
        self.logger.info(json.dumps(asdict(trace), sort_keys=True))
        return trace


class TraceTimer:
    def __enter__(self):
        self.started = perf_counter()
        return self

    def __exit__(self, *_):
        self.latency_ms = (perf_counter() - self.started) * 1_000
