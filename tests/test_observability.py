import json
import logging

from observability.tracing import StructuredTracer, estimated_cost_usd, redact_trace_value


def test_trace_redaction_removes_direct_identifiers_and_secrets():
    value = "Jane Doe\njane@example.com\nAuthorization: Bearer secret-value\n+91 98765 43210"
    redacted = redact_trace_value(value)
    assert "jane@example.com" not in redacted
    assert "98765" not in redacted
    assert "secret-value" not in redacted


def test_cost_is_unavailable_without_explicit_pricing(monkeypatch):
    monkeypatch.delenv("GENAI_INPUT_USD_PER_MILLION_TOKENS", raising=False)
    monkeypatch.delenv("GENAI_OUTPUT_USD_PER_MILLION_TOKENS", raising=False)
    assert estimated_cost_usd(100, 100) is None


def test_cost_estimate_is_explicitly_configured(monkeypatch):
    monkeypatch.setenv("GENAI_INPUT_USD_PER_MILLION_TOKENS", "1")
    monkeypatch.setenv("GENAI_OUTPUT_USD_PER_MILLION_TOKENS", "2")
    assert estimated_cost_usd(400, 400) == 0.0003


def test_structured_tracer_redacts_metadata_at_the_sink(caplog):
    tracer = StructuredTracer(logging.getLogger("test.trace"))
    with caplog.at_level(logging.INFO, logger="test.trace"):
        tracer.emit_generation(
            operation="qa", prompt_version="v1", model_name="api_key=secret-value",
            latency_ms=1.0, input_characters=1, output_characters=1,
            estimated_cost_usd=None, outcome="success",
        )
    logged = json.loads(caplog.records[-1].message)
    assert "secret-value" not in logged["model_name"]
    assert "SECRET REDACTED" in logged["model_name"]
