"""Per-tenant generation policy gate.

Candidate text is sent to an external model only when the server has a provider
key *and* the tenant has an explicit policy entry documenting provider, purpose,
and retention. Missing or malformed configuration fails closed.
"""

from dataclasses import dataclass
import json
import os


GENERATION_ENABLED = "enabled"
GENERATION_UNCONFIGURED = "unconfigured"
GENERATION_POLICY_DENIED = "policy_denied"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class GenerationPolicy:
    provider: str
    purpose: str
    retention: str


def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def tenant_policy(tenant_id: str) -> GenerationPolicy | None:
    try:
        policies = json.loads(os.getenv("GENERATION_TENANT_POLICY_JSON", "{}") or "{}")
    except json.JSONDecodeError:
        return None
    entry = policies.get(tenant_id) if isinstance(policies, dict) else None
    if not isinstance(entry, dict):
        return None
    values = {key: str(entry.get(key, "")).strip() for key in ("provider", "purpose", "retention")}
    if values["provider"] != "gemini" or not values["purpose"] or not values["retention"]:
        return None
    return GenerationPolicy(**values)


def generation_status(tenant_id: str) -> str:
    if not gemini_api_key():
        return GENERATION_UNCONFIGURED
    if tenant_policy(tenant_id) is None:
        return GENERATION_POLICY_DENIED
    return GENERATION_ENABLED
