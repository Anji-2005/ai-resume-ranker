"""Fail-closed development authorization adapter for document endpoints.

Replace this adapter with the deployment identity provider before production.
The API deliberately does not treat the user-supplied tenant header as authority.

``RETRIEVAL_AUTH_TOKENS_JSON`` maps a bearer token to either a list of permitted
tenants (legacy form; all roles) or ``{"tenants": [...], "roles": [...]}`` where
roles are ``reader``, ``editor`` (upload/delete), and ``reviewer`` (approve or
reject AI-generated notes).
"""

import hashlib
import hmac
import json
import os
from typing import Annotated

from fastapi import Header, HTTPException, status

from retrieval.authorization import ALL_ROLES, AuthorizationContext


def subject_for_token(token: str) -> str:
    """A stable, non-reversible principal label; no token material is ever stored."""
    return "token-sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def _token_map():
    try:
        value = json.loads(os.getenv("RETRIEVAL_AUTH_TOKENS_JSON", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("RETRIEVAL_AUTH_TOKENS_JSON must be a JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError("RETRIEVAL_AUTH_TOKENS_JSON must be a JSON object")
    return value


def _grant(entry):
    if isinstance(entry, list) and all(isinstance(item, str) for item in entry):
        return frozenset(entry), ALL_ROLES
    if isinstance(entry, dict):
        tenants, roles = entry.get("tenants"), entry.get("roles", sorted(ALL_ROLES))
        if (isinstance(tenants, list) and all(isinstance(item, str) for item in tenants)
                and isinstance(roles, list) and set(roles) <= ALL_ROLES):
            return frozenset(tenants), frozenset(roles)
    return None


def get_retrieval_authorization(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthorizationContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is required")
    token = authorization.removeprefix("Bearer ").strip()
    grant = None
    for known, entry in _token_map().items():
        # Constant-time comparison avoids leaking token prefixes through timing.
        if token and hmac.compare_digest(known.encode("utf-8"), token.encode("utf-8")):
            grant = _grant(entry)
            break
    if grant is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid retrieval credentials")
    tenants, roles = grant
    return AuthorizationContext(subject=subject_for_token(token), permitted_tenant_ids=tenants, roles=roles)
