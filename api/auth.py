"""Fail-closed development authorization adapter for retrieval endpoints.

Replace this adapter with the deployment identity provider before production.
The API deliberately does not treat the user-supplied tenant header as authority.
"""

import json
import os
from typing import Annotated

from fastapi import Header, HTTPException, status

from retrieval.authorization import AuthorizationContext


def get_retrieval_authorization(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthorizationContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is required")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        token_map = json.loads(os.getenv("RETRIEVAL_AUTH_TOKENS_JSON", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("RETRIEVAL_AUTH_TOKENS_JSON must be a JSON object") from exc
    tenants = token_map.get(token)
    if not isinstance(tenants, list) or not all(isinstance(item, str) for item in tenants):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid retrieval credentials")
    return AuthorizationContext(subject=f"token:{token[:8]}", permitted_tenant_ids=frozenset(tenants))
