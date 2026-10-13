"""Tenant authorization is checked before retrieval can reach a repository."""

from dataclasses import dataclass


class RetrievalAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class AuthorizationContext:
    subject: str
    permitted_tenant_ids: frozenset[str]

    def require_tenant(self, tenant_id: str) -> None:
        if tenant_id not in self.permitted_tenant_ids:
            raise RetrievalAuthorizationError("subject is not authorized for this tenant")
