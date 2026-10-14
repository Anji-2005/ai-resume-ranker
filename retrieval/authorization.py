"""Tenant authorization is checked before retrieval can reach a repository."""

from dataclasses import dataclass, field


ROLE_READER = "reader"
ROLE_EDITOR = "editor"
ROLE_REVIEWER = "reviewer"
ALL_ROLES = frozenset({ROLE_READER, ROLE_EDITOR, ROLE_REVIEWER})


class RetrievalAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class AuthorizationContext:
    subject: str
    permitted_tenant_ids: frozenset[str]
    # Legacy token mappings grant every role; role-scoped mappings narrow this.
    roles: frozenset[str] = field(default=ALL_ROLES)

    def require_tenant(self, tenant_id: str) -> None:
        if tenant_id not in self.permitted_tenant_ids:
            raise RetrievalAuthorizationError("subject is not authorized for this tenant")

    def require_role(self, role: str) -> None:
        if role not in self.roles:
            raise RetrievalAuthorizationError(f"subject lacks the {role} role")
