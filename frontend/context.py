"""Runtime mode detection and per-session API state for the Streamlit app.

Tenant data is cached only in ``st.session_state`` (per browser session), never
in ``st.cache_data``, which is shared across sessions.
"""

from dataclasses import dataclass
import time

import streamlit as st

from api import schemas
from frontend.api_client import ApiClient, ApiError, config_from_env


STANDALONE = "standalone"
CONNECTED = "connected"
UNAVAILABLE = "unavailable"
STATUS_TTL_SECONDS = 20
MODE_LABELS = {
    STANDALONE: "Standalone mode",
    CONNECTED: "V2 Connected mode",
    UNAVAILABLE: "V2 unavailable",
}


@dataclass
class AppContext:
    mode: str
    client: ApiClient | None = None
    health: schemas.HealthResponse | None = None
    session: schemas.SessionResponse | None = None
    error: str | None = None

    @property
    def connected(self) -> bool:
        return self.mode == CONNECTED

    @property
    def generation(self) -> str:
        return self.session.generation if self.session else "unconfigured"

    def has_role(self, role: str) -> bool:
        return bool(self.session and role in self.session.roles)


def _secrets():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_client():
    # Keyed on nothing user-controlled: configuration is read server-side only.
    config = config_from_env(_secrets())
    return ApiClient(config) if config else None


def resolve_context(force: bool = False) -> AppContext:
    client = _cached_client()
    if client is None:
        return AppContext(STANDALONE)
    cached = st.session_state.get("_v2_status")
    if cached and not force and time.monotonic() - cached["at"] < STATUS_TTL_SECONDS:
        return AppContext(cached["mode"], client, cached["health"], cached["session"], cached["error"])
    health = session = error = None
    try:
        health = client.health()
        session = client.session()
        mode = CONNECTED if health.database == "ok" else UNAVAILABLE
        if mode == UNAVAILABLE:
            error = "The V2 API is running but its database is unavailable."
    except ApiError as exc:
        mode, error = UNAVAILABLE, exc.message
    st.session_state["_v2_status"] = {
        "at": time.monotonic(), "mode": mode, "health": health, "session": session, "error": error,
    }
    return AppContext(mode, client, health, session, error)


def load_documents(ctx: AppContext, refresh: bool = False):
    """Return the tenant's document summaries, cached in this browser session only."""
    if not ctx.connected:
        return []
    if refresh or "_v2_documents" not in st.session_state:
        st.session_state["_v2_documents"] = ctx.client.list_documents().documents
    return st.session_state["_v2_documents"]


def invalidate_documents():
    st.session_state.pop("_v2_documents", None)
    st.session_state.pop("_v2_document_detail", None)


def document_labels(documents) -> dict[str, str]:
    """Map document_id strings to opaque, human-scannable labels."""
    return {
        str(doc.document_id): f"{doc.candidate_ref} · v{doc.current_version_number} · {doc.chunk_count} chunks"
        for doc in documents
    }
