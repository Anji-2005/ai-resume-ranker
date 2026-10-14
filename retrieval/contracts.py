"""Framework-independent retrieval contracts."""

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class MetadataFilters:
    document_ids: frozenset[UUID] = field(default_factory=frozenset)
    document_version_ids: frozenset[UUID] = field(default_factory=frozenset)
    source_keys: frozenset[str] = field(default_factory=frozenset)
    media_types: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RetrievalCandidate:
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    chunk_key: str
    section_id: str
    content: str
    source_start: int
    source_end: int
    page_number: int | None
    source_key: str
    media_type: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fusion_score: float = 0.0
    reranker_score: float | None = None

    @property
    def source_id(self) -> str:
        """The single canonical citation identifier: the database chunk UUID."""
        return str(self.chunk_id)


@dataclass(frozen=True)
class ContextSource:
    """Exactly what was supplied to a model for one chunk.

    ``text`` is the bounded, identifier-redacted excerpt; citations are verified
    against it and nothing else. ``source_start``/``source_end`` are offsets into
    the original normalized document text for the supplied (possibly truncated)
    span; redaction may change the excerpt length but not the provenance span.
    """

    source_id: str
    candidate: RetrievalCandidate
    text: str
    source_start: int
    source_end: int
    truncated: bool = False


@dataclass(frozen=True)
class RetrievalContext:
    candidates: tuple[RetrievalCandidate, ...]
    context_text: str
    source_chunk_ids: tuple[str, ...]
    sources: tuple[ContextSource, ...] = ()

    def supplied_text_by_id(self) -> dict[str, str]:
        """Return only the evidence actually supplied to the model, keyed by canonical ID."""
        return {source.source_id: source.text for source in self.sources}
