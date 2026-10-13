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


@dataclass(frozen=True)
class RetrievalContext:
    candidates: tuple[RetrievalCandidate, ...]
    context_text: str
    source_chunk_ids: tuple[str, ...]
