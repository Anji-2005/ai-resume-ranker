"""Context construction that keeps model inputs bounded and citations explicit."""

from retrieval.contracts import ContextSource, RetrievalCandidate, RetrievalContext
from utils.privacy import redact_direct_identifiers


DEFAULT_MAX_CHUNKS = 5
DEFAULT_MAX_CHARACTERS = 8_000
SEPARATOR = "\n\n"


def source_header(source_id: str) -> str:
    return f"[source_chunk_id={source_id}]\n"


def build_context(candidates, max_chunks: int = DEFAULT_MAX_CHUNKS,
                  max_characters: int = DEFAULT_MAX_CHARACTERS) -> RetrievalContext:
    """Select, bound, and redact evidence for a model.

    The character budget covers headers and separators, so
    ``len(context_text) <= max_characters`` always holds. Each supplied excerpt is
    recorded in ``sources`` so citation verification can only accept text that was
    actually sent to the model.
    """
    sources: list[ContextSource] = []
    blocks: list[str] = []
    seen: set[str] = set()
    used = 0
    for candidate in candidates:
        if len(sources) >= max_chunks:
            break
        source_id = candidate.source_id
        if source_id in seen or not candidate.content.strip():
            continue
        overhead = len(source_header(source_id)) + (len(SEPARATOR) if blocks else 0)
        remaining = max_characters - used - overhead
        if remaining <= 0:
            break
        raw = candidate.content[:remaining]
        excerpt = redact_direct_identifiers(raw, at_document_start=candidate.source_start == 0)
        # Redaction can lengthen text; never exceed the budget after masking.
        excerpt = excerpt[:remaining]
        if not excerpt.strip():
            continue
        truncated = len(raw) < len(candidate.content)
        sources.append(ContextSource(
            source_id=source_id, candidate=candidate, text=excerpt,
            source_start=candidate.source_start,
            source_end=candidate.source_start + len(raw) if truncated else candidate.source_end,
            truncated=truncated,
        ))
        blocks.append(source_header(source_id) + excerpt)
        seen.add(source_id)
        used += overhead + len(excerpt)
    return RetrievalContext(
        candidates=tuple(source.candidate for source in sources),
        context_text=SEPARATOR.join(blocks),
        source_chunk_ids=tuple(source.source_id for source in sources),
        sources=tuple(sources),
    )


def select_context(candidates: list[RetrievalCandidate], max_chunks: int = DEFAULT_MAX_CHUNKS,
                   max_characters: int = DEFAULT_MAX_CHARACTERS) -> RetrievalContext:
    """Compatibility name for :func:`build_context`."""
    return build_context(candidates, max_chunks=max_chunks, max_characters=max_characters)


def context_from_sources(sources) -> RetrievalContext:
    """Rebuild a context from previously supplied sources without re-truncating or re-reading content."""
    sources = tuple(sources)
    return RetrievalContext(
        candidates=tuple(source.candidate for source in sources),
        context_text=SEPARATOR.join(source_header(source.source_id) + source.text for source in sources),
        source_chunk_ids=tuple(source.source_id for source in sources),
        sources=sources,
    )
