"""Context construction that keeps model inputs bounded and citations explicit."""

from retrieval.contracts import RetrievalCandidate, RetrievalContext


def select_context(candidates: list[RetrievalCandidate], max_chunks: int = 5, max_characters: int = 8_000):
    selected, excerpts, seen, used = [], [], set(), 0
    for candidate in candidates:
        if candidate.chunk_id in seen or len(selected) >= max_chunks:
            continue
        remaining = max_characters - used
        if remaining <= 0:
            break
        content = candidate.content[:remaining]
        if not content:
            continue
        selected.append(candidate)
        excerpts.append(content)
        seen.add(candidate.chunk_id)
        used += len(content)
    text = "\n\n".join(
        f"[source_chunk_id={item.chunk_key}]\n{excerpt}" for item, excerpt in zip(selected, excerpts, strict=True)
    )
    return RetrievalContext(tuple(selected), text, tuple(str(item.chunk_id) for item in selected))
