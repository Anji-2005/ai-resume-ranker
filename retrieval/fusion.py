"""Rank-only fusion keeps PostgreSQL FTS and vector scores incomparable by design."""

from collections import defaultdict

from retrieval.contracts import RetrievalCandidate


def reciprocal_rank_fusion(dense: list[RetrievalCandidate], lexical: list[RetrievalCandidate], k: int = 60):
    """Fuse candidate lists without claiming a common score scale or BM25."""
    merged: dict[object, RetrievalCandidate] = {}
    scores = defaultdict(float)
    for results, rank_name in ((dense, "dense_rank"), (lexical, "lexical_rank")):
        for rank, candidate in enumerate(results, start=1):
            existing = merged.get(candidate.chunk_id, candidate)
            values = dict(candidate.__dict__)
            values[rank_name] = rank
            if existing is not candidate:
                values["dense_rank"] = existing.dense_rank or values["dense_rank"]
                values["lexical_rank"] = existing.lexical_rank or values["lexical_rank"]
            merged[candidate.chunk_id] = RetrievalCandidate(**values)
            scores[candidate.chunk_id] += 1 / (k + rank)
    fused = []
    for chunk_id, candidate in merged.items():
        fused.append(RetrievalCandidate(**{**candidate.__dict__, "fusion_score": scores[chunk_id]}))
    return sorted(fused, key=lambda item: (-item.fusion_score, str(item.chunk_id)))
