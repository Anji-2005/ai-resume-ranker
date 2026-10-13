"""Small deterministic retrieval metrics with no benchmark-library dependency."""

from math import log2
from statistics import mean


def recall_at_k(ranked_chunk_ids, relevant_chunk_ids, k=5):
    return float(bool(set(ranked_chunk_ids[:k]) & set(relevant_chunk_ids)))


def reciprocal_rank(ranked_chunk_ids, relevant_chunk_ids):
    relevant = set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_chunk_ids, relevant_chunk_ids, k=5):
    """Binary-relevance normalized discounted cumulative gain."""
    relevant = set(relevant_chunk_ids)
    dcg = sum(1.0 / log2(rank + 1) for rank, item in enumerate(ranked_chunk_ids[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    if not ideal_hits:
        return 0.0
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def summarize(rows):
    """Rows contain `ranked_chunk_ids`, `relevant_chunk_ids`, and `latency_ms`."""
    latencies = sorted(row["latency_ms"] for row in rows)
    percentile_index = max(0, int(round(0.95 * len(latencies))) - 1)
    return {
        "queries": len(rows),
        "recall_at_5": round(mean(recall_at_k(row["ranked_chunk_ids"], row["relevant_chunk_ids"]) for row in rows), 4),
        "mrr": round(mean(reciprocal_rank(row["ranked_chunk_ids"], row["relevant_chunk_ids"]) for row in rows), 4),
        "ndcg_at_5": round(mean(ndcg_at_k(row["ranked_chunk_ids"], row["relevant_chunk_ids"]) for row in rows), 4),
        "mean_latency_ms": round(mean(row["latency_ms"] for row in rows), 3),
        "p95_latency_ms": round(latencies[percentile_index], 3),
    }
