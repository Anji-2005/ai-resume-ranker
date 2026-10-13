"""Evaluate the original whole-document cosine matcher on the synthetic set."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from evals.retrieval_metrics import summarize
from utils.matcher import calculate_match_score, get_model


def evaluate(dataset_path: Path):
    dataset = json.loads(dataset_path.read_text())
    model = get_model()  # exact baseline model/loader used by the Streamlit matcher
    rows = []
    for query in dataset["queries"]:
        started = perf_counter()
        ranked = sorted(
            dataset["documents"],
            key=lambda document: calculate_match_score(query["text"], document["text"], model=model),
            reverse=True,
        )
        rows.append({
            "query_id": query["id"], "ranked_chunk_ids": [item["id"] for item in ranked],
            "relevant_chunk_ids": query["relevant_source_keys"],
            "latency_ms": (perf_counter() - started) * 1_000,
        })
    return {
        "dataset_version": dataset["dataset_version"],
        "method": "whole_document_cosine_baseline", "metrics": summarize(rows), "per_query": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_retrieval.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/latest_whole_document_baseline.json"))
    args = parser.parse_args()
    result = evaluate(args.dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
