"""Offline citation-verification regression evaluation; no provider is called."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from generation.citations import CitationVerificationError, verify_answer
from generation.schemas import GroundedAnswer
from retrieval.context import build_context
from retrieval.contracts import RetrievalCandidate


def evaluate(dataset_path):
    dataset = json.loads(dataset_path.read_text())
    rows = []
    for case in dataset["cases"]:
        candidate = RetrievalCandidate(
            document_id=uuid4(), document_version_id=uuid4(), chunk_id=uuid4(), chunk_key="eval-chunk",
            section_id="eval", content=case["source"], source_start=0, source_end=len(case["source"]),
            page_number=None, source_key="eval", media_type="text/plain",
        )
        # Optional fields exercise canonical-ID and context-budget boundaries.
        citation_id = candidate.chunk_key if case.get("citation_id") == "chunk_key" else candidate.source_id
        answer = GroundedAnswer.model_validate({"claims": [{"text": case["claim"], "citations": [{
            "source_chunk_id": citation_id, "quote": case["quote"]
        }]}]})
        context = build_context([candidate], max_characters=case.get("max_characters", 8_000))
        started = perf_counter()
        try:
            verify_answer(answer, context)
            outcome = "accept"
        except CitationVerificationError:
            outcome = "reject"
        rows.append({"id": case["id"], "expected": case["expected"], "outcome": outcome,
                     "latency_ms": (perf_counter() - started) * 1_000})
    return {
        "dataset_version": dataset["dataset_version"],
        "method": "offline_citation_verification",
        "cases": len(rows),
        "accuracy": round(sum(row["expected"] == row["outcome"] for row in rows) / len(rows), 4),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 3),
        "provider_cost_usd": None,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/generation_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/latest_generation_eval.json"))
    args = parser.parse_args()
    result = evaluate(args.dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
