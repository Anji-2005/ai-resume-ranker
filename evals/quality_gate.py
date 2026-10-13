"""Regression gate for measured, synthetic offline evaluations."""

import json
from pathlib import Path


def require_quality(retrieval_path, generation_path, offline_path="evals/latest_offline_quality.json"):
    retrieval = json.loads(Path(retrieval_path).read_text())["metrics"]
    generation = json.loads(Path(generation_path).read_text())
    offline = json.loads(Path(offline_path).read_text())
    failures = []
    if retrieval["recall_at_5"] < 1.0:
        failures.append("whole-document baseline Recall@5 fell below 1.0")
    if retrieval["mrr"] < 0.8:
        failures.append("whole-document baseline MRR fell below 0.8")
    if generation["accuracy"] < 1.0:
        failures.append("citation verification evaluation failed")
    if retrieval["ndcg_at_5"] < 0.8:
        failures.append("whole-document baseline nDCG@5 fell below 0.8")
    for metric in ("parser_fidelity", "redaction_recall", "score_parity", "structured_output_accuracy"):
        if offline[metric] < 1.0:
            failures.append(f"{metric} fell below 1.0")
    if offline["sensitive_content_acceptance_rate"] > 0.0:
        failures.append("sensitive content acceptance rate exceeded 0.0")
    if failures:
        raise SystemExit("Quality gate failed: " + "; ".join(failures))
    print("Synthetic quality gate passed")


if __name__ == "__main__":
    require_quality("evals/latest_whole_document_baseline.json", "evals/latest_generation_eval.json")
