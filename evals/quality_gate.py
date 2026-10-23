"""Regression gate for measured, synthetic offline evaluations."""

import json
from pathlib import Path


def require_quality(
    retrieval_path,
    generation_path,
    offline_path="evals/latest_offline_quality.json",
    investigation_path="evals/latest_investigation_eval.json",
    ingestion_path="evals/latest_ingestion_benchmark.json",
):
    retrieval = json.loads(Path(retrieval_path).read_text())["metrics"]
    generation = json.loads(Path(generation_path).read_text())
    offline = json.loads(Path(offline_path).read_text())
    investigation = json.loads(Path(investigation_path).read_text())["metrics"]
    ingestion = json.loads(Path(ingestion_path).read_text())
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
    for metric in (
        "task_completion_rate",
        "scenario_outcome_accuracy",
        "tool_selection_correctness",
        "citation_validity",
        "unsupported_claim_rejection_rate",
        "authorization_block_rate",
    ):
        if investigation[metric] < 1.0:
            failures.append(f"investigation {metric} fell below 1.0")
    if investigation["authorization_violation_count"] != 0:
        failures.append("investigation authorization violations exceeded 0")
    if ingestion["completed"] != ingestion["documents"] or ingestion["failed"] != 0:
        failures.append("synthetic background ingestion did not complete every document")
    if ingestion["success_rate"] < 1.0:
        failures.append("synthetic background ingestion success rate fell below 1.0")
    if failures:
        raise SystemExit("Quality gate failed: " + "; ".join(failures))
    print("Synthetic quality gate passed")


if __name__ == "__main__":
    require_quality("evals/latest_whole_document_baseline.json", "evals/latest_generation_eval.json")
