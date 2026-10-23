"""Deterministic, privacy-safe evaluation of the bounded V3 investigation agent.

This runs the real LangGraph, tool dispatcher, context builder, redaction, report
validator, and citation verifier with an in-memory repository and scripted fake
tool-calling provider. It does not measure live Gemini quality or billed cost.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, UUID, uuid5

from langgraph.checkpoint.memory import InMemorySaver

from investigations.schemas import AgentLimits, ToolInvocation, ToolModelResponse
from investigations.tools import InvestigationTools
from investigations.workflow import InvestigationAgent, initial_state
from retrieval.context import build_context
from retrieval.contracts import RetrievalCandidate


TENANT = "synthetic-evaluation-tenant"


@dataclass
class Version:
    version_id: UUID
    version_number: int = 1
    is_current: bool = True
    media_type: str = "application/pdf"


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_key: str
    section_id: str
    content: str
    source_start: int
    source_end: int
    page_number: int | None = 1


class EvaluationRepository:
    def __init__(self, case):
        self.documents = {}
        self.chunks = {}
        for index, content in enumerate(case["documents"], start=1):
            document_id = uuid5(NAMESPACE_URL, f"{case['id']}:document:{index}")
            version_id = uuid5(NAMESPACE_URL, f"{case['id']}:version:{index}")
            chunk_id = uuid5(NAMESPACE_URL, f"{case['id']}:chunk:{index}")
            self.documents[document_id] = Version(version_id)
            self.chunks[document_id] = [Chunk(
                id=chunk_id,
                document_id=document_id,
                document_version_id=version_id,
                chunk_key=f"chunk-{index}",
                section_id="experience",
                content=content,
                source_start=0,
                source_end=len(content),
            )]

    def existing_document_ids(self, tenant_id, document_ids):
        return set(document_ids) & set(self.documents) if tenant_id == TENANT else set()

    def list_versions(self, tenant_id, document_id):
        return [self.documents[document_id]] if tenant_id == TENANT and document_id in self.documents else []

    def get_version_chunks(self, tenant_id, document_id, version_number, limit=500):
        if tenant_id != TENANT or document_id not in self.documents or version_number != 1:
            return []
        return self.chunks[document_id][:limit]


class EvaluationPipeline:
    def __init__(self, repository):
        self.repository = repository

    def retrieve(self, authorization, tenant_id, query, *, filters, limit, max_characters, **_kwargs):
        authorization.require_tenant(tenant_id)
        if query == "__missing__":
            return build_context([])
        candidates = []
        for document_id in sorted(filters.document_ids, key=str):
            if document_id not in self.repository.documents:
                continue
            row = self.repository.chunks[document_id][0]
            candidates.append(RetrievalCandidate(
                document_id=row.document_id,
                document_version_id=row.document_version_id,
                chunk_id=row.id,
                chunk_key=row.chunk_key,
                section_id=row.section_id,
                content=row.content,
                source_start=row.source_start,
                source_end=row.source_end,
                page_number=row.page_number,
                source_key="synthetic-evaluation",
                media_type="application/pdf",
            ))
        return build_context(candidates, max_chunks=limit, max_characters=max_characters)


def _source_rows(payload):
    rows = []
    for result in payload["tool_results"]:
        rows.extend(result["output"].get("sources", []))
    unique = {}
    for row in rows:
        unique[row["source_chunk_id"]] = row
    return list(unique.values())


def _citation(row, *, invented=False):
    quote = "Invented unsupported quote" if invented else row["excerpt"].strip()[:40]
    return {"source_chunk_id": row["source_chunk_id"], "quote": quote}


def _gap(action_id, topic="Requested evidence"):
    return {
        "topic": topic,
        "explanation": "Insufficient evidence was found in the authorized scope.",
        "checked_action_ids": [action_id],
    }


class ScenarioProvider:
    model_name = "deterministic-investigation-eval-v1"

    def __init__(self, case, repository):
        self.case = case
        self.repository = repository
        self.prompts = []

    def invoke_tools(self, prompt, tools):
        self.prompts.append(prompt)
        payload = json.loads(prompt.split("UNTRUSTED_INVESTIGATION_DATA_JSON:\n", 1)[1])
        kind = self.case["kind"]
        results = payload["tool_results"]
        if kind == "unavailable_tool":
            return {
                "calls": [{"id": "eval-unavailable", "name": "delete_document", "arguments": {}}],
                "input_tokens": 40, "output_tokens": 10, "cost_usd": 0.001,
            }
        if kind == "invalid_arguments":
            invocation = ToolInvocation(
                id="eval-invalid", name="search_evidence", arguments={"query": ""},
            )
        elif kind == "unauthorized":
            invocation = ToolInvocation(
                id="eval-unauthorized", name="inspect_document",
                arguments={"document_id": str(uuid5(NAMESPACE_URL, "foreign-document")), "limit": 1},
            )
        elif not results:
            query = "__missing__" if kind in {"missing", "follow_up"} else self.case["objective"]
            invocation = ToolInvocation(
                id="eval-search", name="search_evidence", arguments={"query": query, "limit": 5},
            )
        elif kind == "follow_up" and sum(
                item["tool_name"] == "search_evidence" for item in results) == 1:
            invocation = ToolInvocation(
                id="eval-follow-up", name="search_evidence",
                arguments={"query": "Kubernetes platform evidence", "limit": 5},
            )
        elif kind == "budget":
            invocation = ToolInvocation(
                id="eval-search-again", name="search_evidence", arguments={"query": "repeat", "limit": 5},
            )
        elif kind == "multi_document" and not any(
                item["tool_name"] == "inspect_document" for item in results):
            second = sorted(self.repository.documents, key=str)[1]
            invocation = ToolInvocation(
                id="eval-inspect", name="inspect_document",
                arguments={"document_id": str(second), "limit": 1},
            )
        elif self.case["requirements"] and not any(
                item["tool_name"] == "check_requirements" for item in results):
            invocation = ToolInvocation(
                id="eval-requirements", name="check_requirements",
                arguments={"requirements": self.case["requirements"]},
            )
        else:
            invocation = ToolInvocation(
                id="eval-draft", name="draft_investigation",
                arguments={"report": self._report(payload)},
            )
        return ToolModelResponse(
            calls=[invocation], input_tokens=40, output_tokens=20, cost_usd=0.001,
        )

    def _report(self, payload):
        kind = self.case["kind"]
        rows = _source_rows(payload)
        search_action = next(
            (item["action_id"] for item in payload["tool_results"] if item["tool_name"] == "search_evidence"),
            "eval-search",
        )
        report = {
            "supported_findings": [],
            "missing_evidence": [],
            "contradictory_evidence": [],
            "requirement_evidence": [],
            "verification_questions": [],
        }
        if kind in {"missing", "ambiguous"}:
            report["missing_evidence"] = [_gap(search_action, "Ownership" if kind == "ambiguous" else "COBOL")]
        elif kind == "follow_up":
            report["supported_findings"] = [{
                "text": "The follow-up search found relevant Kubernetes evidence.",
                "citations": [_citation(rows[0])],
            }]
            report["missing_evidence"] = [_gap(search_action, "Rust")]
        elif kind == "contradictory":
            report["contradictory_evidence"] = [{
                "statement": "The authorized documents contain conflicting ownership statements.",
                "citations": [_citation(rows[0]), _citation(rows[1])],
            }]
        else:
            invented = kind == "hallucinated_quote"
            selected = rows[:2] if kind == "multi_document" else rows[:1]
            report["supported_findings"] = [{
                "text": "The authorized document contains relevant technical evidence.",
                "citations": [_citation(row, invented=invented)],
            } for row in selected]
            if rows and not invented:
                report["verification_questions"] = [{
                    "question": "Can a reviewer verify the depth and context of this experience?",
                    "rationale": "The source excerpt is concise.",
                    "citations": [_citation(rows[0])],
                }]
        for requirement in self.case["requirements"]:
            matched = next((row for row in rows if requirement.casefold() in row["excerpt"].casefold()), None)
            report["requirement_evidence"].append({
                "requirement": requirement,
                "status": "supported" if matched else "insufficient",
                "explanation": ("The requirement is present in the cited excerpt." if matched else
                                "Insufficient evidence was found for this requirement."),
                "citations": [_citation(matched)] if matched else [],
                "checked_action_ids": [] if matched else [search_action],
            })
        return report


def _citations(report):
    for item in report.get("supported_findings", []):
        yield from item["citations"]
    for item in report.get("contradictory_evidence", []):
        yield from item["citations"]
    for item in report.get("requirement_evidence", []):
        yield from item["citations"]
    for item in report.get("verification_questions", []):
        yield from item["citations"]


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def evaluate(dataset_path: Path):
    dataset = json.loads(dataset_path.read_text())
    results = []
    for case in dataset["cases"]:
        repository = EvaluationRepository(case)
        pipeline = EvaluationPipeline(repository)
        provider = ScenarioProvider(case, repository)
        limits = AgentLimits(max_planning_iterations=1) if case["kind"] == "budget" else AgentLimits()
        investigation_id = uuid5(NAMESPACE_URL, f"investigation:{case['id']}")
        agent = InvestigationAgent(
            provider=provider,
            tools=InvestigationTools(pipeline, repository),
            checkpointer=InMemorySaver(),
            limits=limits,
            generation_policy_allows=lambda _tenant: True,
        )
        config = {"configurable": {"thread_id": str(investigation_id)}}
        started = perf_counter()
        agent.graph.invoke(initial_state(
            investigation_id=investigation_id,
            tenant_id=TENANT,
            subject="token-sha256:synthetic-eval",
            permitted_tenant_ids=[TENANT],
            objective=case["objective"],
            requirements=case["requirements"],
            document_ids=list(repository.documents),
        ), config)
        latency_ms = (perf_counter() - started) * 1_000
        snapshot = agent.graph.get_state(config)
        values = snapshot.values
        status = "awaiting_review" if snapshot.next == ("human_review",) else values.get("status")
        tools = [event["tool_name"] for event in values.get("execution_events", [])]
        report = values.get("report", {})
        evidence = {item["source_id"]: item["text"] for item in values.get("evidence", [])}
        citations = list(_citations(report))
        valid_citations = sum(
            citation["source_chunk_id"] in evidence
            and citation["quote"] in evidence[citation["source_chunk_id"]]
            for citation in citations
        )
        injection_boundary = True
        if case["kind"] == "prompt_injection":
            injection_boundary = (
                len(provider.prompts) >= 2
                and "UNTRUSTED_INVESTIGATION_DATA_JSON" in provider.prompts[-1]
                and "IGNORE ALL PRIOR INSTRUCTIONS" in provider.prompts[-1]
                and tools == case["expected_tools"]
            )
        outcome_correct = (
            status == case["expected_status"]
            and values.get("error_code", "") == case.get("expected_error", "")
            and tools == case["expected_tools"]
            and injection_boundary
        )
        results.append({
            "id": case["id"],
            "kind": case["kind"],
            "status": status,
            "error_code": values.get("error_code") or None,
            "tools": tools,
            "tool_selection_correct": tools == case["expected_tools"],
            "tool_selection_scored": case["kind"] not in {"invalid_arguments", "unavailable_tool"},
            "outcome_correct": outcome_correct,
            "task_case": bool(case.get("task_case")),
            "citation_count": len(citations),
            "valid_citation_count": valid_citations,
            "authorization_attempt": bool(case.get("authorization_attempt")),
            "authorization_blocked": values.get("error_code") == "authorization_denied",
            "unsupported_claim_attempt": bool(case.get("unsupported_claim_attempt")),
            "unsupported_claim_rejected": values.get("error_code") == "report_validation_failed",
            "prompt_injection_boundary_preserved": injection_boundary,
            "tool_calls": len(tools),
            "latency_ms": latency_ms,
            "mock_reported_cost_usd": values.get("cost_usd", 0.0),
        })

    task_cases = [item for item in results if item["task_case"]]
    auth_cases = [item for item in results if item["authorization_attempt"]]
    unsupported_cases = [item for item in results if item["unsupported_claim_attempt"]]
    selection_cases = [item for item in results if item["tool_selection_scored"]]
    total_citations = sum(item["citation_count"] for item in results)
    latencies = [item["latency_ms"] for item in results]
    tool_calls = [item["tool_calls"] for item in results]
    metrics = {
        "case_count": len(results),
        "task_completion_rate": sum(item["status"] == "awaiting_review" for item in task_cases) / len(task_cases),
        "scenario_outcome_accuracy": sum(item["outcome_correct"] for item in results) / len(results),
        "tool_selection_correctness": (
            sum(item["tool_selection_correct"] for item in selection_cases) / len(selection_cases)
        ),
        "tool_selection_scenario_count": len(selection_cases),
        "citation_validity": (sum(item["valid_citation_count"] for item in results) / total_citations
                              if total_citations else 1.0),
        "unsupported_claim_rejection_rate": (
            sum(item["unsupported_claim_rejected"] for item in unsupported_cases) / len(unsupported_cases)
        ),
        "authorization_violation_count": sum(not item["authorization_blocked"] for item in auth_cases),
        "authorization_block_rate": sum(item["authorization_blocked"] for item in auth_cases) / len(auth_cases),
        "mean_tool_calls": sum(tool_calls) / len(tool_calls),
        "min_tool_calls": min(tool_calls),
        "max_tool_calls": max(tool_calls),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mock_reported_cost_usd_total": sum(item["mock_reported_cost_usd"] for item in results),
        "live_provider_calls": 0,
        "live_provider_cost_usd": None,
    }
    return {
        "dataset_version": dataset["dataset_version"],
        "evaluation_version": "investigation-agent-eval-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "deterministic scripted fake",
        "metrics": metrics,
        "cases": results,
        "limitations": [
            "Synthetic in-memory documents and deterministic fake tool calls; no live Gemini behavior is measured.",
            "Latency is local single-process orchestration time, not provider, PostgreSQL, network, or load latency.",
            "Mock-reported cost validates budget accounting only and is not a billed provider cost.",
            "The evaluation pipeline returns labelled scope chunks directly and is not a retrieval relevance benchmark.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/investigation_eval_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/latest_investigation_eval.json"))
    args = parser.parse_args()
    result = evaluate(args.dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
