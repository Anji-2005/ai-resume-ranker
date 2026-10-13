"""Run the production retrieval pipeline against a seeded PostgreSQL/pgvector dataset.

This script intentionally does not emulate PostgreSQL FTS in memory. It measures
the real dense + native-tsvector + RRF pipeline only when a test database exists.
"""

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter

from db.session import get_session_factory
from docx import Document
from evals.retrieval_metrics import summarize
from repositories.documents import DocumentRepository
from retrieval.authorization import AuthorizationContext
from retrieval.pipeline import RetrievalPipeline


def evaluate(dataset_path: Path, tenant_id: str):
    dataset = json.loads(dataset_path.read_text())
    session = get_session_factory()()
    try:
        pipeline = RetrievalPipeline(DocumentRepository(session))
        authorization = AuthorizationContext("evaluation", frozenset({tenant_id}))
        repository = pipeline.repository
        # Seed parser-produced chunks so this evaluates the same ingestion/retrieval path.
        for document in dataset["documents"]:
            output = BytesIO()
            docx = Document()
            docx.add_paragraph(document["text"])
            docx.save(output)
            repository.ingest_bytes(tenant_id, document["id"], f"{document['id']}.docx", output.getvalue())
        session.commit()
        rows = []
        for query in dataset["queries"]:
            started = perf_counter()
            context = pipeline.retrieve(authorization, tenant_id, query["text"], limit=5)
            rows.append({
                "query_id": query["id"], "ranked_chunk_ids": [item.source_key for item in context.candidates],
                "relevant_chunk_ids": query["relevant_source_keys"],
                "latency_ms": (perf_counter() - started) * 1_000,
            })
        return {
            "dataset_version": dataset["dataset_version"],
            "method": "postgres_native_fts_pgvector_rrf", "metrics": summarize(rows), "per_query": rows,
        }
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_retrieval.json"))
    parser.add_argument("--tenant", default="evaluation")
    parser.add_argument("--output", type=Path, default=Path("evals/latest_postgres_retrieval.json"))
    args = parser.parse_args()
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is required; this script measures real PostgreSQL retrieval.")
    result = evaluate(args.dataset, args.tenant)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
