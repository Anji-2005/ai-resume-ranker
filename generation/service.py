"""Grounded recruiter brief and document Q&A orchestration."""

import json

from generation.citations import verify_answer, verify_recruiter_brief
from generation.prompts import (
    DOCUMENT_QA_PROMPT_VERSION,
    RECRUITER_BRIEF_PROMPT_VERSION,
    question_answer_prompt,
    recruiter_brief_prompt,
)
from generation.schemas import GroundedAnswer, RecruiterBrief
from observability.tracing import StructuredTracer, TraceTimer, estimated_cost_usd


class GroundedGenerationService:
    def __init__(self, provider, retrieval_pipeline, tracer=None):
        self.provider = provider
        self.retrieval_pipeline = retrieval_pipeline
        self.tracer = tracer or StructuredTracer()

    def _generate(self, operation, prompt_version, prompt, schema):
        try:
            with TraceTimer() as timer:
                payload = self.provider.generate(prompt, schema)
        except Exception as exc:
            self.tracer.emit_generation(
                operation=operation,
                prompt_version=prompt_version,
                model_name=getattr(self.provider, "model_name", "unknown"),
                latency_ms=round(timer.latency_ms, 3),
                input_characters=len(prompt),
                output_characters=0,
                estimated_cost_usd=estimated_cost_usd(len(prompt), 0),
                outcome="error",
                error_class=type(exc).__name__,
            )
            raise
        output_characters = len(json.dumps(payload, ensure_ascii=False, default=str))
        self.tracer.emit_generation(
            operation=operation,
            prompt_version=prompt_version,
            model_name=getattr(self.provider, "model_name", "unknown"),
            latency_ms=round(timer.latency_ms, 3),
            input_characters=len(prompt),
            output_characters=output_characters,
            estimated_cost_usd=estimated_cost_usd(len(prompt), output_characters),
            outcome="success",
        )
        return payload

    def recruiter_brief(self, authorization, tenant_id, job, *, filters=None, limit=8, use_reranker=False):
        context = self.retrieval_pipeline.retrieve(
            authorization, tenant_id, job.get("description", ""), filters=filters,
            limit=limit, use_reranker=use_reranker,
        )
        if not context.candidates:
            raise ValueError("no authorized evidence was retrieved")
        payload = self._generate(
            "recruiter_brief", RECRUITER_BRIEF_PROMPT_VERSION,
            recruiter_brief_prompt(job, context), RecruiterBrief,
        )
        return verify_recruiter_brief(RecruiterBrief.model_validate(payload), context)

    def answer_question(self, authorization, tenant_id, question, *, filters=None, limit=5, use_reranker=False):
        context = self.retrieval_pipeline.retrieve(
            authorization, tenant_id, question, filters=filters,
            limit=limit, use_reranker=use_reranker,
        )
        if not context.candidates:
            return GroundedAnswer(claims=[], uncertainty="No authorized supporting evidence was retrieved.")
        payload = self._generate(
            "document_qa", DOCUMENT_QA_PROMPT_VERSION,
            question_answer_prompt(question, context), GroundedAnswer,
        )
        return verify_answer(GroundedAnswer.model_validate(payload), context)
