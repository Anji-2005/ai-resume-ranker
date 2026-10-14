"""Grounded recruiter brief and document Q&A orchestration."""

from dataclasses import dataclass
import json

from pydantic import ValidationError

from generation.citations import verify_answer, verify_recruiter_brief
from generation.prompts import (
    DOCUMENT_QA_PROMPT_VERSION,
    RECRUITER_BRIEF_PROMPT_VERSION,
    question_answer_prompt,
    recruiter_brief_prompt,
)
from generation.schemas import GroundedAnswer, RecruiterBrief
from observability.tracing import StructuredTracer, TraceTimer, estimated_cost_usd


OUTPUT_SCHEMA_VERSION = "grounded-output-v1"


class GenerationProviderError(RuntimeError):
    """The provider call failed; the message intentionally carries no prompt or response text."""


class GenerationOutputError(ValueError):
    """The provider returned output that does not match the strict schema."""


class NoEvidenceError(ValueError):
    """No authorized evidence was retrieved, so nothing can be generated."""


@dataclass(frozen=True)
class GroundedResult:
    output: object
    context: object
    prompt_version: str
    model_name: str
    schema_version: str = OUTPUT_SCHEMA_VERSION


class GroundedGenerationService:
    def __init__(self, provider, retrieval_pipeline, tracer=None):
        self.provider = provider
        self.retrieval_pipeline = retrieval_pipeline
        self.tracer = tracer or StructuredTracer()

    @property
    def model_name(self):
        return getattr(self.provider, "model_name", "unknown")

    def _generate(self, operation, prompt_version, prompt, schema):
        try:
            with TraceTimer() as timer:
                payload = self.provider.generate(prompt, schema)
        except Exception as exc:
            self.tracer.emit_generation(
                operation=operation,
                prompt_version=prompt_version,
                model_name=self.model_name,
                latency_ms=round(timer.latency_ms, 3),
                input_characters=len(prompt),
                output_characters=0,
                estimated_cost_usd=estimated_cost_usd(len(prompt), 0),
                outcome="error",
                error_class=type(exc).__name__,
            )
            raise GenerationProviderError("the generation provider request failed") from exc
        output_characters = len(json.dumps(payload, ensure_ascii=False, default=str))
        self.tracer.emit_generation(
            operation=operation,
            prompt_version=prompt_version,
            model_name=self.model_name,
            latency_ms=round(timer.latency_ms, 3),
            input_characters=len(prompt),
            output_characters=output_characters,
            estimated_cost_usd=estimated_cost_usd(len(prompt), output_characters),
            outcome="success",
        )
        return payload

    @staticmethod
    def _validate(schema, payload):
        try:
            return schema.model_validate(payload)
        except ValidationError:
            # Validation messages can echo model output or source text; never propagate them.
            raise GenerationOutputError("generated output did not match the required schema") from None

    def grounded_brief(self, authorization, tenant_id, job, *, filters=None, limit=8,
                       use_reranker=False) -> GroundedResult:
        context = self.retrieval_pipeline.retrieve(
            authorization, tenant_id, job.get("description", ""), filters=filters,
            limit=limit, use_reranker=use_reranker,
        )
        if not context.candidates:
            raise NoEvidenceError("no authorized evidence was retrieved")
        payload = self._generate(
            "recruiter_brief", RECRUITER_BRIEF_PROMPT_VERSION,
            recruiter_brief_prompt(job, context), RecruiterBrief,
        )
        brief = verify_recruiter_brief(self._validate(RecruiterBrief, payload), context)
        return GroundedResult(brief, context, RECRUITER_BRIEF_PROMPT_VERSION, self.model_name)

    def grounded_answer(self, authorization, tenant_id, question, *, filters=None, limit=5,
                        use_reranker=False) -> GroundedResult:
        context = self.retrieval_pipeline.retrieve(
            authorization, tenant_id, question, filters=filters,
            limit=limit, use_reranker=use_reranker,
        )
        if not context.candidates:
            answer = GroundedAnswer(claims=[], uncertainty="No authorized supporting evidence was retrieved.")
            return GroundedResult(answer, context, DOCUMENT_QA_PROMPT_VERSION, self.model_name)
        payload = self._generate(
            "document_qa", DOCUMENT_QA_PROMPT_VERSION,
            question_answer_prompt(question, context), GroundedAnswer,
        )
        answer = verify_answer(self._validate(GroundedAnswer, payload), context)
        return GroundedResult(answer, context, DOCUMENT_QA_PROMPT_VERSION, self.model_name)

    def recruiter_brief(self, authorization, tenant_id, job, *, filters=None, limit=8, use_reranker=False):
        return self.grounded_brief(
            authorization, tenant_id, job, filters=filters, limit=limit, use_reranker=use_reranker,
        ).output

    def answer_question(self, authorization, tenant_id, question, *, filters=None, limit=5, use_reranker=False):
        return self.grounded_answer(
            authorization, tenant_id, question, filters=filters, limit=limit, use_reranker=use_reranker,
        ).output
