"""Deterministic verification of model citations against the evidence actually supplied."""

from generation.schemas import GroundedAnswer, GroundedClaim, RecruiterBrief, RequirementAnalysis


class CitationVerificationError(ValueError):
    pass


def supplied_sources(context) -> dict[str, str]:
    """Map canonical source IDs to the exact excerpt text that was sent to the model.

    Only ``context.sources`` counts as supplied evidence. Full candidate content,
    truncated tails, and chunks excluded by the budget are never accepted.
    """
    return context.supplied_text_by_id()


def verify_recruiter_brief(brief: RecruiterBrief, context) -> RecruiterBrief:
    sources = supplied_sources(context)
    claims = [_verified_claim(claim, sources) for claim in brief.claims]
    questions = []
    for question in brief.interview_questions:
        _verify_citations(question.citations, sources)
        questions.append(question)
    requirements = [_verified_requirement(item, sources) for item in brief.requirement_evidence]
    return RecruiterBrief(claims=claims, requirement_evidence=requirements, interview_questions=questions)


def verify_answer(answer: GroundedAnswer, context) -> GroundedAnswer:
    sources = supplied_sources(context)
    if not answer.claims:
        if not answer.uncertainty:
            raise CitationVerificationError("an answer without claims must explicitly state uncertainty")
        return answer
    return GroundedAnswer(claims=[_verified_claim(claim, sources) for claim in answer.claims], uncertainty=answer.uncertainty)


def _verified_claim(claim: GroundedClaim, sources):
    _verify_citations(claim.citations, sources)
    return claim


def _verified_requirement(item: RequirementAnalysis, sources):
    if item.status == "supported":
        _verify_citations(item.citations, sources)
    elif item.citations:
        _verify_citations(item.citations, sources)
    elif item.status in {"unsupported", "unclear"}:
        wording = item.explanation.casefold()
        if not any(marker in wording for marker in ("no supporting evidence", "unclear", "uncertain")):
            raise CitationVerificationError("unsupported requirement must explicitly state evidence uncertainty")
    return item


def _verify_citations(citations, sources):
    if not citations:
        raise CitationVerificationError("factual content requires at least one citation")
    for citation in citations:
        source = sources.get(citation.source_chunk_id)
        if source is None:
            raise CitationVerificationError("citation references a source that was not supplied to the model")
        if not citation.quote.strip():
            raise CitationVerificationError("citation quote must contain text")
        if citation.quote not in source:
            raise CitationVerificationError("citation quote is not an exact excerpt of the supplied source")
