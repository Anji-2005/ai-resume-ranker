"""Deterministic verification of model citations against retrieved source chunks."""

from generation.schemas import GroundedAnswer, GroundedClaim, RecruiterBrief, RequirementAnalysis


class CitationVerificationError(ValueError):
    pass


def verify_recruiter_brief(brief: RecruiterBrief, context) -> RecruiterBrief:
    sources = {str(candidate.chunk_id): candidate.content for candidate in context.candidates}
    claims = [_verified_claim(claim, sources) for claim in brief.claims]
    questions = []
    for question in brief.interview_questions:
        _verify_citations(question.citations, sources)
        questions.append(question)
    requirements = [_verified_requirement(item, sources) for item in brief.requirement_evidence]
    return RecruiterBrief(claims=claims, requirement_evidence=requirements, interview_questions=questions)


def verify_answer(answer: GroundedAnswer, context) -> GroundedAnswer:
    sources = {str(candidate.chunk_id): candidate.content for candidate in context.candidates}
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
            raise CitationVerificationError("citation references a chunk outside authorized retrieval")
        if citation.quote not in source:
            raise CitationVerificationError("citation quote is not an exact source excerpt")
