"""Report policy and citation validation for investigation output."""

import re

from generation.citations import CitationVerificationError, verify_citations
from investigations.schemas import InvestigationReport
from retrieval.context import context_from_sources


class InvestigationReportError(ValueError):
    pass


_PROHIBITED = re.compile(
    r"\b(?:hire|hiring decision|reject (?:the )?(?:candidate|applicant)|shortlist|"
    r"best candidate|rank(?:ed|ing)? candidates?|recommend (?:the )?(?:candidate|applicant)|"
    r"suitable for employment|infer (?:age|race|gender|religion|disability))\b",
    re.I,
)
_GAP_MARKERS = ("insufficient", "missing", "no evidence", "not found", "unclear", "not established")


def verify_investigation_report(report: InvestigationReport, sources, valid_action_ids,
                                expected_requirements=()) -> InvestigationReport:
    """Verify report content only against evidence and searches from this run."""
    context = context_from_sources(sources)
    texts = []
    try:
        expected = {value.casefold() for value in expected_requirements}
        actual = [item.requirement.casefold() for item in report.requirement_evidence]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise InvestigationReportError("requirement evidence must exactly match the authorized criteria")
        for finding in report.supported_findings:
            verify_citations(finding.citations, context)
            texts.append(finding.text)
        for contradiction in report.contradictory_evidence:
            verify_citations(contradiction.citations, context)
            if len({item.source_chunk_id for item in contradiction.citations}) < 2:
                raise InvestigationReportError("contradictory evidence needs at least two distinct sources")
            texts.append(contradiction.statement)
        for gap in report.missing_evidence:
            if not set(gap.checked_action_ids) <= set(valid_action_ids):
                raise InvestigationReportError("evidence gap references an action that was not executed")
            if not any(marker in gap.explanation.casefold() for marker in _GAP_MARKERS):
                raise InvestigationReportError("evidence gaps must explicitly state uncertainty")
            texts.extend((gap.topic, gap.explanation))
        for question in report.verification_questions:
            if question.citations:
                verify_citations(question.citations, context)
            texts.extend((question.question, question.rationale))
        for item in report.requirement_evidence:
            if item.status == "supported":
                if not item.citations:
                    raise InvestigationReportError("supported requirements need citations")
                verify_citations(item.citations, context)
            elif item.status == "contradictory":
                verify_citations(item.citations, context)
                if len({citation.source_chunk_id for citation in item.citations}) < 2:
                    raise InvestigationReportError("contradictory requirements need at least two sources")
            else:
                if not item.checked_action_ids or not set(item.checked_action_ids) <= set(valid_action_ids):
                    raise InvestigationReportError("insufficient requirements need executed evidence actions")
                if not any(marker in item.explanation.casefold() for marker in _GAP_MARKERS):
                    raise InvestigationReportError("insufficient requirements must explicitly state uncertainty")
            texts.extend((item.requirement, item.explanation))
    except CitationVerificationError as exc:
        raise InvestigationReportError("investigation report contains an unsupported citation") from exc
    if _PROHIBITED.search("\n".join(texts)):
        raise InvestigationReportError("investigation report contains a prohibited employment conclusion")
    return report
