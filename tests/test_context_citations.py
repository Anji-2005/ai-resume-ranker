"""Regression tests for canonical citation IDs and context-budget citation boundaries."""

import json
from uuid import uuid4

import pytest

from generation.citations import CitationVerificationError, verify_answer, verify_recruiter_brief
from generation.prompts import question_answer_prompt, recruiter_brief_prompt
from generation.schemas import GroundedAnswer, RecruiterBrief
from retrieval.context import build_context, context_from_sources, source_header
from retrieval.contracts import RetrievalCandidate
from workflows.recruiter_brief import source_from_dict, source_to_dict


def candidate(content, *, key="section-0001-chunk-0001", start=100, chunk_id=None):
    return RetrievalCandidate(
        document_id=uuid4(), document_version_id=uuid4(), chunk_id=chunk_id or uuid4(), chunk_key=key,
        section_id="section-0001", content=content, source_start=start, source_end=start + len(content),
        page_number=2, source_key="synthetic", media_type="application/pdf",
    )


def answer(source_id, quote):
    return GroundedAnswer.model_validate({"claims": [{
        "text": "A synthetic claim.", "citations": [{"source_chunk_id": source_id, "quote": quote}],
    }]})


def test_canonical_source_id_is_identical_in_context_prompt_and_verifier():
    item = candidate("Built Python APIs with PostgreSQL.")
    context = build_context([item])

    assert context.source_chunk_ids == (str(item.chunk_id),)
    assert source_header(str(item.chunk_id)) in context.context_text
    assert item.chunk_key not in context.context_text
    for prompt in (question_answer_prompt("q", context), recruiter_brief_prompt({"description": "d"}, context)):
        payload = json.loads(prompt.split("SOURCE_DATA_JSON:\n", 1)[1])
        assert payload["available_source_chunk_ids"] == [str(item.chunk_id)]
        assert f"[source_chunk_id={item.chunk_id}]" in payload["retrieved_context"]
    assert verify_answer(answer(str(item.chunk_id), "Python APIs"), context).claims


def test_chunk_key_used_as_citation_id_is_rejected():
    item = candidate("Built Python APIs with PostgreSQL.")
    with pytest.raises(CitationVerificationError):
        verify_answer(answer(item.chunk_key, "Python APIs"), build_context([item]))


@pytest.mark.parametrize("budget", [50, 120, 300, 1_000])
def test_context_text_respects_total_character_budget_including_headers(budget):
    items = [candidate("word " * 200, key=f"k{index}") for index in range(4)]
    context = build_context(items, max_chunks=4, max_characters=budget)
    assert len(context.context_text) <= budget


def test_quote_from_truncated_tail_cannot_pass_verification():
    head = "Led Python migration. " * 3
    tail = "Managed a Kubernetes platform team."
    item = candidate(head + tail)
    budget = len(source_header(item.source_id)) + len(head)
    context = build_context([item], max_characters=budget)

    supplied = context.sources[0]
    assert supplied.truncated is True
    assert tail not in supplied.text
    # Provenance describes only the supplied span of the original document.
    assert supplied.source_start == item.source_start
    assert supplied.source_end == item.source_start + len(supplied.text)
    with pytest.raises(CitationVerificationError):
        verify_answer(answer(item.source_id, "Kubernetes platform team"), context)
    assert verify_answer(answer(item.source_id, "Led Python migration."), context).claims


def test_quote_from_passage_excluded_by_budget_or_chunk_limit_is_rejected():
    first, second = candidate("Python services. " * 5), candidate("Rust compiler work.")
    by_limit = build_context([first, second], max_chunks=1)
    by_budget = build_context([first, second], max_characters=len(source_header(first.source_id)) + 20)

    for context in (by_limit, by_budget):
        assert context.source_chunk_ids == (first.source_id,)
        with pytest.raises(CitationVerificationError):
            verify_answer(answer(second.source_id, "Rust compiler work"), context)


def test_missing_and_whitespace_and_invented_quotes_are_rejected():
    item = candidate("Built Python APIs with PostgreSQL.")
    context = build_context([item])
    brief = RecruiterBrief.model_validate({
        "claims": [{"text": "Python APIs.", "citations": [{"source_chunk_id": item.source_id, "quote": "Python APIs"}]}],
        "requirement_evidence": [{"requirement": "Python", "status": "supported", "explanation": "Stated."}],
        "interview_questions": [{"question": "Q?", "reason": "R.", "citations": [
            {"source_chunk_id": item.source_id, "quote": "Python APIs"}]}],
    })
    with pytest.raises(CitationVerificationError, match="at least one citation"):
        verify_recruiter_brief(brief, context)
    with pytest.raises(CitationVerificationError, match="must contain text"):
        verify_answer(answer(item.source_id, "   "), context)
    with pytest.raises(CitationVerificationError, match="exact excerpt"):
        verify_answer(answer(item.source_id, "Built Java APIs"), context)
    with pytest.raises(CitationVerificationError, match="not supplied"):
        verify_answer(answer(str(uuid4()), "Python APIs"), context)


def test_duplicate_and_empty_chunks_are_supplied_once():
    item = candidate("Python APIs.")
    empty = candidate("   ")
    context = build_context([item, item, empty])
    assert context.source_chunk_ids == (item.source_id,)


def test_direct_identifiers_are_redacted_before_supply_and_cannot_be_quoted():
    item = candidate("Contact jane.doe@example.com for the Python API project.")
    context = build_context([item])
    assert "jane.doe@example.com" not in context.context_text
    with pytest.raises(CitationVerificationError):
        verify_answer(answer(item.source_id, "jane.doe@example.com"), context)
    assert verify_answer(answer(item.source_id, "the Python API project"), context).claims


def test_serialized_workflow_sources_round_trip_without_reexpanding_truncated_text():
    item = candidate("Python. " * 50 + "Hidden tail sentence.")
    context = build_context([item], max_characters=120)
    restored = context_from_sources(source_from_dict(source_to_dict(s)) for s in context.sources)

    assert restored.context_text == context.context_text
    assert restored.supplied_text_by_id() == context.supplied_text_by_id()
    with pytest.raises(CitationVerificationError):
        verify_answer(answer(item.source_id, "Hidden tail sentence."), restored)
