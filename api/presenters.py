"""Map domain objects to HTTP contracts at the confidentiality boundary.

Display text is identifier-redacted here; filenames and source keys never leave
the service.
"""

from uuid import UUID

from api import schemas
from utils.privacy import redact_direct_identifiers


def display_text(content: str, source_start: int) -> str:
    return redact_direct_identifiers(content, at_document_start=source_start == 0)


def search_result(candidate) -> schemas.SearchResult:
    score = candidate.reranker_score if candidate.reranker_score is not None else candidate.fusion_score
    return schemas.SearchResult(
        document_id=candidate.document_id, document_version_id=candidate.document_version_id,
        candidate_ref=schemas.candidate_ref(candidate.document_id), chunk_id=candidate.chunk_id,
        chunk_key=candidate.chunk_key, section_id=candidate.section_id,
        content=display_text(candidate.content, candidate.source_start),
        source_start=candidate.source_start, source_end=candidate.source_end,
        page_number=candidate.page_number, score=score, fusion_score=candidate.fusion_score,
        dense_rank=candidate.dense_rank, lexical_rank=candidate.lexical_rank,
        reranker_score=candidate.reranker_score, source_chunk_id=candidate.source_id,
    )


def supplied_source(source) -> schemas.SuppliedSource:
    candidate = source.candidate
    return schemas.SuppliedSource(
        source_chunk_id=source.source_id, document_id=candidate.document_id,
        candidate_ref=schemas.candidate_ref(candidate.document_id), section_id=candidate.section_id,
        page_number=candidate.page_number, source_start=source.source_start, source_end=source.source_end,
        truncated=source.truncated, excerpt=source.text,
    )


def generation_metadata(result) -> schemas.GenerationMetadata:
    return schemas.GenerationMetadata(
        provider="gemini-langchain", model=result.model_name,
        prompt_version=result.prompt_version, schema_version=result.schema_version,
    )


def chunk_view(chunk) -> schemas.ChunkView:
    return schemas.ChunkView(
        chunk_id=chunk.id, chunk_key=chunk.chunk_key, section_id=chunk.section_id,
        page_number=chunk.page_number, source_start=chunk.source_start, source_end=chunk.source_end,
        content=display_text(chunk.content, chunk.source_start),
    )


def version_info(row) -> schemas.DocumentVersionInfo:
    return schemas.DocumentVersionInfo(
        version_id=row.version_id, version_number=row.version_number, is_current=row.is_current,
        media_type=row.media_type, parser_name=row.parser_name, parser_version=row.parser_version,
        content_sha256_prefix=row.content_hash[:12], chunk_count=row.chunk_count, created_at=row.created_at,
    )


def document_summary(row) -> schemas.DocumentSummary:
    return schemas.DocumentSummary(
        document_id=row.document_id, candidate_ref=schemas.candidate_ref(row.document_id),
        current_version_number=row.current_version_number, version_count=row.version_count,
        media_type=row.media_type, chunk_count=row.chunk_count,
        created_at=row.created_at, updated_at=row.updated_at,
    )


def candidate_refs(document_ids) -> list[str]:
    return [schemas.candidate_ref(UUID(str(value))) for value in document_ids]
