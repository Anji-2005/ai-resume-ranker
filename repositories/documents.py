"""Transactional persistence and tenant-scoped hybrid retrieval."""

from dataclasses import dataclass
import hashlib
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from db.models import StoredDocument, StoredDocumentChunk, StoredDocumentVersion
from services.embeddings import SentenceTransformerEmbedder
from utils.pdf_parser import parse_document_from_bytes
from retrieval.contracts import MetadataFilters, RetrievalCandidate


@dataclass(frozen=True)
class DocumentSummaryRow:
    document_id: UUID
    created_at: object
    updated_at: object
    current_version_number: int
    version_count: int
    media_type: str
    chunk_count: int


@dataclass(frozen=True)
class VersionRow:
    version_id: UUID
    version_number: int
    is_current: bool
    media_type: str
    parser_name: str
    parser_version: str
    content_hash: str
    chunk_count: int
    created_at: object


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    version_id: UUID
    version_number: int
    created: bool
    embedding_model: str


class DocumentTooLargeError(ValueError):
    pass


class DocumentRepository:
    """The only persistence entry point for parsed candidate documents."""

    def __init__(self, session: Session, embedder=None):
        self.session = session
        self.embedder = embedder or SentenceTransformerEmbedder()

    def ingest_bytes(self, tenant_id: str, source_key: str, filename: str, data: bytes,
                     max_chunks: int | None = None) -> IngestionResult:
        _validate_scope(tenant_id, source_key)
        parsed = parse_document_from_bytes(filename, data)
        if max_chunks is not None and len(parsed.chunks) > max_chunks:
            # Bound embedding work and storage before any vector is computed.
            raise DocumentTooLargeError(f"document exceeds the {max_chunks}-chunk ingestion limit")
        digest = hashlib.sha256(data).hexdigest()
        existing = self.session.scalar(
            select(StoredDocument).where(
                StoredDocument.tenant_id == tenant_id, StoredDocument.source_key == source_key
            ).with_for_update()
        )
        if existing is None:
            document = StoredDocument(tenant_id=tenant_id, source_key=source_key)
            self.session.add(document)
            self.session.flush()
        else:
            document = existing
            if document.deleted_at is not None:
                document.deleted_at = None
            known = self.session.scalar(
                select(StoredDocumentVersion).where(
                    StoredDocumentVersion.document_id == document.id,
                    StoredDocumentVersion.content_hash == digest,
                )
            )
            if known is not None:
                return IngestionResult(
                    document.id, known.id, known.version_number, False, self.embedder.model_name
                )
        last_version = self.session.scalar(
            select(func.max(StoredDocumentVersion.version_number)).where(
                StoredDocumentVersion.document_id == document.id
            )
        ) or 0
        vectors = self.embedder.embed([chunk.text for chunk in parsed.chunks])
        if len(vectors) != len(parsed.chunks):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        self.session.query(StoredDocumentVersion).filter(
            StoredDocumentVersion.document_id == document.id
        ).update({StoredDocumentVersion.is_current: False}, synchronize_session=False)
        version = StoredDocumentVersion(
            document_id=document.id, version_number=last_version + 1, content_hash=digest,
            filename=filename, media_type=parsed.media_type, parser_name=parsed.parser_name,
            parser_version=parsed.parser_version, parsed_document=parsed.model_dump(mode="json"), is_current=True,
        )
        self.session.add(version)
        self.session.flush()
        for chunk, vector in zip(parsed.chunks, vectors, strict=True):
            self.session.add(StoredDocumentChunk(
                tenant_id=tenant_id, document_id=document.id, document_version_id=version.id,
                chunk_key=chunk.chunk_id, section_id=chunk.section_id, page_number=chunk.page_number,
                source_start=chunk.source_start, source_end=chunk.source_end,
                content=chunk.text, embedding=vector,
            ))
        return IngestionResult(document.id, version.id, version.version_number, True, self.embedder.model_name)

    def search_dense(self, tenant_id: str, vector: list[float], filters: MetadataFilters,
                     limit: int) -> list[RetrievalCandidate]:
        """Tenant-filtered pgvector cosine search. Called only by RetrievalPipeline."""
        scope = _active_chunks(tenant_id, filters)
        semantic_rows = self.session.execute(
            scope.add_columns((1 - StoredDocumentChunk.embedding.cosine_distance(vector)).label("score"))
            .order_by(StoredDocumentChunk.embedding.cosine_distance(vector)).limit(limit * 3)
        ).all()
        return [_candidate(chunk) for chunk, _ in semantic_rows]

    def search_lexical(self, tenant_id: str, query: str, filters: MetadataFilters,
                       limit: int) -> list[RetrievalCandidate]:
        """Tenant-filtered PostgreSQL native full-text ranking, not BM25."""
        scope = _active_chunks(tenant_id, filters)
        tsquery = func.plainto_tsquery("english", query)
        lexical_score = func.ts_rank_cd(StoredDocumentChunk.search_vector, tsquery)
        lexical_rows = self.session.execute(
            scope.add_columns(lexical_score.label("score"))
            .where(StoredDocumentChunk.search_vector.op("@@")(tsquery))
            .order_by(lexical_score.desc()).limit(limit * 3)
        ).all()
        return [_candidate(chunk) for chunk, _ in lexical_rows]

    def count_chunks(self, version_id: UUID) -> int:
        return self.session.scalar(
            select(func.count(StoredDocumentChunk.id)).where(StoredDocumentChunk.document_version_id == version_id)
        ) or 0

    def list_documents(self, tenant_id: str, limit: int = 100, offset: int = 0) -> tuple[list[DocumentSummaryRow], int]:
        """Tenant-filtered document summaries; never returns filenames or source keys."""
        _validate_tenant(tenant_id)
        version_stats = select(
            StoredDocumentVersion.document_id.label("document_id"),
            func.count(StoredDocumentVersion.id).label("version_count"),
            func.max(StoredDocumentVersion.created_at).label("updated_at"),
        ).group_by(StoredDocumentVersion.document_id).subquery()
        chunk_stats = select(
            StoredDocumentChunk.document_version_id.label("version_id"),
            func.count(StoredDocumentChunk.id).label("chunk_count"),
        ).group_by(StoredDocumentChunk.document_version_id).subquery()
        scope = (StoredDocument.tenant_id == tenant_id, StoredDocument.deleted_at.is_(None))
        total = self.session.scalar(select(func.count(StoredDocument.id)).where(*scope)) or 0
        rows = self.session.execute(
            select(StoredDocument, StoredDocumentVersion, version_stats.c.version_count,
                   version_stats.c.updated_at, func.coalesce(chunk_stats.c.chunk_count, 0))
            .join(StoredDocumentVersion, (StoredDocumentVersion.document_id == StoredDocument.id)
                  & StoredDocumentVersion.is_current.is_(True))
            .join(version_stats, version_stats.c.document_id == StoredDocument.id)
            .outerjoin(chunk_stats, chunk_stats.c.version_id == StoredDocumentVersion.id)
            .where(*scope)
            .order_by(version_stats.c.updated_at.desc(), StoredDocument.id)
            .limit(max(1, min(limit, 500))).offset(max(0, offset))
        ).all()
        return [
            DocumentSummaryRow(
                document_id=document.id, created_at=document.created_at, updated_at=updated_at,
                current_version_number=version.version_number, version_count=version_count,
                media_type=version.media_type, chunk_count=chunk_count,
            )
            for document, version, version_count, updated_at, chunk_count in rows
        ], total

    def get_document(self, tenant_id: str, document_id: UUID) -> StoredDocument | None:
        _validate_tenant(tenant_id)
        return self.session.scalar(select(StoredDocument).where(
            StoredDocument.id == document_id, StoredDocument.tenant_id == tenant_id,
            StoredDocument.deleted_at.is_(None),
        ))

    def list_versions(self, tenant_id: str, document_id: UUID) -> list[VersionRow]:
        document = self.get_document(tenant_id, document_id)
        if document is None:
            return []
        versions = self.session.scalars(
            select(StoredDocumentVersion).where(StoredDocumentVersion.document_id == document.id)
            .order_by(StoredDocumentVersion.version_number.desc())
        ).all()
        return [
            VersionRow(
                version_id=version.id, version_number=version.version_number, is_current=version.is_current,
                media_type=version.media_type, parser_name=version.parser_name,
                parser_version=version.parser_version, content_hash=version.content_hash,
                chunk_count=self.count_chunks(version.id), created_at=version.created_at,
            )
            for version in versions
        ]

    def get_version_chunks(self, tenant_id: str, document_id: UUID, version_number: int,
                           limit: int = 500) -> list[StoredDocumentChunk]:
        """Chunks for one version of a tenant's document, ordered by source offset."""
        _validate_tenant(tenant_id)
        return list(self.session.scalars(
            select(StoredDocumentChunk)
            .join(StoredDocumentVersion, StoredDocumentVersion.id == StoredDocumentChunk.document_version_id)
            .join(StoredDocument, StoredDocument.id == StoredDocumentChunk.document_id)
            .where(
                StoredDocument.id == document_id, StoredDocument.tenant_id == tenant_id,
                StoredDocumentChunk.tenant_id == tenant_id,
                StoredDocumentVersion.version_number == version_number,
            )
            .order_by(StoredDocumentChunk.source_start, StoredDocumentChunk.chunk_key)
            .limit(max(1, min(limit, 1_000)))
        ).all())

    def existing_document_ids(self, tenant_id: str, document_ids) -> set[UUID]:
        """Return which of the given IDs exist for this tenant (used to reject foreign IDs)."""
        _validate_tenant(tenant_id)
        if not document_ids:
            return set()
        return set(self.session.scalars(select(StoredDocument.id).where(
            StoredDocument.tenant_id == tenant_id, StoredDocument.deleted_at.is_(None),
            StoredDocument.id.in_(list(document_ids)),
        )).all())

    def delete_document(self, tenant_id: str, document_id: UUID) -> bool:
        document = self.session.scalar(select(StoredDocument).where(
            StoredDocument.id == document_id, StoredDocument.tenant_id == tenant_id
        ).with_for_update())
        if document is None:
            return False
        self.session.delete(document)  # physical deletion; foreign keys cascade to versions and vectors
        return True


def _active_chunks(tenant_id: str, filters: MetadataFilters) -> Select:
    statement = select(StoredDocumentChunk).join(
        StoredDocument, StoredDocument.id == StoredDocumentChunk.document_id
    ).join(
        StoredDocumentVersion, StoredDocumentVersion.id == StoredDocumentChunk.document_version_id
    ).where(
        StoredDocumentChunk.tenant_id == tenant_id, StoredDocument.deleted_at.is_(None),
        StoredDocumentVersion.is_current.is_(True),
    )
    if filters.document_ids:
        statement = statement.where(StoredDocumentChunk.document_id.in_(filters.document_ids))
    if filters.document_version_ids:
        statement = statement.where(StoredDocumentChunk.document_version_id.in_(filters.document_version_ids))
    if filters.source_keys:
        statement = statement.where(StoredDocument.source_key.in_(filters.source_keys))
    if filters.media_types:
        statement = statement.where(StoredDocumentVersion.media_type.in_(filters.media_types))
    return statement


def _candidate(chunk: StoredDocumentChunk) -> RetrievalCandidate:
    return RetrievalCandidate(
        document_id=chunk.document_id, document_version_id=chunk.document_version_id,
        chunk_id=chunk.id, chunk_key=chunk.chunk_key, section_id=chunk.section_id,
        content=chunk.content, source_start=chunk.source_start, source_end=chunk.source_end,
        page_number=chunk.page_number, source_key=chunk.document_version.document.source_key,
        media_type=chunk.document_version.media_type,
    )


def _validate_tenant(tenant_id: str):
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id is required")


def _validate_scope(tenant_id: str, source_key: str):
    if not tenant_id.strip() or not source_key.strip():
        raise ValueError("tenant_id and source_key are required")
