"""Single, bounded retrieval pipeline; it is not an autonomous agent workflow."""

from retrieval.authorization import AuthorizationContext
from retrieval.context import DEFAULT_MAX_CHARACTERS, build_context
from retrieval.contracts import MetadataFilters, RetrievalCandidate, RetrievalContext
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from services.embeddings import SentenceTransformerEmbedder


MAX_RESULTS = 50
MAX_CANDIDATES = 100


class RetrievalPipeline:
    def __init__(self, repository, embedder=None, reranker=None):
        self.repository = repository
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.reranker = reranker

    def search(self, authorization: AuthorizationContext, tenant_id: str, query: str,
               filters: MetadataFilters | None = None, limit: int = 10,
               candidate_limit: int = 30, use_reranker: bool = False) -> list[RetrievalCandidate]:
        """Return ranked, provenance-bearing chunks for display; no model context budget applies."""
        # This is deliberately the first executable operation: no embedding or database work occurs first.
        authorization.require_tenant(tenant_id)
        if not query.strip():
            return []
        filters = filters or MetadataFilters()
        candidate_limit = max(1, min(candidate_limit, MAX_CANDIDATES))
        vector = self.embedder.embed([query])[0]
        dense = self.repository.search_dense(tenant_id, vector, filters, candidate_limit)
        lexical = self.repository.search_lexical(tenant_id, query, filters, candidate_limit)
        fused = reciprocal_rank_fusion(dense, lexical)
        if use_reranker:
            if self.reranker is None:
                self.reranker = CrossEncoderReranker()
            fused = self.reranker.rerank(query, fused[:candidate_limit])
        return fused[:max(1, min(limit, MAX_RESULTS))]

    def retrieve(self, authorization: AuthorizationContext, tenant_id: str, query: str,
                 filters: MetadataFilters | None = None, limit: int = 10,
                 candidate_limit: int = 30, use_reranker: bool = False,
                 max_characters: int = DEFAULT_MAX_CHARACTERS) -> RetrievalContext:
        """Return bounded model context; ``limit`` is the maximum number of supplied chunks."""
        candidates = self.search(
            authorization, tenant_id, query, filters=filters, limit=limit,
            candidate_limit=candidate_limit, use_reranker=use_reranker,
        )
        return build_context(candidates, max_chunks=max(1, min(limit, MAX_RESULTS)), max_characters=max_characters)
