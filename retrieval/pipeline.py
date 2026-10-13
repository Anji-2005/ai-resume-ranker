"""Single, bounded retrieval pipeline; it is not an autonomous agent workflow."""

from retrieval.authorization import AuthorizationContext
from retrieval.context import select_context
from retrieval.contracts import MetadataFilters, RetrievalContext
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from services.embeddings import SentenceTransformerEmbedder


class RetrievalPipeline:
    def __init__(self, repository, embedder=None, reranker=None):
        self.repository = repository
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.reranker = reranker

    def retrieve(self, authorization: AuthorizationContext, tenant_id: str, query: str,
                 filters: MetadataFilters | None = None, limit: int = 10,
                 candidate_limit: int = 30, use_reranker: bool = False) -> RetrievalContext:
        # This is deliberately the first executable operation: no embedding or database work occurs first.
        authorization.require_tenant(tenant_id)
        if not query.strip():
            return select_context([])
        filters = filters or MetadataFilters()
        candidate_limit = max(1, min(candidate_limit, 100))
        vector = self.embedder.embed([query])[0]
        dense = self.repository.search_dense(tenant_id, vector, filters, candidate_limit)
        lexical = self.repository.search_lexical(tenant_id, query, filters, candidate_limit)
        fused = reciprocal_rank_fusion(dense, lexical)
        if use_reranker:
            if self.reranker is None:
                self.reranker = CrossEncoderReranker()
            fused = self.reranker.rerank(query, fused[:candidate_limit])
        return select_context(fused[:max(1, min(limit, 50))])
