from uuid import uuid4

import pytest

from retrieval.authorization import AuthorizationContext, RetrievalAuthorizationError
from retrieval.contracts import MetadataFilters, RetrievalCandidate
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from evals.retrieval_metrics import ndcg_at_k


def candidate(key, content):
    return RetrievalCandidate(
        document_id=uuid4(), document_version_id=uuid4(), chunk_id=uuid4(), chunk_key=key,
        section_id="section-0001", content=content, source_start=0, source_end=len(content),
        page_number=1, source_key="candidate-1", media_type="application/pdf",
    )


class FakeEmbedder:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.0] * 384 for _ in texts]


class FakeRepository:
    def __init__(self):
        self.dense = [candidate("chunk-dense", "Python API work")]
        self.lexical = [candidate("chunk-lexical", "SQL reporting work")]
        self.filters = None

    def search_dense(self, tenant_id, vector, filters, limit):
        self.filters = filters
        return self.dense

    def search_lexical(self, tenant_id, query, filters, limit):
        return self.lexical


def test_authorization_happens_before_embedding_or_repository_access():
    embedder, repository = FakeEmbedder(), FakeRepository()
    pipeline = RetrievalPipeline(repository, embedder=embedder)

    with pytest.raises(RetrievalAuthorizationError):
        pipeline.retrieve(AuthorizationContext("user", frozenset()), "tenant-a", "python")

    assert embedder.calls == 0
    assert repository.filters is None


def test_pipeline_fuses_sources_preserves_chunk_ids_and_filters():
    repository = FakeRepository()
    pipeline = RetrievalPipeline(repository, embedder=FakeEmbedder())
    filters = MetadataFilters(source_keys=frozenset({"candidate-1"}))

    result = pipeline.retrieve(
        AuthorizationContext("user", frozenset({"tenant-a"})), "tenant-a", "python", filters=filters
    )

    assert repository.filters == filters
    assert set(result.source_chunk_ids) == {str(item.chunk_id) for item in result.candidates}
    assert "source_chunk_id=chunk-dense" in result.context_text
    by_key = {item.chunk_key: item for item in result.candidates}
    assert by_key["chunk-dense"].dense_rank == 1
    assert by_key["chunk-lexical"].lexical_rank == 1


def test_cross_encoder_reranker_reorders_fused_candidates_without_losing_provenance():
    class Model:
        def predict(self, pairs):
            return [0.1, 0.9]

    first, second = candidate("one", "first"), candidate("two", "second")
    reranked = CrossEncoderReranker(model=Model()).rerank("query", [first, second])

    assert [item.chunk_key for item in reranked] == ["two", "one"]
    assert reranked[0].section_id == "section-0001"


def test_ndcg_at_k_uses_binary_relevance_and_rank_discount():
    assert ndcg_at_k(["relevant", "other"], ["relevant"], 2) == 1.0
    assert 0 < ndcg_at_k(["other", "relevant"], ["relevant"], 2) < 1.0
