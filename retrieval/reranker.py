"""Optional lazy cross-encoder reranking of a bounded fused candidate set."""

from retrieval.contracts import RetrievalCandidate


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        self.model_name = model_name
        self._model = model

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, candidate.content) for candidate in candidates])
        ranked = [
            RetrievalCandidate(**{**candidate.__dict__, "reranker_score": float(score)})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(ranked, key=lambda item: (-item.reranker_score, -item.fusion_score))
