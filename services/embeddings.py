"""Baseline embedding adapter backed by the existing SentenceTransformer model."""

from importlib.metadata import PackageNotFoundError, version

from utils.matcher import MODEL_NAME, get_model


class SentenceTransformerEmbedder:
    model_name = MODEL_NAME

    try:
        model_version = version("sentence-transformers")
    except PackageNotFoundError:  # pragma: no cover - package is required in supported installs
        model_version = "unknown"

    def __init__(self, model=None):
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._model or get_model()
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]
