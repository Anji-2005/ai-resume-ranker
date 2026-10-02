"""Semantic similarity utilities with lazy model loading."""

from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once, only when analysis begins."""
    from sentence_transformers import SentenceTransformer

    # Prefer the local cache to avoid slow network metadata checks on every launch.
    # If this is the first run, fall back to the normal download behavior.
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except (OSError, ValueError):
        return SentenceTransformer(MODEL_NAME)


def calculate_match_score(job_description, resume_text, model=None):
    """Return a bounded semantic similarity percentage."""
    if not job_description.strip() or not resume_text.strip():
        return 0.0

    encoder = model or get_model()
    embeddings = encoder.encode(
        [job_description, resume_text],
        show_progress_bar=False,
    )
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return round(min(1.0, max(0.0, similarity)) * 100, 2)
