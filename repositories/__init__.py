"""Repository adapters."""
from repositories.investigations import InvestigationRepository
from repositories.ingestion_jobs import IngestionJobRepository

__all__ = ["IngestionJobRepository", "InvestigationRepository"]
