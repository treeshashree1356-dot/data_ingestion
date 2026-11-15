"""Services package for data ingestion and retrieval orchestration."""

from services import indexing_service, ingestion_service, query_service
from services.indexing_service import (
    IndexingService,
    indexing_service as default_indexing_service,
)
from services.ingestion_service import (
    IngestionResult,
    IngestionService,
    ingestion_service as default_ingestion_service,
)
from services.query_service import (
    QueryService,
    query_service as default_query_service,
)

__all__ = [
    "ingestion_service",
    "IngestionService",
    "IngestionResult",
    "default_ingestion_service",
    "indexing_service",
    "IndexingService",
    "default_indexing_service",
    "query_service",
    "QueryService",
    "default_query_service",
]
