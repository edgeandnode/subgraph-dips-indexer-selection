"""
The Indexing Indexer Selection Algorithm (IISA) module.

This module provides:
- score_loader: Fetches pre-computed indexer scores from BigQuery
- indexer_selection: Selects indexers based on weighted scoring
- iisa_http_endpoints: FastAPI HTTP endpoints for the service
"""

from .indexer_selection import DataProcessor
from .score_loader import BigQueryProvider, DataManager

__all__ = [
    "BigQueryProvider",
    "DataManager",
    "DataProcessor",
]
