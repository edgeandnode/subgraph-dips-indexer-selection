"""
The Indexing Indexer Selection Algorithm (IISA) module.

This module provides:
- score_loader: Loads pre-computed indexer scores from a JSON file
- indexer_selection: Selects indexers based on weighted scoring
- iisa_http_endpoints: FastAPI HTTP endpoints for the service
"""

from .indexer_selection import IndexerSelector
from .score_loader import DataManager

__all__ = [
    "DataManager",
    "IndexerSelector",
]
