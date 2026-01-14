"""
Pydantic models for API request and response objects.

These models match the API contract expected by the Rust HTTP client
in dipper-iisa/src/http_client.rs.
"""

from typing import Optional

from pydantic import BaseModel


class CandidateIndexer(BaseModel):
    """
    A candidate indexer with ID and URL.

    The URL is used for GeoIP resolution to determine geographic diversity.
    """

    id: str
    url: str


class SelectionRequest(BaseModel):
    """
    Request body for indexer selection endpoints.

    Matches the SelectionRequest struct in the Rust HTTP client.
    """

    deployment_id: str
    candidates: Optional[list[CandidateIndexer]] = None
    existing_indexers: Optional[list[str]] = None
    pending_agreements: Optional[dict[str, list[str]]] = None
    num_candidates: Optional[int] = None
    blocklist: Optional[list[str]] = None
    declined_indexers: Optional[dict[str, list[str]]] = None


class SingleSelectionResponse(BaseModel):
    """
    Response for the /select-one endpoint.

    Returns a single selected indexer ID or None if no selection was made.
    """

    indexer_id: Optional[str] = None


class MultiSelectionResponse(BaseModel):
    """
    Response for the /select-many endpoint.

    Returns a list of selected indexer IDs.
    """

    indexer_ids: list[str]


class HealthResponse(BaseModel):
    """
    Response for the /health endpoint.
    """

    status: str
    data_loaded: bool
