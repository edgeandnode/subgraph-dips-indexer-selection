"""
IISA HTTP API - FastAPI endpoints for the Indexing Indexer Selection Algorithm.

This module exposes HTTP endpoints for indexer selection that match the API
contract expected by the Rust HTTP client in dipper-iisa.

Endpoints:
- GET /health - Health check, reports if data is loaded
- POST /refresh - Reload scores from BigQuery
- POST /select-one - Select single best indexer for a deployment
- POST /select-many - Select multiple indexers for a deployment
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .indexer_selection import DataProcessor
from .score_loader import BigQueryProvider, DataManager

__all__ = ["app", "Settings", "get_settings"]


# =============================================================================
# Configuration
# =============================================================================


class Settings(BaseSettings):
    """
    Service configuration loaded from environment variables.

    All settings are prefixed with IISA_ in environment variables.
    For example, IISA_GCP_PROJECT sets the gcp_project field.
    """

    model_config = SettingsConfigDict(
        env_prefix="IISA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Cloud Platform
    gcp_project: str
    gcp_location: str = "US"

    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the process.
    """
    return Settings()


# =============================================================================
# Request/Response Models
# =============================================================================


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


# =============================================================================
# Service State
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("iisa-service")


class IISAState:
    """
    Holds the IISA service state including initialized providers and cached data.
    """

    def __init__(self) -> None:
        self.settings: Optional[Settings] = None
        self.data_manager: Optional[DataManager] = None
        self._history: Optional[pd.DataFrame] = None
        self._initialized: bool = False

    def initialize(self, settings: Settings) -> bool:
        """
        Initialize the IISA providers and fetch initial data.

        Returns True if initialization succeeded, False if we should fall back
        to random selection mode.
        """
        self.settings = settings

        try:
            logger.info("Initializing IISA providers...")

            bigquery = BigQueryProvider(
                project=settings.gcp_project,
                location=settings.gcp_location,
            )

            self.data_manager = DataManager(bigquery)

            logger.info("IISA providers initialized successfully")
            self._initialized = True
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize IISA providers: {e}")
            logger.warning("Service will operate in random selection fallback mode")
            self._initialized = False
            return False

    def refresh_data(self) -> bool:
        """
        Load pre-computed indexer scores from BigQuery.

        This method loads scores from the indexer_scores table (populated by CronJob)
        instead of computing them in-container. This is much faster and uses less memory.

        Returns True if scores were loaded successfully, False otherwise.
        """
        if not self._initialized or self.data_manager is None:
            logger.warning("Cannot refresh data: DataManager not initialized")
            return False

        try:
            logger.info("Loading pre-computed indexer scores from BigQuery...")
            success = self.data_manager.load_scores()

            if success:
                self._history = self.data_manager.get_data()
                if self._history is not None:
                    logger.info(f"Scores loaded successfully: {len(self._history)} indexers")
                    return True

            logger.warning("Failed to load scores from BigQuery")
            return False

        except Exception as e:
            logger.error(f"Failed to load scores: {e}")
            return False

    @property
    def history(self) -> Optional[pd.DataFrame]:
        """Get the cached history DataFrame."""
        return self._history

    @property
    def is_ready(self) -> bool:
        """Check if the service has data loaded and is ready to make selections."""
        return self._history is not None and not self._history.empty


# Global state
_state = IISAState()


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup and shutdown.

    On startup:
    1. Load settings from environment
    2. Initialize BigQuery provider and DataManager
    3. Load pre-computed scores from BigQuery

    If any step fails, the service continues in fallback mode with random selection.
    """
    global _state

    settings = get_settings()

    # Set log level from settings
    logging.getLogger().setLevel(getattr(logging, settings.log_level))
    logger.setLevel(getattr(logging, settings.log_level))

    logger.info("Starting IISA service...")
    logger.info(f"GCP Project: {settings.gcp_project}")
    logger.info(f"GCP Location: {settings.gcp_location}")

    # Initialize providers and load data
    if not _state.initialize(settings):
        logger.error("IISA initialization failed - cannot start service")
        raise RuntimeError("Failed to initialize IISA providers")

    # Load data on startup - service won't accept requests until ready
    logger.info("Loading indexer scores from BigQuery...")
    if not _state.refresh_data():
        logger.error("Failed to load indexer scores - cannot start service")
        raise RuntimeError("Failed to load indexer scores from BigQuery")

    logger.info("IISA service ready")

    yield

    # Cleanup
    logger.info("Shutting down IISA service...")


app = FastAPI(
    title="IISA Service",
    description="Indexing Indexer Selection Algorithm for The Graph DIPs service",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the service status and whether data has been loaded.
    """
    return HealthResponse(
        status="healthy",
        data_loaded=_state.is_ready,
    )


@app.post("/refresh")
async def refresh_data():
    """
    Trigger a data refresh from BigQuery.

    This endpoint fetches fresh performance data from BigQuery.
    It can take several minutes to complete.
    """
    if not _state._initialized:
        raise HTTPException(
            status_code=503,
            detail="IISA providers not initialized. Check configuration.",
        )

    success = _state.refresh_data()
    if success:
        row_count = len(_state.history) if _state.history is not None else 0
        return {"status": "success", "rows": row_count}
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh data from BigQuery")


@app.post("/select-one", response_model=SingleSelectionResponse)
async def select_one(request: SelectionRequest) -> SingleSelectionResponse:
    """
    Select a single indexer for a deployment using weighted scoring algorithm.
    """
    if not _state.is_ready or _state.history is None:
        raise HTTPException(status_code=503, detail="IISA data not loaded")

    try:
        selected = _select_with_processor(request, num_to_select=1)
        indexer_id = selected[0] if selected else None
        logger.info(f"Selected indexer {indexer_id} for deployment {request.deployment_id}")
        return SingleSelectionResponse(indexer_id=indexer_id)
    except Exception as e:
        logger.exception(f"Selection failed for deployment {request.deployment_id}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {e}")


@app.post("/select-many", response_model=MultiSelectionResponse)
async def select_many(request: SelectionRequest) -> MultiSelectionResponse:
    """
    Select multiple indexers for a deployment using weighted scoring algorithm.
    """
    if not _state.is_ready or _state.history is None:
        raise HTTPException(status_code=503, detail="IISA data not loaded")

    if request.num_candidates is None:
        raise HTTPException(
            status_code=400,
            detail="num_candidates is required for select-many",
        )

    if request.num_candidates <= 0:
        return MultiSelectionResponse(indexer_ids=[])

    try:
        selected = _select_with_processor(request, num_to_select=request.num_candidates)
        logger.info(f"Selected {len(selected)} indexers for deployment {request.deployment_id}: {selected}")
        return MultiSelectionResponse(indexer_ids=selected)
    except Exception as e:
        logger.exception(f"Selection failed for deployment {request.deployment_id}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {e}")


def _select_with_processor(request: SelectionRequest, num_to_select: int) -> list[str]:
    """
    Use DataProcessor for intelligent indexer selection.

    The DataProcessor uses weighted scoring based on:
    - Latency linear regression coefficient
    - Uptime score
    - Existing agreements (fewer is better for load balancing)
    - Stake to fees ratio
    - Success rate
    - Sync duration
    - Agreement acceptance latency
    """
    if _state.history is None:
        return []

    # Build existing_agreements dict from request
    existing_agreements: dict[str, list[str]] = {}
    if request.existing_indexers:
        existing_agreements[request.deployment_id] = request.existing_indexers

    # Build pending_agreements dict - convert to expected format
    pending_agreements: dict[str, list[str]] = request.pending_agreements or {}

    # Create DataProcessor instance
    # Note: DataProcessor does its own filtering of candidates based on
    # indexer_denylist, pending agreements, etc.
    processor = DataProcessor(
        history=_state.history,
        deployment_id=request.deployment_id,
        existing_agreements=existing_agreements,
        pending_agreements=pending_agreements,
        declined_indexers=request.declined_indexers or {},
        indexer_denylist=request.blocklist or [],
    )

    # Get selections
    added, _cancelled = processor.get_indexer_selections()

    # Return the selected indexers for this deployment
    selected = added.get(request.deployment_id, [])

    # DataProcessor returns at most 3 - (existing count) indexers
    # If we need more, we may need multiple calls or a different approach
    return selected[:num_to_select]


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    logger.info(f"Starting IISA service on {settings.host}:{settings.port}")
    uvicorn.run(
        "iisa.iisa_http_endpoints:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
